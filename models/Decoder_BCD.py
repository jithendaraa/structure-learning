import sys, pdb
sys.path.append('..')

import numpy as onp
import haiku as hk

import optax, jax
import jax.numpy as jnp
import jax.random as rnd
from jax import pmap, vmap, jit, ops, random, value_and_grad, lax, grad
from jax.ops import index, index_mul, index_update
from jax.tree_util import tree_map, tree_multimap
from typing import Tuple, Optional, cast, Union
import networkx as nx

from modules.GumbelSinkhorn import GumbelSinkhorn
from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

Tensor = Union[onp.ndarray, jnp.ndarray]
PParamType = Union[hk.Params, jnp.ndarray]
PRNGKey = jnp.ndarray
LStateType = Optional[hk.State]


class Decoder_BCD(hk.Module):
    def __init__(self, dim, l_dim, noise_dim, batch_size, hidden_size, max_deviation, 
                do_ev_noise, proj_dims, log_stds_max=10.0, logit_constraint=10, tau=None, subsample=False, 
                s_prior_std=3.0, num_bethe_iters=20, horseshoe_tau=None):
        super().__init__()

        self.dim = dim
        self.l_dim = l_dim
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.do_ev_noise = do_ev_noise
        self.log_stds_max = log_stds_max
        self.logit_constraint = logit_constraint
        self.tau = tau
        self.subsample = subsample
        self.s_prior_std = s_prior_std
        self.proj_dims = proj_dims
        self.num_bethe_iters = num_bethe_iters
        self.horseshoe_tau = horseshoe_tau

        self.p_model = hk.Sequential([
                            hk.Flatten(), 
                            hk.Linear(hidden_size), jax.nn.gelu,
                            hk.Linear(hidden_size), jax.nn.gelu,
                            hk.Linear(dim * dim)
                        ])
        
        self.decoder = hk.Sequential([
                            hk.Flatten(), 
                            hk.Linear(dim * dim), jax.nn.relu,
                            hk.Linear(dim * dim), jax.nn.relu,
                            hk.Linear(proj_dims)
                        ])
        
        self.ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)
        self.l_prior = Horseshoe(scale=jnp.ones(self.l_dim + self.noise_dim) * self.horseshoe_tau)

    def sample_W(self, L, P):
        return (P @ L @ P.T).T

    def lower(self, theta: Tensor, dim: int) -> Tensor:
        """Given n(n-1)/2 parameters theta, form a
        strictly lower-triangular matrix"""
        out = jnp.zeros((self.dim, self.dim))
        out = ops.index_update(out, jnp.triu_indices(self.dim, 1), theta).T
        return out

    def sample_L(self, L_params, L_state, rng_key):
        """
            Performs L, Σ ~ q_ϕ(L, Σ) 
                    where q_ϕ is a Normal if self.do_ev_noise
                    else q_ϕ is a Normalizing Flow (not implemented, ie when `use_flow` is True; see BCD Nets code instead)
            
            L has dim * (dim - 1) / 2 terms
            Σ is a single term referring to noise on each node 
            
            [TODO]
        """

        L_params = cast(jnp.ndarray, L_params)
        means, log_stds = L_params[: self.l_dim + self.noise_dim], L_params[self.l_dim + self.noise_dim :]
        log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max

        # ? Sample L and Σ jointly from the Normal
        if self.do_ev_noise:
            l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
            full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.batch_size,))
            full_l_batch = cast(jnp.ndarray, full_l_batch)
        else:   raise NotImplementedError
        
        # ? log likelihood for q_ϕ(L, Σ)
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)  
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

        return full_l_batch, full_log_prob_l, None

    def get_P_logits(self, L_samples: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
            Computes the P_logits = T = h_ϕ(L, Σ); h_ϕ = self.p_model
            ! Removed code for opt.factorized from BCD Nets
            ! Refer original code if you want to extend to `factorized` True
        """

        p_logits = self.p_model(L_samples)
        if self.logit_constraint is not None:
            # Want to map -inf to -logit_constraint, inf to +logit_constraint
            p_logits = jnp.tanh(p_logits / self.logit_constraint) * self.logit_constraint
        return p_logits.reshape((-1, self.dim, self.dim))
    
    def eltwise_ancestral_sample(self, weighted_adj_mat, perm_mat, eps, rng_key, interv_target=None, interv_value=0.0):
        """
            eps: standard deviation
            Given a single weighted adjacency matrix perform ancestral sampling
            Use the permutation matrix to get the topsorted order.
            Traverse topologically and give one ancestral sample using weighted_adj_mat and eps
        """
        sample = jnp.zeros((self.dim+1))
        ordering = jnp.arange(0, self.dim)

        theta = weighted_adj_mat
        adj_mat = jnp.where(weighted_adj_mat != 0, 1.0, 0.0)
        swapped_ordering = ordering[jnp.where(perm_mat, size=self.dim)[1].argsort()]
        noise_terms = jnp.multiply(eps, random.normal(rng_key, shape=(self.dim,)))

        # Traverse node topologically
        for j in swapped_ordering:
            mean = sample[:-1] @ theta[:, j]
            # ! Use lax.cond if possible
            sample = index_update(sample, index[j], mean + noise_terms[j])
            sample = index_update(sample, index[interv_target], interv_value)

        return sample[:-1]

    def ancestral_sample(self, weighted_adj_mat, perm_mat, eps, rng_key, interv_targets, interv_value=0.0):
        """
            Gives `n_samples` = len(interv_targets) of data generated from one weighted_adj_mat and interventions from `interv_targets`. 
            Each of these samples will be an ancestral sample taking into account the ith intervention in interv_targets
        """

        rng_keys = rnd.split(rng_key, len(interv_targets))
        samples = vmap(self.eltwise_ancestral_sample, (None, None, None, 0, 0, None), (0))(weighted_adj_mat, perm_mat, eps, rng_keys, interv_targets, interv_value)
        return samples

    def get_gradients(self, P_params: PParamType, L_params: hk.Params, L_states: LStateType,
        gt_data: jnp.ndarray, rng_key: PRNGKey, 
        hard: bool, interv_targets: jnp.ndarray) -> Tuple[jnp.ndarray, LStateType]:
        """
            [TODO]
        """
        tau_scaling_factor = 1.0 / self.tau

        # * Get loss and gradients of loss wrt to P and L
        (loss, L_states), grads = value_and_grad(self.elbo, argnums=(0, 1), has_aux=True)(
                P_params, L_params, L_states, gt_data, rng_key, hard, interv_targets)

        elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)

        # * L2 regularization over parameters of P
        l2_elbo_grad_P = grad(
            lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p))
        )(P_params)
        elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)
        return elbo_grad_P, elbo_grad_L

    def __call__(self, hard, rng_key, interv_targets, init=False, 
                P_params=None, L_params=None, L_states=None, gt_data=None):
        """
            [TODO]
        """
        if init:    
            x = jnp.zeros((self.batch_size, self.l_dim + self.noise_dim))
            return self.p_model(x), self.decoder(jnp.ones((self.dim)))

        batched_qz_samples, X_recons = jnp.array([]), jnp.array([])

        # ? 1. Draw (L, Σ) ~ q_ϕ(L, Σ)
        full_l_batch, full_log_prob_l, out_L_states = self.sample_L(L_params, L_states, rng_key)
        l_batch = full_l_batch[:, :-self.noise_dim]
        w_noise = full_l_batch[:, -self.noise_dim:]
        batched_log_noises = jnp.ones((self.batch_size, self.dim)) * w_noise.reshape((self.batch_size, self.noise_dim))
        batched_L = vmap(self.lower, in_axes=(0, None))(l_batch, self.dim)

        # ? 2. Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
        batched_P_logits = self.get_P_logits(full_l_batch)

        # ? 3. Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) or hard P = Hungarian(P̃) 
        if hard:    batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
        else:   batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)

        batched_W = vmap(self.sample_W, (0, 0), (0))(batched_L, batched_P)
        batched_adj_mats = jnp.where(batched_W != 0, 1.0, 0.0)
        batched_qz_samples = vmap(self.ancestral_sample, (0, 0, 0, None, None, None), (0))(batched_W, batched_P, jnp.exp(batched_log_noises), rng_key, interv_targets, 0.0)
        X_recons = vmap(vmap(self.decoder, (0), (0)), (0), (0))(batched_qz_samples)

        return (batched_P, batched_P_logits, batched_L, batched_log_noises, 
                batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, X_recons)


    def log_prob_x(self, data, log_sigmas, P, L, rng_key):
        """
            Calculates log P(X|Z) for latent Zs

            X|Z is Gaussian so easy to calculate

            Args:
                data: an (n_samples x dim)-dimensional array of observations
                log_sigmas: A (dim)-dimension vector of log standard deviations
                P: A (dim x dim)-dimensional permutation matrix
                L: A (dim x dim)-dimensional strictly lower triangular matrix
            Returns:
                log_prob: Log probability of observing Xs given P, L
        """

        adjustment_factor = 1
        # ! To implement, look at this function in the original code of BCD Nets 
        if self.subsample: raise NotImplementedError

        n, dim = data.shape
        W = self.sample_W(L, P)
        precision = ((jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T)
        eye_minus_W_logdet = 0
        log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

        def datapoint_exponent(x):
            return -0.5 * x.T @ precision @ x

        log_exponent = vmap(datapoint_exponent)(data)

        return adjustment_factor * (
            0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
            + jnp.sum(log_exponent)
        )
    

    def elbo(self, P_params: PParamType, L_params: hk.Params, L_states: LStateType, gt_data: jnp.ndarray, 
        rng_key: PRNGKey, hard: bool, interv_targets: jnp.ndarray) -> Tuple[jnp.ndarray, LStateType]:
        """
            [TODO]
        """
        rng_key, rng_key_2 = rnd.split(rng_key, 2)

        # ? 1. Draw (L, Σ) ~ q_ϕ(L, Σ)
        full_l_batch, full_log_prob_l, out_L_states = self.sample_L(L_params, L_states, rng_key)
        l_batch = full_l_batch[:, :-self.noise_dim]
        w_noise = full_l_batch[:, -self.noise_dim:]
        batched_log_noises = jnp.ones((self.batch_size, self.dim)) * w_noise.reshape((self.batch_size, self.noise_dim))
        batched_L = vmap(self.lower, in_axes=(0, None))(l_batch, self.dim)

        # ? 2. Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
        batched_P_logits = self.get_P_logits(full_l_batch)

        # ? 3. Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) or hard P = Hungarian(P̃) 
        if hard:    
            batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key,)
            batched_W = vmap(self.sample_W, (0, 0), (0))(batched_L, batched_P)
            batched_adj_mats = jnp.where(batched_W != 0, 1.0, 0.0)

            batched_qz_samples = vmap(self.ancestral_sample, (0, 0, 0, None, None, None), (0))(batched_W, batched_P, jnp.exp(batched_log_noises), rng_key, interv_targets, 0.0)
            X_recons = vmap(vmap(self.decoder, (0), (0)), (0), (0))(batched_qz_samples)

        else:   
            batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key,)
            

        def bcd_loop(rng_key):
            """
                [TODO]
            """
            
            # ? 4. Get likelihood
            likelihoods = vmap(self.log_prob_x, in_axes=(None, 0, 0, 0, None))(gt_data, batched_log_noises, batched_P, batched_L, rng_key)
            
            # ? 5. Get KL terms
            l_prior_probs = jnp.sum(self.l_prior.log_prob(full_l_batch)[:, :self.l_dim], axis=1)
            s_prior_probs = jnp.sum(full_l_batch[:, self.l_dim:] ** 2 / (2 * self.s_prior_std ** 2), axis=-1)
            logprob_P = vmap(self.ds.logprob, in_axes=(0, 0, None))(batched_P, batched_P_logits, self.num_bethe_iters)
            log_P_prior = -jnp.sum(jnp.log(onp.arange(self.dim) + 1))
            KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
            
            # ? Get ELBO
            final_term = likelihoods - KL_term_L - logprob_P + log_P_prior
            return jnp.mean(final_term), out_L_states

        rng_keys = rnd.split(rng_key, 1)
        _, (elbos, out_L_states) = lax.scan(lambda _, rng_key: (None, bcd_loop(rng_key)), None, rng_keys)
        elbo_estimate = jnp.mean(elbos)
        return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)

    






