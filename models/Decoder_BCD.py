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
                s_prior_std=3.0, num_bethe_iters=20, horseshoe_tau=None, learn_noise=False, noise_sigma=0.1,
                P=None, L=None, decoder_layers='linear', learn_L=True, pred_last_L = 1, fix_decoder=False):
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
        self.learn_noise = learn_noise
        self.learn_L = learn_L
        self.P = P
        self.L = L
        self.num_elems = pred_last_L
        self.fix_decoder = fix_decoder

        if self.do_ev_noise:
            self.noise_sigma = jnp.array([[noise_sigma]] * self.batch_size)
            self.log_noise_sigma = jnp.log(self.noise_sigma)
        else:   self.noise_sigma = noise_sigma

        # self.p_model = hk.Sequential([
        #                     hk.Flatten(), 
        #                     hk.Linear(hidden_size), jax.nn.gelu,
        #                     hk.Linear(hidden_size), jax.nn.gelu,
        #                     hk.Linear(dim * dim)
        #                 ])

        if decoder_layers == 'nonlinear': 
            self.decoder = hk.Sequential([
                hk.Flatten(), 
                hk.Linear(16, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(proj_dims, with_bias=False)
            ])
            
        elif decoder_layers == 'linear':
            self.decoder = hk.Sequential([
                hk.Flatten(), hk.Linear(proj_dims, with_bias=False)
            ])
        
        self.ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)
        self.l_prior = Horseshoe(scale=jnp.ones(self.l_dim) * self.horseshoe_tau)

    def project(self, data, proj_matrix):
        return jnp.matmul(data, proj_matrix)

    def sample_W(self, L, P):
        W = (P @ L @ P.T).T
        return W

    def lower(self, theta: Tensor, dim: int) -> Tensor:
        """
            Given n(n-1)/2 parameters theta, form a
            strictly lower-triangular matrix
        """
        out = jnp.zeros((self.dim, self.dim))
        out = ops.index_update(out, jnp.tril_indices(self.dim, -1), theta)
        return out

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
    
    def eltwise_ancestral_sample(self, weighted_adj_mat, perm_mat, eps, rng_key, interv_target=None, interv_values=None):
        """
            eps: standard deviation
            Given a single weighted adjacency matrix perform ancestral sampling
            Use the permutation matrix to get the topsorted order.
            Traverse topologically and give one ancestral sample using weighted_adj_mat and eps
        """

        sample = jnp.zeros((self.dim+1))
        ordering = jnp.arange(0, self.dim)

        theta = weighted_adj_mat
        swapped_ordering = ordering[jnp.where(perm_mat, size=self.dim)[1].argsort()]
        noise_terms = jnp.multiply(eps, random.normal(rng_key, shape=(self.dim,)))

        # Traverse node topologically
        for j in swapped_ordering:
            mean = sample[:-1] @ theta[:, j]
            # ! Use lax.cond if possible 
            sample = index_update(sample, index[j], mean + noise_terms[j])
            sample = index_update(sample, index[interv_target], interv_values[interv_target])

        return sample[:-1]

    def ancestral_sample(self, weighted_adj_mat, perm_mat, eps, rng_key, interv_targets, interv_values):
        """
            Gives `n_samples` = len(interv_targets) of data generated from one weighted_adj_mat and interventions from `interv_targets`. 
            Each of these samples will be an ancestral sample taking into account the ith intervention in interv_targets
        """

        interv_values = jnp.concatenate( ( interv_values, jnp.zeros((len(interv_targets), 1)) ), axis=1)
        rng_keys = rnd.split(rng_key, len(interv_targets))
        samples = vmap(self.eltwise_ancestral_sample, (None, None, None, 0, 0, 0), (0))(weighted_adj_mat, perm_mat, eps, rng_keys, interv_targets, interv_values)
        return samples

    def get_GT_L(self):
        l_batch = self.L[jnp.tril_indices(self.dim, k=-1)][jnp.newaxis, :]
        l_batch = l_batch.repeat(self.batch_size, axis=0)
        full_l_batch = jnp.array([0.0] * self.batch_size)
        return l_batch, full_l_batch

    def sample_L(self, L_params, rng_key):
        """
            Performs L, Σ ~ q_ϕ(L, Σ) 
                    where q_ϕ is a Normal if self.do_ev_noise
                    else q_ϕ is a Normalizing Flow (not implemented, ie when `use_flow` is True; see BCD Nets code instead)
            
            L has dim * (dim - 1) / 2 terms
            Σ is a single term referring to noise on each node 
            
            [TODO]: fix this
        """
        L_params = cast(jnp.ndarray, L_params)

        if self.learn_noise:    means, log_stds = L_params[: self.l_dim + self.noise_dim], L_params[self.l_dim + self.noise_dim :]
        else:                   means, log_stds = L_params[: self.l_dim], L_params[self.l_dim :]
        if self.log_stds_max is not None:    log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max

        # ? Sample L from the Normal
        if self.do_ev_noise:
            l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
            full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.batch_size,))
            full_l_batch = cast(jnp.ndarray, full_l_batch)
        else:   raise NotImplementedError
        
        # ? log likelihood for q_ϕ(L)
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)  
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)
        return full_l_batch, full_log_prob_l, None

    def sample_partial_L(self, L_params, rng_key):
        """
            [TODO]
            learns only the last parameter in L
        """
        gt_means, _ = self.get_GT_L()
        L_params = cast(jnp.ndarray, L_params)
        means, log_stds = L_params[: self.l_dim], L_params[self.l_dim :]
        log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max

        # ? Sample L from the Normal
        if self.do_ev_noise:
            l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
            full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.batch_size,))
            full_l_batch = cast(jnp.ndarray, full_l_batch)
        else:   raise NotImplementedError
        
        # ? log likelihood for q_ϕ(L)
        full_l_batch = full_l_batch.at[:, :-self.num_elems].set(gt_means[:, :-self.num_elems])
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)  
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

        return full_l_batch, full_log_prob_l
    
    def __call__(self, hard, rng_key, interv_targets, interv_values, init=False, 
                P_params=None, L_params=None, decoder_params=None):
        """
            [TODO]
            hard: bool
            rng_key: jax PRNG Key
            interv_values: jnp.array of shape (opt.num_samples, opt.num_nodes)
        """

        if init:
            x = jnp.ones((self.batch_size, self.l_dim + self.noise_dim))
            return None, self.decoder(jnp.ones((10, self.dim)))

        batched_qz_samples, X_recons = jnp.array([]), jnp.array([])

        # ? 1. Draw L ~ q_ϕ(L); Σ are not learned, they are fixed
        if self.learn_L is True:    
            l_batch, full_log_prob_l, _ = self.sample_L(L_params, rng_key)
        
        # ! TEMPORARY TEST: remove later. Fixes L, learns only the last element.
        elif self.learn_L == 'partial': 
            l_batch, full_log_prob_l = self.sample_partial_L(L_params, rng_key)

        # ! TEMPORARY TEST: remove later. Fixes L, no longer learn it.
        elif self.learn_L is False: 
            l_batch, full_log_prob_l = self.get_GT_L() 
        
        if self.learn_noise:
            full_l_batch = l_batch
            batched_log_noises = full_l_batch[:, -self.noise_dim:]
        else:
            w_noise = self.log_noise_sigma
            full_l_batch = jnp.concatenate((l_batch, w_noise), axis=1)
            batched_log_noises = jnp.ones((self.batch_size, self.dim)) * w_noise.reshape((self.batch_size, self.noise_dim))
        
        batched_L = vmap(self.lower, in_axes=(0, None))(l_batch[:,  :self.l_dim], self.dim)

        # ! TEMPORARY TEST: fixes permutation to Identity; for exps where we are not learning P. Remove later or add support for learning P as well.
        batched_P = jnp.eye(self.dim, self.dim)[jnp.newaxis, :].repeat(self.batch_size, axis=0) 

        # ? 2. Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
        # batched_P_logits = self.get_P_logits(full_l_batch)
        batched_P_logits = None

        # ? 3. Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) or hard P = Hungarian(P̃) 
        # if hard:    batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
        # else:   batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)

        batched_W = vmap(self.sample_W, (0, 0), (0))(batched_L, batched_P)
        
        rng_keys = rnd.split(rng_key, self.batch_size)
        # batched_qz_samples = self.ancestral_sample(batched_W[0], batched_P[0], jnp.exp(batched_log_noises)[0], 
        #                     rng_keys[0], interv_targets, interv_values)

        batched_ancestral_sample = vmap(self.ancestral_sample, (0, 0, 0, 0, None, None), (0))

        batched_qz_samples = batched_ancestral_sample(batched_W, batched_P, jnp.exp(batched_log_noises), 
                                                        rng_keys, interv_targets, interv_values)

        if self.fix_decoder is True:
            print("Fixed decoder")
            X_recons = vmap(self.project, (0, None), (0))(batched_qz_samples, self.P)
        else:
            X_recons = vmap(self.project, (0, None), (0))(batched_qz_samples, decoder_params)

        return (batched_P, batched_P_logits, batched_L, batched_log_noises, 
                batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, X_recons)
    

    