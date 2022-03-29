import sys, pdb
sys.path.append('..')

import numpy as onp
import haiku as hk

import optax, jax
import jax.numpy as jnp
import jax.random as rnd
from jax import pmap, vmap, jit, ops, random
from jax.ops import index, index_mul, index_update
from typing import Tuple, Optional, cast, Union
import networkx as nx

from modules.GumbelSinkhorn import GumbelSinkhorn
from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

Tensor = Union[onp.ndarray, jnp.ndarray]
PParamType = Union[hk.Params, jnp.ndarray]
PRNGKey = jnp.ndarray


class Decoder_BCD(hk.Module):
    def __init__(self, dim, l_dim, noise_dim, batch_size, hidden_size, max_deviation, 
                do_ev_noise, proj_dims, log_stds_max=10.0, logit_constraint=10, tau=None, subsample=False, s_prior_std=3.0):
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

        self.ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)

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
     
    def get_W(self, P, L):
        return P @ L @ P.T

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

        return full_l_batch, full_log_prob_l


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
        for j in reversed(swapped_ordering):
            mean = sample[:-1] @ theta[:, j]
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


    def __call__(self, x, hard, rng_key, interv_targets, init=False, L_params=None, L_state=None):
        """
            [TODO]
        """
        batched_qz_samples, X_recons = jnp.array([]), jnp.array([])
        
        if init: 
            return self.p_model(x), self.decoder(jnp.zeros((self.dim)))

        rng_key, rng_key_1 = rnd.split(rng_key, 2)

        # ? 1. Draw (L, Σ) ~ q_ϕ(L, Σ)
        full_l_batch, full_log_prob_l = self.sample_L(L_params, L_state, rng_key)
        l_batch = full_l_batch[:, :-self.noise_dim]
        w_noise = full_l_batch[:, -self.noise_dim:]
        batched_noises = jnp.ones((self.batch_size, self.dim)) * w_noise.reshape((self.batch_size, self.noise_dim))
        batched_L = vmap(self.lower, in_axes=(0, None))(l_batch, self.dim)
        
        # ? 2. Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
        batched_P_logits = self.get_P_logits(full_l_batch)

        # ? 3. Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ), hard P = Hungarian(P̃) 
        if hard:    
            batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key,)
            batched_W = vmap(self.get_W, (0, 0), (0))(batched_P, batched_L)
            batched_adj_mats = jnp.where(batched_W != 0, 1.0, 0.0)

            batched_qz_samples = vmap(self.ancestral_sample, (0, 0, 0, None, None, None), (0))(batched_W, batched_P, batched_noises, rng_key, interv_targets, 0.0)
            X_recons = vmap(vmap(self.decoder, (0), (0)), (0), (0))(batched_qz_samples)

        else:   
            batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key,)

        return batched_P, batched_L, batched_noises, batched_W, batched_qz_samples





