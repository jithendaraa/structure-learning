import sys, pdb
sys.path.append('..')

import numpy as onp
import haiku as hk

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import vmap, ops, random
from jax.ops import index, index_update
from typing import Tuple, Optional, cast, Union
import networkx as nx

from modules.GumbelSinkhorn import GumbelSinkhorn
from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

Tensor = Union[onp.ndarray, jnp.ndarray]
PParamType = Union[hk.Params, jnp.ndarray]
PRNGKey = jnp.ndarray
LStateType = Optional[hk.State]

class BN(hk.Module):
    def __call__(self, x):
        return hk.BatchNorm(True, True, 0.9)(x, True)

class Discriminator(hk.Module):
    def __init__(self, image_dims):
        super().__init__()

        self.image_dims = image_dims

        self.disc = hk.Sequential([
            hk.Conv2D(16, 3), BN(), jax.nn.leaky_relu,
            hk.Conv2D(32, 3), BN(), jax.nn.leaky_relu,
            hk.Flatten(),
            hk.Linear(256), jax.nn.leaky_relu,
            hk.Linear(128), jax.nn.leaky_relu,
            hk.Linear(64), jax.nn.leaky_relu,
            hk.Linear(1), jax.nn.sigmoid   
        ])

    def __call__(self, x):
        return self.disc(x)


class NodewiseMLP(hk.Module):
    def __init__(self, hidden_dims):
        super().__init__()

        self.mlp = hk.Sequential([
            hk.Linear(hidden_dims), jax.nn.gelu,
            hk.Linear(hidden_dims), jax.nn.gelu,
            hk.Linear(hidden_dims), jax.nn.gelu,
            hk.Linear(hidden_dims),
        ])
    
    def __call__(self, x):
        return self.mlp(x)


class Conv_Decoder_BCD(hk.Module):
    def __init__(self, dim, l_dim, noise_dim, batch_size, do_ev_noise, learn_P, learn_noise, 
                proj_dims, tau, P, max_deviation, logit_constraint=10, log_stds_max=10., 
                learn_L=True, L=None, noise_sigma=None):
        super().__init__()

        self.dim = dim
        self.l_dim = l_dim
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.do_ev_noise = do_ev_noise
        self.log_stds_max = log_stds_max
        self.learn_L = learn_L
        self.learn_P = learn_P
        self.learn_noise = learn_noise
        self.proj_dims = proj_dims
        self.logit_constraint = logit_constraint
        self.tau = tau
        self.P = P
        self.L = L
        self.nodewise_hidden_dim = 16
        self.ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)
        
        if self.do_ev_noise:
            self.noise_sigma = jnp.array([[noise_sigma]] * self.batch_size)
            self.log_noise_sigma = jnp.log(self.noise_sigma)
        else:   self.noise_sigma = noise_sigma

        if self.learn_P:
            self.p_model = hk.Sequential([
                                hk.Flatten(), 
                                hk.Linear(64), jax.nn.gelu,
                                hk.Linear(64), jax.nn.gelu,
                                hk.Linear(64), jax.nn.gelu,
                                hk.Linear(dim * dim)
                            ])
            

        self.linear_decoder = hk.Sequential([
            hk.Linear(16), jax.nn.gelu,
            hk.Linear(64), jax.nn.gelu,
            hk.Linear(256), jax.nn.gelu,
            hk.Linear(512), jax.nn.gelu,
            hk.Linear(1024), jax.nn.gelu,
            hk.Linear(2048), jax.nn.gelu,
            hk.Linear(2500), jax.nn.sigmoid
        ])


    def sample_W(self, L, P):
        W = (P @ L @ P.T).T
        return W

    # def spatial_broadcast(self, z_samples, h_, w_):
    #     """
    #         `z_samples` has shape: [self.batch_size, opt.batches, self.dim]
    #     """
    #     res = jnp.zeros((self.batch_size, z_samples.shape[1], 1, self.nodewise_hidden_dim))

    #     # for i in range(self.dim):
    #     #     out = getattr(self, f"mlp_{i}")(z_samples[:, :, i:i+1])[:, :, None, :]
    #     #     res = jnp.concatenate( (res, out), axis=-2 )
    #     res = self.node_mlp(z_samples)
        
    #     flat_z = res.reshape(-1, res.shape[-1])[:, None, None, :]
    #     broadcasted_image = jnp.tile(flat_z, (1, h_, w_, 1))
    #     return broadcasted_image

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
        """
        p_logits = self.p_model(L_samples)
        if self.logit_constraint is not None:        # Want to map -inf to -logit_constraint, inf to +logit_constraint
            p_logits = jnp.tanh(p_logits / self.logit_constraint) * self.logit_constraint
        return p_logits.reshape((-1, self.dim, self.dim))

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
        return full_l_batch, full_log_prob_l

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

    def __call__(self, rng_key, interv_targets, interv_values, L_params, hard=True):
        h_, w_ = self.proj_dims[-2] // 2, self.proj_dims[-1] // 2  

        num_input_samples = len(interv_targets)
        l_batch, full_log_prob_l = self.sample_L(L_params, rng_key)

        if self.learn_noise:        # * Perform inference over Σ as well
            full_l_batch = l_batch
            batched_log_noises = full_l_batch[:, -self.noise_dim:]
        else:                       # * Don't infer Σ
            w_noise = self.log_noise_sigma
            full_l_batch = jnp.concatenate((l_batch, w_noise), axis=1)
            batched_log_noises = jnp.ones((self.batch_size, self.dim)) * w_noise.reshape((self.batch_size, self.noise_dim))

        if self.learn_L is True:
            batched_L = vmap(self.lower, in_axes=(0, None))(l_batch[:,  :self.l_dim], self.dim)
        else:
            batched_L = self.L[jnp.newaxis, :].repeat(self.batch_size, axis=0) 

        if self.learn_P:
            # ? Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
            batched_P_logits = self.get_P_logits(full_l_batch)

            # ? Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) or hard P = Hungarian(P̃) 
            if hard:    batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
            else:   batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)

        else:   
            batched_P = self.P[jnp.newaxis, :].repeat(self.batch_size, axis=0) 
            batched_P_logits = None

        batched_W = vmap(self.sample_W, (0, 0), (0))(batched_L, batched_P)
        rng_keys = rnd.split(rng_key, self.batch_size)
        batched_ancestral_sample = vmap(self.ancestral_sample, (0, 0, 0, 0, None, None), (0))
        batched_qz_samples = batched_ancestral_sample(batched_W, batched_P, jnp.exp(batched_log_noises), 
                                                        rng_keys, interv_targets, interv_values)

        # spatial_qz = self.spatial_broadcast(batched_qz_samples, h_, w_) # (self.batch_size * self.batches, h_, w_, self.nodewise_hidden_dim)
        
        # spatial_qz = spatial_qz.reshape(self.batch_size, 
        #                                 num_input_samples, 
        #                                 h_, w_, 
        #                                 self.nodewise_hidden_dim) 

        # spatial_qz = spatial_qz.reshape(-1, h_, w_, self.nodewise_hidden_dim)
        
        # X_recons = self.decoder(spatial_qz).reshape(self.batch_size, 
        #                                             num_input_samples, 
        #                                             self.proj_dims[-2], 
        #                                             self.proj_dims[-1], 
        #                                             self.proj_dims[-3]) 
        
        X_recons = self.linear_decoder(batched_qz_samples)
        
        X_recons = X_recons.reshape(self.batch_size, 
                                    num_input_samples, 
                                    self.proj_dims[-2], 
                                    self.proj_dims[-1], 
                                    self.proj_dims[-3])

        return (batched_P, batched_P_logits, batched_L, batched_log_noises, 
                batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, X_recons * 255.)