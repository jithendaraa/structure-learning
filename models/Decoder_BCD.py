import sys, pdb
sys.path.append('..')

import haiku as hk
import jax.numpy as jnp
import optax, jax
from jax import pmap, vmap, jit
from typing import Tuple, Optional, cast, Union


from modules.GumbelSinkhorn import GumbelSinkhorn
from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe


class Decoder_BCD(hk.Module):
    def __init__(self, dim, l_dim, noise_dim, batch_size, hidden_size, lr, max_deviation, do_ev_noise, log_stds_max=10.0):
        super().__init__()

        self.l_dim = l_dim
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.log_stds_max = log_stds_max
        self.do_ev_noise = do_ev_noise

        # self.ds = GumbelSinkhorn(num_nodes, noise_type="gumbel", tol=max_deviation)

        self.p_model = hk.Sequential([
                            hk.Flatten(), 
                            hk.Linear(hidden_size), jax.nn.gelu,
                            hk.Linear(hidden_size), jax.nn.gelu,
                            hk.Linear(dim * dim)
                        ])
     

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


    def __call__(self, x, init, L_params=None, L_state=None, rng_key=None):
        """
            [TODO]
        """

        if init is False:
            full_l_batch, full_log_prob_l = self.sample_L(L_params, L_state, rng_key)
            pdb.set_trace()

        return self.p_model(x)

        # full_l_batch, full_log_prob_l, out_L_states 
        # = sample_L(L_params, L_states, rng_key)








