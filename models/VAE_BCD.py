import sys, pdb
sys.path.append('..')

import haiku as hk
import jax
import jax.numpy as jnp
from jax import vmap, ops, random
from jax.ops import index, index_update

from typing import cast
from modules.GumbelSinkhorn import GumbelSinkhorn
from tensorflow_probability.substrates.jax.distributions import Normal


class VAE_BCD(hk.Module):
    def __init__(self, d, proj_dims, learn_noise, learn_P, hidden_size, log_noise_sigma, batch_size, 
                max_deviation, projection='linear', ev_noise=True, log_stds_max=3.0, 
                learn_L=True, P=None):
        """
            d: int, number of nodes in the latent SCM
            proj_dims: int, dimension D of x
            projection: str, linear or nonlinear projection of z to x
            ev_noise: bool, if True, we are in the equal noise(ϵ) variance of a linear Gaussian SCM
            P: The true projection matrix
        """
        super().__init__()
        assert ev_noise == True
        assert learn_L == True

        self.num_nodes = d
        self.proj_dims = proj_dims
        self.log_stds_max = log_stds_max
        self.learn_noise = learn_noise
        self.learn_P = learn_P

        self.log_noise_sigma = log_noise_sigma
        self.batch_size = batch_size
        self.do_ev_noise = ev_noise

        self.l_dim = d * (d-1) // 2
        if ev_noise: self.noise_dim = 1
        else: self.noise_dim = d
        
        self.doubly_stochastic = GumbelSinkhorn(d, noise_type="gumbel", tol=max_deviation)
        self.num_L_params = 2 * self.l_dim
        self.num_Σ_params = 2 * self.noise_dim * int(self.learn_noise)
        self.P = P
        
        self.encoder = hk.Sequential([
            hk.Flatten(), 
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params), jax.nn.gelu,
            hk.Linear(self.num_L_params+self.num_Σ_params),
        ])

        if self.learn_P:
            self.p_model = hk.Sequential([
                                hk.Flatten(), 
                                hk.Linear(hidden_size), jax.nn.gelu,
                                hk.Linear(hidden_size), jax.nn.gelu,
                                hk.Linear(d * d)
                            ])

        if projection == 'linear':
            self.decoder = hk.Sequential([
                hk.Flatten(), 
                hk.Linear(proj_dims, with_bias=False)
            ])

        elif projection == '2_layer_mlp':
            self.decoder = hk.Sequential([
                hk.Flatten(), 
                hk.Linear(16, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(64, with_bias=False), jax.nn.gelu,
                hk.Linear(proj_dims, with_bias=False)
            ])
        
    def lower(self, theta):
        """
            Given n(n-1)/2 parameters theta, form a
            strictly lower-triangular matrix
        """
        out = jnp.zeros((self.num_nodes, self.num_nodes))
        out = ops.index_update(out, jnp.tril_indices(self.num_nodes, -1), theta)
        return out
    
    def sample_L_and_Σ(self, rng_key, L_Σ_params):
        """
        """
        if self.learn_noise: 
            means, log_stds = L_Σ_params[:self.l_dim+self.noise_dim], L_Σ_params[self.l_dim+self.noise_dim:]
        else:
            means, log_stds = L_Σ_params[:self.l_dim], L_Σ_params[self.l_dim :]
        
        if self.log_stds_max is not None:   
            log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max
        
        if self.do_ev_noise:
            l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
            full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.batch_size,))
            full_l_batch = cast(jnp.ndarray, full_l_batch)
        else: raise NotImplementedError
        
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)  
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)
        return full_l_batch, full_log_prob_l

    def get_P_logits(self, L_samples):
        """
            Computes the P_logits = T = h_ϕ(L, Σ); h_ϕ = self.p_model
        """
        p_logits = self.p_model(L_samples)
        if self.logit_constraint is not None:
            # Want to map -inf to -logit_constraint, inf to +logit_constraint
            p_logits = jnp.tanh(p_logits / self.logit_constraint) * self.logit_constraint
        return p_logits.reshape((-1, self.dim, self.dim))

    def sample_P(self, rng_key, full_l_batch, hard):
        """
        """
        if self.learn_P is False:
            batched_P = self.P[jnp.newaxis, :].repeat(len(full_l_batch), axis=0) 
            batched_P_logits = None
        else:
            # ? Compute logits T = h_ϕ(L, Σ); h_ϕ = p_model
            batched_P_logits = self.get_P_logits(full_l_batch)

            # ? Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) or hard P = Hungarian(P̃) 
            if hard:    
                batched_P = self.doubly_stochastic.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
            
            else:   
                batched_P = self.doubly_stochastic.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)
            
        return batched_P, batched_P_logits

    def sample_W(self, L, P):
        W = (P @ L @ P.T).T
        return W

    def eltwise_ancestral_sample(self, W, P, eps, rng_key, interv_target=None, interv_values=None):
        """
            Given a single weighted adjacency matrix perform ancestral sampling
            Use the permutation matrix to get the topsorted order.
            Traverse topologically and give one ancestral sample using W and eps
            
            W: weighted adjacency matrix jnp.array(d, d)
            P: permutation matrix jnp.array(d, d)
            eps:
            rng_key:
            interv_target:
            interv_values:
        """

        sample = jnp.zeros((self.num_nodes+1))
        ordering = jnp.arange(0, self.num_nodes)
        swapped_ordering = ordering[jnp.where(P, size=self.num_nodes)[1].argsort()]
        noise_terms = jnp.multiply(eps, random.normal(rng_key, shape=(self.num_nodes,)))

        # Traverse node topologically
        for j in swapped_ordering:
            mean = sample[:-1] @ W[:, j]
            # ! Use lax.cond if possible 
            sample = index_update(sample, index[j], mean + noise_terms[j])
            sample = index_update(sample, index[interv_target], interv_values[interv_target])

        return sample[:-1]

    def ancestral_sample(self, W, P, eps, rng_key, interv_targets, interv_values):
        """
            Gives `n_samples` = len(interv_targets) of data generated from one weighted_adj_mat and interventions from `interv_targets`. 
            Each of these samples will be an ancestral sample taking into account the ith intervention in interv_targets
        """
        num_samples = len(interv_targets)
        interv_values = jnp.concatenate( ( interv_values, jnp.zeros((num_samples, 1)) ), axis=1)
        rng_keys = random.split(rng_key, num_samples)
        vmapped_ancestral_sampler = vmap(self.eltwise_ancestral_sample, (None, None, None, 0, 0, 0), (0))
        samples = vmapped_ancestral_sampler(W, P, eps, rng_keys, interv_targets, interv_values)
        return samples

    def __call__(self, hard, rng_key, x_targets, interv_targets=None, interv_values=None, init=False):
        """
        """
        if init is True:
            z = jnp.ones((len(x_targets), self.num_nodes))
            if self.learn_P: self.p_model(jnp.ones((self.l_dim + self.noise_dim)))
            return self.encoder(x_targets), self.decoder(z)
        
        # Get q(L, Σ | X)
        L_Σ_params = self.encoder(x_targets)
        L_Σ_params = jnp.mean(L_Σ_params, axis=0)
        full_l_batch, full_log_prob_l = self.sample_L_and_Σ(rng_key, L_Σ_params)

        if self.learn_noise is False:
            w_noise = jnp.ones((full_l_batch.shape[0], 1)) * self.log_noise_sigma
            full_l_batch = jnp.concatenate((full_l_batch, w_noise), axis=1)

        # Get q(P | L, Σ, X)
        batched_log_noises = full_l_batch[:, -self.noise_dim:]
        batched_L = vmap(self.lower, in_axes=(0))(full_l_batch[:,  :self.l_dim])
        batched_P, batched_P_logits = self.sample_P(rng_key, full_l_batch, hard)
        
        # W = (PLP.T).T
        batched_W = vmap(self.sample_W, (0, 0), (0))(batched_L, batched_P)
        
        # \hat{z} <- Ancestral sample(P, L, Σ, interventional targets)
        rng_keys = random.split(rng_key, len(batched_L))
        
        batched_ancestral_sample = vmap(self.ancestral_sample, (0, 0, 0, 0, None, None), (0))
        batched_qz_samples = batched_ancestral_sample(
                                batched_W, 
                                batched_P, 
                                jnp.exp(batched_log_noises), 
                                rng_keys, 
                                interv_targets, 
                                interv_values
                            )

        # \hat{X} = Decoder(\hat{z})
        decode_along_Ws = vmap(self.decoder, 0, 0)
        decode_along_datasets = vmap(decode_along_Ws, 0, 0)
        X_recons = decode_along_datasets(batched_qz_samples)

        return (batched_P, batched_P_logits, batched_L, batched_log_noises, 
                batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, 
                X_recons)