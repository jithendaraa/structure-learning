from tkinter import W
import haiku as hk
import jax, pdb
import jax.numpy as jnp
import jax.random as rnd
from jax import vmap, ops, random

class VAE(hk.Module):
    def __init__(self, d, D, corr, sigmoid=True):
        super().__init__()

        self.d = d
        self.D = D
        self.corr = corr
        self.sigmoid = sigmoid
        if corr: 
            self.elements_in_chol = int(((d**2 - d)/2) + d)
        else:
            self.elements_in_chol = d

        self.encoder = hk.Sequential([
            hk.Linear(2*(d+D)), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(64), jax.nn.relu,
            hk.Linear(d + self.elements_in_chol)
        ])

        if sigmoid is False:
            self.decoder = hk.Sequential([
            hk.Linear(64), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(D)
        ])

        else:
            self.decoder = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(D), jax.nn.sigmoid
            ])
    
    def lower(self, chols_flat):
        """
            Given d(d-1)/2 parameters, form a
            strictly lower-triangular matrix
        """
        L = jnp.zeros((self.d, self.d))
        L = ops.index_update(L, jnp.tril_indices(self.d), chols_flat)
        return L
    
    def reparam_sample(self, rng_key, z_mu, z_L_chol):
        std_normal_noise = random.normal(rng_key, shape=(self.d,))
        if self.corr:
            z = z_mu + (z_L_chol @ std_normal_noise)
        else: 
            z = z_mu + (jnp.diag(z_L_chol) @ std_normal_noise)
        return z

    def __call__(self, rng_key, X, corr, z_chols_max=10.):
        n = X.shape[0]
        rng_keys = rnd.split(rng_key, n)
        z_outs = self.encoder(X)
        z_mus, z_chols_flat = z_outs[:, :self.d], z_outs[:, self.d:]
        z_chols_flat = jnp.clip(z_chols_flat, -z_chols_max, z_chols_max)
        
        if corr is True:
            z_L_chols = vmap(self.lower, 0, 0)(jnp.exp(z_chols_flat))
            z_pred = vmap(self.reparam_sample, (0, 0, 0), 0)(rng_keys, z_mus, z_L_chols)
        else: 
            z_cov = vmap(jnp.diag, 0, 0)(jnp.exp(z_chols_flat))
            z_L_chols = jnp.sqrt(z_cov)
            z_pred = vmap(self.reparam_sample, (0, 0, 0), 0)(rng_keys, z_mus, jnp.exp(z_chols_flat))

        X_recons = self.decoder(z_pred)
        if self.sigmoid is False: return X_recons, z_pred, z_mus, z_L_chols
        return X_recons*255., z_pred, z_mus, z_L_chols
        