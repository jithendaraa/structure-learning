from flax import linen as nn
from jax import numpy as jnp

class Z_mu_logvar_Net(nn.Module):
    latent_dims: int
    num_cholesky_terms: int

    @nn.compact
    def __call__(self, g):
        z = nn.Dense(10 * self.latent_dims * self.latent_dims, name='encoder_0')(g)
        z = nn.relu(z)
        z = nn.Dense(10 * self.latent_dims * self.latent_dims, name='encoder_1')(z)
        z = nn.relu(z)
        
        z_mu = nn.Dense(10 * self.latent_dims, name='mu_encoder_0')(z)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(10 * self.latent_dims, name='mu_encoder_1')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(5 * self.latent_dims * self.latent_dims, name='mu_encoder_2')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_3')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_4')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_5')(z_mu)

        z_logcholesky = nn.Dense(10 * self.latent_dims*self.latent_dims, name='logcovar_encoder_0')(z)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(5 * self.latent_dims*self.latent_dims, name='logcovar_encoder_1')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(5 * self.latent_dims*self.latent_dims, name='logcovar_encoder_2')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_3')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_4')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_5')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.num_cholesky_terms, name='logcovar_encoder_6')(z_logcholesky)

        return jnp.asarray(z_mu), jnp.asarray(z_logcholesky)


class Decoder(nn.Module):
    dims: int
    linear_decoder: bool

    @nn.compact
    def __call__(self, z):
        if self.linear_decoder:
            z = nn.Dense(self.dims, name='decoder_fc0')(z)
        else:
            z = nn.Dense(self.dims, name='decoder_fc0')(z)
            z = nn.relu(z)
            z = nn.Dense(self.dims, name='decoder_fc1')(z)
            z = nn.relu(z)
            z = nn.Dense(self.dims, name='decoder_fc2')(z)
            z = nn.relu(z)
            z = nn.Dense(self.dims, name='decoder_fc3')(z)
            z = nn.relu(
                z)
            z = nn.Dense(self.dims, name='decoder_fc4')(z)
            z = nn.relu(z)
            z = nn.Dense(self.dims, name='decoder_fc5')(z)
            z = nn.relu(z)
            z = nn.Dense(self.dims, name='decoder_fc6')(z)
            
        return z