import sys
sys.path.append('dibs/')
from time import time
import jax.numpy as jnp
import jax
from jax import random, vmap
from flax import linen as nn
from jax import device_put
from jax.ops import index, index_mul
import numpy as np

from dibs.eval.target import make_graph_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture
from dibs.models.linearGaussianEquivalent import BGeJAX
from dibs.eval.metrics import expected_shd

class Decoder_DIBS(nn.Module):
    key: int
    num_nodes: int
    datatype: str
    h_latent: float
    theta_mu: float
    alpha_mu: float
    alpha_lambd: float 
    alpha_linear: float
    n_particles: int
    proj_dims: int
    latent_prior_std: float = None

    def setup(self):
        self.graph_model = make_graph_model(n_vars=self.num_nodes, graph_prior_str = self.datatype)
        self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.num_nodes), alpha_mu=1.0, alpha_lambd=self.num_nodes + 2)
        self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)
        self.bge_jax = BGeJAX(mean_obs=jnp.array([self.theta_mu]*self.num_nodes), alpha_mu=self.alpha_mu, alpha_lambd=self.alpha_lambd)

        self.dibs = MarginalDiBS(kernel=self.kernel, 
                        target_log_prior=self.log_prior, 
                        target_log_marginal_prob=self.log_likelihood, 
                        alpha_linear=self.alpha_linear)

        # Net to feed in G and predict a Z 
        self.z_net = Z_mu_logvar_Net(self.num_nodes)
        self.decoder = Decoder(self.proj_dims)
        print("INIT")

    def log_prior(self, single_w_prob):
            """log p(G) using edge probabilities as G"""    
            return self.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(self, single_w, z, no_interv_targets=None):
        if no_interv_targets is None:
            no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)
            
        log_lik = self.inference_model.log_marginal_likelihood_given_g(w=single_w, data=z, interv_targets=no_interv_targets)
        return log_lik


    def __call__(self, z_gt, z_rng, init_particles_z):

        print("inside forward")
        z_gt = device_put(z_gt, jax.devices()[0])

        # ? 1. Using init particles z, run one SVGD step on dibs and get updated particles and sample a Graph from it
        key, subk = random.split(z_rng)
        particles_z = self.dibs.sample_particles(key=key, n_steps=1, 
                                            init_particles_z=init_particles_z,
                                            data=z_gt)
        particles_g = self.dibs.particle_to_g_lim(particles_z)
        log_p_z_given_g = []
        q_z_mus, q_z_logvars, q_zs = [], [], []
        
        for g in particles_g:
            # ? 2. Get log P(z_gt|G) calculated as BGe Score ; G ~ q(G) 
            log_p_z_given_gi = self.bge_jax.log_marginal_likelihood_given_g(w=g, data=z_gt)
            log_p_z_given_g.append(log_p_z_given_gi)

            # ? 3. Get graph conditioned predictions on z: q(z|G)
            flattened_g = jnp.array(g.reshape(-1))
            flattened_g = device_put(flattened_g, jax.devices()[0])
            q_z_mu, q_z_logvar = self.z_net(flattened_g)
            q_z_mus.append(q_z_mu)
            q_z_logvars.append(q_z_logvar)

            key, z_rng = random.split(key)
            q_z = reparameterize(z_rng, q_z_mu, q_z_logvar)
            q_zs.append(q_z)

        # ? 4. From q(z|G), decode to get reconstructed samples X in higher dimensions
        recons = [] 
        for q_zi in q_zs:
            recon_xi = self.decoder(q_zi)
            recons.append(recon_xi)

        def log_likelihood(single_w, no_interv_targets=None):
            if no_interv_targets is None: no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)
            log_lik = self.inference_model.log_marginal_likelihood_given_g(w=single_w, data=z_gt, interv_targets=no_interv_targets)
            return log_lik

        eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)

        return recons, log_p_z_given_g, q_zs, q_z_mus, q_z_logvars, particles_g, particles_z, eltwise_log_prob


class Z_mu_logvar_Net(nn.Module):
    latent_dims: int

    @nn.compact
    def __call__(self, g):
        z = nn.Dense(20, name='encoder_0')(g)
        z = nn.relu(z)
        z = nn.Dense(self.latent_dims, name='encoder_1')(z)
        z = nn.relu(z)
        z = nn.Dense(self.latent_dims, name='encoder_2')(z)
        z = nn.relu(z)
        
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_0')(z)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_1')(z_mu)

        z_logvar = nn.Dense(self.latent_dims, name='logvar_encoder_0')(z)
        z_logvar = nn.relu(z_logvar)
        z_logvar = nn.Dense(self.latent_dims, name='logvar_encoder_1')(z_logvar)

        return z_mu, z_logvar


class Decoder(nn.Module):
    dims: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(10, name='decoder_fc0')(z)
        z = nn.relu(z)
        z = nn.Dense(self.dims, name='decoder_fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(self.dims, name='decoder_fc2')(z)
        return z
        

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std