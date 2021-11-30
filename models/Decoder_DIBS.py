import sys
sys.path.append('dibs/')
from time import time
import jax.numpy as jnp
import jax
from jax import random, vmap
from flax import linen as nn
from jax import device_put

from dibs.eval.target import make_graph_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.models.linearGaussianEquivalent import BGeJAX

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
    num_updates: int = 5
    latent_prior_std: float = None

    def setup(self):
        self.graph_model = make_graph_model(n_vars=self.num_nodes, graph_prior_str = self.datatype)
        self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.num_nodes), alpha_mu=1.0, alpha_lambd=self.num_nodes + 2)
        self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)
        self.bge_jax = BGeJAX(mean_obs=jnp.array([self.theta_mu]*self.num_nodes), alpha_mu=self.alpha_mu, alpha_lambd=self.alpha_lambd)
        self.dibs = MarginalDiBS(kernel=self.kernel, target_log_prior=self.log_prior, target_log_marginal_prob=self.log_likelihood, alpha_linear=self.alpha_linear)

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


    def __call__(self, z_gt, z_rng, init_particles_z, opt_state_z, sf_baseline, step=0):
        log_p_z_given_g, q_z_mus, q_z_logvars, q_zs, recons = [], [], [], [], []
        
        # ? 1. Using init particles z, run 'num_updates' SVGD step on dibs and get updated particles and sample a Graph from it
        particles_z, opt_state_z, sf_baseline = self.dibs.sample_particles(n_steps=self.num_updates, init_particles_z=init_particles_z,
                                                                key=self.key, opt_state_z=opt_state_z, sf_baseline=sf_baseline, 
                                                                data=z_gt, start=self.num_updates*step)
        particles_g = self.dibs.particle_to_g_lim(particles_z)
        
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

            key, z_rng = random.split(z_rng)
            q_z = reparameterize(z_rng, q_z_mu, q_z_logvar)
            q_zs.append(q_z)

        # ? 4. From q(z|G), decode to get reconstructed samples X in higher dimensions
        for q_zi in q_zs:   recons.append(self.decoder(q_zi))

        return jnp.array(recons), log_p_z_given_g, jnp.array(q_z_mus), jnp.array(q_z_logvars), particles_g, particles_z, opt_state_z, sf_baseline


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