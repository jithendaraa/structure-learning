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

from time import time

class VAE_DIBS(nn.Module):
  noise_sigma: float
  theta_mu: float
  num_samples: float
  theta_sigma: float
  projection: str 
  num_nodes: int 
  known_ED: bool 
  datatype: str 
  n_particles: int
  n_steps: int
  h_latent: float
  alpha_linear: float
  alpha_mu: float
  alpha_lambd: float
  proj_dims: int
  true_encoder: np.ndarray
  true_decoder: np.ndarray
  latent_prior_std: float = None

  def setup(self):
    self.use_true_encoder = False
    self.use_true_decoder = False

    self.graph_model = make_graph_model(n_vars=self.num_nodes, graph_prior_str = self.datatype)
    self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.num_nodes), alpha_mu=1.0, alpha_lambd=self.num_nodes + 2)
    self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)

    self.bge_jax = BGeJAX(mean_obs=jnp.array([self.theta_mu]*self.num_nodes),
                          alpha_mu=self.alpha_mu,
                          alpha_lambd=self.alpha_lambd)

    self.mean_logvar_net = MeanLogvarNet(self.num_nodes)

    if self.projection == 'linear':
      if self.known_ED is False:  
        self.encoder = LinearEncoder(self.num_nodes)
        self.decoder = LinearDecoder(self.proj_dims)
      
      else:
        # * encoder will be E = ((P.P_T)^-1 PX_T )_T and decoder = P
        self.use_true_encoder, self.use_true_decoder = eval(self.known_ED)
        if self.use_true_encoder is False:  self.encoder = LinearEncoder(self.num_nodes)
        if self.use_true_decoder is False:  self.decoder = LinearDecoder(self.proj_dims)
      
    self.z_post_mean_logvar_net = z_mu_logvar_net(128, self.num_nodes)
    print("Initialised VAE-DIBS")


  def __call__(self, x, z_rng, adjacency_matrix):
    x = device_put(x, jax.devices()[0])
    no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)
    E = jnp.array(self.true_encoder)
    E = device_put(E, jax.devices()[0])
    D = jnp.array(self.true_decoder)
    D = device_put(D, jax.devices()[0])

    # ? 1. Get P(z | X) ~ N(z_mu_post, z_std_post)
    if self.use_true_encoder:   z = jnp.matmul(x, E)
    else:                       z = self.encoder(x)

    z_mean_post, z_logvar_post = self.mean_logvar_net(z)
    z = reparameterize(z_rng, z_mean_post, z_logvar_post)

    def log_prior(single_w_prob):
      """log p(G) using edge probabilities as G"""    
      return self.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(single_w):
      log_lik = self.inference_model.log_marginal_likelihood_given_g(w=single_w, data=z, interv_targets=no_interv_targets)
      return log_lik

    # ? 2. Get graph posterior q(G)
    # ? Currently takes n SVGD steps to update particles and get posterior graphs
    eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)
    
    dibs = MarginalDiBS(kernel=self.kernel, target_log_prior=log_prior, 
                        target_log_marginal_prob=log_likelihood, 
                        alpha_linear=self.alpha_linear)

    key, subk = random.split(z_rng)
    init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=self.n_particles, n_vars=self.num_nodes)
    
    key, subk = random.split(key)
    particles_z = dibs.sample_particles(key=subk, n_steps=self.n_steps, init_particles_z=init_particles_z)
    particles_g = dibs.particle_to_g_lim(particles_z)

    log_p_z_given_g = []
    q_z_mus, q_z_logvars, q_zs = [], [], []
    
    for g in particles_g:
      # ? 3. Get regularized z prior = log P(z|G); G ~ q(G) = marginal log likelihood calculated as BGe Score
      log_p_z_given_gi = self.bge_jax.log_marginal_likelihood_given_g(w=g, data=z)
      log_p_z_given_g.append(log_p_z_given_gi)

      # ? 4. Get graph conditioned latent variable distribution q(z|G)
      flattened_g = jnp.array(g.reshape(-1))
      flattened_g = device_put(flattened_g, jax.devices()[0])
      q_z_mu, q_z_logvar = self.z_post_mean_logvar_net(flattened_g)
      q_z_mus.append(q_z_mu)
      q_z_logvars.append(q_z_logvar)

      key, z_rng = random.split(key)
      q_z = reparameterize(z_rng, q_z_mu, q_z_logvar)
      q_zs.append(q_z)

    recons = [] 
    for q_zi in q_zs:
      if self.use_true_decoder is False: recon_xi = self.decoder(q_zi)
      else: recon_xi = jnp.matmul(q_zi, D)
      recons.append(recon_xi)

    return recons, log_p_z_given_g, q_zs, q_z_mus, q_z_logvars, particles_g, eltwise_log_prob


class MeanLogvarNet(nn.Module):
  latent_dims: int
  
  @nn.compact
  def __call__(self, z):
    # ? 1. Get P(z | X) ~ N(z_mu_post, z_std_post)
    # ? 2. Calculate posterior z_mu and posterior z_logvar and 
    post_z_mean = nn.Dense(self.latent_dims, name='z_post_mean')(z)
    post_z_logvar = nn.Dense(self.latent_dims, name='z_post_logvar')(z)
    return post_z_mean, post_z_logvar

class LinearEncoder(nn.Module):
  latent_dims: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(15, name='fc1')(x)
    z = nn.Dense(self.latent_dims, name='fc2')(x)
    return z

class LinearDecoder(nn.Module):
    dims: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(10, name='decoder_fc1')(z)
        z = nn.Dense(self.dims, name='decoder_fc2')(z)
        return z

class z_mu_logvar_net(nn.Module):
  latents: int
  num_nodes: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32)(x)
    x = nn.relu(x)
    x = nn.Dense(self.latents)(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.num_nodes)(x)
    logvar_x = nn.Dense(self.num_nodes)(x)
    return mean_x, logvar_x

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std