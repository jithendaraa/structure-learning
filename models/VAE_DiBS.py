import jax.numpy as jnp
import jax
from jax import random, vmap
from flax import linen as nn
from jax import device_put
from jax.ops import index, index_mul


from dibs.eval.target import make_graph_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture
from dibs.models.linearGaussianEquivalent import BGeJAX
from dibs.eval.metrics import expected_shd

from time import time

class VAE_DIBS(nn.Module):
  key: int
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
  latent_prior_std: float = None

  def setup(self):
    key, rng = random.split(self.key)

    self.graph_model = make_graph_model(n_vars=self.num_nodes, graph_prior_str = self.datatype)
    self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.num_nodes), alpha_mu=1.0, alpha_lambd=self.num_nodes + 2)

    self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)
    self.bge_jax = BGeJAX(mean_obs=jnp.array([self.theta_mu]*self.num_nodes),
                          alpha_mu=self.alpha_mu,
                          alpha_lambd=self.alpha_lambd)


    if self.projection == 'linear':
      self.encoder = LinearEncoder(self.num_nodes)
      if self.known_ED is False:  self.decoder = LinearDecoder(self.proj_dims)
    

    self.z_post_mean_logvar_net = z_mu_logvar_net(128, self.num_nodes)
    print("Initialised VAE-DIBS")


  def __call__(self, x, z_rng, adjacency_matrix):
    x = device_put(x, jax.devices()[0])
    no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)

    # ? 1. Get P(z | X) ~ N(z_mu_post, z_std_post)
    z_mean_post, z_logvar_post = self.encoder(x)
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
    s = time()
    particles_z = dibs.sample_particles(key=subk, n_steps=self.n_steps, init_particles_z=init_particles_z)
    print(f'DiBS took {time() - s}s')
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
      recon_xi = self.decoder(q_zi)
      recons.append(recon_xi)

    return recons, log_p_z_given_g, q_zs, q_z_mus, q_z_logvars, particles_g, eltwise_log_prob



  def sample_initial_random_particles(self, *, key, n_particles, n_vars, n_dim=None):
    """
    Samples random particles to initialize SVGD

    Args:
        key: rng key
        n_particles: number of particles for SVGD
        n_particles: number of variables `d` in inferred BN 
        n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

    Returns:
        z: batch of latent tensors [n_particles, d, k, 2]    
    """
    # default full rank
    if n_dim is None: n_dim = n_vars 
    std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim)) # like prior

    # sample
    key, subk = random.split(key)
    z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std        
    return z

  def sample_particles(self, *, n_steps, init_particles_z, key, callback=None, callback_every=0):
    """
    Deterministically transforms particles to minimize KL to target using SVGD

    Arguments:
        n_steps (int): number of SVGD steps performed
        init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
        key: prng key
        callback: function to be called every `callback_every` steps of SVGD.
        callback_every: if == 0, `callback` is never called. 

    Returns: 
        `n_particles` samples that approximate the DiBS target density
        particles_z: [n_particles, d, k, 2]
    """
    z = init_particles_z
    # initialize score function baseline (one for each particle)
    n_particles, _, n_dim, _ = z.shape
    sf_baseline = jnp.zeros(n_particles)

    if self.latent_prior_std is None: 
      self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

  def particle_to_g_lim(self, z):
        """
        Returns g corresponding to alpha = infinity for particles `z`

        Args:
            z: latent variables [..., d, k, 2]

        Returns:
            graph adjacency matrices of shape [..., d, d]
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        g_samples = (scores > 0).astype(jnp.int32)

        # zero diagonal
        g_samples = index_mul(g_samples, index[..., jnp.arange(scores.shape[-1]), jnp.arange(scores.shape[-1])], 0)
        return g_samples

class LinearEncoder(nn.Module):
  latent_dims: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(10, name='fc1')(x)
    
    # ? 1. Get P(z | X) ~ N(z_mu_post, z_std_post)
    # ? 2. Calculate posterior z_mu and posterior z_logvar and 
    z = nn.Dense(self.latent_dims, name='fc2')(x)
    post_z_mean = nn.Dense(self.latent_dims, name='z_post_mean')(z)
    post_z_logvar = nn.Dense(self.latent_dims, name='z_post_logvar')(z)
    return post_z_mean, post_z_logvar


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

# @jax.jit
# def train_step(state, batch, z_rng):
#   def loss_fn(params):
#     recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)

#     kld_loss = kl_divergence(mean, logvar).mean()
#     loss = bce_loss + kld_loss
#     return loss
#   grads = jax.grad(loss_fn)(state.params)
#   return state.apply_gradients(grads=grads)

# @jax.vmap
# def kl_divergence(mean, logvar):
#   return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


# def compute_metrics(recon_x, x, mean, logvar):
#   kld_loss = kl_divergence(mean, logvar).mean()
#   return {
#       'kld': kld_loss,
#       'loss': bce_loss + kld_loss
#   }


# def model():
#   return VAE(latents=FLAGS.latents)