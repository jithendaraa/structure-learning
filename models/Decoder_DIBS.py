import sys
sys.path.append('dibs/')
from time import time
import tqdm

import jax.numpy as jnp
import jax
from jax import random, vmap, grad
from flax import linen as nn
from jax.ops import index, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.scipy.special import logsumexp


from dibs.eval.target import make_graph_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS, dibs
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
    grad_estimator_z: str = 'reparam'
    score_function_baseline: float = 0.0
    n_grad_mc_samples: int = 128
    tau: float = 1.0

    def setup(self):
        self.graph_model = make_graph_model(n_vars=self.num_nodes, graph_prior_str = self.datatype)
        self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.num_nodes), alpha_mu=1.0, alpha_lambd=self.num_nodes + 2)
        self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)
        self.bge_jax = BGeJAX(mean_obs=jnp.array([self.theta_mu]*self.num_nodes), alpha_mu=self.alpha_mu, alpha_lambd=self.alpha_lambd)
        self.dibs = MarginalDiBS(kernel=self.kernel, 
                target_log_prior=self.log_prior, 
                target_log_marginal_prob=self.log_likelihood, 
                alpha_linear=self.alpha_linear,
                grad_estimator_z=self.grad_estimator_z)

        self.alpha = lambda t: (self.alpha_linear * t)
        self.target_log_joint_prob = lambda single_z, single_theta, subk, data: self.log_likelihood(single_z, data)

        # Net to feed in G and predict a Z 
        self.z_net = Z_mu_logvar_Net(self.num_nodes)
        self.decoder = Decoder(self.proj_dims)
        print("INIT")

    def log_prior(self, single_w_prob):
            """log p(G) using edge probabilities as G"""    
            return self.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(self, single_w, z, no_interv_targets=None):
        if no_interv_targets is None:   no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)
        print("getting log likelihood", self.grad_estimator_z, ":", single_w.shape, z.shape, no_interv_targets) 
        log_lik = self.inference_model.log_marginal_likelihood_given_g(w=single_w, data=z, interv_targets=no_interv_targets)
        print("F")
        return log_lik

    def vec_to_mat(self, z, n_vars):
        """
        Reshapes particle to latent adjacency matrix form
            last dim gets shaped into matrix
        
        Args:
            w: flattened matrix of shape [..., d * d]

        Returns:
            matrix of shape [..., d, d]
        """
        return z.reshape(*z.shape[:-1], n_vars, n_vars)

    def mat_to_vec(self, w):
        """
        Reshapes latent adjacency matrix form to particle
            last two dims get flattened into vector
        
        Args:
            w: matrix of shape [..., d, d]
        
        Returns:
            flattened matrix of shape [..., d * d]
        """
        n_vars = w.shape[-1]
        return w.reshape(*w.shape[:-2], n_vars * n_vars)

    # * Generative graph model p(G | Z); Z is particles
    def edge_probs(self, z, t):
        """
        Edge probabilities encoded by latent representation 

        Args:
            z: latent tensors Z [..., d, k, 2]
            t: step
        
        Returns:
            edge probabilities of shape [..., d, d]
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        probs =  sigmoid(self.alpha(t) * scores)

        # mask diagonal since it is explicitly not modeled
        probs = index_mul(probs, index[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])], 0.0)
        return probs

    def edge_log_probs(self, z, t):
        """
        Edge log probabilities encoded by latent representation

        Args:
            z: latent tensors Z [..., d, k, 2]
            t: step

        Returns:
            tuple of tensors [..., d, d], [..., d, d] corresponding to log(p) and log(1-p)
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        log_probs, log_probs_neg =  log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)

        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        log_probs = index_mul(log_probs, index[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])], 0.0)
        log_probs_neg = index_mul(log_probs_neg, index[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])], 0.0)
        return log_probs, log_probs_neg

    def latent_log_prob(self, single_g, single_z, t):
        """
        Log likelihood of generative graph model

        Args:
            single_g: single graph adjacency matrix [d, d]    
            single_z: single latent tensor [d, k, 2]
            t: step
        
        Returns:
            log likelihood log p(G | Z) of shape [1,]
        """
        # [d, d], [d, d]
        log_p, log_1_p = self.edge_log_probs(single_z, t)

        # [d, d]
        log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = jnp.sum(log_prob_g_ij)

        return log_prob_g

    def eltwise_grad_latent_log_prob(self, gs, single_z, t):
        """
        Gradient of log likelihood of generative graph model w.r.t. Z
        i.e. d/dz log p(G | Z) 
        Batched over samples of G given a single Z.

        Args:
            gs: batch of graph matrices [n_graphs, d, d]
            single_z: latent variable [d, k, 2] 
            t: step

        Returns:
            batch of gradients of shape [n_graphs, d, k, 2]
        """
        dz_latent_log_prob = grad(self.latent_log_prob, 1)
        return vmap(dz_latent_log_prob, (0, None, None), 0)(gs, single_z, t)

    # * Estimators for scores of log p(theta, D | Z); Z is particles
    def eltwise_log_joint_prob(self, gs, single_theta, rng, data):
        """
        log p(data | G, theta) batched over samples of G

        Args:
            gs: batch of graphs [n_graphs, d, d]
            single_theta: single parameter PyTree
            rng:  [1, ]

        Returns:
            batch of logprobs [n_graphs, ]
        """
        return vmap(self.target_log_joint_prob, (0, None, None, None), 0)(gs, single_theta, rng, data)

    def log_joint_prob_soft(self, single_z, single_theta, eps, t, subk, data):
        """
        This is the composition of 
            log p(theta, D | G) 
        and
            G(Z, U)  (Gumbel-softmax graph sample given Z)

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            eps: i.i.d Logistic noise of shpae [d, d] 
            t: step 
            subk: rng key

        Returns:
            logprob of shape [1, ]
        """
        soft_g_sample = self.particle_to_soft_graph(single_z, eps, t)
        return self.target_log_joint_prob(soft_g_sample, single_theta, subk, data)

    def sample_g(self, p, subk, n_samples):
        """
        Sample Bernoulli matrix according to matrix of probabilities

        Args:
            p: matrix of probabilities [d, d]
            n_samples: number of samples
            subk: rng key
        
        Returns:
            an array of matrices sampled according to `p` of shape [n_samples, d, d]
        """
        n_vars = p.shape[-1]
        g_samples = self.vec_to_mat(random.bernoulli(
            subk, p=self.mat_to_vec(p), shape=(n_samples, n_vars * n_vars)), n_vars).astype(jnp.int32)

        # mask diagonal since it is explicitly not modeled
        g_samples = index_mul(g_samples, index[..., jnp.arange(p.shape[-1]), jnp.arange(p.shape[-1])], 0)

        return g_samples

    # * Estimators for score d/dZ log p(theta, D | Z)  (i.e. w.r.t the latent embeddings Z for graph G)
    def eltwise_grad_z_likelihood(self, zs, thetas, baselines, t, subkeys, data=None):
        if self.grad_estimator_z == 'score':    grad_z_likelihood = self.grad_z_likelihood_score_function
        elif self.grad_estimator_z == 'reparam':    grad_z_likelihood = self.grad_z_likelihood_gumbel
        else:   raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')
        return vmap(grad_z_likelihood, (0, 0, 0, None, 0, None), (0, 0))(zs, thetas, baselines, t, subkeys, data)

    def grad_z_likelihood_score_function(self, single_z, single_theta, single_sf_baseline, t, subk, data=None):
        """
        Calculates d/dZ log p(theta, data | particles z) 
        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]
            t: step
            subk: rng key
        Returns:    tuple gradient, baseline  [d, k, 2], [1, ]
        """
        p = self.edge_probs(single_z, t)
        n_vars, n_dim = single_z.shape[0:2]

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p, subk_, self.n_grad_mc_samples)
        n_mc_numerator, n_mc_denominator = self.n_grad_mc_samples, self.n_grad_mc_samples

        # [n_grad_mc_samples, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, subk_, data)
        logprobs_denominator = logprobs_numerator

        # variance_reduction
        logprobs_numerator_adjusted = lax.cond(
            self.score_function_baseline <= 0.0,
            lambda _: logprobs_numerator,
            lambda _: logprobs_numerator - single_sf_baseline,
            operand=None)
        
        # ? [d * k * 2, n_grad_mc_samples]
        grad_z = self.eltwise_grad_latent_log_prob(g_samples, single_z, t)\
            .reshape(self.n_grad_mc_samples, n_vars * n_dim * 2)\
            .transpose((1, 0))

        # ? stable computation of exp/log/divide  [d * k * 2, ]  [d * k * 2, ]
        log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_z, axis=1, return_sign=True)
        log_denominator = logsumexp(logprobs_denominator, axis=0) # []

        # [d * k * 2, ]
        stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        # [d, k, 2]
        stable_sf_grad_shaped = stable_sf_grad.reshape(n_vars, n_dim, 2)

        # update baseline
        single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                            (1 - self.score_function_baseline) * single_sf_baseline)

        return stable_sf_grad_shaped, single_sf_baseline

    # * reparametrized estimation of d/dZ log p(theta, D | Z)
    def grad_z_likelihood_gumbel(self, single_z, single_theta, single_sf_baseline, t, subk, data=None):
        """
        Reparameterization estimator for the score d/dZ log p(theta, D | Z) 
        Using the Gumbel-softmax / concrete distribution reparameterization trick.
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]

        Returns:
            tuple: gradient, baseline of shape [d, k, 2], [1, ]

        """   
        print("IN")
        n_vars = single_z.shape[0]
        n_mc_numerator, n_mc_denominator = self.n_grad_mc_samples, self.n_grad_mc_samples

        # sample Logistic(0,1) as randomness in reparameterization
        subk, subk_ = random.split(subk)
        eps = random.logistic(subk_, shape=(self.n_grad_mc_samples, n_vars, n_vars))                

        # [n_grad_mc_samples, ]
        # since we don't backprop per se, it leaves us with the option of having
        # `soft` and `hard` versions for evaluating the non-grad p(.))
        subk, subk_ = random.split(subk)
       
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples]
        print("HMM")
        import pdb; pdb.set_trace()
        logprobs_numerator = vmap(self.log_joint_prob_soft, (None, None, 0, None, None, None), 0)(single_z, single_theta, eps, t, subk_, data) 
        logprobs_denominator = logprobs_numerator
        print("HA")
        # [n_grad_mc_samples, d, k, 2]
        # d/dx log p(theta, D | G(x, eps)) for a batch of `eps` samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples, d, k, 2]
        grad_z = vmap(grad(self.log_joint_prob_soft, 0), (None, None, 0, None, None, None), 0)(single_z, single_theta, eps, t, subk_, data)

        # stable computation of exp/log/divide  [d, k, 2], [d, k, 2]
        log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None, None], b=grad_z, axis=0, return_sign=True)
        log_denominator = logsumexp(logprobs_denominator, axis=0)   # []

        # [d, k, 2]
        stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))
        return stable_grad, single_sf_baseline

    def particle_to_soft_graph(self, z, eps, t):
        """ 
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples `eps`

        Args:
            z: a single latent tensor Z of shape [d, k, 2]
            eps: random iid Logistic(0,1) noise  of shape [d, d] 
            t: step
        
        Returns:
            Gumbel-softmax sample of adjacency matrix [d, d]
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        soft_graph = sigmoid(self.tau * (eps + self.alpha(t) * scores))

        # mask diagonal since it is explicitly not modeled
        n_vars = soft_graph.shape[-1]
        soft_graph = index_mul(soft_graph, index[..., jnp.arange(n_vars), jnp.arange(n_vars)], 0.0)
        return soft_graph

    def sample_particles(self, start, z, dibs_params, key, sf_baseline, data=None):
        n_steps = self.num_updates
        it = tqdm.tqdm(jnp.arange(start, start+n_steps), desc='DiBS')
        
        for t in it:
            dibs_params, key, sf_baseline = self.svgd_step(z, dibs_params, key, sf_baseline, t, data)     #
            break
    

    def __call__(self, z_gt, z_rng, init_particles_z, dibs_params, sf_baseline, step=0):
        log_p_z_given_g, q_z_mus, q_z_logvars, q_zs, recons = [], [], [], [], []
        
        # ? 1. Using init particles z, run 'num_updates' SVGD step on dibs; update particles; sample graphs
        particles_z, dibs_params, sf_baseline = self.dibs.sample_particles(n_steps=self.num_updates, init_particles_z=init_particles_z, key=self.key, opt_state_z=dibs_params, sf_baseline=sf_baseline, data=z_gt, start=self.num_updates*step)
        # self.sample_particles(self.num_updates*step, init_particles_z, dibs_params, self.key, sf_baseline, 
        #                     data=z_gt)
        
        # print("GOTCHA")
        # particles_g = self.dibs.particle_to_g_lim(particles_z) # ! dont do this; get soft G's

        # s = time()
        # for g in particles_g:
        #     # ? 2. Get log P(z_gt|G) calculated as BGe Score ; G ~ q(G) 
        #     log_p_z_given_gi = self.bge_jax.log_marginal_likelihood_given_g(w=g, data=z_gt)
        #     log_p_z_given_g.append(log_p_z_given_gi)

        #     # ? 3. Get graph conditioned predictions on z: q(z|G)
        #     flattened_g = jnp.array(g.reshape(-1))
        #     flattened_g = device_put(flattened_g, jax.devices()[0])
        #     q_z_mu, q_z_logvar = self.z_net(flattened_g)
        #     q_z_mus.append(q_z_mu)
        #     q_z_logvars.append(q_z_logvar)

        #     key, z_rng = random.split(z_rng)
        #     q_z = reparameterize(z_rng, q_z_mu, q_z_logvar)
        #     q_zs.append(q_z)

        # # ? 4. From q(z|G), decode to get reconstructed samples X in higher dimensions
        # for q_zi in q_zs:   recons.append(self.decoder(q_zi))
        # print(f"Rest of Decoder_DIBS forward method took {time()-s}s")

        # return jnp.array(recons), log_p_z_given_g, jnp.array(q_z_mus), jnp.array(q_z_logvars), particles_g, particles_z, dibs_params, sf_baseline
        # return None, None, None, None, None, None, None, None

    def svgd_step(self, z, dibs_params, key, sf_baseline, t, data=None):
        h = self.kernel.h

        # ? d/dz log p(data | z) grad log likelihood
        key, *batch_subk = random.split(key, self.n_particles + 1) 
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, None, sf_baseline, t, jnp.array(batch_subk), data)
        print("Got log likelihood P(Data | G)")

        return None, None, None

    

    
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