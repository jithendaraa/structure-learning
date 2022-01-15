from re import A
import sys

from jax._src.numpy.linalg import cholesky

sys.path.append('dibs/')
from time import time
import numpy as np
import functools
import utils

import jax, torch
torch.manual_seed(0)
import jax.numpy as jnp
from jax import random, vmap, grad, device_put, jit
from flax import linen as nn
from jax.ops import index, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.scipy.special import logsumexp, gammaln


from dibs.eval.target import make_graph_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.models.linearGaussianEquivalent import BGeJAX
from dibs.utils.graph import acyclic_constr_nograd
from dibs.utils.func import leftsel
from dibs.graph.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection


def reparameterized_multivariate_normal(rng, mean, cholesky_L, size):
    # Cholesky decomposition: Σ = LL.T where L is lower triangular
    # reparametrised sample `res = mean + eps * L`
    d = cholesky_L.shape[0]
    unit_cov_matrix = jnp.identity(d)
    unit_cov_matrix = jnp.expand_dims(unit_cov_matrix, 0).repeat(size, axis=0)
    
    standard_mean = jnp.zeros((size, d))
    eps = random.multivariate_normal(rng, standard_mean, unit_cov_matrix)
    return mean + jnp.matmul(eps, cholesky_L)

def edge_log_probs(z, t):
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
    log_probs, log_probs_neg =  log_sigmoid(t * scores), log_sigmoid(t * -scores)

    # mask diagonal since it is explicitly not modeled
    # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
    log_probs = index_mul(log_probs, index[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])], 0.0)
    log_probs_neg = index_mul(log_probs_neg, index[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])], 0.0)
    return log_probs, log_probs_neg


class Decoder_DIBS(nn.Module):
    num_nodes: int
    datatype: str
    h_latent: float
    alpha_mu: float
    alpha_lambd: float 
    alpha_linear: float
    n_particles: int
    proj_dims: int
    num_samples: int
    linear_decoder: bool
    latent_prior_std: float
    proj_matrix: jnp.array
    grad_estimator_z: str = 'reparam'
    score_function_baseline: float = 0.0
    n_grad_mc_samples: int = 128
    tau: float = 1.0
    beta_linear: float = 1.0
    n_acyclicity_mc_samples: int = 32
    scale: float = 1.0
    edges_per_node: float = 1.0
    known_ED: bool = False

    def setup(self):
        # For BGeJAX
        self.mean_obs = jnp.zeros(self.num_nodes)
        self.n_vars = self.num_nodes
        self.n_edges = self.edges_per_node * self.num_nodes
        self.p = self.n_edges / ((self.n_vars * (self.n_vars - 1)) / 2)
        self.no_interv_targets = jnp.zeros(self.num_nodes).astype(bool)

        if self.datatype not in ['er']:
            NotImplementedError(f"Not implemented {self.datatype}")

        # Net to feed in G and predict a Z 
        num_cholesky_terms = int(self.num_nodes * (self.num_nodes + 1) / 2.0)
        self.z_net = Z_mu_logvar_Net(self.num_nodes, num_cholesky_terms)
        
        self.inference_model = BGeJAX(mean_obs=jnp.zeros(self.n_vars), alpha_mu=1.0, alpha_lambd=self.n_vars + 2)
        self.kernel = FrobeniusSquaredExponentialKernel(h=self.h_latent)

        if self.known_ED is False:
            self.decoder = Decoder(self.proj_dims, self.linear_decoder)
        print("INIT")

    def f_kernel(self, x_latent, y_latent):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            y_latent: latent tensor [d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [1, ] kernel value
        """
        return self.kernel.eval(x=x_latent, y=y_latent, h=self.h_latent)
        
    def f_kernel_mat(self, x_latents, y_latents):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            y_latents: latent tensor [B, d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, 0), 0), (0, None), 0)(x_latents, y_latents)

    def log_prior(self, soft_g):
        """log p(G) using edge probabilities as G"""    
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = soft_g.sum()
        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)

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

    # * Generative graph model p(G | particles Z)
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
        probs =  sigmoid(self.alpha_linear * t * scores)

        # mask diagonal since it is explicitly not modeled
        probs = index_mul(probs, index[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])], 0.0)
        return probs

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
        log_p, log_1_p = edge_log_probs(single_z, self.alpha_linear * t)

        # [d, d]
        log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = jnp.sum(log_prob_g_ij)

        return log_prob_g

    def target_log_joint_prob(self, single_w, data):
        log_lik = self.inference_model.log_marginal_likelihood_given_g(w=single_w, data=data, interv_targets=self.no_interv_targets)
        return log_lik

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
        return self.target_log_joint_prob(soft_g_sample, data)

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

    # * Estimators for scores of log p(theta, D | particles Z)
    def eltwise_log_joint_prob(self, gs, data):
        """
        log p(data | G, theta) batched over samples of G

        Args:
            gs: batch of graphs [n_graphs, d, d]
            single_theta: single parameter PyTree
            rng:  [1, ]

        Returns:
            batch of logprobs [n_graphs, ]
        """
        return vmap(self.target_log_joint_prob, (0, None), 0)(gs, data)

    def constraint_gumbel(self, single_z, single_eps, alpha_t):
        """ 
        Evaluates continuous acyclicity constraint using Gumbel-softmax instead of Bernoulli samples
        Args:
            single_z: single latent tensor [d, k, 2]
            single_eps: i.i.d. Logistic noise of shape [d, d] for Gumbel-softmax
            t: step
        
        Returns:
            constraint value of shape [1,]
        """
        n_vars = single_z.shape[0]
        G = self.particle_to_soft_graph(single_z, single_eps, alpha_t)
        h = acyclic_constr_nograd(G, n_vars)
        return h

    # * Reparameterization estimator for the gradient d/dZ E_p(G|Z) [constraint(G)]
    def grad_constraint_gumbel(self, single_z, key, alpha_t):
        """
        Using the Gumbel-softmax / concrete distribution reparameterization trick.
        Args:
            z: single latent tensor [d, k, 2]                
            key: rng key [1,]    
            t: step
        Returns         
            gradient of constraint [d, k, 2] 
        """
        n_vars = single_z.shape[0]
        eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, n_vars, n_vars))    # [n_mc_samples, d, d]
        # [n_mc_samples, d, k, 2]
        mc_gradient_samples = vmap(grad(self.constraint_gumbel, 0), (None, 0, None), 0)(single_z, eps, alpha_t)
        return mc_gradient_samples.mean(0) # [d, k, 2]

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

    def eltwise_grad_z_likelihood(self, zs, thetas, baselines, t, subkeys, data):
        if self.grad_estimator_z == 'reparam':    
            grad_z_likelihood = self.grad_z_likelihood_gumbel
        elif self.grad_estimator_z == 'score':
            grad_z_likelihood = self.grad_z_likelihood_score_function
        else:   
            raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')
        
        return vmap(grad_z_likelihood, (0, 0, 0, None, 0, None), (0, 0))(zs, thetas, baselines, t, subkeys, data)

    def grad_z_likelihood_score_function(self, single_z, single_theta, single_sf_baseline, t, subk, data=None):
        """
        Score function estimator (aka REINFORCE) for the score d/dZ log p(theta, D | Z) 
        This does not use d/dG log p(theta, D | G) and is hence applicable when not defined.
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]
            t: step
            subk: rng key
        
        Returns:
            tuple gradient, baseline  [d, k, 2], [1, ]

        """
        p = self.edge_probs(single_z, t)    # [d, d]
        n_vars, n_dim = single_z.shape[0:2]

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p, subk_, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_grad_mc_samples, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, data)
        logprobs_denominator = logprobs_numerator

        # variance_reduction
        logprobs_numerator_adjusted = lax.cond(
            self.score_function_baseline <= 0.0,
            lambda _: logprobs_numerator,
            lambda _: logprobs_numerator - single_sf_baseline,
            operand=None)

        # [d * k * 2, n_grad_mc_samples]
        grad_z = self.eltwise_grad_latent_log_prob(g_samples, single_z, t)\
            .reshape(self.n_grad_mc_samples, n_vars * n_dim * 2)\
            .transpose((1, 0))

        # stable computation of exp/log/divide
        # [d * k * 2, ]  [d * k * 2, ]
        log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_z, axis=1, return_sign=True)
        log_denominator = logsumexp(logprobs_denominator, axis=0)   # []

        # [d * k * 2, ]
        stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        # [d, k, 2]
        stable_sf_grad_shaped = stable_sf_grad.reshape(n_vars, n_dim, 2)

        # update baseline
        single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                            (1 - self.score_function_baseline) * single_sf_baseline)

        return stable_sf_grad_shaped, single_sf_baseline

    def grad_z_likelihood_gumbel(self, single_z, single_theta, single_sf_baseline, t, subk, data):
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
        logprobs_numerator = vmap(self.log_joint_prob_soft, (None, None, 0, None, None, None), 0)(single_z, single_theta, eps, t, subk_, data) 
        logprobs_denominator = logprobs_numerator
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

    # * [target_log_prior_particle, eltwise_grad_latent_prior]
    # * Computes gradient of the prior: d/dZ log p(Z) 
    def target_log_prior_particle(self, single_z, t):
        """
        log p(Z) approx. log p(G) via edge probabilities

        Args:
            single_z: single latent tensor [d, k, 2]
            t: step

        Returns:
            log prior graph probability [1,] log p(G) evaluated with G_\alpha(Z)
                i.e. with the edge probabilities   
        """
        # [d, d] # masking is done inside `edge_probs`
        single_soft_g = self.edge_probs(single_z, t)
        return self.log_prior(single_soft_g) # [1, ]

    def eltwise_grad_latent_prior(self, zs, subkeys, t):
        """
        where log p(Z) = - beta(t) E_p(G|Z) [constraint(G)] + log Gaussian(Z) + log f(Z) 
        and f(Z) is an additional prior factor.

        Args:
            zs: single latent tensor  [n_particles, d, k, 2]
            subkeys: batch of rng keys [n_particles, ...]

        Returns:
            batch of gradients of shape [n_particles, d, k, 2]
        """
        # log f(Z) term [d, k, 2], [1,] -> [d, k, 2]
        alpha_t, beta_t = self.alpha_linear * t, self.beta_linear * t
        grad_target_log_prior_particle = grad(self.target_log_prior_particle, 0)

        # [n_particles, d, k, 2], [1,] -> [n_particles, d, k, 2]
        grad_prior_z = vmap(grad_target_log_prior_particle, (0, None), 0)(zs, t)

        # constraint term  [n_particles, d, k, 2], [n_particles,], [1,] -> [n_particles, d, k, 2]
        eltwise_grad_constraint = vmap(self.grad_constraint_gumbel, (0, 0, None), 0)(zs, subkeys, t)

        return - beta_t * eltwise_grad_constraint \
                - zs / (self.latent_prior_std ** 2.0) \
                + grad_prior_z 
    

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
        soft_graph = sigmoid(self.tau * (eps + self.alpha_linear * t * scores))

        # mask diagonal since it is explicitly not modeled
        n_vars = soft_graph.shape[-1]
        soft_graph = index_mul(soft_graph, index[..., jnp.arange(n_vars), jnp.arange(n_vars)], 0.0)
        return soft_graph

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
    
    # * [z_update, eltwise_grad_kernel_z] 
    # * Calculating Calculate phi_z = Mean ( kxx * grad_log P(particles_z | z_gt) + grad kxx ) for updating particles_z
    def z_update(self, single_z, kxx, z, grad_log_prob_z):
        """
        Computes SVGD update for `single_z` particle given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        
        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self.eltwise_grad_kernel_z(z, single_z)
        res = - (weighted_gradient_ascent + repulsion).mean(axis=0)

        # average and negate (for optimizer)
        return res

    def eltwise_grad_kernel_z(self, x_latents, y_latent):
        """
        Computes gradient d/dz k(z, z') elementwise for each provided particle z

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            y_latent: single latent particle [d, k, 2] (z')
            h (float): kernel bandwidth 
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        """        
        grad_kernel_z = jit(grad(self.f_kernel, 0))
        return vmap(grad_kernel_z, (0, None), 0)(x_latents, y_latent)


    def get_posterior_single_z(self, key, single_g):
        flattened_g = jnp.array(single_g.reshape(-1))
        q_z_mu, q_z_logcholesky = self.z_net(flattened_g)
        q_z_cholesky = jnp.exp(q_z_logcholesky)

        tril_indices = jnp.tril_indices(self.num_nodes)
        i, j = tril_indices[0], tril_indices[1]
        cholesky_L = jnp.zeros((self.num_nodes,self.num_nodes), dtype=float)
        cholesky_L = cholesky_L.at[i, j].set(q_z_cholesky)

        q_z = reparameterized_multivariate_normal(key, q_z_mu, cholesky_L, self.num_samples)
        q_z_covariance = jnp.matmul(cholesky_L, jnp.transpose(cholesky_L))
        return q_z_mu, q_z_covariance, q_z

    def get_posterior_z(self, key, gs):
        return vmap(self.get_posterior_single_z, (None, 0), (0, 0, 0))(key, gs)


    def __call__(self, z_rng, particles_z, sf_baseline, z_data, step=0):
        # ? 1. Sample n_particles graphs from particles_z
        s = time()
        if self.grad_estimator_z == 'score':
            sampled_soft_g = self.particle_to_g_lim(particles_z)
        
        elif self.grad_estimator_z == 'reparam':
            eps = random.logistic(z_rng, shape=(self.n_particles, self.num_nodes, self.num_nodes))  
            sampled_soft_g = self.particle_to_soft_graph(particles_z, eps, step)

        print(f'Part 1 takes: {time() - s}s' )

        # ? 2. Get graph conditioned predictions on z: q(z|G)
        s = time()
        # (num_samples, n_particles, d) (num_samples, n_particles, d, d)
        q_z_mus, q_z_covariance, q_zs = self.get_posterior_z(z_rng, sampled_soft_g)
        print(f'Part 2 takes: {time() - s}s')

        # ? 3. From every distribution q(z_i|G), decode to get reconstructed samples X in higher dimensions. i = 1...num_nodes
        s = time()
        if self.known_ED is False:  decoder = lambda q_z: self.decoder(q_z)
        else:                       decoder = lambda q_z: jnp.matmul(q_z, self.proj_matrix)
        # decoder = lambda q_z: self.decoder(q_z)
        # recon = decoder(z_data)
        recons = vmap(decoder, (0), 0)(q_zs)
        print(f'Part 3 takes: {time() - s}s')

        # ? 4. Calculate phi_z = Mean ( kxx * grad_log P(particles_z | data) + grad kxx ) for updating particles_z
        s = time()
        # ? d/dz log p(data | z) grad log likelihood

        z_rng, *batch_subk = random.split(z_rng, self.n_particles + 1) 
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(particles_z, None, sf_baseline, 
                                            step, jnp.array(batch_subk), z_data)
        # print("Grad. LL", dz_log_likelihood[-1], dz_log_likelihood.shape)
        
        # ? d/dz log p(z) (acyclicity) grad log PRIOR
        z_rng, *batch_subk = random.split(z_rng, self.n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(particles_z, jnp.array(batch_subk), step)
        # print("Grad. prior", dz_log_prior[-1], dz_log_prior.shape)

        # ? d/dz log p(z, D) = d/dz log p(z)  + log p(D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        kxx = self.f_kernel_mat(particles_z, particles_z) # ? k(z, z) for all particles

        # ? transformation phi_z(t)(particle m) applied in batch to each particle individually
        phi_z = vmap(self.z_update, (0, 1, None, None), 0)(particles_z, kxx, particles_z, dz_log_prob)
        print(f'Part 4 takes: {time() - s}s')

        return recons, q_z_mus, q_z_covariance, phi_z, sampled_soft_g, sf_baseline, z_rng, q_zs
        # return None, None, None, phi_z, sampled_soft_g, sf_baseline, z_rng, None


class Z_mu_logvar_Net(nn.Module):
    latent_dims: int
    num_cholesky_terms: int

    @nn.compact
    def __call__(self, g):
        z = nn.Dense(20, name='encoder_0')(g)
        z = nn.relu(z)
        z = nn.Dense(3 * self.latent_dims, name='encoder_1')(z)
        z = nn.relu(z)
        z = nn.Dense(3 * self.latent_dims, name='encoder_2')(z)
        z = nn.relu(z)
        
        z_mu = nn.Dense(3 * self.latent_dims, name='mu_encoder_0')(z)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(3 * self.latent_dims, name='mu_encoder_2')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(3 * self.latent_dims, name='mu_encoder_3')(z_mu)
        z_mu = nn.relu(z_mu)
        z_mu = nn.Dense(self.latent_dims, name='mu_encoder_1')(z_mu)

        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_0')(z)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_2')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.latent_dims*self.latent_dims, name='logcovar_encoder_3')(z_logcholesky)
        z_logcholesky = nn.relu(z_logcholesky)
        z_logcholesky = nn.Dense(self.num_cholesky_terms, name='logcovar_encoder_1')(z_logcholesky)

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











    




