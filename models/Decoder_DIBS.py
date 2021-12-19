from re import A
import sys

sys.path.append('dibs/')
from time import time
import numpy as np
import functools

import jax
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
        return z
        

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def v_reparameterize(rng, mean, logvar):
    return vmap(reparameterize, (None, 0, 0), 0)(rng, mean, logvar)


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
    grad_estimator_z: str = 'reparam'
    score_function_baseline: float = 0.0
    n_grad_mc_samples: int = 128
    tau: float = 1.0
    beta_linear: float = 1.0
    n_acyclicity_mc_samples: int = 32
    scale: float = 1.0
    edges_per_node: float = 1.0

    def setup(self):
        # For BGeJAX
        self.mean_obs = jnp.zeros(self.num_nodes)

        # For ErdosReniDAGDistribution
        self.n_vars = self.num_nodes
        self.n_edges = self.edges_per_node * self.num_nodes
        self.p = self.n_edges / ((self.n_vars * (self.n_vars - 1)) / 2)

        if self.datatype not in ['er']:
            NotImplementedError(f"Not implemented {self.datatype}")

        # Net to feed in G and predict a Z 
        self.z_net = Z_mu_logvar_Net(self.num_nodes)
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
        return eval_kernel(self.scale, x_latent, y_latent, self.h_latent, self.h_latent)
        
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

    def slogdet_jax(self, m, parents, n_parents):
        """
        jax.jit-compatible log determinant of a submatrix

        Done by masking everything but the submatrix and
        adding a diagonal of ones everywhere else for the 
        valid determinant

        Args:
            m: [d, d] matrix
            parents: [d, ] boolean indicator of parents
            n_parents: number of parents total

        Returns:
            natural log of determinant of `m`
        """

        n_vars = parents.shape[0]
        submat = leftsel(m, parents, maskval=np.nan)
        submat = leftsel(submat.T, parents, maskval=np.nan).T
        submat = jnp.where(jnp.isnan(submat), jnp.eye(n_vars), submat)
        return jnp.linalg.slogdet(submat)[1]

    def log_marginal_likelihood_given_g_single(self, j, n_parents, R, w, data, log_gamma_terms):
        """
        Computes node specific term of BGe metric
        jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            R: internal matrix for BGe score [d, d]
            w: adjacency matrix [d, d] 
            data: observations [N, d] 
            log_gamma_terms: internal values for BGe score [d, ]

        Returns:
            BGe score for node j
        """

        N, d = data.shape

        isj = jnp.arange(d) == j
        parents = w[:, j] == 1
        parents_and_j = parents | isj

        # if JAX_DEBUG_NANS raises NaN error here,
        # ignore (happens due to lax.cond evaluating the second clause when n_parents == 0)
        log_term_r = lax.cond(
            n_parents == 0,
            # leaf node case
            lambda _: (
                # log det(R)^(...)
                - 0.5 * (N + self.alpha_lambd - d + 1) * jnp.log(jnp.abs(R[j, j]))
            ),
            # child case
            lambda _: (
                # log det(R_II)^(..) / det(R_JJ)^(..)
                0.5 * (N + self.alpha_lambd - d + n_parents) *
                    self.slogdet_jax(R, parents, n_parents)
                - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
                    self.slogdet_jax(R, parents_and_j, n_parents + 1)
            ),
            operand=None,
        )

        return log_gamma_terms[n_parents] + log_term_r

    def eltwise_log_marginal_likelihood_given_g_single(self, *args):
        """
        Same inputs as `log_marginal_likelihood_given_g_single`,
        but batched over `j` and `n_parents` dimensions
        """
        return vmap(self.log_marginal_likelihood_given_g_single, (0, 0, None, None, None, None), 0)(*args)

    def log_marginal_likelihood_given_g(self, w, data, interv_targets=None):
        """Computes BGe marignal likelihood  log p(x | G) in closed form 
        Args:	
            data: observations [N, d]	
            w: adjacency matrix [d, d]	
            interv_targets: boolean mask of shape [d,] of whether or not a node was intervened on
                    intervened nodes are ignored in likelihood computation
        Returns:
            [1, ] BGe Score
        """

        N, d = data.shape        
        if interv_targets is None:
            interv_targets = jnp.zeros(d).astype(bool)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / \
            (self.alpha_mu + 1)
        T = small_t * jnp.eye(d)

        x_bar = data.mean(axis=0, keepdims=True)
        x_center = data - x_bar
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
        # the supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        # store log gamma terms for all possible values of l
        all_l = jnp.arange(d)
        log_gamma_terms = (
            0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
            + gammaln(0.5 * (N + self.alpha_lambd - d + all_l + 1))
            - gammaln(0.5 * (self.alpha_lambd - d + all_l + 1))
            - 0.5 * N * jnp.log(jnp.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 2 * all_l + 1) * \
            jnp.log(small_t)
        )

        # compute number of parents for each node
        n_parents_all = w.sum(axis=0).astype(jnp.int32)

        # sum scores for all nodes
        res = self.eltwise_log_marginal_likelihood_given_g_single(jnp.arange(d), n_parents_all, R, w, data, log_gamma_terms)
        return jnp.sum(jnp.where(interv_targets, 0.0, res))

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
    def edge_probs(self, z, alpha_t):
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
        probs =  sigmoid(alpha_t * scores)

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
        return self.log_marginal_likelihood_given_g(soft_g_sample, data)

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
        return vmap(self.log_marginal_likelihood_given_g_single, (0, None), 0)(gs, data)

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

    def get_posterior_single_z(self, key, single_g):
        flattened_g = jnp.array(single_g.reshape(-1))
        q_z_mu, q_z_logvar = self.z_net(flattened_g)
        q_z = v_reparameterize(key, jnp.asarray([q_z_mu]*self.num_samples), jnp.asarray([q_z_logvar]*self.num_samples))
        return q_z_mu, q_z_logvar, q_z

    def get_posterior_z(self, key, gs):
        return vmap(self.get_posterior_single_z, (None, 0), (0, 0, 0))(key, gs)

    def eltwise_grad_z_likelihood(self, zs, thetas, baselines, t, subkeys, data):
        if self.grad_estimator_z == 'reparam':    
            grad_z_likelihood = self.grad_z_likelihood_gumbel
        else:   
            raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')
        return vmap(grad_z_likelihood, (0, 0, 0, None, 0, 0), (0, 0))(zs, thetas, baselines, t, subkeys, data)

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
        eltwise_grad_constraint = vmap(self.grad_constraint_gumbel, (0, 0, None), 0)(zs, subkeys, alpha_t)

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



    def __call__(self, z_rng, particles_z, sf_baseline, step=0):
        # ? 1. Sample n_particles graphs from particles_z
        s = time()
        z_rng, key = random.split(z_rng) 
        eps = random.logistic(key, shape=(self.n_particles, self.num_nodes, self.num_nodes))  
        sampled_soft_g = self.particle_to_soft_graph(particles_z, eps, step)
        print(f'Part 1 takes: {time() - s}s')

        # ? 2. Get graph conditioned predictions on z: q(z|G)
        s = time()
        z_rng, key = random.split(z_rng) 
        q_z_mus, q_z_logvars, q_zs = self.get_posterior_z(key, sampled_soft_g)
        print(f'Part 2 takes: {time() - s}s')

        # ? 3. From every distribution q(z_i|G), decode to get reconstructed samples X in higher dimensions. i = 1...num_nodes
        s = time()
        decoder = lambda q_z: self.decoder(q_z)
        recons = vmap(decoder, (0), 0)(q_zs)
        print(f'Part 3 takes: {time() - s}s')

        # ? 4. Calculate phi_z = Mean ( kxx * grad_log P(particles_z | data) + grad kxx ) for updating particles_z
        s = time()
        # ? d/dz log p(data | z) grad log likelihood
        z_rng, *batch_subk = random.split(z_rng, self.n_particles + 1) 
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(particles_z, None, sf_baseline, 
                                            step, jnp.array(batch_subk), jnp.array(q_zs))
        # ? d/dz log p(z) (acyclicity) grad log PRIOR
        z_rng, *batch_subk = random.split(z_rng, self.n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(particles_z, jnp.array(batch_subk), step)

        # ? d/dz log p(z, D) = d/dz log p(z)  + log p(D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        kxx = self.f_kernel_mat(particles_z, particles_z) # ? k(z, z) for all particles

        # ? transformation phi_z(t)(particle m) applied in batch to each particle individually
        phi_z = jit(vmap(self.z_update, (0, 1, None, None), 0))(particles_z, kxx, particles_z, dz_log_prob)
        print(f'Part 4 takes: {time() - s}s')

        return recons, q_z_mus, q_z_logvars, phi_z, sampled_soft_g, sf_baseline, z_rng, q_zs

def eval_kernel(scale, x, y, h, global_h):
    h_ = lax.cond(
        h == -1.0,
        lambda _: global_h,
        lambda _: h,
        operand=None)

    squared_norm = jnp.sum((x - y) ** 2.0)      # compute norm
    return scale * jnp.exp(- squared_norm / h_) # compute kernel










    




