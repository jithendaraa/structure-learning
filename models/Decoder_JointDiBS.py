import sys, pdb, functools
sys.path.append('..')
sys.path.append('exps')

import numpy as onp
import jax.numpy as jnp
from jax import random, vmap, grad, device_put, jit, lax
from flax import linen as nn
from jax.ops import index, index_mul, index_update
from jax.nn import sigmoid, log_sigmoid
from jax.scipy.special import logsumexp, gammaln
import networkx as nx

import datagen, utils, pdb
from dibs_new.dibs.target import make_nonlinear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs_new.dibs.models import DenseNonlinearGaussian
from modules.Decoder_DiBS_nets import Decoder, Z_mu_logvar_Net

class Decoder_JointDiBS(nn.Module):
    num_nodes: int
    num_samples: int
    proj_dims: int
    n_particles: int
    model: DenseNonlinearGaussian
    alpha_linear: float
    dibs_type: str = 'nonlinear'
    latent_prior_std: float = None
    grad_estimator: str = 'score'
    known_ED: bool = False
    linear_decoder: bool = False
    clamp: bool = True
    topsort: bool = False
    obs_noise: float = 0.1
# 
    def setup(self):
        self.n_vars = self.num_nodes
        
        self.dibs = JointDiBS(n_vars=self.num_nodes, inference_model=self.model, 
                            alpha_linear=self.alpha_linear, grad_estimator_z=self.grad_estimator)

        lower_triangular_elems = int(self.num_nodes * (self.num_nodes + 1) / 2)
        
        if self.topsort is False:
            self.z_net = Z_mu_logvar_Net(self.num_nodes, lower_triangular_elems)
        
        if self.known_ED is False:  
            self.decoder = Decoder(self.proj_dims, self.linear_decoder)
        # print("Loaded Decoder Joint DIBS")
    
    def reparameterized_multivariate_normal(self, rng, mean, cholesky_L, samples):
        """
        [TODO]
            # Cholesky decomposition: Σ = LL.T where L is lower triangular
            # reparametrised sample `res = mean + eps * L`
        """
        d = cholesky_L.shape[0]
        unit_cov_matrix = jnp.identity(d)
        unit_cov_matrix = jnp.expand_dims(unit_cov_matrix, 0).repeat(samples, axis=0)
        
        standard_mean = jnp.zeros((samples, d))
        eps = random.multivariate_normal(rng, standard_mean, unit_cov_matrix)
        return mean + jnp.matmul(eps, cholesky_L)

    def unsqueeze_theta(self, theta):
        """
        [TODO]
        theta is usually of the form: 
                [   tuple(jnp.array(n_particles, n_vars, n_vars, x), jnp.array(n_particles, n_vars, x)),
                    (),
                    tuple(jnp.array(n_particles, n_vars, x, 1), jnp.array(n_particles, n_vars, 1))
                ]

        This function would add new axes such that the returned theta is of the form:
                    [   tuple(jnp.array(n_particles, "1", n_vars, n_vars, x), jnp.array(n_particles, "1", n_vars, x)),
                        (),
                        tuple(jnp.array(n_particles, "1", n_vars, x, 1), jnp.array(n_particles, "1", n_vars, 1))
                    ]
        """
        res = theta
        for idx, tuples in enumerate(theta):
            if len(tuples) > 0:
                unsqueezed_tuple = tuple([elem[:, jnp.newaxis, ...] for elem in tuples])
                res[idx] = unsqueezed_tuple
        return res

    def squeeze_theta(self, theta):
        """
        [TODO]
        theta is usually of the form: 
                    [   tuple(jnp.array(n_particles, "1", n_vars, n_vars, x), jnp.array(n_particles, "1", n_vars, x)),
                    (),
                    tuple(jnp.array(n_particles, "1", n_vars, x, 1), jnp.array(n_particles, "1", n_vars, 1))
                ]

        This function would add new axes such that the returned theta is of the form:
                [   tuple(jnp.array(n_particles, n_vars, n_vars, x), jnp.array(n_particles, n_vars, x)),
                    (),
                    tuple(jnp.array(n_particles, n_vars, x, 1), jnp.array(n_particles, n_vars, 1))
                ]
        """
        res = theta
        for idx, tuples in enumerate(theta):
            if len(tuples) > 0:
                squeezed_tuple = tuple([elem[:, 0,...] for elem in tuples])
                res[idx] = squeezed_tuple
        return res

    def eltwise_get_posterior_z(self, key, g, theta, samples):
        """
        [TODO]
        """

        if self.dibs_type == 'linear':
            weighted_adj_matrix = jnp.multiply(g, theta)
            q_z_mu, q_z_logcholesky = self.z_net(weighted_adj_matrix.flatten())

        elif self.dibs_type == 'nonlinear':
            flattened_g = jnp.array(g.reshape(-1))
            flattened_theta = jnp.concatenate((theta[0][0].flatten(), theta[0][1].flatten(), theta[2][0].flatten(), theta[2][1].flatten()), axis=0)
            g_thetas = jnp.concatenate((flattened_g, flattened_theta), axis=0)
            q_z_mu, q_z_logcholesky = self.z_net(g_thetas)
        
        q_z_cholesky = jnp.exp(q_z_logcholesky)
        tril_indices = jnp.tril_indices(self.num_nodes)
        i, j = tril_indices[0], tril_indices[1]
        cholesky_L = jnp.zeros((self.num_nodes,self.num_nodes), dtype=float)
        cholesky_L = cholesky_L.at[i, j].set(q_z_cholesky)

        q_z_covar = jnp.matmul(cholesky_L, jnp.transpose(cholesky_L))
        q_z = self.reparameterized_multivariate_normal(key, q_z_mu, cholesky_L, samples)
        return q_z_mu, q_z_covar, cholesky_L, q_z

    def eltwise_get_grad_dibs_params(self, key, z, theta, t, data, interv_targets, sf_baseline):
        """
        [TODO]
        """
        # d/dtheta log p(theta, D | z)
        key, subk = random.split(key)
        
        dtheta_log_prob = self.dibs.eltwise_grad_theta_likelihood(z, theta, t, subk, data, interv_targets)
        
        # d/dz log p(theta, D | z)
        key, *batch_subk = random.split(key, z.shape[0] + 1)
        dz_log_likelihood, sf_baseline = self.dibs.eltwise_grad_z_likelihood(z, theta, sf_baseline, t, jnp.array(batch_subk), interv_targets, data)
        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, z.shape[0] + 1)
        dz_log_prior = self.dibs.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t, latent_prior= 1.0 / jnp.sqrt(self.n_vars))

        # d/dz log p(z, theta, D) = d/dz log p(z)  + log p(theta, D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        
        # k((z, theta), (z, theta)) for all particles
        kxx = self.dibs._f_kernel_mat(z, theta, z, theta)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.dibs._parallel_update_z(z, theta, kxx, z, theta, dz_log_prob)
        phi_theta = self.dibs._parallel_update_theta(z, theta, kxx, z, theta, dtheta_log_prob)

        return phi_z, phi_theta, sf_baseline
    
    def get_grad_dibs_params(self, key, zs, thetas, t, datas, interv_targets, sf_baselines):
        """
        [TODO]
        """
        return vmap(self.eltwise_get_grad_dibs_params, (None, 0, 0, None, 0, None, 0), (0, 0, 0))(key, zs, thetas, t, datas, interv_targets, sf_baselines)

    def initialise_random_particles(self, key):
        """
            [TODO]
        """

        if self.dibs_type == 'nonlinear':
            particles_z, particles_theta = self.dibs._sample_initial_random_particles(key=key, n_particles=self.n_particles)
        
        elif self.dibs_type == 'linear':
            n_dim = self.n_vars
            std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

            key, subk = random.split(key)
            particles_z = random.normal(subk, shape=(self.n_particles, self.n_vars, n_dim, 2)) * std
            shape = (self.n_particles, n_dim, n_dim)
            particles_theta = self.model.mean_edge + self.model.sig_edge * random.normal(key, shape=shape)

        return particles_z, particles_theta

    def ancestral_sample(self, subk, gs, interv_targets, theta, interv_values=None):
        q_zs, idxs = [], []
        num_samples, num_nodes = interv_targets.shape
        if interv_values is None: interv_values = 0.0
        
        for i in range(self.n_particles):
            # predict q_zs only for DAGs 
            adj_mat_i, theta_i = gs[i], theta[i]
            graph = nx.from_numpy_matrix(onp.array(adj_mat_i), create_using=nx.DiGraph)
            
            # If graph is not a DAG
            while nx.is_directed_acyclic_graph(graph) is False:    
                edges = random.choice(subk, jnp.array(nx.find_cycle(graph)), axis=0)
                if len(edges.shape) == 1: edges = edges[jnp.newaxis, :]

                for edge in edges:
                    adj_mat_i = index_update(adj_mat_i, index[edge[0], edge[1]], 0)
                
                graph = nx.from_numpy_matrix(onp.array(adj_mat_i), create_using=nx.DiGraph)

            gs = index_update(gs, index[i], adj_mat_i)
            samples = jnp.zeros((num_samples, num_nodes))
            eps = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(num_samples, num_nodes) )
            toporder = nx.topological_sort(graph)

            # Traverse node topologically
            for j in toporder:
                parents = jnp.array(jnp.where(adj_mat_i[:, j] == 1)[0])
                
                if len(parents) > 0:
                    mean = samples[:, parents] @ theta_i[parents, j]
                    samples = index_update(samples, index[:, j], mean + eps[:, j])
                else:
                    samples = index_update(samples, index[:, j], eps[:, j])

                # Data indices where node j was intervened upon
                intervened_idxs = jnp.where(interv_targets[:, j] == True)[0]
                samples = index_update(samples, index[intervened_idxs, j], interv_values)
            
            q_zs.append(samples)
            idxs.append(i)

        return jnp.array(q_zs), jnp.array(idxs)

    def __call__(self, key, particles_z, particles_theta, sf_baseline, data, interv_targets, step, dibs_type):
        """
        [TODO]

        key
        particles_z: None or (n_particles, n_vars, n_vars, 2)
        particles_theta: None or [   tuple(jnp.array(n_particles, n_vars, n_vars, x), jnp.array(n_particles, n_vars, x)),
                                    (),
                                    tuple(jnp.array(n_particles, n_vars, x, 1), jnp.array(n_particles, n_vars, 1))
                                ]
        sf_baseline: None or (n_particles, )
        data: None or (n_samples, n_vars)
        intev_targets: bool jnp array of shape (n_samples, n_vars)
        step: int
        """
        q_z_mus, q_z_covars = None, None
        samples = len(interv_targets)
        idxs = jnp.arange(0, self.n_particles)
        
        if particles_z is None and particles_theta is None:
            particles_z, particles_theta = self.initialise_random_particles(key)
            
        # ? 1. Sample n_particles graphs from particles_z
        gs = self.dibs.particle_to_g_lim(particles_z)

        # ? 2. Get graph conditioned predictions on z: q(z|G, theta)
        if self.topsort is False:
            get_posterior_z = vmap(self.eltwise_get_posterior_z, (None, 0, 0, None), (0, 0, 0, 0))
            q_z_mus, q_z_covars, _, q_zs = get_posterior_z(key, gs, particles_theta, samples)
        
        # Topsort and ancestral sample
        elif self.topsort is True:
            q_zs, idxs = self.ancestral_sample(key, gs, interv_targets, particles_theta * gs)

        # ? 3. From every distribution q(z_i|G_i), decode to get reconstructed samples X in higher dimensions. i = 1...num_nodes
        if self.known_ED is False:  decoder = lambda q_z: self.decoder(q_z)
        X_recons = vmap(decoder, (0), (0))(q_zs)

        # ? 4. Get dibs gradients dict with repsect to params: z and theta
        if data is None:    data = lax.stop_gradient(q_zs)
        
        if dibs_type == 'nonlinear': 
            phi_z, phi_theta, sf_baseline = self.get_grad_dibs_params(key, particles_z[:, jnp.newaxis, ...], 
                                                                        self.unsqueeze_theta(particles_theta), 
                                                                        step, data, interv_targets, sf_baseline[:, jnp.newaxis])
            dibs_grads = {'phi_z': jnp.squeeze(phi_z, axis=1), 'phi_theta': self.squeeze_theta(phi_theta)}

        elif dibs_type == 'linear':
            phi_z, phi_theta = jnp.zeros_like(particles_z), jnp.zeros_like(particles_theta)
            
            phi_z_, phi_theta_, updated_sf_baseline = self.get_grad_dibs_params(key, particles_z[idxs, jnp.newaxis, ...], 
                                                                        particles_theta[idxs, jnp.newaxis, ...], 
                                                                        step, data, interv_targets, sf_baseline[idxs, jnp.newaxis])
            phi_z = index_update(phi_z, index[idxs], jnp.squeeze(phi_z_, axis=1))
            phi_theta = index_update(phi_theta, index[idxs], jnp.squeeze(phi_theta_, axis=1))
            sf_baseline = index_update(sf_baseline, index[idxs], jnp.squeeze(updated_sf_baseline, axis=1))
            dibs_grads = {'phi_z': phi_z, 'phi_theta': phi_theta}

        else:
            raise Exception("Decoder dibs type has to be either 'linear' or 'nonlinear'")

        return X_recons, (particles_z, particles_theta), q_z_mus, q_z_covars, dibs_grads, gs, sf_baseline, q_zs

