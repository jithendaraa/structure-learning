import sys
sys.path.append('..')
sys.path.append('exps')

import jax.numpy as jnp
from jax import random, vmap, grad, device_put, jit, lax
from flax import linen as nn
from jax.ops import index, index_mul
from jax.nn import sigmoid, log_sigmoid
from jax.scipy.special import logsumexp, gammaln

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
    latent_prior_std: float = None
    grad_estimator: str = 'score'
    known_ED: bool = False
    linear_decoder: bool = False
    clamp: bool = True
# 
    def setup(self):
        self.n_vars = self.num_nodes
        
        self.dibs = JointDiBS(n_vars=self.num_nodes, 
                                    inference_model=self.model, 
                                    alpha_linear=self.alpha_linear, 
                                    grad_estimator_z=self.grad_estimator)

        if self.known_ED is False:  self.decoder = Decoder(self.proj_dims, self.linear_decoder)
        lower_triangular_elems = int(self.num_nodes * (self.num_nodes + 1) / 2)
        self.z_net = Z_mu_logvar_Net(self.num_nodes, lower_triangular_elems)
        print("Loaded Decoder Joint DIBS")
    
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
        n_particles, _, n_dim, _ = z.shape
        # d/dtheta log p(theta, D | z)
        key, subk = random.split(key)
        dtheta_log_prob = self.dibs.eltwise_grad_theta_likelihood(z, theta, t, subk, data, interv_targets)
        
        # d/dz log p(theta, D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.dibs.eltwise_grad_z_likelihood(z, theta, sf_baseline, t, jnp.array(batch_subk), interv_targets, data)
        
        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.dibs.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t, latent_prior= 1.0 / jnp.sqrt(n_dim))

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
        return vmap(self.eltwise_get_grad_dibs_params, (None, 0, 0, None, 0, None, 0), (0, 0, 0))(key, zs, thetas, t, datas, interv_targets, sf_baselines[:, jnp.newaxis])

    def __call__(self, key, particles_z, particles_theta, sf_baseline, data, interv_targets, step):
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

        samples = len(interv_targets)
        # ? 1. Sample n_particles graphs from particles_z
        if sf_baseline is None:     sf_baseline = jnp.zeros(self.n_particles)
        if particles_z is None and particles_theta is None:
            particles_z, particles_theta = self.dibs._sample_initial_random_particles(key=key, n_particles=self.n_particles)

        if self.grad_estimator == 'score': gs = self.dibs.particle_to_g_lim(particles_z)

        # ? 2. Get graph conditioned predictions on z: q(z|G, theta)
        get_posterior_z = vmap(self.eltwise_get_posterior_z, (None, 0, 0, None), (0, 0, 0, 0))
        q_z_mus, q_z_covars, _, q_zs = get_posterior_z(key, gs, particles_theta, samples)

        # ? 3. From every distribution q(z_i|G_i), decode to get reconstructed samples X in higher dimensions. i = 1...num_nodes
        if self.known_ED is False:  decoder = lambda q_z: self.decoder(q_z)
        X_recons = vmap(decoder, (0), (0))(q_zs)

        # ? 4. Get dibs gradients dict with repsect to params: z and theta
        if data is None:  
            if self.clamp is True:
                interv_filter_fn = lambda q_z: jnp.where(interv_targets, 0.0, q_z)
                q_zs = vmap(interv_filter_fn, (0), (0))(q_zs)
            data = lax.stop_gradient(q_zs)
        
        zs = particles_z[:, jnp.newaxis, ...]
        thetas = self.unsqueeze_theta(particles_theta)
        phi_z, phi_theta, sf_baseline = self.get_grad_dibs_params(key, zs, thetas, step, data, interv_targets, sf_baseline)
        dibs_grads = {'phi_z': jnp.squeeze(phi_z, axis=1), 'phi_theta': self.squeeze_theta(phi_theta)}
        
        return X_recons, (particles_z, self.squeeze_theta(particles_theta)), q_z_mus, q_z_covars, dibs_grads, gs, jnp.squeeze(sf_baseline, axis=1), q_zs

