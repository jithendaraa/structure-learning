import sys
sys.path.append('..')
import os, imageio, utils, datagen, pathlib
from os.path import join
from torch.utils.tensorboard import SummaryWriter

import ruamel.yaml as yaml
from jax import vmap, random, jit, grad

import numpy as np
import jax.numpy as jnp
from dibs.eval.target import make_linear_gaussian_equivalent_model
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture
from dibs.eval.metrics import expected_shd, threshold_metrics

# Set seeds
np.random.seed(0)
key = random.PRNGKey(123)

configs = yaml.safe_load((pathlib.Path('..') / 'configs.yaml').read_text())
opt, exp_config = utils.load_yaml_dibs(configs)
logdir = utils.set_tb_logdir(opt)

train_dataloader = datagen.generate_data(opt, logdir)

target = make_linear_gaussian_equivalent_model(key = key, n_vars = opt.num_nodes, 
            graph_prior_str = opt.datatype, obs_noise = opt.noise_sigma, 
            mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
            n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))
gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

model = target.inference_model
gt_graph = train_dataloader.graph
adjacency_matrix = train_dataloader.adjacency_matrix.astype(int)
no_interv_targets = jnp.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational

print()
print("Adjacency matrix:")
print(adjacency_matrix)
print()

data = jnp.array(train_dataloader.samples)
x = data
print("Data:", x)

def log_prior(single_w_prob):
    """log p(G) using edge probabilities as G"""    
    return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

def log_likelihood(single_w, data, interv_targets):
    log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=data, interv_targets=interv_targets)
    return log_lik


# ? initialize kernel and algorithm
kernel = FrobeniusSquaredExponentialKernel(h=opt.h_latent)
dibs = MarginalDiBS(kernel=kernel, target_log_prior=log_prior, 
                    target_log_marginal_prob=log_likelihood, 
                    alpha_linear=opt.alpha_linear, 
                    grad_estimator_z=opt.grad_estimator)
    
# ? initialize particles
key, subk = random.split(key)
init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=opt.n_particles, n_vars=opt.num_nodes)

key, subk = random.split(key)

particles_g, _, _ = dibs.sample_particles(key=subk, 
                        n_steps=opt.num_updates, 
                        init_particles_z=init_particles_z,
                        opt_state_z = None, 
                        sf_baseline=None, 
                        interv_targets=no_interv_targets,
                        data=x, 
                        start=0)

eltwise_log_prob = vmap(lambda g: log_likelihood(g, x, no_interv_targets), (0), 0)
dibs_empirical = particle_marginal_empirical(particles_g)
dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
eshd_e = expected_shd(dist=dibs_empirical, g=adjacency_matrix)
eshd_m = expected_shd(dist=dibs_mixture, g=adjacency_matrix)

sampled_graph, mec_or_gt_count = utils.log_dags(particles_g, gt_graph, eshd_e, eshd_m, dag_file)
writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, 0, dataformats='HWC')

auroc_empirical = threshold_metrics(dist = dibs_empirical, g=adjacency_matrix)['roc_auc']
auroc_mixture = threshold_metrics(dist = dibs_mixture, g=adjacency_matrix)['roc_auc']

print()
print("ESHD (empirical):", eshd_e)
print("ESHD (marginal mixture):", eshd_m)
print("MEC-GT Recovery %", mec_or_gt_count * 100.0/ opt.n_particles)
print(f"AUROC (Empirical and Marginal): {auroc_empirical} {auroc_mixture}")
