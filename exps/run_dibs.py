import sys
sys.path.append('..')
import os, imageio, utils, datagen, pathlib, graphical_models
from os.path import join
from torch.utils.tensorboard import SummaryWriter

import ruamel.yaml as yaml
from jax import vmap, random, jit, grad
import networkx as nx

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
n_intervention_sets = 10

configs = yaml.safe_load((pathlib.Path('..') / 'configs.yaml').read_text())
opt, exp_config = utils.load_yaml_dibs(configs)

logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))

target = make_linear_gaussian_equivalent_model(key = key, n_vars = opt.num_nodes, 
            graph_prior_str = opt.datatype, obs_noise = opt.noise_sigma, 
            mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
            n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

train_dataloader = datagen.generate_data(opt, logdir)
gt_graph = train_dataloader.graph
model = target.inference_model
adjacency_matrix = train_dataloader.adjacency_matrix.astype(int)
gt_graph_cpdag = graphical_models.DAG.from_nx(gt_graph).cpdag()

def evaluate(data, interv_targets, particles_g, steps, tb_plots=False):
    eltwise_log_prob = vmap(lambda g: log_likelihood(g, data, interv_targets), (0), 0)
    dibs_empirical = particle_marginal_empirical(particles_g)
    dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
    eshd_e = np.array(expected_shd(dist=dibs_empirical, g=adjacency_matrix))
    eshd_m = np.array(expected_shd(dist=dibs_mixture, g=adjacency_matrix))

    sampled_graph, mec_or_gt_count = utils.log_dags(particles_g, gt_graph, eshd_e, eshd_m, dag_file)
    writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, steps, dataformats='HWC')

    auroc_empirical = threshold_metrics(dist = dibs_empirical, g=adjacency_matrix)['roc_auc']
    auroc_mixture = threshold_metrics(dist = dibs_mixture, g=adjacency_matrix)['roc_auc']

    if tb_plots:
        writer.add_scalar('Evaluations/AUROC (empirical)', auroc_empirical, steps)
        writer.add_scalar('Evaluations/AUROC (marginal)', auroc_mixture, steps)
        writer.add_scalar('Evaluations/Exp. SHD (Empirical)', eshd_e, steps)
        writer.add_scalar('Evaluations/Exp. SHD (Marginal)', eshd_m, steps)
        writer.add_scalar('Evaluations/MEC or GT recovery %', mec_or_gt_count * 100.0/ opt.n_particles, steps)

    cpdag_shds = []
    for adj_mat in particles_g:
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
        try:
            G_cpdag = graphical_models.DAG.from_nx(G).cpdag()
            shd = gt_graph_cpdag.shd(G_cpdag)
            cpdag_shds.append(shd)
        except:
            pass

    if len(cpdag_shds) > 0: writer.add_scalar('Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)

    print()
    print(f"Metrics after {int(steps)} steps")
    print()
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    print("MEC-GT Recovery %", mec_or_gt_count * 100.0/ opt.n_particles)
    print(f"AUROC (Empirical and Marginal): {auroc_empirical} {auroc_mixture}")


gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
no_interv_targets = np.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational

data_start, data_end = opt.obs_data, opt.num_samples
num_interv_data = data_end - data_start
interv_data_per_set = int(num_interv_data / n_intervention_sets)
data_ = []

interv_data = np.array(target.x_interv)

if num_interv_data > 0:
    for i in range(n_intervention_sets):
        interv_target = np.array([False] * opt.num_nodes)
        intervened_node = [*interv_data[i][0]][0]
        interv_target[intervened_node] = True
        start_idx, end_idx = int(data_start + (i * interv_data_per_set)), int(data_start + ((i+1) * interv_data_per_set)), 
        no_interv_targets[start_idx : end_idx, :] = interv_target[None, :]
        data_.append(interv_data[i][1][:interv_data_per_set])

data_, no_interv_targets = jnp.array(data_).reshape(-1, opt.num_nodes), jnp.array(no_interv_targets)


print()
print("Adjacency matrix:")
print(adjacency_matrix)
print()

data = jnp.array(train_dataloader.samples)[:opt.obs_data]
data = jnp.concatenate((data, data_), axis=0)
x = data

def log_prior(single_w_prob):
    """log p(G) using edge probabilities as G"""    
    return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

def log_likelihood(single_w, data, interv_targets):
    log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=data, interv_targets=interv_targets)
    return log_lik


# ? initialize kernel and algorithm
kernel = FrobeniusSquaredExponentialKernel(h=opt.h_latent)
dibs = MarginalDiBS(kernel=kernel, target_log_prior=log_prior, target_log_marginal_prob=log_likelihood, 
                    alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)
    
# ? initialize particles
key, subk = random.split(key)
init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=opt.n_particles, n_vars=opt.num_nodes)

key, subk = random.split(key)
n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'Intervention sets {n_intervention_sets} with {interv_data_per_set} data points per intervention set')

particles_g, particles_z, opt_state_z, sf_baseline = dibs.sample_particles(key=subk, 
                                                        n_steps=n_steps, 
                                                        init_particles_z=init_particles_z,
                                                        opt_state_z = None, 
                                                        sf_baseline=None, 
                                                        interv_targets=no_interv_targets[:opt.obs_data],
                                                        data=x[:opt.obs_data], 
                                                        start=0)

evaluate(x[:opt.obs_data], no_interv_targets[:opt.obs_data], particles_g, n_steps)

start_ = n_steps
if num_interv_data > 0:
    for i in range(n_intervention_sets):
        interv_targets = no_interv_targets[:opt.obs_data + ((i+1)*interv_data_per_set)]
        data = x[:opt.obs_data + ((i+1)*interv_data_per_set)]

        particles_g, particles_z, opt_state_z, sf_baseline = dibs.sample_particles(key=subk, 
                                                            n_steps=n_steps, 
                                                            init_particles_z=particles_z,
                                                            opt_state_z = opt_state_z, 
                                                            sf_baseline= sf_baseline, 
                                                            interv_targets=interv_targets,
                                                            data=data, 
                                                            start=start_)
        start_ += n_steps
        evaluate(data, interv_targets, particles_g, int(start_), True)