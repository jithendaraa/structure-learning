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

# experiments
from dibs_bge_old import run_dibs_bge_old
from dibs_bge_new import run_dibs_bge_new

# Set seeds
np.random.seed(0)
key = random.PRNGKey(123)
n_intervention_sets = 10

configs = yaml.safe_load((pathlib.Path('..') / 'configs.yaml').read_text())
opt, exp_config = utils.load_yaml_dibs(configs)

logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))

train_dataloader = datagen.generate_data(opt, logdir)
gt_graph = train_dataloader.graph
adjacency_matrix = train_dataloader.adjacency_matrix.astype(int)
gt_graph_cpdag = graphical_models.DAG.from_nx(gt_graph).cpdag()
num_interv_data = opt.num_samples - opt.obs_data
interv_data_per_set = int(num_interv_data / n_intervention_sets)  
gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates

print()
print("Adjacency matrix:")
print(adjacency_matrix)
print()
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'Intervention sets {n_intervention_sets} with {interv_data_per_set} data points per intervention set')

  

def run_dibs_nonlinear_old(key, opt):
    kernel = JointAdditiveFrobeniusSEKernel(h_latent=opt.h_latent, h_theta=opt.h_latent)
    target = make_nonlinear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, obs_noise = opt.noise_sigma, 
                sig_param = opt.theta_sigma, n_observations = opt.num_samples, 
                n_ho_observations = opt.num_samples)

    model = target.inference_model
    data_, no_interv_targets = datagen.generate_interv_data(opt, n_intervention_sets, target)
    data = jnp.array(train_dataloader.samples)[:opt.obs_data]
    data = jnp.concatenate((data, data_), axis=0)
    x = data

    def log_prior(single_w_prob):
        """log p(G) using edge probabilities as G"""    
        return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(single_w, theta, data, interv_targets):
        log_lik = model.log_likelihood(data=data, theta=theta, w=single_w, interv_targets=interv_targets)
        return log_lik

    dibs = JointDiBS(kernel=kernel, target_log_prior=log_prior, 
                        target_log_joint_prob=log_likelihood, 
                        alpha_linear=opt.alpha_linear, 
                        grad_estimator_z=opt.grad_estimator)

    key, subk = random.split(key)
    init_particles_z, init_particles_theta = dibs.sample_initial_random_particles(key=subk, n_particles=opt.n_particles, n_vars=opt.num_nodes, model=target.inference_model)

    particles_g, particles_z, theta, opt_state_z, sf_baseline = dibs.sample_particles(key=subk, n_steps=n_steps, 
                                                            init_particles_z=init_particles_z,
                                                            init_particles_theta=init_particles_theta,
                                                            opt_state_z = None, sf_baseline=None, 
                                                            interv_targets=no_interv_targets[:opt.obs_data],
                                                            data=x[:opt.obs_data], 
                                                            start=0)
    
    evaluate(x[:opt.obs_data], log_likelihood, no_interv_targets[:opt.obs_data], particles_g, n_steps)

if opt.likelihood == 'bge':
    run_dibs_bge_old(train_dataloader, key, opt, n_intervention_sets, gt_graph, dag_file, writer)

    # run_dibs_bge_new(train_dataloader, key, opt, n_intervention_sets, gt_graph, dag_file, writer)

elif opt.likelihood == 'nonlinear':
    run_dibs_nonlinear_old(key, opt)


# start_ = n_steps
# if num_interv_data > 0:
#     for i in range(n_intervention_sets):
#         interv_targets = no_interv_targets[:opt.obs_data + ((i+1)*interv_data_per_set)]
#         data = x[:opt.obs_data + ((i+1)*interv_data_per_set)]

#         particles_g, particles_z, opt_state_z, sf_baseline = dibs.sample_particles(key=subk, 
#                                                             n_steps=n_steps, 
#                                                             init_particles_z=particles_z,
#                                                             opt_state_z = opt_state_z, 
#                                                             sf_baseline= sf_baseline, 
#                                                             interv_targets=interv_targets,
#                                                             data=data, 
#                                                             start=start_)
#         start_ += n_steps
#         evaluate(data, interv_targets, particles_g, int(start_), True)