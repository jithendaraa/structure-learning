import sys, os, imageio
sys.path.append('../')
from os.path import join
import datagen, utils

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from flax.training import train_state
import jax.numpy as jnp
from jax import vmap, random, jit, grad, device_put
from jax.experimental import optimizers
import jax.scipy.stats as dist
import optax, loss_fns

from models.Decoder_JointDiBS import Decoder_JointDiBS
from dibs_new.dibs.target import make_nonlinear_gaussian_model


def run_decoder_joint_dibs(key, opt, logdir, n_intervention_sets, dag_file, writer):
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)

    target, model = make_nonlinear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    gt_graph = nx.from_numpy_matrix(target.g, create_using=nx.DiGraph)
    nx.draw(gt_graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
    plt.savefig(join(logdir,'gt_graph.png'))
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    print()
    
    obs_data, interv_data, z_gt, no_interv_targets, x, p_z_mu, p_z_covar = datagen.get_data(opt, n_intervention_sets, target)

    dibs = Decoder_JointDiBS(opt.num_nodes, opt.num_samples, 
                                opt.proj_dims, 
                                opt.n_particles,
                                model, opt.alpha_linear, 
                                latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes),
                                grad_estimator=opt.grad_estimator,
                                known_ED=opt.known_ED, 
                                linear_decoder=opt.linear_decoder)

    key, rng = random.split(key)
    dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
    opt_init, opt_update, get_params = dibs_optimizer
    
    state = train_state.TrainState.create(
        apply_fn=dibs.apply,
        params=dibs.init(rng, rng, None, None, None, obs_data, no_interv_targets[:opt.obs_data], 0)['params'],
        tx=optax.adam(opt.lr)
    )

    decoder_train_steps = opt.steps - opt.num_updates
    dibs_update = 0
    z_final, theta_final, sf_baseline = None, None, None

    def train_causal_vars(state, z_rng, z, theta, 
                    sf_baseline, data, interv_targets, step=0):
        
        recons, q_z_mus, q_z_covars, dibs_grads, gs, sf_baseline, _, pred_zs = dibs.apply({'params': state.params}, 
                                                                                                z_rng, z, theta, sf_baseline, 
                                                                                                data, interv_targets, step)
        grads = grad(loss_fns.loss_fn)(state.params, z_rng, z, theta, sf_baseline, data, interv_targets, 
                    step, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs)
        res = state.apply_gradients(grads=grads)
        loss, mse_loss, kl_z_loss, z_dist = loss_fns.calc_loss(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, pred_zs, opt, z_gt)
        return res, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, sf_baseline, z_dist, pred_zs

    def train_causal_graph():
        pass

    for step in range(1):
        multiple = step // decoder_train_steps
        even_multiple = ((multiple % 2) == 0)
        interv_targets = no_interv_targets[:opt.obs_data]

        if step < decoder_train_steps:
            state, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, _, z_dist, q_z = train_causal_vars(state, rng, z_final, theta_final, 
                        sf_baseline, obs_data, interv_targets, step)
        else:
            q_z_mus, q_z_covars = np.array(q_z_mus), np.array(q_z_covars)
            data = vmap(datagen.gen_data_from_dist, (0, 0, None), 0)(rng, q_z_mus, q_z_covars, opt.num_samples)  

            # [TODO] Train causal graph with dibs  

        if (step+1) % 100 == 0:
            # [TODO] Calculate metrics and log to wandb/tb
            pass

            

    