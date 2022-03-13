import sys, os, imageio, graphical_models, pdb
sys.path.append('../')
sys.path.append('../../')
from os.path import join
import datagen, utils
from functools import partial

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from flax.training import train_state
import jax.numpy as jnp
from jax import vmap, random, jit, grad, device_put
from jax.experimental import optimizers
import jax.scipy.stats as dist
import optax, loss_fns, wandb

from models.Decoder_JointDiBS import Decoder_JointDiBS
from dibs_new.dibs.target import make_nonlinear_gaussian_model, make_linear_gaussian_model
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_marginal_likelihood
from dibs_new.dibs.inference import JointDiBS
from eval_ import evaluate

def get_target(key, opt):
    if opt.datagen == 'linear':
        target, model = make_linear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)
    
    elif opt.datagen == 'nonlinear':
        target, model = make_nonlinear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    return target, model

def run_decoder_joint_dibs_across_interv_data(key, opt, logdir, dag_file, writer, exp_config_dict, n_intervention_sets=10):
    rng = key
    target, model = get_target(key, opt)
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
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)
    decoder_train_steps = opt.steps - opt.num_updates
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))

    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, 
                    entity=opt.wandb_entity, 
                    config=exp_config_dict, 
                    settings=wandb.Settings(start_method="fork"))
        wandb.run.name = logdir.split('/')[-1]
        wandb.run.save()
        wandb.log({"graph_structure(GT-pred)/Ground truth": wandb.Image(join(logdir, 'gt_graph.png'))}, step=0)

    for i in range(n_intervention_sets + 1):
        print()
        dibs_update = 0
        z_final, theta_final, sf_baseline = None, None, None
        if i > 0 and num_interv_data == 0: break
        
        if opt.reinit is True:  target, model = get_target(key, opt)
        joint_dibs_model = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)
        dibs = Decoder_JointDiBS(opt.num_nodes, opt.num_samples, opt.proj_dims, opt.n_particles,
                                    model, opt.alpha_linear, latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes),
                                    grad_estimator=opt.grad_estimator, known_ED=opt.known_ED, linear_decoder=opt.linear_decoder, clamp = opt.clamp)

        if i == 0:  interv_targets = no_interv_targets[:opt.obs_data]
        else:       interv_targets = no_interv_targets[:opt.obs_data + (i*interv_data_per_set)]
        data_gt = x[:len(interv_targets), :]
    
        state = train_state.TrainState.create(
            apply_fn=dibs.apply,
            params=dibs.init(rng, rng, None, None, None, None, interv_targets, 0)['params'],
            tx=optax.adam(opt.lr))

        dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
        opt_init, opt_update, get_params = dibs_optimizer

        @jit
        def train_causal_vars(state, key, z, theta, sf_baseline, data, interv_targets, step=0):
            recons, particles, q_z_mus, q_z_covars, _, gs, sf_baseline, pred_zs = dibs.apply({'params': state.params}, key, z, theta, sf_baseline, data, interv_targets, step)
            grads = grad(loss_fns.loss_fn)(state.params, key, z, theta, sf_baseline, data, interv_targets, 
                        step, data_gt, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs)
            res = state.apply_gradients(grads=grads)
            loss, mse_loss, kl_z_loss, z_dist = loss_fns.calc_loss(recons, data_gt, p_z_covar, p_z_mu, q_z_covars, q_z_mus, pred_zs, opt, z_gt[:len(interv_targets)])
            return res, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, sf_baseline, z_dist, pred_zs, particles

        
        @jit
        def train_causal_graph(state, key, z, theta, sf_baseline, data, interv_targets, step, opt_states):
            recons, _, q_z_mus, q_z_covars, dibs_grads, gs, sf_baseline, _ = dibs.apply({'params': state.params}, 
                                                                                            key, z, theta, sf_baseline, 
                                                                                            data, interv_targets, step)
            opt_state_z, opt_state_theta = opt_states
            opt_state_z = opt_update(step, dibs_grads['phi_z'], opt_state_z)
            opt_state_theta = opt_update(step, dibs_grads['phi_theta'], opt_state_theta)
            return opt_state_z, opt_state_theta, gs, sf_baseline

        for step in range(opt.steps):
            if step < decoder_train_steps:
                state, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, _, z_dist, q_z, particles = train_causal_vars(state, rng, z_final, theta_final, sf_baseline, data=None, interv_targets=interv_targets, step=step)
                if step == 0: opt_state_z, opt_state_theta = opt_init(particles[0]), opt_init(particles[1])
            else:
                if step == decoder_train_steps:
                    q_z_mus, q_z_covars = np.array(q_z_mus), np.array(q_z_covars)
                    data = vmap(datagen.gen_data_from_dist, (None, 0, 0, None, None, None), 0)(rng, q_z_mus, q_z_covars, len(interv_targets), interv_targets, opt.clamp)  

                t = dibs_update
                opt_state_z, opt_state_theta, gs, sf_baseline = train_causal_graph(state, rng, z_final, theta_final, sf_baseline, 
                                                                data, interv_targets, t, (opt_state_z, opt_state_theta))
                z_final, theta_final = get_params(opt_state_z), get_params(opt_state_theta)
                dibs_update += 1      
        
        writer.add_scalar('z_losses/MSE', np.array(mse_loss), len(interv_targets) - opt.obs_data)
        writer.add_scalar('Distances/MSE(Predicted z | z_GT)', np.array(z_dist), len(interv_targets) - opt.obs_data)      
        wandb_log_dict = {'z_losses/MSE': np.array(mse_loss),
                           'Distances/MSE(Predicted z | z_GT)':  np.array(z_dist)}

        if opt.supervised is True:
            writer.add_scalar('z_losses/KL', np.array(kl_z_loss), len(interv_targets) - opt.obs_data)
            wandb_log_dict['z_losses/KL'] = np.array(kl_z_loss)

        evaluate(target, joint_dibs_model, gs, theta_final, len(interv_targets) - opt.obs_data, 
                dag_file, writer, opt, z_gt[:len(interv_targets)], interv_targets, True)


