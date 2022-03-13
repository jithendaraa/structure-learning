import os, sys, pdb, graphical_models, imageio, wandb, functools
sys.path.append('../')
sys.path.append('../..')

import numpy as np
from jax import numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join

from flax.training import train_state
from jax import vmap, random, jit, grad, device_put
from jax.experimental import optimizers
import jax.scipy.stats as dist
import optax, loss_fns, datagen, utils

from models.Decoder_JointDiBS import Decoder_JointDiBS
from dibs_new.dibs.target import make_linear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
from eval_ import evaluate


def get_target(key, opt):
    target, model = make_linear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                                            graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                                            obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                                            n_observations = opt.num_samples, n_ho_observations = opt.num_samples)
    return target, model


def run_decoder_joint_dibs_linear(key, opt, logdir, n_interv_sets, dag_file, writer):
    exp_config_dict = vars(opt)
    if opt.topsort is True: assert opt.supervised is False

    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_per_set = int(num_interv_data / n_interv_sets)  
    n_steps = opt.num_updates / n_interv_sets if num_interv_data > 0 else opt.num_updates 

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
    
    obs_data, interv_data, z_gt, no_interv_targets, x, p_z_mu, p_z_covar = datagen.get_data(opt, n_interv_sets, target)
    
    if opt.across_interv is False:
        run_linear_decoder_dibs(opt, target, dag_file, rng, model, x, z_gt, 
                                no_interv_targets, p_z_mu, p_z_covar, writer)
    elif opt.across_interv is True:
        run_linear_decoder_dibs_across_interv(opt, target, n_interv_sets, dag_file, rng, model, x, z_gt, 
                                                no_interv_targets, p_z_mu, p_z_covar, writer)




def run_linear_decoder_dibs(opt, target, dag_file, rng, model, x, z_gt, interv_targets, p_z_mu, p_z_covar, writer):
    z_final, theta_final, sf_baseline = None, None, jnp.zeros((opt.n_particles))
    data = None
    decoder_train_steps = opt.steps - opt.num_updates
    dibs_update = 0 

    joint_dibs_model = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)

    dibs = Decoder_JointDiBS(opt.num_nodes, opt.num_samples, opt.proj_dims, 
                                opt.n_particles, model, opt.alpha_linear, dibs_type=opt.likelihood, 
                                latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes),
                                grad_estimator=opt.grad_estimator, known_ED=opt.known_ED, 
                                linear_decoder=opt.linear_decoder, clamp = opt.clamp,
                                topsort = opt.topsort, obs_noise=opt.noise_sigma)

    state = train_state.TrainState.create(
        apply_fn=dibs.apply,
        params=dibs.init(rng, rng, None, None, sf_baseline, None, interv_targets, 0, 'linear')['params'],
        tx=optax.adam(opt.lr)
    )
    
    dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
    opt_init, opt_update, get_params = dibs_optimizer

    # @jit
    def train_causal_vars(state, key, z, theta, sf_baseline, data, interv_targets, step=0, supervised=False):
        recons, particles, q_z_mus, q_z_covars, _, gs, sf_baseline, pred_zs = dibs.apply({'params': state.params}, 
                                                                                            key, z, theta, sf_baseline, 
                                                                                            data, interv_targets, step, 'linear')
        
        grads = grad(loss_fns.loss_fn)(state.params, key, z, theta, sf_baseline, data, interv_targets, 
                    step, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs, 'linear', supervised=supervised)
        res = state.apply_gradients(grads=grads)
        loss, mse_loss, kl_z_loss, z_dist = loss_fns.calc_loss(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, pred_zs, opt, z_gt)
        return res, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, sf_baseline, z_dist, pred_zs, particles

    # @jit
    def train_causal_graph(state, key, z, theta, sf_baseline, data, interv_targets, step, opt_states):
        recons, _, q_z_mus, q_z_covars, dibs_grads, gs, sf_baseline, _ = dibs.apply({'params': state.params}, 
                                                                                        key, z, theta, sf_baseline, 
                                                                                        data, interv_targets, step, 'linear')
        opt_state_z, opt_state_theta = opt_states
        opt_state_z = opt_update(step, dibs_grads['phi_z'], opt_state_z)
        opt_state_theta = opt_update(step, dibs_grads['phi_theta'], opt_state_theta)
        return opt_state_z, opt_state_theta, gs, sf_baseline

    for step in range(opt.steps):
        if step < decoder_train_steps:
            state, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, _, z_dist, q_z, particles = train_causal_vars(state, rng, z_final, theta_final, 
                                                                                                                sf_baseline, data=None, interv_targets=interv_targets, step=step,
                                                                                                                supervised=opt.supervised)
            
            if step == 0:   opt_state_z, opt_state_theta = opt_init(particles[0]), opt_init(particles[1])

            if (step+1) % 50 == 0:
                if opt.supervised is True:
                    writer.add_scalar('z_losses/ELBO', np.array(loss), step)
                    writer.add_scalar('z_losses/KL', np.array(kl_z_loss), step)

                writer.add_scalar('z_losses/MSE', np.array(mse_loss), step)
                writer.add_scalar('Distances/MSE(Predicted z | z_GT)', np.array(z_dist), step)
        else:
            if step == decoder_train_steps and opt.topsort is False:
                q_z_mus, q_z_covars = np.array(q_z_mus), np.array(q_z_covars)
                data = vmap(datagen.gen_data_from_dist, (None, 0, 0, None, None, None), 0)(rng, q_z_mus, q_z_covars, opt.num_samples, interv_targets, opt.clamp)  
    
            t = dibs_update
            opt_state_z, opt_state_theta, gs, sf_baseline = train_causal_graph(state, rng, z_final, theta_final, sf_baseline, 
                                                            data, interv_targets, t, (opt_state_z, opt_state_theta))
    
            z_final, theta_final = get_params(opt_state_z), get_params(opt_state_theta)
            dibs_update += 1

        if step % 5 == 0:
            print(f"Step {step}: {np.array(mse_loss)}")
        
        if step >= decoder_train_steps and (step+1) % 5 == 0:
            evaluate(target, joint_dibs_model, gs, theta_final, step, 
                    dag_file, writer, opt, z_gt, interv_targets, True)


def run_linear_decoder_dibs_across_interv(opt, target, n_interv_sets, dag_file, rng, model, x, z_gt, no_interv_targets, p_z_mu, p_z_covar, writer):
    decoder_train_steps = opt.steps - opt.num_updates
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_per_set = int(num_interv_data / n_interv_sets)
    z_final, theta_final, sf_baseline = None, None, jnp.zeros((opt.n_particles))
    interv_targets = jnp.zeros((opt.num_samples, opt.num_nodes)).astype(bool)

    joint_dibs_model = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)
    dibs = Decoder_JointDiBS(opt.num_nodes, opt.num_samples, opt.proj_dims, 
                            opt.n_particles, model, opt.alpha_linear, dibs_type=opt.likelihood, 
                            latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes),
                            grad_estimator=opt.grad_estimator, known_ED=opt.known_ED, 
                            linear_decoder=opt.linear_decoder, clamp = opt.clamp)
    
    state = train_state.TrainState.create(
            apply_fn=dibs.apply,
            params=dibs.init(rng, rng, None, None, sf_baseline, None, interv_targets, 0, 'linear')['params'],
            tx=optax.adam(opt.lr))

    dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
    opt_init, opt_update, get_params = dibs_optimizer 
    
    for i in range(n_interv_sets + 1):
        dibs_update = 0
        if i > 0 and num_interv_data == 0: break

        interv_targets = no_interv_targets[:opt.obs_data + (i*interv_data_per_set)]
        data_gt = x[:len(interv_targets), :]
        print("Intervened on node", jnp.where(interv_targets[-1] == True)[0])

        if opt.reinit is True:  target, model = get_target(rng, opt)

        supervised = False
        if len(interv_targets) == opt.obs_data: supervised = opt.supervised
        print("supervised", supervised, i)
        
        @jit
        def train_causal_vars(state, key, z, theta, sf_baseline, data, interv_targets, step=0):
            recons, particles, q_z_mus, q_z_covars, _, gs, sf_baseline, pred_zs = dibs.apply({'params': state.params}, 
                                                                                                key, z, theta, sf_baseline, 
                                                                                                data, interv_targets, step, 'linear')
            
            grads = grad(loss_fns.loss_fn)(state.params, key, z, theta, sf_baseline, data, interv_targets, step, data_gt, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs, 'linear', supervised)
            res = state.apply_gradients(grads=grads)
            loss, mse_loss, kl_z_loss, z_dist = loss_fns.calc_loss(recons, data_gt, p_z_covar, p_z_mu, q_z_covars, q_z_mus, pred_zs, opt, z_gt[:len(interv_targets)], supervised)
            return res, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, sf_baseline, z_dist, pred_zs, particles

        @jit
        def train_causal_graph(state, key, z, theta, sf_baseline, data, interv_targets, step, opt_states):
            recons, _, q_z_mus, q_z_covars, dibs_grads, gs, sf_baseline, pred_zs = dibs.apply({'params': state.params}, 
                                                                                            key, z, theta, sf_baseline, 
                                                                                            data, interv_targets, step, 'linear')
            opt_state_z, opt_state_theta = opt_states
            opt_state_z = opt_update(step, dibs_grads['phi_z'], opt_state_z)
            opt_state_theta = opt_update(step, dibs_grads['phi_theta'], opt_state_theta)
            return opt_state_z, opt_state_theta, gs, sf_baseline, pred_zs

        for step in range(opt.steps):
            if step < decoder_train_steps:
                state, particles, loss, mse_loss, kl_z_loss, q_z_mus, q_z_covars, gs, _, z_dist, q_z, particles = train_causal_vars(state, rng, z_final, theta_final, 
                                                                                                                       sf_baseline, data=None, interv_targets=interv_targets, step=step)
                if step == 0: opt_state_z, opt_state_theta = opt_init(particles[0]), opt_init(particles[1])
            
            else:
                if step == decoder_train_steps:
                    q_z_mus, q_z_covars = np.array(q_z_mus), np.array(q_z_covars)
                    data = vmap(datagen.gen_data_from_dist, (None, 0, 0, None, None, None), 0)(rng, q_z_mus, q_z_covars, len(interv_targets), interv_targets, opt.clamp)  

                t = dibs_update
                opt_state_z, opt_state_theta, gs, sf_baseline, q_z = train_causal_graph(state, rng, z_final, theta_final, sf_baseline, 
                                                                data, interv_targets, t, (opt_state_z, opt_state_theta))
                z_final, theta_final = get_params(opt_state_z), get_params(opt_state_theta)
                dibs_update += 1    

        writer.add_scalar('z_losses/MSE', np.array(mse_loss), len(interv_targets) - opt.obs_data)
        writer.add_scalar('Distances/MSE(Predicted z | z_GT)', np.array(z_dist), len(interv_targets) - opt.obs_data)      
        wandb_log_dict = {'z_losses/MSE': np.array(mse_loss), 
                            'Distances/MSE(Predicted z | z_GT)':  np.array(z_dist)}

        if supervised is True:
            writer.add_scalar('z_losses/KL', np.array(kl_z_loss), len(interv_targets) - opt.obs_data)
            wandb_log_dict['z_losses/KL'] = np.array(kl_z_loss)               

        if i > 0:
            _, _, _, obs_z_dist = loss_fns.calc_loss(None, None, None, None, None, None, q_z[:, :opt.obs_data, ...], opt, 
                                    z_gt[ : opt.obs_data], only_z=True)

            _, _, _, interv_z_dist = loss_fns.calc_loss(None, None, None, None, None, None, q_z[:, opt.obs_data:, ...], opt, 
                                    z_gt[opt.obs_data : opt.obs_data + i*interv_data_per_set ], only_z=True)

            writer.add_scalar('Distances/MSE(Predicted interv. z | interv. z_GT)', np.array(interv_z_dist), len(interv_targets) - opt.obs_data)      
            wandb_log_dict['Distances/MSE(Predicted interv. z | interv. z_GT)'] = np.array(interv_z_dist)

            writer.add_scalar('Distances/MSE(Predicted obs. z | obs. z_GT)', np.array(obs_z_dist), len(interv_targets) - opt.obs_data)      
            wandb_log_dict['Distances/MSE(Predicted obs. z | obs. z_GT)'] = np.array(obs_z_dist)
        
        evaluate(target, joint_dibs_model, gs, theta_final, len(interv_targets) - opt.obs_data, dag_file, writer, opt, z_gt[:len(interv_targets)], interv_targets, True, wandb_log_dict)
        print()
