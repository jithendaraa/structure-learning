import sys, os, imageio, graphical_models, pdb
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
import optax, loss_fns, wandb

from models.Decoder_JointDiBS import Decoder_JointDiBS
from dibs_new.dibs.target import make_nonlinear_gaussian_model, make_linear_gaussian_model
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_marginal_likelihood
from dibs_new.dibs.inference import JointDiBS

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

def run_decoder_joint_dibs_across_interv_data(key, opt, logdir, n_intervention_sets, dag_file, writer, exp_config_dict):
    target, model = get_target(key, opt)
    key, rng = random.split(key)

    gt_graph = nx.from_numpy_matrix(target.g, create_using=nx.DiGraph)
    nx.draw(gt_graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
    plt.savefig(join(logdir,'gt_graph.png'))
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    print()
    
    obs_data, interv_data, z_gt, no_interv_targets, x, _, _ = datagen.get_data(opt, n_intervention_sets, target)
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
        if i > 0 and num_interv_data == 0: break
        dibs_update = 0
        z_final, theta_final, sf_baseline = None, None, None
        joint_dibs_model = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)
        
        if i == 0:  interv_targets = no_interv_targets[:opt.obs_data]
        else:       interv_targets = no_interv_targets[:opt.obs_data + (i*interv_data_per_set)]
        data_gt = x[:len(interv_targets), :]

        dibs = Decoder_JointDiBS(opt.num_nodes, opt.num_samples, opt.proj_dims, opt.n_particles,
                                    model, opt.alpha_linear, latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes),
                                    grad_estimator=opt.grad_estimator, known_ED=opt.known_ED, linear_decoder=opt.linear_decoder)
    
        state = train_state.TrainState.create(
            apply_fn=dibs.apply,
            params=dibs.init(rng, rng, None, None, None, None, interv_targets, 0)['params'],
            tx=optax.adam(opt.lr))

        dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
        opt_init, opt_update, get_params = dibs_optimizer

        @jit
        def train_causal_vars(state, key, z, theta, sf_baseline, data, interv_targets, step=0):
            recons, particles, q_z_mus, q_z_covars, _, gs, sf_baseline, pred_zs = dibs.apply({'params': state.params}, 
                                                                                                key, z, theta, sf_baseline, 
                                                                                                data, interv_targets, step)
            
            grads = grad(loss_fns.loss_fn)(state.params, key, z, theta, sf_baseline, data, interv_targets, 
                        step, data_gt, None, None, q_z_covars, q_z_mus, opt, dibs)
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
        #     else:
        #         if step == decoder_train_steps:
        #             q_z_mus, q_z_covars = np.array(q_z_mus), np.array(q_z_covars)
        #             data = vmap(datagen.gen_data_from_dist, (None, 0, 0, None), 0)(rng, q_z_mus, q_z_covars, len(interv_targets))  

        #         t = dibs_update
        #         opt_state_z, opt_state_theta, gs, sf_baseline = train_causal_graph(state, rng, z_final, theta_final, sf_baseline, 
        #                                                         data, interv_targets, t, (opt_state_z, opt_state_theta))
        #         z_final, theta_final = get_params(opt_state_z), get_params(opt_state_theta)
        #         dibs_update += 1  
        
        # writer.add_scalar('z_losses/MSE', np.array(mse_loss), len(interv_targets) - opt.obs_data)
        # writer.add_scalar('Distances/MSE(Predicted z | z_GT)', np.array(z_dist), len(interv_targets) - opt.obs_data)      
        
        # wandb_log_dict = {'z_losses/MSE': np.array(mse_loss),
        #                    'Distances/MSE(Predicted z | z_GT)':  np.array(z_dist)}

        # evaluate(target, writer, joint_dibs_model, gs, theta_final, x, None, 
        #             gt_graph, dag_file, opt, len(interv_targets) - opt.obs_data, wandb_log_dict, True)


def evaluate(target, writer, joint_dibs_model, gs, theta, data, interv_targets, 
            gt_graph, dag_file, opt, steps, wandb_log_dict, tb_plots=False):
    gt_graph = nx.from_numpy_matrix(np.array(target.g), create_using=nx.DiGraph)
    gt_graph_cpdag = graphical_models.DAG.from_nx(gt_graph).cpdag()
    cpdag_shds = []
    for adj_mat in gs:
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
        try:
            G_cpdag = graphical_models.DAG.from_nx(G).cpdag()
            shd = gt_graph_cpdag.shd(G_cpdag)
            cpdag_shds.append(shd)
        except:
            pass

    dibs_empirical = joint_dibs_model.get_empirical(gs, data)
    eshd_e = np.array(expected_shd(dist=dibs_empirical, g=target.g)) 
    auroc_e = threshold_metrics(dist=dibs_empirical, g=target.g)['roc_auc']
    try:
        dibs_mixture = joint_dibs_model.get_mixture(gs, theta, data, interv_targets)
        eshd_m = np.array(expected_shd(dist=dibs_mixture, g=target.g))     
        auroc_m = threshold_metrics(dist=dibs_mixture, g=target.g)['roc_auc']
        sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_m, dag_file)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Marginal)', eshd_m, steps)
        writer.add_scalar('(Interventional) Evaluations/AUROC (marginal)', auroc_m, steps)

    except:
        sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_e, dag_file)

    if tb_plots:
        writer.add_scalar('(Interventional) Evaluations/AUROC (empirical)', auroc_e, steps)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Empirical)', eshd_e, steps)
        writer.add_scalar('(Interventional) Evaluations/MEC or GT recovery %', mec_or_gt_count * 100.0 / opt.n_particles, steps)
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, steps, dataformats='HWC')

    if opt.off_wandb is False:
        wandb_log_dict['(Interventional) Evaluations/AUROC (empirical)'] = auroc_e
        wandb_log_dict['(Interventional) Evaluations/Exp. SHD (Empirical)'] = eshd_e
        wandb_log_dict['(Interventional) Evaluations/MEC or GT recovery %'] = mec_or_gt_count * 100.0/ opt.n_particles
        wandb_log_dict['graph_structure(GT-pred)/Posterior sampled graphs'] = wandb.Image(sampled_graph)

    print()
    print(f"Metrics after {opt.steps} steps training on {opt.obs_data} obs data and {steps} interv. data")
    print()
    print("ESHD (empirical):", eshd_e)
    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar('(Interventional) Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
        if opt.off_wandb is False: wandb_log_dict['(Interventional) Evaluations/CPDAG SHD'] = np.mean(cpdag_shds)

    print("MEC-GT Recovery %", mec_or_gt_count * 100.0/ opt.n_particles)
    print(f"AUROC (Empirical): {auroc_e}")
    print()

    if opt.off_wandb is False:  wandb.log(wandb_log_dict, step=steps)


