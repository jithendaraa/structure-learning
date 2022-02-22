import os, sys, pdb, graphical_models, imageio, wandb
sys.path.append('../')
sys.path.append('../../')

import numpy as np
from jax import numpy as jnp
from jax import random
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join


from dibs_new.dibs.target import make_linear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
import datagen, utils
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

def evaluate(target, dibs, gs, thetas, steps, dag_file, writer, opt, data, 
            interv_targets, tb_plots=False):
    wandb_log_dict = {}
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

    dibs_empirical = dibs.get_empirical(gs, thetas)
    eshd_e = np.array(expected_shd(dist=dibs_empirical, g=target.g))     
    auroc_e = threshold_metrics(dist=dibs_empirical, g=target.g)['roc_auc']
    
    dibs_mixture = dibs.get_mixture(gs, thetas, data, interv_targets) 
    eshd_m = np.array(expected_shd(dist=dibs_mixture, g=target.g))     
    auroc_m = threshold_metrics(dist=dibs_mixture, g=target.g)['roc_auc']
    
    sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_m, dag_file)
    mec_gt_recovery = mec_or_gt_count * 100.0 / opt.n_particles
    
    if tb_plots:
        writer.add_scalar('(Interventional) Evaluations/AUROC (empirical)', auroc_e, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/AUROC (marginal)', auroc_m, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Empirical)', eshd_e, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Marginal)', eshd_m, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/MEC or GT recovery %', mec_or_gt_count * 100.0/ opt.n_particles, len(data) - opt.obs_data)
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, len(data) - opt.obs_data, dataformats='HWC')

        wandb_log_dict['(Interventional) Evaluations/AUROC (empirical)'] = auroc_e
        wandb_log_dict['(Interventional) Evaluations/AUROC (marginal)'] = auroc_m
        wandb_log_dict['(Interventional) Evaluations/Exp. SHD (Empirical)'] = eshd_e
        wandb_log_dict['(Interventional) Evaluations/Exp. SHD (Marginal)'] = eshd_m
        wandb_log_dict['(Interventional) Evaluations/MEC or GT recovery %'] = mec_gt_recovery
        wandb_log_dict['graph_structure(GT-pred)/Posterior sampled graphs'] = wandb.Image(sampled_graph)

    print()
    print(f"Metrics after training on {opt.obs_data} obs data and {len(data) - opt.obs_data} interv. data")
    print(f"AUROC (Empirical and Marginal): {auroc_e} {auroc_m}")
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar('(Interventional) Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
        wandb_log_dict['Evaluations/CPDAG SHD'] = np.mean(cpdag_shds)

    print("MEC-GT Recovery %", mec_gt_recovery)
    print()
    if opt.off_wandb is False:  wandb.log(wandb_log_dict, step=steps)



def run_dibs_linear(key, opt, n_interv_sets, dag_file, writer, logdir):

    exp_config_dict = vars(opt)
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_per_set = int(num_interv_data / n_interv_sets)  
    n_steps = int(opt.num_updates / n_interv_sets) if num_interv_data > 0 else opt.num_updates

    target, model = make_linear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, 
                mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
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

    interv_data, no_interv_targets = datagen.generate_interv_data(opt, n_interv_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    x = jnp.concatenate((obs_data, interv_data), axis=0)
    key, subk = random.split(key)

    if opt.across_interv is False:
        dibs = JointDiBS(n_vars=opt.num_nodes, inference_model=model,
                        alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)

        gs, z_final, theta_final, opt_state_z, opt_state_theta, sf_baseline = dibs.sample(steps=n_steps, key=subk, data=x,
                                                                            interv_targets=no_interv_targets,
                                                                            n_particles=opt.n_particles, callback_every=100, start=0,
                                                                            callback=dibs.visualize_callback(), jitted=True)


        evaluate(target, dibs, gs, theta_final, 0, dag_file, 
                        writer, opt, x, no_interv_targets, False)

    else:
        z_final, sf_baseline, opt_state_z, theta_final = None, None, None, None

        for i in range(n_interv_sets + 1):
            dibs = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)

            if i == 0:
                interv_targets = no_interv_targets[:opt.obs_data]
                data = x[:opt.obs_data]
            else:
                interv_targets = no_interv_targets[:opt.obs_data + (i*interv_data_per_set)]
                data = x[:opt.obs_data + (i*interv_data_per_set)]

            gs, z_final, theta_final, opt_state_z, opt_state_theta, sf_baseline = dibs.sample(steps=opt.num_updates, key=subk, data=data, 
                                                                                    interv_targets=interv_targets, n_particles=opt.n_particles, 
                                                                                    opt_state_z=opt_state_z, z=z_final, theta=theta_final, 
                                                                                    sf_baseline=sf_baseline, callback_every=100, start=0, 
                                                                                    callback=dibs.visualize_callback(), jitted=False)

            evaluate(target, dibs, gs, theta_final, int(n_steps*(i+1)), dag_file, 
                    writer, opt, data, interv_targets, True)
            
            if num_interv_data == 0: break
