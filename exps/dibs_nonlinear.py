import sys, os, graphical_models
sys.path.append('..')
import datagen, utils

sys.path.append(os.getcwd() + '/dibs_new')
sys.path.append(os.getcwd() + '/dibs_new/dibs')

import networkx as nx
import numpy as np
import jax.numpy as jnp
from jax import vmap, random, jit, grad

from dibs_new.dibs.target import make_nonlinear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood


def evaluate(target, dibs, gs, thetas, steps, dag_file, writer, opt, data, 
            interv_targets, tb_plots=False):
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
    dibs_mixture = dibs.get_mixture(gs, thetas, data, interv_targets) 
    
    eshd_e = np.array(expected_shd(dist=dibs_empirical, g=target.g))     
    eshd_m = np.array(expected_shd(dist=dibs_mixture, g=target.g))     
    auroc_e = threshold_metrics(dist=dibs_empirical, g=target.g)['roc_auc']
    auroc_m = threshold_metrics(dist=dibs_mixture, g=target.g)['roc_auc']
    
    sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_m, dag_file)
    
    if tb_plots:
        writer.add_scalar('(Interventional) Evaluations/AUROC (empirical)', auroc_e, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/AUROC (marginal)', auroc_m, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Empirical)', eshd_e, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/Exp. SHD (Marginal)', eshd_m, len(data) - opt.obs_data)
        writer.add_scalar('(Interventional) Evaluations/MEC or GT recovery %', mec_or_gt_count * 100.0/ opt.n_particles, len(data) - opt.obs_data)
    #     writer.add_scalar('Evaluations/NLL (empirical)', negll_e, steps)
    #     writer.add_scalar('Evaluations/NLL (marginal)', negll_m, steps)


    print()
    print(f"Metrics after {int(steps)} steps training on {opt.obs_data} obs data and {len(data) - opt.obs_data} interv. data")
    print()
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar('(Interventional) Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
    print("MEC-GT Recovery %", mec_or_gt_count * 100.0/ opt.n_particles)
    print(f"AUROC (Empirical and Marginal): {auroc_e} {auroc_m}")
    # print(f"Neg. log likelihood (Empirical and Marginal): {negll_e} {negll_m}")
    print()


def run_dibs_nonlinear(key, opt, n_intervention_sets, dag_file, writer, full_train=False):
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)

    target, model = make_nonlinear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, 
                mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    print()
    
    interv_data, no_interv_targets = datagen.generate_interv_data(opt, n_intervention_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    x = jnp.concatenate((obs_data, interv_data), axis=0)
    key, subk = random.split(key)
    
    if full_train:
        dibs = JointDiBS(n_vars=opt.num_nodes, 
                        inference_model=model,
                        alpha_linear=opt.alpha_linear,
                        grad_estimator_z=opt.grad_estimator)
    
        gs, z_final, theta_final, opt_state_z, opt_state_theta, sf_baseline = dibs.sample(steps=n_steps,
                                                                            key=subk, 
                                                                            data=x,
                                                                            interv_targets=no_interv_targets,
                                                                            n_particles=opt.n_particles,
                                                                            callback_every=100, 
                                                                            callback=dibs.visualize_callback())
    else:
        z_final, sf_baseline, opt_state_z, theta_final = None, None, None, None

        for i in range(n_intervention_sets + 1):
            dibs = JointDiBS(n_vars=opt.num_nodes, inference_model=model, alpha_linear=opt.alpha_linear, grad_estimator_z=opt.grad_estimator)

            if i == 0:
                interv_targets = no_interv_targets[:opt.obs_data]
                data = x[:opt.obs_data]
            else:
                interv_targets = no_interv_targets[:opt.obs_data + (i*interv_data_per_set)]
                data = x[:opt.obs_data + (i*interv_data_per_set)]

            gs, z_final, theta_final, opt_state_z, opt_state_theta, sf_baseline = dibs.sample(steps=n_steps, key=subk, 
                                                                    data=data,
                                                                    interv_targets=interv_targets,
                                                                    n_particles=opt.n_particles,
                                                                    opt_state_z=opt_state_z,
                                                                    z=z_final, theta=theta_final, sf_baseline=sf_baseline,
                                                                    callback_every=100, 
                                                                    callback=dibs.visualize_callback(),
                                                                    start=0)

            evaluate(target, dibs, gs, theta_final, int(n_steps*(i+1)), dag_file, 
                    writer, opt, data, interv_targets, True)
            if num_interv_data == 0: break


