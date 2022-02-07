import sys, os, graphical_models
sys.path.append('..')
import datagen, utils

sys.path.append(os.getcwd() + '/dibs_new')
sys.path.append(os.getcwd() + '/dibs_new/dibs')

import networkx as nx
import numpy as np
import jax.numpy as jnp
from jax import vmap, random, jit, grad

from dibs_new.dibs.inference import MarginalDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.target import make_linear_gaussian_equivalent_model
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_marginal_likelihood

def evaluate(target, dibs, gs, steps, dag_file, writer, opt, data, interv_targets, tb_plots=False):
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

    dibs_empirical = dibs.get_empirical(gs)
    dibs_mixture = dibs.get_mixture(gs, data, interv_targets) 
    eshd_e = expected_shd(dist=dibs_empirical, g=target.g, unique=True)     
    eshd_m = expected_shd(dist=dibs_mixture, g=target.g, unique=False)     
    auroc_e = threshold_metrics(dist=dibs_empirical, g=target.g, unique=True)['roc_auc']
    auroc_m = threshold_metrics(dist=dibs_mixture, g=target.g, unique=False)['roc_auc']
    
    negll_e = neg_ave_log_marginal_likelihood(dist=dibs_empirical, eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood, 
                                                x=target.x_ho, unique=True)
    negll_m = neg_ave_log_marginal_likelihood(dist=dibs_mixture, eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood, 
                                                x=target.x_ho, unique=False)
    
    if tb_plots:
        writer.add_scalar('Evaluations/AUROC (empirical)', auroc_e, steps)
        writer.add_scalar('Evaluations/AUROC (marginal)', auroc_m, steps)
        writer.add_scalar('Evaluations/Exp. SHD (Empirical)', eshd_e, steps)
        writer.add_scalar('Evaluations/Exp. SHD (Marginal)', eshd_m, steps)
        writer.add_scalar('Evaluations/MEC or GT recovery %', mec_or_gt_count * 100.0/ opt.n_particles, steps)
        writer.add_scalar('Evaluations/NLL (empirical)', negll_e, steps)
        writer.add_scalar('Evaluations/NLL (marginal)', negll_m, steps)

    sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_m, dag_file)

    print()
    print(f"Metrics after {int(steps)} steps")
    print()
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar('Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
    print("MEC-GT Recovery %", mec_or_gt_count * 100.0/ opt.n_particles)
    print(f"AUROC (Empirical and Marginal): {auroc_e} {auroc_m}")
    print(f"Neg. log likelihood (Empirical and Marginal): {negll_e} {negll_m}")


def run_dibs_bge_new(key, opt, n_intervention_sets, dag_file, writer):
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)
    
    target, model = make_linear_gaussian_equivalent_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, 
                mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    
    interv_data, no_interv_targets = datagen.generate_interv_data(opt, n_intervention_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    x = jnp.concatenate((obs_data, interv_data), axis=0)

    dibs = MarginalDiBS(n_vars=opt.num_nodes, 
                        inference_model=model,
                        alpha_linear=opt.alpha_linear,
                        grad_estimator_z=opt.grad_estimator)
    key, subk = random.split(key)
    gs, z_final, opt_state_z, sf_baseline = dibs.sample(steps=n_steps,
                                            key=subk, 
                                            data=obs_data,
                                            interv_targets=no_interv_targets[:opt.obs_data],
                                            n_particles=opt.n_particles,
                                            opt_state_z=None,
                                            z=None,
                                            sf_baseline=None,
                                            callback_every=100, 
                                            callback=dibs.visualize_callback(),
                                            start=0)
    
    evaluate(target, dibs, gs, n_steps, dag_file, writer, opt, obs_data, no_interv_targets[:opt.obs_data])

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
    #         evaluate(data, log_likelihood, interv_targets, 
    #                 particles_g, int(start_), gt_graph, dag_file, 
    #                 adj_matrix, writer, opt, True)