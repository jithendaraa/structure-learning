import os, sys, pdb, graphical_models, imageio, wandb
sys.path.append('../')
sys.path.append('../../')
import utils

import networkx as nx
import numpy as np
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood


def evaluate(target, dibs, gs, thetas, steps, dag_file, writer, opt, data, 
            interv_targets, tb_plots=False, wandb_log_dict={}):
    """
        [TODO]
    """
    title = '(Interventional) ' if opt.across_interv is True else ''
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
        writer.add_scalar(title + 'Evaluations/AUROC (empirical)', auroc_e, steps)
        writer.add_scalar(title + 'Evaluations/AUROC (marginal)', auroc_m, steps)
        writer.add_scalar(title + 'Evaluations/Exp. SHD (Empirical)', eshd_e, steps)
        writer.add_scalar(title + 'Evaluations/Exp. SHD (Marginal)', eshd_m, steps)
        writer.add_scalar(title + 'Evaluations/MEC or GT recovery %', mec_gt_recovery, steps)
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, steps, dataformats='HWC')

        wandb_log_dict[title + 'Evaluations/AUROC (empirical)'] = auroc_e
        wandb_log_dict[title + 'Evaluations/AUROC (marginal)'] = auroc_m
        wandb_log_dict[title + 'Evaluations/Exp. SHD (Empirical)'] = eshd_e
        wandb_log_dict[title + 'Evaluations/Exp. SHD (Marginal)'] = eshd_m
        wandb_log_dict[title + 'Evaluations/MEC or GT recovery %'] = mec_gt_recovery
        wandb_log_dict['graph_structure(GT-pred)/Posterior sampled graphs'] = wandb.Image(sampled_graph)

    print()
    if opt.obs_data == opt.num_samples:
        print(f"Metrics after training on {opt.obs_data} obs data")
    else:
        print(f"Metrics after training on {opt.obs_data} obs data and {steps} interv. data")
    
    print(f"AUROC (Empirical and Marginal): {auroc_e} {auroc_m}")
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar(title + 'Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
        wandb_log_dict[title + 'Evaluations/CPDAG SHD'] = np.mean(cpdag_shds)
    print("MEC-GT Recovery %", mec_gt_recovery)
    print()
    if opt.off_wandb is False:  wandb.log(wandb_log_dict, step=steps)