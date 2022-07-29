import os, sys, pdb, graphical_models, imageio, wandb, cdt
sys.path.append('../')
sys.path.append('../../')
sys.path.append("../../modules")
sys.path.append("../../models")

import utils
import matplotlib.pyplot as plt
from os.path import join
import networkx as nx
import numpy as np
import numpy as onp
import jax.numpy as jnp
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from modules.divergences import *
from dag_utils import count_accuracy
from vae_bcd_utils import auroc
from jax import jit
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm



def evaluate(target, dibs, gs, thetas, steps, dag_file, writer, opt, data, 
            interv_targets, tb_plots=False, wandb_log_dict={}, logdir=''):
    """
        [TODO]
    """
    auroc_m, auroc_e = None, None
    title = ''
    if opt.across_interv is True: title = '(Interventional) '
    gt_graph = nx.from_numpy_matrix(np.array(target.g), create_using=nx.DiGraph)
    gt_graph_cpdag = graphical_models.DAG.from_nx(gt_graph).cpdag()

    cpdag_shds = []
    for adj_mat in gs:
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
        try:
            G_cpdag = graphical_models.DAG.from_nx(G).cpdag()
            shd = gt_graph_cpdag.shd(G_cpdag)
            cpdag_shds.append(shd)
        except: pass

    dibs_empirical = dibs.get_empirical(gs, thetas)
    eshd_e = np.array(expected_shd(dist=dibs_empirical, g=target.g))     
    try: auroc_e = threshold_metrics(dist=dibs_empirical, g=target.g)['roc_auc']
    except: pass
    
    dibs_mixture = dibs.get_mixture(gs, thetas, data, interv_targets) 
    eshd_m = np.array(expected_shd(dist=dibs_mixture, g=target.g))     
    try: auroc_m = threshold_metrics(dist=dibs_mixture, g=target.g)['roc_auc']
    except: pass

    sampled_graph, mec_or_gt_count = utils.log_dags(gs, gt_graph, eshd_e, eshd_m, dag_file)
    mec_gt_recovery = mec_or_gt_count * 100.0 / opt.n_particles
    
    if tb_plots:
        if auroc_e:
            writer.add_scalar(title + 'Evaluations/AUROC (empirical)', auroc_e, steps)
            wandb_log_dict[title + 'Evaluations/AUROC (empirical)'] = auroc_e
        
        if auroc_m:
            writer.add_scalar(title + 'Evaluations/AUROC (marginal)', auroc_m, steps)
            wandb_log_dict[title + 'Evaluations/AUROC (marginal)'] = auroc_m
        
        writer.add_scalar(title + 'Evaluations/Exp. SHD (Empirical)', eshd_e, steps)
        writer.add_scalar(title + 'Evaluations/Exp. SHD (Marginal)', eshd_m, steps)
        writer.add_scalar(title + 'Evaluations/MEC or GT recovery %', mec_gt_recovery, steps)
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, steps, dataformats='HWC')

        wandb_log_dict[title + 'Evaluations/Exp. SHD (Empirical)'] = eshd_e
        wandb_log_dict[title + 'Evaluations/Exp. SHD (Marginal)'] = eshd_m
        wandb_log_dict[title + 'Evaluations/MEC or GT recovery %'] = mec_gt_recovery
        wandb_log_dict['graph_structure(GT-pred)/Posterior sampled graphs'] = wandb.Image(sampled_graph)

    print()
    if opt.obs_data == opt.num_samples:
        print(f"Metrics after training on {opt.obs_data} obs data")
    else:
        print(f"Metrics after training on {opt.obs_data} obs data and {steps} interv. data")
    
    if auroc_e and auroc_m: print(f"AUROC (Empirical and Marginal): {auroc_e} {auroc_m}")
    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)

    if len(cpdag_shds) > 0: 
        print("Expected CPDAG SHD:", np.mean(cpdag_shds))
        writer.add_scalar(title + 'Evaluations/CPDAG SHD', np.mean(cpdag_shds), steps)
        wandb_log_dict[title + 'Evaluations/CPDAG SHD'] = np.mean(cpdag_shds)

    print("MEC-GT Recovery %", mec_gt_recovery)
    print()
    if opt.off_wandb is False:  
        wandb.log(wandb_log_dict, step=steps)
    
    return auroc_e, auroc_m, eshd_e, eshd_m, mec_gt_recovery


def log_gt_graph(ground_truth_W, logdir, exp_config_dict, opt, writer=None):
    plt.imshow(ground_truth_W)
    plt.savefig(join(logdir, 'gt_w.png'))

    # ? Logging to wandb
    if opt.off_wandb is False:
        if opt.offline_wandb is True: os.system('wandb offline')
        else:   os.system('wandb online')
        
        wandb.init(project=opt.wandb_project, 
                    entity=opt.wandb_entity, 
                    config=exp_config_dict, 
                    settings=wandb.Settings(start_method="fork"))
        wandb.run.name = logdir.split('/')[-1]
        wandb.run.save()
        wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png'))}, step=0)

    # ? Logging to tensorboard
    gt_graph_image = onp.asarray(imageio.imread(join(logdir, 'gt_w.png')))
    if writer:
        writer.add_image('graph_structure(GT-pred)/Ground truth W', gt_graph_image, 0, dataformats='HWC')


def print_metrics(i, loss, mse_dict, mean_dict, opt):
    print()
    print(f"Step {i} | {loss}")
    print(f"Z_MSE: {mse_dict['z_mse']} | X_MSE: {mse_dict['x_mse']}")
    print(f"L MSE: {mse_dict['L_mse']}")
    print(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
    
    if opt.Z_KL is True:
        print(f"Z_KL: {np.array(mse_dict['Z_KL'])}")



def eval_mean(model_params, x_input, data, rng_key, interv_values, do_shd_c=True, 
            step = None, interv_nodes=None, forward=None, gt_L=None, 
            ground_truth_W=None, ground_truth_sigmas=None, opt=None, P=None):
    
    """
        Computes mean error statistics for P, L parameters and data
        data should be observational
    """
    edge_threshold = 0.3
    hard = True
    dim = opt.num_nodes
    do_ev_noise = opt.do_ev_noise

    if opt.do_ev_noise: noise_dim = 1
    else: noise_dim = dim

    @jit
    def forward_pass(model_params, rng_key, hard, x_input, interv_nodes, interv_values, P):
        (   batched_P, 
            batched_P_logits, 
            batched_L, 
            batched_log_noises, 
            batched_W, 
            batched_qz_samples, 
            full_l_batch, 
            full_log_prob_l, 
            X_recons 
        ) = forward.apply(model_params, rng_key, opt, hard, rng_key, x_input, interv_nodes, interv_values, P=P)

        return batched_W, full_l_batch

    batched_W, full_l_batch = forward_pass(model_params, rng_key, True, x_input, interv_nodes, interv_values, P)
    data = data[:opt.obs_data]
    z_prec = onp.linalg.inv(jnp.cov(data.T))
    w_noise = full_l_batch[:, -noise_dim:]
    Xs = data[:opt.obs_data]
    auprcs_w, auprcs_g = [], []

    def sample_stats(est_W, noise, threshold=0.3, get_wasserstein=False):
        if do_ev_noise is False: raise NotImplementedError("")
        
        est_noise = jnp.ones(dim) * jnp.exp(noise)
        est_W_clipped = jnp.where(jnp.abs(est_W) > threshold, est_W, 0)
        gt_graph_clipped = jnp.where(jnp.abs(ground_truth_W) > threshold, est_W, 0)
        
        binary_est_W = jnp.where(est_W_clipped, 1, 0)
        binary_gt_graph = jnp.where(gt_graph_clipped, 1, 0)

        gt_graph_w = nx.from_numpy_matrix(np.array(gt_graph_clipped), create_using=nx.DiGraph)
        pred_graph_w = nx.from_numpy_matrix(np.array(est_W_clipped), create_using=nx.DiGraph)

        gt_graph_g = nx.from_numpy_matrix(np.array(binary_gt_graph), create_using=nx.DiGraph)
        pred_graph_g = nx.from_numpy_matrix(np.array(binary_est_W), create_using=nx.DiGraph)

        auprcs_w.append(cdt.metrics.precision_recall(gt_graph_w, pred_graph_w)[0])
        auprcs_g.append(cdt.metrics.precision_recall(gt_graph_g, pred_graph_g)[0])

        stats = count_accuracy(ground_truth_W, est_W_clipped)

        if get_wasserstein:
            true_wasserstein_distance = precision_wasserstein_loss(ground_truth_sigmas, ground_truth_W, est_noise, est_W_clipped)
            sample_wasserstein_loss = precision_wasserstein_sample_loss(z_prec, est_noise, est_W_clipped)
        else:
            true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0

        true_KL_divergence = precision_kl_loss(ground_truth_sigmas, ground_truth_W, est_noise, est_W_clipped)
        sample_kl_divergence = precision_kl_sample_loss(z_prec, est_noise, est_W_clipped)

        G_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(onp.array(est_W_clipped))).cpdag()
        gt_graph_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(onp.array(gt_graph_clipped))).cpdag()
        
        stats["shd_c"] = gt_graph_cpdag.shd(G_cpdag)
        # stats["true_kl"] = true_KL_divergence
        stats["sample_kl"] = sample_kl_divergence
        # stats["true_wasserstein"] = true_wasserstein_distance
        stats["sample_wasserstein"] = sample_wasserstein_loss
        stats["MSE"] = np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
        return stats

    stats = sample_stats(batched_W[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(tqdm(batched_W[1:], leave=False)):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(batched_W, ground_truth_W, edge_threshold)
    out_stats["auprc_w"] = np.array(auprcs_w).mean()
    out_stats["auprc_g"] = np.array(auprcs_g).mean()
    return out_stats

def get_cross_correlation(pred_latent, true_latent):
    dim= pred_latent.shape[1]
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (np.cov( pred_latent[:,i], true_latent[:,j] )[0,1]) / ( np.std(pred_latent[:,i])*np.std(true_latent[:,j]) )
    
    cost= -1*np.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*np.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
    return score