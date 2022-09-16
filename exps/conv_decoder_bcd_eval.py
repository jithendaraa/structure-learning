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
from modules.divergences import *
from dag_utils import count_accuracy
from conv_decoder_bcd_exps.conv_decoder_bcd_utils import auroc
from jax import jit
from sklearn.metrics import roc_curve, auc
from scipy.optimize import linear_sum_assignment

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
    if writer:
        gt_graph_image = onp.asarray(imageio.imread(join(logdir, 'gt_w.png')))
        writer.add_image('graph_structure(GT-pred)/Ground truth W', gt_graph_image, 0, dataformats='HWC')


def print_metrics(i, loss, mse_dict, mean_dict, opt, mcc_score):
    print()
    print(f"Step {i} | {loss}")
    print(f"Z_MSE: {mse_dict['z_mse']} | X_MSE: {mse_dict['x_mse']}")
    print(f"L MSE: {mse_dict['L_mse']}")
    print(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
    print(f"MCC: {mcc_score}")

    if opt.Z_KL is True:
        print(f"Z_KL: {np.array(mse_dict['Z_KL'])}")


def eval_mean(model_params, L_params, state, forward, data, rng_key, interv_nodes, interv_values, opt, 
            gt_L=None, ground_truth_W=None, ground_truth_sigmas=None):

    edge_threshold = 0.3
    hard = True
    dim = opt.num_nodes
    do_ev_noise = opt.do_ev_noise

    if opt.do_ev_noise: noise_dim = 1
    else: noise_dim = dim

    data = data[:opt.obs_data]

    res, _ = forward.apply(model_params, 
                            state,
                            rng_key,
                            rng_key,
                            interv_nodes,
                            interv_values,
                            L_params)

    ( batch_P, batch_P_logits, batch_L, batch_log_noises,
        batch_W, z_samples, full_l_batch, full_log_prob_l, 
        X_recons ) = res
        
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
        stats["true_kl"] = true_KL_divergence
        stats["sample_kl"] = sample_kl_divergence
        stats["true_wasserstein"] = true_wasserstein_distance
        stats["sample_wasserstein"] = sample_wasserstein_loss
        stats["MSE"] = np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
        return stats

    stats = sample_stats(batch_W[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    
    for i, W in enumerate(batch_W[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]
    
    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(batch_W, ground_truth_W, edge_threshold)
    out_stats["auprc_w"] = np.array(auprcs_w).mean()
    out_stats["auprc_g"] = np.array(auprcs_g).mean()
    return out_stats


def gan_eval_mean(gen_params, L_params, gen, data, rng_key, interv_nodes, 
                interv_values, opt, ground_truth_W=None, ground_truth_sigmas=None):

    edge_threshold = 0.3
    hard = True
    dim = opt.num_nodes
    do_ev_noise = opt.do_ev_noise

    if opt.do_ev_noise: noise_dim = 1
    else: noise_dim = dim

    data = data[:opt.obs_data]

    res = gen.apply(gen_params, 
                rng_key, 
                rng_key, 
                interv_nodes, 
                interv_values, 
                L_params)

    ( batch_P, batch_P_logits, batch_L, batch_log_noises,
        batch_W, z_samples, full_l_batch, full_log_prob_l, 
        X_recons ) = res
        
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

    stats = sample_stats(batch_W[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(batch_W[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(batch_W, ground_truth_W, edge_threshold)
    out_stats["auprc_w"] = np.array(auprcs_w).mean()
    out_stats["auprc_g"] = np.array(auprcs_g).mean()
    return out_stats


def get_cross_correlation(pred_latent, true_latent):
    dim= pred_latent.shape[1]
    cross_corr= onp.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (onp.cov( pred_latent[:,i], true_latent[:,j] )[0,1]) / ( onp.std(pred_latent[:,i])*onp.std(true_latent[:,j]) )
    
    cost= -1*onp.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*onp.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
    return score