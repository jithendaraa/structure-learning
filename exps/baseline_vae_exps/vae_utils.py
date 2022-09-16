import os, wandb, imageio
from os.path import join
import numpy as onp
from jax import numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, auc
from jax import jit


def get_joint_dist_params(sigma, W):
    """
        Gets the joint distribution for some SCM that performs: 
        z = W.T @ z + eps where eps ~ Normal(0, sigma**2*I)
    """
    dim, _ = W.shape
    Sigma = sigma**2 * jnp.eye(dim)
    inv_matrix = jnp.linalg.inv((jnp.eye(dim) - W))
    mu_joint = jnp.array([0.] * dim)
    Sigma_joint = inv_matrix.T @ Sigma @ inv_matrix
    return mu_joint, Sigma_joint


def log_gt_graph(ground_truth_W, logdir, exp_config_dict, opt):
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


def get_lower_elems(L, dim, k=-1):
    return L[jnp.tril_indices(dim, k=k)]


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])


def auroc(pred_Ws, gt_W, threshold):
    
    
    
    return auroc


def get_auroc(d, gt_W, threshold=0.3):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    
    pred_Ws = jnp.zeros((1, d, d))
    _, dim, dim = pred_Ws.shape
    edge_present = jnp.abs(pred_Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(gt_W) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc


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

@jit
def get_covar(L):
    return L @ L.T