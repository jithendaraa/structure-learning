import pathlib
import pdb
import sys
import time
from os.path import join

import graphical_models
import imageio
import optax
from tqdm import tqdm

import wandb

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../modules")
sys.path.append("../../models")

from typing import Optional, Tuple, Union, cast

import datagen
import haiku as hk
import jax
import jax.random as rnd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as onp
import ruamel.yaml as yaml
import utils
from jax import config, grad, jit, lax
from jax import numpy as jnp
from jax import pmap, value_and_grad, vmap
from jax.ops import index, index_mul, index_update
from jax.tree_util import tree_map, tree_multimap
from PIL import Image

config.update("jax_enable_x64", True)

from dag_utils import SyntheticDataset, count_accuracy
# Data generation procedure
from dibs_new.dibs.target import make_linear_gaussian_model
from divergences import *
from eval_ import *
from haiku._src import data_structures
from loss_fns import *
from loss_fns import get_single_kl
from models.Decoder_BCD import Decoder_BCD
from modules.GumbelSinkhorn import GumbelSinkhorn
from modules.hungarian_callback import batched_hungarian, hungarian
from tensorflow_probability.substrates.jax.distributions import (Horseshoe,
                                                                 Normal)
from torch.utils.tensorboard import SummaryWriter
from bcd_utils import *

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")
# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

# ? Set logdir
logdir = utils.set_tb_logdir(opt)
writer = SummaryWriter(join("..", logdir))

# ? Defining type variables
LStateType = Optional[hk.State]
PParamType = Union[hk.Params, jnp.ndarray]
PRNGKey = jnp.ndarray
L_dist = Normal

# ? Variables
dim = opt.num_nodes
n_data = opt.num_samples
degree = opt.exp_edges
do_ev_noise = opt.do_ev_noise
num_outer = opt.num_outer
s_prior_std = opt.s_prior_std
n_interv_sets = 20
calc_shd_c = False
sem_type = opt.sem_type
eval_eid = opt.eval_eid
num_bethe_iters = 20


if do_ev_noise:
    noise_dim = 1
else:
    noise_dim = dim
l_dim = dim * (dim - 1) // 2
input_dim = l_dim + noise_dim
log_stds_max: Optional[float] = 10.0
tau = opt.fixed_tau
tau_scaling_factor = 1.0 / tau


# noise sigma usually around 0.1 but in original BCD nets code it is set to 1
sd = SyntheticDataset(
    n=n_data,
    d=opt.num_nodes,
    graph_type="erdos-renyi",
    degree=2 * degree,
    sem_type=opt.sem_type,
    dataset_type="linear",
    noise_scale=opt.noise_sigma,
    data_seed=opt.data_seed,
)
ground_truth_W = sd.W
ground_truth_P = sd.P
ground_truth_L = sd.P.T @ sd.W.T @ sd.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

(obs_data, interv_data, z_gt, no_interv_targets, x, 
sample_mean, sample_covariance, proj_matrix, interv_values) = datagen.get_data(opt, n_interv_sets, sd, model="bcd", interv_value=opt.interv_value)

node_mus = jnp.ones((dim)) * opt.noise_mu
node_vars = jnp.ones((dim)) * (opt.noise_sigma**2)
interv_types = onp.eye(dim, dtype=onp.bool)

# ! Get GT observational joint distributions
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(sd.W))

# ? Set parameter for Horseshoe prior on L
horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((dim - 1) - 2 * degree))
if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * dim)
print(f"Horseshoe tau is {horseshoe_tau}")

log_gt_graph(ground_truth_W, logdir, exp_config, opt, writer)

hard = True
max_cols = jnp.max(no_interv_targets.sum(1))
data_idx_array = jnp.array([jnp.arange(opt.num_nodes + 1)] * opt.num_samples)
interv_nodes = onp.split(data_idx_array[no_interv_targets], no_interv_targets.sum(1).cumsum()[:-1])
interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([opt.num_nodes] * (max_cols - len(interv_nodes[i])))))
        for i in range(opt.num_samples)]).astype(int)

ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=opt.max_deviation)
gt_L_elems = get_lower_elems(ground_truth_L, dim)

num_interv_data = int(opt.num_samples - opt.obs_data)
interv_data_per_set = int(num_interv_data / n_interv_sets)
print()

assert opt.num_samples > opt.obs_data


def hard_elbo(P_params, L_params, proj_params, rng_key, interv_nodes, x_data, interv_values_):
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)  # * Horseshoe prior over lower triangular matrix L
    rng_key = rnd.split(rng_key, num_outer)[0]

    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""

        (   batch_P,
            batch_P_logits,
            batch_L,
            batch_log_noises,
            batch_W,
            batch_qz_samples,
            full_l_batch,
            full_log_prob_l,
            X_recons, 
        ) = forward.apply(P_params, rng_key, hard, rng_key, interv_nodes, False, opt, horseshoe_tau, 
                proj_matrix, ground_truth_L, interv_values_, P_params, L_params, proj_params)

        if opt.train_loss == "mse":     # ! MSE Loss
            likelihoods = -jit(vmap(get_mse, (None, 0), 0))(x_data, X_recons)
            z_mse = 0.0
            
            if opt.z_mse and opt.z_mse is True:
                z_mse = 10 * jit(vmap(get_mse, (None, 0), 0))(z_gt, batch_qz_samples)
        
        elif opt.train_loss == "LL":    # ! NLL loss
            vectorized_log_prob_X = vmap(log_prob_X, in_axes=(None, 0, 0, 0, None, None, None, None, None))
            likelihoods = vectorized_log_prob_X(x, batch_log_noises, batch_P, batch_L, decoder_params, 
                                                proj_matrix, opt.fix_decoder, opt.cov_space, opt.s_prior_std)

        final_term = likelihoods - z_mse

        # ! KL over P
        if opt.P_KL is True:
            logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits, num_bethe_iters)
            log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
            KL_term_P = logprob_P - log_P_prior
            final_term -= KL_term_P

        # ! KL over L
        if opt.L_KL is True:
            l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
            KL_term_L = full_log_prob_l - l_prior_probs
            final_term -= KL_term_L

        if opt.obs_Z_KL is True and opt.Z_KL == 'joint':
            batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
            batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
            vmapped_kl = vmap(get_single_kl, (None, None, 0, 0, None), (0))
            obs_KL_term_Z = vmapped_kl(p_z_obs_joint_covar, p_z_obs_joint_mu, 
                                        batch_q_z_obs_joint_covars, batch_q_z_obs_joint_mus, opt) 
            
            final_term -= obs_KL_term_Z

        return -jnp.mean(final_term)
    
    _, elbos = lax.scan(lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys)
    return jnp.mean(elbos)


@jit
def gradient_step(P_params, L_params, proj_params, rng_key, interv_nodes, z_gt_data, x_data, interv_values_):
    loss, grads = value_and_grad(hard_elbo, argnums=(0, 1, 2))(P_params, L_params, proj_params, rng_key, interv_nodes, x_data, interv_values_)
    elbo_grad_P, elbo_grad_L, elbo_grad_decoder = tree_map(lambda x_: x_, grads)

    # * L2 regularization over parameters of P (contains p network and decoder)
    if opt.l2_reg is True:
        l2_elbo_grad_P = grad(lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p)))(P_params)
        elbo_grad_P = tree_multimap(lambda x_, y: x_ + y, elbo_grad_P, l2_elbo_grad_P)

    if opt.reg_decoder is True and opt.decoder_layers == 'linear' and opt.learn_P is False:
        l1_elbo_grad_P = grad(lambda p: sum(jnp.sum(jnp.abs(param)) for param in jax.tree_leaves(p)))(P_params)
        elbo_grad_P = tree_multimap(lambda x_, y: x_ + y, elbo_grad_P, l1_elbo_grad_P)

    rng_key_ = rnd.split(rng_key, num_outer)[0]

    (   batch_P,
        batch_P_logits,
        batch_L,
        batch_log_noises,
        batch_W,
        z_samples,
        full_l_batch,
        full_log_prob_l,
        X_recons, 
    ) = forward.apply(P_params, rng_key, hard, rng_key, interv_nodes, False, opt, horseshoe_tau, 
            proj_matrix, ground_truth_L, interv_values_, P_params, L_params, proj_params)

    L_elems = vmap(get_lower_elems, (0, None), 0)(batch_L, dim)
    batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
    true_obs_KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), (0))(p_z_obs_joint_covar, p_z_obs_joint_mu, 
                                                                        batch_q_z_obs_joint_covars, batch_q_z_obs_joint_mus, opt) 

    log_dict = {
        "L_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(gt_L_elems, L_elems)),
        "z_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(z_gt_data, z_samples)),
        "x_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(x_data, X_recons)),
        "L_elems": jnp.mean(L_elems[:, -1]),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z)
    }
    
    return (loss, rng_key, elbo_grad_P, elbo_grad_L, elbo_grad_decoder, log_dict, batch_W, z_samples, X_recons)



for i in range(n_interv_sets + 1):
    x_data = x[ : opt.obs_data + int(i * interv_data_per_set) ]
    z_gt_data = z_gt[ : opt.obs_data + int(i * interv_data_per_set) ]

    interv_targets_ = interv_nodes[ : opt.obs_data + int(i * interv_data_per_set) ]
    interv_values_ = interv_values[ : opt.obs_data + int(i * interv_data_per_set) ]

    ( P_params, L_params, decoder_params, P_opt_params, 
        L_opt_params, decoder_opt_params, rng_keys, 
        forward, opt_P, opt_L, opt_decoder) = init_parallel_params(rng_key, key, opt, num_devices, interv_targets_, 
                                                horseshoe_tau, proj_matrix, ground_truth_L, interv_values_)
    decoder_sparsity_mask = jnp.where(decoder_params != 0., 1.0, 0.0)

    for step in tqdm(range(opt.num_steps)):
        (loss, rng_key, elbo_grad_P, elbo_grad_L, elbo_grad_decoder, 
        mse_dict, batch_W, z_samples, X_recons) = gradient_step(P_params, L_params, decoder_params, rng_key, interv_targets_, z_gt_data, x_data, interv_values_)
        
        # * Update P and decoder
        P_updates, P_opt_params = opt_P.update(elbo_grad_P, P_opt_params, P_params)
        P_params = optax.apply_updates(P_params, P_updates)

        # * Update L network
        L_updates, L_opt_params = opt_L.update(elbo_grad_L, L_opt_params, L_params)
        L_params = optax.apply_updates(L_params, L_updates)

        # * update decoder
        elbo_grad_decoder = jnp.multiply(elbo_grad_decoder, decoder_sparsity_mask)
        decoder_updates, decoder_opt_params = opt_decoder.update(elbo_grad_decoder, decoder_opt_params, decoder_params)
        decoder_params = optax.apply_updates(decoder_params, decoder_updates)

        if jnp.any(jnp.isnan(ravel_pytree(L_params)[0])):   raise Exception("Got NaNs in L params")

    print(f"Trained on {opt.obs_data} observational data and {int(i * interv_data_per_set)} interventional data")

    mean_dict = eval_mean(P_params, L_params, decoder_params, z_gt_data, rk(step), True, tau, step, 
                interv_targets_, forward, horseshoe_tau, proj_matrix, ground_truth_L, 
                sd.W, ground_truth_sigmas, opt)

    wandb_dict = {
        "ELBO": onp.array(loss),
        "Z_MSE": onp.array(mse_dict["z_mse"]),
        "X_MSE": onp.array(mse_dict["x_mse"]),
        "L_MSE": onp.array(mse_dict["L_mse"]),
        "L_elems_avg": onp.array(mse_dict["L_elems"]),
        "true_obs_KL_term_Z": onp.array(mse_dict["true_obs_KL_term_Z"]),
        "Evaluations/SHD": mean_dict["shd"],
        "Evaluations/SHD_C": mean_dict["shd_c"],
        "Evaluations/AUROC": mean_dict["auroc"],
        "train sample KL": mean_dict["sample_kl"],
    }

    if opt.interv_Z_KL is True and opt.Z_KL == 'joint':    wandb_dict['interv_KL_term_Z'] = onp.array(mse_dict['interv_KL_term_Z'])
    x_axis = int(i * interv_data_per_set)
    print_metrics(x_axis, loss, mse_dict, mean_dict, opt)

    pred_W_means, pred_W_stds = lower(L_params[:l_dim], dim), lower(jnp.exp(L_params[l_dim:]), dim)
    print("pred_W_means: ", pred_W_means)
    print("pred_W_stds: ", pred_W_stds)

    if opt.off_wandb is False:  
        plt.imshow(pred_W_means)
        plt.savefig(join(logdir, 'pred_w.png'))
        wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
        wandb.log(wandb_dict, step=x_axis)