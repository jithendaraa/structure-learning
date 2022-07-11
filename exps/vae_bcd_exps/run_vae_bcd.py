import sys
import pathlib
import ruamel.yaml as yaml
from jax import numpy as jnp
import pdb
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../modules")
sys.path.append("../../models")

import numpy as onp
import haiku as hk
from jax import random, value_and_grad, jit, vmap, grad, lax
import utils, datagen
from dag_utils import SyntheticDataset
from vae_bcd_utils import *
from loss_fns import *
from divergences import *
from vae_bcd_eval import *
from modules.GumbelSinkhorn import GumbelSinkhorn
from torch.utils.tensorboard import SummaryWriter
from tensorflow_probability.substrates.jax.distributions import (Horseshoe,
                                                                 Normal)

configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)

assert opt.use_proxy is False
assert opt.train_loss == 'mse'
if opt.l2_reg or opt.reg_decoder or opt.use_proxy: raise NotImplementedError


rng_key = random.PRNGKey(opt.data_seed)
hk_key = hk.PRNGSequence(opt.data_seed)

dim = opt.num_nodes
l_dim = dim * (dim - 1) // 2
if opt.do_ev_noise: noise_dim = 1
else: noise_dim = dim
degree = opt.exp_edges
num_outer = 1
num_devices = 1
tau = opt.fixed_tau
shd = -1.0


logdir = utils.set_tb_logdir(opt)
writer = SummaryWriter(join("..", logdir))

horseshoe_tau = (1 / onp.sqrt(opt.num_samples)) * (2 * degree / ((dim - 1) - 2 * degree))
if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * dim)
print(f"Horseshoe tau is {horseshoe_tau}")


sd = SyntheticDataset(
    n=opt.num_samples,
    d=opt.num_nodes,
    graph_type="erdos-renyi",
    degree=2 * opt.exp_edges,
    sem_type=opt.sem_type,
    dataset_type="linear",
    noise_scale=opt.noise_sigma,
    data_seed=opt.data_seed,
)

z_gt, no_interv_targets, x, proj_matrix, interv_values = datagen.get_data(
    opt, 
    opt.n_interv_sets, 
    sd, 
    model="bcd", 
    interv_value=opt.interv_value
)

ground_truth_W, ground_truth_P = sd.W, sd.P
ground_truth_L = sd.P.T @ sd.W.T @ sd.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

log_gt_graph(ground_truth_W, logdir, vars(opt), opt, writer)

ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=opt.max_deviation)
gt_L_elems = get_lower_elems(ground_truth_L, dim)

max_cols = jnp.max(no_interv_targets.sum(1))
data_idx_array = jnp.array([jnp.arange(opt.num_nodes + 1)] * opt.num_samples)
interv_nodes = onp.split(data_idx_array[no_interv_targets], no_interv_targets.sum(1).cumsum()[:-1])
interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([opt.num_nodes] * (max_cols - len(interv_nodes[i])))))
        for i in range(opt.num_samples)]).astype(int)

rng_keys = random.split(rng_key, 1)
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(sd.W))


# Initialize model and params
forward, model_params, model_opt_params, opt_model, rng_keys = init_vae_bcd_params(opt, hk_key, False, rng_key, interv_nodes, interv_values, num_devices)


def hard_elbo(model_params, rng_key, x_input, interv_nodes, interv_values):
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)  # * Horseshoe prior over lower triangular matrix L
    rng_key = random.split(rng_key, num_outer)[0]

    obs_KL_term_Z = 0.

    (   batch_P, 
        batch_P_logits, 
        batch_L, 
        batch_log_noises, 
        batch_W, 
        batch_qz_samples, 
        full_l_batch, 
        full_log_prob_l, 
        X_recons 
    ) = forward.apply(model_params, rng_key, opt, True, rng_key, x_input, interv_nodes, interv_values)

    likelihoods = -jit(vmap(get_mse, (None, 0), 0))(x, X_recons) 

    final_term = likelihoods

    if opt.P_KL is True:    # ! KL over permutation P
        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits, num_bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
        KL_term_P = logprob_P - log_P_prior
        final_term -= KL_term_P

    if opt.L_KL is True:    # ! KL over edge weights L
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
        KL_term_L = full_log_prob_l - l_prior_probs
        final_term -= KL_term_L

    if (opt.obs_Z_KL is True):
        batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
        batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
        vmapped_kl = vmap(get_single_kl, (None, None, 0, 0, None), (0))
        obs_KL_term_Z = vmapped_kl(p_z_obs_joint_covar, p_z_obs_joint_mu, batch_q_z_obs_joint_covars, batch_q_z_obs_joint_mus, opt) 
        
    final_term -= obs_KL_term_Z
    elbo = -jnp.mean(final_term)
    return elbo


def gradient_step(model_params, rng_key, x_input, interv_nodes, interv_values):
    loss, grads = value_and_grad(hard_elbo, argnums=(0))(model_params, rng_key, x_input, interv_nodes, interv_values)
    rng_key_ = random.split(rng_key, num_outer)[0]

    (   batch_P, 
        batch_P_logits, 
        batch_L, 
        batch_log_noises, 
        batch_W, 
        batch_qz_samples, 
        full_l_batch, 
        full_log_prob_l, 
        X_recons 
    ) = forward.apply(model_params, rng_key_, opt, True, rng_key, x_input, interv_nodes, interv_values)


    batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
    vmapped_get_kl = vmap(get_single_kl, (None, None, 0, 0, None), (0))
    true_obs_KL_term_Z = vmapped_get_kl(p_z_obs_joint_covar, 
                                        p_z_obs_joint_mu, 
                                        batch_q_z_obs_joint_covars, 
                                        batch_q_z_obs_joint_mus, 
                                        opt) 

    L_elems = vmap(get_lower_elems, (0, None), 0)(batch_L, dim)

    log_dict = {
        "L_mse": get_mse(gt_L_elems, L_elems),
        "z_mse": get_mse(z_gt, batch_qz_samples),
        "x_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(x_input, X_recons)),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z)
    }

    pred_W_means = jnp.mean(batch_W, axis=0)
    return (loss, rng_key, grads, log_dict, batch_W, batch_qz_samples, X_recons, pred_W_means)


with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
    
        (   loss, 
            rng_key, 
            grads, 
            log_dict, 
            batch_W, 
            batch_qz_samples, 
            X_recons,
            pred_W_means
        ) = gradient_step(model_params, rng_key, x, interv_nodes, interv_values)
        
        updates, model_opt_params = opt_model.update(grads, model_opt_params, model_params)
        model_params = optax.apply_updates(model_params, updates)
        
        if i==0 or (i+1) % 100 == 0:        
            mean_dict = eval_mean(  model_params, 
                                    x, 
                                    z_gt, 
                                    rk(i), 
                                    interv_values, 
                                    True, 
                                    i, 
                                    interv_nodes, 
                                    forward, 
                                    ground_truth_L, 
                                    sd.W, 
                                    ground_truth_sigmas, 
                                    opt
                                )

            wandb_dict = {
                "ELBO": onp.array(loss),
                "Z_MSE": onp.array(log_dict["z_mse"]),
                "X_MSE": onp.array(log_dict["x_mse"]),
                "L_MSE": onp.array(log_dict["L_mse"]),
                "true_obs_KL_term_Z": onp.array(log_dict["true_obs_KL_term_Z"]),
                "Evaluations/SHD": mean_dict["shd"],
                "Evaluations/SHD_C": mean_dict["shd_c"],
                "Evaluations/AUROC": mean_dict["auroc"],
                "Evaluations/AUPRC_W": mean_dict["auprc_w"],
                "Evaluations/AUPRC_G": mean_dict["auprc_g"],
                "train sample KL": mean_dict["sample_kl"],
            }
            
            if opt.off_wandb is False:  
                plt.imshow(pred_W_means)
                plt.savefig(join(logdir, 'pred_w.png'))
                wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
                wandb.log(wandb_dict, step=i)

            shd = mean_dict["shd"]
            tqdm.write(f"Step {i} | {loss}")
            tqdm.write(f"Z_MSE: {log_dict['z_mse']} | X_MSE: {log_dict['x_mse']}")
            tqdm.write(f"L MSE: {log_dict['L_mse']}")
            tqdm.write(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
            tqdm.write(f"KL(learned || true): {onp.array(log_dict['true_obs_KL_term_Z'])}")
            tqdm.write(f" ")

        pbar.set_postfix(
            loss=f"{loss:.4f}",
            X_mse=f"{log_dict['x_mse']:.4f}",
            KL=f"{log_dict['true_obs_KL_term_Z']:.4f}", 
            L_mse=f"{log_dict['L_mse']:.3f}",
            SHD=shd
        )
    