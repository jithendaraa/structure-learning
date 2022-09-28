import pathlib, wandb, sys, pdb, optax
from os.path import join
import ruamel.yaml as yaml
from tqdm import tqdm
import haiku as hk

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../models")
sys.path.append("../../modules")

import utils, datagen
from dag_utils import SyntheticDataset
from models.VAE import VAE
from vae_utils import *
from loss_fns import *

import jax
import jax.random as rnd
import numpy as onp
from jax import numpy as jnp
from jax import config, jit, lax, value_and_grad, vmap

configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)
exp_config = vars(opt)

assert opt.use_proxy is False

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

logdir = utils.set_tb_logdir(opt)

opt.num_samples = opt.obs_data + int(opt.n_interv_sets * opt.pts_per_interv)
n = opt.num_samples
d = opt.num_nodes
degree = opt.exp_edges
n_interv_sets = opt.n_interv_sets
edge_threshold = 0.3
corr = opt.corr

sd = SyntheticDataset(
    n=n,
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
ground_truth_sigmas = opt.noise_sigma * jnp.ones(d)
L_mse = jnp.mean(ground_truth_W ** 2)
gt_L_elems = get_lower_elems(ground_truth_L, d)
binary_gt_W = jnp.where(jnp.abs(sd.W) > edge_threshold, 1.0, 0.0)

print(ground_truth_W)
print()
print(binary_gt_W)
print()

(z_gt, no_interv_targets, x, proj_matrix, interv_values) = datagen.get_data(opt, n_interv_sets, sd, model="bcd", interv_value=opt.interv_value)
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(sd.W))
log_gt_graph(ground_truth_W, logdir, exp_config, opt)

max_cols = jnp.max(no_interv_targets.sum(1))
data_idx_array = jnp.array([jnp.arange(opt.num_nodes + 1)] * opt.num_samples)
interv_nodes = onp.split(data_idx_array[no_interv_targets], no_interv_targets.sum(1).cumsum()[:-1])
interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([opt.num_nodes] * (max_cols - len(interv_nodes[i])))))
        for i in range(opt.num_samples)]).astype(int)

p_z_mu = jnp.zeros((d))
p_z_covar = jnp.eye(d)

forward, model_params, model_opt_params, opt_model = numerical_init_vae_params(opt, opt.proj_dims, key, rng_key, x)

def get_elbo(model_params, rng_key, x_data):
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    mse_loss = jnp.mean(get_mse(x_data[:, :opt.proj_dims], X_recons))
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    kl_loss = jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
    return (1e-5 * jnp.mean(mse_loss + kl_loss))

@jit
def gradient_step(model_params, x_data):
    loss, grads = value_and_grad(get_elbo, argnums=(0))(model_params, rng_key, x_data)
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    true_obs_KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_obs_joint_covar, p_z_obs_joint_mu, q_z_covars, q_z_mus, opt)

    log_dict = {
        "z_mse": jnp.mean(get_mse(z_pred, z_gt)),
        "x_mse": jnp.mean(get_mse(x_data[:, :opt.proj_dims], X_recons)),
        "L_MSE": L_mse,
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z)
    }
    return X_recons, loss, grads, z_pred, log_dict


with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        X_recons, loss, grads, z_pred, log_dict = gradient_step(model_params, x)
        
        mcc = get_cross_correlation(onp.array(z_pred), onp.array(z_gt))
        auroc = get_vae_auroc(d, ground_truth_W)

        wandb_dict = {
            "ELBO": onp.array(loss),
            "Z_MSE": onp.array(log_dict["z_mse"]),
            "X_MSE": onp.array(log_dict["x_mse"]),
            "L_MSE": L_mse,
            "true_obs_KL_term_Z": onp.array(log_dict["true_obs_KL_term_Z"]),
            "Evaluations/SHD": jnp.sum(binary_gt_W),
            "Evaluations/AUROC": auroc,
            'Evaluations/MCC': mcc,
        }
        
        if (i+1) % 50 == 0 or i == 0:       
            if opt.off_wandb is False:  
                wandb.log(wandb_dict, step=i)

        pbar.set_postfix(
            ELBO=f"{loss:.4f}",
            SHD=jnp.sum(binary_gt_W),
            MCC=mcc, 
            L_mse=f"{log_dict['L_MSE']:.3f}",
            AUROC=f"{auroc:.2f}",
            KL_Z=f"{onp.array(log_dict['true_obs_KL_term_Z']):.4f}"
        )

        model_updates, model_opt_params = opt_model.update(grads, model_opt_params, model_params)
        model_params = optax.apply_updates(model_params, model_updates)