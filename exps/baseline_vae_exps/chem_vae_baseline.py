import pathlib, wandb, sys, pdb, optax
from os.path import join
import ruamel.yaml as yaml
from tqdm import tqdm
import haiku as hk

sys.path.append("..")
sys.path.append("../..")
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

sys.path.append("../conv_decoder_bcd_exps")
sys.path.append("../../models")
sys.path.append("../../modules")

import envs, utils, datagen
from vae_utils import *
from loss_fns import *
from conv_decoder_bcd_utils import *

import jax
import jax.random as rnd
import numpy as onp
from jax import numpy as jnp
from jax import config, jit, lax, value_and_grad, vmap


# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
n = opt.num_samples
d = opt.num_nodes
degree = opt.exp_edges
l_dim = d * (d - 1) // 2
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)

if opt.do_ev_noise: noise_dim = 1
else:   noise_dim = d

low = -8.
high = 8.
hard = True
num_bethe_iters = 20
assert opt.train_loss == 'mse'
bs = opt.batches
num_batches = n // bs + (not (n % bs == 0))
assert opt.dataset == 'chemdata'
proj_dims = (1, 50, 50)
log_stds_max=10.
logdir = utils.set_tb_logdir(opt)
edge_threshold = 0.3

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high, baseroot=opt.baseroot)
_, h, w, c = images.shape
L_mse = jnp.mean(gt_W ** 2)
binary_gt_W = onp.where(onp.abs(gt_W) >= 0.3, 1.0, 0.0)
shd = onp.sum(binary_gt_W)

flat_images = images[:, :, :, 0:1].reshape(-1, h*w)
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(gt_W))
log_gt_graph(gt_W, logdir, vars(opt), opt)

# ? Set parameter for Horseshoe prior on L
if ((d - 1) - 2 * degree) == 0:
    p_n_over_n = 2 * degree / (d - 1)
    if p_n_over_n > 1:
        p_n_over_n = 1
    horseshoe_tau = p_n_over_n * jnp.sqrt(jnp.log(1.0 / p_n_over_n))
else:
    horseshoe_tau = (1 / onp.sqrt(n)) * (2 * degree / ((d - 1) - 2 * degree) )
if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * d)

p_z_mu = jnp.zeros((d))
p_z_covar = jnp.eye(d)


(forward, 
model_params, 
model_opt_params, 
opt_model) = init_vae_params(opt, h, w, key, rng_key, 
                            flat_images[:bs])

def get_elbo(model_params, rng_key, x_data):
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, h, w, rng_key, x_data, opt.corr)
    mse_loss = jnp.mean(get_mse(x_data, X_recons))
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    kl_loss = jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
    return (1e-5 * jnp.mean(mse_loss + kl_loss))

@jit
def gradient_step(model_params, x_data, z_data):
    loss, grads = value_and_grad(get_elbo, argnums=(0))(model_params, rng_key, x_data)
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, h, w, rng_key, x_data, opt.corr)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    true_obs_KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_obs_joint_covar, p_z_obs_joint_mu, q_z_covars, q_z_mus, opt)
    log_dict = {
        "z_mse": jnp.mean(get_mse(z_pred, z_data)),
        "x_mse": jnp.mean(get_mse(x_data, X_recons)),
        "L_MSE": L_mse,
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
    }
    return X_recons, loss, grads, z_pred, log_dict

def train_batch(model_opt_params, model_params, x_data, z_data):
    X_recons, loss, grads, z_pred, log_dict = gradient_step(model_params, x_data, z_data)
    model_updates, model_opt_params = opt_model.update(grads, model_opt_params, model_params)
    model_params = optax.apply_updates(model_params, model_updates)
    log_dict['Evaluations/MCC'] = get_cross_correlation(onp.array(z_pred), onp.array(z_data))
    log_dict['Evaluations/AUROC'] = get_vae_auroc(d, gt_W)
    return X_recons, loss, z_pred, model_opt_params, model_params, log_dict

with tqdm(range(opt.num_steps)) as pbar:  
    for i in pbar:        
        pred_z = None
        epoch_dict = {}
        epoch_loss = 0.0

        for b in range(num_batches):
            start_idx = b * bs
            end_idx = min(n, (b+1) * bs)
            x_data, z_data = flat_images[start_idx:end_idx], z[start_idx:end_idx]
            (X_recons, loss, z_pred, model_opt_params, 
            model_params, log_dict) = train_batch(model_opt_params, model_params, x_data, z_data)
            epoch_loss += loss

            if b == 0:  epoch_dict = log_dict
            else:       
                for key, val in log_dict.items():
                    epoch_dict[key] += val

        elbo = epoch_loss / num_batches
        for key in epoch_dict:
            epoch_dict[key] = epoch_dict[key] / num_batches 

        if i % 20 == 0:
            wandb_dict = {
                'ELBO': elbo,
                "Z_MSE": epoch_dict["z_mse"],
                "X_MSE": epoch_dict["x_mse"],
                "L_MSE": epoch_dict["L_MSE"],
                "true_obs_KL_term_Z": epoch_dict["true_obs_KL_term_Z"],
                "Evaluations/SHD": shd,
                'Evaluations/MCC': epoch_dict['Evaluations/MCC'],
                'Evaluations/AUROC': epoch_dict['Evaluations/AUROC'],
            }
            
            plt.figure()
            plt.imshow(X_recons[0, :].reshape(h, w)/255.)
            plt.savefig(join(logdir, f'pred_image_vae_chem{opt.learn_P}_seed{opt.data_seed}_d{d}_ee_{int(opt.exp_edges)}_sets{opt.n_interv_sets}_pts{opt.pts_per_interv}.png'))
            wandb_dict["graph_structure(GT-pred)/Reconstructed image"] = wandb.Image(join(logdir, f'pred_image_vae_chem{opt.learn_P}_seed{opt.data_seed}_d{d}_ee_{int(opt.exp_edges)}_sets{opt.n_interv_sets}_pts{opt.pts_per_interv}.png'))
            plt.close('all')

            if opt.off_wandb is False:  
                wandb.log(wandb_dict, step=i)

        pbar.set_postfix(
            ELBO=f"{elbo:.4f}",
            SHD=jnp.sum(binary_gt_W),
            MCC=epoch_dict['Evaluations/MCC'], 
            L_mse=epoch_dict["L_MSE"],
            AUROC=epoch_dict['Evaluations/AUROC'],
            KL_Z=epoch_dict["true_obs_KL_term_Z"]
        )

#         