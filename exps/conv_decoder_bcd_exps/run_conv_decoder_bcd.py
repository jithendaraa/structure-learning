import sys, pathlib, pdb, os
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

from tqdm import tqdm
import utils
import ruamel.yaml as yaml

from conv_decoder_bcd_utils import *
import envs, wandb
import jax
from jax import numpy as jnp
import numpy as onp
import jax.random as rnd
from jax import config, jit, lax, value_and_grad, vmap
from jax.tree_util import tree_map, tree_multimap
import haiku as hk
from modules.GumbelSinkhorn import GumbelSinkhorn

from loss_fns import *
from tensorflow_probability.substrates.jax.distributions import (Horseshoe,
                                                                 Normal)
from conv_decoder_bcd_eval import *

# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
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

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high)
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

ds = GumbelSinkhorn(d, noise_type="gumbel", tol=opt.max_deviation)
gt_L_elems = get_lower_elems(gt_L, d)
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(gt_W))

def f(rng_key, interv_targets, interv_values, L_params):
    model = Conv_Decoder_BCD(   
                                d,
                                l_dim, 
                                noise_dim, 
                                opt.batch_size,
                                opt.do_ev_noise,
                                opt.learn_P,
                                opt.learn_noise,
                                proj_dims,
                                opt.fixed_tau,
                                gt_P,
                                opt.max_deviation,
                                logit_constraint=opt.logit_constraint,
                                log_stds_max=log_stds_max,
                                noise_sigma=opt.noise_sigma
                            )
    return model(rng_key, interv_targets, interv_values, L_params)

f = hk.transform_with_state(f)

def init_params_and_optimizers():
    if opt.learn_noise is False:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(l_dim) - 1, ))
    else:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,))

    model_params, state = jit(f.init)(rng_key, rng_key, interv_nodes[:bs], interv_values[:bs], L_params)
    
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_layers)

    L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_L = optax.chain(*L_layers)

    model_opt_params = opt_model.init(model_params)
    L_opt_params = opt_L.init(L_params)

    return state, model_params, L_params, model_opt_params, L_opt_params, opt_model, opt_L

(state, model_params, L_params, model_opt_params, L_opt_params, opt_model, opt_L) = init_params_and_optimizers()

@jit
def get_loss(model_params, L_params, state, x_data, interv_nodes_, interv_values_):
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)

    res, state = f.apply(model_params, 
                            state,
                            rng_key,
                            rng_key,
                            interv_nodes_[:bs],
                            interv_values_[:bs],
                            L_params)
    (   batch_P, 
        batch_P_logits, 
        batch_L, 
        batch_log_noises,
        batch_W, 
        z_samples, 
        full_l_batch, 
        full_log_prob_l, 
        X_recons 
                        ) = res

    mse_loss = jit(vmap(get_mse, (None, 0), 0))(x_data/255.0, X_recons/255.0)
    likelihoods = mse_loss
    final_term = likelihoods

    if opt.P_KL is True:    # ! KL over permutation P
        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits, num_bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(onp.arange(d) + 1))
        KL_term_P = logprob_P - log_P_prior
        final_term += KL_term_P

    if opt.L_KL is True:    # ! KL over edge weights L
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
        KL_term_L = full_log_prob_l - l_prior_probs
        final_term += KL_term_L

    if (opt.obs_Z_KL is True or opt.use_proxy is True):
        batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
        batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
        vmapped_kl = vmap(get_single_kl, (None, None, 0, 0, None), (0))
        if opt.use_proxy is True: raise NotImplementedError
        else:   obs_KL_term_Z = vmapped_kl(p_z_obs_joint_covar, p_z_obs_joint_mu, batch_q_z_obs_joint_covars, batch_q_z_obs_joint_mus, opt) 
        final_term += obs_KL_term_Z

    elbos = jnp.mean(final_term)
    
    return jnp.mean(elbos), (res, state, jnp.mean(mse_loss))

@jit
def get_log_dict(res, loss, x_mse, z_data):
    
    (   batch_P, 
        batch_P_logits, 
        batch_L, 
        batch_log_noises,
        batch_W, 
        z_samples, 
        full_l_batch, 
        full_log_prob_l, 
        X_recons 
                        ) = res

    L_elems = vmap(get_lower_elems, (0, None), 0)(batch_L, d)

    batch_get_obs_joint_dist_params = vmap(get_joint_dist_params, (0, 0), (0, 0))
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(batch_log_noises), batch_W)
    true_obs_KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), (0))(p_z_obs_joint_covar, 
                                                                            p_z_obs_joint_mu, 
                                                                            batch_q_z_obs_joint_covars, 
                                                                            batch_q_z_obs_joint_mus, 
                                                                            opt) 

    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)
    l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
    KL_term_L = full_log_prob_l - l_prior_probs

    pred_W_means = jnp.mean(batch_W, axis=0)

    log_dict = {
        "ELBO": jnp.mean(loss),
        "x_mse": x_mse,
        "L_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(gt_L_elems, L_elems)),
        "z_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(z_data, z_samples)),
        "L_elems": jnp.mean(L_elems[:, -1]),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
        "KL(L)": jnp.mean(KL_term_L)
    }

    return log_dict, batch_W, z_samples, X_recons, pred_W_means

@jit
def update_params(grads, model_opt_params, model_params, L_opt_params, L_params):
    elbo_grad_model, elbo_grad_L = tree_map(lambda x_: x_, grads)
    
    model_updates, model_opt_params = opt_model.update(elbo_grad_model, model_opt_params, model_params)
    model_params = optax.apply_updates(model_params, model_updates)

    L_updates, L_opt_params = opt_L.update(elbo_grad_L, L_opt_params, L_params)
    L_params = optax.apply_updates(L_params, L_updates)

    return model_opt_params, model_params, L_opt_params, L_params


def train_batch(state, model_opt_params, model_params, L_opt_params, L_params, x_data, z_data, interv_nodes_, interv_values_):
    
    (loss, res), grads = value_and_grad(get_loss, argnums=(0,1), has_aux=True)(model_params, 
                                                                                L_params, 
                                                                                state, 
                                                                                x_data,
                                                                                interv_nodes_, 
                                                                                interv_values_)


    (   model_opt_params, 
        model_params, 
        L_opt_params, 
        L_params        ) = update_params(grads, model_opt_params, model_params, L_opt_params, L_params)

    log_dict, batch_W, z_samples, X_recons, pred_W_means = get_log_dict(res[0], loss, res[2], z_data)

    return res[1], model_opt_params, model_params, L_opt_params, L_params, log_dict, batch_W, z_samples, X_recons, pred_W_means


num_test_samples = 10
(   test_interv_data, 
    test_interv_nodes, 
    test_interv_values, 
    test_images,
    padded_test_images      ) = generate_test_samples(d, onp.array(gt_W), 
                                                opt.sem_type, 
                                                [opt.noise_sigma], 
                                                low, high, 
                                                10)


plt.figure()
plt.imshow(padded_test_images/255.)
plt.savefig(f'/home/mila/j/jithendaraa.subramanian/scratch/test_gt_image_learnP{opt.learn_P}_seed{opt.data_seed}_d{d}_ee_{int(opt.exp_edges)}.png')
plt.close('all')

# Training loop
with tqdm(range(opt.num_steps)) as pbar:  
    for i in pbar:
        pred_z = None
        pred_x = None
        pred_W = None
        epoch_dict = {}

        with tqdm(range(num_batches)) as pbar2:
            for b in pbar2:

                start_idx = b * bs
                end_idx = min(n, (b+1) * bs)
                x_data, z_data = images[start_idx:end_idx], z[start_idx:end_idx]

                (   state, 
                    model_opt_params, 
                    model_params, 
                    L_opt_params, 
                    L_params,
                    log_dict, 
                    batch_W, 
                    z_samples, 
                    X_recons, 
                    pred_W_means    ) = train_batch(state, 
                                                    model_opt_params, 
                                                    model_params, 
                                                    L_opt_params, 
                                                    L_params,
                                                    x_data,
                                                    z_data,
                                                    interv_nodes[start_idx:end_idx],
                                                    interv_values[start_idx:end_idx])
                
                if jnp.any(jnp.isnan(ravel_pytree(L_params)[0])):   raise Exception("Got NaNs in L params")

                if b == 0:
                    pred_z = z_samples
                    pred_x = X_recons
                    epoch_dict = log_dict
                    pred_W = pred_W_means[jnp.newaxis, :]
                else:
                    pred_z = jnp.concatenate((pred_z, z_samples), axis=1)
                    pred_x = jnp.concatenate((pred_x, X_recons), axis=1)
                    pred_W = jnp.concatenate((pred_W, pred_W_means[jnp.newaxis, :]), axis=0)
                    
                    for key, val in log_dict.items():
                        epoch_dict[key] += val

                pbar2.set_postfix(
                    Batch=f"{b}/{num_batches}",
                    X_mse=f"{log_dict['x_mse']:.2f}",
                    KL=f"{log_dict['true_obs_KL_term_Z']:.4f}", 
                    L_mse=f"{log_dict['L_mse']:.3f}"
                )

        for key in epoch_dict:
            epoch_dict[key] = epoch_dict[key] / num_batches 

        if i % 20 == 0:
            random_idxs = onp.random.choice(n, bs, replace=False)
            mean_dict = eval_mean(  model_params, 
                                    L_params,
                                    state,
                                    f, 
                                    z, 
                                    rk(i), 
                                    interv_nodes[random_idxs], 
                                    interv_values[random_idxs],
                                    opt, 
                                    gt_L, 
                                    gt_W, 
                                    ground_truth_sigmas, 
                                )

            mcc_scores = []
            for j in range(len(pred_z)):
                mcc_scores.append(get_cross_correlation(onp.array(pred_z[j]), onp.array(z)))
            mcc_score = onp.mean(onp.array(mcc_scores))

            wandb_dict = {
                # Different loss related metrics and evaluating SCM params (L_MSE)
                "ELBO": epoch_dict['ELBO'],
                "Z_MSE": epoch_dict["z_mse"],
                "X_MSE": epoch_dict["x_mse"],
                "L_MSE": epoch_dict["L_mse"],
                "KL(L)": epoch_dict["KL(L)"],
                "true_obs_KL_term_Z": epoch_dict["true_obs_KL_term_Z"],

                # Distance from interventional distributions
                "train sample KL": mean_dict["sample_kl"],
                "true sample KL": mean_dict["true_kl"],
                "Sample Wasserstein": mean_dict["sample_wasserstein"],
                "True Wasserstein": mean_dict["true_wasserstein"],

                # Evaluating structure and causal variables (MCC)
                "Evaluations/SHD": mean_dict["shd"],
                "Evaluations/SHD_C": mean_dict["shd_c"],
                "Evaluations/AUROC": mean_dict["auroc"],
                "Evaluations/AUPRC_W": mean_dict["auprc_w"],
                "Evaluations/AUPRC_G": mean_dict["auprc_g"],
                'Evaluations/MCC': mcc_score,
                
                # Confusion matrix related metrics for structure
                'Evaluations/TPR': mean_dict["tpr"],
                'Evaluations/FPR': mean_dict["fpr"],
                'Evaluations/TP': mean_dict["tp"],
                'Evaluations/FP': mean_dict["fp"],
                'Evaluations/TN': mean_dict["tn"],
                'Evaluations/FN': mean_dict["fn"],
                'Evaluations/Recall': mean_dict["recall"],
                'Evaluations/Precision': mean_dict["precision"],
                'Evaluations/F1 Score': mean_dict["fscore"],
            }

            if opt.off_wandb is False:  
                plt.imshow(jnp.mean(pred_W, axis=0))
                plt.savefig(join(logdir, 'pred_w.png'))
                wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
                plt.close('all')

                plt.figure()
                plt.imshow(jnp.mean(pred_x[:, start_idx, :, :, 0], axis=0)/255.)
                plt.savefig(join(logdir, 'pred_image.png'))
                wandb_dict["graph_structure(GT-pred)/Reconstructed image"] = wandb.Image(join(logdir, 'pred_image.png'))
                plt.close('all')

                plt.figure()
                plt.imshow(images[start_idx, :, :, 0]/255.)
                plt.savefig(join(logdir, 'gt_image.png'))
                wandb_dict["graph_structure(GT-pred)/GT image"] = wandb.Image(join(logdir, 'gt_image.png'))
                plt.close('all')
                
                wandb.log(wandb_dict, step=i)

            shd = mean_dict["shd"]
            tqdm.write(f"Epoch {i} | {epoch_dict['ELBO']}")
            tqdm.write(f"Z_MSE: {epoch_dict['z_mse']} | X_MSE: {epoch_dict['x_mse']}")
            tqdm.write(f"L MSE: {epoch_dict['L_mse']}")
            tqdm.write(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
            tqdm.write(f"KL(learned || true): {onp.array(epoch_dict['true_obs_KL_term_Z'])}")
            tqdm.write(f" ")


        pbar.set_postfix(
            Epoch=i,
            X_mse=f"{epoch_dict['x_mse']:.2f}",
            KL=f"{epoch_dict['true_obs_KL_term_Z']:.4f}", 
            L_mse=f"{epoch_dict['L_mse']:.3f}",
            SHD=shd
        )

loss, (res, _, _) = get_loss(model_params, L_params, state, test_images, test_interv_nodes, test_interv_values)
pred_image = jnp.mean(res[-1], axis=0)

_, h, w, c = pred_image.shape
padded_pred_images = onp.zeros((5, w, c))

for i in range(num_test_samples):
    padded_pred_images = onp.concatenate((padded_pred_images, pred_image[i]), axis=0)
    padded_pred_images = onp.concatenate((padded_pred_images, onp.zeros((5, w, c))), axis=0)

padded_pred_images = padded_pred_images[:, :, 0]

plt.figure()
plt.imshow(padded_pred_images/255.)
plt.savefig(f'/home/mila/j/jithendaraa.subramanian/scratch/test_pred_image_learnP{opt.learn_P}_seed{opt.data_seed}_d{d}_ee_{int(opt.exp_edges)}.png')
plt.close('all')

print(test_interv_nodes)
print(test_interv_values)
print(test_interv_data)



