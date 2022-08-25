import sys, pathlib, pdb, os
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

from tqdm import tqdm
import utils
import ruamel.yaml as yaml
import math


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
from models.Conv_Decoder_BCD import Conv_Decoder_BCD, Discriminator


# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

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
assert n % bs == 0
num_batches = n // bs
if opt.dataset == 'chemdata': image_dim = 2500
assert opt.dataset == 'chemdata'
proj_dims = (1, 50, 50)
log_stds_max=10.
logdir = utils.set_tb_logdir(opt)

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high)
images = images[:, :, :, 0:1]
log_gt_graph(gt_W, logdir, vars(opt), opt)

# ? Set parameter for Horseshoe prior on L
horseshoe_tau = (1 / onp.sqrt(n)) * (2 * degree / ((d - 1) - 2 * degree))
if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * d)

ds = GumbelSinkhorn(d, noise_type="gumbel", tol=opt.max_deviation)
gt_L_elems = get_lower_elems(gt_L, d)
p_z_obs_joint_mu, p_z_obs_joint_covar = get_joint_dist_params(opt.noise_sigma, jnp.array(gt_W))

def gen(rng_key, interv_targets, interv_values, L_params):
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
                            )
    return model(rng_key, interv_targets, interv_values, L_params)

def disc(image):
    model = Discriminator(proj_dims)
    return model(image)

gen = hk.transform_with_state(gen)
disc = hk.transform_with_state(disc)

def init_params_and_optimizers():
    if opt.learn_noise is False:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(l_dim) - 1, ))
    else:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,))

    gen_params, gen_state = jit(gen.init)(rng_key, rng_key, interv_nodes[:bs], interv_values[:bs], L_params)
    disc_params, disc_state = disc.init(rng_key, images[:bs])
    
    gen_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_gen = optax.chain(*gen_layers)

    L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_L = optax.chain(*L_layers)

    disc_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.disc_lr)]
    opt_disc = optax.chain(*disc_layers)

    gen_opt_params = opt_gen.init(gen_params)
    L_opt_params = opt_L.init(L_params)
    disc_opt_params = opt_disc.init(disc_params)

    return (gen_state, disc_state, gen_params, disc_params, L_params, 
            gen_opt_params, disc_opt_params, L_opt_params,
            opt_gen, opt_disc, opt_L)

( gen_state, 
    disc_state, 
    gen_params, 
    disc_params, 
    L_params, 
    gen_opt_params, 
    disc_opt_params, 
    L_opt_params, 
    opt_gen, 
    opt_disc,
    opt_L,      ) = init_params_and_optimizers()


@jit
def get_log_dict(res, x_mse, z_data):
    
    (   batch_P, 
        batch_P_logits, 
        batch_L, 
        batch_log_noises,
        batch_W, 
        z_samples, 
        full_l_batch, 
        full_log_prob_l, 
        X_recons        ) = res

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
        "x_mse": x_mse,
        "L_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(gt_L_elems, L_elems)),
        "z_mse": jnp.mean(vmap(get_mse, (None, 0), 0)(z_data, z_samples)),
        "L_elems": jnp.mean(L_elems[:, -1]),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
        "KL(L)": jnp.mean(KL_term_L)
    }

    return log_dict, batch_W, z_samples, X_recons, pred_W_means

@jit
def bce_loss(x, y):
    eps = 1e-8
    return - (y * jnp.log(x + eps) + (1 - y) * jnp.log(1 - x + eps))

@jit
def get_disc_loss(rng_key, gen_params, gen_state, disc_params, disc_state, gt_images, interv_nodes_, interv_values_, L_params):
    gen_res, gen_state = gen.apply(gen_params, gen_state, rng_key, rng_key, interv_nodes_, interv_values_, L_params)
    fake_recons = gen_res[-1]

    disc_real, disc_state = disc.apply(disc_params, disc_state, rng_key, gt_images)
    disc_fake, disc_state = disc.apply(disc_params, disc_state, rng_key, fake_recons.reshape(-1, proj_dims[-2], proj_dims[-1], proj_dims[-3]))
    disc_fake = disc_fake.reshape(opt.batch_size, bs)

    loss_disc_real = bce_loss(disc_real, jnp.ones_like(disc_real))
    loss_disc_fake = jnp.mean(bce_loss(disc_fake, jnp.zeros_like(disc_fake)), axis=0)

    discriminator_loss = jnp.mean(loss_disc_real + loss_disc_fake) / 2
    return discriminator_loss, disc_state

@jit
def update_discriminator(disc_grads, disc_opt_params, disc_params):
    disc_updates, disc_opt_params = opt_disc.update(disc_grads, disc_opt_params, disc_params)
    disc_params = optax.apply_updates(disc_params, disc_updates)
    return disc_opt_params, disc_params

@jit
def update_generator(grads, gen_opt_params, gen_params, L_opt_params, L_params):
    gen_grads, L_grads = tree_map(lambda x_: x_, grads)
    gen_updates, gen_opt_params = opt_gen.update(gen_grads, gen_opt_params, gen_params)
    gen_params = optax.apply_updates(gen_params, gen_updates)
    L_updates, L_opt_params = opt_L.update(L_grads, L_opt_params, L_params)
    L_params = optax.apply_updates(L_params, L_updates)
    return gen_opt_params, gen_params, L_opt_params, L_params


@jit
def get_gen_loss(rng_key, gen_params, gen_state, disc_params, disc_state, gt_images, interv_nodes_, interv_values_, L_params):
    gen_res, gen_state = gen.apply(gen_params, gen_state, rng_key, rng_key, interv_nodes_, interv_values_, L_params)
    fake_recons = gen_res[-1]
    x_mse = jnp.mean(vmap(get_mse, (None, 0), 0)(gt_images, fake_recons))

    disc_fake, disc_state = disc.apply(disc_params, disc_state, rng_key, fake_recons.reshape(-1, proj_dims[-2], proj_dims[-1], proj_dims[-3]))
    disc_fake = disc_fake.reshape(opt.batch_size, bs)
    gen_loss = jnp.mean(bce_loss(disc_fake, jnp.ones_like(disc_fake)))
    return gen_loss, (gen_res, gen_state, x_mse)


def train_batch(gen_state, disc_state, gen_opt_params, gen_params, L_opt_params, L_params, disc_opt_params, disc_params,
                 x_data, z_data, interv_nodes_, interv_values_):
    
    (disc_loss, disc_state), disc_grads = value_and_grad(get_disc_loss, argnums=(3), has_aux=True)(rng_key, 
                                                                                            gen_params, 
                                                                                            gen_state, 
                                                                                            disc_params, 
                                                                                            disc_state, 
                                                                                            x_data, 
                                                                                            interv_nodes_, 
                                                                                            interv_values_, 
                                                                                            L_params)
    
    disc_opt_params, disc_params = update_discriminator(disc_grads, disc_opt_params, disc_params)

    (gen_loss, gen_res), gen_grads = value_and_grad(get_gen_loss, argnums=(1, 8), has_aux=True)(rng_key, 
                                                                                                gen_params, 
                                                                                                gen_state, 
                                                                                                disc_params, 
                                                                                                disc_state, 
                                                                                                x_data,
                                                                                                interv_nodes_, 
                                                                                                interv_values_, 
                                                                                                L_params)
    gen_state = gen_res[1]
    gen_opt_params, gen_params, L_opt_params, L_params = update_generator(gen_grads, gen_opt_params, gen_params, L_opt_params, L_params)

    log_dict, batch_W, z_samples, X_recons, pred_W_means = get_log_dict(gen_res[0], gen_res[2], z_data)
    
    log_dict['Generator loss'] = gen_loss
    log_dict['Discriminator loss'] = disc_loss

    return (gen_state, disc_state, gen_opt_params, gen_params, L_opt_params, L_params, 
            disc_opt_params, disc_params, log_dict, batch_W, z_samples, X_recons, pred_W_means)


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

                (gen_state, 
                    disc_state, 
                    gen_opt_params, 
                    gen_params, 
                    L_opt_params, 
                    L_params, 
                    disc_opt_params, 
                    disc_params, 
                    log_dict, 
                    batch_W, 
                    z_samples, 
                    X_recons, 
                    pred_W_means) = train_batch(gen_state, 
                                                disc_state, 
                                                gen_opt_params, 
                                                gen_params, 
                                                L_opt_params, 
                                                L_params, 
                                                disc_opt_params, 
                                                disc_params,
                                                images[start_idx:end_idx], 
                                                z[start_idx:end_idx], 
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

        if i % 5 == 0:
            random_idxs = onp.random.choice(n, bs, replace=False)
            mean_dict = gan_eval_mean(  gen_params, 
                                        L_params,
                                        gen_state,
                                        gen, 
                                        z, 
                                        rk(i), 
                                        interv_nodes[random_idxs], 
                                        interv_values[random_idxs],
                                        opt, 
                                        gt_W, 
                                        ground_truth_sigmas, 
                                    )

            mcc_scores = []
            for j in range(len(pred_z)):
                mcc_scores.append(get_cross_correlation(onp.array(pred_z[j]), onp.array(z)))
            mcc_score = onp.mean(onp.array(mcc_scores))

            wandb_dict = {
                "Z_MSE": epoch_dict["z_mse"],
                "X_MSE": epoch_dict["x_mse"],
                "L_MSE": epoch_dict["L_mse"],
                "KL(L)": epoch_dict["KL(L)"],
                "true_obs_KL_term_Z": epoch_dict["true_obs_KL_term_Z"],
                "Evaluations/SHD": mean_dict["shd"],
                "Evaluations/SHD_C": mean_dict["shd_c"],
                "Evaluations/AUROC": mean_dict["auroc"],
                "Evaluations/AUPRC_W": mean_dict["auprc_w"],
                "Evaluations/AUPRC_G": mean_dict["auprc_g"],
                "train sample KL": mean_dict["sample_kl"],
                'Evaluations/MCC': mcc_score,
                'Generator loss': epoch_dict['Generator loss'],
                'Discriminator loss': epoch_dict['Discriminator loss']
            }

            if opt.off_wandb is False:  
                plt.imshow(jnp.mean(pred_W, axis=0))
                plt.savefig(join(logdir, 'pred_w.png'))
                wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
                plt.close('all')

                plt.figure()
                plt.imshow(jnp.mean(pred_x[:, 0, :, :, 0], axis=0)/255.)
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
            tqdm.write(f"Epoch {i}")
            tqdm.write(f"Z_MSE: {epoch_dict['z_mse']} | X_MSE: {epoch_dict['x_mse']}")
            tqdm.write(f"L MSE: {epoch_dict['L_mse']}")
            tqdm.write(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
            tqdm.write(f"KL(learned || true): {onp.array(epoch_dict['true_obs_KL_term_Z'])}")
            tqdm.write(f" ")


        pbar.set_postfix(
            Epoch=i,
            Gen_loss=f"{epoch_dict['Generator loss']:.2f}",
            Disc_loss=f"{epoch_dict['Discriminator loss']:.2f}",
            KL=f"{epoch_dict['true_obs_KL_term_Z']:.4f}", 
            L_mse=f"{epoch_dict['L_mse']:.3f}",
            SHD=shd
        )