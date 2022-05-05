import sys, pdb, os, imageio, pathlib, wandb, optax, time, graphical_models
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../modules')
sys.path.append('../../models')

import networkx as nx
import utils, datagen
from PIL import Image
import ruamel.yaml as yaml
from typing import Tuple, Optional, cast, Union
import matplotlib.pyplot as plt 

import jax
import jax.random as rnd
from jax import vmap, grad, jit, lax, pmap, value_and_grad, config 
from jax.tree_util import tree_map, tree_multimap
from jax import numpy as jnp
from jax.ops import index, index_mul, index_update
import numpy as onp
import haiku as hk
config.update("jax_enable_x64", True)

from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

from torch.utils.tensorboard import SummaryWriter
from modules.GumbelSinkhorn import GumbelSinkhorn
from modules.hungarian_callback import hungarian, batched_hungarian

# Data generation procedure
from dibs_new.dibs.target import make_linear_gaussian_model
from dag_utils import SyntheticDataset, count_accuracy

from divergences import *
from bcd_utils import *
from models.Decoder_BCD import Decoder_BCD
from loss_fns import *
from haiku._src import data_structures
from loss_fns import get_single_kl


def log_gt_graph(ground_truth_W, logdir, exp_config_dict):
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
    writer.add_image('graph_structure(GT-pred)/Ground truth W', gt_graph_image, 0, dataformats='HWC')

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")

# ? Parse args
configs = yaml.safe_load((pathlib.Path('../..') / 'configs.yaml').read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

# ? Set logdir
logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))
print()

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
n_interv_sets = 10
calc_shd_c = False
sem_type = opt.sem_type
eval_eid = opt.eval_eid
num_bethe_iters = 20


if do_ev_noise: noise_dim = 1
else: noise_dim = dim
l_dim = dim * (dim - 1) // 2
input_dim = l_dim + noise_dim
log_stds_max: Optional[float] = 10.0
if opt.fixed_tau is not None: tau = opt.fixed_tau
else: raise NotImplementedError
tau_scaling_factor = 1.0 / tau


# noise sigma usually around 0.1 but in original BCD nets code it is set to 1
sd = SyntheticDataset(n=n_data, d=opt.num_nodes, graph_type="erdos-renyi", degree= 2 * degree, 
                        sem_type=opt.sem_type, dataset_type='linear', noise_scale=opt.noise_sigma,
                        data_seed=opt.data_seed) 
ground_truth_W = sd.W
ground_truth_P = sd.P
ground_truth_L = sd.P.T @ sd.W.T @ sd.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

obs_data = sd.simulate_sem(ground_truth_W, n_data, sd.sem_type, noise_scale=opt.noise_sigma, dataset_type="linear")
obs_data = cast(jnp.ndarray, obs_data)


# ! [TODO] Supports only single interventions currently; will not work for more than 1-node intervs
( obs_data, interv_data, z_gt, 
no_interv_targets, x, 
sample_mean, sample_covariance, proj_matrix ) = datagen.get_data(opt, n_interv_sets, sd, obs_data, model='bcd')

node_mus = jnp.ones((dim)) * opt.noise_mu
node_vars = jnp.ones((dim)) * (opt.noise_sigma**2)

p_z_mu, p_z_covar = get_joint_dist_params(node_mus, node_vars, sd.W, sd.P)

# else:
#     raise NotImplementedError("")

# ? Set parameter for Horseshoe prior on L
if opt.use_alternative_horseshoe_tau:   raise NotImplementedError
else:   horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((dim - 1) - 2 * degree))
if horseshoe_tau < 0:  horseshoe_tau = 1 / (2 * dim)
print(f"Horseshoe tau is {horseshoe_tau}")

# ? 1. Set optimizers for P and L 
log_gt_graph(ground_truth_W, logdir, exp_config)
P_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
opt_P = optax.chain(*P_layers)
opt_L = optax.chain(*L_layers)

def forward_fn(hard, rng_keys, interv_targets, init, P_params=None, L_params=None):
    model = Decoder_BCD(dim, l_dim, noise_dim, opt.batch_size, opt.hidden_size, opt.max_deviation, do_ev_noise, 
                        opt.proj_dims, log_stds_max, opt.logit_constraint, tau, opt.subsample, opt.s_prior_std, 
                        horseshoe_tau=horseshoe_tau, learn_noise=opt.learn_noise, noise_sigma=opt.noise_sigma,
                        P=proj_matrix, L=jnp.array(ground_truth_L), decoder_layers=opt.decoder_layers, 
                        learn_L=opt.learn_L, z_gt=z_gt)

    return model(hard, rng_keys, interv_targets, init, P_params, L_params) 

forward = hk.transform(forward_fn)

def init_parallel_params(rng_key: PRNGKey):
    # @pmap
    def init_params(rng_key: PRNGKey):
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(l_dim) - 1,))
        P_params = forward.init(next(key), False, rng_key, jnp.array(no_interv_targets), True)
        if opt.factorized:  raise NotImplementedError
        P_opt_params = opt_P.init(P_params)
        L_opt_params = opt_L.init(L_params)
        return P_params, L_params, P_opt_params, L_opt_params

    rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
    output = init_params(rng_keys)
    return output

def get_lower_elems(L):
    return L[jnp.tril_indices(opt.num_nodes, k=-1)]

# ? 2. Init params for P and L and get optimizer states
P_params, L_params, P_opt_params, L_opt_params = init_parallel_params(rng_key)
rng_keys = rnd.split(rng_key, num_devices)
print(f"L model has {ff2(num_params(L_params))} parameters")
print(f"P model has {ff2(num_params(P_params))} parameters")


hard = True
max_cols = jnp.max(no_interv_targets.sum(1))
data_idx_array = jnp.array([jnp.arange(opt.num_nodes + 1)]*opt.num_samples)
interv_nodes = jnp.split(data_idx_array[no_interv_targets], no_interv_targets.sum(1).cumsum()[:-1])
interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([opt.num_nodes] * (max_cols - len(interv_nodes[i]))))) for i in range(opt.num_samples)]).astype(int)
ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=opt.max_deviation)
Sigma = jnp.diag(jnp.array([opt.noise_sigma ** 2] * dim))
gt_L_elems = get_lower_elems(ground_truth_L)

def log_prob_X(Xs, log_sigmas, P, L, rng_key, decoder_matrix, subsample=False, s_prior_std=3.0):
    adjustment_factor = 1
    if subsample: raise NotImplementedError   # ! To implement, look at this function in the original code of BCD Nets  
    n, _ = Xs.shape
    W = (P @ L @ P.T).T

    if opt.fix_decoder: d_cross = jnp.linalg.pinv(proj_matrix)
    else: d_cross = jnp.linalg.pinv(decoder_matrix)

    cov_z = jnp.linalg.inv(jnp.eye(dim) - W).T @ Sigma @ jnp.linalg.inv(jnp.eye(dim) - W)
    prec_z = jnp.linalg.inv(cov_z)
    cov_x = decoder_matrix.T @ cov_z @ decoder_matrix

    if opt.cov_space is True: precision_x = jnp.linalg.inv(cov_x)
    else: precision_x = d_cross @ prec_z @ d_cross.T

    log_det_precision = jnp.log(jnp.linalg.det(precision_x))
    def datapoint_exponent(x_): return -0.5 * x_.T @ precision_x @ x_
    log_exponent = vmap(datapoint_exponent)(Xs)
    return adjustment_factor * (0.5 * (log_det_precision - opt.proj_dims * jnp.log(2 * jnp.pi)) + jnp.sum(log_exponent)/n)


# @jit
def hard_elbo(P_params: PParamType, L_params: hk.Params, rng_key: PRNGKey, interv_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, LStateType]:
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)  # * Horseshoe prior over lower triangular matrix L
    rng_key = rnd.split(rng_key, num_outer)[0]
    hard = True
    
    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        # rng_key, rng_key_1 = rnd.split(rng_key, 2)
        
        if opt.learn_P is False and opt.decoder_layers == 'linear':
            decoder_params = P_params['decoder_bcd/~/linear']['w']
        
        (batched_P, batched_P_logits, batched_L, batched_log_noises, 
        batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, 
        X_recons) = forward.apply(P_params, rng_key, hard, rng_key, 
                        interv_nodes, False, P_params, L_params)
        
        # ! likelihoods over high-dim data X
        if opt.train_loss == 'mse':
            likelihoods = - jit(vmap(get_mse, (None, 0), 0))(x, X_recons)

        elif opt.train_loss == 'LL':
            likelihoods = vmap(log_prob_X, in_axes=(None, 0, 0, 0, None, None, None, None))(
                x, batched_log_noises, batched_P, batched_L, rng_key, 
                decoder_params, opt.subsample, s_prior_std)
        
        final_term = likelihoods
        
        # ! KL over P
        if opt.P_KL is True:
            pdb.set_trace()
            logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(batched_P, batched_P_logits, num_bethe_iters)
            log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
            KL_term_P = logprob_P - log_P_prior
            final_term -= KL_term_P

        # ! KL over L
        if opt.L_KL is True:
            l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
            KL_term_L = full_log_prob_l - l_prior_probs
            final_term -= KL_term_L

        if opt.Z_KL is True:
            node_mus = jnp.ones((dim)) * opt.noise_mu
            node_vars = jnp.ones((opt.batch_size, dim)) * jnp.exp(2 * batched_log_noises)
            batched_get_joint_dist_params = vmap(get_joint_dist_params, (None, 0, 0, 0), (0, 0))
            q_z_mus, q_z_covars = batched_get_joint_dist_params(node_mus, node_vars, batched_W, batched_P)
            KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), (0))(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt)
            final_term -= KL_term_Z

        return (-jnp.mean(final_term))
    
    # elbos = outer_loop(rng_keys[0])
    _, elbos = lax.scan(lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys)
    return jnp.mean(elbos)


def eval_mean(P_params, L_params, data, rng_key, do_shd_c=True, tau=1, step = None):
    """Computes mean error statistics for P, L parameters and data"""
    z_prec = onp.linalg.inv(jnp.cov(data.T))

    (batched_P, batched_P_logits, batched_L, batched_log_noises, 
    batched_W, batched_qz_samples, full_l_batch, 
    full_log_prob_l, X_recons) = forward.apply(P_params, rng_key, True, rng_key, 
                                    interv_nodes, False, P_params, L_params)

    w_noise = full_l_batch[:, -noise_dim:]
    Xs = obs_data

    def sample_stats(est_W, noise, threshold=0.3, get_wasserstein=False):
        if do_ev_noise is False: raise NotImplementedError("")
        
        est_noise = jnp.ones(dim) * jnp.exp(noise)
        est_W_clipped = jnp.where(jnp.abs(est_W) > threshold, est_W, 0)
        gt_graph_clipped = jnp.where(jnp.abs(ground_truth_W) > threshold, est_W, 0)

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
    for i, W in enumerate(batched_W[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(batched_W, ground_truth_W, 0.3)
    return out_stats

# @jit
def gradient_step(P_params, L_params, rng_key):
    # * Get loss and gradients of loss wrt to P and L
    loss, grads = value_and_grad(hard_elbo, argnums=(0, 1))(P_params, L_params, rng_key, interv_nodes)
    
    if opt.tau_scaling is True: 
        if opt.learn_P is False:
            # ! Needs fix 
            elbo_grad_P = None
            elbo_grad_L = tree_map(lambda x_: tau_scaling_factor * x_, grads)
        else:
            elbo_grad_P, elbo_grad_L = tree_map(lambda x_: tau_scaling_factor * x_, grads)
    else: 
        elbo_grad_P, elbo_grad_L = tree_map(lambda x_: x_, grads)
    
    # * L2 regularization over parameters of P (contains p network and decoder)
    if opt.l2_reg is True: 
        l2_elbo_grad_P = grad(lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p)))(P_params)
        elbo_grad_P = tree_multimap(lambda x_, y: x_ + y, elbo_grad_P, l2_elbo_grad_P)

    rng_key_ = rnd.split(rng_key, num_outer)[0]

    (_, _, batch_L, _, batch_W, 
    z_samples, _, _, X_recons) = forward.apply(P_params, rng_key_, hard, rng_key_, 
                                        interv_nodes, False, P_params, L_params)
    L_elems = vmap(get_lower_elems, 0, 0)(batch_L)

    mse_dict = {
        'L_mse': jnp.mean(vmap(get_mse, (None, 0), 0)(gt_L_elems, L_elems)),
        'z_mse': jnp.mean(vmap(get_mse, (None, 0), 0)(z_gt, z_samples)),
        'x_mse': jnp.mean(vmap(get_mse, (None, 0), 0)(x, X_recons)),
        'L_elems': jnp.mean(L_elems[:, -1])
    }
    return ( loss, rng_key, elbo_grad_P, elbo_grad_L, mse_dict)


for i in range(opt.num_steps):

    ( loss, rng_key, elbo_grad_P, elbo_grad_L, mse_dict ) = gradient_step(P_params, L_params, rng_key)

    if i % 200 == 0: 
        if opt.train_loss == 'LL':
            if opt.L_KL and opt.learn_L != False:    print(f"Training on LL loss | learning L with KL")
            elif opt.learn_L != False: print(f"Training on LL loss | learning L no KL")
        else:   print(f"Training on MSE loss")

        if opt.decoder_layers == 'linear':
            if opt.learn_P is False: decoder_params = P_params['decoder_bcd/~/linear']['w']
            else:   
                for param in P_params: decoder_params = param
            decoder_2_norm = jnp.linalg.norm(proj_matrix, ord=2) - jnp.linalg.norm(decoder_params, ord=2)
            decoder_mse = get_mse(proj_matrix, decoder_params)
            decoder_fro_norm = jnp.linalg.norm(proj_matrix, ord='fro') - jnp.linalg.norm(decoder_params, ord='fro')
            print(f"Decoder MSE: {decoder_mse} | Decoder 2-Norm: {decoder_2_norm} | Decoder Fro. norm: {decoder_fro_norm}")
        
        print()
        print(f"Step {i} | {loss}")
        print(f"Z_MSE: {mse_dict['z_mse']} | X_MSE: {mse_dict['x_mse']}")
        print(f"L MSE: {mse_dict['L_mse']}")
        
        if opt.fixed_tau is None:   raise NotImplementedError()
        h_elbo = hard_elbo(P_params, L_params, rng_key, interv_nodes)
        mean_dict = eval_mean(P_params, L_params, z_gt, rk(i), True, tau)

        wandb_dict = {
            "ELBO": onp.array(h_elbo),
            "Z_MSE": onp.array(mse_dict['z_mse']),
            "X_MSE": onp.array(mse_dict['x_mse']),
            "L_MSE": onp.array(mse_dict['L_mse']),
            'L_elems_avg': onp.array(mse_dict['L_elems']),
            "Evaluations/SHD": mean_dict["shd"], 
            "Evaluations/SHD_C": mean_dict["shd_c"],
            "Evaluations/AUROC": mean_dict["auroc"],
            "train sample KL": mean_dict["sample_kl"],
        }

        print(f"SHD: {mean_dict['shd']} | CPDAG SHD: {mean_dict['shd_c']} | AUROC: {mean_dict['auroc']}")
        
        if opt.decoder_layers == 'linear':
            wandb_dict['Decoder MSE'] = onp.array(decoder_mse)
            wandb_dict['Decoder 2-Norm'] = onp.array(decoder_2_norm)
            wandb_dict['Decoder Fro. norm'] = onp.array(decoder_fro_norm)

        if opt.off_wandb is False:
            wandb.log(wandb_dict, step=i)

        

    # * Update P network
    P_updates, P_opt_params = opt_P.update(elbo_grad_P, P_opt_params, P_params)
    P_params = optax.apply_updates(P_params, P_updates)
    
    # * Update L network
    L_updates, L_opt_params = opt_L.update(elbo_grad_L, L_opt_params, L_params)
    L_params = optax.apply_updates(L_params, L_updates)
    if jnp.any(jnp.isnan(ravel_pytree(L_params)[0])):   raise Exception("Got NaNs in L params")
    