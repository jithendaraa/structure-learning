import sys, pdb, os, imageio, pathlib, wandb, optax, time
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
from jax import vmap, grad, jit, lax, pmap, value_and_grad, partial, config 
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
from dag_utils import SyntheticDataset

from bcd_utils import *
from models.Decoder_BCD import Decoder_BCD


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
if do_ev_noise: noise_dim = 1
else: noise_dim = dim
l_dim = dim * (dim - 1) // 2
input_dim = l_dim + noise_dim
log_stds_max: Optional[float] = 10.0
if opt.fixed_tau is not None: tau = opt.fixed_tau
else: raise NotImplementedError


# noise sigma usually around 0.1 but in original BCD nets code it is set to 1
sd = SyntheticDataset(n=n_data, d=opt.num_nodes, graph_type="erdos-renyi", degree= 2 * degree, 
                        sem_type=opt.sem_type, dataset_type='linear', noise_scale=opt.noise_sigma) 
ground_truth_W = sd.W
ground_truth_P = sd.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

Xs = sd.simulate_sem(ground_truth_W, n_data, sd.sem_type, noise_scale=opt.noise_sigma, dataset_type="linear")
Xs = cast(jnp.ndarray, Xs)

# ! [TODO] Supports only single interventions currently; will not work for more than 1-node intervs
( obs_data, interv_data, z_gt, 
no_interv_targets, x, p_z_mu, 
p_z_covar ) = datagen.get_data(opt, n_interv_sets, None, Xs)

# ? Set parameter for Horseshoe prior on L
if opt.use_alternative_horseshoe_tau:   raise NotImplementedError
else:   horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((dim - 1) - 2 * degree))
if horseshoe_tau < 0:  horseshoe_tau = 1 / (2 * dim)
print(f"Horseshoe tau is {horseshoe_tau}")

# ? 1. Set optimizers for P and L 
P_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
opt_P = optax.chain(*P_layers)
opt_L = optax.chain(*L_layers)


def forward_fn(hard, rng_keys, interv_targets, init, 
                P_params=None, L_params=None, L_states=None, gt_data=None):
    model = Decoder_BCD(dim, l_dim, noise_dim, opt.batch_size, opt.hidden_size, opt.max_deviation, do_ev_noise, 
                        opt.proj_dims, log_stds_max, opt.logit_constraint, tau, opt.subsample, opt.s_prior_std, horseshoe_tau=horseshoe_tau)
    return model(hard, rng_keys, interv_targets, init, P_params, L_params, L_states, gt_data) 

forward = hk.transform(forward_fn)
# key = hk.PRN

def init_parallel_params(rng_key: PRNGKey):
    # @pmap
    def init_params(rng_key: PRNGKey):
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,))
        L_states = jnp.array([0.0]) # Would be nice to put none here, but need to pmap well
        P_params = forward.init(next(key), False, rng_key, jnp.array(no_interv_targets), True)
        if opt.factorized:  raise NotImplementedError
        P_opt_params = opt_P.init(P_params)
        L_opt_params = opt_L.init(L_params)
        return P_params, L_params, L_states, P_opt_params, L_opt_params

    rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
    output = init_params(rng_keys)
    return output


# ? 2. Init params for P and L and get optimizer states
P_params, L_params, L_states, P_opt_params, L_opt_params = init_parallel_params(rng_key)
rng_keys = rnd.split(rng_key, num_devices)
print(f"L model has {ff2(num_params(L_params))} parameters")
print(f"P model has {ff2(num_params(P_params))} parameters")

hard = True

# ! Remove this line
no_interv_targets = index_update(no_interv_targets, index[2, :], True)

max_cols = jnp.max(no_interv_targets.sum(1))
data_idx_array = jnp.array([jnp.arange(opt.num_nodes + 1)]*opt.num_samples)
interv_nodes = jnp.split(data_idx_array[no_interv_targets], no_interv_targets.sum(1).cumsum()[:-1])
interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([opt.num_nodes] * (max_cols - len(interv_nodes[i]))))) for i in range(opt.num_samples)]).astype(int)

ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=opt.max_deviation)

def topo(P, W):
    ordering = jnp.arange(0, dim)
    swapped_ordering = ordering[jnp.where(P, size=dim)[1].argsort()]
    g = nx.from_numpy_matrix(np.array(W), create_using=nx.DiGraph)
    toporder = nx.topological_sort(g)
    t = []
    for i in toporder: t.append(i)
    t = jnp.array(t)
    assert jnp.array_equal(swapped_ordering, t)
    return swapped_ordering, t

def elbo(P_params: PParamType, L_params: hk.Params, L_states: LStateType,
    z_gt: jnp.ndarray, rng_key: PRNGKey, tau: float, num_outer: int, 
    hard: bool, interv_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, LStateType]:
    
    num_bethe_iters = 20
    
    # * Horseshoe prior over lower triangular matrix L
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)
    
    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        rng_key, rng_key_1 = rnd.split(rng_key, 2)
        
        (batched_P, batched_P_logits, batched_L, batched_log_noises, 
        batched_W, batched_qz_samples, full_l_batch, full_log_prob_l, 
        X_recons) = forward.apply(P_params, rng_key, hard, rng_key, 
                    interv_nodes, False, P_params, L_params, L_states, 
                    gt_data = z_gt)

        likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None, None, None))(
            z_gt, batched_log_noises, batched_P, batched_L, rng_key, opt.subsample, s_prior_std
        )
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :l_dim], axis=1)
        s_prior_probs = jnp.sum(full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1)
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs

        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
            batched_P, batched_P_logits, num_bethe_iters
        )
        log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), None

    rng_keys = rnd.split(rng_key, num_outer)
    _, (elbos, out_L_states) = lax.scan(lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys)
    elbo_estimate = jnp.mean(elbos)
    return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)

def gradient_step(P_params, L_params, L_states, z_gt, P_opt_state, L_opt_state, rng_key, tau):
    rng_key, rng_key_2 = rnd.split(rng_key, 2)
    tau_scaling_factor = 1.0 / tau

    # * Get loss and gradients of loss wrt to P and L
    (loss, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
        P_params, L_params, L_states, z_gt, rng_key, tau, num_outer, hard=True,
        interv_nodes=interv_nodes)

    elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)
    
    # * L2 regularization over parameters of P
    l2_elbo_grad_P = grad(lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p)))(P_params)
    elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

    # * Update P network
    P_updates, P_opt_state = opt_P.update(elbo_grad_P, P_opt_state, P_params)
    P_params = optax.apply_updates(P_params, P_updates)
    
    # * Update L network
    L_updates, L_opt_state = opt_L.update(elbo_grad_L, L_opt_state, L_params)
    L_params = optax.apply_updates(L_params, L_updates)

    return ( loss, P_params, L_params, L_states, P_opt_state, L_opt_state, rng_key_2)



for i in range(opt.num_steps):
    
    ( loss, P_params, new_L_params, L_states, P_opt_params, 
        new_L_opt_params, new_rng_key ) = gradient_step(P_params, L_params, L_states, 
                                            z_gt, P_opt_params, L_opt_params, rng_key, tau)
    
    if jnp.any(jnp.isnan(ravel_pytree(new_L_params)[0])):   raise Exception("Got NaNs in L params")
    
    L_params, L_opt_params, rng_key = new_L_params, new_L_opt_params, new_rng_key

    break




