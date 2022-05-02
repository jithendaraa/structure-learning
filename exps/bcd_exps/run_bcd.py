import sys, pdb, os, imageio, pathlib, wandb, optax, time
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../modules')

import utils
from PIL import Image
import ruamel.yaml as yaml
from typing import Tuple, Optional, cast, Union
import matplotlib.pyplot as plt 

import jax.random as rnd
from jax import vmap, grad, jit, lax, pmap, value_and_grad, partial, config 
from jax.tree_util import tree_map, tree_multimap
from jax import numpy as jnp
import numpy as onp
import haiku as hk
config.update("jax_enable_x64", True)

from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

from torch.utils.tensorboard import SummaryWriter
from modules.GumbelSinkhorn import GumbelSinkhorn
from dag_utils import SyntheticDataset
from bcd_utils import *

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

def init_parallel_params(rng_key: PRNGKey):
    @pmap
    def init_params(rng_key: PRNGKey):
        L_params = jnp.concatenate(
            (jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,
            )
        )
        # Would be nice to put none here, but need to pmap well
        L_states = jnp.array([0.0])

        P_params = get_model_arrays(dim, batch_size, opt.num_perm_layers, 
                    rng_key, hidden_size=opt.hidden_size, do_ev_noise=opt.do_ev_noise)

        if opt.factorized:  P_params = jnp.zeros((dim, dim))
        P_opt_params = opt_P.init(P_params)
        L_opt_params = opt_L.init(L_params)
        return P_params, L_params, L_states, P_opt_params, L_opt_params

    rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
    output = init_params(rng_keys)
    return output

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")

# ? Parse args
configs = yaml.safe_load((pathlib.Path('../..') / 'configs.yaml').read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)

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

# ? Defining variables
dim = opt.num_nodes
l_dim = dim * (dim - 1) // 2
n_data = opt.n_data
degree = opt.exp_edges
log_stds_max: Optional[float] = 10.0
batch_size = opt.batch_size
logit_constraint = opt.logit_constraint
s_prior_std = opt.s_prior_std
num_outer = opt.num_outer
sem_type = opt.sem_type
fix_L_params = False
calc_shd_c = False
do_ev_noise = opt.do_ev_noise
eval_eid = opt.eval_eid

if opt.do_ev_noise: noise_dim = 1
else:   noise_dim = dim
if opt.use_flow: raise NotImplementedError

ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=opt.max_deviation)

# This may be preferred from 'The horseshoe estimator: Posterior concentration around nearly black vectors'van der Pas et al
if opt.use_alternative_horseshoe_tau:   raise NotImplementedError
else:   horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((dim - 1) - 2 * degree))

if horseshoe_tau < 0:  # can happen for very small graphs
    horseshoe_tau = 1 / (2 * dim)

print(f"Horseshoe tau is {horseshoe_tau}")

sd = SyntheticDataset(n=n_data, d=dim, graph_type="erdos-renyi", degree=2 * degree, sem_type=opt.sem_type, 
                        dataset_type='linear', noise_scale=opt.noise_sigma) # noise sigma usually around 0.1 but in original BCD nets code it is set to 1
ground_truth_W = sd.W
ground_truth_P = sd.P
print(ground_truth_W)

Xs = sd.simulate_sem(ground_truth_W, n_data, sd.sem_type, 
                    noise_scale=opt.noise_sigma, dataset_type="linear")
Xs = cast(jnp.ndarray, Xs)

test_Xs = sd.simulate_sem(ground_truth_W, sd.n, sd.sem_type, sd.w_range, 
                            sd.noise_scale, sd.dataset_type, sd.W_2)
ground_truth_sigmas = opt.noise_sigma * jnp.ones(dim)
print("\n")

log_gt_graph(ground_truth_W, logdir, exp_config)
L_layers = []
P_layers = []
P_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
L_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
opt_P = optax.chain(*P_layers)
opt_L = optax.chain(*L_layers)
opt_joint = None

_, p_model = get_model(dim, batch_size, opt.num_perm_layers, hidden_size=opt.hidden_size, 
                        do_ev_noise=opt.do_ev_noise, rng_key=rng_key)
P_params, L_params, L_states, P_opt_params, L_opt_params = init_parallel_params(rng_key)
rng_key = rnd.split(rng_key, num_devices)
print(f"L model has {ff2(num_params(L_params))} parameters")
print(f"P model has {ff2(num_params(P_params))} parameters")

if opt.fixed_tau is not None: tau = opt.fixed_tau
else: raise NotImplementedError


def get_P_logits(P_params: PParamType, L_samples: jnp.ndarray, rng_key: PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # ! removed code for this condition from BCD Nets. Refer original code if you want to extend to `factorized` True
    if opt.factorized: raise NotImplementedError

    else:
        P_params = cast(hk.Params, P_params)
        p_logits = p_model(P_params, rng_key, L_samples)  # type:ignore

    if logit_constraint is not None:
        # Want to map -inf to -logit_constraint, inf to +logit_constraint
        p_logits = jnp.tanh(p_logits / logit_constraint) * logit_constraint

    return p_logits.reshape((-1, dim, dim))


def sample_L(L_params: PParamType, L_state: LStateType, rng_key: PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, LStateType]:

    # ! removed the part of the code that has `use_flow` is True
    # ! Refer original BCD Nets code to include `use_flow` support
    L_params = cast(jnp.ndarray, L_params)
    means, log_stds = L_params[: l_dim + noise_dim], L_params[l_dim + noise_dim :]
    if log_stds_max is not None:
        # Do a soft-clip here to stop instability
        log_stds = jnp.tanh(log_stds / log_stds_max) * log_stds_max
    l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))

    if L_dist is Normal:
        full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(batch_size,))
        full_l_batch = cast(jnp.ndarray, full_l_batch)
    else:
        full_l_batch = (
            rnd.laplace(rng_key, shape=(batch_size, l_dim + noise_dim))
            * jnp.exp(log_stds)[None, :]
            + means[None, :]
        )
    full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
    full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

    out_L_states = None
    return full_l_batch, full_log_prob_l, out_L_states


def elbo(P_params: PParamType, L_params: hk.Params, L_states: LStateType,
    Xs: jnp.ndarray, rng_key: PRNGKey, tau: float, num_outer: int = 1, hard: bool = False) -> Tuple[jnp.ndarray, LStateType]:
    """Computes ELBO estimate from parameters.
    Computes ELBO(P_params, L_params), given by
    E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
    where L = g_L(L_params, e2) and P = g_P(P_params, e1).
    The derivative of this corresponds to the pathwise gradient estimator
    Args:
        P_params: inputs to sampling path functions
        L_params: inputs parameterising function giving L|P distribution
        Xs: (n x dim)-dimension array of inputs
        rng_key: jax prngkey object
        log_sigma_W: (dim)-dimensional array of log standard deviations
        log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
    Returns:
        ELBO: Estimate of the ELBO
    """
    num_bethe_iters = 20

    # * Horseshoe prior over lower triangular matrix L
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)

    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        rng_key, rng_key_1 = rnd.split(rng_key, 2)

        # ! Sample L ~ Normal(means, logstds)
        full_l_batch, full_log_prob_l, out_L_states = sample_L(L_params, L_states, rng_key)
        w_noise = full_l_batch[:, -noise_dim:]
        l_batch = full_l_batch[:, :-noise_dim]
        batched_noises = jnp.ones((batch_size, dim)) * w_noise.reshape((batch_size, noise_dim))
        batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, dim)
        batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)

        # ! Sample P ~ Gumbel Sinkhorn (either soft or hard samples)
        if hard:    batched_P_samples = ds.sample_hard_batched_logits(batched_P_logits, tau, rng_key,)
        else:   batched_P_samples = ds.sample_soft_batched_logits(batched_P_logits, tau, rng_key,)

        likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None, None, None))(
            Xs, batched_noises, batched_P_samples, batched_lower_samples, rng_key, opt.subsample, s_prior_std
        )
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :l_dim], axis=1)
        s_prior_probs = jnp.sum(
            full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1
        )
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
            batched_P_samples, batched_P_logits, num_bethe_iters
        )
        log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), out_L_states

    rng_keys = rnd.split(rng_key, num_outer)
    _, (elbos, out_L_states) = lax.scan(lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys)
    elbo_estimate = jnp.mean(elbos)
    return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)


def get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key):
    P_params, L_params, L_states, rng_key = un_pmap(P_params), un_pmap(L_params), un_pmap(L_states), un_pmap(rng_key),
    full_l_batch, _, _ = jit(sample_L)(L_params, L_states, rng_key)
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    _, errors = jit(ds.sample_hard_batched_logits_debug)(batched_P_logits, tau, rng_key)
    first_converged = jnp.where(jnp.sum(errors, axis=0) == -batch_size)[0]
    if len(first_converged) == 0:   converged_idx = -1
    else:   converged_idx = first_converged[0]
    return converged_idx


@jit
def compute_grad_variance(P_params, L_params, L_states, Xs, rng_key, tau):
    P_params, L_params, L_states, rng_key = un_pmap(P_params), un_pmap(L_params), un_pmap(L_states), un_pmap(rng_key),
    (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(P_params, L_params, L_states, Xs, rng_key, tau, num_outer, hard=True)
    return get_double_tree_variance(*grads)


@partial(pmap, axis_name="i", in_axes=(0, 0, 0, None, 0, None, None, None, None),
    static_broadcasted_argnums=(6, 7))
def parallel_elbo_estimate(P_params, L_params, L_states, Xs, rng_keys, tau, n, hard):
    elbos, _ = elbo(
        P_params, L_params, L_states, Xs, rng_keys, tau, n // num_devices, hard
    )
    mean_elbos = lax.pmean(elbos, axis_name="i")
    return jnp.mean(mean_elbos)


@partial(pmap, axis_name="i", in_axes=(0, 0, 0, None, 0, 0, 0, None),
    static_broadcasted_argnums=(3))
def parallel_gradient_step(P_params, L_params, L_states, Xs, P_opt_state, L_opt_state, rng_key, tau):
    rng_key, rng_key_2 = rnd.split(rng_key, 2)
    tau_scaling_factor = 1.0 / tau

    # * Get gradients of the loss
    (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
        P_params, L_params, L_states, Xs, rng_key, tau, num_outer, hard=True,
    )
    elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)
    elbo_grad_P = lax.pmean(elbo_grad_P, axis_name="i")
    elbo_grad_L = lax.pmean(elbo_grad_L, axis_name="i")

    # * L2 regularization over parameters of P
    l2_elbo_grad_P = grad(
        lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p))
    )(P_params)
    elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

    # * Network Parameter updates
    P_updates, P_opt_state = opt_P.update(elbo_grad_P, P_opt_state, P_params)
    P_params = optax.apply_updates(P_params, P_updates)
    L_updates, L_opt_state = opt_L.update(elbo_grad_L, L_opt_state, L_params)
    if fix_L_params: pass
    else: L_params = optax.apply_updates(L_params, L_updates)

    return ( P_params, L_params, L_states, P_opt_state, L_opt_state, rng_key_2)


def eval_mean(P_params, L_params, L_states, Xs, rng_key, do_shd_c, tau=1):
    """Computes mean error statistics for P, L parameters and data"""
    P_params, L_params, L_states = un_pmap(P_params), un_pmap(L_params), un_pmap(L_states)

    if do_ev_noise: eval_W_fn = eval_W_ev
    else: eval_W_fn = eval_W_non_ev
    _, dim = Xs.shape
    x_prec = onp.linalg.inv(jnp.cov(Xs.T))
    full_l_batch, _, _ = sample_L(L_params, L_states, rng_key)
    w_noise = full_l_batch[:, -noise_dim:]
    l_batch = full_l_batch[:, :-noise_dim]
    batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(l_batch, dim)
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(ds.sample_hard_batched_logits)(batched_P_logits, tau, rng_key)

    def sample_W(L, P):
        return (P @ L @ P.T).T

    Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)

    def sample_stats(W, noise):
        stats = eval_W_fn(W, ground_truth_W, ground_truth_sigmas, 0.3,
                            Xs, jnp.ones(dim) * jnp.exp(noise), provided_x_prec=x_prec,
                            do_shd_c=do_shd_c, do_sid=do_shd_c)
        return stats

    stats = sample_stats(Ws[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(Ws[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(Ws, ground_truth_W, 0.3)
    return out_stats


soft_elbo = parallel_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau, 100, False)[0]
best_elbo = -jnp.inf
steps_t0 = time.time()
mean_dict = {}
t0 = time.time()
t_prev_batch = t0

for i in range(opt.num_steps):
    ( P_params, new_L_params, L_states, 
    P_opt_params, new_L_opt_params, 
    new_rng_key ) = parallel_gradient_step(P_params, L_params, L_states, Xs, P_opt_params, L_opt_params, rng_key, tau)

    if jnp.any(jnp.isnan(ravel_pytree(new_L_params)[0])):   raise Exception("Got NaNs in L params")
    L_params = new_L_params
    L_opt_params = new_L_opt_params
    rng_key = new_rng_key

    # if i == 0:
    #     print(f"\nCompiled gradient step after {time.time() - steps_t0}s")
    #     t00 = time.time()

    if i % 100 == 0:
        if opt.fixed_tau is None:   raise NotImplementedError()
        # t000 = time.time()

        # current_elbo = parallel_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau, 100, True)[0]
        # soft_elbo = parallel_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau, 100, False)[0]
        # num_steps_to_converge = get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key)
        wandb_dict = {}

        # wandb_dict = {
        #     "ELBO": onp.array(current_elbo),
        #     "soft ELBO": onp.array(soft_elbo),
        #     "tau": onp.array(tau),
        #     "Wall Time": onp.array(time.time() - t0),
        #     # "Sinkhorn steps": onp.array(num_steps_to_converge),
        # }

        # print(wandb_dict)
        # t_prev_batch = time.time()

        if (i % 100 == 0):
            # Log evalutation metrics at most once every two minutes
            if (i % 10_000 == 0) and (i != 0):  _do_shd_c = False
            else:    _do_shd_c = calc_shd_c

            # cache_str = f"_{sem_type.split('-')[1]}_d_{degree}_s_{opt.data_seed}_{opt.max_deviation}_{opt.use_flow}.pkl"

            # # ? Saving model params
            # if time.time() - steps_t0 > 120:
            #     # Don't cache too frequently
            #     save_params(P_params, L_params, L_states, P_opt_params, L_opt_params, cache_str)
            #     print("cached_params")

            # elbo_grad_std = compute_grad_variance(P_params, L_params, L_states, Xs, rng_key, tau)

            try:
                mean_dict = eval_mean(P_params, L_params, L_states, test_Xs, rk(i), _do_shd_c)
                train_mean_dict = eval_mean(P_params, L_params, L_states, Xs, rk(i), _do_shd_c)
            except:
                print("Error occured in evaluating test statistics")
                continue


            # if current_elbo > best_elbo:
            #     best_elbo = current_elbo
            #     best_shd = mean_dict["shd"]
            #     wandb_dict['best elbo'] = onp.array(best_elbo)
            #     wandb_dict['Evaluations/best shd'] = onp.array(mean_dict["shd"])

            # if eval_eid and i % 8_000 == 0:
            #     t4 = time.time()
            #     eid = eval_ID(P_params, L_params, L_states, Xs, rk(i), tau,)
            #     wandb_dict['eid_wass'] = eid
                # print(f"EID_wass is {eid}, after {time.time() - t4}s")

            # print(f"MSE is {ff2(mean_dict['MSE'])}, SHD is {ff2(mean_dict['shd'])}")

            metrics_ = (
                {"Evaluations/SHD": mean_dict["shd"],
                "Evaluations/SHD_C": mean_dict["shd_c"],
                "Evaluations/SID": mean_dict["sid"],
                "mse": mean_dict["MSE"],
                "Evaluations/tpr": mean_dict["tpr"],
                "Evaluations/fdr": mean_dict["fdr"],
                "Evaluations/fpr": mean_dict["fpr"],
                "Evaluations/AUROC": mean_dict["auroc"],
                # "ELBO Grad std": onp.array(elbo_grad_std),
                "true KL": mean_dict["true_kl"],
                "true Wasserstein": mean_dict["true_wasserstein"],
                "sample KL": mean_dict["sample_kl"],
                "sample Wasserstein": mean_dict["sample_wasserstein"],
                "pred_size": mean_dict["pred_size"],
                "train sample KL": train_mean_dict["sample_kl"],
                "train sample Wasserstein": train_mean_dict["sample_wasserstein"],
                "pred_size": mean_dict["pred_size"]},
            )

            wandb_dict.update(metrics_[0])
            print(f"Step {i} | SHD: {metrics_[0]['Evaluations/SHD']} | SHD_C: {metrics_[0]['Evaluations/SHD_C']} | auroc: {metrics_[0]['Evaluations/AUROC']}")
                # print(metrics_)
            # if opt.use_flow:    raise NotImplementedError("")

            # print("Plotting fig...")
            # full_l_batch, _, _ = jit(sample_L, static_argnums=3)(un_pmap(L_params), un_pmap(L_states), rk(i))
            # P_logits = jit(get_P_logits)(un_pmap(P_params), full_l_batch, rk(i))
            print()

        # batched_P_samples = jit(ds.sample_hard_batched_logits)(P_logits, tau, rk(i))
        # batched_soft_P_samples = jit(ds.sample_soft_batched_logits)(P_logits, tau, rk(i))

        # our_W = (batched_P_samples[0] @ lower(full_l_batch[0, :l_dim], dim) @ batched_P_samples[0].T).T
        # our_W_soft = (batched_soft_P_samples[0] @ lower(full_l_batch[0, :l_dim], dim) @ batched_soft_P_samples[0].T).T

        # plt.imshow(our_W)
        # plt.colorbar()
        # if opt.off_wandb is False: plt.savefig(f"{logdir}/tmp_hard{wandb.run.name}.png")
        # plt.close()

        # plt.imshow(our_W_soft)
        # plt.colorbar()
        # if opt.off_wandb is False: plt.savefig(f"{logdir}/tmp_soft{wandb.run.name}.png")
        # plt.close()

        # if opt.off_wandb is False:        
        #     wandb_dict['graph_structure(GT-pred)/Sample'] = wandb.Image(Image.open(f"{logdir}/tmp_hard{wandb.run.name}.png"), caption="W sample")
        #     wandb_dict['graph_structure(GT-pred)/SoftSample'] = wandb.Image(Image.open(f"{logdir}/tmp_soft{wandb.run.name}.png"), caption="W sample_soft")
        #     wandb.log(wandb_dict, step=i)

        # print(f"Max value of P_logits was {ff2(jnp.max(jnp.abs(P_logits)))}")
        # if dim == 3:    print(get_histogram(L_params, L_states, P_params, rng_key))
        # steps_t0 = time.time()

    if i == 3000:     break