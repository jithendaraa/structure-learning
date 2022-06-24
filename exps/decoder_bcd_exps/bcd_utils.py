import sys, optax
sys.path.append("../../models")
sys.path.append("../../modules")
from models.Decoder_BCD import Decoder_BCD
from tensorflow_probability.substrates.jax.distributions import Horseshoe

from typing import Union, Callable, cast, Any, Tuple
import jax.numpy as jnp
import haiku as hk
import jax, pdb, time, cdt, optax
import jax.random as rnd
import numpy as np
from jax import vmap, jit, vjp, ops, grad, lax, config, value_and_grad
from sklearn.metrics import roc_curve, auc

# cdt.SETTINGS.rpath = "/path/to/Rscript/binary""
from cdt.metrics import SHD_CPDAG
import networkx as nx
import pickle as pkl
from dag_utils import count_accuracy

from divergences import *
import haiku as hk
from jax.flatten_util import ravel_pytree
from jax import tree_util
from jax.tree_util import tree_map, tree_multimap
from haiku._src import data_structures
from loss_fns import log_prob_X
from functools import partial

Tensor = Any
PRNGKey = Any
Network = Callable[[hk.Params, PRNGKey, Tensor, bool], Tensor]

def rk(x):  return rnd.PRNGKey(x)

def un_pmap(x): return tree_map(lambda x: x[0], x)


def get_model(dim: int, batch_size: int, num_layers: int, rng_key: PRNGKey,
    hidden_size: int = 32, do_ev_noise=True,) -> Tuple[hk.Params, Network]:
    if do_ev_noise: noise_dim = 1
    else: noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    input_dim = l_dim + noise_dim
    rng_key = rnd.PRNGKey(0)

    def forward_fn(in_data: jnp.ndarray) -> jnp.ndarray:
        # Must have num_heads * key_size (=64) = embedding_size
        x = hk.Linear(hidden_size)(hk.Flatten()(in_data))
        x = jax.nn.gelu(x)
        for _ in range(num_layers - 2):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(hidden_size)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(dim * dim)(x)

    forward_fn_init, forward_fn_apply = hk.transform(forward_fn)
    blank_data = np.zeros((batch_size, input_dim))
    laplace_params = forward_fn_init(rng_key, blank_data)
    return laplace_params, forward_fn_apply

def to_mutable_dict(mapping):
  """Turns an immutable FlatMapping into a mutable dict."""
  out = {}
  for key, value in mapping.items():
    value_type = type(value)
    if value_type is data_structures.FlatMapping:
      value = to_mutable_dict(value)
    out[key] = value
  return out

def to_FlatMapping(mapping):
  """Turns a mutable dict into an immutable FlatMapping."""
  out = {}
  for key, value in mapping.items():
    value_type = type(value)
    if value_type is dict:
      value = data_structures.FlatMapping(value)
    out[key] = value
  return data_structures.FlatMapping(out)

def get_mse(x, x_pred):
    return jnp.mean(jnp.mean((x - x_pred)**2))

def get_model_arrays(dim: int, batch_size: int, num_layers: int, rng_key: PRNGKey,
    hidden_size: int = 32, do_ev_noise=True) -> hk.Params:
    """Only returns parameters so that it can be used in pmap"""
    if do_ev_noise: noise_dim = 1
    else:           noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    input_dim = l_dim + noise_dim

    def forward_fn(in_data: jnp.ndarray) -> jnp.ndarray:
        # Must have num_heads * key_size (=64) = embedding_size
        x = hk.Linear(hidden_size)(hk.Flatten()(in_data))
        x = jax.nn.gelu(x)
        for _ in range(num_layers - 2):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(hidden_size)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(dim * dim)(x)

    # out_stats = eval_mean(params, Xs, np.zeros(dim))
    forward_fn_init, _ = hk.transform(forward_fn)
    blank_data = np.zeros((batch_size, input_dim))
    laplace_params = forward_fn_init(rng_key, blank_data)
    return laplace_params


def log_prob_x(Xs, log_sigmas, P, L, rng_key, subsample=False, s_prior_std=3.0):
    """Calculates log P(X|Z) for latent Zs

    X|Z is Gaussian so easy to calculate

    Args:
        Xs: an (n x dim)-dimensional array of observations
        log_sigmas: A (dim)-dimension vector of log standard deviations
        P: A (dim x dim)-dimensional permutation matrix
        L: A (dim x dim)-dimensional strictly lower triangular matrix
    Returns:
        log_prob: Log probability of observing Xs given P, L
    """
    adjustment_factor = 1
    if subsample: raise NotImplementedError
        
    n, dim = Xs.shape
    W = (P @ L @ P.T).T
    
    precision = ((jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T)
    eye_minus_W_logdet = 0
    log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

    def datapoint_exponent(x):
        return -0.5 * x.T @ precision @ x

    log_exponent = vmap(datapoint_exponent)(Xs)

    return adjustment_factor * (
        0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
        + jnp.sum(log_exponent)
    )

import jax.numpy as np
import numpy as onp
import jax.numpy as jnp

Tensor = Union[onp.ndarray, np.ndarray]

def get_double_tree_variance(w, z) -> jnp.ndarray:
    """Given two pytrees w, z, compute std[w, z]"""

    def tree_size(x):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.size(leaf) for leaf in leaves])

    def tree_sum(x):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.sum(leaf) for leaf in leaves])

    def sum_square_tree(x, mean):
        leaves, _ = tree_util.tree_flatten(x)
        return sum([jnp.sum((leaf - mean) ** 2) for leaf in leaves])

    # Average over num_repeats, then over all params

    total_size = tree_size(w) + tree_size(z)
    grad_mean = (tree_sum(w) + tree_sum(z)) / total_size
    tree_variance = (
        sum_square_tree(w, grad_mean) + sum_square_tree(z, grad_mean)
    ) / total_size
    return jnp.sqrt(tree_variance)


def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])


def make_to_W(dim: int,) -> Callable[[jnp.ndarray], jnp.ndarray]:
    out = np.zeros((dim, dim))
    w_param_dim = dim * (dim - 1)
    upper_idx = np.triu_indices(dim, 1)
    lower_idx = np.tril_indices(dim, -1)

    def to_W(w_params: jnp.ndarray) -> jnp.ndarray:
        """Turns a (d x (d-1)) vector into a d x d matrix with zero diagonal."""
        tmp = ops.index_update(out, upper_idx, w_params[: w_param_dim // 2])
        tmp = ops.index_update(tmp, lower_idx, w_params[w_param_dim // 2 :])
        return tmp

    return to_W


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[np.triu_indices(dim, 1)]
    out_2 = W[np.tril_indices(dim, -1)]
    return np.concatenate([out_1, out_2])


def lower(theta: Tensor, dim: int) -> Tensor:
    """Given n(n-1)/2 parameters theta, form a
    strictly lower-triangular matrix"""
    out = np.zeros((dim, dim))
    out = ops.index_update(out, np.triu_indices(dim, 1), theta).T
    return out


def upper(theta: Tensor, dim: int) -> Tensor:
    """Given n(n-1)/2 parameters theta, form a
    strictly upper-triangular matrix"""
    out = np.zeros((dim, dim))
    out = ops.index_update(out, np.tril_indices(dim, -1), theta).T
    return out


def get_variances(W_params: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """The maximum likelihood estimate of sigma is the sample variance"""
    dim = X.shape[1]
    to_W = make_to_W(dim)
    W = to_W(W_params)
    residuals = X.T - W.T @ X.T
    residuals = cast(jnp.ndarray, residuals)
    return np.mean(residuals ** 2, axis=1)


def get_variances_from_W(W, X):
    """The maximum likelihood estimate of sigma is the sample variance"""
    residuals = X.T - W.T @ X.T
    return np.mean(residuals ** 2, axis=1)


def get_variance(W_params, X):
    """The maximum likelihood estimate in the equal variance case"""
    n, dim = X.shape
    to_W = make_to_W(dim)
    W = to_W(W_params)
    residuals = X.T - W.T @ X.T
    return np.sum(residuals ** 2) / (dim * n)


def samples_near(mode: Tensor, samples: Tensor, tol: float):
    """Returns the number of samples in an l_0 ball around the mode"""
    is_close = np.linalg.norm(samples - mode[None, :], ord=np.inf, axis=-1) < tol
    return np.mean(is_close)


def get_labels(dim):
    w_param_dim = dim * (dim - 1)
    x1s, y1s = np.triu_indices(dim, 1)
    x2s, y2s = np.tril_indices(dim, -1)
    xs = np.concatenate((x1s, x2s))
    ys = np.concatenate((y1s, y2s))
    return [f"{xs[i]}->{ys[i]}" for i in range(w_param_dim)]


def get_permutation(key: jnp.ndarray, d: int) -> Tensor:
    return rnd.permutation(key, np.eye(d))


def our_jacrev(fun):
    def jacfun(x):
        y, pullback = vjp(fun, x)
        jac = vmap(pullback, in_axes=0)(np.eye(len(y)))
        return jac, y

    return jacfun


def save_params(P_params, L_params, L_states, P_opt_params, L_opt_state, filename):
    filenames = []
    filenames.append("./tmp/P_params" + filename)
    filenames.append("./tmp/L_params" + filename)
    filenames.append("./tmp/L_states" + filename)
    filenames.append("./tmp/P_opt" + filename)
    filenames.append("./tmp/L_opt" + filename)
    inputs = [P_params, L_params, L_states, P_opt_params, L_opt_state]
    for name, obj in zip(filenames, inputs):
        pkl.dump(obj, open(name, "wb"))


def load_params(filename):
    filenames = []
    filenames.append("./tmp/P_params" + filename)
    filenames.append("./tmp/L_params" + filename)
    filenames.append("./tmp/L_states" + filename)
    filenames.append("./tmp/P_opt" + filename)
    filenames.append("./tmp/L_opt" + filename)
    outs = []
    for name in filenames:
        outs.append(pkl.load(open(name, "rb")))
    return outs


def eval_W_ev(est_W, true_W, true_noise, threshold, Xs, est_noise=None,
    provided_x_prec=None, do_shd_c=True, get_wasserstein=True, do_sid=True):

    dim = np.shape(est_W)[0]
    if provided_x_prec is None:
        x_prec = onp.linalg.inv(np.cov(Xs.T))
    else:
        x_prec = provided_x_prec
    x_prec = onp.linalg.inv(np.cov(Xs.T))
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    # Can provide noise or use the maximum-likelihood estimate
    if est_noise is None:
        est_noise = np.ones(dim) * get_variance(from_W(est_W_clipped, dim), Xs)
    else:
        est_noise = np.ones(dim) * est_noise
    stats = count_accuracy(true_W, est_W_clipped)

    if get_wasserstein:
        true_wasserstein_distance = precision_wasserstein_loss(true_noise, true_W, est_noise, est_W_clipped)
        sample_wasserstein_loss = precision_wasserstein_sample_loss(x_prec, est_noise, est_W_clipped)
    else:
        true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0
    true_KL_divergence = precision_kl_loss(true_noise, true_W, est_noise, est_W_clipped)
    sample_kl_divergence = precision_kl_sample_loss(x_prec, est_noise, est_W_clipped)
    if do_shd_c:
        shd_c = SHD_CPDAG(nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W)))
        stats["shd_c"] = shd_c
    else:
        stats["shd_c"] = np.nan

    if do_sid: sid = SHD_CPDAG(onp.array(est_W_clipped != 0), onp.array(true_W != 0))
    else: sid = onp.nan

    stats["true_kl"] = true_KL_divergence
    stats["sample_kl"] = sample_kl_divergence
    stats["true_wasserstein"] = true_wasserstein_distance
    stats["sample_wasserstein"] = sample_wasserstein_loss
    stats["MSE"] = np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
    stats["sid"] = sid
    return stats


def auroc(Ws, W_true, threshold):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    _, dim, dim = Ws.shape
    edge_present = jnp.abs(Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc


def eval_W_non_ev(est_W, true_W, true_noise, threshold, Xs, est_noise=None, provided_x_prec=None,
    do_shd_c=True, get_wasserstein=True, do_sid=True):
    dim = np.shape(est_W)[0]
    if provided_x_prec is None: x_prec = onp.linalg.inv(np.cov(Xs.T))
    else: x_prec = provided_x_prec
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    # Can provide noise or use the maximum-likelihood estimate
    if est_noise is None: est_noise = np.ones(dim) * jit(get_variances)(from_W(est_W_clipped, dim), Xs)
    # Else est_noise is already given as a vector
    stats = count_accuracy(true_W, est_W_clipped)
    true_KL_divergence = jit(precision_kl_loss)(true_noise, true_W, est_noise, est_W_clipped)
    sample_kl_divergence = jit(precision_kl_sample_loss)(x_prec, est_noise, est_W_clipped)

    if get_wasserstein:
        true_wasserstein_distance = jit(precision_wasserstein_loss)(true_noise, true_W, est_noise, est_W_clipped)
        sample_wasserstein_loss = jit(precision_wasserstein_sample_loss)(x_prec, est_noise, est_W_clipped)
    else:
        true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0

    if do_shd_c: shd_c = SHD_CPDAG(nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W)))
    else: shd_c = np.nan
    if do_sid: sid = SHD_CPDAG(onp.array(est_W_clipped != 0), onp.array(true_W != 0))
    else: sid = onp.nan

    stats["true_kl"] = float(true_KL_divergence)
    stats["sample_kl"] = float(sample_kl_divergence)
    stats["true_wasserstein"] = float(true_wasserstein_distance)
    stats["sample_wasserstein"] = float(sample_wasserstein_loss)
    stats["MSE"] = float(np.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2))
    stats["shd_c"] = shd_c
    stats["sid"] = sid
    return stats


def eval_W(est_W, true_W, true_noise, threshold, Xs, get_wasserstein=True):
    dim = np.shape(est_W)[0]
    x_cov = np.cov(Xs.T)
    est_W_clipped = np.where(np.abs(est_W) > threshold, est_W, 0)
    est_noise = jit(get_variances)(from_W(est_W_clipped, dim), Xs)
    stats = count_accuracy(true_W, est_W_clipped)
    true_KL_divergence = jit(kl_loss)(true_noise, true_W, est_noise, est_W_clipped,)
    sample_kl_divergence = jit(kl_sample_loss)(x_cov, est_noise, est_W)
    if get_wasserstein:
        true_wasserstein_distance = jit(wasserstein_loss)(true_noise, true_W, est_noise, est_W_clipped)
        sample_wasserstein_loss = jit(wasserstein_sample_loss)(x_cov, est_noise, est_W)
    else:   true_wasserstein_distance, sample_wasserstein_loss = 0.0, 0.0

    shd_c = np.nan
    try:    shd_c = SHD_CPDAG(nx.DiGraph(onp.array(est_W_clipped)), nx.DiGraph(onp.array(true_W)))
    except: stats["shd_c"] = np.nan

    stats["true_kl"] = true_KL_divergence
    stats["sample_kl"] = sample_kl_divergence
    stats["true_wasserstein"] = true_wasserstein_distance
    stats["sample_wasserstein"] = sample_wasserstein_loss
    stats["MSE"] = np.mean((est_W_clipped - true_W) ** 2)
    stats["shd_c"] = shd_c
    return stats


def random_str():
    out = onp.random.randint(1_000_000) + time.time()
    return str(out)


def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"


def fit_known_edges(W_binary: Tensor, Xs: Tensor, tol: float = 1e-3, max_iters: int = 3_000, lr: float = 1e-2, verbose: bool = True, lambda_1: float = 0.0) -> jnp.ndarray:
    """Given a binary adjacency matrix W_binary, fit linear SEM coefficients from data Xs"""
    # Make sure W_binary is a 1-0 adjacency matrix
    mask = np.where(W_binary == 0, np.zeros_like(W_binary), np.ones_like(W_binary))
    dim = len(W_binary)
    # Add a bit of regularization to keep things nicely-conditioned
    lambda_2 = 1e-6

    def make_optimizer():
        """SGD with nesterov momentum and a custom lr schedule.
        We should be able to use Nesterov momentum since the problem is convex"""
        # (Maybe we will run into issues with the masking etc interacting with the nesterov?)
        return optax.sgd(lr, nesterov=True)

    def inner_loss(p):
        W = p * mask
        return ( jnp.linalg.norm(Xs.T - W.T @ Xs.T) - jnp.linalg.slogdet(jnp.eye(dim) - W)[1]
                + lambda_1 * jnp.sum(np.abs(W)) + lambda_2 * jnp.sum(W ** 2))

    @jit
    def step(p, opt_state):
        g = grad(inner_loss)(p)
        updates, opt_state = make_optimizer().update(g, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, g

    p = rnd.normal(rnd.PRNGKey(0), shape=W_binary.shape)
    g = jnp.ones_like(W_binary) * jnp.inf
    opt_state = make_optimizer().init(p)

    for i in range(max_iters):
        if jnp.linalg.norm(g) < tol:
            if verbose:     print(f"Converged to gradient norm <{tol} after {i} iterations")
            return p * mask
        p, opt_state, g = step(p, opt_state)

    if verbose: print(f"Failed to converge to tol {tol}, actual gradient norm: {jnp.linalg.norm(g)}")
    
    return p * mask


def npperm(M):
    # From user lesshaste on github: https://github.com/scipy/scipy/issues/7151
    n = M.shape[0]
    d = onp.ones(n)
    j = 0
    s = 1
    f = onp.arange(n)
    v = M.sum(axis=0)
    p = onp.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * M[j]
        d[j] = -d[j]
        s = -s
        prod = onp.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


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


def get_interv_joint_dist_params(sigma, W):
    """
        Give the joint interventional distribution for every single node intervention
    """
    interv_mu_joints, interv_Sigma_joints = None, None
    dim, _ = W.shape

    for interv_idx in range(dim):
        W_tilde = W.at[:, interv_idx].set(0.)
        interv_mu_joint, interv_Sigma_joint = get_joint_dist_params(sigma, W_tilde)
        
        if interv_mu_joints is None or interv_Sigma_joints is None:
            interv_mu_joints, interv_Sigma_joints = interv_mu_joint[jnp.newaxis, :], interv_Sigma_joint[jnp.newaxis, :]
        else:
            interv_mu_joints = jnp.concatenate((interv_mu_joints, interv_mu_joint[jnp.newaxis, :]), axis=0)
            interv_Sigma_joints = jnp.concatenate((interv_Sigma_joints, interv_Sigma_joint[jnp.newaxis, :]), axis=0)

    return interv_mu_joints, interv_Sigma_joints


def get_cond_dist_params(node_mus, node_vars, W, P):
    """
        Get variances on each causal variable for a given theta in a linear gaussian additive noise model, 
        given noise variances.
        
        node_vars: noise variances on each node
        W: weighted adj matrix, jnp.array
        P: permutation matrix
    """
    dim = node_vars.shape[0]
    ordering = jnp.arange(0, dim)
    swapped_ordering = ordering[jnp.where(P, size=dim)[1].argsort()]

    # Only for linear gaussian setting
    for j in swapped_ordering:
        node_mus = node_mus.at[j].set( node_mus[j] + jnp.dot( W[:, j], node_mus) )
        node_vars = node_vars.at[j].set( node_vars[j] + jnp.dot( W[:, j]**2, node_vars) )

    node_covars = jnp.diag(node_vars)
    return node_mus, node_covars


def get_W_tilde(W, idx):
    W.at[:, idx].set(0.)
    return W


def get_interv_dists_for_interv(node_mu, node_vars, batch_W, batch_P, interv_idx):
    idx = jnp.where(interv_idx, size=1)
    batch_W_tilde = vmap(get_W_tilde, (0, None), (0))(batch_W, idx)
    batch_qz_mu_interv, batch_qz_covar_interv = vmap(get_cond_dist_params, (None, 0, 0, 0), (0, 0))(node_mu, node_vars, batch_W_tilde, batch_P)
    return batch_qz_mu_interv, batch_qz_covar_interv, batch_W_tilde


def get_posterior_interv_dists(node_mus, node_vars, batch_W, batch_P, interv_idxs):
    batch_qz_mu_intervs, batch_qz_covar_intervs, batch_W_tildes = vmap(get_interv_dists_for_interv, (None, None, None, None, 0), (0, 0, 0))(node_mus, node_vars, batch_W, batch_P, interv_idxs)
    return batch_qz_mu_intervs, batch_qz_covar_intervs


def get_prior_interv_dists(node_mus, node_vars, W, P, opt):
    pz_mu_intervs, pz_covar_intervs = None, None

    for i in range(opt.num_nodes):
        interv = onp.array([False] * opt.num_nodes)
        interv[i] = True
        w_tilde = onp.array(W, copy=True)
        w_tilde[:, i] = 0.0

        pz_mu_interv, pz_covar_interv = get_cond_dist_params(node_mus, node_vars, w_tilde, P)

        if pz_mu_intervs is None:
            pz_mu_intervs, pz_covar_intervs = pz_mu_interv[jnp.newaxis, :], pz_covar_interv[jnp.newaxis, :]
        else:
            pz_mu_intervs = jnp.concatenate((pz_mu_intervs, pz_mu_interv[jnp.newaxis, :]), axis=0)
            pz_covar_intervs = jnp.concatenate((pz_covar_intervs, pz_covar_interv[jnp.newaxis, :]), axis=0)

    return pz_mu_intervs, pz_covar_intervs


def forward_fn(hard, rng_keys, interv_targets, init, opt, horseshoe_tau, proj_matrix,
                ground_truth_L, interv_values, P_params=None, L_params=None, decoder_params=None, log_stds_max=10.0):
    dim = opt.num_nodes
    l_dim = dim * (dim - 1) // 2
    do_ev_noise = opt.do_ev_noise
    if do_ev_noise: noise_dim = 1
    else:           noise_dim = dim

    model = Decoder_BCD(dim, l_dim, noise_dim, opt.batch_size, opt.hidden_size, opt.max_deviation, do_ev_noise, 
            opt.proj_dims, log_stds_max, opt.logit_constraint, opt.fixed_tau, opt.subsample, opt.s_prior_std, 
            horseshoe_tau=horseshoe_tau, learn_noise=opt.learn_noise, noise_sigma=opt.noise_sigma, 
            P=proj_matrix, L=jnp.array(ground_truth_L), decoder_layers=opt.decoder_layers, 
            learn_L=opt.learn_L, learn_P=opt.learn_P, pred_last_L=opt.pred_last_L, fix_decoder=opt.fix_decoder)

    return model(hard, rng_keys, interv_targets, interv_values, init, P_params, L_params, decoder_params)

def init_parallel_params(rng_key, key, opt, num_devices, no_interv_targets, 
                        horseshoe_tau, proj_matrix, L, interv_values):
    dim = opt.num_nodes
    if opt.do_ev_noise: noise_dim = 1
    else: noise_dim = dim
    
    l_dim = dim * (dim - 1) // 2
    forward = hk.transform(forward_fn)
    temp_key = rnd.PRNGKey(0)

    P_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    decoder_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]

    opt_P = optax.chain(*P_layers)
    opt_L = optax.chain(*L_layers)
    opt_decoder = optax.chain(*decoder_layers)

    sparsity_mask = jnp.ones((dim, opt.proj_dims))

    if opt.proj_sparsity == 0.0:    pass # * Don't need to do anything

    elif opt.proj_sparsity == 1.0:  
        raise NotImplementedError

    elif opt.proj_sparsity < 1.0:
        raise NotImplementedError
        sparsity_mask = jnp.where(sparsity_mask, 1.0, rnd.bernoulli(key, 1 - opt.proj_sparsity))
    
    decoder_params = jnp.multiply(sparsity_mask, rnd.uniform(temp_key, shape=(dim, opt.proj_dims), minval=-1.0, maxval=1.0))
    
    # @pmap
    def init_params(rng_key: PRNGKey):
        # * mus and stds (indicated by -1) of L
        if opt.learn_noise is False:
            L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(l_dim) - 1, ))
        else:
            L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,)
            )
        P_params = forward.init(next(key), False, rng_key, jnp.array(no_interv_targets), True, opt,
                        horseshoe_tau, proj_matrix, L, interv_values)
        
        if opt.factorized:  raise NotImplementedError
        P_opt_params = opt_P.init(P_params)
        L_opt_params = opt_L.init(L_params)
        return P_params, L_params, P_opt_params, L_opt_params

    decoder_opt_params = opt_decoder.init(decoder_params)
    rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
    P_params, L_params, P_opt_params, L_opt_params = init_params(rng_keys)
    
    rng_keys = rnd.split(rng_key, num_devices)
    print(f"L model has {ff2(num_params(L_params))} parameters")
    print(f"P model has {ff2(num_params(P_params))} parameters")

    return (P_params, L_params, decoder_params, 
            P_opt_params, L_opt_params, decoder_opt_params, 
            rng_keys, forward, opt_P, opt_L, opt_decoder)

def get_lower_elems(L, dim, k=-1):
    return L[jnp.tril_indices(dim, k=k)]

def lower(theta, dim):
    """
        Given n(n-1)/2 parameters theta, form a
        strictly lower-triangular matrix
    """
    out = jnp.zeros((dim, dim))
    out = ops.index_update(out, jnp.tril_indices(dim, -1), theta)
    return out.T

def get_covar(z):
    return jnp.cov(z.T)