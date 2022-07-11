import sys
import pdb
sys.path.append("../../models")
sys.path.append("../../modules")

import jax
import jax.numpy as jnp
import numpy as onp
from models.VAE_BCD import VAE_BCD
import haiku as hk
import optax
from sklearn.metrics import roc_curve, auc
from jax import random


def num_params(params: hk.Params) -> int:
    return len(jax.flatten_util.ravel_pytree(params)[0])

def rk(x):  return random.PRNGKey(x)

def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"


def forward_fn(opt, hard, rng_key, x_targets, interv_targets, interv_values, init=False):
    model = VAE_BCD(
        opt.num_nodes,
        opt.proj_dims,
        opt.learn_noise,
        opt.learn_P,
        jnp.log(opt.noise_sigma),
        opt.batch_size,
        opt.max_deviation,
        projection=opt.proj, 
        ev_noise=opt.do_ev_noise,
        learn_L=opt.learn_L
    )

    return model(hard, rng_key, x_targets, interv_targets, interv_values, init)


def init_vae_bcd_params(opt, hk_key, hard, rng_key, interv_targets, interv_values, num_devices):
    forward = hk.transform(forward_fn)
    model_opt_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_opt_layers)
    sample_x_targets = jnp.ones((opt.num_samples, opt.proj_dims))

    def init_params(rng_key):
        model_params = forward.init(next(hk_key), opt, hard, rng_key, sample_x_targets, interv_targets, interv_values, init=True)
        model_opt_params = opt_model.init(model_params)
        return model_params, model_opt_params

    model_params, model_opt_params = init_params(rng_key)
    rng_keys = random.split(rng_key, num_devices)

    print(f"VAE-BCD model initialized with {ff2(num_params(model_params))} parameters!")

    return forward, model_params, model_opt_params, opt_model, rng_keys


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


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[jnp.triu_indices(dim, 1)]
    out_2 = W[jnp.tril_indices(dim, -1)]
    return jnp.concatenate([out_1, out_2])


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
