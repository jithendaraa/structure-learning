import sys, pdb
sys.path.append("../../models")
sys.path.append("../../modules")
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

from models.Conv_Decoder_BCD import Conv_Decoder_BCD
import gym
from typing import OrderedDict
from tqdm import tqdm
from matplotlib import pyplot as plt
from chem_datagen import generate_colors, generate_samples

import numpy as onp
from jax import numpy as jnp
import jax.random as rnd
from jax.flatten_util import ravel_pytree
import jax
import optax
import haiku as hk
from sklearn.metrics import roc_curve, auc
from modules.ColorGen import LinearGaussianColor

def generate_data(opt, low=-8., high=8.):
    n = opt.num_samples
    d = opt.num_nodes

    if opt.generate:
        chem_data = LinearGaussianColor(
                        n=opt.num_samples,
                        obs_data=opt.obs_data,
                        d=opt.num_nodes,
                        graph_type="erdos-renyi",
                        degree=2 * opt.exp_edges,
                        sem_type=opt.sem_type,
                        dataset_type="linear",
                        noise_scale=opt.noise_sigma,
                        data_seed=opt.data_seed,
                        low=low, high=high
                    )
        gt_W = chem_data.W
        gt_P = chem_data.P
        gt_L = chem_data.P.T @ chem_data.W.T @ chem_data.P

        # ? generate linear gaussian colors
        z, interv_targets, interv_values = generate_colors(opt, chem_data, low, high)
        normalized_z = 255. * ((z / (2 * high)) + 0.5)

        # ? Use above colors to generate images
        images = generate_chem_image_dataset(opt.num_samples, opt.num_nodes, interv_values, interv_targets, z)
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_values-seed{opt.data_seed}.npy', onp.array(interv_values))
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_targets-seed{opt.data_seed}.npy', onp.array(interv_targets))
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/z-seed{opt.data_seed}.npy', onp.array(z))
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/images-seed{opt.data_seed}.npy', onp.array(images))
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/W-seed{opt.data_seed}.npy', onp.array(gt_W))
        onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/P-seed{opt.data_seed}.npy', onp.array(gt_P))


    else:
        interv_targets = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_targets-seed{opt.data_seed}.npy'))
        interv_values = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_values-seed{opt.data_seed}.npy'))
        z = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/z-seed{opt.data_seed}.npy'))
        images = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/images-seed{opt.data_seed}.npy'))
        gt_W = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/W-seed{opt.data_seed}.npy'))
        gt_P = jnp.array(onp.load(f'/home/mila/j/jithendaraa.subramanian/scratch/P-seed{opt.data_seed}.npy'))
        gt_L = jnp.array(gt_P.T @ gt_W.T @ gt_P)

    print(gt_W)
    print()

    max_cols = jnp.max(interv_targets.sum(1))
    data_idx_array = jnp.array([jnp.arange(d + 1)] * n)
    interv_nodes = onp.split(data_idx_array[interv_targets], interv_targets.sum(1).cumsum()[:-1])
    interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([d] * (max_cols - len(interv_nodes[i])))))
        for i in range(n)]).astype(int)

    return z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L

def generate_test_samples(d, W, sem_type, noise_sigma, low, high, num_test_samples):
    
    test_interv_data, test_interv_targets, test_interv_values = generate_samples(d,
                                                                            W, 
                                                                            sem_type,
                                                                            noise_sigma,
                                                                            low, high, 
                                                                            num_test_samples
                                                                            )

    test_images = generate_chem_image_dataset(num_test_samples, 
                                            d, 
                                            test_interv_values, 
                                            test_interv_targets, 
                                            test_interv_data)
    _, h, w, c = test_images.shape
    padded_test_images = onp.zeros((5, w, c))

    for i in range(num_test_samples):
        padded_test_images = onp.concatenate((padded_test_images, test_images[i]), axis=0)
        padded_test_images = onp.concatenate((padded_test_images, onp.zeros((5, w, c))), axis=0)

    max_cols = jnp.max(test_interv_targets.sum(1))
    data_idx_array = jnp.array([jnp.arange(d + 1)] * num_test_samples)
    test_interv_nodes = onp.split(data_idx_array[test_interv_targets], test_interv_targets.sum(1).cumsum()[:-1])
    test_interv_nodes = jnp.array([jnp.concatenate((test_interv_nodes[i], jnp.array([d] * (max_cols - len(test_interv_nodes[i])))))
        for i in range(num_test_samples)]).astype(int)

    return test_interv_data, test_interv_nodes, test_interv_values, test_images[:, :, :, 0:1], padded_test_images[:, :, 0]

def generate_chem_image_dataset(n, d, interv_values, interv_targets, z):
    images = None
    env = gym.make(f'LinGaussColorCubesRL-{d}-{d}-Static-10-v0')

    for i in tqdm(range(n)):
        action = OrderedDict()
        action['nodes'] = onp.where(interv_targets[i])
        action['values'] = interv_values[i]
        ob, _, _, _ = env.step(action, z[i])
        
        if i == 0:
            images = ob[1][jnp.newaxis, :]
        else:
            images = onp.concatenate((images, ob[1][jnp.newaxis, :]), axis=0)

    return images

def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"

def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])

def get_lower_elems(L, dim, k=-1):
    return L[jnp.tril_indices(dim, k=k)]

def rk(x):  return rnd.PRNGKey(x)

def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])

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

def forward_fn(opt, rng_key, init, hard, interv_nodes, interv_values, horseshoe_tau, L, P=None, log_stds_max=10.):
    d = opt.num_nodes
    l_dim = d * (d - 1) // 2
    if opt.do_ev_noise: noise_dim = 1
    else:   noise_dim = d
    assert opt.dataset == 'chemdata'
    proj_dims = (1, 50, 50)

    model = Conv_Decoder_BCD(d, 
                            l_dim, 
                            noise_dim, 
                            opt.batch_size, 
                            opt.hidden_size, 
                            opt.max_deviation,
                            opt.do_ev_noise, 
                            proj_dims, 
                            log_stds_max, 
                            logit_constraint=opt.logit_constraint, 
                            tau=opt.fixed_tau,
                            s_prior_std=opt.s_prior_std,
                            horseshoe_tau=horseshoe_tau,
                            learn_noise=opt.learn_noise, 
                            noise_sigma=opt.noise_sigma,
                            L=jnp.array(L),
                            learn_L=opt.learn_L,
                            learn_P=opt.learn_P,
                            P=P)

    model = Conv_Decoder_BCD()

    return model(rng_key, init, hard, interv_nodes, interv_values)



def init_params(rng_key, key, opt, interv_nodes, interv_values, horseshoe_tau, L, P=None):
    
    dim = opt.num_nodes
    if opt.do_ev_noise: noise_dim = 1
    else: noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    
    forward = hk.transform_with_state(forward_fn)
    
    # ! Optimizers
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_layers)
    opt_L = optax.chain(*L_layers)
    
    if opt.learn_noise is False:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(l_dim) - 1, ))
    else:
        L_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1,))
    
    model_params, state = forward.init(next(key), 
                                        opt, 
                                        rng_key, 
                                        True, 
                                        False, 
                                        interv_nodes, 
                                        interv_values, 
                                        horseshoe_tau, 
                                        L, 
                                        P=P)

    model_opt_params = opt_model.init(model_params)
    L_opt_params = opt_L.init(L_params)

    # print(f"L model has {ff2(num_params(L_params))} parameters")
    print(f"Model has {ff2(num_params(model_params))} parameters")

    return (forward, model_params, state, None, opt_model, model_opt_params, None, None)

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

