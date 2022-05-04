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
from loss_fns import *
from haiku._src import data_structures
from jax import random

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


if do_ev_noise: noise_dim = 1
else: noise_dim = dim
l_dim = dim * (dim - 1) // 2
input_dim = l_dim + noise_dim
log_stds_max: Optional[float] = 10.0
if opt.fixed_tau is not None: tau = opt.fixed_tau
else: raise NotImplementedError


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


# noise sigma usually around 0.1 but in original BCD nets code it is set to 1
sd = SyntheticDataset(n=n_data, d=opt.num_nodes, graph_type="erdos-renyi", degree= 2 * degree, 
                        sem_type=opt.sem_type, dataset_type='linear', noise_scale=opt.noise_sigma,
                        data_seed=opt.data_seed)  
ground_truth_W = sd.W
ground_truth_P = sd.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

obs_data = sd.simulate_sem(ground_truth_W, n_data, sd.sem_type, noise_scale=opt.noise_sigma, dataset_type="linear")
obs_data = cast(jnp.ndarray, obs_data)

# ! [TODO] Supports only single interventions currently; will not work for more than 1-node intervs
( obs_data, interv_data, z_gt, 
no_interv_targets, x, 
sample_mean, sample_covariance, proj_matrix ) = datagen.get_data(opt, n_interv_sets, sd, obs_data, model='bcd')
log_gt_graph(ground_truth_W, logdir, exp_config)


class Decoder(hk.Module):
    def __init__(self, proj_dims):
        super().__init__()

        if opt.decoder_layers == 'linear':
            self.decoder = hk.Sequential([
                hk.Flatten(), 
                # hk.Linear(16, with_bias=False), jax.nn.relu,
                # hk.Linear(64, with_bias=False), jax.nn.relu,
                # hk.Linear(64, with_bias=False), jax.nn.relu,
                # hk.Linear(64, with_bias=False), jax.nn.relu,
                hk.Linear(proj_dims, with_bias=False)
            ])
        elif opt.decoder_layers == 'nonlinear':
            self.decoder = hk.Sequential([
                hk.Flatten(), 
                hk.Linear(16, with_bias=False), jax.nn.relu,
                hk.Linear(64, with_bias=False), jax.nn.relu,
                hk.Linear(64, with_bias=False), jax.nn.relu,
                hk.Linear(64, with_bias=False), jax.nn.relu,
                hk.Linear(proj_dims, with_bias=False)
            ])

    def __call__(self, x):
        return self.decoder(x)

def forward_fn(x):
    model = Decoder(opt.proj_dims)
    return model(x)

forward = hk.transform(forward_fn)

decoder_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
opt_decoder = optax.chain(*decoder_layers)

dummy_input = jnp.ones((10, opt.num_nodes))
decoder_params = forward.init(next(key), dummy_input)
decoder_opt_params = opt_decoder.init(decoder_params)

Sigma = jnp.diag(jnp.array([opt.noise_sigma ** 2] * dim))
cov_z = jnp.linalg.inv((jnp.eye(dim) - sd.W.T)) @ Sigma @ jnp.linalg.inv((jnp.eye(dim) - sd.W.T)).T
prec_z = jnp.linalg.inv(cov_z)
eps = jnp.array([opt.noise_sigma] * dim)

def eltwise_ancestral_sample(rng_key):
        """
            eps: standard deviation
            Given a single weighted adjacency matrix perform ancestral sampling
            Use the permutation matrix to get the topsorted order.
            Traverse topologically and give one ancestral sample using weighted_adj_mat and eps
        """
        sample = jnp.zeros((dim+1))
        ordering = jnp.arange(0, dim)

        theta = sd.W
        adj_mat = jnp.where(sd.W != 0, 1.0, 0.0)
        swapped_ordering = ordering[jnp.where(sd.P, size=dim)[1].argsort()]
        noise_terms = jnp.multiply(eps, random.normal(rng_key, shape=(dim,)))

        # Traverse node topologically
        for j in swapped_ordering:
            mean = sample[:-1] @ theta[:, j]
            sample = index_update(sample, index[j], mean + noise_terms[j])

        return sample[:-1]

def ancestral_sample(rng_key):
    """
        Gives `n_samples` = len(interv_targets) of data generated from one weighted_adj_mat and interventions from `interv_targets`. 
        Each of these samples will be an ancestral sample taking into account the ith intervention in interv_targets
    """

    rng_keys = rnd.split(rng_key, x.shape[0])
    samples = vmap(eltwise_ancestral_sample, (0), (0))(rng_keys)
    return samples


def likelihood_loss(decoder_params, Xs):
    n, _ = Xs.shape
    decoder_matrix = decoder_params['decoder/~/linear']['w']
    cov_x = decoder_matrix.T @ cov_z @ decoder_matrix
    d_cross = jnp.linalg.pinv(decoder_matrix)
    prec_x = d_cross @ prec_z @ d_cross.T
    log_det_precision = jnp.log(jnp.linalg.det(prec_x))
    def datapoint_exponent(x):  return -0.5 * x.T @ prec_x @ x
    log_exponent = vmap(datapoint_exponent)(Xs)
    return (0.5  * (log_det_precision - opt.proj_dims * jnp.log(2 * jnp.pi)) + (jnp.sum(log_exponent) / n))

@jit
def LL_loss(decoder_params, rng_key):
    x_pred = forward.apply(decoder_params, rng_key, z_gt)
    # loss = get_mse(x, x_pred)
    loss = likelihood_loss(decoder_params, x)
    return -jnp.mean(loss), tree_map(lambda x: x, None)

@jit
def mse_loss(decoder_params, rng_key, z_samples):
    x_pred = forward.apply(decoder_params, rng_key, z_samples)
    loss = get_mse(x, x_pred)
    return jnp.mean(loss), tree_map(lambda x: x, None)


loss_type = opt.train_loss


for i in range(40000):
    if opt.z_sample == 'fixed': z_samples = z_gt
    elif opt.z_sample == 'sample': z_samples = ancestral_sample(rng_key)
    
    if loss_type == 'mse':
        (loss, _), decoder_grad = value_and_grad(mse_loss, argnums=(0), has_aux=True)(decoder_params, rng_key, z_samples)
    
    elif loss_type == 'LL':
        (loss, _), decoder_grad = value_and_grad(LL_loss, argnums=(0), has_aux=True)(decoder_params, rng_key)
        decoder_grad = tree_map(lambda x: x, decoder_grad)

    decoder_updates, decoder_opt_params = opt_decoder.update(decoder_grad, decoder_opt_params, decoder_params)
    decoder_params = optax.apply_updates(decoder_params, decoder_updates)
    wandb_dict = {"Loss": onp.array(loss)}

    if (i+1) % 200 == 0 or i == 0:
        # ! Decoder metrics for when it is linear
        if opt.decoder_layers == 'linear':
            decoder_matrix = decoder_params['decoder/~/linear']['w']
            decoder_mse = get_mse(proj_matrix, decoder_matrix)
            decoder_matrix_norm_error = jnp.linalg.norm(decoder_matrix, ord=2) - jnp.linalg.norm(proj_matrix, ord=2)
            wandb_dict['Decoder 2-Norm'] = onp.array(decoder_matrix_norm_error)
            wandb_dict['Decoder MSE'] = onp.array(decoder_mse)
            
            print(f"Step {i} | Loss type: {loss_type} | Loss: {loss} | Decoder Norm Error: {decoder_matrix_norm_error} | Decoder MSE: {decoder_mse}")

        if opt.off_wandb is False:  wandb.log(wandb_dict, step=i)
