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
from models.Conv_Decoder_BCD import Conv_Decoder_BCD, ConvDiscriminator


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

low, high = -8., 8.
interv_low, interv_high = -5., 5.
hard = True
num_bethe_iters = 20
assert opt.train_loss == 'mse'
bs = opt.batches
assert n % bs == 0
num_batches = n // bs
assert opt.dataset == 'chemdata'
log_stds_max=10.
logdir = utils.set_tb_logdir(opt)

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high, interv_low, interv_high)
images = images[:, :, :, 0:1]

_, h, w, _ = images.shape
proj_dims = (1, h, w)
log_gt_graph(gt_W, logdir, vars(opt), opt)

# ? Set parameter for Horseshoe prior on L
horseshoe_tau = (1 / onp.sqrt(n)) * (2 * degree / ((d - 1) - 2 * degree))
if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * d)

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
                                noise_sigma=opt.noise_sigma,
                                learn_L=opt.learn_L
                            )
                        
    return model(rng_key, interv_targets, interv_values, L_params)

def disc(image):
    model = ConvDiscriminator()
    return model(image)

gen = hk.transform(gen)
disc = hk.transform(disc)