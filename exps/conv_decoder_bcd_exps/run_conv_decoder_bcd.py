import sys, pathlib, pdb, os
import gym
from tqdm import tqdm

from conv_decoder_bcd_utils import generate_chem_image_dataset

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

import envs
import utils
import ruamel.yaml as yaml

from modules.ColorGen import LinearGaussianColor
from chem_datagen import generate_colors

import numpy as np
import jax
from jax import numpy as jnp


# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

n = opt.num_samples
d = opt.num_nodes
degree = opt.exp_edges
low = -8.
high = 8.

chem_data = LinearGaussianColor(
                n=opt.num_samples,
                obs_data=opt.obs_data,
                d=d,
                graph_type="erdos-renyi",
                degree=2 * degree,
                sem_type=opt.sem_type,
                dataset_type="linear",
                noise_scale=opt.noise_sigma,
                data_seed=opt.data_seed,
                low=low, high=high
            )

ground_truth_W = chem_data.W
ground_truth_P = chem_data.P
ground_truth_L = chem_data.P.T @ chem_data.W.T @ chem_data.P
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
print(ground_truth_W)
print()

# ? generate linear gaussian colors
z, interv_targets, interv_values = generate_colors(opt, chem_data, low, high)
normalized_z = 255. * ((z / (2 * high)) + 0.5)

# ? Use above colors to generate images
images = generate_chem_image_dataset(n, d, interv_values, interv_targets, z)





