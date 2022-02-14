import sys
sys.path.append('..')

import os, imageio, utils, datagen, pathlib, graphical_models
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import ruamel.yaml as yaml
from jax import vmap, random, jit, grad
import networkx as nx
import numpy as np
import jax.numpy as jnp

from decoder_dibs_nonlinear import run_decoder_joint_dibs
from decoder_dibs_nonlinear_with_interv_data import run_decoder_joint_dibs_across_interv_data

# ? Parse args
configs = yaml.safe_load((pathlib.Path('..') / 'configs.yaml').read_text())
opt, exp_config = utils.load_yaml_dibs(configs)

# ? Set seeds
np.random.seed(0)
key = random.PRNGKey(123+opt.data_seed)
n_intervention_sets = 10

# ? Set logdir
logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))

num_interv_data = opt.num_samples - opt.obs_data
interv_data_per_set = int(num_interv_data / n_intervention_sets)  
n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates

print()
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'Intervention sets {n_intervention_sets} with {interv_data_per_set} data points per intervention set')

if opt.likelihood == 'nonlinear':
    if opt.across_interv is True:
        run_decoder_joint_dibs_across_interv_data(key, opt, logdir, n_intervention_sets, dag_file, writer, exp_config)
    else:
        run_decoder_joint_dibs(key, opt, logdir, n_intervention_sets, dag_file, writer)