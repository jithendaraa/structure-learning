import sys
sys.path.append('..')
sys.path.append('../..')

import os, imageio, utils, datagen, pathlib, graphical_models
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import ruamel.yaml as yaml
from jax import vmap, random, jit, grad
import networkx as nx
import numpy as np
import jax.numpy as jnp

# ? Parse args
configs = yaml.safe_load((pathlib.Path('../..') / 'configs.yaml').read_text())
opt = utils.load_yaml_dibs(configs)
if opt.likelihood == 'linear': opt.datagen = 'linear'
exp_config = vars(opt)

# ? Set seeds
np.random.seed(0)
key = random.PRNGKey(123+opt.data_seed)

# ? Set logdir
logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', logdir))

n_intervention_sets = 10
num_interv_data = opt.num_samples - opt.obs_data
interv_data_per_set = int(num_interv_data / n_intervention_sets)  
n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates

print()
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'{n_intervention_sets} intervention sets with {interv_data_per_set} data points per intervention set')

if opt.likelihood == 'nonlinear':
    if opt.across_interv is True:
        from decoder_dibs_nonlinear_with_interv_data import run_decoder_joint_dibs_across_interv_data
        run_decoder_joint_dibs_across_interv_data(key, opt, logdir, dag_file, writer, exp_config, n_intervention_sets)
    else:
        from decoder_dibs_nonlinear import run_decoder_joint_dibs
        run_decoder_joint_dibs(key, opt, logdir, n_intervention_sets, dag_file, writer, exp_config)

elif opt.likelihood == 'linear':
    from decoder_dibs_linear import run_decoder_joint_dibs_linear
    run_decoder_joint_dibs_linear(key, opt, logdir, n_intervention_sets, dag_file, writer)