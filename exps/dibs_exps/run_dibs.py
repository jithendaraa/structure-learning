import sys
sys.path.append('..')
sys.path.append('../..')
import os, imageio, utils, datagen, pathlib, graphical_models
from os.path import join
from torch.utils.tensorboard import SummaryWriter

import ruamel.yaml as yaml
from jax import vmap, random, jit, grad
import networkx as nx
import numpy as np
import jax.numpy as jnp


# ? Parse args
configs = yaml.safe_load((pathlib.Path('../..') / 'configs.yaml').read_text())
opt = utils.load_yaml_dibs(configs, exp='dibs')
if opt.likelihood == 'linear': opt.datagen = 'linear'

# ? Set seeds
np.random.seed(0)
key = random.PRNGKey(123+opt.data_seed)

# ? Set logdir
logdir = utils.set_tb_logdir(opt)
dag_file = join(logdir, 'sampled_dags.png')
writer = SummaryWriter(join('..', '..', logdir))

n_intervention_sets = 10
num_interv_data = opt.num_samples - opt.obs_data
interv_data_per_set = int(num_interv_data / n_intervention_sets)  
n_steps = (opt.num_updates / n_intervention_sets) if num_interv_data > 0 else opt.num_updates

print()
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'Intervention sets {n_intervention_sets} with {interv_data_per_set} data points per intervention set')

if opt.likelihood == 'bge':
    from dibs_bge import run_dibs_bge
    run_dibs_bge(key, opt, n_intervention_sets, dag_file, writer)

elif opt.likelihood == 'linear':
    from dibs_linear import run_dibs_linear
    run_dibs_linear(key, opt, n_intervention_sets, dag_file, writer, logdir)

elif opt.likelihood == 'nonlinear':
    from dibs_nonlinear import run_dibs_nonlinear
    run_dibs_nonlinear(key, opt, n_intervention_sets, dag_file, writer)

