import sys
sys.path.append('..')
import os, imageio, utils, datagen, pathlib, graphical_models
from os.path import join
from torch.utils.tensorboard import SummaryWriter

import ruamel.yaml as yaml
from jax import vmap, random, jit, grad
import networkx as nx
import numpy as np
import jax.numpy as jnp

# experiments
from dibs_bge_old import run_dibs_bge_old
from dibs_bge_new import run_dibs_bge_new
from dibs_nonlinear import run_dibs_nonlinear

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
# gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
# writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates

print()
print(f'Observational data: {opt.obs_data}')
print(f'Interventional data: {num_interv_data}')
print(f'Intervention sets {n_intervention_sets} with {interv_data_per_set} data points per intervention set')

full_train = False

if opt.likelihood == 'bge':
    run_dibs_bge_new(key, opt, n_intervention_sets, dag_file, writer)

elif opt.likelihood == 'nonlinear':
    run_dibs_nonlinear(key, opt, n_intervention_sets, dag_file, writer, full_train)

