import pathlib, pdb, sys, optax, wandb
from os.path import join
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../modules")
sys.path.append("../../models")

from typing import Optional, Tuple, Union, cast

import haiku as hk
import jax
import jax.random as rnd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as onp
import ruamel.yaml as yaml
import utils, datagen
from jax import config, grad, jit, lax, value_and_grad, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_multimap

config.update("jax_enable_x64", True)

from dag_utils import SyntheticDataset, count_accuracy
# Data generation procedure
from divergences import *
from eval_ import *
from loss_fns import *
from loss_fns import get_single_kl
from bcd_utils import *

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")
# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set logdir
logdir = utils.set_tb_logdir(opt)

# ? Defining type variables
LStateType = Optional[hk.State]
PParamType = Union[hk.Params, jnp.ndarray]
PRNGKey = jnp.ndarray

# ? Variables
dim = opt.num_nodes
n_data = opt.num_samples
degree = opt.exp_edges
do_ev_noise = opt.do_ev_noise
num_outer = opt.num_outer
s_prior_std = opt.s_prior_std
n_interv_sets = opt.n_interv_sets
calc_shd_c = False
sem_type = opt.sem_type
eval_eid = opt.eval_eid
num_bethe_iters = 20
num_interv_data = opt.num_samples - opt.obs_data
assert num_interv_data % n_interv_sets == 0

dataseeds = np.arange(1, 21).tolist()
num_edges_per_dag = [] 

for dataseed in dataseeds:
    # ? Set seeds
    onp.random.seed(0)
    rng_key = rnd.PRNGKey(dataseed)
    key = hk.PRNGSequence(42)

    sd = SyntheticDataset(
        n=n_data,
        d=opt.num_nodes,
        graph_type="erdos-renyi",
        degree=2 * degree,
        sem_type=opt.sem_type,
        dataset_type="linear",
        noise_scale=opt.noise_sigma,
        data_seed=dataseed,
    )

    ground_truth_W = sd.W
    ground_truth_P = sd.P
    ground_truth_L = sd.P.T @ sd.W.T @ sd.P
    ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)
    print(ground_truth_W)
    print()

    log_gt_graph(ground_truth_W, logdir, exp_config, opt)

    binary_gt_dag = onp.where(onp.abs(ground_truth_W) >= 0.3, 1, 0)
    num_edges = onp.sum(binary_gt_dag)
    num_edges_per_dag.append(num_edges)
    # plt.imshow(ground_truth_W)
    # plt.savefig(f'gt_graphs/images/d{opt.num_nodes}_expedge{int(opt.exp_edges)}_seed{dataseed}.png')
    print(f'Number of edges for d={opt.num_nodes}, exp_edges={opt.exp_edges}, seed={dataseed}: {num_edges}')
    print()
    plt.close('all')

print(f"Mean edges across {len(dataseeds)} seeds:", np.mean(np.array(num_edges_per_dag)))
print("Edges per seed:", num_edges_per_dag)