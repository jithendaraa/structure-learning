import os, sys, pdb, graphical_models, imageio, wandb
sys.path.append('../')
sys.path.append('../../')

import numpy as np
from jax import numpy as jnp
from jax import random
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join

from dibs_new.dibs.target import make_linear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
import datagen, utils
from eval_ import evaluate

def run_decoder_joint_dibs_linear(key, opt, logdir, n_interv_sets, dag_file, writer):
    exp_config = vars(opt)
    
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_per_set = int(num_interv_data / n_intervention_sets)  
    n_steps = opt.num_updates / n_intervention_sets if num_interv_data > 0 else opt.num_updates



