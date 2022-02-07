import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from jax import numpy as jnp


def generate_interv_data(opt, n_intervention_sets, target):
    data_ = []
    no_interv_targets = np.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational
    data_start, data_end = opt.obs_data, opt.num_samples
    num_interv_data = data_end - data_start
    interv_data_per_set = int(num_interv_data / n_intervention_sets)    
    interv_data = np.array(target.x_interv)

    if num_interv_data > 0:
        for i in range(n_intervention_sets):
            interv_target = np.array([False] * opt.num_nodes)
            intervened_node = [*interv_data[i][0]][0]
            interv_target[intervened_node] = True
            start_idx, end_idx = int(data_start + (i * interv_data_per_set)), int(data_start + ((i+1) * interv_data_per_set)), 
            no_interv_targets[start_idx : end_idx, :] = interv_target[None, :]
            data_.append(interv_data[i][1][:interv_data_per_set])

    data_, no_interv_targets = jnp.array(data_).reshape(-1, opt.num_nodes), jnp.array(no_interv_targets)
    return data_, no_interv_targets