import sys, pdb, pathlib
sys.path.append("..")
import numpy as np


def generate_colors(opt, chem_data, low, high): 
    n = opt.num_samples
    d = opt.num_nodes
    n_interv_sets = opt.n_interv_sets
    interv_data_per_set = (opt.num_samples - opt.obs_data) // n_interv_sets
    obs_data = chem_data.obs_X

    interv_data = []

    interv_values = np.random.uniform(low=-5., high=5., size=(n, d))
    interv_targets = np.full((n, d), False)

    for i in range(n_interv_sets):
        interv_k_nodes = np.random.randint(1, d)
        intervened_node_idxs = np.random.choice(d, interv_k_nodes, replace=False)
        interv_targets[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set, intervened_node_idxs] = True
        interv_value = interv_values[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set]

        interv_data_ = chem_data.intervene_sem(chem_data.W, 
                                                interv_data_per_set, 
                                                opt.sem_type,
                                                sigmas=[opt.noise_sigma], 
                                                idx_to_fix=intervened_node_idxs, 
                                                values_to_fix=interv_value, 
                                                low=low, 
                                                high=high)
        if i == 0:  interv_data = interv_data_
        else: interv_data = np.concatenate((interv_data, interv_data_), axis=0)

    z = np.concatenate((obs_data, interv_data), axis=0)

    return z, interv_targets, interv_values