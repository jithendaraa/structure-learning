import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from jax import numpy as jnp

def generate_data(opt, logdir):
    train_dataloader = ER(num_nodes = opt.num_nodes, exp_edges = opt.exp_edges, 
                            noise_type = opt.noise_type, noise_sigma = opt.noise_sigma, 
                            num_samples = opt.num_samples, mu_prior = opt.theta_mu, 
                            sigma_prior = opt.theta_sigma, seed = opt.data_seed, 
                            project=opt.proj, proj_dims=opt.proj_dims, noise_mu=opt.noise_mu)

    nx.draw(train_dataloader.graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
    plt.savefig(join(logdir,'gt_graph.png'))

    return train_dataloader

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