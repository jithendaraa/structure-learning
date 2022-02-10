import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from jax import numpy as jnp
import torch
from jax import random

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

def get_data(opt, n_intervention_sets, target):
    interv_data, no_interv_targets = generate_interv_data(opt, n_intervention_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    x = jnp.concatenate((obs_data, interv_data), axis=0)

    if opt.proj == 'linear': 
        projection_matrix = torch.rand(opt.num_nodes, opt.proj_dims)
        P = projection_matrix.numpy()
        P_T = np.transpose(P)
        PP_T = P @ P_T
        PP_T_inv = np.linalg.inv(PP_T)
        true_encoder = P_T @ PP_T_inv
        true_decoder = P
        projected_samples = x @ P
        print(f'Data matrix after linear projection from {opt.num_nodes} dims to {opt.proj_dims} dims: {projected_samples.shape}')  
        sample_mean = np.mean(x, axis=0)
        sample_covariance = jnp.array(torch.cov(torch.transpose(torch.tensor(np.array(x)), 0, 1)))

    return obs_data, interv_data, x, no_interv_targets, projected_samples, sample_mean, sample_covariance

def gen_data_from_dist(rng, q_z_mu, q_z_covar, num_samples):
    q_z_mu = jnp.expand_dims(q_z_mu, 0).repeat(num_samples, axis=0)
    q_z_covar = jnp.expand_dims(q_z_covar, 0).repeat(num_samples, axis=0)
    data = random.multivariate_normal(rng, q_z_mu, q_z_covar)
    return data

