import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from jax import numpy as jnp
import torch, pdb
from jax import random

def generate_interv_data(opt, n_interv_sets, target):
    data_ = []
    no_interv_targets = np.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational
    data_start, data_end = opt.obs_data, opt.num_samples
    
    num_interv_data = data_end - data_start
    interv_data_per_node = int(num_interv_data / opt.num_nodes)
    interv_data_per_node_per_set = int(interv_data_per_node / n_interv_sets)
    interv_data_per_set = int(interv_data_per_node_per_set * opt.num_nodes)    
    interv_data = np.array(target.x_interv)

    if num_interv_data > 0:
        for i in range(n_interv_sets):
            interv_targets = []
            for j in range(opt.num_nodes):
                interv_target = np.array([False] * opt.num_nodes).reshape(1, -1).repeat(interv_data_per_node_per_set, axis=0)
                interv_target[:, j] = True
                data_.append(interv_data[j][1][i*interv_data_per_node_per_set: (i+1)*interv_data_per_node_per_set])
                interv_targets.append(interv_target)
            interv_targets = np.array(interv_targets).reshape(interv_data_per_set, opt.num_nodes)
            no_interv_targets[data_start+i*interv_data_per_set : data_start+(i+1)*interv_data_per_set] = interv_targets

    data_, no_interv_targets = jnp.array(data_).reshape(num_interv_data, opt.num_nodes), jnp.array(no_interv_targets)
    return data_, no_interv_targets

def get_data(opt, n_intervention_sets, target):
    interv_data, no_interv_targets = generate_interv_data(opt, n_intervention_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    
    interv_data_per_node = int((opt.num_samples - opt.obs_data) / opt.num_nodes)
    interv_data_per_node_per_set = int(interv_data_per_node / n_intervention_sets)
    interv_data_per_set = int(interv_data_per_node_per_set * opt.num_nodes)    
    
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
        data = obs_data
        sample_mean = np.mean(data, axis=0)
        sample_covariance = jnp.array(torch.cov(torch.transpose(torch.tensor(np.array(data)), 0, 1)))

    return obs_data, interv_data, x, no_interv_targets, projected_samples, sample_mean, sample_covariance

def gen_data_from_dist(rng, q_z_mu, q_z_covar, num_samples, interv_targets):
    q_z_mu = jnp.expand_dims(q_z_mu, 0).repeat(num_samples, axis=0)
    q_z_covar = jnp.expand_dims(q_z_covar, 0).repeat(num_samples, axis=0)
    data = random.multivariate_normal(rng, q_z_mu, q_z_covar)
    data = jnp.where(interv_targets, 0.0, data)
    return data

