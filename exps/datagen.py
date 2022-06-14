import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from jax import numpy as jnp
import torch, pdb
from random import sample
from jax import random
from typing import Optional, Tuple, Union, cast


def single_node_interv_data(opt, n_interv_sets, no_interv_targets, target, model='dibs', interv_value=0.0, interv_node=None):
    """
        For every `n_interv_sets`, randomly choose an index idx_i in 0, 1.... (opt.num_nodes-1) where n = 
        Let n = interventional data points per set = num_interv_data / n_interv_sets 
        For every set in `n_interv_sets`, we generate n data points  correcsponding to intervention on node idx_i
        [TODO]
    """
    data_ = []
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_pts_per_set = int(num_interv_data / n_interv_sets)
    
    if model in ['dibs']: 
        interv_data = np.array(target.x_interv)
        assert num_interv_data <= len(target.x_interv[0][1])

    for i in range(n_interv_sets):
        idx_i = np.random.randint(0, opt.num_nodes)
        if interv_node is not None: idx_i = interv_node
        no_interv_targets[opt.obs_data + i * interv_data_pts_per_set : opt.obs_data + (i+1) * interv_data_pts_per_set, idx_i] = True
        print(f'Intervened node: {idx_i}')
        
        if model in ['dibs']:
            data_idxs = sample(range(len(interv_data[idx_i][1])), interv_data_pts_per_set)
            data_.append(interv_data[idx_i][1][np.array(data_idxs)])
        
        elif model in ['bcd']:
            interv_data = jnp.array(target.intervene_sem(target.W, interv_data_pts_per_set, opt.sem_type,
                                sigmas=[opt.noise_sigma], idx_to_fix=idx_i, value_to_fix=interv_value))
            data_.append(interv_data)

    return data_, no_interv_targets


def multi_node_interv_data(opt, n_interv_sets, no_interv_targets, target, model, interv_value=0.0):
    """
        Generates `n_interv_sets` sets of interventional data.
        Each set has `interv_data_per_node_per_set` interv. data points per node equally spread
        [TODO]
    """

    if model not in ['bcd']:
        raise NotImplementedError(f'No support for model {model} yet')

    data_ = []
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_pts_per_set = int(num_interv_data / n_interv_sets)
    if model in ['dibs']: interv_data = np.array(target.x_interv)

    for i in range(n_interv_sets):
        interv_k_nodes = np.random.randint(1, opt.num_nodes)
        intervened_node_idxs = np.random.choice(opt.num_nodes, interv_k_nodes, replace=False)
        print(f'Intervened nodes: {intervened_node_idxs}')

        interv_data = target.intervene_sem(target.W, interv_data_pts_per_set, opt.sem_type,
                                            sigmas=[opt.noise_sigma], idx_to_fix=intervened_node_idxs, 
                                            value_to_fix=interv_value)

        data_.append(interv_data)
        no_interv_targets[opt.obs_data + i * interv_data_pts_per_set : opt.obs_data + (i+1) * interv_data_pts_per_set, intervened_node_idxs] = True

    return data_, no_interv_targets


def generate_interv_data(opt, n_interv_sets, target, model='dibs', interv_node=None, interv_value=0.0):
    no_interv_targets = np.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational
    num_interv_data = opt.num_samples - opt.obs_data

    if num_interv_data > 0:
        if opt.interv_type == 'single':
            data_, no_interv_targets = single_node_interv_data(opt, n_interv_sets, no_interv_targets, target, model, interv_value, interv_node)
        elif opt.interv_type == 'multi':
            data_, no_interv_targets = multi_node_interv_data(opt, n_interv_sets, no_interv_targets, target, model, interv_value)
        
    data_, no_interv_targets = jnp.array(data_).reshape(num_interv_data, opt.num_nodes), jnp.array(no_interv_targets)
    return data_, no_interv_targets

def get_data(opt, n_intervention_sets, target, data_=None, model='dibs', interv_node=None, interv_value=0.0):
    if model == 'bcd':
        obs_data = target.simulate_sem(target.W, opt.obs_data, target.sem_type, noise_scale=opt.noise_sigma, dataset_type="linear")
        obs_data = cast(jnp.ndarray, obs_data)

    elif data_ is None:   obs_data = jnp.array(target.x)[:opt.obs_data]
    else:   obs_data = jnp.array(data_)[:opt.obs_data]

    num_interv_data = opt.num_samples - opt.obs_data

    if num_interv_data > 0:
        interv_data, no_interv_targets = generate_interv_data(opt, n_intervention_sets, target, model, interv_node, interv_value=interv_value)
        x = jnp.concatenate((obs_data, interv_data), axis=0)
    else:
        interv_data = None
        x = jnp.array(obs_data)
        no_interv_targets = jnp.zeros((opt.num_samples, opt.num_nodes)).astype(bool)

    if opt.proj == 'linear': 
        # P = jnp.array(10 * np.random.rand(opt.num_nodes, opt.proj_dims)) 
        if opt.identity_proj is True:
            P = jnp.eye(opt.proj_dims)
        projected_samples = x @ P
        print(f'Data matrix after linear projection from {opt.num_nodes} dims to {opt.proj_dims} dims: {projected_samples.shape}')  
        
        z_mean = jnp.mean(obs_data, axis=0)
        z_cov = jnp.cov(obs_data.T)
        print(f"Z Mean: {z_mean}")
        print(f"Det. Z Covariance: {jnp.linalg.det(z_cov)}")

        x_mean = jnp.mean(projected_samples, axis=0)
        x_cov = jnp.cov(projected_samples.T)
        print(f"X Mean: {x_mean}")
        print(f"Det. X Covariance: {jnp.linalg.det(x_cov)}")

    return obs_data, interv_data, x, no_interv_targets, projected_samples, z_mean, z_cov, P


def gen_data_from_dist(rng, q_z_mu, q_z_covar, num_samples, interv_targets, clamp=True):
    q_z_mu = jnp.expand_dims(q_z_mu, 0).repeat(num_samples, axis=0)
    q_z_covar = jnp.expand_dims(q_z_covar, 0).repeat(num_samples, axis=0)
    data = random.multivariate_normal(rng, q_z_mu, q_z_covar)
    if clamp is True:   data = jnp.where(interv_targets, 0.0, data)
    return data

