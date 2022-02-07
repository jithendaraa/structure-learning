import sys, os
from jax import vmap, random, jit, grad
import networkx as nx
import numpy as np
import jax.numpy as jnp

sys.path.append(os.getcwd() + '/dibs_new')
sys.path.append(os.getcwd() + '/dibs_new/dibs')

from dibs_new.dibs.inference import MarginalDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.target import make_linear_gaussian_equivalent_model
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_marginal_likelihood

import datagen, utils

def run_dibs_bge_new(key, opt, n_intervention_sets, dag_file, writer):
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)
    
    target, model = make_linear_gaussian_equivalent_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, 
                mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    
    interv_data, no_interv_targets = datagen.generate_interv_data(opt, n_intervention_sets, target)
    obs_data = jnp.array(target.x)[:opt.obs_data]
    x = jnp.concatenate((obs_data, interv_data), axis=0)

    dibs = MarginalDiBS(x=obs_data, inference_model=model)
    key, subk = random.split(key)
    gs = dibs.sample(key=subk, n_particles=opt.n_particles, 
                        steps=opt.num_updates, 
                        callback_every=50, 
                        callback=dibs.visualize_callback())


    dibs_empirical = dibs.get_empirical(gs)
    dibs_mixture = dibs.get_mixture(gs) 
    print()

    for descr, dist, flag in [('DiBS empirical ', dibs_empirical, True), ('DiBS marginal ', dibs_mixture, False)]:
        eshd = expected_shd(dist=dist, g=target.g, unique=flag)     
        auroc = threshold_metrics(dist=dist, g=target.g, unique=flag)['roc_auc']
        negll = neg_ave_log_marginal_likelihood(dist=dist, 
                    eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood, 
                    x=target.x_ho, unique=flag)
        print(f'{descr} |  E-SHD: {eshd:4.1f}   AUROC: {auroc:5.2f}     neg. LL {negll:5.2f}')

