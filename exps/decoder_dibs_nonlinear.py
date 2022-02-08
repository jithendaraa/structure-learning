import sys, os, imageio
sys.path.append('../')
from os.path import join
import datagen, utils

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from flax.training import train_state
import jax.numpy as jnp
from jax import vmap, random, jit, grad, device_put
from jax.experimental import optimizers
import jax.scipy.stats as dist
import optax


from models.Decoder_JointDiBS import Decoder_JointDiBS
from dibs_new.dibs.target import make_nonlinear_gaussian_model

def train_model():
    pass

def run_decoder_joint_dibs(key, opt, logdir, n_intervention_sets, dag_file, writer):
    num_interv_data = opt.num_samples - opt.obs_data
    n_steps = opt.num_updates
    interv_data_per_set = int(num_interv_data / n_intervention_sets)

    target, model = make_nonlinear_gaussian_model(key = key, n_vars = opt.num_nodes, 
                graph_prior_str = opt.datatype, edges_per_node = opt.exp_edges,
                obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    gt_graph = nx.from_numpy_matrix(target.g, create_using=nx.DiGraph)
    nx.draw(gt_graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
    plt.savefig(join(logdir,'gt_graph.png'))
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    print()
    print("Adjacency matrix")
    print(np.array(target.g))
    print()
    
    obs_data, interv_data, z_gt, no_interv_targets, x, z_means, z_covars = datagen.get_data(opt, n_intervention_sets, target)

    dibs = Decoder_JointDiBS(opt.num_nodes, opt.proj_dims, model, opt.alpha_linear, 
                                grad_estimator=opt.grad_estimator,
                                known_ED=opt.known_ED, 
                                linear_decoder=opt.linear_decoder)

    key, rng = random.split(key)
    dibs_optimizer = optimizers.rmsprop(opt.dibs_lr)
    opt_init, opt_update, get_params = dibs_optimizer
    
    state = train_state.TrainState.create(
        apply_fn=dibs.apply,
        params=dibs.init(rng, rng, jnp.array(z_gt))['params'],
        tx=optax.adam(opt.lr)
    )

    decoder_train_steps = opt.steps - opt.num_updates
    dibs_update = 0

    for step in range(opt.steps):
        s = time()
        multiple = step // decoder_train_steps
        even_multiple = ((multiple % 2) == 0)

        train_model()
    