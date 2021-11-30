import sys
import os
from os.path import join

import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

os.chdir('vcn_pytorch')
import vcn_pytorch
import vcn_pytorch.main as main
os.chdir('..')

import os.path as osp
import numpy as np
from variance import Variance
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from dibs.eval.target import make_linear_gaussian_equivalent_model
from dibs.inference import MarginalDiBS
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.eval.metrics import expected_shd
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture

def parse_args():
	parser = argparse.ArgumentParser(description='Variational Causal Networks')
	parser.add_argument('--save_path', type=str, default = 'results/',
					help='Path to save result files')
	parser.add_argument('--no_autoreg_base', action='store_true', default=False,
					help='Use factorisable disrtibution')
	parser.add_argument('--seed', type=int, default=10,
					help='random seed (default: 10)')
	parser.add_argument('--data_seed', type=int, default=12,
					help='random seed for generating data(default: 20)')
	parser.add_argument('--batch_size', type=int, default=1000,
					help='Batch Size for training')
	parser.add_argument('--lr', type=float, default=1e-2,
					help='Learning rate')
	parser.add_argument('--gibbs_temp', type=float, default=1000.0,
					help='Temperature for the Graph Gibbs Distribution')
	parser.add_argument('--sparsity_factor', type=float, default=0.001,
					help='Hyperparameter for sparsity regularizer')
	parser.add_argument('--epochs', type=int, default=500,
					help='Number of iterations to train')
	parser.add_argument('--num_nodes', type=int, default=4,
					help='Number of nodes in the causal model')
	parser.add_argument('--num_samples', type=int, default=7000,
					help='Total number of samples in the synthetic data')
	parser.add_argument('--noise_type', type=str, default='isotropic-gaussian',
					help='Type of noise of causal model')
	parser.add_argument('--noise_sigma', type=float, default=1.0,
					help='Std of Noise Variables')
	parser.add_argument('--theta_mu', type=float, default=2.0,
					help='Mean of Parameter Variables')
	parser.add_argument('--theta_sigma', type=float, default=1.0,
					help='Std of Parameter Variables')
	parser.add_argument('--data_type', type=str, default='er',
					help='Type of data')
	parser.add_argument('--exp_edges', type=float, default=0.7,
					help='Expected number of edges in the random graph')
	parser.add_argument('--eval_only', action='store_true', default=False,
					help='Perform Just Evaluation')
	parser.add_argument('--anneal', action='store_true', default=False,
					help='Perform gibbs temp annealing')
	parser.add_argument('--budget', type=int, default=10,
					help='Number of possible experiments')
	parser.add_argument('--num_updates', type=int, default=500,
					help='Number of update steps for each interventional update')
	parser.add_argument('--acq', type=str, default='variance',
					help='Type of acquisition function to use {variance, random}')
	parser.add_argument('--h_latent', type=float, default=5.0,
					help='hyperparameter for the RBF kernel')
	parser.add_argument('--alpha_linear', type=float, default=0.1,
					help='inverse temperature parameter schedule of sigmoid')
	parser.add_argument('--algo', type=str, default='dibs',
					help='Algorithm to use {vcn, dibs}')
	args = parser.parse_args()
	args.data_size = args.num_nodes * (args.num_nodes-1)
	root = args.save_path
	list_dir = os.listdir(args.save_path)
	args.save_path = os.path.join(args.save_path, args.data_type + '_' + str(int(args.exp_edges)), str(args.num_nodes) + '_' + str(args.seed) + '_' + str(args.data_seed) + '_' + str(args.num_samples) + '_' + \
	  str(args.sparsity_factor) +'_' + str(args.gibbs_temp) + '_' + str(args.no_autoreg_base)) 
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
	if args.num_nodes == 2:
		args.exp_edges = 0.8
    
	args.gibbs_temp_init = 10.
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	return args

def main_dibs(args):
	key = random.PRNGKey(123)
	target = make_linear_gaussian_equivalent_model(key =  key, n_vars = args.num_nodes, graph_prior_str = args.data_type,
		obs_noise = args.noise_sigma, mean_edge = args.theta_mu, sig_edge = args.theta_sigma, n_observations = args.num_samples, n_ho_observations = args.num_samples)

	model = target.inference_model
	print(type(model).__name__)
	_, train_data = vcn_pytorch.main.load_data(args)
	x = jnp.array(train_data.samples) 
	no_interv_targets = jnp.zeros(args.num_nodes).astype(bool) # observational data

	def log_prior(single_w_prob):
		"""log p(G) using edge probabilities as G"""
		return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

	def log_likelihood(single_w):
		log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=x, interv_targets=no_interv_targets)
		return log_lik

	eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)	
    # SVGD + DiBS hyperparams
	n_particles = 20
	n_steps = args.num_updates
	# initialize kernel and algorithm
	kernel = FrobeniusSquaredExponentialKernel(
	h=args.h_latent)

	dibs = MarginalDiBS(
	kernel=kernel, 
	target_log_prior=log_prior,
	target_log_marginal_prob=log_likelihood,
	alpha_linear=args.alpha_linear)
		
	# initialize particles
	key, subk = random.split(key)
	init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=n_particles, n_vars=args.num_nodes)

	key, subk = random.split(key)
	particles_z = dibs.sample_particles(key=subk, n_steps=n_steps, init_particles_z=init_particles_z)

	particles_g = dibs.particle_to_g_lim(particles_z)
	print(particles_g)
	dibs_empirical = particle_marginal_empirical(particles_g)
	dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
	eshd_e = expected_shd(dist=dibs_empirical, g=train_data.adjacency_matrix)
	eshd_m = expected_shd(dist=dibs_mixture, g=train_data.adjacency_matrix)
	print("ESHD (empirical):", eshd_e)
	print("ESHD (marginal):", eshd_m)


if __name__ == "__main__":
	args = parse_args()
	main_dibs(args)