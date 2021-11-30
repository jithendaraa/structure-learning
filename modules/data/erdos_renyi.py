import numpy as np
from .generator import Generator
import networkx as nx 
import graphical_models
import torch

class ER(Generator):
	"""Generate erdos renyi random graphs using networkx's native random graph builder
		Args: 
		num_nodes - Number of Nodes in the graph
		exp_edges - Expected Number of edges in Erdos Renyi graph
		noise_type - Type of exogenous variables
		noise_sigma - Std of the noise type
		num_sampels - number of observations
		mu_prior - prior of weights mean(gaussian)
		sigma_prior - prior of weights sigma (gaussian)
		seed - random seed for data
	"""
	def __init__(self, num_nodes, exp_edges = 1, noise_type='isotropic-gaussian', noise_sigma = 1.0, num_samples=1000, mu_prior = 2.0, sigma_prior = 1.0, seed = 10, project=None, proj_dims=10, noise_mu=0.0):
		self.noise_sigma = noise_sigma
		self.proj_dims = proj_dims
		p = float(exp_edges) / (num_nodes-1)
		acyclic, mmec, count  = 0, 0, 1
		
		# Sample a random directed graph;
		# if acyclic, set to true
		# if MEC(DAG) >= 2; set mmec true
		# break from while if acyclic and mmec 
		while not (acyclic and mmec):
			if exp_edges <= 2:
				self.graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			else:
				self.graph = nx.generators.random_graphs.gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			
			acyclic = expm_np(nx.to_numpy_matrix(self.graph), num_nodes) == 0
			if acyclic:	mmec = num_mec(self.graph) >=2
			count += 1

		super().__init__(num_nodes, len(self.graph.edges), noise_type, num_samples, noise_mu=noise_mu, mu_prior = mu_prior , sigma_prior = sigma_prior, seed = seed)
		self.init_sampler()
		self.samples = self.sample(self.num_samples)

		if project == 'linear': 
			self.projection_matrix = torch.rand(self.num_nodes, self.proj_dims)

			# * inv(PP_t) * PX_t
			P = self.projection_matrix.numpy()
			P_T = np.transpose(P)
			PP_T = np.matmul(P, P_T)
			PP_T_inv = np.linalg.inv(PP_T)
			self.true_encoder = np.matmul(P_T, PP_T_inv)
			self.true_decoder = P

			self.projected_samples = torch.matmul(self.samples, self.projection_matrix)
			print(f'Data matrix after linear projection from {num_nodes} dims to {proj_dims} dims: {self.projected_samples.size()}')

def matrix_poly_np(matrix, d):
	x = np.eye(d) + matrix/d
	return np.linalg.matrix_power(x, d)

def expm_np(A, m):
	expm_A = matrix_poly_np(A, m)
	h_A = np.trace(expm_A) - m
	return h_A

def num_mec(m):
	a = graphical_models.DAG.from_nx(m)
	skeleton = a.cpdag() ##Find the skeleton
	all_dags = skeleton.all_dags() #Find all DAGs in MEC
	return len(all_dags)
