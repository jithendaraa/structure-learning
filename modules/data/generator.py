import numpy as np
import torch
import networkx as nx
PRESETS = ['chain', 'collider','fork', 'random']
NOISE_TYPES = ['gaussian', 'isotropic-gaussian', 'exponential', 'gumbel']
VARIABLE_TYPES = ['gaussian', 'non-gaussian', 'categorical']

class Generator(torch.utils.data.Dataset):

	""" Base class for generating different graphs and performing ancestral sampling"""

	def __init__(self, num_nodes, num_edges, noise_type, num_samples, noise_mu=0., mu_prior = None, sigma_prior = None, seed = None):
		self.num_nodes = num_nodes
		self.num_edges = num_edges
		self.noise_mu = noise_mu
		assert noise_type in NOISE_TYPES, 'Noise types must correspond to {} but got {}'.format(NOISE_TYPES, noise_type)
		self.noise_type = noise_type
		self.num_samples = num_samples
		self.mu_prior = mu_prior
		self.sigma_prior = sigma_prior
		if seed is not None:
			self.reseed(seed)
		if not "self.weighted_adjacency_matrix" in locals():
			self.sample_weights()
			self.build_graph()
	
	def reseed(self, seed = None):
		torch.manual_seed(seed)
		np.random.seed(seed)

	def __getitem__(self, index):
		raise NotImplementedError

	def build_graph(self):
		""" Initilises the adjacency matrix and the weighted adjacency matrix"""
		self.adjacency_matrix = nx.to_numpy_matrix(self.graph)
		self.weighted_adjacency_matrix = self.adjacency_matrix.copy()
		edge_pointer = 0
		for i in nx.topological_sort(self.graph):
			parents = list(self.graph.predecessors(i))
			if len(parents) == 0:
				continue
			else:
				for j in parents:
					self.weighted_adjacency_matrix[j, i] = self.weights[edge_pointer]
					edge_pointer += 1

	def init_sampler(self):
		if self.noise_type.endswith('gaussian'):
			# Identifiable
			if self.noise_type == 'isotropic-gaussian':
				noise_std= [self.noise_sigma]*self.num_nodes
			elif self.noise_type == 'gaussian':
				noise_std = np.linspace(0.1, 3., self.num_nodes)
			for i in range(self.num_nodes):
				self.graph.nodes[i]['sampler'] = torch.distributions.normal.Normal(self.noise_mu, noise_std[i])

		elif self.noise_type == 'exponential':
			noise_std= [self.noise_sigma]*self.num_nodes
			for i in range(self.num_nodes):
				self.graph.nodes[i]['sampler'] = torch.distributions.exponential.Exponential(noise_std[i])

	def sample_weights(self):
		"""Sample the edge weights"""
		if self.mu_prior is not None:
			self.weights = torch.distributions.normal.Normal(self.mu_prior, self.sigma_prior).sample([self.num_edges])

		else:
			dist = torch.distributions.uniform.Uniform(-5, 5)
			self.weights = torch.zeros(self.num_edges)
			for k in range(self.num_edges):
				sample = 0.
				while sample > -0.5 and sample < 0.5:
					sample = dist.sample()
					self.weights[k] = sample

	def sample(self, num_samples, graph = None, node = None, value = None):
		"""Sample observations given a graph
		num_samples: Scalar
		graph: networkx DiGraph
		node: If intervention is performed, specify which node
		value: value set to node after intervention

		Outputs: Observations [num_samples x num_nodes]
		"""
		if graph is None:	graph = self.graph
		samples = torch.zeros(num_samples, self.num_nodes)
		edge_pointer = 0
		actual_means, actual_vars = [0.] * self.num_nodes, [0.] * self.num_nodes
		sample_means, sample_vars = [0.] * self.num_nodes, [0.] * self.num_nodes

		for i in nx.topological_sort(graph):
			if i == node:	noise = torch.tensor([value]*num_samples)
			else:			noise = self.graph.nodes[i]['sampler'].sample([num_samples])
			parents = list(self.graph.predecessors(i))

			if self.noise_type.endswith('gaussian'):
				actual_mean = self.graph.nodes[i]['sampler'].loc
				actual_var = self.graph.nodes[i]['sampler'].scale ** 2
			else:
				NotImplementedError("Have not implemented for non-gaussian models")

			if len(parents) == 0:	
				samples[:,i] = noise
			else:					
				curr, actual_mean, actual_var = 0., 0., 0.

				for j in parents:
					if self.noise_type.endswith('gaussian'):
						# ? actual µ_i = µ_noise_i + sum edge_weight(k, i) * µ_j (j in pa(i))
						# ? actual variance σ^2_i = σ_noise_i^2 + sum (edge_weight(k, i) * σ_j (j in pa(i))) ** 2
						actual_mean += self.weighted_adjacency_matrix[j, i] * self.graph.nodes[j]['sampler'].loc
						actual_var += (self.weighted_adjacency_matrix[j, i] * self.graph.nodes[j]['sampler'].scale) ** 2

					curr += self.weighted_adjacency_matrix[j, i]*samples[:,j]
					edge_pointer += 1
				curr += noise
				samples[:, i] = curr
			
			actual_means[i] = actual_mean.item()
			actual_vars[i] = actual_var.item()
			sample_means[i] = samples[:, i].mean().item()
			# sample_vars[i] = torch.var(samples[:, i], dim=0, unbiased=False).item() # ? Don't use Bessel's correction
		
		sample_covariance = torch.cov(torch.transpose(samples, 0, 1))
		# TODO: Calculate and return `actual covariance` instead of np.sqrt(actual_vars)
		return samples, actual_means, np.sqrt(actual_vars), sample_means, sample_covariance # (num_samples * num_nodes)

	def intervene(self, num_samples, node = None, value = None):
		
		"""Perform intervention to obtain a mutilated graph"""

		if node is None:
			node = torch.randint(self.num_nodes, (1,))
		if value is None:
			#value = torch.distributions.uniform.Uniform(-5,5).sample()
			value = torch.tensor(2.0)

		mutated_graph = self.adjacency_matrix.copy()
		mutated_graph[:, node] = 0. #Cut off all the parents

		return self.sample(num_samples, nx.DiGraph(mutated_graph), node.item(), value.item()), node, value
	
	def __len__(self):
		return self.num_samples

