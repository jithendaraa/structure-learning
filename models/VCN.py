import sys
from networkx.algorithms.operators.unary import reverse
sys.path.append('..')
import torch
import torch.nn as nn
import math

from os.path import join

from modules.AutoregressiveBase import AutoregressiveBase
from modules.data import distributions
import vcn_utils, utils
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


class VCN(nn.Module):
	def __init__(self, opt, num_nodes, sparsity_factor = 0.0, gibbs_temp_init = 10., device=None):
		super().__init__()
		self.opt = opt
		self.num_nodes = num_nodes
		self.sparsity_factor = sparsity_factor
		self.gibbs_temp = gibbs_temp_init
		self.baseline = 0.
		self.datatype = opt.datatype
		self.device = device

		self.adjacency_lists, self.pres_graph = [], {}
		self.dag_adj_matrices = []
		self.dag_graph_log_likelihoods = []
		self.get_all_digraphs()

		if not opt.no_autoreg_base:
			self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, device = device, temp_rsample = 0.1).to(device)
		else:
			NotImplemented('Have not implemented factorised version yet (only autoregressive works)')
        	# graph_dist = factorised_base.FactorisedBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)

		if self.opt.anneal:	self.gibbs_update = self._gibbs_update
		else:				self.gibbs_update = None

		self.init_gibbs_dist()
		print("Initialised VCN")

	def get_all_digraphs(self, only_dags=True):
		self.directed_graphs, self.dags, self.dag_adj_matrices = [], [], []
		self.adjacency_lists, self.pres_graph = [], {}
		all_possible_edges = self.num_nodes * (self.num_nodes - 1)

		for num_edges in range(all_possible_edges+1):
			adj_list = np.zeros((all_possible_edges), dtype=int).tolist()
			# Get all adjacecny lists for directed graphs
			self.solve(adj_list.copy(), num_edges, 0, len(adj_list)-1)

		self.adjacency_lists = np.array(self.adjacency_lists)
		self.directed_adj_matrices = vcn_utils.vec_to_adj_mat_np(self.adjacency_lists, self.num_nodes)

		print(f'Got all ({len(self.directed_adj_matrices)}) DiGraphs for {self.num_nodes} nodes')

		for adj_matrix in self.directed_adj_matrices:
			DG = nx.DiGraph()
			DG.add_nodes_from(np.arange(0, self.num_nodes).tolist())
			
			for i in range(self.num_nodes):
				for j in range(self.num_nodes):
					if adj_matrix[i][j] == 1.:
						DG.add_edge(i, j)
			self.directed_graphs.append(DG)
			is_acyclic = vcn_utils.expm_np(nx.to_numpy_matrix(DG), self.num_nodes)
			if is_acyclic == 0:	
				self.dags.append(DG)
				self.dag_adj_matrices.append(torch.from_numpy(adj_matrix).to(self.device))

	# Gets all adjacency lists for directed graph with n node and p edges and saves into self.adjacency lists
	# Each element in self.adjacency lists has num_edges*(num_edges - 1) edges from which we can get adj matrices
	def solve(self, adj_list, edges_left, start, end):
		if edges_left == 0:
			key = "".join(str(adj_list))
			if key not in self.pres_graph.keys():
				self.pres_graph[key] = True
				self.adjacency_lists.append(adj_list)
			return
		
		if start > end or edges_left < 0: return
		
		self.solve(adj_list.copy(), edges_left, start + 1, end)
		modif_adj_list = adj_list.copy()
		modif_adj_list[start] = 1
		self.solve(modif_adj_list.copy(), edges_left - 1, start + 1, end)
		return

	def init_gibbs_dist(self):
		if self.opt.num_nodes <= 4:
			self.gibbs_dist = distributions.GibbsDAGDistributionFull(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)
		else:
			self.gibbs_dist = distributions.GibbsUniformDAGDistribution(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)

	def _gibbs_update(self, curr, epoch):
		if epoch < self.opt.steps*0.05:
			return curr
		else:
			return self.opt.gibbs_temp_init + (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))
	
	def get_enumerated_dags(self, n_samples, bge_model, gt_graph, interv_targets=None):
		"""
			For all the enumerated DAGs, get likelihood scores and save it
		"""
		dag_file = join((utils.set_tb_logdir(self.opt)), 'enumerated_dags.png')
		self.dag_graph_log_likelihoods = []
		best_ll = -1e5
		print()
		print(f'{len(self.dag_adj_matrices)} DAGs found!')
		nrows, ncols = int(math.ceil(len(self.dag_adj_matrices) / 5.0)), 5

		with torch.no_grad():
			for graph in self.dag_adj_matrices:
				graph_log_likelihood = bge_model.log_marginal_likelihood_given_g(w = graph.unsqueeze(0).repeat(n_samples, 1, 1), interv_targets=interv_targets).mean().item()
				self.dag_graph_log_likelihoods.append(graph_log_likelihood)
				if graph_log_likelihood > best_ll:
					best_ll = graph_log_likelihood
		
		idxs = np.flip(np.argsort(self.dag_graph_log_likelihoods))
		fig = plt.figure()
		fig.set_size_inches(ncols * 4, nrows * 4)
		count = 0
		for idx in idxs:
			graph = self.dags[idx]
			ax = plt.subplot(nrows, ncols, count+1)
			count += 1
			if round(best_ll, 2) == round(self.dag_graph_log_likelihoods[idx], 2): color = '#00FF00' # neon green
			else: color = '#FFFF00' # yellow
			nx.draw(graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=30, node_color=color)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			same_graph = (list(graph.edges()) == list(gt_graph.edges()))
			mec = utils.is_mec(graph, gt_graph)

			if same_graph is True: color='blue'
			elif mec is True: color='red'
			else: color='black'

			if same_graph is True: 
				ax.set_title(f'LL: {self.dag_graph_log_likelihoods[idx]:.2f} \n Ground truth', fontsize=23, color=color)
			else:
				ax.set_title(f'LL: {self.dag_graph_log_likelihoods[idx]:.2f} \n MEC: {mec}', fontsize=23, color=color)
		
		plt.tight_layout()
		plt.savefig(dag_file, dpi=50)
		print( f'Saved enumarate DAG at {dag_file}' )
		plt.show()

	def forward(self, bge_model, e, gt_graph, interv_targets = None):
		n_samples = self.opt.batch_size
		
		if e == 0 and self.num_nodes < 5:
			self.get_enumerated_dags(n_samples, bge_model, gt_graph, interv_targets)
		# graph_dist is of class AutoregressiveBase()
		# Use init_input & init_state (learnable params) -> embed, rnn, project to get logits and state 
		# State used for further RNN iters; logits used to sample from Bernoulli(.)
		# Return all the saved samples of len T = n*(n-1) where n is number of edges
		samples = self.graph_dist.sample([n_samples]) # all 0 or 1s after sampling from Bern(.)

		# get init state and input; input -> RNN inputs: values=cat(samples, init_input) and init_state
		# feed x through Autoregressive base (embed + RNN + project);
		# get logits and get log_probs of posterior as Bern(logits).log_prob(value)
		# posterior_log_probs = approx. P(G | D) = q_phi_(G)
		posterior_log_probs = self.graph_dist.log_prob(samples)
		predicted_G = vcn_utils.vec_to_adj_mat(samples, self.num_nodes)
		# print()

		# Computes marginal log likelihood (after mariginalising theta ~ normal-Wishart)
		# log p(D | G) in closed form (BGe score)
		log_likelihood = bge_model.log_marginal_likelihood_given_g(w = predicted_G, interv_targets=interv_targets)
		# print()
		# print("likelihood", log_likelihood.size(), log_likelihood.mean().item(), interv_targets)

		# computes DAG constraint (tr[e^A(G)] - d)
		dagness = vcn_utils.expm(predicted_G, self.num_nodes)
		# print("dagness", dagness.size())
		self.update_gibbs_temp(e)
		
		# lambda1 * dagness + lambda2 * || A(G) ||_1 (2nd term for sparsity)
		# Since graph prior is a gibbs distribution.
		log_prior_graph = - (self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(predicted_G, axis = [-1, -2]))
		
		# D_KL = KL (approx posterior || prior)
		kl_graph = posterior_log_probs - log_prior_graph 
		# print("kl_graph", kl_graph.size())
		return log_likelihood, kl_graph, posterior_log_probs

	def get_sampled_graph_frequency_plot(self, bge_model, gt_graph, n_samples=1000, interv_targets = None):
		log_likelihoods, unique_graph_edge_list, graph_counts, mecs = [], [], [], []

		with torch.no_grad():
			samples = self.graph_dist.sample([n_samples])
			sampled_G_adj_mat = vcn_utils.vec_to_adj_mat(samples, self.num_nodes)
			log_likelihood = bge_model.log_marginal_likelihood_given_g(w = sampled_G_adj_mat, interv_targets=interv_targets)

		for adj_mat, ll in zip(sampled_G_adj_mat.cpu().numpy(), log_likelihood):
			graph_edges = list(nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph).edges())
			if graph_edges in unique_graph_edge_list:
				graph_counts[unique_graph_edge_list.index(graph_edges)] += 1
			else:
				log_likelihoods.append(ll.item())
				unique_graph_edge_list.append(graph_edges)
				graph_counts.append(1)

		sampled_graphs = [nx.DiGraph() for _ in range(len(graph_counts))]
		for i in range(len(graph_counts)):
			graph = sampled_graphs[i]
			graph.add_nodes_from([0, self.num_nodes-1])
			for edge in unique_graph_edge_list[i]:
				graph.add_edge(*edge)
			sampled_graphs[i] = graph
			mecs.append(utils.is_mec(graph, gt_graph))

		dag_file = join((utils.set_tb_logdir(self.opt)), 'sampled_dags.png')
		print(f'Predicted {len(graph_counts)} graphs by sampling from posterior {n_samples} times')

		nrows, ncols = int(math.ceil(len(sampled_graphs) / 5.0)), 5
		fig = plt.figure()
		fig.set_size_inches(ncols * 5, nrows * 5)
		count = 0
		gt_graph_edges = list(gt_graph.edges())
		gt_graph_edges.sort()

		for idx in range(len(sampled_graphs)):
			graph = sampled_graphs[idx]
			ax = plt.subplot(nrows, ncols, count+1)
			count += 1
			nx.draw(graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=30, node_color='#FFFF00')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.spines['bottom'].set_visible(False)

			pred_graph_edges = list(graph.edges())
			pred_graph_edges.sort()
			same_graph = (pred_graph_edges == gt_graph_edges)
			
			if same_graph is True: color='blue'
			elif mecs[idx] is True: color='red'
			else: color='black'

			if same_graph is True:
				ax.set_title(f'Freq: {graph_counts[idx]} | Ground truth \n LL: {log_likelihoods[idx]:.2f}', fontsize=23, color=color)
			else:
				ax.set_title(f'Freq: {graph_counts[idx]} | MEC: {mecs[idx]} \n LL: {log_likelihoods[idx]:.2f}', fontsize=23, color=color)
		
		plt.tight_layout()
		plt.savefig(dag_file, dpi=60)
		print( f'Saved sampled DAGs at {dag_file}' )
		plt.show()

		return dag_file

	def get_prediction(self, bge_train, e, gt_graph):
		self.log_likelihood, self.kl_graph, self.posterior_log_probs = self(bge_train, e, gt_graph)
	
	def get_loss(self):
		# ELBO Loss
		kl_loss = self.kl_graph
		score_val = ( - self.log_likelihood + kl_loss ).detach()
		per_sample_elbo = self.posterior_log_probs*(score_val-self.baseline)
		self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
		loss = (per_sample_elbo).mean()
		
		loss_dict = {
			'graph_losses/Neg. log likelihood': - self.log_likelihood.mean().item(),
			'graph_losses/KL loss':	kl_loss.mean().item(),
			'graph_losses/Total loss': (- self.log_likelihood + kl_loss).mean().item(),
			'total_losses/Total loss': (- self.log_likelihood + kl_loss).mean().item(),
			'graph_losses/Per sample loss': loss.item()
		}

		return loss, loss_dict, self.baseline

	def update_gibbs_temp(self, e):
		if self.gibbs_update is None:
			return 0
		else:
			self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)