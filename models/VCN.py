import sys
sys.path.append('..')
import torch
import torch.nn as nn
import utils
from modules.AutoregressiveBase import AutoregressiveBase

class VCN(nn.Module):
	def __init__(self, opt, num_nodes, sparsity_factor = 0.0, gibbs_temp_init = 10., device=None):
		super().__init__()
		self.opt = opt
		self.num_nodes = num_nodes
		self.sparsity_factor = sparsity_factor
		self.gibbs_temp = gibbs_temp_init

		if not opt.no_autoreg_base:
			self.graph_dist = AutoregressiveBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)
		else:
			NotImplemented('Have not implemented factorised version yet (only autoregressive works)')
        	# graph_dist = factorised_base.FactorisedBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)

		if self.opt.anneal: 
			self.gibbs_update = self._gibbs_update
		else:
			self.gibbs_update = None

		print("Initialised VCN")

	def _gibbs_update(self, curr, epoch):
		if epoch < self.opt.steps*0.05:
			return curr
		else:
			return self.opt.gibbs_temp_init+ (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))
	
	def forward(self, n_samples, bge_model, e, curr, epoch, interv_targets = None):
		
		samples = self.graph_dist.sample([n_samples])	
		log_probs = self.graph_dist.log_prob(samples).squeeze()

	# 	G = utils.vec_to_adj_mat(samples, self.num_nodes) 
	# 	likelihood = bge_model.log_marginal_likelihood_given_g(w = G, interv_targets=interv_targets)

	# 	dagness = utils.expm(G, self.num_nodes)
	# 	self.update_gibbs_temp(e)
	# 	kl_graph = log_probs + self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]) 
	# 	return likelihood, kl_graph, log_probs


	# def update_gibbs_temp(self, e):
	# 	if self.gibbs_update is None:
	# 		return 0
	# 	else:
	# 		self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)