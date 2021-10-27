import sys
sys.path.append('..')
import torch
import torch.nn as nn

from modules.AutoregressiveBase import AutoregressiveBase
from modules.data import distributions
import vcn_utils as utils

class VCN(nn.Module):
	def __init__(self, opt, num_nodes, sparsity_factor = 0.0, gibbs_temp_init = 10., device=None):
		super().__init__()
		self.opt = opt
		self.num_nodes = num_nodes
		self.sparsity_factor = sparsity_factor
		self.gibbs_temp = gibbs_temp_init
		self.baseline = 0.

		if not opt.no_autoreg_base:
			self.graph_dist = AutoregressiveBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)
		else:
			NotImplemented('Have not implemented factorised version yet (only autoregressive works)')
        	# graph_dist = factorised_base.FactorisedBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)

		if self.opt.anneal:	self.gibbs_update = self._gibbs_update
		else:				self.gibbs_update = None

		self.init_gibbs_dist()
		print("Initialised VCN")

	def init_gibbs_dist(self):
		if self.opt.num_nodes <= 4:
			self.gibbs_dist = distributions.GibbsDAGDistributionFull(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)
		else:
			self.gibbs_dist = distributions.GibbsUniformDAGDistribution(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)
		
		print("Got Gibbs distribution", self.gibbs_dist)

	def _gibbs_update(self, curr, epoch):
		if epoch < self.opt.steps*0.05:
			return curr
		else:
			return self.opt.gibbs_temp_init+ (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))
	
	def forward(self, bge_model, e, interv_targets = None):
		n_samples = self.opt.batch_size
		samples = self.graph_dist.sample([n_samples])	
		log_probs = self.graph_dist.log_prob(samples).squeeze()

		G = utils.vec_to_adj_mat(samples, self.num_nodes) 
		likelihood = bge_model.log_marginal_likelihood_given_g(w = G, interv_targets=interv_targets)
		dagness = utils.expm(G, self.num_nodes)

		self.update_gibbs_temp(e)
		kl_graph = log_probs + self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]) 
		return likelihood, kl_graph, log_probs

	def get_prediction(self, bge_train, e):
		self.likelihood, self.kl_graph, self.log_probs = self(bge_train, e)
	
	def get_loss(self):
		# ELBO Loss
		reconstruction_loss, kl_loss = - self.likelihood, self.kl_graph
		score_val = ( reconstruction_loss + kl_loss ).detach()
		per_sample_elbo = self.log_probs*(score_val-self.baseline)
		self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
		loss = (per_sample_elbo).mean()
		
		loss_dict = {
			'Reconstruction loss': reconstruction_loss.mean().item(),
			'KL loss':	kl_loss.mean().item(),
			'Total loss': (reconstruction_loss + kl_loss).mean().item(),
			'Per sample loss': loss.item()
		}

		return loss, loss_dict, self.baseline

	def update_gibbs_temp(self, e):
		if self.gibbs_update is None:
			return 0
		else:
			self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)