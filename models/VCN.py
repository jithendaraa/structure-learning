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
		self.datatype = opt.datatype

		if not opt.no_autoreg_base:
			self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, device = device, temp_rsample = 0.1).to(device)
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
			return self.opt.gibbs_temp_init + (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))
	
	def forward(self, bge_model, e, interv_targets = None):
		# print("Inside forward")
		n_samples = self.opt.batch_size

		# graph_dist is of class AutoregressiveBase()
		# Use init_input & init_state (learnable params) -> embed, rnn, project to get logits and state 
		# State used for further RNN iters; logits used to sample from Bernoulli(.)
		# Return all the saved samples of len T = n*(n-1) where n is number of edges
		samples = self.graph_dist.sample([n_samples]) # all 0 or 1s after sampling from Bern(.)
		# print("samples", samples.size())	

		# get init state and input; input -> RNN inputs: values=cat(samples, init_input) and init_state
		# feed x through Autoregressive base (embed + RNN + project);
		# get logits and get log_probs of posterior as Bern(logits).log_prob(value)
		# posterior_log_probs = approx. P(G | D) = q_phi_(G)
		posterior_log_probs = self.graph_dist.log_prob(samples).squeeze()
		# print("posterior lp", posterior_log_probs.shape)

		G = utils.vec_to_adj_mat(samples, self.num_nodes)
		# print("G", G.size()) 
		# print()

		# Computes marginal log likelihood (after mariginalising theta ~ normal-Wishart)
		# log p(D | G) in closed form (BGe score)
		log_likelihood = bge_model.log_marginal_likelihood_given_g(w = G, interv_targets=interv_targets)
		# print()
		# print("likelihood", log_likelihood.size(), log_likelihood.mean().item(), interv_targets)

		# computes DAG constraint (tr[e^A(G)] - d)
		dagness = utils.expm(G, self.num_nodes)
		# print("dagness", dagness.size())
		self.update_gibbs_temp(e)
		
		# lambda1 * dagness + lambda2 * || A(G) ||_1 (2nd term for sparsity)
		# Since graph prior is a gibbs distribution.
		log_prior_graph = - (self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]))
		
		# D_KL = KL (approx posterior || prior)
		kl_graph = posterior_log_probs - log_prior_graph 
		# print("kl_graph", kl_graph.size())
		
		return log_likelihood, kl_graph, posterior_log_probs

	def get_prediction(self, bge_train, e):
		self.log_likelihood, self.kl_graph, self.posterior_log_probs = self(bge_train, e)
	
	def get_loss(self):
		# ELBO Loss
		kl_loss = self.kl_graph
		score_val = ( - self.log_likelihood + kl_loss ).detach()
		per_sample_elbo = self.posterior_log_probs*(score_val-self.baseline)
		self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
		loss = (per_sample_elbo).mean()
		
		loss_dict = {
			'Neg. log likelihood loss': - self.log_likelihood.mean().item(),
			'KL loss':	kl_loss.mean().item(),
			'Total loss': (- self.log_likelihood + kl_loss).mean().item(),
			'Per sample loss': loss.item()
		}

		return loss, loss_dict, self.baseline

	def update_gibbs_temp(self, e):
		if self.gibbs_update is None:
			return 0
		else:
			self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)