import torch
import numpy as np

class BGe(torch.nn.Module):
	"""
	Pytorch implementation of Linear Gaussian-Gaussian Model (Continuous Gaussian data)
	Supports batched version.
	 Each variable distributed as Gaussian with mean being the linear combination of its parents 
	weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
	The parameter prior over (mu, lambda) of the joint Gaussian distribution (mean `mu`, precision `lambda`) over x is Gaussian-Wishart, 
	as introduced in 
		Geiger et al (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981
	Computation is based on
		Kuipers et al (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013 
	Note: 
		- (mu, Sigma) of joint is not factorizable into independent theta, but there exists a one-to-one correspondence.
		- lambda = Sigma^{-1}
		- assumes default diagonal parametric matrix T

		This pytorch implementation is based on Jax implementation from Lars Lorch and Jonas Rothfuss.
	"""
	def __init__(self, *,
			mean_obs,
			alpha_mu,
			alpha_lambd,
			data,
			device = "cpu"
			):
		"""
		mean_obs : [num_nodes] : Mean of each Gaussian distributed variable (Prior)
		alpha_mu : torch.float64 : Hyperparameter of Wishart Prior
		alpha_lambd : 
		data : [n_samples, num_nodes] If provided, precomputes the posterior parameter R
		"""
		super(BGe, self).__init__()

		self.mean_obs = torch.tensor(mean_obs)
		self.alpha_mu = alpha_mu
		self.alpha_lambd = alpha_lambd
		self.device = device

		if not isinstance(data, torch.DoubleTensor):
			data = torch.tensor(data).to(torch.float64)

		self.N, self.d = data.shape        
		# pre-compute matrices
		small_t = (self.alpha_mu * (self.alpha_lambd - self.d - 1)) / \
			(self.alpha_mu + 1)
		T = small_t * torch.eye(self.d)

		x_bar = data.mean(axis=0, keepdims=True)
		x_center = data - x_bar
		s_N = torch.matmul(x_center.t(), x_center)  # [d, d]

		# Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
		# the supplementary contains the correct term
		self.R = (T + s_N + ((self.N * self.alpha_mu) / (self.N + self.alpha_mu)) * \
		(torch.matmul((x_bar - self.mean_obs).t(), x_bar - self.mean_obs))).to(self.device)
		
		all_l = torch.arange(self.d)
		
		self.log_gamma_terms = (
		0.5 * (np.log(self.alpha_mu) - np.log(self.N + self.alpha_mu))
		+ torch.special.gammaln(0.5 * (self.N + self.alpha_lambd - self.d + all_l + 1))
		- torch.special.gammaln(0.5 * (self.alpha_lambd - self.d + all_l + 1))
		- 0.5 * self.N * np.log(np.pi)
		# log det(T_JJ)^(..) / det(T_II)^(..) for default T
		+ 0.5 * (self.alpha_lambd - self.d + 2 * all_l + 1) * \
		np.log(small_t)).to(self.device)
		
	def slogdet_pytorch(self, parents, R = None):
		"""
		Batched log determinant of a submatrix
		Done by masking everything but the submatrix and
		adding a diagonal of ones everywhere else for the 
		valid determinant
		"""

		if R is None:
			R = self.R.clone()
		batch_size = parents.shape[0]
		R = R.unsqueeze(0).expand(batch_size,-1,-1)
		parents = parents.to(torch.float64).to(self.device)
		mask = torch.matmul(parents.unsqueeze(2), parents.unsqueeze(1)).to(torch.bool) #[batch_size, d,d]
		R = torch.where(mask, R, torch.tensor([np.nan], device = self.device).to(torch.float64))
		submat = torch.where(torch.isnan(R), torch.eye(self.d, dtype = torch.float64).unsqueeze(0).expand(batch_size,-1,-1).to(self.device), R)
		return torch.linalg.slogdet(submat)[1]

	def log_marginal_likelihood_given_g_j(self, j, w):
		"""
		Computes node specific terms of BGe metric
		j : Node to compute the marginal likelihood. Marginal Likelihood decomposes over each node.
		w : [batch_size, num_nodes, num_nodes] : {0,1} adjacency matrix 
		"""

		batch_size = w.shape[0]
		isj = (torch.arange(self.d) == j).unsqueeze(0).expand(batch_size, -1).to(self.device)
		parents = w[:, :, j] == 1
		parents_and_j = parents | isj
		n_parents = (w.sum(axis=1)[:,j]).long()
		n_parents_mask = n_parents == 0
		_log_term_r_no_parents = - 0.5 * (self.N + self.alpha_lambd - self.d + 1) * torch.log(torch.abs(self.R[j, j]))

		_log_term_r = 0.5 * (self.N + self.alpha_lambd - self.d + n_parents[~n_parents_mask]) *\
						self.slogdet_pytorch(parents[~n_parents_mask])\
					- 0.5 * (self.N + self.alpha_lambd - self.d + n_parents[~n_parents_mask] + 1) *\
						self.slogdet_pytorch(parents_and_j[~n_parents_mask])     # log det(R_II)^(..) / det(R_JJ)^(..)
	
		log_term_r = torch.zeros(batch_size, dtype = torch.float64, device = self.device)
		log_term_r[n_parents_mask] = _log_term_r_no_parents
		log_term_r[~n_parents_mask] = _log_term_r
		return log_term_r + self.log_gamma_terms[n_parents]


	def log_marginal_likelihood_given_g(self, *, w, interv_targets=None):
		"""Computes log p(x | G) in closed form using conjugacy properties
			w:     [batch_size, num_nodes, num_nodes]	{0,1} adjacency marix
			interv_targets: [batch_size, num_nodes] boolean mask of whether or not a node was intervened on
					intervened nodes are ignored in likelihood computation
		"""
		batch_size = w.shape[0]
		if interv_targets is None:
			interv_targets = torch.zeros(batch_size,self.d).to(torch.bool)
		interv_targets = (~interv_targets).to(self.device)
		# sum scores for all nodes
		mll = torch.zeros(batch_size, dtype = torch.float64, device = self.device)
		for i in range(self.d):
			#print(self.log_marginal_likelihood_given_g_j(i, w)[interv_targets[:,i]]) 
			mll[interv_targets[:,i]] += self.log_marginal_likelihood_given_g_j(i, w)[interv_targets[:,i]]  ##TODO: Possible to use torch.vmap but should be okay for now 
		return mll