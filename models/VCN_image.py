import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.AutoregressiveBase import AutoregressiveBase
from modules.data import distributions
import vcn_utils as utils



class VCN_img(nn.Module):
    def __init__(self, opt, num_nodes, sparsity_factor, gibbs_temp, device=None):
        super(VCN_img, self).__init__()

        self.opt = opt
        self.datatype = opt.datatype
        self.num_nodes = num_nodes
        self.sparsity_factor = sparsity_factor
        self.gibbs_temp = gibbs_temp
        self.device = device
        self.baseline = 0.

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(opt.channels, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, opt.num_nodes * opt.chan_per_node, 3, 1, 1),
        ).to(device)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(opt.num_nodes * opt.chan_per_node, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, opt.channels, 4, 2, 1),
        ).to(device)

        if not opt.no_autoreg_base:
            self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, hidden_dim=opt.hidden_dims, device = device, temp_rsample = 0.1).to(device)
        else:
            NotImplemented('Have not implemented factorised version yet (only autoregressive works)')

        if self.opt.anneal:	self.gibbs_update = self._gibbs_update
        else:				self.gibbs_update = None
        
        self.init_gibbs_dist()
        print("Initialised model VCN_img")
    
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

    def forward(self, bge_model, e, interv_targets = None):
        x = self.input_images
        x = self.conv_encoder(x)
        b, c, h, w = x.size()
        x = x.view(-1, self.opt.num_nodes, self.opt.chan_per_node, h, w)
        # print("After encoding:", x.size())

        n_samples = self.opt.batch_size
        # print(f"inside VCN_img forward bsize(n_samples)")
        
        # 1. Sample L ( n*(n-1) ) graph structures autoregressively, per batch from q_phi
        # for i = 1..L: 
        #   G(i) ~ q_phi(G_i)
        #  samples has dimensions (b, L, 1)
        samples = self.graph_dist.sample([n_samples])
        # print("samples", samples.size())
        
        # 2. Get G = adjacency matrix of samples; batch_size, num_nodes, num_nodes
        G = utils.vec_to_adj_mat(samples, self.num_nodes)
        # print("G", G.size()) 
        # print()

        # 3. compute DAG constraint (tr[e^A(G)] - d) and get log prior
        dagness = utils.expm(G, self.num_nodes)
        # print("dagness", dagness.size())
        self.update_gibbs_temp(e)
        # lambda1 * dagness + lambda2 * || A(G) ||_1 (2nd term for sparsity)
		# Since graph prior is a gibbs distribution.
        log_prior_graph = - (self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]))

        # 4. Get approximate posterior_log_probs = approx. P(G | D) = q_phi_(G)
        posterior_log_probs = self.graph_dist.log_prob(samples).squeeze()
        # print("posterior lp", posterior_log_probs.shape)

        # 5. Get marginal log likelihood (after marginalising theta ~ normal-Wishart)
		# by getting log p(D | G) in closed form (BGe score)
        log_likelihood = bge_model.log_marginal_likelihood_given_g(G, interv_targets, x)
        # print("log_likelihood", log_likelihood.size())

		# 6. Get D_KL = KL (approx posterior || prior)
        kl_graph = posterior_log_probs - log_prior_graph
        # print("kl_graph", kl_graph.size())

        x_pred = self.conv_decoder(x.view(b, -1, h, w))
        
        return x[0], x_pred, log_likelihood, kl_graph, posterior_log_probs

    def get_prediction(self, batch_dict, bge_model, step):
        torch_clamp_convert = lambda x: torch.clamp(((x + 1) / 2) * 255.0, 0., 255.).to(torch.int8)
        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['data_to_predict'].to(self.device) # [-1, 1]
        enc_inp, self.pred, self.log_likelihood, self.kl_graph, self.posterior_log_probs = self(bge_model, step)
        return enc_inp, torch_clamp_convert(self.pred)

    def get_loss(self):
        # ELBO Loss for graph likelihood, posterior and prior
        neg_log_likeli, kl_loss = - self.log_likelihood, self.kl_graph
        score_val = ( neg_log_likeli + kl_loss ).detach()
        per_sample_elbo = self.posterior_log_probs*(score_val-self.baseline)
        self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
        graph_loss = (per_sample_elbo).mean()

        # MSE reconstruction loss for image
        recon_loss = F.mse_loss(self.pred, self.ground_truth) 
        
        # Graph loss + image loss
        total_loss = graph_loss + recon_loss

        loss_dict = {
	    	'Neg. log likelihood (G)': neg_log_likeli.mean().item(),
	    	'KL loss (G)':	kl_loss.mean().item(),
	    	'Graph loss (G)': (neg_log_likeli + kl_loss).mean().item(),
	    	'Per sample ELBO loss (G)': graph_loss.item(),
            'Reconstruction loss': recon_loss.item(),
            'Total loss (G + image)': total_loss.item()
	    }   

        print()
        print(f'Neg. log likelihood (G): {round(loss_dict["Neg. log likelihood (G)"], 2)} | KL loss (G): {round(loss_dict["KL loss (G)"], 2)} | Graph loss (G): {round(loss_dict["Graph loss (G)"], 2)} | Per sample ELBO loss (G): {round(loss_dict["Per sample ELBO loss (G)"], 2)} | Reconstruction loss: {round(loss_dict["Reconstruction loss"], 2)} | Total loss (G + image): {round(loss_dict["Total loss (G + image)"], 2)} ')

        return total_loss, loss_dict, self.baseline   

    def update_gibbs_temp(self, e): 
        if self.gibbs_update is None:
            return 0
        else:
            self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)


