import torch
from torch.distributions import normal 
import torch.nn as nn
import torch.nn.functional as F

from modules.data import distributions
from modules.AutoregressiveBase import AutoregressiveBase
import vcn_utils, utils
import networkx as nx


class VAEVCN(nn.Module):
    def __init__(self, opt, num_nodes, sparsity_factor, gibbs_temp_init=10., device=None):
        super(VAEVCN, self).__init__()
        self.opt = opt
        self.num_nodes = num_nodes
        self.sparsity_factor = sparsity_factor
        self.gibbs_temp = gibbs_temp_init
        self.d = opt.proj_dims
        self.device = device
        self.baseline = 0.

        self.vae_params = []
        self.vcn_params = []

        # Initialise Autoregresive VCN or factorised VCN
        if not opt.no_autoreg_base:
            self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, device = device, temp_rsample = 0.1).to(device)
        else:
            NotImplemented('Have not implemented factorised version yet (only autoregressive works)')
        	# graph_dist = factorised_base.FactorisedBase(opt.num_nodes, device = device, temp_rsample = 0.1).to(device)

        # Encoder and decoder for the VAE
        self.build_encoder_decoder()

        # Learnable params for the gaussian prior on z
        self.prior_params = nn.ParameterList(
            [ nn.Parameter( torch.randn((self.opt.num_samples, num_nodes)), requires_grad=True) ,
              nn.Parameter( torch.randn((self.opt.num_samples, num_nodes)), requires_grad=True)]
        ).to(device)

        # Nets to predict Gaussian params of the approximate posterior q(z|x) 
        self.mean_net = nn.Sequential(nn.Linear(num_nodes, num_nodes)).to(device)
        self.logvar_net = nn.Sequential(nn.Linear(num_nodes, num_nodes)).to(device)

        self.vae_params += list(self.prior_params.parameters())
        self.vae_params += list(self.mean_net.parameters())
        self.vae_params += list(self.logvar_net.parameters())
        self.vcn_params += list(self.graph_dist.parameters())

        if self.opt.anneal:	self.gibbs_update = self._gibbs_update
        else:				self.gibbs_update = None
        self.init_gibbs_dist()
        print(f'Initialised VAEVCN with data projected from {opt.num_nodes} dims to {opt.proj_dims} dims')

    def build_encoder_decoder(self):
        if self.opt.proj in ['linear']:
            self.encoder = LinearED(self.d, self.opt.num_nodes, self.device)
            self.vae_params += list(self.encoder.parameters())
            if self.opt.known_ED is False:
                self.decoder = LinearED(self.opt.num_nodes, self.d, self.device)
                self.vae_params += list(self.decoder.parameters())

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
	
    def update_gibbs_temp(self, e):
        if self.gibbs_update is None:
            return 0
        else:
            self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)

    def forward(self, bge_model, step, gt_graph=None, interv_targets = None):
        # Encode high dimensinal data (d) into lower dims (num_nodes)
        x = self.inputs
        z = self.encoder(x)

        """
            Pass through VCN and calculate log likelihood as BGe score based on 
            z_gt (ground truth causal graph) in the latent space of opt.num_nodes
        """
        n_samples = self.opt.batch_size
        samples = self.graph_dist.sample([n_samples])
        predicted_G = vcn_utils.vec_to_adj_mat(samples, self.num_nodes)
        vcn_log_likelihood = bge_model.log_marginal_likelihood_given_g(w = predicted_G, interv_targets=interv_targets)
        
        # Calculate log posterior and log prior for VCN
        self.vcn_log_posterior = self.graph_dist.log_prob(samples)
        dagness = vcn_utils.expm(predicted_G, self.num_nodes)
        vcn_log_prior = - (self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(predicted_G, axis = [-1, -2]))

        # TODO get log prior
        self.z_prior = torch.distributions.normal.Normal(self.prior_params[0], torch.exp(0.5 * self.prior_params[1]))
        self.prior_sample = self.z_prior.rsample()

        # Sample z1...zN from posterior_zn_mean and posterior_zn_std
        z_mu_posterior, z_std_posterior  = self.mean_net(z), torch.exp(0.5 * self.logvar_net(z))
        
        # TODO get sampled_z and decode
        self.z_posterior = torch.distributions.normal.Normal(z_mu_posterior, z_std_posterior)
        
        # TODO FIX: zi should be function of parents, G, and mu_zi, std_zi. The next line will treat the latent vars z as if they were not connected. This not right, just writing this to run VAEVCN
        self.z_sample = self.z_posterior.rsample()
        if self.opt.known_ED is False: self.x_hat = self.decoder(self.z_sample)
        else: self.x_hat = torch.matmul(z, self.projection_matrix)

        self.vcn_terms = { 'll': vcn_log_likelihood, 'kl': self.vcn_log_posterior - vcn_log_prior }
        self.update_gibbs_temp(step)

    def get_prediction(self, loader_objs, step, interv_targets = None):
        bge_model = loader_objs['bge_train']
        self.projection_matrix = loader_objs['projection_matrix'].to(self.device)
        self.inputs = loader_objs['projected_data'].to(self.device)
        self(bge_model, step)

    def get_loss(self, step):
        """ VAE VCN can be trained by either opt.opti == 'simult' or 'alt' 
            When set to 'alt', we alternate between taking 1 step to optimize the latent variables z (VAE loss)
            and 1 step to optimize the graph structure G (VCN loss), and iterate in this manner.

            When set to 'simult', we jointly optimize latent variable and graph structure using a form of ELBO that combines VAE loss + VCN loss
            VCN ELBO: -log p(z_gt|G) + D_KL( q_phi(G) || p(G) ) .... (1)    log likelihood calculated as BGe Score
            VAE ELBO: -log p(x|z) + D_KL( q(z|x) || p_G(z) )    .... (2)    log likelihood calculated as MSE
            
            VAE_VCN ELBO (for simultaneous optimization)
            ------------
            (1) is a bound on p_G(z); using this in (2),
            combined ELBO: - log p(x|z) + D_KL( q(z|x) || p(z|G) ) + D_KL( q_phi(G) || p(G) ) .... (3)
        """

        if self.opt.opti == 'alt':
            # VCN
            vcn_elbo_loss = - self.vcn_terms['ll'] + self.vcn_terms['kl']
            score_val = vcn_elbo_loss.detach()
            per_sample_elbo = self.vcn_log_posterior * (score_val-self.baseline)
            vcn_loss = (per_sample_elbo).mean()
            
            # VAE
            mse_loss = F.mse_loss(self.x_hat, self.inputs)
            vae_kl = F.kl_div(self.z_posterior.log_prob(self.z_sample), self.z_prior.log_prob(self.prior_sample), log_target=True)
            vae_elbo_loss = (mse_loss + vae_kl).mean()

            if step % 2 == 0:   self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
            reqd_loss = int(step % 2 == 0) * vcn_loss + int(step % 2 == 1) * vae_elbo_loss
            
        elif self.opt.opti == 'simult':
            raise NotImplementedError("Simultaneous optimization not implemented yet")

        total_loss = vcn_elbo_loss.mean().item() + vae_elbo_loss.mean().item()

        loss_dict = {
            'graph_losses/Neg. log likelihood': self.vcn_terms['ll'].mean().item(),
			'graph_losses/KL loss':	self.vcn_terms['kl'].mean().item(),
			'graph_losses/Total loss': vcn_elbo_loss.mean().item(),
			'graph_losses/Per sample loss': per_sample_elbo.mean().item(),
            'vae_losses/Neg. log likelihood': mse_loss.mean().item(),
            'vae_losses/KL loss': vae_kl.mean().item(),
            'vae_losses/Total loss': vae_elbo_loss.mean().item(),
			'total_losses/Total loss': total_loss,
        }

        return reqd_loss, total_loss, loss_dict




         
class LinearED(nn.Module):
    def __init__(self, in_dims, out_dims, device):
        super(LinearED, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dims, out_dims),
        ).to(device)
    
    def forward(self, x):
        return self.layers(x)

