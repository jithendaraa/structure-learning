import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.normal import Normal

from modules.graph_vae_ED import Encoder_BU, Decoder
from modules.graph_vae_ED import MLP_BU_n as MLP_BU
from modules.graph_vae_ED import MLP_TD_n as MLP_TD


class GraphVAE(nn.Module):
    def __init__(self, opt, N, M, device):
        super(GraphVAE, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        
        self.opt = opt
        self.device = device
        self.N = N  # num_nodes
        self.M = M  # total latent dims
        self.D = int(M / N) # N'
        self.epoch = 0
        assert self.D == self.opt.dims_per_node

        # The mean params for the gumbel-softmax; we access μ(i, j) for i>j; lower triangle of the parameter matrix
        self.gs_logits = nn.Parameter( np.log(0.5) * torch.ones((self.N, self.N)) ).to(device)
        self.tau = 1

        # Bottom up networks: to predict μ_tilde and log Σ_tilde    
        self.encoder = Encoder_BU(opt, device)
        self.MLP_BU_n = nn.ModuleList([MLP_BU(opt, device) for _ in range(self.N)])

        # Top down network: to predict μ_hat and log Σ_hat  
        self.MLP_TD_n = nn.ModuleList([MLP_TD(opt, (self.N - 1 - n) * self.D, device) for n in range(self.N - 1)])

        self.decoder = Decoder(opt, device)

    def anneal(self):
        self.tau = 0.99 ** self.epoch
        

    def forward(self, x):
        self.anneal()
        N, c, b = self.N, {}, x.size()[0]
        mu_tildes, std_tildes = [], []
        prior_mu_hats, prior_std_hats = [torch.zeros((b, self.D)).to(self.device)], [torch.ones((b, self.D)).to(self.device)]
        posterior_mu, posterior_std = [], []

        z_prior, z_posterior = 1., 1.

        encoded_x = self.encoder(x.view(b, -1))

        # Get Ψ_tilde = (μ_tilde, log Σ_tilde) in bottom-up manner for n=1..N
        for i in range(self.N):
            mu_tilde, logvar_tilde = self.MLP_BU_n[i](encoded_x)
            mu_tildes.append(mu_tilde)
            std_tildes.append( (0.5 * logvar_tilde).exp() )

        """
            We have latent variables: z_0...z_(N-1)
            
            Priors:
                p(z_(N-1)) ~ N(0, I); p(z_n) ~ N(μ_hat, Σ_hat)    ......(1)
            
            Ψ_n_tilde = (μ_n_hat, Σ_n_hat) 
                    = MLP_TD_n( sum_i=n+1...N { c_i_n * z_i } )   ......(2) 
                    where z_i ~ prior p(z_i) (as given in 1)
        """

        for j in range(N-2, -1, -1):
            c_j = []
            weighted_parents = []

            # Traverse parents of node z_j (basically all z_i with i>j)
            for i in range(j+1, N):
                # Sample gating parameter from Gumbel Softmax
                c_ij = self.gs_logits[i][j] # gating parameter
                c_j.append(F.gumbel_softmax(c_ij, self.tau))
                
                # Use Prior means and std of z_i and sample * gating param.
                mean, std = prior_mu_hats[i-N], prior_std_hats[i-N]
                sampled_z_i = Normal(mean, std).rsample()   # sampled parent
                weighted_parents.append(sampled_z_i * c_ij)
            
            c[j] = c_j

            # Get prior(j) from c and Prior(j+1)..prior(N)
            weighted_parents = torch.stack(weighted_parents, dim=0).view(b, -1).to(self.device)
            mu_tilde, logvar_tilde = self.MLP_TD_n[j](weighted_parents)
            # import pdb; pdb.set_trace()
            prior_mu_hats = [mu_tilde] + prior_mu_hats
            prior_std_hats = [ (0.5 * logvar_tilde).exp() ] + prior_std_hats
        
        # print("gating variables c")
        # for key in c.keys():
        #     val = [v.item() for v in c[key]]
        #     print(f'{key}: {val}')

        assert len(prior_mu_hats) == len(mu_tildes)
        for i in range(N):
            mu_hat, mu_tilde = prior_mu_hats[i], mu_tildes[i]
            std_hat, std_tilde = prior_std_hats[i], std_tildes[i]

            # Get posteriors p(z|x, c)'s distribution params as Precision-weighed fusion of psi_hat and psi_tilde
            std_i = 1. / (std_hat ** (-2) + std_tilde ** (-2))
            mu_i = std_i * ( mu_hat * (std_hat ** (-2)) + mu_tilde * (std_tilde ** (-2)) )
            posterior_mu.append(mu_i)
            posterior_std.append(std_i)
        
        # Prior and post dists
        self.prior_dists = [Normal(mu, std) for mu, std in zip(prior_mu_hats, prior_std_hats)]
        self.post_dists = [Normal(mu, std) for mu, std in zip(posterior_mu, posterior_std)]

        # Get z1..zN prior samples and z1|x...zN|x post samples
        self.sampled_priors = [dist.rsample() for dist in self.prior_dists]
        self.sampled_posteriors = [dist.rsample() for dist in self.post_dists]

        # Full prior p(z|c) = prod_n=1..N( p(z_n | z_pa(n), c_{pa(n),n}) )
        for prior in self.sampled_priors: z_prior *= prior
        # Full posterior p(z|x, c) = prod_n=1..N( p(z_n | x, z_pa(n), c_{pa(n),n}) )
        for posterior in self.sampled_posteriors: z_posterior *= posterior

        pred_x = self.decoder(z_posterior).view(b, self.opt.channels, -1)
        pred_x = pred_x.view(b, self.opt.channels, self.opt.resolution, self.opt.resolution)

        return pred_x, z_posterior, z_prior

    def get_prediction(self, batch_dict, step):
        torch_clamp_convert = lambda x: torch.clamp(((x + 1) / 2) * 255.0, 0., 255.).to(torch.int8)
        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['data_to_predict'].to(self.device) # [-1, 1]
        epoch = 1 + int((step * self.opt.batch_size) // batch_dict[self.opt.phase + '_len'])
        
        if epoch > self.epoch: 
            self.epoch = epoch
            print(f'[Epoch {epoch}]')

        self.pred_x, self.z_posterior, self.z_prior = self.forward(self.input_images)
        return torch_clamp_convert(self.pred_x)

    def get_loss(self):
        # ELBO Loss = recon_loss + KL_loss
        recon_loss = F.mse_loss(self.pred_x, self.ground_truth) 
        
        prior_log_probs =  [dist.log_prob(value) for dist, value in zip(self.prior_dists, self.sampled_priors)]
        prior_log_probs = torch.stack(prior_log_probs, dim=0).sum(dim=0)
        post_log_probs =  [dist.log_prob(value) for dist, value in zip(self.post_dists, self.sampled_posteriors)]
        post_log_probs = torch.stack(post_log_probs, dim=0).sum(dim=0)
        kl_loss = F.kl_div(post_log_probs, prior_log_probs, log_target=True)

        total_loss = recon_loss + kl_loss

        loss_dict = {
            'Reconstruction loss': recon_loss.item(),
            'KL loss': kl_loss.item(),
            'Total loss': total_loss.item()
        }

        print(f'ELBO Loss: {total_loss.item()} | Reconstruction loss: {recon_loss.item()} | KL Loss: {kl_loss.item()}')

        return total_loss, loss_dict