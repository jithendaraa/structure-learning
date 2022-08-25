import sys, pdb
from matplotlib.pyplot import axis
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.graph_vae_ED import Encoder_BU, MLP_TD_n, MLP_BU_n, Decoder
from torch.distributions import MultivariateNormal, Bernoulli
torch.autograd.set_detect_anomaly(True)

class GraphVAE(nn.Module):
    def __init__(self, num_nodes, dims_per_node, batch_size, dataset, device):
        """
            num_nodes: N
            dims_per_node: N' = M / N
            dataset
        """
        super(GraphVAE, self).__init__()

        self.d = num_nodes
        self.dims_per_node = dims_per_node
        self.batch_size = batch_size

        if dataset == 'cifar10': 
            self.input_size = (3, 32, 32)
        elif dataset == 'chemdata':
            self.input_size = (1, 50, 50)

        # ? Bottom Up part
        self.encoder = Encoder_BU(self.input_size, device)
        self.MLP_BU = nn.ModuleList([
            MLP_BU_n(dims_per_node, device) for _ in range(num_nodes)
        ])

        # ? Initialize bernoulli means, all mu_ij, the gating paramerters
        # ? Only upper triangular elements matter; for row i, parents are [I+1...N]
        
        self.gating_params = nn.ParameterList([
            nn.Parameter(torch.empty(self.d - i - 1).fill_(0.5), requires_grad=True) for i in range(num_nodes) 
        ])

        # ? Top Down part
        self.MLP_TD = nn.ModuleList([
            MLP_TD_n(dims_per_node, int(dims_per_node * (self.d - 1 - i)), device) for i in range(num_nodes) 
        ])

        self.decoder = Decoder(int(dims_per_node * self.d), dataset, device)

        
    def precision_weighted_fusion(self, mu_1, mu_2, var_1, var_2):
        sigma = (var_1**(-1) + var_2**(-1)) ** -1
        mu = sigma * (mu_1 * (var_1**(-1))) + (mu_2 * (var_2**(-1)))
        return mu, sigma

    def sample_ancestors(self, begin_idx, prior_mu, prior_cov):
        res = None
        for i in range(begin_idx, self.d):
            mu = prior_mu[:, i].unsqueeze(1)
            covar = prior_cov[:, i].unsqueeze(1)

            if i == begin_idx:
                res = MultivariateNormal(mu, covar).rsample()
            else: 
                res_i = MultivariateNormal(mu, covar).rsample()
                res = torch.cat((res, res_i), 1)

        return res

    def forward(self, input_image, epoch=0):
        posterior_mu = torch.empty(self.batch_size, self.d, self.dims_per_node).fill_(0.0).cuda()
        posterior_cov = torch.eye(self.dims_per_node).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.d, 1, 1).cuda()
        prior_mu = torch.empty(self.batch_size, self.d, self.dims_per_node).fill_(0.0).cuda()
        prior_cov = torch.eye(self.dims_per_node).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.d, 1, 1).cuda()

        # * Bottom up calculcation
        mu_tilde, logvar_tilde = None, None
        common_embedding_BU = self.encoder(input_image)

        for i in range(self.d):
            if i == 0:
                mu_tilde, logvar_tilde = self.MLP_BU[i](common_embedding_BU)
            else:
                mu_tilde_i, logvar_tilde_i = self.MLP_BU[i](common_embedding_BU)
                mu_tilde = torch.cat((mu_tilde, mu_tilde_i), 1)
                logvar_tilde = torch.cat((logvar_tilde, logvar_tilde_i), 1)

        # * Top down calculation
        mu_hat, logvar_hat = None, None

        for i in range(self.d-2, -1, -1):
            if i == self.d-2: 
                mu = prior_mu[:, i+1].unsqueeze(1)
                cov = prior_cov[:, i+1].unsqueeze(1)
                ancestor_samples = MultivariateNormal(mu, cov).rsample().cuda()
            else:
                ancestor_samples = self.sample_ancestors(i+1, prior_mu, prior_cov)

            c_gates = F.gumbel_softmax(self.gating_params[i].unsqueeze(0).repeat(self.batch_size, 1), tau=0.99 ** epoch, hard=True)
            gates = c_gates.unsqueeze(2).repeat(1, 1, self.dims_per_node)
            z_parents = (gates * ancestor_samples).reshape(self.batch_size, -1)

            if mu_hat is None or logvar_hat is None:
                mu_hat, logvar_hat = self.MLP_TD[i](z_parents)
                var_hat = torch.exp(logvar_hat)

                prior_mu[:, i] = mu_hat[:, 0]
                for b in range(self.batch_size):
                    prior_cov[b, i] = torch.diag(var_hat[b, 0])

            else:
                mu_hat_i, logvar_hat_i = self.MLP_TD[i](z_parents)
                var_hat_i = torch.exp(logvar_hat_i)

                prior_mu[:, i] = mu_hat_i[:, 0]
                for b in range(self.batch_size):
                    prior_cov[b, i] = torch.diag(var_hat_i[b, 0])

                mu_hat = torch.cat((mu_hat_i, mu_hat), 1)
                logvar_hat = torch.cat((logvar_hat_i, logvar_hat), 1)

            posterior_mu[:, i], posterior_var = self.precision_weighted_fusion(mu_tilde[:, i, :], 
                                                                        mu_hat[:, 0, :], 
                                                                        torch.exp(logvar_tilde[:, i, :]), 
                                                                        torch.exp(logvar_hat[:, 0, :]))
            
            for b in range(self.batch_size):
                posterior_cov[b, i] = torch.diag(posterior_var[b, :])

        sampled_latents = None

        for i in range(self.d):
            mu = posterior_mu[:, i].unsqueeze(1)
            covar = posterior_cov[:, i].unsqueeze(1)

            if i == 0:
                sampled_latents = MultivariateNormal(mu, covar).rsample()
            else:
                sampled_latents_i = MultivariateNormal(mu, covar).rsample()
                sampled_latents = torch.cat((sampled_latents, sampled_latents_i), 1)
        
        sampled_latents = sampled_latents.reshape(self.batch_size, -1)
        x_mu, x_var = self.decoder(sampled_latents)

        x_cov = torch.ones((self.batch_size, x_var.shape[-1], x_var.shape[-1]))
        for b in range(self.batch_size):
            x_cov[b] = torch.diag(x_var[b])

        x_mu = x_mu.unsqueeze(1)
        x_cov = x_cov.cuda().unsqueeze(1)
        x_hat = MultivariateNormal(x_mu, x_cov).rsample().reshape(self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2])
        return x_hat