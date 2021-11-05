import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.graph_vae_ED import Encoder_BU, MLP_BU_n
from modules.graph_vae_ED import MLP_BU_n as MLP_BU

class GraphVAE(nn.Module):
    def __init__(self, opt, N, M, device):
        super(GraphVAE, self).__init__()

        self.opt = opt
        self.device = device
        self.N = N
        self.M = M
        self.dims_per_node = M/ N # N'
        assert self.dims_per_node == self.opt.dims_per_node

        # Bottom up networks: to predict μ_tilde and Σ_tilde    
        self.encoder = Encoder_BU(opt, device)
        self.MLP_BU_n = MLP_BU_n(opt, device)


    def forward(self, x):
        print("Forward Graph VAE", x.size())

        # z_pa(n) = {z_(n+1),...z_N}
        # gating latent dependencies c(i, j)_i_j; c{i, j} denotes gate vairale from z_i to z_j (i>j) 
        # c takes binary value
        # P(z|c) = prod_n=1_N p(z_n | z_pa(n), c_{pa(n), n})


    def get_prediction(self, batch_dict):
        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.forward(self.input_images)

    def get_loss(self, batch_dict):
        # ELBO loss for the VAE
        
        pass

