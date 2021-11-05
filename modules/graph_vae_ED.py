import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_BU(nn.Module):
    def __init__(self, opt, device):
        super(Encoder_BU, self).__init__()

        self.opt = opt
        self.device = device
        self.input_size = int( (self.opt.resolution**2) * self.opt.channels )

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU(),
            nn.Linear(256, 128),
        ).to(device)

    def forward(self, x):
        return self.fc(x)

class MLP_BU_n(nn.Module):
    def __init__(self, opt, device):
        super(MLP_BU_n, self).__init__() 

        dims_per_node = opt.dims_per_node

        self.mlp = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, dims_per_node)
        ).to(device)

        self.mu_tilde_net = nn.Sequential(
            nn.Linear(dims_per_node, dims_per_node)
        ).to(device)

        self.logvar_tilde_net = nn.Sequential(
            nn.Linear(dims_per_node, dims_per_node)
        ).to(device)
        
    def forward(self, x):
        node_specific_mlp = self.mlp(x)
        mu_tilde, logvar_tilde = self.mu_tilde_net(node_specific_mlp), self.logvar_tilde_net(node_specific_mlp)
        return mu_tilde, logvar_tilde

