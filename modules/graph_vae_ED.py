import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_BU(nn.Module):
    def __init__(self, input_size, device='cuda'):
        super(Encoder_BU, self).__init__()

        self.device = device
        self.input_size = int( input_size[0] * input_size[1] * input_size[2] )

        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.input_size, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.fc(x)

#  MLP to get Ψ_tilde
class MLP_BU_n(nn.Module):
    def __init__(self, dims_per_node, device):
        super(MLP_BU_n, self).__init__() 

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
        return mu_tilde.unsqueeze(1), logvar_tilde.unsqueeze(1)


#  MLP to get Ψ_cap
class MLP_TD_n(nn.Module):
    def __init__(self, dims_per_node, parent_dims, device):
        super(MLP_TD_n, self).__init__() 

        self.dims = parent_dims

        self.mlp = nn.Sequential(
            nn.Linear(parent_dims, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, dims_per_node)
        ).to(device)

        self.mu_cap_net = nn.Sequential(nn.Linear(dims_per_node, dims_per_node)).to(device)
        self.logvar_cap_net = nn.Sequential(nn.Linear(dims_per_node, dims_per_node)).to(device)
        
    def forward(self, x):
        node_specific_mlp = self.mlp(x)
        mu_cap, logvar_cap = self.mu_cap_net(node_specific_mlp), self.logvar_cap_net(node_specific_mlp)
        return mu_cap.unsqueeze(1), logvar_cap.unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, dims, dataset, device):
        super(Decoder, self).__init__()
        self.device = device
        
        if dataset == 'cifar10':
            input_size = (3, 32, 32)
        elif dataset == 'chemdata':
            input_size = (1, 50, 50)
        else:
            NotImplementedError

        self.output_size = int( input_size[0] * input_size[1] * input_size[2])

        self.mlp = nn.Sequential(
            nn.Linear(dims, 256), nn.BatchNorm1d(256), nn.ELU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ELU(),
            nn.Linear(512, self.output_size)
        ).to(device)

        self.logvar_net = nn.Sequential(
            nn.Linear(self.output_size, self.output_size)
        )


    def forward(self, z):
        x_hat = self.mlp(z)
        return torch.sigmoid(x_hat), torch.exp(self.logvar_net(x_hat))