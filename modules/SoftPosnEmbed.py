import torch
import torch.nn as nn
import numpy as np



class SoftPositionEmbed(nn.Module):
    def __init__(self, resolution, hidden_size, device=None):
        super(SoftPositionEmbed, self).__init__()
        self.dense = nn.Linear(4, hidden_size).to(device)
        self.grid = build_grid(resolution).to(device) # 1, c, h, w
    
    def forward(self, x):
        dense_grid = self.dense(self.grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + dense_grid


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
    grid = np.expand_dims(grid, axis=0).astype(np.float32)
    res = np.concatenate([grid, 1.0 - grid], axis=-3)
    res = torch.from_numpy(res)
    return res