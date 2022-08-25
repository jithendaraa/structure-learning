import sys, pathlib, pdb, os
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

from tqdm import tqdm
import utils
import ruamel.yaml as yaml

from modules.ColorGen import LinearGaussianColor
from chem_datagen import generate_colors
from graphvae_utils import *
import envs, wandb
from jax import numpy as jnp
import numpy as onp
import jax.random as rnd
from jax import config, jit, lax, value_and_grad, vmap
from jax.tree_util import tree_map, tree_multimap
import haiku as hk

from models.GraphVAE_torch import GraphVAE
import torch
import torch.nn as nn
import graphical_models
import networkx as nx
from dag_utils import count_accuracy



configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
device = 'cuda'
dims_per_node = int(opt.total_dims // opt.num_nodes)

# ? Set seeds
onp.random.seed(opt.data_seed)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

n = opt.num_samples
d = opt.num_nodes
degree = opt.exp_edges
l_dim = d * (d - 1) // 2

if opt.do_ev_noise:
    noise_dim = 1
else:
    noise_dim = d

low = -8.
high = 8.

model = GraphVAE(opt.num_nodes, 
                dims_per_node, 
                opt.batches, 
                opt.dataset, 
                device).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high)
print(gt_W)
print()

logdir = utils.set_tb_logdir(opt)
log_gt_graph(gt_W, logdir, vars(opt), opt)
gt_L_elems = get_lower_elems(gt_L, d)

bs = opt.batches
num_batches = opt.num_samples // bs
images = images[:, :, :, 0][:, onp.newaxis, :, :]
torched_images = torch.from_numpy(onp.array(images))

gt_graph_clipped = jnp.where(jnp.abs(gt_W) > 0.3, 1, 0)
gt_graph_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(onp.array(gt_graph_clipped))).cpdag()


def train_batch(b, epoch):
    start_idx = b * bs
    end_idx = (b+1) * bs

    soft_adjacency_matrix = onp.zeros((d, d)) 
    adjacency_matrix = onp.zeros((d, d)) 

    optimizer.zero_grad()
    x_hat = model.forward(torched_images[start_idx:end_idx][:, ].to(device))
    loss = criterion(x_hat, torched_images[start_idx:end_idx].cuda())
    loss.backward(retain_graph=True)
    optimizer.step()

    for j in range(opt.num_nodes-1):
        soft_adjacency_matrix[j, j+1:] = model.gating_params[j].cpu().detach().numpy() 
        adjacency_matrix[j, j+1:] = torch.nn.functional.gumbel_softmax(
                                            torch.from_numpy(soft_adjacency_matrix[j, j+1:]), 
                                            tau=0.99 ** epoch, hard=True
                                    ).cpu().numpy()

    G_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(adjacency_matrix)).cpdag()
    
    stats = count_accuracy(gt_W, jnp.array(adjacency_matrix))
    stats['shd_c'] = gt_graph_cpdag.shd(G_cpdag)
    stats["auroc"] = auroc(adjacency_matrix[onp.newaxis, :], gt_W, 0.3)

    return x_hat, loss.item(), stats, adjacency_matrix


with tqdm(range(opt.num_steps)) as pbar:
    for epoch in pbar:

        with tqdm(range(num_batches)) as pbar2:
            for b in pbar2:
                
                x_hat, loss, stats, adjacency_matrix = train_batch(b, epoch)

                pbar2.set_postfix(
                    Epoch=f"{epoch}",
                    Batch=f"{b}/{num_batches}",
                    X_mse=f"{loss:.2f}",
                    SHD=f"{stats['shd']}", 
                    AUROC=f"{stats['auroc']:.3f}",
                )

        wandb_dict = {
            'X_MSE': loss, 
            "Evaluations/SHD": stats["shd"],
            "Evaluations/SHD_C": stats["shd_c"],
            "Evaluations/AUROC": stats["auroc"],
        }

        if opt.off_wandb is False:  
            plt.imshow(adjacency_matrix)
            plt.savefig(join(logdir, 'pred_w.png'))
            wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
            
            plt.figure()
            plt.imshow(x_hat[0, 0].cpu().detach().numpy()/255.)
            plt.savefig(join(logdir, 'pred_image.png'))
            wandb_dict["graph_structure(GT-pred)/Reconstructed image"] = wandb.Image(join(logdir, 'pred_image.png'))

            wandb.log(wandb_dict, step=epoch)

        pbar.set_postfix(
            Epoch=f"{epoch}",
            X_mse=f"{loss:.2f}",
            SHD=f"{stats['shd']}", 
            AUROC=f"{stats['auroc']:.3f}",
        )
