import sys, pdb, wandb, os, imageio
from os.path import join

sys.path.append("../../models")
sys.path.append("../../modules")
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

import gym
from typing import OrderedDict
from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as onp
from jax import numpy as jnp
from sklearn.metrics import roc_curve, auc
from modules.ColorGen import LinearGaussianColor
from chem_datagen import generate_colors, generate_samples


def get_cifar10_dataset(batch_size):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/scratch', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='~/scratch', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def generate_chem_image_dataset(n, d, interv_values, interv_targets, z):
    images = None
    env = gym.make(f'LinGaussColorCubesRL-{d}-{d}-Static-10-v0')

    for i in tqdm(range(n)):
        action = OrderedDict()
        action['nodes'] = onp.where(interv_targets[i])
        action['values'] = interv_values[i]
        ob, _, _, _ = env.step(action, z[i])
        
        if i == 0:
            images = ob[1][jnp.newaxis, :]
        else:
            images = onp.concatenate((images, ob[1][jnp.newaxis, :]), axis=0)

    return images

def generate_data(opt, low=-8., high=8.):
    n = opt.num_samples
    d = opt.num_nodes

    if opt.generate:
        chem_data = LinearGaussianColor(
                        n=opt.num_samples,
                        obs_data=opt.obs_data,
                        d=opt.num_nodes,
                        graph_type="erdos-renyi",
                        degree=2 * opt.exp_edges,
                        sem_type=opt.sem_type,
                        dataset_type="linear",
                        noise_scale=opt.noise_sigma,
                        data_seed=opt.data_seed,
                        low=low, high=high
                    )
        gt_W = chem_data.W
        gt_P = chem_data.P
        gt_L = chem_data.P.T @ chem_data.W.T @ chem_data.P

        # ? generate linear gaussian colors
        z, interv_targets, interv_values = generate_colors(opt, chem_data, low, high)
        normalized_z = 255. * ((z / (2 * high)) + 0.5)

        # ? Use above colors to generate images
        images = generate_chem_image_dataset(opt.num_samples, opt.num_nodes, interv_values, interv_targets, z)
        onp.save(f'{opt.baseroot}/interv_values-seed{opt.data_seed}.npy', onp.array(interv_values))
        onp.save(f'{opt.baseroot}/interv_targets-seed{opt.data_seed}.npy', onp.array(interv_targets))
        onp.save(f'{opt.baseroot}/z-seed{opt.data_seed}.npy', onp.array(z))
        onp.save(f'{opt.baseroot}/images-seed{opt.data_seed}.npy', onp.array(images))
        onp.save(f'{opt.baseroot}/W-seed{opt.data_seed}.npy', onp.array(gt_W))
        onp.save(f'{opt.baseroot}/P-seed{opt.data_seed}.npy', onp.array(gt_P))


    else:
        interv_targets = jnp.array(onp.load(f'{opt.baseroot}/interv_targets-seed{opt.data_seed}.npy'))
        interv_values = jnp.array(onp.load(f'{opt.baseroot}/interv_values-seed{opt.data_seed}.npy'))
        z = jnp.array(onp.load(f'{opt.baseroot}/z-seed{opt.data_seed}.npy'))
        images = jnp.array(onp.load(f'{opt.baseroot}/images-seed{opt.data_seed}.npy'))
        gt_W = jnp.array(onp.load(f'{opt.baseroot}/W-seed{opt.data_seed}.npy'))
        gt_P = jnp.array(onp.load(f'{opt.baseroot}/P-seed{opt.data_seed}.npy'))
        gt_L = jnp.array(gt_P.T @ gt_W.T @ gt_P)

    print(gt_W)
    print()

    max_cols = jnp.max(interv_targets.sum(1))
    data_idx_array = jnp.array([jnp.arange(d + 1)] * n)
    interv_nodes = onp.split(data_idx_array[interv_targets], interv_targets.sum(1).cumsum()[:-1])
    interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([d] * (max_cols - len(interv_nodes[i])))))
        for i in range(n)]).astype(int)

    return z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L

def log_gt_graph(ground_truth_W, logdir, exp_config_dict, opt, writer=None):
    plt.imshow(ground_truth_W)
    plt.savefig(join(logdir, 'gt_w.png'))

    # ? Logging to wandb
    if opt.off_wandb is False:
        if opt.offline_wandb is True: os.system('wandb offline')
        else:   os.system('wandb online')
        
        wandb.init(project=opt.wandb_project, 
                    entity=opt.wandb_entity, 
                    config=exp_config_dict, 
                    settings=wandb.Settings(start_method="fork"))
        wandb.run.name = logdir.split('/')[-1]
        wandb.run.save()
        wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png'))}, step=0)

    # ? Logging to tensorboard
    gt_graph_image = onp.asarray(imageio.imread(join(logdir, 'gt_w.png')))
    if writer:
        writer.add_image('graph_structure(GT-pred)/Ground truth W', gt_graph_image, 0, dataformats='HWC')


def get_lower_elems(L, dim, k=-1):
    return L[jnp.tril_indices(dim, k=k)]

def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])

def auroc(Ws, W_true, threshold):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    _, dim, dim = Ws.shape
    edge_present = jnp.abs(Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc