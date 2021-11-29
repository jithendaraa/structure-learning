import torch
from torchvision.utils import make_grid

import os
import pickle
from os.path import *
from pathlib import Path
import wandb
import numpy as np
import imageio
import graphical_models
import networkx as nx
import matplotlib.pyplot as plt
import math
from jax import numpy as jnp
from jax import random

def set_opts(opt):
    print()
    # Set logdir and create dirs along the way
    path_logdir = Path(opt.logdir)
    opt.logdir = path_logdir / opt.dataset / opt.phase / opt.model
    os.makedirs(opt.logdir, exist_ok=True)

    tb_logdir = set_tb_logdir(opt)
    os.makedirs(tb_logdir, exist_ok=True)

    if opt.dataset == 'clevr':
      opt.data_dir = 'datasets/CLEVR_v1.0/images'
    elif opt.dataset == 'mnist':
      opt.data_dir = 'datasets/MNIST'
      opt.resolution, opt.channels = 28, 1

    opt.data_dir = os.path.join(opt.storage_dir, opt.data_dir)

    # Set model_params_logdir and create dirs along the way
    opt.model_params_logdir = opt.logdir / opt.model_params_logdir
    
    os.makedirs(opt.model_params_logdir, exist_ok=True)
    print("model_params_logdir:", opt.model_params_logdir)
    print()

    if opt.num_nodes == 2:  opt.exp_edges = 0.8
    if opt.num_nodes <=4: opt.alpha_lambd = 10.
    else: opt.alpha_lambd = 1000.

    

    return opt

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


# For dataloading
# Dataloader unpackers: inf_generator, get_data_dict, get_dict_template, get_next_batch
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_data_dict(opt, dataloader):
  data_dict = dataloader.__next__()
  
  if opt.model in ['GraphVAE']: 
    res = {"observed_data": data_dict[0]}
    if opt.dataset in ['mnist']:
      res = {"observed_data": (data_dict[0]*2.0) - 1.0}
      res['train_len'] = 60000
      res['test_len'] = 10000
    return res
  
  return data_dict

def get_dict_template(opt):
  if opt.model in ['SlotAttention_img', 'VCN', 'VCN_img', 'Slot_VCN_img', 'GraphVAE', 'DIBS']:
    return {"observed_data": None, 'train_len': None, 'test_len': None}

def set_batch_dict(opt, data_dict, batch_dict):
  if opt.model in ['SlotAttention_img', 'VCN', 'VCN_img', 'Slot_VCN_img', 'GraphVAE', 'DIBS']:
    # Image reconstruction task
    batch_dict["observed_data"] = data_dict["observed_data"]
    batch_dict["data_to_predict"] = data_dict["observed_data"]


  return batch_dict
  
def get_next_batch(data_dict, opt):
    # device = get_device(data_dict["observed_data"])
    batch_dict = get_dict_template(opt)
    batch_dict = set_batch_dict(opt, data_dict, batch_dict)
    
    # input_t = data_dict["observed_data"].size()[1]
    # output_t = data_dict["data_to_predict"].size()[1]
    # total_t = input_t + output_t

    # Flow motion magnitude labels
    # batch_dict["timesteps"] = torch.tensor(np.arange(0, total_t) / total_t).to(device)
    # batch_dict["observed_tp"] = torch.tensor(batch_dict["timesteps"][:input_t]).to(device)
    # batch_dict["tp_to_predict"] = torch.tensor(batch_dict["timesteps"][input_t:]).to(device)

    return batch_dict


# Save model params
# Save model params every `ckpt_save_freq` steps as model_params_logdir/ID_00000xxxxx.pickle
def save_model_params(model, optimizers, opt, step, ckpt_save_freq):
  if step > 0 and ((step+1) % ckpt_save_freq == 0):
      padded_zeros = '0' * (10 - len(str(step)))
      padded_step = padded_zeros + str(step)

      ckpt_filename = os.path.join(opt.model_params_logdir, opt.ckpt_id + '_' + str(opt.batch_size) + '_' + str(opt.lr)[2:] + '_' + str(opt.steps) + '_' + str(opt.resolution))

      if opt.model in ['VCN']:
        ckpt_filename += '_(' + str(opt.num_nodes) + ')'
      elif opt.model in ['VCN_img']:
        ckpt_filename += '_(' + str(opt.num_nodes) + '-' + str(opt.chan_per_node) + ')'
      elif opt.model in ['Slot_VCN_img']:
        ckpt_filename += '_(' + str(opt.num_nodes) + '-' + str(opt.slot_size) + ')'

      ckpt_filename += '_' + padded_step + '.pickle'

      model_dict = {
          'step': step,
          'state_dict': model.state_dict(),
          'optimizer': [optimizer.state_dict() for optimizer in optimizers]}

      with open(ckpt_filename, 'wb') as handle:
          pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
          print(f"Saved model parameters at step {step} -> {ckpt_filename}")

def get_grad_norm(model_params, alt=False):
  total_norm = 0
  for p in model_params:
    if alt is False and p.grad is None:
      raise NotImplementedError("")
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item() ** 2
  total_norm = total_norm ** 0.5
  return total_norm

# Get device
def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = x.view([batch_size, -1] + list(x.size())[1:])
  channels, masks = torch.split(unstacked, [num_channels, 1], dim=-3)
  return channels, masks

def log_to_tb(opt, writer, loss_dict, step, pred_gt):
  if step % opt.loss_log_freq == 0:
    for key in loss_dict.keys():
        writer.add_scalar(key, loss_dict[key], step)
        
  if step > 0 and step % opt.media_log_freq == 0 and pred_gt is not None:
    pred_gt = torch.from_numpy(np.moveaxis(pred_gt, -1, -3)) / 255.0 # b, c, h, w
    nrow = int(len(pred_gt) // 2)
    grid = make_grid(pred_gt , nrow = nrow)
    writer.add_images("Reconstruction", grid, step, dataformats='CHW')

def log_to_wandb(opt, step, loss_dict, media_dict, pred_gt):
  if opt.off_wandb is False:  # Log losses and pred, gt videos
    if step % opt.loss_log_freq == 0:   wandb.log(loss_dict, step=step)

    if step > 0 and step % opt.media_log_freq == 0 and pred_gt is not None:
        media_dict['Pred_GT'] = [wandb.Image(m) for m in pred_gt]
        wandb.log(media_dict, step=step)
        print("Logged media")

def log_images_to_tb(opt, step, keys, values, writer, dataformats='NHWC'):
  assert len(keys) == len(values)
  if step > 0 and step % opt.media_log_freq == 0:
    for i in range(len(keys)):
      key, val = keys[i], values[i]
      writer.add_images(key, val, step, dataformats=dataformats)

def log_encodings_per_node_to_tb(opt, writer, enc_inp, step):
  d, chan_per_node, h, w = enc_inp.size()

  if step > 0 and step % opt.media_log_freq == 0:
    grid = make_grid(enc_inp.view(-1, 1, h, w), nrow=chan_per_node)
    writer.add_images("Encodings learned per node (rows)", grid, step, dataformats='CHW')

def set_tb_logdir(opt):

  if opt.model not in ['DIBS']:
    logdir = os.path.join(opt.logdir, opt.ckpt_id + '_' + str(opt.batch_size) + '_' + str(opt.lr) + '_' + str(opt.steps) + '_' + str(opt.resolution))
  else:
    logdir = os.path.join(opt.logdir, opt.ckpt_id)

  if opt.model in ['VCN', 'VAEVCN']:
    logdir += f'_({opt.num_nodes})_seed{opt.seed}_{opt.data_seed}_factorised{opt.factorised}_proj{opt.proj}{opt.proj_dims}_samples{opt.num_samples}_expedges{opt.exp_edges}'
  elif opt.model in ['VCN_img']:
    logdir += f'_({opt.num_nodes}-{opt.chan_per_node})_seed{opt.seed}_{opt.data_seed}_factorised{opt.factorised}_proj{opt.proj}{opt.proj_dims}_expedges{opt.exp_edges}'
  elif opt.model in ['Slot_VCN_img']:
    logdir += f'_({opt.num_nodes}-{opt.slot_size})_seed{opt.seed}_{opt.data_seed}_factorised{opt.factorised}_proj{opt.proj}{opt.proj_dims}_expedges{opt.exp_edges}'
  elif opt.model in ['DIBS']:
    logdir += f'_({opt.num_nodes})_seed{opt.seed}_{opt.data_seed}_proj{opt.proj}{opt.proj_dims}_samples{opt.num_samples}_expedges{opt.exp_edges}'
  elif opt.model in ['VAE_DIBS']:
    logdir += f'_({opt.num_nodes})_seed{opt.seed}_{opt.data_seed}_proj{opt.proj}{opt.proj_dims}_samples{opt.num_samples}_expedges{opt.exp_edges}_sftcnstrnt_{opt.soft_constraint}_dibsupdates{opt.num_updates}_knownED{opt.known_ED}'

  print("logdir:", logdir)
  return logdir

def log_enumerated_dags_to_tb(writer, logdir, opt):
  if opt.model in ['VCN']:
    dag_file = os.path.join(logdir, 'enumerated_dags.png')
  else:
    raise NotImplemented('Not implemented yet')
  enumerated_dag = np.asarray(imageio.imread(dag_file))
  writer.add_image('graph_structure(GT-pred)/All DAGs', enumerated_dag, 0, dataformats='HWC')


def is_mec(g1, g2):
  """
    Returns True if graph g1 is a Markov Equivalent Class of graph g2
  """
  g1 = graphical_models.DAG.from_nx(g1)
  g1_skeleton = g1.cpdag() ##Find the skeleton
  all_g1_mecs = g1_skeleton.all_dags() #Find all DAGs in MEC
  g2 = graphical_models.DAG.from_nx(g2)
  g2_skeleton = g2.cpdag() ##Find the skeleton
  all_g2_mecs = g2_skeleton.all_dags() #Find all DAGs in MEC
  return all_g1_mecs == all_g2_mecs

def log_dags(particles_g, gt_graph, opt, eshd_e, eshd_m):
  n_particles, num_nodes, _ = particles_g.shape
  predicted_adj_mat = np.array(particles_g)
  unique_graph_edge_list, graph_counts, mecs = [], [], []

  for adj_mat in predicted_adj_mat:
      graph_edges = list(nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph).edges())
      if graph_edges in unique_graph_edge_list:
          graph_counts[unique_graph_edge_list.index(graph_edges)] += 1
      else:
          unique_graph_edge_list.append(graph_edges)
          graph_counts.append(1)

  sampled_graphs = [nx.DiGraph() for _ in range(len(graph_counts))]
  for i in range(len(graph_counts)):
      graph = sampled_graphs[i]
      graph.add_nodes_from([0, num_nodes-1])
      for edge in unique_graph_edge_list[i]:  graph.add_edge(*edge)
      sampled_graphs[i] = graph
      mecs.append(is_mec(graph, gt_graph))

  dag_file = join((set_tb_logdir(opt)), 'sampled_dags.png')
  print(f'DiBS Predicted {len(graph_counts)} unique graphs from {n_particles} modes')

  nrows, ncols = int(math.ceil(len(sampled_graphs) / 5.0)), 5
  fig = plt.figure()
  fig.set_size_inches(ncols * 5, nrows * 5)
  count = 0
  for idx in range(len(sampled_graphs)):
      graph = sampled_graphs[idx]
      ax = plt.subplot(nrows, ncols, count+1)
      count += 1
      nx.draw(graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=30, node_color='#FFFF00')
      ax.set_xticks([])
      ax.set_yticks([])
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.spines['bottom'].set_visible(False)

      same_graph = (list(graph.edges()) == list(gt_graph.edges()))
      
      if same_graph is True: color='blue'
      elif mecs[idx] is True: color='red'
      else: color='black'

      if same_graph is True:
          ax.set_title(f'Freq: {graph_counts[idx]} | Ground truth', fontsize=23, color=color)
      else:
          ax.set_title(f'Freq: {graph_counts[idx]} | MEC: {mecs[idx]}', fontsize=23, color=color)

  plt.suptitle(f'Exp. SHD (empirical): {eshd_e:.3f} Exp. SHD (marginal mixture): {eshd_m:.3f}', fontsize=20)
  plt.tight_layout()
  plt.savefig(dag_file, dpi=60)
  # print( f'Saved sampled DAGs at {dag_file}' )
  plt.show()
  sampled_graph = np.asarray(imageio.imread(dag_file))
  return sampled_graph


def sample_initial_random_particles(key, n_particles, n_vars, n_dim=None, latent_prior_std=None):
    print('Sampling random z_particles....')
    if n_dim is None:   n_dim = n_vars 
    std = latent_prior_std or (1.0 / jnp.sqrt(n_dim))
    key, subk = random.split(key)
    z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std        
    return z