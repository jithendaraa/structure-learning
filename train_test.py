import sys
from jax._src.random import multivariate_normal
from networkx.readwrite.json_graph import adjacency
sys.path.append('models')

from networkx.linalg.graphmatrix import adjacency_matrix
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.VCN_image import VCN_img
import utils
import wandb
import os
from os.path import join
import time
import numpy as np
import imageio
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import vcn_utils

import jax.numpy as jnp
from jax import vmap, random
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.eval.metrics import expected_shd
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture
import networkx as nx
import math
import matplotlib.pyplot as plt
import optax
import jax
from time import time
import jax.scipy.stats as dist


def set_optimizers(model, opt):
    if opt.opti == 'alt': 
        return [optim.Adam(model.vcn_params, lr=1e-2), optim.Adam(model.vae_params, lr=opt.lr)]
    return [optim.Adam(model.parameters(), lr=opt.lr)]

def train_dibs(target, loader_objs, opt, key):
    # ! Tensorboard setup and log ground truth causal graph
    logdir = utils.set_tb_logdir(opt)
    writer = SummaryWriter(logdir)
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    model = target.inference_model
    gt_graph = loader_objs['train_dataloader'].graph
    adjacency_matrix = loader_objs['adj_matrix']

    data = loader_objs['data'] 
    x = jnp.array(data) 
    no_interv_targets = jnp.zeros(opt.num_nodes).astype(bool) # observational data

    def log_prior(single_w_prob):
        """log p(G) using edge probabilities as G"""    
        return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(single_w):
        log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=x, interv_targets=no_interv_targets)
        return log_lik

    eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)	
    # ? SVGD + DiBS hyperparams
    n_particles, n_steps = opt.n_particles, opt.num_updates
    
	# ? initialize kernel and algorithm
    kernel = FrobeniusSquaredExponentialKernel(h=opt.h_latent)
    dibs = MarginalDiBS(kernel=kernel, target_log_prior=log_prior, 
                        target_log_marginal_prob=log_likelihood, 
                        alpha_linear=opt.alpha_linear)
		
	# ? initialize particles
    key, subk = random.split(key)
    init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=n_particles, n_vars=opt.num_nodes)

    key, subk = random.split(key)
    particles_z = dibs.sample_particles(key=subk, n_steps=n_steps, init_particles_z=init_particles_z)
    particles_g = dibs.particle_to_g_lim(particles_z)

    dibs_empirical = particle_marginal_empirical(particles_g)
    dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
    eshd_e = expected_shd(dist=dibs_empirical, g=adjacency_matrix)
    eshd_m = expected_shd(dist=dibs_mixture, g=adjacency_matrix)
    
    predicted_adj_mat = np.array(particles_g)
    unique_graph_edge_list, graph_counts, mecs = [], [], []

    for adj_mat in predicted_adj_mat:
        graph_edges = list(nx.from_numpy_matrix(adj_mat).edges())
        if graph_edges in unique_graph_edge_list:
            graph_counts[unique_graph_edge_list.index(graph_edges)] += 1
        else:
            unique_graph_edge_list.append(graph_edges)
            graph_counts.append(1)

    sampled_graphs = [nx.DiGraph() for _ in range(len(graph_counts))]
    for i in range(len(graph_counts)):
        graph = sampled_graphs[i]
        graph.add_nodes_from([0, opt.num_nodes-1])
        for edge in unique_graph_edge_list[i]:  graph.add_edge(*edge)
        sampled_graphs[i] = graph
        mecs.append(utils.is_mec(graph, gt_graph))

    dag_file = join((utils.set_tb_logdir(opt)), 'sampled_dags.png')
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
    print( f'Saved sampled DAGs at {dag_file}' )
    plt.show()

    sampled_graph = np.asarray(imageio.imread(dag_file))
    writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, 0, dataformats='HWC')

    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)

def train_vae_dibs(model, loader_objs, opt, key):
    from flax.training import train_state

    # ! Tensorboard setup and log ground truth causal graph
    logdir = utils.set_tb_logdir(opt)
    writer = SummaryWriter(logdir)
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    gt_graph = loader_objs['train_dataloader'].graph
    adjacency_matrix = loader_objs['adj_matrix']
    data = loader_objs['projected_data']

    x = jnp.array(data) 

    key, rng = random.split(key)
    m = model()

    state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=m.init(rng, x, key, adjacency_matrix)['params'],
        tx=optax.adam(opt.lr),
    )

    particles_g, eltwise_log_prob = None, None

    @jax.jit
    def train_step(state, batch, z_rng):
        def loss_fn(params):
            recons, log_p_z_given_g, q_zs, q_z_mus, q_z_logvars, particles_g, eltwise_log_prob = m.apply({'params': params}, batch, z_rng, adjacency_matrix)

            mse_loss = 0.
            for recon in recons:
                err = recon - x
                mse_loss += jnp.mean(jnp.square(err))

            kl_z_loss = 0.
            # ? get KL( q(z | G) || P(z|G_i) )
            for i in range(len(log_p_z_given_g)):
                cov = jnp.diag(jnp.exp(0.5 * q_z_logvars[i]))
                q_z = dist.multivariate_normal.pdf(q_zs[i], q_z_mus[i], cov)
                log_q_z = dist.multivariate_normal.logpdf(q_zs[i], q_z_mus[i], cov)
                kl_z_loss += q_z * (log_q_z - log_p_z_given_g[i])

            soft_constraint = 0.
            if opt.soft_constraint is True:
                for g in particles_g:
                    soft_constraint += jnp.linalg.det(jnp.identity(opt.num_nodes) - g)

            loss = mse_loss + kl_z_loss - soft_constraint
            return loss

        loss = loss_fn(state.params)
        grads = jax.grad(loss_fn)(state.params)
        res = state.apply_gradients(grads=grads)
        return res, loss

    for step in range(opt.steps):
        key, rng = random.split(key)
        state, loss = train_step(state, x, rng)

        if step % 1 == 0:
            s = time()

            _, _, _, _, _, particles_g, eltwise_log_prob = m.apply({'params': state.params}, x, rng, adjacency_matrix)
            dibs_empirical = particle_marginal_empirical(particles_g)
            dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
            eshd_e = expected_shd(dist=dibs_empirical, g=adjacency_matrix)
            eshd_m = expected_shd(dist=dibs_mixture, g=adjacency_matrix)
            
            print(f'Calc SHD took {time() - s}s')
            print(f'Step {step} | Loss {loss}')
            print(f'Expected SHD Marginal: {eshd_m} | Empirical: {eshd_e}')
            print()

def train_model(model, loader_objs, exp_config_dict, opt, device, key=None):
    # * DIBS, or a variant thereof, uses jax so train those in a separate function. This function trains only torch models
    if opt.model in ['DIBS']: 
        train_dibs(model, loader_objs, opt, key)
        return
    elif opt.model in ['VAE_DIBS']:
        train_vae_dibs(model, loader_objs, opt, key)
        return

    pred_gt, time_epoch, likelihood, kl_graph, elbo_train, vae_elbo = None, [], [], [], [], []
    optimizers = set_optimizers(model, opt)
    logdir = utils.set_tb_logdir(opt)
    writer = SummaryWriter(logdir)
    
    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict, settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        wandb.watch(model)

    start_time = time.time()
    for step in tqdm(range(opt.steps)):
        start_step_time = time.time()
        pred, gt, loss, loss_dict, media_dict = train_batch(model, loader_objs, optimizers, opt, writer, step, start_time)
        time_epoch.append(time.time() - start_step_time)

        if opt.model in ['SlotAttention_img', 'VCN_img', 'Slot_VCN_img', 'GraphVAE']:
            pred_gt = torch.cat((pred[:opt.log_batches].detach().cpu(), gt[:opt.log_batches].cpu()), 0).numpy()
            pred_gt = np.moveaxis(pred_gt, -3, -1)
        
        elif opt.model in ['VCN']:
            vae_elbo = evaluate_vcn(opt, writer, model, loader_objs['bge_train'], step+1, vae_elbo, device, loss_dict, time_epoch, loader_objs['train_dataloader'])

        # logging to tensorboard and wandb
        utils.log_to_tb(opt, writer, loss_dict, step, pred_gt)
        utils.log_to_wandb(opt, step, loss_dict, media_dict, pred_gt)
        utils.save_model_params(model, optimizers, opt, step, opt.ckpt_save_freq) # Save model params

    if opt.model in ['Slot_VCN_img']:
        utils.log_enumerated_dags_to_tb(writer, logdir, opt)

    elif opt.model in ['VCN']:
        # Log ground truth graph, enumerated DAGs and posterior sampled graphs
        gt_graph = loader_objs['train_dataloader'].graph
        gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
        writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
        
        utils.log_enumerated_dags_to_tb(writer, logdir, opt)

        dag_file = model.get_sampled_graph_frequency_plot(loader_objs['bge_train'], gt_graph, 1000, None)
        sampled_graph = np.asarray(imageio.imread(dag_file))
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, 0, dataformats='HWC')

    writer.flush()
    writer.close()


def train_batch(model, loader_objs, optimizers, opt, writer, step, start_time):
    loss_dict, media_dict = {}, {}
    prediction, gt = None, None
    
    # Zero out gradients
    for optimizer in optimizers: optimizer.zero_grad()
    model.train()

    if opt.model in ['SlotAttention_img', 'Slot_VCN_img']: 
        train_loss, loss_dict, prediction, gt, media_dict = train_slot(opt, loader_objs, model, writer, step, start_time)

    elif opt.model in ['VCN']: 
        train_loss, loss_dict = train_vcn(opt, loader_objs, model, writer, step, start_time)
    
    elif opt.model in ['VCN_img']:
        prediction, gt, train_loss, loss_dict = train_image_vcn(opt, loader_objs, model, writer, step)

    elif opt.model in ['GraphVAE']:
        prediction, gt, train_loss, loss_dict = train_graph_vae(opt, loader_objs, model, writer, step, start_time)

    elif opt.model in ['VAEVCN']:
        train_loss, loss_dict = train_vae_vcn(opt, loader_objs, model, writer, step, start_time)

    if opt.clip != -1:  torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.clip))
    train_loss.backward()

    if opt.opti == 'alt':
        assert len(optimizers) == 2
        if (step % 2 == 0): 
            optimizers[0].step()
        else: 
            optimizers[1].step()
    else:
        # In this case we always have only one optimizer
        assert len(optimizers) == 1
        optimizers[0].step()

    grad_norm = utils.get_grad_norm(model.parameters(), opt.opti == 'alt')
    loss_dict['Gradient Norm'] = grad_norm
    return prediction, gt, train_loss, loss_dict, media_dict

# Vanilla Slot attention and Slot-Image_VCN
def train_slot(opt, loader_objs, model, writer, step, start_time):
    # For models: Slot attention or Slot Image VCN 
    media_dict = {}
    bge_train = loader_objs['bge_train']
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    keys = ['Slot reconstructions', 'Weighted reconstructions', 'Slotwise masks']
    model.get_prediction(batch_dict, bge_train, step)
    recon_combined, slot_recons, slot_masks, weighted_recon, _ = model.get_prediction(batch_dict, bge_train, step)
    train_loss, loss_dict = model.get_loss()
    
    gt_np = model.ground_truth[0].detach().cpu().numpy()
    gt_np = np.moveaxis(gt_np, -3, -1)
    gt_np = ((gt_np + 1)/2) * 255.0
    media_dict['Slot reconstructions'] = [wandb.Image(m) for m in slot_recons]
    media_dict['Weighted reconstructions'] = [wandb.Image(m) for m in weighted_recon]
    media_dict['Slotwise masks'] = [wandb.Image(m) for m in slot_masks]
    values = [slot_recons / 255.0, weighted_recon / 255.0, slot_masks]
    utils.log_images_to_tb(opt, step, keys, values, writer, 'NHWC')
    prediction, gt = recon_combined, ((model.ground_truth + 1)/2) * 255.0
    
    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict, prediction, gt, media_dict

# Train and evaluate Vanilla VCN
def train_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    model.get_prediction(bge_model, step, loader_objs['train_dataloader'].graph)
    train_loss, loss_dict, _ = model.get_loss()
    if step == 0:   
        logdir = utils.set_tb_logdir(opt)
        utils.log_enumerated_dags_to_tb(writer, logdir, opt)
    if step % 100 == 0: 
        tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {train_loss.item():.5f} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict

def evaluate_vcn(opt, writer, model, bge_test, step, vae_elbo, device, loss_dict, time_epoch, train_data):
    gt_graph = train_data.graph
    model.eval()
    with torch.no_grad():
        model.get_prediction(bge_test, step, gt_graph) 
        _, loss_dict, _ = model.get_loss()
        elbo = loss_dict['graph_losses/Total loss']
        
    vae_elbo.append(elbo)

    if step % 100 == 0:
        kl_full, hellinger_full = 0., 0.
        if opt.num_nodes<=4:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)

        print('Step {}:  TRAIN - ELBO: {:.5f} likelihood: {:.5f} kl graph: {:.5f} VAL-ELBO: {:.5f} Temp Target {:.4f} Time {:.2f}'.\
                    format(step, loss_dict['graph_losses/Per sample loss'], loss_dict['graph_losses/Neg. log likelihood'], \
                    loss_dict['graph_losses/KL loss'], elbo, model.gibbs_temp, np.sum(time_epoch[step-100:step]), \
                    flush = True))

        # shd -> Expected structural hamming distance
        shd, prc, rec = vcn_utils.exp_shd(model, train_data.adjacency_matrix)
        kl_full, hellinger_full, auroc_score  = 0., 0., 0.

        if opt.num_nodes <= 4:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)
            writer.add_scalar('Evaluations/Hellinger Full', hellinger_full, step)
        else:
            auroc_score = vcn_utils.auroc(model, train_data.adjacency_matrix)
            writer.add_scalar('Evaluations/AUROC', auroc_score, step)
        
        writer.add_scalar('Evaluations/Exp. SHD', shd, step)
        writer.add_scalar('Evaluations/Exp. Precision', prc, step)
        writer.add_scalar('Evaluations/Exp. Recall', rec, step)

        print('Exp SHD:', shd,  'Exp Precision:', prc, 'Exp Recall:', rec, \
            'Kl_full:', kl_full, 'hellinger_full:', hellinger_full,\
        'auroc:', auroc_score)
        print()

    return vae_elbo


# Image-VCN
def train_image_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    enc_inp, prediction = model.get_prediction(batch_dict, bge_model, step)
    utils.log_encodings_per_node_to_tb(opt, writer, enc_inp, step)  # enc_inp has shape [num_nodes, chan_per_nodes, h, w]
    train_loss, loss_dict, _ = model.get_loss()
    gt = ((model.ground_truth + 1)/2) * 255.0
    
    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return prediction, gt, train_loss, loss_dict
        
# Graph VAE
def train_graph_vae(opt, loader_objs, model, writer, step, start_time):
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    prediction = model.get_prediction(batch_dict, step)
    train_loss, loss_dict = model.get_loss(step)
    gt = ((model.ground_truth + 1)/2) * 255.0

    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return prediction, gt, train_loss, loss_dict

# VAE VCN
def train_vae_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    model.get_prediction(loader_objs, step)
    train_loss, total_loss, loss_dict = model.get_loss(step)
    if step == 0:   
        logdir = utils.set_tb_logdir(opt)
        # utils.log_enumerated_dags_to_tb(writer, logdir, opt)
    if step % 100 == 0: 
        tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {total_loss:.5f} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict
