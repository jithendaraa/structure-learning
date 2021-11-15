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


def train_model(model, loader_objs, exp_config_dict, opt, device):
    pred_gt, time_epoch, likelihood, kl_graph, elbo_train, vae_elbo = None, [], [], [], [], []

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
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
        pred, gt, loss, loss_dict, media_dict = train_batch(model, loader_objs, optimizer, opt, writer, step, start_time)
        time_epoch.append(time.time() - start_step_time)

        if opt.model in ['SlotAttention_img', 'VCN_img', 'Slot_VCN_img', 'GraphVAE']:
            pred_gt = torch.cat((pred[:opt.log_batches].detach().cpu(), gt[:opt.log_batches].cpu()), 0).numpy()
            pred_gt = np.moveaxis(pred_gt, -3, -1)
        
        elif opt.model in ['VCN']:
            vae_elbo = evaluate_vcn(opt, writer, model, loader_objs['bge_train'], step+1, vae_elbo, device, loss_dict, time_epoch, loader_objs['train_dataloader'])

        # logging to tensorboard and wandb
        utils.log_to_tb(opt, writer, loss_dict, step, pred_gt)
        utils.log_to_wandb(opt, step, loss_dict, media_dict, pred_gt)
        utils.save_model_params(model, optimizer, opt, step, opt.ckpt_save_freq) # Save model params

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


def train_batch(model, loader_objs, optimizer, opt, writer, step, start_time):
    loss_dict, media_dict = {}, {}
    prediction, gt = None, None
    optimizer.zero_grad()
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
    optimizer.step()
    grad_norm = utils.get_grad_norm(model.parameters())
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

# Vanilla VCN
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

def train_vae_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    model.get_prediction(loader_objs, step)
    # train_loss, loss_dict, _ = model.get_loss()
    # if step == 0:   
    #     logdir = utils.set_tb_logdir(opt)
    #     utils.log_enumerated_dags_to_tb(writer, logdir, opt)
    # if step % 100 == 0: 
    #     tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {train_loss.item():.5f} | Time: {round((time.time() - start_time) / 60, 3)}m")
    # return train_loss, loss_dict

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

        