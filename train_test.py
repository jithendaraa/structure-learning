import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import vcn_utils


def train_model(model, loader_objs, exp_config_dict, opt, device):
    pred_gt, time_epoch, likelihood, kl_graph, elbo_train, vae_elbo = None, [], [], [], [], []

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    logdir = os.path.join(opt.logdir, opt.ckpt_id + '_' + str(opt.batch_size) + '_' + str(opt.lr) + '_' + str(opt.steps) + '_' + str(opt.resolution))
    writer = SummaryWriter(logdir)

    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict, settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        wandb.watch(model)

    start_time = time.time()
    
    for step in range(opt.steps):
    
        start_step_time = time.time()
        pred, gt, loss, loss_dict, media_dict = train_batch(model, loader_objs, optimizer, opt, writer, step)
        time_epoch.append(time.time() - start_step_time)

        if opt.model not in ['VCN']:
            pred_gt = torch.cat((pred[:opt.log_batches].detach().cpu(), gt[:opt.log_batches].cpu()), 0).numpy()
            pred_gt = np.moveaxis(pred_gt, -3, -1)
        else:
            vae_elbo = evaluate_batch(opt, model, loader_objs['bge_train'], step, vae_elbo, device, loss_dict, time_epoch, loader_objs['train_dataloader'])

        # logging to tensorboard and wandb
        utils.log_to_tb(opt, writer, loss_dict, step, pred_gt)
        utils.log_to_wandb(opt, step, loss_dict, media_dict, pred_gt)
                
        print(f"[Step {step}/{opt.steps}] | Step loss {round(loss, 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
        utils.save_model_params(model, optimizer, opt, step, opt.ckpt_save_freq) # Save model params

    writer.flush()
    writer.close()


def train_batch(model, loader_objs, optimizer, opt, writer, step):
    
    train_dataloader = loader_objs['train_dataloader']
    loss_dict, media_dict = {}, {}
    optimizer.zero_grad()
    model.train()

    if opt.model in ['SlotAttention_img']:
        keys = ['Slot reconstructions', 'Weighted reconstructions', 'Slotwise masks']

        data_dict = utils.get_data_dict(train_dataloader)
        batch_dict = utils.get_next_batch(data_dict, opt)
        
        recon_combined, slot_recons, slot_masks, weighted_recon = model.get_prediction(batch_dict)
        
        gt_np = model.ground_truth[0].detach().cpu().numpy()
        gt_np = np.moveaxis(gt_np, -3, -1)
        gt_np = ((gt_np + 1)/2) * 255.0

        media_dict['Slot reconstructions'] = [wandb.Image(m) for m in slot_recons]
        media_dict['Weighted reconstructions'] = [wandb.Image(m) for m in weighted_recon]
        media_dict['Slotwise masks'] = [wandb.Image(m) for m in slot_masks]

        values = [slot_recons / 255.0, weighted_recon / 255.0, slot_masks]
        utils.log_images_to_tb(opt, step, keys, values, writer, 'NHWC')

        train_loss, loss_dict = model.get_loss()
        prediction, gt = recon_combined, ((model.ground_truth + 1)/2) * 255.0
    
    elif opt.model in ['VCN']:
        model.get_prediction(loader_objs['bge_train'], step)
        train_loss, loss_dict, _ = model.get_loss()
        prediction, gt = None, None

    if opt.clip != -1:  torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.clip))
    train_loss.backward()
    optimizer.step()
    grad_norm = utils.get_grad_norm(model.parameters())
    loss_dict['Gradient Norm'] = grad_norm

    return prediction, gt, train_loss.item(), loss_dict, media_dict


def evaluate_batch(opt, model, bge_test, step, vae_elbo, device, loss_dict, time_epoch, train_data):
    
    model.eval()
    with torch.no_grad():
        model.get_prediction(bge_test, step) 
        _, loss_dict, _ = model.get_loss()
        elbo = loss_dict['Total loss']
        
    vae_elbo.append(elbo)

    if step % 50 == 0:
        kl_full, hellinger_full = 0., 0.
        if opt.num_nodes<=4:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)

        print('Step {}:  TRAIN - ELBO: {:.5f} likelihood: {:.5f} kl graph: {:.5f} VAL-ELBO: {:.5f} Temp Target {:.4f} Time {:.2f}'.\
                    format(step, loss_dict['Per sample loss'], loss_dict['Reconstruction loss'], \
                    loss_dict['KL loss'], elbo, model.gibbs_temp, np.sum(time_epoch[step-100:step]), \
                    flush = True))

        # shd -> Expected structural hamming distance
        shd, prc, rec = vcn_utils.exp_shd(model, train_data.adjacency_matrix)
        print("Computed metrics")
        kl_full, hellinger_full, auroc_score  = 0., 0., 0.

        if opt.num_nodes <= 4:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)
            print("kl_full", kl_full)
            print("hellinger_full", hellinger_full)   
        else:
            auroc_score = vcn_utils.auroc(model, train_data.adjacency_matrix)

        print('Exp SHD:', shd,  'Exp Precision:', prc, 'Exp Recall:', rec, 'Kl_full:', kl_full, 'hellinger_full:', hellinger_full,\
        'auroc:', auroc_score)

    return vae_elbo

        