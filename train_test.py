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


def train_model(model, loader_objs, exp_config_dict, opt, device):

    # Data loaders
    train_dataloader = loader_objs['train_dataloader']
    n_train_batches = loader_objs['n_train_batches']
    print("Train batches", n_train_batches)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    writer = SummaryWriter(os.path.join(opt.logdir, opt.ckpt_id + '_' + str(opt.batch_size) + '_' + str(opt.lr) + '_' + str(opt.steps)))

    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict, settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        wandb.watch(model)

    start_time = time.time()
    for step in range(opt.steps):
        pred, gt, loss, loss_dict, media_dict = train_batch(model, train_dataloader, optimizer, opt, writer, step)
        
        pred_gt = torch.cat((pred[:opt.log_batches].detach().cpu(), gt[:opt.log_batches].cpu()), 0).numpy()
        pred_gt = np.moveaxis(pred_gt, -3, -1)

        # logging to tensorboard
        for key in loss_dict.keys():
            writer.add_scalar(key, loss_dict[key], step)
        
        # logging to wandb
        if opt.off_wandb is False:  # Log losses and pred, gt videos
            if step % opt.loss_log_freq == 0:   wandb.log(loss_dict, step=step)

            if step > 0 and step % opt.media_log_freq == 0:
                media_dict['Pred_GT'] = [wandb.Image(m) for m in pred_gt]
                wandb.log(media_dict, step=step)
                print("Logged media")
                
        print(f"[Step {step}/{opt.steps}]  | Step loss {round(loss, 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
        utils.save_model_params(model, optimizer, opt, step, opt.ckpt_save_freq) # Save model params


    writer.flush()
    writer.close()


def train_batch(model, train_dataloader, optimizer, opt, writer, step):
    # Get batch data & Get input sequence and output ground truth 
    data_dict = utils.get_data_dict(train_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt)
    loss_dict, media_dict = {}, {}
    optimizer.zero_grad()

    if opt.model in ['SlotAttention_img']:
        recon_combined, slot_recons, slot_masks = model.get_prediction(batch_dict)
        
        gt_np = model.ground_truth[0].detach().cpu().numpy()
        gt_np = np.moveaxis(gt_np, -3, -1)
        gt_np = gt_np * 255.0

        media_dict['Slot reconstructions'] = [wandb.Image(m) for m in slot_recons]
        media_dict['Slotwise masks'] = [wandb.Image(m) for m in slot_masks]

        if step > 0 and step % opt.media_log_freq == 0:
            writer.add_images('Slot reconstructions', slot_recons, step, dataformats='NHWC')
            writer.add_images('Slotwise masks', slot_masks, step, dataformats='NHWC')

        train_loss, loss_dict = model.get_loss()
        prediction, gt = recon_combined, model.ground_truth * 255.0
    
    if opt.clip != -1:  torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.clip))
    train_loss.backward()
    optimizer.step()
    grad_norm = utils.get_grad_norm(model.parameters())
    loss_dict['Gradient Norm'] = grad_norm

    return prediction, gt, train_loss.item(), loss_dict, media_dict