import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
import os
import time

def train_model(model, loader_objs, exp_config_dict, opt, device):

    step = 0
    # Data loaders
    train_dataloader = loader_objs['train_dataloader']
    n_train_batches = loader_objs['n_train_batches']
    print("Train batches", n_train_batches)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict, settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        wandb.watch(model)

    for epoch in range(opt.epochs):
        epoch_loss = 0
        start_time = time.time()

        for it in range(n_train_batches):
            pred, gt, loss, loss_dict, media_dict = train_batch(model, train_dataloader, optimizer, opt, device)
            pred_gt = torch.cat((pred.detach().cpu(), gt.cpu()), 0).numpy()
            epoch_loss += loss
            step += 1

            if opt.off_wandb is False:  # Log losses and pred, gt videos
                if step % opt.loss_log_freq == 0:   wandb.log(loss_dict, step=step)

                if step == 1 or step % opt.media_log_freq == 0:
                    media_dict['Pred_GT'] = wandb.Image(pred_gt) 
                    wandb.log(media_dict, step=step)
                    print("Logged media")
                
            print(f"[Epoch {epoch}] step {step} | Step loss {round(loss, 5)}")
            utils.save_model_params(model, optimizer, epoch, opt, step, opt.ckpt_save_freq) # Save model params

        end_time = time.time()
        print()
        print(f'Epoch Num: {epoch} | Epoch Loss: {round(epoch_loss, 5)} | Time for epoch: {(end_time - start_time) / 60}m')


def train_batch(model, train_dataloader, optimizer, opt, device):
    # Get batch data & Get input sequence and output ground truth 
    data_dict = utils.get_data_dict(train_dataloader)
    batch_dict = utils.get_next_batch(data_dict, opt)
    loss_dict, media_dict = {}, {}
    optimizer.zero_grad()

    if opt.model in ['SlotAttention_img']:
        recon_combined, slot_recons, slot_masks = model.get_prediction(batch_dict)
        if opt.off_wandb is False:
            media_dict['Slot reconstructions'] = wandb.Image(slot_recons[0].detach().cpu().numpy())
            media_dict['Slotwise masks'] = wandb.Image(slot_masks[0].detach().cpu().numpy())

        train_loss, loss_dict = model.get_loss(recon_combined)
        prediction, gt = recon_combined, model.ground_truth
    
    train_loss.backward()
    if opt.clip != -1:  torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.clip))
    optimizer.step()
    loss_dict['Gradient Norm'] = utils.get_grad_norm(model.parameters())
    return prediction * 255.0, gt * 255.0, train_loss.item(), loss_dict, media_dict