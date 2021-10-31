import torch
import os
import pickle
from os.path import *
from pathlib import Path
import wandb

def set_opts(opt):
    print()
    # Set logdir and create dirs along the way
    path_logdir = Path(opt.logdir)
    opt.logdir = path_logdir / opt.dataset / opt.phase / opt.model
    os.makedirs(opt.logdir, exist_ok=True)
    print("logdir:", opt.logdir)

    if opt.dataset == 'clevr':
      opt.data_dir = 'datasets/CLEVR_v1.0/images'

    opt.data_dir = os.path.join(opt.storage_dir, opt.data_dir)
    print("data_dir:", opt.data_dir)

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

def get_data_dict(dataloader):
    data_dict = dataloader.__next__()
    return data_dict

def get_dict_template(opt):
  if opt.model in ['SlotAttention_img', 'VCN', 'VCN_img']:
    return {"observed_data": None}

def set_batch_dict(opt, data_dict, batch_dict):
  if opt.model in ['SlotAttention_img', 'VCN', 'VCN_img']:
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
def save_model_params(model, optimizer, opt, step, ckpt_save_freq):
  if step > 0 and (step % ckpt_save_freq == 0):
      padded_zeros = '0' * (10 - len(str(step)))
      padded_step = padded_zeros + str(step)
      model_params_file_name = os.path.join(opt.model_params_logdir, opt.ckpt_id + '_' + padded_step + '.pickle')

      model_dict = {
          'step': step,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict()}

      with open(model_params_file_name, 'wb') as handle:
          pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
          print(f"Saved model parameters at step {step} -> {model_params_file_name}")





def get_grad_norm(model_params):
  total_norm = 0
  for p in model_params:
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
      writer.add_images("Reconstruction", pred_gt / 255.0, step, dataformats='NHWC')

def log_to_wandb(opt, step, loss_dict, media_dict, pred_gt):
  if opt.off_wandb is False:  # Log losses and pred, gt videos
    if step % opt.loss_log_freq == 0:   wandb.log(loss_dict, step=step)

    if step > 0 and step % opt.media_log_freq == 0 and pred_gt:
        media_dict['Pred_GT'] = [wandb.Image(m) for m in pred_gt]
        wandb.log(media_dict, step=step)
        print("Logged media")

def log_images_to_tb(opt, step, keys, values, writer, dataformats='NHWC'):
  assert len(keys) == len(values)
  
  if step > 0 and step % opt.media_log_freq == 0:
    for i in range(len(keys)):
      key, val = keys[i], val[i]
      writer.add_images(key, val, step, dataformats=dataformats)

