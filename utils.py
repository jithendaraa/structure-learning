import torch
import torch.nn as nn
import os

def set_opts(opt):
    print()
    # Set logdir and create dirs along the way
    if os.path.isdir(opt.logdir) is False:  os.mkdir(opt.logdir)    # mkdir logs
    opt.logdir = os.path.join(opt.logdir, opt.model)                # logs/ConvGRU
    if os.path.isdir(opt.logdir) is False:  os.mkdir(opt.logdir)    # mkdir logs/ConvGRU
    print("logdir:", opt.logdir)

    opt.data_dir = os.path.join(opt.storage_dir, opt.data_dir)
    print("data_dir:", opt.data_dir)

    # Set model_params_logdir and create dirs along the way
    opt.model_params_logdir = os.path.join(opt.logdir, opt.model_params_logdir)     # logs/ConvGRU/model_params
    if os.path.isdir(opt.model_params_logdir) is False:  os.mkdir(opt.model_params_logdir)
    print("model_params_logdir:", opt.model_params_logdir)
    print()
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