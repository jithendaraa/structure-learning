import argparse
import ruamel.yaml as yaml
import sys
import pathlib
import utils
import torch

from models.SlotAttentionAutoEncoder import SlotAttentionAutoEncoder as Slot_Attention
from train_test import train_model
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
      arg_type = utils.args_type(value)
      parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    
    opt = parser.parse_args(remaining)
    opt = utils.set_opts(opt)

    exp_config = {}

    for arg in vars(opt):
      val = getattr(opt, arg)
      exp_config[arg] = val
    #   print(arg, val)
    # print()
    return opt, exp_config

def build_model(opt, device):

  implemented_models = ['SlotAttention_img']
  
  resolution = (opt.resolution, opt.resolution) 
  
  if opt.model in ['SlotAttention_img']:
    model = Slot_Attention(opt, resolution, opt.num_slots, opt.num_iterations, device)
  
  else: 
    raise NotImplementedError(f'Model {opt.model} is not implemented. Try one of {implemented_models}')

  return model
    

def main(opt, exp_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(opt, device)

    # Dataloader
    loader_objs = parse_datasets(opt, device)
    
    train_model(model, loader_objs, exp_config, opt, device)

if __name__ == '__main__':
    opt, exp_config = get_opt()
    main(opt, exp_config)