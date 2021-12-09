import warnings
import sys
sys.path.append('models')

from models.Decoder_DIBS import Decoder_DIBS
warnings.filterwarnings("ignore")
import argparse
import ruamel.yaml as yaml
import sys
import pathlib
import torch
from train_test import train_model
import utils
from dataloader import *


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--data_path', default="/home/jithen/scratch")

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
    opt.data_path = args.data_path
    opt = utils.set_opts(opt)

    exp_config = {}

    for arg in vars(opt):
      val = getattr(opt, arg)
      exp_config[arg] = val
      
    return opt, exp_config

def build_model(opt, device, loader_objs):
  key = None
  resolution = (opt.resolution, opt.resolution) 
  implemented_models = ['SlotAttention_img', 'VCN', 'VCN_img', 
                      'Slot_VCN_img', 'GraphVAE', 'VAEVCN', 
                      'DIBS', 'VAE_DIBS', 'Decoder_DIBS']
  
  if opt.model in ['SlotAttention_img']:
    from models.SlotAttentionAutoEncoder import SlotAttentionAutoEncoder as Slot_Attention
    model = Slot_Attention(opt, resolution, opt.num_slots, opt.num_iterations, device)
  
  elif opt.model in ['VCN', 'VCN_img']:
    if opt.datatype in ['er']: 
      from models.VCN import VCN
      model = VCN(opt, opt.num_nodes, opt.sparsity_factor, opt.gibbs_temp, device)
    
    elif opt.datatype in ['image']: 
      from models.VCN_image import VCN_img as VCN
      model = VCN(opt, resolution, opt.num_slots, opt.num_iterations, opt.sparsity_factor, opt.gibbs_temp, device)
    
    elif opt.datatype in ['video']: pass

  elif opt.model in ['Slot_VCN_img']:
    from models.Slot_VCN_img import Slot_VCN_img
    model = Slot_VCN_img(opt, resolution, opt.num_slots, opt.num_iterations, opt.sparsity_factor, opt.gibbs_temp, device)

  elif opt.model in ['GraphVAE']:
    from models.GraphVAE import GraphVAE
    model = GraphVAE(opt, opt.N, opt.M, device)
  
  elif opt.model in ['VAEVCN']:
    from models.VAEVCN import VAEVCN
    model = VAEVCN(opt, opt.num_nodes, opt.sparsity_factor, opt.gibbs_temp, device)

  elif opt.model in ['DIBS']:
    from jax import random
    from models.dibs.eval.target import make_linear_gaussian_equivalent_model

    key = random.PRNGKey(123)
    target = make_linear_gaussian_equivalent_model(key =  key, n_vars = opt.num_nodes, graph_prior_str = opt.datatype,
		                                              obs_noise = opt.noise_sigma, mean_edge = opt.theta_mu, sig_edge = opt.theta_sigma, 
                                                  n_observations = opt.num_samples, n_ho_observations = opt.num_samples)

    model = target
  
  elif opt.model in ['VAE_DIBS']:
    from models.VAE_DiBS import VAE_DIBS
    from jax import random
    key = random.PRNGKey(123)
    def model():
      return VAE_DIBS(opt.noise_sigma, opt.theta_mu, opt.num_samples,
                        opt.theta_sigma, opt.proj, opt.num_nodes, opt.known_ED, opt.data_type,
                        opt.n_particles, opt.num_updates, opt.h_latent, opt.alpha_linear,
                        opt.alpha_mu, opt.alpha_lambd, opt.proj_dims, 
                        loader_objs['true_encoder'], loader_objs['true_decoder'])

  elif opt.model in ['Decoder_DIBS']:
    from models.Decoder_DIBS import Decoder_DIBS
    from jax import random
    import jax.numpy as jnp
    key = random.PRNGKey(123)
    latent_prior_std = 1.0 / jnp.sqrt(opt.num_nodes)
    
    def model():
      return Decoder_DIBS(key, opt.num_nodes, opt.data_type, opt.h_latent,
                          opt.theta_mu, opt.alpha_mu, opt.alpha_lambd,
                          opt.alpha_linear, opt.n_particles, opt.proj_dims, opt.num_samples,
                          latent_prior_std=latent_prior_std)

  else: 
    raise NotImplementedError(f'Model {opt.model} is not implemented. Try one of {implemented_models}')

  return model, key
    

def main(opt, exp_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_objs = parse_datasets(opt, device) # Dataloader
    model, key = build_model(opt, device, loader_objs)

    train_model(model, loader_objs, exp_config, opt, device, key)

if __name__ == '__main__':
    opt, exp_config = get_opt()
    main(opt, exp_config)