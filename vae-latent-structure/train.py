import os, pdb, sys
import json
import argparse
import torch

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger

sys.path.append('..')
sys.path.append("../exps")
sys.path.append('../CausalMBRL')
sys.path.append('../CausalMBRL/envs')
import envs
import data_loader.data_loaders as module_data
from modules.GumbelSinkhorn import GumbelSinkhorn

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def get_config_dict(config):
    config_dict = {
        "batch_size": config['data_loader']['args']['batch_size'],
        "dataset": config['data_loader']['type'],
        "num_steps": config['trainer']['epochs'],
        "model": "GraphVAE",
        "input_dim": config['arch']['args']['input_dim']
    }

    for key in config.keys():
        if not isinstance(config[key], dict):
            config_dict[key] = config[key]

    return config_dict

def main(config, resume, dataseed, exp_edges, num_nodes):
    train_logger = Logger()
    config['data_seed'] = dataseed
    config['exp_edges'] = exp_edges
    config['arch']['args']['n_nodes'] = num_nodes
    config['num_nodes'] = num_nodes
    exp_config_dict = get_config_dict(config)

    # setup data_loader instances
    if config['dataset'] == 'MNIST':
        data_loader = get_instance(module_data, 'data_loader', config)
        valid_data_loader = data_loader.split_validation()
    
    elif config['dataset'] == 'chemdata':
        data_loader = module_data.ChemDataLoader(config['data_loader']['args']['data_dir'],
                                                 config['data_loader']['args']['batch_size'],
                                                 config['data_loader']['args']['shuffle'],
                                                 config['data_loader']['args']['validation_split'],
                                                 config['data_loader']['args']['num_workers'],
                                                 exp_config_dict,
                                                 training=True
                                                )
    
    elif config['dataset'] == 'vector':
        data_loader = module_data.VectorDataLoader(config['data_loader']['args']['data_dir'],
                                                 config['data_loader']['args']['batch_size'],
                                                 config['data_loader']['args']['shuffle'],
                                                 config['data_loader']['args']['validation_split'],
                                                 config['data_loader']['args']['num_workers'],
                                                 exp_config_dict,
                                                 training=True
                                                )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model.gating_params)
    freeze = [(0,1,0), (0,2,1), (0,3,0), (0,4,1), (1,2,0), (1,3,1), (1,4,0), (2,3,1), (2,4,1), (3,4,1)]

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
    
    for (x, y, v) in freeze:
        print(model.gating_params[x][y - x -1])
        trainer.model.gating_params[x][y - x -1].data = torch.Tensor([[v]]).detach()
        print("Setting {}-{} to {}".format(x, y, v))

    trainer.train(exp_config_dict, data_loader.gt_binary_W, 
                data_loader.gt_W, 
                z_true=data_loader.z, 
                x_true=data_loader.x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--dataseed', type=int, default=13)
    parser.add_argument('--exp_edges', type=float, default=1.0)
    parser.add_argument('--num_nodes', type=int, default=10)
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        # config = torch.load(args.resume)['config']
        with open(args.config) as handle:
            config = json.load(handle)

    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.dataseed, args.exp_edges, args.num_nodes)