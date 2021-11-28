import os
from os.path import join
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets.mnist import MNIST
import utils
import cv2

from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import networkx as nx 

class CLEVR(Dataset):
    def __init__(self, root, opt, is_train, device=None, ch=3):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(CLEVR, self).__init__()
        self.root = root
        self.opt = opt
        self.is_train = is_train
        self.device = device
        self.channels = ch
        self.file_prefix = 'CLEVR_train_'
        self.files = os.listdir(self.root)
        random.shuffle(self.files)
        self.length = len(os.listdir(root))

    def get_resized_torch_image(self, file_path, resolution):
        resolution = (resolution, resolution)
        cv_image = cv2.imread(file_path)
        np_image = np.array(cv2.resize(cv_image, resolution, interpolation=cv2.INTER_AREA))
        np_image = np.moveaxis(np_image, -1, 0) # move channel from dim -1 to dim 0
        np_image = ((np_image / 255.0) - 0.5) * 2
        np_image = np.clip(np_image, -1.0, 1.0)
        torch_image = torch.from_numpy( np_image ).contiguous().float().to(self.device)
        return torch_image

    def get_item_dict(self, file_path):
        if self.opt.model in ['SlotAttention_img', 'VCN', 'VCN_img', 'Slot_VCN_img', 'GraphVAE', 'VAEVCN']:
            image = self.get_resized_torch_image(file_path, self.opt.resolution) # [-0.5, 0.5]
            item_dict = { 'observed_data': image, 'predicted_data': image}
        
        return item_dict
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        item_dict = self.get_item_dict(file_path)
        return item_dict

    def __len__(self):
        return self.length



def parse_datasets(opt, device):
    data_objects, data = {}, None
    train_dir = os.path.join(opt.data_dir, 'train')
    test_dir = os.path.join(opt.data_dir, 'test')

    if opt.dataset == 'clevr':
        trainFolder = CLEVR(train_dir, opt, is_train=True, device=device, ch=opt.channels)
        testFolder = CLEVR(test_dir, opt, is_train=False, device=device, ch=opt.channels)
        train_dataloader = DataLoader(trainFolder, batch_size=opt.batch_size, shuffle=False)
        test_dataloader = DataLoader(testFolder, batch_size=opt.batch_size, shuffle=False)
        data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        data = train_dataloader

    elif opt.dataset == 'mnist':
        from torchvision.datasets import MNIST
        # set download = True for the first time you run and ensure internet access before running for 1st time
        trainFolder = MNIST(train_dir, train=True, transform=T.Compose([T.ToTensor()]))
        testFolder = MNIST(test_dir, train=False, transform=T.Compose([T.ToTensor()]))
        train_dataloader = DataLoader(trainFolder, batch_size=opt.batch_size, shuffle=False)
        test_dataloader = DataLoader(testFolder, batch_size=opt.batch_size, shuffle=False)
        data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader)}
        data = train_dataloader

    elif opt.dataset == 'er':
        train_dataloader = ER(num_nodes = opt.num_nodes, exp_edges = opt.exp_edges, noise_type = opt.noise_type, noise_sigma = opt.noise_sigma, num_samples = opt.num_samples, mu_prior = opt.theta_mu, sigma_prior = opt.theta_sigma, seed = opt.data_seed, project=opt.proj, proj_dims=opt.proj_dims, noise_mu=opt.noise_mu)
        nx.draw(train_dataloader.graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
        logdir = utils.set_tb_logdir(opt)
        plt.savefig(join(logdir,'gt_graph.png'))

        data_objects = {"train_dataloader": train_dataloader}
        data = train_dataloader.samples
    
    else:
        raise NotImplementedError(f"There is no dataset named {opt.dataset}")

    if opt.model in ['VCN', 'VCN_img', 'Slot_VCN_img', 'VAEVCN', 'DIBS', 'VAE_DIBS', 'Decoder_DIBS']:
        if data is None:    data = train_dataloader.samples
        bge_train = BGe(opt,
                        mean_obs = [opt.theta_mu]*opt.num_nodes, 
                        alpha_mu = opt.alpha_mu, 
                        alpha_lambd=opt.alpha_lambd, 
                        data = data, 
                        device = device)
        
        data_objects["bge_train"] = bge_train
        data_objects['data'] = data
        data_objects['adj_matrix'] = train_dataloader.adjacency_matrix
        data_objects['true_encoder'], data_objects['true_decoder'] = None, None

        if opt.proj in ['linear', 'nonlinear']:
            data_objects['projected_data'] = train_dataloader.projected_samples
            data_objects['projection_matrix'] = train_dataloader.projection_matrix
            data_objects['true_encoder'] = train_dataloader.true_encoder
            data_objects['true_decoder'] = train_dataloader.true_decoder
    

    print(f"Loaded dataset {opt.dataset}")
    print()
    return data_objects