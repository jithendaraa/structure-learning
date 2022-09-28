import gym, pdb, sys
from argparse import Namespace

sys.path.append('../models')
sys.path.append('../modules')
sys.path.append('../exps')

from exps.chem_datagen import generate_colors
from exps.conv_decoder_bcd_exps.conv_decoder_bcd_utils import generate_chem_image_dataset

import numpy as np
import torch, os
from base import BaseDataLoader
from modules.ColorGen import LinearGaussianColor
from modules.dag_utils import SyntheticDataset
from datagen import get_data

class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config_dict, train=True):
        self.config_dict = config_dict
        self._setup()

    def _setup(self):
        obs_data = self.config_dict['obs_data']
        n =  (self.config_dict['pts_per_interv'] * self.config_dict['n_interv_sets'])
        d = self.config_dict['num_nodes']
        degree = self.config_dict['exp_edges']
        noise_sigma = self.config_dict['noise_sigma']
        n_interv_sets = self.config_dict['n_interv_sets']

        sd = SyntheticDataset(
                                n=n,
                                d=d,
                                graph_type="erdos-renyi",
                                degree=2 * degree,
                                sem_type=self.config_dict['sem_type'],
                                dataset_type="linear",
                                noise_scale=noise_sigma,
                                data_seed=self.config_dict['data_seed'],
                            )

        self.gt_W = sd.W
        self.gt_binary_W = np.where(np.abs(sd.W) >= 0.3, 1, 0)
        opt = Namespace(
            obs_data=obs_data,
            noise_sigma=noise_sigma,
            num_samples=n,
            num_nodes=d,
            identity_proj=False,
            proj=self.config_dict['proj'],
            proj_dims=self.config_dict['input_dim'],
            interv_value=self.config_dict['interv_value'],
            interv_type=self.config_dict['interv_type'],
            sem_type=self.config_dict['sem_type'],
            data_seed=self.config_dict['data_seed']
            )
        
        ( z, no_interv_targets, 
            x, proj_matrix, 
            interv_values ) = get_data(opt, n_interv_sets, sd, model='bcd', interv_value=opt.interv_value)

        self.z, self.x = np.array(z), torch.from_numpy(np.array(x))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        return x

class VectorDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, config_dict, training=True):
        self.data_dir = data_dir
        self.dataset = VectorDataset(self.data_dir, config_dict, train=training)
        self.gt_binary_W = self.dataset.gt_binary_W
        self.gt_W = self.dataset.gt_W
        self.z = self.dataset.z
        self.x = self.dataset.x
        super(VectorDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ChemDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, config_dict, train, ):
        self.config_dict = config_dict
        self._setup()
        
    def _setup(self, edge_threshold=0.3):
        num_samples = self.config_dict['obs_data'] + (self.config_dict['pts_per_interv'] * self.config_dict['n_interv_sets'])
        low, high = self.config_dict["low"], self.config_dict['high']
        interv_low, interv_high = self.config_dict["interv_low"], self.config_dict['interv_high']
        
        self.chem_data = LinearGaussianColor(
                        n=num_samples,
                        obs_data=self.config_dict['obs_data'],
                        d=self.config_dict['num_nodes'],
                        graph_type="erdos-renyi",
                        degree=2 * self.config_dict['exp_edges'],
                        sem_type=self.config_dict['sem_type'],
                        dataset_type="linear",
                        noise_scale=self.config_dict['noise_sigma'],
                        data_seed=self.config_dict['data_seed'],
                        low=low, 
                        high=high,
                        identity_P=True
                    )
        self.gt_W = self.chem_data.W
        self.gt_P = self.chem_data.P
        self.gt_L = self.chem_data.P.T @ self.chem_data.W.T @ self.chem_data.P

        self.opt = Namespace(num_samples=num_samples,
                        obs_data=self.config_dict['obs_data'],
                        num_nodes=self.config_dict['num_nodes'],
                        n_interv_sets=self.config_dict['n_interv_sets'],
                        noise_sigma=self.config_dict['noise_sigma'],
                        sem_type=self.config_dict['sem_type'])
        ( self.z, 
          self.interv_targets, 
          self.interv_values ) = generate_colors(self.opt, self.chem_data, low, high, interv_low, interv_high)

        self.images = generate_chem_image_dataset(self.opt.num_samples, 
                                                    self.opt.num_nodes, 
                                                    self.interv_values, 
                                                    self.interv_targets, 
                                                    self.z)
        self.flat_images = self.get_flat_images()
        self.gt_binary_W = np.where(self.gt_W >= edge_threshold, 1.0, 0.0)

    def get_flat_images(self):
        x = self.images[:, :, :, 0:1]
        samples = x.shape[0]
        x = x.reshape(samples, -1)
        min_v = 0.0
        range_v = 255. - min_v
        if range_v > 0:
            normalised = (x - min_v) / range_v
        else:
            raise NotImplementedError
            normalised = torch.zeros(x.size()).to(x.device)
        if self.config_dict['dataset'] == 'MNIST':   
            return (normalised > 0.5).type(torch.float)
        
        return torch.from_numpy(normalised)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.config_dict['dataset'] == 'chemdata':   
            x = x[:, :, 0:1]
        x = x.flatten()
        min_v = 0.0
        range_v = 255. - min_v
        if range_v > 0:
            normalised = (x - min_v) / range_v
        else:
            raise NotImplementedError
            normalised = torch.zeros(x.size()).to(x.device)
        
        if self.config_dict['dataset'] == 'MNIST':   
            return (normalised > 0.5).type(torch.float)
        
        return torch.from_numpy(normalised)

class ChemDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, config_dict, training=True):
        self.data_dir = data_dir
        self.dataset = ChemDataset(self.data_dir, config_dict, train=training)
        self.gt_binary_W = self.dataset.gt_binary_W
        self.gt_W = self.dataset.gt_W
        self.z = self.dataset.z
        self.x = self.dataset.flat_images
        super(ChemDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        print(f"batch size: {batch_size}")

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = BinarizedMNISTDataset(self.data_dir, train=training)
        self.gt_binary_W = None
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BinarizedMNISTDataset(torch.utils.data.Dataset):
    """
    MNIST dataset converted to binary values.
    """
    def __init__(self, data_dir, train=True):
        if train:
            self.images = torch.from_numpy(np.load(os.path.join(data_dir, 'train_images.npy')))
            self.labels = torch.from_numpy(np.load(os.path.join(data_dir, 'train_labels.npy')))
        else:
            self.images = torch.from_numpy(np.load(os.path.join(data_dir, 'test_images.npy')))
            self.labels = torch.from_numpy(np.load(os.path.join(data_dir, 'test_labels.npy')))

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx].flatten()
        min_v = x.min()
        range_v = x.max() - min_v
        if range_v > 0:
            normalised = (x - min_v) / range_v
        else:
            normalised = torch.zeros(x.size()).to(x.device)
        return (normalised > 0.5).type(torch.float)