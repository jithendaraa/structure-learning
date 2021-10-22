import os
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import utils
import cv2

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

    def get_item(self, idx):
        pass
        
        # in_frames = torch.from_numpy( (in_frames / 255.0) - 0.5 ).contiguous().float().to(self.device).permute(0, 3, 1, 2)
        # out_frames = torch.from_numpy( (out_frames / 255.0) - 0.5 ).contiguous().float().to(self.device).permute(0, 3, 1, 2)
        
        # out = {
        #     "idx": idx, 
        #     "observed_data": in_frames, 
        #     "data_to_predict": out_frames,
        #     'mask': mask,
        #     'in_flow_labels': in_flow_labels,
        #     'out_flow_labels': out_flow_labels}

        # return out

    def get_resized_torch_image(self, file_path, resolution):
        resolution = (resolution, resolution)
        cv_image = cv2.imread(file_path)
        np_image = np.array(cv2.resize(cv_image, resolution, interpolation=cv2.INTER_AREA))
        np_image = np.moveaxis(np_image, -1, 0) # move channel from dim -1 to dim 0
        torch_image = torch.from_numpy( (np_image / 255.0) - 0.5 ).contiguous().float().to(self.device)
        return torch_image

    def get_item_dict(self, file_path):
        if self.opt.model in ['SlotAttention_img']:
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
    if opt.dataset == 'clevr':
        train_dir = os.path.join(opt.data_dir, 'train')
        test_dir = os.path.join(opt.data_dir, 'test')
        
        trainFolder = CLEVR(train_dir, opt, is_train=True, device=device, ch=opt.channels)
        testFolder = CLEVR(test_dir, opt, is_train=False, device=device, ch=opt.channels)
    
        train_dataloader = DataLoader(trainFolder, batch_size=opt.batch_size, shuffle=False)
        test_dataloader = DataLoader(testFolder, batch_size=opt.batch_size, shuffle=False)

    else:
        raise NotImplementedError(f"There is no dataset named {opt.dataset}")

    data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}
    
    return data_objects