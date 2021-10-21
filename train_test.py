import torch

def train_model(model, opt):
    
    b, c, h, w = opt.batch_size, opt.channels, opt.resolution, opt.resolution
    dummy_data = torch.ones((b, c, h, w)).cuda()

    print("dummy_data", dummy_data.size())
    model(dummy_data)
