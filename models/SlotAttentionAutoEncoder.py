import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.SoftPosnEmbed import SoftPositionEmbed
from modules.SlotAttention import SlotAttention
from utils import unstack_and_split

class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""
    def __init__(self, opt, resolution, num_slots, num_iterations, device):
        super(SlotAttentionAutoEncoder, self).__init__()
        """Builds the Slot Attention-based auto-encoder.

            Args:
            resolution: Tuple of integers specifying width and height of input image.
            num_slots: Number of slots in Slot Attention.
            num_iterations: Number of iterations in Slot Attention.
        """
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.opt = opt
        self.device = device

        encoder_out_ch = opt.encoder_out_ch
        self.decoder_initial_size = (8, 8)

        self.cnn_encoder = CNN_Encoder(in_ch=opt.channels, out_ch=encoder_out_ch, device=device)
        self.pos_encoder = SoftPositionEmbed(self.resolution, encoder_out_ch, device=device)

        self.layer_norm = nn.LayerNorm(encoder_out_ch).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_out_ch, 64), nn.ReLU(),
            nn.Linear(64, encoder_out_ch)
        ).to(device)

        self.slot_attention = SlotAttention(encoder_out_ch, num_iterations, num_slots, opt.slot_size, device=device)

        self.pos_decoder = SoftPositionEmbed(self.decoder_initial_size, opt.slot_size, device=device)
        self.cnn_decoder = CNN_Decoder(out_ch=opt.channels, device=device)
        print("Initialised model: SlotAttention AutoEncoder")

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.pos_encoder(x) # b, c, h, w
        x = spatial_flatten(x)    # FLatten spatial dimension
        x = self.mlp(self.layer_norm(x.permute(0, 2, 1))).permute(0, 2, 1)
        
        # Slot Attention module.
        slots = self.slot_attention(x.permute(0, 2, 1)) # pass as b, spatial_dims, c
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size) # b*num_slots, c, h, w
        # `x` has shape: [b*num_slots, h, w, slot_size].
        x = self.pos_decoder(x)
        x = self.cnn_decoder(x)
        # `x` has shape: [b*num_slots, c+1, h, w].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=self.opt.batch_size)
        # `recons` has shape: [batch_size, num_slots, num_channels, h, w]
        # `masks` has shape: [batch_size, num_slots, 1, h, w].

        # Normalize alpha masks over slots.
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, c, h, w].

        return recon_combined, recons, masks, slots

    def get_prediction(self, batch_dict, dummy1=None, dummy2=None):
        np_clip_convert = lambda x: np.clip(((x + 1) / 2) * 255.0, 0. , 255.).astype(np.uint8)
        torch_clamp_convert = lambda x: torch.clamp(((x + 1) / 2) * 255.0, 0., 255.).to(torch.int8)

        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['data_to_predict'].to(self.device) # [-1, 1]
        recon_combined, recons, masks, slots = self(self.input_images)
        self.pred = recon_combined
        
        weighted_recon = recons * masks
        weighted_recon = weighted_recon[0].detach().cpu().numpy()
        slot_recons, slot_masks = recons[0].detach().cpu().numpy(), masks[0].detach().cpu().numpy()
        slot_recons, slot_masks, weighted_recon = np.moveaxis(slot_recons, -3, -1), np.moveaxis(slot_masks, -3, -1), np.moveaxis(weighted_recon, -3, -1)
        
        return torch_clamp_convert(recon_combined), np_clip_convert(slot_recons), slot_masks, np_clip_convert(weighted_recon), slots
    
    def get_loss(self):
        recon_loss = F.mse_loss(self.pred, self.ground_truth) 
        
        loss_dict = {
            'Reconstruction loss': recon_loss.item()
        }

        return recon_loss, loss_dict


def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  b, num_slots, slot_size = slots.size()
  slots = slots.reshape(-1, slot_size)[:, :, None, None]
  grid = slots.repeat(1, 1, resolution[0], resolution[1])   # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

def spatial_flatten(x):
    b, c, h, w = x.size()
    return x.view(b, c, -1)

class CNN_Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, kernel_size=5, padding='same', stride=1, device=None):
        super(CNN_Encoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size, stride, padding), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size, stride, padding), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size, stride, padding), nn.ReLU(),
            nn.Conv2d(64, out_ch, kernel_size, stride, padding), nn.ReLU()
        ).to(device)

    def forward(self, x):
        return self.conv(x)

class CNN_Decoder(nn.Module):
    def __init__(self, in_ch=64, out_ch=3, kernel_size=5, stride=(2, 2), device=None):
        super(CNN_Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, out_ch + 1, 4, 2, 1),
        ).to(device)

    def forward(self, x):
        return self.deconv(x)