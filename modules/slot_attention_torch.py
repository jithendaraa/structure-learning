import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self):
        super(SlotAttentionAutoEncoder, self).__init__()