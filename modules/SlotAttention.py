import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
  def __init__(self, features, num_iterations, num_slots, slot_size=64, mlp_hidden_size=128, epsilon=1e-8, device=None):
    super(SlotAttention, self).__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon
    self.device = device
    
    self.norm_inputs = nn.LayerNorm(features).to(device)
    self.norm_slots = nn.LayerNorm(self.slot_size).to(device)
    self.norm_mlp = nn.LayerNorm(self.slot_size).to(device)

    # Parameters for Gaussian init (shared by all slots).
    self.slots_mu = nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size), gain=1.0).to(device)
    self.slots_log_sigma  = nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size), gain=1.0).to(device)

    # Linear maps for the attention module.
    self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False).to(device)
    self.project_k = nn.Linear(features, self.slot_size, bias=False).to(device)
    self.project_v = nn.Linear(features, self.slot_size, bias=False).to(device)

    # Slot update functions.
    self.gru = nn.GRUCell(self.slot_size, self.slot_size).to(device)
    self.mlp = nn.Sequential(
        nn.Linear(self.slot_size, self.mlp_hidden_size), nn.ReLU(),
        nn.Linear(self.mlp_hidden_size, self.slot_size)
    ).to(device)

  def forward(self, x):
    b, _, c = x.size()

    # Layer norm 
    inputs = self.norm_inputs(x)     # Shape: [batch_size, num_inputs, input_size].
    k = self.project_k(inputs)       # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)       # Shape: [batch_size, num_inputs, slot_size].

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(b, self.num_slots, self.slot_size).to(self.device)

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
      slots_prev = slots
      slots = self.norm_slots(slots)

      # Attention.
      q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
      q *= self.slot_size ** -0.5  # Normalization.
      attn_logits = torch.bmm(k, torch.transpose(q, 1, 2)) # k.qT (b, num_inputs, slot_size) x (b, slot_size, num_slots)
      attn = F.softmax(attn_logits, dim=-1)   # `attn` has shape: [batch_size, num_inputs, num_slots].
      # `attn` has shape: [batch_size, num_inputs, num_slots].

      # Weigted mean.
      attn = attn + self.epsilon
      attn = attn / torch.sum(attn, dim=-2, keepdim=True)
      updates = torch.bmm(torch.transpose(attn, 1, 2), v)   # `updates` has shape: [batch_size, num_slots, slot_size].
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # Slot update.
      updated_slots = []
      for i in range(self.num_slots):
          update = updates[:, i, :]
          prev_slot = slots_prev[:, i, :]
          updated_slot = self.gru(update, prev_slot)
          updated_slots.append(updated_slot)
      
      slots = torch.stack(updated_slots).permute(1, 0, 2)
      slots = slots + self.mlp(self.norm_mlp(slots))

    return slots