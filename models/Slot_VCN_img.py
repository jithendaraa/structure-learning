import sys
sys.path.append('..')
from os.path import join

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from modules.SoftPosnEmbed import SoftPositionEmbed
from modules.SlotAttention import SlotAttention
from modules.data import distributions
from modules.AutoregressiveBase import AutoregressiveBase

import utils
import vcn_utils 


class Slot_VCN_img(nn.Module):
    def __init__(self, opt, resolution, num_slots, num_iterations, sparsity_factor, gibbs_temp, device):
        super(Slot_VCN_img, self).__init__()

        assert num_slots == opt.num_nodes
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_nodes = opt.num_nodes
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor
        self.num_iterations = num_iterations
        self.opt = opt
        self.device = device
        self.baseline = 0.
        self.adjacency_lists, self.pres_graph = [], {}
        self.dag_adj_matrices = []
        self.dag_graph_log_likelihoods = []
        self.get_all_digraphs()

        if self.opt.anneal:	self.gibbs_update = self._gibbs_update
        else:				self.gibbs_update = None
        self.init_gibbs_dist()
        encoder_out_ch = opt.encoder_out_ch
        self.decoder_initial_size = (8, 8) # should be div by slot_size
        
        # Networks for encoder parts of Slot attention
        self.cnn_encoder = CNN_Encoder(in_ch=opt.channels, out_ch=encoder_out_ch, device=device)
        self.pos_encoder = SoftPositionEmbed(self.resolution, encoder_out_ch, device=device)
        self.layer_norm = nn.LayerNorm(encoder_out_ch).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_out_ch, 64), nn.ReLU(),
            nn.Linear(64, encoder_out_ch)
        ).to(device)
        self.slot_attention = SlotAttention(encoder_out_ch, num_iterations, num_slots, opt.slot_size, device=device)

        # Network q_phi(G) to sample graphs
        if not opt.no_autoreg_base:
            self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, hidden_dim=opt.hidden_dims, device = device, temp_rsample = 0.1).to(device)
        else:
            NotImplemented('Have not implemented factorised version yet (only autoregressive works)')

        # Networks for slotwise decoding: Position decode + CNN decoder
        self.pos_decoder = SoftPositionEmbed(self.decoder_initial_size, opt.slot_size, device=device)
        self.cnn_decoder = CNN_Decoder(out_ch=opt.channels, device=device)

        print("Initialised model: Slot_VCN_img")

    def get_all_digraphs(self, only_dags=True):
        self.directed_graphs, self.dags, self.dag_adj_matrices = [], [], []
        self.adjacency_lists, self.pres_graph = [], {}
        num_nodes = self.num_nodes
        all_possible_edges = num_nodes * (num_nodes - 1)
        for num_edges in range(all_possible_edges+1):
            adj_list = np.zeros((all_possible_edges), dtype=int).tolist()
            self.solve(adj_list.copy(), num_edges, 0, len(adj_list)-1)	# Get all adjacecny lists for directed graphs
        
        self.adjacency_lists = np.array(self.adjacency_lists)
        self.directed_adj_matrices = vcn_utils.vec_to_adj_mat_np(self.adjacency_lists, num_nodes)
        
        for matrix_num in range(len(self.directed_adj_matrices)):
            DG = nx.DiGraph()
            DG.add_nodes_from(np.arange(0, num_nodes).tolist())
            adj_matrix = self.directed_adj_matrices[matrix_num]
			
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i][j] == 1.:
                        DG.add_edge(i, j)
            self.directed_graphs.append(DG)
            is_acyclic = vcn_utils.expm_np(nx.to_numpy_matrix(DG), num_nodes) == 0
            if is_acyclic:	
                self.dags.append(DG)
                self.dag_adj_matrices.append(torch.from_numpy(adj_matrix).to(self.device))
                
    
    def solve(self, adj_list, edges_left, start, end):
        """ Gets all adjacency lists for directed graph with n node and p edges and saves into self.adjacency lists
        Each element in self.adjacency lists has num_edges*(num_edges - 1) edges from which we can get adj matrices
        """
        if edges_left == 0:
            key = "".join(str(adj_list))
            if key not in self.pres_graph.keys():
                self.pres_graph[key] = True
                self.adjacency_lists.append(adj_list)
            return
		
        if start > end or edges_left < 0: return
		
        self.solve(adj_list.copy(), edges_left, start + 1, end)
        modif_adj_list = adj_list.copy()
        modif_adj_list[start] = 1
        self.solve(modif_adj_list.copy(), edges_left - 1, start + 1, end)
        return

    def init_gibbs_dist(self):
        if self.opt.num_nodes <= 4:
            self.gibbs_dist = distributions.GibbsDAGDistributionFull(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)
        else:
            self.gibbs_dist = distributions.GibbsUniformDAGDistribution(self.opt.num_nodes, self.opt.gibbs_temp, self.opt.sparsity_factor)

    def _gibbs_update(self, curr, epoch):
        if epoch < self.opt.steps*0.05:
            return curr
        else:
            return self.opt.gibbs_temp_init + (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))

    def get_enumerated_dags(self, n_samples, bge_model, interv_targets=None):
            """
                For all the enumerated DAGs, get likelihood scores and save it
            """
            dag_file = join((utils.set_tb_logdir(self.opt)), 'enumerated_dags.png')
            self.dag_graph_log_likelihoods = []
            best_ll = -1e5
            print()
            print(f'{len(self.dag_adj_matrices)} DAGs found!')
            nrows, ncols = int(math.ceil(len(self.dag_adj_matrices) / 5.0)), 5

            with torch.no_grad():
                for graph in self.dag_adj_matrices:
                    graph_log_likelihood = bge_model.log_marginal_likelihood_given_g(w = graph.unsqueeze(0).repeat(n_samples, 1, 1), interv_targets=interv_targets).mean().item()
                    self.dag_graph_log_likelihoods.append(graph_log_likelihood)
                    if graph_log_likelihood > best_ll:
                        best_ll = graph_log_likelihood
            
            idxs = np.flip(np.argsort(self.dag_graph_log_likelihoods))
            fig = plt.figure()
            fig.set_size_inches(ncols * 4, nrows * 4)
            count = 0
            for idx in idxs:
                graph = self.dags[idx]
                ax = plt.subplot(nrows, ncols, count+1)
                count += 1
                if round(best_ll, 2) == round(self.dag_graph_log_likelihoods[idx], 2): color = '#00FF00' # neon green
                else: color = '#FFFF00' # yellow
                nx.draw(graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=30, node_color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_title(f'LL: {self.dag_graph_log_likelihoods[idx]:.2f} \n Ground truth', fontsize=23)
            
            plt.tight_layout()
            plt.savefig(dag_file, dpi=50)
            print( f'Saved enumarate DAG at {dag_file}' )
            plt.show()


    def forward(self, bge_model, step, interv_targets = None):

        assert self.opt.num_nodes == self.opt.num_slots
        assert self.opt.slot_space in ['1d'] 

        n_samples = self.opt.batch_size

        if step == 1:
            self.get_enumerated_dags(n_samples, bge_model, interv_targets)

        # 1. Encode image to lower dims, position encode and flatten it and pass through slot attention 
        # (get z1...zk | x) k == num_slots == num_nodes
        x = self.input_images
        b = x.size()[0]
        x = self.cnn_encoder(x)
        x = self.pos_encoder(x) # b, c, h, w
        x = spatial_flatten(x)    # FLatten spatial dimension
        x = self.mlp(self.layer_norm(x.permute(0, 2, 1))).permute(0, 2, 1)
        slots = self.slot_attention(x.permute(0, 2, 1)) # pass as b, spatial_dims, c; `slots` has shape: [batch_size, num_slots, slot_size].

        # 2. Autoregressively sample G1..GL from approximate posterior q_phi(G) and get adj matrices
        samples = self.graph_dist.sample([n_samples]) 
        G = vcn_utils.vec_to_adj_mat(samples, self.num_nodes)
        dagness = vcn_utils.expm(G, self.num_nodes)
        self.update_gibbs_temp(step)

        # 3. Get log prior p(G) and log posterior q_phi(G) for causal graph
        log_prior_graph = - (self.gibbs_temp*dagness + self.sparsity_factor*torch.sum(G, axis = [-1, -2]))
        posterior_log_probs = self.graph_dist.log_prob(samples).squeeze()

        # 4. Using BGe score, analytically get log likelihood P(z1...zk|G) for the causal graph G
        log_likelihood = bge_model.log_marginal_likelihood_given_g(G, interv_targets, slots)

        # 5. Get D_KL for graph = KL (approx posterior || prior)
        kl_graph = posterior_log_probs - log_prior_graph

        # 6. Spatial broadcast, posn decode and cnn decode the slots z1..zk to 
        x = spatial_broadcast(slots, self.decoder_initial_size) # b*num_slots, c, h, w
        x = self.pos_decoder(x)
        x = self.cnn_decoder(x)     # [b*num_slots, c+1, h, w].

        # 7. Get slotwise recons x_hat_1..x_hat_k and masks m1..mk
        recons, masks = utils.unstack_and_split(x, batch_size=b)

        # 8. Normalize alpha masks over slots and combine per slots reconstructions. 
        # Get final prediction x_hat = Sum (x_hat_i) = Sum (recons_i * masks_i)
        masks = F.softmax(masks, dim=1)
        x_hat = torch.sum(recons * masks, dim=1)  # Recombine image.

        return x_hat, recons, masks, slots, log_likelihood, kl_graph, posterior_log_probs

    def get_prediction(self, batch_dict, bge_model, step):
        np_clip_convert = lambda x: np.clip(((x + 1) / 2) * 255.0, 0. , 255.).astype(np.uint8)
        torch_clamp_convert = lambda x: torch.clamp(((x + 1) / 2) * 255.0, 0., 255.).to(torch.int8)
        
        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self(bge_model, step)
        self.pred, recons, masks, slots, self.log_likelihood, self.kl_graph, self.posterior_log_probs = self(bge_model, step)

        weighted_recon = recons * masks
        weighted_recon = weighted_recon[0].detach().cpu().numpy()
        slot_recons, slot_masks = recons[0].detach().cpu().numpy(), masks[0].detach().cpu().numpy()
        slot_recons, slot_masks, weighted_recon = np.moveaxis(slot_recons, -3, -1), np.moveaxis(slot_masks, -3, -1), np.moveaxis(weighted_recon, -3, -1)

        return torch_clamp_convert(self.pred), np_clip_convert(slot_recons), slot_masks, np_clip_convert(weighted_recon), slots

    def modify_edges(self, bge_model, interv_targets=None):
        nrows, ncols = int(math.ceil(len(self.dag_adj_matrices) / 5.0)), 5
        idxs = np.flip(np.argsort(self.dag_graph_log_likelihoods))
        ncols, nrows = 5, 
        fig = plt.figure()
        fig.set_size_inches(ncols * 4, nrows * 4)

        with torch.no_grad():

            for idx in idxs:
                dag_adj_mat = self.dag_adj_matrices[idx]
                x = self.input_images
                b = x.size()[0]
                x = self.cnn_encoder(x)
                x = self.pos_encoder(x) # b, c, h, w
                x = spatial_flatten(x)    # FLatten spatial dimension
                x = self.mlp(self.layer_norm(x.permute(0, 2, 1))).permute(0, 2, 1)
                slots = self.slot_attention(x.permute(0, 2, 1))
                x = spatial_broadcast(slots, self.decoder_initial_size) # b*num_slots, c, h, w
                x = self.pos_decoder(x)
                x = self.cnn_decoder(x)     
                recons, masks = utils.unstack_and_split(x, batch_size=b)
                masks = F.softmax(masks, dim=1)
                x_hat = torch.sum(recons * masks, dim=1)
                print(x_hat)

    def get_loss(self):
        # ELBO loss for graph
        neg_log_likeli, kl_loss = - self.log_likelihood, self.kl_graph
        score_val = ( neg_log_likeli + kl_loss ).detach()   # TODO why .detach()?
        per_sample_elbo = self.posterior_log_probs*(score_val-self.baseline)
        self.baseline = 0.95 * self.baseline + 0.05 * score_val.mean() 
        graph_loss = (per_sample_elbo).mean()

        # Image reconstruction loss
        recon_loss = F.mse_loss(self.pred, self.ground_truth) 

        # Graph loss + image loss
        total_loss = graph_loss + recon_loss

        loss_dict = {
	    	'Neg. log likelihood (G)': neg_log_likeli.mean().item(),
	    	'KL loss (G)':	kl_loss.mean().item(),
	    	'Graph loss (G)': (neg_log_likeli + kl_loss).mean().item(),
	    	'Per sample ELBO loss (G)': graph_loss.item(),
            'Reconstruction loss': recon_loss.item(),
            'Total loss (G + image)': total_loss.item()
	    } 

        print()
        print(f'Neg. log likelihood (G): {round(loss_dict["Neg. log likelihood (G)"], 2)} | KL loss (G): {round(loss_dict["KL loss (G)"], 2)} | Graph loss (G): {round(loss_dict["Graph loss (G)"], 2)} | Per sample ELBO loss (G): {round(loss_dict["Per sample ELBO loss (G)"], 2)} | Reconstruction loss: {round(loss_dict["Reconstruction loss"], 2)} | Total loss (G + image): {round(loss_dict["Total loss (G + image)"], 2)} ')
        return total_loss, loss_dict

    def update_gibbs_temp(self, e): 
        if self.gibbs_update is None:
            return 0
        else:
            self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)


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