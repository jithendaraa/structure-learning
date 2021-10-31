import torch
import torch.nn as nn

from modules.AutoregressiveBase import AutoregressiveBase
import vcn_utils as utils



class VCN_img(nn.Module):
    def __init__(self, opt, num_nodes, sparsity_factor, gibbs_temp, device=None):
        super(VCN_img, self).__init__()

        self.opt = opt
        self.datatype = opt.datatype
        self.num_nodes = num_nodes
        self.sparsity_factor = sparsity_factor
        self.gibbs_temp = gibbs_temp
        self.device = device
        self.baseline = 0.

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(opt.channels, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, opt.num_nodes * opt.chan_per_node, 3, 1, 1),
        ).to(device)

        if not opt.no_autoreg_base:
            self.graph_dist = AutoregressiveBase(opt, opt.num_nodes, hidden_dim=opt.hidden_dims, device = device, temp_rsample = 0.1).to(device)
        else:
            NotImplemented('Have not implemented factorised version yet (only autoregressive works)')

        print("Initialised model VCN_img")
    
    def init_gibbs_dist(self):
        pass

    def _gibbs_update(self, curr, epoch):
        if epoch < self.opt.steps*0.05:
            return curr
        else:
            return self.opt.gibbs_temp_init+ (self.opt.gibbs_temp - self.opt.gibbs_temp_init)*(10**(-2 * max(0, (self.opt.steps - 1.1*epoch)/self.opt.steps)))

    def forward(self, bge_model, e, interv_targets = None):
        x = self.input_images
        print("inp_images", x.size())

        x = self.conv_encoder(x)
        x = x.view(-1, self.opt.num_nodes, self.opt.chan_per_node, x.size()[-2], x.size()[-1])
        print("After encoding:", x.size())

        n_samples = self.opt.batch_size
        print(f"inside VCN_img forward bsize(n_samples)")
        
        # 1. Sample L ( n*(n-1) ) graph structures autoregressively, per batch from q_phi
        # for i = 1..L: 
        #   G(i) ~ q_phi(G_i)
        #  samples has dimensions (b, L, 1)
        samples = self.graph_dist.sample([n_samples])
        print("samples", samples.size())

        # posterior_log_probs = approx. P(G | D) = q_phi_(G)
        posterior_log_probs = self.graph_dist.log_prob(samples).squeeze()
        print("posterior lp", posterior_log_probs.shape)
        
        # Get G = adjacency matrix of samples; batch_size, num_nodes, num_nodes
        G = utils.vec_to_adj_mat(samples, self.num_nodes)
        print("G", G.size()) 
        print()

        # Calculate per sample marginal likelihood P(D | G(i)) using BGe score, size (b)
        likelihood = bge_model.log_marginal_likelihood_given_g(G, interv_targets, x)
		# print("likelihood", likelihood.size())

        return None, None, None

    def get_prediction(self, batch_dict, bge_train, e):
        np_clip_convert = lambda x: np.clip(((x + 1) / 2) * 255.0, 0. , 255.).astype(np.uint8)
        torch_clamp_convert = lambda x: torch.clamp(((x + 1) / 2) * 255.0, 0., 255.).to(torch.int8)

        self.input_images = batch_dict['observed_data'].to(self.device) # [-1, 1]
        self.ground_truth = batch_dict['data_to_predict'].to(self.device) # [-1, 1]

        self.likelihood, self.kl_graph, self.posterior_log_probs = self(bge_train, e)


    def get_loss(self):
        pass

    def update_gibbs_temp(self, e): 
        if self.gibbs_update is None:
            return 0
        else:
            self.gibbs_temp =  self.gibbs_update(self.gibbs_temp, e)


