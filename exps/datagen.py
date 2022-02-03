import networkx as nx 
from modules.data.erdos_renyi import ER
from modules.BGe import BGe
import matplotlib.pyplot as plt
from os.path import join

def generate_data(opt, logdir):
    train_dataloader = ER(num_nodes = opt.num_nodes, exp_edges = opt.exp_edges, 
                            noise_type = opt.noise_type, noise_sigma = opt.noise_sigma, 
                            num_samples = opt.num_samples, mu_prior = opt.theta_mu, 
                            sigma_prior = opt.theta_sigma, seed = opt.data_seed, 
                            project=opt.proj, proj_dims=opt.proj_dims, noise_mu=opt.noise_mu)

    nx.draw(train_dataloader.graph, with_labels=True, font_weight='bold', node_size=1000, font_size=25, arrowsize=40, node_color='#FFFF00') # save ground truth graph
    plt.savefig(join(logdir,'gt_graph.png'))
    return train_dataloader