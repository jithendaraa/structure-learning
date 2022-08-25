import sys, pathlib, pdb, os
from os.path import join

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../CausalMBRL')
sys.path.append('../../CausalMBRL/envs')

from tqdm import tqdm
import utils
import ruamel.yaml as yaml

from modules.ColorGen import LinearGaussianColor
from conv_decoder_bcd_utils import *
import envs, wandb
import jax
from jax import numpy as jnp
import numpy as onp
import jax.random as rnd
from jax import config, jit, lax, value_and_grad, vmap
from jax.tree_util import tree_map, tree_multimap
import haiku as hk
from modules.GumbelSinkhorn import GumbelSinkhorn

from loss_fns import *
from tensorflow_probability.substrates.jax.distributions import (Horseshoe,
                                                                 Normal)
from conv_decoder_bcd_eval import *

# ? Parse args
configs = yaml.safe_load((pathlib.Path("../..") / "configs.yaml").read_text())
opt = utils.load_yaml_dibs(configs)
exp_config = vars(opt)

# ? Set seeds
onp.random.seed(0)
rng_key = rnd.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(42)

n = opt.num_samples
d = opt.num_nodes
degree = opt.exp_edges
l_dim = d * (d - 1) // 2
ground_truth_sigmas = opt.noise_sigma * jnp.ones(opt.num_nodes)

if opt.do_ev_noise: noise_dim = 1
else:   noise_dim = d

low = -8.
high = 8.
hard = True
num_bethe_iters = 20
assert opt.train_loss == 'mse'
bs = opt.batches
assert n % bs == 0
num_batches = n // bs
assert opt.dataset == 'chemdata'
proj_dims = (1, 50, 50)
log_stds_max=10.
logdir = utils.set_tb_logdir(opt)

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(opt, low, high)
images = images[:, :, :, 0:1]
log_gt_graph(gt_W, logdir, vars(opt), opt)

class MLP(hk.Module):
    def __init__(self):
        super().__init__()


        self.mlp = hk.Sequential([
            hk.Linear(16), hk.gelu,
            hk.Linear(16), hk.gelu,
            hk.Linear(16), hk.gelu,
            hk.Linear(16), hk.gelu,
            hk.Linear(16)
        ])
    
    def __call__(self, x):
        return self.mlp(x)

class ConvDecoder(hk.Module): 
    def __init__(self, dim):
        super().__init__()
        self.proj_dims = (1, 50, 50)

        # self.decoder = hk.Sequential([
        #         hk.Conv2DTranspose(dim, 3, stride=2), jax.nn.gelu,
        #         hk.Conv2DTranspose(64, 3), jax.nn.gelu,
        #         hk.Conv2DTranspose(64, 3), jax.nn.gelu,
        #         hk.Conv2DTranspose(32, 3), jax.nn.gelu,
        #         hk.Conv2DTranspose(32, 3), jax.nn.gelu,
        #         hk.Conv2DTranspose(16, 3), jax.nn.gelu,
        #         hk.Conv2DTranspose(1, 3), jax.nn.sigmoid
        #     ])

        self.decoder = hk.Sequential([
            hk.Linear(16), jax.nn.gelu,
            hk.Linear(64), jax.nn.gelu,
            hk.Linear(256), jax.nn.gelu,
            hk.Linear(512), jax.nn.gelu,
            hk.Linear(1024), jax.nn.gelu,
            hk.Linear(2048), jax.nn.gelu,
            hk.Linear(2500), jax.nn.sigmoid
        ])

    # def spatial_broadcast(self, z_samples, h_, w_):
    #     flat_z = z_samples.reshape(-1, z_samples.shape[-1])[:, None, None, :]
    #     broadcasted_image = jnp.tile(flat_z, (1, h_, w_, 1))
    #     return broadcasted_image

    def __call__(self, z_samples):
        # For conv
        # h_, w_ = self.proj_dims[-2] // 2, self.proj_dims[-1] // 2  
        # spatial_qz = self.spatial_broadcast(z_samples, h_, w_) # (self.batch_size * self.batches, h_, w_, self.nodewise_hidden_dim)
        # X_recons = self.decoder(spatial_qz)

        # For lin layers
        X_recons = self.decoder(z_samples).reshape(-1, self.proj_dims[-2], self.proj_dims[-1], self.proj_dims[-3])
        return X_recons * 255.


def f(z_samples):
    model = ConvDecoder(opt.num_nodes)
    return model(z_samples)

f = hk.transform(f)

def init_params_and_optimizers():
    model_params = f.init(rng_key, z)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_layers)
    model_opt_params = opt_model.init(model_params)
    return model_params, model_opt_params, opt_model


model_params, model_opt_params, opt_model = init_params_and_optimizers()

@jit
def get_loss(model_params, z_samples, x_data):
    X_recons = f.apply(model_params, rng_key, z_samples)
    mse_loss = get_mse(x_data/255., X_recons/255.)
    return jnp.mean(mse_loss), X_recons

@jit
def update_params(grads, model_opt_params, model_params):
    model_updates, model_opt_params = opt_model.update(grads, model_opt_params, model_params)
    model_params = optax.apply_updates(model_params, model_updates)
    return model_opt_params, model_params

bs = opt.batches

X_recons = f.apply(model_params, rng_key, z)
print(X_recons.shape)

# with tqdm(range(opt.num_steps)) as pbar:  
#     for i in pbar:
#         with tqdm(range(num_batches)) as pbar2:
#             for b in pbar2:
#                 start_idx = b * bs
#                 end_idx = min(n, (b+1) * bs)

#                 x_data, z_data = images[start_idx:end_idx], z[start_idx:end_idx]

#                 (mse_loss, X_recons), grads = value_and_grad(get_loss, argnums=(0), has_aux=True)(model_params, 
#                                                                                     z_data, 
#                                                                                     x_data)

#                 model_opt_params, model_params = update_params(grads, model_opt_params, model_params)    

#                 pbar2.set_postfix(
#                     Batch=f"{b}/{num_batches}",
#                     X_mse=f"{mse_loss:.4f}",
#                 )

#         random_idxs = onp.random.choice(n, bs, replace=False)
#         epoch_loss, epoch_x_pred = get_loss(model_params, z[random_idxs], images[random_idxs])
#         wandb_dict = {'x_mse': mse_loss}

#         if opt.off_wandb is False: 
#             plt.figure()
#             plt.imshow(epoch_x_pred[start_idx][:, :, 0]/255.)
#             plt.savefig(join(logdir, 'pred_image.png'))
#             wandb_dict["graph_structure(GT-pred)/Reconstructed image"] = wandb.Image(join(logdir, 'pred_image.png'))
#             plt.close('all')

#             plt.figure()
#             plt.imshow(images[start_idx][:, :, 0]/255.)
#             plt.savefig(join(logdir, 'gt_image.png'))
#             wandb_dict["graph_structure(GT-pred)/GT image"] = wandb.Image(join(logdir, 'gt_image.png'))
#             plt.close('all')
#             wandb.log(wandb_dict, step=i)


#         pbar.set_postfix(
#             Epoch=i,
#             X_mse=f"{epoch_loss:.4f}",
#         )