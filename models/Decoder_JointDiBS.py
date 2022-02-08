import sys
sys.path.append('..')
sys.path.append('exps')

import jax.numpy as jnp
from jax import random, vmap, grad, device_put, jit
from flax import linen as nn
from jax.ops import index, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.scipy.special import logsumexp, gammaln

import datagen, utils
from dibs_new.dibs.target import make_nonlinear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs_new.dibs.models import DenseNonlinearGaussian
from modules.Decoder_DiBS_nets import Decoder, Z_mu_logvar_Net

class Decoder_JointDiBS(nn.Module):
    num_nodes: int
    proj_dims: int
    model: DenseNonlinearGaussian
    alpha_linear: float
    grad_estimator: str = 'score'
    known_ED: bool = False
    linear_decoder: bool = False

    def setup(self):
        self.n_vars = self.num_nodes
        
        self.dibs_model = JointDiBS(n_vars=self.num_nodes, 
                                    inference_model=self.model, 
                                    alpha_linear=self.alpha_linear, 
                                    grad_estimator_z=self.grad_estimator)

        if self.known_ED is False:  self.decoder = Decoder(self.proj_dims, self.linear_decoder)
        self.z_net = Z_mu_logvar_Net(self.num_nodes, self.proj_dims)

        print("Loaded Decoder Joint DIBS")

    def __call__(self, z_rng, z):
        return self.decoder(z)

