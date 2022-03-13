import os, sys, pdb, graphical_models, imageio, wandb
sys.path.append('dibs_/')
sys.path.append('../dibs_new/')


from dibs_new.dibs.target import make_nonlinear_gaussian_model, make_linear_gaussian_model
from dibs_new.dibs.inference import JointDiBS
from dibs_new.dibs.utils import visualize_ground_truth
from dibs_new.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

import jax, pdb
import jax.random as random

n_vars = 4
n_samples=100
steps = 20000
grad_estimator_z = 'reparam'

key = random.PRNGKey(123)
print(f"JAX backend: {jax.default_backend()}")

key, subk = random.split(key)
data, model = make_linear_gaussian_model(key = key, n_vars = n_vars, 
                        graph_prior_str = 'sf', 
                        edges_per_node = 1.0,
                        obs_noise = 0.1)
pdb.set_trace()
