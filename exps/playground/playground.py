import os, sys, pdb, graphical_models, imageio, wandb
sys.path.append('dibs_/')

from dibs_.dibs.target import make_nonlinear_gaussian_model, make_linear_gaussian_model
from dibs_.dibs.inference import JointDiBS
from dibs_.dibs.utils import visualize_ground_truth
from dibs_.dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

import jax
import jax.random as random

n_vars = 50
n_samples=100
steps = 20000
grad_estimator_z = 'reparam'

key = random.PRNGKey(123)
print(f"JAX backend: {jax.default_backend()}")

key, subk = random.split(key)
data, model = make_nonlinear_gaussian_model(key=subk, n_vars=n_vars, graph_prior_str="sf")

visualize_ground_truth(data.g)
dibs = JointDiBS(x=data.x, inference_model=model)
key, subk = random.split(key)
print(data.x.shape)
print(grad_estimator_z)
dibs = JointDiBS(x=data.x, inference_model=model, grad_estimator_z=grad_estimator_z)
key, subk = random.split(key)
gs, thetas = dibs.sample(key=subk, n_particles=20, steps=steps, callback_every=100, callback=dibs.visualize_callback())

dibs_empirical = dibs.get_empirical(gs, thetas)
dibs_mixture = dibs.get_mixture(gs, thetas)

for descr, dist in [('DiBS ', dibs_empirical), ('DiBS+', dibs_mixture)]:
    
    eshd = expected_shd(dist=dist, g=data.g)        
    auroc = threshold_metrics(dist=dist, g=data.g)['roc_auc']
    negll = neg_ave_log_likelihood(dist=dist, eltwise_log_likelihood=dibs.eltwise_log_likelihood, x=data.x_ho)
    
    print(f'{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}    neg. LL {negll:5.2f}')