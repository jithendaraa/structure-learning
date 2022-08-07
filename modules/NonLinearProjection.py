import haiku as hk
import jax, pdb
from jax import vmap
import jax.numpy as jnp

class NonLinearProjection(hk.Module):
    def __init__(self, proj_dims):
        super().__init__()

        # TODO: See what kind of nonlinearity to add
        self.nonlinear_projector = hk.Sequential([
                                        hk.Linear(proj_dims), jax.nn.relu,
                                        hk.Linear(proj_dims), jax.nn.relu,
                                        hk.Linear(proj_dims)
                                    ])

    def __call__(self, z):
        x = vmap(self.nonlinear_projector, (0), (0))(z)
        return x

def projection_forward(proj_dims, z):
    projector_model = NonLinearProjection(proj_dims)
    return projector_model(z)

def init_projection_params(rng_key, n, d, proj_dims):
    forward = hk.transform(projection_forward)
    z = jnp.ones((n, d))

    def init_params(rng_key):
        projection_model_params = forward.init(next(rng_key), proj_dims, z)
        return projection_model_params
    
    return forward, init_params(rng_key)