import jax.numpy as jnp
from jax import vmap, jit, grad


def get_single_kl(p_z_covar, p_z_mu, q_z_covar, q_z_mu, opt):
    mu_diff = p_z_mu - q_z_mu
    kl_loss = 0.5 * (   jnp.log(jnp.linalg.det(p_z_covar)) - \
                        jnp.log(jnp.linalg.det(q_z_covar)) - \
                        opt.num_nodes + \
                        jnp.trace( jnp.matmul(jnp.linalg.inv(p_z_covar), q_z_covar) ) + \
                        jnp.matmul(jnp.matmul(jnp.transpose(mu_diff), jnp.linalg.inv(p_z_covar)), mu_diff)
                    )

    return kl_loss

get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 
kl_over_zs = lambda p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt: jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), (0))(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
mse_over_recons = lambda recons, x: jnp.mean(vmap(get_mse, (0, None), 0)(recons, x))

def get_mse_and_kls(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt):
    mse_loss, kl_z_loss, loss = 0., 0., 0.
    mse_loss += mse_over_recons(recons, x) / opt.proj_dims
    kl_z_loss += kl_over_zs(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt) / opt.num_nodes
    loss = (mse_loss + (opt.beta * kl_z_loss)) 
    return mse_loss, kl_z_loss, loss

def loss_fn(params, z_rng, z, theta, sf_baseline, data, interv_targets, 
            step, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs):
    
    recons, _, q_z_mus, q_z_covars, _, _, _, _, _ = dibs.apply({'params': params}, z_rng, z, theta, sf_baseline, data, interv_targets, step)
    mse_loss, kl_z_loss, loss = get_mse_and_kls(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt)
    return loss

def calc_loss(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, 
                pred_zs, opt, z_gt):
    z_dist = 0.
    mse_loss, kl_z_loss, loss = get_mse_and_kls(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt)
    z_dist += mse_over_recons(pred_zs, z_gt)
    return loss, mse_loss, kl_z_loss, z_dist