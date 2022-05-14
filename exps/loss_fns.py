import jax.numpy as jnp
from jax import vmap, jit, grad
import pdb


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

    if opt.supervised is True:
        kl_z_loss += kl_over_zs(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt) / opt.num_nodes
    
    loss = (mse_loss + (opt.beta * kl_z_loss)) 
    return mse_loss, kl_z_loss, loss


def loss_fn(params, z_rng, z, theta, sf_baseline, data, interv_targets, 
            step, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt, dibs, dibs_type):
    
    recons, _, q_z_mus, q_z_covars, _, _, _, _ = dibs.apply({'params': params}, z_rng, z, theta, sf_baseline, data, interv_targets, step, dibs_type)
    mse_loss, kl_z_loss, loss = get_mse_and_kls(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt)
    return loss


def calc_loss(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, 
                pred_zs, opt, z_gt, only_z=False):
    loss, mse_loss, kl_z_loss, z_dist = 0., 0., 0., 0.
    if only_z is False:
        mse_loss, kl_z_loss, loss = get_mse_and_kls(recons, x, p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt)
    z_dist += mse_over_recons(pred_zs, z_gt)
    return loss, mse_loss, kl_z_loss, z_dist


def log_prob_X(Xs, log_sigmas, P, L, decoder_matrix, proj_matrix, fix_decoder=False, cov_space=False, s_prior_std=3.0):
    proj_dims = proj_matrix.shape[1]

    n, dim = Xs.shape
    Sigma = jnp.diag(jnp.array([jnp.exp(log_sigmas) ** 2] * dim))

    W = (P @ L @ P.T).T

    if fix_decoder: d_cross = jnp.linalg.pinv(proj_matrix)
    else:   d_cross = jnp.linalg.pinv(decoder_matrix)

    cov_z = jnp.linalg.inv(jnp.eye(dim) - W).T @ Sigma @ jnp.linalg.inv(jnp.eye(dim) - W)
    prec_z = jnp.linalg.inv(cov_z)
    cov_x = decoder_matrix.T @ cov_z @ decoder_matrix

    if cov_space is True:       precision_x = jnp.linalg.inv(cov_x)
    else:                       precision_x = d_cross @ prec_z @ d_cross.T

    log_det_precision = jnp.log(jnp.linalg.det(precision_x))
    def datapoint_exponent(x_): return -0.5 * x_.T @ precision_x @ x_
    log_exponent = vmap(datapoint_exponent)(Xs)

    return (0.5 * (log_det_precision - proj_dims * jnp.log(2 * jnp.pi)) + jnp.sum(log_exponent) / n)




