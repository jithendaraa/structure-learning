import functools
import sys
from networkx.algorithms.centrality import group
from numpy.random.mtrand import sample
sys.path.append('models')
sys.path.append('dibs/')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.VCN_image import VCN_img
import utils
import wandb
import os
from os.path import join
import time
import numpy as np
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import vcn_utils
from time import time
from sklearn import metrics
import graphical_models
import networkx as nx

import optax
import jax
from jax import device_put
import jax.scipy.stats as dist
from flax.training import train_state
import jax.numpy as jnp
from jax import vmap, random, jit, grad
from dibs.kernel import FrobeniusSquaredExponentialKernel
from dibs.inference import MarginalDiBS
from dibs.eval.metrics import expected_shd
from dibs.utils.func import particle_marginal_empirical, particle_marginal_mixture
from dibs.models.linearGaussianEquivalent import BGeJAX

def set_optimizers(model, opt):
    if opt.opti == 'alt' and opt.model in ['VAEVCN']: 
        return [optim.Adam(model.vcn_params, lr=1e-2), optim.Adam(model.vae_params, lr=opt.lr)]
    return [optim.Adam(model.parameters(), lr=opt.lr)]

def train_dibs(target, loader_objs, opt, key):
    # ! Tensorboard setup and log ground truth causal graph
    logdir = utils.set_tb_logdir(opt)
    dag_file = join(logdir, 'sampled_dags.png')
    writer = SummaryWriter(logdir)
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    model = target.inference_model
    gt_graph = loader_objs['train_dataloader'].graph
    adjacency_matrix = loader_objs['adj_matrix']

    data = loader_objs['data'] 
    x = jnp.array(data) 
    no_interv_targets = jnp.zeros(opt.num_nodes).astype(bool) # observational data

    def log_prior(single_w_prob):
        """log p(G) using edge probabilities as G"""    
        return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_likelihood(single_w, data):
        log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=data, interv_targets=no_interv_targets)
        return log_lik

    # ? SVGD + DiBS hyperparams
    n_particles, n_steps = opt.n_particles, opt.num_updates
    
	# ? initialize kernel and algorithm
    kernel = FrobeniusSquaredExponentialKernel(h=opt.h_latent)
    dibs = MarginalDiBS(kernel=kernel, target_log_prior=log_prior, 
                        target_log_marginal_prob=log_likelihood, 
                        alpha_linear=opt.alpha_linear, 
                        grad_estimator_z=opt.grad_estimator)
		
	# ? initialize particles
    key, subk = random.split(key)
    init_particles_z = dibs.sample_initial_random_particles(key=subk, n_particles=n_particles, n_vars=opt.num_nodes)

    key, subk = random.split(key)
    particles_z, _, _ = dibs.sample_particles(key=subk, n_steps=n_steps, init_particles_z=init_particles_z,
                                            opt_state_z = None, sf_baseline=None, data=x, start=0)
    particles_g = dibs.particle_to_g_lim(particles_z)

    def log_likelihood(single_w):
        log_lik = model.log_marginal_likelihood_given_g(w=single_w, data=x, interv_targets=no_interv_targets)
        return log_lik

    eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)	
    dibs_empirical = particle_marginal_empirical(particles_g)
    dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
    eshd_e = expected_shd(dist=dibs_empirical, g=adjacency_matrix)
    eshd_m = expected_shd(dist=dibs_mixture, g=adjacency_matrix)
    
    sampled_graph, mec_or_gt_count = utils.log_dags(particles_g, gt_graph, eshd_e, eshd_m, dag_file)
    writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, 0, dataformats='HWC')

    print("ESHD (empirical):", eshd_e)
    print("ESHD (marginal mixture):", eshd_m)
    print("MEC-GT Recovery %", mec_or_gt_count)

def train_vae_dibs(model, loader_objs, opt, key):
    particles_g, eltwise_log_prob = None, None

    # ! Tensorboard setup and log ground truth causal graph
    logdir = utils.set_tb_logdir(opt)
    writer = SummaryWriter(logdir)
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')

    gt_graph = loader_objs['train_dataloader'].graph
    adjacency_matrix = loader_objs['adj_matrix']
    data = loader_objs['projected_data']
    x = jnp.array(data) 

    key, rng = random.split(key)
    m = model()
    state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=m.init(rng, x, key, adjacency_matrix)['params'],
        tx=optax.adam(opt.lr),
    )


    def train_step(state, batch, z_rng):
        def loss_fn(params):
            recons, log_p_z_given_g, q_zs, q_z_mus, q_z_logvars, particles_g, eltwise_log_prob = m.apply({'params': params}, batch, z_rng, adjacency_matrix)
            n_particles = len(particles_g)
            mse_loss = 0.
            for recon in recons:
                err = (recon - x)
                mse_loss += jnp.mean(jnp.square(err)) 

            kl_z_loss = 0.
            # ? get KL( q(z | G) || P(z|G_i) )
            for i in range(n_particles):
                cov = jnp.diag(jnp.exp(0.5 * q_z_logvars[i]))
                q_z = dist.multivariate_normal.pdf(q_zs[i], q_z_mus[i], cov)
                log_q_z = dist.multivariate_normal.logpdf(q_zs[i], q_z_mus[i], cov)
                kl_z_loss += (q_z * (log_q_z - log_p_z_given_g[i]))

            soft_constraint = 0.
            if opt.soft_constraint is True:
                for g in particles_g:
                    soft_constraint += jnp.linalg.det(jnp.identity(opt.num_nodes) - g)

            loss = (mse_loss + kl_z_loss - soft_constraint) * 1/n_particles
            return loss

        loss = loss_fn(state.params)
    
        # s = time()
        grads = jax.grad(loss_fn)(state.params)
        # print(f'time to get grads {time() - s}s')
        # s = time()
        res = state.apply_gradients(grads=grads)
        # print(f'Time to apply grads {time() - s}s')
        return res, loss

    for step in tqdm(range(opt.steps)):
        key, rng = random.split(key)
        state, loss = train_step(state, x, rng)


        if step % 5 == 0:
            _, _, _, _, _, particles_g, eltwise_log_prob = m.apply({'params': state.params}, x, rng, adjacency_matrix)
            dibs_empirical = particle_marginal_empirical(particles_g)
            dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
            eshd_e = expected_shd(dist=dibs_empirical, g=adjacency_matrix)
            eshd_m = expected_shd(dist=dibs_mixture, g=adjacency_matrix)
            
            sampled_graph = utils.log_dags(particles_g, gt_graph, opt, eshd_e, eshd_m)
            writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, step, dataformats='HWC')
            # writer.add_scalar('Evaluations/Exp. SHD Empirical', eshd_e, step)
            # writer.add_scalar('Evaluations/Exp. SHD Marginal', eshd_m, step)
            # writer.add_scalar('total_losses/Total loss', loss, step)
            
            print(f'Step {step} | Loss {loss}')
            print(f'Expected SHD Marginal: {eshd_m} | Empirical: {eshd_e}')
            print()

def train_decoder_dibs(model, loader_objs, exp_config_dict, opt, key):
    particles_g, eltwise_log_prob, opt_state_z = None, None, None
    sf_baseline = jnp.zeros(opt.n_particles)
    no_interv_targets = jnp.zeros(opt.num_nodes).astype(bool)

    # ! Tensorboard and wandb setup and log ground truth causal graph
    logdir = utils.set_tb_logdir(opt)
    group_name = logdir.split('/')[-1].replace(f"_seed{opt.data_seed}", "")
    dag_file = join(logdir, 'sampled_dags.png')
    writer = SummaryWriter(logdir)
    gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))

    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, 
                    entity=opt.wandb_entity, 
                    config=exp_config_dict, 
                    settings=wandb.Settings(start_method="fork"))
        wandb.run.name = logdir.split('/')[-1]
        wandb.run.save()

    writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
    wandb.log({"graph_structure(GT-pred)/Ground truth": wandb.Image(join(logdir, 'gt_graph.png'))}, step=0)

    gt_graph = loader_objs['train_dataloader'].graph
    gt_graph_cpdag = graphical_models.DAG.from_nx(gt_graph).cpdag()
    projection_matrix = jnp.array(loader_objs['projection_matrix'])
    adj_matrix = loader_objs['adj_matrix'].astype(int)
    gt_samples = loader_objs['data']
    x = jnp.array(loader_objs['projected_data'])
    z_means, z_stds = loader_objs['means'], loader_objs['stds']
    z_gt = jnp.array(gt_samples) 
    z_gt = device_put(z_gt, jax.devices()[0])
    gt = utils.adj_mat_to_vec(torch.from_numpy(adj_matrix).unsqueeze(0), opt.num_nodes).numpy().squeeze()
    
    # ? Prior P(z | G) for KL term - could be sample params or actual params mu and sigma
    p_z_mus, p_z_stds = jnp.array(z_means[opt.z_prior]), jnp.array(z_stds[opt.z_prior])
    print("mu", p_z_mus, "std", p_z_stds)

    m = model()
    key, rng = random.split(key)
    particles_z = utils.sample_initial_random_particles(key, opt.n_particles, opt.num_nodes)
    inference_model = BGeJAX(mean_obs=jnp.zeros(opt.num_nodes), alpha_mu=1.0, alpha_lambd=opt.num_nodes + 2)
    print(adj_matrix)

    def log_likelihood(single_w):
        log_lik = inference_model.log_marginal_likelihood_given_g(w=single_w, data=z_gt, interv_targets=no_interv_targets)
        return log_lik
    
    eltwise_log_prob = vmap(lambda g: log_likelihood(g), 0, 0)
    optimizer = optax.adam(opt.lr)

    state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=m.init(rng, key, particles_z, sf_baseline)['params'],
        tx=optimizer
    )

    def loss_fn(params, z_rng, particles, sf_baseline, step):
        mse_loss, kl_z_loss = 0., 0.
        recons, q_z_mus, q_z_logvars, _, _, _, _, _ = m.apply({'params': params}, z_rng, particles, sf_baseline, step)

        get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 
        v_get_mse = vmap(get_mse, (0, None), 0)
        mse_loss += jnp.sum(v_get_mse(recons, x)) / opt.n_particles

        get_kl = lambda q_z_mu, q_z_logvar, p_z_mu, p_z_std: jnp.sum(-0.5 + (jnp.log(p_z_std) - 0.5 * q_z_logvar) + (jnp.exp(q_z_logvar) + (q_z_mu - p_z_mu)**2)/(2*(p_z_std**2)))
        v_get_kl = vmap(get_kl, (0, 0, None, None), 0)
        kl_z_loss += jnp.sum(v_get_kl(q_z_mus, q_z_logvars, p_z_mus, p_z_stds)) / opt.n_particles
        loss = (mse_loss + (opt.beta * kl_z_loss)) 
        return loss
    
    
    def fast_slow_train_step(state, z_rng, particles, sf_baseline, step):
        dibs_updates = 1
        if opt.algo == 'fast-slow': dibs_updates = opt.num_updates

        grads = jit(grad(jit(loss_fn)))(state.params, z_rng, particles, sf_baseline, step)   # time per train_step() is 14s with jit and 6.8s without
        res = state.apply_gradients(grads=grads)

        # ? mutliple dibs update for one ELBO update for decoder dibs
        # ? Particles_z updated as SVGD transport step z(t+1)(particle m) = z(t)(m) + step_size * phi_z(t)(m)
        for _ in tqdm(range(dibs_updates)):
            recons, q_z_mus, q_z_logvars, phi_z, soft_g, sf_baseline, z_rng, pred_z = jit(m.apply)({'params': res.params}, z_rng, particles, sf_baseline, step)
            particles = particles - (opt.dibs_lr * phi_z)

        mse_loss, kl_z_loss, z_dist, decoder_dist = 0., 0., 0., 0.
        get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 
        v_get_mse = jit(vmap(get_mse, (0, None), 0))
        mse_loss += jnp.mean(v_get_mse(recons, x))

        get_kl = lambda q_z_mu, q_z_logvar, p_z_mu, p_z_std: jnp.sum(-0.5 + (jnp.log(p_z_std) - 0.5 * q_z_logvar) + (jnp.exp(q_z_logvar) + (q_z_mu - p_z_mu)**2)/(2*(p_z_std**2)))
        v_get_kl = jit(vmap(get_kl, (0, 0, None, None), 0))
        kl_z_loss += jnp.mean(v_get_kl(q_z_mus, q_z_logvars, p_z_mus, p_z_stds)) 
        loss = (mse_loss + (opt.beta * kl_z_loss))

        z_dist += jnp.mean(v_get_mse(pred_z, z_gt)) 

        # decoder_dist = mse(decoder_projection, projection matrix)
        # recons_i = dot(pred_z_i, projection_matrix) 
        z, pred_x = pred_z[0], recons[0]
        zT = jnp.transpose(z)
        zTx = jnp.dot(zT, pred_x)
        zTz_inv = jnp.linalg.inv(jnp.dot(zT, z))
        decoder_projection = jnp.dot(zTz_inv, zTx)
        v_get_mse = jit(vmap(jit(get_mse), (0, 0), 0))
        decoder_dist += jnp.mean(v_get_mse(decoder_projection, projection_matrix)) 

        return res, loss, mse_loss, kl_z_loss, q_z_mus, q_z_logvars, soft_g, particles, sf_baseline, z_dist, decoder_dist

    @jit
    def def_train_step(state, z_rng, particles, sf_baseline, step):
        grads = grad(loss_fn)(state.params, z_rng, particles, sf_baseline, step)   # time per train_step() is 14s with jit and 6.8s without
        res = state.apply_gradients(grads=grads)

        # ? mutliple dibs update for one ELBO update for decoder dibs
        # ? Particles_z updated as SVGD transport step z(t+1)(particle m) = z(t)(m) + step_size * phi_z(t)(m)
        recons, q_z_mus, q_z_logvars, phi_z, soft_g, sf_baseline, z_rng, pred_z = m.apply({'params': res.params}, z_rng, particles, sf_baseline, step)
        particles = particles - opt.dibs_lr * phi_z

        mse_loss, kl_z_loss, z_dist, decoder_dist = 0., 0., 0., 0.
        get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 
        v_get_mse = vmap(get_mse, (0, None), 0)
        mse_loss += jnp.mean(v_get_mse(recons, x))

        get_kl = lambda q_z_mu, q_z_logvar, p_z_mu, p_z_std: jnp.sum(-0.5 + (jnp.log(p_z_std) - 0.5 * q_z_logvar) + (jnp.exp(q_z_logvar) + (q_z_mu - p_z_mu)**2)/(2*(p_z_std**2)))
        v_get_kl = vmap(get_kl, (0, 0, None, None), 0)
        kl_z_loss += jnp.mean(v_get_kl(q_z_mus, q_z_logvars, p_z_mus, p_z_stds)) 
        loss = (mse_loss + (opt.beta * kl_z_loss))

        z_dist += jnp.mean(v_get_mse(pred_z, z_gt)) 

        # decoder_dist = mse(decoder_projection, projection matrix)
        # recons_i = dot(pred_z_i, projection_matrix) 
        z, pred_x = pred_z[0], recons[0]
        zT = jnp.transpose(z)
        zTx = jnp.dot(zT, pred_x)
        zTz_inv = jnp.linalg.inv(jnp.dot(zT, z))
        decoder_projection = jnp.dot(zTz_inv, zTx)
        v_get_mse = vmap(get_mse, (0, 0), 0)
        decoder_dist += jnp.sum(v_get_mse(decoder_projection, projection_matrix)) / opt.num_nodes
        
        return res, loss, mse_loss, kl_z_loss, q_z_mus, q_z_logvars, soft_g, particles, sf_baseline, z_dist, decoder_dist

    if opt.algo == 'fast-slow':
        trainer_fn = fast_slow_train_step
        log_freq = 5
    elif opt.algo == 'def':
        trainer_fn = def_train_step
        log_freq = opt.steps // 500

    for step in range(opt.steps):
        key, rng = random.split(key)
        s = time()
        state, loss, mse_loss, kl_z_loss, _, _, soft_g, particles_z, sf_baseline, z_dist, decoder_dist = trainer_fn(state, rng, particles_z, sf_baseline, step)
        loss, mse_loss, kl_z_loss = np.array(loss), np.array(mse_loss), np.array(kl_z_loss) 
        print(f'Step {step} | Loss {loss:4f} | MSE: {mse_loss:4f} | KL: {kl_z_loss} | Time per train step: {(time() - s):2f}s')
        
        if step % log_freq == 0:
            soft_g = np.array(soft_g)
            particles_g = np.random.binomial(1, soft_g, soft_g.shape)   # todo verify
            dibs_empirical = particle_marginal_empirical(particles_g)
            dibs_mixture = particle_marginal_mixture(particles_g, eltwise_log_prob)
            eshd_e = np.array(expected_shd(dist=dibs_empirical, g=adj_matrix))
            eshd_m = np.array(expected_shd(dist=dibs_mixture, g=adj_matrix))
            sampled_graph, mec_gt_count = utils.log_dags(particles_g, gt_graph, eshd_e, eshd_m, dag_file)
            mec_gt_recovery = 100 * (mec_gt_count / opt.n_particles)
            auroc = utils.dibs_auroc(m, rng, state.params, particles_z, sf_baseline, opt.num_nodes, gt, 500 // opt.n_particles)
            
            cpdag_shds = []
            for adj_mat in particles_g:
                G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
                try:
                    G_cpdag = graphical_models.DAG.from_nx(G).cpdag()
                    shd = gt_graph_cpdag.shd(G_cpdag)
                    cpdag_shds.append(shd)
                except:
                    pass
            
            wandb_log_dict = {
                'graph_structure(GT-pred)/Posterior sampled graphs': wandb.Image(sampled_graph),
                'Evaluations/Exp. SHD (Empirical)': eshd_e,
                'Evaluations/Exp. SHD (Marginal)': eshd_m,
                'Evaluations/MEC or GT recovery %': mec_gt_recovery,
                'Evaluations/AUROC': auroc,
                'z_losses/MSE': mse_loss,
                'z_losses/KL': kl_z_loss,
                'z_losses/ELBO': loss,
                'Distances/MSE(Predicted z | z_GT)': np.array(z_dist),
                'Distances/MSE(decoder | projection matrix)': np.array(decoder_dist)
            }

            # ! tensorboard logs
            if len(cpdag_shds) > 0: 
                writer.add_scalar('Evaluations/CPDAG SHD', np.mean(cpdag_shds), step)
                wandb_log_dict['Evaluations/CPDAG SHD'] = np.mean(cpdag_shds)
            
            wandb.log(wandb_log_dict, step=step)

            writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, step, dataformats='HWC')
            writer.add_scalar('Evaluations/Exp. SHD (Empirical)', np.array(eshd_e), step)
            writer.add_scalar('Evaluations/Exp. SHD (Marginal)', np.array(eshd_m), step)
            writer.add_scalar('Evaluations/MEC or GT recovery %', mec_gt_recovery, step)
            writer.add_scalar('Evaluations/AUROC', auroc, step)
            writer.add_scalar('z_losses/MSE', mse_loss, step)
            writer.add_scalar('z_losses/KL', kl_z_loss, step)
            writer.add_scalar('z_losses/ELBO', loss, step)
            writer.add_scalar('Distances/MSE(Predicted z | z_GT)', np.array(z_dist), step)
            writer.add_scalar('Distances/MSE(decoder | projection matrix)', np.array(decoder_dist), step)

            print(f'Expected SHD Marginal: {eshd_m} | Empirical: {eshd_e}')
            print(f"GT-MEC: {mec_gt_recovery} | AUROC: {auroc}")
            if len(cpdag_shds) > 0:
                print(f"CPDAG SHD: {np.mean(cpdag_shds)}")
        print()


def train_model(model, loader_objs, exp_config_dict, opt, device, key=None):
    # * DIBS, or a variant thereof, uses jax so train those in a separate function. This function trains only torch models
    if opt.model in ['DIBS']: 
        train_dibs(model, loader_objs, opt, key)
        return
    elif opt.model in ['VAE_DIBS']:
        train_vae_dibs(model, loader_objs, opt, key)
        return
    elif opt.model in ['Decoder_DIBS']:
        train_decoder_dibs(model, loader_objs, exp_config_dict, opt, key)
        return

    pred_gt, time_epoch, likelihood, kl_graph, elbo_train, vae_elbo = None, [], [], [], [], []
    optimizers = set_optimizers(model, opt)
    logdir = utils.set_tb_logdir(opt)
    writer = SummaryWriter(logdir)
    
    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')

    if opt.off_wandb is False:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, config=exp_config_dict, settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        wandb.watch(model)

    start_time = time()
    for step in tqdm(range(opt.steps)):
        start_step_time = time()
        pred, gt, loss, loss_dict, media_dict = train_batch(model, loader_objs, optimizers, opt, writer, step, start_time)
        time_epoch.append(time() - start_step_time)

        if opt.model in ['SlotAttention_img', 'VCN_img', 'Slot_VCN_img', 'GraphVAE']:
            pred_gt = torch.cat((pred[:opt.log_batches].detach().cpu(), gt[:opt.log_batches].cpu()), 0).numpy()
            pred_gt = np.moveaxis(pred_gt, -3, -1)
        
        elif opt.model in ['VCN']:
            vae_elbo = evaluate_vcn(opt, writer, model, loader_objs['bge_train'], step+1, vae_elbo, device, loss_dict, time_epoch, loader_objs['train_dataloader'])

        # logging to tensorboard and wandb
        utils.log_to_tb(opt, writer, loss_dict, step, pred_gt)
        utils.log_to_wandb(opt, step, loss_dict, media_dict, pred_gt)
        utils.save_model_params(model, optimizers, opt, step, opt.ckpt_save_freq) # Save model params

    if opt.model in ['Slot_VCN_img']:
        utils.log_enumerated_dags_to_tb(writer, logdir, opt)

    elif opt.model in ['VCN']:
        # Log ground truth graph, enumerated DAGs and posterior sampled graphs
        gt_graph = loader_objs['train_dataloader'].graph
        gt_graph_image = np.asarray(imageio.imread(join(logdir, 'gt_graph.png')))
        writer.add_image('graph_structure(GT-pred)/Ground truth', gt_graph_image, 0, dataformats='HWC')
        
        utils.log_enumerated_dags_to_tb(writer, logdir, opt)

        dag_file = model.get_sampled_graph_frequency_plot(loader_objs['bge_train'], gt_graph, 1000, None)
        sampled_graph = np.asarray(imageio.imread(dag_file))
        writer.add_image('graph_structure(GT-pred)/Posterior sampled graphs', sampled_graph, 0, dataformats='HWC')

    writer.flush()
    writer.close()


def train_batch(model, loader_objs, optimizers, opt, writer, step, start_time):
    loss_dict, media_dict = {}, {}
    prediction, gt = None, None
    
    # Zero out gradients
    for optimizer in optimizers: optimizer.zero_grad()
    model.train()

    if opt.model in ['SlotAttention_img', 'Slot_VCN_img']: 
        train_loss, loss_dict, prediction, gt, media_dict = train_slot(opt, loader_objs, model, writer, step, start_time)

    elif opt.model in ['VCN']: 
        train_loss, loss_dict = train_vcn(opt, loader_objs, model, writer, step, start_time)
    
    elif opt.model in ['VCN_img']:
        prediction, gt, train_loss, loss_dict = train_image_vcn(opt, loader_objs, model, writer, step)

    elif opt.model in ['GraphVAE']:
        prediction, gt, train_loss, loss_dict = train_graph_vae(opt, loader_objs, model, writer, step, start_time)

    elif opt.model in ['VAEVCN']:
        train_loss, loss_dict = train_vae_vcn(opt, loader_objs, model, writer, step, start_time)

    if opt.clip != -1:  torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt.clip))
    train_loss.backward()

    if opt.opti == 'alt':
        assert len(optimizers) == 2
        if (step % 2 == 0): 
            optimizers[0].step()
        else: 
            optimizers[1].step()
    else:
        # In this case we always have only one optimizer
        assert len(optimizers) == 1
        optimizers[0].step()

    grad_norm = utils.get_grad_norm(model.parameters(), opt.opti == 'alt')
    loss_dict['Gradient Norm'] = grad_norm
    return prediction, gt, train_loss, loss_dict, media_dict

# Vanilla Slot attention and Slot-Image_VCN
def train_slot(opt, loader_objs, model, writer, step, start_time):
    # For models: Slot attention or Slot Image VCN 
    media_dict = {}
    bge_train = loader_objs['bge_train']
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    keys = ['Slot reconstructions', 'Weighted reconstructions', 'Slotwise masks']
    model.get_prediction(batch_dict, bge_train, step)
    recon_combined, slot_recons, slot_masks, weighted_recon, _ = model.get_prediction(batch_dict, bge_train, step)
    train_loss, loss_dict = model.get_loss()
    
    gt_np = model.ground_truth[0].detach().cpu().numpy()
    gt_np = np.moveaxis(gt_np, -3, -1)
    gt_np = ((gt_np + 1)/2) * 255.0
    media_dict['Slot reconstructions'] = [wandb.Image(m) for m in slot_recons]
    media_dict['Weighted reconstructions'] = [wandb.Image(m) for m in weighted_recon]
    media_dict['Slotwise masks'] = [wandb.Image(m) for m in slot_masks]
    values = [slot_recons / 255.0, weighted_recon / 255.0, slot_masks]
    utils.log_images_to_tb(opt, step, keys, values, writer, 'NHWC')
    prediction, gt = recon_combined, ((model.ground_truth + 1)/2) * 255.0
    
    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict, prediction, gt, media_dict

# Train and evaluate Vanilla VCN
def train_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    model.get_prediction(bge_model, step, loader_objs['train_dataloader'].graph)
    train_loss, loss_dict, _ = model.get_loss()
    if step == 0:   
        logdir = utils.set_tb_logdir(opt)
        if opt.num_nodes < 5:
            # ! Num. DAGs grow super-exponentially with num_nodes. 643 DAGs for 4 nodes.
            utils.log_enumerated_dags_to_tb(writer, logdir, opt)
    if step % 100 == 0: 
        tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {train_loss.item():.5f} | Time: {round((time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict

def evaluate_vcn(opt, writer, model, bge_test, step, vae_elbo, device, loss_dict, time_epoch, train_data):
    gt_graph = train_data.graph
    model.eval()
    with torch.no_grad():
        model.get_prediction(bge_test, step, gt_graph) 
        _, loss_dict, _ = model.get_loss()
        elbo = loss_dict['graph_losses/Total loss']
        
    vae_elbo.append(elbo)

    if step % 100 == 0:
        kl_full, hellinger_full = 0., 0.
        if opt.num_nodes<=4:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)

        print('Step {}:  TRAIN - ELBO: {:.5f} likelihood: {:.5f} kl graph: {:.5f} VAL-ELBO: {:.5f} Temp Target {:.4f} Time {:.2f}'.\
                    format(step, loss_dict['graph_losses/Per sample loss'], loss_dict['graph_losses/Neg. log likelihood'], \
                    loss_dict['graph_losses/KL loss'], elbo, model.gibbs_temp, np.sum(time_epoch[step-100:step]), \
                    flush = True))

        # shd -> Expected structural hamming distance
        shd, prc, rec = vcn_utils.exp_shd(model, train_data.adjacency_matrix)
        kl_full, hellinger_full, auroc_score  = 0., 0., 0.

        if opt.num_nodes <= 3:
            kl_full, hellinger_full = vcn_utils.full_kl_and_hellinger(model, bge_test, model.gibbs_dist, device)
            writer.add_scalar('Evaluations/Hellinger Full', hellinger_full, step)
        else:
            auroc_score = vcn_utils.auroc(model, train_data.adjacency_matrix)
            writer.add_scalar('Evaluations/AUROC', auroc_score, step)
        
        writer.add_scalar('Evaluations/Exp. SHD', shd, step)
        writer.add_scalar('Evaluations/Exp. Precision', prc, step)
        writer.add_scalar('Evaluations/Exp. Recall', rec, step)
        
        print('Exp SHD:', shd,  'Exp Precision:', prc, 'Exp Recall:', rec, \
            'Kl_full:', kl_full, 'hellinger_full:', hellinger_full,\
        'auroc:', auroc_score)
        print()

    return vae_elbo


# Image-VCN
def train_image_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    enc_inp, prediction = model.get_prediction(batch_dict, bge_model, step)
    utils.log_encodings_per_node_to_tb(opt, writer, enc_inp, step)  # enc_inp has shape [num_nodes, chan_per_nodes, h, w]
    train_loss, loss_dict, _ = model.get_loss()
    gt = ((model.ground_truth + 1)/2) * 255.0
    
    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time() - start_time) / 60, 3)}m")
    return prediction, gt, train_loss, loss_dict
        
# Graph VAE
def train_graph_vae(opt, loader_objs, model, writer, step, start_time):
    data_dict = utils.get_data_dict(opt, loader_objs['train_dataloader'])
    batch_dict = utils.get_next_batch(data_dict, opt)
    prediction = model.get_prediction(batch_dict, step)
    train_loss, loss_dict = model.get_loss(step)
    gt = ((model.ground_truth + 1)/2) * 255.0

    tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {round(train_loss.item(), 5)} | Time: {round((time() - start_time) / 60, 3)}m")
    return prediction, gt, train_loss, loss_dict

# VAE VCN
def train_vae_vcn(opt, loader_objs, model, writer, step, start_time):
    bge_model = loader_objs['bge_train']
    model.get_prediction(loader_objs, step)
    train_loss, total_loss, loss_dict = model.get_loss(step)
    if step == 0:   
        logdir = utils.set_tb_logdir(opt)
        # utils.log_enumerated_dags_to_tb(writer, logdir, opt)
    if step % 100 == 0: 
        tqdm.write(f"[Step {step}/{opt.steps}] | Step loss {total_loss:.5f} | Time: {round((time.time() - start_time) / 60, 3)}m")
    return train_loss, loss_dict
