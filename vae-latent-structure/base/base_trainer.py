import os, pdb, wandb
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, auc


def get_cross_correlation(pred_latent, true_latent):
    dim= pred_latent.shape[1]
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (np.cov( pred_latent[:,i], true_latent[:,j] )[0,1]) / ( np.std(pred_latent[:,i])*np.std(true_latent[:,j]) )
    
    cost= -1*np.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*np.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
    return score

def from_W(W, dim):
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[np.triu_indices(dim, 1)]
    out_2 = W[np.tril_indices(dim, -1)]
    return np.concatenate([out_1, out_2])

def auroc(Ws, W_true, threshold=0.3):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    _, dim, dim = Ws.shape
    edge_present = np.abs(Ws) > threshold
    prob_edge_present = np.mean(edge_present, axis=0)
    true_edges = from_W(np.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_logger = train_logger

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], start_time)

        # setup visualization writer instance
        writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self, config, gt_bin_W, gt_W, z_true=None, x_true=None):
        """
        Full training logic
        """
        d = len(self.model.gating_params) + 1

        if config["off_wandb"] is False:
            if config["offline_wandb"] is True: os.system('wandb offline')
            else:   os.system('wandb online')
            
            wandb.init(project=config["wandb_project"], 
                        entity=config["wandb_entity"],
                        config=config,
                        settings=wandb.Settings(start_method="fork"))
        
            wandb.run.name = f"Graph VAE: {config['dataset']}_seed_{config['data_seed']}" 
            wandb.run.save()
            plt.imshow(gt_bin_W)
            plt.savefig('gt_w.png')
            plt.close('all')
            wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image('gt_w.png')}, step=0)

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch, config)
            adj_matrix = np.zeros((d, d))
            
            for i in range(len(self.model.gating_params)):
                adj_matrix[i, i+1:] = self.model.gating_params[i][0][0].cpu().detach().numpy()

            adj_matrix = np.where(adj_matrix >= 0.5, 1.0, 0.0)
            shd = np.sum(np.abs(adj_matrix - gt_bin_W))
            # save logged informations into log dict
            log = {'epoch': epoch, 'Evaluations/SHD': shd}
            
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value
            
            wandb_log_dict = log
            wandb_log_dict['loss'] = result['loss']
            _, z_pred = self.model(x_true.cuda())
            mcc_score = get_cross_correlation(z_pred.cpu().detach().numpy(), z_true)
            wandb_log_dict['Evaluations/MCC'] = mcc_score
            auroc_score = auroc(adj_matrix[np.newaxis, :], gt_W)
            wandb_log_dict['Evaluations/AUROC'] = auroc_score

            if config["off_wandb"] is False and epoch % 50 == 0: 
                L_mse = np.mean((gt_W - adj_matrix)**2)
                plt.imshow(adj_matrix)
                plt.savefig('pred_w.png')
                plt.close('all')
                wandb_log_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image('pred_w.png')
                
                if config['dataset'] == "chemdata":
                    plt.imshow(result['pred_images'][0][0].cpu().detach().numpy())
                    plt.savefig('pred_image.png')
                    plt.close('all')

                    plt.imshow(result['gt_image'].cpu().detach().numpy())
                    plt.savefig('gt_image.png')
                    plt.close('all')

                    wandb_log_dict['graph_structure(GT-pred)/Reconstructed image'] = wandb.Image('pred_image.png')
                    wandb_log_dict['graph_structure(GT-pred)/GT image'] = wandb.Image('gt_image.png')

                wandb_log_dict['L_MSE'] = L_mse
                wandb.log(wandb_log_dict, step=epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
