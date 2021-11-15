import torch
import numpy as np
from itertools import product
from sklearn import metrics

# VCN
def vec_to_adj_mat(matrix, num_nodes):
	matrix = matrix.view(-1, num_nodes, num_nodes-1)
	matrix_full = torch.cat((torch.zeros(matrix.shape[0], num_nodes,1).to(matrix.device), matrix), dim = -1)
	for xx in range(num_nodes):
		matrix_full[:,xx] = torch.roll(matrix_full[:,xx], xx, -1) 
	return matrix_full

def vec_to_adj_mat_np(matrix, num_nodes):
	matrix = np.reshape(matrix, (-1, num_nodes, num_nodes-1))
	matrix_full = np.concatenate((np.zeros((matrix.shape[0], num_nodes,1), dtype = matrix.dtype), matrix), axis = -1)
	for xx in range(num_nodes):
		matrix_full[:,xx] = np.roll(matrix_full[:,xx], xx, axis = -1) 
	return matrix_full
    
def matrix_poly(matrix, d):
	x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
	return torch.matrix_power(x, d)

def matrix_poly_np(matrix, d):
	x = np.eye(d) + np.divide(matrix, d)
	return np.linalg.matrix_power(x, d)
  
def expm(A, m):
	expm_A = matrix_poly(A, m)
	# DAGness constraint on graph_prior: tr[e^A(G)] - d
	h_A = expm_A.diagonal(dim1=-2, dim2=-1).sum(-1) - m
	return h_A

def expm_np(A, m):
	expm_A = matrix_poly_np(A, m)
	h_A = np.trace(expm_A) - m
	return h_A

def shd(B_est, B_true):
	"""Compute various accuracy metrics for B_est.

	true positive = predicted association exists in condition in correct direction
	reverse = predicted association exists in condition in opposite direction
	false positive = predicted association does not exist in condition

	Args:
		B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
		B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

	Returns:
		fdr: (reverse + false positive) / prediction positive
		tpr: (true positive) / condition positive
		fpr: (reverse + false positive) / condition negative
		shd: undirected extra + undirected missing + reverse
		nnz: prediction positive

		Taken from https://github.com/xunzheng/notears
	"""
	if (B_est == -1).any():  # cpdag
		if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
			raise ValueError('B_est should take value in {0,1,-1}')
		if ((B_est == -1) & (B_est.T == -1)).any():
			raise ValueError('undirected edge should only appear once')
	else:  # dag
		if not ((B_est == 0) | (B_est == 1)).all():
			raise ValueError('B_est should take value in {0,1}')
		#if not is_dag(B_est):
		#    raise ValueError('B_est should be a DAG')
	d = B_true.shape[0]
	# linear index of nonzeros
	pred_und = np.flatnonzero(B_est == -1)
	pred = np.flatnonzero(B_est == 1)
	cond = np.flatnonzero(B_true)
	cond_reversed = np.flatnonzero(B_true.T)
	cond_skeleton = np.concatenate([cond, cond_reversed])
	# true pos
	true_pos = np.intersect1d(pred, cond, assume_unique=True)
	# treat undirected edge favorably
	true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
	true_pos = np.concatenate([true_pos, true_pos_und])
	# false pos
	false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
	false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
	false_pos = np.concatenate([false_pos, false_pos_und])
	# reverse
	extra = np.setdiff1d(pred, cond, assume_unique=True)
	reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
	# compute ratio
	pred_size = len(pred) + len(pred_und)
	cond_neg_size = 0.5 * d * (d - 1) - len(cond)
	fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
	tpr = float(len(true_pos)) / max(len(cond), 1)
	fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
	# structural hamming distance
	pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
	cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
	extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
	missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
	shd = len(extra_lower) + len(missing_lower) + len(reverse)
	shd_wc = shd + len(pred_und)
	prc = float(len(true_pos)) / max(float(len(true_pos)+len(reverse) + len(false_pos)), 1.)
	rec = tpr

	return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'prc': prc, 'rec' : rec, 'shd': shd, 'shd_wc': shd_wc, 'nnz': pred_size}

def all_combinations(num_nodes, num_classes=2, return_adj = False):
	comb = list(product(list(range(num_classes)),repeat = num_nodes*(num_nodes-1)))
	comb = np.array(comb)
	if return_adj:
		comb = vec_to_adj_mat_np(comb, num_nodes)
	return comb

def full_kl_and_hellinger(model, bge_train, g_dist, device):
    """Compute the KL Divergence and Hellinger distance in lower dimensional settings (d<=4)"""
    bs = 100000

    all_adj = all_combinations(model.num_nodes, return_adj = True).astype(np.float32)
    all_adj_vec = all_combinations(model.num_nodes, return_adj = False).astype(np.float32)
    log_posterior_graph = torch.zeros(len(all_adj))
    log_prob_g = torch.zeros(len(all_adj))
    log_prob_model = torch.zeros(len(all_adj))
    with torch.no_grad():
        for tt in range(0,len(all_adj),bs):
            log_posterior_graph[tt:tt+bs] = bge_train.log_marginal_likelihood_given_g(w = torch.tensor(all_adj[tt:tt+bs]).to(device)).cpu() #Unnormalized Log Probabilities
            log_prob_model[tt:tt+bs] = model.graph_dist.log_prob(torch.tensor(all_adj_vec[tt:tt+bs]).to(device).unsqueeze(2)).cpu().squeeze()
        for tt in range(len(all_adj)):
            log_prob_g[tt] = g_dist.unnormalized_log_prob(g=all_adj[tt])
    graph_p = torch.distributions.categorical.Categorical(logits = log_posterior_graph + log_prob_g)
    graph_q = torch.distributions.categorical.Categorical(logits = log_prob_model)
    hellinger = (1./np.sqrt(2)) * torch.sqrt((torch.sqrt(graph_p.probs) - torch.sqrt(graph_q.probs)).pow(2).sum()) 
    return torch.distributions.kl.kl_divergence(graph_q, graph_p).item(), hellinger.item()


"""Compute the Expected Structural Hamming Distance of the model"""
def exp_shd(model, ground_truth, num_samples = 1000):
	shd_ = 0
	prc = 0.
	rec = 0.
	
	with torch.no_grad():
		samples = model.graph_dist.sample([num_samples])
		G = vec_to_adj_mat(samples, model.num_nodes)
		
		for i in range(num_samples):	
			metrics = shd(G[i].cpu().numpy(), ground_truth)
			shd_ += metrics['shd']
			prc += metrics['prc']
			rec += metrics['rec']
	
	return shd_/num_samples, prc/num_samples, rec/num_samples

def adj_mat_to_vec(matrix_full, num_nodes):
	for xx in range(num_nodes):
		matrix_full[:,xx] = torch.roll(matrix_full[:,xx], -xx, -1) 
	matrix = matrix_full[..., 1:]
	return matrix.reshape(-1, num_nodes*(num_nodes-1))

def auroc(model, ground_truth, num_samples = 1000):
    """Compute the AUROC of the model as given in 
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009202"""

    gt = adj_mat_to_vec(torch.from_numpy(ground_truth).unsqueeze(0), model.num_nodes).numpy().squeeze()
    num_nodes = model.num_nodes
    bs = 10000
    i = 0
    samples = []
    with torch.no_grad():
        while i<num_samples:
            curr = min(bs, num_samples-i)
            samples.append(model.graph_dist.sample([curr]).cpu().numpy().squeeze())
            i+=curr
    samples = np.concatenate(samples, axis = 0)
    samples_mean = np.mean(samples, axis = 0)
    sorted_beliefs_index = np.argsort(samples_mean)[::-1]
    fpr = np.zeros((samples_mean.shape[-1]))
    tpr = np.zeros((samples_mean.shape[-1]))
    tnr = np.zeros((samples_mean.shape[-1]))
    for i in range(samples_mean.shape[-1]):
        indexes = np.zeros((samples_mean.shape[-1]))
        indexes[sorted_beliefs_index[:i]] = 1
        tp = np.sum(np.logical_and(gt == 1, indexes == 1))
        fn = np.sum(np.logical_and(indexes==0 , gt != indexes))
        tn = np.sum(np.logical_and(gt==0, indexes==0))
        fp = np.sum(np.logical_and(indexes==1, gt!=indexes))
        fpr[i] = float(fp)/(fp+tn)
        tpr[i] = float(tp)/(tp + fn)
        tnr[i] = float(tn)/(tn + fp)
    auroc = metrics.auc(fpr, tpr)
    return auroc

