import sys, pdb, pathlib
sys.path.append("..")
import numpy as np
import networkx as nx

def generate_colors(opt, chem_data, low, high, interv_low, interv_high): 
    n = opt.num_samples
    d = opt.num_nodes
    n_interv_sets = opt.n_interv_sets
    interv_data_per_set = (opt.num_samples - opt.obs_data) // n_interv_sets
    obs_data = chem_data.obs_X

    interv_data = []
    interv_values = np.random.uniform(low=interv_low, high=interv_high, size=(n, d))
    interv_targets = np.full((n, d), False)

    for i in range(n_interv_sets):
        interv_k_nodes = np.random.randint(1, d)
        intervened_node_idxs = np.random.choice(d, interv_k_nodes, replace=False)
        interv_targets[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set, intervened_node_idxs] = True
        interv_value = interv_values[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set]

        interv_data_ = chem_data.intervene_sem(chem_data.W, 
                                                interv_data_per_set, 
                                                opt.sem_type,
                                                sigmas=[opt.noise_sigma], 
                                                idx_to_fix=intervened_node_idxs, 
                                                values_to_fix=interv_value, 
                                                low=low, 
                                                high=high)
        if i == 0:  interv_data = interv_data_
        else: interv_data = np.concatenate((interv_data, interv_data_), axis=0)

    z = np.concatenate((obs_data, interv_data), axis=0)

    return z, interv_targets, interv_values

def intervene_sem(
        W, n, sem_type, sigmas=None, idx_to_fix=None, values_to_fix=None,
        low=-10., high=10.
    ):
        """Simulate samples from SEM with specified type of noise.
        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM
            idx_to_fix: intervened node or list of intervened nodes
            values_to_fix: intervened values
        Returns:
            X: [n,d] sample matrix
        """

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d])
        if len(sigmas) == 1:
            sigmas = np.ones(d) * sigmas

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        assert sem_type == "linear-gauss"

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if isinstance(idx_to_fix, int) and j == idx_to_fix:
                X[:, j] = values_to_fix[:, j]
            elif len(np.argwhere(idx_to_fix == j)) > 0:
                X[:, j] = values_to_fix[:, j]
            else:
                eta = X[:, parents].dot(W[parents, j])
                X[:, j] = eta + np.random.normal(scale=sigmas[j], size=n)

        return np.clip(X, low, high)


def generate_samples(d, W, sem_type, noise_sigma, low, high, num_test_samples, interv_low, interv_high):
    interv_data = []
    test_interv_values = np.random.uniform(low=interv_low, high=interv_high, size=(num_test_samples, d))
    interv_targets = np.full((num_test_samples, d), False)

    for i in range(num_test_samples):
        interv_k_nodes = np.random.randint(1, d)
        intervened_node_idxs = np.random.choice(d, interv_k_nodes, replace=False)

        interv_targets[i, intervened_node_idxs] = True
        interv_value = test_interv_values[i:i+1]

        interv_data_ = intervene_sem(W, 
                                    1, 
                                    sem_type,
                                    sigmas=noise_sigma, 
                                    idx_to_fix=intervened_node_idxs, 
                                    values_to_fix=interv_value, 
                                    low=low, 
                                    high=high)
        if i == 0:  interv_data = interv_data_
        else: interv_data = np.concatenate((interv_data, interv_data_), axis=0)

    return interv_data, interv_targets, test_interv_values





