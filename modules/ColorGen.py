import numpy as np
import networkx as nx
import pdb

class LinearGaussianColor(object):

    def __init__(self, n, obs_data, d, graph_type, degree, sem_type, noise_scale=1.0, 
        dataset_type="linear", quadratic_scale=None, data_seed=0, 
        low=-10., high=10., identity_P=False):
        
        self.n = n
        self.num_obs_data = obs_data
        self.num_interv_data = n - obs_data


        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.dataset_type = dataset_type
        self.w_range = (0.5, 2.0)
        self.quadratic_scale = quadratic_scale
        self.data_seed = data_seed
        self.low = low
        self.high = high
        self.identity_P = identity_P
        self._setup()

    def _setup(self):
        self.W, self.W_2, self.P = LinearGaussianColor.simulate_random_dag(
            self.d, self.degree, self.graph_type, self.w_range, self.data_seed, 
            (self.dataset_type != "linear"), identity_P=self.identity_P
        )

        if self.dataset_type != "linear":
            assert self.W_2 is not None
            self.W_2 = self.W_2 * self.quadratic_scale

        self.obs_X = LinearGaussianColor.simulate_sem(self.W, self.num_obs_data, self.sem_type, self.w_range, 
                                                self.noise_scale, self.dataset_type, self.W_2, 
                                                low=self.low, high=self.high)

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range, data_seed, return_w_2=False, identity_P=False):
        """Simulate random DAG with some expected degree.
        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
            return_w_2: boolean, whether to return an additional
                weight matrix used for quadratic terms
        Returns:
            W: weighted DAG
            [Optional] W: weighted DAG with same occupancy but different weights
        """
        if graph_type == "erdos-renyi":
            prob = float(degree) / (d - 1)
            np.random.seed(data_seed)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)

        elif graph_type == "barabasi-albert":
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == "full":  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError("Unknown graph type")
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        if identity_P is True: P = np.eye(d, d)
        B_perm = P.T.dot(B).dot(P)

        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        
        W = (B_perm != 0).astype(float) * U
        U_2 = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U_2[np.random.rand(d, d) < 0.5] *= -1
        W_2 = (B_perm != 0).astype(float) * U_2

        # At the moment the generative process is P.T @ lower @ P, we want
        # it to be P' @ upper @ P'.T.
        # We can return W.T, so we are saying W.T = P'.T @ lower @ P.
        # We can then return P.T, as we have
        # (P.T).T @ lower @ P.T = W.T

        if return_w_2:
            return W.T, W_2.T, P.T
        else:
            return W.T, None, P.T

    @staticmethod
    def simulate_sem(
        W, n,
        sem_type,
        w_range=None,
        noise_scale=1.0,
        dataset_type="nonlinear_1",
        W_2=None,
        sigmas=None, low=-10., high=10.0
    ) -> np.ndarray:
        """Simulate samples from SEM withsample specified type of noise.
        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM
        Returns:
            X: [n,d] sample matrix
        """

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d], dtype=np.float64)

        if sigmas is None:
            sigmas = np.ones((d,)) * noise_scale
        
        ordered_vertices = list(nx.topological_sort(G))
        assert dataset_type == "linear"
        assert sem_type == "linear-gauss"
        assert len(ordered_vertices) == d

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            eta = X[:, parents].dot(W[parents, j])
            X[:, j] = eta + np.random.normal(scale=sigmas[j], size=n)

        return np.clip(X, low, high)

    @staticmethod
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

class NonLinearCategoricalColor():
    def __init__(self):
        pass