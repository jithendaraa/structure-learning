import numpy as onp
import pdb
import jax.numpy as jnp
from jax import random, vmap, lax
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal
from jax.scipy.special import gammaln

from dibs.utils.func import leftsel


class BGe:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise.
    Uses Normal-Wishart conjugate parameter prior to allow for closed-form marginal likelihood
    :math:`\\log p(D | G)` and thus allows inference of the marginal posterior :math:`p(G | D)`

    For details on the closed-form expression, refer to

    - Geiger et al. (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981
    - Kuipers et al. (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013

    The default arguments imply commonly-used default hyperparameters for mean and precision
    of the Normal-Wishart and assume a diagonal parameter matrix :math:`T`.
    Inspiration for the implementation was drawn from
    https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py

    This implementation uses properties of the determinant to make the computation of the marginal likelihood
    ``jax.jit``-compilable.

    Args:
        graph_dist: Graph model defining prior :math:`\\log p(G)`. Object *has to implement the method*:
            ``unnormalized_log_prob_soft``.
            For example: :class:`~dibs.graph.ErdosReniDAGDistribution`
            or :class:`~dibs.graph.ScaleFreeDAGDistribution`
        mean_obs (ndarray, optional): mean parameter of Normal
        alpha_mu (float, optional): precision parameter of Normal
        alpha_lambd (float, optional): degrees of freedom parameter of Wishart

    """

    def __init__(self, *,
                 graph_dist,
                 mean_obs=None,
                 alpha_mu=None,
                 alpha_lambd=None,
                 ):
        super(BGe, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars

        self.mean_obs = mean_obs or jnp.zeros(self.n_vars)
        self.alpha_mu = alpha_mu or 1.0
        self.alpha_lambd = alpha_lambd or (self.n_vars + 2)

        assert self.alpha_lambd > self.n_vars + 1
        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)

    def get_theta_shape(self, *, n_vars):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    """
    The following functions need to be functionally pure and jax.jit-compilable
    """

    def _slogdet_jax(self, m, parents, n_parents):
        """
        Log determinant of a submatrix. Made ``jax.jit``-compilable
        by masking everything but the submatrix and adding a diagonal of ones everywhere
        else to obtain the valid determinant

        Args:
            m (ndarray): matrix of shape ``[d, d]``
            parents (ndarray): boolean indicator of parents of shape ``[d, ]``
            n_parents (int): number of parents in ``parents``

        Returns:
            natural log of determinant of submatrix ``m`` indexed by ``parents`` on both dimensions
        """

        n_vars = parents.shape[0]
        submat = leftsel(m, parents, maskval=onp.nan)
        submat = leftsel(submat.T, parents, maskval=onp.nan).T
        submat = jnp.where(jnp.isnan(submat), jnp.eye(n_vars), submat)
        return jnp.linalg.slogdet(submat)[1]

    def _log_marginal_likelihood_single(self, j, n_parents, R, g, x, log_gamma_terms):
        """
        Computes node-specific score of BGe marginal likelihood. ``jax.jit``-compilable

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node ``j``
            R (ndarray): internal matrix for BGe score of shape ``[d, d]``
            g (ndarray): adjacency matrix of shape ``[d, d]
            x (ndarray): observations matrix of shape ``[N, d]``
            log_gamma_terms (ndarray): internal values for BGe score ``[d, ]``

        Returns:
            BGe score for node ``j``
        """

        N, d = x.shape

        isj = jnp.arange(d) == j
        parents = g[:, j] == 1
        parents_and_j = parents | isj

        # if `JAX_DEBUG_NANS` flag raises NaN error here:  ignore
        # happens due to lax.cond evaluating the second clause when n_parents == 0
        log_term_r = lax.cond(
            n_parents == 0,
            # leaf node case
            lambda _: (
                # log det(R)^(...)
                    - 0.5 * (N + self.alpha_lambd - d + 1) * jnp.log(jnp.abs(R[j, j]))
            ),
            # child case
            lambda _: (
                # log det(R_II)^(..) / det(R_JJ)^(..)
                    0.5 * (N + self.alpha_lambd - d + n_parents) *
                    self._slogdet_jax(R, parents, n_parents)
                    - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
                    self._slogdet_jax(R, parents_and_j, n_parents + 1)
            ),
            operand=None,
        )

        return log_gamma_terms[n_parents] + log_term_r

    def _eltwise_log_marginal_likelihood_single(self, *args):
        """
        Same as :func:`~dibs.models.BGe._log_marginal_likelihood_single`
        but batched over `j` and `n_parents` dimensions
        """
        return vmap(self._log_marginal_likelihood_single, (0, 0, None, None, None, None), 0)(*args)

    def log_marginal_likelihood(self, *, g, x, interv_targets):
        """Computes BGe marginal likelihood :math:`\\log p(D | G)`` in closed-form;
        ``jax.jit``-compatible

        Args:
            g (ndarray): adjacency matrix of shape ``[d, d]``
            x (ndarray): data of shape ``[N, d]``
            interv_targets (ndarray, optional): boolean mask of shape ``[N, d]`` of whether or not
                a node was intervened upon. Intervened nodes are ignored in likelihood computation

        Returns:
            BGe Score
        """
        assert g.dtype == jnp.int32, f"g has dtype `{g.dtype}` but should have `int32`. If this is a float type, " \
                                     "make sure you don't use the Gumbel-softmax, but instead the score function " \
                                     "gradient estimator with BGe "
        N, d = x.shape

        # intervention
        if interv_targets is None:
            interv_targets = jnp.zeros(d).astype(bool)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / (self.alpha_mu + 1)
        T = small_t * jnp.eye(d)

        x_bar = x.mean(axis=0, keepdims=True)
        x_center = x - x_bar
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
        # their supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        # store log gamma terms for all possible values of l
        all_l = jnp.arange(d)
        log_gamma_terms = (
                0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
                + gammaln(0.5 * (N + self.alpha_lambd - d + all_l + 1))
                - gammaln(0.5 * (self.alpha_lambd - d + all_l + 1))
                - 0.5 * N * jnp.log(jnp.pi)
                # log det(T_JJ)^(..) / det(T_II)^(..) for default T
                + 0.5 * (self.alpha_lambd - d + 2 * all_l + 1) *
                jnp.log(small_t)
        )

        # compute number of parents for each node
        n_parents_all = g.sum(axis=0)

        res = self._eltwise_log_marginal_likelihood_single(
                    jnp.arange(d), n_parents_all, R, g, x, log_gamma_terms)
        
        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                interv_targets[None, ...],
                0.0,
                res
            )
        )

    """
    Distributions used by DiBS for inference:  prior and marginal likelihood 
    """

    def log_graph_prior(self, g_prob):
        """ Computes graph prior :math:`\\log p(G)` given matrix of edge probabilities.
        This function simply binds the function of the provided ``self.graph_dist``.

        Arguments:
            g_prob (ndarray): edge probabilities in G of shape ``[n_vars, n_vars]``

        Returns:
            log prob
        """
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)

    def observational_log_marginal_prob(self, g, _, data, rng, interv_targets):
        """Computes observational marginal likelihood :math:`\\log p(D | G)`` in closed-form;
        ``jax.jit``-compatible
        [TODO] this function is just log joint prob. 

        To unify the function signatures for the marginal and joint inference classes
        :class:`~dibs.inference.MarginalDiBS` and :class:`~dibs.inference.JointDiBS`,
        this marginal likelihood is defined with dummy ``theta`` inputs as ``_``,
        i.e., like a joint likelihood

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``.
                Entries must be binary and of type ``jnp.int32``
            _:
            data (ndarray): data of shape ``[n_samples, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)
            interv_targets (ndarray): bool targets ``[n_samples, n_vars]`

        Returns:
            BGe score of shape ``[1,]``
        """
        if interv_targets is None:  interv_targets = self.no_interv_targets

        return self.log_marginal_likelihood(g=g, x=data, interv_targets=interv_targets)


class LinearGaussian:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise.

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    The noise variance at each node is equal by default, which implies the causal structure is identifiable.

    Args:
        graph_dist: Graph model defining prior :math:`\\log p(G)`. Object *has to implement the method*:
            ``unnormalized_log_prob_soft``.
            For example: :class:`~dibs.graph.ErdosReniDAGDistribution`
            or :class:`~dibs.graph.ScaleFreeDAGDistribution`
        obs_noise (float, optional): variance of additive observation noise at nodes
        mean_edge (float, optional): mean of Gaussian edge weight
        sig_edge (float, optional): std dev of Gaussian edge weight

    """

    def __init__(self, *, graph_dist, obs_noise=0.1, mean_edge=0.0, sig_edge=1.0):
        super(LinearGaussian, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars
        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)


    def get_theta_shape(self, *, n_vars):
        """Returns tree shape of the parameters of the linear model

        Args:
            n_vars (int): number of variables in model

        Returns:
            PyTree of parameter shape
        """
        return jnp.array((n_vars, n_vars))


    def sample_parameters(self, *, key, n_vars, n_particles, batch_size):
        """Samples batch of random parameters given dimensions of graph from :math:`p(\\Theta | G)`

        Args:
            key (ndarray): rng
            n_vars (int): number of variables in BN
            n_particles (int): number of parameter particles sampled
            batch_size (int): number of batches of particles being sampled

        Returns:
            Parameters ``theta`` of shape ``[batch_size, n_particles, n_vars, n_vars]``, dropping dimensions equal to 0
        """
        shape = (batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars))
        theta = self.mean_edge + self.sig_edge * random.normal(key, shape=tuple(d for d in shape if d != 0))
        return theta

    
    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        """Samples ``n_samples`` observations given graph ``g`` and parameters ``theta``

        Args:
            key (ndarray): rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta (Any): parameters
            interv (dict): intervention specification of the form ``{intervened node : clamp value}``

        Returns:
            observation matrix of shape ``[n_samples, n_vars]``
        """
        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:
            # intervention
            if j in interv.keys():
                x = index_update(x, index[:, j], interv[j])
                continue
            
            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
                x = index_update(x, index[:, j], mean + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x
    
    """
    The following functions need to be functionally pure and @jit-able
    """
    def log_prob_parameters(self, *, theta, g):
        """Computes parameter prior :math:`\\log p(\\Theta | G)``
        In this model, the parameter prior is Gaussian.

        Arguments:
            theta (ndarray): parameter matrix of shape ``[n_vars, n_vars]``
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            
        Returns:
            log prob
        """
        return jnp.sum(g * jax_normal.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge))


    def log_likelihood(self, *, x, theta, g, interv_targets):
        """Computes likelihood :math:`p(D | G, \\Theta)`.
        In this model, the noise per observation and node is additive and Gaussian.

        Arguments:
            x (ndarray): observations of shape ``[n_samples, n_vars]``
            theta (ndarray): parameters of shape ``[n_vars, n_vars]``
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            interv_targets (ndarray): binary intervention indicator vector of shape ``[n_samples, n_vars]``
            [TODO] check this

        Returns:
            log prob
        """

        log_normal = jax_normal.logpdf(x=x, loc=x @ (g * theta), scale=jnp.sqrt(self.obs_noise))

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [n_samples, n_vars]
                interv_targets,
                0.0,
                # [n_samples, n_vars]
                log_normal
            )
        )


    """
    Distributions used by DiBS for inference:  prior and joint likelihood 
    """

    def log_graph_prior(self, g_prob):
        """ Computes graph prior :math:`\\log p(G)` given matrix of edge probabilities.
        This function simply binds the function of the provided ``self.graph_dist``.

        Arguments:
            g_prob (ndarray): edge probabilities in G of shape ``[n_vars, n_vars]``

        Returns:
            log prob
        """
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)


    def observational_log_joint_prob(self, g, theta, data, rng, interv_targets):
        """Computes observational joint likelihood :math:`\\log p(\\Theta, D | G)``
        [TODO] this function is just log joint prob. 
        It is not "observational" anymore since we have added interv_targets support
        Should probably change the name

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            theta (ndarray): parameter matrix of shape ``[n_vars, n_vars]``
            data (ndarray): data of shape ``[n_samples, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)
            interv_targets (bool ndarray): shape ``[n_samples, n_vars]``

        Returns:
            log prob of shape ``[1,]``
        """
        if interv_targets is None:  interv_targets = self.no_interv_targets

        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood(x=data, theta=theta, g=g,  interv_targets=interv_targets)
        return log_prob_theta + log_likelihood

