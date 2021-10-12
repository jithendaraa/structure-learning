<h2><center><a href="https://openreview.net/pdf?id=JvGzKO1QLet">Intervention-based Recurrent causal model for nonstationary video causal discovery</a></center></h2>
<br><br>

![Alt text](ircm.png?raw=true "Title")

$p(x^{1:T}; \theta)$ = $p(x^1; \theta) \prod\limits_{t=2}^{T} p(x^t | x^{1:t-1}; \theta)$

$\theta$ -> model params to learn

$\hat{x}$ -> predicted as the mode of $p(x^t | x^{1:t-1}; \theta)$ and do further prediction conditioning on this prediction.

Decompose density function into Recurrent Network and Intervention based causal model:

$f_{\theta}(x^t | x^{1:t-1}; \theta)) = f_{ICM}(x^t | M^t, I^t, x^{1:t-1}; \theta_{ICM})$

$M^t, I^t$ ~ $Bern(\alpha^t, \beta^t)$ 

$\alpha^t, \beta^t = RN(x^{1:t-1}; \theta_{RN})$

where,

$M^t$ -> DAG structure,

$I^t$ -> intervention set (set of variables we intervene upon at time t)

$\mu^t, \Sigma^t$ -> mean and covariance of the multivariate gaussian for the observation set

$\tilde{\mu^t}, \tilde{\Sigma^t}$ -> mean and covariance of the multivariate gaussian for the intervention set

This work assumes only one intervention family. That is I = {$I_1, I_2$} where the first element has no intervenion (it refers to the observational data) and the second refers to the intervened variables at each timestep t=1..T

We use NN to output params of the density function $\tilde{f}$, eg. Gaussian

$f^{(1)} = \tilde{f}(.; NN(., \phi_j^{t}))$

$f^{(2)} = \tilde{f}(.; NN(., \psi_j^{t}))$

where, $\phi, \psi$ are parameters for the observational and interventional density function.