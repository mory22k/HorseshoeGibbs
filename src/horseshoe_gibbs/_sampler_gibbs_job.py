# Copyright 2025 Keisuke Morita

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from ._sampler_gibbs import (
    _sample_lamb2,
    _sample_beta,
)
from ._sampler_gibbs_improved import _sample_sigma2


def logpdf_tau2_prior(tau2: torch.Tensor):
    return -torch.log(1 + torch.sqrt(tau2))


def pdf_tau2_posterior(
    tau2: torch.Tensor,
    lamb2: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    a_prior: torch.Tensor,
    b_prior: torch.Tensor,
):
    n = X.shape[0]

    XD = lamb2 * tau2 * X
    log_p1 = -torch.logdet(XD @ X.T + torch.eye(n)) / 2
    log_p2 = (-a_prior - n / 2) * torch.log(
        # b_prior + y @ (torch.eye(n) - X @ torch.linalg.solve(A, X.t())) @ y / 2
        b_prior + y @ torch.linalg.solve(XD @ X.T + torch.eye(n), y) / 2
    )
    log_p3 = logpdf_tau2_prior(tau2)

    log_p = log_p1 + log_p2 + log_p3
    return log_p


def _sample_tau2(
    tau2,
    lamb2,
    X,
    y,
    stepsize_tau,
    a_prior=torch.tensor(0.5),
    b_prior=torch.tensor(0.5),
):
    for _ in range(1):
        tau2_new = torch.distributions.LogNormal(0, stepsize_tau).sample()
        log_p_new = pdf_tau2_posterior(tau2_new, lamb2, X, y, a_prior, b_prior)
        log_p_current = pdf_tau2_posterior(tau2, lamb2, X, y, a_prior, b_prior)
        log_ratio = log_p_new - log_p_current + torch.log(tau2_new) - torch.log(tau2)

        if torch.log(torch.rand(1)) < log_ratio:
            tau2 = tau2_new
    return tau2


def _markov_transition(tau2, lamb2, X, y, stepsize_tau=0.1):
    tau2_new = _sample_tau2(tau2, lamb2, X, y, stepsize_tau)
    sigma2_new = _sample_sigma2(tau2_new, lamb2, X, y)
    beta_new = _sample_beta(sigma2_new, tau2_new, lamb2, X, y)
    lamb2_new = _sample_lamb2(beta_new, sigma2_new, tau2_new, lamb2)

    return beta_new, sigma2_new, tau2_new, lamb2_new


def sample(
    X,
    y,
    n_iter=1000,
    n_warmup=100,
    tau2_init=None,
    lamb2_init=None,
    stepsize_tau=0.1,
    debug=False,
):
    """
    Two-block Gibbs sampler for the horseshoe model.

    Args:
        X (torch.Tensor): Design matrix, shape (n, p).
        y (torch.Tensor): Target values, shape (n,).
        n_iter (int, optional): Number of iterations. Defaults to 1000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 100.
        tau2_init (torch.Tensor, optional): Initial value of the tau2. Defaults to None.
        lamb2_init (torch.Tensor, optional): Initial value of the lamb2. Defaults to None.
        stepsize_tau (float, optional): Step size for the Metropolis-Hastings update of tau2. Defaults to 0.1.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Coefficients, noise variance, tau2, lamb2.
    """
    n, p = X.shape

    tau2 = tau2_init if tau2_init is not None else torch.tensor(1.0)
    lamb2 = lamb2_init if lamb2_init is not None else torch.ones(p)

    betas = torch.zeros(n_iter, p)
    sigma2s = torch.zeros(n_iter)
    tau2s = torch.zeros(n_iter)
    lamb2s = torch.zeros(n_iter, p)

    for i in range(n_warmup):
        beta, sigma2, tau2, lamb2 = _markov_transition(tau2, lamb2, X, y, stepsize_tau)
        if debug:
            if i % 10 == 0:
                residual = y - X @ beta
                print(f"Warmup | {i:4}/{n_warmup}")

    for i in range(n_iter):
        beta, sigma2, tau2, lamb2 = _markov_transition(tau2, lamb2, X, y, stepsize_tau)
        betas[i] = beta
        sigma2s[i] = sigma2
        lamb2s[i] = lamb2
        tau2s[i] = tau2
        if debug:
            if i % 10 == 0:
                residual = y - X @ beta
                print(
                    f"mcs:{i:4d} | res: {(residual**2).mean():10.3e} | beta: {beta.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    return betas, sigma2s, tau2s, lamb2s
