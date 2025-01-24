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
from torch.distributions import InverseGamma
from ._sampler_gibbs import (
    _sample_tau2,
    _sample_lamb2,
    _sample_beta,
)


def _sample_sigma2(tau2, lamb2, X, y, a_prior=0.5, b_prior=0.5):
    n = X.shape[0]
    XD = lamb2 * tau2 * X
    concentration = a_prior + n / 2
    try:
        XDX_In_inv_y = torch.linalg.solve(XD @ X.T + torch.eye(n), y)
    except torch.linalg.LinAlgError:
        XDX_In_inv_y = torch.linalg.lstsq(XD @ X.T + torch.eye(n), y).solution
    rate = b_prior + y @ XDX_In_inv_y / 2
    try:
        sigma2_new = InverseGamma(concentration, rate).sample()
    except ValueError:
        sigma2_new = torch.tensor(1.0)
    return sigma2_new


def _markov_transition(beta, sigma2, tau2, lamb2, X, y):
    tau2_new = _sample_tau2(beta, sigma2, tau2, lamb2)
    sigma2_new = _sample_sigma2(tau2_new, lamb2, X, y)
    beta_new = _sample_beta(sigma2_new, tau2_new, lamb2, X, y)
    lamb2_new = _sample_lamb2(beta_new, sigma2_new, tau2_new, lamb2)

    return beta_new, sigma2_new, tau2_new, lamb2_new


def sample(
    X,
    y,
    n_iter=1000,
    n_warmup=100,
    beta_init=None,
    sigma2_init=None,
    tau2_init=None,
    lamb2_init=None,
    debug=False,
):
    """
    Gibbs sampler for the horseshoe model with improved sampling of the noise variance.
    Time complexity per iteration is O(n^3p).

    Args:
        X (torch.Tensor): Design matrix, shape (n, p).
        y (torch.Tensor): Target values, shape (n,).
        n_iter (int, optional): Number of iterations. Defaults to 1000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 100.
        beta_init (torch.Tensor, optional): Initial value of the betaficients. Defaults to None.
        sigma2_init (torch.Tensor, optional): Initial value of the noise variance. Defaults to None.
        tau2_init (torch.Tensor, optional): Initial value of the tau2. Defaults to None.
        lamb2_init (torch.Tensor, optional): Initial value of the lamb2. Defaults to None.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Coefficients, noise variance, tau2, lamb2.
    """

    n, p = X.shape

    beta = beta_init if beta_init is not None else torch.zeros(p)
    sigma2 = sigma2_init if sigma2_init is not None else torch.tensor(1.0)
    tau2 = tau2_init if tau2_init is not None else torch.tensor(1.0)
    lamb2 = lamb2_init if lamb2_init is not None else torch.ones(p)

    betas = torch.zeros(n_iter, p)
    sigma2s = torch.zeros(n_iter)
    lamb2s = torch.zeros(n_iter, p)
    tau2s = torch.zeros(n_iter)

    for i in range(n_warmup):
        beta, sigma2, tau2, lamb2 = _markov_transition(beta, sigma2, tau2, lamb2, X, y)
        if debug:
            if i % 10 == 0:
                residual = y - X @ beta
                print(f"Warmup | {i:4}/{n_warmup}")

    for i in range(n_iter):
        beta, sigma2, tau2, lamb2 = _markov_transition(beta, sigma2, tau2, lamb2, X, y)
        betas[i] = beta
        sigma2s[i] = sigma2
        lamb2s[i] = lamb2
        tau2s[i] = tau2
        if debug:
            if i % 10 == 0:
                residual = y - X @ beta
                print(
                    f"res: {(residual**2).mean():10.3e} | beta: {beta.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    return betas, sigma2s, tau2s, lamb2s
