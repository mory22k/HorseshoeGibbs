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

def sum_duplicate_rows_sqrt(X: torch.Tensor, y: torch.Tensor):
    unique_rows, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
    counts = torch.bincount(inverse_indices)

    X_scaled = torch.zeros_like(unique_rows)
    y_scaled = torch.zeros(unique_rows.size(0), dtype=y.dtype)

    for i in range(unique_rows.size(0)):
        group_mask = (inverse_indices == i)
        c_i = counts[i].float()  # 出現回数（floatに変換）

        # Scale the design matrix by sqrt(c_i)
        X_scaled[i] = unique_rows[i] * torch.sqrt(c_i)

        # Scale the target values by sqrt(c_i)
        y_scaled[i] = y[group_mask].sum(dim=0) / torch.sqrt(c_i)

    return X_scaled, y_scaled


def fast_sampling_gaussian_posterior(
    X: torch.Tensor, y: torch.Tensor, L_diag: torch.Tensor, sigma: float
):
    """
    Fast sampling of the parameters that follow a Gaussian posterior given by

    p(theta | X, y, L, sigma) = N(A^{-1} X^T y, sigma^2 A^{-1})

    where A = X^T X + diag(L)^{-1} and sigma is the estimated standard deviation of noise.

    Args:
        X: torch.Tensor, shape (n, p)
            Design matrix.
        y: torch.Tensor, shape (n,)
            Target values.
        L_diag: torch.Tensor, shape (p,)
            Vector of diagonal elements of the matrix L.
        sigma: torch.Tensor, shape ()
            Standard deviation of the noise.
    """

    n = X.shape[0]

    # L = torch.diag(L_diag)
    XL = L_diag * X

    u = torch.normal(mean=0.0, std=sigma * L_diag.sqrt())
    v = torch.normal(mean=0.0, std=sigma, size=(n,))

    # left_hand = X @ L @ X.T + torch.eye(n)
    left_hand = XL @ X.T + torch.eye(n)
    right_hand = y - X @ u - v
    try:
        linalg_solution = torch.linalg.solve(left_hand, right_hand)
    except torch._C._LinAlgError as e:
        linalg_solution = torch.linalg.lstsq(left_hand, right_hand).solution
    # params = u + L @ X.T @ linalg_solution
    params = u + XL.T @ linalg_solution
    return params


def _sample_beta(sigma2, tau2, lamb2, X, y):
    D_diag = tau2 * lamb2
    beta_new = fast_sampling_gaussian_posterior(X, y, D_diag, sigma2.sqrt())
    return beta_new


def _sample_sigma2(beta, tau2, lamb2, X, y, a_prior=0.5, n_prior=0.5):
    n, p = X.shape
    residual = y - X @ beta

    concentration = a_prior + (n + p) / 2
    rate = n_prior + torch.sum(residual**2) / 2 + torch.sum(beta**2 / lamb2) / tau2 / 2
    sigma2_new = InverseGamma(concentration, rate).sample()
    return sigma2_new


def _sample_lamb2(beta, sigma2, tau2, lamb2):
    concentration = 1.0
    rate = 1.0 + 1.0 / lamb2
    nu = InverseGamma(concentration, rate).sample()

    concentration = 1.0
    rate = 1.0 / nu + beta**2 / sigma2 / tau2 / 2
    lamb2_new = InverseGamma(concentration, rate).sample()
    return lamb2_new


def _sample_tau2(beta, sigma2, tau2, lamb2):
    p = beta.shape[0]

    concentration = 1.0
    rate = 1.0 + 1.0 / tau2
    xi = InverseGamma(concentration, rate).sample()

    concentration = (p + 1.0) / 2
    rate = 1.0 / xi + torch.sum(beta**2 / lamb2) / sigma2 / 2
    tau2_new = InverseGamma(concentration, rate).sample()
    return tau2_new


def _markov_transition(beta, sigma2, tau2, lamb2, X, y):
    tau2_new = _sample_tau2(beta, sigma2, tau2, lamb2)
    sigma2_new = _sample_sigma2(beta, tau2_new, lamb2, X, y)
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
    A naive implementation of the Gibbs sampler for the horseshoe model.
    Time complexity per iteration is O(n^2 p).

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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: betaficients, noise variance, tau2, lamb2.
    """

    n, p = X.shape

    beta = beta_init if beta_init is not None else torch.zeros(p)
    sigma2 = sigma2_init if sigma2_init is not None else 1.0
    tau2 = tau2_init if tau2_init is not None else 1.0
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
