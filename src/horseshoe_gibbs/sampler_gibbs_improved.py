import torch
from torch.distributions import InverseGamma
from .sampler_gibbs import (
    _sample_xi,
    _sample_tau2,
    _sample_nu,
    _sample_lamb2,
    _sample_coef,
)


def _sample_sigma2_improved(X, Y, lamb2, tau2, alpha_prior=0.5, beta_prior=0.5):
    n, p = X.shape
    D_diag = lamb2 * tau2
    XD = D_diag * X
    concentration = alpha_prior + n / 2
    rate = beta_prior + Y @ torch.linalg.solve(XD @ X.T + torch.eye(n), Y) / 2
    return InverseGamma(concentration, rate).sample()


def _update(X, Y, coef, sigma2, lamb2, tau2):
    n, p = X.shape

    xi = _sample_xi(tau2)
    tau2 = _sample_tau2(coef, sigma2, lamb2, xi)
    nu = _sample_nu(lamb2)
    lamb2 = _sample_lamb2(coef, sigma2, tau2, nu)
    sigma2 = _sample_sigma2_improved(X, Y, lamb2, tau2)
    coef_new = _sample_coef(X, Y, lamb2, tau2, sigma2)

    return coef_new, sigma2, lamb2, tau2


def sample(
    X,
    Y,
    n_iter=1000,
    n_warmup=100,
    coef_init=None,
    sigma2_init=None,
    lamb2_init=None,
    tau2_init=None,
    debug=False,
):
    """
    Gibbs sampler for the horseshoe model with improved sampling of the noise variance.
    Time complexity per iteration is O(n^3p).

    Args:
        X (torch.Tensor): Design matrix, shape (n, p).
        Y (torch.Tensor): Target values, shape (n,).
        n_iter (int, optional): Number of iterations. Defaults to 1000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 100.
        coef_init (torch.Tensor, optional): Initial value of the coefficients. Defaults to None.
        sigma2_init (torch.Tensor, optional): Initial value of the noise variance. Defaults to None.
        lamb2_init (torch.Tensor, optional): Initial value of the lamb2. Defaults to None.
        tau2_init (torch.Tensor, optional): Initial value of the tau2. Defaults to None.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Coefficients, noise variance, lamb2, tau2.
    """

    if coef_init is None:
        coef_init = torch.randn(X.shape[1])
    if sigma2_init is None:
        sigma2_init = 1.0
    if lamb2_init is None:
        lamb2_init = torch.ones(X.shape[1])
    if tau2_init is None:
        tau2_init = 1.0

    coefs = torch.zeros(n_iter, X.shape[1])
    sigma2s = torch.zeros(n_iter)
    lamb2s = torch.zeros(n_iter, X.shape[1])
    tau2s = torch.zeros(n_iter)

    coef = coef_init
    sigma2 = sigma2_init
    lamb2 = lamb2_init
    tau2 = tau2_init

    for i in range(n_warmup):
        coef, sigma2, lamb2, tau2 = _update(
            X, Y, coef, sigma2, lamb2, tau2
        )
        if debug:
            if i % 10 == 0:
                residual = Y - X @ coef
                print(
                    f"Warmup | res: {(residual**2).mean():10.3e} | coef: {coef.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    for i in range(n_iter):
        coef, sigma2, lamb2, tau2 = _update(
            X, Y, coef, sigma2, lamb2, tau2
        )
        coefs[i] = coef
        sigma2s[i] = sigma2
        lamb2s[i] = lamb2
        tau2s[i] = tau2
        if debug:
            if i % 10 == 0:
                residual = Y - X @ coef
                print(
                    f"res: {(residual**2).mean():10.3e} | coef: {coef.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    return coefs, sigma2s, lamb2s, tau2s
