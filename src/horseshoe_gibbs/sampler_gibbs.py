import torch
from torch.distributions import InverseGamma
from .fast_gaussian import fast_sampling_gaussian_posterior


def _sample_coef(X, Y, lamb2, tau2, sigma2):
    D_diag = lamb2 * tau2
    return fast_sampling_gaussian_posterior(X, Y, D_diag, sigma2.sqrt())


def _sample_sigma2(n, p, residual, coef, lamb2, tau2, alpha_prior=0.5, beta_prior=0.5):
    concentration = alpha_prior + (n + p) / 2
    rate = (
        beta_prior + torch.sum(residual**2) / 2 + torch.sum(coef**2 / lamb2) / tau2 / 2
    )
    return InverseGamma(concentration, rate).sample()


def _sample_lamb2(coef, sigma2, tau2, nu):
    concentration = 1.0
    rate = 1.0 / nu + coef**2 / tau2 / sigma2 / 2
    return InverseGamma(concentration, rate).sample()


def _sample_tau2(coef, sigma2, lamb2, xi):
    concentration = (coef.shape[0] + 1.0) / 2
    rate = 1.0 / xi + torch.sum(coef**2 / lamb2) / sigma2 / 2
    return InverseGamma(concentration, rate).sample()


def _sample_nu(lamb2):
    concentration = 1.0
    rate = 1.0 + 1.0 / lamb2
    return InverseGamma(concentration, rate).sample()


def _sample_xi(tau2):
    concentration = 1.0
    rate = 1.0 + 1.0 / tau2
    return InverseGamma(concentration, rate).sample()


def _update(X, Y, coef, sigma2, lamb2, tau2, residual):
    # Assuming that residual is defined as
    # residual = Y - X @ coef

    n, p = X.shape

    xi = _sample_xi(tau2)
    tau2 = _sample_tau2(coef, sigma2, lamb2, xi)
    nu = _sample_nu(lamb2)
    lamb2 = _sample_lamb2(coef, sigma2, tau2, nu)
    sigma2 = _sample_sigma2(n, p, residual, coef, lamb2, tau2)
    coef_new = _sample_coef(X, Y, lamb2, tau2, sigma2)

    residual = residual - X @ (coef_new - coef)

    return coef_new, sigma2, lamb2, tau2, residual


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
    A naive implementation of the Gibbs sampler for the horseshoe model.
    Time complexity per iteration is O(n^2 p).

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
    residual = Y - X @ coef

    for i in range(n_warmup):
        coef, sigma2, lamb2, tau2, residual = _update(
            X, Y, coef, sigma2, lamb2, tau2, residual
        )
        if debug:
            if i % 10 == 0:
                print(
                    f"Warmup | res: {(residual**2).mean():10.3e} | coef: {coef.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    for i in range(n_iter):
        coef, sigma2, lamb2, tau2, residual = _update(
            X, Y, coef, sigma2, lamb2, tau2, residual
        )
        coefs[i] = coef
        sigma2s[i] = sigma2
        lamb2s[i] = lamb2
        tau2s[i] = tau2
        if debug:
            if i % 10 == 0:
                print(
                    f"res: {(residual**2).mean():10.3e} | coef: {coef.abs().mean():10.3e} | sigma2: {sigma2:10.3e} lamb2: {lamb2.mean():10.3e} | tau2: {tau2:10.3e}"
                )

    return coefs, sigma2s, lamb2s, tau2s
