import torch


def fast_sampling_gaussian_posterior(
    X: torch.Tensor, Y: torch.Tensor, L_diag: torch.Tensor, sigma: torch.Tensor
):
    """
    Fast sampling of the parameters that follow a Gaussian posterior given by

    p(theta | X, Y, L, sigma) = N(A^{-1} X^T Y, sigma^2 A^{-1})

    where A = X^T X + diag(L)^{-1} and sigma is the estimated standard deviation of noise.

    Args:
        X: torch.Tensor, shape (n, p)
            Design matrix.
        Y: torch.Tensor, shape (n,)
            Target values.
        L_diag: torch.Tensor, shape (p,)
            Diagonal of the lower triangular matrix L.
        sigma: torch.Tensor, shape ()
            Standard deviation of the noise.
    """

    n = X.shape[0]

    # L = torch.diag(L_diag)
    XL = L_diag * X

    u = torch.normal(mean=0, std=sigma * L_diag.sqrt())
    v = torch.normal(mean=0, std=sigma, size=(n,))

    # left_hand = X @ L @ X.T + torch.eye(n)
    left_hand = XL @ X.T + torch.eye(n)
    right_hand = Y - X @ u - v
    linalg_solution = torch.linalg.solve(left_hand, right_hand)

    # params = u + L @ X.T @ linalg_solution
    params = u + XL.T @ linalg_solution
    return params
