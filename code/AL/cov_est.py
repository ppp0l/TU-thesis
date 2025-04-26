import numpy as np
import torch

from sklearn.covariance import empirical_covariance

from typing import List

def shrinkage_estimator(data, Sigma0 :np.ndarray = None):
    """
    Linear shrinkage estimator for the covariance matrix of the data.
    By default the sample covariance matrix is shrinked to the identity matrix.

    Args:
        data: (n_samples, n_dim) array of samples
        Sigma0: (n_dim, n_dim) array of prior covariance matrix
    """

    n_samples, n_dim = data.shape

    if Sigma0 is None:
        Sigma0 = np.eye(n_dim) 

    # Sample covariance matrix
    S = empirical_covariance(data, assume_centered=True) 

    data_ = data.reshape( (n_samples, n_dim, 1) )
    data_T = data.reshape( (n_samples, 1, n_dim) )

    fac1 = np.sum( (S -  Sigma0**2)**2)
    fac2 = 1/n_samples**2 * np.sum( (S.reshape((1, n_dim, n_dim)) - data_*data_T )**2)
    delta = 1/fac1 * np.min( (fac1, fac2) )

    Sigma = delta * Sigma0**2 + (1 - delta) * S

    return Sigma

def estimate_covariance(residuals : List[np.ndarray], tolerances: List[np.ndarray]) -> torch.Tensor:
    """
    Estimate the covariance matrix of the residuals.
    The covariance matrix is estimated using the linear shrinkage estimator by Ledoit and Wolf.
    The covariance matrix is shrinked to the identity matrix.

    Args:
        residuals: list with the residuals' estimates for each evaluation points.
        tolerances: list with the tolerances corresponding to each residual.

    Returns:
        covariance matrix of the residuals (torch.Tensor)
    """
    data = []
    for i in range(len(residuals)):
        res_i = residuals[i]/ tolerances[i].reshape( (-1,1))

        res_i *= np.sqrt(2/np.pi) / np.mean(np.abs(res_i), axis = 0)
        data = data + res_i.tolist()

    data = np.array(data)

    cov = shrinkage_estimator(data)

    return torch.tensor(cov, dtype=torch.float64)


