from typing import Optional

import torch
import numpy as np
from hmmlearn import hmm
from sklearn import cluster
from scipy import linalg

from .. import ops


def get_D_from_L_dense(L_dense: torch.tensor) -> int:
    """ Get number of dimensions from L_dense """
    L_dense_np = L_dense.detach().cpu().numpy()
    return int(np.sqrt(L_dense_np.shape[1] * 8 + 1) - 1) // 2


def L_dense_to_cov(L_dense: torch.tensor) -> np.ndarray:
    """ Make covariance numpy from lower triangular tensor representation """

    K = len(L_dense)
    D = get_D_from_L_dense(L_dense)
    
    triangular = np.tril(np.ones((D, D)))

    L = np.tile(triangular, (K, 1, 1))
    L[:, triangular == 1] = L_dense.detach().cpu().numpy()
    cov = np.array([L_k @ L_k.T for L_k in L])

    return cov


def cov_to_L_dense(cov: np.ndarray, par: dict = {}) -> torch.tensor:
    """ Make lower triangular tensor representation from covariance numpy"""

    # get L_dense
    K, D, _ = cov.shape

    L = [linalg.cholesky(cov[k], lower=True) for k in range(K)]
    L_dense = torch.tensor([L[k][np.tril(np.ones((D, D))) == 1].tolist()
                            for k in range(K)], **par)

    assert np.allclose(cov, L_dense_to_cov(L_dense))

    return L_dense
    

def init_params(K: int, D: int, par: dict = {}, cluster_init: bool = True,
                X: Optional[torch.tensor] = None, perturb: bool = False):
    """ Initialize parameters similar to as done in hmmlearn """
    ulog_T = torch.randn((K, K), **par)
    ulog_t0 = torch.randn((K,), **par)

    if X is not None:
        assert X.shape[1] == D
    
    if isinstance(X, torch.Tensor):
        X_numpy = X.clone().detach().cpu().numpy()
    else:
        X_numpy = X

    # make cluster means
    if cluster_init and X is not None:
        kmeans = cluster.KMeans(n_clusters=K)
        kmeans.fit(X_numpy)
        M = torch.tensor(kmeans.cluster_centers_.T
                         + np.random.normal(size=(D, K)), **par)
    else:
        M = torch.randn((D, K), **par)

    # make covariance matrix
    cv = np.cov(X_numpy.T) + 1e-5 * np.eye(D)
    L = linalg.cholesky(cv, lower=True)
    L_dense = torch.tensor([L[np.tril(np.ones((D, D))) == 1].tolist()
                              for k in range(K)], **par)

    return ulog_T, ulog_t0, M, L_dense

def fill_hmmlearn_params(model: hmm.GaussianHMM, log_T: torch.tensor,
                         log_t0: torch.tensor, M: torch.tensor,
                         L_dense: torch.tensor, device) -> hmm.GaussianHMM:
    """
    Populate an hmmlearn model with parameters defined as in
    Mikkel's original code
    """
    # check and set shapes
    K = log_T.shape[1]
    D = M.shape[0]

    assert M.shape[1] == K
    assert log_t0.shape == (K,)
    assert L_dense.shape[0] == K

    # normalize if not yet normalized
    if not torch.allclose(log_T.exp().sum(0), torch.ones(K).to(device)):
        log_T = log_T - ops.logsum(log_T)
    if not torch.allclose(log_t0.exp().sum(), torch.ones(K).to(device)):
        log_t0 = log_t0 - ops.logsum(log_t0)

    T, t0, M = [param.detach().cpu().numpy().astype(np.float64)
                for param in [log_T.exp().T, log_t0.exp(), M.T]]

    # set probabilities
    model.startprob_ = t0
    model.transmat_ = T

    # set means
    model.means_ = M

    if np.isnan(L_dense_to_cov(L_dense)).any():
        print('woah there!')

    # set covars
    model.covars_ = L_dense_to_cov(L_dense)

    return model

def make_iter(lengths):

    assert isinstance(lengths, list)

    end = np.cumsum(lengths).astype(np.int32)
    start = end - lengths

    for i in range(len(lengths)):
        yield start[i], end[i]
