from typing import Optional

import torch
import numpy as np
from hmmlearn import hmm
from sklearn import cluster

from .. import ops


def init_params(K: int, D: int, par: dict = {}, cluster_init: bool = True,
                X: Optional[torch.tensor] = None, perturb: bool = False):
    """ Initialize parameters similar to as done in hmmlearn """
    ulog_T = torch.randn((K, K), **par)
    ulog_t0 = torch.randn((K,), **par)

    # make cluster means
    if cluster_init:
        kmeans = cluster.KMeans(n_clusters=K)
        kmeans.fit(X)
        M = torch.tensor(kmeans.cluster_centers_.T
                         + np.random.normal(size=(D, K)), **par)
    else:
        M = torch.randn((D, K), **par)

    # make covariance matrix
    V = torch.full((D, K), 10 + np.random.normal(), **par)
    S = torch.full((D, D, K), 10 + np.random.normal(), **par)

    return ulog_T, ulog_t0, M, V, S

def fill_hmmlearn_params(model: hmm.GaussianHMM, K: int, log_T: torch.tensor,
                         log_t0: torch.tensor, M: torch.tensor,
                         V: torch.tensor, S: torch.tensor) -> hmm.GaussianHMM:
    """
    Populate an hmmlearn model with parameters defined as in
    Mikkel's original code
    """
    # normalize if not yet normalized
    if not torch.allclose(log_T.exp().sum(1), torch.tensor([1.] * K), atol=0.01):
        log_T = log_T - ops.logsum(log_T)
    if not torch.allclose(log_t0.exp().sum(), torch.tensor([1.]), atol=0.01):
        log_t0 = log_t0 - ops.logsum(log_t0)

    T, t0, M, V, S = [param.detach().numpy().astype(np.float64)
                      for param in [log_T.exp().T, log_t0.exp(), M.T, V, S]]

    # set probabilities
    model.startprob_ = t0
    model.transmat_ = T

    # set means and covars
    model.means_ = M
    model.covars_ = np.array([np.linalg.inv(np.matmul(S[:, :, i].T,
                                                      S[:, :, i])
                                            + np.diag(V[:, i]**2))
                              for i in range(K)])
    return model

def make_iter(lengths):

    assert isinstance(lengths, list)

    end = np.cumsum(lengths).astype(np.int32)
    start = end - lengths

    for i in range(len(lengths)):
        yield start[i], end[i]
