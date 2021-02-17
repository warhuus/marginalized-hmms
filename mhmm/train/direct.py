from typing import Any, Optional

import torch
import numpy as np
from hmmlearn import hmm

from .. import ops
from . import utils


def calc_emissionprob_mikkel(x, m, v, s, device):
    """
    Calculate the emission probabilities conditioned on the states
    using Mikkel's original function in PyTorch.
    """
    num_observations = x.shape[1]
    K = m.shape[1]
    e = torch.zeros(num_observations, K, device=device)
    for k in range(K):
        mm = m[:, k]
        ss = s[:, :, k]
        vv = v[:, k]
        ld = 0.5*torch.logdet(torch.matmul(ss.t(), ss) + torch.diag(vv**2))
        sx = torch.matmul(ss, x - mm[:, None])
        vx = (x - mm[:, None]) * vv[:, None]
        e[:,k] = - 0.5*torch.sum(sx**2, dim=0) - 0.5*torch.sum(vx**2, dim=0) + ld
    return e


def calc_logprob_optim(x, log_T, log_t0, M, V, S, device):
    """
    Calculate the log-likelihood of the obersvations under the
    model using Mikkel's original function in PyTorch.
    """
    E = calc_emissionprob_mikkel(x, M, V, S, device)
    log_p = log_t0 + E[0]    
    for n in range(1, E.shape[0]):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1)) + E[n]
    L = - ops.logsum(log_p)
    return L

def calc_logprob_save(x, K, log_T, log_t0, M, V, S):
    """
    Calculate the neg log-likelihood of observations under the model
    using hmmlearn.
    """

    # make hmmlearn model and set prob matrices
    model = hmm.GaussianHMM(K, 'full', init_params='')
    model.startprob_ =  log_t0.exp().detach().numpy().astype(np.float64)
    model.transmat_ = log_T.exp().detach().numpy().astype(np.float64).T

    # set means and covars
    model.means_ = M.detach().numpy().astype(np.float64).T
    S_numpy = S.detach().numpy().astype(np.float64)
    V_numpy = V.detach().numpy().astype(np.float64)
    model.covars_ = np.array([np.linalg.inv(np.matmul(S_numpy[:, :, i].T,
                                                      S_numpy[:, :, i])
                                            + np.diag(V_numpy[:, i]**2)
                                            ).astype(np.float64)
                              for i in range(K)])

    return - model.score(x.detach().numpy().astype(np.float64).T)


def run(X: torch.tensor, device: torch.device, lengths: list, N: int, D: int, K: int,
        bs: Optional[int] = None, N_iter: int = 1000, **kwargs):
    """ Train an HMM model using direct optimization """
    # init batch size
    bs = 1 if bs is None else bs
    assert bs <= N

    # init params and optimizer, send X to device
    par = {'requires_grad': True, 'dtype': torch.float32, 'device': device}
    ulog_T, ulog_t0, M, V, S = utils.init_params(K, D, par)
    optimizer = torch.optim.Adam([ulog_T, ulog_t0, M, V, S], lr=0.05)
    
    # transpose X for this part
    X = X.T.to(device)

    # Prepare log-likelihood save and loop generator
    L = np.empty(N_iter)

    for i in range(N_iter):

        for start, end in utils.make_iter(lengths):

            # get sequence
            x = X[:, start:end]

            # normalize log transition matrix
            log_T = ulog_T - ops.logsum(ulog_T)
            log_t0 = ulog_t0 - ops.logsum(ulog_t0)

            # calc and save log-likelihood using hmmlearn
            L[i] += calc_logprob_save(x, K, log_T, log_t0, M, V, S)

            # calc log-likelihood using Mikkel's functions and take a step
            Li_optim = calc_logprob_optim(x, log_T, log_t0, M, V, S, device)
            
            optimizer.zero_grad()
            Li_optim.backward()
            optimizer.step()

    # get covariance matrix
    S_numpy = S.detach().cpu().numpy()
    V_numpy = V.detach().cpu().numpy()
    Cov = np.empty((K, D, D))
    for i in range(K):
        inv_cov_matrix = (np.matmul(S_numpy[:, :, i].T, S_numpy[:, :, i])
                          + np.diag(V_numpy[:, i]**2))
        Cov[i] = np.linalg.inv(inv_cov_matrix)

    # get state probabilities - I wouldn't trust these, at least not
    # at the moment
    log_T = ulog_T - ops.logsum(ulog_T, dim=1)
    log_t0 = ulog_t0 - ops.logsum(ulog_t0)
    E = calc_emissionprob_mikkel(x=X, m=M, v=V, s=S, device=device)

    log_p = log_t0
    log_p_n = np.zeros((N, K))
    log_p_n[0] = log_p.detach().cpu().numpy() + E[0].detach().cpu().numpy()
    for n in range(1, N):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1)) + E[n]
        log_p_n[n] = log_p.detach().cpu().numpy()

    log_p = torch.zeros(K, device=device)
    log_p_b = np.zeros((N, K))
    log_p_b[-1] = log_p.detach().cpu().numpy()
    for n in reversed(range(N - 1)):
        log_p = torch.squeeze(ops.logsum(log_p + E[n + 1] + log_T.t(), dim=1))
        log_p_b[n] = log_p.detach().cpu().numpy()

    log_p_t = log_p_n + log_p_b
    state_probabilites = np.exp(log_p_t - ops.logsum_numpy(log_p_t, dim=1))

    return L, log_T, log_t0, M.detach().cpu().numpy(), Cov, state_probabilites