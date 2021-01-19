import torch
import numpy as np

from . import ops


def calc_loglik(x, m, v, s, device):
    num_observations = x.shape[1]
    K = m.shape[1]
    e = torch.zeros(num_observations, K, device=device)
    for k in range(K):
        mm = m[:, k]
        ss = s[:, :, k]
        vv = v[:, k]
        ld = 0.5 * torch.logdet(torch.matmul(ss.t(), ss) + torch.diag(vv**2))
        sx = torch.matmul(ss, x-mm[:, None])
        vx = (x-mm[:, None]) * vv[:, None]
        e[:, k] = 0.5 * (-torch.sum(sx**2, dim=0)
                         - torch.sum(vx**2, dim=0)
                         + ld)
    return e


def init_params(num_states, num_dimensions, rank_covariance, par):
    ulog_T = torch.randn((num_states, num_states), **par)
    ulog_t0 = torch.randn(num_states, **par)
    M = torch.randn((num_dimensions, num_states), **par)
    V = torch.ones((num_dimensions, num_states), **par)
    S = torch.randn((rank_covariance, num_dimensions, num_states), **par)
    return ulog_T, ulog_t0, M, V, S


def step(X, log_T, log_t0, M, V, S, device, batch_size, optimizer):

    E = calc_loglik(X, M, V, S, device)

    # calculate log-likelihood
    log_p = log_t0 + E[0]
    for n in range(1, batch_size):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1)) + E[n]
    L = -ops.logsum(log_p)
    L_return = L

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    return L_return, log_T, log_t0, M, V, S

def make_iter(lengths):

    assert isinstance(lengths, list)

    end = np.cumsum(lengths).astype(np.int32)
    start = end - lengths

    for i in range(len(lengths)):
        yield start[i], end[i]


def hmm_step():
    NotImplementedError


