import types

import torch
import numpy as np

from .. import ops
from . import utils


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


def run(X0, num_states, num_dimensions, rank_covariance, device, batch_size,
        lengths):
    
    data_iter = utils.make_iter(lengths)

    assert isinstance(data_iter, types.GeneratorType)

    # init params, optimizer, minibatch log-likelihood
    par = {'requires_grad': True, 'dtype': torch.float32, 'device': device}
    params = utils.init_params(num_states, num_dimensions, rank_covariance, par)
    optimizer = torch.optim.Adam(params, lr=0.05)
    X0 = X0.to(device)

    Lr = np.empty(len(lengths))

    # get params
    ulog_T, ulog_t0, M, V, S = params

    for i, (start, end) in enumerate(data_iter):

        # get batch
        X = X0[:, range(start, end)]

        # normalize log transition matrix
        log_T = ulog_T - ops.logsum(ulog_T, dim=0)
        log_t0 = ulog_t0 - ops.logsum(ulog_t0)

        L_, log_T, log_t0, M, V, S = step(X, log_T, log_t0, M, V, S,
                                          device, batch_size, optimizer)
        Lr[i] = L_.detach().cpu().numpy()

    return Lr, log_T, log_t0, ulog_T, ulog_t0, M, V, S