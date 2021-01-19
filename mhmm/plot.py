import matplotlib.pyplot as plt
import numpy as np
import torch

from . import ops
from . import train


def drawnow():
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()


def toy_data(X0):
    plt.figure('Data')
    plt.clf()
    plt.imshow(X0.numpy())
    drawnow()


def diagnostics(X0, ulog_T, ulog_t0, log_T, log_t0, M, V, S,
                num_observations, num_states, Lr, device):
    # log likelihood
    plt.figure('Objective').clf()
    plt.plot(Lr)
    drawnow()

    # covariance matrices
    plt.figure('Parameters').clf()
    S_numpy = S.detach().cpu().numpy()
    V_numpy = V.detach().cpu().numpy()
    for i in range(num_states):
        plt.subplot(2, num_states, i+1)
        inv_cov_matrix = (np.matmul(S_numpy[:, :, i].T, S_numpy[:, :, i])
                            + np.diag(V_numpy[:, i]**2))
        covariance_matrix = np.linalg.inv(inv_cov_matrix)
        plt.imshow(covariance_matrix)
        plt.colorbar()

    # means
    plt.subplot(223)
    plt.imshow(M.detach().cpu().numpy())
    plt.colorbar()
    plt.subplot(224)

    # transition probabilities
    plt.imshow(np.exp(log_T.detach().cpu().numpy()))
    plt.clim(0, 1)
    plt.colorbar()
    drawnow()

    # state probabilities
    plt.figure('State probability')
    plt.clf()
    log_T = ulog_T - ops.logsum(ulog_T, dim=1)
    log_t0 = ulog_t0 - ops.logsum(ulog_t0)
    E = train.calc_loglik(X0.to(device), M, V, S, device)

    log_p = log_t0
    log_p_n = np.zeros((num_observations, num_states))
    log_p_n[0] = log_p.detach().cpu().numpy() + E[0].detach().cpu().numpy()
    for n in range(1, num_observations):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1))+E[n]
        log_p_n[n] = log_p.detach().cpu().numpy()

    log_p = torch.zeros(num_states, device=device)
    log_p_b = np.zeros((num_observations, num_states))
    log_p_b[-1] = log_p.detach().cpu().numpy()
    for n in reversed(range(num_observations - 1)):
        log_p = torch.squeeze(ops.logsum(log_p + E[n + 1] + log_T.t(), dim=1))
        log_p_b[n] = log_p.detach().cpu().numpy()

    log_p_t = log_p_n + log_p_b
    p = np.exp(log_p_t - ops.logsum_numpy(log_p_t, dim=1))
    plt.imshow(p.T, aspect='auto', interpolation='none')
    drawnow()
