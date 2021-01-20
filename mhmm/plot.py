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


def diagnostics(Lr, log_T, log_t0, M, Cov, state_probabilites,
                X0, num_observations, num_states, device):
    # log likelihood
    plt.figure('Objective').clf()
    plt.plot(Lr)
    drawnow()

    # covariance matrices
    plt.figure('Parameters').clf()
    for i in range(num_states):
        plt.subplot(2, num_states, i+1)
        plt.imshow(Cov[i])
        plt.colorbar()

    # means
    plt.subplot(223)
    plt.imshow(M)
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
    plt.imshow(state_probabilites.T, aspect='auto', interpolation='none')
    drawnow()
