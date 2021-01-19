#%%
import os
import pickle
import datetime
import types

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import ops
from . import plot
from . import data
from . import train


def main():

    NUM_OBSERVATIONS = 500
    NUM_DIMENSIONS = 10
    NUM_STATES = 5
    RANK_COVARIANCE = 0
    STATE_LENGTH = 2
    VARIANCE = 0.1
    N_PLOT = 2              # plot every n_plot iterations

    assert (NUM_OBSERVATIONS // (NUM_STATES * STATE_LENGTH)) % 1 == 0

    batch_size = NUM_STATES * STATE_LENGTH
    lengths = [batch_size] * (NUM_OBSERVATIONS // batch_size)

    assert isinstance(lengths, list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make dummy data
    X0 = data.create(
            NUM_OBSERVATIONS, NUM_STATES, NUM_DIMENSIONS, STATE_LENGTH, VARIANCE)

    data_iter = train.make_iter(lengths)

    assert isinstance(data_iter, types.GeneratorType)

# --- specific to mhmm ---

    # init params, optimizer, minibatch log-likelihood
    par = {'requires_grad': True, 'dtype': torch.float32, 'device': device}
    params = train.init_params(NUM_STATES, NUM_DIMENSIONS, RANK_COVARIANCE, par)
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

        L_, log_T, log_t0, M, V, S = train.step(X, log_T, log_t0, M, V, S,
                                                device, batch_size, optimizer)
        Lr[i] = L_.detach().cpu().numpy()

# --- specific to mhmm ---

    # plot every n_plot iterations
    if ((i + 1) % N_PLOT) == 0:
        plot.diagnostics(X0, ulog_T, ulog_t0, log_T, log_t0, M, V, S,
                            NUM_OBSERVATIONS, NUM_STATES, Lr, device)
        
    plt.show()
   
    now = datetime.datetime.now()

    path = os.path.join(os.getcwd(), "output", now.strftime("%m_%d"))
    if not os.path.isdir(path):
        os.mkdir(path)
    
    with open(os.path.join(path, now.strftime("%H_%M_%S") + ".pickle"), 'wb') as f:
        pickle.dump(
            {"ulog_T": ulog_T, "ulog_t0": ulog_t0, "M": M, "V": V, "S": S, "Lr": Lr},
            f)
