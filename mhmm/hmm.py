import os
import pickle
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import ops
from . import plot
from . import data
from . import train


def main():

    NUM_OBSERVATIONS = 250
    NUM_DIMENSIONS = 10
    NUM_STATES = 5
    RANK_COVARIANCE = 0
    BS = 10                 # batch size
    R = 1000                # number of iterations
    N_PLOT = 50             # plot every n_plot iterations

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make dummy data
    X0 = data.create(NUM_OBSERVATIONS, NUM_STATES, NUM_DIMENSIONS)
    X0 = X0.to(device)

    # init params, optimizer, minibatch log-likelihood
    par = {'requires_grad': True, 'dtype': torch.float32, 'device': device}
    params = train.init_params(NUM_STATES, NUM_DIMENSIONS, RANK_COVARIANCE, par)
    optimizer = torch.optim.Adam(params, lr=0.05)
    Lr = np.empty(R)

    # get params
    ulog_T, ulog_t0, M, V, S = params

    for r in range(R):

        # get batch
        i0 = np.random.randint(NUM_OBSERVATIONS - BS + 1)
        X = X0[:, i0:i0+BS]

        # normalize log transition matrix
        log_T = ulog_T - ops.logsum(ulog_T, dim=0)
        log_t0 = ulog_t0 - ops.logsum(ulog_t0)

        L_, log_T, log_t0, M, V, S = train.step(X, log_T, log_t0, M, V, S,
                                                device, BS, optimizer)
        Lr[r] = L_.detach().cpu().numpy()

        # plot every n_plot iterations
        if ((r + 1) % N_PLOT) == 0:
            plot.diagnostics(X0, ulog_T, ulog_t0, log_T, log_t0, M, V, S,
                             NUM_OBSERVATIONS, NUM_STATES, Lr,   device)
   
    now = datetime.datetime.now()

    path = os.path.join(os.getcwd(), "output", now.strftime("%m_%d"))
    if not os.path.isdir(path):
        os.mkdir(path)
    
    with open(os.path.join(path, now.strftime("%H_%M_%S") + ".pickle"), 'wb') as f:
        pickle.dump(
            {"ulog_T": ulog_T, "ulog_t0": ulog_t0, "M": M, "V": V, "S": S, "Lr": Lr},
            f)
