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


def main(model):

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

    # train
    if model == "marginalized-hmm":
        output = train.marghmm.run( 
            X0, NUM_STATES, NUM_DIMENSIONS, RANK_COVARIANCE, device, batch_size, lengths)
    else:
        raise NotImplementedError("the indicated model is not yet implemented")

    # plot
    plot.diagnostics(*output, X0, NUM_OBSERVATIONS, NUM_STATES, device)
    plt.show()
   
    # save
    now = datetime.datetime.now()

    path = os.path.join(os.getcwd(), "output", now.strftime("%m_%d"))
    if not os.path.isdir(path):
        os.mkdir(path)
    
    Lr, _, _, ulog_T, ulog_t0, M, V, S = output
    with open(os.path.join(path, now.strftime("%H_%M_%S") + ".pickle"), 'wb') as f:
        pickle.dump(
            {"ulog_T": ulog_T, "ulog_t0": ulog_t0, "M": M, "V": V, "S": S, "Lr": Lr}, f)
