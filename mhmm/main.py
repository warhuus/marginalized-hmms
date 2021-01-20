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


def main(opt):

    NUM_OBSERVATIONS = 500
    NUM_DIMENSIONS = 10
    NUM_STATES = 5
    RANK_COVARIANCE = 0
    STATE_LENGTH = 10
    VARIANCE = 0.1

    assert (NUM_OBSERVATIONS // (NUM_STATES * STATE_LENGTH)) % 1 == 0

    batch_size = NUM_STATES * STATE_LENGTH
    lengths = [batch_size] * (NUM_OBSERVATIONS // batch_size)

    assert isinstance(lengths, list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make dummy data
    X0 = data.create(
            NUM_OBSERVATIONS, NUM_STATES, NUM_DIMENSIONS, STATE_LENGTH, VARIANCE)
    
    if opt.plotdata:
        plot.toy_data(X0)

    # train
    if opt.algo == "direct":
        output = train.marghmm.run( 
            X0, NUM_STATES, NUM_DIMENSIONS, NUM_OBSERVATIONS, RANK_COVARIANCE,
            device, batch_size, lengths)
    elif opt.algo in ["viterbi", "map"]:
        output = train.hmm_.run(
            X0, NUM_STATES, NUM_DIMENSIONS, lengths, opt.algo)
    else:
        raise NotImplementedError("The indicated model is not implemented")

    # plot
    plot.diagnostics(*output, X0, NUM_OBSERVATIONS, NUM_STATES, device)
    if opt.show:
        plt.show()
   
    # save
    now = datetime.datetime.now()

    path = os.path.join(os.getcwd(), "output", now.strftime("%m_%d"))
    if not os.path.isdir(path):
        os.mkdir(path)
    
    Lr, log_T, log_t0, M, Cov, state_probabilities = output
    with open(os.path.join(path, now.strftime("%H_%M_%S") + ".pickle"), 'wb') as f:
        pickle.dump(
            {"Lr": Lr, "log_T": log_T, "log_t0": log_t0,
             "M": M, "Cov": Cov, "state_probabilities": state_probabilities,
             "algo": opt.algo}, f)
