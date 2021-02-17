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

    N = opt['N']
    D = opt['D']
    K = opt['K']
    cov_rank = opt['K']
    state_length = opt['state_length']
    var = opt['var']

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create data
    X0 = data.create(opt, return_type='tensor')

    # train
    try:
        N_seq = opt['N_seq']
    except KeyError:
        N_seq = 1
    lengths = np.repeat(N, N_seq).tolist()

    if opt["algo"] == "direct":
        output = train.direct.run(X0, device, lengths, **opt)
    elif opt["algo"] in ["viterbi", "map"]:
        output = train.hmm_.run(
            X0, K, D, lengths, opt["algo"])
    else:
        raise NotImplementedError("The indicated model is not implemented")

    # plot
    plot.diagnostics(*output, X0.T, N, K, device)
    if opt["show"]:
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
             "data": opt["data"], "algo": opt["algo"]}, f)
