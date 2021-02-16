import os
import pickle
from datetime import datetime

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
    plt.imshow(X0)
    drawnow()
    plt.show()


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


def get_latest(num, outputdir):

    today = datetime.now().strftime("%m_%d")

    files = os.listdir(os.path.join(outputdir, today))
    files.sort()

    outputs = [ os.path.join(outputdir, today, files[-i])
                for i in range(1, num + 1)
                if files[-i].endswith('.pickle') ]
    return outputs

def plot_latest(opt):

    # get latest files
    outfiles = get_latest(opt.num, opt.outputdir)

    # setup plotting
    fig, ax = plt.subplots()

    # get log-likelihood from latest files
    for outfile in outfiles:
        with open(outfile, 'rb') as f:
            outdict = pickle.load(f)
        _ = ax.plot(np.arange(len(outdict['Lr'])) + 1, outdict['Lr'], label=outdict['algo'])
        _ = ax.annotate('%0.2f' % outdict['Lr'][-1], xy=(len(outdict['Lr']), outdict['Lr'][-1]),
                    textcoords='data')
    ax.legend(loc="upper right")
    ax.set_xlabel("iterations")
    ax.set_ylabel("log-likelihood")
    plt.show()
