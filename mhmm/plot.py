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
                num_states):
    # log likelihood
    plt.figure('Objective').clf()
    plt.plot(Lr.T)
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

    plt.show()


def get_latest(num, outputdir):

    today = datetime.now().strftime("%m_%d")

    files = os.listdir(os.path.join(outputdir, today))
    files = [name for name in files if name.endswith('.pickle')]
    files.sort()

    outputs = [ os.path.join(outputdir, today, files[-i])
                for i in range(1, num + 1) ]
    return outputs

def plot_latest(opt):

    # get latest files
    outfiles = get_latest(opt.num, opt.outputdir)

    # setup plotting
    fig, ax = plt.subplots()

    plot_average = False
    # get log-likelihood from latest files
    min_y, max_y = 1e10, -1e10
    colors = ['r', 'b', 'g', 'm']
    for i, outfile in enumerate(outfiles):
        with open(outfile, 'rb') as f:
            outdict = pickle.load(f)
        if plot_average:
            _ = ax.plot(np.arange(outdict['log_likelihood'].shape[1]) + 1, outdict['log_likelihood'].mean(0),
                        label=f"{outdict['algo']}-{outdict['optimizer']}:\nminimum = {round(np.nanmin(outdict['log_likelihood']), 1)}",
                        color=colors[i])
        min_y, max_y = min(outdict['log_likelihood'].min(), min_y), max(outdict['log_likelihood'].max(), max_y)
        
        for r in range(outdict['log_likelihood'].shape[0]):
            _ = ax.plot(np.arange(outdict['log_likelihood'].shape[1]) + 1, outdict['log_likelihood'][r, :],
                        color=colors[i], alpha=0.2,
                        label=f"{outdict['algo']}-{outdict['optimizer']}:\nminimum = {round(np.nanmin(outdict['log_likelihood']), 1)}" if (r == 0 and not plot_average) else None)
    ax.set_ylim([min_y - 2, max_y + 100])
    ax.legend(loc="upper right")
    ax.set_xlabel("iterations")
    ax.set_ylabel("negative log-likelihood")
    ax.set_title((f"which-hard={outdict.get('which_hard')}, "
                  + f"lr={outdict.get('lrate')}, "
                  + f"seed={outdict.get('seed')}"))

    today = datetime.now().strftime("%m_%d")
    if not os.path.isdir(os.path.join(opt.outputdir, today, "plots")):
        os.mkdir(os.path.join(opt.outputdir, today, "plots"))

    plt.savefig(os.path.join(opt.outputdir, today, "plots",
        (f"hard={outdict.get('which_hard')}_lr={outdict.get('lrate')}_"
         + f"seed={outdict.get('seed')}_reps={outdict.get('reps')}_"
         + f"{datetime.now().strftime('%y%m%d_%H%M')}.png")
        )
    )

    plt.show()
