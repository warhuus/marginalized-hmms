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

    try:
        files = os.listdir(os.path.join(outputdir, today))
        outputdir = os.path.join(outputdir, today)
    except FileNotFoundError:
        outputdir = 'output'
        files = os.listdir(outputdir)
    files = [name for name in files if name.endswith('.pickle')]
    files.sort()

    outputs = [ os.path.join(outputdir, files[-i])
                for i in range(1, num + 1) ]
    return outputs

def plot_latest(opt):

    # get latest files
    outfiles = get_latest(opt['num'], opt['outputdir'])

    # setup plotting
    fig, ax = plt.subplots(figsize=(5, 2))

    plot_average = False
    # get log-likelihood from latest files
    min_y, max_y = 1e10, -1e10
    colors = ['r', 'b', 'g', 'm']
    for i, outfile in enumerate(outfiles):
        with open(outfile, 'rb') as f:
            outdict = pickle.load(f)

        negll = outdict['log_likelihood']
        if plot_average:
            _ = ax.plot(np.arange(negll.shape[1]) + 1, negll.mean(0),
                        label=f"{outdict['algo']}-{outdict['optimizer']}:\nminimum = {round(np.nanmin(negll), 1)}",
                        color=colors[i])
        
        for r in range(negll.shape[0]):
            min_y, max_y = min(negll[r][~np.isnan(negll[r])].min(), min_y), max(negll[r][~np.isnan(negll[r])].max(), max_y)
            _ = ax.plot(np.arange(negll.shape[1]) + 1, negll[r, :],
                        color=colors[i], alpha=0.2,
                        label=f"{outdict['algo']}-{outdict['optimizer']}:\nminimum = {round(np.nanmin(negll), 1)}" if (r == 0 and not plot_average) else None)
    ax.set_ylim([min_y - 2, max_y + 100])
    ax.legend(loc="upper right")
    ax.set_xlabel("iterations")
    ax.set_ylabel("-ll")
    ax.set_title((f"which-hard={outdict.get('which_hard')}, "
                  + f"lr={outdict.get('lrate')}, "
                  + f"seed={outdict.get('seed')}"))
    plt.tight_layout()
    if opt['save']:
        today = datetime.now().strftime("%m_%d")
        if not os.path.isdir(os.path.join(opt['outputdir'], today, "plots")):
            os.mkdir(os.path.join(opt['outputdir'], today, "plots"))
        plt.savefig(os.path.join(opt['outputdir'], today, "plots",
            (f"hard={outdict.get('which_hard')}_lr={outdict.get('lrate')}_"
            + f"seed={outdict.get('seed')}_reps={outdict.get('reps')}_"
            + f"{datetime.now().strftime('%y%m%d_%H%M')}.png")
            )
        )

    plt.show()
