import io
import sys
from typing import Union

import torch
from hmmlearn import hmm

from . import utils

def run(X: torch.tensor, lengths: list, K: int, D: int,
        algo: Union['viterbi', 'map'], N_iter: int = 1000, **kwargs):
    """ Train an HMM model using EM with hmmlearn """
    
    # print convergence report to temporary file
    temperr = io.StringIO()
    sys.stderr = temperr

    # setup model
    model = hmm.GaussianHMM(K, "full", n_iter=N_iter, tol=1e-20,
                            verbose=True, algorithm=algo, init_params='')
    model = utils.fill_hmmlearn_params(model, K, *utils.init_params(K, D))
    _ = model.fit(X, lengths)

    # get convergence report
    contents = temperr.getvalue()
    temperr.close()

    Lr = [ float(contents.split()[i])
           for i in range(1, len(contents.split()), 3) ]
    Lr = [ -lr / len(lengths) for lr in Lr ] 

    # reset stderr
    sys.stderr = sys.__stderr__

    log_t0 = (torch.tensor(model.startprob_) + 1e-20).log()
    log_T = (torch.tensor(model.transmat_) + 1e-20).log().T
    M = model.means_.T
    Cov = model.covars_
    _, state_probabilities = model.score_samples(X, lengths=lengths)   

    return Lr, log_T, log_t0, M, Cov, state_probabilities

