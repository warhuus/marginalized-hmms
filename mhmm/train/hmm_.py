import io
import sys
from typing import Union

import torch
from hmmlearn import hmm
import numpy as np

from . import utils

def run(X: torch.tensor, lengths: list, K: int, D: int,
        algo: Union['viterbi', 'map'], N_iter: int = 1000,
        reps: int = 20, **kwargs):
    """ Train an HMM model using EM with hmmlearn """

    min_neg_loglik = 1e10
    Ls = np.tile(np.nan, (reps, N_iter))
    
    for r in range(reps):

       # print convergence report to temporary file
       temperr = io.StringIO()
       sys.stderr = temperr

       # setup model
       model = hmm.GaussianHMM(K, "full", n_iter=N_iter, tol=1e-20,
                                   verbose=True, algorithm=algo, init_params='')
       model = utils.fill_hmmlearn_params(model, K, *utils.init_params(K, D, X=X,
                                                                       perturb=True))
       _ = model.fit(X, lengths)

       # get convergence report
       contents = temperr.getvalue()
       temperr.close()

       Lr = [ float(contents.split()[i])
              for i in range(1, len(contents.split()), 3) ]
       Lr = [ -lr / len(lengths) for lr in Lr ]

       actual_iters = len(Lr)
       Lr += [np.nan] * (N_iter - actual_iters)
       Ls[r, :] = Lr

       # reset stderr
       sys.stderr = sys.__stderr__

       # update the minimum negative log-likelihood
       min_neg_loglik = min(Lr + [min_neg_loglik])

       # save the best model thus far
       if min_neg_loglik in Lr:
           best_model = model

    log_t0 = (torch.tensor(best_model.startprob_) + 1e-20).log()
    log_T = (torch.tensor(best_model.transmat_) + 1e-20).log().T
    M = best_model.means_.T
    Cov = best_model.covars_
    _, state_probabilities = best_model.score_samples(X, lengths=lengths)

    return Ls, log_T, log_t0, M, Cov, state_probabilities

