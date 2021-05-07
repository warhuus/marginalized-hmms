import io
import sys
from typing import Union, List, Optional

import torch
from hmmlearn import hmm
import numpy as np

from . import utils

def run(X: torch.tensor, algo: Union['viterbi', 'map'], K: int, seed: int, lengths: Optional[int],
        D: Optional[int], N_iter: int = 1000, reps: int = 20, where: str = 'colab', **kwargs):
    """ Train an HMM model using EM with hmmlearn """

    if where == 'colab':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    torch.manual_seed(seed)
    np.random.seed(seed)

    min_neg_loglik = 1e10
    Ls = np.tile(np.nan, (reps, N_iter))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for r in tqdm(range(reps)):

       # print convergence report to temporary file
       temperr = io.StringIO()
       sys.stderr = temperr

       # setup model
       model = hmm.GaussianHMM(K, "full", n_iter=N_iter, tol=1e-20,
                               verbose=True, algorithm=algo, init_params='')
       model = utils.fill_hmmlearn_params(model, *utils.init_params(
              K, D, X=X, par={'dtype': torch.float64}, perturb=True), device)
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

    log_t0 = (torch.tensor(best_model.startprob_, dtype=torch.float64) + 1e-20).log()
    log_T = (torch.tensor(best_model.transmat_, dtype=torch.float64) + 1e-20).log().T
    M = best_model.means_.T
    Cov = best_model.covars_
    _, state_probabilities = best_model.score_samples(X, lengths=lengths)

    return {'log_likelihood': Ls,
            'log_T': log_T,
            'log_t0': log_t0,
            'M': M,
            'covariances': Cov,
            'state_probabilities': state_probabilities}

