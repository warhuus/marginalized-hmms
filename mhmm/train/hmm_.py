import io
import sys

import torch
from hmmlearn import hmm

def run(X0, num_states, num_dimensions, lengths):

    # print convergence report to temporary file
    temperr = io.StringIO()
    sys.stderr = temperr

    assert all(lengths[i] == lengths[0] for i in range(1, len(lengths)))

    model = hmm.GaussianHMM(n_components=num_states, covariance_type="full",
                            n_iter=len(lengths), random_state=0, tol=0.0001,
                            verbose=True)
    _ = model.fit(X0.T, lengths)

    # get convergence report
    contents = temperr.getvalue()
    temperr.close()

    Lr = [ float(contents.split()[i])
           for i in range(1, len(contents.split()), 3) ]
    Lr = [ -lr / len(lengths) for lr in Lr ] 

    # reset stderr
    sys.stderr = sys.__stderr__

    log_t0 = (torch.tensor(model.startprob_) + 1e-10).log()
    log_T = (torch.tensor(model.transmat_) + 1e-10).log()
    M = model.means_.T
    Cov = model.covars_
    _, state_probabilities = model.score_samples(X0.T, lengths=lengths)   

    return Lr, log_T, log_t0, M, Cov, state_probabilities

