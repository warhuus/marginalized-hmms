from typing import Any, Optional, List

import torch
import numpy as np
from hmmlearn import hmm
from tqdm import tqdm

from .. import ops
from . import utils


def log_mv_normal(x: torch.tensor, M: torch.tensor, L_dense: torch.tensor):
    """
    X : D x N
    M : D x K
    L_dense : K x D(D - 1)/2
    """

    D, N = x.shape
    K = M.shape[1]
    log_prob = torch.empty((N, K))

    for k in range(K):
        
        L = torch.zeros((D, D))
        L[torch.tril(torch.ones((D, D))) == 1] = L_dense[k]

        cv_log_det = 2 * torch.log(torch.diagonal(L)).sum()
        cv_sol = torch.triangular_solve((x.t() - M.t()[k]).t(), L,
                                        upper=False).solution.t()
        log_prob[:, k] = - .5 * (torch.sum(cv_sol ** 2, dim=1) + 
                                 D * np.log(2 * np.pi) + cv_log_det)
    
    return log_prob
        

def calc_emissionprob_mikkel(x, m, v, s, device):
    """
    Calculate the emission probabilities conditioned on the states
    using Mikkel's original function in PyTorch.
    """
    num_observations = x.shape[1]
    K = m.shape[1]
    e = torch.zeros(num_observations, K, device=device)
    for k in range(K):
        mm = m[:, k]
        ss = s[:, :, k]
        vv = v[:, k]
        ld = 0.5*torch.logdet(torch.matmul(ss.t(), ss) + torch.diag(vv**2))
        sx = torch.matmul(ss, x - mm[:, None])
        vx = (x - mm[:, None]) * vv[:, None]
        e[:,k] = - 0.5*torch.sum(sx**2, dim=0) - 0.5*torch.sum(vx**2, dim=0) + ld
    return e


def calc_logprob_optim(x, log_T, log_t0, M, L_dense):
    """
    Calculate the log-likelihood of the obersvations under the
    model using Mikkel's original function in PyTorch.
    """
    E = log_mv_normal(x, M, L_dense)
    log_p = log_t0 + E[0]    
    for n in range(1, E.shape[0]):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1)) + E[n]
    L = - ops.logsum(log_p)
    return L


def calc_logprob_save(x, log_T, log_t0, M, L_dense):
    """
    Calculate the neg log-likelihood of observations under the model
    using hmmlearn.
    """

    # make hmmlearn model and fill
    K = M.shape[1]
    model = hmm.GaussianHMM(K, 'full', init_params='')
    model = utils.fill_hmmlearn_params(model, log_T, log_t0, M, L_dense)

    return - model.score(x.detach().cpu().numpy().astype(np.float64).T)


def run(X: torch.tensor, algo: str, K: int, lengths: Optional[List] = None,
        N: Optional[int] = None, D: Optional[int] = None, bs: Optional[int] = None,
        N_iter: int = 1000, reps: int = 20, **kwargs):
    """ Train an HMM model using direct optimization """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init batch size
    bs = 1 if bs is None else bs
    assert bs <= N

    # init params pars
    par = {'requires_grad': True, 'dtype': torch.float32, 'device': device}
    
    # transpose X for this part
    X = X.T.to(device)

    # get log-like. measurements ready
    min_neg_loglik = 1e10
    Ls = np.empty((reps, N_iter))

    assert X.shape == (D, N)

    for r in tqdm(range(reps)):

        # init params and optimizer
        params = utils.init_params(K, D, par, X=X.T, perturb=True)

        ulog_T, ulog_t0, M, L_dense = params
        optimizer = torch.optim.Adam([ulog_T, ulog_t0, M, L_dense], lr=0.05)
        
        # Prepare log-likelihood save and loop generator
        Log_like = np.zeros(N_iter)

        for i in tqdm(range(N_iter)):

            for start, end in utils.make_iter(lengths):

                # get sequence
                x = X[:, start:end]

                # normalize log transition matrix
                log_T = ulog_T - ops.logsum(ulog_T)
                log_t0 = ulog_t0 - ops.logsum(ulog_t0)

                # calc and save log-likelihood using hmmlearn
                Log_like[i] += calc_logprob_save(x, log_T, log_t0, M, L_dense)

                # calc log-likelihood using hmmlearn function and take a step
                Li_optim = calc_logprob_optim(x, log_T, log_t0, M, L_dense)
                
                optimizer.zero_grad()
                Li_optim.backward()
                optimizer.step()
        
        # update the minimum negative log-likelihood
        min_neg_loglik = min(Log_like.tolist() + [min_neg_loglik])

        # save the best model thus far
        if min_neg_loglik in Log_like:
           best_params = log_T, log_t0, M, L_dense
        
        Ls[r, :] = Log_like

    # take observations from the best model
    log_T, log_t0, M, L_dense = best_params

    # get covariance matrix
    Cov = utils.L_dense_to_cov(L_dense)

    # get state probabilities - I wouldn't trust these, at least not
    # at the moment
    model_for_state_probs = utils.fill_hmmlearn_params(
        hmm.GaussianHMM(K, 'full', algorithm=algo, init_params=''),
        log_T, log_t0, M, L_dense
    )
    _, posteriors = model_for_state_probs.score_samples(X.T)

    return Ls, log_T, log_t0, M.detach().cpu().numpy(), Cov, posteriors