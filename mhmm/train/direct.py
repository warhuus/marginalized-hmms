from typing import Any, Optional, List, Union

import torch
import numpy as np
from hmmlearn import hmm

from .. import ops
from . import utils


def log_mv_normal(x: torch.tensor, M: torch.tensor, L_dense: torch.tensor, device):
    """
    X : D x N
    M : D x K
    L_dense : K x D(D - 1)/2
    """

    D, N = x.shape
    K = M.shape[1]
    log_prob = torch.empty((N, K), dtype=torch.float64, device=device)

    for k in range(K):
        L = torch.zeros((D, D), dtype=torch.float64, device=device)
        L[torch.tril(torch.ones((D, D), dtype=torch.float64).to(device)) == 1] = L_dense[k]

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


def calc_logprob_optim(x, log_T, log_t0, M, L_dense, device):
    """
    Calculate the negative log-likelihood of the obersvations under the
    model using Mikkel's original function in PyTorch.
    """
    E = log_mv_normal(x, M, L_dense, device)
    log_p = log_t0 + E[0]    
    for n in range(1, E.shape[0]):
        log_p = torch.squeeze(ops.logsum(log_p + log_T, dim=1)) + E[n]
    L = - ops.logsum(log_p)
    return L


def calc_logprob_save(x, ulog_T, ulog_t0, M, L_dense, device):
    """
    Calculate the neg log-likelihood of observations under the model
    using hmmlearn.
    x: (D, N)
    log_T: (K, K) - unnormalized
    log_t0: (K,) - unnormalized
    M: (D, K)
    L_dense 
    """

    log_T = ulog_T - ops.logsum(ulog_T)
    log_t0 = ulog_t0 - ops.logsum(ulog_t0)

    D, K = M.shape

    # make hmmlearn model and fill
    assert torch.allclose(log_T.exp().sum(0), torch.ones(M.shape[1], dtype=torch.float64).to(device))
    assert torch.allclose(log_t0.exp().sum(), torch.ones(M.shape[1], dtype=torch.float64).to(device))
    assert L_dense.shape[0] == K
    assert utils.get_D_from_L_dense(L_dense) == D

    model = hmm.GaussianHMM(K, 'full', init_params='')
    model = utils.fill_hmmlearn_params(model, log_T, log_t0, M, L_dense, device)

    return - model.score(x.detach().cpu().numpy().astype(np.float64).T)


def run(X: torch.tensor, lengths: list, K: int = 2, optimizer: str = 'adam', momentum: int = 0.9,
        D: int = 2, N_iter: int = 1000, reps: int = 20, lrate: float = 0.001, seed: int = 0, 
        algo: str = 'direct', M: Optional[np.ndarray] = None, Sigma: Optional[np.ndarray] = None,
        where: str = 'colab', **kwargs):
    """ Train an HMM model using direct optimization """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if where == 'colab':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # check params to train
    assert optimizer in ['adam', 'SGD', 'LBFGS']
    assert algo in ['mom-as-initializer', 'mom-then-direct', 'direct']

    # init params pars
    par = {'requires_grad': True, 'dtype': torch.float64, 'device': device}
    
    # transpose X for this part
    Xfull = X.clone()
    Xfull = Xfull.T.to(device)

    # get log-like. measurements ready
    min_neg_loglik = 1e10
    Ls = np.empty((reps, N_iter))

    assert Xfull.shape[0] == D

    Optimizer = {
        'adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'LBFGS': torch.optim.LBFGS
    }[optimizer]
    
    optim_pars = {'lr': lrate}
    if optimizer == 'SGD':
        optim_pars['momentum'] = momentum
    if optimizer == 'LBFGS':
        optim_pars['max_iter'] = 4

    for r in tqdm(range(reps)):

        # init params and optimizer
        params = utils.init_params(K, D, par, X=Xfull.T, perturb=True)

        if algo in ['mom-as-initializer', 'mom-then-direct']:
            assert M.shape == (D, K)
            assert Sigma.shape == (K, D, D)

            ulog_T, ulog_t0, _, _ = params
            M = torch.tensor(M.copy(), **par)
            L_dense = utils.cov_to_L_dense(Sigma, par)
            params_to_optim = [ulog_T, ulog_t0]

            if algo == 'mom-as-initializer':
                params_to_optim += [M, L_dense]
        
        elif algo == 'direct':
            ulog_T, ulog_t0, M, L_dense = params
            params_to_optim = [ulog_T, ulog_t0, M, L_dense]
        
        optimizer_ = Optimizer(params_to_optim, **optim_pars)

        # Prepare log-likelihood save and loop generator
        Log_like = np.zeros(N_iter)

        for i in tqdm(range(N_iter)):

            for start, end in utils.make_iter(lengths):

                # get sequence
                x = Xfull[:, start:end]
                
                # normalize log transition matrix
                log_T = ulog_T - ops.logsum(ulog_T)  # P[i, j] = prob FROM j TO i
                log_t0 = ulog_t0 - ops.logsum(ulog_t0)

                assert torch.allclose(log_T.exp().sum(0), torch.ones(K, dtype=torch.float64).to(device))
                assert torch.allclose(log_t0.exp().sum(), torch.ones(K, dtype=torch.float64).to(device))

                old_ulog_T = ulog_T.clone().detach().requires_grad_()
                old_ulog_t0 = ulog_t0.clone().detach().requires_grad_()
                old_L_dense = L_dense.clone().detach().requires_grad_()
                old_M = M.clone().detach().requires_grad_()
                
                if optimizer != 'LBFGS':
                    optimizer_.zero_grad()
                    Li_optim = calc_logprob_optim(x, log_T, log_t0, M, L_dense, device)
                    Li_optim.backward()
                    optimizer_.step()
                else:
                    def closure():
                        optimizer_.zero_grad()
                        Li_optim = calc_logprob_optim(x, log_T, log_t0, M, L_dense, device)
                        Li_optim.backward(retain_graph=True)
                        return Li_optim
                    optimizer_.step(closure=closure)

                if (torch.isnan(ulog_T).any() or torch.isnan(ulog_t0).any()
                    or torch.isnan(L_dense).any() or torch.isnan(M).any()):

                    ulog_T = old_ulog_T.clone().detach().requires_grad_()
                    ulog_t0 = old_ulog_t0.clone().detach().requires_grad_()
                    L_dense = old_L_dense.clone().detach().requires_grad_()
                    M = old_M.clone().detach().requires_grad_()

                    if algo in ['direct', 'mom-as-initializer']:
                        optimizer_ = Optimizer([ulog_T, ulog_t0, M, L_dense], **optim_pars)
                    else:
                        optimizer_ = Optimizer([ulog_T, ulog_t0], **optim_pars)

                # calc and save log-likelihood using hmmlearn
                Log_like[i] += calc_logprob_save(x, ulog_T, ulog_t0, M, L_dense, device)

                assert not np.isnan(Log_like[i])

            # save the best model thus far
            if Log_like[i] < min_neg_loglik:
                min_neg_loglik = Log_like[i]
                best_params = (ulog_T.clone().detach(),
                               ulog_t0.clone().detach(),
                               M.clone().detach(),
                               L_dense.clone().detach())
                if torch.isnan(best_params[0]).any():
                    hi = 3
        
        Ls[r, :] = Log_like / len(lengths)

    # take observations from the best model
    ulog_T, ulog_t0, M, L_dense = best_params

    # get covariance matrix
    Cov = utils.L_dense_to_cov(L_dense)

    # normalize matrices
    log_T = ulog_T - ops.logsum(ulog_T)  # P[i, j] = prob FROM j TO i
    log_t0 = ulog_t0 - ops.logsum(ulog_t0)

    # get state probabilities - I wouldn't trust these, at least not
    # at the moment
    model_for_state_probs = utils.fill_hmmlearn_params(
        hmm.GaussianHMM(K, 'full', init_params=''),
        log_T, log_t0, M, L_dense, device
    )
    _, posteriors = model_for_state_probs.score_samples(X)

    return {'log_likelihood': Ls,
            'log_T': log_T,
            'log_t0': log_t0,
            'M': M.detach().cpu().numpy(),
            'covariances': Cov,
            'state_probabilities': posteriors}