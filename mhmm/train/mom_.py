from tqdm import tqdm
from mom import mom
import numpy as np
import torch

from . import direct
from . import utils


def run(X: np.ndarray, K: int, D: int, algo: str, lengths: int, verbose: bool = False,
        N_iter: int = 1000, pytorch_N_iter: int = 1000, reps: int = 20,
        lrate: float = 0.001, **kwargs):
    ''' Run multiple iterations and reps of the mom algorithm '''

    assert D == X.shape[1]
    assert K <= D
        
    min_neg_loglik = 1e10
    
    Ls = np.empty((reps, N_iter))
    for r in tqdm(range(reps)):
            
        # transform X to include flattened covariances
        X_tilde = mom.transform(X)
        assert X_tilde.shape == (len(X), D + D*(1 + D) // 2)

        # run Algorithm B
        O = None
        while O is None:
            O, _ = mom.run_algorithm_B(X_tilde, K, verbose=True)
    
        M, Sigma = mom.separate_means_and_sigma(O, D, K)

        output = direct.run(X, algo, K, lengths, D=D, N_iter=N_iter,
                            params_to_train='transition_probabilities',
                            reps=1, lrate=lrate, M=M, Sigma=Sigma)
        Log_like, log_T, log_t0, M_out, Cov_out, posteriors = output

        assert np.allclose(M_out, M)
        assert np.allclose(Cov_out, Sigma)
        assert Log_like.shape == (1, N_iter)

        # update the minimum negative log-likelihood
        if Log_like.min() < min_neg_loglik:
            min_neg_loglik = Log_like.min()
            best_params = log_T, log_t0, M, Cov_out
            best_posteriors = posteriors
        
        Ls[r, :] = Log_like.ravel()
    
    log_T, log_t0, M, Cov_out = best_params
    L_dense = utils.cov_to_L_dense(Cov_out)

    return {'log_likelihood': Ls,
            'log_T': log_T,
            'log_t0': log_t0,
            'M': M,
            'covariances': Cov_out,
            'state_probabilities': best_posteriors}

