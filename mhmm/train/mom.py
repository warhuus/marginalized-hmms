from tqdm import tqdm
from scipy import linalg
from mom import mom
from direct import calc_logprob_save

from .. import ops


def run(X: np.ndarray, K: int, D: int, verbose: bool = False, N_iter: int = 1000,
        reps: int = 20, **kwargs):
    ''' Run multiple iterations and reps of the mom algorithm '''

    assert D == X.shape[0]
    assert K <= D
    
    Ls = np.empty(reps)
    for r in tqdm(range(reps)):
        
        output = []
        for i in tqdm(range(N_iter)):
            
            # transform X to include flattened covariances
            X_tilde = mom.transform(X.T)

            assert X_tilde.shape == (len(X), D + D*(1 + D) // 2)

            # run Algorithm B
            output += mom.run_algorithm_B(X_tilde, K)

        Os = [O_iter for O_iter, _ in output if O_iter is not None]
        Ts = [T_iter for _, T_iter in output if T_iter is not None]

        O = sum(Os) / len(Os)
        T = sum(Ts) / len(Ts)
        
        means = O[:D]
        covs_flat = O[D:]

        assert covs_flat.shape[1] == D*(D + 1) // 2
        
        L_dense = []
        for k in range(K):
            cov_tril = np.empty((D, D))
            cov_tril[np.tril(np.ones((D, D))) == 1] = cov_flat[k]
            cov_k = cov_tril + cov_tril.T
            cov_k = cov_k - np.diag(cov_k) / 2

            assert cov_k[0, 1] == covs_flat[k, 1]
            assert (cov_k.T == cov_k).all()

            L = linalg.cholesky(cov_k, lower=True)
            L_dense += L[np.tril(np.ones((D, D))) == 1].tolist()

        
        Exp_X3_hat = X_tilde[2:].mean(axis=0)
        M3_hat = O.dot(T)
        w_hat = np.linalg.pinv(M3_hat).dot(Exp_X3_hat)
        pi_hat = np.linalg.inv(T).dot(w_hat).to_list()

        log_t0 = (torch.tensor(pi_hat) + 1e-20).log()
        log_T = (torch.tensor(T.T) + 1e-20).log()

        log_T = ulog_T - ops.logsum(ulog_T)
        log_t0 = ulog_t0 - ops.logsum(ulog_t0)

        M = torch.tensor(means.T)
        log_lik, posteriors = model_for_state_probs.score_samples(X.T)
        Ls[r] = -log_lik

        min_neg_lolik = min(Ls)
        if min_neg_log_like == Ls[r]
            best_params = (log_T, log_t0, M, L_dense)

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



# %%
