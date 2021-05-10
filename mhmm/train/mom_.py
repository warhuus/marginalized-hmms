from tqdm.notebook import tqdm
import numpy as np
import torch

from . import direct
from . import utils

from typing import List

import numpy as np
import scipy.linalg


def form_L(B312: List[np.ndarray], k: int, verbose: bool) -> np.ndarray:
    ''' Return L, R3 '''
    L = np.empty((k, k))

    # step 1: compute R3 that diagonalizes B312[0]
    L[0], R3 = scipy.linalg.eig(B312[0])
    R3 = R3.real
    
    # step 2: obtain the diagonals of the matrices R3^-1 @ B312[j] @ R3
    # for all but the first entry.
    if not np.isclose(np.linalg.det(R3), 0):
        try:
            R3_inv = np.linalg.inv(R3)
        except np.linalg.LinAlgError:
            if verbose:
                print(f'failed to invert R3:\n\nR3 = {R3}\n')
            return None, None
    else:
        if verbose:
            print(f'failed to invert R3:\n\nR3 = {R3}\n')
        return None, None    

    for i in range(1, k):
        L[i] = np.diag(R3_inv.dot(B312[i]).dot(R3))
    
    return L, R3


def sample_rotation_matrix(k: int) -> np.ndarray:
    ''' Make sample rotation matrix Theta '''
    theta, R = np.linalg.qr(np.random.normal(scale=5, size=(k, k)))
    theta = theta @ np.diag(np.sign(np.diag(R)))

    if np.linalg.det(theta) < 0:
        theta[:, [0, 1]] = theta[:, [1, 0]]

    return theta


def make_P32(X: np.ndarray) -> np.ndarray:
    ''' Compute P32 '''
    P32 = [np.einsum('i, j -> ij', X[i + 2], X[i + 1])
           for i in range(len(X) - 2)]
        
    return sum(P32) / len(P32)


def make_P31(X: np.ndarray) -> np.ndarray:
    ''' Compute P31 '''
    P31 = [np.einsum('i, j -> ij', X[i + 2], X[i])
           for i in range(len(X) - 2)]
    return sum(P31) / len(P31)


def make_P312(X: np.ndarray) -> np.ndarray:
    ''' Compute P_312 '''
    P312 = [np.einsum('i, j, k -> ijk', X[i + 2], X[i], X[i + 1])
            for i in range(0, len(X) - 2)]
    return sum(P312) / len(P312)


def transform(X: np.ndarray):
    """ Transform X with flattened empircal covariance """
    N, D = X.shape
    Sigma = np.array([np.einsum('i, j -> ij', X[i], X[i])[
                       np.tril(np.ones([D, D])) == 1]
                      for i in range(N)])
    X_tilde = np.hstack([X, Sigma])

    assert X_tilde.shape == (N, D + D*(D + 1) // 2)
    assert X_tilde[1, 1] == X[1, 1]
    assert (X_tilde[1, D:D + 2] == np.einsum('i, j -> ij', X[1], X[1])[:2, 0]).all()

    return X_tilde


def separate_means_and_sigma(O: np.ndarray, D: int, K: int):
    """
    Retreive means and sigma from the outputted O matrix of
    emission probabilities
    """
    assert O.shape == (D + D*(D + 1) // 2, K)

    means = O[:D]
    covs_flat = O[D:]

    assert covs_flat.shape[0] == D*(D + 1) // 2
    
    covs = []
    stencil = np.tril(np.ones((D, D))) == 1
    for k in range(K):

        # set up memory
        cov_tril = np.zeros((D, D))

        # use stencil to fil covariance
        cov_tril[stencil == 1] = covs_flat[:, k]

        # fill out upper triangle
        cov_k = cov_tril + cov_tril.T
        cov_k = cov_k - np.diag(np.diag(cov_k)) / 2

        # checks
        assert cov_k[0, 1] == covs_flat[1, k]
        assert (cov_k.T == cov_k).all()

        covs += [cov_k]

    covs_array = np.array(covs)
    assert (*reversed(means.shape), D) == covs_array.shape

    return means, covs_array


def compute_top_k_singular_values(P31: np.ndarray, P32: np.ndarray, k):
    
    U3, s, U1 = np.linalg.svd(P31)
    assert np.allclose(P31.dot(U1[0]), s[0] * U3[:, 0])

    rightsvec, s, U2 = np.linalg.svd(P32)
    assert np.allclose(P32.dot(U2[0]), s[0] * rightsvec[:, 0])

    U1 = U1.T[:, :k]
    U2 = U2.T[:, :k]
    U3 = U3[:, :k]

    old_Us = {'U1': U1, 'U2': U2, 'U3': U3}
    new_Us = old_Us.copy()

    for key, value in new_Us.items():
        for i, vector in enumerate(value.T):
            new_Us[key][:, i] = -vector if all(vector < 0) else vector

    assert all([(abs(new_Us[key]) == abs(old_Us[key])).all()
                for key in old_Us.keys()])
    
    return [new_Us[key] for key in ['U1', 'U2', 'U3']]


def run_algorithm_B(X: np.ndarray, k: int, verbose: bool = False) -> np.ndarray:
    '''
    Implementation of Algorithm B from Anandkumar, et al. 2012 for HMMs with
    multivariate Gaussian emissions. Code taken partly from maxentile (https://bit.ly/3ualJru)
    with inspiration from cmgithub's Matlab code for discrete emissions (https://bit.ly/3uakvfW).
    
    Returns None, None upon insolvent results. To see errors, use verbose = True.

    Input:
        X: Time series data, ndarray of shape (sample size, dimesions).
        k: A prior number of states, integer.
        verbose: boolean (default False), whether or not to print errors.
        
    Output:
        O: Estimated emission means, ndarray of shape (k, dimensions).
        T: Transition probability matrix, ndarray of shape (k, k). The
            probability of transitioning from the i'th to the j'th state
            is given by cell (j, i).

    '''

    # make P matrices
    P31 = make_P31(X)
    P32 = make_P32(X)
    P312 = make_P312(X)

    # compute top-k singular vectors
    U3_full, _, U1T_full = np.linalg.svd(P31)
    _, _, U2T_full = np.linalg.svd(P32)

    U1 = U1T_full.T[:, :k]
    U2 = U2T_full.T[:, :k]
    U3 = U3_full[:, :k]

    # pick invertible theta
    theta = sample_rotation_matrix(k)

    # form B312(U2 theta_j) for all j
    P312_U3_theta = [P312.dot(U2.dot(theta[j])) for j in range(k)]
    B312 = [
        (U3.T.dot(P312_U3_theta[j]).dot(U1)).dot(
        np.linalg.inv(U3.T.dot(P31).dot(U1)))
        for j in range(k)
    ]

    if verbose:
        print(f'theta:\n{theta}\n')
        print(f'U1:\n{U1}\n') 
        print(f'U2:\n{U2}\n')
        print(f'U3:\n{U3}\n')
        print(f'P312_U3_theta:\n{P312_U3_theta}\n')
        print(f'B312:\n{B312}\n')

    # form matrix L
    L, R3 = form_L(B312, k, verbose)

    if L is None:
        return None, None

    else:

        # form and return M2
        O = U2.dot(np.linalg.inv(theta).dot(L))

        if verbose:
            print(f'O:\n{O}\n')
            print(f'L:\n{L}\n')
            print(f'R:\n{R3}\n')

        # get transition matrix
        try:
            T = np.linalg.inv(U3.T.dot(O)).dot(R3)
        except np.linalg.LinAlgError:
            if verbose:
                print(f'failed to invert U3^T O:\n\nU3^T O =\n{U3.T.dot(O)}\n')
            T = None

        # set to None if any probabilities are negative
        if T is not None:
            if (T < 0).any():
                if verbose:
                    print(f'negative probability in T:\n{T}\n')
                T = None
    
        if T is not None:
            T = T / T.sum(axis=0).T
            if verbose:
                print(f'T:\n{T}\n')
                print(f'-------------\n')

        return O, T


def run(X: np.ndarray, lengths: int, K: int = 2, D: int = 2, seed: int = 0, verbose: bool = False,
        algo: str = 'mom-then-direct', N_iter: int = 1000, reps: int = 20, lrate: float = 0.001,
        **kwargs):
    ''' Run multiple iterations and reps of the mom algorithm '''

    torch.manual_seed(seed)
    np.random.seed(seed)

    assert D == X.shape[1]
    assert K <= D
        
    min_neg_loglik = 1e10
    
    Ls = np.empty((reps, N_iter))

    for r in tqdm(range(reps)):
            
        # transform X to include flattened covariances
        X_tilde = transform(X)
        assert X_tilde.shape == (len(X), D + D*(1 + D) // 2)

        # run Algorithm B
        O = None
        while O is None:
            O, _ = run_algorithm_B(X_tilde, K, verbose=verbose)
    
        M, Sigma = separate_means_and_sigma(O, D, K)

        output = direct.run(X, lengths, D=D, K=K, N_iter=N_iter, algo=algo,
                            reps=1, lrate=lrate, M=M, Sigma=Sigma)
        Log_like = output['log_likelihood']
        log_T = output['log_T']
        log_t0 = output['log_t0']
        M_out = output['M']
        Cov_out = output['covariances']
        posteriors = output['state_probabilities']

        if algo == 'mom-then-direct':
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

