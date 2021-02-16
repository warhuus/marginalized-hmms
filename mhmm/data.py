from typing import Union

import numpy as np
import torch
from hmmlearn import hmm

from . import plot


def create(opt: dict, return_type: Union['tensor', 'numpy'],
           ) -> Union[torch.tensor, np.ndarray]:
    """ Create data for the particular experiment """
    X = {'fake': fake(**opt),
         'dummy': dummy(**opt)}[opt['data']]

    if opt['plotdata']:
        plot.toy_data(X[:opt['N'], :].T)

    return {'tensor': torch.tensor(X, dtype=torch.float32),
            'numpy': X}[return_type]

def fake(N: int, D: int, K: int, cov_rank: int, state_length: int,
         var: float, N_seq: int = 1, **kwargs) -> np.ndarray:
    """
    Make 'fake' sequential data using Mikkel's original example. Shape
    of returned data is (N, D).
    """
    assert (N // (K * state_length)) % 1 == 0

    # define params
    M = np.ones((K, D)) * np.arange(K)[:, None]
    C = np.floor_divide(np.arange(N), state_length) % K
    S = np.tile(np.eye(D), (K, 1, 1)) * np.sqrt(var)

    # make sequences
    X = np.empty((N * N_seq, D))
    last_sample_in_seq = np.cumsum(np.repeat(N, N_seq))

    for last in last_sample_in_seq:

        x = np.empty((N, D))
        for n in range(N):
            i = C[n]
            x[n, :] = np.random.multivariate_normal(M[i], S[i])
        
        X[last - N:last, :] = x

    return X

def dummy(N: int, D: int, K: int, var: float, N_seq: int = 1,
          **kwargs) -> np.ndarray:
    """ Make real dummy data using a `\mathbf{P}` and `\mathbf{P}_0` """

    model = hmm.GaussianHMM(K, 'full', init_params='')

    # make transition probability matrices
    model.startprob_ = np.array([0.8] + [0.2/(K - 1)
                                         for i in range(K - 1)]
                                )

    P = np.tile(0.2/(K - 1), (K, K))
    np.fill_diagonal(P, 0.8)
    model.transmat_ = P

    # make mean and covariance matrices
    model.means_ = np.tile(np.arange(K)[:, None], D)
    model.covars_ = np.tile(np.identity(D) * var,
                            (K, 1, 1))

    # make sequences
    X = np.empty((N * N_seq, D))
    Z = np.empty(N * N_seq)

    last_sample_in_seq = np.cumsum(np.repeat(N, N_seq))

    for last in last_sample_in_seq:
        X[last - N:last, :], Z[last - N:last] = model.sample(N)

    return X
                
