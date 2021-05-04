from typing import Union

import numpy as np
import torch
from hmmlearn import hmm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from .. import plot


def create(opt: dict, return_type: Union['tensor', 'numpy'],
           ) -> Union[torch.tensor, np.ndarray]:
    """ Create data for a particular experiment """

    try:
        X = {'fake': fake, 'dummy': dummy, 'hard_dummy': hard_dummy
            }[opt['data']](**opt)
    except KeyError:
        raise NotImplementedError

    if opt['plotdata']:
        plot.toy_data(X[:opt['N'], :].T)

    return {'tensor': torch.tensor(X, dtype=torch.float32),
            'numpy': X}[return_type]

def fake(N: int, D: int, K: int, cov_rank: int, state_length: int,
         var: float, N_seq: int = 1, seed: int = 0, **kwargs) -> np.ndarray:
    """
    Make 'fake' sequential data using Mikkel's original example. Shape
    of returned data is (N, D).
    """
    assert (N // (K * state_length)) % 1 == 0

    np.random.seed(seed)

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
          seed: int = 0, **kwargs) -> np.ndarray:
    """ Make real dummy data using a `\mathbf{P}` and `\mathbf{P}_0` """

    np.random.seed(seed)

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
                
def hard_dummy(N: int, D: int, K: int, N_seq: int = 1, which_hard: int = 0,
               seed: int = 0, **kwargs) -> np.ndarray:
    """ Make real dummy data using a `\mathbf{P}` and `\mathbf{P}_0` """

    np.random.seed(seed)

    D = 2
    K = 2

    model = hmm.GaussianHMM(K, 'full', init_params='')

    # make transition probability matrices
    model.startprob_ = np.repeat(1/2, K)

    P = np.array([[0.8, 0.2],
                  [0.4, 0.6]])
    model.transmat_ = P

    # make mean and covariance matrices
    model.means_ = np.array([[3, 4],
                             [5, 7]])
    cov_base = np.array([[[2, 0.5],
                          [0.5, 2]],
                         [[1.5, -0.3],
                          [-0.3, 4]]])
    model.covars_ = [cov_base, cov_base/2, cov_base*2][which_hard]

    # make sequences
    X = np.empty((N * N_seq, D))
    Z = np.empty(N * N_seq)

    last_sample_in_seq = np.cumsum(np.repeat(N, N_seq))

    for last in last_sample_in_seq:
        X[last - N:last, :], Z[last - N:last] = model.sample(N)
    
    densities = [multivariate_normal(model.means_[i], model.covars_[i])
                 for i in range(K)]

    x, y = np.meshgrid(np.linspace(-2.5, 9, 1000), np.linspace(-2.5, 11, 1000))
    pos = np.dstack((x, y))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(K):
        ax.contour(x, y, densities[i].pdf(pos), levels=3, colors='k')
        ax.plot(X[:, 0][Z == i], X[:, 1][Z == i], '*' + ['r', 'g', 'b'][i])
    plt.grid()

    return X
                
