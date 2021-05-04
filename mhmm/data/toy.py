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
        X = {
            'dummy-2D': dummy_2D, 'dummy-ND': dummy_ND
        }[opt['data']](**opt)
    except KeyError:
        raise NotImplementedError

    return {'tensor': torch.tensor(X, dtype=torch.float32),
            'numpy': X}[return_type]


def dummy_ND():
    pass


def dummy_2D(N: int, N_seq: int = 1, cov_multiplier: float = 0,
             data_seed: int = 0, plot_data: bool = True, **kwargs) -> np.ndarray:
    """ Make real dummy data using a `\mathbf{P}` and `\mathbf{P}_0` """

    assert cov_multiplier > 0 

    # set random seed
    np.random.seed(data_seed)

    # set dimension and number of states
    D = 2
    K = 2

    # initialize hmmlearn model
    model = hmm.GaussianHMM(K, 'full', init_params='')

    # make transition probability matrices
    model.startprob_ = np.repeat(1/2, K)
    model.transmat_ = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])

    # make mean and covariance matrices
    model.means_ = np.array([[3, 4],
                             [5, 7]])
    cov_base = np.array([
        [[2, 0.5],
         [0.5, 2]],
        [[1.5, -0.3],
         [-0.3, 4]]
    ])
    model.covars_ = cov_base * cov_multiplier

    # make sequences
    X = np.empty((N * N_seq, D))
    Z = np.empty(N * N_seq)

    # fill sequences with randomly generated hmmlearn data
    last_sample_in_seq = np.cumsum(np.repeat(N, N_seq))
    for last in last_sample_in_seq:
        X[last - N:last, :], Z[last - N:last] = model.sample(N)

    # plot if plot_data = True
    if plot_data:
        densities = [multivariate_normal(model.means_[i], model.covars_[i])
                    for i in range(K)]
        x, y = np.meshgrid(np.linspace(-2.5, 9, 1000), np.linspace(-2.5, 11, 1000))
        pos = np.dstack((x, y))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(K):
            ax.contour(x, y, densities[i].pdf(pos), levels=4, colors='k')
            ax.plot(X[:, 0][Z == i], X[:, 1][Z == i], '*' + ['r', 'g', 'b'][i])
        plt.grid()
        plt.show()

    return X


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


                
