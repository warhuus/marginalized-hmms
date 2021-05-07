from typing import Union

import numpy as np
import torch
from hmmlearn import hmm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix        

from .. import plot


def create(opt: dict, return_type: Union['tensor', 'numpy'],
           ) -> Union[torch.tensor, np.ndarray]:
    """ Create data for a particular experiment """

    try:
        data_dict = {
            'dummy': dummy, 'fake': fake
        }[opt['data']](**opt)
    except KeyError:
        raise NotImplementedError
    
    data_dict['X'] = {'tensor': torch.tensor(data_dict['X'], dtype=torch.float64),
                      'numpy': data_dict['X']}[return_type]

    return data_dict


def dummy(N: int, D: int, K: int, cov_structure: str = 'full',
          N_seq: int = 1, data_seed: int = 0, plot_data: bool = True,
          **kwargs) -> np.ndarray:
    
    assert D >= K, "Number of dimensions must be larger than components"

    if cov_structure != 'full':
        raise NotImplementedError

    # set random seed
    np.random.seed(data_seed)

    # initialize hmmlearn model
    model = hmm.GaussianHMM(K, 'full', init_params='')
    
    # make transition probabilities
    P = np.random.randint(1, 21, size=(K, K))
    P = (P / P.sum(0)).T
    assert np.allclose(P.sum(1), 1)
    model.transmat_ = P

    # make start probabilities
    model.startprob_ = np.repeat(1/K, K)

    # make means and covars
    model.means_ = np.random.uniform(-1.5, 1.5, size=(K, D))
    model.covars_ = np.array([make_spd_matrix(D, random_state=data_seed + k)
                              for k in range(K)])

    # make sequences
    X = np.empty((N * N_seq, D))
    Z = np.empty(N * N_seq)

    # fill sequences with randomly generated hmmlearn data
    last_sample_in_seq = np.cumsum(np.repeat(N, N_seq))
    for last in last_sample_in_seq:
        X[last - N:last, :], Z[last - N:last] = model.sample(N)

    if D == 2 and plot_data:
        densities = [multivariate_normal(model.means_[i], model.covars_[i])
                     for i in range(K)]
        x, y = np.meshgrid(np.linspace(-2.5, 9, 1000), np.linspace(-2.5, 11, 1000))
        pos = np.dstack((x, y))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(K):
            ax.contour(x, y, densities[i].pdf(pos), levels=4, colors='k')
            ax.plot(X[:, 0][Z == i], X[:, 1][Z == i], '*' + ['r', 'g', 'b'][i])
        ax.set_xlim([X[:, 0].min(), X[:, 0].max()])
        ax.set_ylim([X[:, 1].min(), X[:, 1].max()])
        plt.grid()
        plt.show()

    return {'X': X, 'Z': Z, 'model': model}


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


                
