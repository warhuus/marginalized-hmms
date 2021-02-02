import numpy as np
import torch

from hmmlearn import hmm


def fake(num_observations, num_states, num_dimensions, state_length, variance):
    m = np.ones((num_states, num_dimensions)) * np.arange(num_states)[:, None]
    c = np.floor_divide(np.arange(num_observations), state_length) % num_states
    s = np.tile(np.eye(num_dimensions), (num_states, 1, 1)) * np.sqrt(variance)

    x = np.zeros((num_dimensions, num_observations))
    for n in range(num_observations):
        i = c[n]
        x[:, n] = np.random.multivariate_normal(m[i], s[i])

    return torch.tensor(x, dtype=torch.float32)


def dummy(num_observations, num_states, num_dimensions, lengths, variance):

    model = hmm.GaussianHMM(n_components=num_states)

    model.startprob_ = np.array([0.8] + [0.2/(num_states - 1)
                                         for i in range(num_states - 1)])
    model.transmat_ = np.tile(0.2/(num_states - 1), (num_states, num_states))

    model.means_ = np.tile(np.arange(num_states)[:, None], num_dimensions)
    model.covars_ = np.tile(np.identity(num_dimensions)*variance,
                            (num_states, 1, 1))

    all_samples = np.empty((num_dimensions, num_observations))

    for i, l in enumerate(lengths):

        sample, _ = model.sample(num_observations, random_state=0)
        all_samples[:, i * l : i * (l + 1)] = sample.T

    return all_samples
                
