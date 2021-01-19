import numpy as np
import torch


def create(num_observations, num_states, num_dimensions, state_length, variance):
    m = np.ones((num_states, num_dimensions)) * np.arange(num_states)[:, None]
    c = np.floor_divide(np.arange(num_observations), state_length) % num_states
    s = np.tile(np.eye(num_dimensions), (num_states, 1, 1)) * np.sqrt(variance)

    x = np.zeros((num_dimensions, num_observations))
    for n in range(num_observations):
        i = c[n]
        x[:, n] = np.random.multivariate_normal(m[i], s[i])

    return torch.tensor(x, dtype=torch.float32)
