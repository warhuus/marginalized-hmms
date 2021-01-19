import torch
import numpy as np

from .. import ops


def init_params(num_states, num_dimensions, rank_covariance, par):
    ulog_T = torch.randn((num_states, num_states), **par)
    ulog_t0 = torch.randn(num_states, **par)
    M = torch.randn((num_dimensions, num_states), **par)
    V = torch.ones((num_dimensions, num_states), **par)
    S = torch.randn((rank_covariance, num_dimensions, num_states), **par)
    return ulog_T, ulog_t0, M, V, S


def make_iter(lengths):

    assert isinstance(lengths, list)

    end = np.cumsum(lengths).astype(np.int32)
    start = end - lengths

    for i in range(len(lengths)):
        yield start[i], end[i]
