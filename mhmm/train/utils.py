import torch
import numpy as np

from .. import ops


def init_params(K, D, par):
    ulog_T = torch.randn((K, K), **par)
    ulog_t0 = torch.randn(K, **par)
    M = torch.randn((D, K), **par)
    V = torch.ones((D, K), **par)
    S = torch.randn((0, D, K), **par)
    return ulog_T, ulog_t0, M, V, S


def make_iter(lengths):

    assert isinstance(lengths, list)

    end = np.cumsum(lengths).astype(np.int32)
    start = end - lengths

    for i in range(len(lengths)):
        yield start[i], end[i]
