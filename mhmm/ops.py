import torch
import numpy as np


def logsum(x, dim=0):
    max_, _ = x.max(dim=dim, keepdim=True)
    return max_ + torch.log(torch.sum(torch.exp(x - max_), dim=dim, keepdim=True))


def logsum_numpy(x, dim=0):
    max_ = x.max(axis=dim, keepdims=True)
    return max_ + np.log(np.sum(np.exp(x - max_), axis=dim, keepdims=True))