#%% import
import os
from functools import wraps
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as io


REPO = ''
DATA_PATH = os.path.join(REPO, 'hcp_data', 'for_python')


def transform_X(func):
    @wraps(func)
    def wrapper(opt: dict, *args, **kwargs):
        X = func(*args, **kwargs)
        subjects, N, D = X.shape

        if opt.get('subjects') is not None:
            X = X[np.random.default_rng(opt.get('seed')).
                  integers(0, subjects, opt.get('subjects'))]

        assert opt['N_train'] + opt['N_valid'] + opt['N_test'] == X.shape[0]
        
        Xtrain = X[:opt['N_train']].reshape((opt['N_train']*N, D))
        Xvalid = X[opt['N_train']:opt['N_valid']].reshape((opt['N_valid']*N, D))
        Xtest = X[opt['N_valid']:opt['N_test']].reshape((opt['N_test']*N, D))

        assert np.all(X[0, 0] == Xtrain[0])
        assert np.all(X[1, 0] == Xtrain[N])
        assert np.all(X[1, 5] == Xtrain[N + 5])

        return (torch.tensor(d, dtype=torch.float64)
                for d in (Xtrain, Xvalid, Xtest))
    return wrapper


def extract_X(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        data = func(*args, **kwargs)
        return data['X']
    return wrapper


@transform_X
@extract_X
def load(verbose: bool = False, only_ica_file: bool = True,
         from_morten: bool = True):
    '''
    Load HCP data from Morten. Returns dict if only_ica_file is
    True, otherwise dict of dicts.
    '''

    if not from_morten:
        raise NotImplementedError

    data = {}
    path = DATA_PATH if opt.get('directory') is None else opt.get('directory')
    for f in os.listdir(path):

        if only_ica_file and 'ica' not in f:
            continue

        if verbose:
            print(f'{f}\n__________________________\n')

        data[f] = {}
        for k, v in io.loadmat(os.path.join(path, f)).items():

            if not isinstance(v, (str, bytes)) and k != '__globals__':
                data[f] = {**data[f], k: v}
            
            if verbose:
                print(k, v if isinstance(v, (str, bytes)) else len(v)
                      if isinstance(v, list) else v.shape)

        if verbose:
            print('\n\n')
    
    return data[f] if only_ica_file else data


def compare_the_two_mat_files_from_morten():
    '''
    Compares the two mat files sent from Morten, checks that over-
    lapping data in one is the same as in the other.
    '''
    data = load.__wrapped__.__wrapped__(True, False)

    data_keys = list(data.keys())
    long_list_of_keys = list(data[data_keys[1]].keys())
    short_list_of_keys = list(data[data_keys[0]].keys())

    assert len(long_list_of_keys) > len(short_list_of_keys)

    for k2 in long_list_of_keys:

        if k2 in short_list_of_keys:
            assert np.all(data[data_keys[1]][k2] == data[data_keys[0]][k2])
    
    return True


class HCPData(Dataset):

    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample