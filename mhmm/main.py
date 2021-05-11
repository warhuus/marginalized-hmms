#%%
import os
import pickle
import datetime
import types

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import ops
from . import plot
from . import data
from . import train


def main(opt, dir_=None):

    # create data
    if opt.get('data') == 'hcp':
        Xtrain, Xvalid, Xtest = data.hcp.load(opt)
        opt['subjects'] = 200 if opt.get('subjects') is None else opt['subjects']
        opt['N'] = 405
        opt['D'] = 50
        opt['N_seq'] = opt.get('subjects')
        data_dict = {'train_data': Xtrain, 'valid_data': Xvalid, 'test_data': Xtest}

    else:
        data_dict = data.toy.create(opt, return_type='tensor')
        assert data_dict['X'].shape == (opt['N']*opt['N_seq'], opt['D'])

    # train
    try:
        method = {'direct': train.direct.run,
                  'map': train.hmm_.run,
                  'viterbi': train.hmm_.run,
                  'mom-as-initializer': train.mom_.run,
                  'mom-then-direct': train.mom_.run,}[opt['algo']]
    except KeyError:
        raise NotImplementedError

    output = method(data_dict, lengths=data.utils.make_lengths(opt), **opt)

    # plot - broken for now
    if opt.get('show') is not None and opt.get('show'):
        plot.diagnostics(*output, opt['K'])
   
    # save
    now = datetime.datetime.now()
    save_name = (f'{now.strftime("%y_%m_%d_%H_%M_%S")}_{opt["algo"]}'
                 + f'_{opt.get("optimizer") if opt["algo"] != "viterbi" else ""}'
                 + f'_{opt.get("data_seed" if opt["data"] != "hcp" else "seed")}.pickle')

    if dir_ is None:
        dir_ = os.path.join(os.getcwd().split('hmm_specialkursus')[0],
            'hmm_specialkursus', 'mhmm', 'output', now.strftime('%m_%d')
        )
        if not os.path.isdir(dir_):
            os.mkdir(dir_)

    with open(os.path.join(dir_, save_name), 'wb') as f:
        dump_dict = {**output, **opt}
        dump_dict = {**dump_dict, **data_dict} if opt['data'] != 'hcp' else dump_dict
        pickle.dump(dump_dict, f)
