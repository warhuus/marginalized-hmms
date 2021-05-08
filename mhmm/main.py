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
        X = data.hcp.load(opt)
        opt['subjects'] = 200 if opt.get('subjects') is None else opt['subjects']
        opt['N'] = 405
        opt['D'] = 50
        opt['N_seq'] = opt.get('subjects')

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

    output = method(data_dict['X'], lengths=data.utils.make_lengths(opt), **opt)

    # plot - broken for now
    if opt['show']:
        plot.diagnostics(*output, opt['K'])
   
    # save
    now = datetime.datetime.now()
    save_name = (f'{now.strftime("%H_%M_%S")}_{opt["algo"]}_{opt.get("optimizer") if opt["algo"] != "viterbi" else ""}'
                 + f'_{opt.get("data_seed")}.pickle')

    with open(os.path.join(dir_, save_name), 'wb') as f:
        pickle.dump({
            **output,
            **opt,
            **data_dict}, f)
