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


def main(opt):

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
                  'mom': train.mom_.run}[opt['algo']]
    except KeyError:
        raise NotImplementedError

    output = method(data_dict['X'], lengths=data.utils.make_lengths(opt), **opt)

    # plot - broken for now
    if opt['show']:
        plot.diagnostics(*output, opt['K'])
   
    # save
    now = datetime.datetime.now()

    path = os.path.join(os.getcwd(), "output", now.strftime("%m_%d"))
    if not os.path.isdir(path):
        os.mkdir(path)

    save_name = f'{now.strftime("%H_%M_%S")}_{opt["algo"]}_{opt.get("optimizer")}.pickle'

    with open(os.path.join(path, save_name), 'wb') as f:
        pickle.dump({
            **output,
            **opt,
            **data_dict}, f)
