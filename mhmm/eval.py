#%%
import pickle
import os
from typing import Optional, Union, Callable, List
from itertools import combinations, product
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

from mhmm.train.utils import make_iter
from mhmm.data.utils import make_lengths


def load_files(path: Optional[str] = None, test: bool = False):
    path = os.path.join('..', 'output', 'colab_output') if path is None else path
    files = os.listdir(path)
    loaded = {}

    for i, filename in enumerate(files):
        valid_file = (filename.endswith('.pickle')
                      and filename[-10:-7] in ['909', '559', '_22'])
        if test:
            valid_file = valid_file and ('viterbi' in filename or 'map' in filename)
        if valid_file:
            with open(os.path.join(path, filename), 'rb') as f:
                loaded[f'{i}-{filename.split(".pickle")[0]}'] = pickle.load(f)
    return loaded


def compare_state_sequences(experiment: dict):
    Z = experiment['Z'].astype(int)
    Z_hat = experiment['state_probabilities'].argmax(1)
    return [normalized_mutual_info_score(Z[s:e], Z_hat[s:e])
            for s, e in make_iter(make_lengths(experiment))]


def transform_loaded_files(func: Callable):
    def wrapper(loaded: dict):
        all_data = None
        for k, experiment in loaded.items():
            if experiment['algo'] in ['viterbi', 'map']:
                experiment['optimizer'] = experiment['algo']
            data = func(experiment)
            if all_data is None:
                all_data = data
            else:
                all_data = all_data.append(data, ignore_index=True)
        return all_data
    return wrapper


@transform_loaded_files
def make_nmi_data(experiment: dict):
    return pd.DataFrame({
        'D': experiment['D'],
        'K': experiment['K'],
        'optimizer': experiment['optimizer'],
        'seed': experiment['data_seed'],
        'algorithm': experiment['algo'],
        'NMI': compare_state_sequences(experiment)
    })


@transform_loaded_files
def make_ll_data(experiment: dict):
    return pd.DataFrame({
        'D': experiment['D'],
        'K': experiment['K'],
        'optimizer': experiment['optimizer'],
        'neg. log likelihood': experiment['log_likelihood'].ravel(),
        'seed': experiment['data_seed'],
        'algorithm': experiment['algo'],
        'training_seed': np.repeat(np.arange(experiment['reps']) + experiment['seed'],
                                    experiment['N_iter']),
        'step': np.tile(np.arange(experiment['N_iter']), experiment['reps'])
    })


def make_report_plots(data: pd.DataFrame, Ds: List[int] = [5, 20, 20, 50],
                      Ks: List[int] = [2, 3, 5, 5], seeds: List[int] = [909, 559, 23],
                      fewer_nmi_points: bool = False, save: bool = False,
                      show: bool = True, save_dir: Optional[str] = None, save_format: str = '.eps'):

    if save:
        assert save_dir is not None

    # set up plots
    fig, ax = plt.subplots(4, 4, figsize=(12, 7), gridspec_kw={'height_ratios': [4, 4, 4, 1]})
    h, l = None, None

    # set relevant plotting function and initial kwargs depending on the input data
    if 'neg. log likelihood' in data.columns:
        plot_func = sns.lineplot
        kwargs = {'x': 'step', 'y': 'neg. log likelihood', 'hue': 'algorithm',
                  'style': 'optimizer', 'units': 'training_seed', 'estimator': None,
                  'alpha': 0.3}
    elif 'NMI' in data.columns:
        plot_func = sns.boxplot
        kwargs = {'x': 'algorithm', 'y': 'NMI', 'hue': 'optimizer'}
    else:
        raise NotImplementedError

    for i in range(4):
        for j in range(4):

            # save last axis for legend
            if i == 3:
                ax[i, j].remove()
                continue
            
            # get relevant plotting data for this axis
            kwargs['data'] = data.query('D == @Ds[@j] & K == @Ks[@j] & seed == @seeds[@i]')

            # Use only average of NMI across iterations if fewer_nmi_points == True
            if ('NMI' in data.columns and fewer_nmi_points):
                kwargs['data'] = kwargs['data'].groupby(
                    ['D', 'K', 'seed', 'algorithm', 'optimizer'], as_index=False).mean()

            # get legend handles and labels by plotting on a dummy axis
            if h is None and l is None:
                _, dummy_ax = plt.subplots()
                plt.sca(dummy_ax)
                g = plot_func(**kwargs)
                h, l = g.get_legend_handles_labels()
                plt.close()
            
            plt.sca(ax[i, j])

            # deal with legends and plot data
            if 'neg. log likelihood' in data.columns:
                kwargs['legend'] = False

            plot_func(**kwargs)

            if 'NMI' in data.columns:
                ax[i, j].get_legend().remove()

            # gid rid of labels, set titles
            if j > 0:
                ax[i, j].set_ylabel(None)
            if ('NMI' in data.columns or i < 2):
                ax[i, j].set_xlabel(None)
            if i == 0:
                ax[i, j].set_title(f'D = {Ds[j]}, K = {Ks[j]}')

    # make legend
    gs = ax[3, 0].get_gridspec()    
    axleg = fig.add_subplot(gs[3:, :])
    axleg.axis(False)
    axleg.legend(h, l, ncol=len(h), loc=10)

    # add tight layout for formatting
    fig.tight_layout()

    # show and/or save
    if save:
        now = datetime.now.strftime('%H_%M_%S')
        save_name = f"{now}_{'NMI' if 'NMI' in data.columns else 'LL'}_plots{save_format}"
        plt.save(os.path.join(save_dir, save_name))
#%%

def plot_all_output(output_dir: str, **kwargs)

    assert os.path.isdir(output_dir)

    all_files = load_files(test=True)
    make_report_plots(make_nmi_data(all_files), **kwargs)
    make_report_plots(make_ll_data(all_files), **kwargs)


# %%
# def compare_states(Z: np.ndarray, Z_hat: np.ndarray):
#     max_state = max(Z)
#     states = range(max_state)
#     minimum_distance = abs(sum(Z - Z_hat))

#     permutation = {}
#     temp = max_state + 1
#     for i, j in combinations(states, 2):
#         candidate_Z_hat = Z_hat.copy()

#         # permute states
#         candidate_Z_hat[candidate_Z_hat == i] = temp
#         candidate_Z_hat[candidate_Z_hat == j] = i
#         candidate_Z_hat[candidate_Z_hat == temp] = j

#         # if candidate permutation is 1/(1 + max_state) better,
#         # then save the permutation
#         distance = abs(sum(Z - candidate_Z_hat))
#         if distance < minimum_distance:
#             permutation[(i, j)] = minimum_distance - distance

#     assert (normalized_mutual_info_score(Z, Z_hat)
#             == normalized_mutual_info_score(Z, Z_hat)


# def make_P_and_P0_from_Z(experiment: dict):

#     P_hat = experiment['log_T'].exp().T
#     P = experiment['model'].transmat_

#     P0_hat = experiment['log_t0'].exp()
#     P0 = experiment['model'].startprob_

#     Z = experiment['Z'].astype(int)

#     P_from_Z = np.zeros_like(P_hat)
#     P0_from_Z = np.zeros_like(P0_hat)
    
#     for start, end in make_iter(make_lengths(experiment)):

#         P_from_Z[Z[start]] += 1

#         for i in range(start + 1, end):
#             P_from_Z[Z[i - 1], Z[i]] += 1

#     P0_from_Z /= P0_from_Z.sum()
#     P_from_Z = (P_from_Z.T /  P_from_Z.T.sum(0)).T





# def compare_matrices(experiment: dict, which: str = 'means', **kwargs):

#     if which == 'means':
#         mat_hat = experiment['M'].T
#         mat = experiment['model'].means_
#     elif which == 'P':
#         mat_hat = experiment['log_T'].exp().T
#         mat = experiment['model'].transmat_
#         Z = experiment['Z']
#     elif which == 'P0':
#         mat_hat = experiment['log_t0'].exp()
#         mat = experiment['model'].startprob_
#         Z = experiment['Z']
#     else:
#         raise NotImplementedError("'which' must be one of 'means', 'P', 'P0'")

#     assert mat_hat.shape == mat.shape
    
#     if isinstance(mat_hat, torch.Tensor):
#         mat_hat = mat_hat.detach().cpu().numpy()

#     return np.linalg.norm(abs(mat - mat_hat), **kwargs)