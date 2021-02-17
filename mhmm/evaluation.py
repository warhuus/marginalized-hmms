# %%
import os
import pickle

import torch
import numpy as np

from mhmm.plot import get_latest

os.chdir((r'c:\\Users\\CHSWA\\OneDrive - Ørsted\\DTU\\semester_2\\' +
          r'hmm_specialkursus\\mhmm'))

def get_data():
    filenames = get_latest(2, 'output')

    items = {}
    for filename in filenames:
        with open(filename, 'rb') as file_:
            items[filename.split('.pickle')[0][-8:]] = pickle.load(file_)

    return items

#%%
items = get_data()

direct, viterbi = None, None

for date in items.keys():
    if items[date]['algo'] == 'direct':
        direct = items[date]
    if items[date]['algo'] == 'viterbi':
        viterbi = items[date]

#%%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(range(len(direct['Lr'])), direct['Lr'], label='direct')
ax.plot(range(len(viterbi['Lr'])), viterbi['Lr'], label='viterbi')
ax.legend()

#%%
plt.subplot(1, 2, 1)
plt.imshow(viterbi['M'])
plt.colorbar()
_ = plt.title('viterbi means')

plt.subplot(1, 2, 2)
plt.imshow(direct['M'])
plt.colorbar()
_ = plt.title('direct means')

#%%
with torch.no_grad():
    plt.subplot(1, 2, 1)
    plt.imshow(viterbi['log_T'].exp())
    plt.colorbar()
    _ = plt.title('viterbi trans_mat')

    plt.subplot(1, 2, 2)
    plt.imshow(direct['log_T'].exp())
    plt.colorbar()
    _ = plt.title('direct trans_mat')

# %%
with torch.no_grad():


    for i in range(5):
        
        plt.subplot(2, 5, 1 + i)
        plt.imshow(np.exp(viterbi['Cov'][i, :, :]))
        plt.colorbar()
        _ = plt.title('viterbi Cov')
        
        plt.subplot(2, 5, 1 + i + 5)
        plt.imshow(np.exp(direct['Cov'][i, :, :]))
        plt.colorbar()
        _ = plt.title('direct Cov')
#%%