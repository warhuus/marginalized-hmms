#%% import
import os
import os.path as path

import git
import nibabel as nib
from nilearn import image
import matplotlib.pyplot as plt
import numpy as np

#%% find data
repo = git.Repo('.', search_parent_directories=True)
upper_data_path = path.join(
    '..',
    'hcpdata',
    'HCP1200_Parcellation_Timeseries_Netmats_recon2',
    'HCP_PTN1200_recon2',
)
ica_data_path = path.join(
    upper_data_path,
    'groupICA_3T_HCP1200_MSMAll.tar',
    'groupICA_3T_HCP1200_MSMAll',
    'groupICA',
    'groupICA_3T_HCP1200_MSMAll_d25.ica'
)

ind_data_path = path.join(
    upper_data_path,
    'NodeTimeseries_3T_HCP1200_MSMAll_ICAd25_ts2.tar',
    'NodeTimeseries_3T_HCP1200_MSMAll_ICAd25_ts2',
    'node_timeseries',
    '3T_HCP1200_MSMAll_d25_ts2'
)

mat_data_path = path.join(
    upper_data_path,
    'netmats_3T_HCP1200_MSMAll_ICAd25_ts2.tar',
    'netmats_3T_HCP1200_MSMAll_ICAd25_ts2',
    'netmats',
    '3T_HCP1200_MSMAll_d25_ts2'
)

#%% get data

# this is the ICA group that each voxel belongs to
dlabel = image.get_data(path.join(repo.working_tree_dir, ica_data_path, 'melodic_IC_ftb.dlabel.nii'))

# this I probably don't really need, it's just the ICA values for each voxel
dscalar = image.get_data(path.join(repo.working_tree_dir, ica_data_path, 'melodic_IC.dscalar.nii'))

# this I also don't think I need
dsum = image.get_data(path.join(repo.working_tree_dir, ica_data_path, 'melodic_IC_sum.nii.gz'))

#%% inspect data
for data in [dlabel, dscalar, dsum]:
    print(type(data))
    print(data.shape)

# %% load individual data
first_ind = os.listdir(path.join(repo.working_tree_dir, ind_data_path))[0]

# this is time series data for one person, with each column representing an ICA component
a = np.loadtxt(path.join(repo.working_tree_dir, ind_data_path, first_ind))
a.shape

# %% load mat data data
a = np.loadtxt(path.join(repo.working_tree_dir, mat_data_path, 'netmats1.txt'))[1].reshape((25, 25))
a.shape

# %%
