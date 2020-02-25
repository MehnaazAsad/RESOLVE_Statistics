"""
{This script combines data over more than one text file in case mcmc chain
 crashes}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import numpy as np
import math

def read_raw(path_to_file):
    colnames = ['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',\
    'scatter']
    emcee_table = pd.read_csv(path_to_file, names=colnames, 
        delim_whitespace=True, header=None)

    emcee_table = emcee_table[emcee_table.mhalo_c.values != '#']
    emcee_table.mhalo_c = emcee_table.mhalo_c.astype(np.float64)
    emcee_table.mstellar_c = emcee_table.mstellar_c.astype(np.float64)
    emcee_table.lowmass_slope = emcee_table.lowmass_slope.astype(np.float64)

    # Cases where last parameter was a NaN and its value was being written to 
    # the first element of the next line followed by 4 NaNs for the other 
    # parameters
    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            row[4] = scatter_val

    # Cases where rows of NANs appear
    emcee_table = emcee_table.dropna(axis='index', how='any').\
    reset_index(drop=True)
    return emcee_table

def read_chi2(path_to_file):
    chi2_df = pd.read_csv(path_to_file,header=None,names=['chisquared'])
    return chi2_df

# Path to pre and post breaking files
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir'] + 'bmhm_run3/'
pre_986_raw_path = path_to_proc + 'mcmc_eco_raw.txt'
post_986_raw_path = path_to_proc + 'mcmc_eco_raw_bmf986.txt'
pre_986_chi2_path = path_to_proc + 'eco_chi2.txt'
post_986_chi2_path = path_to_proc + 'eco_chi2_bmf986.txt'

# Reading in raw chain and chi squared files
pre_986_raw = read_raw(pre_986_raw_path)
post_986_raw = read_raw(post_986_raw_path)
pre_986_chi2 = read_chi2(pre_986_chi2_path)
post_986_chi2 = read_chi2(post_986_chi2_path)

# Combining pre and post breaking files
combined_raw = pre_986_raw.append(post_986_raw, ignore_index=True)
combined_chi2 = pre_986_chi2.append(post_986_chi2, ignore_index=True)

# Writing combined files to text files in the same format as those output from
# mcmc.py
combined_raw.to_csv(path_to_proc+'combined_processed_eco_raw.txt', index=False, 
    sep=' ', header=False)                                                             
combined_chi2.to_csv(path_to_proc+'combined_processed_eco_chi2.txt', 
    index=False, sep=' ', header=False)   