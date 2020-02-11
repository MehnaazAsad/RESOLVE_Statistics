"""
{This script reads in the raw chain and plots times series for all parameters
 in order to identify the burn-in}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import numpy as np
import math

__author__ = '{Mehnaaz Asad}'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=20)
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

survey = 'eco'
mf_type = 'smf'
file_ver = 2.0 # Writing to files as chain runs

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_run5_errmock/'
else:
    path_to_proc = path_to_proc + 'bmhm_run2/'

if mf_type == 'smf' and survey == 'eco' and file_ver == 1.0:   

    chain_fname = path_to_proc + 'mcmc_eco_raw.txt'

    emcee_table = pd.read_csv(chain_fname,delim_whitespace=True,\
    names=['halo','stellar','lowmass','highmass','scatter'],header=None)

    sampler = []
    chunk = []
    for idx,row in emcee_table.iterrows():
        
        if row[0] == '#':
            chunk = np.array(chunk)
            sampler.append(chunk)
            chunk = []
        else:
            row = row.astype(np.float64)
            chunk.append(row)
    sampler = np.array(sampler)

    walker_id_arr = np.zeros(len(emcee_table))
    iteration_id_arr = np.zeros(len(emcee_table))
    counter_wid = 1
    counter_itid = 0
    for idx,row in emcee_table.iterrows():
        counter_itid += 1
        if row[0] == '#':
            counter_wid +=1
            walker_id_arr[idx] = 0
            counter_itid = 0
        else:
            walker_id_arr[idx] = counter_wid
            iteration_id_arr[idx] = counter_itid

else:

    chain_fname = path_to_proc + 'mcmc_{0}_raw.txt'.format(survey)

    emcee_table = pd.read_csv(chain_fname, delim_whitespace=True, 
        names=['halo','stellar','lowmass','highmass','scatter'], 
        header=None)

    emcee_table = emcee_table[emcee_table.halo.values != '#']
    emcee_table.halo = emcee_table.halo.astype(np.float64)
    emcee_table.stellar = emcee_table.stellar.astype(np.float64)
    emcee_table.lowmass = emcee_table.lowmass.astype(np.float64)

    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            row[4] = scatter_val 
    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)

    # Each chunk is now a step and within each chunk, each row is a walker
    # Different from what it used to be where each chunk was a walker and 
    # within each chunk, each row was a step

    walker_id_arr = np.zeros(len(emcee_table))
    iteration_id_arr = np.zeros(len(emcee_table))
    counter_wid = 0
    counter_stepid = 0
    for idx,row in emcee_table.iterrows():
        counter_wid += 1
        if idx % 250 == 0:
            counter_stepid += 1
            counter_wid = 1
        walker_id_arr[idx] = counter_wid
        iteration_id_arr[idx] = counter_stepid

id_data = {'walker_id': walker_id_arr, 'iteration_id': iteration_id_arr}
id_df = pd.DataFrame(id_data, index=emcee_table.index)
emcee_table = emcee_table.assign(**id_df)
emcee_table = emcee_table.loc[emcee_table.walker_id.values > 0]
emcee_table = emcee_table.astype(np.float64)

grps = emcee_table.groupby('iteration_id')
grp_keys = grps.groups.keys()

halo = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
stellar = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
lowmass = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
highmass = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
scatter = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
for idx,key in enumerate(grp_keys):
    group = grps.get_group(key)
    halo_mean = np.mean(group.halo.values)
    halo_std = np.std(group.halo.values)
    halo[0][idx] = halo_mean
    halo[1][idx] = halo_std
    stellar_mean = np.mean(group.stellar.values)
    stellar_std = np.std(group.stellar.values)
    stellar[0][idx] = stellar_mean
    stellar[1][idx] = stellar_std
    lowmass_mean = np.mean(group.lowmass.values)
    lowmass_std = np.std(group.lowmass.values)
    lowmass[0][idx] = lowmass_mean
    lowmass[1][idx] = lowmass_std
    highmass_mean = np.mean(group.highmass.values)
    highmass_std = np.std(group.highmass.values)
    highmass[0][idx] = highmass_mean
    highmass[1][idx] = highmass_std
    scatter_mean = np.mean(group.scatter.values)
    scatter_std = np.std(group.scatter.values)
    scatter[0][idx] = scatter_mean
    scatter[1][idx] = scatter_std

behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]
grp_keys = list(grp_keys)

fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, \
    figsize=(10,10))
ax1.plot(grp_keys, halo[0],c='#941266',ls='--', marker='o')
ax1.axhline(behroozi10_param_vals[0],color='lightgray')
ax2.plot(grp_keys, stellar[0], c='#941266',ls='--', marker='o')
ax2.axhline(behroozi10_param_vals[1],color='lightgray')
ax3.plot(grp_keys, lowmass[0], c='#941266',ls='--', marker='o')
ax3.axhline(behroozi10_param_vals[2],color='lightgray')
ax4.plot(grp_keys, highmass[0], c='#941266',ls='--', marker='o')
ax4.axhline(behroozi10_param_vals[3],color='lightgray')
ax5.plot(grp_keys, scatter[0], c='#941266',ls='--', marker='o')
ax5.axhline(behroozi10_param_vals[4],color='lightgray')

ax1.fill_between(grp_keys, halo[0]-halo[1], halo[0]+halo[1], alpha=0.3, \
    color='#941266')
ax2.fill_between(grp_keys, stellar[0]-stellar[1], stellar[0]+stellar[1], \
    alpha=0.3, color='#941266')
ax3.fill_between(grp_keys, lowmass[0]-lowmass[1], lowmass[0]+lowmass[1], \
    alpha=0.3, color='#941266')
ax4.fill_between(grp_keys, highmass[0]-highmass[1], highmass[0]+highmass[1], \
    alpha=0.3, color='#941266')
ax5.fill_between(grp_keys, scatter[0]-scatter[1], scatter[0]+scatter[1], \
    alpha=0.3, color='#941266')

ax1.set_ylabel(r"$\boldmath{M_{1}}$")
ax2.set_ylabel(r"$\boldmath{M_{*(b),0}}$")
ax3.set_ylabel(r"$\boldsymbol{\beta}$")
ax4.set_ylabel(r"$\boldsymbol{\delta}$")
ax5.set_ylabel(r"$\boldsymbol{\xi}$")

ax1.annotate(behroozi10_param_vals[0], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax2.annotate(behroozi10_param_vals[1], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax3.annotate(behroozi10_param_vals[2], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax4.annotate(behroozi10_param_vals[3], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax5.annotate(behroozi10_param_vals[4], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)

# ax1.text(495,12.4,r'$\mathbf{12.35}$',bbox=dict(boxstyle="square",ec='k',
#     fc='lightgray',alpha=0.5),size=10)
# ax2.text(495,10.78,r'$\mathbf{10.72}$',bbox=dict(boxstyle="square",ec='k',
#     fc='lightgray',alpha=0.5),size=10)
# ax3.text(495,0.49,r'$\mathbf{0.44}$',bbox=dict(boxstyle="square",ec='k',
#     fc='lightgray',alpha=0.5),size=10)
# ax4.text(495,0.9,r'$\mathbf{0.57}$',bbox=dict(boxstyle="square",ec='k',
#     fc='lightgray',alpha=0.5),size=10)
# ax5.text(495,0.25,r'$\mathbf{0.15}$',bbox=dict(boxstyle="square",ec='k',
#     fc='lightgray',alpha=0.5),size=10)
plt.xlabel(r"$\mathbf{iteration\ number}$")
plt.show()
'''
dat_file = pd.read_csv('mcmc_eco.dat', sep='\s+', 
    names=['halo','stellar','lowmass','highmass','scatter']) 

raw_file = pd.read_csv('mcmc_eco_raw.txt', sep='\s+', 
    names=['halo','stellar','lowmass','highmass','scatter'])

for idx,row in enumerate(dat_file.values): 
    if np.isnan(row)[4] == True and np.isnan(row)[3] == False: 
        scatter_val = dat_file.values[idx+1][0] 
        row[4] = scatter_val

dat_file = dat_file.dropna(axis='index',how='any').reset_index(drop=True)

sampler = []
chunk = []
for idx,row in raw_file.iterrows():
    if row[0] == '#':
        chunk = np.array(chunk)
        sampler.append(chunk)
        chunk = []
    else:
        row = row.astype(np.float64)
        chunk.append(row)
sampler = np.array(sampler)

raw_file = sampler.reshape(250000,5)

raw_file = pd.DataFrame(raw_file, 
    columns=['halo','stellar','lowmass','highmass','scatter'])


walker_id_arr = np.zeros(len(dat_file),dtype=int)
iteration_id_arr = np.zeros(len(dat_file),dtype=int)
counter_wid = 0
counter_itid = 0
for idx,row in dat_file.iterrows():
    if idx % 1000 == 0:
        counter_wid += 1
        counter_itid = 0
    counter_itid += 1
    walker_id_arr[idx] = int(counter_wid)
    iteration_id_arr[idx] = int(counter_itid)

dat_file['walker_id'] = walker_id_arr
dat_file['iter_id'] = iteration_id_arr

first_250_raw_compare = dat_file.loc[dat_file.iter_id == 1]
first_250_raw_compare = first_250_raw_compare[['halo','stellar','lowmass','highmass','scatter']]
first_250_raw_compare = first_250_raw_compare.round({'halo':2, 'stellar':2, \
    'lowmass':2, 'highmass':2, 'scatter':2}) 

len(pd.merge(raw_file[:250], first_250_raw_compare, 
    on=['halo','stellar','lowmass','highmass','scatter'], how='inner'))
'''