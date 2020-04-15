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
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

survey = 'eco'
mf_type = 'smf'

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run9/'
# else:
#     path_to_proc = path_to_proc + 'bmhm_run3/'

# chain_fname = path_to_proc + 'combined_processed_{0}_raw.txt'.format(survey)
chain_fname = path_to_proc + 'mcmc_{0}_colour_raw.txt'.format(survey)

emcee_table = pd.read_csv(chain_fname, delim_whitespace=True, 
    names=['Mstar_q','Mhalo_q','mu','nu'], 
    header=None)

emcee_table = emcee_table[emcee_table.Mstar_q.values != '#']
emcee_table.Mstar_q = emcee_table.Mstar_q.astype(np.float64)
emcee_table.Mhalo_q = emcee_table.Mhalo_q.astype(np.float64)
emcee_table.mu = emcee_table.mu.astype(np.float64)
emcee_table.nu = emcee_table.nu.astype(np.float64)

for idx,row in enumerate(emcee_table.values):
    if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
        nu_val = emcee_table.values[idx+1][0]
        row[3] = nu_val 
emcee_table = emcee_table.dropna(axis='index', how='any').\
    reset_index(drop=True)

emcee_table.nu = np.log10(emcee_table.nu)

# Each chunk is now a step and within each chunk, each row is a walker
# Different from what it used to be where each chunk was a walker and 
# within each chunk, each row was a step

walker_id_arr = np.zeros(len(emcee_table))
iteration_id_arr = np.zeros(len(emcee_table))
counter_wid = 0
counter_stepid = 0
for idx,row in emcee_table.iterrows():
    counter_wid += 1
    if idx % 256 == 0:
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

Mstar_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
Mhalo_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
mu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
nu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
for idx,key in enumerate(grp_keys):
    group = grps.get_group(key)
    Mstar_q_mean = np.mean(group.Mstar_q.values)
    Mstar_q_std = np.std(group.Mstar_q.values)
    Mstar_q[0][idx] = Mstar_q_mean
    Mstar_q[1][idx] = Mstar_q_std
    Mhalo_q_mean = np.mean(group.Mhalo_q.values)
    Mhalo_q_std = np.std(group.Mhalo_q.values)
    Mhalo_q[0][idx] = Mhalo_q_mean
    Mhalo_q[1][idx] = Mhalo_q_std
    mu_mean = np.mean(group.mu.values)
    mu_std = np.std(group.mu.values)
    mu[0][idx] = mu_mean
    mu[1][idx] = mu_std
    nu_mean = np.mean(group.nu.values)
    nu_std = np.std(group.nu.values)
    nu[0][idx] = nu_mean
    nu[1][idx] = nu_std

zumandelbaum_param_vals = [10.5, 13.76, 0.69, np.log10(0.15)]
grp_keys = list(grp_keys)

fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, \
    figsize=(10,10))
ax1.plot(grp_keys, Mstar_q[0],c='#941266',ls='--', marker='o')
ax1.axhline(zumandelbaum_param_vals[0],color='lightgray')
ax2.plot(grp_keys, Mhalo_q[0], c='#941266',ls='--', marker='o')
ax2.axhline(zumandelbaum_param_vals[1],color='lightgray')
ax3.plot(grp_keys, mu[0], c='#941266',ls='--', marker='o')
ax3.axhline(zumandelbaum_param_vals[2],color='lightgray')
ax4.plot(grp_keys, nu[0], c='#941266',ls='--', marker='o')
ax4.axhline(zumandelbaum_param_vals[3],color='lightgray')
# ax4.scatter(600, -1.059910, marker='*', c='k')
ax4.scatter(600, -1.165710, marker='*', c='k', s=70)
ax4.scatter(600, 2.301931, marker='*', c='k', s=70)


ax1.fill_between(grp_keys, Mstar_q[0]-Mstar_q[1], Mstar_q[0]+Mstar_q[1], 
    alpha=0.3, color='#941266')
ax2.fill_between(grp_keys, Mhalo_q[0]-Mhalo_q[1], Mhalo_q[0]+Mhalo_q[1], \
    alpha=0.3, color='#941266')
ax3.fill_between(grp_keys, mu[0]-mu[1], mu[0]+mu[1], \
    alpha=0.3, color='#941266')
ax4.fill_between(grp_keys, nu[0]-nu[1], nu[0]+nu[1], \
    alpha=0.3, color='#941266')

ax1.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{*}}$")
ax2.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{h}}$")
ax3.set_ylabel(r"$\boldsymbol{\mu}$")
ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
# ax1.set_yscale('log')
# ax2.set_yscale('log')

ax1.annotate(zumandelbaum_param_vals[0], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax2.annotate(zumandelbaum_param_vals[1], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax3.annotate(zumandelbaum_param_vals[2], (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
ax4.annotate(0.15, (0.95,0.85), xycoords='axes fraction', 
    bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
plt.xlabel(r"$\mathbf{iteration\ number}$")
plt.show()