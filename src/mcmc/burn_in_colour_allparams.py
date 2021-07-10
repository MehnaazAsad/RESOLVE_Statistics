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
import os

__author__ = '{Mehnaaz Asad}'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

def find_nearest(array, value): 
    array = np.asarray(array) 
    idx = (np.abs(array - value)).argmin() 
    return array[idx] 

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
nwalkers = 100
run = 32

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run{0}/'.format(run)
else:
    path_to_proc = path_to_proc + 'bmhm_run3/'

chain_fname = path_to_proc + 'mcmc_{0}_colour_raw.txt'.format(survey)

if quenching == 'hybrid':
    emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
        names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mstar_q','Mhalo_q','mu','nu'], sep='\s+')

    for idx,row in enumerate(emcee_table.values):

        ## For cases where 5 params on one line and 3 on the next
        if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
            mhalo_q_val = emcee_table.values[idx+1][0]
            mu_val = emcee_table.values[idx+1][1]
            nu_val = emcee_table.values[idx+1][2]
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

        ## For cases where 4 params on one line, 4 on the next and 1 on the 
        ## third line (numbers in scientific notation unlike case above)
        elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            mstar_q_val = emcee_table.values[idx+1][1]
            mhalo_q_val = emcee_table.values[idx+1][2]
            mu_val = emcee_table.values[idx+1][3]
            nu_val = emcee_table.values[idx+2][0]
            row[4] = scatter_val
            row[5] = mstar_q_val
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)


elif quenching == 'halo':
    emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
        names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mhalo_qc','Mhalo_qs','mu_c','mu_s'], sep='\s+')

    for idx,row in enumerate(emcee_table.values):

        ## For cases where 5 params on one line and 3 on the next
        if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
            mhalo_qs_val = emcee_table.values[idx+1][0]
            mu_c_val = emcee_table.values[idx+1][1]
            mu_s_val = emcee_table.values[idx+1][2]
            row[6] = mhalo_qs_val
            row[7] = mu_c_val
            row[8] = mu_s_val 

        ## For cases where 4 params on one line, 4 on the next and 1 on the 
        ## third line (numbers in scientific notation unlike case above)
        elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            mhalo_qc_val = emcee_table.values[idx+1][1]
            mhalo_qs_val = emcee_table.values[idx+1][2]
            mu_c_val = emcee_table.values[idx+1][3]
            mu_s_val = emcee_table.values[idx+2][0]
            row[4] = scatter_val
            row[5] = mhalo_qc_val
            row[6] = mhalo_qs_val
            row[7] = mu_c_val
            row[8] = mu_s_val 

    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)


chi2_fname = path_to_proc + '{0}_colour_chi2.txt'.format(survey)
chi2_df = pd.read_csv(chi2_fname,header=None,names=['chisquared'])
# chi2 = np.log10(chi2_df.chisquared.values)
emcee_table['chi2'] = chi2_df['chisquared']

# Each chunk is now a step and within each chunk, each row is a walker
# Different from what it used to be where each chunk was a walker and 
# within each chunk, each row was a step

walker_id_arr = np.zeros(len(emcee_table))
iteration_id_arr = np.zeros(len(emcee_table))
counter_wid = 0
counter_stepid = 0
for idx,row in emcee_table.iterrows():
    counter_wid += 1
    if idx % nwalkers == 0:
        counter_stepid += 1
        counter_wid = 1
    walker_id_arr[idx] = counter_wid
    iteration_id_arr[idx] = counter_stepid

id_data = {'walker_id': walker_id_arr, 'iteration_id': iteration_id_arr}
id_df = pd.DataFrame(id_data, index=emcee_table.index)
emcee_table = emcee_table.assign(**id_df)

grps = emcee_table.groupby('iteration_id')
grp_keys = grps.groups.keys()

if quenching == 'hybrid':
    Mhalo_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mstar_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mlow_slope = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mhigh_slope = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    scatter = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]

    Mstar_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mhalo_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    nu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    chi2 = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]

    for idx,key in enumerate(grp_keys):
        group = grps.get_group(key)
        Mhalo_c_mean = np.mean(group.Mhalo_c.values)
        Mhalo_c_std = np.std(group.Mhalo_c.values)
        Mhalo_c[0][idx] = Mhalo_c_mean
        Mhalo_c[1][idx] = Mhalo_c_std
        Mstar_c_mean = np.mean(group.Mstar_c.values)
        Mstar_c_std = np.std(group.Mstar_c.values)
        Mstar_c[0][idx] = Mstar_c_mean
        Mstar_c[1][idx] = Mstar_c_std
        mlow_slope_mean = np.mean(group.mlow_slope.values)
        mlow_slope_std = np.std(group.mlow_slope.values)
        mlow_slope[0][idx] = mlow_slope_mean
        mlow_slope[1][idx] = mlow_slope_std
        mhigh_slope_mean = np.mean(group.mhigh_slope.values)
        mhigh_slope_std = np.std(group.mhigh_slope.values)
        mhigh_slope[0][idx] = mhigh_slope_mean
        mhigh_slope[1][idx] = mhigh_slope_std
        scatter_mean = np.mean(group.scatter.values)
        scatter_std = np.std(group.scatter.values)
        scatter[0][idx] = scatter_mean
        scatter[1][idx] = scatter_std

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

        chi2_mean = np.mean(group.chi2.values)
        chi2_std = np.std(group.chi2.values)
        chi2[0][idx] = chi2_mean
        chi2[1][idx] = chi2_std
    
    Mhalo_c_fid = 12.35
    Mstar_c_fid = 10.72
    mlow_slope_fid = 0.44
    mhigh_slope_fid = 0.57
    scatter_fid = 0.15
    behroozi10_param_vals = [Mhalo_c_fid, Mstar_c_fid, mlow_slope_fid, \
        mhigh_slope_fid, scatter_fid]

    Mstar_q_fid = 10.5 # Msun/h
    Mh_q_fid = 13.76 # Msun/h
    mu_fid = 0.69
    nu_fid = 0.15
    zumandelbaum_param_vals = [Mstar_q_fid, Mh_q_fid, mu_fid, nu_fid]
    
    grp_keys = list(grp_keys)

    fig1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, 
        sharex=True, figsize=(10,10))
    ax1.plot(grp_keys, Mhalo_c[0],c='#941266',ls='--', marker='o')
    ax1.axhline(behroozi10_param_vals[0],color='lightgray')
    ax2.plot(grp_keys, Mstar_c[0], c='#941266',ls='--', marker='o')
    ax2.axhline(behroozi10_param_vals[1],color='lightgray')
    ax3.plot(grp_keys, mlow_slope[0], c='#941266',ls='--', marker='o')
    ax3.axhline(behroozi10_param_vals[2],color='lightgray')
    ax4.plot(grp_keys, mhigh_slope[0], c='#941266',ls='--', marker='o')
    ax4.axhline(behroozi10_param_vals[3],color='lightgray')
    ax5.plot(grp_keys, scatter[0], c='#941266',ls='--', marker='o')
    ax5.axhline(behroozi10_param_vals[4],color='lightgray')

    ax6.plot(grp_keys, Mstar_q[0],c='#941266',ls='--', marker='o')
    ax6.axhline(zumandelbaum_param_vals[0],color='lightgray')
    ax7.plot(grp_keys, Mhalo_q[0], c='#941266',ls='--', marker='o')
    ax7.axhline(zumandelbaum_param_vals[1],color='lightgray')
    ax8.plot(grp_keys, mu[0], c='#941266',ls='--', marker='o')
    ax8.axhline(zumandelbaum_param_vals[2],color='lightgray')
    ax9.plot(grp_keys, nu[0], c='#941266',ls='--', marker='o')
    ax9.axhline(zumandelbaum_param_vals[3],color='lightgray')
    ax10.plot(grp_keys, chi2[0], c='#941266',ls='--', marker='o')

    ax1.fill_between(grp_keys, Mhalo_c[0]-Mhalo_c[1], Mhalo_c[0]+Mhalo_c[1], 
        alpha=0.3, color='#941266')
    ax2.fill_between(grp_keys, Mstar_c[0]-Mstar_c[1], Mstar_c[0]+Mstar_c[1], \
        alpha=0.3, color='#941266')
    ax3.fill_between(grp_keys, mlow_slope[0]-mlow_slope[1], \
        mlow_slope[0]+mlow_slope[1], alpha=0.3, color='#941266')
    ax4.fill_between(grp_keys, mhigh_slope[0]-mhigh_slope[1], \
        mhigh_slope[0]+mhigh_slope[1], alpha=0.3, color='#941266')
    ax5.fill_between(grp_keys, scatter[0]-scatter[1], scatter[0]+scatter[1], \
        alpha=0.3, color='#941266')

    ax6.fill_between(grp_keys, Mstar_q[0]-Mstar_q[1], Mstar_q[0]+Mstar_q[1], 
        alpha=0.3, color='#941266')
    ax7.fill_between(grp_keys, Mhalo_q[0]-Mhalo_q[1], Mhalo_q[0]+Mhalo_q[1], \
        alpha=0.3, color='#941266')
    ax8.fill_between(grp_keys, mu[0]-mu[1], mu[0]+mu[1], \
        alpha=0.3, color='#941266')
    ax9.fill_between(grp_keys, nu[0]-nu[1], nu[0]+nu[1], \
        alpha=0.3, color='#941266')
    ax10.fill_between(grp_keys, chi2[0]-chi2[1], chi2[0]+chi2[1], \
        alpha=0.3, color='#941266')

    ax1.set_ylabel(r"$\mathbf{log_{10}\ M_{1}}$")
    ax2.set_ylabel(r"$\mathbf{log_{10}\ M_{*}}$")
    ax3.set_ylabel(r"$\boldsymbol{\beta}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax4.set_ylabel(r"$\boldsymbol{\delta}$")
    # ax10.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{{\ \chi}^2}$")
    ax5.set_ylabel(r"$\boldsymbol{\xi}$")

    ax6.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{*}}$")
    ax7.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{h}}$")
    ax8.set_ylabel(r"$\boldsymbol{\mu}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax9.set_ylabel(r"$\boldsymbol{\nu}$")
    # ax10.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{{\ \chi}^2}$")
    ax10.set_ylabel(r"$\boldsymbol{{\chi}^2}$")
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')

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

    ax6.annotate(zumandelbaum_param_vals[0], (0.95,0.85), xycoords='axes fraction', 
        bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
    ax7.annotate(zumandelbaum_param_vals[1], (0.95,0.85), xycoords='axes fraction', 
        bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
    ax8.annotate(zumandelbaum_param_vals[2], (0.95,0.85), xycoords='axes fraction', 
        bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
    ax9.annotate(zumandelbaum_param_vals[3], (0.95,0.85), xycoords='axes fraction', 
        bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
    
    plt.xlabel(r"$\mathbf{iteration\ number}$")
    plt.show()

elif quenching == 'halo':
    Mhalo_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mstar_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mlow_slope = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mhigh_slope = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    scatter = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]

    Mh_qc = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mh_qs = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu_s = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    chi2 = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]

    for idx,key in enumerate(grp_keys):
        group = grps.get_group(key)
        Mhalo_c_mean = np.mean(group.Mhalo_c.values)
        Mhalo_c_std = np.std(group.Mhalo_c.values)
        Mhalo_c[0][idx] = Mhalo_c_mean
        Mhalo_c[1][idx] = Mhalo_c_std
        Mstar_c_mean = np.mean(group.Mstar_c.values)
        Mstar_c_std = np.std(group.Mstar_c.values)
        Mstar_c[0][idx] = Mstar_c_mean
        Mstar_c[1][idx] = Mstar_c_std
        mlow_slope_mean = np.mean(group.mlow_slope.values)
        mlow_slope_std = np.std(group.mlow_slope.values)
        mlow_slope[0][idx] = mlow_slope_mean
        mlow_slope[1][idx] = mlow_slope_std
        mhigh_slope_mean = np.mean(group.mhigh_slope.values)
        mhigh_slope_std = np.std(group.mhigh_slope.values)
        mhigh_slope[0][idx] = mhigh_slope_mean
        mhigh_slope[1][idx] = mhigh_slope_std
        scatter_mean = np.mean(group.scatter.values)
        scatter_std = np.std(group.scatter.values)
        scatter[0][idx] = scatter_mean
        scatter[1][idx] = scatter_std

        Mh_qc_mean = np.mean(group.Mh_qc.values)
        Mh_qc_std = np.std(group.Mh_qc.values)
        Mh_qc[0][idx] = Mh_qc_mean
        Mh_qc[1][idx] = Mh_qc_std
        Mh_qs_mean = np.mean(group.Mh_qs.values)
        Mh_qs_std = np.std(group.Mh_qs.values)
        Mh_qs[0][idx] = Mh_qs_mean
        Mh_qs[1][idx] = Mh_qs_std
        mu_c_mean = np.mean(group.mu_c.values)
        mu_c_std = np.std(group.mu_c.values)
        mu_c[0][idx] = mu_c_mean
        mu_c[1][idx] = mu_c_std
        mu_s_mean = np.mean(group.mu_s.values)
        mu_s_std = np.std(group.mu_s.values)
        mu_s[0][idx] = mu_s_mean
        mu_s[1][idx] = mu_s_std

        chi2_mean = np.mean(group.chi2.values)
        chi2_std = np.std(group.chi2.values)
        chi2[0][idx] = chi2_mean
        chi2[1][idx] = chi2_std

    #TODO modify plot making to add behroozi params
    zumandelbaum_param_vals = [12.2, 12.17, 0.38, 0.15]
    grp_keys = list(grp_keys)

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, \
        figsize=(10,10))
    ax1.plot(grp_keys, Mh_qc[0],c='#941266',ls='--', marker='o')
    ax1.axhline(zumandelbaum_param_vals[0],color='lightgray')
    ax2.plot(grp_keys, Mh_qs[0], c='#941266',ls='--', marker='o')
    ax2.axhline(zumandelbaum_param_vals[1],color='lightgray')
    ax3.plot(grp_keys, mu_c[0], c='#941266',ls='--', marker='o')
    ax3.axhline(zumandelbaum_param_vals[2],color='lightgray')
    ax4.plot(grp_keys, mu_s[0], c='#941266',ls='--', marker='o')
    ax4.axhline(zumandelbaum_param_vals[3],color='lightgray')
    ax5.plot(grp_keys, chi2[0], c='#941266',ls='--', marker='o')

    ax1.fill_between(grp_keys, Mh_qc[0]-Mh_qc[1], Mh_qc[0]+Mh_qc[1], 
        alpha=0.3, color='#941266')
    ax2.fill_between(grp_keys, Mh_qs[0]-Mh_qs[1], Mh_qs[0]+Mh_qs[1], \
        alpha=0.3, color='#941266')
    ax3.fill_between(grp_keys, mu_c[0]-mu_c[1], mu_c[0]+mu_c[1], \
        alpha=0.3, color='#941266')
    ax4.fill_between(grp_keys, mu_s[0]-mu_s[1], mu_s[0]+mu_s[1], \
        alpha=0.3, color='#941266')
    ax5.fill_between(grp_keys, chi2[0]-chi2[1], chi2[0]+chi2[1], \
        alpha=0.3, color='#941266')

    ax1.set_ylabel(r"$\mathbf{log_{10}\ Mh_{qc}}$")
    ax2.set_ylabel(r"$\mathbf{log_{10}\ Mh_{qs}}$")
    ax3.set_ylabel(r"$\boldsymbol{\ mu_{c}}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax4.set_ylabel(r"$\boldsymbol{\ mu_{s}}$")
    ax5.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{{\ \chi}^2}$")
    # ax5.set_ylabel(r"$\boldsymbol{{\chi}^2}$")
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
    plt.xlabel(r"$\mathbf{iteration\ number}$", fontsize=20)
    plt.show()


######################## Calculate acceptance fraction ########################

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run{0}/'.format(run)
else:
    path_to_proc = path_to_proc + 'bmhm_run3/'

chain_fname = path_to_proc + 'mcmc_{0}_colour_raw.txt'.format(survey)

if quenching == 'hybrid':
    emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
        names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mstar_q','Mhalo_q','mu','nu'], sep='\s+')

    for idx,row in enumerate(emcee_table.values):

        ## For cases where 5 params on one line and 3 on the next
        if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
            mhalo_q_val = emcee_table.values[idx+1][0]
            mu_val = emcee_table.values[idx+1][1]
            nu_val = emcee_table.values[idx+1][2]
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

        ## For cases where 4 params on one line, 4 on the next and 1 on the 
        ## third line (numbers in scientific notation unlike case above)
        elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            mstar_q_val = emcee_table.values[idx+1][1]
            mhalo_q_val = emcee_table.values[idx+1][2]
            mu_val = emcee_table.values[idx+1][3]
            nu_val = emcee_table.values[idx+2][0]
            row[4] = scatter_val
            row[5] = mstar_q_val
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)


elif quenching == 'halo':
    emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
        names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mstar_q','Mhalo_q','mu','nu'], sep='\s+')

    for idx,row in enumerate(emcee_table.values):

        ## For cases where 5 params on one line and 3 on the next
        if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
            mhalo_q_val = emcee_table.values[idx+1][0]
            mu_val = emcee_table.values[idx+1][1]
            nu_val = emcee_table.values[idx+1][2]
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

        ## For cases where 4 params on one line, 4 on the next and 1 on the 
        ## third line (numbers in scientific notation unlike case above)
        elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            mstar_q_val = emcee_table.values[idx+1][1]
            mhalo_q_val = emcee_table.values[idx+1][2]
            mu_val = emcee_table.values[idx+1][3]
            nu_val = emcee_table.values[idx+2][0]
            row[4] = scatter_val
            row[5] = mstar_q_val
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)

num_unique_rows = emcee_table[['Mhalo_c', 'Mstar_c', 'mlow_slope', 
    'mhigh_slope', 'scatter','Mstar_q','Mhalo_q','mu','nu']].drop_duplicates().\
    shape[0]
num_rows = len(emcee_table)
acceptance_fraction = num_unique_rows / num_rows
print("Acceptance fraction: {0}%".format(np.round(acceptance_fraction,2)*100))

########################## Calculate autocorrelation ###########################

## Taken from 
## https://github.com/dfm/emcee/blob/b9d6e3e7b1926009baa5bf422ae738d1b06a848a/src/emcee/autocorr.py#L47

import logging
import numpy as np

__all__ = ["function_1d", "integrated_time", "AutocorrError"]

def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i

def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def integrated_time(x, c=5, tol=50, quiet=False):
    """Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.
    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.
    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x[:, k, d])
        f /= n_w
        taus = 2.0 * np.cumsum(f) - 1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est, msg)
        logging.warning(msg)

    return tau_est

class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.
    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.
    """

    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super(AutocorrError, self).__init__(*args, **kwargs)

# Following https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr
chain_eg = emcee_table.values.reshape(500, 30, 9)[:, :, 0].T
integrated_time(chain_eg)

N = np.exp(np.linspace(np.log(100), np.log(chain_eg.shape[1]), 10)).astype(int)
new = np.empty(len(N))
for i, n in enumerate(N):
    new[i] = integrated_time(chain_eg[:, :n])

fig1 = plt.figure()
plt.loglog(N, new, "o-", label="new")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimate")
plt.legend(fontsize=14)
plt.show()
