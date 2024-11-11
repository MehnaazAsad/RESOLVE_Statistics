"""
{This script reads in the raw chain and plots times series for all parameters
 in order to identify the burn-in}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import emcee
import math
import os

__author__ = '{Mehnaaz Asad}'

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{bm}")
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=2, size=7)
plt.rc('ytick.major', width=2, size=7)

def find_nearest(array, value): 
    """Finds the element in array that is closest to the value

    Args:
        array (numpy.array): Array of values
        value (numpy.float): Value to find closest match to

    Returns:
        numpy.float: Closest match found in array
    """
    array = np.asarray(array) 
    idx = (np.abs(array - value)).argmin() 
    return array[idx] 

def get_median_and_percs(df):
    param_subset = df.iloc[:,:9]
    medians = param_subset.median(axis=0)
    percs = param_subset.quantile([0.16, 0.84],axis=0)
    lower_limits = medians - percs.iloc[0,:]
    upper_limits = percs.iloc[1,:] - medians
    medians, lower_limits, upper_limits = np.round(medians, 2), \
        np.round(lower_limits, 2), np.round(upper_limits, 2)
    return medians, lower_limits, upper_limits

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

survey = 'eco'
mf_type = 'smf'
quenching = 'halo'
nwalkers = 500
run = 111

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run{0}/'.format(run)
else:
    path_to_proc = path_to_proc + 'bmhm_colour_run{0}/'.format(run)

if run >= 37:
    reader = emcee.backends.HDFBackend(
        path_to_proc + "chain.h5", read_only=True)
    # reader = emcee.backends.HDFBackend(
    #     "/Users/asadm2/Desktop/chain_{0}_pca.h5".format(quenching), read_only=True)
    flatchain = reader.get_chain(flat=True)

    names_hybrid=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'Mstar_q','Mhalo_q','mu','nu']
    names_halo=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mh_qc','Mh_qs','mu_c','mu_s']

    if quenching == 'hybrid':
        emcee_table = pd.DataFrame(flatchain, columns=names_hybrid)
    elif quenching == 'halo':
        emcee_table = pd.DataFrame(flatchain, columns=names_halo)       

    chi2 = reader.get_blobs(flat=True)
    emcee_table['chi2'] = chi2

    medians, lower_limits, upper_limits = get_median_and_percs(emcee_table)
else:
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
            'Mh_qc','Mh_qs','mu_c','mu_s'], sep='\s+')

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

    bf_params_per_iteration = []

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

        bf_params_per_iteration.append(group.loc[group.chi2 == min(group.chi2)].values[0][:10])
    bf_params_per_iteration = np.array(bf_params_per_iteration)

    Mhalo_c_fid = 12.35
    Mstar_c_fid = 10.72
    mlow_slope_fid = 0.44
    mhigh_slope_fid = 0.57
    scatter_fid = 0.15
    behroozi10_param_vals = [Mhalo_c_fid, Mstar_c_fid, mlow_slope_fid, \
        mhigh_slope_fid, scatter_fid]

    #* Hybrid
    Mstar_q_fid = 10.49 # Msun/h
    Mh_q_fid = 14.03 # Msun/h
    mu_fid = 0.69
    nu_fid = 0.148
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

    ax1.plot(grp_keys, bf_params_per_iteration.T[0], c='k', marker='*')
    ax2.plot(grp_keys, bf_params_per_iteration.T[1], c='k',marker='*')
    ax3.plot(grp_keys, bf_params_per_iteration.T[2], c='k',marker='*')
    ax4.plot(grp_keys, bf_params_per_iteration.T[3], c='k',marker='*')
    ax5.plot(grp_keys, bf_params_per_iteration.T[4], c='k',marker='*')

    ax6.plot(grp_keys, bf_params_per_iteration.T[5], c='k', marker='*')
    ax7.plot(grp_keys, bf_params_per_iteration.T[6], c='k', marker='*')
    ax8.plot(grp_keys, bf_params_per_iteration.T[7], c='k', marker='*')
    ax9.plot(grp_keys, bf_params_per_iteration.T[8], c='k', marker='*')
    ax10.plot(grp_keys, bf_params_per_iteration.T[9], c='k', marker='*')

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
    ax10.set_ylabel(r"$\mathbf{{\chi}^2}$")
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

    bf_params_per_iteration = []

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

        bf_params_per_iteration.append(group.loc[group.chi2 == min(group.chi2)].values[0][:10])
    bf_params_per_iteration = np.array(bf_params_per_iteration)

    Mhalo_c_fid = 12.35
    Mstar_c_fid = 10.72
    mlow_slope_fid = 0.44
    mhigh_slope_fid = 0.57
    scatter_fid = 0.15
    behroozi10_param_vals = [Mhalo_c_fid, Mstar_c_fid, mlow_slope_fid, \
        mhigh_slope_fid, scatter_fid]

    #* Halo 

    Mh_qc_fid = 12.61 # Msun/h
    Mh_qs_fid = 13.5 # Msun/h
    mu_c_fid = 0.40
    mu_s_fid = 0.148
    zumandelbaum_param_vals = [Mh_qc_fid, Mh_qs_fid, mu_c_fid, mu_s_fid]
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

    ax6.plot(grp_keys, Mh_qc[0],c='#941266',ls='--', marker='o')
    ax6.axhline(zumandelbaum_param_vals[0],color='lightgray')
    ax7.plot(grp_keys, Mh_qs[0], c='#941266',ls='--', marker='o')
    ax7.axhline(zumandelbaum_param_vals[1],color='lightgray')
    ax8.plot(grp_keys, mu_c[0], c='#941266',ls='--', marker='o')
    ax8.axhline(zumandelbaum_param_vals[2],color='lightgray')
    ax9.plot(grp_keys, mu_s[0], c='#941266',ls='--', marker='o')
    ax9.axhline(zumandelbaum_param_vals[3],color='lightgray')
    ax10.plot(grp_keys, chi2[0], c='#941266',ls='--', marker='o')

    ax1.plot(grp_keys, bf_params_per_iteration.T[0], c='k', marker='*')
    ax2.plot(grp_keys, bf_params_per_iteration.T[1], c='k',marker='*')
    ax3.plot(grp_keys, bf_params_per_iteration.T[2], c='k',marker='*')
    ax4.plot(grp_keys, bf_params_per_iteration.T[3], c='k',marker='*')
    ax5.plot(grp_keys, bf_params_per_iteration.T[4], c='k',marker='*')

    ax6.plot(grp_keys, bf_params_per_iteration.T[5], c='k', marker='*')
    ax7.plot(grp_keys, bf_params_per_iteration.T[6], c='k', marker='*')
    ax8.plot(grp_keys, bf_params_per_iteration.T[7], c='k', marker='*')
    ax9.plot(grp_keys, bf_params_per_iteration.T[8], c='k', marker='*')
    ax10.plot(grp_keys, bf_params_per_iteration.T[9], c='k', marker='*')

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


    ax6.fill_between(grp_keys, Mh_qc[0]-Mh_qc[1], Mh_qc[0]+Mh_qc[1], 
        alpha=0.3, color='#941266')
    ax7.fill_between(grp_keys, Mh_qs[0]-Mh_qs[1], Mh_qs[0]+Mh_qs[1], \
        alpha=0.3, color='#941266')
    ax8.fill_between(grp_keys, mu_c[0]-mu_c[1], mu_c[0]+mu_c[1], \
        alpha=0.3, color='#941266')
    ax9.fill_between(grp_keys, mu_s[0]-mu_s[1], mu_s[0]+mu_s[1], \
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

    ax6.set_ylabel(r"$\mathbf{log_{10}\ Mh_{qc}}$")
    ax7.set_ylabel(r"$\mathbf{log_{10}\ Mh_{qs}}$")
    ax8.set_ylabel(r"$\boldsymbol{{\mu}_{c}}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax9.set_ylabel(r"$\boldsymbol{{\mu}_{s}}$")
    ax10.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{{\ \chi}^2}$")
    # ax5.set_ylabel(r"$\boldsymbol{{\chi}^2}$")
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
    ax9.annotate(0.15, (0.95,0.85), xycoords='axes fraction', 
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

names_hybrid=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mstar_q','Mhalo_q','mu','nu']
names_halo=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
    'Mh_qc','Mh_qs','mu_c','mu_s']

if quenching == "hybrid":
    num_unique_rows = emcee_table[names_hybrid].drop_duplicates().\
        shape[0]
elif quenching == "halo":
    num_unique_rows = emcee_table[names_halo].drop_duplicates().\
        shape[0]
num_rows = len(emcee_table)
acceptance_fraction = num_unique_rows / num_rows
print("Acceptance fraction: {0}%".format(np.round(acceptance_fraction,2)*100))

