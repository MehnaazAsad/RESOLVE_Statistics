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

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=20)
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
nwalkers = 260

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run27/'
else:
    path_to_proc = path_to_proc + 'bmhm_run3/'

chain_fname = path_to_proc + 'mcmc_{0}_colour_raw.txt'.format(survey)

if quenching == 'hybrid':
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

    # emcee_table.nu = np.log10(emcee_table.nu)

elif quenching == 'halo':
    emcee_table = pd.read_csv(chain_fname, delim_whitespace=True, 
        names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
        header=None)

    emcee_table = emcee_table[emcee_table.Mh_qc.values != '#']
    emcee_table.Mh_qc = emcee_table.Mh_qc.astype(np.float64)
    emcee_table.Mh_qs = emcee_table.Mh_qs.astype(np.float64)
    emcee_table.mu_c = emcee_table.mu_c.astype(np.float64)
    emcee_table.mu_s = emcee_table.mu_s.astype(np.float64)

    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
            mu_s_val = emcee_table.values[idx+1][0]
            row[3] = mu_s_val 
    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)


chi2_fname = path_to_proc + '{0}_colour_chi2.txt'.format(survey)
chi2_df = pd.read_csv(chi2_fname,header=None,names=['chisquared'])
chi2 = np.log10(chi2_df.chisquared.values)
emcee_table['chi2'] = chi2

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
    Mstar_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mhalo_q = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    nu = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    chi2 = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
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
        chi2_mean = np.mean(group.chi2.values)
        chi2_std = np.std(group.chi2.values)
        chi2[0][idx] = chi2_mean
        chi2[1][idx] = chi2_std
    
    zumandelbaum_param_vals = [10.5, 13.76, 0.69, 0.15]
    grp_keys = list(grp_keys)

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, \
        figsize=(10,10))
    ax1.plot(grp_keys, Mstar_q[0],c='#941266',ls='--', marker='o')
    ax1.axhline(zumandelbaum_param_vals[0],color='lightgray')
    ax2.plot(grp_keys, Mhalo_q[0], c='#941266',ls='--', marker='o')
    ax2.axhline(zumandelbaum_param_vals[1],color='lightgray')
    ax3.plot(grp_keys, mu[0], c='#941266',ls='--', marker='o')
    ax3.axhline(zumandelbaum_param_vals[2],color='lightgray')
    ax4.plot(grp_keys, nu[0], c='#941266',ls='--', marker='o')
    ax4.axhline(zumandelbaum_param_vals[3],color='lightgray')
    ax5.plot(grp_keys, chi2[0], c='#941266',ls='--', marker='o')

    ax1.fill_between(grp_keys, Mstar_q[0]-Mstar_q[1], Mstar_q[0]+Mstar_q[1], 
        alpha=0.3, color='#941266')
    ax2.fill_between(grp_keys, Mhalo_q[0]-Mhalo_q[1], Mhalo_q[0]+Mhalo_q[1], \
        alpha=0.3, color='#941266')
    ax3.fill_between(grp_keys, mu[0]-mu[1], mu[0]+mu[1], \
        alpha=0.3, color='#941266')
    ax4.fill_between(grp_keys, nu[0]-nu[1], nu[0]+nu[1], \
        alpha=0.3, color='#941266')
    ax5.fill_between(grp_keys, chi2[0]-chi2[1], chi2[0]+chi2[1], \
        alpha=0.3, color='#941266')

    ax1.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{*}}$")
    ax2.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{h}}$")
    ax3.set_ylabel(r"$\boldsymbol{\mu}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax4.set_ylabel(r"$\boldsymbol{\nu}$")
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
    plt.xlabel(r"$\mathbf{iteration\ number}$")
    plt.show()

elif quenching == 'halo':
    Mh_qc = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    Mh_qs = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu_c = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    mu_s = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    chi2 = [np.zeros(len(grp_keys)),np.zeros(len(grp_keys))]
    for idx,key in enumerate(grp_keys):
        group = grps.get_group(key)
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
    plt.xlabel(r"$\mathbf{iteration\ number}$")
    plt.show()


######################## Calculate acceptance fraction ########################
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_colour_run21/'
else:
    path_to_proc = path_to_proc + 'bmhm_run3/'

chain_fname = path_to_proc + 'mcmc_{0}_colour_raw.txt'.format(survey)

if quenching == 'hybrid':
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

    # emcee_table.nu = np.log10(emcee_table.nu)

elif quenching == 'halo':
    emcee_table = pd.read_csv(chain_fname, delim_whitespace=True, 
        names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
        header=None)

    emcee_table = emcee_table[emcee_table.Mh_qc.values != '#']
    emcee_table.Mh_qc = emcee_table.Mh_qc.astype(np.float64)
    emcee_table.Mh_qs = emcee_table.Mh_qs.astype(np.float64)
    emcee_table.mu_c = emcee_table.mu_c.astype(np.float64)
    emcee_table.mu_s = emcee_table.mu_s.astype(np.float64)

    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
            mu_s_val = emcee_table.values[idx+1][0]
            row[3] = mu_s_val 
    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)

num_unique_rows = emcee_table[['Mstar_q','Mhalo_q','mu','nu']].drop_duplicates().shape[0]
num_rows = len(emcee_table)
acceptance_fraction = num_unique_rows / num_rows
print("Acceptance fraction: {0}%".format(np.round(acceptance_fraction,2)*100))

# For behroozi chains
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']

chain_fname = path_to_proc + 'smhm_run6/mcmc_{0}_raw.txt'.\
    format(survey)
emcee_table = pd.read_csv(chain_fname, 
    names=['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',
    'scatter'],header=None, delim_whitespace=True)
emcee_table = emcee_table[emcee_table.mhalo_c.values != '#']
emcee_table.mhalo_c = emcee_table.mhalo_c.astype(np.float64)
emcee_table.mstellar_c = emcee_table.mstellar_c.astype(np.float64)
emcee_table.lowmass_slope = emcee_table.lowmass_slope.astype(np.float64)

for idx,row in enumerate(emcee_table.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = emcee_table.values[idx+1][0]
          row[4] = scatter_val

emcee_table = emcee_table.dropna(axis='index', how='any').reset_index(drop=True)

num_unique_rows = emcee_table[['mhalo_c','mstellar_c','lowmass_slope',\
    'highmass_slope']].drop_duplicates().shape[0]
num_rows = len(emcee_table)
acceptance_fraction = num_unique_rows / num_rows
print("Acceptance fraction: {0}%".format(np.round(acceptance_fraction,2)*100))

################################################################################


def hybrid_quenching_model(theta, gals_df, mock, randint=None):
    """
    Apply hybrid quenching model from Zu and Mandelbaum 2015

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    """

    # parameter values from Table 1 of Zu and Mandelbaum 2015 "prior case"
    Mstar_q = theta[0] # Msun/h
    Mh_q = theta[1] # Msun/h
    mu = theta[2]
    nu = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)
    cen_stellar_mass_arr, sat_stellar_mass_arr = get_stellar_mock(gals_df, mock, \
        randint)

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/(10**Mh_q))**nu))
    f_red_sat = 1 - (g_Mstar * h_Mh)

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, gals_df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Copy of dataframe
    df = gals_df.copy()
    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

def get_host_halo_mock(gals_df, mock):
    """
    Get host halo mass from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_halos: array
        Array of central host halo masses
    sat_halos: array
        Array of satellite host halo masses
    """

    df = gals_df.copy()

    # groups = df.groupby('halo_id')
    # keys = groups.groups.keys()

    # for key in keys:
    #     group = groups.get_group(key)
    # for index, value in enumerate(group.cs_flag):
    #     if value == 1:
    #         cen_halos.append(group.loghalom.values[index])
    #     else:
    #         sat_halos.append(group.loghalom.values[index])

    if mock == 'vishnu':
        cen_halos = []
        sat_halos = []
        for index, value in enumerate(df.cs_flag):
            if value == 1:
                cen_halos.append(df.halo_mvir.values[index])
            else:
                sat_halos.append(df.halo_mvir.values[index])
    else:
        cen_halos = []
        sat_halos = []
        for index, value in enumerate(df.cs_flag):
            if value == 1:
                cen_halos.append(df.loghalom.values[index])
            else:
                sat_halos.append(df.loghalom.values[index])

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(gals_df, mock, randint=None):
    """
    Get stellar mass from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_gals: array
        Array of central stellar masses
    sat_gals: array
        Array of satellite stellar masses
    """

    df = gals_df.copy()
    if mock == 'vishnu':
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(df['{0}'.format(randint)].values[idx])
            elif value == 0:
                sat_gals.append(df['{0}'.format(randint)].values[idx])

    else:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(df.logmstar.values[idx])
            elif value == 0:
                sat_gals.append(df.logmstar.values[idx])

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

def diff_smf(mstar_arr, volume, h1_bool, colour_flag=False):
    """
    Calculates differential stellar mass function in units of h=1.0

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    phi: array
        Array of y-axis values

    err_tot: array
        Array of error values per bin
    
    bins: array
        Array of bin edge values
    """
    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = np.log10(mstar_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        if survey == 'eco' and colour_flag == 'R':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 6
        elif survey == 'eco' and colour_flag == 'B':
            bin_max = np.round(np.log10((10**11) / 2.041), 1)
            bin_num = 6
        elif survey == 'resolvea':
            # different to avoid nan in inverse corr mat
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7
        else:
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7
        bins = np.linspace(bin_min, bin_max, bin_num)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss
    phi = counts / (volume * dm)  # not a log quantity

    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, False, 'R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, False, 'B')
    else:
        # logmstar_col = 'stellar_mass'
        logmstar_col = '{0}'.format(randint_logmstar)
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(table[logmstar_col], volume, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, True, 'R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, True, 'B')
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def read_data_catl(path_to_file, survey):
    """
    Reads survey catalog from file

    Parameters
    ----------
    path_to_file: `string`
        Path to survey catalog file

    survey: `string`
        Name of survey

    Returns
    ---------
    catl: `pandas.DataFrame`
        Survey catalog with grpcz, abs rmag and stellar mass limits
    
    volume: `float`
        Volume of survey

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms','modelu_rcorr']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            # cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            if mf_type == 'smf':
                # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            # cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl, volume, z_median

def std_func(bins, mass_arr, vel_arr):
    ## Calculate std from mean=0

    last_index = len(bins)-1
    i = 0
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        cen_deltav_arr = []
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])
        N = len(cen_deltav_arr)
        mean = 0
        diff_sqrd_arr = []
        for value in cen_deltav_arr:
            diff = value - mean
            diff_sqrd = diff**2
            diff_sqrd_arr.append(diff_sqrd)
        mean_diff_sqrd = np.mean(diff_sqrd_arr)
        std = np.sqrt(mean_diff_sqrd)
        std_arr.append(std)

    return std_arr

def get_deltav_sigma_vishnu_qmcolour(gals_df, randint):
    """
    Calculate spread in velocity dispersion from Vishnu mock (logmstar already 
    in h=1)

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    std_red_arr: numpy array
        Spread in velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Spread in velocity dispersion of blue galaxies
    centers_blue_arr: numpy array
        Bin centers of central stellar mass for blue galaxies
    """

    mock_pd = gals_df.copy()

    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3


    logmstar_col = '{0}'.format(randint)
    g_galtype_col = 'g_galtype_{0}'.format(randint)
    # Using the same survey definition as in mcmc smf i.e excluding the 
    # buffer except no M_r cut since vishnu mock has no M_r info
    mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
        (mock_pd.cz.values <= max_cz) & \
        (mock_pd[logmstar_col].values >= (10**mstar_limit/2.041))]

    red_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
        colour_label == 'R') & (mock_pd[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
        colour_label == 'B') & (mock_pd[g_galtype_col] == 1)].values)

    # Calculating spread in velocity dispersion for galaxies in groups
    # with a red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_pd.loc[mock_pd.groupid == key]
        cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col].\
            values == 1].values[0]
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        deltav = group.cz.values - len(group)*[mean_cz_grp]
        for val in deltav:
            red_deltav_arr.append(val)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(red_cen_stellar_mass_arr))

    red_cen_stellar_mass_arr = np.log10(red_cen_stellar_mass_arr)
    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
        red_deltav_arr)
    std_red = np.array(std_red)

    # Calculating spread in velocity dispersion for galaxies in groups 
    # with a blue central

    blue_deltav_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = mock_pd.loc[mock_pd.groupid == key]
        cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col]\
            .values == 1].values[0]
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        deltav = group.cz.values - len(group)*[mean_cz_grp]
        for val in deltav:
            blue_deltav_arr.append(val)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(blue_cen_stellar_mass_arr))

    blue_cen_stellar_mass_arr = np.log10(blue_cen_stellar_mass_arr)
    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    std_blue = std_func(blue_stellar_mass_bins, \
        blue_cen_stellar_mass_arr, blue_deltav_arr)    
    std_blue = np.array(std_blue)

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
            
    return std_red, std_blue, centers_red, centers_blue

def get_deltav_sigma_mocks_qmcolour(survey, path):
    """
    Calculate spread in velocity dispersion from survey mocks (logmstar converted
    to h=1 units before analysis)

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    std_red_arr: numpy array
        Spread in velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Spread in velocity dispersion of blue galaxies
    centers_blue_arr: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

    std_red_arr = []
    centers_red_arr = []
    std_blue_arr = []
    centers_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
                (mock_pd.logmstar.values >= mstar_limit)]


            Mstar_q = 10.5 # Msun/h
            Mh_q = 13.76 # Msun/h
            mu = 0.69
            nu = 0.15

            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            mock_pd.logmstar = np.log10((10**mock_pd.logmstar) / 2.041)
            red_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
                colour_label == 'R') & (mock_pd.g_galtype == 1)].values)  
            blue_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
                colour_label == 'B') & (mock_pd.g_galtype == 1)].values)

            # Calculating spread in velocity dispersion for galaxies in groups
            # with a red central

            red_deltav_arr = []
            red_cen_stellar_mass_arr = []
            for key in red_subset_grpids: 
                group = mock_pd.loc[mock_pd.groupid == key]
                cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
                    values == 1].values[0]
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                for val in deltav:
                    red_deltav_arr.append(val)
                    red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # print(max(red_cen_stellar_mass_arr))

            if survey == 'eco' or survey == 'resolvea':
                # TODO : check if this is actually correct for resolve a
                red_stellar_mass_bins = np.linspace(8.6,11.2,6)
            elif survey == 'resolveb':
                red_stellar_mass_bins = np.linspace(8.4,11.0,6)
            std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
                red_deltav_arr)
            std_red = np.array(std_red)
            std_red_arr.append(std_red)

            # Calculating spread in velocity dispersion for galaxies in groups 
            # with a blue central

            blue_deltav_arr = []
            blue_cen_stellar_mass_arr = []
            for key in blue_subset_grpids: 
                group = mock_pd.loc[mock_pd.groupid == key]
                cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                    .values == 1].values[0]
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                for val in deltav:
                    blue_deltav_arr.append(val)
                    blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            # print(max(blue_cen_stellar_mass_arr))

            if survey == 'eco' or survey == 'resolvea':
                # TODO : check if this is actually correct for resolve a
                blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
            elif survey == 'resolveb':
                blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
            std_blue = std_func(blue_stellar_mass_bins, \
                blue_cen_stellar_mass_arr, blue_deltav_arr)    
            std_blue = np.array(std_blue)
            std_blue_arr.append(std_blue)

            centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
                red_stellar_mass_bins[:-1])
            centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
                blue_stellar_mass_bins[:-1])
            
            centers_red_arr.append(centers_red)
            centers_blue_arr.append(centers_blue)
    
    std_red_arr = np.array(std_red_arr)
    centers_red_arr = np.array(centers_red_arr)
    std_blue_arr = np.array(std_blue_arr)
    centers_blue_arr = np.array(centers_blue_arr)
            
    return std_red_arr, std_blue_arr, centers_red_arr, centers_blue_arr

def get_deltav_sigma_data(df):
    """
    Measure spread in velocity dispersion separately for red and blue galaxies 
    by binning up central stellar mass (changes logmstar units from h=0.7 to h=1)

    Parameters
    ----------
    df: pandas Dataframe 
        Data catalog

    Returns
    ---------
    std_red: numpy array
        Spread in velocity dispersion of red galaxies
    centers_red: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue: numpy array
        Spread in velocity dispersion of blue galaxies
    centers_blue: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    catl = df.copy()
    if survey == 'eco' or survey == 'resolvea':
        catl = catl.loc[catl.logmstar >= 8.9]
    elif survey == 'resolveb':
        catl = catl.loc[catl.logmstar >= 8.7]
    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
   
    red_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'R') & (catl.fc == 1)].values)  
    blue_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'B') & (catl.fc == 1)].values)


    # Calculating spread in velocity dispersion for galaxies in groups with a 
    # red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.grp == key]
        cen_stellar_mass = group.logmstar.loc[group.fc.\
            values == 1].values[0]
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        deltav = group.cz.values - len(group)*[mean_cz_grp]
        for val in deltav:
            red_deltav_arr.append(val)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
        red_deltav_arr)
    std_red = np.array(std_red)

    # Calculating spread in velocity dispersion for galaxies in groups with a 
    # blue central

    blue_deltav_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.grp == key]
        cen_stellar_mass = group.logmstar.loc[group.fc\
            .values == 1].values[0]
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        deltav = group.cz.values - len(group)*[mean_cz_grp]
        for val in deltav:
            blue_deltav_arr.append(val)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    std_blue = std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
        blue_deltav_arr)    
    std_blue = np.array(std_blue)

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])

    return std_red, centers_red, std_blue, centers_blue

def reading_catls(filename, catl_format='.hdf5'):
    """
    Function to read ECO/RESOLVE catalogues.

    Parameters
    ----------
    filename: string
        path and name of the ECO/RESOLVE catalogue to read

    catl_format: string, optional (default = '.hdf5')
        type of file to read.
        Options:
            - '.hdf5': Reads in a catalogue in HDF5 format

    Returns
    -------
    mock_pd: pandas DataFrame
        DataFrame with galaxy/group information

    Examples
    --------
    # Specifying `filename`
    >>> filename = 'ECO_catl.hdf5'

    # Reading in Catalogue
    >>> mock_pd = reading_catls(filename, format='.hdf5')

    >>> mock_pd.head()
               x          y         z          vx          vy          vz  \
    0  10.225435  24.778214  3.148386  356.112457 -318.894409  366.721832
    1  20.945772  14.500367 -0.237940  168.731766   37.558834  447.436951
    2  21.335835  14.808488  0.004653  967.204407 -701.556763 -388.055115
    3  11.102760  21.782235  2.947002  611.646484 -179.032089  113.388794
    4  13.217764  21.214905  2.113904  120.689598  -63.448833  400.766541

       loghalom  cs_flag  haloid  halo_ngal    ...        cz_nodist      vel_tot  \
    0    12.170        1  196005          1    ...      2704.599189   602.490355
    1    11.079        1  197110          1    ...      2552.681697   479.667489
    2    11.339        1  197131          1    ...      2602.377466  1256.285409
    3    11.529        1  199056          1    ...      2467.277182   647.318259
    4    10.642        1  199118          1    ...      2513.381124   423.326770

           vel_tan     vel_pec     ra_orig  groupid    M_group g_ngal  g_galtype  \
    0   591.399858 -115.068833  215.025116        0  11.702527      1          1
    1   453.617221  155.924074  182.144134        1  11.524787      4          0
    2  1192.742240  394.485714  182.213220        1  11.524787      4          0
    3   633.928896  130.977416  210.441320        2  11.502205      1          1
    4   421.064495   43.706352  205.525386        3  10.899680      1          1

       halo_rvir
    0   0.184839
    1   0.079997
    2   0.097636
    3   0.113011
    4   0.057210
    """
    ## Checking if file exists
    if not os.path.exists(filename):
        msg = '`filename`: {0} NOT FOUND! Exiting..'.format(filename)
        raise ValueError(msg)
    ## Reading file
    if catl_format=='.hdf5':
        mock_pd = pd.read_hdf(filename)
    else:
        msg = '`catl_format` ({0}) not supported! Exiting...'.format(catl_format)
        raise ValueError(msg)

    return mock_pd

def get_err_data(survey, path):
    """
    Calculate error in data SMF from mocks

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    err_total: array
        Standard deviation of phi values between all mocks and for all galaxies
    err_red: array
        Standard deviation of phi values between all mocks and for red galaxies
    err_blue: array
        Standard deviation of phi values between all mocks and for blue galaxies
    """

    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

    phi_arr_total = []
    phi_arr_red = []
    phi_arr_blue = []
    # logmstar_red_max_arr = []
    # logmstar_blue_max_arr = []
    # colour_err_arr = []
    # colour_corr_mat_inv = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                (mock_pd.logmstar.values >= mstar_limit)]

            Mstar_q = 10.5 # Msun/h
            Mh_q = 13.76 # Msun/h
            mu = 0.69
            nu = 0.15

            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values

            #Measure SMF of mock using diff_smf function
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(logmstar_arr, volume, False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
                volume, False, 'R')
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
                volume, False, 'B')
            phi_arr_total.append(phi_total)
            phi_arr_red.append(phi_red)
            phi_arr_blue.append(phi_blue)

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)

    # phi_arr_colour = np.append(phi_arr_red, phi_arr_blue, axis = 0)

    # Covariance matrix for total phi (all galaxies)
    # cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    # err_total = np.sqrt(cov_mat.diagonal())
    # cov_mat_red = np.cov(phi_arr_red, rowvar=False) # default norm is N-1
    # err_red = np.sqrt(cov_mat_red.diagonal())
    # colour_err_arr.append(err_red)
    # cov_mat_blue = np.cov(phi_arr_blue, rowvar=False) # default norm is N-1
    # err_blue = np.sqrt(cov_mat_blue.diagonal())
    # colour_err_arr.append(err_blue)

    # corr_mat_red = cov_mat_red / np.outer(err_red , err_red)
    # corr_mat_inv_red = np.linalg.inv(corr_mat_red)
    # colour_corr_mat_inv.append(corr_mat_inv_red)
    # corr_mat_blue = cov_mat_blue / np.outer(err_blue , err_blue)
    # corr_mat_inv_blue = np.linalg.inv(corr_mat_blue)
    # colour_corr_mat_inv.append(corr_mat_inv_blue)

    deltav_sig_red, deltav_sig_blue, deltav_sig_cen_red, deltav_sig_cen_blue = \
        get_deltav_sigma_mocks_qmcolour(survey, path)

    phi_red_0 = phi_arr_red[:,0]
    phi_red_1 = phi_arr_red[:,1]
    phi_red_2 = phi_arr_red[:,2]
    phi_red_3 = phi_arr_red[:,3]
    phi_red_4 = phi_arr_red[:,4]

    phi_blue_0 = phi_arr_blue[:,0]
    phi_blue_1 = phi_arr_blue[:,1]
    phi_blue_2 = phi_arr_blue[:,2]
    phi_blue_3 = phi_arr_blue[:,3]
    phi_blue_4 = phi_arr_blue[:,4]

    dv_red_0 = deltav_sig_red[:,0]
    dv_red_1 = deltav_sig_red[:,1]
    dv_red_2 = deltav_sig_red[:,2]
    dv_red_3 = deltav_sig_red[:,3]
    dv_red_4 = deltav_sig_red[:,4]

    dv_blue_0 = deltav_sig_blue[:,0]
    dv_blue_1 = deltav_sig_blue[:,1]
    dv_blue_2 = deltav_sig_blue[:,2]
    dv_blue_3 = deltav_sig_blue[:,3]
    dv_blue_4 = deltav_sig_blue[:,4]

    combined_df = pd.DataFrame({'phi_red_0':phi_red_0, 'phi_red_1':phi_red_1,\
        'phi_red_2':phi_red_2, 'phi_red_3':phi_red_3, 'phi_red_4':phi_red_4, \
        'phi_blue_0':phi_blue_0, 'phi_blue_1':phi_blue_1, 
        'phi_blue_2':phi_blue_2, 'phi_blue_3':phi_blue_3, 
        'phi_blue_4':phi_blue_4, \
        'dv_red_0':dv_red_0, 'dv_red_1':dv_red_1, 'dv_red_2':dv_red_2, \
        'dv_red_3':dv_red_3, 'dv_red_4':dv_red_4, \
        'dv_blue_0':dv_blue_0, 'dv_blue_1':dv_blue_1, 'dv_blue_2':dv_blue_2, \
        'dv_blue_3':dv_blue_3, 'dv_blue_4':dv_blue_4})

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    # deltav_sig_colour = np.append(deltav_sig_red, deltav_sig_blue, axis = 0)
    # cov_mat_colour = np.cov(phi_arr_colour,deltav_sig_colour, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    # cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    return err_colour, corr_mat_inv_colour

def debug_within_outside_1sig(emcee_table, grp_keys, Mstar_q, Mhalo_q, mu, nu, chi2):
    zumandelbaum_param_vals = [10.5, 13.76, 0.69, 0.15]
    iteration = 600.0
    emcee_table_it600 = emcee_table.loc[emcee_table.iteration_id == iteration]  

    chi2_std_it600 = np.std(emcee_table_it600.chi2)
    chi2_mean_it600 = np.mean(emcee_table_it600.chi2)
    # selecting value from within one sigma                                                              
    df_within_sig = emcee_table_it600.loc[(emcee_table_it600.chi2 < chi2_mean_it600 + chi2_std_it600)&(emcee_table_it600.chi2 > chi2_mean_it600 - chi2_std_it600)]
    chi2_within_sig = df_within_sig.chi2.values[3]
    mstar_within_sig = df_within_sig.Mstar_q.values[3]
    mhalo_within_sig = df_within_sig.Mhalo_q.values[3]
    mu_within_sig = df_within_sig.mu.values[3]
    nu_within_sig = df_within_sig.nu.values[3]
    # # selecting value from outside one sigma
    df_outside_sig = emcee_table_it600.loc[emcee_table_it600.chi2 > chi2_mean_it600 + chi2_std_it600]
    chi2_outside_sig = df_outside_sig.chi2.values[3]
    mstar_outside_sig = df_outside_sig.Mstar_q.values[3]
    mhalo_outside_sig = df_outside_sig.Mhalo_q.values[3]
    mu_outside_sig = df_outside_sig.mu.values[3]
    nu_outside_sig = df_outside_sig.nu.values[3]

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, \
        figsize=(10,10))
    ax1.plot(grp_keys, Mstar_q[0],c='#941266',ls='--', marker='o')
    ax1.axhline(zumandelbaum_param_vals[0],color='lightgray')
    ax1.scatter(iteration, mstar_outside_sig, marker='*', c='k', s=70)
    ax1.scatter(iteration, mstar_within_sig, marker='o', c='k', s=70)
    ax2.plot(grp_keys, Mhalo_q[0], c='#941266',ls='--', marker='o')
    ax2.axhline(zumandelbaum_param_vals[1],color='lightgray')
    ax2.scatter(iteration, mhalo_outside_sig, marker='*', c='k', s=70)
    ax2.scatter(iteration, mhalo_within_sig, marker='o', c='k', s=70)
    ax3.plot(grp_keys, mu[0], c='#941266',ls='--', marker='o')
    ax3.axhline(zumandelbaum_param_vals[2],color='lightgray')
    ax3.scatter(iteration, mu_outside_sig, marker='*', c='k', s=70)
    ax3.scatter(iteration, mu_within_sig, marker='o', c='k', s=70)
    ax4.plot(grp_keys, nu[0], c='#941266',ls='--', marker='o')
    ax4.axhline(zumandelbaum_param_vals[3],color='lightgray')
    ax4.scatter(iteration, nu_outside_sig, marker='*', c='k', s=70)
    ax4.scatter(iteration, nu_within_sig, marker='o', c='k', s=70) 
    ax5.plot(grp_keys, chi2[0], c='#941266',ls='--', marker='o')
    ax5.scatter(iteration, chi2_outside_sig, marker='*', c='k', s=70)
    ax5.scatter(iteration, chi2_within_sig, marker='o', c='k', s=70)

    ax1.fill_between(grp_keys, Mstar_q[0]-Mstar_q[1], Mstar_q[0]+Mstar_q[1], 
        alpha=0.3, color='#941266')
    ax2.fill_between(grp_keys, Mhalo_q[0]-Mhalo_q[1], Mhalo_q[0]+Mhalo_q[1], \
        alpha=0.3, color='#941266')
    ax3.fill_between(grp_keys, mu[0]-mu[1], mu[0]+mu[1], \
        alpha=0.3, color='#941266')
    ax4.fill_between(grp_keys, nu[0]-nu[1], nu[0]+nu[1], \
        alpha=0.3, color='#941266')
    ax5.fill_between(grp_keys, chi2[0]-chi2[1], chi2[0]+chi2[1], \
        alpha=0.3, color='#941266')

    ax1.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{*}}$")
    ax2.set_ylabel(r"$\mathbf{log_{10}\ M^{q}_{h}}$")
    ax3.set_ylabel(r"$\boldsymbol{\mu}$")
    # ax4.set_ylabel(r"$\mathbf{log_{10}} \boldsymbol{\ \nu}$")
    ax4.set_ylabel(r"$\boldsymbol{\nu}$")
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
    ax4.annotate(zumandelbaum_param_vals[3], (0.95,0.85), xycoords='axes fraction', 
        bbox=dict(boxstyle="square", ec='k', fc='lightgray', alpha=0.5), size=10)
    plt.xlabel(r"$\mathbf{iteration\ number}$")
    plt.show()

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_interim = dict_of_paths['int_dir']
    path_to_figures = dict_of_paths['plot_dir']
    path_to_data = dict_of_paths['data_dir']

    catl_file = path_to_raw + "eco/eco_all.csv"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    randint_logmstar_file = pd.read_csv("/Users/asadm2/Desktop/randint_logmstar.txt", 
        header=None)
    mock_num = randint_logmstar_file[0].values[int(iteration)-1]
    gals_df_ = reading_catls(path_to_proc + "gal_group.hdf5")
    theta_within = [mstar_within_sig, mhalo_within_sig, mu_within_sig, nu_within_sig]
    f_red_cen, f_red_sat = hybrid_quenching_model(theta_within, gals_df_, 'vishnu', \
        mock_num)
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df_)
    v_sim = 130**3
    total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
    , False, mock_num)  
    sig_red_within, sig_blue_within, cen_red_within, cen_blue_within = \
        get_deltav_sigma_vishnu_qmcolour(gals_df, mock_num)
    total_model_within, red_model_within, blue_model_within = total_model, \
        red_model, blue_model 

    theta_outside = [mstar_outside_sig, mhalo_outside_sig, mu_outside_sig, \
        nu_outside_sig]
    f_red_cen, f_red_sat = hybrid_quenching_model(theta_outside, gals_df_, 'vishnu', \
        mock_num)
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df_)
    v_sim = 130**3
    total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
    , False, mock_num)  
    sig_red_outside, sig_blue_outside, cen_red_outside, cen_blue_outside = \
        get_deltav_sigma_vishnu_qmcolour(gals_df, mock_num)
    total_model_outside, red_model_outside, blue_model_outside = total_model, \
        red_model, blue_model 

    catl, volume, z_median = read_data_catl(catl_file, survey)
    catl = assign_colour_label_data(catl)
    total_data, red_data, blue_data = measure_all_smf(catl, volume, True)
    std_red, centers_red, std_blue, centers_blue = get_deltav_sigma_data(catl)

    sigma, corr_mat_inv = get_err_data(survey, path_to_mocks)

    plt.clf()
    plt.plot(total_model_within[0], total_model_within[1], c='k', linestyle='-', \
        label='total within 1sig')
    plt.plot(total_model_outside[0], total_model_outside[1], c='k', linestyle='--',\
        label='total outside 1sig')
    plt.plot(red_model_within[0], red_model_within[1], color='maroon', 
        linestyle='--', label='within 1sig')
    plt.plot(blue_model_within[0], blue_model_within[1], color='mediumblue', 
        linestyle='--', label='within 1sig')
    plt.plot(red_model_outside[0], red_model_outside[1], color='indianred', 
        linestyle='--', label='outside 1sig')
    plt.plot(blue_model_outside[0], blue_model_outside[1], color='cornflowerblue', 
        linestyle='--', label='outside 1sig')
    plt.errorbar(x=red_data[0], y=red_data[1], yerr=sigma[0:5], xerr=None, 
        color='r', label='data')
    plt.errorbar(x=blue_data[0], y=blue_data[1], yerr=sigma[5:10], xerr=None, 
        color='b', label='data')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
    plt.legend(loc='best')
    plt.title('ECO SMF')
    plt.show()

    plt.clf()
    plt.plot(max_total, phi_total, c='k')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
    plt.legend(loc='best')
    plt.title('ECO SMF')
    plt.show()

    plt.clf()
    plt.scatter(cen_red_within, sig_red_within, c='maroon', label='within 1sig')
    plt.scatter(cen_red_outside, sig_red_outside, c='indianred', label='outside 1sig')
    plt.scatter(cen_blue_within, sig_blue_within, c='mediumblue', label='within 1sig')
    plt.scatter(cen_blue_outside, sig_blue_outside, c='cornflowerblue', \
        label='outside 1sig')
    plt.errorbar(x=centers_red, y=std_red, yerr=sigma[10:15], xerr=None, color='r',\
        label='data', fmt='')
    plt.errorbar(x=centers_blue, y=std_blue, yerr=sigma[15:20], xerr=None, \
        color='b', label='data', fmt='')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    plt.ylabel(r'$\sigma$')
    plt.legend(loc='best')
    plt.title(r'ECO spread in $\delta v$')
    plt.show()
