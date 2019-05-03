"""
{This script plots SMF and SMHM from results of the mcmc including best fit and
 68th percentile of lowest chi-squared values. This is compared to data and is
 done for all 3 surveys: ECO, RESOLVE-A and RESOLVE-B.}
"""

# Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import argparse
import random
import math
import time

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=10)
rc('text', usetex=True)

def read_chi2(path_to_file):
    """
    Reads chi-squared values from file

    Parameters
    ----------
    path_to_file: string
        Path to chi-squared values file

    Returns
    ---------
    chi2: array
        Array of reshaped chi^2 values to match chain values
    """
    chi2_df = pd.read_csv(path_to_file,header=None,names=['chisquared'])

    # Needed to reshape since flattened along wrong axis, 
    # didn't correspond to chain
    if survey == 'resolvea' or survey == 'resolveb':
        test_reshape = chi2_df.chisquared.values.reshape((1000,250))
    if survey == 'eco':
        test_reshape = chi2_df.chisquared.values.reshape((500,250))
    chi2 = np.ndarray.flatten(np.array(test_reshape),'F')

    return chi2

def read_mcmc(path_to_file):
    """
    Reads mcmc chain from file

    Parameters
    ----------
    path_to_file: string
        Path to mcmc chain file

    Returns
    ---------
    emcee_table: pandas dataframe
        Dataframe of mcmc chain values with NANs removed
    """
    colnames = ['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',\
        'scatter']
    emcee_table = pd.read_csv(path_to_file,names=colnames,sep='\s+',\
        dtype=np.float64)

    # Cases where last parameter was a NaN and its value was being written to 
    # the first element of the next line followed by 4 NaNs for the other 
    # parameters
    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            row[4] = scatter_val
    
    # Cases where rows of NANs appear
    emcee_table = emcee_table.dropna().reset_index(drop=True)
    
    return emcee_table

def read_catl(path_to_file,survey):
    """
    Reads survey catalog from file

    Parameters
    ----------
    path_to_file: string
        Path to survey catalog file

    survey: string
        Name of survey

    Returns
    ---------
    catl: pandas dataframe
        Survey catalog with grpcz, abs rmag and stellar mass limits
    
    volume: float
        Volume of survey

    cvar: float
        Cosmic variance of survey

    z_median: float
        Median redshift of survey
    """
    if survey == 'eco':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
            (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &\
                (eco_buff.logmstar.values >= 8.9)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if survey == 'resolvea':
            catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
                (resolve_live18.grpcz.values > 4500) & \
                    (resolve_live18.grpcz.values < 7000) & \
                        (resolve_live18.absrmag.values < -17.33) & \
                            (resolve_live18.logmstar.values >= 8.9)]


            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            # 487 - cz, 369 - grpcz
            catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
                (resolve_live18.grpcz.values > 4500) & \
                    (resolve_live18.grpcz.values < 7000) & \
                        (resolve_live18.absrmag.values < -17) & \
                            (resolve_live18.logmstar.values >= 8.7)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def get_paramvals_percentile(mcmc_table, pctl, chi2):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 1000 sample

    Parameters
    ----------
    mcmc_table: pandas dataframe
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2: array
        Array of chi^2 values

    Returns
    ---------
    mcmc_table_pctl: pandas dataframe
        Random 1000 sample of 68th percentile lowest chi^2 values
    """ 
    pctl = pctl/100
    mcmc_table['chi2'] = chi2
    mcmc_table = mcmc_table.sort_values('chi2')
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(1000)

    return mcmc_table_pctl

def diff_smf(mstar_arr, volume, cvar_err, h1_bool):
    """
    Calculates differential stellar mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

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
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 1.429)
    else:
        logmstar_arr = np.log10(mstar_arr)
    if survey == 'eco' or survey == 'resolvea':
        bins = np.linspace(8.9, 11.8, 12)
    elif survey == 'resolveb':
        bins = np.linspace(8.7, 11.8, 12)
    # Unnormalized histogram and bin edges
    phi, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(phi) / (volume * dm)
    err_cvar = cvar_err / (volume * dm)
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)

    phi = phi / (volume * dm)  # not a log quantity

    return maxis, phi, err_tot, bins

def get_centrals_mock(gals_df):
    """
    Get centrals from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses
    """
    C_S = []
    for idx in range(len(gals_df)):
        if gals_df['halo_hostid'][idx] == gals_df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)
    
    C_S = np.array(C_S)
    gals_df['C_S'] = C_S
    cen_gals = []
    cen_halos = []

    for idx,value in enumerate(gals_df['C_S']):
        if value == 1:
            cen_gals.append(gals_df['stellar_mass'][idx])
            cen_halos.append(gals_df['halo_mvir'][idx])

    cen_gals = np.log10(np.array(cen_gals))
    cen_halos = np.log10(np.array(cen_halos))

    return cen_gals, cen_halos

def get_centrals_data(catl):
    """
    Get centrals from survey catalog

    Parameters
    ----------
    catl: pandas dataframe
        Survey catalog

    Returns
    ---------
    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses
    """ 
    cen_gals = []
    cen_halos = []
    for idx,val in enumerate(catl.fc.values):
        if val == 1:
            stellar_mass_h07 = catl.logmstar.values[idx]
            stellar_mass_h1 = np.log10((10**stellar_mass_h07) / 1.429)
            halo_mass_h07 = catl.logmh_s.values[idx]
            halo_mass_h1 = np.log10((10**halo_mass_h07) / 1.429)
            cen_gals.append(stellar_mass_h1)
            cen_halos.append(halo_mass_h1)

    cen_gals = np.array(cen_gals)
    cen_halos = np.array(cen_halos)

    return cen_gals, cen_halos

def halocat_init(halo_catalog,z_median):
    """
    Initial population of halo catalog using populate_mock function

    Parameters
    ----------
    halo_catalog: string
        Path to halo catalog
    
    z_median: float
        Median redshift of survey

    Returns
    ---------
    model: halotools model instance
        Model based on behroozi 2010 SMHM
    """
    halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_median, \
        prim_haloprop_key='halo_macc')
    model.populate_mock(halocat,seed=5)

    return model

def populate_mock(theta):
    """
    Populate mock based on five parameter values 

    Parameters
    ----------
    theta: array
        Array of parameter values

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model_init.param_dict['smhm_m1_0'] = mhalo_characteristic
    model_init.param_dict['smhm_m0_0'] = mstellar_characteristic
    model_init.param_dict['smhm_beta_0'] = mlow_slope
    model_init.param_dict['smhm_delta_0'] = mhigh_slope
    model_init.param_dict['scatter_model_param1'] = mstellar_scatter

    model_init.mock.populate()

    if survey == 'eco' or survey == 'resolvea':
        sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**8.9
    elif survey == 'resolveb':
        sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model_init.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def mp_func(a_list):
    """
    Populate mock based on five parameter values

    Parameters
    ----------
    a_list: multidimensional array
        Array of five parameter values

    Returns
    ---------
    max_model_arr: array
        Array of x-axis mass values

    phi_model_arr: array
        Array of y-axis values

    err_tot_model_arr: array
        Array of error values per bin

    cen_gals_arr: array
        Array of central galaxy masses

    cen_halos_arr: array
        Array of central halo masses
    """
    v_sim = 130**3

    maxis_arr = []
    phi_arr = []
    err_tot_arr = []
    cen_gals_arr = []
    cen_halos_arr = []

    for theta in a_list:  
        gals_df = populate_mock(theta)
        mstellar_mock = gals_df.stellar_mass.values 
        maxis, phi, err_tot, bins = diff_smf(mstellar_mock, v_sim, 0, True)
        cen_gals, cen_halos = get_centrals_mock(gals_df)

        maxis_arr.append(maxis)
        phi_arr.append(phi)
        err_tot_arr.append(err_tot)
        cen_gals_arr.append(cen_gals)
        cen_halos_arr.append(cen_halos)

    return [maxis_arr, phi_arr, err_tot_arr, cen_gals_arr, cen_halos_arr]

def mp_init(mcmc_table_pctl,nproc):
    """
    Initializes multiprocessing of mocks and smf and smhm measurements

    Parameters
    ----------
    mcmc_table_pctl: pandas dataframe
        Mcmc chain dataframe of 1000 random samples

    nproc: int
        Number of processes to use in multiprocessing

    Returns
    ---------
    result: multidimensional array
        Array of smf and smhm data
    """
    start = time.time()
    chunks = np.array([mcmc_table_pctl.iloc[:,:5].values[i::5] \
        for i in range(5)])
    pool = Pool(processes=nproc)
    result = pool.map(mp_func, chunks)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return result

def get_best_fit_model(survey):
    """
    Get SMF and SMHM information of best fit model given a survey

    Parameters
    ----------
    survey: string
        Name of survey

    Returns
    ---------
    max_model: array
        Array of x-axis mass values

    phi_model: array
        Array of y-axis values

    err_tot_model: array
        Array of error values per bin

    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses
    """
    if survey == 'eco':
        best_fit_params = [12.261,10.662,0.4,0.6,0.37]
    elif survey == 'resolvea':
        best_fit_params = [12.2,10.61,0.384,0.42,0.39]
    elif survey == 'resolveb':
        best_fit_params = [12.25,10.96,0.453,0.66,0.19]   
    v_sim = 130**3
    gals_df = populate_mock(best_fit_params)
    mstellar_mock = gals_df.stellar_mass.values  # Read stellar masses
    max_model, phi_model, err_tot_model, bins_model =\
        diff_smf(mstellar_mock, v_sim, 0, True)
    
    cen_gals, cen_halos = get_centrals_mock(gals_df)

    return max_model, phi_model, err_tot_model, cen_gals, cen_halos

def plot_smf(result, max_model_bf, phi_model_bf, err_tot_model_bf, maxis_data, \
    phi_data, err_data):
    """
    Plot SMF from data, best fit param values and param values corresponding to 
    68th percentile 1000 lowest chi^2 values

    Parameters
    ----------
    result: multidimensional array
        Array of SMF and SMHM information
    
    max_model_bf: array
        Array of x-axis mass values for best fit SMF

    phi_model_bf: array
        Array of y-axis values for best fit SMF
    
    err_tot_model_bf: array
        Array of error values per bin of best fit SMF

    maxis_data: array
        Array of x-axis mass values for data SMF

    phi_data: array
        Array of y-axis values for data SMF

    err_data: array
        Array of error values per bin of data SMF

    Returns
    ---------
    Nothing; SMF plot is saved in figures repository
    """
    if survey == 'resolvea':
        line_label = 'RESOLVE-A'
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
    elif survey == 'eco':
        line_label = 'ECO'

    fig1 = plt.figure(figsize=(10,10))
    plt.errorbar(maxis_data,phi_data,yerr=err_data,color='k',fmt='--s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='{0}'.format(line_label),\
            zorder=10)
    for idx in range(len(result[0][0])):
        plt.plot(result[0][0][idx],result[0][1][idx],color='lightgray',\
            linestyle='-',alpha=0.5,zorder=0,label='model')
    for idx in range(len(result[1][0])):
        plt.plot(result[1][0][idx],result[1][1][idx],color='lightgray',\
            linestyle='-',alpha=0.5,zorder=1)
    for idx in range(len(result[2][0])):
        plt.plot(result[2][0][idx],result[2][1][idx],color='lightgray',\
            linestyle='-',alpha=0.5,zorder=2)
    for idx in range(len(result[3][0])):
        plt.plot(result[3][0][idx],result[3][1][idx],color='lightgray',\
            linestyle='-',alpha=0.5,zorder=3)
    for idx in range(len(result[4][0])):
        plt.plot(result[4][0][idx],result[4][1][idx],color='lightgray',\
            linestyle='-',alpha=0.5,zorder=4)
    plt.errorbar(max_model_bf,phi_model_bf,yerr=err_tot_model_bf,\
        color='r',fmt='--s',ecolor='r',markersize=4,capsize=5,\
            capthick=0.5,label='best fit',zorder=10)
    plt.yscale('log')
    plt.ylim(10**-5,10**-1)
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=15)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
    plt.savefig(path_to_figures + 'smf_emcee_{0}.png'.format(survey))

def plot_smhm(result, gals_bf, halos_bf, gals_data, halos_data, gals_b10, \
    halos_b10):
    """
    Plot SMHM from data, best fit param values, param values corresponding to 
    68th percentile 1000 lowest chi^2 values and behroozi 2010 param values

    Parameters
    ----------
    result: multidimensional array
        Array of central galaxy and halo masses
    
    gals_bf: array
        Array of y-axis stellar mass values for best fit SMHM

    halos_bf: array
        Array of x-axis halo mass values for best fit SMHM
    
    gals_data: array
        Array of y-axis stellar mass values for data SMF

    halos_data: array
        Array of x-axis halo mass values for data SMF

    gals_b10: array
        Array of y-axis stellar mass values for behroozi 2010 SMHM

    halos_b10: array
        Array of x-axis halo mass values for behroozi 2010 SMHM

    Returns
    ---------
    Nothing; SMHM plot is saved in figures repository
    """
    if survey == 'resolvea':
        line_label = 'RESOLVE-A'
        cvar = 0.3
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
        cvar = 0.58
    elif survey == 'eco':
        line_label = 'ECO'
        cvar = 0.125
    
    x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(halos_bf,\
    gals_bf,base=0.4,bin_statval='center')
    x_b10,y_b10,y_std_b10,y_std_err_b10 = Stats_one_arr(halos_b10,\
        gals_b10,base=0.4,bin_statval='center')
    x_data,y_data,y_std_data,y_std_err_data = Stats_one_arr(halos_data,\
        gals_data,base=0.4,bin_statval='center')
    y_std_err_data = np.sqrt(y_std_err_data**2 + cvar**2)

    fig1 = plt.figure(figsize=(10,10))
    plt.errorbar(x_data,y_data,yerr=y_std_err_data,color='k',fmt='-s',\
        ecolor='k',markersize=4,capsize=5,capthick=0.5,\
            label='{0}'.format(line_label),zorder=10)

    plt.errorbar(x_b10,y_b10,yerr=y_std_err_b10,color='k',fmt='--s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='Behroozi10',zorder=10)

    for idx in range(len(result[0][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[0][4][idx],result[0][3][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[1][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[1][4][idx],result[1][3][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=1)
    for idx in range(len(result[2][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[2][4][idx],result[2][3][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=2)
    for idx in range(len(result[3][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[3][4][idx],result[3][3][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=3)
    for idx in range(len(result[4][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[4][4][idx],result[4][3][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=4)

    plt.errorbar(x_bf,y_bf,yerr=y_std_err_bf,color='r',fmt='-s',ecolor='r',\
        markersize=4,capsize=5,capthick=0.5,label='Best fit',zorder=10)

    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(8.9,)
    elif survey == 'resolveb':
        plt.ylim(8.7,)
    plt.xlim(10,)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=15)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
    plt.savefig(path_to_figures + 'smhm_emcee_{0}.png'.format(survey))

def args_parser():
    """
    Parsing arguments passed to populate_mock.py script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('machine', type=str, \
        help='Options: mac/bender')
    parser.add_argument('survey', type=str, \
        help='Options: eco/resolvea/resolveb')
    parser.add_argument('nproc', type=int, help='Number of processes',\
        default=1)
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function that calls all other functions
    
    Parameters
    ----------
    args: 
        Input arguments to the script

    """
    global model_init
    global survey
    global path_to_figures

    survey = args.survey
    machine = args.machine
    nproc = args.nproc

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_interim = dict_of_paths['int_dir']
    path_to_figures = dict_of_paths['plot_dir']

    path_to_proc = path_to_proc + 'smhm/'
    
    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    chi2_file = path_to_proc + '{0}_chi2.txt'.format(survey)
    chain_file = path_to_proc + 'mcmc_{0}.dat'.format(survey)
    if survey == 'eco':
        catl_file = path_to_raw + "eco_all.csv"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    print('Reading chi-squared file')
    chi2 = read_chi2(chi2_file)
    print('Reading mcmc chain file')
    mcmc_table = read_mcmc(chain_file)
    print('Reading catalog')
    catl, volume, cvar, z_median = read_catl(catl_file, survey)
    print('Getting data in specific percentile')
    mcmc_table_pctl = get_paramvals_percentile(mcmc_table, 68, chi2)

    print('Retrieving stellar mass from catalog')
    stellar_mass_arr = catl.logmstar.values
    maxis_data, phi_data, err_data, bins_data = \
        diff_smf(stellar_mass_arr, volume, cvar, False)
    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Retrieving Behroozi 2010 centrals')
    model_init.mock.populate()
    gals_b10 = model_init.mock.galaxy_table
    cen_gals_b10, cen_halos_b10 = get_centrals_mock(gals_b10)

    print('Retrieving survey centrals')
    cen_gals_data, cen_halos_data = get_centrals_data(catl)

    print('Multiprocessing')
    result = mp_init(mcmc_table_pctl, nproc)
    print('Getting best fit model and centrals')
    maxis_bf, phi_bf, err_tot_bf, cen_gals_bf, cen_halos_bf = \
        get_best_fit_model(survey)

    print('Plotting SMF')
    plot_smf(result, maxis_bf, phi_bf, err_tot_bf, maxis_data, phi_data, \
        err_data)

    print('Plotting SMHM')
    plot_smhm(result, cen_gals_bf, cen_halos_bf, cen_gals_data, cen_halos_data, \
        cen_gals_b10, cen_halos_b10)    

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args) 
