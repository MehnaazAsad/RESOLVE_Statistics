"""
{This script plots SMF and SMHM from results of the mcmc including best fit and
 68th percentile of lowest chi-squared values. This is compared to data and is
 done for all 3 surveys: ECO, RESOLVE-A and RESOLVE-B.}
"""

# Matplotlib backend
# import matplotlib
# matplotlib.use('Agg')

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
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

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
    chi2 = chi2_df.chisquared.values

    return chi2

def reading_mock_catl(filename, catl_format='.hdf5'):
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
    colnames = ['mstar_q','mh_q','mu','nu']
    
    emcee_table = pd.read_csv(path_to_file, names=colnames, 
        delim_whitespace=True, header=None)

    emcee_table = emcee_table[emcee_table.mstar_q.values != '#']
    emcee_table.mstar_q = emcee_table.mstar_q.astype(np.float64)
    emcee_table.mh_q = emcee_table.mh_q.astype(np.float64)
    emcee_table.mu = emcee_table.mu.astype(np.float64)
    emcee_table.nu = emcee_table.nu.astype(np.float64)
    
    return emcee_table

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

    cvar: `float`
        Cosmic variance of survey

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0)

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
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33) & 
                    (resolve_live18.logmstar.values >= 8.9)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            if mf_type == 'smf':
                # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17) & 
                    (resolve_live18.logmstar.values >= 8.7)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def get_paramvals_percentile(mcmc_table, pctl, chi2):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 100 sample

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
        Sample of 100 68th percentile lowest chi^2 values
    """ 
    pctl = pctl/100
    mcmc_table['chi2'] = chi2
    mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:4]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][4]
    # Randomly sample 1000 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(1000)

    return mcmc_table_pctl, bf_params, bf_chi2

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

def hybrid_quenching_model(theta, gals_df):
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

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df)
    cen_stellar_mass_arr, sat_stellar_mass_arr = get_stellar_mock(gals_df)

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/10**(Mh_q))**nu))
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
    df.loc[df['C_S'] == 1, 'f_red'] = f_red_cen
    df.loc[df['C_S'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['C_S']):
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
    ##
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

def halocat_init(halo_catalog, z_median):
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

def populate_mock(theta, model):
    """
    Populate mock based on five SMHM parameter values and model

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    model: halotools model instance
        Model based on behroozi 2010 SMHM

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    """"""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.9) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.4) / 2.041), 1)
    elif survey == 'resolveb':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.7) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.1) / 2.041), 1)
    sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def assign_cen_sat_flag(gals_df):
    """
    Assign centrals and satellites flag to dataframe

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    gals_df: pandas dataframe
        Mock catalog with centrals/satellites flag as new column
    """

    C_S = []
    for idx in range(len(gals_df)):
        if gals_df['halo_hostid'][idx] == gals_df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)
    
    C_S = np.array(C_S)
    gals_df['C_S'] = C_S
    return gals_df

def get_host_halo_mock(gals_df):
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

    cen_halos = []
    sat_halos = []
    for idx,value in enumerate(df['C_S']):
        if value == 1:
            cen_halos.append(df['halo_mvir_host_halo'][idx])
        elif value == 0:
            sat_halos.append(df['halo_mvir_host_halo'][idx])

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(gals_df):
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

    cen_gals = []
    sat_gals = []
    for idx,value in enumerate(df['C_S']):
        if value == 1:
            cen_gals.append(df['stellar_mass'][idx])
        elif value == 0:
            sat_gals.append(df['stellar_mass'][idx])

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

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
    cen_gals_red = []
    cen_halos_red = []
    cen_gals_blue = []
    cen_halos_blue = []

    for idx,value in enumerate(gals_df['C_S']):
        if value == 1:
            if gals_df['colour_label'][idx] == 'R':
                cen_gals_red.append(gals_df['stellar_mass'][idx])
                cen_halos_red.append(gals_df['halo_mvir'][idx])
            elif gals_df['colour_label'][idx] == 'B':
                cen_gals_blue.append(gals_df['stellar_mass'][idx])
                cen_halos_blue.append(gals_df['halo_mvir'][idx])

    cen_gals_red = np.log10(np.array(cen_gals_red))
    cen_halos_red = np.log10(np.array(cen_halos_red))
    cen_gals_blue = np.log10(np.array(cen_gals_blue))
    cen_halos_blue = np.log10(np.array(cen_halos_blue))

    return cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue

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
    # cen_gals = []
    # cen_halos = []
    # for idx,val in enumerate(catl.fc.values):
    #     if val == 1:
    #         stellar_mass_h07 = catl.logmstar.values[idx]
    #         stellar_mass_h1 = np.log10((10**stellar_mass_h07) / 2.041)
    #         halo_mass_h07 = catl.logmh_s.values[idx]
    #         halo_mass_h1 = np.log10((10**halo_mass_h07) / 2.041)
    #         cen_gals.append(stellar_mass_h1)
    #         cen_halos.append(halo_mass_h1)
    
    if mf_type == 'smf':
        cen_gals_red = np.log10(10**catl.logmstar.loc[(catl.fc.values == 1)&
            (catl.colour_label.values == 'R')]/2.041)
        cen_halos_red = np.log10(10**catl.logmh_s.loc[(catl.fc.values == 1)&
            (catl.colour_label.values == 'R')]/2.041)
        cen_gals_blue = np.log10(10**catl.logmstar.loc[(catl.fc.values == 1)&
            (catl.colour_label.values == 'B')]/2.041)
        cen_halos_blue = np.log10(10**catl.logmh_s.loc[(catl.fc.values == 1)&
            (catl.colour_label.values == 'B')]/2.041)
    # elif mf_type == 'bmf':
    #     logmstar = catl.logmstar.loc[catl.fc.values == 1]
    #     logmgas = catl.logmgas.loc[catl.fc.values == 1]
    #     logmbary = calc_bary(logmstar, logmgas)
    #     catl['logmbary'] = logmbary
    #     if survey == 'eco' or survey == 'resolvea':
    #         limit = 9.4
    #     elif survey == 'resolveb':
    #         limit = 9.1
    #     cen_gals = np.log10((10**(catl.logmbary.loc[(catl.fc.values == 1) & 
    #         (catl.logmbary.values >= limit)]))/2.041)
    #     cen_halos = np.log10((10**(catl.logmh_s.loc[(catl.fc.values == 1) & 
    #         (catl.logmbary.values >= limit)]))/2.041)

    return cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue

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
            mock_pd = reading_mock_catl(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                (mock_pd.logmstar.values >= mstar_limit)]

            logmstar_arr = mock_pd.logmstar.values 
            u_r_arr = mock_pd.u_r.values

            colour_label_arr = np.empty(len(mock_pd), dtype='str')
            for idx, value in enumerate(logmstar_arr):

                if value <= 9.1:
                    if u_r_arr[idx] > 1.457:
                        colour_label = 'R'
                    else:
                        colour_label = 'B'

                elif value > 9.1 and value < 10.1:
                    divider = 0.24 * value - 0.7
                    if u_r_arr[idx] > divider:
                        colour_label = 'R'
                    else:
                        colour_label = 'B'

                elif value >= 10.1:
                    if u_r_arr[idx] > 1.7:
                        colour_label = 'R'
                    else:
                        colour_label = 'B'
                    
                colour_label_arr[idx] = colour_label

            mock_pd['colour_label'] = colour_label_arr

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

    # Covariance matrix for total phi (all galaxies)
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    err_total = np.sqrt(cov_mat.diagonal())
    cov_mat_red = np.cov(phi_arr_red, rowvar=False) # default norm is N-1
    err_red = np.sqrt(cov_mat_red.diagonal())
    # colour_err_arr.append(err_red)
    cov_mat_blue = np.cov(phi_arr_blue, rowvar=False) # default norm is N-1
    err_blue = np.sqrt(cov_mat_blue.diagonal())
    # colour_err_arr.append(err_blue)

    # corr_mat_red = cov_mat_red / np.outer(err_red , err_red)
    # corr_mat_inv_red = np.linalg.inv(corr_mat_red)
    # colour_corr_mat_inv.append(corr_mat_inv_red)
    # corr_mat_blue = cov_mat_blue / np.outer(err_blue , err_blue)
    # corr_mat_inv_blue = np.linalg.inv(corr_mat_blue)
    # colour_corr_mat_inv.append(corr_mat_inv_blue)

    # cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    return err_total, err_red, err_blue

def get_best_fit_model(best_fit_params):
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
    v_sim = 130**3

    f_red_cen, f_red_sat = hybrid_quenching_model(best_fit_params, gals_df_)
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df_)
    v_sim = 130**3
    total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
    , False)     
    cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue = \
        get_centrals_mock(gals_df)

    max_red = red_model[0]
    phi_red = red_model[1]
    max_blue = blue_model[0]
    phi_blue = blue_model[1]

    return max_red, phi_red, max_blue, phi_blue, cen_gals_red, cen_halos_red,\
        cen_gals_blue, cen_halos_blue

def measure_all_smf(table, volume, data_bool):
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
            volume, False)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, False)
    else:
        logmstar_col = 'stellar_mass'
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(table[logmstar_col], volume, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, True)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, True)
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

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

    maxis_red_arr = []
    phi_red_arr = []
    maxis_blue_arr = []
    phi_blue_arr = []
    cen_gals_red_arr = []
    cen_halos_red_arr = []
    cen_gals_blue_arr = []
    cen_halos_blue_arr = []

    for theta in a_list:  
        f_red_cen, f_red_sat = hybrid_quenching_model(theta, gals_df_)
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df_)
        v_sim = 130**3
        total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
        , False)     
        cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue = \
            get_centrals_mock(gals_df)
        maxis_red_arr.append(red_model[0])
        phi_red_arr.append(red_model[1])
        maxis_blue_arr.append(blue_model[0])
        phi_blue_arr.append(blue_model[1])
        cen_gals_red_arr.append(cen_gals_red)
        cen_halos_red_arr.append(cen_halos_red)
        cen_gals_blue_arr.append(cen_gals_blue)
        cen_halos_blue_arr.append(cen_halos_blue)

    return [maxis_red_arr, phi_red_arr, maxis_blue_arr, phi_blue_arr, 
    cen_gals_red_arr, cen_halos_red_arr, cen_gals_blue_arr, cen_halos_blue_arr]

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
    chunks = np.array([mcmc_table_pctl.iloc[:,:4].values[i::5] \
        for i in range(5)])
    pool = Pool(processes=nproc)
    result = pool.map(mp_func, chunks)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return result

def plot_mf(result, red_data, blue_data, maxis_bf_red, phi_bf_red, 
    maxis_bf_blue, phi_bf_blue, bf_chi2):
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
    fig1 = plt.figure(figsize=(10,10))

    maxis_red_data, phi_red_data, err_red_data = red_data[0], red_data[1], \
        red_data[2]
    maxis_blue_data, phi_blue_data, err_blue_data = blue_data[0], blue_data[1], \
        blue_data[2]
    lower_err = phi_red_data[:-1] - err_red_data
    upper_err = phi_red_data[:-1] + err_red_data
    lower_err = phi_red_data[:-1] - lower_err
    upper_err = upper_err - phi_red_data[:-1]
    asymmetric_err = [lower_err, upper_err]
    plt.errorbar(maxis_red_data[:-1],phi_red_data[:-1],yerr=asymmetric_err,
        color='darkred',fmt='s', ecolor='darkred',markersize=5,capsize=5,
        capthick=0.5,label='data',zorder=10)
    lower_err = phi_blue_data[:-1] - err_blue_data
    upper_err = phi_blue_data[:-1] + err_blue_data
    lower_err = phi_blue_data[:-1] - lower_err
    upper_err = upper_err - phi_blue_data[:-1]
    asymmetric_err = [lower_err, upper_err]
    plt.errorbar(maxis_blue_data[:-1],phi_blue_data[:-1],yerr=asymmetric_err,
        color='darkblue',fmt='s', ecolor='darkblue',markersize=5,capsize=5,
        capthick=0.5,label='data',zorder=10)
    for idx in range(len(result[0][0])):
        plt.plot(result[0][0][idx],result[0][1][idx],color='indianred',
            linestyle='-',alpha=0.3,zorder=0,label='model')
    for idx in range(len(result[0][2])):
        plt.plot(result[0][2][idx],result[0][3][idx],color='cornflowerblue',
            linestyle='-',alpha=0.3,zorder=0,label='model')
    for idx in range(len(result[1][0])):
        plt.plot(result[1][0][idx],result[1][1][idx],color='indianred',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[1][2])):
        plt.plot(result[1][2][idx],result[1][3][idx],color='cornflowerblue',
            linestyle='-',alpha=0.3,zorder=0,label='model')
    for idx in range(len(result[2][0])):
        plt.plot(result[2][0][idx],result[2][1][idx],color='indianred',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[2][2])):
        plt.plot(result[2][2][idx],result[2][3][idx],color='cornflowerblue',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[3][0])):
        plt.plot(result[3][0][idx],result[3][1][idx],color='indianred',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[3][2])):
        plt.plot(result[3][2][idx],result[3][3][idx],color='cornflowerblue',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[4][0])):
        plt.plot(result[4][0][idx],result[4][1][idx],color='indianred',
            linestyle='-',alpha=0.3,zorder=0)
    for idx in range(len(result[4][2])):
        plt.plot(result[4][2][idx],result[4][3][idx],color='cornflowerblue',
            linestyle='-',alpha=0.3,zorder=0)
    # REMOVED BEST FIT ERROR
    plt.errorbar(maxis_bf_red,phi_bf_red,
        color='darkred',fmt='-s',ecolor='darkred',markersize=3,lw=3,
        capsize=5,capthick=0.5,label='best fit',zorder=10)
    plt.errorbar(maxis_bf_blue,phi_bf_blue,
        color='darkblue',fmt='-s',ecolor='darkblue',markersize=3,lw=3,
        capsize=5,capthick=0.5,label='best fit',zorder=10)
    plt.ylim(-5,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
    plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
        xy=(0.1, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=15)
    plt.show()
    if mf_type == 'smf':
        plt.savefig(path_to_figures + 'smf_colour_{0}.png'.format(survey))
    elif mf_type == 'bmf':
        plt.savefig(path_to_figures + 'bmf_colour_{0}.png'.format(survey))

def plot_xmhm(result, gals_bf_red, halos_bf_red, gals_bf_blue, halos_bf_blue,
    gals_data_red, halos_data_red, gals_data_blue, 
    halos_data_blue, bf_chi2):
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
    x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    gals_bf_red,base=0.4,bin_statval='center')
    x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    gals_bf_blue,base=0.4,bin_statval='center')
    # x_b10,y_b10,y_std_b10,y_std_err_b10 = Stats_one_arr(halos_b10,\
    #     gals_b10,base=0.4,bin_statval='center')
    x_data_red,y_data_red,y_std_data_red,y_std_err_data_red = Stats_one_arr(halos_data_red,\
        gals_data_red,base=0.4,bin_statval='center')
    x_data_blue,y_data_blue,y_std_data_blue,y_std_err_data_blue = Stats_one_arr(halos_data_blue,\
        gals_data_blue,base=0.4,bin_statval='center')
    # y_std_err_data = err_data

    fig1 = plt.figure(figsize=(10,10))
    # NOT PLOTTING DATA RELATION
    # plt.errorbar(x_data_red,y_data_red,yerr=y_std_err_data_red,color='r',fmt='s',\
    #     ecolor='r',markersize=5,capsize=5,capthick=0.5,\
    #         label='data',zorder=10)
    # plt.errorbar(x_data_blue,y_data_blue,yerr=y_std_err_data_blue,color='b',fmt='s',\
    #     ecolor='b',markersize=5,capsize=5,capthick=0.5,\
    #         label='data',zorder=10)

    # NOT PLOTTING BEHROOZI RELATION
    # plt.errorbar(x_b10,y_b10, color='k',fmt='--s',\
    #     markersize=3, label='Behroozi10', zorder=10, alpha=0.7)

    for idx in range(len(result[0][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[0][5][idx],result[0][4][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-', \
            alpha=0.5, zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[0][7][idx],result[0][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[1][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[1][5][idx],result[1][4][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[1][7][idx],result[1][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[2][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[2][5][idx],result[2][4][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[2][7][idx],result[2][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[3][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[3][5][idx],result[3][4][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[3][7][idx],result[3][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[4][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[4][5][idx],result[4][4][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[4][7][idx],result[4][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')

    # REMOVED ERROR BAR ON BEST FIT
    plt.errorbar(x_bf_red,y_bf_red,color='darkred',fmt='-s',ecolor='darkred',lw=3,\
        markersize=4,capsize=5,capthick=0.5,label='best fit',zorder=10)
    plt.errorbar(x_bf_blue,y_bf_blue,color='darkblue',fmt='-s',ecolor='darkblue',lw=3,\
        markersize=4,capsize=5,capthick=0.5,label='best fit',zorder=10)

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    if mf_type == 'smf':
        if survey == 'eco':
            plt.ylim(np.log10((10**8.9)/2.041),)
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
    plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
        xy=(0.8, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=15)
    plt.show()
    if mf_type == 'smf':
        plt.savefig(path_to_figures + 'smhm_emcee_{0}.png'.format(survey))
    elif mf_type == 'bmf':
        plt.savefig(path_to_figures + 'bmhm_emcee_{0}.png'.format(survey))

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

global model_init
global gals_df_
global path_to_figures

machine = 'mac'
mf_type = 'smf'
survey = 'eco'
nproc = 32

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

chi2_file = path_to_proc + 'smhm_colour_run9/{0}_colour_chi2.txt'.format(survey)
chain_file = path_to_proc + 'smhm_colour_run9/mcmc_{0}_colour_raw.txt'.format(survey)

if survey == 'eco':
    catl_file = path_to_raw + "eco/eco_all.csv"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)

print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)

print('Reading catalog')
catl, volume, cvar, z_median = read_data_catl(catl_file, survey)

print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)

print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Retrieving survey centrals')
cen_gals_data_red, cen_halos_data_red, cen_gals_data_blue, cen_halos_data_blue =\
     get_centrals_data(catl)

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

print('Measuring error in data from mocks')
total_data, red_data[2], blue_data[2] = \
    get_err_data(survey, path_to_mocks)

model_init = halocat_init(halo_catalog, z_median)

print('Populating halos using best fit shmr params')
bf_params_shmr = np.array([12.32381675, 10.56581819, 0.4276319, 0.7457711, 
    0.34784431])
gals_df_ = populate_mock(bf_params_shmr, model_init)
gals_df_ = assign_cen_sat_flag(gals_df_)
# gals_df_ = gals_df_[['stellar_mass', 'C_S', 'halo_mvir', 'halo_mvir_host_halo',\
#     'halo_macc','halo_hostid', 'halo_id']]

print('Multiprocessing')
result = mp_init(mcmc_table_pctl, nproc)

print('Getting best fit model')
maxis_bf_red, phi_bf_red, maxis_bf_blue, phi_bf_blue, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue = get_best_fit_model(bf_params)

plot_mf(result, red_data, blue_data, maxis_bf_red, phi_bf_red, 
    maxis_bf_blue, phi_bf_blue, bf_chi2)

plot_xmhm(result, cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue,
    cen_gals_data_red, cen_halos_data_red, cen_gals_data_blue, 
    cen_halos_data_blue, bf_chi2)