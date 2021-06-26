from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerBase
from scipy.stats import binned_statistic as bs
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import markers, rc
import pandas as pd
import numpy as np
import argparse
import random
import math
import time
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

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
    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'mstar_q','mh_q','mu','nu']
    
    emcee_table = pd.read_csv(path_to_file, names=colnames, comment='#',
        header=None, sep='\s+')

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

    return emcee_table

def read_mock_catl(filename, catl_format='.hdf5'):
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
        # columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
        #             'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
        #             'fc', 'grpmb', 'grpms','modelu_rcorr']

        # 13878 galaxies
        # eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
        #     usecols=columns)

        eco_buff = read_mock_catl(path_to_file)

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

def get_paramvals_percentile(mcmc_table, pctl, chi2, randints_df=None):
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
    if randints_df is not None: # This returns a bool; True if df has contents
        mcmc_table['mock_num'] = randints_df.mock_num.values.astype(int)
    mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:9]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][9]
    if randints_df is not None:
        bf_randint = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
            values[0][5].astype(int)
        mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)
        return mcmc_table_pctl, bf_params, bf_chi2, bf_randint
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2

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
        return [max_total, phi_total, err_total, counts_total] , \
            [max_red, phi_red, err_red, counts_red] , \
                [max_blue, phi_blue, err_blue, counts_blue]
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'stellar_mass'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')
        return [max_total, phi_total, err_total, counts_total]

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
            # For eco total
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

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """
    if data_bool:
        mstar_arr = catl.logmstar.values
    else:
        mstar_arr = catl.stellar_mass.values

    colour_label_arr = catl.colour_label.values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = mstar_arr

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 7

        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result = bs(logmstar_arr, colour_label_arr, blue_frac_helper, bins=bins)
    edges = result[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue = result[0]

    return maxis, f_blue

def get_sigma_per_group_data(df):

    catl = df.copy()
    if survey == 'eco' or survey == 'resolvea':
        catl = catl.loc[catl.logmstar >= 8.9]
    elif survey == 'resolveb':
        catl = catl.loc[catl.logmstar >= 8.7]
    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    red_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'R') & (catl.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'B') & (catl.g_galtype == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

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

def get_host_halo_mock(df, mock):
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
        cen_halos = df.halo_mvir[df.cs_flag == 1].reset_index(drop=True)
        sat_halos = df.halo_mvir_host_halo[df.cs_flag == 0].reset_index(drop=True)
    else:
        # Loghalom in the mock catalogs is actually host halo mass i.e. 
        # For satellites, the loghalom value will be the value of the central's
        # loghalom in that halo group and the haloids for the satellites are the 
        # haloid of the central 
        cen_halos = 10**(df.loghalom[df.cs_flag == 1]).reset_index(drop=True)
        sat_halos = 10**(df.loghalom[df.cs_flag == 0]).reset_index(drop=True)

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(df, mock, randint=None):
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

    if mock == 'vishnu' and randint:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['{0}'.format(randint)].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['{0}'.format(randint)].values[idx]))

    elif mock == 'vishnu':
        cen_gals = 10**(df.stellar_mass[df.cs_flag == 1]).reset_index(drop=True)
        sat_gals = 10**(df.stellar_mass[df.cs_flag == 0]).reset_index(drop=True)
    
    else:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append((10**(df.logmstar.values[idx]))/2.041)
            elif value == 0:
                sat_gals.append((10**(df.logmstar.values[idx]))/2.041)

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
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
    f_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = read_mock_catl(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            #! Need group cz here
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                (mock_pd.logmstar.values >= mstar_limit)]

            ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            ## Using best-fit found for new ECO data using optimize_hybridqm_eco,py
            Mstar_q = 10.49 # Msun/h
            Mh_q = 14.03 # Msun/h
            mu = 0.69
            nu = 0.148


            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            logmstar_arr = mock_pd.logmstar.values 

            #Measure SMF of mock using diff_smf function
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(logmstar_arr, volume, False)
            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_arr_total.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)

            #Measure blue fraction of galaxies
            max, f_blue = blue_frac(mock_pd, False, True)
            f_blue_arr.append(f_blue)

            phi_red, phi_blue = \
                get_colour_smf_from_fblue(mock_pd, f_blue, max, volume, 
                False)
            phi_arr_red.append(phi_red)
            phi_arr_blue.append(phi_blue)
            


    phi_arr_total = np.array(phi_arr_total)
    f_blue_arr = np.array(f_blue_arr)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)

    std_phi_red = np.std(phi_arr_red, axis=0)
    std_phi_blue = np.std(phi_arr_blue, axis=0)

    # Covariance matrix for total phi (all galaxies)
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    err_total = np.sqrt(cov_mat.diagonal())

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]
    phi_total_4 = phi_arr_total[:,4]
    phi_total_5 = phi_arr_total[:,5]

    f_blue_0 = f_blue_arr[:,0]
    f_blue_1 = f_blue_arr[:,1]
    f_blue_2 = f_blue_arr[:,2]
    f_blue_3 = f_blue_arr[:,3]
    f_blue_4 = f_blue_arr[:,4]
    f_blue_5 = f_blue_arr[:,5]

    combined_df = pd.DataFrame({'phi_tot_0':phi_total_0, \
        'phi_tot_1':phi_total_1, 'phi_tot_2':phi_total_2, \
        'phi_tot_3':phi_total_3, 'phi_tot_4':phi_total_4, \
        'phi_tot_5':phi_total_5,\
        'f_blue_0':f_blue_0, 'f_blue_1':f_blue_1, 
        'f_blue_2':f_blue_2, 'f_blue_3':f_blue_3, 
        'f_blue_4':f_blue_4, 'f_blue_5':f_blue_5})

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, std_phi_red, std_phi_blue

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

    # if survey == 'eco' or survey == 'resolvea':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.9) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.4) / 2.041), 1)
    # elif survey == 'resolveb':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.7) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.1) / 2.041), 1)
    # sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model.mock.galaxy_table#[sample_mask]
    gals_df = pd.DataFrame(np.array(gals))

    return gals_df

def get_centrals_mock(gals_df, randint=None):
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
    cen_gals = []
    cen_halos = []
    cen_gals_red = []
    cen_halos_red = []
    cen_gals_blue = []
    cen_halos_blue = []
    f_red_cen_gals_red = []
    f_red_cen_gals_blue = []

    if randint:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 1:
                cen_gals.append(gals_df['{0}'.format(randint)][idx])
                cen_halos.append(gals_df['halo_mvir'][idx])
                if gals_df['colour_label'][idx] == 'R':
                    cen_gals_red.append(gals_df['{0}'.format(randint)][idx])
                    cen_halos_red.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    cen_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                    cen_halos_blue.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_blue.append(gals_df['f_red'][idx])
    else:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 1:
                cen_gals.append(gals_df['stellar_mass'][idx])
                cen_halos.append(gals_df['halo_mvir'][idx])
                if gals_df['colour_label'][idx] == 'R':
                    cen_gals_red.append(gals_df['stellar_mass'][idx])
                    cen_halos_red.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    cen_gals_blue.append(gals_df['stellar_mass'][idx])
                    cen_halos_blue.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_blue.append(gals_df['f_red'][idx])

    cen_gals = np.array(cen_gals)
    cen_halos = np.log10(np.array(cen_halos))
    cen_gals_red = np.array(cen_gals_red)
    cen_halos_red = np.log10(np.array(cen_halos_red))
    cen_gals_blue = np.array(cen_gals_blue)
    cen_halos_blue = np.log10(np.array(cen_halos_blue))

    return cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
        cen_halos_blue, f_red_cen_gals_red, f_red_cen_gals_blue

def get_satellites_mock(gals_df, randint=None):
    """
    Get satellites and their host halos from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    sat_gals: array
        Array of central galaxy masses

    sat_halos: array
        Array of central halo masses
    """
    sat_gals_red = []
    sat_halos_red = []
    sat_gals_blue = []
    sat_halos_blue = []
    f_red_sat_gals_red = []
    f_red_sat_gals_blue = []

    if randint:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 0:
                if gals_df['colour_label'][idx] == 'R':
                    sat_gals_red.append(gals_df['{0}'.format(randint)][idx])
                    sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    sat_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                    sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_blue.append(gals_df['f_red'][idx])
    else:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 0:
                if gals_df['colour_label'][idx] == 'R':
                    sat_gals_red.append(gals_df['stellar_mass'][idx])
                    sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    sat_gals_blue.append(gals_df['stellar_mass'][idx])
                    sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_blue.append(gals_df['f_red'][idx])


    sat_gals_red = np.array(sat_gals_red)
    sat_halos_red = np.log10(np.array(sat_halos_red))
    sat_gals_blue = np.array(sat_gals_blue)
    sat_halos_blue = np.log10(np.array(sat_halos_blue))

    return sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
        f_red_sat_gals_red, f_red_sat_gals_blue

def mp_init(mcmc_table_pctl, nproc):
    """
    Initializes multiprocessing of mocks and smf and smhm measurements

    Parameters
    ----------
    mcmc_table_pctl: pandas dataframe
        Mcmc chain dataframe of 100 random samples

    nproc: int
        Number of processes to use in multiprocessing

    Returns
    ---------
    result: multidimensional array
        Array of smf and smhm data
    """
    start = time.time()
    params_df = mcmc_table_pctl.iloc[:,:9].reset_index(drop=True)
    if many_behroozi_mocks:
        mock_num_df = mcmc_table_pctl.iloc[:,5].reset_index(drop=True)
        frames = [params_df, mock_num_df]
        mcmc_table_pctl_new = pd.concat(frames, axis=1)
        chunks = np.array([mcmc_table_pctl_new.values[i::5] \
            for i in range(5)])
    else:
        chunks = np.array([params_df.values[i::5] \
            for i in range(5)])
    pool = Pool(processes=nproc)
    result = pool.map(mp_func, chunks)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return result

def mp_func(a_list):
    """
    Apply hybrid quenching model based on four parameter values

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
    # v_sim = 890641.5172927063 

    maxis_total_arr = []
    phi_total_arr = []
    maxis_fblue_arr = []
    f_blue_arr = []
    phi_red_model_arr = []
    phi_blue_model_arr = []
    cen_gals_red_arr = []
    cen_halos_red_arr = []
    cen_gals_blue_arr = []
    cen_halos_blue_arr = []
    f_red_cen_red_arr = []
    f_red_cen_blue_arr = []
    sat_gals_red_arr = [] 
    sat_halos_red_arr = []
    sat_gals_blue_arr = [] 
    sat_halos_blue_arr = [] 
    f_red_sat_red_arr = []
    f_red_sat_blue_arr = []
    cen_gals_arr = []
    cen_halos_arr = []

    for theta in a_list:  
        if many_behroozi_mocks:
            randint_logmstar = int(theta[4])
            cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', \
            '{0}'.format(randint_logmstar), \
            'g_galtype_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar)]
            
            gals_df = gal_group_df_subset[cols_to_use]

            gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
                format(randint_logmstar),'groupid_{0}'.format(randint_logmstar)]).\
                reset_index(drop=True)

            gals_df[['g_galtype_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar)]] = \
                gals_df[['g_galtype_{0}'.format(randint_logmstar),\
                'groupid_{0}'.format(randint_logmstar)]].astype(int)


        else:
            gals_df = populate_mock(theta[:5], model_init)
            gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].\
                reset_index(drop=True)
            gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
                gals_df['halo_id'], 1, 0)

            cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
                'stellar_mass']
            gals_df = gals_df[cols_to_use]
            gals_df.stellar_mass = np.log10(gals_df.stellar_mass)

        # Stellar masses in log but halo masses not in log
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, 
            'vishnu')
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

        ## Observable #1 - Total SMF
        total_model = measure_all_smf(gals_df, v_sim , False)    
        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gals_df, True, False)
        
        cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
            cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
                get_centrals_mock(gals_df)
        sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
            f_red_sat_red, f_red_sat_blue = \
                get_satellites_mock(gals_df)

        phi_red_model, phi_blue_model = \
            get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
            True)


        maxis_total_arr.append(total_model[0])
        phi_total_arr.append(total_model[1])
        maxis_fblue_arr.append(f_blue[0])
        f_blue_arr.append(f_blue[1])
        phi_red_model_arr.append(phi_red_model)
        phi_blue_model_arr.append(phi_blue_model)
        cen_gals_red_arr.append(cen_gals_red)
        cen_halos_red_arr.append(cen_halos_red)
        cen_gals_blue_arr.append(cen_gals_blue)
        cen_halos_blue_arr.append(cen_halos_blue)
        f_red_cen_red_arr.append(f_red_cen_red)
        f_red_cen_blue_arr.append(f_red_cen_blue)
        sat_gals_red_arr.append(sat_gals_red) 
        sat_halos_red_arr.append(sat_halos_red)
        sat_gals_blue_arr.append(sat_gals_blue)
        sat_halos_blue_arr.append(sat_halos_blue)
        f_red_sat_red_arr.append(f_red_sat_red)
        f_red_sat_blue_arr.append(f_red_sat_blue)
        cen_gals_arr.append(cen_gals)
        cen_halos_arr.append(cen_halos)
    

    return [maxis_total_arr, phi_total_arr, maxis_fblue_arr, f_blue_arr, 
    phi_red_model_arr, phi_blue_model_arr,
    cen_gals_red_arr, cen_halos_red_arr, cen_gals_blue_arr, cen_halos_blue_arr,
    f_red_cen_red_arr, f_red_cen_blue_arr, sat_gals_red_arr, sat_halos_red_arr, 
    sat_gals_blue_arr, sat_halos_blue_arr, f_red_sat_red_arr, f_red_sat_blue_arr,
    cen_gals_arr, cen_halos_arr]

def get_best_fit_model(best_fit_params, best_fit_mocknum=None):
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
    if best_fit_mocknum:
        cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', \
            '{0}'.format(best_fit_mocknum), \
            'g_galtype_{0}'.format(best_fit_mocknum), \
            'groupid_{0}'.format(best_fit_mocknum)]

        gals_df = gal_group_df_subset[cols_to_use]

        gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
            format(best_fit_mocknum),'groupid_{0}'.format(best_fit_mocknum)]).\
            reset_index(drop=True)

        gals_df[['g_galtype_{0}'.format(best_fit_mocknum), \
            'groupid_{0}'.format(best_fit_mocknum)]] = \
            gals_df[['g_galtype_{0}'.format(best_fit_mocknum),\
            'groupid_{0}'.format(best_fit_mocknum)]].astype(int)
    else:
        gals_df = populate_mock(best_fit_params[:5], model_init)
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].\
            reset_index(drop=True)
        gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
            gals_df['halo_id'], 1, 0)

        cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
            'stellar_mass']
        gals_df = gals_df[cols_to_use]
        gals_df.stellar_mass = np.log10(gals_df.stellar_mass)

    # Stellar masses in log but halo masses not in log
    f_red_cen, f_red_sat = hybrid_quenching_model(best_fit_params[5:], gals_df, 
        'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
    v_sim = 130**3 
    # v_sim = 890641.5172927063 

    ## Observable #1 - Total SMF
    total_model = measure_all_smf(gals_df, v_sim , False)    
    ## Observable #2 - Blue fraction
    f_blue = blue_frac(gals_df, True, False)

    cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
        cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
            get_centrals_mock(gals_df)
    sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
        f_red_sat_red, f_red_sat_blue = \
            get_satellites_mock(gals_df)

    phi_red_model, phi_blue_model = \
        get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
        True)


    max_total = total_model[0]
    phi_total = total_model[1]
    max_fblue = f_blue[0]
    fblue = f_blue[1]

    return max_total, phi_total, max_fblue, fblue, phi_red_model, \
        phi_blue_model, cen_gals_red, cen_halos_red,\
        cen_gals_blue, cen_halos_blue, f_red_cen_red, f_red_cen_blue, \
        sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, f_red_sat_red, \
        f_red_sat_blue, cen_gals, cen_halos

def get_colour_smf_from_fblue(df, frac_arr, bin_centers, volume, h1_bool):
    
    if h1_bool:
        logmstar_arr = df.stellar_mass.values
    if not h1_bool:
        mstar_arr = df.logmstar.values
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)

    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = bin_centers - (0.5 * bin_width)
    # Done to include the right side of the last bin
    bin_edges = np.insert(bin_edges, len(bin_edges), bin_edges[-1]+bin_width)
    counts, edg = np.histogram(logmstar_arr, bins=bin_edges)  
    
    counts_blue = frac_arr * counts
    counts_red = (1-frac_arr) * counts

    # Normalized by volume and bin width
    phi_red = counts_red / (volume * bin_width)  # not a log quantity
    phi_red = np.log10(phi_red)

    phi_blue = counts_blue / (volume * bin_width)  # not a log quantity
    phi_blue = np.log10(phi_blue)

    ## Check to make sure that the reconstruced mass functions are the same as 
    ## those from diff_smf(). They aren't exactly but they match up if the 
    ## difference in binning is corrected.
    # fig1 = plt.figure()
    # plt.plot(bin_centers, phi_red, 'r+', ls='--', label='reconstructed')
    # plt.plot(bin_centers, phi_blue, 'b+', ls='--', label='reconstructed')
    # plt.plot(red_data[0], red_data[1], 'r+', ls='-', label='measured')
    # plt.plot(blue_data[0], blue_data[1], 'b+', ls='-', label='measured')
    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)
    # plt.legend()
    # plt.show()

    return phi_red, phi_blue

def plot_total_mf(result, total_data, maxis_bf_total, phi_bf_total,
    bf_chi2, err_colour):
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
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    x_phi_total_data, y_phi_total_data = total_data[0], total_data[1]
    x_phi_total_model = result[0][0][0]

    i_outer = 0
    total_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

    tot_phi_max = np.amax(total_mod_arr, axis=0)
    tot_phi_min = np.amin(total_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mt = plt.fill_between(x=x_phi_total_model, y1=tot_phi_max, 
        y2=tot_phi_min, color='silver', alpha=0.4)

    dt = plt.errorbar(x_phi_total_data, y_phi_total_data, yerr=err_colour[0:6],
        color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bft, = plt.plot(maxis_bf_total, phi_bf_total, color='k', ls='--', lw=4, 
        zorder=10)

    plt.ylim(-4,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

    plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if survey == 'eco':
        plt.title('ECO')
    plt.show()

def plot_colour_mf(result, phi_red_data, phi_blue_data, phi_bf_red, phi_bf_blue,
    std_red, std_blue, bf_chi2):
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
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    ## The same bins were used for fblue that were used for total SMF
    x_phi_red_model = result[0][2][0]
    x_phi_blue_model = result[0][2][0]

    i_outer = 0
    red_mod_arr = []
    blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    red_phi_max = np.amax(red_mod_arr, axis=0)
    red_phi_min = np.amin(red_mod_arr, axis=0)
    blue_phi_max = np.amax(blue_mod_arr, axis=0)
    blue_phi_min = np.amin(blue_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mr = plt.fill_between(x=x_phi_red_model, y1=red_phi_max, 
        y2=red_phi_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=x_phi_blue_model, y1=blue_phi_max, 
        y2=blue_phi_min, color='cornflowerblue',alpha=0.4)

    # dr = plt.errorbar(x_phi_red_model, phi_red_data, yerr=std_red,
    #     color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    #     capthick=1.5, zorder=10, marker='^')
    # db = plt.errorbar(x_phi_blue_model, phi_blue_data, yerr=std_blue,
    #     color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    #     capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bfr, = plt.plot(x_phi_red_model, phi_bf_red,
        color='maroon', ls='--', lw=4, zorder=10)
    bfb, = plt.plot(x_phi_blue_model, phi_bf_blue,
        color='mediumblue', ls='--', lw=4, zorder=10)

    plt.ylim(-4,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

    # plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data','Models','Best-fit'],
    #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
    plt.legend([(mr, mb), (bfr, bfb)], ['Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})


    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.80), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if survey == 'eco':
        plt.title('ECO')
    plt.show()

def plot_fblue(result, fblue_data, maxis_bf_fblue, bf_fblue,
    bf_chi2, err_colour):
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
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    x_fblue_data, y_fblue_data = fblue_data[0], fblue_data[1]
    x_fblue_model = result[0][2][0]

    i_outer = 0
    fblue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

    fblue_max = np.amax(fblue_mod_arr, axis=0)
    fblue_min = np.amin(fblue_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mt = plt.fill_between(x=x_fblue_model, y1=fblue_max, 
        y2=fblue_min, color='silver', alpha=0.4)

    dt = plt.errorbar(x_fblue_data, y_fblue_data, yerr=err_colour[6:],
        color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bft, = plt.plot(maxis_bf_fblue, bf_fblue, color='k', ls='--', lw=4, 
        zorder=10)

    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$f_{blue}$', fontsize=30)

    plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if survey == 'eco':
        plt.title('ECO')
    plt.show()

def plot_xmhm(result, gals_bf, halos_bf, bf_chi2):
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
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
    elif survey == 'eco':
        line_label = 'ECO'
    
    x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(halos_bf,\
    gals_bf,base=0.4,bin_statval='center')
    
    fig1 = plt.figure(figsize=(10,10))
    for idx in range(len(result[0][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[0][19][idx],result[0][18][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=0,label='Models')
    for idx in range(len(result[1][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[1][19][idx],result[1][18][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=1)
    for idx in range(len(result[2][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[2][19][idx],result[2][18][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=2)
    for idx in range(len(result[3][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[3][19][idx],result[3][18][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=3)
    for idx in range(len(result[4][0])):
        x_model,y_model,y_std_model,y_std_err_model = \
            Stats_one_arr(result[4][19][idx],result[4][18][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
            zorder=4)

    plt.plot(x_bf, y_bf, color='k', lw=4, label='Best-fit', zorder=10)

    plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
        [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
        plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
        hatch='\\')

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,15)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    if mf_type == 'smf':
        if survey == 'eco':
            plt.ylim(np.log10((10**8.9)/2.041),12.8)
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 30})
    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)
    plt.show()

def plot_colour_xmhm(result, gals_bf_red, halos_bf_red, gals_bf_blue, 
    halos_bf_blue, bf_chi2):
    """
    Plot SMHM from data, best fit param values, param values corresponding to 
    68th percentile 1000 lowest chi^2 values and behroozi 2010 param values

    Parameters
    ----------
    result: multidimensional array
        Array of central galaxy and halo masses
    
    gals_bf_red: array
        Array of y-axis stellar mass values for red SMHM

    halos_bf_red: array
        Array of x-axis halo mass values for red SMHM
    
    gals_bf_blue: array
        Array of y-axis stellar mass values for blue SMHM

    halos_bf_blue: array
        Array of x-axis halo mass values for blue SMHM

    bf_chi2: float
        Chi-squared value of best fit model
    Returns
    ---------
    Nothing; plot is shown on the screen
    """   
    class AnyObjectHandler(HandlerBase): 
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],  
                linestyle=orig_handle[2],
                color=orig_handle[1])
            return [l1, l2]  

    x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    gals_bf_red,base=0.4,bin_statval='center')
    x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    gals_bf_blue,base=0.4,bin_statval='center')

    fig1 = plt.figure(figsize=(10,10))

    for idx in range(len(result[0][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[0][7][idx],result[0][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-', \
            alpha=0.5, zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[0][9][idx],result[0][8][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[1][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[1][7][idx],result[1][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[1][9][idx],result[1][8][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[2][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[2][7][idx],result[2][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[2][9][idx],result[2][8][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[3][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[3][7][idx],result[3][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[3][9][idx],result[3][8][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
    for idx in range(len(result[4][0])):
        x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
            Stats_one_arr(result[4][7][idx],result[4][6][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
            zorder=0,label='model')
        x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
            Stats_one_arr(result[4][9][idx],result[4][8][idx],base=0.4,\
                bin_statval='center')
        plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
            zorder=0,label='model')

    # REMOVED ERROR BAR ON BEST FIT
    plt.plot(x_bf_red,y_bf_red,color='darkred',lw=4,label='Best-fit',zorder=10)
    plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=4,
        label='Best-fit',zorder=10)

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    if mf_type == 'smf':
        if survey == 'eco':
            plt.ylim(np.log10((10**8.9)/2.041),)
            plt.title('ECO')
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
            plt.title('ECO')
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
    plt.legend([("darkred", "darkblue", "-"), \
        ("indianred","cornflowerblue", "-")],\
        ["Best-fit", "Models"], handler_map={tuple: AnyObjectHandler()},\
        loc='best', prop={'size': 30})
    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)
    plt.show()

def plot_red_fraction_cen(result, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
    f_red_cen_blue):


    cen_gals_arr = []
    cen_halos_arr = []
    fred_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        cen_gals_idx_arr = []
        cen_halos_idx_arr = []
        fred_idx_arr = []
        for idx in range(len(result[chunk_counter][0])):
            red_cen_gals_idx = result[chunk_counter][6][idx]
            red_cen_halos_idx = result[chunk_counter][7][idx]
            blue_cen_gals_idx = result[chunk_counter][8][idx]
            blue_cen_halos_idx = result[chunk_counter][9][idx]
            fred_red_cen_idx = result[chunk_counter][10][idx]
            fred_blue_cen_idx = result[chunk_counter][11][idx]  

            cen_gals_idx_arr = list(red_cen_gals_idx) + list(blue_cen_gals_idx)
            cen_gals_arr.append(cen_gals_idx_arr)

            cen_halos_idx_arr = list(red_cen_halos_idx) + list(blue_cen_halos_idx)
            cen_halos_arr.append(cen_halos_idx_arr)

            fred_idx_arr = list(fred_red_cen_idx) + list(fred_blue_cen_idx)
            fred_arr.append(fred_idx_arr)
            
            cen_gals_idx_arr = []
            cen_halos_idx_arr = []
            fred_idx_arr = []

        chunk_counter+=1
    
    cen_gals_bf = []
    cen_halos_bf = []
    fred_bf = []

    cen_gals_bf = list(cen_gals_red) + list(cen_gals_blue)
    cen_halos_bf = list(cen_halos_red) + list(cen_halos_blue)
    fred_bf = list(f_red_cen_red) + list(f_red_cen_blue)


    fig1 = plt.figure(figsize=(10,8))
    if quenching == 'hybrid':
        for idx in range(len(cen_gals_arr)):
            x, y = zip(*sorted(zip(cen_gals_arr[idx],fred_arr[idx])))
            plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, solid_capstyle='round')
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

        x, y = zip(*sorted(zip(cen_gals_arr[0],fred_arr[0])))
        x_bf, y_bf = zip(*sorted(zip(cen_gals_bf,fred_bf)))
        # Plotting again just so that adding label to legend is easier
        plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, solid_capstyle='round')
        plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, solid_capstyle='round')

    elif quenching == 'halo':
        #! Change plotting so its similar to hybrid case above
        for idx in range(len(cen_halos_arr)):
            plt.scatter(cen_halos_arr[idx], fred_arr[idx], alpha=0.4, s=150, c='cornflowerblue') 
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.scatter(cen_halos_arr[0], fred_arr[0], alpha=1.0, s=150, c='cornflowerblue', label='Models')
        plt.scatter(cen_halos_bf, fred_bf, alpha=0.4, s=150, c='mediumorchid', label='Best-fit')

    plt.ylabel(r'\boldmath$f_{red, cen}$', fontsize=30)
    plt.legend(loc='best', prop={'size':30})
    plt.show()

def plot_red_fraction_sat(result, sat_gals_red, sat_halos_red, \
    sat_gals_blue, sat_halos_blue, f_red_sat_red, f_red_sat_blue):

    sat_gals_arr = []
    sat_halos_arr = []
    fred_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        sat_gals_idx_arr = []
        sat_halos_idx_arr = []
        fred_idx_arr = []
        for idx in range(len(result[chunk_counter][0])):
            red_sat_gals_idx = result[chunk_counter][12][idx]
            red_sat_halos_idx = result[chunk_counter][13][idx]
            blue_sat_gals_idx = result[chunk_counter][14][idx]
            blue_sat_halos_idx = result[chunk_counter][15][idx]
            fred_red_sat_idx = result[chunk_counter][16][idx]
            fred_blue_sat_idx = result[chunk_counter][17][idx]  

            sat_gals_idx_arr = list(red_sat_gals_idx) + list(blue_sat_gals_idx)
            sat_gals_arr.append(sat_gals_idx_arr)

            sat_halos_idx_arr = list(red_sat_halos_idx) + list(blue_sat_halos_idx)
            sat_halos_arr.append(sat_halos_idx_arr)

            fred_idx_arr = list(fred_red_sat_idx) + list(fred_blue_sat_idx)
            fred_arr.append(fred_idx_arr)
            
            sat_gals_idx_arr = []
            sat_halos_idx_arr = []
            fred_idx_arr = []

        chunk_counter+=1
    
    sat_gals_bf = []
    sat_halos_bf = []
    fred_bf = []

    sat_gals_bf = list(sat_gals_red) + list(sat_gals_blue)
    sat_halos_bf = list(sat_halos_red) + list(sat_halos_blue)
    fred_bf = list(f_red_sat_red) + list(f_red_sat_blue)


    fig1 = plt.figure(figsize=(10,8))
    if quenching == 'hybrid':
        for idx in range(len(sat_halos_arr)):
            plt.scatter(sat_halos_arr[idx], fred_arr[idx], alpha=0.4, s=150, c='cornflowerblue')
            # x, y = zip(*sorted(zip(sat_halos_arr[idx],fred_arr[idx])))
            # plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, 
            #     solid_capstyle='round')
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\,'\
            r' \mathrm{h}^{-1} \right]$',fontsize=30)

        plt.scatter(sat_halos_arr[0], fred_arr[0], alpha=0.4, s=150, c='cornflowerblue', label='Models')
        plt.scatter(sat_halos_bf, fred_bf, alpha=1.0, s=150, c='mediumorchid', label='Best-fit')

        # x, y = zip(*sorted(zip(sat_halos_arr[0],fred_arr[0])))
        # x_bf, y_bf = zip(*sorted(zip(sat_halos_bf,fred_bf)))
        # # Plotting again just so that adding label to legend is easier
        # plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, 
        #     solid_capstyle='round')
        # plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, 
        #     solid_capstyle='round')

    elif quenching == 'halo':
        #! Change plotting so its similar to hybrid case above
        for idx in range(len(sat_halos_arr)):
            plt.scatter(sat_halos_arr[idx], fred_arr[idx], alpha=0.4, s=150, 
                c='cornflowerblue') 
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.scatter(sat_halos_arr[0], fred_arr[0], alpha=0.4, s=150, 
            c='cornflowerblue', label='Models')
        plt.scatter(sat_halos_bf, fred_bf, alpha=1.0, s=150, 
            c='mediumorchid', label='Best-fit')

    plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
    plt.legend(loc='best', prop={'size':30})
    plt.show()

def plot_zumand_fig4(result, gals_bf_red, halos_bf_red, gals_bf_blue, 
    halos_bf_blue, bf_chi2):

    # if model == 'halo':
    #     sat_halomod_df = gals_df.loc[gals_df.C_S.values == 0] 
    #     cen_halomod_df = gals_df.loc[gals_df.C_S.values == 1]
    # elif model == 'hybrid':
    #     sat_hybmod_df = gals_df.loc[gals_df.C_S.values == 0]
    #     cen_hybmod_df = gals_df.loc[gals_df.C_S.values == 1]

    logmhalo_mod_arr = result[0][7]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][7])
    for idx in range(5):
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][9])
    logmhalo_mod_arr_flat = np.hstack(logmhalo_mod_arr)

    logmstar_mod_arr = result[0][6]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][6])
    for idx in range(5):
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][8])
    logmstar_mod_arr_flat = np.hstack(logmstar_mod_arr)

    fred_mod_arr = result[0][10]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][10])
    for idx in range(5):
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][11])
    fred_mod_arr_flat = np.hstack(fred_mod_arr)

    fig1 = plt.figure()
    plt.hexbin(logmhalo_mod_arr_flat, logmstar_mod_arr_flat, 
        C=fred_mod_arr_flat, cmap='rainbow')
    cb = plt.colorbar()
    cb.set_label(r'\boldmath\ $f_{red}$')


    x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    gals_bf_red,base=0.4,bin_statval='center')
    x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    gals_bf_blue,base=0.4,bin_statval='center')

    bfr, = plt.plot(x_bf_red,y_bf_red,color='darkred',lw=5,zorder=10)
    bfb, = plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=5,zorder=10)

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
        [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
        plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
        hatch='\\')

    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')

    plt.ylim(8.45, 12.3)
    plt.xlim(10, 14.5)

    plt.legend([(bfr, bfb)], ['Best-fit'], 
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='best', 
        prop={'size': 30})

    if survey == 'eco':
        plt.title('ECO')
    
    plt.show()

global survey
global quenching
global model_init
global many_behroozi_mocks
global dof

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

many_behroozi_mocks = False
quenching = 'hybrid'
machine = 'mac'
mf_type = 'smf'
survey = 'eco'
nproc = 2

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

chi2_file = path_to_proc + 'smhm_colour_run30/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run30/mcmc_{0}_colour_raw.txt'.\
    format(survey)

if survey == 'eco':
    # catl_file = path_to_raw + "eco/eco_all.csv"
    ## New catalog with group finder run on subset after applying M* and cz cuts
    catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

print('Reading files')
chi2 = read_chi2(chi2_file)
mcmc_table = read_mcmc(chain_file)
catl, volume, z_median = read_data_catl(catl_file, survey)

print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)

print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

print('Measuring blue fraction for data')
f_blue = blue_frac(catl, False, True)
 
print('Measuring reconstructed red and blue SMF for data')
phi_red_data, phi_blue_data = get_colour_smf_from_fblue(catl, f_blue[1], 
    f_blue[0], volume, False)

print('Measuring new dynamical metric for data')
red_sigma_data, red_cen_stellar_mass_data, blue_sigma_data, \
        blue_cen_stellar_mass_data = get_sigma_per_group_data(catl)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

print('Measuring error in data from mocks')
err_data, err_phi_red, err_phi_blue = get_err_data(survey, path_to_mocks)

dof = len(err_data) - len(bf_params)

print('Getting best fit model')
maxis_bf_total, phi_bf_total, maxis_bf_fblue, bf_fblue, phi_bf_red, \
    phi_bf_blue, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
    f_red_cen_blue, sat_gals_red_bf, sat_halos_red_bf, sat_gals_blue_bf, \
    sat_halos_blue_bf, f_red_sat_red_bf, f_red_sat_blue_bf, cen_gals_bf, \
    cen_halos_bf = \
    get_best_fit_model(bf_params)

print('Multiprocessing') #~11 minutes
result = mp_init(mcmc_table_pctl, nproc)

print('Plotting')
plot_total_mf(result, total_data, maxis_bf_total, phi_bf_total, 
    bf_chi2, err_data)

plot_fblue(result, f_blue, maxis_bf_fblue, bf_fblue, bf_chi2, err_data)

plot_colour_mf(result, phi_red_data, phi_blue_data, phi_bf_red, phi_bf_blue, 
    err_phi_red, err_phi_blue , bf_chi2)

plot_xmhm(result, cen_gals_bf, cen_halos_bf, bf_chi2)

plot_colour_xmhm(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
    cen_halos_blue, bf_chi2)

plot_red_fraction_cen(result, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
    f_red_cen_blue)

plot_red_fraction_sat(result,sat_gals_red_bf, \
    sat_halos_red_bf, sat_gals_blue_bf, sat_halos_blue_bf, f_red_sat_red_bf, \
    f_red_sat_blue_bf)

plot_zumand_fig4(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
    cen_halos_blue, bf_chi2)
