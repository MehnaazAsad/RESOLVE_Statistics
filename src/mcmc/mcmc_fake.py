"""
{This script carries out an MCMC analysis using fake data to test that 
 the known 5 parameter values are recovered}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import emcee 
import time
import math
import os

__author__ = '{Mehnaaz Asad}'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

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

        eco_buff = reading_catls(path_to_file)
        eco_buff = mock_add_grpcz(eco_buff, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_new.values) / (3 * 10**5)
        
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

    return z_median

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
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
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

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        mstar_total_arr = catl.logmstar.values
        censat_col = 'g_galtype'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values
    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:
        mstar_total_arr = catl.logmstar.values
        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values           
    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mstar_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mstar_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mstar_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mstar_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mstar_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mstar_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mstar_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mstar_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_total_arr = np.log10((10**mstar_total_arr) / 2.041)
        logmstar_cen_arr = np.log10((10**mstar_cen_arr) / 2.041)
        logmstar_sat_arr = np.log10((10**mstar_sat_arr) / 2.041)
    else:
        logmstar_total_arr = mstar_total_arr
        logmstar_cen_arr = mstar_cen_arr
        logmstar_sat_arr = mstar_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 7

        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmstar_total_arr, colour_label_total_arr, blue_frac_helper, bins=bins)
    result_cen = bs(logmstar_cen_arr, colour_label_cen_arr, blue_frac_helper, bins=bins)
    result_sat = bs(logmstar_sat_arr, colour_label_sat_arr, blue_frac_helper, bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool):
    """
    Calculates differential baryonic mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of baryonic masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

    sim_bool: boolean
        True if masses are from mock

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
        logmbary_arr = np.log10((10**mass_arr) / 2.041)
        # print("Data ", logmbary_arr.min(), logmbary_arr.max())
    else:
        logmbary_arr = np.log10(mass_arr)
        # print(logmbary_arr.min(), logmbary_arr.max())
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

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

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

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
        cen_gals = 10**(df.logmstar[df.cs_flag == 1]).reset_index(drop=True)
        sat_gals = 10**(df.logmstar[df.cs_flag == 0]).reset_index(drop=True)
    
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

    phi_total_arr = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

            ## Using best-fit found for new ECO data using result from chain 34
            ## i.e. hybrid quenching model
            Mstar_q = 10.54 # Msun/h
            Mh_q = 14.09 # Msun/h
            mu = 0.77
            nu = 0.17

            ## Using best-fit found for new ECO data using result from chain 35
            ## i.e. halo quenching model
            Mh_qc = 12.29 # Msun/h
            Mh_qs = 12.78 # Msun/h
            mu_c = 1.37
            mu_s = 1.48

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            logmstar_arr = mock_pd.logmstar.values

            #Measure SMF of mock using diff_smf function
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(logmstar_arr, volume, False)
            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
    phi_arr_total = np.array(phi_total_arr)
    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]
    phi_total_4 = phi_arr_total[:,4]
    phi_total_5 = phi_arr_total[:,5]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]
    f_blue_cen_4 = f_blue_cen_arr[:,4]
    f_blue_cen_5 = f_blue_cen_arr[:,5]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]
    f_blue_sat_4 = f_blue_sat_arr[:,4]
    f_blue_sat_5 = f_blue_sat_arr[:,5]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'phi_tot_4':phi_total_4, 'phi_tot_5':phi_total_5,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_cen_4':f_blue_cen_4, 'f_blue_cen_5':f_blue_cen_5,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'f_blue_sat_4':f_blue_sat_4, 'f_blue_sat_5':f_blue_sat_5})

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=15)
    # rc('text', usetex=True)
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral')
    # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$', r'$\Phi_5$', r'$\Phi_6$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$', r'$cen_5$', r'$cen_6$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$', r'$sat_5$', r'$sat_6$',]

    # plt.xticks(tick_marks, names, rotation='vertical')
    # plt.yticks(tick_marks, names)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()
    
    # ## Velocity dispersion
    # bins_red=np.linspace(-2,3,5)
    # bins_blue=np.linspace(-1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # fig2 = plt.figure()
    # for idx in range(len(combined_df.values[:,12:16])):
    #     plt.plot(bins_red, combined_df.values[:,12:16][idx])
    # for idx in range(len(combined_df.values[:,16:20])):
    #     plt.plot(bins_blue, combined_df.values[:,16:20][idx])
    # plt.plot(bins_red, mean_mstar_red_data[0], '--r', lw=3, label='data')
    # plt.plot(bins_blue, mean_mstar_blue_data[0], '--b', lw=3, label='data')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=30)
    # plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()

    # ## Blue fraction from mocks
    # fig3 = plt.figure(figsize=(10,10))
    # for idx in range(len(combined_df.values[:,4:8])):
    #     plt.plot(f_blue[0], combined_df.values[:,4:8][idx], '--')
    # plt.plot(f_blue_data[0], f_blue_data[2], 'k--', lw=3, label='cen data')
    # for idx in range(len(combined_df.values[:,8:12])):
    #     plt.plot(f_blue[0], combined_df.values[:,8:12][idx], '-')
    # plt.plot(f_blue_data[0], f_blue_data[3], 'k-', lw=3, label='sat data')

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=25)
    # plt.title(r'Blue fractions from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()

    # ## SMF from mocks and data
    # fig4 = plt.figure()
    # for idx in range(len(combined_df.values[:,:4])):
    #     plt.plot(max_total, combined_df.values[:,:4][idx], '-')
    # plt.plot(total_data[0], total_data[1], 'k--', lw=3, label='data')

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$')
    # plt.title(r'SMFs from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()


    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral')
    # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
    # plt.colorbar(cax)
    # plt.gca().invert_yaxis() 
    # # # put a blue dot at (10, 20)
    # # plt.scatter([10], [20])

    # # # put a red dot, size 40, at 2 locations:
    # # plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)
    # plt.show()

    # fig1, axes = plt.subplots(12,12, sharex=True, sharey=True, figsize=(10,10))
    # for i, ax_i in enumerate(axes.flatten()):
    #     if i == 12:
    #         break
    #     for j, ax_j in enumerate(axes.flatten()):
    #         if j == 12:
    #             break
    #         elif i == j:
    #             axes[i,j].hist(combined_df[combined_df.columns.values[i]], 
    #                 density=True, color='k')
    #         # else:
    #         #     axes[i,j].scatter(combined_df[combined_df.columns.values[i]], 
    #         #         combined_df[combined_df.columns.values[j]], c='k')
    #         axes[i,j].set_xticks([])
    #         axes[i,j].set_yticks([])
    #         axes[i,j].set_aspect('equal')

    # fig1.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    # fig1, ax_main = plt.subplots(1,1)
    # axes = pd.plotting.scatter_matrix(combined_df, alpha=0.8, diagonal='kde', 
    #     c='k', range_padding=0.1)

    # for i, ax in enumerate(axes.flatten()):
    #     c = plt.cm.Spectral(combined_df.corr().values.flatten()[i])
    #     ax.set_facecolor(c)
    # # ax_main.invert_yaxis()
    # plt.title(r'Total mass function and blue fraction')
    # plt.show()


    # rc('text', usetex=True)
    # rc('text.latex', preamble=r"\usepackage{amsmath}")

    # ## Total SMFs from mocks and data for paper
    # tot_phi_max = np.amax(combined_df.values[:,:6], axis=0)
    # tot_phi_min = np.amin(combined_df.values[:,:6], axis=0)
    # error = np.nanstd(combined_df.values[:,:6], axis=0)

    # fig2 = plt.figure()
    # mt = plt.fill_between(x=max_total, y1=tot_phi_max, 
    #     y2=tot_phi_min, color='silver', alpha=0.4)
    # dt = plt.errorbar(total_data[0], total_data[1], yerr=error,
    #     color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    #     capthick=1.5, zorder=10, marker='^')
    # plt.ylim(-4,-1)

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$')

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='best')
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # plt.savefig('/Users/asadm2/Desktop/total_smf.svg', format='svg', 
    #     bbox_inches="tight", dpi=1200)
    # plt.show()


    return err_colour, corr_mat_inv_colour

def mcmc(nproc, nwalkers, nsteps, phi_total_data, f_blue_cen_data, 
    f_blue_sat_data, err, corr_mat_inv):
    """
    MCMC analysis

    Parameters
    ----------
    nproc: int
        Number of processes to spawn
    
    nwalkers: int
        Number of walkers to use
    
    nsteps: int
        Number of steps to run MCMC for
    
    phi: array
        Array of y-axis values of mass function

    err: array
        Array of error per bin of mass function

    Returns
    ---------
    sampler: multidimensional array
        Result of running emcee 

    """
    ## Starting at best-fit values found in optimize_hybridqm_eco.py
    Mstar_q = 10.49 # Msun/h
    Mh_q = 14.03 # Msun/h
    mu = 0.69
    nu = 0.148

    Mh_qc = 12.61 # Msun/h
    Mh_qs = 13.5 # Msun/h
    mu_c = 0.40
    mu_s = 0.148

    Mhalo_c = 12.35
    Mstar_c = 10.72
    mlow_slope = 0.44
    mhigh_slope = 0.57
    scatter = 0.15

    behroozi10_param_vals = [Mhalo_c, Mstar_c, mlow_slope, mhigh_slope, scatter]

    if quenching == 'hybrid':
        hybrid_param_vals = [Mstar_q, Mh_q, mu, nu]
        all_param_vals = behroozi10_param_vals + hybrid_param_vals

    elif quenching == 'halo':
        halo_param_vals = [Mh_qc, Mh_qs, mu_c, mu_s]
        all_param_vals = behroozi10_param_vals + halo_param_vals

    ndim = len(all_param_vals)

    p0 = all_param_vals + 0.1*np.random.rand(ndim*nwalkers).\
        reshape((nwalkers, ndim))

    filename = "chain.h5"
    backend = emcee.backends.HDFBackend(filename)
    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend,
            args=(phi_total_data, f_blue_cen_data, f_blue_sat_data, 
                err, corr_mat_inv), pool=pool)
        start = time.time()
        sampler.run_mcmc(p0, nsteps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return sampler

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)
    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)
    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged

def lnprob(theta, phi_total_data, f_blue_cen_data, f_blue_sat_data,
    err, corr_mat_inv):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    # cz_inner = 2530 # not starting at corner of box
    cz_inner = 11000 # not starting at corner of box
    cz_outer = 200*H0 

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    warnings.simplefilter("error", (UserWarning, RuntimeWarning))
    try: 
        sim_len = 130
        wedge_radius = 71.62

        gals_df = populate_mock(theta[:5], model_init)
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    
        gals_df['distance_from_upperright'] = np.sqrt((gals_df['x']-sim_len)**2 + \
            gals_df['y']**2 + (gals_df['z']-sim_len)**2)

        gals_df = gals_df.loc[gals_df.distance_from_upperright.values <= wedge_radius]
        gals_df = apply_rsd(gals_df)

        #* Changed cz_inner and cz_outer above to encompass the whole wedge since
        #* 3000 - 12000 would only leave 65 galaxies since the cz range of this
        #* wedge is 11389-19916
        # gals_df = gals_df.loc[\
        #     (gals_df['cz'] >= cz_inner) &
        #     (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
        
        gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
            gals_df['halo_id'], 1, 0)

        cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
            'stellar_mass', 'ra', 'dec', 'cz']
        gals_df = gals_df[cols_to_use]

        gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

        gals_df['logmstar'] = np.log10(gals_df['logmstar'])

        if quenching == 'hybrid':
            f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
                'vishnu')
        elif quenching == 'halo':
            f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
                'vishnu')
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, \
            gals_df)

        gal_group_df = group_finding(gals_df,
            path_to_data + 'interim/', param_dict)

        # cz_inner_mod = 3000
        # gal_group_df = gal_group_df.loc[\
        #     (gal_group_df['cz'] >= cz_inner_mod) &
        #     (gal_group_df['cz'] <= cz_outer)].reset_index(drop=True)

        # v_sim = 130**3
        # v_sim = 890641.5172927063 #survey volume used in group_finder.py

        ## Observable #1 - Total SMF
        total_model = measure_all_smf(gal_group_df, survey_vol, False)  
        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        model_arr = []
        model_arr.append(total_model[1])
        model_arr.append(f_blue[2])   
        model_arr.append(f_blue[3])
        err_arr = err

        data_arr, model_arr = np.array(data_arr), np.array(model_arr)
        chi2 = chi_squared(data_arr, model_arr, err_arr, corr_mat_inv)
        lnp = -chi2 / 2

        if math.isnan(lnp):
            raise ValueError
    except (ValueError, RuntimeWarning, UserWarning):
        lnp = -np.inf
        chi2 = np.inf

    return lnp, chi2

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def args_parser():
    """
    Parsing arguments passed to script

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
    parser.add_argument('mf_type', type=str, \
        help='Options: smf/bmf')
    parser.add_argument('quenching', type=str, \
        help='Options: hybrid/halo')
    parser.add_argument('nproc', type=int, nargs='?', 
        help='Number of processes')
    parser.add_argument('nwalkers', type=int, nargs='?', 
        help='Number of walkers')
    parser.add_argument('nsteps', type=int, nargs='?', help='Number of steps')
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
    global mf_type
    global quenching
    global path_to_data
    global level

    rseed = 12
    np.random.seed(rseed)
    survey = args.survey
    machine = args.machine
    nproc = args.nproc
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    mf_type = args.mf_type
    quenching = args.quenching
    level = "group"
    
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_external = dict_of_paths['ext_dir']
    path_to_data = dict_of_paths['data_dir']
    
    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    if survey == 'eco':
        catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

    if survey == 'eco':
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    elif survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    elif survey == 'resolveb':
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

    z_median = read_data_catl(catl_file, survey)

    ## For hybrid using values from chain 32 contour plot
    if quenching == 'hybrid':
        theta = np.array([12.31, 10.605, 0.411, 0.50, 0.335, 10.071, 13.89, 0.498, 0.19])
    model_init_data = halocat_init(halo_catalog, z_median)
    catl = populate_mock(theta[:5], model_init_data)
    catl = catl.loc[catl['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    ### x,y,z = 0,0,0
    catl['distance_from_lowerleft'] = np.sqrt(catl['x']**2 + \
        catl['y']**2 + catl['z']**2)

    sim_len = 130
    # Distance where wedge volume (1/8 sphere volume) = eco volume with buffer
    wedge_radius = 71.62
    ## Volume of eco survey with buffer 
    volume = 192351.6
    cz_inner = 2530
    cz_outer = 7470
    catl = catl.loc[catl.distance_from_lowerleft.values <= wedge_radius]
    catl = apply_rsd(catl)

    catl = catl.loc[\
        (catl['cz'] >= cz_inner) &
        (catl['cz'] <= cz_outer)].reset_index(drop=True)

    catl['cs_flag'] = np.where(catl['halo_hostid'] == \
        catl['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    catl = catl[cols_to_use]

    catl.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    catl['logmstar'] = np.log10(catl['logmstar'])

    print('Assigning colour to data')
    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], catl, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], catl, \
            'vishnu')
    catl = assign_colour_label_mock(f_red_cen, f_red_sat, catl)

    eco = {
        'c': 3*10**5,
        'survey_vol': volume,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gal_group_df = group_finding(catl,
        path_to_data + 'interim/', param_dict)

    cz_inner_mod = 3000
    cz_outer_mod = 7000
    volume_mod = 151829.26
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['cz'] >= cz_inner_mod) &
        (gal_group_df['cz'] <= cz_outer_mod)].reset_index(drop=True)

    print('Measuring SMF for data')
    total_data = measure_all_smf(gal_group_df, volume_mod, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(gal_group_df, True, False)

    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Measuring error in data from mocks')
    sigma, corr_mat_inv = get_err_data(survey, path_to_mocks)

    print('Error in data: \n', sigma)
    print('------------- \n')
    print('Inverse of covariance matrix: \n', corr_mat_inv)
    print('------------- \n')
    print('SMF total data: \n', total_data[1])
    print('------------- \n')
    print('Blue frac cen data: \n', f_blue_data[2])
    print('Blue frac sat data: \n', f_blue_data[3])
    print('------------- \n')

    print('Running MCMC')
    sampler = mcmc(nproc, nwalkers, nsteps, total_data[1],
        f_blue_data[2], f_blue_data[3], sigma, corr_mat_inv)

    print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))
    print("Mean autocorrelation time: {0:.3f} steps".format(
        np.mean(sampler.get_autocorr_time())))

# Main function
if __name__ == '__main__':
    args = args_parser()

    main(args)

###################################TEST#########################################
def test():
    import matplotlib.pyplot as plt

    theta = np.array([12.42877028, 10.7496672 ,  0.49743833,  0.59420229,  0.16614364,
        10.49855279, 14.12568771,  0.70864637,  0.18365174])
    sim_len = 130
    wedge_radius = 71.62

    gals_df = populate_mock(theta[:5], model_init)
    gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    ### x,y,z = 0,0,0
    gals_df['distance_from_lowerleft'] = np.sqrt(gals_df['x']**2 + \
        gals_df['y']**2 + gals_df['z']**2)
    ### x,y,z = 130,0,130
    gals_df['distance_from_upperright'] = np.sqrt((gals_df['x']-sim_len)**2 + \
        gals_df['y']**2 + (gals_df['z']-sim_len)**2)

    gals_df_lowerleft = gals_df.loc[gals_df.distance_from_lowerleft.values <= wedge_radius]
    gals_df_upperright = gals_df.loc[gals_df.distance_from_upperright.values <= wedge_radius]


    xdata = gals_df_lowerleft.x.values
    ydata = gals_df_lowerleft.y.values
    zdata = gals_df_lowerleft.z.values

    xdata_invert = gals_df_upperright.x.values
    ydata_invert = gals_df_upperright.y.values
    zdata_invert = gals_df_upperright.z.values

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.axes.set_xlim3d(left=0, right=sim_len)
    ax.axes.set_ylim3d(bottom=0, top=sim_len)
    ax.axes.set_zlim3d(bottom=0, top=sim_len)

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='coolwarm')
    ax.scatter3D(xdata_invert, ydata_invert, zdata_invert, c=zdata_invert, cmap='coolwarm')
    plt.show()
