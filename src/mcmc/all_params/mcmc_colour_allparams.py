"""
{This script carries out an MCMC analysis and varies Behroozi parameters as well 
 as quenching parameters simultaneously. Observables used are total SMF and blue
 fraction of galaxies both in bins of stellar mass.}
"""

# Built-in/Generic Imports
import multiprocessing
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import h5py
import os

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def models_add_avgrpcz(df, grpid_col=None, galtype_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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
        #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
        eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def assign_colour_label_data_legacy(catl):
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

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
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
        max_total, phi_total, err_total, counts_total = \
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
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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
            #* Changed max bin from 11.5 to 11.1 to be the same as mstar-sigma (10.8)
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
            bin_num = 5

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

    return maxis, phi, err_tot, counts

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
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary_a23.values
            mass_cen_arr = catl.logmbary_a23.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary_a23.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
            #* Changed max bin from 11.5 to 11.1 to be the same as mstar-sigma (10.8)
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)


        elif mf_type == 'bmf':
            mbary_limit = 9.3
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        print(catl.logmbary_a23.min())

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
            
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        print(catl.logmbary_a23.min())

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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
    else:
        logmbary_arr = np.log10(mass_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.3) / 2.041), 1)

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.default_rng(df['galid'][ii])
        rfloat = rng.uniform()
        # Comparing against f_red
        if (rfloat >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
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
        # Both cen and sat are the same mass for a halo i.e. the satellites
        # are assigned a halo mass of the central. 
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

def get_err_data_legacy(survey, path):
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
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
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

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                # mock_pd.ps_grp_censat.loc[mock_pd.index == final_sat_idx] = 0
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            # ## Using best-fit found for new ECO data using result from chain 67
            # ## i.e. hybrid quenching model
            # bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
            #     0.00722556715]

            # ## 75
            # bf_from_last_chain = [10.215486, 13.987752, 0.753758, 0.025111]
            
            ## Using best-fit found for new ECO data using result from chain 59
            ## i.e. hybrid quenching model which was the last time sigma-M* was
            ## used i.e. stacked_stat = True
            bf_from_last_chain = [10.133745, 13.478087, 0.810922,
                0.043523]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 62
            ## i.e. halo quenching model
            bf_from_last_chain = [11.749165, 12.471502, 1.496924, 0.416936]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            # Copied from chain 67
            min_chi2_params = [10.194299, 14.545483, 0.708014, 0.007226] #chi2=35
            max_chi2_params = [10.582321, 14.119958, 0.745622, 0.233953] #chi2=1839
            
            # From chain 61
            min_chi2_params = [10.143908, 13.630966, 0.825897, 0.042685] #chi2=13.66

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are eigenvalues #confirmed by comparing s to 
        ## output of np.linalg.eig(C) where the first array is array of 
        ## eigenvalues.
        ## https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt one more time 
        # to be able to compare directly to values in Sigma since eigenvalues 
        # are squares of singular values 
        # (http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf)
        # https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm#:~:text=The%20SVD%20represents%20an%20expansion,up%20the%20columns%20of%20U.
        max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        VT = VT[:n_elements, :]
        ## reconstruct
        sigma_mat = sigma_mat[:, :n_elements]
        B = U.dot(sigma_mat.dot(VT))
        # print(B)
        ## transform 2 ways (this is how you would transform data, model and sigma
        ## i.e. new_data = data.dot(Sigma) etc.) <- not sure this is correct
        #* Both lines below are equivalent transforms of the matrix BUT to project 
        #* data, model, err in the new space, you need the eigenvectors NOT the 
        #* eigenvalues (sigma_mat) as was used earlier. The eigenvectors are 
        #* VT.T (the right singular vectors) since those have the reduced 
        #* dimensions needed for projection. Data and model should then be 
        #* projected similarly to err_colour below. 
        #* err_colour.dot(sigma_mat) no longer makes sense since that is 
        #* multiplying the error with the eigenvalues. Using sigma_mat only
        #* makes sense in U.dot(sigma_mat) since U and sigma_mat were both derived 
        #* by doing svd on the matrix which is what you're trying to get in the 
        #* new space by doing U.dot(sigma_mat). 
        #* http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        err_colour_pca = err_colour.dot(VT.T)
        eigenvectors = VT.T

        ## Same as err_colour.dot(sigma_mat)
        # err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    from matplotlib.legend_handler import HandlerTuple
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import cm

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
    rc('text', usetex=True)
    rc('text.latex', preamble=r"\usepackage{amsmath}")
    rc('axes', linewidth=2)
    rc('xtick.major', width=4, size=7)
    rc('ytick.major', width=4, size=7)
    rc('xtick.minor', width=2, size=7)
    rc('ytick.minor', width=2, size=7)

    # #* Reduced feature space
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # # #* Reconstructed post-SVD (sub corr_mat_colour for original matrix)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(corr_mat_colour, cmap=cmap, vmin=-1, vmax=1)
    # # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]
    
    # tick_marks=[0, 4, 8, 12, 16]
    # names = [
    #     r"$\boldsymbol\phi$",
    #     r"$\boldsymbol{f_{blue}^{c}}$",
    #     r"$\boldsymbol{f_{blue}^{s}}$",
    #     r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
    #     r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

    # plt.xticks(tick_marks, names, fontsize=10)#, rotation='vertical')
    # plt.yticks(tick_marks, names, fontsize=10)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()
    # plt.savefig('/Users/asadm2/Desktop/matrix_eco_smf.pdf')


    # #* Scree plot

    # percentage_variance = []
    # for val in s:
    #     sum_of_eigenvalues = np.sum(s)
    #     percentage_variance.append((val/sum_of_eigenvalues)*100)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.bar(np.arange(1, len(s)+1, 1), percentage_variance, color='#663399',
    #     zorder=5)
    # ax1.scatter(np.arange(1, len(s)+1, 1), s, c='orange', s=50, zorder=10)
    # ax1.plot(np.arange(1, len(s)+1, 1), s, 'k--', zorder=10)
    # ax1.hlines(max_eigen, 0, 20, colors='orange', zorder=10, lw=2)
    # ax1.set_xlabel('Component number')
    # ax1.set_ylabel('Singular values')
    # ax2.set_ylabel('Percentage of variance')
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.set_frame_on(False)
    # ax1.set_xticks(np.arange(1, len(s)+1, 1))
    # plt.show()

    ############################################################################
    # #* Observable plots for paper
    # #* Total SMFs from mocks and data for paper

    data = combined_df.values[:,:4]
    max_total = total_data[0]

    upper_bound = np.nanmean(data, axis=0) + \
        np.nanstd(data, axis=0)
    lower_bound = np.nanmean(data, axis=0) - \
        np.nanstd(data, axis=0)

    phi_max = []
    phi_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data.T[idx]
            >=lower_bound[idx], data.T[idx]<=upper_bound[idx]))
        nums = data.T[idx][idxs]
        phi_min.append(min(nums))
        phi_max.append(max(nums))

    phi_max = np.array(phi_max)
    phi_min = np.array(phi_min)

    midpoint_phi_err = -((phi_max * phi_min) ** 0.5)
    phi_data_midpoint_diff = 10**total_data[1] - 10**midpoint_phi_err 

    fig1 = plt.figure()

    mt = plt.fill_between(x=max_total, y1=np.log10(10**phi_min + phi_data_midpoint_diff), 
        y2=np.log10(10**phi_max + phi_data_midpoint_diff), color='silver', alpha=0.4)

    # mt = plt.fill_between(x=max_total, y1=phi_max, 
    #     y2=phi_min, color='silver', alpha=0.4)
    dt = plt.scatter(total_data[0], total_data[1],
        color='k', s=150, zorder=10, marker='^')

    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)

    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

    plt.legend([(dt), (mt)], ['ECO','Mocks'],
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
    plt.minorticks_on()
    # plt.title(r'SMFs from mocks')
    if mf_type == 'smf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_smf_total_offsetmocks.pdf', 
            bbox_inches="tight", dpi=1200)
    elif mf_type == 'bmf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_bmf_total.pdf', 
            bbox_inches="tight", dpi=1200)
    plt.show()

    # # #* Blue fraction from mocks and data for paper

    data_cen = combined_df.values[:,4:8]
    data_sat = combined_df.values[:,8:12]

    upper_bound = np.nanmean(data_cen, axis=0) + \
        np.nanstd(data_cen, axis=0)
    lower_bound = np.nanmean(data_cen, axis=0) - \
        np.nanstd(data_cen, axis=0)

    cen_max = []
    cen_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_cen.T[idx]
            >=lower_bound[idx], data_cen.T[idx]<=upper_bound[idx]))
        nums = data_cen.T[idx][idxs]
        cen_min.append(min(nums))
        cen_max.append(max(nums))

    upper_bound = np.nanmean(data_sat, axis=0) + \
        np.nanstd(data_sat, axis=0)
    lower_bound = np.nanmean(data_sat, axis=0) - \
        np.nanstd(data_sat, axis=0)

    sat_max = []
    sat_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_sat.T[idx]
            >=lower_bound[idx], data_sat.T[idx]<=upper_bound[idx]))
        nums = data_sat.T[idx][idxs]
        sat_min.append(min(nums))
        sat_max.append(max(nums))

    cen_max = np.array(cen_max)
    cen_min = np.array(cen_min)
    sat_max = np.array(sat_max)
    sat_min = np.array(sat_min)

    midpoint_cen_err = (cen_max * cen_min) ** 0.5
    midpoint_sat_err = (sat_max * sat_min) ** 0.5
    cen_data_midpoint_diff = f_blue_data[2] - midpoint_cen_err 
    sat_data_midpoint_diff = f_blue_data[3] - midpoint_sat_err

    fig2 = plt.figure()

    #* Points not truly in the middle like in next figure. They're just shifted.
    # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + (f_blue_data[2] - cen_max), 
    #     y2=cen_min + (f_blue_data[2] - cen_min) - (np.array(cen_max) - np.array(cen_min)), color='rebeccapurple', alpha=0.4)
    # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + (f_blue_data[3] - sat_max), 
    #     y2=sat_min + (f_blue_data[3] - sat_min) - (np.array(sat_max) - np.array(sat_min)), color='goldenrod', alpha=0.4)

    mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + cen_data_midpoint_diff, 
        y2=cen_min + cen_data_midpoint_diff, color='rebeccapurple', alpha=0.4)
    mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + sat_data_midpoint_diff, 
        y2=sat_min + sat_data_midpoint_diff, color='goldenrod', alpha=0.4)

    dt_cen = plt.scatter(f_blue_data[0], f_blue_data[2],
        color='rebeccapurple', s=150, zorder=10, marker='^')
    dt_sat = plt.scatter(f_blue_data[0], f_blue_data[3],
        color='goldenrod', s=150, zorder=10, marker='^')

    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
    plt.ylim(0,1)
    # plt.title(r'Blue fractions from mocks and data')
    plt.legend([dt_cen, dt_sat, mt_cen, mt_sat], 
        ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
    plt.minorticks_on()
    if mf_type == 'smf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_offsetmocks.pdf', 
            bbox_inches="tight", dpi=1200)
    elif mf_type == 'bmf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bary.pdf', 
            bbox_inches="tight", dpi=1200)
    plt.show()


    # #* Velocity dispersion from mocks and data for paper

    bins_red=np.linspace(1,2.8,5)
    bins_blue=np.linspace(1,2.5,5)
    bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    data_red = combined_df.values[:,12:16]
    data_blue = combined_df.values[:,16:20]

    upper_bound = np.nanmean(data_red, axis=0) + \
        np.nanstd(data_red, axis=0)
    lower_bound = np.nanmean(data_red, axis=0) - \
        np.nanstd(data_red, axis=0)

    red_max = []
    red_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_red.T[idx]
            >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
        nums = data_red.T[idx][idxs]
        red_min.append(min(nums))
        red_max.append(max(nums))

    upper_bound = np.nanmean(data_blue, axis=0) + \
        np.nanstd(data_blue, axis=0)
    lower_bound = np.nanmean(data_blue, axis=0) - \
        np.nanstd(data_blue, axis=0)

    blue_max = []
    blue_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_blue.T[idx]
            >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
        nums = data_blue.T[idx][idxs]
        blue_min.append(min(nums))
        blue_max.append(max(nums))

    red_max = np.array(red_max)
    red_min = np.array(red_min)
    blue_max = np.array(blue_max)
    blue_min = np.array(blue_min)

    midpoint_red_err = (red_max * red_min) ** 0.5
    midpoint_blue_err = (blue_max * blue_min) ** 0.5
    red_data_midpoint_diff = 10**mean_mstar_red_data[0] - 10**midpoint_red_err 
    blue_data_midpoint_diff = 10**mean_mstar_blue_data[0] - 10**midpoint_blue_err

    fig3 = plt.figure()
    mt_red = plt.fill_between(x=bins_red, y1=np.log10(np.abs(10**red_min + red_data_midpoint_diff)), 
        y2=np.log10(10**red_max + red_data_midpoint_diff), color='indianred', alpha=0.4)
    mt_blue = plt.fill_between(x=bins_blue, y1=np.log10(10**blue_min + blue_data_midpoint_diff), 
        y2=np.log10(10**blue_max + blue_data_midpoint_diff), color='cornflowerblue', alpha=0.4)

    dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
        color='indianred', s=150, zorder=10, marker='^')
    dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
        color='cornflowerblue', s=150, zorder=10, marker='^')

    plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    if mf_type == 'smf':
        plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    elif mf_type == 'bmf':
        plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # plt.title(r'Velocity dispersion from mocks and data')
    plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
        ['ECO','Mocks'],
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
        prop={'size':20})
    plt.minorticks_on()
    if mf_type == 'smf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_avmstar_offsetmocks.pdf', 
            bbox_inches="tight", dpi=1200)
    elif mf_type == 'bmf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_avmstar_bary.pdf', 
            bbox_inches="tight", dpi=1200)

    plt.show()
    ############################################################################
    ## Stacked sigma from mocks and data for paper
    if mf_type == 'smf':
        bin_min = 8.6
    elif mf_type == 'bmf':
        bin_min = 9.1
    bins_red=np.linspace(bin_min,10.8,5)
    bins_blue=np.linspace(bin_min,10.8,5)
    bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    data_red = combined_df.values[:,20:24]
    data_blue = combined_df.values[:,24:28]

    upper_bound = np.nanmean(data_red, axis=0) + \
        np.nanstd(data_red, axis=0)
    lower_bound = np.nanmean(data_red, axis=0) - \
        np.nanstd(data_red, axis=0)

    red_max = []
    red_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_red.T[idx]
            >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
        nums = data_red.T[idx][idxs]
        red_min.append(min(nums))
        red_max.append(max(nums))

    upper_bound = np.nanmean(data_blue, axis=0) + \
        np.nanstd(data_blue, axis=0)
    lower_bound = np.nanmean(data_blue, axis=0) - \
        np.nanstd(data_blue, axis=0)

    blue_max = []
    blue_min = []
    for idx in range(len(upper_bound)):
        idxs = np.where(np.logical_and(data_blue.T[idx]
            >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
        nums = data_blue.T[idx][idxs]
        blue_min.append(min(nums))
        blue_max.append(max(nums))

    red_max = np.array(red_max)
    red_min = np.array(red_min)
    blue_max = np.array(blue_max)
    blue_min = np.array(blue_min)

    midpoint_red_err = (red_max * red_min) ** 0.5
    midpoint_blue_err = (blue_max * blue_min) ** 0.5
    red_data_midpoint_diff = 10**sigma_red_data - 10**midpoint_red_err
    blue_data_midpoint_diff = 10**sigma_blue_data - 10**midpoint_blue_err

    fig3 = plt.figure()
    mt_red = plt.fill_between(x=bins_red, y1=np.log10(np.abs(10**red_min + red_data_midpoint_diff)), 
        y2=np.log10(10**red_max + red_data_midpoint_diff), color='indianred', alpha=0.4)
    mt_blue = plt.fill_between(x=bins_blue, y1=np.log10(10**blue_min + blue_data_midpoint_diff), 
        y2=np.log10(10**blue_max + blue_data_midpoint_diff), color='cornflowerblue', alpha=0.4)

    dt_red = plt.scatter(bins_red, sigma_red_data, 
        color='indianred', s=150, zorder=10, marker='^')
    dt_blue = plt.scatter(bins_blue, sigma_blue_data,
        color='cornflowerblue', s=150, zorder=10, marker='^')

    plt.ylabel(r'\boldmath$\overline{\log_{10}\ \sigma} \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
        ['ECO','Mocks'],
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', 
        prop={'size':20})
    plt.minorticks_on()
    if mf_type == 'smf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_offsetmocks.pdf', 
            bbox_inches="tight", dpi=1200)
    elif mf_type == 'bmf':
        plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_bary.pdf', 
            bbox_inches="tight", dpi=1200)

    plt.show()

    if pca:
        return err_colour_pca, eigenvectors, n_elements
    else:
        return err_colour, corr_mat_inv_colour

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

def get_err_data(path_to_proc):
    # Read in datasets from h5 file and calculate corr matrix
    if stacked_stat == 'both':
        hf_read = h5py.File(path_to_proc + 'corr_matrices_28stats_{0}_{1}.h5'.
            format(quenching, mf_type), 'r')
        hf_read.keys()
        smf = hf_read.get('smf')
        smf = np.squeeze(np.array(smf))
        fblue_cen = hf_read.get('fblue_cen')
        fblue_cen = np.array(fblue_cen)
        fblue_sat = hf_read.get('fblue_sat')
        fblue_sat = np.array(fblue_sat)
        mean_mstar_red = hf_read.get('mean_mstar_red')
        mean_mstar_red = np.array(mean_mstar_red)
        mean_mstar_blue = hf_read.get('mean_mstar_blue')
        mean_mstar_blue = np.array(mean_mstar_blue)
        sigma_red = hf_read.get('sigma_red')
        sigma_red = np.array(sigma_red)
        sigma_blue = hf_read.get('sigma_blue')
        sigma_blue = np.array(sigma_blue)

    elif stacked_stat:
        hf_read = h5py.File(path_to_proc + 'corr_matrices_xy_mstarsigma_{0}.h5'.format(quenching), 'r')
        hf_read.keys()
        smf = hf_read.get('smf')
        smf = np.array(smf)
        fblue_cen = hf_read.get('fblue_cen')
        fblue_cen = np.array(fblue_cen)
        fblue_sat = hf_read.get('fblue_sat')
        fblue_sat = np.array(fblue_sat)
        mean_mstar_red = hf_read.get('mean_sigma_red')
        mean_mstar_red = np.array(mean_mstar_red)
        mean_mstar_blue = hf_read.get('mean_sigma_blue')
        mean_mstar_blue = np.array(mean_mstar_blue)

    else:
        hf_read = h5py.File(path_to_proc + 'corr_matrices_{0}.h5'.format(quenching), 'r')
        hf_read.keys()
        smf = hf_read.get('smf')
        smf = np.array(smf)
        fblue_cen = hf_read.get('fblue_cen')
        fblue_cen = np.array(fblue_cen)
        fblue_sat = hf_read.get('fblue_sat')
        fblue_sat = np.array(fblue_sat)
        mean_mstar_red = hf_read.get('mean_mstar_red')
        mean_mstar_red = np.array(mean_mstar_red)
        mean_mstar_blue = hf_read.get('mean_mstar_blue')
        mean_mstar_blue = np.array(mean_mstar_blue)

    for i in range(100):
        phi_total_0 = smf[i][:,0]
        phi_total_1 = smf[i][:,1]
        phi_total_2 = smf[i][:,2]
        phi_total_3 = smf[i][:,3]

        f_blue_cen_0 = fblue_cen[i][:,0]
        f_blue_cen_1 = fblue_cen[i][:,1]
        f_blue_cen_2 = fblue_cen[i][:,2]
        f_blue_cen_3 = fblue_cen[i][:,3]

        f_blue_sat_0 = fblue_sat[i][:,0]
        f_blue_sat_1 = fblue_sat[i][:,1]
        f_blue_sat_2 = fblue_sat[i][:,2]
        f_blue_sat_3 = fblue_sat[i][:,3]

        mstar_red_cen_0 = mean_mstar_red[i][:,0]
        mstar_red_cen_1 = mean_mstar_red[i][:,1]
        mstar_red_cen_2 = mean_mstar_red[i][:,2]
        mstar_red_cen_3 = mean_mstar_red[i][:,3]

        mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
        mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
        mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
        mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

        sigma_red_0 = sigma_red[i][:,0]
        sigma_red_1 = sigma_red[i][:,1]
        sigma_red_2 = sigma_red[i][:,2]
        sigma_red_3 = sigma_red[i][:,3]

        sigma_blue_0 = sigma_blue[i][:,0]
        sigma_blue_1 = sigma_blue[i][:,1]
        sigma_blue_2 = sigma_blue[i][:,2]
        sigma_blue_3 = sigma_blue[i][:,3]

        combined_df = pd.DataFrame({
            'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
            'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
            'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
            'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
            'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
            'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
            'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
            'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
            'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
            'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3,
            'sigma_red_0':sigma_red_0, 'sigma_red_1':sigma_red_1, 
            'sigma_red_2':sigma_red_2, 'sigma_red_3':sigma_red_3,
            'sigma_blue_0':sigma_blue_0, 'sigma_blue_1':sigma_blue_1, 
            'sigma_blue_2':sigma_blue_2, 'sigma_blue_3':sigma_blue_3})

        if i == 0:
            # Correlation matrix of phi and deltav colour measurements combined
            corr_mat_global = combined_df.corr()
            cov_mat_global = combined_df.cov()

            corr_mat_average = corr_mat_global
            cov_mat_average = cov_mat_global
        else:
            corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
            cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
            

    # Using average cov mat to get correlation matrix
    corr_mat_average = calc_corr_mat(cov_mat_average)
    corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
    sigma_average = np.sqrt(np.diag(cov_mat_average))

    if pca:
        num_mocks = 64
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_average)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_average.shape[0], corr_mat_average.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_average.shape[0], :corr_mat_average.shape[0]] = diag(s)

        ## values in s are eigenvalues #confirmed by comparing s to 
        ## output of np.linalg.eig(C) where the first array is array of 
        ## eigenvalues.
        ## https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt one more time 
        # to be able to compare directly to values in Sigma since eigenvalues 
        # are squares of singular values 
        # (http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf)
        # https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm#:~:text=The%20SVD%20represents%20an%20expansion,up%20the%20columns%20of%20U.
        min_eigen = np.sqrt(np.sqrt(2/(num_mocks)))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* min_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>min_eigen])
        VT = VT[:n_elements, :]

        print("Number of principle components kept: {0}".format(n_elements))

        ## reconstruct
        # sigma_mat = sigma_mat[:, :n_elements]
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        ## transform 2 ways (this is how you would transform data, model and sigma
        ## i.e. new_data = data.dot(Sigma) etc.) <- not sure this is correct
        #* Both lines below are equivalent transforms of the matrix BUT to project 
        #* data, model, err in the new space, you need the eigenvectors NOT the 
        #* eigenvalues (sigma_mat) as was used earlier. The eigenvectors are 
        #* VT.T (the right singular vectors) since those have the reduced 
        #* dimensions needed for projection. Data and model should then be 
        #* projected similarly to err_colour below. 
        #* err_colour.dot(sigma_mat) no longer makes sense since that is 
        #* multiplying the error with the eigenvalues. Using sigma_mat only
        #* makes sense in U.dot(sigma_mat) since U and sigma_mat were both derived 
        #* by doing svd on the matrix which is what you're trying to get in the 
        #* new space by doing U.dot(sigma_mat). 
        #* http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        err_colour_pca = sigma_average.dot(VT.T)
        eigenvectors = VT.T

    if pca:
        return err_colour_pca, eigenvectors
    else:
        return sigma_average, corr_mat_inv_colour_average

def mcmc(nproc, nwalkers, nsteps, data, err, corr_mat_inv):
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

    ## Starting at best-fit parameters from middle column of table 2
    ## https://arxiv.org/pdf/1001.0015.pdf
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
    
    if pca:
        filename = "chain_{0}_pca.h5".format(quenching)
    else:
        filename = "chain_{0}.h5".format(quenching)
    # filename = "memtest.h5"

    if not new_chain:
        print("Resuming chain...")
        new_backend = emcee.backends.HDFBackend(filename)
        with Pool(processes=nproc) as pool:
            new_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                backend=new_backend, args=(data, err, corr_mat_inv), pool=pool)
            start = time.time()
            new_sampler.run_mcmc(None, nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    else:
        print("Starting new chain...")
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        with Pool(processes=nproc) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                backend=backend, args=(data, err, corr_mat_inv), pool=pool)
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

    model.mock.populate(seed=1993)

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
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def lnprob(theta, data, err, corr_mat_inv):
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
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

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
        gals_df = populate_mock(theta[:5], model_init)
        if mf_type == 'smf':
            gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
        elif mf_type == 'bmf':
            gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.0].reset_index(drop=True)
        
        gals_df = apply_rsd(gals_df)

        gals_df = gals_df.loc[\
            (gals_df['cz'] >= cz_inner) &
            (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
        
        gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
            gals_df['halo_id'], 1, 0)

        cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
            'stellar_mass', 'ra', 'dec', 'cz', 'galid']
        gals_df = gals_df[cols_to_use]

        gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

        gals_df['logmstar'] = np.log10(gals_df['logmstar'])


        if quenching == 'hybrid':
            f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
                'vishnu')
        elif quenching == 'halo':
            f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
                'vishnu')
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

        # npmax = 1e5
        # if len(gals_df) >= npmax:
        #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
        #     print("(test) WARNING! Increasing memory allocation\n")
        #     npmax*=1.2

        gal_group_df = group_finding(gals_df,
            path_to_data + 'interim/', param_dict)

        ## Pair splitting
        psgrpid = split_false_pairs(
            np.array(gal_group_df.ra),
            np.array(gal_group_df.dec),
            np.array(gal_group_df.cz), 
            np.array(gal_group_df.groupid))

        gal_group_df["ps_groupid"] = psgrpid

        arr1 = gal_group_df.ps_groupid
        arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
        arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
        mapping = dict(zip(arr1_unq, arr2_unq))   
        new_values = arr1.map(mapping)
        gal_group_df['ps_groupid'] = new_values  

        most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
            .transform(max) == gal_group_df['logmstar']        
        grp_censat_new = most_massive_gal_idxs.astype(int)
        gal_group_df["ps_grp_censat"] = grp_censat_new

        gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
            galtype_col='ps_grp_censat', cen_cz_col='cz')

        ## Making a similar cz cut as in data which is based on grpcz being 
        ## defined as cz of the central of the group "grpcz_cen"
        cz_inner_mod = 3000
        gal_group_df = gal_group_df.loc[\
            (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
            (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

        dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
        dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

        v_inner = vol_sphere(dist_inner)
        v_outer = vol_sphere(dist_outer)

        v_sphere = v_outer-v_inner
        survey_vol = v_sphere/8

        # v_sim = 130**3
        # v_sim = 890641.5172927063 #survey volume used in group_finder.py

        ## Observable #1 - Total SMF
        if mf_type == 'smf':
            total_model = measure_all_smf(gal_group_df, survey_vol, False) 

            ## Observable #2 - Blue fraction
            f_blue = blue_frac(gal_group_df, True, False)
        
            ## Observable #3 
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,10.8,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,10.8,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])

            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))

        elif mf_type == 'bmf':
            logmstar_col = 'logmstar' #No explicit logmbary column
            total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
                survey_vol, True) 

            ## Observable #2 - Blue fraction
            f_blue = blue_frac(gal_group_df, True, False)
        
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.0,11.2,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.0,11.2,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])

            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
 
        model_arr = []
        model_arr.append(total_model[1])
        model_arr.append(f_blue[2])   
        model_arr.append(f_blue[3])
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)

        model_arr = np.array(model_arr)

        if pca:
            chi2 = chi_squared_pca(data, model_arr, err, corr_mat_inv)
        else:
            chi2 = chi_squared(data, model_arr, err, corr_mat_inv)
    
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

def chi_squared_pca(data, model, err_data, mat):
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
    model = model.flatten() # flatten from (5,4) to (1,20)

    # print('data: \n', data)
    # print('model: \n', model)

    model = model.dot(mat)

    #Error is already transformed in get_err_data() and data is already 
    #transformed in main()
    chi_squared_arr = (data - model)**2 / (err_data**2)
    chi_squared = np.sum(chi_squared_arr)
    # print("chi-squared: ", chi_squared)
    
    return chi_squared

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
    global stacked_stat
    global pca
    global new_chain

    rseed = 12
    np.random.seed(rseed)
    level = "group"
    stacked_stat = "both"
    pca = False
    new_chain = True

    survey = args.survey
    machine = args.machine
    nproc = args.nproc
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    mf_type = args.mf_type
    quenching = args.quenching
    
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
        if mf_type == 'smf':
            catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
        elif mf_type == 'bmf':
            catl_file = path_to_proc + \
                "gal_group_eco_bary_buffer_volh1_dr3.hdf5"    
        else:
            print("Incorrect mass function chosen")
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"
    
    if survey == 'eco':
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    elif survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    elif survey == 'resolveb':
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary_a23.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if mf_type == 'smf':
        print('Measuring stacked velocity dispersion for data')
        red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

        sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        
        sigma_red_data = np.log10(sigma_red_data[0])
        sigma_blue_data = np.log10(sigma_blue_data[0])
    elif mf_type == 'bmf':
        print('Measuring stacked velocity dispersion for data')
        red_deltav, red_cen_mbary_sigma, blue_deltav, \
            blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

        sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        
        sigma_red_data = np.log10(sigma_red_data[0])
        sigma_blue_data = np.log10(sigma_blue_data[0])

    if mf_type == 'smf':
        print('Measuring velocity dispersion for data')
        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))

    elif mf_type == 'bmf':
        print('Measuring velocity dispersion for data')
        red_sigma, red_cen_mbary_sigma, blue_sigma, \
            blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))


    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Measuring error in data from mocks')
    sigma, mat = get_err_data(path_to_proc)

    print('Error in data: \n', sigma)
    print('------------- \n')
    print('Matrix: \n', mat)
    print('------------- \n')
    if mf_type == 'smf':
        print('SMF total data: \n', total_data[1])
    elif mf_type == 'bmf':
        print('BMF total data: \n', total_data[1])       
    print('------------- \n')
    print('Blue frac cen data: \n', f_blue_data[2])
    print('Blue frac sat data: \n', f_blue_data[3])
    print('------------- \n')

    print('Dispersion red data: \n', sigma_red_data)
    print('Dispersion blue data: \n', sigma_blue_data)
    print('Mean M* red data: \n', mean_mstar_red_data[0])
    print('Mean M* blue data: \n', mean_mstar_blue_data[0])
    print('------------- \n')

    print('Running MCMC')
    if stacked_stat == "both":

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data, mean_mstar_red_data, mean_mstar_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data,  mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(mean_mstar_red_data)
        data_arr.append(mean_mstar_blue_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            data_arr = data_arr.dot(mat)

    # if stacked_stat == True
    elif stacked_stat: 

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr.append(mean_mstar_red_data)
        data_arr.append(mean_mstar_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            data_arr = data_arr.dot(mat)

    # if stacked_stat == False
    else:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            data_arr = data_arr.dot(mat)

    sampler = mcmc(nproc, nwalkers, nsteps, data_arr, sigma, mat)

    # sampler = mcmc(nproc, nwalkers, nsteps, total_data[1], f_blue_data[2], 
    #     f_blue_data[3], sigma, corr_mat_inv)

    print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))
    print("Mean autocorrelation time: {0:.3f} steps".format(
        np.mean(sampler.get_autocorr_time())))

# Main function
if __name__ == '__main__':
    args = args_parser()
    # pr = cProfile.Profile()
    # pr.enable()

    main(args)

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()

    # with open('profile_eco_2p_16w_5s.txt', 'w+') as f:
    #     f.write(s.getvalue())






