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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerBase
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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
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

def get_paramvals_percentile(mcmc_table, pctl, chi2, randints_df):
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
    mcmc_table['mock_num'] = randints_df.mock_num.values.astype(int)
    mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:4]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][4]
    bf_randint = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][5].astype(int)
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2, bf_randint

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

def std_func(bins, mass_arr, vel_arr):
    """
    Calculate std from mean = 0

    Parameters
    ----------
    bins: array
        Array of bins
    mass_arr: array
        Array of masses to be binned
    vel_arr: array
        Array of velocities

    Returns
    ---------
    std_arr: array
        Standard deviation from 0 of velocity values in each mass bin
    """

    last_index = len(bins)-1
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        cen_deltav_arr = []
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])
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
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
            volume, True, 'R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
            volume, True, 'B')
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

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
    sig_arr_red = []
    sig_arr_blue = []
    cen_arr_red = []
    cen_arr_blue = []
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
            mock_pd = read_mock_catl(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
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
            max_red, phi_red, err_red, bins_red, counts_red = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
                volume, False, 'R')
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
                volume, False, 'B')
            phi_arr_total.append(phi_total)
            phi_arr_red.append(phi_red)
            phi_arr_blue.append(phi_blue)

            sig_red, sig_blue, cen_red, cen_blue = \
                get_deltav_sigma_mocks_qmcolour(survey, mock_pd)

            sig_arr_red.append(sig_red)
            sig_arr_blue.append(sig_blue)
            cen_arr_red.append(cen_red)
            cen_arr_blue.append(cen_blue)

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    sig_arr_red = np.array(sig_arr_red)
    sig_arr_blue = np.array(sig_arr_blue)
    cen_arr_red = np.array(cen_arr_red)
    cen_arr_blue = np.array(cen_arr_blue)

    # Covariance matrix for total phi (all galaxies)
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    err_total = np.sqrt(cov_mat.diagonal())

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

    dv_red_0 = sig_arr_red[:,0]
    dv_red_1 = sig_arr_red[:,1]
    dv_red_2 = sig_arr_red[:,2]
    dv_red_3 = sig_arr_red[:,3]
    dv_red_4 = sig_arr_red[:,4]

    dv_blue_0 = sig_arr_blue[:,0]
    dv_blue_1 = sig_arr_blue[:,1]
    dv_blue_2 = sig_arr_blue[:,2]
    dv_blue_3 = sig_arr_blue[:,3]
    dv_blue_4 = sig_arr_blue[:,4]

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

    return err_total, err_colour

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
    gals_df['cs_flag'] = C_S
    return gals_df

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
    params_df = mcmc_table_pctl.iloc[:,:4].reset_index(drop=True)
    mock_num_df = mcmc_table_pctl.iloc[:,5].reset_index(drop=True)
    frames = [params_df, mock_num_df]
    mcmc_table_pctl_new = pd.concat(frames, axis=1)
    chunks = np.array([mcmc_table_pctl_new.values[i::5] \
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
    # v_sim = 130**3
    v_sim = 890641.5172927063 

    maxis_red_arr = []
    phi_red_arr = []
    maxis_blue_arr = []
    phi_blue_arr = []
    cen_gals_red_arr = []
    cen_halos_red_arr = []
    cen_gals_blue_arr = []
    cen_halos_blue_arr = []
    std_red_arr = []
    std_blue_arr = []
    cen_std_red_arr = []
    cen_std_blue_arr = []
    f_red_cen_red_arr = []
    f_red_cen_blue_arr = []

    for theta in a_list:  
        randint_logmstar = int(theta[4])
        theta = theta[:4]
        cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', 'cz', \
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

        gals_df = assign_cen_sat_flag(gals_df)
        # Stellar masses in log but halo masses not in log
        f_red_cen, f_red_sat = hybrid_quenching_model(theta, gals_df, 'vishnu',\
            randint_logmstar)
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
        total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
        , False, randint_logmstar)    
        cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue, \
            f_red_cen_red, f_red_cen_blue = \
                get_centrals_mock(gals_df, randint_logmstar)
        std_red_model, std_blue_model, centers_red_model, centers_blue_model = \
            get_deltav_sigma_vishnu_qmcolour(gals_df, randint_logmstar)

        maxis_red_arr.append(red_model[0])
        phi_red_arr.append(red_model[1])
        maxis_blue_arr.append(blue_model[0])
        phi_blue_arr.append(blue_model[1])
        cen_gals_red_arr.append(cen_gals_red)
        cen_halos_red_arr.append(cen_halos_red)
        cen_gals_blue_arr.append(cen_gals_blue)
        cen_halos_blue_arr.append(cen_halos_blue)
        std_red_arr.append(std_red_model)
        std_blue_arr.append(std_blue_model)
        cen_std_red_arr.append(centers_red_model)
        cen_std_blue_arr.append(centers_blue_model)
        f_red_cen_red_arr.append(f_red_cen_red)
        f_red_cen_blue_arr.append(f_red_cen_blue)

    return [maxis_red_arr, phi_red_arr, maxis_blue_arr, phi_blue_arr, 
    cen_gals_red_arr, cen_halos_red_arr, cen_gals_blue_arr, cen_halos_blue_arr,
    std_red_arr, std_blue_arr, cen_std_red_arr, cen_std_blue_arr, 
    f_red_cen_red_arr, f_red_cen_blue_arr]

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
                cen_halos.append(10**(df.loghalom.values[index]))
            else:
                sat_halos.append(10**(df.loghalom.values[index]))

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
        # These masses are log
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['{0}'.format(randint)].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['{0}'.format(randint)].values[idx]))

    else:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                # Stellar masses in non-Vishnu mocks are in h=0.7
                cen_gals.append((10**(df.logmstar.values[idx]))/2.041)
            elif value == 0:
                sat_gals.append((10**(df.logmstar.values[idx]))/2.041)

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

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
    f_red_cen_gals_red = []
    f_red_cen_gals_blue = []

    for idx,value in enumerate(gals_df['C_S']):
        if value == 1:
            if gals_df['colour_label'][idx] == 'R':
                cen_gals_red.append(gals_df['{0}'.format(randint)][idx])
                cen_halos_red.append(gals_df['halo_mvir'][idx])
                f_red_cen_gals_red.append(gals_df['f_red'][idx])
            elif gals_df['colour_label'][idx] == 'B':
                cen_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                cen_halos_blue.append(gals_df['halo_mvir'][idx])
                f_red_cen_gals_blue.append(gals_df['f_red'][idx])

    cen_gals_red = np.array(cen_gals_red)
    cen_halos_red = np.log10(np.array(cen_halos_red))
    cen_gals_blue = np.array(cen_gals_blue)
    cen_halos_blue = np.log10(np.array(cen_halos_blue))

    return cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue, \
        f_red_cen_gals_red, f_red_cen_gals_blue

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
   
    red_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'R') & (catl.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'B') & (catl.g_galtype == 1)].values)


    # Calculating spread in velocity dispersion for galaxies in groups with a 
    # red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
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
        group = catl.loc[catl.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype\
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

def get_deltav_sigma_mocks_qmcolour(survey, mock_df):
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
    mock_pd = mock_df.copy()
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

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
                        
    centers_red = np.array(centers_red)
    centers_blue = np.array(centers_blue)
            
    return std_red, std_blue, centers_red, centers_blue

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
    groupid_col = 'groupid_{0}'.format(randint)
    # Using the same survey definition as in mcmc smf i.e excluding the 
    # buffer except no M_r cut since vishnu mock has no M_r info
    mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
        (mock_pd.cz.values <= max_cz) & \
        (mock_pd[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    red_subset_grpids = np.unique(mock_pd[groupid_col].loc[(mock_pd.\
        colour_label == 'R') & (mock_pd[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(mock_pd[groupid_col].loc[(mock_pd.\
        colour_label == 'B') & (mock_pd[g_galtype_col] == 1)].values)

    # Calculating spread in velocity dispersion for galaxies in groups
    # with a red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_pd.loc[mock_pd[groupid_col] == key]
        cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col].\
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

    # Calculating spread in velocity dispersion for galaxies in groups 
    # with a blue central

    blue_deltav_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = mock_pd.loc[mock_pd[groupid_col] == key]
        cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col]\
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

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
            
    return std_red, std_blue, centers_red, centers_blue

def get_best_fit_model(best_fit_params, best_fit_mocknum):
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
    cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', 'cz', \
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

    gals_df = assign_cen_sat_flag(gals_df)
    f_red_cen, f_red_sat = hybrid_quenching_model(best_fit_params, gals_df, 
        'vishnu', best_fit_mocknum)
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
    # v_sim = 130**3 
    v_sim = 890641.5172927063 
    total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
    , False, best_fit_mocknum)     
    cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
        f_red_cen_blue = get_centrals_mock(gals_df, best_fit_mocknum)
    std_red, std_blue, std_centers_red, std_centers_blue = \
        get_deltav_sigma_vishnu_qmcolour(gals_df, best_fit_mocknum)

    max_red = red_model[0]
    phi_red = red_model[1]
    max_blue = blue_model[0]
    phi_blue = blue_model[1]

    return max_red, phi_red, max_blue, phi_blue, cen_gals_red, cen_halos_red,\
        cen_gals_blue, cen_halos_blue, f_red_cen_red, f_red_cen_blue, std_red, \
            std_blue, std_centers_red, std_centers_blue

def plot_mf(result, red_data, blue_data, maxis_bf_red, phi_bf_red, 
    maxis_bf_blue, phi_bf_blue, bf_chi2, err_colour):
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

    maxis_red_data, phi_red_data = red_data[0], red_data[1]
    maxis_blue_data, phi_blue_data = blue_data[0], blue_data[1]

    i_outer = 0
    red_mod_arr = []
    blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][1][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][3][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][1][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][3][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][1][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][3][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][1][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][3][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][1][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][3][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    red_phi_max = np.amax(red_mod_arr, axis=0)
    red_phi_min = np.amin(red_mod_arr, axis=0)
    blue_phi_max = np.amax(blue_mod_arr, axis=0)
    blue_phi_min = np.amin(blue_mod_arr, axis=0)
    
    # alpha_mod = 0.7
    # col_red_mod = 'lightcoral'
    # col_blue_mod = 'cornflowerblue'
    # lw_mod = 3
    # for idx in range(len(result[0][0])):
    #     mr, = plt.plot(result[0][0][idx],result[0][1][idx],color=col_red_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[0][2])):
    #     mb, = plt.plot(result[0][2][idx],result[0][3][idx],color=col_blue_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[1][0])):
    #     plt.plot(result[1][0][idx],result[1][1][idx],color=col_red_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[1][2])):
    #     plt.plot(result[1][2][idx],result[1][3][idx],color=col_blue_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[2][0])):
    #     plt.plot(result[2][0][idx],result[2][1][idx],color=col_red_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[2][2])):
    #     plt.plot(result[2][2][idx],result[2][3][idx],color=col_blue_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[3][0])):
    #     plt.plot(result[3][0][idx],result[3][1][idx],color=col_red_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[3][2])):
    #     plt.plot(result[3][2][idx],result[3][3][idx],color=col_blue_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[4][0])):
    #     plt.plot(result[4][0][idx],result[4][1][idx],color=col_red_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)
    # for idx in range(len(result[4][2])):
    #     plt.plot(result[4][2][idx],result[4][3][idx],color=col_blue_mod,
    #         linestyle='-',alpha=alpha_mod,zorder=5,lw=lw_mod)

    # Data
    # dr = plt.fill_between(x=maxis_red_data, y1=phi_red_data+err_colour[0:5], 
    #     y2=phi_red_data-err_colour[0:5], color='darkred',alpha=0.4)
    # db = plt.fill_between(x=maxis_blue_data, y1=phi_blue_data+err_colour[5:10], 
    #     y2=phi_blue_data-err_colour[5:10], color='darkblue',alpha=0.4)

    fig1= plt.figure(figsize=(10,10))
    mr = plt.fill_between(x=maxis_red_data, y1=red_phi_max, 
        y2=red_phi_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=maxis_blue_data, y1=blue_phi_max, 
        y2=blue_phi_min, color='cornflowerblue',alpha=0.4)

    dr = plt.errorbar(maxis_red_data,phi_red_data,yerr=err_colour[0:5],
        color='darkred',fmt='s-',ecolor='darkred',markersize=5,capsize=5,
        capthick=0.5,zorder=10)
    db = plt.errorbar(maxis_blue_data,phi_blue_data,yerr=err_colour[5:10],
        color='darkblue',fmt='s-',ecolor='darkblue',markersize=5,capsize=5,
        capthick=0.5,zorder=10)
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bfr, = plt.plot(maxis_bf_red,phi_bf_red,
        color='maroon',ls='--',lw=3,zorder=10)
    bfb, = plt.plot(maxis_bf_blue,phi_bf_blue,
        color='darkblue',ls='--',lw=3,zorder=10)

    plt.ylim(-4,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)

    plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
    # plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

    plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
        xy=(0.1, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if survey == 'eco':
        plt.title('ECO')
    plt.show()
    # if mf_type == 'smf':
    #     plt.savefig(path_to_figures + 'smf_colour_{0}.png'.format(survey))
    # elif mf_type == 'bmf':
    #     plt.savefig(path_to_figures + 'bmf_colour_{0}.png'.format(survey))

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
    plt.plot(x_bf_red,y_bf_red,color='darkred',lw=3,label='Best-fit',zorder=10)
    plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=3,
        label='Best-fit',zorder=10)

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
    if mf_type == 'smf':
        if survey == 'eco':
            plt.ylim(np.log10((10**8.9)/2.041),)
            plt.title('ECO')
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
            plt.title('ECO')
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
    plt.legend([("darkred", "darkblue", "-"), \
        ("indianred","cornflowerblue", "-")],\
        ["Best-fit", "Models"], handler_map={tuple: AnyObjectHandler()},\
        loc='best', prop={'size': 30})
    plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
        xy=(0.8, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)
    plt.show()
    # if mf_type == 'smf':
    #     plt.savefig(path_to_figures + 'smhm_emcee_{0}.png'.format(survey))
    # elif mf_type == 'bmf':
    #     plt.savefig(path_to_figures + 'bmhm_emcee_{0}.png'.format(survey))

def plot_sigma_vdiff(result, std_red_data, cen_red_data, std_blue_data, 
    cen_blue_data, std_bf_red, std_bf_blue, std_cen_bf_red, std_cen_bf_blue, 
    bf_chi2, err_colour):
    """[summary]

    Args:
        result ([type]): [description]
        std_red_data ([type]): [description]
        cen_red_data ([type]): [description]
        std_blue_data ([type]): [description]
        cen_blue_data ([type]): [description]
        std_bf_red ([type]): [description]
        std_bf_blue ([type]): [description]
        std_cen_bf_red ([type]): [description]
        std_cen_bf_blue ([type]): [description]
        bf_chi2 ([type]): [description]
    """

    fig1= plt.figure(figsize=(10,10))
    for idx in range(len(result[0][0])):
        mr = plt.scatter(result[0][10][idx],result[0][8][idx],color='indianred',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[0][2])):
        mb = plt.scatter(result[0][11][idx],result[0][9][idx],color='cornflowerblue',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[1][0])):
        plt.scatter(result[1][10][idx],result[1][8][idx],color='indianred',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[1][2])):
        plt.scatter(result[1][11][idx],result[1][9][idx],color='cornflowerblue',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[2][0])):
        plt.scatter(result[2][10][idx],result[2][8][idx],color='indianred',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[2][2])):
        plt.scatter(result[2][11][idx],result[2][9][idx],color='cornflowerblue',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[3][0])):
        plt.scatter(result[3][10][idx],result[3][8][idx],color='indianred',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[3][2])):
        plt.scatter(result[3][11][idx],result[3][9][idx],color='cornflowerblue',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[4][0])):
        plt.scatter(result[4][10][idx],result[4][8][idx],color='indianred',
            alpha=0.3,zorder=5,s=120)
    for idx in range(len(result[4][2])):
        plt.scatter(result[4][11][idx],result[4][9][idx],color='cornflowerblue',
            alpha=0.3,zorder=5,s=120)

    dr = plt.scatter(cen_red_data, std_red_data, c='maroon', marker='p', s=160, 
        zorder=10, edgecolors='darkred')
    db = plt.scatter(cen_blue_data, std_blue_data, c='mediumblue', marker='p', 
        s=160, zorder=10, edgecolors='darkblue')

    sigr = plt.fill_between(x=cen_red_data, y1=std_red_data+err_colour[10:15], 
        y2=std_red_data-err_colour[10:15], color='darkred',alpha=0.3)
    sigb = plt.fill_between(x=cen_blue_data, y1=std_blue_data+err_colour[15:20], 
        y2=std_blue_data-err_colour[15:20], color='darkblue',alpha=0.3)


    bfr = plt.scatter(std_cen_bf_red, std_bf_red, c='maroon', marker='*', 
        s=160, zorder=10, edgecolors='darkred')
    bfb = plt.scatter(std_cen_bf_blue, std_bf_blue, c='mediumblue', 
        marker='*', s=160, zorder=10, edgecolors='darkblue')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
    plt.legend([(dr, db), (mr, mb), (bfr, bfb), (sigr, sigb)], 
        ['Data','Models','Best-fit',r'1$\sigma$'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5)
    if survey == 'eco':
        plt.title('ECO')
    plt.show()

def plot_zumand_fig4(result, gals_bf_red, halos_bf_red, gals_bf_blue, 
    halos_bf_blue, bf_chi2):

    # if model == 'halo':
    #     sat_halomod_df = gals_df.loc[gals_df.C_S.values == 0] 
    #     cen_halomod_df = gals_df.loc[gals_df.C_S.values == 1]
    # elif model == 'hybrid':
    #     sat_hybmod_df = gals_df.loc[gals_df.C_S.values == 0]
    #     cen_hybmod_df = gals_df.loc[gals_df.C_S.values == 1]

    logmhalo_mod_arr = result[0][5]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][5])
    for idx in range(5):
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][7])
    logmhalo_mod_arr_flat = np.hstack(logmhalo_mod_arr)

    logmstar_mod_arr = result[0][4]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][4])
    for idx in range(5):
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][6])
    logmstar_mod_arr_flat = np.hstack(logmstar_mod_arr)

    fred_mod_arr = result[0][12]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][12])
    for idx in range(5):
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][13])
    fred_mod_arr_flat = np.hstack(fred_mod_arr)

    fig1 = plt.figure()
    plt.hexbin(logmhalo_mod_arr_flat, logmstar_mod_arr_flat, 
        C=fred_mod_arr_flat, cmap='rainbow', reduce_C_function=np.median)
    cb = plt.colorbar()
    cb.set_label(r'\boldmath\ median $f_{red}$')

    x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    gals_bf_red,base=0.4,bin_statval='center')
    x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    gals_bf_blue,base=0.4,bin_statval='center')

    bfr, = plt.plot(x_bf_red,y_bf_red,color='darkred',lw=5,zorder=10)
    bfb, = plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=5,zorder=10)

    plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(bf_chi2,2)), 
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
    plt.show()


global survey
global path_to_figures
global gal_group_df_subset

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

machine = 'mac'
mf_type = 'smf'
survey = 'eco'
nproc = 2

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

chi2_file = path_to_proc + 'smhm_colour_run16/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run16/mcmc_{0}_colour_raw.txt'.\
    format(survey)
randint_file = path_to_proc + 'smhm_colour_run16/{0}_colour_mocknum.txt'.\
    format(survey)

if survey == 'eco':
    # catl_file = path_to_raw + "eco/eco_all.csv"
    ## New catalog with group finder run on subset after applying M* and cz cuts
    catl_file = path_to_proc + "gal_group_eco_data.hdf5"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

print('Reading files')
chi2 = read_chi2(chi2_file)
mcmc_table = read_mcmc(chain_file)
mock_nums_df = pd.read_csv(randint_file, header=None, names=['mock_num'], 
    dtype=int)
catl, volume, z_median = read_data_catl(catl_file, survey)
gal_group_df = read_mock_catl(path_to_proc + "gal_group.hdf5") 


print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2, bf_randint = \
    get_paramvals_percentile(mcmc_table, 68, chi2, mock_nums_df)

## Use only the mocks that are in the random sample of 100
# Count the first 20 + 22nd + 123-131 columns of general information from 
# mock catalog (halo + rsd)
idx_arr = np.insert(np.linspace(0,20,21), len(np.linspace(0,20,21)), (22, 123, 
    124, 125, 126, 127, 128, 129, 130, 131)).astype(int)

names_arr = [x for x in gal_group_df.columns.values[idx_arr]]
for idx in mcmc_table_pctl.mock_num.unique():
    names_arr.append('{0}_y'.format(idx))
    names_arr.append('groupid_{0}'.format(idx))
    names_arr.append('g_galtype_{0}'.format(idx))
names_arr = np.array(names_arr)

gal_group_df_subset = gal_group_df[names_arr]

# Renaming the "1_y" column kept from line 1896 because of case where it was
# also in mcmc_table_ptcl.mock_num and was selected twice
gal_group_df_subset.columns.values[30] = "behroozi_bf"

for idx in mcmc_table_pctl.mock_num.unique():
    gal_group_df_subset = gal_group_df_subset.rename(columns=\
        {'{0}_y'.format(idx):'{0}'.format(idx)})


print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Retrieving survey centrals')
# Returns log masses in h=1.0
cen_gals_data_red, cen_halos_data_red, cen_gals_data_blue, cen_halos_data_blue =\
     get_centrals_data(catl)

# Returns measurements in h=1.0
print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

# Returns masses in h=1.0
print('Measuring spread in vel disp for data')
std_red, centers_red, std_blue, centers_blue = get_deltav_sigma_data(catl)

# Measures errors using measurements in h=1.0
print('Measuring error in data from mocks')
err_total_data, err_colour_data = \
    get_err_data(survey, path_to_mocks)

print('Multiprocessing')
result = mp_init(mcmc_table_pctl, nproc)
# Most recent run took 3 minutes 12/1/2021

print('Getting best fit model')
maxis_bf_red, phi_bf_red, maxis_bf_blue, phi_bf_blue, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
        f_red_cen_blue, std_bf_red, std_bf_blue, std_cen_bf_red, \
            std_cen_bf_blue = get_best_fit_model(bf_params, bf_randint)

plot_mf(result, red_data, blue_data, maxis_bf_red, phi_bf_red, 
    maxis_bf_blue, phi_bf_blue, bf_chi2, err_colour_data)

plot_xmhm(result, cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue,
    cen_gals_data_red, cen_halos_data_red, cen_gals_data_blue, 
    cen_halos_data_blue, bf_chi2)

plot_zumand_fig4(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
    cen_halos_blue, bf_chi2)

plot_sigma_vdiff(result, std_red, centers_red, std_blue, centers_blue, 
    std_bf_red, std_bf_blue, std_cen_bf_red, std_cen_bf_blue, bf_chi2, 
    err_colour_data)