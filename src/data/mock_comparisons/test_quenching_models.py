"""
{This script tests given values of the Zu and Mandelbaum quenching model
 parameters to see how the result SMF for blue and red model galaxies compares
 to that of data}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from halotools.sim_manager import CachedHaloCatalog
import matplotlib.animation as animation
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import scipy
import math
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'serif', 'serif': ['Times']}, size=15)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=10)
rc('ytick.major', width=2, size=10)


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

def read_data(path_to_file, survey):
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
        logmstar_arr = mstar_arr

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
    # cov_mat_red = np.cov(phi_arr_red, rowvar=False) # default norm is N-1
    # err_red = np.sqrt(cov_mat_red.diagonal())
    # colour_err_arr.append(err_red)
    # cov_mat_blue = np.cov(phi_arr_blue, rowvar=False) # default norm is N-1
    # err_blue = np.sqrt(cov_mat_blue.diagonal())
    # colour_err_arr.append(err_blue)

    cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    err_colour = np.sqrt(cov_mat_colour.diagonal())

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

    return err_total, err_colour[0:5], err_colour[5:10]

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
    ver = 2.0
    chi2_df = pd.read_csv(path_to_file,header=None,names=['chisquared'])

    # Applies to runs prior to run 5?
    if mf_type == 'smf' and survey == 'eco' and ver==1.0:
        # Needed to reshape since flattened along wrong axis, 
        # didn't correspond to chain
        test_reshape = chi2_df.chisquared.values.reshape((1000,250))
        chi2 = np.ndarray.flatten(np.array(test_reshape),'F')
    
    else:
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
    ver = 2.0
    colnames = ['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',\
        'scatter']
    
    if mf_type == 'smf' and survey == 'eco' and ver==1.0:
        emcee_table = pd.read_csv(path_to_file,names=colnames,sep='\s+',\
            dtype=np.float64)

    else:
        emcee_table = pd.read_csv(path_to_file, names=colnames, 
            delim_whitespace=True, header=None)

        emcee_table = emcee_table[emcee_table.mhalo_c.values != '#']
        emcee_table.mhalo_c = emcee_table.mhalo_c.astype(np.float64)
        emcee_table.mstellar_c = emcee_table.mstellar_c.astype(np.float64)
        emcee_table.lowmass_slope = emcee_table.lowmass_slope.astype(np.float64)

    # Cases where last parameter was a NaN and its value was being written to 
    # the first element of the next line followed by 4 NaNs for the other 
    # parameters
    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            row[4] = scatter_val
    
    # Cases where rows of NANs appear
    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)
    
    return emcee_table

def get_paramvals_percentile(table, percentile, chi2_arr):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 1000 sample

    Parameters
    ----------
    table: pandas dataframe
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2_arr: array
        Array of chi^2 values

    Returns
    ---------
    mcmc_table_pctl: pandas dataframe
        Random 1000 sample of 68th percentile lowest chi^2 values
    """ 
    percentile = percentile/100
    table['chi2'] = chi2_arr
    table = table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(percentile*len(table))
    mcmc_table_pctl = table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:5]

    return bf_params

def halocat_init(halo_cat,z):
    """
    Initial population of halo catalog using populate_mock function

    Parameters
    ----------
    halo_cat: string
        Path to halo catalog
    
    z: float
        Median redshift of survey

    Returns
    ---------
    model: halotools model instance
        Model based on behroozi 2010 SMHM
    """
    halocat = CachedHaloCatalog(fname=halo_cat, update_cached_fname=True)
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z, \
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
        limit = np.round(np.log10((10**8.9) / 2.041), 1)
        sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    elif survey == 'resolveb':
        limit = np.round(np.log10((10**8.7) / 2.041), 1)
        sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model_init.mock.galaxy_table[sample_mask]
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

def halo_quenching_model(gals_df, param=None, param_id=None):
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
    if param_id == 0:
        Mh_qc = 10**param # Msun/h 
    else:
        Mh_qc = 10**12.20 # Msun/h 
    
    if param_id == 1:
        Mh_qs = 10**param
    else: 
        Mh_qs = 10**12.17 # Msun/h

    if param_id == 2:
        mu_c = param
    else: 
        mu_c = 0.38

    if param_id == 3:
        mu_s = param
    else: 
        mu_s = 0.15

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/Mh_qc)**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/Mh_qs)**mu_s))

    return f_red_cen, f_red_sat

def hybrid_quenching_model(gals_df, params):
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
    Mstar_q, Mh_q, mu, nu = params[0], params[1], params[2], params[3]
    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df)
    cen_stellar_mass_arr, sat_stellar_mass_arr = get_stellar_mock(gals_df)

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/Mstar_q)**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/Mstar_q)**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/Mh_q)**nu))
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

def measure_all_smf(table, volume, cvar, data_bool):
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
        logmstar_col = 'stellar_mass'
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(np.log10(table[logmstar_col]), volume, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(np.log10(table[logmstar_col].loc[table[colour_col] == 'R']
            ), volume, True, 'R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(np.log10(table[logmstar_col].loc[table[colour_col] == 'B']
            ), volume, True, 'B')
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

# def args_parser():
#     """
#     Parsing arguments passed to script

#     Returns
#     -------
#     args: 
#         Input arguments to the script
#     """
#     print('Parsing in progress')
#     parser = argparse.ArgumentParser()
#     parser.add_argument('survey', type=str, 
#         help='Options: eco/resolvea/resolveb')
#     parser.add_argument('machine', type=str, 
#         help='Options: mac/bender')   
#     parser.add_argument('quenching_model', type=str,
#         help='Options: hybrid/halo')
#     args = parser.parse_args()
#     return args


global survey
global model_init
global machine
global model
global mf_type

# args = args_parser()
# survey = args.survey
# machine = args.machine
# model = args.quenching_model
survey = 'eco'
machine = 'mac'
model = 'hybrid'
mf_type = 'smf'
vol_sim = 130**3

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

    
if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

chi2_file = path_to_proc + 'smhm_run6/{0}_chi2.txt'.format(survey)
chain_file = path_to_proc + 'smhm_run6/mcmc_{0}_raw.txt'.format(survey)

if survey == 'eco':
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/vishnu/'\
        'rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

catl, volume, cvar, z_median = read_data(catl_file, survey)

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)

print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)

print('Getting best fit params')
bf_params = get_paramvals_percentile(mcmc_table, 68, chi2)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

print('Populating using best fit params')
ax1_catalog = populate_mock(bf_params)

print('Assigning centrals/satellites flag')
ax1_catalog = assign_cen_sat_flag(ax1_catalog)

print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, 0, True)

print('Calculating error in data from mocks')
total_data[2], red_data[2], blue_data[2] = get_err_data(survey, 
    path_to_mocks) 

# params for outside and within 1 sigma for unlog nu values
# params = [10**10.218907,  10**12.929771,  0.521268, 7.652937] 
# params = [10**10.218907,  10**12.929771,  0.521268 , 105.746357]

#params for well outside, midway outside and within 1 sigma for log nu values
# params = [10**10.030012,  10**13.761830,  0.665974, -1.165710]
params = [10**10.159325,  10**12.271083,  0.80802,  0.573578]
# params = [10**9.991129,  10**13.403738,  0.594571,  2.301931]

f_red_cen1, f_red_sat1 = hybrid_quenching_model(ax1_catalog, params)
ax1_catalog = assign_colour_label_mock(f_red_cen1, f_red_sat1, ax1_catalog)
total_model1, red_model1, blue_model1 = measure_all_smf(ax1_catalog, 
    vol_sim , 0, False)


fig1 = plt.figure()
lower_err = red_data[1] - red_data[2]
upper_err = red_data[1] + red_data[2]
lower_err = red_data[1] - lower_err
upper_err = upper_err - red_data[1]
asymmetric_err = [lower_err, upper_err]
plt.fill_between(red_data[0], red_data[1]-lower_err, 
    red_data[1]+upper_err, color='indianred',alpha=0.5)
lower_err = blue_data[1] - blue_data[2]
upper_err = blue_data[1] + blue_data[2]
lower_err = blue_data[1] - lower_err
upper_err = upper_err - blue_data[1]
asymmetric_err = [lower_err, upper_err]
plt.fill_between(blue_data[0], blue_data[1]-lower_err, 
    blue_data[1]+upper_err, color='cornflowerblue',alpha=0.5)
plt.plot(blue_model1[0],blue_model1[1],color='darkblue',
    linestyle='-',zorder=10,label='model',lw=3)
plt.plot(red_model1[0],red_model1[1],color='darkred',
    linestyle='-',zorder=10,label='model',lw=3)
plt.xlabel(r'$\boldmath\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
plt.ylabel(r'$\boldmath\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)
# plt.title(r'\boldmath$\log_{10}\ \boldsymbol{\ $\nu$} ')
plt.title(r'$\boldmath\log_{10}\ \nu = 0.573578$', fontsize=25)
plt.show()


    