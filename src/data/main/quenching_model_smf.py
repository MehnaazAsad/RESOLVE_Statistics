"""
{This script tests best fit SMHM for ECO and compares the resulting SMF for 
 both red and blue galaxies with those from data}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from halotools.sim_manager import CachedHaloCatalog
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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=15)
rc('text', usetex=True)

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
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms', 'modelu_rcorr']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
            (eco_buff.grpcz.values <= 7000) & \
                (eco_buff.absrmag.values <= -17.33) &\
                    (eco_buff.logmstar.values >= 8.9)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b', 
                    'modelu_rcorr']
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
    test_reshape = chi2_df.chisquared.values.reshape((1000,250))
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
    # Sample random 1000 of lowest chi2
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(1000)

    return mcmc_table_pctl, bf_params

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

def halo_quenching_model(gals_df):
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
    Mh_qc = 10**12.20 
    Mh_qs = 10**12.17
    mu_c = 0.38
    mu_s = 0.15

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/Mh_qc)**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/Mh_qs)**mu_s))

    return f_red_cen, f_red_sat

def hybrid_quenching_model(gals_df):
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
    Mstar_q = 10**10.5 
    Mh_q = 10**13.76
    mu = 0.69
    nu = 0.15

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

def assign_colour_mock(gals_df, catl, stat):
    """
    Assign colour to mock catalog

    Parameters
    ----------
    gals_df: pandas Dataframe
        Mock catalog
    catl: pandas Dataframe
        Data catalog
    stat: string
        Specify whether mean or median statistic is used to assign colour
        from data to mock catalog

    Returns
    ---------
    gals_df: pandas Dataframe
        Dataframe with model corrected (u-r) colour assigned as new column
    """

    logmstar_arr_mock = np.log10(gals_df.stellar_mass.values)
    logmstar_arr_data = catl.logmstar.values
    # Both measurements of stellar masses have to be in the same h=1 unit
    logmstar_arr_data = np.log10((10**logmstar_arr_data) / 2.041)
    u_r_arr_data = catl.modelu_rcorr.values

    # Either assign the mean or median colour within each bin of stellar mass
    if stat == 'mean':
        x,y,x_err,y_err = Stats_one_arr(logmstar_arr_data, u_r_arr_data, 0.005, 
        statfunc=np.nanmean)
    elif stat == 'median':
        x,y,x_err,y_err = Stats_one_arr(logmstar_arr_data, u_r_arr_data, 0.005, 
        statfunc=np.nanmedian)       

    # Assign mean or median colour based on which data bin the mock stellar mass
    # falls in
    colour_arr = np.zeros(len(gals_df))
    for idx1, value1 in enumerate(logmstar_arr_mock):
        colour = 0
        for idx2, value2 in enumerate(x):
            if value1 > value2:
                colour = y[idx2]
                break

        colour_arr[idx1] = colour
    
    gals_df['modelu_rcorr'] = colour_arr

    return gals_df

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
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)        
    else:
        logmstar_arr = mstar_arr
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 12)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 12)   


    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    phi = counts / (volume * dm)  # not a log quantity

    return maxis, phi, err_poiss, bins, counts

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

    phi_arr_total = []
    phi_arr_red = []
    phi_arr_blue = []
    for num in range(num_mocks):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        mock_pd = reading_catls(filename) 

        #Using the same survey definition as in mcmc smf i.e excluding the buffer
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

        mock_pd['colour_label'] = colour_label_arr

        #Measure SMF of mock using diff_smf function
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(logmstar_arr, volume, 0, False)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            volume, 0, False)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            volume, 0, False)
        phi_arr_total.append(phi_total)
        phi_arr_red.append(phi_red)
        phi_arr_blue.append(phi_blue)

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)

    err_total = np.std(phi_arr_total, axis=0)
    err_red = np.std(phi_arr_red, axis=0)
    err_blue = np.std(phi_arr_blue, axis=0)

    return err_total, err_red, err_blue

def plot_mstellar_colour_data(catl):
    """
    Plots stellar mass vs colour for data catalog

    Parameters
    ----------
    catl: pandas Dataframe
        Data catalog
    """

    u_r_arr = catl.modelu_rcorr.values
    logmstar_arr = catl.logmstar.values 
    x = logmstar_arr

    # Values from Moffett et al. 2015 equation 1
    if survey == 'eco' or survey == 'resolvea':
        div_lowest_xmin = 8.9
    elif survey == 'resolveb':
        div_lowest_xmin = 8.7

    div_lowest_xmax = 9.1
    div_lowest_y = 1.457
    
    div_mid_xmin = div_lowest_xmax
    div_mid_xmax = 10.1
    div_mid_x = np.unique(x[np.where((x >= div_mid_xmin) & (x <= div_mid_xmax))])
    div_mid_y = 0.24 * div_mid_x - 0.7

    div_max_xmin = div_mid_xmax
    div_max_xmax = x.max()
    div_max_y = 1.7

    # # unique because otherwise when plotting there were too many points and the 
    # # dashed line appeared solid
    # x_new = np.unique(x[np.where((x >= 9.1) & (x <= 10.1))])
    # y = 0.24*x_new - 0.7

    # # Joining arrays
    # div_x_arr = [8.9, 9.09, 10.1, 11.79]
    # div_y_arr = [1.457, 1.457, 1.7, 1.7]

    # div_x_arr = np.concatenate((div_x_arr, x_new))
    # div_y_arr = np.concatenate((div_y_arr, y))
    # # Sorting out values
    # div_x_sort_idx = np.argsort(div_x_arr)
    # div_arr = np.vstack((div_x_arr[div_x_sort_idx], div_y_arr[div_x_sort_idx]))

    plt.clf()
    plt.close()
    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(111)
    ax = sns.kdeplot(logmstar_arr, u_r_arr, ax=ax1, cmap='Blues', shade=True,
    shade_lowest=False)
    ax.scatter(logmstar_arr,u_r_arr,c='#921063',marker='x',alpha=0.1,zorder=1)
    ax1.hlines(y=div_lowest_y,xmin=div_lowest_xmin,xmax=div_lowest_xmax,
    linestyle='--',color='k', linewidth=2,zorder=10)    
    ax1.plot(div_mid_x,div_mid_y, color='k', linestyle='--',linewidth=2)
    # ax1.plot(div_arr[0], div_arr[1], linestyle='--', color='k', linewidth=2)
    ax1.hlines(y=div_max_y,xmin=div_max_xmin,xmax=div_max_xmax,linestyle='--',
    color='k', linewidth=2,zorder=10)
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \right]$')
    plt.ylabel(r'\boldmath$ (u-r)^e$')
    if survey == 'eco':
        plt.title('ECO')
    elif survey == 'resolvea':
        plt.title('RESOLVE-A')
    elif survey == 'resolveb':
        plt.title('RESOLVE-B')
    plt.show()

def plot_eco_mstellar_colour_mock(gals_df, model):
    """
    Plots stellar mass vs colour from mock catalog

    Parameters
    ----------
    gals_df: pandas Dataframe
        Dataframe of mock catalog
    model: string
        Hybrid or halo quenching model
    """

    fig1 = plt.figure(figsize=(10,10))  
    ax1 = fig1.add_subplot(111)
    gals_df_subset = gals_df.loc[gals_df.modelu_rcorr.values > 0]
    ax = sns.kdeplot(np.log10(gals_df_subset.stellar_mass.values), 
    gals_df_subset.modelu_rcorr.values, ax=ax1, cmap='Blues', shade=True, 
    shade_lowest=False)
    ax.scatter(np.log10(gals_df_subset.stellar_mass.values),
    gals_df_subset.modelu_rcorr.values,c='#921063',marker='x',alpha=0.1,zorder=1)
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \right]$') 
    plt.ylabel(r'\boldmath$ (u-r)^e$')
    if model == 'hybrid':
        plt.title(r'Hybrid quenching model')
    elif model == 'halo':
        plt.title(r'Halo quenching model')
    plt.show()

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
            diff_smf(table[logmstar_col], volume, cvar, False)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, cvar, False)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, cvar, False)
    else:
        logmstar_col = 'stellar_mass'
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(np.log10(table[logmstar_col]), volume, cvar, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(np.log10(table[logmstar_col].loc[table[colour_col] == 'R']
            ), volume, cvar, True)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(np.log10(table[logmstar_col].loc[table[colour_col] == 'B']
            ), volume, cvar, True)
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

def plot_smf(total_data, red_data, blue_data, total_model, red_model, 
blue_model, model):
    """
    Plots stellar mass function for all, red and blue galaxies for data and 
    for halo/hybrid model

    Parameters
    ----------
    total_data: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for all galaxies from data
    red_data: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for red galaxies from data
    blue_data: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for blue galaxies from data
    total_model: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for all galaxies from model (hybrid or halo)
    red_model: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for red galaxies from model (hybrid or halo)
    blue_model: array
        Multidimensional array of stellar mass, phi, total error in SMF and 
        counts per bin for blue galaxies from model (hybrid or halo)
    """

    max_total_data, phi_total_data, err_total_data, counts_total_data = \
        total_data[0], total_data[1], total_data[2], total_data[3]
    max_red_data, phi_red_data, err_red_data, counts_red_data = \
        red_data[0], red_data[1], red_data[2], red_data[3]
    max_blue_data, phi_blue_data, err_blue_data, counts_blue_data = \
        blue_data[0], blue_data[1], blue_data[2], blue_data[3]
    max_total, phi_total, err_total, counts_total = \
        total_model[0], total_model[1], total_model[2], total_model[3]
    max_red, phi_red, err_red, counts_red = \
        red_model[0], red_model[1], red_model[2], red_model[3]
    max_blue, phi_blue, err_blue, counts_blue = \
        blue_model[0], blue_model[1], blue_model[2], blue_model[3]

    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(111)
    plt.fill_between(max_total_data,phi_total_data-err_total_data,
    phi_total_data+err_total_data,color='k',alpha=0.3, 
        label=r'$\textrm{total}_{\textrm{d}}$')
    plt.errorbar(max_total,phi_total,yerr=err_total,color='k',
        fmt='-s',ecolor='k',markersize=4,capsize=5,capthick=0.5,
        label=r'$\textrm{total}_{\textrm{m}}$',
        zorder=10)
    plt.fill_between(max_red_data,phi_red_data-err_red_data,
    phi_red_data+err_red_data,color='r',alpha=0.3, 
        label=r'$\textrm{red}_{\textrm{d}}$')
    plt.errorbar(max_red,phi_red,yerr=err_red,color='r',
        fmt='-s',ecolor='r',markersize=4,capsize=5,capthick=0.5,
        label=r'$\textrm{red}_{\textrm{m}}$',
        zorder=10)
    plt.fill_between(max_blue_data,phi_blue_data-err_blue_data,
    phi_blue_data+err_blue_data,color='b',alpha=0.3,
        label=r'$\textrm{blue}_{\textrm{d}}$')
    plt.errorbar(max_blue,phi_blue,yerr=err_blue,color='b',
        fmt='-s',ecolor='b',markersize=4,capsize=5,capthick=0.5,
        label=r'$\textrm{blue}_{\textrm{m}}$',
        zorder=10)
    for i in range(len(phi_total_data)):
        text = ax1.text(max_total_data[i], 10**-1.07, counts_total_data[i],
        ha="center", va="center", color="k", size=7)
        if i == 0 or i == 1:
            text = ax1.text(max_total_data[i] + 0.12, 10**-1.07, '(' + 
            np.str(counts_total[i]) + ')', ha="center", va="center", color="k", 
            size=7)           
        elif i == 10:
            text = ax1.text(max_total_data[i] + 0.05, 10**-1.07, '(' + 
            np.str(counts_total[i]) + ')', ha="center", va="center", color="k", 
            size=7)
        else:
            text = ax1.text(max_total_data[i] + 0.1, 10**-1.07, '(' + 
            np.str(counts_total[i]) + ')', ha="center", va="center", color="k", 
            size=7)
    for i in range(len(phi_red_data)):
        text = ax1.text(max_red_data[i], 10**-1.18, counts_red_data[i],
        ha="center", va="center", color="r", size=7)
        if i == 0 or i == 1:
            text = ax1.text(max_red_data[i] + 0.12, 10**-1.18, '(' + 
            np.str(counts_red[i]) + ')', ha="center", va="center", color="r", 
            size=7)
        elif i == 10:
            text = ax1.text(max_red_data[i] + 0.05, 10**-1.18, '(' + 
            np.str(counts_red[i]) + ')', ha="center", va="center", color="r", 
            size=7)
        else:
            text = ax1.text(max_red_data[i] + 0.1, 10**-1.18, '(' + 
            np.str(counts_red[i]) + ')', ha="center", va="center", color="r", 
            size=7)
    for i in range(len(phi_blue_data)):
        text = ax1.text(max_blue_data[i], 10**-1.28, counts_blue_data[i],
        ha="center", va="center", color="b", size=7)
        if i == 0 or i == 1:
            text = ax1.text(max_blue_data[i] + 0.12, 10**-1.28, '(' + 
            np.str(counts_blue[i]) + ')', ha="center", va="center", color="b", 
            size=7) 
        elif i == 10:
            text = ax1.text(max_blue_data[i] + 0.05, 10**-1.28, '(' + 
            np.str(counts_blue[i]) + ')', ha="center", va="center", color="b", 
            size=7)
        else:
            text = ax1.text(max_blue_data[i] + 0.1, 10**-1.28, '(' + 
            np.str(counts_blue[i]) + ')', ha="center", va="center", color="b", 
            size=7)            
    plt.yscale('log')
    plt.ylim(10**-5,10**-1)
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', 
        fontsize=15)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', 
        fontsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left',
        prop={'size': 10})
    if model == 'halo':
        plt.title(r'Halo quenching model - ECO')
    elif model == 'hybrid':
        plt.title(r'Hybrid quenching model - ECO')
    plt.show()

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
    parser.add_argument('survey', type=str, 
        help='Options: eco/resolvea/resolveb')
    parser.add_argument('quenching_model', type=str,
        help='Options: hybrid/halo')
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
    global survey
    global model
    global model_init
    
    survey = args.survey
    model = args.quenching_model

    # Paths
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_interim = dict_of_paths['int_dir']
    path_to_figures = dict_of_paths['plot_dir']
    path_to_external = dict_of_paths['ext_dir']
    path_to_eco_mocks = path_to_external + 'ECO_mvir_catls/'

    vol_sim = 130**3 # Mpc/h

    chi2_file = path_to_proc + 'smhm_run3/{0}_chi2.txt'.format(survey)
    chain_file = path_to_proc + 'smhm_run3/mcmc_{0}.dat'.format(survey)
    
    if survey == 'eco':
        catl_file = path_to_raw + "eco_all.csv"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
    catl, volume, cvar, z_median = read_data(catl_file, survey)

    print('Reading chi-squared file')
    chi2 = read_chi2(chi2_file)

    print('Reading mcmc chain file')
    mcmc_table = read_mcmc(chain_file)

    print('Getting data in specific percentile')
    mcmc_table_pctl, bf_params = get_paramvals_percentile(mcmc_table, 68, chi2)

    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Populating mock using best fit parameters')
    gals_df = populate_mock(bf_params)

    print('Assigning centrals and satellites flag')
    gals_df = assign_cen_sat_flag(gals_df)

    print('Applying quenching model')
    if model == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(gals_df)
    elif model == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(gals_df)

    print('Assigning colour labels to mock galaxies')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    print('Assigning colour labels to data')
    catl = assign_colour_label_data(catl)

    print('Assigning colour to mock galaxies')
    gals_df = assign_colour_mock(gals_df, catl, 'median')

    print('Measuring SMF for data')
    total_data, red_data, blue_data = measure_all_smf(catl, volume, 0, True)
    total_data[2], red_data[2], blue_data[2] = get_err_data(survey, 
        path_to_eco_mocks)  

    print('Measuring SMF for model')
    total_model, red_model, blue_model = measure_all_smf(gals_df, vol_sim 
    , 0, False)

    print('Plotting SMF')
    plot_smf(total_data, red_data, blue_data, total_model, red_model, blue_model, 
    model)

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args) 