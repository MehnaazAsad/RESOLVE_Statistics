"""
{This script tests different values of the Zu and Mandelbaum quenching model
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
import matplotlib
matplotlib.use('Agg')
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
 #'/fs1/masad/anaconda3/envs/resolve_statistics/bin/magick'


def which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p

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
                    'fc', 'grpmb', 'grpms', 'modelu_rcorr', 'umag', 'rmag']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file, delimiter=",", header=0, 
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
            (eco_buff.grpcz.values <= 7000) & 
            (eco_buff.absrmag.values <= -17.33) &
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
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, 
            usecols=columns)

        if survey == 'resolvea':
            catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                (resolve_live18.grpcz.values > 4500) & 
                (resolve_live18.grpcz.values < 7000) & 
                (resolve_live18.absrmag.values < -17.33) & 
                (resolve_live18.logmstar.values >= 8.9)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            # 487 - cz, 369 - grpcz
            catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                (resolve_live18.grpcz.values > 4500) & 
                (resolve_live18.grpcz.values < 7000) & 
                (resolve_live18.absrmag.values < -17) & 
                (resolve_live18.logmstar.values >= 8.7)]

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
    Calculate error in red and blue data SMF from Victor's mocks

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
    max_arr_blue = []
    err_arr_blue = []
    for num in range(num_mocks):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        mock_pd = reading_catls(filename) 

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

    err_total = np.std(np.log10(phi_arr_total), axis=0)
    err_red = np.std(np.log10(phi_arr_red), axis=0)
    err_blue = np.std(np.log10(phi_arr_blue), axis=0)

    return err_total, err_red, err_blue

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
    
    gals_df_arr = []
    for model_init in [model1, model2, model3, model4]:
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
        gals_df_arr.append(gals_df)

    gals_df1 = gals_df_arr[0]
    gals_df2 = gals_df_arr[1]
    gals_df3 = gals_df_arr[2]
    gals_df4 = gals_df_arr[3]

    return gals_df1, gals_df2, gals_df3, gals_df4

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

def hybrid_quenching_model(gals_df, param=None, param_id=None):
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
    if param_id == 0:
        Mh_q = 10**param # Msun/h 
    else:
        Mh_q = 10**13.76 # Msun/h 
    
    if param_id == 1:
        Mstar_q = 10**param
    else: 
        Mstar_q = 10**10.5 # Msun/h

    if param_id == 2:
        mu = param
    else: 
        mu = 0.69

    if param_id == 3:
        nu = param
    else: 
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
    parser.add_argument('machine', type=str, 
        help='Options: mac/bender')   
    parser.add_argument('quenching_model', type=str,
        help='Options: hybrid/halo')
    args = parser.parse_args()
    return args


global survey
global model1
global model2
global model3
global model4
global machine
global model

args = args_parser()
survey = args.survey
machine = args.machine
model = args.quenching_model

plt.rcParams['animation.convert_path'] = '{0}/magick'.format(os.path.dirname(which('python')))

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']

if survey == 'eco':
    path_to_mocks = path_to_external + 'ECO_mvir_catls/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

chi2_file = path_to_proc + 'smhm_run3/{0}_chi2.txt'.format(survey)
chain_file = path_to_proc + 'smhm_run3/mcmc_{0}.dat'.format(survey)

if survey == 'eco':
    catl_file = path_to_raw + "eco_all.csv"
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

print('Getting data in specific percentile')
mcmc_table_pctl, bf_params = get_paramvals_percentile(mcmc_table, 68, chi2)

print('Initial population of halo catalog')
model1 = halocat_init(halo_catalog, z_median)
model2 = halocat_init(halo_catalog, z_median)
model3 = halocat_init(halo_catalog, z_median)
model4 = halocat_init(halo_catalog, z_median)

print('Populating using best fit params')
ax1_catalog, ax2_catalog, ax3_catalog, ax4_catalog = populate_mock(bf_params)

print('Assigning centrals/satellites flag')
ax1_catalog = assign_cen_sat_flag(ax1_catalog)
ax2_catalog = assign_cen_sat_flag(ax2_catalog)
ax3_catalog = assign_cen_sat_flag(ax3_catalog)
ax4_catalog = assign_cen_sat_flag(ax4_catalog)

print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, 0, True)

print('Calculating error in data from mocks')
total_data[2], red_data[2], blue_data[2] = get_err_data(survey, 
    path_to_mocks) 

counter = 0

# parameter values from Table 1 of Zu and Mandelbaum 2015 "prior case"
if model == 'hybrid':
    Mh_q_arr = np.linspace(11.0, 15.5, 15)[:2] # Msun/h
    Mstar_q_arr = np.linspace(9.0, 12.0, 15)[:2] # Msun/h
    mu_arr = np.linspace(0.0, 3.0, 15)[:2]
    nu_arr = np.linspace(0.0, 3.0, 15)[:2]

elif model == 'halo':
    Mh_qc_arr = np.linspace(11.0, 15.5, 15)
    Mh_qs_arr = np.linspace(11.0, 15.5, 15)
    muc_arr = np.linspace(0.0, 3.0, 15)
    mus_arr = np.linspace(0.0, 3.0, 15)
    
def init():
    
    line1 = ax1.errorbar([],[],yerr=[],color='#E61A27',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#E61A27',capsize=2,capthick=0.5)
    line2 = ax2.errorbar([],[],yerr=[],color='#E61A27',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#E61A27',capsize=2,capthick=0.5)     
    line3 = ax3.errorbar([],[],yerr=[],color='#E61A27',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#E61A27',capsize=2,capthick=0.5)     
    line4 = ax4.errorbar([],[],yerr=[],color='#E61A27',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#E61A27',capsize=2,capthick=0.5)
    line5 = ax1.errorbar([],[],yerr=[],color='#5696B9',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#5696B9',capsize=2,capthick=0.5)
    line6 = ax2.errorbar([],[],yerr=[],color='#5696B9',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#5696B9',capsize=2,capthick=0.5)     
    line7 = ax3.errorbar([],[],yerr=[],color='#5696B9',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#5696B9',capsize=2,capthick=0.5)     
    line8 = ax4.errorbar([],[],yerr=[],color='#5696B9',fmt="s-",linewidth=2,
        elinewidth=0.5,ecolor='#5696B9',capsize=2,capthick=0.5)

    line = [line1,line2,line3,line4,line5,line6,line7,line8]
    
    return line

def make_animation(i,j,k,l,ax1_catalog,ax2_catalog,ax3_catalog,ax4_catalog):
    global counter
    vol_sim = 130**3 # Mpc/h
    if model == 'hybrid':
        Mhalo, Mstar, mu, nu = i, j[counter], k[counter], l[counter]
    elif model == 'halo':
        Mhalo_c, Mhalo_s, mu_c, mu_s = i, j[counter], k[counter], l[counter]

    print('Applying quenching model')
    if model == 'hybrid':
        f_red_cen1, f_red_sat1 = hybrid_quenching_model(ax1_catalog, Mhalo, 0)
        f_red_cen2, f_red_sat2 = hybrid_quenching_model(ax2_catalog, Mstar, 1)
        f_red_cen3, f_red_sat3 = hybrid_quenching_model(ax3_catalog, mu, 2)
        f_red_cen4, f_red_sat4 = hybrid_quenching_model(ax4_catalog, nu, 3)
    elif model == 'halo':
        f_red_cen1, f_red_sat1 = halo_quenching_model(ax1_catalog, Mhalo_c, 0)
        f_red_cen2, f_red_sat2 = halo_quenching_model(ax2_catalog, Mhalo_s, 1)
        f_red_cen3, f_red_sat3 = halo_quenching_model(ax3_catalog, mu_c, 2)
        f_red_cen4, f_red_sat4 = halo_quenching_model(ax4_catalog, mu_s, 3)

    ax1_catalog = assign_colour_label_mock(f_red_cen1, f_red_sat1, ax1_catalog)
    ax2_catalog = assign_colour_label_mock(f_red_cen2, f_red_sat2, ax2_catalog)
    ax3_catalog = assign_colour_label_mock(f_red_cen3, f_red_sat3, ax3_catalog)
    ax4_catalog = assign_colour_label_mock(f_red_cen4, f_red_sat4, ax4_catalog)

    total_model1, red_model1, blue_model1 = measure_all_smf(ax1_catalog, 
        vol_sim , 0, False)
    total_model2, red_model2, blue_model2 = measure_all_smf(ax2_catalog, 
        vol_sim , 0, False)
    total_model3, red_model3, blue_model3 = measure_all_smf(ax3_catalog, 
        vol_sim , 0, False)
    total_model4, red_model4, blue_model4 = measure_all_smf(ax4_catalog, 
        vol_sim , 0, False)
    
    for ax in [ax1,ax2,ax3,ax4]:
        ax.clear()
        smf_data_red = ax.fill_between(red_data[0], 
            np.log10(red_data[1])-red_data[2], 
            np.log10(red_data[1])+red_data[2], color='#E61A27', alpha=0.3, 
            label=r'$\textrm{red}_{\textrm{d}}$')
        smf_data_blue = ax.fill_between(blue_data[0], 
            np.log10(blue_data[1])-blue_data[2], 
            np.log10(blue_data[1])+blue_data[2], color='#5696B9', alpha=0.3, 
            label=r'$\textrm{blue}_{\textrm{d}}$') 
        ax.set_ylim(-5,-1)     
        ax.minorticks_on()
        if ax == ax3 or ax == ax4:
            ax.set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', 
            fontsize=15)
        if ax == ax1 or ax == ax3:
            ax.set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', 
            fontsize=15)
        ax.label_outer()

    rel_err = 0.434*(red_model1[2]/red_model1[1])
    line1 = ax1.errorbar(red_model1[0], np.log10(red_model1[1]), yerr=rel_err,
        color='#E61A27', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#E61A27', capsize=2, capthick=1.5, markersize='3') 
    rel_err = 0.434*(red_model2[2]/red_model2[1])
    line2 = ax2.errorbar(red_model2[0], np.log10(red_model2[1]), yerr=rel_err,
        color='#E61A27', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#E61A27', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(red_model3[2]/red_model3[1])
    line3 = ax3.errorbar(red_model3[0], np.log10(red_model3[1]), yerr=rel_err,
        color='#E61A27', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#E61A27', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(red_model4[2]/red_model4[1])
    line4 = ax4.errorbar(red_model4[0], np.log10(red_model4[1]), yerr=rel_err,
        color='#E61A27', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#E61A27', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(blue_model1[2]/blue_model1[1])
    line5 = ax1.errorbar(blue_model1[0], np.log10(blue_model1[1]), yerr=rel_err,
        color='#5696B9', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#5696B9', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(blue_model2[2]/blue_model2[1])
    line6 = ax2.errorbar(blue_model2[0], np.log10(blue_model2[1]), yerr=rel_err,
        color='#5696B9', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#5696B9', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(blue_model3[2]/blue_model3[1])
    line7 = ax3.errorbar(blue_model3[0], np.log10(blue_model3[1]), yerr=rel_err,
        color='#5696B9', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#5696B9', capsize=2, capthick=1.5, markersize='3')
    rel_err = 0.434*(blue_model4[2]/blue_model4[1])
    line8 = ax4.errorbar(blue_model4[0], np.log10(blue_model4[1]), yerr=rel_err,
        color='#5696B9', fmt="s-", linewidth=2, elinewidth=0.5, 
        ecolor='#5696B9', capsize=2, capthick=1.5, markersize='3')

    if model == 'hybrid':
        ax1.legend([line1,smf_data_red,smf_data_blue],
            [r'$M_{h}^{q}=%4.2f$' % Mhalo, 
            r'$\textrm{red}_{\textrm{d}}$', r'$\textrm{blue}_{\textrm{d}}$'], 
            loc='lower left',prop={'size': 10})
        ax2.legend([line2],[r'$M_{*}^{q}=%4.2f$' % Mstar],
            loc='lower left',prop={'size': 10})
        ax3.legend([line3],[r'$\mu=%4.2f$' % mu], 
            loc='lower left',prop={'size': 10})
        ax4.legend([line4],[r'$\nu=%4.2f$' % nu], 
            loc='lower left',prop={'size': 10})

    elif model == 'halo':
        ax1.legend([line1,smf_data_red,smf_data_blue],
            [r'$M_{h}^{qc}=%4.2f$' % Mhalo_c, 
            r'$\textrm{red}_{\textrm{d}}$', r'$\textrm{blue}_{\textrm{d}}$'], 
            loc='lower left',prop={'size': 10})
        ax2.legend([line2],[r'$M_{h}^{qs}=%4.2f$' % Mhalo_s],
            loc='lower left',prop={'size': 10})
        ax3.legend([line3],[r'${\mu}^{c}=%4.2f$' % mu_c], 
            loc='lower left',prop={'size': 10})
        ax4.legend([line4],[r'${\mu}^{s}=%4.2f$' % mu_s], 
            loc='lower left',prop={'size': 10})

    print('Setting data')
    if model == 'hybrid':
        array_for_counter = Mh_q_arr
    elif model == 'halo':
        array_for_counter = Mh_qc_arr
    print('Frame {0}/{1}'.format(counter+1,len(array_for_counter)))
    
    counter+=1
    
    line = [line1,line2,line3,line4,line5,line6,line7,line8]  
    
    return line

#Setting up the figure, the axis, and the plot element we want to animate
fig, axs = plt.subplots(2,2, figsize=(12,8), sharex='row', sharey='col', 
    gridspec_kw={'hspace': 0.1, 'wspace': 0.0})
(ax1, ax2), (ax3, ax4) = axs
if model == 'hybrid':
    fig.suptitle(r'Hybrid quenching model')
elif model == 'halo':
    fig.suptitle(r'Halo quenching model')

if model == 'hybrid':
    anim = animation.FuncAnimation(plt.gcf(), make_animation, Mh_q_arr, 
        init_func=init,fargs=(Mstar_q_arr, mu_arr, nu_arr, ax1_catalog, 
        ax2_catalog, ax3_catalog, ax4_catalog,), interval=1000, blit=False, 
        repeat=True)
elif model == 'halo':
    anim = animation.FuncAnimation(plt.gcf(), make_animation, Mh_qc_arr, 
        init_func=init,fargs=(Mh_qs_arr, muc_arr, mus_arr, ax1_catalog, 
        ax2_catalog, ax3_catalog, ax4_catalog,), interval=1000, blit=False, 
        repeat=True)
print('Saving animation')
os.chdir(path_to_figures)
if model == 'hybrid':
    anim.save('{0}_smf_hybrid_test.gif'.format(survey),writer='imagemagick',fps=1)
elif model == 'halo':
    anim.save('{0}_smf_halo_test.gif'.format(survey),writer='imagemagick',fps=1)