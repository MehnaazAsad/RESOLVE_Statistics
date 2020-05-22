"""
{This script calculates velocity dispersion for red and blue galaxies from 
 Vishnu mock and compares with measurement from data.}
"""

from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import rc
import pandas as pd
import numpy as np
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

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
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2

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
    for idx,value in enumerate(df['g_galtype']):
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
    for idx,value in enumerate(df['g_galtype']):
        if value == 1:
            cen_gals.append(df['stellar_mass'][idx])
        elif value == 0:
            sat_gals.append(df['stellar_mass'][idx])

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

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
    df.loc[df['g_galtype'] == 1, 'f_red'] = f_red_cen
    df.loc[df['g_galtype'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['g_galtype']):
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

def get_deltav(df, df_type):
    """
    Measure velocity dispersion separately for red and blue galaxies by 
    binning up central stellar mass

    Parameters
    ----------
    catl: pandas Dataframe 
        Data/mock catalog
    df_type: string
        'data' or 'mock' depending on type of catalog

    Returns
    ---------
    deltav_red: numpy array
        Velocity dispersion of red galaxies
    centers_red: numpy array
        Bin centers of central stellar mass for red galaxies
    deltav_blue: numpy array
        Velocity dispersion of blue galaxies
    centers_blue: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    catl = df.copy()

    if df_type == 'data':
        grp_id_colname = 'grp'
        cs_colname = 'fc'
        mstar_colname = 'logmstar'
        cz_colname = 'cz'
        # Change to h=1
        catl[mstar_colname] = np.log10((10**catl[mstar_colname]) / 2.041)
    elif df_type == 'mock':
        grp_id_colname = 'groupid'
        cs_colname = 'g_galtype'
        mstar_colname = 'stellar_mass'
        cz_colname = 'cz'
        catl[mstar_colname] = np.log10(catl[mstar_colname])

    red_subset = catl.loc[catl.colour_label == 'R']
    blue_subset = catl.loc[catl.colour_label == 'B']
    
    red_groups = red_subset.groupby(grp_id_colname)
    red_keys = red_groups.groups.keys()

    # Calculating velocity dispersion for red galaxies in data
    deltav_arr = []
    cen_stellar_mass_arr = []
    for key in red_keys: 
        group = red_groups.get_group(key) 
        if 1 in group[cs_colname].values: 
            cen_stellar_mass = group[mstar_colname].loc[group[cs_colname].\
                values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group[cz_colname].values),2)
            deltav = group[cz_colname].values - len(group)*[mean_cz_grp]
            for val in deltav:
                deltav_arr.append(val)
                cen_stellar_mass_arr.append(cen_stellar_mass)

    red_bin_min = np.log10((10**8.55) / 2.041)
    red_bin_max = np.log10((10**11.5) / 2.041)
    deltav_binned_red = binned_statistic(cen_stellar_mass_arr, deltav_arr,
        bins=np.linspace(red_bin_min,red_bin_max,6)) 
    deltav_red = deltav_binned_red[0]
    edges_red = deltav_binned_red[1]

    blue_groups = blue_subset.groupby(grp_id_colname)
    blue_keys = blue_groups.groups.keys()

    # Calculating velocity dispersion for blue galaxies in data
    deltav_arr = []
    cen_stellar_mass_arr = []
    for key in blue_keys: 
        group = blue_groups.get_group(key)  
        if 1 in group[cs_colname].values: 
            cen_stellar_mass = group[mstar_colname].loc[group[cs_colname]\
                .values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group[cz_colname].values),2)
            deltav = group[cz_colname].values - len(group)*[mean_cz_grp]
            for val in deltav:
                deltav_arr.append(val)
                cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_bin_min = np.log10((10**8.55) / 2.041)
    blue_bin_max = np.log10((10**10.5) / 2.041)
    deltav_binned_blue = binned_statistic(cen_stellar_mass_arr, deltav_arr,
        bins=np.linspace(blue_bin_min,blue_bin_max,6))             
    deltav_blue = deltav_binned_blue[0]
    edges_blue = deltav_binned_blue[1]

    centers_red = 0.5 * (edges_red[1:] + edges_red[:-1])
    centers_blue = 0.5 * (edges_blue[1:] + edges_blue[:-1])

    return deltav_red, deltav_blue, centers_red, centers_blue

def get_deltav_mocks(survey, path):
    """
    Calculate error in data VD from mocks

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    deltav_binned_red_arr: numpy array
        Velocity dispersion of red galaxies
    centers_binned_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    deltav_binned_blue_arr: numpy array
        Velocity dispersion of blue galaxies
    centers_binned_blue_arr: numpy array
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

    deltav_binned_red_arr = []
    edges_binned_red_arr = []
    deltav_binned_blue_arr = []
    edges_binned_blue_arr = []
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

            logmstar_arr = mock_pd.logmstar.values 
            u_r_arr = mock_pd.u_r.values

            colour_label_arr = np.empty(len(mock_pd), dtype='str')
            # Using defintions from Moffett paper
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
            mock_pd.logmstar = np.log10((10**mock_pd.logmstar) / 2.041)
            red_subset = mock_pd.loc[mock_pd.colour_label == 'R']
            blue_subset = mock_pd.loc[mock_pd.colour_label == 'B']

            red_groups = red_subset.groupby('groupid')
            red_keys = red_groups.groups.keys()

            # Calculating velocity dispersion for red galaxies in mock
            deltav_arr = []
            cen_stellar_mass_arr = []
            for key in red_keys: 
                group = red_groups.get_group(key) 
                if 1 in group.cs_flag.values: 
                    cen_stellar_mass = group.logmstar.loc[group.cs_flag.values \
                        == 1].values[0]
                    mean_cz_grp = np.round(np.mean(group.cz.values),2)
                    deltav = group.cz.values - len(group)*[mean_cz_grp]
                    for val in deltav:
                        deltav_arr.append(val)
                        cen_stellar_mass_arr.append(cen_stellar_mass)

            deltav_binned_red = binned_statistic(cen_stellar_mass_arr, deltav_arr,
                bins=np.linspace(8.55,11.5,6)) 
            deltav_binned_red_arr.append(deltav_binned_red[0]) 
            edges_binned_red_arr.append(deltav_binned_red[1])

            blue_groups = blue_subset.groupby('groupid')
            blue_keys = blue_groups.groups.keys()

            # Calculating velocity dispersion for blue galaxies in mock
            deltav_arr = []
            cen_stellar_mass_arr = []
            for key in blue_keys: 
                group = blue_groups.get_group(key)  
                if 1 in group.cs_flag.values:
                    cen_stellar_mass = group.logmstar.loc[group.cs_flag.values \
                        == 1].values[0]
                    mean_cz_grp = np.round(np.mean(group.cz.values),2)
                    deltav = group.cz.values - len(group)*[mean_cz_grp]
                    for val in deltav:
                        deltav_arr.append(val)
                        cen_stellar_mass_arr.append(cen_stellar_mass)
            
            deltav_binned_blue = binned_statistic(cen_stellar_mass_arr, deltav_arr,
                bins=np.linspace(8.55,10.5,6))             
            deltav_binned_blue_arr.append(deltav_binned_blue[0])
            edges_binned_blue_arr.append(deltav_binned_blue[1])

    deltav_binned_red_arr = np.array(deltav_binned_red_arr)
    edges_binned_red_arr = np.array(edges_binned_red_arr)
    deltav_binned_blue_arr = np.array(deltav_binned_blue_arr)
    edges_binned_blue_arr = np.array(edges_binned_blue_arr)

    # Getting bin centers for plotting instead of bin edges
    edg_red = edges_binned_red_arr[0]
    edg_blue = edges_binned_blue_arr[0]
    centers_binned_red_arr = 0.5 * (edg_red[1:] + edg_red[:-1])
    centers_binned_blue_arr = 0.5 * (edg_blue[1:] + edg_blue[:-1])
    centers_binned_red_arr = 64*[centers_binned_red_arr]
    centers_binned_blue_arr = 64*[centers_binned_blue_arr]

    return deltav_binned_red_arr, deltav_binned_blue_arr, centers_binned_red_arr,\
        centers_binned_blue_arr

global survey
global mf_type

survey = 'eco'
machine = 'mac'
mf_type = 'smf'

dict_of_paths = cwpaths.cookiecutter_paths() 
path_to_raw = dict_of_paths['raw_dir'] 
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']

catl_file = path_to_raw + "eco/eco_all.csv"

chi2_file = path_to_proc + 'smhm_colour_run10/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run10/mcmc_{0}_colour_raw.txt'.\
    format(survey)

path_to_mocks = path_to_data + 'mocks/m200b/eco/'

gal_group_df = pd.read_hdf(path_to_proc + 'gal_group.hdf5')
group_df = pd.read_hdf(path_to_proc + 'group.hdf5')

catl, volume, cvar, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)

chi2 = read_chi2(chi2_file)

mcmc_table = read_mcmc(chain_file)

mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)

f_red_cen, f_red_sat = hybrid_quenching_model(bf_params, gal_group_df)
gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gal_group_df)

deltav_red_data, deltav_blue_data, centers_red_data, \
    centers_blue_data = get_deltav(catl, 'data')

deltav_red_vishnu, deltav_blue_vishnu, centers_red_vishnu, \
    centers_blue_vishnu = get_deltav(gals_df, 'mock')

deltav_red_mocks, deltav_blue_mocks, centers_red_mocks, \
    centers_blue_mocks = get_deltav_mocks(survey, path_to_mocks)

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=False,figsize=(12,10),\
     gridspec_kw = {'height_ratios':[7,3]})
# for idx in range(len(centers_red_mocks)):
#     plt.scatter(centers_red_mocks[idx], deltav_red_mocks[idx], 
#         c='indianred')
#     plt.scatter(centers_blue_mocks[idx], deltav_blue_mocks[idx], 
#         c='cornflowerblue')
plt.sca(ax1) # required for plt.gca() to pick up labels
plt.scatter(centers_red_data, deltav_red_data, marker='*', c='indianred', \
    s=100, label='data')
plt.scatter(centers_blue_data, deltav_blue_data, marker='*', \
    c='cornflowerblue', s=100, label='data')
plt.scatter(centers_red_vishnu, deltav_red_vishnu, marker='o', \
    facecolors='none', edgecolors='indianred', s=100, label='vishnu')
plt.scatter(centers_blue_vishnu, deltav_blue_vishnu, marker='o', \
    facecolors='none', edgecolors='cornflowerblue', s=100, label='vishnu')
plt.ylabel(r'\boldmath$<\Delta v\ > \left[km/s\right]$', labelpad=15, 
    fontsize=25)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))

red_residual = deltav_red_vishnu/deltav_red_data
blue_residual = deltav_blue_vishnu/deltav_blue_data

ax2.scatter(centers_red_data,red_residual,c='indianred')
ax2.scatter(centers_blue_data,blue_residual,c='cornflowerblue')
xmin, xmax = ax2.get_xlim()
ax2.plot(np.linspace(xmin, xmax, 6), 6*[0], ls='--', c='k')
ax2.set_ylim(-2,1)
ax2.set_xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$', labelpad=15, 
    fontsize=25)
ax2.set_ylabel(r'$\mathbf{\frac{vishnu}{data}}$', labelpad=30, 
    fontsize=25)

ax1.set_title(r'Vishnu mock vs. data mean velocity dispersion')
ax1.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
fig.tight_layout()
fig.show()