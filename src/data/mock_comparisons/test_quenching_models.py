"""
{This script tests given values of the Zu and Mandelbaum hybrid quenching model
 parameters to see how the resulting SMF for blue and red model galaxies 
 as well as spread in velocity difference of galaxies around red and blue 
 centrals compares to that of data}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)


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
    i = 0
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        cen_deltav_arr = []
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])
        N = len(cen_deltav_arr)
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

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
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
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values

            #Measure SMF of mock using diff_smf function
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(logmstar_arr, volume, h1_bool=False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
                volume, h1_bool=False, colour_flag='R')
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
                volume, h1_bool=False, colour_flag='B')
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

    # phi_arr_colour = np.append(phi_arr_red, phi_arr_blue, axis = 0)

    # Covariance matrix for total phi (all galaxies)
    # cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    # err_total = np.sqrt(cov_mat.diagonal())
    # cov_mat_red = np.cov(phi_arr_red, rowvar=False) # default norm is N-1
    # err_red = np.sqrt(cov_mat_red.diagonal())
    # colour_err_arr.append(err_red)
    # cov_mat_blue = np.cov(phi_arr_blue, rowvar=False) # default norm is N-1
    # err_blue = np.sqrt(cov_mat_blue.diagonal())
    # colour_err_arr.append(err_blue)

    # corr_mat_red = cov_mat_red / np.outer(err_red , err_red)
    # corr_mat_inv_red = np.linalg.inv(corr_mat_red)
    # colour_corr_mat_inv.append(corr_mat_inv_red)
    # corr_mat_blue = cov_mat_blue / np.outer(err_blue , err_blue)
    # corr_mat_inv_blue = np.linalg.inv(corr_mat_blue)
    # colour_corr_mat_inv.append(corr_mat_inv_blue)

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

    # deltav_sig_colour = np.append(deltav_sig_red, deltav_sig_blue, axis = 0)
    # cov_mat_colour = np.cov(phi_arr_colour,deltav_sig_colour, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    # cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    return err_colour, corr_mat_inv_colour

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
    Isolates 68th percentile lowest chi^2 values and takes random 100 sample

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
    subset: ndarray
        Random 100 sample of param values from 68th percentile
    """ 
    percentile = percentile/100
    table['chi2'] = chi2_arr
    table = table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(percentile*len(table))
    mcmc_table_pctl = table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:5]
    subset = mcmc_table_pctl.drop_duplicates().sample(100).values[:,:5] 
    subset = np.insert(subset, 0, bf_params, axis=0)

    return subset

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
                cen_gals.append((10**(df.logmstar.values[idx]))/2.041)
            elif value == 0:
                sat_gals.append((10**(df.logmstar.values[idx]))/2.041)

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

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
            diff_smf(table[logmstar_col], volume, h1_bool=False)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, h1_bool=False, colour_flag='R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, h1_bool=False, colour_flag='B')
    else:
        # logmstar_col = 'stellar_mass'
        logmstar_col = '{0}'.format(randint_logmstar)
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, h1_bool=True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
            volume,h1_bool=True, colour_flag='R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
            volume, h1_bool=True, colour_flag='B')
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

global survey
global machine
global mf_type
global ver

survey = 'eco'
machine = 'mac'
mf_type = 'smf'
ver = 2.0

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

if survey == 'eco':
    # catl_file = path_to_raw + "eco/eco_all.csv"
    catl_file = path_to_proc + "gal_group_eco_data.hdf5"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

catl, volume, z_median = read_data_catl(catl_file, survey)
gal_group_df = reading_catls(path_to_proc + "gal_group.hdf5") 

## Use only the mocks that are in the random sample of 100
# Count the first 20 + 22nd + 123-131 columns of general information from 
# mock catalog (halo + rsd)
idx_arr = np.insert(np.linspace(0,20,21), len(np.linspace(0,20,21)), (22, 123, 
    124, 125, 126, 127, 128, 129, 130, 131)).astype(int)

mock_to_test = 70

names_arr = [x for x in gal_group_df.columns.values[idx_arr]]
names_arr.append('{0}_y'.format(mock_to_test))
names_arr.append('groupid_{0}'.format(mock_to_test))
names_arr.append('g_galtype_{0}'.format(mock_to_test))
names_arr = np.array(names_arr)

gal_group_df_subset = gal_group_df[names_arr]

# Renaming the "1_y" column kept from line 1896 because of case where it was
# also in mcmc_table_ptcl.mock_num and was selected twice
gal_group_df_subset.columns.values[30] = "behroozi_bf"

gal_group_df_subset = gal_group_df_subset.rename(columns=\
    {'{0}_y'.format(mock_to_test):'{0}'.format(mock_to_test)})

print('Assigning centrals/satellites flag')
gals_df = assign_cen_sat_flag(gal_group_df_subset)

print('Assigning colour to data')
catl = assign_colour_label_data(catl)

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

print('Measuring spread in vel disp for data')
std_red, centers_red, std_blue, centers_blue = get_deltav_sigma_data(catl)

print('Measuring error in data from mocks')
err_colour_data, colour_corr_mat_inv = \
    get_err_data(survey, path_to_mocks)

# params for outside and within 1 sigma for unlog nu values
# params = [10**10.218907,  10**12.929771,  0.521268, 7.652937] 
# params = [10**10.218907,  10**12.929771,  0.521268 , 105.746357]

#params for well outside, midway outside and within 1 sigma for log nu values
# params = [10**10.030012,  10**13.761830,  0.665974, -1.165710]
# params = [10**10.159325,  10**12.271083,  0.80802,  0.573578]
# params = [10**9.991129,  10**13.403738,  0.594571,  2.301931]

# Testing extreme values from smhm_colour_run17 (without prior on Mh_q and using
# both observables)
theta = [10.031071, 94.612836, 0.471234, 0.576365] #extreme value for second
theta = [9.987552, 14.343890, 0.472247, 4.266467] #extreme value for fourth
theta = [10.008105, 132.188023, 0.482480, 4.621764] #extreme values for both

f_red_cen, f_red_sat = hybrid_quenching_model(theta, gals_df, 'vishnu',\
    mock_to_test)
gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
v_sim = 890641.5172927063
total_model, red_model, blue_model = measure_all_smf(gals_df, v_sim 
    , False, mock_to_test)    
std_red_model, std_blue_model, centers_red_model, centers_blue_model = \
    get_deltav_sigma_vishnu_qmcolour(gals_df, mock_to_test)


maxis_red_data, phi_red_data = red_data[0], red_data[1]
maxis_blue_data, phi_blue_data = blue_data[0], blue_data[1]
maxis_red_model, phi_red_model = red_model[0], red_model[1]
maxis_blue_model, phi_blue_model = blue_model[0], blue_model[1]


fig1 = plt.figure()

dr = plt.errorbar(maxis_red_data,phi_red_data,yerr=err_colour_data[0:5],
    color='darkred',fmt='s-',ecolor='darkred',markersize=5,capsize=5,
    capthick=0.5,zorder=10)
db = plt.errorbar(maxis_blue_data,phi_blue_data,yerr=err_colour_data[5:10],
    color='darkblue',fmt='s-',ecolor='darkblue',markersize=5,capsize=5,
    capthick=0.5,zorder=10)
mr, = plt.plot(maxis_red_model,phi_red_model,
    color='lightcoral',ls='--',lw=3,zorder=10)
mb, = plt.plot(maxis_blue_model,phi_blue_model,
    color='cornflowerblue',ls='--',lw=3,zorder=10)

plt.legend([(dr, db), (mr, mb)], ['Data','Model'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.4)})

plt.xlabel(r'$\boldmath\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'$\boldmath\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

plt.annotate(r'$\mathbf{M^{q}_{*}} \approx$'+' {0} \n'.format\
    (np.round(theta[0],2))+
    r'$\mathbf{M^{q}_{h}} \approx$'+' {0} \n'.format(np.round(theta[1],2))+
    r'$\boldsymbol{\mu} \approx$'+' {0} \n'.format(np.round(theta[2],2))+
    r'$\boldsymbol{\nu} \approx$'+' {0}'.format(np.round(theta[3],2)), 
    xy=(0.9, 0.70), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    ec='k', fc='lightgray', alpha=0.5), size=25)

# plt.title(r'ECO SMF comparison for extreme value of $\mathbf{M^{q}_{h}}$')
# plt.title(r'ECO SMF comparison for extreme value of $\boldsymbol{\nu}$')
plt.title(r'ECO SMF comparison for extreme value of $\boldsymbol{\nu}$ \& $\mathbf{M^{q}_{h}}$')

plt.show()


fig2= plt.figure(figsize=(10,10))

mr, = plt.plot(centers_red_model,std_red_model,
    color='lightcoral',ls='--',lw=3,zorder=10)
mb, = plt.plot(centers_blue_model,std_blue_model,
    color='cornflowerblue',ls='--',lw=3,zorder=10)

dr = plt.errorbar(centers_red,std_red,yerr=err_colour_data[10:15],
    color='darkred',fmt='p-',ecolor='darkred',markersize=10,capsize=10,
    capthick=1.0,zorder=10)
db = plt.errorbar(centers_blue,std_blue,yerr=err_colour_data[15:20],
    color='darkblue',fmt='p-',ecolor='darkblue',markersize=10,capsize=10,
    capthick=1.0,zorder=10)

plt.legend([(dr, db), (mr, mb)], 
    ['Data','Models'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.4)}, markerscale=1.5)

plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)

plt.annotate(r'$\mathbf{M^{q}_{*}} \approx$'+' {0} \n'.format\
    (np.round(theta[0],2))+
    r'$\mathbf{M^{q}_{h}} \approx$'+' {0} \n'.format(np.round(theta[1],2))+
    r'$\boldsymbol{\mu} \approx$'+' {0} \n'.format(np.round(theta[2],2))+
    r'$\boldsymbol{\nu} \approx$'+' {0}'.format(np.round(theta[3],2)), 
    xy=(0.02, 0.70), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    ec='k', fc='lightgray', alpha=0.5), size=25)

# plt.title(r'ECO spread in velocity difference comparison for extreme value of $\mathbf{M^{q}_{h}}$')
# plt.title(r'ECO spread in velocity difference comparison for extreme value of $\boldsymbol{\nu}$')
plt.title(r'ECO spread in velocity difference comparison for extreme value of $\boldsymbol{\nu}$ \& $\mathbf{M^{q}_{h}}$')

plt.show()


    