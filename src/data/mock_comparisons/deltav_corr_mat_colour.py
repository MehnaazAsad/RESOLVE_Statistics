"""
{This script calculates spread in velocity dispersion (sigma) from mocks for 
 red and blue galaxies as well as smf for red and blue galaxies. It then 
 calculates a full correlation matrix using sigma and smf of both galaxy 
 populations as well as a correlation matrix of just sigma
 measurements of both galaxy populations.
 
 Mean velocity dispersion of red and blue galaxies in bins of central stellar
 mass from data and mocks is also compared.}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import normaltest as nt
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import rc
from scipy.stats import binned_statistic as bs
import random
import pandas as pd
import numpy as np
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
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

def std_func(bins, mass_arr, vel_arr):
    ## Calculate std from mean=0

    last_index = len(bins)-1
    i = 0
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        cen_deltav_arr = []
        if index1 == last_index:
            break
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

def std_func_mod(bins, mass_arr, vel_arr):
    mass_arr_bin_idxs = np.digitize(mass_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean = 0
    std_arr = []
    for idx in range(1, len(bins)):
        cen_deltav_arr = []
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        cen_deltav_arr.append(np.array(vel_arr)[current_bin_idxs])
        
        diff_sqrd_arr = []
        # mean = np.mean(cen_deltav_arr)
        for value in cen_deltav_arr:
            # print(mean)
            # print(np.mean(cen_deltav_arr))
            diff = value - mean
            diff_sqrd = diff**2
            diff_sqrd_arr.append(diff_sqrd)
        mean_diff_sqrd = np.mean(diff_sqrd_arr)
        std = np.sqrt(mean_diff_sqrd)
        # print(std)
        # print(np.std(cen_deltav_arr))
        std_arr.append(std)

    return std_arr

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
    std_red = std_func_mod(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
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
    std_blue = std_func_mod(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
        blue_deltav_arr)    
    std_blue = np.array(std_blue)

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])

    return std_red, centers_red, std_blue, centers_blue

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

    mean_stats_red = bs(red_sigma_arr, red_cen_stellar_mass_arr, 
        statistic='mean', bins=np.linspace(0,250,6))
    mean_stats_blue = bs(blue_sigma_arr, blue_cen_stellar_mass_arr, 
        statistic='mean', bins=np.linspace(0,250,6))

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    return mean_stats_red, centers_red, mean_stats_blue, centers_blue

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

def get_err_smf_mocks(survey, path):
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
    max_arr_red = []
    max_arr_blue = []
    colour_corr_mat_inv = []
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
            max_arr_red.append(max_red)
            max_arr_blue.append(max_blue)

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    max_arr_red = np.array(max_arr_red)
    max_arr_blue = np.array(max_arr_blue)

    # Covariance matrix for total phi (all galaxies)
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    err_total = np.sqrt(cov_mat.diagonal())
    
    return phi_arr_red, phi_arr_blue

def get_deltav_sigma_mocks_urcolour(survey, path):
    """
    Calculate spread in velocity dispersion from survey mocks

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

    std_red_arr = []
    centers_red_arr = []
    std_blue_arr = []
    centers_blue_arr = []
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
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
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
                red_stellar_mass_bins = np.linspace(8.6,11.5,6)
            elif survey == 'resolveb':
                red_stellar_mass_bins = np.linspace(8.4,11.0,6)
            std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
                red_deltav_arr)
            std_red = np.array(std_red)
            std_red_arr.append(std_red)

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
                blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
            elif survey == 'resolveb':
                blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
            std_blue = std_func(blue_stellar_mass_bins, \
                blue_cen_stellar_mass_arr, blue_deltav_arr)    
            std_blue = np.array(std_blue)
            std_blue_arr.append(std_blue)

            centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
                red_stellar_mass_bins[:-1])
            centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
                blue_stellar_mass_bins[:-1])
            
            centers_red_arr.append(centers_red)
            centers_blue_arr.append(centers_blue)
    
    std_red_arr = np.array(std_red_arr)
    centers_red_arr = np.array(centers_red_arr)
    std_blue_arr = np.array(std_blue_arr)
    centers_blue_arr = np.array(centers_blue_arr)
            
    return std_red_arr, std_blue_arr, centers_red_arr, centers_blue_arr

def get_deltav_sigma_mocks_qmcolour_mod(survey, path):
    """
    Calculate spread in velocity dispersion from survey mocks

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

    std_red_arr = []
    centers_red_arr = []
    std_blue_arr = []
    centers_blue_arr = []
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
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
                (mock_pd.logmstar.values >= mstar_limit)]

            f_red_c, f_red_s = hybrid_quenching_model(mock_pd)
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

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
            std_red_arr.append(std_red)

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
            std_blue_arr.append(std_blue)

            centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
                red_stellar_mass_bins[:-1])
            centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
                blue_stellar_mass_bins[:-1])
            
            centers_red_arr.append(centers_red)
            centers_blue_arr.append(centers_blue)
    
    std_red_arr = np.array(std_red_arr)
    centers_red_arr = np.array(centers_red_arr)
    std_blue_arr = np.array(std_blue_arr)
    centers_blue_arr = np.array(centers_blue_arr)
            
    return std_red_arr, std_blue_arr, centers_red_arr, centers_blue_arr

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
    std_red = std_func_mod(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
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
    std_blue = std_func_mod(blue_stellar_mass_bins, \
        blue_cen_stellar_mass_arr, blue_deltav_arr)    
    std_blue = np.array(std_blue)

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
                        
    centers_red = np.array(centers_red)
    centers_blue = np.array(centers_blue)
            
    return std_red, std_blue, centers_red, centers_blue

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

def get_sigma_per_group_mocks_qmcolour(survey, mock_df):
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

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_pd.loc[mock_pd.groupid == key]
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
        group = mock_pd.loc[mock_pd.groupid == key]
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

    mean_stats_red = bs(red_sigma_arr, red_cen_stellar_mass_arr, 
        statistic='mean', bins=np.linspace(0,250,6))
    mean_stats_blue = bs(blue_sigma_arr, blue_cen_stellar_mass_arr, 
        statistic='mean', bins=np.linspace(0,250,6))

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])
            
    return mean_stats_red, centers_red, mean_stats_blue, centers_blue

def get_err_data_mod(survey, path):
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
    mean_cen_arr_red = []
    mean_cen_arr_blue = []
    new_sig_arr_red = []
    new_sig_arr_blue = []
    box_id_arr = np.linspace(5001,5008,8)
    subset_counter = 0
    while subset_counter < 15:
        print(subset_counter)
    # for box in box_id_arr:
        box = random.randint(5001,5008)
        num = random.randint(0,7)
        # box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        # for num in range(num_mocks):
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

        ## Using best-fit found for new ECO data using optimize_qm_eco.py
        ## for hybrid quenching model
        Mstar_q = 10.49 # Msun/h
        Mh_q = 14.03 # Msun/h
        mu = 0.69
        nu = 0.148

        ## Using best-fit found for new ECO data using optimize_qm_eco.py 
        ## for halo quenching model
        Mh_qc = 12.61 # Msun/h
        Mh_qs = 13.5 # Msun/h
        mu_c = 0.40
        mu_s = 0.148

        if quenching == 'hybrid':
            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
        elif quenching == 'halo':
            theta = [Mh_qc, Mh_qs, mu_c, mu_s]
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

        sig_red, sig_blue, cen_red_sig, cen_blue_sig = \
            get_deltav_sigma_mocks_qmcolour(survey, mock_pd)

        new_mean_stats_red, new_centers_red, new_mean_stats_blue, \
            new_centers_blue = \
            get_sigma_per_group_mocks_qmcolour(survey, mock_pd)

        sig_arr_red.append(sig_red)
        sig_arr_blue.append(sig_blue)
        cen_arr_red.append(cen_red_sig)
        cen_arr_blue.append(cen_blue_sig)

        new_sig_arr_red.append(new_centers_red)
        new_sig_arr_blue.append(new_centers_blue)
        mean_cen_arr_red.append(new_mean_stats_red[0])
        mean_cen_arr_blue.append(new_mean_stats_blue[0])

        subset_counter += 1

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    sig_arr_red = np.array(sig_arr_red)
    sig_arr_blue = np.array(sig_arr_blue)
    cen_arr_red = np.array(cen_arr_red)
    cen_arr_blue = np.array(cen_arr_blue)
    new_sig_arr_red = np.array(new_sig_arr_red)
    new_sig_arr_blue = np.array(new_sig_arr_blue)
    mean_cen_arr_red = np.array(mean_cen_arr_red)
    mean_cen_arr_blue = np.array(mean_cen_arr_blue)

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

    av_grpcen_red_0 = mean_cen_arr_red[:,0]
    av_grpcen_red_1 = mean_cen_arr_red[:,1]
    av_grpcen_red_2 = mean_cen_arr_red[:,2]
    av_grpcen_red_3 = mean_cen_arr_red[:,3]
    av_grpcen_red_4 = mean_cen_arr_red[:,4]

    av_grpcen_blue_0 = mean_cen_arr_blue[:,0]
    av_grpcen_blue_1 = mean_cen_arr_blue[:,1]
    av_grpcen_blue_2 = mean_cen_arr_blue[:,2]
    av_grpcen_blue_3 = mean_cen_arr_blue[:,3]
    av_grpcen_blue_4 = mean_cen_arr_blue[:,4]



    combined_df = pd.DataFrame({'phi_red_0':phi_red_0, 'phi_red_1':phi_red_1,\
        'phi_red_2':phi_red_2, 'phi_red_3':phi_red_3, 'phi_red_4':phi_red_4, \
        'phi_blue_0':phi_blue_0, 'phi_blue_1':phi_blue_1, 
        'phi_blue_2':phi_blue_2, 'phi_blue_3':phi_blue_3, 
        'phi_blue_4':phi_blue_4, \
        'dv_red_0':dv_red_0, 'dv_red_1':dv_red_1, 'dv_red_2':dv_red_2, \
        'dv_red_3':dv_red_3, 'dv_red_4':dv_red_4, \
        'dv_blue_0':dv_blue_0, 'dv_blue_1':dv_blue_1, 'dv_blue_2':dv_blue_2, \
        'dv_blue_3':dv_blue_3, 'dv_blue_4':dv_blue_4, \
        'av_grpcen_red_0':av_grpcen_red_0, 'av_grpcen_red_1':av_grpcen_red_1, \
        'av_grpcen_red_2':av_grpcen_red_2, 'av_grpcen_red_3':av_grpcen_red_3, \
        'av_grpcen_red_4':av_grpcen_red_4, 'av_grpcen_blue_0':av_grpcen_blue_0,\
        'av_grpcen_blue_1':av_grpcen_blue_1, 'av_grpcen_blue_2':av_grpcen_blue_2, \
        'av_grpcen_blue_3':av_grpcen_blue_3, 'av_grpcen_blue_4':av_grpcen_blue_4 })



    # deltav_sig_colour = np.append(deltav_sig_red, deltav_sig_blue, axis = 0)
    # cov_mat_colour = np.cov(phi_arr_colour,deltav_sig_colour, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    # cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    return combined_df

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
    mean_cen_arr_red = []
    mean_cen_arr_blue = []
    new_sig_arr_red = []
    new_sig_arr_blue = []
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

            ## Using best-fit found for new ECO data using optimize_qm_eco.py
            ## for hybrid quenching model
            Mstar_q = 10.49 # Msun/h
            Mh_q = 14.03 # Msun/h
            mu = 0.69
            nu = 0.148

            ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            ## for halo quenching model
            Mh_qc = 12.61 # Msun/h
            Mh_qs = 13.5 # Msun/h
            mu_c = 0.40
            mu_s = 0.148

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
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

            sig_red, sig_blue, cen_red_sig, cen_blue_sig = \
                get_deltav_sigma_mocks_qmcolour(survey, mock_pd)

            new_mean_stats_red, new_centers_red, new_mean_stats_blue, \
                new_centers_blue = \
                get_sigma_per_group_mocks_qmcolour(survey, mock_pd)

            sig_arr_red.append(sig_red)
            sig_arr_blue.append(sig_blue)
            cen_arr_red.append(cen_red_sig)
            cen_arr_blue.append(cen_blue_sig)

            new_sig_arr_red.append(new_centers_red)
            new_sig_arr_blue.append(new_centers_blue)
            mean_cen_arr_red.append(new_mean_stats_red[0])
            mean_cen_arr_blue.append(new_mean_stats_blue[0])

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    sig_arr_red = np.array(sig_arr_red)
    sig_arr_blue = np.array(sig_arr_blue)
    cen_arr_red = np.array(cen_arr_red)
    cen_arr_blue = np.array(cen_arr_blue)
    new_sig_arr_red = np.array(new_sig_arr_red)
    new_sig_arr_blue = np.array(new_sig_arr_blue)
    mean_cen_arr_red = np.array(mean_cen_arr_red)
    mean_cen_arr_blue = np.array(mean_cen_arr_blue)

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

    av_grpcen_red_0 = mean_cen_arr_red[:,0]
    av_grpcen_red_1 = mean_cen_arr_red[:,1]
    av_grpcen_red_2 = mean_cen_arr_red[:,2]
    av_grpcen_red_3 = mean_cen_arr_red[:,3]
    av_grpcen_red_4 = mean_cen_arr_red[:,4]

    av_grpcen_blue_0 = mean_cen_arr_blue[:,0]
    av_grpcen_blue_1 = mean_cen_arr_blue[:,1]
    av_grpcen_blue_2 = mean_cen_arr_blue[:,2]
    av_grpcen_blue_3 = mean_cen_arr_blue[:,3]
    av_grpcen_blue_4 = mean_cen_arr_blue[:,4]



    combined_df = pd.DataFrame({'phi_red_0':phi_red_0, 'phi_red_1':phi_red_1,\
        'phi_red_2':phi_red_2, 'phi_red_3':phi_red_3, 'phi_red_4':phi_red_4, \
        'phi_blue_0':phi_blue_0, 'phi_blue_1':phi_blue_1, 
        'phi_blue_2':phi_blue_2, 'phi_blue_3':phi_blue_3, 
        'phi_blue_4':phi_blue_4, \
        'dv_red_0':dv_red_0, 'dv_red_1':dv_red_1, 'dv_red_2':dv_red_2, \
        'dv_red_3':dv_red_3, 'dv_red_4':dv_red_4, \
        'dv_blue_0':dv_blue_0, 'dv_blue_1':dv_blue_1, 'dv_blue_2':dv_blue_2, \
        'dv_blue_3':dv_blue_3, 'dv_blue_4':dv_blue_4, \
        'av_grpcen_red_0':av_grpcen_red_0, 'av_grpcen_red_1':av_grpcen_red_1, \
        'av_grpcen_red_2':av_grpcen_red_2, 'av_grpcen_red_3':av_grpcen_red_3, \
        'av_grpcen_red_4':av_grpcen_red_4, 'av_grpcen_blue_0':av_grpcen_blue_0,\
        'av_grpcen_blue_1':av_grpcen_blue_1, 'av_grpcen_blue_2':av_grpcen_blue_2, \
        'av_grpcen_blue_3':av_grpcen_blue_3, 'av_grpcen_blue_4':av_grpcen_blue_4 })



    # deltav_sig_colour = np.append(deltav_sig_red, deltav_sig_blue, axis = 0)
    # cov_mat_colour = np.cov(phi_arr_colour,deltav_sig_colour, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    # cov_mat_colour = np.cov(phi_arr_red,phi_arr_blue, rowvar=False)
    # err_colour = np.sqrt(cov_mat_colour.diagonal())
    # corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    return combined_df

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

global model_init
global survey
global path_to_proc
global mf_type
global quenching

survey = 'eco'
machine = 'mac'
mf_type = 'smf'
quenching = 'hybrid'

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
    catl_file = path_to_proc + "gal_group_eco_data.hdf5"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_data + 'mocks/m200b/resolvea/'
elif survey == 'resolveb':
    path_to_mocks = path_to_data + 'mocks/m200b/resolveb/'

catl, volume, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)

std_red_mocks, std_blue_mocks, centers_red_mocks, \
    centers_blue_mocks = get_deltav_sigma_mocks_urcolour(survey, path_to_mocks)

std_red_mocks2, std_blue_mocks2, centers_red_mocks2, \
    centers_blue_mocks2 = get_deltav_sigma_mocks_qmcolour_mod(survey, path_to_mocks)

std_red_data, centers_red_data, std_blue_data, centers_blue_data = \
    get_deltav_sigma_data(catl)

phi_red_arr, phi_blue_arr = get_err_smf_mocks(survey, path_to_mocks)

corr_mat_combined_bool = True
if corr_mat_combined_bool:
    phi_red_0 = phi_red_arr[:,0]
    phi_red_1 = phi_red_arr[:,1]
    phi_red_2 = phi_red_arr[:,2]
    phi_red_3 = phi_red_arr[:,3]
    phi_red_4 = phi_red_arr[:,4]

    phi_blue_0 = phi_blue_arr[:,0]
    phi_blue_1 = phi_blue_arr[:,1]
    phi_blue_2 = phi_blue_arr[:,2]
    phi_blue_3 = phi_blue_arr[:,3]
    phi_blue_4 = phi_blue_arr[:,4]

    dv_red_0 = std_red_mocks2[:,0]
    dv_red_1 = std_red_mocks2[:,1]
    dv_red_2 = std_red_mocks2[:,2]
    dv_red_3 = std_red_mocks2[:,3]
    dv_red_4 = std_red_mocks2[:,4]

    dv_blue_0 = std_blue_mocks2[:,0]
    dv_blue_1 = std_blue_mocks2[:,1]
    dv_blue_2 = std_blue_mocks2[:,2]
    dv_blue_3 = std_blue_mocks2[:,3]
    dv_blue_4 = std_blue_mocks2[:,4]
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
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(combined_df.corr(), cmap=cmap)
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig1.colorbar(cax)
    plt.title(r'Combined \boldmath{$\Phi$} and \boldmath{$\sigma$} measurements using ZuMand15 colours')
    plt.show()

if corr_mat_combined_bool:
    phi_red_0 = phi_red_arr[:,0]
    phi_red_1 = phi_red_arr[:,1]
    phi_red_2 = phi_red_arr[:,2]
    phi_red_3 = phi_red_arr[:,3]
    phi_red_4 = phi_red_arr[:,4]

    phi_blue_0 = phi_blue_arr[:,0]
    phi_blue_1 = phi_blue_arr[:,1]
    phi_blue_2 = phi_blue_arr[:,2]
    phi_blue_3 = phi_blue_arr[:,3]
    phi_blue_4 = phi_blue_arr[:,4]

    dv_red_0 = std_red_mocks[:,0]
    dv_red_1 = std_red_mocks[:,1]
    dv_red_2 = std_red_mocks[:,2]
    dv_red_3 = std_red_mocks[:,3]
    dv_red_4 = std_red_mocks[:,4]

    dv_blue_0 = std_blue_mocks[:,0]
    dv_blue_1 = std_blue_mocks[:,1]
    dv_blue_2 = std_blue_mocks[:,2]
    dv_blue_3 = std_blue_mocks[:,3]
    dv_blue_4 = std_blue_mocks[:,4]
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
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(combined_df.corr(), cmap=cmap)
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig1.colorbar(cax)
    plt.title(r'Combined \boldmath{$\Phi$} and \boldmath{$\sigma$} measurements')
    plt.show()

else:
    cov_mat_colour = np.cov(std_red_mocks, std_blue_mocks, 
        rowvar=False)
    err_colour = np.sqrt(cov_mat_colour.diagonal())
    corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)

    # Correlation matrix of just deltav measurements combined
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(corr_mat_colour, cmap=cmap)
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig1.colorbar(cax)
    plt.title(r'Combined \boldmath{$\delta$}v measurements')
    plt.show()

# Plot of spread in velocity dispersion of red and blue galaxies from data and 
# mocks
fig2 = plt.figure()
for idx in range(len(centers_red_mocks)):
    plt.scatter(centers_red_mocks[idx], std_red_mocks[idx], 
        c='indianred', s=80)
    plt.scatter(centers_blue_mocks[idx], std_blue_mocks[idx], 
        c='cornflowerblue', s=80)
plt.scatter(centers_red_data, std_red_data, marker='*', c='darkred', s=80)
plt.scatter(centers_blue_data, std_blue_data, marker='*', c='darkblue', s=80)
plt.xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$', labelpad=15, 
    fontsize=25)
plt.ylabel(r'\boldmath$\sigma \left[km/s\right]$', labelpad=15, 
    fontsize=25)
plt.title(r'mocks vs. data $\sigma$')
plt.show()

## Histogram of red and blue sigma in bins of central stellar mass to see if the
## distribution of values to take std of is normal or lognormal
nrows = 2
ncols = 5
if survey == 'eco' or survey == 'resolvea':
    red_stellar_mass_bins = np.linspace(8.6,11.5,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
fig3, axs = plt.subplots(nrows, ncols)
for i in range(0, nrows, 1):
    for j in range(0, ncols, 1):
        if i == 0: # row 1 for all red bins
            axs[i, j].hist(np.log10(std_red_mocks.T[j]), histtype='step', \
                color='indianred', linewidth=4, linestyle='-') # first red bin
            axs[i, j].set_title('[{0}-{1}]'.format(np.round(
                red_stellar_mass_bins[j],2), np.round(
                red_stellar_mass_bins[j+1],2)), fontsize=20)
            k2, p = nt(np.log10(std_red_mocks.T[j]), nan_policy="omit")
            axs[i, j].text(0.7, 0.7, "{0}".format(np.round(p, 2)),
                transform=axs[i, j].transAxes)
        else: # row 2 for all blue bins
            axs[i, j].hist(np.log10(std_blue_mocks.T[j]), histtype='step', \
                color='cornflowerblue', linewidth=4, linestyle='-')
            axs[i, j].set_title('[{0}-{1}]'.format(np.round(
                blue_stellar_mass_bins[j],2), np.round(
                blue_stellar_mass_bins[j+1],2)), fontsize=20)
            k2, p = nt(np.log10(std_blue_mocks.T[j]), nan_policy="omit")
            axs[i, j].text(0.7, 0.7, "{0}".format(np.round(p, 2)), 
                transform=axs[i, j].transAxes)
for ax in axs.flat:
    ax.set(xlabel=r'\boldmath$\sigma \left[km/s\right]$')

for ax in axs.flat:
    ax.label_outer()

plt.show()

p_red_arr = []
p_blue_arr = []
for idx in range(len(std_red_mocks.T)):
    k2, p = nt(np.log10(std_red_mocks.T[idx]), nan_policy="omit")
    p_red_arr.append(p)
for idx in range(len(std_blue_mocks.T)):
    k2, p = nt(np.log10(std_blue_mocks.T[idx]), nan_policy="omit")
    p_blue_arr.append(p)

# * resolve B - neither log or linear passed null hypothesis of normal dist.
# * eco - log passed null hypothesis 

################################################################################

    ### Randomly sample a subset from 64 mocks and measure corr matrix ###

for idx in range(5):
    print(idx)
    combined_df = get_err_data_mod(survey, path_to_mocks)

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(combined_df.corr(), cmap=cmap)
    # tick_marks = [i for i in range(len(corr_mat_colour.columns))]
    # plt.xticks(tick_marks, corr_mat_colour.columns, rotation='vertical')
    # plt.yticks(tick_marks, corr_mat_colour.columns)    
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig1.colorbar(cax)
    plt.title(r'Mass function and old and new sigma observable')
    plt.savefig('sample_{0}.png'.format(idx+1))


                ### Testing singular value decomposition ###

from scipy import linalg

num_mocks = 64
used_combined_df = get_err_data(survey, path_to_mocks)
used_corr_mat_colour = used_combined_df.corr()
used_err_colour = np.sqrt(np.diag(used_combined_df.cov()))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(used_corr_mat_colour, cmap=cmap)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig1.colorbar(cax)
plt.title(r'Original matrix')
plt.show()


## Help from http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf
## Help from https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

U, s, Vh = linalg.svd(used_corr_mat_colour) # columns of U are the eigenvectors
eigenvalue_threshold = np.sqrt(np.sqrt(2/num_mocks))

idxs_cut = []
for idx,eigenval in enumerate(s):
    if eigenval < eigenvalue_threshold:
        idxs_cut.append(idx)

last_idx_to_keep = min(idxs_cut)-1
reconst = np.matrix(U[:, :last_idx_to_keep]) * np.diag(s[:last_idx_to_keep]) * \
    np.matrix(Vh[:last_idx_to_keep, :])

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(reconst, cmap=cmap)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig2.colorbar(cax)
plt.title(r'Reconstructed matrix post-SVD')
plt.show()

fig3 = plt.figure()
plt.scatter(np.linspace(1,len(s),len(s)), s, s=120, marker='o', 
    facecolors='none', edgecolors='mediumorchid', linewidths=3)
plt.plot(np.linspace(1,len(s),len(s)), s, '-k')
plt.hlines(eigenvalue_threshold, 0, 30, ls='--')
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
plt.show()

## Projecting data onto new orthogonal space

print('Measuring SMF for data')
total_data, red_data, blue_data = measure_all_smf(catl, volume, \
    data_bool=True)

print('Measuring spread in vel disp for data')
std_red, old_centers_red, std_blue, old_centers_blue = get_deltav_sigma_data(catl)

print('Measuring binned spread in vel disp for data')
mean_grp_cen_red, new_centers_red, mean_grp_cen_blue, new_centers_blue = \
    get_sigma_per_group_data(catl)

full_data_arr = []
full_data_arr = np.insert(full_data_arr, 0, red_data[1])
full_data_arr = np.insert(full_data_arr, len(full_data_arr), blue_data[1])
full_data_arr = np.insert(full_data_arr, len(full_data_arr), std_red)
full_data_arr = np.insert(full_data_arr, len(full_data_arr), std_blue)
full_data_arr = np.insert(full_data_arr, len(full_data_arr), mean_grp_cen_red[0])
full_data_arr = np.insert(full_data_arr, len(full_data_arr), mean_grp_cen_blue[0])

full_data_arr = full_data_arr.reshape(1,30) #N,n - N:# of data , n:# of dims
eigenvector_subset = np.matrix(U[:, :last_idx_to_keep]) 

full_data_arr_new_space = full_data_arr @ eigenvector_subset

## Projecting simga (error from mocks) onto new orthogonal space

mock_data_df_new_space = pd.DataFrame(used_combined_df @ eigenvector_subset)
err_colour_new_space = np.sqrt(np.diag(mock_data_df_new_space.cov()))

