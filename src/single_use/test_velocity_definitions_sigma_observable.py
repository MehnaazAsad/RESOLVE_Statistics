"""
{This script tests different measurements of velocity for second observable.}
"""

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import scipy as sp
import random
import math
import os

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

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
        # eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0)#, \
            #usecols=columns)

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
        # volume = 192351.36 # Survey volume with buffer [Mpc/h]^3
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
        Standard deviation from 0 of velocity difference values in each mass bin
    """

    last_index = len(bins)-1
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        cen_deltav_arr = []
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and index1 == last_index:
                cen_deltav_arr.append(vel_arr[index2])
            elif stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])

        mean = 0
        # mean = np.mean(cen_deltav_arr)
        diff_sqrd_arr = []
        for value in cen_deltav_arr:
            diff = value - mean
            diff_sqrd = diff**2
            diff_sqrd_arr.append(diff_sqrd)
        mean_diff_sqrd = np.mean(diff_sqrd_arr)
        std = np.sqrt(mean_diff_sqrd)
        std_arr.append(std)

    return std_arr

def mean_std_func(bins, mass_arr, vel_arr, groupid_arr):
    mass_arr_bin_idxs = np.digitize(mass_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_std_arr = []
    len_std_arr = []
    for idx in range(1, len(bins)):
        cen_deltav_arr = []
        grpid_arr = []
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        cen_deltav_arr.append(np.array(vel_arr)[current_bin_idxs])
        grpid_arr.append(np.array(groupid_arr)[current_bin_idxs])

        mean = 0
        std_arr = []
        
        data_temp = {'group_id': np.array(grpid_arr).flatten(), 
            'deltav': np.array(cen_deltav_arr).flatten()}
        df_temp = pd.DataFrame(data=data_temp)
        groups = df_temp.groupby('group_id')
        keys = groups.groups.keys()

        for key in keys:
            group = groups.get_group(key)
            vels = group.deltav.values
            diff_sqrd_arr = []
            for value in vels:
                diff = value - mean
                diff_sqrd = diff**2
                diff_sqrd_arr.append(diff_sqrd)
            mean_diff_sqrd = np.mean(diff_sqrd_arr)
            std = np.sqrt(mean_diff_sqrd)
            std_arr.append(std)
        len_std_arr.append(len(std_arr))
        mean_std_arr.append(np.mean(std_arr))
    return mean_std_arr

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

def mean_grphalo_func(bins, logmstar_arr, loghalom_arr):
    mass_arr_bin_idxs = np.digitize(logmstar_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_halomass_arr = []
    for idx in range(1, len(bins)):
        halomass_arr = []
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        halomass_arr.append(np.array(loghalom_arr)[current_bin_idxs])
        mean_halomass = np.mean(np.array(halomass_arr).flatten())
        mean_halomass_arr.append(mean_halomass)

    return mean_halomass_arr

def mean_grphalo_vcirc_func(bins, logmstar_arr, loghalom_arr):
    mass_arr_bin_idxs = np.digitize(logmstar_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_vcirc_arr = []
    for idx in range(1, len(bins)):
        halomass_arr = []
        delta_mean = 200
        omega_m = 0.3
        rho_crit = 2.77*10**11 # assuming h=1.0 # h^2 . Msun/Mpc^3
        G = 4.3*10**-9 # Mpc . Msun^-1 . (km/s)^2
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        halomass_arr.append(np.array(loghalom_arr)[current_bin_idxs])
        halomass_arr = np.array(halomass_arr).flatten()
        # radius in Mpc
        halo_radius = ((3*(10**halomass_arr)) / (4*np.pi*delta_mean*omega_m*rho_crit))**(1/3)
        halo_vcirc = np.sqrt((G * (10**halomass_arr))/halo_radius)
        mean_vcirc = np.mean(halo_vcirc)

        mean_vcirc_arr.append(mean_vcirc)

    return mean_vcirc_arr

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

        # Different velocity definitions
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]

        # Velocity difference
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

        # Different velocity definitions
        mean_cz_grp = np.round(np.mean(group.cz.values),2)
        cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]

        # Velocity difference
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
    # last_red_bin = centers_red[-1] + (centers_red[-1] - centers_red[-2])
    # centers_red = np.insert(centers_red, len(centers_red), last_red_bin)

    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
    # last_blue_bin = centers_blue[-1] + (centers_blue[-1] - centers_blue[-2])
    # centers_blue = np.insert(centers_blue, len(centers_blue), last_blue_bin)
               
    centers_red = np.array(centers_red)
    centers_blue = np.array(centers_blue)
            
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

if survey == 'eco':
    # catl_file = path_to_raw + "eco/eco_all.csv"
    ## New catalog with group finder run on subset after applying M* and cz cuts
    # catl_file = path_to_proc + "gal_group_eco_data.hdf5"
    catl_file = path_to_proc + "gal_group_eco_data_vol_update.hdf5"

    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

catl, volume, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)
# std_red, centers_red, std_blue, centers_blue = get_deltav_sigma_data(catl)
# err_total_data, err_colour_data = \
#     get_err_data(survey, path_to_mocks)

catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
catl.M_group = np.log10((10**catl.M_group) / 2.041)
catl.logmh_s = np.log10((10**catl.logmh_s) / 2.041)
catl.logmh = np.log10((10**catl.logmh) / 2.041)



if survey == 'eco' or survey == 'resolvea':
    catl = catl.loc[catl.logmstar >= np.log10((10**8.9)/2.041)]
elif survey == 'resolveb':
    catl = catl.loc[catl.logmstar >= np.log10((10**8.7)/2.041)]

red_subset_grpids = np.unique(catl.groupid.loc[(catl.\
    colour_label == 'R') & (catl.g_galtype == 1)].values)  
blue_subset_grpids = np.unique(catl.groupid.loc[(catl.\
    colour_label == 'B') & (catl.g_galtype == 1)].values)

red_subset_grpids = np.unique(catl.grp.loc[(catl.\
    colour_label == 'R') & (catl.fc == 1)].values)  
blue_subset_grpids = np.unique(catl.grp.loc[(catl.\
    colour_label == 'B') & (catl.fc == 1)].values)

# Calculating spread in velocity dispersion for galaxies in groups with a 
# red central

### USE IF OLD DATA
red_deltav_arr = []
red_cen_stellar_mass_arr = []
grpid_arr = []
red_cen_cz_arr = []
red_mean_cz_arr = []
red_grp_halo_mass_arr = []
for key in red_subset_grpids: 
    group = catl.loc[catl.grp == key]

    grp_halo_mass = np.unique(group.logmh.values)[0]
    cen_stellar_mass = group.logmstar.loc[group.fc.\
        values == 1].values[0]
    
    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.fc == 1].values[0]
    cz_grp = np.unique(group.grpcz.values)[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]
    
    # red_cen_stellar_mass_arr.append(cen_stellar_mass)
    red_grp_halo_mass_arr.append(grp_halo_mass)
    red_cen_cz_arr.append(cen_cz_grp)
    red_mean_cz_arr.append(mean_cz_grp)

    for val in deltav:
        red_deltav_arr.append(val)
        red_cen_stellar_mass_arr.append(cen_stellar_mass)
        grpid_arr.append(key)
    # if len(group) > 5:
    #     break

### USE IF NEW DATA
red_deltav_arr = []
red_cen_stellar_mass_arr = []
grpid_arr = []
red_cen_cz_arr = []
red_mean_cz_arr = []
red_grp_halo_mass_arr = []
for key in red_subset_grpids: 
    group = catl.loc[catl.groupid == key]

    grp_halo_mass = np.unique(group.logmh.values)[0]
    cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
        values == 1].values[0]
    
    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
    cz_grp = np.unique(group.grpcz.values)[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]
    
    # red_cen_stellar_mass_arr.append(cen_stellar_mass)
    red_grp_halo_mass_arr.append(grp_halo_mass)
    red_cen_cz_arr.append(cen_cz_grp)
    red_mean_cz_arr.append(mean_cz_grp)

    for val in deltav:
        red_deltav_arr.append(val)
        red_cen_stellar_mass_arr.append(cen_stellar_mass)
        grpid_arr.append(key)
    # if len(group) > 5:
    #     break

if survey == 'eco' or survey == 'resolvea':
    # TODO : check if this is actually correct for resolve a
    red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    # red_stellar_mass_bins = np.linspace(8.9,11.5,6) # h=0.7
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)

std_red = std_func_mod(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    red_deltav_arr)
std_red = np.array(std_red)

mean_std_red = mean_std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    red_deltav_arr, grpid_arr)
mean_halo_red = mean_grphalo_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    red_grp_halo_mass_arr)
mean_vcirc_red = mean_grphalo_vcirc_func(red_stellar_mass_bins, 
    red_cen_stellar_mass_arr, red_grp_halo_mass_arr)

# Calculating spread in velocity dispersion for galaxies in groups with a 
# blue central

### USE IF OLD DATA
blue_deltav_arr = []
blue_cen_stellar_mass_arr = []
grpid_arr = []
blue_cen_cz_arr = []
blue_mean_cz_arr = []
blue_grp_halo_mass_arr = []
for key in blue_subset_grpids: 
    group = catl.loc[catl.grp == key]

    grp_halo_mass = np.unique(group.logmh.values)[0]
    cen_stellar_mass = group.logmstar.loc[group.fc\
        .values == 1].values[0]

    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.fc == 1].values[0]
    cz_grp = np.unique(group.grpcz.values)[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]

    # blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    blue_grp_halo_mass_arr.append(grp_halo_mass)
    blue_cen_cz_arr.append(cen_cz_grp)
    blue_mean_cz_arr.append(mean_cz_grp)

    for val in deltav:
        blue_deltav_arr.append(val)
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        grpid_arr.append(key)


### USE IF NEW DATA
blue_deltav_arr = []
blue_cen_stellar_mass_arr = []
grpid_arr = []
blue_cen_cz_arr = []
blue_mean_cz_arr = []
blue_grp_halo_mass_arr = []
for key in blue_subset_grpids: 
    group = catl.loc[catl.groupid == key]

    grp_halo_mass = np.unique(group.logmh.values)[0]
    cen_stellar_mass = group.logmstar.loc[group.g_galtype\
        .values == 1].values[0]

    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
    cz_grp = np.unique(group.grpcz.values)[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]

    # blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    blue_grp_halo_mass_arr.append(grp_halo_mass)
    blue_cen_cz_arr.append(cen_cz_grp)
    blue_mean_cz_arr.append(mean_cz_grp)

    for val in deltav:
        blue_deltav_arr.append(val)
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        grpid_arr.append(key)

if survey == 'eco' or survey == 'resolvea':
    # TODO : check if this is actually correct for resolve a
    blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    # blue_stellar_mass_bins = np.linspace(8.9,11,6) #h=0.7

elif survey == 'resolveb':
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)

std_blue = std_func_mod(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    blue_deltav_arr)    
std_blue = np.array(std_blue)

mean_std_blue = mean_std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    blue_deltav_arr, grpid_arr)
mean_halo_blue = mean_grphalo_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    blue_grp_halo_mass_arr)
mean_vcirc_blue = mean_grphalo_vcirc_func(blue_stellar_mass_bins, 
    blue_cen_stellar_mass_arr, blue_grp_halo_mass_arr)

# centers_red = 0.5 * (result_red[1][1:] + \
#     result_red[1][:-1])
# centers_blue = 0.5 * (result_blue[1][1:] + \
#     result_blue[1][:-1])

centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    red_stellar_mass_bins[:-1])
# last_red_bin = centers_red[-1] + (centers_red[-1] - centers_red[-2])
# centers_red = np.insert(centers_red, len(centers_red), last_red_bin)
centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    blue_stellar_mass_bins[:-1])
# last_blue_bin = centers_blue[-1] + (centers_blue[-1] - centers_blue[-2])
# centers_blue = np.insert(centers_blue, len(centers_blue), last_blue_bin)

fig1 = plt.figure()
# plt.errorbar(centers_red,std_red,yerr=err_colour_data[10:15],
#     color='darkred',fmt='p-',ecolor='darkred',markersize=10,capsize=10,
#     capthick=1.0,zorder=10)
# plt.errorbar(centers_blue,std_blue,yerr=err_colour_data[15:20],
#     color='darkblue',fmt='p-',ecolor='darkblue',markersize=10,capsize=10,
#     capthick=1.0,zorder=10)
plt.scatter(centers_red,std_red,color='darkred',s=350,marker='p')
plt.scatter(centers_blue,std_blue,color='darkblue',s=350,marker='p')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
# plt.title('Spread in velocity difference from group cz as included in catalog')
plt.title('Spread in velocity difference from central cz of group')
# plt.title('Spread in velocity difference from mean cz of group')
plt.show()

fig2 = plt.figure()
plt.scatter(centers_red,mean_std_red,color='darkred',s=350, marker='p')
plt.scatter(centers_blue,mean_std_blue,color='darkblue',s=350, marker='p')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.title('Mean of spread in velocity difference from central cz of group')
plt.show()


# Plot of comparison between using mean of all satellite velocities to measure 
# velocity difference and using central velocity
fig3, ax3 = plt.subplots()

ax4 = ax3.twinx()
ax4.set_ylabel(r'Mean cz (red) [km/s]')
imshowax4 = ax4.scatter(red_cen_cz_arr, red_mean_cz_arr, c=red_grp_halo_mass_arr, cmap='Reds', s=50)
# ax4.set_ylim(ax4.get_ylim()[::-1])
plt.gca().invert_yaxis()

imshowax3 = ax3.scatter(blue_cen_cz_arr, blue_mean_cz_arr, c=blue_grp_halo_mass_arr, cmap='Blues', s=50)

ax3.plot(np.linspace(2530, 7470, 20), np.linspace(2530, 7470, 20), '--k')
ax4.plot(np.linspace(2530, 7470, 20), np.linspace(2530, 7470, 20), '--k')

cbb = plt.colorbar(mappable=imshowax3, shrink=0.5, pad=-0.1)
cbr = plt.colorbar(mappable=imshowax4, shrink=0.5, pad=0.15)
cbb.set_label(r'Group halo mass $[{M_\odot}/h]$', rotation=90, labelpad=20)

ax3.set_xlabel(r'Central cz [km/s]')
ax3.set_ylabel(r'Mean cz (blue) [km/s]')
plt.title('Comparison of central cz and mean cz values of groups')
plt.show()


fig4 = plt.figure()
plt.scatter(centers_red, mean_halo_red, color='darkred',s=350,marker='p')
plt.scatter(centers_blue,mean_halo_blue,color='darkblue',s=350,marker='p')
plt.xlabel('Group central M* [Msun/h]')
plt.ylabel('Mean group halo mass [Msun/h]')
plt.title(r'Group halo mass (HAM with $M_{r}$) vs group central stellar mass')
plt.show()

fig5 = plt.figure()
plt.scatter(centers_red, mean_vcirc_red, color='darkred',s=350, marker='p')
plt.scatter(centers_blue,mean_vcirc_blue,color='darkblue',s=350, marker='p')
plt.xlabel('Group central M* [Msun/h]')
plt.ylabel('Mean group halo circular velocity [km/s]')
plt.title(r'Group halo cirvular velocity vs group central stellar mass')
plt.show()

# Plot of comparison between using spread in velocity across all groups per bin
# and mean of spreads per group per bin
fig6 = plt.figure()
plt.scatter(std_red, mean_std_red, c='indianred', s=50)
plt.scatter(std_blue, mean_std_blue, c='cornflowerblue', s=50)
plt.plot(np.linspace(0, 350, 20), np.linspace(0, 350, 20), '--k')
plt.xlabel('Sigma')
plt.ylabel('Mean sigma')
plt.show()

################################################################################
############################### SIMULATION DATA ################################
################################################################################

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

def mean_std_func(bins, mass_arr, vel_arr, groupid_arr):
    mass_arr_bin_idxs = np.digitize(mass_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_std_arr = []
    len_std_arr = []
    for idx in range(1, len(bins)):
        cen_deltav_arr = []
        grpid_arr = []
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        cen_deltav_arr.append(np.array(vel_arr)[current_bin_idxs])
        grpid_arr.append(np.array(groupid_arr)[current_bin_idxs])

        mean = 0
        std_arr = []
        
        data_temp = {'group_id': np.array(grpid_arr).flatten(), 
            'deltav': np.array(cen_deltav_arr).flatten()}
        df_temp = pd.DataFrame(data=data_temp)
        groups = df_temp.groupby('group_id')
        keys = groups.groups.keys()

        for key in keys:
            group = groups.get_group(key)
            vels = group.deltav.values
            diff_sqrd_arr = []
            for value in vels:
                diff = value - mean
                diff_sqrd = diff**2
                diff_sqrd_arr.append(diff_sqrd)
            mean_diff_sqrd = np.mean(diff_sqrd_arr)
            std = np.sqrt(mean_diff_sqrd)
            std_arr.append(std)
        len_std_arr.append(len(std_arr))
        mean_std_arr.append(np.mean(std_arr))
    return mean_std_arr

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

def mean_halo_func(bins, logmstar_arr, loghalom_arr):
    mass_arr_bin_idxs = np.digitize(logmstar_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_halomass_arr = []
    for idx in range(1, len(bins)):
        halomass_arr = []
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        halomass_arr.append(np.array(loghalom_arr)[current_bin_idxs])
        mean_halomass = np.mean(np.array(halomass_arr).flatten())
        mean_halomass_arr.append(mean_halomass)

    return mean_halomass_arr

def mean_halo_vcirc_func(bins, logmstar_arr, loghalom_arr, halor_arr):
    mass_arr_bin_idxs = np.digitize(logmstar_arr, bins)
    # Put all galaxies that would have been in the bin after the last in the 
    # bin as well i.e galaxies with bin number 5 and 6 from previous line all
    # go in one bin
    for idx, value in enumerate(mass_arr_bin_idxs):
        if value == 6:
            mass_arr_bin_idxs[idx] = 5

    mean_vcirc_arr = []
    for idx in range(1, len(bins)):
        halomass_arr = []
        halorvir_arr = []
        delta_mean = 200
        omega_m = 0.3
        rho_crit = 2.77*10**11 # assuming h=1.0 # h^2 . Msun/Mpc^3
        G = 4.3*10**-9 # Mpc . Msun^-1 . (km/s)^2
        current_bin_idxs = np.argwhere(mass_arr_bin_idxs == idx)
        halomass_arr.append(np.array(loghalom_arr)[current_bin_idxs])
        halomass_arr = np.array(halomass_arr).flatten()
        halorvir_arr.append(np.array(halor_arr)[current_bin_idxs])
        halorvir_arr = np.array(halorvir_arr).flatten()
        # radius in Mpc
        # halo_radius = ((3*(10**halomass_arr)) / (4*np.pi*delta_mean*omega_m*rho_crit))**(1/3)
        halo_vcirc = np.sqrt((G * (10**halomass_arr))/halorvir_arr)
        mean_vcirc = np.mean(halo_vcirc)

        mean_vcirc_arr.append(mean_vcirc)

    return mean_vcirc_arr


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

chi2_file = path_to_proc + 'smhm_colour_run17/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run17/mcmc_{0}_colour_raw.txt'.\
    format(survey)
randint_file = path_to_proc + 'smhm_colour_run17/{0}_colour_mocknum.txt'.\
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

cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', 'halo_macc', 'halo_rvir', 
    'cz', \
    '{0}'.format(bf_randint), \
    'g_galtype_{0}'.format(bf_randint), \
    'groupid_{0}'.format(bf_randint)]

gals_df = gal_group_df_subset[cols_to_use]

gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
    format(bf_randint),'groupid_{0}'.format(bf_randint)]).\
    reset_index(drop=True)

gals_df = assign_cen_sat_flag(gals_df)
f_red_cen, f_red_sat = hybrid_quenching_model(best_fit_params, gals_df, 
    'vishnu', best_fit_mocknum)
gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

grpid_col = 'groupid_{0}'.format(bf_randint)
galtype_col = 'g_galtype_{0}'.format(bf_randint)
logmstar_col = '{0}'.format(bf_randint)

red_subset_grpids = np.unique(gals_df.halo_id.loc[(gals_df.\
    colour_label == 'R') & (gals_df.cs_flag == 1)].values)  
blue_subset_grpids = np.unique(gals_df.halo_id.loc[(gals_df.\
    colour_label == 'B') & (gals_df.cs_flag == 1)].values)

red_deltav_arr = []
red_cen_stellar_mass_arr = []
red_grpid_arr = []
red_cen_cz_arr = []
red_mean_cz_arr = []
red_halo_mass_arr = []
red_halo_rvir_arr = []
red_host_halo_mass_arr = []
red_host_halo_rvir_arr = []
for key in red_subset_grpids: 
    group = gals_df.loc[gals_df.halo_hostid == key]

    host_halo_mass = group.halo_mvir.loc[group.cs_flag.\
        values == 1].values[0]
    host_halo_rvir = group.halo_rvir.loc[group.cs_flag.\
        values == 1].values[0]
    # halo_macc = group.halo_mvir.values
    # halo_rvir = group.halo_rvir.values
    cen_stellar_mass = group[logmstar_col].loc[group.cs_flag.\
        values == 1].values[0]

   
    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.cs_flag == 1].values[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]
    
    red_cen_stellar_mass_arr.append(cen_stellar_mass)
    red_host_halo_mass_arr.append(host_halo_mass)
    red_host_halo_rvir_arr.append(host_halo_rvir)
    red_cen_cz_arr.append(cen_cz_grp)
    red_mean_cz_arr.append(mean_cz_grp)

    for idx, val in enumerate(deltav):
        red_deltav_arr.append(val)
        # red_halo_mass_arr.append(halo_macc[idx])
        # red_halo_rvir_arr.append(halo_rvir[idx])
        # red_cen_stellar_mass_arr.append(cen_stellar_mass)
        red_grpid_arr.append(key)
    # if len(group) > 5:
    #     break

if survey == 'eco' or survey == 'resolvea':
    # TODO : check if this is actually correct for resolve a
    red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    # red_stellar_mass_bins = np.linspace(8.6,11.5,6)
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)

blue_deltav_arr = []
blue_cen_stellar_mass_arr = []
blue_grpid_arr = []
blue_cen_cz_arr = []
blue_mean_cz_arr = []
blue_halo_mass_arr = []
blue_halo_rvir_arr = []
blue_host_halo_mass_arr = []
blue_host_halo_rvir_arr = []
for key in blue_subset_grpids: 
    group = gals_df.loc[gals_df.halo_hostid == key]

    host_halo_mass = group.halo_mvir.loc[group.cs_flag.\
        values == 1].values[0]
    host_halo_rvir = group.halo_rvir.loc[group.cs_flag.\
        values == 1].values[0]
    # halo_macc = group.halo_mvir.values
    # halo_rvir = group.halo_rvir.values
    cen_stellar_mass = group[logmstar_col].loc[group.cs_flag.\
        values == 1].values[0]

    
    # Different velocity definitions
    mean_cz_grp = np.round(np.mean(group.cz.values),2)
    cen_cz_grp = group.cz.loc[group.cs_flag == 1].values[0]

    # Velocity difference
    deltav = group.cz.values - len(group)*[cen_cz_grp]

    blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    blue_host_halo_mass_arr.append(host_halo_mass)
    blue_host_halo_rvir_arr.append(host_halo_rvir)
    blue_cen_cz_arr.append(cen_cz_grp)
    blue_mean_cz_arr.append(mean_cz_grp)

    for idx, val in enumerate(deltav):
        blue_deltav_arr.append(val)
        # blue_halo_mass_arr.append(halo_macc[idx])
        # blue_halo_rvir_arr.append(halo_rvir[idx])
        # blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        blue_grpid_arr.append(key)

if survey == 'eco' or survey == 'resolvea':
    # TODO : check if this is actually correct for resolve a
    blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    # blue_stellar_mass_bins = np.linspace(8.9,11,6)

elif survey == 'resolveb':
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)


std_red = std_func_mod(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    red_deltav_arr)
std_red = np.array(std_red)

mean_std_red = mean_std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    red_deltav_arr, red_grpid_arr)

std_blue = std_func_mod(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    blue_deltav_arr)
std_blue = np.array(std_blue)

mean_std_blue = mean_std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    blue_deltav_arr, blue_grpid_arr)


mean_halo_red = mean_halo_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    np.log10(red_host_halo_mass_arr))
mean_vcirc_red = mean_halo_vcirc_func(red_stellar_mass_bins, 
    red_cen_stellar_mass_arr, np.log10(red_host_halo_mass_arr), 
    red_host_halo_rvir_arr)

mean_halo_blue = mean_halo_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    np.log10(blue_host_halo_mass_arr))
mean_vcirc_blue = mean_halo_vcirc_func(blue_stellar_mass_bins, 
    blue_cen_stellar_mass_arr, np.log10(blue_host_halo_mass_arr), 
    blue_host_halo_rvir_arr)

centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    red_stellar_mass_bins[:-1])
centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    blue_stellar_mass_bins[:-1])


fig7 = plt.figure()
plt.scatter(centers_red,std_red,color='darkred',s=350,marker='p')
plt.scatter(centers_blue,std_blue,color='darkblue',s=350,marker='p')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.title('Spread in velocity difference from halo central cz')
plt.show()

fig8 = plt.figure()
plt.scatter(centers_red,mean_std_red,color='darkred',s=350, marker='p')
plt.scatter(centers_blue,mean_std_blue,color='darkblue',s=350, marker='p')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.title('Mean of spread in velocity difference from halo central cz')
plt.show()

fig9 = plt.figure()
plt.scatter(centers_red, mean_halo_red, color='darkred',s=350,marker='p')
plt.scatter(centers_blue,mean_halo_blue,color='darkblue',s=350,marker='p')
plt.xlabel('Halo central M* [Msun/h]')
plt.ylabel('Mean halo mass [Msun/h]')
plt.show()

fig10 = plt.figure()
plt.scatter(centers_red, mean_vcirc_red, color='darkred',s=350, marker='p')
plt.scatter(centers_blue,mean_vcirc_blue,color='darkblue',s=350, marker='p')
plt.xlabel('Halo central M* [Msun/h]')
plt.ylabel('Mean halo circular velocity [km/s]')
plt.show()

################################################################################
# Comparing distribution of Behroozi parameters before and after adding second
# observable
################################################################################

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']

survey = 'eco'
machine = 'mac'
mf_type = 'smf'
ver = 2.0


## Subset of 100 from latest total smf run on which group finding was done
chi2_file = path_to_proc + 'smhm_run6/{0}_chi2.txt'.format(survey)
if mf_type == 'smf' and survey == 'eco' and ver == 1.0:
    chain_file = path_to_proc + 'mcmc_{0}.dat'.format(survey)
else:
    chain_file = path_to_proc + 'smhm_run6/mcmc_{0}_raw.txt'.\
        format(survey)

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)

print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)

print('Getting subset of 100 Behroozi parameters')
mcmc_table_subset = get_paramvals_percentile(mcmc_table, 68, chi2)

## Latest colour run using both observables

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

chi2_file = path_to_proc + 'smhm_colour_run17/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run17/mcmc_{0}_colour_raw.txt'.\
    format(survey)
randint_file = path_to_proc + 'smhm_colour_run17/{0}_colour_mocknum.txt'.\
    format(survey)

chi2 = read_chi2(chi2_file)
mcmc_table = read_mcmc(chain_file)
mock_nums_df = pd.read_csv(randint_file, header=None, names=['mock_num'], 
    dtype=int)

mcmc_table_pctl, bf_params, bf_chi2, bf_randint = \
    get_paramvals_percentile(mcmc_table, 68, chi2, mock_nums_df)

mock_nums_picked = mcmc_table_pctl['mock_num']

mhalo_arr = []
mstar_arr = []
lowslope = []
highslope = []
scatter = []
for idx in mock_nums_picked:
    mhalo_arr.append(mcmc_table_subset.T[0][idx-1])
    mstar_arr.append(mcmc_table_subset.T[1][idx-1])
    lowslope.append(mcmc_table_subset.T[2][idx-1])
    highslope.append(mcmc_table_subset.T[3][idx-1])
    scatter.append(mcmc_table_subset.T[4][idx-1])

plt.clf()
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
# ax1.hist(mcmc_table_b10_full['mhalo_c'], histtype='step', lw=3, color='r', ls='-', label='full chain')
ax1.hist(mcmc_table_subset.T[0], histtype='step', lw=3, color='r', ls='-', label='68% full B10', bins=np.linspace(11.5, 12.8, 8))

ax1.hist(mhalo_arr, histtype='step', lw=3, color='r', ls='dashdot', label='68% (+ 2nd observable)', bins=np.linspace(11.5, 12.8, 8))

# ax2.hist(mcmc_table_b10_full['mstellar_c'], histtype='step', lw=3, color='g', ls='-', label='full chain')
ax2.hist(mcmc_table_subset.T[1], histtype='step', lw=3, color='b', ls='-', label='68% full B10', bins=np.linspace(10.4, 10.9, 8))
ax2.hist(mstar_arr, histtype='step', lw=3, color='b', ls='dashdot', label='68% (+ 2nd observable)', bins=np.linspace(10.4, 10.9, 8))

# ax3.hist(mcmc_table_b10_full['lowmass_slope'], histtype='step', lw=3, color='b', ls='-', label='full chain')
ax3.hist(mcmc_table_subset.T[2], histtype='step', lw=3, color='g', ls='-', label='68% full B10', bins=np.linspace(0.2, 0.5, 8))
ax3.hist(lowslope, histtype='step', lw=3, color='g', ls='dashdot', label='68% (+ 2nd observable)', bins=np.linspace(0.2, 0.5, 8))

# ax4.hist(mcmc_table_b10_full['highmass_slope'], histtype='step', lw=3, color='y', ls='-', label='full chain')
ax4.hist(mcmc_table_subset.T[3], histtype='step', lw=3, color='y', ls='-', label='68% full B10', bins=np.linspace(0.2, 1.2, 8))
ax4.hist(highslope, histtype='step', lw=3, color='y', ls='dashdot', label='68% (+ 2nd observable)', bins=np.linspace(0.2, 1.2, 8))

# ax5.hist(mcmc_table_b10_full['scatter'], histtype='step', lw=3, color='violet', ls='-', label='full chain')
ax5.hist(mcmc_table_subset.T[4], histtype='step', lw=3, color='violet', ls='-', label='68% full B10', bins=np.linspace(0.1, 0.5, 8))
ax5.hist(scatter, histtype='step', lw=3, color='violet', ls='dashdot', label='68% (+ 2nd observable)', bins=np.linspace(0.1, 0.5, 8))


ax1.title.set_text('Characteristic halo mass')
ax2.title.set_text('Characteristic stellar mass')
ax3.title.set_text('Low mass slope')
ax4.title.set_text('High mass slope')
ax5.title.set_text('Log-normal scatter in stellar mass')

plt.legend(loc='best')
plt.show()

################################################################################
# In a bin of M* what does the distribution of M_h look like?
################################################################################

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import os

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

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
    cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', 'cz', 'halo_macc', \
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
    cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red,\
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
    std_red = std_func_mod(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
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
    std_blue = std_func_mod(blue_stellar_mass_bins, \
        blue_cen_stellar_mass_arr, blue_deltav_arr)    
    std_blue = np.array(std_blue)

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
        red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
        blue_stellar_mass_bins[:-1])
            
    return std_red, std_blue, centers_red, centers_blue

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
                
        # if value == 0:
        #     if gals_df['colour_label'][idx] == 'R':
        #         cen_gals_red.append(gals_df['{0}'.format(randint)][idx])
        #         cen_halos_red.append(gals_df['halo_macc'][idx])
        #         f_red_cen_gals_red.append(gals_df['f_red'][idx])
        #     elif gals_df['colour_label'][idx] == 'B':
        #         cen_gals_blue.append(gals_df['{0}'.format(randint)][idx])
        #         cen_halos_blue.append(gals_df['halo_macc'][idx])
        #         f_red_cen_gals_blue.append(gals_df['f_red'][idx])

    cen_gals_red = np.array(cen_gals_red)
    cen_halos_red = np.log10(np.array(cen_halos_red))
    cen_gals_blue = np.array(cen_gals_blue)
    cen_halos_blue = np.log10(np.array(cen_halos_blue))

    return cen_gals_red, cen_halos_red, cen_gals_blue, cen_halos_blue, \
        f_red_cen_gals_red, f_red_cen_gals_blue


global survey
global path_to_figures
global mf_type
global gal_group_df_subset

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']

ver = 2.0
machine = 'mac'
mf_type = 'smf'
survey = 'eco'
nproc = 2

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

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


## Subset of 100 from latest total smf run on which group finding was done
chi2_file = path_to_proc + 'smhm_run6/{0}_chi2.txt'.format(survey)
if mf_type == 'smf' and survey == 'eco' and ver == 1.0:
    chain_file = path_to_proc + 'mcmc_{0}.dat'.format(survey)
else:
    chain_file = path_to_proc + 'smhm_run6/mcmc_{0}_raw.txt'.\
        format(survey)

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)

print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)

print('Getting subset of 100 Behroozi parameters')
mcmc_table_subset = get_paramvals_percentile(mcmc_table, 68, chi2)

## Latest colour run 
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

chi2_file = path_to_proc + 'smhm_colour_run17/{0}_colour_chi2.txt'.\
    format(survey)
chain_file = path_to_proc + 'smhm_colour_run17/mcmc_{0}_colour_raw.txt'.\
    format(survey)
randint_file = path_to_proc + 'smhm_colour_run17/{0}_colour_mocknum.txt'.\
    format(survey)

if survey == 'eco':
    # catl_file = path_to_raw + "eco/eco_all.csv"
    ## New catalog with group finder run on subset after applying M* and cz cuts
    catl_file = path_to_proc + "gal_group_eco_data.hdf5"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

print('Reading files')
chi2_colour = read_chi2(chi2_file)
mcmc_table_colour = read_mcmc(chain_file)
mock_nums_df = pd.read_csv(randint_file, header=None, names=['mock_num'], 
    dtype=int)
gal_group_df = read_mock_catl(path_to_proc + "gal_group.hdf5") 

mcmc_table_pctl, bf_params, bf_chi2, bf_randint = \
    get_paramvals_percentile(mcmc_table_colour, 68, chi2_colour, mock_nums_df)

mock_nums_picked = mcmc_table_pctl['mock_num']

mhalo_arr = []
mstar_arr = []
lowslope = []
highslope = []
scatter = []
for idx in mock_nums_picked:
    mhalo_arr.append(mcmc_table_subset.T[0][idx-1])
    mstar_arr.append(mcmc_table_subset.T[1][idx-1])
    lowslope.append(mcmc_table_subset.T[2][idx-1])
    highslope.append(mcmc_table_subset.T[3][idx-1])
    scatter.append(mcmc_table_subset.T[4][idx-1])
bf_scatter = 

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


maxis_bf_red, phi_bf_red, maxis_bf_blue, phi_bf_blue, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
        f_red_cen_blue, std_bf_red, std_bf_blue, std_cen_bf_red, \
            std_cen_bf_blue = get_best_fit_model(bf_params, bf_randint)

x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red,x_red_data,y_red_data = \
    Stats_one_arr(cen_halos_red,cen_gals_red,base=0.4,bin_statval='center', 
    arr_digit='y')
x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue,x_blue_data,y_blue_data = \
    Stats_one_arr(cen_halos_blue,cen_gals_blue,base=0.4,bin_statval='center', 
    arr_digit='y')

fig1 = plt.figure(figsize=(10,10))
plt.plot(x_bf_red,y_bf_red,color='darkred',lw=3,label='Best-fit',zorder=10)
plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=3,
    label='Best-fit',zorder=10)
plt.fill_between(x_bf_red, y_bf_red+y_std_bf_red, y_bf_red-y_std_bf_red, 
    color='indianred', alpha=0.6)
plt.fill_between(x_bf_blue, y_bf_blue+y_std_bf_blue, y_bf_blue-y_std_bf_blue, 
    color='cornflowerblue', alpha=0.6)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
plt.show()

bw_std_red = std_cen_bf_red[1] - std_cen_bf_red[0]
bw_std_blue = std_cen_bf_blue[1] - std_cen_bf_blue[0]
blue_min = std_cen_bf_blue[3] - 0.5*bw_std_blue
blue_max = std_cen_bf_blue[3] + 0.5*bw_std_blue
red_min = std_cen_bf_red[3] - 0.5*bw_std_red
red_max = std_cen_bf_red[3] + 0.5*bw_std_red


red_halos_in_bin = []
for idx, value in enumerate(cen_gals_red):
    if value >= 10.16 and value <= 10.68:
        red_halos_in_bin.append(cen_halos_red[idx])

blue_halos_in_bin = []
for idx, value in enumerate(cen_gals_blue):
    if value >= 9.86 and value <= 10.28:
        blue_halos_in_bin.append(cen_halos_blue[idx])

fig2 = plt.figure(figsize=(10,10))
plt.hist(red_halos_in_bin, histtype='step', lw=3, color='r', ls='-', label='10.16 - 10.68')
plt.hist(blue_halos_in_bin, histtype='step', lw=3, color='b', ls='-', label='9.86 - 10.28')
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=25)
plt.title('Distribution of halo masses in specified stellar mass bin')
plt.legend()
plt.show()