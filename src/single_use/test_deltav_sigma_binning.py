"""
{This script tests different binning for sigma measurement of red and blue 
 galaxies from both survey mocks and data.}
"""

from cosmo_utils.utils import work_paths as cwpaths
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

def std_func(bins, mass_arr, vel_arr):

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

def get_deltav_sigma(df, df_type):
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
    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_keys: 
        group = red_groups.get_group(key) 
        if 1 in group[cs_colname].values: 
            cen_stellar_mass = group[mstar_colname].loc[group[cs_colname].\
                values == 1].values[0]
            if cen_stellar_mass < 8.0:
                print(group.name)
            mean_cz_grp = np.round(np.mean(group[cz_colname].values),2)
            deltav = group[cz_colname].values - len(group)*[mean_cz_grp]
            for val in deltav:
                red_deltav_arr.append(val)
                red_cen_stellar_mass_arr.append(cen_stellar_mass)

    # red_stellar_mass_bins = np.arange(8.6,11.5,0.25)
    red_stellar_mass_bins = np.linspace(8.55,11.5,6)
    std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
        red_deltav_arr)                                                           
    
    blue_groups = blue_subset.groupby(grp_id_colname)
    blue_keys = blue_groups.groups.keys()

    # Calculating velocity dispersion for blue galaxies in data
    blue_deltav_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_keys: 
        group = blue_groups.get_group(key)  
        if 1 in group[cs_colname].values: 
            cen_stellar_mass = group[mstar_colname].loc[group[cs_colname]\
                .values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group[cz_colname].values),2)
            deltav = group[cz_colname].values - len(group)*[mean_cz_grp]
            for val in deltav:
                blue_deltav_arr.append(val)
                blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    # blue_stellar_mass_bins = np.arange(8.6,11,0.25)
    blue_stellar_mass_bins = np.linspace(8.55,10.5,6)
    std_blue = std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
        blue_deltav_arr)    

    centers_red = 0.5 * (red_stellar_mass_bins[1:] + red_stellar_mass_bins[:-1])
    centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + blue_stellar_mass_bins[:-1])

    std_red, std_blue = np.array(std_red), np.array(std_blue)
    
    return std_red, std_blue, centers_red, centers_blue

def get_deltav_sigma_mocks(survey, path):
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

            # Calculating spread in velocity dispersion for red galaxies
            red_deltav_arr = []
            red_cen_stellar_mass_arr = []
            groupid_arr = []
            for key in red_keys: 
                group = red_groups.get_group(key)               
                if 1 in group.cs_flag.values: 
                    cen_stellar_mass = group.logmstar.loc[group.cs_flag.\
                        values == 1].values[0]
                    mean_cz_grp = np.round(np.mean(group.cz.values),2)
                    deltav = group.cz.values - len(group)*[mean_cz_grp]
                    for val in deltav:
                        red_deltav_arr.append(val)
                        red_cen_stellar_mass_arr.append(cen_stellar_mass)

            print(min(red_cen_stellar_mass_arr))

            # red_stellar_mass_bins = np.arange(8.6,11.5,0.25)
            red_stellar_mass_bins = np.linspace(8.55,11.5,7)
            std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
                red_deltav_arr)
            std_red = np.array(std_red)
            std_red_arr.append(std_red)                                                       
            
            blue_groups = blue_subset.groupby('groupid')
            blue_keys = blue_groups.groups.keys()

            # Calculating spread in velocity dispersion for blue galaxies
            blue_deltav_arr = []
            blue_cen_stellar_mass_arr = []
            for key in blue_keys: 
                group = blue_groups.get_group(key)  
                if 1 in group.cs_flag.values: 
                    cen_stellar_mass = group.logmstar.loc[group.cs_flag\
                        .values == 1].values[0]
                    mean_cz_grp = np.round(np.mean(group.cz.values),2)
                    deltav = group.cz.values - len(group)*[mean_cz_grp]
                    for val in deltav:
                        blue_deltav_arr.append(val)
                        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            print(min(blue_cen_stellar_mass_arr))

            # blue_stellar_mass_bins = np.arange(8.6,11,0.25)
            blue_stellar_mass_bins = np.linspace(8.55,10.5,6)
            std_blue = std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
                blue_deltav_arr)    
            std_blue = np.array(std_blue)
            std_blue_arr.append(std_blue)

            centers_red = 0.5 * (red_stellar_mass_bins[1:] + red_stellar_mass_bins[:-1])
            centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + blue_stellar_mass_bins[:-1])
            
            centers_red_arr.append(centers_red)
            centers_blue_arr.append(centers_blue)

    std_red_arr = np.array(std_red_arr)
    centers_red_arr = np.array(centers_red_arr)
    std_blue_arr = np.array(std_blue_arr)
    centers_blue_arr = np.array(centers_blue_arr)
            
    return std_red_arr, std_blue_arr, centers_red_arr, centers_blue_arr

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
                    (resolve_live18.absrmag.values <= -17.33)]
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


global survey
global mf_type

survey = 'eco'
machine = 'mac'
mf_type = 'smf'

dict_of_paths = cwpaths.cookiecutter_paths() 
path_to_raw = dict_of_paths['raw_dir'] 
path_to_data = dict_of_paths['data_dir']

catl_file = path_to_raw + "eco/eco_all.csv"
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

catl, volume, cvar, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)

std_red_data, std_blue_data, centers_red_data, \
    centers_blue_data = get_deltav_sigma(catl, 'data')

std_red_mocks, std_blue_mocks, centers_red_mocks, \
    centers_blue_mocks = get_deltav_sigma_mocks(survey, path_to_mocks)

fig = plt.figure(figsize=(12,10))
for idx in range(len(centers_red_mocks)):
    plt.scatter(centers_red_mocks[idx], std_red_mocks[idx], 
        c='indianred')
    plt.scatter(centers_blue_mocks[idx], std_blue_mocks[idx], 
        c='cornflowerblue')
plt.scatter(centers_red_data, std_red_data, marker='*', c='darkred', \
    s=100, label='data')
plt.scatter(centers_blue_data, std_blue_data, marker='*', \
    c='darkblue', s=100, label='data')
plt.xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$', labelpad=15, 
    fontsize=25)
plt.ylabel(r'\boldmath$\sigma \left[km/s\right]$', labelpad=15, 
    fontsize=25)
plt.show()