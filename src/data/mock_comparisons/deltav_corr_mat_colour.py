"""
{This script plots stellar mass function for red and blue galaxies from data 
 AND mocks and also plots correlation matrices calculated two different ways.}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import rc
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import math
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

def get_err_smf(survey, path):
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

def get_err_deltav(survey, path):

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

    # logmstar_min_red_arr = []
    # logmstar_min_blue_arr = []
    # logmstar_max_red_arr = []
    # logmstar_max_blue_arr = []
    deltav_binned_red_arr = []
    # logmstar_binned_red_arr = []
    deltav_binned_blue_arr = []
    # logmstar_binned_blue_arr = []
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

            # for index,val in enumerate(deltav_binned_blue[0]):
            #     logmstar_binned_blue_arr.append(val)
            #     deltav_binned_blue_arr.append(deltav_binned_blue[1][index])

            # To get binning
            # min_red = np.min(mock_pd.logmstar.loc[(mock_pd.colour_label == 'R') 
            #     & (mock_pd.cs_flag == 1)])
            # max_red = np.max(mock_pd.logmstar.loc[(mock_pd.colour_label == 'R') 
            #     & (mock_pd.cs_flag == 1)])
            # min_blue = np.min(mock_pd.logmstar.loc[(mock_pd.colour_label == 'B') 
            #     & (mock_pd.cs_flag == 1)])
            # max_blue = np.max(mock_pd.logmstar.loc[(mock_pd.colour_label == 'B') 
            #     & (mock_pd.cs_flag == 1)])
            
            # logmstar_min_red_arr.append(min_red)
            # logmstar_max_red_arr.append(max_red)
            # logmstar_min_blue_arr.append(min_blue)
            # logmstar_max_blue_arr.append(max_blue)

    # logmstar_binned_red_arr = np.array(logmstar_binned_red_arr)
    # logmstar_binned_blue_arr = np.array(logmstar_binned_blue_arr)
    deltav_binned_red_arr = np.array(deltav_binned_red_arr)
    deltav_binned_blue_arr = np.array(deltav_binned_blue_arr)

    return deltav_binned_red_arr, deltav_binned_blue_arr

def measure_all_smf(table, volume, data_bool):
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
            diff_smf(table[logmstar_col], volume, True)
        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
            volume, True, 'R')
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
            volume, True, 'B')
    
    return [max_total, phi_total, err_total, counts_total] , \
        [max_red, phi_red, err_red, counts_red] , \
            [max_blue, phi_blue, err_blue, counts_blue]

global model_init
global survey
global path_to_proc
global mf_type

survey = 'eco'
machine = 'mac'
mf_type = 'smf'

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
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

deltav_binned_red_arr, deltav_binned_blue_arr = get_err_deltav(survey, 
    path_to_mocks)
phi_red_arr, phi_blue_arr = get_err_smf(survey, path_to_mocks)
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

    dv_red_0 = deltav_binned_red_arr[:,0]
    dv_red_1 = deltav_binned_red_arr[:,1]
    dv_red_2 = deltav_binned_red_arr[:,2]
    dv_red_3 = deltav_binned_red_arr[:,3]
    dv_red_4 = deltav_binned_red_arr[:,4]

    dv_blue_0 = deltav_binned_blue_arr[:,0]
    dv_blue_1 = deltav_binned_blue_arr[:,1]
    dv_blue_2 = deltav_binned_blue_arr[:,2]
    dv_blue_3 = deltav_binned_blue_arr[:,3]
    dv_blue_4 = deltav_binned_blue_arr[:,4]
    combined_df = pd.DataFrame({'phi_red_0':phi_red_0, 'phi_red_1':phi_red_1,\
        'phi_red_2':phi_red_2, 'phi_red_3':phi_red_3, 'phi_red_4':phi_red_4, \
        'phi_blue_0':phi_blue_0, 'phi_blue_1':phi_blue_1, 
        'phi_blue_2':phi_blue_2, 'phi_blue_3':phi_blue_3, 
        'phi_blue_4':phi_blue_4, \
        'dv_red_0':dv_red_0, 'dv_red_1':dv_red_1, 'dv_red_2':dv_red_2, \
        'dv_red_3':dv_red_3, 'dv_red_4':dv_red_4, \
        'dv_blue_0':dv_blue_0, 'dv_blue_1':dv_blue_1, 'dv_blue_2':dv_blue_2, \
        'dv_blue_3':dv_blue_3, 'dv_blue_4':dv_blue_4})

else:
    cov_mat_colour = np.cov(deltav_binned_red_arr, deltav_binned_blue_arr, 
        rowvar=False)
    err_colour = np.sqrt(cov_mat_colour.diagonal())
    corr_mat_colour = cov_mat_colour / np.outer(err_colour, err_colour)                                                           
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour)


# Correlation matrix of phi and deltav colour measurements combined
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(combined_df.corr(), cmap=cmap)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig1.colorbar(cax)
plt.title(r'Combined \boldmath{$\Phi$} and \boldmath{$\Delta$}v measurements')
plt.show()

# Correlation matrix of just deltav measurements combined
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(corr_mat_colour, cmap=cmap)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig2.colorbar(cax)
plt.title(r'Combined \boldmath{$\delta$}v measurements')
plt.show()
