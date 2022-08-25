"""
{This script plots all 3 observables used in mcmc analysis for data where the 
buffer is included and for data where the buffer is excluded}
"""

from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
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

def read_data_catl(path_to_file, survey, buffer=False):
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
        eco_catl = pd.read_csv(path_to_file,delimiter=",", header=0)

        # eco_catl = reading_catls(path_to_file)
        if buffer:
            czmin = 2530
            czmax = 7470
            volume = 192351.36 # Survey volume with buffer [Mpc/h]^3
        else:
            czmin = 3000
            czmax = 7000
            volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_catl.loc[(eco_catl.cz.values >= czmin) & 
                (eco_catl.cz.values <= czmax) & 
                (eco_catl.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_catl.loc[(eco_catl.cz.values >= czmin) & 
                (eco_catl.cz.values <= czmax) & 
                (eco_catl.absrmag.values <= -17.33)] 

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
   
    if catl_type == 'old':
        groupid_col = 'grp'
        galtype_col = 'fc'
    elif catl_type == 'new':
        groupid_col = 'groupid'
        galtype_col = 'g_galtype'

    red_subset_grpids = np.unique(catl[groupid_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(catl[groupid_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    # Calculating spread in velocity dispersion for galaxies in groups with a 
    # red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl[groupid_col] == key]
        cen_stellar_mass = group.logmstar.loc[group[galtype_col].\
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
        group = catl.loc[catl[groupid_col] == key]
        cen_stellar_mass = group.logmstar.loc[group[galtype_col]\
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

    if catl_type == 'old':
        groupid_col = 'grp'
        galtype_col = 'fc'
    elif catl_type == 'new':
        groupid_col = 'groupid'
        galtype_col = 'g_galtype'

    red_subset_grpids = np.unique(catl[groupid_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(catl[groupid_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl[groupid_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group[galtype_col]\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[galtype_col] == 1].values[0]
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
        group = catl.loc[catl[groupid_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group[galtype_col]\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[galtype_col] == 1].values[0]
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

global survey
global mf_type
global quenching
global catl_type

survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
catl_type = 'old' # ECO catalog type : old - from wiki, new - Post group-finder

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if survey == 'eco':
    # catl_file = path_to_proc + "gal_group_eco_data.hdf5"
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

catl, volume, z_median = read_data_catl(catl_file, survey, buffer=False)
catl = assign_colour_label_data(catl)

total_data_nobuff, red_data_nobuff, blue_data_nobuff = measure_all_smf(catl, \
    volume, data_bool=True)

print('Measuring spread in vel disp for data')
std_red_nobuff, old_centers_red_nobuff, std_blue_nobuff, \
    old_centers_blue_nobuff = get_deltav_sigma_data(catl)

print('Measuring binned spread in vel disp for data')
mean_grp_cen_red_nobuff, new_centers_red_nobuff, mean_grp_cen_blue_nobuff, \
    new_centers_blue_nobuff = get_sigma_per_group_data(catl)

catl, volume, z_median = read_data_catl(catl_file, survey, buffer=True)
catl = assign_colour_label_data(catl)

total_data_buff, red_data_buff, blue_data_buff = measure_all_smf(catl, volume, \
    data_bool=True)

print('Measuring spread in vel disp for data')
std_red_buff, old_centers_red_buff, std_blue_buff, old_centers_blue_buff = \
    get_deltav_sigma_data(catl)

print('Measuring binned spread in vel disp for data')
mean_grp_cen_red_buff, new_centers_red_buff, mean_grp_cen_blue_buff, \
    new_centers_blue_buff = get_sigma_per_group_data(catl)

fig1= plt.figure(figsize=(10,10))
plt.plot(red_data_nobuff[0], red_data_nobuff[1], c='darkred', ls='-', lw=3, 
    label='Without buffer')
plt.plot(blue_data_nobuff[0], blue_data_nobuff[1], c='darkblue', ls='-', lw=3, 
    label='Without buffer')
plt.plot(red_data_buff[0], red_data_buff[1], c='darkred', ls='--', lw=3, 
    label='With buffer')
plt.plot(blue_data_buff[0], blue_data_buff[1], c='darkblue', ls='--', lw=3, 
    label='With buffer')
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)
plt.title('ECO SMF')
plt.legend(prop={'size': 20})
plt.show()

fig2= plt.figure(figsize=(10,10))
plt.plot(old_centers_red_nobuff, std_red_nobuff, c='darkred', ls='-', lw=3, 
    label='Without buffer')
plt.plot(old_centers_blue_nobuff, std_blue_nobuff, c='darkblue', ls='-', lw=3, 
    label='Without buffer')
plt.plot(old_centers_red_buff, std_red_buff, c='darkred', ls='--', lw=3, 
    label='With buffer')
plt.plot(old_centers_blue_buff, std_blue_buff, c='darkblue', ls='--', lw=3, 
    label='With buffer')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.title(r'ECO spread in velocity difference vs group central $M_{*}$')
plt.legend(prop={'size': 20})
plt.show()

fig3= plt.figure(figsize=(10,10))
plt.plot(new_centers_red_nobuff, mean_grp_cen_red_nobuff[0], c='darkred', ls='-', \
    lw=3, label='Without buffer')
plt.plot(new_centers_blue_nobuff, mean_grp_cen_blue_nobuff[0], c='darkblue', \
    ls='-', lw=3, label='Without buffer')
plt.plot(new_centers_red_buff, mean_grp_cen_red_buff[0], c='darkred', ls='--', \
    lw=3, label='With buffer')
plt.plot(new_centers_blue_buff, mean_grp_cen_blue_buff[0], c='darkblue', \
    ls='--', lw=3, label='With buffer')
plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
plt.title(r'ECO average group central $M_{*}$ vs spread in velocity difference')
plt.legend(prop={'size': 20})
plt.show()

"""
Conclusion: 

There isn't a significant difference between the 3 observables whether you use
data with and without the buffer. The difference between these two cases 
becomes significant (for the 2nd and 3rd observable) at the massive/highest 
spread in velocity in difference end when you filter data by cz instead of 
grpcz in read_data_catl()
"""
