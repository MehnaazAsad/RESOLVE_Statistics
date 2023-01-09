from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import h5py
import os

# from matplotlib.legend_handler import HandlerTuple
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from matplotlib import cm

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
# rc('text', usetex=True)
# rc('text.latex', preamble=r"\usepackage{amsmath}")
# rc('axes', linewidth=2)
# rc('xtick.major', width=4, size=7)
# rc('ytick.major', width=4, size=7)
# rc('xtick.minor', width=2, size=7)
# rc('ytick.minor', width=2, size=7)

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

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

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
            # For eco total
            #* Changed max bin from 11.5 to 11.1 to be the same as mstar-sigma (10.8)
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
            bin_num = 5

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """
    ## Mocks case different than data because of censat_col
    if not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

            #* Changed max bin from 11.5 to 11.1 to be the same as mstar-sigma (10.8)
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
            bin_num = 5
            bins = np.linspace(bin_min, bin_max, bin_num)

        elif mf_type == 'bmf':
            mbary_limit = 9.3
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 5
            bins = np.linspace(bin_min, bin_max, bin_num)


    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
        # if max(red_cen_stellar_mass_arr) > 12:
        #     print(max(red_cen_stellar_mass_arr))
        #     cen_red_subset_df.to_csv("cen_red_subset_df_{0}.csv".format(max(red_cen_stellar_mass_arr)))

    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
    """
    Calculates differential baryonic mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of baryonic masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

    sim_bool: boolean
        True if masses are from mock

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
        logmbary_arr = np.log10((10**mass_arr) / 2.041)
    else:
        logmbary_arr = np.log10(mass_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.3) / 2.041), 1)

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

def halo_quenching_model(theta, gals_df, mock):
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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
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

def get_host_halo_mock(df, mock):
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
        cen_halos = df.halo_mvir[df.cs_flag == 1].reset_index(drop=True)
        sat_halos = df.halo_mvir_host_halo[df.cs_flag == 0].reset_index(drop=True)
    else:
        # Both cen and sat are the same mass for a halo i.e. the satellites
        # are assigned a halo mass of the central. 
        cen_halos = 10**(df.loghalom[df.cs_flag == 1]).reset_index(drop=True)
        sat_halos = 10**(df.loghalom[df.cs_flag == 0]).reset_index(drop=True)

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(df, mock, randint=None):
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

    if mock == 'vishnu' and randint:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['{0}'.format(randint)].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['{0}'.format(randint)].values[idx]))

    elif mock == 'vishnu':
        cen_gals = 10**(df.logmstar[df.cs_flag == 1]).reset_index(drop=True)
        sat_gals = 10**(df.logmstar[df.cs_flag == 0]).reset_index(drop=True)
    
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

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            newgroupid[galsel]=brokenupids[galsel]
        else:
            pass
    return newgroupid 



survey = 'eco'
mf_type = 'smf'
quenching = 'halo'
machine = 'bender'
level = 'group'
stacked_stat = True

print("Survey: {0}\n".format(survey))
print("Mass function: {0}\n".format(mf_type))
print("Quenching: {0}\n".format(quenching))
print("Level: {0}\n".format(level))
print("M*-sigma: {0}\n".format(stacked_stat))

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']
    
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

mock_name = 'ECO'
num_mocks = 8
min_cz = 3000
max_cz = 7000
mag_limit = -17.33
mstar_limit = 8.9
volume = 151829.26 # Survey volume without buffer [Mpc/h]^3

phi_global = []
fblue_cen_global = []
fblue_sat_global = []
mean_mstar_red_global = []
mean_mstar_blue_global = []
sigma_red_global = []
sigma_blue_global = []
for i in tqdm(range(100)):
    phi_total_arr = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    sigma_red_arr = []
    sigma_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in tqdm(box_id_arr):
        box = int(box)
        temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            # print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            # Since assigning new groups means that some numbers are removed 
            # i.e. there are gaps in the range of new groupids, these gaps 
            # need to be filled 
            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)


            if mf_type == "smf":
                if quenching == "hybrid" and not stacked_stat:
                    # from #75 used for hybrid sigma-mstar chain 80
                    bf_from_last_chain = [10.215486, 13.987752, 0.753758, 0.025111]

                    Mstar_q = bf_from_last_chain[0] # Msun/h**2
                    Mh_q = bf_from_last_chain[1] # Msun/h
                    mu = bf_from_last_chain[2]
                    nu = bf_from_last_chain[3]

                elif quenching == "hybrid" and stacked_stat:
                    # from #84
                    bf_from_last_chain = [10.18868649, 13.23990274, 0.72632304, 0.05996219]
                    
                    ## This set was not used in this code to make matrices.
                    ## Using best-fit found for new ECO data using result from chain 59
                    ## i.e. hybrid quenching model which was the last time M*-sigma was
                    ## used i.e. stacked_stat = True
                    # bf_from_last_chain = [10.133745, 13.478087, 0.810922, 0.043523]


                    Mstar_q = bf_from_last_chain[0] # Msun/h**2
                    Mh_q = bf_from_last_chain[1] # Msun/h
                    mu = bf_from_last_chain[2]
                    nu = bf_from_last_chain[3]

                elif quenching == "halo" and not stacked_stat:
                    # from #76 used for halo sigma-mstar chain 81
                    bf_from_last_chain = [11.7784379, 12.1174944, 1.62771601, 0.131290924]
                   
                    Mh_qc = bf_from_last_chain[0] # Msun/h
                    Mh_qs = bf_from_last_chain[1] # Msun/h
                    mu_c = bf_from_last_chain[2]
                    mu_s = bf_from_last_chain[3]

                elif quenching == "halo" and stacked_stat:
                    #from #86
                    bf_from_last_chain = [11.928341 , 12.60596691, 1.63365685, 0.35175002]
                    Mh_qc = bf_from_last_chain[0] # Msun/h
                    Mh_qs = bf_from_last_chain[1] # Msun/h
                    mu_c = bf_from_last_chain[2]
                    mu_s = bf_from_last_chain[3]

            elif mf_type == "bmf":
                if quenching == "hybrid" and stacked_stat:
                    # from #68 
                    bf_from_last_chain = [10.40868054, 14.06677972,  0.83745589,  0.14406545]
                    #! Compare to latest 10.39975208, 14.37426995, 0.94420486, 0.10782901
                    Mstar_q = bf_from_last_chain[0] # Msun/h**2
                    Mh_q = bf_from_last_chain[1] # Msun/h
                    mu = bf_from_last_chain[2]
                    nu = bf_from_last_chain[3]

                elif quenching == "halo" and stacked_stat:
                    #from #83 used for chain 90 
                    #These are the same params as used for halo, smf case 
                    #since this is the very first halo quenching baryonic 
                    #chain to be run
                    bf_from_last_chain = [12.01240587, 12.51050784, 1.40100554, 0.44524407]
                    #! Compare to latest 11.97920231, 12.86108614, 1.75227584, 0.48775956
                    Mh_qc = bf_from_last_chain[0] # Msun/h
                    Mh_qs = bf_from_last_chain[1] # Msun/h
                    mu_c = bf_from_last_chain[2]
                    mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)
            
            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')
                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,10.8,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,10.8,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.0,11.2,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.0,11.2,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                sigma_red_arr.append(sigma_red)
                sigma_blue_arr.append(sigma_blue)
            
            #* Changed from else to if since now we want to include both stats
            if stacked_stat:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    sigma_red_arr = np.array(sigma_red_arr)
    sigma_blue_arr = np.array(sigma_blue_arr)

    phi_global.append(phi_arr_total)
    fblue_cen_global.append(f_blue_cen_arr)
    fblue_sat_global.append(f_blue_sat_arr)
    mean_mstar_red_global.append(mean_mstar_red_arr)
    mean_mstar_blue_global.append(mean_mstar_blue_arr)
    sigma_red_global.append(sigma_red_arr)
    sigma_blue_global.append(sigma_blue_arr)

phi_global = np.array(phi_global)
fblue_cen_global = np.array(fblue_cen_global)
fblue_sat_global = np.array(fblue_sat_global)
mean_mstar_red_global = np.array(mean_mstar_red_global)
mean_mstar_blue_global = np.array(mean_mstar_blue_global)
sigma_red_global = np.array(sigma_red_global)
sigma_blue_global = np.array(sigma_blue_global)

hf = h5py.File(path_to_proc + 'corr_matrices_28stats_{0}_{1}.h5'.format(quenching, mf_type), 'w')
hf.create_dataset('smf', data=phi_global)
hf.create_dataset('fblue_cen', data=fblue_cen_global)
hf.create_dataset('fblue_sat', data=fblue_sat_global)
hf.create_dataset('sigma_red', data=sigma_red_global)
hf.create_dataset('sigma_blue', data=sigma_blue_global)
hf.create_dataset('mean_mstar_red', data=mean_mstar_red_global)
hf.create_dataset('mean_mstar_blue', data=mean_mstar_blue_global)
hf.close()

if stacked_stat:
    hf = h5py.File(path_to_proc + 'corr_matrices_xy_mstarsigma_oldbins_{0}.h5'.format(quenching), 'w')
    hf.create_dataset('smf', data=phi_global)
    hf.create_dataset('fblue_cen', data=fblue_cen_global)
    hf.create_dataset('fblue_sat', data=fblue_sat_global)
    hf.create_dataset('mean_sigma_red', data=mean_mstar_red_global)
    hf.create_dataset('mean_sigma_blue', data=mean_mstar_blue_global)
    hf.close()
else:
    hf = h5py.File(path_to_proc + 'corr_matrices_{0}.h5'.format(quenching), 'w')
    hf.create_dataset('smf', data=phi_global)
    hf.create_dataset('fblue_cen', data=fblue_cen_global)
    hf.create_dataset('fblue_sat', data=fblue_sat_global)
    hf.create_dataset('mean_mstar_red', data=mean_mstar_red_global)
    hf.create_dataset('mean_mstar_blue', data=mean_mstar_blue_global)
    hf.close()

# Read in datasets from h5 file and calculate corr matrix

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat


hf_read = h5py.File(path_to_proc + 'corr_matrices_28stats_{0}_{1}.h5'.
    format(quenching, mf_type), 'r')
hf_read = h5py.File('/Users/asadm2/Desktop/corr_matrices_28stats_{0}_{1}.h5'.
    format(quenching, mf_type), 'r')

hf_read.keys()
smf = hf_read.get('smf')
smf = np.squeeze(np.array(smf))
fblue_cen = hf_read.get('fblue_cen')
fblue_cen = np.array(fblue_cen)
fblue_sat = hf_read.get('fblue_sat')
fblue_sat = np.array(fblue_sat)
mean_mstar_red = hf_read.get('mean_mstar_red')
mean_mstar_red = np.array(mean_mstar_red)
mean_mstar_blue = hf_read.get('mean_mstar_blue')
mean_mstar_blue = np.array(mean_mstar_blue)
sigma_red = hf_read.get('sigma_red')
sigma_red = np.array(sigma_red)
sigma_blue = hf_read.get('sigma_blue')
sigma_blue = np.array(sigma_blue)

for i in range(100):
    phi_total_0 = smf[i][:,0]
    phi_total_1 = smf[i][:,1]
    phi_total_2 = smf[i][:,2]
    phi_total_3 = smf[i][:,3]

    f_blue_cen_0 = fblue_cen[i][:,0]
    f_blue_cen_1 = fblue_cen[i][:,1]
    f_blue_cen_2 = fblue_cen[i][:,2]
    f_blue_cen_3 = fblue_cen[i][:,3]

    f_blue_sat_0 = fblue_sat[i][:,0]
    f_blue_sat_1 = fblue_sat[i][:,1]
    f_blue_sat_2 = fblue_sat[i][:,2]
    f_blue_sat_3 = fblue_sat[i][:,3]

    mstar_red_cen_0 = mean_mstar_red[i][:,0]
    mstar_red_cen_1 = mean_mstar_red[i][:,1]
    mstar_red_cen_2 = mean_mstar_red[i][:,2]
    mstar_red_cen_3 = mean_mstar_red[i][:,3]

    mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
    mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
    mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
    mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

    sigma_red_0 = sigma_red[i][:,0]
    sigma_red_1 = sigma_red[i][:,1]
    sigma_red_2 = sigma_red[i][:,2]
    sigma_red_3 = sigma_red[i][:,3]

    sigma_blue_0 = sigma_blue[i][:,0]
    sigma_blue_1 = sigma_blue[i][:,1]
    sigma_blue_2 = sigma_blue[i][:,2]
    sigma_blue_3 = sigma_blue[i][:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3,
        'sigma_red_0':sigma_red_0, 'sigma_red_1':sigma_red_1, 
        'sigma_red_2':sigma_red_2, 'sigma_red_3':sigma_red_3,
        'sigma_blue_0':sigma_blue_0, 'sigma_blue_1':sigma_blue_1, 
        'sigma_blue_2':sigma_blue_2, 'sigma_blue_3':sigma_blue_3})

    if i == 0:
        # Correlation matrix of phi and deltav colour measurements combined
        corr_mat_global = combined_df.corr()
        cov_mat_global = combined_df.cov()

        corr_mat_average = corr_mat_global
        cov_mat_average = cov_mat_global
    else:
        corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
        cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
        

# Using average cov mat to get correlation matrix
corr_mat_average = calc_corr_mat(cov_mat_average)
corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
sigma_average = np.sqrt(np.diag(cov_mat_average))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cmap = cm.get_cmap('Spectral_r')
cax = ax1.matshow(corr_mat_average, cmap=cmap, vmin=-1, vmax=1)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
plt.colorbar(cax)
plt.title('{0}'.format(quenching))
plt.show()


###################################################################
if stacked_stat:
    hf_read = h5py.File(path_to_proc + 'corr_matrices_xy_mstarsigma_oldbins_{0}.h5'.format(quenching), 'r')
else:
    hf_read = h5py.File(path_to_proc + 'corr_matrices_{0}.h5'.format(quenching), 'r')
hf_read.keys()
smf = hf_read.get('smf')
smf = np.array(smf)
fblue_cen = hf_read.get('fblue_cen')
fblue_cen = np.array(fblue_cen)
fblue_sat = hf_read.get('fblue_sat')
fblue_sat = np.array(fblue_sat)
mean_mstar_red = hf_read.get('mean_mstar_red')
mean_mstar_red = np.array(mean_mstar_red)
mean_mstar_blue = hf_read.get('mean_mstar_blue')
mean_mstar_blue = np.array(mean_mstar_blue)

#* Values from chain 74
#* bf_bin_params = [1.25265896e+01, 1.05945720e+01, 4.53178499e-01, 5.82146045e-01,
#*       3.19537853e-01, 1.01463004e+01, 1.43383461e+01, 6.83946944e-01,
#*       1.41126799e-02]
#* bf_bin_chi2 = 30.84

#* Using DR3 with pair-splitting since matrix included pair splitting + new bins
data_stats = [-1.52518353, -1.67537112, -1.89183513, -2.60866256,  0.82258065,
        0.58109091,  0.32606325,  0.09183673,  0.46529814,  0.31311707,
        0.21776504,  0.04255319, 10.33120738, 10.44089237, 10.49140916,
        10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856]

model_stats = [-1.6519256 , -1.8308534 , -2.06871355, -2.76088258,  0.80957628,
        0.61997687,  0.32985039,  0.06624606,  0.48458904,  0.35380835,
        0.17920657,  0.0754717 , 10.34519482, 10.37549973, 10.56002331,
        10.72787762,  9.92759132,  9.87016296, 10.05955315, 10.15922451]

data_stats = np.array(data_stats)
model_stats = np.array(model_stats)

chi_squared_arr = []
sigma_arr = []
for i in range(100):
    phi_total_0 = smf[i][:,0]
    phi_total_1 = smf[i][:,1]
    phi_total_2 = smf[i][:,2]
    phi_total_3 = smf[i][:,3]

    f_blue_cen_0 = fblue_cen[i][:,0]
    f_blue_cen_1 = fblue_cen[i][:,1]
    f_blue_cen_2 = fblue_cen[i][:,2]
    f_blue_cen_3 = fblue_cen[i][:,3]

    f_blue_sat_0 = fblue_sat[i][:,0]
    f_blue_sat_1 = fblue_sat[i][:,1]
    f_blue_sat_2 = fblue_sat[i][:,2]
    f_blue_sat_3 = fblue_sat[i][:,3]

    mstar_red_cen_0 = mean_mstar_red[i][:,0]
    mstar_red_cen_1 = mean_mstar_red[i][:,1]
    mstar_red_cen_2 = mean_mstar_red[i][:,2]
    mstar_red_cen_3 = mean_mstar_red[i][:,3]

    mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
    mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
    mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
    mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

    sigma_red_0 = sigma_red[i][:,0]
    sigma_red_1 = sigma_red[i][:,1]
    sigma_red_2 = sigma_red[i][:,2]
    sigma_red_3 = sigma_red[i][:,3]

    sigma_blue_0 = sigma_blue[i][:,0]
    sigma_blue_1 = sigma_blue[i][:,1]
    sigma_blue_2 = sigma_blue[i][:,2]
    sigma_blue_3 = sigma_blue[i][:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3,
        'sigma_red_0':sigma_red_0, 'sigma_red_1':sigma_red_1, 
        'sigma_red_2':sigma_red_2, 'sigma_red_3':sigma_red_3,
        'sigma_blue_0':sigma_blue_0, 'sigma_blue_1':sigma_blue_1, 
        'sigma_blue_2':sigma_blue_2, 'sigma_blue_3':sigma_blue_3})

    if i == 0:
        # Correlation matrix of phi and deltav colour measurements combined
        corr_mat_global = combined_df.corr()
        cov_mat_global = combined_df.cov()

        corr_mat_average = corr_mat_global
        cov_mat_average = cov_mat_global
    else:
        corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
        cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
        
        corr_mat_global = corr_mat_global.append(combined_df.corr())
        cov_mat_global = cov_mat_global.append(combined_df.cov())
        

    corr_mat_inv_colour = np.linalg.inv(combined_df.corr().values)  
    sigma = np.sqrt(np.diag(combined_df.cov()))
    sigma_arr.append(sigma)

    chi_squared_i = chi_squared(data_stats, model_stats, sigma, corr_mat_inv_colour)
    chi_squared_arr.append(chi_squared_i)


#* Calculating chi2 using av. corr. mat and av. cov mat calculated independently
corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average.values) 
sigma_average = np.sqrt(np.diag(cov_mat_average))
# sigma_average = np.mean(sigma_arr, axis=0)
chi_squared_average = chi_squared(data_stats, model_stats, sigma_average, corr_mat_inv_colour_average)

#* Calculating chi2 using av. corr. mat derived from av. cov mat

#* Coded covariance matrix and correlation matrix to make sure 
#* df.corr() and np.cov() produce expected results (and they do!)
def calc_cov_mat(df):
    N = len(df)
    fact = 1/(N-1)

    df_mean = df.mean(axis=0).values
    num_cols = df.shape[1]
    cov_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            mean_y_i = df_mean[i]
            mean_y_j = df_mean[j]
            cov_mat[i][j] = fact * np.sum((df.values.T[i] - mean_y_i)*(df.values.T[j] - mean_y_j))
        
            #* Alt code
            # summation = 0
            # for n in range(N):
            #     summation += (df.values.T[i][n] - mean_y_i) * (df.values.T[j][n] - mean_y_j)
            # cov_mat[i][j] = fact * summation

    return cov_mat

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

# my_cov_mat = calc_cov_mat(combined_df)
#Using average cov mat
my_corr_mat = calc_corr_mat(cov_mat_average)
corr_mat_inv_colour_average = np.linalg.inv(my_corr_mat) 
sigma_average = np.sqrt(np.diag(cov_mat_average))
# sigma_average = np.mean(sigma_arr, axis=0)
chi_squared_average = chi_squared(data_stats, model_stats, sigma_average, corr_mat_inv_colour_average)

#* Calculating chi2 by varying models (bf from run 74) instead of matrix
model_init = halocat_init(halo_catalog, z_median)
bf_bin_params = [1.25265896e+01, 1.05945720e+01, 4.53178499e-01, 5.82146045e-01,
    3.19537853e-01, 1.01463004e+01, 1.43383461e+01, 6.83946944e-01,
    1.41126799e-02]
bf_bin_chi2 = 30.84
model_stats_global = []
for i in range(100):
    model_stats = lnprob(bf_bin_params)
    model_stats = model_stats.flatten()
    model_stats_global.append(model_stats)
model_stats_global = np.array(model_stats_global)

model_stats_global = np.genfromtxt('/Users/asadm2/Desktop/100models.csv', delimiter=',')

# Using average corr matrix and sigma from average cov matrix
corr_mat_inv_colour = np.linalg.inv(corr_mat_average.values)  
sigma = np.sqrt(np.diag(cov_mat_average))
chi_squared_100models_arr = []
for i in range(100):
    model_stats = model_stats_global[i]
    chi_squared_100models_i = chi_squared(data_stats, model_stats, sigma, corr_mat_inv_colour)
    chi_squared_100models_arr.append(chi_squared_100models_i)
chi_squared_100models_arr = np.array(chi_squared_100models_arr)

# Removing extremely large chi-squared of 83000+
chi_squared_100models_arr_mod = chi_squared_100models_arr[\
    chi_squared_100models_arr < max(chi_squared_100models_arr)]

fig1 = plt.figure(figsize=(10,8))
plt.hist(chi_squared_arr, histtype='step', lw=2, color='rebeccapurple', 
    label='100 matrices')
plt.hist(chi_squared_100models_arr_mod, histtype='step', lw=2, 
    color='goldenrod', label='100 models')
plt.axvline(np.average(chi_squared_arr), color='rebeccapurple', 
    linestyle='dashed', linewidth=1, 
    label=r'\boldmath$\overline{ \chi^2 }$ =' + 
    '{0}'.format(np.round(np.average(chi_squared_arr), 2)))
plt.axvline(np.average(chi_squared_100models_arr_mod), color='goldenrod',
    linestyle='dashed', linewidth=1, 
    label=r'\boldmath$\overline{ \chi^2 }$ =' + 
    '{0}'.format(np.round(np.average(chi_squared_100models_arr_mod), 2)))
plt.legend(loc="best",prop={'size':20})
plt.xlabel(r'\boldmath$\chi^{2}$')
plt.show()

#* 21 is index of model that's at the center of the yellow distribution

hf_read = h5py.File(path_to_proc +'corr_matrices.h5', 'r')
hf_read.keys()
smf = hf_read.get('smf')
smf = np.array(smf)
fblue_cen = hf_read.get('fblue_cen')
fblue_cen = np.array(fblue_cen)
fblue_sat = hf_read.get('fblue_sat')
fblue_sat = np.array(fblue_sat)
mean_mstar_red = hf_read.get('mean_mstar_red')
mean_mstar_red = np.array(mean_mstar_red)
mean_mstar_blue = hf_read.get('mean_mstar_blue')
mean_mstar_blue = np.array(mean_mstar_blue)

#* Using DR3 with pair-splitting since matrix included pair splitting + new bins
data_stats = [-1.52518353, -1.67537112, -1.89183513, -2.60866256,  0.82258065,
        0.58109091,  0.32606325,  0.09183673,  0.46529814,  0.31311707,
        0.21776504,  0.04255319, 10.33120738, 10.44089237, 10.49140916,
        10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856]

model_stats = [-1.65462665, -1.83406683, -2.06495113, -2.75657834,  0.80264609,
        0.6276114 ,  0.32879606,  0.08945687,  0.45818693,  0.36019536,
        0.18370166,  0.02666667, 10.35981178, 10.36600304, 10.54175568,
       10.73550415,  9.90159321,  9.9658432 , 10.03906345, 10.21994781]
    

data_stats = np.array(data_stats)
model_stats = np.array(model_stats)

chi_squared_arr = []
sigma_arr = []
for i in range(100):
    phi_total_0 = smf[i][:,0]
    phi_total_1 = smf[i][:,1]
    phi_total_2 = smf[i][:,2]
    phi_total_3 = smf[i][:,3]

    f_blue_cen_0 = fblue_cen[i][:,0]
    f_blue_cen_1 = fblue_cen[i][:,1]
    f_blue_cen_2 = fblue_cen[i][:,2]
    f_blue_cen_3 = fblue_cen[i][:,3]

    f_blue_sat_0 = fblue_sat[i][:,0]
    f_blue_sat_1 = fblue_sat[i][:,1]
    f_blue_sat_2 = fblue_sat[i][:,2]
    f_blue_sat_3 = fblue_sat[i][:,3]

    mstar_red_cen_0 = mean_mstar_red[i][:,0]
    mstar_red_cen_1 = mean_mstar_red[i][:,1]
    mstar_red_cen_2 = mean_mstar_red[i][:,2]
    mstar_red_cen_3 = mean_mstar_red[i][:,3]

    mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
    mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
    mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
    mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

    if i == 0:
        # Correlation matrix of phi and deltav colour measurements combined
        corr_mat_global = combined_df.corr()
        cov_mat_global = combined_df.cov()

        corr_mat_average = corr_mat_global
        cov_mat_average = cov_mat_global
    else:
        corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
        cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
        
        corr_mat_global = corr_mat_global.append(combined_df.corr())
        cov_mat_global = cov_mat_global.append(combined_df.cov())
        

    corr_mat_inv_colour = np.linalg.inv(combined_df.corr().values)  
    sigma = np.sqrt(np.diag(combined_df.cov()))
    sigma_arr.append(sigma)

    chi_squared_i = chi_squared(data_stats, model_stats, sigma, corr_mat_inv_colour)
    chi_squared_arr.append(chi_squared_i)

fig1 = plt.figure(figsize=(10,8))
plt.hist(chi_squared_arr, histtype='step', lw=2, color='rebeccapurple', 
    label='100 matrices')
plt.hist(chi_squared_100models_arr_mod, histtype='step', lw=2, 
    color='goldenrod', label='100 models')
plt.axvline(np.average(chi_squared_arr), color='rebeccapurple', 
    linestyle='dashed', linewidth=1, 
    label=r'\boldmath$\overline{ \chi^2 }$ =' + 
    '{0}'.format(np.round(np.average(chi_squared_arr), 2)))
plt.axvline(np.average(chi_squared_100models_arr_mod), color='goldenrod',
    linestyle='dashed', linewidth=1, 
    label=r'\boldmath$\overline{ \chi^2 }$ =' + 
    '{0}'.format(np.round(np.average(chi_squared_100models_arr_mod), 2)))
plt.legend(loc="best",prop={'size':20})
plt.xlabel(r'\boldmath$\chi^{2}$')
plt.show()

# Looking at most massive galaxy
for i in range(100):
    model.mock.populate()
    gals = model.mock.galaxy_table           
    print(np.log10(gals['stellar_mass'].max()))

#* Checking to see if distribution of cells across 64 mocks is gaussian 
from scipy.stats import normaltest
row_indices = corr_mat_global.index.unique()
p_arr = []
for i in row_indices:
    for j in row_indices:
        sk, p = normaltest(corr_mat_global.loc[corr_mat_global.index == i][j].values)
        p_arr.append(p)
p_arr = np.array(p_arr)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cmap = cm.get_cmap('Spectral_r')
cax = ax1.matshow(corr_mat_average, cmap=cmap, vmin=-1, vmax=1)
# cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
tick_marks = [i for i in range(len(combined_df.columns))]
names = [
r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]

tick_marks=[0, 4, 8, 12, 16]
names = [
    r"$\boldsymbol\phi$",
    r"$\boldsymbol{f_{blue}^{c}}$",
    r"$\boldsymbol{f_{blue}^{s}}$",
    r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
    r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

plt.xticks(tick_marks, names, fontsize=10)#, rotation='vertical')
plt.yticks(tick_marks, names, fontsize=10)    
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
plt.colorbar(cax)
plt.title('{0}'.format(quenching))
plt.show()

#* Matrix plot for when all 28 bins are included

hf_read = h5py.File('/Users/asadm2/Desktop/corr_matrices_28stats_{0}.h5'.format(quenching), 'r')
hf_read.keys()
smf = hf_read.get('smf')
smf = np.array(smf)
fblue_cen = hf_read.get('fblue_cen')
fblue_cen = np.array(fblue_cen)
fblue_sat = hf_read.get('fblue_sat')
fblue_sat = np.array(fblue_sat)
mean_mstar_red = hf_read.get('mean_mstar_red')
mean_mstar_red = np.array(mean_mstar_red)
mean_mstar_blue = hf_read.get('mean_mstar_blue')
mean_mstar_blue = np.array(mean_mstar_blue)
sigma_red = hf_read.get('sigma_red')
sigma_red = np.array(sigma_red)
sigma_blue = hf_read.get('sigma_blue')
sigma_blue = np.array(sigma_blue)


for i in range(1):
    phi_total_0 = smf[i][:,0]
    phi_total_1 = smf[i][:,1]
    phi_total_2 = smf[i][:,2]
    phi_total_3 = smf[i][:,3]

    f_blue_cen_0 = fblue_cen[i][:,0]
    f_blue_cen_1 = fblue_cen[i][:,1]
    f_blue_cen_2 = fblue_cen[i][:,2]
    f_blue_cen_3 = fblue_cen[i][:,3]

    f_blue_sat_0 = fblue_sat[i][:,0]
    f_blue_sat_1 = fblue_sat[i][:,1]
    f_blue_sat_2 = fblue_sat[i][:,2]
    f_blue_sat_3 = fblue_sat[i][:,3]

    mstar_red_cen_0 = mean_mstar_red[i][:,0]
    mstar_red_cen_1 = mean_mstar_red[i][:,1]
    mstar_red_cen_2 = mean_mstar_red[i][:,2]
    mstar_red_cen_3 = mean_mstar_red[i][:,3]

    mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
    mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
    mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
    mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

    sigma_red_0 = sigma_red[i][:,0]
    sigma_red_1 = sigma_red[i][:,1]
    sigma_red_2 = sigma_red[i][:,2]
    sigma_red_3 = sigma_red[i][:,3]

    sigma_blue_0 = sigma_blue[i][:,0]
    sigma_blue_1 = sigma_blue[i][:,1]
    sigma_blue_2 = sigma_blue[i][:,2]
    sigma_blue_3 = sigma_blue[i][:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3,
        'sigma_red_0':sigma_red_0, 'sigma_red_1':sigma_red_1, 
        'sigma_red_2':sigma_red_2, 'sigma_red_3':sigma_red_3,
        'sigma_blue_0':sigma_blue_0, 'sigma_blue_1':sigma_blue_1,
        'sigma_blue_2':sigma_blue_2, 'sigma_blue_3':sigma_blue_3})

    if i == 0:
        # Correlation matrix of phi and deltav colour measurements combined
        corr_mat_global = combined_df.corr()
        cov_mat_global = combined_df.cov()

        corr_mat_average = corr_mat_global
        cov_mat_average = cov_mat_global
    else:
        corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
        cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
        

# Using average cov mat to get correlation matrix
corr_mat_average = calc_corr_mat(cov_mat_average)
corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
sigma_average = np.sqrt(np.diag(cov_mat_average))


from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
cmap = cm.get_cmap('Spectral_r')
cax = ax1.matshow(corr_mat_average, cmap=cmap, vmin=-1, vmax=1)
tick_marks = [i for i in range(len(combined_df.columns))]
names = [
r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
r'$sigma\ red_1$', r'$red_2$', r'$red_3$', r'$red_4$',
r'$sigma\ blue_1$', r'$blue_2$', r'$blue_3$', r'$blue_4$']

tick_marks=[0, 4, 8, 12, 16, 20, 24]
names = [
    r"$\boldsymbol\phi$",
    r"$\boldsymbol{f_{blue}^{c}}$",
    r"$\boldsymbol{f_{blue}^{s}}$",
    r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
    r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$",
    r"$\boldsymbol{\sigma_{red}}$",
    r"$\boldsymbol{\sigma_{blue}}$"]

plt.xticks(tick_marks, names, fontsize=10)#, rotation='vertical')
plt.yticks(tick_marks, names, fontsize=10)    
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
plt.colorbar(cax)
plt.title('{0}'.format(quenching))
plt.show()