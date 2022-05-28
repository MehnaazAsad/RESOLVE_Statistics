

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from scipy.stats import norm
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import colors
from matplotlib import ticker

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=7)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=1, labelsize=5)
rc('xtick.major', width=2, size=4)
rc('ytick.major', width=2, size=4)
rc('xtick.minor', width=2, size=4)
rc('ytick.minor', width=2, size=4)

__author__ = '[Mehnaaz Asad]'

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def average_of_log(arr):
    result = np.log10(np.nanmean(10**(arr)))
    return result

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
        eco_buff = mock_add_grpcz(eco_buff, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_new.values) / (3 * 10**5)
        
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
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]

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
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
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

    return maxis, phi, err_tot, bins, counts

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

    if data_bool:
        mstar_total_arr = catl.logmstar.values
        censat_col = 'g_galtype'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values
    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:
        mstar_total_arr = catl.logmstar.values
        if level == 'group':
            censat_col = 'g_galtype'
        elif level == 'halo':
            #* To test fblue of centrals with halo centrals instead of group centrals
            #* to see how group finder misclassification affects the measurement
            censat_col = 'cs_flag'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values           
    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mstar_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mstar_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mstar_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mstar_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mstar_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mstar_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mstar_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mstar_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_total_arr = np.log10((10**mstar_total_arr) / 2.041)
        logmstar_cen_arr = np.log10((10**mstar_cen_arr) / 2.041)
        logmstar_sat_arr = np.log10((10**mstar_sat_arr) / 2.041)
    else:
        logmstar_total_arr = mstar_total_arr
        logmstar_cen_arr = mstar_cen_arr
        logmstar_sat_arr = mstar_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5

        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmstar_total_arr, colour_label_total_arr, blue_frac_helper, bins=bins)
    result_cen = bs(logmstar_cen_arr, colour_label_cen_arr, blue_frac_helper, bins=bins)
    result_sat = bs(logmstar_sat_arr, colour_label_sat_arr, blue_frac_helper, bins=bins)
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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            catl = catl.loc[catl.logmstar >= 8.9]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

    if catl_type == 'data' or catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        logmstar_col = 'logmstar'
        ## Use group level for data even when settings.level == halo
        if catl_type == 'data' or level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        ## No halo level in data
        if catl_type == 'mock':
            if level == 'halo':
                galtype_col = 'cs_flag'
                ## Halo ID is equivalent to halo_hostid in vishnu mock
                id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            mstar_limit = 8.9
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    # start = time.time()
    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    # red_subset_ids = [key for key in Counter(
    #     red_subset_df.groupid).keys() if Counter(
    #         red_subset_df.groupid)[key] > 1]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['deltav'].\
        apply(lambda x: gapper(x)).values
      
    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['deltav'].\
        apply(lambda x: gapper(x)).values

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            catl = catl.loc[catl.logmstar >= 8.9]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

    if catl_type == 'data' or catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        logmstar_col = 'logmstar'
        ## Use group level for data even when settings.level == halo
        if catl_type == 'data' or level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        ## No halo level in data
        if catl_type == 'mock':
            if level == 'halo':
                galtype_col = 'cs_flag'
                ## Halo ID is equivalent to halo_hostid in vishnu mock
                id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            mstar_limit = 8.9
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

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
    red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()
    red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
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
    blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()
    blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)


    return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
        blue_cen_stellar_mass_arr

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

global survey
global quenching
global mf_type
global stacked_stat
global level

survey = 'eco'
quenching = 'hybrid'
mf_type = 'smf'
level = 'group'
stacked_stat = False

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_data = dict_of_paths['data_dir']
path_to_mocks = path_to_data + 'mocks/m200b/eco/'
path_to_proc = dict_of_paths['proc_dir']

catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"

print('Reading catalog') #No Mstar cut needed as catl_file already has it
catl, volume, z_median = read_data_catl(catl_file, survey)

print('Assigning colour to data using u-r colours')
catl = assign_colour_label_data(catl)

print('Measuring SMF for data')
total_data = measure_all_smf(catl, volume, True)

print('Measuring blue fraction for data')
f_blue_data = blue_frac(catl, False, True)

if stacked_stat:
    print('Measuring stacked velocity dispersion for data')
    red_deltav, red_cen_mstar_sigma, blue_deltav, \
        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

    sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
        statistic='std', bins=np.linspace(8.6,11,5))
    sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
        statistic='std', bins=np.linspace(8.6,11,5))
    
    sigma_red_data = np.log10(sigma_red_data[0])
    sigma_blue_data = np.log10(sigma_blue_data[0])

    data_observables = np.array(
        [total_data[1][0], total_data[1][1], total_data[1][2], total_data[1][3],
        f_blue_data[2][0], f_blue_data[2][1], f_blue_data[2][2], f_blue_data[2][3],
        f_blue_data[3][0], f_blue_data[3][0], f_blue_data[3][0], f_blue_data[3][0],
        sigma_red_data[0], sigma_red_data[1], 
        sigma_red_data[2], sigma_red_data[3],
        sigma_blue_data[0], sigma_blue_data[1], 
        sigma_blue_data[2], sigma_blue_data[3]])

else:
    print('Measuring velocity dispersion for data')
    red_sigma, red_cen_mstar_sigma, blue_sigma, \
        blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

    red_sigma = np.log10(red_sigma)
    blue_sigma = np.log10(blue_sigma)

    mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(-2,3,5))
    mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(-1,3,5))

    data_observables = np.array(
        [total_data[1][0], total_data[1][1], total_data[1][2], total_data[1][3],
        f_blue_data[2][0], f_blue_data[2][1], f_blue_data[2][2], f_blue_data[2][3],
        f_blue_data[3][0], f_blue_data[3][0], f_blue_data[3][0], f_blue_data[3][0],
        mean_mstar_red_data[0][0], mean_mstar_red_data[0][1], 
        mean_mstar_red_data[0][2], mean_mstar_red_data[0][3],
        mean_mstar_blue_data[0][0], mean_mstar_blue_data[0][1], 
        mean_mstar_blue_data[0][2], mean_mstar_blue_data[0][3]])

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

phi_total_arr = []
# phi_arr_red = []
# phi_arr_blue = []
f_blue_cen_arr = []
f_blue_sat_arr = []
mean_mstar_red_arr = []
mean_mstar_blue_arr = []
box_id_arr = np.linspace(5001,5008,8)
for box in box_id_arr:
    box = int(box)
    temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
        mock_name) 
    for num in range(num_mocks):
        filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        print('Box {0} : Mock {1}'.format(box, num))
        mock_pd = reading_catls(filename) 
        mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer
        mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
            (mock_pd.grpcz_new.values <= max_cz) &\
            (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
        
        # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
        # Mstar_q = 10.39 # Msun/h
        # Mh_q = 14.85 # Msun/h
        # mu = 0.65
        # nu = 0.16

        # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
        # ## for hybrid quenching model
        # Mstar_q = 10.49 # Msun/h
        # Mh_q = 14.03 # Msun/h
        # mu = 0.69
        # nu = 0.148

        # ## Using best-fit found for new ECO data using result from chain 42
        # ## i.e. hybrid quenching model
        # bf_from_last_chain = [10.1679343, 13.10135398, 0.81869216, 0.13844437]

        ## Using best-fit found for new ECO data using result from chain 50
        ## i.e. hybrid quenching model
        bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]
        
        Mstar_q = bf_from_last_chain[0] # Msun/h**2
        Mh_q = bf_from_last_chain[1] # Msun/h
        mu = bf_from_last_chain[2]
        nu = bf_from_last_chain[3]

        # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
        # ## for halo quenching model
        # Mh_qc = 12.61 # Msun/h
        # Mh_qs = 13.5 # Msun/h
        # mu_c = 0.40
        # mu_s = 0.148

        # ## Using best-fit found for new ECO data using result from chain 41
        # ## i.e. halo quenching model
        # bf_from_last_chain = [11.86645536, 12.54502723, 1.42736618, 0.5261119]

        ## Using best-fit found for new ECO data using result from chain 49
        ## i.e. halo quenching model
        bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

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
        # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
        # logmstar_red_max_arr.append(logmstar_red_max)
        # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
        # logmstar_blue_max_arr.append(logmstar_blue_max)
        logmstar_arr = mock_pd.logmstar.values
        # mhi_arr = mock_pd.mhi.values
        # logmgas_arr = np.log10(1.4 * mhi_arr)
        # logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
        # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

        # max_total, phi_total, err_total, bins_total, counts_total = \
        #     diff_bmf(logmbary_arr, volume, False)
        # phi_total_arr.append(phi_total)

        #Measure SMF of mock using diff_smf function
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(logmstar_arr, volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
        #     volume, False, 'B')
        phi_total_arr.append(phi_total)
        # phi_arr_red.append(phi_red)
        # phi_arr_blue.append(phi_blue)


        #Measure blue fraction of galaxies
        f_blue = blue_frac(mock_pd, False, False)
        f_blue_cen_arr.append(f_blue[2])
        f_blue_sat_arr.append(f_blue[3])

        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    mock_pd, 'mock')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])

            mean_mstar_red_arr.append(sigma_red)
            mean_mstar_blue_arr.append(sigma_blue)

        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-2,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))

            mean_mstar_red_arr.append(mean_mstar_red[0])
            mean_mstar_blue_arr.append(mean_mstar_blue[0])


phi_arr_total = np.array(phi_total_arr)
# phi_arr_red = np.array(phi_arr_red)
# phi_arr_blue = np.array(phi_arr_blue)
f_blue_cen_arr = np.array(f_blue_cen_arr)
f_blue_sat_arr = np.array(f_blue_sat_arr)

mean_mstar_red_arr = np.array(mean_mstar_red_arr)
mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

phi_total_0 = phi_arr_total[:,0]
phi_total_1 = phi_arr_total[:,1]
phi_total_2 = phi_arr_total[:,2]
phi_total_3 = phi_arr_total[:,3]

f_blue_cen_0 = f_blue_cen_arr[:,0]
f_blue_cen_1 = f_blue_cen_arr[:,1]
f_blue_cen_2 = f_blue_cen_arr[:,2]
f_blue_cen_3 = f_blue_cen_arr[:,3]

f_blue_sat_0 = f_blue_sat_arr[:,0]
f_blue_sat_1 = f_blue_sat_arr[:,1]
f_blue_sat_2 = f_blue_sat_arr[:,2]
f_blue_sat_3 = f_blue_sat_arr[:,3]

mstar_red_cen_0 = mean_mstar_red_arr[:,0]
mstar_red_cen_1 = mean_mstar_red_arr[:,1]
mstar_red_cen_2 = mean_mstar_red_arr[:,2]
mstar_red_cen_3 = mean_mstar_red_arr[:,3]

mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

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


# Correlation matrix of phi and deltav colour measurements combined
corr_mat_colour = combined_df.corr()
corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
err_colour = np.sqrt(np.diag(combined_df.cov()))

#Flip the rows so that the diagonal matches the diagonal in the plot
corr_mat_colour = corr_mat_colour.iloc[::-1]

fig, ax = plt.subplots(20, 20, figsize=(13.5,13.5), 
    gridspec_kw={'wspace':0, 'hspace':0})

color_norm = colors.Normalize(vmin=-1, vmax=1)
mapper = cm.ScalarMappable(norm=color_norm, cmap=cm.Spectral_r)
    
for i in range(20):
    for j in range(20):
        if i+j!=19:
            #i+1 because otherwise the first minus value would be -0 which
            #isn't how you access the last column. It has to be the last
            #because we have to plot backwards for the y values i.e. subplot[0][0]
            #is the top left cell of the matrix which has x=first bin of phi and 
            #y=last bin of last observable.
            ax[i][j].scatter(combined_df.iloc[:,j], combined_df.iloc[:,-(i+1)], 
                c='k', s=2)
            ## Don't plot white squares for cells where mock measurements are 
            ## very different from ECO
            if i not in [8, 9, 10] and j not in [9, 10, 11]:
                ax[i][j].scatter(data_observables[j], data_observables[-(i+1)], 
                    marker='+', c='w', s=60, lw=2)
        else:
            #Filtering out nans before fit
            (mu, sigma) = norm.fit(combined_df.iloc[:,j][~combined_df.iloc[:,j].
            isnull()])
            n, bins, patches = ax[i][j].hist(combined_df.iloc[:,j][~combined_df.
            iloc[:,j].isnull()], histtype='step', color='k')
            #Plotting gaussian fit to histogram
            ax01 = ax[i][j].twinx() 
            ax01.plot(bins, norm.pdf(bins, mu, sigma), 'w-', lw=2)
            if i not in [8, 9, 10] and j not in [9, 10, 11]:
                ax01.scatter(data_observables[j], 
                    norm.pdf(data_observables[j], mu, sigma), marker='+', 
                    c='w', s=60, lw=2)
            ax01.tick_params(left=False, labelleft=False, top=False, 
                labeltop=False, right=False, labelright=False, bottom=False, 
                labelbottom=False)
            ax01.spines['top'].set_linewidth(0)
            ax01.spines['bottom'].set_linewidth(0)
            ax01.spines['left'].set_linewidth(0)
            ax01.spines['right'].set_linewidth(0)
        ax[i][j].set_facecolor(mapper.to_rgba(corr_mat_colour.values[i][j]))
        ax[i][j].tick_params(
            axis='both',        # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=False,       # ticks along the bottom edge are off
            top=False,          # ticks along the top edge are off
            left=False,         # ticks along the left edge are off
            right=False,        # ticks along the right edge are off
            labelleft=False,    # labels along the left edge are off
            labeltop=False,     # labels along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelright=False)   # labels along the right edge are off
        if i > 0:
            ax[i][j].spines['top'].set_linewidth(0)
        if j > 0:
            ax[i][j].spines['left'].set_linewidth(0)
        if i >= 0 and i < 19:
            ax[i][j].spines['bottom'].set_linewidth(0)
        if j >= 0 and j < 19:
            ax[i][j].spines['right'].set_linewidth(0)

#Dark box around SMF observable
ax[19][0].spines["bottom"].set_linewidth(4)
ax[19][1].spines["bottom"].set_linewidth(4)
ax[19][2].spines["bottom"].set_linewidth(4)
ax[19][3].spines["bottom"].set_linewidth(4)
ax[19][0].spines["left"].set_linewidth(4)
ax[18][0].spines["left"].set_linewidth(4)
ax[17][0].spines["left"].set_linewidth(4)
ax[16][0].spines["left"].set_linewidth(4)
ax[16][0].spines["top"].set_linewidth(4)
ax[16][1].spines["top"].set_linewidth(4)
ax[16][2].spines["top"].set_linewidth(4)
ax[16][3].spines["top"].set_linewidth(4)
ax[16][4].spines["left"].set_linewidth(4)
ax[17][4].spines["left"].set_linewidth(4)
ax[18][4].spines["left"].set_linewidth(4)
ax[19][4].spines["left"].set_linewidth(4)

#Dark box around blue fraction observable
ax[16][4].spines["top"].set_linewidth(4)
ax[16][5].spines["top"].set_linewidth(4)
ax[16][6].spines["top"].set_linewidth(4)
ax[16][7].spines["top"].set_linewidth(4)
ax[16][8].spines["top"].set_linewidth(4)
ax[16][9].spines["top"].set_linewidth(4)
ax[16][10].spines["top"].set_linewidth(4)
ax[16][11].spines["top"].set_linewidth(4)
ax[15][4].spines["left"].set_linewidth(4)
ax[14][4].spines["left"].set_linewidth(4)
ax[13][4].spines["left"].set_linewidth(4)
ax[12][4].spines["left"].set_linewidth(4)
ax[11][4].spines["left"].set_linewidth(4)
ax[10][4].spines["left"].set_linewidth(4)
ax[9][4].spines["left"].set_linewidth(4)
ax[8][4].spines["left"].set_linewidth(4)
ax[8][4].spines["top"].set_linewidth(4)
ax[8][5].spines["top"].set_linewidth(4)
ax[8][6].spines["top"].set_linewidth(4)
ax[8][7].spines["top"].set_linewidth(4)
ax[8][8].spines["top"].set_linewidth(4)
ax[8][9].spines["top"].set_linewidth(4)
ax[8][10].spines["top"].set_linewidth(4)
ax[8][11].spines["top"].set_linewidth(4)
ax[8][12].spines["left"].set_linewidth(4)
ax[9][12].spines["left"].set_linewidth(4)
ax[10][12].spines["left"].set_linewidth(4)
ax[11][12].spines["left"].set_linewidth(4)
ax[12][12].spines["left"].set_linewidth(4)
ax[13][12].spines["left"].set_linewidth(4)
ax[14][12].spines["left"].set_linewidth(4)
ax[15][12].spines["left"].set_linewidth(4)

#Dark box around velocity dispersion observable
ax[8][12].spines["top"].set_linewidth(4)
ax[8][13].spines["top"].set_linewidth(4)
ax[8][14].spines["top"].set_linewidth(4)
ax[8][15].spines["top"].set_linewidth(4)
ax[8][16].spines["top"].set_linewidth(4)
ax[8][17].spines["top"].set_linewidth(4)
ax[8][18].spines["top"].set_linewidth(4)
ax[8][19].spines["top"].set_linewidth(4)
ax[7][12].spines["left"].set_linewidth(4)
ax[6][12].spines["left"].set_linewidth(4)
ax[5][12].spines["left"].set_linewidth(4)
ax[4][12].spines["left"].set_linewidth(4)
ax[3][12].spines["left"].set_linewidth(4)
ax[2][12].spines["left"].set_linewidth(4)
ax[1][12].spines["left"].set_linewidth(4)
ax[0][12].spines["left"].set_linewidth(4)
ax[0][12].spines["top"].set_linewidth(4)
ax[0][13].spines["top"].set_linewidth(4)
ax[0][14].spines["top"].set_linewidth(4)
ax[0][15].spines["top"].set_linewidth(4)
ax[0][16].spines["top"].set_linewidth(4)
ax[0][17].spines["top"].set_linewidth(4)
ax[0][18].spines["top"].set_linewidth(4)
ax[0][19].spines["top"].set_linewidth(4)
ax[0][19].spines["right"].set_linewidth(4)
ax[1][19].spines["right"].set_linewidth(4)
ax[2][19].spines["right"].set_linewidth(4)
ax[3][19].spines["right"].set_linewidth(4)
ax[4][19].spines["right"].set_linewidth(4)
ax[5][19].spines["right"].set_linewidth(4)
ax[6][19].spines["right"].set_linewidth(4)
ax[7][19].spines["right"].set_linewidth(4)

# for ax in ax.flat:
#     ax.label_outer()

cax = plt.axes([0.27, 0.93, 0.5, 0.04])
cbar = plt.colorbar(mapper, cax=cax, orientation="horizontal")
tick_locator = ticker.MaxNLocator(nbins=10)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params(labelsize=17)

# Horizontal axis tick labels
plt.annotate("8.6", (-1.63, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("9.25", (-1.5, -43.5), fontsize=10, annotation_clip=False)
plt.annotate("9.9", (-1.32, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("10.55", (-1.21, -43.5), fontsize=10, annotation_clip=False)
# plt.annotate("11.2", (-1.06, -43.5), fontsize=12, annotation_clip=False)

plt.annotate("8.6", (-0.95, -43.5), fontsize=12, annotation_clip=False)
plt.annotate("9.9", (-0.69, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("11.2", (-0.44, -43.5), fontsize=12, annotation_clip=False)

plt.annotate("8.6", (-0.33, -43.5), fontsize=12, annotation_clip=False)
plt.annotate("9.9", (-0.08, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("11.2", (0.18, -43.5), fontsize=12, annotation_clip=False)

plt.annotate("-2.0", (0.29, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("-0.75", (0.38, -43.5), fontsize=12, annotation_clip=False)
plt.annotate("0.5", (0.55, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("1.75", (0.64, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("3.0", (0.82, -43.5), fontsize=12, annotation_clip=False)

plt.annotate("-1", (0.91, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("-0.75", (0.38, -43.5), fontsize=12, annotation_clip=False)
plt.annotate("1.0", (1.18, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("1.75", (0.64, -43.5), fontsize=12, annotation_clip=False)
plt.annotate("3.0", (1.45, -43.5), fontsize=12, annotation_clip=False)

# plt.annotate("9.25", (-1.5, -43.5), fontsize=10, annotation_clip=False)
# plt.annotate("9.9", (-0.71, -43.5), fontsize=12, annotation_clip=False)
# # plt.annotate("10.55", (-1.21, -43.5), fontsize=10, annotation_clip=False)
# plt.annotate("11.2", (-0.41, -43.5), fontsize=12, annotation_clip=False)

# # plt.annotate("9.25", (-0.88, -43.5), fontsize=12, annotation_clip=False)
# # plt.annotate("10.55", (-0.58, -43.5), fontsize=12, annotation_clip=False)

# plt.annotate("8.6", (-0.36, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("9.9", (-0.22, -43.5), fontsize=12, annotation_clip=False)
# # plt.annotate("10.55", (0.07, -43.5), fontsize=12, annotation_clip=False)

# plt.annotate("11.2", (-1.03, -43.5), fontsize=12, annotation_clip=False)


# plt.annotate("-2", (0.25, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("-0.75", (0.38, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("0.5", (0.51, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("1.75", (0.64, -43.5), fontsize=12, annotation_clip=False)
# plt.annotate("3", (0.75, -43.5), fontsize=12, annotation_clip=False)

# Vertical axis tick labels
plt.annotate("8.6", (-1.73, -42.3), fontsize=12, annotation_clip=False)
plt.annotate("9.9", (-1.73, -38.3), fontsize=12, annotation_clip=False)
# plt.annotate("11.2", (-1.73, -35.0), fontsize=12, annotation_clip=False)

plt.annotate("8.6", (-1.73, -34.4), fontsize=12, annotation_clip=False)
plt.annotate("9.9", (-1.73, -30.8), fontsize=12, annotation_clip=False)
# plt.annotate("11.2", (-1.73, -27.3), fontsize=12, annotation_clip=False)

plt.annotate("8.6", (-1.73, -26.6), fontsize=12, annotation_clip=False)
plt.annotate("9.9", (-1.73, -22.9), fontsize=12, annotation_clip=False)
# plt.annotate("11.2", (-1.73, -19.5), fontsize=12, annotation_clip=False)

plt.annotate("-2.0", (-1.73, -18.8), fontsize=12, annotation_clip=False)
plt.annotate("0.5", (-1.73, -15.2), fontsize=12, annotation_clip=False)
# plt.annotate("3.0", (-1.73, -11.8), fontsize=12, annotation_clip=False)

plt.annotate("-1.0", (-1.75, -11.1), fontsize=12, annotation_clip=False)
plt.annotate("1.0", (-1.73, -7.6), fontsize=12, annotation_clip=False)
plt.annotate("3.0", (-1.73, -3.9), fontsize=12, annotation_clip=False)

## Horizontal axis labels
plt.annotate(r"$\boldsymbol\phi$", (-1.31, -44.8), fontsize=20, 
    annotation_clip=False)
plt.annotate("", xy=(0.12, 0.06), xytext=(0.19, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2))
plt.annotate("", xy=(0.21, 0.06), xytext=(0.28, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

plt.annotate(r"$\boldsymbol{f_{blue}^{c}}$", (-0.7, -44.8), fontsize=20, 
    annotation_clip=False)
plt.annotate("", xy=(0.28, 0.06), xytext=(0.34, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2)) 
plt.annotate("", xy=(0.37, 0.06), xytext=(0.435, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

plt.annotate(r"$\boldsymbol{f_{blue}^{s}}$", (-0.05, -44.8), fontsize=20, 
    annotation_clip=False)
plt.annotate("", xy=(0.436, 0.06), xytext=(0.505, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2))  
plt.annotate("", xy=(0.53, 0.06), xytext=(0.593, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

# plt.annotate(r"$\boldsymbol{\sigma_{red}}$", (0.55, -44.8), fontsize=20, 
#     annotation_clip=False)
plt.annotate(r"$\boldsymbol{\overline{M_{*,red}^{c}}}$", (0.55, -44.8), 
    fontsize=20, annotation_clip=False)

plt.annotate("", xy=(0.594, 0.06), xytext=(0.65, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2))
plt.annotate("", xy=(0.72, 0.06), xytext=(0.748, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))
    
# plt.annotate(r"$\boldsymbol{\sigma_{blue}}$", (1.14, -44.8), fontsize=20, 
#     annotation_clip=False)
plt.annotate(r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$", (1.14, -44.8), 
    fontsize=20, annotation_clip=False)

plt.annotate("", xy=(0.75, 0.06), xytext=(0.80, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2))
plt.annotate("", xy=(0.875, 0.06), xytext=(0.90, 0.06), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

# Vertical axis  labels
plt.annotate(r"$\boldsymbol\phi$", (-1.85, -38.5), fontsize=20, rotation=90,
    annotation_clip=False)
plt.annotate("", xy=(0.065, 0.11), xytext=(0.065, 0.171), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2))
plt.annotate("", xy=(0.065, 0.197), xytext=(0.065, 0.264), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

plt.annotate(r"$\boldsymbol{f_{blue}^{c}}$", (-1.85, -30.5), fontsize=20, 
    rotation=90, annotation_clip=False)
plt.annotate("", xy=(0.065, 0.266), xytext=(0.065, 0.33), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2)) 
plt.annotate("", xy=(0.065, 0.36), xytext=(0.065, 0.42), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

plt.annotate(r"$\boldsymbol{f_{blue}^{s}}$", (-1.85, -22.5), fontsize=20, 
    rotation=90, annotation_clip=False)
plt.annotate("", xy=(0.065, 0.422), xytext=(0.065, 0.49), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2)) 
plt.annotate("", xy=(0.065, 0.52), xytext=(0.065, 0.575), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

# plt.annotate(r"$\boldsymbol{\sigma_{red}}$", (-1.85, -15.6), fontsize=20, 
#     rotation=90, annotation_clip=False)
plt.annotate(r"$\boldsymbol{\overline{M_{*,red}^{c}}}$", (-1.85, -15.6), 
    fontsize=20, rotation=90, annotation_clip=False)

plt.annotate("", xy=(0.065, 0.577), xytext=(0.065, 0.63), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2)) 
plt.annotate("", xy=(0.065, 0.695), xytext=(0.065, 0.728), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

# plt.annotate(r"$\boldsymbol{\sigma_{blue}}$", (-1.85, -8.1), fontsize=20, 
#     rotation=90, annotation_clip=False)
plt.annotate(r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$", (-1.85, -8.1), fontsize=20, 
    rotation=90, annotation_clip=False)
plt.annotate("", xy=(0.065, 0.73), xytext=(0.065, 0.78), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="->", facecolor='k', linewidth=2)) 
plt.annotate("", xy=(0.065, 0.85), xytext=(0.065, 0.88), 
    xycoords="figure fraction", textcoords="figure fraction", 
    arrowprops=dict(arrowstyle="<-", facecolor='k', linewidth=2))

plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/matrix_hybrid.pdf')

plt.show()