# Libs
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
import pandas as pd
import numpy as np
import h5py
import os

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

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

def models_add_avgrpcz(df, grpid_col=None, galtype_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

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
        #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
        eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def assign_colour_label_data_legacy(catl):
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

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
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
        max_total, phi_total, err_total, counts_total = \
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
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary_a23.values
            mass_cen_arr = catl.logmbary_a23.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary_a23.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

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

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

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

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
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
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
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
    #* DF of only N > 1 groups sorted by ps_groupid
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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

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
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
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
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
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

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

def get_err_data(path_to_proc):
    # Read in datasets from h5 file and calculate corr matrix
    hf_read = h5py.File(path_to_proc + 'final_matrices/corr_matrices_28stats_{0}_{1}.h5'.
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

    #* In bmf halo case, some mocks had -np.inf in some bin(s). 
    #* Converting to nan so it can be ignored by df.corr/df.cov
    sigma_blue[sigma_blue == -np.inf] = np.nan

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
            # corr_mat_global = combined_df.corr()
            cov_mat_global = combined_df.cov()

            # corr_mat_average = corr_mat_global
            cov_mat_average = cov_mat_global
        else:
            # corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
            cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
            

    # Using average cov mat to get correlation matrix
    corr_mat_average = calc_corr_mat(cov_mat_average)
    corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
    sigma_average = np.sqrt(np.diag(cov_mat_average))

    if pca:
        num_mocks = 64
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_average)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_average.shape[0], corr_mat_average.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_average.shape[0], :corr_mat_average.shape[0]] = diag(s)

        ## values in s are eigenvalues #confirmed by comparing s to 
        ## output of np.linalg.eig(C) where the first array is array of 
        ## eigenvalues.
        ## https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt one more time 
        # to be able to compare directly to values in Sigma since eigenvalues 
        # are squares of singular values 
        # (http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf)
        # https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm#:~:text=The%20SVD%20represents%20an%20expansion,up%20the%20columns%20of%20U.
        min_eigen = np.sqrt(np.sqrt(2/(num_mocks)))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* min_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>min_eigen])
        VT = VT[:n_elements, :]

        print("Number of principle components kept: {0}".format(n_elements))

        ## reconstruct
        # sigma_mat = sigma_mat[:, :n_elements]
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        ## transform 2 ways (this is how you would transform data, model and sigma
        ## i.e. new_data = data.dot(Sigma) etc.) <- not sure this is correct
        #* Both lines below are equivalent transforms of the matrix BUT to project 
        #* data, model, err in the new space, you need the eigenvectors NOT the 
        #* eigenvalues (sigma_mat) as was used earlier. The eigenvectors are 
        #* VT.T (the right singular vectors) since those have the reduced 
        #* dimensions needed for projection. Data and model should then be 
        #* projected similarly to err_colour below. 
        #* err_colour.dot(sigma_mat) no longer makes sense since that is 
        #* multiplying the error with the eigenvalues. Using sigma_mat only
        #* makes sense in U.dot(sigma_mat) since U and sigma_mat were both derived 
        #* by doing svd on the matrix which is what you're trying to get in the 
        #* new space by doing U.dot(sigma_mat). 
        #* http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        err_colour_pca = sigma_average.dot(VT.T)
        eigenvectors = VT.T

    if pca:
        return err_colour_pca, eigenvectors
    else:
        return sigma_average, corr_mat_inv_colour_average

global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca

rseed = 12
np.random.seed(rseed)
level = "group"
stacked_stat = "both"
pca = False

survey = 'eco'
machine = 'mac'
mf_type = 'bmf'
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
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
            "gal_group_eco_bary_buffer_volh1_dr3.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

print('Reading catalog') #No Mstar cut needed as catl_file already has it
catl, volume, z_median = read_data_catl(catl_file, survey)

print('Assigning colour label to data')
catl = assign_colour_label_data(catl)

if mf_type == 'smf':
    print('Measuring SMF for data')
    total_data_ss = measure_all_smf(catl, volume, True)
elif mf_type == 'bmf':
    print('Measuring BMF for data')
    logmbary = catl.logmbary_a23.values
    total_data_bs = diff_bmf(logmbary, volume, False)

print('Measuring blue fraction for data')
if mf_type == 'smf':
    f_blue_data_ss = blue_frac(catl, False, True)
elif mf_type == 'bmf':
    f_blue_data_bs = blue_frac(catl, False, True)

if mf_type == 'smf':
    print('Measuring stacked velocity dispersion for data')
    red_deltav, red_cen_mstar_sigma, blue_deltav, \
        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

    sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
        statistic='std', bins=np.linspace(8.6,10.8,5))
    sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
        statistic='std', bins=np.linspace(8.6,10.8,5))
    
    sigma_red_data_ss = np.log10(sigma_red_data[0])
    sigma_blue_data_ss = np.log10(sigma_blue_data[0])
elif mf_type == 'bmf':
    print('Measuring stacked velocity dispersion for data')
    red_deltav, red_cen_mbary_sigma, blue_deltav, \
        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

    sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
        statistic='std', bins=np.linspace(9.0,11.2,5))
    sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
        statistic='std', bins=np.linspace(9.0,11.2,5))
    
    sigma_red_data_bs = np.log10(sigma_red_data[0])
    sigma_blue_data_bs = np.log10(sigma_blue_data[0])

if mf_type == 'smf':
    print('Measuring velocity dispersion for data')
    red_sigma, red_cen_mstar_sigma, blue_sigma, \
        blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

    red_sigma = np.log10(red_sigma)
    blue_sigma = np.log10(blue_sigma)

    mean_mstar_red_data_ss = bs(red_sigma, red_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.8,5))
    mean_mstar_blue_data_ss = bs(blue_sigma, blue_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.5,5))

elif mf_type == 'bmf':
    print('Measuring velocity dispersion for data')
    red_sigma, red_cen_mbary_sigma, blue_sigma, \
        blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

    red_sigma = np.log10(red_sigma)
    blue_sigma = np.log10(blue_sigma)

    mean_mstar_red_data_bs = bs(red_sigma, red_cen_mbary_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.8,5))
    mean_mstar_blue_data_bs = bs(blue_sigma, blue_cen_mbary_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.5,5))

if mf_type == 'smf':
    sigma_average_ss, mat_ss = get_err_data(path_to_proc)
elif mf_type == 'bmf':
    sigma_average_bs, mat_bs = get_err_data(path_to_proc)

def eckert_mf(norm, M_star, alpha, mass_arr):
    # mass_arr = np.log10((10**mass_arr) / 2.041)
    mass_arr = 10**mass_arr
    M_star = 10**M_star
    norm = norm*10**-3
    result = norm * ((mass_arr / M_star)**(alpha+1))*np.exp(-mass_arr/M_star)
    mass_arr = np.log10(mass_arr/2.041) #h: 0.7 -> 1
    result = np.log10(result/0.343) #h: 0.7 -> 1
    return mass_arr, result

if mf_type == 'smf':
    eckert_smf_x, eckert_smf_y = eckert_mf(5.95, 10.92, -1.19, catl.logmstar.values)
elif mf_type == 'bmf':
    eckert_bmf_x, eckert_bmf_y = eckert_mf(7.48, 10.92, -1.28, catl.logmbary_a23.values)

#* Plots of observables
rc('axes', linewidth=4)

#* Mass functions
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

max_total_ss = total_data_ss[0]
max_total_bs = total_data_bs[0]

e16_smf = ax[0].scatter(eckert_smf_x, eckert_smf_y)
# mt = ax[0].fill_between(x=max_total_ss, y1=total_data_ss[1]+sigma_average_ss[:4], 
#     y2=total_data_ss[1]-sigma_average_ss[:4], color='silver', alpha=0.4)
dt = ax[0].scatter(total_data_ss[0], total_data_ss[1],
    color='k', s=150, zorder=10, marker='^')

# ax[1].fill_between(x=max_total_bs, y1=total_data_bs[1]+sigma_average_bs[:4], 
#     y2=total_data_bs[1]-sigma_average_bs[:4], color='silver', alpha=0.4)
e16_bmf = ax[1].scatter(eckert_bmf_x, eckert_bmf_y)

ax[1].scatter(total_data_bs[0], total_data_bs[1],
    color='k', s=150, zorder=10, marker='^')

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

ax[0].legend([dt,  (e16_smf, e16_bmf)], ['ECO', 'E16'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':30})
ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()
plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_mf_total_bothsamples.pdf', 
    bbox_inches="tight", dpi=1200)

# plt.show()

#* Blue fractions
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

mt_cen = ax[0].fill_between(x=f_blue_data_ss[0], y1=f_blue_data_ss[2]+sigma_average_ss[4:8], 
    y2=f_blue_data_ss[2]-sigma_average_ss[4:8], color='rebeccapurple', alpha=0.4)
mt_sat = ax[0].fill_between(x=f_blue_data_ss[0], y1=f_blue_data_ss[3]+sigma_average_ss[8:12], 
    y2=f_blue_data_ss[3]-sigma_average_ss[8:12], color='goldenrod', alpha=0.4)

dt_cen = ax[0].scatter(f_blue_data_ss[0], f_blue_data_ss[2],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_sat = ax[0].scatter(f_blue_data_ss[0], f_blue_data_ss[3],
    color='goldenrod', s=150, zorder=10, marker='^')

ax[1].fill_between(x=f_blue_data_bs[0], y1=f_blue_data_bs[2]+sigma_average_bs[4:8], 
    y2=f_blue_data_bs[2]-sigma_average_bs[4:8], color='rebeccapurple', alpha=0.4)
ax[1].fill_between(x=f_blue_data_bs[0], y1=f_blue_data_bs[3]+sigma_average_bs[8:12], 
    y2=f_blue_data_bs[3]-sigma_average_bs[8:12], color='goldenrod', alpha=0.4)

ax[1].scatter(f_blue_data_bs[0], f_blue_data_bs[2],
    color='rebeccapurple', s=150, zorder=10, marker='^')
ax[1].scatter(f_blue_data_bs[0], f_blue_data_bs[3],
    color='goldenrod', s=150, zorder=10, marker='^')

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$f_{blue}$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$f_{blue}$', fontsize=30)

ax[0].set_ylim(0,1)
ax[1].set_ylim(0,1)

ax[0].legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':30})

ax[0].minorticks_on()
ax[1].minorticks_on()

plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bothsamples.pdf', 
    bbox_inches="tight", dpi=1200)

# plt.show()

#* sigma-mass
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

bins_red=np.linspace(1,2.8,5)
bins_blue=np.linspace(1,2.5,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

mt_red = ax[0].fill_between(x=bins_red, y1=mean_mstar_red_data_ss[0]+sigma_average_ss[12:16], 
    y2=mean_mstar_red_data_ss[0]-sigma_average_ss[12:16], color='indianred', alpha=0.4)
mt_blue = ax[0].fill_between(x=bins_blue, y1=mean_mstar_blue_data_ss[0]+sigma_average_ss[16:20], 
    y2=mean_mstar_blue_data_ss[0]-sigma_average_ss[16:20], color='cornflowerblue', alpha=0.4)

dt_red = ax[0].scatter(bins_red, mean_mstar_red_data_ss[0], 
    color='indianred', s=150, zorder=10, marker='^')
dt_blue = ax[0].scatter(bins_blue, mean_mstar_blue_data_ss[0],
    color='cornflowerblue', s=150, zorder=10, marker='^')

ax[1].fill_between(x=bins_red, y1=mean_mstar_red_data_bs[0]+sigma_average_bs[12:16], 
    y2=mean_mstar_red_data_bs[0]-sigma_average_bs[12:16], color='indianred', alpha=0.4)
ax[1].fill_between(x=bins_blue, y1=mean_mstar_blue_data_bs[0]+sigma_average_bs[16:20], 
    y2=mean_mstar_blue_data_bs[0]-sigma_average_bs[16:20], color='cornflowerblue', alpha=0.4)

ax[1].scatter(bins_red, mean_mstar_red_data_bs[0], 
    color='indianred', s=150, zorder=10, marker='^')
ax[1].scatter(bins_blue, mean_mstar_blue_data_bs[0],
    color='cornflowerblue', s=150, zorder=10, marker='^')

ax[0].set_xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)
ax[1].set_ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)

ax[0].legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    ['ECO','Mocks'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
    prop={'size':30})

ax[0].set_ylim(9.8,11.05)
ax[1].set_ylim(9.8,11.05)

ax[0].minorticks_on()
ax[1].minorticks_on()

plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_sigma_mass_bothsamples.pdf', 
    bbox_inches="tight", dpi=1200)

# plt.show()

#* mass-sigma
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

bin_min = 8.6
bin_max = 10.8
bins_red=np.linspace(bin_min,bin_max,5)
bins_blue=np.linspace(bin_min,bin_max,5)
bins_red_ss = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue_ss = 0.5 * (bins_blue[1:] + bins_blue[:-1])

bin_min = 9.0
bin_max = 11.2
bins_red=np.linspace(bin_min,bin_max,5)
bins_blue=np.linspace(bin_min,bin_max,5)
bins_red_bs = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue_bs = 0.5 * (bins_blue[1:] + bins_blue[:-1])


mt_red = ax[0].fill_between(x=bins_red_ss, y1=sigma_red_data_ss+sigma_average_ss[20:24], 
    y2=sigma_red_data_ss-sigma_average_ss[20:24], color='indianred', alpha=0.4)
mt_blue = ax[0].fill_between(x=bins_blue_ss, y1=sigma_blue_data_ss+sigma_average_ss[24:28], 
    y2=sigma_blue_data_ss-sigma_average_ss[24:28], color='cornflowerblue', alpha=0.4)

dt_red = ax[0].scatter(bins_red_ss, sigma_red_data_ss, 
    color='indianred', s=150, zorder=10, marker='^')
dt_blue = ax[0].scatter(bins_blue_ss, sigma_blue_data_ss,
    color='cornflowerblue', s=150, zorder=10, marker='^')

ax[1].fill_between(x=bins_red_bs, y1=sigma_red_data_bs+sigma_average_bs[20:24], 
    y2=sigma_red_data_bs-sigma_average_bs[20:24], color='indianred', alpha=0.4)
ax[1].fill_between(x=bins_blue_bs, y1=sigma_blue_data_bs+sigma_average_bs[24:28], 
    y2=sigma_blue_data_bs-sigma_average_bs[24:28], color='cornflowerblue', alpha=0.4)

ax[1].scatter(bins_red_bs, sigma_red_data_bs, 
    color='indianred', s=150, zorder=10, marker='^')
ax[1].scatter(bins_blue_bs, sigma_blue_data_bs,
    color='cornflowerblue', s=150, zorder=10, marker='^')

ax[0].set_ylabel(r'\boldmath$\overline{\log_{10}\ \sigma} \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$\overline{\log_{10}\ \sigma} \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)

ax[0].legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    ['ECO','Mocks'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', 
    prop={'size':30})

ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()

plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_mass_sigma_bothsamples.pdf', 
    bbox_inches="tight", dpi=1200)

# plt.show()





###############################################################################
#Plots of mock measurements and ECO

hf_read = h5py.File(path_to_proc + 'final_matrices/corr_matrices_28stats_{0}_{1}.h5'.
    format(quenching, mf_type), 'r')
hf_read.keys()
if mf_type == 'smf':
    mf_s = hf_read.get('smf')
    mf_s = np.squeeze(np.array(mf_s))
    fblue_cen_s = hf_read.get('fblue_cen')
    fblue_cen_s = np.array(fblue_cen_s)
    fblue_sat_s = hf_read.get('fblue_sat')
    fblue_sat_s = np.array(fblue_sat_s)
    mean_mstar_red_s = hf_read.get('mean_mstar_red')
    mean_mstar_red_s = np.array(mean_mstar_red_s)
    mean_mstar_blue_s = hf_read.get('mean_mstar_blue')
    mean_mstar_blue_s = np.array(mean_mstar_blue_s)
    sigma_red_s = hf_read.get('sigma_red')
    sigma_red_s = np.array(sigma_red_s)
    sigma_blue_s = hf_read.get('sigma_blue')
    sigma_blue_s = np.array(sigma_blue_s)

    #* In bmf halo case, some mocks had -np.inf in some bin(s). 
    #* Converting to nan so it can be ignored by df.corr/df.cov
    sigma_blue_s[sigma_blue_s == -np.inf] = np.nan
elif mf_type == 'bmf':
    mf_b = hf_read.get('smf')
    mf_b = np.squeeze(np.array(mf_b))
    fblue_cen_b = hf_read.get('fblue_cen')
    fblue_cen_b = np.array(fblue_cen_b)
    fblue_sat_b = hf_read.get('fblue_sat')
    fblue_sat_b = np.array(fblue_sat_b)
    mean_mstar_red_b = hf_read.get('mean_mstar_red')
    mean_mstar_red_b = np.array(mean_mstar_red_b)
    mean_mstar_blue_b = hf_read.get('mean_mstar_blue')
    mean_mstar_blue_b = np.array(mean_mstar_blue_b)
    sigma_red_b = hf_read.get('sigma_red')
    sigma_red_b = np.array(sigma_red_b)
    sigma_blue_b = hf_read.get('sigma_blue')
    sigma_blue_b = np.array(sigma_blue_b)

    #* In bmf halo case, some mocks had -np.inf in some bin(s). 
    #* Converting to nan so it can be ignored by df.corr/df.cov
    sigma_blue_b[sigma_blue_b == -np.inf] = np.nan

#* Plots of observables
rc('axes', linewidth=4)

#* Mass functions
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

max_total_ss = total_data_ss[0]
max_total_bs = total_data_bs[0]

for i in range(len(mf_s[0])):
    mt, = ax[0].plot(max_total_ss, mf_s[0][i], alpha=0.4)
dt, = ax[0].plot(total_data_ss[0], total_data_ss[1], color='k', zorder=10, lw=3)

for i in range(len(mf_b[0])):
    mt, = ax[1].plot(max_total_bs, mf_b[0][i], alpha=0.4)
dt, = ax[1].plot(total_data_bs[0], total_data_bs[1], color='k', zorder=10, lw=3)

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

ax[0].legend([dt,  mt], ['ECO', 'Mocks'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':30})
ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()

#* Blue fractions
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

for i in range(len(fblue_cen_s[0])):
    mt_cen, = ax[0].plot(f_blue_data_ss[0], fblue_cen_s[0][i], color='rebeccapurple', alpha=0.4)
dt_cen, = ax[0].plot(f_blue_data_ss[0], f_blue_data_ss[2], color='k', zorder=10, lw=3, ls='-')

for i in range(len(fblue_sat_s[0])):
    mt_sat, = ax[0].plot(f_blue_data_ss[0], fblue_sat_s[0][i], color='goldenrod', alpha=0.4)
dt_sat, = ax[0].plot(f_blue_data_ss[0], f_blue_data_ss[3], color='k', zorder=10, lw=3, ls='--')

for i in range(len(fblue_cen_b[0])):
    mt_cen, = ax[1].plot(f_blue_data_bs[0], fblue_cen_b[0][i], color='rebeccapurple', alpha=0.4)
dt_cen, = ax[1].plot(f_blue_data_bs[0], f_blue_data_bs[2], color='k', zorder=10, lw=3, ls='-')

for i in range(len(fblue_sat_b[0])):
    mt_sat, = ax[1].plot(f_blue_data_bs[0], fblue_sat_b[0][i], color='goldenrod', alpha=0.4)
dt_sat, = ax[1].plot(f_blue_data_bs[0], f_blue_data_bs[3], color='k', zorder=10, lw=3, ls='--')


ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$f_{blue}$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$f_{blue}$', fontsize=30)

ax[0].set_ylim(0,1)
ax[1].set_ylim(0,1)

ax[0].legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':30})

ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()

#* sigma-mass
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

bins_red=np.linspace(1,2.8,5)
bins_blue=np.linspace(1,2.5,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

for i in range(len(mean_mstar_red_s[0])):
    mt_red, = ax[0].plot(bins_red, mean_mstar_red_s[0][i], color='indianred', alpha=0.4)
dt_red, = ax[0].plot(bins_red, mean_mstar_red_data_ss[0], color='k', zorder=10, lw=3, ls='-')

for i in range(len(mean_mstar_blue_s[0])):
    mt_blue, = ax[0].plot(bins_blue, mean_mstar_blue_s[0][i], color='cornflowerblue', alpha=0.4)
dt_blue, = ax[0].plot(bins_blue, mean_mstar_blue_data_ss[0], color='k', zorder=10, lw=3, ls='--')

for i in range(len(mean_mstar_red_b[0])):
    mt_red, = ax[1].plot(bins_red, mean_mstar_red_b[0][i], color='indianred', alpha=0.4)
dt_red, = ax[1].plot(bins_red, mean_mstar_red_data_bs[0], color='k', zorder=10, lw=3, ls='-')

for i in range(len(mean_mstar_blue_b[0])):
    mt_blue, = ax[1].plot(bins_blue, mean_mstar_blue_b[0][i], color='cornflowerblue', alpha=0.4)
dt_blue, = ax[1].plot(bins_blue, mean_mstar_blue_data_bs[0], color='k', zorder=10, lw=3, ls='--')


ax[0].set_xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)
ax[1].set_ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)

ax[0].legend([dt_red, dt_blue, mt_red, mt_blue], 
    ['ECO red', 'ECO blue', 'Mocks red', 'Mocks blue'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
    prop={'size':30})

# ax[0].set_ylim(9.8,11.05)
# ax[1].set_ylim(9.8,11.05)

ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()

#* mass-sigma
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

bin_min = 8.6
bin_max = 10.8
bins_red=np.linspace(bin_min,bin_max,5)
bins_blue=np.linspace(bin_min,bin_max,5)
bins_red_ss = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue_ss = 0.5 * (bins_blue[1:] + bins_blue[:-1])

bin_min = 9.0
bin_max = 11.2
bins_red=np.linspace(bin_min,bin_max,5)
bins_blue=np.linspace(bin_min,bin_max,5)
bins_red_bs = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue_bs = 0.5 * (bins_blue[1:] + bins_blue[:-1])

for i in range(len(sigma_red_s[0])):
    mt_red, = ax[0].plot(bins_red_ss, sigma_red_s[0][i], color='indianred', alpha=0.4)
dt_red, = ax[0].plot(bins_red_ss, sigma_red_data_ss, color='k', zorder=10, lw=3, ls='-')

for i in range(len(sigma_blue_s[0])):
    mt_blue, = ax[0].plot(bins_blue_ss, sigma_blue_s[0][i], color='cornflowerblue', alpha=0.4)
dt_blue, = ax[0].plot(bins_blue_ss, sigma_blue_data_ss, color='k', zorder=10, lw=3, ls='--')

for i in range(len(sigma_red_b[0])):
    mt_red, = ax[1].plot(bins_red_bs, sigma_red_b[0][i], color='indianred', alpha=0.4)
dt_red, = ax[1].plot(bins_red_bs, sigma_red_data_bs, color='k', zorder=10, lw=3, ls='-')

for i in range(len(sigma_blue_b[0])):
    mt_blue, = ax[1].plot(bins_blue_bs, sigma_blue_b[0][i], color='cornflowerblue', alpha=0.4)
dt_blue, = ax[1].plot(bins_blue_bs, sigma_blue_data_bs, color='k', zorder=10, lw=3, ls='--')

ax[0].set_ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)
ax[1].set_ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=30)

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)

ax[0].legend([dt_red, dt_blue, mt_red, mt_blue], 
    ['ECO red', 'ECO blue', 'Mocks red', 'Mocks blue'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', 
    prop={'size':30})

ax[0].minorticks_on()
ax[1].minorticks_on()
plt.show()
