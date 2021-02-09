"""
{This script finds the best-fit hybrid quenching model parameters for the ECO
 data (in h=1.0) so that they can be applied to the mocks when measuring 
 error in data}
"""

from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math
import os

__author__ = '{Mehnaaz Asad}'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=20)
rc('text', usetex=True)
plt.rcParams['legend.title_fontsize'] = 'xx-small'
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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
    ## Calculate std from mean=0

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
        # logmstar_col = '{0}'.format(randint_logmstar)
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
   
    red_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'R') & (catl.fc == 1)].values)  
    blue_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'B') & (catl.fc == 1)].values)


    # Calculating spread in velocity dispersion for galaxies in groups with a 
    # red central

    red_deltav_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.grp == key]
        cen_stellar_mass = group.logmstar.loc[group.fc.\
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
        group = catl.loc[catl.grp == key]
        cen_stellar_mass = group.logmstar.loc[group.fc\
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

    data = data.flatten() # from (4,5) to (1,20)
    model = model.flatten() # same as above
    print("data: " , data , "\n")
    print("model: " , model , "\n")
    print("data error: " , err_data , "\n")
    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def lnprob(theta, phi_red_data, phi_blue_data, std_red_data, std_blue_data, 
    err, corr_mat_inv, gals_df):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    
    if quenching == 'hybrid':
        f_red_cen, f_red_sat, cen_mstar, sat_hosthalom, sat_mstar = \
            hybrid_quenching_model(theta, gals_df)
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta, gals_df)
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
    v_survey = volume
    total_model, red_model, blue_model = measure_all_smf(gals_df, v_survey 
    , True)     
    std_red_model, centers_red_model,  std_blue_model, centers_blue_model = \
        get_deltav_sigma_data(gals_df)
    data_arr = []
    data_arr.append(phi_red_data)
    data_arr.append(phi_blue_data)
    data_arr.append(std_red_data)
    data_arr.append(std_blue_data)
    model_arr = []
    model_arr.append(red_model[1])
    model_arr.append(blue_model[1])   
    model_arr.append(std_red_model)
    model_arr.append(std_blue_model)
    err_arr = err

    data_arr, model_arr = np.array(data_arr), np.array(model_arr)
    chi2 = chi_squared(data_arr, model_arr, err_arr, corr_mat_inv)
    
    return chi2

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

    f_red_cen = 1 - np.exp(-((10**cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((10**sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((10**sat_hosthalo_mass_arr/(10**Mh_q))**nu))
    f_red_sat = 1 - (g_Mstar * h_Mh)

    return f_red_cen, f_red_sat, cen_stellar_mass_arr, sat_hosthalo_mass_arr, sat_stellar_mass_arr

def halo_quenching_model(theta, gals_df):
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

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df)

    f_red_cen = 1 - np.exp(-(((10**cen_hosthalo_mass_arr)/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-(((10**sat_hosthalo_mass_arr)/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def get_deltav_sigma_mocks_urcolour(survey, mock_df):
    """
    Calculate spread in velocity dispersion from survey mocks (logmstar converted
    to h=1 units before analysis)

    Parameters
    ----------
    survey: string
        Name of survey
    mock_df: Pandas DataFrame
        Mock catalog

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

def get_err_data_urcolour(survey, path):
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
    deltav_sig_arr_red = []
    deltav_sig_arr_blue = []
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

            deltav_sig_red, deltav_sig_blue, deltav_sig_cen_red, \
                deltav_sig_cen_blue = get_deltav_sigma_mocks_urcolour(survey, 
                mock_pd)
            deltav_sig_arr_red.append(deltav_sig_red)
            deltav_sig_arr_blue.append(deltav_sig_blue)

    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    deltav_sig_arr_red = np.array(deltav_sig_arr_red)
    deltav_sig_arr_blue = np.array(deltav_sig_arr_blue)

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

    dv_red_0 = deltav_sig_arr_red[:,0]
    dv_red_1 = deltav_sig_arr_red[:,1]
    dv_red_2 = deltav_sig_arr_red[:,2]
    dv_red_3 = deltav_sig_arr_red[:,3]
    dv_red_4 = deltav_sig_arr_red[:,4]

    dv_blue_0 = deltav_sig_arr_blue[:,0]
    dv_blue_1 = deltav_sig_arr_blue[:,1]
    dv_blue_2 = deltav_sig_arr_blue[:,2]
    dv_blue_3 = deltav_sig_arr_blue[:,3]
    dv_blue_4 = deltav_sig_arr_blue[:,4]

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

    return err_colour, corr_mat_inv_colour

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
    df.loc[df['fc'] == 1, 'f_red'] = f_red_cen
    df.loc[df['fc'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['fc']):
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

def get_host_halo_mock(catl):
    """
    Get host halo mass from mock catalog

    Parameters
    ----------
    catl: pandas dataframe
        Data catalog

    Returns
    ---------
    cen_halos: array
        Array of central host halo masses
    sat_halos: array
        Array of satellite host halo masses
    """

    df = catl.copy()

    cen_halos = []
    sat_halos = []
    for index, value in enumerate(df.fc):
        if value == 1:
            cen_halos.append(df.logmh_s.values[index])
        else:
            sat_halos.append(df.logmh_s.values[index])

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(catl):
    """
    Get stellar mass from mock catalog

    Parameters
    ----------
    catl: pandas dataframe
        Data catalog

    Returns
    ---------
    cen_gals: array
        Array of central stellar masses
    sat_gals: array
        Array of satellite stellar masses
    """

    df = catl.copy()

    cen_gals = []
    sat_gals = []
    for idx,value in enumerate(df.fc):
        if value == 1:
            cen_gals.append(df.logmstar.values[idx])
        elif value == 0:
            sat_gals.append(df.logmstar.values[idx])

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals


dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_data = dict_of_paths['data_dir']

global volume
global quenching

survey = 'eco'
mf_type = 'smf'
quenching = 'halo'

catl_file = path_to_proc + "gal_group_eco_data.hdf5"
path_to_mocks = path_to_data + 'mocks/m200b/eco/'
catl, volume, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)
# Measurements in h=1.0
total_data, red_data, blue_data = measure_all_smf(catl, volume, True)
# Masses in h=1.0
sigma_red, cen_red, sigma_blue, cen_blue = get_deltav_sigma_data(catl)
# SMF measurements and masses in h=1.0 before matrix and error calculations
err_data_colour, corr_mat_colour_inv = get_err_data_urcolour(survey, 
    path_to_mocks)

Mstar_q = 10.5 # Msun/h
Mh_q = 13.76 # Msun/h
mu = 0.69
nu = 0.15
x0 = [Mstar_q, Mh_q, mu, nu]

Mh_qc = 12.20 # Msun/h
Mh_qs = 12.17 # Msun/h
mu_c = 0.38
mu_s = 0.15
x0 = [Mh_qc, Mh_qs, mu_c, mu_s]


catl, volume, z_median = read_data_catl(catl_file, survey)


res = minimize(lnprob, x0, args=(red_data[1], blue_data[1], sigma_red, 
    sigma_blue, err_data_colour, corr_mat_colour_inv, catl), 
    method='nelder-mead', options={'maxiter':20, 'disp': True})

best_fit = res.x

## After running minimize for 10, 20, 40 and 80 iterations
catl = catl.loc[catl.logmstar.values >= 8.9]
f_red_cen_fid, f_red_sat_fid, mstar_cen_fid, sat_hosthalo_fid, mstar_sat_fid = \
    hybrid_quenching_model(x0, catl)
f_red_cen, f_red_sat, mstar_cen, sat_hosthalo, mstar_sat = \
    hybrid_quenching_model(best_fit, catl)

f_red_cen_fid, f_red_sat_fid = halo_quenching_model(x0, catl)
f_red_cen, f_red_sat = halo_quenching_model(best_fit, catl)

gals_df_fid = assign_colour_label_mock(f_red_cen_fid, f_red_sat_fid, catl)
v_survey = volume
total_model_fid, red_model_fid, blue_model_fid = measure_all_smf(gals_df_fid, 
    v_survey, True)     
std_red_model_fid, centers_red_model_fid, std_blue_model_fid, \
    centers_blue_model_fid = get_deltav_sigma_data(gals_df_fid)

gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, catl)
v_survey = volume
total_model, red_model, blue_model = measure_all_smf(gals_df, v_survey, True)     
std_red_model, centers_red_model,  std_blue_model, centers_blue_model = \
    get_deltav_sigma_data(gals_df)

## Red fraction of centrals
plt.scatter(mstar_cen, f_red_cen, c='mediumorchid', label='ECO best fit')
plt.scatter(mstar_cen_fid, f_red_cen_fid, c='cornflowerblue', label='fiducial')
plt.ylabel(r'$f_{red}$', fontsize=30)
plt.xlabel(r'$M_{*, cen}$',fontsize=20)
plt.legend(loc='best')
plt.show()

## Red fraction of satellites
plt.scatter(mstar_sat, f_red_sat, c=10**sat_hosthalo, label='ECO best fit')
plt.ylabel(r'$f_{red}$', fontsize=30)
plt.xlabel(r'$M_{*, sat}$',fontsize=20)
plt.colorbar()
plt.legend(loc='best')
plt.show()

plt.scatter(mstar_sat_fid, f_red_sat_fid, c=10**sat_hosthalo_fid, 
    label='fiducial')
plt.ylabel(r'$f_{red}$', fontsize=30)
plt.xlabel(r'$M_{*, sat}$',fontsize=20)
plt.legend(loc='best')
plt.show()

## SMF plot - data vs best fit
# plt.plot(total_model[0], total_model[1], c='k', linestyle='--', linewidth=5, label='model')
plt.plot(red_model[0], red_model[1], c='maroon', linestyle='--', linewidth=5, 
    label='model')
plt.plot(blue_model[0], blue_model[1], c='mediumblue', linestyle='--', 
    linewidth=5)
# plt.plot(total_data[0], total_data[1], c='k', linestyle='-', linewidth=5, label='data')
plt.plot(red_data[0], red_data[1], c='indianred', linestyle='-', linewidth=5, 
    label='data')
plt.plot(blue_data[0], blue_data[1], c='cornflowerblue', linestyle='-', 
    linewidth=5)
yerr_red = err_data_colour[0:5]
yerr_blue = err_data_colour[5:10]
plt.fill_between(x=red_data[0], y1=red_data[1]+yerr_red, 
    y2=red_data[1]-yerr_red, color='r', alpha=0.3)
plt.fill_between(x=blue_data[0], y1=blue_data[1]+yerr_blue, 
    y2=blue_data[1]-yerr_blue, color='b', alpha=0.3)
plt.legend(loc='best')
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,'
    r'\mathrm{h}^{3} \right]$', fontsize=20)
plt.title(r'ECO SMF ($\chi^2$\ = {0}) (best-fit vs data)'.\
    format(np.round(res80.fun, 2)))
plt.show()

## Spread in velocity difference plot - data vs best fit
plt.scatter(centers_red_model, std_red_model, c='maroon', s=50, label='model')
plt.scatter(centers_blue_model, std_blue_model, c='mediumblue', s=50, 
    label='model')
plt.scatter(cen_red, sigma_red, c='indianred', s=50, label='data')
plt.scatter(cen_blue, sigma_blue, c='cornflowerblue', s=50, label='data')
yerr_red = err_data_colour[10:15]
yerr_blue = err_data_colour[15:20]
plt.fill_between(x=cen_red, y1=sigma_red+yerr_red, 
    y2=sigma_red-yerr_red, color='r', alpha=0.3)
plt.fill_between(x=cen_blue, y1=sigma_blue+yerr_blue, 
    y2=sigma_blue-yerr_blue, color='b', alpha=0.3)
plt.legend(loc='best')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star,cen} \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\sigma$', fontsize=30)
plt.title(r'ECO spread in $\delta v$\ (best-fit vs data)')
plt.show()

## SMF plot - fiducial vs best fit
# plt.plot(total_model[0], total_model[1], c='k', linestyle='--', linewidth=5, label='model')
plt.plot(red_model_fid[0], red_model_fid[1], c='maroon', linestyle='--', 
    linewidth=5, label='fiducial')
plt.plot(blue_model_fid[0], blue_model_fid[1], c='mediumblue', linestyle='--', 
    linewidth=5)
# plt.plot(total_data[0], total_data[1], c='k', linestyle='-', linewidth=5, label='data')
plt.plot(red_model[0], red_model[1], c='indianred', linestyle='-', linewidth=5, 
    label='best fit')
plt.plot(blue_model[0], blue_model[1], c='cornflowerblue', linestyle='-', 
    linewidth=5)
yerr_red = err_data_colour[0:5]
yerr_blue = err_data_colour[5:10]
plt.fill_between(x=red_data[0], y1=red_data[1]+yerr_red, 
    y2=red_data[1]-yerr_red, color='r', alpha=0.3)
plt.fill_between(x=blue_data[0], y1=blue_data[1]+yerr_blue, 
    y2=blue_data[1]-yerr_blue, color='b', alpha=0.3)
plt.legend(loc='best')
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,'
    r'\mathrm{h}^{3} \right]$', fontsize=20)
plt.title(r'ECO SMF ($\chi^2$\ = {0})'.format(np.round(res80.fun, 2)))
plt.show()

## Spread in velocity difference plot - fiducial vs best fit
plt.scatter(centers_red_model_fid, std_red_model_fid, c='maroon', s=50, 
    label='fiducial')
plt.scatter(centers_blue_model_fid, std_blue_model_fid, c='mediumblue', s=50, 
    label='fiducial')
plt.scatter(centers_red_model, std_red_model, c='indianred', s=50, 
    label='best fit')
plt.scatter(centers_blue_model, std_blue_model, c='cornflowerblue', s=50, 
    label='best fit')
yerr_red = err_data_colour[10:15]
yerr_blue = err_data_colour[15:20]
plt.fill_between(x=cen_red, y1=sigma_red+yerr_red, 
    y2=sigma_red-yerr_red, color='r', alpha=0.3)
plt.fill_between(x=cen_blue, y1=sigma_blue+yerr_blue, 
    y2=sigma_blue-yerr_blue, color='b', alpha=0.3)
plt.legend(loc='best')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star,cen} \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\sigma$', fontsize=30)
plt.title(r'ECO spread in $\delta v$')
plt.show()


## Histogram of M* of galaxies labeled red and blue in data and model
plt.hist(catl.logmstar.loc[catl.colour_label == 'R'], 
    bins=np.linspace(8.9, 12, 10), label='data', 
    histtype='step', color='r', lw=5)
plt.hist(gals_df.logmstar.loc[gals_df.colour_label == 'R'], 
    bins=np.linspace(8.9, 12, 10), label='best-fit', 
    histtype='step', color='indianred', lw=5)
plt.hist(catl.logmstar.loc[catl.colour_label == 'B'], 
    bins=np.linspace(8.9, 11.5, 8), label='data', 
    histtype='step', color='b', lw=5)
plt.hist(gals_df.logmstar.loc[gals_df.colour_label == 'B'],
    bins=np.linspace(8.9, 11.5, 8), label='best-fit', 
    histtype='step', color='cornflowerblue', lw=5)
plt.legend(loc='best')
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$', fontsize=20)
plt.show()

## Histogram of u-r colours of galaxies labeled red and blue in data and model
plt.hist(catl.modelu_rcorr.loc[catl.colour_label == 'R'], 
    bins=np.linspace(1.4, 3.4, 7), label='data', 
    histtype='step', color='r', lw=5)
plt.hist(gals_df.modelu_rcorr.loc[gals_df.colour_label == 'R'], 
    bins=np.linspace(1.4, 3.4, 7), label='best-fit', 
    histtype='step', color='indianred', lw=5)
plt.hist(catl.modelu_rcorr.loc[catl.colour_label == 'B'], 
    bins=np.linspace(0.5, 2.0, 7), label='data', 
    histtype='step', color='b', lw=5)
plt.hist(gals_df.modelu_rcorr.loc[gals_df.colour_label == 'B'],
    bins=np.linspace(0.5, 2.0, 7), label='best-fit', 
    histtype='step', color='cornflowerblue', lw=5)
plt.legend(loc='best')
plt.xlabel(r'\boldmath(u-r)', fontsize=20)
plt.show()

## Plot of change of parameter values and chi-squared at each iteration 
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
labels=['10', '20', '40', '80']
colours=['indianred', 'yellowgreen', 'cornflowerblue', 'orchid']
for idx in range(len(labels)):
    idx2 = 0
    res = vars()['res{0}'.format(labels[idx])]
    ax1.scatter(idx+1, res.x[idx2], c=colours[idx], s=100, label=labels[idx])
ax1.axhline(x0[0], ls='--', color='k')
ax1.minorticks_on()
ax1.set_axisbelow(True)
ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
ax1.set_ylabel(r'$\mathbf{M^{q}_{*}}$')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.legend(loc='best', prop={'size': 8}, title='Iterations')

ax2.scatter([1,2,3,4], [res10.x[1], res20.x[1], res40.x[1], best_fit[1]], 
    c=['indianred', 'yellowgreen', 'cornflowerblue', 'orchid'], s=100)
ax2.axhline(x0[1], ls='--', color='k')
ax2.minorticks_on()
ax2.set_axisbelow(True)
ax2.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
ax2.set_ylabel(r'$\mathbf{M^{q}_{h}}$')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

ax3.scatter([1,2,3,4], [res10.x[2], res20.x[2], res40.x[2], best_fit[2]], 
    c=['indianred', 'yellowgreen', 'cornflowerblue', 'orchid'], s=100)
ax3.axhline(x0[2], ls='--', color='k')
ax3.minorticks_on()
ax3.set_axisbelow(True)
ax3.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
ax3.set_ylabel(r'$\boldsymbol{\mu}$')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

ax4.scatter([1,2,3,4], [res10.x[3], res20.x[3], res40.x[3], best_fit[3]], 
    c=['indianred', 'yellowgreen', 'cornflowerblue', 'orchid'], s=100)
ax4.axhline(x0[3], ls='--', color='k')
ax4.minorticks_on()
ax4.set_axisbelow(True)
ax4.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax4.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
ax4.set_ylabel(r'$\boldsymbol{\nu}$')
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

ax5.scatter([1,2,3,4], [res10.fun, res20.fun, res40.fun, res80.fun], 
    c=['indianred', 'yellowgreen', 'cornflowerblue', 'orchid'], s=100)
ax5.minorticks_on()
ax5.set_axisbelow(True)
ax5.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax5.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
ax5.set_ylabel(r'$\boldsymbol{{\ \chi}^2}$')
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle('Hybrid quenching model parameters')
plt.xlabel('Set')
plt.show()


################################################################################

## Extra check

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import argparse
import random
import math
import time
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])

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
    ver = 2.0
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
    ver = 2.0
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

def get_paramvals_percentile(mcmc_table, pctl, chi2):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 1000 sample

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
        values[0][:5]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][5]
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2

def get_centrals_mock(gals_df):
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
    cen_gals = []
    cen_halos = []

    for idx,value in enumerate(gals_df['C_S']):
        if value == 1:
            cen_gals.append(gals_df['stellar_mass'][idx])
            cen_halos.append(gals_df['halo_mvir'][idx])

    cen_gals = np.log10(np.array(cen_gals))
    cen_halos = np.log10(np.array(cen_halos))

    return cen_gals, cen_halos

def halocat_init(halo_catalog,z_median):
    """
    Initial population of halo catalog using populate_mock function

    Parameters
    ----------
    halo_catalog: string
        Path to halo catalog
    
    z_median: float
        Median redshift of survey

    Returns
    ---------
    model: halotools model instance
        Model based on behroozi 2010 SMHM
    """
    halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_median, \
        prim_haloprop_key='halo_macc')
    model.populate_mock(halocat,seed=5)

    return model

def populate_mock(theta):
    """
    Populate mock based on five parameter values 

    Parameters
    ----------
    theta: array
        Array of parameter values

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model_init.param_dict['smhm_m1_0'] = mhalo_characteristic
    model_init.param_dict['smhm_m0_0'] = mstellar_characteristic
    model_init.param_dict['smhm_beta_0'] = mlow_slope
    model_init.param_dict['smhm_delta_0'] = mhigh_slope
    model_init.param_dict['scatter_model_param1'] = mstellar_scatter

    model_init.mock.populate()

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.9) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.4) / 2.041), 1)
    elif survey == 'resolveb':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.7) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.1) / 2.041), 1)
    sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model_init.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def get_best_fit_model(best_fit_params):
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
    v_sim = 130**3
    gals_df = populate_mock(best_fit_params)
    mstellar_mock = gals_df.stellar_mass.values  # Read stellar masses

    if mf_type == 'smf':
        max_model, phi_model, err_tot_model, bins_model, counts_model =\
            diff_smf(mstellar_mock, v_sim, True)
    elif mf_type == 'bmf':
        max_model, phi_model, err_tot_model, bins_model, counts_model =\
            diff_bmf(mstellar_mock, v_sim, True)
    cen_gals, cen_halos = get_centrals_mock(gals_df)

    return max_model, phi_model, err_tot_model, counts_model, cen_gals, \
        cen_halos

def get_xmhm_mocks(survey, path, mf_type):
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
        Standard deviation of phi values between samples of 8 mocks
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


    x_arr = []
    y_arr = []
    y_std_err_arr = []
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
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)]
                cen_gals = np.log10(10**(mock_pd.logmstar.loc
                    [mock_pd.cs_flag == 1])/2.041)
                # cen_halos = mock_pd.M_group.loc[mock_pd.cs_flag == 1]
                cen_halos = mock_pd.loghalom.loc[mock_pd.cs_flag == 1]

                x,y,y_std,y_std_err = Stats_one_arr(cen_halos, cen_gals, base=0.4,
                    bin_statval='center')

            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)]
                cen_gals_stellar = np.log10(10**(mock_pd.logmstar.loc
                    [mock_pd.cs_flag == 1])/2.041)
                cen_gals_gas = mock_pd.mhi.loc[mock_pd.cs_flag == 1]
                cen_gals_gas = np.log10((1.4 * cen_gals_gas)/2.041)
                cen_gals_bary = calc_bary(cen_gals_stellar, cen_gals_gas)
                mock_pd['cen_gals_bary'] = cen_gals_bary
                if survey == 'eco' or survey == 'resolvea':
                    limit = np.log10((10**9.4) / 2.041)
                    cen_gals_bary = mock_pd.cen_gals_bary.loc\
                        [mock_pd.cen_gals_bary >= limit]
                    cen_halos = mock_pd.M_group.loc[(mock_pd.cs_flag == 1) & 
                        (mock_pd.cen_gals_bary >= limit)]
                elif survey == 'resolveb':
                    limit = np.log10((10**9.1) / 2.041)
                    cen_gals_bary = mock_pd.cen_gals_bary.loc\
                        [mock_pd.cen_gals_bary >= limit]
                    cen_halos = mock_pd.M_group.loc[(mock_pd.cs_flag == 1) & 
                        (mock_pd.cen_gals_bary >= limit)]

                x,y,y_std,y_std_err = Stats_one_arr(cen_halos, cen_gals_bary, 
                    base=0.4, bin_statval='center')        

            x_arr.append(x)
            y_arr.append(y)
            y_std_err_arr.append(y_std_err)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    y_std_err_arr = np.array(y_std_err_arr)

    return [x_arr, y_arr, y_std_err_arr]

global model_init
global survey
global path_to_figures
global mf_type

survey = 'eco'
machine = 'mac'
mf_type = 'smf'
ver = 2.0 # No more .dat

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_run6/'
elif mf_type == 'bmf':
    path_to_proc = path_to_proc + 'bmhm_run3/'

chi2_file = path_to_proc + '{0}_chi2.txt'.format(survey)

if mf_type == 'smf' and survey == 'eco' and ver == 1.0:
    chain_file = path_to_proc + 'mcmc_{0}.dat'.format(survey)
else:
    chain_file = path_to_proc + 'mcmc_{0}_raw.txt'.format(survey)

if survey == 'eco':
    catl_file = path_to_raw + 'eco/eco_all.csv'
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + 'RESOLVE_liveJune2019.csv'
    if survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    else:
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)
print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)
print('Reading catalog')
catl, volume, z_median = read_data_catl(catl_file, survey)
print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

maxis_bf, phi_bf, err_tot_bf, counts_bf, cen_gals_bf, cen_halos_bf = \
get_best_fit_model(bf_params) #mstar cut at 8.9 in h=1

result = get_xmhm_mocks(survey, path_to_mocks, mf_type)

fig1 = plt.figure(figsize=(10,10))
x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(cen_halos_bf,
    cen_gals_bf,base=0.4,bin_statval='center')
x_mocks, y_mocks, y_std_err_mocks = result[0], result[1], \
    result[2]
for i in range(len(x_mocks)):
    plt.errorbar(x_mocks[i],y_mocks[i],yerr=y_std_err_mocks[i],
        color='lightgray',fmt='-s', ecolor='lightgray', markersize=4, 
        capsize=5, capthick=0.5, label=r'mocks',zorder=5)
plt.errorbar(x_bf,y_bf,color='mediumorchid',fmt='-s',
    ecolor='mediumorchid',markersize=5,capsize=5,capthick=0.5,\
        label=r'best-fit',zorder=20)


plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, '
    r'\mathrm{h}^{-1} \right]$',fontsize=20)
if mf_type == 'smf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**8.9)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**8.7)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, '
        r'\mathrm{h}^{-1} \right]$',fontsize=20)
elif mf_type == 'bmf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**9.4)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**9.1)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, '
        r'\mathrm{h}^{-1} \right]$',fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 15})
# plt.savefig(path_to_figures + 'mocks_data_xmhm_{0}.png'.format(survey))
plt.show()
