"""
{This script plots SMF, blue fraction, SMHM and average group central stellar 
 mass vs. velocity dispersion from results of the chain where all 9 params 
 (behroozi and quenching) were varied. Rsd and group-finding is done on a subset 
 of 100 models from the chain that correspond to 68th percentile of lowest 
 chi-squared values so that the dynamical observable can be measured even though 
 it was not used to constrain the modeling. All plots are compared with data. 
 The rsd and group-finding were done separately and the file is simply read in 
 this script.}
"""

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerBase
from scipy.stats import binned_statistic as bs
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import markers, rc
import pandas as pd
import numpy as np
import time
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

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
        Array of chi^2 values to match chain values
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
    emcee_table: pandas.DataFrame
        Dataframe of mcmc chain values with NANs removed
    """
    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'mstar_q','mh_q','mu','nu']
    
    emcee_table = pd.read_csv(path_to_file, names=colnames, comment='#',
        header=None, sep='\s+')

    for idx,row in enumerate(emcee_table.values):

        ## For cases where 5 params on one line and 3 on the next
        if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
            mhalo_q_val = emcee_table.values[idx+1][0]
            mu_val = emcee_table.values[idx+1][1]
            nu_val = emcee_table.values[idx+1][2]
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

        ## For cases where 4 params on one line, 4 on the next and 1 on the 
        ## third line (numbers in scientific notation unlike case above)
        elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            mstar_q_val = emcee_table.values[idx+1][1]
            mhalo_q_val = emcee_table.values[idx+1][2]
            mu_val = emcee_table.values[idx+1][3]
            nu_val = emcee_table.values[idx+2][0]
            row[4] = scatter_val
            row[5] = mstar_q_val
            row[6] = mhalo_q_val
            row[7] = mu_val
            row[8] = nu_val 

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

def mock_add_grpcz(mock_df, data_bool=None, grpid_col=None):
    """Adds column of group cz values to mock catalogues

    Args:
        mock_df (pandas.DataFrame): Mock catalogue

    Returns:
        pandas.DataFrame: Mock catalogue with new column called grpcz added
    """
    if data_bool:
        grpcz = mock_df.groupby('groupid').cz.mean().values
        grp = np.unique(mock_df.groupid.values)
        mydict = dict(zip(grp, grpcz))
        full_grpcz_arr = [np.round(mydict[val],2) for val in mock_df.groupid.values]
        mock_df['grpcz_new'] = full_grpcz_arr
    elif data_bool is None:
        ## Mocks case
        grpcz = mock_df.groupby('groupid').cz.mean().values
        grp = np.unique(mock_df.groupid.values)
        mydict = dict(zip(grp, grpcz))
        full_grpcz_arr = [np.round(mydict[val],2) for val in mock_df.groupid.values]
        mock_df['grpcz'] = full_grpcz_arr
    else:
        ## Models from Vishnu
        grpcz = mock_df.groupby(grpid_col).cz.mean().values
        grp = np.unique(mock_df[grpid_col].values)
        mydict = dict(zip(grp, grpcz))
        full_grpcz_arr = [np.round(mydict[val],2) for val in mock_df[grpid_col].values]
        mock_df['grpcz'] = full_grpcz_arr
    return mock_df

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
        #             'fc', 'grpmb', 'grpms','modelu_rcorr', 'grpsig', 'grpsig_stack']

        # # 13878 galaxies
        # eco_buff = pd.read_csv(path_to_file, delimiter=",", header=0, 
        #     usecols=columns)

        eco_buff = read_mock_catl(path_to_file)
        eco_buff = mock_add_grpcz(eco_buff, True)

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

def get_paramvals_percentile(mcmc_table, pctl, chi2, randints_df=None):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 100 sample

    Parameters
    ----------
    mcmc_table: pandas.DataFrame
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2: array
        Array of chi^2 values
    
    randints_df (optional): pandas.DataFrame
        Dataframe of mock numbers in case many Behroozi mocks were used.
        Defaults to None.

    Returns
    ---------
    mcmc_table_pctl: pandas dataframe
        Sample of 100 68th percentile lowest chi^2 values
    bf_params: numpy array
        Array of parameter values corresponding to the best-fit model
    bf_chi2: float
        Chi-squared value corresponding to the best-fit model
    bf_randint: int
        In case multiple Behroozi mocks were used, this is the mock number
        that corresponds to the best-fit model. Otherwise, this is not returned.
    """ 
    pctl = pctl/100
    mcmc_table['chi2'] = chi2
    if randints_df is not None: # This returns a bool; True if df has contents
        mcmc_table['mock_num'] = randints_df.mock_num.values.astype(int)
    mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:9]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][9]
    if randints_df is not None:
        bf_randint = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
            values[0][5].astype(int)
        mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)
        return mcmc_table_pctl, bf_params, bf_chi2, bf_randint
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas.DataFrame 
        Data catalog

    Returns
    ---------
    catl: pandas.DataFrame
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
    table: pandas.DataFrame
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    data_bool: boolean
        Data or mock
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    3 multidimensional arrays of [stellar mass, phi, total error in SMF and 
    counts per bin] for all, red and blue galaxies
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
        return [max_total, phi_total, err_total, counts_total] , \
            [max_red, phi_red, err_red, counts_red] , \
                [max_blue, phi_blue, err_blue, counts_blue]
    else:
        if randint_logmstar != 1:
            logmstar_col = '{0}'.format(randint_logmstar)
        elif randint_logmstar == 1:
            logmstar_col = 'behroozi_bf'
        else:
            logmstar_col = 'stellar_mass'
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
    
    colour_flag (optional): boolean
        'R' if galaxy masses correspond to red galaxies & 'B' if galaxy masses
        correspond to blue galaxies. Defaults to False.

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
    
    counts: array
        Array of number of things in each bin
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

def blue_frac_helper(arr):
    """Helper function for blue_frac() that calculates the fraction of blue 
    galaxies

    Args:
        arr (numpy array): Array of 'R' and 'B' characters depending on whether
        galaxy is red or blue

    Returns:
        numpy array: Array of floats representing fractions of blue galaxies in 
        each bin
    """
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
        mstar_arr = catl.logmstar.values
    elif randint_logmstar != 1:
        mstar_arr = catl['{0}'.format(randint_logmstar)].values
    elif randint_logmstar == 1:
        mstar_arr = catl['behroozi_bf'].values

    colour_label_arr = catl.colour_label.values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = mstar_arr

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 7

        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result = bs(logmstar_arr, colour_label_arr, blue_frac_helper, bins=bins)
    edges = result[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue = result[0]

    return maxis, f_blue

def halocat_init(halo_catalog, z_median):
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

def hybrid_quenching_model(theta, gals_df, mock, randint=None):
    """
    Apply hybrid quenching model from Zu and Mandelbaum 2015

    Parameters
    ----------
    theta: numpy array
        Array of quenching model parameter values
    gals_df: pandas dataframe
        Mock catalog
    mock: string
        'vishnu' or 'nonvishnu' depending on what mock it is
    randint (optional): int
        Mock number in the case where many Behroozi mocks were used.
        Defaults to None.

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

def get_host_halo_mock(df, mock):
    """
    Get host halo mass from mock catalog

    Parameters
    ----------
    df: pandas dataframe
        Mock catalog
    mock: string
        'vishnu' or 'nonvishnu' depending on what mock it is

    Returns
    ---------
    cen_halos: array
        Array of central host halo masses
    sat_halos: array
        Array of satellite host halo masses
    """
    if mock == 'vishnu':
        cen_halos = df.halo_mvir[df.cs_flag == 1].reset_index(drop=True)
        sat_halos = df.halo_mvir_host_halo[df.cs_flag == 0].reset_index(drop=True)
    else:
        # Loghalom in the mock catalogs is actually host halo mass i.e. 
        # For satellites, the loghalom value will be the value of the central's
        # loghalom in that halo group and the haloids for the satellites are the 
        # haloid of the central 
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
    df: pandas dataframe
        Mock catalog
    mock: string
        'Vishnu' or 'nonVishnu' depending on what mock it is
    randint (optional): int
        Mock number in the case where many Behroozi mocks were used.
        Defaults to None.

    Returns
    ---------
    cen_gals: array
        Array of central stellar masses
    sat_gals: array
        Array of satellite stellar masses
    """

    if mock == 'vishnu' and randint != 1:
        cen_gals = 10**(df['{0}'.format(randint)][df.cs_flag == 1]).\
            reset_index(drop=True)
        sat_gals = 10**(df['{0}'.format(randint)][df.cs_flag == 0]).\
            reset_index(drop=True)

    elif mock == 'vishnu' and randint == 1:
        cen_gals = 10**(df['behroozi_bf'][df.cs_flag == 1]).\
            reset_index(drop=True)
        sat_gals = 10**(df['behroozi_bf'][df.cs_flag == 0]).\
            reset_index(drop=True)

    # elif mock == 'vishnu':
    #     cen_gals = 10**(df.stellar_mass[df.cs_flag == 1]).reset_index(drop=True)
    #     sat_gals = 10**(df.stellar_mass[df.cs_flag == 0]).reset_index(drop=True)
    
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

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    df: pandas Dataframe
        Mock catalog
    drop_fred (optional): boolean
        Whether or not to keep red fraction column after colour has been
        assigned. Defaults to False.

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
    err_colour: array
        Standard deviation from matrix of phi values and blue fractions values
        between all mocks and for all galaxies
    std_phi_red: array
        Standard deviation of phi values between all mocks for red galaxies
    std_phi_blue: array
        Standard deviation of phi values between all mocks for blue galaxies
    std_mean_cen_arr_red: array
        Standard deviation of observable number 3 (mean grp central stellar mass
        in bins of velocity dispersion) for red galaxies
    std_mean_cen_arr_blue: array
        Standard deviation of observable number 3 (mean grp central stellar mass
        in bins of velocity dispersion) for blue galaxies
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
    f_blue_arr = []
    mean_cen_arr_red = []
    mean_cen_arr_blue = []
    veldisp_arr_red = []
    veldisp_arr_blue = []
    veldisp_cen_arr_red = []
    veldisp_cen_arr_blue = []
    red_cen_stellar_mass_arr = []
    red_num_arr = []
    blue_cen_stellar_mass_arr = []
    blue_num_arr = []
    box_id_arr = np.linspace(5001,5008,8)

    start = time.time()
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            print('Box {0} : Mock {1}'.format(box, num))
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = read_mock_catl(filename) 
            mock_pd = mock_add_grpcz(mock_pd)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.grpcz.values >= min_cz) & \
                (mock_pd.grpcz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

            ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16
            
            # ## Using best-fit found for new ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            # Mstar_q = 10.06067888
            # Mh_q = 14.05665242
            # mu = 0.56853249
            # nu = 0.48598653

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 32
            ## i.e. hybrid quenching model
            Mstar_q = 10.06 # Msun/h
            Mh_q = 14.05 # Msun/h
            mu = 0.56
            nu = 0.48

            ## Using best-fit found for new ECO data using result from chain 33
            ## i.e. halo quenching model
            Mh_qc = 11.78 # Msun/h
            Mh_qs = 13.14 # Msun/h
            mu_c = 1.09
            mu_s = 1.99

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')        
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            logmstar_arr = mock_pd.logmstar.values 

            #Measure SMF of mock using diff_smf function
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(logmstar_arr, volume, False)
            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_arr_total.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)

            #Measure blue fraction of galaxies
            max, f_blue = blue_frac(mock_pd, False, True)
            f_blue_arr.append(f_blue)

            phi_red, phi_blue = \
                get_colour_smf_from_fblue(mock_pd, f_blue, max, volume, 
                False)
            phi_arr_red.append(phi_red)
            phi_arr_blue.append(phi_blue)
            
            if level == 'group':
                ## Group level
                new_mean_stats_red, new_centers_red, new_mean_stats_blue, \
                    new_centers_blue = \
                    get_sigma_per_group_mocks_qmcolour(survey, mock_pd)

                ## Group level
                veldisp_red, veldisp_blue, veldisp_cen_red, veldisp_cen_blue = \
                    get_deltav_sigma_mocks_qmcolour(survey, mock_pd)

                ## Group level
                red_cen_stellar_mass, red_num, blue_cen_stellar_mass, \
                    blue_num = \
                    get_N_per_group_mocks_qmcolour(survey, mock_pd, central_bool=True)

            elif level == 'halo':
                ## Halo level
                new_mean_stats_red, new_centers_red, new_mean_stats_blue, \
                    new_centers_blue = \
                    get_sigma_per_halo_mocks_qmcolour(survey, mock_pd)

                ## Halo level
                veldisp_red, veldisp_blue, veldisp_cen_red, veldisp_cen_blue = \
                    get_deltav_sigma_halo_mocks_qmcolour(survey, mock_pd)

                ## Halo level
                red_cen_stellar_mass, red_num, blue_cen_stellar_mass, \
                    blue_num = \
                    get_N_per_halo_mocks_qmcolour(survey, mock_pd, central_bool=True)

            mean_cen_arr_red.append(new_mean_stats_red[0])
            mean_cen_arr_blue.append(new_mean_stats_blue[0])

            veldisp_arr_red.append(veldisp_red)
            veldisp_arr_blue.append(veldisp_blue)

            red_cen_stellar_mass_arr.append(red_cen_stellar_mass)
            red_num_arr.append(red_num)
            blue_cen_stellar_mass_arr.append(blue_cen_stellar_mass)
            blue_num_arr.append(blue_num)

    phi_arr_total = np.array(phi_arr_total)
    f_blue_arr = np.array(f_blue_arr)

    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)

    std_phi_red = np.std(phi_arr_red, axis=0)
    std_phi_blue = np.std(phi_arr_blue, axis=0)

    mean_cen_arr_red = np.array(mean_cen_arr_red)
    mean_cen_arr_blue = np.array(mean_cen_arr_blue)

    std_mean_cen_arr_red = np.nanstd(mean_cen_arr_red, axis=0)
    std_mean_cen_arr_blue = np.nanstd(mean_cen_arr_blue, axis=0)

    std_veldisp_arr_red = np.nanstd(veldisp_arr_red, axis=0)
    std_veldisp_arr_blue = np.nanstd(veldisp_arr_blue, axis=0)

    # Covariance matrix for total phi (all galaxies)
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    err_total = np.sqrt(cov_mat.diagonal())

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]
    phi_total_4 = phi_arr_total[:,4]
    phi_total_5 = phi_arr_total[:,5]

    f_blue_0 = f_blue_arr[:,0]
    f_blue_1 = f_blue_arr[:,1]
    f_blue_2 = f_blue_arr[:,2]
    f_blue_3 = f_blue_arr[:,3]
    f_blue_4 = f_blue_arr[:,4]
    f_blue_5 = f_blue_arr[:,5]

    combined_df = pd.DataFrame({'phi_tot_0':phi_total_0, \
        'phi_tot_1':phi_total_1, 'phi_tot_2':phi_total_2, \
        'phi_tot_3':phi_total_3, 'phi_tot_4':phi_total_4, \
        'phi_tot_5':phi_total_5,\
        'f_blue_0':f_blue_0, 'f_blue_1':f_blue_1, 
        'f_blue_2':f_blue_2, 'f_blue_3':f_blue_3, 
        'f_blue_4':f_blue_4, 'f_blue_5':f_blue_5})

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    # corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    end = time.time()
    total_time = end - start
    print("Mock processing took {0:.1f} seconds".format(total_time))

    return err_colour, std_phi_red, std_phi_blue, std_mean_cen_arr_red, \
        std_mean_cen_arr_blue, std_veldisp_arr_red, std_veldisp_arr_blue, \
        red_cen_stellar_mass_arr, red_num_arr, blue_cen_stellar_mass_arr, \
        blue_num_arr

def populate_mock(theta, model):
    """
    Populate mock based on five SMHM parameter values and model

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    model: halotools model instance
        Model based on behroozi 2010 SMHM

    Returns
    ---------
    gals_df: pandas.DataFrame
        Dataframe of Vishnu mock catalog
    """
    """"""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate(seed=5)

    # if survey == 'eco' or survey == 'resolvea':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.9) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.4) / 2.041), 1)
    # elif survey == 'resolveb':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.7) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.1) / 2.041), 1)
    # sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model.mock.galaxy_table#[sample_mask]
    gals_df = pd.DataFrame(np.array(gals))

    return gals_df

def get_centrals_mock(gals_df, randint=None):
    """
    Get centrals from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    randint (optional): int
        Mock number in the case where many Behroozi mocks were used.
        Defaults to None.

    Returns
    ---------
    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses

    cen_gals_red: array
        Array of red central galaxy masses

    cen_halos_red: array
        Array of red central halo masses

    cen_gals_blue: array
        Array of blue central galaxy masses

    cen_halos_blue: array
        Array of blue central halo masses

    f_red_cen_gals_red: array
        Array of red fractions for red central galaxies

    f_red_cen_gals_blue: array
        Array of red fractions for blue central galaxies
    """
    cen_gals = []
    cen_halos = []
    cen_gals_red = []
    cen_halos_red = []
    cen_gals_blue = []
    cen_halos_blue = []
    f_red_cen_gals_red = []
    f_red_cen_gals_blue = []

    if randint != 1:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 1:
                cen_gals.append(gals_df['{0}'.format(randint)][idx])
                cen_halos.append(gals_df['halo_mvir'][idx])
                if gals_df['colour_label'][idx] == 'R':
                    cen_gals_red.append(gals_df['{0}'.format(randint)][idx])
                    cen_halos_red.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    cen_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                    cen_halos_blue.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_blue.append(gals_df['f_red'][idx])
    elif randint == 1:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 1:
                cen_gals.append(gals_df['behroozi_bf'][idx])
                cen_halos.append(gals_df['halo_mvir'][idx])
                if gals_df['colour_label'][idx] == 'R':
                    cen_gals_red.append(gals_df['behroozi_bf'][idx])
                    cen_halos_red.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    cen_gals_blue.append(gals_df['behroozi_bf'][idx])
                    cen_halos_blue.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_blue.append(gals_df['f_red'][idx])

    else:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 1:
                cen_gals.append(gals_df['stellar_mass'][idx])
                cen_halos.append(gals_df['halo_mvir'][idx])
                if gals_df['colour_label'][idx] == 'R':
                    cen_gals_red.append(gals_df['stellar_mass'][idx])
                    cen_halos_red.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    cen_gals_blue.append(gals_df['stellar_mass'][idx])
                    cen_halos_blue.append(gals_df['halo_mvir'][idx])
                    f_red_cen_gals_blue.append(gals_df['f_red'][idx])

    cen_gals = np.array(cen_gals)
    cen_halos = np.log10(np.array(cen_halos))
    cen_gals_red = np.array(cen_gals_red)
    cen_halos_red = np.log10(np.array(cen_halos_red))
    cen_gals_blue = np.array(cen_gals_blue)
    cen_halos_blue = np.log10(np.array(cen_halos_blue))

    return cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
        cen_halos_blue, f_red_cen_gals_red, f_red_cen_gals_blue

def get_satellites_mock(gals_df, randint=None):
    """
    Get satellites and their host halos from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog
        
    randint (optional): int
        Mock number in the case where many Behroozi mocks were used. 
        Defaults to None.

    Returns
    ---------
    sat_gals_red: array
        Array of red satellite galaxy masses

    sat_halos_red: array
        Array of red satellite host halo masses

    sat_gals_blue: array
        Array of blue satellite galaxy masses

    sat_halos_blue: array
        Array of blue satellite host halo masses

    f_red_sat_gals_red: array
        Array of red fractions for red satellite galaxies

    f_red_sat_gals_blue: array
        Array of red fractions for blue satellite galaxies

    """
    sat_gals_red = []
    sat_halos_red = []
    sat_gals_blue = []
    sat_halos_blue = []
    f_red_sat_gals_red = []
    f_red_sat_gals_blue = []

    if randint != 1:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 0:
                if gals_df['colour_label'][idx] == 'R':
                    sat_gals_red.append(gals_df['{0}'.format(randint)][idx])
                    sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    sat_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                    sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_blue.append(gals_df['f_red'][idx])
    elif randint == 1:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 0:
                if gals_df['colour_label'][idx] == 'R':
                    sat_gals_red.append(gals_df['behroozi_bf'][idx])
                    sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    sat_gals_blue.append(gals_df['behroozi_bf'][idx])
                    sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_blue.append(gals_df['f_red'][idx])

    else:
        for idx,value in enumerate(gals_df['cs_flag']):
            if value == 0:
                if gals_df['colour_label'][idx] == 'R':
                    sat_gals_red.append(gals_df['stellar_mass'][idx])
                    sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_red.append(gals_df['f_red'][idx])
                elif gals_df['colour_label'][idx] == 'B':
                    sat_gals_blue.append(gals_df['stellar_mass'][idx])
                    sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                    f_red_sat_gals_blue.append(gals_df['f_red'][idx])


    sat_gals_red = np.array(sat_gals_red)
    sat_halos_red = np.log10(np.array(sat_halos_red))
    sat_gals_blue = np.array(sat_gals_blue)
    sat_halos_blue = np.log10(np.array(sat_halos_blue))

    return sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
        f_red_sat_gals_red, f_red_sat_gals_blue

def mp_init(mcmc_table_pctl, nproc):
    """
    Initializes multiprocessing of mocks and smf and smhm measurements

    Parameters
    ----------
    mcmc_table_pctl: pandas dataframe
        Mcmc chain dataframe of 100 random samples

    nproc: int
        Number of processes to use in multiprocessing

    Returns
    ---------
    result: multidimensional array
        Arrays of smf and smhm data for all, red and blue galaxies
    """
    start = time.time()
    params_df = mcmc_table_pctl.iloc[:,:9].reset_index(drop=True)
    if many_behroozi_mocks:
        mock_num_df = mcmc_table_pctl.iloc[:,5].reset_index(drop=True)
        frames = [params_df, mock_num_df]
        mcmc_table_pctl_new = pd.concat(frames, axis=1)
        chunks = np.array([mcmc_table_pctl_new.values[i::5] \
            for i in range(5)])
    else:
        # chunks = np.array([params_df.values[i::5] \
        #     for i in range(5)])
        # Chunks are just numbers from 1-100 for the case where rsd + grp finder
        # were run for a selection of 100 random 1sigma models from run 32
        # and all those mocks are used instead. 
        chunks = np.arange(1,101,1).reshape(5, 20, 1) # Mimic shape of chunks above 
    pool = Pool(processes=nproc)
    result = pool.map(mp_func, chunks)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return result

def mp_func(a_list):
    """
    Apply behroozi and hybrid quenching model based on nine parameter values

    Parameters
    ----------
    a_list: multidimensional array
        Array of nine parameter values

    Returns
    ---------
    maxis_total_arr: array
        Array of x-axis mass values for all galaxies

    phi_total_arr: array
        Array of y-axis phi values for all galaxies

    maxis_fblue_arr: array
        Array of x_axis mass values for bleu fraction measurement

    f_blue_arr: array
        Array of blue fraction values

    phi_red_model_arr: array
        Array of y-axis phi values for red galaxies

    phi_blue_model_arr: array
        Array of y-axis phi values for blue galaxies

    cen_gals_red_arr: array
        Array of red central galaxy masses

    cen_halos_red_arr: array
        Array of red central halo masses

    cen_gals_blue_arr: array
        Array of blue central galaxy masses
    
    cen_halos_blue_arr: array
        Array of blue central halo masses

    f_red_cen_red_arr: array
        Array of red fractions for red central galaxies

    f_red_cen_blue_arr: array
        Array of red fractions for blue central galaxies

    sat_gals_red_arr: array
        Array of red satellite galaxy masses

    sat_halos_red_arr: array
        Array of red satellite host halo masses

    sat_gals_blue_arr: array
        Array of blue satellite galaxy masses

    sat_halos_blue_arr: array
        Array of blue satellite host halo masses

    f_red_sat_red_arr: array
        Array of red fractions for red satellite galaxies

    f_red_sat_blue_arr: array
        Array of red fractions for blue satellite galaxies

    cen_gals_arr: array
        Array of central galaxy masses

    cen_halos_arr: array
        Array of central halo masses

    grp_red_cen_arr: array
        Array of red group central stellar masses

    grp_blue_cen_arr: array
        Array of blue group central stellar masses

    red_sigma_arr: array
        Array of velocity dispersion of galaxies with red group centrals

    blue_sigma_arr: array
        Array of velocity dispersion of galaxies with blue group centrals

    """
    # v_sim = 130**3
    v_sim = 890641.5172927063 

    maxis_total_arr = []
    phi_total_arr = []

    maxis_fblue_arr = []
    f_blue_arr = []

    phi_red_model_arr = []
    phi_blue_model_arr = []

    cen_gals_red_arr = []
    cen_halos_red_arr = []
    cen_gals_blue_arr = []
    cen_halos_blue_arr = []

    f_red_cen_red_arr = []
    f_red_cen_blue_arr = []

    sat_gals_red_arr = [] 
    sat_halos_red_arr = []
    sat_gals_blue_arr = [] 
    sat_halos_blue_arr = [] 

    f_red_sat_red_arr = []
    f_red_sat_blue_arr = []
    cen_gals_arr = []
    cen_halos_arr = []
    grp_red_cen_arr = []
    grp_blue_cen_arr = []
    red_sigma_arr = []
    blue_sigma_arr = []
    vdisp_red_arr = []
    vdisp_blue_arr = []
    vdisp_cen_arr_red = []
    vdisp_cen_arr_blue = []

    vdisp_red_points_arr = []
    vdisp_blue_points_arr = []
    red_host_halo_mass_arr_sigma_mh = []
    blue_host_halo_mass_arr_sigma_mh = []

    red_cen_stellar_mass_arr = []
    red_num_arr = []
    blue_cen_stellar_mass_arr = []
    blue_num_arr = []
    red_host_halo_mass_arr_N_mh = []
    blue_host_halo_mass_arr_N_mh = []

    wtd_red_sigma_arr = []
    wtd_red_cen_stellar_mass_arr = []
    wtd_blue_sigma_arr = []
    wtd_blue_cen_stellar_mass_arr = []
    wtd_red_nsat_arr = []
    wtd_blue_nsat_arr = []


    for theta in a_list:  
        if many_behroozi_mocks:
            randint_logmstar = int(theta[4])
            cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', \
            '{0}'.format(randint_logmstar), \
            'g_galtype_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar)]
            
            gals_df = gal_group_df_subset[cols_to_use]

            gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
                format(randint_logmstar),'groupid_{0}'.format(randint_logmstar)]).\
                reset_index(drop=True)

            gals_df[['g_galtype_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar)]] = \
                gals_df[['g_galtype_{0}'.format(randint_logmstar),\
                'groupid_{0}'.format(randint_logmstar)]].astype(int)


        else:
            # gals_df = populate_mock(theta[:5], model_init)
            # gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].\
            #     reset_index(drop=True)
            # gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
            #     gals_df['halo_id'], 1, 0)

            # cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
            #     'stellar_mass']

            # gals_df.stellar_mass = np.log10(gals_df.stellar_mass)

            randint_logmstar = theta[0]
            # 1 is the best-fit model which is calculated separately 
            if randint_logmstar == 1:
                continue

            cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', 'cs_flag', \
            '{0}'.format(randint_logmstar), \
            'g_galtype_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar)]

            gals_df = gal_group_df_subset[cols_to_use]
    
            gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
            format(randint_logmstar),'groupid_{0}'.format(randint_logmstar)]).\
            reset_index(drop=True)

            gals_df[['g_galtype_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar)]] = \
                gals_df[['g_galtype_{0}'.format(randint_logmstar),\
                'groupid_{0}'.format(randint_logmstar)]].astype(int)

        #* Stellar masses in log but halo masses not in log
        # randint_logmstar-2 because the best fit randint is 1 in gal_group_df
        # and in mcmc_table the best fit set of params have been removed and the
        # index was reset so now there is an offset of 2 between the indices 
        # of the two sets of data.
        quenching_params = mcmc_table_pctl_subset.iloc[randint_logmstar-2].\
            values[5:]
        if quenching == 'hybrid':
            f_red_cen, f_red_sat = hybrid_quenching_model(quenching_params, gals_df, 
                'vishnu', randint_logmstar)
        elif quenching == 'halo':
            f_red_cen, f_red_sat = halo_quenching_model(quenching_params, gals_df, 
                'vishnu')
        gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

        ## Observable #1 - Total SMF
        total_model = measure_all_smf(gals_df, v_sim , False, randint_logmstar)    
        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gals_df, True, False, randint_logmstar)
        
        cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
            cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
                get_centrals_mock(gals_df, randint_logmstar)
        sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
            f_red_sat_red, f_red_sat_blue = \
                get_satellites_mock(gals_df, randint_logmstar)

        phi_red_model, phi_blue_model = \
            get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
            True, randint_logmstar)

        if level == 'group':
            ## Group level
            red_sigma, grp_red_cen_stellar_mass, blue_sigma, \
                grp_blue_cen_stellar_mass = \
                get_sigma_per_group_vishnu_qmcolour(gals_df, randint_logmstar)

            ## Group level
            vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
                vdisp_centers_blue_model = \
                get_deltav_sigma_vishnu_qmcolour(gals_df, randint_logmstar)

            ## Group level
            red_cen_stellar_mass_model, red_num_model, blue_cen_stellar_mass_model, \
                blue_num_model = \
                get_N_per_group_vishnu_qmcolour(gals_df, randint_logmstar, \
                    central_bool=True)

            wtd_red_sigma, wtd_red_cen_stellar_mass, wtd_blue_sigma, \
                wtd_blue_cen_stellar_mass, wtd_red_nsat, wtd_blue_nsat = \
                get_satellite_weighted_sigma_group_vishnu(gals_df, randint_logmstar)

        elif level == 'halo':
            ## Halo level
            red_sigma, grp_red_cen_stellar_mass, blue_sigma, \
                grp_blue_cen_stellar_mass = \
                get_sigma_per_halo_vishnu_qmcolour(gals_df, randint_logmstar)

            ## Halo level
            vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
                vdisp_centers_blue_model, vdisp_red, \
                vdisp_blue, red_host_halo_mass_sigma_mh, blue_host_halo_mass_sigma_mh = \
                get_deltav_sigma_halo_vishnu_qmcolour(gals_df, randint_logmstar)

            ## Halo level
            red_cen_stellar_mass_model, red_num_model, blue_cen_stellar_mass_model, \
                blue_num_model, red_host_halo_mass_N_mh, blue_host_halo_mass_N_mh = \
                get_N_per_halo_vishnu_qmcolour(gals_df, randint_logmstar, \
                    central_bool=True)

            wtd_red_sigma, wtd_red_cen_stellar_mass, wtd_blue_sigma, \
                wtd_blue_cen_stellar_mass, wtd_red_nsat, wtd_blue_nsat = \
                get_satellite_weighted_sigma_halo_vishnu(gals_df, randint_logmstar)

        maxis_total_arr.append(total_model[0])
        phi_total_arr.append(total_model[1])
        maxis_fblue_arr.append(f_blue[0])
        f_blue_arr.append(f_blue[1])
        phi_red_model_arr.append(phi_red_model)
        phi_blue_model_arr.append(phi_blue_model)
        cen_gals_red_arr.append(cen_gals_red)
        cen_halos_red_arr.append(cen_halos_red)
        cen_gals_blue_arr.append(cen_gals_blue)
        cen_halos_blue_arr.append(cen_halos_blue)
        f_red_cen_red_arr.append(f_red_cen_red)
        f_red_cen_blue_arr.append(f_red_cen_blue)
        sat_gals_red_arr.append(sat_gals_red) 
        sat_halos_red_arr.append(sat_halos_red)
        sat_gals_blue_arr.append(sat_gals_blue)
        sat_halos_blue_arr.append(sat_halos_blue)
        f_red_sat_red_arr.append(f_red_sat_red)
        f_red_sat_blue_arr.append(f_red_sat_blue)
        cen_gals_arr.append(cen_gals)
        cen_halos_arr.append(cen_halos)
        grp_red_cen_arr.append(grp_red_cen_stellar_mass)
        grp_blue_cen_arr.append(grp_blue_cen_stellar_mass)
        red_sigma_arr.append(red_sigma)
        blue_sigma_arr.append(blue_sigma)
        vdisp_red_arr.append(vdisp_red_model)
        vdisp_blue_arr.append(vdisp_blue_model)
        vdisp_cen_arr_red.append(vdisp_centers_red_model)
        vdisp_cen_arr_blue.append(vdisp_centers_blue_model)

        red_cen_stellar_mass_arr.append(red_cen_stellar_mass_model)
        red_num_arr.append(red_num_model)
        blue_cen_stellar_mass_arr.append(blue_cen_stellar_mass_model)
        blue_num_arr.append(blue_num_model)

        if level == 'halo':
            vdisp_red_points_arr.append(vdisp_red)
            vdisp_blue_points_arr.append(vdisp_blue)
            red_host_halo_mass_arr_sigma_mh.append(red_host_halo_mass_sigma_mh)
            blue_host_halo_mass_arr_sigma_mh.append(blue_host_halo_mass_sigma_mh)

            ## For N-Mh plot
            red_host_halo_mass_arr_N_mh.append(red_host_halo_mass_N_mh)
            blue_host_halo_mass_arr_N_mh.append(blue_host_halo_mass_N_mh)

        wtd_red_sigma_arr.append(wtd_red_sigma)
        wtd_red_cen_stellar_mass_arr.append(wtd_red_cen_stellar_mass)
        wtd_blue_sigma_arr.append(wtd_blue_sigma)
        wtd_blue_cen_stellar_mass_arr.append(wtd_blue_cen_stellar_mass)
        wtd_red_nsat_arr.append(wtd_red_nsat)
        wtd_blue_nsat_arr.append(wtd_blue_nsat)

    if level == 'halo':
        return [maxis_total_arr, phi_total_arr, maxis_fblue_arr, f_blue_arr, 
        phi_red_model_arr, phi_blue_model_arr,
        cen_gals_red_arr, cen_halos_red_arr, cen_gals_blue_arr, 
        cen_halos_blue_arr, f_red_cen_red_arr, f_red_cen_blue_arr, 
        sat_gals_red_arr, sat_halos_red_arr, sat_gals_blue_arr, 
        sat_halos_blue_arr, f_red_sat_red_arr, f_red_sat_blue_arr,
        cen_gals_arr, cen_halos_arr, grp_red_cen_arr, grp_blue_cen_arr, 
        red_sigma_arr, blue_sigma_arr, vdisp_red_arr, vdisp_blue_arr, 
        vdisp_cen_arr_red, vdisp_cen_arr_blue, vdisp_red_points_arr, 
        vdisp_blue_points_arr, red_host_halo_mass_arr_sigma_mh, 
        blue_host_halo_mass_arr_sigma_mh, red_cen_stellar_mass_arr, 
        red_num_arr, blue_cen_stellar_mass_arr, blue_num_arr, 
        red_host_halo_mass_arr_N_mh, blue_host_halo_mass_arr_N_mh, 
        wtd_red_sigma_arr, wtd_red_cen_stellar_mass_arr, wtd_blue_sigma_arr,
        wtd_blue_cen_stellar_mass_arr, wtd_red_nsat_arr, wtd_blue_nsat_arr]

    elif level == 'group':
        return [maxis_total_arr, phi_total_arr, maxis_fblue_arr, f_blue_arr, 
        phi_red_model_arr, phi_blue_model_arr,
        cen_gals_red_arr, cen_halos_red_arr, cen_gals_blue_arr, cen_halos_blue_arr,
        f_red_cen_red_arr, f_red_cen_blue_arr, sat_gals_red_arr, sat_halos_red_arr, 
        sat_gals_blue_arr, sat_halos_blue_arr, f_red_sat_red_arr, f_red_sat_blue_arr,
        cen_gals_arr, cen_halos_arr, grp_red_cen_arr, grp_blue_cen_arr, 
        red_sigma_arr, blue_sigma_arr, vdisp_red_arr, vdisp_blue_arr, 
        vdisp_cen_arr_red, vdisp_cen_arr_blue, red_cen_stellar_mass_arr, 
        red_num_arr, blue_cen_stellar_mass_arr, blue_num_arr, wtd_red_sigma_arr, 
        wtd_red_cen_stellar_mass_arr, wtd_blue_sigma_arr,
        wtd_blue_cen_stellar_mass_arr, wtd_red_nsat_arr, wtd_blue_nsat_arr]

def get_best_fit_model(best_fit_params, best_fit_mocknum=None):
    """
    Get SMF and SMHM information of best fit model given a survey

    Parameters
    ----------
    best_fit_params: array
        Array of parameter values corresponding to the best-fit model
    
    best_fit_mocknum (optional): int
        Mock number corresponding to the best-fit model. Defaults to None.

    Returns
    ---------
    max_total: array
        Array of x-axis mass values for all galaxies

    phi_total: array
        Array of y-axis phi values for all galaxies

    maxis_fblue: array
        Array of x_axis mass values for bleu fraction measurement

    f_blue: array
        Array of blue fraction values

    phi_red_model: array
        Array of y-axis phi values for red galaxies

    phi_blue_model: array
        Array of y-axis phi values for blue galaxies

    cen_gals_red: array
        Array of red central galaxy masses

    cen_halos_red: array
        Array of red central halo masses

    cen_gals_blue: array
        Array of blue central galaxy masses
    
    cen_halos_blue: array
        Array of blue central halo masses

    f_red_cen_red: array
        Array of red fractions for red central galaxies

    f_red_cen_blue: array
        Array of red fractions for blue central galaxies

    sat_gals_red: array
        Array of red satellite galaxy masses

    sat_halos_red: array
        Array of red satellite host halo masses

    sat_gals_blue: array
        Array of blue satellite galaxy masses

    sat_halos_blue: array
        Array of blue satellite host halo masses

    f_red_sat_red: array
        Array of red fractions for red satellite galaxies

    f_red_sat_blue: array
        Array of red fractions for blue satellite galaxies

    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses

    red_sigma: array
        Array of velocity dispersion of galaxies with red group centrals

    grp_red_cen_stellar_mass: array
        Array of red group central stellar masses

    blue_sigma: array
        Array of velocity dispersion of galaxies with blue group centrals

    grp_blue_cen_stellar_mass: array
        Array of blue group central stellar masses

    """   
    if best_fit_mocknum:
        cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', \
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
    else:
        # gals_df = populate_mock(best_fit_params[:5], model_init)
        # gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].\
        #     reset_index(drop=True)
        # gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        #     gals_df['halo_id'], 1, 0)

        # cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        #     'stellar_mass']
        # gals_df = gals_df[cols_to_use]
        # gals_df.stellar_mass = np.log10(gals_df.stellar_mass)

        randint_logmstar = 1
        cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
        'halo_mvir_host_halo', 'cz', 'cs_flag', \
        'behroozi_bf', \
        'g_galtype_{0}'.format(randint_logmstar), \
        'groupid_{0}'.format(randint_logmstar)]

        gals_df = gal_group_df_subset[cols_to_use]

        gals_df = gals_df.dropna(subset=['g_galtype_{0}'.\
        format(randint_logmstar),'groupid_{0}'.format(randint_logmstar)]).\
        reset_index(drop=True)

        gals_df[['g_galtype_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar)]] = \
            gals_df[['g_galtype_{0}'.format(randint_logmstar),\
            'groupid_{0}'.format(randint_logmstar)]].astype(int)

    # Stellar masses in log but halo masses not in log
    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(best_fit_params[5:], gals_df, 
            'vishnu', randint_logmstar)
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(best_fit_params[5:], gals_df, 
            'vishnu')      
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
    # v_sim = 130**3 
    v_sim = 890641.5172927063 

    ## Observable #1 - Total SMF
    total_model = measure_all_smf(gals_df, v_sim , False, randint_logmstar)    
    ## Observable #2 - Blue fraction
    f_blue = blue_frac(gals_df, True, False, randint_logmstar)

    cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
        cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
            get_centrals_mock(gals_df, randint_logmstar)
    sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
        f_red_sat_red, f_red_sat_blue = \
            get_satellites_mock(gals_df, randint_logmstar)

    phi_red_model, phi_blue_model = \
        get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
        True, randint_logmstar)

    max_total = total_model[0]
    phi_total = total_model[1]
    max_fblue = f_blue[0]
    fblue = f_blue[1]

    if level == 'group':
        ## Group level
        red_sigma, grp_red_cen_stellar_mass, blue_sigma, \
            grp_blue_cen_stellar_mass = get_sigma_per_group_vishnu_qmcolour(gals_df, 
            randint_logmstar)

        ## Group level
        vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
            vdisp_centers_blue_model = \
            get_deltav_sigma_vishnu_qmcolour(gals_df, randint_logmstar)

        ## Group level
        red_cen_stellar_mass_model, red_num_model, blue_cen_stellar_mass_model, \
            blue_num_model = \
            get_N_per_group_vishnu_qmcolour(gals_df, randint_logmstar, \
                central_bool=True)

        wtd_red_sigma, wtd_red_cen_stellar_mass, wtd_blue_sigma, \
            wtd_blue_cen_stellar_mass, wtd_red_nsat, wtd_blue_nsat = \
            get_satellite_weighted_sigma_group_vishnu(gals_df, randint_logmstar)

    elif level == 'halo':
        ## Halo level
        red_sigma, grp_red_cen_stellar_mass, blue_sigma, \
            grp_blue_cen_stellar_mass = get_sigma_per_halo_vishnu_qmcolour(gals_df, 
            randint_logmstar)

        ## Halo level
        vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
            vdisp_centers_blue_model, vdisp_red, \
            vdisp_blue, red_host_halo_mass_sigma_mh, blue_host_halo_mass_sigma_mh = \
            get_deltav_sigma_halo_vishnu_qmcolour(gals_df, randint_logmstar)

        ## Halo level
        red_cen_stellar_mass_model, red_num_model, blue_cen_stellar_mass_model, \
            blue_num_model, red_host_halo_mass_N_mh, blue_host_halo_mass_N_mh = \
            get_N_per_halo_vishnu_qmcolour(gals_df, randint_logmstar, \
                central_bool=True)

        wtd_red_sigma, wtd_red_cen_stellar_mass, wtd_blue_sigma, \
            wtd_blue_cen_stellar_mass, wtd_red_nsat, wtd_blue_nsat = \
            get_satellite_weighted_sigma_halo_vishnu(gals_df, randint_logmstar)

        return max_total, phi_total, max_fblue, fblue, phi_red_model, \
            phi_blue_model, cen_gals_red, cen_halos_red,\
            cen_gals_blue, cen_halos_blue, f_red_cen_red, f_red_cen_blue, \
            sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, f_red_sat_red, \
            f_red_sat_blue, cen_gals, cen_halos, red_sigma, \
            grp_red_cen_stellar_mass, blue_sigma, grp_blue_cen_stellar_mass, \
            vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
            vdisp_centers_blue_model, vdisp_red, vdisp_blue, \
            red_host_halo_mass_sigma_mh, blue_host_halo_mass_sigma_mh, \
            red_cen_stellar_mass_model, red_num_model, blue_cen_stellar_mass_model,\
            blue_num_model, red_host_halo_mass_N_mh, blue_host_halo_mass_N_mh, \
            wtd_red_sigma, wtd_red_cen_stellar_mass, wtd_blue_sigma, \
            wtd_blue_cen_stellar_mass, wtd_red_nsat, wtd_blue_nsat


    return max_total, phi_total, max_fblue, fblue, phi_red_model, \
        phi_blue_model, cen_gals_red, cen_halos_red,\
        cen_gals_blue, cen_halos_blue, f_red_cen_red, f_red_cen_blue, \
        sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, f_red_sat_red, \
        f_red_sat_blue, cen_gals, cen_halos, red_sigma, \
        grp_red_cen_stellar_mass, blue_sigma, grp_blue_cen_stellar_mass, \
        vdisp_red_model, vdisp_blue_model, vdisp_centers_red_model, \
        vdisp_centers_blue_model, red_cen_stellar_mass_model, red_num_model, \
        blue_cen_stellar_mass_model, blue_num_model, wtd_red_sigma, \
        wtd_red_cen_stellar_mass, wtd_blue_sigma, wtd_blue_cen_stellar_mass, \
        wtd_red_nsat, wtd_blue_nsat

def get_colour_smf_from_fblue(df, frac_arr, bin_centers, volume, h1_bool, 
    randint_logmstar=None):
    """Reconstruct red and blue SMFs from blue fraction measurement

    Args:
        df (pandas.DataFrame): Data/Mock
        frac_arr (array): Array of blue fraction values
        bin_centers (array): Array of x-axis stellar mass bin center values
        volume (float): Volume of data/mock
        h1_bool (boolean): True if masses in h=1.0, False if not in h=1.0
        randint_logmstar (int, optional): Mock number in the case where many
        Behroozi mocks were used. Defaults to None.

    Returns:
        phi_red (array): Array of phi values for red galaxies
        phi_blue (array): Array of phi values for blue galaxies
    """
    if h1_bool and randint_logmstar != 1:
        logmstar_arr = df['{0}'.format(randint_logmstar)].values
    elif h1_bool and randint_logmstar == 1:
        logmstar_arr = df['behroozi_bf'].values
    if not h1_bool:
        mstar_arr = df.logmstar.values
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)

    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = bin_centers - (0.5 * bin_width)
    # Done to include the right side of the last bin
    bin_edges = np.insert(bin_edges, len(bin_edges), bin_edges[-1]+bin_width)
    counts, edg = np.histogram(logmstar_arr, bins=bin_edges)  
    
    counts_blue = frac_arr * counts
    counts_red = (1-frac_arr) * counts

    # Normalized by volume and bin width
    phi_red = counts_red / (volume * bin_width)  # not a log quantity
    phi_red = np.log10(phi_red)

    phi_blue = counts_blue / (volume * bin_width)  # not a log quantity
    phi_blue = np.log10(phi_blue)

    ## Check to make sure that the reconstruced mass functions are the same as 
    ## those from diff_smf(). They aren't exactly but they match up if the 
    ## difference in binning is corrected.
    # fig1 = plt.figure()
    # plt.plot(bin_centers, phi_red, 'r+', ls='--', label='reconstructed')
    # plt.plot(bin_centers, phi_blue, 'b+', ls='--', label='reconstructed')
    # plt.plot(red_data[0], red_data[1], 'r+', ls='-', label='measured')
    # plt.plot(blue_data[0], blue_data[1], 'b+', ls='-', label='measured')
    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)
    # plt.legend()
    # plt.show()

    return phi_red, phi_blue

def get_sigma_per_group_data(catl):
    """Calculating velocity dispersion of groups from real data

    Args:
        catl (pandas.DataFrame): Data catalogue 

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Group blue central stellar mass

    """
    if survey == 'eco' or survey == 'resolvea':
        catl = catl.loc[catl.logmstar >= 8.9]
    elif survey == 'resolveb':
        catl = catl.loc[catl.logmstar >= 8.7]
    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    red_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'R') & (catl.fc == 1)].values)  
    blue_subset_grpids = np.unique(catl.grp.loc[(catl.\
        colour_label == 'B') & (catl.fc == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    red_sigmagapper_arr = []
    red_sigmagapperstacked_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.grp == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.fc\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.fc == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            red_sigmagapper = np.unique(group.grpsig.values)[0]
            red_sigmagapperstacked = np.unique(group.grpsig_stack.values)[0]
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_sigmagapper_arr.append(red_sigmagapper)
            red_sigmagapperstacked_arr.append(red_sigmagapperstacked)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_sigmagapper_arr = []
    blue_sigmagapperstacked_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.grp == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.fc\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.fc == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            blue_sigmagapper = np.unique(group.grpsig.values)[0]
            blue_sigmagapperstacked = np.unique(group.grpsig_stack.values)[0]
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_sigmagapper_arr.append(blue_sigmagapper)
            blue_sigmagapperstacked_arr.append(blue_sigmagapperstacked)

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

def get_sigma_per_group_mocks_qmcolour(survey, mock_df):
    """
    Calculate velocity dispersion from survey mocks 

    Parameters
    ----------
    survey: string
        Name of survey

    mock_df: string
        Mock catalogue

    Returns
    ---------
    mean_stats_red: numpy array
        Average red group central stellar mass in bins of velocity dispersion

    centers_red: numpy array
        Bin centers of velocity dispersion of galaxies around red centrals

    mean_stats_blue: numpy array
        Average blue group central stellar mass in bins of velocity dispersion

    centers_blue: numpy array
        Bin centers of velocity dispersion of galaxies around blue centrals
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    red_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.g_galtype == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_df.loc[mock_df.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            cz_grp = np.unique(group.grpcz.values)[0]

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
        group = mock_df.loc[mock_df.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            cz_grp = np.unique(group.grpcz.values)[0]

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

def get_sigma_per_group_vishnu_qmcolour(gals_df, randint=None):
    """
    Calculate velocity dispersion from Vishnu mock 

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'R') & (gals_df[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'B') & (gals_df[g_galtype_col] == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = gals_df.loc[gals_df[groupid_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
                values == 1].values[0]
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[g_galtype_col].values == 1].values[0]
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
        group = gals_df.loc[gals_df[groupid_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
                values == 1].values[0]
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[g_galtype_col].values == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

def get_sigma_per_halo_mocks_qmcolour(survey, mock_df):
    """
    Calculate velocity dispersion of halos from survey mocks

    Parameters
    ----------
    survey: string
        Name of survey

    mock_df: string
        Mock catalogue

    Returns
    ---------
    mean_stats_red: numpy array
        Average red group central stellar mass in bins of velocity dispersion

    centers_red: numpy array
        Bin centers of velocity dispersion of galaxies around red centrals

    mean_stats_blue: numpy array
        Average blue group central stellar mass in bins of velocity dispersion

    centers_blue: numpy array
        Bin centers of velocity dispersion of galaxies around blue centrals
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    ## Halo ID is equivalent to halo_hostid in vishnu mock
    red_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.cs_flag == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        if len(halo) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = halo.logmstar.loc[halo.cs_flag\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_halo = np.round(np.mean(halo.cz.values),2)
            cen_cz_halo = halo.cz.loc[halo.cs_flag == 1].values[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[mean_cz_halo]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        if len(halo) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = halo.logmstar.loc[halo.cs_flag\
                .values == 1].values[0]
            
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            cen_cz_grp = halo.cz.loc[halo.cs_flag == 1].values[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
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

def get_sigma_per_halo_vishnu_qmcolour(gals_df, randint=None):
    """
    Calculate velocity dispersion of halos from Vishnu mock (logmstar 
    already in h=1)

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'R') & (gals_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'B') & (gals_df.cs_flag == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        if len(halo) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
                values == 1].values[0]
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            cen_cz_grp = halo.cz.loc[halo.cs_flag.values == 1].values[0]
            # cz_grp = np.unique(halo.grpcz.values)[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        if len(halo) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
                values == 1].values[0]
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            cen_cz_grp = halo.cz.loc[halo.cs_flag.values == 1].values[0]
            # cz_grp = np.unique(halo.grpcz.values)[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

def get_satellite_weighted_sigma_halo_vishnu(gals_df, randint=None):
    """
    Calculate velocity dispersion of halos from Vishnu mock (logmstar 
    already in h=1)

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'R') & (gals_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'B') & (gals_df.cs_flag == 1)].values)

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    red_nsat_arr = []
    for key in red_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        if len(halo) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
                values == 1].values[0]
            nsat = len(halo.loc[halo.cs_flag.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            cen_cz_grp = halo.cz.loc[halo.cs_flag.values == 1].values[0]
            # cz_grp = np.unique(halo.grpcz.values)[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_nsat_arr.append(nsat)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_nsat_arr = []
    for key in blue_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        if len(halo) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
                values == 1].values[0]
            nsat = len(halo.loc[halo.cs_flag.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            cen_cz_grp = halo.cz.loc[halo.cs_flag.values == 1].values[0]
            # cz_grp = np.unique(halo.grpcz.values)[0]

            # Velocity difference
            deltav = halo.cz.values - len(halo)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_nsat_arr.append(nsat)
            

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr, red_nsat_arr, blue_nsat_arr

def get_satellite_weighted_sigma_group_vishnu(gals_df, randint=None):
    """
    Calculate velocity dispersion of halos from Vishnu mock (logmstar 
    already in h=1)

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'R') & (gals_df[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'B') & (gals_df[g_galtype_col] == 1)].values)


    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    red_nsat_arr = []
    for key in red_subset_grpids: 
        group = gals_df.loc[gals_df[groupid_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
                values == 1].values[0]
            nsat = len(group.loc[group[g_galtype_col].values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[g_galtype_col].values == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_nsat_arr.append(nsat)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_nsat_arr = []
    for key in blue_subset_grpids: 
        group = gals_df.loc[gals_df[groupid_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
                values == 1].values[0]
            nsat = len(group.loc[group[g_galtype_col].values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[g_galtype_col].values == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_nsat_arr.append(nsat)            

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr, red_nsat_arr, blue_nsat_arr

def get_satellite_weighted_sigma_data(catl):
    """
    Calculate velocity dispersion of halos from Vishnu mock (logmstar 
    already in h=1)

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
    """
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
    red_nsat_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            nsat = len(group.loc[group.g_galtype.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_nsat_arr.append(nsat)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_nsat_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            nsat = len(group.loc[group.g_galtype.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_nsat_arr.append(nsat)
            
    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr, red_nsat_arr, blue_nsat_arr

def get_satellite_weighted_sigma_mocks(catl):
    """
    Calculate velocity dispersion of halos from ECO mocks

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    red_sigma_arr: numpy array
        Velocity dispersion of galaxies around red centrals

    red_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of red galaxies

    blue_sigma_arr: numpy array
        Velocity dispersion of galaxies around blue centrals

    blue_cen_stellar_mass_arr: numpy array
        Array of central stellar mass of blue galaxies
    """
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
    red_nsat_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            nsat = len(group.loc[group.g_galtype.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_nsat_arr.append(nsat)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_nsat_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            nsat = len(group.loc[group.g_galtype.values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group.g_galtype == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_nsat_arr.append(nsat)
            
    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr, red_nsat_arr, blue_nsat_arr

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
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        cen_deltav_arr = []
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])
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

def get_deltav_sigma_data(catl):
    """
    Measure velocity dispersion separately for red and blue galaxies 
    by binning up central stellar mass (changes logmstar units from h=0.7 to h=1)

    Parameters
    ----------
    df: pandas Dataframe 
        Data catalog

    Returns
    ---------
    std_red: numpy array
        Velocity dispersion of red galaxies
    centers_red: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue: numpy array
        Velocity dispersion of blue galaxies
    centers_blue: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    if survey == 'eco' or survey == 'resolvea':
        catl = catl.loc[catl.logmstar >= 8.9]
    elif survey == 'resolveb':
        catl = catl.loc[catl.logmstar >= 8.7]
        
    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
   
    red_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'R') & (catl.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'B') & (catl.g_galtype == 1)].values)


    # Calculating velocity dispersion for galaxies in groups with a 
    # red central
    red_singleton_counter = 0
    # red_deltav_arr = []
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
                values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     red_deltav_arr.append(val)
            #     red_cen_stellar_mass_arr.append(cen_stellar_mass)

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    # std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    #     red_deltav_arr)
    # std_red = np.array(std_red)

    mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
        statistic='mean', bins=red_stellar_mass_bins)
    std_red = mean_stats_red[0]

    # Calculating velocity dispersion for galaxies in groups with a 
    # blue central
    blue_singleton_counter = 0
    # blue_deltav_arr = []
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     blue_deltav_arr.append(val)
            #     blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    # std_blue = std_func(blue_stellar_mass_bins, blue_cen_stellar_mass_arr, 
    #     blue_deltav_arr)    
    # std_blue = np.array(std_blue)

    mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
        statistic='mean', bins=blue_stellar_mass_bins)
    std_blue = mean_stats_blue[0]
    # centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    #     red_stellar_mass_bins[:-1])
    # centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    #     blue_stellar_mass_bins[:-1])

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    return std_red, centers_red, std_blue, centers_blue

def get_deltav_sigma_mocks_qmcolour(survey, mock_df):
    """
    Calculate velocity dispersion from survey mocks (logmstar converted
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
        Velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Velocity dispersion of blue galaxies
    centers_blue_arr: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    red_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.g_galtype == 1)].values)

    # Calculating velocity dispersion for galaxies in groups
    # with a red central
    red_singleton_counter = 0
    # red_deltav_arr = []
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_df.loc[mock_df.groupid == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
                values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     red_deltav_arr.append(val)
            #     red_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(red_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)

    # std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    #     red_deltav_arr)
    # std_red = np.array(std_red)

    mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
        statistic='mean', bins=red_stellar_mass_bins)
    std_red = mean_stats_red[0]

    # Calculating velocity dispersion for galaxies in groups 
    # with a blue central
    blue_singleton_counter = 0
    # blue_deltav_arr = []
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = mock_df.loc[mock_df.groupid == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                .values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)

            # for val in deltav:
            #     blue_deltav_arr.append(val)
            #     blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(blue_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)

    # std_blue = std_func(blue_stellar_mass_bins, \
    #     blue_cen_stellar_mass_arr, blue_deltav_arr)    
    # std_blue = np.array(std_blue)

    mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
        statistic='mean', bins=blue_stellar_mass_bins)
    std_blue = mean_stats_blue[0]

    # centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    #     red_stellar_mass_bins[:-1])
    # centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    #     blue_stellar_mass_bins[:-1])

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    centers_red = np.array(centers_red)
    centers_blue = np.array(centers_blue)
            
    return std_red, std_blue, centers_red, centers_blue

def get_deltav_sigma_vishnu_qmcolour(mock_df, randint=None):
    """
    Calculate velocity dispersion from Vishnu mock (logmstar already 
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
        Velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Velocity dispersion of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.grpcz.values >= min_cz) & \
            (mock_df.grpcz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.grpcz.values >= min_cz) & \
            (mock_df.grpcz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.grpcz.values >= min_cz) & \
            (mock_df.grpcz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        mock_df[logmstar_col] = np.log10(mock_df[logmstar_col])

    red_subset_grpids = np.unique(mock_df[groupid_col].loc[(mock_df.\
        colour_label == 'R') & (mock_df[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(mock_df[groupid_col].loc[(mock_df.\
        colour_label == 'B') & (mock_df[g_galtype_col] == 1)].values)

    # Calculating velocity dispersion for galaxies in groups
    # with a red central
    red_singleton_counter = 0
    # red_deltav_arr = []
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_df.loc[mock_df[groupid_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            if randint != 1:
                cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col].\
                    values == 1].values[0]
            elif randint == 1:
                cen_stellar_mass = group['behroozi_bf'].loc[group[g_galtype_col].\
                    values == 1].values[0]
            else:
                cen_stellar_mass = group['stellar_mass'].loc[group[g_galtype_col].\
                    values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     red_deltav_arr.append(val)
            #     red_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(red_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    # std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    #     red_deltav_arr)
    # std_red = np.array(std_red)

    mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
        statistic='mean', bins=red_stellar_mass_bins)
    std_red = mean_stats_red[0]

    # Calculating velocity dispersion for galaxies in groups 
    # with a blue central
    blue_singleton_counter = 0
    # blue_deltav_arr = []
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = mock_df.loc[mock_df[groupid_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            if randint != 1:
                cen_stellar_mass = group['{0}'.format(randint)].loc[group[g_galtype_col].\
                    values == 1].values[0]
            elif randint == 1:
                cen_stellar_mass = group['behroozi_bf'].loc[group[g_galtype_col].\
                    values == 1].values[0]
            else:
                cen_stellar_mass = group['stellar_mass'].loc[group[g_galtype_col].\
                    values == 1].values[0]
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            deltav = group.cz.values - len(group)*[mean_cz_grp]
            sigma = deltav.std()
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     blue_deltav_arr.append(val)
            #     blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(blue_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    # std_blue = std_func(blue_stellar_mass_bins, \
    #     blue_cen_stellar_mass_arr, blue_deltav_arr)    
    # std_blue = np.array(std_blue)

    mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
        statistic='mean', bins=blue_stellar_mass_bins)
    std_blue = mean_stats_blue[0]

    # centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    #     red_stellar_mass_bins[:-1])
    # centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    #     blue_stellar_mass_bins[:-1])

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])
            
    return std_red, std_blue, centers_red, centers_blue

def get_deltav_sigma_halo_mocks_qmcolour(survey, mock_df):
    """
    Calculate velocity dispersion from survey mocks (logmstar converted
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
        Velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Velocity dispersion of blue galaxies
    centers_blue_arr: numpy array
        Bin centers of central stellar mass for blue galaxies
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    ## Halo ID is equivalent to halo_hostid in vishnu mock
    red_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.cs_flag == 1)].values)

    # Calculating velocity dispersion for galaxies in groups
    # with a red central
    red_singleton_counter = 0
    # red_deltav_arr = []
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        if len(halo) == 1:
            red_singleton_counter += 1
        else:
            cen_stellar_mass = halo.logmstar.loc[halo.cs_flag.\
                values == 1].values[0]
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            sigma = deltav.std()
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     red_deltav_arr.append(val)
            #     red_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(red_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    # std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    #     red_deltav_arr)
    # std_red = np.array(std_red)

    mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
        statistic='mean', bins=red_stellar_mass_bins)
    std_red = mean_stats_red[0]

    # Calculating velocity dispersion for galaxies in groups 
    # with a blue central
    blue_singleton_counter = 0
    # blue_deltav_arr = []
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        if len(halo) == 1:
            blue_singleton_counter += 1
        else:
            cen_stellar_mass = halo.logmstar.loc[halo.cs_flag\
                .values == 1].values[0]
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            sigma = deltav.std()
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            # for val in deltav:
            #     blue_deltav_arr.append(val)
            #     blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(blue_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    # std_blue = std_func(blue_stellar_mass_bins, \
    #     blue_cen_stellar_mass_arr, blue_deltav_arr)    
    # std_blue = np.array(std_blue)

    mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
        statistic='mean', bins=blue_stellar_mass_bins)
    std_blue = mean_stats_blue[0]

    # centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    #     red_stellar_mass_bins[:-1])
    # centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    #     blue_stellar_mass_bins[:-1])

    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])
                        
    centers_red = np.array(centers_red)
    centers_blue = np.array(centers_blue)
            
    return std_red, std_blue, centers_red, centers_blue

def get_deltav_sigma_halo_vishnu_qmcolour(mock_df, randint=None):
    """
    Calculate velocity dispersion from Vishnu mock (logmstar already 
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
        Velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Velocity dispersion of blue galaxies
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.cz.values >= min_cz) & \
            (mock_df.cz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.cz.values >= min_cz) & \
            (mock_df.cz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        mock_df = mock_add_grpcz(mock_df, False, groupid_col)
        mock_df = mock_df.loc[(mock_df.grpcz.values >= min_cz) & \
            (mock_df.grpcz.values <= max_cz) & \
            (mock_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        mock_df[logmstar_col] = np.log10(mock_df[logmstar_col])

    red_subset_haloids = np.unique(mock_df.halo_hostid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(mock_df.halo_hostid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.cs_flag == 1)].values)

    # Calculating velocity dispersion of satellites per host halo
    # with a red central
    red_singleton_counter = 0
    # red_deltav_arr = []
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    red_host_halo_mass_arr = []
    for key in red_subset_haloids: 
        halo = mock_df.loc[mock_df.halo_hostid == key]
        if len(halo) == 1:
            red_singleton_counter += 1
        else:
            if randint != 1:
                cen_stellar_mass = halo['{0}'.format(randint)].loc[halo.cs_flag.\
                    values == 1].values[0]
            elif randint == 1:
                cen_stellar_mass = halo['behroozi_bf'].loc[halo.cs_flag.\
                    values == 1].values[0]
            else:
                cen_stellar_mass = halo['stellar_mass'].loc[halo.cs_flag.\
                    values == 1].values[0]
            host_halo_mass = np.unique(halo.halo_mvir_host_halo.values)[0]
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            sigma = deltav.std()
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_host_halo_mass_arr.append(host_halo_mass)
            # for val in deltav:
            #     red_deltav_arr.append(val)
            #     red_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(red_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    elif survey == 'resolveb':
        red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    # std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
    #     red_deltav_arr)
    # std_red = np.array(std_red)

    mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
        statistic='mean', bins=red_stellar_mass_bins)
    std_red = mean_stats_red[0]

    # Calculating velocity dispersion of satellites per host halo
    # with a blue central
    blue_singleton_counter = 0
    # blue_deltav_arr = []
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_host_halo_mass_arr = []
    for key in blue_subset_haloids: 
        halo = mock_df.loc[mock_df.halo_hostid == key]
        if len(halo) == 1:
            blue_singleton_counter += 1
        else:
            if randint != 1:
                cen_stellar_mass = halo['{0}'.format(randint)].loc[halo.cs_flag.\
                    values == 1].values[0]
            elif randint == 1:
                cen_stellar_mass = halo['behroozi_bf'].loc[halo.cs_flag.\
                    values == 1].values[0]
            else:
                cen_stellar_mass = halo['stellar_mass'].loc[halo.cs_flag.\
                    values == 1].values[0]
            host_halo_mass = np.unique(halo.halo_mvir_host_halo.values)[0]
            mean_cz_grp = np.round(np.mean(halo.cz.values),2)
            deltav = halo.cz.values - len(halo)*[mean_cz_grp]
            sigma = deltav.std()
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_host_halo_mass_arr.append(host_halo_mass)
            # for val in deltav:
            #     blue_deltav_arr.append(val)
            #     blue_cen_stellar_mass_arr.append(cen_stellar_mass)
    # print(max(blue_cen_stellar_mass_arr))

    if survey == 'eco' or survey == 'resolvea':
        # TODO : check if this is actually correct for resolve a
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
    elif survey == 'resolveb':
        blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    # std_blue = std_func(blue_stellar_mass_bins, \
    #     blue_cen_stellar_mass_arr, blue_deltav_arr)    
    # std_blue = np.array(std_blue)

    mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
        statistic='mean', bins=blue_stellar_mass_bins)
    std_blue = mean_stats_blue[0]


    # centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
    #     red_stellar_mass_bins[:-1])
    # centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
    #     blue_stellar_mass_bins[:-1])
            
    centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    return std_red, std_blue, centers_red, centers_blue, red_sigma_arr, \
        blue_sigma_arr, red_host_halo_mass_arr, blue_host_halo_mass_arr

def get_N_per_group_data(catl, central_bool=None):
    """Calculating velocity dispersion of groups from real data

    Args:
        catl (pandas.DataFrame): Data catalogue 

        central_bool (Boolean): True if central is to be included in count ; 
            False if central is to be excluded in count

    Returns:
        red_num_arr (numpy array): Number of galaxies in groups with red centrals

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_num_arr (numpy array): Number of galaxies in groups with blue centrals
        
        blue_cen_stellar_mass_arr (numpy array): Group blue central stellar mass

    """
    if survey == 'eco' or survey == 'resolvea':
        catl = catl.loc[catl.logmstar >= 8.9]
    elif survey == 'resolveb':
        catl = catl.loc[catl.logmstar >= 8.7]
    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    red_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'R') & (catl.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(catl.groupid.loc[(catl.\
        colour_label == 'B') & (catl.g_galtype == 1)].values)

    red_num_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype\
            .values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        red_cen_stellar_mass_arr.append(cen_stellar_mass)
        red_num_arr.append(num)
            
    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = catl.loc[catl.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype\
            .values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        blue_num_arr.append(num)

    return red_num_arr, red_cen_stellar_mass_arr, blue_num_arr, \
        blue_cen_stellar_mass_arr

def get_N_per_group_mocks_qmcolour(survey, mock_df, central_bool=None):
    """
    Calculate velocity dispersion from survey mocks 

    Parameters
    ----------
    survey: string
        Name of survey

    mock_df: string
        Mock catalogue

    central_bool: Boolean
        True if central is to be included in count
        False if central is to be excluded in count


    Returns
    ---------
    red_cen_stellar_mass_arr: numpy array
        Red group central stellar mass

    red_num_arr: numpy array
        Number of galaxies around red group centrals

    blue_cen_stellar_mass_arr: numpy array
        Blue group central stellar mass

    blue_num_arr: numpy array
        Number of galaxies around blue group centrals
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    red_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.g_galtype == 1)].values)  
    blue_subset_grpids = np.unique(mock_df.groupid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.g_galtype == 1)].values)

    red_num_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = mock_df.loc[mock_df.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype\
            .values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        red_cen_stellar_mass_arr.append(cen_stellar_mass)
        red_num_arr.append(num)

    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = mock_df.loc[mock_df.groupid == key]
        cen_stellar_mass = group.logmstar.loc[group.g_galtype\
            .values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        blue_num_arr.append(num)

    # mean_stats_red = bs(red_sigma_arr, red_cen_stellar_mass_arr, 
    #     statistic='mean', bins=np.linspace(0,250,6))
    # mean_stats_blue = bs(blue_sigma_arr, blue_cen_stellar_mass_arr, 
    #     statistic='mean', bins=np.linspace(0,250,6))

    # centers_red = 0.5 * (mean_stats_red[1][1:] + \
    #     mean_stats_red[1][:-1])
    # centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
    #     mean_stats_blue[1][:-1])
            
    return red_cen_stellar_mass_arr, red_num_arr, blue_cen_stellar_mass_arr, \
        blue_num_arr

def get_N_per_group_vishnu_qmcolour(gals_df, randint=None, central_bool=None):
    """
    Calculate velocity dispersion from Vishnu mock 

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    central_bool: Boolean
        True if central is to be included in count
        False if central is to be excluded in count

    Returns
    ---------
    red_cen_stellar_mass_arr: numpy array
        Red group central stellar mass

    red_num_arr: numpy array
        Number of galaxies around red group centrals

    blue_cen_stellar_mass_arr: numpy array
        Blue group central stellar mass

    blue_num_arr: numpy array
        Number of galaxies around blue group centrals
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'R') & (gals_df[g_galtype_col] == 1)].values)  
    blue_subset_grpids = np.unique(gals_df[groupid_col].loc[(gals_df.\
        colour_label == 'B') & (gals_df[g_galtype_col] == 1)].values)

    red_num_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_grpids: 
        group = gals_df.loc[gals_df[groupid_col] == key]
        cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
            values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        red_num_arr.append(num)
        red_cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_grpids: 
        group = gals_df.loc[gals_df[groupid_col] == key]
        cen_stellar_mass = group[logmstar_col].loc[group[g_galtype_col].\
            values == 1].values[0]
        if central_bool:
            num = len(group)
        elif not central_bool:
            num = len(group) - 1
        blue_num_arr.append(num)
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    return red_cen_stellar_mass_arr, red_num_arr, blue_cen_stellar_mass_arr, \
        blue_num_arr

def get_N_per_halo_mocks_qmcolour(survey, mock_df, central_bool=None):
    """
    Calculate velocity dispersion of halos from survey mocks

    Parameters
    ----------
    survey: string
        Name of survey

    mock_df: string
        Mock catalogue

    central_bool: Boolean
        True if central is to be included in count
        False if central is to be excluded in count

    Returns
    ---------
    red_cen_stellar_mass_arr: numpy array
        Red group central stellar mass

    red_num_arr: numpy array
        Number of galaxies around red group centrals

    blue_cen_stellar_mass_arr: numpy array
        Blue group central stellar mass

    blue_num_arr: numpy array
        Number of galaxies around blue group centrals
    """
    mock_df.logmstar = np.log10((10**mock_df.logmstar) / 2.041)
    ## Halo ID is equivalent to halo_hostid in vishnu mock
    red_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'R') & (mock_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(mock_df.haloid.loc[(mock_df.\
        colour_label == 'B') & (mock_df.cs_flag == 1)].values)

    red_num_arr = []
    red_cen_stellar_mass_arr = []
    for key in red_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        cen_stellar_mass = halo.logmstar.loc[halo.cs_flag\
            .values == 1].values[0]
        if central_bool:
            num = len(halo)
        elif not central_bool:
            num = len(halo) - 1
        red_num_arr.append(num)
        red_cen_stellar_mass_arr.append(cen_stellar_mass)

    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    for key in blue_subset_haloids: 
        halo = mock_df.loc[mock_df.haloid == key]
        cen_stellar_mass = halo.logmstar.loc[halo.cs_flag\
            .values == 1].values[0]
        if central_bool:
            num = len(halo)
        elif not central_bool:
            num = len(halo) - 1          
        blue_num_arr.append(num)
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)

    # mean_stats_red = bs(red_sigma_arr, red_cen_stellar_mass_arr, 
    #     statistic='mean', bins=np.linspace(0,250,6))
    # mean_stats_blue = bs(blue_sigma_arr, blue_cen_stellar_mass_arr, 
    #     statistic='mean', bins=np.linspace(0,250,6))

    # centers_red = 0.5 * (mean_stats_red[1][1:] + \
    #     mean_stats_red[1][:-1])
    # centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
    #     mean_stats_blue[1][:-1])
            
    return red_cen_stellar_mass_arr, red_num_arr, blue_cen_stellar_mass_arr, \
        blue_num_arr

def get_N_per_halo_vishnu_qmcolour(gals_df, randint=None, central_bool=None):
    """
    Calculate velocity dispersion of halos from Vishnu mock (logmstar 
    already in h=1)

    Parameters
    ----------
    gals_df: pandas.DataFrame
        Mock catalogue

    randint (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    central_bool: Boolean
        True if central is to be included in count
        False if central is to be excluded in count

    Returns
    ---------
    red_cen_stellar_mass_arr: numpy array
        Red group central stellar mass

    red_num_arr: numpy array
        Number of galaxies around red group centrals

    blue_cen_stellar_mass_arr: numpy array
        Blue group central stellar mass

    blue_num_arr: numpy array
        Number of galaxies around blue group centrals
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

    if randint != 1:
        logmstar_col = '{0}'.format(randint)
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    elif randint == 1:
        logmstar_col = 'behroozi_bf'
        g_galtype_col = 'g_galtype_{0}'.format(randint)
        groupid_col = 'groupid_{0}'.format(randint)
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        # gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.cz.values >= min_cz) & \
            (gals_df.cz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

    else:
        logmstar_col = 'stellar_mass'
        g_galtype_col = 'g_galtype'
        groupid_col = 'groupid'
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
        # and M* star cuts to mimic mocks and data.
        gals_df = mock_add_grpcz(gals_df, False, groupid_col)
        gals_df = gals_df.loc[(gals_df.grpcz.values >= min_cz) & \
            (gals_df.grpcz.values <= max_cz) & \
            (gals_df[logmstar_col].values >= (10**mstar_limit)/2.041)]
        gals_df[logmstar_col] = np.log10(gals_df[logmstar_col])

    red_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'R') & (gals_df.cs_flag == 1)].values)  
    blue_subset_haloids = np.unique(gals_df.halo_hostid.loc[(gals_df.\
        colour_label == 'B') & (gals_df.cs_flag == 1)].values)

    red_num_arr = []
    red_cen_stellar_mass_arr = []
    red_host_halo_mass_arr = []
    for key in red_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
            values == 1].values[0]
        host_halo_mass = np.unique(halo.halo_mvir_host_halo.values)[0]
        if central_bool:
            num = len(halo)
        elif not central_bool:
            num = len(halo) - 1
        red_num_arr.append(num)
        red_cen_stellar_mass_arr.append(cen_stellar_mass)
        red_host_halo_mass_arr.append(host_halo_mass)

    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    blue_host_halo_mass_arr = []
    for key in blue_subset_haloids: 
        halo = gals_df.loc[gals_df.halo_hostid == key]
        cen_stellar_mass = halo[logmstar_col].loc[halo.cs_flag.\
            values == 1].values[0]
        host_halo_mass = np.unique(halo.halo_mvir_host_halo.values)[0]
        if central_bool:
            num = len(halo)
        elif not central_bool:
            num = len(halo) - 1
        blue_num_arr.append(num)
        blue_cen_stellar_mass_arr.append(cen_stellar_mass)
        blue_host_halo_mass_arr.append(host_halo_mass)

    return red_cen_stellar_mass_arr, red_num_arr, blue_cen_stellar_mass_arr, \
        blue_num_arr, red_host_halo_mass_arr, blue_host_halo_mass_arr

def plot_total_mf(result, total_data, maxis_bf_total, phi_bf_total,
    bf_chi2, err_colour):
    """
    Plot SMF from data, best fit param values and param values corresponding to 
    68th percentile 100 lowest chi^2 values

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    total_data: multidimensional array
        Array of total SMF information

    maxis_bf_total: array
        Array of x-axis mass values for best-fit SMF

    phi_bf_total: array
        Array of y-axis values for best-fit SMF

    bf_chi2: float
        Chi-squared value associated with best-fit model

    err_colour: array
        Array of error values from matrix

    Returns
    ---------
    Plot displayed on screen.
    """
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    x_phi_total_data, y_phi_total_data = total_data[0], total_data[1]
    x_phi_total_model = result[0][0][0]

    i_outer = 0
    total_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            tot_mod_ii = result[i_outer][1][idx]
            total_mod_arr.append(tot_mod_ii)
        i_outer += 1

    tot_phi_max = np.amax(total_mod_arr, axis=0)
    tot_phi_min = np.amin(total_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mt = plt.fill_between(x=x_phi_total_model, y1=tot_phi_max, 
        y2=tot_phi_min, color='silver', alpha=0.4)

    dt = plt.errorbar(x_phi_total_data, y_phi_total_data, yerr=err_colour[0:6],
        color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bft, = plt.plot(maxis_bf_total, phi_bf_total, color='k', ls='--', lw=4, 
        zorder=10)

    plt.ylim(-4,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.78), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='best')

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')
    plt.show()

def plot_colour_mf(result, phi_red_data, phi_blue_data, phi_bf_red, phi_bf_blue,
    std_red, std_blue, bf_chi2):
    """
    Plot red and blue SMF from data, best fit param values and param values 
    corresponding to 68th percentile 100 lowest chi^2 values

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information

    phi_red_data: array
        Array of y-axis values for red SMF from data

    phi_blue_data: array
        Array of y-axis values for blue SMF from data
    
    phi_bf_red: array
        Array of y-axis values for red SMF from best-fit model

    phi_bf_blue: array
        Array of y-axis values for blue SMF from best-fit model

    std_red: array
        Array of std values per bin of red SMF from mocks

    std_blue: array
        Array of std values per bin of blue SMF from mocks

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    ## The same bins were used for fblue that were used for total SMF
    x_phi_red_model = result[0][2][0]
    x_phi_blue_model = result[0][2][0]

    i_outer = 0
    red_mod_arr = []
    blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][4][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][5][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    red_phi_max = np.amax(red_mod_arr, axis=0)
    red_phi_min = np.amin(red_mod_arr, axis=0)
    blue_phi_max = np.amax(blue_mod_arr, axis=0)
    blue_phi_min = np.amin(blue_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mr = plt.fill_between(x=x_phi_red_model, y1=red_phi_max, 
        y2=red_phi_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=x_phi_blue_model, y1=blue_phi_max, 
        y2=blue_phi_min, color='cornflowerblue',alpha=0.4)

    dr = plt.errorbar(x_phi_red_model, phi_red_data, yerr=std_red,
        color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    db = plt.errorbar(x_phi_blue_model, phi_blue_data, yerr=std_blue,
        color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bfr, = plt.plot(x_phi_red_model, phi_bf_red,
        color='maroon', ls='--', lw=4, zorder=10)
    bfb, = plt.plot(x_phi_blue_model, phi_bf_blue,
        color='mediumblue', ls='--', lw=4, zorder=10)

    plt.ylim(-4,-1)
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

    # plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data','Models','Best-fit'],
    #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
    plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data', 'Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})


    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.78), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    # if survey == 'eco':
    #     plt.title('ECO')
    plt.show()

def plot_fblue(result, fblue_data, maxis_bf_fblue, bf_fblue,
    bf_chi2, err_colour):
    """
    Plot blue fraction from data, best fit param values and param values 
    corresponding to 68th percentile 100 lowest chi^2 values

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    fblue_data: array
        Array of y-axis blue fraction values for data

    maxis_bf_fblue: array
        Array of x-axis mass values for best-fit model

    bf_fblue: array
        Array of y-axis blue fraction values for best-fit model

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    err_colour: array
        Array of error values from matrix

    Returns
    ---------
    Plot displayed on screen.
    """
    class AnyObjectHandler(HandlerBase): 
        # https://stackoverflow.com/questions/31478077/how-to-make-two-markers
        # -share-the-same-label-in-the-legend-using-matplotlib
                                    #AND
        #https://stackoverflow.com/questions/41752309/single-legend-
        # item-with-two-lines
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            # if orig_handle[3]:
            #     topcap_r = plt.Line2D([x0,x0+width*0.2], [0.8*height, 0.8*height], 
            #         linestyle='-', color='darkred') 
            #     body_r = plt.Line2D([x0+width*0.1, x0+width*0.1], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkred')
            #     bottomcap_r = plt.Line2D([x0, x0+width*0.2], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkred')
            #     topcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.8*height, 0.8*height], linestyle='-', color='darkblue') 
            #     body_b = plt.Line2D([x0+width*0.5, x0+width*0.5], \
            #         [0.2*height, 0.8*height], linestyle='-', color='darkblue')
            #     bottomcap_b = plt.Line2D([x0+width*0.4, x0+width*0.6], \
            #         [0.2*height, 0.2*height], linestyle='-', color='darkblue')
            #     return [topcap_r, body_r, bottomcap_r, topcap_b, body_b, bottomcap_b]
            l1 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.6*height, 0.6*height],  
                linestyle=orig_handle[2], color=orig_handle[1])
            return [l1, l2] 

    x_fblue_data, y_fblue_data = fblue_data[0], fblue_data[1]
    x_fblue_model = result[0][2][0]

    i_outer = 0
    fblue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            fblue_mod_ii = result[i_outer][3][idx]
            fblue_mod_arr.append(fblue_mod_ii)
        i_outer += 1

    fblue_max = np.amax(fblue_mod_arr, axis=0)
    fblue_min = np.amin(fblue_mod_arr, axis=0)
    
    fig1= plt.figure(figsize=(10,10))
    mt = plt.fill_between(x=x_fblue_model, y1=fblue_max, 
        y2=fblue_min, color='silver', alpha=0.4)

    dt = plt.errorbar(x_fblue_data, y_fblue_data, yerr=err_colour[6:],
        color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
        capthick=1.5, zorder=10, marker='^')
    # Best-fit
    # Need a comma after 'bfr' and 'bfb' to solve this:
    #   AttributeError: 'NoneType' object has no attribute 'create_artists'
    bft, = plt.plot(maxis_bf_fblue, bf_fblue, color='k', ls='--', lw=4, 
        zorder=10)

    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$f_{blue}$', fontsize=30)

    plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.875, 0.78), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    # if survey == 'eco':
    #     plt.title('ECO')
    plt.show()

def plot_xmhm(result, gals_bf, halos_bf, bf_chi2):
    """
    Plot SMHM from data, best fit param values, param values corresponding to 
    68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    gals_bf: array
        Array of y-axis stellar mass values for best fit SMHM

    halos_bf: array
        Array of x-axis halo mass values for best fit SMHM
    
    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """
    if survey == 'resolvea':
        line_label = 'RESOLVE-A'
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
    elif survey == 'eco':
        line_label = 'ECO'
    
    # x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(halos_bf,\
    # gals_bf,base=0.4,bin_statval='center')

    y_bf,x_bf,binnum = bs(halos_bf,\
    gals_bf,'mean',bins=np.linspace(10, 15, 7))


    i_outer = 0
    mod_x_arr = []
    mod_y_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            mod_x_ii = result[i_outer][19][idx]
            mod_y_ii = result[i_outer][18][idx]
            y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                bins=np.linspace(10, 15, 7))
            mod_x_arr.append(x)
            mod_y_arr.append(y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            mod_x_ii = result[i_outer][19][idx]
            mod_y_ii = result[i_outer][18][idx]
            y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                bins=np.linspace(10, 15, 7))
            mod_x_arr.append(x)
            mod_y_arr.append(y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            mod_x_ii = result[i_outer][19][idx]
            mod_y_ii = result[i_outer][18][idx]
            y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                bins=np.linspace(10, 15, 7))
            mod_x_arr.append(x)
            mod_y_arr.append(y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            mod_x_ii = result[i_outer][19][idx]
            mod_y_ii = result[i_outer][18][idx]
            y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                bins=np.linspace(10, 15, 7))
            mod_x_arr.append(x)
            mod_y_arr.append(y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            mod_x_ii = result[i_outer][19][idx]
            mod_y_ii = result[i_outer][18][idx]
            y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                bins=np.linspace(10, 15, 7))
            mod_x_arr.append(x)
            mod_y_arr.append(y)
        i_outer += 1

    y_max = np.nanmax(mod_y_arr, axis=0)
    y_min = np.nanmin(mod_y_arr, axis=0)

    # for idx in range(len(result[0][0])):
    #     x_model,y_model,y_std_model,y_std_err_model = \
    #         Stats_one_arr(result[0][19][idx],result[0][18][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
    #         zorder=0,label='Models')
    # for idx in range(len(result[1][0])):
    #     x_model,y_model,y_std_model,y_std_err_model = \
    #         Stats_one_arr(result[1][19][idx],result[1][18][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
    #         zorder=1)
    # for idx in range(len(result[2][0])):
    #     x_model,y_model,y_std_model,y_std_err_model = \
    #         Stats_one_arr(result[2][19][idx],result[2][18][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
    #         zorder=2)
    # for idx in range(len(result[3][0])):
    #     x_model,y_model,y_std_model,y_std_err_model = \
    #         Stats_one_arr(result[3][19][idx],result[3][18][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
    #         zorder=3)
    # for idx in range(len(result[4][0])):
    #     x_model,y_model,y_std_model,y_std_err_model = \
    #         Stats_one_arr(result[4][19][idx],result[4][18][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model,y_model,color='lightgray',linestyle='-',alpha=0.5,\
    #         zorder=4)

    fig1 = plt.figure(figsize=(10,10))

    x_cen =  0.5 * (mod_x_arr[0][1:] + mod_x_arr[0][:-1])

    plt.fill_between(x=x_cen, y1=y_max, 
        y2=y_min, color='lightgray',alpha=0.4,label='Models')

    x_cen =  0.5 * (x_bf[1:] + x_bf[:-1])

    plt.plot(x_cen, y_bf, color='k', lw=4, label='Best-fit', zorder=10)

    plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
        [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
        plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
        hatch='\\')

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,14.5)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    if mf_type == 'smf':
        if survey == 'eco' and quenching == 'hybrid':
            plt.ylim(np.log10((10**8.9)/2.041),11.9)
        elif survey == 'eco' and quenching == 'halo':
            plt.ylim(np.log10((10**8.9)/2.041),11.56)
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 30})
    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_colour_xmhm(result, gals_bf_red, halos_bf_red, gals_bf_blue, 
    halos_bf_blue, bf_chi2):
    """
    Plot red and blue SMHM from data, best fit param values, param values 
    corresponding to 68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    gals_bf_red: array
        Array of y-axis stellar mass values for red SMHM for best-fit model

    halos_bf_red: array
        Array of x-axis halo mass values for red SMHM for best-fit model
    
    gals_bf_blue: array
        Array of y-axis stellar mass values for blue SMHM for best-fit model

    halos_bf_blue: array
        Array of x-axis halo mass values for blue SMHM for best-fit model

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """   
    class AnyObjectHandler(HandlerBase): 
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],  
                linestyle=orig_handle[2],
                color=orig_handle[1])
            return [l1, l2]  

    # x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    # gals_bf_red,base=0.4,bin_statval='center')
    # x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    # gals_bf_blue,base=0.4,bin_statval='center')

    y_bf_red,x_bf_red,binnum_red = bs(halos_bf_red,\
    gals_bf_red,'mean',bins=np.linspace(10, 15, 15))
    y_bf_blue,x_bf_blue,binnum_blue = bs(halos_bf_blue,\
    gals_bf_blue,'mean',bins=np.linspace(10, 15, 15))

    # for idx in range(5,20,1):
    #     fig1 = plt.figure(figsize=(16,9))

    #     y_bf_red,x_bf_red,binnum_red = bs(halos_bf_red,\
    #     gals_bf_red,'mean',bins=np.linspace(10, 15, idx))
    #     y_bf_blue,x_bf_blue,binnum_blue = bs(halos_bf_blue,\
    #     gals_bf_blue,'mean',bins=np.linspace(10, 15, idx))

    #     red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
    #     blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

    #     # REMOVED ERROR BAR ON BEST FIT
    #     bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=4,label='Best-fit',zorder=10)
    #     bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=4,
    #         label='Best-fit',zorder=10)

    #     plt.vlines(x_bf_red, ymin=9, ymax=12, colors='r')
    #     plt.vlines(x_bf_blue, ymin=9, ymax=12, colors='b')
            
    #     plt.scatter(halos_bf_red, gals_bf_red, c='r', alpha=0.4)
    #     plt.scatter(halos_bf_blue, gals_bf_blue, c='b', alpha=0.4)

    #     plt.annotate(r'$Number of bins: ${0}'.
    #         format(int(idx)-1), 
    #         xy=(0.02, 0.9), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    #         ec='k', fc='lightgray', alpha=0.5), size=25)

    #     # plt.xlim(10,14.5)
    #     plt.ylim(np.log10((10**8.9)/2.041),11.56)
    #     if quenching == 'halo':
    #         plt.title('Halo quenching model')
    #     elif quenching == 'hybrid':
    #         plt.title('Hybrid quenching model')
    #     plt.xlabel('Halo Mass')
    #     plt.ylabel('Stellar Mass')
    #     plt.show()
    #     plt.savefig('/Users/asadm2/Desktop/shmr_binning/{0}.png'.format(idx))

    i_outer = 0
    red_mod_x_arr = []
    red_mod_y_arr = []
    blue_mod_x_arr = []
    blue_mod_y_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_x_ii = result[i_outer][7][idx]
            red_mod_y_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_x_ii = result[i_outer][9][idx]
            blue_mod_y_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_x_ii = result[i_outer][7][idx]
            red_mod_y_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_x_ii = result[i_outer][9][idx]
            blue_mod_y_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_x_ii = result[i_outer][7][idx]
            red_mod_y_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_x_ii = result[i_outer][9][idx]
            blue_mod_y_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_x_ii = result[i_outer][7][idx]
            red_mod_y_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_x_ii = result[i_outer][9][idx]
            blue_mod_y_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_x_ii = result[i_outer][7][idx]
            red_mod_y_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_x_ii = result[i_outer][9][idx]
            blue_mod_y_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(10, 15, 15))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

    red_y_max = np.nanmax(red_mod_y_arr, axis=0)
    red_y_min = np.nanmin(red_mod_y_arr, axis=0)
    blue_y_max = np.nanmax(blue_mod_y_arr, axis=0)
    blue_y_min = np.nanmin(blue_mod_y_arr, axis=0)

    # for idx in range(len(result[0][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[0][7][idx],result[0][6][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-', \
    #         alpha=0.5, zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[0][9][idx],result[0][8][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[1][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[1][7][idx],result[1][6][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[1][9][idx],result[1][8][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[2][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[2][7][idx],result[2][6][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[2][9][idx],result[2][8][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[3][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[3][7][idx],result[3][6][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[3][9][idx],result[3][8][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[4][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[4][7][idx],result[4][6][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[4][9][idx],result[4][8][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')

    fig1 = plt.figure(figsize=(10,10))

    red_x_cen =  0.5 * (red_mod_x_arr[0][1:] + red_mod_x_arr[0][:-1])
    blue_x_cen = 0.5 * (blue_mod_x_arr[0][1:] + blue_mod_x_arr[0][:-1])

    mr = plt.fill_between(x=red_x_cen, y1=red_y_max, 
        y2=red_y_min, color='lightcoral',alpha=0.4,label='Models')
    mb = plt.fill_between(x=blue_x_cen, y1=blue_y_max, 
        y2=blue_y_min, color='cornflowerblue',alpha=0.4,label='Models')

    red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
    blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

    # REMOVED ERROR BAR ON BEST FIT
    bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=4,label='Best-fit',zorder=10)
    bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=4,
        label='Best-fit',zorder=10)

    plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
        [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
        plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
        hatch='\\')

    if survey == 'resolvea' and mf_type == 'smf':
        plt.xlim(10,14)
    else:
        plt.xlim(10,14.5)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    if mf_type == 'smf':
        if survey == 'eco' and quenching == 'hybrid':
            plt.ylim(np.log10((10**8.9)/2.041),11.9)
            # plt.title('ECO')
        elif survey == 'eco' and quenching == 'halo':
            plt.ylim(np.log10((10**8.9)/2.041),11.56)
        elif survey == 'resolvea':
            plt.ylim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**8.7)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.ylim(np.log10((10**9.4)/2.041),)
            plt.title('ECO')
        elif survey == 'resolveb':
            plt.ylim(np.log10((10**9.1)/2.041),)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
    # plt.legend([("darkred", "darkblue", "-"), \
    #     ("indianred","cornflowerblue", "-")],\
    #     ["Best-fit", "Models"], handler_map={tuple: AnyObjectHandler()},\
    #     loc='best', prop={'size': 30})
    plt.legend([(mr, mb), (bfr, bfb)], ['Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, 
        loc='best',prop={'size': 30})

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_colour_hmxm(result, gals_bf_red, halos_bf_red, gals_bf_blue, halos_bf_blue,
    bf_chi2):
    """
    Plot SMHM from data, best fit param values, param values corresponding to 
    68th percentile 1000 lowest chi^2 values and behroozi 2010 param values

    Parameters
    ----------
    result: multidimensional array
        Array of central galaxy and halo masses
    
    gals_bf: array
        Array of y-axis stellar mass values for best fit SMHM

    halos_bf: array
        Array of x-axis halo mass values for best fit SMHM
    
    gals_data: array
        Array of y-axis stellar mass values for data SMF

    halos_data: array
        Array of x-axis halo mass values for data SMF

    gals_b10: array
        Array of y-axis stellar mass values for behroozi 2010 SMHM

    halos_b10: array
        Array of x-axis halo mass values for behroozi 2010 SMHM

    Returns
    ---------
    Nothing; SMHM plot is saved in figures repository
    """   
    class AnyObjectHandler(HandlerBase): 
        def create_artists(self, legend, orig_handle, x0, y0, width, height, 
            fontsize, trans):
            l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height], 
                linestyle=orig_handle[2], color=orig_handle[0]) 
            l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],  
                linestyle=orig_handle[2],
                color=orig_handle[1])
            return [l1, l2]  

    # x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(gals_bf_red,\
    # halos_bf_red,base=0.2,bin_statval='center')
    # x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(gals_bf_blue,\
    # halos_bf_blue,base=0.2,bin_statval='center')

    y_bf_red,x_bf_red,binnum_red = bs(gals_bf_red,\
    halos_bf_red,'mean',bins=np.linspace(8.6, 11.5, 7))
    y_bf_blue,x_bf_blue,binnum_blue = bs(gals_bf_blue,\
    halos_bf_blue,'mean',bins=np.linspace(8.6, 11.5, 7))

    for idx in range(5,20,1):
        fig1 = plt.figure(figsize=(16,9))

        y_bf_red,x_bf_red,binnum_red = bs(gals_bf_red,\
        halos_bf_red,'mean',bins=np.linspace(8.6, 11.5, idx))
        y_bf_blue,x_bf_blue,binnum_blue = bs(gals_bf_blue,\
        halos_bf_blue,'mean',bins=np.linspace(8.6, 11.5, idx))

        red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
        blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

        # REMOVED ERROR BAR ON BEST FIT
        bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=4,label='Best-fit',zorder=10)
        bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=4,
            label='Best-fit',zorder=10)

        plt.annotate(r'$Number of bins: ${0}'.
            format(int(idx)), 
            xy=(0.02, 0.9), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        plt.xlim(np.log10((10**8.9)/2.041),12)
        plt.ylim(10,13.5)
        if quenching == 'halo':
            plt.title('Halo quenching model')
        elif quenching == 'hybrid':
            plt.title('Hybrid quenching model')
        plt.xlabel('Stellar Mass')
        plt.ylabel('Halo Mass')
        plt.savefig('/Users/asadm2/Desktop/hsmr_binning/{0}.png'.format(idx))

    i_outer = 0
    red_mod_x_arr = []
    red_mod_y_arr = []
    blue_mod_x_arr = []
    blue_mod_y_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_y_ii = result[i_outer][7][idx] #halos
            red_mod_x_ii = result[i_outer][6][idx] #galaxies
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_y_ii = result[i_outer][9][idx] #halos
            blue_mod_x_ii = result[i_outer][8][idx] #galaxies
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_y_ii = result[i_outer][7][idx]
            red_mod_x_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_y_ii = result[i_outer][9][idx]
            blue_mod_x_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_y_ii = result[i_outer][7][idx]
            red_mod_x_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_y_ii = result[i_outer][9][idx]
            blue_mod_x_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_y_ii = result[i_outer][7][idx]
            red_mod_x_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_y_ii = result[i_outer][9][idx]
            blue_mod_x_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_y_ii = result[i_outer][7][idx]
            red_mod_x_ii = result[i_outer][6][idx]
            red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            red_mod_x_arr.append(red_x)
            red_mod_y_arr.append(red_y)
            blue_mod_y_ii = result[i_outer][9][idx]
            blue_mod_x_ii = result[i_outer][8][idx]
            blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                bins=np.linspace(8.6, 11.5, 7))
            blue_mod_x_arr.append(blue_x)
            blue_mod_y_arr.append(blue_y)
        i_outer += 1

    red_y_max = np.nanmax(red_mod_y_arr, axis=0)
    red_y_min = np.nanmin(red_mod_y_arr, axis=0)
    blue_y_max = np.nanmax(blue_mod_y_arr, axis=0)
    blue_y_min = np.nanmin(blue_mod_y_arr, axis=0)

    # for idx in range(len(result[0][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[0][4][idx],result[0][5][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-', \
    #         alpha=0.5, zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[0][6][idx],result[0][7][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[1][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[1][4][idx],result[1][5][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[1][6][idx],result[1][7][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[2][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[2][4][idx],result[2][5][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[2][6][idx],result[2][7][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='center')
    # for idx in range(len(result[3][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[3][4][idx],result[3][5][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[3][6][idx],result[3][7][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    # for idx in range(len(result[4][0])):
    #     x_model_red,y_model_red,y_std_model_red,y_std_err_model_red = \
    #         Stats_one_arr(result[4][4][idx],result[4][5][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_red,y_model_red,color='indianred',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')
    #     x_model_blue,y_model_blue,y_std_model_blue,y_std_err_model_blue = \
    #         Stats_one_arr(result[4][6][idx],result[4][7][idx],base=0.4,\
    #             bin_statval='center')
    #     plt.plot(x_model_blue,y_model_blue,color='cornflowerblue',linestyle='-',alpha=0.5,\
    #         zorder=0,label='model')

    fig1 = plt.figure(figsize=(10,10))

    red_x_cen =  0.5 * (red_mod_x_arr[0][1:] + red_mod_x_arr[0][:-1])
    blue_x_cen = 0.5 * (blue_mod_x_arr[0][1:] + blue_mod_x_arr[0][:-1])

    mr = plt.fill_between(x=red_x_cen, y1=red_y_max, 
        y2=red_y_min, color='lightcoral',alpha=0.4,label='Models')
    mb = plt.fill_between(x=blue_x_cen, y1=blue_y_max, 
        y2=blue_y_min, color='cornflowerblue',alpha=0.4,label='Models')

    red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
    blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

    # REMOVED ERROR BAR ON BEST FIT
    bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=3,label='Best-fit',
        zorder=10)
    bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=3,
        label='Best-fit',zorder=10)

    # plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
    #     [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
    #     plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
    #     hatch='\\')


    if survey == 'resolvea' and mf_type == 'smf':
        plt.ylim(10,14)
    else:
        plt.ylim(10,)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    if mf_type == 'smf':
        if survey == 'eco':
            plt.xlim(np.log10((10**8.9)/2.041),)
            plt.title('ECO')
        elif survey == 'resolvea':
            plt.xlim(np.log10((10**8.9)/2.041),13)
        elif survey == 'resolveb':
            plt.xlim(np.log10((10**8.7)/2.041),)
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        if survey == 'eco' or survey == 'resolvea':
            plt.xlim(np.log10((10**9.4)/2.041),)
            plt.title('ECO')
        elif survey == 'resolveb':
            plt.xlim(np.log10((10**9.1)/2.041),)
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

    plt.legend([(mr, mb), (bfr, bfb)], ['Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, 
        loc='upper left',prop={'size': 30})

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')
    
    plt.show()

def plot_red_fraction_cen(result, cen_gals_red, \
    cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
    f_red_cen_blue):
    """
    Plot red fraction of centrals from best fit param values and param values 
    corresponding to 68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    cen_gals_red: array
        Array of red central stellar mass values for best-fit model

    cen_halos_red: array
        Array of red central halo mass values for best-fit model

    cen_gals_blue: array
        Array of blue central stellar mass values for best-fit model

    cen_halos_blue: array
        Array of blue central halo mass values for best-fit model

    f_red_cen_red: array
        Array of red fractions for red centrals for best-fit model

    f_red_cen_blue: array
        Array of red fractions for blue centrals for best-fit model
        
    Returns
    ---------
    Plot displayed on screen.
    """   
    cen_gals_arr = []
    cen_halos_arr = []
    fred_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        cen_gals_idx_arr = []
        cen_halos_idx_arr = []
        fred_idx_arr = []
        for idx in range(len(result[chunk_counter][0])):
            red_cen_gals_idx = result[chunk_counter][6][idx]
            red_cen_halos_idx = result[chunk_counter][7][idx]
            blue_cen_gals_idx = result[chunk_counter][8][idx]
            blue_cen_halos_idx = result[chunk_counter][9][idx]
            fred_red_cen_idx = result[chunk_counter][10][idx]
            fred_blue_cen_idx = result[chunk_counter][11][idx]  

            cen_gals_idx_arr = list(red_cen_gals_idx) + list(blue_cen_gals_idx)
            cen_gals_arr.append(cen_gals_idx_arr)

            cen_halos_idx_arr = list(red_cen_halos_idx) + list(blue_cen_halos_idx)
            cen_halos_arr.append(cen_halos_idx_arr)

            fred_idx_arr = list(fred_red_cen_idx) + list(fred_blue_cen_idx)
            fred_arr.append(fred_idx_arr)
            
            cen_gals_idx_arr = []
            cen_halos_idx_arr = []
            fred_idx_arr = []

        chunk_counter+=1
    
    cen_gals_bf = []
    cen_halos_bf = []
    fred_bf = []

    cen_gals_bf = list(cen_gals_red) + list(cen_gals_blue)
    cen_halos_bf = list(cen_halos_red) + list(cen_halos_blue)
    fred_bf = list(f_red_cen_red) + list(f_red_cen_blue)


    fig1 = plt.figure(figsize=(10,8))
    if quenching == 'hybrid':
        for idx in range(len(cen_gals_arr)):
            x, y = zip(*sorted(zip(cen_gals_arr[idx],fred_arr[idx])))
            plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, solid_capstyle='round')
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

        x, y = zip(*sorted(zip(cen_gals_arr[0],fred_arr[0])))
        x_bf, y_bf = zip(*sorted(zip(cen_gals_bf,fred_bf)))
        # Plotting again just so that adding label to legend is easier
        plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, solid_capstyle='round')
        plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, solid_capstyle='round')

    elif quenching == 'halo':
        for idx in range(len(cen_halos_arr)):
            x, y = zip(*sorted(zip(cen_halos_arr[idx],fred_arr[idx])))
            plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, solid_capstyle='round')
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

        x, y = zip(*sorted(zip(cen_halos_arr[0],fred_arr[0])))
        x_bf, y_bf = zip(*sorted(zip(cen_halos_bf, fred_bf)))
        # Plotting again just so that adding label to legend is easier
        plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, solid_capstyle='round')
        plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, solid_capstyle='round')

    plt.ylabel(r'\boldmath$f_{red, cen}$', fontsize=30)
    plt.legend(loc='best', prop={'size':30})

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_red_fraction_sat(result, sat_gals_red, sat_halos_red, \
    sat_gals_blue, sat_halos_blue, f_red_sat_red, f_red_sat_blue):
    """
    Plot red fraction of satellites from best fit param values and param values 
    corresponding to 68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    sat_gals_red: array
        Array of red satellite stellar mass values for best-fit model

    sat_halos_red: array
        Array of red satellite host halo mass values for best-fit model

    sat_gals_blue: array
        Array of blue satellite stellar mass values for best-fit model

    sat_halos_blue: array
        Array of blue satellite host halo mass values for best-fit model

    f_red_sat_red: array
        Array of red fractions for red satellites for best-fit model

    f_red_sat_blue: array
        Array of red fractions for blue satellites for best-fit model
        
    Returns
    ---------
    Plot displayed on screen.
    """
    sat_gals_arr = []
    sat_halos_arr = []
    fred_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        sat_gals_idx_arr = []
        sat_halos_idx_arr = []
        fred_idx_arr = []
        for idx in range(len(result[chunk_counter][0])):
            red_sat_gals_idx = result[chunk_counter][12][idx]
            red_sat_halos_idx = result[chunk_counter][13][idx]
            blue_sat_gals_idx = result[chunk_counter][14][idx]
            blue_sat_halos_idx = result[chunk_counter][15][idx]
            fred_red_sat_idx = result[chunk_counter][16][idx]
            fred_blue_sat_idx = result[chunk_counter][17][idx]  

            sat_gals_idx_arr = list(red_sat_gals_idx) + list(blue_sat_gals_idx)
            sat_gals_arr.append(sat_gals_idx_arr)

            sat_halos_idx_arr = list(red_sat_halos_idx) + list(blue_sat_halos_idx)
            sat_halos_arr.append(sat_halos_idx_arr)

            fred_idx_arr = list(fred_red_sat_idx) + list(fred_blue_sat_idx)
            fred_arr.append(fred_idx_arr)
            
            sat_gals_idx_arr = []
            sat_halos_idx_arr = []
            fred_idx_arr = []

        chunk_counter+=1
    
    sat_gals_arr = np.array(sat_gals_arr)
    sat_halos_arr = np.array(sat_halos_arr)
    fred_arr = np.array(fred_arr)

    if quenching == 'hybrid':
        sat_mean_stats = bs(np.hstack(sat_gals_arr), np.hstack(fred_arr), bins=10)
        sat_std_stats = bs(np.hstack(sat_gals_arr), np.hstack(fred_arr), 
            statistic='std', bins=10)
        sat_stats_bincens = 0.5 * (sat_mean_stats[1][1:] + sat_mean_stats[1][:-1])

    elif quenching == 'halo':
        sat_mean_stats = bs(np.hstack(sat_halos_arr), np.hstack(fred_arr), bins=10)
        sat_std_stats = bs(np.hstack(sat_halos_arr), np.hstack(fred_arr), 
            statistic='std', bins=10)
        sat_stats_bincens = 0.5 * (sat_mean_stats[1][1:] + sat_mean_stats[1][:-1])

    sat_gals_bf = []
    sat_halos_bf = []
    fred_bf = []
    sat_gals_bf = list(sat_gals_red) + list(sat_gals_blue)
    sat_halos_bf = list(sat_halos_red) + list(sat_halos_blue)
    fred_bf = list(f_red_sat_red) + list(f_red_sat_blue)


    fig1 = plt.figure(figsize=(10,8))
    if quenching == 'hybrid':
        # plt.plot(sat_stats_bincens, sat_mean_stats[0], lw=3, ls='--', 
        #     c='cornflowerblue', marker='p', ms=20)
        # plt.fill_between(sat_stats_bincens, sat_mean_stats[0]+sat_std_stats[0], 
        #     sat_mean_stats[0]-sat_std_stats[0], color='cornflowerblue', 
        #     alpha=0.5)
        plt.errorbar(sat_stats_bincens, sat_mean_stats[0], 
            yerr=sat_std_stats[0], color='navy', fmt='s', 
            ecolor='navy',markersize=12, capsize=7, capthick=1.5, 
            zorder=10, marker='p', label='Model average')
        # for idx in range(len(sat_halos_arr)):
        #     plt.scatter(sat_halos_arr[idx], fred_arr[idx], alpha=0.4, s=150, c='cornflowerblue')
            # x, y = zip(*sorted(zip(sat_halos_arr[idx],fred_arr[idx])))
            # plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, 
            #     solid_capstyle='round')

        # plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\,'\
        #     r' \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, sat} \left[\mathrm{M_\odot}\,'\
                    r' \mathrm{h}^{-1} \right]$',fontsize=30)

        # plotting again just for label
        # plt.scatter(sat_halos_arr[0], fred_arr[0], alpha=0.4, s=150, c='cornflowerblue', label='Models')
        # plt.scatter(sat_halos_bf, fred_bf, alpha=0.4, s=150, c='mediumorchid', 
        #     label='Best-fit')
        plt.scatter(sat_gals_bf, fred_bf, alpha=0.4, s=150, c=sat_halos_bf, 
            cmap='viridis' ,label='Best-fit')
        plt.colorbar(label=r'\boldmath$\log_{10}\ M_{h, host}$')

        # x, y = zip(*sorted(zip(sat_halos_arr[0],fred_arr[0])))
        # x_bf, y_bf = zip(*sorted(zip(sat_halos_bf,fred_bf)))
        # # Plotting again just so that adding label to legend is easier
        # plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, 
        #     solid_capstyle='round')
        # plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, 
        #     solid_capstyle='round')

    elif quenching == 'halo':
        plt.errorbar(sat_stats_bincens, sat_mean_stats[0], 
            yerr=sat_std_stats[0], color='navy', fmt='s', 
            ecolor='navy',markersize=12, capsize=7, capthick=1.5, 
            zorder=10, marker='p', label='Model average')
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\,'\
                    r' \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.scatter(sat_halos_bf, fred_bf, alpha=0.4, s=150, c='mediumorchid',\
            label='Best-fit', zorder=10)
        # for idx in range(len(sat_halos_arr)):
        #     # plt.scatter(sat_halos_arr[idx], fred_arr[idx], alpha=0.4, s=150, 
        #     #     c='cornflowerblue')
        #     x, y = zip(*sorted(zip(sat_halos_arr[idx],fred_arr[idx])))
        #     plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, 
        #         solid_capstyle='round')
        # ## Plotting again just for legend label
        # plt.scatter(sat_halos_arr[0], fred_arr[0], alpha=0.4, s=150, 
        #     c='cornflowerblue', label='Models')

    plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
    plt.legend(loc='best', prop={'size':30})

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_zumand_fig4(result, gals_bf_red, halos_bf_red, gals_bf_blue, 
    halos_bf_blue, bf_chi2):
    """
    Plot red and blue SMHM from best fit param values and param values 
    corresponding to 68th percentile 100 lowest chi^2 values like Fig 4 from 
    Zu and Mandelbaum paper

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information
    
    gals_bf_red: array
        Array of y-axis stellar mass values for red SMHM for best-fit model

    halos_bf_red: array
        Array of x-axis halo mass values for red SMHM for best-fit model
    
    gals_bf_blue: array
        Array of y-axis stellar mass values for blue SMHM for best-fit model

    halos_bf_blue: array
        Array of x-axis halo mass values for blue SMHM for best-fit model

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """   

    # if model == 'halo':
    #     sat_halomod_df = gals_df.loc[gals_df.C_S.values == 0] 
    #     cen_halomod_df = gals_df.loc[gals_df.C_S.values == 1]
    # elif model == 'hybrid':
    #     sat_hybmod_df = gals_df.loc[gals_df.C_S.values == 0]
    #     cen_hybmod_df = gals_df.loc[gals_df.C_S.values == 1]

    logmhalo_mod_arr = result[0][7]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][7])
    for idx in range(5):
        logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, result[idx][9])
    logmhalo_mod_arr_flat = np.hstack(logmhalo_mod_arr)

    logmstar_mod_arr = result[0][6]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][6])
    for idx in range(5):
        logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, result[idx][8])
    logmstar_mod_arr_flat = np.hstack(logmstar_mod_arr)

    fred_mod_arr = result[0][10]
    for idx in range(5):
        idx+=1
        if idx == 5:
            break
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][10])
    for idx in range(5):
        fred_mod_arr = np.insert(fred_mod_arr, -1, result[idx][11])
    fred_mod_arr_flat = np.hstack(fred_mod_arr)

    fig1 = plt.figure()
    plt.hexbin(logmhalo_mod_arr_flat, logmstar_mod_arr_flat, 
        C=fred_mod_arr_flat, cmap='rainbow')
    cb = plt.colorbar()
    cb.set_label(r'\boldmath\ $f_{red}$')


    x_bf_red,y_bf_red,y_std_bf_red,y_std_err_bf_red = Stats_one_arr(halos_bf_red,\
    gals_bf_red,base=0.4,bin_statval='center')
    x_bf_blue,y_bf_blue,y_std_bf_blue,y_std_err_bf_blue = Stats_one_arr(halos_bf_blue,\
    gals_bf_blue,base=0.4,bin_statval='center')

    bfr, = plt.plot(x_bf_red,y_bf_red,color='darkred',lw=5,zorder=10)
    bfb, = plt.plot(x_bf_blue,y_bf_blue,color='darkblue',lw=5,zorder=10)

    plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
        format(np.round(bf_chi2/dof,2)), 
        xy=(0.02, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
        [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
        plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
        hatch='\\')

    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')

    plt.ylim(8.45, 12.3)
    plt.xlim(10, 14.5)

    plt.legend([(bfr, bfb)], ['Best-fit'], 
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='best', 
        prop={'size': 30})

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    # if survey == 'eco':
    #     plt.title('ECO')
    
    plt.show()

def plot_mean_grpcen_vs_sigma(result, red_sigma_bf, \
    grp_red_cen_stellar_mass_bf, blue_sigma_bf, grp_blue_cen_stellar_mass_bf, \
    red_sigma_data, grp_red_cen_stellar_mass_data, blue_sigma_data, \
    grp_blue_cen_stellar_mass_data, err_red, err_blue, bf_chi2):
    """
    Plot average group central stellar mass vs. velocity dispersion from data, 
    best fit param values and param values corresponding to 68th percentile 100 
    lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information

    red_sigma_bf: array
        Array of velocity dispersion around red group centrals for best-fit 
        model

    grp_red_cen_stellar_mass_bf: array
        Array of red group central stellar masses for best-fit model

    blue_sigma_bf: array
        Array of velocity dispersion around blue group centrals for best-fit 
        model

    grp_blue_cen_stellar_mass_bf: array
        Array of blue group central stellar masses for best-fit model

    red_sigma_data: array
        Array of velocity dispersion around red group centrals for data

    grp_red_cen_stellar_mass_data: array
        Array of red group central stellar masses for data

    blue_sigma_data: array
        Array of velocity dispersion around blue group centrals for data

    grp_blue_cen_stellar_mass_data: array
        Array of blue group central stellar masses for data

    err_red: array
        Array of std values per bin of red group central stellar mass vs. 
        velocity dispersion from mocks

    err_blue: array
        Array of std values per bin of blue group central stellar mass vs. 
        velocity dispersion from mocks

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """   
    
    # grp_red_cen_gals_arr = []
    # grp_blue_cen_gals_arr = []
    # red_sigma_arr = []
    # blue_sigma_arr = []
    # chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    # while chunk_counter < 5:
    #     for idx in range(len(result[chunk_counter][0])):
    #         grp_red_cen_gals_idx = result[chunk_counter][16][idx]
    #         grp_blue_cen_gals_idx = result[chunk_counter][17][idx]
    #         red_sigma_idx = result[chunk_counter][18][idx]
    #         blue_sigma_idx = result[chunk_counter][19][idx]

    #         for idx,val in enumerate(grp_red_cen_gals_idx):
    #             grp_red_cen_gals_arr.append(val)
    #             red_sigma_arr.append(red_sigma_idx[idx])
            
    #         for idx,val in enumerate(grp_blue_cen_gals_idx):
    #             grp_blue_cen_gals_arr.append(val)
    #             blue_sigma_arr.append(blue_sigma_idx[idx])

    #         # grp_red_cen_gals_arr.append(grp_red_cen_gals_idx)
    #         # grp_blue_cen_gals_arr.append(grp_blue_cen_gals_idx)
    #         # red_sigma_arr.append(red_sigma_idx)
    #         # blue_sigma_arr.append(blue_sigma_idx)

    #     chunk_counter+=1
    
    # mean_stats_red = bs(red_sigma_arr, grp_red_cen_gals_arr, statistic='mean', 
    #     bins=np.linspace(0,250,6))
    # mean_stats_blue = bs(blue_sigma_arr, grp_blue_cen_gals_arr, statistic='mean', 
    #     bins=np.linspace(0,250,6))

    # std_stats_red = bs(red_sigma_arr, grp_red_cen_gals_arr, statistic='std', 
    #     bins=np.linspace(0,250,6))
    # std_stats_blue = bs(blue_sigma_arr, grp_blue_cen_gals_arr, statistic='std', 
    #     bins=np.linspace(0,250,6))
    global mean_centers_red
    global mean_stats_red_data
    global mean_centers_blue
    global mean_stats_blue_data

    mean_grp_red_cen_gals_arr = []
    mean_grp_blue_cen_gals_arr = []
    red_sigma_arr = []
    blue_sigma_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        for idx in range(len(result[chunk_counter][0])):
            grp_red_cen_gals_idx = result[chunk_counter][20][idx]
            grp_blue_cen_gals_idx = result[chunk_counter][21][idx]
            red_sigma_idx = result[chunk_counter][22][idx]
            blue_sigma_idx = result[chunk_counter][23][idx]

            mean_stats_red = bs(red_sigma_idx, grp_red_cen_gals_idx, 
                statistic='mean', bins=np.linspace(0,250,6))
            mean_stats_blue = bs(blue_sigma_idx, grp_blue_cen_gals_idx, 
                statistic='mean', bins=np.linspace(0,250,6))
            red_sigma_arr.append(mean_stats_red[1])
            blue_sigma_arr.append(mean_stats_blue[1])
            mean_grp_red_cen_gals_arr.append(mean_stats_red[0])
            mean_grp_blue_cen_gals_arr.append(mean_stats_blue[0])

        chunk_counter+=1

    red_models_max = np.nanmax(mean_grp_red_cen_gals_arr, axis=0)
    red_models_min = np.nanmin(mean_grp_red_cen_gals_arr, axis=0)
    blue_models_max = np.nanmax(mean_grp_blue_cen_gals_arr, axis=0)
    blue_models_min = np.nanmin(mean_grp_blue_cen_gals_arr, axis=0)

    ## Same centers used for all sets of lines since binning is the same for 
    ## models, bf and data
    mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    mean_stats_red_bf = bs(red_sigma_bf, grp_red_cen_stellar_mass_bf, 
        statistic='mean', bins=np.linspace(0,250,6))
    mean_stats_blue_bf = bs(blue_sigma_bf, grp_blue_cen_stellar_mass_bf, 
        statistic='mean', bins=np.linspace(0,250,6))

    mean_stats_red_data = bs(red_sigma_data, grp_red_cen_stellar_mass_data, 
        statistic='mean', bins=np.linspace(0,250,6))
    mean_stats_blue_data = bs(blue_sigma_data, grp_blue_cen_stellar_mass_data, 
        statistic='mean', bins=np.linspace(0,250,6))

    # ## Seeing the effects of binning
    # for idx in range(3,10,1):
    #     fig1 = plt.figure(figsize=(16,9))

    #     mean_stats_red_bf = bs(red_sigma_bf, grp_red_cen_stellar_mass_bf, 
    #         statistic='mean', bins=np.linspace(0,250,idx))
    #     mean_stats_blue_bf = bs(blue_sigma_bf, grp_blue_cen_stellar_mass_bf, 
    #         statistic='mean', bins=np.linspace(0,250,idx))

    #     mean_centers_red = 0.5 * (mean_stats_red_bf[1][1:] + \
    #         mean_stats_red_bf[1][:-1])
    #     mean_centers_blue = 0.5 * (mean_stats_blue_bf[1][1:] + \
    #         mean_stats_blue_bf[1][:-1])

    #     bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
    #     bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

    #     plt.scatter(red_sigma_bf, grp_red_cen_stellar_mass_bf, c='r', alpha=0.4)
    #     plt.scatter(blue_sigma_bf, grp_blue_cen_stellar_mass_bf, c='b', alpha=0.4)

    #     plt.annotate(r'$Number of bins: ${0}'.
    #         format(int(idx)-1), 
    #         xy=(0.02, 0.9), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    #         ec='k', fc='lightgray', alpha=0.5), size=25)

    #     # plt.xlim(10,14.5)
    #     plt.ylim(np.log10((10**8.9)/2.041),11.56)
    #     if quenching == 'halo':
    #         plt.title('Halo quenching model')
    #     elif quenching == 'hybrid':
    #         plt.title('Hybrid quenching model')
    #     plt.xlabel('Sigma')
    #     plt.ylabel('Stellar Mass')
    #     plt.show()

    fig1,ax1 = plt.subplots(figsize=(10,8))

    dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=err_red,
            color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
    db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=err_blue,
            color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
            capthick=1.0,zorder=10)

    # dr = plt.scatter(mean_centers_red, mean_stats_red_data[0], marker='o', \
    #     c='darkred', s=200, zorder=10)
    # db = plt.scatter(mean_centers_blue, mean_stats_blue_data[0], marker='o', \
    #     c='darkblue', s=200, zorder=10)

    
    mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
        y2=red_models_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
        y2=blue_models_min, color='cornflowerblue',alpha=0.4)

    bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
    bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

    l = plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
        ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

    chi_squared_red = np.nansum((mean_stats_red_data[0] - 
        mean_stats_red_bf[0])**2 / (err_red**2))
    chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
        mean_stats_blue_bf[0])**2 / (err_blue**2))

    plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
        r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
        chi_squared_red,2),np.round(chi_squared_blue,2)), 
        xy=(0.015, 0.73), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    plt.ylim(8.9, 11.1)
    # plt.annotate(r'$\boldsymbol\chi ^2 \approx$ {0}'.format(np.round(chi_squared_blue/dof,2)), 
    # xy=(0.02, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    # ec='k', fc='lightgray', alpha=0.5), size=25)

    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
    from matplotlib.widgets import Button


    def errors(event):
        global l1
        global l2
        l1 = ax1.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=err_red,
            color='darkred',fmt='o',ecolor='darkred',markersize=13,capsize=10,
            capthick=1.0,zorder=10)
        l2 = ax1.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=err_blue,
            color='darkblue',fmt='o',ecolor='darkblue',markersize=13,capsize=10,
            capthick=1.0,zorder=10)
        ax1.draw()

    def remove_errors(event):
        l1.remove()
        l2.remove()


    # if survey == 'eco':
    #     plt.title('ECO')
   
    plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    
    # axerrors = plt.axes([0, 0, 1, 1])
    # axnoerrors = plt.axes([0.2, 0.05, 1, 1])
    # iperrors = InsetPosition(ax1, [0.05, 0.05, 0.1, 0.08]) #posx, posy, width, height
    # ipnoerrors = InsetPosition(ax1, [0.2, 0.05, 0.12, 0.08]) #posx, posy, width, height
    # axerrors.set_axes_locator(iperrors)
    # axnoerrors.set_axes_locator(ipnoerrors)
    # berrors = Button(axerrors, 'Add errors', color='pink', hovercolor='tomato')
    # berrors.on_clicked(errors)
    # bnoerrors = Button(axnoerrors, 'Remove errors', color='pink', hovercolor='tomato')
    # bnoerrors.on_clicked(remove_errors)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_sigma_vdiff_mod(result, std_red_data, cen_red_data, std_blue_data, \
    cen_blue_data, std_bf_red, std_bf_blue, std_cen_bf_red, std_cen_bf_blue, \
    bf_chi2, err_red, err_blue):
    """[summary]

    Args:
        result ([type]): [description]
        std_red_data ([type]): [description]
        cen_red_data ([type]): [description]
        std_blue_data ([type]): [description]
        cen_blue_data ([type]): [description]
        std_bf_red ([type]): [description]
        std_bf_blue ([type]): [description]
        std_cen_bf_red ([type]): [description]
        std_cen_bf_blue ([type]): [description]
        bf_chi2 ([type]): [description]
    """
    i_outer = 0
    red_mod_arr = []
    blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][24][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][25][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][24][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][25][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][24][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][25][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][24][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][25][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][24][idx]
            red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][25][idx]
            blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    red_std_max = np.nanmax(red_mod_arr, axis=0)
    red_std_min = np.nanmin(red_mod_arr, axis=0)
    blue_std_max = np.nanmax(blue_mod_arr, axis=0)
    blue_std_min = np.nanmin(blue_mod_arr, axis=0)

    fig1= plt.figure(figsize=(10,10))

    mr = plt.fill_between(x=cen_red_data, y1=red_std_min, 
        y2=red_std_max, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=cen_blue_data, y1=blue_std_min, 
        y2=blue_std_max, color='cornflowerblue',alpha=0.4)

    dr = plt.errorbar(cen_red_data,std_red_data,yerr=err_red,
        color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
        capthick=1.0,zorder=10)
    db = plt.errorbar(cen_blue_data,std_blue_data,yerr=err_blue,
        color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
        capthick=1.0,zorder=10)

    bfr, = plt.plot(std_cen_bf_red,std_bf_red,
        color='maroon',ls='-',lw=3,zorder=10)
    bfb, = plt.plot(std_cen_bf_blue,std_bf_blue,
        color='mediumblue',ls='-',lw=3,zorder=10)

    plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)

    plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
        ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
        loc='upper left')

    chi_squared_red = np.nansum((std_red_data - 
        std_bf_red)**2 / (err_red**2))
    chi_squared_blue = np.nansum((std_blue_data - 
        std_bf_blue)**2 / (err_blue**2))

    plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
        r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
        chi_squared_red,2),np.round(chi_squared_blue,2)), 
        xy=(0.015, 0.73), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    # if survey == 'eco':
    #     plt.title('ECO')
    plt.show()

def plot_sigma_host_halo_mass_vishnu(result, vdisp_red_bf, vdisp_blue_bf, \
    red_host_halo_bf, blue_host_halo_bf):

    i_outer = 0
    vdisp_red_mod_arr = []
    vdisp_blue_mod_arr = []
    hosthalo_red_mod_arr = []
    hosthalo_blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][28][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][29][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][30][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][31][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][28][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][29][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][30][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][31][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][28][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][29][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][30][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][31][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][28][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][29][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][30][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][31][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][28][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][29][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][30][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][31][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    flat_list_red_vdisp = [item for sublist in vdisp_red_mod_arr for item in sublist]
    flat_list_red_host = [item for sublist in hosthalo_red_mod_arr for item in sublist]

    flat_list_blue_vdisp = [item for sublist in vdisp_blue_mod_arr for item in sublist]
    flat_list_blue_host = [item for sublist in hosthalo_blue_mod_arr for item in sublist]

    import seaborn as sns
    fig1 = plt.figure(figsize=(11, 9))
    bfr = plt.scatter(vdisp_red_bf, np.log10(red_host_halo_bf), c='maroon', s=120, zorder = 10)
    bfb = plt.scatter(vdisp_blue_bf, np.log10(blue_host_halo_bf), c='darkblue', s=120, zorder=10)

    sns.kdeplot(x=flat_list_red_vdisp, y=np.log10(flat_list_red_host), color='indianred', shade=True)
    sns.kdeplot(x=flat_list_blue_vdisp, y=np.log10(flat_list_blue_host), color='cornflowerblue', shade=True)
    # for idx in range(len(vdisp_red_mod_arr)):
    #     mr = plt.scatter(vdisp_red_mod_arr[idx], np.log10(hosthalo_red_mod_arr[idx]), 
    #         c='indianred', s=120, alpha=0.8, marker='*')
    # for idx in range(len(vdisp_blue_mod_arr)):
    #     mb = plt.scatter(vdisp_blue_mod_arr[idx], np.log10(hosthalo_blue_mod_arr[idx]), 
    #         c='cornflowerblue', s=120, alpha=0.8, marker='*')
    
    plt.legend([(bfr, bfb)], 
        ['Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
        loc='best')

    plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    plt.title('Host halo mass - velocity dispersion in best-fit model (excluding singletons)')
    plt.show()

def plot_N_host_halo_mass_vishnu(result, N_red_bf, N_blue_bf, \
    red_host_halo_bf, blue_host_halo_bf):

    i_outer = 0
    N_red_mod_arr = []
    N_blue_mod_arr = []
    hosthalo_red_mod_arr = []
    hosthalo_blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][33][idx]
            N_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][35][idx]
            N_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][36][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][37][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][33][idx]
            N_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][35][idx]
            N_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][36][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][37][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][33][idx]
            N_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][35][idx]
            N_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][36][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][37][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][33][idx]
            N_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][35][idx]
            N_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][36][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][37][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][33][idx]
            N_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][35][idx]
            N_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][36][idx]
            hosthalo_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][37][idx]
            hosthalo_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    flat_list_red_N = [item for sublist in N_red_mod_arr for item in sublist]
    flat_list_red_host = [item for sublist in hosthalo_red_mod_arr for item in sublist]

    flat_list_blue_N = [item for sublist in N_blue_mod_arr for item in sublist]
    flat_list_blue_host = [item for sublist in hosthalo_blue_mod_arr for item in sublist]

    import seaborn as sns
    fig1 = plt.figure(figsize=(11, 9))
    bfr = plt.scatter(N_red_bf, np.log10(red_host_halo_bf), c='maroon', s=120, zorder = 10)
    N_blue_bf_offset = [x+0.05 for x in N_blue_bf] #For plotting purposes
    bfb = plt.scatter(N_blue_bf_offset, np.log10(blue_host_halo_bf), c='darkblue', s=120, zorder=11, alpha=0.3)

    # sns.kdeplot(x=flat_list_red_N, y=np.log10(flat_list_red_host), color='indianred', shade=True)
    # sns.kdeplot(x=flat_list_blue_N, y=np.log10(flat_list_blue_host), color='cornflowerblue', shade=True)
    # for idx in range(len(vdisp_red_mod_arr)):
    #     mr = plt.scatter(vdisp_red_mod_arr[idx], np.log10(hosthalo_red_mod_arr[idx]), 
    #         c='indianred', s=120, alpha=0.8, marker='*')
    # for idx in range(len(vdisp_blue_mod_arr)):
    #     mb = plt.scatter(vdisp_blue_mod_arr[idx], np.log10(hosthalo_blue_mod_arr[idx]), 
    #         c='cornflowerblue', s=120, alpha=0.8, marker='*')
    
    plt.legend([(bfr, bfb)], 
        ['Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
        loc='best')

    plt.xlabel(r'\boldmath$ {N} $', fontsize=30)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    plt.title('Host halo mass - Number of galaxies in halo in best-fit model (excluding singletons)')
    plt.show()

def plot_mean_grpcen_vs_N(result, red_num_bf, \
    red_cen_stellar_mass_bf, blue_num_bf, blue_cen_stellar_mass_bf, \
    red_num_data, red_cen_stellar_mass_data_N, blue_num_data, \
    blue_cen_stellar_mass_data_N, red_cen_stellar_mass_mocks, red_num_mocks, \
    blue_cen_stellar_mass_mocks, blue_num_mocks):
    """
    Plot average halo/group central stellar mass vs. number of galaxies in 
    halos/groups from data, best fit param values and param values corresponding 
    to 68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information

    red_sigma_bf: array
        Array of velocity dispersion around red group centrals for best-fit 
        model

    grp_red_cen_stellar_mass_bf: array
        Array of red group central stellar masses for best-fit model

    blue_sigma_bf: array
        Array of velocity dispersion around blue group centrals for best-fit 
        model

    grp_blue_cen_stellar_mass_bf: array
        Array of blue group central stellar masses for best-fit model

    red_sigma_data: array
        Array of velocity dispersion around red group centrals for data

    grp_red_cen_stellar_mass_data: array
        Array of red group central stellar masses for data

    blue_sigma_data: array
        Array of velocity dispersion around blue group centrals for data

    grp_blue_cen_stellar_mass_data: array
        Array of blue group central stellar masses for data

    err_red: array
        Array of std values per bin of red group central stellar mass vs. 
        velocity dispersion from mocks

    err_blue: array
        Array of std values per bin of blue group central stellar mass vs. 
        velocity dispersion from mocks

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """   
    
    mean_grp_red_cen_gals_arr = []
    mean_grp_blue_cen_gals_arr = []
    red_num_arr = []
    blue_num_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        for idx in range(len(result[chunk_counter][0])):
            if level == 'halo':
                grp_red_cen_gals_idx = result[chunk_counter][32][idx]
                grp_blue_cen_gals_idx = result[chunk_counter][34][idx]
                red_num_idx = np.log10(result[chunk_counter][33][idx])
                blue_num_idx = np.log10(result[chunk_counter][35][idx])
            elif level == 'group':
                grp_red_cen_gals_idx = result[chunk_counter][28][idx]
                grp_blue_cen_gals_idx = result[chunk_counter][30][idx]
                red_num_idx = np.log10(result[chunk_counter][29][idx])
                blue_num_idx = np.log10(result[chunk_counter][31][idx])

            mean_stats_red = bs(red_num_idx, grp_red_cen_gals_idx, 
                statistic='mean', bins=np.arange(0,2.5,0.5))
            mean_stats_blue = bs(blue_num_idx, grp_blue_cen_gals_idx, 
                statistic='mean', bins=np.arange(0,0.8,0.2))
            red_num_arr.append(mean_stats_red[1])
            blue_num_arr.append(mean_stats_blue[1])
            mean_grp_red_cen_gals_arr.append(mean_stats_red[0])
            mean_grp_blue_cen_gals_arr.append(mean_stats_blue[0])

        chunk_counter+=1

    red_models_max = np.nanmax(mean_grp_red_cen_gals_arr, axis=0)
    red_models_min = np.nanmin(mean_grp_red_cen_gals_arr, axis=0)
    blue_models_max = np.nanmax(mean_grp_blue_cen_gals_arr, axis=0)
    blue_models_min = np.nanmin(mean_grp_blue_cen_gals_arr, axis=0)

    ## Same centers used for all sets of lines since binning is the same for 
    ## models, bf and data
    mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])

    # Max of red_num_bf = 62, Max of red_num_idx above varies (35-145)
    mean_stats_red_bf = bs(np.log10(red_num_bf), red_cen_stellar_mass_bf, 
        statistic='mean', bins=np.arange(0,2.5,0.5))
    # Max of blue_num_bf = 3, Max of blue_num_idx above varies (3-4)
    mean_stats_blue_bf = bs(np.log10(blue_num_bf), blue_cen_stellar_mass_bf, 
        statistic='mean', bins=np.arange(0,0.8,0.2))

    # Max of red_num_data = 388
    mean_stats_red_data = bs(np.log10(red_num_data), red_cen_stellar_mass_data_N, 
        statistic='mean', bins=np.arange(0,2.5,0.5))
    # Max of blue_num_data = 19
    mean_stats_blue_data = bs(np.log10(blue_num_data), blue_cen_stellar_mass_data_N, 
        statistic='mean', bins=np.arange(0,0.8,0.2))

    mean_stats_red_mocks_arr = []
    mean_stats_blue_mocks_arr = []
    for idx in range(len(red_cen_stellar_mass_mocks)):
        # Max of red_num_mocks[idx] = 724
        mean_stats_red_mocks = bs(np.log10(red_num_mocks[idx]), red_cen_stellar_mass_mocks[idx], 
            statistic='mean', bins=np.arange(0,2.5,0.5))
        mean_stats_red_mocks_arr.append(mean_stats_red_mocks[0])
    for idx in range(len(blue_cen_stellar_mass_mocks)):
        # Max of blue_num_mocks[idx] = 8
        mean_stats_blue_mocks = bs(np.log10(blue_num_mocks[idx]), blue_cen_stellar_mass_mocks[idx], 
            statistic='mean', bins=np.arange(0,0.8,0.2))
        mean_stats_blue_mocks_arr.append(mean_stats_blue_mocks[0])

    # fig1 = plt.figure(figsize=(11,9))
    # plt.hist(np.log10(red_num_bf), histtype='step', lw=3, label='Best-fit', 
    #     color='k', ls='--', zorder=10)
    # plt.hist(np.log10(red_num_data), histtype='step', lw=3, label='Data', 
    #     color='k', ls='-.', zorder=10)
    # for idx in range(len(red_cen_stellar_mass_mocks)):
    #     plt.hist(np.log10(red_num_mocks[idx]), histtype='step', lw=3)
    # plt.title('Histogram of log(number of galaxies in halo with red central)')
    # plt.legend()
    # plt.show()

    # fig2 = plt.figure(figsize=(11,9))
    # plt.hist(np.log10(blue_num_bf), histtype='step', lw=3, label='Best-fit', 
    #     color='k', ls='--', zorder=10)
    # plt.hist(np.log10(blue_num_data), histtype='step', lw=3, label='Data', 
    #     color='k', ls='-.', zorder=10)
    # for idx in range(len(blue_cen_stellar_mass_mocks)):
    #     plt.hist(np.log10(blue_num_mocks[idx]), histtype='step', lw=3)
    # plt.title('Histogram of log(number of galaxies in halo with blue central)')
    # plt.legend()
    # plt.show()

    ## Error bars on data points 
    std_mean_cen_arr_red = np.nanstd(mean_stats_red_mocks_arr, axis=0)
    std_mean_cen_arr_blue = np.nanstd(mean_stats_blue_mocks_arr, axis=0)

    fig1,ax1 = plt.subplots(figsize=(10,8))

    dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=std_mean_cen_arr_red,
            color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
    db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=std_mean_cen_arr_blue,
            color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
    
    mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
        y2=red_models_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
        y2=blue_models_min, color='cornflowerblue',alpha=0.4)

    bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
    bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

    l = plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
        ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

    chi_squared_red = np.nansum((mean_stats_red_data[0] - 
        mean_stats_red_bf[0])**2 / (std_mean_cen_arr_red**2))
    chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
        mean_stats_blue_bf[0])**2 / (std_mean_cen_arr_blue**2))

    plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
        r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
        chi_squared_red,2),np.round(chi_squared_blue,2)), 
        xy=(0.015, 0.7), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    plt.ylim(8.9, 11.5)   
    plt.xlabel(r'\boldmath$ \log_{10}\ {N} $', fontsize=30)
    plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    
    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_mean_N_vs_grpcen(result, red_num_bf, \
    red_cen_stellar_mass_bf, blue_num_bf, blue_cen_stellar_mass_bf, \
    red_num_data, red_cen_stellar_mass_data_N, blue_num_data, \
    blue_cen_stellar_mass_data_N, red_cen_stellar_mass_mocks, red_num_mocks, \
    blue_cen_stellar_mass_mocks, blue_num_mocks):
    """
    Plot average number of galaxies in halos/groups vs. halo/group central 
    stellar mass from data, best fit param values and param values corresponding 
    to 68th percentile 100 lowest chi^2 values.

    Parameters
    ----------
    result: multidimensional array
        Array of SMF, blue fraction and SMHM information

    red_sigma_bf: array
        Array of velocity dispersion around red group centrals for best-fit 
        model

    grp_red_cen_stellar_mass_bf: array
        Array of red group central stellar masses for best-fit model

    blue_sigma_bf: array
        Array of velocity dispersion around blue group centrals for best-fit 
        model

    grp_blue_cen_stellar_mass_bf: array
        Array of blue group central stellar masses for best-fit model

    red_sigma_data: array
        Array of velocity dispersion around red group centrals for data

    grp_red_cen_stellar_mass_data: array
        Array of red group central stellar masses for data

    blue_sigma_data: array
        Array of velocity dispersion around blue group centrals for data

    grp_blue_cen_stellar_mass_data: array
        Array of blue group central stellar masses for data

    err_red: array
        Array of std values per bin of red group central stellar mass vs. 
        velocity dispersion from mocks

    err_blue: array
        Array of std values per bin of blue group central stellar mass vs. 
        velocity dispersion from mocks

    bf_chi2: float
        Chi-squared value associated with the best-fit model

    Returns
    ---------
    Plot displayed on screen.
    """   

    stat = 'mean'

    red_stellar_mass_bins = np.linspace(8.6,10.7,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.7,6)

    grp_red_cen_gals_arr = []
    grp_blue_cen_gals_arr = []
    mean_red_num_arr = []
    mean_blue_num_arr = []
    chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
    while chunk_counter < 5:
        for idx in range(len(result[chunk_counter][0])):
            if level == 'halo':
                grp_red_cen_gals_idx = result[chunk_counter][32][idx]
                grp_blue_cen_gals_idx = result[chunk_counter][34][idx]
                red_num_idx = result[chunk_counter][33][idx]
                blue_num_idx = result[chunk_counter][35][idx]
            elif level == 'group':
                grp_red_cen_gals_idx = result[chunk_counter][28][idx]
                grp_blue_cen_gals_idx = result[chunk_counter][30][idx]
                red_num_idx = result[chunk_counter][29][idx]
                blue_num_idx = result[chunk_counter][31][idx]

            mean_stats_red = bs(grp_red_cen_gals_idx, red_num_idx,
                statistic=stat, bins=red_stellar_mass_bins)
            mean_stats_blue = bs(grp_blue_cen_gals_idx, blue_num_idx,
                statistic=stat, bins=blue_stellar_mass_bins)
            mean_red_num_arr.append(mean_stats_red[0])
            mean_blue_num_arr.append(mean_stats_blue[0])
            grp_red_cen_gals_arr.append(mean_stats_red[1])
            grp_blue_cen_gals_arr.append(mean_stats_blue[1])

        chunk_counter+=1

    red_models_max = np.nanmax(mean_red_num_arr, axis=0)
    red_models_min = np.nanmin(mean_red_num_arr, axis=0)
    blue_models_max = np.nanmax(mean_blue_num_arr, axis=0)
    blue_models_min = np.nanmin(mean_blue_num_arr, axis=0)

    ## Same centers used for all sets of lines since binning is the same for 
    ## models, bf and data
    mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
        mean_stats_red[1][:-1])
    mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
        mean_stats_blue[1][:-1])


    mean_stats_red_bf = bs(red_cen_stellar_mass_bf, red_num_bf, 
        statistic=stat, bins=red_stellar_mass_bins)
    mean_stats_blue_bf = bs(blue_cen_stellar_mass_bf, blue_num_bf,
        statistic=stat, bins=blue_stellar_mass_bins)

    mean_stats_red_data = bs(red_cen_stellar_mass_data_N, red_num_data,
        statistic=stat, bins=red_stellar_mass_bins)
    mean_stats_blue_data = bs(blue_cen_stellar_mass_data_N, blue_num_data,
        statistic=stat, bins=blue_stellar_mass_bins)

    mean_stats_red_mocks_arr = []
    mean_stats_blue_mocks_arr = []
    for idx in range(len(red_cen_stellar_mass_mocks)):
        mean_stats_red_mocks = bs(red_cen_stellar_mass_mocks[idx], red_num_mocks[idx],
            statistic=stat, bins=red_stellar_mass_bins)
        mean_stats_red_mocks_arr.append(mean_stats_red_mocks[0])
    for idx in range(len(blue_cen_stellar_mass_mocks)):
        mean_stats_blue_mocks = bs(blue_cen_stellar_mass_mocks[idx], blue_num_mocks[idx],
            statistic=stat, bins=blue_stellar_mass_bins)
        mean_stats_blue_mocks_arr.append(mean_stats_blue_mocks[0])

    ## Error bars on data points 
    std_mean_cen_arr_red = np.nanstd(mean_stats_red_mocks_arr, axis=0)
    std_mean_cen_arr_blue = np.nanstd(mean_stats_blue_mocks_arr, axis=0)

    fig1,ax1 = plt.subplots(figsize=(10,8))

    dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=std_mean_cen_arr_red,
            color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
    db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=std_mean_cen_arr_blue,
            color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
    
    mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
        y2=red_models_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
        y2=blue_models_min, color='cornflowerblue',alpha=0.4)

    bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
    bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

    l = plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
        ['Data','Models','Best-fit'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

    chi_squared_red = np.nansum((mean_stats_red_data[0] - 
        mean_stats_red_bf[0])**2 / (std_mean_cen_arr_red**2))
    chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
        mean_stats_blue_bf[0])**2 / (std_mean_cen_arr_blue**2))

    plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
        r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
        chi_squared_red,2),np.round(chi_squared_blue,2)), 
        xy=(0.015, 0.7), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    if stat == 'mean':
        plt.ylabel(r'\boldmath$\overline{N}$',fontsize=30)
    elif stat == 'median':
        plt.ylabel(r'\boldmath${N_{median}}$',fontsize=30)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

  
    if quenching == 'hybrid':
        plt.title('Hybrid quenching model | ECO')
    elif quenching == 'halo':
        plt.title('Halo quenching model | ECO')

    plt.show()

def plot_satellite_weighted_sigma(result, vdisp_red_bf, vdisp_blue_bf, \
    mstar_red_bf, mstar_blue_bf, nsat_red_bf, nsat_blue_bf, vdisp_red_data, \
    vdisp_blue_data, mstar_red_data, mstar_blue_data, \
    nsat_red_data, nsat_blue_data):

    def sw_func_red(arr):
        result = np.sum(arr)
        return result

    def sw_func_blue(arr):
        result = np.sum(arr)
        return result

    def hw_func_red(arr):
        result = np.sum(arr)/len(arr)
        return result

    def hw_func_blue(arr):
        result = np.sum(arr)/len(arr)
        return result

    if level == 'group':
        vdisp_red_idx = 32
        vdisp_blue_idx = 34
        cen_red_idx = 33
        cen_blue_idx = 35
        nsat_red_idx = 36
        nsat_blue_idx = 37
    elif level == 'halo':
        vdisp_red_idx = 38
        vdisp_blue_idx = 40
        cen_red_idx = 39
        cen_blue_idx = 41
        nsat_red_idx = 42
        nsat_blue_idx = 43
       
    i_outer = 0
    vdisp_red_mod_arr = []
    vdisp_blue_mod_arr = []
    cen_mstar_red_mod_arr = []
    cen_mstar_blue_mod_arr = []
    nsat_red_mod_arr = []
    nsat_blue_mod_arr = []
    while i_outer < 5:
        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][vdisp_red_idx][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][vdisp_blue_idx][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][cen_red_idx][idx]
            cen_mstar_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][cen_blue_idx][idx]
            cen_mstar_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][nsat_red_idx][idx]
            nsat_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][nsat_blue_idx][idx]
            nsat_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][vdisp_red_idx][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][vdisp_blue_idx][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][cen_red_idx][idx]
            cen_mstar_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][cen_blue_idx][idx]
            cen_mstar_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][nsat_red_idx][idx]
            nsat_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][nsat_blue_idx][idx]
            nsat_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][vdisp_red_idx][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][vdisp_blue_idx][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][cen_red_idx][idx]
            cen_mstar_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][cen_blue_idx][idx]
            cen_mstar_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][nsat_red_idx][idx]
            nsat_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][nsat_blue_idx][idx]
            nsat_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][vdisp_red_idx][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][vdisp_blue_idx][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][cen_red_idx][idx]
            cen_mstar_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][cen_blue_idx][idx]
            cen_mstar_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][nsat_red_idx][idx]
            nsat_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][nsat_blue_idx][idx]
            nsat_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

        for idx in range(len(result[i_outer][0])):
            red_mod_ii = result[i_outer][vdisp_red_idx][idx]
            vdisp_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][vdisp_blue_idx][idx]
            vdisp_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][cen_red_idx][idx]
            cen_mstar_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][cen_blue_idx][idx]
            cen_mstar_blue_mod_arr.append(blue_mod_ii)

            red_mod_ii = result[i_outer][nsat_red_idx][idx]
            nsat_red_mod_arr.append(red_mod_ii)
            blue_mod_ii = result[i_outer][nsat_blue_idx][idx]
            nsat_blue_mod_arr.append(blue_mod_ii)
        i_outer += 1

    vdisp_red_mod_arr = np.array(vdisp_red_mod_arr, dtype=object)
    vdisp_blue_mod_arr = np.array(vdisp_blue_mod_arr, dtype=object)
    cen_mstar_red_mod_arr = np.array(cen_mstar_red_mod_arr, dtype=object)
    cen_mstar_blue_mod_arr = np.array(cen_mstar_blue_mod_arr, dtype=object)
    nsat_red_mod_arr = np.array(nsat_red_mod_arr, dtype=object)
    nsat_blue_mod_arr = np.array(nsat_blue_mod_arr, dtype=object)

    red_stellar_mass_bins = np.linspace(8.6,11.2,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.7,6)

    ## Models
    ratio_red_mod_arr = []
    ratio_blue_mod_arr = []
    for idx in range(len(vdisp_red_mod_arr)):
        # nsat_red_total = np.sum(nsat_red_mod_arr[idx])
        # nsat_blue_total = np.sum(nsat_blue_mod_arr[idx])

        nsat_red = bs(cen_mstar_red_mod_arr[idx], nsat_red_mod_arr[idx], 'sum', 
            bins=red_stellar_mass_bins)
        nsat_blue = bs(cen_mstar_blue_mod_arr[idx], nsat_blue_mod_arr[idx], 'sum',
            bins=blue_stellar_mass_bins)

        nsat_vdisp_product_red = np.array(nsat_red_mod_arr[idx]) * \
            (np.array(vdisp_red_mod_arr[idx])**2)
        nsat_vdisp_product_blue = np.array(nsat_blue_mod_arr[idx]) * \
            (np.array(vdisp_blue_mod_arr[idx])**2)

        sw_mean_stats_red_mod = bs(cen_mstar_red_mod_arr[idx], 
            nsat_vdisp_product_red, statistic=sw_func_red, 
            bins=red_stellar_mass_bins)
        ## Nsat is number of satellites stacked in a bin in More et al Eq. 1
        sw_mean_stats_red_mod_ii = sw_mean_stats_red_mod[0]/nsat_red[0]


        sw_mean_stats_blue_mod = bs(cen_mstar_blue_mod_arr[idx], 
            nsat_vdisp_product_blue, statistic=sw_func_blue, 
            bins=blue_stellar_mass_bins)
        sw_mean_stats_blue_mod_ii = sw_mean_stats_blue_mod[0]/nsat_blue[0]


        hw_mean_stats_red_mod = bs(cen_mstar_red_mod_arr[idx], 
            np.array(vdisp_red_mod_arr[idx])**2, statistic=hw_func_red, 
            bins=red_stellar_mass_bins)

        hw_mean_stats_blue_mod = bs(cen_mstar_blue_mod_arr[idx], 
            np.array(vdisp_blue_mod_arr[idx])**2, statistic=hw_func_blue, 
            bins=blue_stellar_mass_bins)
        
        ratio_red_mod = np.log10(hw_mean_stats_red_mod[0])
        ratio_blue_mod = np.log10(hw_mean_stats_blue_mod[0])

        ratio_red_mod_arr.append(ratio_red_mod)
        ratio_blue_mod_arr.append(ratio_blue_mod)

    ratio_red_mod_max = np.nanmax(ratio_red_mod_arr, axis=0)
    ratio_red_mod_min = np.nanmin(ratio_red_mod_arr, axis=0)
    ratio_blue_mod_max = np.nanmax(ratio_blue_mod_arr, axis=0)
    ratio_blue_mod_min = np.nanmin(ratio_blue_mod_arr, axis=0)


    ## Best-fit
    # nsat_red_total = np.sum(nsat_red_bf)
    # nsat_blue_total = np.sum(nsat_blue_bf)

    nsat_vdisp_product_red = np.array(nsat_red_bf) * (np.array(vdisp_red_bf)**2)
    nsat_vdisp_product_blue = np.array(nsat_blue_bf) * (np.array(vdisp_blue_bf)**2)

    nsat_red = bs(mstar_red_bf, nsat_red_bf, 'sum', 
        bins=red_stellar_mass_bins)
    nsat_blue = bs(mstar_blue_bf, nsat_blue_bf, 'sum',
        bins=blue_stellar_mass_bins)

    sw_mean_stats_red = bs(mstar_red_bf, nsat_vdisp_product_red,
        statistic=sw_func_red, bins=red_stellar_mass_bins)
    sw_mean_stats_red_bf = sw_mean_stats_red[0]/nsat_red[0]

    sw_mean_stats_blue = bs(mstar_blue_bf, nsat_vdisp_product_blue,
        statistic=sw_func_blue, bins=blue_stellar_mass_bins)
    sw_mean_stats_blue_bf = sw_mean_stats_blue[0]/nsat_blue[0]

    hw_mean_stats_red = bs(mstar_red_bf, np.array(vdisp_red_bf)**2,
        statistic=hw_func_red, bins=red_stellar_mass_bins)

    hw_mean_stats_blue = bs(mstar_blue_bf, np.array(vdisp_blue_bf)**2,
        statistic=hw_func_blue, bins=blue_stellar_mass_bins)

    ## Data
    # nsat_red_total = np.sum(nsat_red_data)
    # nsat_blue_total = np.sum(nsat_blue_data)

    nsat_vdisp_product_red_data = np.array(nsat_red_data) * (np.array(vdisp_red_data)**2)
    nsat_vdisp_product_blue_data = np.array(nsat_blue_data) * (np.array(vdisp_blue_data)**2)

    nsat_red = bs(mstar_red_data, nsat_red_data, 'sum', 
        bins=red_stellar_mass_bins)
    nsat_blue = bs(mstar_blue_data, nsat_blue_data, 'sum',
        bins=blue_stellar_mass_bins)

    sw_mean_stats_red = bs(mstar_red_data, nsat_vdisp_product_red_data,
        statistic=sw_func_red, bins=red_stellar_mass_bins)
    sw_mean_stats_red_data = sw_mean_stats_red[0]/nsat_red[0]

    sw_mean_stats_blue = bs(mstar_blue_data, nsat_vdisp_product_blue_data,
        statistic=sw_func_blue, bins=blue_stellar_mass_bins)
    sw_mean_stats_blue_data = sw_mean_stats_blue[0]/nsat_blue[0]

    hw_mean_stats_red_data = bs(mstar_red_data, np.array(vdisp_red_data)**2,
        statistic=hw_func_red, bins=red_stellar_mass_bins)

    hw_mean_stats_blue_data = bs(mstar_blue_data, np.array(vdisp_blue_data)**2,
        statistic=hw_func_blue, bins=blue_stellar_mass_bins)

    ## Centers the same for data, models and best-fit since red and blue bins 
    ## are the same for all 3 cases
    centers_red = 0.5 * (sw_mean_stats_red[1][1:] + \
        sw_mean_stats_red[1][:-1])
    centers_blue = 0.5 * (sw_mean_stats_blue[1][1:] + \
        sw_mean_stats_blue[1][:-1])

    fig1 = plt.figure(figsize=(11, 9))    
    # const = 1/(0.9*(2/3))
    # bfr, = plt.plot(centers_red, const*np.log10(sw_mean_stats_red[0]/hw_mean_stats_red[0]), c='maroon', lw=3, zorder = 10)
    # bfb, = plt.plot(centers_blue, const*np.log10(sw_mean_stats_blue[0]/hw_mean_stats_blue[0]), c='darkblue', lw=3, zorder=10)

    ## Ratio of satellite/host weighted
    # bfr, = plt.plot(centers_red, 
    #     np.log10(sw_mean_stats_red_bf/hw_mean_stats_red[0]), c='maroon', lw=3, 
    #     zorder = 10)
    # bfb, = plt.plot(centers_blue, 
    #     np.log10(sw_mean_stats_blue_bf/hw_mean_stats_blue[0]), c='darkblue', 
    #     lw=3, zorder=10)

    # dr = plt.scatter(centers_red, 
    #     np.log10(sw_mean_stats_red_data/hw_mean_stats_red_data[0]), 
    #     c='indianred', s=300, marker='p', zorder = 20)
    # db = plt.scatter(centers_blue, 
    #     np.log10(sw_mean_stats_blue_data/hw_mean_stats_blue_data[0]), 
    #     c='royalblue', s=300, marker='p', zorder=20)

    # mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
    #     y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
    # mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
    #     y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)

    # ## Satellite weighted
    # bfr, = plt.plot(centers_red, np.log10(sw_mean_stats_red_bf), c='maroon', lw=3, zorder = 10)
    # bfb, = plt.plot(centers_blue, np.log10(sw_mean_stats_blue_bf), c='darkblue', lw=3, zorder=10)

    # dr = plt.scatter(centers_red, 
    #     np.log10(sw_mean_stats_red_data), 
    #     c='indianred', s=300, marker='p', zorder = 20)
    # db = plt.scatter(centers_blue, 
    #     np.log10(sw_mean_stats_blue_data), 
    #     c='royalblue', s=300, marker='p', zorder=20)

    # mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
    #     y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
    # mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
    #     y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)

    ## Host weighted
    bfr, = plt.plot(centers_red, np.log10(hw_mean_stats_red[0]), c='maroon', lw=3, zorder = 10)
    bfb, = plt.plot(centers_blue, np.log10(hw_mean_stats_blue[0]), c='darkblue', lw=3, zorder=10)

    dr = plt.scatter(centers_red, 
        np.log10(hw_mean_stats_red_data[0]), 
        c='indianred', s=300, marker='p', zorder = 20)
    db = plt.scatter(centers_blue, 
        np.log10(hw_mean_stats_blue_data[0]), 
        c='royalblue', s=300, marker='p', zorder=20)

    mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
        y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
        y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)


    mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
        y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
    mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
        y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)
    
    plt.legend([(bfr, bfb), (dr, db), (mr, mb)], 
        ['Best-fit', 'Data', 'Models'],
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
        loc='best')

    plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    # plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{sw}^2 / \sigma_{hw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)
    # plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{sw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{hw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)


    if quenching == 'halo':
        plt.title('Host weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
        # plt.title('Satellite weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
        # plt.title('Ratio of satellite-host weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
    elif quenching == 'hybrid':
        plt.title('Host weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
        # plt.title('Satellite weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
        # plt.title('Ratio of satellite-host weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
    plt.show()


def main():
    global survey
    global quenching
    global model_init
    global many_behroozi_mocks
    global gal_group_df_subset
    global dof
    global level

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_data = dict_of_paths['data_dir']

    many_behroozi_mocks = False
    quenching = 'hybrid'
    level = 'group'
    machine = 'mac'
    mf_type = 'smf'
    survey = 'eco'
    nproc = 2

    if quenching == 'halo':
        run = 33
    elif quenching == 'hybrid':
        run = 32

    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    chi2_file = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
        format(run, survey)
    chain_file = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
        format(run, survey)

    if survey == 'eco':
        # catl_file = path_to_raw + "eco/eco_all.csv"
        ## New catalog with group finder run on subset after applying M* and cz cuts
        catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    print('Reading files')
    chi2 = read_chi2(chi2_file)
    mcmc_table = read_mcmc(chain_file)
    catl, volume, z_median = read_data_catl(catl_file, survey)
    ## Group finder run on subset after applying M* cut 8.6 and cz cut 3000-12000
    gal_group_run32 = read_mock_catl(path_to_proc + "gal_group_run{0}.hdf5".format(run)) 

    idx_arr = np.insert(np.linspace(0,20,21), len(np.linspace(0,20,21)), (22, 123, 
        124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134)).astype(int)

    names_arr = [x for x in gal_group_run32.columns.values[idx_arr]]
    for idx in np.arange(2,101,1):
        names_arr.append('{0}_y'.format(idx))
        names_arr.append('groupid_{0}'.format(idx))
        names_arr.append('g_galtype_{0}'.format(idx))
    names_arr = np.array(names_arr)

    gal_group_df_subset = gal_group_run32[names_arr]

    # Renaming the "1_y" column kept from line 1896 because of case where it was
    # also in mcmc_table_ptcl.mock_num and was selected twice
    gal_group_df_subset.columns.values[30] = "behroozi_bf"

    ### Removing "_y" from column names for stellar mass
    # Have to remove the first element because it is 'halo_y' column name
    cols_with_y = np.array([[idx, s] for idx, s in enumerate(gal_group_df_subset.columns.values) if '_y' in s][1:])
    colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
    gal_group_df_subset.columns.values[cols_with_y[:,0].astype(int)] = colnames_without_y

    print('Getting data in specific percentile')
    # get_paramvals called to get bf params and chi2 values
    mcmc_table_pctl, bf_params, bf_chi2 = \
        get_paramvals_percentile(mcmc_table, 68, chi2)
    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter', \
        'mstar_q', 'mh_q', 'mu', 'nu']

    mcmc_table_pctl_subset = pd.read_csv(path_to_proc + 'run{0}_params_subset.txt'.format(run), 
        delim_whitespace=True, names=colnames).iloc[1:,:].reset_index(drop=True)

    print('Assigning colour to data')
    catl = assign_colour_label_data(catl)

    print('Measuring SMF for data')
    total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

    print('Measuring blue fraction for data')
    f_blue = blue_frac(catl, False, True)

    print('Measuring reconstructed red and blue SMF for data')
    phi_red_data, phi_blue_data = get_colour_smf_from_fblue(catl, f_blue[1], 
        f_blue[0], volume, False)

    print('Measuring new dynamical metric for data')
    red_sigma_data, red_cen_stellar_mass_data_sigma, blue_sigma_data, \
        blue_cen_stellar_mass_data_sigma = get_sigma_per_group_data(catl)

    # Returns masses in h=1.0
    print('Measuring vel disp for data')
    veldisp_red, veldisp_centers_red, veldisp_blue, veldisp_centers_blue = \
        get_deltav_sigma_data(catl)

    print('Measuring number of galaxies in groups for data')
    red_num_data, red_cen_stellar_mass_data_N, blue_num_data, \
        blue_cen_stellar_mass_data_N = get_N_per_group_data(catl, central_bool=True)

    print('Measuring satellite weighted velocity dispersion for data')
    wtd_red_sigma_data, red_cen_stellar_mass_data, wtd_blue_sigma_data, \
        blue_cen_stellar_mass_data, red_nsat_data, blue_nsat_data = \
        get_satellite_weighted_sigma_data(catl)

    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Measuring error in data from mocks')
    err_data, err_phi_red, err_phi_blue, err_std_red, err_std_blue, err_vdisp_red, \
        err_vdisp_blue, red_cen_stellar_mass_mocks, red_num_mocks, \
        blue_cen_stellar_mass_mocks, blue_num_mocks = \
        get_err_data(survey, path_to_mocks)

    dof = len(err_data) - len(bf_params)

    print('Getting best fit model')
    if level == 'halo':
        maxis_bf_total, phi_bf_total, maxis_bf_fblue, bf_fblue, phi_bf_red, \
            phi_bf_blue, cen_gals_red, \
            cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
            f_red_cen_blue, sat_gals_red_bf, sat_halos_red_bf, sat_gals_blue_bf, \
            sat_halos_blue_bf, f_red_sat_red_bf, f_red_sat_blue_bf, cen_gals_bf, \
            cen_halos_bf, red_sigma_bf, grp_red_cen_stellar_mass_bf, \
            blue_sigma_bf, grp_blue_cen_stellar_mass_bf, vdisp_red_bf, vdisp_blue_bf, \
            vdisp_centers_red_bf, vdisp_centers_blue_bf, vdisp_red_points_bf, \
            vdisp_blue_points_bf, red_host_halo_mass_bf_sigma_mh, \
            blue_host_halo_mass_bf_sigma_mh, red_cen_stellar_mass_bf, red_num_bf, \
            blue_cen_stellar_mass_bf, blue_num_bf, red_host_halo_mass_bf_N_mh, \
            blue_host_halo_mass_bf_N_mh, wtd_red_sigma_bf, \
            wtd_red_cen_stellar_mass_bf, wtd_blue_sigma_bf, \
            wtd_blue_cen_stellar_mass_bf, wtd_red_nsat_bf, wtd_blue_nsat_bf = \
            get_best_fit_model(bf_params)

    elif level == 'group':
        maxis_bf_total, phi_bf_total, maxis_bf_fblue, bf_fblue, phi_bf_red, \
            phi_bf_blue, cen_gals_red, \
            cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
            f_red_cen_blue, sat_gals_red_bf, sat_halos_red_bf, sat_gals_blue_bf, \
            sat_halos_blue_bf, f_red_sat_red_bf, f_red_sat_blue_bf, cen_gals_bf, \
            cen_halos_bf, red_sigma_bf, grp_red_cen_stellar_mass_bf, \
            blue_sigma_bf, grp_blue_cen_stellar_mass_bf, vdisp_red_bf, vdisp_blue_bf, \
            vdisp_centers_red_bf, vdisp_centers_blue_bf, red_cen_stellar_mass_bf, \
            red_num_bf, blue_cen_stellar_mass_bf, blue_num_bf, wtd_red_sigma_bf, \
            wtd_red_cen_stellar_mass_bf, wtd_blue_sigma_bf, \
            wtd_blue_cen_stellar_mass_bf, wtd_red_nsat_bf, wtd_blue_nsat_bf = \
            get_best_fit_model(bf_params)

    print('Multiprocessing') #~18 minutes
    result = mp_init(mcmc_table_pctl_subset, nproc)

    print('Plotting')
    plot_total_mf(result, total_data, maxis_bf_total, phi_bf_total, 
        bf_chi2, err_data)

    plot_fblue(result, f_blue, maxis_bf_fblue, bf_fblue, bf_chi2, err_data)

    plot_colour_mf(result, phi_red_data, phi_blue_data, phi_bf_red, phi_bf_blue, 
        err_phi_red, err_phi_blue , bf_chi2)

    plot_xmhm(result, cen_gals_bf, cen_halos_bf, bf_chi2)

    plot_colour_xmhm(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
        cen_halos_blue, bf_chi2)

    plot_colour_hmxm(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
        cen_halos_blue, bf_chi2)

    plot_red_fraction_cen(result, cen_gals_red, \
        cen_halos_red, cen_gals_blue, cen_halos_blue, f_red_cen_red, \
        f_red_cen_blue)

    plot_red_fraction_sat(result, sat_gals_red_bf, \
        sat_halos_red_bf, sat_gals_blue_bf, sat_halos_blue_bf, f_red_sat_red_bf, \
        f_red_sat_blue_bf)

    plot_zumand_fig4(result, cen_gals_red, cen_halos_red, cen_gals_blue, 
        cen_halos_blue, bf_chi2)

    plot_mean_grpcen_vs_sigma(result, red_sigma_bf, grp_red_cen_stellar_mass_bf, \
        blue_sigma_bf, grp_blue_cen_stellar_mass_bf, red_sigma_data, \
        red_cen_stellar_mass_data_sigma, blue_sigma_data, \
        blue_cen_stellar_mass_data_sigma, err_std_red, err_std_blue, bf_chi2)

    plot_sigma_vdiff_mod(result, veldisp_red, veldisp_centers_red, veldisp_blue, \
        veldisp_centers_blue, vdisp_red_bf, vdisp_blue_bf, vdisp_centers_red_bf, \
        vdisp_centers_blue_bf, bf_chi2, err_vdisp_red, err_vdisp_blue)

    plot_sigma_host_halo_mass_vishnu(result, vdisp_red_points_bf, \
        vdisp_blue_points_bf, red_host_halo_mass_bf_sigma_mh, \
        blue_host_halo_mass_bf_sigma_mh)

    plot_N_host_halo_mass_vishnu(result, red_num_bf, \
        blue_num_bf, red_host_halo_mass_bf_N_mh, \
        blue_host_halo_mass_bf_N_mh)

    plot_mean_grpcen_vs_N(result, red_num_bf, \
        red_cen_stellar_mass_bf, blue_num_bf, blue_cen_stellar_mass_bf, \
        red_num_data, red_cen_stellar_mass_data_N, blue_num_data, \
        blue_cen_stellar_mass_data_N, red_cen_stellar_mass_mocks, red_num_mocks, \
        blue_cen_stellar_mass_mocks, blue_num_mocks)

    plot_mean_N_vs_grpcen(result, red_num_bf, \
        red_cen_stellar_mass_bf, blue_num_bf, blue_cen_stellar_mass_bf, \
        red_num_data, red_cen_stellar_mass_data_N, blue_num_data, \
        blue_cen_stellar_mass_data_N, red_cen_stellar_mass_mocks, red_num_mocks, \
        blue_cen_stellar_mass_mocks, blue_num_mocks)

    plot_satellite_weighted_sigma(result, wtd_red_sigma_bf, \
        wtd_blue_sigma_bf, wtd_red_cen_stellar_mass_bf, \
        wtd_blue_cen_stellar_mass_bf, wtd_red_nsat_bf, \
        wtd_blue_nsat_bf,\
        wtd_red_sigma_data, wtd_blue_sigma_data, red_cen_stellar_mass_data, \
        blue_cen_stellar_mass_data, red_nsat_data, blue_nsat_data)


if __name__ == 'main':
    main()

# ### Move to plotting function sigma_vdiff_mod from get_deltav functions
# if Settings.survey == 'eco' or Settings.survey == 'resolvea':
#     # TODO : check if this is actually correct for resolve a
#     red_stellar_mass_bins = np.linspace(8.6,11.2,6)
# elif Settings.survey == 'resolveb':
#     red_stellar_mass_bins = np.linspace(8.4,11.0,6)

# mean_stats_red = bs(red_cen_stellar_mass_arr, red_sigma_arr,
#     statistic='mean', bins=red_stellar_mass_bins)
# std_red = mean_stats_red[0]

# if survey == 'eco' or survey == 'resolvea':
#     # TODO : check if this is actually correct for resolve a
#     blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
# elif survey == 'resolveb':
#     blue_stellar_mass_bins = np.linspace(8.4,10.4,6)

# mean_stats_blue = bs(blue_cen_stellar_mass_arr, blue_sigma_arr,
#     statistic='mean', bins=blue_stellar_mass_bins)
# std_blue = mean_stats_blue[0]

# centers_red = 0.5 * (mean_stats_red[1][1:] + \
#     mean_stats_red[1][:-1])
# centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
#     mean_stats_blue[1][:-1])


# ### Move to plotting function mean_grpccen_vs_sigma from sigma_per_group_mocks

# if catl_type == 'mock':
#     mean_stats_red = bs(red_sigma_arr, red_cen_stellar_mass_arr, 
#         statistic='mean', bins=np.linspace(0,250,6))
#     mean_stats_blue = bs(blue_sigma_arr, blue_cen_stellar_mass_arr, 
#         statistic='mean', bins=np.linspace(0,250,6))

#     centers_red = 0.5 * (mean_stats_red[1][1:] + \
#         mean_stats_red[1][:-1])
#     centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
#         mean_stats_blue[1][:-1])
    
#     return mean_stats_red, centers_red, mean_stats_blue, centers_blue
