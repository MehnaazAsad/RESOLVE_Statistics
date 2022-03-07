"""
{This script is a playground for observables being added to the mcmc pipeline}
"""
__author__ = '{Mehnaaz Asad}'

from turtle import st
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binned_statistic as bs
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)



def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

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
        ## Use group level for data even when level == halo
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
            max_cz = 7000
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

    red_singleton_counter = 0
    red_sigma_arr = []
    red_cen_stellar_mass_arr = []
    red_group_mass_arr = []
    # red_sigmagapper_arr = []
    red_nsat_arr = []
    red_keys_arr = []
    red_host_halo_mass_arr = []
    for key in red_subset_ids: 
        group = catl.loc[catl[id_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            red_keys_arr.append(key)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            group_stellar_mass = np.log10(np.sum(10**group[logmstar_col].values))
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                #So that the array is returned either way without the need for 
                #two separate return statements
                host_halo_mass = 0 
            nsat = len(group.loc[group[galtype_col].values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            sigma = deltav.std()
            # red_sigmagapper = np.unique(group.grpsig.values)[0]
            
            red_sigma_arr.append(sigma)
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_group_mass_arr.append(group_stellar_mass)
            # red_sigmagapper_arr.append(red_sigmagapper)
            red_nsat_arr.append(nsat)
            red_host_halo_mass_arr.append(host_halo_mass)

    blue_singleton_counter = 0
    blue_sigma_arr = []
    blue_cen_stellar_mass_arr = []
    blue_group_mass_arr = []
    # blue_sigmagapper_arr = []
    blue_nsat_arr = []
    blue_keys_arr = []
    blue_host_halo_mass_arr = []
    for key in blue_subset_ids: 
        group = catl.loc[catl[id_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            blue_keys_arr.append(key)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            group_stellar_mass = np.log10(np.sum(10**group[logmstar_col].values))
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                #So that the array is returned either way without the need for 
                #two separate return statements
                host_halo_mass = 0 

            nsat = len(group.loc[group[galtype_col].values == 0])
            # Different velocity definitions
            mean_cz_grp = np.round(np.mean(group.cz.values),2)
            cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
            # cz_grp = np.unique(group.grpcz.values)[0]

            # Velocity difference
            deltav = group.cz.values - len(group)*[cen_cz_grp]
            # sigma = deltav[deltav!=0].std()
            sigma = deltav.std()
            # blue_sigmagapper = np.unique(group.grpsig.values)[0]
            
            blue_sigma_arr.append(sigma)
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_group_mass_arr.append(group_stellar_mass)
            # blue_sigmagapper_arr.append(blue_sigmagapper)
            blue_nsat_arr.append(nsat)
            blue_host_halo_mass_arr.append(host_halo_mass)


    return red_keys_arr, red_sigma_arr, red_cen_stellar_mass_arr, red_nsat_arr, \
        blue_keys_arr, blue_sigma_arr, blue_cen_stellar_mass_arr, blue_nsat_arr

def get_richness(catl, catl_type, randint=None, central_bool=None):

    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            catl = catl.loc[catl.logmstar >= 8.9]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

    if catl_type == 'data' or catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        logmstar_col = 'logmstar'
        ## Use group level for data even when level == halo
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
            max_cz = 7000
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

    red_singleton_counter = 0
    red_num_arr = []
    red_cen_stellar_mass_arr = []
    red_host_halo_mass_arr = []
    red_group_halo_mass_arr = []
    red_group_stellar_mass_arr = []
    red_keys_arr = []
    for key in red_subset_ids: 
        group = catl.loc[catl[id_col] == key]
        if len(group) == 1:
            red_singleton_counter += 1
        else:
            red_keys_arr.append(key)
            # print(logmstar_col)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            group_stellar_mass = np.log10(np.sum(10**group[logmstar_col].values))
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                #So that the array is returned either way without the need for 
                #two separate return statements
                host_halo_mass = 0 

            #* This works for both model and data since abundance matching was done
            #* on best-fit model to calculate group halo mass using group Mstar
            group_halo_mass = np.unique(group.M_group.values)[0]

            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_num_arr.append(num)
            red_host_halo_mass_arr.append(host_halo_mass)
            red_group_stellar_mass_arr.append(group_stellar_mass)
            red_group_halo_mass_arr.append(group_halo_mass)

    blue_singleton_counter = 0
    blue_num_arr = []
    blue_cen_stellar_mass_arr = []
    blue_host_halo_mass_arr = []
    blue_group_halo_mass_arr = []
    blue_group_stellar_mass_arr = []
    blue_keys_arr = []
    for key in blue_subset_ids: 
        group = catl.loc[catl[id_col] == key]
        if len(group) == 1:
            blue_singleton_counter += 1
        else:
            blue_keys_arr.append(key)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]

            group_stellar_mass = np.log10(np.sum(10**group[logmstar_col].values))

            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                host_halo_mass = 0 

            if catl_type == 'data':
                group_halo_mass = np.unique(group.M_group.values)[0]
            else:
                # group_halo_mass = host_halo_mass # Calculated above for catl_type model
                group_halo_mass = np.unique(group.M_group.values)[0]
            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1

            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_num_arr.append(num)
            blue_host_halo_mass_arr.append(host_halo_mass)
            blue_group_stellar_mass_arr.append(group_stellar_mass)
            blue_group_halo_mass_arr.append(group_halo_mass)

    if level == 'group':
        return red_keys_arr, red_num_arr, red_group_halo_mass_arr, \
            red_cen_stellar_mass_arr, blue_keys_arr, blue_num_arr, \
            blue_group_halo_mass_arr, blue_cen_stellar_mass_arr
    elif level == 'halo':
        return red_keys_arr, red_num_arr, red_host_halo_mass_arr, \
            red_cen_stellar_mass_arr, blue_keys_arr, blue_num_arr, \
            blue_host_halo_mass_arr, blue_cen_stellar_mass_arr

def get_vdf(red_sigma, blue_sigma, volume):
    bins = np.linspace(0, 250, 7)
    # Unnormalized histogram and bin edges
    counts_red, edg = np.histogram(red_sigma, bins=bins)  # paper used 17 bins
    counts_blue, edg = np.histogram(blue_sigma, bins=bins)  # paper used 17 bins

    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss_red = np.sqrt(counts_red) / (volume * dm)
    err_poiss_blue = np.sqrt(counts_blue)/ (volume * dm)
    phi_red = counts_red / (volume * dm)  # not a log quantity
    phi_blue = counts_blue / (volume * dm)  # not a log quantity
    phi_red = np.log10(phi_red)
    phi_blue = np.log10(phi_blue)
    return maxis, [phi_red, phi_blue], [err_poiss_red, err_poiss_blue], bins, [counts_red, counts_blue]

def abundance_matching_f(dict1, dict2, dict1_names=None, dict2_names=None,
    volume1=None, volume2=None, reverse=True, dens1_opt=False,
    dens2_opt=False):
    """
    Performs abundance matching based on two quantities (dict1 and dict2).
    It assigns values from `dict2` to elements in `dict1`.
    Parameters
    ----------
    dict1: dictionary_like, or array_like
        Dictionary or array-like object of property 1
        If `dens1_opt == True`:
            - Object is a dictionary consisting of the following keys:
                - 'dict1_names': shape (2,)
                - Order of `dict1_names`: [var1_value, dens1_value]
        else:
            - Object is a 1D array or list
            - Density must be calculated for var2
    dict2: dictionary_like, or array_like
        Dictionary or array-like object of property 2
        If `dens2_opt == True`:
            - Object is a dictionary consisting of the following keys:
                - 'dict2_names': shape (2,)
                - Order of `dict2_names`: [var2_value, dens2_value]
        else:
            - Object is a 1D array or list
            - Density must be calculated for var2
    dict1_names: NoneType, or array_list with shape (2,), optional (def: None)
        names of the `dict1` keys, in order of [var1_value, dens1_value]
    dict2_names: NoneType, or array_list with shape (2,), optional (def: None)
        names of the `dict2` keys, in order of [var2_value, dens2_value]
    volume1: NoneType or float, optional (default = None)
        Volume of the 1st variable `var1`
        Required if `dens1_opt == False`
    volume2: NoneType or float, optional (default = None)
        Volume of the 2nd variable `var2`
        Required if `dens2_opt == False`
    reverse: boolean
        Determines the relation between var1 and var2.
        - reverse==True: var1 increases with increasing var2
        - reverse==False: var1 decreases with increasing var2
    dens1_opt: boolean, optional (default = False)
        - If 'True': density is already provided as key for `dict1`
        - If 'False': density must me calculated
        - `dict1_names` must be provided and have length (2,)
    dens2_opt: boolean, optional (default = False)
        - If 'True': density is already provided as key for `dict2`
        - If 'False': density must me calculated
        - `dict2_names` must be provided and have length (2,)
    Returns
    -------
    var1_ab: array_like
        numpy.array of elements matching those of `dict1`, after matching with
        dict2.
    """
    ## Checking input parameters
    # 1st property
    if dens1_opt:
        assert(len(dict1_names) == 2)
        var1  = np.array(dict1[dict1_names[0]])
        dens1 = np.array(dict1[dict1_names[1]])
    else:
        var1        = np.array(dict1)
        assert(volume1 != None)
        ## Calculating Density for `var1`
        if reverse:
            ncounts1 = np.array([np.where(var1<xx)[0].size for xx in var1])+1
        else:
            ncounts1 = np.array([np.where(var1>xx)[0].size for xx in var1])+1
        dens1 = ncounts1.astype(float)/volume1
    # 2nd property
    if dens2_opt:
        assert(len(dict2_names) == 2)
        var2  = np.array(dict2[dict2_names[0]])
        dens2 = np.array(dict2[dict2_names[1]])
    else:
        var2        = np.array(dict2)
        assert(volume2 != None)
        ## Calculating Density for `var1`
        if reverse:
            ncounts2 = np.array([np.where(var2<xx)[0].size for xx in var2])+1
        else:
            ncounts2 = np.array([np.where(var2>xx)[0].size for xx in var2])+1
        dens2 = ncounts2.astype(float)/volume2
    ##
    ## Interpolating densities and values
    interp_var2 = interp1d(dens2, var2, bounds_error=True,assume_sorted=False)
    # Value assignment
    var1_ab = np.array([interp_var2(xx) for xx in dens1])

    return var1_ab

def group_mass_assignment(mockgal_pd, mockgroup_pd, param_dict):
    """
    Assigns a theoretical halo mass to the group based on a group property
    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID
    mockgroup_pd: pandas DataFrame
        DataFame containing information for each galaxy group
    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    -----------
    mockgal_pd_new: pandas DataFrame
        Original info + abundance matched mass of the group, M_group
    mockgroup_pd_new: pandas DataFrame
        Original info of `mockgroup_pd' + abundance matched mass, M_group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Mass Assign. ....')
    ## Copies of DataFrames
    gal_pd   = mockgal_pd.copy()
    group_pd = mockgroup_pd.copy()

    # ## Changing stellar mass to log
    # gal_pd['logmstar'] = np.log10(gal_pd['stellar_mass'])

    ## Constants
    Cens     = int(1)
    Sats     = int(0)
    n_gals   = len(gal_pd  )
    n_groups = len(group_pd)
    ## Type of abundance matching
    if param_dict['catl_type'] == 'mr':
        prop_gal    = 'M_r'
        reverse_opt = True
    elif param_dict['catl_type'] == 'mstar':
        prop_gal    = 'logmstar'
        reverse_opt = False
    # Absolute value of `prop_gal`
    prop_gal_abs = prop_gal + '_abs'
    ##
    ## Selecting only a `few` columns
    # Galaxies
    gal_pd = gal_pd.loc[:,[prop_gal, 'groupid']]
    # Groups
    group_pd = group_pd[['ngals']]
    ##
    ## Total `prop_gal` for groups
    group_prop_arr = [[] for x in range(n_groups)]
    ## Looping over galaxy groups
    # Mstar-based
    if param_dict['catl_type'] == 'mstar':
        for group_zz in tqdm(range(n_groups)):
            ## Stellar mass
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal]
            group_log_prop_tot = np.log10(np.sum(10**group_prop))
            ## Saving to array
            group_prop_arr[group_zz] = group_log_prop_tot
    # Luminosity-based
    elif param_dict['catl_type'] == 'mr':
        for group_zz in tqdm(range(n_groups)):
            ## Total abs. magnitude of the group
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal]
            group_prop_tot = Mr_group_calc(group_prop)
            ## Saving to array
            group_prop_arr[group_zz] = group_prop_tot
    ##
    ## Saving to DataFrame
    group_prop_arr            = np.asarray(group_prop_arr)
    group_pd.loc[:, prop_gal] = group_prop_arr
    if param_dict['verbose']:
        print('Calculating group masses...Done')
    ##
    ## --- Halo Abundance Matching --- ##
    ## Mass function for given cosmology
    path_to_hmf = '/Users/asadm2/Desktop/Planck_H0_100.0_HMF_warren.csv'

    hmf_pd = pd.read_csv(path_to_hmf, sep=',')

    ## Halo mass
    Mh_ab = abundance_matching_f(group_prop_arr,
                                    hmf_pd,
                                    volume1=param_dict['survey_vol'],
                                    reverse=reverse_opt,
                                    dict2_names=['logM', 'ngtm'],
                                    dens2_opt=True)
    # Assigning to DataFrame
    group_pd.loc[:, 'M_group'] = Mh_ab
    ###
    ### ---- Galaxies ---- ###
    # Adding `M_group` to galaxy catalogue
    gal_pd = pd.merge(gal_pd, group_pd[['M_group', 'ngals']],
                        how='left', left_on='groupid', right_index=True)
    # Renaming `ngals` column
    gal_pd = gal_pd.rename(columns={'ngals':'g_ngal'})
    #
    # Selecting `central` and `satellite` galaxies
    gal_pd.loc[:, prop_gal_abs] = np.abs(gal_pd[prop_gal])
    gal_pd.loc[:, 'g_galtype']  = np.ones(n_gals).astype(int)*Sats
    g_galtype_groups            = np.ones(n_groups)*Sats
    ##
    ## Looping over galaxy groups
    for zz in tqdm(range(n_groups)):
        gals_g = gal_pd.loc[gal_pd['groupid']==zz]
        ## Determining group galaxy type
        gals_g_max = gals_g.loc[gals_g[prop_gal_abs]==gals_g[prop_gal_abs].max()]
        g_galtype_groups[zz] = int(np.random.choice(gals_g_max.index.values))
    g_galtype_groups = np.asarray(g_galtype_groups).astype(int)
    ## Assigning group galaxy type
    gal_pd.loc[g_galtype_groups, 'g_galtype'] = Cens
    ##
    ## Dropping columns
    # Galaxies
    gal_col_arr = [prop_gal, prop_gal_abs, 'groupid']
    gal_pd      = gal_pd.drop(gal_col_arr, axis=1)
    # Groups
    group_col_arr = ['ngals']
    group_pd      = group_pd.drop(group_col_arr, axis=1)
    ##
    ## Merging to original DataFrames
    # Galaxies
    mockgal_pd_new = pd.merge(mockgal_pd, gal_pd, how='left', left_index=True,
        right_index=True)
    # Groups
    mockgroup_pd_new = pd.merge(mockgroup_pd, group_pd, how='left',
        left_index=True, right_index=True)
    if param_dict['verbose']:
        print('Group Mass Assign. ....Done')

    return mockgal_pd_new, mockgroup_pd_new

def get_mstar_sigma_halomassbins(catl, gals_df, randint_logmstar):
    catl = preprocess.catl
    randint_logmstar=1
    richness_red_ids, N_red, mhalo_red, richness_mstar_red, richness_blue_ids, \
        N_blue, mhalo_blue, richness_mstar_blue = \
        get_richness(gals_df, 'model', randint_logmstar, central_bool=True)

    veldisp_red_keys, sigma_red, veldisp_mstar_red, nsat_red, \
        veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, nsat_blue = \
        get_velocity_dispersion(gals_df, 'model', randint_logmstar)

    if level == 'halo':
        mhalo_red = np.log10(mhalo_red)
        mhalo_blue = np.log10(mhalo_blue)

    richness_results_red = [richness_red_ids, N_red, mhalo_red, 
        richness_mstar_red]
    richness_results_blue = [richness_blue_ids, N_blue, mhalo_blue, 
        richness_mstar_blue]

    veldisp_results_red = [veldisp_red_keys, sigma_red, veldisp_mstar_red, 
        nsat_red]
    veldisp_results_blue = [veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, 
        nsat_blue]

    richness_dict_red = {z[0]: list(z[1:]) for z in zip(*richness_results_red)} 
    veldisp_dict_red = {z[0]: list(z[1:]) for z in zip(*veldisp_results_red)} 

    richness_dict_blue = {z[0]: list(z[1:]) for z in zip(*richness_results_blue)} 
    veldisp_dict_blue = {z[0]: list(z[1:]) for z in zip(*veldisp_results_blue)} 


    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_red_keys:
        # mhalo_arr.append(np.log10(richness_dict_red[key][1])) 
        mhalo_arr.append(richness_dict_red[key][1])
        sigma_arr.append(veldisp_dict_red[key][0])
        mstar_arr.append(veldisp_dict_red[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    mstar_binned_red_arr = []
    sigma_binned_red_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        mstar_binned_red_arr.append(mstar_ii)
        sigma_binned_red_arr.append(sigma_ii)

    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_blue_keys:
        # mhalo_arr.append(np.log10(richness_dict_blue[key][1])) 
        mhalo_arr.append(richness_dict_blue[key][1])
        sigma_arr.append(veldisp_dict_blue[key][0])
        mstar_arr.append(veldisp_dict_blue[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    mstar_binned_blue_arr = []
    sigma_binned_blue_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        mstar_binned_blue_arr.append(mstar_ii)
        sigma_binned_blue_arr.append(sigma_ii)

    richness_red_ids, N_red, mhalo_red, richness_mstar_red, richness_blue_ids, \
        N_blue, mhalo_blue, richness_mstar_blue = \
        get_richness(catl, 'data')

    veldisp_red_keys, sigma_red, veldisp_mstar_red, nsat_red, \
        veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, nsat_blue = \
        get_velocity_dispersion(catl, 'data')

    richness_results_red = [richness_red_ids, N_red, mhalo_red, 
        richness_mstar_red]
    richness_results_blue = [richness_blue_ids, N_blue, mhalo_blue, 
        richness_mstar_blue]

    veldisp_results_red = [veldisp_red_keys, sigma_red, veldisp_mstar_red, 
        nsat_red]
    veldisp_results_blue = [veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, 
        nsat_blue]

    richness_dict_red = {z[0]: list(z[1:]) for z in zip(*richness_results_red)} 
    veldisp_dict_red = {z[0]: list(z[1:]) for z in zip(*veldisp_results_red)} 

    richness_dict_blue = {z[0]: list(z[1:]) for z in zip(*richness_results_blue)} 
    veldisp_dict_blue = {z[0]: list(z[1:]) for z in zip(*veldisp_results_blue)} 

    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_red_keys:
        # mhalo_arr.append(np.log10(richness_dict_red[key][1])) 
        mhalo_arr.append(richness_dict_red[key][1])
        sigma_arr.append(veldisp_dict_red[key][0])
        mstar_arr.append(veldisp_dict_red[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    d_mstar_binned_red_arr = []
    d_sigma_binned_red_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        d_mstar_binned_red_arr.append(mstar_ii)
        d_sigma_binned_red_arr.append(sigma_ii)

    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_blue_keys:
        # mhalo_arr.append(np.log10(richness_dict_blue[key][1])) 
        mhalo_arr.append(richness_dict_blue[key][1])
        sigma_arr.append(veldisp_dict_blue[key][0])
        mstar_arr.append(veldisp_dict_blue[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    d_mstar_binned_blue_arr = []
    d_sigma_binned_blue_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        d_mstar_binned_blue_arr.append(mstar_ii)
        d_sigma_binned_blue_arr.append(sigma_ii)


    fig, axs = plt.subplots(2, 7, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    sp_idx = 0
    for idx in range(len(sigma_binned_red_arr)):
        if idx < 7:
            if len(sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(sigma_binned_red_arr[idx], 
                    mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[0, idx].plot(mean_centers_red, mean_stats_red[0], 
                    c='indianred', zorder=20, lw=6, ls='--')

            axs[0, idx].scatter(sigma_binned_red_arr[idx], 
                mstar_binned_red_arr[idx],
                c='maroon', s=150, edgecolors='k', zorder=5)
            
            axs[0, idx].set_title("Group Halo bin {0} - {1}".format(
                np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[0, idx].set_xlim(0, 250)        
        else:
            if len(sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(sigma_binned_red_arr[idx], 
                    mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[1, sp_idx].plot(mean_centers_red, mean_stats_red[0], 
                    c='indianred', zorder=20, lw=6, ls='--')
            
            axs[1, sp_idx].scatter(sigma_binned_red_arr[idx], 
                mstar_binned_red_arr[idx], 
                c='maroon', s=150, edgecolors='k', zorder=5)
            
            axs[1, sp_idx].set_title(" Group Halo bin {0} - {1}".format(
                np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[1, sp_idx].set_xlim(0, 250)        

            sp_idx += 1

    sp_idx = 0
    for idx in range(len(sigma_binned_blue_arr)):
        if idx < 7:
            if len(sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(sigma_binned_blue_arr[idx], 
                    mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[0, idx].plot(mean_centers_blue, mean_stats_blue[0], 
                    c='cornflowerblue', zorder=20, lw=6, ls='--')

            axs[0, idx].scatter(sigma_binned_blue_arr[idx], 
                mstar_binned_blue_arr[idx],
                c='darkblue', s=150, edgecolors='k', zorder=10)

            axs[0, idx].set_title("Group Halo bin {0} - {1}".format(
                np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[0, idx].set_xlim(0, 250)        
        else:
            if len(sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(sigma_binned_blue_arr[idx], 
                    mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[1, sp_idx].plot(mean_centers_blue, mean_stats_blue[0], 
                    c='cornflowerblue', zorder=20, lw=6, ls='--')

            axs[1, sp_idx].scatter(sigma_binned_blue_arr[idx], 
                mstar_binned_blue_arr[idx], 
                c='darkblue', s=150, edgecolors='k', zorder=10)

            axs[1, sp_idx].set_title("Group Halo bin {0} - {1}".format(
                np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[1, sp_idx].set_xlim(0, 250)        
            sp_idx += 1

    ## Data 
    sp_idx = 0
    for idx in range(len(d_sigma_binned_red_arr)):
        if idx < 7:
            if len(d_sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(d_sigma_binned_red_arr[idx], 
                    d_mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[0, idx].plot(mean_centers_red, mean_stats_red[0], 
                    c='indianred', zorder=20, lw=6, ls='dotted')
                
        else:
            if len(d_sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(d_sigma_binned_red_arr[idx], 
                    d_mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[1, sp_idx].plot(mean_centers_red, mean_stats_red[0], 
                    c='indianred', zorder=20, lw=6, ls='dotted')
            
            sp_idx += 1

    sp_idx = 0
    for idx in range(len(d_sigma_binned_blue_arr)):
        if idx < 7:
            if len(d_sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(d_sigma_binned_blue_arr[idx], 
                    d_mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[0, idx].plot(mean_centers_blue, mean_stats_blue[0], 
                    c='cornflowerblue', zorder=20, lw=6, ls='dotted')

        else:
            if len(d_sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(d_sigma_binned_blue_arr[idx], 
                    d_mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[1, sp_idx].plot(mean_centers_blue, mean_stats_blue[0], 
                    c='cornflowerblue', zorder=20, lw=6, ls='dotted')

            sp_idx += 1

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, 
        left=False, right=False)
    plt.xlabel(r'\boldmath$\sigma_{group} \left[\mathrm{km/s} \right]$', 
        fontsize=30)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\,'
        r'\mathrm{h}^{-1} \right]$',fontsize=30)
    plt.title('{0} quenching'.format(quenching), pad=40.5)
    plt.show()

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
        mstar_total_arr = catl.logmstar.values
        censat_col = 'fc'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values
    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:
        mstar_total_arr = catl.logmstar.values
        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'
        mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values           
    elif randint_logmstar != 1:
        mstar_total_arr = catl['{0}'.format(randint_logmstar)].values
        censat_col = 'grp_censat_{0}'.format(randint_logmstar)
        mstar_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    elif randint_logmstar == 1:
        mstar_total_arr = catl['behroozi_bf'].values
        censat_col = 'grp_censat_{0}'.format(randint_logmstar)
        mstar_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
        mstar_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values


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

    if level == 'group':
        f_blue_cen = result_cen[0]
        f_blue_sat = result_sat[0]
    elif level == 'halo':
        f_blue_cen = 0
        f_blue_sat = 0

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

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

def get_best_fit_model(bf_params, best_fit_mocknum=None):

    best_fit_results = {}
    best_fit_experimentals = {}

    if best_fit_mocknum:
        cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', \
            '{0}'.format(best_fit_mocknum), \
            'grp_censat_{0}'.format(best_fit_mocknum), \
            'groupid_{0}'.format(best_fit_mocknum)]

        gals_df = globals.gal_group_df_subset[cols_to_use]

        gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
            format(best_fit_mocknum),'groupid_{0}'.format(best_fit_mocknum)]).\
            reset_index(drop=True)

        gals_df[['grp_censat_{0}'.format(best_fit_mocknum), \
            'groupid_{0}'.format(best_fit_mocknum)]] = \
            gals_df[['grp_censat_{0}'.format(best_fit_mocknum),\
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
        'grp_censat_{0}'.format(randint_logmstar), \
        'groupid_{0}'.format(randint_logmstar),\
        'cen_cz_{0}'.format(randint_logmstar)]

        gals_df = gal_group_df_subset[cols_to_use]

        gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
        format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
        'cen_cz_{0}'.format(randint_logmstar)]).\
        reset_index(drop=True)

        gals_df[['grp_censat_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar)]] = \
            gals_df[['grp_censat_{0}'.format(randint_logmstar),\
            'groupid_{0}'.format(randint_logmstar)]].astype(int)
                    
        gals_df['behroozi_bf'] = np.log10(gals_df['behroozi_bf'])

    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(bf_params[5:], gals_df, 
            'vishnu', randint_logmstar)
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(bf_params[5:], gals_df, 
            'vishnu')      
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
    # v_sim = 130**3 
    v_sim = 890641.5172927063  ## cz: 3000-12000
    # v_sim = 165457.21308906242 ## cz: 3000-7000


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

    red_sigma, red_cen_mstar_sigma, blue_sigma, \
        blue_cen_mstar_sigma, red_nsat, blue_nsat, red_host_halo_mass_vd, \
        blue_host_halo_mass_vd = \
        get_velocity_dispersion(gals_df, 'model', randint_logmstar)

    red_num, red_cen_mstar_richness, blue_num, \
        blue_cen_mstar_richness, red_host_halo_mass, \
        blue_host_halo_mass = \
        get_richness(gals_df, 'model', randint_logmstar)
    
    best_fit_results["f_blue"] = {'max_fblue':f_blue[0],
                                    'fblue_total':f_blue[1],
                                    'fblue_cen':f_blue[2],
                                    'fblue_sat':f_blue[3]}
    
    best_fit_results["phi_colour"] = {'phi_red':phi_red_model,
                                        'phi_blue':phi_blue_model}

    best_fit_results["centrals"] = {'gals':cen_gals, 'halos':cen_halos,
                                    'gals_red':cen_gals_red, 
                                    'halos_red':cen_halos_red,
                                    'gals_blue':cen_gals_blue,
                                    'halos_blue':cen_halos_blue}
    
    best_fit_results["satellites"] = {'gals_red':sat_gals_red, 
                                    'halos_red':sat_halos_red,
                                    'gals_blue':sat_gals_blue,
                                    'halos_blue':sat_halos_blue}
    
    best_fit_results["f_red"] = {'cen_red':f_red_cen_red,
                                'cen_blue':f_red_cen_blue,
                                'sat_red':f_red_sat_red,
                                'sat_blue':f_red_sat_blue}
    
    best_fit_experimentals["vel_disp"] = {'red_sigma':red_sigma,
        'red_cen_mstar':red_cen_mstar_sigma,
        'blue_sigma':blue_sigma, 'blue_cen_mstar':blue_cen_mstar_sigma,
        'red_nsat':red_nsat, 'blue_nsat':blue_nsat, 
        'red_hosthalo':red_host_halo_mass_vd, 
        'blue_hosthalo':blue_host_halo_mass_vd}
    
    best_fit_experimentals["richness"] = {'red_num':red_num,
        'red_cen_mstar':red_cen_mstar_richness, 'blue_num':blue_num,
        'blue_cen_mstar':blue_cen_mstar_richness, 
        'red_hosthalo':red_host_halo_mass, 'blue_hosthalo':blue_host_halo_mass}

    return best_fit_results, best_fit_experimentals, gals_df

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
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
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
                (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
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

            ## Using best-fit found for new ECO data using result from chain 34
            ## i.e. hybrid quenching model
            Mstar_q = 10.54 # Msun/h
            Mh_q = 14.09 # Msun/h
            mu = 0.77
            nu = 0.17

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 35
            ## i.e. halo quenching model
            Mh_qc = 12.29 # Msun/h
            Mh_qs = 12.78 # Msun/h
            mu_c = 1.37
            mu_s = 1.48

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
    
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma, red_nsat, blue_nsat, red_host_halo_mass, \
                blue_host_halo_mass = get_velocity_dispersion(mock_pd, 'mock')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic='mean', bins=np.linspace(-2,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic='mean', bins=np.linspace(-1,3,5))

            mean_mstar_red_arr.append(mean_mstar_red[0])
            mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)
    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    # print('Measuring velocity dispersion for data')
    # red_sigma, red_cen_mstar_sigma, blue_sigma, \
    # blue_cen_mstar_sigma = \
    # get_velocity_dispersion(catl, 'data')

    # red_sigma = np.log10(red_sigma)
    # blue_sigma = np.log10(blue_sigma)

    # mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    #     statistic='mean', bins=np.linspace(-2,3,5))
    # mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
    #     statistic='mean', bins=np.linspace(-1,3,5))


    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]
    # phi_total_4 = phi_arr_total[:,4]
    # phi_total_5 = phi_arr_total[:,5]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]
    # f_blue_cen_4 = f_blue_cen_arr[:,4]
    # f_blue_cen_5 = f_blue_cen_arr[:,5]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]
    # f_blue_sat_4 = f_blue_sat_arr[:,4]
    # f_blue_sat_5 = f_blue_sat_arr[:,5]

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

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=15)
    # rc('text', usetex=True)
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral')
    # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]

    # plt.xticks(tick_marks, names, rotation='vertical')
    # plt.yticks(tick_marks, names)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()
    
    # ## Velocity dispersion
    # bins_red=np.linspace(-2,3,5)
    # bins_blue=np.linspace(-1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # fig2 = plt.figure()
    # for idx in range(len(combined_df.values[:,12:16])):
    #     plt.plot(bins_red, combined_df.values[:,12:16][idx])
    # for idx in range(len(combined_df.values[:,16:20])):
    #     plt.plot(bins_blue, combined_df.values[:,16:20][idx])
    # plt.plot(bins_red, mean_mstar_red_data[0], '--r', lw=3, label='data')
    # plt.plot(bins_blue, mean_mstar_blue_data[0], '--b', lw=3, label='data')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=30)
    # plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()

    # ## Blue fraction from mocks
    # fig3 = plt.figure(figsize=(10,10))
    # for idx in range(len(combined_df.values[:,4:8])):
    #     plt.plot(f_blue[0], combined_df.values[:,4:8][idx], '--')
    # plt.plot(f_blue_data[0], f_blue_data[2], 'k--', lw=3, label='cen data')
    # for idx in range(len(combined_df.values[:,8:12])):
    #     plt.plot(f_blue[0], combined_df.values[:,8:12][idx], '-')
    # plt.plot(f_blue_data[0], f_blue_data[3], 'k-', lw=3, label='sat data')

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=25)
    # plt.title(r'Blue fractions from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()

    # ## SMF from mocks and data
    # fig4 = plt.figure()
    # for idx in range(len(combined_df.values[:,:4])):
    #     plt.plot(max_total, combined_df.values[:,:4][idx], '-')
    # plt.plot(total_data[0], total_data[1], 'k--', lw=3, label='data')

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$')
    # plt.title(r'SMFs from mocks and data')
    # plt.legend(loc='best', prop={'size':25})
    # plt.show()


    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral')
    # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
    # plt.colorbar(cax)
    # plt.gca().invert_yaxis() 
    # # # put a blue dot at (10, 20)
    # # plt.scatter([10], [20])

    # # # put a red dot, size 40, at 2 locations:
    # # plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)
    # plt.show()

    # fig1, axes = plt.subplots(12,12, sharex=True, sharey=True, figsize=(10,10))
    # for i, ax_i in enumerate(axes.flatten()):
    #     if i == 12:
    #         break
    #     for j, ax_j in enumerate(axes.flatten()):
    #         if j == 12:
    #             break
    #         elif i == j:
    #             axes[i,j].hist(combined_df[combined_df.columns.values[i]], 
    #                 density=True, color='k')
    #         # else:
    #         #     axes[i,j].scatter(combined_df[combined_df.columns.values[i]], 
    #         #         combined_df[combined_df.columns.values[j]], c='k')
    #         axes[i,j].set_xticks([])
    #         axes[i,j].set_yticks([])
    #         axes[i,j].set_aspect('equal')

    # fig1.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    # fig1, ax_main = plt.subplots(1,1)
    # axes = pd.plotting.scatter_matrix(combined_df, alpha=0.8, diagonal='kde', 
    #     c='k', range_padding=0.1)

    # for i, ax in enumerate(axes.flatten()):
    #     c = plt.cm.Spectral(combined_df.corr().values.flatten()[i])
    #     ax.set_facecolor(c)
    # # ax_main.invert_yaxis()
    # plt.title(r'Total mass function and blue fraction')
    # plt.show()


    # rc('text', usetex=True)
    # rc('text.latex', preamble=r"\usepackage{amsmath}")

    # ## Total SMFs from mocks and data for paper
    # tot_phi_max = np.amax(combined_df.values[:,:6], axis=0)
    # tot_phi_min = np.amin(combined_df.values[:,:6], axis=0)
    # error = np.nanstd(combined_df.values[:,:6], axis=0)

    # fig2 = plt.figure()
    # mt = plt.fill_between(x=max_total, y1=tot_phi_max, 
    #     y2=tot_phi_min, color='silver', alpha=0.4)
    # dt = plt.errorbar(total_data[0], total_data[1], yerr=error,
    #     color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    #     capthick=1.5, zorder=10, marker='^')
    # plt.ylim(-4,-1)

    # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$')

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='best')
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # plt.savefig('/Users/asadm2/Desktop/total_smf.svg', format='svg', 
    #     bbox_inches="tight", dpi=1200)
    # plt.show()


    return err_colour, corr_mat_inv_colour

global survey
global quenching
global level 
global mf_type
global gal_group_df_subset

survey = 'eco'
level = 'group'
quenching = 'halo'
mf_type = 'smf'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"

if quenching == 'halo':
    run = 35
elif quenching == 'hybrid':
    run = 34

chi2_file = path_to_proc + \
    'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
chain_file = path_to_proc + \
    'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)

chi2 = pd.read_csv(chi2_file, header=None, names=['chisquared'])\
    ['chisquared'].values

mcmc_table = read_mcmc(chain_file)

gal_group = reading_catls(path_to_proc + \
    "gal_group_run{0}.hdf5".format(run)) 

# #! Change this if testing with different cz limit
# gal_group = self.read_mock_catl(settings.path_to_proc + \
#     "gal_group_run{0}.hdf5".format(settings.run)) 

idx_arr = np.insert(np.linspace(0,20,21), len(np.linspace(0,20,21)), \
    (22, 123, 124, 125, 126, 127, 128, 129)).\
    astype(int)

names_arr = [x for x in gal_group.columns.values[idx_arr]]
for idx in np.arange(2,102,1):
    names_arr.append('{0}_y'.format(idx))
    names_arr.append('groupid_{0}'.format(idx))
    names_arr.append('grp_censat_{0}'.format(idx))
    names_arr.append('cen_cz_{0}'.format(idx))
names_arr = np.array(names_arr)

gal_group_df_subset = gal_group[names_arr]

# Renaming the "1_y" column kept from line 1896 because of case where it was
# also in mcmc_table_ptcl.mock_num and was selected twice
gal_group_df_subset.columns.values[25] = "behroozi_bf"

### Removing "_y" from column names for stellar mass
# Have to remove the first element because it is 'halo_y' column name
cols_with_y = np.array([[idx, s] for idx, s in enumerate(
    gal_group_df_subset.columns.values) if '_y' in s][1:])
colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
gal_group_df_subset.columns.values[cols_with_y[:,0].\
    astype(int)] = colnames_without_y


print('Reading catalog') #No Mstar cut needed as catl_file already has it
catl, volume, z_median = read_data_catl(catl_file, survey)

print('Assigning colour to data using u-r colours')
catl = assign_colour_label_data(catl)

print('Getting data in specific percentile')
# get_paramvals called to get bf params and chi2 values
mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)


red_keys, red_sigma, red_cen_mstar_sigma, red_nsat, blue_keys, blue_sigma, \
    blue_cen_mstar_sigma, blue_nsat= \
    get_velocity_dispersion(catl, 'data')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_stats_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    statistic='mean', bins=np.linspace(-2,3,5))
mean_stats_blue_data = bs(blue_sigma, blue_cen_mstar_sigma,
    statistic='mean', bins=np.linspace(-1,3,5))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

red_keys, red_sigma, red_cen_mstar_sigma, red_nsat, blue_keys, blue_sigma, \
    blue_cen_mstar_sigma, blue_nsat= \
    get_velocity_dispersion(gals_df, 'model', 1)

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_stats_red_bf = bs(red_sigma, red_cen_mstar_sigma, 
    statistic='mean', bins=np.linspace(-2,3,5))
mean_stats_blue_bf = bs(blue_sigma, blue_cen_mstar_sigma,
    statistic='mean', bins=np.linspace(-1,3,5))

sigma, corr_mat_inv = get_err_data(survey, path_to_mocks)

fig1 = plt.subplots(figsize=(10,8))

error_red = sigma[12:16]
error_blue = sigma[16:]

dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=error_red,
        color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
        capthick=1.0,zorder=10)
db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=error_blue,
        color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
        capthick=1.0,zorder=10)


dr, = plt.plot(mean_centers_red, mean_stats_red_data[0], lw=3, c='darkred', ls='-')
db, = plt.plot(mean_centers_blue, mean_stats_blue_data[0], lw=3, c='darkblue', ls='-')

bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], lw=3, c='indianred', ls='--')
bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], lw=3, c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.ylim(8.9,)

plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

prop_gal = 'behroozi_bf'
reverse_opt = False
gal_pd = gals_df.loc[:,[prop_gal, 'groupid_1']]
# gal_pd.behroozi_bf = np.log10((10**gal_pd.behroozi_bf.values) * 2.041)

groups = gals_df.groupby('groupid_1')
keys = groups.groups.keys()
ngals_arr = []
for key in keys:
    group_n = len(groups.get_group(key))
    ngals_arr.append(group_n)
ngals_arr = np.asarray(ngals_arr)

group_pd = pd.DataFrame(data=ngals_arr, columns=['ngals'])

# gal_pd = gal_pd.sort_values('groupid_1').reset_index(drop=True)
n_groups = len(np.unique(gal_pd.groupid_1))
group_prop_arr = [[] for x in range(n_groups)]


for group_zz in tqdm(range(n_groups)):
    ## Stellar mass
    group_prop = gal_pd.loc[gal_pd['groupid_1']==group_zz, prop_gal]
    group_log_prop_tot = np.log10(np.sum(10**group_prop))
    ## Saving to array
    group_prop_arr[group_zz] = group_log_prop_tot

group_prop_arr = np.asarray(group_prop_arr)

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

H0 = 100 # (km/s)/Mpc
cz_inner = 3000 # not starting at corner of box
cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

v_inner = vol_sphere(dist_inner)
v_outer = vol_sphere(dist_outer)

v_sphere = v_outer-v_inner
survey_vol = v_sphere/8

eco = {
    'c': 3*10**5,
    'survey_vol': survey_vol,
    'min_cz' : cz_inner,
    'max_cz' : cz_outer,
    'zmin': cz_inner/(3*10**5),
    'zmax': cz_outer/(3*10**5),
    'l_perp': 0.07,
    'l_para': 1.1,
    'nmin': 1,
    'verbose': True,
    'catl_type': 'mstar'
}

# Changes string name of survey to variable so that the survey dict can
# be accessed
param_dict = vars()[survey]

path_to_hmf = '/Users/asadm2/Desktop/Planck_H0_100.0_HMF_warren.csv'

hmf_pd = pd.read_csv(path_to_hmf, sep=',')

## Halo mass
Mh_ab = abundance_matching_f(group_prop_arr,
                                hmf_pd,
                                volume1=param_dict['survey_vol'],
                                reverse=reverse_opt,
                                dict2_names=['logM', 'ngtm'],
                                dens2_opt=True)
group_pd.loc[:, 'M_group'] = Mh_ab

gal_pd = pd.merge(gal_pd, group_pd[['M_group', 'ngals']],
                    how='left', left_on='groupid_1', right_index=True)

# gals_df = gals_df.sort_values('groupid_1')
gals_df['M_group'] = gal_pd['M_group']
# gals_df = gals_df.reset_index(drop=True)

### Group halo mass plot of data and best-fit
randint = 1
logmstar_col = 'behroozi_bf'
galtype_col = 'grp_censat_{0}'.format(randint)
cencz_col = 'cen_cz_{0}'.format(randint)
id_col = 'groupid_{0}'.format(randint)


min_cz = 3000
max_cz = 7000
mstar_limit = 8.9


gals_df_subset = gals_df.loc[
    (gals_df[cencz_col].values >= min_cz) & \
    (gals_df[cencz_col].values <= max_cz) & \
    (gals_df[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

# group_mh_model = gals_df_subset.M_group.values
# group_mh_data = catl.M_group.values

group_mh_model = gals_df_subset.M_group.loc[gals_df_subset.grp_censat_1 == 1]
group_mh_data = catl.M_group.loc[catl.g_galtype == 1]

bin_min = 10.9
bin_max = 15
bin_num = 6
bins = np.linspace(bin_min, bin_max, bin_num)

volume = preprocess.volume ## 151829.26

counts, edg = np.histogram(group_mh_data, bins=bins)  # paper used 17 bins
dm = edg[1] - edg[0]  # Bin width
maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
# Normalized to volume and bin width
err_poiss = np.sqrt(counts) / (volume * dm)
err_tot_data = err_poiss
phi = counts / (volume * dm)  # not a log quantity

phi_data = np.log10(phi)

# volume = 890641.5172927063


counts, edg = np.histogram(group_mh_model, bins=bins)  # paper used 17 bins
dm = edg[1] - edg[0]  # Bin width
maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
# Normalized to volume and bin width
err_poiss = np.sqrt(counts) / (volume * dm)
err_tot = err_poiss
phi = counts / (volume * dm)  # not a log quantity

phi_model = np.log10(phi)

plt.errorbar(maxis, phi_data, yerr=err_tot_data,
    color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^', label='Data')
plt.plot(maxis, phi_model, color='mediumorchid', label='Best-fit', lw=3)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
plt.title('Group halo mass function')
plt.legend(loc='best',prop={'size':30})
plt.show()

################################################################################
### Group stellar mass plot of eco data right after re-running group finder

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']

def cumu_num_dens(data,nbins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    bin_min = 8.9
    bin_max = 13
    bins = np.linspace(bin_min, bin_max, nbins)
    freq,edg = np.histogram(data,bins=bins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not bool_mag:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,n_cumu,err_poiss

# Post group-finding catalog
catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
catl_postgrpfind_used = reading_catls(catl_file)
# catl, volume, z_median = read_data_catl(catl_file, survey)

catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1.hdf5"
# catl, volume, z_median = read_data_catl(catl_file, survey)
catl_postgrpfind_new = reading_catls(catl_file)

groups = catl_postgrpfind_used.groupby('groupid')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_postgrpfinding_used = np.asarray(mstar_group_data_arr)

groups = catl_postgrpfind_new.groupby('groupid')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_postgrpfinding_new = np.asarray(mstar_group_data_arr)

# Original ECO catalog
catl_file = path_to_raw + "eco/eco_all.csv"
eco_buff = pd.read_csv(catl_file,delimiter=",", header=0) 
groups = eco_buff.groupby('grp')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_or = np.asarray(mstar_group_data_arr)


# Latest public DR of ECO
ecodr2 = pd.read_csv("/Users/asadm2/Desktop/ecodr2.csv", delimiter=",", header=0)
ecodr2 = ecodr2.loc[ecodr2.name != 'ECO13860']
groups = ecodr2.groupby('grp_e17')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_dr2 = np.asarray(mstar_group_data_arr)

# ECO catalog used in analysis
## grpcz cuts 3000-7000 where group cz is cz of central
## -17.33 cut
catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
catl, volume_analysis, z_median = read_data_catl(catl_file, survey)
groups = catl.groupby('groupid')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_analysis = np.asarray(mstar_group_data_arr)

## Best-fit model
volume_bf = 890641.5172927063 ## cz: 3000-12000
groups = gals_df_subset.groupby('groupid_1')
keys = groups.groups.keys()
mstar_group_bf_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum((10**group.behroozi_bf.values)*2.041))
    mstar_group_bf_arr.append(group_log_prop_tot)


bin_min = 8.9
bin_max = 13
bin_num = 6
bins = np.linspace(bin_min, bin_max, bin_num)

volume_buffer_H70 = 192351.36 * 2.915 # survey volume with buffer in h=0.7

counts, edg = np.histogram(mstar_group_data_or, bins=bins)  # paper used 17 bins
dm = edg[1] - edg[0]  # Bin width
maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
# Normalized to volume and bin width
err_poiss = np.sqrt(counts) / (volume * dm)
err_tot_data = err_poiss
phi = counts / (volume * dm)  # not a log quantity

phi_data = np.log10(phi)


x_or,y_or,err_or = cumu_num_dens(mstar_group_data_or, 6, None, volume_buffer_H70, False)
x_dr2,y_dr2,err_dr2 = cumu_num_dens(mstar_group_data_dr2, 6, None, volume_buffer_H70, False)
x_grpfind_new,y_grpfind_new,err_grpfind_new = cumu_num_dens(mstar_group_data_postgrpfinding_new, 
    6, None, volume_buffer_H70/2.915, False)
x_grpfind_used,y_grpfind_used,err_grpfind_used = cumu_num_dens(mstar_group_data_postgrpfinding_used, 
    6, None, volume_buffer_H70, False)
x_an,y_an,err_an = cumu_num_dens(mstar_group_data_analysis, 6, None, volume_analysis * 2.915, False)
x_bf,y_bf,err_bf = cumu_num_dens(mstar_group_bf_arr, 6, None, volume_analysis*2.915, False)

# plt.errorbar(maxis, phi_data, yerr=err_tot_data,
#     color='k', fmt='s-', ecolor='k', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Differential')
# plt.errorbar(x_or, np.log10(y_or), yerr=err_or,
#     color='k', fmt='s--', ecolor='k', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Cumulative')
plt.scatter(x_or, np.log10(y_or), color='mediumorchid', s=200, marker='^', label='Original catalog')
plt.scatter(x_dr2, np.log10(y_dr2), color='palevioletred', s=200, marker='^', label='DR2')
plt.scatter(x_grpfind_used, np.log10(y_grpfind_used), color='teal', s=200, marker='^', label='Post-group finding (used)')
plt.scatter(x_grpfind_new, np.log10(y_grpfind_new), color='k', s=200, marker='^', label='Post-group finding (new)')
plt.scatter(x_an, np.log10(y_an), color='gold', s=200, marker='^', label='Used in analysis')
plt.scatter(x_bf, np.log10(y_bf), color='cornflowerblue', s=200, marker='^', label='Best-fit hybrid')

# plt.plot(maxis, phi_model, color='mediumorchid', label='Best-fit', lw=3)
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$(n > M) \left[\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
# plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
plt.title('Group stellar mass function')
plt.legend(loc='best',prop={'size':30})
plt.show()
################################################################################
### Group halo mass plot of eco data right after re-running group finder
def cumu_num_dens(data,nbins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    bin_min = 10.9
    bin_max = 15
    bins = np.linspace(bin_min, bin_max, nbins)
    freq,edg = np.histogram(data,bins=bins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not bool_mag:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,n_cumu,err_poiss

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']


# Post group-finding catalog
catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
catl = reading_catls(catl_file)
group_mh_data_postgrpfinding_used = catl.M_group.loc[catl.g_galtype == 1]
volume_buffer_H70 = 192351.36 * 2.915
volume_buffer_H100 = 192351.36

catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1.hdf5"
catl = reading_catls(catl_file)
group_mh_data_postgrpfinding_new = catl.M_group.loc[catl.g_galtype == 1]


# Post group-finding catalog but no mstar cut applied so technically the same 
# as the purple points (original ECO catalog)
catl_file = "/Users/asadm2/Desktop/gal_group_eco_data_buffer_nomstarcut.hdf5"
catl_postgrpfind_nomstarcut = reading_catls(catl_file)
group_mh_data_postgrpfinding_nomstarcut = catl_postgrpfind_nomstarcut.M_group.loc\
    [catl_postgrpfind_nomstarcut.g_galtype == 1]

# Original ECO catalog
catl_file = path_to_raw + "eco/eco_all.csv"
eco_buff = pd.read_csv(catl_file,delimiter=",", header=0) 
# group_mh_data_or = eco_buff.logmh.loc[eco_buff.logmh.values > 0]
# group_mh_data_or = eco_buff.logmh_s.loc[eco_buff.logmh_s.values > 0]
group_mh_data_or = eco_buff.logmh_s.loc[eco_buff.fc == 1]

# Latest public DR of ECO
ecodr2 = pd.read_csv("/Users/asadm2/Desktop/ecodr2.csv", delimiter=",", header=0)
ecodr2 = ecodr2.loc[ecodr2.name != 'ECO13860']
# group_mh_data_dr2 = ecodr2.logmh_s_e17.loc[ecodr2.logmh_s_e17.values > 0]
group_mh_data_dr2 = ecodr2.logmh_s_e17.loc[ecodr2.fc_e17 == 1]

# ECO catalog used in analysis
## grpcz cuts 3000-7000 where group cz is cz of central
## -17.33 cut
volume_nobuffer_H100 = 151829.26
volume_nobuffer_H70 = 151829.26 * 2.915
catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
catl, volume_analysis, z_median = read_data_catl(catl_file, survey)
group_mh_data_analysis = catl.M_group.loc[catl.g_galtype == 1]

volume_bf = 890641.5172927063 ## cz: 3000-12000
group_mh_bestfit = gals_df_subset.M_group.loc[gals_df_subset.grp_censat_1 == 1]
groups = gals_df_subset.groupby('groupid_1')
keys = groups.groups.keys()
keys_arr = []
for key in keys:
    group = groups.get_group(key)
    group_cen = group.loc[group.grp_censat_1 == 1]
    if len(group_cen) > 1:
        keys_arr.append(key)


### HMF
import astropy.cosmology as astrocosmo
import hmf
H0=100
cosmo_model = astrocosmo.Planck15.clone(H0=H0)
hmf_choice_fit_eco_default = hmf.fitting_functions.Warren
mass_func_eco_default = hmf.mass_function.hmf.MassFunction(Mmin=10, Mmax=15, 
    cosmo_model=cosmo_model, hmf_model=hmf_choice_fit_eco_default)

warren = pd.read_csv("/Users/asadm2/Desktop/Planck_H0_100.0_HMF_warren.csv", delimiter=",", header=0)

## Trying cumulative function
x_or,y_or,err_or = cumu_num_dens(group_mh_data_or, 30, None, volume_buffer_H100, False)
x_dr2,y_dr2,err_dr2 = cumu_num_dens(group_mh_data_dr2, 30, None, volume_buffer_H100, False)
x_grpfind_used,y_grpfind_used,err_grpfind_used = cumu_num_dens(group_mh_data_postgrpfinding_used, 
    30, None, volume_buffer_H70, False)
x_grpfind_new,y_grpfind_new,err_grpfind_new = cumu_num_dens(group_mh_data_postgrpfinding_new, 
    30, None, volume_buffer_H100, False)
x_an,y_an,err_an = cumu_num_dens(group_mh_data_analysis, 30, None, volume_nobuffer_H70, False)
x_bf,y_bf,err_bf = cumu_num_dens(group_mh_bestfit, 30, None, volume_nobuffer_H100, False)
x_grpfind_nomstarcut,y_grpfind_nomstarcut,err_grpfind_nomstarcut = \
    cumu_num_dens(group_mh_data_postgrpfinding_nomstarcut, 30, None, volume_buffer_H70, False)

# plt.errorbar(x_or, np.log10(y_or), yerr=err_or,
#     color='mediumorchid', fmt='s', ecolor='mediumorchid', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Original catalog')
# plt.errorbar(x_dr2, np.log10(y_dr2), yerr=err_dr2,
#     color='palevioletred', fmt='s', ecolor='palevioletred', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='DR2')
# plt.errorbar(x_grpfind, np.log10(y_grpfind), yerr=err_grpfind,
#     color='teal', fmt='s', ecolor='teal', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Post-group finding')
# plt.errorbar(x_an, np.log10(y_an), yerr=err_an,
#     color='gold', fmt='s', ecolor='gold', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Used in analysis')
plt.scatter(x_or, np.log10(y_or), color='mediumorchid', s=200, marker='^', label='Original catalog')
plt.scatter(x_dr2, np.log10(y_dr2), color='palevioletred', s=200, marker='^', label='DR2')
plt.scatter(x_grpfind_used, np.log10(y_grpfind_used), color='teal', s=200, marker='^', label='Post-group finding (used)')
plt.scatter(x_grpfind_new, np.log10(y_grpfind_new), color='teal', s=200, marker='*', label='Post-group finding (new)')
plt.scatter(x_an, np.log10(y_an), color='gold', s=200, marker='^', label='Used in analysis')
plt.scatter(x_bf, np.log10(y_bf), color='k', s=200, marker='^', label='Best-fit hybrid')
plt.scatter(x_grpfind_nomstarcut, np.log10(y_grpfind_nomstarcut), color='cornflowerblue', 
    s=200, marker='*', label='Post-group finding (no mstar cut)')

# plt.plot(np.log10(mass_func_eco_default.m), 
#     np.log10(mass_func_eco_default.ngtm), lw=5, 
#     color='cornflowerblue', label='Warren h=0.7', solid_capstyle='round')

plt.plot(warren.logM.values, np.log10(warren.ngtm.values), color='k', label='Warren h=1.0', lw=5)
# plt.xlim(11,15)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$(n > M) \left[\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
# plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
plt.title('Group halo mass function from various ECO catalogs')
plt.legend(loc='best',prop={'size':30})
plt.show()


def diff_func(group_mh_data, volume):
    ## Trying differential function
    bin_min = 10.5
    bin_max = 15
    bin_num = 6
    bins = np.linspace(bin_min, bin_max, bin_num)
    counts, edg = np.histogram(group_mh_data, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot_data = np.log10(err_poiss)
    phi = counts / (volume * dm)  # not a log quantity

    phi_data = np.log10(phi)

    return maxis, phi_data, err_tot_data

## Trying cumulative function
x_or,y_or,err_or = diff_func(group_mh_data_or, volume)
x_dr2,y_dr2,err_dr2 = diff_func(group_mh_data_dr2, volume)
x_grpfind,y_grpfind,err_grpfind = diff_func(group_mh_data_postgrpfinding, 
    volume*2.915)
x_an,y_an,err_an = diff_func(group_mh_data_analysis, volume*2.915)

plt.errorbar(x_or, y_or, yerr=err_or,
    color='mediumorchid', fmt='s', ecolor='mediumorchid', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^', label='Original catalog')
plt.errorbar(x_dr2, y_dr2, yerr=err_dr2,
    color='palevioletred', fmt='s', ecolor='palevioletred', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^', label='DR2')
plt.errorbar(x_grpfind, np.log10(y_grpfind), yerr=err_grpfind,
    color='teal', fmt='s', ecolor='teal', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^', label='Post-group finding')
plt.errorbar(x_an, np.log10(y_an), yerr=err_an,
    color='gold', fmt='s', ecolor='gold', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^', label='Used in analysis')
plt.plot(warren.logM.values, np.log10(warren.ngtm.values), color='k', label='Warren+06', lw=3)

# plt.errorbar(maxis, phi_data, yerr=err_tot_data,
#     color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
#     capthick=1.5, zorder=10, marker='^', label='Data')
# plt.plot(warren.logM.values, np.log10(warren.ngtm.values), color='mediumorchid', label='Warren+06', lw=3)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
plt.title('Group halo mass function')
plt.legend(loc='best',prop={'size':30})
plt.show()

################################################################################
# Plot of satellite fractions in best-fit and data

def sat_frac_helper(arr):
    total_num = len(arr)
    sat_counter = list(arr).count(0)
    # print("Poisson error: {0}".format(np.sqrt(sat_counter)/total_num))
    return sat_counter/total_num


bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5

bins = np.linspace(bin_min, bin_max, bin_num)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

catl['logmstar_h100'] = np.log10((10**catl.logmstar.values)/2.041)
catl_subset = catl.loc[catl.logmstar_h100 > np.round(np.log10((10**8.9) / 2.041), 1)]


logmstar_red_data = catl_subset.logmstar.loc[catl_subset.colour_label == 'R']
logmstar_blue_data = catl_subset.logmstar.loc[catl_subset.colour_label == 'B']

grp_cen_sat_red_data = catl_subset.g_galtype.loc[catl_subset.colour_label == 'R']
grp_cen_sat_blue_data = catl_subset.g_galtype.loc[catl_subset.colour_label == 'B']

result_red_data= bs(logmstar_red_data, grp_cen_sat_red_data, sat_frac_helper, bins=bins)
result_blue_data = bs(logmstar_blue_data, grp_cen_sat_blue_data, sat_frac_helper, bins=bins)

logmstar_red_bf = gals_df_subset.behroozi_bf.loc[gals_df_subset.colour_label == 'R']
logmstar_blue_bf = gals_df_subset.behroozi_bf.loc[gals_df_subset.colour_label == 'B']

grp_cen_sat_red_bf = gals_df_subset.grp_censat_1.loc[gals_df_subset.colour_label == 'R']
grp_cen_sat_blue_bf = gals_df_subset.grp_censat_1.loc[gals_df_subset.colour_label == 'B']

result_red_bf = bs(logmstar_red_bf, grp_cen_sat_red_bf, sat_frac_helper, bins=bins)
result_blue_bf = bs(logmstar_blue_bf, grp_cen_sat_blue_bf, sat_frac_helper, bins=bins)


plt.scatter(bin_centers, result_red_data[0], color='firebrick', s=200, marker='^', label='Early types (data)')
plt.scatter(bin_centers, result_blue_data[0], color='cornflowerblue', s=200, marker='^', label='Late types (data)')
plt.scatter(bin_centers, result_red_bf[0], color='firebrick', s=200, marker='o', label='Early types (best-fit halo)')
plt.scatter(bin_centers, result_blue_bf[0], color='cornflowerblue', s=200, marker='o', label='Late types (best-fit halo)')
plt.xlabel(r'\boldmath$\log_{10}\ M_{\star,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$f_{sat}$', fontsize=30)
plt.title('Group satellite fraction in stellar mass bins')
plt.legend(loc='best', prop={'size':20})
plt.show()

################################################################################
catl_file = path_to_proc + "gal_group_eco_data_buffer.hdf5"
catl, volume_analysis, z_median = read_data_catl(catl_file, survey)
catl = assign_colour_label_data(catl)

veldisp_red_keys, sigma_red_data, veldisp_mstar_red, nsat_red, \
    veldisp_blue_keys, sigma_blue_data, veldisp_mstar_blue, nsat_blue = \
    get_velocity_dispersion(catl, 'data')

richness_red_ids, N_red, mhalo_red_data, richness_mstar_red, richness_blue_ids, \
    N_blue, mhalo_blue_data, richness_mstar_blue = \
    get_richness(catl, 'data', central_bool=True)

N_red_cen = len(veldisp_red_keys)
N_blue_cen = len(veldisp_blue_keys)
bins = np.linspace(0, 250, 7)
sigma_all = sigma_red + sigma_blue
colour_label_all = ["R"]*N_red_cen + ["B"]*N_blue_cen
result_data = bs(sigma_all, colour_label_all, blue_frac_helper, bins=bins)

maxis, [phi_red_d, phi_blue_d], [err_poiss_red_d, err_poiss_blue_d], bins, \
    [counts_red_d, counts_blue_d] = get_vdf(sigma_red, sigma_blue, volume_analysis)

randint_logmstar = 1
veldisp_red_keys, sigma_red, veldisp_mstar_red, nsat_red, \
    veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, nsat_blue = \
    get_velocity_dispersion(gals_df, 'model', randint_logmstar)

richness_red_ids, N_red, mhalo_red, richness_mstar_red, richness_blue_ids, \
    N_blue, mhalo_blue, richness_mstar_blue = \
    get_richness(gals_df, 'model', randint_logmstar, central_bool=True)

N_red_cen = len(veldisp_red_keys)
N_blue_cen = len(veldisp_blue_keys)
bins = np.linspace(0, 250, 7)
sigma_all = sigma_red + sigma_blue
colour_label_all = ["R"]*N_red_cen + ["B"]*N_blue_cen
result_bf = bs(sigma_all, colour_label_all, blue_frac_helper, bins=bins)

## Volume analysis used since measurements are made on data where cz: 3000-7000
maxis, [phi_red_bf, phi_blue_bf], [err_poiss_red_bf, err_poiss_blue_bf], bins, \
    [counts_red, counts_blue] = get_vdf(sigma_red, sigma_blue, volume_analysis)

## VDF
plt.plot(maxis, phi_red_bf, lw=5, ls='--', color='maroon', label='Best-fit')
plt.plot(maxis, phi_red_d, lw=5, ls='-', color='maroon', label='Data')
plt.plot(maxis, phi_blue_bf, lw=5, ls='--', color='cornflowerblue', label='Best-fit')
plt.plot(maxis, phi_blue_d, lw=5, ls='-', color='cornflowerblue', label='Data')
plt.legend(loc='best', prop={'size': 30})
plt.show()

## Blue fraction vs velocity dispersion
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.plot(bin_centers, result_bf[0], lw=5, ls='--', color='k', label='Best-fit')
plt.plot(bin_centers, result_data[0], lw=5, ls='-', color='k', label='Data')
plt.xlabel('Velocity dispersion')
plt.ylabel('Blue fraction')
plt.legend(loc='best', prop={'size': 30})
plt.show()


## velocity dispersion vs group halo mass for red and blue group centrals 
## Binned both ways

bins_grp_halo = np.linspace(11, 14, 7)
result_red_bf = bs(mhalo_red, sigma_red, statistic='median', bins=bins_grp_halo)
result_blue_bf = bs(mhalo_blue, sigma_blue, statistic='median', bins=bins_grp_halo)
result_red_data = bs(mhalo_red_data, sigma_red_data, statistic='median', bins=bins_grp_halo)
result_blue_data = bs(mhalo_blue_data, sigma_blue_data, statistic='median', bins=bins_grp_halo)
bin_centers = 0.5 * (bins_grp_halo[1:] + bins_grp_halo[:-1])

plt.plot(bin_centers, result_red_bf[0], lw=5, ls='-', color='maroon', label='Best-fit')
plt.plot(bin_centers, result_blue_bf[0], lw=5, ls='-', color='cornflowerblue', label='Best-fit')
plt.plot(bin_centers, result_red_data[0], lw=5, ls='--', color='maroon', label='Data')
plt.plot(bin_centers, result_blue_data[0], lw=5, ls='--', color='cornflowerblue', label='Data')
plt.xlabel('Group Halo Mass')
plt.ylabel('Average velocity dispersion')
plt.title('Sigma vs group halo mass (sigma calculated per group)')
plt.legend(loc='best', prop={'size': 30})
plt.show()

bins_sigma = np.arange(0, 600, 50)
result_red_bf = bs(sigma_red, mhalo_red, statistic='median', bins=bins_sigma)
result_blue_bf = bs(sigma_blue, mhalo_blue, statistic='median', bins=bins_sigma)
result_red_data = bs(sigma_red_data, mhalo_red_data, statistic='median', bins=bins_sigma)
result_blue_data = bs(sigma_blue_data, mhalo_blue_data, statistic='median', bins=bins_sigma)
bin_centers = 0.5 * (bins_sigma[1:] + bins_sigma[:-1])

plt.plot(bin_centers, result_red_bf[0], lw=5, ls='-', color='maroon', label='Best-fit')
plt.plot(bin_centers, result_blue_bf[0], lw=5, ls='-', color='cornflowerblue', label='Best-fit')
plt.plot(bin_centers, result_red_data[0], lw=5, ls='--', color='maroon', label='Data')
plt.plot(bin_centers, result_blue_data[0], lw=5, ls='--', color='cornflowerblue', label='Data')
plt.xlabel('Average group halo mass')
plt.xlabel('Velocity dispersion')
plt.title('Group halo mass vs sigma (sigma calculated per group)')
plt.legend(loc='best', prop={'size': 30})
plt.show()

################################################################################
#* Delta V (binned both ways) centrals not included

#* Bins of Delta V

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

stat_red_mocks = []
stat_blue_mocks = []
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
            (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

        ## Using best-fit found for new ECO data using result from chain 34
        ## i.e. hybrid quenching model
        Mstar_q = 10.54 # Msun/h
        Mh_q = 14.09 # Msun/h
        mu = 0.77
        nu = 0.17

        ## Using best-fit found for new ECO data using result from chain 35
        ## i.e. halo quenching model
        Mh_qc = 12.29 # Msun/h
        Mh_qs = 12.78 # Msun/h
        mu_c = 1.37
        mu_s = 1.48

        if quenching == 'hybrid':
            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                'nonvishnu')
        elif quenching == 'halo':
            theta = [Mh_qc, Mh_qs, mu_c, mu_s]
            f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                'nonvishnu')
        mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

        red_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='R')]
        blue_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='B')]

        deltav_red_mock = []
        deltav_blue_mock = []
        red_cen_mstar_mock = []
        blue_cen_mstar_mock = []

        for key in red_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                if val > 0 :
                    deltav_red_mock.append(val)
                    red_cen_mstar_mock.append(cen_mstar)

        for key in blue_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                if val > 0 :
                    deltav_blue_mock.append(val)
                    blue_cen_mstar_mock.append(cen_mstar)

        deltav_red_mock = np.asarray(deltav_red_mock)
        deltav_blue_mock = np.asarray(deltav_blue_mock)
        red_cen_mstar_mock = np.asarray(red_cen_mstar_mock)
        blue_cen_mstar_mock = np.asarray(blue_cen_mstar_mock)

        deltav_red_mock = np.log10(deltav_red_mock)
        deltav_blue_mock = np.log10(deltav_blue_mock)

        mean_stats_red_mock = bs(deltav_red_mock, red_cen_mstar_mock, 
            statistic='mean', bins=np.linspace(-1,4,10))
        mean_stats_blue_mock = bs(deltav_blue_mock, blue_cen_mstar_mock,
            statistic='mean', bins=np.linspace(-1,4,10))

        stat_red_mocks.append(mean_stats_red_mock[0])
        stat_blue_mocks.append(mean_stats_blue_mock[0])

deltav_red_data = []
deltav_blue_data = []
red_cen_mstar_data = []
blue_cen_mstar_data = []

red_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='R')]
blue_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='B')]

for key in red_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_red_data.append(val)
            red_cen_mstar_data.append(cen_mstar)

for key in blue_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_blue_data.append(val)
            blue_cen_mstar_data.append(cen_mstar)

deltav_red_data = np.asarray(deltav_red_data)
deltav_blue_data = np.asarray(deltav_blue_data)
red_cen_mstar_data = np.asarray(red_cen_mstar_data)
blue_cen_mstar_data = np.asarray(blue_cen_mstar_data)

deltav_red_data = np.log10(deltav_red_data)
deltav_blue_data = np.log10(deltav_blue_data)


deltav_red_bf = []
deltav_blue_bf = []
red_cen_mstar_bf = []
blue_cen_mstar_bf = []

red_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='R')]
blue_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='B')]

for key in red_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_red_bf.append(val)
            red_cen_mstar_bf.append(cen_mstar)

for key in blue_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])

    for val in deltav:
        if val > 0 :
            deltav_blue_bf.append(val)
            blue_cen_mstar_bf.append(cen_mstar)

deltav_red_bf = np.asarray(deltav_red_bf)
deltav_blue_bf = np.asarray(deltav_blue_bf)
red_cen_mstar_bf = np.asarray(red_cen_mstar_bf)
blue_cen_mstar_bf = np.asarray(blue_cen_mstar_bf)

deltav_red_bf = np.log10(deltav_red_bf)
deltav_blue_bf = np.log10(deltav_blue_bf)

mean_stats_red_data = bs(deltav_red_data, red_cen_mstar_data, 
    statistic='mean', bins=np.linspace(-1,4,10))
mean_stats_blue_data = bs(deltav_blue_data, blue_cen_mstar_data,
    statistic='mean', bins=np.linspace(-1,4,10))

mean_stats_red_bf = bs(deltav_red_bf, red_cen_mstar_bf, 
    statistic='mean', bins=np.linspace(-1,4,10))
mean_stats_blue_bf = bs(deltav_blue_bf, blue_cen_mstar_bf,
    statistic='mean', bins=np.linspace(-1,4,10))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

error_red = np.nanstd(stat_red_mocks, axis=0)
error_blue = np.nanstd(stat_blue_mocks, axis=0)

# dr, = plt.plot(mean_centers_red, mean_stats_red_data[0], lw=3, c='darkred', ls='-')
# db, = plt.plot(mean_centers_blue, mean_stats_blue_data[0], lw=3, c='darkblue', ls='-')
dr = plt.errorbar(mean_centers_red, mean_stats_red_data[0], yerr=error_red, 
    lw=3, c='darkred', ls='none', capsize=10, marker='^', markersize=10)
db = plt.errorbar(mean_centers_blue, mean_stats_blue_data[0], yerr=error_blue, 
    lw=3, c='darkblue', ls='none', capsize=10, marker='^', markersize=10)

bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], lw=3, c='indianred', ls='--')
bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], lw=3, c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.ylim(8.9,)

plt.xlabel(r'\boldmath$\log_{10}\ \Delta v \left[\mathrm{km/s} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

#* Bins of M* group cen

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

stat_red_mocks = []
stat_blue_mocks = []
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
            (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

        ## Using best-fit found for new ECO data using result from chain 34
        ## i.e. hybrid quenching model
        Mstar_q = 10.54 # Msun/h
        Mh_q = 14.09 # Msun/h
        mu = 0.77
        nu = 0.17

        ## Using best-fit found for new ECO data using result from chain 35
        ## i.e. halo quenching model
        Mh_qc = 12.29 # Msun/h
        Mh_qs = 12.78 # Msun/h
        mu_c = 1.37
        mu_s = 1.48

        if quenching == 'hybrid':
            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                'nonvishnu')
        elif quenching == 'halo':
            theta = [Mh_qc, Mh_qs, mu_c, mu_s]
            f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                'nonvishnu')
        mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

        red_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='R')]
        blue_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='B')]

        deltav_red_mock = []
        deltav_blue_mock = []
        red_cen_mstar_mock = []
        blue_cen_mstar_mock = []

        for key in red_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                if val > 0 :
                    deltav_red_mock.append(val)
                    red_cen_mstar_mock.append(cen_mstar)

        for key in blue_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                if val > 0 :
                    deltav_blue_mock.append(val)
                    blue_cen_mstar_mock.append(cen_mstar)

        deltav_red_mock = np.asarray(deltav_red_mock)
        deltav_blue_mock = np.asarray(deltav_blue_mock)
        red_cen_mstar_mock = np.asarray(red_cen_mstar_mock)
        blue_cen_mstar_mock = np.asarray(blue_cen_mstar_mock)

        deltav_red_mock = np.log10(deltav_red_mock)
        deltav_blue_mock = np.log10(deltav_blue_mock)

        mean_stats_red_mock = bs(red_cen_mstar_mock, deltav_red_mock, 
            statistic='mean', bins=np.linspace(8.9,12,8))
        mean_stats_blue_mock = bs(blue_cen_mstar_mock, deltav_blue_mock,
            statistic='mean', bins=np.linspace(8.9,12,8))

        stat_red_mocks.append(mean_stats_red_mock[0])
        stat_blue_mocks.append(mean_stats_blue_mock[0])

deltav_red_data = []
deltav_blue_data = []
red_cen_mstar_data = []
blue_cen_mstar_data = []

red_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='R')]
blue_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='B')]

for key in red_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_red_data.append(val)
            red_cen_mstar_data.append(cen_mstar)

for key in blue_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_blue_data.append(val)
            blue_cen_mstar_data.append(cen_mstar)

deltav_red_data = np.asarray(deltav_red_data)
deltav_blue_data = np.asarray(deltav_blue_data)
red_cen_mstar_data = np.asarray(red_cen_mstar_data)
blue_cen_mstar_data = np.asarray(blue_cen_mstar_data)

deltav_red_data = np.log10(deltav_red_data)
deltav_blue_data = np.log10(deltav_blue_data)


deltav_red_bf = []
deltav_blue_bf = []
red_cen_mstar_bf = []
blue_cen_mstar_bf = []

red_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='R')]
blue_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='B')]

for key in red_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
    for val in deltav:
        if val > 0 :
            deltav_red_bf.append(val)
            red_cen_mstar_bf.append(cen_mstar)

for key in blue_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])

    for val in deltav:
        if val > 0 :
            deltav_blue_bf.append(val)
            blue_cen_mstar_bf.append(cen_mstar)

deltav_red_bf = np.asarray(deltav_red_bf)
deltav_blue_bf = np.asarray(deltav_blue_bf)
red_cen_mstar_bf = np.asarray(red_cen_mstar_bf)
blue_cen_mstar_bf = np.asarray(blue_cen_mstar_bf)

deltav_red_bf = np.log10(deltav_red_bf)
deltav_blue_bf = np.log10(deltav_blue_bf)

## Stack delta V and measure mean
mean_stats_red_data = bs(red_cen_mstar_data, deltav_red_data, 
    statistic='mean', bins=np.linspace(8.9,12,8))
mean_stats_blue_data = bs(blue_cen_mstar_data, deltav_blue_data,
    statistic='mean', bins=np.linspace(8.9,12,8))

mean_stats_red_bf = bs(red_cen_mstar_bf, deltav_red_bf,
    statistic='mean', bins=np.linspace(8.9,12,8))
mean_stats_blue_bf = bs(blue_cen_mstar_bf, deltav_blue_bf,
    statistic='mean', bins=np.linspace(8.9,12,8))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

error_red = np.nanstd(stat_red_mocks, axis=0)
error_blue = np.nanstd(stat_blue_mocks, axis=0)

# dr, = plt.plot(mean_centers_red, mean_stats_red_data[0], lw=3, c='darkred', ls='-')
# db, = plt.plot(mean_centers_blue, mean_stats_blue_data[0], lw=3, c='darkblue', ls='-')
dr = plt.errorbar(mean_centers_red, mean_stats_red_data[0], yerr=error_red, 
    lw=3, c='darkred', ls='none', capsize=10, marker='^', markersize=10)
db = plt.errorbar(mean_centers_blue, mean_stats_blue_data[0], yerr=error_blue, 
    lw=3, c='darkblue', ls='none', capsize=10, marker='^', markersize=10)

bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], lw=3, c='indianred', ls='--')
bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], lw=3, c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.ylabel(r'\boldmath$\overline{\log_{10}\ \Delta v} \left[\mathrm{km/s} \right]$', fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

## Measuring velocity dispersion per bin of central stellar mass (stacked) 
## instead of measuring it per group and then averaging sigma per bin of central
## stellar mass

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

std_stat_red_mocks = []
std_stat_blue_mocks = []
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
            (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

        ## Using best-fit found for new ECO data using result from chain 34
        ## i.e. hybrid quenching model
        Mstar_q = 10.54 # Msun/h
        Mh_q = 14.09 # Msun/h
        mu = 0.77
        nu = 0.17

        ## Using best-fit found for new ECO data using result from chain 35
        ## i.e. halo quenching model
        Mh_qc = 12.29 # Msun/h
        Mh_qs = 12.78 # Msun/h
        mu_c = 1.37
        mu_s = 1.48

        if quenching == 'hybrid':
            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                'nonvishnu')
        elif quenching == 'halo':
            theta = [Mh_qc, Mh_qs, mu_c, mu_s]
            f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                'nonvishnu')
        mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

        red_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='R')]
        blue_keys = mock_pd.groupid.loc[(mock_pd.g_galtype == 1)&(mock_pd.colour_label=='B')]

        deltav_red_mock = []
        deltav_blue_mock = []
        red_cen_mstar_mock = []
        blue_cen_mstar_mock = []

        for key in red_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            n_members_red = len(group)
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                # if val > 0 :
                deltav_red_mock.append(val)
                red_cen_mstar_mock.append(cen_mstar)

        for key in blue_keys:
            group = mock_pd.loc[mock_pd.groupid == key]
            n_members_blue = len(group)
            if len(group) == 1:
                continue
            cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
            cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
            deltav = np.abs(group.cz.values - len(group)*[cen_cz_grp])
            for val in deltav:
                # if val > 0 :
                deltav_blue_mock.append(val)
                blue_cen_mstar_mock.append(cen_mstar)

        deltav_red_mock = np.asarray(deltav_red_mock)
        deltav_blue_mock = np.asarray(deltav_blue_mock)
        red_cen_mstar_mock = np.asarray(red_cen_mstar_mock)
        blue_cen_mstar_mock = np.asarray(blue_cen_mstar_mock)

        # deltav_red_mock = np.log10(deltav_red_mock)
        # deltav_blue_mock = np.log10(deltav_blue_mock)

        mean_stats_red_mock = bs(red_cen_mstar_mock, deltav_red_mock,
            statistic='std', bins=np.linspace(8.9,12,8))
        mean_stats_blue_mock = bs(blue_cen_mstar_mock, deltav_blue_mock,
            statistic='std', bins=np.linspace(8.9,12,8))

        std_stat_red_mocks.append(mean_stats_red_mock[0])
        std_stat_blue_mocks.append(mean_stats_blue_mock[0])

std_stat_red_mocks = np.asarray(std_stat_red_mocks)
std_stat_blue_mocks = np.asarray(std_stat_blue_mocks)

deltav_red_data = []
deltav_blue_data = []
red_cen_mstar_data = []
blue_cen_mstar_data = []

std_red_data = []
std_blue_data = []
red_cen_mstar_data = []
blue_cen_mstar_data = []

red_grp_mstar_data = []
blue_grp_mstar_data = []

red_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='R')]
blue_keys = catl.groupid.loc[(catl.g_galtype == 1)&(catl.colour_label=='B')]

for key in red_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    mean_cz_grp = np.average(group.cz.values)
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    group_mstar = np.log10(np.sum(10**(group.logmstar.values)))
    deltav = np.abs(group.cz.values - len(group)*[mean_cz_grp])
    std = np.std(deltav)
    std_red_data.append(std)
    red_cen_mstar_data.append(cen_mstar)
    red_grp_mstar_data.append(group_mstar)

    # for val in deltav:
    #     # if val > 0 :
    #     deltav_red_data.append(val)
    #     red_cen_mstar_data.append(cen_mstar)

for key in blue_keys:
    group = catl.loc[catl.groupid == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['g_galtype'].values == 1].values[0]
    mean_cz_grp = np.average(group.cz.values)
    cen_mstar = group.logmstar.loc[group['g_galtype'].values == 1].values[0]
    group_mstar = np.log10(np.sum(10**(group.logmstar.values)))
    deltav = np.abs(group.cz.values - len(group)*[mean_cz_grp])
    std = np.std(deltav)
    std_blue_data.append(std)
    blue_cen_mstar_data.append(cen_mstar)
    blue_grp_mstar_data.append(group_mstar)

    # for val in deltav:
    #     # if val > 0 :
    #     deltav_blue_data.append(val)
    #     blue_cen_mstar_data.append(cen_mstar)

deltav_red_data = np.asarray(deltav_red_data)
deltav_blue_data = np.asarray(deltav_blue_data)
red_cen_mstar_data = np.asarray(red_cen_mstar_data)
blue_cen_mstar_data = np.asarray(blue_cen_mstar_data)

std_red_data = np.asarray(std_red_data)
std_blue_data = np.asarray(std_blue_data)
red_grp_mstar_data = np.asarray(red_grp_mstar_data)
blue_grp_mstar_data = np.asarray(blue_grp_mstar_data)

# deltav_red_data = np.log10(deltav_red_data)
# deltav_blue_data = np.log10(deltav_blue_data)


deltav_red_bf = []
deltav_blue_bf = []
red_cen_mstar_bf = []
blue_cen_mstar_bf = []

std_red_bf = []
std_blue_bf = []
red_cen_mstar_bf = []
blue_cen_mstar_bf = []

red_grp_mstar_bf = []
blue_grp_mstar_bf = []

red_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='R')]
blue_keys = gals_df_subset.groupid_1.loc[(gals_df_subset.grp_censat_1 == 1)&(gals_df_subset.colour_label=='B')]

for key in red_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    mean_cz_grp = np.average(group.cz.values)
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    group_mstar = np.log10(np.sum(10**(group.behroozi_bf.values)))
    deltav = np.abs(group.cz.values - len(group)*[mean_cz_grp])
    std = np.std(deltav)
    std_red_bf.append(std)
    red_cen_mstar_bf.append(cen_mstar)
    red_grp_mstar_bf.append(group_mstar)
    # for val in deltav:
    #     # if val > 0 :
    #     deltav_red_bf.append(val)
    #     red_cen_mstar_bf.append(cen_mstar)

for key in blue_keys:
    group = gals_df_subset.loc[gals_df_subset.groupid_1 == key]
    if len(group) == 1:
        continue
    cen_cz_grp = group.cz.loc[group['grp_censat_1'].values == 1].values[0]
    mean_cz_grp = np.average(group.cz.values)
    cen_mstar = group.behroozi_bf.loc[group['grp_censat_1'].values == 1].values[0]
    group_mstar = np.log10(np.sum(10**(group.behroozi_bf.values)))
    deltav = np.abs(group.cz.values - len(group)*[mean_cz_grp])
    std = np.std(deltav)
    std_blue_bf.append(std)
    blue_cen_mstar_bf.append(cen_mstar)
    blue_grp_mstar_bf.append(group_mstar)

    # for val in deltav:
    #     # if val > 0 :
    #     deltav_blue_bf.append(val)
    #     blue_cen_mstar_bf.append(cen_mstar)

deltav_red_bf = np.asarray(deltav_red_bf)
deltav_blue_bf = np.asarray(deltav_blue_bf)
red_cen_mstar_bf = np.asarray(red_cen_mstar_bf)
blue_cen_mstar_bf = np.asarray(blue_cen_mstar_bf)

std_red_bf = np.asarray(std_red_bf)
std_blue_bf = np.asarray(std_blue_bf)
red_grp_mstar_bf = np.asarray(red_grp_mstar_bf)
blue_grp_mstar_bf = np.asarray(blue_grp_mstar_bf)

# deltav_red_bf = np.log10(deltav_red_bf)
# deltav_blue_bf = np.log10(deltav_blue_bf)

mean_stats_red_data = bs(red_cen_mstar_data, deltav_red_data, 
    statistic='std', bins=np.linspace(8.9,12,8))
mean_stats_blue_data = bs(blue_cen_mstar_data, deltav_blue_data,
    statistic='std', bins=np.linspace(8.9,12,8))

mean_stats_red_bf = bs(red_cen_mstar_bf, deltav_red_bf,
    statistic='std', bins=np.linspace(8.9,12,8))
mean_stats_blue_bf = bs(blue_cen_mstar_bf, deltav_blue_bf,
    statistic='std', bins=np.linspace(8.9,12,8))

mean_stats_red_data = bs(red_grp_mstar_data, std_red_data, 
    statistic='mean', bins=np.linspace(8.9,12,5))
mean_stats_blue_data = bs(blue_grp_mstar_data, std_blue_data,
    statistic='mean', bins=np.linspace(8.9,12,5))

mean_stats_red_bf = bs(red_grp_mstar_bf, std_red_bf,
    statistic='mean', bins=np.linspace(8.9,12,5))
mean_stats_blue_bf = bs(blue_grp_mstar_bf, std_blue_bf,
    statistic='mean', bins=np.linspace(8.9,12,5))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

error_red = np.nanstd(std_stat_red_mocks, axis=0)
error_blue = np.nanstd(std_stat_blue_mocks, axis=0)

dr, = plt.plot(mean_centers_red, np.log10(mean_stats_red_data[0]), lw=3, c='darkred', ls='-')
db, = plt.plot(mean_centers_blue, np.log10(mean_stats_blue_data[0]), lw=3, c='darkblue', ls='-')

# dr = plt.errorbar(mean_centers_red, mean_stats_red_data[0], yerr=error_red, 
#     lw=3, c='darkred', ls='none', capsize=10, marker='^', markersize=10)
# db = plt.errorbar(mean_centers_blue, mean_stats_blue_data[0], yerr=error_blue, 
    # lw=3, c='darkblue', ls='none', capsize=10, marker='^', markersize=10)

bfr, = plt.plot(mean_centers_red, np.log10(mean_stats_red_bf[0]), lw=3, c='indianred', 
    ls='--')
bfb, = plt.plot(mean_centers_blue, np.log10(mean_stats_blue_bf[0]), lw=3, 
    c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

################################################################################
# Group multiplicity function from data and best-fit

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

n_members_mocks = []
box_id_arr = np.linspace(5001,5008,8)
for box in box_id_arr:
    box = int(box)
    temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
        mock_name) 
    for num in range(num_mocks):
        deltav_red_mocks = []
        deltav_blue_mocks = []
        red_cen_mstar_mocks = []
        blue_cen_mstar_mocks = []
        filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        print('Box {0} : Mock {1}'.format(box, num))
        mock_pd = reading_catls(filename) 
        mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer
        mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
            (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

        ## Using best-fit found for new ECO data using result from chain 34
        ## i.e. hybrid quenching model
        Mstar_q = 10.54 # Msun/h
        Mh_q = 14.09 # Msun/h
        mu = 0.77
        nu = 0.17

        ## Using best-fit found for new ECO data using result from chain 35
        ## i.e. halo quenching model
        Mh_qc = 12.29 # Msun/h
        Mh_qs = 12.78 # Msun/h
        mu_c = 1.37
        mu_s = 1.48

        if quenching == 'hybrid':
            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                'nonvishnu')
        elif quenching == 'halo':
            theta = [Mh_qc, Mh_qs, mu_c, mu_s]
            f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                'nonvishnu')
        mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

        groups = mock_pd.groupby('groupid')
        n_members_mock = []
        keys = groups.groups.keys()
        for key in keys:
            group = groups.get_group(key)
            n_members_mock.append(len(group))
        n_members_mocks.append(n_members_mock)


groups = catl.groupby('groupid')
keys_data = groups.groups.keys()
n_members_data = []
singleton_data_counter = 0
for key in keys_data:
    group = groups.get_group(key)
    if len(group) == 1:
        singleton_data_counter += 1
    n_members_data.append(len(group))

groups = gals_df_subset.groupby('groupid_1')
keys_bf = groups.groups.keys()
n_members_bf = []
singleton_bf_counter = 0
for key in keys_bf:
    group = groups.get_group(key)
    if len(group) == 1:
        singleton_bf_counter += 1
    n_members_bf.append(len(group))

bins = np.linspace(0,30,30)
plt.hist(n_members_data, bins, histtype='step', lw=3, ls='--', color='mediumorchid', 
    label='Data')
plt.hist(n_members_bf, bins, histtype='step', lw=3, ls='--', color='teal', 
    label='Best-fit hybrid')
plt.yscale('log')
plt.xlabel(r'Number of galaxies in group')
plt.ylabel(r'Number of groups')
plt.title('Group multiplicity function')
plt.legend(loc='best', prop={'size':20})
plt.show()

binned_mocks = []
for idx in range(len(n_members_mocks)):
    y_mocks,binEdges = np.histogram(n_members_mocks[idx],bins)
    binned_mocks.append(y_mocks)

y_data,binEdges = np.histogram(n_members_data,bins)
y_bf,binEdges = np.histogram(n_members_bf,bins)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
menStd     = np.std(binned_mocks, axis=0)
width      = 1
plt.bar(bincenters, y_data, width=width, fill=False, edgecolor='mediumorchid',
    linewidth=3, yerr=menStd, label='Data')
plt.bar(bincenters, y_bf, width=width, fill=False, edgecolor='teal', 
    linewidth=3, label='Best-fit halo')
plt.yscale('log')
plt.legend(loc='best', prop={'size':20})
plt.title('Group multiplicity function')
plt.xlabel(r'Number of galaxies in group')
plt.ylabel(r'Number of groups')
plt.show()

df_data = pd.DataFrame(data={'n_members_data':n_members_data})
df_bf = pd.DataFrame(data={'n_members_bf':n_members_bf})

singleton_data_counter/len(keys_data)
singleton_bf_counter/len(keys_bf)


##* Estimating effective radius and stellar velocity dispersion
# Msun = 1.989*10**30 #kg
# Rsun = 6.95*10**8 #m
# rho_crit = 9.47*10**-27 #kg/m3
# omega_m = 0.25
# rho_m = rho_crit * omega_m #kg/m3
# gals_df['R_e'] = 0.015*((((gals_df.halo_mvir.values*Msun*3)/(4*np.pi*rho_m))**(1/3))/Rsun) #Rsun
# gals_df['sigma'] = np.log10(10**(gals_df.behroozi_bf.values)/gals_df.R_e.values)

# catl['sigma'] = np.log10((10**(catl.logmstar.values)/2.041)/(((3.08*10**16)/catl.r50.values)/Rsun))

gals_df['R_e'] = 0.015*(gals_df.halo_mvir.values**(1/3))
gals_df['sigma'] = np.log10(10**(gals_df.behroozi_bf.values)/gals_df.R_e.values)

catl['sigma'] = np.log10(10**(catl.logmstar.values)/catl.r50.values)


mean_stats_red_data = bs(catl.sigma.loc[(catl.colour_label=='R')&(catl.g_galtype==1)], catl.logmstar.loc[(catl.colour_label=='R')&(catl.g_galtype==1)], 
    statistic='mean', bins=np.linspace(5,10,10))
mean_stats_blue_data = bs(catl.sigma.loc[(catl.colour_label=='B')&(catl.g_galtype==1)], catl.logmstar.loc[(catl.colour_label=='B')&(catl.g_galtype==1)],
    statistic='mean', bins=np.linspace(5,10,10))

mean_stats_red_bf = bs(gals_df.sigma.loc[(gals_df.colour_label=='R')&(gals_df.grp_censat_1==1)], gals_df.behroozi_bf.loc[(gals_df.colour_label=='R')&(gals_df.grp_censat_1==1)], 
    statistic='mean', bins=np.linspace(5,10,10))
mean_stats_blue_bf = bs(gals_df.sigma.loc[(gals_df.colour_label=='B')&(gals_df.grp_censat_1==1)], gals_df.behroozi_bf.loc[(gals_df.colour_label=='B')&(gals_df.grp_censat_1==1)],
    statistic='mean', bins=np.linspace(5,10,10))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

dr, = plt.plot(mean_centers_red, mean_stats_red_data[0], lw=3, c='darkred', ls='-')
db, = plt.plot(mean_centers_blue, mean_stats_blue_data[0], lw=3, c='darkblue', ls='-')

bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], lw=3, c='indianred', ls='--')
bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], lw=3, c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

mean_stats_red_data = bs(catl.logmstar.loc[(catl.colour_label=='R')&(catl.g_galtype==1)], catl.sigma.loc[(catl.colour_label=='R')&(catl.g_galtype==1)],
    statistic='mean', bins=np.linspace(8.9,12,10))
mean_stats_blue_data = bs(catl.logmstar.loc[(catl.colour_label=='B')&(catl.g_galtype==1)], catl.sigma.loc[(catl.colour_label=='B')&(catl.g_galtype==1)], 
    statistic='mean', bins=np.linspace(8.9,12,10))

mean_stats_red_bf = bs(gals_df.behroozi_bf.loc[(gals_df.colour_label=='R')&(gals_df.grp_censat_1==1)], gals_df.sigma.loc[(gals_df.colour_label=='R')&(gals_df.grp_censat_1==1)], 
    statistic='mean', bins=np.linspace(8.9,12,10))
mean_stats_blue_bf = bs(gals_df.behroozi_bf.loc[(gals_df.colour_label=='B')&(gals_df.grp_censat_1==1)], gals_df.sigma.loc[(gals_df.colour_label=='B')&(gals_df.grp_censat_1==1)], 
    statistic='mean', bins=np.linspace(8.9,12,10))

mean_centers_red = 0.5 * (mean_stats_red_data[1][1:] + \
    mean_stats_red_data[1][:-1])
mean_centers_blue = 0.5 * (mean_stats_blue_data[1][1:] + \
    mean_stats_blue_data[1][:-1])

dr, = plt.plot(mean_centers_red, mean_stats_red_data[0], lw=3, c='darkred', ls='-')
db, = plt.plot(mean_centers_blue, mean_stats_blue_data[0], lw=3, c='darkblue', ls='-')

bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], lw=3, c='indianred', ls='--')
bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], lw=3, c='cornflowerblue', ls='--')

l = plt.legend([(dr, db),(bfr, bfb)], 
    ['Data', 'Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
plt.xlabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()

################################################################################
##* Central stellar mass vs group integrated stellar mass

groups = catl.groupby('groupid')
keys_data = groups.groups.keys()
cen_mstar_data = []
group_mstar_data = []
group_mhalo_data = []
for key in keys_data:
    group = groups.get_group(key)
    cen_mstar = group.logmstar.loc[group.g_galtype==1]
    group_mstar = np.log10(np.sum(10**group.logmstar.values))
    group_mhalo = group.M_group.loc[group.g_galtype==1]
    cen_mstar_data.append(cen_mstar)
    group_mstar_data.append(group_mstar)
    group_mhalo_data.append(group_mhalo)

groups = gals_df_subset.groupby('groupid_1')
keys_bf = groups.groups.keys()
cen_mstar_bf = []
group_mstar_bf = []
group_mhalo_bf = []
for key in keys_bf:
    group = groups.get_group(key)
    cen_mstar = group.behroozi_bf.loc[group.grp_censat_1==1]
    group_mstar = np.log10(np.sum(10**group.behroozi_bf.values))
    group_mhalo = group.M_group.loc[group.grp_censat_1==1]
    cen_mstar_bf.append(cen_mstar)
    group_mstar_bf.append(group_mstar)
    group_mhalo_bf.append(group_mhalo)

group_mstar_data, cen_mstar_data = np.asarray(group_mstar_data), \
    np.asarray(cen_mstar_data)
group_mstar_bf, cen_mstar_bf = np.asarray(group_mstar_bf), \
    np.asarray(cen_mstar_bf)
group_mhalo_bf, group_mhalo_data = np.asarray(group_mhalo_bf), \
    np.asarray(group_mhalo_data)

d = plt.scatter(np.log10((10**group_mstar_data)/2.041), 
    np.log10((10**cen_mstar_data)/2.041), facecolors="None", 
    edgecolors='mediumorchid', s=150, label='Data', zorder=10)
bf = plt.scatter(group_mstar_bf, cen_mstar_bf, facecolors="None", 
    edgecolors='teal', s=150, label='Best-fit', zorder=5)
plt.plot(np.linspace(8.5,12.5,5), np.linspace(8.5,12.5,5), c='k', lw=3, ls='--', 
    zorder=15)

plt.ylabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)

plt.legend(loc='best', prop={'size':30})

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()


##* Group halo mass vs group integrated stellar mass

d = plt.scatter(np.log10((10**group_mstar_data)/2.041), 
    group_mhalo_data, facecolors="None", 
    edgecolors='mediumorchid', s=150, label='Data', zorder=10)
bf = plt.scatter(group_mstar_bf, group_mhalo_bf, facecolors="None", 
    edgecolors='teal', s=150, label='Best-fit', zorder=5)
# plt.plot(np.linspace(8.5,12.5,5), np.linspace(8.5,12.5,5), c='k', lw=3, ls='--', 
#     zorder=15)

plt.ylabel(r'\boldmath$\log_{10}\ M_{h, group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)

plt.legend(loc='best', prop={'size':30})

if quenching == 'hybrid':
    plt.title('Hybrid quenching model | ECO')
elif quenching == 'halo':
    plt.title('Halo quenching model | ECO')

plt.show()




##* Data vs best-fit group stellar mass function

groups = catl.groupby('groupid')
keys = groups.groups.keys()
mstar_group_data_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum(10**group.logmstar.values))
    mstar_group_data_arr.append(group_log_prop_tot)
mstar_group_data_postgrpfinding_new = np.asarray(mstar_group_data_arr)


## Best-fit model

H0 = 100 # (km/s)/Mpc
cz_inner = 3000 # not starting at corner of box
cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

v_inner = vol_sphere(dist_inner)
v_outer = vol_sphere(dist_outer)

v_sphere = v_outer-v_inner
survey_vol = v_sphere/8

groups = gals_df.groupby('groupid_1')
keys = groups.groups.keys()
mstar_group_bf_arr = []
for key in keys:
    group = groups.get_group(key)
    group_log_prop_tot = np.log10(np.sum((10**group.behroozi_bf.values)*2.041))
    mstar_group_bf_arr.append(group_log_prop_tot)


x_grpfind_new,y_grpfind_new,err_grpfind_new = cumu_num_dens(mstar_group_data_postgrpfinding_new, 
    6, None, preprocess.volume, False)
x_bf,y_bf,err_bf = cumu_num_dens(mstar_group_bf_arr, 6, None, survey_vol, False)


plt.scatter(x_grpfind_new, np.log10(y_grpfind_new), color='k', s=200, marker='^', label='Post-group finding (new)')
plt.scatter(x_bf, np.log10(y_bf), color='cornflowerblue', s=200, marker='^', label='Best-fit halo')

plt.xlabel(r'\boldmath$\log_{10}\ M_{\star,group} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$(n > M) \left[\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
plt.title('Group stellar mass function')
plt.legend(loc='best',prop={'size':30})
plt.show()



eco_buff = eco_buff.loc[eco_buff.logmstar.values > 0]
Mr = eco_buff.absrmag.values
mstar = eco_buff.logmstar.values

fig, axis = plt.subplots(1,1)
plt.scatter(Mr, mstar, facecolors="None", edgecolors='mediumorchid')
plt.vlines(-17.33, 6, 12, colors='k', lw=3)
plt.hlines(8.9, -13, -25, colors='k', lw=3)
axis.invert_xaxis()
plt.xlabel(r'\boldmath$M_{r}$')
plt.ylabel(r'\boldmath$\log_{10}\ M_{\star} \left[\mathrm{M_\odot}\ \right]$', fontsize=30)
plt.show()

## 24 galaxies that are above the mass completeness but below the luminosity 
## completeness

#* Measuring grp sigma by stacking central sigma in resolve like in Seo et al 2020
#? Not possible since very few groups where vdisp measurements are not 0

resolve = pd.read_csv('/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
    'resolve_statistics/data/raw/resolve/RESOLVE_liveJune2018.csv')
resolve = assign_colour_label_data(resolve)

resolveb = resolve.loc[(resolve.f_b.values == 1) &
    (resolve.grpcz.values >= 4500) & 
    (resolve.grpcz.values <= 7000) & 
    (resolve.absrmag.values <= -17)]

resolvea = resolve.loc[(resolve.f_a.values == 1) &
    (resolve.grpcz.values >= 4500) & 
    (resolve.grpcz.values <= 7000) & 
    (resolve.absrmag.values <= -17.33)]


red_keys = resolveb.grp.loc[(resolveb.fc == 1)&(resolveb.colour_label=='R')]
blue_keys = resolveb.grp.loc[(resolveb.fc == 1)&(resolveb.colour_label=='B')]

cenvdisp_red_data = []
cenvdisp_blue_data = []
satvdisp_red_data = []
satvdisp_blue_data = []
for key in red_keys:
    group = resolveb.loc[resolveb.grp == key]
    len_grp = len(group)
    if len_grp == 1:
        continue
    cen_vdisp = group.vdisp.loc[group.fc.values==1].values[0]
    sat_vdisp = group.vdisp.loc[group.fc.values==0].values
    if 0 in sat_vdisp:
        continue
    for idx in range(len(sat_vdisp)):
        satvdisp_red_data.append(sat_vdisp[idx])
        cenvdisp_red_data.append(cen_vdisp)

for key in blue_keys:
    group = resolveb.loc[resolveb.grp == key]
    if len(group) == 1:
        continue
    cen_vdisp = group.vdisp.loc[group.fc.values==1].values[0]
    sat_vdisp = group.vdisp.loc[group.fc.values==0].values
    if 0 in sat_vdisp:
        continue
    for idx in range(len(sat_vdisp)):
        satvdisp_blue_data.append(sat_vdisp[idx])
        cenvdisp_blue_data.append(cen_vdisp)

cenvdisp_red_data = np.asarray(cenvdisp_red_data)
cenvdisp_blue_data = np.asarray(cenvdisp_blue_data)

mean_stats_red_data = bs(red_cen_mstar_data, deltav_red_data, 
    statistic='std', bins=np.linspace(8.9,12,8))
mean_stats_blue_data = bs(blue_cen_mstar_data, deltav_blue_data,
    statistic='std', bins=np.linspace(8.9,12,8))
