"""
{This script compares best-fit shmr from forward modeling, data shmr as well 
as shmr if abundance matching was done using individual galaxies as opposed 
to groups in data}
"""

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
from progressbar import ProgressBar
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import interpolate
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
                (eco_buff.absrmag.values <= -17.33) &
                (eco_buff.logmstar.values >= 8.9)]
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
                    (resolve_live18.absrmag.values <= -17.33) & 
                    (resolve_live18.logmstar.values >= 8.9)]
                # catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                #     (resolve_live18.grpcz.values >= 4500) & 
                #     (resolve_live18.grpcz.values <= 7000) & 
                #     (resolve_live18.absrmag.values <= -17.33) & 
                #     (resolve_live18.logmstar.values >= 8.9)]
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
    """
    prim_haloprop_key : String giving the column name of the primary halo 
    property governing stellar mass.
    """
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_median, \
        prim_haloprop_key='halo_macc')
    model.populate_mock(halocat,seed=5)

    return model

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def cumu_num_dens(data, bins, weights, volume, mag_bool, h1_bool):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        data = np.log10((10**data) / 2.041)
    freq,edg = np.histogram(data, bins=bins, weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not mag_bool:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,edg,n_cumu,err_poiss,bin_width

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

    if mf_type == 'smf' and survey == 'eco':
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
    
    if mf_type == 'smf' and survey == 'eco':
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

def diff_smf(mstar_arr, volume, h1_bool):
    """
    Calculates differential stellar mass function

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
        if survey == 'eco':
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        elif survey == 'resolvea':
            # different to avoid nan in inverse corr mat
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
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
    for num in range(num_mocks):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
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
            cen_halos = mock_pd.M_group.loc[mock_pd.cs_flag == 1]

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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_mocks = path_to_external + 'ECO_mvir_catls/'

global survey 
global mf_type
survey = 'eco'
mf_type = 'smf'

halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
catl_file = path_to_raw + "eco_all.csv"
catl, volume, cvar, z_median = read_data_catl(catl_file, survey)

logmstar = np.log10((10**catl.logmstar.values)/2.041)

model_init = halocat_init(halo_catalog, z_median)
halo_table = model_init.mock.halo_table.to_pandas()
halo_macc = np.log10(halo_table.halo_macc.values)
v_sim = 130**3 #(Mpc/h)^3

## HAM 
logmstar_sorted = np.sort(logmstar)[::-1]
halomacc_sorted = np.sort(halo_macc)[::-1]
n_logmstar = []
for idx,val in enumerate(logmstar_sorted): 
    n = (idx+1)/volume 
    n_logmstar.append(n)
n_halomacc = []
for idx,val in enumerate(halomacc_sorted): 
    n = (idx+1)/v_sim 
    n_halomacc.append(n)

hmf_interp_func = interpolate.interp1d(n_halomacc, halomacc_sorted)
ham_mass = [hmf_interp_func(val) for val in n_logmstar]

if mf_type == 'smf':
    path_to_proc = path_to_proc + 'smhm_run4_errjk/'
elif mf_type == 'bmf':
    path_to_proc = path_to_proc + 'bmhm_run2/'

chi2_file = path_to_proc + '{0}_chi2.txt'.format(survey)

if mf_type == 'smf' and survey == 'eco':
    chain_file = path_to_proc + 'mcmc_{0}.dat'.format(survey)
else:
    chain_file = path_to_proc + 'mcmc_{0}_raw.txt'.format(survey)

print('Reading chi-squared file')
chi2 = read_chi2(chi2_file)

print('Reading mcmc chain file')
mcmc_table = read_mcmc(chain_file)

print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2 = get_paramvals_percentile(mcmc_table, 68, 
chi2)

print('Getting best fit model and centrals')
maxis_bf, phi_bf, err_tot_bf, counts_bf, cen_gals_bf, cen_halos_bf = \
    get_best_fit_model(bf_params)

# Using abundance matched group halo mass from mocks for errors in data
xmhm_mocks = get_xmhm_mocks(survey, path_to_mocks, mf_type)

logmhalo_cen = np.log10((10**catl.logmh_s.loc[catl.fc.values == 1].values)/2.041)
logmstar_cen = np.log10((10**catl.logmstar.loc[catl.fc.values == 1].values)/2.041)
x_data,y_data,y_std_data,y_std_err_data = Stats_one_arr(logmhalo_cen,
    logmstar_cen,base=0.4,bin_statval='center')
x_mocks, y_mocks, y_std_err_mocks = xmhm_mocks[0], xmhm_mocks[1], \
    xmhm_mocks[2]
y_std_err_data = np.std(y_mocks, axis=0)

x_ham,y_ham,y_std_ham,y_std_err_ham = Stats_one_arr(ham_mass,
    logmstar_sorted,base=0.4,bin_statval='center')

x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(cen_halos_bf,
    cen_gals_bf,base=0.4,bin_statval='center')

fig1 = plt.figure(figsize=(10,10))
plt.fill_between(x_data,y_data-y_std_err_data,y_data+y_std_err_data,
    color='lightgray',label=r'data')
plt.errorbar(x_ham,y_ham,yerr=y_std_err_ham,color='k',fmt='-s',
    ecolor='k',markersize=5,capsize=5,capthick=0.5,
    label=r'galaxy HAM',zorder=10)
plt.errorbar(x_bf,y_bf,color='mediumorchid',fmt='-s',
    ecolor='mediumorchid',markersize=4,capsize=5,capthick=0.5,
    label=r'best-fit',zorder=20)

plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
if mf_type == 'smf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**8.9)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**8.7)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
elif mf_type == 'bmf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**9.4)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**9.1)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 15})
plt.show()