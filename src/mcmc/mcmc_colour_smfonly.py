"""
{This script carries out an MCMC analysis to parametrize the SMHM for red and 
 blue galaxies in ECO}
"""

# Built-in/Generic Imports
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from multiprocessing import Pool, Queue
import pandas as pd
import numpy as np
import argparse
import warnings
import random
import emcee 
import math
import os

__author__ = '[Mehnaaz Asad]'

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

def get_paramvals_percentile(table, percentile, chi2_arr):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 100 sample

    Parameters
    ----------
    table: pandas dataframe
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2_arr: array
        Array of chi^2 values

    Returns
    ---------
    subset: ndarray
        Random 100 sample of param values from 68th percentile
    """ 
    percentile = percentile/100
    table['chi2'] = chi2_arr
    table = table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(percentile*len(table))
    mcmc_table_pctl = table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:5]
    subset = mcmc_table_pctl.drop_duplicates().sample(100).values[:,:5] 
    subset = np.insert(subset, 0, bf_params, axis=0)

    return subset

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

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool):
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
        # print("Data ", logmbary_arr.min(), logmbary_arr.max())
    else:
        logmbary_arr = np.log10(mass_arr)
        # print(logmbary_arr.min(), logmbary_arr.max())
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        if survey == 'eco':
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

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

def mcmc(nproc, nwalkers, nsteps, phi_red_data, phi_blue_data, err, corr_mat_inv):
    """
    MCMC analysis

    Parameters
    ----------
    nproc: int
        Number of processes to spawn
    
    nwalkers: int
        Number of walkers to use
    
    nsteps: int
        Number of steps to run MCMC for
    
    phi: array
        Array of y-axis values of mass function

    err: array
        Array of error per bin of mass function

    Returns
    ---------
    sampler: multidimensional array
        Result of running emcee 

    """
    ## Starting at best-fit values found in optimize_hybridqm_eco.py
    Mstar_q = 10.49 # Msun/h
    Mh_q = 14.03 # Msun/h
    mu = 0.69
    nu = 0.148

    hybrid_param_vals = [Mstar_q, Mh_q, mu, nu]

    ndim = 4
    p0 = hybrid_param_vals + 0.1*np.random.rand(ndim*nwalkers).\
        reshape((nwalkers, ndim))

    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
            args=(phi_red_data, phi_blue_data, err, corr_mat_inv), pool=pool)
        start = time.time()
        for i,result in enumerate(sampler.sample(p0, iterations=nsteps, 
            storechain=False)):
            position = result[0]
            chi2 = np.array(result[3])[:,0]
            # mock_num = np.array(result[3])[:,1].astype(int)
            print("Iteration number {0} of {1}".format(i+1,nsteps))
            chain_fname = open("mcmc_{0}_colour_raw.txt".format(survey), "a")
            chi2_fname = open("{0}_colour_chi2.txt".format(survey), "a")
            # mocknum_fname = open("{0}_colour_mocknum.txt".format(survey), "a")
            for k in range(position.shape[0]):
                chain_fname.write(str(position[k]).strip("[]"))
                chain_fname.write("\n")
            chain_fname.write("# New slice\n")
            for k in range(chi2.shape[0]):
                chi2_fname.write(str(chi2[k]).strip("[]"))
                chi2_fname.write("\n")
            # for k in range(mock_num.shape[0]):
            #     mocknum_fname.write(str(mock_num[k]).strip("[]"))
            #     mocknum_fname.write("\n")
            chain_fname.close()
            chi2_fname.close()
            # mocknum_fname.close()
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    
    return sampler

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
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    """"""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

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

def get_host_halo_mock(gals_df, mock):
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

    df = gals_df.copy()

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
        cen_halos = []
        sat_halos = []
        for index, value in enumerate(df.cs_flag):
            if value == 1:
                cen_halos.append(df.halo_mvir.values[index])
            else:
                sat_halos.append(df.halo_mvir.values[index])
    else:
        cen_halos = []
        sat_halos = []
        for index, value in enumerate(df.cs_flag):
            if value == 1:
                cen_halos.append(10**(df.loghalom.values[index]))
            else:
                sat_halos.append(10**(df.loghalom.values[index]))

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(gals_df, mock, randint=None):
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

    df = gals_df.copy()
    if mock == 'vishnu' and randint:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['{0}'.format(randint)].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['{0}'.format(randint)].values[idx]))

    elif mock == 'vishnu':
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['stellar_mass'].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['stellar_mass'].values[idx]))
    
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

def lnprob(theta, phi_red_data, phi_blue_data, err, corr_mat_inv):
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
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, [chi2, randint_logmstar]
    if theta[1] < 0 or theta[1] > 16:
        chi2 = -np.inf
        return -np.inf, [chi2, randint_logmstar]
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, [chi2, randint_logmstar]
    if theta[3] < 0 or theta[3] > 5:
        chi2 = -np.inf
        return -np.inf, [chi2, randint_logmstar]       

    warnings.simplefilter("error", (UserWarning, RuntimeWarning))
    try: 
        if randint_logmstar:
            cols_to_use = ['halo_mvir', 'cs_flag', 'cz', \
                '{0}_y'.format(randint_logmstar), \
                'g_galtype_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar)]

            if randint_logmstar < 51:
                gals_df_mock = gal_group_df_one.loc[gal_group_df_one['{0}_y'.\
                    format(randint_logmstar)] >= 8.6][cols_to_use]
            else:
                gals_df_mock = gal_group_df_two.loc[gal_group_df_two['{0}_y'.\
                    format(randint_logmstar)] >= 8.6][cols_to_use]

            gals_df_mock = gals_df_mock.rename(columns=\
                {'{0}_y'.format(randint_logmstar):'{0}'.format(randint_logmstar)})
        else:
            cols_to_use = ['halo_mvir', 'cs_flag', 'cz', \
                'stellar_mass', 'g_galtype', 'groupid']
            gals_df_mock = gal_group_df.loc[gal_group_df['stellar_mass'] >= 10**8.6][cols_to_use]
            gals_df_mock.stellar_mass = np.log10(gals_df_mock.stellar_mass)

        f_red_cen, f_red_sat = hybrid_quenching_model(theta, gals_df_mock, \
            'vishnu', randint_logmstar)
        gals_df_mock = assign_colour_label_mock(f_red_cen, f_red_sat, \
            gals_df_mock)
        # v_sim = 130**3
        v_sim = 890641.5172927063 #survey volume used in group_finder.py
        total_model, red_model, blue_model = measure_all_smf(gals_df_mock, v_sim 
        , False, randint_logmstar)     
        data_arr = []
        data_arr.append(phi_red_data)
        data_arr.append(phi_blue_data)
        model_arr = []
        model_arr.append(red_model[1])
        model_arr.append(blue_model[1])   
        err_arr = err

        data_arr, model_arr = np.array(data_arr), np.array(model_arr)
        chi2 = chi_squared(data_arr, model_arr, err_arr, corr_mat_inv)
        lnp = -chi2 / 2

        if math.isnan(lnp):
            raise ValueError
    except (ValueError, RuntimeWarning, UserWarning):
        lnp = -np.inf
        chi2 = np.inf

    return lnp, [chi2, randint_logmstar]

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

    phi_arr_total = []
    phi_arr_red = []
    phi_arr_blue = []
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

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            ## Using best-fit found for new ECO data using optimize_hybridqm_eco,py
            Mstar_q = 10.49 # Msun/h
            Mh_q = 14.03 # Msun/h
            mu = 0.69
            nu = 0.148



            theta = [Mstar_q, Mh_q, mu, nu]
            f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values

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
    phi_arr_total = np.array(phi_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)

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

    combined_df = pd.DataFrame({'phi_red_0':phi_red_0, 'phi_red_1':phi_red_1,\
        'phi_red_2':phi_red_2, 'phi_red_3':phi_red_3, 'phi_red_4':phi_red_4, \
        'phi_blue_0':phi_blue_0, 'phi_blue_1':phi_blue_1, 
        'phi_blue_2':phi_blue_2, 'phi_blue_3':phi_blue_3, 
        'phi_blue_4':phi_blue_4})

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, corr_mat_inv_colour

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
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'stellar_mass'
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

def args_parser():
    """
    Parsing arguments passed to script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('machine', type=str, \
        help='Options: mac/bender')
    parser.add_argument('survey', type=str, \
        help='Options: eco/resolvea/resolveb')
    parser.add_argument('mf_type', type=str, \
        help='Options: smf/bmf')
    parser.add_argument('nproc', type=int, nargs='?', 
        help='Number of processes')
    parser.add_argument('nwalkers', type=int, nargs='?', 
        help='Number of walkers')
    parser.add_argument('nsteps', type=int, nargs='?', help='Number of steps')
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function that calls all other functions
    
    Parameters
    ----------
    args: 
        Input arguments to the script

    """
    # global model_init
    global gal_group_df
    global survey
    global path_to_proc
    global mf_type
    global ver
    # global mocknum_queue

    rseed = 12
    np.random.seed(rseed)
    survey = args.survey
    machine = args.machine
    nproc = args.nproc
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    mf_type = args.mf_type
    use_behroozi_sample = False
    ver = 2.0
    
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_external = dict_of_paths['ext_dir']
    path_to_data = dict_of_paths['data_dir']
    path_to_int = dict_of_paths['int_dir']
    
    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    if survey == 'eco':
        catl_file = path_to_proc + "gal_group_eco_data.hdf5"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"
    
    if survey == 'eco':
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    elif survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    elif survey == 'resolveb':
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

    print('Reading catalog') #No Mstar cut
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour to data')
    catl = assign_colour_label_data(catl)

    print('Measuring SMF for data')
    total_data, red_data, blue_data = measure_all_smf(catl, volume, True)

    print('Measuring error in data from mocks')
    sigma, corr_mat_inv = get_err_data(survey, path_to_mocks)

    print('sigma: \n', sigma)
    print('inv corr mat: \n', corr_mat_inv)
    print('red phi data: \n', red_data[1])
    print('blue phi data: \n', blue_data[1])

    if use_behroozi_sample:
        print('Reading 100 vishnu group catalogs')

        gal_group_df = reading_catls(path_to_proc + "gal_group.hdf5") 
        col_idxs = [str(int(x)) for x in np.linspace(1,101,101)] 
        cols_to_keep_set_one = [] 
        for idx in range(len(col_idxs)): 
            idx+=1 
            cols_to_keep_set_one.append('g_galtype_{0}'.format(idx)) 
            cols_to_keep_set_one.append('groupid_{0}'.format(idx)) 
            cols_to_keep_set_one.append('{0}_y'.format(idx)) 
            if idx == 50:
                break
        cols_to_keep_set_one.append('cz') 
        cols_to_keep_set_one.append('halo_mvir')
        cols_to_keep_set_one.append('cs_flag')

        cols_to_keep_set_two = [] 
        for idx in range(len(col_idxs)): 
            idx+=51
            cols_to_keep_set_two.append('g_galtype_{0}'.format(idx)) 
            cols_to_keep_set_two.append('groupid_{0}'.format(idx)) 
            cols_to_keep_set_two.append('{0}_y'.format(idx)) 
            if idx == 101:
                break
        cols_to_keep_set_two.append('cz') 
        cols_to_keep_set_two.append('halo_mvir')
        cols_to_keep_set_two.append('cs_flag')

        global gal_group_df_one
        global gal_group_df_two

        gal_group_df_one = gal_group_df[cols_to_keep_set_one]
        gal_group_df_two = gal_group_df[cols_to_keep_set_two]
    else:
        ## Stellar mass in this is not in log whereas in the df in the if it is
        print('Reading single Vishnu group catalog')
        gal_group_df = reading_catls(path_to_proc + "gal_group_95perc.hdf5") 



    print('Running MCMC')
    sampler = mcmc(nproc, nwalkers, nsteps, red_data[1], blue_data[1], 
        sigma, corr_mat_inv)

# Main function
if __name__ == '__main__':
    args = args_parser()
    # pr = cProfile.Profile()
    # pr.enable()

    main(args)

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()

    # with open('profile_eco_2p_16w_5s.txt', 'w+') as f:
    #     f.write(s.getvalue())