"""
{This script carries out forward modeling to parametrize the SMHM and BMHM for 
 ECO, RESOLVE-A and RESOLVE-B}
"""

# Built-in/Generic Imports
import argparse
import warnings
import time
import os

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from multiprocessing import Pool
import pandas as pd
import numpy as np
import emcee 
import math

__author__ = '[Mehnaaz Asad]'

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
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

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
                    (resolve_live18.absrmag.values <= -17.33) & 
                    (resolve_live18.logmstar.values >= 8.9)]
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
                    (resolve_live18.absrmag.values <= -17) & 
                    (resolve_live18.logmstar.values >= 8.7)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            # cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl, volume, z_median

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

def diff_smf(mstar_arr, volume, h1_bool):
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
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        # print(box)
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.\
                format(mock_name, num)
            mock_pd = reading_catls(filename) 

            if mf_type == 'smf':
                # Using the same survey definition as data
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & 
                    (mock_pd.cz.values <= max_cz) & 
                    (mock_pd.M_r.values <= mag_limit) &
                    (mock_pd.logmstar.values >= mstar_limit)]

                logmstar_arr = mock_pd.logmstar.values 
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                # Using the same survey definition as data - *no mstar cut*
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & 
                    (mock_pd.cz.values <= max_cz) & 
                    (mock_pd.M_r.values <= mag_limit)]

                logmstar_arr = mock_pd.logmstar.values 
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_arr_total.append(phi_total)

    phi_arr_total = np.array(phi_arr_total)
    # np.std(phi_arr_total, axis=0)
    # Covariance matrix
    # A variable here is a bin so each row is a bin and each column is one 
    # observation of all bins.
    cov_mat = np.cov(phi_arr_total, rowvar=False) # default norm is N-1
    stddev = np.sqrt(cov_mat.diagonal())
    # Correlation matrix
    corr_mat = cov_mat / np.outer(stddev , stddev)
    # Inverse of correlation matrix
    corr_mat_inv = np.linalg.inv(corr_mat)
    return stddev, corr_mat_inv

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

def mcmc(nproc, nwalkers, nsteps, phi, err, inv_corr_mat):
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
    behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]
    bmf_resume = [12.03949177, 10.65053122, 0.50320246, 0.26588596, 0.22353107]
    ndim = 5
    p0 = bmf_resume + 0.1*np.random.rand(ndim*nwalkers).\
        reshape((nwalkers, ndim))

    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
            args=(phi, err, inv_corr_mat), pool=pool)
        start = time.time()
        for i,result in enumerate(sampler.sample(p0, iterations=nsteps, 
            storechain=False)):
            position = result[0]
            chi2 = np.array(result[3])
            # print(chi2)
            print("Iteration number {0} of {1}".format(i+1,nsteps))
            chain_fname = open("mcmc_{0}_raw_bmf986.txt".format(survey), "a")
            chi2_fname = open("{0}_chi2_bmf986.txt".format(survey), "a")
            for k in range(position.shape[0]):
                chain_fname.write(str(position[k]).strip("[]"))
                chain_fname.write("\n")
            chain_fname.write("# New slice\n")
            for k in range(chi2.shape[0]):
                chi2_fname.write(str(chi2[k]).strip("[]"))
                chi2_fname.write("\n")
            chain_fname.close()
            chi2_fname.close()
        # sampler.run_mcmc(p0, nsteps)
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
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

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

    # return chi_squared
    first_term = ((data - model) / (err_data)).reshape(1,len(data))
    third_term = np.transpose(first_term)
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)
    return chi_squared[0][0]

def lnprob(theta, phi, err_tot, inv_corr_mat):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err_tot: array
        Array of error values of mass function

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2
    warnings.simplefilter("error", (UserWarning, RuntimeWarning))
    try:
        gals_df = populate_mock(theta, model_init)
        v_sim = 130**3
        mstellar_mock = gals_df.stellar_mass.values 
        if mf_type == 'smf':
            max_model, phi_model, err_tot_model, bins_model, counts_model = \
                diff_smf(mstellar_mock, v_sim, True)
        elif mf_type == 'bmf':
            max_model, phi_model, err_tot_model, bins_model, counts_model = \
                diff_bmf(mstellar_mock, v_sim, True)           
        chi2 = chi_squared(phi, phi_model, err_tot, inv_corr_mat)
        lnp = -chi2 / 2
        if math.isnan(lnp):
            raise ValueError
    except (ValueError, RuntimeWarning, UserWarning):
        lnp = -np.inf
        chi2 = np.inf

    return lnp, chi2

def write_to_files(sampler):
    """
    Writes chain information to files

    Parameters
    ----------
    sampler: multidimensional array
        Result of running MCMC

    Returns
    ---------
    Nothing; chain information written to files
        
    """
    print("\t - Writing raw chain")
    data = sampler.chain
    fname = path_to_proc + 'mcmc_{0}_raw.txt'.format(survey)
    with open(fname, 'w') as outfile:
        # outfile.write('# Array shape: {0}\n'.format(sampler.chain.shape))
        for data_slice in data:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')

    print("\t - Writing flattened chain")
    chain = sampler.flatchain
    fname = path_to_proc + 'mcmc_{0}.dat'.format(survey)
    with open(fname, 'w') as chain_fname:
        for idx in range(len(chain)):
            chain_fname.write(str(chain[idx]).strip("[]"))
            chain_fname.write("\n")

    print("\t - Writing chi squared")
    blobs = np.ndarray.flatten(np.array(sampler.blobs))
    fname = path_to_proc + '{0}_chi2.txt'.format(survey)
    with open(fname, 'w') as outfile:
        for value in blobs:
            outfile.write(str(value))
            outfile.write("\n")
    
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
        help='Number of processes', default=20)
    parser.add_argument('nwalkers', type=int, nargs='?', 
        help='Number of walkers', default=250)
    parser.add_argument('nsteps', type=int, nargs='?', help='Number of steps',
        default=1000)
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
    global model_init
    global survey
    global path_to_proc
    global mf_type
    rseed = 12
    np.random.seed(rseed)

    survey = args.survey
    machine = args.machine
    nproc = args.nproc
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    mf_type = args.mf_type
    
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
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    if survey == 'eco':
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
    elif survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    elif survey == 'resolveb':
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

    print('Reading catalog')
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Retrieving stellar mass from catalog')
    stellar_mass_arr = catl.logmstar.values
    if mf_type == 'smf':
        maxis_data, phi_data, err_data, bins_data, counts_data = \
            diff_smf(stellar_mass_arr, volume, False)
    elif mf_type == 'bmf':
        gas_mass_arr = catl.logmgas.values
        bary_mass_arr = calc_bary(stellar_mass_arr, gas_mass_arr)
        maxis_data, phi_data, err_data, bins_data, counts_data = \
            diff_bmf(bary_mass_arr, volume, False)
    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    print('Measuring error in data from mocks')
    err_data, inv_corr_mat = get_err_data(survey, path_to_mocks)
    print(err_data, inv_corr_mat)
    print('Running MCMC')
    sampler = mcmc(nproc, nwalkers, nsteps, phi_data, err_data, inv_corr_mat)


# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args)
