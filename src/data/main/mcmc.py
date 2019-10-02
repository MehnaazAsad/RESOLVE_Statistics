"""
{This script carries out an MCMC analysis to parametrize the SMHM for ECO, 
 RESOLVE-A and RESOLVE-B}
"""

# Built-in/Generic Imports
import time

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from multiprocessing import Pool
import pandas as pd
import numpy as np
import argparse
import warnings
import emcee 
import math

__author__ = '[Mehnaaz Asad]'

def read_catl(path_to_file, survey):
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
        cvar = 0.125
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

def diff_smf(mstar_arr, volume, h1_bool):
    """
    Calculates differential stellar mass function in units of h=1.0

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

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
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        # bins = np.linspace(bin_min, bin_max, 7)
        bins = np.linspace(bin_min, bin_max, 8)
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
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 6)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
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
    # print(phi)
    return maxis, phi, err_tot, bins, counts

def jackknife(catl, volume):
    """
    Jackknife ECO survey to get data in error and correlation matrix for 
    chi-squared calculation

    Parameters
    ----------
    catl: Pandas DataFrame
        Survey catalog

    Returns
    ---------
    stddev_jk: numpy array
        Array of sigmas
    corr_mat_inv: numpy matrix
        Inverse of correlation matrix
    """

    ra = catl.radeg.values # degrees
    dec = catl.dedeg.values # degrees

    sin_dec_all = np.rad2deg(np.sin(np.deg2rad(dec))) # degrees

    sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 8)
    ra_arr = np.linspace(ra.min(), ra.max(), 8)

    grid_id_arr = []
    gal_id_arr = []
    grid_id = 1
    max_bin_id = len(sin_dec_arr)-2 # left edge of max bin
    for dec_idx in range(len(sin_dec_arr)):
        for ra_idx in range(len(ra_arr)):
            try:
                if dec_idx == max_bin_id and ra_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
                elif dec_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values < ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
                elif ra_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
                else:                
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values < ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
                # Append dec and sin  
                for gal_id in catl_subset.name.values:
                    gal_id_arr.append(gal_id)
                for grid_id in [grid_id] * len(catl_subset):
                    grid_id_arr.append(grid_id)
                grid_id += 1
            except IndexError:
                break

    gal_grid_id_data = {'grid_id': grid_id_arr, 'name': gal_id_arr}
    df_gal_grid = pd.DataFrame(data=gal_grid_id_data)

    catl = catl.join(df_gal_grid.set_index('name'), on='name')
    catl = catl.reset_index(drop=True)

    # Loop over all sub grids, remove one and measure global smf
    jackknife_phi_arr = []
    for grid_id in range(len(np.unique(catl.grid_id.values))):
        grid_id += 1
        catl_subset = catl.loc[catl.grid_id.values != grid_id]  
        logmstar = catl_subset.logmstar.values
        logmgas = catl_subset.logmgas.values
        logmbary = calc_bary(logmstar, logmgas)
        if mf_type == 'smf':
            maxis, phi, err, bins, counts = diff_smf(logmstar, volume, False)
        elif mf_type == 'bmf':
            maxis, phi, err, bins, counts = diff_bmf(logmbary, volume, False)
        jackknife_phi_arr.append(phi)

    jackknife_phi_arr = np.array(jackknife_phi_arr)

    N = len(jackknife_phi_arr)

    # Covariance matrix
    cov_mat = np.cov(jackknife_phi_arr.T, bias=True)*(N-1)
    stddev_jk = np.sqrt(cov_mat.diagonal())

    # Correlation matrix
    corr_mat = cov_mat / np.outer(stddev_jk , stddev_jk)
    # Inverse of correlation matrix
    corr_mat_inv = np.linalg.inv(corr_mat)
    # print(corr_mat)
    return stddev_jk, corr_mat_inv

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

def mcmc(nproc, nwalkers, nsteps, phi, err, corr_mat_inv):
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
    behroozi10_param_vals = [12.46174527, 10.61989256 , 0.53968546  ,\
        0.85463982,  0.10656538]#[12.35,10.72,0.44,0.57,0.15]
    ndim = 5
    p0 = behroozi10_param_vals + 0.1*np.random.rand(ndim*nwalkers).\
        reshape((nwalkers, ndim))

    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
            args=(phi, err, corr_mat_inv), pool=pool)
        start = time.time()
        for i,result in enumerate(sampler.sample(p0, iterations=nsteps, storechain=False)):
            # position = result[0]
            # chi2 = np.array(result[3])
            print("Iteration number {0} of {1}".format(i+1,nsteps))
            # chain_fname = open("mcmc_{0}_raw.txt".format(survey), "a")
            # chi2_fname = open("{0}_chi2.txt".format(survey), "a")
            # for k in range(position.shape[0]):
            #     chain_fname.write(str(position[k]).strip("[]"))
            #     chain_fname.write("\n")
            # chain_fname.write("# New slice\n")
            # for k in range(chi2.shape[0]):
            #     chi2_fname.write(str(chi2[k]).strip("[]"))
            #     chi2_fname.write("\n")
            # chain_fname.close()
            # chi2_fname.close()
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
        limit = np.round(np.log10((10**8.9) / 2.041), 1)
        sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    elif survey == 'resolveb':
        limit = np.round(np.log10((10**8.7) / 2.041), 1)
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
    first_term = ((data - model) / (err_data)).reshape(1,len(data))
    print("phi_model: ", model)
    third_term = np.transpose(first_term)
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)
    print("chi2: ", chi_squared)
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
        max_model, phi_model, err_tot_model, bins_model, counts_model = \
            diff_smf(mstellar_mock, v_sim, True)
        chi2 = chi_squared(phi, phi_model, err_tot, inv_corr_mat)
        lnp = -chi2 / 2

    except ValueError:
        print("in except clause")
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

    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    if survey == 'eco':
        catl_file = path_to_raw + "eco_all.csv"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    print('Reading catalog')
    catl, volume, cvar, z_median = read_catl(catl_file, survey)

    print('Retrieving stellar mass from catalog')
    stellar_mass_arr = catl.logmstar.values
    gas_mass_arr = catl.logmgas.values
    bary_mass_arr = calc_bary(stellar_mass_arr, gas_mass_arr)
    if mf_type == 'smf':
        maxis_data, phi_data, err_data, bins_data, counts_data = \
            diff_smf(stellar_mass_arr, volume, False)
    elif mf_type == 'bmf':
        maxis_data, phi_data, err_data, bins_data, counts_data = \
            diff_bmf(bary_mass_arr, volume, False)
    print('Initial population of halo catalog')
    model_init = halocat_init(halo_catalog, z_median)

    if survey == 'eco':
        print('Jackknife ECO survey')
        err_data, inv_corr_mat = jackknife(catl, volume)

    print('Running MCMC')
    sampler = mcmc(nproc, nwalkers, nsteps, phi_data, err_data, inv_corr_mat)
    print('Writing to files:')
    # write_to_files(sampler)

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args)
