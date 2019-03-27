"""
{This script carries out an MCMC analysis using fake data to test that 
 the known 5 parameter values are recovered}
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
import emcee
import math

__author__ = '{Mehnaaz Asad}'

def diff_smf(mstar_arr, volume, cvar_err, h1_bool):
    """Calculates differential stellar mass function given stellar masses."""
    
    if not h1_bool:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 1.429)
    else:
        logmstar_arr = np.log10(mstar_arr)
    bins = np.linspace(8.9, 11.5, 51)
    # Unnormalized histogram and bin edges
    phi, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(phi) / (volume * dm)
    err_cvar = cvar_err / (volume * dm)
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    phi = phi / (volume * dm)  # not a log quantity
    return maxis, phi, err_tot, bins

def populate_mock(theta, model):
    """Populates halo catalog with galaxies given SMHM parameters and model."""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['u’scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def chi_squared(data_y, model_y, err_data):
    """Calculates chi squared."""
    chi_squared = (data_y - model_y)**2 / (err_data**2)

    return np.sum(chi_squared)

def lnprob(theta, phi, err_tot):
    """Calculates log probability for emcee."""
    if theta[0] < 0:
        return -np.inf
    if theta[1] < 0:
        return -np.inf
    if theta[4] < 0:
        return -np.inf
    try:
        gals_df = populate_mock(theta, model)
        v_sim = 130**3
        mstellar_mock = gals_df.stellar_mass.values  # Read stellar masses
        max_model, phi_model, err_tot_model, bins_model =\
            diff_smf(mstellar_mock, v_sim, 0, True)
        chi2 = chi_squared(phi, phi_model, err_tot)
        lnp = -chi2 / 2
    except Exception:
        lnp = -np.inf
    return lnp

# ensure reproducibility
rseed = 12
np.random.seed(rseed)

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
# halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
#                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
chain_fname = path_to_proc + 'emcee_SMFRB_fake.dat'

## Generating fake data from simulation
halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
a = 0.98169
z_model = (1/a) - 1
model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_model,
                                    prim_haloprop_key='halo_macc')
model.populate_mock(halocat)

v_sim = 130**3

phi_arr = np.zeros((10,50))
for i in range(10):
    model.mock.populate()
    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()
    mstellar = gals_df.stellar_mass.values 
    maxis, phi, err_tot, bins = diff_smf(mstellar,v_sim,0,True)
    phi_arr[i] = phi

fake_err = np.std(phi_arr,axis=0,dtype=np.float64)
fake_x = maxis
fake_y = phi

behroozi10_param_vals = [12.25,10.62,0.54,0.67,0.3]

nwalkers = 250
ndim = 5
p0 = behroozi10_param_vals + 0.1*np.random.rand(ndim*nwalkers).\
    reshape((nwalkers,ndim))
with Pool(processes=20) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(fake_y, \
            fake_err))
    start = time.time()
    sampler.run_mcmc(p0, 500)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

print("Writing chain to file")
chain = sampler.flatchain
f = open(chain_fname, "w")
for idx in range(len(chain)):
    f.write(str(chain[idx]).strip("[]"))
    f.write("\n")
f.close()

# nsteps = 10
# for i,result in enumerate(sampler.sample(p0, iterations=nsteps, storechain=True)):
#     # position = result[0]
#     print("Iteration number {0} of {1}".format(i+1,nsteps))
#     # f = open(chain_fname, "a")
#     # for k in range(position.shape[0]):
#     #     f.write(str(position[k]).strip("[]"))
#     #     f.write("\n")
#     # f.close()

#behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]