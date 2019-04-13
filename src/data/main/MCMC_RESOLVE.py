"""
{This script carries out an MCMC analysis to parametrize the RESOLVE-B SMHM}
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
    bins = np.linspace(8.9, 11.5, 12)
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
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.9
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def chi_squared(data_y, model_y, err_data):
    """Calculates chi squared."""
    chi_squared = (data_y - model_y)**2 / (err_data**2)

    return np.sum(chi_squared)

def lnprob(theta, phi_resolveb, err_tot_b):
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
        chi2 = chi_squared(phi_resolveb, phi_model, err_tot_b)
        lnp = -chi2 / 2
    except Exception:
        lnp = -np.inf
    return lnp

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
               'vishnu/rockstar/vishnu_rockstar_test.hdf5'
# halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
chain_fname = path_to_proc + 'emcee_resolveA_mp_corrscatter.dat'

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
           'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 'logmh_s', 'fc',
           'grpmb', 'grpms', 'f_a', 'f_b']

# 2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv",
                             delimiter=",", header=0, usecols=columns)

# 956 - cz, 634 - grpcz
resolve_A = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
    (resolve_live18.grpcz.values > 4500) & \
        (resolve_live18.grpcz.values < 7000) & \
            (resolve_live18.absrmag.values < -17.33) & \
                (resolve_live18.logmstar.values >= 8.9)]


v_resolveA = 13172.384  # Survey volume without buffer [Mpc/h]^3
cvar_resolveA = 0.30

# 487 - cz, 369 - grpcz
resolve_B = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
    (resolve_live18.grpcz.values > 4500) & \
        (resolve_live18.grpcz.values < 7000) & \
            (resolve_live18.absrmag.values < -17) & \
                (resolve_live18.logmstar.values >= 8.7)]

v_resolveB = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
cvar_resolveB = 0.58

# SMFs
resa_m_stellar = resolve_A.logmstar.values
max_resolveA, phi_resolveA, err_tot_A, bins_A = diff_smf(resa_m_stellar,
                                                         v_resolveA,
                                                         cvar_resolveA, False)

resb_m_stellar = resolve_B.logmstar.values
max_resolveB, phi_resolveB, err_tot_B, bins_B = diff_smf(resb_m_stellar,
                                                         v_resolveB,
                                                         cvar_resolveB, False)

halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
z_model = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_model,
                                    prim_haloprop_key='halo_macc')
model.populate_mock(halocat)

# ensure reproducibility
rseed = 12
np.random.seed(rseed)

behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]
nwalkers = 250
ndim = 5
p0 = behroozi10_param_vals + 0.1*np.random.rand(ndim*nwalkers).\
    reshape((nwalkers,ndim)) 

with Pool(20) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(phi_resolveA,\
          err_tot_A),pool=pool)
    start = time.time()
    sampler.run_mcmc(p0, 1000)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

print("Writing raw chain to file")
data = sampler.chain
with open(path_to_proc + 'resolveA_mcmc_raw_corrscatter.txt', 'w') as outfile:
    # outfile.write('# Array shape: {0}\n'.format(sampler.chain.shape))
    for data_slice in data:
        np.savetxt(outfile, data_slice, fmt='%-7.2f')
        outfile.write('# New slice\n')

print("Writing flattened chain to file")
chain = sampler.flatchain
f = open(chain_fname, "w")
for idx in range(len(chain)):
    f.write(str(chain[idx]).strip("[]"))
    f.write("\n")
f.close()
'''
nsteps = 5
# f = open(chain_fname, "w")
# f.close()

for i,result in enumerate(sampler.sample(p0, iterations=nsteps, storechain=False)):
    # position = result[0]
    print("Iteration number {0} of {1}".format(i+1,nsteps))
    # f = open(chain_fname, "a")
    # for k in range(position.shape[0]):
    #     f.write(str(position[k]).strip("[]"))
    #     f.write("\n")
    # f.close()
'''
'''
resolve_A_grp = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
    (resolve_live18.grpcz.values > 4500) & \
        (resolve_live18.grpcz.values < 7000) & \
            (resolve_live18.absrmag.values < -17.33) & \
                (resolve_live18.logmstar.values >= 8.9)]

resolve_B_grp = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
    (resolve_live18.grpcz.values > 4500) & \
        (resolve_live18.grpcz.values < 7000) & \
            (resolve_live18.absrmag.values < -17) & \
                (resolve_live18.logmstar.values >= 8.7)]

resa_m_stellar_grp = resolve_A_grp.logmstar.values
max_resolveA_grp, phi_resolveA_grp, err_tot_A_grp, bins_A_grp = diff_smf(resa_m_stellar_grp,
                                                         v_resolveA,
                                                         cvar_resolveA, False)
resb_m_stellar_grp = resolve_B_grp.logmstar.values
max_resolveB_grp, phi_resolveB_grp, err_tot_B_grp, bins_B_grp = diff_smf(resb_m_stellar_grp,
                                                         v_resolveB,
                                                         cvar_resolveB, False)
fig1 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveA_grp,phi_resolveA_grp,yerr=err_tot_A_grp,\
    color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
        capthick=0.5,label=r'group cz')
plt.errorbar(max_resolveA,phi_resolveA,yerr=err_tot_A,\
    color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
        capthick=0.5,label=r'galaxy cz')

plt.xlabel(r'\boldmath$\log\ M_{\star} \left[M_\odot \right]$',size=15)
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$',size=15)
plt.yscale('log')
plt.title(r'\boldmath RESOLVE-A',size=15)
plt.legend(loc='best',prop={'size': 10})
plt.show()

fig2 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveB_grp,phi_resolveB_grp,yerr=err_tot_B_grp,\
    color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
        capthick=0.5,label=r'group cz')
plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,\
    color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
        capthick=0.5,label=r'galaxy cz')

plt.xlabel(r'\boldmath$\log\ M_{\star} \left[M_\odot \right]$',size=15)
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$',size=15)
plt.yscale('log')
plt.title(r'\boldmath RESOLVE-B',size=15)
plt.legend(loc='best',prop={'size': 10})
plt.show()
'''