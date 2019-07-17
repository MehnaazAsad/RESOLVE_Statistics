"""
{This script carries out an MCMC analysis to parametrize the RESOLVE-B SMHM}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math
import time

__author__ = '{Mehnaaz Asad}'

def diff_smf(mstar_arr, volume, cvar_err, h1_bool):
    """Calculates differential stellar mass function given stellar masses."""
    
    if not h1_bool:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 1.429)
    else:
        logmstar_arr = np.log10(mstar_arr)
    bins = np.linspace(8.9, 11.8, 12)
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

def populate_mock(theta):
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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=10)
rc('text', usetex=True)

halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
# halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
#                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
chi2_df = pd.read_csv(path_to_proc + 'resolveb_chi2.txt',header=None,names=['chisquared'])
test_reshape = chi2_df.chisquared.values.reshape((1000,250))
chi2 = np.ndarray.flatten(np.array(test_reshape),'F')

chain_fname = chain_fname = path_to_proc + 'mcmc_resolveb.dat'
emcee_table = pd.read_csv(chain_fname,names=['mhalo_c','mstellar_c',\
                                             'lowmass_slope','highmass_slope',\
                                             'scatter'],sep='\s+',dtype=np.float64)

# Cases where last parameter was a NaN and its value was being written to 
# the first element of the next line followed by 4 NaNs for the other parameters
for idx,row in enumerate(emcee_table.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = emcee_table.values[idx+1][0]
          row[4] = scatter_val

emcee_table = emcee_table.dropna().reset_index(drop=True)

emcee_table['chi2'] = chi2
emcee_table = emcee_table.sort_values('chi2')
slice_end = int(0.68*len(emcee_table))
emcee_table_68_percent = emcee_table[:slice_end]
emcee_table_68_percent = emcee_table_68_percent[:10]

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
           'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 'fc',
           'grpmb', 'grpms']

# 13878 galaxies
eco_buff = pd.read_csv(path_to_raw + "eco_all.csv",delimiter=",", header=0, \
    usecols=columns)

# 6456 galaxies                       
eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
    (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &\
        (eco_buff.logmstar.values >= 8.9)]

v_eco = 151829.26 # Survey volume without buffer [Mpc/h]^3
cvar_eco = 0.125

eco_nobuff_mstellar = eco_nobuff.logmstar.values
max_eco, phi_eco, err_eco, bins_eco = diff_smf(eco_nobuff_mstellar, v_eco, 
                                               cvar_eco, False)

halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
z_model = np.median(eco_nobuff.grpcz.values) / (3 * 10**5)
model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_model,
                                    prim_haloprop_key='halo_macc')
model.populate_mock(halocat,seed=5)

v_sim = 130**3

def mp_func(a_list):
    max_model_arr = []
    phi_model_arr = []
    err_tot_model_arr = []
    for theta in a_list:  
        gals_df = populate_mock(theta)
        mstellar_mock = gals_df.stellar_mass.values  # Read stellar masses
        max_model, phi_model, err_tot_model, bins_model =\
            diff_smf(mstellar_mock, v_sim, 0, True)
        max_model_arr.append(max_model)
        phi_model_arr.append(phi_model)
        err_tot_model_arr.append(err_tot_model)
    return [max_model_arr,phi_model_arr,err_tot_model_arr]

start = time.time()
chunks = np.array([emcee_table_68_percent.iloc[:,:5].values[i::5] for i in range(5)])
pool = Pool(processes=1)
result = pool.map(mp_func, chunks)
end = time.time()
multi_time = end - start
print("Multiprocessing took {0:.1f} seconds".format(multi_time))

best_fit_eco = [12.261,10.662,0.401,0.58,0.37]
gals_df = populate_mock(best_fit_eco)
mstellar_mock = gals_df.stellar_mass.values  # Read stellar masses
max_model_bf, phi_model_bf, err_tot_model_bf, bins_model_bf =\
    diff_smf(mstellar_mock, v_sim, 0, True)


fig1 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,color='k',fmt='--s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='Resolve B',zorder=10)
for idx in range(len(result[0][0])):
    plt.errorbar(result[0][0][idx],result[0][1][idx],\
        yerr=result[0][2][idx],color='lightgray',fmt='-s',\
            ecolor='lightgray',markersize=4,capsize=5,capthick=0.5,\
                label='model',alpha=0.1,zorder=0)
for idx in range(len(result[1][0])):
    plt.errorbar(result[1][0][idx],result[1][1][idx],\
        yerr=result[1][2][idx],color='lightgray',fmt='-s',\
            ecolor='lightgray',markersize=4,capsize=5,capthick=0.5,\
                label='model',alpha=0.1,zorder=1)
for idx in range(len(result[2][0])):
    plt.errorbar(result[2][0][idx],result[2][1][idx],\
        yerr=result[2][2][idx],color='lightgray',fmt='-s',\
            ecolor='lightgray',markersize=4,capsize=5,capthick=0.5,\
                label='model',alpha=0.1,zorder=2)
for idx in range(len(result[3][0])):
    plt.errorbar(result[3][0][idx],result[3][1][idx],\
        yerr=result[3][2][idx],color='lightgray',fmt='-s',\
            ecolor='lightgray',markersize=4,capsize=5,capthick=0.5,\
                label='model',alpha=0.1,zorder=3)
for idx in range(len(result[4][0])):
    plt.errorbar(result[4][0][idx],result[4][1][idx],\
        yerr=result[4][2][idx],color='lightgray',fmt='-s',\
            ecolor='lightgray',markersize=4,capsize=5,capthick=0.5,\
                label='model',alpha=0.1,zorder=4)
plt.errorbar(max_model_bf,phi_model_bf,yerr=err_tot_model_bf,color='r',\
    fmt='--s',ecolor='r',markersize=4,capsize=5,capthick=0.5,\
        label='best fit',zorder=15)
plt.yscale('log')
plt.ylim(10**-5,10**-1)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-1} \right]$')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
plt.show()



 
