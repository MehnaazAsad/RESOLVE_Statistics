"""
{This script compares and plots SMF and SMHM from data and simulation 
 given a choice of Behroozi10 SMHM parameter values}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np

__author__ = '{Mehnaaz Asad}'

def diff_smf(mstar_arr, volume, cvar_err, h1_bool, hdependence):
    """Calculates differential stellar mass function given stellar masses.""" 
    if not h1_bool and hdependence == 1:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 1.429)
    elif not h1_bool and hdependence == 2:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
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

    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def get_centrals(gals_df):
    """Isolates centrals and retrieves their stellar mass and halo mass"""
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

    for idx,value in enumerate(gals_df.C_S.values):
        if value == 1:
            cen_gals.append(gals_df.stellar_mass.values[idx])
            cen_halos.append(gals_df.halo_mvir.values[idx])

    cen_gals = np.log10(np.array(cen_gals))
    cen_halos = np.log10(np.array(cen_halos))
    return cen_gals,cen_halos

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

# Formatting for plots and animation
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=18)
rc('text', usetex=True)

halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
           'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 'logmh_s', 'fc',
           'grpmb', 'grpms', 'f_a', 'f_b']

# 2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv",
                             delimiter=",", header=0, usecols=columns)

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
           'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 'fc',
           'grpmb', 'grpms']

# 13878 galaxies
eco_buff = pd.read_csv(path_to_raw + "eco_all.csv",delimiter=",", header=0, \
    usecols=columns)

# 956 galaxies
resolve_A = resolve_live18.loc[(resolve_live18.f_a.values == 1) &
                               (resolve_live18.grpcz.values > 4500) &
                               (resolve_live18.grpcz.values < 7000) &
                               (resolve_live18.absrmag.values < -17.33)]

v_resolveA = 13172.384  # Survey volume without buffer [Mpc/h]^3
cvar_resolveA = 0.30

# 487
resolve_B = resolve_live18.loc[(resolve_live18.f_b.values == 1) &
                               (resolve_live18.grpcz.values > 4500) &
                               (resolve_live18.grpcz.values < 7000) &
                               (resolve_live18.absrmag.values < -17)]

v_resolveB = 4709.8373 # Survey volume without buffer [Mpc/h]^3
cvar_resolveB = 0.58

# 6456 galaxies                       
eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
    (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &\
        (eco_buff.logmstar.values >= 8.9)]

v_eco = 151829.26 # Survey volume without buffer [Mpc/h]^3
cvar_eco = 0.125

# SMFs
resa_m_stellar = resolve_A.logmstar.values
max_resolveA, phi_resolveA, err_tot_A, bins_A = diff_smf(resa_m_stellar,
                                                         v_resolveA,
                                                         cvar_resolveA, False,
                                                         2)

resb_m_stellar = resolve_B.logmstar.values
max_resolveB, phi_resolveB, err_tot_B, bins_B = diff_smf(resb_m_stellar,
                                                         v_resolveB,
                                                         cvar_resolveB, False,
                                                         2)

eco_nobuff_mstellar = eco_nobuff.logmstar.values
max_eco, phi_eco, err_eco, bins_eco = diff_smf(eco_nobuff_mstellar, v_eco, 
                                               cvar_eco, False, 1)
max_eco_, phi_eco_, err_eco_, bins_eco_ = diff_smf(eco_nobuff_mstellar, v_eco, 
                                               cvar_eco, False, 2)

halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
z_model = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_model,
                                    prim_haloprop_key='halo_macc')

# Populate using default b10 parameters                                   
model.populate_mock(halocat)
sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
gals = model.mock.galaxy_table[sample_mask]
b10_gals_df = gals.to_pandas()

# Populate using best fit eco parameters
v_sim = 130**3
theta = [12.266,10.66,0.4,0.57,0.37]
bf_gals_df = populate_mock(theta)
mstellar_mock = bf_gals_df.stellar_mass.values  # Read stellar masses
max_model, phi_model, err_tot_model, bins_model =\
    diff_smf(mstellar_mock, v_sim, 0, True, 1)

# Get central stellar mass and halo mass for both cases above to plot SMHM
cen_gals_bf,cen_halos_bf = get_centrals(bf_gals_df)
cen_gals_b10,cen_halos_b10 = get_centrals(b10_gals_df)

# Get central stellar mass and halo mass from eco to plot SMHM
cen_gals_eco_h1 = []
cen_halos_eco_h1 = []
for idx,val in enumerate(eco_nobuff.fc.values):
    if val == 1:
        stellar_mass_h07 = eco_nobuff.logmstar.values[idx]
        stellar_mass_h1 = np.log10((10**stellar_mass_h07) / 1.429)
        halo_mass_h07 = eco_nobuff.logmh_s.values[idx]
        halo_mass_h1 = np.log10((10**halo_mass_h07) / 1.429)
        cen_gals_eco_h1.append(stellar_mass_h1)
        cen_halos_eco_h1.append(halo_mass_h1)

cen_gals_eco_h2 = np.array(cen_gals_eco_h1)
cen_halos_eco_h2 = np.array(cen_halos_eco_h1)

cen_gals_eco_h2 = []
cen_halos_eco_h2 = []
for idx,val in enumerate(eco_nobuff.fc.values):
    if val == 1:
        stellar_mass_h07 = eco_nobuff.logmstar.values[idx]
        stellar_mass_h1 = np.log10((10**stellar_mass_h07) / 2.041)
        halo_mass_h07 = eco_nobuff.logmh_s.values[idx]
        halo_mass_h1 = np.log10((10**halo_mass_h07) / 2.041)
        cen_gals_eco_h2.append(stellar_mass_h1)
        cen_halos_eco_h2.append(halo_mass_h1)

cen_gals_eco_h2 = np.array(cen_gals_eco_h2)
cen_halos_eco_h2 = np.array(cen_halos_eco_h2)

# Bin data for all 3 cases: b10, best fit and data
x_bf,y_bf,y_std_bf,y_std_err_bf = Stats_one_arr(cen_halos_bf,\
    cen_gals_bf,base=0.4,bin_statval='center')
x_b10,y_b10,y_std_b10,y_std_err_b10 = Stats_one_arr(cen_halos_b10,\
    cen_gals_b10,base=0.4,bin_statval='center')
x_eco_h1,y_eco_h1,y_std_eco_h1,y_std_err_eco_h1 = Stats_one_arr(cen_halos_eco_h1,\
    cen_gals_eco_h1,base=0.4,bin_statval='center')
x_eco_h2,y_eco_h2,y_std_eco_h2,y_std_err_eco_h2 = Stats_one_arr(cen_halos_eco_h2,\
    cen_gals_eco_h2,base=0.4,bin_statval='center')

y_std_err_eco_h1 = np.sqrt(y_std_err_eco_h1**2 + cvar_eco**2)
y_std_err_eco_h2 = np.sqrt(y_std_err_eco_h2**2 + cvar_eco**2)
# Plot SMF comparisons between best fit and data
fig1 = plt.figure(figsize=(10,10))
plt.errorbar(max_eco,phi_eco,yerr=err_eco,color='#921063',fmt='-s',ecolor='#921063',markersize=4,capsize=5,capthick=0.5,label=r'ECO data $h^{-1}$\ dependence')
plt.errorbar(max_eco_,phi_eco_,yerr=err_eco_,color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,capthick=0.5,label=r'ECO data $h^{-2}$\ dependence')
# plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,capthick=0.5,label='RESOLVE B')
# plt.errorbar(max_resolveA,phi_resolveA,yerr=err_tot_A,color='#442b88',fmt='--s',ecolor='#442b88',markersize=4,capsize=5,capthick=0.5,label='RESOLVE A')
plt.errorbar(max_model,phi_model,yerr=err_tot_model,color='k',fmt='-s',ecolor='k',markersize=4,capsize=5,capthick=0.5,label='ECO best fit')
plt.yscale('log')
plt.ylim(10**-5,10**-1)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
plt.show()

# Plot SMHM comparisons between best fit, behroozi10 and data
fig2 = plt.figure(figsize=(10,10))
plt.errorbar(x_bf,y_bf,yerr=y_std_err_bf,color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,capthick=0.5,label='Best fit')
plt.errorbar(x_b10,y_b10,yerr=y_std_err_b10,color='k',fmt='--s',ecolor='k',markersize=4,capsize=5,capthick=0.5,label='Behroozi10')
plt.errorbar(x_eco_h1,y_eco_h1,yerr=y_std_err_eco_h1,color='r',fmt='-s',ecolor='r',markersize=4,capsize=5,capthick=0.5,label=r'ECO data $h^{-1}$\ dependence')
plt.errorbar(x_eco_h2,y_eco_h2,yerr=y_std_err_eco_h2,color='r',fmt='--s',ecolor='r',markersize=4,capsize=5,capthick=0.5,label=r'ECO data $h^{-2}$\ dependence')
plt.ylim(8.9,)
plt.xlim(10.8,)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
plt.savefig('/Users/asadm2/Desktop/smhm_smass_withcvarerr.png')
plt.close()
