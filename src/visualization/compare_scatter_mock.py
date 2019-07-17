"""
{This script makes mocks for a given set of parameter values and plots the 
 distribution of galaxies' stellar mass in a narrow halo mass range for low and
 high scatter parameter values}
"""

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math

__author__ = '{Mehnaaz Asad}'

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
    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()
    return gals_df

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

chain_fname = path_to_proc + 'emcee_eco_mp_r2.dat'
halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
           'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 'fc',
           'grpmb', 'grpms']

# 13878 galaxies
eco_buff = pd.read_csv(path_to_raw + "eco_all.csv",delimiter=",", header=0, \
    usecols=columns)

# 6456 galaxies                       
eco_nobuff = eco_buff.loc[(eco_buff.cz.values >= 3000) & \
    (eco_buff.cz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &\
        (eco_buff.logmstar.values >= 8.9)]

emcee_table = pd.read_csv(chain_fname,names=['mhalo_c','mstellar_c',\
                                             'lowmass_slope','highmass_slope',\
                                             'scatter'],sep='\s+',dtype=np.float64)

# Cases where last parameter was a NaN and its value was being written to 
# the first element of the next line followed by 4 NaNs for the other parameters
for idx,row in enumerate(emcee_table.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = emcee_table.values[idx+1][0]
          row[4] = scatter_val
 
emcee_table_rev = emcee_table.dropna(axis='index',how='any')
emcee_table_rev = emcee_table_rev.reset_index(drop=True)

## Generating fake data from simulation
halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
z_model = np.median(eco_nobuff.grpcz.values) / (3 * 10**5)
model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_model,
                                    prim_haloprop_key='halo_macc')
model.populate_mock(halocat)
#Testing with values that are normal (0.15), >10 , >100 and >400
theta_arr = [emcee_table_rev.values[0],emcee_table_rev.values[384],\
    emcee_table_rev.values[393],emcee_table_rev.values[18866]]


smass_arr = [[],[],[],[]]
ngals_arr = np.zeros(len(theta_arr))
num_nans_arr = np.zeros(len(theta_arr))
for idx,theta in enumerate(theta_arr):
    gals_df = populate_mock(theta,model)
    smass = gals_df.stellar_mass.loc[(np.log10(gals_df.halo_macc.values) > 12) & \
        (np.log10(gals_df.halo_macc.values) < 12.1)]
    ngals_arr[idx] = len(smass)
    smass = smass.replace([np.inf, -np.inf], np.nan)
    num_nans = smass.isna().sum()
    smass = smass.dropna(axis='index',how='any')
    log_smass = np.log10(smass)
    smass_arr[idx] = log_smass.values
    num_nans_arr[idx] = num_nans
smass_arr = np.array(smass_arr)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10,10))

for idx,ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.hist(smass_arr[idx],histtype='step')
    ax.set_xlabel(r'$\log(M_\star\,/\,M_\odot)$')
    ax.set_title('Scatter of {0} dex'.format(theta_arr[idx][4]),fontsize=12)

ax1.text(10.7,500,r'${0}/{1}$'.format(num_nans_arr[0],ngals_arr[0]),\
    bbox=dict(boxstyle="square",ec='k',fc='lightgray',alpha=0.5),size=10)
ax2.text(31,110,r'${0}/{1}$'.format(num_nans_arr[1],ngals_arr[1]),\
    bbox=dict(boxstyle="square",ec='k',fc='lightgray',alpha=0.5),size=10)
ax3.text(31,21,r'${0}/{1}$'.format(num_nans_arr[2],ngals_arr[2]),\
    bbox=dict(boxstyle="square",ec='k',fc='lightgray',alpha=0.5),size=10)
ax4.text(31,10.5,r'${0}/{1}$'.format(num_nans_arr[3],ngals_arr[3]),\
    bbox=dict(boxstyle="square",ec='k',fc='lightgray',alpha=0.5),size=10)
fig.show()

