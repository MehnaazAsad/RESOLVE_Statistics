"""
{This script carries out an MCMC analysis to parametrize the ECO SMHM}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from chainconsumer import ChainConsumer 
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import corner

__author__ = '{Mehnaaz Asad}'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=10)
rc('text', usetex=True)

chain_fname_smf = path_to_proc + 'smhm_run3/mcmc_eco.dat'
chain_fname_bmf = path_to_proc + 'bmhm/mcmc_eco.dat'

emcee_table_smf = pd.read_csv(chain_fname_smf,names=['mhalo_c','mstellar_c',\
                                             'lowmass_slope','highmass_slope',\
                                             'scatter'],sep='\s+',dtype=np.float64)

emcee_table_bmf = pd.read_csv(chain_fname_bmf,names=['mhalo_c','mstellar_c',\
                                             'lowmass_slope','highmass_slope',\
                                             'scatter'],sep='\s+',dtype=np.float64)

# Cases where last parameter was a NaN and its value was being written to 
# the first element of the next line followed by 4 NaNs for the other parameters
for idx,row in enumerate(emcee_table_smf.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = emcee_table_smf.values[idx+1][0]
          row[4] = scatter_val

for idx,row in enumerate(emcee_table_bmf.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = emcee_table_bmf.values[idx+1][0]
          row[4] = scatter_val 

emcee_table_smf = emcee_table_smf.dropna().reset_index(drop=True)
emcee_table_bmf = emcee_table_bmf.dropna().reset_index(drop=True)

emcee_table_smf = emcee_table_smf.dropna(axis='index',how='any')
emcee_table_bmf = emcee_table_bmf.dropna(axis='index',how='any')
sampler_smf = emcee_table_smf.values.reshape(250,1000,5)
sampler_bmf = emcee_table_bmf.values.reshape(250,500,5)
ndim = 5
samples_smf = sampler_smf[:, 100:, :].reshape((-1, ndim))
samples_bmf = sampler_bmf[:, 100:, :].reshape((-1, ndim))

behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]
best_fit_eco_smhm = [12.356,10.601,0.446,0.62,0.315]

c = ChainConsumer()
c.add_chain(samples_smf,parameters=[r"$\boldmath{M_{halo}}$", \
     r"$\boldmath{M_{stellar}}$", r"$\boldmath{{low\ mass}}$",\
          r"$\boldmath{high\ mass}$", r"$\boldmath{scatter}$"],\
               name="ECO SMHM", color='r')
c.add_chain(samples_bmf,parameters=[r"$\boldmath{M_{halo}}$", \
     r"$\boldmath{M_{stellar}}$", r"$\boldmath{{low\ mass}}$",\
          r"$\boldmath{high\ mass}$", r"$\boldmath{scatter}$"],\
               name="ECO BMHM", color='b')
c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
     sigmas=[1,2], shade_alpha=0.4)
c.configure(smooth=True,label_font_size=12, tick_font_size=8,summary=True,\
     sigma2d=True, sigmas=[1, 2]) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=behroozi10_param_vals)
# fig2 = c.plotter.plot(filename=path_to_figures+'emcee_cc_mp_eco_corrscatter.png',\
#      truth=behroozi10_param_vals)

