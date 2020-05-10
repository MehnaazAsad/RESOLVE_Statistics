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

__author__ = '{Mehnaaz Asad}'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=[r"\usepackage{amsmath}"])
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

survey = 'eco'
file_ver = 2.0


chain_fname_smf_1 = path_to_proc + 'smhm_colour_run9/mcmc_{0}_colour_raw.txt'.\
    format(survey)
mcmc_smf_1 = pd.read_csv(chain_fname_smf_1, 
    names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)
mcmc_smf_1 = mcmc_smf_1[mcmc_smf_1.Mstar_q.values != '#']
mcmc_smf_1.Mstar_q = mcmc_smf_1.Mstar_q.astype(np.float64)
mcmc_smf_1.Mhalo_q = mcmc_smf_1.Mhalo_q.astype(np.float64)
mcmc_smf_1.mu = mcmc_smf_1.mu.astype(np.float64)
mcmc_smf_1.nu = mcmc_smf_1.nu.astype(np.float64)

# chain_fname_bmf_1 = path_to_proc + 'bmhm_run3/combined_processed_{0}_raw.txt'.\
#      format(survey)
# mcmc_bmf_1 = pd.read_csv(chain_fname_bmf_1,
#     names=['mhalo_c','mstellar_c','lowmass_slope','highmass_slope','scatter'], 
#     header=None, delim_whitespace=True)
# # mcmc_bmf_1 = mcmc_bmf_1[mcmc_bmf_1.mhalo_c.values != '#']
# mcmc_bmf_1.mhalo_c = mcmc_bmf_1.mhalo_c.astype(np.float64)
# mcmc_bmf_1.mstellar_c = mcmc_bmf_1.mstellar_c.astype(np.float64)
# mcmc_bmf_1.lowmass_slope = mcmc_bmf_1.lowmass_slope.astype(np.float64)

# Cases where last parameter was a NaN and its value was being written to 
# the first element of the next line followed by 4 NaNs for the other parameters
# for idx,row in enumerate(mcmc_bmf_1.values):
#      if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
#           scatter_val = mcmc_bmf_1.values[idx+1][0]
#           row[4] = scatter_val 

for idx,row in enumerate(mcmc_smf_1.values):
     if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
          nu_val = mcmc_smf_1.values[idx+1][0]
          row[3] = nu_val

# mcmc_bmf_1 = mcmc_bmf_1.dropna(axis='index', how='any').reset_index(drop=True)
mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)
mcmc_smf_1.nu = np.log10(mcmc_smf_1.nu)

# sampler_bmf_1 = mcmc_bmf_1.values.reshape(250,1000,5)
sampler_smf_1 = mcmc_smf_1.values.reshape(700,256,4)

# Removing burn-in
ndim = 4
# samples_bmf_1 = sampler_bmf_1[:, 130:, :].reshape((-1, ndim))
samples_smf_1 = sampler_smf_1[250:, :, :].reshape((-1, ndim))

survey = 'eco'
file_ver = 2.0

if survey == 'eco' and file_ver == 1.0:
     chain_fname_smf_2 = path_to_proc + 'smhm_run4_errjk/mcmc_eco.dat'
     mcmc_smf_2 = pd.read_csv(chain_fname_smf_2,
          names=['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',
          'scatter'],sep='\s+',dtype=np.float64)
else:
     chain_fname_smf_2 = '~/Desktop/mcmc_{0}_colour_raw.txt'.\
          format(survey)
     # chain_fname_smf_2 = path_to_proc + 'smhm_run5_errmock/mcmc_{0}_raw.txt'.\
     #      format(survey)
     mcmc_smf_2 = pd.read_csv(chain_fname_smf_2, 
     names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)
     mcmc_smf_2 = mcmc_smf_2[mcmc_smf_2.Mstar_q.values != '#']
     mcmc_smf_2.Mstar_q = mcmc_smf_2.Mstar_q.astype(np.float64)
     mcmc_smf_2.Mhalo_q = mcmc_smf_2.Mhalo_q.astype(np.float64)
     mcmc_smf_2.mu = mcmc_smf_2.mu.astype(np.float64)
     mcmc_smf_2.nu = mcmc_smf_2.nu.astype(np.float64)

# chain_fname_bmf_2 = path_to_proc + 'bmhm_run3/combined_processed_{0}_raw.txt'.format(survey)
# mcmc_bmf_2 = pd.read_csv(chain_fname_bmf_2,
#     names=['mhalo_c','mstellar_c','lowmass_slope','highmass_slope','scatter'], 
#     header=None, delim_whitespace=True)
# mcmc_bmf_2 = mcmc_bmf_2[mcmc_bmf_2.mhalo_c.values != '#']
# mcmc_bmf_2.mhalo_c = mcmc_bmf_2.mhalo_c.astype(np.float64)
# mcmc_bmf_2.mstellar_c = mcmc_bmf_2.mstellar_c.astype(np.float64)
# mcmc_bmf_2.lowmass_slope = mcmc_bmf_2.lowmass_slope.astype(np.float64)

# # Cases where last parameter was a NaN and its value was being written to 
# # the first element of the next line followed by 4 NaNs for the other parameters
# for idx,row in enumerate(mcmc_bmf_2.values):
#      if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
#           scatter_val = mcmc_bmf_2.values[idx+1][0]
#           row[4] = scatter_val 

for idx,row in enumerate(mcmc_smf_1.values):
     if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
          nu_val = mcmc_smf_1.values[idx+1][0]
          row[3] = nu_val

# mcmc_bmf_2 = mcmc_bmf_2.dropna(axis='index', how='any').reset_index(drop=True)
mcmc_smf_2 = mcmc_smf_2.dropna(axis='index', how='any').reset_index(drop=True)
mcmc_smf_2.nu = np.log10(mcmc_smf_2.nu) # so that both run 9 and run 10 chains can be plotted 

# sampler_bmf_2 = mcmc_bmf_2.values.reshape(250,1000,5)
sampler_smf_2 = mcmc_smf_2.values.reshape(700,256,4)

# Removing burn-in
ndim = 4
# samples_bmf_2 = sampler_bmf_2[:, 130:, :].reshape((-1, ndim))
samples_smf_2 = sampler_smf_2[200:, :, :].reshape((-1, ndim))

zumandelbaum_param_vals = [10.5, 13.76, 0.69, 0.15]

c = ChainConsumer()
# c.add_chain(samples_smf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", \
#      r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$",\
#           r"$\boldsymbol{\nu}$"],\
#                name="ECO run 9", color='#E766EA', zorder=10)
# c.add_chain(samples_bmf_1,parameters=[r"$\mathbf{M_{1}}$", \
#      r"$\mathbf{M_{*(b),0}}$", r"$\boldsymbol{\beta}$",\
#           r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$"],\
#                name="ECO baryonic", color='#53A48D')
c.add_chain(samples_smf_2,parameters=[r"$\mathbf{M^{q}_{*}}$", \
     r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$",\
          r"$\boldsymbol{\nu}$"],\
               name="ECO run 10", color='#53A48D', zorder=10)
# c.add_chain(samples_bmf_2,parameters=[r"$\mathbf{M_{1}}$", \
#      r"$\mathbf{M_{*(b),0}}$", r"$\boldsymbol{\beta}$",\
#           r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$"],\
#                name="RESOLVE A BMHM", color='b')
# c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
#      sigmas=[1,2], shade_alpha=0.4)
c.configure(smooth=True,label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=True, sigmas=[1, 2]) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=zumandelbaum_param_vals)
# fig2 = c.plotter.plot(filename=path_to_figures+'emcee_cc_mp_eco_corrscatter.png',\
#      truth=behroozi10_param_vals)

