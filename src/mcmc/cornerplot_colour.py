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
quenching = 'hybrid'
mf_type = 'both'
nwalkers = 260
nsteps = 1000
burnin = 200
ndim = 4
run_smf = 23
# run_bmf = 

## For SMF
if mf_type == 'smf' or mf_type == 'both':
     chain_fname_smf = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
     format(run_smf, survey)

     if quenching == 'hybrid':
          mcmc_smf_1 = pd.read_csv(chain_fname_smf, 
          names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)

          mcmc_smf_1 = mcmc_smf_1[mcmc_smf_1.Mstar_q.values != '#']
          mcmc_smf_1.Mstar_q = mcmc_smf_1.Mstar_q.astype(np.float64)
          mcmc_smf_1.Mhalo_q = mcmc_smf_1.Mhalo_q.astype(np.float64)
          mcmc_smf_1.mu = mcmc_smf_1.mu.astype(np.float64)
          mcmc_smf_1.nu = mcmc_smf_1.nu.astype(np.float64)

          for idx,row in enumerate(mcmc_smf_1.values):
               if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                    nu_val = mcmc_smf_1.values[idx+1][0]
                    row[3] = nu_val

     elif quenching == 'halo':
          mcmc_smf_1 = pd.read_csv(chain_fname_smf, delim_whitespace=True, 
               names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
               header=None)

          mcmc_smf_1 = mcmc_smf_1[mcmc_smf_1.Mh_qc.values != '#']
          mcmc_smf_1.Mh_qc = mcmc_smf_1.Mh_qc.astype(np.float64)
          mcmc_smf_1.Mh_qs = mcmc_smf_1.Mh_qs.astype(np.float64)
          mcmc_smf_1.mu_c = mcmc_smf_1.mu_c.astype(np.float64)
          mcmc_smf_1.mu_s = mcmc_smf_1.mu_s.astype(np.float64)

          for idx,row in enumerate(mcmc_smf_1.values):
               if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                    mu_s_val = mcmc_smf_1.values[idx+1][0]
                    row[3] = mu_s_val 


     mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)
     sampler_smf_1 = mcmc_smf_1.values.reshape(nsteps,nwalkers,ndim)

     # Removing burn-in
     samples_smf_1 = sampler_smf_1[burnin:, :, :].reshape((-1, ndim))

if mf_type == 'bmf' or mf_type == 'both':
     nsteps = 1000
     nwalkers = 260
     ndim = 4
     burnin = 200
     chain_fname_bmf = path_to_proc + 'smhm_colour_run24/mcmc_{0}_colour_raw.txt'.\
     format(survey)

     ## For BMF
     if quenching == 'hybrid':
          mcmc_bmf_1 = pd.read_csv(chain_fname_bmf, 
          names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)

          mcmc_bmf_1 = mcmc_bmf_1[mcmc_bmf_1.Mstar_q.values != '#']
          mcmc_bmf_1.Mstar_q = mcmc_bmf_1.Mstar_q.astype(np.float64)
          mcmc_bmf_1.Mhalo_q = mcmc_bmf_1.Mhalo_q.astype(np.float64)
          mcmc_bmf_1.mu = mcmc_bmf_1.mu.astype(np.float64)
          mcmc_bmf_1.nu = mcmc_bmf_1.nu.astype(np.float64)

          for idx,row in enumerate(mcmc_bmf_1.values):
               if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                    nu_val = mcmc_bmf_1.values[idx+1][0]
                    row[3] = nu_val

     elif quenching == 'halo':
          mcmc_bmf_1 = pd.read_csv(chain_fname_bmf, delim_whitespace=True, 
               names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
               header=None)

          mcmc_bmf_1 = mcmc_bmf_1[mcmc_bmf_1.Mh_qc.values != '#']
          mcmc_bmf_1.Mh_qc = mcmc_bmf_1.Mh_qc.astype(np.float64)
          mcmc_bmf_1.Mh_qs = mcmc_bmf_1.Mh_qs.astype(np.float64)
          mcmc_bmf_1.mu_c = mcmc_bmf_1.mu_c.astype(np.float64)
          mcmc_bmf_1.mu_s = mcmc_bmf_1.mu_s.astype(np.float64)

          for idx,row in enumerate(mcmc_bmf_1.values):
               if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                    mu_s_val = mcmc_bmf_1.values[idx+1][0]
                    row[3] = mu_s_val 


     mcmc_bmf_1 = mcmc_bmf_1.dropna(axis='index', how='any').reset_index(drop=True)
     sampler_bmf_1 = mcmc_bmf_1.values.reshape(nsteps,nwalkers,ndim)

     # Removing burn-in
     samples_bmf_1 = sampler_bmf_1[burnin:, :, :].reshape((-1, ndim))



nsteps = 1000
nwalkers = 260
ndim = 4
burnin = 200

chain_fname_smf = path_to_proc + 'smhm_colour_run25/mcmc_{0}_colour_raw.txt'.\
format(survey)

if quenching == 'hybrid':
     mcmc_smf_2 = pd.read_csv(chain_fname_smf, 
     names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)

     mcmc_smf_2 = mcmc_smf_2[mcmc_smf_2.Mstar_q.values != '#']
     mcmc_smf_2.Mstar_q = mcmc_smf_2.Mstar_q.astype(np.float64)
     mcmc_smf_2.Mhalo_q = mcmc_smf_2.Mhalo_q.astype(np.float64)
     mcmc_smf_2.mu = mcmc_smf_2.mu.astype(np.float64)
     mcmc_smf_2.nu = mcmc_smf_2.nu.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_2.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               nu_val = mcmc_smf_2.values[idx+1][0]
               row[3] = nu_val

elif quenching == 'halo':
     mcmc_smf_2 = pd.read_csv(chain_fname_smf, delim_whitespace=True, 
          names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
          header=None)

     mcmc_smf_2 = mcmc_smf_2[mcmc_smf_2.Mh_qc.values != '#']
     mcmc_smf_2.Mh_qc = mcmc_smf_2.Mh_qc.astype(np.float64)
     mcmc_smf_2.Mh_qs = mcmc_smf_2.Mh_qs.astype(np.float64)
     mcmc_smf_2.mu_c = mcmc_smf_2.mu_c.astype(np.float64)
     mcmc_smf_2.mu_s = mcmc_smf_2.mu_s.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_2.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               mu_s_val = mcmc_smf_2.values[idx+1][0]
               row[3] = mu_s_val 


mcmc_smf_2 = mcmc_smf_2.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_2 = mcmc_smf_2.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_2 = sampler_smf_2[burnin:, :, :].reshape((-1, ndim))

nsteps = 1000
nwalkers = 260
ndim = 4
burnin = 200

chain_fname_smf = path_to_proc + 'smhm_colour_run26/mcmc_{0}_colour_raw.txt'.\
format(survey)

if quenching == 'hybrid':
     mcmc_smf_3 = pd.read_csv(chain_fname_smf, 
     names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)

     mcmc_smf_3 = mcmc_smf_3[mcmc_smf_3.Mstar_q.values != '#']
     mcmc_smf_3.Mstar_q = mcmc_smf_3.Mstar_q.astype(np.float64)
     mcmc_smf_3.Mhalo_q = mcmc_smf_3.Mhalo_q.astype(np.float64)
     mcmc_smf_3.mu = mcmc_smf_3.mu.astype(np.float64)
     mcmc_smf_3.nu = mcmc_smf_3.nu.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_3.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               nu_val = mcmc_smf_3.values[idx+1][0]
               row[3] = nu_val

elif quenching == 'halo':
     mcmc_smf_3 = pd.read_csv(chain_fname_smf, delim_whitespace=True, 
          names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
          header=None)

     mcmc_smf_3 = mcmc_smf_3[mcmc_smf_3.Mh_qc.values != '#']
     mcmc_smf_3.Mh_qc = mcmc_smf_3.Mh_qc.astype(np.float64)
     mcmc_smf_3.Mh_qs = mcmc_smf_3.Mh_qs.astype(np.float64)
     mcmc_smf_3.mu_c = mcmc_smf_3.mu_c.astype(np.float64)
     mcmc_smf_3.mu_s = mcmc_smf_3.mu_s.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_3.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               mu_s_val = mcmc_smf_3.values[idx+1][0]
               row[3] = mu_s_val 


mcmc_smf_3 = mcmc_smf_3.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_3 = mcmc_smf_3.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_3 = sampler_smf_3[burnin:, :, :].reshape((-1, ndim))

nsteps = 1000
nwalkers = 260
ndim = 4
burnin = 200

chain_fname_smf = path_to_proc + 'smhm_colour_run27/mcmc_{0}_colour_raw.txt'.\
format(survey)

if quenching == 'hybrid':
     mcmc_smf_4 = pd.read_csv(chain_fname_smf, 
     names=['Mstar_q','Mhalo_q','mu','nu'],header=None, delim_whitespace=True)

     mcmc_smf_4 = mcmc_smf_4[mcmc_smf_4.Mstar_q.values != '#']
     mcmc_smf_4.Mstar_q = mcmc_smf_4.Mstar_q.astype(np.float64)
     mcmc_smf_4.Mhalo_q = mcmc_smf_4.Mhalo_q.astype(np.float64)
     mcmc_smf_4.mu = mcmc_smf_4.mu.astype(np.float64)
     mcmc_smf_4.nu = mcmc_smf_4.nu.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_4.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               nu_val = mcmc_smf_4.values[idx+1][0]
               row[3] = nu_val

elif quenching == 'halo':
     mcmc_smf_4 = pd.read_csv(chain_fname_smf, delim_whitespace=True, 
          names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
          header=None)

     mcmc_smf_4 = mcmc_smf_4[mcmc_smf_4.Mh_qc.values != '#']
     mcmc_smf_4.Mh_qc = mcmc_smf_4.Mh_qc.astype(np.float64)
     mcmc_smf_4.Mh_qs = mcmc_smf_4.Mh_qs.astype(np.float64)
     mcmc_smf_4.mu_c = mcmc_smf_4.mu_c.astype(np.float64)
     mcmc_smf_4.mu_s = mcmc_smf_4.mu_s.astype(np.float64)

     for idx,row in enumerate(mcmc_smf_4.values):
          if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
               mu_s_val = mcmc_smf_4.values[idx+1][0]
               row[3] = mu_s_val 


mcmc_smf_4 = mcmc_smf_4.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_4 = mcmc_smf_4.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_4 = sampler_smf_4[burnin:, :, :].reshape((-1, ndim))

zumandelbaum_param_vals_hybrid = [10.5, 13.76, 0.69, 0.15] # For hybrid model
optimizer_best_fit_eco_smf_hybrid = [10.49, 14.03, 0.69, 0.14] # For hybrid model
zumandelbaum_param_vals_halo = [12.20, 0.38, 12.17, 0.15] # For halo model
optimizer_best_fit_eco_smf_halo = [12.61, 13.5, 0.40, 0.148] # For halo model

c = ChainConsumer()
if mf_type == 'smf' or mf_type == 'both':
     c.add_chain(samples_smf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi", 
          color='#E766EA', zorder=9)
if mf_type == 'bmf' or mf_type == 'both':
     c.add_chain(samples_bmf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO best-fit Behroozi", 
          color='#53A48D', zorder=10)
c.add_chain(samples_smf_2,parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi ", color='#1f77b4', zorder=11)
c.add_chain(samples_smf_3,parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi  ", color='#DB7093', zorder=12)
c.add_chain(samples_smf_4,parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi   ", color='#FFD700', zorder=13)

# c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
#      sigmas=[1,2], shade_alpha=0.4)
c.configure(smooth=5,label_font_size=25,tick_font_size=10,summary=True,\
     sigma2d=False,legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
if quenching == 'hybrid':
     if mf_type == 'smf' or mf_type == 'both':
          fig1 = c.plotter.plot(display=True,truth=optimizer_best_fit_eco_smf_hybrid)
elif quenching == 'halo':
     if mf_type == 'smf' or mf_type == 'both':
          fig1 = c.plotter.plot(display=True,truth=optimizer_best_fit_eco_smf_halo)
# fig2 = c.plotter.plot(filename=path_to_figures+'emcee_cc_mp_eco_corrscatter.png',\
#      truth=behroozi10_param_vals)
