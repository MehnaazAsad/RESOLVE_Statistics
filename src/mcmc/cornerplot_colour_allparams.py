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
quenching = 'hybrid'
mf_type = 'smf'
nwalkers = 30
nsteps = 500
burnin = 100
ndim = 9
run = 28


def get_samples(chain_file, nsteps, nwalkers, ndim):
    if quenching == 'hybrid':
        emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
            names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'Mstar_q','Mhalo_q','mu','nu'], sep='\s+')

        for idx,row in enumerate(emcee_table.values):

            ## For cases where 5 params on one line and 3 on the next
            if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
                mhalo_q_val = emcee_table.values[idx+1][0]
                mu_val = emcee_table.values[idx+1][1]
                nu_val = emcee_table.values[idx+1][2]
                row[6] = mhalo_q_val
                row[7] = mu_val
                row[8] = nu_val 

            ## For cases where 4 params on one line, 4 on the next and 1 on the 
            ## third line (numbers in scientific notation unlike case above)
            elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
                scatter_val = emcee_table.values[idx+1][0]
                mstar_q_val = emcee_table.values[idx+1][1]
                mhalo_q_val = emcee_table.values[idx+1][2]
                mu_val = emcee_table.values[idx+1][3]
                nu_val = emcee_table.values[idx+2][0]
                row[4] = scatter_val
                row[5] = mstar_q_val
                row[6] = mhalo_q_val
                row[7] = mu_val
                row[8] = nu_val 

    elif quenching == 'halo':
        emcee_table = pd.read_csv(chain_fname, header=None, comment='#', 
            names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'Mhalo_qc','Mhalo_qs','mu_c','mu_s'], sep='\s+')

        for idx,row in enumerate(emcee_table.values):

            ## For cases where 5 params on one line and 3 on the next
            if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
                mhalo_qs_val = emcee_table.values[idx+1][0]
                mu_c_val = emcee_table.values[idx+1][1]
                mu_s_val = emcee_table.values[idx+1][2]
                row[6] = mhalo_qs_val
                row[7] = mu_c_val
                row[8] = mu_s_val 

            ## For cases where 4 params on one line, 4 on the next and 1 on the 
            ## third line (numbers in scientific notation unlike case above)
            elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
                scatter_val = emcee_table.values[idx+1][0]
                mhalo_qc_val = emcee_table.values[idx+1][1]
                mhalo_qs_val = emcee_table.values[idx+1][2]
                mu_c_val = emcee_table.values[idx+1][3]
                mu_s_val = emcee_table.values[idx+2][0]
                row[4] = scatter_val
                row[5] = mhalo_qc_val
                row[6] = mhalo_qs_val
                row[7] = mu_c_val
                row[8] = mu_s_val 


    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)    
    sampler = emcee_table.values.reshape(nsteps,nwalkers,ndim)

    # Removing burn-in
    samples = sampler[burnin:, :, :].reshape((-1, ndim))

    return samples

chain_fname = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)

samples = get_samples(chain_fname, nsteps, nwalkers, ndim)

Mhalo_c_fid = 12.35
Mstar_c_fid = 10.72
mlow_slope_fid = 0.44
mhigh_slope_fid = 0.57
scatter_fid = 0.15
behroozi10_param_vals = [Mhalo_c_fid, Mstar_c_fid, mlow_slope_fid, \
    mhigh_slope_fid, scatter_fid]
zumandelbaum_param_vals_hybrid = [10.5, 13.76, 0.69, 0.15] # For hybrid model
optimizer_best_fit_eco_smf_hybrid = [10.49, 14.03, 0.69, 0.14] # For hybrid model
zumandelbaum_param_vals_halo = [12.20, 0.38, 12.17, 0.15] # For halo model
optimizer_best_fit_eco_smf_halo = [12.61, 13.5, 0.40, 0.148] # For halo model

c = ChainConsumer()
c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
    r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name=r"ECO: $\mathbf{\Phi}$  + $\mathbf{f_{blue}}$", color='#1f77b4', 
    zorder=13)

# c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
#      sigmas=[1,2], shade_alpha=0.4)

# sigma levels for 1D gaussian showing 68%,95% conf intervals
c.configure(smooth=5, label_font_size=20, tick_font_size=10, summary=True, 
    sigma2d=False, legend_kwargs={"fontsize": 30}) 
if quenching == 'hybrid':
    fig1 = c.plotter.plot(display=True, 
        truth=behroozi10_param_vals+optimizer_best_fit_eco_smf_hybrid)
elif quenching == 'halo':
    fig1 = c.plotter.plot(display=True, 
        truth=behroozi10_param_vals+optimizer_best_fit_eco_smf_halo)
# fig2 = c.plotter.plot(filename=path_to_figures+'emcee_cc_mp_eco_corrscatter.png',\
#      truth=behroozi10_param_vals)
