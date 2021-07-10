"""
{This script samples from multiple chains to create and plot a master chain}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from chainconsumer import ChainConsumer 
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

def get_samples(chain_file, chi2_file, nsteps, nwalkers, ndim, burnin):
    if quenching == 'hybrid':
        emcee_table = pd.read_csv(chain_file,  comment='#',
            names=['Mstar_q','Mhalo_q','mu','nu'], header=None,  
            delim_whitespace=True)

        for idx,row in enumerate(emcee_table.values):
            if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                nu_val = emcee_table.values[idx+1][0]
                row[3] = nu_val

    elif quenching == 'halo':
        emcee_table = pd.read_csv(chain_file, delim_whitespace=True, 
            names=['Mh_qc','Mh_qs','mu_c','mu_s'], 
            header=None)

        for idx,row in enumerate(emcee_table.values):
            if np.isnan(row)[3] == True and np.isnan(row)[2] == False:
                mu_s_val = emcee_table.values[idx+1][0]
                row[3] = mu_s_val 


    emcee_table = emcee_table.dropna(axis='index', how='any').reset_index(drop=True)
    
    chi2_df = pd.read_csv(chi2_file,header=None,names=['chisquared'])
    chi2 = np.log10(chi2_df.chisquared.values)
    emcee_table['chi2'] = chi2

    p = np.exp(-(emcee_table['chi2']/2))
    emcee_table['P'] = p
    threshold = np.round((np.mean(emcee_table.P.values) - \
        np.std(emcee_table.P.values)),2)

    # ndim+2 because there are 4 parameters + chi2 + P columns now
    ndim = ndim+2

    sampler = emcee_table.values.reshape(nsteps, nwalkers, ndim)

    # Removing burn-in
    samples = sampler[burnin:, :, :].reshape((-1, ndim))

    subset_samples = samples[samples[:,-1]>threshold]

    return subset_samples

nwalkers = 260
nsteps = 1000
burnin = 200
ndim = 4
run = 24
chain_bestfit = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)
chi2_fname = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
samples_bestfit = get_samples(chain_bestfit, chi2_fname, nsteps, nwalkers, ndim,
    burnin)

run = 23
chain = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)
chi2_fname = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
samples_23 = get_samples(chain, chi2_fname, nsteps, nwalkers, ndim, burnin)

run = 25
chain = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)
chi2_fname = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
samples_25 = get_samples(chain, chi2_fname, nsteps, nwalkers, ndim, burnin)

run = 26
chain = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)
chi2_fname = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
samples_26 = get_samples(chain, chi2_fname, nsteps, nwalkers, ndim, burnin)

run = 27
chain = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run, survey)
chi2_fname = path_to_proc + 'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
    format(run, survey)
samples_27 = get_samples(chain, chi2_fname, nsteps, nwalkers, ndim, burnin)

master_chain = np.vstack((np.vstack((np.vstack((np.vstack((samples_bestfit, 
    samples_23)),samples_25)),samples_26)),samples_27))[:,0:4]

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
c.add_chain(master_chain, parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"Master chain", color='#1f77b4', zorder=15, linewidth=4)

c.add_chain(samples_bestfit[:,0:4],parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"Best fit", color='#DB7093', zorder=14, linewidth=2, 
          linestyle='dotted')

c.add_chain(samples_23[:,0:4],parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi", color='#FFD700', zorder=13, 
          linewidth=2, linestyle='dashed')
c.add_chain(samples_25[:,0:4],parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi ", color='#E766EA', zorder=13, 
          linewidth=2, linestyle='dashdot')
c.add_chain(samples_26[:,0:4],parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi  ", color='#53A48D', zorder=13, 
          linewidth=2)
c.add_chain(samples_27[:,0:4],parameters=[r"$\mathbf{M^{q}_{*}}$", 
          r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
          name=r"ECO 2$\sigma$ Behroozi   ", color='#C0C0C0', zorder=13, 
          linewidth=2, linestyle=':')


# sigma levels for 1D gaussian showing 68%,95% conf intervals
c.configure(smooth=5, label_font_size=20, tick_font_size=10, summary=True, 
    sigma2d=False, legend_kwargs={"fontsize": 30}) 
if quenching == 'hybrid':
    fig1 = c.plotter.plot(display=True, 
        truth=optimizer_best_fit_eco_smf_hybrid)
elif quenching == 'halo':
    fig1 = c.plotter.plot(display=True, 
        truth=optimizer_best_fit_eco_smf_halo)
