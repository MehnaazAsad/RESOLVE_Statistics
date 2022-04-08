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
import emcee

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
quenching = 'halo'
mf_type = 'smf'
nwalkers = 100
nsteps = 1000
burnin = 300
ndim = 9
run = 46
    
def get_samples(chain_file, nsteps, nwalkers, ndim, burnin):
    if quenching == 'hybrid':
        emcee_table = pd.read_csv(chain_file, header=None, comment='#', 
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
        emcee_table = pd.read_csv(chain_file, header=None, comment='#', 
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


if run >= 37:
    reader = emcee.backends.HDFBackend(
        path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
        read_only=True)
    samples = reader.get_chain(flat=True, discard=burnin) 

else:
    chain_fname = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
        format(run, survey)

    samples = get_samples(chain_fname, nsteps, nwalkers, ndim, burnin)

nwalkers = 100
nsteps = 1000
burnin = 300
ndim = 9
run = 44

if run >= 37:
    reader = emcee.backends.HDFBackend(
        path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
        read_only=True)
    samples_44 = reader.get_chain(flat=True, discard=burnin) 

else:
    chain_fname = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
        format(run, survey)

    samples = get_samples(chain_fname, nsteps, nwalkers, ndim, burnin)

# chain_fname_halo_35 = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
#     format(run_smf, survey)
# samples_chain35 = get_samples(chain_fname_halo_35, nsteps, nwalkers, ndim, burnin)

Mhalo_c_fid = 12.35
Mstar_c_fid = 10.72
mlow_slope_fid = 0.44
mhigh_slope_fid = 0.57
scatter_fid = 0.15
behroozi10_param_vals = [Mhalo_c_fid, Mstar_c_fid, mlow_slope_fid, \
    mhigh_slope_fid, scatter_fid]
    
zumandelbaum_param_vals_hybrid = [10.5, 13.76, 0.69, 0.15] # For hybrid model
optimizer_best_fit_eco_smf_hybrid = [10.49, 14.03, 0.69, 0.148] # For hybrid model
zumandelbaum_param_vals_halo = [12.20, 0.38, 12.17, 0.15] # For halo model
optimizer_best_fit_eco_smf_halo = [12.61, 13.5, 0.40, 0.148] # For halo model

#45
best_fit_hybrid = [12.29868808, 10.48771974,  0.41364778,  0.41375455,  0.30506659,
       10.1679343 , 13.10135398,  0.81869216,  0.13844437]
#46
best_fit_halo = [12.37399415, 10.57683767,  0.42357192,  0.50163458,  0.31593679,
       11.86645536, 12.54502723,  1.42736618,  0.5261119 ]
# parameters=[r"${log_{10}\ M_{1}}$", 
#         r"${log_{10}\ M_{*}}$", r"${\beta}$",
#         r"${\delta}$", r"${\xi}$", 
#         r"${log_{10}\ M^{q}_{*}}$", r"${log_{10}\ M^{q}_{h}}$", 
#         r"${\mu}$", r"${\nu}$"]

c = ChainConsumer()
if quenching == 'hybrid':
    c.add_chain(samples, parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
        r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
        name=r"ECO hybrid (45)", color="#663399", zorder=10)

    # for i in range(len(best_fit_hybrid)):
    #     for j in range(len(best_fit_hybrid)):
    #         if i==j:
    #             continue
    #         else:
    #             c.add_marker([best_fit_hybrid[i],best_fit_hybrid[j]], 
    #             [parameters[i], parameters[j]], marker_style="*", 
    #             marker_size=100, color='#1f77b4')
        

    c.add_chain(samples_43,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
        r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
        name="ECO hybrid (43)", color='#E766EA', 
        zorder=13)

elif quenching == 'halo':
    c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{qc}_{h}}$", r"$\mathbf{log_{10}\ M^{qs}_{h}}$", 
        r"$\boldsymbol{{\mu}_c}$", r"$\boldsymbol{{\mu}_s}$"],
        name=r"ECO halo (46)", color='#663399', zorder=13)

    c.add_chain(samples_44,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{qc}_{h}}$", r"$\mathbf{log_{10}\ M^{qs}_{h}}$", 
        r"$\boldsymbol{{\mu}_c}$", r"$\boldsymbol{{\mu}_s}$"],
        name=r"ECO halo (44)", color='#E766EA', 
        zorder=10)


    # c.add_chain(samples_chain35,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    #     r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    #     r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    #     r"$\mathbf{log_{10}\ M^{qc}_{h}}$", r"$\mathbf{log_{10}\ M^{qs}_{h}}$", 
    #     r"$\boldsymbol{{\mu}_c}$", r"$\boldsymbol{{\mu}_s}$"],
    #     name=r"ECO halo: $\mathbf{\Phi}$  + $\mathbf{f_{blue, cen}}$ + $\mathbf{f_{blue, sat}}$", color='#E766EA', 
    #     zorder=10)

# c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
#      sigmas=[1,2], shade_alpha=0.4)

# sigma levels for 1D gaussian showing 68%,95% conf intervals
c.configure(kde=2.0, shade_gradient = 2.0, shade_alpha=0.8, label_font_size=15, 
    tick_font_size=10, summary=False, sigma2d=False, diagonal_tick_labels=False, 
    max_ticks=4, linewidths=2, legend_kwargs={"fontsize": 15})
c.configure_truth(color='goldenrod', lw=1.7)
fig1 = c.plotter.plot(display=True)
# c.configure(label_font_size=15, tick_font_size=10, summary=True, 
#     sigma2d=False, legend_kwargs={"fontsize": 15}) 
if quenching == 'hybrid':
    fig1 = c.plotter.plot(display=True, truth=best_fit_hybrid)
elif quenching == 'halo':
    fig1 = c.plotter.plot(display=True, truth=best_fit_halo)

if quenching == 'hybrid':
    fig1 = c.plotter.plot(filename='/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/contours_{0}.pdf'.format(quenching), 
    truth=best_fit_hybrid)
elif quenching == 'halo':
    fig1 = c.plotter.plot(filename='/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/contours_{0}.pdf'.format(quenching), 
        truth=best_fit_halo)

# fig2 = c.plotter.plot(filename=path_to_figures+'emcee_cc_mp_eco_corrscatter.png',\
#      truth=behroozi10_param_vals)
################################################################################
# CHAIN 33 and 35 (last and current halo quenching)
nwalkers = 100
nsteps = 1000
burnin = 200
ndim = 9
run_smf = 33

chain_fname_halo_33 = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run_smf, survey)
samples_chain33 = get_samples(chain_fname_halo_33, nsteps, nwalkers, ndim, burnin)

c = ChainConsumer()
# c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
#     r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
#     r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
#     r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
#     r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
#     name=r"ECO: $\mathbf{\Phi}$  + $\mathbf{f_{blue, tot}}$", color='#1f77b4', 
#     zorder=13)

# c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
#     r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
#     r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
#     r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
#     r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
#     name=r"ECO: $\mathbf{\Phi}$  + $\mathbf{f_{blue}}$", 
#     color='#E766EA', zorder=10)
c.add_chain(samples_chain33,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
    r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name="SMF + Total blue fraction", color='#1f77b4', 
    zorder=10)

c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
    r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
    name="SMF + Blue fraction of centrals and satellites", 
    color='#E766EA', zorder=13)

c.configure(kde=2.0,label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=False, legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=behroozi10_param_vals+zumandelbaum_param_vals_hybrid)

################################################################################

################################################################################
# CHAIN 32 and 34 (last and current hybrid quenching)
nwalkers = 100
nsteps = 1000
burnin = 200
ndim = 9
run_smf = 32

chain_fname_hybrid_32 = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run_smf, survey)
samples_chain32 = get_samples(chain_fname_hybrid_32, nsteps, nwalkers, ndim, burnin)

nwalkers = 100
nsteps = 1000
burnin = 200
ndim = 9
run_smf = 34

chain_fname_hybrid_34 = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
    format(run_smf, survey)
samples_chain34 = get_samples(chain_fname_hybrid_34, nsteps, nwalkers, ndim, burnin)

c = ChainConsumer()
# c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
#     r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
#     r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
#     r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
#     r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
#     name=r"ECO: $\mathbf{\Phi}$  + $\mathbf{f_{blue, tot}}$", color='#1f77b4', 
#     zorder=13)

# c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
#     r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
#     r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
#     r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
#     r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
#     name=r"ECO: $\mathbf{\Phi}$  + $\mathbf{f_{blue}}$", 
#     color='#E766EA', zorder=10)
c.add_chain(samples_chain32,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
    r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name="SMF + Total blue fraction", color='#1f77b4', 
    zorder=10)

c.add_chain(samples,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
    r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
    r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
    name="SMF + Blue fraction of centrals and satellites", 
    color='#E766EA', zorder=13)

c.configure(kde=2.0,label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=False, legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=behroozi10_param_vals+zumandelbaum_param_vals_hybrid)

################################################################################
# CHAIN 6 + CHAIN 32 BEHROOZI

behroozi_subset = samples[:,:5]

chain_six_behroozi = path_to_proc + 'smhm_run6/mcmc_{0}_raw.txt'.\
    format(survey)
mcmc_smf_1 = pd.read_csv(chain_six_behroozi, 
    names=['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',
    'scatter'],header=None, delim_whitespace=True)
mcmc_smf_1 = mcmc_smf_1[mcmc_smf_1.mhalo_c.values != '#']
mcmc_smf_1.mhalo_c = mcmc_smf_1.mhalo_c.astype(np.float64)
mcmc_smf_1.mstellar_c = mcmc_smf_1.mstellar_c.astype(np.float64)
mcmc_smf_1.lowmass_slope = mcmc_smf_1.lowmass_slope.astype(np.float64)

for idx,row in enumerate(mcmc_smf_1.values):
     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
          scatter_val = mcmc_smf_1.values[idx+1][0]
          row[4] = scatter_val

mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)

sampler_smf_1 = mcmc_smf_1.values.reshape(1000,250,5)

# Removing burn-in
ndim = 5
# samples_bmf_1 = sampler_bmf_1[:, 130:, :].reshape((-1, ndim))
samples_smf_1 = sampler_smf_1[130:, :, :].reshape((-1, ndim))

c = ChainConsumer()
c.add_chain(samples_smf_1,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$"],\
    name=r"ECO Behroozi: $\mathbf{\Phi}$", color='#E766EA', zorder=15)
c.add_chain(behroozi_subset,parameters=[r"$\mathbf{log_{10}\ M_{1}}$", 
    r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
    r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$"],
    name=r"ECO Behroozi: $\mathbf{\Phi}$  + $\mathbf{f_{blue}}$", color='#1f77b4', 
    zorder=13)
c.configure(smooth=[5, False], kde=[False, 2.0], label_font_size=20, 
    tick_font_size=8,summary=True, sigma2d=False, 
    legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=behroozi10_param_vals)

################################################################################
# CHAIN 17 + CHAIN 32 quenching

quenching_subset = samples[:,5:]
zumandelbaum_param_vals_hybrid = [10.5, 13.76, 0.69, 0.15] # For hybrid model

nwalkers = 260
nsteps = 780
burnin = 130
ndim = 4
run_smf = 17

## For SMF
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

mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_1 = mcmc_smf_1.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_1 = sampler_smf_1[burnin:, :, :].reshape((-1, ndim))

c = ChainConsumer()
c.add_chain(samples_smf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
    name=r"ECO quenching: $\mathbf{\Phi}$ + $\mathbf{\sigma}$-$\mathbf{M_{*}}$", 
    color='#E766EA', zorder=10)
c.add_chain(quenching_subset,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name=r"ECO quenching: $\mathbf{\Phi}$ + $\mathbf{f_{blue}}$", color='#1f77b4', 
    zorder=13)
c.configure(smooth=[5, 5],label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=False, legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=zumandelbaum_param_vals_hybrid)

################################################################################
# CHAIN 16 + CHAIN 32 quenching

nwalkers = 260
nsteps = 780
burnin = 130
ndim = 4
run_smf = 16

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

mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_1 = mcmc_smf_1.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_1 = sampler_smf_1[burnin:, :, :].reshape((-1, ndim))

c = ChainConsumer()
c.add_chain(samples_smf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
    name=r"ECO quenching: $\mathbf{\Phi}$", 
    color='#E766EA', zorder=10)
c.add_chain(quenching_subset,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name=r"ECO quenching: $\mathbf{\Phi}$ + $\mathbf{f_{blue}}$", color='#1f77b4', 
    zorder=13)
c.configure(smooth=[5, 5],label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=False, legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=zumandelbaum_param_vals_hybrid)

################################################################################
# CHAIN 21 + CHAIN 32 quenching

nwalkers = 260
nsteps = 1000
burnin = 200
ndim = 4
run_smf = 21

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

mcmc_smf_1 = mcmc_smf_1.dropna(axis='index', how='any').reset_index(drop=True)
sampler_smf_1 = mcmc_smf_1.values.reshape(nsteps,nwalkers,ndim)

# Removing burn-in
samples_smf_1 = sampler_smf_1[burnin:, :, :].reshape((-1, ndim))

c = ChainConsumer()
c.add_chain(samples_smf_1,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],\
    name=r"ECO quenching: $\mathbf{\Phi}$ + $\mathbf{\sigma}$-$\mathbf{M_{*}}$ + $\mathbf{M_{*}}$-$\mathbf{\sigma}$", 
    color='#E766EA', zorder=10)
c.add_chain(quenching_subset,parameters=[r"$\mathbf{M^{q}_{*}}$", 
    r"$\mathbf{M^{q}_{h}}$", r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"],
    name=r"ECO quenching: $\mathbf{\Phi}$ + $\mathbf{f_{blue}}$", color='#1f77b4', 
    zorder=13)
c.configure(smooth=[5, 5],label_font_size=20, tick_font_size=8,summary=True,\
     sigma2d=False, legend_kwargs={"fontsize": 30}) #1d gaussian showing 68%,95% conf intervals
fig2 = c.plotter.plot(display=True,truth=zumandelbaum_param_vals_hybrid)
