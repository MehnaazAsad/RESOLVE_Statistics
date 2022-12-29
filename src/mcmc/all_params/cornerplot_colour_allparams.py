"""
{This script carries out an MCMC analysis to parametrize the ECO SMHM}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from chainconsumer import ChainConsumer 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import emcee

__author__ = '{Mehnaaz Asad}'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{bm}")
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=2, size=7)
plt.rc('ytick.major', width=2, size=7)

survey = 'eco'
quenching = 'hybrid'
mf_type = 'bmf'
nwalkers = 500
nsteps = 200
burnin = 125
ndim = 9
run = 89
    
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
        path_to_proc + "bmhm_colour_run{0}/chain_hybrid_pca.h5".format(run), 
        read_only=True)
        # "/Users/asadm2/Desktop/chain_hybrid.h5", 
        # read_only=True)
    samples = reader.get_chain(flat=True, discard=burnin) 

else:
    chain_fname = path_to_proc + 'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
        format(run, survey)

    samples = get_samples(chain_fname, nsteps, nwalkers, ndim, burnin)

if quenching == 'hybrid':

    run = 88
    burnin = 125

    reader = emcee.backends.HDFBackend(
        path_to_proc + "bmhm_colour_run{0}/chain_{1}.h5".format(run, quenching), 
        read_only=True)
    samples_88 = reader.get_chain(flat=True, discard=burnin) 

    # run = 80
    # burnin = 100

    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_80 = reader.get_chain(flat=True, discard=burnin) 

    # run = 82
    # burnin = 100

    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_82 = reader.get_chain(flat=True, discard=burnin) 


    # run = 77
    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_77 = reader.get_chain(flat=True, discard=burnin) 

    # run = 78
    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_78 = reader.get_chain(flat=True, discard=burnin) 

elif quenching == 'halo':
    run = 86
    burnin = 125

    reader = emcee.backends.HDFBackend(
        path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
        read_only=True)
    samples_86 = reader.get_chain(flat=True, discard=burnin) 

    # run = 81
    # burnin = 125
    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_81 = reader.get_chain(flat=True, discard=burnin) 

    # run = 58
    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_58 = reader.get_chain(flat=True, discard=burnin) 

    # run = 60
    # reader = emcee.backends.HDFBackend(
    #     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
    #     read_only=True)
    # samples_60 = reader.get_chain(flat=True, discard=burnin) 

#* Baryonic
# quenching = 'hybrid'
# mf_type = 'bmf'
# nwalkers = 100
# nsteps = 1000
# burnin = 300
# ndim = 9
# run = 4

# reader = emcee.backends.HDFBackend(
#     path_to_proc + "bmhm_run{0}/chain.h5".format(run), 
#     read_only=True)
# samples_bmf = reader.get_chain(flat=True, discard=burnin) 

# run = 45
# reader = emcee.backends.HDFBackend(
#     path_to_proc + "smhm_colour_run{0}/chain.h5".format(run), 
#     read_only=True)
# samples_45 = reader.get_chain(flat=True, discard=burnin) 

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

#85
best_fit_hybrid = [12.35985583, 10.73234115, 0.48693041, 0.85309347, 0.17081168,
10.33681049, 11.64015182, 0.96749897, 0.16770749]

best_fit_84 = [12.64869191, 10.68374592,  0.4640583 ,  0.43375466,  
        0.22177846, 10.18868649, 13.23990274,  0.72632304,  0.05996219] 

best_fit_80 = [1.26205996e+01, 1.07021820e+01, 4.76597732e-01, 3.92284946e-01,
       1.68466879e-01, 1.01816555e+01, 1.34914230e+01, 7.29355889e-01,
       3.71080916e-02]

best_fit_69 = [1.25592081e+01, 1.06803444e+01, 4.68010987e-01, 3.46991714e-01,
       2.14114852e-01, 1.01642766e+01, 1.25618083e+01, 7.36058655e-01,
       1.37321398e-01]

best_fit_82 = [1.21988696e+01, 1.05941781e+01, 4.20468082e-01, 6.75529891e-01,
    2.83431492e-01, 1.01152952e+01, 1.49390877e+01, 6.22740146e-01,
    5.79299772e-03]

#87
best_fit_halo = [12.43515588, 10.65978211, 0.46731992, 0.35808034, 0.14540052,
12.26158655, 11.98913432, 0.63355372, 0.44848781]

best_fit_86 = [12.45003142, 10.49533185, 0.42905612, 0.49491889, 0.38661993,
11.928341 , 12.60596691, 1.63365685, 0.35175002]


best_fit_70 = [1.24943815e+01, 1.05162001e+01, 4.16658235e-01, 4.00419773e-01,
3.41828945e-01, 1.20060599e+01, 1.27011410e+01, 1.56991081e+00, 8.70258220e-01]

best_fit_83 = [12.53792836, 10.56614154, 0.4564306 , 0.6724157 , 0.33512296,
    12.01240587, 12.51050784, 1.40100554, 0.44524407]

best_fit_81 = [12.34987088, 10.41560466, 0.39277316, 0.40647588, 0.39271989,
11.8638783 , 12.59569064, 1.85941172, 0.40251681]

#* BARYONIC
#89
best_fit_hybrid_bmf = [12.42762598, 10.98243745, 0.44312913, 0.42222334, 0.15705579,
10.47769704, 13.84888006, 0.78807027, 0.29083456]

best_fit_88 = [1.17887431e+01, 1.06493529e+01, 4.16143172e-01, 3.96345226e-01,
4.50756145e-01, 1.07404229e+01, 1.42884117e+01, 4.13953837e-01,
5.70691342e-02]

# parameters=[r"${log_{10}\ M_{1}}$", 
#         r"${log_{10}\ M_{*}}$", r"${\beta}$",
#         r"${\delta}$", r"${\xi}$", 
#         r"${log_{10}\ M^{q}_{*}}$", r"${log_{10}\ M^{q}_{h}}$", 
#         r"${\mu}$", r"${\nu}$"]

param_names_hybrid = [r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{q}_{*}}$", r"$\mathbf{log_{10}\ M^{q}_{h}}$", 
        r"$\boldsymbol{\mu}$", r"$\boldsymbol{\nu}$"]

param_names_halo = [r"$\mathbf{log_{10}\ M_{1}}$", 
        r"$\mathbf{log_{10}\ M_{*}}$", r"$\boldsymbol{\beta}$",
        r"$\boldsymbol{\delta}$", r"$\boldsymbol{\xi}$", 
        r"$\mathbf{log_{10}\ M^{qc}_{h}}$", r"$\mathbf{log_{10}\ M^{qs}_{h}}$", 
        r"$\boldsymbol{{\mu}_c}$", r"$\boldsymbol{{\mu}_s}$"]

c = ChainConsumer()
if quenching == 'hybrid':
    c.add_chain(samples[:,:], parameters=param_names_hybrid[:],
        name=r"Hybrid 28 stats baryonic pca", color='#E766EA', zorder=-20)

    # c.add_marker(best_fit_83, parameters=param_names_hybrid, 
    #     name="Best-fit 83", marker_style="*", marker_size=200, 
    #     color="#FE420F")

    c.add_chain(samples_88[:,:], parameters=param_names_hybrid[:],
        name=r"Hybrid 28 stats baryonic non-pca", color='#FE420F', zorder=-10)

    c.add_marker(best_fit_88, parameters=param_names_hybrid, 
        name="Best-fit non-pca", marker_style="*", marker_size=200, 
        color="#FE420F")

    # c.add_chain(samples_81[:,:], parameters=param_names_hybrid[:],
    #     name=r"Halo sigma-M* (x-y)", color='#663399', zorder=-20)

    # c.add_marker(best_fit_82, parameters=param_names_hybrid, 
    #     name="Best-fit 82", marker_style="*", marker_size=200, 
    #     color="#663399")

    # c.add_chain(samples_82[:,:], parameters=param_names_hybrid[:],
    #     name=r"Chain 82 mstar-sigma (x-y)", color='#663399', zorder=25)

    # c.add_chain(samples_73[:,:], parameters=param_names_hybrid[:],
    #     name=r"Pair-splitting", color='#663399', zorder=22)

    # c.add_chain(samples_77[:,:],parameters=param_names_hybrid[:],
    #     name="$\mathbf{M_{*} - \sigma}$", color="#069AF3", 
    #     zorder=15)

    # c.add_chain(samples_78[:,:],parameters=param_names_hybrid[:],
    #     name="Pair-splitting removed from current mcmc version", 
    #     color="#2E8B57", 
    #     zorder=30)

    # c.add_chain(samples_59[:,:],parameters=param_names_hybrid[:],
    #     name="$\mathbf{\Phi}$  + $\mathbf{f_{blue, cen}}$ + $\mathbf{f_{blue, sat}}$ + $\mathbf{M_{*}-\sigma}$", color="#663399", 
    #     zorder=15)

    # c.add_chain(samples_55[:,:],parameters=param_names_hybrid[:],
    #     name="$\mathbf{\Phi}$", color='#FE420F', 
    #     zorder=5)

    # c.add_chain(samples_57[:,:],parameters=param_names_hybrid[:],
    #     name="$\mathbf{\Phi}$  + $\mathbf{f_{blue, cen}}$ + $\mathbf{f_{blue, sat}}$", color='#069AF3', 
    #     zorder=10)


elif quenching == 'halo':
    c.add_chain(samples[:,:], parameters=param_names_halo,
        name=r"halo 28 stats pca", color='#E766EA', zorder=-20)

    c.add_chain(samples_86[:,:], parameters=param_names_halo,
        name=r"halo 28 stats non-pca", color='#FE420F', zorder=-10)

    c.add_marker(best_fit_86, parameters=param_names_halo, 
        name="Best-fit non-pca", marker_style="*", marker_size=200, 
        color="#FE420F")

    # c.add_chain(samples_60[:,:],parameters=param_names_halo[:],
    #     name="$\mathbf{\Phi}$  + $\mathbf{f_{blue, cen}}$ + $\mathbf{f_{blue, sat}}$ + $\mathbf{M_{*}-\sigma}$", color="#663399", 
    #     zorder=15)

    # c.add_chain(samples_56[:,:],parameters=param_names_halo[:],
    #     name="$\mathbf{\Phi}$", color='#FE420F', 
    #     zorder=5)

    # c.add_chain(samples_58[:,:],parameters=param_names_halo[:],
    #     name="$\mathbf{\Phi}$  + $\mathbf{f_{blue, cen}}$ + $\mathbf{f_{blue, sat}}$", color='#069AF3', 
    #     zorder=10)
    # c.add_marker(best_fit_83, parameters=param_names_halo, 
    #     name="Best-fit halo mstar-sigma", marker_style="*", marker_size=200, 
    #     color="#663399")

# c.configure(shade_gradient=[0.1, 3.0], colors=['r', 'b'], \
#      sigmas=[1,2], shade_alpha=0.4)


# sigma levels for 1D gaussian showing 68%,95% conf intervals
c.configure(kde=2.0, shade_gradient = 2.0, shade_alpha=0.8, label_font_size=15, 
    tick_font_size=10, summary=True, sigma2d=False, diagonal_tick_labels=False, 
    max_ticks=4, linewidths=2, legend_kwargs={"fontsize": 15})
c.configure_truth(color='goldenrod', lw=1.7)
# fig1 = c.plotter.plot(display=True)
# c.configure(label_font_size=15, tick_font_size=10, summary=True, 
#     sigma2d=False, legend_kwargs={"fontsize": 15}) 
if quenching == 'hybrid':
    fig1 = c.plotter.plot(display=True, truth=best_fit_hybrid_bmf)
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
