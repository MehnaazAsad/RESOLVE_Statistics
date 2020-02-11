"""
{This script calculates correlation matrix from mocks}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import rc
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

def reading_catls(filename, catl_format='.hdf5'):
    """
    Function to read ECO/RESOLVE catalogues.

    Parameters
    ----------
    filename: string
        path and name of the ECO/RESOLVE catalogue to read

    catl_format: string, optional (default = '.hdf5')
        type of file to read.
        Options:
            - '.hdf5': Reads in a catalogue in HDF5 format

    Returns
    -------
    mock_pd: pandas DataFrame
        DataFrame with galaxy/group information

    Examples
    --------
    # Specifying `filename`
    >>> filename = 'ECO_catl.hdf5'

    # Reading in Catalogue
    >>> mock_pd = reading_catls(filename, format='.hdf5')

    >>> mock_pd.head()
               x          y         z          vx          vy          vz  \
    0  10.225435  24.778214  3.148386  356.112457 -318.894409  366.721832
    1  20.945772  14.500367 -0.237940  168.731766   37.558834  447.436951
    2  21.335835  14.808488  0.004653  967.204407 -701.556763 -388.055115
    3  11.102760  21.782235  2.947002  611.646484 -179.032089  113.388794
    4  13.217764  21.214905  2.113904  120.689598  -63.448833  400.766541

       loghalom  cs_flag  haloid  halo_ngal    ...        cz_nodist      vel_tot  \
    0    12.170        1  196005          1    ...      2704.599189   602.490355
    1    11.079        1  197110          1    ...      2552.681697   479.667489
    2    11.339        1  197131          1    ...      2602.377466  1256.285409
    3    11.529        1  199056          1    ...      2467.277182   647.318259
    4    10.642        1  199118          1    ...      2513.381124   423.326770

           vel_tan     vel_pec     ra_orig  groupid    M_group g_ngal  g_galtype  \
    0   591.399858 -115.068833  215.025116        0  11.702527      1          1
    1   453.617221  155.924074  182.144134        1  11.524787      4          0
    2  1192.742240  394.485714  182.213220        1  11.524787      4          0
    3   633.928896  130.977416  210.441320        2  11.502205      1          1
    4   421.064495   43.706352  205.525386        3  10.899680      1          1

       halo_rvir
    0   0.184839
    1   0.079997
    2   0.097636
    3   0.113011
    4   0.057210
    """
    ## Checking if file exists
    if not os.path.exists(filename):
        msg = '`filename`: {0} NOT FOUND! Exiting..'.format(filename)
        raise ValueError(msg)
    ## Reading file
    if catl_format=='.hdf5':
        mock_pd = pd.read_hdf(filename)
    else:
        msg = '`catl_format` ({0}) not supported! Exiting...'.format(catl_format)
        raise ValueError(msg)

    return mock_pd

def read_data(path_to_file, survey):
    """
    Reads survey catalog from file

    Parameters
    ----------
    path_to_file: `string`
        Path to survey catalog file

    survey: `string`
        Name of survey

    Returns
    ---------
    catl: `pandas.DataFrame`
        Survey catalog with grpcz, abs rmag and stellar mass limits
    
    volume: `float`
        Volume of survey

    cvar: `float`
        Cosmic variance of survey

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.cz.values >= 3000) & \
            (eco_buff.cz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &\
                (eco_buff.logmstar.values >= 8.9)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if survey == 'resolvea':
            catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
                (resolve_live18.grpcz.values > 4500) & \
                    (resolve_live18.grpcz.values < 7000) & \
                        (resolve_live18.absrmag.values < -17.33) & \
                            (resolve_live18.logmstar.values >= 8.9)]


            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            # 487 - cz, 369 - grpcz
            catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
                (resolve_live18.grpcz.values > 4500) & \
                    (resolve_live18.grpcz.values < 7000) & \
                        (resolve_live18.absrmag.values < -17) & \
                            (resolve_live18.logmstar.values >= 8.7)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def diff_smf(mstar_arr, volume, cvar_err, h1_bool):
    """
    Calculates differential stellar mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    phi: array
        Array of y-axis values

    err_tot: array
        Array of error values per bin
    
    bins: array
        Array of bin edge values
    """
    if not h1_bool:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)        
    else:
        logmstar_arr = mstar_arr
    if survey == 'eco':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    
    elif survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)   


    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    phi = counts / (volume * dm)  # not a log quantity
    return maxis, phi, err_poiss, bins, counts

def diff_bmf(mass_arr, volume, cvar_err, sim_bool, h1_bool):
    """Calculates differential stellar mass function given stellar/baryonic
     masses."""  
    if sim_bool:
        mass_arr = np.log10(mass_arr)
    
    if not h1_bool:
        # changing from h=0.7 to h=1
        mass_arr = np.log10((10**mass_arr) / 2.041)
    
    if survey == 'eco':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    
    if survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)    

    if survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7) 
       

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(mass_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)

    phi = counts / (volume * dm)  # not a log quantity
    return maxis, phi, err_poiss, bins, counts

def cumu_num_dens(data,weights,volume,bool_mag): 
    if weights is None: 
        weights = np.ones(len(data)) 
    else: 
        weights = np.array(weights) 
    #Unnormalized histogram and bin edges 
    data += 0.775 #change mags from h=0.7 to h=1
    bins = np.arange(data.min(), data.max(), 0.2) 
    freq,edg = np.histogram(data,bins=bins,weights=weights) 
    bin_centers = 0.5*(edg[1:]+edg[:-1]) 
    bin_width = edg[1] - edg[0] 
    if not bool_mag: 
        N_cumu = np.cumsum(freq[::-1])[::-1]  
    else: 
        N_cumu = np.cumsum(freq) 
    n_cumu = N_cumu/volume 
    err_poiss = np.sqrt(N_cumu)/volume 
    return bin_centers,edg,n_cumu,err_poiss,bin_width 
                                                               
def frac_error(phi_arr):
    # 3D array where each element is all the values of phi per bin
    phi_arr_trans = np.transpose(phi_arr)

    std_arr = []
    mean_arr = []
    frac_err_arr_mean = []
    frac_err_arr = []
    for idx,value in enumerate(phi_arr_trans):
        #Measure std between each bin for all mocks
        std = np.std(value)
        #Measure mean between each bin for all mocks
        mean = np.mean(value)
        
        #Fractional error on the mean of all mocks
        frac_error_mean = std/mean

        #Fractional error for each bin of each mock
        temp_arr = []
        for value2 in value:
            # if value2 == 0:
            #     value2 = 10**-99
            frac_error = std/value2 
            temp_arr.append(frac_error)   

        std_arr.append(std)
        mean_arr.append(mean)
        frac_err_arr_mean.append(frac_error_mean)
        frac_err_arr.append(temp_arr)

    frac_err_arr = np.array(frac_err_arr)
    return frac_err_arr, frac_err_arr_mean

def sigma_from_frac(frac_err_arr, frac_err_arr_mean, phi_data):
    # Within each bin multiply frac_error with phi value for each bin
    # Using the mean of each bin
    sigma_final_arr_mean = frac_err_arr_mean * phi_data

    # Within each bin of each mock multiply frac_error with phi value from data
    # for each bin
    sigma_final_arr = []
    for idx,value in enumerate(frac_err_arr):
        value = value * phi_data[idx]
        sigma_final_arr.append(value)

    # Transpose fractional error array because to use df.corr function the columns
    # have to be the different variables i.e. bin numbers
    frac_err_arr_trans = np.transpose(frac_err_arr)
    sigma_final_arr_trans = frac_err_arr_trans * phi_data
    return sigma_final_arr_trans, sigma_final_arr

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']
# path_to_eco = path_to_external + 'm200b/eco/old/ECO_m200b_catls/'
path_to_resolvea = path_to_external + 'RESOLVE_A_mvir_catls/'
path_to_resolveb = path_to_external + 'RESOLVE_B_mvir_catls/'

global survey
survey = 'eco'

phi_arr_smf = []
max_arr_smf = []
err_arr_smf = []
counts_arr_smf = []
phi_arr_bmf = []
max_arr_bmf = []
err_arr_bmf = []
counts_arr_bmf = []
box_id_arr = np.linspace(5001,5008,8)
for box in box_id_arr:
    box = int(box)
    path_to_mocks = '/Users/asadm2/Documents/Grad_School/Research/'\
        'Repositories/resolve_statistics/data/external/m200b/eco/{0}/'\
        'ECO_m200b_catls/'.format(box) 
    for num in range(8):
        filename = path_to_mocks + 'ECO_cat_{0}_Planck_memb_cat.hdf5'.format(num)
        mock_pd = reading_catls(filename) 

        #Using the same survey definition as in mcmc smf i.e excluding the buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= 3000) & \
            (mock_pd.cz.values <= 7000) & (mock_pd.M_r.values <= -17.33) &\
                (mock_pd.logmstar.values >= 8.9)]
        
        logmstar_arr = mock_pd.logmstar.values
        mhi_arr = mock_pd.mhi.values
        logmgas_arr = np.log10(1.4 * mhi_arr)
        logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))
        
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        #Measure SMF of mock using diff_smf function
        maxis, phi, err_poiss, bins, counts = \
            diff_smf(logmstar_arr, volume, 0, False)
        phi_arr_smf.append(phi)
        max_arr_smf.append(maxis)
        err_arr_smf.append(err_poiss)
        counts_arr_smf.append(counts)
        maxis, phi, err_poiss, bins, counts = \
            diff_bmf(logmbary_arr, volume, 0, False, False) 
        phi_arr_bmf.append(phi) 
        max_arr_bmf.append(maxis)
        err_arr_bmf.append(err_poiss)
        counts_arr_bmf.append(counts) 

phi_arr_smf = np.array(phi_arr_smf)
max_arr_smf = np.array(max_arr_smf)
err_arr_smf = np.array(err_arr_smf)
counts_arr_smf = np.array(counts_arr_smf)
phi_arr_bmf = np.array(phi_arr_bmf)
max_arr_bmf = np.array(max_arr_bmf)
err_arr_bmf = np.array(err_arr_bmf)
counts_arr_bmf = np.array(counts_arr_bmf)

columns = ['1', '2', '3', '4', '5', '6'] 
df = pd.DataFrame(phi_arr_smf, columns=columns)
df.index += 1

columns = ['1', '2', '3', '4', '5', '6'] 
df2 = pd.DataFrame(phi_arr_bmf, columns=columns)
df2.index += 1
###################### Covariance and correlation matrix #######################
N = len(phi_arr_smf)
cov_mat_eco = np.cov(phi_arr_smf.T, bias=True)*(N-1)
fig1 = plt.figure()
ax = sns.heatmap(cov_mat_eco, linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(df.corr(method='spearman'), cmap=cmap)
# Loop over data dimensions and create text annotations.
# for i in range(df.corr().shape[0]):
#     for j in range(df.corr().shape[1]):
#         text = ax1.text(j, i, np.round(df.corr().values[i, j],2),
#                        ha="center", va="center", color="w", size=10)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig2.colorbar(cax)
plt.title('ECO SMF correlation matrix')
plt.show()

fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
cmap = cm.get_cmap('Spectral')
cax = ax1.matshow(df2.corr(), cmap=cmap)
# Loop over data dimensions and create text annotations.
# for i in range(df2.corr().shape[0]):
#     for j in range(df2.corr().shape[1]):
#         text = ax1.text(j, i, np.round(df2.corr().values[i, j],2),
#                        ha="center", va="center", color="w", size=10)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig3.colorbar(cax)
plt.title('ECO BMF correlation matrix')
plt.show() 
######################### SMFs and BMFs from 64 ECO mocks ######################
if survey == 'eco':
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

catl, volume, cvar, z_median = read_data(catl_file, survey)
logmstar_arr = catl.logmstar.values
logmgas_arr = catl.logmgas.values
logmbary_arr = np.log10((10**logmstar_arr)+(10**logmgas_arr))
max_data_smf, phi_data_smf, err_data_smf, bins_data_smf, counts_smf = \
    diff_smf(logmstar_arr, volume, cvar, False)

err_data_smf = np.std(phi_arr_smf, axis=0)

max_data_bmf, phi_data_bmf, err_data_bmf, bins_data_bmf, counts_bmf = \
    diff_bmf(logmbary_arr, volume, 0, False, False)

err_data_bmf = np.std(phi_arr_bmf, axis=0)

cm = plt.get_cmap('Spectral')
n_catls = 64
col_arr = [cm(idx/float(n_catls)) for idx in range(n_catls)]
fig4 = plt.figure(figsize=(10,12))
ax1 = fig4.add_subplot(111)
for idx in range(len(phi_arr_smf)):
    plt.errorbar(max_arr_smf[idx],phi_arr_smf[idx],yerr=err_arr_smf[idx],
    color=col_arr[idx],markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_smf,phi_data_smf,yerr=err_data_smf,markersize=4,capsize=5,
capthick=2, label='data',color='k',fmt='-s',ecolor='k',zorder=10,linewidth=2)
for i in range(len(phi_data_smf)):
    text = ax1.text(max_data_smf[i], 10**-1.06, counts_smf[i],
    ha="center", va="center", color="k", size=15)
plt.xlabel(r'\boldmath $\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath $\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.ylim(10**-5,10**-0.9)
plt.title('ECO SMF')
plt.legend(loc='lower left', prop={'size': 20})
plt.show() 

fig5 = plt.figure(figsize=(10,12))
ax1 = fig5.add_subplot(111)
for idx in range(len(phi_arr_bmf)):
    plt.errorbar(max_arr_bmf[idx],phi_arr_bmf[idx],yerr=err_arr_bmf[idx],
    color=col_arr[idx],markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_bmf,phi_data_bmf,yerr=err_data_bmf,markersize=4,capsize=5,
capthick=2, label='data',color='k',fmt='-s',ecolor='k',zorder=10,linewidth=2)
for i in range(len(phi_data_bmf)):
    text = ax1.text(max_data_bmf[i], 10**-1.06, counts_bmf[i],
    ha="center", va="center", color="k", size=15)
plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.ylim(10**-5,10**-0.9)
plt.title('ECO BMF')
plt.legend(loc='lower left', prop={'size': 20})
plt.show() 

## Plot of luminosity function of all mocks
mag_cen_arr = []
mag_n_arr = []
mag_err_arr = []
for box in box_id_arr:
    box = int(box)
    path_to_mocks = '/Users/asadm2/Documents/Grad_School/Research/'\
        'Repositories/resolve_statistics/data/external/m200b/eco/{0}/'\
        'ECO_m200b_catls/'.format(box) 
    for num in range(8):
        filename = path_to_mocks + 'ECO_cat_{0}_Planck_memb_cat.hdf5'.format(num)
        mock_pd = reading_catls(filename) 

        #Using the same survey definition as in mcmc smf i.e excluding the buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= 3000) & \
            (mock_pd.cz.values <= 7000) & (mock_pd.M_r.values <= -17.33) &\
                (mock_pd.logmstar.values >= 8.9)]
        mag_cen, mag_edg, mag_n, mag_err, bw = cumu_num_dens(mock_pd.M_r.
            values, None, volume, True) 
        mag_cen_arr.append(mag_cen)
        mag_n_arr.append(mag_n)
        mag_err_arr.append(mag_err)

mag_cen_arr = np.array(mag_cen_arr)
mag_n_arr = np.array(mag_n_arr)
mag_err_arr = np.array(mag_err_arr)

mag_cen, mag_edg, mag_n, mag_err, bw = cumu_num_dens(catl.absrmag.
    values, None, volume, True)    

fig6 = plt.figure(figsize=(10,10))
for idx in range(n_catls):
    plt.errorbar(mag_cen_arr[idx],mag_n_arr[idx],yerr=mag_err_arr[idx],
    color=col_arr[idx],marker='o',markersize=4,capsize=5,capthick=0.5)
plt.errorbar(mag_cen,mag_n,yerr=mag_err,markersize=4,capsize=5,
    capthick=2, marker='o', label='data',color='k',fmt='-s',ecolor='k',
    linewidth=2,zorder=10)
plt.yscale('log')
plt.gca().invert_xaxis() 
plt.xlabel(r'\boldmath $M_{r}$')
plt.ylabel(r'\boldmath $\mathrm{(n < M_{r})} [\mathrm{h}^{3}\mathrm{Mpc}^{-3}]$')
plt.title(r'ECO Luminosity Function')
plt.legend(loc='lower left', prop={'size': 20})
plt.show()
'''
################################################################################
survey = 'resolvea'
phi_arr_smf = []
max_arr_smf = []
err_arr_smf = []
counts_arr_smf = []
phi_arr_bmf = []
max_arr_bmf = []
err_arr_bmf = []
counts_arr_bmf = []
for num in range(59):
    filename = path_to_resolvea + 'A_cat_{0}_Planck_memb_cat.hdf5'.format(num)
    mock_pd = reading_catls(filename) 

    #Using the same survey definition as in mcmc smf i.e excluding the buffer
    mock_pd = mock_pd.loc[(mock_pd.cz.values >= 4500) & \
        (mock_pd.cz.values <= 7000) & (mock_pd.M_r.values < -17.33) & \
                (mock_pd.logmstar.values >= 8.9)]
    
    logmstar_arr = mock_pd.logmstar.values
    mhi_arr = mock_pd.mhi.values
    logmgas_arr = np.log10(1.4 * mhi_arr)
    logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))

    # print("bary: ", logmbary_arr.max())
    # print("star: ", logmstar_arr.max())

    volume = 13172.384 # Survey volume without buffer [Mpc/h]^3
    #Measure SMF of mock using diff_smf function
    maxis, phi, err_poiss, bins, counts = \
        diff_smf(logmstar_arr, volume, 0, False)
    phi_arr_smf.append(phi)
    max_arr_smf.append(maxis)
    err_arr_smf.append(err_poiss)
    counts_arr_smf.append(counts)
    maxis, phi, err_poiss, bins, counts = \
        diff_bmf(logmbary_arr, volume, 0, False, False) 
    phi_arr_bmf.append(phi) 
    max_arr_bmf.append(maxis)
    err_arr_bmf.append(err_poiss)
    counts_arr_bmf.append(counts) 
    
phi_arr_smf = np.array(phi_arr_smf)
max_arr_smf = np.array(max_arr_smf)
err_arr_smf = np.array(err_arr_smf)
counts_arr_smf = np.array(counts_arr_smf)
phi_arr_bmf = np.array(phi_arr_bmf)
max_arr_bmf = np.array(max_arr_bmf)
err_arr_bmf = np.array(err_arr_bmf)
counts_arr_bmf = np.array(counts_arr_bmf)


columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'] 
df3 = pd.DataFrame(phi_arr_smf, columns=columns)
df3.index += 1

columns = ['1', '2', '3', '4', '5', '6', '7', '8'] 
df4 = pd.DataFrame(phi_arr_bmf, columns=columns)
df4.index += 1

cov_mat_resolvea = np.cov(phi_arr_smf)
fig6 = plt.figure()
ax = sns.heatmap(cov_mat_resolvea, linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

fig7 = plt.figure()
ax1 = fig7.add_subplot(111)
cmap = cm.get_cmap('rainbow')
cax = ax1.matshow(df3.corr(), cmap=cmap)
# Loop over data dimensions and create text annotations.
for i in range(df3.corr().shape[0]):
    for j in range(df3.corr().shape[1]):
        text = ax1.text(j, i, np.round(df3.corr().values[i, j],2),
                       ha="center", va="center", color="w", size=7)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig7.colorbar(cax)
plt.title('RESOLVE-A SMF correlation matrix')
plt.show() 

fig8 = plt.figure()
ax1 = fig8.add_subplot(111)
cmap = cm.get_cmap('rainbow')
cax = ax1.matshow(df4.corr(), cmap=cmap)
# Loop over data dimensions and create text annotations.
for i in range(df4.corr().shape[0]):
    for j in range(df4.corr().shape[1]):
        text = ax1.text(j, i, np.round(df4.corr().values[i, j],2),
                       ha="center", va="center", color="w", size=7)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig8.colorbar(cax)
plt.title('RESOLVE-A BMF correlation matrix')
plt.show() 
##################### SMFs and BMFs from 59 RESOLVE-A mocks ####################
if survey == 'eco':
    catl_file = path_to_raw + "eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

catl, volume, cvar, z_median = read_data(catl_file, survey)
logmstar_arr = catl.logmstar.values
logmgas_arr = catl.logmgas.values
logmbary_arr = np.log10((10**logmstar_arr)+(10**logmgas_arr))
max_data_smf, phi_data_smf, err_data_smf, bins_data_smf, counts_smf = \
    diff_smf(logmstar_arr, volume, cvar, False)

max_data_bmf, phi_data_bmf, err_data_bmf, bins_data_bmf, counts_bmf = \
    diff_bmf(logmbary_arr, volume, 0, False, False)

fig9 = plt.figure()
ax1 = fig9.add_subplot(111)
for idx in range(len(phi_arr_smf)):
    plt.errorbar(max_arr_smf[idx],phi_arr_smf[idx],yerr=err_arr_smf[idx],
    markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_smf,phi_data_smf,yerr=err_data_smf,markersize=4,capsize=5,
capthick=0.5, label='data',color='k',fmt='-s',ecolor='k')
for i in range(len(phi_data_smf)):
    text = ax1.text(max_data_smf[i], 10**-0.9, counts_smf[i],
    ha="center", va="center", color="k", size=7)
plt.xlabel(r'\boldmath $\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath $\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.title('RESOLVE A SMF')
plt.legend(loc='lower left', prop={'size': 10})
plt.show() 

fig10 = plt.figure()
ax1 = fig10.add_subplot(111)
for idx in range(len(phi_arr_bmf)):
    plt.errorbar(max_arr_bmf[idx],phi_arr_bmf[idx],yerr=err_arr_bmf[idx],
    markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_bmf,phi_data_bmf,yerr=err_data_bmf,markersize=4,capsize=5,
capthick=0.5, label='data',color='k',fmt='-s',ecolor='k')
for i in range(len(phi_data_bmf)):
    text = ax1.text(max_data_bmf[i], 10**-0.85, counts_bmf[i],
    ha="center", va="center", color="k", size=7)
plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.title('RESOLVE A BMF')
plt.legend(loc='lower left', prop={'size': 10})
plt.show()
################################################################################
survey = 'resolveb'
phi_arr_smf = []
max_arr_smf = []
err_arr_smf = []
counts_arr_smf = []
phi_arr_bmf = []
max_arr_bmf = []
err_arr_bmf = []
counts_arr_bmf = []
for num in range(104):
    filename = path_to_resolveb + 'B_cat_{0}_Planck_memb_cat.hdf5'.format(num)
    mock_pd = reading_catls(filename) 

    #Using the same survey definition as in mcmc smf i.e excluding the buffer
    mock_pd = mock_pd.loc[(mock_pd.cz.values >= 4500) & \
        (mock_pd.cz.values <= 7000) & (mock_pd.M_r.values < -17) & \
                (mock_pd.logmstar.values >= 8.7)]
    
    logmstar_arr = mock_pd.logmstar.values
    mhi_arr = mock_pd.mhi.values
    logmgas_arr = np.log10(1.4 * mhi_arr)
    logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))

    # print("bary: ", logmbary_arr.max())
    # print("star: ", logmstar_arr.max())

    volume = 4709.8373 # Survey volume without buffer [Mpc/h]^3
    #Measure SMF of mock using diff_smf function
    maxis, phi, err_poiss, bins, counts = \
        diff_smf(logmstar_arr, volume, 0, False)
    phi_arr_smf.append(phi)
    max_arr_smf.append(maxis)
    err_arr_smf.append(err_poiss)
    counts_arr_smf.append(counts)
    maxis, phi, err_poiss, bins, counts = \
        diff_bmf(logmbary_arr, volume, 0, False, False) 
    phi_arr_bmf.append(phi) 
    max_arr_bmf.append(maxis)
    err_arr_bmf.append(err_poiss)
    counts_arr_bmf.append(counts) 
    
phi_arr_smf = np.array(phi_arr_smf)
max_arr_smf = np.array(max_arr_smf)
err_arr_smf = np.array(err_arr_smf)
counts_arr_smf = np.array(counts_arr_smf)
phi_arr_bmf = np.array(phi_arr_bmf)
max_arr_bmf = np.array(max_arr_bmf)
err_arr_bmf = np.array(err_arr_bmf)
counts_arr_bmf = np.array(counts_arr_bmf)


columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'] 
df5 = pd.DataFrame(phi_arr_smf, columns=columns)
df5.index += 1

columns = ['1', '2', '3', '4', '5', '6', '7', '8'] 
df6 = pd.DataFrame(phi_arr_bmf, columns=columns)
df6.index += 1

cov_mat_resolveb = np.cov(phi_arr_smf)
fig11 = plt.figure()
ax = sns.heatmap(cov_mat_resolveb, linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

fig12 = plt.figure()
ax1 = fig12.add_subplot(111)
cmap = cm.get_cmap('rainbow')
cax = ax1.matshow(df5.corr(), cmap=cmap)
# Loop over data dimensions and create text annotations.
for i in range(df5.corr().shape[0]):
    for j in range(df5.corr().shape[1]):
        text = ax1.text(j, i, np.round(df5.corr().values[i, j],2),
                       ha="center", va="center", color="w", size=7)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig12.colorbar(cax)
plt.title('RESOLVE-B SMF correlation matrix')
plt.show() 

fig13 = plt.figure()
ax1 = fig13.add_subplot(111)
cmap = cm.get_cmap('rainbow')
cax = ax1.matshow(df6.corr(), cmap=cmap)
# Loop over data dimensions and create text annotations.
for i in range(df6.corr().shape[0]):
    for j in range(df6.corr().shape[1]):
        text = ax1.text(j, i, np.round(df6.corr().values[i, j],2),
                       ha="center", va="center", color="w", size=7)
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig13.colorbar(cax)
plt.title('RESOLVE-B BMF correlation matrix')
plt.show() 
##################### SMFs and BMFs from 104 RESOLVE-B mocks ###################
if survey == 'eco':
    catl_file = path_to_raw + "eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

catl, volume, cvar, z_median = read_data(catl_file, survey)
logmstar_arr = catl.logmstar.values
logmgas_arr = catl.logmgas.values
logmbary_arr = np.log10((10**logmstar_arr)+(10**logmgas_arr))
max_data_smf, phi_data_smf, err_data_smf, bins_data_smf, counts_smf = \
    diff_smf(logmstar_arr, volume, cvar, False)

max_data_bmf, phi_data_bmf, err_data_bmf, bins_data_bmf, counts_bmf = \
    diff_bmf(logmbary_arr, volume, 0, False, False)

fig14 = plt.figure()
ax1 = fig14.add_subplot(111)
for idx in range(len(phi_arr_smf)):
    plt.errorbar(max_arr_smf[idx],phi_arr_smf[idx],yerr=err_arr_smf[idx],
    markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_smf,phi_data_smf,yerr=err_data_smf,markersize=4,capsize=5,
capthick=0.5, label='data',color='k',fmt='-s',ecolor='k')
for i in range(len(phi_data_smf)):
    text = ax1.text(max_data_smf[i], 10**-0.8, counts_smf[i],
    ha="center", va="center", color="k", size=7)
plt.xlabel(r'\boldmath $\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath $\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.title('RESOLVE B SMF')
plt.legend(loc='lower left', prop={'size': 10})
plt.show() 

fig15 = plt.figure()
ax1 = fig15.add_subplot(111)
for idx in range(len(phi_arr_bmf)):
    plt.errorbar(max_arr_bmf[idx],phi_arr_bmf[idx],yerr=err_arr_bmf[idx],
    markersize=4,capsize=5,capthick=0.5)
plt.errorbar(max_data_bmf,phi_data_bmf,yerr=err_data_bmf,markersize=4,capsize=5,
capthick=0.5, label='data',color='k',fmt='-s',ecolor='k')
for i in range(len(phi_data_bmf)):
    text = ax1.text(max_data_bmf[i], 10**-0.73, counts_bmf[i],
    ha="center", va="center", color="k", size=7)
plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
plt.yscale('log')
plt.title('RESOLVE B BMF')
plt.legend(loc='lower left', prop={'size': 10})
plt.show()
################################################################################
########## Comparison between cosmic variance error and poisson error ##########

# ECO
survey = 'eco'
data_file = path_to_raw + "eco_all.csv"
catl, volume, cvar, z_median = read_data(data_file, survey)
stellar_mass_arr = catl.logmstar.values
#Measure SMF of data using diff_smf function
maxis_data, phi_data, err_poiss, bins_data = \
    diff_smf(stellar_mass_arr, volume, 0, False)

print("ECO smf \n")
print("poisson: \n" , err_poiss, "\n\n", "cvar: \n", cvar*phi_data, "\n")

# RESOLVE
survey = 'resolvea'
data_file = path_to_raw + "RESOLVE_liveJune2018.csv"
catl, volume, cvar, z_median = read_data(data_file, survey)
stellar_mass_arr = catl.logmstar.values
#Measure SMF of data using diff_smf function
maxis_data, phi_data, err_poiss, bins_data = \
    diff_smf(stellar_mass_arr, volume, 0, False)

print("RESOLVE-A smf \n")
print("poisson: \n" , err_poiss, "\n\n", "cvar: \n", cvar*phi_data, "\n")

survey = 'resolveb'
data_file = path_to_raw + "RESOLVE_liveJune2018.csv"
catl, volume, cvar, z_median = read_data(data_file, survey)
stellar_mass_arr = catl.logmstar.values
#Measure SMF of data using diff_smf function
maxis_data, phi_data, err_poiss, bins_data = \
    diff_smf(stellar_mass_arr, volume, 0, False)

print("RESOLVE-B smf \n")
print("poisson: \n" , err_poiss, "\n\n", "cvar: \n", cvar*phi_data)

################################################################################
'''