"""
{This script carries out the main checks for mocks : luminosity function, 
 number density, SMF & BMF and correlation matrices. First 3 are compared with
 data.}
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
import argparse
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

def diff_smf(mstar_arr, volume, h1_bool):
    """
    Calculates differential stellar mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

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

def diff_bmf(mass_arr, volume, sim_bool, h1_bool):
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

def cumu_num_dens(data, weights, volume, bool_mag): 
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

def measure_corr_mat(path_to_mocks):

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
        temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
            temp_dict.get('mock_name')) 
        for num in range(temp_dict.get('num_mocks')):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.\
                format(temp_dict.get('mock_name'), num)
            mock_pd = reading_catls(filename) 
            if num == 0:
                print("cz min: ", mock_pd.cz.min())
                print("cz max: ",mock_pd.cz.max())
            #Using the same survey definition as in mcmc smf i.e excluding 
            # the buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= temp_dict.get('min_cz'))
                & (mock_pd.cz.values <= temp_dict.get('max_cz')) & 
                (mock_pd.M_r.values <= temp_dict.get('mag_limit')) &
                (mock_pd.logmstar.values >= temp_dict.get('mstar_limit'))]
            
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))
            
            volume =  temp_dict.get('volume') 
            #Measure SMF of mock using diff_smf function
            maxis, phi, err_poiss, bins, counts = \
                diff_smf(logmstar_arr, volume, False)
            phi_arr_smf.append(phi)
            max_arr_smf.append(maxis)
            err_arr_smf.append(err_poiss)
            counts_arr_smf.append(counts)
            maxis, phi, err_poiss, bins, counts = \
                diff_bmf(logmbary_arr, volume, False, False) 
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

    N = len(phi_arr_smf)
    cov_mat_eco = np.cov(phi_arr_smf, rowvar=False)
    fig1 = plt.figure()
    ax = sns.heatmap(cov_mat_eco, linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.show()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(df.corr(), cmap=cmap)
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig2.colorbar(cax)
    if survey == 'resolvea' or survey == 'resolveb':
        plt.title(r'RESOLVE-{0} SMF correlation matrix'.\
            format(temp_dict.get('mock_name')))
    else:
        plt.title(r'{0} SMF correlation matrix'.format(temp_dict.get('mock_name')))
    plt.show()

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(111)
    cmap = cm.get_cmap('Spectral')
    cax = ax1.matshow(df2.corr(), cmap=cmap)
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig3.colorbar(cax)
    if survey == 'resolvea' or survey == 'resolveb':
        plt.title(r'RESOLVE-{0} BMF correlation matrix'.\
            format(temp_dict.get('mock_name')))
    else:
        plt.title(r'{0} BMF correlation matrix'.format(temp_dict.get('mock_name')))
    plt.show() 

    return [max_arr_smf, phi_arr_smf, err_arr_smf], [max_arr_bmf, phi_arr_bmf, 
    err_arr_bmf]

def measure_mass_funcs(smf, bmf, catl):
    
    box_id_arr = np.linspace(5001,5008,8)
    volume =  temp_dict.get('volume') 

    max_arr_smf, phi_arr_smf, err_arr_smf = smf[0], smf[1], smf[2]
    max_arr_bmf, phi_arr_bmf, err_arr_bmf = bmf[0], bmf[1], bmf[2]
    logmstar_arr = catl.logmstar.values
    logmgas_arr = catl.logmgas.values
    logmbary_arr = np.log10((10**logmstar_arr)+(10**logmgas_arr))
    max_data_smf, phi_data_smf, err_data_smf, bins_data_smf, counts_smf = \
        diff_smf(logmstar_arr, volume, False)

    err_data_smf = np.std(phi_arr_smf, axis=0)

    max_data_bmf, phi_data_bmf, err_data_bmf, bins_data_bmf, counts_bmf = \
        diff_bmf(logmbary_arr, volume, False, False)

    err_data_bmf = np.std(phi_arr_bmf, axis=0)

    cm = plt.get_cmap('Spectral')
    n_catls = temp_dict.get('num_mocks')*len(box_id_arr)
    col_arr = [cm(idx/float(n_catls)) for idx in range(n_catls)]
    fig4 = plt.figure(figsize=(10,12))
    ax1 = fig4.add_subplot(111)
    for idx in range(len(phi_arr_smf)):
        plt.errorbar(max_arr_smf[idx],phi_arr_smf[idx],yerr=err_arr_smf[idx],
        color=col_arr[idx],markersize=4,capsize=5,capthick=0.5)
    plt.errorbar(max_data_smf,phi_data_smf,yerr=err_data_smf,markersize=4,
        capsize=5,capthick=2, label='data',color='k',fmt='-s',ecolor='k',
        zorder=10,linewidth=2)
    for i in range(len(phi_data_smf)):
        text = ax1.text(max_data_smf[i], ax1.get_ylim()[1]+0.01, counts_smf[i],
        ha="center", va="center", color="k", size=15)
    plt.xlabel(r'\boldmath $\log_{10}\ M_\star \left[\mathrm{M_\odot}\, '
        r'\mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath $\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,'
        r'\mathrm{h}^{-3} \right]$')
    plt.yscale('log')
    # plt.ylim(10**-5,10**-0.9)
    if survey == 'resolvea' or survey == 'resolveb':
        plt.title(r'RESOLVE-{0} SMF'.format(temp_dict.get('mock_name')))
    else:
        plt.title(r'{0} SMF'.format(temp_dict.get('mock_name')))
    plt.legend(loc='lower left', prop={'size': 20})
    plt.show() 

    fig5 = plt.figure(figsize=(10,12))
    ax1 = fig5.add_subplot(111)
    for idx in range(len(phi_arr_bmf)):
        plt.errorbar(max_arr_bmf[idx],phi_arr_bmf[idx],yerr=err_arr_bmf[idx],
        color=col_arr[idx],markersize=4,capsize=5,capthick=0.5)
    plt.errorbar(max_data_bmf,phi_data_bmf,yerr=err_data_bmf,markersize=4,
        capsize=5,capthick=2, label='data',color='k',fmt='-s',ecolor='k',
        zorder=10,linewidth=2)
    for i in range(len(phi_data_bmf)):
        text = ax1.text(max_data_bmf[i], ax1.get_ylim()[1]+0.01, counts_bmf[i],
        ha="center", va="center", color="k", size=15)
    plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, '
        r'\mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,'
        r'\mathrm{h}^{-3} \right]$')
    plt.yscale('log')
    # plt.ylim(10**-5,10**-0.9)
    if survey == 'resolvea' or survey == 'resolveb':
        plt.title(r'RESOLVE-{0} BMF'.format(temp_dict.get('mock_name')))
    else:
        plt.title(r'{0} BMF'.format(temp_dict.get('mock_name')))
    plt.legend(loc='lower left', prop={'size': 20})
    plt.show() 

def measure_lum_funcs(catl, path_to_mocks):

    volume =  temp_dict.get('volume') 

    mag_cen_arr = []
    mag_n_arr = []
    mag_err_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
            temp_dict.get('mock_name')) 
        for num in range(temp_dict.get('num_mocks')):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.\
                format(temp_dict.get('mock_name'), num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf 
            # i.e excluding the buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= temp_dict.get('min_cz')) 
                & (mock_pd.cz.values <= temp_dict.get('max_cz')) & 
                (mock_pd.M_r.values <= temp_dict.get('mag_limit')) &
                (mock_pd.logmstar.values >= temp_dict.get('mstar_limit'))]
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

    cm = plt.get_cmap('Spectral')
    n_catls = temp_dict.get('num_mocks')*len(box_id_arr)
    col_arr = [cm(idx/float(n_catls)) for idx in range(n_catls)]
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
    plt.ylabel(r'\boldmath $\mathrm{(n < M_{r})} [\mathrm{h}^{3}'
        r'\mathrm{Mpc}^{-3}]$')
    if survey == 'resolvea' or survey == 'resolveb':
        plt.title(r'RESOLVE-{0} Luminosity Function'.\
            format(temp_dict.get('mock_name')))
    else:
        plt.title(r'{0} Luminosity Function'.format(temp_dict.get('mock_name')))
    plt.legend(loc='lower left', prop={'size': 20})
    plt.show()

def measure_num_dens(catl, path_to_mocks):

    volume =  temp_dict.get('volume') 

    num_arr = [] 
    box_arr = [] 
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr: 
        box = int(box) 
        temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
            temp_dict.get('mock_name'))
        num_arr_temp = [] 
        box_arr_temp = [] 
        for num in range(temp_dict.get('num_mocks')): 
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format( 
                    temp_dict.get('mock_name'), num)  
            mock_pd = reading_catls(filename) 
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= temp_dict.get('min_cz'))
                & (mock_pd.cz.values <= temp_dict.get('max_cz')) & 
                (mock_pd.M_r.values <= temp_dict.get('mag_limit')) & 
                (mock_pd.logmstar.values >= temp_dict.get('mstar_limit'))] 
            num_arr_temp.append(len(mock_pd)) 
            box_arr_temp.append(box) 
        box_arr.append(box_arr_temp) 
        num_arr.append(num_arr_temp) 
    num_arr = np.array(num_arr)

    num_dens_arr = num_arr/volume

    ms = ['o','^','s','d','<','>','v','p'] 
    colors = ['salmon', 'orangered','darkorange','mediumorchid','cornflowerblue',
        'gold', 'mediumslateblue','mediumturquoise'] 
    fig7 = plt.figure(figsize=(10,8)) 
    for i in np.linspace(0,7,8): 
        i = int(i) 
        plt.scatter(box_arr[i],num_dens_arr[i],marker=ms[i],
            edgecolors=colors[i],facecolors='none',s=150, linewidths=2) 
    plt.scatter(np.linspace(5001,5008,8),8*[len(catl)/volume],marker='*',
        edgecolors='k',facecolors='none',s=200,linewidths=2,
        label=r'{0}'.format(temp_dict.get('mock_name'))) 
    plt.ylabel(r'\boldmath $n\ {[Mpc/h]}^{-3}$', fontsize=20)
    plt.legend(loc='best', prop={'size': 20})
    plt.show()

def args_parser():
    """
    Parsing arguments passed to script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('survey', type=str, \
        help='Options: eco/resolvea/resolveb')
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function that calls all other functions
    
    Parameters
    ----------
    args: 
        Input arguments to the script

    """
    # Paths
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_data = dict_of_paths['data_dir']
    path_to_raw = dict_of_paths['raw_dir']

    global survey
    global temp_dict
    survey = args.survey

    # Path to mocks
    if survey == 'eco':
        path_to_mocks = path_to_data + 'mocks/m200b/eco/'
        catl_file = path_to_raw + 'eco/eco_all.csv'
    elif survey == 'resolvea':
        path_to_mocks = path_to_data + 'mocks/m200b/resolve_a/'
        catl_file = path_to_raw + 'resolve/RESOLVE_liveJune2018.csv'
    elif survey == 'resolveb':  
        path_to_mocks = path_to_data + 'mocks/m200b/resolve_b/'
        catl_file = path_to_raw + 'resolve/RESOLVE_liveJune2018.csv'

    # Survey definition dictionaries - all without buffer
    eco = {
        'mock_name' : 'ECO',
        'num_mocks' : 8,
        'min_cz' : 3000,
        'max_cz' : 7000,
        'mag_limit' : -17.33,
        'mstar_limit' : 8.9,
        'volume' : 151829.26 #[Mpc/h]^3
    }   

    resolvea = {
        'mock_name' : 'A',
        'num_mocks' : 59,
        'min_cz' : 4500,
        'max_cz' : 7000,
        'mag_limit' : -17.33,
        'mstar_limit' : 8.9,
        'volume' : 13172.384  #[Mpc/h]^3 
    }

    resolveb = {
        'mock_name' : 'B',
        'num_mocks' : 104,
        'min_cz' : 4500,
        'max_cz' : 7000,
        'mag_limit' : -17,
        'mstar_limit' : 8.7,
        'volume' : 4709.8373  #[Mpc/h]^3
    }

    # Changes string name of survey to variable so that the survey dict can 
    # be accessed
    temp_dict = vars()[survey]

    catl, volume, cvar, z_median = read_data(catl_file, survey)
    smf, bmf = measure_corr_mat(path_to_mocks)
    measure_mass_funcs(smf, bmf, catl)
    measure_lum_funcs(catl, path_to_mocks)
    measure_num_dens(catl, path_to_mocks)

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args)