"""
{This script calculates cosmic variance from mocks and plots SMF for red and 
 blue galaxies from mocks}
"""

# Libs
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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=10)
rc('text', usetex=True)

def survey_definitions():
    
    eco = {
        "mock_name": "ECO",
        "num_mocks": 8,
        "min_cz": 3000,
        "max_cz": 7000,
        "mag_limit": -17.33,
        "mstar_limit": 8.9,
        "volume": 151829.26 # Survey volume without buffer [Mpc/h]^3
    }

    resolvea = {
        "mock_name": "A",
        "num_mocks": 59,
        "min_cz": 4500,
        "max_cz": 7000,
        "mag_limit": -17.33,
        "mstar_limit": 8.9,
        "volume": 13172.384 # Survey volume without buffer [Mpc/h]^3
    }

    resolveb = {
        "mock_name": "B",
        "num_mocks": 104,
        "min_cz": 4500,
        "max_cz": 7000,
        "mag_limit": -17,
        "mstar_limit": 8.7,
        "volume": 4709.8373 # Survey volume without buffer [Mpc/h]^3
    }

    if survey == 'eco':
        return eco
    elif survey == 'resolvea':
        return resolvea
    elif survey == 'resolveb':
        return resolveb
    
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
                    'fc', 'grpmb', 'grpms', 'modelu_rcorr', 'umag', 'rmag']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
            (eco_buff.grpcz.values <= 7000) & 
            (eco_buff.absrmag.values <= -17.33) &
            (eco_buff.logmstar.values >= 8.9)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b', 
                    'modelu_rcorr']
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
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 12)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 12)   

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
    
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 9)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(mass_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)

    phi = counts / (volume * dm)  # not a log quantity
    return maxis, phi, err_poiss, bins, counts

def measure_mock_total_mf(path):
    """
    Calculate error in data SMF from mocks

    Parameters
    ----------
    path: string
        Path to mock catalogs

    Returns
    ---------
    err_total: array
        Standard deviation of phi values between all mocks and for all galaxies
    err_red: array
        Standard deviation of phi values between all mocks and for red galaxies
    err_blue: array
        Standard deviation of phi values between all mocks and for blue galaxies
    """
    phi_smf_arr = []
    phi_bmf_arr = []
    max_smf_arr = []
    max_bmf_arr = []
    err_smf_arr = []
    err_bmf_arr = []

    for num in range(survey_dict["num_mocks"]):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            survey_dict["mock_name"], num)
        mock_pd = reading_catls(filename) 

        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= survey_dict["min_cz"]) & \
            (mock_pd.cz.values <= survey_dict["max_cz"]) & \
            (mock_pd.M_r.values <= survey_dict["mag_limit"]) & \
            (mock_pd.logmstar.values >= survey_dict["mstar_limit"])]

        # Convert mhI and logmstar masses to h=1.0 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mock_pd.logmstar.values) / 2.041)
        logmhi_arr = mock_pd.mhi.values / 2.041
        logmgas_arr = np.log10(1.4 * logmhi_arr)
        logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))

        # Measure SMF of mock using diff_smf function
        # Volume in h=1 since stellar masses have been changed to be in h=1
        max_smf, phi_smf, err_smf, bins_smf, counts_smf = \
            diff_smf(logmstar_arr, survey_dict["volume"], 0, True)
        max_bmf, phi_bmf, err_bmf, bins_bmf, counts_bmf = \
            diff_bmf(logmbary_arr, survey_dict["volume"], 0, False, True)

        phi_smf_arr.append(np.log10(phi_smf))
        phi_bmf_arr.append(np.log10(phi_bmf))
        max_smf_arr.append(np.log10(max_smf))
        max_bmf_arr.append(np.log10(max_bmf))
        err_smf_arr.append(np.log10(err_smf))
        err_bmf_arr.append(np.log10(err_bmf))
    
    if mf_type == 'smf':
        return max_smf_arr, phi_smf_arr, err_smf_arr

    elif mf_type == 'bmf':
        return max_bmf_arr, phi_bmf_arr, err_bmf_arr

def measure_mock_colour_mf(path):
    """
    Assign colour label to data

    Parameters
    ----------
    path: string
        Path to mock catalogs

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """
    cols = survey_dict["num_mocks"]
    rows = 3 # store mass, phi, error measurements from SMF
    len_arr = 5 # number of bins in SMF
    smf_red = [[[0 for k in range(len_arr)] for j in range(cols)] 
        for i in range(rows)]
    smf_blue = [[[0 for k in range(len_arr)] for j in range(cols)] 
        for i in range(rows)]

    for num in range(survey_dict["num_mocks"]):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            survey_dict["mock_name"], num)
        mock_pd = reading_catls(filename) 

        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= survey_dict["min_cz"]) & \
            (mock_pd.cz.values <= survey_dict["max_cz"]) & \
            (mock_pd.M_r.values <= survey_dict["mag_limit"]) & \
            (mock_pd.logmstar.values >= survey_dict["mstar_limit"])]

        logmstar_arr = mock_pd.logmstar.values
        u_r_arr = mock_pd.u_r.values

        colour_label_arr = np.empty(len(mock_pd), dtype='str')
        for idx, value in enumerate(logmstar_arr):
            # Divisions taken from Moffett et al. 2015 equation 1
            if value <= 9.1:
                if u_r_arr[idx] > 1.457:
                    colour_label = 'R'
                else:
                    colour_label = 'B'

            elif value > 9.1 and value < 10.1:
                divider = 0.24 * value - 0.7
                if u_r_arr[idx] > divider:
                    colour_label = 'R'
                else:
                    colour_label = 'B'

            elif value >= 10.1:
                if u_r_arr[idx] > 1.7:
                    colour_label = 'R'
                else:
                    colour_label = 'B'
                
            colour_label_arr[idx] = colour_label
        
        mock_pd['colour_label'] = colour_label_arr

        max_red, phi_red, err_red, bins_red, counts_red = \
            diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
                survey_dict["volume"], 0, False)
        max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'], 
                survey_dict["volume"], 0, False)

        smf_red[0][num] = max_red
        smf_red[1][num] = phi_red
        smf_red[2][num] = err_red
        smf_blue[0][num] = max_blue
        smf_blue[1][num] = phi_blue
        smf_blue[2][num] = err_blue       

    return smf_red, smf_blue

def measure_cov_mat(mf_array):

    cov_mat = np.cov(mf_array.T)
    print(cov_mat.shape)
    fig1 = plt.figure()
    ax = sns.heatmap(cov_mat, linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.show()

def measure_corr_mat(mf_array):

    if mf_type == 'smf':
        columns = ['1', '2', '3', '4', '5']#, '6', '7', '8', '9', '10', '11'] 
    elif mf_type == 'bmf':
        columns = ['1', '2', '3', '4', '5', '6', '7', '8']

    df = pd.DataFrame(mf_array, columns=columns)
    df.index += 1

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)
    cmap = cm.get_cmap('rainbow')
    cax = ax1.matshow(df.corr(), cmap=cmap)
    # Loop over data dimensions and create text annotations.
    for i in range(df.corr().shape[0]):
        for j in range(df.corr().shape[1]):
            text = ax1.text(j, i, np.round(df.corr().values[i, j],2),
                        ha="center", va="center", color="w", size='10')
    plt.gca().invert_yaxis() 
    plt.gca().xaxis.tick_bottom()
    fig2.colorbar(cax)
    if survey == 'eco':
        if mf_type == 'smf':
            plt.title('ECO SMF')
        elif mf_type == 'bmf':
            plt.title('ECO BMF')
    elif survey == 'resolvea':
        if mf_type == 'smf':
            plt.title('RESOLVE-A SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-A BMF')
    elif survey == 'resolveb':
        if mf_type == 'smf':
            plt.title('RESOLVE-B SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-B BMF')
    plt.show()

def plot_total_mf(max_1, phi_1, err_1, max_2, phi_2, err_2, 
    counts_2):

    fig4 = plt.figure()
    for idx in range(len(phi_1)):
        plt.errorbar(max_1[idx],phi_1[idx],yerr=err_1[idx],
        markersize=4,capsize=5,capthick=0.5)
    for i in range(len(phi_2)):
        text = plt.text(max_2[i], 10**-1.07, counts_2[i],
        ha="center", va="center", color="k", size=7)
    plt.errorbar(max_2,phi_2,yerr=err_2,markersize=4,capsize=5, capthick=0.5, 
        label='data',color='k',fmt='-s',ecolor='k')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
    plt.yscale('log')
    
    if survey == 'eco':
        if mf_type == 'smf':
            plt.title('ECO SMF')
        elif mf_type == 'bmf':
            plt.title('ECO BMF')
    elif survey == 'resolvea':
        if mf_type == 'smf':
            plt.title('RESOLVE-A SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-A BMF')
    elif survey == 'resolveb':
        if mf_type == 'smf':
            plt.title('RESOLVE-B SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-B BMF')

    plt.legend(loc='best', prop={'size': 10})
    plt.show() 

def plot_colour_mf(smf_1, smf_2):

    fig5 = plt.figure()
    for idx in range(len(smf_1[0])):
        plt.errorbar(smf_1[0][idx],smf_1[1][idx],yerr=smf_1[2][idx],
        markersize=4,capsize=5,capthick=0.5,color='r')
    for idx in range(len(smf_2[0])):
        plt.errorbar(smf_2[0][idx],smf_2[1][idx],yerr=smf_2[2][idx],
        markersize=4,capsize=5,capthick=0.5,color='b')
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-3} \right]$')
    plt.yscale('log')
    
    if survey == 'eco':
        if mf_type == 'smf':
            plt.title('ECO SMF')
        elif mf_type == 'bmf':
            plt.title('ECO BMF')
    elif survey == 'resolvea':
        if mf_type == 'smf':
            plt.title('RESOLVE-A SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-A BMF')
    elif survey == 'resolveb':
        if mf_type == 'smf':
            plt.title('RESOLVE-B SMF')
        if mf_type == 'bmf':
            plt.title('RESOLVE-B BMF')

    plt.show()

def cosmic_vs_poisson(catl_file):

    catl, volume, cvar, z_median = read_data(catl_file, survey)
    stellar_mass_arr = catl.logmstar.values
    # Measure SMF of data using diff_smf function
    maxis_data, phi_data, err_data, bins_data, counts_data = \
        diff_smf(stellar_mass_arr, volume, 0, False)

    print("poisson: \n" , err_data, "\n\n", "cvar: \n", cvar*phi_data)

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
    parser.add_argument('survey', type=str, 
        help='Options: eco/resolvea/resolveb')
    parser.add_argument('mf_type', type=str,
        help='Options: smf/bmf')
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
    global survey
    global mf_type
    global survey_dict
    
    survey = args.survey
    mf_type = args.mf_type
    survey_dict = survey_definitions()

    # Paths
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_external = dict_of_paths['ext_dir']

    if survey == 'eco':
        path_to_mocks = path_to_external + 'ECO_mvir_catls/'
    elif survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    elif survey == 'resolveb':
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

    if survey == 'eco':
        catl_file = path_to_raw + "eco_all.csv"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"


    catl, volume, cvar, z_median = read_data(catl_file, survey)
    stellar_mass_arr = np.log10((10**catl.logmstar.values) / 2.041)
    gas_mass_arr = np.log10((10**catl.logmgas.values) / 2.041)
    bary_mass_arr = np.log10(10**(stellar_mass_arr) + 10**(gas_mass_arr))

    print("Measuring mass function for data")
    if mf_type == 'smf':
        max_data, phi_data, err_data, bins_data, counts_data = \
            diff_smf(stellar_mass_arr, volume, cvar, True)
    elif mf_type == 'bmf':
        max_data, phi_data, err_data, bins_data, counts_data = \
            diff_bmf(bary_mass_arr, volume, cvar, False, True)

    print("Measuring total mass function for mock")   
    max_mock, phi_mock, err_mock = measure_mock_total_mf(path_to_mocks)

    print("Measuring colour mass function")
    smf_red, smf_blue  = measure_mock_colour_mf(path_to_mocks)

    print("Measuring covariance matrix")
    measure_cov_mat(phi_mock)

    print("Measuring correlation matrix")
    measure_corr_mat(phi_mock)

    print("Plotting total mass functions for mocks and data")
    plot_total_mf(max_mock, phi_mock, err_mock, max_data, phi_data, err_data, 
        counts_data)
    
    print("Plotting colour mass functions for mocks")
    plot_colour_mf(smf_red, smf_blue)
    
    print("Comparing cosmic and poisson error for {0}".format(survey))
    cosmic_vs_poisson(catl_file)

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args) 
