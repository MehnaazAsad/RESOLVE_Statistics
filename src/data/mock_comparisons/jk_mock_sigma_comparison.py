"""
{This temp script compares sigma from std of phi from mocks with that from 
 the sqrt(diagonals of covariance matrix) from the jackknife of ECO survey}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from random import randint
from matplotlib import cm
from matplotlib import rc
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['axes.linewidth'] = 1.0

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

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        if survey == 'eco':
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.3) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)   

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)
    # print(counts)
    # print(phi)
    return maxis, phi, err_tot, bins, counts

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool):
    """
    Calculates differential baryonic mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of baryonic masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

    sim_bool: boolean
        True if masses are from mock

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
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmbary_arr = np.log10((10**mass_arr) / 2.041)
    else:
        logmbary_arr = np.log10(mass_arr)
    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.3) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

def get_std_phi_mocks(survey, path, mf_type):
    """
    Calculate error in data SMF from mocks

    Parameters
    ----------
    survey: string
        Name of survey
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

    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

    phi_arr_total = []
    max_arr_total = []
    phi_arr_red = []
    max_arr_red = []
    phi_arr_blue = []
    max_arr_blue = []
    for num in range(num_mocks):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        mock_pd = reading_catls(filename) 

        # Using the same survey definition as in mcmc mf i.e excluding the 
        # buffer
        if mf_type == 'smf':
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
                (mock_pd.logmstar.values >= mstar_limit)]
            logmstar_arr = mock_pd.logmstar.values
            mass_arr = logmstar_arr

        elif mf_type == 'bmf':
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit)]
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            mass_arr = calc_bary(logmstar_arr, logmgas_arr)
            mock_pd['logmbary'] = mass_arr
         
        u_r_arr = mock_pd.u_r.values

        colour_label_arr = np.empty(len(mock_pd), dtype='str')
        for idx, value in enumerate(logmstar_arr):

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

        #Measure SMF of mock using diff_smf function
        if mf_type == 'smf':
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_smf(mass_arr, volume, False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
                volume, False)
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
                volume, False)

        if mf_type == 'bmf':
            max_total, phi_total, err_total, bins_total, counts_total = \
                diff_bmf(mass_arr, volume, False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                diff_bmf(mock_pd.logmbary.loc[mock_pd.colour_label.values == 'R'],
                volume, False)
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                diff_bmf(mock_pd.logmbary.loc[mock_pd.colour_label.values == 'B'],
                volume, False)

        phi_arr_total.append(phi_total)
        max_arr_total.append(max_total)
        phi_arr_red.append(phi_red)
        max_arr_red.append(max_red)
        phi_arr_blue.append(phi_blue)
        max_arr_blue.append(max_blue)

    phi_arr_total = np.array(phi_arr_total)
    max_arr_total = np.array(max_arr_total)
    phi_arr_red = np.array(phi_arr_red)
    max_arr_red = np.array(max_arr_red)
    phi_arr_blue = np.array(phi_arr_blue)
    max_arr_blue = np.array(max_arr_blue)

    # Std of log(phi) - (already in log)
    err_total = np.std(phi_arr_total, axis=0)
    err_red = np.std(phi_arr_red, axis=0)
    err_blue = np.std(phi_arr_blue, axis=0)

    return [err_total, err_red, err_blue], \
        [phi_arr_total,phi_arr_red,phi_arr_blue], \
        [max_arr_total,max_arr_red,max_arr_blue]

def get_jk_mocks(survey, path, mf_type, data_catl):
    """
    Calculate error in data SMF from mocks

    Parameters
    ----------
    survey: string
        Name of survey
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

    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

    stddev_jk_arr = []
    err_total_arr = []
    for i in range(25):
        num_mocks_sample = np.random.choice(num_mocks, 8)
        for num in num_mocks_sample:
            filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
                (mock_pd.logmstar.values >= mstar_limit)]
            print(len(mock_pd))

            if survey == 'resolveb':
                data_catl.radeg.loc[data_catl.radeg.values > 300] -= 360
                ra = data_catl.radeg.values # degrees
                dec = data_catl.dedeg.values # degrees
                sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
                # Grid only in declination
                sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 7)
                ra_arr = np.linspace(ra.min(), ra.max(), 2)

            elif survey == 'resolvea':
                ra = data_catl.radeg.values # degrees
                dec = data_catl.dedeg.values # degrees
                sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
                sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 5)
                ra_arr = np.linspace(ra.min(), ra.max(), 26)

            elif survey == 'eco':
                ra = data_catl.radeg.values # degrees
                dec = data_catl.dedeg.values # degrees
                sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
                sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 7)
                ra_arr = np.linspace(ra.min(), ra.max(), 7)

            grid_id_arr = []
            gal_id_arr = []
            grid_id = 1
            max_bin_id = len(sin_dec_arr)-2 # left edge of max bin
            for dec_idx in range(len(sin_dec_arr)):
                for ra_idx in range(len(ra_arr)):
                    try:
                        if dec_idx == max_bin_id and ra_idx == max_bin_id:
                            mock_pd_subset = mock_pd.loc[(mock_pd.ra.values >= 
                                ra_arr[ra_idx]) &
                                (mock_pd.ra.values <= ra_arr[ra_idx+1]) & 
                                (np.sin(np.deg2rad(mock_pd.dec.values)) >= 
                                sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                                mock_pd.dec.values)) <= sin_dec_arr[dec_idx+1])] 
                        elif dec_idx == max_bin_id:
                            mock_pd_subset = mock_pd.loc[(mock_pd.ra.values >= 
                                ra_arr[ra_idx]) &
                                (mock_pd.ra.values < ra_arr[ra_idx+1]) & 
                                (np.sin(np.deg2rad(mock_pd.dec.values)) >= 
                                sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                                mock_pd.dec.values)) <= sin_dec_arr[dec_idx+1])] 
                        elif ra_idx == max_bin_id:
                            mock_pd_subset = mock_pd.loc[(mock_pd.ra.values >= 
                                ra_arr[ra_idx]) &
                                (mock_pd.ra.values <= ra_arr[ra_idx+1]) & 
                                (np.sin(np.deg2rad(mock_pd.dec.values)) >= 
                                sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                                mock_pd.dec.values)) < sin_dec_arr[dec_idx+1])] 
                        else:                
                            mock_pd_subset = mock_pd.loc[(mock_pd.ra.values >= 
                                ra_arr[ra_idx]) &
                                (mock_pd.ra.values < ra_arr[ra_idx+1]) & 
                                (np.sin(np.deg2rad(mock_pd.dec.values)) >= 
                                sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                                mock_pd.dec.values)) < sin_dec_arr[dec_idx+1])] 
                        # Append dec and sin  
                        for gal_id in mock_pd_subset.groupid.values:
                            gal_id_arr.append(gal_id)
                        for grid_id in [grid_id] * len(mock_pd_subset):
                            grid_id_arr.append(grid_id)
                        grid_id += 1
                    except IndexError:
                        break

            gal_grid_id_data = {'grid_id': grid_id_arr, 'groupid': gal_id_arr}
            df_gal_grid = pd.DataFrame(data=gal_grid_id_data)

            mock_pd = mock_pd.join(df_gal_grid.set_index('groupid'), on='groupid')
            mock_pd = mock_pd.reset_index(drop=True)

            # Loop over all sub grids, remove one and measure global smf
            jackknife_phi_arr = []
            jackknife_max_arr = []
            jackknife_err_arr = []
            gals_count_arr = []
            jackknife_counts_arr = []
            N_grids = (len(ra_arr)-1)*(len(sin_dec_arr)-1)

            # volume = ((N_grids - 1)/N_grids)*volume
            for grid_id in range(N_grids):
                grid_id += 1
                mock_pd_subset = mock_pd.loc[mock_pd.grid_id.values != grid_id]  
                logmstar = mock_pd_subset.logmstar.values
                if mf_type == 'smf':
                    maxis, phi, err, bins, counts = diff_smf(logmstar, volume, False)
                jackknife_phi_arr.append(phi)
                jackknife_max_arr.append(maxis) 
                jackknife_err_arr.append(err)
                jackknife_counts_arr.append(counts)
                gals_count_arr.append(len(mock_pd_subset))

            jackknife_phi_arr = np.array(jackknife_phi_arr)
            jackknife_max_arr = np.array(jackknife_max_arr)
            jackknife_err_arr = np.array(jackknife_err_arr)
            jackknife_counts_arr = np.array(jackknife_counts_arr)
            gals_count_arr = np.array(gals_count_arr)

            # bias = True sets normalization to N instead of default N-1
            cov_mat = np.cov(jackknife_phi_arr.T, bias=True)*(N_grids-1)
            stddev_jk = np.sqrt(cov_mat.diagonal())
            stddev_jk_arr.append(stddev_jk)

        phi_total_arr = []
        for num in num_mocks_sample:
            filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)]
                logmstar_arr = mock_pd.logmstar.values
                mass_arr = logmstar_arr

            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)]
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                mass_arr = calc_bary(logmstar_arr, logmgas_arr)
                mock_pd['logmbary'] = mass_arr
            
            u_r_arr = mock_pd.u_r.values

            colour_label_arr = np.empty(len(mock_pd), dtype='str')
            for idx, value in enumerate(logmstar_arr):

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

            #Measure SMF of mock using diff_smf function
            if mf_type == 'smf':
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_smf(mass_arr, volume, False)

            if mf_type == 'bmf':
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_bmf(mass_arr, volume, False)

            phi_total_arr.append(phi_total)

        # Std of log(phi) - (already in log)
        err_total = np.std(phi_total_arr, axis=0)
        err_total_arr.append(err_total)
    
    err_total_arr = np.array(err_total_arr)
    stddev_jk_arr = np.array(stddev_jk_arr)

    return stddev_jk_arr, err_total_arr

def get_sampled_err_data(survey, path, mf_type):
    """
    Calculate error in data SMF from randomly sampled mocks

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    err_total: array
        Standard deviation of phi values between samples of 8 mocks
    """

    if survey == 'eco':
        mock_name = 'ECO'
        num_mocks = 8
        min_cz = 3000
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    elif survey == 'resolvea':
        mock_name = 'A'
        num_mocks = 59
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17.33
        mstar_limit = 8.9
        volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
    elif survey == 'resolveb':
        mock_name = 'B'
        num_mocks = 104
        min_cz = 4500
        max_cz = 7000
        mag_limit = -17
        mstar_limit = 8.7
        volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

    phi_total_arr = []
    err_total_arr = []
    for i in range(100):
        num_mocks_sample = np.random.choice(num_mocks, 8)
        for num in num_mocks_sample:
            filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)]
                logmstar_arr = mock_pd.logmstar.values
                mass_arr = logmstar_arr

            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                    (mock_pd.cz.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)]
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                mass_arr = calc_bary(logmstar_arr, logmgas_arr)
                mock_pd['logmbary'] = mass_arr
            
            u_r_arr = mock_pd.u_r.values

            colour_label_arr = np.empty(len(mock_pd), dtype='str')
            for idx, value in enumerate(logmstar_arr):

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

            #Measure SMF of mock using diff_smf function
            if mf_type == 'smf':
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_smf(mass_arr, volume, False)

            if mf_type == 'bmf':
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_bmf(mass_arr, volume, False)

            phi_total_arr.append(phi_total)

        # Std of log(phi) - (already in log)
        err_total = np.std(phi_total_arr, axis=0)
        err_total_arr.append(err_total)
    
    err_total_arr = np.array(err_total_arr)

    return err_total_arr

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
                (resolve_live18.grpcz.values >= 4500) & \
                    (resolve_live18.grpcz.values <= 7000) & \
                        (resolve_live18.absrmag.values <= -17.33) & \
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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']

global survey
global mf_type
survey = 'eco'
mf_type = 'smf'

if survey == 'eco':
    path_to_old_mocks = path_to_external + 'm200b/eco/old/ECO_m200b_catls/'
    path_to_new_mocks = path_to_external + 'm200b/eco/'
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

catl, volume, cvar, z_median = read_data(catl_file, survey)

if survey == 'resolveb':
    catl.radeg.loc[catl.radeg.values > 300] -= 360
    ra = catl.radeg.values # degrees
    dec = catl.dedeg.values # degrees
    sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
    # Grid only in declination
    sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 7)
    ra_arr = np.linspace(ra.min(), ra.max(), 2)

elif survey == 'resolvea':
    ra = catl.radeg.values # degrees
    dec = catl.dedeg.values # degrees
    sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
    sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 5)
    ra_arr = np.linspace(ra.min(), ra.max(), 26)

elif survey == 'eco':
    ra = catl.radeg.values # degrees
    dec = catl.dedeg.values # degrees
    sin_dec_all = np.sin(np.deg2rad(dec)) # degrees
    sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 7)
    ra_arr = np.linspace(ra.min(), ra.max(), 7)

grid_id_arr = []
gal_id_arr = []
grid_id = 1
max_bin_id = len(sin_dec_arr)-2 # left edge of max bin
for dec_idx in range(len(sin_dec_arr)):
    for ra_idx in range(len(ra_arr)):
        try:
            if dec_idx == max_bin_id and ra_idx == max_bin_id:
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                    (np.sin(np.deg2rad(catl.dedeg.values)) >= 
                        sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                            catl.dedeg.values)) <= sin_dec_arr[dec_idx+1])] 
            elif dec_idx == max_bin_id:
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values < ra_arr[ra_idx+1]) & 
                    (np.sin(np.deg2rad(catl.dedeg.values)) >= 
                        sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                            catl.dedeg.values)) <= sin_dec_arr[dec_idx+1])] 
            elif ra_idx == max_bin_id:
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                    (np.sin(np.deg2rad(catl.dedeg.values)) >= 
                        sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                            catl.dedeg.values)) < sin_dec_arr[dec_idx+1])] 
            else:                
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values < ra_arr[ra_idx+1]) & 
                    (np.sin(np.deg2rad(catl.dedeg.values)) >= 
                        sin_dec_arr[dec_idx]) & (np.sin(np.deg2rad(
                            catl.dedeg.values)) < sin_dec_arr[dec_idx+1])] 
            # Append dec and sin  
            for gal_id in catl_subset.name.values:
                gal_id_arr.append(gal_id)
            for grid_id in [grid_id] * len(catl_subset):
                grid_id_arr.append(grid_id)
            grid_id += 1
        except IndexError:
            break

gal_grid_id_data = {'grid_id': grid_id_arr, 'name': gal_id_arr}
df_gal_grid = pd.DataFrame(data=gal_grid_id_data)

catl = catl.join(df_gal_grid.set_index('name'), on='name')
catl = catl.reset_index(drop=True)

# Loop over all sub grids, remove one and measure global smf
jackknife_phi_arr = []
jackknife_max_arr = []
jackknife_err_arr = []
gals_count_arr = []
jackknife_counts_arr = []
N_grids = (len(ra_arr)-1)*(len(sin_dec_arr)-1)
print(N_grids)
# volume = ((N_grids - 1)/N_grids)*volume
for grid_id in range(N_grids):
    grid_id += 1
    catl_subset = catl.loc[catl.grid_id.values != grid_id]  
    logmstar = catl_subset.logmstar.values
    logmgas = catl_subset.logmgas.values
    logmbary = calc_bary(logmstar, logmgas)
    if mf_type == 'smf':
        maxis, phi, err, bins, counts = diff_smf(logmstar, volume, False)
    elif mf_type == 'bmf':
        maxis, phi, err, bins, counts = diff_bmf(logmbary, volume, False)
    jackknife_phi_arr.append(phi)
    jackknife_max_arr.append(maxis) 
    jackknife_err_arr.append(err)
    jackknife_counts_arr.append(counts)
    gals_count_arr.append(len(catl_subset))

jackknife_phi_arr = np.array(jackknife_phi_arr)
jackknife_max_arr = np.array(jackknife_max_arr)
jackknife_err_arr = np.array(jackknife_err_arr)
jackknife_counts_arr = np.array(jackknife_counts_arr)
gals_count_arr = np.array(gals_count_arr)

# bias = True sets normalization to N instead of default N-1
cov_mat = np.cov(jackknife_phi_arr.T, bias=True)*(N_grids-1)
stddev_jk_data = np.sqrt(cov_mat.diagonal())
corr_mat = cov_mat / np.outer(stddev_jk_data , stddev_jk_data)
corr_mat_inv = np.linalg.inv(corr_mat)

# stddev_mocks = get_std_phi_mocks(survey, path_to_mocks, mf_type)
jk_mock_arr, std_mock_arr = get_jk_mocks(survey, path_to_mocks, mf_type, catl)
# sampled_stddev_mocks = get_sampled_err_data(survey,path_to_mocks,mf_type)
'''
fig1 = plt.figure()
plt.scatter(ra,sin_dec_all,c=catl.logmstar.values,marker='x',s=10)
for ycoords in sin_dec_arr:
    plt.axhline(y=ycoords, color='k', ls='--')
for xcoords in ra_arr:
    plt.axvline(x=xcoords, color='k', ls='--')
# To stop offset in y being plotted
plt.ylim(sin_dec_all.min(), sin_dec_all.max())
plt.colorbar()
plt.ylabel(r'\boldmath$\sin(\delta) [deg]$')
plt.xlabel(r'\boldmath$\alpha [deg]$')
plt.show()

# For comparing jk one mock to sigma_jk (data) as well as std(phi) of mocks
fig2, ax = plt.subplots()  
# ax.scatter(stddev_jk, stddev_mocks[0][0], c='#EF379F')
ax.scatter(stddev_jk_data, stddev_mocks[0][0], c='#37EF88', s=10, label='mock std(phi)')
ax.scatter(stddev_jk_data, jk_mock_arr, c='#EF379F', s=10, label='mock jackknife')
n = np.arange(1,len(stddev_jk_data)+1,1) 
for i, txt in enumerate(n): 
    ax.annotate(txt, (stddev_jk_data[i], stddev_mocks[0][0][i]), size=8)
    ax.annotate(txt, (stddev_jk_data[i], jk_mock_arr[0][i]), size=8)  
plt.plot(np.linspace(0,0.4,10),np.linspace(0,0.4,10), '--k')
plt.xlabel(r'${\sigma}_{jackknife}^{data}$') 
plt.ylabel(r'${\sigma}_{mocks}$') 
plt.title(r'{0}'.format(survey))
plt.legend(loc='best')
plt.show()

# For comparing jk all mocks to std(phi) of mocks
fig3, ax = plt.subplots(figsize=(10,7))  
for i in range(len(jk_mock_arr.transpose())):
    # Each line here is a bin
    ax.scatter(jk_mock_arr.transpose()[i], 104*[stddev_mocks[0][0][i]], 
        c=np.random.rand(3,), s=30, label='{0}'.format(i+1))
plt.plot(np.linspace(0,0.8,10),np.linspace(0,0.8,10), '--k')
plt.xlabel(r'${\sigma}_{jackknife}$') 
plt.ylabel(r'${\sigma}_{phi}$') 
plt.title(r'{0}'.format(survey))
plt.legend(loc='best')
plt.show()

# For comparing 1 of 25 samples of 8 RESOLVE B mocks to ECO (jackknife vs std(phi))
jk_mock_arr_reshape = jk_mock_arr.reshape(25,8,6)
fig4, ax = plt.subplots(figsize=(10,7)) 
for i in range(8):
    ax.scatter(jk_mock_arr_reshape[0][i],std_mock_arr[0])
plt.plot(np.linspace(0,0.4,10),np.linspace(0,0.4,10), '--k') 
plt.xlabel(r'${\sigma}_{jackknife}^{data}$')  
plt.ylabel(r'${\sigma}_{phi}$')  
plt.title(r'{0}'.format(survey)) 
plt.legend(loc='best') 
'''

# For comparing all 25 samples of 8 RESOLVE B mocks to ECO (jackknife vs std(phi))
jk_mock_arr_reshape = jk_mock_arr.reshape(25,8,6)
fig5, axs = plt.subplots(5, 5, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
fig5.suptitle('25 samples of 8 RESOLVE B mocks')
row = 0
col = 0
for i in range(25):
    for j in range(8):
        axs[row, col].scatter(jk_mock_arr_reshape[i][j],std_mock_arr[i])
        axs[row, col].plot(np.linspace(0,0.6,10),np.linspace(0,0.6,10), '--k')
        axs[row, col].set_ylim(0,0.6)
    if col == 4:
        row += 1
        col = 0
    else:
        col += 1
for ax in axs.flat:
    ax.label_outer()
axs[4,2].set_xlabel(r'$\boldsymbol{\sigma}_\mathbf{jackknife}$', fontsize=20)
axs[2,0].set_ylabel(r'${\boldsymbol{\sigma}_\mathbf{phi}}$', fontsize=20)
plt.show()