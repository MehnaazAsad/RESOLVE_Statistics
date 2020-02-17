"""
{This script carries out jackknife resampling for selected survey}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from random import randint
from matplotlib import rc
import seaborn as sns
import pandas as pd
import numpy as np
import math


__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=15)
rc('text', usetex=True)

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
                    'fc', 'grpmb', 'grpms', 'modelu_rcorr']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
            (eco_buff.grpcz.values <= 7000) & \
                (eco_buff.absrmag.values <= -17.33) &\
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

def get_area_on_sphere(ra_min,ra_max,sin_dec_min,sin_dec_max):
    """Calculate area on sphere given ra and dec in degrees"""
    # area = (180/np.pi)*(ra_max-ra_min)* \
    #     (np.rad2deg(np.sin(np.deg2rad(dec_max)))- \
    #         np.rad2deg(np.sin(np.deg2rad(dec_min)))) 
    area = (180/np.pi) * (ra_max-ra_min) * (sin_dec_max - sin_dec_min)
    return area

def get_ra_max(area,dec_min,dec_max,ra_min):
    """Calculate ra width given a fixed area, dec range and lower limit of ra"""
    const = (area/((180/np.pi)*(np.rad2deg(np.sin(np.deg2rad(dec_max)))- \
            np.rad2deg(np.sin(np.deg2rad(dec_min))))))
    ra_max = const + ra_min
    return ra_max

def fixed_dec_method():
    ra_slices_or = np.linspace(ra.min(), ra.max(), 10)
    dec_slices = np.linspace(dec.min(), dec.max(), 10)

    # Calculating area of boxes in row 1 to get fixed area to use
    area_row1 = []
    for i in range(len(dec_slices)):
        if i == 9:
            break
        area = get_area_on_sphere(ra_slices_or[i], ra_slices_or[i+1], 
            dec_slices[0], dec_slices[1] )
        area_row1.append(area)

    dec_width = dec_slices[1] - dec_slices[0]

    # On sphere, as dec changes the area gets smaller so by forcing the area to 
    # be the same as the first slice as well as keeping DEC ranges fixed, get 
    # new ra width for each slice (difference in longitude is more pronounced 
    # than different in latitude as you move up/down)
    ra_slices_arr = []
    ra_slices_arr.append(ra_slices_or)
    for idx, val in enumerate(dec_slices[1:]):
        fixed_area = area_row1[0]
        dec_min_ = val
        dec_max_ = val + dec_width
        ra_min_ = ra_slices_or[0]
        ra_max_ = get_ra_max(fixed_area, dec_min_, dec_max_, ra_min_)
        new_ra_width = ra_max_ - ra_min_
        new_ra_slice = np.arange(ra_min_, np.ceil(ra.max()), new_ra_width)
        ra_slices_arr.append(new_ra_slice)
    ra_slices_arr = np.array(ra_slices_arr)

    # Checking that all boxes are the same area
    area_rows = []
    for idx, val in enumerate(ra_slices_arr):
        for idx2, val2 in enumerate(val):
            try:
                area = get_area_on_sphere(val2, val[idx2+1], 
                    dec_slices[idx], dec_slices[idx+1])
                area_rows.append(area)
            except IndexError:
                break

    plt.clf()
    fig1 = plt.figure()
    plt.scatter(ra,dec,c='cyan',marker='x',s=10)
    for idx, ycoords in enumerate(dec_slices):
        plt.axhline(y=ycoords, color='k', ls='--')
        for xcoords in ra_slices_arr[idx]:
            y_norm_min = (ycoords - dec.min())/(dec.max() - dec.min())
            y_norm_max = ((ycoords + dec_width)-dec.min())/(dec.max() - 
                dec.min())
            plt.axvline(x=xcoords, ymin=y_norm_min, ymax=y_norm_max,
                color='k', ls='--')
            # To stop offset in y being plotted
            plt.ylim(dec.min(), dec.max())

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

def get_err_data(survey, path):
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
    phi_arr_red = []
    phi_arr_blue = []
    max_arr_blue = []
    err_arr_blue = []
    for num in range(num_mocks):
        filename = path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        mock_pd = reading_catls(filename) 

        # Using the same survey definition as in mcmc smf i.e excluding the 
        # buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
            (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)]

        logmstar_arr = mock_pd.logmstar.values 
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
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(logmstar_arr, volume, False)

        phi_arr_total.append(phi_total)
 
    phi_arr_total = np.array(phi_arr_total)
    err_total = np.std(phi_arr_total, axis=0)

    return err_total

def measure_cov_mat(mf_array):

    cov_mat = np.cov(mf_array)
    fig1 = plt.figure()
    ax = sns.heatmap(cov_mat, linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.show()

def measure_corr_mat(mf_array):

    if mf_type == 'smf':
        columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'] 
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

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']

global survey
global mf_type
survey = 'resolvea'
mf_type = 'smf'

if survey == 'eco':
    path_to_mocks = path_to_external + 'ECO_mvir_catls/'
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

catl, volume, cvar, z_median = read_data(catl_file, survey)

ra = catl.radeg.values # degrees
dec = catl.dedeg.values # degrees

sin_dec_all = np.rad2deg(np.sin(np.deg2rad(dec))) # degrees

sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 3)
ra_arr = np.linspace(ra.min(), ra.max(), 3)

area_boxes = []
for dec_idx in range(len(sin_dec_arr)):
    for ra_idx in range(len(ra_arr)):
        try:
            area = get_area_on_sphere(ra_arr[ra_idx], ra_arr[ra_idx+1], 
                sin_dec_arr[dec_idx], sin_dec_arr[dec_idx+1])
            area_boxes.append(area)
        except IndexError:
            break

# Plotting sin(dec) vs RA - all boxes same size
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

# Plotting sin inverse of sin(dec) vs RA - boxes slowly get bigger as you move
# up in declination
fig2 = plt.figure()
plt.scatter(ra,dec,c='cyan',marker='x',s=10)
for ycoords in sin_dec_arr:
    ycoords = np.rad2deg(math.asin(np.deg2rad(ycoords)))
    plt.axhline(y=ycoords, color='k', ls='--')
for xcoords in ra_arr:
    plt.axvline(x=xcoords, color='k', ls='--')
# To stop offset in y being plotted
plt.ylim(dec.min(), dec.max())
plt.ylabel(r'\boldmath$\delta [deg]$')
plt.xlabel(r'\boldmath$\alpha [deg]$')

area_boxes_ = []
for dec_idx in range(len(sin_dec_arr)):
    for ra_idx in range(len(ra_arr)):
        try:
            area = get_area_on_sphere(ra_arr[ra_idx], ra_arr[ra_idx+1], 
                np.rad2deg(math.asin(np.deg2rad(sin_dec_arr[dec_idx]))), 
                np.rad2deg(math.asin(np.deg2rad(sin_dec_arr[dec_idx+1]))))
            area_boxes_.append(area)
        except IndexError:
            break

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
                    (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                        sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                            catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
            elif dec_idx == max_bin_id:
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values < ra_arr[ra_idx+1]) & 
                    (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                        sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                            catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
            elif ra_idx == max_bin_id:
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                    (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                        sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                            catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
            else:                
                catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                    (catl.radeg.values < ra_arr[ra_idx+1]) & 
                    (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                        sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                            catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
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
jackknife_smf_phi_arr = []
jackknife_smf_max_arr = []
jackknife_smf_err_arr = []
for grid_id in range(len(np.unique(catl.grid_id.values))):
    grid_id += 1
    catl_subset = catl.loc[catl.grid_id.values != grid_id]  
    logmstar = catl_subset.logmstar.values
    maxis, phi, err, bins, counts = diff_smf(logmstar, volume, False)
    jackknife_smf_phi_arr.append(phi)
    jackknife_smf_max_arr.append(maxis) 
    jackknife_smf_err_arr.append(err)

jackknife_smf_phi_arr = np.array(jackknife_smf_phi_arr)
jackknife_smf_max_arr = np.array(jackknife_smf_max_arr)
jackknife_smf_err_arr = np.array(jackknife_smf_err_arr)

N = len(jackknife_smf_phi_arr)

# bias = True sets normalization to N instead of default N-1
cov_mat = np.matrix(np.cov(jackknife_smf_phi_arr.T, bias=True))*(N-1)
stddev = np.sqrt(cov_mat.diagonal())
corr_mat = cov_mat / np.outer(stddev , stddev)

corr_mat_inv = np.linalg.inv(corr_mat)
fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
cmap = cm.get_cmap('rainbow')
cax = ax1.matshow(corr_mat, cmap=cmap)
# Loop over data dimensions and create text annotations.
for i in range(corr_mat.shape[0]):
    for j in range(corr_mat.shape[1]):
        text = ax1.text(j, i, np.round(corr_mat[i, j],2),
                    ha="center", va="center", color="w", size='10')
plt.gca().invert_yaxis() 
plt.gca().xaxis.tick_bottom()
fig3.colorbar(cax)
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

# Plotting sin(dec) vs RA - all boxes same size
fig4 = plt.figure()
colours = []
for i in range(100):
    colours.append('#%06X' % randint(0, 0xFFFFFF))
for idx, grid_id in enumerate(np.unique(catl.grid_id.values)):
    ra = catl.radeg.loc[catl.grid_id.values == grid_id]
    sin_dec = np.rad2deg(np.sin(np.deg2rad(catl.dedeg.loc
        [catl.grid_id.values == grid_id])))
    plt.scatter(ra,sin_dec,c=colours[idx],marker='x',s=10)
for ycoords in sin_dec_arr:
    plt.axhline(y=ycoords, color='k', ls='--')
for xcoords in ra_arr:
    plt.axvline(x=xcoords, color='k', ls='--')
# To stop offset in y being plotted
plt.ylim(sin_dec_all.min(), sin_dec_all.max())
plt.ylabel(r'\boldmath$\sin(\delta) [deg]$')
plt.xlabel(r'\boldmath$\alpha [deg]$')

# Plotting all SMFs from jackknife
fig5 = plt.figure()
for idx in range(len(jackknife_smf_phi_arr)):
    plt.errorbar(jackknife_smf_max_arr[idx],jackknife_smf_phi_arr[idx],
        yerr=jackknife_smf_err_arr[idx], markersize=4, capsize=5, capthick=0.5, 
        color=colours[idx],fmt='-s',ecolor=colours[idx])
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
