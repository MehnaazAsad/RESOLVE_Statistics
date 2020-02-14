"""
{This script checks whether log(phi) or phi values within bins represent a 
 normal distribution}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
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
   
# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_data = dict_of_paths['data_dir']
path_to_raw = dict_of_paths['raw_dir']

global survey
survey = 'eco'

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
        #Using the same survey definition as in mcmc smf i.e excluding the buffer
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= temp_dict.get('min_cz')) & 
            (mock_pd.cz.values <= temp_dict.get('max_cz')) & 
            (mock_pd.M_r.values <= temp_dict.get('mag_limit')) &
            (mock_pd.logmstar.values >= temp_dict.get('mstar_limit'))]
        
        logmstar_arr = mock_pd.logmstar.values
        mhi_arr = mock_pd.mhi.values
        logmgas_arr = np.log10(1.4 * mhi_arr)
        logmbary_arr = np.log10(10**(logmstar_arr) + 10**(logmgas_arr))
        
        volume =  temp_dict.get('volume') 
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

log_pvalues = [] 
for i in range(len(phi_arr_smf.T)): 
    result = normaltest(np.log10(phi_arr_smf.T[i])) 
    log_pvalues.append(result[1]) 
log_pvalues = np.array(log_pvalues) 

not_log_pvalues = [] 
for i in range(len(phi_arr_smf.T)): 
    result = normaltest(phi_arr_smf.T[i]) 
    not_log_pvalues.append(result[1]) 
not_log_pvalues = np.array(not_log_pvalues) 

fig1 = plt.figure(figsize=(8,8)) 
ax1 = fig1.add_subplot(2,3,1)   
ax2 = fig1.add_subplot(2,3,2)   
ax3 = fig1.add_subplot(2,3,3)   
ax4 = fig1.add_subplot(2,3,4)   
ax5 = fig1.add_subplot(2,3,5)   
ax6 = fig1.add_subplot(2,3,6)   
ax_arr = [ax1,ax2,ax3,ax4,ax5,ax6] 
for i in range(len(ax_arr)): 
    ax_arr[i].hist(phi_arr_smf.T[i], histtype='step') 
    ax_arr[i].text(1,1,np.round(not_log_pvalues[i],2),
        horizontalalignment='right',verticalalignment='top',
        transform=ax_arr[i].transAxes) 
fig1.suptitle(r'\boldmath $\Phi$ {0}'.format(survey)) 
plt.show()

fig2 = plt.figure(figsize=(8,8)) 
ax1 = fig2.add_subplot(2,3,1)   
ax2 = fig2.add_subplot(2,3,2)   
ax3 = fig2.add_subplot(2,3,3)   
ax4 = fig2.add_subplot(2,3,4)   
ax5 = fig2.add_subplot(2,3,5)   
ax6 = fig2.add_subplot(2,3,6)   
ax_arr = [ax1,ax2,ax3,ax4,ax5,ax6] 
for i in range(len(ax_arr)): 
    ax_arr[i].hist(np.log10(phi_arr_smf.T[i]), histtype='step') 
    ax_arr[i].text(1,1,np.round(log_pvalues[i],2),
        horizontalalignment='right',verticalalignment='top',
        transform=ax_arr[i].transAxes) 
fig2.suptitle(r'\boldmath log\ $\Phi$ {0}'.format(survey)) 
plt.show()