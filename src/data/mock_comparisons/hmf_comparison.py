"""
The purpose of this script is to compare the HMF that RESOLVE used (Warren et.
al 2006) with the HMF I use for abundance matching from the Vishnu simulation 
to see if a difference exists and if so, perhaps accounts for the difference
in logmh_s vs grpms for centrals between my AM and that of RESOLVE
"""

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def cumu_num_dens(data,bins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
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

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_data = dict_of_paths['data_dir']

#HMF from hmfcalc; Warren et. al 2006
cumunumdens_table = pd.read_csv(path_to_data + 'external/all_plots/'\
    'mVector_PLANCK-SMT .txt', usecols=[0,8],names=['Mass','n'],comment='#',
    delim_whitespace=True)

halo_table = pd.read_csv(path_to_interim + 'id_macc.csv',header=0)
v_sim = 130**3 #(Mpc/h)^3
halo_mass = halo_table.halo_macc.values
nbins = num_bins(halo_mass)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
bin_width_hmass = cumu_num_dens(halo_mass,nbins,None,v_sim,False)

fig1 = plt.figure(figsize=(10,8))
plt.plot(cumunumdens_table.Mass.values,cumunumdens_table.n.values,\
         label=r'Warren et. al 2006',color='#128486')
plt.plot(bin_centers_hmass,n_hmass,label=r'Vishnu simulation',color='#eb4f1c')
plt.xlabel(r'$\mathbf {Mass \left[M_\odot\ h^{-1}\right ]}$')
plt.ylabel(r'$\mathbf {(n > M)} \left[\mathbf{{h}^{3}{Mpc}^{-3}}\right]$')
plt.title(r'$\mathbf {HMF\ comparison}$')
plt.legend(loc='best',prop={'size': 10})
plt.xscale('log')
plt.yscale('log')
plt.show()

################################################################################

## HMF comparison between ECO mock catalogs and Vishnu simulation (whole box)

from halotools.sim_manager import CachedHaloCatalog
from collections import OrderedDict
import os

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

def assign_cen_sat_flag(gals_df):
    """
    Assign centrals and satellites flag to dataframe

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    gals_df: pandas dataframe
        Mock catalog with centrals/satellites flag as new column
    """

    C_S = []
    for idx in range(len(gals_df)):
        if gals_df['halo_hostid'][idx] == gals_df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)

    C_S = np.array(C_S)
    gals_df['cs_flag'] = C_S
    return gals_df

path_to_mocks = path_to_data + 'mocks/m200b/eco/'
survey = 'eco'

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

bin_centers_arr = []
n_arr = []
box_id_arr = np.linspace(5001,5008,8)
for box in box_id_arr:
    box = int(box)
    temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
        mock_name) 
    for num in range(num_mocks):
        filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
            mock_name, num)
        mock_pd = reading_catls(filename) 
        mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
            (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)]
        halo_mass_cen = 10**(mock_pd.loghalom.loc[mock_pd.cs_flag == 1])
        bins = np.logspace(10.5, 14.5, 8)
        bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
            bin_width_hmass = cumu_num_dens(halo_mass_cen,bins,None,volume,
            False)
        bin_centers_arr.append(bin_centers_hmass)
        n_arr.append(n_hmass)


halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
vishnu_df = pd.DataFrame(np.array(halocat.halo_table))
vishnu_df = assign_cen_sat_flag(vishnu_df)
v_sim = 130**3
halo_mass = vishnu_df.halo_mvir.loc[vishnu_df.cs_flag == 1]
bins = np.logspace(10.5, 14.5, 8)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
    bin_width_hmass = cumu_num_dens(halo_mass,bins,None,v_sim,False)

fig1 = plt.figure(figsize=(10,8))
for idx in range(len(n_arr)):
    plt.plot(np.log10(bin_centers_arr[idx]),np.log10(n_arr[idx]),
        color='lightgray', label='mocks', marker='s')
plt.plot(np.log10(bin_centers_hmass),np.log10(n_hmass),
    label=r'Vishnu simulation',color='#eb4f1c', marker='s')
plt.xlabel(r'\boldmath$\log_{10}\ M_h \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'$\mathbf {(n > M)} \left[\mathbf{{h}^{3}{Mpc}^{-3}}\right]$')
plt.title(r'$\mathbf {HMF\ comparison}$')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
plt.show()