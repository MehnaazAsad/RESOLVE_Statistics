"""
{This script compares shmr, differential smf and cumulative smf between 
 mocks and best fit. The motivation for shmr comparison was to see how 
 abundance matching M_r + closest M_r match from survey to get M* in mocks 
 compares with using Behroozi to get M* in modeling. If you account for 
 standard deviation the grey lines fall within 1sigma even though the best fit
 is different at intermediate masses.}
"""

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import random
import emcee
import os


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

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

def read_data_catl(path_to_file, survey):
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

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        # columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
        #             'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
        #             'fc', 'grpmb', 'grpms','modelu_rcorr']

        # 13878 galaxies
        # eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
        #     usecols=columns)

        eco_buff = reading_catls(path_to_file)
        #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
        eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
                (eco_buff.grpcz_new.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_new.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            # cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            if mf_type == 'smf':
                # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            # cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl, volume, z_median

def read_mcmc(path_to_file):
    """
    Reads mcmc chain from file

    Parameters
    ----------
    path_to_file: string
        Path to mcmc chain file

    Returns
    ---------
    emcee_table: pandas.DataFrame
        Dataframe of mcmc chain values with NANs removed
    """
    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'mstar_q','mh_q','mu','nu']
    
    if run >= 37:
        reader = emcee.backends.HDFBackend(path_to_file, read_only=True)
        flatchain = reader.get_chain(flat=True)
        emcee_table = pd.DataFrame(flatchain, columns=colnames)
    elif run < 37:
        emcee_table = pd.read_csv(path_to_file, names=colnames, comment='#',
            header=None, sep='\s+')

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

        emcee_table = emcee_table.dropna(axis='index', how='any').\
            reset_index(drop=True)

    return emcee_table

def get_paramvals_percentile(mcmc_table, pctl, chi2, randints_df=None):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 100 sample

    Parameters
    ----------
    mcmc_table: pandas.DataFrame
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2: array
        Array of chi^2 values
    
    randints_df (optional): pandas.DataFrame
        Dataframe of mock numbers in case many Behroozi mocks were used.
        Defaults to None.

    Returns
    ---------
    mcmc_table_pctl: pandas dataframe
        Sample of 100 68th percentile lowest chi^2 values
    bf_params: numpy array
        Array of parameter values corresponding to the best-fit model
    bf_chi2: float
        Chi-squared value corresponding to the best-fit model
    bf_randint: int
        In case multiple Behroozi mocks were used, this is the mock number
        that corresponds to the best-fit model. Otherwise, this is not returned.
    """ 
    pctl = pctl/100
    mcmc_table['chi2'] = chi2
    if randints_df is not None: # This returns a bool; True if df has contents
        mcmc_table['mock_num'] = randints_df.mock_num.values.astype(int)
    mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(pctl*len(mcmc_table))
    mcmc_table_pctl = mcmc_table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:9]
    bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][9]
    if randints_df is not None:
        bf_randint = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
            values[0][5].astype(int)
        mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)
        return mcmc_table_pctl, bf_params, bf_chi2, bf_randint
    # Randomly sample 100 lowest chi2 
    mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

    return mcmc_table_pctl, bf_params, bf_chi2

def halocat_init(halo_catalog, z_median):
    """
    Initial population of halo catalog using populate_mock function

    Parameters
    ----------
    halo_catalog: string
        Path to halo catalog
    
    z_median: float
        Median redshift of survey

    Returns
    ---------
    model: halotools model instance
        Model based on behroozi 2010 SMHM
    """
    halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_median, \
        prim_haloprop_key='halo_macc')
    model.populate_mock(halocat,seed=5)

    return model

def diff_smf(mstar_arr, volume, h1_bool, colour_flag=False):
    """
    Calculates differential stellar mass function in units of h=1.0

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
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = np.log10(mstar_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        
        if survey == 'eco' and colour_flag == 'R':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 6
        elif survey == 'eco' and colour_flag == 'B':
            bin_max = np.round(np.log10((10**11) / 2.041), 1)
            bin_num = 6

        elif survey == 'resolvea':
            # different to avoid nan in inverse corr mat
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7

        else:
            # For eco total
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 5

        bins = np.linspace(bin_min, bin_max, bin_num)

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
    err_tot = err_poiss
    phi = counts / (volume * dm)  # not a log quantity

    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

def get_host_halo_mock(df, mock):
    """
    Get host halo mass from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_halos: array
        Array of central host halo masses
    sat_halos: array
        Array of satellite host halo masses
    """

    # groups = df.groupby('halo_id')
    # keys = groups.groups.keys()

    # for key in keys:
    #     group = groups.get_group(key)
    # for index, value in enumerate(group.cs_flag):
    #     if value == 1:
    #         cen_halos.append(group.loghalom.values[index])
    #     else:
    #         sat_halos.append(group.loghalom.values[index])

    if mock == 'vishnu':
        cen_halos = np.log10(df.halo_mvir[df.cs_flag == 1]).reset_index(drop=True)
        sat_halos = np.log10(df.halo_mvir_host_halo[df.cs_flag == 0]).reset_index(drop=True)
    else:
        # Both cen and sat are the same mass for a halo i.e. the satellites
        # are assigned a halo mass of the central. 
        cen_halos = df.loghalom[df.cs_flag == 1].reset_index(drop=True)
        sat_halos = df.loghalom[df.cs_flag == 0].reset_index(drop=True)

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(df, mock, randint=None):
    """
    Get stellar mass from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_gals: array
        Array of central stellar masses
    sat_gals: array
        Array of satellite stellar masses
    """

    if mock == 'vishnu' and randint:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(10**(df['{0}'.format(randint)].values[idx]))
            elif value == 0:
                sat_gals.append(10**(df['{0}'.format(randint)].values[idx]))

    elif mock == 'vishnu':
        # cen_gals = 10**(df.logmstar[df.cs_flag == 1]).reset_index(drop=True)
        # sat_gals = 10**(df.logmstar[df.cs_flag == 0]).reset_index(drop=True)
        cen_gals = np.log10(df.stellar_mass[df.cs_flag == 1].reset_index(drop=True))
        sat_gals = np.log10(df.stellar_mass[df.cs_flag == 0].reset_index(drop=True))
   
    else:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append(np.log10((10**(df.logmstar.values[idx]))/2.041))
            elif value == 0:
                sat_gals.append(np.log10((10**(df.logmstar.values[idx]))/2.041))

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

def populate_mock(theta, model):
    """
    Populate mock based on five SMHM parameter values and model

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    model: halotools model instance
        Model based on behroozi 2010 SMHM

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    """"""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

    # if survey == 'eco' or survey == 'resolvea':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.9) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.4) / 2.041), 1)
    # elif survey == 'resolveb':
    #     if mf_type == 'smf':
    #         limit = np.round(np.log10((10**8.7) / 2.041), 1)
    #     elif mf_type == 'bmf':
    #         limit = np.round(np.log10((10**9.1) / 2.041), 1)
    # sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model.mock.galaxy_table#[sample_mask]
    gals_df = pd.DataFrame(np.array(gals))

    return gals_df

def cumu_num_dens(data, bins, weights, volume, bool_mag):
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
        N_cumu = np.cumsum(freq) #magnitudes case
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers ,edg, n_cumu, err_poiss, bin_width

global model_init
global survey
global mf_type
global quenching
global level 

survey = 'eco'
machine = 'mac'
mf_type = 'smf'
quenching = 'hybrid'
ver = 2.0 # No more .dat

if quenching == 'halo':
    run = 44
elif quenching == 'hybrid':
    run = 43


dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_external = dict_of_paths['ext_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

chi2_file = path_to_proc + 'smhm_colour_run{0}/chain.h5'.format(run)
chain_file = path_to_proc + 'smhm_colour_run{0}/chain.h5'.format(run)

if survey == 'eco':
    catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + 'RESOLVE_liveJune2019.csv'
    if survey == 'resolvea':
        path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
    else:
        path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

reader = emcee.backends.HDFBackend(chi2_file , read_only=True)
chi2 = reader.get_blobs(flat=True)

mcmc_table = read_mcmc(chain_file)

print('Reading catalog')
catl, volume, z_median = read_data_catl(catl_file, survey)

print('Getting data in specific percentile')
mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table, 68, chi2)

model_init = halocat_init(halo_catalog, z_median)

gals_df =  populate_mock(bf_params[:5], model_init) #mstar cut at 8.9 in h=1 (8.6)
gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
    gals_df['halo_id'], 1, 0)

cen_halos_bf, sat_halos_bf = get_host_halo_mock(gals_df, 'vishnu')
cen_gals_bf, sat_gals_bf = get_stellar_mock(gals_df, 'vishnu')

bins = np.linspace(10, 15, 7)
shmr = bs(cen_halos_bf, cen_gals_bf, statistic='mean', bins = bins)
centers = 0.5 * (shmr[1][1:] + shmr[1][:-1])
x_bf = centers
y_bf = shmr[0]    

H0 = 100
cz_inner = 3000
cz_outer = 12000
dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

v_inner = vol_sphere(dist_inner)
v_outer = vol_sphere(dist_outer)

v_sphere = v_outer-v_inner
v_sim = v_sphere/8

v_sim = 130**3
mass_bf, phi_bf, err_tot_bf, bins_bf, counts_bf = \
    diff_smf(gals_df.stellar_mass.values, v_sim, True)

cumu_smf_result_bf = cumu_num_dens(gals_df.stellar_mass.values,bins,None,v_sim,False)

## catl_grpcz_cut and catl_cz_cut were initialized in the terminal so if this 
## script is run from scratch the following lines will result in an error and 
## those objects will have to be initialized again
mass_data, phi_data, err_tot_data, bins_data, counts_data = \
    diff_smf(catl_grpcz_cut.logmstar.values, volume, False)
bins = np.logspace(8.6, 11.2, 7)
cumu_smf_result_data = cumu_num_dens(10**catl_grpcz_cut.logmstar.values,bins,None,volume,
    False)
cen_halos_data = catl_grpcz_cut.logmh.loc[catl_grpcz_cut.fc==1]
cen_gals_data = catl_grpcz_cut.logmstar.loc[catl_grpcz_cut.fc==1]
cen_gals_data = np.log10((10**cen_gals_data)/ 2.041)
x,y,y_std,y_std_err = Stats_one_arr(cen_halos_data,
    cen_gals_data,base=0.4,bin_statval='center')

mass_data2, phi_data2, err_tot_data2, bins_data2, counts_data2 = \
    diff_smf(catl_cz_cut.logmstar.values, volume, False)
bins = np.logspace(8.6, 11.2, 7)
cumu_smf_result_data2 = cumu_num_dens(10**catl_cz_cut.logmstar.values,bins,None,volume,
    False)
cen_halos_data2 = catl_cz_cut.logmh.loc[catl_cz_cut.fc==1]
cen_gals_data2 = catl_cz_cut.logmstar.loc[catl_cz_cut.fc==1]
cen_gals_data2 = np.log10((10**cen_gals_data2)/ 2.041)
x2,y2,y_std2,y_std_err2 = Stats_one_arr(cen_halos_data2,
    cen_gals_data2,base=0.4,bin_statval='center')


if survey == 'eco':
    mock_name = 'ECO'
    num_mocks = 8
    min_cz = 3000
    max_cz = 7000
    mag_limit = -17.33
    mstar_limit = 8.9
    # volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
elif survey == 'resolvea':
    mock_name = 'A'
    num_mocks = 59
    min_cz = 4500
    max_cz = 7000
    mag_limit = -17.33
    mstar_limit = 8.9
    # volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
elif survey == 'resolveb':
    mock_name = 'B'
    num_mocks = 104
    min_cz = 4500
    max_cz = 7000
    mag_limit = -17
    mstar_limit = 8.7
    # volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units     as u

def get_survey_vol(ra_arr, dec_arr, rho_arr):
    """
    Computes the volume of a "sphere" with given limits for 
    ra, dec, and distance
    Parameters
    ----------
    ra_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final right ascension coordinates
        Unit: degrees
    dec_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final declination coordinates
        Unit: degrees
    rho_arr: list or numpy.ndarray, shape (N,2)
        arrray with initial and final distance
        Unit: distance units
    
    Returns
    ----------
    survey_vol: float
        volume of the survey being analyzed
        Unit: distance**(3)
    """
    # Right ascension - Radians  theta coordinate
    theta_min_rad, theta_max_rad = np.radians(np.array(ra_arr))
    # Declination - Radians - phi coordinate
    phi_min_rad, phi_max_rad = np.radians(90.-np.array(dec_arr))[::-1]
    # Distance
    rho_min, rho_max = rho_arr
    # Calculating volume
    vol  = (1./3.)*(np.cos(phi_min_rad)-np.cos(phi_max_rad))
    vol *= (theta_max_rad) - (theta_min_rad)
    vol *= (rho_max**3) - rho_min**3
    vol  = np.abs(vol)

    return vol

ra_min_real = 130.05
ra_max_real = 237.45
dec_min     = -1
dec_max     = 49.85
# Extras
dec_range   = dec_max - dec_min
ra_range    = ra_max_real - ra_min_real
H0 = 100
cosmo_model = astrocosmo.Planck15.clone(H0=H0)
km_s       = u.km/u.s
z_arr      = (np.array([min_cz, max_cz])*km_s/(ac.c.to(km_s))).value
z_arr      = (np.array([min_cz, max_cz])*km_s/(3e5*km_s)).value
r_arr      = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
survey_vol_used = get_survey_vol( [0, ra_range], [dec_min, dec_max], r_arr)

x_arr = []
y_arr = []
phi_arr = []
mass_arr = []
cumu_mass_arr = []
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
        mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')

        mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
            (mock_pd.grpcz_new.values <= max_cz) & \
            (mock_pd.M_r.values <= mag_limit) &\
            (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

        cen_halos, sat_halos = get_host_halo_mock(mock_pd, 'other')
        cen_gals, sat_gals = get_stellar_mock(mock_pd, 'other')
        
        bins = np.linspace(10, 15, 7)

        shmr = bs(cen_halos, cen_gals, statistic='mean', bins = bins)
        centers = 0.5 * (shmr[1][1:] + shmr[1][:-1])

        smf_result = diff_smf(mock_pd.logmstar.values, survey_vol_used, False)
        cumu_smf_result = cumu_num_dens(((10**mock_pd.logmstar.values)/2.041),bins,None,
            survey_vol_used,False)
        x_arr.append(centers)
        y_arr.append(shmr[0])        
        phi_arr.append(smf_result[1])
        mass_arr.append(smf_result[0])
        cumu_mass_arr.append(cumu_smf_result[0])
        n_arr.append(cumu_smf_result[2])


## Plot of shmr comparison between mocks and behroozi best fit
fig1 = plt.figure(figsize=(10,10))
for i in range(len(y_arr)):
    plt.plot(centers,y_arr[i],
        color='lightgray', ls='-', lw=2, label=r'mocks',zorder=5)
plt.errorbar(centers,y_bf,color='mediumorchid',fmt='-s',
    ecolor='mediumorchid',markersize=5,capsize=5,capthick=0.5,\
        label=r'best-fit',zorder=20)
# plt.errorbar(x,y,
#     color='k',fmt='-s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with grpcz cut',zorder=10)
# plt.errorbar(x2,y2,
#     color='k',fmt='--s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with cz cut',zorder=10)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
if mf_type == 'smf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**8.9)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**8.7)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
elif mf_type == 'bmf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**9.4)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**9.1)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 15})
plt.show()

## Plot of differential stellar mass function
fig2 = plt.figure(figsize=(10,10))
for i in range(len(phi_arr)):
    plt.errorbar(mass_arr[i],phi_arr[i],
        color='lightgray',fmt='-s', ecolor='lightgray', markersize=4, 
        capsize=5, capthick=0.5, label=r'mocks',zorder=5)
# plt.errorbar(mass_data,phi_data,
#     color='k',fmt='-s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with grpcz cut',zorder=10)
# plt.errorbar(mass_data2,phi_data2,
#     color='k',fmt='--s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with cz cut',zorder=10)
plt.errorbar(mass_bf,phi_bf,
    color='mediumorchid',fmt='-s',ecolor='mediumorchid',markersize=3,
    capsize=5,capthick=0.5,label='best fit',zorder=10)
plt.ylim(-4,-1)
if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
plt.show()

## Plot of cumulative stellar mass function
fig3 = plt.figure(figsize=(10,10))
for i in range(len(n_arr)):
    plt.errorbar(np.log10(cumu_mass_arr[i]),np.log10(n_arr[i]),
        color='lightgray',fmt='-s', ecolor='lightgray', markersize=4, 
        capsize=5, capthick=0.5, label=r'mocks',zorder=5)
# plt.errorbar(np.log10(cumu_mass_arr[60]),np.log10(n_arr[60]),
#     color='lightgray',fmt='-s', ecolor='lightgray', markersize=4, 
#     capsize=5, capthick=0.5, label=r'mocks',zorder=5)
# plt.errorbar(np.log10(cumu_smf_result_data[0]),np.log10(cumu_smf_result_data[2]),
#     color='k',fmt='-s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with grpcz cut',zorder=10)
# plt.errorbar(np.log10(cumu_smf_result_data2[0]),np.log10(cumu_smf_result_data2[2]),
#     color='k',fmt='--s',ecolor='k',markersize=3,
#     capsize=5,capthick=0.5,label='data with cz cut',zorder=10)
plt.errorbar(np.log10(cumu_smf_result_bf[0]),np.log10(cumu_smf_result_bf[2]),
    color='mediumorchid',fmt='-s',ecolor='mediumorchid',markersize=3,
    capsize=5,capthick=0.5,label='best fit',zorder=10)
if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
plt.ylabel(r'$\mathbf {(n > M)} \left[\mathbf{{h}^{3}{Mpc}^{-3}}\right]$')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
plt.show()

## Pick one mock and plot central SHMR where each point is a galaxy. Plot SHMR 
## Vishnu with error bars (std of points in bin) to see if scatter
## (maybe assymetric) could explain different SHMR.
cen_halos_arr = []
cen_gals_arr = []
box_id_arr = np.linspace(5001,5008,8)
num_mocks_arr = np.linspace(0,7,8)

box = int(random.choice(box_id_arr))
num = int(random.choice(num_mocks_arr))
temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
    mock_name) 
filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
    mock_name, num)
mock_pd = reading_catls(filename) 

mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
    (mock_pd.cz.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
    (mock_pd.logmstar.values >= mstar_limit)]
cen_halos, sat_halos = get_host_halo_mock(mock_pd, 'other')
cen_gals, sat_gals = get_stellar_mock(mock_pd, 'other')
cen_gals = np.log10((10**cen_gals)/ 2.041)
cen_halos_arr.append(cen_halos)
cen_gals_arr.append(cen_gals)

x_bf,y_bf,x_std_bf,y_std_err_bf,x_data,y_data = Stats_one_arr(cen_halos_bf,
    cen_gals_bf,base=0.4,bin_statval='center',arr_digit='y')

y_std_arr = []
for arr in y_data:
    y_std_arr.append(np.std(arr))

fig1 = plt.figure(figsize=(10,10))
plt.scatter(cen_halos_arr,cen_gals_arr, color='lightgray', s=20, 
    label=r'mock', zorder=10)
# plt.scatter(cen_halos_bf, cen_gals_bf, color='mediumorchid', s=20, zorder=5, 
#     alpha=0.2)
plt.errorbar(x_bf,y_bf,yerr=y_std_arr, color='darkviolet',fmt='-s',
    ecolor='darkviolet',markersize=7,capsize=7,capthick=1,linewidth=3,
    label=r'best-fit',zorder=20)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
if mf_type == 'smf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**8.9)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**8.7)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
elif mf_type == 'bmf':
    if survey == 'eco' or survey == 'resolvea':
        plt.ylim(np.log10((10**9.4)/2.041),)
    elif survey == 'resolveb':
        plt.ylim(np.log10((10**9.1)/2.041),)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 20})
plt.show()


