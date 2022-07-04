from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import os


__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)


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

def models_add_avgrpcz(df, grpid_col=None, galtype_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)

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

def behroozi10(logmstar, bf_params):
    """ 
    This function calculates the B10 stellar to halo mass relation 
    using the functional form 
    """
    M_1, Mstar_0, beta, delta = bf_params[:4]
    gamma = model_init.param_dict['smhm_gamma_0']
    second_term = (beta*np.log10((10**logmstar)/(10**Mstar_0)))
    third_term_num = (((10**logmstar)/(10**Mstar_0))**delta)
    third_term_denom = (1 + (((10**logmstar)/(10**Mstar_0))**(-gamma)))
    logmh = M_1 + second_term + (third_term_num/third_term_denom) - 0.5

    return logmh

def halocat_init(halo_catalog,z_median):
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

def populate_mock(theta):
    """
    Populate mock based on five parameter values 

    Parameters
    ----------
    theta: array
        Array of parameter values

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model_init.param_dict['smhm_m1_0'] = mhalo_characteristic
    model_init.param_dict['smhm_m0_0'] = mstellar_characteristic
    model_init.param_dict['smhm_beta_0'] = mlow_slope
    model_init.param_dict['smhm_delta_0'] = mhigh_slope
    model_init.param_dict['scatter_model_param1'] = mstellar_scatter

    model_init.mock.populate()

    # if survey == 'eco' or survey == 'resolvea':
    #     limit = np.round(np.log10((10**8.9) / 2.041), 1)
    #     sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    # elif survey == 'resolveb':
    #     limit = np.round(np.log10((10**8.7) / 2.041), 1)
    #     sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model_init.mock.galaxy_table#[sample_mask]
    gals_df = pd.DataFrame(np.array(gals))

    return gals_df

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

def hybrid_quenching_model(theta, gals_df, mock, randint=None):
    """
    Apply hybrid quenching model from Zu and Mandelbaum 2015

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    """

    # parameter values from Table 1 of Zu and Mandelbaum 2015 "prior case"
    Mstar_q = theta[0] # Msun/h
    Mh_q = theta[1] # Msun/h
    mu = theta[2]
    nu = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)
    cen_stellar_mass_arr, sat_stellar_mass_arr = get_stellar_mock(gals_df, mock, \
        randint)

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/(10**Mh_q))**nu))
    f_red_sat = 1 - (g_Mstar * h_Mh)

    return f_red_cen, f_red_sat, cen_stellar_mass_arr

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
        cen_halos = df.halo_mvir[df.cs_flag == 1].reset_index(drop=True)
        sat_halos = df.halo_mvir_host_halo[df.cs_flag == 0].reset_index(drop=True)
    else:
        # Both cen and sat are the same mass for a halo i.e. the satellites
        # are assigned a halo mass of the central. 
        cen_halos = 10**(df.loghalom[df.cs_flag == 1]).reset_index(drop=True)
        sat_halos = 10**(df.loghalom[df.cs_flag == 0]).reset_index(drop=True)

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
        cen_gals = 10**(df.logmstar[df.cs_flag == 1]).reset_index(drop=True)
        sat_gals = 10**(df.logmstar[df.cs_flag == 0]).reset_index(drop=True)
    
    else:
        cen_gals = []
        sat_gals = []
        for idx,value in enumerate(df.cs_flag):
            if value == 1:
                cen_gals.append((10**(df.logmstar.values[idx]))/2.041)
            elif value == 0:
                sat_gals.append((10**(df.logmstar.values[idx]))/2.041)

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

global model_init
global survey
global mf_type
survey = 'eco'
mf_type = 'smf'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
catl_original = path_to_raw + 'eco/eco_all.csv'

behroozi10_params = np.array([12.35, 10.72, 0.44, 0.57, 0.15])
mstar_min = np.round(np.log10((10**8)/2.041),1) 
mstar_max = np.round(np.log10((10**12)/2.041),1) 
logmstar_arr_or = np.linspace(mstar_min, mstar_max, 500)
catl, volume, z_median = read_data_catl(catl_file, survey)
catl_or = pd.read_csv(catl_original)
catl_or = catl_or.loc[catl_or.logmstar > 0]
model_init = halocat_init(halo_catalog, z_median)

logmh_total = behroozi10(logmstar_arr_or, behroozi10_params)

rc('text', usetex=False)
with plt.xkcd():
    # Random SHMR
    fig1 = plt.figure(figsize=(5,5))
    plt.plot(logmh_total, logmstar_arr_or, c='k', lw=5)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel('Halo Mass', fontsize=20)
    plt.ylabel('Stellar Mass', fontsize=20)
    plt.savefig('/Users/asadm2/Desktop/cartoon_shmr.pdf')

with plt.xkcd():
    fig2 = plt.figure(figsize=(5,5))
    cm = plt.cm.get_cmap('tab20b')
    sc = plt.scatter(catl_or.radeg, catl_or.dedeg, c=catl_or.logmstar, cmap=cm, s=20)
    #*Modified catalog was made after applying stellar mass cut so all galaxies
    #* are above h=1 equivalent of 8.9 for ECO
    plt.scatter(catl_or.radeg.loc[catl_or.logmstar < 8.9], 
        catl_or.dedeg.loc[catl_or.logmstar < 8.9], c='lightgray', s=20)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel('RA', fontsize=60)
    plt.ylabel('DEC', fontsize=60)
    plt.colorbar(sc)
    plt.show()

    # ECO SMF
    fig3 = plt.figure(figsize=(7,5))
    volume = 151829.26
    maxis, phi, err_tot, bins, counts = diff_smf(catl.logmstar.values, volume, False)
    plt.plot(maxis, phi, c='k', lw=5)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel(r'$M_{*}$', fontsize=20)
    plt.ylabel(r'$\Phi$', fontsize=20)
    plt.show()

    mstar_cen, f_red_sat, mstar_mhalo_sat, mstar_sat

    # Quenching function cartoon
    #* Params from chain 50
    bf_params = [12.10864609, 10.50681389, 0.37964697, 0.65261742, 0.34880853,
        10.11453861, 13.69516435, 0.7229029 , 0.05319513]
    gals_df = populate_mock(bf_params[:5])
    gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)
    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)
    gals_df['logmstar'] = np.log10(gals_df['logmstar'])
    f_red_cen, f_red_sat, mstar_cen = \
        hybrid_quenching_model(bf_params[5:], gals_df, 'vishnu')
with plt.xkcd():
    fig4 = plt.figure(figsize=(6,6))
    plt.scatter(np.log10(mstar_cen), f_red_cen)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel(r'$f_{red}$', fontsize=40)
    plt.xlabel(r'$M_{*, cen}$',fontsize=40)
    plt.savefig('/Users/asadm2/Desktop/cartoon_fred.pdf')

    # ECO SMF red and blue cartoon
    fig5 = plt.figure(figsize=(7,5))
    volume = 151829.26
    maxis, phi, err_tot, bins, counts = diff_smf(catl.logmstar.values, volume, False)
    plt.plot(maxis, phi, c='r', lw=5)
    plt.plot(maxis + 1, phi, c='b', lw=5)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel(r'$M_{*}$', fontsize=20)
    plt.ylabel(r'$\Phi$', fontsize=20)
    plt.show()

with plt.xkcd():
    #Blue fraction of centrals and satellites - values copied from data
    fig4 = plt.figure(figsize=(6,6))
    x = [ 9.3625,  9.8875, 10.4125, 10.9375]
    y_cen = [0.85365854, 0.64826176, 0.3389313 , 0.07236842]
    y_sat = [0.59922929, 0.47792998, 0.25229358, 0.2173913 ]
    plt.plot(x, y_cen, ls='-', lw=2, c='k', label='cen')
    plt.plot(x, y_sat, ls='--', lw=2, c='k', label='sat')
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel(r'$M$',fontsize=20)
    plt.ylabel(r'$f_{blue}$', fontsize=20)
    plt.legend(loc='best')
    plt.savefig('/Users/asadm2/Desktop/cartoon_fblue_cen_sat.pdf')

with plt.xkcd():
    #Velocity dispersion around red and blue group centrals - values copied 
    #from data
    fig4 = plt.figure(figsize=(6,6))
    x_red = np.array([-2.  , -0.75,  0.5 ,  1.75,  3.  ])
    x_blue = np.array([-1.,  0.,  1.,  2.,  3.])
    x_red = 0.5 * (x_red[1:] + x_red[:-1])
    x_blue = 0.5 * (x_blue[1:] + x_blue[:-1])

    y_red = [ 9.835157  , 10.30420445, 10.45091222, 10.57060556]
    y_blue = [10.05761284,  9.96185817, 10.05961179, 10.13817874]
    plt.plot(x_red, y_red, ls='-', lw=2, c='r')
    plt.plot(x_blue, y_blue, ls='-', lw=2, c='b')
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel(r'$\sigma}$', fontsize=20)
    plt.ylabel(r'$\overline{M}$',fontsize=20)
    plt.savefig('/Users/asadm2/Desktop/cartoon_sigma_mstar.pdf')

rc('text', usetex=True)
# SHMR plot to show all 5 parameters
fig4 = plt.figure(figsize=(7,5))
plt.plot(logmh_blue, logmstar_arr_or, c='k', lw=2, zorder=10)   
plt.plot(logmh_rand, logmstar_rand, c='lightgray', lw=10, zorder=5)
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
plt.ylabel(r'\boldmath$\log_{10}\ M_{*} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
plt.xlim(min(logmh_blue),15)
plt.ylim(min(logmstar_arr_or),)

# Inset of fig4 showing error parameter
left, bottom, width, height = [0.60, 0.17, 0.25, 0.25]
ax2 = fig4.add_axes([left, bottom, width, height])
ax2.plot(logmh_blue[250:], logmstar_arr_or[250:], c='k', lw=2, zorder=10)
for i in range(250):
    ax2.plot(logmh_rand[:250][i][250:], logmstar_rand[:250][i][250:], 
    c='lightgray', lw=20, zorder=5)
ax2.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
ax2.set_xlim(13,15)
plt.show()

# Plot of best fit mass relations for ECO and RESOLVE A
x_bf_eco_smf = [10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8,
    14.2, 14.6, 15.0]
y_bf_eco_smf = [8.666693, 8.748637, 8.958516, 9.500914, 9.99621, 10.248682,
    10.415832, 10.524332, 10.633187, 10.736437, 10.901241, 10.973494, 10.818796]

x_bf_eco_bmf = [10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8,
    14.2, 14.6, 15.0]
y_bf_eco_bmf = [9.183136, 9.263569, 9.430301, 9.741157, 10.0494585, 10.257879,
    10.418935, 10.558106, 10.729276, 10.766456, 10.943247, 10.977773,
    11.014101]

x_bf_ra_smf = [10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8,
    14.2, 14.6, 15.0]
y_bf_ra_smf = [8.65164, 8.755341, 9.001323, 9.570955, 10.021729, 10.236013,
    10.362319, 10.463475, 10.582119, 10.681992, 10.7281475, 10.706401,
    10.917749]

x_bf_ra_bmf = [10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8,
    14.2, 14.6, 15.0]
y_bf_ra_bmf = [9.176975, 9.247481, 9.423248, 9.794446, 10.132633, 10.333286,
    10.478376, 10.582316, 10.692095, 10.754516, 10.839354, 11.069578, 10.517319]

fig5 = plt.figure(figsize=(10,10))
plt.errorbar(x_bf_eco_smf,y_bf_eco_smf,color='#8511C0',fmt='-s',ecolor='#8511C0',\
    markersize=4,capsize=5,capthick=0.5,label='ECO shmr',zorder=10)
plt.errorbar(x_bf_eco_bmf,y_bf_eco_bmf,color='darkcyan',fmt='-s',ecolor='darkcyan',\
    markersize=4,capsize=5,capthick=0.5,label='ECO bhmr',zorder=10)
plt.errorbar(x_bf_ra_smf,y_bf_ra_smf,color='#E766EA',fmt='--s',ecolor='#E766EA',\
    markersize=4,capsize=5,capthick=0.5,label='RESOLVE A shmr',zorder=10)
plt.errorbar(x_bf_ra_bmf,y_bf_ra_bmf,color='#53A48D',fmt='--s',ecolor='#53A48D',\
    markersize=4,capsize=5,capthick=0.5,label='RESOLVE A bhmr',zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
plt.ylabel(r'\boldmath$\log_{10}\ M_\star (M_{b}) \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
plt.legend(loc='best',prop={'size': 20})
plt.show()

# Quenching function plot
chi2 = read_chi2(chi2_file)
mcmc_table = read_mcmc(chain_file)
bf_params = get_paramvals_percentile(mcmc_table, 68, chi2)
gals_df = populate_mock(bf_params)
gals_df = assign_cen_sat_flag(gals_df)
f_red_cen, mstar_cen, f_red_sat, mstar_mhalo_sat, mstar_sat = \
    hybrid_quenching_model(gals_df)
fig6 = plt.figure(figsize=(10,8))
plt.scatter(np.log10(mstar_cen), f_red_cen)
plt.ylabel(r'\boldmath$f_{red}$', fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
plt.show()
fig5 = plt.figure(figsize=(10,8))
plt.scatter(np.log10(mstar_sat), f_red_sat)
plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, sat} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)