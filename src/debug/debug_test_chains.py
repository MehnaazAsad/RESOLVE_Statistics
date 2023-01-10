"""
{This script plots SMF from test chains where pair-splitting, dr3 and 
obs 3 new binning were tested.}
"""
import emcee
import pandas as pd
import os
import numpy as np
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

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

        eco_buff = read_mock_catl(path_to_file)
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

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas.DataFrame
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    data_bool: boolean
        Data or mock
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    3 multidimensional arrays of [stellar mass, phi, total error in SMF and 
    counts per bin] for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
        return [max_total, phi_total, err_total, counts_total]# , \
            # [max_red, phi_red, err_red, counts_red] , \
            #     [max_blue, phi_blue, err_blue, counts_blue]
    else:
        if randint_logmstar != 1:
            logmstar_col = '{0}'.format(randint_logmstar)
        elif randint_logmstar == 1:
            logmstar_col = 'behroozi_bf'
        else:
            logmstar_col = 'stellar_mass'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, bins_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')
        return [max_total, phi_total, err_total, counts_total]

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
    
    colour_flag (optional): boolean
        'R' if galaxy masses correspond to red galaxies & 'B' if galaxy masses
        correspond to blue galaxies. Defaults to False.

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
    
    counts: array
        Array of number of things in each bin
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

def read_mock_catl(filename, catl_format='.hdf5'):
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

def get_chain_df(filepath):
    reader = emcee.backends.HDFBackend(filepath, read_only=True)
    chi2 = reader.get_blobs(flat=True)

    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope'
    , 'mhigh_slope', 'scatter', 'mstar_q','mh_q','mu','nu']

    flatchain = reader.get_chain(flat=True)
    mcmc_table = pd.DataFrame(flatchain, columns=colnames)

    return mcmc_table, chi2

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def get_err_data_71(survey, path):
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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = read_mock_catl(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 50
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 49
            ## i.e. halo quenching model
            bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

            # max_total, phi_total, err_total, bins_total, counts_total = \
            #     diff_bmf(logmbary_arr, volume, False)
            # phi_total_arr.append(phi_total)
            if mf_type == 'smf':
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_total_arr.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)


            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1.4,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1,3,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1.4,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, corr_mat_inv_colour


survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
level = 'group'
stacked_stat = False

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

file_dr3 = "/Users/asadm2/Desktop/chain_hybrid.h5"
file_ps = "/Users/asadm2/Desktop/chain_hybrid_pairsplitting.h5"
file_bin = "/Users/asadm2/Desktop/chain_hybrid_newbinning.h5"

mcmc_table_dr3, chi2_dr3 = get_chain_df(file_dr3)
mcmc_table_ps, chi2_ps = get_chain_df(file_ps)
mcmc_table_bin, chi2_bin = get_chain_df(file_bin)

mcmc_table_pctl, bf_params, bf_chi2 = \
    get_paramvals_percentile(mcmc_table_bin, 68, chi2_bin)

#########################################################################
#* CHAIN 71 - DR3

gal_group = read_mock_catl("/Users/asadm2/Desktop/gal_group_run71.hdf5") 

idx_arr = np.insert(np.linspace(1,20,21), len(np.linspace(0,20,21)), \
    (22, 123, 124, 125, 126, 127, 128, 129)).\
    astype(int)

names_arr = [x for x in gal_group.columns.values[idx_arr]]
for idx in np.arange(2,102,1):
    names_arr.append('{0}_y'.format(idx))
    names_arr.append('groupid_{0}'.format(idx))
    names_arr.append('grp_censat_{0}'.format(idx))
    names_arr.append('cen_cz_{0}'.format(idx))
names_arr = np.array(names_arr)

gal_group_df_subset = gal_group[names_arr]

# Renaming the "1_y" column kept from line 1896 because of case where it was
# also in mcmc_table_ptcl.mock_num and was selected twice
gal_group_df_subset.columns.values[25] = "behroozi_bf"

### Removing "_y" from column names for stellar mass
# Have to remove the first element because it is 'halo_y' column name
cols_with_y = np.array([[idx, s] for idx, s in enumerate(
    gal_group_df_subset.columns.values) if '_y' in s][1:])
colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
gal_group_df_subset.columns.values[cols_with_y[:,0].\
    astype(int)] = colnames_without_y

v_sim = 890641.5172927063  ## cz: 3000-12000
randint_logmstar = 1
cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
'halo_mvir_host_halo', 'cz', 'cs_flag', \
'behroozi_bf', \
'grp_censat_{0}'.format(randint_logmstar), \
'groupid_{0}'.format(randint_logmstar),\
'cen_cz_{0}'.format(randint_logmstar)]

gals_df = gal_group_df_subset[cols_to_use]

gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
'cen_cz_{0}'.format(randint_logmstar)]).\
reset_index(drop=True)

gals_df[['grp_censat_{0}'.format(randint_logmstar), \
    'groupid_{0}'.format(randint_logmstar)]] = \
    gals_df[['grp_censat_{0}'.format(randint_logmstar),\
    'groupid_{0}'.format(randint_logmstar)]].astype(int)
            
gals_df['behroozi_bf'] = np.log10(gals_df['behroozi_bf'])

total_bf = measure_all_smf(gals_df, v_sim , False, 
    randint_logmstar)    

total_models = []
for i in range(2, 102, 1):
    randint_logmstar = i
    cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
    'halo_mvir_host_halo', 'cz', 'cs_flag', \
    '{0}'.format(randint_logmstar), \
    'grp_censat_{0}'.format(randint_logmstar), \
    'groupid_{0}'.format(randint_logmstar), \
    'cen_cz_{0}'.format(randint_logmstar)]

    gals_df = gal_group_df_subset[cols_to_use]

    gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
    format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
    'cen_cz_{0}'.format(randint_logmstar)]).\
    reset_index(drop=True)

    gals_df[['grp_censat_{0}'.format(randint_logmstar), \
        'groupid_{0}'.format(randint_logmstar)]] = \
        gals_df[['grp_censat_{0}'.format(randint_logmstar),\
        'groupid_{0}'.format(randint_logmstar)]].astype(int)

    gals_df['{0}'.format(randint_logmstar)] = \
        np.log10(gals_df['{0}'.format(randint_logmstar)])

    total_model = measure_all_smf(gals_df, v_sim , False, 
        randint_logmstar)    
    total_models.append(total_model[1])

catl_file = catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3_nops.hdf5"
catl, volume, z_median = read_data_catl(catl_file, survey)
total_data = measure_all_smf(catl, volume, True)

path_to_mocks = path_to_data + 'mocks/m200b/eco/'
sigma, mat = get_err_data_71(survey, path_to_mocks)

### Plotting
x_phi_total_data, y_phi_total_data = total_data[0], total_data[1]

x_phi_total_model = total_data[0]
tot_phi_max = np.amax(total_models, axis=0)
tot_phi_min = np.amin(total_models, axis=0)

x_phi_total_bf, y_phi_total_bf = total_bf[0], total_bf[1]

fig1= plt.figure(figsize=(10,10))
mt = plt.fill_between(x=x_phi_total_model, y1=tot_phi_max, 
    y2=tot_phi_min, color='silver', alpha=0.4)

dt = plt.errorbar(x_phi_total_data, y_phi_total_data, yerr=sigma[:4],
    color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')

# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bft, = plt.plot(x_phi_total_bf, y_phi_total_bf, color='k', ls='--', lw=4, 
    zorder=10)


plt.ylim(-4,-1)

if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='lower left')
plt.show()

#* CHAIN 72 - New binning

def get_err_data_72(survey, path):
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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = read_mock_catl(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 50
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 49
            ## i.e. halo quenching model
            bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

            # max_total, phi_total, err_total, bins_total, counts_total = \
            #     diff_bmf(logmbary_arr, volume, False)
            # phi_total_arr.append(phi_total)
            if mf_type == 'smf':
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, bins_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_total_arr.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)


            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, corr_mat_inv_colour

gal_group = read_mock_catl("/Users/asadm2/Desktop/gal_group_run72.hdf5") 

idx_arr = np.insert(np.linspace(1,20,21), len(np.linspace(0,20,21)), \
    (22, 123, 124, 125, 126, 127, 128, 129)).\
    astype(int)

names_arr = [x for x in gal_group.columns.values[idx_arr]]
for idx in np.arange(2,102,1):
    names_arr.append('{0}_y'.format(idx))
    names_arr.append('groupid_{0}'.format(idx))
    names_arr.append('grp_censat_{0}'.format(idx))
    names_arr.append('cen_cz_{0}'.format(idx))
names_arr = np.array(names_arr)

gal_group_df_subset = gal_group[names_arr]

# Renaming the "1_y" column kept from line 1896 because of case where it was
# also in mcmc_table_ptcl.mock_num and was selected twice
gal_group_df_subset.columns.values[25] = "behroozi_bf"

### Removing "_y" from column names for stellar mass
# Have to remove the first element because it is 'halo_y' column name
cols_with_y = np.array([[idx, s] for idx, s in enumerate(
    gal_group_df_subset.columns.values) if '_y' in s][1:])
colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
gal_group_df_subset.columns.values[cols_with_y[:,0].\
    astype(int)] = colnames_without_y

v_sim = 890641.5172927063  ## cz: 3000-12000
randint_logmstar = 1
cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
'halo_mvir_host_halo', 'cz', 'cs_flag', \
'behroozi_bf', \
'grp_censat_{0}'.format(randint_logmstar), \
'groupid_{0}'.format(randint_logmstar),\
'cen_cz_{0}'.format(randint_logmstar)]

gals_df = gal_group_df_subset[cols_to_use]

gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
'cen_cz_{0}'.format(randint_logmstar)]).\
reset_index(drop=True)

gals_df[['grp_censat_{0}'.format(randint_logmstar), \
    'groupid_{0}'.format(randint_logmstar)]] = \
    gals_df[['grp_censat_{0}'.format(randint_logmstar),\
    'groupid_{0}'.format(randint_logmstar)]].astype(int)
            
gals_df['behroozi_bf'] = np.log10(gals_df['behroozi_bf'])

total_bf = measure_all_smf(gals_df, v_sim , False, 
    randint_logmstar)    

total_models = []
for i in range(2, 102, 1):
    randint_logmstar = i
    cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
    'halo_mvir_host_halo', 'cz', 'cs_flag', \
    '{0}'.format(randint_logmstar), \
    'grp_censat_{0}'.format(randint_logmstar), \
    'groupid_{0}'.format(randint_logmstar), \
    'cen_cz_{0}'.format(randint_logmstar)]

    gals_df = gal_group_df_subset[cols_to_use]

    gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
    format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
    'cen_cz_{0}'.format(randint_logmstar)]).\
    reset_index(drop=True)

    gals_df[['grp_censat_{0}'.format(randint_logmstar), \
        'groupid_{0}'.format(randint_logmstar)]] = \
        gals_df[['grp_censat_{0}'.format(randint_logmstar),\
        'groupid_{0}'.format(randint_logmstar)]].astype(int)

    gals_df['{0}'.format(randint_logmstar)] = \
        np.log10(gals_df['{0}'.format(randint_logmstar)])

    total_model = measure_all_smf(gals_df, v_sim , False, 
        randint_logmstar)    
    total_models.append(total_model[1])

catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
catl, volume, z_median = read_data_catl(catl_file, survey)
total_data = measure_all_smf(catl, volume, True)

path_to_mocks = path_to_data + 'mocks/m200b/eco/'
sigma, mat = get_err_data_72(survey, path_to_mocks)

### Plotting
x_phi_total_data, y_phi_total_data = total_data[0], total_data[1]

x_phi_total_model = total_data[0]
tot_phi_max = np.amax(total_models, axis=0)
tot_phi_min = np.amin(total_models, axis=0)

x_phi_total_bf, y_phi_total_bf = total_bf[0], total_bf[1]

fig1= plt.figure(figsize=(10,10))
mt = plt.fill_between(x=x_phi_total_model, y1=tot_phi_max, 
    y2=tot_phi_min, color='silver', alpha=0.4)

dt = plt.errorbar(x_phi_total_data, y_phi_total_data, yerr=sigma[:4],
    color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')

# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bft, = plt.plot(x_phi_total_bf, y_phi_total_bf, color='k', ls='--', lw=4, 
    zorder=10)


plt.ylim(-4,-1)

if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='lower left')
plt.show()

#####################################################################

counts_dr3 = np.array([16046, 11449,  7264,  1188]) # DR3
counts_newbins = np.array([23690, 18225, 11770,  2115]) # New binning

counts_newbins/counts_dr3

x = [8.925,  9.575, 10.225, 10.875]
y_ps = [-1.45277329, -1.59191464, -1.77499471, -2.44247001]
y_dr3 = [-1.5572495 , -1.70384874, -1.90144046, -2.68779985]
y_newbin = [-1.38784962, -1.51080779, -1.69763447, -2.4348488]

fig1= plt.figure(figsize=(10,10))

# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
plt.plot(x, y_ps, color='k', ls='--', lw=4, label='pair-splitting bf')
plt.plot(x, y_newbin, color='k', ls='-', lw=4, label='new binning bf')
plt.plot(x, y_dr3, color='k', ls='-.', lw=4, label='dr3 bf')

plt.ylim(-4,-1)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=30)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

plt.legend(loc='lower left', prop={'size':25})
plt.show()

#####################################################################
#* Chi-squared test using chain 72 : new binning and plotting mstar-sigma

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import emcee 
import math
import os

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

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

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

def get_err_data(survey, path, bin_i):
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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 50
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 49
            ## i.e. halo quenching model
            bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

            # max_total, phi_total, err_total, bins_total, counts_total = \
            #     diff_bmf(logmbary_arr, volume, False)
            # phi_total_arr.append(phi_total)
            if mf_type == 'smf':
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_total_arr.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)


            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

    num_cols_by_idx = combined_df.shape[1] - 1
    combined_df = combined_df.drop(combined_df.columns[num_cols_by_idx - bin_i], 
        axis=1)

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, \
        gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_new"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['cen_cz'] >= cz_inner_mod) &
        (gal_group_df['cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar'
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.9,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour to data using u-r colours')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if stacked_stat:
        if mf_type == 'smf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])
        elif mf_type == 'bmf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mbary_sigma, blue_deltav, \
                blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])

    else:
        if mf_type == 'smf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.9,5))
        elif mf_type == 'bmf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mbary_sigma, blue_sigma, \
                blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.9,5))

    if stacked_stat:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    else:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    return data_arr, z_median

def get_err_data_original(survey, path):
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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 50
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 49
            ## i.e. halo quenching model
            bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

            # max_total, phi_total, err_total, bins_total, counts_total = \
            #     diff_bmf(logmbary_arr, volume, False)
            # phi_total_arr.append(phi_total)
            if mf_type == 'smf':
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_total_arr.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)


            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.9,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
    # rc('text', usetex=True)
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # #* Reduced feature space
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(T.corr(), cmap=cmap)
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Reconstructed post-SVD (sub corr_mat_colour for original matrix)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(corr_mat_colour, cmap=cmap, vmin=-1, vmax=1)
    # # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]

    # plt.xticks(tick_marks, names, rotation='vertical')
    # plt.yticks(tick_marks, names)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Scree plot

    # percentage_variance = []
    # for val in s:
    #     sum_of_eigenvalues = np.sum(s)
    #     percentage_variance.append((val/sum_of_eigenvalues)*100)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.bar(np.arange(1, len(s)+1, 1), percentage_variance, color='#663399',
    #     zorder=5)
    # ax1.scatter(np.arange(1, len(s)+1, 1), s, c='orange', s=50, zorder=10)
    # ax1.plot(np.arange(1, len(s)+1, 1), s, 'k--', zorder=10)
    # ax1.hlines(max_eigen, 0, 20, colors='orange', zorder=10, lw=2)
    # ax1.set_xlabel('Component number')
    # ax1.set_ylabel('Singular values')
    # ax2.set_ylabel('Percentage of variance')
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.set_frame_on(False)
    # ax1.set_xticks(np.arange(1, len(s)+1, 1))
    # plt.show()

    ############################################################################
    #* Observable plots for paper
    #* Total SMFs from mocks and data for paper

    # data = combined_df.values[:,:4]

    # upper_bound = np.nanmean(data, axis=0) + \
    #     np.nanstd(data, axis=0)
    # lower_bound = np.nanmean(data, axis=0) - \
    #     np.nanstd(data, axis=0)

    # phi_max = []
    # phi_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data.T[idx]
    #         >=lower_bound[idx], data.T[idx]<=upper_bound[idx]))
    #     nums = data.T[idx][idxs]
    #     phi_min.append(min(nums))
    #     phi_max.append(max(nums))

    # fig1 = plt.figure()
    # mt = plt.fill_between(x=max_total, y1=phi_max, 
    #     y2=phi_min, color='silver', alpha=0.4)
    # dt = plt.scatter(total_data[0], total_data[1],
    #     color='k', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)

    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_smf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_bmf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()

    # #* Blue fraction from mocks and data for paper

    # data_cen = combined_df.values[:,4:8]
    # data_sat = combined_df.values[:,8:12]

    # upper_bound = np.nanmean(data_cen, axis=0) + \
    #     np.nanstd(data_cen, axis=0)
    # lower_bound = np.nanmean(data_cen, axis=0) - \
    #     np.nanstd(data_cen, axis=0)

    # cen_max = []
    # cen_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_cen.T[idx]
    #         >=lower_bound[idx], data_cen.T[idx]<=upper_bound[idx]))
    #     nums = data_cen.T[idx][idxs]
    #     cen_min.append(min(nums))
    #     cen_max.append(max(nums))

    # upper_bound = np.nanmean(data_sat, axis=0) + \
    #     np.nanstd(data_sat, axis=0)
    # lower_bound = np.nanmean(data_sat, axis=0) - \
    #     np.nanstd(data_sat, axis=0)

    # sat_max = []
    # sat_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_sat.T[idx]
    #         >=lower_bound[idx], data_sat.T[idx]<=upper_bound[idx]))
    #     nums = data_sat.T[idx][idxs]
    #     sat_min.append(min(nums))
    #     sat_max.append(max(nums))

    # fig2 = plt.figure()

    # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max, 
    #     y2=cen_min, color='rebeccapurple', alpha=0.4)
    # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max, 
    #     y2=sat_min, color='goldenrod', alpha=0.4)

    # dt_cen = plt.scatter(f_blue_data[0], f_blue_data[2],
    #     color='rebeccapurple', s=150, zorder=10, marker='^')
    # dt_sat = plt.scatter(f_blue_data[0], f_blue_data[3],
    #     color='goldenrod', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
    # plt.ylim(0,1)
    # # plt.title(r'Blue fractions from mocks and data')
    # plt.legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    #     ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()


    # #* Velocity dispersion from mocks and data for paper

    # bins_red=np.linspace(-2,3,5)
    # bins_blue=np.linspace(-1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # data_red = combined_df.values[:,12:16]
    # data_blue = combined_df.values[:,16:20]

    # upper_bound = np.nanmean(data_red, axis=0) + \
    #     np.nanstd(data_red, axis=0)
    # lower_bound = np.nanmean(data_red, axis=0) - \
    #     np.nanstd(data_red, axis=0)

    # red_max = []
    # red_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_red.T[idx]
    #         >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
    #     nums = data_red.T[idx][idxs]
    #     red_min.append(min(nums))
    #     red_max.append(max(nums))

    # upper_bound = np.nanmean(data_blue, axis=0) + \
    #     np.nanstd(data_blue, axis=0)
    # lower_bound = np.nanmean(data_blue, axis=0) - \
    #     np.nanstd(data_blue, axis=0)

    # blue_max = []
    # blue_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_blue.T[idx]
    #         >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
    #     nums = data_blue.T[idx][idxs]
    #     blue_min.append(min(nums))
    #     blue_max.append(max(nums))

    # fig3 = plt.figure()
    # mt_red = plt.fill_between(x=bins_red, y1=red_min, 
    #     y2=red_max, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=blue_min, 
    #     y2=blue_max, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
    #     color='indianred', s=150, zorder=10, marker='^')
    # dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
    #     color='cornflowerblue', s=150, zorder=10, marker='^')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)

    # plt.show()
    ############################################################################
    # ## Stacked sigma from mocks and data for paper
    # if mf_type == 'smf':
    #     bin_min = 8.6
    # elif mf_type == 'bmf':
    #     bin_min = 9.1
    # bins_red=np.linspace(bin_min,11.2,5)
    # bins_blue=np.linspace(bin_min,11.2,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # mean_mstar_red_max = np.nanmax(combined_df.values[:,12:16], axis=0)
    # mean_mstar_red_min = np.nanmin(combined_df.values[:,12:16], axis=0)
    # mean_mstar_blue_max = np.nanmax(combined_df.values[:,16:20], axis=0)
    # mean_mstar_blue_min = np.nanmin(combined_df.values[:,16:20], axis=0)

    # error = np.nanstd(combined_df.values[:,12:], axis=0)

    # fig2 = plt.figure()

    # mt_red = plt.fill_between(x=bins_red, y1=mean_mstar_red_max, 
    #     y2=mean_mstar_red_min, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=mean_mstar_blue_max, 
    #     y2=mean_mstar_blue_min, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.errorbar(bins_red, sigma_red_data, yerr=error[:4],
    #     color='indianred', fmt='s', ecolor='indianred', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')
    # dt_blue = plt.errorbar(bins_blue, sigma_blue_data, yerr=error[4:8],
    #     color='cornflowerblue', fmt='s', ecolor='cornflowerblue', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')


    # plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)       
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # # plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    # #     bbox_inches="tight", dpi=1200)

    # plt.show()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour


global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global new_chain
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = False
pca = False
new_chain = True

survey = 'eco'
machine = 'bender'
mf_type = 'smf'
quenching = 'hybrid'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_data_buffer_volh1_dr2.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'


data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

bf_bin_params = [12.0521335 , 10.53874947,  0.38734879,  0.59748402,  0.30480815,
            10.1291546 , 13.56838266,  0.72976532,  0.06375868]
bf_bin_chi2 = 30.50556557
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, mat = get_err_data_original(survey, path_to_mocks)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

# Plot of sigma observable (with model binned both ways to compare with new data
# binning)
fig1= plt.figure(figsize=(10,10))

bins_red_data = np.linspace(1,3,5)
bins_blue_data = np.linspace(1,2.9,5)
bin_centers_red_data = 0.5 * (bins_red_data[1:] + bins_red_data[:-1])
bin_centers_blue_data = 0.5 * (bins_blue_data[1:] + bins_blue_data[:-1])

bins_red_model = np.linspace(1,3,5)
bins_blue_model = np.linspace(1,2.9,5)
bin_centers_red_model = 0.5 * (bins_red_model[1:] + bins_red_model[:-1])
bin_centers_blue_model = 0.5 * (bins_blue_model[1:] + bins_blue_model[:-1])

dr = plt.errorbar(bin_centers_red_data, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue_data, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red_model, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue_model, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

chi2_arr = []
for i in range(20):
    if i == 0:
        chi2 = chi_squared(data_stats, model_stats, sigma, mat)
        chi2_arr.append(chi2)
    print(i)
    sigma, mat = get_err_data(survey, path_to_mocks, i)

    num_elems_by_idx = len(data_stats) - 1
    data_stats_mod  = np.delete(data_stats, num_elems_by_idx-i) 
    model_stats_mod  = np.delete(model_stats, num_elems_by_idx-i) 

    chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma, mat)
    chi2_arr.append(chi2)


chi2_arr = [36.496817210024474,
 32.67650162913964,
 30.74097619726093,
 36.64552267225759,
 41.70822664333266,
 41.40382508794237,
 35.16674219103154,
 37.925933969244625,
 46.21525205540388,
 40.84557516866762,
 39.729700790855894,
 39.98032743868651,
 45.91925067479555,
 47.21846423637512,
 56.60347917214687,
 43.57305665955035,
 31.586429855607268,
 51.531495050246896,
 31.782679189783554,
 38.59250709403704,
 41.106775952974594]


fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = [
r'$\Phi 1$', r'$\Phi 2$', r'$\Phi 3$', r'$\Phi 4$',
r'$fblue_{cen}\ 1$', r'$fblue_{cen}\ 2$', r'$fblue_{cen}\ 3$', r'$fblue_{cen}\ 4$',
r'$fblue_{sat}\ 1$', r'$fblue_{sat}\ 2$', r'$fblue_{sat}\ 3$', r'$fblue_{sat}\ 4$',
r'$\sigma_{red}\ 1$', r'$\sigma_{red}\ 2$', 
r'$\sigma_{red}\ 3$', r'$\sigma_{red}\ 4$',
r'$\sigma_{blue}\ 1$', r'$\sigma_{blue}\ 2$', 
r'$\sigma_{blue}\ 3$', r'$\sigma_{blue}\ 4$', 
'No bins removed']

plt.plot(tick_marks, chi2_arr[::-1])
plt.scatter(tick_marks, chi2_arr[::-1])
plt.xlabel("Bin removed")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names[::-1], fontsize=20, rotation='vertical')
plt.show()

# Plot of distribution of sigma and mstar for both binnings 
# (code run on bender)
red_sigma_data, red_cen_mstar_sigma_data, blue_sigma_data, \
    blue_cen_mstar_sigma_data = get_velocity_dispersion(catl, 'data')

red_sigma_data = np.log10(red_sigma_data)
blue_sigma_data = np.log10(blue_sigma_data)

mean_mstar_red_data = bs(red_sigma_data, red_cen_mstar_sigma_data, 
    statistic='count', bins=np.linspace(1,2.8,5))
mean_mstar_blue_data = bs(blue_sigma_data, blue_cen_mstar_sigma_data, 
    statistic='count', bins=np.linspace(1,2.5,5))

red_sigma_bf, red_cen_mstar_sigma_bf, blue_sigma_bf, \
    blue_cen_mstar_sigma_bf = get_velocity_dispersion(
        gal_group_df, 'model')

red_sigma_bf = np.log10(red_sigma_bf)
blue_sigma_bf = np.log10(blue_sigma_bf)

mean_mstar_red_bf = bs(red_sigma_bf, red_cen_mstar_sigma_bf, 
    statistic='count', bins=np.linspace(1,2.8,5))
mean_mstar_blue_bf = bs(blue_sigma_bf, blue_cen_mstar_sigma_bf, 
    statistic='count', bins=np.linspace(1,2.5,5))

nbins=4
red_sigma_data_arr = []
blue_sigma_data_arr = []
red_sigma_bf_arr = []
blue_sigma_bf_arr = []
for idx in range(nbins):
    red_sigma_data_i = red_sigma_data[np.where(np.array(mean_mstar_red_data[2]) == idx)[0]]            
    red_sigma_data_arr.append(red_sigma_data_i)
    blue_sigma_data_i = blue_sigma_data[np.where(np.array(mean_mstar_blue_data[2]) == idx)[0]]            
    blue_sigma_data_arr.append(blue_sigma_data_i)

    red_sigma_bf_i = red_sigma_bf[np.where(np.array(mean_mstar_red_bf[2]) == idx)[0]]            
    red_sigma_bf_arr.append(red_sigma_bf_i)
    blue_sigma_bf_i = blue_sigma_bf[np.where(np.array(mean_mstar_blue_bf[2]) == idx)[0]]            
    blue_sigma_bf_arr.append(blue_sigma_bf_i)

nbins=4
red_mstar_data_arr = []
blue_mstar_data_arr = []
red_mstar_bf_arr = []
blue_mstar_bf_arr = []
for idx in range(nbins):
    red_mstar_data_i = red_cen_mstar_sigma_data[np.where(np.array(mean_mstar_red_data[2]) == idx)[0]]            
    red_mstar_data_arr.append(red_mstar_data_i)
    blue_mstar_data_i = blue_cen_mstar_sigma_data[np.where(np.array(mean_mstar_blue_data[2]) == idx)[0]]            
    blue_mstar_data_arr.append(blue_mstar_data_i)

    red_mstar_bf_i = red_cen_mstar_sigma_bf[np.where(np.array(mean_mstar_red_bf[2]) == idx)[0]]            
    red_mstar_bf_arr.append(red_mstar_bf_i)
    blue_mstar_bf_i = red_cen_mstar_sigma_bf[np.where(np.array(mean_mstar_blue_bf[2]) == idx)[0]]            
    blue_mstar_bf_arr.append(blue_mstar_bf_i)

n_groups_data = len(red_sigma_data) + len(blue_sigma_data)
n_groups_bf = len(red_sigma_bf) + len(blue_sigma_bf)

fig1 = plt.figure()
colours = ['red', 'blue', 'orange', 'green']
for idx in range(nbins):
    plt.hist(red_sigma_data_arr[idx], density=True, histtype='step', ls='-', color=colours[idx])
    plt.hist(red_sigma_bf_arr[idx], density=True, histtype='step', ls='--', color=colours[idx])
plt.xlabel(r'$\log_{10}\ \sigma$')
plt.savefig('red_sigma_dist_bf_data_72_modbin.png')

fig2 = plt.figure()
colours = ['red', 'blue', 'orange', 'green']
for idx in range(nbins):
    plt.hist(blue_sigma_data_arr[idx][~np.isinf(blue_sigma_data_arr[idx])], density=True, histtype='step', ls='-', color=colours[idx])
    plt.hist(blue_sigma_bf_arr[idx], density=True, histtype='step', ls='--', color=colours[idx])
plt.xlabel(r'$\log_{10}\ \sigma$')
plt.savefig('blue_sigma_dist_bf_data_72_modbin.png')

fig3 = plt.figure()
colours = ['red', 'blue', 'orange', 'green']
for idx in range(nbins):
    plt.hist(red_mstar_data_arr[idx], density=True, histtype='step', ls='-', color=colours[idx])
    plt.hist(red_mstar_bf_arr[idx], density=True, histtype='step', ls='--', color=colours[idx])
plt.xlabel(r'$\log_{10}\ \sigma$')
plt.savefig('red_mstar_dist_bf_data_72_modbin.png')

fig4 = plt.figure()
colours = ['red', 'blue', 'orange', 'green']
for idx in range(nbins):
    plt.hist(blue_mstar_data_arr[idx], density=True, histtype='step', ls='-', color=colours[idx])
    plt.hist(blue_mstar_bf_arr[idx], density=True, histtype='step', ls='--', color=colours[idx])
plt.xlabel(r'$\log_{10}\ \sigma$')
plt.savefig('blue_mstar_dist_bf_data_72_modbin.png')

#####################################################################
#* Chi-squared test using chain 71 : dr3 and plotting mstar-sigma

import multiprocessing
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import emcee 
import math
import os

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

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

def assign_colour_label_data_dr2(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 
            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 50
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.11453861, 13.69516435, 0.7229029 , 0.05319513]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 49
            ## i.e. halo quenching model
            bf_from_last_chain = [12.00859308, 12.62730517, 1.48669053, 0.66870568]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)
            # logmstar_red_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'R'].max() 
            # logmstar_red_max_arr.append(logmstar_red_max)
            # logmstar_blue_max = mock_pd.logmstar.loc[mock_pd.colour_label == 'B'].max() 
            # logmstar_blue_max_arr.append(logmstar_blue_max)
            logmstar_arr = mock_pd.logmstar.values
            mhi_arr = mock_pd.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            # print("Max of baryonic mass in {0}_{1}:{2}".format(box, num, max(logmbary_arr)))

            # max_total, phi_total, err_total, bins_total, counts_total = \
            #     diff_bmf(logmbary_arr, volume, False)
            # phi_total_arr.append(phi_total)
            if mf_type == 'smf':
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'R'],
            #     volume, False, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(mock_pd.logmstar.loc[mock_pd.colour_label.values == 'B'],
            #     volume, False, 'B')
            phi_total_arr.append(phi_total)
            # phi_arr_red.append(phi_red)
            # phi_arr_blue.append(phi_blue)


            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1.4,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1,3,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1.4,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(-1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
    # rc('text', usetex=True)
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # #* Reduced feature space
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(T.corr(), cmap=cmap)
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Reconstructed post-SVD (sub corr_mat_colour for original matrix)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(corr_mat_colour, cmap=cmap, vmin=-1, vmax=1)
    # # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]

    # plt.xticks(tick_marks, names, rotation='vertical')
    # plt.yticks(tick_marks, names)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Scree plot

    # percentage_variance = []
    # for val in s:
    #     sum_of_eigenvalues = np.sum(s)
    #     percentage_variance.append((val/sum_of_eigenvalues)*100)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.bar(np.arange(1, len(s)+1, 1), percentage_variance, color='#663399',
    #     zorder=5)
    # ax1.scatter(np.arange(1, len(s)+1, 1), s, c='orange', s=50, zorder=10)
    # ax1.plot(np.arange(1, len(s)+1, 1), s, 'k--', zorder=10)
    # ax1.hlines(max_eigen, 0, 20, colors='orange', zorder=10, lw=2)
    # ax1.set_xlabel('Component number')
    # ax1.set_ylabel('Singular values')
    # ax2.set_ylabel('Percentage of variance')
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.set_frame_on(False)
    # ax1.set_xticks(np.arange(1, len(s)+1, 1))
    # plt.show()

    ############################################################################
    #* Observable plots for paper
    #* Total SMFs from mocks and data for paper

    # data = combined_df.values[:,:4]

    # upper_bound = np.nanmean(data, axis=0) + \
    #     np.nanstd(data, axis=0)
    # lower_bound = np.nanmean(data, axis=0) - \
    #     np.nanstd(data, axis=0)

    # phi_max = []
    # phi_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data.T[idx]
    #         >=lower_bound[idx], data.T[idx]<=upper_bound[idx]))
    #     nums = data.T[idx][idxs]
    #     phi_min.append(min(nums))
    #     phi_max.append(max(nums))

    # fig1 = plt.figure()
    # mt = plt.fill_between(x=max_total, y1=phi_max, 
    #     y2=phi_min, color='silver', alpha=0.4)
    # dt = plt.scatter(total_data[0], total_data[1],
    #     color='k', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)

    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_smf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_bmf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()

    # #* Blue fraction from mocks and data for paper

    # data_cen = combined_df.values[:,4:8]
    # data_sat = combined_df.values[:,8:12]

    # upper_bound = np.nanmean(data_cen, axis=0) + \
    #     np.nanstd(data_cen, axis=0)
    # lower_bound = np.nanmean(data_cen, axis=0) - \
    #     np.nanstd(data_cen, axis=0)

    # cen_max = []
    # cen_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_cen.T[idx]
    #         >=lower_bound[idx], data_cen.T[idx]<=upper_bound[idx]))
    #     nums = data_cen.T[idx][idxs]
    #     cen_min.append(min(nums))
    #     cen_max.append(max(nums))

    # upper_bound = np.nanmean(data_sat, axis=0) + \
    #     np.nanstd(data_sat, axis=0)
    # lower_bound = np.nanmean(data_sat, axis=0) - \
    #     np.nanstd(data_sat, axis=0)

    # sat_max = []
    # sat_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_sat.T[idx]
    #         >=lower_bound[idx], data_sat.T[idx]<=upper_bound[idx]))
    #     nums = data_sat.T[idx][idxs]
    #     sat_min.append(min(nums))
    #     sat_max.append(max(nums))

    # fig2 = plt.figure()

    # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max, 
    #     y2=cen_min, color='rebeccapurple', alpha=0.4)
    # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max, 
    #     y2=sat_min, color='goldenrod', alpha=0.4)

    # dt_cen = plt.scatter(f_blue_data[0], f_blue_data[2],
    #     color='rebeccapurple', s=150, zorder=10, marker='^')
    # dt_sat = plt.scatter(f_blue_data[0], f_blue_data[3],
    #     color='goldenrod', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
    # plt.ylim(0,1)
    # # plt.title(r'Blue fractions from mocks and data')
    # plt.legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    #     ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()


    # #* Velocity dispersion from mocks and data for paper

    # bins_red=np.linspace(-2,3,5)
    # bins_blue=np.linspace(-1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # data_red = combined_df.values[:,12:16]
    # data_blue = combined_df.values[:,16:20]

    # upper_bound = np.nanmean(data_red, axis=0) + \
    #     np.nanstd(data_red, axis=0)
    # lower_bound = np.nanmean(data_red, axis=0) - \
    #     np.nanstd(data_red, axis=0)

    # red_max = []
    # red_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_red.T[idx]
    #         >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
    #     nums = data_red.T[idx][idxs]
    #     red_min.append(min(nums))
    #     red_max.append(max(nums))

    # upper_bound = np.nanmean(data_blue, axis=0) + \
    #     np.nanstd(data_blue, axis=0)
    # lower_bound = np.nanmean(data_blue, axis=0) - \
    #     np.nanstd(data_blue, axis=0)

    # blue_max = []
    # blue_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_blue.T[idx]
    #         >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
    #     nums = data_blue.T[idx][idxs]
    #     blue_min.append(min(nums))
    #     blue_max.append(max(nums))

    # fig3 = plt.figure()
    # mt_red = plt.fill_between(x=bins_red, y1=red_min, 
    #     y2=red_max, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=blue_min, 
    #     y2=blue_max, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
    #     color='indianred', s=150, zorder=10, marker='^')
    # dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
    #     color='cornflowerblue', s=150, zorder=10, marker='^')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)

    # plt.show()
    ############################################################################
    # ## Stacked sigma from mocks and data for paper
    # if mf_type == 'smf':
    #     bin_min = 8.6
    # elif mf_type == 'bmf':
    #     bin_min = 9.1
    # bins_red=np.linspace(bin_min,11.2,5)
    # bins_blue=np.linspace(bin_min,11.2,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # mean_mstar_red_max = np.nanmax(combined_df.values[:,12:16], axis=0)
    # mean_mstar_red_min = np.nanmin(combined_df.values[:,12:16], axis=0)
    # mean_mstar_blue_max = np.nanmax(combined_df.values[:,16:20], axis=0)
    # mean_mstar_blue_min = np.nanmin(combined_df.values[:,16:20], axis=0)

    # error = np.nanstd(combined_df.values[:,12:], axis=0)

    # fig2 = plt.figure()

    # mt_red = plt.fill_between(x=bins_red, y1=mean_mstar_red_max, 
    #     y2=mean_mstar_red_min, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=mean_mstar_blue_max, 
    #     y2=mean_mstar_blue_min, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.errorbar(bins_red, sigma_red_data, yerr=error[:4],
    #     color='indianred', fmt='s', ecolor='indianred', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')
    # dt_blue = plt.errorbar(bins_blue, sigma_blue_data, yerr=error[4:8],
    #     color='cornflowerblue', fmt='s', ecolor='cornflowerblue', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')


    # plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)       
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # # plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    # #     bbox_inches="tight", dpi=1200)

    # plt.show()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, \
        gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_new"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['cen_cz'] >= cz_inner_mod) &
        (gal_group_df['cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1.4,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar'
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1.4,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour to data using u-r colours')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if stacked_stat:
        if mf_type == 'smf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])
        elif mf_type == 'bmf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mbary_sigma, blue_deltav, \
                blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])

    else:
        if mf_type == 'smf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1.4,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))
        elif mf_type == 'bmf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mbary_sigma, blue_sigma, \
                blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(-1.4,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))

    if stacked_stat:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    else:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()
    
    return data_arr, z_median

global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global new_chain
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = False
pca = False
new_chain = True

survey = 'eco'
machine = 'bender'
mf_type = 'smf'
quenching = 'hybrid'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3_nops.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_data_buffer_volh1_dr2.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)


bf_dr3_params = [12.32556134, 10.56730341,  0.41905932,  0.60697411,  0.27271022,
       10.14532316, 14.38424043,  0.67580719,  0.02607859]
bf_dr3_chi2 = 15.389214829218805

model_stats = lnprob(bf_dr3_params)
model_stats = model_stats.flatten()

sigma, mat = get_err_data(survey, path_to_mocks)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

# Plot of sigma observable
fig1= plt.figure(figsize=(10,10))

bins_red = np.linspace(-1.4,3,5)
bins_blue = np.linspace(-1,3,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# Plot of fblue observable
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

#####################################################################
#* M*-sigma plot from chain 74 (best-fit vs data only)

import multiprocessing
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def assign_colour_label_data_legacy(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'
        # censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_velocity_dispersion_total(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    cen_subset_ids = np.unique(catl[id_col].loc[(catl[galtype_col] == 1)].values) 
    ## 4416 groups

    cen_subset_df = catl.loc[catl[id_col].isin(cen_subset_ids)]
    #* Excluding N=1 groups (655 groups / 4416 groups remain)
    cen_subset_ids = cen_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    cen_subset_df = catl.loc[catl[id_col].isin(
        cen_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_subset_df_final = cen_subset_df.loc[cen_subset_df[galtype_col] == 1]
    #* np.sum doesn't actually add anything since there is only one central 
    #* per group (checked)
    #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
    #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
    cen_stellar_mass_arr = cen_subset_df_final.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    sigma_arr = cen_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values


    return sigma_arr, cen_stellar_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

def get_err_data(survey, path, bin_i=None):
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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                # mock_pd.ps_grp_censat.loc[mock_pd.index == final_sat_idx] = 0
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 67
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
                0.00722556715]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 62
            ## i.e. halo quenching model
            bf_from_last_chain = [11.749165, 12.471502, 1.496924, 0.416936]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            # Copied from chain 67
            min_chi2_params = [10.194299, 14.545483, 0.708014, 0.007226] #chi2=35
            max_chi2_params = [10.582321, 14.119958, 0.745622, 0.233953] #chi2=1839
            
            # From chain 61
            min_chi2_params = [10.143908, 13.630966, 0.825897, 0.042685] #chi2=13.66

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

    # Drop multiple columns simultaneously
    if type(bin_i) != "NoneType" and type(bin_i) == np.ndarray:
        num_to_remove = len(bin_i)
        num_cols_by_idx = combined_df.shape[1] - 1
        combined_df = combined_df.drop(combined_df.columns[\
            np.array(num_to_remove*[num_cols_by_idx]) - bin_i], 
            axis=1)     
    # Drop only one column
    elif type(bin_i) != "NoneType" and type(bin_i) == int:
        num_cols_by_idx = combined_df.shape[1] - 1
        combined_df = combined_df.drop(combined_df.columns[num_cols_by_idx - bin_i], 
            axis=1)

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r"\usepackage{amsmath}")
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # # #* Reduced feature space
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(T.corr(), cmap=cmap)
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Reconstructed post-SVD (sub corr_mat_colour for original matrix)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(corr_mat_colour, cmap=cmap, vmin=-1, vmax=1)
    # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]
    
    # tick_marks=[0, 4, 8, 12, 16]
    # names = [
    #     r"$\boldsymbol\phi$",
    #     r"$\boldsymbol{f_{blue}^{c}}$",
    #     r"$\boldsymbol{f_{blue}^{s}}$",
    #     r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
    #     r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

    # plt.xticks(tick_marks, names, fontsize=10)#, rotation='vertical')
    # plt.yticks(tick_marks, names, fontsize=10)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()
    # plt.savefig('/Users/asadm2/Desktop/matrix_eco_smf.pdf')


    # #* Scree plot

    # percentage_variance = []
    # for val in s:
    #     sum_of_eigenvalues = np.sum(s)
    #     percentage_variance.append((val/sum_of_eigenvalues)*100)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.bar(np.arange(1, len(s)+1, 1), percentage_variance, color='#663399',
    #     zorder=5)
    # ax1.scatter(np.arange(1, len(s)+1, 1), s, c='orange', s=50, zorder=10)
    # ax1.plot(np.arange(1, len(s)+1, 1), s, 'k--', zorder=10)
    # ax1.hlines(max_eigen, 0, 20, colors='orange', zorder=10, lw=2)
    # ax1.set_xlabel('Component number')
    # ax1.set_ylabel('Singular values')
    # ax2.set_ylabel('Percentage of variance')
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.set_frame_on(False)
    # ax1.set_xticks(np.arange(1, len(s)+1, 1))
    # plt.show()

    ############################################################################
    # #* Observable plots for paper
    # #* Total SMFs from mocks and data for paper

    # data = combined_df.values[:,:4]

    # upper_bound = np.nanmean(data, axis=0) + \
    #     np.nanstd(data, axis=0)
    # lower_bound = np.nanmean(data, axis=0) - \
    #     np.nanstd(data, axis=0)

    # phi_max = []
    # phi_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data.T[idx]
    #         >=lower_bound[idx], data.T[idx]<=upper_bound[idx]))
    #     nums = data.T[idx][idxs]
    #     phi_min.append(min(nums))
    #     phi_max.append(max(nums))

    # phi_max = np.array(phi_max)
    # phi_min = np.array(phi_min)

    # midpoint_phi_err = -((phi_max * phi_min) ** 0.5)
    # phi_data_midpoint_diff = 10**total_data[1] - 10**midpoint_phi_err 

    # fig1 = plt.figure()

    # mt = plt.fill_between(x=max_total, y1=np.log10(10**phi_min + phi_data_midpoint_diff), 
    #     y2=np.log10(10**phi_max + phi_data_midpoint_diff), color='silver', alpha=0.4)

    # # mt = plt.fill_between(x=max_total, y1=phi_max, 
    # #     y2=phi_min, color='silver', alpha=0.4)
    # dt = plt.scatter(total_data[0], total_data[1],
    #     color='k', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)

    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_smf_total_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_bmf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()

    # # #* Blue fraction from mocks and data for paper

    # data_cen = combined_df.values[:,4:8]
    # data_sat = combined_df.values[:,8:12]

    # upper_bound = np.nanmean(data_cen, axis=0) + \
    #     np.nanstd(data_cen, axis=0)
    # lower_bound = np.nanmean(data_cen, axis=0) - \
    #     np.nanstd(data_cen, axis=0)

    # cen_max = []
    # cen_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_cen.T[idx]
    #         >=lower_bound[idx], data_cen.T[idx]<=upper_bound[idx]))
    #     nums = data_cen.T[idx][idxs]
    #     cen_min.append(min(nums))
    #     cen_max.append(max(nums))

    # upper_bound = np.nanmean(data_sat, axis=0) + \
    #     np.nanstd(data_sat, axis=0)
    # lower_bound = np.nanmean(data_sat, axis=0) - \
    #     np.nanstd(data_sat, axis=0)

    # sat_max = []
    # sat_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_sat.T[idx]
    #         >=lower_bound[idx], data_sat.T[idx]<=upper_bound[idx]))
    #     nums = data_sat.T[idx][idxs]
    #     sat_min.append(min(nums))
    #     sat_max.append(max(nums))

    # cen_max = np.array(cen_max)
    # cen_min = np.array(cen_min)
    # sat_max = np.array(sat_max)
    # sat_min = np.array(sat_min)

    # midpoint_cen_err = (cen_max * cen_min) ** 0.5
    # midpoint_sat_err = (sat_max * sat_min) ** 0.5
    # cen_data_midpoint_diff = f_blue_data[2] - midpoint_cen_err 
    # sat_data_midpoint_diff = f_blue_data[3] - midpoint_sat_err

    # fig2 = plt.figure()

    # #* Points not truly in the middle like in next figure. They're just shifted.
    # # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + (f_blue_data[2] - cen_max), 
    # #     y2=cen_min + (f_blue_data[2] - cen_min) - (np.array(cen_max) - np.array(cen_min)), color='rebeccapurple', alpha=0.4)
    # # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + (f_blue_data[3] - sat_max), 
    # #     y2=sat_min + (f_blue_data[3] - sat_min) - (np.array(sat_max) - np.array(sat_min)), color='goldenrod', alpha=0.4)

    # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + cen_data_midpoint_diff, 
    #     y2=cen_min + cen_data_midpoint_diff, color='rebeccapurple', alpha=0.4)
    # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + sat_data_midpoint_diff, 
    #     y2=sat_min + sat_data_midpoint_diff, color='goldenrod', alpha=0.4)

    # dt_cen = plt.scatter(f_blue_data[0], f_blue_data[2],
    #     color='rebeccapurple', s=150, zorder=10, marker='^')
    # dt_sat = plt.scatter(f_blue_data[0], f_blue_data[3],
    #     color='goldenrod', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
    # plt.ylim(0,1)
    # # plt.title(r'Blue fractions from mocks and data')
    # plt.legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    #     ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()


    # #* Velocity dispersion from mocks and data for paper

    # bins_red=np.linspace(1,3,5)
    # bins_blue=np.linspace(1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # data_red = combined_df.values[:,12:16]
    # data_blue = combined_df.values[:,16:20]

    # upper_bound = np.nanmean(data_red, axis=0) + \
    #     np.nanstd(data_red, axis=0)
    # lower_bound = np.nanmean(data_red, axis=0) - \
    #     np.nanstd(data_red, axis=0)

    # red_max = []
    # red_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_red.T[idx]
    #         >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
    #     nums = data_red.T[idx][idxs]
    #     red_min.append(min(nums))
    #     red_max.append(max(nums))

    # upper_bound = np.nanmean(data_blue, axis=0) + \
    #     np.nanstd(data_blue, axis=0)
    # lower_bound = np.nanmean(data_blue, axis=0) - \
    #     np.nanstd(data_blue, axis=0)

    # blue_max = []
    # blue_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_blue.T[idx]
    #         >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
    #     nums = data_blue.T[idx][idxs]
    #     blue_min.append(min(nums))
    #     blue_max.append(max(nums))

    # red_max = np.array(red_max)
    # red_min = np.array(red_min)
    # blue_max = np.array(blue_max)
    # blue_min = np.array(blue_min)

    # midpoint_red_err = (red_max * red_min) ** 0.5
    # midpoint_blue_err = (blue_max * blue_min) ** 0.5
    # red_data_midpoint_diff = 10**mean_mstar_red_data[0] - 10**midpoint_red_err 
    # blue_data_midpoint_diff = 10**mean_mstar_blue_data[0] - 10**midpoint_blue_err

    # fig3 = plt.figure()
    # mt_red = plt.fill_between(x=bins_red, y1=np.log10(np.abs(10**red_min + red_data_midpoint_diff)), 
    #     y2=np.log10(10**red_max + red_data_midpoint_diff), color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=np.log10(10**blue_min + blue_data_midpoint_diff), 
    #     y2=np.log10(10**blue_max + blue_data_midpoint_diff), color='cornflowerblue', alpha=0.4)

    # dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
    #     color='indianred', s=150, zorder=10, marker='^')
    # dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
    #     color='cornflowerblue', s=150, zorder=10, marker='^')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
    #     prop={'size':20})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)

    # plt.show()
    ############################################################################
    # ## Stacked sigma from mocks and data for paper
    # if mf_type == 'smf':
    #     bin_min = 8.6
    # elif mf_type == 'bmf':
    #     bin_min = 9.1
    # bins_red=np.linspace(bin_min,11.2,5)
    # bins_blue=np.linspace(bin_min,11.2,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # mean_mstar_red_max = np.nanmax(combined_df.values[:,12:16], axis=0)
    # mean_mstar_red_min = np.nanmin(combined_df.values[:,12:16], axis=0)
    # mean_mstar_blue_max = np.nanmax(combined_df.values[:,16:20], axis=0)
    # mean_mstar_blue_min = np.nanmin(combined_df.values[:,16:20], axis=0)

    # error = np.nanstd(combined_df.values[:,12:], axis=0)

    # fig2 = plt.figure()

    # mt_red = plt.fill_between(x=bins_red, y1=mean_mstar_red_max, 
    #     y2=mean_mstar_red_min, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=mean_mstar_blue_max, 
    #     y2=mean_mstar_blue_min, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.errorbar(bins_red, sigma_red_data, yerr=error[:4],
    #     color='indianred', fmt='s', ecolor='indianred', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')
    # dt_blue = plt.errorbar(bins_blue, sigma_blue_data, yerr=error[4:8],
    #     color='cornflowerblue', fmt='s', ecolor='cornflowerblue', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')

    # plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)       
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # # plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    # #     bbox_inches="tight", dpi=1200)

    # plt.show()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour to data using u-r colours')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if stacked_stat:
        if mf_type == 'smf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])
        elif mf_type == 'bmf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mbary_sigma, blue_deltav, \
                blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])

    else:
        if mf_type == 'smf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
        elif mf_type == 'bmf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mbary_sigma, blue_sigma, \
                blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.9,5))

    if stacked_stat:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    else:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    return data_arr, z_median

global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global new_chain
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = False
pca = False
new_chain = True

survey = 'eco'
machine = 'bender'
mf_type = 'smf'
quenching = 'hybrid'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_data_buffer_volh1_dr2.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'


data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

bf_bin_params = [1.25265896e+01, 1.05945720e+01, 4.53178499e-01, 5.82146045e-01,
       3.19537853e-01, 1.01463004e+01, 1.43383461e+01, 6.83946944e-01,
       1.41126799e-02]
bf_bin_chi2 = 30.84
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, mat = get_err_data(survey, path_to_mocks)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

data_stats = [-1.52518353, -1.67537112, -1.89183513, -2.60866256,  0.82258065,
        0.58109091,  0.32606325,  0.09183673,  0.46529814,  0.31311707,
        0.21776504,  0.04255319, 10.33120738, 10.44089237, 10.49140916,
       10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856]

# Using bf above
model_stats = [-1.65516888, -1.83463035, -2.06269189, -2.76348575,  0.8043718 ,
        0.6102489 ,  0.32755795,  0.05555556,  0.47000706,  0.35566382,
        0.17972973,  0.0483871 , 10.30747509, 10.39169121, 10.52210903,
       10.79344082,  9.91377831, 10.00564194, 10.01888371, 10.16165733]

sigma = [0.12114725, 0.14457261, 0.16116129, 0.18796077, 0.02704955,
       0.03569706, 0.02380825, 0.02363771, 0.01933245, 0.01470065,
       0.01735792, 0.02629646, 0.05751571, 0.04541055, 0.02493778,
       0.06555989, 0.07026583, 0.04466467, 0.04253629, 0.06514535]

# Plot of sigma observable (with model binned both ways to compare with new data
# binning)
fig1= plt.figure(figsize=(10,10))

bins_red = np.linspace(1,2.8,5)
bins_blue = np.linspace(1,2.5,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

## Calculating chi-squared without matrix to compare to chi-squared 
## with matrix (they are similar)
data_stats = np.array(data_stats)
model_stats = np.array(model_stats)
sigma = np.array(sigma)
chi_i_arr = (data_stats - model_stats)**2 / (sigma**2)
chi_squared_val = np.sum(chi_i_arr)

## Plot of other 2 observables (smf & fblue)
fig2= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

fig3= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

dc = plt.errorbar(x, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(x, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(x, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(x, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([dc, ds, bfc, bfs], [ 'Data cen', 'Data sat', 'Best-fit cen', 'Best-fit sat'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

#* Remove visually bad bins (fblue_sat #2 & #3 & sigma_red #4)
bins_to_remove = [9, 10, 15]
# Subtraction is required from 19 since np.delete is removing idxs backwards 
# i.e it's not i that's removed but 19-i so if you want to remove bin 4, i can't
# be 4. Instead, i has to be 15.
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])
idxs_to_remove = np.insert(idxs_to_remove, 0, 0)
chi2_arr = []
for i in idxs_to_remove:
    if i == 0:
        chi2 = chi_squared(data_stats, model_stats, sigma, mat)
        chi2_arr.append(chi2)
    else:
        print(i)
        sigma_mod, mat_mod = get_err_data(survey, path_to_mocks, i)

        num_elems_by_idx = len(data_stats) - 1
        data_stats_mod  = np.delete(data_stats, num_elems_by_idx-i) 
        model_stats_mod  = np.delete(model_stats, num_elems_by_idx-i) 

        chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
        chi2_arr.append(chi2)

## chi2_arr: 
#[68.07603259157973, 30.079954101188385, 54.75680297796448, 42.17193344327803]

#* Remove all red sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = [12, 13, 14, 15]
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, mat_mod = get_err_data(survey, path_to_mocks, idxs_to_remove)

num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 
model_stats_mod  = np.delete(model_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)

## chi2_arr: 
#[68.07603259157973, 40.730628369467894]

#* Remove all blue sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = [16, 17, 18, 19]
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, mat_mod = get_err_data(survey, path_to_mocks, idxs_to_remove)

num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 
model_stats_mod  = np.delete(model_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)


#* Remove all red+blue sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = [12, 13, 14, 15, 16, 17, 18, 19]
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, mat_mod = get_err_data(survey, path_to_mocks, idxs_to_remove)

num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 
model_stats_mod  = np.delete(model_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)

#* Remove all fblue sat bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = [8, 9, 10, 11]
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, mat_mod = get_err_data(survey, path_to_mocks, idxs_to_remove)

num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 
model_stats_mod  = np.delete(model_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)

#* Replace red+blue sigma-mstar with total sigma-mstar
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)

def calc_data_stats_mod(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour to data using u-r colours')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    sigma_total_data, cen_mstar_sigma_total_data = \
        get_velocity_dispersion_total(catl, 'data')

    sigma_total_data = np.log10(sigma_total_data)

    mean_mstar_total_data = bs(sigma_total_data, cen_mstar_sigma_total_data, 
        statistic=average_of_log, bins=np.linspace(1,2.8,5))

    phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_total_data = \
        total_data[1], f_blue_data[2], f_blue_data[3], \
        mean_mstar_total_data[0]

    data_stats = []
    data_stats.append(phi_total_data)
    data_stats.append(f_blue_cen_data)
    data_stats.append(f_blue_sat_data)
    data_stats.append(vdisp_total_data)
    data_stats = np.array(data_stats)
    data_stats = data_stats.flatten()

    return data_stats, z_median

def lnprob_mod(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    total_model = measure_all_smf(gal_group_df, survey_vol, False) 

    ## Observable #2 - Blue fraction
    f_blue = blue_frac(gal_group_df, True, False)
    
    sigma_total, cen_mstar_sigma_total = \
        get_velocity_dispersion_total(gal_group_df, 'model')

    sigma_total = np.log10(sigma_total)

    mean_mstar_total = bs(sigma_total, cen_mstar_sigma_total, 
        statistic=average_of_log, bins=np.linspace(1,2.8,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    model_arr.append(mean_mstar_total[0])

    model_arr = np.array(model_arr)

    return model_arr

def get_err_data_mod(survey, path, bin_i=None):
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

    phi_total_arr = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_total_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                # mock_pd.ps_grp_censat.loc[mock_pd.index == final_sat_idx] = 0
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            ## Using best-fit found for new ECO data using result from chain 67
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
                0.00722556715]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 62
            ## i.e. halo quenching model
            bf_from_last_chain = [11.749165, 12.471502, 1.496924, 0.416936]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            # Copied from chain 67
            min_chi2_params = [10.194299, 14.545483, 0.708014, 0.007226] #chi2=35
            max_chi2_params = [10.582321, 14.119958, 0.745622, 0.233953] #chi2=1839
            
            # From chain 61
            min_chi2_params = [10.143908, 13.630966, 0.825897, 0.042685] #chi2=13.66

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])

            #Total sigma-mstar
            sigma_total, cen_mstar_sigma_total = \
                get_velocity_dispersion_total(mock_pd, 'mock')

            sigma_total = np.log10(sigma_total)

            mean_mstar_total = bs(sigma_total, cen_mstar_sigma_total, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))

            mean_mstar_total_arr.append(mean_mstar_total[0])

    phi_arr_total = np.array(phi_total_arr)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_total_arr = np.array(mean_mstar_total_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_total_cen_0 = mean_mstar_total_arr[:,0]
    mstar_total_cen_1 = mean_mstar_total_arr[:,1]
    mstar_total_cen_2 = mean_mstar_total_arr[:,2]
    mstar_total_cen_3 = mean_mstar_total_arr[:,3]


    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_total_cen_0':mstar_total_cen_0, 'mstar_total_cen_1':mstar_total_cen_1, 
        'mstar_total_cen_2':mstar_total_cen_2, 'mstar_total_cen_3':mstar_total_cen_3})

    # Drop multiple columns simultaneously
    if type(bin_i) != "NoneType" and type(bin_i) == np.ndarray:
        num_to_remove = len(bin_i)
        num_cols_by_idx = combined_df.shape[1] - 1
        combined_df = combined_df.drop(combined_df.columns[\
            np.array(num_to_remove*[num_cols_by_idx]) - bin_i], 
            axis=1)     
    # Drop only one column
    elif type(bin_i) != "NoneType" and type(bin_i) == int:
        num_cols_by_idx = combined_df.shape[1] - 1
        combined_df = combined_df.drop(combined_df.columns[num_cols_by_idx - bin_i], 
            axis=1)

    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    # from matplotlib.legend_handler import HandlerTuple
    # import matplotlib.pyplot as plt
    # from matplotlib import rc
    # from matplotlib import cm

    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r"\usepackage{amsmath}")
    # rc('axes', linewidth=2)
    # rc('xtick.major', width=4, size=7)
    # rc('ytick.major', width=4, size=7)
    # rc('xtick.minor', width=2, size=7)
    # rc('ytick.minor', width=2, size=7)

    # # #* Reduced feature space
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(T.corr(), cmap=cmap)
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()

    # #* Reconstructed post-SVD (sub corr_mat_colour for original matrix)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # cmap = cm.get_cmap('Spectral_r')
    # cax = ax1.matshow(corr_mat_colour, cmap=cmap, vmin=-1, vmax=1)
    # cax = ax1.matshow(B, cmap=cmap, vmin=-1, vmax=1)
    # tick_marks = [i for i in range(len(combined_df.columns))]
    # names = [
    # r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$',
    # r'$fblue\ cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$',
    # r'$fblue\ sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$',
    # r'$mstar\ red\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',
    # r'$mstar\ blue\ grpcen_1$', r'$grpcen_2$', r'$grpcen_3$', r'$grpcen_4$',]
    
    # tick_marks=[0, 4, 8, 12, 16]
    # names = [
    #     r"$\boldsymbol\phi$",
    #     r"$\boldsymbol{f_{blue}^{c}}$",
    #     r"$\boldsymbol{f_{blue}^{s}}$",
    #     r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
    #     r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

    # plt.xticks(tick_marks, names, fontsize=10)#, rotation='vertical')
    # plt.yticks(tick_marks, names, fontsize=10)    
    # plt.gca().invert_yaxis() 
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(cax)
    # plt.title('{0}'.format(quenching))
    # plt.show()
    # plt.savefig('/Users/asadm2/Desktop/matrix_eco_smf.pdf')


    # #* Scree plot

    # percentage_variance = []
    # for val in s:
    #     sum_of_eigenvalues = np.sum(s)
    #     percentage_variance.append((val/sum_of_eigenvalues)*100)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.bar(np.arange(1, len(s)+1, 1), percentage_variance, color='#663399',
    #     zorder=5)
    # ax1.scatter(np.arange(1, len(s)+1, 1), s, c='orange', s=50, zorder=10)
    # ax1.plot(np.arange(1, len(s)+1, 1), s, 'k--', zorder=10)
    # ax1.hlines(max_eigen, 0, 20, colors='orange', zorder=10, lw=2)
    # ax1.set_xlabel('Component number')
    # ax1.set_ylabel('Singular values')
    # ax2.set_ylabel('Percentage of variance')
    # ax1.set_zorder(ax2.get_zorder()+1)
    # ax1.set_frame_on(False)
    # ax1.set_xticks(np.arange(1, len(s)+1, 1))
    # plt.show()

    ############################################################################
    # #* Observable plots for paper
    # #* Total SMFs from mocks and data for paper

    # data = combined_df.values[:,:4]

    # upper_bound = np.nanmean(data, axis=0) + \
    #     np.nanstd(data, axis=0)
    # lower_bound = np.nanmean(data, axis=0) - \
    #     np.nanstd(data, axis=0)

    # phi_max = []
    # phi_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data.T[idx]
    #         >=lower_bound[idx], data.T[idx]<=upper_bound[idx]))
    #     nums = data.T[idx][idxs]
    #     phi_min.append(min(nums))
    #     phi_max.append(max(nums))

    # phi_max = np.array(phi_max)
    # phi_min = np.array(phi_min)

    # midpoint_phi_err = -((phi_max * phi_min) ** 0.5)
    # phi_data_midpoint_diff = 10**total_data[1] - 10**midpoint_phi_err 

    # fig1 = plt.figure()

    # mt = plt.fill_between(x=max_total, y1=np.log10(10**phi_min + phi_data_midpoint_diff), 
    #     y2=np.log10(10**phi_max + phi_data_midpoint_diff), color='silver', alpha=0.4)

    # # mt = plt.fill_between(x=max_total, y1=phi_max, 
    # #     y2=phi_min, color='silver', alpha=0.4)
    # dt = plt.scatter(total_data[0], total_data[1],
    #     color='k', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)

    # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

    # plt.legend([(dt), (mt)], ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
    # plt.minorticks_on()
    # # plt.title(r'SMFs from mocks')
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_smf_total_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_bmf_total.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()

    # # #* Blue fraction from mocks and data for paper

    # data_cen = combined_df.values[:,4:8]
    # data_sat = combined_df.values[:,8:12]

    # upper_bound = np.nanmean(data_cen, axis=0) + \
    #     np.nanstd(data_cen, axis=0)
    # lower_bound = np.nanmean(data_cen, axis=0) - \
    #     np.nanstd(data_cen, axis=0)

    # cen_max = []
    # cen_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_cen.T[idx]
    #         >=lower_bound[idx], data_cen.T[idx]<=upper_bound[idx]))
    #     nums = data_cen.T[idx][idxs]
    #     cen_min.append(min(nums))
    #     cen_max.append(max(nums))

    # upper_bound = np.nanmean(data_sat, axis=0) + \
    #     np.nanstd(data_sat, axis=0)
    # lower_bound = np.nanmean(data_sat, axis=0) - \
    #     np.nanstd(data_sat, axis=0)

    # sat_max = []
    # sat_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_sat.T[idx]
    #         >=lower_bound[idx], data_sat.T[idx]<=upper_bound[idx]))
    #     nums = data_sat.T[idx][idxs]
    #     sat_min.append(min(nums))
    #     sat_max.append(max(nums))

    # cen_max = np.array(cen_max)
    # cen_min = np.array(cen_min)
    # sat_max = np.array(sat_max)
    # sat_min = np.array(sat_min)

    # midpoint_cen_err = (cen_max * cen_min) ** 0.5
    # midpoint_sat_err = (sat_max * sat_min) ** 0.5
    # cen_data_midpoint_diff = f_blue_data[2] - midpoint_cen_err 
    # sat_data_midpoint_diff = f_blue_data[3] - midpoint_sat_err

    # fig2 = plt.figure()

    # #* Points not truly in the middle like in next figure. They're just shifted.
    # # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + (f_blue_data[2] - cen_max), 
    # #     y2=cen_min + (f_blue_data[2] - cen_min) - (np.array(cen_max) - np.array(cen_min)), color='rebeccapurple', alpha=0.4)
    # # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + (f_blue_data[3] - sat_max), 
    # #     y2=sat_min + (f_blue_data[3] - sat_min) - (np.array(sat_max) - np.array(sat_min)), color='goldenrod', alpha=0.4)

    # mt_cen = plt.fill_between(x=f_blue_data[0], y1=cen_max + cen_data_midpoint_diff, 
    #     y2=cen_min + cen_data_midpoint_diff, color='rebeccapurple', alpha=0.4)
    # mt_sat = plt.fill_between(x=f_blue_data[0], y1=sat_max + sat_data_midpoint_diff, 
    #     y2=sat_min + sat_data_midpoint_diff, color='goldenrod', alpha=0.4)

    # dt_cen = plt.scatter(f_blue_data[0], f_blue_data[2],
    #     color='rebeccapurple', s=150, zorder=10, marker='^')
    # dt_sat = plt.scatter(f_blue_data[0], f_blue_data[3],
    #     color='goldenrod', s=150, zorder=10, marker='^')

    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
    # plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
    # plt.ylim(0,1)
    # # plt.title(r'Blue fractions from mocks and data')
    # plt.legend([dt_cen, dt_sat, mt_cen, mt_sat], 
    #     ['ECO cen', 'ECO sat', 'Mocks cen', 'Mocks sat'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_fblue_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # plt.show()


    # #* Velocity dispersion from mocks and data for paper

    # bins_red=np.linspace(1,3,5)
    # bins_blue=np.linspace(1,3,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # data_red = combined_df.values[:,12:16]
    # data_blue = combined_df.values[:,16:20]

    # upper_bound = np.nanmean(data_red, axis=0) + \
    #     np.nanstd(data_red, axis=0)
    # lower_bound = np.nanmean(data_red, axis=0) - \
    #     np.nanstd(data_red, axis=0)

    # red_max = []
    # red_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_red.T[idx]
    #         >=lower_bound[idx], data_red.T[idx]<=upper_bound[idx]))
    #     nums = data_red.T[idx][idxs]
    #     red_min.append(min(nums))
    #     red_max.append(max(nums))

    # upper_bound = np.nanmean(data_blue, axis=0) + \
    #     np.nanstd(data_blue, axis=0)
    # lower_bound = np.nanmean(data_blue, axis=0) - \
    #     np.nanstd(data_blue, axis=0)

    # blue_max = []
    # blue_min = []
    # for idx in range(len(upper_bound)):
    #     idxs = np.where(np.logical_and(data_blue.T[idx]
    #         >=lower_bound[idx], data_blue.T[idx]<=upper_bound[idx]))
    #     nums = data_blue.T[idx][idxs]
    #     blue_min.append(min(nums))
    #     blue_max.append(max(nums))

    # red_max = np.array(red_max)
    # red_min = np.array(red_min)
    # blue_max = np.array(blue_max)
    # blue_min = np.array(blue_min)

    # midpoint_red_err = (red_max * red_min) ** 0.5
    # midpoint_blue_err = (blue_max * blue_min) ** 0.5
    # red_data_midpoint_diff = 10**mean_mstar_red_data[0] - 10**midpoint_red_err 
    # blue_data_midpoint_diff = 10**mean_mstar_blue_data[0] - 10**midpoint_blue_err

    # fig3 = plt.figure()
    # mt_red = plt.fill_between(x=bins_red, y1=np.log10(np.abs(10**red_min + red_data_midpoint_diff)), 
    #     y2=np.log10(10**red_max + red_data_midpoint_diff), color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=np.log10(10**blue_min + blue_data_midpoint_diff), 
    #     y2=np.log10(10**blue_max + blue_data_midpoint_diff), color='cornflowerblue', alpha=0.4)

    # dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
    #     color='indianred', s=150, zorder=10, marker='^')
    # dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
    #     color='cornflowerblue', s=150, zorder=10, marker='^')

    # plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
    #     prop={'size':20})
    # plt.minorticks_on()
    # if mf_type == 'smf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_offsetmocks.pdf', 
    #         bbox_inches="tight", dpi=1200)
    # elif mf_type == 'bmf':
    #     plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp_bary.pdf', 
    #         bbox_inches="tight", dpi=1200)

    # plt.show()
    ############################################################################
    # ## Stacked sigma from mocks and data for paper
    # if mf_type == 'smf':
    #     bin_min = 8.6
    # elif mf_type == 'bmf':
    #     bin_min = 9.1
    # bins_red=np.linspace(bin_min,11.2,5)
    # bins_blue=np.linspace(bin_min,11.2,5)
    # bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    # bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

    # mean_mstar_red_max = np.nanmax(combined_df.values[:,12:16], axis=0)
    # mean_mstar_red_min = np.nanmin(combined_df.values[:,12:16], axis=0)
    # mean_mstar_blue_max = np.nanmax(combined_df.values[:,16:20], axis=0)
    # mean_mstar_blue_min = np.nanmin(combined_df.values[:,16:20], axis=0)

    # error = np.nanstd(combined_df.values[:,12:], axis=0)

    # fig2 = plt.figure()

    # mt_red = plt.fill_between(x=bins_red, y1=mean_mstar_red_max, 
    #     y2=mean_mstar_red_min, color='indianred', alpha=0.4)
    # mt_blue = plt.fill_between(x=bins_blue, y1=mean_mstar_blue_max, 
    #     y2=mean_mstar_blue_min, color='cornflowerblue', alpha=0.4)

    # dt_red = plt.errorbar(bins_red, sigma_red_data, yerr=error[:4],
    #     color='indianred', fmt='s', ecolor='indianred', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')
    # dt_blue = plt.errorbar(bins_blue, sigma_blue_data, yerr=error[4:8],
    #     color='cornflowerblue', fmt='s', ecolor='cornflowerblue', markersize=5, capsize=3,
    #     capthick=1.5, zorder=10, marker='^')

    # plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
    # if mf_type == 'smf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
    # elif mf_type == 'bmf':
    #     plt.xlabel(r'\boldmath$\log_{10}\ M_{b, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)       
    # # plt.title(r'Velocity dispersion from mocks and data')
    # plt.legend([(dt_red, dt_blue), (mt_red, mt_blue)], 
    #     ['ECO','Mocks'],
    #     handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
    # plt.minorticks_on()
    # # plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_vdisp.pdf', 
    # #     bbox_inches="tight", dpi=1200)

    # plt.show()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour

sigma_mod, mat_mod = get_err_data_mod(survey, path_to_mocks)
data_stats_mod, z_median = calc_data_stats_mod(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

bf_bin_params = [1.25265896e+01, 1.05945720e+01, 4.53178499e-01, 5.82146045e-01,
       3.19537853e-01, 1.01463004e+01, 1.43383461e+01, 6.83946944e-01,
       1.41126799e-02]
bf_bin_chi2 = 30.84
model_stats_mod = lnprob_mod(bf_bin_params)
model_stats_mod = model_stats_mod.flatten()

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)

#####################################################################
#* Obs. plots from chain 77 (best-fit vs data only)

import multiprocessing
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def assign_colour_label_data_legacy(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'
        # censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                # mock_pd.ps_grp_censat.loc[mock_pd.index == final_sat_idx] = 0
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            # ## Using best-fit found for new ECO data using result from chain 67
            # ## i.e. hybrid quenching model
            # bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
            #     0.00722556715]

            # ## 75
            # bf_from_last_chain = [10.215486, 13.987752, 0.753758, 0.025111]
            
            ## Using best-fit found for new ECO data using result from chain 59
            ## i.e. hybrid quenching model which was the last time sigma-M* was
            ## used i.e. stacked_stat = True
            bf_from_last_chain = [10.133745, 13.478087, 0.810922,
                0.043523]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 62
            ## i.e. halo quenching model
            bf_from_last_chain = [11.749165, 12.471502, 1.496924, 0.416936]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            # Copied from chain 67
            min_chi2_params = [10.194299, 14.545483, 0.708014, 0.007226] #chi2=35
            max_chi2_params = [10.582321, 14.119958, 0.745622, 0.233953] #chi2=1839
            
            # From chain 61
            min_chi2_params = [10.143908, 13.630966, 0.825897, 0.042685] #chi2=13.66

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, corr_mat_inv_colour

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    print('Measuring SMF for data')
    total_data = measure_all_smf(catl, volume, True)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    print('Measuring stacked velocity dispersion for data')
    red_deltav, red_cen_mstar_sigma, blue_deltav, \
        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

    sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
        statistic='std', bins=np.linspace(8.6,11,5))
    sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
        statistic='std', bins=np.linspace(8.6,11,5))

    sigma_red_data = np.log10(sigma_red_data[0])
    sigma_blue_data = np.log10(sigma_blue_data[0])

    phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
        vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
        sigma_red_data, sigma_blue_data

    data_arr = []
    data_arr.append(phi_total_data)
    data_arr.append(f_blue_cen_data)
    data_arr.append(f_blue_sat_data)
    data_arr.append(vdisp_red_data)
    data_arr.append(vdisp_blue_data)
    data_arr = np.array(data_arr)
    data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

    return data_arr, z_median
    

survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
level = 'group'
machine = 'bender'
stacked_stat = True

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_buffer_volh1_dr3.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'

data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

bf_bin_params = [12.412109, 10.603190, 0.432347, 0.597501, 0.262579, 
    10.133802, 13.973785, 0.687669, 0.025981]
bf_bin_chi2 = 29.71
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, mat = get_err_data(survey, path_to_mocks)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

# SMF
fig1= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# fblue
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# Mstar-sigma
bins_red=np.linspace(8.6,11,5)
bins_blue=np.linspace(8.6,11,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

fig3 = plt.figure()

bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')

plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, 
    loc='lower right', prop={'size':20})
plt.show()

################################################################################
#* Comparing red and blue logmstar group cen between dr2 and dr3
from scipy.stats import binned_statistic as bs
from cosmo_utils.utils import work_paths as cwpaths
import pandas as pd
import numpy as np
import os

def assign_colour_label_data_legacy(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'

        colour_label_arr[idx] = colour_label

    catl['colour_label'] = colour_label_arr

    return catl

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def get_stacked_velocity_dispersion(catl, catl_type, galtype_col, id_col, 
    randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, galtype_col, id_col, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

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


survey = 'eco'
mf_type = 'smf'
## Running group-finding on data including the buffer
cz_inner = 2530
cz_outer = 7470
#* Should have the same units as Warren (h=1.0)
volume = 192351.36 # survey volume with buffer in h=1.0


eco = {
    'c': 3*10**5,
    'survey_vol': volume,
    'min_cz' : cz_inner,
    'max_cz' : cz_outer,
    'zmin': cz_inner/(3*10**5),
    'zmax': cz_outer/(3*10**5),
    'l_perp': 0.07,
    'l_para': 1.1,
    'nmin': 1,
    'verbose': True,
    'catl_type': 'mstar'
}


param_dict = vars()[survey]

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_proc = dict_of_paths['proc_dir']
catl_file = path_to_raw + "eco/ecodr3.csv"
duplicate_file = path_to_raw + "ECO_duplicate_classification.csv"
catl_colours_file = path_to_raw + "eco_dr3_colours.csv"

ecodr3_buff = pd.read_csv(catl_file, delimiter=",", header=0)
eco_duplicates = pd.read_csv(duplicate_file, header=0)
eco_colours = pd.read_csv(catl_colours_file, header=0, index_col=0)
duplicate_names = eco_duplicates.NAME.loc[eco_duplicates.DUP >0]
ecodr3_buff = ecodr3_buff[~ecodr3_buff.name.isin(duplicate_names)].reset_index(drop=True)
ecodr3_buff = pd.merge(ecodr3_buff, eco_colours, left_on='name', right_on='galname').\
    drop(columns=["galname"])

ecodr3_buff = mock_add_grpcz(ecodr3_buff, grpid_col='grp', 
    galtype_col='fc', cen_cz_col='cz')

ecodr3_subset_df = ecodr3_buff.loc[(ecodr3_buff.grpcz >= cz_inner) & \
    (ecodr3_buff.grpcz <= cz_outer) & (ecodr3_buff.logmstar >= 8.9) & \
    (ecodr3_buff.absrmag.values <= -17.33)].reset_index(drop=True)

red_dr3 = ecodr3_subset_df.loc[ecodr3_subset_df.red == 1]
blue_dr3 = ecodr3_subset_df.loc[ecodr3_subset_df.blue == 0]

ecodr3_subset_df = assign_colour_label_data(ecodr3_subset_df)

print('Measuring stacked velocity dispersion for data')
red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(ecodr3_subset_df, 
    'data', 'fc', 'grp')

sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data = np.log10(sigma_red_data[0])
sigma_blue_data = np.log10(sigma_blue_data[0])

print('Measuring velocity dispersion for data')
red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(ecodr3_subset_df, 
    'data', 'fc', 'grp')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.8,5))
mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.5,5))

dr2_file = path_to_raw + "eco/ecodr2.csv"
ecodr2_buff = pd.read_csv(dr2_file,delimiter=",", header=0)
ecodr2_buff = mock_add_grpcz(ecodr2_buff, grpid_col='grp_e17', 
    galtype_col='fc_e17', cen_cz_col='cz')
ecodr2_subset_df = ecodr2_buff.loc[(ecodr2_buff.grpcz_e17 >= cz_inner) & \
    (ecodr2_buff.grpcz_e17 <= cz_outer) & (ecodr2_buff.logmstar >= 8.9) & \
    (ecodr2_buff.absrmag.values <= -17.33)].\
    reset_index(drop=True)
ecodr2_subset_df = ecodr2_subset_df.rename(columns={'radeg':'ra'})
ecodr2_subset_df = ecodr2_subset_df.rename(columns={'dedeg':'dec'})
ecodr2_subset_df = assign_colour_label_data_legacy(ecodr2_subset_df)
red_dr2 = ecodr2_subset_df.loc[ecodr2_subset_df.colour_label == 'R']
blue_dr2 = ecodr2_subset_df.loc[ecodr2_subset_df.colour_label == 'B']

red_grpcen_mstar_dr2 = red_dr2.logmstar.loc[red_dr2.fc_e17 == 1]
blue_grpcen_mstar_dr2 = blue_dr2.logmstar.loc[blue_dr2.fc_e17 == 1]
red_grpcen_mstar_dr3 = red_dr3.logmstar.loc[red_dr3.fc == 1]
blue_grpcen_mstar_dr3 = blue_dr3.logmstar.loc[blue_dr3.fc == 1]


print('Measuring stacked velocity dispersion for data')
red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(ecodr2_subset_df, 
    'data', 'fc_e17', 'grp_e17')

sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data = np.log10(sigma_red_data[0])
sigma_blue_data = np.log10(sigma_blue_data[0])

#* Checking M*-sigma after group-finding and after pair-splitting
catl_file_nops = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3_nops.hdf5"

eco_buff_nops = reading_catls(catl_file_nops)
eco_buff_nops = mock_add_grpcz(eco_buff_nops, grpid_col='groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')

eco_buff_nops = eco_buff_nops.loc[(eco_buff_nops.grpcz_cen.values >= 3000) & 
    (eco_buff_nops.grpcz_cen.values <= 7000) & 
    (eco_buff_nops.logmstar >= 8.9) &
    (eco_buff_nops.absrmag.values <= -17.33)]

eco_buff_nops = assign_colour_label_data(eco_buff_nops)

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(eco_buff_nops, 
    'data', 'g_galtype', 'groupid')

sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data = np.log10(sigma_red_data[0])
sigma_blue_data = np.log10(sigma_blue_data[0])

print('Measuring velocity dispersion for data')
red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(eco_buff_nops, 
    'data', 'g_galtype', 'groupid')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.8,5))
mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.5,5))

#* Results of comparing M*-sigma (x-y)
#* DR3 raw data
sigma_red_data
array([2.94816866, 3.16573379, 3.21781261, 3.17880676])

sigma_blue_data
array([3.18170746, 3.24037577, 3.13837627, 3.04341951])

#* DR3 post-group finding (no pair splitting)
sigma_red_data
array([1.90665511, 1.97997373, 1.96469075, 2.15631488])

sigma_blue_data
array([1.82761246, 1.84888208, 1.93491728, 2.32294531])

#* DR3 post-group finding AND pair splitting
sigma_red_data
array([1.87495504, 1.98773671, 1.96936469, 2.16607416])

sigma_blue_data
array([1.74316835, 1.79402739, 1.95085768, 2.33335803])



#* Results of comparing sigma-M* (x-y)
#* DR3 raw data
mean_mstar_red_data[0]
array([10.33589414, 10.45925564, 10.51963471, 10.90178673])

mean_mstar_blue_data[0]
array([ 9.97957968,  9.91990397, 10.00695241, 10.1475587 ])

#* DR3 post-group finding (no pair splitting)
mean_mstar_red_data[0]
array([10.32296892, 10.41302058, 10.43689304, 10.80875171])

mean_mstar_blue_data[0]
array([ 9.92036448,  9.98886097, 10.02533109, 10.05904768])

#* DR3 post-group finding AND pair splitting
mean_mstar_red_data[0]
array([10.33120738, 10.44089237, 10.49140916, 10.90520085])

mean_mstar_blue_data[0]
array([ 9.96134779, 10.02206739, 10.08378187, 10.19648856])

#####################################################################
#* Obs. plots from chain 78 (best-fit vs data only)

import multiprocessing
import time
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

__author__ = '[Mehnaaz Asad]'

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def assign_colour_label_data_legacy(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    logmstar_arr = catl.logmstar.values
    u_r_arr = catl.modelu_rcorr.values

    colour_label_arr = np.empty(len(catl), dtype='str')
    for idx, value in enumerate(logmstar_arr):

        # Divisions taken from Moffett et al. 2015 equation 1
        if value <= 9.1:
            if u_r_arr[idx] > 1.457:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value > 9.1 and value < 10.1:
            divider = 0.24 * value - 0.7
            if u_r_arr[idx] > divider:
                colour_label = 'R'
            else:
                colour_label = 'B'

        if value >= 10.1:
            if u_r_arr[idx] > 1.7:
                colour_label = 'R'
            else:
                colour_label = 'B'
            
        colour_label_arr[idx] = colour_label
    
    catl['colour_label'] = colour_label_arr

    return catl

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        # censat_col = 'ps_grp_censat'
        censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'g_galtype'
            id_col = 'groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'grp_censat'
            id_col = 'groupid'
            cencz_col = 'cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
        rng_arr[ii] = rng
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    df.loc[:, 'rng'] = rng_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    phi_total_arr = []
    # phi_arr_red = []
    # phi_arr_blue = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            # ## Using best-fit found for old ECO data using optimize_hybridqm_eco,py
            # Mstar_q = 10.39 # Msun/h
            # Mh_q = 14.85 # Msun/h
            # mu = 0.65
            # nu = 0.16

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for hybrid quenching model
            # Mstar_q = 10.49 # Msun/h
            # Mh_q = 14.03 # Msun/h
            # mu = 0.69
            # nu = 0.148

            # ## Using best-fit found for new ECO data using result from chain 67
            # ## i.e. hybrid quenching model
            # bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
            #     0.00722556715]

            # ## 75
            # bf_from_last_chain = [10.215486, 13.987752, 0.753758, 0.025111]
            
            ## Using best-fit found for new ECO data using result from chain 59
            ## i.e. hybrid quenching model which was the last time sigma-M* was
            ## used i.e. stacked_stat = True
            bf_from_last_chain = [10.133745, 13.478087, 0.810922, 0.043523]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            # ## Using best-fit found for new ECO data using optimize_qm_eco.py 
            # ## for halo quenching model
            # Mh_qc = 12.61 # Msun/h
            # Mh_qs = 13.5 # Msun/h
            # mu_c = 0.40
            # mu_s = 0.148

            ## Using best-fit found for new ECO data using result from chain 62
            ## i.e. halo quenching model
            bf_from_last_chain = [11.749165, 12.471502, 1.496924, 0.416936]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            # Copied from chain 67
            min_chi2_params = [10.194299, 14.545483, 0.708014, 0.007226] #chi2=35
            max_chi2_params = [10.582321, 14.119958, 0.745622, 0.233953] #chi2=1839
            
            # From chain 61
            min_chi2_params = [10.143908, 13.630966, 0.825897, 0.042685] #chi2=13.66

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)
    # phi_arr_red = np.array(phi_arr_red)
    # phi_arr_blue = np.array(phi_arr_blue)

    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    return err_colour, corr_mat_inv_colour

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['cen_cz'] >= cz_inner_mod) &
        (gal_group_df['cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def calc_data_stats(catl_file):

    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    print('Measuring velocity dispersion for data')
    red_sigma, red_cen_mstar_sigma, blue_sigma, \
        blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

    red_sigma = np.log10(red_sigma)
    blue_sigma = np.log10(blue_sigma)

    mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.8,5))
    mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,2.5,5))

    phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
        vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
        mean_mstar_red_data[0], mean_mstar_blue_data[0]

    data_arr = []
    data_arr.append(phi_total_data)
    data_arr.append(f_blue_cen_data)
    data_arr.append(f_blue_sat_data)
    data_arr.append(vdisp_red_data)
    data_arr.append(vdisp_blue_data)
    data_arr = np.array(data_arr)
    data_arr = data_arr.flatten()

    return data_arr, z_median


survey = 'eco'
mf_type = 'smf'
quenching = 'hybrid'
level = 'group'
machine = 'bender'
stacked_stat = False

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3_nops.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_buffer_volh1_dr3.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_external + 'RESOLVE_A_mvir_catls/'
elif survey == 'resolveb':
    path_to_mocks = path_to_external + 'RESOLVE_B_mvir_catls/'


data_stats, z_median = calc_data_stats(catl_file)

model_init = halocat_init(halo_catalog, z_median)

bf_bin_params = [12.614564, 10.622212, 0.453382, 0.384225, 0.210445, 10.182908,
    14.063352, 0.728482, 0.023669]
bf_bin_chi2 = 18.78
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, mat = get_err_data(survey, path_to_mocks)

chi_squared(data_stats, model_stats, sigma, mat)

data_stats = [-1.52636489, -1.67495453, -1.89217831, -2.60866256,  0.82645562, 
    0.58006042,  0.32635061,  0.09183673,  0.50428082,  0.33464567,
    0.22067039,  0.04255319, 10.32296892, 10.41302058, 10.43689304,
    10.80875171,  9.92036448,  9.98886097, 10.02533109, 10.05904768]

model_stats = [-1.73441117, -1.9072183 , -2.13926361, -2.94373088,  0.83503268,
    0.65494978,  0.34580754,  0.09195402,  0.51870242,  0.37688442,
    0.21483771,  0.12      , 10.29579639, 10.31774426, 10.45108414,
    10.61262035,  9.88168716,  9.90425777,  9.9949131 , 10.08223915]

sigma = [0.12096691, 0.14461279, 0.16109724, 0.18811391, 0.02877629,
    0.03783503, 0.02460691, 0.01696295, 0.01958187, 0.01558674,
    0.01662875, 0.02255644, 0.05394899, 0.03946977, 0.02376962,
    0.08010174, 0.07213797, 0.05349873, 0.04179512, 0.0685496 ]

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

# SMF
fig1= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# fblue
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# sigma-M*
fig1= plt.figure(figsize=(10,10))

bins_red = np.linspace(1,2.8,5)
bins_blue = np.linspace(1,2.5,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

########################################################################
#* Looking at chi-squared distribution in a small region around best-fit point

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import emcee
import math
import os

__author__ = '{Mehnaaz Asad}'

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{bm}")
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', width=2, size=7)
plt.rc('ytick.major', width=2, size=7)

def find_nearest(array, value): 
    """Finds the element in array that is closest to the value

    Args:
        array (numpy.array): Array of values
        value (numpy.float): Value to find closest match to

    Returns:
        numpy.float: Closest match found in array
    """
    array = np.asarray(array) 
    idx = (np.abs(array - value)).argmin() 
    return array[idx] 

nwalkers_arr = [100, 100, 500, 100] #change to 500 if run==75
runs_arr = [61, 67, 75, 78]

for vals in zip(nwalkers_arr, runs_arr):
    nwalkers = vals[0]
    run = vals[1]
    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_proc = dict_of_paths['proc_dir']
    path_to_interim = dict_of_paths['int_dir']
    path_to_figures = dict_of_paths['plot_dir']

    survey = 'eco'
    mf_type = 'smf'
    quenching = 'hybrid'
    path_to_proc = path_to_proc + 'smhm_colour_run{0}/'.format(run)

    reader = emcee.backends.HDFBackend(
        path_to_proc + "chain.h5", read_only=True)
    flatchain = reader.get_chain(flat=True)

    names_hybrid=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'Mstar_q','Mhalo_q','mu','nu']
    names_halo=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'Mh_qc','Mh_qs','mu_c','mu_s']

    if quenching == 'hybrid':
        emcee_table = pd.DataFrame(flatchain, columns=names_hybrid)
    elif quenching == 'halo':
        emcee_table = pd.DataFrame(flatchain, columns=names_halo)       

    chi2 = reader.get_blobs(flat=True)
    emcee_table['chi2'] = chi2

    walker_id_arr = np.zeros(len(emcee_table))
    iteration_id_arr = np.zeros(len(emcee_table))
    counter_wid = 0
    counter_stepid = 0
    for idx,row in emcee_table.iterrows():
        counter_wid += 1
        if idx % nwalkers == 0:
            counter_stepid += 1
            counter_wid = 1
        walker_id_arr[idx] = counter_wid
        iteration_id_arr[idx] = counter_stepid

    id_data = {'walker_id': walker_id_arr, 'iteration_id': iteration_id_arr}
    id_df = pd.DataFrame(id_data, index=emcee_table.index)
    emcee_table = emcee_table.assign(**id_df)

    bf_params = emcee_table.loc[emcee_table.chi2 == emcee_table.chi2.min()][emcee_table.columns[:9]].values[0]
    emcee_table_params = emcee_table[emcee_table.columns[0:9]]

    # tol=0.1

    # region_around_bf = \
    # np.isclose(bf_params[0], emcee_table_params.iloc[:,0], atol=tol) & \
    # np.isclose(bf_params[1], emcee_table_params.iloc[:,1], atol=tol) & \
    # np.isclose(bf_params[2], emcee_table_params.iloc[:,2], atol=tol) & \
    # np.isclose(bf_params[3], emcee_table_params.iloc[:,3], atol=tol) & \
    # np.isclose(bf_params[4], emcee_table_params.iloc[:,4], atol=tol) & \
    # np.isclose(bf_params[5], emcee_table_params.iloc[:,5], atol=tol) & \
    # np.isclose(bf_params[6], emcee_table_params.iloc[:,6], atol=tol) & \
    # np.isclose(bf_params[7], emcee_table_params.iloc[:,7], atol=tol) & \
    # np.isclose(bf_params[8], emcee_table_params.iloc[:,8], atol=tol)

    # idx_where_true = np.array([i for i, x in enumerate(region_around_bf) if x])
    # voxel = emcee_table.loc[idx_where_true]

    # if run == 78:
    #     chi2_dist_78 = voxel.chi2.values
    # elif run == 74:
    #     chi2_dist_74 = voxel.chi2.values

    # plt.hist(chi2_dist_78, histtype='step', lw=2, density=True)
    # plt.hist(chi2_dist_74, histtype='step', lw=2, ls='--', density=True)
    # plt.show()

    std = np.std(emcee_table_params, axis=0).values
    fraction_of_std = 0.5
    tol = fraction_of_std*std

    region_around_bf = \
    np.isclose(bf_params[0], emcee_table_params.iloc[:,0], atol=tol[0]) & \
    np.isclose(bf_params[1], emcee_table_params.iloc[:,1], atol=tol[1]) & \
    np.isclose(bf_params[2], emcee_table_params.iloc[:,2], atol=tol[2]) & \
    np.isclose(bf_params[3], emcee_table_params.iloc[:,3], atol=tol[3]) & \
    np.isclose(bf_params[4], emcee_table_params.iloc[:,4], atol=tol[4]) & \
    np.isclose(bf_params[5], emcee_table_params.iloc[:,5], atol=tol[5]) & \
    np.isclose(bf_params[6], emcee_table_params.iloc[:,6], atol=tol[6]) & \
    np.isclose(bf_params[7], emcee_table_params.iloc[:,7], atol=tol[7]) & \
    np.isclose(bf_params[8], emcee_table_params.iloc[:,8], atol=tol[8])

    idx_where_true = np.array([i for i, x in enumerate(region_around_bf) if x])
    voxel = emcee_table.loc[idx_where_true]


    if run == 61: #last good chain
        chi2_dist_61 = voxel.chi2.values #155 points
    elif run == 67: #first bad chain
        chi2_dist_67 = voxel.chi2.values #176 points
    elif run == 75: #chain with 500 walkers
        chi2_dist_75 = voxel.chi2.values #31 points
    elif run == 78: #latest chain with improvement when ps was removed
        chi2_dist_78 = voxel.chi2.values #168 points

fig1 = plt.figure()
plt.hist(chi2_dist_61, histtype='step', lw=2, ls='-', label='Last good chain (dr2, no ps, old bins)')
plt.hist(chi2_dist_67, histtype='step', lw=2, ls='--', label='First bad chain (dr3, ps, new bins)')
plt.hist(chi2_dist_75, histtype='step', lw=2, ls='-.', label='500 walkers (dr3, ps, new bins)')
plt.hist(chi2_dist_78, histtype='step', lw=2, ls='dotted', label='Latest good chain (dr3, no ps, new bins)')
plt.legend(loc='best', prop={'size':20})
plt.title(r"{0}\% of std(param)".format(fraction_of_std*100))
plt.show()

################################################################################
#* Testing effect on chi-squared of removing bins using chain 80

import multiprocessing
import time
import h5py
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

def get_err_data(path_to_proc, bin_i=None):
    # Read in datasets from h5 file and calculate corr matrix
    hf_read = h5py.File(path_to_proc +'corr_matrices.h5', 'r')
    hf_read.keys()
    smf = hf_read.get('smf')
    smf = np.array(smf)
    fblue_cen = hf_read.get('fblue_cen')
    fblue_cen = np.array(fblue_cen)
    fblue_sat = hf_read.get('fblue_sat')
    fblue_sat = np.array(fblue_sat)
    mean_mstar_red = hf_read.get('mean_mstar_red')
    mean_mstar_red = np.array(mean_mstar_red)
    mean_mstar_blue = hf_read.get('mean_mstar_blue')
    mean_mstar_blue = np.array(mean_mstar_blue)

    for i in range(100):
        phi_total_0 = smf[i][:,0]
        phi_total_1 = smf[i][:,1]
        phi_total_2 = smf[i][:,2]
        phi_total_3 = smf[i][:,3]

        f_blue_cen_0 = fblue_cen[i][:,0]
        f_blue_cen_1 = fblue_cen[i][:,1]
        f_blue_cen_2 = fblue_cen[i][:,2]
        f_blue_cen_3 = fblue_cen[i][:,3]

        f_blue_sat_0 = fblue_sat[i][:,0]
        f_blue_sat_1 = fblue_sat[i][:,1]
        f_blue_sat_2 = fblue_sat[i][:,2]
        f_blue_sat_3 = fblue_sat[i][:,3]

        mstar_red_cen_0 = mean_mstar_red[i][:,0]
        mstar_red_cen_1 = mean_mstar_red[i][:,1]
        mstar_red_cen_2 = mean_mstar_red[i][:,2]
        mstar_red_cen_3 = mean_mstar_red[i][:,3]

        mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
        mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
        mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
        mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

        combined_df = pd.DataFrame({
            'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
            'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
            'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
            'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
            'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
            'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
            'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
            'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
            'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
            'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

        # Drop multiple columns simultaneously
        if type(bin_i) != "NoneType" and type(bin_i) == np.ndarray:
            combined_df = combined_df.drop(combined_df.columns[bin_i], 
                axis=1)
        # Drop only one column
        elif type(bin_i) != "NoneType" and type(bin_i) == np.int64:
            combined_df = combined_df.drop(combined_df.columns[bin_i], 
                axis=1)

        if i == 0:
            # Correlation matrix of phi and deltav colour measurements combined
            # corr_mat_global = combined_df.corr()
            cov_mat_global = combined_df.cov()

            # corr_mat_average = corr_mat_global
            cov_mat_average = cov_mat_global
        else:
            # corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
            cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
            

    # Using average cov mat to get correlation matrix
    corr_mat_average = calc_corr_mat(cov_mat_average)
    corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
    sigma_average = np.sqrt(np.diag(cov_mat_average))

    # corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average.values) 
    # sigma_average = np.sqrt(np.diag(cov_mat_average))

    return sigma_average, corr_mat_inv_colour_average, corr_mat_average

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if stacked_stat:
        if mf_type == 'smf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])
        elif mf_type == 'bmf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mbary_sigma, blue_deltav, \
                blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])

    else:
        if mf_type == 'smf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))

        elif mf_type == 'bmf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mbary_sigma, blue_sigma, \
                blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))


    if stacked_stat:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    else:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()
    
    return data_arr, z_median

def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz', 'galid']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])

    model_arr = np.array(model_arr)

    return model_arr

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

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'
        # censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
            
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.default_rng(df['galid'][ii])
        rfloat = rng.uniform()
        # Comparing against f_red
        if (rfloat >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    model.mock.populate(seed=1993)

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global new_chain
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = False
pca = False
new_chain = True

survey = 'eco'
machine = 'bender'
mf_type = 'smf'
quenching = 'hybrid'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_data_buffer_volh1_dr2.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

# best-fit from chain 80
bf_bin_params = [1.26205996e+01, 1.07021820e+01, 4.76597732e-01, 3.92284946e-01,
       1.68466879e-01, 1.01816555e+01, 1.34914230e+01, 7.29355889e-01,
       3.71080916e-02]
bf_bin_chi2 = 30.68
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, inv_corr_mat, corr_mat = get_err_data(path_to_proc)

data_stats = [-1.52518353, -1.67537112, -1.89183513, -2.60866256,  0.82258065,
        0.58109091,  0.32606325,  0.09183673,  0.46529814,  0.31311707,
        0.21776504,  0.04255319, 10.33120738, 10.44089237, 10.49140916,
       10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856]

model_stats = [-1.66317738, -1.85990648, -2.07118114, -2.83833701,  0.82231532,
        0.65212028,  0.34490961,  0.06306306,  0.46302943,  0.35278515,
        0.20985915,  0.03174603, 10.30132771, 10.38368034, 10.51834202,
       10.75250721,  9.9557085 ,  9.9486084 , 10.05703831, 10.18317699]

sigma = [0.12096691, 0.14461279, 0.16109724, 0.18811391, 0.02793883,
       0.03976063, 0.02793187, 0.02040563, 0.01832583, 0.01750134,
       0.01705853, 0.02817948, 0.0566326 , 0.04073082, 0.02219082,
       0.07895785, 0.08783216, 0.04570364, 0.04221103, 0.0590798 ]

data_stats = np.array(data_stats)
model_stats = np.array(model_stats)
sigma = np.array(sigma)

#* Remove all bins one by one (with replacement)
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.arange(0, 20, 1)
for bin in bins_to_remove:
    sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bin)
    data_stats_mod  = np.delete(data_stats, bin) 
    model_stats_mod  = np.delete(model_stats, bin) 
    chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
    chi2_arr.append(chi2)

chi2_arr = [30.68179416042088,
 30.680306555741588,
 30.234877245304528,
 30.433398175450954,
 30.674358114987946,
 22.121928128691323,
 23.711346372044147,
 30.674070689880907,
 29.35214057727323,
 30.08353621893616,
 25.570350994447317,
 30.645039631820715,
 30.347681882682664,
 30.496386772423406,
 30.168943765737097,
 28.844115301573574,
 27.335989721630824,
 30.45199945476248,
 27.791912728280614,
 30.142060507362977,
 30.548253975521515]

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['No bins removed',
r'$\Phi 1$', r'$\Phi 2$', r'$\Phi 3$', r'$\Phi 4$',
r'$fblue_{cen}\ 1$', r'$fblue_{cen}\ 2$', r'$fblue_{cen}\ 3$', r'$fblue_{cen}\ 4$',
r'$fblue_{sat}\ 1$', r'$fblue_{sat}\ 2$', r'$fblue_{sat}\ 3$', r'$fblue_{sat}\ 4$',
r'$\sigma_{red}\ 1$', r'$\sigma_{red}\ 2$', 
r'$\sigma_{red}\ 3$', r'$\sigma_{red}\ 4$',
r'$\sigma_{blue}\ 1$', r'$\sigma_{blue}\ 2$', 
r'$\sigma_{blue}\ 3$', r'$\sigma_{blue}\ 4$']

plt.axhline(chi2_arr[0], ls='--', c='k', label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))

plt.legend(loc='best', prop={'size':20})
plt.plot(tick_marks, chi2_arr)
plt.scatter(tick_marks, chi2_arr)
plt.xlabel("Bin removed")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()

#* Instead of removing one stat at a time (below), just test one stat at a time
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)

bins_to_remove = [np.arange(4, 20, 1), #keep smf
                  np.append(np.arange(0, 4, 1), np.arange(8, 20, 1)), #fblue_cen
                  np.append(np.arange(0, 8, 1), np.arange(12, 20, 1)), #fblue_sat
                  np.append(np.arange(0, 12, 1), np.arange(16, 20, 1)), #sigma_red
                  np.arange(0, 16, 1)] #sigma_blue

for bins in bins_to_remove:
    sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins)

    data_stats_mod  = np.delete(data_stats, bins) 
    model_stats_mod  = np.delete(model_stats, bins) 

    chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
    chi2_arr.append(chi2)

chi2_arr = [30.68179416042088,
 2.355438804262331,
 13.596167837070684,
 6.41580351797884,
 5.419720974011917,
 2.8210676311750373]

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['All stats', r"$\boldsymbol\phi$", r"$\boldsymbol{f_{blue}^{c}}$", 
r"$\boldsymbol{f_{blue}^{s}}$", r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

plt.axhline(chi2_arr[0], ls='--', c='k', lw=3,
    label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))
plt.plot(tick_marks, chi2_arr, lw=3)
plt.scatter(tick_marks, chi2_arr, s=120)

plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(loc='best', prop={'size':30})
plt.xlabel("Stat used")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()


#* Remove all red sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([12, 13, 14, 15])
# idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

# num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, bins_to_remove) 
model_stats_mod  = np.delete(model_stats, bins_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr: 
#[30.68179416042088, 23.747718419456405]

#* Remove all blue sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([16, 17, 18, 19])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[30.68179416042088, 26.789455777355037]

#* Remove all red+blue sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = [12, 13, 14, 15, 16, 17, 18, 19]
idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, mat_mod = get_err_data(path_to_proc, idxs_to_remove)

num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 
model_stats_mod  = np.delete(model_stats, len(bins_to_remove)*[num_elems_by_idx]-idxs_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[30.681793210609147, 19.31249299291156]

#* Remove all fblue cen bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([4, 5, 6, 7])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[30.68179416042088, 14.935304005227987]

#* Remove all fblue sat bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([8, 9, 10, 11])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[30.68179416042088, 24.924382985813264]

#* Remove all smf bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([0, 1, 2, 3])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[30.68179416042088, 30.129999030923532]


chi2_arr = [30.68179416042088, #no bins removed
    30.129999030923532, #smf removed
    14.935304005227987, #fblue cen removed
    24.924382985813264, #fblue sat removed
    23.747718419456405, #red sigma removed
    26.789455777355037] #blue sigma removed

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['None', r"$\boldsymbol\phi$", r"$\boldsymbol{f_{blue}^{c}}$", 
r"$\boldsymbol{f_{blue}^{s}}$", r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

plt.axhline(chi2_arr[0], ls='--', c='k', lw=3,
    label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))
plt.plot(tick_marks, chi2_arr, lw=3)
plt.scatter(tick_marks, chi2_arr, s=120)

plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(loc='best', prop={'size':30})
plt.xlabel("Stat removed")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()


#* Plotting observables from chain 80

# SMF
fig1= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# fblue
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# sigma-M*
fig1= plt.figure(figsize=(10,10))

bins_red = np.linspace(1,2.8,5)
bins_blue = np.linspace(1,2.5,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()
##########################################################################
#* Testing effect on chi-squared of removing bins using chain 81 (halo)

import multiprocessing
import time
import h5py
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

def get_err_data(path_to_proc, bin_i=None):
    # Read in datasets from h5 file and calculate corr matrix
    hf_read = h5py.File(path_to_proc +'corr_matrices_{0}.h5'.format(quenching), 'r')
    hf_read.keys()
    smf = hf_read.get('smf')
    smf = np.array(smf)
    fblue_cen = hf_read.get('fblue_cen')
    fblue_cen = np.array(fblue_cen)
    fblue_sat = hf_read.get('fblue_sat')
    fblue_sat = np.array(fblue_sat)
    mean_mstar_red = hf_read.get('mean_mstar_red')
    mean_mstar_red = np.array(mean_mstar_red)
    mean_mstar_blue = hf_read.get('mean_mstar_blue')
    mean_mstar_blue = np.array(mean_mstar_blue)

    for i in range(100):
        phi_total_0 = smf[i][:,0]
        phi_total_1 = smf[i][:,1]
        phi_total_2 = smf[i][:,2]
        phi_total_3 = smf[i][:,3]

        f_blue_cen_0 = fblue_cen[i][:,0]
        f_blue_cen_1 = fblue_cen[i][:,1]
        f_blue_cen_2 = fblue_cen[i][:,2]
        f_blue_cen_3 = fblue_cen[i][:,3]

        f_blue_sat_0 = fblue_sat[i][:,0]
        f_blue_sat_1 = fblue_sat[i][:,1]
        f_blue_sat_2 = fblue_sat[i][:,2]
        f_blue_sat_3 = fblue_sat[i][:,3]

        mstar_red_cen_0 = mean_mstar_red[i][:,0]
        mstar_red_cen_1 = mean_mstar_red[i][:,1]
        mstar_red_cen_2 = mean_mstar_red[i][:,2]
        mstar_red_cen_3 = mean_mstar_red[i][:,3]

        mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
        mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
        mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
        mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

        combined_df = pd.DataFrame({
            'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
            'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
            'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
            'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
            'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
            'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
            'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
            'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
            'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
            'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

        # Drop multiple columns simultaneously
        if type(bin_i) != "NoneType" and type(bin_i) == np.ndarray:
            combined_df = combined_df.drop(combined_df.columns[bin_i], 
                axis=1)
        # Drop only one column
        elif type(bin_i) != "NoneType" and type(bin_i) == np.int64:
            combined_df = combined_df.drop(combined_df.columns[bin_i], 
                axis=1)

        if i == 0:
            # Correlation matrix of phi and deltav colour measurements combined
            # corr_mat_global = combined_df.corr()
            cov_mat_global = combined_df.cov()

            # corr_mat_average = corr_mat_global
            cov_mat_average = cov_mat_global
        else:
            # corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
            cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
            

    # Using average cov mat to get correlation matrix
    corr_mat_average = calc_corr_mat(cov_mat_average)
    corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
    sigma_average = np.sqrt(np.diag(cov_mat_average))

    # corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average.values) 
    # sigma_average = np.sqrt(np.diag(cov_mat_average))

    return sigma_average, corr_mat_inv_colour_average, corr_mat_average

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if stacked_stat:
        if mf_type == 'smf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])
        elif mf_type == 'bmf':
            print('Measuring stacked velocity dispersion for data')
            red_deltav, red_cen_mbary_sigma, blue_deltav, \
                blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

            sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red_data = np.log10(sigma_red_data[0])
            sigma_blue_data = np.log10(sigma_blue_data[0])

    else:
        if mf_type == 'smf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))

        elif mf_type == 'bmf':
            print('Measuring velocity dispersion for data')
            red_sigma, red_cen_mbary_sigma, blue_sigma, \
                blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))


    if stacked_stat:

        phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
            sigma_red_data, sigma_blue_data

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()

    else:

        phi_total_data, f_blue_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
            vdisp_blue_data = total_data[1], f_blue_data[1], f_blue_data[2], f_blue_data[3], \
            mean_mstar_red_data[0], mean_mstar_blue_data[0]

        data_arr = []
        data_arr.append(phi_total_data)
        data_arr.append(f_blue_cen_data)
        data_arr.append(f_blue_sat_data)
        data_arr.append(vdisp_red_data)
        data_arr.append(vdisp_blue_data)
        data_arr.append(f_blue_total_data)
        data_arr = np.array(data_arr)
        data_arr = data_arr.flatten()

        if pca:
            #* Same as data_arr = data_arr.dot(mat)
            data_arr = data_arr[:n_eigen]*mat.diagonal()
    
    return data_arr, z_median

def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)
    randint_logmstar = None

    if theta[0] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[1] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[2] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
    if theta[3] < 0:
        chi2 = -np.inf
        return -np.inf, chi2       
    if theta[4] < 0.1:
        chi2 = -np.inf
        return -np.inf, chi2

    if theta[5] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[6] < 0:# or theta[6] > 16:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[7] < 0:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]
    if theta[8] < 0:# or theta[8] > 5:
        chi2 = -np.inf
        return -np.inf, chi2
        # return -np.inf, [chi2, randint_logmstar]       

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.1].reset_index(drop=True)
    
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz', 'galid']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')

            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(8.6,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.8,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,2.5,5))
    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        if stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                    gal_group_df, 'model')
            #! Max bin not the same as in obs 1&2
            sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                statistic='std', bins=np.linspace(9.1,11,5))
            
            sigma_red = np.log10(sigma_red[0])
            sigma_blue = np.log10(sigma_blue[0])
        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = get_velocity_dispersion(
                    gal_group_df, 'model')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    if stacked_stat:
        model_arr.append(sigma_red)
        model_arr.append(sigma_blue)
    else:
        model_arr.append(mean_mstar_red[0])
        model_arr.append(mean_mstar_blue[0])
    model_arr.append(f_blue[1]) 

    model_arr = np.array(model_arr)

    return model_arr

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

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'
        # censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.4
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)

        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.4
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary >= mbary_limit]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        catl.logmbary = np.log10((10**catl.logmbary) / 2.041)

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.4
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
            
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # rng = np.random.default_rng(df['galid'][ii])
        # rfloat = rng.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    model.mock.populate(seed=1993)

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

def get_err_data_extra(survey, path):
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

    phi_total_arr = []
    f_blue_tot_arr = []
    f_blue_cen_arr = []
    f_blue_sat_arr = []
    mean_mstar_red_arr = []
    mean_mstar_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 

            ## Pair splitting
            psgrpid = split_false_pairs(
                np.array(mock_pd.ra),
                np.array(mock_pd.dec),
                np.array(mock_pd.cz), 
                np.array(mock_pd.groupid))

            mock_pd["ps_groupid"] = psgrpid

            arr1 = mock_pd.ps_groupid
            arr1_unq = mock_pd.ps_groupid.drop_duplicates()  
            arr2_unq = np.arange(len(np.unique(mock_pd.ps_groupid))) 
            mapping = dict(zip(arr1_unq, arr2_unq))   
            new_values = arr1.map(mapping)
            mock_pd['ps_groupid'] = new_values  

            most_massive_gal_idxs = mock_pd.groupby(['ps_groupid'])['logmstar']\
                .transform(max) == mock_pd['logmstar']        
            grp_censat_new = most_massive_gal_idxs.astype(int)
            mock_pd["ps_grp_censat"] = grp_censat_new

            # Deal with the case where one group has two equally massive galaxies
            groups = mock_pd.groupby('ps_groupid')
            keys = groups.groups.keys()
            groupids = []
            for key in keys:
                group = groups.get_group(key)
                if np.sum(group.ps_grp_censat.values)>1:
                    groupids.append(key)
            
            final_sat_idxs = []
            for key in groupids:
                group = groups.get_group(key)
                cens = group.loc[group.ps_grp_censat.values == 1]
                num_cens = len(cens)
                final_sat_idx = random.sample(list(cens.index.values), num_cens-1)
                # mock_pd.ps_grp_censat.loc[mock_pd.index == final_sat_idx] = 0
                final_sat_idxs.append(final_sat_idx)
            final_sat_idxs = np.hstack(final_sat_idxs)

            mock_pd.loc[final_sat_idxs, 'ps_grp_censat'] = 0
            #

            mock_pd = mock_add_grpcz(mock_pd, grpid_col='ps_groupid', 
                galtype_col='ps_grp_censat', cen_cz_col='cz')
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            if mf_type == 'smf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit) & \
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)
            elif mf_type == 'bmf':
                mock_pd = mock_pd.loc[(mock_pd.grpcz_cen.values >= min_cz) & \
                    (mock_pd.grpcz_cen.values <= max_cz) & \
                    (mock_pd.M_r.values <= mag_limit)].reset_index(drop=True)

            ## Using best-fit found for new ECO data using result from chain 59
            ## i.e. hybrid quenching model which was the last time sigma-M* was
            ## used i.e. stacked_stat = True
            bf_from_last_chain = [10.133745, 13.478087, 0.810922,
                0.043523]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            ## Using best-fit found for new ECO data using result from chain 76
            ## i.e. halo quenching model
            bf_from_last_chain = [11.7784379, 12.1174944, 1.62771601, 0.131290924]

            Mh_qc = bf_from_last_chain[0] # Msun/h
            Mh_qs = bf_from_last_chain[1] # Msun/h
            mu_c = bf_from_last_chain[2]
            mu_s = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock_extra(f_red_c, f_red_s, mock_pd)

            if mf_type == 'smf':
                logmstar_arr = mock_pd.logmstar.values
                #Measure SMF of mock using diff_smf function
                max_total, phi_total, err_total, counts_total = \
                    diff_smf(logmstar_arr, volume, False)
            elif mf_type == 'bmf':
                logmstar_arr = mock_pd.logmstar.values
                mhi_arr = mock_pd.mhi.values
                logmgas_arr = np.log10(1.4 * mhi_arr)
                logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
                #Measure BMF of mock using diff_bmf function
                max_total, phi_total, err_total, counts_total = \
                    diff_bmf(logmbary_arr, volume, False)

            phi_total_arr.append(phi_total)

            #Measure blue fraction of galaxies
            f_blue = blue_frac(mock_pd, False, False)
            f_blue_tot_arr.append(f_blue[1])
            f_blue_cen_arr.append(f_blue[2])
            f_blue_sat_arr.append(f_blue[3])
    
            #Measure dynamics of red and blue galaxy groups
            if stacked_stat:
                if mf_type == 'smf':
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                elif mf_type == 'bmf':
                    red_deltav, red_cen_mbary_sigma, blue_deltav, \
                        blue_cen_mbary_sigma = get_stacked_velocity_dispersion(mock_pd, 'mock')

                    sigma_red = bs(red_cen_mbary_sigma, red_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    sigma_blue = bs( blue_cen_mbary_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(9.1,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                mean_mstar_red_arr.append(sigma_red)
                mean_mstar_blue_arr.append(sigma_blue)

            else:
                if mf_type == 'smf':
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.8,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,2.5,5))
                
                elif mf_type == 'bmf':
                    red_sigma, red_cen_mbary_sigma, blue_sigma, \
                        blue_cen_mbary_sigma = get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mbary_sigma, 
                        statistic=average_of_log, bins=np.linspace(1,3,5))
                
                mean_mstar_red_arr.append(mean_mstar_red[0])
                mean_mstar_blue_arr.append(mean_mstar_blue[0])

    phi_arr_total = np.array(phi_total_arr)

    f_blue_tot_arr = np.array(f_blue_tot_arr)
    f_blue_cen_arr = np.array(f_blue_cen_arr)
    f_blue_sat_arr = np.array(f_blue_sat_arr)

    mean_mstar_red_arr = np.array(mean_mstar_red_arr)
    mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

    phi_total_0 = phi_arr_total[:,0]
    phi_total_1 = phi_arr_total[:,1]
    phi_total_2 = phi_arr_total[:,2]
    phi_total_3 = phi_arr_total[:,3]

    f_blue_tot_0 = f_blue_tot_arr[:,0]
    f_blue_tot_1 = f_blue_tot_arr[:,1]
    f_blue_tot_2 = f_blue_tot_arr[:,2]
    f_blue_tot_3 = f_blue_tot_arr[:,3]

    f_blue_cen_0 = f_blue_cen_arr[:,0]
    f_blue_cen_1 = f_blue_cen_arr[:,1]
    f_blue_cen_2 = f_blue_cen_arr[:,2]
    f_blue_cen_3 = f_blue_cen_arr[:,3]

    f_blue_sat_0 = f_blue_sat_arr[:,0]
    f_blue_sat_1 = f_blue_sat_arr[:,1]
    f_blue_sat_2 = f_blue_sat_arr[:,2]
    f_blue_sat_3 = f_blue_sat_arr[:,3]

    mstar_red_cen_0 = mean_mstar_red_arr[:,0]
    mstar_red_cen_1 = mean_mstar_red_arr[:,1]
    mstar_red_cen_2 = mean_mstar_red_arr[:,2]
    mstar_red_cen_3 = mean_mstar_red_arr[:,3]

    mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
    mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
    mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
    mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

    combined_df = pd.DataFrame({
        'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
        'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
        'f_blue_tot_0':f_blue_tot_0, 'f_blue_tot_1':f_blue_tot_1, 
        'f_blue_tot_2':f_blue_tot_2, 'f_blue_tot_3':f_blue_tot_3,
        'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
        'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
        'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
        'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
        'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
        'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
        'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
        'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})


    # Correlation matrix of phi and deltav colour measurements combined
    corr_mat_colour = combined_df.corr()
    corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
    err_colour = np.sqrt(np.diag(combined_df.cov()))

    if pca:
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_colour)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_colour.shape[0], corr_mat_colour.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_colour.shape[0], :corr_mat_colour.shape[0]] = diag(s)

        ## values in s are singular values. Corresponding (possibly non-zero) 
        ## eigenvalues are given by s**2.

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt two more times 
        # to be able to compare directly to values in Sigma 
        max_eigen = np.sqrt(np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr)))))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* max_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>max_eigen])
        sigma_mat = sigma_mat[:, :n_elements]
        # VT = VT[:n_elements, :]
        # reconstruct
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        # transform 2 ways (this is how you would transform data, model and sigma
        # i.e. new_data = data.dot(Sigma) etc.)
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        #* Same as err_colour.dot(sigma_mat)
        err_colour = err_colour[:n_elements]*sigma_mat.diagonal()

    if pca:
        return err_colour, sigma_mat, n_elements
    else:
        return err_colour, corr_mat_inv_colour, corr_mat_colour

def assign_colour_label_mock_extra(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.uniform()
        # Comparing against f_red
        if (rng >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df



global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global new_chain
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = False
pca = False
new_chain = True

survey = 'eco'
machine = 'bender'
mf_type = 'smf'
quenching = 'halo'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_data_buffer_volh1_dr2.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

# best-fit from chain 81
bf_bin_params = [ 12.35698883,  10.46610398,   0.39712252,   0.44163207,
         0.35142279,  11.90634744,  12.52440607,   1.49938572,
         0.56093193]
bf_bin_chi2 = 29.84
model_stats = lnprob(bf_bin_params)
model_stats = model_stats.flatten()

sigma, inv_corr_mat, corr_mat = get_err_data(path_to_proc)

data_stats = [-1.52518353, -1.67537112, -1.89183513, -2.60866256,  0.82258065,
        0.58109091,  0.32606325,  0.09183673,  0.46529814,  0.31311707,
        0.21776504,  0.04255319, 10.33120738, 10.44089237, 10.49140916,
       10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856]
       
model_stats = [-1.6564254 , -1.7807524 , -2.03369465, -2.76348575,  0.81732903,
        0.6344226 ,  0.3176573 ,  0.08406114,  0.47584034,  0.35026178,
        0.20680628,  0.06097561, 10.34584904, 10.40047073, 10.51291656,
       10.71528149,  9.89633656,  9.94518757, 10.01201057, 10.07794762]

sigma = [0.12096691, 0.14461279, 0.16109724, 0.18811391, 0.03344477,
       0.02951727, 0.01276146, 0.01933712, 0.02596754, 0.01971784,
       0.02179486, 0.05598426, 0.05052889, 0.03923217, 0.02351261,
       0.08510228, 0.12295059, 0.0928405 , 0.08819096, 0.13137178]

data_stats = np.array(data_stats)
model_stats = np.array(model_stats)
sigma = np.array(sigma)

#* Remove all bins one by one (with replacement)
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.arange(0, 20, 1)
for bin in bins_to_remove:
    sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bin)
    data_stats_mod  = np.delete(data_stats, bin) 
    model_stats_mod  = np.delete(model_stats, bin) 
    chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
    chi2_arr.append(chi2)

chi2_arr = [29.849581281406106,
 28.41726287576236,
 28.227429256869904,
 29.7631834352416,
 29.336080230997936,
 24.143234200954986,
 26.10127123720048,
 29.412421596777406,
 29.56784059428626,
 29.815616248618852,
 26.12935589120969,
 28.546463127569126,
 29.805856867188144,
 29.624627112869216,
 28.280353390298767,
 28.767293023554057,
 25.099175209472055,
 29.290235797394566,
 29.820647515861673,
 28.19828045236985,
 29.30267938705152]

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['No bins removed',
r'$\Phi 1$', r'$\Phi 2$', r'$\Phi 3$', r'$\Phi 4$',
r'$fblue_{cen}\ 1$', r'$fblue_{cen}\ 2$', r'$fblue_{cen}\ 3$', r'$fblue_{cen}\ 4$',
r'$fblue_{sat}\ 1$', r'$fblue_{sat}\ 2$', r'$fblue_{sat}\ 3$', r'$fblue_{sat}\ 4$',
r'$\sigma_{red}\ 1$', r'$\sigma_{red}\ 2$', 
r'$\sigma_{red}\ 3$', r'$\sigma_{red}\ 4$',
r'$\sigma_{blue}\ 1$', r'$\sigma_{blue}\ 2$', 
r'$\sigma_{blue}\ 3$', r'$\sigma_{blue}\ 4$']

plt.axhline(chi2_arr[0], ls='--', c='k', label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))

plt.legend(loc='best', prop={'size':20})
plt.plot(tick_marks, chi2_arr)
plt.scatter(tick_marks, chi2_arr)
plt.xlabel("Bin removed")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()

#* Instead of removing one stat at a time (below), just test one stat at a time
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)

bins_to_remove = [np.arange(4, 20, 1), #keep smf
                  np.append(np.arange(0, 4, 1), np.arange(8, 20, 1)), #fblue_cen
                  np.append(np.arange(0, 8, 1), np.arange(12, 20, 1)), #fblue_sat
                  np.append(np.arange(0, 12, 1), np.arange(16, 20, 1)), #sigma_red
                  np.arange(0, 16, 1)] #sigma_blue

for bins in bins_to_remove:
    sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins)

    data_stats_mod  = np.delete(data_stats, bins) 
    model_stats_mod  = np.delete(model_stats, bins) 

    chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
    chi2_arr.append(chi2)

chi2_arr = [29.849581281406106,
 4.2251542947599665,
 10.541617050901456,
 6.150057592766247,
 7.535391724061271,
 1.4824080703889633]

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['All stats', r"$\boldsymbol\phi$", r"$\boldsymbol{f_{blue}^{c}}$", 
r"$\boldsymbol{f_{blue}^{s}}$", r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

plt.axhline(chi2_arr[0], ls='--', c='k', lw=3,
    label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))
plt.plot(tick_marks, chi2_arr, lw=3)
plt.scatter(tick_marks, chi2_arr, s=120)

plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(loc='best', prop={'size':30})
plt.xlabel("Stat used")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()


#* Remove all red sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([12, 13, 14, 15])
# idxs_to_remove = np.array([19]*len(bins_to_remove)) - np.array([bins_to_remove])[0]
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

# num_elems_by_idx = len(data_stats) - 1
data_stats_mod  = np.delete(data_stats, bins_to_remove) 
model_stats_mod  = np.delete(model_stats, bins_to_remove) 

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr: 
#[29.849581281406106, 20.370967045867847]

#* Remove all blue sigma bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([16, 17, 18, 19])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[29.849581281406106, 27.56767457718615]

#* Remove all fblue cen bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([4, 5, 6, 7])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[29.849581281406106, 20.737832565249892]

#* Remove all fblue sat bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([8, 9, 10, 11])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[29.849581281406106, 25.553599256392054]

#* Remove all smf bins
chi2_arr = []
chi2 = chi_squared(data_stats, model_stats, sigma, inv_corr_mat)
chi2_arr.append(chi2)
bins_to_remove = np.array([0, 1, 2, 3])
sigma_mod, inv_corr_mat_mod, corr_mat_mod = get_err_data(path_to_proc, bins_to_remove)

data_stats_mod  = np.delete(data_stats, bins_to_remove)
model_stats_mod  = np.delete(model_stats, bins_to_remove)

chi2 = chi_squared(data_stats_mod, model_stats_mod, sigma_mod, inv_corr_mat_mod)
chi2_arr.append(chi2)

## chi2_arr:
#[29.849581281406106, 25.823274588090428]


chi2_arr = [29.849581281406106, #no bins removed
    25.823274588090428, #smf removed
    20.737832565249892, #fblue cen removed
    25.553599256392054, #fblue sat removed
    20.370967045867847, #red sigma removed
    27.56767457718615] #blue sigma removed

fig1 = plt.figure()
tick_marks = [i for i in range(len(chi2_arr))]
names = ['None', r"$\boldsymbol\phi$", r"$\boldsymbol{f_{blue}^{c}}$", 
r"$\boldsymbol{f_{blue}^{s}}$", r"$\boldsymbol{\overline{M_{*,red}^{c}}}$",
r"$\boldsymbol{\overline{M_{*,blue}^{c}}}$"]

plt.axhline(chi2_arr[0], ls='--', c='k', lw=3,
    label=r'$\chi^2$={0}'.format(np.round(chi2_arr[0], 2)))
plt.plot(tick_marks, chi2_arr, lw=3)
plt.scatter(tick_marks, chi2_arr, s=120)

plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(loc='best', prop={'size':30})
plt.xlabel("Stat removed")
plt.ylabel(r'$\chi^2$')
plt.xticks(tick_marks, names, fontsize=20, rotation='vertical')
plt.show()

#* fblue total comparison between data and best-fit model

fblue_total_data = [0.69847199,  0.48992322,  0.29620853,  0.08230453]
fblue_total_bf = [0.74095536,  0.57783339,  0.30184805,  0.08216433]
sigma_extra, inv_corr_mat_extra, corr_mat_extra = get_err_data_extra(survey, path_to_mocks)
#4-8 is extra fblue total stat
sigma_extra = [0.12119863, 0.14473986, 0.16135636, 0.18798836, 0.03102185,
       0.02240753, 0.01098   , 0.01781961, 0.0320873 , 0.03061238,
       0.01383636, 0.01851325, 0.02765851, 0.01917606, 0.01971427,
       0.04815691, 0.05380581, 0.04124026, 0.02225705, 0.07028498,
       0.1060082 , 0.08239894, 0.07757622, 0.13916155]

sigma = [0.12096691, 0.14461279, 0.16109724, 0.18811391, 0.03344477,
       0.02951727, 0.01276146, 0.01933712, 0.02596754, 0.01971784,
       0.02179486, 0.05598426, 0.05052889, 0.03923217, 0.02351261,
       0.08510228, 0.12295059, 0.0928405 , 0.08819096, 0.13137178]

fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dct= plt.errorbar(bin_centers, fblue_total_data, yerr=sigma_extra[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')

# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bft, = plt.plot(bin_centers, fblue_total_bf,
    color='rebeccapurple', ls='--', lw=4, zorder=10)


plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([dt, bft], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

#* Plotting observables from chain 81

# SMF
fig1= plt.figure(figsize=(10,10))

x = [ 8.925,  9.575, 10.225, 10.875]

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# fblue
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# sigma-M*
fig1= plt.figure(figsize=(10,10))

bins_red = np.linspace(1,2.8,5)
bins_blue = np.linspace(1,2.5,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:], yerr=sigma[16:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()


#* Plotting observables from chain 84 (28 stats stellar hybrid)

import multiprocessing
import time
import h5py
# import cProfile
# import pstats
# import io

# Libs
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import random
import emcee 
import math
import os

def calc_corr_mat(df):
    num_cols = df.shape[1]
    corr_mat = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            num = df.values[i][j]
            denom = np.sqrt(df.values[i][i] * df.values[j][j])
            corr_mat[i][j] = num/denom
    return corr_mat

def get_err_data(path_to_file):
    # Read in datasets from h5 file and calculate corr matrix
    hf_read = h5py.File(path_to_file, 'r')
    hf_read.keys()
    smf = hf_read.get('smf')
    smf = np.squeeze(np.array(smf))
    fblue_cen = hf_read.get('fblue_cen')
    fblue_cen = np.array(fblue_cen)
    fblue_sat = hf_read.get('fblue_sat')
    fblue_sat = np.array(fblue_sat)
    mean_mstar_red = hf_read.get('mean_mstar_red')
    mean_mstar_red = np.array(mean_mstar_red)
    mean_mstar_blue = hf_read.get('mean_mstar_blue')
    mean_mstar_blue = np.array(mean_mstar_blue)
    sigma_red = hf_read.get('sigma_red')
    sigma_red = np.array(sigma_red)
    sigma_blue = hf_read.get('sigma_blue')
    sigma_blue = np.array(sigma_blue)

    for i in range(100):
        phi_total_0 = smf[i][:,0]
        phi_total_1 = smf[i][:,1]
        phi_total_2 = smf[i][:,2]
        phi_total_3 = smf[i][:,3]

        f_blue_cen_0 = fblue_cen[i][:,0]
        f_blue_cen_1 = fblue_cen[i][:,1]
        f_blue_cen_2 = fblue_cen[i][:,2]
        f_blue_cen_3 = fblue_cen[i][:,3]

        f_blue_sat_0 = fblue_sat[i][:,0]
        f_blue_sat_1 = fblue_sat[i][:,1]
        f_blue_sat_2 = fblue_sat[i][:,2]
        f_blue_sat_3 = fblue_sat[i][:,3]

        mstar_red_cen_0 = mean_mstar_red[i][:,0]
        mstar_red_cen_1 = mean_mstar_red[i][:,1]
        mstar_red_cen_2 = mean_mstar_red[i][:,2]
        mstar_red_cen_3 = mean_mstar_red[i][:,3]

        mstar_blue_cen_0 = mean_mstar_blue[i][:,0]
        mstar_blue_cen_1 = mean_mstar_blue[i][:,1]
        mstar_blue_cen_2 = mean_mstar_blue[i][:,2]
        mstar_blue_cen_3 = mean_mstar_blue[i][:,3]

        sigma_red_0 = sigma_red[i][:,0]
        sigma_red_1 = sigma_red[i][:,1]
        sigma_red_2 = sigma_red[i][:,2]
        sigma_red_3 = sigma_red[i][:,3]

        sigma_blue_0 = sigma_blue[i][:,0]
        sigma_blue_1 = sigma_blue[i][:,1]
        sigma_blue_2 = sigma_blue[i][:,2]
        sigma_blue_3 = sigma_blue[i][:,3]

        combined_df = pd.DataFrame({
            'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
            'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
            'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
            'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
            'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
            'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
            'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
            'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
            'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
            'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3,
            'sigma_red_0':sigma_red_0, 'sigma_red_1':sigma_red_1, 
            'sigma_red_2':sigma_red_2, 'sigma_red_3':sigma_red_3,
            'sigma_blue_0':sigma_blue_0, 'sigma_blue_1':sigma_blue_1, 
            'sigma_blue_2':sigma_blue_2, 'sigma_blue_3':sigma_blue_3})

        if i == 0:
            # Correlation matrix of phi and deltav colour measurements combined
            corr_mat_global = combined_df.corr()
            cov_mat_global = combined_df.cov()

            corr_mat_average = corr_mat_global
            cov_mat_average = cov_mat_global
        else:
            corr_mat_average = pd.concat([corr_mat_average, combined_df.corr()]).groupby(level=0, sort=False).mean()
            cov_mat_average = pd.concat([cov_mat_average, combined_df.cov()]).groupby(level=0, sort=False).mean()
            

    # Using average cov mat to get correlation matrix
    corr_mat_average = calc_corr_mat(cov_mat_average)
    corr_mat_inv_colour_average = np.linalg.inv(corr_mat_average) 
    sigma_average = np.sqrt(np.diag(cov_mat_average))

    if pca:
        num_mocks = 64
        #* Testing SVD
        from scipy.linalg import svd
        from numpy import zeros
        from numpy import diag
        # Singular-value decomposition
        U, s, VT = svd(corr_mat_average)
        # create m x n Sigma matrix
        sigma_mat = zeros((corr_mat_average.shape[0], corr_mat_average.shape[1]))
        # populate Sigma with n x n diagonal matrix
        sigma_mat[:corr_mat_average.shape[0], :corr_mat_average.shape[0]] = diag(s)

        ## values in s are eigenvalues #confirmed by comparing s to 
        ## output of np.linalg.eig(C) where the first array is array of 
        ## eigenvalues.
        ## https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

        # Equation 10 from Sinha et. al 
        # LHS is actually eigenvalue**2 so need to take the sqrt one more time 
        # to be able to compare directly to values in Sigma since eigenvalues 
        # are squares of singular values 
        # (http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf)
        # https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm#:~:text=The%20SVD%20represents%20an%20expansion,up%20the%20columns%20of%20U.
        min_eigen = np.sqrt(np.sqrt(2/(num_mocks)))
        #* Note: for a symmetric matrix, the singular values are absolute values of 
        #* the eigenvalues which means 
        #* min_eigen = np.sqrt(np.sqrt(2/(num_mocks*len(box_id_arr))))
        n_elements = len(s[s>min_eigen])
        VT = VT[:n_elements, :]

        print("Number of principle components kept: {0}".format(n_elements))

        ## reconstruct
        # sigma_mat = sigma_mat[:, :n_elements]
        # B = U.dot(sigma_mat.dot(VT))
        # print(B)
        ## transform 2 ways (this is how you would transform data, model and sigma
        ## i.e. new_data = data.dot(Sigma) etc.) <- not sure this is correct
        #* Both lines below are equivalent transforms of the matrix BUT to project 
        #* data, model, err in the new space, you need the eigenvectors NOT the 
        #* eigenvalues (sigma_mat) as was used earlier. The eigenvectors are 
        #* VT.T (the right singular vectors) since those have the reduced 
        #* dimensions needed for projection. Data and model should then be 
        #* projected similarly to err_colour below. 
        #* err_colour.dot(sigma_mat) no longer makes sense since that is 
        #* multiplying the error with the eigenvalues. Using sigma_mat only
        #* makes sense in U.dot(sigma_mat) since U and sigma_mat were both derived 
        #* by doing svd on the matrix which is what you're trying to get in the 
        #* new space by doing U.dot(sigma_mat). 
        #* http://www.math.usm.edu/lambers/cos702/cos702_files/docs/PCA.pdf
        # T = U.dot(sigma_mat)
        # print(T)
        # T = corr_mat_colour.dot(VT.T)
        # print(T)

        err_colour_pca = sigma_average.dot(VT.T)
        eigenvectors = VT.T

    if pca:
        return err_colour_pca, eigenvectors
    else:
        return sigma_average, corr_mat_inv_colour_average

def calc_data_stats(catl_file):
    print('Reading catalog') #No Mstar cut needed as catl_file already has it
    catl, volume, z_median = read_data_catl(catl_file, survey)

    print('Assigning colour label to data')
    catl = assign_colour_label_data(catl)

    if mf_type == 'smf':
        print('Measuring SMF for data')
        total_data = measure_all_smf(catl, volume, True)
    elif mf_type == 'bmf':
        print('Measuring BMF for data')
        logmbary = catl.logmbary_a23.values
        total_data = diff_bmf(logmbary, volume, False)

    print('Measuring blue fraction for data')
    f_blue_data = blue_frac(catl, False, True)

    if mf_type == 'smf':
        print('Measuring stacked velocity dispersion for data')
        red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 'data')

        sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        
        sigma_red_data = np.log10(sigma_red_data[0])
        sigma_blue_data = np.log10(sigma_blue_data[0])
    elif mf_type == 'bmf':
        print('Measuring stacked velocity dispersion for data')
        red_deltav, red_cen_mbary_sigma, blue_deltav, \
            blue_cen_mbary_sigma = get_stacked_velocity_dispersion(catl, 'data')

        sigma_red_data = bs(red_cen_mbary_sigma, red_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        sigma_blue_data = bs( blue_cen_mbary_sigma, blue_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        
        sigma_red_data = np.log10(sigma_red_data[0])
        sigma_blue_data = np.log10(sigma_blue_data[0])

    if mf_type == 'smf':
        print('Measuring velocity dispersion for data')
        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma = get_velocity_dispersion(catl, 'data')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))

    elif mf_type == 'bmf':
        print('Measuring velocity dispersion for data')
        red_sigma, red_cen_mbary_sigma, blue_sigma, \
            blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'data')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))

    phi_total_data, f_blue_cen_data, f_blue_sat_data, vdisp_red_data, \
        vdisp_blue_data, mean_mstar_red_data, mean_mstar_blue_data = total_data[1], f_blue_data[2], f_blue_data[3], \
        sigma_red_data, sigma_blue_data,  mean_mstar_red_data[0], mean_mstar_blue_data[0]

    data_arr = []
    data_arr.append(phi_total_data)
    data_arr.append(f_blue_cen_data)
    data_arr.append(f_blue_sat_data)
    data_arr.append(mean_mstar_red_data)
    data_arr.append(mean_mstar_blue_data)
    data_arr.append(vdisp_red_data)
    data_arr.append(vdisp_blue_data)
    data_arr = np.array(data_arr)
    data_arr = data_arr.flatten() # flatten from (5,4) to (1,20)

    if pca:
        data_arr = data_arr.dot(mat)
    
    return data_arr, z_median

def lnprob(theta):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values
    
    phi: array
        Array of y-axis values of mass function
    
    err: numpy.array
        Array of error values of red and blue mass function

    corr_mat: array
        Array of inverse of correlation matrix

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    # Moved to outside the try clause for cases where parameter values are 
    # outside the prior (specific one was when theta[1] was > 14)
    # randint_logmstar = np.random.randint(1,101)

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': False,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    gals_df = populate_mock(theta[:5], model_init)
    if mf_type == 'smf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].reset_index(drop=True)
    elif mf_type == 'bmf':
        gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**9.0].reset_index(drop=True)
    
    gals_df = apply_rsd(gals_df)

    gals_df = gals_df.loc[\
        (gals_df['cz'] >= cz_inner) &
        (gals_df['cz'] <= cz_outer)].reset_index(drop=True)
    
    gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
        gals_df['halo_id'], 1, 0)

    cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
        'stellar_mass', 'ra', 'dec', 'cz', 'galid']
    gals_df = gals_df[cols_to_use]

    gals_df.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

    gals_df['logmstar'] = np.log10(gals_df['logmstar'])


    if quenching == 'hybrid':
        f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, \
            'vishnu')
    elif quenching == 'halo':
        f_red_cen, f_red_sat = halo_quenching_model(theta[5:], gals_df, \
            'vishnu')
    gals_df = assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

    # npmax = 1e5
    # if len(gals_df) >= npmax:
    #     print("size of df, {0}, is >= npmax {1}\n".format(len(gals_df), npmax))
    #     print("(test) WARNING! Increasing memory allocation\n")
    #     npmax*=1.2

    gal_group_df = group_finding(gals_df,
        path_to_data + 'interim/', param_dict)

    ## Pair splitting
    psgrpid = split_false_pairs(
        np.array(gal_group_df.ra),
        np.array(gal_group_df.dec),
        np.array(gal_group_df.cz), 
        np.array(gal_group_df.groupid))

    gal_group_df["ps_groupid"] = psgrpid

    arr1 = gal_group_df.ps_groupid
    arr1_unq = gal_group_df.ps_groupid.drop_duplicates()  
    arr2_unq = np.arange(len(np.unique(gal_group_df.ps_groupid))) 
    mapping = dict(zip(arr1_unq, arr2_unq))   
    new_values = arr1.map(mapping)
    gal_group_df['ps_groupid'] = new_values  

    most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
        .transform(max) == gal_group_df['logmstar']        
    grp_censat_new = most_massive_gal_idxs.astype(int)
    gal_group_df["ps_grp_censat"] = grp_censat_new

    gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='ps_groupid', 
        galtype_col='ps_grp_censat', cen_cz_col='cz')

    ## Making a similar cz cut as in data which is based on grpcz being 
    ## defined as cz of the central of the group "grpcz_cen"
    cz_inner_mod = 3000
    gal_group_df = gal_group_df.loc[\
        (gal_group_df['ps_cen_cz'] >= cz_inner_mod) &
        (gal_group_df['ps_cen_cz'] <= cz_outer)].reset_index(drop=True)

    dist_inner = kms_to_Mpc(H0,cz_inner_mod) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    # v_sim = 130**3
    # v_sim = 890641.5172927063 #survey volume used in group_finder.py

    ## Observable #1 - Total SMF
    if mf_type == 'smf':
        total_model = measure_all_smf(gal_group_df, survey_vol, False) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        ## Observable #3 
        red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                gal_group_df, 'model')

        sigma_red = bs(red_cen_mstar_sigma, red_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
            statistic='std', bins=np.linspace(8.6,10.8,5))
        
        sigma_red = np.log10(sigma_red[0])
        sigma_blue = np.log10(sigma_blue[0])

        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma = get_velocity_dispersion(
                gal_group_df, 'model')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))

    elif mf_type == 'bmf':
        logmstar_col = 'logmstar' #No explicit logmbary column
        total_model = diff_bmf(10**(gal_group_df[logmstar_col]), 
            survey_vol, True) 

        ## Observable #2 - Blue fraction
        f_blue = blue_frac(gal_group_df, True, False)
    
        red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = get_stacked_velocity_dispersion(
                gal_group_df, 'model')
        sigma_red = bs(red_cen_mstar_sigma, red_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
            statistic='std', bins=np.linspace(9.0,11.2,5))
        
        sigma_red = np.log10(sigma_red[0])
        sigma_blue = np.log10(sigma_blue[0])

        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma = get_velocity_dispersion(
                gal_group_df, 'model')

        red_sigma = np.log10(red_sigma)
        blue_sigma = np.log10(blue_sigma)

        mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.8,5))
        mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
            statistic=average_of_log, bins=np.linspace(1,2.5,5))

    model_arr = []
    model_arr.append(total_model[1])
    model_arr.append(f_blue[2])   
    model_arr.append(f_blue[3])
    model_arr.append(mean_mstar_red[0])
    model_arr.append(mean_mstar_blue[0])
    model_arr.append(sigma_red)
    model_arr.append(sigma_blue)

    model_arr = np.array(model_arr)

    return model_arr

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

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')
        
        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) & 
                (eco_buff.grpcz_cen.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)
        
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

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def assign_colour_label_data(catl):
    """
    Assign colour label to data

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    Returns
    ---------
    catl: pandas Dataframe
        Data catalog with colour label assigned as new column
    """

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
    """
    Calculates differential stellar mass function for all, red and blue galaxies
    from mock/data

    Parameters
    ----------
    table: pandas Dataframe
        Dataframe of either mock or data 
    volume: float
        Volume of simulation/survey
    cvar: float
        Cosmic variance error
    data_bool: Boolean
        Data or mock

    Returns
    ---------
    3 multidimensional arrays of stellar mass, phi, total error in SMF and 
    counts per bin for all, red and blue galaxies
    """

    colour_col = 'colour_label'

    if data_bool:
        logmstar_col = 'logmstar'
        max_total, phi_total, err_total, counts_total = \
            diff_smf(table[logmstar_col], volume, False)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
        #     volume, False, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
        #     volume, False, 'B')
    else:
        if randint_logmstar:
            logmstar_col = '{0}'.format(randint_logmstar)
        else:
            logmstar_col = 'logmstar'
        ## Changed to 10**X because Behroozi mocks now have M* values in log
        max_total, phi_total, err_total, counts_total = \
            diff_smf(10**(table[logmstar_col]), volume, True)
        # max_red, phi_red, err_red, bins_red, counts_red = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
        #     volume, True, 'R')
        # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
        #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
        #     volume, True, 'B')

    return [max_total, phi_total, err_total, counts_total]
    # return [max_total, phi_total, err_total, counts_total] , \
    #     [max_red, phi_red, err_red, counts_red] , \
    #         [max_blue, phi_blue, err_blue, counts_blue]

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
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
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

    return maxis, phi, err_tot, counts

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
    """
    Calculates blue fraction in bins of stellar mass (which are converted to h=1)

    Parameters
    ----------
    catl: pandas Dataframe 
        Data catalog

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1
    
    data_bool: boolean
        True if data, False if mocks
    
    randint_logmstar (optional): int
        Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    f_blue: array
        Array of y-axis blue fraction values
    """

    if data_bool:
        censat_col = 'g_galtype'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            mass_total_arr = catl.logmbary_a23.values
            mass_cen_arr = catl.logmbary_a23.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary_a23.loc[catl[censat_col] == 0].values

    ## Mocks case different than data because of censat_col
    elif not data_bool and not h1_bool:

        # Not g_galtype anymore after applying pair splitting
        censat_col = 'ps_grp_censat'
        # censat_col = 'g_galtype'
        # censat_col = 'cs_flag'

        if mf_type == 'smf':
            mass_total_arr = catl.logmstar.values
            mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

        elif mf_type == 'bmf':
            logmstar_arr = catl.logmstar.values
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10(1.4 * mhi_arr)
            logmbary_arr = calc_bary(logmstar_arr, logmgas_arr)
            catl["logmbary"] = logmbary_arr

            mass_total_arr = catl.logmbary.values
            mass_cen_arr = catl.logmbary.loc[catl[censat_col] == 1].values
            mass_sat_arr = catl.logmbary.loc[catl[censat_col] == 0].values

    # elif randint_logmstar != 1 and randint_logmstar is not None:
    #     mass_total_arr = catl['{0}'.format(randint_logmstar)].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
    # elif randint_logmstar == 1:
    #     mass_total_arr = catl['behroozi_bf'].values
    #     censat_col = 'g_galtype_{0}'.format(randint_logmstar)
    #     mass_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
    #     mass_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values
    
    # New case where no subset of mocks are used and group finding is done within
    # mcmc framework
    elif randint_logmstar is None:
        mass_total_arr = catl['logmstar'].values
        censat_col = 'ps_grp_censat'
        mass_cen_arr = catl['logmstar'].loc[catl[censat_col] == 1].values
        mass_sat_arr = catl['logmstar'].loc[catl[censat_col] == 0].values
    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
        logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
        logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)
    else:
        logmass_total_arr = mass_total_arr
        logmass_cen_arr = mass_cen_arr
        logmass_sat_arr = mass_sat_arr

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            mstar_limit = 8.9
            bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
            #* Changed max bin from 11.5 to 11.1 to be the same as mstar-sigma (10.8)
            bin_max = np.round(np.log10((10**11.1) / 2.041), 1)

        elif mf_type == 'bmf':
            mbary_limit = 9.3
            bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bin_num = 5
        bins = np.linspace(bin_min, bin_max, bin_num)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)

    result_total = bs(logmass_total_arr, colour_label_total_arr, 
        blue_frac_helper, bins=bins)
    result_cen = bs(logmass_cen_arr, colour_label_cen_arr, blue_frac_helper, 
        bins=bins)
    result_sat = bs(logmass_sat_arr, colour_label_sat_arr, blue_frac_helper, 
        bins=bins)
    edges = result_total[1]
    dm = edges[1] - edges[0]  # Bin width
    maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
    f_blue_total = result_total[0]

    f_blue_cen = result_cen[0]
    f_blue_sat = result_sat[0]

    return maxis, f_blue_total, f_blue_cen, f_blue_sat

def gapper(vel_arr):
    n = len(vel_arr)
    factor = np.sqrt(np.pi)/(n*(n-1))

    summation = 0
    sorted_vel = np.sort(vel_arr)
    for i in range(len(sorted_vel)):
        i += 1
        if i == len(sorted_vel):
            break
        
        deltav_i = sorted_vel[i] - sorted_vel[i-1]
        weight_i = i*(n-i)
        prod = deltav_i * weight_i
        summation += prod

    sigma_gapper = factor * summation

    return sigma_gapper

def get_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)

        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        elif level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    #* DF of only N > 1 groups sorted by ps_groupid
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        #* np.sum doesn't actually add anything since there is only one central 
        #* per group (checked)
        #* Could use np.concatenate(cen_red_subset_df.groupby(['{0}'.format(id_col),
        #*    '{0}'.format(galtype_col)])[logmstar_col].apply(np.array).ravel())
        red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
            
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    #* The gapper method does not exclude the central 
    red_sigma_arr = red_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # red_sigma_arr = gapper(red_subset_df['deltav'])

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
            '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    if mf_type == 'smf':
        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
            blue_cen_bary_mass_arr

def get_stacked_velocity_dispersion(catl, catl_type, randint=None):
    """Calculating velocity dispersion of groups from real data, model or 
        mock

    Args:
        catl (pandas.DataFrame): Data catalogue 

        catl_type (string): 'data', 'mock', 'model'

        randint (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

    Returns:
        red_sigma_arr (numpy array): Velocity dispersion of red galaxies

        red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

        blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
        
        blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

        red_nsat_arr (numpy array): Number of satellites around red centrals

        blue_nsat_arr (numpy array): Number of satellites around blue centrals

    """
    mstar_limit = 8.9
    mbary_limit = 9.3
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            if mf_type == 'smf':
                catl = catl.loc[catl.logmstar >= mstar_limit]
                catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            elif mf_type == 'bmf':
                catl = catl.loc[catl.logmbary_a23 >= mbary_limit]
                catl.logmbary_a23 = np.log10((10**catl.logmbary_a23) / 2.041)
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary_a23'

        ## Use group level for data even when settings.level == halo
        galtype_col = 'g_galtype'
        id_col = 'ps_groupid'

    if catl_type == 'mock':
        catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
        mhi_arr = catl.mhi.values
        logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
        logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
        catl["logmbary"] = logmbary_arr

        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            catl = catl.loc[catl.logmbary >= np.log10((10**mbary_limit)/2.041)]

        logmstar_col = 'logmstar'
        logmbary_col = 'logmbary'

        if level == 'group':
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
        if level == 'halo':
            galtype_col = 'cs_flag'
            ## Halo ID is equivalent to halo_hostid in vishnu mock
            id_col = 'haloid'

    if catl_type == 'model':
        if survey == 'eco':
            min_cz = 3000
            max_cz = 12000
            if mf_type == 'smf':
                mstar_limit = 8.9
            elif mf_type == 'bmf':
                mstar_limit = 9.3
        elif survey == 'resolvea':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.9
        elif survey == 'resolveb':
            min_cz = 4500
            max_cz = 7000
            mstar_limit = 8.7

        if randint is None:
            logmstar_col = 'logmstar'
            logmbary_col = 'logmstar'
            galtype_col = 'ps_grp_censat'
            id_col = 'ps_groupid'
            cencz_col = 'ps_cen_cz'
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]
            # catl[logmstar_col] = np.log10(catl[logmstar_col])

        elif isinstance(randint, int) and randint != 1:
            logmstar_col = '{0}'.format(randint)
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        elif isinstance(randint, int) and randint == 1:
            logmstar_col = 'behroozi_bf'
            galtype_col = 'grp_censat_{0}'.format(randint)
            cencz_col = 'cen_cz_{0}'.format(randint)
            id_col = 'groupid_{0}'.format(randint)
            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
            # and M* star cuts to mimic mocks and data.
            # catl = mock_add_grpcz(catl, id_col, False, galtype_col, cencz_col)
            catl = catl.loc[
                (catl[cencz_col].values >= min_cz) & \
                (catl[cencz_col].values <= max_cz) & \
                (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

        if level == 'halo':
            galtype_col = 'cs_flag'
            id_col = 'halo_hostid'

        catl = models_add_avgrpcz(catl, id_col, galtype_col)

    red_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'R') & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
        colour_label == 'B') & (catl[galtype_col] == 1)].values)


    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    # red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_av']
    if mf_type == 'smf':
        red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = red_subset_df[logmbary_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
    elif mf_type == 'bmf':
        red_cen_bary_mass_arr = np.repeat(red_cen_bary_mass_arr, red_g_ngal_arr)

    red_deltav_arr = np.hstack(red_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    #* Using ddof = 1 means the denom in std is N-1 instead of N which is 
    #* another way to exclude the central from the measurement of sigma 
    #* We can no longer use deltav > 0 since deltav is wrt to average grpcz 
    #* instead of central cz.
    # red_sigma_arr = red_subset_df.groupby(id_col)['cz'].apply(np.std, ddof=1).values

    blue_subset_df = catl.loc[catl[id_col].isin(blue_subset_ids)]
    #* Excluding N=1 groups
    blue_subset_ids = blue_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    blue_subset_df = catl.loc[catl[id_col].isin(
        blue_subset_ids)].sort_values(by='{0}'.format(id_col))
    # cen_blue_subset_df = blue_subset_df.loc[blue_subset_df[galtype_col] == 1]
    # blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
    #     '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = blue_subset_df[logmbary_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    if mf_type == 'smf':
        blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    elif mf_type == 'bmf':
        blue_cen_bary_mass_arr = np.repeat(blue_cen_bary_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    if mf_type == 'smf':
        return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
            blue_cen_stellar_mass_arr
    elif mf_type == 'bmf':
        return red_deltav_arr, red_cen_bary_mass_arr, blue_deltav_arr, \
            blue_cen_bary_mass_arr

def calc_bary(logmstar_arr, logmgas_arr):
    """Calculates baryonic mass of galaxies from survey"""
    logmbary = np.log10((10**logmstar_arr) + (10**logmgas_arr))
    return logmbary

def diff_bmf(mass_arr, volume, h1_bool, colour_flag=False):
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
        bin_min = np.round(np.log10((10**9.3) / 2.041), 1)

        if survey == 'eco':
            # *checked 
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        elif survey == 'resolvea':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)

        bins = np.linspace(bin_min, bin_max, 5)

    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**9.1) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 5)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss

    phi = counts / (volume * dm)  # not a log quantity
    phi = np.log10(phi)

    return [maxis, phi, err_tot, counts]

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

    return f_red_cen, f_red_sat

def halo_quenching_model(theta, gals_df, mock):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(gals_df, \
        mock)

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

def assign_colour_label_mock(f_red_cen, f_red_sat, df, drop_fred=False):
    """
    Assign colour label to mock catalog

    Parameters
    ----------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    gals_df: pandas Dataframe
        Mock catalog
    drop_fred: boolean
        Whether or not to keep red fraction column after colour has been
        assigned

    Returns
    ---------
    df: pandas Dataframe
        Dataframe with colour label and random number assigned as 
        new columns
    """

    # Saving labels
    color_label_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
    df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['cs_flag']):
        # Draw a random number
        rng = np.random.default_rng(df['galid'][ii])
        rfloat = rng.uniform()
        # Comparing against f_red
        if (rfloat >= f_red_arr[ii]):
            color_label = 'B'
        else:
            color_label = 'R'
        # Saving to list
        color_label_arr[ii] = color_label
    
    ## Assigning to DataFrame
    df.loc[:, 'colour_label'] = color_label_arr
    # Dropping 'f_red` column
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    return df

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

    model.mock.populate(seed=1993)

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

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    proc_id = multiprocessing.current_process().pid
    # print(proc_id)
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    mock_coord_path = '{0}.galcatl_radecczlogmstar_{1}.{2}'.format(mock_zz_file, proc_id, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','logmstar']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)

    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)

    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged
    
def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

def chi_squared(data, model, err_data, inv_corr_mat):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: array
        Array of error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    # chi_squared_arr = (data - model)**2 / (err_data**2)
    # chi_squared = np.sum(chi_squared_arr)

    data = data.flatten() # from (2,5) to (1,10)
    model = model.flatten() # same as above

    # print('data: \n', data)
    # print('model: \n', model)

    first_term = ((data - model) / (err_data)).reshape(1,data.size)
    third_term = np.transpose(first_term)

    # chi_squared is saved as [[value]]
    chi_squared = np.dot(np.dot(first_term,inv_corr_mat),third_term)

    return chi_squared[0][0]

global model_init
global survey
global mf_type
global quenching
global path_to_data
global level
global stacked_stat
global pca
global n_eigen

# rseed = 12
# np.random.seed(rseed)
level = "group"
stacked_stat = "both"
pca = False

survey = 'eco'
machine = 'bender'
mf_type = 'bmf'
quenching = 'halo'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_external = dict_of_paths['ext_dir']
path_to_data = dict_of_paths['data_dir']
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

if machine == 'bender':
    halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
elif machine == 'mac':
    halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

if survey == 'eco':
    if mf_type == 'smf':
        catl_file = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
    elif mf_type == 'bmf':
        catl_file = path_to_proc + \
        "gal_group_eco_bary_buffer_volh1_dr3.hdf5"    
    else:
        print("Incorrect mass function chosen")
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

data_stats, z_median = calc_data_stats(catl_file)

print('Initial population of halo catalog')
model_init = halocat_init(halo_catalog, z_median)

# best-fit from chain 84 stellar hybrid
bf_params = [12.64869191, 10.68374592, 0.4640583 , 0.43375466, 0.22177846,
10.18868649, 13.23990274, 0.72632304, 0.05996219]
bf_chi2 = 43.14
model_stats = lnprob(bf_params)
model_stats = model_stats.flatten()

path_to_matrix_file = path_to_proc + 'corr_matrices_28stats_{0}_{1}_nonps.h5'.format(quenching, mf_type)
sigma, mat = get_err_data(path_to_matrix_file)

data_stats = [-1.50641087, -1.66717009, -1.75855679, -2.14068123,  0.83244838,
        0.65177066,  0.40886203,  0.21748401,  0.46857773,  0.34697218,
        0.26781857,  0.11111111, 10.33120738, 10.44089237, 10.49140916,
       10.90520085,  9.96134779, 10.02206739, 10.08378187, 10.19648856,
        1.90950953,  2.01440507,  1.94387242,  2.10831986,  1.74316835,
        1.79926695,  1.93256557,  1.97005728]

model_stats = [-1.70195279, -1.87296322, -2.00209101, -2.37598763,  0.82998679,
        0.70775047,  0.46346248,  0.18514473,  0.45787037,  0.38256088,
        0.28105395,  0.10869565, 10.33463573, 10.41190052, 10.52964211,
       10.72712326,  9.88106728,  9.92773151, 10.04058456, 10.17961121,
        1.93286022,  1.94543751,  2.01022921,  2.13724606,  1.89070318,
        1.84061227,  1.91560009,  2.04094885]

sigma = [0.12096691, 0.14461279, 0.16109724, 0.18811391, 0.02683347,
       0.03742902, 0.0266406 , 0.0197476 , 0.01853414, 0.01434367,
       0.0197633 , 0.02624559, 0.0536389 , 0.0397238 , 0.0223247 ,
       0.07396154, 0.10179693, 0.06076056, 0.04809142, 0.06816211,
       0.13539179, 0.19428472, 0.03608344, 0.06012444, 0.16559101,
       0.2396115 , 0.04601781, 0.11904627]

# best-fit from chain 86 stellar halo
bf_params = [12.45003142, 10.49533185, 0.42905612, 0.49491889, 0.38661993,
11.928341 , 12.60596691, 1.63365685, 0.35175002]
bf_chi2 = 35.94
model_stats = lnprob(bf_params)
model_stats = model_stats.flatten()

path_to_matrix_file = path_to_proc + 'corr_matrices_28stats_{0}_{1}_nonps.h5'.format(quenching, mf_type)
sigma, mat = get_err_data(path_to_matrix_file)

model_stats = [-1.64349243, -1.7843237 , -1.94234316, -2.33136953,  0.84144082,
        0.70512821,  0.44420738,  0.20167653,  0.48678943,  0.39503932,
        0.28335171,  0.140625  , 10.33659172, 10.41757107, 10.53171158,
       10.67783356,  9.80300045,  9.95078373, 10.03885365, 10.12372112,
        1.83072994,  1.90328147,  2.00537302,  2.14842957,  1.74507088,
        1.82174325,  1.90501129,  2.03934095]

sigma = [0.11934883, 0.14286006, 0.1527114 , 0.17191049, 0.03702661,
       0.0537891 , 0.03863945, 0.02016892, 0.04017546, 0.0326368 ,
       0.02795144, 0.02434214, 0.05562563, 0.03860483, 0.02231773,
       0.07289891, 0.10084592, 0.07080061, 0.06209389, 0.11454521,
       0.1970029 , 0.23551372, 0.03475882, 0.05648364, 0.04871227,
       0.03694325, 0.04600347, 0.10839431]

# best-fit from chain 90 baryonic hybrid
bf_params = [12.48588217, 10.57524804, 0.58167099, 0.30082553, 0.27581956,
10.39975208, 14.37426995, 0.94420486, 0.10782901]
bf_chi2 = 54
model_stats = lnprob(bf_params)
model_stats = model_stats.flatten()

path_to_matrix_file = path_to_proc + 'corr_matrices_28stats_{0}_{1}_ps.h5'.format(quenching, mf_type)
sigma, mat = get_err_data(path_to_matrix_file)

data_stats = [-1.37100074, -1.56283196, -1.93629169, -2.66403959,  0.8465711 ,
        0.62694611,  0.37299465,  0.10691824,  0.55704008,  0.41626016,
        0.2283105 ,  0.09090909, 10.43607131, 10.48323239, 10.53705475,
       10.94332686, 10.0527434 , 10.01044694, 10.08551048, 10.27737551,
        1.83568257,  1.91195255,  2.07040896,  2.21856796,  1.64732242,
        1.88781988,  1.98457075,  2.43277928]

model_stats = [-1.51304886, -1.73780609, -2.14044939, -2.91409129,  0.86014641,
        0.70555937,  0.39954853,  0.079566  ,  0.56425091,  0.47849462,
        0.26576577,  0.11363636, 10.3751421 , 10.45606422, 10.59863472,
       10.7876749 , 10.00358677,  9.99996185, 10.09361458, 10.19808674,
        1.88775566,  1.94386461,  2.07966066,  2.23421911,  1.72938958,
        1.88854423,  1.99212893,  2.21316067]

sigma = [0.11961597, 0.1456231 , 0.16493722, 0.19162415, 0.02341389,
       0.04068774, 0.03412875, 0.03190922, 0.02444046, 0.02398993,
       0.02742082, 0.07505574, 0.05943878, 0.03782555, 0.02320546,
       0.06111394, 0.04873099, 0.03278858, 0.02829484, 0.04232432,
       0.07638922, 0.04130497, 0.04501977, 0.05627282, 0.03014632,
       0.03442977, 0.04950584, 0.11632295]

# best-fit from chain 91 baryonic halo
bf_params = [12.35324148, 10.52477267, 0.55656734, 0.41804463, 0.33896999,
11.97920231, 12.86108614, 1.75227584, 0.48775956]
bf_chi2 = 35
model_stats = lnprob(bf_params)
model_stats = model_stats.flatten()

path_to_matrix_file = path_to_proc + 'corr_matrices_28stats_{0}_{1}_ps_old.h5'.format(quenching, mf_type)
sigma, mat = get_err_data(path_to_matrix_file)

model_stats = [-1.44787217, -1.66373669, -2.03869068, -2.77890802,  0.86415262,
        0.69530703,  0.38558235,  0.12073491,  0.56395494,  0.43564837,
        0.25385935,  0.1509434 , 10.35595989, 10.43233871, 10.56695461,
       10.73343086, 10.02511787, 10.03193283, 10.10438251, 10.17681313,
        1.89489435,  1.9573198 ,  2.07053247,  2.23720224,  1.67020685,
        1.8618456 ,  1.93081347,  2.12055988]

sigma = [0.11963384, 0.14566991, 0.16501053, 0.19174992, 0.03304501,
       0.05192721, 0.02612118, 0.01629128, 0.03618085, 0.02804649,
       0.02602542, 0.0414303 , 0.04366212, 0.02863161, 0.02060698,
       0.06310771, 0.05936532, 0.04120691, 0.03497909, 0.07694819,
       0.0536965 , 0.03510162, 0.03798428, 0.05461796, 0.03170525,
       0.04084354, 0.09199716, 0.31607368]
       
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

# SMF/BMF
fig1= plt.figure(figsize=(10,10))

bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)
x = 0.5 * (bins[1:] + bins[:-1])

d = plt.errorbar(x, data_stats[:4], yerr=sigma[:4],
    color='k', fmt='s', ecolor='k',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bf, = plt.plot(x, model_stats[:4],
    color='k', ls='--', lw=4, zorder=10)

if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(d), (bf)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# fblue
fig2= plt.figure(figsize=(10,10))

mstar_limit = 8.9
bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
bin_max = np.round(np.log10((10**11.1) / 2.041), 1)
bin_num = 5
bins = np.linspace(bin_min, bin_max, bin_num)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

dc = plt.errorbar(bin_centers, data_stats[4:8], yerr=sigma[4:8],
    color='rebeccapurple', fmt='s', ecolor='rebeccapurple',markersize=12, 
    capsize=7, capthick=1.5, zorder=10, marker='^')
ds = plt.errorbar(bin_centers, data_stats[8:12], yerr=sigma[8:12],
    color='goldenrod', fmt='s', ecolor='goldenrod',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfc, = plt.plot(bin_centers, model_stats[4:8],
    color='rebeccapurple', ls='--', lw=4, zorder=10)
bfs, = plt.plot(bin_centers, model_stats[8:12],
    color='goldenrod', ls='--', lw=4, zorder=10)

if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)

plt.legend([(dc, ds), (bfc, bfs)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# sigma-M*/sigma-Mb
fig3= plt.figure(figsize=(10,10))

#Same binning for both M* and Mb
bins_red = np.linspace(1,2.8,5)
bins_blue = np.linspace(1,2.5,5)
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dr = plt.errorbar(bin_centers_red, data_stats[12:16], yerr=sigma[12:16],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[16:20], yerr=sigma[16:20],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[12:16],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[16:20],
    color='mediumblue', ls='--', lw=4, zorder=10)

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
if mf_type == 'smf':
    plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
elif mf_type == 'bmf':
    plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

# M*-sigma
fig4= plt.figure(figsize=(10,10))

if mf_type == 'smf':
    bins_red = np.linspace(8.6,10.8,5)
    bins_blue = np.linspace(8.6,10.8,5)
    bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])
elif mf_type == 'bmf':
    bins_red = np.linspace(9.0,11.2,5)
    bins_blue = np.linspace(9.0,11.2,5)
    bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])
    bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])
  
dr = plt.errorbar(bin_centers_red, data_stats[20:24], yerr=sigma[20:24],
    color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
db = plt.errorbar(bin_centers_blue, data_stats[24:], yerr=sigma[24:],
    color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bfr, = plt.plot(bin_centers_red, model_stats[20:24],
    color='maroon', ls='--', lw=4, zorder=10)
bfb, = plt.plot(bin_centers_blue, model_stats[24:],
    color='mediumblue', ls='--', lw=4, zorder=10)

if mf_type == 'smf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{* , group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
elif mf_type == 'bmf':
    plt.xlabel(r'\boldmath$\log_{10}\ M_{b , group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)  
plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=30)

plt.legend([(dr, db), (bfr, bfb)], ['Data','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
plt.show()

def hybrid_quenching_model(theta):
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

    sat_hosthalo_mass_arr = 10**(np.linspace(10.5, 14.5, 200))
    sat_stellar_mass_arr = 10**(np.linspace(8.6, 12, 200))
    cen_stellar_mass_arr = 10**(np.linspace(8.6, 12, 200))

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/(10**Mh_q))**nu))
    f_red_sat = 1 - (g_Mstar * h_Mh)

    return f_red_cen, f_red_sat

def halo_quenching_model(theta):
    """
    Apply halo quenching model from Zu and Mandelbaum 2015

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
    Mh_qc = theta[0] # Msun/h 
    Mh_qs = theta[1] # Msun/h
    mu_c = theta[2]
    mu_s = theta[3]

    cen_hosthalo_mass_arr = 10**(np.linspace(10, 15, 200))
    sat_hosthalo_mass_arr = 10**(np.linspace(10, 15, 200))

    f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
    f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

    return f_red_cen, f_red_sat

if quenching == "hybrid":
    if mf_type == 'smf':
        cen_stellar_mass_arr = np.linspace(8.6, 12, 200)
    elif mf_type == 'bmf':
        cen_stellar_mass_arr = np.linspace(9.0, 12, 200)
    an_fred_cen, an_fred_sat = hybrid_quenching_model(bf_params[5:])

    if mf_type == 'smf':
        antonio_data = pd.read_csv(path_to_external + "fquench_stellar/fqlogTSM_cen_DS_TNG_Salim_z0.csv", 
            index_col=0, skiprows=1, names=['fred_ds','logmstar','fred_tng','fred_salim'])
        plt.plot(antonio_data.logmstar.values, antonio_data.fred_ds.values, lw=5, c='k', ls='dashed', label='Dark Sage')
        plt.plot(antonio_data.logmstar.values, antonio_data.fred_salim.values, lw=5, c='k', ls='dotted', label='Salim+18')
        plt.plot(antonio_data.logmstar.values, antonio_data.fred_tng.values, lw=5, c='k', ls='dashdot', label='TNG')

    plt.plot(cen_stellar_mass_arr, an_fred_cen, lw=5, c='peru', 
        ls='dotted', label='analytical')
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=30)    
    plt.ylabel(r'\boldmath$f_{red, cen}$', fontsize=30)
    plt.legend(loc='best', prop={'size':25})
    plt.show()

    sat_hosthalo_mass_arr = np.linspace(10.5, 14.5, 200)
    if mf_type == 'smf':
        sat_stellar_mass_arr = np.linspace(8.6, 12, 200)
    elif mf_type == 'bmf':
        sat_stellar_mass_arr = np.linspace(9.0, 12, 200)
    an_fred_cen, an_fred_sat = hybrid_quenching_model(bf_params[5:])

    if mf_type == 'smf':
        antonio_data = pd.read_csv(path_to_external + "fquench_stellar/fqlogTSM_sat_DS_TNG_Salim_z0.csv", 
            index_col=0, skiprows=1, 
            names=['fred_ds','logmstar','fred_tng'])
        hosthalo_data = pd.read_csv(path_to_external + "fquench_halo/fqlogMvirhost_sat_DS_TNG_z0.csv", 
            index_col=0, skiprows=1, names=['fred_ds','logmhalo','fred_tng'])

        dss = plt.scatter(antonio_data.logmstar.values, antonio_data.fred_ds.values, 
            lw=5, c=hosthalo_data.logmhalo.values, cmap='viridis', marker='^', 
            s=120, label='Dark Sage', zorder=10)
        tngs = plt.scatter(antonio_data.logmstar.values, antonio_data.fred_tng.values, 
            lw=5, c=hosthalo_data.logmhalo.values, cmap='viridis', marker='s', 
            s=120, label='TNG', zorder=10)
        dsp, = plt.plot(antonio_data.logmstar.values, antonio_data.fred_ds.values, lw=3, c='k', ls='dashed')
        tngp, = plt.plot(antonio_data.logmstar.values, antonio_data.fred_tng.values, lw=3, c='k', ls='dashdot')

    plt.scatter(sat_stellar_mass_arr, an_fred_sat, alpha=0.4, s=150,
        c=sat_hosthalo_mass_arr, cmap='viridis',
        label='analytical')
    if mf_type == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, sat} \left[\mathrm{M_\odot}\,'\
                    r' \mathrm{h}^{-2} \right]$',fontsize=30)
    elif mf_type == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{b, sat} \left[\mathrm{M_\odot}\,'\
                    r' \mathrm{h}^{-2} \right]$',fontsize=30)    
    plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
    plt.colorbar(label=r'\boldmath$\log_{10}\ M_{h, host}$')
    plt.legend(loc='best', prop={'size':25})
    plt.show()

elif quenching == "halo":
    cen_hosthalo_mass_arr = np.linspace(10, 15, 200)
    an_fred_cen, an_fred_sat = halo_quenching_model(bf_params[5:])

    antonio_data = pd.read_csv(path_to_external + "fquench_halo/fqlogMvir_cen_DS_TNG_z0.csv", 
        index_col=0, skiprows=1, names=['fred_ds','logmhalo','fred_tng'])
    plt.plot(antonio_data.logmhalo.values, antonio_data.fred_ds.values, lw=5, c='k', ls='dashed', label='Dark Sage')
    plt.plot(antonio_data.logmhalo.values, antonio_data.fred_tng.values, lw=5, c='k', ls='dashdot', label='TNG')

    plt.plot(cen_hosthalo_mass_arr, an_fred_cen, lw=5, c='peru', 
        ls='dotted', label='analytical')
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    plt.ylabel(r'\boldmath$f_{red, cen}$', fontsize=30)
    plt.legend(loc='best', prop={'size':25})
    plt.show()

    sat_hosthalo_mass_arr = np.linspace(10, 15, 200)
    an_fred_cen, an_fred_sat = halo_quenching_model(bf_params[5:])

    antonio_data = pd.read_csv(path_to_external + "fquench_halo/fqlogMvirhost_sat_DS_TNG_z0.csv", 
        index_col=0, skiprows=1, names=['fred_ds','logmhalo','fred_tng'])
    dsp, = plt.plot(antonio_data.logmhalo.values, antonio_data.fred_ds.values, lw=5, c='k', ls='dashed', label='Dark Sage')
    tngp, = plt.plot(antonio_data.logmhalo.values, antonio_data.fred_tng.values, lw=5, c='k', ls='dashdot', label='TNG')

    plt.plot(sat_hosthalo_mass_arr, an_fred_sat, lw=5, c='peru', 
        ls='dotted', label='analytical')
    plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\,'\
                r' \mathrm{h}^{-1} \right]$',fontsize=30)
    plt.legend(loc='best', prop={'size':25})
    plt.show()

