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
#### CHAIN 71 - DR3

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

#### CHAIN 72 - New binning

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
# Chi-squared test using chain 72 : new binning and plotting mstar-sigma

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
# Chi-squared test using chain 71 : dr3 and plotting mstar-sigma

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




