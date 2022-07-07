"""
{This script:
1. Uses ECODR2 to test how much pair splitting affects observables
2. Uses ECODR3 to test how much new colours affect observables}
"""
__author__ = '{Mehnaaz Asad}'

from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
import pandas as pd
import numpy as np

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

def diff_smf(mstar_arr, volume, h1_bool):
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

    bin_min = np.round(np.log10((10**8.9) / 2.041), 1)        
    bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
    bin_num = 5

    bins = np.linspace(bin_min, bin_max, bin_num)

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

def blue_frac(catl, censat_col):
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
    mass_total_arr = catl.logmstar.values
    mass_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
    mass_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
    logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
    logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)

    mstar_limit = 8.9
    bin_min = np.round(np.log10((10**mstar_limit) / 2.041), 1)
    bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
    bin_num = 5
    bins = np.linspace(bin_min, bin_max, bin_num)


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

def get_stacked_velocity_dispersion(catl, galtype_col, id_col):
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

    catl = catl.loc[catl.logmstar >= mstar_limit]

    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    logmstar_col = 'logmstar'

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
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz_e16']
    red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()
    red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)
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
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_e16']

    blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()
    blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)
    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
        blue_cen_stellar_mass_arr

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

def get_velocity_dispersion(catl, galtype_col, id_col):
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
    catl = catl.loc[catl.logmstar >= mstar_limit]

    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    logmstar_col = 'logmstar'

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
    red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
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
    blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_data = dict_of_paths['data_dir']

ecodr2 = pd.read_csv(path_to_data + "raw/eco/ecodr2.csv")
ecodr2 = ecodr2.loc[ecodr2.name != 'ECO13860']

e16 = ecodr2[['name', 'grp_e16', 'fc_e16']]
e17 = ecodr2[['name', 'grp_e17', 'fc_e17']]

##############################*DR2:E16*###########################

ecodr2 = ecodr2.loc[(ecodr2.grpcz_e16.values >= 3000) & 
    (ecodr2.grpcz_e16.values <= 7000) & 
    (ecodr2.absrmag.values <= -17.33)]

volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
z_median = np.median(ecodr2.grpcz_e16.values) / (3 * 10**5)

ecodr2 = assign_colour_label_data(ecodr2)

total_data_e16 = measure_all_smf(ecodr2, volume, True)

f_blue_data_e16 = blue_frac(ecodr2, censat_col='fc_e16')

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(ecodr2, 
        galtype_col='fc_e16', id_col='grp_e16')

sigma_red_data_e16 = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data_e16 = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data_e16 = np.log10(sigma_red_data_e16[0])
sigma_blue_data_e16 = np.log10(sigma_blue_data_e16[0])

red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(ecodr2, 
        galtype_col='fc_e16', id_col='grp_e16')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data_e16 = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-2,3,5))
mean_mstar_blue_data_e16 = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))

ecodr2 = ecodr2.loc[(ecodr2.grpcz_e16.values >= 3000) & 
    (ecodr2.grpcz_e16.values <= 7000) & 
    (ecodr2.absrmag.values <= -17.33)]

##############################*DR2:E17*###########################

ecodr2 = pd.read_csv(path_to_data + "raw/eco/ecodr2.csv")
ecodr2 = ecodr2.loc[ecodr2.name != 'ECO13860']

ecodr2 = ecodr2.loc[(ecodr2.grpcz_e17.values >= 3000) & 
    (ecodr2.grpcz_e17.values <= 7000) & 
    (ecodr2.absrmag.values <= -17.33)]

volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
z_median = np.median(ecodr2.grpcz_e17.values) / (3 * 10**5)

ecodr2 = assign_colour_label_data(ecodr2)

total_data_e17 = measure_all_smf(ecodr2, volume, True)

f_blue_data_e17 = blue_frac(ecodr2, censat_col='fc_e17')

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(ecodr2, 
        galtype_col='fc_e17', id_col='grp_e17')

sigma_red_data_e17 = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data_e17 = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data_e17 = np.log10(sigma_red_data_e17[0])
sigma_blue_data_e17 = np.log10(sigma_blue_data_e17[0])

red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(ecodr2, 
        galtype_col='fc_e17', id_col='grp_e17')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data_e17 = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-2,3,5))
mean_mstar_blue_data_e17 = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))

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

fig1 = plt.figure()
dt_e16 = plt.scatter(total_data_e16[0], total_data_e16[1],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_e17 = plt.scatter(total_data_e17[0], total_data_e17[1],
    color='goldenrod', s=75, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(dt_e16), (dt_e17)], ['no pair splitting','pair splitting'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

fig2 = plt.figure()
dt_cen_e16 = plt.scatter(f_blue_data_e16[0], f_blue_data_e16[2],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_sat_e16 = plt.scatter(f_blue_data_e16[0], f_blue_data_e16[3],
    color='goldenrod', s=150, zorder=10, marker='^')

dt_cen_e17 = plt.scatter(f_blue_data_e17[0], f_blue_data_e17[2],
    color='rebeccapurple', s=50, zorder=15, marker='s')
dt_sat_e17 = plt.scatter(f_blue_data_e17[0], f_blue_data_e17[3],
    color='goldenrod', s=50, zorder=15, marker='s')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
plt.ylim(0,1)
# plt.title(r'Blue fractions from mocks and data')
plt.legend([dt_cen_e16, dt_sat_e16, dt_cen_e17, dt_sat_e17], 
    ['nps cen', 'nps sat', 'ps cen', 'ps sat'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
plt.minorticks_on()
plt.show()

fig3 = plt.figure()
bins_red=np.linspace(-2,3,5)
bins_blue=np.linspace(-1,3,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dt_red_e16 = plt.scatter(bins_red, mean_mstar_red_data_e16[0], 
    color='indianred', s=150, zorder=10, marker='^')
dt_blue_e16 = plt.scatter(bins_blue, mean_mstar_blue_data_e16[0],
    color='cornflowerblue', s=150, zorder=10, marker='^')

dt_red_e17 = plt.scatter(bins_red, mean_mstar_red_data_e17[0], 
    color='indianred', s=50, zorder=15, marker='s')
dt_blue_e17 = plt.scatter(bins_blue, mean_mstar_blue_data_e17[0],
    color='cornflowerblue', s=50, zorder=15, marker='s')

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
# plt.title(r'Velocity dispersion from mocks and data')
plt.legend([(dt_red_e16, dt_blue_e16), (dt_red_e17, dt_blue_e17)], 
    ['no pair splitting', 'pair splitting'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
plt.minorticks_on()
plt.show()

fig4 = plt.figure()
bin_min = 8.6
bins_red=np.linspace(bin_min,11,5)
bins_blue=np.linspace(bin_min,11,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dt_red_e16 = plt.scatter(bins_red, sigma_red_data_e16, 
    facecolors='none', edgecolors='indianred', s=150, 
    zorder=15, marker='^')
dt_blue_e16 = plt.scatter(bins_blue, sigma_blue_data_e16,
    facecolors='none', edgecolors='cornflowerblue', s=150,
    zorder=15, marker='^')

dt_red_e17 = plt.scatter(bins_red, sigma_red_data_e17, 
    color='indianred', s=150, 
    zorder=10, marker='s')
dt_blue_e17 = plt.scatter(bins_blue, sigma_blue_data_e17,
    color='cornflowerblue', s=150,
    zorder=10, marker='s')

plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
plt.legend([(dt_red_e16, dt_blue_e16), (dt_red_e17, dt_blue_e17)], 
    ['no pair splitting', 'pair splitting'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
plt.minorticks_on()
plt.show()

##############################*DR3 vs mcmc eco catalog*#########################
from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
import pandas as pd
import numpy as np
import os

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

def read_data_catl(path_to_file):
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

    eco_buff = reading_catls(path_to_file)
    #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
    eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

    eco_buff = mock_add_grpcz(eco_buff, grpid_col='groupid', 
        galtype_col='g_galtype', cen_cz_col='cz')
    
    # 6456 galaxies                       
    catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) & 
        (eco_buff.grpcz_new.values <= 7000) & 
        (eco_buff.absrmag.values <= -17.33)]

    volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    # cvar = 0.125
    z_median = np.median(catl.grpcz_new.values) / (3 * 10**5)
        
    return catl, volume, z_median

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

def get_velocity_dispersion(catl, colour_col):
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
    catl = catl.loc[catl.logmstar >= mstar_limit]

    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    logmstar_col = 'logmstar'
    ## Use group level for data even when settings.level == halo
    galtype_col = 'g_galtype'
    id_col = 'groupid'

    red_subset_ids = np.unique(catl[id_col].loc[(catl[colour_col] == 'R') 
        & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl[colour_col] == 'B') 
        & (catl[galtype_col] == 1)].values)

    red_subset_df = catl.loc[catl[id_col].isin(red_subset_ids)]
    #* Excluding N=1 groups
    red_subset_ids = red_subset_df.groupby([id_col]).filter\
        (lambda x: len(x) > 1)[id_col].unique()
    red_subset_df = catl.loc[catl[id_col].isin(
        red_subset_ids)].sort_values(by='{0}'.format(id_col))
    cen_red_subset_df = red_subset_df.loc[red_subset_df[galtype_col] == 1]
    red_cen_stellar_mass_arr = cen_red_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
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
    blue_cen_stellar_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmstar_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
        blue_cen_stellar_mass_arr

def get_stacked_velocity_dispersion(catl, colour_col):
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
    catl = catl.loc[catl.logmstar >= mstar_limit]

    catl.logmstar = np.log10((10**catl.logmstar) / 2.041)

    logmstar_col = 'logmstar'

    ## Use group level for data even when settings.level == halo
    galtype_col = 'g_galtype'
    id_col = 'groupid'

    red_subset_ids = np.unique(catl[id_col].loc[(catl[colour_col] == 'R') 
        & (catl[galtype_col] == 1)].values) 
    blue_subset_ids = np.unique(catl[id_col].loc[(catl[colour_col] == 'B') 
        & (catl[galtype_col] == 1)].values)

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

    red_cen_stellar_mass_arr = red_subset_df[logmstar_col].loc[red_subset_df[galtype_col] == 1]

    red_g_ngal_arr = red_subset_df.groupby([id_col]).size()

    red_cen_stellar_mass_arr = np.repeat(red_cen_stellar_mass_arr, red_g_ngal_arr)

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

    blue_cen_stellar_mass_arr = blue_subset_df[logmstar_col].loc[blue_subset_df[galtype_col] == 1]

    blue_g_ngal_arr = blue_subset_df.groupby([id_col]).size()

    blue_cen_stellar_mass_arr = np.repeat(blue_cen_stellar_mass_arr, blue_g_ngal_arr)

    blue_deltav_arr = np.hstack(blue_subset_df.groupby([id_col])['deltav'].apply(np.array).values)

    return red_deltav_arr, red_cen_stellar_mass_arr, blue_deltav_arr, \
        blue_cen_stellar_mass_arr

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

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']

eco_analysis = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"

catl, volume, z_median = read_data_catl(eco_analysis)

catl = assign_colour_label_data(catl)

# M* - sigma
red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 
        colour_col='colour_label')

sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data = np.log10(sigma_red_data[0])
sigma_blue_data = np.log10(sigma_blue_data[0])

# sigma - M*
red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(catl, 
        colour_col='colour_label')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-2,3,5))
mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))


ecodr3 = pd.read_csv(path_to_data + "external/ecodr3.csv", index_col=0)

new_colours = []
for galaxy in catl.name.values:
    ecodr3_entry = ecodr3.loc[ecodr3.name == galaxy]
    blue = ecodr3_entry.blue.values[0]
    red = ecodr3_entry.red.values[0]
    if blue == 1 and red ==0:
        new_colours.append('B')
    elif blue==0 and red==1:
        new_colours.append('R')
new_colours = np.array(new_colours)

catl["colour_label_new"] = new_colours

counter =0
for idx, entry in catl.iterrows():
    if entry.colour_label != entry.colour_label_new:
        counter += 1
#* 414 galaxies

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 
        colour_col='colour_label_new')

sigma_red_data_new = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))
sigma_blue_data_new = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.6,11,5))

sigma_red_data_new = np.log10(sigma_red_data_new[0])
sigma_blue_data_new = np.log10(sigma_blue_data_new[0])

# sigma - M*
red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(catl, 
        colour_col='colour_label_new')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data_new = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-2,3,5))
mean_mstar_blue_data_new = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))


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

fig1 = plt.figure()
bins_red=np.linspace(-2,3,5)
bins_blue=np.linspace(-1,3,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dt_red = plt.scatter(bins_red, mean_mstar_red_data[0], 
    color='indianred', s=150, zorder=10, marker='^')
dt_blue = plt.scatter(bins_blue, mean_mstar_blue_data[0],
    color='cornflowerblue', s=150, zorder=10, marker='^')

dt_red_new = plt.scatter(bins_red, mean_mstar_red_data_new[0], 
    facecolors='none', edgecolors='indianred', s=150, zorder=15, marker='s')
dt_blue_new = plt.scatter(bins_blue, mean_mstar_blue_data_new[0],
    facecolors='none', edgecolors='cornflowerblue', s=150, zorder=15, marker='s')

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
# plt.title(r'Velocity dispersion from mocks and data')
plt.legend([(dt_red, dt_blue), (dt_red_new, dt_blue_new)], 
    ['original', 'new colours'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
plt.minorticks_on()
plt.show()

fig2 = plt.figure()
bin_min = 8.6
bins_red=np.linspace(bin_min,11,5)
bins_blue=np.linspace(bin_min,11,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dt_red = plt.scatter(bins_red, sigma_red_data, 
    color='indianred', s=150, 
    zorder=10, marker='^')
dt_blue = plt.scatter(bins_blue, sigma_blue_data,
    color='cornflowerblue', s=150,
    zorder=10, marker='^')

dt_red_new = plt.scatter(bins_red, sigma_red_data_new, 
    facecolors='none', edgecolors='indianred', s=150, 
    zorder=15, marker='s')
dt_blue_new = plt.scatter(bins_blue, sigma_blue_data_new,
    facecolors='none', edgecolors='cornflowerblue', s=150, 
    zorder=15, marker='s')

plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=20)
plt.xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
plt.legend([(dt_red, dt_blue), (dt_red_new, dt_blue_new)], 
    ['original', 'new colours'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', prop={'size':20})
plt.minorticks_on()
plt.show()

#* Compare number of galaxies in bin
red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 
        colour_col='colour_label_new')

sigma_red_data_new_count = bs(red_cen_mstar_sigma, red_deltav,
    statistic='count', bins=np.linspace(8.6,11,5))
sigma_blue_data_new_count = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='count', bins=np.linspace(8.6,11,5))

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(catl, 
        colour_col='colour_label')

sigma_red_data_count = bs(red_cen_mstar_sigma, red_deltav,
    statistic='count', bins=np.linspace(8.6,11,5))
sigma_blue_data_count = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='count', bins=np.linspace(8.6,11,5))
