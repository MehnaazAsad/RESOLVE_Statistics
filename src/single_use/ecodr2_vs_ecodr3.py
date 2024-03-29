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

#* Using ECODR2 (Katie's pair splitting version and non-pair splitting version)
#* to compare all 3 main observables + M*-sigma
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_data = dict_of_paths['data_dir']

ecodr2 = pd.read_csv(path_to_data + "raw/eco/ecodr2.csv")
ecodr2 = ecodr2.loc[ecodr2.name != 'ECO13860']

e16 = ecodr2[['name', 'grp_e16', 'fc_e16']]
e17 = ecodr2[['name', 'grp_e17', 'fc_e17']]

#*DR2:E16
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

#*DR2:E17
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
#* Comparing velocity measurements using old colours and new colours
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

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', 
    fontsize=20)
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

plt.ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', 
    fontsize=20)
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

##############################################################
#* Comparing effect of different baryonic calculations on obs.

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

    bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
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

def blue_frac_helper(arr):
    total_num = len(arr)
    blue_counter = list(arr).count('B')
    return blue_counter/total_num

def blue_frac(catl, col):
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

    censat_col = 'g_galtype'

    mass_total_arr = catl[col]
    mass_cen_arr = catl[col].loc[catl[censat_col] == 1].values
    mass_sat_arr = catl[col].loc[catl[censat_col] == 0].values    

    colour_label_total_arr = catl.colour_label.values
    colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
    colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

    # changing from h=0.7 to h=1 assuming h^-2 dependence
    logmass_total_arr = np.log10((10**mass_total_arr) / 2.041)
    logmass_cen_arr = np.log10((10**mass_cen_arr) / 2.041)
    logmass_sat_arr = np.log10((10**mass_sat_arr) / 2.041)

    mbary_limit = 9.4
    bin_min = np.round(np.log10((10**mbary_limit) / 2.041), 1)
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

def get_velocity_dispersion(catl, logmbary_col):
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

    mbary_limit = 9.4
    catl = catl.loc[catl[logmbary_col] >= mbary_limit]

    catl[logmbary_col] = np.log10((10**catl[logmbary_col]) / 2.041)

    ## Use group level for data even when settings.level == halo
    galtype_col = 'g_galtype'
    id_col = 'ps_groupid'

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
    blue_cen_bary_mass_arr = cen_blue_subset_df.groupby(['{0}'.format(id_col),
        '{0}'.format(galtype_col)])[logmbary_col].apply(np.sum).values
    # blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz_av']
    blue_sigma_arr = blue_subset_df.groupby(['{0}'.format(id_col)])['cz'].\
        apply(lambda x: gapper(x)).values
    # blue_sigma_arr = gapper(blue_subset_df['deltav'])

    # blue_sigma_arr = blue_subset_df.groupby('{0}'.format(id_col))['cz'].apply(np.std, ddof=1).values

    return red_sigma_arr, red_cen_bary_mass_arr, blue_sigma_arr, \
        blue_cen_bary_mass_arr


logmstar = catl.logmstar.values
logmgas = catl.logmgas.values
logmgas_100 = catl.logmgas_a100.values

logmbary_mod = np.log10(10**logmstar + 10**logmgas)
logmbary_a100 = np.log10(10**logmstar + 10**logmgas_100)

catl["logmbary_mod"] = logmbary_mod
catl["logmbary_a100"] = logmbary_a100


bmf_logmbary = diff_bmf(catl.logmbary.values, volume, False)
bmf_logmbary_mod = diff_bmf(catl.logmbary_mod.values, volume, False)
bmf_logmbary_a100 = diff_bmf(catl.logmbary_a100.values, volume, False)


fig1 = plt.figure()

logmbary_or = plt.scatter(bmf_logmbary[0], bmf_logmbary[1],
    edgecolors='k', facecolors="None", s=150, zorder=10, marker='^')
logmbary_mod = plt.scatter(bmf_logmbary_mod[0], bmf_logmbary_mod[1],
    edgecolors='k', facecolors="None", s=150, zorder=10, marker='s')
logmbary_a100 = plt.scatter(bmf_logmbary_a100[0], bmf_logmbary_a100[1],
    edgecolors='k', facecolors="None", s=150, zorder=10, marker='*')

plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(logmbary_or), (logmbary_mod), (logmbary_a100)], 
    ['Original from catalog','Calc. using logmgas', 'Calc. using logmgas100'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', prop={'size':20})
plt.minorticks_on()
plt.show()


fblue_logmbary = blue_frac(catl, 'logmbary')
fblue_logmbary_mod = blue_frac(catl, 'logmbary_mod')
fblue_logmbary_a100 = blue_frac(catl, 'logmbary_a100')

fig2 = plt.figure()

logmbary_or_cen = plt.scatter(fblue_logmbary[0], fblue_logmbary[2],
    edgecolors='rebeccapurple', facecolors="None", s=150, zorder=10, marker='^')
logmbary_or_sat = plt.scatter(fblue_logmbary[0], fblue_logmbary[3],
    edgecolors='goldenrod', facecolors="None", s=150, zorder=10, marker='^')
logmbary_mod_cen = plt.scatter(fblue_logmbary_mod[0], fblue_logmbary_mod[2],
    edgecolors='rebeccapurple', facecolors="None", s=150, zorder=10, marker='s')
logmbary_mod_sat = plt.scatter(fblue_logmbary_mod[0], fblue_logmbary_mod[3],
    edgecolors='goldenrod', facecolors="None", s=150, zorder=10, marker='s')
logmbary_a100_cen = plt.scatter(fblue_logmbary_a100[0], fblue_logmbary_a100[2],
    edgecolors='rebeccapurple', facecolors="None", s=150, zorder=10, marker='*')
logmbary_a100_sat = plt.scatter(fblue_logmbary_a100[0], fblue_logmbary_a100[3],
    edgecolors='goldenrod', facecolors="None", s=150, zorder=10, marker='*')

plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
plt.ylim(0,1)
plt.legend([logmbary_or_cen, logmbary_or_sat, logmbary_mod_cen, logmbary_mod_sat,\
    logmbary_a100_cen, logmbary_a100_sat], 
    ['Original cen', 'Original sat', 'Logmgas cen', 'Logmgas sat', 
    'Logmgas100 cen', 'Logmgas100 sat'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper right', prop={'size':17})
plt.minorticks_on()
plt.show()


red_sigma, red_cen_mbary_sigma, blue_sigma, \
    blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'logmbary')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

logmbary_mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
logmbary_mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))

red_sigma, red_cen_mbary_sigma, blue_sigma, \
    blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'logmbary_mod')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

logmbary_mod_mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
logmbary_mod_mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))

red_sigma, red_cen_mbary_sigma, blue_sigma, \
    blue_cen_mbary_sigma = get_velocity_dispersion(catl, 'logmbary_a100')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

logmbary_a100_mean_mstar_red_data = bs(red_sigma, red_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
logmbary_a100_mean_mstar_blue_data = bs(blue_sigma, blue_cen_mbary_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))

fig3 = plt.figure()

bins_red = logmbary_mean_mstar_red_data[1]
bins_blue = logmbary_mean_mstar_blue_data[1]
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

logmbary_red = plt.scatter(bins_red, logmbary_mean_mstar_red_data[0], 
    edgecolors='indianred', facecolors="None", s=150, zorder=10, marker='^')
logmbary_blue = plt.scatter(bins_blue, logmbary_mean_mstar_blue_data[0],
    edgecolors='cornflowerblue', facecolors="None", s=150, zorder=10, marker='^')

logmbary_mod_red = plt.scatter(bins_red, logmbary_mod_mean_mstar_red_data[0], 
    edgecolors='indianred', facecolors="None", s=150, zorder=10, marker='s')
logmbary_mod_blue = plt.scatter(bins_blue, logmbary_mod_mean_mstar_blue_data[0],
    edgecolors='cornflowerblue', facecolors="None", s=150, zorder=10, marker='s')

logmbary_a100_red = plt.scatter(bins_red, logmbary_a100_mean_mstar_red_data[0], 
    edgecolors='indianred', facecolors="None", s=150, zorder=10, marker='*')
logmbary_a100_blue = plt.scatter(bins_blue, logmbary_a100_mean_mstar_blue_data[0],
    edgecolors='cornflowerblue', facecolors="None", s=150, zorder=10, marker='*')

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{b, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)
plt.legend([(logmbary_red, logmbary_blue), (logmbary_mod_red, logmbary_mod_blue),
    (logmbary_a100_red, logmbary_a100_blue)], 
    ['Original','Calc. using logmgas', 'Calc. using logmgas100'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='upper left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

################################################################################
#* Compare pair-splitting DR3 (my version) smf vs non-pair-splitting DR2 smf
#* (my version)

from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import pandas as pd
import numpy as np
import random
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

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

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
    # changing from h=0.7 to h=1 assuming h^-2 dependence
    logmbary_arr = np.log10((10**mass_arr) / 2.041)

    bin_min = np.round(np.log10((10**9.4) / 2.041), 1)
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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']

dr3_filename = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
dr2_filename = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

dr2 = reading_catls(dr2_filename)
dr3 = reading_catls(dr3_filename)

dr2 = mock_add_grpcz(dr2, grpid_col='groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')
dr3 = mock_add_grpcz(dr3, grpid_col='ps_groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')

dr2 = dr2.loc[(dr2.grpcz_cen.values >= 3000) & 
    (dr2.grpcz_cen.values <= 7000) & 
    (dr2.absrmag.values <= -17.33)]
dr3 = dr3.loc[(dr3.grpcz_cen.values >= 3000) & 
    (dr3.grpcz_cen.values <= 7000) & 
    (dr3.absrmag.values <= -17.33)]

volume = 151829.26 # Survey volume without buffer [Mpc/h]^3

dr2_smf = diff_smf(dr2.logmstar.values, volume, False)
dr3_smf = diff_smf(dr3.logmstar.values, volume, False)

fig1 = plt.figure()
dt_dr2 = plt.scatter(dr2_smf[0], dr2_smf[1],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_dr3 = plt.scatter(dr3_smf[0], dr3_smf[1],
    color='goldenrod', s=75, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(dt_dr2), (dt_dr3)], ['no pair splitting','pair splitting'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

#* Compare pair-splitting DR3 smf (my implementation) vs pair-splitting DR2 smf
#* (katie's implementation)

ecodr2_or = pd.read_csv(path_to_data + "raw/eco/ecodr2.csv")
ecodr2_or = ecodr2_or.loc[ecodr2_or.name != 'ECO13860']

ecodr2_or = ecodr2_or.loc[(ecodr2_or.grpcz_e17.values >= 3000) & 
    (ecodr2_or.grpcz_e17.values <= 7000) & 
    (ecodr2_or.absrmag.values <= -17.33)]

e17_smf = diff_smf(ecodr2_or.logmstar.values, volume, False)

fig1 = plt.figure()
dt_dr2_or = plt.scatter(e17_smf[0], e17_smf[1],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_dr3 = plt.scatter(dr3_smf[0], dr3_smf[1],
    color='goldenrod', s=75, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(dt_dr2_or), (dt_dr3)], ['original pair splitting','my pair splitting'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

#* Compare DR3 smf/bmf vs mocks smf/bmf

mock_name = 'ECO'
num_mocks = 8
min_cz = 3000
max_cz = 7000
mag_limit = -17.33
mstar_limit = 8.9
volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
mf_type = 'bmf'

phi_total_arr = []
box_id_arr = np.linspace(5001,5008,8)
for box in box_id_arr:
    box = int(box)
    temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
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
    #     break
    # break
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

phi_arr_total = np.array(phi_total_arr)

fig1 = plt.figure()
for i in range(len(phi_arr_total)):
    mocks = plt.scatter(dr3_smf[0], phi_arr_total[i],
        color='rebeccapurple', s=75, zorder=10, marker='^')
dt_dr3 = plt.scatter(dr3_smf[0], dr3_smf[1],
    color='goldenrod', s=150, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(mocks), (dt_dr3)], ['mocks (ps)','dr3 (ps)'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()


dr3_bmf = diff_bmf(dr3.logmbary.values, volume, False)

logmbary_calc_arr = calc_bary(dr3.logmstar.values, dr3.logmgas_a100.values)
dr3_bmf_calc = diff_bmf(logmbary_calc_arr, volume, False)

fig1 = plt.figure()
for i in range(len(phi_arr_total)):
    mocks = plt.scatter(dr3_bmf_calc[0], phi_arr_total[i],
        color='rebeccapurple', s=75, zorder=10, marker='^')
# dt_dr3 = plt.scatter(dr3_bmf[0], dr3_bmf[1],
#     color='goldenrod', s=150, zorder=15, marker='^')
dt_dr3 = plt.scatter(dr3_bmf_calc[0], dr3_bmf_calc[1],
    color='goldenrod', s=150, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_b \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(mocks), (dt_dr3)], ['mocks (ps)','dr3 (ps)'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

################################################################################
#* Comparing best-fit DR3 smf chi-squared vs best-fit DR2 smf chi-squared

from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import binned_statistic as bs
import emcee
import pandas as pd
import numpy as np
import os
import multiprocessing
import subprocess
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
import random

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

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

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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
    
    catl['colour_label_old'] = colour_label_arr

    return catl

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

def get_velocity_dispersion(catl, catl_type, color_type=None, randint=None):
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
        if mf_type == 'smf':
            catl = catl.loc[catl.logmstar >= np.log10((10**mstar_limit)/2.041)]
        elif mf_type == 'bmf':
            mhi_arr = catl.mhi.values
            logmgas_arr = np.log10((1.4 * mhi_arr) / 2.041)
            logmbary_arr = calc_bary(catl.logmstar.values, logmgas_arr)
            catl["logmbary"] = logmbary_arr
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

        # catl = models_add_avgrpcz(catl, id_col, galtype_col)

    if color_type == 'old':
        print("Using old colour info")
        red_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label_old == 'R') & (catl[galtype_col] == 1)].values) 
        blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label_old == 'B') & (catl[galtype_col] == 1)].values)
    else:
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

def get_err(pair_splitting):
    mock_name = 'ECO'
    num_mocks = 8
    min_cz = 3000
    max_cz = 7000
    mag_limit = -17.33
    mstar_limit = 8.9
    volume = 151829.26 # Survey volume without buffer [Mpc/h]^3

    phi_total_arr = []
    red_sigma_new_arr = []
    blue_sigma_new_arr = []
    red_sigma_old_arr = []
    blue_sigma_old_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path_to_mocks + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            print('Box {0} : Mock {1}'.format(box, num))
            mock_pd = reading_catls(filename) 
            
            if pair_splitting:
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
            else:
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

            ## Using best-fit found for new ECO data using result from chain 67
            ## i.e. hybrid quenching model
            bf_from_last_chain = [10.1942986, 14.5454828, 0.708013630,
                0.00722556715]

            Mstar_q = bf_from_last_chain[0] # Msun/h**2
            Mh_q = bf_from_last_chain[1] # Msun/h
            mu = bf_from_last_chain[2]
            nu = bf_from_last_chain[3]

            if quenching == 'hybrid':
                theta = [Mstar_q, Mh_q, mu, nu]
                f_red_c, f_red_s = hybrid_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            elif quenching == 'halo':
                theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                f_red_c, f_red_s = halo_quenching_model(theta, mock_pd, 
                    'nonvishnu')
            mock_pd = assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

            red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma = get_velocity_dispersion(
                mock_pd, 'mock')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            mean_mstar_red_new = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))
            mean_mstar_blue_new = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(1,3,5))

            mean_mstar_red_old = bs(red_sigma, red_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1.4,3,5))
            mean_mstar_blue_old = bs(blue_sigma, blue_cen_mstar_sigma, 
                statistic=average_of_log, bins=np.linspace(-1,3,5))

            red_sigma_new_arr.append(mean_mstar_red_new[0])
            blue_sigma_new_arr.append(mean_mstar_blue_new[0])
            red_sigma_old_arr.append(mean_mstar_red_old[0])
            blue_sigma_old_arr.append(mean_mstar_blue_old[0])

    phi_arr_total = np.array(phi_total_arr)
    red_sigma_new_arr = np.array(red_sigma_new_arr)
    blue_sigma_new_arr = np.array(blue_sigma_new_arr)
    red_sigma_old_arr = np.array(red_sigma_old_arr)
    blue_sigma_old_arr = np.array(blue_sigma_old_arr)

    return phi_arr_total, [red_sigma_new_arr, blue_sigma_new_arr], [red_sigma_old_arr, blue_sigma_old_arr]

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

def get_bestfit(theta, pair_splitting):
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

    if pair_splitting:
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
            (gal_group_df['cen_cz'] >= cz_inner_mod) &
            (gal_group_df['cen_cz'] <= cz_outer)].reset_index(drop=True)
    else:
        gal_group_df = models_add_cengrpcz(gal_group_df, grpid_col='groupid', 
            galtype_col='grp_censat', cen_cz_col='cz')

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

    ## Observable #1 - Total SMF
    smf_model = diff_smf(10**(gal_group_df.logmstar.values), survey_vol, True) 

    # red_sigma, red_cen_mstar_sigma, blue_sigma, \
    # blue_cen_mstar_sigma = get_velocity_dispersion(
    #     gal_group_df, 'model')

    # red_sigma = np.log10(red_sigma)
    # blue_sigma = np.log10(blue_sigma)

    # mean_mstar_red_new = bs(red_sigma, red_cen_mstar_sigma, 
    #     statistic=average_of_log, bins=np.linspace(1,3,5))
    # mean_mstar_blue_new = bs(blue_sigma, blue_cen_mstar_sigma, 
    #     statistic=average_of_log, bins=np.linspace(1,3,5))

    # mean_mstar_red_old = bs(red_sigma, red_cen_mstar_sigma, 
    #     statistic=average_of_log, bins=np.linspace(-1.4,3,5))
    # mean_mstar_blue_old = bs(blue_sigma, blue_cen_mstar_sigma, 
    #     statistic=average_of_log, bins=np.linspace(-1,3,5))

    return smf_model#, [mean_mstar_red_new, mean_mstar_blue_new], [mean_mstar_red_old, mean_mstar_blue_old]

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
global quenching
global survey
global mf_type 

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']
path_to_data = dict_of_paths['data_dir']
path_to_mocks = path_to_data + 'mocks/m200b/eco/'

halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
            'vishnu/rockstar/vishnu_rockstar_test.hdf5'

quenching = 'hybrid'
survey = 'eco'
mf_type = 'smf'
level = 'group'

#*DR2
#*data (No pair splitting)
dr2_filename = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
dr2 = reading_catls(dr2_filename)
dr2 = mock_add_grpcz(dr2, grpid_col='groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')
dr2 = dr2.loc[(dr2.grpcz_cen.values >= 3000) & 
    (dr2.grpcz_cen.values <= 7000) & 
    (dr2.absrmag.values <= -17.33)]
volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
z_median = np.median(dr2.grpcz_cen.values) / (3 * 10**5)
dr2_smf = diff_smf(dr2.logmstar.values, volume, False)

#*model (No pair splitting)
file=path_to_proc + 'smhm_colour_run61/chain.h5'
reader = emcee.backends.HDFBackend(file , read_only=True)
chi2 = reader.get_blobs(flat=True)
flatchain = reader.get_chain(flat=True)
colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
    'mstar_q','mh_q','mu','nu']
emcee_table = pd.DataFrame(flatchain, columns=colnames)
emcee_table['chi2'] = chi2

bf_params_dr2 = emcee_table.loc[emcee_table.chi2 == min(emcee_table.chi2)].\
    iloc[0].values[:9]

model_init = halocat_init(halo_catalog, z_median)
dr2_bf = get_bestfit(bf_params_dr2, pair_splitting=False)
dr2_bf_smf = dr2_bf[0]

#*error (No pair splitting)
dr2_mocks = get_err(pair_splitting=False)
dr2_mocks_smf = dr2_mocks[0]
phi_total_0 = dr2_mocks_smf[:,0]
phi_total_1 = dr2_mocks_smf[:,1]
phi_total_2 = dr2_mocks_smf[:,2]
phi_total_3 = dr2_mocks_smf[:,3]
combined_df = pd.DataFrame({
    'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
    'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,})
corr_mat_colour = combined_df.corr()
corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
dr2_err = np.sqrt(np.diag(combined_df.cov()))

dr2_chisquared = chi_squared(dr2_smf[1], dr2_bf_smf[1], dr2_err, 
    corr_mat_inv_colour)

#*DR3
#*data (pair splitting)
dr3_filename = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
dr3 = reading_catls(dr3_filename)
dr3 = mock_add_grpcz(dr3, grpid_col='ps_groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')
#### Testing non-pair splitting
# most_massive_gal_idxs = dr3.groupby(['groupid'])['logmstar']\
#     .transform(max) == dr3['logmstar']        
# grp_censat_new = most_massive_gal_idxs.astype(int)
# dr3["groupid_censat"] = grp_censat_new
# dr3 = mock_add_grpcz(dr3, grpid_col='groupid', 
#     galtype_col='groupid_censat', cen_cz_col='cz')
####
dr3 = dr3.loc[(dr3.grpcz_cen.values >= 3000) & 
    (dr3.grpcz_cen.values <= 7000) & 
    (dr3.absrmag.values <= -17.33)]
volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
z_median = np.median(dr3.grpcz_cen.values) / (3 * 10**5)
dr3_smf = diff_smf(dr3.logmstar.values, volume, False)

#*model (pair splitting)
file=path_to_proc + 'smhm_colour_run67/chain.h5'
reader = emcee.backends.HDFBackend(file , read_only=True)
chi2 = reader.get_blobs(flat=True)
flatchain = reader.get_chain(flat=True)
colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
    'mstar_q','mh_q','mu','nu']
emcee_table = pd.DataFrame(flatchain, columns=colnames)
emcee_table['chi2'] = chi2

bf_params_dr3 = emcee_table.loc[emcee_table.chi2 == min(emcee_table.chi2)].\
    iloc[0].values[:9]
dr3_bf = get_bestfit(bf_params_dr3, pair_splitting=True)
dr3_bf_smf = dr3_bf[0]

#*error (pair splitting)
dr3_mocks = get_err(pair_splitting=True)
dr3_mocks_smf = dr3_mocks[0]
phi_total_0 = dr3_mocks_smf[:,0]
phi_total_1 = dr3_mocks_smf[:,1]
phi_total_2 = dr3_mocks_smf[:,2]
phi_total_3 = dr3_mocks_smf[:,3]
combined_df = pd.DataFrame({
    'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
    'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,})

corr_mat_colour = combined_df.corr()
corr_mat_inv_colour = np.linalg.inv(corr_mat_colour.values)  
dr3_err = np.sqrt(np.diag(combined_df.cov()))

dr3_chisquared = chi_squared(dr3_smf[1], dr3_bf_smf[1], dr3_err, 
    corr_mat_inv_colour)

#* BF DR3 sigma-M* old binning chi-squared vs new binning chi-squared

#*data - new
dr3 = assign_colour_label_data(dr3)

red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(
        dr3, 'data')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

dr3_mean_mstar_red_new = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
dr3_mean_mstar_blue_new = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
data_new = []
data_new.append(dr3_mean_mstar_red_new[0])
data_new.append(dr3_mean_mstar_blue_new[0])
data_new = np.array(data_new)

#*error - new
dr3_mocks = get_err(pair_splitting=True)

mstar_red_cen_0 = dr3_mocks[1][0][:,0]
mstar_red_cen_1 = dr3_mocks[1][0][:,1]
mstar_red_cen_2 = dr3_mocks[1][0][:,2]
mstar_red_cen_3 = dr3_mocks[1][0][:,3]

mstar_blue_cen_0 = dr3_mocks[1][1][:,0]
mstar_blue_cen_1 = dr3_mocks[1][1][:,1]
mstar_blue_cen_2 = dr3_mocks[1][1][:,2]
mstar_blue_cen_3 = dr3_mocks[1][1][:,3]

combined_df = pd.DataFrame({
    'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
    'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
    'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
    'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

corr_mat_colour = combined_df.corr()
corr_mat_inv_colour_new = np.linalg.inv(corr_mat_colour.values)  
dr3_new_err = np.sqrt(np.diag(combined_df.cov()))

#*model - new
dr3_bf = get_bestfit(bf_params_dr3, pair_splitting=True)
dr3_bf_vdisp_new = dr3_bf[1]
model_new = []
model_new.append(dr3_bf_vdisp_new[0][0]) #red
model_new.append(dr3_bf_vdisp_new[1][0]) #blue
model_new = np.array(model_new)

dr3_chisquared_new = chi_squared(data_new, model_new, dr3_new_err, 
    corr_mat_inv_colour_new)

#*error - old binning (mod)
mstar_red_cen_0 = dr3_mocks[2][0][:,0]
mstar_red_cen_1 = dr3_mocks[2][0][:,1]
mstar_red_cen_2 = dr3_mocks[2][0][:,2]
mstar_red_cen_3 = dr3_mocks[2][0][:,3]

mstar_blue_cen_0 = dr3_mocks[2][1][:,0]
mstar_blue_cen_1 = dr3_mocks[2][1][:,1]
mstar_blue_cen_2 = dr3_mocks[2][1][:,2]
mstar_blue_cen_3 = dr3_mocks[2][1][:,3]

combined_df = pd.DataFrame({
    'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
    'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
    'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
    'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})

corr_mat_colour = combined_df.corr()
corr_mat_inv_colour_old = np.linalg.inv(corr_mat_colour.values)  
dr3_old_err = np.sqrt(np.diag(combined_df.cov()))

#*model - old binning (mod)
dr3_bf_vdisp_old = dr3_bf[2]
model_old = []
model_old.append(dr3_bf_vdisp_old[0][0]) #red
model_old.append(dr3_bf_vdisp_old[1][0]) #blue
model_old = np.array(model_old)

#*data - old binning (mod)
# dr3 = assign_colour_label_data_legacy(dr3)

# red_sigma, red_cen_mstar_sigma, blue_sigma, \
#     blue_cen_mstar_sigma = get_velocity_dispersion(
#         dr3, 'data', 'old')

dr3_mean_mstar_red_old = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1.4,3,5))
dr3_mean_mstar_blue_old = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))
data_old = []
data_old.append(dr3_mean_mstar_red_old[0])
data_old.append(dr3_mean_mstar_blue_old[0])
data_old = np.array(data_old)

dr3_chisquared_old = chi_squared(data_old, model_old, dr3_old_err, 
    corr_mat_inv_colour_old)

#####
# Trying binning test on DR2 instead
dr2_filename = path_to_proc + "gal_group_eco_data_buffer_volh1_dr2.hdf5"
dr2 = reading_catls(dr2_filename)
dr2 = mock_add_grpcz(dr2, grpid_col='groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')
dr2 = dr2.loc[(dr2.grpcz_cen.values >= 3000) & 
    (dr2.grpcz_cen.values <= 7000) & 
    (dr2.absrmag.values <= -17.33)]
volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
z_median = np.median(dr2.grpcz_cen.values) / (3 * 10**5)
dr2_smf = diff_smf(dr2.logmstar.values, volume, False)
dr2 = assign_colour_label_data_legacy(dr2)
red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(
        dr2, 'data', 'old')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

dr2_mean_mstar_red_old = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-2,3,5))
dr2_mean_mstar_blue_old = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(-1,3,5))
dr2_data_old = []
dr2_data_old.append(dr2_mean_mstar_red_old[0])
dr2_data_old.append(dr2_mean_mstar_blue_old[0])
dr2_data_old = np.array(dr2_data_old)

dr2_mean_mstar_red_new = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
dr2_mean_mstar_blue_new = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
dr2_data_new = []
dr2_data_new.append(dr2_mean_mstar_red_new[0])
dr2_data_new.append(dr2_mean_mstar_blue_new[0])
dr2_data_new = np.array(dr2_data_new)

###############################################################################
#* Compare observables between DR2 and DR3

from cosmo_utils.utils import work_paths as cwpaths
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binned_statistic as bs
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import pandas as pd
import numpy as np
import random
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

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

def average_of_log(arr):
    result = np.log10(np.mean(np.power(10, arr)))
    return result

def assign_colour_label_dr2(catl):
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
    u_r_arr = catl.modelu_r.values

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

def assign_colour_label_dr3(catl):
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


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_proc = dict_of_paths['proc_dir']

dr3_filename = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
dr2_filename = path_to_proc + "gal_group_eco_stellar_buffer_volh1_dr2_ps.hdf5"

dr2 = reading_catls(dr2_filename)
dr3 = reading_catls(dr3_filename)

dr2 = mock_add_grpcz(dr2, grpid_col='ps_groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')
dr3 = mock_add_grpcz(dr3, grpid_col='ps_groupid', 
    galtype_col='g_galtype', cen_cz_col='cz')

dr2 = dr2.loc[(dr2.grpcz_cen.values >= 3000) & 
    (dr2.grpcz_cen.values <= 7000) & 
    (dr2.absrmag.values <= -17.33)]
dr3 = dr3.loc[(dr3.grpcz_cen.values >= 3000) & 
    (dr3.grpcz_cen.values <= 7000) & 
    (dr3.absrmag.values <= -17.33)]

volume = 151829.26 # Survey volume without buffer [Mpc/h]^3

dr2_smf = diff_smf(dr2.logmstar.values, volume, False)
dr3_smf = diff_smf(dr3.logmstar.values, volume, False)

fig1 = plt.figure()
dt_dr2 = plt.scatter(dr2_smf[0], dr2_smf[1],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_dr3 = plt.scatter(dr3_smf[0], dr3_smf[1],
    color='goldenrod', s=75, zorder=15, marker='^')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dlogM}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)

plt.legend([(dt_dr2), (dt_dr3)], ['dr2','dr3'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()

dr2 = assign_colour_label_dr2(dr2)
dr3 = assign_colour_label_dr3(dr3)

f_blue_dr2 = blue_frac(dr2, censat_col='g_galtype')
f_blue_dr3 = blue_frac(dr3, censat_col='g_galtype')

fig2 = plt.figure()
dt_cen_dr2 = plt.scatter(f_blue_dr2[0], f_blue_dr2[2],
    color='rebeccapurple', s=150, zorder=10, marker='^')
dt_cen_dr3 = plt.scatter(f_blue_dr3[0], f_blue_dr3[2],
    color='rebeccapurple', s=150, zorder=15, marker='*')

dt_sat_dr2 = plt.scatter(f_blue_dr2[0], f_blue_dr2[3],
    color='goldenrod', s=150, zorder=10, marker='^')
dt_sat_dr3 = plt.scatter(f_blue_dr3[0], f_blue_dr3[3],
    color='goldenrod', s=150, zorder=15, marker='*')

plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$f_{blue}$', fontsize=20)
plt.ylim(0,1)
plt.legend([dt_cen_dr2, dt_cen_dr3, dt_sat_dr2, dt_sat_dr3], 
    ['dr2 cen','dr3 cen', 'dr2 sat', 'dr3 sat'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower left', 
    prop={'size':20})
plt.minorticks_on()
plt.show()


red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(dr2, 
        galtype_col='g_galtype', id_col='ps_groupid')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_dr2 = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.8,5))
mean_mstar_blue_dr2 = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.5,5))

red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(dr3, 
        galtype_col='g_galtype', id_col='ps_groupid')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_dr3 = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.8,5))
mean_mstar_blue_dr3 = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,2.5,5))

fig3 = plt.figure()

bins_red=np.linspace(1,2.8,5)
bins_blue=np.linspace(1,2.5,5)
bins_red = 0.5 * (bins_red[1:] + bins_red[:-1])
bins_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

dt_red_dr2 = plt.scatter(bins_red, mean_mstar_red_dr2[0],
    color='indianred', s=200, zorder=10, marker='^')
dt_red_dr3 = plt.scatter(bins_red, mean_mstar_red_dr3[0],
    color='indianred', s=200, zorder=15, marker='*')

dt_blue_dr2 = plt.scatter(bins_blue, mean_mstar_blue_dr2[0],
    color='cornflowerblue', s=200, zorder=10, marker='^')
dt_blue_dr3 = plt.scatter(bins_blue, mean_mstar_blue_dr3[0],
    color='cornflowerblue', s=200, zorder=15, marker='*')

plt.xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=20)
plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-2} \right]$',fontsize=20)

plt.legend([dt_red_dr2, dt_red_dr3, dt_blue_dr2, dt_blue_dr3], 
    ['dr2', 'dr3', 'dr2', 'dr3'],
    handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='lower right', 
    prop={'size':20})
plt.minorticks_on()
plt.show()


#* Galaxies in common between DR2 and DR3
common_gals = dr2.merge(dr3['name']).name.values

dr2_subset = dr2[dr2['name'].isin(common_gals)].reset_index(drop=True)
red_dr2 = dr2_subset.loc[dr2_subset.colour_label == 'R']
blue_dr2 = dr2_subset.loc[dr2_subset.colour_label == 'B']

dr3_subset = dr3[dr3['name'].isin(common_gals)].reset_index(drop=True)
red_dr3 = dr3_subset.loc[dr3_subset.colour_label == 'R']
blue_dr3 = dr3_subset.loc[dr3_subset.colour_label == 'B']

#213 galaxies that are labeled as group central in one dr and group satellite in
#other dr. So, if we want to make matrix just for group centrals, we have to exclude
#these.
galtype_mismatch_idxs = np.where(dr2_subset.g_galtype.values 
    != dr3_subset.g_galtype.values)
dr2_subset = dr2_subset[~dr2_subset.index.isin(galtype_mismatch_idxs[0])]
dr3_subset = dr3_subset[~dr3_subset.index.isin(galtype_mismatch_idxs[0])]

dr2_dr3_colour_labels = dr2_subset[['name', 'colour_label', 'g_galtype']].copy()
dr2_dr3_colour_labels['colour_label_dr3'] = dr3_subset['colour_label']
dr2_dr3_colour_labels.rename(columns={"colour_label": "colour_label_dr2"}, inplace=True)

red_dr2_red_dr3 = dr2_dr3_colour_labels.loc[(dr2_dr3_colour_labels.colour_label_dr2 == 'R') & 
    (dr2_dr3_colour_labels.colour_label_dr3 == 'R') & (dr2_dr3_colour_labels.g_galtype == 1)]
red_dr2_blue_dr3 = dr2_dr3_colour_labels.loc[(dr2_dr3_colour_labels.colour_label_dr2 == 'R') & 
    (dr2_dr3_colour_labels.colour_label_dr3 == 'B') & (dr2_dr3_colour_labels.g_galtype == 1)]

blue_dr2_blue_dr3 = dr2_dr3_colour_labels.loc[(dr2_dr3_colour_labels.colour_label_dr2 == 'B') & 
    (dr2_dr3_colour_labels.colour_label_dr3 == 'B') & (dr2_dr3_colour_labels.g_galtype == 1)]
blue_dr2_red_dr3 = dr2_dr3_colour_labels.loc[(dr2_dr3_colour_labels.colour_label_dr2 == 'B') & 
    (dr2_dr3_colour_labels.colour_label_dr3 == 'R') & (dr2_dr3_colour_labels.g_galtype == 1)]

print(red_dr2_red_dr3.shape[0])
print(red_dr2_blue_dr3.shape[0])
print(blue_dr2_blue_dr3.shape[0])
print(blue_dr2_red_dr3.shape[0])


confusion_matrix = np.zeros((2,2))
confusion_matrix[0][0] = red_dr2_red_dr3.shape[0]
confusion_matrix[0][1] = blue_dr2_red_dr3.shape[0]
confusion_matrix[1][0] = red_dr2_blue_dr3.shape[0]
confusion_matrix[1][1] = blue_dr2_blue_dr3.shape[0]

# from sklearn import metrics
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Red", "Blue"])
n_grpcen = len(dr2_dr3_colour_labels.loc[dr2_dr3_colour_labels.g_galtype == 1])
labels = ['Red', 'Blue']
fig, ax = plt.subplots()
#Show percentage instead
confusion_matrix_percent = np.round(((confusion_matrix/n_grpcen)*100), 2)
# ax.matshow(confusion_matrix_percent)
ax.matshow(confusion_matrix)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
plt.xlabel('DR2')
plt.ylabel('DR3')
for (i, j), z in np.ndenumerate(confusion_matrix):
    ax.text(j, i, '{0} ({1}\%)'.format(int(z), confusion_matrix_percent[i][j]), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))# cm_display.plot()
plt.title('Confusion matrix for {0} group centrals in common between DR2 and DR3'.format(n_grpcen))
plt.show()