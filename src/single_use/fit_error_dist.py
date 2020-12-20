"""
{This script calculates spread in velocity dispersion (sigma) from mocks for 
 red and blue galaxies as well as smf for red and blue galaxies. It then 
 finds a non-gaussian distribution that best fits the error in spread 
 distributions in each bin.}
"""

from cosmo_utils.utils import work_paths as cwpaths
from scipy.stats import normaltest as nt
from chainconsumer import ChainConsumer
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import chi2 
from matplotlib import rc
from scipy import stats
import pandas as pd
import numpy as np
import emcee
import math
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

def std_func(bins, mass_arr, vel_arr):

    last_index = len(bins)-1
    i = 0
    std_arr = []
    for index1, bin_edge in enumerate(bins):
        cen_deltav_arr = []
        if index1 == last_index:
            break
        for index2, stellar_mass in enumerate(mass_arr):
            if stellar_mass >= bin_edge and stellar_mass < bins[index1+1]:
                cen_deltav_arr.append(vel_arr[index2])
        N = len(cen_deltav_arr)
        mean = 0
        diff_sqrd_arr = []
        for value in cen_deltav_arr:
            diff = value - mean
            diff_sqrd = diff**2
            diff_sqrd_arr.append(diff_sqrd)
        mean_diff_sqrd = np.mean(diff_sqrd_arr)
        std = np.sqrt(mean_diff_sqrd)
        std_arr.append(std)

    return std_arr

def get_deltav_sigma_mocks(survey, path):
    """
    Calculate spread in velocity dispersion from survey mocks

    Parameters
    ----------
    survey: string
        Name of survey
    path: string
        Path to mock catalogs

    Returns
    ---------
    std_red_arr: numpy array
        Spread in velocity dispersion of red galaxies
    centers_red_arr: numpy array
        Bin centers of central stellar mass for red galaxies
    std_blue_arr: numpy array
        Spread in velocity dispersion of blue galaxies
    centers_blue_arr: numpy array
        Bin centers of central stellar mass for blue galaxies
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

    std_red_arr = []
    centers_red_arr = []
    std_blue_arr = []
    centers_blue_arr = []
    box_id_arr = np.linspace(5001,5008,8)
    for box in box_id_arr:
        box = int(box)
        temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
            mock_name) 
        for num in range(num_mocks):
            filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                mock_name, num)
            mock_pd = reading_catls(filename) 

            # Using the same survey definition as in mcmc smf i.e excluding the 
            # buffer
            mock_pd = mock_pd.loc[(mock_pd.cz.values >= min_cz) & \
                (mock_pd.cz.values <= max_cz) & \
                (mock_pd.M_r.values <= mag_limit) & \
                (mock_pd.logmstar.values >= mstar_limit)]

            logmstar_arr = mock_pd.logmstar.values 
            u_r_arr = mock_pd.u_r.values

            colour_label_arr = np.empty(len(mock_pd), dtype='str')
            # Using defintions from Moffett paper
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
            mock_pd.logmstar = np.log10((10**mock_pd.logmstar) / 2.041)
            red_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
                colour_label == 'R') & (mock_pd.g_galtype == 1)].values)  
            blue_subset_grpids = np.unique(mock_pd.groupid.loc[(mock_pd.\
                colour_label == 'B') & (mock_pd.g_galtype == 1)].values)

            # Calculating spread in velocity dispersion for galaxies in groups
            # with a red central

            red_deltav_arr = []
            red_cen_stellar_mass_arr = []
            for key in red_subset_grpids: 
                group = mock_pd.loc[mock_pd.groupid == key]
                cen_stellar_mass = group.logmstar.loc[group.g_galtype.\
                    values == 1].values[0]
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                for val in deltav:
                    red_deltav_arr.append(val)
                    red_cen_stellar_mass_arr.append(cen_stellar_mass)
            # print(max(red_cen_stellar_mass_arr))

            if survey == 'eco' or survey == 'resolvea':
                # TODO : check if this is actually correct for resolve a
                red_stellar_mass_bins = np.linspace(8.6,11.5,6)
            elif survey == 'resolveb':
                red_stellar_mass_bins = np.linspace(8.4,11.0,6)
            std_red = std_func(red_stellar_mass_bins, red_cen_stellar_mass_arr, 
                red_deltav_arr)
            std_red = np.array(std_red)
            std_red_arr.append(std_red)

            # Calculating spread in velocity dispersion for galaxies in groups 
            # with a blue central

            blue_deltav_arr = []
            blue_cen_stellar_mass_arr = []
            for key in blue_subset_grpids: 
                group = mock_pd.loc[mock_pd.groupid == key]
                cen_stellar_mass = group.logmstar.loc[group.g_galtype\
                    .values == 1].values[0]
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                for val in deltav:
                    blue_deltav_arr.append(val)
                    blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            # print(max(blue_cen_stellar_mass_arr))

            if survey == 'eco' or survey == 'resolvea':
                # TODO : check if this is actually correct for resolve a
                blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
            elif survey == 'resolveb':
                blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
            std_blue = std_func(blue_stellar_mass_bins, \
                blue_cen_stellar_mass_arr, blue_deltav_arr)    
            std_blue = np.array(std_blue)
            std_blue_arr.append(std_blue)

            centers_red = 0.5 * (red_stellar_mass_bins[1:] + \
                red_stellar_mass_bins[:-1])
            centers_blue = 0.5 * (blue_stellar_mass_bins[1:] + \
                blue_stellar_mass_bins[:-1])
            
            centers_red_arr.append(centers_red)
            centers_blue_arr.append(centers_blue)
    
    std_red_arr = np.array(std_red_arr)
    centers_red_arr = np.array(centers_red_arr)
    std_blue_arr = np.array(std_blue_arr)
    centers_blue_arr = np.array(centers_blue_arr)
            
    return std_red_arr, std_blue_arr, centers_red_arr, centers_blue_arr

def lnprob(theta, x_vals, y_vals, err_tot):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values

    x_vals: array
        Array of x-axis values

    y_vals: array
        Array of y-axis values
    
    err_tot: array
        Array of error values of mass function

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        try:
            model = m * x_vals + b 
            chi2 = chi_squared(y_vals, model, err_tot)
            lnp = -chi2 / 2
            if math.isnan(lnp):
                raise ValueError
        except (ValueError, RuntimeWarning, UserWarning):
            lnp = -np.inf
            chi2 = -np.inf
    else:
        chi2 = -np.inf
        lnp = -np.inf
        
    return lnp, chi2

def chi_squared(data, model, err_data):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: float
        Error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    chi_squared_arr = (data - model)**2 / (err_data**2)
    chi_squared = np.sum(chi_squared_arr)

    return chi_squared

global model_init
global survey
global path_to_proc
global mf_type

survey = 'resolveb'
machine = 'mac'
mf_type = 'smf'

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
    catl_file = path_to_raw + "eco/eco_all.csv"
elif survey == 'resolvea' or survey == 'resolveb':
    catl_file = path_to_raw + "resolve/RESOLVE_liveJune2018.csv"

if survey == 'eco':
    path_to_mocks = path_to_data + 'mocks/m200b/eco/'
elif survey == 'resolvea':
    path_to_mocks = path_to_data + 'mocks/m200b/resolvea/'
elif survey == 'resolveb':
    path_to_mocks = path_to_data + 'mocks/m200b/resolveb/'

std_red_mocks, std_blue_mocks, centers_red_mocks, \
    centers_blue_mocks = get_deltav_sigma_mocks(survey, path_to_mocks)

## Histogram of red and blue sigma in bins of central stellar mass to see if the
## distribution of values to take std of is normal or lognormal
nrows = 2
ncols = 5
if survey == 'eco' or survey == 'resolvea':
    red_stellar_mass_bins = np.linspace(8.6,11.5,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
fig3, axs = plt.subplots(nrows, ncols)
for i in range(0, nrows, 1):
    for j in range(0, ncols, 1):
        if i == 0: # row 1 for all red bins
            axs[i, j].hist(np.log10(std_red_mocks.T[j]), histtype='step', \
                color='indianred', linewidth=4, linestyle='-') # first red bin
            axs[i, j].set_title('[{0}-{1}]'.format(np.round(
                red_stellar_mass_bins[j],2), np.round(
                red_stellar_mass_bins[j+1],2)), fontsize=20)
            k2, p = nt(np.log10(std_red_mocks.T[j]), nan_policy="omit")
            axs[i, j].text(0.7, 0.7, "{0}".format(np.round(p, 2)),
                transform=axs[i, j].transAxes)
        else: # row 2 for all blue bins
            axs[i, j].hist(np.log10(std_blue_mocks.T[j]), histtype='step', \
                color='cornflowerblue', linewidth=4, linestyle='-')
            axs[i, j].set_title('[{0}-{1}]'.format(np.round(
                blue_stellar_mass_bins[j],2), np.round(
                blue_stellar_mass_bins[j+1],2)), fontsize=20)
            k2, p = nt(np.log10(std_blue_mocks.T[j]), nan_policy="omit")
            axs[i, j].text(0.7, 0.7, "{0}".format(np.round(p, 2)), 
                transform=axs[i, j].transAxes)
for ax in axs.flat:
    ax.set(xlabel=r'\boldmath$\sigma \left[km/s\right]$')

for ax in axs.flat:
    ax.label_outer()

plt.show()

## Measuring fractional error in sigma of red and blue galaxies in all 5 bins
sigma_av_red = [] 
frac_err_red = [] 
for idx in range(len(std_red_mocks.T)): 
    mean = np.mean(std_red_mocks.T[idx][~np.isnan(std_red_mocks.T[idx])]) 
    sigma_av_red.append(mean) 
    frac_err = (std_red_mocks.T[idx][~np.isnan(std_red_mocks.T[idx])] \
        - mean)/mean 
    frac_err_red.append(frac_err) 
frac_err_red = np.array(frac_err_red, dtype=list)

sigma_av_blue = [] 
frac_err_blue = [] 
for idx in range(len(std_blue_mocks.T)): 
    mean = np.mean(std_blue_mocks.T[idx][~np.isnan(std_blue_mocks.T[idx])]) 
    sigma_av_blue.append(mean) 
    frac_err = (std_blue_mocks.T[idx][~np.isnan(std_blue_mocks.T[idx])] \
        - mean)/mean 
    frac_err_blue.append(frac_err) 
frac_err_blue = np.array(frac_err_blue, dtype=list)


## Fit fractional error distributions
nrows = 2
ncols = 5
if survey == 'eco' or survey == 'resolvea':
    red_stellar_mass_bins = np.linspace(8.6,11.5,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)

max_red_arr = np.empty(len(frac_err_red))
max_blue_arr = np.empty(len(frac_err_blue))
for idx in range(len(frac_err_red)):
    max_red = plt.hist(frac_err_red[idx], density=True)[0].max()
    max_blue = plt.hist(frac_err_blue[idx], density=True)[0].max()
    max_red_arr[idx] = max_red + 0.05
    max_blue_arr[idx] = max_blue + 0.05
    print(np.mean(frac_err_red[idx]))
    print(np.mean(frac_err_blue[idx]))
plt.clf()


red_a = []
red_loc = []
red_scale = []
blue_a = []
blue_loc = []
blue_scale = []
fig3, axs = plt.subplots(nrows, ncols)
for i in range(0, nrows, 1):
    for j in range(0, ncols, 1):
        if i == 0: # row 1 for all red bins
            frac_err_arr = frac_err_red[j]
            axs[i,j].hist(frac_err_arr, density=True, histtype='step', 
                linewidth=3, color='k')

            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            xt = axs[i,j].get_xticks()  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(frac_err_arr))

            loc_log, scale_log = stats.logistic.fit(frac_err_arr)
            pdf_logistic = stats.logistic.pdf(lnspc, loc_log, scale_log)  
            axs[i,j].plot(lnspc, pdf_logistic, label="Logistic")

            # a_beta, b_beta, loc_beta, scale_beta = stats.beta.fit(frac_err_arr)
            # pdf_beta = stats.beta.pdf(lnspc, a_beta, b_beta, loc_beta, 
            #     scale_beta)  
            # axs[i,j].plot(lnspc, pdf_beta, label="Beta")

            loc_norm, scale_norm = stats.norm.fit(frac_err_arr)
            pdf_norm = stats.norm.pdf(lnspc, loc_norm, scale_norm)  
            axs[i,j].plot(lnspc, pdf_norm, label="Normal")

            a_sn, loc_sn, scale_sn = stats.skewnorm.fit(frac_err_arr)
            pdf_skewnorm = stats.skewnorm.pdf(lnspc, a_sn, loc_sn, scale_sn)  
            axs[i,j].plot(lnspc, pdf_skewnorm, label="Skew-normal")
            red_a.append(a_sn)
            red_loc.append(loc_sn)
            red_scale.append(scale_sn)

            # a_w, loc_w, scale_w = stats.weibull_min.fit(frac_err_arr)
            # pdf_weibull = stats.weibull_min.pdf(lnspc, a_w, loc_w, scale_w)  
            # axs[i,j].plot(lnspc, pdf_weibull, label="Weibull")

            # a_g,loc_g,scale_g = stats.gamma.fit(frac_err_arr)  
            # pdf_gamma = stats.gamma.pdf(lnspc, a_g, loc_g, scale_g)  
            # axs[i,j].plot(lnspc, pdf_gamma, label="Gamma")

            axs[i,j].set_title('[{0}-{1}]'.format(np.round(
                red_stellar_mass_bins[j],2), np.round(
                red_stellar_mass_bins[j+1],2)),fontsize=20, 
                color='indianred')
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (a_sn, ),
                r'$loc=%.2f$' % (loc_sn, ),
                r'$scale=%.2f$' % (scale_sn, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[i,j].set_ylim(0, max_red_arr[j])
            # axs[i, j].text(0.4, 0.8, textstr, fontsize=12, bbox=props, 
            #     transform=axs[i, j].transAxes)

        else: # row 2 for all blue bins
            frac_err_arr = frac_err_blue[j]
            axs[i,j].hist(frac_err_arr, density=True, histtype='step', 
                linewidth=3, color='k')

            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            xt = axs[i,j].get_xticks() 
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(frac_err_arr))

            loc_log, scale_log = stats.logistic.fit(frac_err_arr)
            pdf_logistic = stats.logistic.pdf(lnspc, loc_log, scale_log)  
            axs[i,j].plot(lnspc, pdf_logistic, label="Logistic")

            # a_beta, b_beta, loc_beta, scale_beta = stats.beta.fit(frac_err_arr)
            # pdf_beta = stats.beta.pdf(lnspc, a_beta, b_beta, loc_beta, 
            #     scale_beta)  
            # axs[i,j].plot(lnspc, pdf_beta, label="Beta")

            loc_norm, scale_norm = stats.norm.fit(frac_err_arr)
            pdf_norm = stats.norm.pdf(lnspc, loc_norm, scale_norm)  
            axs[i,j].plot(lnspc, pdf_norm, label="Normal")

            a_sn, loc_sn, scale_sn = stats.skewnorm.fit(frac_err_arr)
            pdf_skewnorm = stats.skewnorm.pdf(lnspc, a_sn, loc_sn, scale_sn)  
            axs[i,j].plot(lnspc, pdf_skewnorm, label="Skew-normal")
            blue_a.append(a_sn)
            blue_loc.append(loc_sn)
            blue_scale.append(scale_sn)

            # a_w, loc_w, scale_w = stats.weibull_min.fit(frac_err_arr)
            # pdf_weibull = stats.weibull_min.pdf(lnspc, a_w, loc_w, scale_w)  
            # axs[i,j].plot(lnspc, pdf_weibull, label="Weibull")

            # a_g, loc_g, scale_g = stats.gamma.fit(frac_err_arr)  
            # pdf_gamma = stats.gamma.pdf(lnspc, a_g, loc_g,scale_g)  
            # axs[i,j].plot(lnspc, pdf_gamma, label="Gamma")

            axs[i,j].set_title('[{0}-{1}]'.format(np.round(
                blue_stellar_mass_bins[j],2), np.round(
                blue_stellar_mass_bins[j+1],2)), fontsize=20, 
                color='cornflowerblue')
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (a_sn, ),
                r'$loc=%.2f$' % (loc_sn, ),
                r'$scale=%.2f$' % (scale_sn, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[i,j].set_ylim(0, max_blue_arr[j])
            # axs[i, j].text(0.4, 0.8, textstr, fontsize=12, bbox=props, 
            #     transform=axs[i, j].transAxes)

red_a = np.array(red_a) 
red_loc = np.array(red_loc)
red_scale = np.array(red_scale)
blue_a = np.array(blue_a) 
blue_loc = np.array(blue_loc)
blue_scale = np.array(blue_scale)

a_arr = (np.array((red_a, blue_a))).flatten()
loc_arr = (np.array((red_loc, blue_loc))).flatten()
scale_arr = (np.array((red_scale, blue_scale))).flatten()

axs[0,0].legend(loc='center right', prop={'size': 8})

axs[1,2].set(xlabel=r'\boldmath$(\sigma - \bar \sigma )/ \bar \sigma$')

plt.show()


## Simulating errors
np.random.seed(30)
m_true_arr = np.round(np.random.uniform(-4.9, 0.4, size=500),2)
b_true_arr = np.round(np.random.uniform(1, 7, size=500),2)

## Keeping data fixed
m_true = m_true_arr[50]
b_true = b_true_arr[50]
N=10
x = np.sort(10*np.random.rand(N))

samples_arr = []
chi2_arr = []
yerr_arr = []
for i in range(500):
    print(i)

    ## Mimicking non-gaussian errors from mocks
    # yerr = stats.skewnorm.rvs(a_arr, loc_arr, scale_arr) 

    ## Corresponding gaussian distributions
    var_arr = stats.skewnorm.stats(a_arr, loc_arr, scale_arr,  moments='mvsk')[1]
    # mu_arr = stats.skewnorm.stats(a_arr, loc_arr, scale_arr,  moments='mvsk')[0]
    std_arr = np.sqrt(var_arr)

    ## Simulating gaussian errors with mean of 0 and same sigma as corresponding
    ## non-gaussian fits
    yerr = stats.norm.rvs(np.zeros(10), std_arr)

    y = m_true * x + b_true
    y_new = y + y*yerr

    pos = [0,5] + 1e-4 * np.random.randn(64, 2) 
    nwalkers, ndim = pos.shape
    nsteps = 5000

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
        args=(x, y_new, std_arr))
    sampler.run_mcmc(pos, nsteps, store=True, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    chi2 = sampler.get_blobs(discard=100, thin=15, flat=True)

    samples_arr.append(flat_samples)
    chi2_arr.append(chi2)
    yerr_arr.append(yerr)

non_gaussian_samples_arr = np.array(samples_arr)
non_gaussian_chi2_arr = np.array(chi2_arr)
non_gaussian_yerr_arr = np.array(yerr_arr)

gaussian_samples_arr = np.array(samples_arr)
gaussian_chi2_arr = np.array(chi2_arr)
gaussian_yerr_arr = np.array(yerr_arr)

# Get parameter values for each realization that correspond to lowest 
# chi-squared value
gaussian_minchi2_arr = []
gaussian_minm_arr = []
gaussian_minb_arr = []
for idx in range(len(gaussian_chi2_arr)):
    gaussian_minchi2_arr.append(min(gaussian_chi2_arr[idx]))
    gaussian_minm_arr.append(gaussian_samples_arr[idx][:,0][np.argmin(gaussian_chi2_arr[idx])])
    gaussian_minb_arr.append(gaussian_samples_arr[idx][:,1][np.argmin(gaussian_chi2_arr[idx])])

samples = np.column_stack((gaussian_minm_arr,gaussian_minb_arr)) 

non_gaussian_minchi2_arr = []
non_gaussian_minm_arr = []
non_gaussian_minb_arr = []
for idx in range(len(non_gaussian_chi2_arr)):
    non_gaussian_minchi2_arr.append(min(non_gaussian_chi2_arr[idx]))
    non_gaussian_minm_arr.append(non_gaussian_samples_arr[idx][:,0][np.argmin(non_gaussian_chi2_arr[idx])])
    non_gaussian_minb_arr.append(non_gaussian_samples_arr[idx][:,1][np.argmin(non_gaussian_chi2_arr[idx])])

samples = np.column_stack((non_gaussian_minm_arr,non_gaussian_minb_arr)) 


## Plot of maximum likelihood points
m_mean = samples[:,0].mean()
m_std = samples[:,0].std()
b_mean = samples[:,1].mean()
b_std = samples[:,1].std()

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
fig1 = plt.figure(figsize=(8, 8))

textstr_m = '\n'.join((
    r'$\mu=%.2f$' % (m_mean, ),
    r'$\sigma=%.2f$' % (m_std, )))

textstr_b = '\n'.join((
    r'$\mu=%.2f$' % (b_mean, ),
    r'$\sigma=%.2f$' % (b_std, )))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax_histx.text(0.20, 0.95, textstr_m, transform=ax_histx.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
# place a text box in upper left in axes coords
ax_histy.text(0.50, 0.98, textstr_b, transform=ax_histy.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax_scatter.scatter(samples[:,0], samples[:,1], c='mediumorchid', marker='+')
ax_histx.hist(samples[:,0], histtype='step', color='cornflowerblue',
    linewidth=5)
ax_histy.hist(samples[:,1], histtype='step', orientation='horizontal', 
    color='cornflowerblue', linewidth=5)

ax_scatter.set_xlabel('m')
ax_scatter.set_ylabel('b')

ax_histx.set_title('Distribution of maximum likelihood points given non-gaussian errors')
fig1.show()

gaussian_minchi2_arr = np.array(gaussian_minchi2_arr)
non_gaussian_minchi2_arr = np.array(non_gaussian_minchi2_arr)
min_bin = math.floor(min([min(gaussian_minchi2_arr), min(non_gaussian_minchi2_arr)]))
max_bin = math.ceil(max([max(gaussian_minchi2_arr), max(non_gaussian_minchi2_arr)]))
num_bins = 20
dof = N - ndim
## Plot of distributions of minimum chi-squared
fig2 = plt.figure()
x_arr = np.linspace(chi2.ppf(0.01, dof),chi2.ppf(0.99, dof), 100)
# plt.plot(x_arr, 168*chi2.pdf(x_arr, dof), 'r-', lw=5, alpha=0.6, label='scaled chi2 pdf')
plt.hist(gaussian_minchi2_arr[~np.isinf(gaussian_minchi2_arr)], histtype='step',
    bins=np.linspace(min_bin, max_bin, num_bins),
    color='cornflowerblue', label='Gaussian', linewidth=5)                                          
plt.hist(non_gaussian_minchi2_arr[~np.isinf(non_gaussian_minchi2_arr)],
    bins=np.linspace(min_bin, max_bin, num_bins),
    histtype='step', color='mediumorchid', label='Skew normal', linewidth=5)
plt.title(r'Distribution of min(${\chi}^{2}$)') 
plt.legend(loc='upper right')     
fig2.show() 

truth = {"$m$": m_true, "$b$": b_true}  
c = ChainConsumer()  
c.add_chain(gaussian_samples_arr.mean(axis=1), 
    parameters=[r"$\bar m$", r"$\bar b$"], color='#6495ed', name='Gaussian') 
c.add_chain(non_gaussian_samples_arr.mean(axis=1), 
    parameters=[r"$\bar m$", r"$\bar b$"], color='#ba55d3', name='Skew normal') 
c.configure(diagonal_tick_labels=False, tick_font_size=8, label_font_size=25, 
    max_ticks=8, legend_location=(0,-1))  
fig = c.plotter.plot(truth=truth, display=True)      

## Plot of combined std of chains
c = ChainConsumer()  
c.add_chain(gaussian_samples_arr.std(axis=1), 
    parameters=[r"${\sigma}_{m}$", r"${\sigma}_{b}$"], color='#6495ed', 
    name='Gaussian') 
c.add_chain(non_gaussian_samples_arr.std(axis=1), 
    parameters=[r"${\sigma}_{m}$", r"${\sigma}_{b}$"], color='#ba55d3', 
    name='Skew normal') 
c.configure(diagonal_tick_labels=False, tick_font_size=8, label_font_size=25, 
    max_ticks=8, legend_location=(0,-1))  
fig = c.plotter.plot(display=True)


var_arr = stats.skewnorm.stats(a_arr, loc_arr, scale_arr,  moments='mvsk')[1]
std_arr = np.sqrt(var_arr)

nrows = 2
ncols = 5
if survey == 'eco' or survey == 'resolvea':
    red_stellar_mass_bins = np.linspace(8.6,11.5,6)
    blue_stellar_mass_bins = np.linspace(8.6,10.5,6)
elif survey == 'resolveb':
    red_stellar_mass_bins = np.linspace(8.4,11.0,6)
    blue_stellar_mass_bins = np.linspace(8.4,10.4,6)
    
fig3, axs = plt.subplots(nrows, ncols)
for i in range(0, nrows, 1):
    for j in range(0, ncols, 1):
        if i == 0: # row 1 for all red bins
            frac_err_arr = frac_err_red[j]
            frac_err_arr = frac_err_arr[~np.isnan(frac_err_arr)]
            axs[i,j].hist(frac_err_arr, density=True, histtype='step', color='k', 
                label='Mocks', linewidth=2)

            axs[i,j].hist(stats.norm.rvs(0, std_arr[j], size=100000), 
                density=True, histtype='step', label='Gaussian', linewidth=2)
            axs[i,j].hist(stats.skewnorm.rvs(a_arr[j], loc_arr[j], scale_arr[j], 
                size=100000), density=True, histtype='step', linewidth=2, 
                label='Skew normal') 

            axs[i,j].set_title('[{0}-{1}]'.format(np.round(
                red_stellar_mass_bins[j],2), np.round(
                red_stellar_mass_bins[j+1],2)), color='indianred')
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (0, ),
                r'$\sigma=%.2f$' % (std_arr[j], ),
                r'$a=%.2f$' % (a_arr[j], ),
                r'$loc=%.2f$' % (loc_arr[j], ),
                r'$scale=%.2f$' % (scale_arr[j], )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            if i == 0 and j >= 2:
                axs[i, j].text(0.5, 0.7, textstr, fontsize=10, bbox=props, 
                    transform=axs[i, j].transAxes) 
            else:
                axs[i, j].text(0.4, 0.7, textstr, fontsize=10, bbox=props, 
                    transform=axs[i, j].transAxes)


        else: # row 2 for all blue bins
            frac_err_arr = frac_err_blue[j]
            frac_err_arr = frac_err_arr[~np.isnan(frac_err_arr)]
            axs[i,j].hist(frac_err_arr, density=True, histtype='step', color='k', 
                label='Mocks', linewidth=2)

            axs[i,j].hist(stats.norm.rvs(0, std_arr[j+5], size=100000), 
                density=True, histtype='step', linewidth=2, label='Gaussian')
            axs[i,j].hist(stats.skewnorm.rvs(a_arr[j+5], loc_arr[j+5], 
                scale_arr[j+5], size=100000), density=True, histtype='step', 
                label='Skew normal', linewidth=2) 

            axs[i,j].set_title('[{0}-{1}]'.format(np.round(
                blue_stellar_mass_bins[j],2), np.round(
                blue_stellar_mass_bins[j+1],2)), color='cornflowerblue')
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (0, ),
                r'$\sigma=%.2f$' % (std_arr[j+5], ),
                r'$a=%.2f$' % (a_arr[j+5], ),
                r'$loc=%.2f$' % (loc_arr[j+5], ),
                r'$scale=%.2f$' % (scale_arr[j+5], )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)                  
            axs[i, j].text(0.4, 0.7, textstr, fontsize=10, bbox=props, 
                transform=axs[i, j].transAxes)

axs[0,0].legend(loc='center right', prop={'size': 10})

axs[1,2].set(xlabel=r'\boldmath$(\sigma - \bar \sigma )/ \bar \sigma$')

plt.show()
