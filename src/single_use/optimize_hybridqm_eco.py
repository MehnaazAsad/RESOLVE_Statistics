from cosmo_utils.utils import work_paths as cwpaths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import numpy as np
import math
import os

def hybrid_quenching_model(x_arr, Mstar_q, Mh_q, mu, nu):
    """
    Apply hybrid quenching model from Zu and Mandelbaum 2015

    Parameters
    ----------
    catl: pandas dataframe
        Data catalog

    Returns
    ---------
    f_red_cen: array
        Array of central red fractions
    f_red_sat: array
        Array of satellite red fractions
    """

    df = catl.copy()

    cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = get_host_halo_mock(df)
    cen_stellar_mass_arr, sat_stellar_mass_arr = get_stellar_mock(df)

    f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

    g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
    h_Mh = np.exp(-((sat_hosthalo_mass_arr/(10**Mh_q))**nu))
    f_red_sat = 1 - (g_Mstar * h_Mh)

    color_label_arr = [[] for x in range(len(df))]
    rng_arr = [[] for x in range(len(df))]
    # Adding columns for f_red to df
    df.loc[:, 'f_red'] = np.zeros(len(df))
    df.loc[df['fc'] == 1, 'f_red'] = f_red_cen
    df.loc[df['fc'] == 0, 'f_red'] = f_red_sat
    # Converting to array
    f_red_arr = df['f_red'].values
    # Looping over galaxies
    for ii, cs_ii in enumerate(df['fc']):
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
    drop_fred=False
    if drop_fred:
        df.drop('f_red', axis=1, inplace=True)

    mstar_red_arr = df.logmstar.loc[df.colour_label == 'R']
    mstar_blue_arr = df.logmstar.loc[df.colour_label == 'B']

    # changing from h=0.7 to h=1 assuming h^-2 dependence
    logmstar_red_arr = np.log10((mstar_red_arr) / 2.041)

    bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
    bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
    bin_num = 6
    bins_red = np.linspace(bin_min, bin_max, bin_num)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_red_arr, bins=bins_red)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = x_arr[0:5]  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss
    phi_red = counts / (volume * dm)  # not a log quantity

    phi_red = np.log10(phi_red)

    logmstar_blue_arr = np.log10((mstar_blue_arr) / 2.041)

    bin_max = np.round(np.log10((10**11) / 2.041), 1)
    bin_num = 6
    bins_blue = np.linspace(bin_min, bin_max, bin_num)

    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_blue_arr, bins=bins_blue)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = x_arr[5:10]  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss
    phi_blue = counts / (volume * dm)  # not a log quantity

    phi_blue = np.log10(phi_blue)

    phi_colour = [phi_red, phi_blue]
    phi_colour = np.array(phi_colour).flatten()

    return phi_colour

def get_host_halo_mock(catl):
    """
    Get host halo mass from mock catalog

    Parameters
    ----------
    catl: pandas dataframe
        Data catalog

    Returns
    ---------
    cen_halos: array
        Array of central host halo masses
    sat_halos: array
        Array of satellite host halo masses
    """

    df = catl.copy()

    cen_halos = []
    sat_halos = []
    for index, value in enumerate(df.fc):
        if value == 1:
            cen_halos.append(df.logmh.values[index])
        else:
            sat_halos.append(df.logmh.values[index])

    cen_halos = np.array(cen_halos)
    sat_halos = np.array(sat_halos)

    return cen_halos, sat_halos

def get_stellar_mock(catl):
    """
    Get stellar mass from mock catalog

    Parameters
    ----------
    catl: pandas dataframe
        Data catalog

    Returns
    ---------
    cen_gals: array
        Array of central stellar masses
    sat_gals: array
        Array of satellite stellar masses
    """

    df = catl.copy()

    cen_gals = []
    sat_gals = []
    for idx,value in enumerate(df.fc):
        if value == 1:
            cen_gals.append(df.logmstar.values[idx])
        elif value == 0:
            sat_gals.append(df.logmstar.values[idx])

    cen_gals = np.array(cen_gals)
    sat_gals = np.array(sat_gals)

    return cen_gals, sat_gals

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
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms','modelu_rcorr']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
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

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
path_to_data = dict_of_paths['data_dir']

global catl
global volume
survey = 'eco'
mf_type = 'smf'
catl_file = path_to_raw + "eco/eco_all.csv"
catl, volume, z_median = read_data_catl(catl_file, survey)
catl.logmh = 10**(catl.logmh.values)
catl.logmstar = 10**(catl.logmstar.values)
xdata = np.array([ 8.86,  9.38,  9.9 , 10.42, 10.94, 8.81,  9.23,  9.65, 10.07, 10.49])
theta =  np.array([10.5 , 13.76 , 0.69 , 0.15])
# y = hybrid_quenching_model(xdata, theta[0], theta[1], theta[2], theta[3])
y = np.array([-2.12357237, -2.0610347 , -1.97307953, -2.14607971, -2.73599082, -1.62996357, -1.78882601, -1.95090655, -2.26177934, -2.80028339])
popt, pcov = curve_fit(hybrid_quenching_model, xdata, y)