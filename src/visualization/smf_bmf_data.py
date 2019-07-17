"""
{This script plots SMF and BMF from all 3 surveys}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import argparse
import math

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=10)
rc('text', usetex=True)

def read_catl(path_to_file, survey):
    """
    Reads survey catalog from file

    Parameters
    ----------
    path_to_file: string
        Path to survey catalog file

    survey: string
        Name of survey

    Returns
    ---------
    catl: pandas dataframe
        Survey catalog with grpcz, abs rmag and stellar mass limits
    
    volume: float
        Volume of survey

    cvar: float
        Cosmic variance of survey
    """
    if survey == 'eco':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        if h == 1.0:
            volume = 151829.26 # Survey volume without buffer [Mpc/h]^3 in h=1.0
            cz_measurement = eco_buff.grpcz.values
        elif h == 0.7:
            #Survey volume without buffer [Mpc/h]^3
            volume = 151829.26 * 2.915 # convert from h = 1.0 to 0.7
            cz_measurement = eco_buff.cz.values
        cvar = 0.125

        if mass == 'smf':
        # 6456 galaxies                       
            catl = eco_buff.loc[(cz_measurement >= 3000) & \
                (cz_measurement <= 7000) & (eco_buff.absrmag.values <= -17.33) & \
                        (eco_buff.logmstar.values >= 8.9)]
        elif mass == 'bmf':
            # Removing stellar mass cut
            catl = eco_buff.loc[(cz_measurement >= 3000) & \
                (cz_measurement <= 7000) & (eco_buff.absrmag.values <= -17.33)]
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if h == 1.0:
            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cz_measurement = resolve_live18.grpcz.values
        elif h == 0.7:
            #Survey volume without buffer [Mpc/h]^3
            volume = 13172.384 * 2.915 # convert from h = 1.0 to 0.7
            cz_measurement = resolve_live18.cz.values
        cvar = 0.30

        if survey == 'resolvea':
            if mass == 'smf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
                    (cz_measurement > 4500) & (cz_measurement < 7000) & \
                            (resolve_live18.absrmag.values < -17.33) & \
                                (resolve_live18.logmstar.values >= 8.9)]
            elif mass == 'bmf':
                 catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
                    (cz_measurement > 4500) & (cz_measurement < 7000) & \
                            (resolve_live18.absrmag.values < -17.33)]          
        
        elif survey == 'resolveb':
            if mass == 'smf':
            # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
                    (cz_measurement > 4500) & (cz_measurement < 7000) & \
                            (resolve_live18.absrmag.values < -17) & \
                                (resolve_live18.logmstar.values >= 8.7)]
            elif mass == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
                    (cz_measurement > 4500) & (cz_measurement < 7000) & \
                            (resolve_live18.absrmag.values < -17)]

            if h == 1.0:
                volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3
            elif h == 0.7:
                #Survey volume without buffer [Mpc/h]^3
                volume = 4709.8373 * 2.915 # convert from h = 1.0 to 0.7
            cvar = 0.58

    return catl, volume, cvar

def diff_smf(mstar_arr, volume, cvar_err):
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
    if h == 1.0:
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
        bin_num = 12
    elif h == 0.7: 
        logmstar_arr = mstar_arr
        bin_num = 16
    if survey == 'eco' or survey == 'resolvea':
        bins = np.linspace(8.9, 11.8, bin_num)
        print("{0} : {1}".format(survey,len(logmstar_arr[logmstar_arr>=8.9])))
    elif survey == 'resolveb':
        bins = np.linspace(8.7, 11.8, bin_num)
        print("{0} : {1}".format(survey,len(logmstar_arr[logmstar_arr>=8.7])))
    # Unnormalized histogram and bin edges
    phi, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(phi) / (volume * dm)
    err_cvar = cvar_err #/ (volume * dm)
    print('Poisson error: {0}'.format(err_poiss))
    
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    phi = phi / (volume * dm)  # not a log quantity
    err_cvar = err_cvar*phi
    print('Cosmic variance error: {0}'.format(err_cvar))
    return maxis, phi, err_tot, bins

def calc_bary(mstar_arr, mgas_arr):
    """
    Calculates baryonic mass from stellar and gas mass

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    mgass_arr: numpy array
        Array of gas masses

    Returns
    ---------
    logmbary: numpy array
        Array of baryonic masses

    bin_num: int
        Number of bins to use
    """
    if h == 1.0:
        logmbary = np.log10(((10**mstar_arr) + (10**mgas_arr)) / 2.041)
        bin_num = 12
    elif h == 0.7:
        logmbary = np.log10((10**mstar_arr) + (10**mgas_arr))
        bin_num = 16
    return logmbary, bin_num

def diff_bmf(logmbary_arr, volume, cvar_err, bin_num):
    """
    Calculates differential baryonic mass function

    Parameters
    ----------
    mass_arr: numpy array
        Array of baryonic masses

    volume: float
        Volume of survey

    cvar_err: float
        Cosmic variance of survey
    
    bin_num: int
        Number of bins to use

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
    # Unnormalized histogram and bin edges
    
    if survey == 'eco' or survey == 'resolvea':
        bins = np.linspace(9.4,12.0,bin_num)
        print("{0} : {1}".format(survey,len(logmbary_arr[logmbary_arr>=9.4])))
    if survey == 'resolveb':
        bins = np.linspace(9.1,12.0,bin_num)
        print("{0} : {1}".format(survey,len(logmbary_arr[logmbary_arr>=9.1])))
    phi, edg = np.histogram(logmbary_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(phi) / (volume * dm)
    err_cvar = cvar_err #/ (volume * dm)
    print('Poisson error: {0}'.format(err_poiss))
    
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    phi = phi / (volume * dm)  # not a log quantity
    err_cvar = phi * err_cvar
    print('Cosmic variance error: {0}'.format(err_cvar))
    err_tot = err_cvar * phi
    return maxis, phi, err_tot, bins

def plot_massfunc(maxis_70, phi_70, err_70, maxis_100, phi_100, err_100):
    """
    Plot SMF from data, best fit param values and param values corresponding to 
    68th percentile 1000 lowest chi^2 values

    Parameters
    ----------
    maxis_70: array
        Array of x-axis mass values for data SMF assuming h=0.7

    phi_70: array
        Array of y-axis values for data SMF assuming h=0.7

    err_70: array
        Array of error values per bin of data SMF assuming h=0.7

    maxis_100: array
        Array of x-axis mass values for data SMF assuming h=1.0

    phi_100: array
        Array of y-axis values for data SMF assuming h=1.0

    err_100: array
        Array of error values per bin of data SMF assuming h=1.0

    Returns
    ---------
    Nothing; SMF plot is saved in figures repository
    """
    if survey == 'resolvea':
        line_label = 'RESOLVE-A'
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
    elif survey == 'eco':
        line_label = 'ECO'

    fig1 = plt.figure(figsize=(10,10))
    plt.plot(maxis_70,phi_70,'k-')
    plt.fill_between(maxis_70,phi_70-err_70,phi_70+err_70,color='g',alpha=0.3)
    plt.errorbar(maxis_70,phi_70,yerr=err_70,color='k',fmt='-s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='{0} h=0.7'.format(line_label),\
            zorder=10)
    plt.plot(maxis_100,phi_100,'k--')
    plt.fill_between(maxis_100,phi_100-err_100,phi_100+err_100,color='b',alpha=0.3)
    plt.errorbar(maxis_100,phi_100,yerr=err_100,color='k',fmt='--s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='{0} h=1.0'.format(line_label),\
            zorder=10)
    plt.yscale('log')
    plt.ylim(10**-5,10**-1)
    if mass == 'smf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h^{-2}} \right]$', fontsize=15)
        # if h == 0.7:
        #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h_{70}}^{-2} \right]$', fontsize=15)
        # elif h == 1.0:
        #     plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h_{100}}^{-2} \right]$', fontsize=15)
    elif mass == 'bmf':
        plt.xlabel(r'\boldmath$\log_{10}\ M_{bary} \left[\mathrm{M_\odot}\, \mathrm{h^{-2}} \right]$', fontsize=15)
        # if h == 0.7:
        #     plt.xlabel(r'\boldmath$\log_{10}\ M_{bary} \left[\mathrm{M_\odot}\, \mathrm{h_{70}}^{-2} \right]$', fontsize=15)
        # elif h == 1.0:
        #     plt.xlabel(r'\boldmath$\log_{10}\ M_{bary} \left[\mathrm{M_\odot}\, \mathrm{h_{100}}^{-2} \right]$', fontsize=15)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h^{3}} \right]$', fontsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
    plt.show()
    # plt.savefig(path_to_figures + '{0}_{1}.png'.format(mass,survey))

def plot_smf_bmf(maxis_smf, maxis_bmf, phi_smf, phi_bmf, err_smf, err_bmf):
    """
    Plot SMF and BMF from data

    Parameters
    ----------
    maxis_smf: array
        Array of x-axis mass values for data SMF assuming h=1.0

    phi_smf: array
        Array of y-axis values for data SMF assuming h=1.0

    err_smf: array
        Array of error values per bin of data SMF assuming h=1.0

    maxis_bmf: array
        Array of x-axis mass values for data BMF assuming h=1.0

    phi_bmf: array
        Array of y-axis values for data BMF assuming h=1.0

    err_bmf: array
        Array of error values per bin of data BMF assuming h=1.0

    Returns
    ---------
    Nothing; SMF plot is saved in figures repository
    """
    if survey == 'resolvea':
        line_label = 'RESOLVE-A'
    elif survey == 'resolveb':
        line_label = 'RESOLVE-B'
    elif survey == 'eco':
        line_label = 'ECO'

    fig2 = plt.figure(figsize=(10,10))
    plt.plot(maxis_smf,phi_smf,'k-')
    plt.fill_between(maxis_smf,phi_smf-err_smf,phi_smf+err_smf,color='g',alpha=0.3)
    plt.errorbar(maxis_smf,phi_smf,yerr=err_smf,color='k',fmt='-s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='{0} smf'.format(line_label),\
            zorder=10)
    plt.plot(maxis_bmf,phi_bmf,'k--')
    plt.fill_between(maxis_bmf,phi_bmf-err_bmf,phi_bmf+err_bmf,color='b',alpha=0.3)
    plt.errorbar(maxis_bmf,phi_bmf,yerr=err_bmf,color='k',fmt='--s',ecolor='k',\
        markersize=4,capsize=5,capthick=0.5,label='{0} bmf'.format(line_label),\
            zorder=10)
    plt.yscale('log')
    plt.ylim(10**-5,10**-1)
    plt.xlabel(r'\boldmath$\log_{10}\ M \left[\mathrm{M_\odot}\, \mathrm{h^{-2}} \right]$', fontsize=15)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h^{3}} \right]$', fontsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 10})
    plt.show()
    
def args_parser():
    """
    Parsing arguments passed to script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('survey', type=str, \
        help='Options: eco/resolvea/resolveb')
    parser.add_argument('type', type=str, \
        help='Options: type of mass function (smf/bmf)')
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function that calls all other functions
    
    Parameters
    ----------
    args: 
        Input arguments to the script

    """
    global survey
    global mass
    global h
    global path_to_figures

    survey = args.survey
    mass = args.type

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_figures = dict_of_paths['plot_dir']

    if survey == 'eco':
        catl_file = path_to_raw + "eco_all.csv"
    elif survey == 'resolvea' or survey == 'resolveb':
        catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"

    h = 0.7
    print('Reading catalog')
    catl, volume, cvar = read_catl(catl_file, survey)
    print("{0} : {1} total".format(survey, len(catl)))

    print('Retrieving masses from catalog')
    if mass == 'smf':
        mstellar_arr = catl.logmstar.values
        maxis_70, phi_70, err_70, bins_70 = \
            diff_smf(mstellar_arr, volume, cvar)
    elif mass == 'bmf':
        mstellar_arr = catl.logmstar.values
        
        mgas_arr = catl.logmgas.values
        mbary_arr, bin_num = calc_bary(mstellar_arr, mgas_arr)
        maxis_70, phi_70, err_70, bins_70 = \
            diff_bmf(mbary_arr, volume, cvar, bin_num)

    h = 1.0
    print('Reading catalog')
    catl, volume, cvar = read_catl(catl_file, survey)
    print("{0} : {1} total".format(survey, len(catl)))
    
    print('Retrieving masses from catalog')
    if mass == 'smf':
        mstellar_arr = catl.logmstar.values
        maxis_100, phi_100, err_100, bins_100 = \
            diff_smf(mstellar_arr, volume, cvar)
    elif mass == 'bmf':
        mstellar_arr = catl.logmstar.values
        mgas_arr = catl.logmgas.values
        mbary_arr, bin_num = calc_bary(mstellar_arr, mgas_arr)
        maxis_100, phi_100, err_100, bins_100 = \
            diff_bmf(mbary_arr, volume, cvar, bin_num)
    """
    print('Plotting')
    plot_massfunc(maxis_70, phi_70, err_70, maxis_100, phi_100, err_100)  

    mstellar_arr = catl.logmstar.values
    maxis_smf, phi_smf, err_smf, bins_smf = \
        diff_smf(mstellar_arr, volume, cvar)

    mstellar_arr = catl.logmstar.values
    mgas_arr = catl.logmgas.values
    mbary_arr, bin_num = calc_bary(mstellar_arr, mgas_arr)
    maxis_bmf, phi_bmf, err_bmf, bins_bmf = \
        diff_bmf(mbary_arr, volume, cvar, bin_num)
    
    #Assuming h=1.0 which is what is used in MCMC
    plot_smf_bmf(maxis_smf, maxis_bmf, phi_smf, phi_bmf, err_smf, err_bmf)
"""
# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args) 
