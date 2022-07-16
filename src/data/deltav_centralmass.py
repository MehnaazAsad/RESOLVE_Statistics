#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:45:22 2018

@author: asadm2
"""

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from collections import Counter
from scipy.stats import binned_statistic as bs
import matplotlib.gridspec as gridspec  
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import random

def std_func(num_arr):
    mean = 0
    diff_sqrd_arr = []
    for value in num_arr:
        diff = value - mean
        diff_sqrd = diff**2
        diff_sqrd_arr.append(diff_sqrd)
    mean_diff_sqrd = np.mean(diff_sqrd_arr)
    std = np.sqrt(mean_diff_sqrd)
    return std

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

# Bootstrap error in sigma using 20 samples and all galaxies in survey
def get_sigma_error(eco_keys):
    grp_key_arr = list(eco_keys)
    N_samples = 20
    std_blue_cen_arr_global = []
    std_red_cen_arr_global = []
    for i in range(N_samples):
        deltav_arr = []
        cen_stellar_mass_arr = []
        cen_colour_arr = []
        for j in range(len(eco_keys)):
            rng = random.choice(grp_key_arr)
            group = eco_groups.get_group(rng)
            cz_arr = group.cz.values
            grpcz = np.unique(group.grpcz.values)
            deltav = cz_arr-grpcz
            # Since some groups don't have a central
            if 1 in group.fc.values:
                cen_stellar_mass = group.logmstar.loc[group.fc.values == 1].values[0]
                cen_colour_label = group.colour_label.loc[group.fc.values == 1].values[0]
            for val in deltav:
                deltav_arr.append(val)
                cen_stellar_mass_arr.append(cen_stellar_mass)
                cen_colour_arr.append(cen_colour_label)
        data = {'deltav': deltav_arr, 'log_cen_stellar_mass': cen_stellar_mass_arr,\
            'cen_colour_label': cen_colour_arr}
        vel_col_mass_df = pd.DataFrame(data)
        df_blue_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'B']
        df_red_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_colour_label'] == 'R']
        stellar_mass_bins = np.arange(7,12,0.5)
        i = 0
        std_blue_cen_arr = []
        for index1,bin_edge in enumerate(stellar_mass_bins):
            df_blue_cen_deltav_arr = []
            if index1 == 9:
                break
            for index2,stellar_mass in enumerate(df_blue_cen.log_cen_stellar_mass.values):
                if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
                    df_blue_cen_deltav_arr.append(df_blue_cen.deltav.values[index2])
            std = std_func(df_blue_cen_deltav_arr)
            std_blue_cen_arr.append(std)
        i = 0
        std_red_cen_arr = []
        for index1,bin_edge in enumerate(stellar_mass_bins):
            df_red_cen_deltav_arr = []
            if index1 == 9:
                break
            for index2,stellar_mass in enumerate(df_red_cen.log_cen_stellar_mass.values):
                if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
                    df_red_cen_deltav_arr.append(df_red_cen.deltav.values[index2])
            std = std_func(df_red_cen_deltav_arr)
            std_red_cen_arr.append(std)
        std_blue_cen_arr_global.append(std_blue_cen_arr)
        std_red_cen_arr_global.append(std_red_cen_arr)

    std_red_cen_arr_global = np.array(std_red_cen_arr_global)
    std_blue_cen_arr_global = np.array(std_blue_cen_arr_global)

    std_red_cen_err = np.std(std_red_cen_arr_global, axis=0)
    std_blue_cen_err = np.std(std_blue_cen_arr_global, axis=0)

    return std_red_cen_err, std_blue_cen_err

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']

path_to_mocks = path_to_data + 'mocks/m200b/eco/'

eco = pd.read_csv(path_to_raw + 'eco/eco_all.csv', delimiter=',', header=0)
eco_dr2 = pd.read_csv(path_to_raw + 'eco/ecodr2.csv')
eco_dr2 = eco_dr2.loc[eco_dr2.name != 'ECO13860']
eco_nobuff = eco_dr2.loc[(eco_dr2.grpcz_e16.values >= 3000) & 
    (eco_dr2.grpcz_e16.values <= 7000) & (eco_dr2.absrmag.values <= -17.33) &
    (eco_dr2.logmstar.values >= 8.9)]

eco_nobuff = assign_colour_label_data(eco_nobuff)

eco_groups = eco_nobuff.groupby('grp_e16')
eco_keys = eco_groups.groups.keys()
deltav_arr = []
cen_stellar_mass_arr = []
cen_colour_label_arr = []
cen_colour_arr = []
grpn_arr = []
no_group_cen_counter = 0
for index,key in enumerate(eco_keys):
    group = eco_groups.get_group(key)
    # Since some groups don't have a central
    if 1 in group.fc_e16.values:
        cz_arr = group.cz.values
        cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
        av_cz_grp = np.average(cz_arr)
        deltav = cz_arr - av_cz_grp

        cen_stellar_mass = group.logmstar.loc[group.fc_e16.values == 1].values[0]
        
        cen_colour_label = group.colour_label.loc[group.fc_e16.values == 1].values[0]
        cen_colour = group.modelu_rcorr.loc[group.fc_e16.values == 1].values[0]
        
        grp_N = group.grpn_e16.loc[group.fc_e16.values == 1].values[0]
    # if 1 not in group.fc.values:
    #     no_group_cen_counter+=1 #14 groups out of 4427
        for val in deltav:
            deltav_arr.append(val)
            cen_stellar_mass_arr.append(cen_stellar_mass)
            cen_colour_label_arr.append(cen_colour_label)
            cen_colour_arr.append(cen_colour)
            grpn_arr.append(grp_N)

catl = eco_nobuff

deltav_red_data = []
deltav_blue_data = []
red_cen_mstar_data = []
blue_cen_mstar_data = []
sigma_red_data = []
sigma_blue_data = []

red_keys = catl.grp_e16.loc[(catl.fc_e16 == 1)&(catl.colour_label=='R')]
blue_keys = catl.grp_e16.loc[(catl.fc_e16 == 1)&(catl.colour_label=='B')]

for key in red_keys:
    group = catl.loc[catl.grp_e16 == key]
    if len(group) == 1:
        continue
    mean_cz_grp = np.average(group.cz.values)
    cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['fc_e16'].values == 1].values[0]
    deltav = group.cz.values - len(group)*[mean_cz_grp]
    for val in deltav:
        if val != 0 :
            deltav_red_data.append(val)
            red_cen_mstar_data.append(cen_mstar)
    sigma = np.std(deltav[deltav!=0])

    # red_cen_mstar_data.append(cen_mstar)
    sigma_red_data.append(sigma)

for key in blue_keys:
    group = catl.loc[catl.grp_e16 == key]
    if len(group) == 1:
        continue
    mean_cz_grp = np.average(group.cz.values)
    cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
    cen_mstar = group.logmstar.loc[group['fc_e16'].values == 1].values[0]
    deltav = group.cz.values - len(group)*[mean_cz_grp]
    for val in deltav:
        if val != 0 :
            deltav_blue_data.append(val)
            blue_cen_mstar_data.append(cen_mstar)
    sigma = np.std(deltav[deltav!=0])
    # blue_cen_mstar_data.append(cen_mstar)
    sigma_blue_data.append(sigma)

deltav_red_data = np.asarray(deltav_red_data)
deltav_blue_data = np.asarray(deltav_blue_data)
red_cen_mstar_data = np.asarray(red_cen_mstar_data)
blue_cen_mstar_data = np.asarray(blue_cen_mstar_data)
sigma_red_data = np.asarray(sigma_red_data)
sigma_blue_data = np.asarray(sigma_blue_data)

# deltav_red_data = np.log10(deltav_red_data)
# deltav_blue_data = np.log10(deltav_blue_data)

std_stats_red_data = bs(red_cen_mstar_data, deltav_red_data, 
    statistic='std', bins=np.linspace(8.9,11.5,5))
std_stats_blue_data = bs(blue_cen_mstar_data, deltav_blue_data,
    statistic='std', bins=np.linspace(8.9,11.5,5))

count_stats_red_data = bs(red_cen_mstar_data, deltav_red_data, 
    statistic='count', bins=np.linspace(8.9,11.5,5))
count_stats_blue_data = bs(blue_cen_mstar_data, deltav_blue_data,
    statistic='count', bins=np.linspace(8.9,11.5,5))

bins=np.linspace(8.9,11.5,5)
bins = 0.5 * (bins[1:] + bins[:-1])

poiss_err_red = std_stats_red_data[0]/np.sqrt(count_stats_red_data[0])
poiss_err_blue = std_stats_blue_data[0]/np.sqrt(count_stats_blue_data[0])

fig1 = plt.figure()
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0,0])
plt.scatter(cen_stellar_mass_arr, deltav_arr, s=50, c=cen_colour_arr, 
    cmap='coolwarm')
cbar = plt.colorbar()
cbar.set_label(r'$\mathbf{(u-r)^e_{group\ cen}}$', labelpad=15, fontsize=20)
plt.xlabel(r'$\mathbf{log_{10}\ M_{*,group\ cen}}\ [\mathbf{M_{\odot}}]$', 
    labelpad=10, fontsize=20)
plt.ylabel(r'$\mathbf{\Delta v\ \left[km/s\right]}$', labelpad=10, fontsize=20)
# plt.ylim(-2000, 2000)
plt.minorticks_on()
plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_deltav.pdf', 
    bbox_inches="tight", dpi=1200)

# left, bottom, width, height = [0.25, 0.28, 0.25, 0.25]
# ax2 = fig1.add_axes([left, bottom, width, height])
# ax2.errorbar(bins, std_stats_red_data[0], poiss_err_red,
#     color='#f46542',fmt='s',ecolor='#f46542',markersize=4,capsize=5,
#     capthick=0.5)
# ax2.errorbar(bins, std_stats_blue_data[0], poiss_err_blue,
#     color='#2d77e5',fmt='s',ecolor='#2d77e5',markersize=4,capsize=5,
#     capthick=0.5)
# ax2.set_ylabel(r'\boldmath$\sigma\ \left[km/s\right]$', fontsize=20)
fig1.tight_layout()
plt.show()


eco_nobuff = eco_dr2.loc[(eco_dr2.grpcz_e16.values >= 3000) & 
    (eco_dr2.grpcz_e16.values <= 7000) & (eco_dr2.absrmag.values <= -17.33) &
    (eco_dr2.logmbary.values >= 9.4)]

eco_nobuff = assign_colour_label_data(eco_nobuff)
eco_nobuff_realgas = eco_nobuff.loc[eco_nobuff.hitype_e16 == 1]

eco_groups = eco_nobuff_realgas.groupby('grp_e16')
eco_keys = eco_groups.groups.keys()
deltav_arr = []
cen_baryonic_mass_arr = []
cen_colour_label_arr = []
cen_colour_arr = []
grpn_arr = []
no_group_cen_counter = 0
for index,key in enumerate(eco_keys):
    group = eco_groups.get_group(key)
    # Since some groups don't have a central
    if 1 in group.fc_e16.values:
        cz_arr = group.cz.values
        cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
        deltav = cz_arr - cen_cz_grp

        # cen_baryonic_mass = np.log10(10**cen_stellar_mass + 10**cen_gas_mass)
        cen_baryonic_mass = group.logmbary.loc[group.fc_e16.values == 1].values[0]
        
        cen_colour_label = group.colour_label.loc[group.fc_e16.values == 1].values[0]
        cen_colour = group.modelu_rcorr.loc[group.fc_e16.values == 1].values[0]
        
        grp_N = group.grpn_e16.loc[group.fc_e16.values == 1].values[0]
    # if 1 not in group.fc.values:
    #     no_group_cen_counter+=1 #14 groups out of 4427
        for val in deltav:
            deltav_arr.append(val)
            cen_baryonic_mass_arr.append(cen_baryonic_mass)
            cen_colour_label_arr.append(cen_colour_label)
            cen_colour_arr.append(cen_colour)
            grpn_arr.append(grp_N)

catl = eco_nobuff_realgas

deltav_red_data = []
deltav_blue_data = []
red_cen_mbary_data = []
blue_cen_mbary_data = []
sigma_red_data = []
sigma_blue_data = []

red_keys = catl.grp_e16.loc[(catl.fc_e16 == 1)&(catl.colour_label=='R')]
blue_keys = catl.grp_e16.loc[(catl.fc_e16 == 1)&(catl.colour_label=='B')]

for key in red_keys:
    group = catl.loc[catl.grp_e16 == key]
    if len(group) == 1:
        continue
    mean_cz_grp = np.average(group.cz.values)
    cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
    cen_mbary = group.logmbary.loc[group['fc_e16'].values == 1].values[0]
    deltav = group.cz.values - len(group)*[cen_cz_grp]
    for val in deltav:
        if val != 0 :
            deltav_red_data.append(val)
            red_cen_mbary_data.append(cen_mbary)
    sigma = np.std(deltav[deltav!=0])

    # red_cen_mstar_data.append(cen_mstar)
    sigma_red_data.append(sigma)

for key in blue_keys:
    group = catl.loc[catl.grp_e16 == key]
    if len(group) == 1:
        continue
    mean_cz_grp = np.average(group.cz.values)
    cen_cz_grp = group.cz.loc[group['fc_e16'].values == 1].values[0]
    cen_mbary = group.logmbary.loc[group['fc_e16'].values == 1].values[0]
    deltav = group.cz.values - len(group)*[cen_cz_grp]
    for val in deltav:
        if val != 0 :
            deltav_blue_data.append(val)
            blue_cen_mbary_data.append(cen_mbary)
    sigma = np.std(deltav[deltav!=0])
    # blue_cen_mstar_data.append(cen_mstar)
    sigma_blue_data.append(sigma)

deltav_red_data = np.asarray(deltav_red_data)
deltav_blue_data = np.asarray(deltav_blue_data)
red_cen_mbary_data = np.asarray(red_cen_mbary_data)
blue_cen_mbary_data = np.asarray(blue_cen_mbary_data)
sigma_red_data = np.asarray(sigma_red_data)
sigma_blue_data = np.asarray(sigma_blue_data)

# deltav_red_data = np.log10(deltav_red_data)
# deltav_blue_data = np.log10(deltav_blue_data)

std_stats_red_data = bs(red_cen_mbary_data, deltav_red_data, 
    statistic='std', bins=np.linspace(9.4,11.5,5))
std_stats_blue_data = bs(blue_cen_mbary_data, deltav_blue_data,
    statistic='std', bins=np.linspace(9.4,11.5,5))

count_stats_red_data = bs(red_cen_mbary_data, deltav_red_data, 
    statistic='count', bins=np.linspace(9.4,11.5,5))
count_stats_blue_data = bs(blue_cen_mbary_data, deltav_blue_data,
    statistic='count', bins=np.linspace(9.4,11.5,5))

bins=np.linspace(9.4,11.5,5)
bins = 0.5 * (bins[1:] + bins[:-1])

poiss_err_red = std_stats_red_data[0]/np.sqrt(count_stats_red_data[0])
poiss_err_blue = std_stats_blue_data[0]/np.sqrt(count_stats_blue_data[0])

fig1 = plt.figure()
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0,0])
plt.scatter(cen_baryonic_mass_arr, deltav_arr, s=50, c=cen_colour_arr, 
    cmap='coolwarm')
cbar = plt.colorbar()
cbar.set_label(r'$\mathbf{(u-r)^e_{group\ cen}}$', labelpad=15, fontsize=20)
plt.xlabel(r'$\mathbf{log_{10}\ M_{b,group\ cen}}\ [\mathbf{M_{\odot}}]$', 
    labelpad=10, fontsize=20)

plt.ylabel(r'$\mathbf{\Delta v\ \left[km/s\right]}$', labelpad=10, fontsize=20)
# plt.ylim(-2000, 2000)
plt.minorticks_on()
# plt.savefig('eco_deltav.pdf', bbox_inches='tight', dpi=1200)
# plt.show()

left, bottom, width, height = [0.25, 0.28, 0.25, 0.25]
ax2 = fig1.add_axes([left, bottom, width, height])
ax2.errorbar(bins, std_stats_red_data[0], poiss_err_red,
    color='#f46542',fmt='s',ecolor='#f46542',markersize=4,capsize=5,
    capthick=0.5)
ax2.errorbar(bins, std_stats_blue_data[0], poiss_err_blue,
    color='#2d77e5',fmt='s',ecolor='#2d77e5',markersize=4,capsize=5,
    capthick=0.5)
ax2.set_ylabel(r'\boldmath$\sigma\ \left[km/s\right]$', fontsize=20)
fig1.tight_layout()
plt.show()

################################################################################
#* Panel plots of sigma-mstar and mstar-sigma for ECODR3 only (Figures 2a and 
#* 2b in paper)

survey = 'eco'
level = 'group'

def average_of_log(arr):
    result = np.log10(np.mean(10**(arr)))
    return result

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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            catl = catl.loc[catl.logmstar >= 8.9]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

    if catl_type == 'data' or catl_type == 'mock':
        logmstar_col = 'logmstar'
        ## Use group level for data even when settings.level == halo
        if catl_type == 'data' or level == 'group':
            galtype_col = 'fc'
            id_col = 'grp'
        ## No halo level in data
        if catl_type == 'mock':
            if level == 'halo':
                galtype_col = 'cs_flag'
                ## Halo ID is equivalent to halo_hostid in vishnu mock
                id_col = 'haloid'

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
    if catl_type == 'data':
        if survey == 'eco' or survey == 'resolvea':
            catl = catl.loc[catl.logmstar >= 8.9]
        elif survey == 'resolveb':
            catl = catl.loc[catl.logmstar >= 8.7]

    if catl_type == 'data' or catl_type == 'mock':
        logmstar_col = 'logmstar'
        ## Use group level for data even when settings.level == halo
        if catl_type == 'data' or level == 'group':
            galtype_col = 'fc'
            id_col = 'grp'
        ## No halo level in data
        if catl_type == 'mock':
            if level == 'halo':
                galtype_col = 'cs_flag'
                ## Halo ID is equivalent to halo_hostid in vishnu mock
                id_col = 'haloid'

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
    red_subset_df['deltav'] = red_subset_df['cz'] - red_subset_df['grpcz']
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
    blue_subset_df['deltav'] = blue_subset_df['cz'] - blue_subset_df['grpcz']
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

    colour_label_arr = np.array(['R' if x==1 else 'B' for x in catl.red.values])    
    catl['colour_label'] = colour_label_arr

    return catl

catl_file = path_to_raw + "eco/ecodr3.csv"
duplicate_file = path_to_raw + "ECO_duplicate_classification.csv"
catl_colours_file = path_to_raw + "eco_dr3_colours.csv"

eco_buff = pd.read_csv(catl_file, delimiter=",", header=0)
eco_duplicates = pd.read_csv(duplicate_file, header=0)
eco_colours = pd.read_csv(catl_colours_file, header=0, index_col=0)

# Removing duplicate entries
duplicate_names = eco_duplicates.NAME.loc[eco_duplicates.DUP > 0]
eco_buff = eco_buff[~eco_buff.name.isin(duplicate_names)].reset_index(drop=True)

# Combining colour information
eco_buff = pd.merge(eco_buff, eco_colours, left_on='name', right_on='galname')\
    .drop(columns=["galname"])

eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) &
    (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) &
    (eco_buff.logmstar.values >= 8.9)]

eco_nobuff = assign_colour_label_data(eco_nobuff)

red_deltav, red_cen_mstar_sigma, blue_deltav, \
    blue_cen_mstar_sigma = get_stacked_velocity_dispersion(eco_nobuff, 'data')

sigma_red_data = bs(red_cen_mstar_sigma, red_deltav,
    statistic='std', bins=np.linspace(8.9,11.5,5))
sigma_blue_data = bs( blue_cen_mstar_sigma, blue_deltav,
    statistic='std', bins=np.linspace(8.9,11.5,5))

# counter = Counter(sigma_red_data[2]).most_common()
# red_poisson_error = np.log10(np.sqrt(np.flip(np.array(list(counter)))[:,0]))

# counter = Counter(sigma_blue_data[2]).most_common()
# blue_poisson_error = np.log10(np.sqrt(np.flip(np.array(list(counter)))[:,0]))

sigma_red_data = np.log10(sigma_red_data[0])
sigma_blue_data = np.log10(sigma_blue_data[0])

red_sigma, red_cen_mstar_sigma, blue_sigma, \
    blue_cen_mstar_sigma = get_velocity_dispersion(eco_nobuff, 'data')

red_sigma = np.log10(red_sigma)
blue_sigma = np.log10(blue_sigma)

mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))
mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
    statistic=average_of_log, bins=np.linspace(1,3,5))

#* Bootstrap errors
from tqdm import tqdm
iterations = 100
sigma_red_data_global = []
sigma_blue_data_global = []
mean_mstar_red_data_global = []
mean_mstar_blue_data_global = []
for i in tqdm(range(iterations)):       
    galtype_col = 'fc'
    id_col = 'grp'

    all_group_ids = np.unique(eco_nobuff[id_col].loc[eco_nobuff[galtype_col] == 1].values) 
    n_group_samples = len(all_group_ids)

    sample = np.array(random.choices(all_group_ids, k=n_group_samples))
    df = eco_nobuff.loc[eco_nobuff[id_col].isin(sample)]
    
    red_deltav, red_cen_mstar_sigma, blue_deltav, \
        blue_cen_mstar_sigma = get_stacked_velocity_dispersion(df, 'data')

    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
        statistic='std', bins=np.linspace(8.9,11.5,5))
    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
        statistic='std', bins=np.linspace(8.9,11.5,5))

    sigma_red = np.log10(sigma_red[0])
    sigma_blue = np.log10(sigma_blue[0])

    sigma_red_data_global.append(sigma_red)
    sigma_blue_data_global.append(sigma_blue)

    red_sigma, red_cen_mstar_sigma, blue_sigma, \
        blue_cen_mstar_sigma = get_velocity_dispersion(df, 'data')

    red_sigma = np.log10(red_sigma)
    blue_sigma = np.log10(blue_sigma)

    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,3,5))
    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
        statistic=average_of_log, bins=np.linspace(1,3,5))

    mean_mstar_red_counts = bs(red_sigma, red_cen_mstar_sigma, 
        statistic='count', bins=np.linspace(1,3,5))
    mean_mstar_blue_counts = bs(blue_sigma, blue_cen_mstar_sigma, 
        statistic='count', bins=np.linspace(1,3,5))

    mean_mstar_red_data_global.append(mean_mstar_red[0])
    mean_mstar_blue_data_global.append(mean_mstar_blue[0])

mean_mstar_red_err = np.nanstd(mean_mstar_red_data_global, axis=0)
mean_mstar_blue_err = np.nanstd(mean_mstar_blue_data_global, axis=0)

sigma_red_err = np.nanstd(sigma_red_data_global, axis=0)
sigma_blue_err = np.nanstd(sigma_blue_data_global, axis=0)

##Plot

rc('axes', linewidth=4)
fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharex=False, sharey=False, 
    gridspec_kw={'wspace':0.25})

bins = np.linspace(8.9,11.5,5)
bin_centers = 0.5 * (bins[1:] + bins[:-1])


ax[0].errorbar(bin_centers, sigma_red_data, yerr=sigma_red_err,
    color='indianred', fmt='s-', ecolor='darkred', mfc='darkred', mec='darkred',
    markersize=17, capsize=10, elinewidth=5,
    capthick=2.0, zorder=10, marker='^', ls='--', lw=7)
ax[0].errorbar(bin_centers, sigma_blue_data, yerr=sigma_blue_err,
    color='cornflowerblue', fmt='s-', ecolor='darkblue', mfc='darkblue',
    mec='darkblue', markersize=17, elinewidth=5,
    capsize=10, capthick=2.0, zorder=12, marker='^', ls='--', lw=7)

# ax[0].plot(bin_centers, sigma_red_data,
#     color='indianred', zorder=10, ls='--', lw=7)
# ax[0].plot(bin_centers, sigma_blue_data,
#     color='cornflowerblue', zorder=10, ls='--', lw=7)
# ax[0].scatter(bin_centers, sigma_red_data,
#     color='indianred', marker='s', s=400)
# ax[0].scatter(bin_centers, sigma_blue_data,
#     color='cornflowerblue', marker='s', s=400)

bins_red = mean_mstar_red_data[1]
bin_centers_red = 0.5 * (bins_red[1:] + bins_red[:-1])

bins_blue = mean_mstar_blue_data[1]
bin_centers_blue = 0.5 * (bins_blue[1:] + bins_blue[:-1])

# counter = Counter(mean_mstar_red_data[2]).most_common()
# red_poisson_error = np.log10(np.sqrt(np.flip(np.array(list(counter)))[:,0]))

# #skipping the last one in the Counter where bin idx = 0 because that is 
# #where sigma blue is below the left most bin's left edge. (Only one galaxy too)
# counter = Counter(mean_mstar_blue_data[2]).most_common()[:-1]
# blue_poisson_error = np.log10(np.sqrt(np.flip(np.array(list(counter)))[:,0]))

ax[1].errorbar(bin_centers_red, mean_mstar_red_data[0], 
    yerr=mean_mstar_red_err, color='indianred', fmt='s-', 
    ecolor='darkred', mfc='darkred', mec='darkred', markersize=17, elinewidth=5, 
    capsize=10, capthick=2.0, 
    zorder=10, marker='^', ls='--', lw=7)
ax[1].errorbar(bin_centers_blue, mean_mstar_blue_data[0],
    yerr=mean_mstar_blue_err, color='cornflowerblue', fmt='s-', ecolor='darkblue',
    mfc='darkblue', mec='darkblue', markersize=17, elinewidth=5, capsize=10, capthick=2.0, 
    zorder=12, marker='^', ls='--', lw=7)

# avmstar_red, = ax[1].plot(bin_centers_red, mean_mstar_red_data[0], 
#     color='indianred', zorder=10, ls='--', lw=7)
# avmstar_blue, = ax[1].plot(bin_centers_blue, mean_mstar_blue_data[0],
#     color='cornflowerblue', zorder=10, ls='--', lw=7)
# ax[1].scatter(bin_centers_red, mean_mstar_red_data[0], 
#     color='indianred', marker='s', s=400)
# ax[1].scatter(bin_centers_blue, mean_mstar_blue_data[0],
#     color='cornflowerblue', marker='s', s=400)

ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_{*, group\ cen} \left[\mathrm{M_\odot}\ \right]$', fontsize=35, labelpad=15)
ax[1].set_xlabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km\ s^{-1}} \right]$', fontsize=35, labelpad=15)

ax[0].set_ylabel(r'\boldmath$\log_{10}\ \sigma \left[\mathrm{km/s} \right]$', fontsize=35, labelpad=15)
ax[1].set_ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, group\ cen}} \left[\mathrm{M_\odot}\ \right]$', fontsize=35, labelpad=15)

ax[0].minorticks_on()
ax[1].minorticks_on()

plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/eco_sigma_mstar_panel.pdf', 
    bbox_inches="tight", dpi=1200)

# plt.savefig('/Users/asadm2/Desktop/eco_sigma_mstar_panel_origbin.pdf', 
#     bbox_inches="tight", dpi=1200)

plt.show()





