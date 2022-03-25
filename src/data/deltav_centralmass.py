#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:45:22 2018

@author: asadm2
"""

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=15)
rc('text', usetex=True)
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
eco_dr2 = pd.read_csv('/Users/asadm2/Desktop/ecodr2.csv')
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
        deltav = cz_arr - cen_cz_grp

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
    deltav = group.cz.values - len(group)*[cen_cz_grp]
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
    deltav = group.cz.values - len(group)*[cen_cz_grp]
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
