#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:45:22 2018

@author: asadm2
"""

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np

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

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']


resolve_df = pd.read_csv(path_to_raw + 'RESOLVE_liveJune2018.csv', \
                         delimiter=',',header=0)

resolve_groups = resolve_df.groupby('grp')
resolve_keys = resolve_groups.groups.keys()
deltav_arr = []
cen_stellar_mass_arr = []
cen_colour_arr = []
grpn_arr = []
for index,key in enumerate(resolve_keys):
    group = resolve_groups.get_group(key)
    cz_arr = group.cz.values
    grpcz = np.unique(group.grpcz.values)
    deltav = cz_arr-grpcz
    if 1 in group.fc.values:
        cen_stellar_mass = group.logmstar.loc[group.fc.values == 1].values[0]
        cen_colour = group.modelg_rcorr.loc[group.fc.values == 1].values[0]
        grp_N = group.grpn.loc[group.fc.values == 1].values[0]
    else:
        continue
    for val in deltav:
        deltav_arr.append(val)
        cen_stellar_mass_arr.append(cen_stellar_mass)
        cen_colour_arr.append(cen_colour)
        grpn_arr.append(grp_N)
       
#fig1 = plt.figure(figsize=(10,10))
#plt.scatter(cen_stellar_mass_arr,deltav_arr,s=7,c=cen_colour_arr,cmap='RdBu_r')
#cbar = plt.colorbar()
#cbar.set_label(r'$\mathbf{central\ g-r}$')
#plt.xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$')
#plt.ylabel(r'\boldmath$\Delta v\ \left[km/s\right]$')

#fig2 = plt.figure(figsize=(10,10))
#plt.scatter(grpn_arr,deltav_arr,s=7,c=cen_colour_arr,cmap='RdBu_r')
#cbar = plt.colorbar()
#cbar.set_label(r'$\mathbf{central\ g-r}$')
#plt.xlabel(r'$\mathbf{N}$')
#plt.ylabel(r'\boldmath$\Delta v\ \left[km/s\right]$')

data = {'deltav': deltav_arr, 'log_cen_stellar_mass': cen_stellar_mass_arr,\
        'cen_gr_colour': cen_colour_arr}
vel_col_mass_df = pd.DataFrame(data)
df_blue_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_gr_colour'] <= 0.7]
df_red_cen = vel_col_mass_df.loc[vel_col_mass_df['cen_gr_colour'] > 0.7]
 
# Stats_one_arr returns std from the mean   
x_blue,y_blue,y_std_blue,y_std_err_blue = Stats_one_arr(df_blue_cen.\
                                                        log_cen_stellar_mass,\
                                                        df_blue_cen.deltav,\
                                                        base=0.5,\
                                                        statfunc=np.std,\
                                                        bin_statval='left')

x_red,y_red,y_std_red,y_std_err_red = Stats_one_arr(df_red_cen.\
                                                    log_cen_stellar_mass,\
                                                    df_red_cen.deltav,\
                                                    base=0.5,statfunc=np.std,\
                                                    bin_statval='left')

# Calculate same as above but std from 0 km/s
stellar_mass_bins = np.arange(7,12,0.5)
i = 0
std_blue_cen_arr = []
err_std_blue_cen_arr = [] 
for index1,bin_edge in enumerate(stellar_mass_bins):
    df_blue_cen_deltav_arr = []
    if index1 == 9:
        break
    for index2,stellar_mass in enumerate(df_blue_cen.log_cen_stellar_mass.values):
        if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
            df_blue_cen_deltav_arr.append(df_blue_cen.deltav.values[index2])
    N = len(df_blue_cen_deltav_arr)
    std = std_func(df_blue_cen_deltav_arr)
    err_std = std/np.sqrt(2*N)
    std_blue_cen_arr.append(std)
    err_std_blue_cen_arr.append(err_std)

i = 0
std_red_cen_arr = []
err_std_red_cen_arr = []
for index1,bin_edge in enumerate(stellar_mass_bins):
    df_red_cen_deltav_arr = []
    if index1 == 9:
        break
    for index2,stellar_mass in enumerate(df_red_cen.log_cen_stellar_mass.values):
        if stellar_mass >= bin_edge and stellar_mass < stellar_mass_bins[index1+1]:
            df_red_cen_deltav_arr.append(df_red_cen.deltav.values[index2])
    N = len(df_red_cen_deltav_arr)
    std = std_func(df_red_cen_deltav_arr)
    err_std = std/np.sqrt(2*N)
    std_red_cen_arr.append(std)
    err_std_red_cen_arr.append(err_std)
        
fig3,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=False,figsize=(20,20),\
     gridspec_kw = {'height_ratios':[6.5,3.5]})
fig3.subplots_adjust(right=0.8)
main = ax1.scatter(cen_stellar_mass_arr,deltav_arr,s=7,c=cen_colour_arr,cmap='RdBu_r')
cbar_ax = fig3.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = plt.colorbar(main,ax=ax1,cax=cbar_ax)
cbar.set_label(r'$\mathbf{central\ g-r}$')
ax2.errorbar(stellar_mass_bins[:-1],std_blue_cen_arr,yerr=err_std_blue_cen_arr,\
             color='#2d77e5',fmt='s',ecolor='#2d77e5',markersize=4,capsize=5,\
             capthick=0.5)
ax2.errorbar(stellar_mass_bins[:-1],std_red_cen_arr,yerr=err_std_red_cen_arr,\
             color='#f46542',fmt='s',ecolor='#f46542',markersize=4,capsize=5,\
             capthick=0.5)
ax2.set_xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$')
ax1.set_ylabel(r'\boldmath$\Delta v\ \left[km/s\right]$')
ax2.set_ylabel(r'\boldmath$\sigma\ \left[km/s\right]$')
ax2.text(7.5, 153,r'$\mathbf{(red)\ 0.7 < g-r \leq 0.7 (blue)}$',\
         {'color': 'k', 'fontsize': 15, 'ha': 'center', 'va': 'center',\
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})

fig4 = plt.figure(figsize=(10,10))
plt.errorbar(stellar_mass_bins[:-1],std_blue_cen_arr,yerr=err_std_blue_cen_arr,\
             color='#2d77e5',fmt='s',ecolor='#2d77e5',markersize=4,capsize=5,\
             capthick=0.5)
plt.errorbar(stellar_mass_bins[:-1],std_red_cen_arr,yerr=err_std_red_cen_arr,\
             color='#f46542',fmt='s',ecolor='#f46542',markersize=4,capsize=5,\
             capthick=0.5)
plt.xlabel(r'$\mathbf{log\ M_{*,cen}}\ [\mathbf{M_{\odot}}]$')
plt.ylabel(r'\boldmath$\sigma\ \left[km/s\right]$')
