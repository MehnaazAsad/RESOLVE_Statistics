#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:22:01 2019

@author: asadm2
"""

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from collections import Counter
from scipy import interpolate
from matplotlib import rc
import pandas as pd
import numpy as np
import math

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def cumu_num_dens(data,nbins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=nbins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not bool_mag:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,edg,n_cumu,err_poiss,bin_width

def get_grp_df(df,groupby_keyword):
    grps = df.groupby(groupby_keyword)
    grp_keys = grps.groups.keys()
    total_mstar_grp_arr = []
    grp_logmhs_arr = []
    grp_n_arr = []
    key_arr = []
    fc_arr = []
    for key in grp_keys:
        group = grps.get_group(key)
        total_mstar_grp = [np.log10((10**group.logmstar.values).sum())]*len(group)
        grp_logmhs = [group.logmh_s.values[0]]*len(group)
        grp_n = [group.grpn.values[0]]*len(group)
        key = [key]*len(group)
        fc = group.fc.values
        total_mstar_grp_arr.extend(total_mstar_grp)
        grp_logmhs_arr.extend(grp_logmhs)
        grp_n_arr.extend(grp_n)
        key_arr.extend(key)
        fc_arr.extend(fc)
    grp_data = np.array(list(zip(key_arr,total_mstar_grp_arr,grp_logmhs_arr,\
                                 grp_n_arr,fc_arr)))
    grp_df = pd.DataFrame(grp_data,columns=['grp','grpms','logmh_s','grpn',\
                                            'fc'])
    return grp_df


### Paths
path_to_raw = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/raw/'
path_to_interim = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/interim/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

columns = ['name','radeg','dedeg','cz','grpcz','logmstar','logmgas','grp',\
           'grpn','grpnassoc','logmh','logmh_s','fc','grpmb','grpms','f_a','f_b']

#2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv", \
                             delimiter=",", header=0, usecols=columns)

grps = resolve_live18.groupby('grp') #group by group ID
grp_keys = grps.groups.keys()

#Isolating groups that don't have designated central
grp_id_no_central_arr = []
for key in grp_keys:
    group = grps.get_group(key)
    if 1 not in group.fc.values:
        grp_id = group.grp.values
        grp_id_no_central_arr.append(np.unique(grp_id)[0])

resolve_live18_subset = resolve_live18.loc[~resolve_live18['grp'].\
                                           isin(grp_id_no_central_arr)]

resolve_live18_subset = resolve_live18_subset.loc[(resolve_live18_subset\
                                                   ['grpcz'] >= 4500) & \
                                                    (resolve_live18_subset\
                                                     ['grpcz'] <= 6000)]

v_resolve = 52100/2.915 #(Mpc/h)^3 (h from 0.7 to 1) *Eckert 2015*

### Halo data (not in log)
halo_table = pd.read_csv('../../data/interim/id_macc.csv',header=0)
v_sim = 130**3 #(Mpc/h)^3

halo_mass_c = halo_table.halo_macc.loc[halo_table.C_S.values==1].values
nbins = num_bins(halo_mass_c)
bin_centers_hmassc,bin_edges_hmassc,n_hmassc,err_poiss_hmassc,\
bin_width_hmassc = cumu_num_dens(halo_mass_c,nbins,None,v_sim,False)

### Group integrated stellar mass for HAM
grpms_c = 10**(resolve_live18_subset.grpms.loc[resolve_live18_subset.fc.\
                                               values==1])
nbins = num_bins(grpms_c)
bin_centers_grpmsc,bin_edges_grpmsc,n_grpmsc,err_poiss_grpmsc,bin_width_grpmsc = \
cumu_num_dens(grpms_c,nbins,None,v_resolve,False)


grpmsc_n_interp_func = interpolate.interp1d(bin_centers_grpmsc,n_grpmsc,\
                                            fill_value='extrapolate')

hmassc_n_interp_func = interpolate.interp1d(n_hmassc,bin_centers_hmassc,\
                                            fill_value='extrapolate')

pbar = ProgressBar(maxval=len(grpms_c))
n_grpmsc_arr = [grpmsc_n_interp_func(val) for val in pbar(grpms_c)]
pbar = ProgressBar(maxval=len(n_grpmsc_arr))
hmassc_grpmsc_ham = [hmassc_n_interp_func(val) for val in pbar(n_grpmsc_arr)]

hmassc_loggrpmsc = np.log10(hmassc_grpmsc_ham)

loggrpmsc = np.log10(grpms_c)

x_grpmsc,y_grpmsc,y_std_grpmsc,y_std_err_grpmsc = Stats_one_arr\
(hmassc_loggrpmsc,loggrpmsc,base=0.3)

x_mhsc,y_mhsc,y_std_mhsc,y_std_err_mhsc = Stats_one_arr\
(resolve_live18_subset.logmh_s.loc[resolve_live18_subset.fc.values==1],\
 loggrpmsc,base=0.3)

fig1 = plt.figure(figsize=(10,10))
plt.errorbar(x_grpmsc,y_grpmsc,yerr=y_std_grpmsc,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label='Group mstar central')
plt.errorbar(x_mhsc,y_mhsc,yerr=y_std_mhsc,\
             color='#a0298d',fmt='--s',ecolor='#a0298d',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE stellar mass derived')
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{*,grp} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 10})


## Debugging plot of group SMHM
grpms_counter = Counter(resolve_live18_subset.grpms.values)
grpms_most_common = grpms_counter.most_common(10)

#Get all groups in top line 
top_horizontal_line_groups = resolve_live18_subset.loc[resolve_live18_subset.\
                                                       grpms.values==\
                                                       grpms_most_common[1][0]]
num_groups_topline = len(np.unique(top_horizontal_line_groups.grp.values))

#Get all groups in bottom line 
bottom_horizontal_line_groups = resolve_live18_subset.loc[resolve_live18_subset.\
                                                       grpms.values==\
                                                       grpms_most_common[0][0]]
num_groups_topline = len(np.unique(bottom_horizontal_line_groups.grp.values))

top_grp_df = get_grp_df(top_horizontal_line_groups,'grp')
bottom_grp_df = get_grp_df(bottom_horizontal_line_groups,'grp')
all_grp_df = get_grp_df(resolve_live18_subset,'grp')

fig2 = plt.figure(figsize=(10,10))
plt.scatter(resolve_live18_subset.logmh_s.values,\
            resolve_live18_subset.grpms.values,color='#96c82b',s=12,\
            label='RESOLVE')
plt.scatter(top_grp_df.logmh_s.values,top_grp_df.grpms.values,color='#670033',\
            s=12,label='top row')
plt.scatter(bottom_grp_df.logmh_s.values,bottom_grp_df.grpms.values,\
            color='#f34b19',s=12,label='bottom line')
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{*,grp} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 15})

fig3 = plt.figure(figsize=(10,10))
plt.scatter(resolve_live18_subset.logmh_s.values,\
            resolve_live18_subset.grpms.values,color='#1c2575',s=12,\
            label='RESOLVE (before)')
plt.scatter(all_grp_df.logmh_s.values,all_grp_df.grpms.values,color='#a0a395',\
            s=8,label='all data (after)')
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{*,grp} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 15})

### Using my calculated grpms values to do abundance matching
grpms_c_new = 10**(all_grp_df.grpms.loc[all_grp_df.fc.values==1])
nbins = num_bins(grpms_c_new)
bin_centers_grpmsc_new,bin_edges_grpmsc_new,n_grpmsc_new,err_poiss_grpmsc_new,\
bin_width_grpmsc_new = cumu_num_dens(grpms_c_new,nbins,None,v_resolve,False)

grpmsc_n_interp_func_new = interpolate.interp1d(bin_centers_grpmsc_new,\
                                                n_grpmsc_new,\
                                                fill_value='extrapolate')

pbar = ProgressBar(maxval=len(grpms_c_new))
n_grpmsc_new_arr = [grpmsc_n_interp_func_new(val) for val in pbar(grpms_c_new)]
pbar = ProgressBar(maxval=len(n_grpmsc_new_arr))
hmassc_grpmsc_ham_new = [hmassc_n_interp_func(val) for val in \
                         pbar(n_grpmsc_new_arr)]

hmassc_loggrpmsc_new = np.log10(hmassc_grpmsc_ham_new)
loggrpmsc_new = np.log10(grpms_c_new)

x_grpmsc_new,y_grpmsc_new,y_std_grpmsc_new,y_std_err_grpmsc_new = Stats_one_arr\
(hmassc_loggrpmsc_new,loggrpmsc_new,base=0.3)

x_mhsc_new,y_mhsc_new,y_std_mhsc_new,y_std_err_mhsc_new = Stats_one_arr\
(all_grp_df.logmh_s.loc[all_grp_df.fc.values==1],\
 loggrpmsc_new,base=0.3)


fig4 = plt.figure(figsize=(10,10))
plt.errorbar(x_grpmsc,y_grpmsc,yerr=y_std_grpmsc,\
             color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,\
             capthick=0.5,label='Group mstar central')
plt.errorbar(x_grpmsc_new,y_grpmsc_new,yerr=y_std_grpmsc_new,\
             color='#442b88',fmt='--s',ecolor='#442b88',markersize=4,capsize=5,\
             capthick=0.5,label='Group mstar central new')
plt.errorbar(x_mhsc,y_mhsc,yerr=y_std_mhsc,\
             color='#a3cd3b',fmt='--s',ecolor='#a3cd3b',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE stellar mass derived')
plt.errorbar(x_mhsc_new,y_mhsc_new,yerr=y_std_mhsc_new,\
             color='#20a74e',fmt='--s',ecolor='#20a74e',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE stellar mass derived new')
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{*,grp} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 10})

