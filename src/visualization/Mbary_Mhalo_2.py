#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:48:41 2018

@author: asadm2
"""

from cosmo_utils.utils.stats_funcs import Stats_one_arr
from progressbar import ProgressBar
import matplotlib.pyplot as plt
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

def centrals_satellites_flag(halo_table):
    N_halo = len(halo_table)

    ID   = halo_table['halo_id']
    UPID = halo_table['halo_upid' ]
    
    # Arrrays #
    C_S_FLAG = [ [] for x in range(N_halo)]
    HALO_ID  = [ [] for x in range(N_halo)]
        
    pbar = ProgressBar(maxval=N_halo)
    for Halo_Cat in pbar(range(0, N_halo)):
    	if UPID[Halo_Cat] == -1:
    		C_S_FLAG[Halo_Cat] = 1
    		HALO_ID[Halo_Cat] = int(ID[Halo_Cat])
    	else:
    		C_S_FLAG[Halo_Cat] = 0
    		HALO_ID [Halo_Cat] = int(UPID[Halo_Cat])
    return C_S_FLAG

### Paths
path_to_raw = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/raw/'
path_to_interim = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/interim/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

columns = ['name','radeg','dedeg','cz','grpcz','logmstar','logmgas','grp',\
           'grpn','logmh','logmh_s','fc','grpmb','grpms','f_a','f_b']

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

grps = resolve_live18_subset.groupby('grp') 
grp_keys = grps.groups.keys()

mstar_c_arr = []
for key in grp_keys:
    group = grps.get_group(key)
    if 1 in group.fc.values:
        mstar_c = group.logmstar.loc[group.fc.values == 1].values[0]
        temp_arr = [mstar_c]*len(group)
    else:
        temp_arr = [0]*len(group)
    mstar_c_arr.extend(temp_arr)
mstar_c_arr = np.array(mstar_c_arr)
    
mgas_c_arr = []
for key in grp_keys:
    group = grps.get_group(key)
    if 1 in group.fc.values:
        mgas_c = group.logmgas.loc[group.fc.values == 1].values[0]
        temp_arr = [mgas_c]*len(group)
    else:
        temp_arr = [0]*len(group)
    mgas_c_arr.extend(temp_arr)
mgas_c_arr = np.array(mgas_c_arr)

centrals_resolve = resolve_live18_subset.loc[resolve_live18_subset['fc'] == 1] #1594 centrals

RESOLVE_B = centrals_resolve.loc[centrals_resolve['f_b'] == 1] #537 galaxies
RESOLVE_A = centrals_resolve.loc[centrals_resolve['f_a'] == 1] #1057 galaxies
RESOLVE_B = RESOLVE_B.reset_index(drop=True)
RESOLVE_A = RESOLVE_A.reset_index(drop=True)

v_resolve = 50000/2.915 #(Mpc/h)^3 (h from 0.7 to 1)

### Group integrated baryonic mass for HAM
grpmb = 10**(resolve_live18_subset.grpmb.values)
nbins = num_bins(grpmb)
bin_centers_grpmb,bin_edges_grpmb,n_grpmb,err_poiss_grpmb,bin_width_grpmb = \
cumu_num_dens(grpmb,nbins,None,v_resolve,False)

### Using baryonic mass of each galaxy's group central for HAM
mstar_c = 10**mstar_c_arr
mgas_c = 10**mgas_c_arr
mbary_c = mstar_c + mgas_c
nbins = num_bins(mbary_c)
bin_centers_mbaryc,bin_edges_mbaryc,n_mbaryc,err_poiss_mbaryc,\
bin_width_mbaryc = cumu_num_dens(mbary_c,nbins,None,v_resolve,False)

### Group integrated stellar mass for HAM
grpms = 10**(resolve_live18_subset.grpms.values)
nbins = num_bins(grpms)
bin_centers_grpms,bin_edges_grpms,n_grpms,err_poiss_grpms,bin_width_grpms = \
cumu_num_dens(grpms,nbins,None,v_resolve,False)

### Using stellar mass of each galaxy's group central for HAM
mstar_c = 10**mstar_c_arr
nbins = num_bins(mstar_c)
bin_centers_mstarc,bin_edges_mstarc,n_mstarc,err_poiss_mstarc,\
bin_width_mstarc = cumu_num_dens(mstar_c,nbins,None,v_resolve,False)

### Halo data (not in log)
halo_table = pd.read_csv('../../data/interim/id_macc.csv',header=0)
v_sim = 130**3 #(Mpc/h)^3

halo_mass_c = halo_table.halo_macc.loc[halo_table.C_S.values==1].values
nbins = num_bins(halo_mass_c)
bin_centers_hmassc,bin_edges_hmassc,n_hmassc,err_poiss_hmassc,\
bin_width_hmassc = cumu_num_dens(halo_mass_c,nbins,None,v_sim,False)

############################### IN PROGRESS (HAM) ############################
### Interpolating between grpmb and n
grpmb_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb,\
                                           fill_value='extrapolate')
### Interpolating between central mbary and n
mbaryc_n_interp_func = interpolate.interp1d(bin_centers_mbaryc,n_mbaryc,\
                                            fill_value='extrapolate')

### Interpolating between grpmstar and n
grpms_n_interp_func = interpolate.interp1d(bin_centers_grpms,n_grpms,\
                                           fill_value='extrapolate')
### Interpolating between central mstar and n
mstarc_n_interp_func = interpolate.interp1d(bin_centers_mstarc,n_mstarc,\
                                            fill_value='extrapolate')

### Interpolating between central hmass and n and reversing it so you can pass 
### an n and get central hmass value
hmassc_n_interp_func = interpolate.interp1d\
(n_hmassc,bin_centers_hmassc,fill_value='extrapolate')

### HAM
pbar = ProgressBar(maxval=len(grpmb))
n_grpmb_arr = [grpmb_n_interp_func(val) for val in pbar(grpmb)]
pbar = ProgressBar(maxval=len(n_grpmb_arr))
hmass_grpmb_ham = [hmassc_n_interp_func(val) for val in pbar(n_grpmb_arr)]

pbar = ProgressBar(maxval=len(mbary_c))
n_mbaryc_arr = [mbaryc_n_interp_func(val) for val in pbar(mbary_c)]
pbar = ProgressBar(maxval=len(n_mbaryc_arr))
hmass_mbaryc_ham = [hmassc_n_interp_func(val) for val in pbar(n_mbaryc_arr)]

pbar = ProgressBar(maxval=len(grpms))
n_grpms_arr = [grpms_n_interp_func(val) for val in pbar(grpms)]
pbar = ProgressBar(maxval=len(n_grpms_arr))
hmass_grpms_ham = [hmassc_n_interp_func(val) for val in pbar(n_grpms_arr)]

pbar = ProgressBar(maxval=len(mstar_c))
n_mstarc_arr = [mstarc_n_interp_func(val) for val in pbar(mstar_c)]
pbar = ProgressBar(maxval=len(n_mstarc_arr))
hmass_mstarc_ham = [hmassc_n_interp_func(val) for val in pbar(n_mstarc_arr)]

### Convert to log
hmass_logmbaryc = np.log10(hmass_mbaryc_ham)
hmass_loggrpmb = np.log10(hmass_grpmb_ham)
logmbaryc = np.log10(mbary_c)
loggrpmb = np.log10(grpmb)

hmass_logmstarc = np.log10(hmass_mstarc_ham)
hmass_loggrpms = np.log10(hmass_grpms_ham)
logmstarc = np.log10(mstar_c)
loggrpms = np.log10(grpms)

### Get error bars
x_mbaryc,y_mbaryc,y_std_mbaryc,y_std_err_mbaryc = Stats_one_arr(hmass_logmbaryc,\
                                                                logmbaryc,\
                                                                base=0.05)
x_grpmb,y_grpmb,y_std_grpmb,y_std_err_grpmb = Stats_one_arr(hmass_loggrpmb,\
                                                            logmbaryc,base=0.05)

x_mstarc,y_mstarc,y_std_mstarc,y_std_err_mstarc = Stats_one_arr(hmass_logmstarc,\
                                                                logmbaryc,\
                                                                base=0.05)
x_grpms,y_grpms,y_std_grpms,y_std_err_grpms = Stats_one_arr(hmass_loggrpms,\
                                                            logmbaryc,\
                                                            base=0.05)

x_mhs,y_mhs,y_std_mhs,y_std_err_mhs = Stats_one_arr(resolve_live18_subset.logmh_s.\
                                                    values,logmbaryc,base=0.05)
x_mh,y_mh,y_std_mh,y_std_err_mh = Stats_one_arr(resolve_live18_subset.logmh.values,\
                                                logmbaryc,base=0.05)
### Plotting BMHM and SMHM from HAM
fig1 = plt.figure(figsize=(10,10))
#plt.scatter(hmass_loggrpmb,logmbaryc,color='#1baeab',s=5)
#plt.scatter(hmass_logmbaryc,logmbaryc,color='#f6a631',s=5)
#plt.scatter(hmass_logmstarc,logmbaryc,color='k',s=5)
#plt.scatter(hmass_loggrpms,logmbaryc,color='r',s=5)
#plt.scatter(resolve_live18_subset.logmh_s.values,logmbaryc,color='b',s=5)
#plt.scatter(resolve_live18_subset.logmh.values,logmbaryc,color='g',s=5)

plt.errorbar(x_grpmb,y_grpmb,yerr=y_std_grpmb,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label='Group mbary')
plt.errorbar(x_mbaryc,y_mbaryc,yerr=y_std_mbaryc,\
             color='#f6a631',fmt='s',ecolor='#f6a631',markersize=4,capsize=5,\
             capthick=0.5,label='Central mbary')
plt.errorbar(x_mhs,y_mhs,yerr=y_std_mhs,\
             color='#a0298d',fmt='s',ecolor='#a0298d',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE stellar mass derived')
plt.errorbar(x_mh,y_mh,yerr=y_std_mh,\
             color='k',fmt='s',ecolor='k',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE rband lum derived')
plt.xlabel(r'Halo mass')
plt.ylabel(r'Baryonic mass')
plt.legend(loc='best',prop={'size': 10})

#x_mhs,y_mhs,y_std_mhs,y_std_err_mhs = Stats_one_arr(resolve_live18.logmh_s.\
#                                                    values,resolve_live18.\
#                                                    grpms.values,base=0.05)
#
#fig2 = plt.figure(figsize=(10,10))
#plt.scatter(hmass_loggrpms,loggrpms,color='#1baeab',s=5)
#plt.scatter(hmass_logmstarc,logmstarc,color='#f6a631',s=5)
#plt.errorbar(x_grpms,y_grpms,yerr=y_std_grpms,\
#             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
#             capthick=0.5,label='Group mstar')
#plt.errorbar(x_mstarc,y_mstarc,yerr=y_std_mstarc,\
#             color='#f6a631',fmt='s',ecolor='#f6a631',markersize=4,capsize=5,\
#             capthick=0.5,label='Central mstar')
#plt.errorbar(x_mhs,y_mhs,yerr=y_std_mhs,\
#             color='#a0298d',fmt='s',ecolor='#a0298d',markersize=4,capsize=5,\
#             capthick=0.5,label='RESOLVE stellar mass derived')
#plt.xlabel(r'Halo mass')
#plt.ylabel(r'Stellar mass')
#plt.legend(loc='best',prop={'size': 10})
'''
################################# SHAM ########################################
### Calculate baryonic mass of all galaxies for SHAM
mbary = (10**(resolve_live18.logmstar.values) + 10**(resolve_live18.logmgas.\
         values))
nbins = num_bins(mbary)
bin_centers_mbary,bin_edges_mbary,n_mbary,err_poiss_mbary,bin_width_mbary = \
cumu_num_dens(mbary,nbins,None,v_resolve,False)

### Stellar mass of all galaxies for SHAM
mstar = 10**(resolve_live18.logmstar.values)
nbins = num_bins(mstar)
bin_centers_mstar,bin_edges_mstar,n_mstar,err_poiss_mstar,bin_width_mstar = \
cumu_num_dens(mstar,nbins,None,v_resolve,False)

halo_mass = halo_table.halo_macc.values        
nbins = num_bins(halo_mass)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,bin_width_hmass = \
cumu_num_dens(halo_mass,nbins,None,v_sim,False)

### Interpolating between mbary and n
mbary_n_interp_func = interpolate.interp1d(bin_centers_mbary,n_mbary,\
                                           fill_value='extrapolate')

### Interpolating between smass and n
mstar_n_interp_func = interpolate.interp1d(bin_centers_mstar,n_mstar,\
                                           fill_value='extrapolate')

### Interpolating between hmass and n and reversing it so you can pass an n and
### get an hmass value
hmass_n_interp_func = interpolate.interp1d\
(n_hmass,bin_centers_hmass,fill_value='extrapolate')

### SHAM
pbar = ProgressBar(maxval=len(mbary))
n_mbary_arr = [mbary_n_interp_func(val) for val in pbar(mbary)]
pbar = ProgressBar(maxval=len(n_mbary_arr))
hmass_mbary_sham = [hmass_n_interp_func(val) for val in pbar(n_mbary_arr)]
pbar = ProgressBar(maxval=len(mstar))
n_mstar_arr = [mstar_n_interp_func(val) for val in pbar(mstar)]
pbar = ProgressBar(maxval=len(n_mstar_arr))
hmass_mstar_sham = [hmass_n_interp_func(val) for val in pbar(n_mstar_arr)]

### Convert to log
hmass_logmbary = np.log10(hmass_mbary_sham)
logmbary = np.log10(mbary)

hmass_logmstar = np.log10(hmass_mstar_sham)
logmstar = np.log10(mstar)

### Get error bars
x_mbary,y_mbary,y_std_mbary,y_std_err_mbary = Stats_one_arr(hmass_logmbary,\
                                                            logmbary,base=0.5)

x_mstar,y_mstar,y_std_mstar,y_std_err_mstar = Stats_one_arr(hmass_logmstar,\
                                                            logmstar,base=0.5)

### Plotting
fig3 = plt.figure(figsize=(10,10))
plt.scatter(hmass_logmbary,logmbary,color='#1baeab',s=5)
plt.errorbar(x_mbary,y_mbary,yerr=y_std_mbary,\
             color='#f6a631',fmt='s',ecolor='#f6a631',markersize=4,capsize=5,\
             capthick=0.5)
plt.xlabel('Halo mass')
plt.ylabel('Baryonic mass')
plt.legend(loc='best',prop={'size': 10})


x_mhs,y_mhs,y_std_mhs,y_std_err_mhs = Stats_one_arr(resolve_live18.logmh_s.\
                                                    values,resolve_live18.\
                                                    logmstar.values,base=0.5)

fig4 = plt.figure(figsize=(10,10))
plt.errorbar(x_mstar,y_mstar,yerr=y_std_mstar,\
             color='#f6a631',fmt='s',ecolor='#f6a631',markersize=4,capsize=5,\
             capthick=0.5)
plt.scatter(resolve_live18.logmh_s.values,resolve_live18.logmstar.values,\
            color='k',s=5)
plt.errorbar(x_mhs,y_mhs,yerr=y_std_mhs,\
             color='#a0298d',fmt='s',ecolor='#a0298d',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE stellar mass derived')
plt.xlabel('Halo mass')
plt.ylabel('Stellar mass')
plt.legend(loc='best',prop={'size': 10})
'''