#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:37:10 2018

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

### Paths
path_to_raw = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/raw/'
path_to_interim = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/interim/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=25)
rc('text', usetex=True)

columns = ['name','radeg','dedeg','cz','grpcz','logmstar','logmgas','grp',\
           'grpn','logmh','logmh_s','fc','grpmb','grpms','f_a','f_b']

#2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv", \
                             delimiter=",", header=0,usecols=columns)

centrals_resolve = resolve_live18.loc[resolve_live18['fc'] == 1] #1594 centrals

RESOLVE_B = centrals_resolve.loc[centrals_resolve['f_b'] == 1] #537 galaxies
RESOLVE_A = centrals_resolve.loc[centrals_resolve['f_a'] == 1] #1057 galaxies
RESOLVE_B = RESOLVE_B.reset_index(drop=True)
RESOLVE_A = RESOLVE_A.reset_index(drop=True)

### Baryonic mass and BMF
mbary = (10**(resolve_live18.logmstar.values) + 10**(resolve_live18.logmgas.values))
v_resolve = 50000/2.915 #(Mpc/h)^3 (h from 0.7 to 1)
nbins = num_bins(mbary)
bin_centers_mbary,bin_edges_mbary,n_mbary,err_poiss_mbary,bin_width_mbary = \
cumu_num_dens(mbary,nbins,None,v_resolve,False)

### Stellar mass and SMF
stellarmass = 10**(resolve_live18.logmstar.values)
nbins = num_bins(stellarmass)
bin_centers_smass,bin_edges_smass,n_smass,err_poiss_smass,bin_width_smass = \
cumu_num_dens(stellarmass,nbins,None,v_resolve,False)

### Group integrated baryonic mass and group integrated BMF
grpmb = 10**(resolve_live18.grpmb.values)
nbins = num_bins(grpmb)
bin_centers_grpmb,bin_edges_grpmb,n_grpmb,err_poiss_grpmb,bin_width_grpmb = \
cumu_num_dens(grpmb,nbins,None,v_resolve,False)

### Halo data
halo_table = pd.read_csv('../../data/interim/macc.csv',names=['halo_macc'])
v_sim = 130**3 #(Mpc/h)^3
halo_mass = halo_table.halo_macc.values
nbins = num_bins(halo_mass)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,bin_width_hmass = \
cumu_num_dens(halo_mass,nbins,None,v_sim,False)

### Interpolating between mbary and n
mbary_n_interp_func = interpolate.interp1d(bin_centers_mbary,n_mbary,fill_value='extrapolate')

### Interpolating between grpmb and n
grpmb_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb,fill_value='extrapolate')

### Interpolating between smass and n
smass_n_interp_func = interpolate.interp1d(bin_centers_smass,n_smass,fill_value='extrapolate')

### Interpolating between hmass and n and reversing it so you can pass an n and
### get an hmass value
hmass_n_interp_func = interpolate.interp1d\
(n_hmass,bin_centers_hmass,fill_value='extrapolate')

### Abundance matching individual galaxies
pbar = ProgressBar(maxval=len(mbary))
n_mbary_arr = [mbary_n_interp_func(val) for val in pbar(mbary)]
pbar = ProgressBar(maxval=len(stellarmass))
n_smass_arr = [smass_n_interp_func(val) for val in pbar(stellarmass)]

pbar = ProgressBar(maxval=len(n_mbary_arr))
hmass_mbary_sham = [hmass_n_interp_func(val) for val in pbar(n_mbary_arr)]
pbar = ProgressBar(maxval=len(n_smass_arr))
hmass_smass_sham = [hmass_n_interp_func(val) for val in pbar(n_smass_arr)]

### Abundance matching group baryonic masses
pbar = ProgressBar(maxval=len(grpmb))
n_grpmb_arr = [grpmb_n_interp_func(val) for val in pbar(grpmb)]

pbar = ProgressBar(maxval=len(n_grpmb_arr))
hmass_grpmb_sham = [hmass_n_interp_func(val) for val in pbar(n_grpmb_arr)]

### Log scaling
hmass_logmbary = np.log10(hmass_mbary_sham)
hmass_loggrpmb = np.log10(hmass_grpmb_sham)
hmass_logsmass = np.log10(hmass_smass_sham)
logmbary = np.log10(mbary)
logsmass = np.log10(stellarmass)

### Retrieving log values from resolve catalog
hmass_rlum = resolve_live18.logmh.values
hmass_stellarmass = resolve_live18.logmh_s.values
loggrpmb = resolve_live18.grpmb.values

### Using stats_one_arr to get error bars on mean
x_mbary,y_mbary,y_std_mbary,y_std_err_mbary = Stats_one_arr(hmass_logmbary,logmbary,\
                                                            base=0.5)

x_rlum,y_rlum,y_std_rlum,y_std_err_rlum = Stats_one_arr(hmass_rlum,loggrpmb,\
                                         base=0.5)

x_smass,y_smass,y_std_smass,y_std_err_smass = Stats_one_arr(hmass_stellarmass,loggrpmb,\
                                            base=0.5)

x_grpmb,y_grpmb,y_std_grpmb,y_std_err_grpmb = Stats_one_arr(hmass_loggrpmb,loggrpmb,\
                                                            base=0.5)

x_smass_sham,y_smass_sham,y_std_smass_sham,y_std_err_smass_sham = \
Stats_one_arr(hmass_logsmass,logsmass,base=0.5)


### Halo mass derived from different methods vs group baryonic mass
fig1 = plt.figure(figsize=(10,10))
plt.errorbar(x_mbary,y_mbary,yerr=y_std_mbary,\
             color='k',fmt='s',ecolor='k',markersize=4,capsize=5,\
             capthick=0.5,label='baryonic mass derived')
plt.errorbar(x_grpmb,y_grpmb,yerr=y_std_grpmb,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label='grp baryonic mass derived')
plt.errorbar(x_rlum,y_rlum,yerr=y_std_rlum,\
             color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
             capthick=0.5,label='grp r band luminosity derived')
plt.errorbar(x_smass,y_smass,yerr=y_std_smass,\
             color='#a0298d',fmt='--s',ecolor='#a0298d',markersize=4,capsize=5,\
             capthick=0.5,label='grp stellar mass derived')
plt.legend(loc='best',prop={'size': 10})
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{b,grp} \left[M_\odot \right]$')

### Plot of BMHM where logmbary is used to derive halo mass
fig2 = plt.figure(figsize=(10,10))
plt.errorbar(x_mbary,y_mbary,yerr=y_std_mbary,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label='baryonic mass derived')
plt.legend(loc='best',prop={'size': 10})
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_b \left[M_\odot \right]$')

### Plot of SMHM where stellar mass is used to derive halo mass
fig3 = plt.figure(figsize=(10,10))
plt.errorbar(x_smass_sham,y_smass_sham,yerr=y_std_smass_sham,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label='stellar mass derived')
plt.legend(loc='best',prop={'size': 10})
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_s \left[M_\odot \right]$')

