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

### Calculate baryonic mass
mbary = (10**(resolve_live18.logmstar.values) + 10**(resolve_live18.logmgas.values))
v_resolve = 50000/2.915 #(Mpc/h)^3 (h from 0.7 to 1)
nbins = num_bins(mbary)
bin_centers_mbary,bin_edges_mbary,n_mbary,err_poiss_mbary,bin_width_mbary = \
cumu_num_dens(mbary,nbins,None,v_resolve,False)

### Stellar mass from resovle
stellarmass = 10**(resolve_live18.logmstar.values)
nbins = num_bins(stellarmass)
bin_centers_smass,bin_edges_smass,n_smass,err_poiss_smass,bin_width_smass = \
cumu_num_dens(stellarmass,nbins,None,v_resolve,False)


### Group integrated baryonic mass
grpmb = 10**(resolve_live18.grpmb.values)
nbins = num_bins(grpmb)
bin_centers_grpmb,bin_edges_grpmb,n_grpmb,err_poiss_grpmb,bin_width_grpmb = \
cumu_num_dens(grpmb,nbins,None,v_resolve,False)

#### Group integrated baryonic mass for centrals
#grpmb_c = 10**(centrals_resolve.grpmb.values)
#nbins = num_bins(grpmb)
#bin_centers_grpmb_c,bin_edges_grpmb_c,n_grpmb_c,err_poiss_grpmb_c,bin_width_grpmb_c = \
#cumu_num_dens(grpmb_c,nbins,None,v_resolve,False)


### Halo data (not in log)
halo_table = pd.read_csv('../../data/interim/macc.csv',names=['halo_macc'])

v_sim = 130**3 #(Mpc/h)^3
halo_mass = halo_table.halo_macc.values
nbins = num_bins(halo_mass)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,bin_width_hmass = \
cumu_num_dens(halo_mass,nbins,None,v_sim,False)

### Interpolating between mbary and n
mbary_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb,fill_value='extrapolate')

### Interpolating between smass and n
smass_n_interp_func = interpolate.interp1d(bin_centers_smass,n_smass,fill_value='extrapolate')

### Interpolating between hmass and n and reversing it so you can pass an n and
### get an hmass value
hmass_n_interp_func = interpolate.interp1d\
(n_hmass,bin_centers_hmass,fill_value='extrapolate')

pbar = ProgressBar(maxval=len(mbary))
n_mbary_arr = [mbary_n_interp_func(val) for val in pbar(mbary)]
pbar = ProgressBar(maxval=len(n_mbary_arr))
hmass_mbary_sham = [hmass_n_interp_func(val) for val in pbar(n_mbary_arr)]
pbar = ProgressBar(maxval=len(stellarmass))
n_smass_arr = [smass_n_interp_func(val) for val in pbar(stellarmass)]
pbar = ProgressBar(maxval=len(n_smass_arr))
hmass_smass_sham = [hmass_n_interp_func(val) for val in pbar(n_smass_arr)]

fig3 = plt.figure(figsize=(10,10))
plt.scatter(np.log10(hmass_mbary_sham),np.log10(mbary),color='#1baeab',label='SHAM')
plt.xlabel('Halo mass')
plt.ylabel('Baryonic mass')

fig4 = plt.figure(figsize=(10,10))
plt.scatter(np.log10(hmass_smass_sham),np.log10(stellarmass),color='#1baeab',label='SHAM')
plt.scatter(resolve_live18.logmh_s.values,resolve_live18.logmstar.values,color='#f6a631',label='RESOLVE')
plt.xlabel('Halo mass')
plt.ylabel('Stellar mass')
plt.legend(loc='best',prop={'size': 10})
'''
############################### IN PROGRESS (HAM) ############################
mock_catalog = pd.read_hdf('../../../ECO_Resolve/ECO_Vishnu_mock_catalogues/data/ECO_Vishnu_mock_catalog_tiled_rsd.h5',\
                           key='mock_catalog_tiled')
v_sim = 130**3 #(Mpc/h)^3
centrals_halos = mock_catalog.loc[mock_catalog['C_S'] == 1] 
halo_mass_c = centrals_halos.halo_macc.values
nbins_hmass_c = num_bins(halo_mass_c)
bin_centers_hmass_c,bin_edges_hmass_c,n_hmass_c,err_poiss_hmass_c,bin_width_hmass_c = \
cumu_num_dens(halo_mass_c,nbins_hmass_c,None,v_sim,False)

### Interpolating between grpmb and n
grpmb_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb,fill_value='extrapolate')

### Interpolating between central hmass and n and reversing it so you can pass 
### an n and get central hmass value
c_hmass_n_interp_func = interpolate.interp1d\
(n_hmass_c,bin_centers_hmass_c,fill_value='extrapolate')

pbar = ProgressBar(maxval=len(grpmb))
n_grpmb_arr = [grpmb_n_interp_func(val) for val in pbar(grpmb)]
pbar = ProgressBar(maxval=len(n_grpmb_arr))
hmass_grpmb_ham = [c_hmass_n_interp_func(val) for val in pbar(n_grpmb_arr)]

fig1 = plt.figure(figsize=(10,10))
plt.scatter(np.log10(bin_centers_hmass_c),np.log10(n_hmass_c))
plt.title('HMF (centrals only)')

fig2 = plt.figure(figsize=(10,10))
plt.scatter(np.log10(bin_centers_grpmb),np.log10(n_grpmb))
plt.title('group BMF')
##############################################################################
'''
        


