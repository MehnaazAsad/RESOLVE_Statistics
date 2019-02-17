#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:39:34 2019

@author: asadm2
"""
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
from matplotlib import rc
import pandas as pd
import numpy as np
import math
#import os

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h = 2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def diff_SMF(mstar_arr,volume,cvar_err):
    logmstar_arr = np.log10((10**mstar_arr)/1.429) #changing from h=0.7 to h=1
    nbins = num_bins(logmstar_arr)  #Number of bins to divide data into
    #Unnormalized histogram and bin edges
    phi,edg = np.histogram(logmstar_arr,bins=nbins)  #paper used 17 bins
    dM = edg[1] - edg[0]  #Bin width
    maxis = 0.5*(edg[1:]+edg[:-1])  #Mass axis i.e. bin centers
    #Normalized to volume and bin width
    err_poiss = np.sqrt(phi)/(volume*dM)
    err_cvar = cvar_err/(volume*dM)
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    
    phi = phi/(volume*dM) 
    
    return maxis,phi,err_tot,nbins



### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
#halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
#                'vishnu/rockstar/vishnu_rockstar_test.hdf5'
halo_catalog = '/Users/asadm2/Desktop/vishnu_rockstar_test.hdf5'


###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

columns = ['name','radeg','dedeg','cz','grpcz','absrmag','logmstar','logmgas','grp',\
           'grpn','grpnassoc','logmh','logmh_s','fc','grpmb','grpms','f_a','f_b']

#2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv", \
                             delimiter=",", header=0, usecols=columns)

#956 galaxies
resolve_A = resolve_live18.loc[(resolve_live18.f_a.values == 1) & \
                               (resolve_live18.grpcz.values > 4500) & \
                               (resolve_live18.grpcz.values < 7000) & \
                               (resolve_live18.absrmag.values < -17.33)]

v_resolveA = 13172.384 #Survey volume without buffer [Mpc/h]^3
cvar_resolveA = 0.30

#487
resolve_B = resolve_live18.loc[(resolve_live18.f_b.values == 1) & \
                               (resolve_live18.grpcz.values > 4500) & \
                               (resolve_live18.grpcz.values < 7000) & \
                               (resolve_live18.absrmag.values < -17)]

v_resolveB = 4709.8373#*2.915 #Survey volume without buffer [Mpc/h]^3
cvar_resolveB = 0.58

### SMFs     
resa_m_stellar = resolve_A.logmstar.values     
max_resolveA,phi_resolveA,err_tot_A,nbins_A = diff_SMF(resa_m_stellar,\
                                                       v_resolveA,cvar_resolveA)

resb_m_stellar = resolve_B.logmstar.values   
max_resolveB,phi_resolveB,err_tot_B,nbins_B = diff_SMF(resb_m_stellar,\
                                                       v_resolveB,cvar_resolveB)

### Plot SMFs
fig1 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,\
             color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE B')
plt.errorbar(max_resolveA,phi_resolveA,yerr=err_tot_A,\
             color='#442b88',fmt='--s',ecolor='#442b88',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE A')
plt.yscale('log')
xlim_left,xlim_right = plt.xlim()
plt.axvline(x=8.7,color='#921063')
plt.axvline(x=8.9,color='#442b88')
plt.axvspan(xlim_left, 8.7, color='gray', alpha=0.7, lw=0)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-1} \right]$')
plt.legend(loc='best',prop={'size': 10})


### Simulation method
mhalo_characteristic = np.arange(11.5,13.0,0.1) #13.0 not included
mstellar_characteristic = np.arange(9.5,11.0,0.1) #11.0 not included
mlow_slope = np.arange(0.35,0.50,0.01)[:-1] #0.5 included by default
mhigh_slope = np.arange(0.50,0.65,0.01)[:-1]
mstellar_scatter = np.arange(0.02,0.095,0.005)

z_model = np.median(resolve_live18.grpcz.values)/(3*10**5)
#model = Moster13SmHm('moster13',prim_haloprop_key='halo_macc')
model = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_model,\
                                     prim_haloprop_key='halo_macc')
halocat = CachedHaloCatalog(fname=halo_catalog,update_cached_fname = True)

model.param_dict['smhm_m1_0'] = mhalo_characteristic[0]
model.param_dict['smhm_m0_0'] = mstellar_characteristic[0]
model.param_dict['smhm_beta_0'] = mlow_slope[0]
model.param_dict['smhm_delta_0'] = mhigh_slope[0]
model.param_dict['u’scatter_model_param1'] = mstellar_scatter[0]

model.populate_mock(halocat)

sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
gals = model.mock.galaxy_table[sample_mask]
gals_df = gals.to_pandas()
logM_mock  = np.log10(gals_df.stellar_mass.values)  #Read stellar masses
nbins_mock = nbins_B  #Number of bins to divide data into
V_sim = 130**3 #Vishnu volume [Mpc/h]^3 
#Unnormalized histogram and bin edges
Phi,edg = np.histogram(logM_mock,bins=nbins_mock)  
dM = edg[1] - edg[0]  #Bin width
Max = 0.5*(edg[1:]+edg[:-1])  #Mass axis i.e. bin centers
#Normalized to volume and bin width
err_poiss = np.sqrt(Phi)/(V_sim*dM)
Phi = Phi/(V_sim*dM) 

fig2 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,\
             color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE B')
plt.errorbar(max_resolveA,phi_resolveA,yerr=err_tot_A,\
             color='#442b88',fmt='--s',ecolor='#442b88',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE A')

plt.plot(Max,Phi,color='#442b88',linestyle='--',label='Vishnu')

plt.yscale('log')
plt.axvline(x=8.7,color='#921063')
plt.axvline(x=8.9,color='#442b88')
plt.axvspan(xlim_left, 8.7, color='gray', alpha=0.7, lw=0)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-1} \right]$')
plt.legend(loc='best',prop={'size': 10})

