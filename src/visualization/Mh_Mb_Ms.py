#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:52:43 2018

@author: asadm2
"""

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
import pandas as pd
import numpy as np

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font',**{'family':'sans-serif','sans-serif':['Tahoma']},size=20)
rc('text', usetex=True)

resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv", \
                             delimiter=",", header=0,\
                             usecols=['name','fc','logmh_s','grpmb','grpms',\
                                      'modelg_r','logmstar','logmgas','f_b',\
                                      'f_a'])

centrals = resolve_live18.loc[resolve_live18['fc'] == 1]
RESOLVE_B = centrals.loc[centrals['f_b'] == 1] 
RESOLVE_A = centrals.loc[centrals['f_a'] == 1]  #969
RESOLVE_B = RESOLVE_B.reset_index(drop=True)
RESOLVE_A = RESOLVE_A.reset_index(drop=True)
#Calculate baryonic mass
logmb_RB = np.log10(10**RESOLVE_B.logmstar.values + 10**RESOLVE_B.logmgas.values)
logmb_RA = np.log10(10**RESOLVE_A.logmstar.values + 10**RESOLVE_A.logmgas.values)

df_logmb_RB = pd.DataFrame({'logmb': logmb_RB})
RESOLVE_B = pd.concat([RESOLVE_B, df_logmb_RB], axis=1)
RESOLVE_B = RESOLVE_B.loc[RESOLVE_B['logmb'] >= 9.1]

df_logmb_RA = pd.DataFrame({'logmb': logmb_RA})
RESOLVE_A = pd.concat([RESOLVE_A, df_logmb_RA], axis=1)
RESOLVE_A = RESOLVE_A.loc[RESOLVE_A['logmb'] >= 9.3]

nbins_resolve = 20  #Number of bins to divide data into
#V_resolve = 62973.86559766765 #RESOLVE-A w/RESOLVE buffer + RESOLVE-B w/buffer
V_RESOLVEB = 16586.11924198251 #RESOLVE-B w/ buffer
#Unnormalized histogram and bin edges
Phi_resolvebary,edg_resolvebary = np.histogram(RESOLVE_B['logmb'],\
                                               bins=nbins_resolve)
dM_resolve = edg_resolvebary[1] - edg_resolvebary[0]  #Bin width
Max_resolvebary = 0.5*(edg_resolvebary[1:]+edg_resolvebary[:-1])  #Mass axis i.e. bin centers
Phi_resolvebary = Phi_resolvebary/V_RESOLVEB
#Cumulative number density plot for stellar mass
n_b = np.cumsum(np.flip(Phi_resolvebary,0))
n_b = np.flip(n_b,0)

xs_b = np.linspace(8.07,11.15,num=200)
f_b = interpolate.interp1d(Max_resolvebary, n_b,fill_value="extrapolate",kind=3)
ynew_b = f_b(xs_b)

fig1 = plt.figure()
plt.yscale('log')
plt.xlabel(r'$\log(M_{b}\,/\,M_\odot)$')
plt.ylabel(r'$n(> M)/\mathrm{Mpc}^{-3}$')
plt.scatter(Max_resolvebary,n_b,label='data')
plt.plot(xs_b,ynew_b,'--k',label='interp/exterp')
plt.legend(loc='best')
plt.show()

halo_mass_table = pd.read_csv(path_to_interim + 'macc.csv',header=None,\
                              names=['halomass'])
V_sim = 130**3
logM_h  = np.log10(halo_mass_table.halomass.values/0.7) #solar mass
Phi_h,edg_h = np.histogram(logM_h,bins=20)
dM_h = edg_h[1] - edg_h[0]  #Bin width
Max_h = 0.5*(edg_h[1:]+edg_h[:-1])  #Mass axis i.e. bin centers
Phi_h = Phi_h/V_sim
#Cumulative number density plot for halo mass
n_h = np.cumsum(np.flip(Phi_h,0))
n_h = np.flip(n_h,0)

###SHAM###
def SHAM(obs_x,obs_y,sim_x,sim_y,data_x,sort_key):
    f_halo = interpolate.interp1d(sim_y, sim_x,fill_value="extrapolate")
    f_obs = interpolate.interp1d(obs_x, obs_y,\
                                 fill_value="extrapolate",kind=3)
    
    sham_result = []
    n_obs_arr = []
    for value in sorted(data_x, key=sort_key):
        n_obs = f_obs(value)
        n_obs_arr.append(n_obs)
    for value in n_obs_arr:
        match_x = f_halo(value)
        sham_result.append(match_x)

    return(sham_result)

halo_mass_sham = SHAM(Max_resolvebary,n_b,Max_h,n_h,RESOLVE_B['logmb'],float)
fig2 = plt.figure(figsize=(10,8))
plt.scatter(halo_mass_sham,RESOLVE_B['logmb'],\
            c=RESOLVE_B.modelg_r.values,cmap='RdBu_r')
cbar = plt.colorbar()
cbar.set_label(r'\boldmath\ g - r', rotation=270,labelpad=20,fontsize=15)
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_b \left[M_\odot \right]$')
plt.title('Baryonic vs halo mass RESOLVE-B')


#fig2 = plt.figure(figsize=(10,8))
#plt.scatter(centrals.logmh_s.values,centrals.logmstar.values,\
#            c=centrals.modelg_r.values,cmap='RdBu_r')
#cbar = plt.colorbar()
#cbar.set_label(r'\boldmath\ g - r', rotation=270,labelpad=20,fontsize=15)
#plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
#plt.ylabel(r'\boldmath$\log\ M_s \left[M_\odot \right]$')
#plt.title('Stellar vs halo mass')
#

def SHAM(obs_x,obs_y,sim_x,sim_y,data_x,sort_key):
    f_halo = interpolate.interp1d(sim_y, sim_x,fill_value="extrapolate")
    f_obs = interpolate.interp1d(obs_x, obs_y,\
                                 fill_value="extrapolate",kind=3)
    
    sham_result = []
    n_obs_arr = []
    for value in sorted(data_x, key=sort_key):
        n_obs = f_obs(value)
        n_obs_arr.append(n_obs)
    for value in n_obs_arr:
        match_x = f_halo(value)
        sham_result.append(match_x)

    return(sham_result)

halo_mass_sham = SHAM(Max_resolvebary,n_b,Max_h,n_h,RESOLVE_B['logmb'],float)
fig2 = plt.figure(figsize=(10,8))
plt.scatter(halo_mass_sham,RESOLVE_B['logmb'],\
            c=RESOLVE_B.modelg_r.values,cmap='RdBu_r')
cbar = plt.colorbar()
cbar.set_label(r'\boldmath\ g - r', rotation=270,labelpad=20,fontsize=15)
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_b \left[M_\odot \right]$')
plt.title('Baryonic vs halo mass RESOLVE-B')


