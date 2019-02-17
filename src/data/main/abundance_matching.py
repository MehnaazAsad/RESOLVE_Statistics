#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:25:59 2018

@author: asadm2
"""

from cosmo_utils.utils import work_paths as cwpaths
from scipy.interpolate import CubicSpline
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
from matplotlib import rc
import pandas as pd
import numpy as np

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)

#Latest release of RESOLVE catalog
resolve_live = pd.read_csv(path_to_raw + 'RESOLVE_liveJune2018.csv',\
                           delimiter=',') #2286
RESOLVE_A = resolve_live.loc[resolve_live['f_a'] == 1] #1493
RESOLVE_B = resolve_live.loc[resolve_live['f_b'] == 1] #793

#Read stellar masses
logM_resolve  = [value for value in RESOLVE_B.logmstar.values if value >= 8.7]  
nbins_resolve = 20  #Number of bins to divide data into
#V_resolve = 62973.86559766765 #RESOLVE-A w/RESOLVE buffer + RESOLVE-B w/buffer
V_RESOLVEB = 16586.11924198251 #RESOLVE-B w/ buffer
#Unnormalized histogram and bin edges
Phi_resolve,edg_resolve = np.histogram(logM_resolve,bins=nbins_resolve)
dM_resolve = edg_resolve[1] - edg_resolve[0]  #Bin width
Max_resolve = 0.5*(edg_resolve[1:]+edg_resolve[:-1])  #Mass axis i.e. bin centers
Phi_resolve = Phi_resolve/V_RESOLVEB
#Cumulative number density plot for stellar mass
n_s = np.cumsum(np.flip(Phi_resolve,0))
n_s = np.flip(n_s,0)

##Spline interpolation
#tck = interpolate.splrep(Max_resolve,n)
#result = interpolate.splev(np.linspace(8.7,11.18,num=200),tck)
##Cubic spline interpolation
#cs = CubicSpline(Max_resolve,n)
##1-D interpolation
xs_s = np.linspace(8.7,11.25,num=200)
f_s = interpolate.interp1d(Max_resolve, n_s,fill_value="extrapolate",kind=3)
ynew_s = f_s(xs_s)

##Schecter fit
#def fit_func(x,phi_star,x_star,alpha):
#    return phi_star*((x/x_star)**(alpha+1))*(np.e**(-(x/x_star)))
#p0 = [10**-2,9.6,-1.25]
#bounds=((10**-3,9,-np.inf),(10**-2,10.5,np.inf))
#params = curve_fit(fit_func, Max_resolve, n, maxfev=20000)
#fit = fit_func(np.linspace(8.7,11.18),params[0][0],params[0][1],params[0][2])

fig1 = plt.figure()
plt.yscale('log')
plt.axvline(8.7,ls='--',c='r',label='RESOLVE-B')
#plt.axvline(8.9,ls='--',c='g',label='RESOLVE-A')
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$n(> M)/\mathrm{Mpc}^{-3}$')
plt.scatter(Max_resolve,n_s,label='data')
#plt.plot(np.linspace(8.7,11.18),fit,label='fit')
plt.plot(xs_s,ynew_s,'--k',label='interp/exterp')
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

xs_h = np.linspace(8.13,14.81,num=200)
f_h = interpolate.interp1d(Max_h, n_h,fill_value="extrapolate",kind=3)
ynew_h = f_h(xs_h)

fig2 = plt.figure()
plt.yscale('log')
plt.xlabel(r'$\log(M_{h}\,/\,M_\odot)$')
plt.ylabel(r'$n(> M)/\mathrm{Mpc}^{-3}$')
plt.scatter(Max_h,n_h,label='data')
plt.plot(xs_h,ynew_h,'--k',label='interp/exterp')
plt.legend(loc='best')
plt.show()

#interp_fn2 = lambda x,a: f_h(x) - a
#halo_mass_sham = []
#n_sm_arr = []
#for value in sorted(logM_resolve, key=float):
#    n_sm = f_s(value)
#    matched_hm = optimize.newton(interp_fn2, value+2, args=(n_sm,),maxiter=50000)
#    halo_mass_sham.append(matched_hm)

###SHAM###
##Stellar to halo mass relation
f_h = interpolate.interp1d(n_h, Max_h)
f_s = interpolate.interp1d(Max_resolve, n_s,fill_value="extrapolate",kind=3)
halo_mass_sham = []
n_sm_arr = []
for value in sorted(logM_resolve, key=float):
    n_sm = f_s(value)
    n_sm_arr.append(n_sm)
for value in n_sm_arr:
    halo_mass = f_h(value)
    halo_mass_sham.append(halo_mass)
    
logM_resolve = sorted(logM_resolve, key=float) 
logM_resolve = np.array(logM_resolve)

log_halomass_Behroozi = np.log10(10**12.35) + \
(0.44*np.log10(10**logM_resolve/(10**10.72))) + \
(((10**logM_resolve/(10**10.72))**0.57)/\
 (1+((10**logM_resolve/(10**10.72))**-1.56))) - 0.5

fig3 = plt.figure()
plt.scatter(halo_mass_sham,logM_resolve,s=5)
plt.plot(log_halomass_Behroozi,logM_resolve,'--k',label='Behroozi 2010')
plt.xlabel(r'\boldmath Halo Mass $\left[M_\odot \right]$')
plt.ylabel(r'\boldmath Stellar Mass $\left[M_\odot \right]$')
plt.legend(loc='best')
plt.show()

