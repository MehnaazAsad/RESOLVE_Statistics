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
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#plt.ioff()
from matplotlib import rc
import pandas as pd
import numpy as np
import emcee
import math
#import os

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h = 2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def diff_SMF(mstar_arr,volume,cvar_err,h1_bool):
    if not h1_bool:
        logmstar_arr = np.log10((10**mstar_arr)/1.429) #changing from h=0.7 to h=1
    else: 
        logmstar_arr = np.log10(mstar_arr)
    bins = np.linspace(8.9,11.5,11)
    # nbins = num_bins(logmstar_arr)  #Number of bins to divide data into
    #Unnormalized histogram and bin edges
    phi,edg = np.histogram(logmstar_arr,bins=bins)  #paper used 17 bins
    dM = edg[1] - edg[0]  #Bin width
    maxis = 0.5*(edg[1:]+edg[:-1])  #Mass axis i.e. bin centers
    #Normalized to volume and bin width
    err_poiss = np.sqrt(phi)/(volume*dM)
    err_cvar = cvar_err/(volume*dM)
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    
    phi = phi/(volume*dM) #not a log quantity
    
    return maxis,phi,err_tot,bins

def populate_mock(model,theta):
    halocat = CachedHaloCatalog(fname=halo_catalog,update_cached_fname = True)
    z_model = np.median(resolve_live18.grpcz.values)/(3*10**5)
    if model=='Moster':
        model = Moster13SmHm(prim_haloprop_key='halo_macc')
    elif model=='Behroozi':
        model = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_model,\
                                     prim_haloprop_key='halo_macc')
    
    mhalo_characteristic,mstellar_characteristic,mlow_slope,mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['u’scatter_model_param1'] = mstellar_scatter
    model.populate_mock(halocat)

    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()
    return gals_df

def chi_squared(data_y,model_y,err_data):
    chi_squared = (data_y - model_y)**2/(err_data**2)
    return np.sum(chi_squared)

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_proc = dict_of_paths['proc_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
               'vishnu/rockstar/vishnu_rockstar_test.hdf5'
# halo_catalog = '/Users/asadm2/Desktop/vishnu_rockstar_test.hdf5'
chain_fname = path_to_proc+'emcee_SMFRB.dat'

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
max_resolveA,phi_resolveA,err_tot_A,bins_A = diff_SMF(resa_m_stellar,\
                                                       v_resolveA,cvar_resolveA,False)

resb_m_stellar = resolve_B.logmstar.values   
max_resolveB,phi_resolveB,err_tot_B,bins_B = diff_SMF(resb_m_stellar,\
                                                       v_resolveB,cvar_resolveB,False)

arr = np.arange(1,1000001,1)
list = arr.tolist()

def lnprob(theta,phi_resolveB,err_tot_B):
    list.pop(0)
    if theta[0] < 0:
        return -np.inf
    if theta[1] < 0:
        return -np.inf
    if theta[4] < 0:
        return -np.inf
    gals_df = populate_mock('Behroozi',theta)
    v_sim = 130**3
    mstellar_mock  = gals_df.stellar_mass.values  #Read stellar masses
    max_model,phi_model,err_tot_model,bins_model = diff_SMF(mstellar_mock,\
                                                        v_sim,0,True)
    chi2 = chi_squared(phi_resolveB,phi_model,err_tot_B)
    lnP = -chi2/2
    return lnP

behroozi10_param_vals = [12.35,10.72,0.44,0.57,0.15]
nwalkers = 250
ndim = 5
p0 = behroozi10_param_vals + np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) 
# p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(phi_resolveB, err_tot_B),threads=1)
sampler.run_mcmc(p0, 4000)
np.savetxt(chain_fname,sampler.flatchain)
'''
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
# plt.axvline(x=8.7,color='#921063')
# plt.axvline(x=8.9,color='#442b88')
# plt.axvspan(xlim_left, 8.7, color='gray', alpha=0.7, lw=0)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-1} \right]$')
plt.legend(loc='best',prop={'size': 10})
plt.show()

fig2 = plt.figure(figsize=(10,10))
plt.errorbar(max_resolveB,phi_resolveB,yerr=err_tot_B,\
             color='#921063',fmt='--s',ecolor='#921063',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE B')
plt.errorbar(max_resolveA,phi_resolveA,yerr=err_tot_A,\
             color='#442b88',fmt='--s',ecolor='#442b88',markersize=4,capsize=5,\
             capthick=0.5,label='RESOLVE A')

plt.plot(Max,Phi,color='k',linestyle='-',label='Vishnu')

plt.yscale('log')
plt.ylim(10**-5,10**-1)
# plt.axvline(x=8.7,color='#921063')
# plt.axvline(x=8.9,color='#442b88')
# plt.axvspan(xlim_left, 8.7, color='gray', alpha=0.7, lw=0)
plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{-1} \right]$')
plt.legend(loc='best',prop={'size': 10})
plt.show()
'''
# import emcee
# import numpy as np



# def lnprob(x, mu, icov):
#     diff = x-mu
#     list.pop(0)
#     return -np.dot(diff,np.dot(icov,diff))/2.0


# chain_fname = 'nd_gaussian_chain_pyversion.dat'
# ndim = 5

# # ensure reproducibility for tests against python emcee
# rseed = 10
# np.random.seed(rseed)

# means = np.random.rand(ndim)

# cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
# cov = np.triu(cov)
# cov += cov.T - np.diag(cov.diagonal())
# cov = np.dot(cov,cov)

# icov = np.linalg.inv(cov)
# nwalkers = 250
# nsteps = 4000
# p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
# # for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
# #     if (i+1) % 100 == 0:
# #         print("{0:5.1%}".format(float(i) / nsteps))

# sampler.run_mcmc(p0, 4000)

# np.savetxt(chain_fname,sampler.flatchain)