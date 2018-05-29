#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:22:17 2018

@author: asadm2
"""

### DESCRIPTION
#This script parametrizes the SMHM relation to produce a SMF which is compared
#to the SMF from RESOVLE
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from halotools.sim_manager import FakeSim
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import rc
import pandas as pd
import numpy as np

### Paths
path_to_raw = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/raw/'
path_to_interim = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/interim/'

#path_to_raw = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/raw/'
#path_to_interim = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/'\
#'interim/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

resolve_dr1_1 = path_to_raw + 'RESOLVE_DR1_1.txt'
resolve_dr2_1 = path_to_raw + 'RESOLVE_DR2_1.txt'
resolve_dr2_2 = path_to_raw + 'RESOLVE_DR2_2.txt'


dr11 = pd.read_csv(resolve_dr1_1,delimiter='\s+',header=None,\
                           skiprows=79,usecols=[0,3,4,44],names=['Name','cz',\
                                               'groupcz','logM*']) 

dr21 = pd.read_csv(resolve_dr2_1,delimiter='\s+',header=None,\
                           skiprows=45,names=['Name','RA','DEC','Tel','F21',\
                                              'e_F21','rms','limflag',\
                                              'confused','F21corr',\
                                              'eF21corr-rand','eF21corr-sys',\
                                              'deconf-mode'])   

dr22 = pd.read_csv(resolve_dr2_2,delimiter='\s+',header=None,\
                           skiprows=61,names=['Name','groupid-s','groupmass-s',
                                              'groupid-b','groupmass-b',
                                              'dens-s','dens-corr-s','dens-b',
                                              'dens-corr-b','dist-s',
                                              'f_dist-s','dist-b',
                                              'f_dist-b'])

#Common galaxies
dfs = [dr11, dr21, dr22]
merged_dr = reduce(lambda left,right: pd.merge(left,right,on='Name'), dfs)


merged_dr_units = ['--','km/s','km/s','solMass','deg','deg','--','Jy.km/s',
                   'Jy.km/s','mJy','--','--','Jy.km/s','Jy.km/s','Jy.km/s',
                   '--','--','solMass','--','solMass','solMass/Mpc2',
                   'solMass/Mpc2','solMass/Mpc2','solMass/Mpc2','--','--',
                   '--','--']

merged_dr_description = ['RESOLVE galaxy identifier',\
                         'Local group corrected velocity of galaxy',\
                         'Velocity of group center','Log stellar mass',\
                         'Right Ascension of photometric center',\
                         'Declination of photometric center',\
                         'Telescope used for observation (GBT = Green Bank 100'
                         'm; AO = Arecibo 300m; ALFALFA = Arecibo Legacy Fast '
                         'ALFA Survey (Arecibo 300m); S05-AO = Springob et '
                         'al. (2005) catalog (Arecibo 300m); S05-GB300 = '
                         'Springob et al. (2005) catalog '
                         '(Green Bank 300ft) )','Total 21cm flux',\
                         'Uncertainty on total 21cm flux',\
                         'rms noise of spectrum assuming 10 km/s channels',\
                         'Flag indicating upper limit on F21 '
                         '(0 = detection; 1=upper limit)',\
                         'Flag indicating 21cm emission likely confused '
                         '(0 = not confused; 1 = confused)',\
                         'Total 21cm flux after correction for confusion',\
                         'Statistical uncertainty on total 21cm flux after '
                         'correction for confusion',\
                         'Additional systematic uncertainty on total 21cm '
                         'flux after correction for confusion',\
                         'Method used to correct 21cm flux for confusion and '
                         'determine the additional systematic uncertainty',\
                         'Group dark matter halo ID, stellar-mass limited '
                         'sample','Group dark matter halo mass, stellar-mass '
                         'limited sample','Group dark matter halo ID, '
                         'baryonic mass-limited sample','Group dark matter '
                         'halo mass, baryonic mass-limited sample',\
                         'Large-scale structure density, stellar mass-limited '
                         'sample groups, uncorrected for edge-effects',\
                         'Large-scale structure density, stellar mass-limited '
                         'sample groups, corrected for edge-effects',\
                         'Large-scale structure density, baryonic '
                         'mass-limited sample groups, uncorrected for '
                         'edge-effects','Large-scale structure density, '
                         'baryonic mass-limited sample groups, corrected '
                         'for edge-effects','Projected distance to nearest '
                         'M_h>10^12 halo, stellar mass-limited sample groups',\
                         'Reliability Flag for dist_s (Flag=1 indicates that '
                         'the corresponding distance may be unreliable due to '
                         'proximity to survey boundary.)',\
                         'Projected distance to nearest M_h>10^12 halo, '
                         'baryonic mass-limited sample groups ',\
                         'Reliability Flag for dist_b (Flag=1 indicates that '
                         'the corresponding distance may be unreliable due to '
                         'proximity to survey boundary.)']

units = {}
describe = {}
for index,value in enumerate(merged_dr.columns.values):
    name = value
    units[name] = merged_dr_units[index]
    describe[name] = merged_dr_description[index]

M_HI = []
M_stellar = []
M_halo_s = [] #group halo mass from stellar mass limited sample
M_halo_b = [] #group halo mass from baryonic mass limited sample
H_0 = 70 #km/s/Mpc
for galaxy in merged_dr.Name.values:
    f21 = merged_dr.F21.loc[merged_dr.Name == galaxy].values[0]
    cz = merged_dr.cz.loc[merged_dr.Name == galaxy].values[0]
    D = cz/H_0 #Mpc
    m_HI = (2.36*10**5)*(D**2)*(f21)
    M_HI.append(np.log10(m_HI))
    M_stellar.append(merged_dr['logM*'].loc[merged_dr.Name == galaxy].\
                     values[0])
    M_halo_s.append(merged_dr['groupmass-s'].loc[merged_dr.Name == galaxy]\
                    .values[0])
    M_halo_b.append(merged_dr['groupmass-b'].loc[merged_dr.Name == galaxy]\
                    .values[0])

logM_resolve  = M_stellar  #Read stellar masses
nbins_resolve = 13  #Number of bins to divide data into
V_resolve = 5.2e4  #Survey volume in Mpc3
#Unnormalized histogram and bin edges
Phi_resolve,edg_resolve = np.histogram(logM_resolve,bins=nbins_resolve)  
dM_resolve = edg_resolve[1] - edg_resolve[0]  #Bin width
Max_resolve = edg_resolve[0:-1] + dM_resolve/2.  #Mass axis
#Normalized to volume and bin width
Phi_resolve = Phi_resolve/(float(V_resolve)*dM_resolve)  

fig1 = plt.figure(figsize=(10,8))
plt.yscale('log')
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
plt.plot(Max_resolve,Phi_resolve)
plt.title('Stellar mass function')


Mhalo_characteristic = np.arange(8,11,0.1)

def gals(value):   
    model = PrebuiltSubhaloModelFactory('behroozi10')
#    halocat = FakeSim()
    halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')
    model.param_dict['smhm_m1_0'] = value
    model.populate_mock(halocat)
    #Applying RESOLVE-A stellar mass limit
    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**8.7
    gals = model.mock.galaxy_table[sample_mask]
    return gals

fig,ax = plt.subplots(figsize=(10,8))
catalog = gals(Mhalo_characteristic[-1])
logM = np.log10(catalog['stellar_mass'])
nbins = 13
Volume = 5.2e4  #Volume of RESOLVE Mpc^3
Phi,edg = np.histogram(logM,bins=nbins) 
dM = edg[1] - edg[0]  
Max = edg[0:-1] + dM/2.
Phi = Phi/(float(Volume)*dM)
line, = plt.plot(Max,Phi)
SMF_RESOLVE = plt.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE')
plt.yscale('log')
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
plt.title('Stellar mass function')

def init():
    line, = ax.plot([],[],lw=2,color='k')    
    return line,

def make_animation(i):
    catalog = gals(i)
    logM = np.log10(catalog['stellar_mass'])
    nbins = 13
    Volume = 5.2e4  #Volume of RESOLVE Mpc^3
    Phi,edg = np.histogram(logM,bins=nbins) 
    dM = edg[1] - edg[0]  
    Max = edg[0:-1] + dM/2.
    Phi = Phi/(float(Volume)*dM)    
    plt.legend([r'$M_{h}=%4.2f$' % i],loc='lower right')
    line.set_data(Max,Phi)
    return line,    
    
anim = animation.FuncAnimation(plt.gcf(), make_animation, frames = \
                               Mhalo_characteristic,init_func = init,\
                               interval=1000,blit=False)
plt.show()