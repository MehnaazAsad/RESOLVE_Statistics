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
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rc
from shutil import which
import pandas as pd
import numpy as np
import os


### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/bolshoi/'\
'rockstar/bolshoi_test_v1.hdf5'

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=20)
rc('text', usetex=True)
plt.rcParams['animation.convert_path'] = '/fs1/masad/anaconda3/envs/resolve_statistics/bin/magick'
#'{0}/magick'.format(os.path.dirname(which('python')))

RESOLVE = pd.read_csv(path_to_interim + 'RESOLVE_formatted.txt',delimiter='\t')

M_HI = []
M_stellar = []
M_halo_s = [] #group halo mass from stellar mass limited sample
M_halo_b = [] #group halo mass from baryonic mass limited sample
H_0 = 70 #km/s/Mpc
for galaxy in RESOLVE.Name.values:
    f21 = RESOLVE.F21.loc[RESOLVE.Name == galaxy].values[0]
    cz = RESOLVE.cz.loc[RESOLVE.Name == galaxy].values[0]
    D = cz/H_0 #Mpc
    m_HI = (2.36*10**5)*(D**2)*(f21)
    M_HI.append(np.log10(m_HI))
    M_stellar.append(RESOLVE['logstellarmass'].loc[RESOLVE.Name == galaxy].\
                     values[0])
    M_halo_s.append(RESOLVE['groupmass-s'].loc[RESOLVE.Name == galaxy]\
                    .values[0])
    M_halo_b.append(RESOLVE['groupmass-b'].loc[RESOLVE.Name == galaxy]\
                    .values[0])

logM_resolve  = M_stellar  #Read stellar masses
nbins_resolve = 10  #Number of bins to divide data into
V_resolve = 5.2e4  #Survey volume in Mpc3
#Unnormalized histogram and bin edges
Phi_resolve,edg_resolve = np.histogram(logM_resolve,bins=nbins_resolve)  
dM_resolve = edg_resolve[1] - edg_resolve[0]  #Bin width
Max_resolve = edg_resolve[0:-1] + dM_resolve/2.  #Mass axis
#Normalized to volume and bin width
Phi_resolve = Phi_resolve/(V_resolve*dM_resolve)  

fig1 = plt.figure(figsize=(10,8))
plt.yscale('log')
plt.axvline(x=8.9, ls = '--', c = 'r')
#plt.xlim(8.9,11.5)
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
plt.plot(Max_resolve,Phi_resolve)
plt.title('Stellar mass function')

counter = 0
nbins = 10
Volume = 5.2e4  #Volume of RESOLVE Mpc^3
Volume_FK = 250.**3
Mhalo_characteristic = np.arange(11.5,13.0,1.5) #13.0 not included


def gals(Mhalo_value):   
    ###Models
    model1 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    
    ###Halocats
#    halocat1 = CachedHaloCatalog(fname=halo_catalog)

    halocat1 = FakeSim()
#    halocat2 = FakeSim()
#    halocat3 = FakeSim()
#    halocat4 = FakeSim()
#    halocat5 = FakeSim()
    
    ###Parameter values
    model1.param_dict['smhm_m1_0'] = Mhalo_value   
    ###Populate mocks
    model1.populate_mock(halocat1)    
    #Applying RESOLVE-A stellar mass limit
    sample_mask1 = model1.mock.galaxy_table['stellar_mass'] >= 10**8.9   
    Mhalo_gals            = model1.mock.galaxy_table[sample_mask1] 
    return Mhalo_gals
    
def init():
    line1, = ax1.plot([],[],lw=2,color='k')   
    return line1,

def make_animation(i):
    global counter
    Mhalo = i
    ax1_catalog = gals(Mhalo)
    
    ax1_logM = np.log10(ax1_catalog['stellar_mass'])
    
    Phi1,ax1_edg = np.histogram(ax1_logM,bins=nbins) 
    
    ax1_dM = ax1_edg[1] - ax1_edg[0] 
    
    ax1_Max = ax1_edg[0:-1] + ax1_dM/2.
    
    Phi1 = Phi1/(float(Volume_FK)*ax1_dM)
    
    ax1.legend([line1,SMF_RESOLVE],[r'$M_{h}=%4.2f$' % Mhalo,'RESOLVE'],\
               loc='lower left',prop={'size': 20})
    
    print('Setting data')
    print('Frame {0}/{1}'.format(counter+1,1))
    line1.set_data(ax1_Max,Phi1)
    
    counter+=1
    
    return line1, 

#Setting up the figure, the axis, and the plot element we want to animate
fig,ax1 = plt.subplots(figsize=(10,8))
#ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax1_catalog = gals(Mhalo_characteristic[0])
ax1_logM = np.log10(ax1_catalog['stellar_mass'])
ax1_Phi,ax1_edg = np.histogram(ax1_logM,bins=nbins) 
ax1_dM = ax1_edg[1] - ax1_edg[0]  
ax1_Max = ax1_edg[0:-1] + ax1_dM/2.
ax1_Phi = ax1_Phi/(float(Volume_FK)*ax1_dM)
line1, = ax1.plot(ax1_Max,ax1_Phi)
SMF_RESOLVE, = ax1.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

ax1.set_xlim(8.9,)
ax1.minorticks_on()
ax1.set_yscale('log')
ax1.set_xlabel(r'$\log(M_\star\,/\,M_\odot)$')
ax1.set_ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')

anim = animation.FuncAnimation(plt.gcf(), make_animation, \
                               Mhalo_characteristic,init_func=init,\
                               interval=1000,blit=False,repeat=True)
plt.tight_layout()
print('Saving animation')
#plt.show()
#Writer = animation.writers['imagemagick']
#writer = Writer(fps=15, bitrate=1800)
#anim.save('SMF.html')
#animation.verbose.set_level('helpful')
os.chdir(path_to_figures)
#writer = ImageMagickFileWriter()
anim.save('SMF_test.gif',writer='imagemagick',fps=1)
#anim.save('SMF_5params.html',fps=1)