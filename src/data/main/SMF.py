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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
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
Mhalo_characteristic = np.arange(11.5,13.0,0.1) #13.0 not included
Mstellar_characteristic = np.arange(9.5,11.0,0.1) #11.0 not included
Mlow_slope = np.arange(0.35,0.50,0.01)[:-1] #0.5 included by default
Mhigh_slope = np.arange(0.50,0.65,0.01)[:-1]
Mstellar_scatter = np.arange(0.02,0.095,0.005)


def gals(Mhalo_value,Mstellar_value,Mlow_slope,Mhigh_slope,Mstellar_scatter):   
    ###Models
    model1 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_mvir')
    model2 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_mvir')
    model3 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_mvir')
    model4 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_mvir')
    model5 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_mvir')
    
    ###Halocats
    halocat1 = CachedHaloCatalog(fname=halo_catalog)
    halocat2 = CachedHaloCatalog(fname=halo_catalog)
    halocat3 = CachedHaloCatalog(fname=halo_catalog)
    halocat4 = CachedHaloCatalog(fname=halo_catalog)
    halocat5 = CachedHaloCatalog(fname=halo_catalog)
#    halocat1 = FakeSim()
#    halocat2 = FakeSim()
#    halocat3 = FakeSim()
#    halocat4 = FakeSim()
#    halocat5 = FakeSim()
    
    ###Parameter values
    model1.param_dict['smhm_m1_0'] = Mhalo_value
    model2.param_dict['smhm_m0_0'] = Mstellar_value
    model3.param_dict['smhm_beta_0'] = Mlow_slope
    model4.param_dict['smhm_delta_0'] = Mhigh_slope
    model5.param_dict['u’scatter_model_param1'] = Mstellar_scatter
    
    ###Populate mocks
    model1.populate_mock(halocat1)
    model2.populate_mock(halocat2)
    model3.populate_mock(halocat3)
    model4.populate_mock(halocat4)
    model5.populate_mock(halocat5)
    
    #Applying RESOLVE-A stellar mass limit
    sample_mask1 = model1.mock.galaxy_table['stellar_mass'] >= 10**8.9
    sample_mask2 = model2.mock.galaxy_table['stellar_mass'] >= 10**8.9
    sample_mask3 = model3.mock.galaxy_table['stellar_mass'] >= 10**8.9
    sample_mask4 = model4.mock.galaxy_table['stellar_mass'] >= 10**8.9
    sample_mask5 = model5.mock.galaxy_table['stellar_mass'] >= 10**8.9
    
    Mhalo_gals            = model1.mock.galaxy_table[sample_mask1]
    Mstellar_gals         = model2.mock.galaxy_table[sample_mask2]
    Mlowslope_gals        = model3.mock.galaxy_table[sample_mask3]
    Mhighslope_gals       = model4.mock.galaxy_table[sample_mask4]
    Mstellarscatter_gals  = model5.mock.galaxy_table[sample_mask5]
    
    return Mhalo_gals,Mstellar_gals,Mlowslope_gals,Mhighslope_gals,Mstellarscatter_gals
    
def init():
    line1, = ax1.plot([],[],lw=2,color='k')   
    line2, = ax2.plot([],[],lw=2,color='k')
    line3, = ax3.plot([],[],lw=2,color='k')
    line4, = ax4.plot([],[],lw=2,color='k')
    line5, = ax5.plot([],[],lw=2,color='k')
    line = [line1,line2,line3,line4,line5]
    return line

def make_animation(i,j,k,l,m):
    global counter
    Mhalo, Mstellar, Mlowslope, Mhighslope, Mstellarscatter = i,j[counter],\
    k[counter],l[counter],m[counter]
    ax1_catalog,ax2_catalog,ax3_catalog,ax4_catalog,ax5_catalog = \
    gals(Mhalo,Mstellar,Mlowslope,Mhighslope,Mstellarscatter)
    
    ax1_logM = np.log10(ax1_catalog['stellar_mass'])
    ax2_logM = np.log10(ax2_catalog['stellar_mass'])
    ax3_logM = np.log10(ax3_catalog['stellar_mass'])
    ax4_logM = np.log10(ax4_catalog['stellar_mass'])
    ax5_logM = np.log10(ax5_catalog['stellar_mass'])
    
    Phi1,ax1_edg = np.histogram(ax1_logM,bins=nbins) 
    Phi2,ax2_edg = np.histogram(ax2_logM,bins=nbins)
    Phi3,ax3_edg = np.histogram(ax3_logM,bins=nbins)
    Phi4,ax4_edg = np.histogram(ax4_logM,bins=nbins)
    Phi5,ax5_edg = np.histogram(ax5_logM,bins=nbins)
    
    ax1_dM = ax1_edg[1] - ax1_edg[0] 
    ax2_dM = ax2_edg[1] - ax2_edg[0] 
    ax3_dM = ax3_edg[1] - ax3_edg[0]
    ax4_dM = ax4_edg[1] - ax4_edg[0]
    ax5_dM = ax5_edg[1] - ax5_edg[0]
    
    ax1_Max = ax1_edg[0:-1] + ax1_dM/2.
    ax2_Max = ax2_edg[0:-1] + ax2_dM/2.
    ax3_Max = ax3_edg[0:-1] + ax3_dM/2.
    ax4_Max = ax4_edg[0:-1] + ax4_dM/2.
    ax5_Max = ax5_edg[0:-1] + ax5_dM/2.
    
    Phi1 = Phi1/(float(Volume_FK)*ax1_dM)
    Phi2 = Phi2/(float(Volume_FK)*ax2_dM)
    Phi3 = Phi3/(float(Volume_FK)*ax3_dM)
    Phi4 = Phi4/(float(Volume_FK)*ax4_dM)
    Phi5 = Phi5/(float(Volume_FK)*ax5_dM)
    
    ax1.legend([line1,SMF_RESOLVE],[r'$M_{h}=%4.2f$' % Mhalo,'RESOLVE'],\
               loc='lower left',prop={'size': 10})
    ax2.legend([line2,SMF_RESOLVE],[r'$M_{*}=%4.2f$' % Mstellar,'RESOLVE'],\
               loc='lower left',prop={'size': 10})
    ax3.legend([line3,SMF_RESOLVE],[r'$\beta=%4.2f$' % Mlowslope,'RESOLVE'],\
               loc='lower left',prop={'size': 10})
    ax4.legend([line4,SMF_RESOLVE],[r'$\delta=%4.2f$' % Mhighslope,'RESOLVE'],\
               loc='lower left',prop={'size': 10})
    ax5.legend([line5,SMF_RESOLVE],[r'$\xi=%4.3f$' % Mstellarscatter,'RESOLVE']\
               ,loc='lower left',prop={'size': 10})
    
    print('Setting data')
    print(counter)
    line1.set_data(ax1_Max,Phi1)
    line2.set_data(ax2_Max,Phi2)
    line3.set_data(ax3_Max,Phi3)
    line4.set_data(ax4_Max,Phi4)
    line5.set_data(ax5_Max,Phi5)
    
    counter+=1
    
    return [line1,line2,line3,line4,line5]  

#Setting up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

ax1_catalog,ax2_catalog,ax3_catalog,ax4_catalog,ax5_catalog = \
gals(Mhalo_characteristic[0],Mstellar_characteristic[0],Mlow_slope[0],\
     Mhigh_slope[0],Mstellar_scatter[0])

ax1_logM = np.log10(ax1_catalog['stellar_mass'])
ax1_Phi,ax1_edg = np.histogram(ax1_logM,bins=nbins) 
ax1_dM = ax1_edg[1] - ax1_edg[0]  
ax1_Max = ax1_edg[0:-1] + ax1_dM/2.
ax1_Phi = ax1_Phi/(float(Volume_FK)*ax1_dM)
line1, = ax1.plot(ax1_Max,ax1_Phi)
SMF_RESOLVE, = ax1.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

ax2_logM = np.log10(ax2_catalog['stellar_mass'])
ax2_Phi,ax2_edg = np.histogram(ax2_logM,bins=nbins) 
ax2_dM = ax2_edg[1] - ax2_edg[0]  
ax2_Max = ax2_edg[0:-1] + ax2_dM/2.
ax2_Phi = ax2_Phi/(float(Volume_FK)*ax2_dM)
line2, = ax2.plot(ax2_Max,ax2_Phi)
SMF_RESOLVE, = ax2.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

ax3_logM = np.log10(ax3_catalog['stellar_mass'])
ax3_Phi,ax3_edg = np.histogram(ax3_logM,bins=nbins) 
ax3_dM = ax3_edg[1] - ax3_edg[0]  
ax3_Max = ax3_edg[0:-1] + ax3_dM/2.
ax3_Phi = ax3_Phi/(float(Volume_FK)*ax3_dM)
line3, = ax3.plot(ax3_Max,ax3_Phi)
SMF_RESOLVE, = ax3.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

ax4_logM = np.log10(ax4_catalog['stellar_mass'])
ax4_Phi,ax4_edg = np.histogram(ax4_logM,bins=nbins) 
ax4_dM = ax4_edg[1] - ax4_edg[0]  
ax4_Max = ax4_edg[0:-1] + ax4_dM/2.
ax4_Phi = ax4_Phi/(float(Volume_FK)*ax4_dM)
line4, = ax4.plot(ax4_Max,ax4_Phi)
SMF_RESOLVE, = ax4.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

ax5_logM = np.log10(ax5_catalog['stellar_mass'])
ax5_Phi,ax5_edg = np.histogram(ax5_logM,bins=nbins) 
ax5_dM = ax5_edg[1] - ax5_edg[0]  
ax5_Max = ax5_edg[0:-1] + ax5_dM/2.
ax5_Phi = ax5_Phi/(float(Volume_FK)*ax5_dM)
line5, = ax5.plot(ax5_Max,ax5_Phi)
SMF_RESOLVE, = ax5.plot(Max_resolve,Phi_resolve,label='SMF RESOLVE',c='r')

for ax in [ax1,ax2,ax3,ax4,ax5]:
    ax.set_xlim(8.9,)
    ax.minorticks_on()
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log(M_\star\,/\,M_\odot)$')
    ax.set_ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')

anim = animation.FuncAnimation(plt.gcf(), make_animation, \
                               Mhalo_characteristic,init_func=init,\
                               fargs=(Mstellar_characteristic,Mlow_slope,\
                                      Mhigh_slope,Mstellar_scatter,),\
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
anim.save('SMF_5params.gif',writer='imagemagick',fps=1)
#anim.save('SMF_5params.html',fps=1)