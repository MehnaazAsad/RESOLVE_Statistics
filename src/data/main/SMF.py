#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:22:17 2018

@author: asadm2
"""
#%%
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
RESOLVE_A = RESOLVE[RESOLVE['Name'].str.contains("rs")] #969
RESOLVE_B = RESOLVE[RESOLVE['Name'].str.contains("rf")] #402


M_HI = []
M_stellar = []
M_halo_s = [] #group halo mass from stellar mass limited sample
M_halo_b = [] #group halo mass from baryonic mass limited sample
H_0 = 70 #km/s/Mpc
for galaxy in RESOLVE_B.Name.values:
    f21 = RESOLVE_B.F21.loc[RESOLVE_B.Name == galaxy].values[0]
    cz = RESOLVE_B.cz.loc[RESOLVE_B.Name == galaxy].values[0]
    D = cz/H_0 #Mpc
    m_HI = (2.36*10**5)*(D**2)*(f21)
    M_HI.append(np.log10(m_HI))
    M_stellar.append(RESOLVE_B['logstellarmass'].loc[RESOLVE_B.Name == galaxy].\
                     values[0])
    M_halo_s.append(RESOLVE_B['groupmass-s'].loc[RESOLVE_B.Name == galaxy]\
                    .values[0])
    M_halo_b.append(RESOLVE_B['groupmass-b'].loc[RESOLVE_B.Name == galaxy]\
                    .values[0])


logM_resolve  = M_stellar  #Read stellar masses
nbins_resolve = 10  #Number of bins to divide data into
V_resolve = 13700 #RESOLVE-B volume in Mpc3
#Unnormalized histogram and bin edges
Phi_resolve,edg_resolve = np.histogram(logM_resolve,bins=nbins_resolve)  
dM_resolve = edg_resolve[1] - edg_resolve[0]  #Bin width
Max_resolve = 0.5*(edg_resolve[1:]+edg_resolve[:-1])  #Mass axis i.e. bin centers
#Normalized to volume and bin width
err_poiss = np.sqrt(Phi_resolve)/(V_resolve*dM_resolve)
err_cvar = 0.58/(V_resolve*dM_resolve)
err_tot = np.sqrt(err_cvar**2 + err_poiss**2)

Phi_resolve = Phi_resolve/(V_resolve*dM_resolve)  

fig1 = plt.figure(figsize=(10,8))
plt.yscale('log')
plt.xlim(8.9,11.5)
plt.errorbar(Max_resolve,Phi_resolve,yerr=err_tot,fmt="rs--",    
    linewidth=2,   # width of plot line
    elinewidth=0.5,# width of error bar line
    ecolor='k',    # color of error bar
    capsize=5,     # cap length for error bar
    capthick=0.5 )
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
plt.title('Stellar mass function')
###############################################################################
###COSMIC VARIANCE
eco = pd.read_csv(path_to_interim + 'ECO_DR1.txt',delimiter='\s+',header=None,usecols=\
                            [0,1,2,3,4,5,13,15],\
                            names=['ECOID','RA','DEC','velocity','M_r',\
                                   'log_stellarmass','groupcz',\
                                   'log_halogroupmass'])
    
dec_slices = np.arange(-1,49.85,2.5)
minz_resB = np.min(RESOLVE_B.cz/(3*10**5))
maxz_resB = np.max(RESOLVE_B.cz/(3*10**5))
mincz_resB = 4500
maxcz_resB = 7000
H_0 = 70
slice_area = 2.5*75 #deg^2
deg2_in_sphere = 41252.96 #deg^2

num_in_slice = []
z = []
for index,value in enumerate(dec_slices):
    if index == 20:
        break
    counter = 0
    for index2,galaxy in enumerate(eco.ECOID.values):
        if 134.99 < eco.RA.values[index2] < 209.99:
            if value < eco.DEC.values[index2] < dec_slices[index+1]:
                if minz_resB < (eco.velocity.values[index2]/(3*10**5)) < maxz_resB:
                    z.append(eco.velocity.values[index2]/(3*10**5))
                    counter += 1
    num_in_slice.append(counter)
num_in_slice = np.array(num_in_slice)

#med_z = np.median(z)
#d = med_z*3*10**5/H_0 #Mpc
#vol_sphere = (4/3)*np.pi*(d**3) #Mpc^3

inner_vol = (4/3)*np.pi*((mincz_resB/H_0)**3)
outer_vol = (4/3)*np.pi*((maxcz_resB/H_0)**3)
vol_sphere = outer_vol-inner_vol #Mpc^3

vol_slice = (slice_area/deg2_in_sphere)*vol_sphere
gal_dens = num_in_slice/vol_slice
cosmic_variance = (np.std(gal_dens)*100)
print('Cosmic variance: {0}%'.format(np.round(cosmic_variance,2)))
###############################################################################

#%%
counter = 0
nbins = 10
Volume =  V_resolve #Volume of RESOLVE-B Mpc^3
Volume_FK = 250.**3
Mhalo_characteristic = np.arange(11.5,13.0,0.1) #13.0 not included
Mstellar_characteristic = np.arange(9.5,11.0,0.1) #11.0 not included
Mlow_slope = np.arange(0.35,0.50,0.01)[:-1] #0.5 included by default
Mhigh_slope = np.arange(0.50,0.65,0.01)[:-1]
Mstellar_scatter = np.arange(0.02,0.095,0.005)

def gals(Mhalo_value,Mstellar_value,Mlow_slope,Mhigh_slope,Mstellar_scatter):   
    ###Models
    model1 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    model2 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    model3 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    model4 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    model5 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
    
    ###Halocats
    halocat1 = CachedHaloCatalog(fname=halo_catalog)
    halocat2 = CachedHaloCatalog(fname=halo_catalog)
    halocat3 = CachedHaloCatalog(fname=halo_catalog)
    halocat4 = CachedHaloCatalog(fname=halo_catalog)
    halocat5 = CachedHaloCatalog(fname=halo_catalog)

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
    
    #Applying RESOLVE-B stellar mass limit
    sample_mask1 = model1.mock.galaxy_table['stellar_mass'] >= 10**8.7
    sample_mask2 = model2.mock.galaxy_table['stellar_mass'] >= 10**8.7
    sample_mask3 = model3.mock.galaxy_table['stellar_mass'] >= 10**8.7
    sample_mask4 = model4.mock.galaxy_table['stellar_mass'] >= 10**8.7
    sample_mask5 = model5.mock.galaxy_table['stellar_mass'] >= 10**8.7
    
    Mhalo_gals            = model1.mock.galaxy_table[sample_mask1]
    Mstellar_gals         = model2.mock.galaxy_table[sample_mask2]
    Mlowslope_gals        = model3.mock.galaxy_table[sample_mask3]
    Mhighslope_gals       = model4.mock.galaxy_table[sample_mask4]
    Mstellarscatter_gals  = model5.mock.galaxy_table[sample_mask5]
    
    return Mhalo_gals,Mstellar_gals,Mlowslope_gals,Mhighslope_gals,Mstellarscatter_gals
    
def init():
    
    line1 = ax1.errorbar([],[],yerr=[],fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=0.5)
    line2 = ax2.errorbar([],[],yerr=[],fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=0.5)     
    line3 = ax3.errorbar([],[],yerr=[],fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=0.5)     
    line4 = ax4.errorbar([],[],yerr=[],fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=0.5)     
    line5 = ax5.errorbar([],[],yerr=[],fmt="bs--",\
                        linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                        capthick=0.5) 
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
    
    ax1_Max = 0.5*(ax1_edg[1:]+ax1_edg[:-1])
    ax2_Max = 0.5*(ax2_edg[1:]+ax2_edg[:-1])
    ax3_Max = 0.5*(ax3_edg[1:]+ax3_edg[:-1])
    ax4_Max = 0.5*(ax4_edg[1:]+ax4_edg[:-1])
    ax5_Max = 0.5*(ax5_edg[1:]+ax5_edg[:-1])
    
    ax1_menStd = np.sqrt(Phi1)/(Volume_FK*ax1_dM)
    ax2_menStd = np.sqrt(Phi2)/(Volume_FK*ax2_dM)
    ax3_menStd = np.sqrt(Phi3)/(Volume_FK*ax3_dM)
    ax4_menStd = np.sqrt(Phi4)/(Volume_FK*ax4_dM)
    ax5_menStd = np.sqrt(Phi5)/(Volume_FK*ax5_dM)
   
    Phi1 = Phi1/(float(Volume_FK)*ax1_dM)
    Phi2 = Phi2/(float(Volume_FK)*ax2_dM)
    Phi3 = Phi3/(float(Volume_FK)*ax3_dM)
    Phi4 = Phi4/(float(Volume_FK)*ax4_dM)
    Phi5 = Phi5/(float(Volume_FK)*ax5_dM)
    
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.clear()
        SMF_RESOLVEB = ax.errorbar(Max_resolve,Phi_resolve,yerr=err_tot,\
                                  fmt="rs--",linewidth=2,elinewidth=0.5,\
                                  ecolor='k',capsize=2,capthick=1.5,\
                                  markersize='3')
        ax.set_xlim(8.9,)
        ax.minorticks_on()
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log(M_\star\,/\,M_\odot)$')
        ax.set_ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')

    line1 = ax1.errorbar(ax1_Max,Phi1,yerr=ax1_menStd,fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=1.5,markersize='3') 
    line2 = ax2.errorbar(ax2_Max,Phi2,yerr=ax2_menStd,fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=1.5,markersize='3') 
    line3 = ax3.errorbar(ax3_Max,Phi3,yerr=ax3_menStd,fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=1.5,markersize='3') 
    line4 = ax4.errorbar(ax4_Max,Phi4,yerr=ax4_menStd,fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=1.5,markersize='3') 
    line5 = ax5.errorbar(ax5_Max,Phi5,yerr=ax5_menStd,fmt="bs--",\
                            linewidth=2,elinewidth=0.5,ecolor='k',capsize=2,\
                            capthick=1.5,markersize='3') 

    
    ax1.legend([line1,SMF_RESOLVEB],[r'$M_{h}=%4.2f$' % Mhalo,'RESOLVE-B'],\
               loc='lower left',prop={'size': 10})
    ax2.legend([line2,SMF_RESOLVEB],[r'$M_{*}=%4.2f$' % Mstellar,'RESOLVE-B'],\
               loc='lower left',prop={'size': 10})
    ax3.legend([line3,SMF_RESOLVEB],[r'$\beta=%4.2f$' % Mlowslope,'RESOLVE-B'],\
               loc='lower left',prop={'size': 10})
    ax4.legend([line4,SMF_RESOLVEB],[r'$\delta=%4.2f$' % Mhighslope,'RESOLVE-B'],\
               loc='lower left',prop={'size': 10})
    ax5.legend([line5,SMF_RESOLVEB],[r'$\xi=%4.3f$' % Mstellarscatter,'RESOLVE-B']\
               ,loc='lower left',prop={'size': 10})
    
    print('Setting data')
    print('Frame {0}/{1}'.format(counter+1,len(Mhalo_characteristic)))
    
    counter+=1
    
    line = [line1,line2,line3,line4,line5]  
    
    return line

#Setting up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

anim = animation.FuncAnimation(plt.gcf(), make_animation, \
                               Mhalo_characteristic,init_func=init,\
                               fargs=(Mstellar_characteristic,Mlow_slope,\
                                      Mhigh_slope,Mstellar_scatter,),\
                                      interval=1000,blit=False,repeat=True)
plt.tight_layout()
print('Saving animation')
os.chdir(path_to_figures)
anim.save('SMF_RB_macc.gif',writer='imagemagick',fps=1)

