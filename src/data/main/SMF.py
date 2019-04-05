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
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import os

def which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/vishnu/'\
'rockstar/vishnu_rockstar_test.hdf5'
# halo_catalog = '/Users/asadm2/Desktop/vishnu_rockstar_test.hdf5'

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)
plt.rcParams['animation.convert_path'] = '/fs1/masad/anaconda3/envs/resolve_statistics/bin/magick'
#'{0}/magick'.format(os.path.dirname(which('python')))

def use_resolvea(path_to_file):
    resolve_live18 = pd.read_csv(path_to_raw + 'RESOLVE_liveJune2018.csv',
                                delimiter=',')
    #956 galaxies
    resolve_A = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                                    (resolve_live18.grpcz.values >= 4500) & 
                                    (resolve_live18.grpcz.values <= 7000) &
                                    (resolve_live18.absrmag.values <= -17.33)] 

    volume = 13172.384 #Survey volume without buffer [Mpc/h]^3
    cvar = 0.30
    return resolve_A,volume,cvar

def use_resolveb(path_to_file):
    resolve_live18 = pd.read_csv(path_to_raw + 'RESOLVE_liveJune2018.csv',
                                delimiter=',')
    #487
    resolve_B = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                                    (resolve_live18.grpcz.values >= 4500) & 
                                    (resolve_live18.grpcz.values <= 7000) & 
                                    (resolve_live18.absrmag.values <= -17)]

    volume = 4709.8373#*2.915 #Survey volume without buffer [Mpc/h]^3
    cvar = 0.58
    return resolve_B,volume,cvar

def use_eco(path_to_file):
    columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 'logmstar',
                'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 'fc',
                'grpmb', 'grpms']
    # 13878 galaxies
    eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0,
                            usecols=columns)
    # 6456 galaxies                       
    eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                                (eco_buff.grpcz.values <= 7000) & 
                                (eco_buff.absrmag.values <= -17.33)]

    volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
    cvar = 0.125
    return eco_nobuff,volume,cvar

def diff_smf(mstar_arr, volume, cvar_err, h1_bool):
    """Calculates differential stellar mass function given stellar masses."""
    
    if not h1_bool:
        # changing from h=0.7 to h=1
        logmstar_arr = np.log10((10**mstar_arr) / 1.429)
    else:
        logmstar_arr = np.log10(mstar_arr)
    # Unnormalized histogram and bin edges
    bins = np.linspace(8.9,11.8,12)
    phi, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(phi) / (volume * dm)
    err_cvar = cvar_err / (volume * dm)
    err_tot = np.sqrt(err_cvar**2 + err_poiss**2)
    phi = phi / (volume * dm)  # not a log quantity
    return maxis, phi, err_tot, bins

path_to_eco = path_to_raw + "eco_all.csv"
eco_nobuff, volume, cvar = use_eco(path_to_eco)
eco_nobuff_mstellar = eco_nobuff.logmstar.values
maxis, phi, err, bins = diff_smf(eco_nobuff_mstellar, volume, cvar, False)

fig1 = plt.figure(figsize=(10,8))
plt.yscale('log')
plt.xlim(8.9,11.5)
plt.errorbar(maxis,phi,yerr=err,fmt="rs--",    
    linewidth=2,   # width of plot line
    elinewidth=0.5,# width of error bar line
    ecolor='k',    # color of error bar
    capsize=5,     # cap length for error bar
    capthick=0.5 )
plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$')
plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
plt.title('Stellar mass function')

counter = 0
nbins = 10
Volume_sim = 130**3
Mhalo_characteristic = np.arange(11.5,13.0,0.1) #13.0 not included
Mstellar_characteristic = np.arange(9.5,11.0,0.1) #11.0 not included
Mlow_slope = np.arange(0.35,0.50,0.01)[:-1] #0.5 included by default
Mhigh_slope = np.arange(0.50,0.65,0.01)[:-1]
#testing with extreme scatter numbers from ECO's MCMC run
Mstellar_scatter = np.linspace(10,800,15) 

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

###Populate mocks
model1.populate_mock(halocat1)
model2.populate_mock(halocat2)
model3.populate_mock(halocat3)
model4.populate_mock(halocat4)
model5.populate_mock(halocat5)

def gals(Mhalo_value,Mstellar_value,Mlow_slope,Mhigh_slope,Mstellar_scatter):   
    ###Parameter values
    model1.param_dict['smhm_m1_0'] = Mhalo_value
    model2.param_dict['smhm_m0_0'] = Mstellar_value
    model3.param_dict['smhm_beta_0'] = Mlow_slope
    model4.param_dict['smhm_delta_0'] = Mhigh_slope
    model5.param_dict['u’scatter_model_param1'] = Mstellar_scatter
    
    model1.mock.populate()
    model2.mock.populate()
    model3.mock.populate()
    model4.mock.populate()
    model5.mock.populate()

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
    
    ax1_menStd = np.sqrt(Phi1)/(Volume_sim*ax1_dM)
    ax2_menStd = np.sqrt(Phi2)/(Volume_sim*ax2_dM)
    ax3_menStd = np.sqrt(Phi3)/(Volume_sim*ax3_dM)
    ax4_menStd = np.sqrt(Phi4)/(Volume_sim*ax4_dM)
    ax5_menStd = np.sqrt(Phi5)/(Volume_sim*ax5_dM)
   
    Phi1 = Phi1/(float(Volume_sim)*ax1_dM)
    Phi2 = Phi2/(float(Volume_sim)*ax2_dM)
    Phi3 = Phi3/(float(Volume_sim)*ax3_dM)
    Phi4 = Phi4/(float(Volume_sim)*ax4_dM)
    Phi5 = Phi5/(float(Volume_sim)*ax5_dM)
    
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.clear()
        SMF_RESOLVEB = ax.errorbar(maxis,phi,yerr=err,\
                                  fmt="rs--",linewidth=2,elinewidth=0.5,\
                                  ecolor='k',capsize=2,capthick=1.5,\
                                  markersize='3')
        ax.set_xlim(8.9,)
        ax.set_ylim(10**-5,10**-1)
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
anim.save('SMF_ECO_macc_MCMC.gif',writer='imagemagick',fps=1)