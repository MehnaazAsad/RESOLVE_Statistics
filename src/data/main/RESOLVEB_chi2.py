#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:55:42 2018

@author: asadm2
"""

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
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import matplotlib
import os


### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']
halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/bolshoi/'\
'rockstar/bolshoi_test_v1.hdf5'

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=10)
rc('text', usetex=True)

RESOLVE = pd.read_csv(path_to_interim + 'RESOLVE_formatted.txt',delimiter='\t')
RESOLVE_A = RESOLVE[RESOLVE['Name'].str.contains("rs")] #969
RESOLVE_B = RESOLVE[RESOLVE['Name'].str.contains("rf")] #402

RESB_M_stellar = []
for galaxy in RESOLVE_B.Name.values:
    RESB_M_stellar.append(RESOLVE_B['logstellarmass'].loc\
                          [RESOLVE_B.Name == galaxy].values[0])
    
    
logM_resolveB  = RESB_M_stellar  #Read stellar masses
nbins_resolveB = 10  #Number of bins to divide data into
V_resolveB = 13700# 5.2e4  #Survey volume in Mpc3
#Unnormalized histogram and bin edges
Phi_resolveB,edg_resolveB = np.histogram(logM_resolveB,bins=nbins_resolveB)  
dM_resolveB = edg_resolveB[1] - edg_resolveB[0]  #Bin width
Max_resolveB = 0.5*(edg_resolveB[1:]+edg_resolveB[:-1])  #Mass axis i.e. bin centers
#Normalized to volume and bin width
err_poiss_B = np.sqrt(Phi_resolveB)/(V_resolveB*dM_resolveB)
err_cvar_B = 0.58/(V_resolveB*dM_resolveB)
err_tot_B = np.sqrt(err_cvar_B**2 + err_poiss_B**2)

Phi_resolveB = Phi_resolveB/(V_resolveB*dM_resolveB) 

RESA_M_stellar = []
for galaxy in RESOLVE_A.Name.values:
    RESA_M_stellar.append(RESOLVE_A['logstellarmass'].loc\
                          [RESOLVE_A.Name == galaxy].values[0])
 
logM_resolveA  = RESA_M_stellar  #Read stellar masses
nbins_resolveA = 10  #Number of bins to divide data into
V_resolveA = 38400 #Survey volume in Mpc3
#Unnormalized histogram and bin edges
Phi_resolveA,edg_resolveA = np.histogram(logM_resolveA,bins=nbins_resolveA)  
dM_resolveA = edg_resolveA[1] - edg_resolveA[0]  #Bin width
Max_resolveA = 0.5*(edg_resolveA[1:]+edg_resolveA[:-1])  #Mass axis i.e. bin centers
#Normalized to volume and bin width
err_poiss_A = np.sqrt(Phi_resolveA)/(V_resolveA*dM_resolveA)
err_cvar_A = 0.30/(V_resolveA*dM_resolveA)
err_tot_A = np.sqrt(err_cvar_A**2 + err_poiss_A**2)

Phi_resolveA = Phi_resolveA/(V_resolveA*dM_resolveA)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

nbins = 10
Volume_FK = 250.**3
Mhalo_characteristic = np.arange(11.5,13.0,0.1) #13.0 not included
Mstellar_characteristic = np.arange(10,11.5,0.1) #11.0 not included

chi2_arrs = []
Mhalo_arr = []
Mstellar_arr = []
for index,stellar_value in enumerate(Mstellar_characteristic):
    print('{0}/{1} characteristic stellar'.format(index+1,len(Mstellar_characteristic)))
    chi2_arr = []
    for index2,halo_value in enumerate(Mhalo_characteristic):
        print('{0}/{1} characteristic halo'.format(index2+1,len(Mhalo_characteristic)))
        Mhalo_arr.append(halo_value)
        Mstellar_arr.append(stellar_value)
        model1 = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                         prim_haloprop_key='halo_macc')
        halocat1 = CachedHaloCatalog(fname=halo_catalog)
#        halocat1 = FakeSim()
        model1.param_dict['smhm_m1_0'] = halo_value
        model1.param_dict['smhm_m0_0'] = stellar_value
        model1.populate_mock(halocat1)
        sample_mask1 = model1.mock.galaxy_table['stellar_mass'] >= 10**8.7
        gals = model1.mock.galaxy_table[sample_mask1]
        
        logM = np.log10(gals['stellar_mass'])    
        Phi,edg = np.histogram(logM,bins=nbins)    
        dM = edg[1] - edg[0]    
        M_ax = 0.5*(edg[1:]+edg[:-1])   
        menStd = np.sqrt(Phi)/(Volume_FK*dM)  
        Phi = Phi/(float(Volume_FK)*dM)
        
        print('Calculating chi squared')
        chi2 = 0
        for i in range(nbins):
            chi2_i = ((Phi_resolveB[i]-Phi[i])**2)/((err_tot_B[i])**2)
            chi2 += chi2_i
        chi2_arr.append(chi2)
    chi2_arrs.append(chi2_arr)
    
chi2_arrs = np.array(chi2_arrs)

print('Plotting')
fig, ax = plt.subplots()
im, cbar = heatmap(chi2_arrs, np.round(Mstellar_characteristic,2), \
                   np.round(Mhalo_characteristic,2), ax=ax, cmap="RdYlBu", \
                   cbarlabel=r'$\chi^{2}$')

plt.xlabel(r'Characteristic halo mass')
plt.ylabel(r'Characteristic stellar mass')

fig.tight_layout()
os.chdir(path_to_figures)
plt.savefig('chi-squared.png')     