"""
The purpose of this script is to compare the HMF that RESOLVE used (Warren et.
al 2006) with the HMF I use for abundance matching from the Vishnu simulation 
to see if a difference exists and if so, perhaps accounts for the difference
in logmh_s vs grpms for centrals between my AM and that of RESOLVE
"""

from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def cumu_num_dens(data,nbins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=nbins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not bool_mag:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,edg,n_cumu,err_poiss,bin_width

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

#HMF from hmfcalc; Warren et. al 2006
cumunumdens_table = pd.read_csv('~/Desktop/all_plots/mVector_PLANCK-SMT .txt',\
                                usecols=[0,8],names=['Mass','n'],comment='#',\
                                delim_whitespace=True)

halo_table = pd.read_csv(path_to_interim + 'id_macc.csv',header=0)
v_sim = 130**3 #(Mpc/h)^3
halo_mass = halo_table.halo_macc.values
nbins = num_bins(halo_mass)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
bin_width_hmass = cumu_num_dens(halo_mass,nbins,None,v_sim,False)

fig1 = plt.figure(figsize=(10,8))
plt.plot(cumunumdens_table.Mass.values,cumunumdens_table.n.values,\
         label=r'Warren et. al 2006',color='#128486')
plt.plot(bin_centers_hmass,n_hmass,label=r'Vishnu simulation',color='#eb4f1c')
plt.xlabel(r'$\mathbf {Mass \left[M_\odot\ h^{-1}\right ]}$')
plt.ylabel(r'$\mathbf {(n > M)} \left[\mathbf{{h}^{3}{Mpc}^{-3}}\right]$')
plt.title(r'$\mathbf {HMF\ comparison}$')
plt.legend(loc='best',prop={'size': 10})
plt.xscale('log')
plt.yscale('log')
plt.show()