"""
{This script carries out HAM and SHAM using baryonic and stellar masses of 
 groups and individual galaxies and compares to the values from RESOLVE}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from cosmo_utils.utils import work_paths as cwpaths
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
import pandas as pd
import numpy as np
import math

__author__ = '{Mehnaaz Asad}'

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def cumu_num_dens(data,bins,weights,volume,bool_mag):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=bins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    if not bool_mag:
        N_cumu = np.cumsum(freq[::-1])[::-1] 
    else:
        N_cumu = np.cumsum(freq)
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,edg,n_cumu,err_poiss,bin_width

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

columns = ['name','radeg','dedeg','cz','grpcz','logmstar','logmgas','grp',\
    'grpn','logmh','logmh_s','fc','grpmb','grpms','f_a','f_b',\
        'grpabsrmag']

# 2286 galaxies
resolve_live18 = pd.read_csv(path_to_raw + "RESOLVE_liveJune2018.csv", \
    delimiter=",", header=0, usecols=columns)

grps = resolve_live18.groupby('grp') #group by group ID
grp_keys = grps.groups.keys()

# Isolating groups that don't have designated central
grp_id_no_central_arr = []
for key in grp_keys:
    group = grps.get_group(key)
    if 1 not in group.fc.values:
        grp_id = group.grp.values
        grp_id_no_central_arr.append(np.unique(grp_id)[0])

resolve_live18 = resolve_live18.loc[~resolve_live18['grp'].\
    isin(grp_id_no_central_arr)]

#################################### (HAM) ################################
cols_subset = ['grp','grpmb','grpms','logmh_s','logmh','grpabsrmag']
resolve_live18_subset = resolve_live18[cols_subset]
grps = resolve_live18_subset.groupby('grp') 
grp_keys = grps.groups.keys()

# Get integrated baryonic and stellar mass and have one entry per group
grpmb_arr = np.zeros(len(grp_keys))
grpms_arr = np.zeros(len(grp_keys))
logmh_s_arr = np.zeros(len(grp_keys))
logmh_arr = np.zeros(len(grp_keys))
grprmag_arr = np.zeros(len(grp_keys))
for idx,key in enumerate(grp_keys):
    group = grps.get_group(key)
    grpmb = np.unique(group.grpmb.values)[0]
    grpms = np.unique(group.grpms.values)[0]
    logmh_s = np.unique(group.logmh_s.values)[0]
    logmh = np.unique(group.logmh.values)[0]
    grprmag = np.unique(group.grpabsrmag.values)[0]
    grpms_arr[idx] = grpms
    grpmb_arr[idx] = grpmb
    logmh_s_arr[idx] = logmh_s
    logmh_arr[idx] = logmh
    grprmag_arr[idx] = grprmag

# Create cumulative baryonic and stellar mass functions
bins = np.linspace(6.7,12.5,15)
v_resolve = 50000/2.915 #(Mpc/h)^3 (h from 0.7 to 1)
bin_centers_grpmb,bin_edges_grpmb,n_grpmb,err_poiss_grpmb,bin_width_grpmb = \
    cumu_num_dens(grpmb_arr,bins,None,v_resolve,False)
bin_centers_grpms,bin_edges_grpms,n_grpms,err_poiss_grpms,bin_width_grpms = \
    cumu_num_dens(grpms_arr,bins,None,v_resolve,False)

# Load halo catalog
halo_table = pd.read_csv(path_to_interim + 'id_macc.csv',header=0)
v_sim = 130**3 #(Mpc/h)^3

# Use only host halos for HAM (PID and UPID of -1)
halo_mass_hh = halo_table.halo_macc.loc[halo_table.C_S.values==1]

# Create HMF
bins = num_bins(halo_mass_hh)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
    bin_width_hmass = cumu_num_dens(halo_mass_hh,bins,None,v_sim,False)

# Interpolating between grpmb and n
grpmb_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb)

# Interpolating between grpmstar and n
grpms_n_interp_func = interpolate.interp1d(bin_centers_grpms,n_grpms,\
    fill_value='extrapolate')

# Interpolating between central hmass and n and reversing it so you can pass 
# an n and get central hmass value
hmass_n_interp_func = interpolate.interp1d(n_hmass,bin_centers_hmass)

pbar = ProgressBar(maxval=len(grpmb_arr))
n_grpmb_arr = [grpmb_n_interp_func(val) for val in pbar(grpmb_arr)]
pbar = ProgressBar(maxval=len(n_grpmb_arr))
hmass_grpmb_ham = [hmass_n_interp_func(val) for val in pbar(n_grpmb_arr)]

pbar = ProgressBar(maxval=len(grpms_arr))
n_grpms_arr = [grpms_n_interp_func(val) for val in pbar(grpms_arr)]
pbar = ProgressBar(maxval=len(n_grpms_arr))
hmass_grpms_ham = [hmass_n_interp_func(val) for val in pbar(n_grpms_arr)]

### Convert to log
hmass_loggrpmb = np.log10(hmass_grpmb_ham)
hmass_loggrpms = np.log10(hmass_grpms_ham)

### Get error bars
x_grpmb,y_grpmb,y_std_grpmb,y_std_err_grpmb = Stats_one_arr(hmass_loggrpmb,\
    grpmb_arr,base=0.3,bin_statval='center')

x_grpms,y_grpms,y_std_grpms,y_std_err_grpms = Stats_one_arr(hmass_loggrpms,\
    grpms_arr,base=0.3,bin_statval='center')

x_mhs,y_mhs,y_std_mhs,y_std_err_mhs = Stats_one_arr(logmh_s_arr,grpms_arr,\
    base=0.3,bin_statval='center')

x_mh,y_mh,y_std_mh,y_std_err_mh = Stats_one_arr(logmh_arr,grprmag_arr,base=0.3,\
    bin_statval='center')

fig1 = plt.figure(figsize=(10,10))
plt.errorbar(x_grpmb,y_grpmb,yerr=y_std_err_grpmb,\
             color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
             capthick=0.5,label=r'$M_{bary,grp}$')
plt.errorbar(x_grpms,y_grpmb,yerr=y_std_err_grpms,\
             color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
             capthick=0.5,label=r'$M_{\star ,grp}$')
plt.errorbar(x_mhs,y_grpmb,yerr=y_std_err_mhs,\
             color='#a0298d',fmt='--s',ecolor='#a0298d',markersize=4,capsize=5,\
             capthick=0.5,label=r'RESOLVE stellar mass derived')
plt.errorbar(x_mh,y_grpmb,yerr=y_std_err_mh,\
             color='k',fmt='--s',ecolor='k',markersize=4,capsize=5,\
             capthick=0.5,label=r'RESOLVE r-band mag derived')
plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{bary,grp} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 10})
plt.show()