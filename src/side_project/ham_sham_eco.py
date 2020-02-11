"""
{This script carries out HAM and SHAM using baryonic and stellar masses of 
 groups and individual galaxies and compares to the values from ECO}
"""

# Libs
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from cosmo_utils.utils import work_paths as cwpaths
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
from numpy import random
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

def get_wp(RA,DEC,CZ):
    N = len(RA)
    weights = np.ones_like(RA)

    # Random points
    rand_num = eco_nobuff.size*5
    rand_RA = np.round(random.uniform(eco_nobuff.radeg.min(),eco_nobuff.radeg.max(),\
        rand_num), 5)
    rand_DEC = np.round(random.uniform(eco_nobuff.dedeg.min(),eco_nobuff.dedeg.max(),\
        rand_num), 7)
    rand_CZ = np.round(random.uniform(eco_nobuff.grpcz.min(),eco_nobuff.grpcz.max(),\
        rand_num), 1)
    rand_N = len(rand_RA)
    rand_weights = np.ones_like(rand_RA)

    nbins = 10
    bins = np.logspace(np.log10(0.1), np.log10(20.0), nbins + 1)
    cosmology = 2 # Planck
    nthreads = 2
    pimax = 25.0

    # Auto pair counts in DD
    autocorr = 1
    DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA, \
        DEC, CZ, weights1=weights, weight_type='pair_product')

    # Auto pair counts in RR
    RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, rand_RA, \
        rand_DEC, rand_CZ, weights1=rand_weights, weight_type='pair_product')

    # Cross pair counts in DR
    autocorr=0
    DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA, DEC, \
        CZ, RA2=rand_RA, DEC2=rand_DEC, CZ2=rand_CZ, weights1=weights, \
            weights2=rand_weights, weight_type='pair_product')

    wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N, DD_counts, DR_counts, \
        DR_counts, RR_counts, nbins, pimax)

    return bins,wp

# Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=18)
rc('text', usetex=True)

columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'grpabsrmag', 'absrmag', 
            'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
            'fc', 'grpmb', 'grpms']

# 13878 galaxies
eco_buff = pd.read_csv(path_to_raw+'eco_all.csv',delimiter=",", header=0, \
    usecols=columns)

eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
    (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33) & \
            (eco_buff.logmstar.values >= 8.9)]

grps = eco_nobuff.groupby('grp') #group by group ID
grp_keys = grps.groups.keys()

# Isolating groups that don't have designated central
grp_id_no_central_arr = []
for key in grp_keys:
    group = grps.get_group(key)
    if 1 not in group.fc.values:
        grp_id = group.grp.values
        grp_id_no_central_arr.append(np.unique(grp_id)[0])

eco_nobuff = eco_nobuff.loc[~eco_nobuff['grp'].isin(grp_id_no_central_arr)]
#################################### (HAM) ################################
grps = eco_nobuff.groupby('grp') 
grp_keys = grps.groups.keys()

# Get integrated baryonic and stellar mass and have one entry per group
grpmb_arr = np.zeros(len(grp_keys))
grpms_arr = np.zeros(len(grp_keys))
logmh_s_arr = np.zeros(len(grp_keys))
logmh_arr = np.zeros(len(grp_keys))
grprmag_arr = np.zeros(len(grp_keys))
cenmstar_arr = np.zeros(len(grp_keys))
ra_arr = np.zeros(len(grp_keys))
dec_arr = np.zeros(len(grp_keys))
cz_arr = np.zeros(len(grp_keys))
for idx,key in enumerate(grp_keys):
    group = grps.get_group(key)
    grpmb = np.unique(group.grpmb.values)[0]
    grpms = np.unique(group.grpms.values)[0]
    logmh_s = np.unique(group.logmh_s.values)[0] # same number for all
    logmh = np.unique(group.logmh.values)[0] # same number for all
    grprmag = np.unique(group.grpabsrmag.values)[0]
    cenmstar = group.logmstar.loc[group.fc.values == 1].values[0]
    ra = group.radeg.loc[group.fc.values == 1].values[0] # central
    dec = group.dedeg.loc[group.fc.values == 1].values[0] # central
    cz = np.unique(group.grpcz.values)[0]
    grpms_arr[idx] = grpms
    grpmb_arr[idx] = grpmb
    logmh_s_arr[idx] = logmh_s
    logmh_arr[idx] = logmh
    grprmag_arr[idx] = grprmag
    cenmstar_arr[idx] = cenmstar
    ra_arr[idx] = ra
    dec_arr[idx] = dec
    cz_arr[idx] = cz

# Create cumulative baryonic and stellar mass functions
bins_sm = np.linspace(8.9, 12.2, 12)
bins_bm = np.linspace(9.4, 12.2, 12)
v_eco = 151829.26 * 2.915
bin_centers_grpmb,bin_edges_grpmb,n_grpmb,err_poiss_grpmb,bin_width_grpmb = \
    cumu_num_dens(grpmb_arr,bins_bm,None,v_eco,False)
bin_centers_grpms,bin_edges_grpms,n_grpms,err_poiss_grpms,bin_width_grpms = \
    cumu_num_dens(grpms_arr,bins_sm,None,v_eco,False)

# Load halo catalog
halo_table = pd.read_csv(path_to_interim + 'id_macc.csv',header=0)
v_sim = 130**3 * 2.915 # Vishnu simulation volume (Mpc/h)^3 h=0.7

# Use only host halos for HAM (PID and UPID of -1) with masses in h=0.7
halo_mass_hh = halo_table.halo_macc.loc[halo_table.C_S.values==1] * 1.429 

# Create HMF
bins = num_bins(halo_mass_hh)
bin_centers_hmass,bin_edges_hmass,n_hmass,err_poiss_hmass,\
    bin_width_hmass = cumu_num_dens(halo_mass_hh,bins,None,v_sim,False)

# Interpolating between grpmb and n
grpmb_n_interp_func = interpolate.interp1d(bin_centers_grpmb,n_grpmb,\
    fill_value='extrapolate')

# Interpolating between grpmstar and n
grpms_n_interp_func = interpolate.interp1d(bin_centers_grpms,n_grpms,\
    fill_value='extrapolate')

# Interpolating between central hmass and n and reversing it so you can pass 
# an n and get central hmass value
hmass_n_interp_func = interpolate.interp1d(n_hmass,bin_centers_hmass, \
    fill_value='extrapolate')

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

df_halomasses = {'baryonic': hmass_loggrpmb, 'stellar': hmass_loggrpms, 'absmag': logmh_arr,
'resolve_stellar': logmh_s_arr}
df_halomasses = pd.DataFrame(data=df_halomasses)

### Get error bars
x_grpmb,y_grpmb,y_std_grpmb,y_std_err_grpmb = Stats_one_arr(hmass_loggrpmb,\
    cenmstar_arr,base=0.3)

x_grpms,y_grpms,y_std_grpms,y_std_err_grpms = Stats_one_arr(hmass_loggrpms,\
    cenmstar_arr,base=0.3)

x_mhs,y_mhs,y_std_mhs,y_std_err_mhs = Stats_one_arr(logmh_s_arr,cenmstar_arr,\
    base=0.3)

x_mh,y_mh,y_std_mh,y_std_err_mh = Stats_one_arr(logmh_arr,cenmstar_arr,base=0.3)

y_std_err_grpmb = np.sqrt(y_std_err_grpmb**2 + (0.30**2))
y_std_err_grpms = np.sqrt(y_std_err_grpms**2 + (0.30**2))
y_std_err_mhs = np.sqrt(y_std_err_mhs**2 + (0.30**2))
y_std_err_mh = np.sqrt((y_std_err_mh**2) + (0.30**2))

fig1 = plt.figure(figsize=(10,10))
plt.errorbar(x_grpmb,y_grpmb,yerr=y_std_err_grpmb,\
    color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
        capthick=0.5,label=r'$M_{bary,grp}$')
plt.errorbar(x_grpms,y_grpms,yerr=y_std_err_grpms,\
    color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
        capthick=0.5,label=r'$M_{\star ,grp}$')
plt.errorbar(x_mhs,y_mhs,yerr=y_std_err_mhs,\
    color='#a0298d',fmt='--s',ecolor='#a0298d',markersize=4,capsize=5,\
        capthick=0.5,label=r'RESOLVE stellar mass derived')
plt.errorbar(x_mh,y_mh,yerr=y_std_err_mh,\
    color='k',fmt='--s',ecolor='k',markersize=4,capsize=5,\
        capthick=0.5,label=r'RESOLVE r-band mag derived')

plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'\boldmath$\log\ M_{\star,c} \left[M_\odot \right]$')
plt.legend(loc='best',prop={'size': 10})
plt.show()

################################### Corrfunc ###################################
dict_corrfunc = {'RA':ra_arr, 'DEC':dec_arr, 'grpcz':cz_arr, \
    'logmh_s':logmh_s_arr, 'logmh':logmh_arr, 'logmh_grpmb':hmass_loggrpmb, \
        'logmh_grpms':hmass_loggrpms}
df_corrfunc = pd.DataFrame(dict_corrfunc)

idx = int(10/100*(len(df_corrfunc)))
df_corrfunc = df_corrfunc.sort_values('logmh', ascending=False)
logmh_high10 = df_corrfunc[:idx]
logmh_low10 = df_corrfunc[-idx:]
df_corrfunc = df_corrfunc.sort_values('logmh_s', ascending=False)
logmhs_high10 = df_corrfunc[:idx]
logmhs_low10 = df_corrfunc[-idx:]
df_corrfunc = df_corrfunc.sort_values('logmh_grpmb', ascending=False)
logmhgrpb_high10 = df_corrfunc[:idx]
logmhgrpb_low10 = df_corrfunc[-idx:]
df_corrfunc = df_corrfunc.sort_values('logmh_grpms', ascending=False)
logmhgrps_high10 = df_corrfunc[:idx]
logmhgrps_low10 = df_corrfunc[-idx:]

# Real galaxies
RA = logmhs_high10.RA.values
DEC = logmhs_high10.DEC.values
CZ = logmhs_high10.grpcz.values
hm = [logmhs_high10, logmhs_low10, logmh_high10, logmh_low10, logmhgrpb_high10,\
    logmhgrpb_low10, logmhgrps_high10, logmhgrps_low10]
bins_arr = []
wp_arr = []
for idx,value in enumerate(hm):
    print(value)
    RA = value.RA.values
    DEC = value.DEC.values
    CZ = value.grpcz.values
    bins,wp = get_wp(RA,DEC,CZ)
    bins_arr.append(bins)
    wp_arr.append(wp)

"""
fracdiff_bary_arr = np.zeros(460)
for idx,predicted in enumerate(halomass_df.baryonic.values):
    truth = halomass_df.resolve_stellar.values[idx]
    fracdiff_bary = 100*((predicted/truth)-1)
    fracdiff_bary_arr[idx] = fracdiff_bary

fracdiff_stellar_arr = np.zeros(460)
for idx,predicted in enumerate(halomass_df.stellar.values):
    truth = halomass_df.resolve_stellar.values[idx]
    fracdiff_stellar = 100*((predicted/truth)-1)
    fracdiff_stellar_arr[idx] = fracdiff_stellar

fracdiff_absmag_arr = np.zeros(460)
for idx,predicted in enumerate(halomass_df.absmag.values):
    truth = halomass_df.resolve_stellar.values[idx]
    fracdiff_absmag = 100*((predicted/truth)-1)
    fracdiff_absmag_arr[idx] = fracdiff_absmag

fracdiff_rstellar_arr = np.zeros(460)
for idx,predicted in enumerate(halomass_df.resolve_stellar.values):
    truth = halomass_df.resolve_stellar.values[idx]
    fracdiff_rstellar = 100*((predicted/truth)-1)
    fracdiff_rstellar_arr[idx] = fracdiff_rstellar


### Use stats_one_arr with true halo mass and fractional difference
x_grpmb,y_grpmb,y_std_grpmb,y_std_err_grpmb = Stats_one_arr(hmass_loggrpmb,\
    fracdiff_bary_arr,base=0.3,bin_statval='left')
x_grpms,y_grpms,y_std_grpms,y_std_err_grpms = Stats_one_arr(hmass_loggrpms,\
    fracdiff_stellar_arr,base=0.3,bin_statval='left')
x_grpabsmag,y_grpabsmag,y_std_grpabsmag,y_std_err_grpabsmag = Stats_one_arr(logmh_arr,\
    fracdiff_absmag_arr,base=0.3,bin_statval='left')
x_grprs,y_grprs,y_std_grprs,y_std_err_grprs = Stats_one_arr(logmh_s_arr,\
    fracdiff_rstellar_arr,base=0.3,bin_statval='left')

fig1 = plt.figure(figsize=(10,10))
plt.errorbar(x_grpmb,y_grpmb,yerr=y_std_err_grpmb,\
    color='#1baeab',fmt='--s',ecolor='#1baeab',markersize=4,capsize=5,\
        capthick=0.5,label=r'$M_{bary,grp}$')
plt.errorbar(x_grpms,y_grpms,yerr=y_std_err_grpms,\
    color='#f6a631',fmt='--s',ecolor='#f6a631',markersize=4,capsize=5,\
        capthick=0.5,label=r'$M_{\star ,grp}$')
plt.errorbar(x_grpabsmag,y_grpabsmag,yerr=y_std_err_grpabsmag,\
    color='#a0298d',fmt='--s',ecolor='#a0298d',markersize=4,capsize=5,\
        capthick=0.5,label=r'RESOLVE r-band mag derived')
plt.errorbar(x_grprs,y_grprs,yerr=y_std_err_grprs,\
    color='k',fmt='--s',ecolor='k',markersize=4,capsize=5,\
        capthick=0.5,label=r'RESOLVE stellar mass derived')

plt.xlabel(r'\boldmath$\log\ M_h \left[M_\odot \right]$')
plt.ylabel(r'Fractional difference')
plt.legend(loc='best',prop={'size': 10})
plt.show()
"""