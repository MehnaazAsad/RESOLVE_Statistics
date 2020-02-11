"""
{This script carries out HAM and SHAM using baryonic and stellar masses of 
 groups and individual galaxies and compares to the values from RESOLVE B}
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

RESOLVE_B = resolve_live18.loc[resolve_live18['f_b'] == 1] #793 galaxies
RESOLVE_B = RESOLVE_B.reset_index(drop=True)
