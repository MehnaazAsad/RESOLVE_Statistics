"""
{This script compares the cumulative luminosity function used for AM in Victor's 
 mock-making script and the data I use to compare with mocks now.}
"""

from cosmo_utils.utils import file_readers as freader
from cosmo_utils.utils import work_paths as cwpaths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import math

def read_data(path_to_file, survey):
    """
    Reads survey catalog from file

    Parameters
    ----------
    path_to_file: `string`
        Path to survey catalog file

    survey: `string`
        Name of survey

    Returns
    ---------
    catl: `pandas.DataFrame`
        Survey catalog with grpcz, abs rmag and stellar mass limits
    
    volume: `float`
        Volume of survey

    cvar: `float`
        Cosmic variance of survey

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
                    'fc', 'grpmb', 'grpms']

        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0, \
            usecols=columns)

        # 6456 galaxies                       
        catl = eco_buff.loc[(eco_buff.cz.values >= 3000) & \
            (eco_buff.cz.values <= 7000) & (eco_buff.absrmag.values <= -17.33)
            & (eco_buff.absrmag.values >= -23.5)]

        volume = 192351.36 # Survey volume with buffer [Mpc/h]^3
        # volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def cumu_num_dens(data, weights, volume, bool_mag): 
    if weights is None: 
        weights = np.ones(len(data)) 
    else: 
        weights = np.array(weights) 
    #Unnormalized histogram and bin edges 
    data += 0.775 #change mags from h=0.7 to h=1
    bins = np.arange(math.floor(data.min()), math.ceil(data.max()), 0.2) 
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

def Schechter_func(M,phi_star,M_star,alpha):
    const = 0.4*np.log(10)*phi_star
    first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
    second_exp_term = np.exp(-10**(0.4*(M_star-M)))
    return const*first_exp_term*second_exp_term
    
__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_data  = dict_of_paths['data_dir']
path_to_raw   = dict_of_paths['raw_dir']

survey = 'eco'

# Path to files
catl_current     = path_to_raw + 'eco/eco_all.csv'
catl_katie_mocks = path_to_raw + 'gal_Lr_Mb_Re.txt'
catl_vc_mocks    = path_to_raw + 'eco_wresa_050815.dat'
weights_vc_mocks = path_to_raw + 'eco_wresa_050815_weightmuiso.dat'

# * NOTE: volume_current is without the buffer
catl_current, volume_current, cvar, z_median = read_data(catl_current, survey)
mr_current      = catl_current.absrmag.values

vc_data         = pd.DataFrame(freader.IDL_read_file(catl_vc_mocks))
vc_weights      = pd.DataFrame(freader.IDL_read_file(weights_vc_mocks))
volume_vc       = 192294.221932 #(Mpc/h)^3
mr_vc           = np.array(vc_data['goodnewabsr'])
vc_weights      = np.array(vc_weights['corrvecinterp'])
# mr_cut_vc       = [(index,value) for index,value in enumerate(mr_vc) if \
#                     value <= -17.33]
mr_cut_vc       = [(index,value) for index,value in enumerate(mr_vc) if \
                    value <= -17.33 and value >= -23.5]
### Getting indices of values that fall between cuts
mr_cut_idx      = [pair[0] for pair in mr_cut_vc]
vc_weights      = vc_weights[mr_cut_idx]

# catl_katie_mocks   = pd.read_csv(catl_katie_mocks, delimiter='\s+', 
#     header=None, skiprows=2, names=['M_r','logmbary','Re'])
# volume_katie = 192351.36 #Volume of ECO with buffer in (Mpc/h)^3

mag_cen, mag_edg, mag_n, mag_err, bw = cumu_num_dens(mr_current, None, 
    volume_current, True) 

mag_cen_vc, mag_edg_vc, mag_n_vc, mag_err_vc, bw_vc = \
    cumu_num_dens(np.array([pair[1] for pair in mr_cut_vc]), 
    vc_weights, volume_vc, True) 

## Plotting
fig1 = plt.figure(figsize=(10,8))
plt.errorbar(mag_cen,mag_n,yerr=mag_err,markersize=4,capsize=5,
    capthick=2, marker='o', label='current',color='r',fmt='-s',ecolor='r',
    linewidth=2,zorder=10)
plt.errorbar(mag_cen_vc,mag_n_vc,yerr=mag_err_vc,markersize=4,capsize=5,
    capthick=2, marker='o', label='vc mock AM',color='g',fmt='-s',ecolor='g',
    linewidth=2,zorder=10)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.legend(loc='best')
plt.title('Luminosity function of data catalogs used')
plt.show()
