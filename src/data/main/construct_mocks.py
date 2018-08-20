#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:10:35 2018

@author: asadm2
"""
import matplotlib as mpl
mpl.use('agg')
from cosmo_utils.utils import work_paths as cwpaths
#from cosmo_utils.utils import file_readers as freader
from scipy.optimize import curve_fit
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rc
import pandas as pd
import numpy as np
import sympy
import math
import csv

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)

### Schechter function
def Schechter_func(M,phi_star,M_star,alpha):
    const = 0.4*np.log(10)*phi_star
    first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
    second_exp_term = np.exp(-10**(0.4*(M_star-M)))
    return const*first_exp_term*second_exp_term

### Differential method
def diff_num_dens(data,nbins,weights,volume,mag_bool):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=nbins,weights=weights)     
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    err_poiss = np.sqrt(freq)/(volume*bin_width)
    n_diff = freq/(volume*bin_width)
    return bin_centers,edg,n_diff,err_poiss

### Given data calculate how many bins should be used
def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

### Data file used
eco_obs_catalog = pd.read_csv(path_to_raw + 'gal_Lr_Mb_Re.txt',\
                              delimiter='\s+',header=None,skiprows=2,\
                              names=['M_r','logmbary','Re'])

Mr_all = eco_obs_catalog.M_r.values
Mr_unique = np.unique(eco_obs_catalog.M_r.values)
v_eco = 192351.36 #Volume of ECO with buffer in (Mpc/h)^3
min_Mr = min(Mr_all)
max_Mr = max(Mr_all)

### Data and weights from VC
#VC_data_filename = 'eco_wresa_050815.dat'
#VC_weights_filename = 'eco_wresa_050815_weightmuiso.dat'
#VC_data = pd.DataFrame(freader.IDL_read_file(path_to_raw + VC_data_filename))
#VC_weights = pd.DataFrame(freader.IDL_read_file(path_to_raw + \
#                                                VC_weights_filename))
#Mr_VC = np.array(VC_data['goodnewabsr'])
#weights_VC = np.array(VC_weights['corrvecinterp'])
#volume_VC = 192294.221932 #(Mpc/h)^3

### Using SF to fit data between -17.33 and -23.5 and extrapolating both ends
Mr_cut = [value for value in Mr_all if value <= -17.33 and value >= -23.5]
max_Mr_cut = max(Mr_cut)
min_Mr_cut = min(Mr_cut)
nbins = 25 #better chi-squared than 35 from num_bins function

### Calculate differential number density using function
bin_centers_cut,bin_edges_cut,n_Mr_cut,err_poiss_cut = diff_num_dens(Mr_cut,\
                                                                     nbins,\
                                                                     None,\
                                                                     v_eco,\
                                                                     mag_bool=\
                                                                     True)

p0 = [10**-2,-20,-1.2] #initial guess for phi_star,M_star,alpha

### Curve fit using Schechter function defined
params_noextrap,pcov = curve_fit(Schechter_func,bin_centers_cut,n_Mr_cut,p0,\
                                 sigma=err_poiss_cut,absolute_sigma=True,\
                                 maxfev=20000,method='lm')

### Best fit parameters from curve fit
Phi_star_bestfit = params_noextrap[0]
M_star_bestfit = params_noextrap[1]
Alpha_bestfit = params_noextrap[2]

### Fitting data between -17.33 and -23.5
fit_noextrap = Schechter_func(np.linspace(min_Mr_cut,max_Mr_cut),\
                        Phi_star_bestfit,M_star_bestfit,\
                        Alpha_bestfit)

### Extrapolations on both ends
fit_extrap_dim = Schechter_func(np.linspace(max_Mr_cut,max_Mr),\
                      Phi_star_bestfit,M_star_bestfit,\
                      Alpha_bestfit)

fit_extrap_bright = Schechter_func(np.linspace(min_Mr,min_Mr_cut),\
                      Phi_star_bestfit,M_star_bestfit,\
                      Alpha_bestfit)

### Percentage fractional difference
data_fit_frac = []
for index,value in enumerate(bin_centers_cut):
    n_data = n_Mr_cut[index]
    n_fit = Schechter_func(value,Phi_star_bestfit,M_star_bestfit,\
                     Alpha_bestfit)
    frac_diff = np.abs(((n_fit-n_data)/n_data)*100)
    data_fit_frac.append(frac_diff)
    
### Chi_squared 
model_fit = Schechter_func(bin_centers_cut,\
                        Phi_star_bestfit,M_star_bestfit,\
                        Alpha_bestfit)
chi2 = 0
for i in range(len(bin_centers_cut)):
    chi2_i = ((n_Mr_cut[i]-model_fit[i])**2)/((err_poiss_cut[i])**2)
    chi2 += chi2_i
###############################################################################
'''
### Using SF to fit VC's data and fit it
### Applying same cut as VC
Mr_cut_VC = [(index,value) for index,value in enumerate(Mr_VC) if \
             value <= -17.33 and value >= -23.5]
### Getting indices of values that fall between cuts
Mr_cut_idx = [pair[0] for pair in Mr_cut_VC]
### Getting values that fall between cuts
Mr_cut_data = [pair[1] for pair in Mr_cut_VC]
### Using indices to get weights for values that fall between cuts
weights_VC = weights_VC[Mr_cut_idx]

max_Mr_cut_VC = max(Mr_cut_data)
min_Mr_cut_VC = min(Mr_cut_data)
nbins_VC = 25

### Calculating differential number density using function
bin_centers_cut_VC,bin_edges_cut_VC,n_Mr_cut_VC,\
err_poiss_cut_VC = diff_num_dens(Mr_cut_data,nbins_VC,weights_VC,volume_VC,\
                                 mag_bool=True)

### Deleting last edge to avoid shape mismatch while plotting
bin_edges_cut_VC = np.delete( bin_edges_cut_VC[::1], -1)

### Curve fit using Schechter function defined
params_VC,pcov_VC = curve_fit(Schechter_func,bin_centers_cut_VC,n_Mr_cut_VC,\
                              p0,sigma=err_poiss_cut_VC,absolute_sigma=True,\
                              maxfev=20000,method='lm')

### Fitting data between -17.33 and -23.5
fit_VC = Schechter_func(np.linspace(min_Mr_cut_VC,max_Mr_cut_VC),\
                        params_VC[0],params_VC[1],\
                        params_VC[2])

### Chi_squared 
model_fit_VC = Schechter_func(bin_edges_cut_VC,\
                        params_VC[0],params_VC[1],\
                        params_VC[2])
chi2_VC = 0
for i in range(len(bin_edges_cut_VC)):
    chi2_i = ((n_Mr_cut_VC[i]-model_fit_VC[i])**2)/((err_poiss_cut_VC[i])**2)
    chi2_VC += chi2_i
'''   
#################################PLOTS#########################################
### Plotting data + fit and extrapolation
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=False,figsize=(20,20),\
     gridspec_kw = {'height_ratios':[8,2]})
ax1.set_yscale('log')
ax1.axvline(-17.33,ls='--',c='g',label='ECO/RESOLVE-A')
ax1.axvline(-17,ls='--',c='r',label='RESOLVE-B')
ax1.invert_xaxis()
### Data + errorbars
ax1.errorbar(bin_centers_cut,n_Mr_cut,yerr=err_poiss_cut,fmt="ks--",ls='None',\
             elinewidth=0.5,ecolor='k',capsize=5,capthick=0.5,markersize=4,\
             label='data between {0} and -17.33'.format\
             (np.round(min_Mr_cut,1)))
### Data fit
ax1.plot(np.linspace(min_Mr_cut,max_Mr_cut),fit_noextrap,'--k',\
         label=r'$\mathrm\chi^{2} = %s$' %(np.round(chi2,2)))
### Extrapolation on dim end
ax1.plot(np.linspace(max_Mr_cut,max_Mr),fit_extrap_dim,'--y',label='extrap')
### Extrapolation on bright end
ax1.plot(np.linspace(min_Mr,min_Mr_cut),fit_extrap_bright,'--y')
### VC data and fit
#ax1.errorbar(bin_edges_cut_VC,n_Mr_cut_VC,yerr=err_poiss_cut_VC,fmt="gs--",\
#             ls='None',elinewidth=0.5,ecolor='g',capsize=5,capthick=0.5,\
#             markersize=4,label='VC data between {0} and -17.33'.format\
#             (np.round(min_Mr_cut_VC,1)))
#ax1.plot(np.linspace(min_Mr_cut_VC,max_Mr_cut_VC),fit_VC,'--g',\
#         label=r'$\mathrm\chi^{2} = %s$' %(np.round(chi2_VC,2)))

ax1.set_ylabel(r'$\mathrm{[\frac{dn}{dmag}]}/[\mathrm{h}^{3}\mathrm{Mpc}^{-3}'
                '\mathrm{mag}^{-1}]$')
ax1.legend(loc='upper right', prop={'size': 12})

### Percentage fractional difference
ax2.scatter(bin_centers_cut,data_fit_frac,c='k')
ax2.set_xlabel(r'$M_{r}$')
ax2.set_ylabel(r'$\mathrm{[\frac{model-data}{data}]} \%$')
fig.tight_layout()
fig.show()
###############################################################################
### Inverting the Schechter function using sympy so that it will return 
### magnitudes given a number density

### Make all parameters symbols
n,M,phi_star,M_star,alpha = sympy.symbols('n,M,phi_star,M_star,alpha')
const = 0.4*sympy.log(10)*phi_star
first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
second_exp_term = sympy.exp(-10**(0.4*(M_star-M)))
### Make expression that will be an input for sympy.solve
expr = (const*first_exp_term*second_exp_term)-n
### Get schechter function in terms of M
result = sympy.solve(expr,M,quick=True)
### Make result a lambda function so you can pass parameter and variable values
result_func = sympy.lambdify((n,phi_star,M_star,alpha),result,\
                             modules=["sympy"])

### Get parameter values and change variable names because the original ones
### are now symbols
phi_star_num = params_noextrap[0]
M_star_num = params_noextrap[1]
alpha_num = params_noextrap[2]
### Pick a magnitude range
M_num = np.linspace(-23.5,-17.33,200)
### Calculate n given range of M values
n_num = Schechter_func(M_num,phi_star_num,M_star_num,alpha_num)
### Given the n values just calculated use the lambda function in terms of n to 
### test whether you get back the same magnitude values as M_num
M_test = [result_func(val,phi_star_num,M_star_num,alpha_num) for val in n_num]
### Plot both to make sure they overlap
#fig2 = plt.figure(figsize=(10,10))
#plt.scatter(M_num,n_num,c='b',label='n given M',s=15)
#plt.scatter(M_test,n_num,c='r',label='M given n',s=5)
#plt.legend(loc='best')
#plt.show()

## WORK IN PROGRESS
### Halo data
halo_prop_table = pd.read_csv(path_to_interim + 'halo_vpeak.csv',header=None,\
                              names=['vpeak'])
v_sim = 130**3 #(Mpc/h)^3
vpeak = halo_prop_table.vpeak.values
nbins = num_bins(vpeak)
bin_centers_vpeak,bin_edges_vpeak,n_vpeak,err_poiss = diff_num_dens(vpeak,\
                                                                    nbins,\
                                                                    None,\
                                                                    v_sim,\
                                                                    mag_bool=\
                                                                    False)

f_h = interpolate.InterpolatedUnivariateSpline(bin_centers_vpeak,n_vpeak)

pbar = ProgressBar(maxval=len(vpeak))
n_vpeak_arr = [f_h(val) for val in pbar(vpeak)]
pbar = ProgressBar(maxval=len(n_vpeak_arr))
halo_Mr_sham = np.array([result_func(val,phi_star_num,M_star_num,alpha_num) \
                         for val in pbar(n_vpeak_arr)])
halo_Mr_sham = np.ndarray.flatten(halo_Mr_sham)

with open(path_to_interim + 'SHAM.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(vpeak,halo_Mr_sham))
#fig3 = plt.figure()
##plt.xscale('log')
#plt.gca().invert_yaxis()
#plt.scatter(vpeak,halo_Mr_sham,s=5)
#plt.ylabel(r'$M_{r}$')
#plt.xlabel(r'$v_{peak} /\mathrm{km\ s^{-1}}$')
#plt.show()

##############################################################################
#plt.scatter(Mr_unique,n_Mr,c='r',s=5,label='unique bins')
#plt.scatter(bin_centers,n_Mr_2,c='g',s=5,label='cum sum with unique bins')
#plt.scatter(bin_centers_2,n_Mr_3,c='b',s=5,label='cum sum with FD bins')
#plt.fill_between(bin_centers_cut,np.log10(n_Mr_cut-err_poiss_cut),\
#                 np.log10(n_Mr_cut+err_poiss_cut),alpha=0.5,\
#                 label='data until -17.33')
##plt.plot(np.linspace(min_Mr,max_Mr),fit_alldata,'--g',label='all data fit')
#f_h = interpolate.interp1d(n_vpeak,bin_centers_vpeak,fill_value="extrapolate")
#    mbary = eco_obs_catalog.loc[eco_obs_catalog['M_r']==mag_value,'logmbary']
#    mbary_arr.append(mbary)
#    re = eco_obs_catalog.loc[eco_obs_catalog['M_r'] == mag_value, 'Re']
#    re_arr.append(re)

#    if mag_bool:
#        n_diff = np.cumsum(n_diff) 
#    else:
#        n_diff = np.cumsum(np.flip(n_diff,0))
#        n_diff = np.flip(n_diff,0)
'''
## Parameter grid search 
Phi_star_Steps = 50
M_star_Steps = 60
Alpha_Steps = 100
Number_Sigmas = 20
Phi_star_min = Phi_star_bestfit - 0.5*Number_Sigmas * np.sqrt( pcov[0,0] )
Phi_star_max = Phi_star_bestfit + 2.0*Number_Sigmas * np.sqrt( pcov[0,0] )
Phi_star_array = np.arange( Phi_star_min, Phi_star_max, \
                           (Phi_star_max - Phi_star_min )/( Phi_star_Steps) )

M_star_min = M_star_bestfit - 0.5*Number_Sigmas * np.sqrt( pcov[1,1] )
M_star_max = M_star_bestfit + 2.0*Number_Sigmas * np.sqrt( pcov[1,1] )
M_star_array = np.arange( M_star_min, M_star_max, \
                           (M_star_max - M_star_min )/( M_star_Steps) )

Alpha_min = Alpha_bestfit - ( 1.0*Number_Sigmas + 0) * np.sqrt( pcov[2,2] )
Alpha_max = Alpha_bestfit + ( 1.0*Number_Sigmas + 5) * np.sqrt( pcov[2,2] )
Alpha_array = np.arange( Alpha_min, Alpha_max, \
                        (Alpha_max - Alpha_min )/( Alpha_Steps) )

Chi2_3D_array = np.zeros([Phi_star_Steps, M_star_Steps, Alpha_Steps])
for s1 in range(Phi_star_Steps):
    for s2 in range(M_star_Steps):
        for s3 in range(Alpha_Steps):
            Phi_star_Test = Phi_star_array[s1]
            M_star_Test = M_star_array[s2]
            Alpha_Test = Alpha_array[s3]
            chi2 = 0
            dens_array = [ [] for x in range(len(n_Mr_cut))]
            Mag_array = [ [] for x in range(len(n_Mr_cut))]
            for i in range(len(n_Mr_cut)):
                dens_array[i] = Schechter_func(bin_centers_cut[i],Phi_star_Test,\
                                            M_star_Test,Alpha_Test)
                Residual = (n_Mr_cut[i]-dens_array[i])/err_poiss_cut[i]
                chi2 += Residual**2
            dens_array = np.array(dens_array)
            Chi2_3D_array[s1, s2, s3] = chi2
'''
#x_h = np.linspace(min(vpeak),max(vpeak),num=200)
#ynew_h = f_h(x_h)

#fig2 = plt.figure()
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(r'$v_{peak} /\mathrm{km\ s^{-1}}$')
#plt.ylabel(r'$n(> v_{peak})/\mathrm{Mpc}^{-3}\mathrm{{v}_{peak}}^{-1}$')
#plt.scatter(bin_centers_vpeak,n_vpeak,label='data',s=2)
#plt.plot(x_h,ynew_h,'--k',label='interp/exterp')
#plt.legend(loc='best')
#plt.show()
#print(ii/np.float(size),"/033[1A]")
