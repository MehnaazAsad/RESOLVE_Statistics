
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
# halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/vishnu/'\
# 'rockstar/vishnu_rockstar_test.hdf5'
halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)
plt.rcParams['animation.convert_path'] = '{0}/magick'.format(os.path.dirname(which('python')))
#'/fs1/masad/anaconda3/envs/resolve_statistics/bin/magick'


def diff_smf(mstar_arr, volume, h1_bool):
    """
    Calculates differential stellar mass function in units of h=1.0

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

    cvar_err: float
        Cosmic variance of survey

    h1_bool: boolean
        True if units of masses are h=1, False if units of masses are not h=1

    Returns
    ---------
    maxis: array
        Array of x-axis mass values

    phi: array
        Array of y-axis values

    err_tot: array
        Array of error values per bin
    
    bins: array
        Array of bin edge values
    """
    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = np.log10(mstar_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        if survey == 'eco':
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        elif survey == 'resolvea':
            # different to avoid nan in inverse corr mat
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss
    phi = counts / (volume * dm)  # not a log quantity

    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

def read_catl(path_to_file):
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

        if mf_type == 'smf':
            # 6456 galaxies                       
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33) &
                (eco_buff.logmstar.values >= 8.9)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & 
                (eco_buff.grpcz.values <= 7000) & 
                (eco_buff.absrmag.values <= -17.33)] 

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)
        
    elif survey == 'resolvea' or survey == 'resolveb':
        columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh', 
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
            usecols=columns)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33) & 
                    (resolve_live18.logmstar.values >= 8.9)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)
        
        elif survey == 'resolveb':
            if mf_type == 'smf':
                # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17) & 
                    (resolve_live18.logmstar.values >= 8.7)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def jackknife(catl, volume):
    """
    Jackknife ECO survey to get data in error and correlation matrix for 
    chi-squared calculation

    Parameters
    ----------
    catl: Pandas DataFrame
        Survey catalog

    Returns
    ---------
    stddev_jk: numpy array
        Array of sigmas
    corr_mat_inv: numpy matrix
        Inverse of correlation matrix
    """

    ra = catl.radeg.values # degrees
    dec = catl.dedeg.values # degrees

    sin_dec_all = np.rad2deg(np.sin(np.deg2rad(dec))) # degrees

    sin_dec_arr = np.linspace(sin_dec_all.min(), sin_dec_all.max(), 11)
    ra_arr = np.linspace(ra.min(), ra.max(), 11)

    grid_id_arr = []
    gal_id_arr = []
    grid_id = 1
    max_bin_id = len(sin_dec_arr)-2 # left edge of max bin
    for dec_idx in range(len(sin_dec_arr)):
        for ra_idx in range(len(ra_arr)):
            try:
                if dec_idx == max_bin_id and ra_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
                elif dec_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values < ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) <= sin_dec_arr[dec_idx+1])] 
                elif ra_idx == max_bin_id:
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values <= ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
                else:                
                    catl_subset = catl.loc[(catl.radeg.values >= ra_arr[ra_idx]) &
                        (catl.radeg.values < ra_arr[ra_idx+1]) & 
                        (np.rad2deg(np.sin(np.deg2rad(catl.dedeg.values))) >= 
                            sin_dec_arr[dec_idx]) & (np.rad2deg(np.sin(np.deg2rad(
                                catl.dedeg.values))) < sin_dec_arr[dec_idx+1])] 
                # Append dec and sin  
                for gal_id in catl_subset.name.values:
                    gal_id_arr.append(gal_id)
                for grid_id in [grid_id] * len(catl_subset):
                    grid_id_arr.append(grid_id)
                grid_id += 1
            except IndexError:
                break

    gal_grid_id_data = {'grid_id': grid_id_arr, 'name': gal_id_arr}
    df_gal_grid = pd.DataFrame(data=gal_grid_id_data)

    catl = catl.join(df_gal_grid.set_index('name'), on='name')
    catl = catl.reset_index(drop=True)

    # Loop over all sub grids, remove one and measure global smf
    jackknife_phi_arr = []
    for grid_id in range(len(np.unique(catl.grid_id.values))):
        grid_id += 1
        catl_subset = catl.loc[catl.grid_id.values != grid_id]  
        logmstar = catl_subset.logmstar.values
        maxis, phi, err, bins, counts = diff_smf(logmstar, volume, False)
        jackknife_phi_arr.append(phi)

    jackknife_phi_arr = np.array(jackknife_phi_arr)

    N = (len(ra_arr)-1)**2

    # Covariance matrix
    cov_mat = np.cov(jackknife_phi_arr.T, bias=True)*(N-1)
    stddev_jk = np.sqrt(cov_mat.diagonal())

    return stddev_jk

global survey
global mf_type
survey = 'eco'
mf_type = 'smf'

path_to_eco = path_to_raw + "eco_all.csv"
catl, volume, cvar, z_median = read_catl(path_to_eco)
logmstar = catl.logmstar.values
maxis_data, phi_data, err_data, bins, counts = diff_smf(logmstar, volume, False)
err_data = jackknife(catl, volume)
lower_err = phi_data - err_data
upper_err = phi_data + err_data
lower_err = phi_data - lower_err
upper_err = upper_err - phi_data
asymmetric_err = [lower_err, upper_err]

# fig1 = plt.figure(figsize=(10,8))
# plt.xlim(np.log10((10**8.9)/2.041),np.log10((10**11.5)/2.041))
# plt.errorbar(maxis_data, phi_data, yerr=asymmetric_err, fmt="rs--", linewidth=2, 
#     elinewidth=0.5, ecolor='r', capsize=5, capthick=0.5 )
# plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=15)
# plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=15)
# plt.title('Stellar mass function')


counter = 0
nbins = bins
volume_sim = 130**3

Mhalo_characteristic = np.linspace(11.5,13.0,30)
Mstellar_characteristic = np.linspace(9.5,11.0,30)
Mlow_slope = np.linspace(0.35,0.50,30)
Mhigh_slope = np.linspace(0.50,0.65,30)
Mstellar_scatter = np.linspace(0.1,0.2,30)

print('Setting up models')
###Models
model1 = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_median,
    prim_haloprop_key='halo_macc')
model2 = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_median,
    prim_haloprop_key='halo_macc')
model3 = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_median,
    prim_haloprop_key='halo_macc')
model4 = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_median,
    prim_haloprop_key='halo_macc')
model5 = PrebuiltSubhaloModelFactory('behroozi10',redshift=z_median,
    prim_haloprop_key='halo_macc')

print('Setting up halocats')
###Halocats
halocat1 = CachedHaloCatalog(fname=halo_catalog)
halocat2 = CachedHaloCatalog(fname=halo_catalog)
halocat3 = CachedHaloCatalog(fname=halo_catalog)
halocat4 = CachedHaloCatalog(fname=halo_catalog)
halocat5 = CachedHaloCatalog(fname=halo_catalog)

print('Initial mock population')
###Populate mocks
model1.populate_mock(halocat1)
model2.populate_mock(halocat2)
model3.populate_mock(halocat3)
model4.populate_mock(halocat4)
model5.populate_mock(halocat5)

def gals(Mhalo_value,Mstellar_value,Mlow_slope,Mhigh_slope,Mstellar_scatter):
    ###Parameter values
    model1.param_dict['smhm_m1_0']              = Mhalo_value
    model2.param_dict['smhm_m0_0']              = Mstellar_value
    model3.param_dict['smhm_beta_0']            = Mlow_slope
    model4.param_dict['smhm_delta_0']           = Mhigh_slope
    model5.param_dict['scatter_model_param1']   = Mstellar_scatter
    
    model1.mock.populate()
    model2.mock.populate()
    model3.mock.populate()
    model4.mock.populate()
    model5.mock.populate()

    #Applying ECO stellar mass limit
    limit = np.log10((10**8.9)/2.041)
    sample_mask1 = model1.mock.galaxy_table['stellar_mass'] >= 10**limit
    sample_mask2 = model2.mock.galaxy_table['stellar_mass'] >= 10**limit
    sample_mask3 = model3.mock.galaxy_table['stellar_mass'] >= 10**limit
    sample_mask4 = model4.mock.galaxy_table['stellar_mass'] >= 10**limit
    sample_mask5 = model5.mock.galaxy_table['stellar_mass'] >= 10**limit
    
    Mhalo_gals            = model1.mock.galaxy_table[sample_mask1]
    Mstellar_gals         = model2.mock.galaxy_table[sample_mask2]
    Mlowslope_gals        = model3.mock.galaxy_table[sample_mask3]
    Mhighslope_gals       = model4.mock.galaxy_table[sample_mask4]
    Mstellarscatter_gals  = model5.mock.galaxy_table[sample_mask5]
    
    return Mhalo_gals, Mstellar_gals, Mlowslope_gals, Mhighslope_gals, \
        Mstellarscatter_gals
    
def init():
    
    line1 = ax1.errorbar([], [], yerr=[], fmt="s-", linewidth=2, 
        elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', capsize=2, 
        capthick=0.5)
    line2 = ax2.errorbar([], [], yerr=[], fmt="s-", linewidth=2, 
        elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', capsize=2, 
        capthick=0.5)   
    line3 = ax3.errorbar([], [], yerr=[], fmt="s-", linewidth=2, 
        elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', capsize=2, 
        capthick=0.5)    
    line4 = ax4.errorbar([], [], yerr=[], fmt="s-", linewidth=2, 
        elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', capsize=2, 
        capthick=0.5)    
    line5 = ax5.errorbar([], [], yerr=[], fmt="s-", linewidth=2, 
        elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', capsize=2, 
        capthick=0.5)
    line = [line1,line2,line3,line4,line5]
    
    return line

def make_animation(i,j,k,l,m):
    global counter
    Mhalo, Mstellar, Mlowslope, Mhighslope, Mstellarscatter = i,j[counter],\
    k[counter],l[counter],m[counter]
    ax1_catalog,ax2_catalog,ax3_catalog,ax4_catalog,ax5_catalog = \
    gals(Mhalo,Mstellar,Mlowslope,Mhighslope,Mstellarscatter)
    
    catalog_arr = [ax1_catalog,ax2_catalog,ax3_catalog,ax4_catalog,ax5_catalog]
    
    maxis_arr = []
    phi_arr = []
    err_arr = []
    for i in range(5):
        maxis, phi, err, bins, counts = diff_smf(catalog_arr[i]['stellar_mass'], 
            volume_sim, True)
        maxis_arr.append(maxis)
        phi_arr.append(phi)
        err_arr.append(err)
    
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.clear()
        SMF_ECO = ax.errorbar(maxis_data, phi_data, yerr=asymmetric_err, 
            fmt="ks", linewidth=2, elinewidth=0.5, ecolor='k', capsize=2, 
            capthick=1.5, markersize='5')
        ax.set_xlim(np.log10((10**8.9)/2.041),np.log10((10**11.8)/2.041))
        ax.set_ylim(-5,-1)
        ax.minorticks_on()
        fig.tight_layout()
        fig.subplots_adjust(left=0.1, bottom=0.1, wspace=0) 
        if ax in [ax2, ax3, ax5]:
            plt.setp(ax.get_yticklabels(), visible=False)
            # ax.set_xticks(ax.get_xticks()[1:]) 
    
    # Putting fig. lines here instead doesn't work
    ax1.set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=15)
    ax4.set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=15)
    ax4.set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=15)
    ax5.set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=15)
    
    line1 = ax1.errorbar(maxis_arr[0], phi_arr[0], fmt="s-",
        linewidth=2, elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', 
        capsize=2, capthick=1.5, markersize='3') 
    line2 = ax2.errorbar(maxis_arr[1], phi_arr[1], fmt="s-",
        linewidth=2, elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', 
        capsize=2, capthick=1.5, markersize='3')
    line3 = ax3.errorbar(maxis_arr[2], phi_arr[2], fmt="s-",
        linewidth=2, elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', 
        capsize=2, capthick=1.5, markersize='3') 
    line4 = ax4.errorbar(maxis_arr[3], phi_arr[3], fmt="s-",
        linewidth=2, elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', 
        capsize=2, capthick=1.5, markersize='3') 
    line5 = ax5.errorbar(maxis_arr[4], phi_arr[4], fmt="s-",
        linewidth=2, elinewidth=0.5, color='mediumorchid', ecolor='mediumorchid', 
        capsize=2, capthick=1.5, markersize='3') 

    ax1.legend([line1,SMF_ECO],[r'$M_{h}=%4.2f$' % Mhalo,'ECO'], 
        loc='lower left',prop={'size': 15})
    ax2.legend([line2,SMF_ECO],[r'$M_{*}=%4.2f$' % Mstellar,'ECO'],
        loc='lower left',prop={'size': 15})
    ax3.legend([line3,SMF_ECO],[r'$\beta=%4.2f$' % Mlowslope,'ECO'],
        loc='lower left',prop={'size': 15})
    ax4.legend([line4,SMF_ECO],[r'$\delta=%4.2f$' % Mhighslope,'ECO'],
        loc='lower left',prop={'size': 15})
    ax5.legend([line5,SMF_ECO],[r'$\xi=%4.3f$' % Mstellarscatter,'ECO'],
        loc='lower left',prop={'size': 15})
    
    print('Setting data')
    print('Frame {0}/{1}'.format(counter+1,len(Mhalo_characteristic)))
    
    counter+=1
    
    line = [line1,line2,line3,line4,line5]  
    
    return line

#Setting up the figure, the axis, and the plot element to animate
fig = plt.figure(figsize=(10,8))
# These two lines don't work unless they are also in make_animation
fig.tight_layout()
fig.subplots_adjust(left=0.1, bottom=0.1, wspace=0) 
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2, sharey=ax1)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2, sharey=ax1)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2, sharey=ax4)  
for ax in [ax2, ax3, ax5]:
    plt.setp(ax.get_yticklabels(), visible=False)
    # The x-ticks will overlap with "wspace=0", so hide the first 
    # bottom tick
    # ax.set_xticks(ax.get_xticks()[1:])  

anim = animation.FuncAnimation(plt.gcf(), make_animation, 
    Mhalo_characteristic, init_func=init, fargs=(Mstellar_characteristic, 
    Mlow_slope,Mhigh_slope,Mstellar_scatter,), interval=500, blit=False, 
    repeat=True)
plt.tight_layout()
print('Saving animation')
os.chdir(path_to_figures)
anim.save('smf_eco_tenpercent.gif',writer='imagemagick')