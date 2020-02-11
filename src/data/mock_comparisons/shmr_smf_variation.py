"""{Test the affect of changing SMF parameters by a certain percentage on SHMR}"""


from halotools.empirical_models import PrebuiltSubhaloModelFactory
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

def halocat_init(halo_catalog,z_median):
    """
    Initial population of halo catalog using populate_mock function

    Parameters
    ----------
    halo_catalog: string
        Path to halo catalog
    
    z_median: float
        Median redshift of survey

    Returns
    ---------
    model: halotools model instance
        Model based on behroozi 2010 SMHM
    """
    halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
    model = PrebuiltSubhaloModelFactory('behroozi10', redshift=z_median, \
        prim_haloprop_key='halo_macc')
    model.populate_mock(halocat,seed=5)

    return model

def populate_mock(theta):
    """
    Populate mock based on five parameter values 

    Parameters
    ----------
    theta: array
        Array of parameter values

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model_init.param_dict['smhm_m1_0'] = mhalo_characteristic
    model_init.param_dict['smhm_m0_0'] = mstellar_characteristic
    model_init.param_dict['smhm_beta_0'] = mlow_slope
    model_init.param_dict['smhm_delta_0'] = mhigh_slope
    model_init.param_dict['scatter_model_param1'] = mstellar_scatter

    model_init.mock.populate()

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.9) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.4) / 2.041), 1)
    elif survey == 'resolveb':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.7) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.1) / 2.041), 1)
    sample_mask = model_init.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model_init.mock.galaxy_table[sample_mask]
    gals_df = gals.to_pandas()

    return gals_df

def read_data_catl(path_to_file, survey):
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
        # 13878 galaxies
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0)

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
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[
                    (resolve_live18.grpcz.values >= 4500) & 
                    (resolve_live18.grpcz.values <= 7000) & 
                    (resolve_live18.absrmag.values <= -17.33) & 
                    (resolve_live18.logmstar.values >= 8.9)]
                # catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) & 
                #     (resolve_live18.grpcz.values >= 4500) & 
                #     (resolve_live18.grpcz.values <= 7000) & 
                #     (resolve_live18.absrmag.values <= -17.33) & 
                #     (resolve_live18.logmstar.values >= 8.9)]
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

def get_centrals_mock(gals_df):
    """
    Get centrals from mock catalog

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    cen_gals: array
        Array of central galaxy masses

    cen_halos: array
        Array of central halo masses
    """
    C_S = []
    for idx in range(len(gals_df)):
        if gals_df['halo_hostid'][idx] == gals_df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)
    
    C_S = np.array(C_S)
    gals_df['C_S'] = C_S
    cen_gals = []
    cen_halos = []

    for idx,value in enumerate(gals_df['C_S']):
        if value == 1:
            cen_gals.append(gals_df['stellar_mass'][idx])
            cen_halos.append(gals_df['halo_mvir'][idx])

    cen_gals = np.log10(np.array(cen_gals))
    cen_halos = np.log10(np.array(cen_halos))

    return cen_gals, cen_halos

def diff_smf(mstar_arr, volume, h1_bool):
    """
    Calculates differential stellar mass function

    Parameters
    ----------
    mstar_arr: numpy array
        Array of stellar masses

    volume: float
        Volume of survey or simulation

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

global survey
global mf_type
survey = 'eco'
mf_type = 'smf'

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']

halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
catl_file = path_to_raw + "eco_all.csv"

eco_bf_params = [12.32381675, 10.56581819, 0.4276319, 0.7457711, 0.34784431]
v_sim = 130**3
## Experiment 10% above and below best fit SHMR to see if SMF changes by the 
## same amount
Mhalo_characteristic = np.array([eco_bf_params[0],1.05*eco_bf_params[0],0.95*eco_bf_params[0]])
Mstellar_characteristic = np.array([eco_bf_params[1],1.05*eco_bf_params[1],0.95*eco_bf_params[1]])
Mlow_slope = np.array([eco_bf_params[2],1.05*eco_bf_params[2],0.95*eco_bf_params[2]])
Mhigh_slope = np.array([eco_bf_params[3],1.05*eco_bf_params[3],0.95*eco_bf_params[3]])
Mstellar_scatter = np.array([eco_bf_params[4],1.05*eco_bf_params[4],0.95*eco_bf_params[4]])

catl, volume, cvar, z_median = read_data_catl(catl_file, survey)
model_init = halocat_init(halo_catalog, z_median)

max_arr = []
phi_arr = []
cen_gals_arr = []
cen_halos_arr = []
for i in range(len(Mhalo_characteristic)):
    eco_params = np.array([Mhalo_characteristic[i], Mstellar_characteristic[i], 
        Mlow_slope[i], Mhigh_slope[i], Mstellar_scatter[i]])
    gals_df = populate_mock(eco_params)
    cen_gals, cen_halos = get_centrals_mock(gals_df)
    mstellar_mock = gals_df.stellar_mass.values
    max_model, phi_model, err_tot_model, bins_model, counts_model =\
        diff_smf(mstellar_mock, v_sim, True)
    max_arr.append(max_model)
    phi_arr.append(phi_model)
    cen_gals_arr.append(cen_gals)
    cen_halos_arr.append(cen_halos)

fig1 = plt.figure(figsize=(10,8))
for i in range(len(Mhalo_characteristic)):
    colour_arr = ['#53A48D', 'mediumorchid', 'cornflowerblue']
    label_arr = [r'$best\ fit$', r'$+5\% \Theta$', r'$-5\% \Theta$']  
    plt.errorbar(max_arr[i],phi_arr[i],
        color=colour_arr[i],fmt='-s',ecolor=colour_arr[i],markersize=10,
        capsize=5,capthick=0.5,label=label_arr[i],linewidth=5)
    plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
    plt.legend(loc='best', prop={'size': 20})

fig2 = plt.figure(figsize=(10,8))
y_arr = []
x_arr = []
for i in range(len(Mhalo_characteristic)):
    x,y,y_std,y_std_err = Stats_one_arr(cen_halos_arr[i],
        cen_gals_arr[i],base=0.4,bin_statval='center')
    y_arr.append(y)
    x_arr.append(x)
    colour_arr = ['#53A48D', 'mediumorchid', 'cornflowerblue']
    label_arr = [r'$best\ fit$', r'$+5\% \Theta$', r'$-5\% \Theta$']  
    plt.errorbar(x,y,color=colour_arr[i],fmt='-s',ecolor=colour_arr[i],\
        markersize=10,capsize=5,capthick=0.5,label=label_arr[i],linewidth=5)
    plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    plt.legend(loc='best', prop={'size': 20})

## SMF percentage difference
p_percent_diff = ((phi_arr[1] - phi_arr[0])/np.abs(phi_arr[0]))*100
n_percent_diff = ((phi_arr[2] - phi_arr[0])/np.abs(phi_arr[0]))*100
fig3 = plt.figure(figsize=(10,8))
plt.plot(max_arr[0], p_percent_diff, c='mediumorchid', label=r'$+5\% \Theta$', lw=5)
plt.plot(max_arr[0], np.zeros(len(p_percent_diff)), c='#53A48D', label=r'$best\ fit$', lw=5)
plt.plot(max_arr[0], n_percent_diff, c='cornflowerblue', label=r'$-5\% \Theta$', lw=5)
plt.legend(loc='best', prop={'size': 20})
plt.ylabel('\% change in SMF')

## SHMR perecentage difference
p_percent_diff = ((y_arr[1] - y_arr[0])/y_arr[0])*100
n_percent_diff = ((y_arr[2][2:] - y_arr[0])/y_arr[0])*100
fig4 = plt.figure(figsize=(10,8))
plt.plot(x_arr[0], p_percent_diff, c='mediumorchid', label=r'$+5\% \Theta$', lw=5)
plt.plot(x_arr[0], np.zeros(len(p_percent_diff)), c='#53A48D', label=r'$best\ fit$', lw=5)
plt.plot(x_arr[0], n_percent_diff, c='cornflowerblue', label=r'$-5\% \Theta$', lw=5)
plt.legend(loc='best', prop={'size': 20})
plt.ylabel('\% change in SHMR')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for i in range(len(Mhalo_characteristic)):
    colour_arr = ['#53A48D', 'mediumorchid', 'cornflowerblue']
    label_arr = [r'$best\ fit$', r'$+5\% \Theta$', r'$-5\% \Theta$']  
    ax1.errorbar(max_arr[i],phi_arr[i],
        color=colour_arr[i],fmt='-s',ecolor=colour_arr[i],markersize=10,
        capsize=5,capthick=0.5,label=label_arr[i],linewidth=5)
    ax1.set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
    ax1.set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=20)
    ax1.legend(loc='best', prop={'size': 20})
    ax1.set_title(r'$\textbf{SMF}$')

y_arr = []
x_arr = []
for i in range(len(Mhalo_characteristic)):
    x,y,y_std,y_std_err = Stats_one_arr(cen_halos_arr[i],
        cen_gals_arr[i],base=0.4,bin_statval='center')
    y_arr.append(y)
    x_arr.append(x)
    colour_arr = ['#53A48D', 'mediumorchid', 'cornflowerblue']
    label_arr = [r'$best\ fit$', r'$+5\% \Theta$', r'$-5\% \Theta$']  
    ax2.errorbar(x,y,color=colour_arr[i],fmt='-s',ecolor=colour_arr[i],\
        markersize=10,capsize=5,capthick=0.5,label=label_arr[i],linewidth=5)
    ax2.set_xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    ax2.set_ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
    ax2.set_title(r'$\textbf{SHMR}$')

p_percent_diff = ((phi_arr[1] - phi_arr[0])/np.abs(phi_arr[0]))*100
n_percent_diff = ((phi_arr[2] - phi_arr[0])/np.abs(phi_arr[0]))*100
ax3.plot(max_arr[0], p_percent_diff, c='mediumorchid', label=r'$+5\% \Theta$', lw=5)
ax3.plot(max_arr[0], np.zeros(len(p_percent_diff)), c='#53A48D', label=r'$best\ fit$', lw=5)
ax3.plot(max_arr[0], n_percent_diff, c='cornflowerblue', label=r'$-5\% \Theta$', lw=5)
ax3.set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=20)
ax3.set_ylabel('\% change in SMF')

## SHMR perecentage difference
p_percent_diff = ((y_arr[1] - y_arr[0])/y_arr[0])*100
n_percent_diff = ((y_arr[2][2:] - y_arr[0])/y_arr[0])*100
ax4.plot(x_arr[0], p_percent_diff, c='mediumorchid', label=r'$+5\% \Theta$', lw=5)
ax4.plot(x_arr[0], np.zeros(len(p_percent_diff)), c='#53A48D', label=r'$best\ fit$', lw=5)
ax4.plot(x_arr[0], n_percent_diff, c='cornflowerblue', label=r'$-5\% \Theta$', lw=5)
ax4.set_xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=20)
ax4.set_ylabel('\% change in SHMR')
