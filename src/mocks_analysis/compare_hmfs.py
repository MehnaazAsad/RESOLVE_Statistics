"""
This script plots different HMFs: Warren (default used in making ECO mocks), 
Tinker08 (default spherical overdensity of 200m) and Tinker08 (spherical 
overdensity of 337m which is what we want to use instead of Warren to make ECO
mvir mocks)
"""
import astropy.cosmology as astrocosmo
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import hmf

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

__author__ = '{Mehnaaz Asad}'

H0=100
# hmf_type = 'cumulative' 
# hmf_type = 'numdens'
hmf_type = 'differential'

cosmo_model = astrocosmo.Planck15.clone(H0=H0)

hmf_choice_fit_eco_default = hmf.fitting_functions.Warren
hmf_choice_fit_tinker_default = hmf.fitting_functions.Tinker08

mass_func_eco_default = hmf.mass_function.hmf.MassFunction(Mmin=10, Mmax=15, 
    cosmo_model=cosmo_model, hmf_model=hmf_choice_fit_eco_default)

mass_func_tinker_default = hmf.mass_function.hmf.MassFunction(Mmin=10, Mmax=15, 
    cosmo_model=cosmo_model, hmf_model=hmf_choice_fit_tinker_default)

mass_func_tinker_337 = hmf.mass_function.hmf.MassFunction(Mmin=10, Mmax=15, 
    cosmo_model=cosmo_model, hmf_model=hmf_choice_fit_tinker_default, 
    mdef_model='SOMean', mdef_params={"overdensity":337})

if hmf_type == 'cumulative':
    ngtm_fid = mass_func_eco_default.ngtm
    fig1 = plt.figure(figsize=(10,10))
    plt.plot(np.log10(mass_func_eco_default.m), np.log10(mass_func_eco_default.ngtm/ngtm_fid), lw=5, 
        color='cornflowerblue', label='Warren (FOF)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_default.m), np.log10(mass_func_tinker_default.ngtm/ngtm_fid), lw=5, 
        color='mediumorchid', label='Tinker (SO Mean - 200)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_337.m), np.log10(mass_func_tinker_337.ngtm/ngtm_fid), lw=5, color='teal', 
        label='Tinker (SO Mean - 337)', solid_capstyle='round')

    plt.xlabel(r"Halo Mass [$M_\odot/h$]", fontsize=30)
    # plt.ylabel(r"Ratio of hmfs to Warren", fontsize=30)

    plt.annotate('Planck15\n\n $h=1.0$\n$\Omega_{m}=0.3075$\n$\Omega_{\Lambda}'\
        '=0.69$\n$\Omega_{b}=0.0486$\n$n_{s}=0.968$', 
        xy=(0.02, 0.25), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('Ratio of HMFs to Warren (default ECO)')
    plt.legend(prop={"size":25})
    plt.show()

elif hmf_type == 'differential':
    dndlog10m_fid = mass_func_eco_default.dndlog10m

    fig1 = plt.figure(figsize=(10,10))
    plt.plot(np.log10(mass_func_eco_default.m), np.log10(mass_func_eco_default.dndlog10m/dndlog10m_fid), lw=5, 
        color='cornflowerblue', label='Warren (FOF)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_default.m), np.log10(mass_func_tinker_default.dndlog10m/dndlog10m_fid), lw=5, 
        color='mediumorchid', label='Tinker (SO Mean - 200)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_337.m), np.log10(mass_func_tinker_337.dndlog10m/dndlog10m_fid), lw=5, color='teal', 
        label='Tinker (SO Mean - 337)', solid_capstyle='round')

    plt.xlabel(r"Halo Mass [$M_\odot/h$]", fontsize=30)
    # plt.ylabel(r"Ratio of hmfs to Warren", fontsize=30)

    plt.annotate('Planck15\n\n $h=1.0$\n$\Omega_{m}=0.3075$\n$\Omega_{\Lambda}'\
        '=0.69$\n$\Omega_{b}=0.0486$\n$n_{s}=0.968$', 
        xy=(0.02, 0.25), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('Ratio of HMFs to Warren (default ECO)')
    plt.legend(prop={"size":25})
    plt.show()

elif hmf_type == 'numdens':
    dndm_fid = mass_func_eco_default.dndm

    fig1 = plt.figure(figsize=(10,10))
    plt.plot(np.log10(mass_func_eco_default.m), np.log10(mass_func_eco_default.dndm/dndm_fid), lw=5, 
        color='cornflowerblue', label='Warren (FOF)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_default.m), np.log10(mass_func_tinker_default.dndm/dndm_fid), lw=5, 
        color='mediumorchid', label='Tinker (SO Mean - 200)', solid_capstyle='round')

    plt.plot(np.log10(mass_func_tinker_337.m), np.log10(mass_func_tinker_337.dndm/dndm_fid), lw=5, color='teal', 
        label='Tinker (SO Mean - 337)', solid_capstyle='round')

    plt.xlabel(r"Halo Mass [$M_\odot/h$]", fontsize=30)
    # plt.ylabel(r"Ratio of hmfs to Warren", fontsize=30)

    plt.annotate('Planck15\n\n $h=1.0$\n$\Omega_{m}=0.3075$\n$\Omega_{\Lambda}'\
        '=0.69$\n$\Omega_{b}=0.0486$\n$n_{s}=0.968$', 
        xy=(0.02, 0.25), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        ec='k', fc='lightgray', alpha=0.5), size=25)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('Ratio of HMFs to Warren (default ECO)')
    plt.legend(prop={"size":25})
    plt.show()
