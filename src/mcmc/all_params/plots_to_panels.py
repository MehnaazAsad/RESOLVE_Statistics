import numpy as np
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binned_statistic as bs
from collections import OrderedDict

from scipy.stats import chi2
import pandas as pd

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

mf_type = 'smf'

hybrid_smf_total = [np.array([ 8.925,  9.575, 10.225, 10.875]),
 np.array([-1.52829146, -1.67184278, -1.89046511, -2.60333372]),
 np.array([0.00054792, 0.00046446, 0.0003611 , 0.00015893]),
 np.array([2924, 2101, 1270,  246])]

hybrid_error = np.array([0.12096691, 0.14461279, 0.16109724, 0.18811391])

hybrid_x_phi_total_data, hybrid_y_phi_total_data = hybrid_smf_total[0], hybrid_smf_total[1]

hybrid_y_phi_total_bf = np.array([-1.44271943, -1.58037288, -1.7888421 , -2.5240684 ])

dof = 11

hybrid_tot_phi_max = np.array([-1.31766636, -1.39692244, -1.60325968, -2.27083452])
hybrid_tot_phi_min = np.array([-1.55687075, -1.69905777, -1.91178633, -2.69853286])


halo_smf_total = [np.array([ 8.925,  9.575, 10.225, 10.875]),
 np.array([-1.52829146, -1.67184278, -1.89046511, -2.60333372]),
 np.array([0.00054792, 0.00046446, 0.0003611 , 0.00015893]),
 np.array([2924, 2101, 1270,  246])]

halo_error = np.array([0.12096691, 0.14461279, 0.16109724, 0.18811391])

halo_x_phi_total_data, halo_y_phi_total_data = halo_smf_total[0], halo_smf_total[1]

halo_y_phi_total_bf = np.array([-1.4769015 , -1.59156301, -1.80582377, -2.54434344])

halo_tot_phi_max = np.array([-1.20657928, -1.31496221, -1.493781  , -2.22341469])
halo_tot_phi_min = np.array([-1.71150238, -1.88306169, -2.12844142, -2.89751232])


fig, ax = plt.subplots(1, 2, figsize=(24,13.5), sharey=True, gridspec_kw={'wspace':0})
mt = ax[0].fill_between(x=hybrid_x_phi_total_data, y1=hybrid_tot_phi_max, 
    y2=hybrid_tot_phi_min, color='silver', alpha=0.4)

dt = ax[0].errorbar(hybrid_x_phi_total_data, hybrid_y_phi_total_data, 
    yerr=hybrid_error, color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bft, = ax[0].plot(hybrid_x_phi_total_data, hybrid_y_phi_total_bf, color='k', 
    ls='--', lw=4, zorder=10)


mt = ax[1].fill_between(x=halo_x_phi_total_data, y1=halo_tot_phi_max, 
    y2=halo_tot_phi_min, color='silver', alpha=0.4)

dt = ax[1].errorbar(halo_x_phi_total_data, halo_y_phi_total_data, 
    yerr=halo_error, color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
    capthick=1.5, zorder=10, marker='^')
# Best-fit
# Need a comma after 'bfr' and 'bfb' to solve this:
#   AttributeError: 'NoneType' object has no attribute 'create_artists'
bft, = ax[1].plot(halo_x_phi_total_data, halo_y_phi_total_bf, color='k', 
    ls='--', lw=4, zorder=10)

halo_bf_chi2 = 14.959239060578579
hybrid_bf_chi2 = 11.145115359202434

ax[0].set_ylim(-3,-1)
ax[1].set_ylim(-3,-1)

if mf_type == 'smf':
    ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
elif mf_type == 'bmf':
    ax[0].set_xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
    ax[1].set_xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)

ax[0].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)
# ax[1].set_ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

ax[0].annotate(r'$\boldsymbol\chi ^2 / dof \approx {0}/{1}$'.format(np.round(hybrid_bf_chi2,2),dof) \
    + '\n' + \
    r'$p \approx$ {0}'.format(np.round((1 - chi2.cdf(hybrid_bf_chi2, dof)),2)), 
    xy=(0.03, 0.23), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    ec='k', fc='lightgray', alpha=0.5), size=25)

ax[0].annotate('Hybrid quenching', xy=(0.63, 0.95), xycoords='axes fraction', 
    color='rebeccapurple', size=30)

ax[1].annotate(r'$\boldsymbol\chi ^2 / dof \approx {0}/{1}$'.format(np.round(halo_bf_chi2,2),dof) \
    + '\n' + \
    r'$p \approx$ {0}'.format(np.round((1 - chi2.cdf(halo_bf_chi2, dof)),2)), 
    xy=(0.02, 0.03), xycoords='axes fraction', bbox=dict(boxstyle="square", 
    ec='k', fc='lightgray', alpha=0.5), size=25)

ax[1].annotate('Halo quenching', xy=(0.67, 0.95), xycoords='axes fraction', 
    color='rebeccapurple', size=30)

ax[0].legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
    handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='lower left')


# figM = plt.get_current_fig_manager()
# figM.resize(*figM.window.maxsize())
plt.savefig('/Users/asadm2/Documents/Grad_School/Research/Papers/RESOLVE_Statistics_paper/Figures/smf_total_panel.pdf', 
    bbox_inches="tight", dpi=1200)
# plt.show()
