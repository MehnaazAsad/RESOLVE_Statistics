"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binned_statistic as bs
from collections import OrderedDict

from matplotlib.legend_handler import HandlerTuple
from scipy.stats import chi2
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


class Plotting():

    def __init__(self, preprocess) -> None:
        self.preprocess = preprocess
        self.settings = preprocess.settings

    def plot_total_mf(self, models, data, best_fit):
        """
        Plot SMF from data, best fit param values and param values corresponding to 
        68th percentile 100 lowest chi^2 values

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        total_data: multidimensional array
            Array of total SMF information

        maxis_bf_total: array
            Array of x-axis mass values for best-fit SMF

        phi_bf_total: array
            Array of y-axis values for best-fit SMF

        bf_chi2: float
            Chi-squared value associated with best-fit model

        err_colour: array
            Array of error values from matrix

        Returns
        ---------
        Plot displayed on screen.
        """
        settings = self.settings
        preprocess = self.preprocess

        smf_total = data[0] #x, y, error, counts
        error = data[4][0:6]

        x_phi_total_data, y_phi_total_data = smf_total[0], smf_total[1]
        x_phi_total_model = models[0][0]['smf_total']['max_total'][0]

        x_phi_total_bf, y_phi_total_bf = best_fit[0]['smf_total']['max_total'],\
            best_fit[0]['smf_total']['phi_total']

        dof = data[6]

        i_outer = 0
        mod_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][0]['smf_total']['max_total'])):
                tot_mod_ii = models[i_outer][0]['smf_total']['phi_total'][idx]
                mod_arr.append(tot_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['smf_total']['max_total'])):
                tot_mod_ii = models[i_outer][0]['smf_total']['phi_total'][idx]
                mod_arr.append(tot_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['smf_total']['max_total'])):
                tot_mod_ii = models[i_outer][0]['smf_total']['phi_total'][idx]
                mod_arr.append(tot_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['smf_total']['max_total'])):
                tot_mod_ii = models[i_outer][0]['smf_total']['phi_total'][idx]
                mod_arr.append(tot_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['smf_total']['max_total'])):
                tot_mod_ii = models[i_outer][0]['smf_total']['phi_total'][idx]
                mod_arr.append(tot_mod_ii)
            i_outer += 1

        tot_phi_max = np.amax(mod_arr, axis=0)
        tot_phi_min = np.amin(mod_arr, axis=0)
        
        fig1= plt.figure(figsize=(10,10))
        mt = plt.fill_between(x=x_phi_total_model, y1=tot_phi_max, 
            y2=tot_phi_min, color='silver', alpha=0.4)

        dt = plt.errorbar(x_phi_total_data, y_phi_total_data, yerr=error,
            color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')
        # Best-fit
        # Need a comma after 'bfr' and 'bfb' to solve this:
        #   AttributeError: 'NoneType' object has no attribute 'create_artists'
        bft, = plt.plot(x_phi_total_bf, y_phi_total_bf, color='k', ls='--', lw=4, 
            zorder=10)

        
        plt.ylim(-4,-1)
        if settings.mf_type == 'smf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        elif settings.mf_type == 'bmf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.87, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        plt.annotate(r'$ p \approx$ {0}'.
            format(np.round((1 - chi2.cdf(preprocess.bf_chi2, dof)),2)), 
            xy=(0.87, 0.69), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)


        plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='best')

        if settings.survey == 'eco':
            if settings.quenching == 'hybrid':
                plt.title('Hybrid quenching model | ECO')
            elif settings.quenching == 'halo':
                plt.title('Halo quenching model | ECO')
        plt.show()

    def plot_colour_mf(self, models, data, best_fit):
        """
        Plot red and blue SMF from data, best fit param values and param values 
        corresponding to 68th percentile 100 lowest chi^2 values

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information

        phi_red_data: array
            Array of y-axis values for red SMF from data

        phi_blue_data: array
            Array of y-axis values for blue SMF from data
        
        phi_bf_red: array
            Array of y-axis values for red SMF from best-fit model

        phi_bf_blue: array
            Array of y-axis values for blue SMF from best-fit model

        std_red: array
            Array of std values per bin of red SMF from mocks

        std_blue: array
            Array of std values per bin of blue SMF from mocks

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """
        settings = self.settings
        preprocess = self.preprocess

        # x axis values same as those used in blue fraction for both red and 
        # blue since colour MFs were reconstructed from blue fraction
        fblue_data = data[1]

        error_red = data[5]['std_phi_colour']['std_phi_red']
        error_blue = data[5]['std_phi_colour']['std_phi_blue']

        x_phi_red_model = fblue_data[0]
        x_phi_red_data, y_phi_red_data = x_phi_red_model, data[2]

        x_phi_blue_model = fblue_data[0]
        x_phi_blue_data, y_phi_blue_data = x_phi_blue_model, data[3]

        x_phi_red_bf, y_phi_red_bf = fblue_data[0],\
            best_fit[0]['phi_colour']['phi_red']

        x_phi_blue_bf, y_phi_blue_bf = fblue_data[0],\
            best_fit[0]['phi_colour']['phi_blue']

        dof = data[6]

        i_outer = 0
        red_mod_arr = []
        blue_mod_arr = []
        while i_outer < 5:
            # Length of phi_red is the same as phi_blue
            for idx in range(len(models[i_outer][0]['phi_colour']['phi_red'])):
                red_mod_ii = models[i_outer][0]['phi_colour']['phi_red'][idx]
                red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][0]['phi_colour']['phi_blue'][idx]
                blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['phi_colour']['phi_red'])):
                red_mod_ii = models[i_outer][0]['phi_colour']['phi_red'][idx]
                red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][0]['phi_colour']['phi_blue'][idx]
                blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['phi_colour']['phi_red'])):
                red_mod_ii = models[i_outer][0]['phi_colour']['phi_red'][idx]
                red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][0]['phi_colour']['phi_blue'][idx]
                blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['phi_colour']['phi_red'])):
                red_mod_ii = models[i_outer][0]['phi_colour']['phi_red'][idx]
                red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][0]['phi_colour']['phi_blue'][idx]
                blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['phi_colour']['phi_red'])):
                red_mod_ii = models[i_outer][0]['phi_colour']['phi_red'][idx]
                red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][0]['phi_colour']['phi_blue'][idx]
                blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

        red_phi_max = np.amax(red_mod_arr, axis=0)
        red_phi_min = np.amin(red_mod_arr, axis=0)
        blue_phi_max = np.amax(blue_mod_arr, axis=0)
        blue_phi_min = np.amin(blue_mod_arr, axis=0)
        
        fig1= plt.figure(figsize=(10,10))
        mr = plt.fill_between(x=x_phi_red_model, y1=red_phi_max, 
            y2=red_phi_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=x_phi_blue_model, y1=blue_phi_max, 
            y2=blue_phi_min, color='cornflowerblue',alpha=0.4)

        dr = plt.errorbar(x_phi_red_data, y_phi_red_data, yerr=error_red,
            color='darkred', fmt='s', ecolor='darkred',markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')
        db = plt.errorbar(x_phi_blue_data, y_phi_blue_data, yerr=error_blue,
            color='darkblue', fmt='s', ecolor='darkblue',markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')
        # Best-fit
        # Need a comma after 'bfr' and 'bfb' to solve this:
        #   AttributeError: 'NoneType' object has no attribute 'create_artists'
        bfr, = plt.plot(x_phi_red_bf, y_phi_red_bf,
            color='maroon', ls='--', lw=4, zorder=10)
        bfb, = plt.plot(x_phi_blue_bf, y_phi_blue_bf,
            color='mediumblue', ls='--', lw=4, zorder=10)

        plt.ylim(-4,-1)
        if settings.mf_type == 'smf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        elif settings.mf_type == 'bmf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

        # plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data','Models','Best-fit'],
        #     handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})
        plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data', 'Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})


        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.87, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.survey == 'eco':
            if settings.quenching == 'hybrid':
                plt.title('Hybrid quenching model | ECO')
            elif settings.quenching == 'halo':
                plt.title('Halo quenching model | ECO')

        # if survey == 'eco':
        #     plt.title('ECO')
        plt.show()

    def plot_fblue(self, models, data, best_fit):
        """
        Plot blue fraction from data, best fit param values and param values 
        corresponding to 68th percentile 100 lowest chi^2 values

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        fblue_data: array
            Array of y-axis blue fraction values for data

        maxis_bf_fblue: array
            Array of x-axis mass values for best-fit model

        bf_fblue: array
            Array of y-axis blue fraction values for best-fit model

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        err_colour: array
            Array of error values from matrix

        Returns
        ---------
        Plot displayed on screen.
        """
        settings = self.settings
        preprocess = self.preprocess

        fblue_data = data[1]
        error_total = data[5]['std_fblue']['std_fblue']
        # error_cen = data[5]['std_fblue']['std_fblue_cen']
        # error_sat = data[5]['std_fblue']['std_fblue_sat']
        error_cen = data[4][6:12]
        error_sat = data[4][12:]
        dof = data[6]

        x_fblue_total_data, y_fblue_total_data = fblue_data[0], fblue_data[1]
        y_fblue_cen_data = fblue_data[2]
        y_fblue_sat_data = fblue_data[3]

        x_fblue_model = models[0][0]['f_blue']['max_fblue'][0]

        x_fblue_total_bf, y_fblue_total_bf = best_fit[0]['f_blue']['max_fblue'],\
            best_fit[0]['f_blue']['fblue_total']
        y_fblue_cen_bf = best_fit[0]['f_blue']['fblue_cen']
        y_fblue_sat_bf = best_fit[0]['f_blue']['fblue_sat']

        i_outer = 0
        total_mod_arr = []
        cen_mod_arr = []
        sat_mod_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][0]['f_blue']['max_fblue'])):
                tot_mod_ii = models[i_outer][0]['f_blue']['fblue_total'][idx]
                cen_mod_ii = models[i_outer][0]['f_blue']['fblue_cen'][idx]
                sat_mod_ii = models[i_outer][0]['f_blue']['fblue_sat'][idx]
                total_mod_arr.append(tot_mod_ii)
                cen_mod_arr.append(cen_mod_ii)
                sat_mod_arr.append(sat_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['f_blue']['max_fblue'])):
                tot_mod_ii = models[i_outer][0]['f_blue']['fblue_total'][idx]
                cen_mod_ii = models[i_outer][0]['f_blue']['fblue_cen'][idx]
                sat_mod_ii = models[i_outer][0]['f_blue']['fblue_sat'][idx]
                total_mod_arr.append(tot_mod_ii)
                cen_mod_arr.append(cen_mod_ii)
                sat_mod_arr.append(sat_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['f_blue']['max_fblue'])):
                tot_mod_ii = models[i_outer][0]['f_blue']['fblue_total'][idx]
                cen_mod_ii = models[i_outer][0]['f_blue']['fblue_cen'][idx]
                sat_mod_ii = models[i_outer][0]['f_blue']['fblue_sat'][idx]
                total_mod_arr.append(tot_mod_ii)
                cen_mod_arr.append(cen_mod_ii)
                sat_mod_arr.append(sat_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['f_blue']['max_fblue'])):
                tot_mod_ii = models[i_outer][0]['f_blue']['fblue_total'][idx]
                cen_mod_ii = models[i_outer][0]['f_blue']['fblue_cen'][idx]
                sat_mod_ii = models[i_outer][0]['f_blue']['fblue_sat'][idx]
                total_mod_arr.append(tot_mod_ii)
                cen_mod_arr.append(cen_mod_ii)
                sat_mod_arr.append(sat_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['f_blue']['max_fblue'])):
                tot_mod_ii = models[i_outer][0]['f_blue']['fblue_total'][idx]
                cen_mod_ii = models[i_outer][0]['f_blue']['fblue_cen'][idx]
                sat_mod_ii = models[i_outer][0]['f_blue']['fblue_sat'][idx]
                total_mod_arr.append(tot_mod_ii)
                cen_mod_arr.append(cen_mod_ii)
                sat_mod_arr.append(sat_mod_ii)
            i_outer += 1

        fblue_total_max = np.amax(total_mod_arr, axis=0)
        fblue_total_min = np.amin(total_mod_arr, axis=0)
        fblue_cen_max = np.nanmax(cen_mod_arr, axis=0)
        fblue_cen_min = np.nanmin(cen_mod_arr, axis=0)
        fblue_sat_max = np.nanmax(sat_mod_arr, axis=0)
        fblue_sat_min = np.nanmin(sat_mod_arr, axis=0)

        
        fig1= plt.figure(figsize=(10,10))
        mt = plt.fill_between(x=x_fblue_model, y1=fblue_total_max, 
            y2=fblue_total_min, color='silver', alpha=0.4)

        dt = plt.errorbar(x_fblue_total_data, y_fblue_total_data, yerr=error_total,
            color='k', fmt='s', ecolor='k', markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')
        # dt = plt.scatter(x_fblue_total_data, y_fblue_total_data,
        #     color='k', s=120, zorder=10, marker='^')
        # Best-fit
        # Need a comma after 'bfr' and 'bfb' to solve this:
        #   AttributeError: 'NoneType' object has no attribute 'create_artists'
        bft, = plt.plot(x_fblue_total_bf, y_fblue_total_bf, color='k', ls='--', lw=4, 
            zorder=10)

        if settings.mf_type == 'smf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        elif settings.mf_type == 'bmf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$f_{blue}$', fontsize=30)

        plt.legend([(dt), (mt), (bft)], ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.87, 0.75), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)
        plt.annotate(r'$ p \approx$ {0}'.
            format(np.round((1 - chi2.cdf(preprocess.bf_chi2, dof)),2)), 
            xy=(0.87, 0.69), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.survey == 'eco':
            if settings.quenching == 'hybrid':
                plt.title('Hybrid quenching model | ECO')
            elif settings.quenching == 'halo':
                plt.title('Halo quenching model | ECO')

        plt.show()

        fig2= plt.figure(figsize=(10,10))
        mc = plt.fill_between(x=x_fblue_model, y1=fblue_cen_max, 
            y2=fblue_cen_min, color='thistle', alpha=0.5)
        ms = plt.fill_between(x=x_fblue_model, y1=fblue_sat_max, 
            y2=fblue_sat_min, color='khaki', alpha=0.5)


        dc = plt.errorbar(x_fblue_total_data, y_fblue_cen_data, yerr=error_cen,
            color='rebeccapurple', fmt='s', ecolor='rebeccapurple', markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')
        ds = plt.errorbar(x_fblue_total_data, y_fblue_sat_data, yerr=error_sat,
            color='goldenrod', fmt='s', ecolor='goldenrod', markersize=12, capsize=7,
            capthick=1.5, zorder=10, marker='^')

        # Best-fit
        # Need a comma after 'bfr' and 'bfb' to solve this:
        #   AttributeError: 'NoneType' object has no attribute 'create_artists'
        bfc, = plt.plot(x_fblue_total_bf, y_fblue_cen_bf, color='rebeccapurple', ls='--', lw=4, 
            zorder=10)
        bfs, = plt.plot(x_fblue_total_bf, y_fblue_sat_bf, color='goldenrod', ls='--', lw=4, 
            zorder=10)


        if settings.mf_type == 'smf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        elif settings.mf_type == 'bmf':
            plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$f_{blue}$', fontsize=30)

        plt.legend([(dc), (mc), (bfc), (ds), (ms), (bfs)], 
            ['Data - cen','Models - cen','Best-fit - cen',
            'Data - sat','Models - sat','Best-fit - sat'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)})

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)),
            xy=(0.87, 0.55), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)
        plt.annotate(r'$ p \approx$ {0}'.
            format(np.round((1 - chi2.cdf(preprocess.bf_chi2, dof)),2)), 
            xy=(0.87, 0.49), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.survey == 'eco':
            if settings.quenching == 'hybrid':
                plt.title('Hybrid quenching model | ECO')
            elif settings.quenching == 'halo':
                plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_xmhm(self, models, data, best_fit):
        """
        Plot SMHM from data, best fit param values, param values corresponding to 
        68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        gals_bf: array
            Array of y-axis stellar mass values for best fit SMHM

        halos_bf: array
            Array of x-axis halo mass values for best fit SMHM
        
        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """
        settings = self.settings
        preprocess = self.preprocess

        if settings.survey == 'resolvea':
            line_label = 'RESOLVE-A'
        elif settings.survey == 'resolveb':
            line_label = 'RESOLVE-B'
        elif settings.survey == 'eco':
            line_label = 'ECO'

        dof = data[6]

        halos_bf = best_fit[0]['centrals']['halos']
        gals_bf = best_fit[0]['centrals']['gals']

        y_bf,x_bf,binnum = bs(halos_bf,
            gals_bf,'mean',bins=np.linspace(10, 15, 15))


        i_outer = 0
        mod_x_arr = []
        mod_y_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][0]['centrals']['halos'])):
                mod_x_ii = models[i_outer][0]['centrals']['halos'][idx]
                mod_y_ii = models[i_outer][0]['centrals']['gals'][idx]
                y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                mod_x_arr.append(x)
                mod_y_arr.append(y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos'])):
                mod_x_ii = models[i_outer][0]['centrals']['halos'][idx]
                mod_y_ii = models[i_outer][0]['centrals']['gals'][idx]
                y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                mod_x_arr.append(x)
                mod_y_arr.append(y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos'])):
                mod_x_ii = models[i_outer][0]['centrals']['halos'][idx]
                mod_y_ii = models[i_outer][0]['centrals']['gals'][idx]
                y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                mod_x_arr.append(x)
                mod_y_arr.append(y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos'])):
                mod_x_ii = models[i_outer][0]['centrals']['halos'][idx]
                mod_y_ii = models[i_outer][0]['centrals']['gals'][idx]
                y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                mod_x_arr.append(x)
                mod_y_arr.append(y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos'])):
                mod_x_ii = models[i_outer][0]['centrals']['halos'][idx]
                mod_y_ii = models[i_outer][0]['centrals']['gals'][idx]
                y,x,binnum = bs(mod_x_ii,mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                mod_x_arr.append(x)
                mod_y_arr.append(y)
            i_outer += 1

        y_max = np.nanmax(mod_y_arr, axis=0)
        y_min = np.nanmin(mod_y_arr, axis=0)

        fig1 = plt.figure(figsize=(10,10))

        x_cen =  0.5 * (mod_x_arr[0][1:] + mod_x_arr[0][:-1])

        plt.fill_between(x=x_cen, y1=y_max, 
            y2=y_min, color='lightgray',alpha=0.4,label='Models')

        x_cen =  0.5 * (x_bf[1:] + x_bf[:-1])

        plt.plot(x_cen, y_bf, color='k', lw=4, label='Best-fit', zorder=10)

        plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
            [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
            plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
            hatch='\\')

        if settings.survey == 'resolvea' and settings.mf_type == 'smf':
            plt.xlim(10,14)
        else:
            plt.xlim(10,14.5)
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        if settings.mf_type == 'smf':
            if settings.survey == 'eco' and settings.quenching == 'hybrid':
                plt.ylim(np.log10((10**8.9)/2.041),11.9)
            elif settings.survey == 'eco' and settings.quenching == 'halo':
                plt.ylim(np.log10((10**8.9)/2.041),11.56)
            elif settings.survey == 'resolvea':
                plt.ylim(np.log10((10**8.9)/2.041),13)
            elif settings.survey == 'resolveb':
                plt.ylim(np.log10((10**8.7)/2.041),)
            plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        elif settings.mf_type == 'bmf':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                plt.ylim(np.log10((10**9.4)/2.041),)
            elif settings.survey == 'resolveb':
                plt.ylim(np.log10((10**9.1)/2.041),)
            plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best',prop={'size': 30})
        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_colour_xmhm(self, models, data, best_fit):
        """
        Plot red and blue SMHM from data, best fit param values, param values 
        corresponding to 68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        gals_bf_red: array
            Array of y-axis stellar mass values for red SMHM for best-fit model

        halos_bf_red: array
            Array of x-axis halo mass values for red SMHM for best-fit model
        
        gals_bf_blue: array
            Array of y-axis stellar mass values for blue SMHM for best-fit model

        halos_bf_blue: array
            Array of x-axis halo mass values for blue SMHM for best-fit model

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings
        preprocess = self.preprocess

        dof = data[6]

        halos_bf_red = best_fit[0]['centrals']['halos_red']
        gals_bf_red = best_fit[0]['centrals']['gals_red']

        halos_bf_blue = best_fit[0]['centrals']['halos_blue']
        gals_bf_blue = best_fit[0]['centrals']['gals_blue']

        y_bf_red,x_bf_red,binnum_red = bs(halos_bf_red,\
        gals_bf_red,'mean',bins=np.linspace(10, 15, 15))
        y_bf_blue,x_bf_blue,binnum_blue = bs(halos_bf_blue,\
        gals_bf_blue,'mean',bins=np.linspace(10, 15, 15))

        i_outer = 0
        red_mod_x_arr = []
        red_mod_y_arr = []
        blue_mod_x_arr = []
        blue_mod_y_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_x_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_y_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_x_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_y_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_x_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_y_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_x_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_y_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_x_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_y_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_x_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_y_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_x_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_y_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_x_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_y_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_x_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_y_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_x_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_y_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(10, 15, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

        red_y_max = np.nanmax(red_mod_y_arr, axis=0)
        red_y_min = np.nanmin(red_mod_y_arr, axis=0)
        blue_y_max = np.nanmax(blue_mod_y_arr, axis=0)
        blue_y_min = np.nanmin(blue_mod_y_arr, axis=0)

        fig1 = plt.figure(figsize=(10,10))

        red_x_cen =  0.5 * (red_mod_x_arr[0][1:] + red_mod_x_arr[0][:-1])
        blue_x_cen = 0.5 * (blue_mod_x_arr[0][1:] + blue_mod_x_arr[0][:-1])

        mr = plt.fill_between(x=red_x_cen, y1=red_y_max, 
            y2=red_y_min, color='lightcoral',alpha=0.4,label='Models')
        mb = plt.fill_between(x=blue_x_cen, y1=blue_y_max, 
            y2=blue_y_min, color='cornflowerblue',alpha=0.4,label='Models')

        red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
        blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

        # REMOVED ERROR BAR ON BEST FIT
        bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=4,label='Best-fit',zorder=10)
        bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=4,
            label='Best-fit',zorder=10)

        plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
            [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
            plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
            hatch='\\')

        if settings.survey == 'resolvea' and settings.mf_type == 'smf':
            plt.xlim(10,14)
        else:
            plt.xlim(10,14.5)
        plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        if settings.mf_type == 'smf':
            if settings.survey == 'eco' and settings.quenching == 'hybrid':
                plt.ylim(np.log10((10**8.9)/2.041),11.9)
                # plt.title('ECO')
            elif settings.survey == 'eco' and settings.quenching == 'halo':
                plt.ylim(np.log10((10**8.9)/2.041),11.56)
            elif settings.survey == 'resolvea':
                plt.ylim(np.log10((10**8.9)/2.041),13)
            elif settings.survey == 'resolveb':
                plt.ylim(np.log10((10**8.7)/2.041),)
            plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        elif settings.mf_type == 'bmf':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                plt.ylim(np.log10((10**9.4)/2.041),)
                plt.title('ECO')
            elif settings.survey == 'resolveb':
                plt.ylim(np.log10((10**9.1)/2.041),)
            plt.ylabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.legend([(mr, mb), (bfr, bfb)], ['Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, 
            loc='best',prop={'size': 30})

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_colour_hmxm(self, models, data, best_fit):
        """
        Plot SMHM from data, best fit param values, param values corresponding to 
        68th percentile 1000 lowest chi^2 values and behroozi 2010 param values

        Parameters
        ----------
        result: multidimensional array
            Array of central galaxy and halo masses
        
        gals_bf: array
            Array of y-axis stellar mass values for best fit SMHM

        halos_bf: array
            Array of x-axis halo mass values for best fit SMHM
        
        gals_data: array
            Array of y-axis stellar mass values for data SMF

        halos_data: array
            Array of x-axis halo mass values for data SMF

        gals_b10: array
            Array of y-axis stellar mass values for behroozi 2010 SMHM

        halos_b10: array
            Array of x-axis halo mass values for behroozi 2010 SMHM

        Returns
        ---------
        Nothing; SMHM plot is saved in figures repository
        """   
        settings = self.settings
        preprocess = self.preprocess

        dof = data[6]

        halos_bf_red = best_fit[0]['centrals']['halos_red']
        gals_bf_red = best_fit[0]['centrals']['gals_red']

        halos_bf_blue = best_fit[0]['centrals']['halos_blue']
        gals_bf_blue = best_fit[0]['centrals']['gals_blue']

        y_bf_red,x_bf_red,binnum_red = bs(gals_bf_red,\
        halos_bf_red,'mean',bins=np.linspace(8.6, 12, 15))
        y_bf_blue,x_bf_blue,binnum_blue = bs(gals_bf_blue,\
        halos_bf_blue,'mean',bins=np.linspace(8.6, 12, 15))

        i_outer = 0
        red_mod_x_arr = []
        red_mod_y_arr = []
        blue_mod_x_arr = []
        blue_mod_y_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_y_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_x_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_y_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_x_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_y_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_x_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_y_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_x_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_y_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_x_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_y_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_x_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_y_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_x_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_y_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_x_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 11.5, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_mod_y_ii = models[i_outer][0]['centrals']['halos_red'][idx]
                red_mod_x_ii = models[i_outer][0]['centrals']['gals_red'][idx]
                red_y,red_x,binnum = bs(red_mod_x_ii,red_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                red_mod_x_arr.append(red_x)
                red_mod_y_arr.append(red_y)
                blue_mod_y_ii = models[i_outer][0]['centrals']['halos_blue'][idx]
                blue_mod_x_ii = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_y,blue_x,binnum = bs(blue_mod_x_ii,blue_mod_y_ii,'mean',
                    bins=np.linspace(8.6, 12, 15))
                blue_mod_x_arr.append(blue_x)
                blue_mod_y_arr.append(blue_y)
            i_outer += 1

        red_y_max = np.nanmax(red_mod_y_arr, axis=0)
        red_y_min = np.nanmin(red_mod_y_arr, axis=0)
        blue_y_max = np.nanmax(blue_mod_y_arr, axis=0)
        blue_y_min = np.nanmin(blue_mod_y_arr, axis=0)

        fig1 = plt.figure(figsize=(10,10))

        red_x_cen =  0.5 * (red_mod_x_arr[0][1:] + red_mod_x_arr[0][:-1])
        blue_x_cen = 0.5 * (blue_mod_x_arr[0][1:] + blue_mod_x_arr[0][:-1])

        mr = plt.fill_between(x=red_x_cen, y1=red_y_max, 
            y2=red_y_min, color='lightcoral',alpha=0.4,label='Models')
        mb = plt.fill_between(x=blue_x_cen, y1=blue_y_max, 
            y2=blue_y_min, color='cornflowerblue',alpha=0.4,label='Models')

        red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
        blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

        # REMOVED ERROR BAR ON BEST FIT
        bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=3,label='Best-fit',
            zorder=10)
        bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=3,
            label='Best-fit',zorder=10)

        if settings.survey == 'resolvea' and settings.mf_type == 'smf':
            plt.ylim(10,14)
        else:
            plt.ylim(10,)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        if settings.mf_type == 'smf':
            if settings.survey == 'eco':
                plt.xlim(np.log10((10**8.9)/2.041),)
                plt.title('ECO')
            elif settings.survey == 'resolvea':
                plt.xlim(np.log10((10**8.9)/2.041),13)
            elif settings.survey == 'resolveb':
                plt.xlim(np.log10((10**8.7)/2.041),)
            plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        elif settings.mf_type == 'bmf':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                plt.xlim(np.log10((10**9.4)/2.041),)
                plt.title('ECO')
            elif settings.survey == 'resolveb':
                plt.xlim(np.log10((10**9.1)/2.041),)
            plt.xlabel(r'\boldmath$\log_{10}\ M_{b} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

        plt.legend([(mr, mb), (bfr, bfb)], ['Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, 
            loc='upper left',prop={'size': 30})

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')
        
        plt.show()

    def plot_red_fraction_cen(self, models, best_fit):
        """
        Plot red fraction of centrals from best fit param values and param values 
        corresponding to 68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        cen_gals_red: array
            Array of red central stellar mass values for best-fit model

        cen_halos_red: array
            Array of red central halo mass values for best-fit model

        cen_gals_blue: array
            Array of blue central stellar mass values for best-fit model

        cen_halos_blue: array
            Array of blue central halo mass values for best-fit model

        f_red_cen_red: array
            Array of red fractions for red centrals for best-fit model

        f_red_cen_blue: array
            Array of red fractions for blue centrals for best-fit model
            
        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings
        preprocess = self.preprocess

        halos_bf_red = best_fit[0]['centrals']['halos_red']
        gals_bf_red = best_fit[0]['centrals']['gals_red']

        halos_bf_blue = best_fit[0]['centrals']['halos_blue']
        gals_bf_blue = best_fit[0]['centrals']['gals_blue']

        fred_bf_red = best_fit[0]['f_red']['cen_red']
        fred_bf_blue = best_fit[0]['f_red']['cen_blue']

        cen_gals_arr = []
        cen_halos_arr = []
        fred_arr = []
        i_outer = 0 # There are 5 chunks of all statistics each with len 20
        while i_outer < 5:
            cen_gals_idx_arr = []
            cen_halos_idx_arr = []
            fred_idx_arr = []
            for idx in range(len(models[i_outer][0]['centrals']['halos_red'])):
                red_cen_gals_idx = models[i_outer][0]['centrals']['gals_red'][idx]
                red_cen_halos_idx = models[i_outer][0]['centrals']['halos_red'][idx]
                blue_cen_gals_idx = models[i_outer][0]['centrals']['gals_blue'][idx]
                blue_cen_halos_idx = models[i_outer][0]['centrals']['halos_blue'][idx]
                fred_red_cen_idx = models[i_outer][0]['f_red']['cen_red'][idx]
                fred_blue_cen_idx = models[i_outer][0]['f_red']['cen_blue'][idx]

                cen_gals_idx_arr = list(red_cen_gals_idx) + list(blue_cen_gals_idx)
                cen_gals_arr.append(cen_gals_idx_arr)

                cen_halos_idx_arr = list(red_cen_halos_idx) + list(blue_cen_halos_idx)
                cen_halos_arr.append(cen_halos_idx_arr)

                fred_idx_arr = list(fred_red_cen_idx) + list(fred_blue_cen_idx)
                fred_arr.append(fred_idx_arr)
                
                cen_gals_idx_arr = []
                cen_halos_idx_arr = []
                fred_idx_arr = []

            i_outer+=1
        
        cen_gals_bf = []
        cen_halos_bf = []
        fred_bf = []

        cen_gals_bf = list(gals_bf_red) + list(gals_bf_blue)
        cen_halos_bf = list(halos_bf_red) + list(halos_bf_blue)
        fred_bf = list(fred_bf_red) + list(fred_bf_blue)

        fig1 = plt.figure(figsize=(10,8))
        if settings.quenching == 'hybrid':
            for idx in range(len(cen_gals_arr)):
                x, y = zip(*sorted(zip(cen_gals_arr[idx],fred_arr[idx])))
                plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, solid_capstyle='round')
            plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

            x, y = zip(*sorted(zip(cen_gals_arr[0],fred_arr[0])))
            x_bf, y_bf = zip(*sorted(zip(cen_gals_bf,fred_bf)))
            # Plotting again just so that adding label to legend is easier
            plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, solid_capstyle='round')
            plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, solid_capstyle='round')

        elif settings.quenching == 'halo':
            for idx in range(len(cen_halos_arr)):
                x, y = zip(*sorted(zip(cen_halos_arr[idx],fred_arr[idx])))
                plt.plot(x, y, alpha=0.4, c='cornflowerblue', lw=10, solid_capstyle='round')
            plt.xlabel(r'\boldmath$\log_{10}\ M_{h, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

            x, y = zip(*sorted(zip(cen_halos_arr[0],fred_arr[0])))
            x_bf, y_bf = zip(*sorted(zip(cen_halos_bf, fred_bf)))
            # Plotting again just so that adding label to legend is easier
            plt.plot(x, y, alpha=0.4, c='cornflowerblue', label='Models', lw=10, solid_capstyle='round')
            plt.plot(x_bf, y_bf, c='mediumorchid', label='Best-fit', lw=10, solid_capstyle='round')

        plt.ylabel(r'\boldmath$f_{red, cen}$', fontsize=30)
        plt.legend(loc='best', prop={'size':30})

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_red_fraction_sat(self, models, best_fit):
        """
        Plot red fraction of satellites from best fit param values and param values 
        corresponding to 68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        sat_gals_red: array
            Array of red satellite stellar mass values for best-fit model

        sat_halos_red: array
            Array of red satellite host halo mass values for best-fit model

        sat_gals_blue: array
            Array of blue satellite stellar mass values for best-fit model

        sat_halos_blue: array
            Array of blue satellite host halo mass values for best-fit model

        f_red_sat_red: array
            Array of red fractions for red satellites for best-fit model

        f_red_sat_blue: array
            Array of red fractions for blue satellites for best-fit model
            
        Returns
        ---------
        Plot displayed on screen.
        """
        settings = self.settings

        halos_bf_red = best_fit[0]['satellites']['halos_red']
        gals_bf_red = best_fit[0]['satellites']['gals_red']

        halos_bf_blue = best_fit[0]['satellites']['halos_blue']
        gals_bf_blue = best_fit[0]['satellites']['gals_blue']

        fred_bf_red = best_fit[0]['f_red']['sat_red']
        fred_bf_blue = best_fit[0]['f_red']['sat_blue']

        sat_gals_arr = []
        sat_halos_arr = []
        fred_arr = []
        i_outer = 0 # There are 5 chunks of all 16 statistics each with len 20
        while i_outer < 5:
            sat_gals_idx_arr = []
            sat_halos_idx_arr = []
            fred_idx_arr = []
            for idx in range(len(models[i_outer][0]['satellites']['halos_red'])):
                red_sat_gals_idx = models[i_outer][0]['satellites']['gals_red'][idx]
                red_sat_halos_idx = models[i_outer][0]['satellites']['halos_red'][idx]
                blue_sat_gals_idx = models[i_outer][0]['satellites']['gals_blue'][idx]
                blue_sat_halos_idx = models[i_outer][0]['satellites']['halos_blue'][idx]
                fred_red_sat_idx = models[i_outer][0]['f_red']['sat_red'][idx]
                fred_blue_sat_idx = models[i_outer][0]['f_red']['sat_blue'][idx]

                sat_gals_idx_arr = list(red_sat_gals_idx) + list(blue_sat_gals_idx)
                sat_gals_arr.append(sat_gals_idx_arr)

                sat_halos_idx_arr = list(red_sat_halos_idx) + list(blue_sat_halos_idx)
                sat_halos_arr.append(sat_halos_idx_arr)

                fred_idx_arr = list(fred_red_sat_idx) + list(fred_blue_sat_idx)
                fred_arr.append(fred_idx_arr)
                
                sat_gals_idx_arr = []
                sat_halos_idx_arr = []
                fred_idx_arr = []

            i_outer+=1
        
        sat_gals_arr = np.array(sat_gals_arr)
        sat_halos_arr = np.array(sat_halos_arr)
        fred_arr = np.array(fred_arr)

        if settings.quenching == 'hybrid':
            sat_mean_stats = bs(np.hstack(sat_gals_arr), np.hstack(fred_arr), bins=10)
            sat_std_stats = bs(np.hstack(sat_gals_arr), np.hstack(fred_arr), 
                statistic='std', bins=10)
            sat_stats_bincens = 0.5 * (sat_mean_stats[1][1:] + sat_mean_stats[1][:-1])

        elif settings.quenching == 'halo':
            sat_mean_stats = bs(np.hstack(sat_halos_arr), np.hstack(fred_arr), bins=10)
            sat_std_stats = bs(np.hstack(sat_halos_arr), np.hstack(fred_arr), 
                statistic='std', bins=10)
            sat_stats_bincens = 0.5 * (sat_mean_stats[1][1:] + sat_mean_stats[1][:-1])

        sat_gals_bf = []
        sat_halos_bf = []
        fred_bf = []
        sat_gals_bf = list(gals_bf_red) + list(gals_bf_blue)
        sat_halos_bf = list(halos_bf_red) + list(halos_bf_blue)
        fred_bf = list(fred_bf_red) + list(fred_bf_blue)


        fig1 = plt.figure(figsize=(10,8))
        if settings.quenching == 'hybrid':
            plt.errorbar(sat_stats_bincens, sat_mean_stats[0], 
                yerr=sat_std_stats[0], color='navy', fmt='s', 
                ecolor='navy',markersize=12, capsize=7, capthick=1.5, 
                zorder=10, marker='p', label='Model average')
            plt.xlabel(r'\boldmath$\log_{10}\ M_{*, sat} \left[\mathrm{M_\odot}\,'\
                        r' \mathrm{h}^{-1} \right]$',fontsize=30)

            plt.scatter(sat_gals_bf, fred_bf, alpha=0.4, s=150, c=sat_halos_bf, 
                cmap='viridis' ,label='Best-fit')
            plt.colorbar(label=r'\boldmath$\log_{10}\ M_{h, host}$')

        elif settings.quenching == 'halo':
            plt.errorbar(sat_stats_bincens, sat_mean_stats[0], 
                yerr=sat_std_stats[0], color='navy', fmt='s', 
                ecolor='navy',markersize=12, capsize=7, capthick=1.5, 
                zorder=10, marker='p', label='Model average')
            plt.xlabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\,'\
                        r' \mathrm{h}^{-1} \right]$',fontsize=30)
            plt.scatter(sat_halos_bf, fred_bf, alpha=0.4, s=150, c='mediumorchid',\
                label='Best-fit', zorder=10)

        plt.ylabel(r'\boldmath$f_{red, sat}$', fontsize=30)
        plt.legend(loc='best', prop={'size':30})

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_zumand_fig4(self, models, data, best_fit):
        """
        Plot red and blue SMHM from best fit param values and param values 
        corresponding to 68th percentile 100 lowest chi^2 values like Fig 4 from 
        Zu and Mandelbaum paper

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information
        
        gals_bf_red: array
            Array of y-axis stellar mass values for red SMHM for best-fit model

        halos_bf_red: array
            Array of x-axis halo mass values for red SMHM for best-fit model
        
        gals_bf_blue: array
            Array of y-axis stellar mass values for blue SMHM for best-fit model

        halos_bf_blue: array
            Array of x-axis halo mass values for blue SMHM for best-fit model

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings
        preprocess = self.preprocess

        dof = data[6]

        logmhalo_mod_arr = models[0][0]['centrals']['halos_red']
        for idx in range(5):
            idx+=1
            if idx == 5:
                break
            logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, models[idx][0]['centrals']['halos_red'])
        for idx in range(5):
            logmhalo_mod_arr = np.insert(logmhalo_mod_arr, -1, models[idx][0]['centrals']['halos_blue'])
        logmhalo_mod_arr_flat = np.hstack(logmhalo_mod_arr)

        logmstar_mod_arr = models[0][0]['centrals']['gals_red']
        for idx in range(5):
            idx+=1
            if idx == 5:
                break
            logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, models[idx][0]['centrals']['gals_red'])
        for idx in range(5):
            logmstar_mod_arr = np.insert(logmstar_mod_arr, -1, models[idx][0]['centrals']['gals_blue'])
        logmstar_mod_arr_flat = np.hstack(logmstar_mod_arr)

        fred_mod_arr = models[0][0]['f_red']['cen_red']
        for idx in range(5):
            idx+=1
            if idx == 5:
                break
            fred_mod_arr = np.insert(fred_mod_arr, -1, models[idx][0]['f_red']['cen_red'])
        for idx in range(5):
            fred_mod_arr = np.insert(fred_mod_arr, -1, models[idx][0]['f_red']['cen_blue'])
        fred_mod_arr_flat = np.hstack(fred_mod_arr)

        halos_bf_red = best_fit[0]['centrals']['halos_red']
        gals_bf_red = best_fit[0]['centrals']['gals_red']

        halos_bf_blue = best_fit[0]['centrals']['halos_blue']
        gals_bf_blue = best_fit[0]['centrals']['gals_blue']

        y_bf_red,x_bf_red,binnum_red = bs(halos_bf_red,\
        gals_bf_red,'mean',bins=np.linspace(10, 15, 15))

        y_bf_blue,x_bf_blue,binnum_blue = bs(halos_bf_blue,\
        gals_bf_blue,'mean',bins=np.linspace(10, 15, 15))
        
        fig1 = plt.figure()
        plt.hexbin(logmhalo_mod_arr_flat, logmstar_mod_arr_flat, 
            C=fred_mod_arr_flat, cmap='rainbow')
        cb = plt.colorbar()
        cb.set_label(r'\boldmath\ $f_{red}$')

        red_x_cen =  0.5 * (x_bf_red[1:] + x_bf_red[:-1])
        blue_x_cen = 0.5 * (x_bf_blue[1:] + x_bf_blue[:-1])

        # REMOVED ERROR BAR ON BEST FIT
        bfr, = plt.plot(red_x_cen,y_bf_red,color='darkred',lw=5,zorder=10)
        bfb, = plt.plot(blue_x_cen,y_bf_blue,color='darkblue',lw=5, zorder=10)

        plt.annotate(r'$\boldsymbol\chi ^2 / dof \approx$ {0}'.
            format(np.round(preprocess.bf_chi2/dof,2)), 
            xy=(0.02, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        plt.ylim(8.45, 12.3)
        plt.xlim(10, 14.5)

        plt.fill([13.5, plt.gca().get_xlim()[1], plt.gca().get_xlim()[1], 13.5], 
            [plt.gca().get_ylim()[0], plt.gca().get_ylim()[0], 
            plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]], fill=False, 
            hatch='\\')

        plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')
        plt.ylabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$')


        plt.legend([(bfr, bfb)], ['Best-fit'], 
            handler_map={tuple: HandlerTuple(ndivide=2, pad=0.3)}, loc='best', 
            prop={'size': 30})

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        # if survey == 'eco':
        #     plt.title('ECO')
        
        plt.show()

    def plot_mean_grpcen_vs_sigma(self, models, data, data_experiments, best_fit):
        """
        Plot average group central stellar mass vs. velocity dispersion from data, 
        best fit param values and param values corresponding to 68th percentile 100 
        lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information

        red_sigma_bf: array
            Array of velocity dispersion around red group centrals for best-fit 
            model

        grp_red_cen_stellar_mass_bf: array
            Array of red group central stellar masses for best-fit model

        blue_sigma_bf: array
            Array of velocity dispersion around blue group centrals for best-fit 
            model

        grp_blue_cen_stellar_mass_bf: array
            Array of blue group central stellar masses for best-fit model

        red_sigma_data: array
            Array of velocity dispersion around red group centrals for data

        grp_red_cen_stellar_mass_data: array
            Array of red group central stellar masses for data

        blue_sigma_data: array
            Array of velocity dispersion around blue group centrals for data

        grp_blue_cen_stellar_mass_data: array
            Array of blue group central stellar masses for data

        err_red: array
            Array of std values per bin of red group central stellar mass vs. 
            velocity dispersion from mocks

        err_blue: array
            Array of std values per bin of blue group central stellar mass vs. 
            velocity dispersion from mocks

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings

        error_red = data[5]['vel_disp']['std_mean_mstar_red']
        error_blue = data[5]['vel_disp']['std_mean_mstar_blue']

        x_sigma_red_data, y_sigma_red_data = data_experiments['vel_disp']['red_sigma'],\
            data_experiments['vel_disp']['red_cen_mstar']
        x_sigma_blue_data, y_sigma_blue_data = data_experiments['vel_disp']['blue_sigma'],\
            data_experiments['vel_disp']['blue_cen_mstar']

        x_sigma_red_bf, y_sigma_red_bf = best_fit[1]['vel_disp']['red_sigma'],\
            best_fit[1]['vel_disp']['red_cen_mstar']
        x_sigma_blue_bf, y_sigma_blue_bf = best_fit[1]['vel_disp']['blue_sigma'],\
            best_fit[1]['vel_disp']['blue_cen_mstar']

        mean_grp_red_cen_gals_arr = []
        mean_grp_blue_cen_gals_arr = []
        red_sigma_arr = []
        blue_sigma_arr = []
        chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
        while chunk_counter < 5:
            for idx in range(len(models[chunk_counter][1]['vel_disp'])):
                grp_red_cen_gals_idx = models[chunk_counter][1]['vel_disp']['red_cen_mstar'][idx]
                grp_blue_cen_gals_idx = models[chunk_counter][1]['vel_disp']['blue_cen_mstar'][idx]
                red_sigma_idx = models[chunk_counter][1]['vel_disp']['red_sigma'][idx]
                blue_sigma_idx = models[chunk_counter][1]['vel_disp']['blue_sigma'][idx]

                mean_stats_red = bs(red_sigma_idx, grp_red_cen_gals_idx, 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_stats_blue = bs(blue_sigma_idx, grp_blue_cen_gals_idx, 
                    statistic='mean', bins=np.linspace(0,250,6))
                red_sigma_arr.append(mean_stats_red[1])
                blue_sigma_arr.append(mean_stats_blue[1])
                mean_grp_red_cen_gals_arr.append(mean_stats_red[0])
                mean_grp_blue_cen_gals_arr.append(mean_stats_blue[0])

            chunk_counter+=1

        red_models_max = np.nanmax(mean_grp_red_cen_gals_arr, axis=0)
        red_models_min = np.nanmin(mean_grp_red_cen_gals_arr, axis=0)
        blue_models_max = np.nanmax(mean_grp_blue_cen_gals_arr, axis=0)
        blue_models_min = np.nanmin(mean_grp_blue_cen_gals_arr, axis=0)

        ## Same centers used for all sets of lines since binning is the same for 
        ## models, bf and data
        mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
            mean_stats_red[1][:-1])
        mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
            mean_stats_blue[1][:-1])

        mean_stats_red_bf = bs(x_sigma_red_bf, y_sigma_red_bf, 
            statistic='mean', bins=np.linspace(0,250,6))
        mean_stats_blue_bf = bs(x_sigma_blue_bf, y_sigma_blue_bf, 
            statistic='mean', bins=np.linspace(0,250,6))

        mean_stats_red_data = bs(x_sigma_red_data, y_sigma_red_data, 
            statistic='mean', bins=np.linspace(0,250,6))
        mean_stats_blue_data = bs(x_sigma_blue_data, y_sigma_blue_data,
            statistic='mean', bins=np.linspace(0,250,6))

        fig1 = plt.subplots(figsize=(10,8))

        dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=error_red,
                color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=error_blue,
                color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        
        mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
            y2=red_models_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
            y2=blue_models_min, color='cornflowerblue',alpha=0.4)

        bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
        bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

        l = plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
            ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

        chi_squared_red = np.nansum((mean_stats_red_data[0] - 
            mean_stats_red_bf[0])**2 / (error_red**2))
        chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
            mean_stats_blue_bf[0])**2 / (error_blue**2))

        plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
            r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
            chi_squared_red,2),np.round(chi_squared_blue,2)), 
            xy=(0.015, 0.65), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        dof_red = 5 # Nparams=0, Nbins=5
        dof_blue = 5 
        plt.annotate(r'$ p \approx$ {0}''\n'\
            r'$ p \approx$ {1}'.format(
            np.round((1 - chi2.cdf(chi_squared_red, dof_red)),2),
            np.round((1 - chi2.cdf(chi_squared_blue, dof_blue)),2)), 
            xy=(0.015, 0.55), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)


        plt.ylim(8.9,)

        plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        
        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_mean_sigma_vs_grpcen(self, models, data, data_experiments, best_fit):
        """[summary]

        Args:
            result ([type]): [description]
            std_red_data ([type]): [description]
            cen_red_data ([type]): [description]
            std_blue_data ([type]): [description]
            cen_blue_data ([type]): [description]
            std_bf_red ([type]): [description]
            std_bf_blue ([type]): [description]
            std_cen_bf_red ([type]): [description]
            std_cen_bf_blue ([type]): [description]
            bf_chi2 ([type]): [description]
        """
        settings = self.settings

        error_red = data[5]['vel_disp']['std_mean_sigma_red']
        error_blue = data[5]['vel_disp']['std_mean_sigma_blue']

        dof = data[6]

        x_sigma_red_data, y_sigma_red_data = data_experiments['vel_disp']['red_cen_mstar'],\
            data_experiments['vel_disp']['red_sigma']
        x_sigma_blue_data, y_sigma_blue_data = data_experiments['vel_disp']['blue_cen_mstar'],\
            data_experiments['vel_disp']['blue_sigma']

        x_sigma_red_bf, y_sigma_red_bf = best_fit[1]['vel_disp']['red_cen_mstar'],\
            best_fit[1]['vel_disp']['red_sigma']
        x_sigma_blue_bf, y_sigma_blue_bf = best_fit[1]['vel_disp']['blue_cen_mstar'],\
            best_fit[1]['vel_disp']['blue_sigma']

        if settings.survey == 'eco' or settings.survey == 'resolvea':
            # TODO : check if this is actually correct for resolve a
            red_stellar_mass_bins = np.linspace(8.6,11.2,6)
        elif settings.survey == 'resolveb':
            red_stellar_mass_bins = np.linspace(8.4,11.0,6)

        if settings.survey == 'eco' or settings.survey == 'resolvea':
            # TODO : check if this is actually correct for resolve a
            blue_stellar_mass_bins = np.linspace(8.6,10.7,6)
        elif settings.survey == 'resolveb':
            blue_stellar_mass_bins = np.linspace(8.4,10.4,6)


        grp_red_cen_gals_arr = []
        grp_blue_cen_gals_arr = []
        mean_red_sigma_arr = []
        mean_blue_sigma_arr = []
        chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
        while chunk_counter < 5:
            for idx in range(len(models[chunk_counter][1]['vel_disp'])):
                grp_red_cen_gals_idx = models[chunk_counter][1]['vel_disp']['red_cen_mstar'][idx]
                grp_blue_cen_gals_idx = models[chunk_counter][1]['vel_disp']['blue_cen_mstar'][idx]
                red_sigma_idx = models[chunk_counter][1]['vel_disp']['red_sigma'][idx]
                blue_sigma_idx = models[chunk_counter][1]['vel_disp']['blue_sigma'][idx]

                mean_stats_red = bs(grp_red_cen_gals_idx, red_sigma_idx,
                    statistic='mean', bins=red_stellar_mass_bins)
                mean_stats_blue = bs(grp_blue_cen_gals_idx, blue_sigma_idx,
                    statistic='mean', bins=blue_stellar_mass_bins)
                mean_red_sigma_arr.append(mean_stats_red[0])
                mean_blue_sigma_arr.append(mean_stats_blue[0])
                grp_red_cen_gals_arr.append(mean_stats_red[1])
                grp_blue_cen_gals_arr.append(mean_stats_blue[1])

            chunk_counter+=1

        red_models_max = np.nanmax(mean_red_sigma_arr, axis=0)
        red_models_min = np.nanmin(mean_red_sigma_arr, axis=0)
        blue_models_max = np.nanmax(mean_blue_sigma_arr, axis=0)
        blue_models_min = np.nanmin(mean_blue_sigma_arr, axis=0)

        ## Same centers used for all sets of lines since binning is the same for 
        ## models, bf and data
        mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
            mean_stats_red[1][:-1])
        mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
            mean_stats_blue[1][:-1])

        mean_stats_red_bf = bs(x_sigma_red_bf, y_sigma_red_bf, 
            statistic='mean', bins=red_stellar_mass_bins)
        mean_stats_blue_bf = bs(x_sigma_blue_bf, y_sigma_blue_bf, 
            statistic='mean', bins=blue_stellar_mass_bins)

        mean_stats_red_data = bs(x_sigma_red_data, y_sigma_red_data, 
            statistic='mean', bins=red_stellar_mass_bins)
        mean_stats_blue_data = bs(x_sigma_blue_data, y_sigma_blue_data,
            statistic='mean', bins=blue_stellar_mass_bins)

        fig1= plt.figure(figsize=(10,10))

        mr = plt.fill_between(x=mean_centers_red, y1=red_models_min, 
            y2=red_models_max, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_min, 
            y2=blue_models_max, color='cornflowerblue',alpha=0.4)

        dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],
            yerr=error_red, color='darkred',fmt='^',ecolor='darkred', 
            markersize=12,capsize=10,capthick=1.0,zorder=10)
        db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],
            yerr=error_blue, color='darkblue',fmt='^',ecolor='darkblue',
            markersize=12,capsize=10,capthick=1.0,zorder=10)

        bfr, = plt.plot(mean_centers_red,mean_stats_red_bf[0],
            color='maroon',ls='-',lw=3,zorder=10)
        bfb, = plt.plot(mean_centers_blue,mean_stats_blue_bf[0],
            color='mediumblue',ls='-',lw=3,zorder=10)

        plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
        plt.ylabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)

        plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
            ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
            loc='upper left')

        chi_squared_red = np.nansum((mean_stats_red_data[0] - 
            mean_stats_red_bf[0])**2 / (error_red**2))
        chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
            mean_stats_blue_bf[0])**2 / (error_blue**2))

        plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
            r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
            chi_squared_red,2),np.round(chi_squared_blue,2)), 
            xy=(0.015, 0.65), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        # plt.annotate(r'$ p \approx$ {0}''\n'\
        #     r'$ p \approx$ {1}'.format(
        #     np.round((1 - chi2.cdf(chi_squared_red, dof)),2),
        #     np.round((1 - chi2.cdf(chi_squared_blue, dof)),2)), 
        #     xy=(0.015, 0.55), xycoords='axes fraction', bbox=dict(boxstyle="square", 
        #     ec='k', fc='lightgray', alpha=0.5), size=25)

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        # if survey == 'eco':
        #     plt.title('ECO')
        plt.show()

    def plot_sigma_host_halo_mass_vishnu(self, models, best_fit):

        i_outer = 0
        vdisp_red_mod_arr = []
        vdisp_blue_mod_arr = []
        hosthalo_red_mod_arr = []
        hosthalo_blue_mod_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

        flat_list_red_vdisp = [item for sublist in vdisp_red_mod_arr for item in sublist]
        flat_list_red_host = [item for sublist in hosthalo_red_mod_arr for item in sublist]

        flat_list_blue_vdisp = [item for sublist in vdisp_blue_mod_arr for item in sublist]
        flat_list_blue_host = [item for sublist in hosthalo_blue_mod_arr for item in sublist]

        x_sigma_red_bf, y_hosthalo_red_bf = best_fit[1]['vel_disp']['red_sigma'],\
            best_fit[1]['vel_disp']['red_hosthalo']
        x_sigma_blue_bf, y_hosthalo_blue_bf = best_fit[1]['vel_disp']['blue_sigma'],\
            best_fit[1]['vel_disp']['blue_hosthalo']

        import seaborn as sns
        fig1 = plt.figure(figsize=(11, 9))
        bfr = plt.scatter(x_sigma_red_bf, np.log10(y_hosthalo_red_bf), c='maroon', s=120, zorder = 10)
        bfb = plt.scatter(x_sigma_blue_bf, np.log10(y_hosthalo_blue_bf), c='darkblue', s=120, zorder=10)

        sns.kdeplot(x=flat_list_red_vdisp, y=np.log10(flat_list_red_host), color='indianred', shade=True)
        sns.kdeplot(x=flat_list_blue_vdisp, y=np.log10(flat_list_blue_host), color='cornflowerblue', shade=True)
        # for idx in range(len(vdisp_red_mod_arr)):
        #     mr = plt.scatter(vdisp_red_mod_arr[idx], np.log10(hosthalo_red_mod_arr[idx]), 
        #         c='indianred', s=120, alpha=0.8, marker='*')
        # for idx in range(len(vdisp_blue_mod_arr)):
        #     mb = plt.scatter(vdisp_blue_mod_arr[idx], np.log10(hosthalo_blue_mod_arr[idx]), 
        #         c='cornflowerblue', s=120, alpha=0.8, marker='*')
        
        plt.legend([(bfr, bfb)], 
            ['Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
            loc='best')

        plt.xlabel(r'\boldmath$\sigma \left[\mathrm{km/s} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.title('Host halo mass - velocity dispersion in best-fit model (excluding singletons)')
        plt.show()

    def plot_N_host_halo_mass_vishnu(self, models, best_fit):

        i_outer = 0
        N_red_mod_arr = []
        N_blue_mod_arr = []
        hosthalo_red_mod_arr = []
        hosthalo_blue_mod_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][1]['richness']['red_num'])):
                red_mod_ii = models[i_outer][1]['richness']['red_num'][idx]
                N_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_num'][idx]
                N_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['richness']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['richness']['red_num'])):
                red_mod_ii = models[i_outer][1]['richness']['red_num'][idx]
                N_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_num'][idx]
                N_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['richness']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['richness']['red_num'])):
                red_mod_ii = models[i_outer][1]['richness']['red_num'][idx]
                N_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_num'][idx]
                N_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['richness']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['richness']['red_num'])):
                red_mod_ii = models[i_outer][1]['richness']['red_num'][idx]
                N_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_num'][idx]
                N_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['richness']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['richness']['red_num'])):
                red_mod_ii = models[i_outer][1]['richness']['red_num'][idx]
                N_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_num'][idx]
                N_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['richness']['red_hosthalo'][idx]
                hosthalo_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['richness']['blue_hosthalo'][idx]
                hosthalo_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

        flat_list_red_N = [item for sublist in N_red_mod_arr for item in sublist]
        flat_list_red_host = [item for sublist in hosthalo_red_mod_arr for item in sublist]

        flat_list_blue_N = [item for sublist in N_blue_mod_arr for item in sublist]
        flat_list_blue_host = [item for sublist in hosthalo_blue_mod_arr for item in sublist]

        x_N_red_bf, y_hosthalo_red_bf = best_fit[1]['richness']['red_num'],\
            best_fit[1]['richness']['red_hosthalo']
        x_N_blue_bf, y_hosthalo_blue_bf = best_fit[1]['richness']['blue_num'],\
            best_fit[1]['richness']['blue_hosthalo']

        import seaborn as sns
        fig1 = plt.figure(figsize=(11, 9))
        bfr = plt.scatter(x_N_red_bf, np.log10(y_hosthalo_red_bf), c='maroon', s=120, zorder = 10)
        N_blue_bf_offset = [x+0.05 for x in x_N_blue_bf] #For plotting purposes
        bfb = plt.scatter(N_blue_bf_offset, np.log10(y_hosthalo_blue_bf), c='darkblue', s=120, zorder=11, alpha=0.3)

        # sns.kdeplot(x=flat_list_red_N, y=np.log10(flat_list_red_host), color='indianred', shade=True)
        # sns.kdeplot(x=flat_list_blue_N, y=np.log10(flat_list_blue_host), color='cornflowerblue', shade=True)
        # for idx in range(len(vdisp_red_mod_arr)):
        #     mr = plt.scatter(vdisp_red_mod_arr[idx], np.log10(hosthalo_red_mod_arr[idx]), 
        #         c='indianred', s=120, alpha=0.8, marker='*')
        # for idx in range(len(vdisp_blue_mod_arr)):
        #     mb = plt.scatter(vdisp_blue_mod_arr[idx], np.log10(hosthalo_blue_mod_arr[idx]), 
        #         c='cornflowerblue', s=120, alpha=0.8, marker='*')
        
        plt.legend([(bfr, bfb)], 
            ['Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
            loc='best')

        plt.xlabel(r'\boldmath$ {N} $', fontsize=30)
        plt.ylabel(r'\boldmath$\log_{10}\ M_{h, host} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        plt.title('Host halo mass - Number of galaxies in halo in best-fit model (excluding singletons)')
        plt.show()

    def plot_mean_grpcen_vs_N(self, models, data, data_experiments, best_fit):
        """
        Plot average halo/group central stellar mass vs. number of galaxies in 
        halos/groups from data, best fit param values and param values corresponding 
        to 68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information

        red_sigma_bf: array
            Array of velocity dispersion around red group centrals for best-fit 
            model

        grp_red_cen_stellar_mass_bf: array
            Array of red group central stellar masses for best-fit model

        blue_sigma_bf: array
            Array of velocity dispersion around blue group centrals for best-fit 
            model

        grp_blue_cen_stellar_mass_bf: array
            Array of blue group central stellar masses for best-fit model

        red_sigma_data: array
            Array of velocity dispersion around red group centrals for data

        grp_red_cen_stellar_mass_data: array
            Array of red group central stellar masses for data

        blue_sigma_data: array
            Array of velocity dispersion around blue group centrals for data

        grp_blue_cen_stellar_mass_data: array
            Array of blue group central stellar masses for data

        err_red: array
            Array of std values per bin of red group central stellar mass vs. 
            velocity dispersion from mocks

        err_blue: array
            Array of std values per bin of blue group central stellar mass vs. 
            velocity dispersion from mocks

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings

        mean_grp_red_cen_gals_arr = []
        mean_grp_blue_cen_gals_arr = []
        red_num_arr = []
        blue_num_arr = []
        chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
        while chunk_counter < 5:
            for idx in range(len(models[chunk_counter][1]['richness'])):
                grp_red_cen_gals_idx = models[chunk_counter][1]['richness']['red_cen_mstar'][idx]
                grp_blue_cen_gals_idx = models[chunk_counter][1]['richness']['blue_cen_mstar'][idx]
                red_num_idx = np.log10(models[chunk_counter][1]['richness']['red_num'][idx])
                blue_num_idx = np.log10(models[chunk_counter][1]['richness']['blue_num'][idx])

                mean_stats_red = bs(red_num_idx, grp_red_cen_gals_idx, 
                    statistic='mean', bins=np.arange(0,2.5,0.5))
                mean_stats_blue = bs(blue_num_idx, grp_blue_cen_gals_idx, 
                    statistic='mean', bins=np.arange(0,0.8,0.2))
                red_num_arr.append(mean_stats_red[1])
                blue_num_arr.append(mean_stats_blue[1])
                mean_grp_red_cen_gals_arr.append(mean_stats_red[0])
                mean_grp_blue_cen_gals_arr.append(mean_stats_blue[0])

            chunk_counter+=1

        red_models_max = np.nanmax(mean_grp_red_cen_gals_arr, axis=0)
        red_models_min = np.nanmin(mean_grp_red_cen_gals_arr, axis=0)
        blue_models_max = np.nanmax(mean_grp_blue_cen_gals_arr, axis=0)
        blue_models_min = np.nanmin(mean_grp_blue_cen_gals_arr, axis=0)

        ## Same centers used for all sets of lines since binning is the same for 
        ## models, bf and data
        mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
            mean_stats_red[1][:-1])
        mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
            mean_stats_blue[1][:-1])

        red_num_bf, red_cen_stellar_mass_bf = best_fit[1]['richness']['red_num'],\
            best_fit[1]['richness']['red_cen_mstar']
        blue_num_bf, blue_cen_stellar_mass_bf = best_fit[1]['richness']['blue_num'],\
            best_fit[1]['richness']['blue_cen_mstar']

        # Max of red_num_bf = 62, Max of red_num_idx above varies (35-145)
        mean_stats_red_bf = bs(np.log10(red_num_bf), red_cen_stellar_mass_bf, 
            statistic='mean', bins=np.arange(0,2.5,0.5))
        # Max of blue_num_bf = 3, Max of blue_num_idx above varies (3-4)
        mean_stats_blue_bf = bs(np.log10(blue_num_bf), blue_cen_stellar_mass_bf, 
            statistic='mean', bins=np.arange(0,0.8,0.2))

        red_num_data, red_cen_stellar_mass_data = data_experiments['richness']['red_num'],\
            data_experiments['richness']['red_cen_mstar']
        blue_num_data, blue_cen_stellar_mass_data = data_experiments['richness']['blue_num'],\
            data_experiments['richness']['blue_cen_mstar']

        # Max of red_num_data = 388
        mean_stats_red_data = bs(np.log10(red_num_data), red_cen_stellar_mass_data, 
            statistic='mean', bins=np.arange(0,2.5,0.5))
        # Max of blue_num_data = 19
        mean_stats_blue_data = bs(np.log10(blue_num_data), blue_cen_stellar_mass_data, 
            statistic='mean', bins=np.arange(0,0.8,0.2))
        
        red_num_mocks = data[5]['richness']['red_num']
        red_cen_stellar_mass_mocks = data[5]['richness']['red_cen_mstar']
        blue_num_mocks = data[5]['richness']['blue_num']
        blue_cen_stellar_mass_mocks = data[5]['richness']['blue_cen_mstar']

        mean_stats_red_mocks_arr = []
        mean_stats_blue_mocks_arr = []
        for idx in range(len(red_cen_stellar_mass_mocks)):
            # Max of red_num_mocks[idx] = 724
            mean_stats_red_mocks = bs(np.log10(red_num_mocks[idx]), red_cen_stellar_mass_mocks[idx], 
                statistic='mean', bins=np.arange(0,2.5,0.5))
            mean_stats_red_mocks_arr.append(mean_stats_red_mocks[0])
        for idx in range(len(blue_cen_stellar_mass_mocks)):
            # Max of blue_num_mocks[idx] = 8
            mean_stats_blue_mocks = bs(np.log10(blue_num_mocks[idx]), blue_cen_stellar_mass_mocks[idx], 
                statistic='mean', bins=np.arange(0,0.8,0.2))
            mean_stats_blue_mocks_arr.append(mean_stats_blue_mocks[0])

        ## Error bars on data points 
        std_mean_cen_arr_red = np.nanstd(mean_stats_red_mocks_arr, axis=0)
        std_mean_cen_arr_blue = np.nanstd(mean_stats_blue_mocks_arr, axis=0)

        fig1 = plt.subplots(figsize=(10,8))

        dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=std_mean_cen_arr_red,
                color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=std_mean_cen_arr_blue,
                color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        
        mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
            y2=red_models_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
            y2=blue_models_min, color='cornflowerblue',alpha=0.4)

        bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
        bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

        plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
            ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

        chi_squared_red = np.nansum((mean_stats_red_data[0] - 
            mean_stats_red_bf[0])**2 / (std_mean_cen_arr_red**2))
        chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
            mean_stats_blue_bf[0])**2 / (std_mean_cen_arr_blue**2))

        plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
            r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
            chi_squared_red,2),np.round(chi_squared_blue,2)), 
            xy=(0.015, 0.7), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        plt.ylim(8.9, 11.5)   
        plt.xlabel(r'\boldmath$ \log_{10}\ {N} $', fontsize=30)
        plt.ylabel(r'\boldmath$\overline{\log_{10}\ M_{*, cen}} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
        
        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_mean_N_vs_grpcen(self, models, data, data_experiments, best_fit):
        """
        Plot average number of galaxies in halos/groups vs. halo/group central 
        stellar mass from data, best fit param values and param values corresponding 
        to 68th percentile 100 lowest chi^2 values.

        Parameters
        ----------
        result: multidimensional array
            Array of SMF, blue fraction and SMHM information

        red_sigma_bf: array
            Array of velocity dispersion around red group centrals for best-fit 
            model

        grp_red_cen_stellar_mass_bf: array
            Array of red group central stellar masses for best-fit model

        blue_sigma_bf: array
            Array of velocity dispersion around blue group centrals for best-fit 
            model

        grp_blue_cen_stellar_mass_bf: array
            Array of blue group central stellar masses for best-fit model

        red_sigma_data: array
            Array of velocity dispersion around red group centrals for data

        grp_red_cen_stellar_mass_data: array
            Array of red group central stellar masses for data

        blue_sigma_data: array
            Array of velocity dispersion around blue group centrals for data

        grp_blue_cen_stellar_mass_data: array
            Array of blue group central stellar masses for data

        err_red: array
            Array of std values per bin of red group central stellar mass vs. 
            velocity dispersion from mocks

        err_blue: array
            Array of std values per bin of blue group central stellar mass vs. 
            velocity dispersion from mocks

        bf_chi2: float
            Chi-squared value associated with the best-fit model

        Returns
        ---------
        Plot displayed on screen.
        """   
        settings = self.settings

        stat = 'mean'

        red_stellar_mass_bins = np.linspace(8.6,10.7,6)
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)

        grp_red_cen_gals_arr = []
        grp_blue_cen_gals_arr = []
        mean_red_num_arr = []
        mean_blue_num_arr = []
        chunk_counter = 0 # There are 5 chunks of all 16 statistics each with len 20
        while chunk_counter < 5:
            for idx in range(len(models[chunk_counter][1]['richness'])):
                grp_red_cen_gals_idx = models[chunk_counter][1]['richness']['red_cen_mstar'][idx]
                grp_blue_cen_gals_idx = models[chunk_counter][1]['richness']['blue_cen_mstar'][idx]
                red_num_idx = models[chunk_counter][1]['richness']['red_num'][idx]
                blue_num_idx = models[chunk_counter][1]['richness']['blue_num'][idx]

                mean_stats_red = bs(grp_red_cen_gals_idx, red_num_idx,
                    statistic=stat, bins=red_stellar_mass_bins)
                mean_stats_blue = bs(grp_blue_cen_gals_idx, blue_num_idx,
                    statistic=stat, bins=blue_stellar_mass_bins)
                mean_red_num_arr.append(mean_stats_red[0])
                mean_blue_num_arr.append(mean_stats_blue[0])
                grp_red_cen_gals_arr.append(mean_stats_red[1])
                grp_blue_cen_gals_arr.append(mean_stats_blue[1])

            chunk_counter+=1

        red_models_max = np.nanmax(mean_red_num_arr, axis=0)
        red_models_min = np.nanmin(mean_red_num_arr, axis=0)
        blue_models_max = np.nanmax(mean_blue_num_arr, axis=0)
        blue_models_min = np.nanmin(mean_blue_num_arr, axis=0)

        ## Same centers used for all sets of lines since binning is the same for 
        ## models, bf and data
        mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
            mean_stats_red[1][:-1])
        mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
            mean_stats_blue[1][:-1])

        red_num_bf, red_cen_stellar_mass_bf = best_fit[1]['richness']['red_num'],\
            best_fit[1]['richness']['red_cen_mstar']
        blue_num_bf, blue_cen_stellar_mass_bf = best_fit[1]['richness']['blue_num'],\
            best_fit[1]['richness']['blue_cen_mstar']

        mean_stats_red_bf = bs(red_cen_stellar_mass_bf, red_num_bf, 
            statistic=stat, bins=red_stellar_mass_bins)
        mean_stats_blue_bf = bs(blue_cen_stellar_mass_bf, blue_num_bf,
            statistic=stat, bins=blue_stellar_mass_bins)

        red_num_data, red_cen_stellar_mass_data = data_experiments['richness']['red_num'],\
            data_experiments['richness']['red_cen_mstar']
        blue_num_data, blue_cen_stellar_mass_data = data_experiments['richness']['blue_num'],\
            data_experiments['richness']['blue_cen_mstar']

        mean_stats_red_data = bs(red_cen_stellar_mass_data, red_num_data,
            statistic=stat, bins=red_stellar_mass_bins)
        mean_stats_blue_data = bs(blue_cen_stellar_mass_data, blue_num_data,
            statistic=stat, bins=blue_stellar_mass_bins)

        red_num_mocks = data[5]['richness']['red_num']
        red_cen_stellar_mass_mocks = data[5]['richness']['red_cen_mstar']
        blue_num_mocks = data[5]['richness']['blue_num']
        blue_cen_stellar_mass_mocks = data[5]['richness']['blue_cen_mstar']

        mean_stats_red_mocks_arr = []
        mean_stats_blue_mocks_arr = []
        for idx in range(len(red_cen_stellar_mass_mocks)):
            mean_stats_red_mocks = bs(red_cen_stellar_mass_mocks[idx], red_num_mocks[idx],
                statistic=stat, bins=red_stellar_mass_bins)
            mean_stats_red_mocks_arr.append(mean_stats_red_mocks[0])
        for idx in range(len(blue_cen_stellar_mass_mocks)):
            mean_stats_blue_mocks = bs(blue_cen_stellar_mass_mocks[idx], blue_num_mocks[idx],
                statistic=stat, bins=blue_stellar_mass_bins)
            mean_stats_blue_mocks_arr.append(mean_stats_blue_mocks[0])

        ## Error bars on data points 
        std_mean_cen_arr_red = np.nanstd(mean_stats_red_mocks_arr, axis=0)
        std_mean_cen_arr_blue = np.nanstd(mean_stats_blue_mocks_arr, axis=0)

        fig1 = plt.subplots(figsize=(10,8))

        dr = plt.errorbar(mean_centers_red,mean_stats_red_data[0],yerr=std_mean_cen_arr_red,
                color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        db = plt.errorbar(mean_centers_blue,mean_stats_blue_data[0],yerr=std_mean_cen_arr_blue,
                color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
                capthick=1.0,zorder=10)
        
        mr = plt.fill_between(x=mean_centers_red, y1=red_models_max, 
            y2=red_models_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=mean_centers_blue, y1=blue_models_max, 
            y2=blue_models_min, color='cornflowerblue',alpha=0.4)

        bfr, = plt.plot(mean_centers_red, mean_stats_red_bf[0], c='indianred', zorder=9)
        bfb, = plt.plot(mean_centers_blue, mean_stats_blue_bf[0], c='cornflowerblue', zorder=9)

        plt.legend([(dr, db), (mr, mb), (bfr, bfb)], 
            ['Data','Models','Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, loc='upper left')

        chi_squared_red = np.nansum((mean_stats_red_data[0] - 
            mean_stats_red_bf[0])**2 / (std_mean_cen_arr_red**2))
        chi_squared_blue = np.nansum((mean_stats_blue_data[0] - 
            mean_stats_blue_bf[0])**2 / (std_mean_cen_arr_blue**2))

        plt.annotate(r'$\boldsymbol\chi ^2_{{red}} \approx$ {0}''\n'\
            r'$\boldsymbol\chi ^2_{{blue}} \approx$ {1}'.format(np.round(\
            chi_squared_red,2),np.round(chi_squared_blue,2)), 
            xy=(0.015, 0.7), xycoords='axes fraction', bbox=dict(boxstyle="square", 
            ec='k', fc='lightgray', alpha=0.5), size=25)

        if stat == 'mean':
            plt.ylabel(r'\boldmath$\overline{N}$',fontsize=30)
        elif stat == 'median':
            plt.ylabel(r'\boldmath${N_{median}}$',fontsize=30)
        plt.xlabel(r'\boldmath$\log_{10}\ M_{*, cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)

    
        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')

        plt.show()

    def plot_satellite_weighted_sigma(self, models, data_experiments, best_fit):

        settings = self.settings

        def sw_func_red(arr):
            result = np.sum(arr)
            return result

        def sw_func_blue(arr):
            result = np.sum(arr)
            return result

        def hw_func_red(arr):
            result = np.sum(arr)/len(arr)
            return result

        def hw_func_blue(arr):
            result = np.sum(arr)/len(arr)
            return result
        
        i_outer = 0
        vdisp_red_mod_arr = []
        vdisp_blue_mod_arr = []
        cen_mstar_red_mod_arr = []
        cen_mstar_blue_mod_arr = []
        nsat_red_mod_arr = []
        nsat_blue_mod_arr = []
        while i_outer < 5:
            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_cen_mstar'][idx]
                cen_mstar_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_cen_mstar'][idx]
                cen_mstar_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_nsat'][idx]
                nsat_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_nsat'][idx]
                nsat_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_cen_mstar'][idx]
                cen_mstar_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_cen_mstar'][idx]
                cen_mstar_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_nsat'][idx]
                nsat_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_nsat'][idx]
                nsat_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_cen_mstar'][idx]
                cen_mstar_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_cen_mstar'][idx]
                cen_mstar_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_nsat'][idx]
                nsat_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_nsat'][idx]
                nsat_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_cen_mstar'][idx]
                cen_mstar_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_cen_mstar'][idx]
                cen_mstar_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_nsat'][idx]
                nsat_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_nsat'][idx]
                nsat_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

            for idx in range(len(models[i_outer][1]['vel_disp']['red_sigma'])):
                red_mod_ii = models[i_outer][1]['vel_disp']['red_sigma'][idx]
                vdisp_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_sigma'][idx]
                vdisp_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_cen_mstar'][idx]
                cen_mstar_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_cen_mstar'][idx]
                cen_mstar_blue_mod_arr.append(blue_mod_ii)

                red_mod_ii = models[i_outer][1]['vel_disp']['red_nsat'][idx]
                nsat_red_mod_arr.append(red_mod_ii)
                blue_mod_ii = models[i_outer][1]['vel_disp']['blue_nsat'][idx]
                nsat_blue_mod_arr.append(blue_mod_ii)
            i_outer += 1

        vdisp_red_mod_arr = np.array(vdisp_red_mod_arr, dtype=object)
        vdisp_blue_mod_arr = np.array(vdisp_blue_mod_arr, dtype=object)
        cen_mstar_red_mod_arr = np.array(cen_mstar_red_mod_arr, dtype=object)
        cen_mstar_blue_mod_arr = np.array(cen_mstar_blue_mod_arr, dtype=object)
        nsat_red_mod_arr = np.array(nsat_red_mod_arr, dtype=object)
        nsat_blue_mod_arr = np.array(nsat_blue_mod_arr, dtype=object)

        red_stellar_mass_bins = np.linspace(8.6,11.2,6)
        blue_stellar_mass_bins = np.linspace(8.6,10.7,6)

        ## Models
        ratio_red_mod_arr = []
        ratio_blue_mod_arr = []
        for idx in range(len(vdisp_red_mod_arr)):
            # nsat_red_total = np.sum(nsat_red_mod_arr[idx])
            # nsat_blue_total = np.sum(nsat_blue_mod_arr[idx])

            nsat_red = bs(cen_mstar_red_mod_arr[idx], nsat_red_mod_arr[idx], 'sum', 
                bins=red_stellar_mass_bins)
            nsat_blue = bs(cen_mstar_blue_mod_arr[idx], nsat_blue_mod_arr[idx], 'sum',
                bins=blue_stellar_mass_bins)

            nsat_vdisp_product_red = np.array(nsat_red_mod_arr[idx]) * \
                (np.array(vdisp_red_mod_arr[idx])**2)
            nsat_vdisp_product_blue = np.array(nsat_blue_mod_arr[idx]) * \
                (np.array(vdisp_blue_mod_arr[idx])**2)

            sw_mean_stats_red_mod = bs(cen_mstar_red_mod_arr[idx], 
                nsat_vdisp_product_red, statistic=sw_func_red, 
                bins=red_stellar_mass_bins)
            ## Nsat is number of satellites stacked in a bin in More et al Eq. 1
            sw_mean_stats_red_mod_ii = sw_mean_stats_red_mod[0]/nsat_red[0]


            sw_mean_stats_blue_mod = bs(cen_mstar_blue_mod_arr[idx], 
                nsat_vdisp_product_blue, statistic=sw_func_blue, 
                bins=blue_stellar_mass_bins)
            sw_mean_stats_blue_mod_ii = sw_mean_stats_blue_mod[0]/nsat_blue[0]


            hw_mean_stats_red_mod = bs(cen_mstar_red_mod_arr[idx], 
                np.array(vdisp_red_mod_arr[idx])**2, statistic=hw_func_red, 
                bins=red_stellar_mass_bins)

            hw_mean_stats_blue_mod = bs(cen_mstar_blue_mod_arr[idx], 
                np.array(vdisp_blue_mod_arr[idx])**2, statistic=hw_func_blue, 
                bins=blue_stellar_mass_bins)
            
            ratio_red_mod = np.log10(hw_mean_stats_red_mod[0])
            ratio_blue_mod = np.log10(hw_mean_stats_blue_mod[0])

            ratio_red_mod_arr.append(ratio_red_mod)
            ratio_blue_mod_arr.append(ratio_blue_mod)

        ratio_red_mod_max = np.nanmax(ratio_red_mod_arr, axis=0)
        ratio_red_mod_min = np.nanmin(ratio_red_mod_arr, axis=0)
        ratio_blue_mod_max = np.nanmax(ratio_blue_mod_arr, axis=0)
        ratio_blue_mod_min = np.nanmin(ratio_blue_mod_arr, axis=0)


        ## Best-fit
        # nsat_red_total = np.sum(nsat_red_bf)
        # nsat_blue_total = np.sum(nsat_blue_bf)
        nsat_red_bf, vdisp_red_bf, mstar_red_bf = best_fit[1]['vel_disp']['red_nsat'],\
            best_fit[1]['vel_disp']['red_sigma'], best_fit[1]['vel_disp']['red_cen_mstar']
        nsat_blue_bf, vdisp_blue_bf, mstar_blue_bf = best_fit[1]['vel_disp']['blue_nsat'],\
            best_fit[1]['vel_disp']['blue_sigma'], best_fit[1]['vel_disp']['blue_cen_mstar']

        nsat_vdisp_product_red = np.array(nsat_red_bf) * (np.array(vdisp_red_bf)**2)
        nsat_vdisp_product_blue = np.array(nsat_blue_bf) * (np.array(vdisp_blue_bf)**2)

        nsat_red = bs(mstar_red_bf, nsat_red_bf, 'sum', 
            bins=red_stellar_mass_bins)
        nsat_blue = bs(mstar_blue_bf, nsat_blue_bf, 'sum',
            bins=blue_stellar_mass_bins)

        sw_mean_stats_red = bs(mstar_red_bf, nsat_vdisp_product_red,
            statistic=sw_func_red, bins=red_stellar_mass_bins)
        sw_mean_stats_red_bf = sw_mean_stats_red[0]/nsat_red[0]

        sw_mean_stats_blue = bs(mstar_blue_bf, nsat_vdisp_product_blue,
            statistic=sw_func_blue, bins=blue_stellar_mass_bins)
        sw_mean_stats_blue_bf = sw_mean_stats_blue[0]/nsat_blue[0]

        hw_mean_stats_red = bs(mstar_red_bf, np.array(vdisp_red_bf)**2,
            statistic=hw_func_red, bins=red_stellar_mass_bins)

        hw_mean_stats_blue = bs(mstar_blue_bf, np.array(vdisp_blue_bf)**2,
            statistic=hw_func_blue, bins=blue_stellar_mass_bins)

        ## Data
        # nsat_red_total = np.sum(nsat_red_data)
        # nsat_blue_total = np.sum(nsat_blue_data)
        nsat_red_data, vdisp_red_data, mstar_red_data = data_experiments['vel_disp']['red_nsat'],\
            data_experiments['vel_disp']['red_sigma'], data_experiments['vel_disp']['red_cen_mstar']
        nsat_blue_data, vdisp_blue_data, mstar_blue_data = data_experiments['vel_disp']['blue_nsat'],\
            data_experiments['vel_disp']['blue_sigma'], data_experiments['vel_disp']['blue_cen_mstar']

        nsat_vdisp_product_red_data = np.array(nsat_red_data) * (np.array(vdisp_red_data)**2)
        nsat_vdisp_product_blue_data = np.array(nsat_blue_data) * (np.array(vdisp_blue_data)**2)

        nsat_red = bs(mstar_red_data, nsat_red_data, 'sum', 
            bins=red_stellar_mass_bins)
        nsat_blue = bs(mstar_blue_data, nsat_blue_data, 'sum',
            bins=blue_stellar_mass_bins)

        sw_mean_stats_red = bs(mstar_red_data, nsat_vdisp_product_red_data,
            statistic=sw_func_red, bins=red_stellar_mass_bins)
        sw_mean_stats_red_data = sw_mean_stats_red[0]/nsat_red[0]

        sw_mean_stats_blue = bs(mstar_blue_data, nsat_vdisp_product_blue_data,
            statistic=sw_func_blue, bins=blue_stellar_mass_bins)
        sw_mean_stats_blue_data = sw_mean_stats_blue[0]/nsat_blue[0]

        hw_mean_stats_red_data = bs(mstar_red_data, np.array(vdisp_red_data)**2,
            statistic=hw_func_red, bins=red_stellar_mass_bins)

        hw_mean_stats_blue_data = bs(mstar_blue_data, np.array(vdisp_blue_data)**2,
            statistic=hw_func_blue, bins=blue_stellar_mass_bins)

        ## Centers the same for data, models and best-fit since red and blue bins 
        ## are the same for all 3 cases
        centers_red = 0.5 * (sw_mean_stats_red[1][1:] + \
            sw_mean_stats_red[1][:-1])
        centers_blue = 0.5 * (sw_mean_stats_blue[1][1:] + \
            sw_mean_stats_blue[1][:-1])

        fig1 = plt.figure(figsize=(11, 9))    
        # const = 1/(0.9*(2/3))
        # bfr, = plt.plot(centers_red, const*np.log10(sw_mean_stats_red[0]/hw_mean_stats_red[0]), c='maroon', lw=3, zorder = 10)
        # bfb, = plt.plot(centers_blue, const*np.log10(sw_mean_stats_blue[0]/hw_mean_stats_blue[0]), c='darkblue', lw=3, zorder=10)

        ## Ratio of satellite/host weighted
        # bfr, = plt.plot(centers_red, 
        #     np.log10(sw_mean_stats_red_bf/hw_mean_stats_red[0]), c='maroon', lw=3, 
        #     zorder = 10)
        # bfb, = plt.plot(centers_blue, 
        #     np.log10(sw_mean_stats_blue_bf/hw_mean_stats_blue[0]), c='darkblue', 
        #     lw=3, zorder=10)

        # dr = plt.scatter(centers_red, 
        #     np.log10(sw_mean_stats_red_data/hw_mean_stats_red_data[0]), 
        #     c='indianred', s=300, marker='p', zorder = 20)
        # db = plt.scatter(centers_blue, 
        #     np.log10(sw_mean_stats_blue_data/hw_mean_stats_blue_data[0]), 
        #     c='royalblue', s=300, marker='p', zorder=20)

        # mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
        #     y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
        # mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
        #     y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)

        # ## Satellite weighted
        # bfr, = plt.plot(centers_red, np.log10(sw_mean_stats_red_bf), c='maroon', lw=3, zorder = 10)
        # bfb, = plt.plot(centers_blue, np.log10(sw_mean_stats_blue_bf), c='darkblue', lw=3, zorder=10)

        # dr = plt.scatter(centers_red, 
        #     np.log10(sw_mean_stats_red_data), 
        #     c='indianred', s=300, marker='p', zorder = 20)
        # db = plt.scatter(centers_blue, 
        #     np.log10(sw_mean_stats_blue_data), 
        #     c='royalblue', s=300, marker='p', zorder=20)

        # mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
        #     y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
        # mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
        #     y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)

        ## Host weighted
        bfr, = plt.plot(centers_red, np.log10(hw_mean_stats_red[0]), c='maroon', lw=3, zorder = 10)
        bfb, = plt.plot(centers_blue, np.log10(hw_mean_stats_blue[0]), c='darkblue', lw=3, zorder=10)

        dr = plt.scatter(centers_red, 
            np.log10(hw_mean_stats_red_data[0]), 
            c='indianred', s=300, marker='p', zorder = 20)
        db = plt.scatter(centers_blue, 
            np.log10(hw_mean_stats_blue_data[0]), 
            c='royalblue', s=300, marker='p', zorder=20)

        mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
            y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
            y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)


        mr = plt.fill_between(x=centers_red, y1=ratio_red_mod_max, 
            y2=ratio_red_mod_min, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=centers_blue, y1=ratio_blue_mod_max, 
            y2=ratio_blue_mod_min, color='cornflowerblue',alpha=0.4)
        
        plt.legend([(bfr, bfb), (dr, db), (mr, mb)], 
            ['Best-fit', 'Data', 'Models'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, markerscale=1.5, 
            loc='best')

        plt.xlabel(r'\boldmath$\log_{10}\ M_{\star , cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=30)
        # plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{sw}^2 / \sigma_{hw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)
        # plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{sw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)
        plt.ylabel(r'\boldmath$\log_{10}\ (\sigma_{hw}^2) \left[\mathrm{km/s} \right]$', fontsize=30)


        if settings.quenching == 'halo':
            plt.title('Host weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
            # plt.title('Satellite weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
            # plt.title('Ratio of satellite-host weighted velocity dispersion in halo quenching model (excluding singletons)', fontsize=25)
        elif settings.quenching == 'hybrid':
            plt.title('Host weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
            # plt.title('Satellite weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
            # plt.title('Ratio of satellite-host weighted velocity dispersion in hybrid quenching model (excluding singletons)', fontsize=25)
        plt.show()

    def plot_vdf(self, models, data, data_experiments, best_fit):
        
        settings = self.settings

        fig1= plt.figure(figsize=(10,10))

        data_vdf_dict = data_experiments['vdf']
        data_vdf_error = data[5]['vdf']
        dr = plt.errorbar(np.log10(data_vdf_dict['x_vdf']),data_vdf_dict['phi_vdf'][0],
            yerr=data_vdf_error['std_phi_red'],
            color='darkred',fmt='^',ecolor='darkred',markersize=12,capsize=10,
            capthick=1.0,zorder=10)
        db = plt.errorbar(np.log10(data_vdf_dict['x_vdf']),data_vdf_dict['phi_vdf'][1],
            yerr=data_vdf_error['std_phi_blue'],
            color='darkblue',fmt='^',ecolor='darkblue',markersize=12,capsize=10,
            capthick=1.0,zorder=10)

        bf_vdf_dict = best_fit[1]['vdf']
        # Best-fit
        # Need a comma after 'bfr' and 'bfb' to solve this:
        #   AttributeError: 'NoneType' object has no attribute 'create_artists'
        bfr, = plt.plot(np.log10(bf_vdf_dict['x_vdf']), bf_vdf_dict['phi_vdf'][0], color='maroon', ls='-', lw=4, 
            zorder=10)
        bfb, = plt.plot(np.log10(bf_vdf_dict['x_vdf']), bf_vdf_dict['phi_vdf'][1], color='cornflowerblue', ls='-', lw=4, 
            zorder=10)

        mod_vdf_red_arr = []
        mod_vdf_blue_arr = []
        for idx in range(len(models)):
            mod_vdf_dict = models[idx][1]['vdf']
            mod_vdf_red_chunk = mod_vdf_dict['phi_red']
            for idx in range(len(mod_vdf_red_chunk)):
                mod_vdf_red_arr.append(mod_vdf_red_chunk[idx])
            mod_vdf_blue_chunk = mod_vdf_dict['phi_blue']
            for idx in range(len(mod_vdf_blue_chunk)):
                ## Converting -inf to nan due to log(counts=0) for some blue bins
                mod_vdf_blue_chunk[idx][mod_vdf_blue_chunk[idx] == -np.inf] = np.nan
                mod_vdf_blue_arr.append(mod_vdf_blue_chunk[idx])

        red_std_max = np.nanmax(mod_vdf_red_arr, axis=0)
        red_std_min = np.nanmin(mod_vdf_red_arr, axis=0)
        blue_std_max = np.nanmax(mod_vdf_blue_arr, axis=0)
        blue_std_min = np.nanmin(mod_vdf_blue_arr, axis=0)

        mr = plt.fill_between(x=np.log10(bf_vdf_dict['x_vdf']), y1=red_std_min, 
            y2=red_std_max, color='lightcoral',alpha=0.4)
        mb = plt.fill_between(x=np.log10(bf_vdf_dict['x_vdf']), y1=blue_std_min, 
            y2=blue_std_max, color='cornflowerblue',alpha=0.4)

        # plt.ylim(-4,-1)
        if settings.level == 'group':
            plt.xlabel(r'\boldmath$\log_{10}\ \sigma_{group} [kms^{-1}]$', fontsize=30)
        elif settings.level == 'halo':
            plt.xlabel(r'\boldmath$\log_{10}\ \sigma_{halo} [kms^{-1}]$', fontsize=30)
        plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=30)

        plt.legend([(dr, db), (mr, mb), (bfr, bfb)], ['Data', 'Models', 'Best-fit'],
            handler_map={tuple: HandlerTuple(ndivide=3, pad=0.3)}, loc='best')

        if settings.quenching == 'hybrid':
            plt.title('Hybrid quenching model | ECO')
        elif settings.quenching == 'halo':
            plt.title('Halo quenching model | ECO')
        plt.show()

    def Plot_Core(self, data, models, best_fit):
        self.plot_total_mf(models, data, best_fit)

        self.plot_fblue(models, data, best_fit)

        self.plot_colour_mf(models, data, best_fit)

        self.plot_xmhm(models, data, best_fit)

        self.plot_colour_xmhm(models, data, best_fit)

        self.plot_colour_hmxm(models, data, best_fit)

        self.plot_red_fraction_cen(models, best_fit)

        self.plot_red_fraction_sat(models, best_fit)

        self.plot_zumand_fig4(models, data, best_fit)

    def Plot_Experiments(self, data, data_experiments, models, best_fit):
        self.plot_mean_grpcen_vs_sigma(models, data, data_experiments, best_fit)

        self.plot_mean_sigma_vs_grpcen(models, data, data_experiments, best_fit)

        # self.plot_sigma_host_halo_mass_vishnu(models, best_fit)

        # self.plot_N_host_halo_mass_vishnu(models, best_fit)

        # self.plot_mean_grpcen_vs_N(models, data, data_experiments, best_fit)

        # self.plot_mean_N_vs_grpcen(models, data, data_experiments, best_fit)

        # self.plot_satellite_weighted_sigma(models, data_experiments, best_fit)

        # self.plot_vdf(models, data, data_experiments, best_fit)