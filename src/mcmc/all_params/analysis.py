"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from scipy.stats import binned_statistic as bs
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from itertools import repeat
import numpy as np
import pandas as pd
import globals
import time

class Analysis():

    def __init__(self, preprocess) -> None:
        self.catl = None
        self.total_data = None
        self.f_blue = None
        self.phi_red_data = None
        self.phi_blue_data = None
        self.mean_mstar_red_data = None
        self.mean_mstar_blue_data = None
        self.model_init = None
        self.best_fit_core = None
        self.best_fit_experimentals = None
        self.dof = None
        self.error_data = None
        self.mocks_stdevs = None
        self.result = None
        self.preprocess = preprocess
        self.settings = preprocess.settings
        self.vdisp_red_data = None
        self.vdisp_blue_data = None

    def assign_colour_label_data(self, catl):
        """
        Assign colour label to data

        Parameters
        ----------
        catl: pandas.DataFrame 
            Data catalog

        Returns
        ---------
        catl: pandas.DataFrame
            Data catalog with colour label assigned as new column
        """

        logmstar_arr = catl.logmstar.values
        u_r_arr = catl.modelu_rcorr.values

        colour_label_arr = np.empty(len(catl), dtype='str')
        for idx, value in enumerate(logmstar_arr):

            # Divisions taken from Moffett et al. 2015 equation 1
            if value <= 9.1:
                if u_r_arr[idx] > 1.457:
                    colour_label = 'R'
                else:
                    colour_label = 'B'

            if value > 9.1 and value < 10.1:
                divider = 0.24 * value - 0.7
                if u_r_arr[idx] > divider:
                    colour_label = 'R'
                else:
                    colour_label = 'B'

            if value >= 10.1:
                if u_r_arr[idx] > 1.7:
                    colour_label = 'R'
                else:
                    colour_label = 'B'
                
            colour_label_arr[idx] = colour_label
        
        catl['colour_label'] = colour_label_arr

        return catl

    def measure_all_smf(self, table, volume, data_bool, randint_logmstar=None):
        """
        Calculates differential stellar mass function for all, red and blue galaxies
        from mock/data

        Parameters
        ----------
        table: pandas.DataFrame
            Dataframe of either mock or data 
        volume: float
            Volume of simulation/survey
        data_bool: boolean
            Data or mock
        randint_logmstar (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

        Returns
        ---------
        3 multidimensional arrays of [stellar mass, phi, total error in SMF and 
        counts per bin] for all, red and blue galaxies
        """

        colour_col = 'colour_label'

        if data_bool:
            logmstar_col = 'logmstar'
            max_total, phi_total, err_total, bins_total, counts_total = \
                self.diff_smf(table[logmstar_col], volume, False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                self.diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
                volume, False, 'R')
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                self.diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
                volume, False, 'B')
            return [max_total, phi_total, err_total, counts_total] , \
                [max_red, phi_red, err_red, counts_red] , \
                    [max_blue, phi_blue, err_blue, counts_blue]
        else:
            if randint_logmstar != 1:
                logmstar_col = '{0}'.format(randint_logmstar)
            elif randint_logmstar == 1:
                logmstar_col = 'behroozi_bf'
            else:
                logmstar_col = 'stellar_mass'
            ## Changed to 10**X because Behroozi mocks now have M* values in log
            max_total, phi_total, err_total, bins_total, counts_total = \
                self.diff_smf(10**(table[logmstar_col]), volume, True)
            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
            #     volume, True, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
            #     volume, True, 'B')
            return [max_total, phi_total, err_total, counts_total]

    def diff_smf(self, mstar_arr, volume, h1_bool, colour_flag=False):
        """
        Calculates differential stellar mass function in units of h=1.0

        Parameters
        ----------
        mstar_arr: numpy array
            Array of stellar masses

        volume: float
            Volume of survey or simulation

        h1_bool: boolean
            True if units of masses are h=1, False if units of masses are not h=1
        
        colour_flag (optional): boolean
            'R' if galaxy masses correspond to red galaxies & 'B' if galaxy masses
            correspond to blue galaxies. Defaults to False.

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
        
        counts: array
            Array of number of things in each bin
        """
        settings = self.settings

        if not h1_bool:
            # changing from h=0.7 to h=1 assuming h^-2 dependence
            logmstar_arr = np.log10((10**mstar_arr) / 2.041)
        else:
            logmstar_arr = np.log10(mstar_arr)

        if settings.survey == 'eco' or settings.survey == 'resolvea':
            bin_min = np.round(np.log10((10**8.9) / 2.041), 1)

            if settings.survey == 'eco' and colour_flag == 'R':
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 6
                
            elif settings.survey == 'eco' and colour_flag == 'B':
                bin_max = np.round(np.log10((10**11) / 2.041), 1)
                bin_num = 6

            elif settings.survey == 'resolvea':
                # different to avoid nan in inverse corr mat
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 7

            else:
                # For eco total
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 5

            bins = np.linspace(bin_min, bin_max, bin_num)

        elif settings.survey == 'resolveb':
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

    def blue_frac_helper(self, arr):
        """Helper function for blue_frac() that calculates the fraction of blue 
        galaxies

        Args:
            arr (numpy array): Array of 'R' and 'B' characters depending on whether
            galaxy is red or blue

        Returns:
            numpy array: Array of floats representing fractions of blue galaxies in 
            each bin
        """
        total_num = len(arr)
        blue_counter = list(arr).count('B')
        return blue_counter/total_num

    def blue_frac(self, catl, h1_bool, data_bool, randint_logmstar=None):
        """
        Calculates blue fraction in bins of stellar mass (which are converted to h=1)

        Parameters
        ----------
        catl: pandas Dataframe 
            Data catalog

        h1_bool: boolean
            True if units of masses are h=1, False if units of masses are not h=1
        
        data_bool: boolean
            True if data, False if mocks
        
        randint_logmstar (optional): int
            Mock number in case many Behroozi mocks were used. Defaults to None.

        Returns
        ---------
        maxis: array
            Array of x-axis mass values

        f_blue: array
            Array of y-axis blue fraction values
        """
        settings = self.settings
        # print("before type set")
        # print(type(randint_logmstar))

        # randint_logmstar = int(randint_logmstar)

        # print(type(randint_logmstar))
        # print("after type set")

        if data_bool:
            mstar_total_arr = catl.logmstar.values
            censat_col = 'g_galtype'
            mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values
        ## Mocks case different than data because of censat_col
        elif not data_bool and not h1_bool:
            mstar_total_arr = catl.logmstar.values
            censat_col = 'g_galtype'
            # censat_col = 'cs_flag'
            mstar_cen_arr = catl.logmstar.loc[catl[censat_col] == 1].values
            mstar_sat_arr = catl.logmstar.loc[catl[censat_col] == 0].values           
        elif isinstance(randint_logmstar, int) and randint_logmstar != 1:
            # print("mp_func brought me here")
            mstar_total_arr = catl['{0}'.format(randint_logmstar)].values
            censat_col = 'grp_censat_{0}'.format(randint_logmstar)
            mstar_cen_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 1].values
            mstar_sat_arr = catl['{0}'.format(randint_logmstar)].loc[catl[censat_col] == 0].values
        elif isinstance(randint_logmstar, int) and randint_logmstar == 1:
            mstar_total_arr = catl['behroozi_bf'].values
            censat_col = 'grp_censat_{0}'.format(randint_logmstar)
            mstar_cen_arr = catl['behroozi_bf'].loc[catl[censat_col] == 1].values
            mstar_sat_arr = catl['behroozi_bf'].loc[catl[censat_col] == 0].values

        # print("censat_col:{0}".format(censat_col))
        # print(type(randint_logmstar))
        colour_label_total_arr = catl.colour_label.values
        colour_label_cen_arr = catl.colour_label.loc[catl[censat_col] == 1].values
        colour_label_sat_arr = catl.colour_label.loc[catl[censat_col] == 0].values

        if not h1_bool:
            # changing from h=0.7 to h=1 assuming h^-2 dependence
            logmstar_total_arr = np.log10((10**mstar_total_arr) / 2.041)
            logmstar_cen_arr = np.log10((10**mstar_cen_arr) / 2.041)
            logmstar_sat_arr = np.log10((10**mstar_sat_arr) / 2.041)
        else:
            logmstar_total_arr = mstar_total_arr
            logmstar_cen_arr = mstar_cen_arr
            logmstar_sat_arr = mstar_sat_arr

        if settings.survey == 'eco' or settings.survey == 'resolvea':
            bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 5

            bins = np.linspace(bin_min, bin_max, bin_num)

        elif settings.survey == 'resolveb':
            bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
            bins = np.linspace(bin_min, bin_max, 7)

        result_total = bs(logmstar_total_arr, colour_label_total_arr, self.blue_frac_helper, bins=bins)
        result_cen = bs(logmstar_cen_arr, colour_label_cen_arr, self.blue_frac_helper, bins=bins)
        result_sat = bs(logmstar_sat_arr, colour_label_sat_arr, self.blue_frac_helper, bins=bins)
        edges = result_total[1]
        dm = edges[1] - edges[0]  # Bin width
        maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
        f_blue_total = result_total[0]

        if settings.level == 'group':
            f_blue_cen = result_cen[0]
            f_blue_sat = result_sat[0]
        elif settings.level == 'halo':
            f_blue_cen = 0
            f_blue_sat = 0
 
        return maxis, f_blue_total, f_blue_cen, f_blue_sat

    def get_colour_smf_from_fblue(self, df, frac_arr, bin_centers, volume, h1_bool, 
        randint_logmstar=None):
        """Reconstruct red and blue SMFs from blue fraction measurement

        Args:
            df (pandas.DataFrame): Data/Mock
            frac_arr (array): Array of blue fraction values
            bin_centers (array): Array of x-axis stellar mass bin center values
            volume (float): Volume of data/mock
            h1_bool (boolean): True if masses in h=1.0, False if not in h=1.0
            randint_logmstar (int, optional): Mock number in the case where many
            Behroozi mocks were used. Defaults to None.

        Returns:
            phi_red (array): Array of phi values for red galaxies
            phi_blue (array): Array of phi values for blue galaxies
        """
        if h1_bool and randint_logmstar != 1:
            logmstar_arr = df['{0}'.format(randint_logmstar)].values
        elif h1_bool and randint_logmstar == 1:
            logmstar_arr = df['behroozi_bf'].values
        if not h1_bool:
            mstar_arr = df.logmstar.values
            logmstar_arr = np.log10((10**mstar_arr) / 2.041)

        bin_width = bin_centers[1] - bin_centers[0]
        bin_edges = bin_centers - (0.5 * bin_width)
        # Done to include the right side of the last bin
        bin_edges = np.insert(bin_edges, len(bin_edges), bin_edges[-1]+bin_width)
        counts, edg = np.histogram(logmstar_arr, bins=bin_edges)  
        
        counts_blue = frac_arr * counts
        counts_red = (1-frac_arr) * counts

        # Normalized by volume and bin width
        phi_red = counts_red / (volume * bin_width)  # not a log quantity
        phi_red = np.log10(phi_red)

        phi_blue = counts_blue / (volume * bin_width)  # not a log quantity
        phi_blue = np.log10(phi_blue)

        ## Check to make sure that the reconstruced mass functions are the same as 
        ## those from diff_smf(). They aren't exactly but they match up if the 
        ## difference in binning is corrected.
        # fig1 = plt.figure()
        # plt.plot(bin_centers, phi_red, 'r+', ls='--', label='reconstructed')
        # plt.plot(bin_centers, phi_blue, 'b+', ls='--', label='reconstructed')
        # plt.plot(red_data[0], red_data[1], 'r+', ls='-', label='measured')
        # plt.plot(blue_data[0], blue_data[1], 'b+', ls='-', label='measured')
        # plt.xlabel(r'\boldmath$\log_{10}\ M_\star \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$', fontsize=25)
        # plt.ylabel(r'\boldmath$\Phi \left[\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{h}^{3} \right]$', fontsize=25)
        # plt.legend()
        # plt.show()

        return phi_red, phi_blue

    def halocat_init(self, halo_catalog, z_median):
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

    def hybrid_quenching_model(self, theta, gals_df, mock, randint=None):
        """
        Apply hybrid quenching model from Zu and Mandelbaum 2015

        Parameters
        ----------
        theta: numpy array
            Array of quenching model parameter values
        gals_df: pandas dataframe
            Mock catalog
        mock: string
            'vishnu' or 'nonvishnu' depending on what mock it is
        randint (optional): int
            Mock number in the case where many Behroozi mocks were used.
            Defaults to None.

        Returns
        ---------
        f_red_cen: array
            Array of central red fractions
        f_red_sat: array
            Array of satellite red fractions
        """

        # parameter values from Table 1 of Zu and Mandelbaum 2015 "prior case"
        Mstar_q = theta[0] # Msun/h
        Mh_q = theta[1] # Msun/h
        mu = theta[2]
        nu = theta[3]

        cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = self.get_host_halo_mock(gals_df, \
            mock)
        cen_stellar_mass_arr, sat_stellar_mass_arr = self.get_stellar_mock(gals_df, mock, \
            randint)

        f_red_cen = 1 - np.exp(-((cen_stellar_mass_arr/(10**Mstar_q))**mu))

        g_Mstar = np.exp(-((sat_stellar_mass_arr/(10**Mstar_q))**mu))
        h_Mh = np.exp(-((sat_hosthalo_mass_arr/(10**Mh_q))**nu))
        f_red_sat = 1 - (g_Mstar * h_Mh)

        return f_red_cen, f_red_sat

    def halo_quenching_model(self, theta, gals_df, mock):
        """
        Apply halo quenching model from Zu and Mandelbaum 2015

        Parameters
        ----------
        gals_df: pandas dataframe
            Mock catalog

        Returns
        ---------
        f_red_cen: array
            Array of central red fractions
        f_red_sat: array
            Array of satellite red fractions
        """

        # parameter values from Table 1 of Zu and Mandelbaum 2015 "prior case"
        Mh_qc = theta[0] # Msun/h 
        Mh_qs = theta[1] # Msun/h
        mu_c = theta[2]
        mu_s = theta[3]

        cen_hosthalo_mass_arr, sat_hosthalo_mass_arr = self.get_host_halo_mock(gals_df, \
            mock)

        f_red_cen = 1 - np.exp(-((cen_hosthalo_mass_arr/(10**Mh_qc))**mu_c))
        f_red_sat = 1 - np.exp(-((sat_hosthalo_mass_arr/(10**Mh_qs))**mu_s))

        return f_red_cen, f_red_sat

    def get_host_halo_mock(self, df, mock):
        """
        Get host halo mass from mock catalog

        Parameters
        ----------
        df: pandas dataframe
            Mock catalog
        mock: string
            'vishnu' or 'nonvishnu' depending on what mock it is

        Returns
        ---------
        cen_halos: array
            Array of central host halo masses
        sat_halos: array
            Array of satellite host halo masses
        """
        if mock == 'vishnu':
            cen_halos = df.halo_mvir[df.cs_flag == 1].reset_index(drop=True)
            sat_halos = df.halo_mvir_host_halo[df.cs_flag == 0].reset_index(drop=True)
        else:
            # Loghalom in the mock catalogs is actually host halo mass i.e. 
            # For satellites, the loghalom value will be the value of the central's
            # loghalom in that halo group and the haloids for the satellites are the 
            # haloid of the central 
            cen_halos = 10**(df.loghalom[df.cs_flag == 1]).reset_index(drop=True)
            sat_halos = 10**(df.loghalom[df.cs_flag == 0]).reset_index(drop=True)

        cen_halos = np.array(cen_halos)
        sat_halos = np.array(sat_halos)

        return cen_halos, sat_halos

    def get_stellar_mock(self, df, mock, randint=None):
        """
        Get stellar mass from mock catalog

        Parameters
        ----------
        df: pandas dataframe
            Mock catalog
        mock: string
            'Vishnu' or 'nonVishnu' depending on what mock it is
        randint (optional): int
            Mock number in the case where many Behroozi mocks were used.
            Defaults to None.

        Returns
        ---------
        cen_gals: array
            Array of central stellar masses
        sat_gals: array
            Array of satellite stellar masses
        """

        if mock == 'vishnu' and randint != 1:
            cen_gals = 10**(df['{0}'.format(randint)][df.cs_flag == 1]).\
                reset_index(drop=True)
            sat_gals = 10**(df['{0}'.format(randint)][df.cs_flag == 0]).\
                reset_index(drop=True)

        elif mock == 'vishnu' and randint == 1:
            cen_gals = 10**(df['behroozi_bf'][df.cs_flag == 1]).\
                reset_index(drop=True)
            sat_gals = 10**(df['behroozi_bf'][df.cs_flag == 0]).\
                reset_index(drop=True)

        # elif mock == 'vishnu':
        #     cen_gals = 10**(df.stellar_mass[df.cs_flag == 1]).reset_index(drop=True)
        #     sat_gals = 10**(df.stellar_mass[df.cs_flag == 0]).reset_index(drop=True)
        
        else:
            cen_gals = []
            sat_gals = []
            for idx,value in enumerate(df.cs_flag):
                if value == 1:
                    cen_gals.append((10**(df.logmstar.values[idx]))/2.041)
                elif value == 0:
                    sat_gals.append((10**(df.logmstar.values[idx]))/2.041)

        cen_gals = np.array(cen_gals)
        sat_gals = np.array(sat_gals)

        return cen_gals, sat_gals

    def assign_colour_label_mock(self, f_red_cen, f_red_sat, df, drop_fred=False):
        """
        Assign colour label to mock catalog

        Parameters
        ----------
        f_red_cen: array
            Array of central red fractions
        f_red_sat: array
            Array of satellite red fractions
        df: pandas Dataframe
            Mock catalog
        drop_fred (optional): boolean
            Whether or not to keep red fraction column after colour has been
            assigned. Defaults to False.

        Returns
        ---------
        df: pandas Dataframe
            Dataframe with colour label and random number assigned as 
            new columns
        """

        # Saving labels
        color_label_arr = [[] for x in range(len(df))]
        rng_arr = [[] for x in range(len(df))]
        # Adding columns for f_red to df
        df.loc[:, 'f_red'] = np.zeros(len(df))
        df.loc[df['cs_flag'] == 1, 'f_red'] = f_red_cen
        df.loc[df['cs_flag'] == 0, 'f_red'] = f_red_sat
        # Converting to array
        f_red_arr = df['f_red'].values
        # Looping over galaxies
        for ii, cs_ii in enumerate(df['cs_flag']):
            # Draw a random number
            rng = np.random.uniform()
            # Comparing against f_red
            if (rng >= f_red_arr[ii]):
                color_label = 'B'
            else:
                color_label = 'R'
            # Saving to list
            color_label_arr[ii] = color_label
            rng_arr[ii] = rng
        
        ## Assigning to DataFrame
        df.loc[:, 'colour_label'] = color_label_arr
        df.loc[:, 'rng'] = rng_arr
        # Dropping 'f_red` column
        if drop_fred:
            df.drop('f_red', axis=1, inplace=True)

        return df

    def get_centrals_mock(self, gals_df, randint=None):
        """
        Get centrals from mock catalog

        Parameters
        ----------
        gals_df: pandas dataframe
            Mock catalog

        randint (optional): int
            Mock number in the case where many Behroozi mocks were used.
            Defaults to None.

        Returns
        ---------
        cen_gals: array
            Array of central galaxy masses

        cen_halos: array
            Array of central halo masses

        cen_gals_red: array
            Array of red central galaxy masses

        cen_halos_red: array
            Array of red central halo masses

        cen_gals_blue: array
            Array of blue central galaxy masses

        cen_halos_blue: array
            Array of blue central halo masses

        f_red_cen_gals_red: array
            Array of red fractions for red central galaxies

        f_red_cen_gals_blue: array
            Array of red fractions for blue central galaxies
        """
        cen_gals = []
        cen_halos = []
        cen_gals_red = []
        cen_halos_red = []
        cen_gals_blue = []
        cen_halos_blue = []
        f_red_cen_gals_red = []
        f_red_cen_gals_blue = []

        if randint != 1:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 1:
                    cen_gals.append(gals_df['{0}'.format(randint)][idx])
                    cen_halos.append(gals_df['halo_mvir'][idx])
                    if gals_df['colour_label'][idx] == 'R':
                        cen_gals_red.append(gals_df['{0}'.format(randint)][idx])
                        cen_halos_red.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        cen_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                        cen_halos_blue.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_blue.append(gals_df['f_red'][idx])
        elif randint == 1:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 1:
                    cen_gals.append(gals_df['behroozi_bf'][idx])
                    cen_halos.append(gals_df['halo_mvir'][idx])
                    if gals_df['colour_label'][idx] == 'R':
                        cen_gals_red.append(gals_df['behroozi_bf'][idx])
                        cen_halos_red.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        cen_gals_blue.append(gals_df['behroozi_bf'][idx])
                        cen_halos_blue.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_blue.append(gals_df['f_red'][idx])

        else:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 1:
                    cen_gals.append(gals_df['stellar_mass'][idx])
                    cen_halos.append(gals_df['halo_mvir'][idx])
                    if gals_df['colour_label'][idx] == 'R':
                        cen_gals_red.append(gals_df['stellar_mass'][idx])
                        cen_halos_red.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        cen_gals_blue.append(gals_df['stellar_mass'][idx])
                        cen_halos_blue.append(gals_df['halo_mvir'][idx])
                        f_red_cen_gals_blue.append(gals_df['f_red'][idx])

        cen_gals = np.array(cen_gals)
        cen_halos = np.log10(np.array(cen_halos))
        cen_gals_red = np.array(cen_gals_red)
        cen_halos_red = np.log10(np.array(cen_halos_red))
        cen_gals_blue = np.array(cen_gals_blue)
        cen_halos_blue = np.log10(np.array(cen_halos_blue))

        return cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
            cen_halos_blue, f_red_cen_gals_red, f_red_cen_gals_blue

    def get_satellites_mock(self, gals_df, randint=None):
        """
        Get satellites and their host halos from mock catalog

        Parameters
        ----------
        gals_df: pandas dataframe
            Mock catalog
            
        randint (optional): int
            Mock number in the case where many Behroozi mocks were used. 
            Defaults to None.

        Returns
        ---------
        sat_gals_red: array
            Array of red satellite galaxy masses

        sat_halos_red: array
            Array of red satellite host halo masses

        sat_gals_blue: array
            Array of blue satellite galaxy masses

        sat_halos_blue: array
            Array of blue satellite host halo masses

        f_red_sat_gals_red: array
            Array of red fractions for red satellite galaxies

        f_red_sat_gals_blue: array
            Array of red fractions for blue satellite galaxies

        """
        sat_gals_red = []
        sat_halos_red = []
        sat_gals_blue = []
        sat_halos_blue = []
        f_red_sat_gals_red = []
        f_red_sat_gals_blue = []

        if randint != 1:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 0:
                    if gals_df['colour_label'][idx] == 'R':
                        sat_gals_red.append(gals_df['{0}'.format(randint)][idx])
                        sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        sat_gals_blue.append(gals_df['{0}'.format(randint)][idx])
                        sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_blue.append(gals_df['f_red'][idx])
        elif randint == 1:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 0:
                    if gals_df['colour_label'][idx] == 'R':
                        sat_gals_red.append(gals_df['behroozi_bf'][idx])
                        sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        sat_gals_blue.append(gals_df['behroozi_bf'][idx])
                        sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_blue.append(gals_df['f_red'][idx])

        else:
            for idx,value in enumerate(gals_df['cs_flag']):
                if value == 0:
                    if gals_df['colour_label'][idx] == 'R':
                        sat_gals_red.append(gals_df['stellar_mass'][idx])
                        sat_halos_red.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_red.append(gals_df['f_red'][idx])
                    elif gals_df['colour_label'][idx] == 'B':
                        sat_gals_blue.append(gals_df['stellar_mass'][idx])
                        sat_halos_blue.append(gals_df['halo_mvir_host_halo'][idx])
                        f_red_sat_gals_blue.append(gals_df['f_red'][idx])


        sat_gals_red = np.array(sat_gals_red)
        sat_halos_red = np.log10(np.array(sat_halos_red))
        sat_gals_blue = np.array(sat_gals_blue)
        sat_halos_blue = np.log10(np.array(sat_halos_blue))

        return sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
            f_red_sat_gals_red, f_red_sat_gals_blue

    def get_best_fit_model(self, bf_params, experiments, best_fit_mocknum=None):
        settings = self.settings
        preprocess = self.preprocess

        best_fit_results = {}
        best_fit_experimentals = {}

        if best_fit_mocknum:
            cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
                'halo_mvir_host_halo', 'cz', \
                '{0}'.format(best_fit_mocknum), \
                'grp_censat_{0}'.format(best_fit_mocknum), \
                'groupid_{0}'.format(best_fit_mocknum)]

            gals_df = globals.gal_group_df_subset[cols_to_use]

            gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
                format(best_fit_mocknum),'groupid_{0}'.format(best_fit_mocknum)]).\
                reset_index(drop=True)

            gals_df[['grp_censat_{0}'.format(best_fit_mocknum), \
                'groupid_{0}'.format(best_fit_mocknum)]] = \
                gals_df[['grp_censat_{0}'.format(best_fit_mocknum),\
                'groupid_{0}'.format(best_fit_mocknum)]].astype(int)
        else:
            # gals_df = populate_mock(best_fit_params[:5], model_init)
            # gals_df = gals_df.loc[gals_df['stellar_mass'] >= 10**8.6].\
            #     reset_index(drop=True)
            # gals_df['cs_flag'] = np.where(gals_df['halo_hostid'] == \
            #     gals_df['halo_id'], 1, 0)

            # cols_to_use = ['halo_mvir', 'halo_mvir_host_halo', 'cs_flag', 
            #     'stellar_mass']
            # gals_df = gals_df[cols_to_use]
            # gals_df.stellar_mass = np.log10(gals_df.stellar_mass)

            randint_logmstar = 1
            cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
            'halo_mvir_host_halo', 'cz', 'cs_flag', \
            'behroozi_bf', \
            'grp_censat_{0}'.format(randint_logmstar), \
            'groupid_{0}'.format(randint_logmstar),\
            'cen_cz_{0}'.format(randint_logmstar)]

            gals_df = globals.gal_group_df_subset[cols_to_use]

            gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
            format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
            'cen_cz_{0}'.format(randint_logmstar)]).\
            reset_index(drop=True)

            gals_df[['grp_censat_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar)]] = \
                gals_df[['grp_censat_{0}'.format(randint_logmstar),\
                'groupid_{0}'.format(randint_logmstar)]].astype(int)
                        
            gals_df['behroozi_bf'] = np.log10(gals_df['behroozi_bf'])

        if settings.quenching == 'hybrid':
            f_red_cen, f_red_sat = self.hybrid_quenching_model(bf_params[5:], 
                gals_df, 'vishnu', randint_logmstar)
        elif settings.quenching == 'halo':
            f_red_cen, f_red_sat = self.halo_quenching_model(bf_params[5:], 
                gals_df, 'vishnu')      
        gals_df = self.assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)
        # v_sim = 130**3 
        v_sim = 890641.5172927063  ## cz: 3000-12000
        # v_sim = 165457.21308906242 ## cz: 3000-7000

        ## Observable #1 - Total SMF
        total_model = self.measure_all_smf(gals_df, v_sim , False, 
            randint_logmstar)    
        ## Observable #2 - Blue fraction
        f_blue = self.blue_frac(gals_df, True, False, randint_logmstar)

        ## Observable #3 - sigma-M* or M*-sigma
        if settings.stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = experiments.get_stacked_velocity_dispersion(
                gals_df, 'model', randint_logmstar)

            best_fit_experimentals["vel_disp"] = {'red_sigma':red_deltav,
                'red_cen_mstar':red_cen_mstar_sigma,
                'blue_sigma':blue_deltav, 'blue_cen_mstar':blue_cen_mstar_sigma}

        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = experiments.get_velocity_dispersion(
                    gals_df, 'model', randint_logmstar)

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            best_fit_experimentals["vel_disp"] = {'red_sigma':red_sigma,
                'red_cen_mstar':red_cen_mstar_sigma,
                'blue_sigma':blue_sigma, 'blue_cen_mstar':blue_cen_mstar_sigma}


        # mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
        #     statistic='mean', bins=np.linspace(-2,3,5))
        # mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
        #     statistic='mean', bins=np.linspace(-1,3,5))

        cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
            cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
                self.get_centrals_mock(gals_df, randint_logmstar)
        sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
            f_red_sat_red, f_red_sat_blue = \
                self.get_satellites_mock(gals_df, randint_logmstar)

        phi_red_model, phi_blue_model = \
            self.get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
            True, randint_logmstar)

        # red_sigma, red_cen_mstar_sigma, blue_sigma, \
        #     blue_cen_mstar_sigma, red_nsat, blue_nsat, red_host_halo_mass_vd, \
        #     blue_host_halo_mass_vd = \
        #     experiments.get_velocity_dispersion(gals_df, 'model', randint_logmstar)
        red_num, red_cen_mstar_richness, blue_num, \
            blue_cen_mstar_richness, red_host_halo_mass, \
            blue_host_halo_mass = \
            experiments.get_richness(gals_df, 'model', randint_logmstar)

        # v_sim = 890641.5172927063 
        if not settings.stacked_stat:
            x_vdf, phi_vdf, error, bins, counts = experiments.get_vdf(
                red_sigma, blue_sigma, v_sim)

            best_fit_experimentals["vdf"] = {'x_vdf':x_vdf,
                'phi_vdf':phi_vdf,
                'error':error, 'bins':bins,
                'counts':counts}

        best_fit_results["smf_total"] = {'max_total':total_model[0],
                                        'phi_total':total_model[1]}
        
        best_fit_results["f_blue"] = {'max_fblue':f_blue[0],
                                      'fblue_total':f_blue[1],
                                      'fblue_cen':f_blue[2],
                                      'fblue_sat':f_blue[3]}
        
        best_fit_results["phi_colour"] = {'phi_red':phi_red_model,
                                          'phi_blue':phi_blue_model}

        best_fit_results["centrals"] = {'gals':cen_gals, 'halos':cen_halos,
                                        'gals_red':cen_gals_red, 
                                        'halos_red':cen_halos_red,
                                        'gals_blue':cen_gals_blue,
                                        'halos_blue':cen_halos_blue}
        
        best_fit_results["satellites"] = {'gals_red':sat_gals_red, 
                                        'halos_red':sat_halos_red,
                                        'gals_blue':sat_gals_blue,
                                        'halos_blue':sat_halos_blue}
        
        best_fit_results["f_red"] = {'cen_red':f_red_cen_red,
                                    'cen_blue':f_red_cen_blue,
                                    'sat_red':f_red_sat_red,
                                    'sat_blue':f_red_sat_blue}


        # best_fit_experimentals["vel_disp"] = {'red_sigma':mean_mstar_red[1],
        #     'red_cen_mstar':mean_mstar_red[0],
        #     'blue_sigma':mean_mstar_blue[1], 'blue_cen_mstar':mean_mstar_blue[0]}
            # 'red_nsat':red_nsat, 'blue_nsat':blue_nsat, 
            # 'red_hosthalo':red_host_halo_mass_vd, 
            # 'blue_hosthalo':blue_host_halo_mass_vd}
        
        best_fit_experimentals["richness"] = {'red_num':red_num,
            'red_cen_mstar':red_cen_mstar_richness, 'blue_num':blue_num,
            'blue_cen_mstar':blue_cen_mstar_richness, 
            'red_hosthalo':red_host_halo_mass, 'blue_hosthalo':blue_host_halo_mass}

        # best_fit_experimentals["vdf"] = {'x_vdf':x_vdf,
        #     'phi_vdf':phi_vdf,
        #     'error':error, 'bins':bins,
        #     'counts':counts}

        return best_fit_results, best_fit_experimentals, gals_df

    def get_err_data(self, path, experiments):
        """
        Calculate error in data SMF from mocks

        Parameters
        ----------
        survey: string
            Name of survey
        path: string
            Path to mock catalogs

        Returns
        ---------
        err_colour: array
            Standard deviation from matrix of phi values and blue fractions values
            between all mocks and for all galaxies
        std_phi_red: array
            Standard deviation of phi values between all mocks for red galaxies
        std_phi_blue: array
            Standard deviation of phi values between all mocks for blue galaxies
        std_mean_cen_arr_red: array
            Standard deviation of observable number 3 (mean grp central stellar mass
            in bins of velocity dispersion) for red galaxies
        std_mean_cen_arr_blue: array
            Standard deviation of observable number 3 (mean grp central stellar mass
            in bins of velocity dispersion) for blue galaxies
        """
        settings = self.settings
        preprocess = self.preprocess

        mocks_experimentals = {}

        if settings.survey == 'eco':
            mock_name = 'ECO'
            num_mocks = 8
            min_cz = 3000
            max_cz = 7000
            mag_limit = -17.33
            mstar_limit = 8.9
            volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        elif settings.survey == 'resolvea':
            mock_name = 'A'
            num_mocks = 59
            min_cz = 4500
            max_cz = 7000
            mag_limit = -17.33
            mstar_limit = 8.9
            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3 
        elif settings.survey == 'resolveb':
            mock_name = 'B'
            num_mocks = 104
            min_cz = 4500
            max_cz = 7000
            mag_limit = -17
            mstar_limit = 8.7
            volume = 4709.8373  # Survey volume without buffer [Mpc/h]^3

        phi_total_arr = []
        phi_red_arr = []

        phi_blue_arr = []
        f_blue_arr = []
        f_blue_cen_arr = []
        f_blue_sat_arr = []

        mean_mstar_red_arr = []
        mean_mstar_blue_arr = []

        mean_sigma_red_arr = []
        mean_sigma_blue_arr = []

        red_cen_mstar_richness_arr = []
        red_num_arr = []
        blue_cen_mstar_richness_arr = []
        blue_num_arr = []

        x_vdf_arr = []
        red_phi_vdf_arr = []
        blue_phi_vdf_arr = []

        box_id_arr = np.linspace(5001,5008,8)
        for box in box_id_arr:
            box = int(box)
            temp_path = path + '{0}/{1}_m200b_catls/'.format(box, 
                mock_name) 
            for num in range(num_mocks):
                print('Box {0} : Mock {1}'.format(box, num))
                filename = temp_path + '{0}_cat_{1}_Planck_memb_cat.hdf5'.format(
                    mock_name, num)
                mock_pd = preprocess.read_mock_catl(filename) 
                mock_pd = preprocess.mock_add_grpcz(mock_pd, grpid_col='groupid', 
                    galtype_col='g_galtype', cen_cz_col='cz')
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer
                mock_pd = mock_pd.loc[(mock_pd.grpcz_new.values >= min_cz) & \
                    (mock_pd.grpcz_new.values <= max_cz) & (mock_pd.M_r.values <= mag_limit) &\
                    (mock_pd.logmstar.values >= mstar_limit)].reset_index(drop=True)

                ## Using best-fit found for new ECO data using result from chain 42
                ## i.e. hybrid quenching model
                Mstar_q = 10.11652049 # Msun/h**2
                Mh_q = 13.86684472 # Msun/h
                mu = 0.76086959
                nu = 0.04489465

                ## Using best-fit found for new ECO data using result from chain 43
                ## i.e. halo quenching model
                Mh_qc = 11.68499777 # Msun/h
                Mh_qs = 12.3832308 # Msun/h
                mu_c = 1.41969021
                mu_s = 0.46442463

                if settings.quenching == 'hybrid':
                    theta = [Mstar_q, Mh_q, mu, nu]
                    f_red_c, f_red_s = self.hybrid_quenching_model(theta, mock_pd, 
                        'nonvishnu')
                elif settings.quenching == 'halo':
                    theta = [Mh_qc, Mh_qs, mu_c, mu_s]
                    f_red_c, f_red_s = self.halo_quenching_model(theta, mock_pd, 
                        'nonvishnu')        
                mock_pd = self.assign_colour_label_mock(f_red_c, f_red_s, mock_pd)

                logmstar_arr = mock_pd.logmstar.values 


                ### Statistics for correlation matrix
                #Measure SMF of mock using diff_smf function
                mass_total, phi_total, err_total, bins_total, counts_total = \
                    self.diff_smf(logmstar_arr, volume, False)
                phi_total_arr.append(phi_total)

                #Measure blue fraction of galaxies
                mass, f_blue, f_blue_cen, f_blue_sat = self.blue_frac(mock_pd, 
                    False, False)
                f_blue_arr.append(f_blue)
                f_blue_cen_arr.append(f_blue_cen)
                f_blue_sat_arr.append(f_blue_sat)

                if settings.stacked_stat:
                    red_deltav, red_cen_mstar_sigma, blue_deltav, \
                        blue_cen_mstar_sigma = \
                        experiments.get_stacked_velocity_dispersion(
                        mock_pd, 'mock')

                    sigma_red = bs(red_cen_mstar_sigma, red_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    sigma_blue = bs( blue_cen_mstar_sigma, blue_deltav,
                        statistic='std', bins=np.linspace(8.6,11,5))
                    
                    sigma_red = np.log10(sigma_red[0])
                    sigma_blue = np.log10(sigma_blue[0])

                    mean_mstar_red_arr.append(sigma_red)
                    mean_mstar_blue_arr.append(sigma_blue)

                else:
                    red_sigma, red_cen_mstar_sigma, blue_sigma, \
                        blue_cen_mstar_sigma = \
                        experiments.get_velocity_dispersion(mock_pd, 'mock')

                    red_sigma = np.log10(red_sigma)
                    blue_sigma = np.log10(blue_sigma)

                    mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                        statistic='mean', bins=np.linspace(-2,3,5))
                    mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                        statistic='mean', bins=np.linspace(-1,3,5))

                    mean_mstar_red_arr.append(mean_mstar_red[0])
                    mean_mstar_blue_arr.append(mean_mstar_blue[0])

                ### Statistics for measuring std. dev. to plot error bars
                phi_red, phi_blue = \
                    self.get_colour_smf_from_fblue(mock_pd, f_blue, mass, volume, 
                    False)
                phi_red_arr.append(phi_red)
                phi_blue_arr.append(phi_blue)
                
                # red_sigma, red_cen_mstar_sigma, blue_sigma, \
                #     blue_cen_mstar_sigma, red_nsat, blue_nsat, \
                #     red_host_halo_mass, blue_host_halo_mass = \
                #     experiments.get_velocity_dispersion(mock_pd, 'mock')

                red_num, red_cen_mstar_richness, blue_num, \
                    blue_cen_mstar_richness, red_host_halo_mass, \
                    blue_host_halo_mass = \
                    experiments.get_richness(mock_pd, 'mock')
                
                # mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
                #     statistic='mean', bins=np.linspace(0,250,6))
                # mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
                #     statistic='mean', bins=np.linspace(0,250,6))

                red_cen_mstar_richness_arr.append(red_cen_mstar_richness)
                red_num_arr.append(red_num)
                blue_cen_mstar_richness_arr.append(blue_cen_mstar_richness)
                blue_num_arr.append(blue_num)

                if not settings.stacked_stat:
                    x_vdf, phi_vdf, error, bins, counts = experiments.\
                        get_vdf(red_sigma, blue_sigma, volume)

                    x_vdf_arr.append(x_vdf)
                    red_phi_vdf_arr.append(phi_vdf[0])

                    ## Converting -inf to nan due to log(counts=0) for some blue bins
                    phi_vdf[1][phi_vdf[1] == -np.inf] = np.nan
                    blue_phi_vdf_arr.append(phi_vdf[1])
                
        phi_total_arr = np.array(phi_total_arr)

        f_blue_arr = np.array(f_blue_arr)
        std_fblue = np.nanstd(f_blue_arr, axis=0)

        ### For calculating std. dev. for:
        ## Blue fraction split by group centrals and satellites
        f_blue_cen_arr = np.array(f_blue_cen_arr)
        f_blue_sat_arr = np.array(f_blue_sat_arr)

        std_fblue_cen = np.nanstd(f_blue_cen_arr, axis=0)
        std_fblue_sat = np.nanstd(f_blue_sat_arr, axis=0)

        ## Colour mass functions
        phi_red_arr = np.array(phi_red_arr)
        phi_blue_arr = np.array(phi_blue_arr)

        std_phi_red = np.std(phi_red_arr, axis=0)
        std_phi_blue = np.std(phi_blue_arr, axis=0)

        ## Average group/halo central mstar vs velocity dispersion
        mean_mstar_red_arr = np.array(mean_mstar_red_arr)
        mean_mstar_blue_arr = np.array(mean_mstar_blue_arr)

        std_mean_mstar_red_arr = np.nanstd(mean_mstar_red_arr, axis=0)
        std_mean_mstar_blue_arr = np.nanstd(mean_mstar_blue_arr, axis=0)

        ## Average velocity dispersion vs group/halo central mstar
        mean_sigma_red_arr = np.array(mean_sigma_red_arr)
        mean_sigma_blue_arr = np.array(mean_sigma_blue_arr)

        std_mean_sigma_red_arr = np.nanstd(mean_sigma_red_arr, axis=0)
        std_mean_sigma_blue_arr = np.nanstd(mean_sigma_blue_arr, axis=0)
        
        if not settings.stacked_stat:
            ## Velocity dispersion function
            red_phi_vdf_arr = np.array(red_phi_vdf_arr)
            blue_phi_vdf_arr = np.array(blue_phi_vdf_arr)

            std_phi_vdf_red_arr = np.nanstd(red_phi_vdf_arr, axis=0)
            std_phi_vdf_blue_arr = np.nanstd(blue_phi_vdf_arr, axis=0)

            mocks_experimentals["vdf"] = {'std_phi_red':std_phi_vdf_red_arr,
                'std_phi_blue':std_phi_vdf_blue_arr}

        mocks_experimentals["std_fblue"] = {'std_fblue': std_fblue,
                                            'std_fblue_cen':std_fblue_cen,
                                            'std_fblue_sat':std_fblue_sat}

        mocks_experimentals["std_phi_colour"] = {'std_phi_red':std_phi_red,
                                                'std_phi_blue':std_phi_blue}
        
        mocks_experimentals["vel_disp"] = {'std_mean_mstar_red':std_mean_mstar_red_arr,
                                            'std_mean_mstar_blue':std_mean_mstar_blue_arr,
                                            'std_mean_sigma_red':std_mean_sigma_red_arr,
                                            'std_mean_sigma_blue':std_mean_sigma_blue_arr}
        
        mocks_experimentals["richness"] = {'red_num':red_num_arr,
            'red_cen_mstar':red_cen_mstar_richness_arr, 'blue_num':blue_num_arr,
            'blue_cen_mstar':blue_cen_mstar_richness_arr}


        phi_total_0 = phi_total_arr[:,0]
        phi_total_1 = phi_total_arr[:,1]
        phi_total_2 = phi_total_arr[:,2]
        phi_total_3 = phi_total_arr[:,3]
        # phi_total_4 = phi_total_arr[:,4]
        # phi_total_5 = phi_total_arr[:,5]

        ## Plotting correlation matrix for fblue split by cen and sat
        f_blue_cen_0 = f_blue_cen_arr[:,0]
        f_blue_cen_1 = f_blue_cen_arr[:,1]
        f_blue_cen_2 = f_blue_cen_arr[:,2]
        f_blue_cen_3 = f_blue_cen_arr[:,3]
        # f_blue_cen_4 = f_blue_cen_arr[:,4]
        # f_blue_cen_5 = f_blue_cen_arr[:,5]

        f_blue_sat_0 = f_blue_sat_arr[:,0]
        f_blue_sat_1 = f_blue_sat_arr[:,1]
        f_blue_sat_2 = f_blue_sat_arr[:,2]
        f_blue_sat_3 = f_blue_sat_arr[:,3]
        # f_blue_sat_4 = f_blue_sat_arr[:,4]
        # f_blue_sat_5 = f_blue_sat_arr[:,5]

        mstar_red_cen_0 = mean_mstar_red_arr[:,0]
        mstar_red_cen_1 = mean_mstar_red_arr[:,1]
        mstar_red_cen_2 = mean_mstar_red_arr[:,2]
        mstar_red_cen_3 = mean_mstar_red_arr[:,3]

        mstar_blue_cen_0 = mean_mstar_blue_arr[:,0]
        mstar_blue_cen_1 = mean_mstar_blue_arr[:,1]
        mstar_blue_cen_2 = mean_mstar_blue_arr[:,2]
        mstar_blue_cen_3 = mean_mstar_blue_arr[:,3]

        combined_df = pd.DataFrame({
            'phi_tot_0':phi_total_0, 'phi_tot_1':phi_total_1, 
            'phi_tot_2':phi_total_2, 'phi_tot_3':phi_total_3,
            'f_blue_cen_0':f_blue_cen_0, 'f_blue_cen_1':f_blue_cen_1, 
            'f_blue_cen_2':f_blue_cen_2, 'f_blue_cen_3':f_blue_cen_3,
            'f_blue_sat_0':f_blue_sat_0, 'f_blue_sat_1':f_blue_sat_1, 
            'f_blue_sat_2':f_blue_sat_2, 'f_blue_sat_3':f_blue_sat_3,
            'mstar_red_cen_0':mstar_red_cen_0, 'mstar_red_cen_1':mstar_red_cen_1, 
            'mstar_red_cen_2':mstar_red_cen_2, 'mstar_red_cen_3':mstar_red_cen_3,
            'mstar_blue_cen_0':mstar_blue_cen_0, 'mstar_blue_cen_1':mstar_blue_cen_1, 
            'mstar_blue_cen_2':mstar_blue_cen_2, 'mstar_blue_cen_3':mstar_blue_cen_3})
        
        err_colour = np.sqrt(np.diag(combined_df.cov()))

        # import matplotlib.pyplot as plt
        # from matplotlib import rc
        # from matplotlib import cm

        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
        # rc('text', usetex=False)
        # rc('axes', linewidth=2)
        # rc('xtick.major', width=4, size=7)
        # rc('ytick.major', width=4, size=7)
        # rc('xtick.minor', width=2, size=7)
        # rc('ytick.minor', width=2, size=7)

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # cmap = cm.get_cmap('Spectral')
        # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
        # tick_marks = [i for i in range(len(combined_df.columns))]
        # names = [r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$', r'$\Phi_4$', 
        # r'$\Phi_5$', r'$\Phi_6$',
        # r'$cen_1$', r'$cen_2$', r'$cen_3$', r'$cen_4$', r'$cen_5$',
        # r'$cen_6$', r'$sat_1$', r'$sat_2$', r'$sat_3$', r'$sat_4$', r'$sat_5$',
        # r'$sat_6$']
        # plt.xticks(tick_marks, names, rotation='vertical')
        # plt.yticks(tick_marks, names)    
        # plt.gca().invert_yaxis() 
        # plt.gca().xaxis.tick_bottom()
        # plt.colorbar(cax)
        # plt.title(r'SMF and blue fraction of centrals and satellites | {0}'.format(settings.quenching))
        # plt.show()

        # ## Plotting correlation matrix for vdf
        # vdf_red_0 = red_phi_vdf_arr[:,0]
        # vdf_red_1 = red_phi_vdf_arr[:,1]
        # vdf_red_2 = red_phi_vdf_arr[:,2]
        # vdf_red_3 = red_phi_vdf_arr[:,3]
        # vdf_red_4 = red_phi_vdf_arr[:,4]
        # vdf_red_5 = red_phi_vdf_arr[:,5]

        # vdf_blue_0 = blue_phi_vdf_arr[:,0]
        # vdf_blue_1 = blue_phi_vdf_arr[:,1]
        # vdf_blue_2 = blue_phi_vdf_arr[:,2]
        # vdf_blue_3 = blue_phi_vdf_arr[:,3]
        # vdf_blue_4 = blue_phi_vdf_arr[:,4]
        # vdf_blue_5 = blue_phi_vdf_arr[:,5]

        # combined_df = pd.DataFrame({
        #     'vdf_red_0':vdf_red_0, 
        #     'vdf_red_1':vdf_red_1, 
        #     'vdf_red_2':vdf_red_2, 
        #     'vdf_red_3':vdf_red_3, 
        #     'vdf_red_4':vdf_red_4, 
        #     'vdf_red_5':vdf_red_5,
        #     'vdf_blue_0':vdf_blue_0, 
        #     'vdf_blue_1':vdf_blue_1, 
        #     'vdf_blue_2':vdf_blue_2, 
        #     'vdf_blue_3':vdf_blue_3, 
        #     'vdf_blue_4':vdf_blue_4, 
        #     'vdf_blue_5':vdf_blue_5})

        # import matplotlib.pyplot as plt
        # from matplotlib import rc
        # from matplotlib import cm

        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=25)
        # rc('text', usetex=False)
        # rc('axes', linewidth=2)
        # rc('xtick.major', width=4, size=7)
        # rc('ytick.major', width=4, size=7)
        # rc('xtick.minor', width=2, size=7)
        # rc('ytick.minor', width=2, size=7)

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # cmap = cm.get_cmap('Spectral')
        # cax = ax1.matshow(combined_df.corr(), cmap=cmap, vmin=-1, vmax=1)
        # tick_marks = [i for i in range(len(combined_df.columns))]
        # plt.xticks(tick_marks, combined_df.columns, rotation='vertical')
        # plt.yticks(tick_marks, combined_df.columns)    
        # plt.gca().invert_yaxis() 
        # plt.gca().xaxis.tick_bottom()
        # plt.colorbar(cax)
        # plt.title(r'Velocity dispersion function')
        # plt.show()

        return err_colour, mocks_experimentals

    def mp_init(self, mcmc_table_pctl, nproc, experiments):
        """
        Initializes multiprocessing of mocks and smf and smhm measurements

        Parameters
        ----------
        mcmc_table_pctl: pandas dataframe
            Mcmc chain dataframe of 100 random samples

        nproc: int
            Number of processes to use in multiprocessing

        Returns
        ---------
        result: multidimensional array
            Arrays of smf and smhm data for all, red and blue galaxies
        """
        settings = self.settings

        start = time.time()
        params_df = mcmc_table_pctl.iloc[:,:9].reset_index(drop=True)
        if settings.many_behroozi_mocks:
            mock_num_df = mcmc_table_pctl.iloc[:,5].reset_index(drop=True)
            frames = [params_df, mock_num_df]
            mcmc_table_pctl_new = pd.concat(frames, axis=1)
            chunks = np.array([mcmc_table_pctl_new.values[i::5] \
                for i in range(5)])
        else:
            # chunks = np.array([params_df.values[i::5] \
            #     for i in range(5)])
            # Chunks are just numbers from 1-100 for the case where rsd + grp finder
            # were run for a selection of 100 random 1sigma models from run 32
            # and all those mocks are used instead. 
            chunks = np.arange(1,101,1).reshape(5, 20, 1) # Mimic shape of chunks above 
        pool = Pool(processes=nproc)
        # mp_dict = {"experiments":experiments, "analysis":self}
        # nargs = [(chunk, experiments) for chunk in chunks]
        experiments_list = [experiments for x in range(len(chunks))]
        result = pool.map(self.mp_func, chunks, experiments_list)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        return result

    def mp_func(self, a_list, experiments):
        """
        Apply behroozi and hybrid quenching model based on nine parameter values

        Parameters
        ----------
        a_list: multidimensional array
            Array of nine parameter values

        Returns
        ---------
        model_results: dictionary

        model_experimentals: dictionary
        """
        # a_list = nargs[:-1]
        # experiments = nargs[-1]
        # print(experiments)
        # print(a_list)
        settings = self.settings
        preprocess = self.preprocess

        # v_sim = 130**3
        v_sim = 890641.5172927063 ## cz: 3000 - 12000
        # v_sim = 165457.21308906242 ## cz: 3000-7000

        print('Reloaded')

        main_keys = ["smf_total","f_blue","phi_colour","centrals","satellites",\
            "f_red"]
        sub_keys = [{"max_total":[],"phi_total":[]},{"max_fblue":[],"fblue_total":[],\
            "fblue_cen":[],"fblue_sat":[]},\
            {"phi_red":[],"phi_blue":[]},{"gals":[],"halos":[],"gals_red":[],\
            "halos_red":[],"gals_blue":[],"halos_blue":[]},{"gals_red":[],\
            "halos_red":[],"gals_blue":[],"halos_blue":[]},{"cen_red":[],\
            "cen_blue":[],"sat_red":[],"sat_blue":[]}]
        model_results = dict(zip(main_keys,sub_keys))

        main_keys = ["vel_disp","richness","vdf"]
        sub_keys = [{"red_sigma":[],"red_cen_mstar":[],"blue_sigma":[],\
            "blue_cen_mstar":[],"red_nsat":[],"blue_nsat":[],"red_hosthalo":[],\
            "blue_hosthalo":[]},{"red_num":[],\
            "red_cen_mstar":[],"blue_num":[],"blue_cen_mstar":[],\
            "red_hosthalo":[],"blue_hosthalo":[]},{"phi_red":[],"phi_blue":[]}]

        model_experimentals = dict(zip(main_keys,sub_keys))

        for theta in a_list:  
            if settings.many_behroozi_mocks:
                randint_logmstar = int(theta[4])
                cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
                'halo_mvir_host_halo', 'cz', \
                '{0}'.format(randint_logmstar), \
                'grp_censat_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar),\
                'cen_cz_{0}'.format(randint_logmstar)]
                
                gals_df = globals.gal_group_df_subset[cols_to_use]

                gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
                    format(randint_logmstar),'groupid_{0}'.format(
                    randint_logmstar)]).reset_index(drop=True)

                gals_df[['grp_censat_{0}'.format(randint_logmstar), \
                    'groupid_{0}'.format(randint_logmstar)]] = \
                    gals_df[['grp_censat_{0}'.format(randint_logmstar),\
                    'groupid_{0}'.format(randint_logmstar)]].astype(int)


            else:
                randint_logmstar = theta[0]
                ## Change from np.int64 to int so that isinstance if statements
                ## in functions called below execute
                randint_logmstar = int(randint_logmstar)
                # 1 is the best-fit model which is calculated separately 
                if randint_logmstar == 1:
                    continue

                cols_to_use = ['halo_hostid', 'halo_id', 'halo_mvir', \
                'halo_mvir_host_halo', 'cz', 'cs_flag', \
                '{0}'.format(randint_logmstar), \
                'grp_censat_{0}'.format(randint_logmstar), \
                'groupid_{0}'.format(randint_logmstar), \
                'cen_cz_{0}'.format(randint_logmstar)]

                gals_df = globals.gal_group_df_subset[cols_to_use]
        
                gals_df = gals_df.dropna(subset=['grp_censat_{0}'.\
                format(randint_logmstar),'groupid_{0}'.format(randint_logmstar),\
                'cen_cz_{0}'.format(randint_logmstar)]).\
                reset_index(drop=True)

                gals_df[['grp_censat_{0}'.format(randint_logmstar), \
                    'groupid_{0}'.format(randint_logmstar)]] = \
                    gals_df[['grp_censat_{0}'.format(randint_logmstar),\
                    'groupid_{0}'.format(randint_logmstar)]].astype(int)

                gals_df['{0}'.format(randint_logmstar)] = \
                    np.log10(gals_df['{0}'.format(randint_logmstar)])

            #* Stellar masses in log but halo masses not in log
            # randint_logmstar-2 because the best fit randint is 1 in gal_group_df
            # and in mcmc_table the best fit set of params have been removed and the
            # index was reset so now there is an offset of 2 between the indices 
            # of the two sets of data.
            quenching_params = preprocess.mcmc_table_pctl_subset.iloc[randint_logmstar-2].\
                values[5:]
            # f_red_cen, f_red_sat = hybrid_quenching_model(theta[5:], gals_df, 
            #     'vishnu')
            if settings.quenching == 'hybrid':
                f_red_cen, f_red_sat = self.hybrid_quenching_model(quenching_params, gals_df, 
                    'vishnu', randint_logmstar)
            elif settings.quenching == 'halo':
                f_red_cen, f_red_sat = self.halo_quenching_model(quenching_params, gals_df, 
                    'vishnu')
            gals_df = self.assign_colour_label_mock(f_red_cen, f_red_sat, gals_df)

            ## Observable #1 - Total SMF
            total_model = self.measure_all_smf(gals_df, v_sim , False, randint_logmstar)    
            ## Observable #2 - Blue fraction
            # print(type(randint_logmstar))
            f_blue = self.blue_frac(gals_df, True, False, randint_logmstar)

            ## Observable #3 - sigma-M* or M*-sigma
            if settings.stacked_stat:
                red_deltav, red_cen_mstar_sigma, blue_deltav, \
                blue_cen_mstar_sigma = experiments.get_stacked_velocity_dispersion(
                    gals_df, 'model', randint_logmstar)

                model_experimentals["vel_disp"]["red_sigma"].append(red_deltav)
                model_experimentals["vel_disp"]["red_cen_mstar"].append(red_cen_mstar_sigma)
                model_experimentals["vel_disp"]["blue_sigma"].append(blue_deltav)
                model_experimentals["vel_disp"]["blue_cen_mstar"].append(blue_cen_mstar_sigma)

            else:
                red_sigma, red_cen_mstar_sigma, blue_sigma, \
                    blue_cen_mstar_sigma = experiments.get_velocity_dispersion(
                    gals_df, 'model', randint_logmstar)

                red_sigma = np.log10(red_sigma)
                blue_sigma = np.log10(blue_sigma)

                model_experimentals["vel_disp"]["red_sigma"].append(red_sigma)
                model_experimentals["vel_disp"]["red_cen_mstar"].append(red_cen_mstar_sigma)
                model_experimentals["vel_disp"]["blue_sigma"].append(blue_sigma)
                model_experimentals["vel_disp"]["blue_cen_mstar"].append(blue_cen_mstar_sigma)

            # mean_mstar_red = bs(red_sigma, red_cen_mstar_sigma, 
            #     statistic='mean', bins=np.linspace(-2,3,5))
            # mean_mstar_blue = bs(blue_sigma, blue_cen_mstar_sigma, 
            #     statistic='mean', bins=np.linspace(-1,3,5))

            cen_gals, cen_halos, cen_gals_red, cen_halos_red, cen_gals_blue, \
                cen_halos_blue, f_red_cen_red, f_red_cen_blue = \
                    self.get_centrals_mock(gals_df, randint_logmstar)
            sat_gals_red, sat_halos_red, sat_gals_blue, sat_halos_blue, \
                f_red_sat_red, f_red_sat_blue = \
                    self.get_satellites_mock(gals_df, randint_logmstar)

            phi_red_model, phi_blue_model = \
                self.get_colour_smf_from_fblue(gals_df, f_blue[1], f_blue[0], v_sim, 
                True, randint_logmstar)

            # red_sigma, red_cen_mstar_sigma, blue_sigma, \
            #     blue_cen_mstar_sigma, red_nsat, blue_nsat, \
            #     red_host_halo_mass_vd, blue_host_halo_mass_vd = \
            #     experiments.get_velocity_dispersion(gals_df, 'model', 
            #         randint_logmstar)

            red_num, red_cen_mstar_richness, blue_num, \
                blue_cen_mstar_richness, red_host_halo_mass, \
                blue_host_halo_mass = \
                experiments.get_richness(gals_df, 'model', randint_logmstar)

            # v_sim = 890641.5172927063 
            if not settings.stacked_stat:
                x_vdf, phi_vdf, error, bins, counts = experiments.\
                    get_vdf(red_sigma, blue_sigma, v_sim)

                model_experimentals["vdf"]["phi_red"].append(phi_vdf[0])
                model_experimentals["vdf"]["phi_blue"].append(phi_vdf[1])

            model_results["smf_total"]["max_total"].append(total_model[0])
            model_results["smf_total"]["phi_total"].append(total_model[1])

            model_results["f_blue"]["max_fblue"].append(f_blue[0])
            model_results["f_blue"]["fblue_total"].append(f_blue[1])
            model_results["f_blue"]["fblue_cen"].append(f_blue[2])
            model_results["f_blue"]["fblue_sat"].append(f_blue[3])

            model_results["phi_colour"]["phi_red"].append(phi_red_model)
            model_results["phi_colour"]["phi_blue"].append(phi_blue_model)

            model_results["centrals"]["gals"].append(cen_gals)
            model_results["centrals"]["halos"].append(cen_halos)
            model_results["centrals"]["gals_red"].append(cen_gals_red)
            model_results["centrals"]["halos_red"].append(cen_halos_red)
            model_results["centrals"]["gals_blue"].append(cen_gals_blue)
            model_results["centrals"]["halos_blue"].append(cen_halos_blue)

            model_results["satellites"]["gals_red"].append(sat_gals_red)
            model_results["satellites"]["halos_red"].append(sat_halos_red)
            model_results["satellites"]["gals_blue"].append(sat_gals_blue)
            model_results["satellites"]["halos_blue"].append(sat_halos_blue)

            model_results["f_red"]["cen_red"].append(f_red_cen_red)
            model_results["f_red"]["sat_red"].append(f_red_sat_red)
            model_results["f_red"]["cen_blue"].append(f_red_cen_blue)
            model_results["f_red"]["sat_blue"].append(f_red_sat_blue)

            # model_experimentals["vel_disp"]["red_sigma"].append(mean_mstar_red[1])
            # model_experimentals["vel_disp"]["red_cen_mstar"].append(mean_mstar_red[0])
            # model_experimentals["vel_disp"]["blue_sigma"].append(mean_mstar_blue[1])
            # model_experimentals["vel_disp"]["blue_cen_mstar"].append(mean_mstar_blue[0])
            # model_experimentals["vel_disp"]["red_nsat"].append(red_nsat)
            # model_experimentals["vel_disp"]["blue_nsat"].append(blue_nsat)
            # model_experimentals["vel_disp"]["red_hosthalo"].append(red_host_halo_mass_vd)
            # model_experimentals["vel_disp"]["blue_hosthalo"].append(blue_host_halo_mass_vd)

            model_experimentals["richness"]["red_num"].append(red_num)
            model_experimentals["richness"]["red_cen_mstar"].append(red_cen_mstar_richness)
            model_experimentals["richness"]["blue_num"].append(blue_num)
            model_experimentals["richness"]["blue_cen_mstar"].append(blue_cen_mstar_richness)
            model_experimentals["richness"]["red_hosthalo"].append(red_host_halo_mass)
            model_experimentals["richness"]["blue_hosthalo"].append(blue_host_halo_mass)

        return [model_results, model_experimentals]

    def Core(self, experiments):
        settings = self.settings
        preprocess = self.preprocess

        print('Assigning colour to data')
        self.catl = self.assign_colour_label_data(preprocess.catl)

        print('Measuring SMF for data')
        self.total_data, red_data, blue_data = self.measure_all_smf(self.catl, preprocess.volume, True)

        print('Measuring blue fraction for data')
        self.f_blue = self.blue_frac(self.catl, False, True)

        print('Measuring velocity dispersion for data')
        if settings.stacked_stat:
            red_deltav, red_cen_mstar_sigma, blue_deltav, \
            blue_cen_mstar_sigma = experiments.get_stacked_velocity_dispersion(
                self.catl, 'data')

            # red_sigma_stat = bs(red_cen_mstar_sigma, red_deltav,
            #     statistic='std', bins=np.linspace(8.6,11,5))
            # blue_sigma_stat = bs( blue_cen_mstar_sigma, blue_deltav,
            #     statistic='std', bins=np.linspace(8.6,11,5))
            
            # red_sigma = np.log10(red_sigma_stat[0])
            # blue_sigma = np.log10(blue_sigma_stat[0])

            self.vdisp_red_data = [red_deltav, red_cen_mstar_sigma]
            self.vdisp_blue_data = [blue_deltav, blue_cen_mstar_sigma]

        else:
            red_sigma, red_cen_mstar_sigma, blue_sigma, \
                blue_cen_mstar_sigma = \
                experiments.get_velocity_dispersion(self.catl, 'data')

            red_sigma = np.log10(red_sigma)
            blue_sigma = np.log10(blue_sigma)

            # #! Implement average of log function for next chains (49 onwards)
            # mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
            #     statistic='mean', bins=np.linspace(-2,3,5))
            # mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
            #     statistic='mean', bins=np.linspace(-1,3,5))

            self.vdisp_red_data = [red_sigma, red_cen_mstar_sigma]
            self.vdisp_blue_data = [blue_sigma, blue_cen_mstar_sigma]


        # red_sigma, red_cen_mstar_sigma, blue_sigma, \
        #     blue_cen_mstar_sigma = experiments.get_velocity_dispersion(self.catl, 'data')

        # red_sigma = np.log10(red_sigma)
        # blue_sigma = np.log10(blue_sigma)

        # # self.mean_mstar_red_data = bs(red_sigma, red_cen_mstar_sigma, 
        # #     statistic='mean', bins=np.linspace(-2,3,5))
        # # self.mean_mstar_blue_data = bs(blue_sigma, blue_cen_mstar_sigma, 
        # #     statistic='mean', bins=np.linspace(-1,3,5))

        # self.vdisp_red_data = [red_sigma, red_cen_mstar_sigma]
        # self.vdisp_blue_data = [blue_sigma, blue_cen_mstar_sigma]

        print('Measuring reconstructed red and blue SMF for data')
        self.phi_red_data, self.phi_blue_data = self.get_colour_smf_from_fblue(self.catl, self.f_blue[1], 
            self.f_blue[0], preprocess.volume, False)

        print('Initial population of halo catalog')
        self.model_init = self.halocat_init(settings.halo_catalog, preprocess.z_median)

        print('Measuring error in data from mocks')
        self.error_data, self.mocks_stdevs = self.get_err_data(settings.path_to_mocks, experiments)
        self.dof = len(self.error_data) - len(preprocess.bf_params)

        # return [self.total_data, self.f_blue, self.mean_mstar_red_data, self.mean_mstar_blue_data, self.phi_red_data, self.phi_blue_data, self.error_data, self.mocks_stdevs, self.dof]

        return [self.total_data, self.f_blue, self.vdisp_red_data, self.vdisp_blue_data, self.phi_red_data, self.phi_blue_data, self.error_data, self.mocks_stdevs, self.dof]

    def Mocks_And_Models(self, experiments):
        settings = self.settings
        preprocess = self.preprocess

        self.best_fit_core, self.best_fit_experimentals, gals_df = self.get_best_fit_model(preprocess.bf_params, experiments)

        print('Multiprocessing') #~18 minutes
        ## self.result has shape [5,2]: 5 chunks of 2 dictionaries
        self.result = self.mp_init(preprocess.mcmc_table_pctl_subset, settings.nproc, experiments)
        return self.result, [self.best_fit_core, self.best_fit_experimentals], gals_df
        # self.mp_models, self.mp_experimentals = self.mp_init(preprocess.mcmc_table_pctl_subset, settings.nproc)

    






