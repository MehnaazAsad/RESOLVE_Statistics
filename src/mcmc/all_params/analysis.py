"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from preprocess import Preprocess
from settings import Settings
import numpy as np
from scipy.stats import binned_statistic as bs


class Analysis():

    catl = None
    total_data = None
    f_blue = []
    phi_red_data = None
    phi_blue_data = None

    @staticmethod
    def assign_colour_label_data(catl):
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

    @staticmethod
    def measure_all_smf(table, volume, data_bool, randint_logmstar=None):
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
                Analysis.diff_smf(table[logmstar_col], volume, False)
            max_red, phi_red, err_red, bins_red, counts_red = \
                Analysis.diff_smf(table[logmstar_col].loc[table[colour_col] == 'R'], 
                volume, False, 'R')
            max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
                Analysis.diff_smf(table[logmstar_col].loc[table[colour_col] == 'B'], 
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
                Analysis.diff_smf(10**(table[logmstar_col]), volume, True)
            # max_red, phi_red, err_red, bins_red, counts_red = \
            #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'R']), 
            #     volume, True, 'R')
            # max_blue, phi_blue, err_blue, bins_blue, counts_blue = \
            #     diff_smf(10**(table[logmstar_col].loc[table[colour_col] == 'B']), 
            #     volume, True, 'B')
            return [max_total, phi_total, err_total, counts_total]

    @staticmethod
    def diff_smf(mstar_arr, volume, h1_bool, colour_flag=False):
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
        if not h1_bool:
            # changing from h=0.7 to h=1 assuming h^-2 dependence
            logmstar_arr = np.log10((10**mstar_arr) / 2.041)
        else:
            logmstar_arr = np.log10(mstar_arr)

        if Settings.survey == 'eco' or Settings.survey == 'resolvea':
            bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
            if Settings.survey == 'eco' and colour_flag == 'R':
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 6
                
            elif Settings.survey == 'eco' and colour_flag == 'B':
                bin_max = np.round(np.log10((10**11) / 2.041), 1)
                bin_num = 6

            elif Settings.survey == 'resolvea':
                # different to avoid nan in inverse corr mat
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 7

            else:
                # For eco total
                bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
                bin_num = 7

            bins = np.linspace(bin_min, bin_max, bin_num)

        elif Settings.survey == 'resolveb':
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

    @staticmethod
    def blue_frac_helper(arr):
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

    @staticmethod
    def blue_frac(catl, h1_bool, data_bool, randint_logmstar=None):
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
        if data_bool:
            mstar_arr = catl.logmstar.values
        elif randint_logmstar != 1:
            mstar_arr = catl['{0}'.format(randint_logmstar)].values
        elif randint_logmstar == 1:
            mstar_arr = catl['behroozi_bf'].values

        colour_label_arr = catl.colour_label.values

        if not h1_bool:
            # changing from h=0.7 to h=1 assuming h^-2 dependence
            logmstar_arr = np.log10((10**mstar_arr) / 2.041)
        else:
            logmstar_arr = mstar_arr

        if Settings.survey == 'eco' or Settings.survey == 'resolvea':
            bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7

            bins = np.linspace(bin_min, bin_max, bin_num)

        elif Settings.survey == 'resolveb':
            bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
            bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
            bins = np.linspace(bin_min, bin_max, 7)

        result = bs(logmstar_arr, colour_label_arr, Analysis.blue_frac_helper, bins=bins)
        edges = result[1]
        dm = edges[1] - edges[0]  # Bin width
        maxis = 0.5 * (edges[1:] + edges[:-1])  # Mass axis i.e. bin centers
        f_blue = result[0]

        return maxis, f_blue

    @staticmethod
    def get_colour_smf_from_fblue(df, frac_arr, bin_centers, volume, h1_bool, 
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

    @staticmethod
    def Core():
        print('Assigning colour to data')
        Analysis.catl = Analysis.assign_colour_label_data(Preprocess.catl)

        print('Measuring SMF for data')
        Analysis.total_data, red_data, blue_data = Analysis.measure_all_smf(Analysis.catl, Preprocess.volume, True)

        print('Measuring blue fraction for data')
        Analysis.f_blue = Analysis.blue_frac(Analysis.catl, False, True)

        print('Measuring reconstructed red and blue SMF for data')
        Analysis.phi_red_data, Analysis.phi_blue_data = Analysis.get_colour_smf_from_fblue(Analysis.catl, Analysis.f_blue[1], 
            Analysis.f_blue[0], Preprocess.volume, False)


