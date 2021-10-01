"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from scipy.stats import binned_statistic as bs
import numpy as np

class Experiments():

    def __init__(self, analysis) -> None:
        self.data_experimentals = {}
        self.settings = analysis.settings
        self.preprocess = analysis.preprocess
        self.analysis = analysis

    def get_velocity_dispersion(self, catl, catl_type, randint=None):
        """Calculating velocity dispersion of groups from real data, model or 
            mock

        Args:
            catl (pandas.DataFrame): Data catalogue 

            catl_type (string): 'data', 'mock', 'model'

            randint (optional): int
                Mock number in case many Behroozi mocks were used. Defaults to None.

        Returns:
            red_sigma_arr (numpy array): Velocity dispersion of red galaxies

            red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

            blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
            
            blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

            red_nsat_arr (numpy array): Number of satellites around red centrals

            blue_nsat_arr (numpy array): Number of satellites around blue centrals

        """
        settings = self.settings
        preprocess = self.preprocess

        if catl_type == 'data':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                catl = catl.loc[catl.logmstar >= 8.9]
            elif settings.survey == 'resolveb':
                catl = catl.loc[catl.logmstar >= 8.7]

        if catl_type == 'data' or catl_type == 'mock':
            catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            logmstar_col = 'logmstar'
            ## Use group level for data even when settings.level == halo
            if catl_type == 'data' or settings.level == 'group':
                galtype_col = 'g_galtype'
                id_col = 'groupid'
            ## No halo level in data
            if catl_type == 'mock':
                if settings.level == 'halo':
                    galtype_col = 'cs_flag'
                    ## Halo ID is equivalent to halo_hostid in vishnu mock
                    id_col = 'haloid'
    
        if catl_type == 'model':
            if settings.survey == 'eco':
                min_cz = 3000
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolvea':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolveb':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.7

            if randint != 1:
                logmstar_col = '{0}'.format(randint)
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            elif randint == 1:
                logmstar_col = 'behroozi_bf'
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            else:
                logmstar_col = 'stellar_mass'
                galtype_col = 'g_galtype'
                id_col = 'groupid'
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= (10**mstar_limit)/2.041)]
                catl[logmstar_col] = np.log10(catl[logmstar_col])

            if settings.level == 'halo':
                galtype_col = 'cs_flag'
                id_col = 'halo_hostid'

        red_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'R') & (catl[galtype_col] == 1)].values)  
        blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'B') & (catl[galtype_col] == 1)].values)

        red_singleton_counter = 0
        red_sigma_arr = []
        red_cen_stellar_mass_arr = []
        # red_sigmagapper_arr = []
        red_nsat_arr = []
        for key in red_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            if len(group) == 1:
                red_singleton_counter += 1
            else:
                cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                    .values == 1].values[0]
                nsat = len(group.loc[group[galtype_col].values == 0])
                # Different velocity definitions
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
                # cz_grp = np.unique(group.grpcz.values)[0]

                # Velocity difference
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                sigma = deltav.std()
                # red_sigmagapper = np.unique(group.grpsig.values)[0]
                
                red_sigma_arr.append(sigma)
                red_cen_stellar_mass_arr.append(cen_stellar_mass)
                # red_sigmagapper_arr.append(red_sigmagapper)
                red_nsat_arr.append(nsat)

        blue_singleton_counter = 0
        blue_sigma_arr = []
        blue_cen_stellar_mass_arr = []
        # blue_sigmagapper_arr = []
        blue_nsat_arr = []
        for key in blue_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            if len(group) == 1:
                blue_singleton_counter += 1
            else:
                cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                    .values == 1].values[0]
                nsat = len(group.loc[group[galtype_col].values == 0])
                # Different velocity definitions
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
                # cz_grp = np.unique(group.grpcz.values)[0]

                # Velocity difference
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                # sigma = deltav[deltav!=0].std()
                sigma = deltav.std()
                # blue_sigmagapper = np.unique(group.grpsig.values)[0]
                
                blue_sigma_arr.append(sigma)
                blue_cen_stellar_mass_arr.append(cen_stellar_mass)
                # blue_sigmagapper_arr.append(blue_sigmagapper)
                blue_nsat_arr.append(nsat)


        return red_sigma_arr, red_cen_stellar_mass_arr, blue_sigma_arr, \
            blue_cen_stellar_mass_arr, red_nsat_arr, blue_nsat_arr

    def get_richness(self, catl, catl_type, randint=None, central_bool=True):
        settings = self.settings
        preprocess = self.preprocess

        if catl_type == 'data':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                catl = catl.loc[catl.logmstar >= 8.9]
            elif settings.survey == 'resolveb':
                catl = catl.loc[catl.logmstar >= 8.7]

        if catl_type == 'data' or catl_type == 'mock':
            catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            logmstar_col = 'logmstar'
            ## Use group level for data even when settings.level == halo
            if catl_type == 'data' or settings.level == 'group':
                galtype_col = 'g_galtype'
                id_col = 'groupid'
            ## No halo level in data
            if catl_type == 'mock':
                if settings.level == 'halo':
                    galtype_col = 'cs_flag'
                    ## Halo ID is equivalent to halo_hostid in vishnu mock
                    id_col = 'haloid'
    
        if catl_type == 'model':
            if settings.survey == 'eco':
                min_cz = 3000
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolvea':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolveb':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.7

            if randint != 1:
                logmstar_col = '{0}'.format(randint)
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            elif randint == 1:
                logmstar_col = 'behroozi_bf'
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            else:
                logmstar_col = 'stellar_mass'
                galtype_col = 'g_galtype'
                id_col = 'groupid'
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= (10**mstar_limit)/2.041)]
                catl[logmstar_col] = np.log10(catl[logmstar_col])

            if settings.level == 'halo':
                galtype_col = 'cs_flag'
                id_col = 'halo_hostid'

        red_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'R') & (catl[galtype_col] == 1)].values)  
        blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'B') & (catl[galtype_col] == 1)].values)

        red_num_arr = []
        red_cen_stellar_mass_arr = []
        red_host_halo_mass_arr = []
        for key in red_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            # print(logmstar_col)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                #So that the array is returned either way without the need for 
                #two separate return statements
                host_halo_mass = 0 
            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_num_arr.append(num)
            red_host_halo_mass_arr.append(host_halo_mass)

                
        blue_num_arr = []
        blue_cen_stellar_mass_arr = []
        blue_host_halo_mass_arr = []
        for key in blue_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                host_halo_mass = 0 
            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_num_arr.append(num)
            blue_host_halo_mass_arr.append(host_halo_mass)

        return red_num_arr, red_cen_stellar_mass_arr, blue_num_arr, \
            blue_cen_stellar_mass_arr, red_host_halo_mass_arr, \
            blue_host_halo_mass_arr

    def get_vdf(self, red_sigma, blue_sigma, volume):
        bins = np.linspace(0, 250, 7)
        # Unnormalized histogram and bin edges
        counts_red, edg = np.histogram(red_sigma, bins=bins)  # paper used 17 bins
        counts_blue, edg = np.histogram(blue_sigma, bins=bins)  # paper used 17 bins

        dm = edg[1] - edg[0]  # Bin width
        maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
        # Normalized to volume and bin width
        err_poiss_red = np.sqrt(counts_red) / (volume * dm)
        err_poiss_blue = np.sqrt(counts_blue)/ (volume * dm)
        phi_red = counts_red / (volume * dm)  # not a log quantity
        phi_blue = counts_blue / (volume * dm)  # not a log quantity
        phi_red = np.log10(phi_red)
        phi_blue = np.log10(phi_blue)
        return maxis, [phi_red, phi_blue], [err_poiss_red, err_poiss_blue], bins, [counts_red, counts_blue]

    def Run_Experiments(self):
        analysis = self.analysis
        preprocess = self.preprocess

        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma, red_nsat, blue_nsat = \
            self.get_velocity_dispersion(analysis.catl, 'data')

        red_num, red_cen_mstar_richness, blue_num, \
            blue_cen_mstar_richness, red_host_halo_mass, \
            blue_host_halo_mass = \
            self.get_richness(analysis.catl, 'data')

        x_vdf, phi_vdf, error, bins, counts = self.\
            get_vdf(red_sigma, blue_sigma, preprocess.volume)

        self.data_experimentals["vel_disp"] = {'red_sigma':red_sigma,
            'red_cen_mstar':red_cen_mstar_sigma,
            'blue_sigma':blue_sigma, 'blue_cen_mstar':blue_cen_mstar_sigma,
            'red_nsat':red_nsat, 'blue_nsat':blue_nsat}
        
        self.data_experimentals["richness"] = {'red_num':red_num,
            'red_cen_mstar':red_cen_mstar_richness, 'blue_num':blue_num,
            'blue_cen_mstar':blue_cen_mstar_richness, 
            'red_hosthalo':red_host_halo_mass, 'blue_hosthalo':blue_host_halo_mass}

        self.data_experimentals["vdf"] = {'x_vdf':x_vdf,
            'phi_vdf':phi_vdf,
            'error':error, 'bins':bins,
            'counts':counts}

        return self.data_experimentals
        
    def get_richness_test(self, catl, catl_type, randint=None, central_bool=True):
        settings = self.settings
        preprocess = self.preprocess

        if catl_type == 'data':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                catl = catl.loc[catl.logmstar >= 8.9]
            elif settings.survey == 'resolveb':
                catl = catl.loc[catl.logmstar >= 8.7]

        if catl_type == 'data' or catl_type == 'mock':
            catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            logmstar_col = 'logmstar'
            ## Use group level for data even when settings.level == halo
            if catl_type == 'data' or settings.level == 'group':
                galtype_col = 'g_galtype'
                id_col = 'groupid'
            ## No halo level in data
            if catl_type == 'mock':
                if settings.level == 'halo':
                    galtype_col = 'cs_flag'
                    ## Halo ID is equivalent to halo_hostid in vishnu mock
                    id_col = 'haloid'
    
        if catl_type == 'model':
            if settings.survey == 'eco':
                min_cz = 3000
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolvea':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolveb':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.7

            if randint != 1:
                logmstar_col = '{0}'.format(randint)
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            elif randint == 1:
                logmstar_col = 'behroozi_bf'
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            else:
                logmstar_col = 'stellar_mass'
                galtype_col = 'g_galtype'
                id_col = 'groupid'
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= (10**mstar_limit)/2.041)]
                catl[logmstar_col] = np.log10(catl[logmstar_col])

            if settings.level == 'halo':
                galtype_col = 'cs_flag'
                id_col = 'halo_hostid'

        red_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'R') & (catl[galtype_col] == 1)].values)  
        blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'B') & (catl[galtype_col] == 1)].values)

        red_num_arr = []
        red_cen_stellar_mass_arr = []
        red_host_halo_mass_arr = []
        for key in red_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            # print(logmstar_col)
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                #So that the array is returned either way without the need for 
                #two separate return statements
                host_halo_mass = 0 
            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1
            red_cen_stellar_mass_arr.append(cen_stellar_mass)
            red_num_arr.append(num)
            red_host_halo_mass_arr.append(host_halo_mass)

        blue_num_arr = []
        blue_cen_stellar_mass_arr = []
        blue_host_halo_mass_arr = []
        for key in blue_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                .values == 1].values[0]
            if catl_type == 'model':
                host_halo_mass = np.unique(group.halo_mvir_host_halo.values)[0]
            else: 
                host_halo_mass = 0 
            if central_bool:
                num = len(group)
            elif not central_bool:
                num = len(group) - 1
            blue_cen_stellar_mass_arr.append(cen_stellar_mass)
            blue_num_arr.append(num)
            blue_host_halo_mass_arr.append(host_halo_mass)

        return red_subset_ids, red_num_arr, red_host_halo_mass_arr, red_cen_stellar_mass_arr, \
            blue_subset_ids, blue_num_arr, blue_host_halo_mass_arr, blue_cen_stellar_mass_arr

    def get_velocity_dispersion_test(self, catl, catl_type, randint=None):
        """Calculating velocity dispersion of groups from real data, model or 
            mock

        Args:
            catl (pandas.DataFrame): Data catalogue 

            catl_type (string): 'data', 'mock', 'model'

            randint (optional): int
                Mock number in case many Behroozi mocks were used. Defaults to None.

        Returns:
            red_sigma_arr (numpy array): Velocity dispersion of red galaxies

            red_cen_stellar_mass_arr (numpy array): Group red central stellar mass

            blue_sigma_arr (numpy array): Velocity dispersion of blue galaxies
            
            blue_cen_stellar_mass_arr (numpy array): Blue central stellar mass

            red_nsat_arr (numpy array): Number of satellites around red centrals

            blue_nsat_arr (numpy array): Number of satellites around blue centrals

        """
        settings = self.settings
        preprocess = self.preprocess

        if catl_type == 'data':
            if settings.survey == 'eco' or settings.survey == 'resolvea':
                catl = catl.loc[catl.logmstar >= 8.9]
            elif settings.survey == 'resolveb':
                catl = catl.loc[catl.logmstar >= 8.7]

        if catl_type == 'data' or catl_type == 'mock':
            catl.logmstar = np.log10((10**catl.logmstar) / 2.041)
            logmstar_col = 'logmstar'
            ## Use group level for data even when settings.level == halo
            if catl_type == 'data' or settings.level == 'group':
                galtype_col = 'g_galtype'
                id_col = 'groupid'
            ## No halo level in data
            if catl_type == 'mock':
                if settings.level == 'halo':
                    galtype_col = 'cs_flag'
                    ## Halo ID is equivalent to halo_hostid in vishnu mock
                    id_col = 'haloid'
    
        if catl_type == 'model':
            if settings.survey == 'eco':
                min_cz = 3000
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolvea':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.9
            elif settings.survey == 'resolveb':
                min_cz = 4500
                max_cz = 7000
                mstar_limit = 8.7

            if randint != 1:
                logmstar_col = '{0}'.format(randint)
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            elif randint == 1:
                logmstar_col = 'behroozi_bf'
                galtype_col = 'g_galtype_{0}'.format(randint)
                id_col = 'groupid_{0}'.format(randint)
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= np.log10((10**mstar_limit)/2.041))]

            else:
                logmstar_col = 'stellar_mass'
                galtype_col = 'g_galtype'
                id_col = 'groupid'
                # Using the same survey definition as in mcmc smf i.e excluding the 
                # buffer except no M_r cut since vishnu mock has no M_r info. Only grpcz
                # and M* star cuts to mimic mocks and data.
                catl = preprocess.mock_add_grpcz(catl, False, id_col)
                catl = catl.loc[(catl.grpcz.values >= min_cz) & \
                    (catl.grpcz.values <= max_cz) & \
                    (catl[logmstar_col].values >= (10**mstar_limit)/2.041)]
                catl[logmstar_col] = np.log10(catl[logmstar_col])

            if settings.level == 'halo':
                galtype_col = 'cs_flag'
                id_col = 'halo_hostid'

        red_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'R') & (catl[galtype_col] == 1)].values)  
        blue_subset_ids = np.unique(catl[id_col].loc[(catl.\
            colour_label == 'B') & (catl[galtype_col] == 1)].values)

        red_singleton_counter = 0
        red_sigma_arr = []
        red_cen_stellar_mass_arr = []
        # red_sigmagapper_arr = []
        red_nsat_arr = []
        red_keys_arr = []
        for key in red_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            if len(group) == 1:
                red_singleton_counter += 1
            else:
                red_keys_arr.append(key)
                cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                    .values == 1].values[0]
                nsat = len(group.loc[group[galtype_col].values == 0])
                # Different velocity definitions
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
                # cz_grp = np.unique(group.grpcz.values)[0]

                # Velocity difference
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                sigma = deltav.std()
                # red_sigmagapper = np.unique(group.grpsig.values)[0]
                
                red_sigma_arr.append(sigma)
                red_cen_stellar_mass_arr.append(cen_stellar_mass)
                # red_sigmagapper_arr.append(red_sigmagapper)
                red_nsat_arr.append(nsat)

        blue_singleton_counter = 0
        blue_sigma_arr = []
        blue_cen_stellar_mass_arr = []
        # blue_sigmagapper_arr = []
        blue_nsat_arr = []
        blue_keys_arr = []
        for key in blue_subset_ids: 
            group = catl.loc[catl[id_col] == key]
            if len(group) == 1:
                blue_singleton_counter += 1
            else:
                blue_keys_arr.append(key)
                cen_stellar_mass = group[logmstar_col].loc[group[galtype_col]\
                    .values == 1].values[0]
                nsat = len(group.loc[group[galtype_col].values == 0])
                # Different velocity definitions
                mean_cz_grp = np.round(np.mean(group.cz.values),2)
                cen_cz_grp = group.cz.loc[group[galtype_col].values == 1].values[0]
                # cz_grp = np.unique(group.grpcz.values)[0]

                # Velocity difference
                deltav = group.cz.values - len(group)*[mean_cz_grp]
                # sigma = deltav[deltav!=0].std()
                sigma = deltav.std()
                # blue_sigmagapper = np.unique(group.grpsig.values)[0]
                
                blue_sigma_arr.append(sigma)
                blue_cen_stellar_mass_arr.append(cen_stellar_mass)
                # blue_sigmagapper_arr.append(blue_sigmagapper)
                blue_nsat_arr.append(nsat)


        return red_subset_ids, red_sigma_arr, red_cen_stellar_mass_arr, red_nsat_arr, red_keys_arr, \
            blue_subset_ids, blue_sigma_arr, blue_cen_stellar_mass_arr, blue_nsat_arr, blue_keys_arr

def experiment_func():
    richness_red_ids, N_red, mhalo_red, richness_mstar_red, richness_blue_ids, \
        N_blue, mhalo_blue, richness_mstar_blue = \
        experiments.get_richness_test(gals_df, 'model', randint_logmstar)

    veldisp_red_ids, sigma_red, veldisp_mstar_red, nsat_red, veldisp_red_keys, \
        veldisp_blue_ids, sigma_blue, veldisp_mstar_blue, nsat_blue, veldisp_blue_keys = \
        experiments.get_velocity_dispersion_test(gals_df, 'model', randint_logmstar)

    richness_results_red = [richness_red_ids, N_red, mhalo_red, richness_mstar_red]
    richness_results_blue = [richness_blue_ids, N_blue, mhalo_blue, richness_mstar_blue]

    veldisp_results_red = [veldisp_red_keys, sigma_red, veldisp_mstar_red, nsat_red]
    veldisp_results_blue = [veldisp_blue_keys, sigma_blue, veldisp_mstar_blue, nsat_blue]

    richness_dict_red = {z[0]: list(z[1:]) for z in zip(*richness_results_red)} 
    veldisp_dict_red = {z[0]: list(z[1:]) for z in zip(*veldisp_results_red)} 

    richness_dict_blue = {z[0]: list(z[1:]) for z in zip(*richness_results_blue)} 
    veldisp_dict_blue = {z[0]: list(z[1:]) for z in zip(*veldisp_results_blue)} 


    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_red_keys:
        mhalo_arr.append(np.log10(richness_dict_red[key][1])) 
        sigma_arr.append(veldisp_dict_red[key][0])
        mstar_arr.append(veldisp_dict_red[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    mstar_binned_red_arr = []
    sigma_binned_red_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        mstar_binned_red_arr.append(mstar_ii)
        sigma_binned_red_arr.append(sigma_ii)

    mhalo_arr = []
    mstar_arr = []
    sigma_arr = []
    for key in veldisp_blue_keys:
        mhalo_arr.append(np.log10(richness_dict_blue[key][1])) 
        sigma_arr.append(veldisp_dict_blue[key][0])
        mstar_arr.append(veldisp_dict_blue[key][1])

        
    bins=np.linspace(10, 15, 15)
    last_index = len(bins)-1
    mstar_binned_blue_arr = []
    sigma_binned_blue_arr = []
    for index1, bin_edge in enumerate(bins):
        if index1 == last_index:
            break
        mstar_ii = []
        sigma_ii = []
        for index2, mhalo in enumerate(mhalo_arr):
            if mhalo >= bin_edge and mhalo < bins[index1+1]:
                mstar_ii.append(mstar_arr[index2])
                sigma_ii.append(sigma_arr[index2])
        mstar_binned_blue_arr.append(mstar_ii)
        sigma_binned_blue_arr.append(sigma_ii)

    fig, axs = plt.subplots(2, 7, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    sp_idx = 0
    for idx in range(len(sigma_binned_red_arr)):
        if idx < 7:
            if len(sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(sigma_binned_red_arr[idx], mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[0, idx].plot(mean_centers_red, mean_stats_red[0], c='indianred', zorder=20, lw=6, ls='--')
                
            axs[0, idx].scatter(sigma_binned_red_arr[idx], mstar_binned_red_arr[idx],
                c='maroon', s=150, edgecolors='k', zorder=5)
            
            axs[0, idx].set_title("Halo bin {0} - {1}".format(np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[0, idx].set_xlim(0, 250)        
        else:
            if len(sigma_binned_red_arr[idx]) > 0:
                mean_stats_red = bs(sigma_binned_red_arr[idx], mstar_binned_red_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_red = 0.5 * (mean_stats_red[1][1:] + \
                    mean_stats_red[1][:-1])

                axs[1, sp_idx].plot(mean_centers_red, mean_stats_red[0], c='indianred', zorder=20, lw=6, ls='--')
            
            axs[1, sp_idx].scatter(sigma_binned_red_arr[idx], mstar_binned_red_arr[idx], 
                c='maroon', s=150, edgecolors='k', zorder=5)
            
            axs[1, sp_idx].set_title("Halo bin {0} - {1}".format(np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[1, sp_idx].set_xlim(0, 250)        

            sp_idx += 1

    sp_idx = 0
    for idx in range(len(sigma_binned_blue_arr)):
        if idx < 7:
            if len(sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(sigma_binned_blue_arr[idx], mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[0, idx].plot(mean_centers_blue, mean_stats_blue[0], c='cornflowerblue', zorder=20, lw=6, ls='--')

            axs[0, idx].scatter(sigma_binned_blue_arr[idx], mstar_binned_blue_arr[idx],
                c='darkblue', s=150, edgecolors='k', zorder=10)

            axs[0, idx].set_title("Halo bin {0} - {1}".format(np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[0, idx].set_xlim(0, 250)        
        else:
            if len(sigma_binned_blue_arr[idx]) > 0:
                mean_stats_blue = bs(sigma_binned_blue_arr[idx], mstar_binned_blue_arr[idx], 
                    statistic='mean', bins=np.linspace(0,250,6))
                mean_centers_blue = 0.5 * (mean_stats_blue[1][1:] + \
                    mean_stats_blue[1][:-1])

                axs[1, sp_idx].plot(mean_centers_blue, mean_stats_blue[0], c='cornflowerblue', zorder=20, lw=6, ls='--')

            axs[1, sp_idx].scatter(sigma_binned_blue_arr[idx], mstar_binned_blue_arr[idx], 
                c='darkblue', s=150, edgecolors='k', zorder=10)

            axs[1, sp_idx].set_title("Halo bin {0} - {1}".format(np.round(bins[idx],2), np.round(bins[idx+1],2)), fontsize=15)
            axs[1, sp_idx].set_xlim(0, 250)        
            sp_idx += 1

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r'\boldmath$\sigma_{halo} \left[\mathrm{km/s} \right]$', fontsize=30)
    plt.ylabel(r'\boldmath$\log_{10}\ M_{*, halo\ cen} \left[\mathrm{M_\odot}\, \mathrm{h}^{-1} \right]$',fontsize=30)
    plt.title('{0} quenching'.format(settings.quenching), pad=40.5)
    plt.show
