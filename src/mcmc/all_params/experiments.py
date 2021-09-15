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
            if settings.level == 'group':
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
            if settings.level == 'group':
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
            blue_host_halo_mass_arr.append(blue_host_halo_mass_arr)

        return red_num_arr, red_cen_stellar_mass_arr, blue_num_arr, \
            blue_cen_stellar_mass_arr, red_host_halo_mass_arr, \
            blue_host_halo_mass_arr

    def Run_Experiments(self):
        analysis = self.analysis

        red_sigma, red_cen_mstar_sigma, blue_sigma, \
            blue_cen_mstar_sigma, red_nsat, blue_nsat = \
            self.get_velocity_dispersion(analysis.catl, 'data')

        red_num, red_cen_mstar_richness, blue_num, \
            blue_cen_mstar_richness, red_host_halo_mass, \
            blue_host_halo_mass = \
            self.get_richness(analysis.catl, 'data')

        self.data_experimentals["vel_disp"] = {'red_sigma':red_sigma,
            'red_cen_mstar':red_cen_mstar_sigma,
            'blue_sigma':blue_sigma, 'blue_cen_mstar':blue_cen_mstar_sigma,
            'red_nsat':red_nsat, 'blue_nsat':blue_nsat}
        
        self.data_experimentals["richness"] = {'red_num':red_num,
            'red_cen_mstar':red_cen_mstar_richness, 'blue_num':blue_num,
            'blue_cen_mstar':blue_cen_mstar_richness, 
            'red_hosthalo':red_host_halo_mass, 'blue_hosthalo':blue_host_halo_mass}

        