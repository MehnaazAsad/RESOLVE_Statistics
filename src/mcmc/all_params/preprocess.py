"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

import pandas as pd
import numpy as np
import globals
import emcee
import os

class Preprocess():

    def __init__(self, settings) -> None:
        self.catl = None
        self.volume = None
        self.z_median = None
        self.bf_params = None
        self.bf_chi2 = None
        # self.mcmc_table_pctl_subset = None
        self.mcmc_table_pctl = None
        self.settings = settings

    def read_mcmc(self, path_to_file):
        """
        Reads mcmc chain from file

        Parameters
        ----------
        path_to_file: string
            Path to mcmc chain file

        Returns
        ---------
        emcee_table: pandas.DataFrame
            Dataframe of mcmc chain values with NANs removed
        """
        settings = self.settings

        colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'mstar_q','mh_q','mu','nu']
        
        if settings.run >= 37:
            reader = emcee.backends.HDFBackend(path_to_file, read_only=True)
            flatchain = reader.get_chain(flat=True)
            emcee_table = pd.DataFrame(flatchain, columns=colnames)
        elif settings.run < 37:
            emcee_table = pd.read_csv(path_to_file, names=colnames, comment='#',
                header=None, sep='\s+')

            for idx,row in enumerate(emcee_table.values):

                ## For cases where 5 params on one line and 3 on the next
                if np.isnan(row)[6] == True and np.isnan(row)[5] == False:
                    mhalo_q_val = emcee_table.values[idx+1][0]
                    mu_val = emcee_table.values[idx+1][1]
                    nu_val = emcee_table.values[idx+1][2]
                    row[6] = mhalo_q_val
                    row[7] = mu_val
                    row[8] = nu_val 

                ## For cases where 4 params on one line, 4 on the next and 1 on the 
                ## third line (numbers in scientific notation unlike case above)
                elif np.isnan(row)[4] == True and np.isnan(row)[3] == False:
                    scatter_val = emcee_table.values[idx+1][0]
                    mstar_q_val = emcee_table.values[idx+1][1]
                    mhalo_q_val = emcee_table.values[idx+1][2]
                    mu_val = emcee_table.values[idx+1][3]
                    nu_val = emcee_table.values[idx+2][0]
                    row[4] = scatter_val
                    row[5] = mstar_q_val
                    row[6] = mhalo_q_val
                    row[7] = mu_val
                    row[8] = nu_val 

            emcee_table = emcee_table.dropna(axis='index', how='any').\
                reset_index(drop=True)

        return emcee_table

    def read_mock_catl(self, filename, catl_format='.hdf5'):
        """
        Function to read ECO/RESOLVE catalogues.

        Parameters
        ----------
        filename: string
            path and name of the ECO/RESOLVE catalogue to read

        catl_format: string, optional (default = '.hdf5')
            type of file to read.
            Options:
                - '.hdf5': Reads in a catalogue in HDF5 format

        Returns
        -------
        mock_pd: pandas DataFrame
            DataFrame with galaxy/group information

        Examples
        --------
        # Specifying `filename`
        >>> filename = 'ECO_catl.hdf5'

        # Reading in Catalogue
        >>> mock_pd = reading_catls(filename, format='.hdf5')

        >>> mock_pd.head()
                x          y         z          vx          vy          vz  \
        0  10.225435  24.778214  3.148386  356.112457 -318.894409  366.721832
        1  20.945772  14.500367 -0.237940  168.731766   37.558834  447.436951
        2  21.335835  14.808488  0.004653  967.204407 -701.556763 -388.055115
        3  11.102760  21.782235  2.947002  611.646484 -179.032089  113.388794
        4  13.217764  21.214905  2.113904  120.689598  -63.448833  400.766541

        loghalom  cs_flag  haloid  halo_ngal    ...        cz_nodist      vel_tot  \
        0    12.170        1  196005          1    ...      2704.599189   602.490355
        1    11.079        1  197110          1    ...      2552.681697   479.667489
        2    11.339        1  197131          1    ...      2602.377466  1256.285409
        3    11.529        1  199056          1    ...      2467.277182   647.318259
        4    10.642        1  199118          1    ...      2513.381124   423.326770

            vel_tan     vel_pec     ra_orig  groupid    M_group g_ngal  g_galtype  \
        0   591.399858 -115.068833  215.025116        0  11.702527      1          1
        1   453.617221  155.924074  182.144134        1  11.524787      4          0
        2  1192.742240  394.485714  182.213220        1  11.524787      4          0
        3   633.928896  130.977416  210.441320        2  11.502205      1          1
        4   421.064495   43.706352  205.525386        3  10.899680      1          1

        halo_rvir
        0   0.184839
        1   0.079997
        2   0.097636
        3   0.113011
        4   0.057210
        """
        ## Checking if file exists
        if not os.path.exists(filename):
            msg = '`filename`: {0} NOT FOUND! Exiting..'.format(filename)
            raise ValueError(msg)
        ## Reading file
        if catl_format=='.hdf5':
            mock_pd = pd.read_hdf(filename)
        else:
            msg = '`catl_format` ({0}) not supported! Exiting...'.format(catl_format)
            raise ValueError(msg)

        return mock_pd

    def mock_add_grpcz_old(self, mock_df, data_bool=None, grpid_col=None, 
        censat_col=None, cencz_col=None):
        """Adds column of group cz values to mock catalogues

        Args:
            mock_df (pandas.DataFrame): Mock catalogue

        Returns:
            pandas.DataFrame: Mock catalogue with new column called grpcz added
        """
        if data_bool:
            groups = mock_df.groupby('groupid') 
            keys = groups.groups.keys() 
            grpcz_new = [] 
            grpn = []
            for key in keys: 
                group = groups.get_group(key) 
                cen_cz = group.cz.loc[group.g_galtype == 1].values[0] 
                grpcz_new.append(cen_cz) 
                grpn.append(len(group))

            full_grpcz_arr = np.repeat(grpcz_new, grpn)
            mock_df['grpcz_new'] = full_grpcz_arr
        elif data_bool is None:
            ## Mocks case
            groups = mock_df.groupby('groupid') 
            keys = groups.groups.keys() 
            grpcz_new = [] 
            grpn = []
            for key in keys: 
                group = groups.get_group(key) 
                cen_cz = group.cz.loc[group.g_galtype == 1].values[0] 
                grpcz_new.append(cen_cz) 
                grpn.append(len(group))

            full_grpcz_arr = np.repeat(grpcz_new, grpn)
            mock_df['grpcz'] = full_grpcz_arr
        else:
            ## Models from Vishnu
            groups = mock_df.groupby(grpid_col) 
            keys = groups.groups.keys() 
            grpcz_new = [] 
            grpn = []
            for key in keys: 
                group = groups.get_group(key)
                cen_cz = group[cencz_col].loc[group[censat_col] == 1].values[0] 
                grpcz_new.append(cen_cz) 
                grpn.append(len(group))

            full_grpcz_arr = np.repeat(grpcz_new, grpn)
            mock_df['grpcz'] = full_grpcz_arr
        return mock_df

    def mock_add_grpcz(self, df, grpid_col=None, galtype_col=None, cen_cz_col=None):
        cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
        # Sum doesn't actually add up anything here but I didn't know how to get
        # each row as is so I used .apply
        cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
            galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
        zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
        a_dictionary = dict(zip_iterator)
        df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)

        av_cz = df.groupby(['{0}'.format(grpid_col)])\
            ['cz'].apply(np.average).values
        zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
        a_dictionary = dict(zip_iterator)
        df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

        return df

    def read_data_catl(self, path_to_file, survey):
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

        z_median: `float`
            Median redshift of survey
        """
        settings = self.settings
        if survey == 'eco':
            # columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag', 
            #             'logmstar', 'logmgas', 'grp', 'grpn', 'logmh', 'logmh_s', 
            #             'fc', 'grpmb', 'grpms','modelu_rcorr', 'grpsig', 'grpsig_stack']

            # # 13878 galaxies
            # eco_buff = pd.read_csv(path_to_file, delimiter=",", header=0, \
            #     usecols=columns)

            eco_buff = self.read_mock_catl(path_to_file)
            #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
            eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

            eco_buff = self.mock_add_grpcz(eco_buff, grpid_col='groupid', 
                galtype_col='g_galtype', cen_cz_col='cz')

            if settings.mf_type == 'smf':
                # 6456 galaxies
                catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) &\
                    (eco_buff.grpcz_new.values <= 7000) &\
                    (eco_buff.absrmag.values <= -17.33)]
            elif settings.mf_type == 'bmf':
                catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) &\
                    (eco_buff.grpcz_new.values <= 7000) &\
                    (eco_buff.absrmag.values <= -17.33)]

            volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
            # cvar = 0.125
            z_median = np.median(catl.grpcz_new.values) / (3 * 10**5)

        elif survey == 'resolvea' or survey == 'resolveb':
            columns = ['name', 'radeg', 'dedeg', 'cz', 'grpcz', 'absrmag',
                    'logmstar', 'logmgas', 'grp', 'grpn', 'grpnassoc', 'logmh',
                    'logmh_s', 'fc', 'grpmb', 'grpms', 'f_a', 'f_b']
            # 2286 galaxies
            resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0, \
                usecols=columns)

            if survey == 'resolvea':
                if settings.mf_type == 'smf':
                    catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) &\
                        (resolve_live18.grpcz.values >= 4500) &\
                        (resolve_live18.grpcz.values <= 7000) &\
                        (resolve_live18.absrmag.values <= -17.33)]
                elif settings.mf_type == 'bmf':
                    catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) &\
                        (resolve_live18.grpcz.values >= 4500) &\
                        (resolve_live18.grpcz.values <= 7000) &\
                        (resolve_live18.absrmag.values <= -17.33)]

                volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
                # cvar = 0.30
                z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

            elif survey == 'resolveb':
                if settings.mf_type == 'smf':
                    # 487 - cz, 369 - grpcz
                    catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) &\
                        (resolve_live18.grpcz.values >= 4500) &\
                        (resolve_live18.grpcz.values <= 7000) &\
                        (resolve_live18.absrmag.values <= -17)]
                elif settings.mf_type == 'bmf':
                    catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) &\
                        (resolve_live18.grpcz.values >= 4500) &\
                        (resolve_live18.grpcz.values <= 7000) &\
                        (resolve_live18.absrmag.values <= -17)]

                volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
                # cvar = 0.58
                z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

        return catl, volume, z_median

    def get_paramvals_percentile(self, mcmc_table, pctl, chi2, randints_df=None):
        """
        Isolates 68th percentile lowest chi^2 values and takes random 100 sample

        Parameters
        ----------
        mcmc_table: pandas.DataFrame
            Mcmc chain dataframe

        pctl: int
            Percentile to use

        chi2: array
            Array of chi^2 values
        
        randints_df (optional): pandas.DataFrame
            Dataframe of mock numbers in case many Behroozi mocks were used.
            Defaults to None.

        Returns
        ---------
        mcmc_table_pctl: pandas dataframe
            Sample of 100 68th percentile lowest chi^2 values
        bf_params: numpy array
            Array of parameter values corresponding to the best-fit model
        bf_chi2: float
            Chi-squared value corresponding to the best-fit model
        bf_randint: int
            In case multiple Behroozi mocks were used, this is the mock number
            that corresponds to the best-fit model. Otherwise, this is not returned.
        """ 
        pctl = pctl/100
        mcmc_table['chi2'] = chi2
        if randints_df is not None: # This returns a bool; True if df has contents
            mcmc_table['mock_num'] = randints_df.mock_num.values.astype(int)
        mcmc_table = mcmc_table.sort_values('chi2').reset_index(drop=True)
        slice_end = int(pctl*len(mcmc_table))
        mcmc_table_pctl = mcmc_table[:slice_end]
        # Best fit params are the parameters that correspond to the smallest chi2
        bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
            values[0][:9]
        bf_chi2 = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
            values[0][9]
        if randints_df is not None:
            bf_randint = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
                values[0][5].astype(int)
            mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)
            return mcmc_table_pctl, bf_params, bf_chi2, bf_randint
        # Randomly sample 100 lowest chi2 
        mcmc_table_pctl = mcmc_table_pctl.drop_duplicates().sample(100)

        return mcmc_table_pctl, bf_params, bf_chi2

    def Load_Into_Dataframes(self):
        settings = self.settings

        print('Reading files')
        # chi2 = pd.read_csv(settings.chi2_file, header=None, names=['chisquared'])\
        #     ['chisquared'].values

        reader = emcee.backends.HDFBackend(settings.chi2_file , read_only=True)
        chi2 = reader.get_blobs(flat=True)

        mcmc_table = self.read_mcmc(settings.chain_file)
        self.catl, self.volume, self.z_median = self.\
            read_data_catl(settings.catl_file, settings.survey)
        ## Group finder run on subset after applying M* cut 8.6 and cz cut 3000-12000
        gal_group = self.read_mock_catl(settings.path_to_proc + \
            "gal_group_run{0}.hdf5".format(settings.run)) 

        # #! Change this if testing with different cz limit
        # gal_group = self.read_mock_catl(settings.path_to_proc + \
        #     "gal_group_run{0}.hdf5".format(settings.run)) 

        idx_arr = np.insert(np.linspace(0,20,21), len(np.linspace(0,20,21)), \
            (22, 123, 124, 125, 126, 127, 128, 129)).\
            astype(int)

        names_arr = [x for x in gal_group.columns.values[idx_arr]]
        for idx in np.arange(2,102,1):
            names_arr.append('{0}_y'.format(idx))
            names_arr.append('groupid_{0}'.format(idx))
            names_arr.append('grp_censat_{0}'.format(idx))
            names_arr.append('cen_cz_{0}'.format(idx))
        names_arr = np.array(names_arr)

        globals.gal_group_df_subset = gal_group[names_arr]

        # Renaming the "1_y" column kept from line 1896 because of case where it was
        # also in mcmc_table_ptcl.mock_num and was selected twice
        globals.gal_group_df_subset.columns.values[25] = "behroozi_bf"

        ### Removing "_y" from column names for stellar mass
        # Have to remove the first element because it is 'halo_y' column name
        cols_with_y = np.array([[idx, s] for idx, s in enumerate(
            globals.gal_group_df_subset.columns.values) if '_y' in s][1:])
        colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
        globals.gal_group_df_subset.columns.values[cols_with_y[:,0].\
            astype(int)] = colnames_without_y

        print('Getting data in specific percentile')
        # get_paramvals called to get bf params and chi2 values
        self.mcmc_table_pctl, self.bf_params, self.bf_chi2 = \
            self.get_paramvals_percentile(mcmc_table, 68, chi2)

        colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter', \
            'mstar_q', 'mh_q', 'mu', 'nu']
        #! Change this if testing with different cz limit
        self.mcmc_table_pctl_subset = pd.read_csv(settings.path_to_proc + 
            'run{0}_params_subset.txt'.format(settings.run), 
            delim_whitespace=True, names=colnames)\
            .iloc[1:,:].reset_index(drop=True)