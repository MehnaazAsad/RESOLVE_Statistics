"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from cosmo_utils.utils import work_paths as cwpaths


class Settings():

    def __init__(self) -> None:
        self.quenching = 'hybrid'
        self.machine = 'mac'
        self.survey = 'eco'
        self.mf_type = 'bmf'
        self.level = 'group'
        self.pca = True
        self.stacked_stat = True
        self.nproc = 2
        self.run = 0
        self.many_behroozi_mocks = False
        self.halo_catalog = None
        self.chi2_file = None
        self.chain_file = None
        self.catl_file = None
        self.path_to_mocks = None
        self.path_to_proc = None

    def Initalize_Global_Settings(self):
        dict_of_paths = cwpaths.cookiecutter_paths()
        path_to_raw = dict_of_paths['raw_dir']
        path_to_data = dict_of_paths['data_dir']
        self.path_to_proc = dict_of_paths['proc_dir']

        if self.machine == 'bender':
            self.halo_catalog = '/home/asadm2/.astropy/cache/halotools/'\
                'halo_catalogs/vishnu/rockstar/vishnu_rockstar_test.hdf5'
        elif self.machine == 'mac':
            self.halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

        if self.quenching == 'halo':
            self.run = 110
        elif self.quenching == 'hybrid':
            if self.mf_type == 'smf':
                self.run = 108
            elif self.mf_type == 'bmf':
                self.run = 109

        if self.run >= 37:
            if self.mf_type == 'smf':
                self.chi2_file = self.path_to_proc + \
                    'smhm_colour_run{0}/chain.h5'.\
                    format(self.run, self.survey)
                self.chain_file = self.path_to_proc + \
                    'smhm_colour_run{0}/chain.h5'.\
                    format(self.run, self.survey)
            elif self.mf_type == 'bmf':
                self.chi2_file = self.path_to_proc + \
                    'bmhm_colour_run{0}/chain.h5'.\
                    format(self.run, self.survey)
                self.chain_file = self.path_to_proc + \
                    'bmhm_colour_run{0}/chain.h5'.\
                    format(self.run, self.survey)
        elif self.run < 37:
            self.chi2_file = self.path_to_proc + \
                'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
                format(self.run, self.survey)
            self.chain_file = self.path_to_proc + \
                'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
                format(self.run, self.survey)

        if self.survey == 'eco':
            if self.mf_type == "smf":
            # catl_file = path_to_raw + "eco/eco_all.csv"
            ## Updated catalog with group finder run on subset after applying M* 
            # and cz cuts: changed volume to be in h=1 instead of 0.7
                self.catl_file = self.path_to_proc + \
                    "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
            elif self.mf_type == "bmf":
                self.catl_file = self.path_to_proc + \
                    "gal_group_eco_bary_buffer_volh1_dr3.hdf5"
            self.path_to_mocks = path_to_data + 'mocks/m200b/eco/'
        elif self.survey == 'resolvea' or self.survey == 'resolveb':
            self.catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"