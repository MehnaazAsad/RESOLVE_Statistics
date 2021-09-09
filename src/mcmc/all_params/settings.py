"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from cosmo_utils.utils import work_paths as cwpaths

class Settings():
    quenching = 'hybrid'
    machine = 'mac'
    survey = 'eco'
    mf_type = 'smf'
    run = 0
    halo_catalog = None
    chi2_file = None
    chain_file = None
    catl_file = None
    path_to_mocks = None
    path_to_proc = None

    @staticmethod
    def Initalize_Global_Settings():
        dict_of_paths = cwpaths.cookiecutter_paths()
        path_to_raw = dict_of_paths['raw_dir']
        path_to_data = dict_of_paths['data_dir']
        Settings.path_to_proc = dict_of_paths['proc_dir']

        if Settings.machine == 'bender':
            Settings.halo_catalog = '/home/asadm2/.astropy/cache/halotools/'\
                'halo_catalogs/vishnu/rockstar/vishnu_rockstar_test.hdf5'
        elif Settings.machine == 'mac':
            Settings.halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

        if Settings.quenching == 'halo':
            Settings.run = 33
        elif Settings.quenching == 'hybrid':
            Settings.run = 32

        Settings.chi2_file = Settings.path_to_proc + \
            'smhm_colour_run{0}/{1}_colour_chi2.txt'.\
            format(Settings.run, Settings.survey)
        Settings.chain_file = Settings.path_to_proc + \
            'smhm_colour_run{0}/mcmc_{1}_colour_raw.txt'.\
            format(Settings.run, Settings.survey)

        if Settings.survey == 'eco':
            # catl_file = path_to_raw + "eco/eco_all.csv"
            ## New catalog with group finder run on subset after applying M* 
            # and cz cuts
            Settings.catl_file = Settings.path_to_proc + \
                "gal_group_eco_data_buffer.hdf5"
            Settings.path_to_mocks = path_to_data + 'mocks/m200b/eco/'
        elif Settings.survey == 'resolvea' or Settings.survey == 'resolveb':
            Settings.catl_file = path_to_raw + "RESOLVE_liveJune2018.csv"
