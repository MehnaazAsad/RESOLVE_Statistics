"""
{This script makes a mock using Moster 2013 empirical model.}
"""

from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
import pandas as pd
import numpy as np
import tarfile

__author__ = '{Mehnaaz Asad}'

def assign_cen_sat(data_file):
    """
    Assign central/satellite flag

    Parameters
    ----------
    data_file: pandas DataFrame
        Mock catalog

    Returns
    ---------
    df: pandas DataFrame
        Mock catalog with cs_flag column added
    """
    df = data_file.copy()
    C_S = []
    for idx in range(len(df)):
        if df['halo_hostid'][idx] == df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)
    df['cs_flag'] = C_S
    return df

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_int = dict_of_paths['int_dir']

halo_catalog = path_to_raw + 'vishnu_rockstar_z0.hdf5'

halocat = CachedHaloCatalog(fname=halo_catalog, update_cached_fname=True)
z = halocat.redshift

## Don't need seed here. Running this many times will still return the same
## stellar masses as long as model.mc_stellar_mass() uses a seed
model = Moster13SmHm(redshift=z, prim_haloprop_key='halo_macc')

# mc_stellar_mass method returns different stellar_mass distributions for when 
# scatter_model_param1 is set to 0.2 and 0.001 but mean_stellar_mass returns 
# the same distributions which doesn't make sense
halocat.halo_table['stellar_mass'] = model.mc_stellar_mass( 
    prim_haloprop=halocat.halo_table['halo_macc'], redshift=z, seed=1) 
scatter = model.param_dict['scatter_model_param1']

mock_catalog = pd.DataFrame(np.array(halocat.halo_table[['halo_id','halo_pid',\
    'halo_upid', 'halo_hostid','halo_x','halo_y','halo_z', 'halo_vx', 'halo_vy',\
    'halo_vz', 'halo_mvir', 'halo_macc', 'stellar_mass']]))
mock_catalog = assign_cen_sat(mock_catalog)

mock_catalog.to_hdf(path_to_int+'vishnu_rockstar_moster_z{0}_{1}dex_v3.hdf5'.\
    format(z, scatter), key='\gal_catl', mode='w', complevel=8)

tf = tarfile.open(path_to_int+'vishnu_rockstar_moster_z{0}_{1}dex_v3.tar.gz'.\
    format(z, scatter), mode="w:gz")
tf.add(path_to_int+'vishnu_rockstar_moster_z{0}_{1}dex_v3.hdf5'.format(z, scatter))
tf.close()
