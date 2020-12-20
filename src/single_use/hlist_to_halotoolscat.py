    """This script is for preparing hlist files to be read by halotools
    """

from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.sim_manager import CachedHaloCatalog
import pandas as pd

rockstar_table = pd.read_table('hlist_1.00000.list.gz',delimiter='\s+',\
compression='gzip',comment='#',usecols=[1,5,6,10,11,17,18,19,20,21,22,39,60,61],\
names=['halo_id','halo_pid','halo_upid','halo_mvir','halo_rvir','halo_x',\
    'halo_y','halo_z','halo_vx','halo_vy','halo_vz','halo_m200b','halo_macc',\
    'halo_mpeak'])

# To convert default rockstar kpc/h units to Mpc/h to avoid bugs with halotools
rockstar_table.halo_rvir = rockstar_table.halo_rvir/1000

redshift = 0
Lbox, particle_mass = 130,3.215e7

halocat = UserSuppliedHaloCatalog(redshift=redshift,
    halo_rvir=rockstar_table.halo_rvir.values,
    halo_vx=rockstar_table.halo_vx.values,
    halo_vy=rockstar_table.halo_vy.values,
    halo_vz=rockstar_table.halo_vz.values,
    halo_upid=rockstar_table.halo_upid.values,
    halo_m200=rockstar_table.halo_m200b.values,
    halo_macc=rockstar_table.halo_macc.values,
    halo_mpeak=rockstar_table.halo_mpeak.values,
    Lbox=Lbox,particle_mass=particle_mass,halo_x=rockstar_table.halo_x.values,
    halo_y=rockstar_table.halo_y.values,halo_z=rockstar_table.halo_z.values,
    halo_id=rockstar_table.halo_id.values,
    halo_pid=rockstar_table.halo_pid.values,
    halo_mvir=rockstar_table.halo_mvir.values)
halocat.add_halocat_to_cache('/home/asadm2/.astropy/cache/halotools/'\
    'halo_catalogs/vishnu/rockstar/vishnu_rockstar_z0.hdf5',simname='vishnu',
    halo_finder='rockstar',version_name='keshawn_v1.0_no_cuts',
    processing_notes='Positions, velocities, radius and mass information'\
    'extracted (mvir,macc,m200,mpeak) from original snapshot',redshift='0')

# Only required to test loading catalog from cache to memory
halocat = CachedHaloCatalog(simname='vishnu', halo_finder='rockstar',
    version_name='keshawn_v1.0_no_cuts', redshift=0,
    update_cached_fname = False)