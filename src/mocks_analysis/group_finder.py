"""
{This script applies redshift-space distortions to Vishnu mock and runs
 group finder so that velocity dispersion measurements can be obtained since
 we are including that as a second observable in constraining red and blue
 SHMRs.}
"""

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import argparse
import emcee
import os

__author__ = '{Mehnaaz Asad}'
__collaborator__ = '{Victor Calderon}'

def pandas_df_to_hdf5_file(data, hdf5_file, key=None, mode='w',
        complevel=8):
        """
        Saves a pandas DataFrame into a normal or a `pandas` hdf5 file.

        Parameters
        ----------
        data: pandas DataFrame object
                DataFrame with the necessary data

        hdf5_file: string
                Path to output file (HDF5 format)

        key: string
                Location, under which to save the pandas DataFrame

        mode: string, optional (default = 'w')
                mode to handle the file.

        complevel: int, range(0-9), optional (default = 8)
                level of compression for the HDF5 file
        """
        ##
        ## Saving DataFrame to HDF5 file
        try:
            data.to_hdf(hdf5_file, key, mode=mode, complevel=complevel)
        except:
            msg = 'Could not create HDF5 file'
            raise ValueError(msg)

def kms_to_Mpc(H0,v):
    return v/H0

def vol_sphere(r):
    volume = (4/3)*np.pi*(r**3)
    return volume

def read_mock_catl(filename, catl_format='.hdf5'):
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

def mock_add_grpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_new'] = df['{0}'.format(grpid_col)].map(a_dictionary)
    return df

def read_data_catl(path_to_file, survey):
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
        Survey catalog with grpcz, abs rmag and stellar mass limits and data 
        in h=0.7

    volume: `float`
        Volume of survey

    cvar: `float`
        Cosmic variance of survey

    z_median: `float`
        Median redshift of survey
    """
    if survey == 'eco':
        # 13878 galaxies
        # eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0)

        eco_buff = read_mock_catl(path_to_file)
        eco_buff = mock_add_grpcz(eco_buff, grpid_col='groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')

        if mf_type == 'smf':
            # 6456 galaxies
            catl = eco_buff.loc[(eco_buff.grpcz_new.values >= 3000) &
                (eco_buff.grpcz_new.values <= 7000) &
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) &
                (eco_buff.grpcz.values <= 7000) &
                (eco_buff.absrmag.values <= -17.33)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        cvar = 0.125
        z_median = np.median(catl.grpcz.values) / (3 * 10**5)

    elif survey == 'resolvea' or survey == 'resolveb':
        # 2286 galaxies
        resolve_live18 = pd.read_csv(path_to_file, delimiter=",", header=0)

        if survey == 'resolvea':
            if mf_type == 'smf':
                catl = resolve_live18.loc[
                    (resolve_live18.grpcz.values >= 4500) &
                    (resolve_live18.grpcz.values <= 7000) &
                    (resolve_live18.absrmag.values <= -17.33)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_a.values == 1) &
                    (resolve_live18.grpcz.values >= 4500) &
                    (resolve_live18.grpcz.values <= 7000) &
                    (resolve_live18.absrmag.values <= -17.33)]

            volume = 13172.384  # Survey volume without buffer [Mpc/h]^3
            cvar = 0.30
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

        elif survey == 'resolveb':
            if mf_type == 'smf':
                # 487 - cz, 369 - grpcz
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) &
                    (resolve_live18.grpcz.values >= 4500) &
                    (resolve_live18.grpcz.values <= 7000) &
                    (resolve_live18.absrmag.values <= -17)]
            elif mf_type == 'bmf':
                catl = resolve_live18.loc[(resolve_live18.f_b.values == 1) &
                    (resolve_live18.grpcz.values >= 4500) &
                    (resolve_live18.grpcz.values <= 7000) &
                    (resolve_live18.absrmag.values <= -17)]

            volume = 4709.8373  # *2.915 #Survey volume without buffer [Mpc/h]^3
            cvar = 0.58
            z_median = np.median(resolve_live18.grpcz.values) / (3 * 10**5)

    return catl,volume,cvar,z_median

def read_chi2(path_to_file):
    """
    Reads chi-squared values from file

    Parameters
    ----------
    path_to_file: string
        Path to chi-squared values file

    Returns
    ---------
    chi2: array
        Array of reshaped chi^2 values to match chain values
    """
    chi2_df = pd.read_csv(path_to_file,header=None,names=['chisquared'])

    # Applies to runs prior to run 5?
    if mf_type == 'smf' and survey == 'eco' and ver==1.0:
        # Needed to reshape since flattened along wrong axis, 
        # didn't correspond to chain
        test_reshape = chi2_df.chisquared.values.reshape((1000,250))
        chi2 = np.ndarray.flatten(np.array(test_reshape),'F')
    
    else:
        chi2 = chi2_df.chisquared.values

    return chi2

def read_mcmc(path_to_file):
    """
    Reads mcmc chain from file

    Parameters
    ----------
    path_to_file: string
        Path to mcmc chain file

    Returns
    ---------
    emcee_table: pandas dataframe
        Dataframe of mcmc chain values with NANs removed
    """
    # colnames = ['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',\
    #     'scatter']
    
    # if mf_type == 'smf' and survey == 'eco' and ver==1.0:
    #     emcee_table = pd.read_csv(path_to_file,names=colnames,sep='\s+',\
    #         dtype=np.float64)

    # else:
    #     emcee_table = pd.read_csv(path_to_file, names=colnames, 
    #         delim_whitespace=True, header=None)

    #     emcee_table = emcee_table[emcee_table.mhalo_c.values != '#']
    #     emcee_table.mhalo_c = emcee_table.mhalo_c.astype(np.float64)
    #     emcee_table.mstellar_c = emcee_table.mstellar_c.astype(np.float64)
    #     emcee_table.lowmass_slope = emcee_table.lowmass_slope.astype(np.float64)

    # # Cases where last parameter was a NaN and its value was being written to 
    # # the first element of the next line followed by 4 NaNs for the other 
    # # parameters
    # for idx,row in enumerate(emcee_table.values):
    #     if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
    #         scatter_val = emcee_table.values[idx+1][0]
    #         row[4] = scatter_val
    
    # # Cases where rows of NANs appear
    # emcee_table = emcee_table.dropna(axis='index', how='any').\
    #     reset_index(drop=True)
    
    colnames = ['mhalo_c', 'mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
        'mstar_q','mh_q','mu','nu']

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

def get_paramvals_percentile(table, percentile, chi2_arr):
    """
    Isolates 68th percentile lowest chi^2 values and takes random 1000 sample

    Parameters
    ----------
    table: pandas dataframe
        Mcmc chain dataframe

    pctl: int
        Percentile to use

    chi2_arr: array
        Array of chi^2 values

    Returns
    ---------
    subset: ndarray
        Random 100 sample of param values from 68th percentile
    """ 
    percentile = percentile/100
    table['chi2'] = chi2_arr
    table = table.sort_values('chi2').reset_index(drop=True)
    slice_end = int(percentile*len(table))
    mcmc_table_pctl = table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    bf_params = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:9]
    # subset = mcmc_table_pctl.drop_duplicates().sample(100).values[:,:5] 
    subset = mcmc_table_pctl.drop_duplicates().sample(100).values[:,:9]
    subset = np.insert(subset, 0, bf_params, axis=0)

    return subset

def halocat_init(halo_catalog, z_median):
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

def populate_mock(theta, model, remove_cols=False):
    """
    Populate mock based on five SMHM parameter values and model

    Parameters
    ----------
    theta: array
        Array of parameter values

    model: halotools model instance
        Model based on behroozi 2010 SMHM

    Returns
    ---------
    gals_df: pandas dataframe
        Dataframe of mock catalog
    """
    """"""

    mhalo_characteristic, mstellar_characteristic, mlow_slope, mhigh_slope,\
        mstellar_scatter = theta
    model.param_dict['smhm_m1_0'] = mhalo_characteristic
    model.param_dict['smhm_m0_0'] = mstellar_characteristic
    model.param_dict['smhm_beta_0'] = mlow_slope
    model.param_dict['smhm_delta_0'] = mhigh_slope
    model.param_dict['scatter_model_param1'] = mstellar_scatter

    model.mock.populate()

    gals = model.mock.galaxy_table
    gals_df = pd.DataFrame(np.array(gals))
    if remove_cols:
        gals_df = gals_df[['halo_mvir_host_halo','stellar_mass']]

    return gals_df

def assign_cen_sat_flag(gals_df):
    """
    Assign centrals and satellites flag to dataframe

    Parameters
    ----------
    gals_df: pandas dataframe
        Mock catalog

    Returns
    ---------
    gals_df: pandas dataframe
        Mock catalog with centrals/satellites flag as new column
    """

    C_S = []
    for idx in range(len(gals_df)):
        if gals_df['halo_hostid'][idx] == gals_df['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)

    C_S = np.array(C_S)
    gals_df['cs_flag'] = C_S
    return gals_df

def cart_to_spherical_coords(cart_arr, dist_arr):
    """
    Computes the right ascension and declination for the given
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """

    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_arr,
        y_arr,
        z_arr) = (cart_arr/np.vstack(dist_arr)).T
    ## Declination
    dec_arr = 90. - np.degrees(np.arccos(z_arr))
    ## Right ascension
    ra_arr = np.ones(len(cart_arr))
    idx_ra_90 = np.where((x_arr==0) & (y_arr>0))
    idx_ra_minus90 = np.where((x_arr==0) & (y_arr<0))
    ra_arr[idx_ra_90] = 90.
    ra_arr[idx_ra_minus90] = -90.
    idx_ones = np.where(ra_arr==1)
    ra_arr[idx_ones] = np.degrees(np.arctan(y_arr[idx_ones]/x_arr[idx_ones]))

    ## Seeing on which quadrant the point is at
    idx_ra_plus180 = np.where(x_arr<0)
    ra_arr[idx_ra_plus180] += 180.
    idx_ra_plus360 = np.where((x_arr>=0) & (y_arr<0))
    ra_arr[idx_ra_plus360] += 360.

    return ra_arr, dec_arr

def apply_rsd(mock_catalog):
    """
    Applies redshift-space distortions

    Parameters
    ----------
    mock_catalog: Pandas dataframe
        Galaxy catalog

    Returns
    ---------
    mock_catalog: Pandas dataframe
        Mock catalog with redshift-space distortions now applied and
        ra,dec,rsd positions and velocity information added
    """

    speed_c = 3*10**5 #km/s
    z_min = 0
    z_max = 0.5
    dz = 10**-3
    H0 = 100
    omega_m = 0.25
    omega_b = 0.04
    Tcmb0 = 2.7255

    redshift_arr = np.arange(z_min,z_max,dz)
    cosmo = LambdaCDM(H0,omega_m,omega_b,Tcmb0)
    como_dist = cosmo.comoving_distance(redshift_arr)
    comodist_z_interp = interp1d(como_dist,redshift_arr)

    cart_gals = mock_catalog[['x','y','z']].values #Mpc/h
    vel_gals = mock_catalog[['vx','vy','vz']].values #km/s

    dist_from_obs = (np.sum(cart_gals**2, axis=1))**.5
    z_cosm_arr  = comodist_z_interp(dist_from_obs)
    cz_cosm_arr = speed_c * z_cosm_arr
    cz_arr  = cz_cosm_arr
    ra_arr, dec_arr = cart_to_spherical_coords(cart_gals,dist_from_obs)
    vr_arr = np.sum(cart_gals*vel_gals, axis=1)/dist_from_obs
    #this cz includes hubble flow and peculiar motion
    cz_arr += vr_arr*(1+z_cosm_arr)

    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr

    return mock_catalog

def group_finding(mock_pd, col_id, mock_zz_file, param_dict, file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to
    galaxy groups
    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the
        galaxies that made it into the catalogue
    mock_zz_file: string
        path to the galaxy catalogue
    param_dict: python dictionary
        dictionary with `project` variables
    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products
    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties
    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Finding ....')
    # Speed of light - in km/s
    speed_c = param_dict['c']
    ##
    ## Running FoF
    # File prefix

    #* Added to make sure that by running the exe from two different screen
    #* sessions simultaneously, the files wouldn't get mixed up when being 
    #* written to in the same location. Filenames in the other session had no i.
    i=1393
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, i, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, i, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, i, file_ext)
    mock_coord_path = '{0}.galcatl_radeccz_{1}.{2}'.format(mock_zz_file, i, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz','{0}'.format(col_id)]]
    mock_coord_pd = mock_coord_pd.rename(columns={"{0}".format(col_id): "logmstar"}).to_csv(mock_coord_path,
        sep=' ', header=None, index=False)
    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    # cu.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)
    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z', 'grp_censat']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
        index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None,
        names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd[['groupid','grp_censat']]], axis=1)
    ## Add cen_cz column from mockgroup_pd to final DF
    mockgal_pd_merged = pd.merge(mockgal_pd_merged, mockgroup_pd[['groupid','cen_cz']], on="groupid")    
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1

    rename_dict = {'groupid': 'groupid_{0}'.format(col_id),
                    'grp_censat': 'grp_censat_{0}'.format(col_id),
                    'cen_cz': 'cen_cz_{0}'.format(col_id)}
    mockgal_pd_merged.rename(columns=rename_dict, inplace=True)

    ## Removing FoF files
    if param_dict['verbose']:
        print('Removing group-finding related files')
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    if param_dict['verbose']:
        print('Group Finding ....Done')

    return mockgal_pd_merged, mockgroup_pd

def abundance_matching_f(dict1, dict2, dict1_names=None, dict2_names=None,
    volume1=None, volume2=None, reverse=True, dens1_opt=False,
    dens2_opt=False):
    """
    Performs abundance matching based on two quantities (dict1 and dict2).
    It assigns values from `dict2` to elements in `dict1`.
    Parameters
    ----------
    dict1: dictionary_like, or array_like
        Dictionary or array-like object of property 1
        If `dens1_opt == True`:
            - Object is a dictionary consisting of the following keys:
                - 'dict1_names': shape (2,)
                - Order of `dict1_names`: [var1_value, dens1_value]
        else:
            - Object is a 1D array or list
            - Density must be calculated for var2
    dict2: dictionary_like, or array_like
        Dictionary or array-like object of property 2
        If `dens2_opt == True`:
            - Object is a dictionary consisting of the following keys:
                - 'dict2_names': shape (2,)
                - Order of `dict2_names`: [var2_value, dens2_value]
        else:
            - Object is a 1D array or list
            - Density must be calculated for var2
    dict1_names: NoneType, or array_list with shape (2,), optional (def: None)
        names of the `dict1` keys, in order of [var1_value, dens1_value]
    dict2_names: NoneType, or array_list with shape (2,), optional (def: None)
        names of the `dict2` keys, in order of [var2_value, dens2_value]
    volume1: NoneType or float, optional (default = None)
        Volume of the 1st variable `var1`
        Required if `dens1_opt == False`
    volume2: NoneType or float, optional (default = None)
        Volume of the 2nd variable `var2`
        Required if `dens2_opt == False`
    reverse: boolean
        Determines the relation between var1 and var2.
        - reverse==True: var1 increases with increasing var2
        - reverse==False: var1 decreases with increasing var2
    dens1_opt: boolean, optional (default = False)
        - If 'True': density is already provided as key for `dict1`
        - If 'False': density must me calculated
        - `dict1_names` must be provided and have length (2,)
    dens2_opt: boolean, optional (default = False)
        - If 'True': density is already provided as key for `dict2`
        - If 'False': density must me calculated
        - `dict2_names` must be provided and have length (2,)
    Returns
    -------
    var1_ab: array_like
        numpy.array of elements matching those of `dict1`, after matching with
        dict2.
    """
    ## Checking input parameters
    # 1st property
    if dens1_opt:
        assert(len(dict1_names) == 2)
        var1  = np.array(dict1[dict1_names[0]])
        dens1 = np.array(dict1[dict1_names[1]])
    else:
        var1        = np.array(dict1)
        assert(volume1 != None)
        ## Calculating Density for `var1`
        if reverse:
            ncounts1 = np.array([np.where(var1<xx)[0].size for xx in var1])+1
        else:
            ncounts1 = np.array([np.where(var1>xx)[0].size for xx in var1])+1
        dens1 = ncounts1.astype(float)/volume1
    # 2nd property
    if dens2_opt:
        assert(len(dict2_names) == 2)
        var2  = np.array(dict2[dict2_names[0]])
        dens2 = np.array(dict2[dict2_names[1]])
    else:
        var2        = np.array(dict2)
        assert(volume2 != None)
        ## Calculating Density for `var1`
        if reverse:
            ncounts2 = np.array([np.where(var2<xx)[0].size for xx in var2])+1
        else:
            ncounts2 = np.array([np.where(var2>xx)[0].size for xx in var2])+1
        dens2 = ncounts2.astype(float)/volume2
    ##
    ## Interpolating densities and values
    interp_var2 = interp1d(dens2, var2, bounds_error=True,assume_sorted=False)
    # Value assignment
    var1_ab = np.array([interp_var2(xx) for xx in dens1])

    return var1_ab

def group_mass_assignment(mockgal_pd, mockgroup_pd, param_dict):
    """
    Assigns a theoretical halo mass to the group based on a group property
    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID
    mockgroup_pd: pandas DataFrame
        DataFame containing information for each galaxy group
    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    -----------
    mockgal_pd_new: pandas DataFrame
        Original info + abundance matched mass of the group, M_group
    mockgroup_pd_new: pandas DataFrame
        Original info of `mockgroup_pd' + abundance matched mass, M_group
    """
    ## Constants
    if param_dict['verbose']:
        print('Group Mass Assign. ....')
    ## Copies of DataFrames
    gal_pd   = mockgal_pd.copy()
    group_pd = mockgroup_pd.copy()

    ## Changing stellar mass to log
    gal_pd['logmstar'] = np.log10(gal_pd['stellar_mass'])

    ## Constants
    Cens     = int(1)
    Sats     = int(0)
    n_gals   = len(gal_pd  )
    n_groups = len(group_pd)
    ## Type of abundance matching
    if param_dict['catl_type'] == 'mr':
        prop_gal    = 'M_r'
        reverse_opt = True
    elif param_dict['catl_type'] == 'mstar':
        prop_gal    = 'logmstar'
        reverse_opt = False
    # Absolute value of `prop_gal`
    prop_gal_abs = prop_gal + '_abs'
    ##
    ## Selecting only a `few` columns
    # Galaxies
    gal_pd = gal_pd.loc[:,[prop_gal, 'groupid']]
    # Groups
    group_pd = group_pd[['ngals']]
    ##
    ## Total `prop_gal` for groups
    group_prop_arr = [[] for x in range(n_groups)]
    ## Looping over galaxy groups
    # Mstar-based
    if param_dict['catl_type'] == 'mstar':
        for group_zz in tqdm(range(n_groups)):
            ## Stellar mass
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal]
            group_log_prop_tot = np.log10(np.sum(10**group_prop))
            ## Saving to array
            group_prop_arr[group_zz] = group_log_prop_tot
    # Luminosity-based
    elif param_dict['catl_type'] == 'mr':
        for group_zz in tqdm(range(n_groups)):
            ## Total abs. magnitude of the group
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal]
            group_prop_tot = Mr_group_calc(group_prop)
            ## Saving to array
            group_prop_arr[group_zz] = group_prop_tot
    ##
    ## Saving to DataFrame
    group_prop_arr            = np.asarray(group_prop_arr)
    group_pd.loc[:, prop_gal] = group_prop_arr
    if param_dict['verbose']:
        print('Calculating group masses...Done')
    ##
    ## --- Halo Abundance Matching --- ##
    ## Mass function for given cosmology
    path_to_hmf = '/fs1/caldervf/Repositories/Large_Scale_Structure/ECO/'\
        'ECO_Mocks_Catls/data/interim/MF/Planck/ECO/Planck_H0_100.0_HMF_warren.csv'

    hmf_pd = pd.read_csv(path_to_hmf, sep=',')

    ## Halo mass
    Mh_ab = abundance_matching_f(group_prop_arr,
                                    hmf_pd,
                                    volume1=param_dict['survey_vol'],
                                    reverse=reverse_opt,
                                    dict2_names=['logM', 'ngtm'],
                                    dens2_opt=True)
    # Assigning to DataFrame
    group_pd.loc[:, 'M_group'] = Mh_ab
    ###
    ### ---- Galaxies ---- ###
    # Adding `M_group` to galaxy catalogue
    gal_pd = pd.merge(gal_pd, group_pd[['M_group', 'ngals']],
                        how='left', left_on='groupid', right_index=True)
    # Renaming `ngals` column
    gal_pd = gal_pd.rename(columns={'ngals':'g_ngal'})
    #
    # Selecting `central` and `satellite` galaxies
    gal_pd.loc[:, prop_gal_abs] = np.abs(gal_pd[prop_gal])
    gal_pd.loc[:, 'g_galtype']  = np.ones(n_gals).astype(int)*Sats
    g_galtype_groups            = np.ones(n_groups)*Sats
    ##
    ## Looping over galaxy groups
    for zz in tqdm(range(n_groups)):
        gals_g = gal_pd.loc[gal_pd['groupid']==zz]
        ## Determining group galaxy type
        gals_g_max = gals_g.loc[gals_g[prop_gal_abs]==gals_g[prop_gal_abs].max()]
        g_galtype_groups[zz] = int(np.random.choice(gals_g_max.index.values))
    g_galtype_groups = np.asarray(g_galtype_groups).astype(int)
    ## Assigning group galaxy type
    gal_pd.loc[g_galtype_groups, 'g_galtype'] = Cens
    ##
    ## Dropping columns
    # Galaxies
    gal_col_arr = [prop_gal, prop_gal_abs, 'groupid']
    gal_pd      = gal_pd.drop(gal_col_arr, axis=1)
    # Groups
    group_col_arr = ['ngals']
    group_pd      = group_pd.drop(group_col_arr, axis=1)
    ##
    ## Merging to original DataFrames
    # Galaxies
    mockgal_pd_new = pd.merge(mockgal_pd, gal_pd, how='left', left_index=True,
        right_index=True)
    # Groups
    mockgroup_pd_new = pd.merge(mockgroup_pd, group_pd, how='left',
        left_index=True, right_index=True)
    if param_dict['verbose']:
        print('Group Mass Assign. ....Done')

    return mockgal_pd_new, mockgroup_pd_new

def group_mass_assignment_rev(mockgal_pd, mockgroup_pd, param_dict, mock_num):
    """
    Assigns a theoretical halo mass to the group based on a group property
    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID
    mockgroup_pd: pandas DataFrame
        DataFame containing information for each galaxy group
    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    -----------
    mockgal_pd_new: pandas DataFrame
        Original info + abundance matched mass of the group, M_group
    mockgroup_pd_new: pandas DataFrame
        Original info of `mockgroup_pd' + abundance matched mass, M_group
    """
    ## Constants
    # if param_dict['verbose']:
    #     print('Group Mass Assign. ....')
    ## Copies of DataFrames
    gal_pd   = mockgal_pd.copy()
    group_pd = mockgroup_pd.copy()

    ## Changing stellar mass to log
    gal_pd['{0}'.format(mock_num)] = np.log10(gal_pd['{0}'.format(mock_num)])

    ## Constants
    Cens     = int(1)
    Sats     = int(0)
    n_gals   = len(gal_pd)
    n_groups = len(group_pd)
    ## Type of abundance matching
    if param_dict['catl_type'] == 'mr':
        prop_gal    = 'M_r'
        reverse_opt = True
    elif param_dict['catl_type'] == 'mstar':
        prop_gal    = '{0}'.format(mock_num)
        reverse_opt = False
    # Absolute value of `prop_gal`
    prop_gal_abs = prop_gal + '_abs'
    ##
    ## Selecting only a `few` columns
    # Galaxies
    gal_pd = gal_pd.loc[:,[prop_gal, 'groupid', 'index']]
    # Groups
    group_pd = group_pd[['ngals']]
    
    gal_pd = pd.merge(gal_pd, group_pd[['ngals']],
                how='left', left_on='groupid', right_index=True)
    # Renaming `ngals` column
    gal_pd = gal_pd.rename(columns={'ngals':'g_ngal_{0}'.format(mock_num)})
    
    #
    # Selecting `central` and `satellite` galaxies
    gal_pd.loc[:, prop_gal_abs] = np.abs(gal_pd[prop_gal])
    gal_pd.loc[:, 'g_galtype_{0}'.format(mock_num)] = np.ones(n_gals).astype(int)*Sats
    g_galtype_groups = np.ones(n_groups)*Sats
    ##
    ## Looping over galaxy groups
    for zz in tqdm(range(n_groups)):
        gals_g = gal_pd.loc[gal_pd['groupid']==zz]
        ## Determining group galaxy type
        gals_g_max = gals_g.loc[gals_g[prop_gal_abs]==gals_g[prop_gal_abs].max()]
        g_galtype_groups[zz] = int(np.random.choice(gals_g_max.index.values))
    g_galtype_groups = np.asarray(g_galtype_groups).astype(int)

    ## Assigning group galaxy type
    gal_pd.loc[g_galtype_groups, 'g_galtype_{0}'.format(mock_num)] = Cens
    
    ## Renaming 'groupid' column
    gal_pd = gal_pd.rename(columns={'groupid':'groupid_{0}'.format(mock_num)})
    ## Dropping 'prop_gal_abs' column
    gal_pd = gal_pd.loc[:,['{0}'.format(mock_num), \
        'g_galtype_{0}'.format(mock_num), 'groupid_{0}'.format(mock_num), \
        'g_ngal_{0}'.format(mock_num), 'index']]
    ##
    ## Dropping columns
    # Galaxies
    # gal_col_arr = [prop_gal, prop_gal_abs, 'groupid']
    # gal_pd_temp      = gal_pd_temp.drop(gal_col_arr, axis=1)
    # Groups
    # group_col_arr = ['ngals']
    # group_pd_temp      = group_pd_temp.drop(group_col_arr, axis=1)
        
    ## Merging to original DataFrames
    # Galaxies
    # col_idxs_new = ['g_galtype_'+str(int(x)) for x in np.linspace(1,101,101)] 
    # mockgal_pd_new = pd.merge(mockgal_pd, gal_pd[col_idxs_new], how='left', left_index=True,
    #     right_index=True)
    # Groups
    # mockgroup_pd_new = pd.merge(mockgroup_pd, group_pd, how='left',
    #     left_index=True, right_index=True)
    # if param_dict['verbose']:
    #     print('Group Mass Assign. ....Done')

    return gal_pd, group_pd

def diff_smf_mod(mstar_arr, volume, h1_bool, colour_flag=False):
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
    """
    if not h1_bool:
        # changing from h=0.7 to h=1 assuming h^-2 dependence
        logmstar_arr = np.log10((10**mstar_arr) / 2.041)
    else:
        logmstar_arr = np.log10(mstar_arr)

    if survey == 'eco' or survey == 'resolvea':
        bin_min = np.round(np.log10((10**8.9) / 2.041), 1)
        if survey == 'eco' and colour_flag == 'R':
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 6
        elif survey == 'eco' and colour_flag == 'B':
            bin_max = np.round(np.log10((10**11) / 2.041), 1)
            bin_num = 6
        elif survey == 'resolvea':
            # different to avoid nan in inverse corr mat
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7
        else:
            bin_max = np.round(np.log10((10**11.5) / 2.041), 1)
            bin_num = 7
        bins = np.linspace(bin_min, bin_max, bin_num)
    elif survey == 'resolveb':
        bin_min = np.round(np.log10((10**8.7) / 2.041), 1)
        bin_max = np.round(np.log10((10**11.8) / 2.041), 1)
        bins = np.linspace(bin_min, bin_max, 7)
    # Unnormalized histogram and bin edges
    counts, edg = np.histogram(logmstar_arr[~np.isnan(logmstar_arr)], bins=bins)
    # counts, edg = np.histogram(logmstar_arr, bins=bins)  # paper used 17 bins
    dm = edg[1] - edg[0]  # Bin width
    maxis = 0.5 * (edg[1:] + edg[:-1])  # Mass axis i.e. bin centers
    # Normalized to volume and bin width
    err_poiss = np.sqrt(counts) / (volume * dm)
    err_tot = err_poiss
    phi = counts / (volume * dm)  # not a log quantity

    phi = np.log10(phi)

    return maxis, phi, err_tot, bins, counts

def args_parser():
    """
    Parsing arguments passed to script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=int, nargs='?', help='Chain number')
    args = parser.parse_args()
    return args

def main(args):
    global survey
    global mf_type
    global ver
    survey = 'eco'
    mf_type = 'smf'
    machine = 'bender'
    ver = 2.0
    run = args.run

    H0 = 100 # (km/s)/Mpc
    cz_inner = 2530 # not starting at corner of box
    # cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box

    dist_inner = kms_to_Mpc(H0,cz_inner) #Mpc/h
    dist_outer = kms_to_Mpc(H0,cz_outer) #Mpc/h

    v_inner = vol_sphere(dist_inner)
    v_outer = vol_sphere(dist_outer)

    v_sphere = v_outer-v_inner
    survey_vol = v_sphere/8

    eco = {
        'c': 3*10**5,
        'survey_vol': survey_vol,
        'min_cz' : cz_inner,
        'max_cz' : cz_outer,
        'zmin': cz_inner/(3*10**5),
        'zmax': cz_outer/(3*10**5),
        'l_perp': 0.07,
        'l_para': 1.1,
        'nmin': 1,
        'verbose': True,
        'catl_type': 'mstar'
    }

    # Changes string name of survey to variable so that the survey dict can
    # be accessed
    param_dict = vars()[survey]

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_raw = dict_of_paths['raw_dir']
    path_to_data = dict_of_paths['data_dir']
    path_to_processed = dict_of_paths['proc_dir']

    if machine == 'bender':
        halo_catalog = '/home/asadm2/.astropy/cache/halotools/halo_catalogs/'\
                    'vishnu/rockstar/vishnu_rockstar_test.hdf5'
    elif machine == 'mac':
        halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'

    if survey == 'eco':
        # catl_file = path_to_raw + "eco/eco_all.csv"
        catl_file = path_to_processed + "gal_group_eco_data_buffer.hdf5"

    # chi2_file = path_to_processed + 'smhm_colour_run35/{0}_colour_chi2.txt'.format(survey)
    # if mf_type == 'smf' and survey == 'eco' and ver == 1.0:
    #     chain_file = path_to_processed + 'mcmc_{0}.dat'.format(survey)
    # else:
    #     chain_file = path_to_processed + 'smhm_colour_run35/mcmc_{0}_colour_raw.txt'.\
    #         format(survey)

    print('Reading chi-squared file')
    # chi2 = read_chi2(chi2_file)
    reader = emcee.backends.HDFBackend(
        path_to_processed + "smhm_colour_run{0}/chain.h5".format(run), read_only=True)
    chi2 = reader.get_blobs(flat=True)

    print('Reading mcmc chain file')
    # mcmc_table = read_mcmc(chain_file)
    names=['Mhalo_c', 'Mstar_c', 'mlow_slope', 'mhigh_slope', 'scatter',
            'Mstar_q','Mhalo_q','mu','nu']
    flatchain = reader.get_chain(flat=True)
    mcmc_table = pd.DataFrame(flatchain, columns=names)

    print('Getting subset of 100 Behroozi parameters')
    mcmc_table_subset = get_paramvals_percentile(mcmc_table, 68, chi2)

    params_df = pd.DataFrame(mcmc_table_subset)
    params_df.to_csv(path_to_processed + 'run{0}_params_subset.txt'.format(run), 
        header=None, index=None, sep=' ', mode='w')

    print('Reading survey data')
    # No M* cut
    catl, volume, cvar, z_median = read_data_catl(catl_file, survey)

    print('Populating halos')
    # Populating halos with best fit set of params
    model_init = halocat_init(halo_catalog, z_median)
    bf_params = mcmc_table_subset[0][:5]
    gals_df_ = populate_mock(bf_params, model_init, False)
    gals_df_['cs_flag'] = np.where(gals_df_['halo_hostid'] == \
        gals_df_['halo_id'], 1, 0)
    gals_df_ = gals_df_.rename(columns={"stellar_mass": "1"})
    gals_df_ = gals_df_.sort_values(by='halo_mvir_host_halo')


    # Populating 100 out of the 101 set of params
    i=2
    for params in mcmc_table_subset[1:]:
        print(i)
        params = params[:5]
        mock = populate_mock(params, model_init, True)
        mock = mock.sort_values(by='halo_mvir_host_halo')
        gals_df_['{0}'.format(i)] = mock.stellar_mass.values
        i+=1
    gals_df_.reset_index(inplace=True, drop=True)

    # def mp_func(a_list):
    #     for theta in enumerate(a_list):
    #         params = theta[:5]
    #         mock = populate_mock(params, model_init)
    #         mock = mock.sort_values(by='halo_mvir_host_halo')
    #         gals_arr = np.insert(gals_arr, mock.stellar_mass.values, axis=1)

    # chunks = np.array([mcmc_table_subset_test[1:][i::5] \
    #     for i in range(5)])
    # from multiprocessing import Pool
    # nproc = 4
    # pool = Pool(processes=nproc)
    # global gals_arr 
    # gals_arr = np.zeros(shape=(7450441,10))

    # pool.map(mp_func, chunks)

    h5File = path_to_processed + "mocks101_run{0}.h5".format(run)
    gals_df_.to_hdf(h5File, "/gals_df_/d1")

    print('Applying RSD')
    gals_df_ = pd.read_hdf(h5File, "/gals_df_/d1")
    gals_rsd_df = apply_rsd(gals_df_)

    print('Applying velocity and stellar mass cuts')
    col_idxs = [str(int(x)) for x in np.linspace(1,101,101)]   
    gals_rsd_subset_df = gals_rsd_df.loc[(gals_rsd_df.cz >= cz_inner) & \
        (gals_rsd_df.cz <= cz_outer)].reset_index(drop=True)
    
    for col in col_idxs:
        print('{0} out of {1}'.format(col, len(col_idxs)))
        # Keep track of index from gals_rsd_subset_df
        gals_rsd_grpfinder_df = gals_rsd_subset_df.loc[gals_rsd_subset_df\
            [col]>10**8.6][['{0}'.format(col),'ra','dec','cz']].\
            reset_index(drop=False)
        # * Make sure that df that group finding is run on has its indices reset
        gal_group_df, group_df = group_finding(gals_rsd_grpfinder_df, col, 
            path_to_data + 'interim/', param_dict)
        ## No need to run this function since group_finding now returns 
        ## grp_censat flag
        # gal_group_df_new, group_df_new = \
        #     group_mass_assignment_rev(gal_group_df, group_df, param_dict, col)
        gals_rsd_subset_df = pd.merge(gals_rsd_subset_df, gal_group_df, 
            how='left', left_on = gals_rsd_subset_df.index, right_on='index')

        rename_dict = {'ra_x': 'ra',
                        'dec_x': 'dec',
                        'cz_x': 'cz'}
        gals_rsd_subset_df.rename(columns=rename_dict, inplace=True)
        ## 'index_x' and 'index_y' are only created when col > 1. ra, dec and cz
        ## are calculated for all 101 mocks and the information is only needed
        ## for group finding so those columns can be dropped.
        if int(col) > 1:
            gals_rsd_subset_df.drop(columns=['index_x','index_y','ra_y','dec_y',
                'cz_y'], inplace=True)

    print('Writing to output files')
    pandas_df_to_hdf5_file(data=gals_rsd_subset_df,
        hdf5_file=path_to_processed + 'gal_group_run{0}.hdf5'.format(run), 
        key='gal_group_df')
    # pandas_df_to_hdf5_file(data=group_df_new,
    #     hdf5_file=path_to_processed + 'group.hdf5', key='group_df')

# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args)