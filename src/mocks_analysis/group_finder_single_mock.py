"""
{This script applies redshift-space distortions to a single Vishnu mock and runs
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
        eco_buff = pd.read_csv(path_to_file,delimiter=",", header=0)

        if mf_type == 'smf':
            # 6456 galaxies
            catl = eco_buff.loc[(eco_buff.grpcz.values >= 3000) &
                (eco_buff.grpcz.values <= 7000) &
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
                    (resolve_live18.absrmag.values <= -17.33) &
                    (resolve_live18.logmstar.values >= 8.9)]
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
                    (resolve_live18.absrmag.values <= -17) &
                    (resolve_live18.logmstar.values >= 8.7)]
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
    colnames = ['mhalo_c','mstellar_c','lowmass_slope','highmass_slope',\
        'scatter']
    
    if mf_type == 'smf' and survey == 'eco' and ver==1.0:
        emcee_table = pd.read_csv(path_to_file,names=colnames,sep='\s+',\
            dtype=np.float64)

    else:
        emcee_table = pd.read_csv(path_to_file, names=colnames, 
            delim_whitespace=True, header=None)

        emcee_table = emcee_table[emcee_table.mhalo_c.values != '#']
        emcee_table.mhalo_c = emcee_table.mhalo_c.astype(np.float64)
        emcee_table.mstellar_c = emcee_table.mstellar_c.astype(np.float64)
        emcee_table.lowmass_slope = emcee_table.lowmass_slope.astype(np.float64)

    # Cases where last parameter was a NaN and its value was being written to 
    # the first element of the next line followed by 4 NaNs for the other 
    # parameters
    for idx,row in enumerate(emcee_table.values):
        if np.isnan(row)[4] == True and np.isnan(row)[3] == False:
            scatter_val = emcee_table.values[idx+1][0]
            row[4] = scatter_val
    
    # Cases where rows of NANs appear
    emcee_table = emcee_table.dropna(axis='index', how='any').\
        reset_index(drop=True)
    
    return emcee_table

def get_paramvals_percentile(table, chi2_arr, percentile1, percentile2=None):
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
        Random sample of param values between 68th and 95th percentile
    """ 
    table['chi2'] = chi2_arr
    percentile1 = percentile1/100
    slice_end = int(percentile1*len(table))
    table = table.sort_values('chi2').reset_index(drop=True)
    mcmc_table_pctl = table[:slice_end]
    # Best fit params are the parameters that correspond to the smallest chi2
    subset = mcmc_table_pctl.drop_duplicates().reset_index(drop=True).\
        values[0][:5]

    if percentile2:
        percentile2 = percentile2/100
        mcmc_table_pctl = table.loc[(table.chi2 > 
            np.percentile(table.chi2, 68)) & (table.chi2 <= np.percentile(table\
            .chi2, 95))].reset_index(drop=True)
        subset = mcmc_table_pctl.drop_duplicates().sample(1).values[:,:5][0]
        print("Behroozi parameters: {0}".format(subset))

    return subset

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

def populate_mock(theta, model):
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

    if survey == 'eco' or survey == 'resolvea':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.9) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.4) / 2.041), 1)
    elif survey == 'resolveb':
        if mf_type == 'smf':
            limit = np.round(np.log10((10**8.7) / 2.041), 1)
        elif mf_type == 'bmf':
            limit = np.round(np.log10((10**9.1) / 2.041), 1)
    sample_mask = model.mock.galaxy_table['stellar_mass'] >= 10**limit
    gals = model.mock.galaxy_table[sample_mask]
    gals_df = pd.DataFrame(np.array(gals))

    return gals_df

def cart_to_spherical_coords(cart_arr, dist):
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
    (   x_val,
        y_val,
        z_val) = cart_arr/float(dist)
    # Distance to object
    dist = float(dist)
    ## Declination
    dec_val = 90. - np.degrees(np.arccos(z_val))
    ## Right ascension
    if x_val == 0:
        if y_val > 0.:
            ra_val = 90.
        elif y_val < 0.:
            ra_val = -90.
    else:
        ra_val = np.degrees(np.arctan(y_val/x_val))

    ## Seeing on which quadrant the point is at
    if x_val < 0.:
        ra_val += 180.
    elif (x_val >= 0.) and (y_val < 0.):
        ra_val += 360.

    return ra_val, dec_val

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

    ngal = len(mock_catalog)
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

    dist_from_obs_arr = np.zeros(ngal)
    ra_arr = np.zeros(ngal)
    dec_arr = np.zeros(ngal)
    cz_arr = np.zeros(ngal)
    cz_nodist_arr = np.zeros(ngal)
    vel_tan_arr = np.zeros(ngal)
    vel_tot_arr = np.zeros(ngal)
    vel_pec_arr = np.zeros(ngal)
    for x in tqdm(range(ngal)):
        dist_from_obs = (np.sum(cart_gals[x]**2))**.5
        z_cosm = comodist_z_interp(dist_from_obs)
        cz_cosm = speed_c * z_cosm
        cz_val = cz_cosm
        ra,dec = cart_to_spherical_coords(cart_gals[x],dist_from_obs)
        vr = np.dot(cart_gals[x], vel_gals[x])/dist_from_obs
        #this cz includes hubble flow and peculiar motion
        cz_val += vr*(1+z_cosm)
        vel_tot = (np.sum(vel_gals[x]**2))**.5
        vel_tan = (vel_tot**2 - vr**2)**.5
        vel_pec  = (cz_val - cz_cosm)/(1 + z_cosm)
        dist_from_obs_arr[x] = dist_from_obs
        ra_arr[x] = ra
        dec_arr[x] = dec
        cz_arr[x] = cz_val
        cz_nodist_arr[x] = cz_cosm
        vel_tot_arr[x] = vel_tot
        vel_tan_arr[x] = vel_tan
        vel_pec_arr[x] = vel_pec

    mock_catalog['r_dist'] = dist_from_obs_arr
    mock_catalog['ra'] = ra_arr
    mock_catalog['dec'] = dec_arr
    mock_catalog['cz'] = cz_arr
    mock_catalog['cz_nodist'] = cz_nodist_arr
    mock_catalog['vel_tot'] = vel_tot_arr
    mock_catalog['vel_tan'] = vel_tan_arr
    mock_catalog['vel_pec'] = vel_pec_arr

    return mock_catalog

def group_finding(mock_pd, mock_zz_file, param_dict, file_ext='csv'):
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

    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof.{1}'.format(mock_zz_file, file_ext)
    grep_file       = '{0}.galcatl_grep.{1}'.format(mock_zz_file, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g.{1}'.format(mock_zz_file, file_ext)
    mock_coord_path = '{0}.galcatl_radeccz.{1}'.format(mock_zz_file, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)
    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
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
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z']
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
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd['groupid']], axis=1)
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
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
    # Remaining `ngals` column
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

def main():
    global survey
    global mf_type
    global ver
    survey = 'eco'
    mf_type = 'smf'
    machine = 'bender'
    ver=2.0

    H0 = 100 # h(km/s)/Mpc
    cz_inner = 3000 # not starting at corner of box
    cz_outer = 120*H0 # utilizing until 120 Mpc of Vishnu box

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
        catl_file = path_to_raw + "eco/eco_all.csv"

    chi2_file = path_to_processed + 'smhm_run6/{0}_chi2.txt'.format(survey)
    if mf_type == 'smf' and survey == 'eco' and ver == 1.0:
        chain_file = path_to_processed + 'mcmc_{0}.dat'.format(survey)
    else:
        chain_file = path_to_processed + 'smhm_run6/mcmc_{0}_raw.txt'.\
            format(survey)

    print('Reading chi-squared file')
    chi2 = read_chi2(chi2_file)

    print('Reading mcmc chain file')
    mcmc_table = read_mcmc(chain_file)

    print('Getting parameters')
    params = get_paramvals_percentile(mcmc_table, chi2, 68, 95)

    print('Reading survey data')
    catl, volume, cvar, z_median = read_data_catl(catl_file, survey)

    print('Populating halos')
    model_init = halocat_init(halo_catalog, z_median)
    gals_df = populate_mock(params, model_init)
    gals_df = assign_cen_sat_flag(gals_df)

    print('Applying RSD')
    gals_rsd_df = apply_rsd(gals_df)
    gals_rsd_subset_df = gals_rsd_df.loc[(gals_rsd_df.cz >= cz_inner) & \
        (gals_rsd_df.cz <= cz_outer) &
        (gals_rsd_df.stellar_mass >= (10**8.9/2.041))].reset_index(drop=True)

    gal_group_df, group_df = group_finding(gals_rsd_subset_df,
        path_to_data + 'interim/', param_dict)
    gal_group_df_new, group_df_new = \
        group_mass_assignment(gal_group_df, group_df, param_dict)

    print('Writing to output files')
    pandas_df_to_hdf5_file(data=gal_group_df_new,
        hdf5_file=path_to_processed + 'gal_group_95perc_r4.hdf5', key='gal_group_df')
    pandas_df_to_hdf5_file(data=group_df_new,
        hdf5_file=path_to_processed + 'group_95perc_r4.hdf5', key='group_df')

# Main function
if __name__ == '__main__':
    main()