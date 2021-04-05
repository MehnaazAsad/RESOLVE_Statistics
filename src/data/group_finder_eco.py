"""
{This script runs group finder on ECO data after having applied M* and cz cuts.}
"""

from cosmo_utils.utils import work_paths as cwpaths
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import os

__author__ = '{Mehnaaz Asad}'


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

    # ## Changing stellar mass to log
    # gal_pd['logmstar'] = np.log10(gal_pd['stellar_mass'])

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


survey = 'eco'
cz_inner = 3000
cz_outer = 7000
volume = 151829.26 * 2.915 # survey volume without buffer in h=0.7

eco = {
    'c': 3*10**5,
    'survey_vol': volume,
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

param_dict = vars()[survey]

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_processed = dict_of_paths['proc_dir']

catl_file = path_to_raw + "eco/eco_all.csv"

eco_buff = pd.read_csv(catl_file,delimiter=",", header=0) 

eco_subset_df = eco_buff.loc[(eco_buff.cz >= cz_inner) & \
    (eco_buff.cz <= cz_outer) & (eco_buff.logmstar >= 8.9)].\
    reset_index(drop=True)

eco_subset_df = eco_subset_df.rename(columns={'radeg':'ra'})
eco_subset_df = eco_subset_df.rename(columns={'dedeg':'dec'})

# * Make sure that df that group finding is run on has its indices reset
gal_group_df, group_df = group_finding(eco_subset_df,
    path_to_data + 'interim/', param_dict)
gal_group_df_new, group_df_new = \
    group_mass_assignment(gal_group_df, group_df, param_dict)

print('Writing to output files')
pandas_df_to_hdf5_file(data=gal_group_df_new,
    hdf5_file=path_to_processed + 'gal_group_eco_data_vol_update.hdf5', key='gal_group_df')