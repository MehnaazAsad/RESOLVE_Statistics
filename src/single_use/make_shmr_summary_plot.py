import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import CachedHaloCatalog
from cosmo_utils.utils import work_paths as cwpaths
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
import pandas as pd
import subprocess
import os
from scipy.stats import binned_statistic as bs

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=30)
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{amsmath}")
rc('axes', linewidth=2)
rc('xtick.major', width=4, size=7)
rc('ytick.major', width=4, size=7)
rc('xtick.minor', width=2, size=7)
rc('ytick.minor', width=2, size=7)

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
    df['grpcz_cen'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def models_add_avgrpcz(df, grpid_col=None, galtype_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)

    av_cz = df.groupby(['{0}'.format(grpid_col)])\
        ['cz'].apply(np.average).values
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(av_cz))
    a_dictionary = dict(zip_iterator)
    df['grpcz_av'] = df['{0}'.format(grpid_col)].map(a_dictionary)

    return df

def models_add_cengrpcz(df, grpid_col=None, galtype_col=None, cen_cz_col=None):
    cen_subset_df = df.loc[df[galtype_col] == 1].sort_values(by=grpid_col)
    # Sum doesn't actually add up anything here but I didn't know how to get
    # each row as is so I used .apply
    cen_cz = cen_subset_df.groupby(['{0}'.format(grpid_col),'{0}'.format(
        galtype_col)])['{0}'.format(cen_cz_col)].apply(np.sum).values    
    zip_iterator = zip(list(cen_subset_df[grpid_col]), list(cen_cz))
    a_dictionary = dict(zip_iterator)
    df['ps_cen_cz'] = df['{0}'.format(grpid_col)].map(a_dictionary)

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

        #* Recommended to exclude this galaxy in erratum to Hood et. al 2018
        eco_buff = eco_buff.loc[eco_buff.name != 'ECO13860']

        eco_buff = mock_add_grpcz(eco_buff, grpid_col='ps_groupid', 
            galtype_col='g_galtype', cen_cz_col='cz')

        if mf_type == 'smf':
            # 6456 galaxies
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) &
                (eco_buff.grpcz_cen.values <= 7000) &
                (eco_buff.absrmag.values <= -17.33)]
        elif mf_type == 'bmf':
            catl = eco_buff.loc[(eco_buff.grpcz_cen.values >= 3000) &
                (eco_buff.grpcz_cen.values <= 7000) &
                (eco_buff.absrmag.values <= -17.33)]

        volume = 151829.26 # Survey volume without buffer [Mpc/h]^3
        # cvar = 0.125
        z_median = np.median(catl.grpcz_cen.values) / (3 * 10**5)

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

    return catl, volume, z_median

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
    subset = mcmc_table_pctl.drop_duplicates().sample(200).values[:,:9]
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

    model.mock.populate(seed=1993)

    gals = model.mock.galaxy_table
    gals_df = pd.DataFrame(np.array(gals))
    if remove_cols:
        gals_df = gals_df[['halo_mvir_host_halo','stellar_mass']]

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

    i=1993
    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof_{1}.{2}'.format(mock_zz_file, i, file_ext)
    grep_file       = '{0}.galcatl_grep_{1}.{2}'.format(mock_zz_file, i, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g_{1}.{2}'.format(mock_zz_file, i, file_ext)
    mock_coord_path = '{0}.galcatl_radeccz_{1}.{2}'.format(mock_zz_file, i, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz', 'logmstar']]
    mock_coord_pd = mock_coord_pd.to_csv(mock_coord_path,
        sep=' ', header=None, index=False)
    # cu.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    # fof_exe = '/fs1/caldervf/custom_utilities_c/group_finder_fof/fof9_ascii'
    # fof_exe = '/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/interim/fof/fof9_ascii'
    fof_exe = '/Users/asadm2/Desktop/fof/fof9_ascii'
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

def group_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    """
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupcz : Redshift velocity in km/s of galaxy i's group center.
    
    Note: the FoF code of AA Berlind uses theta_i = declination, with theta_cen = 
    the central declination. This version uses theta_i = pi/2-dec, with some trig functions
    changed so that the output *matches* that of Berlind's FoF code (my "deccen" is the same as
    his "thetacen", to be exact.)
    -----
    """
    # Prepare cartesian coordinates of input galaxies
    ngalaxies = len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        nmembers = len(galaxygrpid[sel])
        xcen=np.sum(galaxycz[sel]*galaxyx[sel])/nmembers
        ycen=np.sum(galaxycz[sel]*galaxyy[sel])/nmembers
        zcen=np.sum(galaxycz[sel]*galaxyz[sel])/nmembers
        czcen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        deccen = np.arcsin(zcen/czcen)*180.0/np.pi # degrees
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Return counts for binning based on group ID numbers.
    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid = np.unique(grpids)
    if return_by_galaxy:
        grpn_by_gal=np.zeros(len(grpids)).astype(int)
        for idv in grpids:
            sel = np.where(grpids==idv)
            grpn_by_gal[sel]=len(sel[0])
        return grpn_by_gal
    else:
        occurences=[]
        for uid in uniqid:
            sel = np.where(grpids==uid)
            occurences.append(len(grpids[sel]))
        return occurences

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def split_false_pairs(galra, galde, galcz, galgroupid):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract
    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.
    """
    groupra,groupde,groupcz=group_skycoords(galra,galde,galcz,galgroupid)
    groupn = multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    # brokenupids_start = np.max(galgroupid)+1
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    ## parameters corresponding to Katie's dividing line in cz-rproj space
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)

    for ii,gg in enumerate(n2grps):
        # pair of indices where group's ngal == 2
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            # break
            newgroupid[galsel]=brokenupids[galsel]
            # newgroupid[galsel] = np.array([brokenupids_start, brokenupids_start+1])
            # brokenupids_start+=2
        else:
            pass
    return newgroupid 

H0 = 100 # (km/s)/Mpc
cz_inner = 2530 # not starting at corner of box
# cz_inner = 3000 # not starting at corner of box
cz_outer = 120*H0 # utilizing 120 Mpc of Vishnu box
survey = 'eco'
mf_type = 'smf'

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
    'verbose': False,
    'catl_type': 'mstar'
}

# Changes string name of survey to variable so that the survey dict can
# be accessed
param_dict = vars()[survey]


dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_data = dict_of_paths['data_dir']
path_to_processed = dict_of_paths['proc_dir']

halo_catalog = path_to_raw + 'vishnu_rockstar_test.hdf5'
catl_file = path_to_processed + "gal_group_eco_stellar_buffer_volh1_dr3.hdf5"
catl, volume, z_median = read_data_catl(catl_file, survey)
model_init = halocat_init(halo_catalog, z_median)

bf_params_hybrid = [12.64869191, 10.68374592, 0.4640583 , 0.43375466, 0.22177846,
10.18868649, 13.23990274, 0.72632304, 0.05996219]
bf_params_halo = [12.45003142, 10.49533185, 0.42905612, 0.49491889, 0.38661993,
11.928341 , 12.60596691, 1.63365685, 0.35175002]
gals_df_ = populate_mock(bf_params_halo[:5], model_init, False)
gals_df_['cs_flag'] = np.where(gals_df_['halo_hostid'] == \
    gals_df_['halo_id'], 1, 0)
gals_df_.rename(columns={'stellar_mass':'logmstar'}, inplace=True)

gals_df_['logmstar'] = np.log10(gals_df_['logmstar'])

#Sorts in ascending order
gals_df_ = gals_df_.sort_values(by='halo_mvir_host_halo')

print('Applying RSD')
gals_rsd_df = apply_rsd(gals_df_)

print('Applying velocity and stellar mass cuts')
gals_rsd_subset_df = gals_rsd_df.loc[(gals_rsd_df.cz >= cz_inner) & \
    (gals_rsd_df.cz <= cz_outer)].reset_index(drop=True)
gals_rsd_grpfinder_df = gals_rsd_subset_df.loc[gals_rsd_subset_df\
    ['logmstar']>8.6][['logmstar','ra','dec','cz']].\
    reset_index(drop=False)

# * Make sure that df that group finding is run on has its indices reset
gal_group_df, group_df_or = group_finding(gals_rsd_grpfinder_df, 
    path_to_data + 'interim/', param_dict)

#* Pair splitting
psgrpid = split_false_pairs(
    np.array(gal_group_df.ra),
    np.array(gal_group_df.dec),
    np.array(gal_group_df.cz), 
    np.array(gal_group_df.groupid))

gal_group_df["ps_groupid"] = psgrpid

arr1 = gal_group_df["ps_groupid"]
arr1_unq = gal_group_df["ps_groupid"].drop_duplicates()  
arr2_unq = np.arange(len(np.unique(gal_group_df["ps_groupid"]))) 
mapping = dict(zip(arr1_unq, arr2_unq))   
new_values = arr1.map(mapping)
gal_group_df['ps_groupid'] = new_values  

most_massive_gal_idxs = gal_group_df.groupby(['ps_groupid'])['logmstar']\
    .transform(max) == gal_group_df['logmstar']        
grp_censat_new = most_massive_gal_idxs.astype(int)
gal_group_df["ps_grp_censat"] = grp_censat_new

gal_group_df = models_add_cengrpcz(
    gal_group_df, 
    grpid_col='ps_groupid', 
    galtype_col='ps_grp_censat', 
    cen_cz_col='cz')

# Dropping pre-pair splitting groupids and related info and replacing
# with post-pait splitting info
gal_group_df.drop(columns=[
    'groupid',
    'grp_censat',
    'cen_cz'], 
    inplace=True)

rename_dict = {'ps_groupid': 'groupid',
                'ps_grp_censat': 'grp_censat',
                'ps_cen_cz': 'cen_cz'}
gal_group_df.rename(columns=rename_dict, inplace=True)

gals_rsd_subset_df = pd.merge(gals_rsd_subset_df, gal_group_df, 
    how='left', left_on = gals_rsd_subset_df.index.values, right_on='index')

rename_dict = {'ra_x': 'ra',
                'dec_x': 'dec',
                'cz_x': 'cz',}
gals_rsd_subset_df.rename(columns=rename_dict, inplace=True)

cols_with_y = np.array([[idx, s] for idx, s in enumerate(
    gals_rsd_subset_df.columns.values) if '_y' in s][1:])
colnames_without_y = [s.replace("_y", "") for s in cols_with_y[:,1]]
gals_rsd_subset_df.columns.values[cols_with_y[:,0].\
    astype(int)] = colnames_without_y

def H100_to_H70(arr, h_exp):
    #Assuming h^-1 units
    if h_exp == -1:
        result = np.log10((10**arr) * 1.429)
    #Assuming h^-2 units
    elif h_exp == -2:
        result = np.log10((10**arr) * 2.041)
    return result


## Making a similar cz cut as in data which is based on grpcz being 
## defined as cz of the central of the group "grpcz_cen"
cz_inner_mod = 3000
gals_rsd_subset_df = gals_rsd_subset_df.loc[\
    (gals_rsd_subset_df['cen_cz'] >= cz_inner_mod) &
    (gals_rsd_subset_df['cen_cz'] <= cz_outer)].reset_index(drop=True)

extra_halo_masses_df = pd.read_csv('/Users/asadm2/Desktop/extra_halo_masses.csv')
extra_halo_masses_df = extra_halo_masses_df.drop(columns='Unnamed: 0')
subset = extra_halo_masses_df[['halo_id', 'halo_mpeak']]
gals_rsd_subset_df = gals_rsd_subset_df.merge(subset, how='left', on='halo_id')

cen_halos = gals_rsd_subset_df.halo_mvir.loc[gals_rsd_subset_df.cs_flag == 1].values
cen_gals = gals_rsd_subset_df.logmstar.loc[gals_rsd_subset_df.cs_flag == 1].values
cen_gals = H100_to_H70(cen_gals, -2)
cen_halos = H100_to_H70(np.log10(cen_halos), -1)

sat_halos = gals_rsd_subset_df.halo_macc.loc[gals_rsd_subset_df.cs_flag == 0].values
sat_gals = gals_rsd_subset_df.logmstar.loc[gals_rsd_subset_df.cs_flag == 0].values
sat_gals = H100_to_H70(sat_gals, -2)
sat_halos = H100_to_H70(np.log10(sat_halos), -1)

halos = np.hstack((cen_halos, sat_halos))
gals = np.hstack((cen_gals, sat_gals))

mock_ratio = np.log10((10**gals)/(10**halos))

my_binned_shmr = bs(halos, mock_ratio, statistic='median',
    bins=np.linspace(10.6, 14.6, 10))
bin_centers = 0.5 * (my_binned_shmr[1][1:] + my_binned_shmr[1][:-1])


def behroozi10(logmstar, bf_params):
    """ 
    This function calculates the B10 stellar to halo mass relation 
    using the functional form and best-fit parameters 
    https://arxiv.org/pdf/1001.0015.pdf
    """
    M_1, Mstar_0, beta, delta, gamma = bf_params[:5]
    second_term = (beta*np.log10((10**logmstar)/(10**Mstar_0)))
    third_term_num = (((10**logmstar)/(10**Mstar_0))**delta)
    third_term_denom = (1 + (((10**logmstar)/(10**Mstar_0))**(-gamma)))
    logmh = M_1 + second_term + (third_term_num/third_term_denom) - 0.5

    return logmh


mstar_min = 8.9
mstar_max = 11.8
logmstar_behroozi10 = np.linspace(mstar_min, mstar_max, 500)
logmh_behroozi10 = behroozi10(logmstar_behroozi10, bf_params_hybrid)
analytical_ratio = np.log10((10**logmstar_behroozi10)/(10**logmh_behroozi10))

with open('/Users/asadm2/Desktop/behroozi2018_girelli2020_commonshmrs.json', 'r') as f:
  data = json.load(f)

data.keys()
num_shmrs = np.array(data['datasetColl']).shape[0] #(9,) since there are 9 SHMRs 

all_shmr_data = []
names = []
for shmr_idx in range(num_shmrs):
    name = data['datasetColl'][shmr_idx]['name']
    shmr_data = data['datasetColl'][shmr_idx]['data']
    res = np.array(sorted([sub['value'] for sub in shmr_data], key=lambda x: x[0]))
    names.append(name)
    all_shmr_data.append(res)


palette = cm.get_cmap('Paired', num_shmrs)
palette = np.array([[0.65098039, 0.80784314, 0.89019608, 1.        ],
                    [0.12156863, 0.47058824, 0.70588235, 1.        ],
                    [0.2       , 0.62745098, 0.17254902, 1.        ],
                    [0.98431373, 0.60392157, 0.6       , 1.        ],
                    [0.99215686, 0.74901961, 0.43529412, 1.        ],
                    [1.        , 0.49803922, 0.        , 1.        ],
                    [0.41568627, 0.23921569, 0.60392157, 1.        ],
                    [0.96470588, 0.74509804, 0.        , 1.        ],
                    [0.69411765, 0.34901961, 0.15686275, 1.        ]])
fig1 = plt.figure()
for plot_idx in range(num_shmrs):
    plt.plot(all_shmr_data[plot_idx][:,0], all_shmr_data[plot_idx][:,1], 
        ls="-.", lw=4, c=palette[plot_idx], label=names[plot_idx])
plt.plot(bin_centers, my_binned_shmr[0], ls='-', lw=4, c='k', label='Mock (Asad2023)')
plt.plot(logmh_behroozi10, analytical_ratio, ls='--', lw=4, c='k', label='Analytical (Asad2023)')
plt.xlabel(r'\boldmath$\log_{10}\ M_{h} \left[\mathrm{M_\odot} \right]$',fontsize=30)
plt.ylabel(r'\boldmath$\log_{10}(\ M_\star / M_h)$',fontsize=30)
plt.legend(loc='best',prop={'size': 20})
plt.show()

