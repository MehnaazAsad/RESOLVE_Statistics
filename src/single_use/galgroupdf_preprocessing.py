"""
{This script does some preprocessing on the output from running
group finder on the results from the chain. Specifically, it 
chooses to retain only the columns needed to make plots so that 
the size of the dataframe is as small as needed.}
"""
__author__ = '{Mehnaaz Asad}'

import argparse
import os

from cosmo_utils.utils import work_paths as cwpaths
import pandas as pd
import numpy as np
import pickle

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
    parser.add_argument('run', type=int, nargs='?', 
        help='Chain number')
    args = parser.parse_args()
    return args

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

def main(args):

    dict_of_paths = cwpaths.cookiecutter_paths()
    path_to_proc = dict_of_paths['proc_dir']
    run = args.run

    print('Reading file')
    gal_group = read_mock_catl(path_to_proc + \
        "gal_group_run{0}.hdf5".format(run)) 

    print('Creating subset')
    idx_arr = np.insert(np.linspace(1, 20, 20), len(np.linspace(1, 20, 20)), \
        (22, 223, 224, 225, 226, 227, 228, 229)).\
        astype(int)

    names_arr = [x for x in gal_group.columns.values[idx_arr]]
    for idx in np.arange(2, 202, 1):
        names_arr.append('{0}_y'.format(idx))
        names_arr.append('groupid_{0}'.format(idx))
        names_arr.append('grp_censat_{0}'.format(idx))
        names_arr.append('cen_cz_{0}'.format(idx))
    names_arr = np.array(names_arr)

    gal_group_df_subset = gal_group[names_arr]

    print('Writing to output files')
    with open(path_to_proc + 'gal_group_run{0}.pickle'.format(run), 'wb') as handle:
        pickle.dump(gal_group_df_subset, handle, 
        protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = args_parser()
    main(args)
