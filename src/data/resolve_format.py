#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:02:04 2018

@author: asadm2
"""

from functools import reduce
from matplotlib import rc
import pandas as pd

### Paths
path_to_raw = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/raw/'
path_to_interim = '/Users/asadm2/Documents/Grad_School/Research/Repositories/'\
'resolve_statistics/data/interim/'

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=25)
rc('text', usetex=True)

resolve_dr1_1 = path_to_raw + 'RESOLVE_DR1_1.txt'
resolve_dr2_1 = path_to_raw + 'RESOLVE_DR2_1.txt'
resolve_dr2_2 = path_to_raw + 'RESOLVE_DR2_2.txt'


dr11 = pd.read_csv(resolve_dr1_1,delimiter='\s+',header=None,\
                           skiprows=79,usecols=[0,3,4,44],names=['Name','cz',
                                               'groupcz','logstellarmass']) 

dr21 = pd.read_csv(resolve_dr2_1,delimiter='\s+',header=None,\
                           skiprows=45,names=['Name','RA','DEC','Tel','F21',
                                              'e_F21','rms','limflag',
                                              'confused','F21corr',
                                              'eF21corr-rand','eF21corr-sys',
                                              'deconf-mode'])   

dr22 = pd.read_csv(resolve_dr2_2,delimiter='\s+',header=None,\
                           skiprows=61,names=['Name','groupid-s','groupmass-s',
                                              'groupid-b','groupmass-b',
                                              'dens-s','dens-corr-s','dens-b',
                                              'dens-corr-b','dist-s',
                                              'f_dist-s','dist-b',
                                              'f_dist-b'])

#Common galaxies
dfs = [dr11, dr21, dr22] #2286,2164,1384 number of galaxies respectively
merged_dr = reduce(lambda left,right: pd.merge(left,right,on='Name'), dfs)


merged_dr_units = ['--','km/s','km/s','solMass','deg','deg','--','Jy.km/s',
                   'Jy.km/s','mJy','--','--','Jy.km/s','Jy.km/s','Jy.km/s',
                   '--','--','solMass','--','solMass','solMass/Mpc2',
                   'solMass/Mpc2','solMass/Mpc2','solMass/Mpc2','--','--',
                   '--','--']

merged_dr_description = ['RESOLVE galaxy identifier',\
                         'Local group corrected velocity of galaxy',\
                         'Velocity of group center','Log stellar mass',\
                         'Right Ascension of photometric center',\
                         'Declination of photometric center',\
                         'Telescope used for observation (GBT = Green Bank 100'
                         'm; AO = Arecibo 300m; ALFALFA = Arecibo Legacy Fast '
                         'ALFA Survey (Arecibo 300m); S05-AO = Springob et '
                         'al. (2005) catalog (Arecibo 300m); S05-GB300 = '
                         'Springob et al. (2005) catalog '
                         '(Green Bank 300ft) )','Total 21cm flux',\
                         'Uncertainty on total 21cm flux',\
                         'rms noise of spectrum assuming 10 km/s channels',\
                         'Flag indicating upper limit on F21 '
                         '(0 = detection; 1=upper limit)',\
                         'Flag indicating 21cm emission likely confused '
                         '(0 = not confused; 1 = confused)',\
                         'Total 21cm flux after correction for confusion',\
                         'Statistical uncertainty on total 21cm flux after '
                         'correction for confusion',\
                         'Additional systematic uncertainty on total 21cm '
                         'flux after correction for confusion',\
                         'Method used to correct 21cm flux for confusion and '
                         'determine the additional systematic uncertainty',\
                         'Group dark matter halo ID, stellar-mass limited '
                         'sample','Group dark matter halo mass, stellar-mass '
                         'limited sample','Group dark matter halo ID, '
                         'baryonic mass-limited sample','Group dark matter '
                         'halo mass, baryonic mass-limited sample',\
                         'Large-scale structure density, stellar mass-limited '
                         'sample groups, uncorrected for edge-effects',\
                         'Large-scale structure density, stellar mass-limited '
                         'sample groups, corrected for edge-effects',\
                         'Large-scale structure density, baryonic '
                         'mass-limited sample groups, uncorrected for '
                         'edge-effects','Large-scale structure density, '
                         'baryonic mass-limited sample groups, corrected '
                         'for edge-effects','Projected distance to nearest '
                         'M_h>10^12 halo, stellar mass-limited sample groups',\
                         'Reliability Flag for dist_s (Flag=1 indicates that '
                         'the corresponding distance may be unreliable due to '
                         'proximity to survey boundary.)',\
                         'Projected distance to nearest M_h>10^12 halo, '
                         'baryonic mass-limited sample groups ',\
                         'Reliability Flag for dist_b (Flag=1 indicates that '
                         'the corresponding distance may be unreliable due to '
                         'proximity to survey boundary.)']

units = {}
describe = {}
for index,value in enumerate(merged_dr.columns.values):
    name = value
    units[name] = merged_dr_units[index]
    describe[name] = merged_dr_description[index]
    
#merged_dr.to_csv(path_to_interim+'RESOLVE_formatted.txt',index=False, sep='\t',\
#           columns=merged_dr.columns)