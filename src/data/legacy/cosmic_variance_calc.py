"""
{This script calculates cosmic variance of RESOLVE-B by fitting it within ECO}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__author__ = '{Mehnaaz Asad}'

def get_area_on_sphere(ra_min,ra_max,dec_min,dec_max):
    """Calculate area on sphere given ra and dec in degrees"""
    area = (180/np.pi)*(ra_max-ra_min)* \
        (np.rad2deg(np.sin(np.deg2rad(dec_max)))- \
            np.rad2deg(np.sin(np.deg2rad(dec_min)))) 
    return area

def get_dec_width(area,ra_min,ra_max,dec_min):
    """Calculate dec width given a fixed area, ra range and lower limit of dec"""
    const = (area/((180/np.pi)*(ra_max-ra_min)))
    dec_max = np.rad2deg(np.arcsin(np.deg2rad(const) + np.sin(np.deg2rad(dec_min))))
    dec_width = np.abs(dec_max - dec_min)
    return dec_width

### Paths
dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_interim = dict_of_paths['int_dir']
path_to_figures = dict_of_paths['plot_dir']

# 13878 galaxies
eco_buff = pd.read_csv(path_to_raw + "eco_all.csv",delimiter=",", header=0)

# 6456 galaxies                       
eco_nobuff = eco_buff.loc[(eco_buff.grpcz.values >= 3000) & \
    (eco_buff.grpcz.values <= 7000) & (eco_buff.absrmag.values <= -17.33)]

resolve_live18 = pd.read_csv(path_to_raw + 'RESOLVE_liveJune2018.csv',
                            delimiter=',')
resolve_logmbary = np.log10((10**resolve_live18.logmstar.values) + 
                            (10**resolve_live18.logmgas.values))
resolve_live18['logmbary'] = resolve_logmbary
#487
resolve_B = resolve_live18.loc[(resolve_live18.f_b.values == 1) & 
                                (resolve_live18.grpcz.values >= 4500) & 
                                (resolve_live18.grpcz.values <= 7000) & 
                                (resolve_live18.absrmag.values <= -17)]
                                # (resolve_live18.logmstar >= 8.7) & 
                                # (resolve_live18.logmbary >= 9.1)]

# Using dec width in degrees of RESOLVE-B, see how many such slices can fit 
# within dec range of ECO
dec_slices = np.arange(-1,49.85,2.5) #slices of 2.5 degrees width
ra_min = 134.99
ra_max = 209.99
dec_min = dec_slices[0]
dec_max = dec_slices[1]
area = get_area_on_sphere(ra_min,ra_max,dec_min,dec_max)

# On sphere, as dec changes the area gets smaller so by forcing the area to be 
# the same as the first slice as well as keeping RA ranges fixed, get new 
# dec width for each slice
dec_width_arr = np.zeros(len(dec_slices))
for index,dec_min in enumerate(dec_slices):
    dec_width = get_dec_width(area,ra_min,ra_max,dec_min)
    dec_width_arr[index] = dec_width

# Calculate new dec slices using new dec widths for each slice
new_dec_slices = np.zeros(len(dec_slices))
for index,value in enumerate(dec_width_arr):
    if index == 0:
        current_dec_min = dec_slices[0]
        new_dec_slices[index] = current_dec_min
    else: 
        new_dec_max = current_dec_min + value
        new_dec_slices[index] = new_dec_max
        current_dec_min = new_dec_max

# Max of new slices goes until 60+ degrees which is outside the range of ECO.
# Keeping dec values until 49.85 i.e. max range of ECO
new_dec_slices = new_dec_slices[:18]
dec_width_arr = dec_width_arr[:18]
new_dec_slices = np.append(new_dec_slices,49.85)
dec_width_arr = np.append(dec_width_arr,49.85-new_dec_slices[-2])

minz_resB = np.min(resolve_B.cz/(3*10**5))
maxz_resB = np.max(resolve_B.cz/(3*10**5))
mincz_resB = 4500
maxcz_resB = 7000
H_0 = 70
deg2_in_sphere = 41252.96 #deg^2

num_in_slice = []
cz = []
for index,value in enumerate(new_dec_slices):
   if index == 21:
       break
   counter = 0
   RA_arr = []
   DEC_arr = []
   for index2,galaxy in enumerate(eco_nobuff.name.values):
       # Fixed RA span of 75 degrees which corresponds to ra span of RESOLVE-B 
       # of 5 hours. Using same span in ECO i.e. 9h - 14h corresponds to 
       # 134.99 - 209.99 in degrees
       if 134.99 < eco_nobuff.radeg.values[index2] < 209.99: 
           if value < eco_nobuff.dedeg.values[index2] < dec_slices[index+1]:
               if mincz_resB < (eco_nobuff.cz.values[index2]) < maxcz_resB:
                   cz.append(eco_nobuff.cz.values[index2])
                   RA_arr.append(eco_nobuff.radeg.values[index2])
                   DEC_arr.append(eco_nobuff.dedeg.values[index2])
                   counter += 1

#    fig = plt.figure(figsize=(10,8))
#    ax = plt.gca()
#    plt.ylim(-1,49.85)
#    plt.xlim(0,75)
#    plt.scatter(RA_arr,DEC_arr)
#    plt.xlabel('RA (deg)')
#    plt.ylabel('DEC (deg)')
   num_in_slice.append(counter)
num_in_slice = np.array(num_in_slice)

inner_vol = (4/3)*np.pi*((mincz_resB/H_0)**3)
outer_vol = (4/3)*np.pi*((maxcz_resB/H_0)**3)
vol_sphere = outer_vol-inner_vol #Mpc^3

# Using function to calculate area of each slice with increasing dec width
slice_area_arr = []
for index,value in enumerate(new_dec_slices):
    if index == 18:
        break
    slice_area = get_area_on_sphere(ra_min,ra_max,value,new_dec_slices[index+1])
    slice_area_arr.append(slice_area)
slice_area_arr = np.array(slice_area_arr)

vol_slice_arr = (slice_area_arr/deg2_in_sphere)*vol_sphere
gal_dens_arr = num_in_slice[:18]/vol_slice_arr # Shape ,19 vs ,18 originally
# How cosmic variance was calculated in Eckert et. al 2016 using mocks.
# Originally, this just involved using standard deviation without mean.
cosmic_variance = (np.std(gal_dens_arr)/np.mean(gal_dens_arr))*100
print('Cosmic variance: {0}%'.format(np.round(cosmic_variance,2)))