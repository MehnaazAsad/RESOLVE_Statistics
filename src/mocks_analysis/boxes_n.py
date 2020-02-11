from cosmo_utils.utils.file_readers import fast_food_reader as ffr
from cosmo_utils.utils import work_paths as cwpaths
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os 

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

dict_of_paths = cwpaths.cookiecutter_paths()
path_to_raw = dict_of_paths['raw_dir']
path_to_ff = path_to_raw + 'ff_files/m200b/'
os.chdir(path_to_ff)

ngal_arr = [] 
for hb_local in np.sort(os.listdir()): 
    with open(path_to_ff + hb_local, 'rb') as hb: 
        idat    = ffr('int'  , 5, hb) 
        fdat    = ffr('float', 9, hb) 
        znow    = ffr('float', 1, hb)[0] 
        ngal    = int(idat[1]) 
        lbox    = int(fdat[0]) 
        x_arr   = ffr('float' , ngal, hb)
        y_arr   = ffr('float' , ngal, hb)
        z_arr   = ffr('float' , ngal, hb)
        vx_arr  = ffr('float' , ngal, hb)
        vy_arr  = ffr('float' , ngal, hb)
        vz_arr  = ffr('float' , ngal, hb)
        halom   = ffr('double' , ngal, hb)
        cs_flag = ffr('int'   , ngal, hb)
        haloid  = ffr('long'   , ngal, hb)
        ngal_arr.append(ngal) 

ngal_arr = np.array(ngal_arr)

ngal_boxes = ngal_arr/(lbox**3)
ngal_5001_or = 0.0831
ngal_5001 = ngal_boxes[0]

frac_diff = np.abs(((ngal_boxes-ngal_5001)/ngal_5001)*100)

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=False,figsize=(10,8),\
     gridspec_kw = {'height_ratios':[8,2]})
ax1.scatter(np.linspace(5002,5008,7), ngal_boxes[1:], s=150, 
    c='mediumorchid', alpha=0.7) 
ax1.scatter(5001, ngal_5001, marker='*', s=200, c='cornflowerblue', 
    alpha=0.7, label=r'new 5001')  
ax1.scatter(5001, ngal_5001_or, marker='s', facecolors='none', s=200, 
    edgecolors='k', label=r'original 5001')
ax1.set_ylabel(r'$n\ {[Mpc/h]}^{-3}$', fontsize=20)
ax1.legend(loc='best', prop={'size': 20})

ax2.scatter(np.linspace(5001,5008,8), frac_diff,c='k')
ax2.set_ylabel(r'$\mathrm{[\frac{new-5001}{5001}]} \%$', fontsize=15)
fig.tight_layout()
fig.show()
                                      