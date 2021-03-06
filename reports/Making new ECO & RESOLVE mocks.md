# Making new ECO & RESOLVE mocks

```
Author:   Mehnaaz Asad
```



### Notes about the general process, using ECO as an example



- Number density, `n`, needed to calculate HOD parameters is obtained after applying geometry and redshift cuts and calculating the volume of the space carved out. 

  

- `.bgc2` files are particle and halo files obtained from simulation boxes after running the Rockstar halo finder. 

  

- The boxes’ ICs are set by re-scaling the present day power spectrum to the desired initial redshift of the simulation. **Note: The halo files we have are z=0.**

  

- The `.bgc2` files are used as input to the halobias code along with the HOD parameters and the output are `.ff`  (fast food format) files, which contain galaxy positions and velocities.



The `.ff` files are required by `eco_mocks_create.py` to make mocks which involves:

- CLF and abundance matching with ECO survey to get M<sub>r</sub>.
- Matching to closest M<sub>r</sub> in survey to assign all other galaxy properties to mock galaxies.
- Carving out the geometry of the survey.
- Calculating redshift-space distortions and distances.
- Carrying out group finding and group mass assignment which also includes group galaxy type and group M<sub>r</sub> / M<sub>*</sub> being calculated depending on the type of abundance matching that was carried out. 



------



### Steps run to make the new set of mocks from 8 boxes:



1. The HOD parameter values that match `n` of ECO down to RESOLVE B M<sub>r</sub> limit, **n = 0.0831 (Mpc h<sup>-1</sup>)<sup>-3</sup>**, are shown below:

   | HOD parameter                                                | Value |
   | ------------------------------------------------------------ | ----- |
   | <img src="https://render.githubusercontent.com/render/math?math=\log \M_{min}"> | 10.81 |
   | <img src="https://render.githubusercontent.com/render/math?math=\alpha \log \M"> | 0.2   |
   | <img src="https://render.githubusercontent.com/render/math?math=\log \M_{0}"> | 10.0  |
   | <img src="https://render.githubusercontent.com/render/math?math=\log \M_{1}"> | 12.05 |
   | <img src="https://render.githubusercontent.com/render/math?math=\alpha"> | 1.0   |

   

2. Halobias was run on `.bgc2` files, located in  `/zpool0/fs2/lss/LasDamas/Resolve/`, from boxes 5001-5008 for a specific halo definition (M<sub>200</sub>). The halobias executable used was `halobias_so_part_extra` located in`/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/raw`.  This was originally `halobias_so_part` located in`/fs1/szewciw/galaxy_clustering/codes/halobias` , but it was modified to include three additional columns that are required to execute the mock-making script - `halo_mass`, `central_satellite_flag`, and `halo_id`. A sample execution of halobias is shown below:

   

> `./halobias_so_part_extra 3 4 1 10.81 0.2 10.0 12.05 1.0 1 0. 1. outputfile.pnm -1 /zpool0/fs2/lss/LasDamas/Resolve/5003/rockstar/so_m200b/*.bgc2 > /fs1/masad/Research/Repositories/RESOLVE_Statistics/data/raw/ff_files/m200b/5003_200b_outfile.ff`



​	   A more general usage example is shown below:

> `./halobias_so_part_extra Ncenopt Nsatopt PNNopt logMmin siglogM logM0 logM1 alpha center Dgamma Dv PNMfile seed Halofiles > Galaxies`



​       The arguments along with their definitions and options are shown below:

| Argument    | Definition                                                   | Specified options (if applicable)                            |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `Ncenopt`   | The mean occupation function of central galaxies             | 0 : Ncen = 0<br /><br />1 : Ncen = 1 for M > M<sub>min</sub><br /><br />2 : \<Ncen> = <img src="https://render.githubusercontent.com/render/math?math=exp(-M_{min}/M)"> (Zehavi et al. 2005; Tinker et al. 2005)<br /><br />3 : \<Ncen> = <img src="https://render.githubusercontent.com/render/math?math=0.5*[1"> +<img src="https://render.githubusercontent.com/render/math?math=\erf((\log M - \log M_{min})/\sigma \log M)]">  (Zheng, Coil, & Zehavi 2007) |
| `Nsatopt`   | The mean occupation of satellite galaxies                    | 0 : Nsat = 0<br /><br />1 : \<Nsat> = <img src="https://render.githubusercontent.com/render/math?math=(M/M_{1})^\alpha \ \for M > M_{min} "> (Kravtsov et al. 2004)<br /><br />2 : \<Nsat> = <img src="https://render.githubusercontent.com/render/math?math=(M/M_{1})^\alpha \ \for M > M_{0} "><br /><br />3 : \<Nsat> =<img src="https://render.githubusercontent.com/render/math?math=exp(-M_{0}/(M - M_{min}))(M / M_{1})">(Zehavi et al. 2005; Tinker et al. 2005)<br /><br />4 : \<Nsat> = <img src="https://render.githubusercontent.com/render/math?math=0.5*[1">+<img src="https://render.githubusercontent.com/render/math?math=\erf((\log M -\log M_{min})/\sigma \log M)]*"> <img src="https://render.githubusercontent.com/render/math?math=((M - M_{0})/M_{1})^\alpha \ \for M > M_{0}"> (Zheng, Coil & Zehavi 2007) |
| `Pnnopt`    | Second moment of the occupation number which for satellites is assumed to be a poisson distribution | 0 : Average  (Nsat = nint(\<Nsat>) - with frequencies to preserve mean)<br /><br />1 : Poisson  (Nsat drawn from Poisson distribution)<br /><br />2 : Binomial (Nsat drawn from a Binomial distribution)<br /><br />3 : Negative Binomial (Nsat drawn from a Negative Binomial distribution) |
| `logMmin`   | Minimum mass of halo that can contain a galaxy (in units of M<sub>sun</sub>h<sup>-1</sup>) |                                                              |
| `siglogM`   | Width of cutoff at M<sub>min</sub> (scatter in the mass-luminosity relation) |                                                              |
| `logM0`     | Minimum halo mass that can contain a satellite galaxy (in units of M<sub>sun</sub>h<sup>-1</sup>) |                                                              |
| `logM1`     | Halo mass for which \<Nsat>=1 (in units of M<sub>sun</sub>h<sup>-1</sup>) |                                                              |
| `alpha`     | Slope of the \<Nsat> - M relation                            |                                                              |
| `center`    | Position of central galaxy relative to its halo              | 0 : Don't force the "central" galaxy to be at center of its halo<br /><br />1 : Force the "central" galaxy to be at center of its halo |
| `Dgamma`    | Difference between galaxy and mass density profiles (gamma<sub>g</sub>-gamma<sub>m</sub>) |                                                              |
| `Dv`        | Difference between galaxy and mass velocity profiles (v<sub>g</sub> - v<sub>c</sub>)/(v<sub>m</sub> - v<sub>c</sub>) |                                                              |
| `PNMfile`   | Output file containing number of galaxies selected for each halo |                                                              |
| `seed`      | Seed for gsl rng                                             |                                                              |
| `Halofiles` | Halo particle distribution files (`.bgc2 `)                  |                                                              |
| `Galaxies`  | Biased galaxy distribution (`.ff`)                           |                                                              |



3. Resulting `.ff` files located in `/fs1/masad/Research/Repositories/RESOLVE_Statistics/data/raw/ff_files/m200b` were required as input for the mock-making script, `eco_mocks_create.py`, located in `/fs1/masad/Research/Repositories/ECO_Mocks_Catls/src/data/mocks_create`.

4. Since mocks **without** the buffer region were required, the cz ranges in `eco_mocks_create.py` were modified (lines `2250` and `2251`). `RA` and `DEC` ranges remained the same. A sample execution of `eco_mocks_create` for ECO is shown below :



> `make clean`
>
> `make delete_mock_catls`
>
> `make CPU_FRAC=0.30 COSMO_CHOICE="Planck" HALOTYPE="m200b" REMOVE_FILES="True" SURVEY="ECO" catl_mr_make`



Relevant arguments of `eco_mocks_create.py` which can be modified from the command line are as follows along with their definitions, options and default values:

| Argument       | Definition                                                   | Specified options (if applicable)      | Default value         |
| -------------- | ------------------------------------------------------------ | -------------------------------------- | --------------------- |
| `size_cube`    | Length of simulation cube in Mpc h<sup>-1</sup>              |                                        | 180                   |
| `catl_type`    | Type of abundance matching used in catalog                   | [`mr` , `mstar`]                       | M<sub>r</sub>         |
| `zmedian`      | Median redshift of the survey                                |                                        | 0                     |
| `survey`       | Type of survey to produce                                    | [`A`, `B`, `ECO`]                      | ECO                   |
| `halotype`     | Type of halo definition                                      | [`mvir`, `m200b`]                      | mvir                  |
| `cosmo_choice` | Cosmology to use                                             | [`Planck`,`LasDamas`]                  | Planck                |
| `hmf_model`    | Halo Mass Function choice                                    | [`warren`,`tinker08`]                  | Warren                |
| `clf_type`     | Type of CLF                                                  | `1` - Cacciato `2` - LasDamas best-fit | (2) LasDamas best-fit |
| `zspace`       | Option for adding redshift-space distortions                 | `1` - No RSD `2` - With RSD            | (2) **with** RSD      |
| `nmin`         | Minimum number of galaxies in a galaxy group                 | [1-1000]                               | 1                     |
| `seed`         | Random seed to be used for the analysis                      |                                        | 1                     |
| `remove_files` | Delete files created by the script, in case they exist already | True/False                             | False                 |
| `cpu_frac`     | Fraction of total number of CPUs to use                      | [0-1]                                  | 0.75                  |



------

### Additional notes:

- The steps above can be reproduced for RESOLVE by changing the `survey` argument value.
- The script has been modified to run on all 8 boxes automatically.
- The resulting mock catalogs are located in `/fs1/masad/Research/Repositories/ECO_Mocks_Catls/data/processed/TAR_files/Planck/m200b` as `.tar.gz` files. 