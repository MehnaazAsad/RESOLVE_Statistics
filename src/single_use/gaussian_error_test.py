"""
{This script tests the effects of non-gaussian and gaussian errors on 
 chi-squared statistic}
"""

# Libs
from cosmo_utils.utils import work_paths as cwpaths
from chainconsumer import ChainConsumer
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import chi2 
from matplotlib import rc
import pandas as pd
import numpy as np
import emcee
import math
import os

__author__ = '{Mehnaaz Asad}'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']}, size=20)
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('xtick.major', width=2, size=7)
rc('ytick.major', width=2, size=7)

def lnprob(theta, x_vals, y_vals, err_tot):
    """
    Calculates log probability for emcee

    Parameters
    ----------
    theta: array
        Array of parameter values

    x_vals: array
        Array of x-axis values

    y_vals: array
        Array of y-axis values
    
    err_tot: array
        Array of error values of mass function

    Returns
    ---------
    lnp: float
        Log probability given a model

    chi2: float
        Value of chi-squared given a model 
        
    """
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        try:
            model = m * x_vals + b 
            chi2 = chi_squared(y_vals, model, err_tot)
            lnp = -chi2 / 2
            if math.isnan(lnp):
                raise ValueError
        except (ValueError, RuntimeWarning, UserWarning):
            lnp = -np.inf
            chi2 = -np.inf
    else:
        chi2 = -np.inf
        lnp = -np.inf
        
    return lnp, chi2

def chi_squared(data, model, err_data):
    """
    Calculates chi squared

    Parameters
    ----------
    data: array
        Array of data values
    
    model: array
        Array of model values
    
    err_data: float
        Error in data values

    Returns
    ---------
    chi_squared: float
        Value of chi-squared given a model 

    """
    chi_squared_arr = (data - model)**2 / (err_data**2)
    chi_squared = np.sum(chi_squared_arr)

    return chi_squared

np.random.seed(30)
m_true_arr = np.round(np.random.uniform(-4.9, 0.4, size=500),2)
b_true_arr = np.round(np.random.uniform(1, 7, size=500),2)
sigma_arr = np.round(np.random.uniform(0.5, 3.5, size=500),2)

## Keeping data fixed
m_true = m_true_arr[50]
b_true = b_true_arr[50]
N=50
x = np.sort(10 * np.random.rand(N))

samples_arr = []
chi2_arr = []
yerr_arr = []
for i in range(500):
    print(i)

    # Generate some synthetic data from the model.
    N = 50
    mu = 0
    sigma = sigma_arr[200]
    # shape = (mu/sigma)**2
    # scale = (sigma**2)/mu
    loc = mu
    scale = np.sqrt(3)*(sigma/np.pi) 

    ## Gaussian vs non-gaussian errors
    # yerr = np.random.normal(mu, sigma, N)
    yerr = np.random.logistic(loc, scale, N)

    y = m_true * x + b_true
    y += yerr

    pos = [0,5] + 1e-4 * np.random.randn(64, 2) #sigma = 1e-4 and mu = [0,5]
    nwalkers, ndim = pos.shape
    nsteps = 5000

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
        args=(x, y, sigma))
    sampler.run_mcmc(pos, nsteps, store=True, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    chi2 = sampler.get_blobs(discard=100, thin=15, flat=True)

    samples_arr.append(flat_samples)
    chi2_arr.append(chi2)
    yerr_arr.append(yerr)

gaussian_samples_arr = np.array(samples_arr)
gaussian_chi2_arr = np.array(chi2_arr)
gaussian_yerr_arr = np.array(yerr_arr)

non_gaussian_samples_arr = np.array(samples_arr)
non_gaussian_chi2_arr = np.array(chi2_arr)
non_gaussian_yerr_arr = np.array(yerr_arr)

# Get parameter values for each realization that correspond to lowest 
# chi-squared value
gaussian_minchi2_arr = []
gaussian_minm_arr = []
gaussian_minb_arr = []
for idx in range(len(gaussian_chi2_arr)):
    gaussian_minchi2_arr.append(min(gaussian_chi2_arr[idx]))
    gaussian_minm_arr.append(gaussian_samples_arr[idx][:,0][np.argmin(gaussian_chi2_arr[idx])])
    gaussian_minb_arr.append(gaussian_samples_arr[idx][:,1][np.argmin(gaussian_chi2_arr[idx])])

non_gaussian_minchi2_arr = []
non_gaussian_minm_arr = []
non_gaussian_minb_arr = []
for idx in range(len(non_gaussian_chi2_arr)):
    non_gaussian_minchi2_arr.append(min(non_gaussian_chi2_arr[idx]))
    non_gaussian_minm_arr.append(non_gaussian_samples_arr[idx][:,0][np.argmin(non_gaussian_chi2_arr[idx])])
    non_gaussian_minb_arr.append(non_gaussian_samples_arr[idx][:,1][np.argmin(non_gaussian_chi2_arr[idx])])

samples = np.column_stack((non_gaussian_minm_arr,non_gaussian_minb_arr)) 
samples = np.column_stack((gaussian_minm_arr,gaussian_minb_arr)) 

## Plot of maximum likelihood points
m_mean = samples[:,0].mean()
m_std = samples[:,0].std()
b_mean = samples[:,1].mean()
b_std = samples[:,1].std()

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
fig1 = plt.figure(figsize=(8, 8))

textstr_m = '\n'.join((
    r'$\mu=%.2f$' % (m_mean, ),
    r'$\sigma=%.2f$' % (m_std, )))

textstr_b = '\n'.join((
    r'$\mu=%.2f$' % (b_mean, ),
    r'$\sigma=%.2f$' % (b_std, )))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax_histx.text(0.20, 0.95, textstr_m, transform=ax_histx.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
# place a text box in upper left in axes coords
ax_histy.text(0.50, 0.98, textstr_b, transform=ax_histy.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax_scatter.scatter(samples[:,0], samples[:,1], c='mediumorchid', marker='+')
ax_histx.hist(samples[:,0], histtype='step', color='cornflowerblue',
    linewidth=5)
ax_histy.hist(samples[:,1], histtype='step', orientation='horizontal', 
    color='cornflowerblue', linewidth=5)

ax_scatter.set_xlabel('m')
ax_scatter.set_ylabel('b')

ax_histx.set_title('Distribution of maximum likelihood points given non-gaussian errors')
fig1.show()


gaussian_minchi2_arr = np.array(gaussian_minchi2_arr)
non_gaussian_minchi2_arr = np.array(non_gaussian_minchi2_arr)
min_bin = math.floor(min([min(gaussian_minchi2_arr), min(non_gaussian_minchi2_arr)]))
max_bin = math.ceil(max([max(gaussian_minchi2_arr), max(non_gaussian_minchi2_arr)]))
num_bins = 20
dof = N - ndim
## Plot of distributions of minimum chi-squared
fig2 = plt.figure()
x = np.linspace(chi2.ppf(0.01, dof),chi2.ppf(0.99, dof), 100)
plt.plot(x, 2225*chi2.pdf(x, dof), 'r-', lw=5, alpha=0.6, label='scaled chi2 pdf')
plt.hist(gaussian_minchi2_arr[~np.isinf(gaussian_minchi2_arr)], histtype='step',
    bins=np.linspace(min_bin, max_bin, num_bins),
    color='cornflowerblue', label='Gaussian', linewidth=5)                                          
plt.hist(non_gaussian_minchi2_arr[~np.isinf(non_gaussian_minchi2_arr)],
    bins=np.linspace(min_bin, max_bin, num_bins),
    histtype='step', color='mediumorchid', label='Logistic', linewidth=5)
plt.title(r'Distribution of min(${\chi}^{2}$)') 
plt.legend(loc='upper right')     
fig2.show() 

## Plot of combined mean of chains
truth = {"$m$": m_true, "$b$": b_true}  
c = ChainConsumer()  
c.add_chain(gaussian_samples_arr.mean(axis=1), 
    parameters=[r"$\bar m$", r"$\bar b$"], color='#6495ed', name='Gaussian') 
c.add_chain(non_gaussian_samples_arr.mean(axis=1), 
    parameters=[r"$\bar m$", r"$\bar b$"], color='#ba55d3', name='Logistic') 
c.configure(diagonal_tick_labels=False, tick_font_size=8, label_font_size=25, 
    max_ticks=8)  
fig = c.plotter.plot(truth=truth, display=True)      

## Plot of combined std of chains
c = ChainConsumer()  
c.add_chain(gaussian_samples_arr.std(axis=1), 
    parameters=[r"${\sigma}_{m}$", r"${\sigma}_{b}$"], color='#6495ed', 
    name='Gaussian') 
c.add_chain(non_gaussian_samples_arr.std(axis=1), 
    parameters=[r"${\sigma}_{m}$", r"${\sigma}_{b}$"], color='#ba55d3', 
    name='Logistic') 
c.configure(diagonal_tick_labels=False, tick_font_size=8, label_font_size=25, 
    max_ticks=8, legend_location=(0,-1))  
fig = c.plotter.plot(display=True, extents={r"${\sigma}_{b}$": (0.975,1.016), \
    r"${\sigma}_{m}$":(0.176,0.184)})   


# Plot sample that corresponds to min and max of minimum chi-squared for 
# logistic distribution

idx_min = np.argmin(non_gaussian_minchi2_arr) 
m_true_min = samples[idx_min][0]
b_true_min = samples[idx_min][1]
yerr_min = non_gaussian_yerr_arr[idx_min]
x_min = x
y_min = m_true_min * x_min + b_true_min
y_min += yerr_min
samples_min = non_gaussian_samples_arr[idx_min]
chi2_min = non_gaussian_chi2_arr[idx_min]
df_min = pd.DataFrame({"m": samples_min[:,0], "b": samples_min[:,1], \
    "chi2":chi2_min})
df_min_sorted = df_min.sort_values('chi2').reset_index(drop=True)
slice_end = int(68*len(df_min_sorted))
df_min_pctl = df_min_sorted[:slice_end]
df_min_pctl_sample = df_min_pctl.drop_duplicates().sample(100)

idx_max = np.argmax(non_gaussian_minchi2_arr) 
m_true_max = samples[idx_max][0]
b_true_max = samples[idx_max][1]
yerr_max = non_gaussian_yerr_arr[idx_max]
x_max = x
y_max = m_true_max * x_max + b_true_max
y_max += yerr_max
samples_max = non_gaussian_samples_arr[idx_max]
chi2_max = non_gaussian_chi2_arr[idx_max]
df_max = pd.DataFrame({"m": samples_max[:,0], "b": samples_max[:,1], \
    "chi2":chi2_max})
df_max_sorted = df_max.sort_values('chi2').reset_index(drop=True)
slice_end = int(68*len(df_max_sorted))
df_max_pctl = df_max_sorted[:slice_end]
df_max_pctl_sample = df_max_pctl.drop_duplicates().sample(100)

fig3 = plt.figure()
x0 = np.linspace(0, 10, 500)
for idx in range(len(df_min_pctl_sample)):
    plt.plot(x0, df_min_pctl_sample.m.values[idx] * x0 + \
        df_min_pctl_sample.b.values[idx], "palegoldenrod", alpha=0.3)
plt.plot(x0, m_true_min * x0 + b_true_min, "gold", label="low min", zorder=10,
    linewidth=5)
plt.scatter(x_min, y_min, c='darkgoldenrod', marker='o', label='data (min)', 
    s=45)
for idx in range(len(df_max_pctl_sample)):
    plt.plot(x0, df_max_pctl_sample.m.values[idx] * x0 + \
        df_max_pctl_sample.b.values[idx], "peachpuff", alpha=0.3)
plt.plot(x0, m_true_max * x0 + b_true_max, "maroon", label="high min", 
    zorder=11, linewidth=5)
plt.scatter(x_max, y_max, c='firebrick', marker='*', label='data (max)', s=45)
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Data and model comparison between min and max of min(${\chi}^{2}$)")
fig3.show()

# error_pdf_comparison.png plot
sns.distplot(np.random.normal(mu, sigma, 100000), label='gaussian')
sns.distplot(np.random.logistic(loc, scale, 100000), label='logistic')                             
plt.legend(loc='best')
plt.show() 

# Weibull pdf plot for different parameter values
from scipy.stats import weibull_min 
fig, ax = plt.subplots(1, 1)
c_arr = np.linspace(1,3,20)
mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
x = np.linspace(weibull_min.ppf(0.01, c), weibull_min.ppf(0.99, c), 100) 
ax.set_prop_cycle(color=[
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5'])
for idx in range(len(c_arr)):
    ax.plot(x, weibull_min.pdf(x, c_arr[idx]), lw=5, alpha=0.6, label='c={0}'.format(np.round(c_arr[idx], 2)))
plt.legend(loc='best', prop={'size': 10})
plt.title('Weibull pdf')
plt.show()