#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:40:06 2019

@author: asadm2
"""
#from cosmo_utils.utils import Stats_one_arr
from scipy.stats import binned_statistic as bs
import numpy as np
import random
import math

def get_y(m,b,x):    
    return m*x+b

def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h = 2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

def chi_squared(data,model):
    counts_data,edg_data = np.histogram(data,bins=num_bins(data)) 
    err_poiss = np.sqrt(counts_data)
    counts_model,edg_model = np.histogram(model,bins=num_bins(data))
    chi_squared = (counts_data - counts_model)**2/(err_poiss**2)
    print(counts_data)
    return np.sum(chi_squared)


m = -1.57
b = 6.8

N=30
x_data = np.sort(10*np.random.rand(N))
y = m*x_data + b
yerr = 5*np.random.rand(N)
y_data = y + yerr


y_model = get_y(m,b,x_data)
chi_squared_old = chi_squared(y_data,y_model)

iter = 0
old_m = m
old_b = b
chain = [[],[]]
while chi_squared_old > 1 and iter < 5:
    iter += 1
    new_m = np.random.normal(old_m,1)
    new_b = np.random.normal(old_b,2)
    print(chi_squared_old, old_m, old_b, iter)
    y_model = get_y(new_m,new_b,x_data)
    chi_squared_new = chi_squared(y_data,y_model)
    
    
    if chi_squared_new < chi_squared_old:
        chain[0].append(new_m)
        chain[1].append(new_b)
        old_m = new_m
        old_b = new_b
        chi_squared_old = chi_squared_new
    else:
        delta_chisquared = chi_squared_new - chi_squared_old
        prob = (np.e**(-delta_chisquared))/2
        print(prob, delta_chisquared, chi_squared_new, chi_squared_old)
        rand = random.uniform(0,1)
        if rand < prob:
            chain[0].append(new_m)
            chain[1].append(new_b)
            old_m = new_m
            old_b = new_b
            chi_squared_old = chi_squared_new
        else:
            chain[0].append(old_m)
            chain[1].append(old_b)

plt.scatter(x_data, y_data)
plt.plot(x_data,get_y(old_m, old_b,x_data))