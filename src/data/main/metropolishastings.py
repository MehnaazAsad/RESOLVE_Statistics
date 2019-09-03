#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:40:06 2019

@author: asadm2
"""
#from cosmo_utils.utils import Stats_one_arr
from scipy.stats import binned_statistic as bs
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random
import corner
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

def chi_squared(data_x,data_y,model_x,model_y):
    data_stats,data_edges,data_binnum = bs(data_x,data_y,bins=num_bins(data_x))
    bin_counter = Counter(np.sort(data_binnum))
    data_counts = []
    for key,value in bin_counter.items():
        data_counts.append(value)
    data_counts = np.array(data_counts)
    err_poiss = np.sqrt(data_counts)
    model_stats,model_edges,model_binnum = bs(model_x,model_y,bins=num_bins(data_x))

    chi_squared = (data_stats - model_stats)**2/(err_poiss**2)
    return np.sum(chi_squared)


m = -1.57
b = 6.8

N=50
x_data = np.sort(10*np.random.rand(N))
y_data = m*x_data + b
yerr = 5*np.random.rand(N)
y_data = y_data + yerr

y_model = get_y(m,b,x_data)
chi_squared_old = chi_squared(x_data,y_data,x_data,y_model)

iter = 0
old_m = m
old_b = b
chain = []
while chi_squared_old > 0.05: #and len(chain) < 3500: #iter < 5:
    iter += 1
    #get new m and b from gaussian
    new_m = np.random.normal(old_m,1)
    new_b = np.random.normal(old_b,0.5)
    #generate y data
    y_model = get_y(new_m,new_b,x_data)
    #calculate new chi squared
    chi_squared_new = chi_squared(x_data,y_data,x_data,y_model)
    
    
    if chi_squared_new < chi_squared_old:
        chain.append([new_m,new_b])
#        chain[0].append(new_m)
#        chain[1].append(new_b)
        old_m = new_m
        old_b = new_b
        chi_squared_old = chi_squared_new
    else:
        delta_chisquared = chi_squared_new - chi_squared_old
        prob = np.exp(-delta_chisquared/2)
        rand = random.uniform(0,1)
        if rand < prob:
            chain.append([new_m,new_b])
#            chain[0].append(new_m)
#            chain[1].append(new_b)
            old_m = new_m
            old_b = new_b
            chi_squared_old = chi_squared_new
        else:
            chain.append([old_m,old_b])

#            chain[0].append(old_m)
#            chain[1].append(old_b)
#    print(iter,chi_squared_new)

fig1 = plt.figure(figsize=(10,10))
plt.scatter(x_data,y_data)
plt.plot(x_data,get_y(old_m, old_b,x_data))

#fig2 = plt.figure(figsize=(10,10))
#plt.scatter(chain[0], chain[1])

fig3 = corner.corner(chain, labels=["$m$", "$b$"], \
                     truths=[m,b])

