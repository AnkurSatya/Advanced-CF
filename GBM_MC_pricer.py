# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:03:54 2022

@author: LPras
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.linear_model import LinearRegression

days_in_year = 365

npath = 1000
strike = 99
s = 100
r = 0.06 #anual risk free rate
sigma = 0.2 #anual volatility
n_days = 10
#used for risk free rate over all the days when calculating option price for european option
T = n_days/days_in_year


# set attributes for underlying price process simulation
def initiate_path(npath, s, r, sigma, days, strike):

    path = pd.DataFrame(np.zeros(npath)+s)
    
    #to get daily volatility and risk free rate
    dt = 1/days_in_year

    for i in range(days):
        this_day = path.iloc[:,i]
        next_day = this_day * np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)* np.random.normal(size = npath))
        path = pd.concat([path,next_day], axis = 1)

    # clean dataframe
    path.columns = list(range(days+1))
    path['pay_off_at_maturity'] = np.maximum(0,path[days] - strike)
    path['price_option'] = path['pay_off_at_maturity'] * np.exp(-r * T)
    average_P = path['price_option'].mean()
    sterr_P = path['price_option'].std()/np.sqrt(npath)

    return path, average_P, sterr_P

path, average_P, sterr_P = initiate_path(npath, s, r, sigma, n_days, strike)
print(average_P, sterr_P)

stock_paths = path.iloc[:,0:-4].T


