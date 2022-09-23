# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:39:56 2022

@author: LPras
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def LSM_montecarlo(T,N,steps,S0,r,sigma,K, typeCP):

    typeCP = 1 if typeCP == "call" else -1    
    dt = np.float(T) / steps
    df = np.exp(-r * dt)
    
    # simulation of index levels
    stock_paths = np.zeros((steps + 1, N))

    stock_paths[0] = S0

    for t in range(1, steps + 1):
        RV_paths = np.random.normal(size = N)
        stock_paths[t] = stock_paths[t-1]*np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*RV_paths)

    

    # case-based calculation of payoff
    payoff_T = np.maximum(typeCP*(stock_paths - K), 0)
    
    #print(np.round(payoff_T,0))
    
    # LSM algorithm
    V = np.copy(payoff_T)

    for t in range(steps-1, 0, -1):
        #Least squares polynomial fit.
        #reg = np.polyfit(stock_paths[t], V[t+1]*df, 7)
        reg = np.polynomial.legendre.legfit(stock_paths[t], V[t+1]*df, 7)

        #Evaluate polynomial at specific values
        #expected_value_option = np.polyval(reg, stock_paths[t])
        expected_value_option = np.polynomial.legendre.legval(stock_paths[t],reg)

        
        #payoffs and if option was exercised earlier because the regressed
        #price was better than payoff also the discounted price of the option
        if (t % frequency_use_bermudan) == 0:
            print(t)
            V[t] = np.where(expected_value_option > payoff_T[t], V[t+1]*df, payoff_T[t])
        else:
            V[t] = V[t+1]*df
            
    # V[1] is value of the option of those paths
    #sum over all the final option prices divided by the number op paths
    #times one df because the first step was not regressed
    option = df * np.sum(V[1])/ N
    return option

T=1
N=100000
steps=365
S=100
r=0.06
sigma=0.2
K=99
typeCP='put'
frequency_use_bermudan = 30

print('Bermudan option where you can exercise every', frequency_use_bermudan,'days, price:', LSM_montecarlo(T,N,steps,S,r,sigma,K, typeCP))