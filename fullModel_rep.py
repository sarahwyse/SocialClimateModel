#the full model using opinion response functions as the social model and the SDE energy balance as the climate model
#this version saves data from repeated simulations to an csv file

import numpy as np
import numpy.random as rnd
from scipy.stats import binom
import pandas as pd
import time

t1 = time.time()

#time parameters
yrToSec = 365*24*60*60   #original model was in seconds, use this as part of C to convert to years
years = 80               #years to run model for to match IPCC projections (over 80 years)
dt = 1/365               #step size for solving deterministic part, use dt=1/365 to make each time step a day

#climate parameters
C = (3.985e6)*50/yrToSec    #upper-ocean heat capacity to 50m depth [J/m^2/°C] 
S = 1372                    #solar radiation (divide by 4 to get insolation) [W/m^2]  
alpha = 0.33                #albedo, or planetary reflectivity [unitless]
eta = 202                   #climate feedback paramter [W/m^2]
beta = 1.9                  #climate feedback parameter [W/m^2/°C]
a = 5.35                    #CO2 forcing coefficient [W/m^2]
pre_ind = 280               #preindustrial CO2 concentration [parts per million; ppm];
T_PI = 13.7                 #pre-industrial average temperature

#SDE parameters
theta = 1/5                 #5 year window for autocorrelation
sigma = np.sqrt(dt/yrToSec) #magnitude of stochasticity (1 degree over 80 years)

#social model parameters
M = 25                      #memory bank length - we choose a fixed value here
CM = np.zeros(int(years/dt))                    #committed minority
CM[0] = 0.1
r = CM[0]                       #speaking rate of opinion A is initially E
r_track = np.zeros(int(years/dt))  #to track the speaking rate of A over time without interferring with root finding
r_track[0] = r
error = 1e-6                #error bound in root finding search

#extreme events (EE) parameters
lambda_0 = 16
#lambda_0 = 48/12            #number of baseline EE per year from 2011-2022 data (no human influence)
EE = np.zeros(int(years/dt)) #to track number of extreme events per timestep
EE[0] = 0
mu = 0.0045
#mu = 0.014                #shift in E after extreme event occurs
delta = 0.002                #decay rate of the shift in E

#initial conditions
T = np.zeros(int(years/dt)) #temperature
T[0] = 14.9                 #temp in 2020
conc = np.zeros(int(years/dt)) #concentration of CO2 in ppm
conc[0] = 410   #2020 CO2 concentration from IPCC report
emis_ppm_actual = np.zeros(int(years/dt)) #emissions as determined by social model
W = np.zeros(int(years/dt)) #noise term for SDE

#import and interpolate IPCC CO2 emissions projections, data.___ indicates scenario used
count = int(10/dt) #10 is number of years between IPCC projections and divided by dt gives size of linspace needed to interpolate data
data = pd.read_csv(r'Carbon_dioxide_Gt_CO2_yr.csv')
emis_best = np.zeros(int(years/dt))
emis_worst = np.zeros(int(years/dt))
for i in range(1,len(data.years)-1):
    emis_best[(i-1)*count:(i-1)*count+count] = np.linspace(data.ssp119[i], data.ssp119[i+1], num=count, endpoint=False)
    emis_worst[(i-1)*count:(i-1)*count+count] = np.linspace(data.ssp585[i], data.ssp585[i+1], num=count, endpoint=False)
emis_ppm_best = dt*emis_best/7.76 #convert from Gt to ppm CO2, multiply by dt to average emissions across the year
emis_ppm_worst = dt*emis_worst/7.76

repetitions=10
r_end = np.zeros(repetitions) #HERE HERE HERE
CM_max = np.zeros(repetitions) #HERE HERE HERE
#run simulation
for rep in range(repetitions):
    #set things to 0 as needed
    r = CM[0]
    for t in range(1, int(years/dt)):
        #get equilibrium value from social model by iterating until within error bound
        r_new = -1  #place holder value so that we can go into the while loop
        while abs(r-r_new)>error:   #iterate until we reach a fixed point
            r_temp = r
            if r>1:         #so that numerical errors don't cause r_new=nan
                r=1
            if (M%2):
                r_new = CM[t-1] + (1-CM[t-1])*(sum([binom.pmf(i,M,r) for i in range(int((M+1)/2), M+1)]))
            else:
                r_new = CM[t-1] + (1-CM[t-1])*(sum([binom.pmf(i,M,r) for i in range(int(M/2)+1, M+1)]) + 0.5*binom.pmf(int(M/2),M,r))
            r = r_new
            r_new = r_temp
        r_track[t] = r
        if r==1: #HERE HERE HERE
            break #HERE HERE HERE
        #get emissions level to plug into climate model
        emis_ppm_actual[t] = r*emis_ppm_best[t] + (1-r)*emis_ppm_worst[t]
        conc[t] = conc[t-1] + emis_ppm_actual[t]    
        #define a random number to add stochasticity
        rand = np.random.normal(0, 1, None)
        #solve stoch via Euler Maruyama
        W[t] = W[t-1] - theta*W[t-1]*dt + sigma*rand
        #solve climate model via Euler Maruyama
        T[t] = T[t-1] + dt*(S*(1-alpha)/4 - (eta+beta*T[t-1]) + a*np.log(conc[t-1]/pre_ind))/C + W[t]
        #temperature impacts the social model
        lam = dt*lambda_0*(1.1**(T[t]-T_PI))  #current rate of EE per timestep
        EE[t] = rnd.poisson(lam)                 #number of extreme events in the timestep
        CM[t] = CM[0] + mu*sum(EE[i]*np.exp(-delta*(t-i)) for i in range(t+1))    
    #save data, each new row is a new simulation
    r_end[rep] = r     #HERE HERE HERE
    CM_max[rep] = np.max(CM[0:t]) #HERE HERE HERE
    #pd.DataFrame(T).T.to_csv(r'/home/swyse/Documents/Graduate School/Thesis/Figures/SCM/data_temp2.csv', index=False, header=False, mode='a')    #temperature
    #pd.DataFrame(r_track).T.to_csv(r'/home/swyse/Documents/Graduate School/Thesis/Figures/SCM/data_r2.csv', index=False, header=False, mode='a') #the speaking rate of opinion A over time
    #pd.DataFrame(CM).T.to_csv(r'/home/swyse/Documents/Graduate School/Thesis/Figures/SCM/data_CM2.csv', index=False, header=False, mode='a')     #committed minority proportion (drives speaking rate of A)
    #pd.DataFrame(EE).T.to_csv(r'/home/swyse/Documents/Graduate School/Thesis/Figures/SCM/data_EE2.csv', index=False, header=False, mode='a')     #extreme event occurance
        

t2 = time.time() - t1
