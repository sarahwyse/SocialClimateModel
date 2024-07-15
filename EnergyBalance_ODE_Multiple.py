#climate ODE model - solved using Euler method for all climate scenarios

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#constants
yrToSec = 365.25*24*60*60   #model is in seconds, use this as part of C to convert to years
C = (3.985e6)*50/yrToSec    #upper-ocean heat capacity to 50m depth [J/m^2/°C] 
S = 1372                    #solar radiation (divide by 4 to get insolation) [W/m^2]  
alpha = 0.33                #albedo, or planetary reflectivity [unitless]
A = 202                     #climate feedback paramter [W/m^2]
B = 1.9                     #climate feedback parameter [W/m^2/°C]
a = 5.35                    #CO2 forcing coefficient [W/m^2]
pre_ind = 280               #preindustrial CO2 concentration [parts per million; ppm];

years = 80                  #years to run model for to match IPCC projections (over 80 years)
T_PI = 13.7                 #pre-industrial average temperature
dt = 1/365.25               #step size for solving deterministic part

count = int(10/dt) #10 is number of years between IPCC projections and divided by dt gives size of linspace needed to interpolate data
#import IPCC CO2 emissions projections, data.___ indicates scenario used
data = pd.read_csv(r'Carbon_dioxide_Gt_CO2_yr.csv')

plt.figure()
col = ["deepskyblue", "darkblue", "orange", "red", "darkred"]

for j in range(5,0,-1):
    emis = np.zeros(int(years/dt))
    for i in range(1,len(data.years)-1):
        emis[(i-1)*count:(i-1)*count+count] = np.linspace(data.iloc[i,j], data.iloc[i+1,j], num=count, endpoint=False)
    emis_ppm = dt*emis/7.76 #convert from Gt to ppm CO2, multiply by dt to average emissions across the year
    
    #add net emissions each year to compute CO2 concentration
    conc = np.zeros(int(years/dt)) #track concentration of CO2 in ppm
    conc[0] = 410   #2020 CO2 concentration from IPCC report
    for i in range(1,int(years/dt)):
        conc[i] = conc[i-1] + emis_ppm[i]
    
    #initial conditions
    T = np.zeros(int(years/dt))
    T[0] = 14.9  #pre-ind was 13.7
    
    #run simulation
    for t in range(1, int(years/dt)):
        T[t] = T[t-1] + dt*(S*(1-alpha)/4 - (A+B*T[t-1]) + a*np.log(conc[t-1]/pre_ind))/C
            
    #plot
    plt.plot(np.linspace(2020,2020+years, int(years/dt)), T-T_PI, color = col[j-1])

plt.legend(['SSP5-8.5', 'SSP3-7.0', 'SSP2-4.5', 'SSP1-2.6', 'SSP1-1.9'])    
plt.xlabel('Time (years)')
plt.ylabel('Temperature change ($\degree$C)')

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize

plt.show()