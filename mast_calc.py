#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:28:34 2022

@author: mmwr
"""

import matplotlib.pyplot as plt
import numpy as np
import math as m
# from tabulate import tabulate

# I want to find the modulus of elaticity of pipe and plot the mode shapes
# The natural frequency should be 1/2 the time period of wave
# The wave time period is 1 sec



## Initial parameters

num_modes = 2
nodes = 100
L = 2                                      # Length  [m]
print('The length of the beam in m is:', L)
wall_thickness_mm = 2.5                   # [mm]
dia_mm = 125                                     # diameter [mm]
x = np.linspace(0, L, nodes+1)

## Convert mm to m
dia = dia_mm/1000                                   # diameter [m]
wall_thick = wall_thickness_mm /1000                # [m]


r_out = dia/2                                       # outer radius [m]
r_in = r_out - wall_thick                           # inner radius [m]
area = m.pi * ( (r_out )**2 - (r_in)**2 )           # [m^2]


# material properties
rho = 1400 #1450                                            # density [ kg/m^3 ]
m_L =  rho * area                                   # mass per length [kg m^-1] 
E = 3e+09                                           # Modulus Elasticity [N/m^2]

## Fig settings
tsize = 18 # font size of image title
size = 16 # font size of image axes

### Area moments of inertia about principal axis

Ixc  = m.pi/4 * ( (r_out )**4 - (r_in)**4 )
print('Ixc = Iyc =', Ixc)


### Calculate lamda 
lam = np.zeros(num_modes+1)

i = 0
while i <= num_modes:   

    lam[i] = (2*i - 1) * m.pi/2 
    i+= 1
    
## calculate sigma ( σ_i )
sigma= np.ones(num_modes+1)
 
if num_modes == 0:
      lam[0] = 1
      sigma[0] = 0
      
elif num_modes == 1:
    lam[0] = 1
    lam[1] = 1.87510407
    
    sigma[0] = 0
    sigma[1] = 0.734095514
elif num_modes == 2:
    lam[0] = 1
    lam[1] = 1.87510407
    lam[2]  = 4.69409113
    
    sigma[0] = 0
    sigma[1] = 0.734095514
    sigma[2] = 1.018467319
elif num_modes == 3:
    lam[0] = 1
    lam[1] = 1.87510407
    lam[2]  = 4.69409113
    lam[3] = 7.85475744
    
    sigma[0] = 0
    sigma[1] = 0.734095514
    sigma[2] = 1.018467319
    sigma[3] = 0.999224497
elif num_modes == 4:
    lam[0] = 1
    lam[1] = 1.87510407
    lam[2]  = 4.69409113
    lam[3] = 7.85475744
    lam[4] = 10.99554073
    
    sigma[0] = 0
    sigma[1] = 0.734095514
    sigma[2] = 1.018467319
    sigma[3] = 0.999224497
    sigma[4] = 1.000033551
else:    
    lam[0] = 1
    lam[1] = 1.87510407
    lam[2]  = 4.69409113
    lam[3] = 7.85475744
    lam[4] = 10.99554073
    lam[5] = 14.13716839
    
    sigma[0] = 0
    sigma[1] = 0.734095514
    sigma[2] = 1.018467319
    sigma[3] = 0.999224497
    sigma[4] = 1.000033551
    sigma[5] = 0.99999855


## calculate modes
y = np.zeros((num_modes+1, nodes+1))
i = 0

#  modes = cosh(λ_i *x/L) - np.cos(λ_i *x/L) -  σ_i *( np.sinh(λ * x/L) - sin(λ * x/L))
while i <= num_modes: 
    y[i, :] = np.cosh(lam[i] *x/L) - np.cos(lam[i] *x/L)\
                - sigma[i]*( np.sinh(lam[i] * x/L) - np.sin(lam[i]*x/L))
    
    i+= 1

## Plot the modes of the beam    
plt.title('Modes of the beam', fontsize = tsize )
i = 1
while i <= num_modes: 
    plt.plot(x, y[i], label = f'mode = {i: .1f}')
    i+= 1
    
plt.xlabel('x [m]', fontsize = size )
plt.ylabel('y ', fontsize = size )
plt.grid()
plt.legend(loc=4)

# calculate frequency and time period corresponding to each mode
# frequency = ( λ_i^2 / 2* pi * L^2)  * (EI/m)^1/2
fre = np.zeros(num_modes+1) 
i = 0
while i <= num_modes: 
    fre[i] = ( (lam[i]**2)/(2*m.pi* (L**2))) * m.sqrt( E * Ixc /m_L)
    i+= 1

fr = fre[1:]      # remove the first elemet as it corresponds to zero mode

print('Frequencies = ', fr)
T_n = 1/fr          # natural time period
print('Time period = ', T_n)


plt.show()

print('*************** PROGRAM ENDS ******************')
