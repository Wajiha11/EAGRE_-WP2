#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:20:39 2021

@author: mmwr
"""

import firedrake as fd
import math
import numpy as np
from matplotlib import animation, pyplot as plt
'''
Select the case:
    case 1 : VP solved by firedrake to compute the eqs of motion by using fd. derivative
    case 2 : VP solved to weak forms of the imposed eqs of motions manually
    Note: First Uncomment the (EXACT SOLUTION) section to get the exact solution, then comment it and run either case 1 or case 2.
    If I run the code after uncommenting the (EXACT SOLUTION) then case 1 and case 2 does not produce results. Still searching why.
    case 1 gives wrong results while case 2 gives results close to exact solution.
'''
case = 2

##  mesh ##
n = 15
mesh = fd.UnitSquareMesh(n, n)
x = fd.SpatialCoordinate(mesh)


######### PARAMETERS   ###############

g = 9.8 # gravitational acceleration
H = 1 # water depth
t = 0
m = 3
k = (2* fd.pi * m) /(2* fd.pi)
print('k =',k)
Tp = (2* fd.pi ) /k
print('Tp =',Tp)
t_end = Tp# time of simulation in sec
print('End time =', t_end)
# dt = 0.005 # time step [s]  n/t_end
dx= 1/n
dt = 0.005 # dx/(16*np.pi)
print('time step size =', dt)
ts = int(t_end/dt)
print('time_steps =', ts)
theta = fd.pi/4
xvals = np.linspace(0, 0.99, 100) 

## Define function spaces  ##
V = fd.FunctionSpace(mesh, "CG", 1)

## Define Functions ##
trial = fd.TrialFunction(V)
print( 'The shape of trail funcion is= ',np.shape(trial))
v = fd.TestFunction(V)
phi = fd.Function(V, name = "phi") # phi^n
phi_new = fd.Function(V, name = "phi_new") # phi^n+1
eta = fd.Function(V, name =  "eta") # eta^n
eta_new = fd.Function(V, name =  "eta_new") # eta^n+1

######  Define initial conditions  ############
Cc = 1
Dc = 1
E = (Cc*Cc + Dc*Dc)**0.5 # E', E=E'A, the constant A is chosen to be 1.
theta = np.arctan(Cc/Dc)

ic1 = phi.interpolate(E * fd.cos(k*x[0])*fd.sin(k*t + theta))
ic2 = eta.interpolate(-E* k * fd.cos(k*x[0])*fd.cos(k*t + theta))


phi.assign(ic1)
phi_new.assign(ic1)

x_slice = np.linspace(0, 1,n)

eta.assign(ic2)
eta_new.assign(ic2)

plt.figure(1)
plt.title(r'Eta value at the centre of the domain')
plt.xlabel(r'$x$ ')
plt.ylabel(r'$\eta$ ')

# ## EXACT SOLUTION #########
# outfile_phi_exact = fd.File("results_exact/phi.pvd")
# outfile_eta_exact = fd.File("results_exact/eta.pvd")

# while ( t <= t_end):
#     phi_exact= phi.interpolate(E * fd.cos(k*x[0]) * fd.sin(k*t + theta))
#     eta_exact = eta.interpolate(-E * k * fd.cos(k*x[0]) * fd.cos(k*t + theta))
#     t += dt
#     outfile_eta_exact.write( eta_exact )
#     outfile_phi_exact.write( phi_exact )
# etavals = np.array([eta_exact.at(x, 0.5) for x in xvals])
# plt.plot(xvals, etavals)


### VARIATIONAL PRINCIPLE #########
if case ==1:
    print("You have selected case 1 : VP solved by firedrake to compute the eqs of motion ")

    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) - (1/2 * H * fd.inner(fd.grad(phi), fd.grad(phi))) - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx
    
    eta_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1
    print('eta_expression=',eta_expr)
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    print('phi_expression=',phi_expr)
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_case1/phi.pvd")
    outfile_eta = fd.File("results_case1/eta.pvd")
    
    
    ######### TIME LOOP ############
    while (t <= t_end):
        fd.solve(eta_expr == 0, eta_new)
        eta.assign(eta_new)
        fd.solve(phi_expr == 0 , phi_new)
        phi.assign(phi_new)
        t += dt
        outfile_eta.write( eta )
        outfile_phi.write( phi )
    etavals = np.array([eta.at(x, 0.5) for x in xvals])
    plt.plot(xvals, etavals) 
        
elif case == 2:
    print("You have selected case 2 : VP solved to weak forms of the imposed eqs of motions manually")
    a_phi = fd.inner( trial, v ) * fd.dx
    # L_phi = fd.inner( (phi - dt * eta_new), v ) * fd.dx 
    L_phi = fd.inner( (phi - dt * eta), v ) * fd.dx
    LVP_phi = fd.LinearVariationalProblem(a_phi, L_phi, phi)
    
    a_eta = fd.inner( trial, v ) * fd.dx
    # L_eta = (v * eta + dt * fd.inner(fd.grad(phi), fd.grad(v)) ) * fd.dx
    L_eta =  ( fd.inner( eta,v) + dt * fd.inner(fd.grad(phi), fd.grad(v)) ) * fd.dx 
    LVP_eta = fd.LinearVariationalProblem(a_eta, L_eta, eta)
    LVS_phi = fd.LinearVariationalSolver(LVP_phi)
    LVS_eta = fd.LinearVariationalSolver(LVP_eta)
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_case2/phi.pvd")
    outfile_eta = fd.File("results_case2/eta.pvd")   
    
    while (t <= t_end):
        LVS_phi.solve()
        LVS_eta.solve()
        # LVS_phi.solve()
        t += dt
        outfile_eta.write( eta )
        outfile_phi.write( phi )
        
    
    etavals = np.array([eta.at(x, 0.5) for x in xvals])
    plt.plot(xvals, etavals)
    plt.title(r'Eta value at the centre of the domain')
    plt.xlabel(r'$x$ ')
    plt.ylabel(r'$\eta$ ')
        
    
    
    
    
plt.show()     
    
    
    

    
print('*************** PROGRAM ENDS ******************')
