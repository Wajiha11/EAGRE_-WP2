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
t_end = 1*Tp# time of simulation in sec
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

phie= fd.Function(V, name = "phi_exact") 
etae = fd.Function(V, name = "eta_exact") 

######  Define initial conditions  ############
Cc = 1
Dc = 1
E = (Cc*Cc + Dc*Dc)**0.5 # E', E=E'A, the constant A is chosen to be 1.
theta = np.arctan(Cc/Dc)

ic1 = phi.interpolate(E * fd.cos(k*x[0]) * fd.sin(k*t + theta))
ic2 = eta.interpolate(-E* k * fd.cos(k*x[0])*fd.cos(k*t + theta))


phi.assign(ic1)
phi_new.assign(ic1)

x_slice = np.linspace(0, 1,n)

eta.assign(ic2)
eta_new.assign(ic2)

######## FIGURE SETTINGS ###########

# plt.figure(1)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title(r'$\eta$ value at the centre of the domain')
ax2.set_title(r'$\phi$ value at the centre of the domain')
ax1.set_xlabel(r'$x$ ')
ax1.set_ylabel(r'$\eta$ ')
ax2.set_xlabel(r'$x$ ')
ax2.set_ylabel(r'$\phi$ ')

# ###### EXACT SOLUTION #########
# outfile_phi_exact = fd.File("results_exact/phi.pvd")
# outfile_eta_exact = fd.File("results_exact/eta.pvd")

# while ( t <= t_end):
#     phi_exact= phie.interpolate(E * fd.cos(k*x[0]) * fd.sin(k*t + theta))
#     eta_exact = etae.interpolate(-E * k * fd.cos(k*x[0]) * fd.cos(k*t + theta))
#     t += dt
#     # outfile_eta_exact.write( eta_exact )
#     # outfile_phi_exact.write( phi_exact )
# etaevals = np.array([eta_exact.at(x, 0.5) for x in xvals])
# phievals = np.array([phi_exact.at(x, 0.5) for x in xvals])
# print('phi_exact =', phievals)
# print('eta_exact =', etaevals)
# ax1.plot(xvals, etaevals, '--',label = '$\eta$ Exact')
# # #ax1.legend(loc='upper right')
# ax2.plot(xvals, phievals, '--',label = '$\phi$ Exact')
# # #ax2.legend(loc='upper left')


### VARIATIONAL PRINCIPLE #########
if case ==1:
    print("You have selected case 1 : VP solved by firedrake to compute the eqs of motion ")

    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) - (1/2 * H * fd.inner(fd.grad(phi), fd.grad(phi))) - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx
    
    eta_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1
    print('eta_expression=',eta_expr)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new))
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    print('phi_expression=',phi_expr)
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_case1/phi.pvd")
    outfile_eta = fd.File("results_case1/eta.pvd")
    
    
    ######### TIME LOOP ############
    while (t <= t_end):
        eta_expr.solve()
        t+= dt
        phi_expr.solve()
        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )


    eta1vals = np.array([eta_new.at(x, 0.5) for x in xvals])
    phi1vals = np.array([phi_new.at(x, 0.5) for x in xvals])

    
    ax1.plot(xvals, eta1vals, label = 'Case1 : $\eta$')
    ax1.legend(loc=2)
    ax2.plot(xvals,phi1vals, label = 'Case1 : $\phi$')
    ax2.legend(loc=1)
    
    
elif case == 2:
    print("You have selected case 2 : VP solved to weak forms of the imposed eqs of motions manually")
      
    phi_full = (v* (phi_new - phi)/dt  + g*v*eta) * fd.dx
    phi_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_full, phi_new))
    
    eta_full = (v * (eta_new - eta)/dt - H * fd.inner(fd.grad(v), fd.grad(phi_new)))* fd.dx
    eta_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_full, eta_new))
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_case3/phi.pvd")
    outfile_eta = fd.File("results_case3/eta.pvd")   
    
    while t<= t_end:
        phi_full.solve()
        t+= dt
        eta_full.solve()
        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )
        
    eta2vals = np.array([eta_new.at(x, 0.5) for x in xvals])
    phi2vals = np.array([phi_new.at(x, 0.5) for x in xvals])
    # print('phi_case2 =', phi2vals)
    # print('eta_case2 =', eta2vals)
    
    ax1.plot(xvals, eta2vals, label = 'Case2 : $\eta$')
    ax1.legend(loc=2)
    ax2.plot(xvals,phi2vals, label = 'Case2 : $\phi$')
    ax2.legend(loc=1)
    
plt.show()     
    
    
    

    
print('*************** PROGRAM ENDS ******************')
