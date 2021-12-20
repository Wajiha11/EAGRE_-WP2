#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:35:43 2021

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
case = 1

##  mesh ##
n = 79 #99
mesh = fd.UnitSquareMesh(n, n)
x,y = fd.SpatialCoordinate(mesh)
Lx = 1
Ly = 1
xvals = np.linspace(0, 0.99, 100)
yvals = np.linspace(0, 0.99, 100) 
yslice = 0.5
xslice = 0.5

######### PARAMETERS   ###############

g = 9.8 # gravitational acceleration
H = 1 # water depth
t = 0
m = 2
m1 = 2
m2 = 2

k1 = (2* fd.pi * m1) /Lx
print('k1 =',k1)

k2 = (2* fd.pi * m2) /Ly
print('k2 =',k2)

c = np.sqrt(g*H)

w = c * np.sqrt(k1**2 + k2**2)

k = np.sqrt(k1**2 + k2**2)
print('k =',k)

Tp = (2* fd.pi ) /w
print('Tp =',Tp)

t_end = 2*Tp# time of simulation in sec
print('End time =', t_end)

# dt = 0.005 # time step [s]  n/t_end
dx= 1/n
dt = 0.001 # dx/(16*np.pi)
print('time step size =', dt)

ts = int(t_end/dt)
print('time_steps =', ts)


## Define function spaces  ##
V = fd.FunctionSpace(mesh, "CG", 1)


#### Define Function Spaces ####
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
A = 1
B = 1


ic1 = phi.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( -A*fd.sin(w*t) + B*fd.cos(w*t) ) * (g/w) )
ic2 = eta.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( A*fd.cos(w*t) + B*fd.sin(w*t) ) )


phi.assign(ic1)
phi_new.assign(ic1)


eta.assign(ic2)
eta_new.assign(ic2)

######## FIGURE SETTINGS ###########

# plt.figure(1)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title(r'$\eta$ value in $x$ direction')
ax2.set_title(r'$\phi$ value in $x$ direction')
ax3.set_title(r'$\eta$ value in $y$ direction')
ax4.set_title(r'$\phi$ value in $y$ direction')

ax1.set_xlabel(r'$x$ ')
ax1.set_ylabel(r'$\eta$ ')
ax2.set_xlabel(r'$x$ ')
ax2.set_ylabel(r'$\phi$ ')
ax3.set_xlabel(r'$y$ ')
ax3.set_ylabel(r'$\eta$ ')
ax4.set_xlabel(r'$y$ ')
ax4.set_ylabel(r'$\phi$ ')

###### EXACT SOLUTION #########

phi_exact= phie.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( -A*fd.sin(w*t_end) + B*fd.cos(w*t_end) )*(g/w) )
eta_exact = etae.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( A*fd.cos(w*t_end) + B*fd.sin(w*t_end) ) )

# outfile_phi_exact = fd.File("results_exact/phi.pvd")
# outfile_eta_exact = fd.File("results_exact/eta.pvd")

# while ( t <= t_end):
#     phi_exact= phie.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( -A*fd.sin(w*t) + B*fd.cos(w*t) ) * (g/w) )
#     eta_exact = etae.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( A*fd.cos(w*t) + B*fd.sin(w*t) ) )
#     t += dt
#     outfile_eta_exact.write( eta_exact )
#     outfile_phi_exact.write( phi_exact )


etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
phievals = np.array([phi_exact.at(x, yslice) for x in xvals])

etaevalsy = np.array([eta_exact.at(xslice, y) for y in yvals])
phievalsy =  np.array([phi_exact.at(xslice, y) for y in yvals])

ax1.plot(xvals, etaevals, '--',label = '$Exact: \eta_x$ ')
ax2.plot(xvals, phievals, '--',label = '$Exact: \phi_x$ ')
ax3.plot(yvals, etaevalsy, '--',label = '$Exact: \eta_y$ ')
ax4.plot(yvals, phievalsy, '--',label = '$Exact: \phi_y$ ')

# #ax1.legend(loc='upper right')
# #ax2.legend(loc='upper left')


### VARIATIONAL PRINCIPLE #########
if case == 1:
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
        
        # set-up next time-step
        phi.assign(phi_new)
        eta.assign(eta_new)


    eta1vals = np.array([eta_new.at(x, yslice) for x in xvals])
    phi1vals = np.array([phi_new.at(x, yslice) for x in xvals])
    
    eta1valsy = np.array([eta_new.at(xslice, y) for y in yvals])
    phi1valsy =  np.array([phi_new.at(xslice, y) for y in yvals])
    
    ax1.plot(xvals, eta1vals, label = 'Case1 : $\eta_x$')
    ax2.plot(xvals,phi1vals, label = 'Case1 : $\phi_x$')
    ax3.plot(yvals, eta1valsy,label = 'Case1 : $\eta_y$')
    ax4.plot(yvals,phi1valsy, label = 'Case1 : $\phi_y$')
    
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)

    
elif case == 2:
      
    print('You have selected case 2: First calculates eta^n+1 and then phi^(n+1) like the given problem')
    eta2_full = (v * (eta_new - eta)/dt - H * fd.inner(fd.grad(v), fd.grad(phi)))* fd.dx
    eta2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta2_full, eta_new))
    
    phi2_full = (v* (phi_new - phi)/dt  + g*v*eta_new) * fd.dx
    phi2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi2_full, phi_new))
    
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_case2/phi.pvd") # case 4 in shallow2.py
    outfile_eta = fd.File("results_case2/eta.pvd")
    
    while t<= t_end:
        eta2_full.solve()
        t+= dt
        phi2_full.solve()
        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        phi.assign(phi_new)
        eta.assign(eta_new)
        
    eta2vals = np.array([eta_new.at(x, yslice) for x in xvals])
    phi2vals = np.array([phi_new.at(x, yslice) for x in xvals])
    
    eta2valsy = np.array([eta_new.at(xslice, y) for y in yvals])
    phi2valsy =  np.array([phi_new.at(xslice, y) for y in yvals])
    # print('phi_case2 =', phi3vals)
    # print('eta_case2 =', eta3vals)
    ax1.plot(xvals, eta2vals, label = 'Case2 : $\eta_x$')
    ax2.plot(xvals,phi2vals, label = 'Case2 : $\phi_x$')
    ax3.plot(yvals, eta2valsy,label = 'Case2 : $\eta_y$')
    ax4.plot(yvals,phi2valsy, label = 'Case2 : $\phi_y$')
    
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)

    
    
plt.show()     
    
    
    

    
print('*************** PROGRAM ENDS ******************')
