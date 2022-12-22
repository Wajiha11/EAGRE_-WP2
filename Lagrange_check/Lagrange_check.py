!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:08:36 2022

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
'''
case = 1

##  mesh ##
nx = 30#99
ny = 1#99


# n = 90
# mesh = fd.UnitSquareMesh(n, n)
# V = fd.FunctionSpace(mesh, "CG", 1)
# x,y = fd.SpatialCoordinate(mesh)

Lx = 1
Ly = 1
xvals = np.linspace(0, 0.99, 100)
yvals = np.linspace(0, 0.99, 100) 
yslice = 0.5
xslice = 0.5

mesh = fd.IntervalMesh(nx, Lx)
extmesh = fd.ExtrudedMesh(mesh, layers = ny,
                         layer_height= 1)
V = fd.FunctionSpace(extmesh, "CG", 1, vfamily= "Lagrange", vdegree = 20)
x,y = fd.SpatialCoordinate(extmesh)

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
# dx= 1/n
dt = 0.001 # dx/(16*np.pi)
print('time step size =', dt)

ts = int(t_end/dt)
print('time_steps =', ts)


## Define function spaces  ##
# V = fd.FunctionSpace(mesh, "CG", 1)


#### Define Function Spaces ####
trial = fd.TrialFunction(V)
# print( 'The shape of trail funcion is= ',np.shape(trial))

v = fd.TestFunction(V)
v2 = fd.TestFunction(V)

phi1 = fd.Function(V, name = "phi1") # phi^n
phi1_new = fd.Function(V, name = "phi1_new") # phi^n+1
eta1 = fd.Function(V, name =  "eta1") # eta^n
eta1_new = fd.Function(V, name =  "eta1_new") # eta^n+1


phi2 = fd.Function(V, name = "phi2") # phi^n
phi2_new = fd.Function(V, name = "phi2_new") # phi^n+1
eta2 = fd.Function(V, name =  "eta2") # eta^n
eta2_new = fd.Function(V, name =  "eta2_new") # eta^n+1


phie= fd.Function(V, name = "phi_exact") 
etae = fd.Function(V, name = "eta_exact") 

######  Define initial conditions  ############
A = 1
B = 1

phi = fd.Function(V, name = "phi") # phi^n
eta = fd.Function(V, name =  "eta") # eta^n


ic1 = phi.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( -A*fd.sin(w*t) + B*fd.cos(w*t) ) * (g/w) )
ic2 = eta.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( A*fd.cos(w*t) + B*fd.sin(w*t) ) )



phi1.assign(ic1)
phi1_new.assign(ic1)
eta1.assign(ic2)
eta1_new.assign(ic2)

phi2.assign(ic1)
phi2_new.assign(ic1)
eta2.assign(ic2)
eta2_new.assign(ic2)



###### EXACT SOLUTION #########

phi_exact= phie.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( -A*fd.sin(w*t_end) + B*fd.cos(w*t_end) )*(g/w) )
eta_exact = etae.interpolate( fd.cos(k1 * x) * fd.cos(k2 * y) * ( A*fd.cos(w*t_end) + B*fd.sin(w*t_end) ) )


etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
phievals = np.array([phi_exact.at(x, yslice) for x in xvals])

etaevalsy = np.array([eta_exact.at(xslice, y) for y in yvals])
phievalsy =  np.array([phi_exact.at(xslice, y) for y in yvals])


### VARIATIONAL PRINCIPLE #########
if case == 1:
    print("You have selected case 1 : VP solved by firedrake to compute the eqs of motion ")

    VP = ( fd.inner ((eta1_new - eta1)/dt , phi1) - fd.inner(phi1_new , (eta1_new/dt)) - (1/2 * H * fd.inner(fd.grad(phi1), fd.grad(phi1))) - (1/2 * g * fd.inner(eta1_new,eta1_new)) ) * fd.dx
    
    eta1_expr = fd.derivative(VP, phi1, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1
    print('eta_expression=',eta1_expr)
    eta1_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta1_expr, eta1_new))
    phi1_expr = fd.derivative(VP, eta1_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    print('phi_expression=',phi1_expr)
    phi1_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi1_expr, phi1_new))
    
    ###### OUTPUT FILES ##########
    outfile_phi1 = fd.File("results_case1/phi1.pvd")
    outfile_eta1 = fd.File("results_case1/eta1.pvd")
    
    
    ######### TIME LOOP ############
    while (t <= t_end):
        eta1_expr.solve()
        t+= dt
        phi1_expr.solve()
        
        outfile_eta1.write( eta1_new )
        outfile_phi1.write( phi1_new )
        
        # set-up next time-step
        phi1.assign(phi1_new)
        eta1.assign(eta1_new)


    eta1vals = np.array([eta1_new.at(x, yslice) for x in xvals])
    phi1vals = np.array([phi1_new.at(x, yslice) for x in xvals])
    
    eta1valsy = np.array([eta1_new.at(xslice, y) for y in yvals])
    phi1valsy =  np.array([phi1_new.at(xslice, y) for y in yvals])
    
    error_e1x = etaevals - eta1vals
    error_p1x = phievals - phi1vals
    error_e1y = etaevalsy - eta1valsy
    error_p1y =  phievalsy - phi1valsy 
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
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
    
    ax1.plot(xvals, eta1vals, label = 'Case 1 : $\eta_x$')
    ax2.plot(xvals,phi1vals, label = 'Case 1 : $\phi_x$')
    ax3.plot(yvals, eta1valsy,label = 'Case 1 : $\eta_y$')
    ax4.plot(yvals,phi1valsy, label = 'Case 1 : $\phi_y$')
    
    ax1.plot(xvals, etaevals, '--',label = '$Exact: \eta_x$ ')
    ax2.plot(xvals, phievals, '--',label = '$Exact: \phi_x$ ')
    ax3.plot(yvals, etaevalsy, '--',label = '$Exact: \eta_y$ ')
    ax4.plot(yvals, phievalsy, '--',label = '$Exact: \phi_y$ ')
    
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)
    
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    
    ax1.set_title('$ Relative error for $\eta$ in $x$ direction')
    ax2.set_title('$ Relative error  for $\phi$ in $x$ direction')
    ax3.set_title('$ Relative error for $\eta$ in $y$ direction')
    ax4.set_title('$ Relative error for $\phi$ in $y$ direction')
    
    ax1.plot(xvals, error_e1x, label = 'Case1 : $\eta_x$')
    ax2.plot(xvals, error_p1x, label = 'Case1 : $\phi_x$')
    ax3.plot(yvals, error_e1y, label = 'Case1 : $\eta_y$')
    ax4.plot(yvals, error_p1y, label = 'Case1 : $\phi_y$')
    
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)

    
elif case == 2:
      
    print('You have selected case 2: First calculates eta^n+1 and then phi^(n+1) like the given problem')
    eta2_full = (v * (eta2_new - eta2)/dt - H * fd.inner(fd.grad(v), fd.grad(phi2)))* fd.dx
    eta2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta2_full, eta2_new))
    
    phi2_full = (v* (phi2_new - phi2)/dt  + g*v*eta2_new) * fd.dx
    phi2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi2_full, phi2_new))
    
    
    ###### OUTPUT FILES ##########
    outfile_phi2 = fd.File("results_case2/phi.pvd") # case 4 in shallow2.py
    outfile_eta2 = fd.File("results_case2/eta.pvd")
    
    while t<= t_end:
        eta2_full.solve()
        t+= dt
        phi2_full.solve()
        outfile_eta2.write( eta2_new )
        outfile_phi2.write( phi2_new )
        
        # set-up next time-step
        phi2.assign(phi2_new)
        eta2.assign(eta2_new)
        
    eta2vals = np.array([eta2_new.at(x, yslice) for x in xvals])
    phi2vals = np.array([phi2_new.at(x, yslice) for x in xvals])
    
    eta2valsy = np.array([eta2_new.at(xslice, y) for y in yvals])
    phi2valsy =  np.array([phi2_new.at(xslice, y) for y in yvals])
    # print('phi_case2 =', phi3vals)
    # print('eta_case2 =', eta3vals)
    
    error_e2x = etaevals - eta2vals
    error_p2x = phievals - phi2vals  
    error_e2y = etaevalsy - eta2valsy
    error_p2y =  phievalsy - phi2valsy
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
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
    
    ax1.plot(xvals, eta2vals, label = 'Case 2 : $\eta_x$')
    ax2.plot(xvals,phi2vals, label = 'Case 2 : $\phi_x$')
    ax3.plot(yvals, eta2valsy,label = 'Case 2  : $\eta_y$')
    ax4.plot(yvals,phi2valsy, label = 'Case 2 : $\phi_y$')
    
    ax1.plot(xvals, etaevals, '--',label = '$Exact: \eta_x$ ')
    ax2.plot(xvals, phievals, '--',label = '$Exact: \phi_x$ ')
    ax3.plot(yvals, etaevalsy, '--',label = '$Exact: \eta_y$ ')
    ax4.plot(yvals, phievalsy, '--',label = '$Exact: \phi_y$ ')
    
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)
    
    
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


    
    ax1.set_title('$ Relative error for $\eta$ in $x$ direction')
    ax2.set_title('$ Relative error  for $\phi$ in $x$ direction')
    ax3.set_title('$ Relative error for $\eta$ in $y$ direction')
    ax4.set_title('$ Relative error for $\phi$ in $y$ direction')
    

    
    ax1.plot(xvals, error_e2x, label = 'Case 2 : $\eta_x$')
    ax2.plot(xvals, error_p2x, label = 'Case 2 : $\phi_x$')
    ax3.plot(yvals, error_e2y, label = 'Case 2 : $\eta_y$')
    ax4.plot(yvals, error_p2y, label = 'Case 2 : $\phi_y$')
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    ax3.legend(loc=2)
    ax4.legend(loc=1)
    
    

elif case == 3:
    
     print("Case 1 and Case 2 will be solved and compared with the exact solution")
     print('Case 2 has been initialised')
    
     eta2_full = (v2 * (eta2_new - eta2)/dt - H * fd.inner(fd.grad(v2), fd.grad(phi2)))* fd.dx
     eta2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta2_full, eta2_new))
    
     phi2_full = (v2 * (phi2_new - phi2)/dt  + g*v2*eta2_new) * fd.dx
     phi2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi2_full, phi2_new))
    
    
     ###### OUTPUT FILES ##########
     outfile_phi2 = fd.File("results_case2/phi2.pvd") # case 4 in shallow2.py
     outfile_eta2 = fd.File("results_case2/eta2.pvd")
    
     while t<= t_end:
        eta2_full.solve()
        t+= dt
        phi2_full.solve()
        outfile_eta2.write( eta2_new )
        outfile_phi2.write( phi2_new )
        
        # set-up next time-step
        phi2.assign(phi2_new)
        eta2.assign(eta2_new)
    
    
        
     eta2vals = np.array([eta2_new.at(x, yslice) for x in xvals])
     phi2vals = np.array([phi2_new.at(x, yslice) for x in xvals])
    
     eta2valsy = np.array([eta2_new.at(xslice, y) for y in yvals])
     phi2valsy =  np.array([phi2_new.at(xslice, y) for y in yvals])
   
     error_e2x = etaevals - eta2vals
     error_p2x = phievals - phi2vals  
     error_e2y = etaevalsy - eta2valsy
     error_p2y =  phievalsy - phi2valsy
    
     print('Case 2 has been completed')
    
   

     print("Case 1 has been initialized")     
    
     VP = ( fd.inner ((eta1_new - eta1)/dt , phi1) - fd.inner(phi1_new , (eta1_new/dt))\
           - (1/2 * H * fd.inner(fd.grad(phi1), fd.grad(phi1))) - (1/2 * g * fd.inner(eta1_new,eta1_new)) ) * fd.dx
    
     eta1_expr = fd.derivative(VP, phi1, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1
     print('eta_expression=',eta1_expr)
     eta1_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta1_expr, eta1_new))
     phi1_expr = fd.derivative(VP, eta1_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
     print('phi_expression=',phi1_expr)
     phi1_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi1_expr, phi1_new))
    
    ###### OUTPUT FILES ##########
     outfile_phi1 = fd.File("results_case1/phi1.pvd")
     outfile_eta1 = fd.File("results_case1/eta1.pvd")
    
    
    ######### TIME LOOP ############
     t = 0
     while (t <= t_end):
        eta1_expr.solve()
        t+= dt
        phi1_expr.solve()
        
        outfile_eta1.write( eta1_new )
        outfile_phi1.write( phi1_new )
        
        # set-up next time-step
        phi1.assign(phi1_new)
        eta1.assign(eta1_new)

    
     eta1vals = np.array([eta1_new.at(x, yslice) for x in xvals])
     phi1vals = np.array([phi1_new.at(x, yslice) for x in xvals])
    
     eta1valsy = np.array([eta1_new.at(xslice, y) for y in yvals])
     phi1valsy =  np.array([phi1_new.at(xslice, y) for y in yvals])
    
     error_e1x = etaevals - eta1vals
     error_p1x = phievals - phi1vals
     error_e1y = etaevalsy - eta1valsy
     error_p1y =  phievalsy - phi1valsy
    
     print("Case 1 has been completed") 

    # print('phi_case2 =', phi3vals)
    # print('eta_case2 =', eta3vals)
    
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
    
     ax1.plot(xvals, eta1vals, 'c-', label = 'Case 1 : $\eta_x$')
     ax2.plot(xvals,phi1vals, 'c-', label = 'Case 1 : $\phi_x$')
     ax3.plot(yvals, eta1valsy, 'c-',label = 'Case 1 : $\eta_y$')
     ax4.plot(yvals,phi1valsy, 'c-', label = 'Case 1 : $\phi_y$')
    
     ax1.plot(xvals, eta2vals, 'r:', label = 'Case 2 : $\eta_x$')
     ax2.plot(xvals,phi2vals, 'r:',label = 'Case 2 : $\phi_x$')
     ax3.plot(yvals, eta2valsy,'r:',label = 'Case 2  : $\eta_y$')
     ax4.plot(yvals,phi2valsy, 'r:',label = 'Case 2 : $\phi_y$')
    
     ax1.plot(xvals, etaevals, '--',label = '$Exact: \eta_x$ ')
     ax2.plot(xvals, phievals, '--',label = '$Exact: \phi_x$ ')
     ax3.plot(yvals, etaevalsy, '--',label = '$Exact: \eta_y$ ')
     ax4.plot(yvals, phievalsy, '--',label = '$Exact: \phi_y$ ')
    
    
     ax1.legend(loc=2)
     ax2.legend(loc=1)
     ax3.legend(loc=2)
     ax4.legend(loc=1)
    
     fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
     ax1.set_title('$ Relative error for $\eta$ in $x$ direction')
     ax2.set_title('$ Relative error  for $\phi$ in $x$ direction')
     ax3.set_title('$ Relative error for $\eta$ in $y$ direction')
     ax4.set_title('$ Relative error for $\phi$ in $y$ direction')
    
     ax1.plot(xvals, error_e1x, 'c-',label = 'Case1 : $\eta_x$')
     ax2.plot(xvals, error_p1x, 'c-',label = 'Case1 : $\phi_x$')
     ax3.plot(yvals, error_e1y, 'c-',label = 'Case1 : $\eta_y$')
     ax4.plot(yvals, error_p1y, 'c-',label = 'Case1 : $\phi_y$')
    
     ax1.plot(xvals, error_e2x, 'r:', label = 'Case 2 : $\eta_x$')
     ax2.plot(xvals, error_p2x, 'r:', label = 'Case 2 : $\phi_x$')
     ax3.plot(yvals, error_e2y, 'r:', label = 'Case 2 : $\eta_y$')
     ax4.plot(yvals, error_p2y, 'r:', label = 'Case 2 : $\phi_y$')
    
     ax1.legend(loc=2)
     ax2.legend(loc=1)
     ax3.legend(loc=2)
     ax4.legend(loc=1)
    
    
plt.show()     
        
    

    
print('*************** PROGRAM ENDS ******************')
