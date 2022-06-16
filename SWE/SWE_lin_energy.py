#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:39:04 2022

@author: mmwr
"""


import firedrake as fd
import math as m
import numpy as np
from matplotlib import animation, pyplot as plt

print('#####################################################################')
print('######################  Initial parameters  #########################')
print('#####################################################################')


case = 1
start_wavemaker = 1 # (start_wavemaker = 1 => wavemaker started to move, start_wavemaker = 2 => Wavemaker starts and then stops)
ic = 1                                                     #  ic = 1 to use ics = func, ic = 0 use ics as 0 
settings = 2                                               # settings for wavemaker, 1 == original , 2 == yangs settings
alp = 1
dt = 0.02 #0.0005 # dx/(16*np.pi)
print('Time step size =', dt)

H0 = 1                                                      # water depth
g = 9.8                                                     # gravitational acceleration
c = np.sqrt(g*H0)                                           # wave speed  

#________________________ MESH  _______________________#

nx = 200 
n = nx
ny = 1

dx= 1/nx

Lx =  100 #26 # 33 # make it equal to wavelength 
Ly = 1
print("Lx =", Lx)
print('Ly =', Ly)
print("Nodes in x direction =", nx)

mesh = fd.RectangleMesh(nx, ny, Lx, Ly)
x,y = fd.SpatialCoordinate(mesh)
Lw =  0.3 # 1/n * 20                                        # Point till which coordinates trandformation will happen
print('Lw =', Lw)

xvals = np.linspace(0, Lx - 0.1 , 100)
yvals = np.linspace(0, Lx - 0.1 , 100) 
yslice = Ly/2
xslice = Lx/2

wavemaker_id = 1                 # 1 => left side of the domain

#__________________  Define function spaces  __________________##

V = fd.FunctionSpace(mesh, "CG", 1)                         # scalar function space

trial = fd.TrialFunction(V)                                 # trail function

v = fd.TestFunction(V)

phi = fd.Function(V, name = "phi")                          # phi^n
phi_new = fd.Function(V, name = "phi_new")                  # phi^n+1

eta = fd.Function(V, name =  "eta")                         # eta^n
eta_new = fd.Function(V, name =  "eta_new")                 # eta^n+1

h = fd.Function(V, name =  "eta")                           # h^n
h_new = fd.Function(V, name =  "eta_new")                   # h^n+1

#______________________ Exact solution _______________________#

phie= fd.Function(V, name = "phi_exact") 
he = fd.Function(V, name = "h_exact") 
etae = fd.Function(V, name = "eta_exact") 

#_______________________ Wavemaker _______________________#

R = fd.Function(V, name = "wavemaker")                       # Wavemaker motion
Rt = fd.Function(V, name = "wavemaker motion")               # Wavemaker velocity

Rh = fd.Function(V, name = "wavemaker")                      # Wavemaker motion till Lw
Rht = fd.Function(V, name = "wavemaker_velocity")            # Wavemaker velocity with Heaviside

Ek = fd.Function(V , name = "kinetic energy")
Ep = fd.Function(V, name = "potential energy")
E1t = fd.Function(V, name = "Total energy for case 1")
Et = fd.Function(V)

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16 # font size of image axes
t = 0


##__________________   Parameters for wave   _____________________##
print("#####################################################################")
print('########################   PARAMETERS  of Wave  #####################')
print("#####################################################################")

t = 0 # start time
m1 = 1
m2 = 0

k1 = (2* fd.pi * m1) /Lx
print('Wavenumber in x direction (k1) =',k1)

k2 = 0 #(2* fd.pi * m2) /Ly
print('Wavenumber in y direction (k2)  =',k2)


w = c * np.sqrt(k1**2 + k2**2)
print('wave frequency (w)',w )

k = np.sqrt(k1**2 + k2**2)
print('Total wavenumber (k) =',k)

Tp = (2* fd.pi ) /w
print('Time period of wave (Tp) =',Tp)

##__________________ Parameters for wavemaker _____________________##
print("#####################################################################")
print('######################### Parameters for wavemaker ##################')
print("#####################################################################")
        
gamma = 0.002 # 0.002 #0.0001                                # coefficient to determine the amplitude of wavemaker
print('Gamma =', gamma)

if gamma >= Lw:
    print(" The wavelength of the wavemaker should be less than Lw")

lamb = 15 #13 #13 #15 #40 #60 # 0.5                                 # Wavelength
print('Wavelength of wavemaker=', lamb)

kp = 2*fd.pi/lamb                                           # Wave number
print('Wavemaker wave number (kp) =',kp)

sigma =   c * fd.sqrt(kp**2) #fd.sqrt(g*kp*fd.tanh(kp*H0)) # c * np.sqrt(kp**2) #0.62 #2*np.pi/Tw #fd.sqrt(g*kw*fd.tanh(kw*H0)) #calculated 0.62 by plotting 
print('Wavemaker frequency (sigma) =', sigma)

# dt = 0.1 * (2/sigma)

Tw = 2*fd.pi/sigma  #2*Tp #3*Tp #fd.pi/(2*ww)               # Wavemaker  period
print('Time period of wavemaker (Tw )=', Tw)

t_end =  4*Tw # time of simulation in sec
print('End time =', t_end)

ts = int(t_end/dt)
print('time_steps =', ts)

##______________ Plot to spot the region of wavemaker frequency ____________##

lam = np.linspace(1, 200,200) 
k_plot = 2*fd.pi/lam 
w_pot =  ( np.sqrt(g* k_plot * np.tanh(k_plot*H0)))
w_shallow = c * np.sqrt(k_plot**2)
Time_period = 2*fd.pi/w_pot


fig, ((ax1, ax2)) = plt.subplots(2)
ax1.set_title(" Wave frequency ($\omega$) vs. wave number (k)")
ax1.plot(k_plot, w_pot,  'k--',label = '$Potential$')
ax1.plot(kp, sigma,  'ro')
ax1.plot(k_plot, w_shallow , 'r--', label = '$Shallow$')
ax1.set_xlabel('k ')
ax1.set_ylabel('$\omega $ ')
# ax2.set_xlim([19, 1000])
ax1.legend(loc=1)

ax2.plot(k_plot,w_pot,   'k--',label = '$Potential$')
ax2.plot(k_plot, w_shallow , 'r--', label = '$Shallow $')
ax2.plot(kp, sigma,  'ro')
ax2.set_xlabel('k ')
ax2.set_ylabel('$\omega $ ')
ax2.set_xlim([0.05, 3])
ax2.legend(loc=1)

##______________  To get results at different time steps ______________##

time = []
while (t <= t_end):  
        t+= dt
        time.append(t)


x2 = int(len(time)/2)
t_plot = np.array([ time[1], time[x2], time[-1] ])
print("t_plot =", t_plot)


##___________________ Parameters for IC _________________________##
if ic == 1:
    print('#################################################################')
    print('################ Parameters of ICs and Exact ####################')
    print('#################################################################')
    if case ==1 : 
        A0 = 0.009 #0.01 # 0.0009#0.0005
        B0 =0.009# 0.01 # 0.0009 #0.0005
    else:
        A0 = 0.009 # 0.0009
        B0 = 0.009 # 0.0009
    print('A0 =', A0)
    print('B0 =', B0)
    
    Uo = gamma
    tic = 0
    aic = np.exp(-1j * sigma * tic)
    print('aic =', aic)
    
##______________________  Parameters for Exact Sol _______________________##
    
    # P = (kp * fd.sin(int(kp) * Lx))
    P = (kp * fd.sin(int(kp * Lx)))
    # P = (kp * fd.sin(kp * Lx))
    print('P = (kp * fd.sin(int(kp) * Lx)) =',P)

    U_0 = Uo * 1j * sigma  
    
    a = np.exp(-1j * sigma * t_end)
    print("Real part of exp(-1j * sigma * t_end) =",a.real)
            ############################################
            #          Initial Conditions              #
            ############################################
            
print('Initial conditions')

if ic ==1:    
    ic1 = phi.interpolate( (U_0.real)/P * aic.real* fd.cos(kp * (x - Lx)) \
                          + (g/w) * fd.cos(k1 * x) * ( -A0*fd.sin(w * tic) + B0*fd.cos(w * tic) ))    
    if case == 1:
        ic2 = eta.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) )
    else:
        ic2 =  h.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) + H0)
else:
    ic1 = phi.assign (0)    
    if case == 1:
        ic2 = eta.assign(0) 
    else:
        ic2 = h.assign(1.0)
    
phi.assign(ic1)
phi_new.assign(ic1)

if case ==1:
    eta.assign(ic2)
    eta_new.assign(ic2)
else:
    h.assign(ic2)
    h_new.assign(ic2)

etavals = np.array([ic2.at(x, yslice) for x in xvals])
phivals = np.array([ic1.at(x, yslice) for x in xvals])

fig, ((ax1, ax2)) = plt.subplots(2)
ax2.plot(xvals, phivals , label = '$\phi$')
if case ==1:
    ax1.plot(xvals, etavals, label = '$\eta$')
    ax1.set_ylabel('$\eta$ ')
else: 
    ax1.plot(xvals, etavals, label = '$h$')
    ax1.set_ylabel('$h(x,t)$ ')
    if ic == 1:
        pass
    else:
        ax1.set_ylim([0.0, 1.5])
    
ax1.set_xlabel('$x$ ',fontsize=size)
ax1.set
ax2.set_xlabel('$x$ ',fontsize=size)
ax2.set_ylabel('$\phi$ ',fontsize=size)

            ############################################
            #                  Wavemaker               #
            ############################################

print('############### Wavemaker motion calculations block #################')

nt = 0
nnt = np.linspace(0, t_end, ts+1)
print('length of nnt =', len(nnt))
     
if start_wavemaker  == 1: # wavemaker moving from t = 0 to t = t_end
    if settings ==1:
        Rt = fd.Constant( gamma * ((1j * sigma)* np.exp(-1j * sigma * t)).real )    
        Rh = Rh.interpolate(fd.conditional( fd.le(x,Lw) , -gamma*(np.exp(-1j * sigma * t)).real , 0.0 )) 
        Rht =  Rht.interpolate(fd.conditional(fd.le(x,Lw) , gamma* ((1j * sigma) * np.exp(-1j * sigma * t)).real , 0.0) )
    else: 
        
        Rt = fd.Constant( gamma* sigma * fd.sin(sigma*t))  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw), -gamma * fd.cos(sigma*t), 0.0) )
        Rht = Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma * sigma * fd.sin(sigma*t),0.0))
   
elif start_wavemaker  == 2: # wavemaker moves at first and then stops after some time
    
    if settings ==1:
        Rt = fd.Constant( gamma* ((1j * sigma)* np.exp(-1j * sigma * t)).real )  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma * (np.exp(-1j * sigma * t)).real , 0.0)) 
        Rht =  Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma * ((1j * sigma) * np.exp(-1j * sigma * t)).real , 0.0) )
    else: 
        
        Rt = fd.Constant( gamma * sigma * fd.sin(sigma*t))  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma * fd.cos(sigma*t),0.0))
        Rht = Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma * sigma * fd.sin(sigma*t),0.0))   
else: # wavemaker does not move at all
    Rt = fd.Constant(0)
    Rh = fd.Constant(0)
    Rht = fd.Constant(0)

##__________________  Plot of wavemaker motion  _____________________##
print('Plot of wavemaker motion')
Rt1=[]
Rh1 = []
lim = int(len(time)/4)          # time after which wavemaker stops

for nt in range(len(nnt)): 
    if start_wavemaker  == 1:
        if settings == 1:
            R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
            Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
        else:
            R_h1 = -gamma*fd.cos(sigma*t)
            Rt_1 = gamma*sigma*fd.sin(sigma*t)
            
    elif start_wavemaker  == 2:
        print('The wavemaker will stop after time step =',lim) 
        if nt <= lim: 
            if settings == 1:
                R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
                Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
            else:
                R_h1 = -gamma*fd.cos(sigma*t)
                Rt_1 = gamma*sigma*fd.sin(sigma*t)
        elif nt > lim:
            R_h1 = fd.Constant(0)
            Rt_1 = fd.Constant(0)

    else:
        R_h1 = fd.Constant(0)
        Rt_1 = fd.Constant(0)

    t+=dt
    Rt1.append(Rt_1)
    Rh1.append(R_h1)


if start_wavemaker == 1:
    Amp_wave = max(Rh1)
    print('Maximum amplitude of wavemaker =', Amp_wave)
    
    vel_wave = max(Rt1)
    print('Maximum velocity of wavemaker =', vel_wave)
else:
    pass   

# plt.figure(3)
fig, (ax1, ax2) = plt.subplots(2)

ax1.set_title('Wavemaker motion',fontsize=tsize)
ax1.plot(nnt, Rh1, 'r-', label = f'$h_e: t = {t:.3f}$ ')
ax1.set_xlabel('$Time [s]$ ',fontsize=size)
ax1.set_ylabel('$R(t)[m/s]$ ',fontsize=size)

ax2.set_title('Wavemaker velocity',fontsize=tsize)
ax2.plot(nnt, Rt1,  'r-', label = f'$\phi_e: t = {t:.3f}$ ')
ax2.set_xlabel('$Time [s]$ ',fontsize=size)
ax2.set_ylabel('$R_{t} [m/s^2]$ ',fontsize=size) 

##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

plt.figure(2)
fig, (ax1, ax2) = plt.subplots(2)

ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)

if case == 1:
    ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)

    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$\eta (x,t)$ ',fontsize=size)
    
    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)

else:
    ax1.set_title(r'$h $ value in $x$ direction',fontsize=tsize)
    
    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$h(x,t)$ ',fontsize=size)

    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size) 
    
#######################  VARIATIONAL PRINCIPLE  ##############################
print("#####################################################################")
print('######################### Numerical Calculations   ##################')
print("#####################################################################")

t = 0
if case == 1:
    print("You have selected case 1 : Linear (alpha = 0) /Nonlinear (alpha = 1) SWE VP solved by firedrake by using fd.derivative ")

    E1_t = []
    
    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) \
          - (1/2 * (H0 + alp*eta_new) * fd.inner(fd.grad(phi), fd.grad(phi))) \
          - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx - (H0 + alp*eta_new)*Rt*phi*fd.ds(1)
        
    eta_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1 first
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new))
    
    
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    ##__________ OUTPUT FILES ______________________##
    
    outfile_phi = fd.File("results_Lin_SWE_case1/phi.pvd")
    outfile_eta = fd.File("results_Lin_SWE_case1/eta.pvd")
    
    ###________________  TIME LOOP _________________##
 
    while (t <= t_end):
        eta_expr.solve()
        
        phi_expr.solve()
        
        t+= dt
        

        Epp = fd.assemble(( 1/2 * g * fd.inner(eta,eta) )* fd.dx)
        Ekk = fd.assemble(0.5 * H0* (fd.grad(phi)**2 * fd.dx))
        Et = abs(Ekk) + abs(Epp)
        
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            
            if ic == 1:
                phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x - Lx)) \
                                  + (g/w) * fd.cos(k1 * x ) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                    
                eta_exact = etae.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x ) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) )
                    
                phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
                etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
            else:
                pass
                
            
            eta1vals = np.array([eta_new.at(x, 0.5) for x in xvals])   
            phi1vals = np.array([phi_new.at(x, 0.5) for x in xvals])
        
            ax1.plot(xvals, eta1vals, label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals,phi1vals, label = f' $\phi_n: dt = {t:.3f}$')

            if ic == 1:
                ax1.plot(xvals, etaevals, 'k--', label = f'$h_e: t = {t:.3f}$ ')
                ax2.plot(xvals, phievals,  'k--', label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4)
            ax2.legend(loc=4)
                    

        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )

        E1_t.append(Et)
        phi.assign(phi_new)
        eta.assign(eta_new)
        
        
    plt.figure(5)
    plt.title('Total Energy evolution with time',fontsize= tsize )
    plt.xlabel( '$t$', fontsize= size)
    plt.ylabel( '$ Energy $', fontsize= size)
    plt.plot(time, E1_t )
    plt.grid()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
plt.show()     
print('*************** PROGRAM ENDS ******************')



