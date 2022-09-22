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
import os

'''
Case 1 => Solves NL_SWE case by using time-discrete VP based on SV scheme.
Case 2 => Solves NL_SWE case by using time-discrete  weak formulations VP based on SV scheme.

'''


print('#####################################################################')
print('######################  Initial parameters  #########################')
print('#####################################################################')


case = 2 #  Case = 1 (Non-linear SWE by SV) & case = 3 (NLSWE by SV2)
start_wavemaker = 2 # (start_wavemaker = 1 => wavemaker started to move, start_wavemaker = 2 => Wavemaker starts and then stops)
ic = 0                                                     #  ic = 1 to use ics = func, ic = 0 use ics as 0 
settings = 2                                               # settings for wavemaker, 1 == original , 2 == yangs settings
alp = 1
dt = 0.02                            # time step
print('Time step size =', dt)
save_path =  'data_SWE_SV'
if not os.path.exists(save_path):
            os.makedirs(save_path)

        
H0 = 1                                                      # water depth
g = 9.8                                                     # gravitational acceleration
c = np.sqrt(g*H0)                                           # wave speed  

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16  # font size of image axes
factor = 2
t = 0

#________________________ MESH  _______________________#

nx = 200 
n = nx
ny = 1

dx= 1/nx

Lx =  140  # make it equal to wavelength 
Ly = 40 
print("Lx =", Lx)
print('Ly =', Ly)
print("Nodes in x direction =", nx)

mesh = fd.RectangleMesh(nx, ny, Lx, Ly)
x,y = fd.SpatialCoordinate(mesh)
Lw =  1                                         # Point till which coordinates trandformation will happen
print('Lw =', Lw)

xvals = np.linspace(0, Lx-0.001  , nx)
yvals = np.linspace(0, Ly- 0.001  , ny) 
yslice = Ly/2
xslice = Lx/2

wavemaker_id = 1                                # 1 => left side of the domain

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
A_max = 0.002

lamb =  70                                                  # Wavelength
print('Wavelength of wavemaker=', lamb)

kp = 2*fd.pi/lamb                                           # Wave number
print('Wavemaker wave number (kp) =',kp)

sigma =   c * fd.sqrt(kp**2) #fd.sqrt(g*kp*fd.tanh(kp*H0))  # Wavemaker frequency 
print('Wavemaker frequency (sigma) =', sigma)

Tw = 2*fd.pi/sigma  #2*Tp #3*Tp #fd.pi/(2*ww)               # Wavemaker  period
print('Time period of wavemaker (Tw )=', Tw)

t_end =  2*Tw                                               # time of simulation in sec
print('End time =', t_end)

t_steps = int(t_end/dt)
print('time_steps =', t_steps)

t_stop = Tw

gamma = A_max
    
##______________ Plot to spot the region of wavemaker frequency ____________##

lam = np.linspace(1, 200,200) 
k_plot = 2*fd.pi/lam 
w_pot =  ( np.sqrt(g* k_plot * np.tanh(k_plot*H0)))
w_shallow = c * np.sqrt(k_plot**2)
Time_period = 2*fd.pi/w_pot

fig, ((ax1, ax2)) = plt.subplots(2)
ax1.set_title(" Wave frequency ($\omega$) vs. wave number (k)",fontsize=tsize)
ax1.plot(k_plot, w_pot,  'k--',label = '$Potential$')
ax1.plot(kp, sigma,  'ro')
ax1.plot(k_plot, w_shallow , 'r--', label = '$Shallow$')
ax1.set_ylabel('$\omega $ ',fontsize=size)
ax1.legend(loc=1)
ax1.grid()

ax2.plot(k_plot,w_pot,   'k--',label = '$Potential$')
ax2.plot(k_plot, w_shallow , 'r--', label = '$Shallow $')
ax2.plot(kp, sigma,  'ro')
ax2.set_xlabel('k ',fontsize=size)
ax2.set_ylabel('$\omega $ ',fontsize=size)
ax2.set_xlim([0.05, 3])
ax2.legend(loc=1)
ax2.grid()

##______________  To get results at different time steps ______________##

time = []
while (t <= t_end):  
        t+= dt
        time.append(t)

x2 = int(len(time)/2)
t_plot = np.array([ time[0], time[x2], time[-1] ])
print("t_plot =", t_plot)

lim1 = t_stop
i = 0
color= np.array(['g-', 'b--', 'r:'])
colore= np.array(['k:', 'c--', 'm:'])

##___________________ Parameters for IC _________________________##
if ic == 1:
    print('#################################################################')
    print('################ Parameters of ICs and Exact ####################')
    print('#################################################################')
    if case ==1 : 
        A0 = 0.009 
        B0 =0.009
    else:
        A0 = 0.009 
        B0 = 0.009 
    print('A0 =', A0)
    print('B0 =', B0)
    
    Uo = gamma
    tic = 0
    aic = np.exp(-1j * sigma * tic)
    print('aic =', aic)
    
##______________________  Parameters for Exact Sol _______________________##
    

    P = (kp * fd.sin(kp * Lx))
    print('P = (kp * fd.sin(int(kp) * Lx)) =',P)

    U_0 = Uo * 1j * sigma  
    
    a = np.exp(-1j * sigma * t_end)
    print("Real part of exp(-1j * sigma * t_end) =",a.real)
#__________________  Define function spaces  __________________##

V = fd.FunctionSpace(mesh, "CG", 1)                         # scalar function space

trial = fd.TrialFunction(V)                                 # trail function

v = fd.TestFunction(V)

phi = fd.Function(V, name = "phi")                          # phi^n
phi_half = fd.Function(V, name = "phi")                     # phi^n+1/2
phi_new = fd.Function(V, name = "phi_new")                  # phi^n+1


h = fd.Function(V, name =  "eta")                           # h^n
h_new = fd.Function(V, name =  "eta_new")                   # h^n+1

#______________________ Exact solution _______________________#

phie= fd.Function(V, name = "phi_exact") 
he = fd.Function(V, name = "h_exact") 
etae = fd.Function(V, name = "eta_exact") 

#_______________________ Wavemaker _______________________#

R        = fd.Function(V, name = "wavemaker")                      # Wavemaker motion
Rt       = fd.Function(V, name = "wavemaker motion")               # Wavemaker velocity


Rt_half       = fd.Function(V, name = "Rt_half")                   # Wavemaker velocity

Rh       = fd.Function(V, name = "wavemaker")                      # Wavemaker motion till Lw
Rh_new   = fd.Function(V, name = "wavemaker")                      # Wavemaker motion till Lw at t+1
Rh_half  = fd.Function(V, name = "Rh_half ") 

# Rht      = fd.Function(V, name = "wavemaker_velocity")             # Wavemaker velocity with Heaviside
Rht_half = fd.Function(V, name = "Rht_half") 

W        = fd.Function(V, name = "Lw - Rh")
W_half   = fd.Function(V, name = "Lw - Rh") 
W_new    = fd.Function(V, name = "Lw - Rh_new") 

X = fd.Function(V, name = "x_coord - Lw")
            ############################################
            #          Initial Conditions              #
            ############################################
            
print('Initial conditions')

if ic ==1:    
    ic1 = phi.interpolate( (U_0.real)/P * aic.real* fd.cos(kp * (x - Lx)) \
                          + (g/w) * fd.cos(k1 * x) * ( -A0*fd.sin(w * tic) + B0*fd.cos(w * tic) ))    

    ic2 =  h.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) + H0)
else:
    ic1 = phi.assign (0)    
    ic2 = h.assign(1.0)
    
phi.assign(ic1)
phi_new.assign(ic1)



h.assign(ic2)
h_new.assign(ic2)

etavals = np.array([ic2.at(x, yslice) for x in xvals])
phivals = np.array([ic1.at(x, yslice) for x in xvals])

fig, ((ax1, ax2)) = plt.subplots(2)
ax2.plot(xvals, phivals , label = '$\phi$')

ax1.plot(xvals, etavals, label = '$h$')
ax1.set_ylabel('$h(x,t)$ [m] ',fontsize=size)
if ic == 1:
    pass
else:
    ax1.set_ylim([0.0, 1.5])
    
ax1.grid()
ax2.set_xlabel('$x$ [m] ',fontsize=size)
ax2.set_ylabel('$\phi (x,t)$ ',fontsize=size)
ax2.grid()

            ############################################
            #                  Wavemaker               #
            ############################################

print('############### Wavemaker motion calculations block #################')

nt = 0
nnt = np.linspace(0, t_end, t_steps+1)

##__________________  Plot of wavemaker motion  _____________________##
print('Plot of wavemaker motion')
Rt1 = []
Rh1 = []
lim = t_stop #int(len(time)/2)          # time after which wavemaker stops

if start_wavemaker == 2:
        print('The wavemaker will stop after time  =',lim) 
        
t = 0 

for nt in range(len(nnt)): 
    if start_wavemaker  == 1:
        if settings == 1:
            R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
            Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
        else:
            R_h1 = -gamma*fd.cos(sigma*t)
            Rt_1 = gamma*sigma*fd.sin(sigma*t)
            
    elif start_wavemaker  == 2:
        if settings == 1:
                R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
                Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
                
                if t >= t_stop:
                    R_h1 = -gamma *(np.exp(-1j * sigma * t_stop)).real 
                    Rt_1 = 0 *  gamma * ((1j * sigma) * np.exp(-1j * sigma *t_stop)).real
        elif settings == 2:
                R_h1 = -gamma*fd.cos(sigma*t)
                Rt_1 = gamma*sigma*fd.sin(sigma*t)
                
                if t >= t_stop:
                    R_h1 = -gamma*fd.cos(sigma*t_stop)
                    Rt_1 = 0*gamma*sigma*fd.sin(sigma*t_stop)
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

fig, (ax1, ax2) = plt.subplots(2)

ax1.set_title('Wavemaker Position',fontsize=tsize)
ax1.plot(nnt, Rh1, 'r-', label = f'$h_e: t = {t:.3f}$ ')
ax1.set_ylabel('$R(t)[m]$ ',fontsize=size)
ax1.grid()

ax2.set_title('Wavemaker velocity',fontsize=tsize)
ax2.plot(nnt, Rt1,  'r-', label = f'$\phi_e: t = {t:.3f}$ ')
ax2.set_xlabel('$Time [s]$ ',fontsize=size)
ax2.set_ylabel('$R_{t} [m/s]$ ',fontsize=size) 
ax2.grid()
##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title('Initial Conditions',fontsize=tsize)
ax1.set_title(r'$h $ value in $x$ direction',fontsize=tsize)
ax1.set_ylabel(r'$h(x,t)\times 10^{-2} [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)\times 10^{-2} $ ',fontsize=size) 
ax2.grid()
    
#######################  VARIATIONAL PRINCIPLE  ##############################
print("#####################################################################")
print('######################### Numerical Calculations   ##################')
print("#####################################################################")

t = 0
        
if case == 1:
    print('##############################################################################################################')
    print("You have selected case 1 : Non_Linear SWE VP with piston wavemaker solved by firedrake by using fd.derivative ")
    print(" Time discrete VP is based on Symplectic-Euler scheme  ")
    print('##############################################################################################################')
    E2_t = [] 
    E2_k = []
    E2_p = [] 
    
    pot_ener = Lx * Ly * 9.8 * 0.5
    x = fd.SpatialCoordinate(mesh)
    x_coord = fd.Function(V).interpolate(x[0])
    
    
    ##################################### VP #################################
    
    
    VP =( - 2  * h     * ( (W_half*phi_half) - (W*phi) )/dt \
          - 2  * h_new * ( (W_new*phi_new) - (W_half*phi_half) )/dt \
          + X * Rht_half * ( h_new.dx(0) + h.dx(0)) * phi_half \
          - (1/2 * (Lw**2/ W_half) * fd.inner(fd.grad(phi_half), fd.grad(phi_half)) * h_new\
            + 1/2 * (Lw**2/ W_half) * fd.inner(fd.grad(phi_half), fd.grad(phi_half)) * h) \
          - (1/2 * g  * (Lw - Rh_half) * (fd.inner(h_new,h_new)  + fd.inner(h,h) )  ) \
              + g  * H0 * (Lw - Rh_half) * (h_new + h) )* fd.dx \
          - (Lw * Rt_half * phi_half* (h_new + h) ) *fd.ds(1)
          
    
    ##########################################################################
    phi_half_expr = fd.derivative(VP, h ,v)   # derivative of VP wrt h^n to get the expression for phi^n+1/2
    h_expr = fd.derivative(VP, phi_half , v)  # derivative of VP wrt phi^n+1/2 to get the expression for h^n+1
    phi_expr = fd.derivative(VP, h_new , v)  # derivative of VP wrt h^n+1 to get the value of phi^n+1
    
    
    phi_half_expr  = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem( phi_half_expr, phi_half))
    h_expr  = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem( h_expr, h_new))
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    
    ###________________  OUTPUT FILES _________________###
    if start_wavemaker ==1:
            outfile_phi = fd.File("results_SV_NLSWE_wm1_case2/phi.pvd")
            outfile_eta = fd.File("results_SV_NLSWE_wm1_case2/eta.pvd")
    elif start_wavemaker == 2:
            outfile_phi = fd.File("results_SV_NLSWE_wm2_case2/phi.pvd")
            outfile_eta = fd.File("results_SV_NLSWE_wm2_case2/eta.pvd")
    elif start_wavemaker == 0:
            outfile_phi = fd.File("results_SV_NonLinSWE_wm0_case2/phi.pvd")
            outfile_eta = fd.File("results_SV_NonLinSWE_wm0_case2/eta.pvd")
    ###________________  TXT FILES _________________###
    if start_wavemaker == 1:
        filename1 = "NLSWE_SV_wm1.txt"
        filename2 = "eta_NLSWE_SV_wm1.txt"
        filename3 = "phi_NLSWE_SV_wm1.txt"
    elif start_wavemaker == 2:
        filename1 = "NLSWE_SV_wm2.txt"
        filename2 = "eta_NLSWE_SV_wm2.txt"
        filename3 = "phi_NLSWE_SV_wm2.txt"
    elif start_wavemaker == 0:
        filename1 = "NLSWE_SV_wm0.txt"
        filename2 = "eta_NLSWE_SV_wm0.txt"
        filename3 = "phi_NLSWE_SV_wm0.txt"

    f = open(filename1 , 'w+')
    ######### TIME LOOP ############
    
    while (t <= t_end):
        tt = format(t, '.3f') 
        print('t =', t)
        
        t_new = t + dt
        t_half = t + dt/2
        
        X.interpolate( x_coord - Lw)
        W.interpolate(Lw - Rh)
        W_new.interpolate(Lw - Rh_new)
        W_half.interpolate(Lw - Rh_half)
    ## ______________________  wavemaker motion  _________________________ ##
        if start_wavemaker == 1:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))
                        
                        Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t), 0.0) )
                        Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_half), 0.0) )
                        Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_new), 0.0) )
                        
                        Rht_half.interpolate(fd.conditional(fd.le(x_coord,Lw),gamma  * sigma * fd.sin(sigma*t_half),0.0))
                        
            # # wavemaker moves at first and then stops after some time
        if start_wavemaker == 2:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))
                        
                        Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t), 0.0) )
                        Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_half), 0.0) )
                        Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_new), 0.0) )
                        
                        Rht_half.interpolate(fd.conditional(fd.le(x_coord,Lw),gamma  * sigma * fd.sin(sigma* t_half),0.0))

                        if t >= t_stop:
                            R.assign(-gamma  * fd.cos(sigma*t_stop))
                            Rt.assign(0)
                            
                            Rt_half.assign(0)
                            
                            Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_stop), 0.0) )
                            Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * (t_stop + dt/2)), 0.0) )
                            Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * (t_stop + dt)), 0.0) )
                            Rht_half.assign(0)

             # # wavemaker does not move at all
        elif start_wavemaker == 0:
                        R.assign(0)
                        Rt.assign(0)
                        
                        Rt_half.assign(0)
                        Rh.assign(0)
                        Rh_half.assign(0)
                        Rh_new.assign(0)
                        Rht_half.assign(0)
            
    ## ___________________________________________________________________ ##
                  
        phi_half_expr.solve()        
        h_expr.solve()
        phi_expr.solve()
        
        t+= dt

        Epp1 = fd.assemble( ( 1/2 * g * fd.inner(h,h))* ((Lw - Rh)/Lw) * fd.dx)
        Epp2 = fd.assemble( ( g *h * H0)* ((Lw - Rh)/Lw) * fd.dx )

        Epp = fd.assemble( (Lw - Rh)*( g*h*(0.5*h - H0) )* fd.dx  )
        Ekk = fd.assemble(0.5 * (Lw**2/(Lw - Rh)) * h * fd.inner(fd.grad(phi), fd.grad(phi)) * fd.dx )


        Et = abs(Ekk) + abs(Epp)
        
        f.write('%-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s\n' \
                % (str(t), str(R.dat.data[2]), str(Rt.dat.data[2]), str(phi.at(0,0)),\
                    str(h.at(0,0)), str(Epp), str(Ekk), str(Et) , str(Epp1), str(Epp2) ) )
                # % (str(t), str(R.dat.data[2]), str(Rt.dat.data[2]), str(phi.at(0,0)), str(h.at(0,0)), str(4410.367500000001 - Epp), str(Ekk), str(Et) ) )
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            i += 1
            if ic == 1:
                phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x_coord - Lx)) \
                              + (g/w) * fd.cos(k1 * x_coord) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
                h_exact = he.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x_coord - Lx))\
                                  +  fd.cos(k1 * x_coord) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) + H0 )
                    
                phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
                etaevals = np.array([h_exact.at(x, yslice) for x in xvals])    
                
            else:
                pass

            phi1vals = np.array([phi_new.at(x, Ly/2) for x in xvals])
            h1vals = np.array([h_new.at(x, Ly/2) for x in xvals])
            
            if start_wavemaker == 1:
                h_file_name = 'h_SV_nlswe_wm1_'+tt+'.txt'
                phi_file_name = 'phi_SV_nlswe_wm1_'+tt+'.txt'
            elif start_wavemaker == 2:
                 h_file_name = 'h_SV_nlswe_wm2_'+tt+'.txt'
                 phi_file_name = 'phi_SV_nlswe_wm2_'+tt+'.txt'
            elif start_wavemaker == 0:
                 h_file_name = 'h_SV_nlswe_wm0_'+tt+'.txt'
                 phi_file_name = 'phi_SV_nlswe_wm0_'+tt+'.txt'
                 
            h_file = open(os.path.join(save_path, h_file_name), 'w')
            phi_file = open(os.path.join(save_path, phi_file_name), 'w')
            
            y_slice = Ly/2
            x_coarse = np.linspace(0,Lx-0.001,200)
            for ix in x_coarse:
                h_file.write('%-25s  %-25s\n' %(str(ix), str(h.at(ix,y_slice))))
                phi_file.write('%-25s  %-25s\n' %(str(ix), str(phi.at(ix,y_slice))))
                
            ax1.plot(xvals, h1vals * (10 ** factor), color[i-1],label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals,phi1vals, color[i-1], label = f' $\phi_n: t = {t:.3f}$')
                
 
            if ic == 1:
                ax1.plot(xvals, etaevals* (10 ** factor), colore[i-1], label = f'$h_e: t = {t:.3f}$ ')
                ax2.plot(xvals, phievals,  colore[i-1], label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4)
            ax2.legend(loc=4)            


        outfile_eta.write( h_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        E2_t.append(abs(Et)) 
        E2_p.append(abs(Epp))
        E2_k.append(abs(Ekk))
        
        phi.assign(phi_new)
        h.assign(h_new)
        
    f.close() 
    h_file.close()
    phi_file.close()
 
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Energy evolution with time',fontsize= tsize)
    ax1.plot(time, E2_k)
    ax1.set_ylabel('Kinetic energy[J] ',fontsize=size)
    ax1.grid()
    
    ax2.plot(time, E2_p)
    ax2.set_ylabel('Potential Energy [J]',fontsize=size) 
    ax2.grid()
    
    ax3.plot(time, E2_t)
    ax3.set_xlabel('$Time [s]$ ',fontsize=size)
    ax3.set_ylabel('Total energy [J] ',fontsize=size) 
    ax3.grid()
    
    
elif case == 2:
    
    print('##############################################################################################################')
    print("You have selected case 2 : Non_Linear SWE VP with piston wavemaker solved by firedrake by using fd.derivative. ")
    print("Time discrete weak formulations based on Symplectic-Euler scheme.  ")
    print('##############################################################################################################')
    E2_t = [] 
    E2_k = []
    E2_p = [] 
    
    pot_ener = Lx * Ly * 9.8 * 0.5
    DBC = fd.DirichletBC( V, Rht_half , wavemaker_id)
    x = fd.SpatialCoordinate(mesh)
    y = 0
    x_coord = fd.Function(V).interpolate(x[0])
    
    
    ##################################### VP #################################
    
    VP =( - 2  * h     * ( (W_half*phi_half) - (W*phi) )/dt \
          - 2  * h_new * ( (W_new*phi_new) - (W_half*phi_half) )/dt \
          + X * Rht_half * ( h_new.dx(0) + h.dx(0)) * phi_half \
          - (1/2 * (Lw**2/ W_half) * fd.inner(fd.grad(phi_half), fd.grad(phi_half)) * h_new\
            + 1/2 * (Lw**2/ W_half) * fd.inner(fd.grad(phi_half), fd.grad(phi_half)) * h) \
          - (1/2 * g  * (Lw - Rh_half) * (fd.inner(h_new,h_new)  + fd.inner(h,h) )  ) \
              + g  * H0 * (Lw - Rh_half) * (h_new + h) )* fd.dx \
          - (Lw * Rt_half * phi_half* (h_new + h) ) *fd.ds(1)

    phi_half_expr = ( -2 * fd.inner( (W_half*phi_half - W*phi )/dt, v )\
                      + ( X * Rht_half * phi_half * v.dx(0) ) \
                      - 0.5* (Lw**2/W_half) * (phi_half.dx(0))**2 * v  \
                      - g * W_half * (h - H0)* v  )* fd.dx \
                      - ( Lw * Rt_half * phi_half * v )*fd.ds(1)
                      
                      
    h_expr = (+2* fd.inner( W_half* (h_new - h)/dt, v) \
              + ( X * Rht_half * fd.inner( h_new.dx(0) + h.dx(0), v)) \
              - (Lw**2/W_half) * phi_half.dx(0) * v.dx(0) * (h_new + h) )*fd.dx\
              - (Lw * Rt_half * (h_new + h) * v ) * fd.ds(1)
              
    phi_new_expr = ( - 2 * fd.inner( ( (W_new*phi_new) - (W_half*phi_half) )/dt, v ) \
                  + ( X * Rht_half * phi_half * v.dx(0) )\
                    - 0.5 * (Lw**2/(W_half)) * (phi_half.dx(0))**2 * v \
                    - g*(W_half)*(h_new - H0)* v)*fd.dx\
                    - (Lw * Rt_half * phi_half * v) * fd.ds(1)

            

    phi_half_expr  = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem( phi_half_expr, phi_half))
    h_expr  = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem( h_expr, h_new))
    phi_new_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_new_expr, phi_new))
            

    
    ###________________  OUTPUT FILES _________________###
    if start_wavemaker ==1:
            outfile_phi = fd.File("results_SV2_NLSWE_wm1/phi.pvd")
            outfile_eta = fd.File("results_SV2_NLSWE_wm1/eta.pvd")
    elif start_wavemaker == 2:
            outfile_phi = fd.File("results_SV2_NLSWE_wm2_case2/phi.pvd")
            outfile_eta = fd.File("results_SV2_NLSWE_wm2_case2/eta.pvd")
    elif start_wavemaker == 0:
            outfile_phi = fd.File("results_SV2_NLSWE_wm0/phi.pvd")
            outfile_eta = fd.File("results_SV2_NLSWE_wm0/eta.pvd")
    ###________________  TXT FILES _________________###
    if start_wavemaker == 1:
        filename1 = "NLSWE_SV2_wm1.txt"
        filename2 = "eta_NLSWE_SV2_wm1.txt"
        filename3 = "phi_NLSWE_SV2_wm1.txt"
    elif start_wavemaker == 2:
        filename1 = "NLSWE_SV2_wm2.txt"
        filename2 = "eta_NLSWE_SV2_wm2.txt"
        filename3 = "phi_NLSWE_SV2_wm2.txt"
    elif start_wavemaker == 0:
        filename1 = "NLSWE_SV2_wm0.txt"
        filename2 = "eta_NLSWE_SV2_wm0.txt"
        filename3 = "phi_NLSWE_SV2_wm0.txt"

    f = open(filename1 , 'w+')
    ######### TIME LOOP ############
    
    while (t <= t_end):
        tt = format(t, '.3f')  
        t_new = t + dt
        t_half = t + dt/2
        
        X.interpolate( x_coord - Lw)
        W.interpolate(Lw - Rh)
        W_new.interpolate(Lw - Rh_new)
        W_half.interpolate(Lw - Rh_half)
    ## ______________________  wavemaker motion  _________________________ ##

        if start_wavemaker == 1:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))
                        
                        Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t), 0.0) )
                        Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_half), 0.0) )
                        Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_new), 0.0) )
                        
                        Rht_half.interpolate(fd.conditional(fd.le(x_coord,Lw),gamma  * sigma * fd.sin(sigma*t_half),0.0))
                        
            # # wavemaker moves at first and then stops after some time
        if start_wavemaker == 2:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))
                        
                        Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t), 0.0) )
                        Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_half), 0.0) )
                        Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_new), 0.0) )
                        
                        Rht_half.interpolate(fd.conditional(fd.le(x_coord,Lw),gamma  * sigma * fd.sin(sigma* t_half),0.0))

                        if t >= t_stop:
                            R.assign(-gamma  * fd.cos(sigma*t_stop))
                            Rt.assign(0)
                            
                            Rt_half.assign(0)
                            
                            Rh.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * t_stop), 0.0) )
                            Rh_half.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * (t_stop + dt/2)), 0.0) )
                            Rh_new.interpolate(fd.conditional(fd.le(x_coord,Lw), -gamma  * fd.cos(sigma * (t_stop + dt)), 0.0) )
                            
                            Rht_half.assign(0)

             # # wavemaker does not move at all
        elif start_wavemaker == 0:
                        R.assign(0)
                        Rt.assign(0)
                        
                        Rt_half.assign(0)
                        
                        Rh.assign(0)
                        Rh_half.assign(0)
                        Rh_new.assign(0)
                        
                        Rht_half.assign(0)
            
    ## ___________________________________________________________________ ##
                
        phi_half_expr.solve()        
        h_expr.solve()
        phi_new_expr.solve()
        
        t+= dt

        
        Epp1 = fd.assemble( ( 1/2 * g * fd.inner(h,h))* ((Lw - Rh)/Lw) * fd.dx)
        Epp2 = fd.assemble( ( g *h * H0)* ((Lw - Rh)/Lw) * fd.dx )

        
        Epp = fd.assemble( (Lw - Rh)*( g*h*(0.5*h - H0) )* fd.dx  )
        Ekk = fd.assemble(0.5 * (Lw**2/(Lw - Rh)) * h * fd.inner(fd.grad(phi), fd.grad(phi)) * fd.dx )


        Et = abs(Ekk) + abs(Epp)
        
        f.write('%-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s\n' \
                % (str(t), str(R.dat.data[2]), str(Rt.dat.data[2]), str(phi.at(0,0)),\
                   str(h.at(0,0)), str(Epp), str(Ekk), str(Et) , str(Epp1), str(Epp2) ) )
               
        
        if (t in t_plot):
            i+=1
            print('Plotting starts')
            print('t =', t)
            if ic == 1:
                phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x_coord - Lx)) \
                              + (g/w) * fd.cos(k1 * x_coord) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
                h_exact = he.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x_coord - Lx))\
                                  +  fd.cos(k1 * x_coord) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) + H0 )
                    
                phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
                etaevals = np.array([h_exact.at(x, yslice) for x in xvals])    
                
            else:
                pass

            phi1vals = np.array([phi_new.at(x, Ly/2) for x in xvals])
            h1vals = np.array([h_new.at(x, Ly/2) for x in xvals])
            
            if start_wavemaker == 1:
                h_file_name = 'h_SV2_nlswe_wm1_'+tt+'.txt'
                phi_file_name = 'phi_SV2_nlswe_wm1_'+tt+'.txt'
            elif start_wavemaker == 2:
                 h_file_name = 'h_SV2_nlswe_wm2_'+tt+'.txt'
                 phi_file_name = 'phi_SV2_nlswe_wm2_'+tt+'.txt'
            elif start_wavemaker == 0:
                 h_file_name = 'h_SV2_nlswe_wm0_'+tt+'.txt'
                 phi_file_name = 'phi_SV2_nlswe_wm0_'+tt+'.txt'
                 
            h_file = open(os.path.join(save_path, h_file_name), 'w')
            phi_file = open(os.path.join(save_path, phi_file_name), 'w')
            
            y_slice = Ly/2
            x_coarse = np.linspace(0,Lx-0.001,200)
            for ix in x_coarse:
                h_file.write('%-25s  %-25s\n' %(str(ix), str(h.at(ix,y_slice))))
                phi_file.write('%-25s  %-25s\n' %(str(ix), str(phi.at(ix,y_slice))))
                
            ax1.plot(xvals, h1vals * (10 ** factor), color[i-1],label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals,phi1vals, color[i-1], label = f' $\phi_n: t = {t:.3f}$')
                
 
            if ic == 1:
                ax1.plot(xvals, etaevals* (10 ** factor), colore[i-1], label = f'$h_e: t = {t:.3f}$ ')
                ax2.plot(xvals, phievals,  colore[i-1], label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4)
            ax2.legend(loc=4) 


        outfile_eta.write( h_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        E2_t.append(abs(Et)) 
        E2_p.append(abs(Epp))
        E2_k.append(abs(Ekk))
        
        phi.assign(phi_new)
        h.assign(h_new)
        
    f.close() 
    h_file.close()
    phi_file.close()
 
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Energy evolution with time',fontsize= tsize)
    ax1.plot(time, E2_k)
    ax1.set_ylabel('Kinetic energy[J] ',fontsize=size)
    ax1.grid()
    
    ax2.plot(time, E2_p)
    ax2.set_ylabel('Potential Energy [J]',fontsize=size) 
    ax2.grid()
    
    ax3.plot(time, E2_t)
    ax3.set_xlabel('$Time [s]$ ',fontsize=size)
    ax3.set_ylabel('Total energy [J] ',fontsize=size) 
    ax3.grid()
    

    
    
else:
    print(" The selected number does not match any case")       
        
          
        
        
        
plt.show()     
print('*************** PROGRAM ENDS ******************')