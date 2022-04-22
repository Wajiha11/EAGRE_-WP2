
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmwr
"""

import firedrake as fd
import math
import numpy as np
from matplotlib import animation, pyplot as plt
'''
Select the case:
    case 1 : Linear SWE VP solved by firedrake to compute the weak formulations by using fd. derivative 
            ( can be turned into nonlinear by putting the value of alp = 1 )
    case 2 : Non-Linear SWE VP solved by firedrake to compute the weak formulations by using fd. derivative
    
    Note: you can turn the wave-maker on and off for both cases by putting start_wavemaker = 1 and 0, respectively.
'''
case = 1
start_wavemaker = 0 # (start_wavemaker = 1 => wavemaker started to move)
alp = 0
dt = 0.0005 # dx/(16*np.pi)
print('time step size =', dt)


#######  mesh #########

n = 99#151
m = 2
mesh = fd.UnitSquareMesh(n, m)
x,y = fd.SpatialCoordinate(mesh)


Lx = 1
Ly = 1
Lw =  1/n * 20                                              # Point till which coordinates trandformation will happen
print('Lw =', Lw)

xvals = np.linspace(0, 0.99, 100)
yvals = np.linspace(0, 0.99, 100) 
yslice = 0.5
xslice = 0.5

wavemaker_id = 1 # 1 => left side of the domain

#########  FIGURE PARAMETERS   ###############

tsize = 18 # font size of image title
size = 16 # font size of image axes

######### PARAMETERS   ###############

g = 9.8 # gravitational acceleration

H = 1 # water depth
t = 0 # start time
m = 2
m1 = 2
m2 = 0

k1 = (2* fd.pi * m1) /Lx
print('Wavenumber in x direction (k1) =',k1)

k2 = 0 #(2* fd.pi * m2) /Ly
print('Wavenumber in y direction (k2)  =',k2)

c = np.sqrt(g*H)

w = c * np.sqrt(k1**2 + k2**2)
print('wave frequency ()',w )

k = np.sqrt(k1**2 + k2**2)
print('Total wavenumber (k) =',k)

Tp = (2* fd.pi ) /w
print('Tp =',Tp)

t_end = 2*Tp# time of simulation in sec
print('End time =', t_end)

# dt = 0.005 # time step [s]  n/t_end
dx= 1/n

ts = int(t_end/dt)
print('time_steps =', ts)

####### To get results at different time steps ######

time = []
while (t <= t_end):  
        t+= dt
        time.append(t)



t_plot = np.array([ time[100], time[200], time[-1] ])
print("t_plot =", t_plot)
################### Parameters for wavemaker #################################

H0 = 1  
          
gamma = 0.0001                                                # amplitude of wavemaker
print('Gamma=', gamma)

lamb = 0.5                                                   # Wavelength
print('Wavelength of wavemaker=', lamb)

kw = 2*fd.pi/lamb                                            # Wave number
print('Wavemaker wave number =',kw)


ww = fd.sqrt(g*H0)*kw #np.sqrt(g*kw* np.tanh(kw*H0))        # Wave frequency
print('Wave frequency of wavemaker (ww)=', ww)

Tw =  2*Tp #3*Tp #fd.pi/(2*ww)                                           # Wavemaker  period
print('Time period of wavemaker (Tw )=', Tw)

sigma = 2*np.pi/Tw
print('Wavemaker frequency (sigma) =', sigma)

kp = sigma/c
print('kp =', kp)

if gamma >= Lw:
    print(" The wavelength of the wavemaker should be less than Lw")
    

########################## Parameters for IC #################################
if case ==1 : 
    A0 = 0.01 # 0.0009#0.0005
    B0 =0.01 # 0.0009 #0.0005
else:
    A0 = 0.0009
    B0 = 0.0009

Uo = gamma

tic = 0
aic = np.exp(-1j * sigma * tic)
print('aic =', aic)
P = (kp * np.sin(int(kp) * Lx))

########################## Parameters for Exact Sol ##########################

P = (kp * np.sin(int(kp) * Lx))
print('P= ',P)

U_0 = Uo * 1j * sigma  

a = np.exp(-1j * sigma * t_end)
print(" Real part of exp(-1j * sigma * t_end) =",a.real)

A = U_0 * 1/P #(1/(kp * fd.sin(kp * Lx)))
print("A =", A)

A1 =  A * a.real #* fd.cos(k1 * (x - Lx))
print("A1 =", A1)
A2 = (g/w) * ( -A0*fd.sin(w*t_end) + B0*fd.cos(w*t_end) )
print("A2 =", A2)

B1 = ((1j*sigma)/ g) * Uo/(k1 * fd.sin(k1 * Lx))
print("B1 =", B1)
print("B1 real=", B1.real)

B2 = B1 * a # * fd.cos(k1 * (x - Lx))
print("B2 =", B2)
print("B2_real =", B2.real)

B3 =  ( A0*fd.cos(w*t_end) + B0*fd.sin(w*t_end) )
print("B3 =", B3)
B4 = B2 + B3
print("B4 =", B4)


##############################################################################

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

h = fd.Function(V, name =  "eta")                           # h^n
h_new = fd.Function(V, name =  "eta_new")                   # h^n+1

#_______________________Exact solution _______________________#
phie= fd.Function(V, name = "phi_exact") 
he = fd.Function(V, name = "h_exact") 
etae = fd.Function(V, name = "eta_exact") 

#_______________________ Wavemaker _______________________#
R = fd.Function(V, name = "wavemaker")                       # Wavemaker motion
Rt = fd.Function(V, name = "wavemaker motion")
Rh = fd.Function(V, name = "wavemaker")                      # Wavemaker motion till Lw
R_h = fd.Function(V, name = "wavemaker")                      # Wavemaker motion till Lw (used for plotting only)
Rht = fd.Function(V, name = "wavemaker_velocity")            # Wavemaker velocity with Heaviside


Rt = fd.Function(V, name = "wavemaker motion")
Rt1 = fd.Function(V)

# x_coord = fd.Function(V).interpolate(x[0])

            ############################################
            #                  Wavemaker              #
            ############################################
print('Wavemaker calculations block')

    
if start_wavemaker  == 1:
    
    Rt = fd.Constant( gamma* ((1j * sigma)* np.exp(-1j * sigma *t)).real )    
    
    ### use the below ones when x,y = fd.SpatialCoordinate(mesh)
    Rh = R.interpolate(fd.conditional(fd.le(x,Lw),-gamma*(np.exp(-1j * sigma *t)).real , 0.0)) 

    Rht =  Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma* ((1j * sigma) * np.exp(-1j * sigma *t)).real , 0.0))
else:
    Rt = fd.Constant(0)
    Rh = fd.Constant(0)
    Rht = fd.Constant(0)
    
############ Plot of wavemaker motion ###########
print('Plot of wavemaker motion')
Rt1=[]
Rh1 = []
nt = 0
nnt = np.linspace(0, t_end, ts+1)

for nt in range(len(nnt)):
    if start_wavemaker  == 1:
        R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 

        Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 

        R_h = -gamma*(np.exp(-1j * sigma *t)).real  #fd.cos(ww*t)

    else:
        Rt_1 = fd.Constant(0)
        Rh_1 = fd.Constant(0)
        R_h1 = fd.Constant(0)
        Rht = fd.Constant(0)
    t+=dt
    Rt1.append(Rt_1)
    Rh1.append(R_h1)
    
    
plt.figure()    
plt.title('Evolution of wavemaker motion over time',fontsize=tsize)    
plt.plot(nnt, Rt1, 'r-')
plt.ylabel('$R_{t}$',fontsize=size)
plt.xlabel('time',fontsize=size)
plt.grid()


if start_wavemaker == 1:
    Amp_wave = max(Rh1)
    print('Maximum amplitude of wavemaker =', Amp_wave)
    
    vel_wave = max(Rt1)
    print('Maximum velocity of wavemaker =', vel_wave)
else:
    pass 

plt.figure(2)    
plt.title('Displacement of wavemaker',fontsize=tsize)    
plt.plot(nnt, Rh1, 'r-')
plt.ylabel('$R(t)$',fontsize=size)
plt.xlabel('time',fontsize=size)
plt.grid()

            ############################################
            #          Initial Conditions              #
            ############################################
            
print('Initial conditions')

    
ic1 = phi.interpolate( (U_0.real)/P * aic.real* fd.cos(kp * (x - Lx)) \
                      + (g/w) * fd.cos(k1 * x) * ( -A0*fd.sin(w * tic) + B0*fd.cos(w * tic) ))
    
if case == 1:
    ic2 = eta.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                              +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) )
else:
    ic2 =  h.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                              +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) + H)


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

plt.figure(1)    

fig, ((ax1, ax2)) = plt.subplots(2)
ax2.plot(xvals, phivals , label = '$\phi$')
ax1.plot(xvals, etavals, label = '$\eta$')

ax1.set_xlabel(r'$x$ ')
ax1.set_ylabel(r'$\eta$ ')
ax2.set_xlabel(r'$x$ ')
ax2.set_ylabel(r'$\phi$ ')

######## FIGURE SETTINGS ###########
print('Figure settings')

plt.figure(2)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig, (ax1, ax2) = plt.subplots(2)


ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)

# ax4.set_title(r'$\phi$ value in $y$ direction',fontsize=tsize)


if case == 1:
    ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
    # ax3.set_title(r'$\eta$ value in $y$ direction',fontsize=tsize)
    
    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$\eta (x,t)$ ',fontsize=size)
    
    
    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
    
    
    # ax3.set_xlabel(r'$y$ ',fontsize=size)
    # ax3.set_ylabel(r'$\eta (x,t)$ ',fontsize=size)
    
    
    # ax4.set_xlabel(r'$y$ ',fontsize=size)
    # ax4.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)

else:
    ax1.set_title(r'$h $ value in $x$ direction',fontsize=tsize)
    # ax3.set_title(r'$h $ value in $y$ direction',fontsize=tsize)
    
    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$h(x,t)$ ',fontsize=size)
    
    
    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
    
    
    # ax3.set_xlabel(r'$y$ ',fontsize=size)
    # ax3.set_ylabel(r'$h(x,t)$ ',fontsize=size)
    
    
    # ax4.set_xlabel(r'$y$ ',fontsize=size)
    # ax4.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)    


#######################  VARIATIONAL PRINCIPLE  ##############################
t = 0
if case == 1:
    print("You have selected case 1 : Linear (alpha = 0) /Nonlinear (alpha = 1) SWE VP solved by firedrake by using fd.derivative ")

    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) \
          - (1/2 * (H + alp*eta_new) * fd.inner(fd.grad(phi), fd.grad(phi))) \
          - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx - (H + alp*eta_new)*Rt*phi*fd.ds(1)
        
          
    eta_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1 first
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new))
    
    
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_Lin_SWE_case1/phi.pvd")
    outfile_eta = fd.File("results_Lin_SWE_case1/eta.pvd")
    
    
    ######### TIME LOOP ############

    
    while (t <= t_end):
        eta_expr.solve()
        
        phi_expr.solve()
        
        t+= dt
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x - Lx)) \
                              + (g/w) * fd.cos(k1 * x ) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
            eta_exact = etae.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x - Lx))\
                              +  fd.cos(k1 * x ) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) )
                
            phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
            h1vals = np.array([eta_new.at(x, 0.5) for x in xvals])
            
            phi1vals = np.array([phi_new.at(x, 0.5) for x in xvals])
            etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
            
            
            ax1.plot(xvals, h1vals, label = f' $\eta_n: t = {t:.3f}$')
            ax1.plot(xvals, etaevals, 'k--', label = f'$\eta_e: t = {t:.3f}$ ')
            ax1.legend(loc=2)
            
            ax2.plot(xvals,phi1vals, label = f' $\phi_n: dt = {t:.3f}$')
            ax2.plot(xvals, phievals,  'k--', label = f'$\phi_e: t = {t:.3f}$ ') 
            ax2.legend(loc=1)

        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        phi.assign(phi_new)
        eta.assign(eta_new)
    
elif case == 2:
    print("You have selected case 2 : Non_Linear SWE VP with piston wavemaker solved by firedrake by using fd.derivative ")
    print(" VP of Nonlinear SWE with Piston wavemaker  ")
    DBC = fd.DirichletBC( V, Rht , wavemaker_id)
    x = fd.SpatialCoordinate(mesh)
    y = 0
    x_coord = fd.Function(V).interpolate(x[0])
    
    ##################################### VP #################################
    
    VP = ( fd.inner( Rht * ((x_coord - Lw)/(Lw - Rh) ) * h_new.dx(0), phi)  +  fd.inner ((h_new - h)/dt , phi )\
              - fd.inner(phi_new , (h_new/dt)) - 1/2 * (Lw/(Lw - Rh))**2 * h_new * fd.inner(fd.grad(phi), fd.grad(phi))\
              - (1/2 * g * fd.inner(h_new,h_new))  + g*H*h_new ) * (Lw - Rht)/Lw *fd.dx \
              - (Rt*h_new*phi)*fd.ds(1)
          
          
    ##########################################################################
    
    h_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for h^n+1
    phi_expr = fd.derivative(VP, h_new, v)  # derivative of VP wrt h^n+1 to get the value of phi^n+1
    
    
    h_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(h_expr, h_new))

    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_NonLinSWE_case2/phi.pvd")
    outfile_eta = fd.File("results_NonLinSWE_case2/eta.pvd")
    
    
    ######### TIME LOOP ############
    
    while (t <= t_end):
        h_expr.solve()
        
        phi_expr.solve()
        
        t+= dt
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x_coord - Lx)) \
                              + (g/w) * fd.cos(k1 * x_coord) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
            h_exact = he.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x_coord - Lx))\
                              +  fd.cos(k1 * x_coord) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) + H )
                
            phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
            h1vals = np.array([h_new.at(x, 0.5) for x in xvals])
            
            phi1vals = np.array([phi_new.at(x, 0.5) for x in xvals])
            etaevals = np.array([h_exact.at(x, yslice) for x in xvals])
            
            
            ax1.plot(xvals, h1vals, label = f' $h_n: t = {t:.3f}$')
            ax1.plot(xvals, etaevals, 'k--', label = f'$h_e: t = {t:.3f}$ ')
            ax1.legend(loc=2)
            
            ax2.plot(xvals,phi1vals, label = f' $\phi_n: dt = {t:.3f}$')
            ax2.plot(xvals, phievals,  'k--', label = f'$\phi_e: t = {t:.3f}$ ') 
            ax2.legend(loc=1)

        outfile_eta.write( h_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        phi.assign(phi_new)
        h.assign(h_new)

else:
    print(" The selected number does not match any case")
    
    
    
plt.show()     
    
    
    

    
print('*************** PROGRAM ENDS ******************')
