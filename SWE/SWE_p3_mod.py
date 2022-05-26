
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:35:43 2021

@author: mmwr
"""

import firedrake as fd
import math as m
import numpy as np
from matplotlib import animation, pyplot as plt
'''
Select the case:
    case 1 : Linear SWE VP solved by firedrake to compute the weak formulations by using fd. derivative 
            ( can be turned into nonlinear by putting the value of alp = 1 )
    case 2 : Non-Linear SWE VP solved by firedrake to compute the weak formulations by using fd. derivative
'''
print('#####################################################################')
print('######################  Initial parameters  #########################')
print('#####################################################################')


case = 2
start_wavemaker = 1 # (start_wavemaker = 1 => wavemaker started to move, start_wavemaker = 2 => Wavemaker starts and then stops)
ic = 0   #  ic = 1 to use ics = func, ic = 0 use ics as 0 
settings = 2  # settings for wavemaker, 1 == original , 2 == yangs settings
alp = 0
dt = 0.002 #0.0005 # dx/(16*np.pi)
print('time step size =', dt)


#######  mesh #########

n = nx = 80 #99#151
ny = 1
# mesh = fd.UnitSquareMesh(nx, ny)
# x,y = fd.SpatialCoordinate(mesh)

Lx = 40# 33 # make it equal to wavelength 
Ly = 1
print("Lx =", Lx)
print('Ly =', Ly)
print(" nodes in x direction =", nx)

mesh = fd.RectangleMesh(nx, ny,Lx, Ly)
x,y = fd.SpatialCoordinate(mesh)
Lw =  0.3# 1/n * 20                                              # Point till which coordinates trandformation will happen
print('Lw =', Lw)

xvals = np.linspace(0, Lx - 0.1 , 100)
yvals = np.linspace(0, Lx -0.1 , 100) 
yslice = 0.5
xslice = 0.5

wavemaker_id = 1 # 1 => left side of the domain

#########  FIGURE PARAMETERS   ###############

tsize = 18 # font size of image title
size = 16 # font size of image axes

################### Parameters for wavemaker #################################
print("###############################################################")
print('################### Parameters for wavemaker ##################')
print("###############################################################")
H0 = 1  # water depth
g = 9.8 # gravitational acceleration

c = np.sqrt(g*H0)  #wave speed
          
gamma = 0.02 # 0.002 #0.0001                                                # amplitude of wavemaker
print('Gamma=', gamma)

lamb = 15 #40 #60 # 0.5                                                   # Wavelength
print('Wavelength of wavemaker=', lamb)

kp = 2*fd.pi/lamb                                            # Wave number
# print('Wavemaker wave number =',kw)
print('Wavemaker wave number =',kp)

# ww = fd.sqrt(g*kw*fd.tanh(kp*H0)) #fd.sqrt(g*H0)*kp #np.sqrt(g*kp* np.tanh(kp*H0))        # Wave frequency
# print('Wave frequency of wavemaker (ww)=', ww)


# sigma = 2*np.pi/Tw 

sigma =  fd.sqrt(g*kp*fd.tanh(kp*H0)) # c * np.sqrt(kp**2) #0.62 #2*np.pi/Tw #fd.sqrt(g*kw*fd.tanh(kw*H0)) #calculated 0.62 by plotting 
print('Wavemaker frequency (sigma) =', sigma)

Tw = 2*fd.pi/sigma  #2*Tp #3*Tp #fd.pi/(2*ww)                                           # Wavemaker  period
print('Time period of wavemaker (Tw )=', Tw)

# kp = sigma/c
# print('kp =', kp)

if gamma >= Lw:
    print(" The wavelength of the wavemaker should be less than Lw")

t_end =  2*Tw # time of simulation in sec
print('End time =', t_end)

# dt = 0.005 # time step [s]  n/t_end
dx= 1/n

ts = int(t_end/dt)
print('time_steps =', ts)

# H0 = 1 
# g = 9.81                                             # Gravitational constant
# lamb = 0.5 #2.0                                                       # Wavelength
# k = 2*fd.pi/lamb                                                   # Wave number
# w = fd.sqrt(g*k*fd.tanh(k*H0)) # sigma was w                                    # Wave frequency
# Tw = 2*fd.pi/w                                                     # Wave period
# gamma = 0.001 # 0.02 

######### PARAMETERS   ###############
print("#############################################################")
print('################   PARAMETERS  of Wave  #####################')
print("#############################################################")
g = 9.8 # gravitational acceleration

# H = 1 # water depth
t = 0 # start time
m = 2
m1 = 1
m2 = 0

k1 = (2* fd.pi * m1) /Lx
print('Wavenumber in x direction (k1) =',k1)

k2 = 0 #(2* fd.pi * m2) /Ly
print('Wavenumber in y direction (k2)  =',k2)


w = c * np.sqrt(k1**2 + k2**2)
print('wave frequency ()',w )

k = np.sqrt(k1**2 + k2**2)
print('Total wavenumber (k) =',k)

Tp = (2* fd.pi ) /w
print('Tp =',Tp)

# t_end = 2*Tp # time of simulation in sec
# print('End time =', t_end)

# # dt = 0.005 # time step [s]  n/t_end
# dx= 1/n

# ts = int(t_end/dt)
# print('time_steps =', ts)

################### Parameters for wavemaker #################################

# gamma = 0.0001                                                # amplitude of wavemaker
# print('Gamma=', gamma)

# lamb = 0.5                                                   # Wavelength
# print('Wavelength of wavemaker=', lamb)

# kw = 2*fd.pi/lamb                                            # Wave number
# print('Wavemaker wave number =',kw)


# ww = fd.sqrt(g*H0)*kw #np.sqrt(g*kw* np.tanh(kw*H0))        # Wave frequency
# print('Wave frequency of wavemaker (ww)=', ww)

# Tw =  2*Tp #3*Tp #fd.pi/(2*ww)                                           # Wavemaker  period
# print('Time period of wavemaker (Tw )=', Tw)

# sigma = 2*np.pi/Tw
# print('Wavemaker frequency (sigma) =', sigma)

# kp = sigma/c
# print('kp =', kp)

# if gamma >= Lw:
#     print(" The wavelength of the wavemaker should be less than Lw")


####### To get results at different time steps ######

time = []
while (t <= t_end):  
        t+= dt
        time.append(t)


x2 = int(len(time)/2)
t_plot = np.array([ time[100], time[x2], time[-1] ])
print("t_plot =", t_plot)


lim = int(len(time)/4)
print(lim) 
lim1 = time[lim]


########################## Parameters for IC #################################
if ic ==1:
    print('#################################################################')
    print('##################### Parameters of ICs  ########################')
    print('#################################################################')
if case ==1 : 
    A0 = 0.01 # 0.0009#0.0005
    B0 =0.01 # 0.0009 #0.0005
else:
    A0 = 0.009# 0.0009
    B0 = 0.009# 0.0009

Uo = gamma

tic = 0
aic = np.exp(-1j * sigma * tic)
print('aic =', aic)


########################## Parameters for Exact Sol ##########################

# P = (kp * np.sin(int(kp) * Lx))
P = (kp * fd.sin(kp * Lx))
print('P= ',P)

U_0 = Uo * 1j * sigma  

a = np.exp(-1j * sigma * t_end)
print(" Real part of exp(-1j * sigma * t_end) =",a.real)

# A = U_0 * 1/P #(1/(kp * fd.sin(kp * Lx)))
# print("A =", A)

# A1 =  A * a.real #* fd.cos(k1 * (x - Lx))
# print("A1 =", A1)
# A2 = (g/w) * ( -A0*fd.sin(w*t_end) + B0*fd.cos(w*t_end) )
# print("A2 =", A2)

# B1 = ((1j*sigma)/ g) * Uo/(k1 * fd.sin(k1 * Lx))
# print("B1 =", B1)
# print("B1 real=", B1.real)

# B2 = B1 * a # * fd.cos(k1 * (x - Lx))
# print("B2 =", B2)
# print("B2_real =", B2.real)

# B3 =  ( A0*fd.cos(w*t_end) + B0*fd.sin(w*t_end) )
# print("B3 =", B3)
# B4 = B2 + B3
# print("B4 =", B4)


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

Ek = fd.Function(V , name = "kinetic energy")
Ep = fd.Function(V, name = "potential energy")
E1t = fd.Function(V, name = "Total energy for case 1")
Et = fd.Function(V)

# x_coord = fd.Function(V).interpolate(x[0])

            ############################################
            #                  Wavemaker               #
            ############################################
print('############### Wavemaker calculations block #######################')

# lim = int(len(time)/4)
# print(lim) 

nt = 0
nnt = np.linspace(0, t_end, ts+1)
print('length of nnt =', len(nnt))
 
    
if start_wavemaker  == 1:
    if settings ==1:
        Rt = fd.Constant( gamma* ((1j * sigma)* np.exp(-1j * sigma * t)).real )    
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma*(np.exp(-1j * sigma * t)).real , 0.0)) 
        Rht =  Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma* ((1j * sigma) * np.exp(-1j * sigma * t)).real , 0.0) )
    else: 
        # lamb = 0.5 #2.0                                                       # Wavelength
        # k = 2*fd.pi/lamb  
        # Tw = 2*fd.pi/w   
        # gamma = 0.02 # 0.02                                                 # Wave amplitude
        # # t_stop = 2 * Tp #7.0  
        # w =  fd.sqrt(g*k*fd.tanh(k*H0))
        
        Rt = fd.Constant( gamma*w*fd.sin(sigma*t))  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw), -gamma*fd.cos(sigma*t), 0.0) )
        Rht = Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma*w*fd.sin(sigma*t),0.0))

    
elif start_wavemaker  == 2:
    
    if settings ==1:
        Rt = fd.Constant( gamma* ((1j * sigma)* np.exp(-1j * sigma * t)).real )  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma*(np.exp(-1j * sigma * t)).real , 0.0)) 
        Rht =  Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma* ((1j * sigma) * np.exp(-1j * sigma * t)).real , 0.0) )
    else: 
        # lamb = 0.5 #2.0                                                       # Wavelength
        # k = 2*fd.pi/lamb  
        # Tw = 2*fd.pi/w   
        # gamma = 0.02 # 0.02                                                 # Wave amplitude
        # t_stop = 2 * Tp #7.0  
        # w =  fd.sqrt(g*k*fd.tanh(k*H0))
        
        Rt = fd.Constant( gamma*sigma*fd.sin(sigma*t))  
        Rh = Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma*fd.cos(sigma*t),0.0))
        Rht = Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma*w*fd.sin(sigma*t),0.0))
    
    
elif start_wavemaker  == 3:
    Rt = fd.Constant(0)
    Rh = fd.Constant(0)
    Rht = fd.Constant(0)    
    
else:
    Rt = fd.Constant(0)
    Rh = fd.Constant(0)
    Rht = fd.Constant(0)
    

############ Plot of wavemaker motion ###########
print('Plot of wavemaker motion')
Rt1=[]
Rh1 = []
nt = 0


for nt in range(len(nnt)): 
    if start_wavemaker  == 1:
        if settings == 1:
            R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
            Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
            R_h = -gamma*(np.exp(-1j * sigma *t)).real  #fd.cos(ww*t)
        else:
            # lamb = 0.5 #2.0                                                       # Wavelength
            # k = 2*fd.pi/lamb  
            # Tw = 2*fd.pi/w   
            # gamma = 0.02 # 0.02 
            R_h = -gamma*fd.cos(sigma*t)
            R_h1 = -gamma*fd.cos(sigma*t)
            Rt_1 = gamma*w*fd.sin(sigma*t)
            
    elif start_wavemaker  == 2:
        # print(nt)
        if nt <= lim: 
            if settings == 1:
                R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 
                Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 
                R_h = -gamma*(np.exp(-1j * sigma *t)).real  #fd.cos(ww*t)
            else:
                # lamb = 0.5 #2.0                                                       # Wavelength
                # k = 2*fd.pi/lamb  
                # Tw = 2*fd.pi/w   
                # gamma = 0.02 # 0.02 
                R_h = -gamma*fd.cos(sigma*t)
                R_h1 = -gamma*fd.cos(sigma*t)
                Rt_1 = gamma*w*fd.sin(sigma*t)
        elif nt > lim:
            Rt_1 = fd.Constant(0)
            Rh_1 = fd.Constant(0)
            R_h1 = fd.Constant(0)
            Rht =  fd.Constant(0)

    else:
        Rt_1 = fd.Constant(0)
        Rh_1 = fd.Constant(0)
        R_h1 = fd.Constant(0)
        Rht = fd.Constant(0)
    t+=dt
    Rt1.append(Rt_1)
    Rh1.append(R_h1)
    

# for nt in range(len(nnt)):
#     if start_wavemaker  == 1:
#         R_h1 = -gamma *(np.exp(-1j * sigma *t)).real 

#         Rt_1 = gamma * ((1j * sigma) * np.exp(-1j * sigma *t)).real 

#         R_h = -gamma*(np.exp(-1j * sigma *t)).real  #fd.cos(ww*t)

#     else:
#         Rt_1 = fd.Constant(0)
#         Rh_1 = fd.Constant(0)
#         R_h1 = fd.Constant(0)
#         Rht = fd.Constant(0)
#     t+=dt
#     Rt1.append(Rt_1)
#     Rh1.append(R_h1)
    
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
    
    ic1 = phi.interpolate( ((U_0.real)/P * aic.real* fd.cos(kp * (x - Lx)) \
                           + (g/w) * fd.cos(k1 * x) * ( -A0*fd.sin(w * tic) + B0*fd.cos(w * tic) )) * 0)
        
    if case == 1:
        ic2 = eta.interpolate(0 * ( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                                   +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) )) )
    else:
        # ic2 =  h.interpolate( 0 *  ((((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
        #                           +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) )) + H)
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

plt.figure(1)    

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
    
ax1.set_xlabel('$x$ ')
ax2.set_xlabel('$x$ ')
ax2.set_ylabel('$\phi$ ')

######## FIGURE SETTINGS ###########
print('Figure settings')

plt.figure(2)
fig, (ax1, ax2) = plt.subplots(2)

ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)



if case == 1:
    ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
    # ax3.set_title(r'$\eta$ value in $y$ direction',fontsize=tsize)
    
    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$\eta (x,t)$ ',fontsize=size)
    
    
    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
    


else:
    ax1.set_title(r'$h $ value in $x$ direction',fontsize=tsize)
    # ax3.set_title(r'$h $ value in $y$ direction',fontsize=tsize)
    
    ax1.set_xlabel(r'$x$ ',fontsize=size)
    ax1.set_ylabel(r'$h(x,t)$ ',fontsize=size)
    
    
    ax2.set_xlabel(r'$x$ ',fontsize=size)
    ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
  


#######################  VARIATIONAL PRINCIPLE  ##############################
t = 0
if case == 1:
    print("You have selected case 1 : Linear (alpha = 0) /Nonlinear (alpha = 1) SWE VP solved by firedrake by using fd.derivative ")
    
    # E1_p = []
    # E1_k = []
    E1_t = [] #np.zeros(len(time))#[]
    # def Ek_expr(phi):
    #     #return Function(V).interpolate(H0-0.5*(1+sign(x[0]-xb))*sb*(x[0]-xb))
    #     return 0.5 * fd.inner(fd.grad(phi_new), fd.grad(phi_new)) 

    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) \
          - (1/2 * (H0 + alp*eta_new) * fd.inner(fd.grad(phi), fd.grad(phi))) \
          - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx - (H0 + alp*eta_new)*Rt*phi*fd.ds(1)
        
          
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
        

        Epp = fd.assemble(( 1/2 * g * fd.inner(eta,eta) )* fd.dx)
        Ekk = fd.assemble(0.5 * H0* (fd.grad(phi)**2 * fd.dx))
        # Epp = fd.assemble(( 1/2 * g * fd.inner(eta,eta) )* dx)
        Et = Ekk + Epp
        
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x - Lx)) \
                              + (g/w) * fd.cos(k1 * x ) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
            eta_exact = etae.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x - Lx))\
                              +  fd.cos(k1 * x ) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) )
                
            phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
            etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
        
            eta1vals = np.array([eta_new.at(x, 0.5) for x in xvals])   
            phi1vals = np.array([phi_new.at(x, 0.5) for x in xvals])
        
            
            
            ax1.plot(xvals, eta1vals, label = f' $\eta_n: t = {t:.3f}$')
            ax1.plot(xvals, etaevals, 'k--', label = f'$\eta_e: t = {t:.3f}$ ')
            ax1.legend(loc=2)
            
            ax2.plot(xvals,phi1vals, label = f' $\phi_n: dt = {t:.3f}$')
            ax2.plot(xvals, phievals,  'k--', label = f'$\phi_e: t = {t:.3f}$ ') 
            ax2.legend(loc=1)
        
            
            

        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        # Ek.assign(Ek1)
        # Ep.assign(Ep1)
        
        
        E1_t.append(Et)
        phi.assign(phi_new)
        eta.assign(eta_new)
    print('length of Et= ', len(E1_t))
    # print('Et =', E1_t)
    print('length of time= ', len(time)) 
    # print('Epvals =', Epvals)
    # print('Ekvals =', Ekvals)
    
    plt.figure(5)
    plt.title('Total Energy evolution with time',fontsize= tsize )
    plt.xlabel( '$t$', fontsize= size)
    plt.ylabel( '$ Energy $', fontsize= size)
    plt.plot(time, E1_t )
    plt.grid()
        
    
elif case == 2:
    print('##############################################################################################################')
    print("You have selected case 2 : Non_Linear SWE VP with piston wavemaker solved by firedrake by using fd.derivative ")
    print(" Time discrete VP of Nonlinear SWE with Piston wavemaker  ")
    print('##############################################################################################################')
    E2_t = [] 
    Rh_c2 = []
    Rht_c2 = []
    DBC = fd.DirichletBC( V, Rht , wavemaker_id)
    x = fd.SpatialCoordinate(mesh)
    y = 0
    x_coord = fd.Function(V).interpolate(x[0])
    
    print('value of lim1 is', lim1)
    
    ##################################### VP #################################
    
    VP = ( fd.inner( Rht * ((x_coord - Lw)/(Lw - Rh) ) * h_new.dx(0), phi)  +  fd.inner ((h_new - h)/dt , phi )\
              - fd.inner(phi_new , (h_new/dt)) - 1/2 * (Lw/(Lw - Rh))**2 * h_new * fd.inner(fd.grad(phi), fd.grad(phi))\
              - (1/2 * g * fd.inner(h_new,h_new))  + g*H0*h_new ) * (Lw - Rht)/Lw *fd.dx \
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
        
        if start_wavemaker  == 2:       
            if t > lim1:
                Rt.assign(0) #= fd.Constant(0)
                Rh.assign(0) #= fd.Constant(0)
                Rht.assign(0) #= fd.Constant(0)
        elif start_wavemaker  == 3:
            Rt.assign(gamma* ((1j * sigma)* np.exp(-1j * sigma * t)).real) #= fd.Constant(0)
            Rh.assign(Rh.interpolate(fd.conditional(fd.le(x,Lw),-gamma*(np.exp(-1j * sigma * t)).real , 0.0)) )#= fd.Constant(0)
            Rht.assign(Rht.interpolate(fd.conditional(fd.le(x,Lw),gamma* ((1j * sigma) * np.exp(-1j * sigma * t)).real , 0.0) )) #= fd.Constant(0)
            

        h_expr.solve()
        
        phi_expr.solve()
        
        t+= dt
        Epp = fd.assemble(( 1/2 * g * fd.inner(h,h) )* ((Lw - Rh)/Lw) * fd.dx)
        Ekk = fd.assemble(0.5 * fd.inner(h , (fd.grad(phi)**2) )* (Lw/(Lw - Rh))* fd.dx )

        Et = abs(Ekk) + abs(Epp)
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x_coord - Lx)) \
                              + (g/w) * fd.cos(k1 * x_coord) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                
            h_exact = he.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x_coord - Lx))\
                              +  fd.cos(k1 * x_coord) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) + H0 )
                
            phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
            h1vals = np.array([h_new.at(x, yslice) for x in xvals])
            
            phi1vals = np.array([phi_new.at(x, yslice) for x in xvals])
            etaevals = np.array([h_exact.at(x, yslice) for x in xvals])
            
            
            ax1.plot(xvals, h1vals, label = f' $h_n: t = {t:.3f}$')
            if ic == 1:
                ax1.plot(xvals, etaevals, 'k--', label = f'$h_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4) #2
            
            ax2.plot(xvals,phi1vals, label = f' $\phi_n: dt = {t:.3f}$')
            if ic ==1:
                ax2.plot(xvals, phievals,  'k--', label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax2.legend(loc=4) #1

        outfile_eta.write( h_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        E2_t.append(abs(Et))
        
        # Rh_c2.append(Rh)
        # Rht_c2.append(Rht)
        
        phi.assign(phi_new)
        h.assign(h_new)
        
    plt.figure(5)
    plt.title('Total Energy evolution with time',fontsize= tsize )
    plt.xlabel( '$time$', fontsize= size)
    plt.ylabel( '$ Energy $', fontsize= size)
    plt.plot(time, E2_t )
    plt.grid()
    
    # print('Rht =', Rht_c2)
    
    # print('length of Rht =', len(Rht_c2))
    # print(' length of time vector', len(time))
    
    # plt.figure(6)
    # plt.title('Evolution of wavemaker motion with time',fontsize = tsize)
    # plt.xlabel( '$time$', fontsize= size)
    # plt.ylabel( '$ R_{t} $', fontsize= size)
    # plt.plot(time, Rht_c2 )
    # plt.grid() 
    
    # plt.figure(7)
    # plt.title('Wavemaker displacement with time',fontsize = tsize)
    # plt.xlabel( '$time$', fontsize= size)
    # plt.ylabel( '$ R(t) $', fontsize= size)
    # plt.plot(time, Rh_c2 )
    # plt.grid() 
        

    
elif case == 3:
      
    print('You have selected case 3: Solves weak formulations; first calculates h^n+1 and then calculates phi^(n+1)')
    
    x = fd.SpatialCoordinate(mesh)
    y = 0
    x_coord = fd.Function(V).interpolate(x[0])
 
    piston = 1 #( Choose piston = 1 to solve for Nonlinear SWE with piston wavemaker  )
    
    if piston == 1:
        ### Nonlinear SWE with piston wavemaker  ####
        print(" Nonlinear SWE with piston wavemaker ")
        h2_full =  ( fd.inner((h_new - h)/dt,v) + fd.inner(Rht * ((x_coord - Lw)/(Lw - Rh)) * h.dx(0), v) \
                +  ((Lw/(Lw - Rh)))**2 * fd.inner((h_new * phi.dx(0)).dx(0),v)  )* fd.dx + ((Lw/(Lw - Rh))*h_new*phi.dx(0) *v \
                - Rt*h_new*v)*fd.ds(1) # - Rht*h*v *fd.ds(1)
                ### + v * ((Lw/(Lw - Rh)))**2 * fd.inner(fd.grad(h,fd.grad(phi))  )* fd.dx
    
        phi2_full = ( fd.inner((phi_new - phi)/dt, v) + v * ((x_coord - Lw)/(Lw - Rh))*Rht * phi.dx(0) \
                  + 0.5 * ((Lw)/(Lw - Rh))**2 *(fd.inner(fd.grad(phi), fd.grad(phi))) * v + g*(h_new - H0)*v ) * fd.dx #- Rht*h*fd.grad(phi)*fd.ds(1)
    else:             
        ### Nonlinear SWE without piston wavemaker  ####
        print(" Nonlinear SWE without piston wavemaker ")
        h2_full =   (fd.inner((h_new - h)/dt, v) + fd.inner((h * phi.dx(0)).dx(0),v))* fd.dx          
    
        phi2_full = (fd.inner((phi_new - phi)/dt, v) + 1/2 * fd.inner(phi.dx(0), phi.dx(0))*v + g*(h - H0)*v)*fd.dx
        
    
        ###### Solve #####
    
    h2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(h2_full, h_new))
    phi2_full = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi2_full, phi_new) )
    
    
    ###### OUTPUT FILES ##########
    outfile_phi = fd.File("results_NonLinSWE_case3/phi.pvd")
    outfile_eta = fd.File("results_NonLinSWE_case3/h.pvd")
    
    while t<= t_end:
        h2_full.solve()
        t+= dt
        phi2_full.solve()
        outfile_eta.write( h_new )
        outfile_phi.write( phi_new )
        
        # set-up next time-step
        phi.assign(phi_new)
        h.assign(h_new)
        
    eta2vals = np.array([h_new.at(x, 0.5) for x in xvals])
    phi2vals = np.array([phi_new.at(x, 0.5) for x in xvals])
    # print('phi_case2 =', phi2vals)
    # print('h_case2 =', eta2vals)
    
    ax1.plot(xvals, eta2vals, label = 'Case3 : $h$')
    ax1.legend(loc=2)
    ax2.plot(xvals,phi2vals, label = 'Case3 : $\phi$')
    ax2.legend(loc=1)
else:
    print(" The selected number does not match any case")
    
    
# ax1.plot(xvals, etaevals, '--',label = '$Exact: \eta_x$ ')
# ax1.legend(loc=2)
# ax2.plot(xvals, phievals, '--',label = '$Exact: \phi_x$ ')
# ax2.legend(loc=1)
# ax3.plot(yvals, etaevalsy, '--',label = '$Exact: \eta_y$ ')
# ax4.plot(yvals, phievalsy, '--',label = '$Exact: \phi_y$ ')
    
plt.show()     
    
    
    

    
print('*************** PROGRAM ENDS ******************')