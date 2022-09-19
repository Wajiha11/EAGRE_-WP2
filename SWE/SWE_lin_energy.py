import firedrake as fd
import math as m
import numpy as np
from matplotlib import animation, pyplot as plt
import os
# import logging
# from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
print('#####################################################################')
print('######################  Initial parameters  #########################')
print('#####################################################################')

'''
Case 1 => Solves Linear shallow water case by using time-discrete VP based on SE scheme.
Case 2 => Solves Linear shallow water case by using time-discrete VP based on SV scheme.

'''

case = 2
start_wavemaker = 2 # (start_wavemaker = 0 => Does not move at all  , start_wavemaker = 1 => wavemaker keeps moving, start_wavemaker = 2 => Wavemaker starts and then stops)
ic = 0                                                     #  initial conditions (ic) = 1 to use ic = func, ic = 0 use ic as 0 
settings = 1                                               # settings for wavemaker motion, 1 == original , 2 == yangs settings
alp = 0
dt = 0.02 
print('Time step size =', dt)
save_path =  "data_Lin_SWE" 
if not os.path.exists(save_path):
    os.makedirs(save_path)

H0 = 1                                                      # water depth
g = 9.8                                                     # gravitational acceleration
c = np.sqrt(g*H0)                                           # wave speed  
#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16 # font size of image axes
factor = 2
t = 0
tt = format(t, '.3f') 
#________________________ MESH  _______________________#

nx = 200 #30
n = nx
ny = 1

dx= 1/nx

Lx =  140
Ly = 40
print("Lx =", Lx)
print('Ly =', Ly)
print("Nodes in x direction =", nx)

mesh = fd.RectangleMesh(nx, ny, Lx, Ly)
x,y = fd.SpatialCoordinate(mesh)
Lw =  5                                       # Point till which coordinates trandformation will happen
print('Lw =', Lw)

xvals = np.linspace(0, Lx-0.001  , nx)
yvals = np.linspace(0, Ly- 0.001  , ny) 
yslice = Ly/2
xslice = Lx/2

wavemaker_id = 1                 # 1 => left side of the domain

#__________________  Define function spaces  __________________##

V = fd.FunctionSpace(mesh, "CG", 1)                         # scalar function space

trial = fd.TrialFunction(V)                                 # trail function

v = fd.TestFunction(V)

phi = fd.Function(V, name = "phi")                          # phi^n
phi_new = fd.Function(V, name = "phi_new")                  # phi^n+1

phi_half = fd.Function(V, name = "phi")                     # phi^n

eta_half = fd.Function(V, name = "phi")                     # phi^n

eta = fd.Function(V, name =  "eta")                         # eta^n
eta_new = fd.Function(V, name =  "eta_new")                 # eta^n+1


#______________________ Exact solution _______________________#

phie= fd.Function(V, name = "phi_exact") 
he = fd.Function(V, name = "h_exact") 
etae = fd.Function(V, name = "eta_exact") 

#_______________________ Wavemaker _______________________#

R = fd.Function(V, name = "wavemaker")                       # Wavemaker motion
Rt = fd.Function(V, name = "wavemaker motion")               # Wavemaker velocity
Rt_half = fd.Function(V, name = "wavemaker motion")          # Wavemaker velocity

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
gamma = 0.002

lamb = 70                                                   # Wavelength
print('Wavelength of wavemaker=', lamb)

kp = 2*fd.pi/lamb                                           # Wave number
print('Wavemaker wave number (kp) =',kp)

sigma =   c * fd.sqrt(kp**2) #fd.sqrt(g*kp*fd.tanh(kp*H0))  # Wavemaker frequency 
print('Wavemaker frequency (sigma) =', sigma)

Tw = 2*fd.pi/sigma  #2*Tp #3*Tp #fd.pi/(2*ww)               # Wavemaker  period
print('Time period of wavemaker (Tw )=', Tw)

t_end =  2*Tw                                               # time of simulation in sec
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
i = 0
if ic ==1:
    color= np.array(['g-', 'b-', 'k-'])
    colore= np.array(['m--', 'c--', 'r--'])
else:    
    color= np.array(['g-', 'b--', 'r:'])
    colore= np.array(['k:', 'c--', 'm:'])

t_stop = t_end/2

##___________________ Parameters for IC _________________________##
if ic == 1:
    print('#################################################################')
    print('################ Parameters of ICs and Exact ####################')
    print('#################################################################')

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
            ############################################
            #          Initial Conditions              #
            ############################################
            
print('Initial conditions')

if ic ==1:    
    ic1 = phi.interpolate( (U_0.real)/P * aic.real* fd.cos(kp * (x - Lx)) \
                          + (g/w) * fd.cos(k1 * x) * ( -A0*fd.sin(w * tic) + B0*fd.cos(w * tic) ))    
    ic2 = eta.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * tic)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x) * ( A0*fd.cos(w * tic) + B0*fd.sin(w * tic) ) )
    
else:
    ic1 = phi.assign (0)    
    ic2 = eta.assign(0) 
    
phi.assign(ic1)
phi_new.assign(ic1)
phi_half.assign(ic1)

eta.assign(ic2)
eta_new.assign(ic2)
eta_half.assign(ic2)

phivals = np.array([ic1.at(x, yslice) for x in xvals])
etavals = np.array([ic2.at(x, yslice) for x in xvals])
print(etavals)

fig, ((ax1, ax2)) = plt.subplots(2)
ax1.set_title('Initial Conditions',fontsize=tsize)
ax1.plot(xvals, etavals, label = '$\eta$')
ax1.set_ylabel('$\eta (x,t)$ [m] ',fontsize=size)
ax1.grid()

ax2.plot(xvals, phivals , label = '$\phi$')
ax2.set_xlabel('$x$ [m] ',fontsize=size)
ax2.set_ylabel('$\phi (x,t)$ ',fontsize=size)
ax2.grid()


            ############################################
            #                  Wavemaker               #
            ############################################

print('############### Wavemaker motion calculations block #################')

nt = 0
nnt = np.linspace(0, t_end, ts+1)

##__________________  Plot of wavemaker motion  _____________________##
print('Plot of wavemaker motion')
Rt1=[]
Rh1 = []

if start_wavemaker == 2:
        print('The wavemaker will stop after time step =',t_stop) 
        
t = 0    
for nt in range(len(nnt)): 
    if start_wavemaker  == 1:
            R_h1 = -gamma*fd.cos(sigma*t)
            Rt_1 = gamma*sigma*fd.sin(sigma*t)
            
    elif start_wavemaker == 2:

            R_h1 = -gamma *fd.cos(sigma*t)
            Rt_1 = gamma *sigma*fd.sin(sigma*t)  
            
            if t >= t_stop:
                    R_h1 = -gamma*fd.cos(sigma*t_stop)
                    Rt_1 = 0*gamma*sigma*fd.sin(sigma*t_stop)
                    
    elif start_wavemaker == 0:
        
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

plt.figure(2)
fig, (ax1, ax2) = plt.subplots(2)

ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)

ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
ax1.set_ylabel(r'$\eta (x,t) \times 10^{-2} [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()

#######################  VARIATIONAL PRINCIPLE  ##############################
print("#####################################################################")
print('######################### Numerical Calculations   ##################')
print("#####################################################################")

t = 0
if case == 1:
    print("You have selected case 1 : Linear (alpha = 0) /Nonlinear (alpha = 1) SWE VP solved by firedrake by using fd.derivative ")

    E1_t = []
    E1_p = []
    E1_k = []

    VP = ( fd.inner ((eta_new - eta)/dt , phi) - fd.inner(phi_new , (eta_new/dt)) \
          - (1/2 * (H0 + alp*eta_new) * fd.inner(fd.grad(phi), fd.grad(phi))) \
          - (1/2 * g * fd.inner(eta_new,eta_new)) ) * fd.dx - (H0 + alp*eta_new)* Rt *phi*fd.ds(1)
   
        
    eta_expr = fd.derivative(VP, phi, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1 first
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new))
    
    
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    ##__________ OUTPUT FILES ______________________##
    if start_wavemaker ==1:
        outfile_phi = fd.File("results_LinSWE_SE_wm1_case1/phi.pvd")
        outfile_eta = fd.File("results_LinSWE_SE_wm1_case1/eta.pvd")
    elif start_wavemaker == 2:
            outfile_phi = fd.File("results_LinSWE_SE_wm2_case1/phi.pvd")
            outfile_eta = fd.File("results_LinSWE_SE_wm2_case1/eta.pvd")    
    elif start_wavemaker == 0:
            outfile_phi = fd.File("results_LinSWE_SE_wm0_case1/phi.pvd")
            outfile_eta = fd.File("results_LinSWE_SE_wm0_case1/eta.pvd") 
    
    ###________________  TXT FILES _________________###
    if start_wavemaker == 1:
        filename1 = "Linear_SWE_SE_wm1.txt"
    elif start_wavemaker == 2:
        filename1 = "Linear_SWE_SE_wm2.txt"
    elif start_wavemaker == 0:
        filename1 = "Linear_SWE_SE_wm0.txt"

    f = open(filename1 , 'w+')

    
    ## ________________  TIME LOOP _________________ ##
    
    while (t <= t_end):
        tt = format(t, '.3f')     
    ## ______________________  wavemaker motion  _________________________ ##
    # # wavemaker moving from t = 0 to t = t_end
        if start_wavemaker == 1:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))

            # # wavemaker moves at first and then stops after some time
        if start_wavemaker == 2:
                        R.assign(-gamma * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))

                        
                        if t >= t_stop:
                            R.assign(-gamma *fd.cos(sigma*t_stop))
                            Rt.assign(0) 
                            
             # # wavemaker does not move at all
        elif start_wavemaker == 0:
                        Rt.assign(0) 
                        R.assign(0) 
    ## ___________________________________________________________________ ##

        eta_expr.solve()
        phi_expr.solve()
        
        t+= dt
        # print('velocity of wavemaker',Rt.dat.data) #Rt.dat.data

        Epp = fd.assemble(( 1/2 * g * fd.inner(eta,eta) )* fd.dx)
        Ekk = fd.assemble(0.5 * H0* (fd.grad(phi)**2 * fd.dx))
        Et = abs(Ekk) + abs(Epp)
        
        f.write('%-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s\n' \
                % (str(t), str(R.dat.data[2]), str(Rt.dat.data[2]), str(phi.at(0,0)), str(eta.at(0,0)), str(Epp), str(Ekk), str(Et) ) )
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            i += 1
            
            
            if ic == 1:
                phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x - Lx)) \
                                  + (g/w) * fd.cos(k1 * x ) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                    
                eta_exact = etae.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x ) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) )
                    
                phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
                etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
            else:
                pass

            eta1vals = np.array([eta_new.at(x, Ly/2) for x in xvals])   
            phi1vals = np.array([phi_new.at(x, Ly/2) for x in xvals])
            
            
            if start_wavemaker == 1:
                eta_file_name = 'eta_lswe_SE_wm1_'+tt+'.txt'
                phi1_file_name = 'phi_lswe_SE_wm1_'+tt+'.txt'
            elif start_wavemaker == 2:
                 eta_file_name = 'eta_lswe_SE_wm2_'+tt+'.txt'
                 phi1_file_name = 'phi_lswe_SE_wm2_'+tt+'.txt'
            elif start_wavemaker == 0:
                 eta_file_name = 'eta_lswe_SE_wm0_'+tt+'.txt'
                 phi1_file_name = 'phi_lswe_SE_wm0_'+tt+'.txt'
                 
            eta_file = open(os.path.join(save_path, eta_file_name), 'w')
            phi1_file = open(os.path.join(save_path, phi1_file_name), 'w')
            
            y_slice = Ly/2
            x_coarse = np.linspace(0,Lx-0.001,200)
            for ix in x_coarse:
                eta_file.write('%-25s  %-25s %-25s\n' %(str(ix), str(H0 + eta.at(ix,y_slice)), str(eta.at(ix,y_slice))))
                phi1_file.write('%-25s %-25s\n' %(str(ix), str(phi.at(ix,y_slice))))
                

            ax1.plot(xvals, eta1vals * (10 ** factor), color[i-1],label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals,phi1vals, color[i-1], label = f' $\phi_n: t = {t:.3f}$')
                
 
            if ic == 1:
                ax1.plot(xvals, etaevals* (10 ** factor), colore[i-1], label = f'$h_e: t = {t:.3f}$ ')
                ax2.plot(xvals, phievals,  colore[i-1], label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4)
            ax2.legend(loc=4)

        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )

        E1_t.append(Et)
        E1_k.append(Ekk)
        E1_p.append(Epp)
        
        phi.assign(phi_new)
        eta.assign(eta_new)

    f.close() 
    eta_file.close()
    phi1_file.close()
     
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Energy evolution with time',fontsize= tsize)
    ax1.plot(time, E1_k)
    ax1.set_ylabel('Kinetic energy[J] ',fontsize=size)
    ax1.grid()
    
    ax2.plot(time, E1_p)
    ax2.set_ylabel('Potential Energy [J]',fontsize=size) 
    ax2.grid()
    
    ax3.plot(time, E1_t)
    ax3.set_xlabel('$Time [s]$ ',fontsize=size)
    ax3.set_ylabel('Total energy [J] ',fontsize=size) 
    ax3.grid()
    
elif case ==2:
    print('Case 2: The linear SWE will be solved with time discrete VO based on SV scheme.')
    
    E1_t = []
    E1_p = []
    E1_k = []

          
    # VP =( -fd.inner (2 * (phi_half - phi)/dt     , eta ) \
    #       - fd.inner(2 * (phi_new - phi_half)/dt , eta_new ) \
    #      - (1/2 * g * (fd.inner(eta_new,eta_new) + fd.inner(eta,eta)  ) ) \
    #       - (1 * (H0) * fd.grad(phi_half)**2 ) ) * fd.dx\
    #       - (2* H0 * Rt_half * phi_half ) *fd.ds(1)
          
    VP =( -fd.inner (2 * (phi_half - phi)/dt     , eta ) \
          - fd.inner(2 * (phi_new - phi_half)/dt , eta_new ) \
          - (1/2 * (H0) * fd.inner(fd.grad(phi_half), fd.grad(phi_half)) \
            + 1/2 * (H0) * fd.inner(fd.grad(phi_half), fd.grad(phi_half))) \
          - (1/2 * g * (fd.inner(eta_new,eta_new)  + fd.inner(eta,eta) )  ) )* fd.dx\
          - (H0 * Rt_half *phi_half + H0 * Rt_half *phi_half ) *fd.ds(1)
    
    
    phi_half_expr = fd.derivative(VP, eta, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    phi_half_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_half_expr, phi_half))
    
    eta_expr = fd.derivative(VP, phi_half, v)  # derivative of VP wrt phi^n to get the expression for eta^n+1 first
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new))
    
    
    phi_expr = fd.derivative(VP, eta_new, v)  # derivative of VP wrt eta^n+1 to get the value of phi^n+1
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new))
    
    
    ##__________ OUTPUT FILES ______________________##
    if start_wavemaker ==1:
        outfile_phi = fd.File("results_LinSWE_SV_wm1_case1/phi.pvd")
        outfile_eta = fd.File("results_LinSWE_SV_wm1_case1/eta.pvd")
    elif start_wavemaker == 2:
            outfile_phi = fd.File("results_LinSWE_SV_wm2_case1/phi.pvd")
            outfile_eta = fd.File("results_LinSWE_SV_wm2_case1/eta.pvd")    
    elif start_wavemaker == 0:
            outfile_phi = fd.File("results_LinSWE_SV_wm0_case1/phi.pvd")
            outfile_eta = fd.File("results_LinSWE_SV_wm0_case1/eta.pvd") 
    
    ###________________  TXT FILES _________________###
    if start_wavemaker == 1:
        filename1 = "Linear_SWE_SV_wm1.txt"
    elif start_wavemaker == 2:
        filename1 = "Linear_SWE_SV_wm2.txt"
    elif start_wavemaker == 0:
        filename1 = "Linear_SWE_SV_wm0.txt"
        
    exact_sol = "exact_sol.txt"

    f = open(filename1 , 'w+')
    e = open(exact_sol, 'w+')

    
    ## ________________  TIME LOOP _________________ ##
    
    while (t <= t_end):
        tt = format(t, '.3f')
        t_half = t + dt/2
    ## ______________________  wavemaker motion  _________________________ ##
    # # wavemaker moving from t = 0 to t = t_end
        if start_wavemaker == 1:
                        R.assign(-gamma  * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))

            # # wavemaker moves at first and then stops after some time
        if start_wavemaker == 2:
                        R.assign(-gamma * fd.cos(sigma*t))
                        Rt.assign( gamma * sigma * fd.sin(sigma*t))
                        Rt_half.assign( gamma * sigma * fd.sin(sigma*t_half))
            
                        if t >= t_stop:
                            R.assign(-gamma *fd.cos(sigma*t_stop))
                            Rt_half.assign( gamma * sigma * fd.sin(sigma * t_stop))
                            Rt.assign(0) 
                            
             # # wavemaker does not move at all
        elif start_wavemaker == 0:
                        Rt.assign(0) 
                        R.assign(0) 
                        Rt_half.assign(0)
    ## ___________________________________________________________________ ##
        phi_half_expr.solve()
        eta_expr.solve()
        phi_expr.solve()

        t+= dt
        # print('velocity of wavemaker',Rt.dat.data) #Rt.dat.data

        Epp = fd.assemble(( 1/2 * g * fd.inner(eta,eta) )* fd.dx)
        Ekk = fd.assemble(0.5 * H0* (fd.grad(phi)**2 * fd.dx))
        Et = abs(Ekk) + abs(Epp)
        
        f.write('%-25s %-25s %-25s %-25s %-25s %-25s %-25s %-25s\n' \
                % (str(t), str(R.dat.data[2]), str(Rt.dat.data[2]), str(phi.at(0,0)), str(eta.at(0,0)), str(Epp), str(Ekk), str(Et) ) )
        
        if (t in t_plot):
            print('Plotting starts')
            print('t =', t)
            i += 1
            
            if ic == 1:
                phi_exact = phie.interpolate( (U_0.real)/P * a.real* fd.cos(kp * (x - Lx)) \
                                  + (g/w) * fd.cos(k1 * x ) * ( -A0*fd.sin(w * t) + B0*fd.cos(w * t) ) )
                    
                eta_exact = etae.interpolate( (((1j*sigma)/ g) * (U_0.real)/(P) * np.exp(-1j * sigma * t_end)).real * fd.cos(kp * (x - Lx))\
                                  +  fd.cos(k1 * x ) * ( A0*fd.cos(w * t) + B0*fd.sin(w * t) ) )
                    
                phievals = np.array([phi_exact.at(x, yslice) for x in xvals])
                etaevals = np.array([eta_exact.at(x, yslice) for x in xvals])
            else:
                pass

            eta1vals = np.array([eta_new.at(x, Ly/2) for x in xvals])   
            phi1vals = np.array([phi_new.at(x, Ly/2) for x in xvals])
            
            
            if start_wavemaker == 1:
                eta_file_name = 'eta_lswe_SV_wm1_'+tt+'.txt'
                phi1_file_name = 'phi_lswe_SV_wm1_'+tt+'.txt'
            elif start_wavemaker == 2:
                 eta_file_name = 'eta_lswe_SV_wm2_'+tt+'.txt'
                 phi1_file_name = 'phi_lswe_SV_wm2_'+tt+'.txt'
            elif start_wavemaker == 0:
                 eta_file_name = 'eta_lswe_SV_wm0_'+tt+'.txt'
                 phi1_file_name = 'phi_lswe_SV_wm0_'+tt+'.txt'
            if ic == 1:
                etae_file_name = 'etae_lswe_SV_wm1_'+tt+'.txt'
                phi1e_file_name = 'phie_lswe_SV_wm1_'+tt+'.txt'

                 
            eta_file = open(os.path.join(save_path, eta_file_name), 'w')
            phi1_file = open(os.path.join(save_path, phi1_file_name), 'w')
            
            y_slice = Ly/2
            x_coarse = np.linspace(0,Lx-0.001,200)
            for ix in x_coarse:
                eta_file.write('%-25s  %-25s %-25s\n' %(str(ix), str(H0 + eta.at(ix,y_slice)), str(eta.at(ix,y_slice))))
                phi1_file.write('%-25s %-25s\n' %(str(ix), str(phi.at(ix,y_slice))))
                

            ax1.plot(xvals, eta1vals * (10 ** factor), color[i-1],label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals,phi1vals, color[i-1], label = f' $\phi_n: t = {t:.3f}$')
            
            if ic == 1:
                ax1.plot(xvals, etaevals* (10 ** factor), colore[i-1], label = f'$h_e: t = {t:.3f}$ ')
                ax2.plot(xvals, phievals,  colore[i-1], label = f'$\phi_e: t = {t:.3f}$ ')
            else:
                pass
            ax1.legend(loc=4)
            ax2.legend(loc=4)

        outfile_eta.write( eta_new )
        outfile_phi.write( phi_new )

        E1_t.append(Et)
        E1_k.append(Ekk)
        E1_p.append(Epp)
        
        phi.assign(phi_new)
        eta.assign(eta_new)


    f.close() 
    eta_file.close()
    phi1_file.close()
     
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Energy evolution with time',fontsize= tsize)
    ax1.plot(time, E1_k)
    ax1.set_ylabel('Kinetic energy[J] ',fontsize=size)
    ax1.grid()
    
    ax2.plot(time, E1_p)
    ax2.set_ylabel('Potential Energy [J]',fontsize=size) 
    ax2.grid()
    
    ax3.plot(time, E1_t)
    ax3.set_xlabel('$Time [s]$ ',fontsize=size)
    ax3.set_ylabel('Total energy [J] ',fontsize=size) 
    ax3.grid()
    
    
else:
    print(" The selected number does not match any case")       
        
plt.show()     
print('*************** PROGRAM ENDS ******************')
