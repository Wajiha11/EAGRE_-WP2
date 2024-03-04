

import firedrake as fd
from firedrake import (
    min_value
)
import math
from math import *
import time as tijd
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule


print('#####################################################################')
print('######################  Initial parameters  #########################')
print('#####################################################################')

# parameters in SI units
g = 9.81  # gravitational acceleration [m/s^2]

# water
Lx = 140 # length of the tank [m] in x-direction
Lz = 1
nx = 120 # no. of nodes in x-direction
nz = 6   # no. of nodes in z-direction
nCG = 2  # function space order horizontal
nCGvert = 6 # function space order vertical
H0 = Lz     # rest water depth [m]

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step
                                      

top_id = 'top'

nvpcase = 1 # case 1 (SE linear), 2 (SE nonlinear), 21 (SE nonlinear piston wavemaker), 22 (SV nonlinear piston wavemaker)


if nvpcase == 1: 
    save_path =  "lin_PF_no_wavemaker" 
elif nvpcase == 2:
    save_path =  "NL_PF_SE" 
elif nvpcase == 21:
    save_path =  "NL_PF_SE_piston" 
elif nvpcase == 22:
    save_path =  "NL_PF_SV_piston" 
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#
# Extruded mesh; see example:
# https://www.firedrakeproject.org/demos/extruded_continuity.py.html
# https://www.firedrakeproject.org/extruded-meshes.html
# CG x R for surface eta and phi
# CG x CG for interior phi or varphi
# Use at for visualisation at point; use several at's.

mesh1d = fd.IntervalMesh(nx, Lx)
mesh = fd.ExtrudedMesh(mesh1d, nz, layer_height=Lz/nz, extrusion_type='uniform')

x, z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0.0, Lx-10**(-10), nx)
zvals = np.linspace(0.0, Lz-10**(-10), nz) # 
zslice = H0-10**(-10)
xslice = 0.5*Lx

# The equations are in nondimensional units, hence we 
L = 1
T = 1
Lx /= L
Lz /= L
gg = g

## initial condition nic=0 in fluid based on analytical solution

x = mesh.coordinates

t0 = 0.0
nic = 0
if nvpcase == 21: # wavemaker
    nic = 1
elif nvpcase == 22: # wavemaker
    nic = 1
elif nvpcase == 23: # wavemaker
    nic = 1
elif nvpcase == 233 or ncpcase == 234: # wavemaker case 23 to 25 plus waveflap
    nic = 1
    nowaveflap = 1.0 # 0: pure piston case; 1: pure waveflap case
    norfullgrav = 0.0

time = []
t = 0
    
if nic == 0:
    print('########################   PAarameters of standing-wave exact sol #####################')
    n_mode = 2
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    A = 0.2*4
    D = -gg*A/(omega*np.cosh(kx*H0))
    Tperiod = 2*np.pi/omega
    print('Period: ', Tperiod)
    phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
    phi_exact_exprH0 = D * fd.cos(kx * x[0]) * fd.cosh(kx * H0) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
    eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)
    t_end = Tperiod  # time of simulation [s]
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 0.5 # run at a) 0.125 and b) 0.5*0.125
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    print('dtt=',dtt, t_end/dtt,dt,2/omega) 
    
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 4
    
elif nic == 1: 
    print("#####################################################################")
    print('########################   PARAMETERS  of Wave  #####################')
    print("#####################################################################")
    lambd = 70
    n_mode = Lx/lambd #
    print('n_mode',n_mode)
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    Tperiod = 2.0*np.pi/omega
  
    nTfac = 35
    tstop = (nTfac-7)*Tperiod
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 30*Tperiod

    
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 0.5
    dt = CFL*Tperiod/Nt    # time step 
    print('dtt=',dtt, t_end/dtt,dt,2/omega)
    D = 0.0
    phi_exact_expr = D * x[0] * x[1]
    phi_exact_exprH0 = D * x[0]
    eta_exact_expr = D * x[0]
    
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+1*dt):
        time.append(t)
        t+= dt

    nplot = nTfac
    
dtmeet = t_end/nplot # (0:nplot)*dtmeet
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
print(' S: tmeet gvd', dtmeet, tmeet)
print('tmeas', tmeas)
epsmeet = 10.0**(-10)
nt = int(len(time)/nplot)
t_plot = time[0::nt]
print('t_plot', t_plot,nt,nplot, t_end)
print('gg:',gg)
color = np.array(['g-', 'b--', 'r:', 'm:'])
colore = np.array(['k:', 'c--', 'm:'])


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2) = plt.subplots(2)
if nvpcase == 1:
    ax1.set_title(r'VP linear SE',fontsize=tsize2)
elif nvpcase == 2:
    ax1.set_title(r'VP nonlinear SE ',fontsize=tsize2)
elif nvpcase == 21:
    ax1.set_title(r'VP nonlinear SE wavemaker:',fontsize=tsize2)
elif nvpcase == 22:
    ax1.set_title(r'VP nonlinear SV wavemaker:',fontsize=tsize2)

ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________#

# interior potential varphi; can have mix degrees in horizontal and vertical dimension
V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG', vdegree=nCGvert) 

# free surface eta and surface potential phi extended uniformly in vertical: vdegree=0
V_R = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='R', vdegree=0) 


varphi = fd.Function(V_W, name="varphi")

phi_f = fd.Function(V_R, name="phi_f")  # at the free surface
phiii = fd.Function(V_R, name="phi")
phif_new = fd.Function(V_R, name="phi_f") 

eta = fd.Function(V_R, name="eta")
eta_new = fd.Function(V_R, name="eta_new")

heta = fd.Function(V_R, name="heta")
h_new = fd.Function(V_R, name="h_new")


# Variables for Stormer-Verlet waves
mixed_V = V_R * V_W
result_mixed = fd.Function(mixed_V)
phii, varphii = fd.split(result_mixed)

#__________________  Define test functions  __________________#

# Test functions
v_W = fd.TestFunction(V_W)
v_R = fd.TestFunction(V_R)

# Stormer-Verlet waves
vvp = fd.TestFunction(mixed_V)
vvp0, vvp1 = fd.split(vvp)  # These represent "blocks".



##_________________  Boundary Conditions __________________________##
                                      
BC_phi_f = fd.DirichletBC(V_R, phi_f, top_id)
BC_phif_new = fd.DirichletBC(V_R, phif_new, top_id)

BC_varphi = fd.DirichletBC(V_W, 0, top_id)
BC_varphi_mixed = fd.DirichletBC(mixed_V.sub(1), 0, top_id)  # Wave SV



##_________________ Define Variational Principle __________________________##

# SE linear potential flow without wavemaker
if nvpcase==1: 
    print('##############################################################################################################')
    print("You have selected case 1 : Linear PF VP without piston wavemaker solved by firedrake by using fd.derivative. ")
    print("Time discrete VP is based on Symplectic-Euler scheme.  ")
    print('##############################################################################################################')
    
    ##______________________________ Time-discrete Variational Principle __________________________________## 
    
    
    VP11 = ( fd.inner(phii, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds_t \
        - ( 1/2 * fd.inner(fd.grad(phii+varphii), fd.grad(phii+varphii))  ) * fd.dx
        
    ##_________________  Automatic derivation of time-discrete weak formulations __________________________##
    
    # Step-1 and 2 must be solved in tandem: f-derivative VP wrt eta to find update of phi at free surface
    # int -phi/dt + phif/dt - gg*et) delta eta ds a=0 -> (phi-phif)/dt = -gg * eta
    phif_expr1 = fd.derivative(VP11, eta, du=vvp0)  # du represents perturbation

    # Step-2: f-derivative VP wrt varphi to get interior phi given surface update phi
    # int nabla (phi+varphi) cdot nabla delta varphi dx = 0
    phi_expr1 = fd.derivative(VP11, varphii, du=vvp1)
    Fexpr = phif_expr1+phi_expr1
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = BC_varphi_mixed))

    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    eta_expr2 = fd.derivative(VP11, phii, du=v_R)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2,eta_new)) 

    
elif nvpcase==2: # Steps 1 and 2 need solving in unison
    print('##############################################################################################################')
    print("You have selected case 2: Non-Linear PF VP without piston wavemaker solved by firedrake by using fd.derivative.")
    print("Time discrete VP is based on Symplectic-Euler scheme.  ")
    print('##############################################################################################################')

    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    # 
    Lw = 0.5*Lx
    Ww = Lw # Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check; 
    
    ##______________________________ Time-discrete Variational Principle __________________________________## 
    
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) + H0*Ww*fd.inner(phi_f, eta/dt) \
            - gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) )* fd.ds_t \
            - 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
            + Ww * (H0**2/(H0+fac*eta)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx
                
                
    ##_________________  Automatic derivation of time-discrete weak formulations __________________________##                
                
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?

    #  Step-2: linear solve; 
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: linear solve; 
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))
    
    
# SE nonlinear potential flow with wavemaker   
elif nvpcase==21: # SE Steps 1 and 2 need solving in unison case with wavemaker initial condition nic=1
    # Using ideas form here for time-dependence of wavemaker: https://www.firedrakeproject.org/demos/higher_order_mass_lumping.py.html
    print('##############################################################################################################')
    print("You have selected case 21: Non-Linear PF VP with piston wavemaker solved by firedrake by using fd.derivative.")
    print("Time discrete VP is based on Symplectic-Euler scheme.  ")
    print('##############################################################################################################')
    
    param_psi1 = {'ksp_converged_reason':None,'ksp_type': 'preonly', 'pc_type': 'lu'}
    param_psi2  = {'ksp_converged_reason':None,'ksp_type': 'preonly', 'pc_type': 'lu','snes_type': 'newtonls'}
    param_psi  = {'ksp_type': 'preonly', 'pc_type': 'lu','snes_type': 'newtonls','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'}
    param_psi5 = {'ksp_converged_reason':None, 'snes_type': 'newtonls','ksp_type': 'gmres', 'pc_type': 'jacobi'}
    
    ##__________________ Parameters for wavemaker _____________________##
    t = 0
    gam = 0.05
    sigm = omega


    tstop = Tperiod
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    Rwave = fd.Constant(0.0)
    Rwave.assign(Rwavemaker(t,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign(dRwavemakerdt(t,gam,sigm,tstop))
    Lw = 0.5*Lx
    Ww = fd.Constant(0.0)
    Wwn = fd.Constant(0.0)
    Ww.assign(Lw-Rwavemaker(t,gam,sigm,tstop))      #  Lw-Ww
    Wwn.assign(Lw-Rwavemaker(t-1.0*dt,gam,sigm,tstop))      #  Lw-Wwn
    
    ##______________________________ Time-discrete Variational Principle __________________________________## 
    

    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check; 
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) + H0*Wwn*fd.inner(phi_f, eta/dt) - gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) \
             -H0*phii*(x[0]-Lw)*dRwavedt*eta.dx(0) )* fd.ds_t \
        - 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                  + Ww * (H0**2/(H0+fac*eta)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx - Lw*dRwavedt*(phii+varphii)* (H0+eta) * fd.ds_v(1) 
            
            
    ##_________________  Automatic derivation of time-discrete weak formulations __________________________##  
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) 

    #  Step-2: solve; 
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: linear solve; 
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))

# SV nonlinear potential flow with wavemaker 
elif nvpcase==22: # Steps 1 and 2 need solving in unison; Stormer-Verlet
    print('##############################################################################################################')
    print("You have selected case 22: Non-Linear PF VP with piston wavemaker solved by firedrake by using fd.derivative. ")
    print("Time discrete VP is based on SV scheme.  ")
    print('##############################################################################################################')
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    ##__________________ Parameters for wavemaker _____________________##
    Lw = 0.5*Lx
    t = 0
    sigm = omega
    gam = 0.02
    
    tstop = Tperiod
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    Rwave = fd.Constant(0.0)
    Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
    Lw = 0.5*Lx
    Lw = Lx
    Ww = fd.Constant(0.0) 
    Wwn = fd.Constant(0.0) 
    Wwp = fd.Constant(0.0) 
    Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2
    Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))            #  Lw-Wwn n
    Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1
    
    # Ww = Lw  Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check;

    ##______________________________ Time-discrete Variational Principle __________________________________## 
    
    # phii = psi^n+1/2; phi_f = psi^n; phiii= psi^n+1
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) -H0*Wwp*fd.inner(phiii,eta_new/dt) + H0*Wwn*fd.inner(phi_f, eta/dt) \
             - 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
             - 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) \
             + 0.5*H0*phii*(x[0]-Lw)*dRwavedt*(eta.dx(0)+eta_new.dx(0)) )* fd.ds_t \
             - 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(x[1]/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
             + (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(x[1]/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)) )**2 \
             + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx\
             - 0.5*Lw*dRwavedt*(phii+varphii)*(H0+eta+H0+eta_new)*fd.ds_v(1)
             
    ##__________________ Parameters for wavemaker _____________________##
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?

    #  Step-2: solve; for phi^{n+1/2} and varphi^{n+1/2}
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: solve; for h^{n+1}
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))

    #  Step-4: linear solve; for phi^{n+1}
    phif_exprnl4 = fd.derivative(VPnl, eta_new, du=v_R) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?
    phif_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_exprnl4 ,phiii))


phii, varphii = result_mixed.split()

bc_top = fd.DirichletBC(V_R, 0, top_id)

eta_exact = fd.Function(V_R)
eta_exact.interpolate(eta_exact_expr)
eta.interpolate(eta_exact)


phi_f.interpolate(phi_exact_expr)
phii.interpolate(phi_exact_exprH0)
varphii.interpolate(phi_exact_expr-phi_exact_exprH0)

if nvpcase==22:
    eta_new.interpolate(eta_exact)


###### OUTPUT FILES ##########
outfile_Jaco = fd.File("results/Jaco.pvd")
outfile_phi = fd.File("results/phi.pvd")
outfile_eta = fd.File("results/eta.pvd")
outfile_varphi = fd.File("results/varphi.pvd")

t = 0.0
i = 0.0


print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([eta.at(x, zslice) for x in xvals]) #  pphi1vals = np.array([phii.at(xvals, zslice)])
phi1vals = np.array([phii.at(x, zslice) for x in xvals])

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)

# output_data()
if nvpcase == 1: #(SE linear)
    EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
    EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
elif nvpcase == 2:  # (SE nonlinear)  
    EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
elif nvpcase == 21: #(SE nonlinear wavemaker)
    EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)

elif nvpcase == 22: # 
    EKin = fd.assemble( 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                 +(Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                 + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx )
    EPot = fd.assemble( (0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
                        + 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) ) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)


E = EKin+EPot
plt.figure(2)

E0 = E
plt.plot(t,E-E0,'.k')
plt.plot(t,EPot-E0,'.b')
plt.plot(t,EKin,'.r')
plt.ylabel(f'$E(t), K(t) (r), P(t) (b)$',fontsize=size)
plt.xlabel(f'$t$ [s]',fontsize=size)
        
if nvpcase == 1:
    plt.title(r'Functional derivative VP used steps 1+2 & 3:',fontsize=tsize) # phi_expr.solve() # ?
elif nvpcase == 2:
    plt.title(r'VP nonlinear used steps 1+2 & 3:',fontsize=tsize)
elif nvpcase == 21: # plt.title(r'VP nonlinear used steps 1+2 & 3 wavemaker:',fontsize=tsize)
    plt.title(r'VP nonlinear SE, wavemaker:',fontsize=tsize)
elif nvpcase == 22: # plt.title(r'VP nonlinear used steps 1+2 & 3 wavemaker:',fontsize=tsize)
    plt.title(r'VP nonlinear SV, wavemaker:',fontsize=tsize)


print('E0=',E-E0,EKin,EPot-E0)

print('Time Loop starts')
tic = tijd.time()
while t <= t_end + dt: #

    tt = format(t, '.3f') 

    if nvpcase == 1: # VP linear steps 1 and 2 combined # solve of phi everywhere steps 1 and 2 combined
        phi_combo.solve() # 
        phii, varphii = result_mixed.split()
        eta_expr.solve()
        
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
        
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined with wavemaker
        Rwave.assign(Rwavemaker(t+1.0*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+1.0*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))      # Lw-Ww
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))      # Lw-Wwn
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
        
    elif nvpcase == 22:
        Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))     #  Lw-Wwn n
        Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
        phif_exprnl.solve()


    if nvpcase == 1:  # VP linear steps 1 and 2 combined
        phi_f.assign(phii)
        eta.assign(eta_new)
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        phi_f.assign(phii)
        eta.assign(eta_new)
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined
        phi_f.assign(phii)
        eta.assign(eta_new)
    elif nvpcase == 22:
        phi_f.assign(phiii)
        eta.assign(eta_new)

    
    # Energy monitoring:
    if nvpcase == 1: # VP linear steps 1 and 2 combined
        EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
        EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2\
                                   + Ww * (H0**2/(H0+fac*eta)) * (facc*phii.dx(1)+varphii.dx(1))**2) * fd.dx )
        EPot = fd.assemble( gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined
        EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                   + Ww * (H0**2/(H0+fac*eta)) * (facc*phii.dx(1)+varphii.dx(1))**2) * fd.dx )
        EPot = fd.assemble( gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 22:
        EKin = fd.assemble( 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                     +(Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                     + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( (0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
                            + 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) )* fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)

        
    E = EKin+EPot
        
    plt.figure(2)

    plt.plot(t,E-E0,'.k')
    plt.plot(t,EPot-E0,'.b')
    plt.plot(t,EKin,'.r')
    plt.ylabel(f'$E(t), K(t), P(t)$',fontsize=size)
    plt.xlabel(f'$t$ [s]',fontsize=size)


    t+= dt
    if (t in t_plot): # 
        print('Plotting starts')
        plt.figure(1)
        i += 1
        tmeet = tmeet+dtmeet

        eta1vals = np.array([eta.at(x, zslice) for x in xvals])
        if nvpcase == 1: # VP linear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals]) #phi1vals = np.array([phi.at(x, zslice) for x in xvals])
        elif nvpcase == 2: # VP nonlinear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        elif nvpcase == 21: # VP nonlinear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        else: 
            phi1vals = np.array([phi_f.at(x, zslice) for x in xvals])

        if nic == 0:
            ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
            phi_exact_exprv = D * np.cos(kx * xvals) * np.cosh(kx * H0) * np.sin(omega * t) #
            eta_exact_exprv = A * np.cos(kx * xvals) * np.cos(omega * t)
         
            ax1.plot(xvals, eta_exact_exprv, '-c', linewidth=1) # 
            ax2.plot(xvals, phi_exact_exprv, '-c', linewidth=1) #
            ax1.legend(loc=4)
            ax2.legend(loc=4)
            print('t =', t, tmeet, i)
            
        elif nic == 1:
            if t>=Tstartmeas: # t >= 0*tstop:
                ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
                ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
                ax2.legend(loc=4)
                print('t =', t, tmeet, i) 

                    
        
        outfile_eta.write(eta, time=t)
        outfile_phi.write(phi_f, time=t)

     

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
