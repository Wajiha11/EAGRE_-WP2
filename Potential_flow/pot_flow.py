import firedrake as fd
import math
import numpy as np
import matplotlib.pyplot as plt
import os


# parameters in SI units
t_end = 5.0  # time of simulation [s]
dt = 0.005  # time step [s]
g = 9.8  # gravitational acceleration
# water
Lx = 20.0  # length of the tank [m] in x-direction; needed for computing initial condition
Lz = 10.0  # height of the tank [m]; needed for computing initial condition

nx = 120
nz = 6

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

top_id = 4  

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16 # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#

mesh = fd.RectangleMesh(nx, nz, Lx, Lz)
x,z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0, Lx-0.001  , nx)
zvals = np.linspace(0, Lz- 0.001  , nz) 
zslice = Lz
xslice = Lx/2
# The equations are in nondimensional units, hence we transform::

L = Lz
T = L / math.sqrt(g * L)
t_end /= T
dt /= T
Lx /= L
Lz /= L

##______________  To get results at different time steps ______________##

time = []
t = 0
while (t <= t_end+dt):
        time.append(t)
        t+= dt

t2 = int(len(time)/2)
t_plot = np.array([ time[0], time[t2], time[-1] ])

print('t_plot', t_plot)

color= np.array(['g-', 'b--', 'r:'])
colore= np.array(['k:', 'c--', 'm:'])


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')



fig, (ax1, ax2) = plt.subplots(2)

ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)

ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________##

V_W = fd.FunctionSpace(mesh, "CG", 1)

phi = fd.Function(V_W, name="phi")
phi_new = fd.Function(V_W, name="phi")

phi_f = fd.Function(V_W, name="phi_f")  # at the free surface
phif_new = fd.Function(V_W, name="phi_f") 

eta = fd.Function(V_W, name="eta")
eta_new = fd.Function(V_W, name="eta")

trial_W = fd.TrialFunction(V_W)
v_W = fd.TestFunction(V_W)


##_________________  Boundary Conditions __________________________##

class MyBC(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])


def surface_BC():
    bc = fd.DirichletBC(V_W, 1, top_id)
    f = fd.Function(V_W, dtype=np.int32)
    ## f is now 0 everywhere, except on the boundary
    bc.apply(f)
    return MyBC(V_W, 0, f)


BC_exclude_beyond_surface = surface_BC()
BC_phi_f = fd.DirichletBC(V_W, phi_f, top_id)


phif_expr =  (v_W * (phif_new - phi_f)/dt  + v_W *  eta ) * fd.ds(top_id) # derivative of VP wrt eta^n+1 to get the value of phi^n+1
phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_expr, phif_new, bcs= BC_exclude_beyond_surface))

phi1_expr = fd.dot(fd.grad(phi), fd.grad(v_W)) * fd.dx
phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi1_expr, phi, bcs = BC_phi_f))

eta_expr = v_W *  (eta_new - eta)/dt * fd.ds(top_id) - fd.dot(fd.grad(phi_new), fd.grad(v_W)) * fd.dx
eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new, bcs= BC_exclude_beyond_surface ))

## initial condition in fluid based on analytical solution
## compute analytical initial phi and eta
n_mode = 1
a = 0.0 * T / L ** 2  # in nondim units
b = 5.0 * T / L ** 2  # in nondim units
lambda_x = np.pi * n_mode / Lx
omega = np.sqrt(lambda_x * np.tanh(lambda_x * Lz))
x = mesh.coordinates
phi_exact_expr = a * fd.cos(lambda_x * x[0]) * fd.cosh(lambda_x * x[1])
eta_exact_expr = -omega * b * fd.cos(lambda_x * x[0]) * fd.cosh(lambda_x * Lz)

bc_top = fd.DirichletBC(V_W, 0, top_id)
eta.assign(0.0)
phi.assign(0.0)
eta_exact = fd.Function(V_W)
eta_exact.interpolate(eta_exact_expr)
eta.assign(eta_exact, bc_top.node_set)
phi.interpolate(phi_exact_expr)
phi_f.assign(phi, bc_top.node_set)



###### OUTPUT FILES ##########
outfile_phi = fd.File("results/phi.pvd")
outfile_eta = fd.File("results/eta.pvd")


def output_data():
    output_data.counter += 1
    if output_data.counter % output_data_every_x_time_steps != 0:
        return
    mesh_static = mesh.coordinates.vector().get_local()
    mesh.coordinates.dat.data[:, 1] += eta.dat.data_ro
    
    outfile_eta.write( eta_new )
    outfile_phi.write( phi_new )
    mesh.coordinates.vector().set_local(mesh_static)
    
output_data.counter = -1  # -1 to exclude counting print of initial state


t = 0.0
i = 0.0


output_data()
print('Time Loop starts')

while t <= t_end + dt:
    # print("time = ", t * T)
    # symplectic Euler scheme
    tt = format(t, '.3f') 
    
    phif_expr.solve()
    phi_expr.solve()
    eta_expr.solve()

    if (t in t_plot):
                print('Plotting starts')
                print('t =', t)
                i += 1
                
                eta1vals = np.array([eta_new.at(x, zslice) for x in xvals])
                phi1vals = np.array([phi.at(x, zslice) for x in xvals])
                
                
                ax1.plot(xvals, eta1vals , color[int(i-1)],label = f' $\eta_n: t = {t:.3f}$')
                ax2.plot(xvals,phi1vals, color[int(i-1)], label = f' $\phi_n: t = {t:.3f}$')
                
                ax1.legend(loc=4)
                ax2.legend(loc=4)
    
    t+= dt
        
    phi_f.assign(phif_new)
    phi_new.assign(phi)
    eta.assign(eta_new)

    output_data()



plt.show() 
print('*************** PROGRAM ENDS ******************')
