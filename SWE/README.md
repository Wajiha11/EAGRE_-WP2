# Linear Shallow water equations
Use shallow2_2D.py for the modelling of linear shallow water equations without piston wave-maker in two dimensional domain.
It is done to compare the two approaches to solve the system of equations. One of these aprroaches is novel; it uses a time discrete VP top derive the weak formulations by using fd.derivative function of Firedrake. The second of these approaches uses the conventinal method, in which the user manually type the weak formulations. The results prove that both approaches are equal as the L_infity norm obtained from the comparison of both approaches shows that the difference is within the machine precesion.

# Numerical wave tank with piston wave-maker by using linear/non-linear Shallow water equations
Use SWE_p3.py for the modelling of numerical wave tank with piston wave-maker by using linear/non-linear Shallow water equations in one dimensional domain.
