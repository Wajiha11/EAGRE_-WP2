# Linear Shallow water equations
Use shallow2_2D.py for the modelling of linear shallow water equations without piston wave-maker in two dimensional domain.
It is done to compare the two approaches to solve the system of equations. One of these aprroaches is novel; it uses a time discrete VP top derive the weak formulations by using fd.derivative function of Firedrake. The second of these approaches uses the conventinal method, in which the user manually type the weak formulations. The results prove that both approaches are equal as the L_infity norm obtained from the comparison of both approaches shows that the difference is within the machine precesion.

# Numerical wave tank with piston wave-maker by using linear/non-linear Shallow water equations
Use SWE_p3.py for the modelling of numerical wave tank with piston wave-maker by using linear/non-linear Shallow water equations in one dimensional domain.


# Energy monitoring for linear shallow water equations with piston wavemaker (SWE_lin_energy.py)
Use SWE_lin_energy.py file to monitor the enrgy of the system with linear shallow water equations. In input paramerter section, you will find following options:

- case = 1
- start_wavemaker = 1 # (start_wavemaker = 1 => wavemaker started to move, start_wavemaker = 2 => Wavemaker starts and then stops)
- ic = 1                                                     #  ic = 1 to use ics = func, ic = 0 use ics as 0 
- settings = 2                                               # settings for wavemaker, 1 == original , 2 == yangs settings
- alp = 1

The description of initial parameters is as follows:

- ic value corresponds to either you want to assign zero or non-zero initial conditions for eta and phi. If ic = 1 then non-zero initial conditions will be   assigned for eta and phi while ic = 0 will assign zero initial conditions to eta and phi.
- settings = 1 corresponds to wavemaker motion, i.e. gamma Re (exp^(- i * sigma * t)), while settings = 2 corresponds to wavemaker motion mentioned in Yang's code i.e. -gamma*cos(w*t). The purpose to settings = 2 is to compare the current code with Yang's case. 
- alp = 0 corresponds to fully linear equations while alp = 1 corresponds to nonlinear equations with linear domain.

# Energy monitoring for non-linear shallow water equations with piston wavemaker (SWE_energy.py)
The code file SWE_energy.py can solve linear as well as nonlinear shallow water equations with and without piston wavemaker. The user should coose
- case = 1 for linear case and then put alp = 0 
- case = 1 for linear case and then put alp = 1 to do a case with nonlinear equations with linear domain
- case = 2 for fully nonlinear SWE
