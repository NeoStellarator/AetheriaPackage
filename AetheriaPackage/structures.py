import os
from functools import partial
from typing import List

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns

import AetheriaPackage.GeneralConstants as const
import AetheriaPackage.tail_plotter_tools as tpt
from AetheriaPackage.GeneralConstants import *
from AetheriaPackage.data_structs import *
from AetheriaPackage.basic_functions import Linear


def get_tank_radius(l_tank:float, V_tank:float, n:int, **kwargs)->float:
    '''
    Function that determines the radius of the tank given the length, volume and number of cylinders.
    It makes use of the Newton method implemented in scipy.optimize.root_scalar.
    
    Parameters
    ----------
    l_tank : FLOAT
        Tank length
    V_tank : FLOAT
        Tank volume
    n : INT
        Number of cylinders in the tank
    **kwargs
        Any other arguments to be passed to root-finding algorithm.
    '''

    def tank_volume_function(r_tank, l_tank, V_tank, n):
        '''
        Function f(x) whose root is the tank radius.
        x    ->  r_tank
        args ->  [l_tank, V_tank, n]

        f(x) = V_tank - n(pi r_tank^2 (l_tank-2r_tank) + 4/3 pi r_tank^3) = 0
            = V_tank/(n pi) - r_tank^2 (l_tank - 2/3 r_tank) = 0        
        '''
        return V_tank/(n*np.pi) -r_tank**2 *(l_tank - 2/3*r_tank)

    def derivative_tank_volume_function(r_tank, l_tank, V_tank, n):
        '''
        Derivative of 'tank_volume_function' with respect to r_tank.
        x    ->  r_tank
        args ->  [l_tank, V_tank, n]
        '''
        derivative = 2*r_tank*(r_tank-l_tank)

        if derivative == 0:
            # Derivative is zero when r_tank = 0 or r_tank = l_tank
            raise ValueError(f'Zero derivative in Newton Method! r_tank={r_tank:.3f} | l_tank={l_tank:.3f}')
        
        return derivative
    
    result = sp.optimize.root_scalar(tank_volume_function, fprime=derivative_tank_volume_function, 
                                     method='newton', args=(l_tank, V_tank, n), x0=l_tank/4, **kwargs) 
    if result.converged:
        return result.root
    else:
        raise ValueError(f'Cannot compute tank radius. Cause of termination: {result.flag}')



def optimize_tail_length(beta:float, V_tank:float, h0:float, b0:float, 
                         hf:float, bf:float, n:int, linear_rel:str='AR', 
                         plot:bool=False):
    '''
    Function to minimize the tail length, given the crash coefficient, tank volume, start-
    of-tail and end-of-tail dimensions, number of cylinders in the tank and the assumed second
    linear constraint (AR or width).

    The SLSQP algorithm implemented in scipy.optimize.minimize is used. The tail can be plotted 
    if the parameter plot is set to True.

    Parameters
    ----------
    beta: FLOAT [-]
        Crash diameter coefficient, between 0 and 1.
    V_tank : FLOAT [m3]
        H2 tank volume.
    h0 : FLOAT [m]
        Start-of-tail height (inner fuselage height)
    b0 : FLOAT [m]
        Start-of-tail width (inner fuselage width).
    hf : FLOAT [m]
        End-of-tail height.
    bf : FLOAT [m]
        End-of-tail width. If linear_rel='AR', this value is reset such that the tail width is tangent
        to the rest of the fuselage, and avoiding the bulging effect caused by the quadratic nature of b
        that arises when linear_rel='AR'.
    n : INT [-]
        Number of cylinders in the H2 tank.
    linear_rel : STR ['AR' or 'b', 'AR' by default]
        Assumed second linear relationship for the optimizer, implemented as a constraint. Can be either
        'AR' for a linear variation of aspect ratio with tailwise position (implying a quadratic 
        vaiation of the width with tailwise position) or 'b' for a linear variation of
        width with tailwise position.
    plot: BOOL (default FALSE)
        Specify whether to plot the tail.
    
    Inner Variables
    ---------------
    The design vector x used in the optimizer is structured as follows:
        x = [l_tank, hk, bk, hc, bc]
        0      1   2   3   4 
    l_tank: FLOAT [m]
        H2 tank length.
    hk : FLOAT [m]
        Tail height at the end of the H2 tank.
    bk : FLOAT [m]
        Tail width at the end of the H2 tank.
    hc : FLOAT [m]
        Tail height at the end of the H2 tank in the event of a crash.
    bc : FLOAT [m]
        Tail width at the end of the H2 tank in the event of a crash.
    
    
    Returns
    -------
    x = l_tail, l_tank, hk, bk, hc, bc, upsweep, r_tank
    l_tail : FLOAT [m]
        Tail length
    l_tank : see Inner Variables
    hk : see Inner Variables
    bk : see Inner Variables
    hc : see Inner Variables
    bc : see Inner Variables
    upsweep : FLOAT [rad]
        Upsweep angle
    r_tank : FLOAT [m]
        H2 cylinder radius.
    '''

    def get_tail_size(x:List, h0:float, hf:float)->float:
        '''
        Function to compute the length of the tail, assuming a linear relationship
        of height with tail-wise position. This is the objective function used in 
        the optimizer.

        Recall that the design vector is structured as follows
        x = [l_tank, hk, bk, hc, bc]
            0      1   2   3   4 
        
        The computed function is:
        l_tail = l_tank*(hf-h0)/(hk-h0)
        '''
        return x[0]*(hf-h0)/(x[1]-h0)


    def second_linear_constraint(x:List, h0:float, b0:float, hf:float, bf:float, var:str='AR')->float:
        '''
        Function for the second linear constraint. The objective function accounts for 
        a linear variation of height; this function is used as a constraint in the
        optimizer and accounts for either a linear varition of width or a linear variation
        of aspect ratio.

        Recall that the design vector is structured as follows
        x = [l_tank, hk, bk, hc, bc]
               0      1   2   3   4 
        
        '''

        l_tail = get_tail_size(x, h0, hf)
        l_tank, hk, bk, = x[0], x[1], x[2]

        if var == 'b':
            b = Linear(0, l_tail, b0, bf)
            f2 = b(l_tank)-bk
        elif var == 'AR':
            AR = Linear(0, l_tail, b0/h0, bf/hf) 
            f2 = AR(l_tank) - bk/hk
        
        return f2


    # ensure one-time continuity of width distri bution at beginning of tial
    if linear_rel == 'AR':
        bf = hf*(b0/h0)*(2-hf/h0)
    
    # Already setting the known values into the functions
    p_get_tank_radius = partial(get_tank_radius, V_tank=V_tank, n=n)
    p_get_tail_size   = partial(get_tail_size, h0=h0, hf=hf)
    p_second_linear_constraint  = partial(second_linear_constraint, h0=h0, b0=b0, hf=hf, bf=bf, var=linear_rel)

    # Defining Bounds
    bnds = ((0, None), (0.0001, h0), (0.0001, b0), (0.0001, None), (0.0001, None))

    # Defining Constraints: recall that 'inequality means that it is to be non-negative' 
    cons = (
            {'type': 'ineq', 'fun': lambda x: x[3]-2*p_get_tank_radius(x[0])},   # crash region larger than hydrogen tank (height)
            {'type': 'ineq', 'fun': lambda x: x[4]-2*n*p_get_tank_radius(x[0])}, # crash region larger than hydrogen tank (width)
            {'type': 'ineq', 'fun': lambda x: x[0]-2*p_get_tank_radius(x[0])}  , # prevent tank from achieving unfeasable volume
            {'type': 'eq'  , 'fun': lambda x: beta**2-(x[3]*x[4])/(x[1]*x[2])} , # beta relationship crashed-uncrashed region
            {'type': 'eq'  , 'fun': lambda x: x[2]/x[1] - x[4]/x[3]}           , # assumption that the aspect ratio during crash is constant
            {'type': 'eq'  , 'fun': p_second_linear_constraint},                 # second linear constraint
            )
    
    k1, k2  = 0.8, 0.6
    x0 = [V_tank, k1*h0, k1*b0, k2*h0, k2*b0]
    
    result = sp.optimize.minimize(p_get_tail_size, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':1000}, tol=0.001)
    
    if result.success:
        l_tank, hk, bk, hc, bc  = result.x
        r_tank = get_tank_radius(l_tank, V_tank, 2)
        l_tail = result.fun
        upsweep = np.arctan2((h0 - hf), l_tank) # upsweep angle

        if plot: 
            try:
                tpt.plot_complete_tail(l_tail, l_tank, h0, b0, hc, bc, hf, bf, r_tank, linear_rel)
            except IndexError:
                pass

        return l_tail, l_tank, hk, bk, hc, bc, hf, bf, upsweep, r_tank
    
    else:
        raise RuntimeError(f'Tail optimization failed: {result.message}')

def stress_strain_curve(stress_peak, strain_peak, plain_stress, e_d):
    E = stress_peak/strain_peak
    e = np.arange(0,1,0.001)
    densification = e_d
    strain_p = stress_peak / E
    crush_strength = np.zeros(len(e))

    for i in range(len(e)):
        if e[i] <= strain_p:
            crush_strength[i] = e[i]*E
        if e[i] > strain_p and e[i] <= densification:
            crush_strength[i] = plain_stress
        if e[i] > densification:
            crush_strength[i] = plain_stress + (e[i]-densification)*E

    #plt.plot(e, crush_strength/(10**6))
    #plt.show()
    return crush_strength, e


def decel_calculation(s_p, s_y, e_0, v0, e_d, height,m, PRINT=False):
    a, v, s, t  = [0], [v0], [0], [0]
    Ek = [0.5*m*v[-1]**2]
    s_tot = height
    dt = 0.000001
    g = 9.81
    A = m*20*g/(s_p)
    Strain = [0]
    Strainrate = [0]

    sigma_cr, e = stress_strain_curve(s_y, e_0, s_p, e_d)

    while v[-1] > 0:
        strain = s[-1]/s_tot
        index = np.abs(e - strain).argmin()
        Fcrush = sigma_cr[index]*A
        #Fcrush = 0.315*10**6
        ds = v[-1]*dt
        work_done = Fcrush*ds
        Ek.append(Ek[-1]- work_done)

        if Ek[-1] > 0:
            v.append(np.sqrt(2*Ek[-1]/m))
            a.append((v[-1]-v[-2])/dt)
            t.append(t[-1]+dt)
            s.append(s[-1] + ds)
            Strain.append(strain)
            Strainrate.append((Strain[-1]-Strain[-2])/dt)
        else:
            Ek.pop(-1)
            break
    if PRINT:
        plt.plot(t[1:-1], np.array(Strain[1:-1]))
        plt.xlabel("Time")
        plt.ylabel("Strain")
        plt.show()

        plt.plot(t[1:-1], np.array(Strainrate[1:-1]))
        plt.xlabel("Time")
        plt.ylabel("Strain rate")
        plt.show()

        plt.plot(t[1:-1], np.array(a[1:-1])/g)
        plt.xlabel("Time")
        plt.ylabel("Acceleration")
        plt.show()

        plt.plot(t, v)
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.show()

        plt.plot(t, s)
        plt.plot([0, t[-1]],[e_d*s_tot, e_d*s_tot])
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.show()
    max_g = min(np.array(a))

    return s[-1], A, max_g

def simple_crash_box(m, a, sigma_cr, v):
    s = v**2/(2*a)
    A = m*a/sigma_cr
    print(s, A)
    return s, A

def crash_box_height_convergerence(plateau_stress, yield_stress, e_0, e_d, v0, s0, m):
    I, s_arr, i = [], [], 0
    height = s0
    error = 1
    while error > 0.0005:
        travel_distance, A, max_g = decel_calculation(plateau_stress, yield_stress, e_0, v0, e_d, height, m, False)
        new_height = travel_distance/e_d
        error = abs((new_height-height)/height)
        height = new_height
        I.append(i)
        s_arr.append(height)
        i += 1
    #print(height, A, max_g/9.81)
    #plt.plot(I, s_arr)
    #plt.show()
    return travel_distance, A




def get_fuselage_sizing(h2tank, fuelcell, perf_par, fuselage, power, plot_tail=False):

    crash_box_height, crash_box_area = crash_box_height_convergerence(const.s_p, const.s_y, const.e_0, const.e_d, const.v0, const.s0, perf_par.MTOM)
    fuselage.height_fuselage_inner = fuselage.height_cabin + crash_box_height
    fuselage.height_fuselage_outer = fuselage.height_fuselage_inner + const.fuselage_margin

    power.h2_tank_volume = h2tank.volume(perf_par.mission_energy)

    l_tail, l_tank, hk, bk, hc, bc, hf, bf, upsweep, r_tank = optimize_tail_length(beta=fuselage.beta_crash,
                                                                           V_tank=power.h2_tank_volume,
                                                                           h0=fuselage.height_fuselage_inner,
                                                                           b0=fuselage.width_fuselage_inner,
                                                                           hf=const.hf,
                                                                           bf=const.bf,
                                                                           n=const.n_tanks,
                                                                           linear_rel=const.linear_rel,
                                                                           plot=plot_tail)

    fuselage.length_tail = l_tail
    power.h2_tank_length = l_tank
    power.h2_tank_radius    = r_tank
    fuselage.upsweep = upsweep 
    fuselage.bc = bc
    fuselage.crash_box_area =  crash_box_area
    fuselage.hc = hc
    fuselage.bf = bf
    fuselage.hf = hf
    fuselage.length_fuselage = fuselage.length_cockpit + fuselage.length_cabin + l_tail + fuelcell.depth + const.fuselage_margin 

    return fuselage

class PylonSizing():
    def __init__(self, engine, L):
        self.mass_eng = engine.mass_pertotalengine
        self.L = L
        self.Tmax =  2.5*2200*9.81/6
        self.moment = self.Tmax*L

    def I_xx(self, x): return np.pi/4 *  ((x[0] + x[1])**4 - x[0]**4)

    def get_area(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)

    def weight_func(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)*const.rho_composite*self.L


    def get_stress(self, x):
        return (self.moment*(x[1] + x[0]))/self.I_xx(x)

    # def r2_larger_than_r1(self, x):
    #     # print(f"r2>r1 = {x[1] - x[0]}")
    #     return x[1] - x[0]

    def column_buckling_constraint(self, x):
        # print(f"r1, r2 = {x[0], x[1]}")
        # print(f"column buckling = {(np.pi**2*const.E_alu*self.I_xx(x))/(self.L**2*self.get_area(x))- self.get_stress(x)}")
        return (np.pi**2*const.E_composite*self.I_xx(x))/(self.L**2*self.get_area(x)) - self.get_stress(x)

    def von_mises_constraint(self, x):
        # print(f"Von Mises = {const.sigma_yield -1/np.sqrt(2)*self.get_stress(x)} ")
        return const.sigma_yield - 1/np.sqrt(2)*self.get_stress(x)

    def eigenfreq_constraint(self, x):
        # print(f"Eigenfrequency = {1/(2*np.pi)*np.sqrt((3*const.E_alu*self.I_xx(x))/(self.L**3*self.mass_eng))}")
        print(f"Ixx = {self.I_xx(x)}")
        return 1/(2*np.pi)*np.sqrt((3*const.E_composite*self.I_xx(x))/(self.L**3*self.mass_eng)) - const.eigenfrequency_lim_pylon


    def  optimize_pylon_sizing(self, x0):

        cons = (
            {'type': 'ineq', 'fun': self.column_buckling_constraint },
                {'type': 'ineq', 'fun': self.von_mises_constraint }
                # {'type': 'ineq', 'fun': self.eigenfreq_constraint}
                )
        bnds = ((0.095, 0.1), (0.001,0.2))

        res = sp.optimize.minimize(self.weight_func, x0, method='SLSQP', bounds=bnds, constraints=cons)

        return res

#Moments of Inertia
def i_xx_solid(width,height):
    return width*height*height*height/12
def i_yy_solid(width,height):
    return width*width*width*height/12
def j_z_solid(width,height):
    return width*height*(width*width + height*height)/12

def i_xx_thinwalled(width,height,thickness):
    return 1/3 * width*height*height*thickness
def i_yy_thinwalled(width,height,thickness):
    return 1/3 * width*width*height*thickness
def j_z_thinwalled(width,height,thickness):
    return (height+width)*height*width*thickness/3


"""NORMAL STRESS"""
def bending_stress(moment_x,moment_y,i_xx,i_yy,i_xy,x,y):
    return((moment_x*i_yy-moment_y*i_xy)*y + (moment_y*i_xx - moment_x*i_xy)*x)/(i_xx*i_yy-i_xy*i_xy)
def normal_stress(force,area):
    return force/area
    

"""SHEAR STRESS"""
def torsion_circular(torque,dens,j_z):
    return torque*dens/j_z

def torsion_thinwalled_closed(torque,thickness,area):
    return torque/(2*thickness*area)

def maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, pos=True):
    n = lambda CL, V, WoS: 0.5 * const.rho_cr* V ** 2 * CL / WoS
    Vc, VD = Vs
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return min(n(CLmax, V, WoS), nmax) if pos else \
    ( max(-n(CLmax, V, WoS), nmin) if V <= Vc else interpolate(V, Vc, VD, nmin, 0))

def plotmaneuvrenv(WoS, Vc, CLmax, nmin, nmax):
    VD = 1.2*Vc
    Vs = Vc, VD
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x], color='blue', zorder=3)
    sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, False) for V in x], color='blue', label='Manoeuvre Envelope',zorder=3)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    plt.plot([VD, VD], [maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True), maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True)], color='blue',zorder=3)
    plt.plot([VD, VD],[0, nmax], color='blue',zorder=3)
    plt.grid(True)
    plt.xlim(0,VD+7)
    plt.plot([-5,VD+7],[0,0], color='black', lw=1)


    #plt.savefig('manoeuvre_env.png')
    return np.max([maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x])

def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + const.rho_cr * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

neggustload = lambda V, Vs, us, ns, CLalpha, WoS: 2 - posgustload(V, Vs, us, ns, CLalpha, WoS)


def plot_dash(V, n):
    plt.plot([0, V],[1, n], linestyle='dashed', color='black', zorder=1, alpha=0.5)


def plotgustenv(V_s, Vc, CLalpha, WoS, TEXT=False):
    n = lambda V, u: 1 + const.rho_cr * V * CLalpha * u / (2 * WoS)
    #Vb = np.sqrt(n(Vc, uc))*V_s
    Vb = Vc - 22.12
    Vb, Vc, VD = Vs = (Vb, Vc, 1.2*Vc) # Change if VD Changes
    us = const.ub, uc, ud  # Obtained from CS
    nb, nc, nd = ns = n(Vb, ub), n(Vc, uc), n(VD, ud)
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', zorder=2)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    sns.lineplot(x=x, y=[neggustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', label='Gust Load Envelope',zorder=2)
    plt.plot([VD, VD], [neggustload(VD, Vs, us, ns, CLalpha, WoS), posgustload(VD, Vs, us, ns, CLalpha, WoS)], color='black',zorder=2)
    plot_dash(Vc, nc)
    plot_dash(Vc, 2 - nc)
    plot_dash(VD, nd)
    plot_dash(VD, 2 - nd)
    plt.plot([Vb, Vb], [2-nb, nb], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    plt.plot([Vc, Vc], [2-nc, nc], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    if TEXT:
        plt.text(Vb + 1, 0.1, 'Vb', fontsize = 11, weight='bold')
        plt.text(V_s + 1, 0.1, 'Vs', fontsize=11, weight='bold')
        plt.text(Vc + 1, 0.1, 'Vc', fontsize=11, weight='bold')
        plt.text(VD + 1, 0.1, 'Vd', fontsize=11, weight='bold')
        plt.plot([V_s,V_s],[0, 0.05], color='black')
        plt.plot([Vb, Vb], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')



    return np.max([posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x])
    # plt.savefig('gust.png')

def get_gust_manoeuvr_loadings(perf_par, aero):

    nm = plotmaneuvrenv(perf_par.wing_loading_cruise, const.v_cr, aero.cL_max, const.n_min_req, const.n_max_req)
    ng = plotgustenv(const.v_stall, const.v_cr , aero.cL_alpha, perf_par.wing_loading_cruise, TEXT=False)

    perf_par.n_max, perf_par.n_ult = max(nm, ng), max(nm, ng)*1.5

    return perf_par



def calc_wing_weight(mtom:float, S:float, n_ult:float, A:float) -> float:
    """Returns the mass of the wing subsystem, using Cessna method for cantilever wings.
    See Eq. 5.2 (pg 67) of Pt 5. Component Weight Estimation (Roskam).

    :param mtom: maximum take off mass (Kg)
    :type mtom: float
    :param S: Wing area (m2)
    :type S: float
    :param n_ult: Ultimate load factor (-)
    :type n_ult: float
    :param A: Aspect ratio (-)
    :type A: float
    """
    # Convert to Imperial Units
    S_ft = S*10.7639104
    n_ult = n_ult
    A = A
    mtow_lbs = 2.20462 * mtom

    return 0.04674*(mtow_lbs**0.397)*(S_ft**0.36)*(n_ult**0.397)*(A**1.712)*0.453592

def calc_fuselage_weight(mtom:float, lf:float, nult:float, wf:float, 
                         hf:float, v_cr:float, rho_cr:float, rho_sl:float) -> float:
    """ Returns mass of fuselage subsystem, using USAF method. 
    See Eq 5.25 (pg 76) Pt 5. Component Weight Estimation (Roskam).

    :param mtom: Maximum take off mass (Kg)
    :type mtom: float
    :param lf: Fuselage length (m)
    :type lf: float
    :param nult: Ultimate load factor (-)
    :type nult: float
    :param wf: Maximum fuselage width (m)
    :type wf: float
    :param hf: Maximum fuselage height (m)
    :type hf: float
    :param v_cr: design cruise speed  (m/s)
    :type v_cr: float
    :param rho_cr: density at cruise altitude (Kg/m3)
    :type rho_cr: float
    :param rho_sl: density at sea-level altitude (Kg/m3)
    :type rho_sl: float

    # THE FOLLOWING IS FOR CESSNA METHOD, WHICH IS NOT USED
    # :param max_per:  Maximium perimeter of the fuselage
    # :type max_per: float
    # :param npax: Amount of passengers including pilot
    # :type npax: int
    """

    # Convert to Imperial Units
    mtow_lbs = 2.20462 * mtom
    lf_ft = lf*3.28084
    nult = nult # ultimate load factor
    wf_ft = wf*3.28084 # width fuselage [ft]
    hf_ft = hf*3.28084 # height fuselage [ft]
    Vc_kts = v_cr*(rho_cr/rho_sl)**0.5*1.94384449 # design cruise speed [kts] (convert TAS -> EAS)

    fweigh_USAF = 200*((mtow_lbs*nult/10**5)**0.286*(lf_ft/10)**0.857*((wf_ft+hf_ft)/10)*(Vc_kts/100)**0.338)**1.1
    return fweigh_USAF*0.453592  

    #if identifier == "J1":
    #    # THIS IS CESSNA METHOD
    #    fweight_high = 14.86*(mtow_lbs**0.144)*((lf_ft/max_per_ft)**0.778)*(lf_ft**0.383)*(npax**0.455)
    #    mass = fweight_high*0.453592
    #else:
    #    fweight_high = 14.86*(mtow_lbs**0.144)*((lf_ft/max_per_ft)**0.778)*(lf_ft**0.383)*(npax**0.455)
    #    fweight_low = 0.04682*(mtow_lbs**0.692)*(max_per_ft**0.374)*(lf_ft**0.590)
    #    fweight = (fweight_high + fweight_low)/2
    #    mass = fweight*0.453592

def calc_landing_gear_weight(mtom:float) -> float:
    """Returns the mass of the landing gear subsystem, using simplified Cessna method
    for retractable landing gears. Adapted from Eq. 5.38 (pg 81) Pt 5. Component 
    Weight Estimation (Roskam).

    :param mtom: maximum take off mass (Kg)
    :type mtom: float
    """        
    # Convert to imperial units
    mtow_lbs = 2.20462 * mtom

    return (0.04*mtow_lbs + 6.2)*0.453592


def calc_powertrain_weight(n_engines:int)->float:
    """Returns the mass of the powertrain, including:
    - engine (scimo https://sci-mo.de/motors/)   13 Kg
    - inverter (scimo https://sci-mo.de/motors/) 10 Kg
    - propeller (online source TODO add source)  20 Kg
    
    
    :param n_engines: number of engines
    :type n_engines: int
    """

    return (2*n_engines)*(13 + 10) + n_engines*20 

def calc_htail_weight(mtom:float, S_h:float, A_h:float, t_r_h:float) -> float:
    """Returns mass of the horizontal tail subsystem, based on Cessna method.
    See Eq. 5.12 (pg 71) Pt 5. Component Weight Estimation (Roskam).
    (Midterm note: only used for Joby)

    :param mtom: maximum take off mass (Kg)
    :type mtom: float
    :param S_h: horizontal tail area (m2)
    :type S_h: float
    :param A_h: aspect ratio horizontal tail (-)
    :type A_h: float
    :param t_r_h: Horizontal tail maximum root thickness in (m)
    :type t_r_h: float
    """
    # Convert to Imperial Units

    w_to_lbs = 2.20462262*mtom
    S_h_ft = 10.7639104*S_h
    t_r_h_ft = 3.2808399*t_r_h

    return (3.184*w_to_lbs**0.887*S_h_ft**0.101*A_h**0.138)/(174.04*t_r_h_ft**0.223)*0.45359237

def calc_nacelle_weight(p_to:float)->float:
    """Returns mass of nacelles, based on Cessna Mehod. See Eq. 5.29 (pg 78)
    Pt 5. Component Weight Estimation (Roskam).

    :param p_to: Take-off power (W)
    :type p_to: float
    """
    # Convert to Imperial Units
    p_to_hp = 0.001341*p_to

    # Factor 0.24 is originally intended for horizontally opposed engines - 
    # since aircraft is electric, it could be decreased.
    return 0.24*p_to_hp*0.45359237


def calc_misc_weight(mtom:float, oew:float, npax:int) -> float:
    """Returns the mass of miscallenous subsystems:
    - Flight control system (Cessna method Eq 7.2 pg 98)
    - Electrical system (Cessna method Eq 7.13 pg 101)
    - Avionics system (Torenbeek method Eq. 7.23 pg 103)
    - Airconditioning system (Torenbeek method Eq 7.29 pg 104)
    - Furnishing mass (Cessna method Eq 7.41 pg 107)
     
    All equations and pages refer to Pt 5. Component Weight Estimation (Roskam

    :param mtom: Maximum take-off mass (Kg)
    :type mtom: float
    :param oew: Operating empty weight (Kg)
    :type oew: float
    :param npax: Number of passengers (including pilots)
    :type npax: int
    """
    # Convert to Imperial Units
    w_to_lbs = 2.20462262*mtom
    w_oew_lbs = 2.20462262*oew

    mass_fc = 0.0168*w_to_lbs                    # Flight control system 
    mass_elec = 0.0268*w_to_lbs                  # Electrical system   
    mass_avionics = 40 + 0.008*w_to_lbs          # Avionics system
    mass_airco = 0.018*w_oew_lbs                 # Airconditioning 
    mass_fur = 0.412*npax**1.145*w_to_lbs**0.489 # Furnishing 

    return np.sum([mass_fur, mass_airco, mass_avionics, mass_elec, mass_fc])*0.45359237


        
def get_weight_vtol(perf_par:AircraftParameters, fuselage:Fuselage, wing:Wing, engine:Engine, vtail:VeeTail, power:Power, test=False):
    """ This function computes the weight of all components, and updates the data structures accordingly.

    It uses the following weight components
    -----------------------------------------
    Powersystem mass -> Sized in power sizing, retrieved from perf class
    Engine mass -> Scimo engines and inverters used
    wing mass -> class II/wingbox code
    vtail mass -> Class II/wingbox code
    fuselage mass -> Class II
    landing gear mass -> Class II
    nacelle mass -> class II
    misc mass -> class II
    -----------------------------------------

    """

    # Wing mass 
    #This is automatically updated in the wing box calculations if they work
    wing.wing_weight = calc_wing_weight(perf_par.MTOM, 
                                        wing.surface, 
                                        perf_par.n_ult, 
                                        wing.aspect_ratio) 

    # Vtail mass
    # Wing equation is used instead of horizontal tail because of the heavy load of the engine which is attached
    vtail.vtail_weight = calc_wing_weight(perf_par.MTOM, 
                                          vtail.surface, 
                                          perf_par.n_ult, 
                                          vtail.aspect_ratio)

    #fuselage mass
    fuselage.fuselage_weight = calc_fuselage_weight(perf_par.MTOM, 
                                                    fuselage.length_fuselage, 
                                                    perf_par.n_ult, 
                                                    fuselage.width_fuselage_outer, 
                                                    fuselage.height_fuselage_outer, 
                                                    const.v_cr,
                                                    const.rho_cr,
                                                    const.rho_sl) # TODO update

    #landing gear mass
    perf_par.lg_mass = calc_landing_gear_weight(perf_par.MTOM)

    # Nacelle and engine mass

    total_engine_mass = calc_powertrain_weight(const.n_engines) + 90 # 90 kg is for the pylon length
    nacelle_mass = calc_nacelle_weight(perf_par.hoverPower)

    engine.totalmass = nacelle_mass + total_engine_mass
    engine.mass_perpowertrain = (engine.totalmass)/const.n_engines
    engine.mass_pernacelle = nacelle_mass/const.n_engines
    engine.mass_pertotalengine = total_engine_mass/const.n_engines

    # Misc mass
    perf_par.misc_mass = calc_misc_weight(perf_par.MTOM, perf_par.OEM, const.npax+1)

    perf_par.OEM = np.sum([power.powersystem_mass, 
                           wing.wing_weight, 
                           vtail.vtail_weight, 
                           fuselage.fuselage_weight, 
                           nacelle_mass, 
                           total_engine_mass, 
                           perf_par.lg_mass, 
                           perf_par.misc_mass])

    perf_par.MTOM =  perf_par.OEM + const.m_pl

    # Update weight not part of a data structure

    return perf_par, wing, vtail, fuselage, engine


