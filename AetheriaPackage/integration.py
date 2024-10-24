
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AetheriaPackage.data_structs import *
from AetheriaPackage.sim_contr import size_vtail_opt, span_vtail
from AetheriaPackage.aerodynamics import wing_planform, vtail_planform, component_drag_estimation, get_aero_planform, weissinger_l
from AetheriaPackage.performance import get_wing_power_loading,get_performance_updated
from AetheriaPackage.propulsion import propcalc
from AetheriaPackage.structures import get_gust_manoeuvr_loadings, get_weight_vtol, get_fuselage_sizing
from AetheriaPackage.power import power_system_convergences

def run_integration(file_path, counter_tuple=(1,1), json_path=None, dir_path=None):
    """ Runs an entire integraton loop

    :param file_path: Path to the initial estmate
    :type file_path: str
    :param counter_tuple: tuple with two integers, defaults to (1,1)
    :type counter_tuple: tuple, optional
    :param json_path: Path to json (storing) file, defaults to None
    :type json_path: str, optional
    :param dir_path: Path to working directory, defaults to None
    :type dir_path: str, optional
    """    
    #----------------------------- Initialize classes --------------------------------
    if counter_tuple == (1,1):
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank()
        mission = AircraftParameters.load(file_path)
        wing  =  Wing.load(file_path)
        engine = Engine.load(file_path)
        aero = Aero.load(file_path)
        fuselage = Fuselage.load(file_path)
        vtail = VeeTail.load(file_path)
        stability = Stab.load(file_path)
        power = Power.load(file_path)
    else:
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank()
        mission = AircraftParameters.load(json_path)
        wing  =  Wing.load(json_path)
        engine = Engine.load(json_path)
        aero = Aero.load(json_path)
        fuselage = Fuselage.load(json_path)
        vtail = VeeTail.load(json_path)
        stability = Stab.load(json_path)
        power = Power.load(json_path)
    #----------------------------------------------------------------------------------

    #-------------------- Preliminary Sizing ------------------------------------------
    get_wing_power_loading(mission, wing, engine, aero)
    get_gust_manoeuvr_loadings(mission, aero)
    
    #-------------------- Planform Sizing ---------------------------------------------
    wing_planform(wing, mission.MTOM, mission.wing_loading_cruise)

    #-------------------- Aerodynamic Sizing ------------------------------------------
    alpha_arr,cL_lst, induced_drag_lst =  get_aero_planform(aero, wing, 20)
    component_drag_estimation(wing, fuselage, vtail, aero) #
    
    # Compute releant parameters for emergency approach
    lift_over_drag_arr = cL_lst/(induced_drag_lst + aero.cd0_cruise)
    aero.ld_max = np.max(lift_over_drag_arr)
    aero.cl_ld_max = cL_lst[np.argmax(lift_over_drag_arr)]
    aero.alpha_approach= alpha_arr[np.argmax(lift_over_drag_arr)]
    aero.cL_alpha0_approach= cL_lst[np.argmax(lift_over_drag_arr)]

    mission.glide_slope = np.arctan(1/aero.ld_max)
    CL_CD_endurance_opt = (cL_lst**(3/2))/(induced_drag_lst + aero.cd0_cruise)
    aero.cL_endurance = cL_lst[np.argmax(CL_CD_endurance_opt)]
    aero.downwash_angle_stall =  np.average(weissinger_l(wing, aero.alpha_approach, 20)[3])

    #-------------------- Propulsion --------------------------------------------------
    propcalc(aero, mission=mission, engine=engine, h_cruise= const.h_cruise)
    
    #-------------------- Flight Performance ------------------------------------------
    get_performance_updated(aero, mission, wing,engine, power)

    #-------------------- Power System Sizing -----------------------------------------
    power_system_convergences(power, mission) 

    #-------------------- Stability and Control ---------------------------------------
    size_vtail_opt( WingClass=wing,
                    AircraftClass=mission,
                    PowerClass=power,
                    EngineClass=engine,
                    Aeroclass=aero,
                    FuseClass=fuselage,
                    VTailClass=vtail, 
                    StabClass=stability,
                    CLh_initguess=-0.1,
                    stepsize = 5e-2,
                ) 

    #-------------------- Structures --------------------------------------------------
    vtail_planform(vtail)
    get_fuselage_sizing(Tank, Pstack, mission, fuselage, power)

    #-------------------- Weight Estimation -------------------------------------------
    get_weight_vtol(mission, fuselage, wing, aero, engine, vtail, power)

    #-------------------- dumping update parameters to json file ----------------------
    if dir_path is not None:
        mission.dump(json_path)
        wing.dump(json_path)
        engine.dump(json_path)
        aero.dump(json_path)
        fuselage.dump(json_path)
        vtail.dump(json_path)
        fuselage.dump(json_path)
        stability.dump(json_path)
        power.dump(json_path)

    #-------------------- Log all variables from current iterations -------------------

    if dir_path is not None:
        for data_struct in [mission, wing, engine, aero, fuselage, vtail, stability, power]:
            save_path = os.path.join(dir_path, "aetheria" + "_" + data_struct.label + "_hist.csv")
            data = data_struct.model_dump()
            
            if os.path.exists(save_path):
                pd.DataFrame(np.array(list(data.values()), dtype=object).reshape(1, -1)).to_csv(save_path, mode="a", header=False, index= False)
            else: 
                pd.DataFrame([data]).to_csv(save_path, columns= list(data.keys()), index=False)
                    # Read the output from the subprocess
    else:
        return mission, wing, engine, aero, fuselage, stability, power


def multi_run(file_path, outer_loop_counter, json_path, dir_path, save_inner_convergence=False,
              max_inner_loops=25, eps_exit=0.001, n_min_eps=3):
        print(f"{'='*30}\nOuter loop iteration = {outer_loop_counter}\n{'='*30}")
        eps_lst = [] # store the consecutive differences

        with open(json_path, 'r') as f:
            data = json.load(f)
            MTOM_one = data["AircraftParameters"]["MTOM"]
        print(f"MTOM: {MTOM_one} [Kg]")
        for i in range(1, max_inner_loops+1):
            print(f'\nInner loop Iteration = {i}') 
            run_integration(file_path  ,(outer_loop_counter, i),json_path, dir_path) # run integration files which can be looped

            # load data so that convergences can be checked
            with open(json_path, 'r') as f:
                data = json.load(f)
            MTOM_two = data["AircraftParameters"]["MTOM"]

            #log data so that convergences can be monitored live
            print(f"MTOM: {MTOM_two} [Kg]")

            #break out of the convergences loop if the mtom convergences below eps_exit
            eps = abs(MTOM_two - MTOM_one) / MTOM_one
            eps_lst.append(eps)

            if len(eps) >= n_min_eps:
                if max(eps[-n_min_eps:]) < eps_exit: 
                    print(f" Inner loop has converged -> epsilon is: {100*eps:.4f} %")
                    break
            MTOM_one = MTOM_two
            
        if save_inner_convergence:
            iteration_count = range(1, len(eps_lst)+1)

            fig, ax = plt.subplots()
            ax.semilogy(iteration_count, eps_lst)
            ax.set_title(f'Outer Loop: {outer_loop_counter}')
            ax.set_xlabel('Inner Loop Iteration')
            ax.set_ylabel('eps')
            
            save_dir = os.path.join(dir_path, 'inner_convergence_plots')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            plt.savefig(os.path.join(save_dir, f'outer_{outer_loop_counter:>02}_convergence_plot.png'), 
                        dpi=300, bbox_inches='tight')

