
import numpy as np
import os
import json
import sys
import pathlib as pl
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from modules.powersizing import PropulsionSystem
from input import GeneralConstants
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size # Well this should have probably been used
from modules.planform.planformsizing import wing_planform
from modules.preliminary_sizing.wing_power_loading_functions import get_wing_power_loading
from modules.structures.Flight_Envelope import get_gust_manoeuvr_loadings
from modules.aero.clean_class2drag import integrated_drag_estimation
from modules.aero.slipstream_cruise_function import slipstream_cruise
from modules.aero.slipstream_stall_function import slipstream_stall
from modules.flight_perf.performance  import get_energy_power_perf
from modules.structures.fuselage_length import get_fuselage_sizing
from modules.structures.ClassIIWeightEstimation import get_weight_vtol
from modules.structures.wingbox_optimizer import GetWingWeight
# from modules.propellor.propellor_sizing import propcalc
from scripts.structures.vtail_span import span_vtail
import input.GeneralConstants as const
from scripts.power.finalPowersizing import power_system_convergences





def run_integration(label, file_path, counter_tuple=(0,0)):
    #----------------------------- Initialize classes --------------------------------
    if counter_tuple == (1,1):
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
        mission = AircraftParameters.load(r"input/initial_estimate.json")
        wing  =  Wing.load(r"input/initial_estimate.json")
        engine = Engine.load(r"input/initial_estimate.json")
        aero = Aero.load(r"input/initial_estimate.json")
        fuselage = Fuselage.load(r"input/initial_estimate.json")
        vtail = VeeTail.load(r"input/initial_estimate.json")
        stability = Stab.load(r"input/initial_estimate.json")
        power = Power.load(r"input/initial_estimate.json")
    else:
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
        mission = AircraftParameters.load(file_path)
        wing  =  Wing.load(file_path)
        engine = Engine.load(file_path)
        aero = Aero.load(file_path)
        fuselage = Fuselage.load(file_path)
        vtail = VeeTail.load(file_path)
        stability = Stab.load(file_path)
        power = Power.load(file_path)
    #----------------------------------------------------------------------------------


    # Preliminary Sizing
    mission, wing,  engine, aero = get_wing_power_loading(mission, wing, engine, aero)
    mission =  get_gust_manoeuvr_loadings(mission, aero)
    

    #planform sizing
    wing = wing_planform(wing, mission.MTOM, mission.wing_loading_cruise)


    #-------------------- propulsion ----------------------------
    # mission, engine = propcalc( clcd= aero.ld_cruise, mission=mission, engine= engine, h_cruise= GeneralConstants.h_cruise)

    #-------------------- Aerodynamic sizing--------------------
    wing, fuselage, vtail, aero, horizontal_tail =  integrated_drag_estimation(wing, fuselage, vtail, aero) #TODO go through this functions and find inital estime
    aero = slipstream_cruise(wing, engine, aero, mission) # TODO the effect of of cl on the angle of attack

    #-------------------- Flight Performance --------------------
    wing, engine, aero, mission = get_energy_power_perf(wing, engine, aero, mission)





    #-------------------- power system sizing--------------------
    power, mission = power_system_convergences(power, mission) #


    #-------------------- stability and control--------------------
    #The function here loads and dumps a couple of times so to make sure nothing goes wrong therefore
    #before this class everything has to be dumped and loaded again

    #loading
    mission.dump()
    wing.dump()  
    engine.dump() 
    aero.dump() 
    horizontal_tail.dump() 
    fuselage.dump() 
    vtail.dump() 
    stability.dump() 


    #dumping
    mission.load()
    wing.load()  
    engine.load() 
    aero.load() 
    horizontal_tail.load() 
    fuselage.load() 
    vtail.load() 
    stability.load() 
    
    b_ref = span_vtail(1,fuselage.diameter_fuselage,30*np.pi/180)
    wing,horizontal_tail,fuselage,vtail, stability = size_vtail_opt(WingClass=  wing,
                                                                    Aeroclass= aero,
                                                                    HorTailClass= horizontal_tail,
                                                                    FuseClass= fuselage,
                                                                    VTailClass= vtail, 
                                                                    StabClass=stability,
                                                                    b_ref= b_ref, #!!!!!!!!! please update value when we get it
                                                                    stepsize = 5e-2,
                                                                    ) 

    #loading
    mission.load()
    wing.load()  
    engine.load() 
    aero.load() 
    horizontal_tail.load() 
    fuselage.load() 
    vtail.load() 
    stability.load() 

    #------------- Structures------------------

    # Fuselage sizing
    fuselage = get_fuselage_sizing(Tank,Pstack, mission, fuselage)
    ''' 
    try: 
        wing = GetWingWeight(wing, engine, material, aero )
    except TypeError:
        wing.taper += 0.05
        print(f"Wing box design failed, adding 0.05 to taper. New taper = {wing.taper}")
    '''


    #------------- weight_estimation------------------
    mission, fuselage, wing, engine, vtail =  get_weight_vtol(mission, fuselage, wing, engine, vtail)


    #Final dump
    mission.dump()
    wing.dump()  
    engine.dump() 
    aero.dump() 
    horizontal_tail.dump() 
    fuselage.dump() 
    vtail.dump() 
    stability.dump()
    power.dump()
    

    #--------------------------------- Log all variables from current iterations ----------------------------------
    # Load data from JSON file
    save_path = r"output\final_convergence_history"
    with open(const.json_path) as jsonFile:
        data = json.load(jsonFile)

    if os.path.exists(os.path.join(save_path, "aetheria" + "_" + label + "_hist.csv")):
        pd.DataFrame(np.array(list(data.values()), dtype= object).reshape(1, -1)).to_csv(os.path.join(save_path, "aetheria" + "_" + label + "_hist.csv") , mode="a", header=False, index= False)
    else: 
        pd.DataFrame([data]).to_csv(os.path.join(save_path, "aetheria" + "_" + label + "_hist.csv"), columns= list(data.keys()), index=False)
            # Read the output from the subprocess



if __name__ == "__main__":
    
    run_integration()

