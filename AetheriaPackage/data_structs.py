from dataclasses import dataclass
import json
import pandas as pd
from pydantic import BaseModel, FilePath
from typing import Optional
import pathlib as pl
import AetheriaPackage.GeneralConstants as const

class Aero(BaseModel):
    label : str = "Aero"
    cd0_cruise: float 
    cL_max: float 
    # cL_max_flaps60: float 
    cL_alpha: float 
    e: float 
    cL_endurance: float | None   = None
    cL_alpha0_approach: float | None  = None
    alpha_approach: float | None   = None
    cd_cruise: float | None = None
    cd_upsweep: float | None  = None
    cd_base: float | None  = None
    cL_cruise: float | None  = None
    cm_ac: float | None  = None
    cm_alpha: float | None  = None
    ld_cruise: float | None  = None
    ld_max: float | None  = None
    cl_ld_max: float | None  = None
    downwash_angle_stall: float   = 0.2219
    ld_stall: float | None  = None
    cd_stall: float | None  = None
    cd0_stall: float | None  = None
    mach_stall: float | None  = None
    deps_da: float | None  = None
    mach_cruise: float | None  = None
    cL_plus_slipstream: float | None  = None
    cL_plus_slipstream_stall: float | None  = None
    cd_tot_cruise: float | None  = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Aero"])
        except:
            raise Exception(f"There was an error when loading in {cls}")

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Aero"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

class AircraftParameters(BaseModel):
    label: str = "Aircraft"
    MTOM: float
    OEM: float | None = None
    Stots: float # Total area of wing reference area

    #energy 
    mission_energy: float | None = None
    mission_time: float | None = None
    climb_energy: float | None = None
    cruise_energy: float | None = None
    descend_energy: float | None = None
    hover_energy: float | None = None
    hor_loiter_energy: float | None = None

    #power
    prop_eff: float  # Propulsive efficiency
    cruisePower : float | None = None
    hoverPower : float | None = None
    max_thrust: float | None = None
    max_thrust_per_engine: float | None = None
    
    #performance
    glide_slope: float 
    v_stall: float = const.v_stall
    v_climb: float | None = None
    v_approach: float | None = None
    v_max: float | None = None
    wing_loading_cruise: float | None = None
    TW_max: float | None = None

    # Load factors
    n_max: float | None = None
    n_ult : float | None = None
    turn_loadfactor: float | None = None # Turning load factor

    #CG and weight
    oem_cg : float | None = None
    cg_front : float | None = None
    cg_rear : float | None = None
    cg_front_bar : float | None = None
    cg_rear_bar : float | None = None
    wing_loc: float | None = None
    misc_mass: float | None = None
    lg_mass: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["AircraftParameters"])
        except:
            raise Exception(f"There was an error when loading in {cls}")


    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["AircraftParameters"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


@dataclass
class Battery:
    """
    This class is to estimate the parameters battery.

    :param EnergyDensity: Specific energy density [kWh/kg]
    :param PowerDensity: Power density [kW/kg]
    :param VolumeDensity: Specific volumetric density [kWh/l]
    :param CostDensity: Cost density [US$/kWh]
    :param Efficiency: Efficiency of the tank
    :param Depth_of_discharge: Depth of discharge (DOD)
    :param ChargingEfficiency: Charging efficiency
    :param End_of_life
    """
    #densities
    EnergyDensity : float = 0.3
    PowerDensity : float  = 3.8
    VolumeDensity : float = .85
    CostDensity : float = None

    #extra parameters
    Efficiency : float = None
    Depth_of_discharge : float = 1
    End_of_life : float = 0.8
    ChargingEfficiency : float = None


    def energymass(self, Energy):
        """
        :return: Mass of the battery [kg]
        """
        return Energy/ self.EnergyDensity /self.Efficiency / self.End_of_life
    
    def powermass(self, Power):
        """
        :return: Mass of the battery [kg]
        """
        return Power/ self.PowerDensity / self.Depth_of_discharge /self.End_of_life 


    def volume(self, Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param vol_en_den: Volumetric energy density of the battery [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return Energy /self.VolumeDensity * 0.001

    def price(self, Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per Wh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return Energy *self.CostDensity

    def heat(self, power):
        """
        :param: power[kW]: electric power generated by the battery
        :return: heat generated by the battery
        """
        return power * (1-self.Efficiency
)

class Engine(BaseModel):
    label : str = "Engine"
    x_rotor_loc: list 
    y_rotor_loc: list 
    total_disk_area: float   
    

    pylon_length: float | None = None 
    totalmass: float | None   = None
    # in the following three, mass_perx = mass of x per number of engines
    mass_perpowertrain: float | None  = None
    mass_pernacelle: float | None  = None
    mass_pertotalengine: float | None  = None
    thrust_coefficient: float | None  = None
    t_factor: float | None  = None
    hub_radius: float | None  = None
    prop_radius: float | None  = None
    prop_area: float | None  = None
    prop_n_blades : int | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Engine"])
        except:
            raise Exception(f"There was an error when loading in {cls}")

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Engine"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


@dataclass
class Fluid:
    """Stores fluid paramers
    """
    viscosity: float = None
    thermal_conductivity: float = None
    heat_capacity: float = None
    density: float = None

@dataclass
class FuelCell:
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param maxpower: max power from FC [KW]
    :param mass: mass [kg]
    :param Cost: Cost of the fuel cell [US$] (not loaded)
    :param Efficiency: Efficiency of the fuel cell s
    """
    maxpower = 125 #kW
    mass = 42 #kg
    efficiency = .55
    length = 0.582 #m
    width = 0.43 #m
    depth = 0.156 #m

    def heat(self, power):
        """
        :param: power[kW]: electric power generated by the fuel cell
        :return: heat generated by the fuel cell
        """
        return (1 - self.efficiency)/ self.efficiency * power

    @property
    def volume(self):
        return self.length * self.width * self.depth #volume in m^3
    
    @property
    def price(self):
        return self.maxpower * 75 # 75 dollars per kW


class Fuselage(BaseModel):
    label : str = "Fuselage"
    fuselage_weight: float | None = None
    length_fuselage: float
    beta_crash: float
    length_tail: float 
    diameter_fuselage: float 
    upsweep: float 
    length_cabin: float = 2.7 # Length of the cabin
    height_cabin: float = 1.6 # Length of the cabin
    height_fuselage_outer: float  = 1.6 + const.fuselage_margin
    height_fuselage_inner: float  = 1.88
    width_fuselage_inner: float = 1.88 + const.fuselage_margin 
    width_fuselage_outer: float | None = None
    volume_fuselage: float 
    length_cockpit: float = 2.103
    crash_box_area: float | None = None
    

    # Crash diameter related
    bc: float | None = None # width crash area
    hc: float | None = None # height crash area
    bf: float | None = None # width crash area
    hf: float | None = None # height crash area

    @property
    def max_perimeter(self):
        #TODO Please disucss a better explanation with Jorrick
        return 2*self.height_fuselage_outer + 2*self.width_fuselage_outer


    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Fuselage"])
        except:
            raise Exception(f"There was an error when loading in {cls}")

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Fuselage"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


class HydrogenTank(BaseModel):
    """
    This class is to estimate the parameters of a Hydrogen tank.

    :param EnergyDensity  [kWh/kg]
    :param VolumeDensity [kWh/l]
    :param cost [US$/kWh] (not loaded)
    """
    label: str = "Tank"
    energyDensity: float = 1.8
    volumeDensity: float = 0.6*3.6e6*1000 # J/m^3
    volumeDensity_h2: float = 2e6*1000 # J/m^3
    cost: float =  16
    fuel_cell_eff: float =  0.55


    def mass(self,energy) -> float:
        """
        :return: Mass of the battery
        """
        return energy / self.energyDensity

    def volume(self,energy) -> float:
        """
        :return: Volume of the tank [m^3]
        """
        return energy / self.volumeDensity / self.fuel_cell_eff 

    def price(self,energy) -> float:
        """
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Tank"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

class Power(BaseModel):
    label : str = "Power"
    powersystem_mass: float | None = None
    battery_pos: float 
    battery_mass: float | None = None
    battery_power : float | None = None
    battery_energy : float | None = None
    battery_volume: float | None = None
    fuelcell_volume : float | None = None
    fuelcell_mass: float | None = None
    h2_tank_mass: float | None = None
    h2_tank_volume : float | None = None
    h2_tank_radius: float | None = None
    h2_tank_length: float 
    cooling_mass: float | None = None
    nu_FC_cruise_fraction: float | None = None  
    
    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Power"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Power"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

@dataclass
class Radiator:

    #needed parameters
    W_HX : float
    H_HX: float
    Z_HX: float
    h_tube : float = None
    t_tube : float = None
    t_channel : float = None
    s_fin: float = None
    l_fin: float = None
    h_fin: float = None
    t_fin : float = None

    #calcalated parameters
    
    HX_alpha: float = None
    HX_gamma : float = None
    HX_delta : float= None

    #numbers
    N_channel: int = None
    N_fin : int = None

    #surface area's
    A_cold: float = None
    A_cross_hot: float = None
    A_hot: float = None
    A_fin : float = None
    A_fs_cross : float = None
    W_channel: float = None


    def load(self):
        df = pd.read_csv(r"input/radiator_values/HX.csv")
        self.h_tube = df['h_tube'][0]
        self.t_tube = df['t_tube'][0]
        self.t_channel = df['t_channel'][0]
        self.s_fin = df['s_fin'][0]
        self.t_fin = df['t_fin'][0]
        self.h_fin = df['h_fin'][0]
        self.l_fin = df['l_fin'][0]

    def dump(self):

        column =  [ 'h_tube', 't_tube', 't_channel', 's_fin', 'l_fin', 'h_fin', 't_fin']
        data = [ self.h_tube, self.t_tube, self.t_channel, self.s_fin, self.l_fin, self.h_fin, self.t_fin] 
        data = dict(zip(column,data))
        df = pd.DataFrame.from_dict([data])
        df.to_csv("radiator_values/HX.csv", columns=list(data.keys()), index= False)



class Stab(BaseModel):
    label : str = "Stability"
    Cm_de: float | None = None
    Cn_dr: float | None = None
    Cxa: float | None = None
    Cxq: float | None = None
    Cza: float | None = None
    Czq: float | None = None
    Cma: float | None = None
    Cmq: float | None = None
    Cz_adot: float | None = None
    Cm_adot: float | None = None
    muc: float | None = None
    Cxu: float | None = None
    Czu: float | None = None
    Cx0: float | None = None
    Cz0: float | None = None
    Cmu: float | None = None
    Cyb: float | None = None
    Cyp: float | None = None
    Cyr: float | None = None
    Clb: float | None = None
    Clp: float | None = None
    Clr: float | None = None
    Cnb: float | None = None
    Cnp: float | None = None
    Cnr: float | None = None
    Cy_dr: float | None = None
    Cy_beta_dot: float | None = None
    Cn_beta_dot: float | None = None
    mub: float | None = None
    sym_eigvals: float | None = None
    asym_eigvals: float | None = None
    cg_front_bar: float | None = None
    cg_rear_bar: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Stability"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Stability"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

class VeeTail(BaseModel):
    label : str = "Vtail"
    vtail_weight: float | None  = None
    surface: float 
    virtual_hor_surface: float | None  = None
    virtual_ver_surface: float | None  = None
    aspect_ratio: float | None = None
    span: float | None  = None
    chord_root: float | None = None
    chord_tip: float | None = None
    chord_mac: float | None = None
    thickness_to_chord: float | None  = 0.12
    quarterchord_sweep: float 
    taper: float | None  = 1
    dihedral: float | None  = None
    ruddervator_efficiency: float | None  = None
    shs: float | None  = None
    elevator_min: float | None  = None
    rudder_max: float | None  = None
    max_clh: float | None  = None
    c_control_surface_to_c_vee_ratio: float | None  = None
    cL_cruise: float | None  = None
    length_wing2vtail: float | None  = None # position of LEMAC of veetail relative to LEMAC of main wing
    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Veetail"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Veetail"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

class Wing(BaseModel):
    """"ANGLES IN RADIANS PLEASE"""
    label : str = "Wing"
    wing_weight: float| None = None
    surface: float | None = None
    aspect_ratio: float 
    span: float | None = None
    chord_root: float | None = None
    chord_tip: float | None = None
    chord_mac: float | None = None
    thickness_to_chord: float = 0.12
    quarterchord_sweep: float 
    sweep_LE: float| None = None
    taper: float 
    x_lemac: float | None= None
    x_lewing: float| None = None # position of LEMAC of main wing relative to nose of ac
    y_mac: float| None = None
    washout: float 

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Wing"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Wing"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


# Propeller structure from tuduam module
class Propeller(BaseModel):
    n_blades:float
    """The number of blades on the propellor"""    
    r_prop:float
    """"Propeller radius"""
    rpm_cruise:float 
    """"The rotations per minute during cruise flight"""
    xi_0:float
    """"Non-dimensional hub radius (r_hub/R) [-]"""
    chord_arr: Optional[list] = None
    """"Array with the chords at each station"""
    rad_arr: Optional[list] = None
    """"Radial coordinates for each station"""
    pitch_arr: Optional[list] = None
    """"Array with the pitch at each station"""
    tc_ratio:Optional[float] = None
    """Thickness over chord ratio of the airfoil"""    

    # to be consistent with the rest of datastructure file, these methods have been copied from above 
    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Propeller"])
        except:
            raise Exception(f"There was an error when loading in {cls}")

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Propeller"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)