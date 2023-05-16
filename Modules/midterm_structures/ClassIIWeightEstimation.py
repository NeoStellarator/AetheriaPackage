import numpy as np
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
sys.path.append(os.path.join(list(pl.Path(__file__).parents)[2], "modules","powersizing"))

from modules.powersizing.battery import BatterySizing
from modules.powersizing.fuellCell import FuellCellSizing
from modules.powersizing.hydrogenTank import HydrogenTankSizing
from modules.powersizing.energypowerrequirement import MissionRequirements
from modules.powersizing.powersystem import PropulsionSystem, onlyFuelCellSizng
import input.GeneralConstants as  const




class VtolWeightEstimation:
    def __init__(self) -> None:
        self.components = []

    def add_component(self, CompObject):
        """ Method for adding a component to the VTOL

        :param CompObject: The component to be added to the VTOL
        :type CompObject: Component parent class
        """        
        self.components.append(CompObject)  

    def compute_mass(self):
        """ Computes the mass of entire vtol

        :return: Entire mass of VTOL
        :rtype: float
        """        
        mass_lst = [i.return_mass() for i in self.components]
        return np.sum(mass_lst)

class Component():
    """ This is the parent class for all weight components, it initalized the mass
    attribute and a way of easily returning it. This is used in VtolWeightEstimation.
    """    
    def __init__(self) -> None:
        self.mass = None

    def return_mass(self): return self.mass


class Wing(Component):
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, S, n_ult, A):
        """Retunrs the weight of the wing

        :param mtom: maximum take off mass
        :type mtom: float
        :param S: Wing area
        :type S: float
        :param n_ult: Ultimate load factor
        :type n_ult: float
        :param A: Aspect ratio
        :type A: float
        """        
        super().__init__()
        self.id = "wing"
        self.S_ft = S*3.28084
        self.n_ult = n_ult
        self.A = A
        self.mtow_lbs = 2.20462 * mtom
        self.mass = 0.04674*((self.mtow_lbs/2)**0.397)*(self.S_ft**0.36)*(self.n_ult**0.397)*(self.A**1.712)*0.453592

class Fuselage(Component):
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self,identifier, mtom, max_per, lf, npax):
        """ Returns fuselage weight

        :param mtom: Maximum take off weight
        :type mtom: float
        :param max_per:  Maximium perimeter of the fuselage
        :type max_per: float
        :param lf: Fuselage length
        :type lf: float
        :param npax: Amount of passengers including pilot
        :type npax: int
        """        
        super().__init__()
        self.id = "fuselage"
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf
        self.Pmax_ft = max_per*3.28084
        self.npax = npax
        if identifier == "J1":
            self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
            self.mass = self.fweight_high*0.453592
        else:
            self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
            self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.Pmax_ft**0.379)*(self.lf_ft**0.590)
            self.fweight = (self.fweight_high + self.fweight_low)/2
            self.mass = self.fweight*0.453592

class LandingGear(Component):
    def __init__(self, mtom):
        """Computes the mass of the landing gear 

        :param mtom:
        :type mtom: float
        """        
        super().__init__()
        self.id = "landing gear"
        self.mass = 0.04*mtom

class Engines(Component):
    def __init__(self,p_max, p_dense ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :type p_max: float
        :param p_dense: Power density [w]
        :type p_dense: float
        """        
        super().__init__()
        self.id = "engines"
        self.mass = p_max/p_dense

class HorizontalTail(Component):
    def __init__(self, w_to, S_h, A_h, t_r_h ):
        """Computes the mass of the horizontal tail, only used for Joby

        :param W_to: take off weight in  kg
        :type W_to: float
        :param S_h: Horizontal tail area in  m^2
        :type S_h: float
        :param A_h: Aspect ratio horizontal tail
        :type A_h: float
        :param t_r_h: Horizontal tail maximum root thickness in m 
        :type t_r_h: float
        """        

        self.id = "Horizontal tail"
        w_to_lbs = 2.20462262*w_to
        S_h_ft = 10.7639104*S_h
        t_r_h_ft = 3.2808399*t_r_h

        super().__init__()
        self.mass =  (3.184*w_to_lbs**0.887*S_h_ft**0.101*A_h**0.138)/(57.5*t_r_h_ft**0.223)*0.45359237

class Nacelle(Component):
    def __init__(self, w_to):
        """ Returns nacelle weight

        :param w_to: Total take off weight aka MTOM
        :type w_to: float
        """        
        super().__init__()
        self.id = "Nacelles"
        w_to_lbs = 2.20462262*w_to
        self.mass = 0.1*w_to_lbs*0.45359237 # Original was 0.24 but decreased it since the electric aircraft would require less structural weight0

class H2System(Component):
    def __init__(self, energy, cruisePower, hoverPower):
        """Returns the lightest solutions of the hydrogen 

        :param energy: Amount of energy consumed
        :type energy: float
        :param cruisePower: Power during cruise
        :type cruisePower: float
        :param hoverPower: Power during hover
        :type hoverPower: float
        """        
        super().__init__()
        self.id = "Hydrogen system"
        echo = np.arange(0,1.5,0.05)

        #batteries
        Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        #HydrogenBat = BatterySizing(sp_en_den=1.85,vol_en_den=3.25,sp_pow_den=2.9,cost=0,discharge_effiency=0.6,charging_efficiency=1,depth_of_discharge=1)


        #-----------------------Model-----------------
        BatteryUsed = Liionbat
        FirstFC = FuellCellSizing(const.PowerDensityFuellCell,const.VolumeDensityFuellCell,const.effiencyFuellCell, 0)
        FuelTank = HydrogenTankSizing(const.EnergyDensityTank,const.VolumeDensityTank,0)
        InitialMission = MissionRequirements(EnergyRequired= energy, CruisePower= cruisePower, HoverPower= hoverPower )


        #calculating mass
        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= InitialMission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank)

        TotalMass, TankMass, FuelCellMass, BatteryMass = Mass

        OnlyH2Tank, OnlyH2FC = onlyFuelCellSizng(InitialMission, FuelTank, FirstFC)

        self.mass = min(OnlyH2FC + OnlyH2Tank, max(TotalMass))





#TODO add battery/turbine engine system
#TODO Think of penalty for weight of fuselage for crashworthiness, firewall et cetera  

        
