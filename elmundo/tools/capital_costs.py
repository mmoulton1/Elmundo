import numpy as np
from numpy import interp, mean, dot
from math import log10, ceil, log
from geopy.distance import geodesic 
from geopy.point import Point
from shapely.geometry import Point as ShapelyPoint, Polygon, MultiPolygon

#define a function for the capital cost of H2 storage in salt caverns. I am using the model from Papadias
def capital_cost_salt_cavern(storage_capacity): 
    """
    Calculate the capital cost of H2 storage in salt caverns using the Papadias model.

    Parameters:
    storage_capacity (float): Storage capacity of the cavern in kg.

    Returns:
    float: Adjusted installed capital cost in 2018 USD.
    """
    # Installed capital cost
    a = 0.092548
    b = 1.6432
    c = 10.161
    d = 641057.2079 #cost in USD per km
    cost_per_kg_H2 = np.exp(a*(np.log(storage_capacity/1000))**2 - b*np.log(storage_capacity/1000) + c)  # 2019 [USD] from Papadias
    installed_capex = cost_per_kg_H2 * storage_capacity   
    cepci_overall = 1.29/1.30 # Convert from $2019 to $2018
    adjusted_installed_capex = installed_capex * cepci_overall
    return adjusted_installed_capex, cost_per_kg_H2

def capital_cost_underground_pipes(storage_capacity):
    """
    Calculate the capital cost of H2 storage in underground pipes using a similar model to Papadias.

    Parameters:
    storage_capacity (float): Storage capacity of the pipes in kg.

    Returns:
    float: Adjusted installed capital cost in 2018 USD.
    float: Cost per kg H2 in 2019 USD.
    """
    # Installed capital cost constants for underground pipes
    a = 0.001559
    b = 0.035313
    c = 4.5183

    # Calculate cost per kg H2 in 2019 USD
    cost_per_kg_H2 = np.exp(a * (np.log(storage_capacity / 1000)) ** 2 - b * np.log(storage_capacity / 1000) + c)
    
    # Calculate installed capital expenditure (CAPEX) in 2019 USD
    installed_capex = cost_per_kg_H2 * storage_capacity
    
    # Cost adjustment from 2019 to 2018 USD
    cepci_overall = 1.29 / 1.30
    adjusted_installed_capex = installed_capex * cepci_overall
    
    return adjusted_installed_capex, cost_per_kg_H2

'''
    Author: Jamie Kee
    Feb 7, 2023
    Source for most equations is HDSAM3.1, H2 Compressor sheet
    Output is in 2016 USD
'''


class Compressor:
    def __init__(self, p_outlet, flow_rate_kg_d, p_inlet=20, n_compressors=2, sizing_safety_factor=1.1):
        '''
            Parameters:
            ---------------
            p_outlet: oulet pressure (bar)
            flow_Rate_kg_d: mass flow rate in kg/day
        '''
        self.p_inlet = p_inlet # bar
        self.p_outlet = p_outlet # bar
        self.flow_rate_kg_d = flow_rate_kg_d # kg/day

        self.n_compressors = n_compressors # At least 2 compressors are recommended for operation at any given time
        self.n_comp_back_up = 1 # Often times, an extra compressor is purchased and installed so that the system can operate at a higher availability.
        self.sizing_safety_factor = sizing_safety_factor # typically oversized. Default to oversize by 10%

        if flow_rate_kg_d*(1/24)*(1/60**2)/n_compressors > 5.4:
            # largest compressors can only do up to about 5.4 kg/s
            """
            H2A Hydrogen Delivery Infrastructure Analysis Models and Conventional Pathway Options Analysis Results
            DE-FG36-05GO15032
            Interim Report
            Nexant, Inc., Air Liquide, Argonne National Laboratory, Chevron Technology Venture, Gas Technology Institute, National Renewable Energy Laboratory, Pacific Northwest National Laboratory, and TIAX LLC
            May 2008
            """
            raise ValueError("Invalid compressor design. Flow rate must be less than 5.4 kg/s per compressor")
            
    def compressor_power(self):
        R = 8.314 # Universal gas constant in J/mol-K
        T = 25 + 273.15 # Temperature in Kelvin (25°C)
    
        cpcv = 1.41 # Specific heat ratio (Cp/Cv) for hydrogen
        sizing = self.sizing_safety_factor # Sizing safety factor, typically 110%
        isentropic_efficiency = 0.88 # Isentropic efficiency for a reciprocating compressor

        # Hydrogen compressibility factor (Z) at different pressures (bar) and 25°C
        Z_pressures = [1, 10, 50, 100, 300, 500, 1000]
        Z_z = [1.0006, 1.0059, 1.0297, 1.0601, 1.1879, 1.3197, 1.6454]
        Z = np.mean(np.interp([self.p_inlet, self.p_outlet], Z_pressures, Z_z)) # Average Z factor between inlet and outlet pressures

        # Compression ratio per stage
        c_ratio_per_stage = 2.1
        # Number of stages required
        self.stages = np.ceil((np.log10(self.p_outlet) - np.log10(self.p_inlet)) / np.log10(c_ratio_per_stage))
        
        # Flow rate per compressor (kg/day) converted to kg-mols/sec
        flow_per_compressor = self.flow_rate_kg_d / self.n_compressors # kg/day
        flow_per_compressor_kg_mols_sec = flow_per_compressor / (24 * 60 * 60) / 2.0158 # kg-mols/sec

        # Pressure ratio
        p_ratio = self.p_outlet / self.p_inlet

        # Theoretical power required for compression (kW)
        theoretical_power = Z * flow_per_compressor_kg_mols_sec * R * T * self.stages * (cpcv / (cpcv - 1)) * ((p_ratio) ** ((cpcv - 1) / (self.stages * cpcv)) - 1)
        # Actual power required considering isentropic efficiency (kW)
        actual_power = theoretical_power / isentropic_efficiency

        # Motor efficiency based on empirical formula
        motor_efficiency = np.dot(
            [0.00008, -0.0015, 0.0061, 0.0311, 0.7617],
            [np.log(actual_power) ** x for x in [4, 3, 2, 1, 0]]
        )
        # Motor rating with sizing safety factor (kW)
        self.motor_rating = sizing * actual_power / motor_efficiency
    def compressor_system_power(self):
            return self.motor_rating, self.motor_rating*self.n_compressors # [kW] total system power
    
    def compressor_costs(self):
        n_comp_total = self.n_compressors + self.n_comp_back_up # 2 compressors + 1 backup for reliability
        production_volume_factor = 0.55 # Assume high production volume
        CEPCI = 1.29/1.1 #Convert from 2007 to 2016$

        cost_per_unit = 1962.2*self.motor_rating**0.8225*production_volume_factor*CEPCI
        if self.stages>2:
            cost_per_unit = cost_per_unit * (1+0.2*(self.stages-2))

        install_cost_factor = 2

        direct_capex = cost_per_unit*n_comp_total*install_cost_factor

        land_required = 10000 #m^2 This doesn't change at all in HDSAM...?
        land_cost = 12.35 #$/m2
        land = land_required*land_cost

        other_capital_pct = [0.05,0.1,0.1,0,0.03,0.12] # These are all percentages of direct capex (site,E&D,contingency,licensing,permitting,owners cost)
        other_capital = dot(other_capital_pct,[direct_capex]*len(other_capital_pct)) + land
        
        total_capex = direct_capex + other_capital
        return total_capex

def pipeline(distance):
    d = 641057.2079 #cost per km in USD
    pipe_cost = d * distance 
    return pipe_cost
