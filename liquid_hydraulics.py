



def liquid_hydraulics():
    import math
    line_diameter = 0.076860358 #meters
    line_length = 99.97459995 #meters
    roughness = 4.572e-5 #meters
    elevation_change = -1 #meters
    API_gravity = 55.0 
    viscosity = 0.001 # pascal-second
    flow_rate = 0.003680261 #cubic meters per second
    grav_constant = 9.8066 #(m/s^2)

    #calculate the following:
    flow_area = math.pi * line_diameter ** 2 / 4
    velocity = flow_rate / flow_area
    line_volume = flow_area * line_length
    density = 1000 * 141.5/(API_gravity + 131.5) #kilograms per cubic meter
    Re = density * velocity * line_diameter/viscosity

    print(f'flow area = {flow_area} m^2')
    print(f'velocity = {velocity} m/s')
    print(f'density = {density} kg/m^3')
    print(f'Reynolds number = {Re}')
    if Re > 2000:
        #Turbulent flow correlation from "Fluid Mechanics for Chemical Engineers, Third Edition" by Noel de Nevers p.187. 
        #Note that the Fanning friction factor is one fourth of the Darcy friction factor
        fanning_friction = 0.001375 * (1 + (20000.0 * roughness / line_diameter + 1e6 / Re)**(1.0/3.0))
        print(f'Turbulent fanning friction = {fanning_friction}')
    else:
        #laminar flow
        fanning_friction = 16.0/Re
        print(f'Laminar fanning friction = {fanning_friction}')
    
    #pressure drop due to friction = 4 * f * (L/D) * v^2 / 2 . If all inputs are SI units, output will be in Pascals

    dP_friction = 2 * fanning_friction * (line_length/line_diameter) * velocity ** 2 * density

    dP_elevation = elevation_change * grav_constant * density

    dP = dP_friction + dP_elevation

    print(f'Total pressure drop = {dP} Pa')

# def main_cadlc():
    # import pandas as pd
    # from CoolProp.CoolProp import PropsSI
    # from scipy.optimize import brentq
    # import math
    # from pint import UnitRegistry
    # ureg = UnitRegistry()
    # line_diameter = 1.0
    # line_length = 1000.0
    # roughness = 4.572e-5
    # elevation_change = -5.0
    # API_gravity = 55.0
    # density = ureg.Quantity
    # density(1000 * 141.5/(API_gravity + 131.5),ureg.kg/(ureg.meter ** 3))
    # print(f'Density: {density.magnitude}')

if __name__ == "__main__":
    liquid_hydraulics()

