from component_classes import (
    ureg,
)
import math
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition

def phase_env():
    """Generates a phase envelope diagram for a given composition"""
    import matplotlib.pyplot as plt
    AS_g = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    try:
        AS_g.build_phase_envelope("dummy")
        PE = AS_g.get_phase_envelope_data()
        plt.plot(PE.T, PE.p, '-', label='Composition')
        plt.xlabel('Temperature [K]')
    except ValueError as VE:
        print(VE)

    plt.ylabel('Pressure [Pa]')
    plt.yscale('log')
    plt.title('Phase Envelope for Selected Composition')
    plt.legend(loc='lower right', shadow=True)
    plt.savefig('methane-ethane.png')


def gas_properties():
    """Calculates basic gas properties for gas at a given pressure, temperature, and composition. Optionally can also take a line ID and flow rate to calculate velocity and Mach number"""
    P    = ureg.Quantity(3000, "psi")   #Absoulute pressure, NOT GAUGE
    T    = ureg.Quantity(300, "K")   

    AS = composition.define_composition(
        y_Methane = 0.8,
        y_Ethane = 0.11,
        y_Propane=0.05,
        y_n_Butane = 0.02,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )

    ID    = ureg.Quantity(8.125, "inch")
    A    = math.pi * ID**2 / 4.0
    flow_rate = ureg.Quantity(1000, "mscf/day")

    AS.update(CP.PT_INPUTS, P.to("Pa").magnitude, T.to("K").magnitude)
    rho_mass = ureg.Quantity(AS.rhomass(), "kg/m^3")
    rho_molar = ureg.Quantity(AS.rhomolar(), "mol/m^3")
    molar_mass = ureg.Quantity(AS.molar_mass(), "kg/mol")
    velocity = flow_rate/(rho_molar * A)
    speed_of_sound = ureg.Quantity(AS.speed_sound(), "m/s")

    print('\n')
    print(f'Inputs: P = {P.to('psi')}, T = {T.to('degF')}')
    print('\n')
    print(f'Outputs: \n'
        f'Molar mass of gas = {molar_mass.to("g/mol")}\n'
        f'Mass Density = {rho_mass.to("lb/ft^3")}\n'
        f'Molar density = {rho_molar.to("mol/ft^3")}\n'
        f'Compressibility = {AS.compressibility_factor()}\n'
        f'Flowing velocity = {velocity.to("ft/s")}\n'
        f'Speed of sound = {speed_of_sound.to("ft/s")}\n'
        f'Mach number = {(velocity/speed_of_sound).to("")}' #converts to dimensionless

        )


def amount_of_gas_static_line():
    """
    Calculates the amount of gas (molar/standard volume and mass) in a pipeline at a given pressure and temperature at static conditions. 
    You can change the units in the print statements to whatever makes sense for your particular situation.
    Note that this calculation does not account for density changes due to elevation change or flowing pressure drop.
    """

    P    = ureg.Quantity(1000, "psi")    #Absoulute pressure, NOT GAUGE
    T    = ureg.Quantity(300, "K")   
    ID    = ureg.Quantity(8.125, "inch")
    A    = math.pi * ID**2 / 4.0
    dL   = ureg.Quantity(10000.0,    "feet")
    AS = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    AS.update(CP.PT_INPUTS, P.to("Pa").magnitude, T.to("K").magnitude)
    rho_mass = AS.rhomass()
    rho_molar = AS.rhomolar()
    #mass = volume * density = A * L * rho_mass
    mass_gas = A * dL * ureg.Quantity(rho_mass, "kg/m^3")
    moles_gas = A * dL * ureg.Quantity(rho_molar, "mol/m^3") 

    print('\n')
    print(f'Inputs: P = {P.to('psi')}, T = {T.to('degF')}')
    print('\n')
    print(f'Contents at static conditions:')
    print(f' Volume = {(A*dL).to("oil_bbl")}') 
    print(f' Gas mass in line = {mass_gas.to("lb")}')
    print(f' Gas quantity = {moles_gas.to("mscf") }')
    

if __name__ == "__main__":
    #amount_of_gas_static_line()
    gas_properties()
