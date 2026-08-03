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


def amount_of_gas_static_line():
    """
    Calculates the amount of gas (molar/standard volume and mass) in a pipeline at a given pressure and temperature at static conditions. 
    You can change the units in the print statements to whatever makes sense for your particular situation.
    Note that this calculation does not account for density changes due to elevation change or flowing pressure drop.
    """

    P    = ureg.Quantity(1000, "psi")   
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
    amount_of_gas_static_line()
