import csv
import math
import os
import warnings
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

import composition
import ht
from component_classes import ureg


def calc_heat_transfer():
    T_gas    = ureg.Quantity(70, "degF").to("degK").magnitude    # K
    D_h    =  ureg.Quantity(29.0, "inch").to("m").magnitude
    eps  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    T_pipe = ureg.Quantity(1100, "degF").to("degK").magnitude    # K
    L = ureg.Quantity(4, "ft").to("m").magnitude    # Pa 
    Q_scfd   = ureg.Quantity(10.0, "mmscf/day")
    P_gas = ureg.Quantity(100, "psi").to("Pa").magnitude 
    flow_area = D_h **2 * math.pi / 4

    AS = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.08,
        y_Propane=0.01,
        y_n_Butane = 0.0,
        y_CarbonDioxide= 0.01,
        y_n_Decane= 0.0,
        y_Water = 0.0,
        eos = "HEOS"
        )


    Qdot = []
    deltaT = []
    Flow_range = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    for i in Flow_range:
        Q_scfd   = ureg.Quantity(i, "mmscf/day")    # Pa

        AS.update(CP.PT_INPUTS, P_gas, T_gas)
        rho = AS.rhomass()
        mdot     = Q_scfd.to("mol/s").magnitude * AS.molar_mass()   # kg/s from mol wt
        mu = AS.viscosity()
        v  = mdot / (rho * flow_area)   # m/s    
        Re      = fluids_Reynolds(V=v, D=D_h, rho=rho, mu=mu)
        Pr = AS.Prandtl()
        f_darcy = fluids_friction_factor(Re=Re, eD=eps / D_h)
        Nu = ht.conv_internal.turbulent_Gnielinski(Re = Re, Pr = Pr, fd = f_darcy)
        k = AS.conductivity()
        h = Nu * k / D_h

        Qdot.append(h * math.pi*D_h * L * (T_pipe - T_gas))

        deltaT.append(Qdot[-1] / mdot / AS.cpmass())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'Consolas'
    plt.title(f'Heat transfer at {ureg.Quantity(P_gas, "Pa").to("psi").magnitude:.0f} psia')
    l1, = ax.plot(Flow_range, Qdot, label='Heat transferred [Watts]', color = 'red')
    ax2 = ax.twinx()
    l2, = ax2.plot(Flow_range, deltaT, label='Gas temperature rise [deg K]', color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlabel('Flow rate (MMSCFD)')  # Add an x-label to the Axes.
    ax.set_ylabel('Heat transferred [Watts]')  # Add a y-label to the Axes.
    ax2.set_ylabel('Gas temperature rise [deg K]')
    ax2.legend([l1, l2], ['Heat transferred [Watts]', 'Gas temperature rise [deg K]'])
    plt.show()

    print(Qdot)
    print(deltaT)

if __name__ == "__main__":
    calc_heat_transfer()