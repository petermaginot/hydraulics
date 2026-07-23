"""
Dumping ground for various test functions used while developing things
"""

import math
import os
import time
import warnings

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

import composition
from component_classes import ureg
from compressible_flow import (
    Bend as Compressible_Bend,
    CheckValve as Compressible_CheckValve,
    ChokedFlowError,
    Contraction_Expansion as Compressible_Contraction_Expansion,
    Line_Segment as Compressible_Line_Segment,
    Valve as Compressible_Valve,
    Orifice as Compressible_Orifice,
    FlowState,
    _build_phase_limits,
    _resolve_mdot,
    _safe_update_PT,
    compressible_pipe_segment,
    compressible_dA,
)
from incompressible import (
    Bend as Incompressible_Bend,
    CheckValve as Incompressible_CheckValve,
    Contraction_Expansion as Incompressible_Contraction_Expansion,
    Incompressible_Fluid,
    Line_Segment as Incompressible_Line_Segment,
    Orifice as Incompressible_Orifice,
    Valve as Incompressible_Valve,
    _resolve_flow_rate,
    export_pressure_profile,
    print_results,
)
from compressible_network import Compressible_Network


def test_comp_hydraulics():

    P_gas    = ureg.Quantity(1000, "psi").to("Pa").magnitude    # Pa
    T_gas    = 300.0    # K
    D_gas    = ureg.Quantity(4.026, "inch").to("m").magnitude
    A_gas    = math.pi * D_gas**2 / 4.0
    eps_gas  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    dL_gas   = ureg.Quantity(40.0,    "feet").to("m").magnitude
    dz_gas   = ureg.Quantity(40.0,    "feet").to("m").magnitude
    AS_g = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    AS_g.update(CP.PT_INPUTS, P_gas, T_gas)
    rho_gas = AS_g.rhomass()
    S_in = AS_g.smass()
    Q_scfd   = ureg.Quantity(200.0, "mmscf/day")
    mdot     = Q_scfd.to("mol/s").magnitude * AS_g.molar_mass()   # kg/s from mol wt
    G_gas    = mdot / A_gas
    a_in   = AS_g.speed_sound()                # m/s
    v_in   = mdot / (rho_gas * A_gas)          # m/s
    Ma_in = v_in/a_in

    print('\n')
    print(f'inputs: P = {P_gas}, Smass= {S_in}, velocity = {v_in}, Mach number = {Ma_in}')
    # print('\n')
    fs_iso = FlowState(AS_g, mdot, A=A_gas, z=0.0)
    compressible_pipe_segment(
        fs_iso,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        isothermal=True,
    )
    outlet_P = AS_g.p()
    outlet_T = AS_g.T()
    rho_out = AS_g.rhomass()
    a_out   = AS_g.speed_sound()                # m/s
    v_out   = mdot / (rho_out * A_gas)          # m/s
    Ma_out = v_out/a_out
    S_out = AS_g.smass()

    # print('\n')
    print('Isothermal case')
    print(f'outputs: P = {outlet_P}, T = {outlet_T}, Smass= {S_out}, velocity = {v_out}, Mach number = {Ma_out}')
    print('\n')

    AS_g.update(CP.PT_INPUTS, P_gas, T_gas) #reinitialize abstract state

    fs_ad = FlowState(AS_g, mdot, A=A_gas, z=0.0)
    compressible_pipe_segment(
        fs_ad,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        isothermal=False,
        q_wall = ureg.Quantity(0, "Btu/hr").to("watt").magnitude,
    )
    outlet_P = AS_g.p()
    outlet_T = AS_g.T()
    rho_out = AS_g.rhomass()
    a_out   = AS_g.speed_sound()                # m/s
    v_out   = mdot / (rho_out * A_gas)          # m/s
    Ma_out = v_out/a_out
    S_out = AS_g.smass()

    print('Adiabatic case')
    print(f'outputs: P = {outlet_P}, T = {outlet_T}, Smass= {S_out}, velocity = {v_out}, Mach number = {Ma_out}')

def test_compressible_line_segment_csv():
    csv_path = os.path.join(os.path.dirname(__file__), "testprofile_crane.csv")
    roughness = ureg.Quantity(0.00015, "ft")

    seg = Compressible_Line_Segment.from_csv(csv_path, roughness=roughness, name='1')

    P_in = ureg.Quantity(1300, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(40, "degF").to("degK").magnitude   # K
    Q_scfd = ureg.Quantity(125.5, "mmscf/day")

    AS = composition.define_composition(
        y_Methane = 0.75,
        y_Ethane = 0.21,
        y_Propane=0.04,
        # y_n_Butane = 0.01,
        # y_CarbonDioxide= 0.02,
        # y_Water = 1.0,
        eos = "HEOS"
    )
    AS.update(CP.PT_INPUTS, P_in, T_in)
    rho_in = AS.rhomass()
    area_in = seg.profile[0][3]
    v_in = _resolve_mdot(Q_scfd, AS) / (rho_in * area_in)
    Ma_in = v_in / AS.speed_sound()

    print("\ntest_line_segment_csv")
    print(f"  inlet: P={P_in:.4g} Pa, T={T_in} K, Ma={Ma_in:.4f}")

    fs_seg = FlowState(AS, _resolve_mdot(Q_scfd, AS), A=seg.inlet_area_si, z=0.0)
    profile_points = seg.dP_dT(fs_seg, isothermal=True, mu=1.1e-5)
    P_out = AS.p()
    T_out = AS.T()

    rho_out = AS.rhomass()
    area_out = seg.profile[-1][3]
    v_out = _resolve_mdot(Q_scfd, AS) / (rho_out * area_out)
    Ma_out = v_out / AS.speed_sound()

    P_out_psi = ureg.Quantity(P_out, "Pa").to("psi").magnitude
    dP_psi = ureg.Quantity(P_out - P_in, "Pa").to("psi").magnitude
    print(f"  outlet: P={P_out_psi:.4g} psi, T={T_out:.4g} K, Ma={Ma_out:.4f}")
    print(f"  dP = {dP_psi:.4f} psi")

    import matplotlib.pyplot as plt

    dist_ft = [ureg.Quantity(pt[0], "m").to("ft").magnitude  for pt in profile_points]
    P_psi   = [ureg.Quantity(pt[1], "Pa").to("psi").magnitude for pt in profile_points]
    T_degF  = [ureg.Quantity(pt[2], "degK").to("degF").magnitude for pt in profile_points]
    v_fts = [ureg.Quantity(pt[3], "m").to("ft").magnitude for pt in profile_points]

    plt.rcParams["font.family"] = "Consolas"
    fig, ax1 = plt.subplots()

    l1, = ax1.plot(dist_ft, P_psi,  color="black", label="Pressure [psi]")
    ax1.set_xlabel("Distance [ft]")
    ax1.set_ylabel("Pressure [psi]")


    ax2 = ax1.twinx()

    l2, = ax2.plot(dist_ft, T_degF, color="red",   label="Temperature [°F]")
    ax2.set_ylabel("Temperature [°F]")

    # l2, = ax2.plot(dist_ft, v_fts, color="red",   label="Velocity(ft/s)")
    # ax2.set_ylabel("Velocity (ft/s)")

    ax2.legend([l1, l2], ["Pressure [psi]", "Temperature (°F)"])
    plt.title(f'Pressure and temperature profile at {Q_scfd}')
    plt.tight_layout()
    plt.show()




def test_compressible_fittings():
    D_small = ureg.Quantity(3.826, "in")
    D_large = ureg.Quantity(4.026, "in")
    contraction_test = Compressible_Contraction_Expansion(Di_US=D_large, Di_DS= D_small)
    expansion_test = Compressible_Contraction_Expansion(Di_US=D_small, Di_DS= D_large)
    elbow = Compressible_Bend(D_large, 90, 1.5)
    P_in = ureg.Quantity(1000, "psi").to("Pa").magnitude
    T_in = 300.0   # K
    Q_scfd = ureg.Quantity(60, "mmscf/day")
    print(f'Inlet P:{P_in}, T:{T_in}')
    AS = composition.define_composition(
        y_Methane = 0.0,
        y_Ethane = 0.00,
        y_Propane=0.00,
        y_n_Butane = 1,
        y_CarbonDioxide= 0.0,
        eos = "HEOS"
    )
    AS.update(CP.PT_INPUTS, P_in, T_in)

    mdot_fittings = _resolve_mdot(Q_scfd, AS)
    fs_fit = FlowState(AS, mdot_fittings, A=contraction_test.inlet_area_si, z=0.0)
    contraction_test.dP_dT(fs_fit)
    print(f'After contraction P:{AS.p()}, T:{AS.T()}')
    expansion_test.dP_dT(fs_fit)
    print(f'After expansion P:{AS.p()}, T:{AS.T()}')
    elbow.dP_dT(fs_fit)
    print(f'After elbow P:{AS.p()}, T:{AS.T()}')


def test_incompressible_p2p():
    """Simple two-point pressure drop with entrance and exit losses."""

    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(3.068, "inch")
    length           = ureg.Quantity(2000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft")

    segment = Incompressible_Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
    )

    fluid = Incompressible_Fluid(
        density=ureg.Quantity(1000.0, "kg/m^3"),
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    flow_rate = ureg.Quantity(60, "oil_bbl/day")

    dP_line = segment.dP(fluid, flow_rate)

    # Entrance and exit losses computed manually (K = 0.5 and 1.0 respectively,
    # both referenced to the pipe velocity head).
    Q   = _resolve_flow_rate(flow_rate, fluid.density_si)
    A   = segment.profile[0][3]
    v   = Q / A
    rho = fluid.density_si

    dP_entrance = -0.5 * 0.5 * rho * v ** 2   # K = 0.5 for sharp-edged entrance
    dP_exit     = -1.0 * 0.5 * rho * v ** 2   # K = 1.0 for abrupt exit

    dP_q = ureg.Quantity(dP_line + dP_entrance + dP_exit, "Pa")

    print(f"  dP entrance : {ureg.Quantity(dP_entrance, 'Pa').to('psi'):.4f~P}")
    print(f"  dP line     : {ureg.Quantity(dP_line,     'Pa').to('psi'):.4f~P}")
    print(f"  dP exit     : {ureg.Quantity(dP_exit,     'Pa').to('psi'):.4f~P}")
    print(f"  dP total    : {dP_q.to('psi'):.4f~P}")


def test_incompressible_cont():
    """Segment + bend + contraction/expansion example."""

    id_val = ureg.Quantity(4.026, "inch")

    segment = Incompressible_Line_Segment.from_csv(
        csv_path="testprofile_ID_OD_WT.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
    )

    bend1  = Incompressible_Bend(Di=id_val, ang_deg=90.0, bend_dias=1.5)
    cont1  = Incompressible_Contraction_Expansion(
        Di_US=id_val, Di_DS=ureg.Quantity(3.826, "inch")
    )
    exp1   = Incompressible_Contraction_Expansion(
        Di_DS=id_val, Di_US=ureg.Quantity(3.826, "inch")
    )

    fluid = Incompressible_Fluid(
        density=ureg.Quantity(1000.0, "kg/m^3"),
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    flow_rate = ureg.Quantity(20000, "oil_bbl/day")

    dP_seg   = segment.dP(fluid, flow_rate)
    dP_bend  = bend1.dP(fluid, flow_rate)
    dP_cont1 = cont1.dP(fluid, flow_rate)
    dP_exp1  = exp1.dP(fluid, flow_rate)
    dP_tot   = dP_seg + dP_bend + dP_cont1 + dP_exp1

    print(f"  {segment}")
    print(f"  {bend1}")
    print(f"  {cont1}")
    print(f"  {exp1}")
    print(f"  dP segment  : {ureg.Quantity(dP_seg,   'Pa').to('psi'):.4f~P}")
    print(f"  dP bend     : {ureg.Quantity(dP_bend,  'Pa').to('psi'):.4f~P}")
    print(f"  dP cont1    : {ureg.Quantity(dP_cont1, 'Pa').to('psi'):.4f~P}")
    print(f"  dP exp1     : {ureg.Quantity(dP_exp1,  'Pa').to('psi'):.4f~P}")
    print(f"  dP total    : {ureg.Quantity(dP_tot,   'Pa').to('psi'):.4f~P}")


def test_incompressible_csv_profile():
    """Load a variable-geometry profile from a CSV and export a pressure
    profile."""

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    segment = Incompressible_Line_Segment.from_csv(
        csv_path="testprofile_Dh_Flow_area.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
    )

    flow_rate = ureg.Quantity(6000, "oil_bbl/day")
    P0        = ureg.Quantity(100.0, "psi").to("Pa").magnitude

    dP_total = segment.dP(fluid, flow_rate)
    print_results(fluid, segment, flow_rate, dP_total)

    output_csv = "testprofile_Dh_Flow_area_pressure_profile.csv"
    export_pressure_profile(segment, fluid, flow_rate, output_csv, P0=P0)


def test_K():
    """Tests compressible_K via Valve.dP_dT across three flow regimes.

    Valve: Cv=1.0, Di=0.5 inch.
    Fluid: 100 psi / 150 °F natural gas from example_composition.csv.

    Scenario 1 — linear path:  small mdot, |dP|/P_in << 5%, linearized formula used.
    Scenario 2 — fallback:     moderate mdot, |dP|/P_in > 5%,
                               compressible_changing_area_K invoked.
    Scenario 3 — choked inlet: near-sonic velocity, RuntimeError raised.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits

    valve = Compressible_Valve(Di=ureg.Quantity(0.5, "inch"), Cv=1.0)
    A = math.pi * valve.Di_si ** 2 / 4.0

    print("\ntest_K")

    # --- Scenario 1: linear path (|dP|/P_in ~ 0.06%, denominator ~ 1.0) ---
    mdot_s1 = ureg.Quantity(0.001, "kg/s")
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    H_in   = AS.hmass()
    rho_in = AS.rhomass()
    v_in   = mdot_s1.to("kg/s").magnitude / (rho_in * A)

    fs_s1 = FlowState(
        AS, mdot_s1.to("kg/s").magnitude, A=A, z=0.0,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )
    valve.dP_dT(fs_s1)

    P_out1    = AS.p()
    rel_dP1   = (P_in - P_out1) / P_in
    v_out1    = mdot_s1.to("kg/s").magnitude / (AS.rhomass() * A)
    energy_err = abs((H_in + v_in**2 / 2.0) - (AS.hmass() + v_out1**2 / 2.0))

    assert P_out1 < P_in, f"Scenario 1: expected P_out < P_in"
    assert rel_dP1 < 0.05, f"Scenario 1: |dP|/P_in={rel_dP1:.3%} should be < 5% (linear regime)"
    assert energy_err < 5.0, f"Scenario 1: stagnation-enthalpy error {energy_err:.2f} J/kg > 5 J/kg"
    print(f"  Scenario 1 (linear):   P_out={P_out1:.0f} Pa, |dP|/P_in={rel_dP1:.3%}, "
          f"energy residual={energy_err:.3f} J/kg")

    # --- Scenario 2: fallback via |dP|/P_in > dPmax (~12-13% at this flow rate) ---
    mdot_s2 = ureg.Quantity(0.015, "kg/s")
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    fs_s2 = FlowState(
        AS, mdot_s2.to("kg/s").magnitude, A=A, z=0.0,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )
    valve.dP_dT(fs_s2)

    P_out2  = AS.p()
    rel_dP2 = (P_in - P_out2) / P_in

    assert P_out2 < P_in, f"Scenario 2: expected P_out < P_in"
    assert rel_dP2 > 0.05, f"Scenario 2: |dP|/P_in={rel_dP2:.3%} should be > 5% (fallback regime)"
    print(f"  Scenario 2 (fallback): P_out={P_out2:.0f} Pa, |dP|/P_in={rel_dP2:.3%}")

    # --- Scenario 3: Ma_in >= 0.98, RuntimeError expected ---
    mdot_s3 = ureg.Quantity(0.25, "kg/s")
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    try:
        fs_s3 = FlowState(
            AS, mdot_s3.to("kg/s").magnitude, A=A, z=0.0,
            T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
            T_critical=T_critical, P_critical=P_critical,
        )
        valve.dP_dT(fs_s3)
        raise AssertionError("Scenario 3: expected RuntimeError for choked flow but none was raised")
    except RuntimeError as e:
        print(f"  Scenario 3 (choked):   RuntimeError correctly raised: {e}")


def pseudo_orifice():
    from compressible_flow import compressible_changing_area_K, choked_mass_flux
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(114.696, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits
    
    OD = ureg.Quantity(4.026, "in")
    ID = ureg.Quantity(0.25, "in")

    A = OD.magnitude**2*math.pi/4
    A_orifice = ID.to("m").magnitude**2*math.pi/4

    Beta = ID/OD

    Cd = 0.62

    K = (1-Beta.magnitude**4)/Cd**2
    
    mdot_s3 = ureg.Quantity(0.000001, "kg/s")
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    
    fs_s3 = FlowState(
        AS, mdot_s3.to("kg/s").magnitude, A=A, z=0.0,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )
    results =  choked_mass_flux(fs_s3, A_orifice*.62,A)
    mdot_choked, P_throat, T_throat, rho_throat, P_outlet, T_outlet = results

    print(results)

def trying_orifices():
    from compressible_flow import compressible_changing_area_K, choked_mass_flux
    import fluids.fittings
    import fluids.flow_meter
    import numpy as np
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits

    valve_Cv = 1.76
    orifice_Cd = 0.62

    ID = ureg.Quantity(1.939, "inch")
    D_orifice_trim = ureg.Quantity(0.25, "inch")
    ID_si             = ID.to("m").magnitude
    D_orifice_trim_si = D_orifice_trim.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4

    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    mdot = _resolve_mdot(flow_rate = ureg.Quantity(0.0001, "mmscf/day"), abstract_state=AS)
    fs = FlowState(
        AS=AS, mdot=mdot, A=A_pipe, z=0.0,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )

    valve = Compressible_Valve(Di=ID, Cv=valve_Cv) #100% open 2" D4 with 1" trim

    orifice = Compressible_Orifice(Di = ID, Do = D_orifice_trim)
    #find equivalent discharge coefficient for this valve
    Cd_valve = fluids.flow_meter.K_to_discharge_coefficient(ID_si, D_orifice_trim_si, valve.K)

    #choked flow rate through valve
    A_throat_valve = Cd_valve * math.pi * D_orifice_trim_si ** 2 / 4

    valve_choked_rate = choked_mass_flux(fs, A_throat_valve, A_outlet=A_pipe)
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions
    orifice_choked_rate = choked_mass_flux(fs, orifice_Cd*orifice.Do_si**2/4*math.pi, A_outlet=A_pipe)


    print(f'Cd of the valve: {Cd_valve}')
    print(f'mdot: {mdot}, Pin = {fs.P}, Tin = {fs.T}')
    print(f'valve choked flow rate: {valve_choked_rate}')
    print(f'orifice choked flow rate: {orifice_choked_rate}')
    print(f'\nmdot, P_orifice, P_valve')

    for i in np.arange(0.002, .05 , 0.002):
        flow_rate = ureg.Quantity(i, "kg/s")

        mdot = _resolve_mdot(flow_rate = flow_rate, abstract_state=AS)
        fs.mdot = mdot

        _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

        valve.dP_dT(fs)
        valve_P = fs.P
        # print(f'Valve outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
        _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions


        orifice.dP_dT(fs)
        orifice_P = fs.P
        # print(f'orifice outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
        print(f'{mdot}, {orifice_P}, {valve_P}')


def test_contraction_expansion():
    import fluids
    OD = 60.32
    WTs = [1.65,	2.77,3.18,3.91,	5.54,8.74,	11.07]
    ID_min = OD - 2 * max(WTs)
    for i in WTs:
        ID = OD - i*2
        K_ds = fluids.fittings.contraction_sharp(Di1=ID, Di2=ID_min)
        Kcont    = K_ds * (ID_min / ID) ** 4
    
        # Expansion: fluids returns K w.r.t. upstream velocity directly.
        Kexp = fluids.fittings.diffuser_sharp(Di1=ID_min, Di2=ID)
        print(f'WT:{i},K contraction: {Kcont}, K expansion: {Kexp}')

def test_compressible_dA():
    from compressible_flow import compressible_changing_area_K, choked_mass_flux
    import fluids.fittings
    import fluids.flow_meter
    import numpy as np
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    P_out_test = ureg.Quantity(50, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude
    
    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits

    valve_Cv = 1.76
    orifice_Cd = 0.62

    ID = ureg.Quantity(1.939, "inch")
    D_orifice_trim = ureg.Quantity(0.25, "inch")
    ID_si             = ID.to("m").magnitude
    D_orifice_trim_si = D_orifice_trim.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4

    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    mdot = _resolve_mdot(flow_rate = ureg.Quantity(0.07, "mmscf/day"), abstract_state=AS)

    fs = FlowState(
        AS=AS, mdot=mdot, A=A_pipe, z=0.0,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )

    valve = Compressible_Valve(Di=ID, Cv=valve_Cv) #100% open 2" D4 with 1" trim

    orifice = Compressible_Orifice(Di = ID, Do = D_orifice_trim)
    #find equivalent discharge coefficient for this valve
    Cd_valve = fluids.flow_meter.K_to_discharge_coefficient(ID_si, D_orifice_trim_si, valve.K)

    #choked flow rate through valve
    A_throat_valve = Cd_valve * math.pi * D_orifice_trim_si ** 2 / 4

    valve_choked_rate = choked_mass_flux(fs, A_throat_valve, A_outlet=A_pipe)
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions
    orifice_choked_rate = choked_mass_flux(fs, orifice_Cd*orifice.Do_si**2/4*math.pi, A_outlet=A_pipe)
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

    #Compare results of compressible_dA to the valve and orifice dP_dT functions. Might be slightly different due to the differences in how entropy is accounted for 
    # (dP_dT functions currently used average of inlet/outlet temperature for calculating entropy. compressible_dA function uses average of throat/outlet temperature)
    #Orifice first.
    print(f'mdot = {mdot}')

    valve.dP_dT(fs)

    print(f'dP_dT function results')
    print(f'Valve outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

    orifice.dP_dT(fs)

    print(f'orifice outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

    print(f'compressible_dA function results')

    fs.mdot = mdot
    compressible_dA(fs, A_throat_valve, K = valve.K, A2 = None, P2 = None)
    P2_valve_mode1 = fs.P
    print(f'Valve outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

    # NOTE: Orifice should expose a .K property derived from Cd; for now compute it inline.
    K_orifice = fluids.flow_meter.discharge_coefficient_to_K(ID_si, D_orifice_trim_si, orifice_Cd)
    A_throat_orifice = orifice_Cd * math.pi * D_orifice_trim_si ** 2 / 4

    fs.mdot = mdot
    compressible_dA(fs, A_throat_orifice, K = K_orifice, A2 = None, P2 = None)
    P2_orifice_mode1 = fs.P
    print(f'orifice outlet pressure: {fs.P}, pressure ratio = {fs.P/P_in}, temperature: {fs.T}')
    _safe_update_PT(AS, P_in, T_in, *phase_limits) #return to initial conditions

    # -------------------------------------------------------------------
    # Mode 2 round-trip: feed the Mode 1 P_out back as P2, verify the
    # solver recovers the same mdot (within tight tolerance) and lands at
    # the same outlet pressure.
    # -------------------------------------------------------------------
    print('\nMode 2 round-trip (dictate P2, solve mdot)')

    fs.mdot = mdot   # seed; only enters via h_stagnation in choked_mass_flux.
    compressible_dA(fs, A_throat_valve, K=valve.K, A2=None, P2=P2_valve_mode1)
    mdot_valve_recovered = fs.mdot
    P_valve_recovered    = fs.P
    rel_mdot_err_valve = abs(mdot_valve_recovered - mdot) / mdot
    print(f'  Valve   : mdot_in={mdot:.6g}, mdot_out={mdot_valve_recovered:.6g}, '
          f'rel err={rel_mdot_err_valve:.2e}, P_out={P_valve_recovered:.6g} '
          f'(target {P2_valve_mode1:.6g})')
    assert rel_mdot_err_valve < 5e-3, (
        f"Mode 2 valve round-trip: mdot drift {rel_mdot_err_valve:.2%} exceeds 0.5%"
    )
    assert abs(P_valve_recovered - P2_valve_mode1) < 100.0, (
        f"Mode 2 valve round-trip: P_out drift {abs(P_valve_recovered - P2_valve_mode1):.4g} Pa "
        f"exceeds 100 Pa"
    )
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    fs.mdot = mdot
    compressible_dA(fs, A_throat_orifice, K=K_orifice, A2=None, P2=P2_orifice_mode1)
    mdot_orifice_recovered = fs.mdot
    P_orifice_recovered    = fs.P
    rel_mdot_err_orifice = abs(mdot_orifice_recovered - mdot) / mdot
    print(f'  Orifice : mdot_in={mdot:.6g}, mdot_out={mdot_orifice_recovered:.6g}, '
          f'rel err={rel_mdot_err_orifice:.2e}, P_out={P_orifice_recovered:.6g} '
          f'(target {P2_orifice_mode1:.6g})')
    assert rel_mdot_err_orifice < 5e-3, (
        f"Mode 2 orifice round-trip: mdot drift {rel_mdot_err_orifice:.2%} exceeds 0.5%"
    )
    assert abs(P_orifice_recovered - P2_orifice_mode1) < 100.0, (
        f"Mode 2 orifice round-trip: P_out drift "
        f"{abs(P_orifice_recovered - P2_orifice_mode1):.4g} Pa exceeds 100 Pa"
    )
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    # -------------------------------------------------------------------
    # Mode 2 choked branch: pick a P2 well below P2_choked. Solver should
    # clamp mdot to the choked value and the outlet stagnation enthalpy
    # should match the inlet.
    # -------------------------------------------------------------------
    print('\nMode 2 choked branch (P2 << P2_choked)')
    h_in_static = AS.hmass()
    # Capture stagnation enthalpy at the *Mode 1* mdot (small v_in => h0 ~ h_static).
    v_in_seed   = mdot / (AS.rhomass() * A_pipe)
    h0_inlet    = h_in_static + 0.5 * v_in_seed**2

    fs.mdot = mdot
    P2_low = 0.2 * P_in
    compressible_dA(fs, A_throat_valve, K=valve.K, A2=None, P2=P2_low)
    mdot_choked_solved = fs.mdot
    v_out_chk = fs.v
    h_out_chk = fs.AS.hmass()
    h0_out    = h_out_chk + 0.5 * v_out_chk**2

    # Independent choked-rate reference from choked_mass_flux at the inlet
    # stagnation state for comparison.
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    fs.A    = A_pipe
    fs.mdot = mdot
    mdot_ref, *_ = choked_mass_flux(fs, A_throat_valve)
    rel_mdot_choked_err = abs(mdot_choked_solved - mdot_ref) / mdot_ref
    h0_rel_err = abs(h0_out - h0_inlet) / max(abs(h0_inlet), 1.0)
    print(f'  mdot_choked solved={mdot_choked_solved:.6g}, reference={mdot_ref:.6g}, '
          f'rel err={rel_mdot_choked_err:.2e}')
    print(f'  h0 drift: inlet={h0_inlet:.6g}, outlet={h0_out:.6g}, '
          f'rel err={h0_rel_err:.2e}')
    assert rel_mdot_choked_err < 1e-2, (
        f"Mode 2 choked branch: mdot mismatch {rel_mdot_choked_err:.2%} exceeds 1%"
    )
    assert h0_rel_err < 1e-4, (
        f"Mode 2 choked branch: stagnation enthalpy drift {h0_rel_err:.2%} exceeds 0.01%"
    )
    _safe_update_PT(AS, P_in, T_in, *phase_limits)

    # -------------------------------------------------------------------
    # Input validation guards.
    # -------------------------------------------------------------------
    print('\nInput validation')
    fs.mdot = mdot
    fs.A    = A_pipe
    for label, kwargs, expected_substr in [
        ("P2 >= P_in", dict(A_throat=A_throat_valve, K=valve.K, A2=A_pipe, P2=P_in + 1.0),
         "P2 must be strictly less than"),
        ("K < 0",     dict(A_throat=A_throat_valve, K=-1.0, A2=A_pipe, P2=None),
         "K must be non-negative"),
        ("A_throat > fs.A", dict(A_throat=A_pipe * 2.0, K=valve.K, A2=A_pipe, P2=None),
         "A_throat must be in"),
    ]:
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        fs.A = A_pipe
        try:
            compressible_dA(fs, **kwargs)
        except ValueError as e:
            assert expected_substr in str(e), (
                f"Validation case {label!r} raised ValueError but message "
                f"missing {expected_substr!r}: {e}"
            )
            print(f'  {label}: ValueError raised correctly')
        else:
            raise AssertionError(
                f"Validation case {label!r} did not raise ValueError"
            )


def test_dmdot_dT_roundtrip():
    """Round-trip every compressible component's dP_dT against its
    dmdot_dT inverse.

    For each component: pick a representative geometry and inlet state,
    run forward dP_dT at a chosen mdot to capture P_out, then run
    dmdot_dT(fs_fresh, P2=P_out) and assert the recovered mdot matches
    the original to tight tolerance.

    Covers: Valve (constricted + plain), CheckValve (forward flow),
    Orifice, Bend, Contraction_Expansion (contraction direction),
    Line_Segment (adiabatic + isothermal).  Expansion is excluded -- its
    dmdot_dT raises NotImplementedError by design (kinetic recovery
    typically dominates K-loss, reversing the residual sign).
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)

    ID    = ureg.Quantity(1.939, "inch")
    ID_si = ID.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4.0
    D_trim = ureg.Quantity(0.25, "inch")

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot_target = _resolve_mdot(ureg.Quantity(0.07, "mmscf/day"), AS)

    fs = FlowState(
        AS=AS, mdot=mdot_target, A=A_pipe, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2], P_critical=phase_limits[3],
    )

    def reset(A=A_pipe, mdot=mdot_target):
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        fs.A    = A
        fs.z    = 0.0
        fs.mdot = mdot

    print("\ntest_dmdot_dT_roundtrip")

    TOL = 5e-3
    cases = []

    def roundtrip(label, component, dP_dT_kwargs=None, dmdot_dT_kwargs=None,
                  area=A_pipe):
        dP_dT_kwargs   = dP_dT_kwargs   or {}
        dmdot_dT_kwargs = dmdot_dT_kwargs or {}
        reset(A=area)
        component.dP_dT(fs, **dP_dT_kwargs)
        P2 = fs.P
        reset(A=area)
        fs.mdot = 0.0
        component.dmdot_dT(fs, P2=P2, **dmdot_dT_kwargs)
        err = abs(fs.mdot - mdot_target) / mdot_target
        cases.append((label, err, fs.mdot, fs.P, P2))
        assert err < TOL, f"{label}: rel err {err:.2e} > {TOL:.0e}"

    roundtrip("Valve+throat", Compressible_Valve(Di=ID, Cv=1.76, minimum_diameter=D_trim))
    roundtrip("Valve plain",  Compressible_Valve(Di=ID, K=10.0))
    roundtrip("CheckValve fwd", Compressible_CheckValve(Di=ID, K=20.0))
    roundtrip("Orifice",      Compressible_Orifice(Di=ID, Do=D_trim))
    roundtrip("Bend",         Compressible_Bend(Di=ID, ang_deg=90.0, bend_dias=3.0))
    roundtrip("Contraction 2->1\"",
              Compressible_Contraction_Expansion(
                  Di_US=ID, Di_DS=ureg.Quantity(1.0, "inch")))

    seg = Compressible_Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ID,
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="seg",
    )
    roundtrip("Line_Segment adiabatic",  seg, area=seg.inlet_area_si)
    roundtrip("Line_Segment isothermal", seg,
              dP_dT_kwargs={"isothermal": True},
              dmdot_dT_kwargs={"isothermal": True},
              area=seg.inlet_area_si)

    for label, err, mdot_out, P_out, P_target in cases:
        print(f"  {label:25s}: rel err={err:.2e}, mdot={mdot_out:.6g} "
              f"(target {mdot_target:.6g}), P_out={P_out:.6g} "
              f"(target {P_target:.6g})")


def test_dmdot_dT_choke_raises():
    """Verify dmdot_dT raises ChokedFlowError when P2 is below the
    component's choke limit, with mdot_choked populated on the exception
    so the network solver can clamp against it.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude
    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)

    ID    = ureg.Quantity(1.939, "inch")
    ID_si = ID.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4.0

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot_seed = _resolve_mdot(ureg.Quantity(0.07, "mmscf/day"), AS)
    fs = FlowState(
        AS=AS, mdot=mdot_seed, A=A_pipe, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2], P_critical=phase_limits[3],
    )

    print("\ntest_dmdot_dT_choke_raises")

    # P2 = 1 kPa is well below the choke pressure for any component at
    # 100 psi inlet -- subsonic resolution must fail.
    P2_below_choke = 1.0e3

    def assert_chokes(label, component):
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        fs.A    = A_pipe
        fs.z    = 0.0
        fs.mdot = mdot_seed
        try:
            component.dmdot_dT(fs, P2=P2_below_choke)
        except ChokedFlowError as exc:
            assert exc.mdot_choked > 0.0, (
                f"{label}: ChokedFlowError raised but mdot_choked is non-positive "
                f"({exc.mdot_choked!r})"
            )
            print(f"  {label:13s}: ChokedFlowError raised, "
                  f"mdot_choked={exc.mdot_choked:.4g} kg/s")
        else:
            raise AssertionError(f"{label}: ChokedFlowError not raised")

    assert_chokes("Plain Valve", Compressible_Valve(Di=ID, K=10.0))
    assert_chokes("Bend",        Compressible_Bend(Di=ID, ang_deg=90.0, bend_dias=3.0))


def test_orifice_dmdot_dT_vs_dA():
    """Confirm Orifice.dmdot_dT agrees with a direct compressible_dA Mode 2
    call when Cd_override is supplied -- which eliminates the RHG Cd<->mdot
    fixed point so the two paths reduce to the same numerical problem.

    Regression guard: the Cd fixed point should converge to the same mdot
    the direct call returns; any drift signals a bug in the loop's
    convergence criterion or state restoration.
    """
    import fluids.flow_meter

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_composition.csv")
    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude
    AS = composition.define_composition_from_csv(csv_path)
    phase_limits = _build_phase_limits(AS)

    ID    = ureg.Quantity(1.939, "inch")
    D_o   = ureg.Quantity(0.25,  "inch")
    ID_si = ID.to("m").magnitude
    Do_si = D_o.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4.0
    A_bore = math.pi * Do_si ** 2 / 4.0
    Cd_fixed = 0.62

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot_target = _resolve_mdot(ureg.Quantity(0.07, "mmscf/day"), AS)

    fs = FlowState(
        AS=AS, mdot=mdot_target, A=A_pipe, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2], P_critical=phase_limits[3],
    )

    orf = Compressible_Orifice(Di=ID, Do=D_o, Cd_override=Cd_fixed)
    orf.dP_dT(fs)
    P2 = fs.P

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    fs.A    = A_pipe
    fs.mdot = 0.0
    orf.dmdot_dT(fs, P2=P2)
    mdot_via_orf = fs.mdot

    K_orf = fluids.flow_meter.discharge_coefficient_to_K(
        D=ID_si, Do=Do_si, C=Cd_fixed,
    )
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    fs.A    = A_pipe
    fs.mdot = 0.0
    compressible_dA(fs, A_throat=Cd_fixed * A_bore, K=K_orf, A2=A_pipe, P2=P2)
    mdot_via_dA = fs.mdot

    rel_drift = abs(mdot_via_orf - mdot_via_dA) / max(mdot_via_dA, 1e-30)

    print("\ntest_orifice_dmdot_dT_vs_dA")
    print(f"  forward mdot (target) : {mdot_target:.6g} kg/s, P2 target {P2:.6g} Pa")
    print(f"  Orifice.dmdot_dT      : {mdot_via_orf:.6g} kg/s")
    print(f"  compressible_dA(P2=)  : {mdot_via_dA:.6g} kg/s")
    print(f"  rel drift             : {rel_drift:.2e}")
    assert rel_drift < 1e-6, (
        f"Orifice/dA drift {rel_drift:.2e} exceeds 1e-6 -- Cd fixed point may "
        f"have a convergence-criterion or state-restoration bug."
    )


def test_incompressible_dmdot_roundtrip():
    """Round-trip every incompressible component's dP against its
    dmdot inverse.

    For each component: pick a representative geometry and a forward
    mass flow rate, compute the resulting outlet pressure via dP(), then
    invert via dmdot(P_inlet, P_outlet) and assert the recovered mdot
    matches the original.

    Analytic components (Valve, CheckValve, Contraction_Expansion) round-
    trip to machine precision (~1e-12).  Fixed-point components (Bend,
    Orifice) round-trip to the loop tolerance (~1e-7).  Line_Segment
    uses brentq and round-trips to ~1e-5.
    """
    print("\ntest_incompressible_dmdot_roundtrip")

    fluid = Incompressible_Fluid(density=1000.0, viscosity=1e-3)
    ID    = ureg.Quantity(2.0, "inch")
    ID_si = ID.to("m").magnitude
    A     = math.pi * ID_si ** 2 / 4.0
    rho   = fluid.density_si

    P_inlet = 5.0e5   # 5 bar absolute, well above any cavitation
    mdot_target_pq = ureg.Quantity(20.0, "kg/s")
    mdot_target    = mdot_target_pq.to("kg/s").magnitude

    cases  = []
    TOL    = 5e-3

    def roundtrip(label, component, mdot_pq=mdot_target_pq, tol=TOL,
                  P_in=P_inlet):
        dP        = component.dP(fluid, mdot_pq)
        P_out     = P_in + dP
        mdot_inv  = component.dmdot(fluid, P_in, P_out)
        mdot_in   = mdot_pq.to("kg/s").magnitude
        err       = abs(mdot_inv - mdot_in) / mdot_in
        cases.append((label, err, mdot_inv, mdot_in, dP))
        assert err < tol, f"{label}: rel err {err:.2e} > {tol:.0e}"

    roundtrip("Valve K=10",
              Incompressible_Valve(Di=ID, K=10.0))
    roundtrip("CheckValve K=20 fwd",
              Incompressible_CheckValve(Di=ID, K=20.0))
    roundtrip("Bend 90 deg",
              Incompressible_Bend(Di=ID, ang_deg=90.0, bend_dias=3.0))
    roundtrip("Orifice 0.75\" bore",
              Incompressible_Orifice(
                  Di=ID, Do=ureg.Quantity(0.75, "inch")))
    roundtrip("Contraction 2->1\"",
              Incompressible_Contraction_Expansion(
                  Di_US=ID, Di_DS=ureg.Quantity(1.0, "inch")))
    roundtrip("Expansion 1\"->2\"",
              Incompressible_Contraction_Expansion(
                  Di_US=ureg.Quantity(1.0, "inch"), Di_DS=ID),
              mdot_pq=ureg.Quantity(5.0, "kg/s"))

    seg = Incompressible_Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ID,
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="seg",
    )
    roundtrip("Line_Segment flat",
              seg, mdot_pq=ureg.Quantity(15.0, "kg/s"), tol=1e-4)

    seg_down = Incompressible_Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ID,
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(-10.0, "ft"),
        name="seg_down",
    )
    roundtrip("Line_Segment downhill",
              seg_down, mdot_pq=ureg.Quantity(15.0, "kg/s"), tol=1e-4)

    for label, err, mdot_inv, mdot_in, dP in cases:
        print(f"  {label:25s}: rel err={err:.2e}, mdot_inv={mdot_inv:.6g} "
              f"(target {mdot_in:.6g}), dP={dP:.4g} Pa")


def test_incompressible_valve_cavitation():
    """Exercise the three-regime ISA-75.01 cavitation gate added to
    incompressible Valve / CheckValve.

    Covers: silent path (F_L=None), flashing, choked cavitating,
    incipient warning, and round-trip preservation when F_L is set in the
    non-cavitating regime.
    """
    import warnings as _warnings

    print("\ntest_incompressible_valve_cavitation")

    Pv = 10.0e3     # 10 kPa vapor pressure (water-like at ambient)
    fluid = Incompressible_Fluid(
        density=1000.0, viscosity=1e-3, vapor_pressure=Pv,
    )

    ID = ureg.Quantity(2.0, "inch")

    # --- Silent path: F_L is None, no warning or error fires even at
    # extreme cavitation conditions.
    valve_silent = Incompressible_Valve(Di=ID, K=10.0, name="silent")
    mdot_pq      = ureg.Quantity(40.0, "kg/s")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        valve_silent.dP(fluid, mdot_pq, P_inlet=2.0e5)
        valve_silent.dmdot(fluid, P_inlet=2.0e5, P_outlet=1.5e5)
    cav_warnings = [w for w in caught if "cavitation" in str(w.message).lower()]
    assert not cav_warnings, (
        f"silent path leaked a cavitation warning: {[str(w.message) for w in cav_warnings]}"
    )
    print(f"  silent (F_L=None)        : no warnings/errors (got "
          f"{len(caught)} unrelated warnings)")

    # --- Regime 1: flashing.  P_outlet below P_v -> hard error.
    valve = Incompressible_Valve(Di=ID, K=10.0, F_L=0.9, name="flashing-test")
    try:
        valve.dmdot(fluid, P_inlet=1.0e5, P_outlet=5.0e3)
    except RuntimeError as exc:
        assert "flashing" in str(exc).lower(), f"wrong message: {exc}"
        print(f"  regime 1 (flashing)      : RuntimeError raised -- ok")
    else:
        raise AssertionError("flashing regime did not raise RuntimeError")

    # --- Regime 2: choked cavitating flow.
    # F_L=0.7, P_in=2 bar, Pv=10 kPa, F_F~=0.96 (no Pc on fluid)
    # dP_choked = 0.49 * (2e5 - 0.96*1e4) = 0.49 * 1.904e5 = 9.33e4 Pa
    # Choose P_outlet that drives |dP| just over 9.33e4 Pa but still
    # leaves P_outlet > Pv (so it's choked-cavitating, not flashing).
    valve_choked = Incompressible_Valve(Di=ID, K=10.0, F_L=0.7, name="choked-test")
    try:
        # |dP| = 1.0e5 Pa > 9.33e4 Pa, P_outlet = 1.0e5 Pa > Pv = 1.0e4 Pa.
        valve_choked.dmdot(fluid, P_inlet=2.0e5, P_outlet=1.0e5)
    except RuntimeError as exc:
        assert "choked" in str(exc).lower(), f"wrong message: {exc}"
        print(f"  regime 2 (choked cav.)   : RuntimeError raised -- ok")
    else:
        raise AssertionError("choked-cavitating regime did not raise RuntimeError")

    # --- Regime 3: incipient cavitation warning.
    # The incipient-but-not-choked window is
    #   F_L^2 * (P_in - Pv)  <  |dP|  <  F_L^2 * (P_in - F_F*Pv)
    # i.e. width F_L^2 * (1-F_F) * Pv.  For small Pv/P_in the window
    # collapses to a few hundred Pa, so we use a fluid where Pv is a
    # substantial fraction of P_in (a near-saturation high-vapor-pressure
    # liquid) to get a workable test margin.
    Pv_high = 5.0e5
    fluid_hot = Incompressible_Fluid(
        density=900.0, viscosity=5e-4, vapor_pressure=Pv_high,
    )
    valve_incip = Incompressible_Valve(Di=ID, K=10.0, F_L=0.95, name="incip-test")
    # F_L=0.95 -> F_L^2 = 0.9025.  P_in=1e6, Pv=5e5, F_F=0.96.
    # dP_incip  = 0.9025 * (1e6 - 5e5)      = 4.51e5 Pa
    # dP_choked = 0.9025 * (1e6 - 0.96*5e5) = 4.69e5 Pa
    # |dP|=4.60e5 -> sigma = 0.5e6/4.6e5 = 1.087 < 1/F_L^2 = 1.108 (incipient)
    # P_outlet  = 1e6 - 4.6e5 = 5.4e5 > Pv = 5e5 (no flashing).
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        valve_incip.dmdot(fluid_hot, P_inlet=1.0e6, P_outlet=5.40e5)
    incip = [w for w in caught
             if "incipient cavitation" in str(w.message).lower()]
    assert incip, f"no incipient warning; caught: {[str(w.message) for w in caught]}"
    print(f"  regime 3 (incipient)     : UserWarning raised -- ok")

    # --- CheckValve mirrors Valve: flashing path.
    cv = Incompressible_CheckValve(Di=ID, K=20.0, F_L=0.9, name="cv-flash")
    try:
        cv.dmdot(fluid, P_inlet=1.0e5, P_outlet=5.0e3)
    except RuntimeError as exc:
        assert "flashing" in str(exc).lower()
        print(f"  CheckValve flashing      : RuntimeError raised -- ok")
    else:
        raise AssertionError("CheckValve flashing did not raise")

    # --- Round-trip preservation: in the non-cavitating regime, F_L set
    # does not change the math, only adds the gate.
    valve_safe = Incompressible_Valve(Di=ID, K=10.0, F_L=0.9, name="safe")
    P_in = 2.0e6    # 20 bar -- gives headroom above the choked threshold
                    # (F_L^2*(P_in-F_F*Pv) ~= 1.6 MPa) for |dP| ~ 0.5 MPa.
    dP   = valve_safe.dP(fluid, ureg.Quantity(20.0, "kg/s"), P_inlet=P_in)
    mdot = valve_safe.dmdot(fluid, P_inlet=P_in, P_outlet=P_in + dP)
    err  = abs(mdot - 20.0) / 20.0
    assert err < 1e-10, f"round-trip drift with F_L set: {err:.2e}"
    print(f"  round-trip with F_L=0.9  : rel err {err:.2e} -- ok")


def test_incompressible_orifice_cavitation():
    """Verify Orifice.dP fires the cavitation gate on both forward
    (dP) and inverse (dmdot) paths.

    Regression guard for the prior sign bug: Orifice.dP used to gate the
    cavitation check on `if dP_perm > 0.0`, which never fired since
    dP_perm is always <= 0 for forward flow.  This test exercises a
    parameter combination known to trip both the choked and the incipient
    thresholds, and asserts the right outcome on both `dP` and `dmdot`.
    """
    import warnings as _warnings

    print("\ntest_incompressible_orifice_cavitation")

    # Water-like at 1 bar / 65 C (Pv ~ 25 kPa) flowing through a small
    # restriction at high velocity.
    Pv    = 25.0e3
    fluid = Incompressible_Fluid(
        density=1000.0, viscosity=1e-3, vapor_pressure=Pv,
    )
    orif = Incompressible_Orifice(
        Di=ureg.Quantity(2.0, "inch"),
        Do=ureg.Quantity(0.5, "inch"),
        name="cav-test",
    )

    # --- Forward path: a high mdot at modest inlet pressure produces a
    # |dP| large enough to push sigma below sigma_choked.
    P_in_choked = 1.0e5     # 1 bar absolute
    mdot_high   = ureg.Quantity(15.0, "kg/s")
    try:
        orif.dP(fluid, mdot_high, P_inlet=P_in_choked)
    except RuntimeError as exc:
        assert "choked cavitation" in str(exc).lower(), f"wrong message: {exc}"
        print(f"  dP   forward choked     : RuntimeError raised -- ok")
    else:
        raise AssertionError(
            "Orifice.dP did not raise RuntimeError on choked cavitation"
        )

    # --- Inverse path: same regime via dmdot.  Use a P_outlet that
    # drives |dP| above the choked threshold.  At P_in=1 bar, Pv=25 kPa,
    # a near-zero outlet (vacuum) gives sigma = (1e5-2.5e4)/9.9e4 ~ 0.76,
    # well below typical sigma_choked ~ 2.5.
    try:
        orif.dmdot(fluid, P_inlet=P_in_choked, P_outlet=2.6e4)
    except RuntimeError as exc:
        assert "choked cavitation" in str(exc).lower(), f"wrong message: {exc}"
        print(f"  dmdot inverse choked    : RuntimeError raised -- ok")
    else:
        raise AssertionError(
            "Orifice.dmdot did not raise RuntimeError on choked cavitation"
        )

    # --- Silent path: high inlet pressure keeps sigma well above all
    # thresholds.  dP must not raise or warn about cavitation.
    P_in_safe = 5.0e6     # 50 bar -- huge sigma headroom
    mdot_low  = ureg.Quantity(1.0, "kg/s")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        dP = orif.dP(fluid, mdot_low, P_inlet=P_in_safe)
    cav = [w for w in caught if "cavitation" in str(w.message).lower()]
    assert not cav, (
        f"silent path leaked cavitation warning: "
        f"{[str(w.message) for w in cav]}"
    )
    print(f"  dP   safe regime        : no cavitation warning (dP={dP:.3g} Pa)")


# --- Choke / integrator debugging tests (migrated from test.py) ---


def test_choked_mass_flux_ideal_gas_air():
    """Ideal-gas air nozzle sanity check for choked_mass_flux.

    For an ideal gas with constant gamma = 1.4, classical isentropic-nozzle
    theory gives the critical pressure ratio P*/P0 = (2/(gamma+1))^(gamma/
    (gamma-1)) approx 0.5283 and the critical temperature ratio T*/T0 =
    2/(gamma+1) approx 0.8333.  The mass-flux coefficient is
        G_max = P0 * sqrt(gamma/(R_s*T0)) * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
    With P0 = 10 bar, T0 = 300 K, and A_throat = 1 cm^2 = 1e-4 m^2, dry air
    (M = 28.96 g/mol => R_s = 287.0 J/(kg.K)) gives mdot_choked ~ 0.2354 kg/s.

    CoolProp's HEOS-air pseudo-pure backend is close to but not exactly
    ideal-gas (slight Z deviation, slight gamma deviation), so we allow
    a 1% tolerance on each quantity.
    """
    from compressible_flow import (
        choked_mass_flux, _build_phase_limits, _safe_update_PT, FlowState,
    )

    P0 = ureg.Quantity(10.0, "bar").to("Pa").magnitude   # 1.0e6 Pa
    T0 = 300.0                                           # K
    A_throat = 1.0e-4                                    # m^2

    AS = AbstractState("HEOS", "Air")
    phase_limits = _build_phase_limits(AS)
    _safe_update_PT(AS, P0, T0, *phase_limits)

    # choked_mass_flux's new convention: fs.AS at the static inlet and
    # fs.mdot used to build the stagnation enthalpy reference internally.
    # We set fs.mdot to a small probe value so v_in approx 0 -- i.e. the
    # inlet is effectively stagnation, matching the textbook reference.
    fs = FlowState(
        AS, mdot=1e-12, A=A_throat, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2],       P_critical=phase_limits[3],
    )
    mdot_choked, P_star, T_star, rho_star, P_out, T_out = choked_mass_flux(
        fs, A_throat, A_outlet=A_throat,
    )

    # Closed-form ideal-gas reference values for gamma = 1.4, M_air = 28.96 g/mol.
    gamma = 1.4
    M_air = 28.96e-3
    R_s   = 8.31446261815324 / M_air
    P_ratio_ref = (2.0/(gamma+1.0)) ** (gamma/(gamma-1.0))             # ~0.5283
    T_ratio_ref = 2.0/(gamma+1.0)                                       # ~0.8333
    crit_coeff  = (2.0/(gamma+1.0)) ** ((gamma+1.0)/(2.0*(gamma-1.0)))
    G_ref       = P0 * math.sqrt(gamma/(R_s*T0)) * crit_coeff
    mdot_ref    = G_ref * A_throat

    def check(label, value, ref, tol_rel):
        err = abs(value - ref) / abs(ref)
        status = "OK  " if err < tol_rel else "FAIL"
        print(f"  [{status}] {label}: got {value:.6g}, ref {ref:.6g}, "
              f"rel err {err:.2%} (tol {tol_rel:.0%})")

    print("Ideal-gas air-nozzle choke validation (P0=10 bar, T0=300 K, A=1 cm^2):")
    check("P*/P0",        P_star/P0,  P_ratio_ref, 0.01)
    check("T*/T0",        T_star/T0,  T_ratio_ref, 0.01)
    check("mdot_choked",  mdot_choked, mdot_ref,   0.02)
    print(f"  (raw: P*={P_star:.4g} Pa, T*={T_star:.4g} K, "
          f"rho*={rho_star:.4g} kg/m^3, mdot={mdot_choked:.4g} kg/s)")


def test_compressible_K_choke_roundtrip():
    """Validate the FlowState-aware stagnation reference inside choked_mass_flux.

    The cross-cutting bug fix that introducing FlowState delivered: before,
    choked_mass_flux read AS.hmass() directly and labelled it h0,
    silently treating the static inlet as stagnation.  Now it reads
    fs.h_stagnation = h_static + 0.5*v_in**2, which correctly grows the
    stagnation reference when the inlet carries non-trivial kinetic
    energy.  Consequence: for a fixed inlet (P, T, A), the choked mass
    flow grows monotonically with v_in (more accessible enthalpy in the
    expansion).

    Test: at the same inlet (P_in, T_in) and area A, evaluate
    choked_mass_flux at two FlowStates -- one with v_in approx 0 (the
    classical stagnation case) and one with v_in approx half the local
    speed of sound -- and verify mdot_choked rises by a few percent in
    the latter.  Under the old (buggy) code these two would have
    returned the same value.

    Also verify the lower-Ma case round-trips against the textbook
    stagnation formula to within 1%.
    """
    from compressible_flow import (
        choked_mass_flux, _build_phase_limits, _safe_update_PT, FlowState,
    )

    P_in = ureg.Quantity(20.0, "bar").to("Pa").magnitude
    T_in = 300.0
    Di   = ureg.Quantity(0.5, "inch").to("m").magnitude
    A    = math.pi * Di**2 / 4.0

    AS = composition.define_composition(
        y_Methane=0.95, y_Ethane=0.04, y_CarbonDioxide=0.01, eos="HEOS",
    )
    phase_limits = _build_phase_limits(AS)

    def _mdot_choked_at(v_in_target):
        """Build a FlowState whose v_in is approximately v_in_target and
        return the choked mass flow for that inlet state."""
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        rho_in_seed = AS.rhomass()
        mdot_seed = max(v_in_target * rho_in_seed * A, 1e-12)
        fs = FlowState(
            AS, mdot=mdot_seed, A=A, z=0.0,
            T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
            T_critical=phase_limits[2],       P_critical=phase_limits[3],
        )
        mdot_ch, *_ = choked_mass_flux(fs, A, A_outlet=A)
        return mdot_ch, fs.Ma

    # Probe at v_in ~ 0 (essentially stagnation inlet).
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    a_in = AS.speed_sound()
    mdot_stag, Ma_stag = _mdot_choked_at(0.0)

    # Probe at v_in ~ 0.5 * a_in.  With the FlowState-aware h0 the choked
    # mass flow should rise by a few percent vs the stagnation reference.
    mdot_hot, Ma_hot = _mdot_choked_at(0.5 * a_in)

    rel_lift = (mdot_hot - mdot_stag) / mdot_stag

    print("FlowState-aware choked_mass_flux (methane-rich mixture, P_in=20 bar, T_in=300 K):")
    print(f"  v_in ~ 0    (Ma_in={Ma_stag:.3f}):  mdot_choked = {mdot_stag:.6g} kg/s")
    print(f"  v_in ~ a/2  (Ma_in={Ma_hot:.3f}):  mdot_choked = {mdot_hot:.6g} kg/s")
    print(f"  relative lift from v_in**2/2 in h0:  {rel_lift:+.2%}")

    status = "OK  " if 0.0 < rel_lift < 0.25 else "FAIL"
    print(f"  [{status}] mdot_choked rises with v_in (expected: 0 < lift < ~25%)")


def test_valve_minimum_diameter_choke():
    """Verify Valve.minimum_diameter sharpens internal-throat choke detection.

    Sets up a 2" pipe carrying a methane-rich mixture at 20 bar / 300 K, a
    globe-valve-like K=5, and an internal throat half the pipe diameter
    (A_throat = A_pipe/4).  Picks mdot in the band where:
        mdot_choked_at_pipe_area  > mdot  > mdot_choked_at_throat_area
    so the pipe-area screen (Valve without minimum_diameter)
    silently passes, but the throat-area screen (Valve with
    minimum_diameter) raises ChokedFlowError.  This is the gap that
    improvements.md R1.5 / R2 flagged and that minimum_diameter closes.
    """
    from compressible_flow import (
        Valve, FlowState, ChokedFlowError, choked_mass_flux,
        _build_phase_limits, _safe_update_PT,
    )

    P_in = ureg.Quantity(20.0, "bar").to("Pa").magnitude
    T_in = 300.0
    D_pipe = ureg.Quantity(2.0, "inch").to("m").magnitude
    D_min  = ureg.Quantity(1.0, "inch").to("m").magnitude
    A_pipe   = math.pi * D_pipe ** 2 / 4.0
    A_throat = math.pi * D_min  ** 2 / 4.0

    AS = composition.define_composition(
        y_Methane=0.95, y_Ethane=0.04, y_CarbonDioxide=0.01, eos="HEOS",
    )
    phase_limits = _build_phase_limits(AS)

    def fresh_fs(mdot):
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        return FlowState(
            AS, mdot=mdot, A=A_pipe, z=0.0,
            T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
            T_critical=phase_limits[2],       P_critical=phase_limits[3],
        )

    # Reference choked mass flows at the two candidate areas (using a
    # near-stagnation inlet for both -- the test mdot is set well below
    # mdot_choked_pipe and well above mdot_choked_throat so the small
    # KE-in-stagnation correction cannot flip either comparison).
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    fs_probe = FlowState(
        AS, mdot=1e-6, A=A_pipe, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2],       P_critical=phase_limits[3],
    )
    mdot_pipe,   *_ = choked_mass_flux(fs_probe, A_pipe,   A_outlet=A_pipe)
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    fs_probe = FlowState(
        AS, mdot=1e-6, A=A_pipe, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2],       P_critical=phase_limits[3],
    )
    mdot_throat, *_ = choked_mass_flux(fs_probe, A_throat, A_outlet=A_pipe)

    # Pick mdot squarely in the band [mdot_throat, mdot_pipe].
    mdot_test = 0.5 * (mdot_pipe + mdot_throat)

    print("Valve internal-throat choke detection (methane mix, 20 bar, 300 K):")
    print(f"  D_pipe = 2 in,  D_min = 1 in  (A_throat = A_pipe/4)")
    print(f"  mdot_choked at A_pipe   = {mdot_pipe:.4f} kg/s  (legacy screen)")
    print(f"  mdot_choked at A_throat = {mdot_throat:.4f} kg/s  (new screen)")
    print(f"  test mdot               = {mdot_test:.4f} kg/s  (between the two)")

    # 1) Valve with minimum_diameter: should raise ChokedFlowError.
    v_throat = Valve(Di=D_pipe, K=20.0, minimum_diameter=D_min)
    fs = fresh_fs(mdot_test)
    new_raised = False
    try:
        v_throat.dP_dT(fs)
    except ChokedFlowError as e:
        new_raised = True
        mdot_clamp = e.mdot_choked

    # 2) Valve without minimum_diameter: legacy path, should NOT raise.
    v_legacy = Valve(Di=D_pipe, K=5.0)
    fs2 = fresh_fs(mdot_test)
    legacy_raised = False
    try:
        v_legacy.dP_dT(fs2)
    except ChokedFlowError:
        legacy_raised = True

    print(f"  Valve(minimum_diameter=1in):  ChokedFlowError raised? {new_raised}"
          + (f"  (mdot_choked={mdot_clamp:.4f} kg/s)" if new_raised else ""))
    print(f"  Valve(no minimum_diameter):   ChokedFlowError raised? {legacy_raised}")

    status = "OK  " if (new_raised and not legacy_raised) else "FAIL"
    print(f"  [{status}] minimum_diameter catches internal choke that the "
          f"pipe-area screen misses")


def test_line_segment_choke_diagnostic():
    """Verify the predictive Fanno / isothermal choke diagnostic on
    Line_Segment.dP_dT (improvements.md R7).

    Sets up a methane-rich mixture at P_in = 5 bar, T_in = 300 K in 1"
    Sch 40 pipe with mdot chosen so inlet Ma ~ 0.2.  Under ideal-gas
    Fanno theory L_max ~ 20 m for these inputs, so a 200 m segment
    should trip the predictive warning while a 5 m segment should not.
    The isothermal branch is checked separately at the same long
    length.
    """
    import warnings as _warnings
    from compressible_flow import (
        Line_Segment, FlowState, _build_phase_limits, _safe_update_PT,
    )

    P_in = ureg.Quantity(5.0, "bar").to("Pa").magnitude
    T_in = 300.0
    OD = ureg.Quantity(1.315, "inch").to("m").magnitude
    WT = ureg.Quantity(0.133, "inch").to("m").magnitude
    ID = OD - 2 * WT
    A_pipe = math.pi * ID ** 2 / 4.0
    roughness = ureg.Quantity(0.00015, "ft").to("m").magnitude

    AS = composition.define_composition(
        y_Methane=0.95, y_Ethane=0.04, y_CarbonDioxide=0.01, eos="HEOS",
    )
    phase_limits = _build_phase_limits(AS)

    # Pick mdot so inlet Ma ~ 0.2 at the static (P_in, T_in).
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    rho_in = AS.rhomass()
    a_in   = AS.speed_sound()
    Ma_target = 0.2
    mdot = Ma_target * a_in * rho_in * A_pipe

    def fresh_fs():
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        return FlowState(
            AS, mdot=mdot, A=A_pipe, z=0.0,
            T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
            T_critical=phase_limits[2],       P_critical=phase_limits[3],
        )

    def run(length_m, isothermal):
        seg = Line_Segment(
            roughness=ureg.Quantity(roughness, "m"),
            id_val=ureg.Quantity(ID, "m"),
            length=ureg.Quantity(length_m, "m"),
            elevation_change=ureg.Quantity(0.0, "m"),
        )
        fs = fresh_fs()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            try:
                seg.dP_dT(fs, isothermal=isothermal)
            except RuntimeError:
                # Reactive choke in compressible_pipe_segment is still
                # possible at very long lengths; the diagnostic should
                # have fired BEFORE the RuntimeError.
                pass
        choke_warnings = [w for w in caught
                          if issubclass(w.category, UserWarning)
                          and "chok" in str(w.message).lower()]
        return choke_warnings

    print(f"Line_Segment predictive choke diagnostic "
          f"(methane mix, P_in=5 bar, T_in=300 K, ID=1\" Sch 40, Ma_in~{Ma_target}):")
    print(f"  mdot = {mdot:.4f} kg/s")

    # 1) Adiabatic long pipe -- should warn.
    warns_long_adi = run(200.0, isothermal=False)
    long_adi_ok = len(warns_long_adi) >= 1 and "fanno" in str(warns_long_adi[0].message).lower()
    print(f"  adiabatic, L=200 m:  warnings={len(warns_long_adi)}  "
          f"[{'OK  ' if long_adi_ok else 'FAIL'}] expect Fanno-choke UserWarning")
    if warns_long_adi:
        print(f"    -> {warns_long_adi[0].message}")

    # 2) Adiabatic short pipe -- should NOT warn.
    warns_short_adi = run(5.0, isothermal=False)
    short_adi_ok = len(warns_short_adi) == 0
    print(f"  adiabatic, L=5 m:    warnings={len(warns_short_adi)}  "
          f"[{'OK  ' if short_adi_ok else 'FAIL'}] expect no choke UserWarning")

    # 3) Isothermal long pipe -- should warn (different branch / text).
    warns_long_iso = run(200.0, isothermal=True)
    long_iso_ok = len(warns_long_iso) >= 1 and "isothermal" in str(warns_long_iso[0].message).lower()
    print(f"  isothermal, L=200 m: warnings={len(warns_long_iso)}  "
          f"[{'OK  ' if long_iso_ok else 'FAIL'}] expect isothermal-choke UserWarning")
    if warns_long_iso:
        print(f"    -> {warns_long_iso[0].message}")

    overall = long_adi_ok and short_adi_ok and long_iso_ok
    print(f"  [{'OK  ' if overall else 'FAIL'}] R7 predictive diagnostic behaves as expected")


def test_pipe_segment_convergence_order():
    """Verify the Heun (trapezoidal) predictor-corrector in
    compressible_pipe_segment integrates at better than first order.

    Methane-rich mixture at 10 bar / 300 K in 1" Sch 40 pipe, mdot chosen
    so the run develops a steep gradient (~10-15% pressure drop over the
    test length).  The slice is integrated as 1, 2, and 4 equal sub-steps
    with the adaptive splitter and the correction-skip disabled, and
    compared against a 512-sub-step reference.

    Pass criteria (chosen to be robust against the reference's own noise
    floor rather than asserting an exact error ratio of 4):
      - the single-step error is < 0.1% of the total pressure drop
        (forward Euler fails this by an order of magnitude), and
      - 4 sub-steps reduce the error by at least 10x vs 1 sub-step
        (first order would give exactly 4x).
    """
    from compressible_flow import (
        FlowState, _build_phase_limits, _safe_update_PT,
        compressible_pipe_segment,
    )

    P_in = ureg.Quantity(10.0, "bar").to("Pa").magnitude
    T_in = 300.0
    OD = ureg.Quantity(1.315, "inch").to("m").magnitude
    WT = ureg.Quantity(0.133, "inch").to("m").magnitude
    ID = OD - 2 * WT
    A_pipe = math.pi * ID ** 2 / 4.0
    roughness = ureg.Quantity(0.00015, "ft").to("m").magnitude

    AS = composition.define_composition(
        y_Methane=0.95, y_Ethane=0.04, y_CarbonDioxide=0.01, eos="HEOS",
    )
    phase_limits = _build_phase_limits(AS)

    # mdot for inlet Ma ~ 0.15; over dL_total the flow then develops a
    # ~10-15% pressure drop -- steep enough that truncation error is well
    # above the EOS noise floor.
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot = 0.15 * AS.speed_sound() * AS.rhomass() * A_pipe
    dL_total = 8.0   # m

    huge = 1e18  # disables the adaptive splitter so step size is controlled

    def run(n_steps):
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        fs = FlowState(
            AS, mdot=mdot, A=A_pipe, z=0.0,
            T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
            T_critical=phase_limits[2],       P_critical=phase_limits[3],
        )
        for _ in range(n_steps):
            compressible_pipe_segment(
                fs, dL=dL_total / n_steps, dz=0.0, D_h=ID,
                roughness=roughness,
                energy_tol=huge, dPdL_rel_tol=huge, Ma_change_tol=huge,
                correction_skip_rel_tol=0.0,
            )
        return fs.P, fs.T

    P_ref, T_ref = run(512)
    dP_total = P_in - P_ref
    print(f"Heun convergence order (methane mix, 10 bar / 300 K, "
          f"1\" Sch 40, mdot={mdot:.4f} kg/s, dL={dL_total} m):")
    print(f"  reference (512 steps): P_out={P_ref:.2f} Pa "
          f"(dP={dP_total:.0f} Pa, {100*dP_total/P_in:.1f}% of inlet), "
          f"T_out={T_ref:.4f} K")

    errs = {}
    for n in (1, 2, 4):
        P_n, T_n = run(n)
        errs[n] = abs(P_n - P_ref)
        print(f"  n={n}: |P error| = {errs[n]:10.4g} Pa "
              f"({errs[n]/dP_total*100:.4f}% of dP), "
              f"|T error| = {abs(T_n - T_ref):.3g} K")

    single_step_ok = errs[1] < 1e-3 * dP_total
    reduction_ok   = errs[4] < errs[1] / 10.0
    print(f"  [{'OK  ' if single_step_ok else 'FAIL'}] single-step error "
          f"< 0.1% of total dP (got {errs[1]/dP_total*100:.4f}%)")
    print(f"  [{'OK  ' if reduction_ok else 'FAIL'}] error reduction 1->4 "
          f"steps >= 10x (got {errs[1]/max(errs[4], 1e-300):.1f}x; "
          f"first order would give 4x)")


def test_isothermal_choke_gate():
    """Verify the isothermal Mach gate in compressible_pipe_segment.

    Isothermal pipe flow goes singular at the isothermal sound speed
    a_T = sqrt((dP/drho)_T) = a/sqrt(gamma) for an ideal gas, *below* the
    isentropic sound speed.  The original gate tested v/a >= 0.98 in both
    modes, so in isothermal mode it could never fire -- the dP/dL
    singularity (at v/a ~ 0.88 for gamma ~ 1.3) was hit first and
    surfaced as a misleading "failed to converge after 8 splits".

    Three cases on a methane-rich mixture at 10 bar / 300 K, 1" Sch 40:
      1. Inlet Ma_T = 1.02 (isentropic Ma ~ 0.9, which the old gate
         passed): expect an immediate RuntimeError naming the isothermal
         Mach number.
      2. Inlet Ma_T = 0.85 with a long slice (singularity mid-pipe):
         expect a RuntimeError whose message names the isothermal choke.
      3. Mild flow: integrates normally.
    """
    from compressible_flow import (
        FlowState, _build_phase_limits, _safe_update_PT,
        compressible_pipe_segment,
    )

    P_in = ureg.Quantity(10.0, "bar").to("Pa").magnitude
    T_in = 300.0
    OD = ureg.Quantity(1.315, "inch").to("m").magnitude
    WT = ureg.Quantity(0.133, "inch").to("m").magnitude
    ID = OD - 2 * WT
    A_pipe = math.pi * ID ** 2 / 4.0
    roughness = ureg.Quantity(0.00015, "ft").to("m").magnitude

    AS = composition.define_composition(
        y_Methane=0.95, y_Ethane=0.04, y_CarbonDioxide=0.01, eos="HEOS",
    )
    phase_limits = _build_phase_limits(AS)
    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    rho_in = AS.rhomass()
    a_in   = AS.speed_sound()
    a_T    = 1.0 / math.sqrt(AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT))

    def run(Ma_T_inlet, dL):
        mdot = Ma_T_inlet * a_T * rho_in * A_pipe
        _safe_update_PT(AS, P_in, T_in, *phase_limits)
        fs = FlowState(
            AS, mdot=mdot, A=A_pipe, z=0.0,
            T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
            T_critical=phase_limits[2],       P_critical=phase_limits[3],
        )
        compressible_pipe_segment(
            fs, dL=dL, dz=0.0, D_h=ID, roughness=roughness, isothermal=True,
        )
        return fs

    print(f"Isothermal choke gate (methane mix, 10 bar / 300 K, 1\" Sch 40): "
          f"a={a_in:.1f} m/s, a_T={a_T:.1f} m/s (ratio {a_in/a_T:.3f})")

    # 1) Inlet gate: Ma_T = 1.02 but isentropic Ma = 1.02*a_T/a < 0.98.
    try:
        run(1.02, dL=5.0)
        gate_ok, gate_msg = False, "no error raised"
    except RuntimeError as exc:
        gate_msg = str(exc)
        gate_ok = "isothermal mach" in gate_msg.lower() and "inlet" in gate_msg.lower()
    print(f"  [{'OK  ' if gate_ok else 'FAIL'}] inlet Ma_T=1.02 "
          f"(isentropic Ma={1.02*a_T/a_in:.3f}, old gate silent) raises "
          f"isothermal-Mach RuntimeError")
    print(f"    -> {gate_msg[:120]}")

    # 2) Singularity mid-pipe: choke origin should be named in the error.
    try:
        run(0.85, dL=30.0)
        mid_ok, mid_msg = False, "no error raised"
    except RuntimeError as exc:
        mid_msg = str(exc)
        mid_ok = "isothermal" in mid_msg.lower()
    print(f"  [{'OK  ' if mid_ok else 'FAIL'}] inlet Ma_T=0.85, L=30 m "
          f"(chokes mid-pipe) raises an error naming the isothermal choke")
    print(f"    -> ...{mid_msg[-160:]}")

    # 3) Mild flow integrates normally.
    fs = run(0.10, dL=30.0)
    mild_ok = fs.P < P_in and fs.T == T_in
    print(f"  [{'OK  ' if mild_ok else 'FAIL'}] mild flow (Ma_T=0.10) "
          f"integrates: P_out={fs.P:.0f} Pa, T_out={fs.T:.1f} K")


# --- dmdot_dT vs dP_dT performance benchmark (migrated from benchmark_dmdot.py) ---
#
# Benchmark the cost of dmdot_dT vs dP_dT at three levels per component:
#   1. Forward dP_dT call (baseline cost of one walk).
#   2. Component-level dmdot_dT call (brentq + forward closure).
#   3. Single-edge network solve with auto-detected inverse mode
#      (`Compressible_Network.solve` calling `walk_edge_inverse`).
# For each level we record wall-clock time and the number of times the
# component's dP_dT was actually invoked (instrumented via a monkeypatch).
# Entry point: benchmark_dmdot_dT().


def _build_AS():
    csv_path = os.path.join(os.path.dirname(__file__), "example_composition.csv")
    return composition.define_composition_from_csv(csv_path)


def _make_fs(AS, P, T, mdot, A, phase_limits):
    _safe_update_PT(AS, P, T, *phase_limits)
    return FlowState(
        AS, mdot, A=A, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2], P_critical=phase_limits[3],
    )


class _DPCounter:
    """Monkeypatch wrapper that counts dP_dT calls on a component instance."""
    def __init__(self, component):
        self.component = component
        self.calls = 0
        self._original = component.dP_dT
        component.dP_dT = self._wrapped

    def _wrapped(self, fs, *args, **kwargs):
        self.calls += 1
        return self._original(fs, *args, **kwargs)

    def restore(self):
        self.component.dP_dT = self._original


def benchmark_component(name, component, AS, P_in, T_in, mdot_target,
                        A_pipe, phase_limits, P_target,
                        component_class, di_for_node=None,
                        dmdot_dT_kwargs=None):
    """Run the three timing levels for one component type."""
    dmdot_dT_kwargs = dmdot_dT_kwargs or {}

    # Level 1: forward dP_dT.
    counter = _DPCounter(component)
    counter.calls = 0
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    t0 = time.perf_counter()
    component.dP_dT(fs, **dmdot_dT_kwargs)
    dt_fwd = time.perf_counter() - t0
    fwd_calls = counter.calls
    counter.restore()

    # Level 2: component-level dmdot_dT (brentq inside).
    counter = _DPCounter(component)
    counter.calls = 0
    fs = _make_fs(AS, P_in, T_in, 0.0, A_pipe, phase_limits)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        component.dmdot_dT(fs, P2=P_target, **dmdot_dT_kwargs)
    dt_dmdot = time.perf_counter() - t0
    dmdot_calls = counter.calls
    dmdot_mdot = fs.mdot
    counter.restore()

    # Level 3: 1-edge Compressible_Network.solve with inverse mode
    # (auto-detected because both endpoints are P-spec).
    net = Compressible_Network()
    # If the component carries a non-trivial diameter, use it as the
    # node area too (saves the area-mismatch _area_match call).
    if di_for_node is not None:
        net.add_node("inlet",  P=P_in, T=T_in, diameter=di_for_node)
        net.add_node("outlet", P=P_target,    diameter=di_for_node)
    else:
        net.add_node("inlet",  P=P_in, T=T_in)
        net.add_node("outlet", P=P_target)
    net.add_edge("edge", "inlet", "outlet", component)

    counter = _DPCounter(component)
    counter.calls = 0
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = net.solve(AS, mdot_init_kgs=max(mdot_target * 0.5, 0.1))
    dt_net = time.perf_counter() - t0
    net_calls = counter.calls
    net_mdot = result["mdot_kgs"]["edge"]
    net_converged = result["converged"]
    counter.restore()

    # Per-call cost: time per dP_dT invocation, for cross-component comparison.
    dt_per_fwd = (dt_fwd / fwd_calls) if fwd_calls else float("nan")

    return {
        "name": name,
        "dt_fwd_s":   dt_fwd,
        "dt_dmdot_s": dt_dmdot,
        "dt_net_s":   dt_net,
        "fwd_calls": fwd_calls,
        "dmdot_calls": dmdot_calls,
        "net_calls":   net_calls,
        "dt_per_fwd_s": dt_per_fwd,
        "dmdot_mdot":  dmdot_mdot,
        "net_mdot":    net_mdot,
        "mdot_target": mdot_target,
        "net_converged": net_converged,
    }


def benchmark_dmdot_dT():
    AS = _build_AS()
    phase_limits = _build_phase_limits(AS)

    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    ID    = ureg.Quantity(1.939, "inch")
    ID_si = ID.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4.0
    D_trim = ureg.Quantity(0.25, "inch")

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot_target = _resolve_mdot(ureg.Quantity(0.07, "mmscf/day"), AS)

    # For each component, first run forward to get the target P2, then
    # benchmark all three levels with that P2 as the target.
    cases = []

    # 1. Constricted Valve (forwards to compressible_dA Mode 2)
    v_throat = Compressible_Valve(Di=ID, Cv=1.76, minimum_diameter=D_trim)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    v_throat.dP_dT(fs)
    P2 = fs.P
    cases.append(("Valve constricted", v_throat, P2, ID, None))

    # 2. Plain Valve (drives compressible_changing_area_K via helper)
    v_plain = Compressible_Valve(Di=ID, K=10.0)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    v_plain.dP_dT(fs)
    P2 = fs.P
    cases.append(("Valve plain", v_plain, P2, ID, None))

    # 3. Orifice (compressible_dA + Cd<->mdot fixed point)
    orf = Compressible_Orifice(Di=ID, Do=D_trim)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    orf.dP_dT(fs)
    P2 = fs.P
    cases.append(("Orifice", orf, P2, ID, None))

    # 4. Bend (compressible_K + K<->mdot fixed point)
    bend = Compressible_Bend(Di=ID, ang_deg=90.0, bend_dias=3.0)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    bend.dP_dT(fs)
    P2 = fs.P
    cases.append(("Bend", bend, P2, ID, None))

    # 5. Line_Segment 200 ft 2" (slice loop inside)
    seg = Compressible_Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ID,
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="seg",
    )
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    seg.dP_dT(fs)
    P2 = fs.P
    cases.append(("Line_Segment 200ft", seg, P2, ID, None))

    rows = []
    for name, component, P2, di, kwargs in cases:
        print(f"\nBenchmarking {name} ...", flush=True)
        row = benchmark_component(
            name=name,
            component=component,
            AS=AS, P_in=P_in, T_in=T_in,
            mdot_target=mdot_target,
            A_pipe=A_pipe,
            phase_limits=phase_limits,
            P_target=P2,
            component_class=type(component),
            di_for_node=di,
        )
        rows.append(row)
        print(
            f"  fwd dP_dT      : {row['dt_fwd_s']*1000:8.2f} ms "
            f"({row['fwd_calls']:3d} dP_dT call{'s' if row['fwd_calls']!=1 else ''})",
            flush=True,
        )
        print(
            f"  dmdot_dT       : {row['dt_dmdot_s']*1000:8.2f} ms "
            f"({row['dmdot_calls']:3d} dP_dT calls) "
            f"-> mdot {row['dmdot_mdot']:.6g}  (target {row['mdot_target']:.6g})",
            flush=True,
        )
        print(
            f"  network solve  : {row['dt_net_s']*1000:8.2f} ms "
            f"({row['net_calls']:3d} dP_dT calls) "
            f"-> mdot {row['net_mdot']:.6g}  converged={row['net_converged']}",
            flush=True,
        )
        # Network amplification factor: dP_dT calls in network solve vs
        # in component-level dmdot_dT.  Tells us how much overhead LM +
        # FD adds on top of a single brentq.
        if row["dmdot_calls"]:
            amp = row["net_calls"] / row["dmdot_calls"]
            print(f"  amplification  : net/dmdot = {amp:.1f}x", flush=True)

    print("\n=== Summary ===")
    print(f"{'component':22s} {'fwd ms':>8s} {'dmdot ms':>10s} {'net ms':>10s} "
          f"{'dP_dT/dmdot':>12s} {'dP_dT/net':>10s} {'us/dP_dT':>10s}")
    for r in rows:
        print(
            f"{r['name']:22s} "
            f"{r['dt_fwd_s']*1000:8.2f} "
            f"{r['dt_dmdot_s']*1000:10.2f} "
            f"{r['dt_net_s']*1000:10.2f} "
            f"{r['dmdot_calls']:12d} "
            f"{r['net_calls']:10d} "
            f"{r['dt_per_fwd_s']*1e6:10.1f}"
        )


if __name__ == "__main__":

    # test_compressible_fittings()
    # test_compressible_line_segment_csv()
    # test_comp_hydraulics()
    # test_incompressible_p2p()
    # test_incompressible_cont()
    # test_incompressible_csv_profile()
    #test_K()
    # test_contraction_expansion()
    # pseudo_orifice()
    # trying_orifices()
    test_compressible_dA()
    test_dmdot_dT_roundtrip()
    test_dmdot_dT_choke_raises()
    test_orifice_dmdot_dT_vs_dA()
    test_incompressible_dmdot_roundtrip()
    test_incompressible_valve_cavitation()
    test_incompressible_orifice_cavitation()

    # --- Choke / integrator debugging tests (migrated from test.py) ---
    print('--------------------------------------------------------')
    print('\nChoked-flow ideal-gas air nozzle')
    test_choked_mass_flux_ideal_gas_air()

    print('--------------------------------------------------------')
    print('\ncompressible_K choke round-trip')
    test_compressible_K_choke_roundtrip()

    print('--------------------------------------------------------')
    print('\nValve internal-throat (minimum_diameter) choke detection')
    test_valve_minimum_diameter_choke()

    print('--------------------------------------------------------')
    print('\nLine_Segment predictive Fanno / isothermal choke diagnostic (R7)')
    test_line_segment_choke_diagnostic()

    print('--------------------------------------------------------')
    print('\ncompressible_pipe_segment Heun integrator convergence order')
    test_pipe_segment_convergence_order()

    print('--------------------------------------------------------')
    print('\ncompressible_pipe_segment isothermal choke gate')
    test_isothermal_choke_gate()

    # Slow timing run (several seconds, no pass/fail assertions) -- run on demand.
    # benchmark_dmdot_dT()
