import math
import os

import CoolProp.CoolProp as CP

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
