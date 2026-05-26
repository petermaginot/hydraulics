import math
import os

import CoolProp.CoolProp as CP

import composition
from component_classes import ureg
from compressible_flow import (
    Bend as Compressible_Bend,
    Contraction_Expansion as Compressible_Contraction_Expansion,
    Line_Segment as Compressible_Line_Segment,
    Valve as Compressible_Valve,
    Orifice as Compressible_Orifice,
    FlowState,
    _build_phase_limits,
    _resolve_mdot,
    _safe_update_PT,
    compressible_pipe_segment,
)
from incompressible import (
    Bend as Incompressible_Bend,
    Contraction_Expansion as Incompressible_Contraction_Expansion,
    Incompressible_Fluid,
    Line_Segment as Incompressible_Line_Segment,
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
    trying_orifices()
