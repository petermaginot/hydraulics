import math
import os

import CoolProp.CoolProp as CP

import composition
from component_classes import ureg
from compressible_flow import (
    Bend as Compressible_Bend,
    Contraction_Expansion as Compressible_Contraction_Expansion,
    Line_Segment as Compressible_Line_Segment,
    _resolve_mdot,
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
    compressible_pipe_segment(
        abstract_state=AS_g,   # already updated to (P_gas, T_gas) above
        mdot=mdot,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        flow_area=A_gas,
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

    compressible_pipe_segment(
        abstract_state=AS_g,   # already updated to (P_gas, T_gas) above
        mdot=mdot,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        flow_area=A_gas,
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

    profile_points = seg.dP_dT(abstract_state=AS, flow_rate=Q_scfd, isothermal=True, mu = 1.1e-5)
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

    contraction_test.dP_dT(AS, Q_scfd)
    print(f'After contraction P:{AS.p()}, T:{AS.T()}')
    expansion_test.dP_dT(AS, Q_scfd)
    print(f'After expansion P:{AS.p()}, T:{AS.T()}')
    elbow.dP_dT(abstract_state=AS, flow_rate=Q_scfd)
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


if __name__ == "__main__":
    # test_compressible_fittings()
    # test_compressible_line_segment_csv()
    # test_comp_hydraulics()
    # test_incompressible_p2p()
    # test_incompressible_cont()
    test_incompressible_csv_profile()
