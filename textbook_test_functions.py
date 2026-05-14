"""
A selection of solved problems from various textbooks, used to validate the program's output for different scenarios
"""

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition
from component_classes import ureg
import math
import os

def test_deNevers8_10():
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment
    #Example 8.10 on pages 315-317 in "Fluid Mechanics for Chemical Engineers, 3rd Ed" by Noel de Nevers
    #Adiabadic flow with friction (Fanno flow)
    #Given P0 (stagnation) = 30 psia, T0 (stagnation) = 200 F find flow rate if receiving reservoir P3 = 18 psia
    #Flow accelerates through a frictionless nozzle into an 8 ft long section of 1" Schedule 40 pipe
    #Example uses a two-iteration Fanno flow calculation and finds mass flow rate to be 0.317 lb/s. 
    #The textbook solution for the outlet conditions is T=622 degrees R, outlet velocity = 675.5 ft/s and the Mach number = 0.553, mass flow rate = 0.317 lbm/s
    #Note that the textbook edition I have has a typo for the SI unit solution, listing it as 1.44 kg/s rather than the correct 0.144 kg/s.
    P_target = ureg.Quantity(18, "psi").to("Pa").magnitude
    P0    = ureg.Quantity(30, "psi").to("Pa").magnitude    # Pa
    T0 = ureg.Quantity(200, "degF").to("degK").magnitude    # K
    OD_pipe    = ureg.Quantity(1.315, "inch").to("m").magnitude
    WT_pipe = ureg.Quantity(0.133, "inch").to("m").magnitude
    ID_pipe = OD_pipe - 2 * WT_pipe
    eps_gas  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    dL_gas   = ureg.Quantity(8.0,    "feet").to("m").magnitude
    dz_gas   = ureg.Quantity(0.0,    "feet").to("m").magnitude
    AS = composition.define_composition(
        y_Nitrogen = 0.79,
        y_Oxygen = 0.21,
        eos = "HEOS"
        )
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits
    _safe_update_PT(AS, P0, T0, *phase_limits)

    A_0 = 10000
    A_1 = math.pi * ID_pipe**2/4

    Q_scfd = ureg.Quantity(100, "mscf/day")  # initial guess
    tolerance_Pa = ureg.Quantity(0.01, "psi").to("Pa").magnitude

    print(f'\nInputs: P0 = {ureg.Quantity(P0,"Pa").to("psi"):.3f}, T0 = {ureg.Quantity(T0,"degK").to("degF"):.3f}')
    print(f'Target: P_target = {ureg.Quantity(P_target,"Pa").to("psi"):.3f}\n')

    # Bisection bracket in mscf/day; None until that side has been evaluated
    Q_low = None   # highest Q tried that gives P_2 > P_target (need more flow)
    Q_high = None  # lowest Q tried that gives P_2 < P_target (need less flow)

    for iteration in range(30):
        mdot = Q_scfd.to("mol/s").magnitude * AS.molar_mass()

        _safe_update_PT(AS, P0, T0, *phase_limits)

        #For point 0 to point 1, gas undergoes isentropic acceleration through ideal converging nozzle.
        #Assume a very large upstream area so velocity is approximately zero; K = 0 (lossless)
        compressible_changing_area_K(
            abstract_state=AS, mdot=mdot, A_in=A_0, A_out=A_1, K=0,
            T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
            T_critical=T_critical, P_critical=P_critical,
        )
        P_1 = AS.p()

        compressible_pipe_segment(
            abstract_state=AS,   # already updated to (P_1, T_1) by compressible_changing_area_K
            mdot=mdot,
            dL=dL_gas, dz=dz_gas, D_h=ID_pipe, roughness=eps_gas,
            flow_area=A_1, isothermal=False,
            T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
            T_critical=T_critical, P_critical=P_critical,
        )

        #At point 2, the pressure will be the same as point 3. The temperature won't be the same,
        #since it will lose all of its velocity head and dump that energy into heat, but the problem doesn't ask for that.
        P_2 = AS.p()
        T_2 = AS.T()
        v_2 = mdot / AS.rhomass() / A_1
        m_2 = v_2 / AS.speed_sound()
        error_psi = ureg.Quantity(P_2 - P_target, "Pa").to("psi").magnitude

        print(f'Iteration {iteration + 1}: Q = {Q_scfd.to("mscf/day"):.3f}, mdot = {ureg.Quantity(mdot,"kg/s").to("lb/s"):.4f}')
        print(f'  P_1 = {ureg.Quantity(P_1,"Pa").to("psi"):.4f}')
        print(f'  P_2 = {ureg.Quantity(P_2,"Pa").to("psi"):.4f}, T_2 = {ureg.Quantity(T_2,"degK").to("degR"):.2f}')
        print(f'  v_2 = {ureg.Quantity(v_2,"m/s").to("ft/s"):.2f}, Mach = {m_2:.4f}, P_error = {error_psi:+.4f} psi\n')


        if abs(P_2 - P_target) < tolerance_Pa:
            print(f'Converged after {iteration + 1} iteration(s).')
            break

        # Update bisection bracket: higher Q → more pressure drop → lower P_2
        Q_val = Q_scfd.to("mscf/day").magnitude
        if P_2 > P_target:
            Q_low = Q_val   # need more flow; record as lower bound
        else:
            Q_high = Q_val  # need less flow; record as upper bound

        if Q_low is not None and Q_high is not None:
            # Both bracket sides known — bisect to guarantee convergence
            Q_scfd = ureg.Quantity((Q_low + Q_high) / 2, "mscf/day")
        else:
            # Only one side bracketed yet — scale proportionally to find the other side quickly
            Q_scfd = Q_scfd * (P_2 / P_target)
    print('Textbook solution: mass flow rate = 0.317 lbm/s, outlet velocity = 675.5 ft/s, outlet temperature = 622 deg R, Mach #=0.553')

def test_ZuckerBiblarz5_7():
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment
    #Example on pages 124-126 in "Fundamentals of Gas Dynamics, 2nd Ed" by Robert Zucker and Oscar Biblarz
    #Isentropic changing area flow
    #Air at stagnation conditions of 100 psia and 600 deg Rflows through a nozzle into a receiver at 80.2 psia
    #Find final Mach number, temperature and velocity
    #Assume an arbitrary large area initially for stagnation conditions, and a significantly smaller area for final conditions
    A0 = 1000
    A1 = 1

    mdot = 1000 #kg/s, initial guess, iterate it until pressure matches target

    P0    = ureg.Quantity(100, "psi").to("Pa").magnitude    # Pa
    T0 = ureg.Quantity(600, "degR").to("degK").magnitude    # K

    P_target  = ureg.Quantity(80.2, "psi").to("Pa").magnitude    # Pa

    AS = composition.define_composition(
        y_Nitrogen = 0.79,
        y_Oxygen = 0.21,
        eos = "HEOS"
        )

    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits

    tolerance_Pa = ureg.Quantity(0.01, "psi").to("Pa").magnitude

    # Bisection bracket in mscf/day; None until that side has been evaluated
    mdot_low = None   # highest flow rate tried that gives P_2 > P_target (need more flow)
    mdot_high = None  # lowest flow rate tried that gives P_2 < P_target (need less flow)

    for iteration in range(30):
        _safe_update_PT(AS, P0, T0, *phase_limits)

        #For point 0 to point 1, gas undergoes isentropic acceleration through ideal converging nozzle.
        #Assume a very large upstream area so velocity is approximately zero; K = 0 (lossless)
        compressible_changing_area_K(
            abstract_state=AS, mdot=mdot, A_in=A0, A_out=A1, K=0,
            T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
            T_critical=T_critical, P_critical=P_critical,
        )

        P_1 = AS.p()
        T_1 = AS.T()
        v_1 = mdot / AS.rhomass() / A1
        m_1 = v_1 / AS.speed_sound()
        error_psi = ureg.Quantity(P_1 - P_target, "Pa").to("psi").magnitude

        print(f'Iteration {iteration + 1}: mdot = {ureg.Quantity(mdot,"kg/s").to("lb/s"):.4f}')
        print(f'  P_0 = {ureg.Quantity(P0,"Pa").to("psi"):.4f}')
        print(f'  P_1 = {ureg.Quantity(P_1,"Pa").to("psi"):.4f}, T_1 = {ureg.Quantity(T_1,"degK").to("degR"):.2f}')
        print(f'  v_1 = {ureg.Quantity(v_1,"m/s").to("ft/s"):.2f}, Mach = {m_1:.4f}, P_error = {error_psi:+.4f} psi\n')


        if abs(P_1 - P_target) < tolerance_Pa:
            print(f'Converged after {iteration + 1} iteration(s).')
            break

        # Update bisection bracket: higher mdot → more pressure drop → lower P_2
        mdot_val = mdot
        if P_1 > P_target:
            mdot_low = mdot_val   # need more flow; record as lower bound
        else:
            mdot_high = mdot_val  # need less flow; record as upper bound

        if mdot_low is not None and mdot_high is not None:
            # Both bracket sides known — bisect to guarantee convergence
            mdot = (mdot_low + mdot_high) / 2
        else:
            # Only one side bracketed yet — scale proportionally to find the other side quickly
            mdot = mdot * (P_1 / P_target)
    print('Textbook solution: outlet velocity = 663 ft/s, outlet temperature = 563 deg R, Mach #=0.57')

def test_ZuckerBiblarz9_3():
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment
    #Example 9.3 on pages 259-260 in "Fundamentals of Gas Dynamics, 2nd Ed" by Robert Zucker and Oscar Biblarz
    #Adiabadic flow with friction (Fanno flow)
    #Air flowing at P1 = 20 psia, T1 = 70 F , v1 = 406 ft/s in a 6" diameter galvanized iron duct (absolute roughness = 0.0005 ft)
    #Find final Mach number, temperature, pressure after 70 feet
    
    P0    = ureg.Quantity(20, "psi").to("Pa").magnitude    # Pa
    T0 = ureg.Quantity(70, "degF").to("degK").magnitude    # K
    ID_pipe = ureg.Quantity(6, "inch").to("m").magnitude    # m
    eps_gas  = ureg.Quantity(0.0005, "ft").to("m").magnitude
    v_1   = ureg.Quantity(406.0,    "ft/s").to("m/s").magnitude
    dL_gas   = ureg.Quantity(70.0,    "feet").to("m").magnitude
    dz_gas   = ureg.Quantity(0.0,    "feet").to("m").magnitude
    AS = composition.define_composition(
        y_Nitrogen = 0.79,
        y_Oxygen = 0.21,
        eos = "HEOS"
        )
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits
    _safe_update_PT(AS, P0, T0, *phase_limits)

    A_1 = math.pi * ID_pipe**2/4
    #First find volumetric flow rate
    Vdot_1 = v_1 * A_1

    #convert to mass flow rate
    mdot = Vdot_1 * AS.rhomass()

    print(f'\nInputs: P0 = {ureg.Quantity(P0,"Pa").to("psi"):.3f}, T0 = {ureg.Quantity(T0,"degK").to("degF"):.3f}')

    compressible_pipe_segment(
        abstract_state=AS,
        mdot=mdot,
        dL=dL_gas, dz=dz_gas, D_h=ID_pipe, roughness=eps_gas,
        flow_area=A_1, isothermal=False,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )

    P_2 = AS.p()
    T_2 = AS.T()
    v_2 = mdot / AS.rhomass() / A_1
    m_2 = v_2 / AS.speed_sound()

    print(f'Outputs:  P_2 = {ureg.Quantity(P_2,"Pa").to("psi"):.4f}, T_2 = {ureg.Quantity(T_2,"degK").to("degR"):.2f}, Mach = {m_2:.4f}')

    print(f'Textbook ideal gas solution: P_2 = 11.28 psia, T_2 = 505 deg R, Mach #=0.623')

def test_ZuckerBiblarz10_3():
    #Example 10.3 on pages 296-297 in "Fundamentals of Gas Dynamics, 2nd Ed" by Robert Zucker and Oscar Biblarz
    #Flow with heat transfer (Rayleigh flow)
    #Air flowing at P1 = 10.0 psia, T1 = 400 R , v1 = 402 ft/s. 50 btu/lbm of heat is added to the gas.
    #Rayleigh flow assumes no friction, so we will use roughness = 0, L = 0.01 m, and ID = 1 m
    #Find final Mach number, temperature, pressure
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment
    P2    = ureg.Quantity(10, "psi").to("Pa").magnitude    # Pa
    T2 = ureg.Quantity(400, "degR").to("degK").magnitude    # K
    ID_pipe = 1.0    # m
    eps_gas  = ureg.Quantity(0.0, "ft").to("m").magnitude
    v_2   = ureg.Quantity(402.0,    "ft/s").to("m/s").magnitude
    dL_gas   = 0.01 #m
    dz_gas   = ureg.Quantity(0.0,    "feet").to("m").magnitude
    AS = composition.define_composition(
        y_Nitrogen = 0.79,
        y_Oxygen = 0.21,
        eos = "HEOS"
        )
    phase_limits = _build_phase_limits(AS)
    T_cricondentherm, P_cricondenbar, T_critical, P_critical = phase_limits
    _safe_update_PT(AS, P2, T2, *phase_limits)

    A_2 = math.pi * ID_pipe**2/4
    mdot = v_2 * A_2 * AS.rhomass()

    dq = ureg.Quantity(50, "Btu/lb").to("J/kg").magnitude * mdot

    print(f'\nInputs: P2 = {ureg.Quantity(P2,"Pa").to("psi"):.3f}, T2 = {ureg.Quantity(T2,"degK").to("degR"):.3f}')

    compressible_pipe_segment(
        abstract_state=AS,
        mdot=mdot,
        dL=dL_gas, dz=dz_gas, D_h=ID_pipe, roughness=eps_gas,
        flow_area=A_2, isothermal=False, q_wall= dq,
        T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
        T_critical=T_critical, P_critical=P_critical,
    )

    P_3 = AS.p()
    T_3 = AS.T()
    v_3 = mdot / AS.rhomass() / A_2
    m_3 = v_3 / AS.speed_sound()

    print(f'Outputs:  P_3 = {ureg.Quantity(P_3,"Pa").to("psi"):.4f}, T_3 = {ureg.Quantity(T_3,"degK").to("degR"):.2f}, Mach = {m_3:.4f}')

    print(f'Textbook ideal gas solution: P_3 = 8.19 psia, T_3 = 580 deg R, Mach #=0.603') 

def dP_fittings():
    from math import pi
    import fluids.units as fittings
    from fluids.units import u as ureg

    ID_pipe = ureg.Quantity(3.068, "inch")

    rho = ureg.Quantity(49.0, "lb/ft^3")
    flow_rate = ureg.Quantity(10000, "bbl/day")
    area = (ID_pipe**2)/4*pi

    velocity = flow_rate / area

    K_globe = fittings.K_globe_valve_Crane(D1=ID_pipe, D2=ID_pipe)
    K_swing_check = fittings.K_swing_check_valve_Crane(D= ID_pipe, angled=True)

    dP_globe = velocity**2 * K_globe * rho / 2
    dP_check = velocity**2 * K_swing_check * rho / 2

    print(f'Globe valve K factor: {K_globe}, pressure drop: {dP_globe.to("psi")}')
    print(f'Check valve K factor: {K_swing_check}, pressure drop: {dP_check.to("psi")}')


def test_deNevers6_11():
    #Example 6.11 and 6.12 on page 203-204 in "Fluid Mechanics for Chemical Engineers, 3rd Ed" by Noel de Nevers
    #Calculate the pressure drop across 3000 ft of 3" pipe, two globe valves, a swing check valve, and nine standard radius 90 degree elbows.abs
    #Note that in my copy of the book, the solution to 6.12 has a typo, it multiplies the globe valve K factor by 3 instead of 2, which affects the results. Instead of 31 psi, it should be 23 psi drop across the fittings.

    from parallel import parallel_incompressible
    from incompressible import Line_Segment, Bend, Incompressible_Fluid, Valve
    from component_classes import ureg

    from fluids import fittings
    ID_pipe = ureg.Quantity(3.068, "inch").to("m").magnitude
    eps  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    pipe_length = ureg.Quantity(3000, "ft").to("m").magnitude
    mu = ureg.Quantity(50, "cP").to("Pa*s").magnitude
    rho = ureg.Quantity(62.3, "lb/ft^3").to("kg/m^3").magnitude
    flow_rate = ureg.Quantity(300, "gal/min")


    #Initialize classes
    fittinglist = []    
    #Pipe
    PipeSeg = Line_Segment(roughness= eps, id_val = ID_pipe, length= pipe_length, elevation_change = 0)
    fittinglist.append(PipeSeg)

    #globe valves
    K_globe = fittings.K_globe_valve_Crane(D1=ID_pipe, D2=ID_pipe)
    print(f'Globe valve K = {K_globe}')
    GlobeValve1 = Valve(ID_pipe, K_globe, 'Globe valve 1')
    GlobeValve2 = Valve(ID_pipe, K_globe, 'Globe valve 2')
    fittinglist.append(GlobeValve1)
    fittinglist.append(GlobeValve2)

    #swing check
    K_swing_check = fittings.K_swing_check_valve_Crane(D= ID_pipe, angled=True)
    SwingCheck = Valve(ID_pipe, K_swing_check, 'Check')
    fittinglist.append(SwingCheck)

    print(f'Check valve K = {K_swing_check}')
    #elbows
    Elbows = []
    for i in range(9):
        fittinglist.append(Bend(ID_pipe, 90, 1, str(i)))

    Fld = Incompressible_Fluid(density = rho, viscosity = mu)

    dP, flow_fractions = parallel_incompressible([fittinglist], fluid = Fld, total_flow_rate = flow_rate )
    dP_pipe = ureg.Quantity(PipeSeg.dP(Fld, flow_rate), "Pa")
    dP_fittings = ureg.Quantity(dP, "Pa") - dP_pipe
    print(f'Overall pressure drop:{ ureg.Quantity(dP, "Pa").to("psi")}')
    print(f'Pipe friction = {dP_pipe.to("psi")}, fitting friction = {dP_fittings.to("psi")}')
    print('Textbook solution = 529 psi overall, 484 psi due to pipe, 45 psi due to fittings using equivalent lengths (6.11) and 23 psi* using sum of K values (6.12)')
    print('Pipe friction estimate agrees well, but the textbook K values for fittings are appreciably different from those calculated by the fluids library (Kglobe = 6.3 vs 5.9, Kcheck = 2.0 vs 1.7, Kelbow = 0.74 vs 0.54.')
    print('*Note that in my copy of the book, the solution to 6.12 has a typo, it multiplies the globe valve K factor by 3 instead of 2. The 31 psi drop across the fittings shown in the text is incorrect, it should be 23 psi drop across the fittings.')

def test_deNevers6_4():
    #Example 6.4 on page 188 in "Fluid Mechanics for Chemical Engineers, 3rd Ed" by Noel de Nevers
    #Calculate the pressure drop across 3000 ft of 3" pipe

    from incompressible import Line_Segment, Incompressible_Fluid
    from component_classes import ureg
    ID_pipe = ureg.Quantity(3.068, "inch").to("m").magnitude
    eps  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    pipe_length = ureg.Quantity(3000, "ft").to("m").magnitude
    mu = ureg.Quantity(50, "cP").to("Pa*s").magnitude
    rho = ureg.Quantity(62.3, "lb/ft^3").to("kg/m^3").magnitude
    flow_rate = ureg.Quantity(300, "gal/min")

    PipeSeg = Line_Segment(roughness= eps, id_val = ID_pipe, length= pipe_length, elevation_change = 0)
    Fld = Incompressible_Fluid(density = rho, viscosity = mu)
    dP = PipeSeg.dP(Fld, flow_rate)
    print( ureg.Quantity(dP, "Pa").to("psi"))
    print('Textbook solution dP = 484 psi')

def test_Crane_air_line():
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment, _resolve_mdot

    roughness = ureg.Quantity(0.00015, "ft")
    seg_length = ureg.Quantity(75, "ft")
    OD = ureg.Quantity(1.315, "inch")
    WT = ureg.Quantity(0.133, "inch")
    ID = OD - 2*WT
    viscosity = None #use the abstract state viscosity
    area = ID**2/4*math.pi

    seg = Line_Segment(roughness = roughness, id_val = ID, length = seg_length, elevation_change=0.0)

    P_in = ureg.Quantity(65+14.696, "psi").to("Pa").magnitude #65 psig
    T_in = ureg.Quantity(115, "degF").to("degK").magnitude   # K

    AS = composition.define_composition(
        y_Nitrogen = 0.79,
        y_Oxygen = 0.21,
        eos = "HEOS"
    )
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

    AS.update(CP.PT_INPUTS, P_in, T_in)

    Q_scfd = ureg.Quantity(100.0, "scf/min")

    mdot = _resolve_mdot(Q_scfd, AS)

    v_in = mdot / AS.molar_mass()

    Ma_in = v_in /AS.speed_sound()

    print(f'  inlet: P={ureg.Quantity(P_in,"Pa").to("psi"):.4f}, T={T_in:.4g} K, Ma={Ma_in:.4f}')
    seg.dP_dT(abstract_state=AS, flow_rate=Q_scfd, isothermal=True, mu=viscosity,
    T_cricondentherm=T_cric,
    P_cricondenbar=P_bar,
    T_critical=T_c,
    P_critical=P_c,)
 
    P_out = AS.p()
    T_out = AS.T()

    rho_out = AS.rhomass()
    area_out = seg.profile[-1][3]
    v_out = _resolve_mdot(Q_scfd, AS) / (rho_out * area_out)
    Ma_out = v_out / AS.speed_sound()

    print(f'  outlet: P={ureg.Quantity(P_out,"Pa").to("psi"):.4f} psi, v = {ureg.Quantity(v_out,"m/s").to("ft/min"):.4f}, T = {T_out}K')
    print(f'dP = {ureg.Quantity(P_out-P_in,"Pa").to("psi"):.4f}')




def test_Crane_gas_pipeline():
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment, _resolve_mdot
    csv_path = os.path.join(os.path.dirname(__file__), "testprofile_crane.csv")
    roughness = ureg.Quantity(0.00015, "ft")
    seg_length = ureg.Quantity(100, "miles")
    OD = ureg.Quantity(14.0, "inch")
    WT = ureg.Quantity(0.312, "inch")
    ID = OD - 2*WT
    # viscosity = 1.1e-5 #Pa*s, given in problem.
    viscosity = None #use the abstract state viscosity

    seg = Line_Segment(roughness = roughness, id_val = ID, length = seg_length, elevation_change=0.0)
    P_target = ureg.Quantity(300, "psi").to("Pa").magnitude
    P_in = ureg.Quantity(1300, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(40, "degF").to("degK").magnitude   # K

    AS = composition.define_composition(
        y_Methane = 0.75,
        y_Ethane = 0.21,
        y_Propane=0.04,
        eos = "HEOS"
    )
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

    tolerance_Pa = ureg.Quantity(0.01, "psi").to("Pa").magnitude

    Q_mmscfd = 125.5  # initial guess, iterate until P_out matches P_target

    # Bisection bracket in mmscf/day; None until that side has been evaluated
    Q_low = None   # highest Q tried that gives P_out > P_target (need more flow)
    Q_high = None  # lowest Q tried that gives P_out < P_target (need less flow)

    for iteration in range(30):
        AS.update(CP.PT_INPUTS, P_in, T_in)
        Q_scfd = ureg.Quantity(Q_mmscfd, "mmscf/day")
        print(f'Guess: {Q_scfd}')
        rho_in = AS.rhomass()
        area_in = seg.profile[0][3]
        v_in = _resolve_mdot(Q_scfd, AS) / (rho_in * area_in)
        Ma_in = v_in / AS.speed_sound()

        try:
            seg.dP_dT(abstract_state=AS, flow_rate=Q_scfd, isothermal=True, mu=viscosity, 
            T_cricondentherm=T_cric, 
            P_cricondenbar=P_bar, 
            T_critical=T_c,
            P_critical=P_c,)
        except RuntimeError as e:
            # Flow too high — segment failed to converge (likely approaching choked flow)
            print(f'Iteration {iteration + 1}: Q = {Q_mmscfd:.4f} mmscf/day — solver failed, treating as upper bound')
            print(f'  ({e})\n')
            Q_high = Q_mmscfd
            if Q_low is not None:
                Q_mmscfd = (Q_low + Q_high) / 2
            else:
                Q_mmscfd = Q_mmscfd / 2
            continue

        P_out = AS.p()
        T_out = AS.T()

        rho_out = AS.rhomass()
        area_out = seg.profile[-1][3]
        v_out = _resolve_mdot(Q_scfd, AS) / (rho_out * area_out)
        Ma_out = v_out / AS.speed_sound()

        P_out_psi = ureg.Quantity(P_out, "Pa").to("psi").magnitude
        dP_psi = ureg.Quantity(P_out - P_in, "Pa").to("psi").magnitude
        error_psi = ureg.Quantity(P_out - P_target, "Pa").to("psi").magnitude

        print(f'Iteration {iteration + 1}: Q = {Q_mmscfd:.4f} mmscf/day')
        print(f'  inlet: P={ureg.Quantity(P_in,"Pa").to("psi"):.4f}, T={T_in:.4g} K, Ma={Ma_in:.4f}')
        print(f'  outlet: P={P_out_psi:.4g} psi, T={T_out:.4g} K, Ma={Ma_out:.4f}')
        print(f'  dP = {dP_psi:.4f} psi, P_error = {error_psi:+.4f} psi\n')

        if abs(P_out - P_target) < tolerance_Pa:
            print(f'Converged after {iteration + 1} iteration(s).')
            break

        # Higher Q → more pressure drop → lower P_out
        if P_out > P_target:
            Q_low = Q_mmscfd   # need more flow; record as lower bound
        else:
            Q_high = Q_mmscfd  # need less flow; record as upper bound

        if Q_low is not None and Q_high is not None:
            Q_mmscfd = (Q_low + Q_high) / 2
        else:
            Q_mmscfd = Q_mmscfd * (P_out / P_target)

def test_Crane_choked_steam():
    #NOTE this problem is a dead end for this program, because it starts as saturated vapor and immediately goes two-phase upon expansion.
    from compressible_flow import Line_Segment, Bend, Contraction_Expansion, Valve, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment, _resolve_mdot
    from fluids import fittings
    roughness = ureg.Quantity(0.00015, "ft")
    seg_length = ureg.Quantity(30, "ft")
    OD = ureg.Quantity(2.375, "inch")
    WT = ureg.Quantity(0.154, "inch")
    P_in = ureg.Quantity(170, "psi").to("Pa").magnitude
    ID = OD - 2*WT

    fitting_list = []

    seg = Line_Segment(roughness = roughness, id_val = ID, length = seg_length, elevation_change=0.0)
    fitting_list.append(seg)

    elbow = Bend(Di = ID, ang_deg = 90, bend_dias = 1.0)
    fitting_list.append(elbow)

    k_valve = fittings.K_globe_valve_Crane(D1 = ID.to("m").magnitude, D2 = ID.to("m").magnitude)

    valve = Valve(Di = ID, K = k_valve)
    fitting_list.append(valve)


    M_target = 0.97


    AS = composition.define_composition(
        y_Water = 1,
        eos = "HEOS"
    )
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    
    tolerance_Ma = 0.002

    mdot =   ureg.Quantity(100, "lb/hr") # initial guess, iterate until P_out matches P_target

    # Bisection bracket on mass flow rate; None until that side has been evaluated
    mdot_low = None   # highest mdot tried that gives P_out > P_target (need more flow)
    mdot_high = None  # lowest mdot tried that gives P_out < P_target (need less flow)

    for iteration in range(30):
        AS.update(CP.PQ_INPUTS, P_in, 1.0)
        print(f'Guess: {mdot}')
        rho_in = AS.rhomass()
        area_in = seg.profile[0][3]
        v_in = _resolve_mdot(mdot, AS) / (rho_in * area_in)
        Ma_in = v_in / AS.speed_sound()

        try:
            seg.dP_dT(abstract_state=AS, flow_rate=mdot, isothermal=False,
            T_cricondentherm=T_cric,
            P_cricondenbar=P_bar,
            T_critical=T_c,
            P_critical=P_c,)
            #next, the elbow
            elbow.dP_dT(abstract_state=AS, flow_rate=mdot,
            T_cricondentherm=T_cric,
            P_cricondenbar=P_bar,
            T_critical=T_c,
            P_critical=P_c,)
            #and the valve
            valve.dP_dT(abstract_state=AS, flow_rate=mdot,
            T_cricondentherm=T_cric,
            P_cricondenbar=P_bar,
            T_critical=T_c,
            P_critical=P_c,)
        except RuntimeError as e:
            # Flow too high — segment failed to converge (likely approaching choked flow)
            print(f'Iteration {iteration + 1}: mdot = {mdot.to("lb/hr").magnitude:.4f} lb/hr — solver failed, treating as upper bound')
            print(f'  ({e})\n')
            mdot_high = mdot
            if mdot_low is not None:
                mdot = (mdot_low + mdot_high) / 2
            else:
                mdot = mdot / 2
            continue

        P_out = AS.p()
        T_out = AS.T()

        rho_out = AS.rhomass()
        area_out = seg.profile[-1][3]
        v_out = _resolve_mdot(mdot, AS) / (rho_out * area_out)
        Ma_out = v_out / AS.speed_sound()


        error_Ma = Ma_out - M_target

        print(f'Iteration {iteration + 1}: mdot = {mdot.to("lb/hr").magnitude:.4f} lb/hr')
        print(f'  inlet:  P={ureg.Quantity(P_in,"Pa").to("psi"):.4f}, Ma={Ma_in:.4f}')
        print(f'  outlet: P={ureg.Quantity(P_out,"Pa").to("psi"):.4f}, T={T_out:.4g} K, Ma={Ma_out:.4f}')
        print(f'  Ma_error = {error_Ma:+.4f}\n')

        if abs(error_Ma) < tolerance_Ma:
            print(f'Converged after {iteration + 1} iteration(s).')
            break

        # Higher mdot → higher outlet Mach
        if Ma_out < M_target:
            mdot_low = mdot   # need more flow; record as lower bound
        else:
            mdot_high = mdot  # need less flow; record as upper bound

        if mdot_low is not None and mdot_high is not None:
            mdot = (mdot_low + mdot_high) / 2
        else:
            mdot = mdot * (M_target / Ma_out)


if __name__ == "__main__":

    print('Zucker & Biblarz unnumbered example in section 5.7 (isentropic converging nozzle):')
    test_ZuckerBiblarz5_7()

    print('--------------------------------------------------------')
    print('Zucker & Biblarz example 9.3 (Fanno flow):')
    test_ZuckerBiblarz9_3()

    print('--------------------------------------------------------')
    print('de Nevers example 8.10 (isentropic converging nozzle and Fanno flow):')
    test_deNevers8_10()

    print('--------------------------------------------------------')
    print('Zucker & Biblarz example 10.3 (Rayleigh flow):')
    test_ZuckerBiblarz10_3()

    print('--------------------------------------------------------')
    print('de Nevers example 6.11 and 6.12 (incompressible fluid friction with pipe and fittings):')
    test_deNevers6_11()

    print('--------------------------------------------------------')
    print('de Nevers example 6.4 (incompressible fluid friction with pipe):')
    test_deNevers6_4()

    print('--------------------------------------------------------')
    print('\nCrane TP410 example')
    test_Crane_gas_pipeline()
    test_Crane_air_line()

    # test_Crane_choked_steam()