from compressible import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT, compressible_changing_area_K, _safe_update_PT, compressible_pipe_segment
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition
from component_classes import ureg
import math
import os

def test_deNevers8_10():
    #Example 8.10 on pages 315-317 in "Fluid Mechanics for Chemical Engineers, 3rd Ed" by Noel de Nevers
    #Adiabadic flow with friction (Fanno flow)
    #Given P0 (stagnation) = 30 psia, T0 (stagnation) = 200 F find flow rate if receiving reservoir P3 = 18 psia
    #Flow accelerates through a frictionless nozzle into an 8 ft long section of 1" Schedule 40 pipe
    #Example uses a two-iteration Fanno flow calculation and finds mass flow rate to be 0.317 lb/s. 
    #The temperature at the outlet of the pipe is found to be 622 degrees R.
    #The outlet velocity is 675.5 ft/s and the Mach number is 0.553
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

    print(f'  P_2 = {ureg.Quantity(P_2,"Pa").to("psi"):.4f}, T_2 = {ureg.Quantity(T_2,"degK").to("degR"):.2f}, Mach = {m_2:.4f}')

    print(f'Textbook ideal gas solution: P_2 = 11.28 psia, T_2 = 505 deg R, Mach #=0.623')

def test_ZuckerBiblarz10_3():
    #Example 10.3 on pages 296-297 in "Fundamentals of Gas Dynamics, 2nd Ed" by Robert Zucker and Oscar Biblarz
    #Flow with heat transfer (Rayleigh flow)
    #Air flowing at P1 = 10.0 psia, T1 = 400 R , v1 = 402 ft/s. 50 btu/lbm of heat is added to the gas.
    #Rayleigh flow assumes no friction, so we will use roughness = 0, L = 0.01 m, and ID = 1 m
    #Find final Mach number, temperature, pressure
    
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

    print(f'  P_3 = {ureg.Quantity(P_3,"Pa").to("psi"):.4f}, T_3 = {ureg.Quantity(T_3,"degK").to("degR"):.2f}, Mach = {m_3:.4f}')

    print(f'Textbook ideal gas solution: P_3 = 8.19 psia, T_3 = 580 deg R, Mach #=0.603') 



if __name__ == "__main__":
    print('\nZucker & Biblarz unnumbered example in section 5.7 (isentropic converging nozzle):')
    test_ZuckerBiblarz5_7()

    # print('\nZucker & Biblarz example 9.3 (Fanno flow):')
    # test_ZuckerBiblarz9_3()
    
    # print('de Nevers problem 8.10 (isentropic converging nozzle and Fanno flow):')
    # test_deNevers8_10()

    # print('\nZucker & Biblarz example 10.3 (Rayleigh flow):')
    # test_ZuckerBiblarz10_3()

