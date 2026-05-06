"""
Calculates flow partitioning on parallel line segments
"""
def parallel_incompressible(line_segment_list, fluid, total_flow_rate):
    # line_segment_list is a list of at least two line segment class objects
    # fluid is an incompressible fluid class object
    # total_flow_rate is a flow rate in mass or volumetric flow rate

    #Returns total differential pressure, list of flow fractions on each segment


    #For initial guess, assume identical friction factors for all line segments and use average cross sectional area, so that resistance to flow simplifies to:
    #dP = f Li/Di * mdoti/(rho^2 * Ai ^2) 
    #Resistance proportional to Li/(Ai^2.5) [not exact for noncircular geometry or variable diameters, but close enough for a first guess]
    #Total flow resistance = 1/(sum of 1/individual segment resistances)
    #Xi = flow fraction on segment i = total flow ressitance / individual line resistance
    sum_recip_R = 0.0
    for i in line_segment_list:
        avg_area = i.volume_m3/i.total_length_m

        sum_recip_R += avg_area**2.5 / i.total_length_m
    
    Rtotal = 1/sum_recip_R

    flow_fraction = []
    for i in line_segment_list:
        avg_area = i.volume_m3/i.total_length_m
        flow_fraction.append(Rtotal/(i.total_length_m / avg_area**2.5))

    # Iterate flow fractions until dP across every segment converges to a common value.
    # Physics constraint: parallel segments share the same inlet/outlet, so dP_i must
    # be equal for all i.
    #
    # Newton's method: estimate d(dP_i)/d(ff_i) numerically via a small perturbation.
    # Because elevation is a flow-independent offset in dP, only the friction term
    # appears in the slope. This makes Newton robust when elevation >> friction
    # (low flow rates), where a ratio-based correction would stall.
    #
    # Given slopes s_i, the Newton target dP_common and correction delta_ff_i are:
    #   dP_common   = sum(dP_i / s_i) / sum(1 / s_i)
    #   delta_ff_i  = (dP_common - dP_i) / s_i
    # By construction sum(delta_ff_i) = 0, so total flow is exactly preserved.
    MAX_ITER = 20
    TOL = 1e-6
    EPS = 1e-4  # dimensionless perturbation to each flow fraction for slope estimation

    dP_list = []
    dP_target = 0.0

    for k in range(MAX_ITER):
        dP_list = [
            seg.dP(fluid=fluid, flow_rate=total_flow_rate * ff)
            for seg, ff in zip(line_segment_list, flow_fraction)
        ]

        slopes = [
            (seg.dP(fluid=fluid, flow_rate=total_flow_rate * (ff + EPS)) - dP) / EPS
            for seg, ff, dP in zip(line_segment_list, flow_fraction, dP_list)
        ]

        sum_inv_s = sum(1.0 / s for s in slopes)
        dP_target = sum(dp / s for dp, s in zip(dP_list, slopes)) / sum_inv_s

        max_rel_dev = max(abs(dp - dP_target) / abs(dP_target) for dp in dP_list)
        if max_rel_dev < TOL:
            break

        delta_ff = [(dP_target - dp) / s for dp, s in zip(dP_list, slopes)]
        new_fractions = [max(ff + dff, 1e-10) for ff, dff in zip(flow_fraction, delta_ff)]
        total_ff = sum(new_fractions)
        flow_fraction = [ff / total_ff for ff in new_fractions]
        # print(f'iteration {k}, {flow_fraction}, {dP_list}')
    return dP_target, flow_fraction

def parallel_compressible(line_segment_list, AS, total_flow_rate):
    # line_segment_list is a list of at least two line segment class objects
    # fluid is a CoolProp Abstract State class object
    # total_flow_rate is a flow rate in mass, molar, or volumetric flow rate

    #Returns list of outlet pressures on each segment, a list of temperatures on each segment, and a list of flow fractions on each segment


    #For initial guess, assume identical friction factors for all line segments and use average cross sectional area, so that resistance to flow simplifies to:
    #dP = f Li/Di * mdoti/(rho^2 * Ai ^2) 
    #Resistance proportional to Li/(Ai^2.5) [not exact for noncircular geometry or variable diameters, but close enough for a first guess]
    #Total flow resistance = 1/(sum of 1/individual segment resistances)
    #Xi = flow fraction on segment i = total flow ressitance / individual line resistance
    sum_recip_R = 0.0
    for i in line_segment_list:
        avg_area = i.volume_m3/i.total_length_m

        sum_recip_R += avg_area**2.5 / i.total_length_m
    
    Rtotal = 1/sum_recip_R

    flow_fraction = []
    for i in line_segment_list:
        avg_area = i.volume_m3/i.total_length_m
        flow_fraction.append(Rtotal/(i.total_length_m / avg_area**2.5))

    P0 = AS.p()
    T0 = AS.T()

    # Precompute the phase envelope once.  build_phase_envelope is expensive
    # for multicomponent HEOS mixtures (seconds per call), and the composition
    # does not change across segments or iterations, so we forward the four
    # scalars into every dP_dT call to skip the per-call envelope build.
    from compressible import _build_phase_limits
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    print(f"phase limits: T_cric={T_cric}, P_bar={P_bar}, T_c={T_c}, P_c={P_c}")

    MAX_ITER = 20
    TOL = 1e-6
    EPS = 1e-4  # dimensionless perturbation to each flow fraction for slope estimation

    def _outlet(seg, ff):
        AS_out, _ = seg.dP_dT(
            AS, total_flow_rate * ff, P0, T0,
            T_cricondentherm=T_cric, P_cricondenbar=P_bar,
            T_critical=T_c, P_critical=P_c,
        )
        return AS_out.p(), AS_out.T()

    P_out_list = []
    T_out_list = []
    P_target = 0.0

    for k in range(MAX_ITER):
        P_out_list = []
        T_out_list = []
        for seg, ff in zip(line_segment_list, flow_fraction):
            P_out, T_out = _outlet(seg, ff)
            P_out_list.append(P_out)
            T_out_list.append(T_out)

        slopes = [
            (_outlet(seg, ff + EPS)[0] - P_out) / EPS
            for seg, ff, P_out in zip(line_segment_list, flow_fraction, P_out_list)
        ]

        sum_inv_s = sum(1.0 / s for s in slopes)
        P_target = sum(p / s for p, s in zip(P_out_list, slopes)) / sum_inv_s

        max_rel_dev = max(abs(p - P_target) / abs(P_target) for p in P_out_list)
        if max_rel_dev < TOL:
            break

        delta_ff = [(P_target - p) / s for p, s in zip(P_out_list, slopes)]
        new_fractions = [max(ff + dff, 1e-10) for ff, dff in zip(flow_fraction, delta_ff)]
        total_ff = sum(new_fractions)
        flow_fraction = [ff / total_ff for ff in new_fractions]

    return P_out_list, T_out_list, flow_fraction


def test_parallel_compressible():
    from compressible import Line_Segment
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import AbstractState
    import composition
    from component_classes import ureg
    P_in = ureg.Quantity(1000, "psi").to("Pa").magnitude
    T_in = 300.0   # K
    Q_scfd = ureg.Quantity(30, "mmscf/day")
    print(f'Inlet P:{P_in}, T:{T_in}')
    AS = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
    )
    AS.update(CP.PT_INPUTS, P_in, T_in)

    segment_list = []
    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(3.068, "inch")
    length           = ureg.Quantity(2000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft") #all parallel lines should have identical net elevation changes

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    id_val           = ureg.Quantity(4.026, "inch")
    length           = ureg.Quantity(1000.0, "ft")

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    id_val           = ureg.Quantity(1.939, "inch")
    length           = ureg.Quantity(1500.0, "ft")

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    outlet_press, outlet_temp, flow_fractions = parallel_compressible(segment_list, AS, Q_scfd)
    print(f'outlet pressures = {outlet_press}')
    print(f'outlet temperatures = {outlet_temp}')
    print(f'Flow fractions = {flow_fractions}')


def test_parallel_incompressible():
    from incompressible import Line_Segment, Incompressible_Fluid
    from component_classes import ureg
    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )
    segment_list = []
    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(3.068, "inch")
    length           = ureg.Quantity(2000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft")

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    id_val           = ureg.Quantity(4.026, "inch")
    length           = ureg.Quantity(1000.0, "ft")

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    id_val           = ureg.Quantity(1.939, "inch")
    length           = ureg.Quantity(1500.0, "ft")

    segment_list.append(Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        )
    )

    flow_rate = ureg.Quantity(10000, "oil_bbl/day")
    P0        = ureg.Quantity(100.0, "psi").to("Pa").magnitude

    dP, flow_fractions = parallel_incompressible(segment_list, fluid, flow_rate)
    print(f'dP = {dP}')
    print(f'Flow fractions = {flow_fractions}')


if __name__ == "__main__":
    test_parallel_compressible()