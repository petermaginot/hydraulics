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
    # be equal for all i. Correction uses the turbulent approximation dP ~ Q^2, giving
    # new_Q_i = Q_i * sqrt(dP_target / dP_i), then renormalize to preserve total flow.
    MAX_ITER = 100
    TOL = 1e-6
    print(f'initial fraction guess: {flow_fraction}')
    dP_list = []
    for k in range(MAX_ITER):
        dP_list = [
            seg.dP(fluid=fluid, flow_rate=total_flow_rate * ff)
            for seg, ff in zip(line_segment_list, flow_fraction)
        ]

        dP_target = sum(ff * dp for ff, dp in zip(flow_fraction, dP_list))

        max_rel_dev = max(abs(dp - dP_target) / abs(dP_target) for dp in dP_list)
        if max_rel_dev < TOL:
            break

        new_fractions = [
            ff * (dP_target / dp) ** 0.5
            for ff, dp in zip(flow_fraction, dP_list)
        ]
        total = sum(new_fractions)
        flow_fraction = [ff / total for ff in new_fractions]
        print(f'iteration {k}, {flow_fraction}, {dP_list}')
    return dP_target, flow_fraction

def test_parallel():
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

    flow_rate = ureg.Quantity(200, "oil_bbl/day")
    P0        = ureg.Quantity(100.0, "psi").to("Pa").magnitude

    dP, flow_fractions = parallel_incompressible(segment_list, fluid, flow_rate)
    print(f'dP = {dP}')
    print(f'Flow fractions = {flow_fractions}')


if __name__ == "__main__":
    test_parallel()