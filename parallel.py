"""
Calculates flow partitioning on parallel line segments
"""
import warnings


def parallel_incompressible(line_segment_list, fluid, total_flow_rate):
    # line_segment_list is a list of at least two parallel branches.  Each
    # branch is one of:
    #   - a Line_Segment instance,
    #   - a fitting instance (Bend, Contraction_Expansion), or
    #   - a list of the above run in series (branch dP = sum of component dPs).
    # Every branch must implement .dP(fluid=..., flow_rate=...) returning Pa,
    # or, for series-list branches, every contained component must.
    # fluid is an incompressible fluid class object
    # total_flow_rate is a flow rate in mass or volumetric flow rate

    #Returns total differential pressure, list of flow fractions on each branch


    def _branch_dP(branch, flow_rate):
        if isinstance(branch, list):
            return sum(c.dP(fluid=fluid, flow_rate=flow_rate) for c in branch)
        return branch.dP(fluid=fluid, flow_rate=flow_rate)

    # Sanity check: parallel branches share inlet and outlet nodes, so each
    # branch's net elevation change must agree.  Fittings carry no elevation
    # data and contribute zero.  Warn (rather than raise) so existing callers
    # aren't broken; physically inconsistent inputs will still produce results
    # that don't match a real piping configuration.
    ELEV_TOL_M = 0.1
    branch_dz = []
    for branch in line_segment_list:
        items = branch if isinstance(branch, list) else [branch]
        dz = 0.0
        for c in items:
            if hasattr(c, "net_elevation_change_m"):
                dz += c.net_elevation_change_m
        branch_dz.append(dz)

    if max(branch_dz) - min(branch_dz) > ELEV_TOL_M:
        warnings.warn(
            f"parallel_incompressible: branch net elevation changes disagree "
            f"by more than {ELEV_TOL_M} m (per-branch dz [m] = "
            f"{[round(dz, 4) for dz in branch_dz]}).  Parallel branches must "
            f"share inlet/outlet elevation; results will not correspond to a "
            f"physically realizable piping layout.",
            stacklevel=2,
        )

    #For initial guess, assume identical friction factors for all line segments and use average cross sectional area, so that resistance to flow simplifies to:
    #dP = f Li/Di * mdoti/(rho^2 * Ai ^2)
    #Resistance proportional to Li/(Ai^2.5) [not exact for noncircular geometry or variable diameters, but close enough for a first guess]
    #For a series branch, sum L_k / A_k^2.5 over its line segments only;
    #fittings contribute negligible length compared to pipe runs and are ignored
    #in the initial guess (they still appear in the iterated dP).
    #Total flow resistance = 1/(sum of 1/individual branch resistances)
    #Xi = flow fraction on branch i = total flow resistance / individual branch resistance
    branch_R = []
    for branch in line_segment_list:
        items = branch if isinstance(branch, list) else [branch]
        R = 0.0
        for c in items:
            if hasattr(c, "total_length_m") and hasattr(c, "volume_m3"):
                avg_area = c.volume_m3 / c.total_length_m
                R += c.total_length_m / avg_area ** 2.5
        branch_R.append(R)

    if all(R > 0.0 for R in branch_R):
        sum_recip_R = sum(1.0 / R for R in branch_R)
        flow_fraction = [(1.0 / R) / sum_recip_R for R in branch_R]
    else:
        # A branch with no line-segment length (only fittings) cannot anchor the
        # initial guess; fall back to an equal split and let Newton converge.
        n = len(line_segment_list)
        flow_fraction = [1.0 / n] * n

    # Iterate flow fractions until dP across every branch converges to a common value.
    # Physics constraint: parallel branches share the same inlet/outlet, so dP_i must
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
            _branch_dP(b, total_flow_rate * ff)
            for b, ff in zip(line_segment_list, flow_fraction)
        ]

        slopes = [
            (_branch_dP(b, total_flow_rate * (ff + EPS)) - dP) / EPS
            for b, ff, dP in zip(line_segment_list, flow_fraction, dP_list)
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
    # line_segment_list is a list of at least two parallel branches.  Each
    # branch is one of:
    #   - a Line_Segment instance (compressible),
    #   - a fitting instance (Bend, Contraction_Expansion), or
    #   - a list of the above run in series (AS is chained outlet-to-inlet
    #     through the components in order).
    # AS is a CoolProp Abstract State pre-updated to the common inlet (P, T).
    # total_flow_rate is a flow rate in mass, molar, or volumetric flow rate

    #Returns list of outlet pressures on each branch, a list of outlet
    #temperatures on each branch, and a list of flow fractions on each branch


    #For initial guess, assume identical friction factors for all line segments and use average cross sectional area, so that resistance to flow simplifies to:
    #dP = f Li/Di * mdoti/(rho^2 * Ai ^2)
    #Resistance proportional to Li/(Ai^2.5) [not exact for noncircular geometry or variable diameters, but close enough for a first guess]
    #For a series branch, sum L_k / A_k^2.5 over its line segments only;
    #fittings contribute negligible length compared to pipe runs and are ignored
    #in the initial guess (they still appear in the iterated outlet calc).
    #Total flow resistance = 1/(sum of 1/individual branch resistances)
    #Xi = flow fraction on branch i = total flow resistance / individual branch resistance
    branch_R = []
    for branch in line_segment_list:
        items = branch if isinstance(branch, list) else [branch]
        R = 0.0
        for c in items:
            if hasattr(c, "total_length_m") and hasattr(c, "volume_m3"):
                avg_area = c.volume_m3 / c.total_length_m
                R += c.total_length_m / avg_area ** 2.5
        branch_R.append(R)

    if all(R > 0.0 for R in branch_R):
        sum_recip_R = sum(1.0 / R for R in branch_R)
        flow_fraction = [(1.0 / R) / sum_recip_R for R in branch_R]
    else:
        n = len(line_segment_list)
        flow_fraction = [1.0 / n] * n

    # Sanity check: parallel branches share inlet/outlet nodes, so each branch's
    # net elevation change must agree.  Fittings carry no elevation and
    # contribute zero.  Warn (rather than raise) so existing callers aren't
    # broken; physically inconsistent inputs will still produce results that
    # don't match a real piping configuration.
    ELEV_TOL_M = 0.1
    branch_dz = []
    for branch in line_segment_list:
        items = branch if isinstance(branch, list) else [branch]
        dz = 0.0
        for c in items:
            if hasattr(c, "net_elevation_change_m"):
                dz += c.net_elevation_change_m
        branch_dz.append(dz)

    if max(branch_dz) - min(branch_dz) > ELEV_TOL_M:
        warnings.warn(
            f"parallel_compressible: branch net elevation changes disagree "
            f"by more than {ELEV_TOL_M} m (per-branch dz [m] = "
            f"{[round(dz, 4) for dz in branch_dz]}).  Parallel branches must "
            f"share inlet/outlet elevation; results will not correspond to a "
            f"physically realizable piping layout.",
            stacklevel=2,
        )

    P0 = AS.p()
    T0 = AS.T()

    # Precompute the phase envelope once.  build_phase_envelope is expensive
    # for multicomponent HEOS mixtures (seconds per call), and the composition
    # does not change across segments or iterations, so we forward the four
    # scalars into every dP_dT call to skip the per-call envelope build.
    from compressible import _build_phase_limits, _safe_update_PT
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    print(f"phase limits: T_cric={T_cric}, P_bar={P_bar}, T_c={T_c}, P_c={P_c}")

    MAX_ITER = 20
    TOL = 1e-6
    EPS = 1e-4  # dimensionless perturbation to each flow fraction for slope estimation

    def _branch_outlet(branch, flow_rate):
        # Reset AS to common inlet conditions before walking the branch.  AS is
        # mutated in place by every dP_dT call, so we must reseed it for each
        # branch evaluation.  Every dP_dT (Line_Segment, Bend,
        # Contraction_Expansion) now reads inlet conditions from AS as
        # provided, so the protocol is uniform: ensure AS is at the upstream
        # state, then call .dP_dT(AS, flow_rate).
        _safe_update_PT(AS, P0, T0, T_cric, P_bar, T_c, P_c)
        items = branch if isinstance(branch, list) else [branch]
        for c in items:
            if hasattr(c, "total_length_m"):
                # Line_Segment: forward precomputed phase-envelope limits.
                c.dP_dT(
                    AS, flow_rate,
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )
            else:
                # Fitting (Bend / Contraction_Expansion).  Forward phase-envelope
                # limits so internal PT updates can apply the same supercritical
                # phase hint -- needed for some mixtures where HEOS phase
                # stability analysis fails.
                c.dP_dT(
                    AS, flow_rate,
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )
        return AS.p(), AS.T()

    def _outlet(branch, ff):
        return _branch_outlet(branch, total_flow_rate * ff)

    P_out_list = []
    T_out_list = []
    P_target = 0.0

    for k in range(MAX_ITER):
        P_out_list = []
        T_out_list = []
        for branch, ff in zip(line_segment_list, flow_fraction):
            P_out, T_out = _outlet(branch, ff)
            P_out_list.append(P_out)
            T_out_list.append(T_out)

        slopes = [
            (_outlet(branch, ff + EPS)[0] - P_out) / EPS
            for branch, ff, P_out in zip(line_segment_list, flow_fraction, P_out_list)
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
    from compressible import Line_Segment, Bend, Contraction_Expansion, _build_phase_limits, _safe_update_PT
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import AbstractState
    import composition
    from component_classes import ureg
    import os
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
        # y_Hydrogen=0.8,
        eos = "PR"
    )

    _safe_update_PT(AS, P_in, T_in, *_build_phase_limits(AS))

    segment_list = []
    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(3.068, "inch")
    # length           = ureg.Quantity(2000.0, "ft")
    # elevation_change = ureg.Quantity(-1.3359375, "m") #all parallel lines should have identical net elevation changes
    csv_path = os.path.join(os.path.dirname(__file__), "testprofile_3inS40.csv")
    Seg_1 = Line_Segment.from_csv(csv_path, roughness=roughness, name = '1')
    segment_list.append(Seg_1)

    Bend_1 = Bend(id_val, 90, 1.5)

    csv_path = os.path.join(os.path.dirname(__file__), "testprofile_ID_OD_WT.csv")

    Seg_2 = Line_Segment.from_csv(csv_path, roughness=roughness, name = '2')

    series_list = [Bend_1, Seg_2]
    segment_list.append(series_list)

    # id_val           = ureg.Quantity(4.026, "inch")
    # length           = ureg.Quantity(1000.0, "ft")

    # segment_list.append(Line_Segment(
    #     roughness=roughness,
    #     id_val=id_val,
    #     length=length,
    #     elevation_change=elevation_change,
    #     )
    # )

    # id_val           = ureg.Quantity(1.939, "inch")
    # length           = ureg.Quantity(1500.0, "ft")

    # segment_list.append(Line_Segment(
    #     roughness=roughness,
    #     id_val=id_val,
    #     length=length,
    #     elevation_change=elevation_change,
    #     )
    # )

    outlet_press, outlet_temp, flow_fractions = parallel_compressible(segment_list, AS, Q_scfd)
    print(f'outlet pressures = {outlet_press}')
    print(f'outlet temperatures = {outlet_temp}')
    print(f'Flow fractions = {flow_fractions}')


def test_parallel_incompressible():
    from incompressible import Line_Segment, Bend, Incompressible_Fluid
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
    Seg_1 = Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        name = '1',
        )

    segment_list.append(Seg_1)

    id_val           = ureg.Quantity(4.026, "inch")
    length           = ureg.Quantity(1000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft")

    series_list = []
    Seg_2 = Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        name = '2',
        )
    series_list.append(Seg_2)

    Bend_1 = Bend(id_val, 90, 1.5)
    series_list.append(Bend_1)

    id_val           = ureg.Quantity(1.939, "inch")
    length           = ureg.Quantity(1500.0, "ft")
    elevation_change = ureg.Quantity(0, "ft")
    Seg_3 = Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
        name = '3',
        )

    series_list.append(Seg_3)

    segment_list.append(series_list)

    flow_rate = ureg.Quantity(10000, "oil_bbl/day")
    P0        = ureg.Quantity(100.0, "psi").to("Pa").magnitude

    dP, flow_fractions = parallel_incompressible(segment_list, fluid, flow_rate)
    print(f'dP = {dP}')
    print(f'Flow fractions = {flow_fractions}')


if __name__ == "__main__":
    test_parallel_compressible()