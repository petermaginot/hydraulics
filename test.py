
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition
from component_classes import ureg
import math
import os


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


if __name__ == "__main__":
    
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