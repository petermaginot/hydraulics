"""Throat-state helpers for the single-fitting calculator.

Computes the conditions at the vena-contracta / choke throat for
orifices and reducing valves.  Called by the single-fitting screens
after a successful forward or inverse solve to populate the throat
display in the results panel.

No solver-side signatures are changed; the state is re-derived on a
snapshot AS so the caller's FlowState is not disturbed.
"""

import math

from component_classes import ureg


def _fresh_AS_like(AS):
    """Return a fresh AbstractState with the same backend / composition.

    CoolProp's AbstractState is a Cython object that cannot be deepcopied
    (no __reduce__).  We rebuild it from fluid_names + mole fractions.
    """
    from CoolProp.CoolProp import AbstractState
    backend = AS.backend_name()
    # backend_name() returns e.g. "HelmholtzEOSMixtureBackend"; map common ones.
    if "Helmholtz" in backend:
        backend_arg = "HEOS"
    elif "Peng" in backend or "PR" in backend:
        backend_arg = "PR"
    elif "SRK" in backend:
        backend_arg = "SRK"
    else:
        backend_arg = "HEOS"
    AS_new = AbstractState(backend_arg, "&".join(AS.fluid_names()))
    AS_new.set_mole_fractions(list(AS.get_mole_fractions()))
    return AS_new


def incompressible_throat_state(fitting, fluid, mdot_kgs, P_inlet_Pa):
    """Return throat conditions for an incompressible orifice or reducing valve.

    Args:
        fitting    : incompressible component (Orifice, Valve, etc.)
        fluid      : Incompressible_Fluid
        mdot_kgs   : float, solved mass flow rate [kg/s]
        P_inlet_Pa : float, inlet static pressure [Pa]

    Returns:
        dict with keys ("A_m2", "v_ms", "P_Pa") or None if the component
        has no meaningful throat (e.g. a Bend with no reducer).
    """
    from incompressible import Orifice as _IOrif
    from incompressible import Valve   as _IValve

    rho = fluid.density_si
    Q   = mdot_kgs / rho

    if isinstance(fitting, _IOrif):
        import fluids.flow_meter
        Di = fitting.Di_si
        Do = fitting.Do_si
        A_pipe = math.pi * Di ** 2 / 4.0
        mu = fluid.viscosity_si
        m  = mdot_kgs
        if fitting.Cd_override is not None:
            Cd = fitting.Cd_override
        else:
            try:
                Cd = fluids.flow_meter.C_Reader_Harris_Gallagher(
                    D=Di, Do=Do, rho=rho, mu=mu, m=m, taps=fitting.taps,
                )
            except Exception:
                Cd = 0.61
        A_bore   = math.pi * Do ** 2 / 4.0
        A_throat = Cd * A_bore
        v_throat = Q / A_throat if A_throat > 0 else 0.0
        P_throat = P_inlet_Pa - 0.5 * rho * v_throat ** 2
        return {"A_m2": A_throat, "v_ms": v_throat, "P_Pa": P_throat}

    if isinstance(fitting, _IValve):
        D_min = getattr(fitting, "D_min_si", None)
        Di    = getattr(fitting, "Di_si", None)
        if D_min is None or Di is None or D_min >= Di * 0.9999:
            return None
        A_throat = math.pi * D_min ** 2 / 4.0
        v_throat = Q / A_throat if A_throat > 0 else 0.0
        P_throat = P_inlet_Pa - 0.5 * rho * v_throat ** 2
        return {"A_m2": A_throat, "v_ms": v_throat, "P_Pa": P_throat}

    return None


def compressible_throat_state(fitting, AS, mdot_kgs, P_inlet_Pa, T_inlet_K,
                               phase_limits=None):
    """Return throat conditions for a compressible orifice or reducing valve.

    Re-runs the isentropic acceleration to the throat on a snapshot of the
    AbstractState so the caller's FlowState is not disturbed.

    Args:
        fitting      : compressible component (Orifice, Valve, etc.)
        AS           : CoolProp AbstractState (will be deep-copied — not mutated)
        mdot_kgs     : float, solved mass flow rate [kg/s]
        P_inlet_Pa   : float, inlet static pressure [Pa]
        T_inlet_K    : float, inlet static temperature [K]
        phase_limits : phase-envelope limits dict from _build_phase_limits

    Returns:
        dict with keys ("A_m2", "v_ms", "P_Pa", "T_K", "Ma") or None.
    """
    import CoolProp.CoolProp as CP
    import compressible_flow as cf

    # Determine the throat area.
    import fluids.flow_meter

    throat_area = None

    from compressible_flow import Orifice as _COrif
    from compressible_flow import Valve   as _CValve

    if isinstance(fitting, _COrif):
        Di = fitting.Di_si
        Do = fitting.Do_si
        A_pipe = math.pi * Di ** 2 / 4.0
        A_bore = math.pi * Do ** 2 / 4.0
        # Estimate Cd at inlet conditions for the throat area.
        try:
            AS_snap = _fresh_AS_like(AS)
            AS_snap.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
            rho_in = AS_snap.rhomass()
            mu     = AS_snap.viscosity()
            if fitting.Cd_override is not None:
                Cd = fitting.Cd_override
            else:
                Cd = fluids.flow_meter.C_Reader_Harris_Gallagher(
                    D=Di, Do=Do, rho=rho_in, mu=mu, m=mdot_kgs, taps=fitting.taps,
                )
        except Exception:
            Cd = 0.61
        throat_area = Cd * A_bore
        inlet_area  = A_pipe

    elif isinstance(fitting, _CValve):
        D_min = getattr(fitting, "D_min_si", None)
        Di    = getattr(fitting, "Di_si", None)
        if D_min is None or Di is None or D_min >= Di * 0.9999:
            return None
        throat_area = math.pi * D_min ** 2 / 4.0
        inlet_area  = math.pi * Di ** 2 / 4.0

    else:
        return None

    if phase_limits is not None:
        T_cric, P_bar, T_c, P_c = phase_limits
    else:
        T_cric = P_bar = T_c = P_c = None

    # First try the choke check.  choked_mass_flux returns the mass flow at
    # which Ma=1 at A_throat for the inlet stagnation state, plus the throat
    # static (P, T, rho).  If our solved mdot meets or exceeds it, the
    # throat is sonic -- the subsonic root of compressible_changing_area_K
    # degenerates to Ma=1 here and won't converge cleanly.
    AS_choke = _fresh_AS_like(AS)
    AS_choke.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
    fs_choke = cf.FlowState(
        AS_choke, mdot=mdot_kgs, A=inlet_area, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
    try:
        mdot_chk, P_thr_chk, T_thr_chk, rho_thr_chk, _, _ = cf.choked_mass_flux(
            fs=fs_choke, A_throat=throat_area,
        )
    except RuntimeError:
        # _NoChokeBracketError -- geometry cannot reach Mach 1 along the
        # accessible isentrope.  Fall through to subsonic march.
        mdot_chk = float("inf")
        P_thr_chk = T_thr_chk = rho_thr_chk = None

    # Tolerate a small overshoot from the solver's choke cap (~0.1 %).
    if mdot_kgs >= mdot_chk * (1.0 - 1.0e-3):
        v_thr = mdot_kgs / (rho_thr_chk * throat_area) if rho_thr_chk else 0.0
        AS_son = _fresh_AS_like(AS)
        AS_son.update(CP.PT_INPUTS, P_thr_chk, T_thr_chk)
        try:
            a_thr = AS_son.speed_sound()
            Ma    = v_thr / a_thr if a_thr > 0 else 1.0
        except (ValueError, RuntimeError):
            Ma = 1.0
        return {
            "A_m2": throat_area,
            "v_ms": v_thr,
            "P_Pa": P_thr_chk,
            "T_K":  T_thr_chk,
            "Ma":   Ma,
        }

    # Subsonic: isentropically accelerate from inlet to A_throat.
    AS_snap = _fresh_AS_like(AS)
    AS_snap.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
    fs = cf.FlowState(
        AS_snap, mdot=mdot_kgs, A=inlet_area, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
    cf.compressible_changing_area_K(fs, A_out=throat_area, K=0.0)
    return {
        "A_m2": throat_area,
        "v_ms": fs.v,
        "P_Pa": fs.P,
        "T_K":  fs.T,
        "Ma":   fs.Ma,
    }
