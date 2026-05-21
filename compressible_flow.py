"""compressible_flow.py

Single-phase Compressible (gas) pipeline hydraulics.

This module provides component classes and helper functions for computing
pressure and temperature profiles along gas pipelines.  CoolProp AbstractState
objects are used throughout for real-gas equation-of-state calculations, so
any fluid or mixture supported by CoolProp can be used without modification.

The component classes (Line_Segment, Bend, Contraction_Expansion) inherit
geometry storage and CSV-loading logic from the base classes in
component_classes.py, adding compressible-flow dP_dT() methods that update a
CoolProp AbstractState in-place as conditions evolve along the pipe.

Classes
-------
Line_Segment  (inherits Base_Line_Segment)
    Adds dP_dT() for compressible flow.  Steps through consecutive profile
    point pairs via compressible_pipe_segment(), applying isentropic
    area-change corrections at inter-slice boundaries.  Updates the
    AbstractState in place and returns a list of (distance, pressure,
    temperature, velocity) tuples for profile plotting.

Bend  (inherits Base_Bend)
    Adds dP_dT() using fluids.fittings.bend_rounded() to obtain K, then
    delegates to compressible_K().

Valve  (inherits Base_Valve)
    Adds dP_dT() using the pre-computed K-factor stored on the instance,
    delegating to compressible_K().

CheckValve  (inherits Base_CheckValve)
    Adds dP_dT() identical to Valve's, using the forward-flow K-factor
    stored on the instance.  Carries check_valve=True from the base class,
    so network._reversed_component substitutes K=1e9 under reverse flow
    and the valve presents near-infinite resistance.

Contraction_Expansion  (inherits Base_Contraction_Expansion)
    Adds dP_dT() using fluids.fittings.contraction_sharp() or
    diffuser_sharp() to obtain K, then delegates to
    compressible_changing_area_K().

Module-level functions
----------------------
_resolve_mdot(flow_rate, abstract_state)
    Convert a pint Quantity flow rate (mass, molar, standard-volume, or
    actual volumetric) to mass flow rate [kg/s].

viscosity_LGE(T, mol_wt, density)
    Lee-Gonzalez-Eakin correlation for hydrocarbon gas viscosity.  Used as
    a fallback when the chosen equation of state does not support viscosity
    (e.g. Peng-Robinson).

_build_phase_limits(AS)
    Build the phase envelope on a scratch AbstractState and return
    (T_cricondentherm, P_cricondenbar, T_critical, P_critical).

_safe_update_PT(AS, P, T, ...)
    Call AS.update(PT_INPUTS, P, T) with an explicit phase hint when the
    state is determinably outside the two-phase region, bypassing CoolProp's
    internal phase-stability analysis to avoid false two-phase detection.

compressible_changing_area(abstract_state, mdot, A_in, A_out)
    Isentropic pressure and temperature correction for a compressible fluid
    passing through a change in flow area.  Uses the area-Mach relation to
    find the subsonic outlet Mach number, then recovers static conditions
    from total-condition ratios.

compressible_changing_area_K(abstract_state, mdot, A_in, A_out, K)
    Outlet conditions for an area change with a known loss coefficient K.
    Solves the simultaneous stagnation-enthalpy and entropy-generation
    balance equations via scipy.optimize.root, using the isentropic result
    as the initial guess.

compressible_K(abstract_state, mdot, flow_area, K)
    Outlet conditions for a constant-area fitting with a known loss
    coefficient K.  Applies a single-step analytical result from the
    combined energy, continuity, entropy, and EOS derivation, then corrects
    temperature to satisfy the stagnation-enthalpy balance.

compressible_pipe_segment(abstract_state, mdot, dL, dz, D_h, roughness,
                          flow_area, ...)
    Core compressible pipe-flow integration over a single pipe slice.
    Solves coupled dP/dL and dT/dL ODEs (or the isothermal dP/dL equation)
    using a forward-Euler step with a one-iteration energy-balance
    correction.  Updates the AbstractState in-place to outlet conditions.

downsample_profile(profile, max_step_m=1000.0, ...)
    Reduce a dense pipe profile to the points that actually matter for
    compressible flow: diameter changes, polyline slope breaks, and a
    spacing cap.  Returns a new profile list in the same 4-tuple format.
"""

import csv
import math
import os
import warnings
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import fluids.fittings
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition
from component_classes import (
    Base_Line_Segment,
    Base_Bend,
    Base_Contraction_Expansion,
    Base_Valve,
    Base_CheckValve,
    downsample_profile,
    ureg,
)


class Line_Segment(Base_Line_Segment):
    """Pipe segment with compressible-flow pressure and temperature calculation.

    Inherits geometry storage, CSV loading, and convenience properties from
    Base_Line_Segment.  Adds dP_dT() for compressible flow, stepping through
    consecutive profile slices via compressible_pipe_segment() and applying
    isentropic area-change corrections at inter-slice boundaries.

    Constructor arguments and behavior are identical to Base_Line_Segment.
    See Base_Line_Segment for full argument documentation.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        isothermal=False,
        q_wall=0.0,
        mu=None,
        T_cricondentherm=None,
        P_cricondenbar=None,
        T_critical=None,
        P_critical=None,
        energy_tol=10.0,
        dPdL_rel_tol=0.05,
        max_split_depth=8,
        verbose=False,
    ):
        """Calculate outlet pressure and temperature for compressible flow
        through the segment.

        Steps through consecutive profile point pairs, calling
        compressible_pipe_segment() for each slice and applying isentropic
        area-change corrections at boundaries where the flow area changes.

        Heat input q_wall is distributed uniformly per unit pipe length across
        all slices.

        Args:
            abstract_state : CoolProp AbstractState pre-configured for the
                             working fluid AND already updated to the segment
                             inlet (P, T).  Read in-place to obtain the inlet
                             conditions, and mutated in place during
                             integration; must not be shared across threads.
            flow_rate      : pint Quantity -- mass ([mass]/[time]), molar or
                             standard-volume ([substance]/[time], e.g. mol/s,
                             mmscf/day), or actual volumetric
                             ([length]^3/[time]) flow rate.
            isothermal     : bool, if True temperature is held constant
                             through each slice.  Default False.
            q_wall         : float, total heat input to the fluid over the
                             entire segment [W].  Distributed uniformly per
                             unit length.  Ignored when isothermal=True.
                             Default 0.0 (adiabatic).
            mu             : float or None, viscosity [Pa*s] forwarded to
                             compressible_pipe_segment() for every slice.  If
                             None (default), CoolProp is queried at each slice
                             and falls back to Lee-Gonzalez-Eakin on failure.
            T_cricondentherm,
            P_cricondenbar,
            T_critical,
            P_critical     : optional precomputed phase-envelope limits [K, Pa].
                             If all four are supplied, the internal call to
                             _build_phase_limits is skipped.  Useful when
                             dP_dT is invoked many times on the same fluid
                             (e.g. parallel-flow iteration), since
                             build_phase_envelope is expensive for
                             multicomponent HEOS mixtures.  Default None
                             (envelope is built on each call).
            energy_tol     : float, stagnation-enthalpy residual tolerance
                             [J/kg] forwarded to compressible_pipe_segment()'s
                             adaptive splitter.  Default 10.0.  Ignored when
                             isothermal=True.
            dPdL_rel_tol   : float, relative dP/dL-change tolerance forwarded
                             to compressible_pipe_segment()'s adaptive
                             splitter.  Default 0.05 (5%).
            max_split_depth: int, maximum recursive bisection depth permitted
                             per profile slice.  Default 8 (=256x refinement).

        Returns:
            profile_points: a list of (distance_m, pressure_Pa, temperature_K,
            velocity_ms) tuples, one per profile point (inlet through outlet),
            suitable for constructing pressure, temperature, or velocity
            profile plots.  abstract_state is updated in place to outlet
            conditions.

        Raises:
            ValueError   : if the profile has fewer than two points or
                           flow_rate dimensions are unrecognized.
            RuntimeError : if two-phase conditions, choked flow, or a
                           CoolProp failure occur during integration.
        """
        if len(self.profile) < 2:
            raise ValueError(
                "Line_Segment.dP_dT: profile must have at least two points."
            )

        AS = abstract_state
        # If the caller supplied any limits, trust them and skip the rebuild --
        # _build_phase_limits is allowed to return Nones for fields it couldn't
        # compute (e.g. envelope failed but critical point succeeded), and we
        # don't want to retry on every call.
        if (
            T_cricondentherm is None
            and P_cricondenbar is None
            and T_critical is None
            and P_critical is None
        ):
            T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
        else:
            T_cric, P_bar, T_c, P_c = (
                T_cricondentherm,
                P_cricondenbar,
                T_critical,
                P_critical,
            )
        # Inlet conditions come from AS as the caller provided it; consistent
        # with Bend.dP_dT and Contraction_Expansion.dP_dT, which also assume
        # the caller has updated AS to the inlet (P, T) before calling.
        P0   = AS.p()
        T0   = AS.T()
        mdot = _resolve_mdot(flow_rate, AS)

        total_length  = self.total_length_m
        q_per_length  = q_wall / total_length if total_length > 0.0 else 0.0

        _AREA_TOL = 1e-6   # fractional area-change threshold
        n     = len(self.profile)

        # Record inlet conditions at profile point 0.
        dist0, _elev0, _D_h0, area0 = self.profile[0]
        v0 = mdot / (AS.rhomass() * area0)
        profile_points = [(dist0, P0, T0, v0)]

        for i in range(n - 1):

            dist_in,  elev_in,  D_h_in,  area_in  = self.profile[i]
            dist_out, elev_out, D_h_out, area_out = self.profile[i + 1]

            dL      = dist_out - dist_in   # m, along-pipe slice length
            dz      = elev_out - elev_in   # m, elevation rise (positive = uphill)
            q_slice = q_per_length * dL    # W, heat for this slice

            compressible_pipe_segment(
                abstract_state=AS,
                mdot=mdot,
                dL=dL,
                dz=dz,
                D_h=D_h_in,
                roughness=self.roughness_si,
                flow_area=area_in,
                isothermal=isothermal,
                q_wall=q_slice,
                mu=mu,
                T_cricondentherm=T_cric,
                P_cricondenbar=P_bar,
                T_critical=T_c,
                P_critical=P_c,
                energy_tol=energy_tol,
                dPdL_rel_tol=dPdL_rel_tol,
                _max_split_depth=max_split_depth,
            )

            # Area-change correction at the boundary to the next slice.
            area_ratio = abs(area_out - area_in) / max(area_in, area_out)
            if area_ratio > _AREA_TOL:
                compressible_changing_area_K(
                    AS, mdot, area_in, area_out, K=0.0,
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )

            # Record conditions at this profile point after all corrections.
            P_cur   = AS.p()
            T_cur   = AS.T()
            v_cur   = mdot / (AS.rhomass() * area_out)
            profile_points.append((dist_out, P_cur, T_cur, v_cur))
            if verbose:
                msg = str(f"Segment {self.name} Step: {i+1} of {n-1}, P = {P_cur}, T = {T_cur}")
                print(msg, end="\r")
        if verbose:
            print(f" "*len(msg), end="\r")
        return profile_points


class Bend(Base_Bend):
    """Rounded pipe bend fitting with compressible pressure/temperature
    calculation.

    Modeled as adiabatic.  Inherits geometry storage and validation from
    Base_Bend.

    Constructor arguments are identical to Base_Bend:
        Di        : pint Quantity or float (m if float).  Pipe inner diameter.
        ang_deg   : float.  Bend angle [degrees].
        bend_dias : float.  Bend radius as a multiple of Di.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        mu=None,
        T_cricondentherm=None,
        P_cricondenbar=None,
        T_critical=None,
        P_critical=None,
    ):
        """Outlet conditions for a compressible fluid passing through the bend.

        The caller must update abstract_state to the inlet (P, T) conditions
        before calling.  Uses fluids.fittings.bend_rounded() to obtain K, then
        delegates to compressible_K().

        Args:
            abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                             by the caller.  Updated in-place on return.
            flow_rate      : pint Quantity -- mass, molar, or volumetric flow.
            mu             : float or None, viscosity [Pa*s] used in the
                             Reynolds number calculation.  If None (default),
                             CoolProp is queried and falls back to
                             Lee-Gonzalez-Eakin on failure.
            T_cricondentherm,
            P_cricondenbar,
            T_critical,
            P_critical     : optional precomputed phase-envelope limits, forwarded
                             to compressible_K() so PT updates can specify a
                             supercritical phase hint and bypass HEOS phase
                             stability analysis (which fails for some mixtures).
        Returns:
            None.  abstract_state is updated in place to outlet conditions.
        """
        AS   = abstract_state
        mdot = _resolve_mdot(flow_rate, AS)

        Di = self.Di_si
        A  = math.pi * Di ** 2 / 4.0

        rho_in = AS.rhomass()
        if mu is None:
            try:
                mu = AS.viscosity()
            except Exception:
                mu = viscosity_LGE(AS.T(), AS.molar_mass() * 1000.0, rho_in)

        v_in = mdot / (rho_in * A)
        Re   = fluids_Reynolds(V=v_in, D=Di, rho=rho_in, mu=mu)
        K    = fluids.fittings.bend_rounded(
            Di=Di,
            bend_diameters=self.bend_dias,
            angle=self.ang_deg,
            Re=Re,
        )

        compressible_K(
            AS, mdot, A, K,
            T_cricondentherm=T_cricondentherm,
            P_cricondenbar=P_cricondenbar,
            T_critical=T_critical,
            P_critical=P_critical,
        )


class Valve(Base_Valve):
    """Valve fitting with compressible pressure/temperature calculation.

    Modeled as adiabatic.  Inherits geometry storage and validation from
    Base_Valve.  Uses the pre-computed K-factor stored on the instance, so
    no Reynolds-number or correlation lookup is performed.

    Constructor arguments are identical to Base_Valve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Resistance coefficient (K-factor), referenced to the
             pipe velocity head.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        T_cricondentherm=None,
        P_cricondenbar=None,
        T_critical=None,
        P_critical=None,
    ):
        """Outlet conditions for a compressible fluid passing through the valve.

        The caller must update abstract_state to the inlet (P, T) conditions
        before calling.  Delegates directly to compressible_K() using the
        K-factor stored on the instance.

        Args:
            abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                             by the caller.  Updated in-place on return.
            flow_rate      : pint Quantity -- mass, molar, or volumetric flow.
            T_cricondentherm,
            P_cricondenbar,
            T_critical,
            P_critical     : optional precomputed phase-envelope limits, forwarded
                             to compressible_K() so PT updates can specify a
                             supercritical phase hint and bypass HEOS phase
                             stability analysis (which fails for some mixtures).

        Returns:
            None.  abstract_state is updated in place to outlet conditions.
        """
        AS   = abstract_state
        mdot = _resolve_mdot(flow_rate, AS)

        Di = self.Di_si
        A  = math.pi * Di ** 2 / 4.0

        compressible_K(
            AS, mdot, A, self.K,
            T_cricondentherm=T_cricondentherm,
            P_cricondenbar=P_cricondenbar,
            T_critical=T_critical,
            P_critical=P_critical,
        )


# When network._reversed_component substitutes K = _SEALING_K (~ 1e9) on a
# check-valve copy for a reverse-flow walk, the inertial compressible_K()
# formula drives P_out and T_out to nonsense values (~ -5e9 Pa) that
# CoolProp can't update against.  CheckValve.dP_dT short-circuits to a
# clamped sealed-state outlet whenever K >= _SEALED_K_THRESHOLD: P_out is
# pulled to _SEALED_P_OUT_FRACTION * P_in (floored at _SEALED_P_FLOOR_PA so
# downstream components on the same edge don't see a sub-triple-point
# state), T_out is left at T_in.  The big walked_outlet_P - P_node residual
# still drives the network solver back toward mdot >= 0.
_SEALED_K_THRESHOLD     = 1.0e6   # K above this is taken to be the sealing shadow
_SEALED_P_OUT_FRACTION  = 0.5     # outlet P as fraction of inlet P
_SEALED_P_FLOOR_PA      = 1.0e5   # absolute floor on outlet P [Pa]


def _is_sealed_check_valve(component):
    """True if `component` is a check-valve shadow carrying the sealing K
    (substituted by network._reversed_component for reverse flow).  Used by
    the compressible network walk to short-circuit any edge whose path
    contains a sealed CV: walking downstream components at the clamped
    sealed-state outlet (P pulled to a fraction of inlet P) puts them into
    a low-density / high-velocity regime where compressible_pipe_segment
    can fail to converge.
    """
    return (
        getattr(component, "check_valve", False)
        and getattr(component, "K", 0.0) >= _SEALED_K_THRESHOLD
    )


def _sealed_outlet_PT(P_in, T_in):
    """Return (P_out, T_out) for a sealed-edge clamped outlet state.

    Used by both CheckValve.dP_dT and the compressible network walk so the
    clamp formula lives in exactly one place.
    """
    return (max(_SEALED_P_OUT_FRACTION * P_in, _SEALED_P_FLOOR_PA), T_in)


class CheckValve(Base_CheckValve):
    """Check valve with compressible pressure/temperature calculation.

    Forward flow uses the K-factor stored on the instance and delegates to
    compressible_K().  Reverse flow is handled upstream by
    network._reversed_component, which returns a shallow copy with K replaced
    by _SEALING_K (~ 1e9); dP_dT() detects that case (K >= _SEALED_K_THRESHOLD)
    and short-circuits to a clamped sealed-state outlet instead of running
    the inertial formula, which would otherwise produce an unphysical
    negative P_out that crashes CoolProp.

    Constructor arguments are identical to Base_CheckValve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Forward-flow K-factor.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        T_cricondentherm=None,
        P_cricondenbar=None,
        T_critical=None,
        P_critical=None,
    ):
        """Outlet conditions for a compressible fluid passing through the
        check valve.

        The caller must update abstract_state to the inlet (P, T) conditions
        before calling.  Forward flow (physical K) delegates to
        compressible_K().  Reverse flow (sealing K substituted by
        _reversed_component) is short-circuited to a clamped sealed-state
        outlet, see module-level _SEALED_* constants.

        Args:
            abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                             by the caller.  Updated in-place on return.
            flow_rate      : pint Quantity -- mass, molar, or volumetric flow.
            T_cricondentherm,
            P_cricondenbar,
            T_critical,
            P_critical     : optional precomputed phase-envelope limits,
                             forwarded to compressible_K() / _safe_update_PT().

        Returns:
            None.  abstract_state is updated in place to outlet conditions.
        """
        AS   = abstract_state
        mdot = _resolve_mdot(flow_rate, AS)

        if self.K >= _SEALED_K_THRESHOLD:
            P_out, T_out = _sealed_outlet_PT(AS.p(), AS.T())
            _safe_update_PT(
                AS, P_out, T_out,
                T_cricondentherm, P_cricondenbar, T_critical, P_critical,
            )
            return

        Di = self.Di_si
        A  = math.pi * Di ** 2 / 4.0

        compressible_K(
            AS, mdot, A, self.K,
            T_cricondentherm=T_cricondentherm,
            P_cricondenbar=P_cricondenbar,
            T_critical=T_critical,
            P_critical=P_critical,
        )


class Contraction_Expansion(Base_Contraction_Expansion):
    """Abrupt contraction or expansion with compressible pressure/temperature
    calculation.

    Modeled as adiabatic.  Inherits geometry storage and validation from
    Base_Contraction_Expansion.  The dP_dT() method is not yet implemented.

    Constructor arguments are identical to Base_Contraction_Expansion:
        Di_US : pint Quantity or float (m if float).  Upstream inner diameter.
        Di_DS : pint Quantity or float (m if float).  Downstream inner diameter.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        T_cricondentherm=None,
        P_cricondenbar=None,
        T_critical=None,
        P_critical=None,
    ):
        """Outlet abstract state for a compressible fluid passing through the
        contraction/expansion.

        The caller must update abstract_state to the inlet (P, T) conditions
        before calling.

        Uses fluids.fittings.contraction_sharp() or diffuser_sharp() to obtain
        the K-factor, then calls compressible_changing_area_K() with that K
        referenced to the upstream (inlet) velocity head.

        Args:
            abstract_state : CoolProp AbstractState instance, pre-updated to
                             inlet (P, T) by the caller.  Updated in-place on
                             return to outlet conditions.
            flow_rate      : pint Quantity -- mass, molar, or volumetric flow.
            T_cricondentherm,
            P_cricondenbar,
            T_critical,
            P_critical     : optional precomputed phase-envelope limits, forwarded
                             to compressible_changing_area_K() for the
                             supercritical phase hint.

        Returns:
            None.  abstract_state is updated in place to outlet conditions.
        """
        AS   = abstract_state
        mdot = _resolve_mdot(flow_rate, AS)

        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        if abs(Di_US - Di_DS) < 1e-12:
            return

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0

        if Di_US > Di_DS:
            # Contraction: fluids returns K w.r.t. downstream; convert to upstream.
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K    = K_ds * (A_DS / A_US) ** 2
        else:
            # Expansion: fluids returns K w.r.t. upstream velocity directly.
            K = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)

        compressible_changing_area_K(
            AS, mdot, A_US, A_DS, K,
            T_cricondentherm=T_cricondentherm,
            P_cricondenbar=P_cricondenbar,
            T_critical=T_critical,
            P_critical=P_critical,
        )


# downsample_profile is defined in component_classes and imported above.

# ---------------------------------------------------------------------------
# Flow-rate helper
# ---------------------------------------------------------------------------

def _resolve_mdot(flow_rate, abstract_state):
    """Convert a pint Quantity flow rate to mass flow rate [kg/s].

    Accepts:
      - Mass flow:           [mass]/[time]        e.g. kg/s, lb/hr
      - Molar / std-volume:  [substance]/[time]   e.g. mol/s, scf/day, mmscf/day
        Standard-volume units (scf, mmscf, scm) are defined as mol equivalents
        in the unit registry, so they fall into this branch automatically.
      - Actual volumetric:   [length]^3/[time]    e.g. m^3/s, ft^3/min
        Requires abstract_state to be updated to the relevant (P, T) so that
        rhomass() returns the correct in-situ density.

    Args:
        flow_rate      : pint Quantity.
        abstract_state : CoolProp AbstractState, used for molar_mass() or
                         rhomass() when the input is not already in kg/s.

    Returns:
        float, mass flow rate [kg/s].

    Raises:
        ValueError : if the dimensionality of flow_rate is not recognized.
    """
    dim = flow_rate.dimensionality
    if dim == {"[mass]": 1, "[time]": -1}:
        return flow_rate.to("kg/s").magnitude
    elif dim == {"[substance]": 1, "[time]": -1}:
        return flow_rate.to("mol/s").magnitude * abstract_state.molar_mass()
    elif dim == {"[length]": 3, "[time]": -1}:
        rho = abstract_state.rhomass()
        return flow_rate.to("m^3/s").magnitude * rho
    else:
        raise ValueError(
            f"flow_rate has unrecognized dimensions {dict(dim)}.  "
            "Expected [mass]/[time] (kg/s, …), [substance]/[time] (mol/s, "
            "scf/day, mmscf/day, …), or [length]^3/[time] (m^3/s, …)."
        )


# ---------------------------------------------------------------------------
# Compressible component classes
# ---------------------------------------------------------------------------

# Unit-system definitions for compressible profile export.
# Each entry maps a system name to conversion factors and column header labels.
# Conversions are applied as: output_value = SI_value * factor  (except
# temperature, which uses an offset conversion via pint).
_COMP_OUTPUT_UNITS = {
    "US_Common": {
        "dist_label":    "distance_ft",
        "elev_label":    "elevation_ft",
        "P_label":       "P_psia",
        "T_label":       "T_degF",
        "v_label":       "v_fps",
        "q_wall_label":  "q_wall_W",      #  watts -- TODO should change this to btu/hr
        "dist_unit":     "ft",
        "elev_unit":     "ft",
        "P_unit":        "psi",
        "T_unit":        "degF",
        "v_unit":        "ft/s",
    },
    "SI": {
        "dist_label":    "distance_m",
        "elev_label":    "elevation_m",
        "P_label":       "P_Pa",
        "T_label":       "T_K",
        "v_label":       "v_ms",
        "q_wall_label":  "q_wall_W",
        "dist_unit":     "m",
        "elev_unit":     "m",
        "P_unit":        "Pa",
        "T_unit":        "K",
        "v_unit":        "m/s",
    },
    "metric": {
        "dist_label":    "distance_m",
        "elev_label":    "elevation_m",
        "P_label":       "P_kPa",
        "T_label":       "T_degC",
        "v_label":       "v_ms",
        "q_wall_label":  "q_wall_W",
        "dist_unit":     "m",
        "elev_unit":     "m",
        "P_unit":        "kPa",
        "T_unit":        "degC",
        "v_unit":        "m/s",
    },
}

def viscosity_LGE(T, mol_wt, density):
    """
    Lee, Gonzalez, and Eakin correlation for hydrocarbon gas viscosity
    Use this for gas viscosity if using a CoolProp equation of state that doesn't support viscosity calculation (like Peng-Robinson)
    T = Temperature, degrees K
    mol_wt = gas molecular weight, kg/kmol
    density = kg/m^3

    Uses the pint library for unit conversion
    """
    T = ureg.Quantity(T, "degK").to("degR").magnitude
    density = ureg.Quantity(density, "kg/m^3").to("g/cm^3").magnitude
    
    #Temperature input to equation is in degrees R, density input is g/cm^3, returns centipoise. Good gravy those are some ridiculous units.

    x = 3.5+986/(T)+0.01*mol_wt
    k = (9.4+0.02*mol_wt)*((T)**1.5)/(209+19*mol_wt+T)
    y = 2.4 - 0.2 * x
    
    mu = k * math.exp(x * density ** y)/10000.0

    return ureg.Quantity(mu, "cP").to("Pa*s").magnitude


# CoolProp integer phase codes returned by AbstractState.phase().
# Retained here as named constants so comparisons read clearly.
_CP_PHASE_LIQUID               = CP.iphase_liquid               # 0
_CP_PHASE_SUPERCRITICAL        = CP.iphase_supercritical         # 1
_CP_PHASE_SUPERCRITICAL_GAS    = CP.iphase_supercritical_gas     # 2
_CP_PHASE_SUPERCRITICAL_LIQUID = CP.iphase_supercritical_liquid  # 3
_CP_PHASE_GAS                  = CP.iphase_gas                   # 5
_CP_PHASE_TWOPHASE             = CP.iphase_twophase              # 6

# Phases that are acceptable for single-phase hydraulics.
_SINGLE_PHASE_CODES = frozenset([
    _CP_PHASE_LIQUID,
    _CP_PHASE_GAS,
    _CP_PHASE_SUPERCRITICAL,
    _CP_PHASE_SUPERCRITICAL_GAS,
    _CP_PHASE_SUPERCRITICAL_LIQUID,
])


def _build_phase_limits(AS, verbose=False):
    """Return (T_cricondentherm, P_cricondenbar, T_critical, P_critical) [K, Pa].
    This function builds a phase envelope for a CoolProp abstract state to calculate the critical properties to aid in determining if
    a pressure/temperature combination is obviously in a single phase state or not by comparing it to the critical pressure/temperature 
    and/or cricondenbar and cricondentherm. This enables us to pass a phase hint to CoolProp's
    abstract state update function, which increases its speed appreciably. If you don't supply a phase hint, CoolProp needs
    to determine the phase with every update calculation. Its routine for determining the phase also fails to converge for some cases 
    even though it is clearly in a single phase region. This process helps avoid that error from rearing its head.

    The increase in calculation speed is very helpful when you need to perform thousands of abstract state updates to iterate over a 
    segment or solve a convergence problem. 

    This function builds the phase envelope on a temporary abstract state so the working abstract state's internal
    solver state is not corrupted.  CoolProp's build_phase_envelope leaves the AbstractState at the last envelope point it visited; subsequent update()
    calls on the same object then fail unpredictably.

    The envelope tracer is fragile (some HEOS mixtures and the PR backend can
    fail to converge), so the critical-point query is wrapped separately.  When
    the envelope fails but the critical point succeeds, returns
    (None, None, T_critical, P_critical) -- callers can still use the critical
    point as a coarser phase hint via _safe_update_PT.

    Returns (None, None, None, None) only when both queries fail.
    """
    AS_tmp = AbstractState("HEOS", "&".join(AS.fluid_names()))
    AS_tmp.set_mole_fractions(list(AS.get_mole_fractions()))
    if verbose:
        msg = str('Building phase limits - this can take a while')
        print(msg, end='\r')
    try:
        AS_tmp.build_phase_envelope("")
        PE = AS_tmp.get_phase_envelope_data()
        T_cric = max(PE.T)
        P_bar  = max(PE.p)
    except Exception:
        T_cric, P_bar = None, None

    try:
        T_c = AS_tmp.T_critical()
        P_c = AS_tmp.p_critical()
    except Exception:
        T_c, P_c = None, None

    if verbose:
        print(f" "*len(msg), end="\r")

    return T_cric, P_bar, T_c, P_c


def _safe_update_PT(AS, P, T, T_cricondentherm=None, P_cricondenbar=None,
                    T_critical=None, P_critical=None):
    """Call AS.update(PT_INPUTS, P, T) with an explicit phase hint when the
    phase is determinable from the phase envelope limits.

    CoolProp's HEOS mixture backend runs an internal phase stability analysis
    before solving for density.  That analysis can fail numerically at conditions
    that are outside but near the phase envelope (false two-phase detection),
    producing 'No density solutions'.  Supplying an explicit phase bypasses the
    stability analysis entirely.

    Phase selection rules (applied in order):
      T > T_cricondentherm, P > P_critical  → iphase_supercritical
      T > T_cricondentherm, P <= P_critical → iphase_supercritical_gas
      P > P_cricondenbar                    → iphase_supercritical
      T > T_critical, P > P_critical        → iphase_supercritical    (fallback)
      T > T_critical, P <= P_critical       → iphase_supercritical_gas (fallback)
      otherwise                             → no hint; CoolProp determines phase

    The two trailing fallback rules apply only when T_cricondentherm /
    P_cricondenbar are unavailable (e.g. the envelope tracer failed for the
    mixture).  They use just the mixture critical point.  This is *coarser*
    than the cricondentherm bound -- there is a small region above T_critical
    but below T_cricondentherm where a multicomponent mixture can still be
    two-phase -- so callers near the envelope should still prefer the full
    cricondentherm/cricondenbar limits when available.
    """
    phase = None
    if T_cricondentherm is not None and T > T_cricondentherm:
        if P_critical is not None and P > P_critical:
            phase = CP.iphase_supercritical
        else:
            phase = CP.iphase_supercritical_gas
    elif P_cricondenbar is not None and P > P_cricondenbar:
        phase = CP.iphase_supercritical
    elif (
        T_cricondentherm is None
        and P_cricondenbar is None
        and T_critical is not None
        and P_critical is not None
        and T > T_critical
    ):
        phase = CP.iphase_supercritical if P > P_critical else CP.iphase_supercritical_gas

    if phase is not None:
        AS.specify_phase(phase)
        try:
            AS.update(CP.PT_INPUTS, P, T)
        finally:
            AS.unspecify_phase()
    else:
        try:
            AS.update(CP.PT_INPUTS, P, T)
        except ValueError as exc:
            msg = f"CoolProp PT update failed at P={P:.4g} Pa, T={T:.4g} K"
            if T_cricondentherm is not None:
                msg += (
                    f"; conditions are within the possible two-phase region "
                    f"(T_cricondentherm={T_cricondentherm:.4g} K, "
                    f"P_cricondenbar={P_cricondenbar:.4g} Pa)"
                )
            raise RuntimeError(msg) from exc

def compressible_changing_area(abstract_state, mdot, A_in, A_out):
    """Isentropic pressure and temperature correction for a ideal gas compressible fluid
    passing through a change in flow area.

    The caller must update abstract_state to the inlet (P, T) conditions before
    calling.  On return, abstract_state is NOT updated to the outlet
    (P_out, T_out) conditions - the caller must do this manually based on the returned P & T. 

    Uses the isentropic area-Mach relation to find the
    outlet Mach number satisfying continuity on the same isentropic curve, then
    recovers outlet static conditions from total-condition ratios.  
    The heat-capacity ratio gamma is obtained from the
    AbstractState at inlet conditions. Note that this is only valid for an ideal gas - this function
    is only used for an initial guess for the non-ideal solver function.

    Area-Mach relation (from https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html):

        A / A* = (1/M) * {[2/(gamma+1)] * [1 + (gamma-1)/2 * M^2]}
                         ^ [(gamma+1) / (2*(gamma-1))]

    Total-condition ratios (Eqs #6, #7 from NASA):

        P / P_total = [1 + (gamma-1)/2 * M^2] ^ [-gamma/(gamma-1)]
        T / T_total = [1 + (gamma-1)/2 * M^2] ^ [-1]

    Args:
        abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                         by the caller.  Updated in-place to outlet conditions
                         on return.  Must not be shared across threads.
        mdot           : float, mass flow rate [kg/s].
        A_in           : float, inlet flow area [m^2].
        A_out          : float, outlet flow area [m^2].

    Returns:
        P_out
        T_out

    Raises:
        ValueError   : if mdot, A_in, or A_out are non-positive, or if the
                       computed inlet Mach number is outside (0, 1).
        RuntimeError : if the numerical solver fails to find a subsonic root.
    """
    from scipy.optimize import brentq

    if mdot <= 0.0:
        raise ValueError(
            f"compressible_changing_area: mdot must be positive (got {mdot})."
        )
    if A_in <= 0.0:
        raise ValueError(
            f"compressible_changing_area: A_in must be positive (got {A_in})."
        )
    if A_out <= 0.0:
        raise ValueError(
            f"compressible_changing_area: A_out must be positive (got {A_out})."
        )

    # ------------------------------------------------------------------
    # Read inlet conditions and compute Ma_in from the abstract state.
    # ------------------------------------------------------------------
    AS = abstract_state
    P_in  = AS.p()
    T_in  = AS.T()
    rho_in = AS.rhomass()
    a_in   = AS.speed_sound()
    v_in   = mdot / (rho_in * A_in)
    Ma_in  = v_in / a_in

    if not (0.0 < Ma_in < 1.0):
        raise ValueError(
            f"compressible_changing_area: inlet Mach number must be in (0, 1) for "
            f"subsonic flow (got {Ma_in:.6f}).  Supersonic area changes are not supported."
        )

    # ------------------------------------------------------------------
    # Obtain gamma from CoolProp at inlet conditions.
    # ------------------------------------------------------------------
    gamma = AS.cpmass() / AS.cvmass()     # isentropic exponent, dimensionless

    # ------------------------------------------------------------------
    # Isentropic area-Mach function  A/A* = f(M, gamma)
    # NASA Eq #9.
    # ------------------------------------------------------------------
    exp_num = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    def _area_ratio(M):
        bracket = 1.0 + (gamma - 1.0) / 2.0 * M**2
        coeff   = (2.0 / (gamma + 1.0)) * bracket
        return (1.0 / M) * coeff**exp_num

    # ------------------------------------------------------------------
    # Total-condition ratios  (NASA Eqs #6, #7)
    # ------------------------------------------------------------------
    def _p_ratio(M):
        return (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (-(gamma / (gamma - 1.0)))

    def _t_ratio(M):
        return (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (-1.0)

    # ------------------------------------------------------------------
    # Recover total conditions at inlet.
    # ------------------------------------------------------------------
    P_total = P_in / _p_ratio(Ma_in)
    T_total = T_in / _t_ratio(Ma_in)

    # ------------------------------------------------------------------
    # A/A* at outlet equals inlet A/A* scaled by A_out/A_in.
    # ------------------------------------------------------------------
    A_star_ratio_in  = _area_ratio(Ma_in)
    A_star_ratio_out = A_star_ratio_in * (A_out / A_in)

    # ------------------------------------------------------------------
    # Solve for subsonic Ma_out such that _area_ratio(Ma_out) == A_star_ratio_out.
    # ------------------------------------------------------------------
    Ma_lo = 1e-9
    Ma_hi = 1.0 - 1e-9

    f_lo = _area_ratio(Ma_lo) - A_star_ratio_out
    f_hi = _area_ratio(Ma_hi) - A_star_ratio_out

    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            f"compressible_changing_area: could not bracket a subsonic root for "
            f"A/A*={A_star_ratio_out:.6f} (f_lo={f_lo:.4g}, f_hi={f_hi:.4g}).  "
            f"Check that the area ratio is physically realizable."
        )

    Ma_out, solver_result = brentq(
        lambda M: _area_ratio(M) - A_star_ratio_out,
        Ma_lo, Ma_hi,
        xtol=1e-10, rtol=1e-10,
        full_output=True,
    )

    if not solver_result.converged:
        raise RuntimeError(
            f"compressible_changing_area: brentq solver did not converge "
            f"(A/A*={A_star_ratio_out:.6f}, Ma_in={Ma_in:.6f})."
        )

    # ------------------------------------------------------------------
    # Recover outlet static conditions from total conditions and Ma_out.
    # ------------------------------------------------------------------
    P_out = P_total * _p_ratio(Ma_out)
    T_out = T_total * _t_ratio(Ma_out)

    return (P_out, T_out)


def compressible_changing_area_K(
    abstract_state, mdot, A_in, A_out, K,
    T_cricondentherm=None, P_cricondenbar=None,
    T_critical=None, P_critical=None,
):
    """Outlet pressure and temperature for a compressible fluid passing through
    an area change with a known loss coefficient K applied to inlet velocity.

    The caller must update abstract_state to the inlet (P, T) conditions before
    calling.  The abstract state is updated in place to outlet conditions, so
    the caller reads outlet pressure and temperature from the same object after
    the call returns.

    Enforces two integrated balance equations simultaneously:

      1. Stagnation-enthalpy conservation (adiabatic, no elevation change):
             H(P_out, T_out) + v_out^2/2 = H_in + v_in^2/2

      2. Entropy generation from the irreversible loss:
             S(P_out, T_out) - S_in = K * v_in^2 / (2 * T_avg)
         where T_avg = (T_in + T_out) / 2 and the mechanical energy dissipated
         per unit mass is e_loss = K * v_in^2 / 2.

    Mass continuity is satisfied implicitly: v_out = mdot / (rho_out * A_out).

    The two-equation system in (P_out, T_out) is solved with
    scipy.optimize.root using the isentropic area-change result as the
    initial guess. Note that this routine can take a lot of iterations (and corresponding
    abstract updates) to converge, which can take quite a long time!

    Args:
        abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                         by the caller.  Updated in-place to outlet conditions
                         on return.  Must not be shared across threads.
        mdot           : float, mass flow rate [kg/s].
        A_in           : float, inlet flow area [m^2].
        A_out          : float, outlet flow area [m^2].
        K              : float, loss coefficient referenced to inlet velocity
                         head (dimensionless, >= 0).

    Returns:
        None.  abstract_state is updated in place to outlet conditions.

    Raises:
        ValueError   : if mdot, A_in, or A_out are non-positive, K is negative,
                       or the inlet Mach number is outside (0, 1).
        RuntimeError : if the numerical solver fails to converge.
    """
    from scipy.optimize import root

    #input validation
    if mdot <= 0.0:
        raise ValueError(
            f"compressible_changing_area_K: mdot must be positive (got {mdot})."
        )
    if A_in <= 0.0:
        raise ValueError(
            f"compressible_changing_area_K: A_in must be positive (got {A_in})."
        )
    if A_out <= 0.0:
        raise ValueError(
            f"compressible_changing_area_K: A_out must be positive (got {A_out})."
        )
    if K < 0.0:
        raise ValueError(
            f"compressible_changing_area_K: K must be non-negative (got {K})."
        )

    AS = abstract_state

    P_in   = AS.p()
    T_in   = AS.T()
    rho_in = AS.rhomass()
    H_in   = AS.hmass()
    S_in   = AS.smass()
    v_in   = mdot / (rho_in * A_in)
    Ma_in  = v_in / AS.speed_sound()

    if not (0.0 < Ma_in < 1.0):
        raise ValueError(
            f"compressible_changing_area_K: inlet Mach number must be between 0 and 1. "
            f"(got {Ma_in:.6f}).  Supersonic area changes are not supported."
        )

    H_total = H_in + 0.5 * v_in**2    # stagnation enthalpy [J/kg], conserved as no work or heat input is assumed
    e_loss  = 0.5 * K * v_in**2       # mechanical energy dissipated per unit mass [J/kg]

    def residuals(x):
        P, T = x
        _safe_update_PT(AS, P, T, T_cricondentherm, P_cricondenbar, T_critical, P_critical)
        v   = mdot / (AS.rhomass() * A_out)
        T_avg = 0.5 * (T_in + T)
        #energy balance: Stagnation enthalpy = outlet enthalpy + outlet kinetic energy
        r_energy  = AS.hmass() + 0.5 * v**2 - H_total
        #entropy accounting: Outlet entropy = inlet entropy + entropy generated from friction heating [(K*v^2/2) / average temperature]
        #note that if there are significant temperature changes, this method will lose some accuracy.
        r_entropy = AS.smass() - S_in - e_loss / T_avg
        return [r_energy, r_entropy]

    # Initial guess: isentropic area change (K=0 limit).
    # compressible_changing_area leaves AS at inlet conditions, 
    # so AS is still valid for the root solver after this call.
    P0, T0 = compressible_changing_area(AS, mdot, A_in, A_out)

    sol = root(residuals, [P0, T0], method="hybr")

    if not sol.success:
        raise RuntimeError(
            f"compressible_changing_area_K: root solver did not converge "
            f"(P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, K={K:.4g}, "
            f"A_in={A_in:.4g} m^2, A_out={A_out:.4g} m^2).  "
            f"Solver message: {sol.message}"
        )

    P_out, T_out = sol.x
    _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)


def compressible_K(
    abstract_state, mdot, flow_area, K,
    T_cricondentherm=None, P_cricondenbar=None,
    T_critical=None, P_critical=None,
):
    """Outlet conditions for a compressible fluid passing through a fitting
    with a known loss coefficient K and no area change.

    Assumes adiabatic conditions and that fluid properties are roughly constant
    across the fitting.  Applies the single-step result from the combined
    energy, continuity, entropy, and EOS derivation (dA = 0 branch):

        dP = -K * v^2 * rho / 2
             / [1 - v^2*(drho/dP)_H / (1 - (v^2/rho)*(drho/dH)_P)]

        dT = [K*v^2/2 + (1/rho - (dH/dP)_T) * dP] / Cp

    For low Mach numbers the dP formula reduces to the familiar
    incompressible result dP = -K*rho*v^2/2.

    See the hand-derivation in /Derivation_images/dP_for_K
    Note that for the entropy accounting, no change in temperature is assumed across the fitting.
    The error introduced by this is probably less than the uncertainty of the K-factor correlation for most realistic
    scenarios. If a dramatic temperature change occurs over the fitting, you may be better off with
    the compressible_changing_area_K function which uses the average temperature in its entropy balance.
    That function is more rigorous but substantially more computationally intensive.

    Args:
        abstract_state : CoolProp AbstractState, pre-updated to inlet (P, T)
                         by the caller.  Updated in-place to outlet conditions
                         on return.  Must not be shared across threads.
        mdot           : float, mass flow rate [kg/s].
        flow_area      : float, flow area [m^2].
        K              : float, loss coefficient referenced to inlet velocity
                         head (dimensionless, >= 0).

    Returns:
        None.  abstract_state is updated in place to outlet conditions.

    Raises:
        ValueError   : if mdot or flow_area are non-positive, or K is negative.
        RuntimeError : if two-phase conditions or near-sonic flow are detected.
    """
    choke_mach_limit = 0.98

    if mdot <= 0.0:
        raise ValueError(f"compressible_K: mdot must be positive (got {mdot}).")
    if flow_area <= 0.0:
        raise ValueError(f"compressible_K: flow_area must be positive (got {flow_area}).")
    if K < 0.0:
        raise ValueError(f"compressible_K: K must be non-negative (got {K}).")

    AS = abstract_state
    P_in  = AS.p()
    T_in  = AS.T()

    if AS.phase() == _CP_PHASE_TWOPHASE:
        raise RuntimeError(
            f"compressible_K: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase flow only."
        )

    rho_in = AS.rhomass()
    Cp     = AS.cpmass()
    H_in   = AS.hmass()
    v_in   = mdot / (rho_in * flow_area)
    Ma_in  = v_in / AS.speed_sound()

    if Ma_in >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_K: inlet Mach number Ma={Ma_in:.4f} is near-sonic.  "
            f"Reduce flow rate or check geometry."
        )

    # Partial derivatives at inlet conditions
    drhodP_H = AS.first_partial_deriv(CP.iDmass, CP.iP,     CP.iHmass)  # (drho/dP)_H
    drhodH_P = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)      # (drho/dH)_P
    dHdP_T   = AS.first_partial_deriv(CP.iHmass, CP.iP,     CP.iT)      # (dH/dP)_T

    # dP: derived from energy + continuity + entropy + EOS with dA = 0. See full derivation in Derivation_images/dP_for_K
    inner  = 1.0 - (v_in**2 / rho_in) * drhodH_P
    dP     = (-K * v_in**2 * rho_in / 2.0) / (1.0 - v_in**2 * drhodP_H / inner)
    P_out  = P_in + dP

    # dT: from thermodynamic identity dH = Cp*dT + (dH/dP)_T * dP
    dT    = (K * v_in**2 / 2.0 + (1.0 / rho_in - dHdP_T) * dP) / Cp
    T_out = T_in + dT

    _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)
    #This is an initial guess for T_out, but we can perform an energy balance to make sure energy is conserved.
    # Stagnation enthalpy is conserved (adiabatic, no elevation change).
    # Nudge T_out so the computed state satisfies the energy balance exactly.
    H_stag = H_in + v_in**2 / 2.0
    rho_out_calc = AS.rhomass()
    v_out_calc   = mdot / (flow_area * rho_out_calc)
    energy_error = H_stag - (AS.hmass() + v_out_calc**2 / 2.0)
    Cp_out    = AS.cpmass()
    drhodT_P  = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
    T_out = T_out + energy_error / (Cp_out - mdot**2/(flow_area**2*rho_out_calc**3) * drhodT_P)
    _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)

    rho_out = AS.rhomass()
    Ma_out  = (mdot / (rho_out * flow_area)) / AS.speed_sound()
    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_K: outlet Mach number Ma={Ma_out:.4f} is near-sonic.  "
            f"Reduce flow rate or check geometry."
        )


def compressible_pipe_segment(
    abstract_state,
    mdot,
    dL,
    dz,
    D_h,
    roughness,
    flow_area,
    q_wall=0.0,
    isothermal=False,
    mu=None,
    T_cricondentherm=None,
    P_cricondenbar=None,
    T_critical=None,
    P_critical=None,
    energy_tol=10.0,
    dPdL_rel_tol=0.05,
    _split_depth=0,
    _max_split_depth=8,
    ):
    """Calculate compressible pipe-flow hydraulics over a single pipe slice
    using the Euler method with an adaptive bisection refinement.

    The slice is taken as a single forward-Euler step using inlet-evaluated
    properties.  After the step, two convergence metrics are evaluated at the
    trial outlet state:

      1. (Non-isothermal only)  The stagnation-enthalpy residual of the
         uncorrected Euler step, |energy_error|, compared against energy_tol.
      2. The relative change in dP/dL between inlet and trial outlet,
         compared against dPdL_rel_tol.

    If either metric exceeds its tolerance, abstract_state is restored to
    inlet conditions and the slice is recursively bisected (halving dL, dz,
    and q_wall) until both metrics fall within tolerance or _max_split_depth
    is reached.  When the slice converges, a one-iteration Newton correction
    is applied to T_out (non-isothermal case) so the final state satisfies
    the stagnation-enthalpy balance exactly.

    The caller must update abstract_state to the inlet (P, T) conditions before
    calling this function.  On return, abstract_state is updated in-place to the
    outlet (P_out, T_out) conditions, so the next call can proceed without any
    additional state update.

    Args:
        abstract_state  : CoolProp AbstractState instance, pre-configured for
                          the working fluid and already updated to inlet (P, T)
                          by the caller.  Updated in-place during integration
                          and left at outlet conditions on return.
                          Must not be shared across threads.
        mdot            : float, mass flow rate [kg/s].
        dL              : float, pipe slice length [m].
        dz              : float, elevation rise over the slice [m].
                          Positive = uphill; negative = downhill.
        D_h             : float, hydraulic diameter [m].
        roughness       : float, absolute pipe-wall roughness [m].
        flow_area       : float, cross-section flow area [m^2].
        q_wall          : float, heat flow into fluid [W] (default 0 =
                          adiabatic).  Ignored when isothermal=True.
        isothermal      : bool, if True the temperature ODE returns 0 and
                          T_out == T_in.  Default False.
        mu              : float or None, viscosity [Pa*s].  If None, CoolProp
                          is queried at each stage; falls back to the
                          Lee-Gonzalez-Eakin correlation if CoolProp raises error.
                          A None value is preserved across recursive splits so
                          each sub-slice queries its own inlet viscosity.
        energy_tol      : float, maximum allowed pre-correction stagnation
                          enthalpy residual [J/kg] before the slice is split.
                          Default 10.0 J/kg (~5e-3 K of T error for typical
                          natural-gas Cp; the post-split Newton correction
                          cleans up the remaining residual exactly).  Ignored
                          when isothermal=True.
        dPdL_rel_tol    : float, maximum allowed relative change in dP/dL
                          between inlet and trial outlet before the slice is
                          split.  Default 0.05 (5%).
        _split_depth    : int, internal recursion counter.  Do not set from
                          calling code; used to enforce _max_split_depth.
        _max_split_depth: int, maximum recursive bisection depth.  Default 8
                          (= up to 256x refinement of the caller's slice).
                          A RuntimeError is raised if convergence is not
                          achieved within this many splits.

    Returns:
        None.  abstract_state is updated in place to outlet conditions.
    """
    grav_constant    = 9.8066
    choke_mach_limit = 0.98

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    _checks = [  # (name, value, unit, strictly_positive)
        ("mdot",      mdot,         "kg/s", True),
        ("dL",        dL,           "m",    True),
        ("D_h",       D_h,          "m",    True),
        ("flow_area", flow_area,    "m^2",  True),
        ("roughness", roughness,    "m",    False),  # zero is valid (smooth pipe)
    ]
    _invalid = [
        f"{name}={val:.6g} {unit}"
        for name, val, unit, strict in _checks
        if (strict and val <= 0.0) or (not strict and val < 0.0)
    ]
    if _invalid:
        raise ValueError(
            f"compressible_pipe_segment: invalid parameter values — all must be "
            f"positive (roughness may be zero): {', '.join(_invalid)}."
        )

    # ------------------------------------------------------------------
    # Read inlet conditions from abstract_state (caller must have set it).
    # ------------------------------------------------------------------
    AS   = abstract_state
    P_in = AS.p()
    T_in = AS.T()

    phase = AS.phase()
    if phase == _CP_PHASE_TWOPHASE:
        raise RuntimeError(
            f"compressible_pipe_segment: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase hydraulics "
            f"only.  Consider checking your inlet conditions."
        )

    rho_in = AS.rhomass()       # kg/m^3
    Cp  = AS.cpmass()        # J/(kg*K)

    # Preserve the caller's mu argument (which may be None) for forwarding to
    # recursive split calls -- each sub-slice should re-query CoolProp at its
    # own inlet rather than reuse the parent slice's inlet viscosity.
    mu_user = mu
    if mu is None:
        try:
            mu = AS.viscosity()   # Pa*s
        except Exception:
            # Fall back to Lee-Gonzalez-Eakin if the EOS (e.g. Peng-Robinson)
            # does not support viscosity.  LGE is for hydrocarbon gases only;
            # supply mu explicitly for other fluids.
            mu = viscosity_LGE(T_in, AS.molar_mass() * 1000.0, rho_in)
    H_in = AS.hmass()   # J/kg
    a   = AS.speed_sound()   # m/s  (isentropic speed of sound)
    v_in  = mdot / (rho_in * flow_area)   # m/s
    Ma = v_in / a                       # Mach number

    if Ma >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_pipe_segment: inlet Mach number Ma={Ma:.4f} "
            f"is sonic or near-sonic.  Reduce flow rate or check geometry."
        )

    # Friction factor and Reynolds number at inlet conditions.
    Re = fluids_Reynolds(V=v_in, D=D_h, rho=rho_in, mu=mu)
    f_darcy = fluids_friction_factor(Re=Re, eD=roughness / D_h)
    # print(f_darcy)


    if not isothermal:

        # We will first calculate dP for a known dL, using fluid properties at the inlet conditions and assuming they don't change enough over the length slice to affect the calculation.
        # From the energy balance, dq = mdot * dH + mdot * d(v^2)/2 + mdot * g * dz
        # From the continuity equation mdot = rho * flow_area * v
        # Combine these two, take the derivative with respect to length, and do some rearranging (substituting dv^2 = 2vdv and dv = -v/rho * drho)
        # 1/mdot * dq/dL = dH/dL - v^2/rho * drho/dL + g * dz/dL
        # We will need to use an equation of state to relate pressure and density. If rho = f(H, P):
        # drho/dL = (∂rho/∂P)_H * dP/dL + (∂rho/∂H)_P * dH/dL [chain rule for partial derivatives]
        # We'll call (∂rho/∂P)_H  = A and (∂rho/∂H)_P = B and assume they are relatively constant over the length slice.

        # Accounting for entropy, from "Fundamentals of Gas Dynamics, 2nd Ed." by Zucker and Biblarz", equation 3.1
        # dS = dSe + dSi
        # where dSe = entropy change due to heat transfer = 1/mdot * dq/T
        # and dSi = entropy change due to friction = 1/T * K * v^2/2
        # and K = f * dL/D_h, where f is Darcy friciton factor and D_h is the hydraulic diameter (or simply the diameter for a round pipe)

        # We can use a thermodynamic identity to relate enthalpy, entropy, and pressure:
        # dH = T*dS + dP/rho (from table 6.2-1 in "Chemical, Biochemical, and Engineering Thermodynmics, 4th ed." by Sandler)
        # Substituting for dS and taking derivative with respect to length
        # dH/dL = 1/mdot * dq/dL + f * v^2 / (2 * D_h) + 1/rho * dP/dL

        # Then, taking the dH/dL from the partial derivatives chain rule above and plugging it in to eliminate drho/dL and eliminating dH/dL with the two equations for DH/dL,
        # after MUCH REARRANGING, you get:
        # dP/dL = (f * rho * v^2/(2*D_h) * (1-v^2 * B / rho) + rho * g * dz/dL - v^2 * B/mdot * dq/dL)/(v^2*A + v^2 * B/rho - 1)
        # where     ^friction contribution                     ^elevation change contrib      ^heat transfer contribution

        # For full details, see a hand-derivation in the /Derivation_images/dP_dL folder in the repo

        # We can use the Euler method to estimate the pressure at the end of a length slice dL


        #First, calculate those oddball partial derivatives
        A = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        B = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        #Now, calculate each contributing component of the dP/dL.
        dP_dL_denominator_factor = v_in**2*A + v_in**2*B/rho_in - 1
        dP_dL_friction = (f_darcy*rho_in*v_in**2/(2*D_h)*(1-v_in**2*B/rho_in))/dP_dL_denominator_factor
        dP_dL_gravity = (rho_in * grav_constant * dz/dL)/dP_dL_denominator_factor
        dP_dL_heatxfr = (-v_in**2*B*q_wall/(mdot * dL))/dP_dL_denominator_factor
        dPdL_in = dP_dL_friction + dP_dL_gravity + dP_dL_heatxfr

        dP = dL * dPdL_in
        P_out = P_in + dP

        #Ideally we would just calculate the output entropy and be good to go with that as our second state variable, and this works fine for single component systems.
        # change in entropy = change in entropy due to heat transfer (q_wall/(mdot * T)) + change in entropy due to friction (f_darcy * v^2/(2*D_h * Tin))
        # S_out = S_in + dS
        #   (From chapter 3 of "Fundamentals of Gas Dynamics, 2nd Ed." by Zucker and Biblarz". See equations 3.1 and 3.64)

        # However, with multicomponent systems, CoolProp has trouble using entropy as one of the inputs for an abstract state update,
        #  so for robustness we need to use a thermodynamic identity to convert it to something easier to deal with, like temperature.
        # Use dH = TdS + VdP = Cp dT + [V - T (∂V/∂T)_P] dP (from table 6.2-1 in "Chemical, Biochemical, and Engineering Thermodynmics, 4th ed." by Sandler)
        #                                   ^ [V - T (∂V/∂T)_P] is the Joule Thompson coefficient μ times negative heat capacity [-μCp]
        #substitute (∂V/∂T)_P = -1/ρ^2 (∂ρ/∂T)_P and solve for dT
        # Then, use the Euler method to calculate T_out.
        drhodT_P = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        dT = (1.0 / Cp) * (
            q_wall / mdot - (T_in / rho_in**2) * drhodT_P * dP + f_darcy * v_in**2 / (2.0 * D_h) * dL
        )
        T_out = T_in + dT

        #Next, we will update the abstract state to the calculated outlet conditions. After that, we will evaluate all of the properties and recalculate what dP/dL and dT/dL
        #would be at the outlet conditions. If it is dramatically different (an energy balance error is greater than energy_tol and the difference in calculated 
        # dP/dL is greater than dPdL_rel_tol), we split the segment in half and re-run the Euler step on each, again checking against the tolerance specs at the end. This is 
        # done recursively as necessary until the errors are smaller than the tolerance and/or we reach the specified number of splits (up to 8 splits/256x reduction in segment length by default)
        # If the Euler step over-extrapolated into an unphysical or two-phase region
        # (P < 0, two-phase, EOS failure) the AS update raises an error -- also treat that as an unambiguous "needs split" signal.
        
        trial_eval_error = None
        try:
            _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar,
                            T_critical, P_critical)

            # Pre-correction stagnation-enthalpy residual (first splitter metric).
            H_total_out  = H_in + q_wall/mdot + v_in**2/2 - grav_constant*dz
            rho_out_trial = AS.rhomass()
            v_out_trial   = mdot / (flow_area * rho_out_trial)
            H_total_calc  = AS.hmass() + v_out_trial**2/2
            energy_error  = H_total_out - H_total_calc

            # Recompute dP/dL at the trial outlet using the same derivation as
            # at the inlet (second splitter metric).
            if mu_user is not None:
                mu_out = mu_user
            else:
                try:
                    mu_out = AS.viscosity()
                except Exception:
                    mu_out = viscosity_LGE(T_out, AS.molar_mass() * 1000.0, rho_out_trial)
            Re_out      = fluids_Reynolds(V=v_out_trial, D=D_h, rho=rho_out_trial, mu=mu_out)
            f_darcy_out = fluids_friction_factor(Re=Re_out, eD=roughness / D_h)
            A_out = AS.first_partial_deriv(CP.iDmass, CP.iP,     CP.iHmass)
            B_out = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
            denom_out     = v_out_trial**2 * A_out + v_out_trial**2 * B_out / rho_out_trial - 1.0
            dPdL_fric_out = (f_darcy_out * rho_out_trial * v_out_trial**2 / (2 * D_h)
                             * (1 - v_out_trial**2 * B_out / rho_out_trial)) / denom_out
            dPdL_grav_out = (rho_out_trial * grav_constant * dz / dL) / denom_out
            dPdL_heat_out = (-v_out_trial**2 * B_out * q_wall / (mdot * dL)) / denom_out
            dPdL_out      = dPdL_fric_out + dPdL_grav_out + dPdL_heat_out

            # Splitter decision: split if either metric is out of tolerance.
            # The 1.0 Pa/m floor in dPdL_avg prevents division-by-zero artifacts
            # when both slopes happen to be tiny (e.g. zero-q horizontal slice).
            dPdL_avg    = max(abs(dPdL_in), abs(dPdL_out), 1.0)
            dPdL_relchg = abs(dPdL_out - dPdL_in) / dPdL_avg
            needs_split = (abs(energy_error) > energy_tol) or (dPdL_relchg > dPdL_rel_tol)
        except (RuntimeError, ValueError) as exc:
            # Trial state unevaluable -- definitely need to split.  Mark the
            # metrics as undefined so the depth-exceeded message is informative.
            trial_eval_error = exc
            energy_error     = float('nan')
            dPdL_relchg      = float('nan')
            needs_split      = True

        if needs_split:
            if _split_depth >= _max_split_depth:
                if trial_eval_error is not None:
                    raise RuntimeError(
                        f"compressible_pipe_segment: trial outlet state "
                        f"(P={P_out:.4g} Pa, T={T_out:.4g} K) is unphysical "
                        f"after {_max_split_depth} recursive splits "
                        f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K).  "
                        f"Underlying error: {trial_eval_error}"
                    ) from trial_eval_error
                raise RuntimeError(
                    f"compressible_pipe_segment: slice failed to converge after "
                    f"{_max_split_depth} recursive splits "
                    f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, "
                    f"energy_error={energy_error:.3g} J/kg [tol={energy_tol:.3g}], "
                    f"dPdL_relchg={dPdL_relchg:.3g} [tol={dPdL_rel_tol:.3g}]).  "
                    f"Reduce upstream profile slice length, loosen tolerances, "
                    f"or raise _max_split_depth."
                )
            # Restore AS to inlet conditions so the recursive halves start
            # from the correct state.  This is the one extra EOS update the
            # splitter costs us per split event.
            _safe_update_PT(AS, P_in, T_in, T_cricondentherm, P_cricondenbar,
                            T_critical, P_critical)
            half_kwargs = dict(
                mdot=mdot, dz=dz/2.0, D_h=D_h,
                roughness=roughness, flow_area=flow_area,
                q_wall=q_wall/2.0, isothermal=isothermal, mu=mu_user,
                T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
                T_critical=T_critical, P_critical=P_critical,
                energy_tol=energy_tol, dPdL_rel_tol=dPdL_rel_tol,
                _split_depth=_split_depth + 1,
                _max_split_depth=_max_split_depth,
            )
            # First half: AS (P_in, T_in) -> (P_mid, T_mid)
            compressible_pipe_segment(AS, dL=dL/2.0, **half_kwargs)
            # Second half: AS (P_mid, T_mid) -> (P_out, T_out)
            compressible_pipe_segment(AS, dL=dL/2.0, **half_kwargs)
            return

        # Converged: Finally, apply a one-step energy balance correction to the temperature so that the final state satisfies an energy balance. This costs one additional
        # abstract state update but does substantially improve the solution when operating at Mach numbers close to 1.
        
        # H_stagnation = H + v^2/2 + gz
        # H_stagnation_in + q_in = H_stagnation_out
        # Error = H_stagnation_out_calculated - H_stagnation_in - q_in
        # Differentiating the energy error with respect to temperature at constant pressure:
        #   q_in, H_stagnation_in, and the gz term in the outlet stagnation enthalpy are constants with respect to temperature. H_out_calc and v_out vary with a change in outlet temperature. 
        # d(error)/dT = (∂H/∂T)_P + (1/2) * (∂v^2/∂T)_P
        # Use continuity to convert v to rho, and use Cp = (∂H/∂T)_P:
        # d(error)/dT = Cp - mdot^2/(A^2*rho^3) * (∂ρ/∂T)_P
        Cp_out_state  = AS.cpmass()
        drhodT_P_out  = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        T_out = T_out + energy_error / (Cp_out_state - mdot**2/(flow_area**2*rho_out_trial**3)*drhodT_P_out)
        _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)

    else:
        #For the isothermal case, we know the outlet temperature but need to estimate the outlet pressure.
        #We can go back to our energy and entropy balances:
        #dH/dL = 1/mdot * dq/dL + f * v^2 / (2 * D_h) + 1/rho * dP/dL (from entropy balance)
        #1/mdot * dq/dL = dH/dL - v^2/rho * drho/dL + g * dz/dL (from energy balance)
        #Hey look, if we add those two equations, the dH/dL and dq/dL terms cancel

        #0 = f * v^2/(2*D_h) + (1/rho) * dP/dL - (v^2/rho) * drho/dL + g * dz/dL

        #now we need to eliminate drho/dL. If rho = f(P, T):
        # drho/dL = (∂ρ/∂T)_P * dT/dL + (∂ρ/∂P)_T * dP/dL (chain rule for partial derivatives)
        # Since this case is isothermal, dT/dL = 0 so we can eliminate the first term.
        # drho/dL = (∂ρ/∂P)_T * dP/dL
        #Substitute this back in to our energy/entropy balance equation and solve for dP/dL
        # dP/dL = (rho * f * v^2 / (2 * D_h) - rho * g * dz/dL) / (1 - v^2 * (∂ρ/∂P)_T)
        # We can now Euler method a dP/dL step
        drhodP_T = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)

        dP = (-rho_in * f_darcy*v_in**2*dL/(2*D_h) - rho_in * grav_constant * dz)/(1- v_in**2 * drhodP_T)
        dPdL_in = dP / dL
        P_out = P_in + dP
        T_out = T_in

        #Next, we will update the abstract state to the calculated outlet conditions. After that, we will evaluate all of the properties and recalculate what dP/dL and dT/dL
        #would be at the outlet conditions. If it is dramatically different (an energy balance error is greater than energy_tol and the difference in calculated 
        # dP/dL is greater than dPdL_rel_tol), we split the segment in half and re-run the Euler step on each, again checking against the tolerance specs at the end. This is 
        # done recursively as necessary until the errors are smaller than the tolerance and/or we reach the specified number of splits (up to 8 splits/256x reduction in segment length by default)
        # If the Euler step over-extrapolated into an unphysical or two-phase region
        # (P < 0, two-phase, EOS failure) the AS update raises an error -- also treat that as an unambiguous "needs split" signal.
        
        trial_eval_error = None
        try:
            _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar,
                            T_critical, P_critical)
            # Recompute dP/dL at the trial outlet for the splitter check.  There's
            # no energy balance to check in the isothermal case (T is fixed by
            # assumption), so dP/dL change is the only convergence metric.
            rho_out_trial = AS.rhomass()
            v_out_trial   = mdot / (flow_area * rho_out_trial)
            if mu_user is not None:
                mu_out = mu_user
            else:
                try:
                    mu_out = AS.viscosity()
                except Exception:
                    mu_out = viscosity_LGE(T_out, AS.molar_mass() * 1000.0, rho_out_trial)
            Re_out       = fluids_Reynolds(V=v_out_trial, D=D_h, rho=rho_out_trial, mu=mu_out)
            f_darcy_out  = fluids_friction_factor(Re=Re_out, eD=roughness / D_h)
            drhodP_T_out = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
            dPdL_out = (-rho_out_trial * f_darcy_out * v_out_trial**2 / (2 * D_h)
                        - rho_out_trial * grav_constant * dz / dL) / (1 - v_out_trial**2 * drhodP_T_out)

            dPdL_avg    = max(abs(dPdL_in), abs(dPdL_out), 1.0)
            dPdL_relchg = abs(dPdL_out - dPdL_in) / dPdL_avg
            needs_split = dPdL_relchg > dPdL_rel_tol
        except (RuntimeError, ValueError) as exc:
            trial_eval_error = exc
            dPdL_relchg      = float('nan')
            needs_split      = True

        if needs_split:
            if _split_depth >= _max_split_depth:
                if trial_eval_error is not None:
                    raise RuntimeError(
                        f"compressible_pipe_segment: isothermal trial outlet state "
                        f"(P={P_out:.4g} Pa, T={T_out:.4g} K) is unphysical after "
                        f"{_max_split_depth} recursive splits "
                        f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K).  "
                        f"Underlying error: {trial_eval_error}"
                    ) from trial_eval_error
                raise RuntimeError(
                    f"compressible_pipe_segment: isothermal slice failed to converge "
                    f"after {_max_split_depth} recursive splits "
                    f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, "
                    f"dPdL_relchg={dPdL_relchg:.3g} [tol={dPdL_rel_tol:.3g}]).  "
                    f"Reduce upstream profile slice length, loosen tolerances, "
                    f"or raise _max_split_depth."
                )
            _safe_update_PT(AS, P_in, T_in, T_cricondentherm, P_cricondenbar,
                            T_critical, P_critical)
            half_kwargs = dict(
                mdot=mdot, dz=dz/2.0, D_h=D_h,
                roughness=roughness, flow_area=flow_area,
                q_wall=q_wall/2.0, isothermal=isothermal, mu=mu_user,
                T_cricondentherm=T_cricondentherm, P_cricondenbar=P_cricondenbar,
                T_critical=T_critical, P_critical=P_critical,
                energy_tol=energy_tol, dPdL_rel_tol=dPdL_rel_tol,
                _split_depth=_split_depth + 1,
                _max_split_depth=_max_split_depth,
            )
            compressible_pipe_segment(AS, dL=dL/2.0, **half_kwargs)
            compressible_pipe_segment(AS, dL=dL/2.0, **half_kwargs)
            return

    # Outlet Mach check.  AS is at the (converged) outlet state in both branches.
    rho_out = AS.rhomass()
    a_out   = AS.speed_sound()
    v_out   = mdot / (rho_out * flow_area)
    Ma_out  = v_out / a_out
    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_pipe_segment: outlet Mach number Ma={Ma_out:.4f} "
            f"is sonic or near-sonic.  Reduce flow rate or check geometry."
        )
