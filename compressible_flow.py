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
    temperature to satisfy the stagnation-enthalpy balance. As of this writing,
    the single-step is taken based on inlet conditions, so this function is only 
    accurate for relatively small pressure changes (<5% ish) where properties do
    not appreciably change across the fitting.

compressible_pipe_segment(fs, dL, dz, D_h, roughness, ...)
    Core compressible pipe-flow integration over a single pipe slice.
    Solves coupled dP/dL and dT/dL ODEs (or the isothermal dP/dL equation)
    using a Heun (Euler predictor / trapezoidal corrector) step with a
    one-iteration energy-balance projection for T.  Updates the FlowState
    in-place to outlet conditions.

downsample_profile(profile, max_step_m=1000.0, ...)
    Reduce a dense pipe profile to the points that actually matter for
    compressible flow: diameter changes, polyline slope breaks, and a
    spacing cap.  Returns a new profile list in the same 4-tuple format.
"""

import csv
import math
import os
import warnings
import numpy as np
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import fluids.fittings
import fluids.flow_meter
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import composition
from component_classes import (
    Base_Line_Segment,
    Base_Bend,
    Base_Contraction_Expansion,
    Base_Valve,
    Base_CheckValve,
    Base_Orifice,
    downsample_profile,
    ureg,
)


# ---------------------------------------------------------------------------
# FlowState
# ---------------------------------------------------------------------------
#
# Convention for the compressible layer:
#   - AS is always at the STATIC thermodynamic state (P, T).
#   - mdot, A, and z ride along on the FlowState so that v, Ma, stagnation
#     enthalpy, and gravitational PE can be derived consistently.
#   - Physics functions (compressible_K, compressible_changing_area_K,
#     compressible_changing_area, compressible_pipe_segment,
#     choked_mass_flux) take a FlowState and mutate it in place: AS to a
#     new static state, and A and/or z when the function changes the
#     geometry of the flow (area-change boundaries, line-segment slices).
#   - mdot is invariant once set.
#
# Phase-envelope limits are cached on the FlowState so callers never need
# to forward T_cricondentherm / P_cricondenbar / T_critical / P_critical
# explicitly through every layer.

_GRAVITY_MS2 = 9.80665


class FlowState:
    """Flowing-fluid state at one location.

    Carries the static thermodynamic state (mutable AS), the conserved mass
    flow, the local flow area and elevation, and the cached phase-envelope
    limits.  Velocity, Mach number, stagnation properties, and gravitational
    PE are derived properties so they always reflect the current AS after
    any in-place mutation.

    Args:
        AS    : CoolProp AbstractState at the static (P, T).  Mutated in
                place by the compressible-flow functions.
        mdot  : float, mass flow rate [kg/s].  Magnitude, not signed --
                reverse-flow geometry is handled by network._reversed_component.
        A     : float, local flow cross-section area [m^2].
        z     : float, local elevation above datum [m].  Default 0.
        T_cricondentherm, P_cricondenbar, T_critical, P_critical :
                optional pre-computed phase-envelope limits.  If all four
                are None and build_envelope is True, the envelope is built
                via _build_phase_limits on construction.
        build_envelope : bool.  If False and the four limits are all None,
                construction skips _build_phase_limits and the FlowState
                carries None for all four (callers that don't need the
                phase hint can skip the build cost).  Default True.
    """

    __slots__ = ("AS", "mdot", "A", "z",
                 "T_cricondentherm", "P_cricondenbar",
                 "T_critical", "P_critical")

    def __init__(self, AS, mdot, A, z=0.0, *,
                 T_cricondentherm=None, P_cricondenbar=None,
                 T_critical=None, P_critical=None,
                 build_envelope=True):
        self.AS   = AS
        self.mdot = float(mdot)
        self.A    = float(A)
        self.z    = float(z)
        if (T_cricondentherm is None and P_cricondenbar is None
                and T_critical is None and P_critical is None
                and build_envelope):
            (self.T_cricondentherm, self.P_cricondenbar,
             self.T_critical, self.P_critical) = _build_phase_limits(AS)
        else:
            self.T_cricondentherm = T_cricondentherm
            self.P_cricondenbar   = P_cricondenbar
            self.T_critical       = T_critical
            self.P_critical       = P_critical

    # ---- Direct AS reads --------------------------------------------------
    @property
    def P(self):
        return self.AS.p()

    @property
    def T(self):
        return self.AS.T()

    @property
    def rho(self):
        return self.AS.rhomass()

    @property
    def h(self):
        return self.AS.hmass()

    @property
    def s(self):
        # Entropy is unchanged by kinetic energy, so static s == stagnation s.
        return self.AS.smass()

    # ---- Derived flow quantities -----------------------------------------
    @property
    def v(self):
        return self.mdot / (self.rho * self.A)

    @property
    def Ma(self):
        return self.v / self.AS.speed_sound()

    @property
    def h_stagnation(self):
        # Gas-dynamic stagnation enthalpy: h_static + v^2/2.  Excludes
        # gravitational PE on purpose; use h_total_with_g for full Bernoulli.
        return self.AS.hmass() + 0.5 * self.v ** 2

    @property
    def h_total_with_g(self):
        return self.h_stagnation + _GRAVITY_MS2 * self.z


def _safe_flowstate_update_PT(fs, P, T):
    """Update fs.AS to (P, T) using fs's cached phase-envelope limits."""
    _safe_update_PT(fs.AS, P, T,
                    fs.T_cricondentherm, fs.P_cricondenbar,
                    fs.T_critical, fs.P_critical)


def _area_match(fs, A_target, tol=1e-6):
    """If fs.A != A_target within tol, run an isentropic area change so
    that fs.AS reflects the state at A_target and fs.A == A_target.

    No-op when fs.A is already at A_target.  Used at the top of every
    component dP_dT method to absorb any area discontinuity between
    consecutive components automatically.
    """
    if abs(fs.A - A_target) / max(fs.A, A_target) > tol:
        compressible_changing_area_K(fs, A_target, K=0.0)


def _line_segment_choke_diagnostic(fs, profile, roughness, isothermal, mu, name):
    """Cheap ideal-gas predictive choke check at the start of
    Line_Segment.dP_dT.  Emits a UserWarning if the segment is predicted
    to choke before reaching its outlet; never raises.

    Adiabatic branch: ideal-gas Fanno closed-form ("Fluid
    Mechanics for Chemical Engineers, 2nd ed" by Noel de Nevers §8.4.1 eq. 8.30) at outlet
    Mach=1, in Darcy convention:

        fLmax/D = (1 - M^2)/(k*M^2)
                + (k+1)/(2k) * ln[(k+1)*M^2 / (2 + (k-1)*M^2)]

    compared against the cumulative geometric integral
    Σ f_i * dL_i / D_h_i along the profile.

    Isothermal branch: simplified long-pipeline form ("Fluid Mechanics
    for Chemical Engineers, 2nd ed" by Noel de Nevers §8.4.2 eq. 8.33, kinetic-energy
    term dropped per the textbook's "long pipeline" assumption).
    Integrating P*dP at constant T and ideal-gas rho gives

        P1^2 - P2^2 = mdot^2 * (R_univ T / M_molar)
                      * Σ f_i * dL_i / (D_h_i * A_i^2)

    so a real P2 requires cum_fL_over_DA2 < P1^2 * M_molar /
    (mdot^2 * R_univ * T).

    Best-effort: any internal failure is swallowed (the diagnostic must
    never block a real evaluation).
    """
    try:
        Ma_in = fs.Ma
        # Skip stagnant or already-choked: the existing reactive gate
        # in compressible_pipe_segment handles Ma >= 0.98 with a clear
        # RuntimeError; stagnant flow has no choke risk worth warning.
        if Ma_in <= 0.01 or Ma_in >= 0.98:
            return

        AS    = fs.AS
        P_in  = AS.p()
        T_in  = AS.T()
        rho_in = AS.rhomass()
        mdot  = fs.mdot
        M_molar = AS.molar_mass()

        # Ideal-gas gamma (mirror _ideal_gas_G_max): use Cp0 when
        # available, fall back to real-gas Cp*M for HEOS mixtures.
        R_univ = 8.31446261815324
        try:
            cp0 = AS.cp0molar()
        except (ValueError, RuntimeError):
            cp0 = AS.cpmass() * M_molar
        k = cp0 / (cp0 - R_univ)   # ideal-gas Cp/Cv

        # Viscosity: prefer caller's kwarg if supplied (matches
        # compressible_pipe_segment's contract), else CoolProp/LGE.
        mu_in = mu if mu is not None else _viscosity_or_LGE(AS, T_in, rho_in)

        # Cumulative geometric integrals along the profile.  Inlet
        # rho/mu held constant for the Re computation -- Re_i depends
        # only on D_h_i since mdot is constant.
        cum_fL_over_D    = 0.0
        cum_fL_over_DA2  = 0.0
        for i in range(len(profile) - 1):
            dist_in,  _e_in,  D_h_in,  area_in  = profile[i]
            dist_out, _e_out, _D_h_out, _a_out  = profile[i + 1]
            dL_i = dist_out - dist_in
            if dL_i <= 0.0 or D_h_in <= 0.0 or area_in <= 0.0:
                continue
            v_i  = mdot / (rho_in * area_in)
            Re_i = fluids_Reynolds(V=v_i, D=D_h_in, rho=rho_in, mu=mu_in)
            f_i  = fluids_friction_factor(Re=Re_i, eD=roughness / D_h_in)
            cum_fL_over_D   += f_i * dL_i / D_h_in
            cum_fL_over_DA2 += f_i * dL_i / (D_h_in * area_in * area_in)

        if isothermal:
            threshold = (P_in * P_in * M_molar
                         / (mdot * mdot * R_univ * T_in))   # [m^-5]
            if cum_fL_over_DA2 >= threshold:
                warnings.warn(
                    f"Line_Segment {name!r}: simplified isothermal "
                    f"pipeline equation predicts no real outlet pressure "
                    f"(cumulative f*dL/(D*A^2) = {cum_fL_over_DA2:.4g} m^-5 "
                    f">= inlet-conditions limit {threshold:.4g} m^-5 at "
                    f"P_in={P_in:.4g} Pa, T={T_in:.4g} K, mdot={mdot:.4g} "
                    f"kg/s).  Real-gas behavior may differ; if integration "
                    f"fails, isothermal-flow choking is the likely cause.",
                    UserWarning, stacklevel=3,
                )
        else:
            fLmax_over_D = (
                (1.0 - Ma_in * Ma_in) / (k * Ma_in * Ma_in)
                + (k + 1.0) / (2.0 * k)
                  * math.log((k + 1.0) * Ma_in * Ma_in
                             / (2.0 + (k - 1.0) * Ma_in * Ma_in))
            )
            if cum_fL_over_D >= fLmax_over_D:
                warnings.warn(
                    f"Line_Segment {name!r}: ideal-gas Fanno flow "
                    f"predicts choke before segment end (cumulative "
                    f"f*dL/D = {cum_fL_over_D:.4g} >= inlet-Mach limit "
                    f"{fLmax_over_D:.4g} at Ma_in={Ma_in:.4f}, "
                    f"gamma={k:.4f}).  Real-gas behavior may differ; "
                    f"if integration fails, adiabatic-flow choking is "
                    f"the likely cause.",
                    UserWarning, stacklevel=3,
                )
    except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
        # Diagnostic only -- never block the real evaluation.
        pass


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
        fs,
        isothermal=False,
        q_wall=0.0,
        mu=None,
        energy_tol=10.0,
        dPdL_rel_tol=0.05,
        Ma_change_tol=0.1,
        correction_skip_rel_tol=1e-9,
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

        On entry fs.AS must be at the segment inlet (P, T).  If fs.A does
        not match the profile-inlet area, an isentropic area-change is
        applied first to absorb the discontinuity.  On return fs.AS is at
        the outlet (P, T), fs.A == profile-outlet area, and fs.z is
        advanced by the segment's total elevation change.

        Args:
            fs              : FlowState.  fs.AS must be at the segment
                              inlet (P, T) when called; mutated in place.
            isothermal      : bool, if True temperature is held constant
                              through each slice.  Default False.
            q_wall          : float, total heat input to the fluid over the
                              entire segment [W].  Distributed uniformly per
                              unit length.  Ignored when isothermal=True.
                              Default 0.0 (adiabatic).
            mu              : float or None, viscosity [Pa*s] forwarded to
                              compressible_pipe_segment() for every slice.  If
                              None (default), CoolProp is queried at each slice
                              and falls back to Lee-Gonzalez-Eakin on failure.
            energy_tol      : float, stagnation-enthalpy residual tolerance
                              [J/kg] forwarded to compressible_pipe_segment()'s
                              adaptive splitter.  Default 10.0.  Ignored when
                              isothermal=True.
            dPdL_rel_tol    : float, relative dP/dL-change tolerance forwarded
                              to compressible_pipe_segment()'s adaptive
                              splitter.  Default 0.05 (5%).
            Ma_change_tol   : float, Mach-number-change tolerance forwarded to
                              compressible_pipe_segment()'s adaptive splitter.
                              Default 0.1.
            correction_skip_rel_tol : float, relative threshold below which
                              compressible_pipe_segment() skips the flash to
                              the Heun-corrected state and keeps the Euler
                              trial state.  Default 1e-9.
            max_split_depth : int, maximum recursive bisection depth permitted
                              per profile slice.  Default 8 (=256x refinement).

        Returns:
            profile_points: a list of (distance_m, pressure_Pa, temperature_K,
            velocity_ms) tuples, one per profile point (inlet through outlet),
            suitable for constructing pressure, temperature, or velocity
            profile plots.  fs is mutated in place to outlet conditions.

        Raises:
            ValueError   : if the profile has fewer than two points.
            RuntimeError : if two-phase conditions, choked flow, or a
                           CoolProp failure occur during integration.
        """
        if len(self.profile) < 2:
            raise ValueError(
                "Line_Segment.dP_dT: profile must have at least two points."
            )

        # Absorb any area discontinuity with the upstream component, then read
        # inlet (P, T) for the inlet-point record.
        _area_match(fs, self.profile[0][3])
        P0   = fs.AS.p()
        T0   = fs.AS.T()
        mdot = fs.mdot

        # Cheap ideal-gas predictive choke check.  Warns (does not raise)
        # if the segment is predicted to choke before its outlet.
        _line_segment_choke_diagnostic(
            fs, self.profile, self.roughness_si, isothermal, mu, self.name,
        )

        total_length  = self.total_length_m
        q_per_length  = q_wall / total_length if total_length > 0.0 else 0.0

        _AREA_TOL = 1e-6   # fractional area-change threshold
        n     = len(self.profile)

        # Record inlet conditions at profile point 0.
        dist0, _elev0, _D_h0, area0 = self.profile[0]
        v0 = mdot / (fs.AS.rhomass() * area0)
        profile_points = [(dist0, P0, T0, v0)]

        for i in range(n - 1):

            dist_in,  elev_in,  D_h_in,  area_in  = self.profile[i]
            dist_out, elev_out, D_h_out, area_out = self.profile[i + 1]

            dL      = dist_out - dist_in   # m, along-pipe slice length
            dz      = elev_out - elev_in   # m, elevation rise (positive = uphill)
            q_slice = q_per_length * dL    # W, heat for this slice

            # Invariant: fs.A == area_in at this point (set either by the
            # outer _area_match for the first slice, or by the previous
            # iteration's area-change correction).
            compressible_pipe_segment(
                fs,
                dL=dL,
                dz=dz,
                D_h=D_h_in,
                roughness=self.roughness_si,
                isothermal=isothermal,
                q_wall=q_slice,
                mu=mu,
                energy_tol=energy_tol,
                dPdL_rel_tol=dPdL_rel_tol,
                Ma_change_tol=Ma_change_tol,
                correction_skip_rel_tol=correction_skip_rel_tol,
                _max_split_depth=max_split_depth,
            )

            # Area-change correction at the boundary to the next slice.
            area_ratio = abs(area_out - area_in) / max(area_in, area_out)
            if area_ratio > _AREA_TOL:
                compressible_changing_area_K(fs, area_out, K=0.0)

            # Record conditions at this profile point after all corrections.
            P_cur   = fs.AS.p()
            T_cur   = fs.AS.T()
            v_cur   = mdot / (fs.AS.rhomass() * area_out)
            profile_points.append((dist_out, P_cur, T_cur, v_cur))
            if verbose:
                msg = str(f"Segment {self.name} Step: {i+1} of {n-1}, P = {P_cur}, T = {T_cur}")
                print(msg, end="\r")
        if verbose:
            print(f" "*len(msg), end="\r")
        return profile_points

    def dmdot_dT(
        self,
        fs,
        P2,
        isothermal=False,
        q_wall=0.0,
        mu=None,
        energy_tol=10.0,
        dPdL_rel_tol=0.05,
        Ma_change_tol=0.1,
        correction_skip_rel_tol=1e-9,
        max_split_depth=8,
        verbose=False,
    ):
        """Inverse of dP_dT: solve for the mass flow rate that produces
        outlet pressure P2 along the segment.

        Wraps the entire dP_dT slice loop in a forward closure and drives
        mdot via _solve_mdot_for_outlet_P.  The choke bound is the
        ideal-gas sonic mass flux at the smallest profile area --
        deliberately loose; the helper's retreat loop handles real-gas
        drift and the Fanno-style choke that friction induces well below
        the isentropic isentropic bound.

        The forward closure restores fs.AS, fs.A, and fs.z to the inlet
        state on each call so the slice loop sees the same starting
        conditions every iteration.  Heat input q_wall and integrator
        tolerances are forwarded verbatim to the inner dP_dT.

        Args:
            fs : FlowState at the segment inlet (static).  fs.AS must be
                 at the inlet (P, T); fs.mdot is consulted only as an
                 optional seed and is overwritten.  Mutated in place to
                 the outlet state on return.
            P2 : float, target outlet pressure [Pa].
            isothermal, q_wall, mu, energy_tol, dPdL_rel_tol,
            Ma_change_tol, correction_skip_rel_tol, max_split_depth,
            verbose : forwarded to self.dP_dT.

        Returns:
            profile_points : list, same format as Line_Segment.dP_dT --
            inlet through outlet (distance, P, T, v) tuples taken from
            the final converged solve.

        Raises:
            ValueError      : if the profile has fewer than two points,
                              or P2 >= inlet pressure.
            ChokedFlowError : if P2 is below the segment's choke limit.
        """
        if len(self.profile) < 2:
            raise ValueError(
                "Line_Segment.dmdot_dT: profile must have at least two points."
            )

        A_inlet = self.profile[0][3]
        _area_match(fs, A_inlet)

        if P2 >= fs.P:
            raise ValueError(
                f"Line_Segment.dmdot_dT: P2 must be strictly less than the "
                f"inlet pressure (got P2={P2:.4g} Pa, P_in={fs.P:.4g} Pa)."
            )

        # Snapshot the inlet state so the forward closure can restore it
        # on every brentq evaluation.
        P0 = fs.P
        T0 = fs.T
        A0 = fs.A
        z0 = fs.z

        # Crude upper bound: ideal-gas isentropic sonic mass flux at the
        # smallest profile area.  True Fanno choke under friction is
        # lower; the helper's retreat handles the gap.
        G_max = _ideal_gas_G_max(fs.AS)
        A_min = min(p[3] for p in self.profile)
        mdot_choked = G_max * A_min

        # Seed: 10% of the loose upper bound.  Only used to floor mdot_lo.
        mdot_guess = 0.1 * mdot_choked

        # Mutable container so the closure can publish the final
        # profile_points list out to the caller.
        last_profile_points = [None]

        def forward_at_mdot(mdot_trial):
            _safe_flowstate_update_PT(fs, P0, T0)
            fs.A    = A0
            fs.z    = z0
            fs.mdot = mdot_trial
            last_profile_points[0] = self.dP_dT(
                fs,
                isothermal=isothermal,
                q_wall=q_wall,
                mu=mu,
                energy_tol=energy_tol,
                dPdL_rel_tol=dPdL_rel_tol,
                Ma_change_tol=Ma_change_tol,
                correction_skip_rel_tol=correction_skip_rel_tol,
                max_split_depth=max_split_depth,
                verbose=verbose,
            )

        # Suppress the per-iteration ideal-gas choke diagnostic emitted
        # by dP_dT.  The diagnostic is a forward-direction informational
        # tool; under the inverse solve it fires on every brentq
        # iteration with slightly different cumulative integrals (which
        # defeats Python's default dedupe).  The reactive choke
        # detection inside compressible_pipe_segment still fires
        # normally.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _solve_mdot_for_outlet_P(
                    fs, P2,
                    forward_at_mdot=forward_at_mdot,
                    mdot_choked=mdot_choked,
                    mdot_guess=mdot_guess,
                    caller_name="Line_Segment.dmdot_dT",
                )
        except RuntimeError as exc:
            # Bracket failure -- the requested P2 is below what the segment
            # can deliver subsonically (Fanno-style friction choke at the
            # pipe exit).  Report the true Fanno choke (the largest mdot the
            # segment passes before its Ma>=0.98 reactive gate fires) rather
            # than the loose isentropic-nozzle bound at A_min: the latter
            # ignores wall friction and overstates the choke for a pipe.
            # forward_at_mdot already restores fs to inlet on each call, so
            # no manual re-anchor is needed before the bisection.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    choke = _fanno_choke_mdot(
                        forward_at_mdot, fs,
                        mdot_seed=mdot_guess,
                        mdot_infeasible=mdot_choked,
                    )
                raise ChokedFlowError(*choke) from exc
            except _NoChokeBracketError:
                # Could not establish a feasible flow -- propagate the
                # original failure rather than fabricate a structured payload.
                raise exc

        return last_profile_points[0]


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

    def dP_dT(self, fs, mu=None):
        """Outlet conditions for a compressible fluid passing through the bend.

        Absorbs any inlet-area discontinuity isentropically, then uses
        fluids.fittings.bend_rounded() to obtain K and delegates to
        compressible_K().  fs.A is unchanged on return (point fitting with
        equal inlet/outlet area).

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS must
                 be at the inlet (P, T); mutated in place to the outlet state.
            mu : float or None, viscosity [Pa*s] used in the Reynolds number
                 calculation.  If None (default), CoolProp is queried and
                 falls back to Lee-Gonzalez-Eakin on failure.

        Returns:
            None.  fs is mutated in place.
        """
        Di = self.Di_si
        A  = math.pi * Di ** 2 / 4.0

        _area_match(fs, A)

        rho_in = fs.rho
        if mu is None:
            mu = _viscosity_or_LGE(fs.AS, fs.T, rho_in)

        Re = fluids_Reynolds(V=fs.v, D=Di, rho=rho_in, mu=mu)
        K  = fluids.fittings.bend_rounded(
            Di=Di,
            bend_diameters=self.bend_dias,
            angle=self.ang_deg,
            Re=Re,
        )

        compressible_K(fs, K)

    def dmdot_dT(self, fs, P2):
        """Inverse of dP_dT: solve for the mass flow rate that produces
        outlet pressure P2 through the bend.

        K depends on Reynolds number which depends on mdot, so a
        K <-> mdot fixed-point loop wraps the per-iteration brentq
        inversion: pick K at the current mdot seed, drive
        `_solve_mdot_for_outlet_P` with a `compressible_K(fs, K)` forward
        closure, then re-evaluate K at the converged mdot and repeat
        until self-consistent.  Re enters bend_rounded only logarithmically,
        so 2-3 outer iterations typically suffice.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS
                 must be at the inlet (P, T); fs.mdot is consulted only
                 as an optional seed and is overwritten.  Mutated in
                 place.
            P2 : float, target outlet pressure [Pa].
            mu : float or None, viscosity [Pa*s].  If None (default),
                 CoolProp / LGE is used at each K evaluation.

        Returns:
            None.  fs is mutated in place.

        Raises:
            ValueError      : if P2 >= inlet pressure.
            ChokedFlowError : if P2 is below the bend's choke limit.
        """
        Di = self.Di_si
        A  = math.pi * Di ** 2 / 4.0
        _area_match(fs, A)

        if P2 >= fs.P:
            raise ValueError(
                f"Bend.dmdot_dT: P2 must be strictly less than the inlet "
                f"pressure (got P2={P2:.4g} Pa, P_in={fs.P:.4g} Pa)."
            )

        P1     = fs.P
        T1     = fs.T
        rho_in = fs.rho
        mu     = _viscosity_or_LGE(fs.AS, T1, rho_in)

        # Cheap ideal-gas choke bound -- the expensive real-gas
        # choked_mass_flux is deferred to the bracket-failure branch
        # below (only fires on actual choke).
        mdot_choked = _ideal_gas_G_max(fs.AS) * A

        # Seed mdot for the first Re/K evaluation.
        if fs.mdot > 0.0:
            mdot_seed = fs.mdot
        else:
            # Rough incompressible guess assuming K ~ 0.3 (typical bend);
            # any positive value works since K is re-fit on each pass.
            mdot_seed = A * math.sqrt(2.0 * (P1 - P2) * rho_in / 0.3)

        _MDOT_REL_TOL = 1.0e-4
        for _ in range(8):
            Re = fluids_Reynolds(
                V=mdot_seed / (rho_in * A), D=Di, rho=rho_in, mu=mu,
            )
            K = fluids.fittings.bend_rounded(
                Di=Di, bend_diameters=self.bend_dias,
                angle=self.ang_deg, Re=Re,
            )

            def forward_at_mdot(mdot_trial, K=K):
                # skip_choke_check=True: mdot_hi capped at 0.95*mdot_choked
                # by _solve_mdot_for_outlet_P, so the per-iteration
                # choked_mass_flux inside compressible_K is redundant.
                _safe_flowstate_update_PT(fs, P1, T1)
                fs.A    = A
                fs.mdot = mdot_trial
                compressible_K(fs, K, skip_choke_check=True)

            mdot_guess = A * math.sqrt(2.0 * (P1 - P2) * rho_in / max(K, 1e-6))

            try:
                mdot_new = _solve_mdot_for_outlet_P(
                    fs, P2,
                    forward_at_mdot=forward_at_mdot,
                    mdot_choked=mdot_choked,
                    mdot_guess=mdot_guess,
                    caller_name="Bend.dmdot_dT",
                )
            except RuntimeError as exc:
                _safe_flowstate_update_PT(fs, P1, T1)
                fs.A = A
                choke = choked_mass_flux(fs=fs, A_throat=A)
                raise ChokedFlowError(*choke) from exc

            if abs(mdot_new - mdot_seed) <= _MDOT_REL_TOL * max(mdot_new, 1e-30):
                return
            mdot_seed = mdot_new

        warnings.warn(
            f"Bend.dmdot_dT: K <-> mdot fixed point did not converge within "
            f"8 iterations (last mdot={mdot_seed:.4g} kg/s, target "
            f"P2={P2:.4g} Pa, inlet P1={P1:.4g} Pa).  Result reflects the "
            f"final iterate.",
            UserWarning,
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

    def dP_dT(self, fs):
        """Outlet conditions for a compressible fluid passing through the valve.

        Absorbs any inlet-area discontinuity isentropically, then dispatches
        on whether the valve carries a geometric constriction:

          * No constriction (``minimum_diameter`` is None or equal to ``Di``):
            delegates to compressible_changing_area_K() with A_out = A_pipe.
            The K-loss is applied across the pipe-area control volume; the
            entropy balance uses T_avg.
          * Constriction (``minimum_diameter`` < ``Di``): delegates to
            compressible_dA() with A_throat = pi*D_min^2/4 and A2 = A_pipe.
            Internal acceleration to the throat is isentropic; the K-loss
            recovery from throat to body uses the rigorous coupled (P, T)
            solve.  Internal choking at the trim is caught even when the
            body outlet is comfortably subsonic.

        fs.A is the pipe area on return in both branches.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS must
                 be at the inlet (P, T); mutated in place to outlet state.

        Returns:
            None.  fs is mutated in place.
        """
        A_pipe = math.pi * self.Di_si ** 2 / 4.0
        _area_match(fs, A_pipe)

        if self.D_min_si is None or self.D_min_si >= self.Di_si:
            compressible_changing_area_K(fs, A_pipe, K=self.K)
        else:
            A_throat = math.pi * self.D_min_si ** 2 / 4.0
            compressible_dA(fs, A_throat, K=self.K, A2=A_pipe)

    def dmdot_dT(self, fs, P2):
        """Inverse of dP_dT: solve for the mass flow rate that produces
        outlet pressure P2, then mutate fs to the outlet state.

        Dispatches on the same constriction test as dP_dT:

          * No constriction (``minimum_diameter`` is None or equal to ``Di``):
            wraps compressible_changing_area_K(fs, A_pipe, K) in a forward
            closure and drives it via _solve_mdot_for_outlet_P.  The choke
            bound is the isentropic sonic mass flux at the pipe area; the
            true Fanno choke under K-loss is lower, which the helper's
            retreat loop discovers.
          * Constriction (``minimum_diameter`` < ``Di``): forwards directly
            to compressible_dA in Mode 2, which already implements the
            throat-then-recovery inversion.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS
                 must be at the inlet (P, T); fs.mdot is consulted only as
                 an optional seed and is overwritten.  Mutated in place to
                 the outlet state; fs.mdot holds the solved value on
                 return.
            P2 : float, target outlet pressure [Pa].

        Returns:
            None.  fs is mutated in place.

        Raises:
            ValueError      : if P2 >= inlet pressure, or if the no-constriction
                              branch is called with K = 0 (lossless: P2 must
                              equal P_in for subsonic flow).
            ChokedFlowError : if P2 is below the component's choke limit
                              (raised by compressible_dA on the constricted
                              branch, or by this method on the no-constriction
                              branch after a bracket failure).
        """
        A_pipe = math.pi * self.Di_si ** 2 / 4.0
        _area_match(fs, A_pipe)

        if P2 >= fs.P:
            raise ValueError(
                f"{type(self).__name__}.dmdot_dT: P2 must be strictly less than "
                f"the inlet pressure (got P2={P2:.4g} Pa, P_in={fs.P:.4g} Pa)."
            )

        if self.D_min_si is not None and self.D_min_si < self.Di_si:
            A_throat = math.pi * self.D_min_si ** 2 / 4.0
            compressible_dA(fs, A_throat, K=self.K, A2=A_pipe, P2=P2)
            return

        if self.K <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.dmdot_dT: cannot solve for mdot with "
                f"K=0 (lossless valve has no P2 < P_in subsonic solution)."
            )

        P1     = fs.P
        T1     = fs.T
        A1     = fs.A
        rho_in = fs.rho

        # Cheap ideal-gas choke bound -- choked_mass_flux is expensive
        # (~1.5 s on HEOS mixtures because it grid-scans an isentrope
        # with ~40 CoolProp PT updates) but only its scalar mdot_choked
        # is needed for brentq's upper bracket, and the helper's
        # retreat loop tolerates a loose overestimate.  The full real-
        # gas payload is only needed when actually choked, so defer
        # that call to the bracket-failure branch.
        mdot_choked = _ideal_gas_G_max(fs.AS) * A1

        mdot_guess = A1 * math.sqrt(2.0 * (P1 - P2) * rho_in / self.K)

        def forward_at_mdot(mdot_trial):
            # skip_choke_check=True: mdot_hi capped at 0.95*mdot_choked
            # by _solve_mdot_for_outlet_P, so the per-iteration
            # choked_mass_flux is redundant with the outer estimate.
            _safe_flowstate_update_PT(fs, P1, T1)
            fs.A    = A1
            fs.mdot = mdot_trial
            compressible_changing_area_K(fs=fs, A_out=A1, K=self.K,
                                         skip_choke_check=True)

        try:
            _solve_mdot_for_outlet_P(
                fs, P2,
                forward_at_mdot=forward_at_mdot,
                mdot_choked=mdot_choked,
                mdot_guess=mdot_guess,
                caller_name=f"{type(self).__name__}.dmdot_dT",
            )
        except RuntimeError as exc:
            # Bracket failure on a monotonically-decreasing residual means
            # the component cannot subsonically attain P2 at this inlet
            # state.  Get the real-gas choke payload now (expensive but
            # we only run this on the failure path) and surface as
            # ChokedFlowError so the network solver can clamp mdot.
            _safe_flowstate_update_PT(fs, P1, T1)
            fs.A = A1
            choke = choked_mass_flux(fs=fs, A_throat=A1)
            raise ChokedFlowError(*choke) from exc


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

    Forward flow uses the K-factor stored on the instance and dispatches to
    either compressible_changing_area_K() (no constriction) or
    compressible_dA() (constriction at the trim), mirroring Valve.dP_dT.
    Reverse flow is handled upstream by network._reversed_component, which
    returns a shallow copy with K replaced by _SEALING_K (~ 1e9); dP_dT()
    detects that case (K >= _SEALED_K_THRESHOLD) and short-circuits to a
    clamped sealed-state outlet instead of running the forward-flow path,
    which would otherwise produce an unphysical negative P_out that crashes
    CoolProp.

    Constructor arguments are identical to Base_CheckValve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Forward-flow K-factor.
    """

    def dP_dT(self, fs):
        """Outlet conditions for a compressible fluid passing through the
        check valve.

        Reverse flow (sealing K substituted by _reversed_component) is
        short-circuited to a clamped sealed-state outlet, see module-level
        _SEALED_* constants.

        Forward flow (physical K) absorbs any inlet-area discontinuity, then
        dispatches on whether the check valve carries a geometric constriction:

          * No constriction (``minimum_diameter`` is None or equal to ``Di``):
            delegates to compressible_changing_area_K() with A_out = A_pipe.
          * Constriction (``minimum_diameter`` < ``Di``): delegates to
            compressible_dA() with A_throat = pi*D_min^2/4 and A2 = A_pipe.
            Internal choking at the trim is caught even when the body outlet
            is comfortably subsonic.

        fs.A is the pipe area on return in the forward-flow branches.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS must
                 be at the inlet (P, T); mutated in place to outlet state.

        Returns:
            None.  fs is mutated in place.
        """
        if self.K >= _SEALED_K_THRESHOLD:
            P_out, T_out = _sealed_outlet_PT(fs.AS.p(), fs.AS.T())
            _safe_flowstate_update_PT(fs, P_out, T_out)
            return

        A_pipe = math.pi * self.Di_si ** 2 / 4.0
        _area_match(fs, A_pipe)

        if self.D_min_si is None or self.D_min_si >= self.Di_si:
            compressible_changing_area_K(fs, A_pipe, K=self.K)
        else:
            A_throat = math.pi * self.D_min_si ** 2 / 4.0
            compressible_dA(fs, A_throat, K=self.K, A2=A_pipe)

    def dmdot_dT(self, fs, P2):
        """Inverse of dP_dT for the check valve: solve for the mass flow
        rate that produces outlet pressure P2.

        Mirrors Valve.dmdot_dT exactly on the forward-flow branches.  The
        sealed-K short-circuit (used by network._reversed_component for
        reverse flow) has no inverse -- a sealed valve passes no flow at
        any P2 -- so this method raises ValueError when invoked on a
        sealed shadow.  The network solver does not currently call
        dmdot_dT on reversed shadows; the guard is defensive.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS
                 must be at the inlet (P, T); fs.mdot is consulted only as
                 an optional seed and is overwritten.  Mutated in place.
            P2 : float, target outlet pressure [Pa].

        Returns:
            None.  fs is mutated in place.

        Raises:
            ValueError      : on a sealed-K shadow, or for the same
                              conditions as Valve.dmdot_dT.
            ChokedFlowError : if P2 is below the component's choke limit.
        """
        if self.K >= _SEALED_K_THRESHOLD:
            raise ValueError(
                "CheckValve.dmdot_dT: undefined for sealed check valve "
                "(K >= _SEALED_K_THRESHOLD); reverse-flow handling is "
                "governed at the network level."
            )
        # Forward-flow inversion is identical to Valve.dmdot_dT.
        Valve.dmdot_dT(self, fs, P2)


class Contraction_Expansion(Base_Contraction_Expansion):
    """Abrupt contraction or expansion with compressible pressure/temperature
    calculation.

    Modeled as adiabatic.  Inherits geometry storage and validation from
    Base_Contraction_Expansion.  The dP_dT() method is not yet implemented.

    Constructor arguments are identical to Base_Contraction_Expansion:
        Di_US : pint Quantity or float (m if float).  Upstream inner diameter.
        Di_DS : pint Quantity or float (m if float).  Downstream inner diameter.
    """

    def dP_dT(self, fs):
        """Outlet abstract state for a compressible fluid passing through the
        contraction/expansion.

        Absorbs any inlet-area discontinuity with the upstream component
        (so that fs.A == A_US before the K is applied), then uses
        fluids.fittings.contraction_sharp() or diffuser_sharp() to obtain
        the K-factor referenced to the upstream velocity head, and calls
        compressible_changing_area_K().  On return fs.A == A_DS.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS must
                 be at the inlet (P, T); mutated in place to outlet state.

        Returns:
            None.  fs is mutated in place.
        """
        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        if abs(Di_US - Di_DS) < 1e-12:
            return

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0

        _area_match(fs, A_US)

        if Di_US > Di_DS:
            # Contraction: fluids returns K w.r.t. downstream; convert to upstream.
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K    = K_ds * (A_DS / A_US) ** 2
        else:
            # Expansion: fluids returns K w.r.t. upstream velocity directly.
            K = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)

        compressible_changing_area_K(fs, A_DS, K)

    def dmdot_dT(self, fs, P2):
        """Inverse of dP_dT: solve for the mass flow rate that produces
        outlet pressure P2 across the area change.

        K is computed from the diameter ratio only (no mdot dependence),
        so this is a single brentq-on-mdot drive of
        compressible_changing_area_K via _solve_mdot_for_outlet_P -- no
        outer fixed-point loop is needed.  Choke is at the smaller of
        the two areas (the throat of a contraction; the inlet of an
        expansion).

        Only the contraction direction (Di_US > Di_DS) is supported.
        Expansions are deferred: for a sharp diffuser the kinetic-energy
        recovery term in Bernoulli typically exceeds the K-loss term, so
        P_out > P_in and the residual sign convention used by
        _solve_mdot_for_outlet_P is reversed.  Equal-diameter is
        degenerate (K=0, all mdot satisfy P2 = P_in).

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS
                 must be at the inlet (P, T); fs.mdot is consulted only
                 as an optional seed and is overwritten.  Mutated in
                 place to the outlet (A_DS, P2, T_out) state.
            P2 : float, target outlet pressure [Pa].

        Returns:
            None.  fs is mutated in place.

        Raises:
            ValueError         : if P2 >= inlet pressure, or if
                                 Di_US == Di_DS.
            NotImplementedError: for the expansion direction (Di_US < Di_DS).
            ChokedFlowError    : if P2 is below the throat choke limit.
        """
        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        if abs(Di_US - Di_DS) < 1e-12:
            raise ValueError(
                "Contraction_Expansion.dmdot_dT: equal-diameter case is "
                "degenerate (K=0, no friction -- P_out always equals P_in)."
            )
        if Di_US < Di_DS:
            raise NotImplementedError(
                "Contraction_Expansion.dmdot_dT: expansion inversion is "
                "not implemented.  For a sharp expansion the kinetic-recovery "
                "term in Bernoulli usually exceeds the K-loss term (P_out > "
                "P_in), reversing the residual sign convention used by the "
                "shared mdot solver.  Wrap the expansion in a downstream "
                "frictional component and invert that instead."
            )

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0
        _area_match(fs, A_US)

        if P2 >= fs.P:
            raise ValueError(
                f"Contraction_Expansion.dmdot_dT: P2 must be strictly less "
                f"than the inlet pressure (got P2={P2:.4g} Pa, "
                f"P_in={fs.P:.4g} Pa)."
            )

        if Di_US > Di_DS:
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K    = K_ds * (A_DS / A_US) ** 2
        else:
            K = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)

        P1     = fs.P
        T1     = fs.T
        rho_in = fs.rho

        # Choke is at the smaller area.  Cheap ideal-gas estimate now;
        # the expensive real-gas choked_mass_flux runs only on the
        # bracket-failure path below.
        A_throat = min(A_US, A_DS)
        mdot_choked = _ideal_gas_G_max(fs.AS) * A_throat

        mdot_guess = A_US * math.sqrt(2.0 * (P1 - P2) * rho_in / max(K, 1e-6))

        def forward_at_mdot(mdot_trial):
            # skip_choke_check=True: mdot_hi capped at 0.95*mdot_choked
            # by _solve_mdot_for_outlet_P, so the per-iteration
            # choked_mass_flux is redundant with the outer estimate.
            _safe_flowstate_update_PT(fs, P1, T1)
            fs.A    = A_US
            fs.mdot = mdot_trial
            compressible_changing_area_K(fs, A_DS, K, skip_choke_check=True)

        try:
            _solve_mdot_for_outlet_P(
                fs, P2,
                forward_at_mdot=forward_at_mdot,
                mdot_choked=mdot_choked,
                mdot_guess=mdot_guess,
                caller_name="Contraction_Expansion.dmdot_dT",
            )
        except RuntimeError as exc:
            _safe_flowstate_update_PT(fs, P1, T1)
            fs.A = A_US
            choke = choked_mass_flux(fs=fs, A_throat=A_throat)
            raise ChokedFlowError(*choke) from exc


# ---------------------------------------------------------------------------
# Orifice -- compressible child class
# ---------------------------------------------------------------------------

class Orifice(Base_Orifice):
    """Square-edged concentric orifice plate, compressible (gas) flow.

    Inherits geometry storage and validation from Base_Orifice.  Physics are
    handled by compressible_dA: isentropic acceleration to the vena-contracta
    throat (area = Cd * A_bore) followed by K-loss entropy recovery to the
    downstream pipe area.  The discharge coefficient Cd is computed via the
    ISO 5167-2 / Reader-Harris-Gallagher correlation or taken from
    Cd_override; the K-factor is derived from Cd via
    fluids.flow_meter.discharge_coefficient_to_K.

    Constructor arguments are identical to Base_Orifice:
        Di          : pint Quantity or float (m if float).  Pipe inner diameter.
        Do          : pint Quantity or float (m if float).  Orifice bore diameter.
        taps        : str.  Tap type: 'corner', 'D and D/2', or 'flange'.
        Cd_override : float or None.  Fixed Cd; bypasses the RHG correlation.
    """

    def dP_dT(self, fs):
        """Advance fs to the orifice outlet state.

        Absorbs any inlet-area mismatch isentropically, resolves the
        discharge coefficient at the current flow conditions, converts it to
        a K-factor, then calls compressible_dA to perform the two-stage
        (isentropic throat + K-loss recovery) solve.  fs.A is restored to
        the pipe area on return.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS must
                 be at the inlet (P, T); mutated in place to outlet state.

        Returns:
            None.  fs is mutated in place.
        """
        A_pipe = math.pi * self.Di_si ** 2 / 4.0
        A_bore = math.pi * self.Do_si ** 2 / 4.0
        _area_match(fs, A_pipe)

        rho_in = fs.rho
        mu = _viscosity_or_LGE(fs.AS, fs.T, rho_in)

        if self.Cd_override is not None:
            Cd = self.Cd_override
        else:
            Cd = fluids.flow_meter.C_Reader_Harris_Gallagher(
                D=self.Di_si, Do=self.Do_si,
                rho=rho_in, mu=mu, m=fs.mdot,
                taps=self.taps,
            )

        K        = fluids.flow_meter.discharge_coefficient_to_K(D=self.Di_si, Do=self.Do_si, C=Cd)
        A_throat = Cd * A_bore
        compressible_dA(fs, A_throat, K=K, A2=A_pipe)

    def dmdot_dT(self, fs, P2):
        """Inverse of dP_dT: solve for the mass flow rate that produces
        outlet pressure P2 across the orifice.

        With ``Cd_override`` set, this is a single direct call to
        compressible_dA in Mode 2.  With the RHG correlation in effect,
        Cd depends on Reynolds number which depends on mdot, so a Cd
        fixed-point loop wraps the dA call: seed mdot from fs.mdot (or
        an incompressible-orifice estimate), evaluate Cd at that mdot,
        invert dA at that K, repeat until mdot is self-consistent.  Cd
        is weakly Re-sensitive so 2-3 iterations typically suffice.

        Args:
            fs : FlowState at the upstream connection (static).  fs.AS
                 must be at the inlet (P, T); fs.mdot is consulted only
                 as an optional seed and is overwritten.  Mutated in
                 place to the outlet state.
            P2 : float, target outlet pressure [Pa].

        Returns:
            None.  fs is mutated in place; fs.mdot holds the solved
            value.

        Raises:
            ValueError      : if P2 >= inlet pressure.
            ChokedFlowError : if P2 is below the orifice choke limit.
        """
        A_pipe = math.pi * self.Di_si ** 2 / 4.0
        A_bore = math.pi * self.Do_si ** 2 / 4.0
        _area_match(fs, A_pipe)

        if P2 >= fs.P:
            raise ValueError(
                f"Orifice.dmdot_dT: P2 must be strictly less than the "
                f"inlet pressure (got P2={P2:.4g} Pa, P_in={fs.P:.4g} Pa)."
            )

        P1, T1, A1 = fs.P, fs.T, fs.A
        rho_in = fs.rho
        mu     = _viscosity_or_LGE(fs.AS, T1, rho_in)

        if self.Cd_override is not None:
            Cd = self.Cd_override
            K  = fluids.flow_meter.discharge_coefficient_to_K(
                D=self.Di_si, Do=self.Do_si, C=Cd,
            )
            compressible_dA(fs, Cd * A_bore, K=K, A2=A_pipe, P2=P2)
            return

        # RHG fixed point on Cd <-> mdot.  Cd_iter typically settles in
        # 2-3 passes; cap at 8 for pathological cases (warn if hit).
        mdot_seed = fs.mdot if fs.mdot > 0.0 else (
            # Incompressible orifice equation, Cd=0.6 placeholder.
            0.6 * A_bore * math.sqrt(2.0 * rho_in * (P1 - P2))
        )
        _MDOT_REL_TOL = 1.0e-4
        for _ in range(8):
            Cd = fluids.flow_meter.C_Reader_Harris_Gallagher(
                D=self.Di_si, Do=self.Do_si,
                rho=rho_in, mu=mu, m=mdot_seed,
                taps=self.taps,
            )
            K = fluids.flow_meter.discharge_coefficient_to_K(
                D=self.Di_si, Do=self.Do_si, C=Cd,
            )
            # Restore inlet before each dA call (dA mutates fs in place).
            _safe_flowstate_update_PT(fs, P1, T1)
            fs.A    = A1
            fs.mdot = mdot_seed
            compressible_dA(fs, Cd * A_bore, K=K, A2=A_pipe, P2=P2)
            mdot_new = fs.mdot
            if abs(mdot_new - mdot_seed) <= _MDOT_REL_TOL * max(mdot_new, 1e-30):
                return
            mdot_seed = mdot_new
        warnings.warn(
            f"Orifice.dmdot_dT: Cd <-> mdot fixed point did not converge "
            f"within 8 iterations (last mdot={mdot_seed:.4g} kg/s, "
            f"target P2={P2:.4g} Pa, inlet P1={P1:.4g} Pa). "
            f"Result reflects the final iterate.",
            UserWarning,
        )


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


def _viscosity_or_LGE(AS, T, rho):
    """CoolProp viscosity if available, else Lee-Gonzalez-Eakin fallback.

    Centralizes the kg/mol -> kg/kmol unit conversion that viscosity_LGE
    requires (CoolProp's molar_mass() returns kg/mol; LGE wants kg/kmol).
    """
    try:
        return AS.viscosity()
    except Exception:
        return viscosity_LGE(T, AS.molar_mass() * 1000.0, rho)


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


# ---------------------------------------------------------------------------
# Choked-flow primitives
# ---------------------------------------------------------------------------

class ChokedFlowError(RuntimeError):
    """Raised by compressible_K / compressible_changing_area_K when the
    requested mdot exceeds the real-gas choked mass flow at the fitting's
    throat.

    Subclasses RuntimeError so existing solver-side handlers that catch
    RuntimeError (e.g. compressible_network.walk_edge) continue to treat
    choked conditions as a walk failure under the penalty path. Callers
    that want to clamp mdot can catch ChokedFlowError specifically and
    read mdot_choked off the exception instance.

    On raise, the abstract_state has been updated to the component outlet
    state: the throat state when A_throat == A_outlet, or the isentropically
    recovered state at A_outlet when A_outlet > A_throat.
    """
    def __init__(self, mdot_choked, P_throat, T_throat, rho_throat,
                 P_outlet, T_outlet):
        super().__init__(
            f"choked: mdot_choked={mdot_choked:.4g} kg/s, "
            f"P*={P_throat:.4g} Pa, T*={T_throat:.4g} K"
        )
        self.mdot_choked = mdot_choked
        self.P_throat    = P_throat
        self.T_throat    = T_throat
        self.rho_throat  = rho_throat
        self.P_outlet    = P_outlet
        self.T_outlet    = T_outlet


class _NoChokeBracketError(RuntimeError):
    """Raised by choked_mass_flux when the isentropic grid scan finds no
    Mach=1 sign change on the accessible isentrope.  Distinct from a
    generic RuntimeError so that pre-screen callers can swallow only the
    "not choked here" outcome and let real CoolProp / numerical failures
    propagate.
    """
    pass


class TwoPhaseIsentropeError(RuntimeError):
    """Raised by _T_at_P_along_isentrope (and thus choked_mass_flux) when the
    single-phase isentrope root at a given pressure lies below the mixture's
    two-phase envelope, i.e. the isentropic expansion would condense.

    This program models single-phase compressible flow only; rather than let
    CoolProp emit a cryptic "stationary point" error from inside the two-phase
    dome, the isentrope solver floors its temperature search at the
    cricondentherm and raises this when no single-phase root exists above it.
    Distinct from _NoChokeBracketError ("not choked here") so callers can tell
    "out of single-phase scope" apart from "subsonic everywhere".
    """
    pass


# Tolerance band around Ma=1 for treating an inlet flow state as "exactly
# choked".  Chained area-change solves can drift the throat Ma a few Newton
# tolerances above or below 1.0; this constant lets compressible_changing_area
# and compressible_changing_area_K accept a sonic-throat inlet for the
# downstream subsonic-recovery solve.
_MA_SONIC_TOL = 1.0e-6


def _T_at_P_along_isentrope(
    AS, P, s_target, T_seed, T_lo, T_hi,
    T_cricondentherm=None, P_cricondenbar=None,
    T_critical=None, P_critical=None,
    T_tol=1e-6,
):
    """Root-solve T such that s(P, T) == s_target on a CoolProp AbstractState.

    CoolProp's (P, s) input pair is unreliable for HEOS mixtures, so this
    routes every EOS update through _safe_update_PT (PT_INPUTS + phase
    hint) and root-solves T externally. AS is left at the solved
    (P, T_sol) state on return.

    The caller supplies an initial T bracket. For a normal-gas isentrope
    expanding from stagnation, T(P) decreases monotonically with P, so
    the previous step's converged T is a tight upper bracket. If the
    initial bracket does not contain a sign change, this function
    progressively widens the lower bracket (handles retrograde-
    condensation excursions and large pressure steps); it raises
    RuntimeError if widening fails to bracket within 8 attempts.
    """
    from scipy.optimize import brentq

    def s_residual(T):
        _safe_update_PT(AS, P, T,
                        T_cricondentherm, P_cricondenbar,
                        T_critical, P_critical)
        return AS.smass() - s_target

    # Single-phase floor: above the cricondentherm the mixture is guaranteed
    # single-phase at any pressure, so the entropy search must never descend
    # below it (CoolProp's PT update fails inside the two-phase dome).  Floor
    # a hair (0.1%) ABOVE the cricondentherm: exactly on the envelope's
    # max-temperature point CoolProp's PT update still fails, so a strict
    # boundary value would re-introduce the crash.  When the envelope tracer
    # didn't yield a cricondentherm, fall back to the historical 0.1*T_hi
    # floor so pure-fluid / air callers are unaffected.
    T_floor = (T_cricondentherm * 1.001
               if T_cricondentherm is not None else 0.1 * T_hi)
    T_lo = max(T_lo, T_floor)

    f_hi = s_residual(T_hi)
    f_lo = s_residual(T_lo)
    tries = 0
    T_lo_curr = T_lo
    while f_hi * f_lo > 0 and T_lo_curr > T_floor and tries < 8:
        T_lo_curr = max(T_floor, T_lo_curr * 0.7)
        f_lo = s_residual(T_lo_curr)
        tries += 1
    if f_hi * f_lo > 0:
        if T_cricondentherm is not None:
            raise TwoPhaseIsentropeError(
                f"_T_at_P_along_isentrope: isentropic expansion to P={P:.4g} Pa "
                f"(s_target={s_target:.4g}) has no single-phase root above "
                f"T_cricondentherm={T_cricondentherm:.4g} K; the expansion enters "
                f"the two-phase region. The single-phase choke solver cannot continue."
            )
        raise RuntimeError(
            f"_T_at_P_along_isentrope: could not bracket isentropic T at "
            f"P={P:.4g} Pa (s_target={s_target:.4g}, "
            f"T range [{T_lo_curr:.4g}, {T_hi:.4g}] K)."
        )

    T_sol = brentq(s_residual, T_lo_curr, T_hi, xtol=T_tol, rtol=1e-9)
    # Leave AS at the solved (P, T) state. brentq's last s_residual call
    # may have been at a different T -- restore here.
    _safe_update_PT(AS, P, T_sol,
                    T_cricondentherm, P_cricondenbar,
                    T_critical, P_critical)
    return T_sol


def _ideal_gas_G_max(AS):
    """Choked mass flux estimate at the AS's current stagnation state from
    the ideal-gas isentropic formula. Returns G [kg/(s*m^2)] only --
    multiply by the throat area to get mdot_choked.

    No EOS update: uses the real-gas gamma = cp/cv read at the current
    (already-flashed) state. Used as a cheap pre-screen to avoid the
    rigorous check on the common (subsonic-by-a-lot) path.

        G_id = P0 * sqrt(gamma / (R_s * T0))
               * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))

    where R_s = R_universal / M is the specific gas constant.
    """
    R_univ = 8.31446261815324
    M      = AS.molar_mass()
    # cp/cv can spike where cp diverges (near the mixture critical point);
    # clamp to a physically plausible band so the estimate stays sane.
    gamma  = min(max(AS.cpmass() / AS.cvmass(), 1.01), 3.0)
    R_s    = R_univ / M
    P0    = AS.p()
    T0    = AS.T()
    crit_ratio = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return P0 * math.sqrt(gamma / (R_s * T0)) * crit_ratio


def choked_mass_flux(fs, A_throat, A_outlet=None, n_grid=40):
    """Calcualtes Real-gas choked mass flow through a throat of area A_throat
    by isentropically reducing pressure until the calculated flow state velocity
    at the throat area equals the speed of sound at the throat conditions. 
    CoolProp's Helmholz equation of state (HEOS) model doesn't handle pressure + 
    specific entropy as inputs for mixtures, so this function runs a solver
    that varies the temperature to find isentropic roots.

    Energy balance: Process is adiabatic so stagnation enthalpy is conserved
    Stagnation enthalpy = h0 = h_in+v_in^2/2 =  h_out+v_out^2/2

    The throat state is located where v = sqrt(2*(h0 - h)) equals the
    local speed of sound a along the constant-s isentrope from the inlet
    stagnation state.

    Post-throat recovery: if A_outlet is given and exceeds A_throat (only
    physically relevant for an expansion fitting whose smaller area is
    upstream), fs.AS is updated to an isentropically recovered state at
    A_outlet -- s_out = s_throat = s_inlet, mdot conserved, h_out +
    v_out^2/2 = h0. Otherwise fs.AS is left at the throat state.

    Args:
        fs         : FlowState at the upstream STATIC inlet state. fs.AS
                     is mutated in place during the march and left at the
                     component outlet state on return. The stagnation
                     reference enthalpy h0 = fs.h_stagnation = h_static +
                     0.5 * v_in**2 is built from the static state plus fs.v
        A_throat   : float, throat area [m^2].
        A_outlet   : float or None, component outlet area [m^2]. If None
                     or equal to A_throat, no post-throat recovery is
                     performed.
        n_grid     : int, number of log-spaced pressure grid points for
                     the initial bracketing scan.

    Returns:
        (mdot_choked, P_throat, T_throat, rho_throat, P_outlet, T_outlet)
        where mdot_choked = rho_throat * a_throat * A_throat and
        (P_outlet, T_outlet) = (P_throat, T_throat) when no recovery is
        performed.

    Raises:
        _NoChokeBracketError if no Mach=1 point is bracketed within
        [0.05*P_in, 0.99*P_in]. The caller should treat this as "not
        choked for this geometry" -- the requested mdot does not exceed
        the real-gas G_max anywhere along the accessible isentrope.
        Genuine CoolProp / numerical failures propagate as RuntimeError
        (or whatever the underlying call raises).
    """
    from scipy.optimize import brentq

    AS = fs.AS
    T_cricondentherm = fs.T_cricondentherm
    P_cricondenbar   = fs.P_cricondenbar
    T_critical       = fs.T_critical
    P_critical       = fs.P_critical

    P_in = AS.p()
    T_in = AS.T()
    # True stagnation enthalpy: h_static + v_in**2/2. 
    h0 = fs.h_stagnation
    s0 = AS.smass()         # entropy unchanged by KE: static == stagnation
    # Aliases preserved for the inner closures below.
    P0 = P_in
    T0 = T_in

    def state_at_P(P, T_seed, T_tol=1e-6):
        """Advance AS to (P, T_on_isentrope), return (h, rho, v, a, T).

        T_hi is fixed at T0 (isentropic expansion from stagnation can only
        cool the gas, so T0 is always a safe upper bound for s(P, T) = s0).
        T_lo seeds from a fraction of T_seed (the previous step's T or
        another warm guess) and is widened automatically by
        _T_at_P_along_isentrope if needed.  Using T0 as the upper bracket
        is critical: a too-tight T_hi can let the bracketing loop walk
        into a spurious low-T root in the two-phase region.

        T_tol loosens the inner entropy-root tolerance; the bracketing
        scan below passes ~1e-2 K (sign detection of v - a doesn't need
        micro-Kelvin roots) while the final brentq polish keeps the tight
        default.
        """
        T_lo_init = max(0.5 * T_seed, 0.1 * T0)
        T = _T_at_P_along_isentrope(
            AS, P, s0, T_seed, T_lo_init, T0,
            T_cricondentherm, P_cricondenbar, T_critical, P_critical,
            T_tol=T_tol,
        )
        h   = AS.hmass()
        rho = AS.rhomass()
        v   = math.sqrt(max(0.0, 2.0 * (h0 - h)))

        # Analytic speed of sound. CoolProp issue #1836 returns 0 inside
        # the two-phase dome on some HEOS backends; fall back to a
        # finite-difference d(P)/d(rho)|_s via two extra isentropic
        # T-root-solves at P +/- dP. Restore AS to (P, T) afterward.
        try:
            a = AS.speed_sound()
            if a is None or a <= 0 or not math.isfinite(a):
                raise ValueError("non-positive speed of sound")
        except (ValueError, RuntimeError):
            dP = max(1.0, 1e-4 * P)
            _T_at_P_along_isentrope(
                AS, P + dP, s0, T, max(0.5 * T, 0.1 * T0), T0,
                T_cricondentherm, P_cricondenbar, T_critical, P_critical,
            )
            rho_p = AS.rhomass()
            _T_at_P_along_isentrope(
                AS, P - dP, s0, T, max(0.5 * T, 0.1 * T0), T0,
                T_cricondentherm, P_cricondenbar, T_critical, P_critical,
            )
            rho_m = AS.rhomass()
            _T_at_P_along_isentrope(
                AS, P, s0, T, max(0.5 * T, 0.1 * T0), T0,
                T_cricondentherm, P_cricondenbar, T_critical, P_critical,
            )
            a = math.sqrt(2.0 * dP / max(rho_p - rho_m, 1e-30))
        return h, rho, v, a, T

    # ------------------------------------------------------------------
    # Bracket the Mach=1 point on the isentrope.  A directed walk seeded
    # at the critical pressure ratio r* = (2/(gamma+1))^(gamma/(gamma-1))
    # finds the sign change of f = v - a in 2-5 evaluations for typical
    # gases; the historical top-down log grid is retained as a fallback
    # for non-monotonic real-gas isentropes near the phase envelope.
    # Bracketing runs with a loosened inner entropy-root tolerance
    # (T_TOL_SCAN); the brentq polish below keeps the tight default, so
    # the returned throat state is unchanged.
    # ------------------------------------------------------------------
    T_TOL_SCAN = 1e-2   # [K] sign-detection only
    P_lo_lim = 0.05 * P0
    P_hi_lim = 0.99 * P0

    def scan_for_bracket(P_values, T_seed_init, T_tol):
        """Evaluate f = v - a along P_values (in walk order); return
        (P_bracket_lo, P_bracket_hi, T_seed_warm) on the first sign
        change, or None if f never changes sign."""
        f_prev = None
        P_prev = None
        T_seed = T_seed_init
        for P in P_values:
            _, _, v, a, T_seed = state_at_P(P, T_seed, T_tol=T_tol)
            f = v - a
            if f_prev is not None and f_prev * f < 0:
                return min(P, P_prev), max(P, P_prev), T_seed
            f_prev, P_prev = f, P
        return None

    # Directed walk: first evaluation at the seed decides the direction
    # (f < 0 = still subsonic there, so the choke -- if any -- lies at
    # lower P; f > 0 = already past it, walk up toward P0).  AS is at the
    # inlet here, so the real-gas gamma = cp/cv is a free read; clamp it
    # against the cp spike near the mixture critical point.
    gamma = min(max(AS.cpmass() / AS.cvmass(), 1.01), 3.0)
    r_star = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    P_seed = min(max(r_star * P0, P_lo_lim), P_hi_lim)

    bracket = None
    _, _, v_s, a_s, T_seed = state_at_P(P_seed, T0, T_tol=T_TOL_SCAN)
    P_curr, f_curr = P_seed, v_s - a_s
    going_down = f_curr < 0.0
    while bracket is None:
        if going_down:
            P_next = max(P_curr / 1.25, P_lo_lim)
        else:
            P_next = min(P_curr * 1.25, P_hi_lim)
        if P_next == P_curr:
            break   # window wall already evaluated; no sign change found
        _, _, v_n, a_n, T_seed = state_at_P(P_next, T_seed, T_tol=T_TOL_SCAN)
        f_next = v_n - a_n
        if f_curr * f_next < 0.0:
            bracket = (min(P_curr, P_next), max(P_curr, P_next), T_seed)
        P_curr, f_curr = P_next, f_next

    if bracket is None:
        # Fallback: coarse top-down log grid (choke usually lands at
        # P/P0 in [0.3, 0.6] but the directed walk can miss a sign change
        # if f is non-monotonic along the isentrope).
        P_grid = P0 * np.logspace(np.log10(0.99), np.log10(0.05), n_grid)
        bracket = scan_for_bracket(P_grid, T0, T_TOL_SCAN)

    if bracket is None:
        raise _NoChokeBracketError(
            f"choked_mass_flux: no Mach=1 point found in "
            f"P in [{0.05*P0:.4g}, {0.99*P0:.4g}] Pa at this stagnation state."
        )
    P_bracket_lo, P_bracket_hi, T_seed_at_lo = bracket

    # Brent on (v - a) over the bracket, at the tight inner tolerance.
    # Inner T-root-solve uses T0 as the upper bracket (state_at_P enforces
    # this); we only need to forward a warm T_seed for the lower-bracket
    # initialization.
    T_seed_holder = [T_seed_at_lo]

    def f_root(P):
        _, _, v, a, T = state_at_P(P, T_seed_holder[0])
        T_seed_holder[0] = T
        return v - a

    try:
        P_star = brentq(
            f_root, P_bracket_lo, P_bracket_hi,
            xtol=max(1.0, 1e-4 * P0), rtol=1e-6,
        )
    except ValueError as exc:
        if "different signs" not in str(exc):
            raise
        # Rare: the loose-tolerance scan handed brentq a bracket whose
        # endpoint signs flip under the tight tolerance (root within the
        # scan noise of an endpoint).  Re-scan the full grid tight.
        P_grid = P0 * np.logspace(np.log10(0.99), np.log10(0.05), n_grid)
        bracket = scan_for_bracket(P_grid, T0, 1e-6)
        if bracket is None:
            raise _NoChokeBracketError(
                f"choked_mass_flux: no Mach=1 point found in "
                f"P in [{0.05*P0:.4g}, {0.99*P0:.4g}] Pa at this stagnation state."
            ) from exc
        P_bracket_lo, P_bracket_hi, T_seed_holder[0] = bracket
        P_star = brentq(
            f_root, P_bracket_lo, P_bracket_hi,
            xtol=max(1.0, 1e-4 * P0), rtol=1e-6,
        )
    h_star, rho_star, v_star, a_star, T_star = state_at_P(P_star, T_seed_holder[0])
    mdot_choked = rho_star * a_star * A_throat

    # Default: no post-throat recovery. AS left at throat state.
    if A_outlet is None or A_outlet <= A_throat * (1.0 + 1e-12):
        return mdot_choked, P_star, T_star, rho_star, P_star, T_star

    # Post-throat isentropic recovery to A_outlet > A_throat.
    # Find P in (P_star, P0) such that v_out = mdot_choked / (rho * A_outlet)
    # and h + v_out^2/2 = h0, with s = s0.  state_at_P enforces T_hi = T0
    # so the entropy root-solve cannot wander into a spurious two-phase
    # low-T root above P*.
    def recovery_residual(P):
        _, rho, _, _, _ = state_at_P(P, T_star)
        h = AS.hmass()
        v_out = mdot_choked / (rho * A_outlet)
        return h + 0.5 * v_out * v_out - h0

    # At P = P_star: v_out < v_star (A_outlet > A_throat, rho similar) so
    # residual is negative. At P -> P0: rho -> rho0, v_out -> small,
    # h -> h0 so residual -> 0 from above. Bracket [P_star, ~0.999*P0].
    P_hi_recov = 0.999 * P0
    f_lo = recovery_residual(P_star)
    f_hi = recovery_residual(P_hi_recov)
    if f_lo * f_hi > 0:
        # No sign change -- can't recover. Leave AS at throat state and
        # return throat as outlet.
        state_at_P(P_star, T_star)
        return mdot_choked, P_star, T_star, rho_star, P_star, T_star

    P_out = brentq(
        recovery_residual, P_star, P_hi_recov,
        xtol=max(1.0, 1e-4 * P0), rtol=1e-6,
    )
    _, _, _, _, T_out = state_at_P(P_out, T_star)
    return mdot_choked, P_star, T_star, rho_star, P_out, T_out


def _choke_pre_screen(fs, A_throat, A_outlet, A_on_choke=None):
    """Pre-screen for choked flow at A_throat and raise ChokedFlowError if so.

    Two-stage gate used by compressible_K, compressible_changing_area_K

      1. Cheap ideal-gas G_max bound at A_throat -- no-op when fs.mdot is
         below ~50% of it (cannot be choked at this stagnation state).
      2. Above the bound, run the rigorous real-gas choked_mass_flux march.
         On a real choke, fs.AS is left at the recovered outlet state, fs.A
         is set to A_on_choke (if given), and ChokedFlowError is raised.
         Otherwise fs.AS (which the march mutated) is restored to inlet so
         the caller's subsonic solve can proceed.

    _NoChokeBracketError is swallowed (treated as "not choked here"); real
    CoolProp / numerical failures propagate.

    Args:
        fs         : FlowState at the inlet (static).  Mutated in place.
        A_throat   : float, throat area for the choke check [m^2].
        A_outlet   : float, area for post-throat isentropic recovery [m^2].
                     Pass A_throat itself when there is no recovery
                     (constant-area fitting).
        A_on_choke : float or None.  When given, fs.A is set to this value
                     immediately before ChokedFlowError is raised.  Use it
                     when the caller's geometry frame differs from the
                     inlet frame (e.g. orifice / changing-area: outlet =
                     pipe area, not throat).
    """
    if fs.mdot <= 0.5 * _ideal_gas_G_max(fs.AS) * A_throat:
        return

    # Sonic-inlet expansion shortcut: when fs is already at (or numerically
    # past) Ma=1 at the inlet area and the requested throat is no smaller
    # than the inlet, the caller is asking for a subsonic recovery from a
    # choked-throat starting state.  The rigorous march below would either
    # oscillate or spuriously raise (fs.mdot is essentially equal to
    # mdot_choked at this state); skip it and let the caller's subsonic
    # solve proceed.
    if fs.Ma >= 1.0 - _MA_SONIC_TOL and A_throat >= fs.A * (1.0 - 1e-12):
        return

    P_in = fs.AS.p()
    T_in = fs.AS.T()
    try:
        mdot_choked, P_t, T_t, rho_t, P_o, T_o = choked_mass_flux(
            fs, A_throat, A_outlet=A_outlet,
        )
    except _NoChokeBracketError:
        # Grid scan found no Mach=1 bracket -- not choked.  AS may have
        # wandered along the isentrope during the scan; restore to inlet.
        _safe_flowstate_update_PT(fs, P_in, T_in)
        return

    if fs.mdot > mdot_choked:
        if A_on_choke is not None:
            fs.A = A_on_choke
        raise ChokedFlowError(mdot_choked, P_t, T_t, rho_t, P_o, T_o)
    # Not choked: the march moved AS; restore to inlet for caller's solve.
    _safe_flowstate_update_PT(fs, P_in, T_in)


def compressible_changing_area(fs, A_out):
    """Isentropic pressure and temperature for an ideal gas passing through
    a change in flow area.

    Used as the initial-guess generator for the non-ideal solver in
    compressible_changing_area_K.  Does NOT mutate fs (neither AS, nor A) --
    the caller decides what to do with the returned (P_out, T_out).

    Uses the isentropic area-Mach relation to find the outlet Mach number
    satisfying continuity on the same isentropic curve, then recovers
    outlet static conditions from total-condition ratios.  gamma is taken
    from the AbstractState at inlet conditions, so this is strictly valid
    only for an ideal gas and serves as an initial guess for the real-gas
    solver.

    Area-Mach relation (from https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html):

        A / A* = (1/M) * {[2/(gamma+1)] * [1 + (gamma-1)/2 * M^2]}
                         ^ [(gamma+1) / (2*(gamma-1))]

    Total-condition ratios (Eqs #6, #7 from NASA):

        P / P_total = [1 + (gamma-1)/2 * M^2] ^ [-gamma/(gamma-1)]
        T / T_total = [1 + (gamma-1)/2 * M^2] ^ [-1]

    Args:
        fs    : FlowState at the inlet (static), carrying mdot and A_in.
                Not mutated.
        A_out : float, outlet flow area [m^2].

    Returns:
        (P_out, T_out) -- floats.

    Raises:
        ValueError   : if A_out is non-positive or the inlet Mach number
                       is outside (0, 1).
        RuntimeError : if the numerical solver fails to find a subsonic root.
    """
    from scipy.optimize import brentq

    if A_out <= 0.0:
        raise ValueError(
            f"compressible_changing_area: A_out must be positive (got {A_out})."
        )

    # ------------------------------------------------------------------
    # Read inlet conditions and compute Ma_in from the FlowState.
    # ------------------------------------------------------------------
    AS    = fs.AS
    A_in  = fs.A
    P_in  = AS.p()
    T_in  = AS.T()
    Ma_in = fs.Ma

    if not (0.0 < Ma_in <= 1.0 + _MA_SONIC_TOL):
        raise ValueError(
            f"compressible_changing_area: inlet Mach number must satisfy "
            f"0 < Ma_in <= 1 (got {Ma_in:.6f}).  Supersonic area changes "
            f"are not supported."
        )

    # Clamp a sonic (or marginally-supersonic by numerical noise) inlet a hair
    # below 1.0 so the isentropic area-Mach algebra below stays on the
    # subsonic branch.  The brentq search is also restricted to (0, 1), so the
    # returned root is the subsonic recovery.
    Ma_in = min(Ma_in, 1.0 - 1e-12)

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


def compressible_changing_area_K(fs, A_out, K=0.0, e_loss=None,
                                 skip_choke_check=False):
    """Outlet pressure and temperature for a compressible fluid passing through
    an area change with a known loss coefficient K applied to inlet velocity.
    Alternatively, e_loss, the mechanical energy loss per unit mass, can be supplied
    in lieu of K for computing the entropy change. If e_loss is supplied, K is ignored.
    If e_loss or K are supplied as zero, the process is evaluated as isentropic.

    Mutates fs in place: fs.AS goes to the outlet static state and fs.A
    becomes A_out.  fs.z is unchanged (area-change fittings are point
    fittings; no elevation change).

    Enforces two integrated balance equations simultaneously:

      1. Stagnation-enthalpy conservation (adiabatic, no elevation change):
             H(P_out, T_out) + v_out^2/2 = H_in + v_in^2/2

      2. Entropy generation from the irreversible loss:
             S(P_out, T_out) - S_in = e_loss / (T_avg)
         where T_avg = (T_in + T_out) / 2.  e_loss [J/kg] is either supplied
         directly by the caller or, if e_loss is None, computed from K as
         e_loss = 0.5 * K * v_in^2.

    Mass continuity is satisfied implicitly: v_out = mdot / (rho_out * A_out).

    The two-equation system in (P_out, T_out) is solved with a damped
    Newton iteration using an analytic Jacobian assembled from CoolProp
    partials; with the HEOS mixture backend each PT_INPUTS update costs
    ~tens of ms, but the partials at the just-computed state are ~1 us, so
    each Newton step costs exactly one EOS flash.  Initial guess:
    isentropic area-change result plus the linearized constant-area
    K-correction (same derivation as compressible_K), so the residual at
    the guess is already small and most calls converge in 1-3 flashes.
    If Newton stalls, the solve falls back to scipy.optimize.root (hybr)
    so robustness matches the original implementation.

    Args:
        fs    : FlowState at the inlet (static).  fs.AS, fs.mdot, fs.A are
                read as the inlet state; on return fs.AS is at the outlet
                state and fs.A == A_out.
        A_out : float, outlet flow area [m^2].
        K     : float, loss coefficient referenced to inlet velocity
                head (dimensionless, >= 0).  Default 0 (isentropic area
                change with the real-gas EOS); used by _area_match.
        e_loss: float, mechanical energy lost per unit mass in J/kg

    Returns:
        None.  fs is mutated in place.

    Raises:
        ValueError   : if A_out is non-positive, K is negative, or the
                       inlet Mach number is outside (0, 1).
        RuntimeError : if the numerical solver fails to converge.
    """
    from scipy.optimize import root

    #input validation
    if A_out <= 0.0:
        raise ValueError(
            f"compressible_changing_area_K: A_out must be positive (got {A_out})."
        )
    if K < 0.0:
        raise ValueError(
            f"compressible_changing_area_K: K must be non-negative (got {K})."
        )
    if e_loss is not None and e_loss < 0.0:
        raise ValueError(
            f"compressible_changing_area_K: e_loss must be non-negative (got {e_loss})."
        )

    AS    = fs.AS
    mdot  = fs.mdot
    A_in  = fs.A

    P_in   = AS.p()
    T_in   = AS.T()
    rho_in = AS.rhomass()
    H_in   = AS.hmass()
    S_in   = AS.smass()
    Cp_in  = AS.cpmass()
    v_in   = fs.v
    Ma_in  = fs.Ma

    if not (0.0 < Ma_in <= 1.0 + _MA_SONIC_TOL):
        raise ValueError(
            f"compressible_changing_area_K: inlet Mach number must satisfy "
            f"0 < Ma_in <= 1 (got {Ma_in:.6f}).  Supersonic area changes "
            f"are not supported."
        )

    # Inlet partials for the initial-guess K-correction below.  Read here
    # (~1 us each at the already-flashed inlet state) because the choke
    # pre-screen mutates AS away from the inlet.
    dRho_dP_H_in = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    dRho_dH_P_in = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    dH_dP_T_in   = AS.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)

    # Choked-flow pre-screen at A_throat = min(A_in, A_out), with recovery
    # to A_out.  On choke, fs.A is updated to A_out and ChokedFlowError
    # is raised; otherwise AS is restored to inlet for the subsonic Newton
    # solve below.  Callers that have already verified mdot is well
    # below the choke point (e.g. compressible_dA Mode 2's brentq, which
    # caps mdot_hi at 0.95 * mdot_choked) can pass skip_choke_check=True
    # to bypass this -- the pre-screen runs choked_mass_flux which costs
    # ~150-200 ms on HEOS mixtures, and amortizing it over a brentq's
    # ~10-20 trial mdots is what makes per-iteration inversion practical.
    if not skip_choke_check:
        _choke_pre_screen(fs, min(A_in, A_out), A_out, A_on_choke=A_out)

    H_total = H_in + 0.5 * v_in**2    # stagnation enthalpy [J/kg], conserved as no work or heat input is assumed
    if e_loss is None:
        e_loss  = 0.5 * K * v_in**2       # mechanical energy dissipated per unit mass [J/kg]

    # Residual scaling: r_energy is in J/kg (~v_in^2 magnitude), r_entropy is in
    # J/(kg·K) (~e_loss/T_in magnitude).  Without scaling the two equations
    # differ by 2-3 orders of magnitude and hybr wastes steps on the small one.
    r_energy_scale  = max(v_in**2, 1.0)
    r_entropy_scale = max(e_loss / T_in, 1e-3)

    # Cache state info inside the residual so the Jacobian callback can reuse
    # the AS partials computed at the same (P, T).  scipy's hybr calls jac
    # separately from fun, so we key on the input vector.
    cache = {"x": None}

    def residuals(x):
        P, T = x
        _safe_flowstate_update_PT(fs, P, T)
        rho = AS.rhomass()
        v   = mdot / (rho * A_out)
        T_avg = 0.5 * (T_in + T)
        #energy balance: Stagnation enthalpy = outlet enthalpy + outlet kinetic energy
        r_energy  = AS.hmass() + 0.5 * v**2 - H_total
        #entropy accounting: Outlet entropy = inlet entropy + entropy generated from friction heating [(K*v^2/2) / average temperature]
        #note that if there are significant temperature changes, this method will lose some accuracy.
        r_entropy = AS.smass() - S_in - e_loss / T_avg
        # Cache state-dependent quantities for the Jacobian.  AS partials are
        # ~1 us each vs. ~100 ms for a fresh PT update on HEOS mixtures, so
        # grabbing them now (instead of re-updating in jacobian()) is essentially free.
        cache["x"]       = (P, T)
        cache["v"]       = v
        cache["rho"]     = rho
        cache["T_avg"]   = T_avg
        cache["dH_dP_T"] = AS.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)
        cache["dH_dT_P"] = AS.first_partial_deriv(CP.iHmass, CP.iT, CP.iP)
        cache["dS_dP_T"] = AS.first_partial_deriv(CP.iSmass, CP.iP, CP.iT)
        cache["dS_dT_P"] = AS.first_partial_deriv(CP.iSmass, CP.iT, CP.iP)
        cache["dRho_dP_T"] = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
        cache["dRho_dT_P"] = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        return [r_energy / r_energy_scale, r_entropy / r_entropy_scale]

    def jacobian(x):
        if cache["x"] != tuple(x):
            #cache miss: force a residual eval to repopulate partials at this x.
            residuals(x)
        v       = cache["v"]
        rho     = cache["rho"]
        T_avg   = cache["T_avg"]
        # v = mdot/(rho*A_out)  =>  dv/dP = -v/rho * dRho/dP,  dv/dT = -v/rho * dRho/dT
        # r_energy = h + v^2/2 - H_total
        dEdP = cache["dH_dP_T"] - (v * v / rho) * cache["dRho_dP_T"]
        dEdT = cache["dH_dT_P"] - (v * v / rho) * cache["dRho_dT_P"]
        # r_entropy = s - S_in - e_loss / T_avg,  d(1/T_avg)/dT = -0.5/T_avg^2
        dSdP = cache["dS_dP_T"]
        dSdT = cache["dS_dT_P"] + 0.5 * e_loss / (T_avg * T_avg)
        return [[dEdP / r_energy_scale,  dEdT / r_energy_scale],
                [dSdP / r_entropy_scale, dSdT / r_entropy_scale]]

    # Initial guess: isentropic area change, then apply the linearized
    # constant-area K-correction (same derivation as compressible_K, see
    # /Derivation_images/dP_for_K) evaluated with the inlet partials:
    #
    #     dP_K = -e_loss * rho / [1 - v^2*(drho/dP)_H / (1 - (v^2/rho)*(drho/dH)_P)]
    #     dT_K = [e_loss + (1/rho - (dH/dP)_T) * dP_K] / Cp
    #
    # The dT form matters: for an ideal gas at constant area the dissipation
    # does NOT raise T_out (it is already inside the conserved stagnation
    # enthalpy; dT_K -> 0 as dP_K -> -rho*e_loss), so the naive +e_loss/Cp
    # correction used previously seeded T0 systematically high.
    # compressible_changing_area does not mutate fs (no AS update), so fs is
    # still at inlet conditions when the residuals below first flash.
    P0, T0 = compressible_changing_area(fs, A_out)
    inner = 1.0 - (v_in**2 / rho_in) * dRho_dH_P_in
    denom = 1.0 - v_in**2 * dRho_dP_H_in / inner if inner > 0.0 else 0.0
    if denom >= 0.25:
        dP_K = -e_loss * rho_in / denom
        dT_K = (e_loss + (1.0 / rho_in - dH_dP_T_in) * dP_K) / Cp_in
    else:
        # Near the acceleration-feedback singularity (Ma -> 1, e.g. the
        # subsonic recovery from a sonic throat) the linearization
        # over-shoots wildly; keep the historical crude correction there.
        dP_K = -e_loss * rho_in
        dT_K = e_loss / Cp_in
    P0 += dP_K
    T0 += dT_K

    # ------------------------------------------------------------------
    # Damped Newton with the analytic Jacobian.  Each iteration costs
    # exactly one EOS flash (inside residuals); the Jacobian reuses the
    # partials cached by that flash.  Termination is on either the scaled
    # O(1) residuals or a tiny relative Newton step -- the step test
    # catches the K=0 case where the entropy residual bottoms out at the
    # double-precision noise floor of AS.smass().  Both tolerances sit
    # far below the network solver's FD noise floor (diff_step=1e-5).
    # If Newton wanders somewhere CoolProp cannot evaluate, or fails to
    # converge, the solve restarts with scipy's hybr from the original
    # guess -- exactly what the pre-Newton implementation always did --
    # so robustness is unchanged.
    # ------------------------------------------------------------------
    NEWTON_RES_TOL  = 1e-9
    NEWTON_STEP_TOL = 1e-9
    NEWTON_MAX_ITER = 10
    T_floor = 0.3 * T_in   # step guard only, not a physical bound

    x_P, x_T = P0, T0
    converged = False
    # The first evaluation (at the guess) is outside the try: if CoolProp
    # cannot evaluate the guess itself, the hybr fallback would fail at the
    # same point, so let that propagate as before.
    r_e, r_s = residuals((x_P, x_T))
    try:
        for _ in range(NEWTON_MAX_ITER):
            if abs(r_e) < NEWTON_RES_TOL and abs(r_s) < NEWTON_RES_TOL:
                converged = True
                break
            (jA, jB), (jC, jD) = jacobian((x_P, x_T))
            det = jA * jD - jB * jC
            if det == 0.0 or not math.isfinite(det):
                break
            dP = -(r_e * jD - r_s * jB) / det
            dT = -(r_s * jA - r_e * jC) / det
            if abs(dP) < NEWTON_STEP_TOL * x_P and abs(dT) < NEWTON_STEP_TOL * x_T:
                converged = True   # fs.AS is already at (x_P, x_T)
                break
            # Per-coordinate trust region: keep P within a factor of ~2 and
            # T above the floor so one bad step can't push CoolProp into an
            # unevaluable state.
            x_P = min(max(x_P + dP, 0.5 * x_P), 1.5 * x_P)
            x_T = min(max(x_T + dT, T_floor), 1.5 * x_T)
            r_e, r_s = residuals((x_P, x_T))
    except RuntimeError:
        converged = False   # CoolProp failed along the walk; restart below

    if converged:
        # residuals() left fs.AS at (x_P, x_T) -- no extra flash needed.
        P_out, T_out = x_P, x_T
    else:
        # Fallback: scipy's hybr (trust-region) from the original guess,
        # matching the pre-Newton implementation's robustness.
        sol = root(residuals, [P0, T0], jac=jacobian, method="hybr")
        if not sol.success:
            raise RuntimeError(
                f"compressible_changing_area_K: root solver did not converge "
                f"(P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, K={K:.4g}, "
                f"e_loss={e_loss:.4g} J/kg, "
                f"A_in={A_in:.4g} m^2, A_out={A_out:.4g} m^2).  "
                f"Solver message: {sol.message}"
            )
        P_out, T_out = sol.x
        if cache["x"] != (P_out, T_out):
            _safe_flowstate_update_PT(fs, P_out, T_out)
    fs.A = A_out

    # The subsonic initial guess from compressible_changing_area keeps hybr on
    # the subsonic branch in normal use, but for unusual K / e_loss combinations
    # the solver can in principle settle on the supersonic root.  Reject that
    # here -- only the subsonic recovery is physically meaningful for our use.
    rho_out = AS.rhomass()
    v_out   = mdot / (rho_out * A_out)
    try:
        a_out = AS.speed_sound()
    except (ValueError, RuntimeError):
        a_out = 0.0
    if a_out > 0.0 and v_out / a_out > 1.0 + _MA_SONIC_TOL:
        raise RuntimeError(
            f"compressible_changing_area_K: Newton converged to a supersonic "
            f"outlet (Ma_out={v_out / a_out:.4f}); the subsonic root was not "
            f"found.  (P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, "
            f"A_in={A_in:.4g} m^2, A_out={A_out:.4g} m^2, K={K:.4g})."
        )


def compressible_K(fs, K, dPmax=0.05, A_throat=None, skip_choke_check=False):
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
    The error introduced by this is probably less than the uncertainty of the K-factor correlation for scenarios with minor pressure
    changes across the fitting. If a dramatic temperature or pressure change occurs over the fitting, this method will give inaccurate results.
    You are better off with the compressible_changing_area_K function which uses the average temperature
    in its entropy balance. That function is more rigorous but substantially more computationally intensive.

    Mutates fs in place: fs.AS to the outlet static state.  fs.A and fs.z
    are unchanged (constant-area, adiabatic, point fitting).

    Args:
        fs       : FlowState at the inlet (static).  fs.AS, fs.mdot, fs.A are
                   read as the inlet state; on return fs.AS is at the outlet
                   state.
        K        : float, loss coefficient referenced to inlet velocity
                   head (dimensionless, >= 0).
        dPmax    : Maximum relative dP (|dP|/P_in) before switching to the
                   rigorous compressible_changing_area_K solver.
        A_throat : float, optional.  Internal throat area [m^2] for the
                   choke-flow pre-screen (e.g. valve seat or trim area).
                   When smaller than fs.A, this is passed to choked_mass_flux
                   as the choke throat while fs.A is used as the recovery
                   outlet area, so internal choking is detected even when the
                   body outlet (fs.A) is comfortably subsonic.  Default None
                   = use fs.A for both (pipe-area-only check).

    Returns:
        None.  fs is mutated in place.

    Raises:
        ValueError   : if K is negative.
        RuntimeError : if two-phase conditions or near-sonic flow are detected.
    """
    choke_mach_limit = 0.98

    if K < 0.0:
        raise ValueError(f"compressible_K: K must be non-negative (got {K}).")

    AS        = fs.AS
    mdot      = fs.mdot
    flow_area = fs.A
    P_in      = AS.p()
    T_in      = AS.T()

    if AS.phase() == _CP_PHASE_TWOPHASE:
        raise RuntimeError(
            f"compressible_K: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase flow only."
        )

    rho_in = AS.rhomass()
    Cp     = AS.cpmass()
    H_in   = AS.hmass()
    v_in   = fs.v
    Ma_in  = fs.Ma

    if Ma_in >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_K: inlet Mach number Ma={Ma_in:.4f} is near-sonic.  "
            f"Reduce flow rate or check geometry."
        )

    # Internal-throat choke pre-screen, hoisted above the dP linearization
    # so it fires regardless of whether the fast (linearized) or slow
    # (compressible_changing_area_K) branch is taken below.  The inner
    # screens of those branches use the body area and would miss internal
    # choking when A_throat < flow_area; this one closes that gap.
    # Callers running an outer brentq with mdot capped well below
    # mdot_choked can pass skip_choke_check=True to bypass the ~150 ms
    # choked_mass_flux call here (and propagate the skip to the slow-path
    # fallback below).  Used by Bend.dmdot_dT's forward closure.
    throat_screened = False
    if (not skip_choke_check
            and A_throat is not None and 0.0 < A_throat < flow_area):
        _choke_pre_screen(fs, A_throat, flow_area)
        throat_screened = True

    # Partial derivatives at inlet conditions
    drhodP_H = AS.first_partial_deriv(CP.iDmass, CP.iP,     CP.iHmass)  # (drho/dP)_H
    drhodH_P = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)      # (drho/dH)_P
    dHdP_T   = AS.first_partial_deriv(CP.iHmass, CP.iP,     CP.iT)      # (dH/dP)_T

    # dP: derived from energy + continuity + entropy + EOS with dA = 0. See full derivation in Derivation_images/dP_for_K
    inner       = 1.0 - (v_in**2 / rho_in) * drhodH_P
    denominator = 1.0 - v_in**2 * drhodP_H / inner
    dP          = (-K * v_in**2 * rho_in / 2.0) / denominator

    #Check if calculated dP is greater than maximum or if near choked conditions (denominator < 0.5) If it is, switch to a more rigorous method.
    # The fallback path (compressible_changing_area_K) runs its own choke
    # pre-screen, so we do NOT run one here -- otherwise the computationally intensive isentropic choke finding function
    # would execute twice for the same inlet state.
    if (abs(dP) / P_in > dPmax) or (denominator < 0.5):
        compressible_changing_area_K(fs, A_out=flow_area, K=K,
                                     skip_choke_check=skip_choke_check)

    else: #small dP, not near choked

        # Body-area choke pre-screen (fast-path only).  Skip if the
        # hoisted A_throat screen above already covered a smaller area --
        # that's strictly more restrictive than this one.
        if not throat_screened and not skip_choke_check:
            _choke_pre_screen(fs, flow_area, flow_area)

        P_out  = P_in + dP

        # dT: from thermodynamic identity dH = Cp*dT + (dH/dP)_T * dP
        dT    = (K * v_in**2 / 2.0 + (1.0 / rho_in - dHdP_T) * dP) / Cp
        T_out = T_in + dT

        _safe_flowstate_update_PT(fs, P_out, T_out)
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
        _safe_flowstate_update_PT(fs, P_out, T_out)


def compressible_pipe_segment(
    fs,
    dL,
    dz,
    D_h,
    roughness,
    q_wall=0.0,
    isothermal=False,
    mu=None,
    energy_tol=10.0,
    dPdL_rel_tol=0.05,
    Ma_change_tol=0.1,
    correction_skip_rel_tol=1e-9,
    _split_depth=0,
    _max_split_depth=8,
    ):
    """Calculate compressible pipe-flow hydraulics over a single pipe slice
    using a Heun (trapezoidal) predictor-corrector with adaptive bisection
    refinement.

    The slice is first taken as a forward-Euler predictor step using
    inlet-evaluated properties.  After the step, three convergence metrics
    are evaluated at the trial outlet state:

      1. (Non-isothermal only)  The stagnation-enthalpy residual of the
         uncorrected Euler step, |energy_error|, compared against energy_tol.
      2. The relative change in dP/dL between inlet and trial outlet,
         compared against dPdL_rel_tol.
      3. The change in Mach number between inlet and trial outlet, compared
         against Ma_change_tol (catches slices where dP/dL and energy both
         pass while Ma climbs enough to invalidate the linearization).

    If any metric exceeds its tolerance, fs.AS is restored to inlet
    conditions and the slice is recursively bisected (halving dL, dz, and
    q_wall) until all metrics fall within tolerance or _max_split_depth
    is reached.  When the slice converges, the Heun corrector is applied:
    the outlet pressure is recomputed with the average of the inlet and
    trial-outlet dP/dL slopes (second-order accurate in dL), and the
    outlet temperature is set by a one-iteration Newton projection onto
    the stagnation-enthalpy balance at the corrected pressure
    (non-isothermal case).  Both reuse properties already evaluated at the
    trial state, so the corrector costs no additional EOS flash beyond the
    final update to the corrected (P, T) -- and that flash is itself
    skipped when the correction is below correction_skip_rel_tol
    (e.g. near-stagnant slices during network solver probing).

    Choke gating is mode-dependent: adiabatic (Fanno) flow chokes at
    Ma = v/a = 1, but isothermal flow goes singular at the isothermal
    sound speed a_T = sqrt((dP/drho)_T) = a/sqrt(gamma) for an ideal gas,
    so in isothermal mode the inlet/outlet gates test
    Ma_T = v*sqrt((drho/dP)_T) instead of v/a.

    On entry fs.AS must be at the inlet (P, T) -- the caller's job.  On
    success fs.AS is at the outlet (P, T) and fs.z is advanced by dz.
    fs.A is unchanged (slice is uniform-area; area transitions are handled
    by the calling component via compressible_changing_area_K).

    Args:
        fs              : FlowState at the inlet (static).  fs.AS, fs.mdot,
                          fs.A are read; fs.AS is mutated and fs.z is
                          advanced by dz on success.
        dL              : float, pipe slice length [m].
        dz              : float, elevation rise over the slice [m].
                          Positive = uphill; negative = downhill.
        D_h             : float, hydraulic diameter [m].
        roughness       : float, absolute pipe-wall roughness [m].
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
        Ma_change_tol   : float, maximum allowed change in Mach number
                          between inlet and trial outlet before the slice is
                          split.  Default 0.1.
        correction_skip_rel_tol : float, relative threshold below which the
                          Heun-corrected (P, T) is considered identical to
                          the Euler trial state and the final EOS flash is
                          skipped (the trial state is kept).  Deliberately
                          tiny (default 1e-9): the discarded correction is a
                          signed truncation term that accumulates across
                          slices, and a skip/no-skip flip must stay below
                          the network solver's finite-difference noise
                          floor (diff_step=1e-5 on inverse edges).
        _split_depth    : int, internal recursion counter.  Do not set from
                          calling code; used to enforce _max_split_depth.
        _max_split_depth: int, maximum recursive bisection depth.  Default 8
                          (= up to 256x refinement of the caller's slice).
                          A RuntimeError is raised if convergence is not
                          achieved within this many splits.

    Returns:
        None.  fs is mutated in place.
    """
    grav_constant    = 9.8066
    choke_mach_limit = 0.98

    AS        = fs.AS
    mdot      = fs.mdot
    flow_area = fs.A

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
    # Read inlet conditions from fs.AS (caller must have set it).
    # ------------------------------------------------------------------
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
        # LGE fallback (inside _viscosity_or_LGE) is hydrocarbon-only; supply
        # mu explicitly for other fluids whose EOS lacks viscosity (e.g. PR).
        mu = _viscosity_or_LGE(AS, T_in, rho_in)
    H_in = AS.hmass()   # J/kg
    a   = AS.speed_sound()   # m/s  (isentropic speed of sound)
    v_in  = mdot / (rho_in * flow_area)   # m/s
    Ma = v_in / a                       # Mach number

    # Inlet choke gate.  Adiabatic (Fanno) flow chokes at Ma = v/a = 1, but
    # the isothermal dP/dL denominator goes singular earlier, at the
    # isothermal sound speed a_T = sqrt((dP/drho)_T) = a/sqrt(gamma) for an
    # ideal gas (~0.88*a for gamma=1.3) -- so the isothermal branch must
    # gate on Ma_T = v*sqrt((drho/dP)_T), not v/a, or the gate never fires
    # before the singularity.
    if isothermal:
        drhodP_T = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
        Ma_gate    = v_in * math.sqrt(drhodP_T)
        gate_label = "isothermal Mach number Ma_T = v/a_T"
    else:
        Ma_gate    = Ma
        gate_label = "Mach number Ma"
    if Ma_gate >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_pipe_segment: inlet {gate_label}={Ma_gate:.4f} "
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
            _safe_flowstate_update_PT(fs, P_out, T_out)

            # Pre-correction stagnation-enthalpy residual (first splitter metric).
            H_total_out  = H_in + q_wall/mdot + v_in**2/2 - grav_constant*dz
            rho_out_trial = AS.rhomass()
            v_out_trial   = mdot / (flow_area * rho_out_trial)
            H_total_calc  = AS.hmass() + v_out_trial**2/2
            energy_error  = H_total_out - H_total_calc

            # Recompute dP/dL at the trial outlet using the same derivation as
            # at the inlet (second splitter metric).
            mu_out = mu_user if mu_user is not None else _viscosity_or_LGE(AS, T_out, rho_out_trial)
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

            # Third splitter metric: Mach-number change across the slice.
            # dP/dL and energy can both pass while Ma climbs enough to
            # invalidate the inlet-property linearization (e.g. 0.3 -> 0.7).
            Ma_out_trial = v_out_trial / AS.speed_sound()
            Ma_chg       = abs(Ma_out_trial - Ma)

            # Splitter decision: split if any metric is out of tolerance.
            # The 1.0 Pa/m floor in dPdL_avg prevents division-by-zero artifacts
            # when both slopes happen to be tiny (e.g. zero-q horizontal slice).
            dPdL_avg    = max(abs(dPdL_in), abs(dPdL_out), 1.0)
            dPdL_relchg = abs(dPdL_out - dPdL_in) / dPdL_avg
            needs_split = ((abs(energy_error) > energy_tol)
                           or (dPdL_relchg > dPdL_rel_tol)
                           or (Ma_chg > Ma_change_tol))
        except (RuntimeError, ValueError) as exc:
            # Trial state unevaluable -- definitely need to split.  Mark the
            # metrics as undefined so the depth-exceeded message is informative.
            trial_eval_error = exc
            energy_error     = float('nan')
            dPdL_relchg      = float('nan')
            Ma_chg           = float('nan')
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
                # The dP/dL denominator is Ma^2 - 1 for an ideal gas; a value
                # near zero at the (deepest) failing slice's inlet means the
                # slice is straddling the Fanno singularity -- name the
                # physical origin instead of leaving only "failed to
                # converge".  -0.3 corresponds to Ma ~ 0.84.
                choke_hint = ""
                if dP_dL_denominator_factor > -0.3:
                    choke_hint = (
                        "  The dP/dL denominator is near zero -- the flow is "
                        "likely choking (Fanno) within this slice; reduce "
                        "the flow rate."
                    )
                raise RuntimeError(
                    f"compressible_pipe_segment: slice failed to converge after "
                    f"{_max_split_depth} recursive splits "
                    f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, "
                    f"energy_error={energy_error:.3g} J/kg [tol={energy_tol:.3g}], "
                    f"dPdL_relchg={dPdL_relchg:.3g} [tol={dPdL_rel_tol:.3g}], "
                    f"Ma_change={Ma_chg:.3g} [tol={Ma_change_tol:.3g}]).  "
                    f"Reduce upstream profile slice length, loosen tolerances, "
                    f"or raise _max_split_depth.{choke_hint}"
                )
            # Restore AS to inlet conditions so the recursive halves start
            # from the correct state.  This is the one extra EOS update the
            # splitter costs us per split event.
            _safe_flowstate_update_PT(fs, P_in, T_in)
            half_kwargs = dict(
                dz=dz/2.0, D_h=D_h, roughness=roughness,
                q_wall=q_wall/2.0, isothermal=isothermal, mu=mu_user,
                energy_tol=energy_tol, dPdL_rel_tol=dPdL_rel_tol,
                Ma_change_tol=Ma_change_tol,
                correction_skip_rel_tol=correction_skip_rel_tol,
                _split_depth=_split_depth + 1,
                _max_split_depth=_max_split_depth,
            )
            # First half: fs (P_in, T_in, z_in) -> (P_mid, T_mid, z_in + dz/2)
            compressible_pipe_segment(fs, dL=dL/2.0, **half_kwargs)
            # Second half: fs (P_mid, T_mid, z_in + dz/2) -> (P_out, T_out, z_in + dz)
            compressible_pipe_segment(fs, dL=dL/2.0, **half_kwargs)
            return

        # Converged: apply the Heun (trapezoidal) corrector.  The Euler
        # predictor used inlet-only slopes; averaging the inlet and
        # trial-outlet dP/dL upgrades the pressure step to second order in
        # dL.  The trial-outlet slope was already computed for the splitter,
        # and the flash to the corrected state below replaces the final
        # flash the energy correction always required -- so the corrector
        # costs no additional EOS update.
        P_corr = P_in + dL * (dPdL_in + dPdL_out) / 2.0

        # The outlet temperature comes from the stagnation-enthalpy balance
        # (one Newton step from the trial state), not a dT slope average --
        # energy conservation is the authoritative T equation.
        # H_stagnation = H + v^2/2 + gz
        # H_stagnation_in + q_in = H_stagnation_out
        # error(P, T) = H_stagnation_out_target - [H(P,T) + v(P,T)^2/2]
        # Differentiating the calculated stagnation enthalpy (q_in,
        # H_stagnation_in, and the gz term are constants):
        #   d(H_calc)/dT|_P = (∂H/∂T)_P + (1/2)(∂v^2/∂T)_P
        #                   = Cp - mdot^2/(A^2*rho^3) * (∂ρ/∂T)_P
        #   d(H_calc)/dP|_T = (∂H/∂P)_T + (1/2)(∂v^2/∂P)_T
        #                   = (∂H/∂P)_T - v^2/rho * (∂ρ/∂P)_T
        # The dP cross-term accounts for P moving from the Euler trial P_out
        # to the Heun P_corr while the derivatives stay anchored at the
        # already-flashed trial state.
        Cp_out_state  = AS.cpmass()
        drhodT_P_out  = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        dHdP_T_out    = AS.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)
        drhodP_T_out  = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
        dHcalc_dT = Cp_out_state - mdot**2/(flow_area**2*rho_out_trial**3)*drhodT_P_out
        dHcalc_dP = dHdP_T_out - v_out_trial**2/rho_out_trial*drhodP_T_out
        error_at_corr = energy_error - dHcalc_dP * (P_corr - P_out)
        T_corr = T_out + error_at_corr / dHcalc_dT

        # Skip the corrected flash when the correction is negligible (e.g.
        # near-stagnant slices during network solver probing); the trial
        # state then stands as the outlet.
        if (abs(P_corr - P_out) > correction_skip_rel_tol * P_in
                or abs(T_corr - T_out) > correction_skip_rel_tol * T_in):
            _safe_flowstate_update_PT(fs, P_corr, T_corr)

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
        # We can now Euler method a dP/dL step.
        # (drhodP_T was already evaluated at the inlet for the choke gate.)
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
            _safe_flowstate_update_PT(fs, P_out, T_out)
            # Recompute dP/dL at the trial outlet for the splitter check.  There's
            # no energy balance to check in the isothermal case (T is fixed by
            # assumption), so dP/dL change is the only convergence metric.
            rho_out_trial = AS.rhomass()
            v_out_trial   = mdot / (flow_area * rho_out_trial)
            mu_out = mu_user if mu_user is not None else _viscosity_or_LGE(AS, T_out, rho_out_trial)
            Re_out       = fluids_Reynolds(V=v_out_trial, D=D_h, rho=rho_out_trial, mu=mu_out)
            f_darcy_out  = fluids_friction_factor(Re=Re_out, eD=roughness / D_h)
            drhodP_T_out = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
            denom_out = 1.0 - v_out_trial**2 * drhodP_T_out
            dPdL_out = (-rho_out_trial * f_darcy_out * v_out_trial**2 / (2 * D_h)
                        - rho_out_trial * grav_constant * dz / dL) / denom_out

            # Second splitter metric: Mach-number change across the slice
            # (same role as in the non-isothermal branch).
            Ma_out_trial = v_out_trial / AS.speed_sound()
            Ma_chg       = abs(Ma_out_trial - Ma)

            dPdL_avg    = max(abs(dPdL_in), abs(dPdL_out), 1.0)
            dPdL_relchg = abs(dPdL_out - dPdL_in) / dPdL_avg
            needs_split = (dPdL_relchg > dPdL_rel_tol) or (Ma_chg > Ma_change_tol)
        except (RuntimeError, ValueError) as exc:
            trial_eval_error = exc
            dPdL_relchg      = float('nan')
            Ma_chg           = float('nan')
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
                # The isothermal dP/dL denominator 1 - v^2*(drho/dP)_T goes
                # through zero at the isothermal choke (v = a_T); a value
                # near zero at the (deepest) failing slice's inlet means the
                # singularity sits inside this slice -- name the physical
                # origin.  0.3 corresponds to Ma_T ~ 0.84.
                choke_hint = ""
                if 1.0 - v_in**2 * drhodP_T < 0.3:
                    choke_hint = (
                        "  The dP/dL denominator is near zero -- the flow is "
                        "likely reaching the isothermal choke (v = a_T) "
                        "within this slice; reduce the flow rate."
                    )
                raise RuntimeError(
                    f"compressible_pipe_segment: isothermal slice failed to converge "
                    f"after {_max_split_depth} recursive splits "
                    f"(dL={dL:.4g} m, P_in={P_in:.4g} Pa, T_in={T_in:.4g} K, "
                    f"dPdL_relchg={dPdL_relchg:.3g} [tol={dPdL_rel_tol:.3g}], "
                    f"Ma_change={Ma_chg:.3g} [tol={Ma_change_tol:.3g}]).  "
                    f"Reduce upstream profile slice length, loosen tolerances, "
                    f"or raise _max_split_depth.{choke_hint}"
                )
            _safe_flowstate_update_PT(fs, P_in, T_in)
            half_kwargs = dict(
                dz=dz/2.0, D_h=D_h, roughness=roughness,
                q_wall=q_wall/2.0, isothermal=isothermal, mu=mu_user,
                energy_tol=energy_tol, dPdL_rel_tol=dPdL_rel_tol,
                Ma_change_tol=Ma_change_tol,
                correction_skip_rel_tol=correction_skip_rel_tol,
                _split_depth=_split_depth + 1,
                _max_split_depth=_max_split_depth,
            )
            compressible_pipe_segment(fs, dL=dL/2.0, **half_kwargs)
            compressible_pipe_segment(fs, dL=dL/2.0, **half_kwargs)
            return

        # Converged: Heun (trapezoidal) corrector on P.  T is fixed by the
        # isothermal assumption, so only the pressure step is upgraded to
        # second order.  Unlike the non-isothermal branch this adds one
        # flash (the Euler trial state used to stand as the outlet), traded
        # for second-order accuracy and fewer recursive splits; the skip
        # threshold keeps near-stagnant slices at a single flash.
        P_corr = P_in + dL * (dPdL_in + dPdL_out) / 2.0
        if abs(P_corr - P_out) > correction_skip_rel_tol * P_in:
            _safe_flowstate_update_PT(fs, P_corr, T_out)

    # Outlet choke gate.  fs.AS is at the (converged) outlet state in both
    # branches.  Same mode-dependent Mach definition as the inlet gate.
    rho_out = AS.rhomass()
    v_out   = mdot / (rho_out * flow_area)
    if isothermal:
        Ma_out = v_out * math.sqrt(AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT))
        gate_label = "isothermal Mach number Ma_T = v/a_T"
    else:
        Ma_out = v_out / AS.speed_sound()
        gate_label = "Mach number Ma"
    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_pipe_segment: outlet {gate_label}={Ma_out:.4f} "
            f"is sonic or near-sonic.  Reduce flow rate or check geometry."
        )

    # Advance fs.z by dz so subsequent components see the post-slice elevation.
    # In the recursive-split path each child slice advances its own dz/2, so
    # the parent's bookkeeping does not double-count.
    fs.z += dz


def _fanno_choke_mdot(forward_at_mdot, fs, mdot_seed, mdot_infeasible):
    """Locate the Fanno (friction-limited) choke mass flow of a pipe segment
    by bisecting the feasibility boundary of its forward solve.

    For a distributed-friction pipe the choke is not the frictionless
    isentropic-nozzle mass flux at the throat area -- it is lower, set by the
    Fanno relation, and occurs at the pipe exit.  compressible_pipe_segment
    raises at its Ma>=0.98 reactive gate (see Line_Segment.dP_dT), so the
    largest mdot for which `forward_at_mdot` does *not* raise is the choke.
    The gate's Mach definition is mode-dependent: v/a for adiabatic slices,
    the isothermal Mach v*sqrt((drho/dP)_T) for isothermal slices -- so for
    isothermal segments the feasibility boundary this helper bisects sits
    at the (earlier) isothermal choke, as it physically should.

    Args:
        forward_at_mdot : callable(mdot) -> None.  Restores fs to the segment
                          inlet, sets fs.mdot, and runs the forward physics;
                          leaves fs at the outlet state on success.  Raises
                          (ChokedFlowError / RuntimeError / ValueError) when
                          the requested mdot drives the segment past choke.
        fs              : FlowState the closure drives.  Left at the choke
                          (exit) state on return.
        mdot_seed       : float, a starting guess expected to be feasible
                          (e.g. 0.1 * the ideal-gas nozzle bound).
        mdot_infeasible : float, an upper bound known to exceed the choke
                          (the ideal-gas nozzle bound G_max * A_min); the
                          forward solve is expected to raise at this mdot.

    Returns:
        (mdot_choke, P_exit, T_exit, rho_exit, P_exit, T_exit) -- the 6-tuple
        ChokedFlowError expects.  The pipe exit is the sonic "throat", so the
        throat and outlet states coincide.

    Raises:
        _NoChokeBracketError : if no feasible mdot can be established (the
                               segment fails to integrate even at vanishing
                               flow), so the caller can fall back rather than
                               fabricate a payload.
    """
    def feasible(mdot_trial):
        try:
            forward_at_mdot(mdot_trial)
            return True
        except (ChokedFlowError, RuntimeError, ValueError):
            return False

    # Establish a feasible lower bracket: shrink the seed until the forward
    # solve integrates to the segment end.  Floor the retreat so a genuinely
    # un-integrable segment surfaces as a no-bracket fallback rather than an
    # infinite loop.
    mdot_lo = mdot_seed
    mdot_floor = 1e-9 * mdot_infeasible
    while mdot_lo > mdot_floor and not feasible(mdot_lo):
        mdot_lo *= 0.5
    if mdot_lo <= mdot_floor:
        raise _NoChokeBracketError(
            "_fanno_choke_mdot: could not establish a feasible mass flow; "
            "the segment fails to integrate even near zero flow."
        )

    # Bisect [mdot_lo (feasible), mdot_hi (infeasible)] on feasibility.  The
    # converged mdot_lo is the largest flow the pipe passes subsonically.
    mdot_hi = mdot_infeasible
    for _ in range(60):
        mdot_mid = 0.5 * (mdot_lo + mdot_hi)
        if feasible(mdot_mid):
            mdot_lo = mdot_mid
        else:
            mdot_hi = mdot_mid
        if mdot_hi - mdot_lo <= 1e-6 * mdot_hi:
            break

    # Leave fs at the choke (exit) state at the converged feasible mdot.
    forward_at_mdot(mdot_lo)
    return (mdot_lo, fs.P, fs.T, fs.rho, fs.P, fs.T)


def _solve_mdot_for_outlet_P(
    fs, P2,
    forward_at_mdot,
    mdot_choked,
    mdot_lo=None,
    mdot_hi_frac=0.95,
    mdot_guess=None,
    xtol_factor=1e-6,
    rtol=1e-6,
    maxiter=50,
    caller_name="_solve_mdot_for_outlet_P",
):
    """Brent's-method drive: iterate `mdot` until `forward_at_mdot(mdot)`
    leaves `fs.P` equal to `P2` within tolerance.  Shared engine behind
    Mode 2 of `compressible_dA` and the `dmdot_dT` methods on every
    compressible component class.

    The residual `fs.P - P2` is monotonically decreasing in mdot for any
    K-based friction loss (higher mdot => larger dP => lower P_out), so a
    single bracket on `[mdot_lo, mdot_hi]` is sufficient.  The upper
    bracket is set just below `mdot_choked` and retreats toward zero on
    kernel failure (the subsonic ideal-gas initial-guess generators
    inside the area-change kernel can drift off the real-gas A* near the
    choke and raise).

    Args:
        fs              : FlowState being driven.  `forward_at_mdot` is
                          responsible for restoring `fs` to inlet on each
                          call (brentq evaluates non-monotonically and
                          out of order).
        P2              : float, target outlet pressure [Pa].
        forward_at_mdot : callable(mdot_trial: float) -> None.  Restores
                          `fs` to inlet, sets `fs.mdot = mdot_trial`, and
                          runs the forward physics; on return `fs.P` is
                          the model-predicted outlet pressure at that
                          mdot.
        mdot_choked     : float, upper physical bound on subsonic mdot
                          [kg/s].  Used to set the initial upper bracket.
        mdot_lo         : float or None.  Lower bracket [kg/s].  Default
                          `max(1e-9 * mdot_choked, 1e-3 * mdot_guess)`
                          if `mdot_guess` is supplied, else
                          `1e-9 * mdot_choked`.
        mdot_hi_frac    : float.  Initial upper bracket as a fraction of
                          mdot_choked.  Default 0.95.
        mdot_guess      : float or None.  Incompressible initial estimate
                          [kg/s]; used only to floor `mdot_lo`.
        xtol_factor     : float.  brentq xtol = max(xtol_factor*mdot_choked, 1e-9).
        rtol, maxiter   : forwarded to brentq.
        caller_name     : str.  Used in diagnostic messages.

    Returns:
        float, the converged mdot.  `fs` is left at the outlet state
        corresponding to that mdot (also stored on `fs.mdot`).

    Raises:
        RuntimeError : if the bracket cannot be established (kernel
                       fails at every retreated upper bound), or if the
                       residual has the same sign at both ends of the
                       bracket.

    Caller responsibilities:
        - Pre-screen the choked branch (P2 < P2_choked) before calling;
          this helper only handles the subsonic non-choked solution.
        - Pre-screen the lossless / degenerate K = 0 case (the residual
          is identically P1 - P2 for all mdot and has no root).
    """
    from scipy.optimize import brentq

    if mdot_lo is None:
        if mdot_guess is not None:
            mdot_lo = max(1e-9 * mdot_choked, 1e-3 * mdot_guess)
        else:
            mdot_lo = 1e-9 * mdot_choked

    def residual(mdot_trial):
        forward_at_mdot(mdot_trial)
        return fs.P - P2

    mdot_hi = mdot_hi_frac * mdot_choked

    def _safe_residual_at(mdot_try):
        # If the kernel fails (ChokedFlowError from the pre-screen, a
        # RuntimeError from the ideal-gas initial-guess generator near
        # the real-gas choke, or a ValueError from CoolProp's internal
        # Brent when constant-area K-loss inversion is pushed past the
        # Fanno choke), retreat the upper bracket toward zero and retry
        # until either the kernel succeeds or the retreat budget is
        # exhausted.  Linear shrinks (1.0 -> 0.2) cover near-choke
        # bracket failures; the log-spaced tail (0.1 -> 0.001) covers
        # long pipe segments whose true Fanno choke is orders of
        # magnitude below the ideal-gas G_max * A_min initial bound.
        for shrink in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2,
                       0.1, 0.03, 0.01, 0.003, 0.001):
            mdot_t = mdot_try * shrink
            if mdot_t <= mdot_lo:
                break
            try:
                return mdot_t, residual(mdot_t)
            except (ChokedFlowError, RuntimeError, ValueError):
                continue
        raise RuntimeError(
            f"{caller_name}: kernel failed at every retreated upper bracket "
            f"between {mdot_lo:.4g} and {mdot_try:.4g} kg/s."
        )

    mdot_hi, r_hi = _safe_residual_at(mdot_hi)
    r_lo = residual(mdot_lo)
    # If r_hi is still on the same side as r_lo (positive -- need more mdot
    # to drive fs.P down to P2), the solution sits between mdot_hi and
    # mdot_choked.  This happens whenever P2 is only slightly above the
    # choke recovery pressure.  Progressively push the upper bracket
    # toward mdot_choked until either the sign flips or the kernel
    # consistently fails near choke.
    if r_lo * r_hi > 0.0 and r_hi > 0.0:
        for frac in (0.99, 0.999, 0.9999):
            mdot_try = frac * mdot_choked
            if mdot_try <= mdot_hi:
                continue
            try:
                r_try = residual(mdot_try)
            except (ChokedFlowError, RuntimeError, ValueError):
                break
            mdot_hi, r_hi = mdot_try, r_try
            if r_lo * r_hi <= 0.0:
                break
    if r_lo * r_hi > 0.0:
        raise RuntimeError(
            f"{caller_name}: failed to bracket the non-choked solution "
            f"(residual at mdot_lo={mdot_lo:.4g}: {r_lo:.4g}; "
            f"at mdot_hi={mdot_hi:.4g}: {r_hi:.4g}). "
            f"P2={P2:.4g} Pa, mdot_choked={mdot_choked:.4g} kg/s."
        )

    mdot_solution = brentq(
        residual,
        mdot_lo, mdot_hi,
        xtol=max(xtol_factor * mdot_choked, 1e-9),
        rtol=rtol,
        maxiter=maxiter,
    )

    # brentq leaves fs at the state of its last function eval, which is
    # not necessarily the converged root.  One extra eval guarantees fs
    # is at the outlet state corresponding to mdot_solution.
    residual(mdot_solution)
    return mdot_solution


def compressible_dA(fs, A_throat, K = 0.0, A2 = None, P2 = None):
    """
    Models a compressible fitting/component with a constriction (valve, orifice). 
    State 1 is the inlet conditions (pressure, temperature, area)
    State th is at the throat/vena cava condition
    State 2 is the outlet condition
        -----------------
                |
    --> 1       th       2
                |
        -----------------
    There are two operating modes, dictated by whether P2 is passed or not. For both modes, the
    fluid is assumed to accelerate isentropically from State 1 to the throat state. From the throat state to State 2, the
    passed K value is used to perform an entropy balance to find the outlet condition (unless flow is choked, as discussed below).
    For most constrictions, most of the friction energy loss occus on the downstream side of the constriction. Using the average
    of the throat conditions and the downstream conditions for calculating the entropy change of the process therefore
    gives us a better estimate of the real process than using the inlet and outlet conditions for this calculation.

    Mode 1 - Dictate mass flow, solve for downstream pressure P2
    If no P2 value is passed, the function operates to calculate the downstream pressure and update the flowstate
    to this pressure. Two conservation equations are satisfied, stagnation enthalpy (energy) and entropy. 
    The function first performs a choke check - if the given mass flow rate fs.mdot is greater than the choked 
    flow rate, the function raises an error.

    If no choking occurs, the fluid is assumed to isentropically accelerate to the throat condition, also satisfying
    an energy balance (adiabadic, no elevation change - no change in stagnation enthalpy):

        H_th + 0.5*v_th^2 = H1 + 0.5*v1^2
        S_th = S1

    From the throat condition to the outlet condition, the change in entropy is given by:

        S2 = S1 + 0.5 * (1/T) * K * v1**2 * (rho_in/rho_throat)**2

    where K is the incompressible permanent-loss coefficient (referenced to the inlet velocity head at the INLET density)
    and the temperature is the average of the throat temperature and the outlet temperature. The (rho_in/rho_throat)**2
    factor scales the inlet-referenced velocity head to the actual throat velocity head: in compressible flow the throat
    gas has expanded, so the real throat velocity (and the dissipated head) exceeds the incompressible estimate by this
    factor. It equals 1.0 in the low-Mach limit, leaving low-Mach results unchanged, and rises toward full throat-KE
    dissipation as the throat approaches sonic -- without it the sonic throat KE is nearly fully recovered as pressure,
    over-predicting downstream pressure recovery and the choke threshold.

    And again, energy is conserved from the inlet conditions:
        H2 + 0.5 * v2^2 = H1 + 0.5 * v1^2

    Mode 2 - Dictate downstream pressure P2, solve for mass flow
    If P2 is passed, the function operates to calculate the mass flow rate that satisfies the given pressure change.
    First, the function performs a choked flow check to estimate the choked flow rate and choked throat pressure at the inlet condition
    and establish an upper limit for the solver. 
    
    An energy and entropy balance at choked conditions is then performed to solve for the downstream pressure that would be expected at
    choked flow, P2_choked. If P2_choked is greater than the P2 value passed to the function, then this means that flow is choked. 
    Another energy balance is performed to solve for the actual outlet temperature. 

    If the choked P2 is less than the P2 passed to the function, the flow is not choked. Entropy accounting calculates outlet entropy
    that should be expected and the solver iterates the temperature until the outlet T2, P2 combo gives the required outlet entropy.

        S2 = S1 + 0.5 * (1/T) * K * v1**2 * (rho_in/rho_throat)**2


    """
    # ---- Input validation
    if K < 0.0:
        raise ValueError(
            f"compressible_dA: K must be non-negative (got {K})."
        )
    if A_throat <= 0.0 or A_throat > fs.A:
        raise ValueError(
            f"compressible_dA: A_throat must be in (0, fs.A]; "
            f"got A_throat={A_throat:.4g} m^2, fs.A={fs.A:.4g} m^2."
        )
    if P2 is not None and P2 >= fs.P:
        raise ValueError(
            f"compressible_dA: P2 must be strictly less than the inlet pressure "
            f"(got P2={P2:.4g} Pa, P_in={fs.P:.4g} Pa)."
        )

    # Record inlet state.
    P1     = fs.P
    T1     = fs.T
    A1     = fs.A
    rho_in = fs.rho
    v1     = fs.mdot / (A1 * rho_in)

    # If A2 is not passed, it is assumed to be equal to fs.A.
    if A2 is None:
        A2 = fs.A

    if P2 is None:
        # ---- Mode 1: dictate mdot, solve for outlet pressure.
        # Isentropic acceleration to throat (kernel performs its own choke check).
        compressible_changing_area_K(fs=fs, A_out=A_throat, K=0.0)

        # K is the incompressible permanent-loss coefficient (inlet velocity
        # head evaluated at the INLET density).  In compressible flow the
        # throat gas has expanded, so the actual throat velocity -- and the
        # dissipated velocity head -- is larger by (rho_in/rho_throat)**2.
        # This factor is 1.0 in the low-Mach limit, so low-Mach results are
        # unchanged.  fs is now at the throat, so fs.rho is rho_throat.
        rho_throat = fs.rho
        e_loss = 0.5 * K * v1**2 * (rho_in / rho_throat)**2

        # Recovery to A2 with the K-derived dissipation.
        compressible_changing_area_K(fs=fs, A_out=A2, e_loss=e_loss)
        return

    # ---- Mode 2: dictate P2, solve for mdot.
    # Find the choked mass flow rate for the given upstream stagnation state.
    # choked_mass_flux leaves fs.AS at the throat state when A_outlet is None.
    mdot_choked, P_choke, T_choke, rho_choke, _, _ = choked_mass_flux(
        fs=fs, A_throat=A_throat,
    )

    # March throat -> A2 with K-based dissipation to find the outlet pressure
    # produced by the choked-flow scenario.  At this point fs.AS is at the
    # throat (sonic) state; re-marching inlet -> throat at mdot_choked would
    # be degenerate (the subsonic root collapses onto Ma=1), so we instead
    # advance directly from the throat.  The K reference velocity is the inlet
    # velocity at the choked rate, scaled to the actual throat velocity head by
    # (rho_in/rho_throat)**2 (see the Mode 1 comment).  Without this factor the
    # sonic throat KE is nearly fully recovered as pressure, putting P2_choked
    # far above the true sonic throat pressure and declaring choke at too high
    # an outlet pressure.
    fs.mdot = mdot_choked
    fs.A    = A_throat
    v1_choked  = mdot_choked / (rho_in * A1)
    e_loss_chk = 0.5 * K * v1_choked**2 * (rho_in / rho_choke)**2
    compressible_changing_area_K(fs=fs, A_out=A2, e_loss=e_loss_chk)
    P2_choked = fs.P

    def _take_choked_branch(fs_at_recovery):
        """Adiabatically expand from the choked-recovery state to P2 and
        enforce the supersonic-outlet guard.  fs is mutated in place."""
        adiabatic_expansion_solver(fs=fs_at_recovery, P2=P2)
        try:
            a_out = fs_at_recovery.AS.speed_sound()
        except (ValueError, RuntimeError):
            a_out = 0.0
        if a_out > 0.0 and fs_at_recovery.v / a_out > 1.0 + _MA_SONIC_TOL:
            raise RuntimeError(
                f"compressible_dA: choked-branch adiabatic expansion to "
                f"P2={P2:.4g} Pa produced a supersonic outlet "
                f"(Ma_out={fs_at_recovery.v / a_out:.4f}) at A2={A2:.4g} m^2. "
                f"P2 is below the post-throat free-expansion limit at this "
                f"geometry; the result would require a shock or additional "
                f"dissipation that this function does not model."
            )

    if P2 < P2_choked:
        # ---- Choked branch: mdot is fixed at mdot_choked. The downstream
        # state expands adiabatically (stagnation enthalpy conserved) to
        # the user-supplied P2. fs is currently at the choked-recovery
        # state, so adiabatic_expansion_solver's snapshot of fs.h_stagnation
        # equals the inlet stagnation enthalpy (the entire path is adiabatic).
        _take_choked_branch(fs)
        return

    # ---- Non-choked branch: iterate mdot until the two-stage march reaches
    # P2. K = 0 + non-choked is physically degenerate (no loss => P2 == P1
    # in the subsonic limit), so reject it before entering brentq.
    if K <= 0.0:
        raise ValueError(
            f"compressible_dA: non-choked solution with K=0 is degenerate "
            f"(zero loss requires P2 == P_in for subsonic flow). "
            f"P2={P2:.4g} Pa, P_in={P1:.4g} Pa, P2_choked={P2_choked:.4g} Pa."
        )

    # Incompressible initial-guess for mdot:
    #   dP = 0.5 * K * rho * v^2,  mdot = rho * A * v
    #   => mdot = A1 * sqrt(2 * dP * rho_in / K)
    mdot_guess = A1 * math.sqrt(2.0 * (P1 - P2) * rho_in / K)

    def forward_at_mdot(mdot_trial):
        # Restore inlet conditions; the kernels mutate fs in place.
        # skip_choke_check=True on both kernel calls: mdot_hi is capped
        # at 0.95*mdot_choked by _solve_mdot_for_outlet_P, so the
        # ~150 ms per-call choked_mass_flux pre-screen is wasted work
        # here -- mdot_choked is already known from the outer
        # choked_mass_flux call above.
        _safe_flowstate_update_PT(fs, P1, T1)
        fs.A    = A1
        fs.mdot = mdot_trial
        v1_t   = mdot_trial / (rho_in * A1)
        compressible_changing_area_K(fs=fs, A_out=A_throat, K=0.0,
                                     skip_choke_check=True)
        # Scale the inlet-referenced K loss to the actual throat velocity head
        # (see the Mode 1 comment); fs is at the throat after the call above.
        rho_throat_t = fs.rho
        e_loss = 0.5 * K * v1_t**2 * (rho_in / rho_throat_t)**2
        compressible_changing_area_K(fs=fs, A_out=A2, e_loss=e_loss,
                                     skip_choke_check=True)

    try:
        _solve_mdot_for_outlet_P(
            fs, P2,
            forward_at_mdot=forward_at_mdot,
            mdot_choked=mdot_choked,
            mdot_guess=mdot_guess,
            caller_name="compressible_dA",
        )
    except RuntimeError as e:
        if "failed to bracket" not in str(e):
            raise
        # Near-choke fallback: P2 sits in the numerically indeterminate
        # band just above P2_choked, where the subsonic root of the
        # two-stage march degenerates toward Ma=1 at the throat and the
        # kernel can no longer evaluate cleanly enough to bracket.  Re-
        # anchor fs to the cached throat sonic state, march to A2, then
        # adiabatically expand to P2 -- the choked-branch path.
        import warnings as _warnings
        _safe_flowstate_update_PT(fs, P_choke, T_choke)
        fs.mdot = mdot_choked
        fs.A    = A_throat
        compressible_changing_area_K(fs=fs, A_out=A2, e_loss=e_loss_chk)
        _take_choked_branch(fs)
        _warnings.warn(
            f"compressible_dA: outlet pressure P2={P2:.4g} Pa is within "
            f"the near-choke numerical band above P2_choked="
            f"{P2_choked:.4g} Pa; reporting choked conditions "
            f"(mdot={mdot_choked:.4g} kg/s).",
            UserWarning,
            stacklevel=2,
        )


def adiabatic_expansion_solver(fs, P2, T_ITER_MAX = 8, H_ABS_TOL = 1.0, H_REL_TOL = 1.0e-9):
    """
    Perform an adiabadic expansion to P2 at area fs.A with stagnation enthalpy conserved. Iterate the outlet temperature until the energy conservation is satisfied.
    
    v_out is recomputed each iteration from the current rho_out so the
    residual is on full h_stagnation. 
    Args:
    fs      :FlowState at the inlet (static). fs.AS is mutated
    P2      :float, target pressure (Pa)
    T_ITER_MAX  : Maximum number of Newton solver iterations
    H_ABS_TOL   : Minimum absolute stagnation enthalpy error since last solve step before solver returns
    H_REL_TOL   : Minimum relative stagnation enthalpy error since last solve step before solver returns
    """
    # Adiabatic: stagnation enthalpy is invariant along the process, so the
    # snapshot taken at entry is the correct target for every Newton step.
    h_stagnation = fs.h_stagnation
    T_out = fs.T
    for _ in range(T_ITER_MAX):
        _safe_flowstate_update_PT(fs, P2, T_out)
        v_out  = fs.v
        h_calc = fs.AS.hmass() + 0.5 * v_out ** 2
        err    = h_calc - h_stagnation
        if abs(err) <= max(H_ABS_TOL, H_REL_TOL * abs(h_stagnation)):
            return
        cp = fs.AS.cpmass()
        if cp <= 0.0:
            break
        T_out -= err / cp
    #NOTE if not converged, should we warn?