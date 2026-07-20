"""incompressible.py

Incompressible (liquid) pipeline hydraulics.

This module provides fluid property storage, pipe-component classes with
pressure-drop calculations, and helper functions for computing and exporting
pressure profiles along liquid pipelines.

The component classes (Line_Segment, Bend, Contraction_Expansion) inherit
geometry storage and CSV-loading logic from the base classes in
component_classes.py, adding incompressible-flow pressure-drop methods.

Classes
-------
Incompressible_Fluid
    Stores density and dynamic viscosity for an incompressible liquid.

Line_Segment  (inherits Base_Line_Segment)
    Adds pressure_profile() and dP() for incompressible flow.

Bend  (inherits Base_Bend)
    Adds dP() using the fluids.fittings.bend_rounded() K-factor correlation.

Valve  (inherits Base_Valve)
    Adds dP() using the pre-computed K-factor stored on the instance.

Contraction_Expansion  (inherits Base_Contraction_Expansion)
    Adds dP() using Bernoulli velocity-head exchange plus K-factor loss.

Module-level functions
----------------------
_resolve_flow_rate(flow_rate, rho_si)
    Convert a pint Quantity flow rate to m^3/s.

dP_friction(fluid, flow_rate, flow_area, eps, D_h, dL)
    Darcy-Weisbach friction loss for a uniform pipe length.

dP_bend(fluid, flow_rate, Di, angle_deg, bend_dias)
    Pressure loss through a rounded bend.

dP_contraction_expansion(fluid, flow_rate, Di_US, Di_DS)
    Total pressure change through an abrupt contraction or expansion.

print_results(fluid, segment, flow_rate, dP_total_Pa)
    Print a formatted summary of a pressure-drop result.

export_pressure_profile(segment, fluid, flow_rate, output_path, P0)
    Write a per-point pressure and velocity profile to a CSV file.
"""

import csv
import math
import warnings
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import fluids.fittings
import fluids.flow_meter
from scipy.optimize import brentq

from component_classes import (
    Base_Line_Segment,
    Base_Bend,
    Base_Contraction_Expansion,
    Base_Valve,
    Base_CheckValve,
    Base_Orifice,
    _to_si,
    _resolve_id,
    _flow_props_from_id,
    ureg,
)


# ---------------------------------------------------------------------------
# Incompressible_Fluid class
# ---------------------------------------------------------------------------

class Incompressible_Fluid:
    """Properties of an incompressible (liquid) fluid.

    Density is the primary stored property (kg/m^3, SI).  The fluid can be
    constructed directly from a density value, or via the class method
    from_api_gravity() for petroleum liquids.

    Args:
        density          : pint Quantity or plain float (kg/m^3 if float).
        viscosity        : pint Quantity or plain float (Pa*s if float).
        vapor_pressure   : pint Quantity or plain float (Pa if float),
                           optional.  Required by the Orifice / Valve /
                           CheckValve cavitation checks.
        critical_pressure: pint Quantity or plain float (Pa if float),
                           optional.  Used only by the Valve / CheckValve
                           cavitation check to compute the ISA F_F factor
                           (F_F = 0.96 - 0.28*sqrt(Pv/Pc)).  When omitted,
                           F_F defaults to 0.96 -- the low-Pv/Pc asymptote,
                           which is mildly conservative.
    """

    def __init__(self, density, viscosity, vapor_pressure=None,
                 critical_pressure=None):
        if hasattr(density, "to"):
            self.density_si = density.to("kg/m^3").magnitude
        else:
            self.density_si = float(density)

        if hasattr(viscosity, "to"):
            self.viscosity_si = viscosity.to("Pa*s").magnitude
        else:
            self.viscosity_si = float(viscosity)

        if vapor_pressure is not None:
            if hasattr(vapor_pressure, "to"):
                self.vapor_pressure_si = vapor_pressure.to("Pa").magnitude
            else:
                self.vapor_pressure_si = float(vapor_pressure)
        else:
            self.vapor_pressure_si = None

        if critical_pressure is not None:
            if hasattr(critical_pressure, "to"):
                self.critical_pressure_si = critical_pressure.to("Pa").magnitude
            else:
                self.critical_pressure_si = float(critical_pressure)
        else:
            self.critical_pressure_si = None

    @classmethod
    def from_api_gravity(cls, api_gravity, viscosity):
        """Construct an Incompressible_Fluid from API gravity.

        Density is derived using the standard API formula:
            rho [kg/m^3] = 1000 * 141.5 / (API + 131.5)

        Args:
            api_gravity: float, dimensionless API degrees.
            viscosity  : pint Quantity or plain float (Pa*s if float).
        """
        rho_si = 1000.0 * 141.5 / (float(api_gravity) + 131.5)
        return cls(density=rho_si, viscosity=viscosity)

    def __repr__(self):
        rho_q = ureg.Quantity(self.density_si, "kg/m^3")
        mu_q  = ureg.Quantity(self.viscosity_si, "Pa*s")
        return (
            f"Incompressible_Fluid("
            f"density={rho_q.to('lb/ft^3'):.4f~P} [{rho_q:.4f~P}], "
            f"viscosity={mu_q.to('cP'):.4f~P} [{mu_q:.6f~P}])"
        )


# ---------------------------------------------------------------------------
# Internal flow-rate helper
# ---------------------------------------------------------------------------

def _resolve_flow_rate(flow_rate, rho_si):
    """Return volumetric flow rate [m^3/s] from a pint Quantity.

    Accepts volumetric ([length]^3/[time]) or mass ([mass]/[time]) flow rates.

    Args:
        flow_rate : pint Quantity.
        rho_si    : float, fluid density [kg/m^3], used for mass-to-volume
                    conversion.

    Returns:
        float, volumetric flow rate [m^3/s].

    Raises:
        ValueError : if flow_rate is not a pint Quantity or has unrecognized
                     dimensions.
    """
    if not hasattr(flow_rate, "dimensionality"):
        raise ValueError(
            "flow_rate must be a pint Quantity with dimensions of "
            "[length]^3/[time] (volumetric) or [mass]/[time] (mass flow)."
        )
    dim = flow_rate.dimensionality
    if dim == {"[length]": 3, "[time]": -1}:
        return flow_rate.to("m^3/s").magnitude
    elif dim == {"[mass]": 1, "[time]": -1}:
        return flow_rate.to("kg/s").magnitude / rho_si
    else:
        raise ValueError(
            f"flow_rate has unrecognized dimensions {dict(dim)}. "
            "Expected [length]^3/[time] (volumetric) or [mass]/[time] "
            "(mass flow rate)."
        )


# ---------------------------------------------------------------------------
# Line_Segment -- incompressible child class
# ---------------------------------------------------------------------------

class Line_Segment(Base_Line_Segment):
    """Pipe segment with incompressible-flow pressure-drop calculations.

    Inherits all geometry storage, CSV loading, and convenience properties
    from Base_Line_Segment.  Adds pressure_profile() and dP() for
    incompressible (liquid) flow using the Darcy-Weisbach friction equation,
    hydrostatic elevation head, and Bernoulli / K-factor area-change
    corrections.

    Constructor arguments and behavior are identical to Base_Line_Segment.
    See Base_Line_Segment for full argument documentation.
    """

    # ------------------------------------------------------------------
    # Pressure and velocity profile
    # ------------------------------------------------------------------

    def pressure_profile(self, fluid, P0, flow_rate):
        """Calculate the absolute static pressure and velocity at each profile
        point along the segment, starting from a known inlet pressure.

        Steps through consecutive profile point pairs.  For each slice the
        pressure change has up to three contributions:

        1. Friction loss (Darcy-Weisbach):
               dP_friction = -(f_D / 2) * (rho * v^2 / D_h) * dL   [Pa]
           Always negative (opposes flow).

        2. Elevation (hydrostatic):
               dP_elevation = -rho * g * dz                          [Pa]
           Positive if flowing downhill (dz < 0), negative if uphill.

        3. Area-change at the inter-slice boundary (only when the flow area
           changes between consecutive slices):

           a. Bernoulli velocity-head exchange (recoverable, affects static P):
                  dP_Bernoulli = (1/2) * rho * (v_in^2 - v_out^2)   [Pa]
              Positive for an expansion (v_out < v_in, static P rises),
              negative for a contraction (v_out > v_in, static P falls).

           b. Permanent loss (K-factor, always negative; circular pipes only):
                  dP_loss = -K * (1/2) * rho * v_ref^2              [Pa]
              K and v_ref are determined by fluids.fittings:
                - Contraction (Di_in > Di_out): contraction_sharp, K w.r.t.
                  downstream velocity; converted to upstream reference for a
                  consistent sign convention.
                - Expansion (Di_in < Di_out): diffuser_sharp, K w.r.t.
                  upstream velocity.

        The velocity at each profile point is computed from the flow area at
        that point.  The per-slice friction and elevation contributions use the
        inlet-point velocity, consistent with the forward-Euler convention used
        throughout.

        For noncircular=True, contribution 3b is omitted entirely because the
        sharp-transition correlations assume a circular bore.

        Args:
            fluid     : Incompressible_Fluid instance.
            P0        : float, absolute static pressure at the segment inlet
                        [Pa].
            flow_rate : pint Quantity -- volumetric ([length]^3/[time]) or
                        mass ([mass]/[time]).  Mass flow is converted to
                        volumetric using fluid.density_si.

        Returns:
            list of dicts, one per profile point (including the inlet).
            Each dict contains:
                "distance_m"  : float, along-pipe distance from segment
                                inlet [m].
                "elevation_m" : float, elevation [m].
                "P_Pa"        : float, absolute static pressure [Pa].
                "v_ms"        : float, mean flow velocity [m/s].

        Raises:
            ValueError : if flow_rate has unrecognized dimensions, the
                         profile contains only one point, or any slice has
                         a non-positive length.
        """
        if len(self.profile) < 2:
            raise ValueError(
                "Line_Segment.pressure_profile: profile must have at least "
                "two points."
            )

        grav_constant = 9.8066   # m/s^2
        _AREA_TOL     = 1e-9     # m^2, absolute tolerance for area equality

        rho = fluid.density_si
        mu  = fluid.viscosity_si
        eps = self.roughness_si
        Q   = _resolve_flow_rate(flow_rate, rho)

        # --- Inlet point (index 0) ---
        dist_0, elev_0, D_h_0, area_0 = self.profile[0]
        results = [{
            "distance_m":  dist_0,
            "elevation_m": elev_0,
            "P_Pa":        float(P0),
            "v_ms":        Q / area_0,
        }]

        P_cur = float(P0)
        n     = len(self.profile)

        for i in range(n - 1):
            dist_in,  elev_in,  D_h_in,  area_in  = self.profile[i]
            dist_out, elev_out, D_h_out, area_out = self.profile[i + 1]

            dL = dist_out - dist_in   # m, along-pipe slice length
            dz = elev_out - elev_in   # m, positive = uphill

            if dL <= 0.0:
                raise ValueError(
                    f"Line_Segment.pressure_profile: non-positive slice length "
                    f"dL={dL:.6g} m between profile points {i} and {i+1}.  "
                    f"Ensure the profile is sorted by distance with no "
                    f"duplicate distances."
                )

            v_in    = Q / area_in
            Re      = fluids_Reynolds(V=v_in, D=D_h_in, rho=rho, mu=mu)
            f_darcy = fluids_friction_factor(Re=Re, eD=eps / D_h_in)

            # 1. Friction loss over this slice.
            dP_friction = -(f_darcy / 2.0) * rho * v_in ** 2 / D_h_in * dL

            # 2. Elevation (hydrostatic) over this slice.
            dP_elevation = -rho * grav_constant * dz

            P_cur += dP_friction + dP_elevation

            # 3. Area-change correction at the boundary to the next slice.
            if abs(area_out - area_in) > _AREA_TOL:
                v_out = Q / area_out

                # 3a. Bernoulli velocity-head exchange.
                dP_bernoulli = 0.5 * rho * (v_in ** 2 - v_out ** 2)

                # 3b. Permanent K-factor loss (circular pipes only).
                dP_loss = 0.0
                if not self.noncircular:
                    Di_in  = math.sqrt(4.0 * area_in  / math.pi)
                    Di_out = math.sqrt(4.0 * area_out / math.pi)

                    if Di_in > Di_out:
                        # Contraction: K expressed w.r.t. downstream velocity;
                        # convert to upstream reference for consistent sign.
                        K_ds  = fluids.fittings.contraction_sharp(
                            Di1=Di_in, Di2=Di_out
                        )
                        K_us  = K_ds * (Di_out / Di_in) ** 4   # (A_out/A_in)^2
                        dP_loss = -K_us * 0.5 * rho * v_in ** 2
                    else:
                        # Expansion: K expressed w.r.t. upstream velocity.
                        K_us    = fluids.fittings.diffuser_sharp(
                            Di1=Di_in, Di2=Di_out
                        )
                        dP_loss = -K_us * 0.5 * rho * v_in ** 2

                P_cur += dP_bernoulli + dP_loss

            results.append({
                "distance_m":  dist_out,
                "elevation_m": elev_out,
                "P_Pa":        P_cur,
                "v_ms":        Q / area_out,
            })

        return results

    # ------------------------------------------------------------------
    # Scalar pressure drop (convenience wrapper)
    # ------------------------------------------------------------------

    def dP(self, fluid, flow_rate):
        """Return the total static pressure change along the segment [Pa].

        Delegates to pressure_profile() using an arbitrary inlet pressure of
        0 Pa, then returns outlet_P - inlet_P.  Negative means pressure
        decreases in the flow direction.

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.

        Returns:
            float, total static pressure change [Pa].
        """
        profile = self.pressure_profile(fluid, P0=0.0, flow_rate=flow_rate)
        return profile[-1]["P_Pa"] - profile[0]["P_Pa"]

    # ------------------------------------------------------------------
    # Inverse: solve mass flow rate for a target outlet pressure
    # ------------------------------------------------------------------

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Forward mass flow rate [kg/s] producing the requested outlet pressure.

        Inverts pressure_profile() via a 1-D brentq on mdot:

            residual(mdot) = pressure_profile(P0=P_inlet, mdot)[-1]['P_Pa']
                             - P_outlet

        Closed-form inversion is not feasible because each slice carries
        its own f(Re), the staircase profile can change area between
        slices (Bernoulli + K-loss corrections), and elevation `-rho*g*dz`
        is independent of mdot.  A 1-D bracketing solve is robust and
        handles all of these uniformly.

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at the segment
                       inlet [Pa].
            P_outlet : float, absolute static pressure at the segment
                       outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Positive for forward flow.

        Raises:
            ValueError : if the brentq bracket cannot be established (e.g.
                         elevation gain exceeds available head, or the
                         segment cannot reach P_outlet for any forward
                         flow).
        """
        rho = fluid.density_si

        def residual(mdot_si):
            Q_pq = ureg.Quantity(mdot_si, "kg/s")
            prof = self.pressure_profile(fluid, P0=P_inlet, flow_rate=Q_pq)
            return prof[-1]["P_Pa"] - P_outlet

        # Initial upper bound from a fully turbulent friction estimate
        # (f ~ 0.02) using the *minimum* hydraulic diameter / area across
        # the profile -- yields the loosest, most generous upper bound for
        # the staircase pipe.  A small positive headroom protects against
        # negative `P_inlet - P_outlet` when elevation flips the sign.
        D_min   = min(pt[2] for pt in self.profile)
        A_min   = min(pt[3] for pt in self.profile)
        L_total = self.profile[-1][0] - self.profile[0][0]
        dP_avail = max(P_inlet - P_outlet, 1.0)
        Q_hi_est = A_min * math.sqrt(
            2.0 * dP_avail * D_min / (0.02 * rho * L_total)
        )
        mdot_hi  = max(rho * Q_hi_est * 5.0, 1.0)

        # Verify the bracket; expand mdot_hi up to ~3 decades if needed.
        f_lo = residual(1e-9)
        f_hi = residual(mdot_hi)
        for _ in range(3):
            if f_lo * f_hi < 0.0:
                break
            mdot_hi *= 10.0
            f_hi = residual(mdot_hi)
        else:
            raise ValueError(
                f"Line_Segment.dmdot: could not bracket a forward mdot "
                f"(P_inlet={P_inlet:.6g} Pa, P_outlet={P_outlet:.6g} Pa, "
                f"residual(0)={f_lo:.3g}, residual({mdot_hi:.3g})={f_hi:.3g}).  "
                f"Elevation head may exceed available dP, or the segment "
                f"cannot reach P_outlet in forward flow."
            )

        return brentq(residual, 1e-9, mdot_hi, xtol=1e-6, rtol=1e-8)


# ---------------------------------------------------------------------------
# Bend -- incompressible child class
# ---------------------------------------------------------------------------

class Bend(Base_Bend):
    """Rounded pipe bend fitting with incompressible pressure-drop calculation.

    Inherits geometry storage and validation from Base_Bend.  Adds dP() using
    the fluids.fittings.bend_rounded() K-factor correlation.

    Constructor arguments are identical to Base_Bend:
        Di        : pint Quantity or float (m if float).  Pipe inner diameter.
        ang_deg   : float.  Bend angle [degrees].
        bend_dias : float.  Bend radius as a multiple of Di.
    """

    def dP(self, fluid, flow_rate):
        """Permanent pressure loss through the bend [Pa].

        Uses the fluids.fittings.bend_rounded() K-factor correlation, which
        accounts for bend angle, bend radius, and Reynolds number:

            dP = -K * (1/2) * rho * v^2

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.
                        Same conventions as Line_Segment.dP().

        Returns:
            float, pressure change [Pa].  Always <= 0 (loss).
        """
        rho = fluid.density_si
        mu  = fluid.viscosity_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0

        Q = _resolve_flow_rate(flow_rate, rho)
        v = Q / A

        Re = fluids_Reynolds(V=v, D=Di, rho=rho, mu=mu)
        K  = fluids.fittings.bend_rounded(
            Di=Di,
            bend_diameters=self.bend_dias,
            angle=self.ang_deg,
            Re=Re,
        )
        return -K * 0.5 * rho * v ** 2

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Mass flow rate [kg/s] producing the requested pressure change.

        K depends on Reynolds number, which depends on Q.  Solved by a
        short fixed-point loop (typically 2-3 iterations to converge):
        seed K at a high-Re value, solve Q analytically from K, recompute
        Re and K at the new Q, repeat until K stabilizes.

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at inlet [Pa].
                       Unused by the bend physics; kept for signature
                       uniformity across components.
            P_outlet : float, absolute static pressure at outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Positive for forward flow.

        Raises:
            ValueError : if P_inlet <= P_outlet.
        """
        dP_drop = P_inlet - P_outlet
        if dP_drop <= 0.0:
            raise ValueError(
                f"Bend.dmdot: requires P_inlet > P_outlet "
                f"(got dP_drop={dP_drop:.6g} Pa).  Forward flow only."
            )
        rho = fluid.density_si
        mu  = fluid.viscosity_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0

        K = fluids.fittings.bend_rounded(
            Di=Di, bend_diameters=self.bend_dias,
            angle=self.ang_deg, Re=1e6,
        )
        for _ in range(8):
            Q_new  = A * math.sqrt(2.0 * dP_drop / (K * rho))
            Re_new = fluids_Reynolds(V=Q_new / A, D=Di, rho=rho, mu=mu)
            K_new  = fluids.fittings.bend_rounded(
                Di=Di, bend_diameters=self.bend_dias,
                angle=self.ang_deg, Re=Re_new,
            )
            if abs(K_new - K) / max(K, 1e-12) < 1e-8:
                K = K_new
                break
            K = K_new
        else:
            warnings.warn(
                f"Bend.dmdot: K(Re) fixed-point did not converge in 8 "
                f"iterations (last drift {abs(K_new-K):.2e}).",
                UserWarning, stacklevel=2,
            )
        Q = A * math.sqrt(2.0 * dP_drop / (K * rho))
        return rho * Q


# ---------------------------------------------------------------------------
# Cavitation check (shared by Valve and CheckValve)
# ---------------------------------------------------------------------------

def _valve_cavitation_check(component, fluid, P_inlet, dP_perm):
    """Three-regime ISA-75.01 cavitation check for a liquid valve fitting.

    Reference: https://www.osti.gov/biblio/10155405 (LANL valve cavitation
    review, page 13 and following).

    Silent (no-op) unless all of the following are present:
        (1) component.F_L is not None
        (2) fluid.vapor_pressure_si is not None
        (3) P_inlet is not None
    and dP_perm < 0 (an actual loss).

    Regimes (in order of severity):
        1. Flashing: P_out < P_v.  Downstream is two-phase; incompressible
           model invalid.  Raises RuntimeError.
        2. Choked cavitating: |dP| >= F_L^2 * (P_in - F_F * P_v), with
           F_F = 0.96 - 0.28*sqrt(P_v/P_c) when fluid.critical_pressure_si
           is known (else F_F = 0.96 -- the low-Pv/Pc asymptote).  Mass
           flow no longer responds to downstream pressure; solver
           derivative invalid.  Raises RuntimeError.
        3. Incipient: sigma = (P_in - P_v) / |dP| < 1 / F_L^2.  Trim
           erosion expected; warns UserWarning.

    Args:
        component : Valve or CheckValve instance (provides F_L and name).
        fluid     : Incompressible_Fluid instance.
        P_inlet   : float, absolute static pressure at inlet [Pa], or None.
        dP_perm   : float, permanent pressure loss [Pa] (<= 0, sign matches
                    dP() return convention).
    """
    Pv = getattr(fluid, "vapor_pressure_si", None)
    F_L = getattr(component, "F_L", None)
    if Pv is None or P_inlet is None or F_L is None:
        return
    if dP_perm >= 0.0:
        return

    dP_abs   = -dP_perm
    P_out    = P_inlet + dP_perm
    cls_name = type(component).__name__
    name_str = f"'{component.name}'" if component.name else cls_name.lower()

    if P_out < Pv:
        raise RuntimeError(
            f"{cls_name} {name_str}: flashing detected "
            f"(P_out={P_out:.4g} Pa < Pv={Pv:.4g} Pa).  Downstream is "
            f"two-phase; incompressible model invalid."
        )

    Pc  = getattr(fluid, "critical_pressure_si", None)
    F_F = 0.96 - 0.28 * math.sqrt(Pv / Pc) if Pc else 0.96

    dP_choked = F_L ** 2 * (P_inlet - F_F * Pv)
    if dP_abs >= dP_choked:
        raise RuntimeError(
            f"{cls_name} {name_str}: choked cavitating flow "
            f"(|dP|={dP_abs:.4g} Pa >= F_L^2*(P1-F_F*Pv)="
            f"{dP_choked:.4g} Pa, F_L={F_L:.3f}, F_F={F_F:.3f}).  "
            f"Mass flow no longer responds to downstream pressure; "
            f"reduce flow or raise back-pressure."
        )

    sigma           = (P_inlet - Pv) / dP_abs
    sigma_incipient = 1.0 / F_L ** 2
    if sigma < sigma_incipient:
        warnings.warn(
            f"{cls_name} {name_str}: incipient cavitation possible "
            f"(sigma={sigma:.3f} < 1/F_L^2={sigma_incipient:.3f}, "
            f"F_L={F_L:.3f}).  Trim erosion likely; consider raising "
            f"back-pressure.",
            UserWarning,
            stacklevel=4,
        )


# ---------------------------------------------------------------------------
# Valve -- incompressible child class
# ---------------------------------------------------------------------------

class Valve(Base_Valve):
    """Valve fitting with incompressible pressure-drop calculation.

    Inherits geometry storage and validation from Base_Valve.  Adds dP() using
    the pre-computed K-factor stored on the instance.

    When the fluid carries vapor_pressure_si and the valve carries F_L, an
    ISA-75.01 three-regime cavitation check (flashing / choked cavitating /
    incipient) fires whenever P_inlet is supplied to dP() or dmdot() is
    called.  See Valve._cavitation_check for the criteria and references.

    Constructor arguments are inherited from Base_Valve (Di, K|Cv|Kv, name,
    minimum_diameter, F_L).
    """

    def dP(self, fluid, flow_rate, P_inlet=None):
        """Permanent pressure loss through the valve [Pa].

        Uses the K-factor stored on the instance:

            dP = -K * (1/2) * rho * v^2

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.
                        Same conventions as Line_Segment.dP().
            P_inlet   : float or None.  Absolute static pressure at the
                        valve inlet [Pa].  Required for the cavitation
                        check; ignored if None, if fluid.vapor_pressure_si
                        is None, or if self.F_L is None.

        Returns:
            float, pressure change [Pa].  Always <= 0 (loss).

        Raises:
            RuntimeError : if flashing or choked cavitation is detected
                           (only when P_inlet, fluid.vapor_pressure_si, and
                           self.F_L are all available).

        Warns:
            UserWarning  : if incipient cavitation is possible.
        """
        rho = fluid.density_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0

        Q = _resolve_flow_rate(flow_rate, rho)
        v = Q / A

        dP_perm = -self.K * 0.5 * rho * v ** 2
        self._cavitation_check(fluid, P_inlet, dP_perm)
        return dP_perm

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Mass flow rate [kg/s] that produces the requested pressure change.

        Analytic inverse of dP() since K is constant:

            Q = A * sqrt(2 * (P_inlet - P_outlet) / (K * rho))
            mdot = rho * Q

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at inlet [Pa].
            P_outlet : float, absolute static pressure at outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Positive for forward flow.

        Raises:
            ValueError   : if P_inlet <= P_outlet (no forward solution).
            RuntimeError : if cavitation gating (see dP()) trips.
        """
        dP_drop = P_inlet - P_outlet
        if dP_drop <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.dmdot: requires P_inlet > P_outlet "
                f"(got dP_drop={dP_drop:.6g} Pa).  Forward flow only."
            )
        if self.K <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.dmdot: K must be > 0 to invert "
                f"(got K={self.K})."
            )
        rho = fluid.density_si
        A   = math.pi * self.Di_si ** 2 / 4.0
        Q   = A * math.sqrt(2.0 * dP_drop / (self.K * rho))
        self._cavitation_check(fluid, P_inlet, -dP_drop)
        return rho * Q

    def _cavitation_check(self, fluid, P_inlet, dP_perm):
        """Three-regime ISA-75.01 cavitation check (see _valve_cavitation_check)."""
        _valve_cavitation_check(self, fluid, P_inlet, dP_perm)


# ---------------------------------------------------------------------------
# CheckValve -- incompressible child class
# ---------------------------------------------------------------------------

class CheckValve(Base_CheckValve):
    """Check valve with incompressible pressure-drop calculation.

    Forward flow uses the K-factor stored on the instance (computed from a
    Crane check-valve correlation and passed at construction).  Reverse flow
    is a network-level concern: Network.solve treats a check valve as a
    perfect seal and pins its edge to exactly zero reverse flow via a
    complementarity residual, so dP()/dmdot() model forward passage only.

    Constructor arguments are identical to Base_CheckValve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Forward-flow K-factor (from a Crane correlation).
    """

    def dP(self, fluid, flow_rate, P_inlet=None):
        """Forward pressure loss through the check valve [Pa].

            dP = -K * (1/2) * rho * v^2

        Only called for positive (forward) flow; the network solver seals
        the edge at exactly zero flow under reverse conditions.

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.
            P_inlet   : float or None.  Absolute static pressure at the
                        valve inlet [Pa].  Required for the cavitation
                        check (see Valve.dP for details); ignored if None,
                        if fluid.vapor_pressure_si is None, or if
                        self.F_L is None.

        Returns:
            float, pressure change [Pa].  Always <= 0 (loss).

        Raises:
            RuntimeError : if flashing or choked cavitation is detected.

        Warns:
            UserWarning  : if incipient cavitation is possible.
        """
        rho = fluid.density_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0
        Q   = _resolve_flow_rate(flow_rate, rho)
        v   = Q / A
        dP_perm = -self.K * 0.5 * rho * v ** 2
        _valve_cavitation_check(self, fluid, P_inlet, dP_perm)
        return dP_perm

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Forward mass flow rate [kg/s] producing the requested pressure change.

        Analytic inverse of dP() since K is constant.  Forward flow only;
        reverse conditions are a network-level perfect seal (zero flow).

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at inlet [Pa].
            P_outlet : float, absolute static pressure at outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Positive for forward flow.

        Raises:
            ValueError   : if P_inlet <= P_outlet (no forward solution).
            RuntimeError : if cavitation gating trips (see dP()).
        """
        dP_drop = P_inlet - P_outlet
        if dP_drop <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.dmdot: requires P_inlet > P_outlet "
                f"(got dP_drop={dP_drop:.6g} Pa).  Forward flow only."
            )
        if self.K <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.dmdot: K must be > 0 to invert "
                f"(got K={self.K})."
            )
        rho = fluid.density_si
        A   = math.pi * self.Di_si ** 2 / 4.0
        Q   = A * math.sqrt(2.0 * dP_drop / (self.K * rho))
        _valve_cavitation_check(self, fluid, P_inlet, -dP_drop)
        return rho * Q


# ---------------------------------------------------------------------------
# Orifice -- incompressible child class
# ---------------------------------------------------------------------------

class Orifice(Base_Orifice):
    """Square-edged concentric orifice plate with incompressible pressure-drop
    calculation.

    Inherits geometry storage and validation from Base_Orifice.  Adds dP()
    using the fluids library Reader-Harris-Gallagher discharge-coefficient
    correlation to compute the non-recoverable (permanent) pressure loss.

    A cavitation index, sigma, is calculated based on the following reference (see page 13)
    https://www.osti.gov/biblio/10155405

    sigma = (upstream pressure - vapor pressure)/(permanent pressure drop across restriction)

    Pages 74-77 of this reference give correlations for incipient and choked cavitation for a circular, concentric, sharp, thin orifice
    
    sigma_incipient = 1.55 + 4.88 * Cd + 5.66 * Cd**2 + 1.95 * Cd**3
    sigma_choked = 1.08 + 2.28 * Cd - 4.38 * Cd**2 + 7.57 * Cd ** 3

    Constructor arguments are identical to Base_Orifice:
        Di          : pint Quantity or float (m if float).  Pipe inner diameter.
        Do          : pint Quantity or float (m if float).  Orifice bore diameter.
        Cd_override : float or None.  Fixed Cd; bypasses the RHG correlation.
    """

    def dP(self, fluid, flow_rate, P_inlet=None):
        """Non-recoverable pressure loss through the orifice plate [Pa].

        Computes the discharge coefficient via the ISO 5167-2 Reader-Harris-
        Gallagher correlation (or uses Cd_override if set), converts to a
        K-factor referenced to the upstream pipe velocity, and returns:

            dP = -K * (1/2) * rho * v_pipe^2

        If P_inlet is supplied and fluid.vapor_pressure_si is set, a
        cavitation check is performed using the sigma index.

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.
            P_inlet   : float or None.  Absolute static pressure at the
                        orifice inlet [Pa].  Required for cavitation check;
                        ignored if None or if fluid.vapor_pressure_si is None.

        Returns:
            float, permanent pressure change [Pa].  Always <= 0 (loss).

        Raises:
            RuntimeError : if choked cavitation is detected
                           (sigma < sigma_choked).

        Warns:
            UserWarning  : if incipient cavitation is possible
                           (sigma < sigma_incipient).
        """
        rho = fluid.density_si
        mu  = fluid.viscosity_si
        Di  = self.Di_si
        Do  = self.Do_si

        A_pipe = math.pi * Di ** 2 / 4.0
        Q      = _resolve_flow_rate(flow_rate, rho)
        v_pipe = Q / A_pipe
        m      = rho * Q

        if self.Cd_override is not None:
            Cd = self.Cd_override
        else:
            Cd = fluids.flow_meter.C_Reader_Harris_Gallagher(
                D=Di, Do=Do, rho=rho, mu=mu, m=m, taps=self.taps
            )
        sigma_incipient = 1.55 + 4.88 * Cd + 5.66 * Cd**2 + 1.95 * Cd**3
        sigma_choked = 1.08 + 2.28 * Cd - 4.38 * Cd**2 + 7.57 * Cd ** 3
        K        = fluids.flow_meter.discharge_coefficient_to_K(D=Di, Do=Do, C=Cd)
        dP_perm  = -K * 0.5 * rho * v_pipe ** 2

        if (
            P_inlet is not None
            and getattr(fluid, "vapor_pressure_si", None) is not None
            and dP_perm < 0.0
        ):
            Pv       = fluid.vapor_pressure_si
            dP_abs   = -dP_perm
            sigma    = (P_inlet - Pv) / dP_abs
            name_str = f"'{self.name}'" if self.name else "orifice"
            if sigma < sigma_choked:
                raise RuntimeError(
                    f"Orifice {name_str}: choked cavitation detected "
                    f"(sigma={sigma:.3f} < {sigma_choked:.3f}).  "
                    f"Reduce flow rate or increase system back-pressure."
                )
            if sigma < sigma_incipient:
                warnings.warn(
                    f"Orifice {name_str}: incipient cavitation possible "
                    f"(sigma={sigma:.3f} < {sigma_incipient:.3f}).  "
                    f"Consider increasing back-pressure.",
                    UserWarning,
                    stacklevel=2,
                )

        return dP_perm

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Mass flow rate [kg/s] producing the requested pressure change.

        Cd from the Reader-Harris-Gallagher correlation depends on Reynolds
        number, which depends on mass flow rate.  Solved by a short
        fixed-point loop (typically 2-3 iterations) on Cd <-> mdot.  When
        self.Cd_override is set the loop collapses to a single analytic
        evaluation.

        If fluid.vapor_pressure_si is set, the same sigma-based cavitation
        check used by dP() fires after the mdot solve (sigma_choked raises
        RuntimeError; sigma_incipient warns).

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at inlet [Pa].
            P_outlet : float, absolute static pressure at outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Positive for forward flow.

        Raises:
            ValueError   : if P_inlet <= P_outlet.
            RuntimeError : if choked cavitation is detected.

        Warns:
            UserWarning  : if incipient cavitation is possible, or if the
                           Cd fixed-point does not converge in 8 iterations.
        """
        dP_drop = P_inlet - P_outlet
        if dP_drop <= 0.0:
            raise ValueError(
                f"Orifice.dmdot: requires P_inlet > P_outlet "
                f"(got dP_drop={dP_drop:.6g} Pa).  Forward flow only."
            )
        rho    = fluid.density_si
        mu     = fluid.viscosity_si
        Di     = self.Di_si
        Do     = self.Do_si
        A_pipe = math.pi * Di ** 2 / 4.0

        if self.Cd_override is not None:
            Cd = self.Cd_override
        else:
            Cd = 0.6   # typical RHG asymptote
            for _ in range(8):
                K_iter = fluids.flow_meter.discharge_coefficient_to_K(
                    D=Di, Do=Do, C=Cd,
                )
                Q_iter = A_pipe * math.sqrt(2.0 * dP_drop / (K_iter * rho))
                m_iter = rho * Q_iter
                Cd_new = fluids.flow_meter.C_Reader_Harris_Gallagher(
                    D=Di, Do=Do, rho=rho, mu=mu, m=m_iter, taps=self.taps,
                )
                if abs(Cd_new - Cd) / max(Cd, 1e-12) < 1e-8:
                    Cd = Cd_new
                    break
                Cd = Cd_new
            else:
                warnings.warn(
                    f"Orifice {self.name!r}: dmdot Cd fixed-point did not "
                    f"converge in 8 iterations (last drift "
                    f"{abs(Cd_new - Cd):.2e}).",
                    UserWarning, stacklevel=2,
                )

        K = fluids.flow_meter.discharge_coefficient_to_K(D=Di, Do=Do, C=Cd)
        Q = A_pipe * math.sqrt(2.0 * dP_drop / (K * rho))

        Pv = getattr(fluid, "vapor_pressure_si", None)
        if Pv is not None:
            sigma_incipient = 1.55 + 4.88 * Cd + 5.66 * Cd ** 2 + 1.95 * Cd ** 3
            sigma_choked    = 1.08 + 2.28 * Cd - 4.38 * Cd ** 2 + 7.57 * Cd ** 3
            sigma           = (P_inlet - Pv) / dP_drop
            name_str = f"'{self.name}'" if self.name else "orifice"
            if sigma < sigma_choked:
                raise RuntimeError(
                    f"Orifice {name_str}: choked cavitation detected "
                    f"(sigma={sigma:.3f} < {sigma_choked:.3f}).  "
                    f"Reduce flow rate or increase system back-pressure."
                )
            if sigma < sigma_incipient:
                warnings.warn(
                    f"Orifice {name_str}: incipient cavitation possible "
                    f"(sigma={sigma:.3f} < {sigma_incipient:.3f}).  "
                    f"Consider increasing back-pressure.",
                    UserWarning, stacklevel=2,
                )

        return rho * Q


# ---------------------------------------------------------------------------
# Contraction_Expansion -- incompressible child class
# ---------------------------------------------------------------------------

class Contraction_Expansion(Base_Contraction_Expansion):
    """Abrupt contraction or expansion with incompressible pressure-drop
    calculation.

    Inherits geometry storage and validation from Base_Contraction_Expansion.
    Adds dP() using Bernoulli velocity-head exchange plus a K-factor
    permanent loss.

    Constructor arguments are identical to Base_Contraction_Expansion:
        Di_US : pint Quantity or float (m if float).  Upstream inner diameter.
        Di_DS : pint Quantity or float (m if float).  Downstream inner diameter.
    """

    def dP(self, fluid, flow_rate):
        """Total static pressure change through the contraction/expansion [Pa].

        Two contributions are summed:

        1. Bernoulli velocity-head exchange (recoverable, affects static P):
               dP_Bernoulli = (1/2) * rho * (v_US^2 - v_DS^2)   [Pa]

        2. Permanent K-factor loss (always <= 0):
               Contraction (Di_US > Di_DS): fluids.fittings.contraction_sharp()
               Expansion   (Di_US < Di_DS): fluids.fittings.diffuser_sharp()

        If Di_US == Di_DS both contributions are zero.

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.

        Returns:
            float, pressure change [Pa].
        """
        rho   = fluid.density_si
        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0

        Q    = _resolve_flow_rate(flow_rate, rho)
        v_US = Q / A_US
        v_DS = Q / A_DS

        # Bernoulli: static pressure change due to velocity change.
        dP_bernoulli = 0.5 * rho * (v_US ** 2 - v_DS ** 2)

        if abs(Di_US - Di_DS) < 1e-12:
            dP_loss = 0.0
        elif Di_US > Di_DS:
            # Contraction: K w.r.t. downstream velocity; convert to upstream.
            K_ds    = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K_us    = K_ds * (A_DS / A_US) ** 2
            dP_loss = -K_us * 0.5 * rho * v_US ** 2
        else:
            # Expansion: K w.r.t. upstream velocity.
            K_us    = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)
            dP_loss = -K_us * 0.5 * rho * v_US ** 2

        return dP_bernoulli + dP_loss

    def dmdot(self, fluid, P_inlet, P_outlet):
        """Mass flow rate [kg/s] producing the requested pressure change.

        Analytic inverse of dP().  Both Bernoulli and K-factor terms are
        proportional to Q^2, with the proportionality determined entirely
        by geometry:

            dP_static = P_outlet - P_inlet = 0.5 * rho * Q^2 * beta
            beta      = (1/A_US^2 - 1/A_DS^2) - K_us/A_US^2

        For a contraction (Di_US > Di_DS) beta < 0, so a forward solution
        requires P_inlet > P_outlet.  For an expansion (Di_US < Di_DS)
        beta can be either sign depending on whether Bernoulli recovery
        outweighs the diffuser loss; the formula handles either case.

        Args:
            fluid    : Incompressible_Fluid instance.
            P_inlet  : float, absolute static pressure at inlet [Pa].
            P_outlet : float, absolute static pressure at outlet [Pa].

        Returns:
            float, mass flow rate [kg/s].  Always positive (forward flow).

        Raises:
            ValueError : if Di_US == Di_DS (any mdot satisfies P_in == P_out),
                         or if no real forward solution exists for the
                         requested pressure pair.
        """
        rho   = fluid.density_si
        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        if abs(Di_US - Di_DS) < 1e-12:
            raise ValueError(
                "Contraction_Expansion.dmdot: undefined for equal diameters "
                "(any mdot satisfies P_inlet == P_outlet)."
            )

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0

        if Di_US > Di_DS:
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K_us = K_ds * (A_DS / A_US) ** 2
        else:
            K_us = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)

        beta      = (1.0 / A_US ** 2 - 1.0 / A_DS ** 2) - K_us / A_US ** 2
        dP_static = P_outlet - P_inlet
        Q_squared = 2.0 * dP_static / (rho * beta)
        if Q_squared <= 0.0:
            raise ValueError(
                f"Contraction_Expansion.dmdot: no real forward solution "
                f"(P_inlet={P_inlet:.6g} Pa, P_outlet={P_outlet:.6g} Pa, "
                f"beta={beta:.6g}).  For contractions, P_inlet must exceed "
                f"P_outlet; for expansions, the sign of (P_outlet - P_inlet) "
                f"must match the sign of beta."
            )
        Q = math.sqrt(Q_squared)
        return rho * Q


# ---------------------------------------------------------------------------
# Module-level convenience pressure-drop functions
# ---------------------------------------------------------------------------

def dP_friction(fluid, flow_rate, flow_area, eps, D_h, dL):
    """Permanent friction pressure loss for a uniform pipe length [Pa].

    Darcy-Weisbach equation:
        dP = -(f_D / 2) * (rho * v^2 / D_h) * dL

    Args:
        fluid     : Incompressible_Fluid instance.
        flow_rate : pint Quantity, volumetric or mass flow rate.
        flow_area : float, pipe flow cross-section area [m^2].
        eps       : float, absolute pipe roughness [m].
        D_h       : float, hydraulic diameter [m].
        dL        : float, pipe length [m].

    Returns:
        float, pressure change [Pa].  Always <= 0.
    """
    for name, val in [("flow_area", flow_area), ("eps", eps),
                      ("D_h", D_h), ("dL", dL)]:
        if val < 0.0:
            raise ValueError(
                f"dP_friction: {name} must be >= 0, got {val}."
            )
    rho     = fluid.density_si
    mu      = fluid.viscosity_si
    Q       = _resolve_flow_rate(flow_rate, rho)
    v       = Q / flow_area
    Re      = fluids_Reynolds(V=v, D=D_h, rho=rho, mu=mu)
    f_darcy = fluids_friction_factor(Re=Re, eD=(eps / D_h))
    return -(f_darcy / 2.0) * rho * v ** 2 / D_h * dL


def dP_bend(fluid, flow_rate, Di, angle_deg, bend_dias):
    """Permanent pressure loss through a rounded pipe bend [Pa].

    Uses the fluids.fittings.bend_rounded() K-factor correlation.

    Args:
        fluid     : Incompressible_Fluid instance.
        flow_rate : pint Quantity, volumetric or mass flow rate.
        Di        : float, pipe inner diameter [m].
        angle_deg : float, bend angle [degrees].
        bend_dias : float, bend radius in multiples of Di.

    Returns:
        float, pressure change [Pa].  Always <= 0.
    """
    for name, val in [("Di", Di), ("angle_deg", angle_deg),
                      ("bend_dias", bend_dias)]:
        if val < 0.0:
            raise ValueError(
                f"dP_bend: {name} must be >= 0, got {val}."
            )
    rho = fluid.density_si
    mu  = fluid.viscosity_si
    Q   = _resolve_flow_rate(flow_rate, rho)
    A   = math.pi * Di ** 2 / 4.0
    v   = Q / A
    Re  = fluids_Reynolds(V=v, D=Di, rho=rho, mu=mu)
    K   = fluids.fittings.bend_rounded(
        Di=Di, bend_diameters=bend_dias, angle=angle_deg, Re=Re
    )
    return -K * 0.5 * rho * v ** 2


def dP_contraction_expansion(fluid, flow_rate, Di_US, Di_DS):
    """Total static pressure change through an abrupt contraction or expansion
    [Pa].

    Includes both the Bernoulli velocity-head exchange (recoverable) and the
    permanent K-factor loss.  If Di_US == Di_DS, returns 0.

    Args:
        fluid     : Incompressible_Fluid instance.
        flow_rate : pint Quantity, volumetric or mass flow rate.
        Di_US     : float, upstream inner diameter [m].
        Di_DS     : float, downstream inner diameter [m].

    Returns:
        float, pressure change [Pa].
    """
    for name, val in [("Di_US", Di_US), ("Di_DS", Di_DS)]:
        if val <= 0.0:
            raise ValueError(
                f"dP_contraction_expansion: {name} must be positive, got {val}."
            )
    return Contraction_Expansion(Di_US, Di_DS).dP(fluid, flow_rate)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(fluid, segment, flow_rate, dP_total_Pa):
    """Print a formatted summary of a Line_Segment.dP() result.

    Args:
        fluid        : Incompressible_Fluid instance.
        segment      : Line_Segment instance.
        flow_rate    : pint Quantity passed to dP().
        dP_total_Pa  : float, value returned by segment.dP().
    """
    rho   = fluid.density_si
    Q     = _resolve_flow_rate(flow_rate, rho)
    first = segment.profile[0]
    A_in  = first[3]
    v     = Q / A_in

    rho_q = ureg.Quantity(rho,         "kg/m^3")
    A_q   = ureg.Quantity(A_in,        "m^2")
    vol_q = ureg.Quantity(segment.volume_m3, "m^3")
    v_q   = ureg.Quantity(v,           "m/s")
    dP_q  = ureg.Quantity(dP_total_Pa, "Pa")

    print("=== Liquid Hydraulics Results ===")
    print(f"  Flow area (inlet)    : {A_q.to('in^2'):.4f~P}  ({A_q:.6f~P})")
    print(f"  Velocity (inlet)     : {v_q.to('ft/s'):.4f~P}  ({v_q:.4f~P})")
    print(f"  Line volume          : {vol_q.to('oil_bbl'):.4f~P}  ({vol_q:.4f~P})")
    print(f"  Fluid density        : {rho_q.to('lb/ft^3'):.4f~P}  ({rho_q:.4f~P})")
    print(f"  dP total             : {dP_q.to('psi'):.4f~P}  ({dP_q:.2f~P})")


def export_pressure_profile(segment, fluid, flow_rate, output_path, P0=0.0):
    """Write a per-point pressure and velocity profile to a CSV file.

    Calls segment.pressure_profile() to compute the profile, then writes one
    row per profile point.  The first row is always the inlet (point 0).

    Columns written:
        point        : integer point index (0 = inlet).
        distance_m   : along-pipe distance from inlet [m].
        elevation_m  : elevation [m].
        P_Pa         : absolute static pressure [Pa].
        v_ms         : mean flow velocity [m/s].

    Args:
        segment     : Line_Segment instance.
        fluid       : Incompressible_Fluid instance.
        flow_rate   : pint Quantity, volumetric or mass flow rate.
        output_path : str, path for the output CSV.
        P0          : float, absolute inlet pressure [Pa].  Default 0.0
                      (produces a gauge-pressure-change profile when no
                      absolute reference is available).
    """
    profile_pts = segment.pressure_profile(fluid, P0=P0, flow_rate=flow_rate)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point",
            "distance_m",
            "elevation_m",
            "P_Pa",
            "v_ms",
        ])
        for idx, pt in enumerate(profile_pts):
            writer.writerow([
                idx,
                f"{pt['distance_m']:.6f}",
                f"{pt['elevation_m']:.6f}",
                f"{pt['P_Pa']:.4f}",
                f"{pt['v_ms']:.6f}",
            ])

    print(f"  Pressure profile exported to: {output_path}")

