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

from component_classes import (
    Base_Line_Segment,
    Base_Bend,
    Base_Contraction_Expansion,
    Base_Valve,
    Base_CheckValve,
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
        density  : pint Quantity or plain float (kg/m^3 if float).
        viscosity: pint Quantity or plain float (Pa*s if float).
    """

    def __init__(self, density, viscosity):
        if hasattr(density, "to"):
            self.density_si = density.to("kg/m^3").magnitude
        else:
            self.density_si = float(density)

        if hasattr(viscosity, "to"):
            self.viscosity_si = viscosity.to("Pa*s").magnitude
        else:
            self.viscosity_si = float(viscosity)

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


# ---------------------------------------------------------------------------
# Valve -- incompressible child class
# ---------------------------------------------------------------------------

class Valve(Base_Valve):
    """Valve fitting with incompressible pressure-drop calculation.

    Inherits geometry storage and validation from Base_Valve.  Adds dP() using
    the pre-computed K-factor stored on the instance.

    Constructor arguments are identical to Base_Valve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Resistance coefficient (K-factor), referenced to the
             pipe velocity head.
    """

    def dP(self, fluid, flow_rate):
        """Permanent pressure loss through the valve [Pa].

        Uses the K-factor stored on the instance:

            dP = -K * (1/2) * rho * v^2

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.
                        Same conventions as Line_Segment.dP().

        Returns:
            float, pressure change [Pa].  Always <= 0 (loss).
        """
        rho = fluid.density_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0

        Q = _resolve_flow_rate(flow_rate, rho)
        v = Q / A

        return -self.K * 0.5 * rho * v ** 2


# ---------------------------------------------------------------------------
# CheckValve -- incompressible child class
# ---------------------------------------------------------------------------

class CheckValve(Base_CheckValve):
    """Check valve with incompressible pressure-drop calculation.

    Forward flow uses the K-factor stored on the instance (computed from a
    Crane check-valve correlation and passed at construction).  Reverse flow
    is handled by network._reversed_component, which returns a shadow copy
    with K = _SEALING_K (≈ 1e9), making the valve act as a near-perfect seal.

    Constructor arguments are identical to Base_CheckValve:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
        K  : float.  Forward-flow K-factor (from a Crane correlation).
    """

    def dP(self, fluid, flow_rate):
        """Forward pressure loss through the check valve [Pa].

            dP = -K * (1/2) * rho * v^2

        Only called for positive (forward) flow; the network solver invokes a
        high-K shadow for reverse flow via network._reversed_component().

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric or mass flow rate.

        Returns:
            float, pressure change [Pa].  Always <= 0 (loss).
        """
        rho = fluid.density_si
        Di  = self.Di_si
        A   = math.pi * Di ** 2 / 4.0
        Q   = _resolve_flow_rate(flow_rate, rho)
        v   = Q / A
        return -self.K * 0.5 * rho * v ** 2


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

