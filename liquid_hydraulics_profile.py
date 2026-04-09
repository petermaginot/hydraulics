import csv
import math
import os
import warnings
from pint import UnitRegistry

ureg = UnitRegistry()


# ---------------------------------------------------------------------------
# Fluid class
# ---------------------------------------------------------------------------

class Incompressible_Fluid:
    """Properties of an incompressible (liquid) fluid.

    Density is the primary stored property (kg/m^3, SI).  The fluid can be
    constructed directly from a density value, or via the class method
    from_api_gravity() for petroleum liquids.

    Args:
        density  : pint Quantity or plain float (kg/m^3 if float).
                   The density of the fluid.
        viscosity: pint Quantity or plain float (Pa*s if float).
                   Dynamic viscosity of the fluid.
    """

    def __init__(self, density, viscosity):
        # Accept pint Quantities or plain floats (assumed SI).
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
# Pipe segment class
# ---------------------------------------------------------------------------

class Line_Segment:
    """Geometric and profile properties of a single pipe segment.

    Inner diameter can be specified directly or derived from outer diameter
    and wall thickness.  Exactly one of the following must be true:
      - Only id_val is supplied.
      - Both od_val and wt_val are supplied (id is derived as od - 2*wt).
      - All three are supplied: a consistency check is performed and a warning
        is issued if they disagree by more than the tolerance; id_val is used.

    The cross-section is assumed circular unless a custom flow_area and
    hydraulic_diameter are supplied, in which case those values are used
    directly (non-circular / hydraulic-radius mode).

    Args:
        id_val            : pint Quantity or float (m if float). Inner diameter.
                            Optional if od_val and wt_val are both given.
        od_val            : pint Quantity or float (m if float). Outer diameter.
                            Optional.
        wt_val            : pint Quantity or float (m if float). Wall thickness.
                            Optional.
        roughness         : pint Quantity or float (m if float). Absolute
                            pipe-wall roughness.
        profile           : list of (distance, elevation) tuples as pint
                            Quantities or plain floats (m if float).
                            If None, a two-point profile [(0, 0), (L, dz)] is
                            generated from length and elevation_change.
        length            : pint Quantity or float (m if float). Used only when
                            profile is None.
        elevation_change  : pint Quantity or float (m if float). Net elevation
                            change over the segment, positive = uphill.
                            Used only when profile is None.
        flow_area         : pint Quantity or float (m^2 if float). Override the
                            computed circular cross-section area.  Supply
                            together with hydraulic_diameter for non-circular
                            geometries.
        hydraulic_diameter: pint Quantity or float (m if float). Override the
                            hydraulic diameter used for Re and friction factor.
                            For a circular pipe this equals the inner diameter.
                            For non-circular sections: D_h = 4 * A / P_wetted.
        id_tolerance      : float, fractional tolerance for the id/od/wt
                            consistency check (default 0.001 = 0.1 %).
    """

    def __init__(
        self,
        roughness,
        id_val=None,
        od_val=None,
        wt_val=None,
        profile=None,
        length=None,
        elevation_change=None,
        flow_area=None,
        hydraulic_diameter=None,
        id_tolerance=0.001,
    ):
        # --- Resolve inner diameter (SI, metres) ---
        id_si = self._resolve_id(id_val, od_val, wt_val, id_tolerance)
        self.id_si = id_si

        # Store od/wt if provided (for reference only; not used in calculations).
        self.od_si = self._to_si_or_none(od_val, "m")
        self.wt_si = self._to_si_or_none(wt_val, "m")

        # --- Roughness ---
        if hasattr(roughness, "to"):
            self.roughness_si = roughness.to("m").magnitude
        else:
            self.roughness_si = float(roughness)

        # --- Flow area and hydraulic diameter ---
        # For a circular pipe both default to functions of the inner diameter.
        # Non-circular callers supply explicit overrides.
        if flow_area is not None:
            if hasattr(flow_area, "to"):
                self.flow_area_si = flow_area.to("m^2").magnitude
            else:
                self.flow_area_si = float(flow_area)
        else:
            # Circular cross-section: A = pi/4 * d^2
            self.flow_area_si = math.pi * id_si ** 2 / 4.0

        if hydraulic_diameter is not None:
            if hasattr(hydraulic_diameter, "to"):
                self.hydraulic_diameter_si = hydraulic_diameter.to("m").magnitude
            else:
                self.hydraulic_diameter_si = float(hydraulic_diameter)
        else:
            # Circular pipe: hydraulic diameter equals inner diameter.
            self.hydraulic_diameter_si = id_si

        # --- Elevation profile (stored in SI: metres) ---
        # profile takes precedence over length / elevation_change.
        if profile is not None:
            raw = self._normalize_profile(profile)
        elif length is not None:
            L_si  = length.to("m").magnitude if hasattr(length, "to") else float(length)
            dz_si = 0.0
            if elevation_change is not None:
                dz_si = (
                    elevation_change.to("m").magnitude
                    if hasattr(elevation_change, "to")
                    else float(elevation_change)
                )
            raw = [(0.0, 0.0), (L_si, dz_si)]
        else:
            raise ValueError(
                "Line_Segment requires either a profile list or a length."
            )

        # _normalize_profile returns a list sorted by distance with a
        # zero-distance point prepended if needed.
        self.profile = raw   # list of (distance_m, elevation_m) float tuples

    # --- Internal helpers ---

    @staticmethod
    def _to_si(val, unit):
        """Convert a pint Quantity or plain float to SI magnitude."""
        if val is None:
            return None
        return val.to(unit).magnitude if hasattr(val, "to") else float(val)

    @staticmethod
    def _to_si_or_none(val, unit):
        if val is None:
            return None
        return val.to(unit).magnitude if hasattr(val, "to") else float(val)

    @staticmethod
    def _resolve_id(id_val, od_val, wt_val, tol):
        """Return inner diameter in metres, performing consistency checks."""
        id_si = Line_Segment._to_si(id_val, "m")
        od_si = Line_Segment._to_si(od_val, "m")
        wt_si = Line_Segment._to_si(wt_val, "m")

        if id_si is None and od_si is None:
            raise ValueError(
                "Line_Segment: supply id_val, or both od_val and wt_val."
            )

        if od_si is not None and wt_si is None:
            raise ValueError(
                "Line_Segment: od_val supplied without wt_val. "
                "Provide both to derive inner diameter."
            )

        if od_si is not None and wt_si is not None:
            id_derived = od_si - 2.0 * wt_si
            if id_si is not None:
                # All three supplied -- check consistency.
                if abs(id_si - id_derived) > tol * max(id_si, id_derived):
                    warnings.warn(
                        f"Line_Segment: id/od/wt are inconsistent "
                        f"(id={id_si:.6f} m, od-2*wt={id_derived:.6f} m). "
                        f"Proceeding with the supplied id_val.",
                        UserWarning,
                        stacklevel=3,
                    )
                # id_val takes precedence when all three are supplied.
                return id_si
            return id_derived

        # Only id_val supplied.
        return id_si

    @staticmethod
    def _normalize_profile(raw_profile):
        """Sort profile by distance, convert Quantities to float metres, and
        prepend a synthetic zero-distance point if the first entry is not
        already at distance = 0.

        Each entry in raw_profile may be either:
          - a (distance, elevation) tuple of pint Quantities, or
          - a (distance_m, elevation_m) tuple of plain floats.
        """
        converted = []
        for dist, elev in raw_profile:
            d_m = dist.to("m").magnitude if hasattr(dist, "to") else float(dist)
            e_m = elev.to("m").magnitude if hasattr(elev, "to") else float(elev)
            converted.append((d_m, e_m))

        converted.sort(key=lambda r: r[0])

        if converted[0][0] != 0.0:
            converted = [(0.0, converted[0][1])] + converted

        return converted

    @classmethod
    def from_csv(
        cls,
        csv_path,
        roughness,
        id_val=None,
        od_val=None,
        wt_val=None,
        flow_area=None,
        hydraulic_diameter=None,
        id_tolerance=0.001,
    ):
        """Construct a Line_Segment by loading an elevation profile from a CSV.

        Expected CSV columns (by position, header row is skipped):
          col 0 - ignored (layer ID or similar)
          col 1 - distance from origin (m).
                  This assumes an absolute along-pipe distance (e.g., from an
                  inline inspection odometer log) that already accounts for
                  elevation changes.  Survey X-Y plane distances must be
                  pre-processed to include the elevation component:
                      s = sqrt(delta_z^2 + delta_xy^2)
                  Survey data is often oversampled; consider downsampling to
                  standard ~13 m (40 ft) pipe joint lengths before use.
          col 2 - elevation (m).

        Args:
            csv_path          : path to the elevation profile CSV file.
            roughness         : pint Quantity or float (m if float).
            id_val, od_val, wt_val, flow_area, hydraulic_diameter, id_tolerance:
                                same as Line_Segment.__init__.
        """
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)                          # skip header row
            for line in reader:
                dist_m = float(line[1])
                elev_m = float(line[2])
                rows.append((dist_m, elev_m))

        if not rows:
            raise ValueError(f"Elevation profile CSV '{csv_path}' contains no data rows.")

        total_dist_m = max(r[0] for r in rows)
        net_elev_m   = rows[-1][1] - rows[0][1]  # approximate; profile stores exact values
        print(
            f"  Elevation profile loaded from '{csv_path}': {len(rows)} points, "
            f"total distance = {total_dist_m:.2f} m, "
            f"net elevation change = {net_elev_m:.4f} m"
        )

        return cls(
            roughness=roughness,
            id_val=id_val,
            od_val=od_val,
            wt_val=wt_val,
            profile=rows,
            flow_area=flow_area,
            hydraulic_diameter=hydraulic_diameter,
            id_tolerance=id_tolerance,
        )

    # --- Convenience properties ---

    @property
    def total_length_m(self):
        """Total along-pipe distance of this segment (m)."""
        return self.profile[-1][0]

    @property
    def net_elevation_change_m(self):
        """Net elevation change (last point minus first point) in metres.
        Positive = uphill overall."""
        return self.profile[-1][1] - self.profile[0][1]

    @property
    def volume_m3(self):
        """Internal volume of the segment (m^3)."""
        return self.flow_area_si * self.total_length_m

    def __repr__(self):
        id_q  = ureg.Quantity(self.id_si, "m")
        dh_q  = ureg.Quantity(self.hydraulic_diameter_si, "m")
        eps_q = ureg.Quantity(self.roughness_si, "m")
        return (
            f"Line_Segment("
            f"id={id_q.to('inch'):.4f~P} [{id_q:.6f~P}], "
            f"D_h={dh_q.to('inch'):.4f~P} [{dh_q:.6f~P}], "
            f"roughness={eps_q.to('ft'):.6f~P} [{eps_q:.8f~P}], "
            f"length={self.total_length_m:.2f} m, "
            f"points={len(self.profile)})"
        )


# ---------------------------------------------------------------------------
# Hydraulics calculation
# ---------------------------------------------------------------------------

def _fanning_friction(Re, eps, d_h):
    """Return the Fanning friction factor and flow regime string.

    For turbulent flow the correlation is from:
        "Fluid Mechanics for Chemical Engineers, Third Edition"
        Noel de Nevers, p. 187.
    Note: Fanning friction factor = (1/4) * Darcy friction factor.

    For laminar flow: f_Fanning = 16 / Re.

    Args:
        Re  : Reynolds number (dimensionless).
        eps : absolute roughness (m).
        d_h : hydraulic diameter (m).

    Returns:
        (f_fanning, regime_string)
    """
    if Re > 2000:
        f = 0.001375 * (
            1.0 + (20000.0 * eps / d_h + 1.0e6 / Re) ** (1.0 / 3.0)
        )
        return f, "turbulent"
    else:
        return 16.0 / Re, "laminar"


def liquid_hydraulics(fluid, segment, flow_rate, grav_constant=None):
    """Calculate incompressible liquid pressure drop along a pipe segment.

    The calculation is performed one profile step at a time so that a
    per-point pressure profile is produced.  For a liquid the friction
    gradient and density are uniform, so the step-by-step approach yields
    the same total result as a single-pass calculation; however, the
    structure is intentionally preserved here to ease future extension to
    compressible (gas/supercritical) flow where both will vary with pressure.

    Sign convention (all dP terms):
      - Positive dP means pressure at this point is HIGHER than at the start.
      - Negative dP means pressure at this point is LOWER than at the start.

    Friction pressure loss always opposes flow (always reduces pressure in
    the flow direction), so dP_friction is always <= 0.

    dP_static at point i (relative to segment inlet):
        dP_static_i = rho * g * (elev_0 - elev_i)             [Pa]

    dP_friction at point i (relative to segment inlet):
        dP_friction_i = -2 * f * (dist_i / D_h) * rho * v^2  [Pa]

    dP_total_i = dP_static_i + dP_friction_i                  [Pa]

    Args:
        fluid         : Incompressible_Fluid instance.
        segment       : Line_Segment instance.
        flow_rate     : pint Quantity -- either a volumetric flow rate
                        (dimensions [length]^3/[time], e.g. m^3/s, bbl/day)
                        or a mass flow rate (dimensions [mass]/[time],
                        e.g. kg/s, lb/min).  Mass flow rate is converted to
                        volumetric using the fluid density.  A plain float or
                        a Quantity with any other dimensions raises ValueError.
        grav_constant : pint Quantity (acceleration). Defaults to 9.8066 m/s^2.

    Returns:
        dict with keys:
            'flow_regime'      : str
            'velocity_si'      : float (m/s)
            'Re'               : float
            'fanning_friction' : float
            'dP_friction_Pa'   : float (total friction dP, Pa, <= 0)
            'dP_elevation_Pa'  : float (total elevation dP, Pa)
            'dP_total_Pa'      : float (Pa)
            'profile_results'  : list of dicts, one per profile point:
                                 {distance_m, elevation_m,
                                  dP_static_Pa, dP_friction_Pa, dP_total_Pa}
    """
    if grav_constant is None:
        grav_constant = ureg.Quantity(9.8066, "m/s^2")

    g   = grav_constant.to("m/s^2").magnitude
    rho = fluid.density_si

    # --- Resolve flow rate to volumetric (m^3/s) ---
    # Pint dimensionality keys: [length], [mass], [time].
    if not hasattr(flow_rate, "dimensionality"):
        raise ValueError(
            "flow_rate must be a pint Quantity with dimensions of "
            "[length]^3/[time] (volumetric) or [mass]/[time] (mass flow)."
        )
    dim = flow_rate.dimensionality
    if dim == {"[length]": 3, "[time]": -1}:
        # Volumetric flow rate -- convert directly.
        Q = flow_rate.to("m^3/s").magnitude
    elif dim == {"[mass]": 1, "[time]": -1}:
        # Mass flow rate -- convert to volumetric using fluid density.
        # Q [m^3/s] = m_dot [kg/s] / rho [kg/m^3]
        Q = flow_rate.to("kg/s").magnitude / rho
    else:
        raise ValueError(
            f"flow_rate has unrecognized dimensions {dict(dim)}. "
            "Expected [length]^3/[time] (volumetric) or [mass]/[time] "
            "(mass flow rate)."
        )
    mu  = fluid.viscosity_si
    eps = segment.roughness_si
    A   = segment.flow_area_si         # actual cross-section area for velocity
    D_h = segment.hydraulic_diameter_si  # hydraulic diameter for Re and friction

    # --- Velocity and Reynolds number ---
    velocity = Q / A
    Re       = rho * velocity * D_h / mu

    # --- Friction factor (uniform along segment for liquid) ---
    f_fanning, flow_regime = _fanning_friction(Re, eps, D_h)

    # Friction pressure gradient (Pa/m), positive magnitude -- loss per unit length.
    # dP_friction = 2 * f * (L / D_h) * rho * v^2
    # => gradient  = 2 * f * rho * v^2 / D_h
    friction_gradient = 2.0 * f_fanning * rho * velocity ** 2 / D_h

    # --- Step through profile points ---
    profile    = segment.profile
    elev_ref   = profile[0][1]     # elevation at segment inlet (m)
    dist_prev  = profile[0][0]     # should always be 0.0 after normalization

    profile_results = []

    for dist_m, elev_m in profile:
        dP_static   =  rho * g * (elev_ref - elev_m)
        dP_friction = -friction_gradient * dist_m    # negative = pressure loss
        dP_total    =  dP_static + dP_friction
        profile_results.append({
            "distance_m":   dist_m,
            "elevation_m":  elev_m,
            "dP_static_Pa":  dP_static,
            "dP_friction_Pa": dP_friction,
            "dP_total_Pa":   dP_total,
        })

    # Totals at the last profile point.
    last  = profile_results[-1]
    dP_friction_total   = last["dP_friction_Pa"]
    dP_elevation_total  = last["dP_static_Pa"]
    dP_total            = last["dP_total_Pa"]

    return {
        "flow_regime":      flow_regime,
        "velocity_si":      velocity,
        "Re":               Re,
        "fanning_friction": f_fanning,
        "dP_friction_Pa":   dP_friction_total,
        "dP_elevation_Pa":  dP_elevation_total,
        "dP_total_Pa":      dP_total,
        "profile_results":  profile_results,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(results, fluid, segment, flow_rate):
    """Print a formatted summary of liquid_hydraulics() results.

    Args:
        results   : dict returned by liquid_hydraulics().
        fluid     : Incompressible_Fluid instance (for display).
        segment   : Line_Segment instance (for display).
        flow_rate : pint Quantity passed to liquid_hydraulics() (for display).
    """
    rho_q   = ureg.Quantity(fluid.density_si,          "kg/m^3")
    A_q     = ureg.Quantity(segment.flow_area_si,       "m^2")
    vol_q   = ureg.Quantity(segment.volume_m3,          "m^3")
    v_q     = ureg.Quantity(results["velocity_si"],     "m/s")
    dP_f_q  = ureg.Quantity(results["dP_friction_Pa"],  "Pa")
    dP_e_q  = ureg.Quantity(results["dP_elevation_Pa"], "Pa")
    dP_t_q  = ureg.Quantity(results["dP_total_Pa"],     "Pa")

    print("=== Liquid Hydraulics Results ===")
    print(f"  Flow regime          : {results['flow_regime']}")
    print(f"  Flow area            : {A_q.to('in^2'):.4f~P}  ({A_q:.6f~P})")
    print(f"  Velocity             : {v_q.to('ft/s'):.4f~P}  ({v_q:.4f~P})")
    print(f"  Line volume          : {vol_q.to('oil_bbl'):.4f~P}  ({vol_q:.4f~P})")
    print(f"  Fluid density        : {rho_q.to('lb/ft^3'):.4f~P}  ({rho_q:.4f~P})")
    print(f"  Reynolds number      : {results['Re']:.1f}")
    print(f"  Fanning friction     : {results['fanning_friction']:.6f}  ({results['flow_regime']})")
    print(f"  dP friction          : {dP_f_q.to('psi'):.4f~P}  ({dP_f_q:.2f~P})")
    print(f"  dP elevation         : {dP_e_q.to('psi'):.4f~P}  ({dP_e_q:.2f~P})")
    print(f"  dP total             : {dP_t_q.to('psi'):.4f~P}  ({dP_t_q:.2f~P})")


def export_pressure_profile(results, output_path):
    """Write the per-point pressure profile from liquid_hydraulics() to a CSV.

    Columns written:
        distance_from_start_m    : along-pipe distance from segment inlet.
        elevation_m              : elevation at this point.
        dP_static_from_start_Pa  : elevation-head pressure differential vs inlet.
        dP_friction_from_start_Pa: frictional pressure differential vs inlet
                                   (always <= 0).
        dP_total_from_start_Pa   : sum of static and friction differentials.

    Sign convention:
        Positive dP means pressure at this point is higher than at the inlet.
        Negative dP means pressure at this point is lower than at the inlet.

    Args:
        results    : dict returned by liquid_hydraulics().
        output_path: file path for the output CSV.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance_from_start_m",
            "elevation_m",
            "dP_static_from_start_Pa",
            "dP_friction_from_start_Pa",
            "dP_total_from_start_Pa",
        ])
        for pt in results["profile_results"]:
            writer.writerow([
                f"{pt['distance_m']:.6f}",
                f"{pt['elevation_m']:.6f}",
                f"{pt['dP_static_Pa']:.4f}",
                f"{pt['dP_friction_Pa']:.4f}",
                f"{pt['dP_total_Pa']:.4f}",
            ])
    print(f"  Pressure profile exported to: {output_path}")


# ---------------------------------------------------------------------------
# Example / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Fluid definition ---
    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    # --- Pipe segment loaded from CSV ---
    segment = Line_Segment.from_csv(
        csv_path="testprofile1.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
    )

    # --- Flow rate ---
    flow_rate = ureg.Quantity(2000, "oil_bbl/day")

    # --- Run calculation ---
    results = liquid_hydraulics(fluid, segment, flow_rate)

    # --- Display and export ---
    print_results(results, fluid, segment, flow_rate)

    output_csv = os.path.splitext("testprofile1.csv")[0] + "_pressure_profile.csv"
    export_pressure_profile(results, output_csv)
