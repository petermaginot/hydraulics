import csv
import math
import os
import warnings
from pint import UnitRegistry
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import fluids.fittings


ureg = UnitRegistry()

# ---------------------------------------------------------------------------
# Custom standard-volume unit definitions
# ---------------------------------------------------------------------------
# Standard conditions:
#   SCM  : 0 deg C (273.15 K), 101325 Pa  (SI/metric standard)
#   SCF  : 60 deg F (288.706 K), 14.696 psia (US upstream gas standard)
#
# Molar volume at standard conditions via ideal gas law:
#   V_std = R * T_std / P_std   [m^3/mol]

_R = 8.31446261815324  # J/(mol*K) -- exact CODATA 2018 value

_T_scm = 273.15        # K
_P_scm = 101325.0      # Pa
_V_scm = _R * _T_scm / _P_scm   # m^3/mol (~0.022414)

_T_scf = (60.0 - 32.0) * 5.0 / 9.0 + 273.15   # K (~288.706)
_P_scf = 14.696 * 6894.757293168               # Pa
_V_scf = ureg.Quantity(_R * _T_scf / _P_scf, 'm^3').to("ft^3").magnitude

ureg.define(f'scm  = {1.0/_V_scm} * mol ')
ureg.define(f'scf  = {1.0/_V_scf} * mol ')
ureg.define(f'mscf = {1e3/_V_scf}  *   mol ')
ureg.define(f'mmscf = {1e6/_V_scf}*  mol ')


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
# Internal helpers shared by Line_Segment and from_csv
# ---------------------------------------------------------------------------

def _to_si(val, unit):
    """Convert a pint Quantity or plain float to SI magnitude.  Returns None
    if val is None."""
    if val is None:
        return None
    return val.to(unit).magnitude if hasattr(val, "to") else float(val)


def _resolve_id(id_si, od_si, wt_si, tol, stacklevel=3):
    """Return inner diameter in meters from any combination of id/od/wt.

    At least one of id_si or (od_si and wt_si) must be non-None.

    Args:
        id_si      : float or None, inner diameter [m].
        od_si      : float or None, outer diameter [m].
        wt_si      : float or None, wall thickness [m].
        tol        : float, fractional tolerance for consistency check.
        stacklevel : int, passed to warnings.warn for correct call-site
                     attribution.

    Returns:
        float, inner diameter [m].

    Raises:
        ValueError : if insufficient geometry is supplied.
    """
    if id_si is None and od_si is None:
        raise ValueError(
            "Geometry error: supply id_val, or both od_val and wt_val."
        )
    if od_si is not None and wt_si is None:
        raise ValueError(
            "Geometry error: od_val supplied without wt_val. "
            "Provide both to derive inner diameter."
        )
    if od_si is not None and wt_si is not None:
        id_derived = od_si - 2.0 * wt_si
        if id_si is not None:
            if abs(id_si - id_derived) > tol * max(id_si, id_derived):
                warnings.warn(
                    f"id/od/wt are inconsistent "
                    f"(id={id_si:.6f} m, od-2*wt={id_derived:.6f} m). "
                    f"Proceeding with id_val.",
                    UserWarning,
                    stacklevel=stacklevel,
                )
            return id_si
        return id_derived
    return id_si


def _flow_props_from_id(id_m):
    """Return (D_h, flow_area) for a circular pipe given inner diameter."""
    return id_m, math.pi * id_m ** 2 / 4.0


# ---------------------------------------------------------------------------
# Line_Segment class
# ---------------------------------------------------------------------------

class Line_Segment:
    """Geometric and elevation-profile properties of a single pipe segment.

    The segment is internally represented as a list of 4-tuples:
        (distance_m, elevation_m, D_h_m, flow_area_m2)

    where properties at each point apply from that point forward to the next
    (forward Euler / inlet-point convention).

    Geometry can be specified in two ways:

    1. Uniform geometry (scalar args):
       Supply id_val, or od_val+wt_val, or D_h+flow_area.  These values are
       broadcast to every profile point.  profile must then be a list of
       2-tuples (distance, elevation) or 4-tuples (distance, elevation, D_h,
       flow_area).  If 4-tuples are supplied together with scalar geometry
       args, a warning is issued and the 4-tuple geometry takes precedence.

    2. Per-point geometry (4-tuple profile):
       Supply profile as a list of 4-tuples (distance, elevation, D_h,
       flow_area).  Scalar geometry args should be omitted; a warning is
       issued if they are supplied.

    For non-circular cross-sections, supply D_h and flow_area directly and
    set noncircular=True.  When noncircular is True, area-change corrections
    at inter-slice boundaries are skipped in dP().

    Args:
        roughness         : pint Quantity or float (m if float).  Absolute
                            pipe-wall roughness; uniform for the whole segment.
        id_val            : pint Quantity or float (m if float).  Inner
                            diameter.  Optional if od_val+wt_val or D_h+
                            flow_area are supplied.
        od_val            : pint Quantity or float (m if float).  Outer
                            diameter.  Optional.
        wt_val            : pint Quantity or float (m if float).  Wall
                            thickness.  Optional.
        profile           : list of 2-tuples (dist, elev) or 4-tuples
                            (dist, elev, D_h, flow_area).  pint Quantities
                            or plain floats (m if float) are accepted.
                            Takes precedence over length/elevation_change.
        length            : pint Quantity or float (m if float).  Segment
                            length.  Used only when profile is None.
        elevation_change  : pint Quantity or float (m if float).  Net
                            elevation change.  Used only when profile is None.
        flow_area         : pint Quantity or float (m^2 if float).  Flow
                            cross-section area.  Supply with D_h for non-
                            circular geometries or to override the circular
                            area derived from id_val.
        hydraulic_diameter: pint Quantity or float (m if float).  Hydraulic
                            diameter used for Re and friction.  For a circular
                            pipe equals the inner diameter.  For non-circular
                            sections: D_h = 4*A / P_wetted.
        noncircular       : bool.  If True, area-change Bernoulli and loss
                            corrections are skipped at inter-slice boundaries
                            in dP().  Default False.
        id_tolerance      : float, fractional tolerance for id/od/wt
                            consistency check.  Default 0.001 (0.1%).
        k_wall            : pint Quantity or float (W/(m*K) if float).
                            Thermal conductivity of the pipe wall.  Stored for
                            future heat-transfer support; not used in any
                            current calculation.  Optional, default None.
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
        noncircular=False,
        id_tolerance=0.001,
        k_wall=None,
    ):
        # --- Roughness (uniform, SI) ---
        self.roughness_si = _to_si(roughness, "m")
        if self.roughness_si is None:
            raise ValueError("Line_Segment: roughness must be supplied.")

        self.noncircular = bool(noncircular)

        # --- Wall thermal conductivity (stored, not yet used) ---
        self.k_wall_si = _to_si(k_wall, "W/(m*K)") if k_wall is not None else None

        # --- Build the elevation/geometry profile ---
        if profile is not None:
            raw = self._normalize_profile(
                profile,
                id_val=id_val,
                od_val=od_val,
                wt_val=wt_val,
                flow_area=flow_area,
                hydraulic_diameter=hydraulic_diameter,
                id_tolerance=id_tolerance,
            )
        elif length is not None:
            # Build a two-point profile from length and elevation_change.
            L_si  = _to_si(length, "m")
            dz_si = _to_si(elevation_change, "m") if elevation_change is not None else 0.0
            D_h_si, A_si = self._scalar_geom(
                id_val, od_val, wt_val, flow_area, hydraulic_diameter, id_tolerance
            )
            raw = [
                (0.0,  0.0,  D_h_si, A_si),
                (L_si, dz_si, D_h_si, A_si),
            ]
        else:
            raise ValueError(
                "Line_Segment requires either a profile list or a length."
            )

        self.profile = raw   # list of (dist_m, elev_m, D_h_m, area_m2)

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_geom(id_val, od_val, wt_val, flow_area, hydraulic_diameter,
                     id_tolerance):
        """Resolve scalar geometry arguments to (D_h_si, area_si).

        Either id_val / od_val+wt_val or D_h+flow_area must be supplied.
        """
        id_si = _to_si(id_val, "m")
        od_si = _to_si(od_val, "m")
        wt_si = _to_si(wt_val, "m")
        A_si  = _to_si(flow_area, "m^2")
        Dh_si = _to_si(hydraulic_diameter, "m")

        if Dh_si is not None and A_si is not None:
            # Non-circular override supplied directly.
            return Dh_si, A_si

        if id_si is not None or (od_si is not None and wt_si is not None):
            id_m = _resolve_id(id_si, od_si, wt_si, id_tolerance, stacklevel=4)
            D_h, area = _flow_props_from_id(id_m)
            if A_si is not None:
                area = A_si
            if Dh_si is not None:
                D_h = Dh_si
            return D_h, area

        raise ValueError(
            "Line_Segment: supply id_val, od_val+wt_val, or "
            "hydraulic_diameter+flow_area."
        )

    @staticmethod
    def _normalize_profile(
        raw_profile,
        id_val=None,
        od_val=None,
        wt_val=None,
        flow_area=None,
        hydraulic_diameter=None,
        id_tolerance=0.001,
    ):
        """Convert raw_profile to a list of (dist_m, elev_m, D_h_m, area_m2).

        Accepts:
          - 2-tuple rows: (dist, elev).  Scalar geometry args are required.
          - 4-tuple rows: (dist, elev, D_h, flow_area).  Scalar geometry args
            are ignored (with a warning if supplied).

        pint Quantities and plain floats (assumed SI) are both accepted.
        The list is sorted by distance; a zero-distance row is prepended if
        the first entry is not already at distance 0.
        """
        if not raw_profile:
            raise ValueError("profile list must not be empty.")

        # Detect tuple width from the first row.
        first = raw_profile[0]
        n_cols = len(first)

        if n_cols == 4:
            # Per-point geometry embedded in the profile.
            scalar_geom_supplied = any(
                v is not None
                for v in (id_val, od_val, wt_val, flow_area, hydraulic_diameter)
            )
            if scalar_geom_supplied:
                warnings.warn(
                    "Line_Segment: scalar geometry args (id_val, od_val, etc.) "
                    "are ignored when a 4-tuple profile is supplied; geometry "
                    "is taken from the profile rows.",
                    UserWarning,
                    stacklevel=4,
                )
            converted = []
            for row in raw_profile:
                d_m   = _to_si(row[0], "m") if hasattr(row[0], "to") else float(row[0])
                e_m   = _to_si(row[1], "m") if hasattr(row[1], "to") else float(row[1])
                Dh_m  = _to_si(row[2], "m") if hasattr(row[2], "to") else float(row[2])
                A_m2  = _to_si(row[3], "m^2") if hasattr(row[3], "to") else float(row[3])
                converted.append((d_m, e_m, Dh_m, A_m2))

        elif n_cols == 2:
            # Uniform geometry broadcast from scalar args.
            D_h_si, A_si = Line_Segment._scalar_geom(
                id_val, od_val, wt_val, flow_area, hydraulic_diameter, id_tolerance
            )
            converted = []
            for row in raw_profile:
                d_m = _to_si(row[0], "m") if hasattr(row[0], "to") else float(row[0])
                e_m = _to_si(row[1], "m") if hasattr(row[1], "to") else float(row[1])
                converted.append((d_m, e_m, D_h_si, A_si))

        else:
            raise ValueError(
                f"Line_Segment: profile rows must be 2-tuples (dist, elev) or "
                f"4-tuples (dist, elev, D_h, flow_area); got {n_cols}-tuple."
            )

        converted.sort(key=lambda r: r[0])

        if converted[0][0] != 0.0:
            # Prepend a synthetic zero-distance point using the geometry of the
            # first real point.
            first_row = converted[0]
            converted = [(0.0, first_row[1], first_row[2], first_row[3])] + converted

        return converted

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path,
        roughness,
        noncircular=False,
        id_tolerance=0.001,
        k_wall=None,
    ):
        """Construct a Line_Segment by loading a profile from a CSV file.

        The first row of the CSV must be a header row whose column names
        determine the geometry mode.  The following column names are
        recognized (case-insensitive):

            Required columns (always):
                layer    - ignored (row identifier or survey layer label)
                distance - along-pipe distance from origin [m]
                elevation - elevation [m]

            Geometry mode A -- circular pipe via diameters:
                ID  - inner diameter [m]
                OD  - outer diameter [m]  (optional if ID is present)
                WT  - wall thickness [m]  (optional if ID is present)
                Inner diameter is resolved per the same id/od/wt rules as
                the constructor: ID takes precedence; OD-2*WT is used as a
                fallback; a consistency warning is issued if both are present
                and disagree.

            Geometry mode B -- direct hydraulic properties:
                D_h       - hydraulic diameter [m]
                flow_area - flow cross-section area [m^2]

        All geometry values in the CSV are assumed to be in SI units (meters,
        square meters).

        Args:
            csv_path     : str, path to the CSV file.
            roughness    : pint Quantity or float (m if float).  Uniform
                           pipe-wall roughness for the whole segment.
            noncircular  : bool.  Passed through to the constructor.
                           Default False.
            id_tolerance : float.  Fractional tolerance for id/od/wt
                           consistency check.  Default 0.001.
            k_wall       : pint Quantity or float (W/(m*K)) or None.
                           Pipe-wall thermal conductivity, stored for future
                           heat-transfer support.  Default None.

        Returns:
            Line_Segment instance.

        Raises:
            ValueError : if the CSV is empty, the header is unrecognized, or
                         geometry cannot be resolved for any row.
        """
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            # Normalize header names to lowercase for case-insensitive matching.
            if reader.fieldnames is None:
                raise ValueError(f"CSV '{csv_path}' appears to be empty.")
            headers = [h.strip().lower() for h in reader.fieldnames]

            # Detect geometry mode.
            has_id  = "id" in headers
            has_od  = "od" in headers
            has_wt  = "wt" in headers
            has_dh  = "d_h" in headers
            has_fa  = "flow_area" in headers

            if has_dh and has_fa:
                mode = "DH_FA"
            elif has_id or (has_od and has_wt):
                mode = "ID_OD_WT"
            else:
                raise ValueError(
                    f"CSV '{csv_path}': cannot determine geometry mode from "
                    f"headers {reader.fieldnames}.  Expected columns "
                    f"'ID'/'OD'/'WT' or 'D_h'/'flow_area'."
                )

            for raw_row in reader:
                # Normalize keys to lowercase for lookup.
                row = {k.strip().lower(): v.strip() for k, v in raw_row.items()
                       if k is not None}

                dist_m = float(row["distance"])
                elev_m = float(row["elevation"])

                if mode == "DH_FA":
                    D_h_m = float(row["d_h"])
                    A_m2  = float(row["flow_area"])
                else:
                    # ID_OD_WT mode -- resolve per _resolve_id rules.
                    id_m  = float(row["id"])  if has_id else None
                    od_m  = float(row["od"])  if has_od else None
                    wt_m  = float(row["wt"])  if has_wt else None
                    # Treat zero as absent (some exports write 0 for missing).
                    if id_m == 0.0:
                        id_m = None
                    if od_m == 0.0:
                        od_m = None
                    if wt_m == 0.0:
                        wt_m = None
                    id_resolved = _resolve_id(
                        id_m, od_m, wt_m, id_tolerance, stacklevel=2
                    )
                    D_h_m, A_m2 = _flow_props_from_id(id_resolved)

                rows.append((dist_m, elev_m, D_h_m, A_m2))

        if not rows:
            raise ValueError(
                f"Elevation profile CSV '{csv_path}' contains no data rows."
            )

        total_dist_m = max(r[0] for r in rows)
        net_elev_m   = rows[-1][1] - rows[0][1]
        print(
            f"  Profile loaded from '{csv_path}': {len(rows)} points, "
            f"total distance = {total_dist_m:.2f} m, "
            f"net elevation change = {net_elev_m:.4f} m  [mode={mode}]"
        )

        return cls(
            roughness=roughness,
            profile=rows,
            noncircular=noncircular,
            id_tolerance=id_tolerance,
            k_wall=k_wall,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_length_m(self):
        """Total along-pipe distance of this segment [m]."""
        return self.profile[-1][0]

    @property
    def net_elevation_change_m(self):
        """Net elevation change (last minus first point) [m].
        Positive = uphill."""
        return self.profile[-1][1] - self.profile[0][1]

    @property
    def volume_m3(self):
        """Approximate internal volume [m^3], computed as the sum of per-slice
        volumes using the inlet-point area for each slice."""
        vol = 0.0
        for i in range(len(self.profile) - 1):
            dist_in,  _,  _, area_in  = self.profile[i]
            dist_out, _, _,  _        = self.profile[i + 1]
            vol += area_in * (dist_out - dist_in)
        return vol

    def __repr__(self):
        first = self.profile[0]
        last  = self.profile[-1]
        id_q  = ureg.Quantity(first[2], "m")   # D_h at inlet
        eps_q = ureg.Quantity(self.roughness_si, "m")
        return (
            f"Line_Segment("
            f"D_h_inlet={id_q.to('inch'):.4f~P} [{id_q:.6f~P}], "
            f"roughness={eps_q.to('ft'):.6f~P} [{eps_q:.8f~P}], "
            f"length={self.total_length_m:.2f} m, "
            f"points={len(self.profile)}, "
            f"noncircular={self.noncircular})"
        )

    # ------------------------------------------------------------------
    # Pressure drop calculation
    # ------------------------------------------------------------------

    def dP(self, fluid, flow_rate):
        """Calculate the total static pressure change along the segment.

        Steps through consecutive profile point pairs.  For each slice the
        pressure change has up to three contributions:

        1. Friction loss (Darcy-Weisbach):
               dP_friction = -(f_D / 2) * (rho * v^2 / D_h) * dL   [Pa]
           Always negative (opposes flow).

        2. Elevation (hydrostatic):
               dP_elevation = -rho * g * dz                          [Pa]
           Positive if flowing downhill (dz < 0), negative if uphill.

        3. Area-change at the slice boundary (applied at the outlet of each
           slice when the next slice has a different flow area, and only when
           noncircular=False):

           a. Bernoulli velocity-head exchange (recoverable, affects static P):
                  dP_Bernoulli = (1/2) * rho * (v_in^2 - v_out^2)   [Pa]
              Positive for an expansion (v_out < v_in, static P rises),
              negative for a contraction (v_out > v_in, static P falls).

           b. Permanent loss (K-factor, always negative):
                  dP_loss = -K * (1/2) * rho * v_ref^2              [Pa]
              K and v_ref are determined by fluids.fittings:
                - Contraction (Di_in > Di_out): contraction_sharp, v_ref = v_out
                  (K expressed w.r.t. downstream velocity; converted to
                  upstream here for consistent sign convention).
                - Expansion   (Di_in < Di_out): diffuser_sharp, v_ref = v_in.

        The per-slice velocity is computed from the inlet-point flow area,
        consistent with the forward-Euler convention used throughout.

        For noncircular=True, contribution 3 is omitted entirely.

        Args:
            fluid     : Incompressible_Fluid instance.
            flow_rate : pint Quantity -- volumetric ([length]^3/[time]) or
                        mass ([mass]/[time]).  Mass flow is converted to
                        volumetric using fluid.density_si.

        Returns:
            float, total static pressure change [Pa].
            Negative means outlet pressure is lower than inlet pressure.

        Raises:
            ValueError : if flow_rate has unrecognized dimensions or the
                         profile contains only one point.
        """
        if len(self.profile) < 2:
            raise ValueError(
                "Line_Segment.dP: profile must have at least two points."
            )

        grav_constant = 9.8066   # m/s^2

        rho = fluid.density_si
        mu  = fluid.viscosity_si
        eps = self.roughness_si

        # --- Resolve flow rate to volumetric (m^3/s) ---
        if not hasattr(flow_rate, "dimensionality"):
            raise ValueError(
                "flow_rate must be a pint Quantity with dimensions of "
                "[length]^3/[time] (volumetric) or [mass]/[time] (mass flow)."
            )
        dim = flow_rate.dimensionality
        if dim == {"[length]": 3, "[time]": -1}:
            Q = flow_rate.to("m^3/s").magnitude
        elif dim == {"[mass]": 1, "[time]": -1}:
            Q = flow_rate.to("kg/s").magnitude / rho
        else:
            raise ValueError(
                f"flow_rate has unrecognized dimensions {dict(dim)}. "
                "Expected [length]^3/[time] (volumetric) or [mass]/[time] "
                "(mass flow rate)."
            )

        dP_total = 0.0
        n = len(self.profile)

        for i in range(n - 1):
            dist_in,  elev_in,  D_h_in,  area_in  = self.profile[i]
            dist_out, elev_out, D_h_out, area_out = self.profile[i + 1]

            dL = dist_out - dist_in   # m, slice length
            dz = elev_out - elev_in   # m, positive = uphill

            if dL <= 0.0:
                raise ValueError(
                    f"Line_Segment.dP: non-positive slice length dL={dL:.6g} m "
                    f"between profile points {i} and {i+1}.  Ensure profile "
                    f"is sorted by distance with no duplicate distances."
                )

            # Inlet velocity for this slice.
            v_in = Q / area_in

            Re       = fluids_Reynolds(V=v_in, D=D_h_in, rho=rho, mu=mu)
            f_darcy  = fluids_friction_factor(Re=Re, eD=eps / D_h_in)

            # 1. Friction loss.
            dP_friction = -(f_darcy / 2.0) * rho * v_in ** 2 / D_h_in * dL

            # 2. Elevation (hydrostatic).
            dP_elevation = -rho * grav_constant * dz

            dP_total += dP_friction + dP_elevation

            # 3. Area-change correction at the boundary to the next slice.
            # Applied using the outlet-point geometry (area_out) vs the current
            # inlet-point geometry (area_in).
            _AREA_TOL = 1e-9   # m^2, absolute tolerance for area equality
            if (not self.noncircular) and (abs(area_out - area_in) > _AREA_TOL):
                v_out = Q / area_out

                # 3a. Bernoulli velocity-head exchange (static pressure change
                # due to velocity change -- recoverable for a perfect transition).
                dP_bernoulli = 0.5 * rho * (v_in ** 2 - v_out ** 2)

                # 3b. Permanent loss via K-factor.
                # Derive equivalent inner diameters from flow areas (circular
                # equivalent, used only for K-factor lookup).
                Di_in  = math.sqrt(4.0 * area_in  / math.pi)
                Di_out = math.sqrt(4.0 * area_out / math.pi)

                if Di_in > Di_out:
                    # Contraction: K is w.r.t. downstream (higher) velocity.
                    K_ds = fluids.fittings.contraction_sharp(Di1=Di_in, Di2=Di_out)
                    # Convert K to upstream reference velocity so the loss is
                    # expressed consistently as K_us * (1/2) * rho * v_in^2.
                    K_us = K_ds * (Di_out / Di_in) ** 4   # (A_out/A_in)^2 ratio
                    dP_loss = -K_us * 0.5 * rho * v_in ** 2
                else:
                    # Expansion: K is w.r.t. upstream velocity.
                    K_us = fluids.fittings.diffuser_sharp(Di1=Di_in, Di2=Di_out)
                    dP_loss = -K_us * 0.5 * rho * v_in ** 2

                dP_total += dP_bernoulli + dP_loss

        return dP_total


# ---------------------------------------------------------------------------
# Bend class
# ---------------------------------------------------------------------------

class Bend:
    """A pipe bend fitting.

    Pressure drop is calculated using the fluids.fittings.bend_rounded()
    K-factor correlation, which accounts for the bend angle, bend radius, and
    Reynolds number.

    Args:
        Di        : pint Quantity or float (m if float).  Pipe inner diameter.
        ang_deg   : float.  Bend angle [degrees].  0 = straight; 90 = elbow.
        bend_dias : float.  Bend centerline radius expressed as a multiple of
                    the pipe inner diameter (e.g. 1.5 for a standard long-
                    radius elbow: R_bend = 1.5 * Di).
    """

    def __init__(self, Di, ang_deg, bend_dias):
        self.Di_si     = _to_si(Di, "m")
        self.ang_deg   = float(ang_deg)
        self.bend_dias = float(bend_dias)

        if self.Di_si is None or self.Di_si <= 0.0:
            raise ValueError("Bend: Di must be a positive length.")
        if self.ang_deg < 0.0:
            raise ValueError("Bend: ang_deg must be >= 0.")
        if self.bend_dias <= 0.0:
            raise ValueError("Bend: bend_dias must be positive.")

    def dP(self, fluid, flow_rate):
        """Permanent pressure loss through the bend [Pa].

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

    def __repr__(self):
        Di_q = ureg.Quantity(self.Di_si, "m")
        return (
            f"Bend(Di={Di_q.to('inch'):.4f~P}, "
            f"angle={self.ang_deg:.1f} deg, "
            f"bend_diameters={self.bend_dias:.2f})"
        )


# ---------------------------------------------------------------------------
# Contraction_Expansion class
# ---------------------------------------------------------------------------

class Contraction_Expansion:
    """An abrupt contraction or expansion between two pipe diameters.

    Computes the total static pressure change, which has two components:

    1. Bernoulli velocity-head exchange (recoverable, affects static P):
           dP_Bernoulli = (1/2) * rho * (v_US^2 - v_DS^2)   [Pa]

    2. Permanent K-factor loss (always <= 0):
           Contraction: uses fluids.fittings.contraction_sharp()
           Expansion  : uses fluids.fittings.diffuser_sharp()

    If Di_US == Di_DS (no area change) both contributions are zero.

    Args:
        Di_US : pint Quantity or float (m if float).  Upstream inner diameter.
        Di_DS : pint Quantity or float (m if float).  Downstream inner diameter.
    """

    def __init__(self, Di_US, Di_DS):
        self.Di_US_si = _to_si(Di_US, "m")
        self.Di_DS_si = _to_si(Di_DS, "m")

        for name, val in (("Di_US", self.Di_US_si), ("Di_DS", self.Di_DS_si)):
            if val is None or val <= 0.0:
                raise ValueError(
                    f"Contraction_Expansion: {name} must be a positive length."
                )

    def dP(self, fluid, flow_rate):
        """Total static pressure change through the contraction/expansion [Pa].

        Includes both the Bernoulli velocity-head exchange and the permanent
        K-factor loss.

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

        Q   = _resolve_flow_rate(flow_rate, rho)
        v_US = Q / A_US
        v_DS = Q / A_DS

        # Bernoulli: static pressure change due to velocity change.
        dP_bernoulli = 0.5 * rho * (v_US ** 2 - v_DS ** 2)

        if abs(Di_US - Di_DS) < 1e-12:
            dP_loss = 0.0
        elif Di_US > Di_DS:
            # Contraction: K w.r.t. downstream velocity; convert to upstream.
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K_us = K_ds * (A_DS / A_US) ** 2
            dP_loss = -K_us * 0.5 * rho * v_US ** 2
        else:
            # Expansion: K w.r.t. upstream velocity.
            K_us = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)
            dP_loss = -K_us * 0.5 * rho * v_US ** 2

        return dP_bernoulli + dP_loss

    def __repr__(self):
        US_q = ureg.Quantity(self.Di_US_si, "m")
        DS_q = ureg.Quantity(self.Di_DS_si, "m")
        return (
            f"Contraction_Expansion("
            f"Di_US={US_q.to('inch'):.4f~P}, "
            f"Di_DS={DS_q.to('inch'):.4f~P})"
        )


# ---------------------------------------------------------------------------
# Global pressure drop functions
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


def dP_friction(fluid, flow_rate, flow_area, eps, D_h, dL):
    """Permanent friction pressure loss for a uniform pipe length.

    Darcy-Weisbach equation:
        dP = -(f_D / 2) * (rho * v^2 / D_h) * dL   [Pa]

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
    rho = fluid.density_si
    mu  = fluid.viscosity_si
    Q   = _resolve_flow_rate(flow_rate, rho)
    v   = Q / flow_area
    Re  = fluids_Reynolds(V=v, D=D_h, rho=rho, mu=mu)
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
    """Total static pressure change through an abrupt contraction or expansion.

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
    rho = fluid.density_si
    Q   = _resolve_flow_rate(flow_rate, rho)
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

    rho_q  = ureg.Quantity(rho,          "kg/m^3")
    A_q    = ureg.Quantity(A_in,         "m^2")
    vol_q  = ureg.Quantity(segment.volume_m3, "m^3")
    v_q    = ureg.Quantity(v,            "m/s")
    dP_q   = ureg.Quantity(dP_total_Pa,  "Pa")

    print("=== Liquid Hydraulics Results ===")
    print(f"  Flow area (inlet)    : {A_q.to('in^2'):.4f~P}  ({A_q:.6f~P})")
    print(f"  Velocity (inlet)     : {v_q.to('ft/s'):.4f~P}  ({v_q:.4f~P})")
    print(f"  Line volume          : {vol_q.to('oil_bbl'):.4f~P}  ({vol_q:.4f~P})")
    print(f"  Fluid density        : {rho_q.to('lb/ft^3'):.4f~P}  ({rho_q:.4f~P})")
    print(f"  dP total             : {dP_q.to('psi'):.4f~P}  ({dP_q:.2f~P})")


def export_pressure_profile(segment, fluid, flow_rate, output_path):
    """Write a per-point pressure profile to a CSV file.

    Walks the segment profile and accumulates pressure contributions
    slice by slice, writing one row per profile point.  The first row
    is always the inlet (distance 0, dP = 0).

    Columns:
        point                    : integer point index (0 = inlet).
        distance_m               : along-pipe distance from inlet [m].
        elevation_m              : elevation [m].
        D_h_m                    : hydraulic diameter at this point [m].
        flow_area_m2             : flow area at this point [m^2].
        velocity_ms              : mean flow velocity at this point [m/s].
        dP_friction_Pa           : cumulative friction dP from inlet [Pa].
        dP_elevation_Pa          : cumulative elevation dP from inlet [Pa].
        dP_area_change_Pa        : cumulative area-change dP from inlet [Pa].
        dP_total_Pa              : cumulative total dP from inlet [Pa].

    Args:
        segment     : Line_Segment instance.
        fluid       : Incompressible_Fluid instance.
        flow_rate   : pint Quantity, volumetric or mass flow rate.
        output_path : str, path for the output CSV.
    """
    grav_constant = 9.8066

    rho = fluid.density_si
    mu  = fluid.viscosity_si
    eps = segment.roughness_si
    Q   = _resolve_flow_rate(flow_rate, rho)

    cum_fric = 0.0
    cum_elev = 0.0
    cum_area = 0.0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point",
            "distance_m",
            "elevation_m",
            "D_h_m",
            "flow_area_m2",
            "velocity_ms",
            "dP_friction_Pa",
            "dP_elevation_Pa",
            "dP_area_change_Pa",
            "dP_total_Pa",
        ])

        # Inlet row (point 0).
        dist_0, elev_0, D_h_0, area_0 = segment.profile[0]
        v_0 = Q / area_0
        writer.writerow([
            0,
            f"{dist_0:.6f}",
            f"{elev_0:.6f}",
            f"{D_h_0:.6f}",
            f"{area_0:.8f}",
            f"{v_0:.6f}",
            f"{cum_fric:.4f}",
            f"{cum_elev:.4f}",
            f"{cum_area:.4f}",
            f"{cum_fric + cum_elev + cum_area:.4f}",
        ])

        n = len(segment.profile)
        _AREA_TOL = 1e-9

        for i in range(n - 1):
            dist_in,  elev_in,  D_h_in,  area_in  = segment.profile[i]
            dist_out, elev_out, D_h_out, area_out = segment.profile[i + 1]

            dL = dist_out - dist_in
            dz = elev_out - elev_in

            v_in  = Q / area_in
            Re    = fluids_Reynolds(V=v_in, D=D_h_in, rho=rho, mu=mu)
            f_D   = fluids_friction_factor(Re=Re, eD=eps / D_h_in)

            cum_fric += -(f_D / 2.0) * rho * v_in ** 2 / D_h_in * dL
            cum_elev += -rho * grav_constant * dz

            if (not segment.noncircular) and (abs(area_out - area_in) > _AREA_TOL):
                v_out        = Q / area_out
                dP_bernoulli = 0.5 * rho * (v_in ** 2 - v_out ** 2)
                Di_in  = math.sqrt(4.0 * area_in  / math.pi)
                Di_out = math.sqrt(4.0 * area_out / math.pi)
                if Di_in > Di_out:
                    K_ds = fluids.fittings.contraction_sharp(Di1=Di_in, Di2=Di_out)
                    K_us = K_ds * (area_out / area_in) ** 2
                    dP_loss = -K_us * 0.5 * rho * v_in ** 2
                else:
                    K_us  = fluids.fittings.diffuser_sharp(Di1=Di_in, Di2=Di_out)
                    dP_loss = -K_us * 0.5 * rho * v_in ** 2
                cum_area += dP_bernoulli + dP_loss

            v_out_pt = Q / area_out
            writer.writerow([
                i + 1,
                f"{dist_out:.6f}",
                f"{elev_out:.6f}",
                f"{D_h_out:.6f}",
                f"{area_out:.8f}",
                f"{v_out_pt:.6f}",
                f"{cum_fric:.4f}",
                f"{cum_elev:.4f}",
                f"{cum_area:.4f}",
                f"{cum_fric + cum_elev + cum_area:.4f}",
            ])

    print(f"  Pressure profile exported to: {output_path}")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_p2p():
    """Simple two-point pressure drop with entrance and exit losses."""

    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(3.068, "inch")
    length           = ureg.Quantity(2000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft")

    segment = Line_Segment(
        roughness=roughness,
        id_val=id_val,
        length=length,
        elevation_change=elevation_change,
    )

    fluid = Incompressible_Fluid(
        density=ureg.Quantity(1000.0, "kg/m^3"),
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    flow_rate = ureg.Quantity(60, "oil_bbl/day")

    dP_line    = segment.dP(fluid, flow_rate)

    # Entrance and exit losses computed manually (K = 0.5 and 1.0 respectively,
    # both referenced to the pipe velocity head).
    Q   = _resolve_flow_rate(flow_rate, fluid.density_si)
    A   = segment.profile[0][3]
    v   = Q / A
    rho = fluid.density_si

    dP_entrance = -0.5 * 0.5 * rho * v ** 2   # K = 0.5 for sharp-edged entrance
    dP_exit     = -1.0 * 0.5 * rho * v ** 2   # K = 1.0 for abrupt exit

    dP_q = ureg.Quantity(dP_line + dP_entrance + dP_exit, "Pa")

    print(f"  dP entrance : {ureg.Quantity(dP_entrance, 'Pa').to('psi'):.4f~P}")
    print(f"  dP line     : {ureg.Quantity(dP_line,     'Pa').to('psi'):.4f~P}")
    print(f"  dP exit     : {ureg.Quantity(dP_exit,     'Pa').to('psi'):.4f~P}")
    print(f"  dP total    : {dP_q.to('psi'):.4f~P}")


def testcont():
    """Segment + bend example demonstrating class-based dP composition."""

    roughness        = ureg.Quantity(0.00015, "ft")
    id_val           = ureg.Quantity(4.026, "inch")
    # length           = ureg.Quantity(2000.0, "ft")
    # elevation_change = ureg.Quantity(25.0, "ft")

    segment = Line_Segment.from_csv(
        csv_path="testprofile_ID_OD_WT.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
    )

    bend1 = Bend(Di=id_val, ang_deg=90.0, bend_dias=1.5)

    fluid = Incompressible_Fluid(
        density=ureg.Quantity(1000.0, "kg/m^3"),
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    flow_rate = ureg.Quantity(20000, "oil_bbl/day")

    dP_seg  = segment.dP(fluid, flow_rate)
    dP_bend = bend1.dP(fluid, flow_rate)
    dP_tot  = dP_seg + dP_bend

    print(f"  {segment}")
    print(f"  {bend1}")
    print(f"  dP segment  : {ureg.Quantity(dP_seg,  'Pa').to('psi'):.4f~P}")
    print(f"  dP bend     : {ureg.Quantity(dP_bend, 'Pa').to('psi'):.4f~P}")
    print(f"  dP total    : {ureg.Quantity(dP_tot,  'Pa').to('psi'):.4f~P}")


def test_csv_profile():
    """Load a variable-geometry profile from a CSV and export a pressure profile."""

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    segment = Line_Segment.from_csv(
        csv_path="testprofile_ID_OD_WT.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
    )

    flow_rate = ureg.Quantity(2000, "oil_bbl/day")

    dP_total = segment.dP(fluid, flow_rate)
    print_results(fluid, segment, flow_rate, dP_total)

    output_csv = "testprofile_ID_OD_WT_pressure_profile.csv"
    export_pressure_profile(segment, fluid, flow_rate, output_csv)


if __name__ == "__main__":
    # test_p2p()
    testcont()
    # test_csv_profile()
