"""component_classes.py

Base (parent) classes for pipeline hydraulic components.

These classes handle geometry storage, input validation, CSV loading, and
convenience properties, but contain no fluid-mechanics calculations.
Physics-specific subclasses (incompressible, compressible) are defined in
separate modules and inherit from the bases here.

Classes
-------
Base_Line_Segment
    A pipe segment defined by an elevation/geometry profile.  Stores roughness,
    the per-point (distance, elevation, D_h, flow_area) profile, and optional
    wall thermal conductivity.  Provides geometry helpers, a CSV loader, and
    read-only convenience properties.

Base_Bend
    A rounded pipe bend fitting.  Stores inner diameter, bend angle, and
    bend-radius-to-diameter ratio.

Base_Contraction_Expansion
    An abrupt contraction or expansion between two pipe diameters.  Stores
    upstream and downstream inner diameters.

Module-level helpers
--------------------
_to_si(val, unit)
    Convert a pint Quantity or plain float to an SI float magnitude.

_resolve_id(id_si, od_si, wt_si, tol, stacklevel)
    Resolve inner diameter from any combination of id / od / wt values.

_flow_props_from_id(id_m)
    Return (D_h, flow_area) for a circular pipe given its inner diameter.
"""

import csv
import math
import warnings
from pint import UnitRegistry

ureg = UnitRegistry()
#need to register standard cubic foot (scf), thousand standard cubic feet (mscf), million standard cubic feet (mmscf), and standard cubic meter (scm) as custom units in the unit registry

# ---------------------------------------------------------------------------
# Custom standard-volume unit definitions
# ---------------------------------------------------------------------------
# Standard conditions:
#   SCM  : 0 deg C (273.15 K), 101325 Pa  (SI/metric standard)
#   SCF  : 60 deg F (288.706 K), 14.696 psia (US upstream gas standard)
#
# Molar volume at standard conditions via ideal gas law:
#   V_std = R * T_std / P_std   [m^3/mol]
#
# The unit is defined as that molar volume, so 1 scm == V_scm m^3/mol of gas,
# allowing pint to convert between standard-volume and molar flow rates.
# ---------------------------------------------------------------------------

_R = 8.31446261815324  # J/(mol*K) -- exact CODATA 2018 value

# SCM: 0 degC, 101325 Pa
_T_scm = 273.15        # K
_P_scm = 101325.0      # Pa
_V_scm = _R * _T_scm / _P_scm   # m^3/mol  (~0.022414)

# SCF: 60 degF, 14.696 psia
_T_scf = (60.0 - 32.0) * 5.0 / 9.0 + 273.15   # K (~288.706)
_P_scf = 14.696 * 6894.757293168               # Pa (14.696 psi is not quite exactly 101325 Pa)
_V_scf = ureg.Quantity(_R * _T_scf / _P_scf, 'm^3')          #1 scf = 1.20 moles, but let the calculation be done in case you need to use some oddball standard conditions
_V_scf = _V_scf.to("ft^3")
_V_scf = _V_scf.magnitude

ureg.define(f'scm  = {1.0/_V_scm} * mol ')
ureg.define(f'scf  = {1.0/_V_scf} * mol ')
ureg.define(f'mscf = {1e3/_V_scf}  *   mol ')
ureg.define(f'mmscf = {1e6/_V_scf}*  mol ')

# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _to_si(val, unit):
    """Convert a pint Quantity or plain float to an SI float magnitude.

    Returns None if val is None.

    Args:
        val  : pint Quantity, plain float, or None.
        unit : str, target SI unit string (e.g. 'm', 'm^2', 'W/(m*K)').

    Returns:
        float or None.
    """
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
        tol        : float, fractional tolerance for the consistency check
                     when all three values are supplied.
        stacklevel : int, passed to warnings.warn for correct call-site
                     attribution.

    Returns:
        float, inner diameter [m].

    Raises:
        ValueError : if insufficient geometry is supplied, or if od is given
                     without wt.
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
    """Return (D_h, flow_area) for a circular pipe given its inner diameter.

    Args:
        id_m : float, inner diameter [m].

    Returns:
        (D_h_m, area_m2) : tuple of floats.
            D_h_m   -- hydraulic diameter [m], equal to id_m for a circle.
            area_m2 -- cross-section flow area [m^2] = pi/4 * id_m^2.
    """
    return id_m, math.pi * id_m ** 2 / 4.0


# ---------------------------------------------------------------------------
# Base_Line_Segment
# ---------------------------------------------------------------------------

class Base_Line_Segment:
    """Geometry and elevation-profile storage for a single pipe segment.

    The segment is internally represented as a list of 4-tuples:

        (distance_m, elevation_m, D_h_m, flow_area_m2)

    where the values at each point apply from that point forward to the next
    (inlet-point / forward-Euler convention).

    Geometry can be specified in two ways:

    1. Uniform geometry (scalar args):
       Supply id_val, or od_val+wt_val, or hydraulic_diameter+flow_area.
       These values are broadcast to every profile point.  profile must then
       be a list of 2-tuples (distance, elevation) or 4-tuples (distance,
       elevation, D_h, flow_area).  If 4-tuples are supplied together with
       scalar geometry args, a warning is issued and the 4-tuple geometry
       takes precedence.

    2. Per-point geometry (4-tuple profile):
       Supply profile as a list of 4-tuples (distance, elevation, D_h,
       flow_area).  Scalar geometry args should be omitted; a warning is
       issued if they are supplied.

    For non-circular cross-sections, supply hydraulic_diameter and flow_area
    directly and set noncircular=True.  Subclasses may use this flag to skip
    fitting-loss correlations that assume a circular bore.

    Args:
        roughness         : pint Quantity or float (m if float).  Absolute
                            pipe-wall roughness; uniform for the whole segment.
        id_val            : pint Quantity or float (m if float).  Inner
                            diameter.  Optional if od_val+wt_val or
                            hydraulic_diameter+flow_area are supplied.
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
                            elevation change (positive = uphill).  Used only
                            when profile is None.
        flow_area         : pint Quantity or float (m^2 if float).  Flow
                            cross-section area.  Supply with hydraulic_diameter
                            for non-circular geometries or to override the
                            circular area derived from id_val.
        hydraulic_diameter: pint Quantity or float (m if float).  Hydraulic
                            diameter used for Re and friction calculations.
                            For a circular pipe equals the inner diameter.
                            For non-circular sections: D_h = 4*A / P_wetted.
        noncircular       : bool.  Set True for non-circular cross-sections.
                            Default False.
        id_tolerance      : float, fractional tolerance for the id/od/wt
                            consistency check.  Default 0.001 (0.1%).
        k_wall            : pint Quantity or float (W/(m*K) if float).
                            Pipe-wall thermal conductivity.  Stored for future
                            heat-transfer support; not used in any current
                            calculation.  Optional, default None.
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
            raise ValueError(
                f"{self.__class__.__name__}: roughness must be supplied."
            )

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
            L_si  = _to_si(length, "m")
            dz_si = _to_si(elevation_change, "m") if elevation_change is not None else 0.0
            D_h_si, A_si = self._scalar_geom(
                id_val, od_val, wt_val, flow_area, hydraulic_diameter, id_tolerance
            )
            raw = [
                (0.0,  0.0,   D_h_si, A_si),
                (L_si, dz_si, D_h_si, A_si),
            ]
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires either a profile list "
                f"or a length."
            )

        self.profile = raw   # list of (dist_m, elev_m, D_h_m, area_m2) floats

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_geom(id_val, od_val, wt_val, flow_area, hydraulic_diameter,
                     id_tolerance):
        """Resolve scalar geometry arguments to (D_h_si, area_si).

        Either id_val / od_val+wt_val or hydraulic_diameter+flow_area must
        be supplied.

        Returns:
            (D_h_m, area_m2) : tuple of floats [m, m^2].
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
            "Supply id_val, od_val+wt_val, or "
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

        Returns:
            list of (dist_m, elev_m, D_h_m, area_m2) float 4-tuples.
        """
        if not raw_profile:
            raise ValueError("profile list must not be empty.")

        first  = raw_profile[0]
        n_cols = len(first)

        if n_cols == 4:
            # Per-point geometry embedded in the profile.
            scalar_geom_supplied = any(
                v is not None
                for v in (id_val, od_val, wt_val, flow_area, hydraulic_diameter)
            )
            if scalar_geom_supplied:
                warnings.warn(
                    "Scalar geometry args (id_val, od_val, etc.) are ignored "
                    "when a 4-tuple profile is supplied; geometry is taken "
                    "from the profile rows.",
                    UserWarning,
                    stacklevel=4,
                )
            converted = []
            for row in raw_profile:
                d_m  = _to_si(row[0], "m")   if hasattr(row[0], "to") else float(row[0])
                e_m  = _to_si(row[1], "m")   if hasattr(row[1], "to") else float(row[1])
                Dh_m = _to_si(row[2], "m")   if hasattr(row[2], "to") else float(row[2])
                A_m2 = _to_si(row[3], "m^2") if hasattr(row[3], "to") else float(row[3])
                converted.append((d_m, e_m, Dh_m, A_m2))

        elif n_cols == 2:
            # Uniform geometry broadcast from scalar args.
            D_h_si, A_si = Base_Line_Segment._scalar_geom(
                id_val, od_val, wt_val, flow_area, hydraulic_diameter, id_tolerance
            )
            converted = []
            for row in raw_profile:
                d_m = _to_si(row[0], "m") if hasattr(row[0], "to") else float(row[0])
                e_m = _to_si(row[1], "m") if hasattr(row[1], "to") else float(row[1])
                converted.append((d_m, e_m, D_h_si, A_si))

        else:
            raise ValueError(
                f"Profile rows must be 2-tuples (dist, elev) or "
                f"4-tuples (dist, elev, D_h, flow_area); got {n_cols}-tuple."
            )

        converted.sort(key=lambda r: r[0])

        if converted[0][0] != 0.0:
            # Prepend a synthetic zero-distance point using the geometry of
            # the first real point.
            first_row = converted[0]
            converted = [(0.0, first_row[1], first_row[2], first_row[3])] + converted

        return converted

    # ------------------------------------------------------------------
    # Class method: CSV loader
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
        """Construct a segment by loading a profile from a CSV file.

        The first row of the CSV must be a header row whose column names
        determine the geometry mode.  The following column names are
        recognized (case-insensitive):

            Required columns (always):
                layer     - ignored (row identifier or survey layer label)
                distance  - along-pipe distance from origin [m]
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
            Instance of the calling class (or subclass).

        Raises:
            ValueError : if the CSV is empty, the header is unrecognized, or
                         geometry cannot be resolved for any row.
        """
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV '{csv_path}' appears to be empty.")
            headers = [h.strip().lower() for h in reader.fieldnames]

            # Detect geometry mode from column headers.
            has_id = "id"        in headers
            has_od = "od"        in headers
            has_wt = "wt"        in headers
            has_dh = "d_h"       in headers
            has_fa = "flow_area" in headers

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
                row = {k.strip().lower(): v.strip() for k, v in raw_row.items()
                       if k is not None}

                dist_m = float(row["distance"])
                elev_m = float(row["elevation"])

                if mode == "DH_FA":
                    D_h_m = float(row["d_h"])
                    A_m2  = float(row["flow_area"])
                else:
                    # ID_OD_WT mode -- resolve per _resolve_id rules.
                    id_m = float(row["id"]) if has_id else None
                    od_m = float(row["od"]) if has_od else None
                    wt_m = float(row["wt"]) if has_wt else None
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
                f"Profile CSV '{csv_path}' contains no data rows."
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
        """Net elevation change (last point minus first point) [m].
        Positive = uphill overall."""
        return self.profile[-1][1] - self.profile[0][1]

    @property
    def volume_m3(self):
        """Approximate internal volume [m^3].

        Computed as the sum of per-slice volumes using the inlet-point flow
        area for each slice (forward-Euler convention).
        """
        vol = 0.0
        for i in range(len(self.profile) - 1):
            dist_in,  _, _, area_in = self.profile[i]
            dist_out, _, _, _       = self.profile[i + 1]
            vol += area_in * (dist_out - dist_in)
        return vol

    def __repr__(self):
        first = self.profile[0]
        dh_q  = ureg.Quantity(first[2], "m")
        eps_q = ureg.Quantity(self.roughness_si, "m")
        return (
            f"{self.__class__.__name__}("
            f"D_h_inlet={dh_q.to('inch'):.4f~P} [{dh_q:.6f~P}], "
            f"roughness={eps_q.to('ft'):.6f~P} [{eps_q:.8f~P}], "
            f"length={self.total_length_m:.2f} m, "
            f"points={len(self.profile)}, "
            f"noncircular={self.noncircular})"
        )


# ---------------------------------------------------------------------------
# Base_Bend
# ---------------------------------------------------------------------------

class Base_Bend:
    """Geometry storage for a rounded pipe bend fitting.

    Stores the pipe inner diameter, bend angle, and bend-radius-to-diameter
    ratio.  Subclasses add fluid-mechanics pressure-drop calculations
    appropriate for the flow regime (incompressible, compressible, etc.).

    Args:
        Di        : pint Quantity or float (m if float).  Pipe inner diameter.
                    Must be positive.
        ang_deg   : float.  Bend angle [degrees].  0 = straight; 90 = elbow.
                    Must be >= 0.
        bend_dias : float.  Bend centerline radius expressed as a multiple of
                    the pipe inner diameter (e.g. 1.5 for a standard long-
                    radius elbow: R_bend = 1.5 * Di).  Must be positive.
    """

    def __init__(self, Di, ang_deg, bend_dias):
        self.Di_si     = _to_si(Di, "m")
        self.ang_deg   = float(ang_deg)
        self.bend_dias = float(bend_dias)

        if self.Di_si is None or self.Di_si <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: Di must be a positive length."
            )
        if self.ang_deg < 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: ang_deg must be >= 0."
            )
        if self.bend_dias <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: bend_dias must be positive."
            )

    def __repr__(self):
        Di_q = ureg.Quantity(self.Di_si, "m")
        return (
            f"{self.__class__.__name__}("
            f"Di={Di_q.to('inch'):.4f~P}, "
            f"angle={self.ang_deg:.1f} deg, "
            f"bend_diameters={self.bend_dias:.2f})"
        )


# ---------------------------------------------------------------------------
# Base_Contraction_Expansion
# ---------------------------------------------------------------------------

class Base_Contraction_Expansion:
    """Geometry storage for an abrupt contraction or expansion fitting.

    Stores the upstream and downstream inner diameters.  Subclasses add
    fluid-mechanics pressure-drop calculations appropriate for the flow
    regime (incompressible, compressible, etc.).

    If Di_US == Di_DS there is no area change and the fitting has no effect.

    Args:
        Di_US : pint Quantity or float (m if float).  Upstream inner diameter.
                Must be positive.
        Di_DS : pint Quantity or float (m if float).  Downstream inner diameter.
                Must be positive.
    """

    def __init__(self, Di_US, Di_DS):
        self.Di_US_si = _to_si(Di_US, "m")
        self.Di_DS_si = _to_si(Di_DS, "m")

        for name, val in (("Di_US", self.Di_US_si), ("Di_DS", self.Di_DS_si)):
            if val is None or val <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}: {name} must be a positive "
                    f"length."
                )

    def __repr__(self):
        US_q = ureg.Quantity(self.Di_US_si, "m")
        DS_q = ureg.Quantity(self.Di_DS_si, "m")
        return (
            f"{self.__class__.__name__}("
            f"Di_US={US_q.to('inch'):.4f~P}, "
            f"Di_DS={DS_q.to('inch'):.4f~P})"
        )
