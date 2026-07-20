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

Base_Valve
    A valve fitting.  Stores pipe inner diameter and a pre-computed K-factor
    (resistance coefficient).

Base_CheckValve
    A check valve fitting.  Same layout as Base_Valve (Di + K_forward) but
    carries check_valve=True so the network solvers treat the edge as a
    perfect seal under reverse flow (exactly zero backflow).

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

# Remove pint's default 'bbl' / 'barrel' (31.5 US gal, the fluid barrel).  In
# pipeline work "barrel" always means the 42-gal petroleum barrel, which pint
# spells 'oil_bbl'.  Deleting forces a UndefinedUnitError on accidental use
# rather than silently scaling flow rates by 0.75.  Touches a pint internal
# (ureg._units); revisit if pint changes its registry layout.
del ureg._units['bbl']
del ureg._units['barrel']

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
# Profile downsampling
# ---------------------------------------------------------------------------

def downsample_profile(profile, max_step_m=1000.0,
                       slope_tol=1e-6, diameter_tol=1e-9, elev_tol=0.0):
    """Downsample a pipe-segment profile, keeping only geometrically
    meaningful points and enforcing a maximum spacing between them.

    Profile is the list-of-tuples format used by Base_Line_Segment: each
    row is (distance_m, elevation_m, D_h_m, flow_area_m2).  The same
    format is returned.

    Closely-spaced profile points dramatically slow compressible-flow
    integrations because the solver is called once per consecutive point
    pair.  This helper produces a coarser profile that retains the
    features that actually affect the result:

      * The first and last point are always kept.
      * Both points bounding any diameter change are kept, so area-change
        corrections see the correct A_in/A_out.
      * Any interior point whose left-slope (to its previous neighbor)
        differs from its right-slope (to its next neighbor) by more than
        slope_tol is kept -- i.e. grade breaks and other polyline vertices.
      * When elev_tol > 0, slope-break-only points are then filtered: a
        point is dropped if its elevation is within elev_tol of the last
        retained point's elevation.  Diameter-change points are exempt.
        This removes the redundant pairs that quantized elevation models
        (e.g. QGIS polyline exports) produce at every step boundary.
      * Between any two kept points whose spacing exceeds max_step_m,
        evenly-spaced intermediate points are inserted from the original
        profile to cap the spacing.

    Args:
        profile      : list of (dist_m, elev_m, D_h_m, A_m2) tuples,
                       sorted by ascending distance.
        max_step_m   : float, maximum allowed spacing [m] between
                       retained points.  Default 1000.0.
        slope_tol    : float, slope-difference threshold for declaring
                       a polyline vertex (dimensionless m/m).
                       Default 1e-6.
        diameter_tol : float, |dD_h| threshold for declaring a
                       diameter change [m].  Default 1e-9.
        elev_tol     : float, minimum elevation change [m] from the last
                       retained point required to keep a slope-break
                       point.  0.0 (default) disables this filter.

    Returns:
        New list of profile rows.  Always at least min(2, len(profile))
        points.
    """
    n = len(profile)
    if n <= 2:
        return list(profile)

    keep = [False] * n
    keep[0]  = True
    keep[-1] = True

    # Diameter changes -- keep both sides of the step so an area-change
    # correction can be applied across the discontinuity.
    diameter_required = {0, n - 1}
    for i in range(1, n):
        if abs(profile[i][2] - profile[i - 1][2]) > diameter_tol:
            keep[i - 1] = True
            keep[i]     = True
            diameter_required.add(i - 1)
            diameter_required.add(i)

    # Polyline slope breaks.
    for i in range(1, n - 1):
        d0, e0 = profile[i - 1][0], profile[i - 1][1]
        d1, e1 = profile[i    ][0], profile[i    ][1]
        d2, e2 = profile[i + 1][0], profile[i + 1][1]
        # Guard against duplicate-distance rows: undefined slope, keep the
        # point rather than silently dropping a possible vertex.
        if d1 <= d0 or d2 <= d1:
            keep[i] = True
            continue
        slope_left  = (e1 - e0) / (d1 - d0)
        slope_right = (e2 - e1) / (d2 - d1)
        if abs(slope_right - slope_left) > slope_tol:
            keep[i] = True

    # Elevation-tolerance filter: drop slope-break-only points whose
    # elevation is within elev_tol of the last retained point's elevation.
    # Diameter-change points are always exempt.
    if elev_tol > 0.0:
        last_elev = profile[0][1]
        for i in range(1, n - 1):
            if not keep[i]:
                continue
            if i in diameter_required:
                last_elev = profile[i][1]
            elif abs(profile[i][1] - last_elev) < elev_tol:
                keep[i] = False
            else:
                last_elev = profile[i][1]

    # Enforce max_step by inserting points from the original profile
    # wherever the spacing between consecutive kept points is too large.
    kept_idx = [i for i, k in enumerate(keep) if k]
    final_idx = [kept_idx[0]]
    for k in range(1, len(kept_idx)):
        i0 = final_idx[-1]
        i1 = kept_idx[k]
        gap = profile[i1][0] - profile[i0][0]
        if gap > max_step_m:
            n_sub = int(math.ceil(gap / max_step_m))
            for j in range(1, n_sub):
                target = profile[i0][0] + j * gap / n_sub
                lo = final_idx[-1] + 1
                hi = i1 - 1
                if lo > hi:
                    break
                best = lo
                best_diff = abs(profile[best][0] - target)
                for m in range(lo + 1, hi + 1):
                    diff = abs(profile[m][0] - target)
                    if diff < best_diff:
                        best = m
                        best_diff = diff
                if best > final_idx[-1]:
                    final_idx.append(best)
        final_idx.append(i1)

    return [profile[i] for i in final_idx]


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
        name=None,
    ):
        self.name = name
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
        name=None,
        downsample=False,
        elev_tol=0.0,
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
            name         : str or None.  Optional label for the segment.
                           Default None.
            downsample   : bool or float.  If False (default) the full
                           profile is used.  If True, downsample_profile()
                           is called with its default max_step_m=1000 m.
                           If a positive float, that value is used as
                           max_step_m [m].
            elev_tol     : float [m].  Passed to downsample_profile() as
                           the minimum elevation change required to retain
                           a slope-break point.  Only used when downsample
                           is not False.  Default 0.0 (disabled).

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

        if downsample is not False:
            max_step_m = 1000.0 if downsample is True else float(downsample)
            n_before = len(rows)
            rows = downsample_profile(rows, max_step_m=max_step_m,
                                      elev_tol=elev_tol)
            print(
                f"  Downsampled profile: {n_before} → {len(rows)} points "
                f"(max_step={max_step_m:.0f} m, elev_tol={elev_tol:.3f} m)"
            )

        total_dist_m = max(r[0] for r in rows)
        net_elev_m   = rows[-1][1] - rows[0][1]
        print(
            f"  Profile loaded from '{csv_path}': {len(rows)} points, "
            f"total distance = {total_dist_m:.2f} m, "
            f"net elevation change = {net_elev_m:.4f} m  [mode={mode}]"
        )

        seg = cls(
            roughness=roughness,
            profile=rows,
            noncircular=noncircular,
            id_tolerance=id_tolerance,
            k_wall=k_wall,
            name=name,
        )
        # Tag the segment with its CSV origin so to_dict() can save the path
        # instead of the full profile (per the network-save spec: CSV-backed
        # segments re-load from the CSV at load time, so edits to the CSV
        # propagate without needing to re-save the network file).
        seg._csv_path = csv_path
        return seg

    # ------------------------------------------------------------------
    # JSON (de)serialization for Network.save / Network.load.
    #
    # CSV-backed segments are stored as {"csv_path": ...} and reloaded via
    # from_csv() so a user can tweak the CSV between runs without re-saving
    # the network.  Manually-constructed segments are stored as a full
    # profile list (SI floats only).
    # ------------------------------------------------------------------

    def to_dict(self):
        d = {
            "kind":        "line_segment",
            "name":        self.name,
            "roughness_m": float(self.roughness_si),
            "noncircular": bool(self.noncircular),
            "k_wall":      None if self.k_wall_si is None else float(self.k_wall_si),
        }
        csv_path = getattr(self, "_csv_path", None)
        if csv_path:
            d["csv_path"] = csv_path
        else:
            d["profile"] = [list(map(float, row)) for row in self.profile]
        return d

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "line_segment":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='line_segment', "
                f"got {payload.get('kind')!r}."
            )
        roughness = payload["roughness_m"]
        noncircular = bool(payload.get("noncircular", False))
        k_wall = payload.get("k_wall")
        name = payload.get("name")
        if "csv_path" in payload:
            return cls.from_csv(
                payload["csv_path"],
                roughness=roughness,
                noncircular=noncircular,
                k_wall=k_wall,
                name=name,
            )
        profile = [tuple(row) for row in payload["profile"]]
        return cls(
            roughness=roughness,
            profile=profile,
            noncircular=noncircular,
            k_wall=k_wall,
            name=name,
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

    @property
    def inlet_area_si(self):
        return self.profile[0][3]

    @property
    def outlet_area_si(self):
        return self.profile[-1][3]

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

    def __init__(self, Di, ang_deg, bend_dias, name=None):
        self.name      = name
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

    @property
    def inlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    @property
    def outlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    def to_dict(self):
        return {
            "kind":      "bend",
            "name":      self.name,
            "Di_m":      float(self.Di_si),
            "ang_deg":   float(self.ang_deg),
            "bend_dias": float(self.bend_dias),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "bend":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='bend', "
                f"got {payload.get('kind')!r}."
            )
        return cls(
            Di=payload["Di_m"],
            ang_deg=payload["ang_deg"],
            bend_dias=payload["bend_dias"],
            name=payload.get("name"),
        )


# ---------------------------------------------------------------------------
# Base_Valve
# ---------------------------------------------------------------------------

class Base_Valve:
    """Geometry storage for a valve fitting.

    Stores the pipe inner diameter and a resistance coefficient (K-factor).
    The K-factor may be supplied directly, or back-calculated from a flow
    coefficient: Cv (US, gpm/psi^0.5) or Kv (metric, m^3/h/bar^0.5).
    Subclasses add fluid-mechanics pressure-drop calculations appropriate
    for the flow regime (incompressible, compressible, etc.).

    Exactly one of K, Cv, or Kv must be supplied.

    Cv is defined as Cv = Q [gpm] * sqrt (specific gravity / ΔP [psi])
    Kv is define similarly in metric units Kv = Q [m^3/h] * sqrt (specific gravity / ΔP [bar])

    Converting units allows us to relate Cv and Kv:

        Cv = 1.156 * Kv

    The fluids library has correlations for valves in terms of the dimensionless K factor, so
    we would ideally want to convert a user-specified Cv or Kv to the dimensionless K to keep 
    the valve pressure drop calculations consistent. It's a units mess but it is doable.

    You can relate the Cv-termed pressure drop to the pressure drop term
    with the dimensionless K factor:

        ΔP = K * rho * v^2/2 = (Q [gpm]/Cv)^2 * rho [lb/ft^3]/62.4

    Solving for K and shaking out the units gives us:

        K  = 2.166e9 * Di^4 / Cv^2     (Di in m, Cv in gpm/psi^0.5)

    The Python fluids library also has a convenient conversion utility that I didn't find until after I had 
    plowed through the units conversion. Goes to show that six weeks in the lab can save six hours in the library.
    
    fluids.fittings.Cv_to_K(Cv: float, D: float)

    Args:
        Di : pint Quantity or float (m if float).  Pipe inner diameter.
             Must be positive.
        K  : float, optional.  Resistance coefficient supplied directly.
             Must be >= 0.
        Cv : float, optional.  US flow coefficient [gpm/psi^0.5].
             Must be positive.
        Kv : float, optional.  Metric flow coefficient [m^3/h/bar^0.5].
             Must be positive.
        minimum_diameter : pint Quantity or float (m if float), optional.
             Geometric minimum flow cross-section inside the trim (seat
             diameter for globe, port for ball, etc.).  Sharpens the
             compressible-flow choke check by giving the real throat area
             rather than the pipe area.  Geometric only -- does not capture
             pressure-recovery (F_L) effects, so valves with strong recovery
             may choke earlier than this predicts.  None uses a pipe-area check.
        F_L : float, optional.  ISA-75.01 liquid pressure-recovery factor
             (dimensionless, typical 0.6 - 1.0, manufacturer-supplied).
             Consumed only by the incompressible Valve.dP / Valve.dmdot
             cavitation check.  When supplied together with the fluid's
             vapor_pressure_si, enables the three-regime cavitation gate
             (flashing / choked cavitating / incipient).  None disables the
             check.  The compressible Valve ignores F_L (it uses
             minimum_diameter for sonic-choke detection).
    """

    def __init__(self, Di, K=None, name=None, Cv=None, Kv=None,
                 minimum_diameter=None, F_L=None):
        self.name  = name
        self.Di_si = _to_si(Di, "m")

        if self.Di_si is None or self.Di_si <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: Di must be a positive length."
            )

        n_specified = sum(v is not None for v in (K, Cv, Kv))
        if n_specified != 1:
            raise ValueError(
                f"{self.__class__.__name__}: supply exactly one of K, Cv, or Kv "
                f"(got {n_specified})."
            )

        self.Cv = float(Cv) if Cv is not None else None
        self.Kv = float(Kv) if Kv is not None else None

        if K is not None:
            self.K = float(K)
        else:
            cv_used = self.Cv if self.Cv is not None else 1.156 * self.Kv
            if cv_used <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}: Cv/Kv must be positive."
                )
            self.K = 2.166e9 * self.Di_si ** 4 / cv_used ** 2

        if self.K < 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: K must be >= 0."
            )

        self.D_min_si = _to_si(minimum_diameter, "m") if minimum_diameter is not None else None
        if self.D_min_si is not None:
            if self.D_min_si <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}: minimum_diameter must be positive."
                )
            if self.D_min_si > self.Di_si:
                raise ValueError(
                    f"{self.__class__.__name__}: minimum_diameter "
                    f"({self.D_min_si:.4g} m) cannot exceed pipe Di "
                    f"({self.Di_si:.4g} m)."
                )

        # Effective discharge coefficient implied by (K, D_min, Di).  From
        # dP = K*rho*v_pipe^2/2 and mdot = Cd*A_min*sqrt(2*rho*dP):
        #     Cd_eff = (Di/D_min)^2 / sqrt(K)
        # Useful only as an input-sanity diagnostic; the actual choke check
        # uses A_throat directly and does not consume Cd_eff.
        self.Cd_eff = None
        if self.D_min_si is not None and self.K > 0.0:
            self.Cd_eff = (self.Di_si / self.D_min_si) ** 2 / math.sqrt(self.K)
            if self.Cd_eff > 1.05:
                warnings.warn(
                    f"{self.__class__.__name__}: implied Cd_eff="
                    f"{self.Cd_eff:.3f} > 1.05 from (K={self.K:.3g}, "
                    f"Di={self.Di_si:.4g} m, D_min={self.D_min_si:.4g} m).  "
                    f"minimum_diameter is likely too small for the specified K.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.Cd_eff < 0.3:
                warnings.warn(
                    f"{self.__class__.__name__}: implied Cd_eff="
                    f"{self.Cd_eff:.3f} < 0.3 from (K={self.K:.3g}, "
                    f"Di={self.Di_si:.4g} m, D_min={self.D_min_si:.4g} m).  "
                    f"K seems high for the specified minimum_diameter; the "
                    f"trim may have significant downstream pressure recovery "
                    f"(F_L < 1) that this geometric check cannot model.",
                    UserWarning,
                    stacklevel=2,
                )

        self.F_L = float(F_L) if F_L is not None else None
        if self.F_L is not None and not (0.0 < self.F_L <= 1.0):
            raise ValueError(
                f"{self.__class__.__name__}: F_L must be in (0, 1], got "
                f"{self.F_L}."
            )

    def __repr__(self):
        Di_q = ureg.Quantity(self.Di_si, "m")
        extra = ""
        if self.Cv is not None:
            extra = f", Cv={self.Cv:.3f}"
        elif self.Kv is not None:
            extra = f", Kv={self.Kv:.3f}"
        if self.D_min_si is not None:
            Dmin_q = ureg.Quantity(self.D_min_si, "m")
            extra += f", D_min={Dmin_q.to('inch'):.4f~P}"
        if self.F_L is not None:
            extra += f", F_L={self.F_L:.3f}"
        return (
            f"{self.__class__.__name__}("
            f"Di={Di_q.to('inch'):.4f~P}, "
            f"K={self.K:.4f}{extra})"
        )

    @property
    def inlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    @property
    def outlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    def to_dict(self):
        d = {
            "kind": "valve",
            "name": self.name,
            "Di_m": float(self.Di_si),
            "K":    float(self.K),
        }
        if self.Cv is not None:
            d["Cv"] = self.Cv
        if self.Kv is not None:
            d["Kv"] = self.Kv
        if self.D_min_si is not None:
            d["D_min_m"] = float(self.D_min_si)
        if self.F_L is not None:
            d["F_L"] = float(self.F_L)
        return d

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "valve":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='valve', "
                f"got {payload.get('kind')!r}."
            )
        kwargs = {
            "Di":   payload["Di_m"],
            "name": payload.get("name"),
        }
        if "D_min_m" in payload:
            kwargs["minimum_diameter"] = payload["D_min_m"]
        if "F_L" in payload:
            kwargs["F_L"] = payload["F_L"]
        if "Cv" in payload:
            return cls(Cv=payload["Cv"], **kwargs)
        if "Kv" in payload:
            return cls(Kv=payload["Kv"], **kwargs)
        return cls(K=payload["K"], **kwargs)


# ---------------------------------------------------------------------------
# Base_CheckValve
# ---------------------------------------------------------------------------

class Base_CheckValve:
    """Geometry storage for a check valve fitting.

    Identical in layout to Base_Valve (Di + K) but carries a class-level
    ``check_valve = True`` marker.  The network solvers detect this and
    treat the valve as a perfect seal: an edge carrying a check valve is
    pinned to exactly zero reverse flow by a complementarity residual
    (see network.md, "Reverse-flow handling").  K applies to forward
    flow only.

    Args:
        Di : pint Quantity or float (m if float).  Pipe inner diameter used
             as the velocity-head reference for dP.  Must be positive.
        K  : float.  Forward-flow resistance coefficient.  Must be >= 0.
        minimum_diameter : pint Quantity or float (m if float), optional.
             Geometric minimum flow cross-section inside the trim.  See
             Base_Valve docstring for details and caveats.  Forward flow
             only (reverse flow through a check valve is zero).
        F_L : float, optional.  ISA-75.01 liquid pressure-recovery factor.
             See Base_Valve docstring.  Used only by the incompressible
             CheckValve cavitation check on the forward-flow path.
    """

    check_valve = True

    def __init__(self, Di, K, name=None, minimum_diameter=None, F_L=None):
        self.name  = name
        self.Di_si = _to_si(Di, "m")
        self.K     = float(K)

        if self.Di_si is None or self.Di_si <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: Di must be a positive length."
            )
        if self.K < 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: K must be >= 0."
            )

        self.D_min_si = _to_si(minimum_diameter, "m") if minimum_diameter is not None else None
        if self.D_min_si is not None:
            if self.D_min_si <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}: minimum_diameter must be positive."
                )
            if self.D_min_si > self.Di_si:
                raise ValueError(
                    f"{self.__class__.__name__}: minimum_diameter "
                    f"({self.D_min_si:.4g} m) cannot exceed pipe Di "
                    f"({self.Di_si:.4g} m)."
                )

        # See Base_Valve for the derivation of Cd_eff.
        self.Cd_eff = None
        if self.D_min_si is not None and self.K > 0.0:
            self.Cd_eff = (self.Di_si / self.D_min_si) ** 2 / math.sqrt(self.K)
            if self.Cd_eff > 1.05:
                warnings.warn(
                    f"{self.__class__.__name__}: implied Cd_eff="
                    f"{self.Cd_eff:.3f} > 1.05 from (K={self.K:.3g}, "
                    f"Di={self.Di_si:.4g} m, D_min={self.D_min_si:.4g} m).  "
                    f"minimum_diameter is likely too small for the specified K.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.Cd_eff < 0.3:
                warnings.warn(
                    f"{self.__class__.__name__}: implied Cd_eff="
                    f"{self.Cd_eff:.3f} < 0.3 from (K={self.K:.3g}, "
                    f"Di={self.Di_si:.4g} m, D_min={self.D_min_si:.4g} m).  "
                    f"K seems high for the specified minimum_diameter; the "
                    f"trim may have significant downstream pressure recovery "
                    f"(F_L < 1) that this geometric check cannot model.",
                    UserWarning,
                    stacklevel=2,
                )

        self.F_L = float(F_L) if F_L is not None else None
        if self.F_L is not None and not (0.0 < self.F_L <= 1.0):
            raise ValueError(
                f"{self.__class__.__name__}: F_L must be in (0, 1], got "
                f"{self.F_L}."
            )

    def __repr__(self):
        Di_q = ureg.Quantity(self.Di_si, "m")
        extra = ""
        if self.D_min_si is not None:
            Dmin_q = ureg.Quantity(self.D_min_si, "m")
            extra = f", D_min={Dmin_q.to('inch'):.4f~P}"
        if self.F_L is not None:
            extra += f", F_L={self.F_L:.3f}"
        return (
            f"{self.__class__.__name__}("
            f"Di={Di_q.to('inch'):.4f~P}, "
            f"K_fwd={self.K:.4f}{extra})"
        )

    @property
    def inlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    @property
    def outlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    def to_dict(self):
        d = {
            "kind": "check_valve",
            "name": self.name,
            "Di_m": float(self.Di_si),
            "K":    float(self.K),
        }
        if self.D_min_si is not None:
            d["D_min_m"] = float(self.D_min_si)
        if self.F_L is not None:
            d["F_L"] = float(self.F_L)
        return d

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "check_valve":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='check_valve', "
                f"got {payload.get('kind')!r}."
            )
        kwargs = {
            "Di":   payload["Di_m"],
            "K":    payload["K"],
            "name": payload.get("name"),
        }
        if "D_min_m" in payload:
            kwargs["minimum_diameter"] = payload["D_min_m"]
        if "F_L" in payload:
            kwargs["F_L"] = payload["F_L"]
        return cls(**kwargs)


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

    def __init__(self, Di_US, Di_DS, name=None):
        self.name     = name
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

    @property
    def inlet_area_si(self):
        return math.pi * self.Di_US_si ** 2 / 4.0

    @property
    def outlet_area_si(self):
        return math.pi * self.Di_DS_si ** 2 / 4.0

    def to_dict(self):
        return {
            "kind":    "contraction_expansion",
            "name":    self.name,
            "Di_US_m": float(self.Di_US_si),
            "Di_DS_m": float(self.Di_DS_si),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "contraction_expansion":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='contraction_expansion', "
                f"got {payload.get('kind')!r}."
            )
        return cls(
            Di_US=payload["Di_US_m"],
            Di_DS=payload["Di_DS_m"],
            name=payload.get("name"),
        )


# ---------------------------------------------------------------------------
# Base_Orifice
# ---------------------------------------------------------------------------

_ORIFICE_VALID_TAPS = {"corner", "D and D/2", "flange"}


class Base_Orifice:
    """Geometry storage for a square-edged concentric orifice plate.

    Stores the pipe inner diameter, orifice bore diameter, and tap type.
    Subclasses add fluid-mechanics pressure-drop calculations appropriate
    for the flow regime (incompressible, compressible, etc.).

    The discharge coefficient is computed per ISO 5167-2 / Reader-Harris-
    Gallagher unless overridden.  Valid beta range for the RHG correlation
    is 0.10–0.75; a warning is issued outside this range.

    Args:
        Di          : pint Quantity or float (m if float).  Pipe inner
                      diameter.  Must be positive.
        Do          : pint Quantity or float (m if float).  Orifice bore
                      diameter.  Must satisfy 0 < Do < Di.
        taps        : str.  Pressure tap type: 'corner' (default),
                      'D and D/2', or 'flange'.
        Cd_override : float or None.  Fixed discharge coefficient.  If
                      supplied, the RHG correlation is bypassed entirely.
                      Must be in (0, 1].  Default None.
    """

    def __init__(self, Di, Do, taps="corner", Cd_override=None, name=None):
        self.name  = name
        self.Di_si = _to_si(Di, "m")
        self.Do_si = _to_si(Do, "m")

        if self.Di_si is None or self.Di_si <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: Di must be a positive length."
            )
        if self.Do_si is None or self.Do_si <= 0.0 or self.Do_si >= self.Di_si:
            raise ValueError(
                f"{self.__class__.__name__}: Do must satisfy 0 < Do < Di "
                f"(got Di={self.Di_si:.6f} m, Do={self.Do_si:.6f} m)."
            )

        self.beta = self.Do_si / self.Di_si

        if self.beta < 0.10 or self.beta > 0.75:
            warnings.warn(
                f"{self.__class__.__name__}: beta={self.beta:.4f} is outside "
                f"the ISO 5167-2 / RHG correlation range [0.10, 0.75].  "
                f"Discharge coefficient accuracy may be reduced.",
                UserWarning,
                stacklevel=2,
            )

        if taps not in _ORIFICE_VALID_TAPS:
            raise ValueError(
                f"{self.__class__.__name__}: taps must be one of "
                f"{sorted(_ORIFICE_VALID_TAPS)}, got {taps!r}."
            )
        self.taps = taps

        if Cd_override is not None:
            Cd_override = float(Cd_override)
            if not (0.0 < Cd_override <= 1.0):
                raise ValueError(
                    f"{self.__class__.__name__}: Cd_override must be in (0, 1], "
                    f"got {Cd_override}."
                )
        self.Cd_override = Cd_override

    @property
    def inlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    @property
    def outlet_area_si(self):
        return math.pi * self.Di_si ** 2 / 4.0

    def __repr__(self):
        Di_q   = ureg.Quantity(self.Di_si, "m")
        Do_q   = ureg.Quantity(self.Do_si, "m")
        cd_str = (f", Cd_override={self.Cd_override:.4f}"
                  if self.Cd_override is not None else "")
        return (
            f"{self.__class__.__name__}("
            f"Di={Di_q.to('inch'):.4f~P}, "
            f"Do={Do_q.to('inch'):.4f~P}, "
            f"beta={self.beta:.4f}, "
            f"taps={self.taps!r}"
            f"{cd_str})"
        )

    def to_dict(self):
        d = {
            "kind": "orifice",
            "name": self.name,
            "Di_m": float(self.Di_si),
            "Do_m": float(self.Do_si),
            "taps": self.taps,
        }
        if self.Cd_override is not None:
            d["Cd_override"] = self.Cd_override
        return d

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") != "orifice":
            raise ValueError(
                f"{cls.__name__}.from_dict: expected kind='orifice', "
                f"got {payload.get('kind')!r}."
            )
        return cls(
            Di=payload["Di_m"],
            Do=payload["Do_m"],
            taps=payload.get("taps", "corner"),
            Cd_override=payload.get("Cd_override"),
            name=payload.get("name"),
        )
