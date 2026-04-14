import csv
import math
import os
import warnings
from pint import UnitRegistry
from fluids.friction import friction_factor as fluids_friction_factor
from fluids.core import Reynolds as fluids_Reynolds
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

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
_V_scf = ureg.Quantity(_R * _T_scf / _P_scf, 'm^3')                  # m^3/mol  (~0.028317 * 1/1000 adjusted)
_V_scf = _V_scf.to("ft^3")
_V_scf = _V_scf.magnitude

ureg.define(f'scm  = 1.0 * mol / {_V_scm}')
ureg.define(f'scf  = 1.0 * mol / {_V_scf} ')
ureg.define(f'mscf = {1e3}  *   mol / {_V_scf}')
ureg.define(f'mmscf = {1e6} *  mol / {_V_scf}')


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

def _darcy_friction(Re, eps, d_h):
    """Return the Darcy friction factor and flow regime string.

    Delegates to fluids.friction.friction_factor(), which selects the best
    available correlation automatically.  For Re < 2040 the laminar solution
    f_D = 64 / Re is returned by the library; for Re >= 2040 a high-accuracy
    turbulent correlation (Colebrook exact solution by default) is used.

    The Darcy friction factor relates to the Fanning friction factor by:
        f_Darcy = 4 * f_Fanning

    Args:
        Re  : Reynolds number (dimensionless).
        eps : absolute roughness (m).
        d_h : hydraulic diameter (m).

    Returns:
        (f_darcy, regime_string)
    """
    f_darcy = fluids_friction_factor(Re=Re, eD=eps / d_h)
    regime  = "laminar" if Re < 2040 else "turbulent"
    return f_darcy, regime

# ---------------------------------------------------------------------------
# Two-phase / phase-boundary guard
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compressible dL slice hydraulics
# ---------------------------------------------------------------------------

def compressible_hydraulics(
    abstract_state,
    P_in,
    T_in,
    mdot,
    dL,
    dz,
    D_h,
    roughness,
    flow_area,
    q_wall=0.0,
    isothermal = False
    ):
    #Inputs
    """ 
    Args:
        abstract_state  : CoolProp AbstractState instance, pre-configured for
                          the working fluid (e.g. AbstractState('HEOS','Methane')).
                          This object is updated in-place at (P_in, T_in) on
                          each call; it must NOT be shared across threads.
        P_in            : float, inlet pressure [Pa].
        T_in            : float, inlet temperature [K].
        mdot            : float, mass flow rate (kg/s)
        dL              : float, pipe segment length [m]
        dz              : float, elevation rise over the slice [m].
                          Positive = uphill; negative = downhill.
        D_h             : float, hydraulic diameter [m].
        roughness       : float, absolute pipe-wall roughness [m].
        flow_area       : float, cross-section flow area [m^2].
        q_wall          : float, heat flow into fluid [W] (default 0 =
                          adiabatic).
        isothermal      : boolean, if True then properties are evaluated isothermally (ignoring q_wall if supplied). If false, uses q_wall.

        Calculated:
        rho             : float, mass density [kg/m^3]
        v            : float, flow velocity in direction of flow [m/s]
        mu              : float, kinematic viscosity [Pa*s], from CoolProp
        a               : float, speed of sound [m/s], from CoolProp
        f_darcy          : float, Darcy friction factor, from fluids library correlation
        Ma              : float, Mach number [dimensionless] Ma = v/a
        Re              : float, Reynolds number [dimensionless] Re = rho * v * D_h/mu

    """
    grav_constant=9.8066
    choke_mach_limit = 0.98
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if mdot <= 0.0:
        raise ValueError(
            f"compressible_slice_hydraulics: mass flow rate mdot must be positive "
            f"(received G={mdot:.6g} kg/s)."
        )
    if dL <= 0.0:
        raise ValueError(
            f"compressible_slice_hydraulics: slice length dL must be positive "
            f"(received dL={dL:.6g} m)."
        )
    #need to validate P, T, roughness, flow area too. I feel like this can be done with a short pythonic loop through all of the inputs that need to be positive, raising an error message that mentions all of the invalid inputs.

    # ------------------------------------------------------------------
    # CoolProp state update at inlet conditions
    # ------------------------------------------------------------------
    AS = abstract_state
    AS.update(CP.PT_INPUTS, P_in, T_in)

    phase = AS.phase()
    if phase == _CP_PHASE_TWOPHASE:
        raise RuntimeError(
            f"compressible_slice_hydraulics: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase hydraulics "
            f"only.  Consider checking your inlet conditions."
        )

    rho   = AS.rhomass()                  # kg/m^3
    Cp    = AS.cpmass()                   # J/(kg*K)
    a     = AS.speed_sound()              # m/s  (isentropic speed of sound)
    mu    = AS.viscosity()                # Pa*s
    S_in = AS.smass() #specific entropy in J/kg K
    H_in = AS.hmass() #specific enthalpy in J/kg

    v  = mdot / (rho*flow_area)     # m/s
    Ma = v / a       # Mach number

    if Ma >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_hydraulics: inlet Mach number Ma={Ma:.4f} "
            f"Flow is sonic or near-sonic at inlet conditions already. This program really isn't intended for sonic conditions."
            f"Reduce flow rate or check segment geometry."
        )

    # ------------------------------------------------------------------
    # Reynolds number and friction factor
    # ------------------------------------------------------------------
    Re = fluids_Reynolds(V=v, D=D_h, rho=rho, mu=mu)
    f_darcy, flow_regime = _darcy_friction(Re, roughness, D_h)

    #calculate the change in pressure - individual contributions for troubleshooting.
    dP_contrib_friction = -f_darcy * dL * v**2/(2 * D_h) * rho
    dP_contrib_elevation = - grav_constant * dz * rho

    #and overall change in pressure
    dP = (rho/(1-Ma**2)) * (-f_darcy * dL * v**2/(2 * D_h) - grav_constant * dz)

    #Euler method to find pressure at outlet
    P_out = P_in + dP
    
    
    if isothermal:
        T_out = T_in

        
    else:
        #calculate the change in entropy by using Euler method
        # change in entropy = change in entropy due to heat transfer (q_wall/(mdot * T)) + change in entropy due to friction (f_darcy * v^2/(2*D_h * Tin))
        #Assumes temperature is relatively constant over this slice. If significant temperature changes occur, need to cut down the length slice.
        dS = (q_wall/mdot +f_darcy*v**2 * dL / (2*D_h))/T_in
        
        S_out = S_in + dS

    #next, update the abstract state with the now known outlet specific entropy (or temperature if isothermal) and pressure. Use this to calculate other variables.
    try:
        
        if isothermal:
            AS.update(CP.PT_INPUTS, P_out, T_out)
            S_out = AS.smass()
            

        else:
            AS.update(CP.PSmass_INPUTS, P_out, S_out)
            T_out = AS.T()
        H_out = AS.hmass()
        phase_out = AS.phase()
       
        if phase_out == _CP_PHASE_TWOPHASE:
            raise RuntimeError(
                f"compressible_hydraulics: fluid enters two-phase region at "
                f"outlet (P_out={P_out:.4g} Pa, T_out={T_out:.4g} K).  "
                f"Reduce step size dL={dL:.4g} m or check operating conditions."
            )
        rho_out = AS.rhomass()
        a_out   = AS.speed_sound()
        v_out   = mdot / (rho_out*flow_area)
        Ma_out  = v_out / a_out
        #Energy balance - calculate actual heat leaving wall (Watts). For adiabadic case, should = 0.
        q_wall_actual = ((H_out - H_in) + (v_out**2 - v**2) / 2 + grav_constant*dz)*mdot
        # if isothermal:
        #     q_wall_actual = ((H_out - H_in) + (v_out**2 - v**2) / 2 + grav_constant*dz)*mdot
        # else:
        #     q_wall_actual = q_wall   
    except RuntimeError:
        raise   # re-raise our own RuntimeErrors as-is
    except Exception as exc:
        raise RuntimeError(
            f"compressible_hydraulics: CoolProp failed to evaluate outlet "
            f"state (P_out={P_out:.4g} Pa, S_out={S_out:.4g} J/kg*k).  "
            f"Try a smaller step size? dL={dL:.4g} m is likely too large.  Detail: {exc}"
        ) from exc

    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_hydraulics: outlet Mach number Ma_out={Ma_out:.4f} "
            f">= choke limit {choke_mach_limit}.  Flow choked within this slice "
            f"(dL={dL:.4g} m).  Use smaller steps."
        )

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    return {
        "P_out":        P_out,
        "T_out":        T_out,
        "rho_out":      rho_out,
        "v_in":          v,
        "v_out":         v_out,
        "Ma_in":         Ma,
        "Ma_out":        Ma_out,
        "Re_in":        Re,
        "f_darcy":      f_darcy,
        "flow_regime":  flow_regime,
        "phase":        phase,
        "dP friction":  dP_contrib_friction,
        "dP gravity":   dP_contrib_elevation,
        "S_out"     :   S_out,
        "q_wall_actual" : q_wall_actual,    
    }

def viscosity_LGE(T, mol_wt, density):
    """
    Lee, Gonzalez, and Eakin correlation for hydrocarbon gas viscosity
    T = pint quantity with temperature units
    mol_wt = gas molecular weight, kg/kmol
    density = pint quantity with mass/vol units
    """
    T = T.to("degR").magnitude
    density = density.to("g/cm^3").magnitude
    
    #Temperature input to equation is in degrees R, density input is g/cm^3, returns centipoise. Good gravy those are some ridiculous units.

    x = 3.5+986/(T)+0.01*mol_wt
    k = (9.4+0.02*mol_wt)*((T)**1.5)/(209+19*mol_wt+T)
    y = 2.4 - 0.2 * x
    
    mu = k * math.exp(x * density ** y)/10000.0

    return ureg.Quantity(mu, "cP")

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
        dP_friction_i = -(f_D / 2) * (dist_i / D_h) * rho * v^2  [Pa]

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
            'darcy_friction'   : float
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
    Re       = fluids_Reynolds(V=velocity, D=D_h, rho=rho, mu=mu)

    # --- Friction factor (uniform along segment for liquid) ---
    f_darcy, flow_regime = _darcy_friction(Re, eps, D_h)

    # Friction pressure gradient (Pa/m), positive magnitude -- loss per unit length.
    # Darcy-Weisbach:  dP/dx = (f_D / 2) * rho * v^2 / D_h
    friction_gradient = (f_darcy / 2.0) * rho * velocity ** 2 / D_h

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
        "darcy_friction":   f_darcy,
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
    print(f"  Darcy friction       : {results['darcy_friction']:.6f}  ({results['flow_regime']})")
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

def test_p2p():
    #simple 2 point pressure drop example
    
    #pipe segment definition
    roughness=ureg.Quantity(0.00015, "ft")
    id_val=ureg.Quantity(3.068, "inch")
    length = ureg.Quantity(2000.0, "ft")
    elevation_change = ureg.Quantity(25.0, "ft")
    segment = Line_Segment(
            roughness=roughness,
            id_val=id_val,
            length = length,
            elevation_change = elevation_change
        )

    #fluid definition
    fluid = Incompressible_Fluid(
        density=ureg.Quantity(1000.0, "kg/m^3"),
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    flow_rate = ureg.Quantity(2000, "oil_bbl/day")
    # --- Run calculation ---
    results = liquid_hydraulics(fluid, segment, flow_rate)

    # --- Display and export ---
    print_results(results, fluid, segment, flow_rate)

    output_csv = os.path.splitext("simple.csv")[0] + "_pressure_profile.csv"
    export_pressure_profile(results, output_csv)

def test_csv_profile():
    #Example with csv file defined profile
    # --- Fluid definition ---
    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )

    # --- Pipe segment loaded from CSV ---
    segment = Line_Segment.from_csv(
        csv_path="testprofile.csv",
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
    )

    # --- Flow rate ---
    flow_rate = ureg.Quantity(2000, "oil_bbl/day")

    # --- Run calculation ---
    results = liquid_hydraulics(fluid, segment, flow_rate)

    # --- Display and export ---
    print_results(results, fluid, segment, flow_rate)

    output_csv = os.path.splitext("testprofile.csv")[0] + "_pressure_profile.csv"
    export_pressure_profile(results, output_csv)

    
def test_comp_hydraulics():

    P_gas    = 790798.0    # Pa
    T_gas    = 299.8    # K
    D_gas    = ureg.Quantity(4.026, "inch").to("m").magnitude
    A_gas    = math.pi * D_gas**2 / 4.0
    eps_gas  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    dL_gas   = ureg.Quantity(100.0,    "feet").to("m").magnitude
    dz_gas   = ureg.Quantity(-100.0,    "feet").to("m").magnitude
    AS_g = AbstractState("HEOS", "Methane")
    AS_g.update(CP.PT_INPUTS, P_gas, T_gas)
    rho_gas = AS_g.rhomass()
    S_in = AS_g.smass()
    Q_scfd   = ureg.Quantity(1.0, "mmscf/day")
    mdot     = Q_scfd.to("mol/s").magnitude * 16.043e-3   # kg/s from mol wt
    G_gas    = mdot / A_gas

    print('\n')
    print(f'inputs: P = {P_gas}, Smass_in= {S_in}')
    print('\n')
    result_g = compressible_hydraulics(
        abstract_state=AS_g,
        P_in=P_gas,
        T_in=T_gas,
        mdot = mdot,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        flow_area=A_gas,
        isothermal = True,
    )

    print(result_g)

def test_compressible_slices():
    slicecount = 20
    total_length = ureg.Quantity(100.0,    "feet").to("m").magnitude
    total_elev_change = ureg.Quantity(100.0,    "feet").to("m").magnitude
    D_pipe    = ureg.Quantity(4.026, "inch").to("m").magnitude
    eps_pipe  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    Q_scfd   = ureg.Quantity(1.0, "mmscf/day")

    dL   = total_length / slicecount
    dz  = total_elev_change/slicecount

    #generate profile assuming uniform slices of length and elevation change

    profile = []

    for i in range(slicecount + 1):
        profile.append([i * dL, i * dz])

    segment = Line_Segment(
            roughness=eps_pipe,
            id_val=D_pipe,
            profile= profile
        )

    #initial conditions
    P_in    = 790798.0    # Pa
    T_in    = 299.8    # K

    A_pipe    = math.pi * D_gas**2 / 4.0

    AS_g = AbstractState("HEOS", "Methane")
    AS_g.update(CP.PT_INPUTS, P_gas, T_gas)
    rho_in = AS_g.rhomass()
    S_in = AS_g.smass()

    mdot     = Q_scfd.to("mol/s").magnitude * 16.043e-3   # kg/s from mol wt

    #loop through and call compressible_hydraulics on each slice of the line segment, using the output of each slice as the inputs for the next



if __name__ == "__main__":

    # test_p2p()
    #test_comp_hydraulics():
    test_compressible_slices()
