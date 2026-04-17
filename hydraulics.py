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
_V_scf = ureg.Quantity(_R * _T_scf / _P_scf, 'm^3')          #1 scf = 1.20 moles, but let the calculation be done in case you need to use some oddball standard conditions
_V_scf = _V_scf.to("ft^3")
_V_scf = _V_scf.magnitude

ureg.define(f'scm  = {1.0/_V_scm} * mol ')
ureg.define(f'scf  = {1.0/_V_scf} * mol ')
ureg.define(f'mscf = {1e3/_V_scf}  *   mol ')
ureg.define(f'mmscf = {1e6/_V_scf}*  mol ')


# ---------------------------------------------------------------------------
# Incompressible Fluid class
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
    isothermal = False,
    mu = None 
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
        mu              : float, viscosity in Pascal-seconds. If not supplied, first tries to calculate from the abstract state, and if that doesn't work, falls back to the Lee, Gonzalez, and Eakin correlation

        Calculated:
        rho             : float, mass density [kg/m^3]
        v            : float, flow velocity in direction of flow [m/s]
        mu              : float, dynamic viscosity [Pa*s], from CoolProp
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
            f"(received mdot={mdot:.6g} kg/s)."
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
    if mu is None:
        try:
            mu    = AS.viscosity()                # Pa*s
        except:
            #If the abstract state is not able to calculate mixture viscosity (like if the Peng Robinson equation is used), use a backup viscosity correlation.
            #Note that the Lee, Gonzalez, Eakin equation is really only intended for hydrocarbon gas mixtures, so if you have some other material, you should supply mu to the function.
            mu = viscosity_LGE(T_in, AS.molar_mass() * 1000.0, AS.rhomass())
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

    #Euler method to find pressure at outlet. 
    # Assumes density, Mach number/velocity, and friction factor are relatively constant over this slice. 
    # If they're not, need to use a smaller length slice or a better numerical integration scheme like Runge-Kutta.
    P_out = P_in + dP
    
    #Now we have P but need one more thermodynamic state variable to be able to calculate everything else.
    if isothermal:
        #Easy peasy for the isothermal case
        T_out = T_in
        
    else:
        #Ideally we would just calculate the output entropy and be good to go with that as our second state variable, and this works fine for single component systems. 
        # change in entropy = change in entropy due to heat transfer (q_wall/(mdot * T)) + change in entropy due to friction (f_darcy * v^2/(2*D_h * Tin))
        # S_out = S_in + dS
        # However, with multicomponent systems, CoolProp has trouble using entropy as one of the inputs, so for robustness we need to use a thermodynamic identity to convert it to something easier to deal with, like temperature.
        # Use dH = TdS + VdP = Cp dT + [V - T (∂V/∂T)_P] dP (from table 6.2-1 in "Chemical, Biochemical, and Engineering Thermodynmics, 4th ed." by Sandler)
        #                                   ^ [V - T (∂V/∂T)_P] is the Joule Thompson coefficient μ times negative heat capacity [-μCp]
        #substitute (∂V/∂T)_P = -1/ρ^2 (∂ρ/∂T)_P and solve for dT
        # Then, use the Euler method to calculate T_out        
        #Assumes temperature, density, heat capacity, velocity, and our weird isobaric thermal expansion (∂ρ/∂T)_P property are all relatively constant over this slice. 
        # If significant changes occur over the slice, need to cut down the length slice (or implement a better numerical integration scheme like Runge-Kutta).
        
        Cp = AS.cpmass()
        drhodT_P = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

        dT = (1/Cp) * (q_wall/mdot - (T_in/rho**2) * drhodT_P * dP + (f_darcy * dL * v**2)/(2 * D_h))

        T_out = T_in + dT

    #next, update the abstract state with the now known outlet temperature and pressure. Use this to calculate other variables.
    try:
        
        AS.update(CP.PT_INPUTS, P_out, T_out)

        H_out = AS.hmass()
        S_out = AS.smass()
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
        #Energy balance - calculate actual heat leaving wall (Watts) as a check on our results. For adiabadic case, should = 0. 
        # It won't actually be exactly that, due to numerical integration erros, but it's a decent measure of the accuracy of 
        # the approximation and tells you if you need to cut down your length step size or if you're getting outside of the zone of validity of the model.
        q_wall_actual = ((H_out - H_in) + (v_out**2 - v**2) / 2 + grav_constant*dz)*mdot

    except RuntimeError:
        raise   # re-raise our own RuntimeErrors as-is
    except Exception as exc:
        raise RuntimeError(
            f"compressible_hydraulics: CoolProp failed to evaluate outlet "
            f"state (P_out={P_out:.4g} Pa, T_out={T_out:.4g} K).  "
            f"step size dL={dL:.4g} m  Detail: {exc}"
        ) from exc

    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_hydraulics: outlet Mach number Ma_out={Ma_out:.4f} "
            f">= choke limit {choke_mach_limit}.  Flow choked within this slice "

        )

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    # Maybe just change this to return the abstract state, from which the P, T, S, rho, phase, and whatever else can be extracted, along with maybe Ma and Re too and q_wall_actual.
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


def liquid_hydraulics(fluid, segment, flow_rate):
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

    grav_constant   = 9.8066
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
        dP_static   =  rho * grav_constant * (elev_ref - elev_m)
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
# Compressible profile CSV runner
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


def _convert_comp_output(value_si, quantity_type, unit_spec):
    """Convert a single SI value to the output unit specified in unit_spec.

    Args:
        value_si      : float, value in SI units.
        quantity_type : str, one of 'dist', 'elev', 'P', 'T', 'v'.
        unit_spec     : dict, one entry from _COMP_OUTPUT_UNITS.

    Returns:
        float, converted value.
    """
    unit_key = quantity_type + "_unit"
    #need to add power for converting watt to btu/hr
    src_units = {
        "dist": "m",
        "elev": "m",
        "P":    "Pa",
        "T":    "K",
        "v":    "m/s",
    }
    src = src_units[quantity_type]
    dst = unit_spec[unit_key]
    if src == dst:
        return value_si
    return ureg.Quantity(value_si, src).to(dst).magnitude


def compressible_hydraulics_from_csv(
    csv_path,
    output_csv_path,
    abstract_state,
    P_in,
    T_in,
    mdot,
    roughness,
    D_h = None,
    flow_area = None,
    isothermal = True,
    q_wall=0.0,
    output_units="US_Common",
):
    """Run compressible hydraulics slice-by-slice along a profile loaded from
    a CSV file and write the per-point results to an output CSV.

    The inlet CSV is expected to have the same three-column format used by
    Line_Segment.from_csv():
        col 0 - ignored (layer ID or similar label)
        col 1 - along-pipe distance from origin [m]
        col 2 - elevation [m]
        col 3 - hydraulic diameter [m] (optional - if D_h is supplied to the function, ignore column in file and use the supplied D_h for the entire length)
        col 4 - flow area [m^2] (optional - if flow_area is supplied to the function, ignore column in file and use the supplied flow_area for the entire length)
        col 5 - Surrounding ground temperature for performing heat transfer calculation TODO not yet implemented
    A header row is expected and skipped.

    One output row is written per profile point (including the inlet as
    point 0).  The inlet row includes distance, elevation, P, T, v, and Ma
    computed from the supplied inlet conditions; the q_wall_W column is left
    blank for point 0 because no slice has yet been traversed.

    Args:
        csv_path          : str, path to the elevation profile CSV.
        output_csv_path   : str, path for the output results CSV.
        abstract_state    : CoolProp AbstractState instance pre-configured for
                            the working fluid.  Updated in-place on each call
                            to compressible_hydraulics(); must not be shared
                            across threads.
        P_in              : float, inlet pressure [Pa].
        T_in              : float, inlet temperature [K].
        mdot              : float, mass flow rate [kg/s].
        D_h               : float, hydraulic diameter [m].
        roughness         : float, absolute pipe-wall roughness [m].
        flow_area         : float, cross-section flow area [m^2].
        isothermal        : bool, if True run isothermal; if False uses q_wall.  Default True.
        q_wall            : float, heat flow into fluid [W].  Only used when
                            isothermal=False.  Default 0.0 (adiabatic).
        output_units      : str, one of 'US_Common' (ft / psia / degF),
                            'SI' (m / Pa / K), or 'metric' (m / kPa / degC).
                            Default 'US_Common'.

    Returns:
        None.  Results are written to output_csv_path.

    Raises:
        ValueError   : unrecognized output_units string, empty CSV, or mdot
                       / geometry issues propagated from compressible_hydraulics.
        RuntimeError : two-phase inlet, choked flow, or CoolProp failure
                       propagated from compressible_hydraulics.
    """
    if output_units not in _COMP_OUTPUT_UNITS:
        raise ValueError(
            f"compressible_hydraulics_from_csv: unrecognized output_units "
            f"'{output_units}'.  Choose from: "
            f"{list(_COMP_OUTPUT_UNITS.keys())}."
        )
    uspec = _COMP_OUTPUT_UNITS[output_units]

    # ------------------------------------------------------------------
    # Load elevation profile from CSV (same format as Line_Segment.from_csv)
    # ------------------------------------------------------------------
    profile = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                      # skip header row
        for row in reader:
            dist_m = float(row[1])
            elev_m = float(row[2])

            if D_h is None:
                D_h_m = float(row[3])
            else:
                D_h_m = D_h

            if flow_area is None:
                flow_area_m = float(row[4])
            else:
                flow_area_m = flow_area
            #T_K = float(row[5]) #TODO for implementing heat transfer
            profile.append((dist_m, elev_m, D_h_m,flow_area_m))

    if not profile:
        raise ValueError(
            f"compressible_hydraulics_from_csv: CSV '{csv_path}' contains no "
            f"data rows."
        )

    # Sort by distance and prepend a zero-distance point if needed,
    # mirroring Line_Segment._normalize_profile().
    profile.sort(key=lambda r: r[0])
    if profile[0][0] != 0.0:
        profile = [(0.0, profile[0][1], profile[0][2], profile[0][3])] + profile

    n_points = len(profile)
    print(
        f"  Profile loaded from '{csv_path}': {n_points} points, "
        f"total distance = {profile[-1][0]:.2f} m, "
        f"net elevation change = {profile[-1][1] - profile[0][1]:.4f} m"
    )

    # ------------------------------------------------------------------
    # Compute inlet velocity and Mach number from inlet conditions.
    # abstract_state is updated in-place here; compressible_hydraulics()
    # will update it again on the first slice call.
    # ------------------------------------------------------------------
    AS = abstract_state
    AS.update(CP.PT_INPUTS, P_in, T_in)

    phase_in = AS.phase()
    if phase_in == _CP_PHASE_TWOPHASE:
        raise RuntimeError(
            f"compressible_hydraulics_from_csv: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase hydraulics only."
        )

    rho_in = AS.rhomass()          # kg/m^3
    a_in   = AS.speed_sound()      # m/s
    v_in   = mdot / (rho_in * profile[0][3])   # m/s  -- use inlet row flow area
    Ma_in  = v_in / a_in
    S_in = AS.smass()

    # ------------------------------------------------------------------
    # Open output CSV and write header + inlet row (point 0)
    # ------------------------------------------------------------------
    with open(output_csv_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)

        writer.writerow([
            "point",
            uspec["dist_label"],
            uspec["elev_label"],
            uspec["P_label"],
            uspec["T_label"],
            uspec["v_label"],
            "Ma",
            uspec["q_wall_label"],
            "Spec. Entropy"
        ])

        dist_0, elev_0, _, _ = profile[0]
        writer.writerow([
            0,
            f"{_convert_comp_output(dist_0, 'dist', uspec):.4f}",
            f"{_convert_comp_output(elev_0, 'elev', uspec):.4f}",
            f"{_convert_comp_output(P_in,   'P',    uspec):.4f}",
            f"{_convert_comp_output(T_in,   'T',    uspec):.4f}",
            f"{_convert_comp_output(v_in,   'v',    uspec):.4f}",
            f"{Ma_in:.6f}",
            "",                           # q_wall blank for inlet row
            f"{S_in:.6f}",
        ])

        # ------------------------------------------------------------------
        # Step through consecutive profile point pairs
        # ------------------------------------------------------------------
        P_cur = P_in
        T_cur = T_in

        for i in range(n_points - 1):
            dist_in,  elev_in,  D_h_in,  area_in  = profile[i]
            dist_out, elev_out, D_h_out, area_out = profile[i + 1]

            slice_dL = dist_out - dist_in   # m, along-pipe length of this slice
            slice_dz = elev_out - elev_in   # m, elevation rise (positive = uphill)
            print(f"Step: {i} of {n_points}", end="\r")

            result = compressible_hydraulics(
                abstract_state=AS,
                P_in=P_cur,
                T_in=T_cur,
                mdot=mdot,
                dL=slice_dL,
                dz=slice_dz,
                D_h=D_h_in,        # hydraulic diameter at slice inlet. 
                roughness=roughness,
                flow_area=area_in,  # flow area at slice inlet. Note that this program assumes perfect pressure/velocity recovery when the flow area changes between slices (no discharge coefficient)
                isothermal=isothermal,
                q_wall=q_wall,
            )
            # S_cur = S_out
            P_cur = result["P_out"]
            T_cur = result["T_out"]

            writer.writerow([
                i + 1,
                f"{_convert_comp_output(dist_out,          'dist', uspec):.4f}",
                f"{_convert_comp_output(elev_out,          'elev', uspec):.4f}",
                f"{_convert_comp_output(P_cur,             'P',    uspec):.4f}",
                f"{_convert_comp_output(T_cur,             'T',    uspec):.4f}",
                f"{_convert_comp_output(result['v_out'],   'v',    uspec):.4f}",
                f"{result['Ma_out']:.6f}",
                f"{result['q_wall_actual']:.4f}",
                f"{result['S_out']:.6f}",
            ])

    print(f"  Compressible profile results exported to: {output_csv_path}")


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

    # output_csv = os.path.splitext("simple.csv")[0] + "_pressure_profile.csv"
    # export_pressure_profile(results, output_csv)

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
    AS_g = AbstractState("HEOS", "Air")
    AS_g.update(CP.PT_INPUTS, P_gas, T_gas)
    rho_gas = AS_g.rhomass()
    S_in = AS_g.smass()
    Q_scfd   = ureg.Quantity(1.0, "mmscf/day")
    mdot     = Q_scfd.to("mol/s").magnitude * AS_g.molar_mass()   # kg/s from mol wt
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
    slicecount = 100
    total_length      = ureg.Quantity(2, "miles").to("m").magnitude
    total_elev_change = ureg.Quantity(2, "miles").to("m").magnitude
    D_pipe  = ureg.Quantity(1.995,  "inch").to("m").magnitude
    eps_pipe = ureg.Quantity(0.00015, "ft").to("m").magnitude
    Q_scfd  = ureg.Quantity(3, "mmscf/day")

    
    # Initial conditions.
    initial_pressure = ureg.Quantity(2000.0, "psi")
    P_in = initial_pressure.to("Pa").magnitude   # Pa
    T_in = 425      # K

    isothermal=False

    AS_g = define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )

    dL = total_length      / slicecount
    dz = total_elev_change / slicecount

    # Generate profile assuming uniform slices of length and elevation change.
    profile = []
    for i in range(slicecount + 1):
        profile.append([i * dL, i * dz])

    segment = Line_Segment(
        roughness=eps_pipe,
        id_val=D_pipe,
        profile=profile,
    )

    A_pipe = math.pi * D_pipe**2 / 4.0


    AS_g.update(CP.PT_INPUTS, P_in, T_in)

    mdot = Q_scfd.to("mol/s").magnitude * AS_g.molar_mass()   # kg/s

    # Compute inlet velocity and Mach number for the header row.
    rho_in = AS_g.rhomass()                    # kg/m^3
    a_in   = AS_g.speed_sound()                # m/s
    v_in   = mdot / (rho_in * A_pipe)          # m/s
    Ma_in  = v_in / a_in

    # Column widths for formatted table output.
    col_w = 14

    header = (
        f"{'Point':>{col_w}}"
        f"{'Dist (ft)':>{col_w}}"
        f"{'Elev (ft)':>{col_w}}"
        f"{'P (psia)':>{col_w}}"
        f"{'T (degF)':>{col_w}}"
        f"{'v (ft/s)':>{col_w}}"
        f"{'Ma':>{col_w}}"
        f"{'q_wall (W)':>{col_w}}"
    )
    sep = "-" * len(header)

    print("\n=== Compressible Isothermal Hydraulics -- Slice-by-Slice Results ===")
    print(f"  Pipe ID: {ureg.Quantity(D_pipe, 'm').to('inch'):.4f~P}")
    print(f"  Flow   : {Q_scfd:.4f~P}  =  {mdot:.4f} kg/s")
    print(f"  Slices : {slicecount}")
    print()
    print(header)
    print(sep)

    # Convert helper: Pa -> psia, K -> degF, m -> ft.
    def pa_to_psia(p):
        return ureg.Quantity(p, "Pa").to("psi").magnitude

    def k_to_degf(t):
        return ureg.Quantity(t, "K").to("degF").magnitude

    def m_to_ft(x):
        return ureg.Quantity(x, "m").to("ft").magnitude

    def ms_to_fts(v):
        return ureg.Quantity(v, "m/s").to("ft/s").magnitude

    # Print the inlet (point 0) row -- no q_wall yet, so leave that column blank.
    dist_0, elev_0 = segment.profile[0]
    print(
        f"{'0':>{col_w}}"
        f"{m_to_ft(dist_0):>{col_w}.3f}"
        f"{m_to_ft(elev_0):>{col_w}.3f}"
        f"{pa_to_psia(P_in):>{col_w}.3f}"
        f"{k_to_degf(T_in):>{col_w}.3f}"
        f"{ms_to_fts(v_in):>{col_w}.4f}"
        f"{Ma_in:>{col_w}.6f}"
        f"{'--':>{col_w}}"
    )

    # Iterate through consecutive profile point pairs (one slice per pair).
    P_cur = P_in
    T_cur = T_in

    for i in range(len(segment.profile) - 1):
        dist_in,  elev_in  = segment.profile[i]
        dist_out, elev_out = segment.profile[i + 1]

        slice_dL = dist_out - dist_in   # m, along-pipe length of this slice
        slice_dz = elev_out - elev_in   # m, elevation rise over this slice

        result = compressible_hydraulics(
            abstract_state=AS_g,
            P_in=P_cur,
            T_in=T_cur,
            mdot=mdot,
            dL=slice_dL,
            dz=slice_dz,
            D_h=D_pipe,
            roughness=eps_pipe,
            flow_area=A_pipe,
            isothermal=isothermal,
        )

        P_cur = result["P_out"]
        T_cur = result["T_out"]

        print(
            f"{i + 1:>{col_w}}"
            f"{m_to_ft(dist_out):>{col_w}.3f}"
            f"{m_to_ft(elev_out):>{col_w}.3f}"
            f"{pa_to_psia(P_cur):>{col_w}.3f}"
            f"{k_to_degf(T_cur):>{col_w}.3f}"
            f"{ms_to_fts(result['v_out']):>{col_w}.4f}"
            f"{result['Ma_out']:>{col_w}.6f}"
            f"{result['q_wall_actual']:>{col_w}.3f}"
        )

    print(sep)
    print()



def test_comp_csv_profile():
    """Exercise compressible_hydraulics_from_csv() using testprofile1.csv."""

    # Pipe geometry
    # D_pipe   = ureg.Quantity(1.995, "inch").to("m").magnitude   # m
    eps_pipe = ureg.Quantity(0.00015, "ft").to("m").magnitude   # m
    # A_pipe   = math.pi * D_pipe**2 / 4.0                        # m^2

    # Inlet conditions.
    P_in = ureg.Quantity(7300.0, "psi").to("Pa").magnitude   # Pa
    T_in = ureg.Quantity(338.0, "degF").to("degK").magnitude                                            # K

    # Flow rate
    Q_scfd = ureg.Quantity(193, "mmscf/day")

    AS = define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    mdot = Q_scfd.to("mol/s").magnitude * AS.molar_mass()          # kg/s

    compressible_hydraulics_from_csv(
        csv_path="Example_Well_Survey.csv",
        output_csv_path="EWS_comp_results.csv",
        abstract_state=AS,
        P_in=P_in,
        T_in=T_in,
        mdot=mdot,
        # D_h=D_pipe,
        roughness=eps_pipe,
        # flow_area=A_pipe,
        isothermal = False,
        output_units="US_Common"
    )

def define_combination(
    AS_gas    = None,
    AS_oil    = None,
    AS_water  = None,
    gas_rate  = None,
    oil_rate  = None,
    water_rate= None,
    eos       = "HEOS",
):
    """Combine up to three single-phase AbstractState streams into one mixed
    AbstractState whose mole fractions reflect the weighted average composition
    of all active streams.

    Each stream is described by a CoolProp AbstractState (already configured
    with mole fractions via set_mole_fractions()) and a total molar flow rate
    in mol (or mol/s, or any consistent molar quantity -- the units cancel
    during normalization, so only the ratio between rates matters).

    Streams where both the AbstractState and rate are non-None are treated as
    active.  A stream is silently skipped if either its AbstractState or its
    rate is None.  At least one stream must be active.

    Components that appear in more than one stream (e.g., n-Butane in both gas
    and oil) are merged by summing their molar contributions before
    normalization.

    Args:
        AS_gas     : CoolProp AbstractState for the gas stream, or None.
        AS_oil     : CoolProp AbstractState for the liquid hydrocarbon stream,
                     or None.
        AS_water   : CoolProp AbstractState for the water stream, or None.
        gas_rate   : float, total moles of gas stream.  Ignored if AS_gas is
                     None.
        oil_rate   : float, total moles of oil stream.  Ignored if AS_oil is
                     None.
        water_rate : float, total moles of water stream.  Ignored if AS_water
                     is None.
        eos        : str, CoolProp equation-of-state backend to use for the
                     combined AbstractState (default 'HEOS').

    Returns:
        CoolProp AbstractState configured with the merged mole fractions.
        Note: the returned AbstractState has NOT been updated to a (P, T) state
        yet -- call AS.update(CP.PT_INPUTS, P, T) before querying properties.

    Raises:
        ValueError : if no active streams are present, or if any active stream
                     has a non-positive rate.
    """
    # Build a list of (AbstractState, rate) pairs for active streams only.
    streams = []
    for label, AS, rate in [
        ("gas",   AS_gas,   gas_rate),
        ("oil",   AS_oil,   oil_rate),
        ("water", AS_water, water_rate),
    ]:
        if AS is None or rate is None:
            continue  # silently skip absent streams
        if rate <= 0.0:
            raise ValueError(
                f"define_combination: {label}_rate must be positive "
                f"(received {rate})."
            )
        streams.append((AS, rate))

    if not streams:
        raise ValueError(
            "define_combination: at least one stream (gas, oil, or water) "
            "must be provided with a non-None AbstractState and rate."
        )

    # Accumulate molar contributions from each stream into a dict keyed by
    # CoolProp component name (hyphenated, e.g. 'n-Butane').
    # contribution[name] = sum over streams of (stream_mole_fraction * stream_rate)
    contribution = {}
    for AS, rate in streams:
        names     = AS.fluid_names()          # list of str, e.g. ['Methane', 'n-Butane']
        fractions = AS.get_mole_fractions()   # list of float, same length

        for name, frac in zip(names, fractions):
            contribution[name] = contribution.get(name, 0.0) + frac * rate

    # Normalize so that mole fractions sum to exactly 1.0.
    total_moles = sum(contribution.values())
    combined_names     = list(contribution.keys())
    combined_fractions = [contribution[n] / total_moles for n in combined_names]

    # Build and return the combined AbstractState.
    fluid_string = "&".join(combined_names)
    AS_combined  = AbstractState(eos, fluid_string)
    AS_combined.set_mole_fractions(combined_fractions)

    return AS_combined


def test_define_combination():
    AS_gas = define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    AS_oil = define_composition(
        y_n_Butane        = 0.02,
        y_IsoButane       = 0.0,
        y_n_Pentane       = 0.05,
        y_Isopentane      = 0.0,
        y_n_Hexane        = 0.08,
        y_n_Heptane       = 0.2,
        y_n_Octane        = 0.3,
        y_n_Nonane        = 0.25,
        y_n_Decane        = 0.1,
        eos = "HEOS"
    )
    AS_water = AbstractState("HEOS", "Water")
    AS_gas.update(CP.PT_INPUTS, 101325, 288.7)
    AS_oil.update(CP.PT_INPUTS, 101325, 288.7)
    AS_water.update(CP.PT_INPUTS, 101325, 288.7)

    gas_rate   = ureg.Quantity(3813,  "mscf").to("mol").magnitude
    oil_rate   = ureg.Quantity(148.0, "oil_bbl").to("m^3").magnitude * AS_oil.rhomolar()
    water_rate = ureg.Quantity(119.0, "oil_bbl").to("m^3").magnitude * AS_water.rhomolar()

    AS_combined = define_combination(
        AS_gas    = AS_gas,
        AS_oil    = AS_oil,
        AS_water  = AS_water,
        gas_rate  = gas_rate,
        oil_rate  = oil_rate,
        water_rate= water_rate,
        eos       = "HEOS",
    )

    AS_combined.update(CP.PT_INPUTS, 1*101325, 300)

    print(AS_combined.Q())




def define_composition(
    # --- USER INPUTS (Mole Fractions) ---
    y_Methane         = 0.0,
    y_Ethane          = 0.0,
    y_Propane         = 0.0,
    y_n_Butane        = 0.0,
    y_IsoButane       = 0.0,
    y_n_Pentane       = 0.0,
    y_Isopentane      = 0.0,
    y_n_Hexane        = 0.0,
    y_n_Heptane       = 0.0,
    y_n_Octane        = 0.0,
    y_n_Nonane        = 0.0,
    y_n_Decane        = 0.0,
    y_CarbonDioxide   = 0.0,
    y_Water           = 0.0,
    y_Nitrogen        = 0.0,
    y_Oxygen          = 0.0,
    y_Argon           = 0.0,
    y_Hydrogen        = 0.0,
    y_HydrogenSulfide = 0.0,
    eos = "HEOS"              #equation of state. HEOS is CoolProp's default Helmholz equation of state. Can also use Peng Robinson (PR) which is faster, although it doesn't allow the calculation of viscosity.
    ):
    # ------------------------------------

    # The list of suffixes based on CoolProp's registry names
    components = [
        "Methane", "Ethane", "Propane", "n_Butane", "IsoButane",
        "n_Pentane", "Isopentane", "n_Hexane", "n_Heptane", "n_Octane",
        "n_Nonane", "n_Decane", "CarbonDioxide", "Water", "Nitrogen",
        "Oxygen", "Argon", "Hydrogen", "HydrogenSulfide"
    ]

    active_cp_names = []
    fractions = []

    for comp in components:
        val = locals().get(f"y_{comp}", 0.0)
        if val > 0:
            # CoolProp uses hyphens (n-Butane) while Python uses underscores (n_Butane)
            cp_ready_name = comp.replace("_", "-")
            active_cp_names.append(cp_ready_name)
            fractions.append(val)

    # Normalize to ensure the sum is exactly 1.0 (prevents CoolProp errors)
    total = sum(fractions)
    fractions = [f / total for f in fractions]

    # Generate State
    fluid_string = "&".join(active_cp_names)

    #Set abstract state model. HEOS is CoolProp's default Helmholz equation of state. Can also use Peng Robinson, although it doesn't allow the calculation of viscosity.
    AS = AbstractState(eos, fluid_string)
    AS.set_mole_fractions(fractions)

    return AS

if __name__ == "__main__":

    #test_p2p()
    # test_comp_hydraulics()
    # test_compressible_slices()
    test_comp_csv_profile()
    # test_define_combination()
