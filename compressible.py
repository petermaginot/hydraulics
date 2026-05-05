import csv
import math
import os
import warnings
from pint import UnitRegistry
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
)


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



class Line_Segment(Base_Line_Segment):
    """Pipe segment with compressible-flow pressure and temperature calculation.

    Inherits geometry storage, CSV loading, and convenience properties from
    Base_Line_Segment.  Adds dP_dT() for compressible flow, stepping through
    consecutive profile slices via compressible_hydraulics() and applying
    isentropic area-change corrections at inter-slice boundaries.

    Constructor arguments and behavior are identical to Base_Line_Segment.
    See Base_Line_Segment for full argument documentation.
    """

    def dP_dT(
        self,
        abstract_state,
        flow_rate,
        P0,
        T0,
        isothermal=False,
        q_wall=0.0,
    ):
        """Calculate outlet pressure and temperature for compressible flow
        through the segment.

        Steps through consecutive profile point pairs, calling
        compressible_hydraulics() for each slice and applying isentropic
        area-change corrections at boundaries where the flow area changes.

        Heat input q_wall is distributed uniformly per unit pipe length across
        all slices.

        Args:
            abstract_state : CoolProp AbstractState pre-configured for the
                             working fluid.  Updated in-place during
                             integration; must not be shared across threads.
            flow_rate      : pint Quantity -- mass ([mass]/[time]), molar or
                             standard-volume ([substance]/[time], e.g. mol/s,
                             mmscf/day), or actual volumetric
                             ([length]^3/[time]) flow rate.
            P0             : float, inlet static pressure [Pa].
            T0             : float, inlet static temperature [K].
            isothermal     : bool, if True temperature is held constant
                             through each slice.  Default False.
            q_wall         : float, total heat input to the fluid over the
                             entire segment [W].  Distributed uniformly per
                             unit length.  Ignored when isothermal=True.
                             Default 0.0 (adiabatic).

        Returns:
            (P_out, T_out) : tuple of floats [Pa, K].

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
        T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
        _safe_update_PT(AS, P0, T0, T_cric, P_bar, T_c, P_c)
        mdot = _resolve_mdot(flow_rate, AS)

        total_length  = self.total_length_m
        q_per_length  = q_wall / total_length if total_length > 0.0 else 0.0

        _AREA_TOL = 1e-6   # fractional area-change threshold
        n     = len(self.profile)
        P_cur = P0
        T_cur = T0

        for i in range(n - 1):

            dist_in,  elev_in,  D_h_in,  area_in  = self.profile[i]
            dist_out, elev_out, D_h_out, area_out = self.profile[i + 1]

            dL      = dist_out - dist_in   # m, along-pipe slice length
            dz      = elev_out - elev_in   # m, elevation rise (positive = uphill)
            q_slice = q_per_length * dL    # W, heat for this slice

            compressible_hydraulics2(
                abstract_state=AS,
                mdot=mdot,
                dL=dL,
                dz=dz,
                D_h=D_h_in,
                roughness=self.roughness_si,
                flow_area=area_in,
                isothermal=isothermal,
                q_wall=q_slice,
                T_cricondentherm=T_cric,
                P_cricondenbar=P_bar,
                T_critical=T_c,
                P_critical=P_c,
            )

            P_cur   = AS.p()
            T_cur   = AS.T()
            rho_cur = AS.rhomass()
            a_cur   = AS.speed_sound()
            v_cur   = mdot / (rho_cur * area_in)
            Ma_cur  = v_cur / a_cur

            # Area-change correction at the boundary to the next slice.
            area_ratio = abs(area_out - area_in) / max(area_in, area_out)
            if area_ratio > _AREA_TOL:
                AS = compressible_changing_area_K(
                    AS, mdot, area_in, area_out, K=0.0
                ) #is it necessary to set AS = compressible_changing_area_K since the abstract state is updated in place?
            print(f" Step: {i+1} of {n-1}, P = {P_cur}, T = {T_cur}")
            # print(f" Step: {i+1} of {n-1}, P = {P_cur}, T = {T_cur}", end="\r")               
        print(f"")
        return P_cur, T_cur


class Bend(Base_Bend):
    """Rounded pipe bend fitting with compressible pressure/temperature
    calculation.

    Modeled as adiabatic.  Inherits geometry storage and validation from
    Base_Bend.  The dP_dT() method is not yet implemented.

    Constructor arguments are identical to Base_Bend:
        Di        : pint Quantity or float (m if float).  Pipe inner diameter.
        ang_deg   : float.  Bend angle [degrees].
        bend_dias : float.  Bend radius as a multiple of Di.
    """

    def dP_dT(self, abstract_state, flow_rate, P0, T0):
        """Outlet pressure and temperature through the bend [Pa, K].

        Args:
            abstract_state : CoolProp AbstractState instance.
            flow_rate      : pint Quantity -- mass, molar, or volumetric flow.
            P0             : float, inlet static pressure [Pa].
            T0             : float, inlet static temperature [K].

        Returns:
            (P_out, T_out) : tuple of floats [Pa, K].

        Raises:
            NotImplementedError : always (not yet implemented).
        """
        raise NotImplementedError(
            "Bend.dP_dT is not yet implemented for compressible flow."
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

    def dP_dT(self, abstract_state, flow_rate):
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

        Returns:
            abstract_state updated to outlet (P, T) conditions.
        """
        AS   = abstract_state
        mdot = _resolve_mdot(flow_rate, AS)

        Di_US = self.Di_US_si
        Di_DS = self.Di_DS_si

        if abs(Di_US - Di_DS) < 1e-12:
            return AS

        A_US = math.pi * Di_US ** 2 / 4.0
        A_DS = math.pi * Di_DS ** 2 / 4.0

        if Di_US > Di_DS:
            # Contraction: fluids returns K w.r.t. downstream; convert to upstream.
            K_ds = fluids.fittings.contraction_sharp(Di1=Di_US, Di2=Di_DS)
            K    = K_ds * (A_DS / A_US) ** 2
        else:
            # Expansion: fluids returns K w.r.t. upstream velocity directly.
            K = fluids.fittings.diffuser_sharp(Di1=Di_US, Di2=Di_DS)

        return compressible_changing_area_K(AS, mdot, A_US, A_DS, K)


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


def _build_phase_limits(AS):
    """Return (T_cricondentherm, P_cricondenbar, T_critical, P_critical) [K, Pa].

    Builds the phase envelope on a temporary AS so the working AS's internal
    solver state is not corrupted.  CoolProp's build_phase_envelope leaves the
    AbstractState at the last envelope point it visited; subsequent update()
    calls on the same object then fail unpredictably.

    Returns (None, None, None, None) if the envelope cannot be built.
    """
    try:
        AS_tmp = AbstractState("HEOS", "&".join(AS.fluid_names()))
        AS_tmp.set_mole_fractions(list(AS.get_mole_fractions()))
        AS_tmp.build_phase_envelope("")
        PE = AS_tmp.get_phase_envelope_data()
        return max(PE.T), max(PE.p), AS_tmp.T_critical(), AS_tmp.p_critical()
    except Exception:
        return None, None, None, None


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
      otherwise                             → no hint; CoolProp determines phase
    """
    phase = None
    if T_cricondentherm is not None and T > T_cricondentherm:
        if P_critical is not None and P > P_critical:
            phase = CP.iphase_supercritical
        else:
            phase = CP.iphase_supercritical_gas
    elif P_cricondenbar is not None and P > P_cricondenbar:
        phase = CP.iphase_supercritical

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
    AbstractState at inlet conditions.

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
    # NASA Glenn Eq #9.
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


def compressible_changing_area_K(abstract_state, mdot, A_in, A_out, K):
    """Outlet pressure and temperature for a compressible fluid passing through
    an area change with a known loss coefficient K applied to inlet velocity.

    The caller must update abstract_state to the inlet (P, T) conditions before
    calling.  The abstract state is updated before return - the calling function can 
    retrieve the updated pressure and temperature from the returned abstract state.

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
    initial guess.

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
        (P_out, T_out) -- tuple of floats [Pa, K].

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
        AS.update(CP.PT_INPUTS, P, T)
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
    AS.update(CP.PT_INPUTS, P_out, T_out)
    return AS


def compressible_hydraulics2(
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
    ):
    """Calculate compressible pipe-flow hydraulics over a single pipe slice
    using either the Euler method

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
            f"compressible_hydraulics2: invalid parameter values — all must be "
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
            f"compressible_hydraulics: fluid is two-phase at inlet "
            f"(P={P_in:.4g} Pa, T={T_in:.4g} K).  Single-phase hydraulics "
            f"only.  Consider checking your inlet conditions."
        )

    rho_in = AS.rhomass()       # kg/m^3
    Cp  = AS.cpmass()        # J/(kg*K)

    if mu is None:
        try:
            mu = AS.viscosity()   # Pa*s
        except:
            # Fall back to Lee-Gonzalez-Eakin if the EOS (e.g. Peng-Robinson)
            # does not support viscosity.  LGE is for hydrocarbon gases only;
            # supply mu explicitly for other fluids.
            mu = viscosity_LGE(T_in, AS.molar_mass() * 1000.0, rho_in)
    S_in = AS.smass()   # J/(kg*K) #NOTE not used, may be able to delete
    H_in = AS.hmass()   # J/kg
    a   = AS.speed_sound()   # m/s  (isentropic speed of sound)
    v_in  = mdot / (rho_in * flow_area)   # m/s
    Ma = v_in / a                       # Mach number

    if Ma >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_hydraulics: inlet Mach number Ma={Ma:.4f} "
            f"is sonic or near-sonic.  Reduce flow rate or check geometry."
        )

    # Friction factor and Reynolds number at inlet conditions.
    Re = fluids_Reynolds(V=v_in, D=D_h, rho=rho_in, mu=mu)
    f_darcy = fluids_friction_factor(Re=Re, eD=roughness / D_h)

    if not isothermal:
        #We will first calculate dP for a known dL, using fluid properties at the inlet conditions and assuming they don't enough over the length slice to affect the calculation.
        #From the energy balance, dq = mdot * dH + mdot/2 * d(v^2)/2 + mdot * g * dz
        #From the continuity equation mdot = rho * flow_area * v
        #Combine these two, take the derivative with respect to length, and do some rearranging (substituting dv^2 = 2vdv and dv = -v/rho * drho)
        # 1/mdot * dq/dL = dH/dL - v^2/rho * drho/dL + g * dz/dL
        # We will need to use an equation of state to relate pressure and density. If rho = f(H, P):
        # drho/dL = (∂rho/∂P)_H * dP/dL + (∂rho/∂H)_P * dH/dL [chain rule for partial derivatives]
        # We'll call (∂rho/∂P)_H  = A and (∂rho/∂H)_P = B and assume they are relatively constant over the length slice.

        #Accounting for entropy, from "Fundamentals of Gas Dynamics, 2nd Ed." by Zucker and Biblarz", equation 3.1
        # dS = dSe + dSi
        #where dSe = entropy change due to heat transfer = 1/mdot * dq/T
        # and dSi = entropy change due to friction = 1/T * K * v^2/2
        # and K = f * dL/D_h, where f is Darcy friciton factor and D_h is the hydraulic diameter (or simply the diameter for a round pipe)

        #We can use a thermodynamic identity to relate enthalpy, entropy, and pressure:
        # dH = T*dS + dP/rho (from table 6.2-1 in "Chemical, Biochemical, and Engineering Thermodynmics, 4th ed." by Sandler)
        #Substituting for dS and taking derivative with respect to length
        # dH/dL = 1/mdot * dq/dL + f * v^2 / (2 * D_h) + 1/rho * dP/dL

        #Then, taking the dH/dL from the partial derivatives chain rule above and plugging it in to eliminate drho/dL and eliminating dH/dL with the two equations for DH/dL,
        # after MUCH REARRANGING, you get:
        # dP/dL = (f * rho * v^2/(2*D_h) * (1-v^2 * B / rho) + rho * g * dz/dL - v^2 * B/mdot * dq/dL)/(v^2*A + v^2 * B/rho - 1)
        # where     ^friction contribution                     ^elevation change contrib      ^heat transfer contribution
        
        #We can use the Euler method to estimate the pressure at the end of a length slice dL

        #First, calculate those oddball partial derivatives
        A = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass) 
        B = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        #Now, calculate each contributing component of the dP/dL.
        dP_dL_denominator_factor = v_in**2*A + v_in**2*B/rho_in - 1
        dP_dL_friction = (f_darcy*rho_in*v_in**2/(2*D_h)*(1-v_in**2*B/rho_in))/dP_dL_denominator_factor
        dP_dL_gravity = (rho_in * grav_constant * dz/dL)/dP_dL_denominator_factor
        dP_dL_heatxfr = (-v_in**2*B*q_wall/(mdot * dL))/dP_dL_denominator_factor

        # print(f'friction contrib: {dP_dL_friction}')
        # print(f'gravity contrib:{dP_dL_gravity}')
        # print(f'heat contrib: {dP_dL_heatxfr}')

        dP = dL * (dP_dL_friction+ dP_dL_gravity + dP_dL_heatxfr)

        P_out = P_in + dP

        #Ideally we would just calculate the output entropy and be good to go with that as our second state variable, and this works fine for single component systems. 
        # change in entropy = change in entropy due to heat transfer (q_wall/(mdot * T)) + change in entropy due to friction (f_darcy * v^2/(2*D_h * Tin)) 
        # S_out = S_in + dS
        #   (From chapter 3 of "Fundamentals of Gas Dynamics, 2nd Ed." by Zucker and Biblarz". See equations 3.1 and 3.64)

        # However, with multicomponent systems, CoolProp has trouble using entropy as one of the inputs, so for robustness we need to use a thermodynamic identity to convert it to something easier to deal with, like temperature.
        # Use dH = TdS + VdP = Cp dT + [V - T (∂V/∂T)_P] dP (from table 6.2-1 in "Chemical, Biochemical, and Engineering Thermodynmics, 4th ed." by Sandler)
        #                                   ^ [V - T (∂V/∂T)_P] is the Joule Thompson coefficient μ times negative heat capacity [-μCp]
        #substitute (∂V/∂T)_P = -1/ρ^2 (∂ρ/∂T)_P and solve for dT
        # Then, use the Euler method to calculate T_out.
        drhodT_P = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        dT = (1.0 / Cp) * (
            q_wall / mdot - (T_in / rho_in**2) * drhodT_P * dP + f_darcy * v_in**2 / (2.0 * D_h) * dL
        )
        T_out = T_in + dT
        _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)

        #We do however know the total (stagnation) enthalpy from our energy balance. Compare H(P, T) to what we would expect from an energy balance.
        H_total_out = H_in + q_wall/mdot + v_in**2/2 - grav_constant*dz

        #And we can compare it to what would be given by our calculated temperature
        rho_out_calc = AS.rhomass()
        v_out_calc = mdot / (flow_area * rho_out_calc)
        H_total_calc = AS.hmass() + v_out_calc**2/2
        energy_error = H_total_out - H_total_calc
        #We can figure out about how much we need to adjust our temperature to make our energy balance for a second guess.
        #Differentiating the energy error:
        # d(error)/dT = (∂H/∂T)_P + (1/2) * (∂v/∂T)_P
        #Use the continuity to convert v to rho
        #d(error)/dT = Cp - mdot/(A*rho^2) * (∂ρ/∂T)_P <-- (that same isobaric compressibilityish factor from earlier)
        # Use this to nudge T back closer to what it should be and update the abstract state
        Cp = AS.cpmass()
        drhodT_P = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)
        T_out = T_out + energy_error / (Cp - (mdot/(flow_area*rho_out_calc))*drhodT_P)

        #compute again
        _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)
        rho_out_calc = AS.rhomass()
        v_out_calc = mdot / (flow_area * rho_out_calc)
        H_total_calc = AS.hmass() + v_out_calc**2/2
        energy_error2 = H_total_out - H_total_calc

        # print(f'Energy error 1: {energy_error}')
        # print(f'Energy error 2: {energy_error2}')
        #If the energy error is still significant at this point, you really need a smaller length slice
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
        #Substitute this back in and solve for dP/dL
        # dP/dL = (rho * f * v^2 / (2 * D_h) - rho * g * dz/dL) / (1 - v^2 * (∂ρ/∂P)_T)
        # We can now Euler method a dP/dL step
        drhodP_T = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)

        dP = (-rho_in * f_darcy*v_in**2*dL/(2*D_h) - rho_in * grav_constant * dz)/(1- v_in**2 * drhodP_T)
        P_out = P_in + dP
        T_out = T_in
        _safe_update_PT(AS, P_out, T_out, T_cricondentherm, P_cricondenbar, T_critical, P_critical)

    #Check outlet Mach number
    rho_out = AS.rhomass()   
    a_out   = AS.speed_sound()   # m/s  (isentropic speed of sound)
    v_out  = mdot / (rho_out * flow_area)   # m/s
    Ma_out = v_out / a_out                       # Mach number

    if Ma_out >= choke_mach_limit:
        raise RuntimeError(
            f"compressible_hydraulics: outlet Mach number Ma={Ma_out:.4f} "
            f"is sonic or near-sonic.  Reduce flow rate or check geometry."
        )

    return AS


def test_comp_hydraulics():

    P_gas    = ureg.Quantity(1000, "psi").to("Pa").magnitude    # Pa
    T_gas    = 300.0    # K
    D_gas    = ureg.Quantity(4.026, "inch").to("m").magnitude
    A_gas    = math.pi * D_gas**2 / 4.0
    eps_gas  = ureg.Quantity(0.00015, "ft").to("m").magnitude
    dL_gas   = ureg.Quantity(40.0,    "feet").to("m").magnitude
    dz_gas   = ureg.Quantity(40.0,    "feet").to("m").magnitude
    AS_g = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    AS_g.update(CP.PT_INPUTS, P_gas, T_gas)
    rho_gas = AS_g.rhomass()
    S_in = AS_g.smass()
    Q_scfd   = ureg.Quantity(200.0, "mmscf/day")
    mdot     = Q_scfd.to("mol/s").magnitude * AS_g.molar_mass()   # kg/s from mol wt
    G_gas    = mdot / A_gas
    a_in   = AS_g.speed_sound()                # m/s
    v_in   = mdot / (rho_gas * A_gas)          # m/s
    Ma_in = v_in/a_in

    print('\n')
    print(f'inputs: P = {P_gas}, Smass= {S_in}, velocity = {v_in}, Mach number = {Ma_in}')
    # print('\n')
    result_g = compressible_hydraulics2(
        abstract_state=AS_g,   # already updated to (P_gas, T_gas) above
        mdot=mdot,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        flow_area=A_gas,
        isothermal=True,
    )
    outlet_P = result_g.p()
    outlet_T = result_g.T()
    rho_out = result_g.rhomass()
    a_out   = result_g.speed_sound()                # m/s
    v_out   = mdot / (rho_out * A_gas)          # m/s
    Ma_out = v_out/a_out
    S_out = result_g.smass()

    # print('\n')
    print('Isothermal case')
    print(f'outputs: P = {outlet_P}, T = {outlet_T}, Smass= {S_out}, velocity = {v_out}, Mach number = {Ma_out}')
    print('\n')

    AS_g.update(CP.PT_INPUTS, P_gas, T_gas) #reinitialize abstract state

    result_g = compressible_hydraulics2(
        abstract_state=AS_g,   # already updated to (P_gas, T_gas) above
        mdot=mdot,
        dL=dL_gas,
        dz=dz_gas,
        D_h=D_gas,
        roughness=eps_gas,
        flow_area=A_gas,
        isothermal=False,
        q_wall = ureg.Quantity(0, "Btu/hr").to("watt").magnitude,
    )
    outlet_P = result_g.p()
    outlet_T = result_g.T()
    rho_out = result_g.rhomass()
    a_out   = result_g.speed_sound()                # m/s
    v_out   = mdot / (rho_out * A_gas)          # m/s
    Ma_out = v_out/a_out
    S_out = result_g.smass()

    print('Adiabadic case')
    print(f'outputs: P = {outlet_P}, T = {outlet_T}, Smass= {S_out}, velocity = {v_out}, Mach number = {Ma_out}')


def test_line_segment_csv():
    csv_path = os.path.join(os.path.dirname(__file__), "Example_Well_Survey.csv")
    roughness = ureg.Quantity(0.00015, "ft")

    seg = Line_Segment.from_csv(csv_path, roughness=roughness)

    P_in = ureg.Quantity(9000, "psi").to("Pa").magnitude
    T_in = 437.0   # K
    Q_scfd = ureg.Quantity(25.6, "mmscf/day")

    AS = composition.define_composition(
        y_Methane = 0.95,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
    )
    AS.update(CP.PT_INPUTS, P_in, T_in)
    rho_in = AS.rhomass()
    area_in = seg.profile[0][3]
    v_in = _resolve_mdot(Q_scfd, AS) / (rho_in * area_in)
    Ma_in = v_in / AS.speed_sound()

    print("\ntest_line_segment_csv")
    print(f"  inlet: P={P_in:.4g} Pa, T={T_in} K, Ma={Ma_in:.4f}")

    P_out, T_out = seg.dP_dT(AS, Q_scfd, P_in, T_in)

    AS.update(CP.PT_INPUTS, P_out, T_out)
    rho_out = AS.rhomass()
    area_out = seg.profile[-1][3]
    v_out = _resolve_mdot(Q_scfd, AS) / (rho_out * area_out)
    Ma_out = v_out / AS.speed_sound()

    P_out_psi = ureg.Quantity(P_out, "Pa").to("psi").magnitude
    dP_psi = ureg.Quantity(P_out - P_in, "Pa").to("psi").magnitude
    print(f"  outlet: P={P_out_psi:.4g} psi, T={T_out:.4g} K, Ma={Ma_out:.4f}")
    print(f"  dP = {dP_psi:.4f} psi")


def test_liq_plot():
    import matplotlib.pyplot as plt

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
    x = []
    y = []
    z = []
    for pt in results["profile_results"]:
        x.append(ureg.Quantity(pt['distance_m'], "m").to("ft").magnitude)
        y.append(ureg.Quantity(pt['dP_total_Pa'], "Pa").to("psi").magnitude)
        z.append(ureg.Quantity(pt['elevation_m'], "m").to("ft").magnitude)
    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'Consolas'
    l1, = ax.plot(x, y, label='Pressure change [psi]', color = 'black')
    ax2 = ax.twinx()
    l2, = ax2.plot(x, z, label='Elevation [ft]', color = 'red')
    ax.set_xlabel('Distance [ft]')  # Add an x-label to the Axes.
    ax.set_ylabel('Pressure change [psi]')  # Add a y-label to the Axes.
    ax2.set_ylabel('Elevation [m]')
    ax2.legend([l1, l2], ['Pressure change [psi]', 'Elevation [m]'])
    plt.show()

def testasdf():
    import matplotlib.pyplot as plt
    AS_g = composition.define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
        )
    try:
        AS_g.build_phase_envelope("dummy")
        PE = AS_g.get_phase_envelope_data()
        plt.plot(PE.T, PE.p, '-', label='Composition')
        plt.xlabel('Temperature [K]')
    except ValueError as VE:
        print(VE)

    plt.ylabel('Pressure [Pa]')
    plt.yscale('log')
    plt.title('Phase Envelope for Selected Composition')
    plt.legend(loc='lower right', shadow=True)
    plt.savefig('methane-ethane.png')

    # AS_g.update(CP.PT_INPUTS, 4.32036e+06, 285.88)
    # phase = AS_g.phase()
    # if phase == _CP_PHASE_TWOPHASE:
    #     print('two phase')
    
def phase_env():
    import matplotlib.pyplot as plt

    HEOS = CP.AbstractState('HEOS', 'Methane&Ethane')

    for x0 in [0.02, 0.2, 0.4, 0.6, 0.8, 0.98]:
        HEOS.set_mole_fractions([x0, 1 - x0])
        try:
            HEOS.build_phase_envelope("dummy")
            PE = HEOS.get_phase_envelope_data()
            PELabel = 'Methane, x = ' + str(x0)
            plt.plot(PE.T, PE.p, '-', label=PELabel)
        except ValueError as VE:
            print(VE)

    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [Pa]')
    plt.yscale('log')
    plt.title('Phase Envelope for Methane/Ethane Mixtures')
    plt.legend(loc='lower right', shadow=True)
    plt.savefig('methane-ethane.pdf')
    plt.savefig('methane-ethane.png')
    plt.close()

def test_twophase():
    AS_g = composition.define_composition(
        y_Methane = 0.95,
        y_Ethane = 0.05,

        eos = "HEOS"
        )
    P0 =6.895e+06
    T0 = 300.0
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS_g)
    # print(f'T cr: {T_cric}, P cr: {P_bar}')
    # _safe_update_PT(AS_g, P0, T0, T_cric, P_bar, T_c, P_c)
    AS_g.update(CP.PT_INPUTS, P0, T0)
    phase = AS_g.phase()
    print(f'Phase: {phase}')
    compressibility_factor = AS_g.compressibility_factor()
    print(f'Z: {compressibility_factor}')

def test_contract_expand():
    D_small = ureg.Quantity(3.826, "in")
    D_large = ureg.Quantity(4.026, "in")
    contraction_test = Contraction_Expansion(Di_US=D_large, Di_DS= D_small)
    expansion_test = Contraction_Expansion(Di_US=D_small, Di_DS= D_large)
    P_in = ureg.Quantity(1000, "psi").to("Pa").magnitude
    T_in = 300.0   # K
    Q_scfd = ureg.Quantity(60, "mmscf/day")
    print(f'Inlet P:{P_in}, T:{T_in}')
    AS = composition.define_composition(
        y_Methane = 0.95,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.02,
        eos = "HEOS"
    )
    AS.update(CP.PT_INPUTS, P_in, T_in)

    AS = contraction_test.dP_dT(AS, Q_scfd)
    print(f'After contraction P:{AS.p()}, T:{AS.T()}')
    AS = expansion_test.dP_dT(AS, Q_scfd)
    print(f'After expansion P:{AS.p()}, T:{AS.T()}')


if __name__ == "__main__":
    test_contract_expand()
    # test_twophase()
    # phase_env()
    # testasdf()
    # test_line_segment_csv()
    #test_p2p()
    # test_comp_hydraulics()
    # test_compressible_slices()
    # test_comp_csv_profile()
    # test_liq_plot()
