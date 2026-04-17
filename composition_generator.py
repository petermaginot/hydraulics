from pint import UnitRegistry
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

    test_define_combination()