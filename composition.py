import csv

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
from component_classes import ureg

# Component names recognised by define_composition / parse_composition_csv.
# Underscored CoolProp aliases (n_Butane -> CoolProp's n-Butane) -- the
# underscore form is what the y_<Component> kwargs use.
KNOWN_COMPONENTS = [
    "Methane", "Ethane", "Propane", "n_Butane", "IsoButane",
    "n_Pentane", "Isopentane", "n_Hexane", "n_Heptane", "n_Octane",
    "n_Nonane", "n_Decane", "Benzene", "CarbonDioxide", "Water", "Nitrogen",
    "Oxygen", "Argon", "Hydrogen", "HydrogenSulfide", "CarbonMonoxide",
]


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
    y_Benzene         = 0.0,
    y_CarbonDioxide   = 0.0,
    y_Water           = 0.0,
    y_Nitrogen        = 0.0,
    y_Oxygen          = 0.0,
    y_Argon           = 0.0,
    y_Hydrogen        = 0.0,
    y_HydrogenSulfide = 0.0,
    y_CarbonMonoxide  = 0.0,
    eos = "HEOS"              #equation of state. HEOS is CoolProp's default Helmholz equation of state. Can also use Peng Robinson (PR) which is faster, although it doesn't allow the calculation of viscosity.
    ):
    # ------------------------------------

    active_cp_names = []
    fractions = []

    for comp in KNOWN_COMPONENTS:
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


def parse_composition_csv(csv_path):
    """Read a composition CSV into a {component: mole_fraction} dict.

    The CSV must have a header row with columns named 'Component' and
    'Mole fraction' (matched case-insensitively).  Blank mole-fraction cells
    are skipped, so a single template CSV listing every supported component
    can be used by leaving the unused rows empty.  Component names must
    match KNOWN_COMPONENTS exactly (underscore form, e.g. 'n_Butane',
    'CarbonDioxide').

    Fractions are NOT normalised here -- the dict is meant to feed
    define_composition (directly or via define_composition_from_csv), which
    normalises internally.

    Args:
        csv_path : path to the composition CSV.

    Returns:
        dict mapping component name to a positive mole fraction (float).

    Raises:
        ValueError : empty file, missing header columns, unknown component,
                     non-numeric fraction, or no rows with a positive value.
    """
    out = {}
    # utf-8-sig transparently strips a UTF-8 BOM, which Excel routinely
    # writes at the start of CSV exports and would otherwise attach to
    # the first header name ('﻿Component' != 'Component').
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Composition CSV '{csv_path}' is empty.")

        # Match the two required headers case-insensitively while keeping
        # the original key (DictReader keys by the exact header text).
        header_map = {h.strip().lower(): h for h in reader.fieldnames if h}
        comp_col = header_map.get("component")
        frac_col = header_map.get("mole fraction") or header_map.get("fraction")
        if comp_col is None or frac_col is None:
            raise ValueError(
                f"Composition CSV '{csv_path}' must have 'Component' and "
                f"'Mole fraction' columns (got {reader.fieldnames})."
            )

        for row in reader:
            name = (row.get(comp_col) or "").strip()
            frac_str = (row.get(frac_col) or "").strip()
            if not name or not frac_str:
                continue
            if name not in KNOWN_COMPONENTS:
                raise ValueError(
                    f"Composition CSV '{csv_path}': unknown component "
                    f"'{name}'.  Valid names: {KNOWN_COMPONENTS}."
                )
            try:
                val = float(frac_str)
            except ValueError as e:
                raise ValueError(
                    f"Composition CSV '{csv_path}': non-numeric mole "
                    f"fraction '{frac_str}' for component '{name}'."
                ) from e
            if val > 0:
                # Duplicate detection mirrors the GUI table backstop: a
                # repeated component is almost always a user mistake, and
                # silently overwriting (or summing) hides it.  Reject so
                # the caller knows to fix the CSV.
                if name in out:
                    raise ValueError(
                        f"Composition CSV '{csv_path}': duplicate component "
                        f"'{name}'.  Combine the rows or remove the redundant "
                        f"entry."
                    )
                out[name] = val

    if not out:
        raise ValueError(
            f"Composition CSV '{csv_path}': no components with a positive "
            f"mole fraction."
        )
    return out


def define_composition_from_csv(csv_path, eos="HEOS"):
    """Build an AbstractState by loading mole fractions from a CSV file.

    Convenience wrapper around parse_composition_csv + define_composition,
    intended for headless / scripting use.  GUI callers typically want to
    use parse_composition_csv directly so the parsed values can be shown
    (and edited) before the AbstractState is built.

    Args:
        csv_path : path to the composition CSV.
        eos      : CoolProp backend (default 'HEOS').

    Returns:
        CoolProp AbstractState with the mole fractions applied.  As with
        define_composition, the state has NOT been updated to a (P, T)
        yet -- call AS.update(CP.PT_INPUTS, P, T) before querying.
    """
    fractions = parse_composition_csv(csv_path)
    kwargs = {f"y_{name}": val for name, val in fractions.items()}
    kwargs["eos"] = eos
    return define_composition(**kwargs)


def mass_flow_rate():
    Q_scfd = ureg.Quantity(1, "mmscf/day")

    AS = define_composition(
        y_Methane = 0.9,
        y_Ethane = 0.05,
        y_Propane=0.02,
        y_n_Butane = 0.01,
        y_CarbonDioxide= 0.01,
        y_Water= 0.0,
        eos = "HEOS"
        )
    
    mdot = Q_scfd.to("mol/s").magnitude * AS.molar_mass()          # kg/s
    print(mdot)

def calc_viscosity():
    AS = define_composition(
        y_Methane         = 0.95,
        y_Ethane          = 0.05,
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
        y_Benzene         = 0.0,
        y_CarbonDioxide   = 0.0,
        y_Water           = 0.0,
        y_Nitrogen        = 0.0,
        y_Oxygen          = 0.0,
        y_Argon           = 0.0,
        y_Hydrogen        = 0.0,
        y_HydrogenSulfide = 0.0,
        y_CarbonMonoxide  = 0.0,
        eos = "HEOS")
    
    T_in = ureg.Quantity(60, "degF").to("degK").magnitude
    P_in = ureg.Quantity(10000, "psi").to("Pa").magnitude

    AS.update(CP.PT_INPUTS, P_in, T_in)

    print(AS.cpmass())
    print(ureg.Quantity(AS.rhomass(), "kg/m^3"))
    print(AS.rhomolar())

if __name__ == "__main__":
    calc_viscosity()
    # test_define_combination()