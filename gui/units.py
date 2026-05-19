"""Unit-dropdown choices used across the input screens.

Each list is ordered with the most likely default first.  Strings are
display labels -- mostly identical to pint's registered unit name, but a
few (e.g. "BBL/D" instead of "oil_bbl/day") are renamed for readability.
Use `to_pint(display)` to translate a display label back to the pint
identifier before constructing a Quantity.

For temperature, pint treats degC/degF as offset units; absolute-T conversion
via ureg.Quantity(value, unit).to("K") works as expected.
"""

# Display label -> pint unit identifier.  Only entries that differ go in
# here; to_pint() falls back to identity for everything else.
_DISPLAY_TO_PINT = {
    "BBL/D": "oil_bbl/day",
}


def to_pint(unit):
    """Translate a display label to its pint unit identifier.

    Pass-through for any unit not in the rename table, so call sites can
    use this unconditionally before handing the string to pint.
    """
    return _DISPLAY_TO_PINT.get(unit, unit)


LENGTH      = ["m", "ft", "inch", "mm", "cm", "km", "mile"]
DIAMETER    = ["inch", "mm", "m", "cm", "ft"]
ROUGHNESS   = ["ft", "m", "mm", "inch", "micrometer"]

DENSITY     = ["kg/m^3", "lb/ft^3", "g/cm^3"]
VISCOSITY   = ["cP", "Pa*s", "mPa*s", "lb/(ft*s)"]

PRESSURE    = ["psi", "Pa", "kPa", "MPa", "bar", "atm"]
TEMPERATURE = ["degF", "degC", "K", "degR"]

# Flow-rate dropdowns are split by flow-type so the user is not offered
# molar/standard-volume units for an incompressible liquid (the
# incompressible solver would reject them).
FLOW_RATE_INCOMPRESSIBLE = [
    "m^3/s", "m^3/h", "gal/min", "BBL/D", "ft^3/s",          # volumetric
    "kg/s", "kg/h", "lb/h",                                  # mass
]

FLOW_RATE_COMPRESSIBLE = [
    "mmscf/day", "mscf/day", "scf/min", "scm/h",             # standard volume
    "mol/s", "mol/h",                                        # molar
    "kg/s", "kg/h", "lb/h",                                  # mass
    "m^3/s", "m^3/h",                                        # actual volumetric
]
