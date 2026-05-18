"""Unit-dropdown choices used across the input screens.

Each list is ordered with the most likely default first.  All strings are
valid pint unit names (using the ureg from component_classes, which defines
the custom standard-volume units scm/scf/mscf/mmscf and removes pint's
default 'bbl').

For temperature, pint treats degC/degF as offset units; absolute-T conversion
via ureg.Quantity(value, unit).to("K") works as expected.
"""

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
    "m^3/s", "m^3/h", "gal/min", "oil_bbl/day", "ft^3/s",   # volumetric
    "kg/s", "kg/h", "lb/h",                                  # mass
]

FLOW_RATE_COMPRESSIBLE = [
    "mmscf/day", "mscf/day", "scf/min", "scm/h",             # standard volume
    "mol/s", "mol/h",                                        # molar
    "kg/s", "kg/h", "lb/h",                                  # mass
    "m^3/s", "m^3/h",                                        # actual volumetric
]
