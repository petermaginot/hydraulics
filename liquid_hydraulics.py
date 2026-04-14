import math
from pint import UnitRegistry

ureg = UnitRegistry()
#simple incompressible fluid hydraulics calculation program

def liquid_hydraulics():
    # --- Inputs (edit value and unit string to change units) ---
    line_diameter    = ureg.Quantity(1.939,   "inch")
    line_length      = ureg.Quantity(3000.0,   "ft")
    roughness        = ureg.Quantity(0.00015, "ft")
    elevation_change = ureg.Quantity(-10,  "ft")
    viscosity        = ureg.Quantity(1.0,     "cP")        # dynamic viscosity
    flow_rate        = ureg.Quantity(5000,   "oil_bbl/day")
    API_gravity      = 10.0                                # dimensionless, always API degrees
    grav_constant    = ureg.Quantity(9.8066,  "m/s^2")

    # --- Convert all dimensional inputs to SI ---
    d  = line_diameter.to("m").magnitude
    L  = line_length.to("m").magnitude
    eps = roughness.to("m").magnitude
    dz = elevation_change.to("m").magnitude
    mu = viscosity.to("Pa*s").magnitude
    Q  = flow_rate.to("m^3/s").magnitude
    g  = grav_constant.to("m/s^2").magnitude

    # Density from API gravity (kg/m^3)
    # rho = 1000 * 141.5 / (API + 131.5)
    rho = 1000.0 * 141.5 / (API_gravity + 131.5)

    # --- Intermediate calculations (all SI) ---
    flow_area   = math.pi * d ** 2 / 4.0     # m^2
    velocity    = Q / flow_area              # m/s
    line_volume = flow_area * L              # m^3
    Re          = rho * velocity * d / mu    # dimensionless

    # --- Fanning friction factor ---
    if Re > 2000:
        # Turbulent correlation from "Fluid Mechanics for Chemical Engineers,
        # Third Edition" by Noel de Nevers, p. 187.
        # Note: Fanning friction factor = (1/4) * Darcy friction factor.
        fanning_friction = 0.001375 * (1.0 + (20000.0 * eps / d + 1.0e6 / Re) ** (1.0 / 3.0)
        )
        flow_regime = "turbulent"
    else:
        # Laminar: f_Fanning = 16 / Re
        fanning_friction = 16.0 / Re
        flow_regime = "laminar"

    # --- Pressure drops (Pa) ---
    dP_friction   = 2.0 * fanning_friction * (L / d) * rho * velocity ** 2
    dP_elevation  = dz * g * rho
    dP_total      = dP_friction + dP_elevation

    # Wrap results as pint Quantities for unit conversion on output
    dP_friction_q  = ureg.Quantity(dP_friction,  "Pa")
    dP_elevation_q = ureg.Quantity(dP_elevation, "Pa")
    dP_total_q     = ureg.Quantity(dP_total,     "Pa")
    flow_area_q    = ureg.Quantity(flow_area,    "m^2")
    velocity_q     = ureg.Quantity(velocity,     "m/s")
    rho_q          = ureg.Quantity(rho,          "kg/m^3")
    line_volume_q  = ureg.Quantity(line_volume,  "m^3")

    # --- Print results ---
    print("=== Liquid Hydraulics Results ===")
    print(f"  Flow regime          : {flow_regime}")
    print(f"  Flow area            : {flow_area_q.to('in^2'):.4f~P}  ({flow_area_q:.6f~P})")
    print(f"  Velocity             : {velocity_q.to('ft/s'):.4f~P}  ({velocity_q:.4f~P})")
    print(f"  Line volume          : {line_volume_q.to('oil_bbl'):.4f~P}  ({line_volume_q:.4f~P})")
    print(f"  Fluid density        : {rho_q.to('lb/ft^3'):.4f~P}  ({rho_q:.4f~P})")
    print(f"  Reynolds number      : {Re:.1f}")
    print(f"  Fanning friction     : {fanning_friction:.6f}  ({flow_regime})")
    print(f"  dP friction          : {dP_friction_q.to('psi'):.4f~P}  ({dP_friction_q:.2f~P})")
    print(f"  dP elevation         : {dP_elevation_q.to('psi'):.4f~P}  ({dP_elevation_q:.2f~P})")
    print(f"  dP total             : {dP_total_q.to('psi'):.4f~P}  ({dP_total_q:.2f~P})")


if __name__ == "__main__":
    liquid_hydraulics()
