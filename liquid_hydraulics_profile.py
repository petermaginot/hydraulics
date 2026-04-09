import csv
import math
import os
from pint import UnitRegistry

ureg = UnitRegistry()


def load_elevation_profile(csv_path):
    """Read an elevation profile CSV and return a list of (distance_m, elevation_m) tuples.

    Expected CSV columns (by position, header row is skipped):
      col 0 - ignored (layer ID or similar)
      col 1 - distance from origin (m)
      col 2 - elevation (m)

    A synthetic zero-distance point is prepended using the elevation of the
    first data row, so the profile always begins at distance = 0.
    The list is sorted by distance before being returned.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                              # skip header
        for line in reader:
            dist_m = float(line[1])
            elev_m = float(line[2])
            rows.append((dist_m, elev_m))

    if not rows:
        raise ValueError("Elevation profile CSV contains no data rows.")

    rows.sort(key=lambda r: r[0])

    # Prepend zero-distance point at the elevation of the first entry.
    elev_at_zero = rows[0][1]
    profile = [(0.0, elev_at_zero)] + rows
    return profile


def export_pressure_profile(profile, rho, g, output_path):
    """Write a CSV with static pressure differential at each profile point.

    The differential pressure at each point is the static head relative to
    the start of the line (distance = 0):

        dP_static_i = rho * g * (elev_ref - elev_i)   [Pa]

    A positive dP means the fluid pressure is higher than at the start
    (i.e., the point is lower in elevation than the start).
    A negative dP means the fluid pressure is lower than at the start
    (i.e., the point is higher in elevation than the start).

    Args:
        profile    : list of (distance_m, elevation_m) tuples, as returned by
                     load_elevation_profile().
        rho        : fluid density (kg/m^3).
        g          : gravitational acceleration (m/s^2).
        output_path: path for the output CSV file.
    """
    elev_ref = profile[0][1]                      # elevation at distance = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance_from_start_m",
            "elevation_m",
            "dP_static_from_start_Pa",
        ])
        for dist_m, elev_m in profile:
            dP_static = rho * g * (elev_ref - elev_m)
            writer.writerow([f"{dist_m:.6f}", f"{elev_m:.6f}", f"{dP_static:.4f}"])

    print(f"  Pressure profile exported to: {output_path}")


def liquid_hydraulics(elevation_profile_csv=None):
    # --- Inputs (edit value and unit string to change units) ---
    line_diameter    = ureg.Quantity(1.939,   "inch")
    line_length      = ureg.Quantity(3000.0,   "ft")
    roughness        = ureg.Quantity(0.00015, "ft")
    elevation_change = ureg.Quantity(-10,  "ft")
    viscosity        = ureg.Quantity(1.0,     "cP")        # dynamic viscosity
    flow_rate        = ureg.Quantity(5000,   "oil_bbl/day")
    API_gravity      = 10.0                                # dimensionless, always API degrees
    grav_constant    = ureg.Quantity(9.8066,  "m/s^2")

    # --- Optional elevation profile CSV ---
    # When a CSV path is supplied, the total distance in the file overrides
    # line_length, and the net elevation change (last point minus first point)
    # overrides elevation_change.
    profile = None
    if elevation_profile_csv is not None:
        profile = load_elevation_profile(elevation_profile_csv)
        total_dist_m  = profile[-1][0]            # last distance value (m)
        net_elev_m    = profile[-1][1] - profile[0][1]   # net elevation change (m)
        line_length      = ureg.Quantity(total_dist_m, "m")
        elevation_change = ureg.Quantity(net_elev_m,   "m")
        print(f"  Elevation profile loaded: {len(profile) - 1} points, "
              f"total distance = {total_dist_m:.2f} m, "
              f"net elevation change = {net_elev_m:.4f} m")

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

    # --- Export pressure profile CSV (only when a profile was loaded) ---
    if profile is not None:
        output_path = os.path.splitext(elevation_profile_csv)[0] + "_pressure_profile.csv"
        export_pressure_profile(profile, rho, g, output_path)


if __name__ == "__main__":
    liquid_hydraulics(elevation_profile_csv="testprofile1.csv")
