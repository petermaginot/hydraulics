# Hydraulics

This software performs hydraulic analysis of piping systems for both incompressible (liquid) and compressible (gas) fluids. The general workflow is:

1. Build the fluid conduit out of components (pipe segments with elevation profiles, bends, and abrupt contractions/expansions).
2. Set up the fluid properties — either via an `Incompressible_Fluid` instance (for liquids) or a CoolProp `AbstractState` (for gases/mixtures).
3. Call the component's calculating method (`dP` for incompressible, `dP_dT` for compressible) to solve for outlet conditions at a given flow rate.

Pipeline branches can be wired in parallel via the helpers in `parallel.py`, which solve the flow-split balance such that every branch ends up at the same outlet pressure.

All quantities accept [pint](https://pint.readthedocs.io/) `Quantity` inputs for unit-safe entry, and the unit registry includes custom standard-volume units (`scm`, `scf`, `mscf`, `mmscf`) defined as mole equivalents so they can be supplied directly as flow rates.

---

## Base component classes — [component_classes.py](component_classes.py)

Geometry storage, input validation, CSV loading, and convenience properties for the three component types. These classes contain no fluid-mechanics calculations; the physics-specific subclasses for liquid and gas flow live in [incompressible.py](incompressible.py) and [compressible.py](compressible.py) and inherit from them.

### `Base_Line_Segment`
A pipe segment defined by a distance/elevation/geometry profile. The segment is internally stored as a list of `(distance_m, elevation_m, D_h_m, flow_area_m2)` 4-tuples using the inlet-point / forward-Euler convention (each point's geometry applies from that point forward to the next).

Geometry can be supplied two ways:
- **Uniform geometry** — pass scalar `id_val`, `od_val`+`wt_val`, or `hydraulic_diameter`+`flow_area`, plus a `profile` of 2-tuples `(distance, elevation)` or a `length`+`elevation_change`.
- **Per-point geometry** — pass a `profile` of 4-tuples `(distance, elevation, D_h, flow_area)` for variable-diameter runs (reducers, custom bore profiles, etc.).

Non-circular cross-sections are supported by supplying `hydraulic_diameter` and `flow_area` directly with `noncircular=True`. Subclasses skip fitting-loss correlations that assume a circular bore when this flag is set.

`Base_Line_Segment.from_csv()` constructs a segment from a CSV profile. The header row determines the geometry mode: `ID` (with optional `OD`/`WT`) for circular pipes, or `D_h`/`flow_area` for direct hydraulic properties. All CSV values are assumed to be SI.

Convenience properties: `total_length_m`, `net_elevation_change_m`, `volume_m3`.

### `Base_Bend`
Geometry storage for a rounded pipe bend fitting. Stores the inner diameter, bend angle (degrees), and bend-radius-to-diameter ratio (e.g. 1.5 for a standard long-radius elbow). All values must be positive.

### `Base_Contraction_Expansion`
Geometry storage for an abrupt contraction or expansion fitting. Stores upstream and downstream inner diameters. If `Di_US == Di_DS` there is no area change and the fitting has no effect.

### Module-level helpers
- `_to_si(val, unit)` — convert a pint `Quantity` or plain float to an SI float magnitude.
- `_resolve_id(id_si, od_si, wt_si, tol)` — resolve inner diameter from any combination of ID / OD / WT, warning when all three are supplied and disagree.
- `_flow_props_from_id(id_m)` — return `(D_h, flow_area)` for a circular pipe.

The module also registers custom pint units `scm`, `scf`, `mscf`, and `mmscf` as mole equivalents at their respective standard conditions (0 °C / 101325 Pa for `scm`; 60 °F / 14.696 psia for `scf`).

---

## Incompressible hydraulics — [incompressible.py](incompressible.py)

Liquid pipeline hydraulics. Components inherit geometry from `Base_*` and add Darcy-Weisbach friction, hydrostatic elevation head, and Bernoulli / K-factor area-change losses.

### `Incompressible_Fluid`
Stores density (kg/m³) and dynamic viscosity (Pa·s) as the working fluid properties. Both can be supplied as pint `Quantity` objects or plain SI floats.

`Incompressible_Fluid.from_api_gravity(api_gravity, viscosity)` constructs the fluid from petroleum API gravity using `ρ = 1000 · 141.5 / (API + 131.5)`.

### `Line_Segment` (inherits `Base_Line_Segment`)
Adds the methods:

- **`pressure_profile(fluid, P0, flow_rate)`** — walks consecutive profile points and returns a list of dicts (`distance_m`, `elevation_m`, `P_Pa`, `v_ms`). Each slice contributes:
  1. Friction loss: `dP = -(f_D/2)·ρ·v²/D_h · dL` (Darcy-Weisbach, always negative).
  2. Elevation: `dP = -ρ·g·dz` (positive downhill).
  3. Area-change at the boundary, when applicable: Bernoulli velocity-head exchange `½ρ(v_in² − v_out²)` plus a permanent K-factor loss from `fluids.fittings.contraction_sharp()` or `diffuser_sharp()`. The K-factor loss is omitted when `noncircular=True`.

- **`dP(fluid, flow_rate)`** — convenience wrapper returning the total static pressure change across the segment (negative when pressure decreases in the flow direction).

`flow_rate` is a pint `Quantity` and may be volumetric (`[length]³/[time]`) or mass (`[mass]/[time]`).

### `Bend` (inherits `Base_Bend`)
Adds `dP(fluid, flow_rate)` using the `fluids.fittings.bend_rounded()` K-factor correlation, which accounts for bend angle, bend radius, and Reynolds number. Result is always ≤ 0.

### `Contraction_Expansion` (inherits `Base_Contraction_Expansion`)
Adds `dP(fluid, flow_rate)` returning the total static pressure change (Bernoulli velocity-head exchange + permanent K-factor loss). Returns 0 when `Di_US == Di_DS`.

### Module-level functions
- **`dP_friction(fluid, flow_rate, flow_area, eps, D_h, dL)`** — Darcy-Weisbach friction loss for a uniform pipe length.
- **`dP_bend(fluid, flow_rate, Di, angle_deg, bend_dias)`** — bend loss for callers who don't want to build a `Bend` instance.
- **`dP_contraction_expansion(fluid, flow_rate, Di_US, Di_DS)`** — total static pressure change through a contraction or expansion.
- **`print_results(fluid, segment, flow_rate, dP_total_Pa)`** — prints a formatted summary (inlet velocity, line volume, dP) in mixed US / SI units.
- **`export_pressure_profile(segment, fluid, flow_rate, output_path, P0=0.0)`** — calls `pressure_profile()` and writes a per-point CSV (`point`, `distance_m`, `elevation_m`, `P_Pa`, `v_ms`).

---

## Compressible hydraulics — [compressible.py](compressible.py)

Single-phase compressible (gas) pipeline hydraulics. CoolProp `AbstractState` objects are used throughout for real-gas equation-of-state calculations, so any fluid or mixture supported by CoolProp can be used without modification. Component `dP_dT` methods mutate the `AbstractState` in place as conditions evolve along the pipe.

> **Note:** the caller must initialize the abstract state to the inlet conditions by running `AS.update(CP.PT_INPUTS, P_in, T_in)` before calling any `dP_dT` method. The same `AbstractState` is read for inlet conditions and updated in place to outlet conditions on return.

### `Line_Segment` (inherits `Base_Line_Segment`)
Adds **`dP_dT(abstract_state, flow_rate, isothermal=False, q_wall=0.0, mu=None, ...)`**, which walks consecutive profile slices via `compressible_pipe_segment()` and applies an isentropic area-change correction (`compressible_changing_area_K` with K=0) at every inter-slice boundary where the flow area changes.

Optional arguments:
- `isothermal` — hold temperature constant per slice (solves a simpler one-equation form).
- `q_wall` — total heat input to the fluid over the entire segment [W], distributed uniformly per unit length. Ignored when `isothermal=True`.
- `mu` — fixed viscosity in Pa·s; if `None`, CoolProp is queried each slice with a Lee-Gonzalez-Eakin fallback for EOS backends that don't support viscosity (e.g. Peng-Robinson).
- `T_cricondentherm`, `P_cricondenbar`, `T_critical`, `P_critical` — precomputed phase-envelope limits. Forward these on repeated calls (e.g. parallel-branch iteration) to skip the computationally-expensive per-call `build_phase_envelope`.
- `energy_tol`, `dPdL_rel_tol`, `max_split_depth` — adaptive bisection tolerances forwarded into each slice.

Returns a list of `(distance_m, P_Pa, T_K, v_ms)` tuples — one per profile point — suitable for plotting profiles. The `AbstractState` is left at outlet conditions.

### `Bend` (inherits `Base_Bend`)
Modeled as adiabatic. Adds **`dP_dT(abstract_state, flow_rate, mu=None, ...)`** which uses `fluids.fittings.bend_rounded()` to obtain the K-factor and delegates to `compressible_K()`.

### `Contraction_Expansion` (inherits `Base_Contraction_Expansion`)
Modeled as adiabatic. Adds **`dP_dT(abstract_state, flow_rate, ...)`** which obtains the K-factor from `fluids.fittings.contraction_sharp()` (for contractions; the result is converted from a downstream- to an upstream-velocity reference) or `diffuser_sharp()` (for expansions), then delegates to `compressible_changing_area_K()`.

### Core integration functions

- **`compressible_pipe_segment(abstract_state, mdot, dL, dz, D_h, roughness, flow_area, q_wall=0.0, isothermal=False, mu=None, ...)`** — the workhorse. Solves the coupled dP/dL and dT/dL ODEs (or the simpler isothermal dP/dL equation) over a single pipe slice using a forward-Euler step with a one-iteration energy-balance correction.

  After the trial step, two convergence metrics are evaluated at the trial outlet:
  1. (Non-isothermal only) stagnation-enthalpy residual `|H_in + q/mdot + v_in²/2 − g·dz − (H_out + v_out²/2)|` vs `energy_tol`.
  2. Relative change in dP/dL between inlet and trial outlet vs `dPdL_rel_tol`.

  If either metric is out of tolerance, the slice is recursively bisected (halving `dL`, `dz`, and `q_wall`) until convergence or `_max_split_depth` is exhausted. A final one-step Newton correction is applied to `T_out` so the converged state satisfies stagnation enthalpy energy balance exactly. Sonic / two-phase / unphysical-EOS-update conditions raise `RuntimeError`.

  The dP/dL derivation begins from the energy balance `1/mdot · dq/dL = dH/dL − v²/ρ · dρ/dL + g·dz/dL`, combined with the continuity and entropy balances (Zucker & Biblarz, *Fundamentals of Gas Dynamics* 2nd ed., eqs 3.1 / 3.64) and the thermodynamic identity `dH = T·dS + dP/ρ`. After substitution, the result is:

  ```
  dP/dL = [ f·ρ·v²/(2·D_h)·(1 − v²·B/ρ)  +  ρ·g·dz/dL  −  v²·B/mdot · dq/dL ]
          / [ v²·A + v²·B/ρ − 1 ]
  ```

  where `A = (∂ρ/∂P)_H` and `B = (∂ρ/∂H)_P`. `T` then steps via the identity `dH = Cp·dT + [V − T·(∂V/∂T)_P]·dP`.

- **`compressible_changing_area(abstract_state, mdot, A_in, A_out)`** — ideal gas isentropic outlet `(P, T)` for an area change, using the area-Mach relation to find the subsonic outlet Mach number and recovering static conditions from total-condition ratios. `gamma = Cp/Cv` is taken from the `AbstractState` at inlet. Returns `(P_out, T_out)` without mutating the state (used as the initial guess for the loss-coefficient form).

- **`compressible_changing_area_K(abstract_state, mdot, A_in, A_out, K, ...)`** — outlet conditions for an area change with a known loss coefficient `K` (referenced to inlet velocity head). Solves the simultaneous balance equations:

  1. Stagnation-enthalpy conservation: `H_out + v_out²/2 = H_in + v_in²/2`.
  2. Entropy generation from the irreversible loss: `S_out − S_in = K·v_in² / (2·T_avg)`.

  via `scipy.optimize.root` (hybrid method), with the isentropic result as the initial guess. Updates the `AbstractState` in place.

- **`compressible_K(abstract_state, mdot, flow_area, K, ...)`** — outlet conditions for a constant-area fitting (e.g. bend) with a known `K`. Applies a single-step analytical result derived from the combined energy + continuity + entropy + EOS equations with `dA = 0`:

  ```
  dP = -K·v²·ρ/2  /  [1  −  v²·(∂ρ/∂P)_H / (1 − (v²/ρ)·(∂ρ/∂H)_P)]
  dT = [K·v²/2 + (1/ρ − (∂H/∂P)_T)·dP] / Cp
  ```

  For low Mach numbers `dP` reduces to the familiar `−K·ρ·v²/2`. A one-iteration Newton correction is then applied to `T_out` to enforce stagnation-enthalpy conservation exactly.

### Helpers

- **`_resolve_mdot(flow_rate, abstract_state)`** — convert a pint `Quantity` to mass flow rate [kg/s]. Accepts mass (`kg/s`, `lb/hr`), molar / standard-volume (`mol/s`, `scf/day`, `mmscf/day`), or actual volumetric (`m³/s`, `ft³/min`). Standard-volume units are defined as mole equivalents in the unit registry, so they fall through the molar branch automatically.
- **`viscosity_LGE(T, mol_wt, density)`** — Lee-Gonzalez-Eakin correlation for hydrocarbon gas viscosity. Used as a fallback when the chosen EOS (e.g. Peng-Robinson) does not support viscosity calculation.
- **`_build_phase_limits(AS)`** — builds the phase envelope on a temporary `AbstractState` (so the working state's solver isn't corrupted) and returns `(T_cricondentherm, P_cricondenbar, T_critical, P_critical)`. Returns `None` for any field that couldn't be computed; both queries failing yields `(None, None, None, None)`. The envelope tracer is fragile for some HEOS mixtures and the PR backend, which is why the critical-point query is wrapped separately.
- **`_safe_update_PT(AS, P, T, ...)`** — wraps `AS.update(PT_INPUTS, ...)` with an explicit phase hint when the state is definitely outside the two-phase region. Bypasses CoolProp's internal phase-stability analysis, which can return false two-phase detections near the envelope for some mixtures.

---

## Composition generator — [composition.py](composition.py)

Helpers for building CoolProp `AbstractState` objects from component mole fractions and for combining multiple streams.

- **`define_composition(y_Methane=..., y_Ethane=..., ..., eos="HEOS")`** — convenience constructor: pass mole fractions by component name (Methane, Ethane, Propane, n_Butane, IsoButane, n_Pentane, Isopentane, n_Hexane … n_Decane, Benzene, CarbonDioxide, Water, Nitrogen, Oxygen, Argon, Hydrogen, HydrogenSulfide). Components with `y_* > 0` are kept; the fractions are normalized to sum to 1. Returns an `AbstractState` configured with the chosen EOS (`HEOS` by default; `PR` is faster but doesn't support viscosity).

- **`define_combination(AS_gas, AS_oil, AS_water, gas_rate, oil_rate, water_rate, eos="HEOS")`** — combine up to three single-phase `AbstractState` streams into one mixed `AbstractState` whose mole fractions reflect the molar-weighted average of all active streams. Streams with either `AS` or rate set to `None` are silently skipped; components present in more than one stream are merged by summing molar contributions before normalization. The returned `AbstractState` has not yet been updated to a `(P, T)` state.

---

## Heat transfer — [heat.py](heat.py)

Experimental wall heat-transfer estimation for a hot pipe with internal gas flow.

- **`calc_heat_transfer()`** — sweeps a range of gas flow rates and, for each, computes the inside-film heat-transfer coefficient via the Gnielinski turbulent correlation (`ht.conv_internal.turbulent_Gnielinski`) using CoolProp-derived `Re`, `Pr`, and `k`. Returns/plots total heat transferred and the corresponding gas temperature rise across the pipe length.

This module is currently a standalone calculator; the `q_wall` parameter in `compressible.Line_Segment.dP_dT()` accepts a fixed total wall heat input rather than calculating it from a coupled film-coefficient calculation - at some point I'll work on improving that.

---

## Parallel branches — [parallel.py](parallel.py)

Solve the flow split between two or more parallel branches that share inlet and outlet nodes. Each branch may be a single component (`Line_Segment`, `Bend`, `Contraction_Expansion`) or a list of components run in series — in the series case the branch dP is the sum of its components, and (for compressible) the `AbstractState` is chained outlet-to-inlet through the components in order.

Both functions issue a warning when branch net elevation changes disagree by more than 0.1 m, since parallel branches must share inlet/outlet elevation to correspond to a physically realizable layout.

### `parallel_incompressible(line_segment_list, fluid, total_flow_rate)`
Returns `(dP_target, flow_fraction_list)` such that every branch sees the same dP.

The initial guess assumes identical friction factors and uses each branch's average flow area, so its resistance scales as `Σ L_k / A_k^2.5` over its pipe segments (fittings are ignored in the initial guess but kept in the iterated dP). The flow fractions are then refined by Newton's method with numerical slope estimation:

```
dP_target  = Σ(dP_i / s_i) / Σ(1 / s_i)
Δff_i      = (dP_target − dP_i) / s_i
```

Because elevation head is flow-independent, only the friction term enters the slope — this keeps Newton stable at low flow rates where elevation dominates and a ratio-based correction would stall. `Σ Δff_i = 0` by construction, so total flow is exactly preserved.

### `parallel_compressible(line_segment_list, AS, total_flow_rate)`
Returns `(P_out_list, T_out_list, flow_fraction_list)`. The phase envelope is built once and forwarded into every `dP_dT` call so per-call `build_phase_envelope` is skipped (the composition does not change across iterations).

Each iteration resets `AS` to the common inlet `(P0, T0)` before walking a branch (since `dP_dT` mutates `AS` in place), then walks every component in the branch in series. Newton's method on flow fractions converges every branch to the same outlet pressure.

---

## Test functions — [test_functions.py](test_functions.py)

End-to-end examples that check the program's calculations against textbook problems to validate its output.
