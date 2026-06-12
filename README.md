# Hydraulics

This software performs hydraulic analysis of piping systems for both incompressible and compressible fluids. The general workflow is:

1. Build the fluid conduit out of components (pipe segments with elevation profiles, bends, valves, and abrupt contractions/expansions).
2. Set up the fluid properties — either via an `Incompressible_Fluid` instance (for liquids) or a CoolProp `AbstractState` (for gases/mixtures).
3. Call the component's calculating method (`dP` for incompressible, `dP_dT` for compressible) to solve for outlet conditions at a given flow rate.

All functions can be run directly in the terminal or via dedicated python scripts. For examples, see [textbook_test_functions.py](textbook_test_functions.py). Additionally, a GUI was Claude'd together to enable more user-friendly setup of single-line-segment and network scenarios. Run it by running `python run_gui.py`.

Pipeline branches can be wired in series and/or parallel via the helpers in `parallel.py`, which solve the flow-split balance such that every branch ends up at the same outlet pressure. For more complex networks (including potential for reverse flow), `network.py` and `compressible_network.py` can be used to construct a pipeline and component network and solve material and energy balances at node. For more details on the network solving methods, see [network.md](network.md)

Most quantities accept [pint](https://pint.readthedocs.io/) `Quantity` inputs for unit-safe entry, and the unit registry includes custom standard-volume units (`scm`, `scf`, `mscf`, `mmscf`) defined as mole equivalents so they can be supplied directly as flow rates.

---

## Base component classes — [component_classes.py](component_classes.py)

Dimension storage, input validation, CSV loading, and convenience properties for the three component types. These classes contain no fluid-mechanics calculations; the physics-specific subclasses for liquid and gas flow live in [incompressible.py](incompressible.py) and [compressible_flow.py](compressible_flow.py) and inherit from them.

### `Base_Line_Segment`
A pipe segment defined by a distance/elevation/geometry profile. The segment is internally stored as a list of `(distance_m, elevation_m, D_h_m, flow_area_m2)` 4-tuples using the inlet-point / forward-Euler convention (each point's geometry applies from that point forward to the next).

Dimensions can be supplied two ways:
- **Uniform geometry** — pass scalar `id_val`, `od_val`+`wt_val`, or `hydraulic_diameter`+`flow_area`, plus a `profile` of 2-tuples `(distance, elevation)` or a `length`+`elevation_change`.
- **Per-point geometry** — pass a `profile` of 4-tuples `(distance, elevation, D_h, flow_area)` for variable-diameter runs (reducers, custom bore profiles, etc.).

Non-circular cross-sections are supported by supplying `hydraulic_diameter` and `flow_area` directly with `noncircular=True`. Subclasses skip fitting-loss correlations that assume a circular bore when this flag is set.

`Base_Line_Segment.from_csv()` constructs a segment from a CSV profile. The header row determines the geometry mode: `ID` (with optional `OD`/`WT`) for circular pipes, or `D_h`/`flow_area` for direct hydraulic properties. All CSV values are assumed to be SI.

Convenience properties: `total_length_m`, `net_elevation_change_m`, `volume_m3`.

### `Base_Bend`
Dimension storage for a rounded (non-mitred) pipe bend/elbow fitting. Stores the inner diameter, bend angle (degrees), and bend-radius-to-diameter ratio (e.g. 1.5 for a standard long-radius elbow). All values must be positive.

### `Base_Valve`
Dimension storage for a valve fitting. Stores the pipe inner diameter and a pre-computed K-factor (resistance coefficient) referenced to the pipe velocity head. The K-factor is supplied at initialization rather than computed by the class, leaving the caller free to use whatever correlation or vendor data is appropriate for the valve type and position. `Di` must be positive and `K` must be `>= 0`. This pairs nicely with the fluids library's K factor correlations for valves, and you can call that library when you create a Valve class object.

### `Base_Contraction_Expansion`
Dimension storage for an abrupt contraction or expansion fitting. Stores upstream and downstream inner diameters. If `Di_US == Di_DS` there is no area change and the fitting has no effect.

### Module-level helpers
- `_to_si(val, unit)` — convert a pint `Quantity` or plain float to an SI float magnitude.
- `_resolve_id(id_si, od_si, wt_si, tol)` — resolve inner diameter from any combination of ID / OD / WT, warning when all three are supplied and disagree.
- `_flow_props_from_id(id_m)` — return `(D_h, flow_area)` for a circular pipe.

The module also registers custom pint units `scm`, `scf`, `mscf`, and `mmscf` as mole equivalents at their respective standard conditions (0 °C / 101325 Pa for `scm`; 60 °F / 14.696 psia for `scf`).

---

## Incompressible hydraulics — [incompressible.py](incompressible.py)

Liquid pipeline hydraulics. Components inherit geometry from `Base_*` and add Darcy-Weisbach friction, hydrostatic elevation head, and Bernoulli / K-factor area-change losses.

### `Incompressible_Fluid`
Stores density (kg/m³) and dynamic viscosity (Pa·s) as the working fluid properties. Both can be supplied as pint `Quantity` objects or plain SI floats. Optional `vapor_pressure` and `critical_pressure` slots (`vapor_pressure_si`, `critical_pressure_si`) are consumed by the cavitation checks on `Valve`, `CheckValve`, and `Orifice`; when omitted those checks are silently skipped.

`Incompressible_Fluid.from_api_gravity(api_gravity, viscosity)` constructs the fluid from petroleum API gravity using `ρ = 1000 · 141.5 / (API + 131.5)`.

### `Line_Segment` (inherits `Base_Line_Segment`)
Adds the methods:

- **`pressure_profile(fluid, P0, flow_rate)`** — walks consecutive profile points and returns a list of dicts (`distance_m`, `elevation_m`, `P_Pa`, `v_ms`). Each slice contributes:
  1. Friction loss: `dP = -(f_D/2)·ρ·v²/D_h · dL` (Darcy-Weisbach, always negative).
  2. Elevation: `dP = -ρ·g·dz` (positive downhill).
  3. Area-change at the boundary, when applicable: Bernoulli velocity-head exchange `½ρ(v_in² − v_out²)` plus a permanent K-factor loss from `fluids.fittings.contraction_sharp()` or `diffuser_sharp()`. The K-factor loss is omitted when `noncircular=True`.

- **`dP(fluid, flow_rate)`** — convenience wrapper returning the total static pressure change across the segment (negative when pressure decreases in the flow direction).

- **`dmdot(fluid, P_inlet, P_outlet)`** — inverse of `dP`. Returns the forward-flow mass flow rate (kg/s) that produces the requested pressure pair. Implemented as a 1-D `scipy.optimize.brentq` on `pressure_profile()` so it handles staircase profiles and elevation transparently. Raises `ValueError` if the bracket cannot be established (e.g. uphill segment with insufficient head).

`flow_rate` is a pint `Quantity` and may be volumetric (`[length]³/[time]`) or mass (`[mass]/[time]`).

### `Bend` (inherits `Base_Bend`)
Adds `dP(fluid, flow_rate)` using the `fluids.fittings.bend_rounded()` K-factor correlation, which accounts for bend angle, bend radius, and Reynolds number. Result is always ≤ 0. Also adds `dmdot(fluid, P_inlet, P_outlet)`, the inverse — a short K(Re) fixed point (typically 2–3 iterations) solves for mdot from the target pressure pair.

### `Valve` (inherits `Base_Valve`)
Adds `dP(fluid, flow_rate, P_inlet=None)` using the pre-computed K-factor stored on the instance: `dP = -K · ½ρv²`. No correlation or Reynolds-number lookup is performed. Result is always ≤ 0. Adds `dmdot(fluid, P_inlet, P_outlet)` — an analytic inverse since K is constant: `Q = A · sqrt(2 · (P_inlet − P_outlet) / (K · ρ))`.

If the valve was constructed with `F_L` (ISA-75.01 liquid pressure-recovery factor) and the fluid carries `vapor_pressure_si`, both `dP` (when called with `P_inlet`) and `dmdot` run a three-regime cavitation gate: flashing (`P_out < P_v`) and choked cavitating (`|dP| ≥ F_L² · (P_in − F_F · P_v)`, where `F_F = 0.96 − 0.28·sqrt(P_v / P_c)` if the fluid has `critical_pressure_si`, else 0.96) both raise `RuntimeError`; incipient cavitation (`σ = (P_in − P_v)/|dP| < 1/F_L²`) warns. With `F_L=None` the check is silent.

### `CheckValve` (inherits `Base_CheckValve`)
Same `dP(fluid, flow_rate, P_inlet=None)` and `dmdot(fluid, P_inlet, P_outlet)` shape as `Valve`, with the same optional `F_L` cavitation gate (forward flow only). Reverse flow is handled by the network solver's `_reversed_component()` machinery, which substitutes a sealing-K shadow.

### `Orifice` (inherits `Base_Orifice`)
Adds `dP(fluid, flow_rate, P_inlet=None)` using the Reader-Harris-Gallagher discharge-coefficient correlation (or `Cd_override` if supplied) converted to a K-factor referenced to the upstream pipe velocity. When `P_inlet` is supplied and the fluid carries `vapor_pressure_si`, a Cd-based sigma cavitation check fires (`RuntimeError` for choked, `UserWarning` for incipient). Adds `dmdot(fluid, P_inlet, P_outlet)` — a short Cd(Re) fixed point solves for mdot.

### `Contraction_Expansion` (inherits `Base_Contraction_Expansion`)
Adds `dP(fluid, flow_rate)` returning the total static pressure change (Bernoulli velocity-head exchange + permanent K-factor loss). Returns 0 when `Di_US == Di_DS`. Adds `dmdot(fluid, P_inlet, P_outlet)` — an analytic inverse of the quadratic relationship `dP_static = 0.5·ρ·Q²·β` (β is geometry-only), supports both contraction and expansion directions.

### Module-level functions
- **`dP_friction(fluid, flow_rate, flow_area, eps, D_h, dL)`** — Darcy-Weisbach friction loss for a uniform pipe length.
- **`dP_bend(fluid, flow_rate, Di, angle_deg, bend_dias)`** — bend loss for callers who don't want to build a `Bend` instance.
- **`dP_contraction_expansion(fluid, flow_rate, Di_US, Di_DS)`** — total static pressure change through a contraction or expansion.
- **`print_results(fluid, segment, flow_rate, dP_total_Pa)`** — prints a formatted summary (inlet velocity, line volume, dP) in mixed US / SI units.
- **`export_pressure_profile(segment, fluid, flow_rate, output_path, P0=0.0)`** — calls `pressure_profile()` and writes a per-point CSV (`point`, `distance_m`, `elevation_m`, `P_Pa`, `v_ms`).

---

## Compressible hydraulics — [compressible_flow.py](compressible_flow.py)

Single-phase compressible (gas) pipeline hydraulics. CoolProp `AbstractState` objects are used throughout for real-gas equation-of-state calculations, so any fluid or mixture supported by CoolProp can be used without modification. Component `dP_dT` methods mutate the `AbstractState` in place as conditions evolve along the pipe.

### `FlowState` — the calling unit across the compressible layer

Every public function and every component `dP_dT` method in this module takes a single `FlowState` argument instead of a loose `(AbstractState, flow_rate, …)` tuple. A FlowState bundles:

- `AS` — a CoolProp `AbstractState` at the **static** thermodynamic state.
- `mdot` — mass flow rate [kg/s], invariant once set.
- `A` — local flow cross-section area [m²].
- `z` — local elevation above datum [m]; advanced by `compressible_pipe_segment` so gravitational PE bookkeeping stays current along a pipe.
- Cached phase-envelope limits (`T_cricondentherm`, `P_cricondenbar`, `T_critical`, `P_critical`) so the four kwargs no longer thread through every signature. Built once via `_build_phase_limits(AS)` at construction by default.

Derived properties are recomputed from `AS` on every access so they cannot desync after an in-place AS mutation: `P`, `T`, `rho`, `v = mdot/(rho·A)`, `Ma = v/a_sound`, `h_stagnation = h_static + v²/2`, `s_static`, and `h_total_with_g = h_stagnation + g·z`.

```python
from compressible_flow import FlowState
AS.update(CP.PT_INPUTS, P_in, T_in)                  # caller anchors AS at static inlet
fs = FlowState(AS, mdot=2.0, A=area, z=0.0)          # builds phase envelope automatically
bend.dP_dT(fs)                                       # mutates AS to outlet static
```

Convention: AS is always at the **static** state. `v` and stagnation properties are derived. Functions that change geometry (`compressible_changing_area_K`, `compressible_pipe_segment`) mutate `fs.A` and/or `fs.z` so the FlowState stays self-consistent with its current location.

### `Line_Segment` (inherits `Base_Line_Segment`)
Adds **`dP_dT(fs, isothermal=False, q_wall=0.0, mu=None, energy_tol=10.0, dPdL_rel_tol=0.05, max_split_depth=8, verbose=False)`**. On entry it absorbs any inlet-area discontinuity with the upstream component via an isentropic area-change (`_area_match` → `compressible_changing_area_K(K=0)`), then walks consecutive profile slices via `compressible_pipe_segment()`, applying another isentropic area-change correction at every inter-slice boundary where the flow area changes.

Optional arguments:
- `isothermal` — hold temperature constant per slice (solves a simpler one-equation form).
- `q_wall` — total heat input to the fluid over the entire segment [W], distributed uniformly per unit length. Ignored when `isothermal=True`.
- `mu` — fixed viscosity in Pa·s; if `None`, CoolProp is queried each slice with a Lee-Gonzalez-Eakin fallback for EOS backends that don't support viscosity (e.g. Peng-Robinson).
- `energy_tol`, `dPdL_rel_tol`, `max_split_depth` — adaptive bisection tolerances forwarded into each slice.

Returns a list of `(distance_m, P_Pa, T_K, v_ms)` tuples — one per profile point — suitable for plotting profiles. `fs.AS` is left at outlet conditions; `fs.A` is updated to the profile's outlet area; `fs.z` is advanced by the segment's total elevation change.

### `Bend` (inherits `Base_Bend`)
Modeled as adiabatic. Adds **`dP_dT(fs, mu=None)`** which absorbs any inlet-area discontinuity, then uses `fluids.fittings.bend_rounded()` to obtain the K-factor and delegates to `compressible_K()`. `fs.A` is unchanged on return (equal inlet/outlet area).

### `Valve` (inherits `Base_Valve`)
Modeled as adiabatic. Adds **`dP_dT(fs)`** which absorbs any inlet-area discontinuity, then passes the pre-computed K-factor stored on the instance directly to `compressible_K()` — no viscosity, Reynolds-number, or correlation lookup is performed.

### `Contraction_Expansion` (inherits `Base_Contraction_Expansion`)
Modeled as adiabatic. Adds **`dP_dT(fs)`** which absorbs any inlet-area discontinuity to land at `A_US`, obtains the K-factor from `fluids.fittings.contraction_sharp()` (for contractions; the result is converted from a downstream- to an upstream-velocity reference) or `diffuser_sharp()` (for expansions), then delegates to `compressible_changing_area_K()`. On return `fs.A == A_DS`.

### Core functions

All five take a `FlowState` as their first argument; `mdot`, `A`, and the phase-envelope limits are read from it. `fs.AS` is mutated in place to the outlet static state, and area/elevation are updated when the function changes the geometry of the flow.

- **`compressible_pipe_segment(fs, dL, dz, D_h, roughness, q_wall=0.0, isothermal=False, mu=None, ...)`** — This one is the real beast!. It solves coupled dP/dL and dT/dL ODEs (or the simpler isothermal dP/dL equation) over a single pipe slice using a forward-Euler step with a one-iteration energy-balance correction.

  After the trial step, two convergence metrics are evaluated at the trial outlet:
  1. If not isothermal, an energy balance is performed (Stagnation enthalpy + heat in in minus stagnation enthalpy out = residual) `|H_in + q/mdot + v_in²/2 − g·dz − (H_out + v_out²/2)|` vs `energy_tol`.
  2. For all cases, the relative change in dP/dL between inlet and outlet fluid properties is compared to `dPdL_rel_tol`.

  If either metric is out of tolerance, the slice is recursively bisected (halving `dL`, `dz`, and `q_wall`) until convergence or `_max_split_depth` is exhausted. A final one-step Newton correction is applied to `T_out` so the converged state satisfies stagnation enthalpy energy balance (almost) exactly. Sonic / two-phase / unphysical-EOS-update conditions raise `RuntimeError`.

  The dP/dL derivation begins from the energy balance `1/mdot · dq/dL = dH/dL − v²/ρ · dρ/dL + g·dz/dL`, combined with the continuity and entropy balances (Zucker & Biblarz, *Fundamentals of Gas Dynamics* 2nd ed., eqs 3.1 / 3.64) and the thermodynamic identity `dH = T·dS + dP/ρ`.

  You can see a chicken scratch version of this derivation in the Derivation_images folder, or a more detailed discussion in the comments for the function.

- **`compressible_changing_area(fs, A_out)`** — ideal gas isentropic outlet `(P, T)` for an area change, using the area-Mach relation to find the subsonic outlet Mach number and recovering static conditions from total-condition ratios. `gamma = Cp/Cv` is taken from `fs.AS` at inlet. Returns `(P_out, T_out)` without mutating the FlowState. This function is used to compute the initial guess for the compressible_changing_area_K function.

- **`compressible_changing_area_K(fs, A_out, K=0.0)`** — outlet conditions for an area change with a known loss coefficient `K` (referenced to inlet velocity head); `fs.A` is the inlet area and is updated to `A_out` on return. Solves the simultaneous balance equations:

  1. Stagnation-enthalpy conservation: `H_out + v_out²/2 = H_in + v_in²/2`.
  2. Entropy generation from the irreversible loss: `S_out − S_in = K·v_in² / (2·T_avg)`.

  via a damped Newton iteration with an analytic Jacobian assembled from CoolProp partial derivatives (each Newton step costs exactly one `PT_INPUTS` flash; the partials at the just-flashed state are ~1 µs). The initial guess is the isentropic ideal gas result plus the linearized constant-area K-correction from `compressible_K`'s derivation, so typical calls converge in 1–3 flashes. If Newton stalls or walks somewhere CoolProp cannot evaluate, the solve restarts with `scipy.optimize.root` (hybrid method) from the original guess. Updates the `AbstractState` in place.

- **`compressible_K(fs, K, dPmax=0.05)`** — outlet conditions for a constant-area fitting (e.g. bend, valve) with a known `K`. Applies a single-step analytical result derived from the combined energy + continuity + entropy + EOS equations with `dA = 0`:

  ```
  dP = -K·v²·ρ/2  /  [1  −  v²·(∂ρ/∂P)_H / (1 − (v²/ρ)·(∂ρ/∂H)_P)]
  dT = [K·v²/2 + (1/ρ − (∂H/∂P)_T)·dP] / Cp
  ```

  For low Mach numbers `dP` reduces to the familiar `−K·ρ·v²/2`. A one-iteration Newton correction is then applied to `T_out` to enforce stagnation-enthalpy conservation exactly. A somewhat illegible handwritten derivation appears in the Derivation_images folder.

  **Adaptive fallback:** after computing the linearized `dP`, if `|dP|/P_in > dPmax` (default 5%) or the compressibility denominator `< 0.5` (indicating near-sonic conditions where the linearization breaks down), the function automatically delegates to `compressible_changing_area_K` with `A_out = fs.A`. This makes `Valve.dP_dT` and `Bend.dP_dT` regime-aware without any API change. Pass a larger `dPmax` to force the fast path, or smaller to be more conservative.

- **`choked_mass_flux(fs, A_throat, A_outlet=None, n_grid=40)`** — real-gas choked mass flow through a throat of area `A_throat`, using the an isentropic expansion. The stagnation enthalpy reference is built from `fs.h_stagnation = h_static + 0.5·v_in²`, so the choked mass flow correctly grows when the inlet carries non-trivial kinetic energy. The Mach=1 bracketing is a directed walk seeded at the critical pressure ratio `(2/(γ+1))^(γ/(γ−1))` (real-gas `γ = cp/cv` at the inlet) with a loosened inner entropy-root tolerance, falling back to an `n_grid`-point top-down scan if the walk finds no sign change; the final `brentq` polish runs at the tight tolerance, so the returned throat state is unaffected by the loose scan. Used internally by `compressible_K` and `compressible_changing_area_K` as a real-gas choke pre-screen; on choke they raise `ChokedFlowError` carrying `mdot_choked` and the (recovered) outlet state. The internal isentrope temperature search is floored just above the mixture cricondentherm so it cannot wander into the two-phase dome (where CoolProp's PT update fails); an expansion whose single-phase root would lie below that floor raises `TwoPhaseIsentropeError` rather than a cryptic EOS "stationary point" error, since this layer models single-phase flow only.

- **`_fanno_choke_mdot(forward_at_mdot, fs, mdot_seed, mdot_infeasible)`** — locate the Fanno (friction-limited) choke mass flow of a pipe segment by bisecting the feasibility boundary of its forward solve: `compressible_pipe_segment` raises at its `Ma ≥ 0.98` reactive gate, so the largest `mdot` the segment passes subsonically *is* the choke. Returns the 6-tuple `ChokedFlowError` expects (the pipe exit is the sonic throat). Used by `Line_Segment.dmdot_dT` to report the true Fanno choke in its `ChokedFlowError` rather than the looser isentropic-nozzle bound `choked_mass_flux` gives at the pipe area — the latter ignores wall friction and overstates the choke for a distributed-friction pipe.

- **`compressible_dA(fs, A_throat, K=0.0, A2=None, P2=None)`** — single-entry-point constriction solver (valve, orifice, restriction). Splits the process into two physically distinct stages: an *isentropic* acceleration from inlet to the throat / vena contracta (`compressible_changing_area_K(K=0)`), then a *K-dissipative* recovery from throat to outlet area `A2` (`compressible_changing_area_K(e_loss=0.5·K·v_in²)`). The entropy balance on the recovery step uses the throat-to-outlet average temperature, which better matches real geometry (most of the entropy generation lives in the post-vena-contracta turbulence) than averaging inlet to outlet.

  Two operating modes selected by whether `P2` is passed:

  - **Mode 1 — dictate `mdot`, solve for outlet `P`.** Mirrors the existing `dP_dT` calling convention; runs the two `compressible_changing_area_K` calls in sequence and leaves `fs` at the outlet static state with `fs.A == A2`.
  - **Mode 2 — dictate `P2`, solve for `mdot`.** Runs `choked_mass_flux` to bound the answer, marches the choked-throat state to `A2` with the K-recovery to compute `P2_choked`, then branches:
    - If `P2 < P2_choked`: flow is choked. `mdot` is clamped to `mdot_choked` and the outlet state is found by adiabatic expansion from the throat to the user-supplied `P2` (stagnation enthalpy conserved). Supersonic outlets raise `RuntimeError`.
    - Otherwise: subsonic. A 1-D `scipy.optimize.brentq` on `mdot` drives the two-stage march's outlet pressure to `P2`. Bracket is `[~0, 0.95·mdot_choked]` with automatic retreat if the kernel's ideal-gas initial-guess generator fails near the real-gas choke.

  Input validation enforces `K ≥ 0`, `0 < A_throat ≤ fs.A`, and `P2 < fs.P` when supplied. Covered by `test_compressible_dA` in [examples.py](examples.py).

- **`adiabatic_expansion_solver(fs, P2, T_ITER_MAX=8, H_ABS_TOL=1.0, H_REL_TOL=1.0e-9)`** — small Newton helper that expands `fs.AS` adiabatically to a target pressure `P2` at the current `fs.A`, iterating `T_out` until stagnation-enthalpy conservation `H(P2, T_out) + 0.5·v_out² = fs.h_stagnation` is satisfied. Used by `compressible_dA`'s choked branch to find the outlet temperature when `mdot` is clamped to `mdot_choked`.

### Helpers

- **`_safe_flowstate_update_PT(fs, P, T)`** — thin wrapper around `_safe_update_PT` that pulls the cached phase-envelope limits off `fs`. Used inside every physics function that needs to step `fs.AS` to a new `(P, T)` while respecting the supercritical phase hint.
- **`_area_match(fs, A_target, tol=1e-6)`** — if `fs.A` differs from `A_target` beyond `tol`, run `compressible_changing_area_K(fs, A_target, K=0)` so the state at `A_target` is reached isentropically. Every component `dP_dT` calls this first so chains of mixed-diameter components no longer need an explicit `Contraction_Expansion` between them.
- **`_resolve_mdot(flow_rate, abstract_state)`** — convert a pint `Quantity` to mass flow rate [kg/s]. Accepts mass (`kg/s`, `lb/hr`), molar / standard-volume (`mol/s`, `scf/day`, `mmscf/day`), or actual volumetric (`m³/s`, `ft³/min`). Standard-volume units are defined as mole equivalents in the unit registry, so they fall through the molar branch automatically. Used by callers that have a pint flow Quantity and want a plain `mdot` to feed into a `FlowState`.
- **`viscosity_LGE(T, mol_wt, density)`** — Lee-Gonzalez-Eakin correlation for hydrocarbon gas viscosity. Used as a fallback when the chosen EOS (e.g. Peng-Robinson) does not support viscosity calculation. Note that this correlation is only valid for light hydrocarbon gases, so be careful. I should probably build in some sort of warning that displays if it is used with non-hydrocarbon components.
- **`_build_phase_limits(AS)`** —     This function builds a phase envelope for a CoolProp abstract state to calculate the critical properties to aid in determining if a pressure/temperature combination is obviously in a single phase state or not by comparing it to the critical pressure/temperature and/or cricondenbar and cricondentherm. This enables us to pass a phase hint to CoolProp's abstract state update function, which increases its speed appreciably. If you don't supply a phase hint, CoolProp needs to determine the phase with every update calculation. Its routine for determining the phase also fails to converge for some cases even though it is clearly in a single phase region. This process helps avoid that error from rearing its head.

  The increase in calculation speed is very helpful when you need to perform thousands of abstract state updates to iterate over a segment or solve a convergence problem. 

  This function builds the phase envelope on a temporary abstract state so the working abstract state's internal solver state is not corrupted.  CoolProp's build_phase_envelope leaves the AbstractState at the last envelope point it visited; subsequent update() calls on the same object then fail unpredictably.

  The envelope tracer is fragile (some HEOS or Peng Robinson mixtures can fail to converge), so the critical-point query is wrapped separately.  When the envelope fails but the critical point succeeds, returns (None, None, T_critical, P_critical) -- callers can still use the critical point as a coarser phase hint via _safe_update_PT.
- **`_safe_update_PT(AS, P, T, ...)`** — wraps `AS.update(PT_INPUTS, ...)` with an explicit phase hint when the state is definitely outside the two-phase region. Bypasses CoolProp's internal phase-stability analysis, which can return false two-phase detections near the envelope for some mixtures. This function is dramatically faster and more reliable than using CoolProp's native abstract state update function when operating well outside of the two-phase region.

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

Solve the flow split between two or more parallel branches that share inlet and outlet nodes. Each branch may be a single component (`Line_Segment`, `Bend`, `Valve`, `Contraction_Expansion`) or a list of components run in series — in the series case the branch dP is the sum of its components, and (for compressible) the `AbstractState` is chained outlet-to-inlet through the components in order.

Both functions issue a warning when branch net elevation changes disagree by more than 0.1 m, since parallel branches must share inlet/outlet elevation to correspond to a physically realizable layout.

### `parallel_incompressible(line_segment_list, fluid, total_flow_rate)`
Returns `(dP_target, flow_fraction_list)` such that every branch sees the same dP.

The initial guess assumes identical friction factors and uses each branch's average flow area, so its resistance scales as `Σ L_k / A_k^2.5` over its pipe segments (fittings are ignored in the initial guess but kept in the iterated dP). The flow fractions are then refined by Newton's method with numerical slope estimation:

```
dP_target  = Σ(dP_i / s_i) / Σ(1 / s_i)
Δff_i      = (dP_target − dP_i) / s_i
```

Because elevation head is flow-independent, only the friction term enters the slope — this keeps Newton stable at low flow rates where elevation dominates and a ratio-based correction would stall. 

### `parallel_compressible(line_segment_list, AS, total_flow_rate)`
Returns `(P_out_list, T_out_list, flow_fraction_list)`. The phase envelope is built once and cached on every `FlowState` constructed during the iteration, so per-call `build_phase_envelope` is skipped (the composition does not change across iterations).

Each iteration resets `AS` to the common inlet `(P0, T0)`, constructs one `FlowState` per branch at that inlet (the branch's mass flow comes from `_resolve_mdot(total_flow_rate * flow_fraction, AS)`), then walks every component in the branch in series via `c.dP_dT(fs)`. Newton's method on flow fractions converges every branch to the same outlet pressure.

---

## Textbook test functions — [textbook_test_functions.py](textbook_test_functions.py)

End-to-end examples that check the program's calculations against textbook problems to validate its output.

## Debugging test functions - [test.py](test.py)

Debugging tests run to test new features


## Utilities - [utilities.py](utilities.py)

File for containing utility functions.

## To do's
-Add orifice plates (`Orifice` class exists; rebuild its `dP_dT` on top of `compressible_dA` for the throat-then-K-recovery split, and expose a `.K` property derived from `Cd`)
-Switch `Valve.dP_dT` to `compressible_dA` so the throat / K-recovery split replaces the inlet-linearized `compressible_K` path for high-dP service
-Wire the downstream-pressure-dictated mode of `compressible_dA` into the network solver (today only the flow-rate-dictated path is exercised)
-Handle flow choking due to pipe area, friction, or heat transfer on pipe segments
-Surface the incompressible Valve/CheckValve cavitation check at the network level — thread `P_inlet` through `_component_signed_dP` so the ISA-75.01 three-regime gate fires automatically during a network solve (today it fires only on direct `dP(..., P_inlet=...)` / `dmdot(...)` calls). See R5 in [improvements.md](improvements.md).
-Add heat transfer calculation to pipe segments
