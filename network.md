# Pipe-network solver — status

A general directed-graph solver for pipe networks built on top of the existing component classes in [incompressible.py](incompressible.py) and [compressible_flow.py](compressible_flow.py) (and the base geometry / CSV-loading code in [component_classes.py](component_classes.py)).  `component_classes.py` also exposes `downsample_profile(profile, max_step_m=1000, elev_tol=0, ...)`, a helper that reduces dense survey CSVs (e.g. QGIS polyline exports) to the points that actually affect the solver: diameter-change boundaries, elevation slope-breaks above `elev_tol`, and a maximum spacing cap — substantially cutting the number of pipe slices and therefore solve time for compressible cases. Two solvers live alongside the existing per-component physics:

- [network.py](network.py) — `Network`, the incompressible-flow solver.
- [compressible_network.py](compressible_network.py) — `Compressible_Network`, a subclass that handles compressible flow with uniform composition.

The original motivation was to enable a GUI in which pipe segments and fittings can be arranged into series/parallel topologies, fluid properties defined, and one of three calculation modes run (find outlet P, find inlet P, find Q). The network solvers are the back end the GUI would call.

---

## Scope

Both solvers handle an arbitrary directed graph of pipe components, including:

- multiple inlets and multiple outlets,
- arbitrary series + parallel arrangements,
- edges whose **flow direction is not known in advance** — the sign of the edge flow is part of the solution. For instance, one branch's flow direction may depend on the relative back-pressures at two outlets.

The reference DWSIM topology that motivated the design:

```
        Inlets: 1, 5          Outlets: 15, 16

        PIPE-1 :  1     -> MIX-1
        conn-5 :  5     -> MIX-1
        PIPE-2 :  MIX-1 -> SPL-1
        PIPE-3 :  SPL-1 -> MIX-2
        PIPE-4 :  SPL-1 -> SPL-2
        PIPE-5 :  SPL-2 -> MIX-2    (flow direction emerges from the solve)
        conn-16:  SPL-2 -> 16
        conn-15:  MIX-2 -> 15
```

Before committing to a custom solver, [pandapipes](https://github.com/e2nIEE/pandapipes) was evaluated. Verdict: useful as a reference for the signed-branch-variable solver pattern, but a poor fit as a direct dependency — it strips per-point pipe profiles, drops pint units in favor of plain SI floats, collapses Bend/Valve/Contraction_Expansion into a single scalar loss coefficient, and pulls in pandapower as a transitive dependency.

---

## Data model — [Node](network.py), [Edge](network.py)

A **`Node`** is a junction, inlet, or outlet. It carries an elevation and an optional combination of boundary conditions:

- `P_spec_Pa` — specified absolute pressure (mutually exclusive with `Q_ext`).
- `Q_ext_spec_qty` — specified external supply, stored as a raw pint `Quantity` so unit conversion can be deferred until `solve()` time (when the fluid is available). Positive = flow into the node from outside; negative = withdrawal. Both solvers convert internally to mass flow [kg/s].
- `T_spec_K` — specified temperature [K]. Ignored by `Network`; used by `Compressible_Network` (inlets typically have it, outlets typically don't).

An **`Edge`** is a directed connection between two existing nodes, carrying a list of components evaluated in series in the `from_node -> to_node` direction. The list may be empty (a zero-pressure-drop connector that pins `P_from == P_to`) or contain one or more `Line_Segment` / `Bend` / `Valve` / `CheckValve` / `Contraction_Expansion` instances. The same component set is available in both regimes — `compressible_flow.CheckValve` mirrors `incompressible.CheckValve` (inheriting the `check_valve = True` marker from `Base_CheckValve`), with a `dP_dT()` body identical to `Valve.dP_dT()`.

Both networks are built imperatively:

```python
# Incompressible
net = Network()
net.add_node("inlet",   P=ureg.Quantity(200, "psi"))
net.add_node("outlet",  P=ureg.Quantity( 50, "psi"))
net.add_edge("PIPE-1",  "inlet", "outlet", [seg1, bend1, seg2])
result = net.solve(fluid)

# Compressible
gnet = Compressible_Network()
gnet.add_node("inlet",  P=ureg.Quantity(1000, "psi"), T=300.0)
gnet.add_node("outlet", Q_ext=ureg.Quantity(2.0, "kg/s") * -1)
gnet.add_edge("PIPE-1", "inlet", "outlet", segment)
result = gnet.solve(abstract_state)
```

Sign and unit conventions (shared by both solvers):

- A positive `mdot_e` on an edge means flow in the nominal `from_node -> to_node` direction; negative means flow runs `to_node -> from_node`.
- `Q_ext` at a node is positive for supply into the network, negative for withdrawal.
- `Q_ext` accepts any pint flow Quantity. For the incompressible solver: mass or actual volumetric (converted via `fluid.density_si`). For the compressible solver: mass or molar / standard-volume (converted via `AbstractState.molar_mass()`); actual volumetric is rejected because density varies along the network.
- Bare floats (no units) at `add_node()` default to **kg/s** for both solvers.

---

## Incompressible solver — `Network.solve`

Unknowns:

- pressure `P_n` at every node whose pressure is not specified;
- signed mass flow `mdot_e` [kg/s] on every edge.

Equations:

- **Pipe equation** per edge: `P_from - P_to = -dP_inlet_to_outlet(mdot_e)`, where the per-edge dP is the signed sum across the edge's components walked in flow direction. The component `dP(fluid, flow_rate)` methods accept a mass-flow Quantity directly.
- **Mass balance** at each non-P-spec node: sum of signed edge mdots + spec'd `Q_ext` (in kg/s) = 0. Interior junctions default to 0.

The combined system is solved with `scipy.optimize.least_squares` (Trust Region Reflective, `x_scale='jac'`) on the concatenated `(free P, edge mdot)` vector. The Jacobian is built by finite differences — adequate for networks up to a few dozen edges; a sparse analytic Jacobian would be needed for larger systems. Earlier versions used `fsolve` (Powell hybrid), but it stalled on any topology mixing tiny check-valve dP with much larger pipe dP: the trust region collapsed and `fsolve` would return `ier=1` with the residual still in the kPa range. TRF handles that ill-conditioning gracefully.

After convergence, the external flow at each P-spec node is recovered from its mass balance and reported alongside the spec'd / interior values.

**Initial guesses:**
- Free node pressures = mean of P-spec values.
- Edge mdots = the largest spec'd `|Q_ext|` (in kg/s) by default, or 10 kg/s if no Q_ext is spec'd. This is intentionally **scale-matched** to the boundary flow: a too-small uniform guess produces a noisy finite-difference Jacobian (friction goes like mdot², so its derivative near zero is tiny), which can leave the trust region stuck.
- Near `|mdot| / rho < 1e-9 m³/s` (effectively zero flow), friction is linearized through zero to avoid the `friction_factor(Re=0)` singularity and to keep the residual smooth.

**Convergence** is declared when the residual L2 norm falls below `1e-4`, regardless of the solver's status code. Trusting the residual rather than the status code catches both false negatives (status reporting "no progress" when the residual has actually hit floating-point noise) and false positives (status reporting "step below xtol" when the trust region collapsed without driving the residual to zero — the failure mode that motivated switching off fsolve).

At least one P-spec node is required to anchor pressures.

**Per-component `dmdot` — analytic inverse, not wired into the solver.** Every incompressible component class carries a `dmdot(fluid, P_inlet, P_outlet) -> mdot_kg_s` method that mirrors the compressible `dmdot_dT(fs, P2)` contract. `Valve` / `CheckValve` / `Contraction_Expansion` are analytic closed-form inversions of `dP`; `Bend` and `Orifice` use a 2–3 iteration K(Re) or Cd(Re) fixed point; `Line_Segment` uses a 1-D `brentq` on `mdot` against `pressure_profile()` because per-slice f(Re) plus staircase area changes plus elevation rule out closed form. Unlike the compressible case, these methods are **not** plumbed into a `walk_edge_inverse` analog inside `Network.solve()` — the incompressible residual `(P_from - P_to) + dP(Q) == 0` is already symmetric in which end is pinned (no special inverse mode needed for P-spec-downstream edges). The `dmdot` methods exist as standalone analytic helpers for post-solve diagnostics, GUI flow-rate-driven workflows, and as the natural inverse for any future control-loop or sensitivity tooling.

---

## Compressible solver — `Compressible_Network.solve`

A subclass of `Network` that re-uses the `Node` / `Edge` data classes, the reversed-shadow geometry handling, and the general graph machinery. It replaces the residual function with a joint Newton on `(P_n, T_n, mdot_e)` and calls each component's `dP_dT()` method via a CoolProp `AbstractState` instead of `dP()`.

**Assumptions** (for the v1 implementation):
- **Uniform composition.** All streams share the same composition, supplied as a single `AbstractState` argument to `solve()`. Heterogeneous-composition mixing is not supported.
- **No heat transfer.** `q_wall` is not exposed by the network solver yet; every edge runs adiabatic.

**Unknowns:**
- `P_n` at each non-P-spec node;
- `T_n` at each non-T-spec node;
- signed `mdot_e` on each edge.

**Equations:**
- **Pipe equation per edge** — one residual: walk the components in flow direction starting from the flow-inlet node's `(P, T)` by constructing a single `FlowState` (carrying the `AbstractState`, mass flow, the chain's first component's inlet area, the inlet node's elevation, and the pre-built phase-envelope limits) and threading it through every component's `dP_dT(fs)`; the resulting walked outlet P must equal the flow-outlet node's P. The walked outlet T is *not* directly matched (because at a mixer the node T differs from any single edge's walked T); it feeds into the energy balance instead. Mixed-diameter chains are absorbed automatically — each `dP_dT` calls `_area_match` first to isentropically reconcile any inlet-area discontinuity with the previous component's outlet.
- **Mass balance** at each non-P-spec node (kg/s).
- **Energy balance** at each non-T-spec node:
  - inflows (edges + supply): contribute `|mdot| * h_inflow` where `h_inflow` is the walked outlet enthalpy of an incoming edge, or `h(P_n, T_spec)` for a supply boundary;
  - outflows (edges + withdrawal): contribute `|mdot| * h(P_n, T_n)` at the node's mixed state;
  - **including the implicit withdrawal at a P-spec outlet** — `Q_ext` is derived from mass balance (`Q_ext = -net_signed_mdot_in`) and added as an outflow. Without this term the equation collapses to "inflow enthalpy = 0", which only the trivial all-zero-flow solution satisfies, producing a spurious zero-flow attractor. This was the v1 bug that prevented mixing tests from converging until fixed.

**Residual scaling.** Pipe / mass / energy balance residuals are normalized to O(1) magnitude (divided by `P_init`, `mdot_ref`, and `mdot_ref * 1e6` respectively) so the Jacobian is not dominated by any one equation class.

**Solver.** `scipy.optimize.least_squares` with the Trust Region Reflective algorithm. TRF accepts variable bounds, which keep trial pressures positive and temperatures above absolute zero — otherwise the ill-conditioned early Newton steps drive `T` deeply negative and the CoolProp PT update fails. `x_scale='jac'` normalizes unknown magnitudes automatically from the Jacobian.

**Phase envelope.** Built once per `solve()` via `_build_phase_limits()` and cached on every `FlowState` constructed inside the walk, so the four limits no longer need to thread through every call signature (rebuilding the envelope is expensive for HEOS mixtures).

**Per-component outlet readout.** After convergence, one additional walk is performed per edge using the converged solution to record the flow-direction outlet `(P_Pa, T_K)` of each component (same `FlowState`-based pattern as `walk_edge`). The list is reindexed back to original `edge.components` order before being returned, so callers can map results to their own per-component objects without needing to know whether the converged flow ended up running forward or reverse on that edge. The data lands in `result["component_outlet_PT"]: dict[edge_name -> list of (P_Pa, T_K)]` and powers the per-block P / T annotations on the compressible GUI canvas (see [GUI.md](GUI.md)).

**Per-edge initial guess.** `mdot_init_kgs` accepts either a scalar (broadcast to every edge) or a `dict {edge_name: kg/s}`. For networks with mixer/splitter junctions, a uniform scalar is mass-imbalanced at the junction and can trap the solver at the zero-flow minimum; pass a mass-balanced dict to seed properly.

**Boundary-condition rule** (validated at solve time): every node with positive `Q_ext` (a supply into the network) must specify `T` so the inflow enthalpy can be evaluated.

**Pressure-driven (inverse-mode) edge dispatch.** Edges whose downstream node is P-spec'd are auto-detected at the top of `solve()` (`inverse_of = [self._nodes[e.to_node].P_spec_Pa is not None for e in self._edges]`) and routed through `walk_edge_inverse` instead of `walk_edge`. The pipe equation for those edges is `(mdot[e_idx] - mdot_solved) / mdot_ref` where `mdot_solved` comes from a local brentq solve of the downstream-P boundary condition — instead of the forward `(walked_P - P_outlet) / P_ref`. This delivers an analytic-quality `+1 / mdot_ref` Jacobian column for the edge's mdot (only the upstream-node coupling needs finite differences) and a bracketed, brentq-converged local solve that is robust near choke. Edges feeding a junction (downstream not P-spec) keep the forward path.

`walk_edge_inverse` dispatches on edge shape:

- **Single-component edge** (the common relief-valve case): delegates directly to `component.dmdot_dT(fs, P2=P_target)`. Required for correctness on components with internal throats (`Orifice`, `Valve` / `CheckValve` with `minimum_diameter`) — the component knows its own throat geometry and routes through `compressible_dA` Mode 2's exact bracketing. A generic-bracket implementation that overestimated `mdot_choked` from the pipe area would consistently fail to bracket subsonically on these and fall through to a penalty `mdot_solved = 0`, which LM happily accepts as a trivial solution on single-edge two-P-spec networks.
- **Multi-component edge**: builds a forward closure that walks the whole chain in flow order and drives mdot via the shared `_solve_mdot_for_outlet_P` helper. Choke bound uses the cheap `_ideal_gas_G_max * A_min` estimate over component inlet areas; the helper's retreat schedule handles overestimates.
- **Check valve anywhere on the edge** (highest priority): if the trial is a reverse drive (`mdot_trial < 0` **or** `P_target ≥ P_in`), the valve seals — returns the inlet-node state as a pass-through with `sealed=True` and `mdot_solved = 0`, warning-free (it is the correct physical answer, not an infeasibility). The residual dispatcher then uses the sealed-edge `mdot / mdot_ref` form rather than the inverse form.
- **Reverse-flow trial** (`mdot_trial < 0`): not supported on inverse edges. Returns `mdot_solved = 0` (the inverse residual `mdot/mdot_ref` then has zero at `mdot = 0` with positive slope, pulling LM back toward positive territory) plus a once-per-solve UserWarning via `_warn_once`. Relief valves are physically forward-only so this is sufficient for the canonical use case; a network whose edges genuinely need to reverse direction must use the forward path (downstream junction instead of P-spec).
- **`P_target >= P_in_trial`** (transient infeasibility during LM): same `mdot_solved = 0` + once-per-solve warning.
- **`ChokedFlowError`** from `component.dmdot_dT` (single-component path) or from the inner brentq (multi-component path): clamps `mdot_solved = exc.mdot_choked` and returns the choked-state outlet. The `(mdot - mdot_choked) / mdot_ref` residual then drives Newton to the clamp value in one outer step — the physically correct answer when a relief valve sees backpressure below the choke point. For a single `Line_Segment` edge, `exc.mdot_choked` is now the true **Fanno** (friction-limited) choke from `_fanno_choke_mdot` (the largest mdot the pipe passes before its `Ma ≥ 0.98` reactive gate), not the looser isentropic-nozzle bound at the pipe area — so the clamp matches the physically realizable pipe-exit-sonic flow.
- **`RuntimeError` or `ValueError`** from the kernel (numerical infeasibility): falls through to `mdot_solved = 0` penalty.

**Finite-difference step on networks with inverse edges.** When `any(inverse_of)`, `solve()` passes `diff_step=1e-5` (relative) to `least_squares` rather than the default `sqrt(machine_eps) ~ 1.5e-8`. The brentq inside `walk_edge_inverse` has a tolerance floor (set by `xtol_factor=1e-8 * mdot_choked` and the helper's bracket retreat) that swamps the FD signal at the default step size, stalling LM on a non-zero residual. The 1e-5 step sits well above the brentq noise floor with negligible truncation-error penalty since the residual is still O(1) at LM tolerances. Forward-only networks keep the default step (preserving byte-identical behavior on pre-existing test suites).

**Energy-balance guard at P-spec outlets.** A trial step that flips a connected edge's mdot temporarily negative drives the implicit `Q_ext_eff` at a P-spec outlet node positive (apparent supply into the network). The energy-balance loop previously read `node.T_spec_K` to evaluate the supply enthalpy and crashed with `TypeError` on a P-spec-only outlet (no `T_spec_K`). The supply branch now `pass`-es when `T_spec_K is None`, treating that transient configuration as zero supply contribution so LM can step away rather than crash. At the converged solution the branch never fires (real P-spec outlets always have `Q_ext_eff < 0`).

**Choke pre-screen skip flag.** Both `compressible_changing_area_K` and `compressible_K` accept `skip_choke_check=False`. When `True`, the `_choke_pre_screen` call (which runs `choked_mass_flux` for ~150 ms on HEOS mixtures) is bypassed. Used by every brentq driver that has already capped `mdot ≤ 0.95 * mdot_choked` from a prior choke determination — `compressible_dA` Mode 2, `Valve.dmdot_dT` (no-throat branch), `Bend.dmdot_dT`, and `Contraction_Expansion.dmdot_dT`. Without this flag the per-iteration pre-screen dominates per-call cost (100-1000x slowdown on `dmdot_dT`); with it, brentq runs are limited by the actual kernel solve rather than redundant choke checks. The simple component `dmdot_dT` methods also use the cheap `_ideal_gas_G_max * A` for their brentq upper bound instead of the expensive real-gas `choked_mass_flux`; the real-gas call is deferred to the bracket-failure branch where the structured `ChokedFlowError` payload is needed.

**Verbose-flag housekeeping.** `Line_Segment.dP_dT()` and `_build_phase_limits()` accept a `verbose=False` argument; the per-step progress prints and "building phase envelope" messages are silent by default and only fire when explicitly enabled. (Network-solver test output is therefore clean.)

---

## Reverse-flow handling (shared)

The hard part of both solvers. When an edge's solved flow comes out negative, each of its components is evaluated **against a reversed shadow of itself**, not via a `sign(Q) * friction(|Q|)` shortcut. This matters because:

- a sharp contraction in the reverse direction is a sharp **expansion**, with a different K-factor correlation;
- a variable-diameter `Line_Segment` produces different Bernoulli/loss contributions depending on traversal direction;
- compressible flow is even more direction-sensitive — the exact sequence of states along the actual flow path governs the state evolution.

`_reversed_component()` in [network.py](network.py) dispatches by duck typing:

- a component with a `.profile` attribute (Line_Segment-like) gets a shallow copy with the profile list rebuilt: distances flipped (`new_dist = total_L - old_dist`), elevations / hydraulic diameters / flow areas reversed in order. All segment properties (`total_length_m`, `net_elevation_change_m`, `volume_m3`) follow automatically from the new profile.
- a component with `.Di_US_si` and `.Di_DS_si` (Contraction_Expansion-like) gets the two diameters swapped — a contraction becomes an expansion.
- everything else (Bend, Valve, CheckValve) is treated as symmetric and returned unchanged. A check valve's asymmetry is not a resistance asymmetry at all — reverse flow through it is simply forbidden (see below), so its forward K is fine for the trial evaluations the solvers make at negative mdot.

The reversed copies are cached by `id()` of the original so they're built at most once per solve. The originals are never mutated.

For the incompressible solver: the reversed shadow's `dP()` at `+|mdot|` gives the dP in flow direction directly; negating restores the forward inlet-to-outlet convention. For the compressible solver: walking the reversed shadow at `+|mdot|` from the new-flow-inlet's `(P, T)` produces the right outlet state directly.

**Check valves: perfect seal, zero reverse flow.** A check valve is modeled as sealing perfectly — an edge carrying one passes *exactly* zero reverse flow. (The previous model substituted a huge sealing K ≈ 1e9, which leaked more as reverse ΔP grew — backwards from real valve behavior, where higher reverse ΔP seats the disc harder.) A sealed valve is a **complementarity condition**, not a resistance: either the edge is open (`mdot > 0` and the ordinary pressure balance holds) or it is sealed (`mdot = 0` and any reverse driving pressure is held across the seat). The two solvers realize this differently:

- **Incompressible: smoothed Fischer-Burmeister residual row.** Every CV-carrying edge replaces its pipe equation with `phi(a, b) = a + b − sqrt(a² + b² + eps²)` where `a = mdot/mdot_ref`, `b = −r_P/P_ref` (`r_P` being the ordinary pipe-equation residual), and `eps = _CHECKVALVE_FB_EPS = 1e-6`. Its zero set is the hyperbola `a·b = eps²/2` with `a, b > 0` — the open branch enforces the pressure balance to ~`eps²/(2a)·P_ref`, the sealed branch leaves a residual leak ~`eps²/(2b)·mdot_ref` that *shrinks* as reverse ΔP grows (the numerical artifact points the physically-correct way), and `phi < 0` strictly for every `a < 0`, so no backflow root exists at all. One smooth C∞ formula, no branch at Q=0, no retry heuristics; `eps` sits well above `least_squares`' ~1.5e-8 FD step so the finite-difference Jacobian resolves the corner. Post-solve, sealed-edge mdot (< `1e-6·mdot_ref`) is snapped to exactly 0.0 before Q_ext recovery, and `ComponentResult` bridges the whole-edge pressure imbalance onto the first CV so the seat visibly holds the full ΔP while other components on the edge show their ordinary (at zero flow, elevation-only) dP.
- **Compressible: sealed-edge detection + residual swap.** `walk_edge()` checks, before doing anything else, whether the edge contains a CV and the trial `mdot < 0`; if so it returns the inlet-node state as a pass-through (no components are walked — nothing downstream of a sealed seat should be asked to integrate a near-vacuum state) and flags `sealed=True`. The residual dispatcher swaps the pipe equation for `mdot / mdot_ref`, zero iff `mdot = 0` — an exact complementarity residual that is always satisfiable, so sealed configurations converge cleanly with zero residual (the old clamp-based model left an unavoidable leftover pressure mismatch and a spurious "did not converge" warning). `walk_edge_inverse()` seals on `mdot_trial < 0` **or** `P_target ≥ P_in` for CV edges, warning-free (it is the correct physical answer, not an infeasibility). Post-solve, sealed-edge mdot is snapped to exactly 0.0, and `component_outlet_PT` reports components upstream of the CV at the from-node state and the CV plus everything downstream at the to-node state.

**Dead legs.** A free node whose every connection to the P-spec'd part of the network crosses a sealed CV has nothing pinning its pressure (incompressible) or pressure/temperature (compressible) — the sealed rows are satisfied regardless. Both solvers run a post-solve reachability sweep from the P-spec nodes across non-sealed edges and warn that such nodes' reported values are indeterminate. A junction whose *only* inflow paths seal off while a `Q_ext` demands flow remains a genuinely infeasible spec and still produces the ordinary "did not converge" warning.

For robustness against general numerical infeasibility during LM trial steps (e.g. a slice that can't be split to within `energy_tol`, a CoolProp PT update at a pathological state, etc.), `walk_edge()` also wraps the per-component loop in a `try/except (RuntimeError, ValueError)` that resets the AS to inlet conditions and returns a penalty walked-outlet of `_WALK_FAIL_P_PA = 1.0` Pa. The resulting `~ -1` pipe-equation residual reads as "step rejected" to the trust-region machinery, which shrinks its radius and tries elsewhere instead of aborting the whole solve.

---

## Result query layer

`Network.solve()` returns a `NetworkResult` that supports **two access styles**.

**Dict-style** (matches the original return shape, kept for backwards compatibility — including the legacy `Q_m3s` keys, which are derived from `mdot_kgs / fluid.density_si`):
```python
result["P_Pa"]["MIX-1"]            # node pressure [Pa]
result["mdot_kgs"]["PIPE-3"]       # signed edge mass flow [kg/s]
result["mdot_ext_kgs"]["outlet"]   # external supply [kg/s]
result["Q_m3s"]["PIPE-3"]          # derived volumetric flow [m^3/s]
result["Q_ext_m3s"]["outlet"]      # derived
result["converged"]                # bool
```

**Per-object views** (the more ergonomic API for downstream code and plotting):

`result.edge(name)` returns an `EdgeResult` exposing both mass and volumetric flow alongside flow-direction-aligned pressures:
- `mdot_kgs`, `mdot_abs_kgs`, `flow_direction` (`"forward"` / `"reverse"`),
- `Q_m3s`, `Q_abs_m3s` (derived from density),
- `P_from_Pa`, `P_to_Pa` (nominal-orientation),
- `P_inlet_Pa`, `P_outlet_Pa` (flow-direction-aligned).

`result.component(component_obj)` returns a `ComponentResult` (identity-based lookup; raises if the component appears on more than one edge) with:
- `P_in_Pa`, `P_out_Pa` — always **flow-direction-aligned**, not nominal,
- `dP_Pa = P_out - P_in` (negative for a friction loss),
- `mdot_kgs`, `mdot_abs_kgs`, `Q_m3s`, `Q_abs_m3s`, `flow_direction`,
- `pressure_profile()` for Line_Segment-like components — returns the same `[{distance_m, elevation_m, P_Pa, v_ms}, ...]` shape as `Line_Segment.pressure_profile()`, walked in flow direction (using the reversed shadow under reverse flow, so distances run 0 → L from the flow inlet to the flow outlet with elevations and Re/friction computed correctly along the way). Fittings raise `AttributeError`.

For multi-component edges (e.g. `[bend, segment]`), `ComponentResult` walks the edge's components in flow order to find this component's endpoint pressures — so a `ComponentResult` on the bend reports the pressure at the bend inlet / outlet, not at the whole edge's inlet / outlet.

`Compressible_Network.solve()` currently returns a plain dict with `P_Pa` / `T_K` / `mdot_kgs` / `Q_ext_mdot_kgs` / `component_outlet_PT` / `converged` keys. The `component_outlet_PT` entry (added so the compressible GUI canvas can label per-block P / T without re-walking the chain itself — see the "Per-component outlet readout" subsection above) covers the most common per-component query. Lifting the full `NetworkResult` / `EdgeResult` / `ComponentResult` accessors to also wrap compressible results (with an additional `T_K` / `pressure_and_temperature_profile()` surface) is still on the open-work list.

---

## Three calculation modes — incompressible only (so far)

The general `Network.solve()` accepts any well-posed mix of P-spec and Q-spec boundary conditions. Three thin wrappers expose the common single-inlet/single-outlet patterns explicitly. Each one mutates only the two named nodes' boundary specs (other boundary conditions in the network are untouched) and returns a regular `NetworkResult`.

```python
# Mode 1 — find outlet P given inlet P and total Q.
result = net.solve_for_outlet_pressure(
    fluid, inlet="in", outlet="out",
    P_inlet=ureg.Quantity(200, "psi"),
    Q=ureg.Quantity(10000, "oil_bbl/day"),
)
P_out = result["P_Pa"]["out"]

# Mode 2 — find inlet P given outlet P and total Q.
result = net.solve_for_inlet_pressure(
    fluid, inlet="in", outlet="out",
    P_outlet=ureg.Quantity(50, "psi"),
    Q=ureg.Quantity(10000, "oil_bbl/day"),
)
P_in = result["P_Pa"]["in"]

# Mode 3 — find total Q given inlet P and outlet P.
result = net.solve_for_flow_rate(
    fluid, inlet="in", outlet="out",
    P_inlet=ureg.Quantity(200, "psi"),
    P_outlet=ureg.Quantity(50, "psi"),
)
mdot = result["mdot_ext_kgs"]["in"]   # positive = into network at inlet
# or volumetric:
Q_m3s = result["Q_ext_m3s"]["in"]
```

All three forward `**solve_kwargs` (P_init, verbose, xtol, maxfev) to `solve()`. For multi-inlet/multi-outlet topologies, the same wrappers still work but only touch two of the nodes — the rest keep their existing specs; the "answer" interpretation only makes sense when the named pair is the dominant flow path.

The compressible analogues do not exist yet (the third item in open work).

---

## Validation

`__main__` blocks contain self-tests in both modules.

**Incompressible** ([network.py](network.py)):

- `_test_single_segment` — one Line_Segment between an inlet (Q-spec) and outlet (P-spec); recovers the same dP as a direct call to `Line_Segment.dP()` (error ~1e-13 psi).
- `_test_parallel_two_branches` — two pipes in parallel; matches `parallel.parallel_incompressible()` on dP and flow fractions to 4 significant figures.
- `_test_screenshot_forward_PIPE5` — the DWSIM topology with outlet-15 at 40 psi and outlet-16 at 80 psi; PIPE-5 carries forward flow (SPL-2 → MIX-2).
- `_test_screenshot_reverse_PIPE5` — same topology with outlet-15 throttled to 175 psi; PIPE-5 reverses on its own, and the solver also figures out which connectors should reverse. Mass balance closes to ~1e-12 BPD.
- `_test_specflow` — DWSIM-screenshot topology driven by Q-spec'd inlets instead of P-spec'd ones.
- `_test_query_api` — exercises `result.component()` / `result.edge()` accessors on a small multi-component-edge network, including a Line_Segment that ends up reversed (so `pressure_profile()` is walked on the reversed shadow with distance 0 at the flow-inlet = nominal-outlet elevation).
- `_test_check_valve_forward_and_sealed` — forward CV flow matches the direct `dP()` calculation; a back-pressured CV edge seals at exactly zero flow (post-snap); an open CV converges from the worst-case `mdot_init_kgs=0` start; sealed `ComponentResult` semantics (pipe shows zero dP, the CV holds the full ΔP, `pressure_profile()` returns flat hydrostatic rows); a junction routes the full supply around a sealed branch; and the dead-leg indeterminate-pressure warning fires.
- `_test_three_modes` — runs the same physical scenario through all three mode wrappers and verifies round-trip consistency: Mode 1 → Mode 2 recovers the original P_in to ~1e-10 psi; Mode 3 recovers the original Q to ~1e-12 relative error.

**Compressible** ([compressible_network.py](compressible_network.py)):

- `_test_single_segment_forward` — matches a direct `dP_dT()` walk to ~7e-8 psi, 3e-9 K, 4e-13 kg/s.
- `_test_parallel_two_branches` — matches `parallel.parallel_compressible()` on outlet P, T, and flow fractions to ~1e-4.
- `_test_mixing_junction` — two streams at different temperatures (350 K and 280 K) merge into one outlet; the solver finds the mixed `T_MIX = 309.5 K`, weighted toward the cold stream because cold gas has higher density (hence higher mass flow for the same pipe geometry and pressure drive). Mass balance closes to 3e-12 kg/s and the enthalpy mixing balance to 5e-12 relative error.
- `_test_orifice_subsonic` — single-edge `Orifice` between a P/T-spec inlet and Q-spec outlet; recovers the direct forward `dP_dT()` walk's P/T/mdot to ~1e-9.
- `_test_orifice_choke` — same `Orifice` topology with a flow rate that chokes the trim; verifies `ChokedFlowError` is raised with `mdot_choked` within ~25% of an ideal-gas critical-ratio reference.
- `_test_inverse_single_relief_valve` — single `Valve` edge between two P-spec nodes (vessel → atmosphere); auto-detected as inverse-mode, recovers the same mdot as a direct forward `Valve.dP_dT()` call to ~1e-5.
- `_test_check_valve_forward` — forward flow through a single `CheckValve` edge matches a direct `dP_dT()` walk.
- `_test_check_valve_sealed` — CV edge between two P-spec nodes with `P_to > P_from` (the configuration that was "over-determined" under the old sealing-K model): converges warning-free with `mdot == 0.0` exactly, and `component_outlet_PT` shows the CV holding the full ΔP. Also covers the dead-leg case — a free node isolated between two sealed CVs triggers the indeterminate-P/T warning.
- `_test_check_valve_junction_sealed` — supply → junction → (withdrawal + CV edge holding back a 100 psi higher region through an interior junction). The CV edge is forward-mode, so this exercises the `walk_edge` sealed branch rather than the inverse one; the full supply routes through the withdrawal and the CV edge reports exactly zero flow.
- `_test_inverse_relief_from_junction` — three-node topology with an inlet feeding a junction, then two outgoing edges to P-spec outlets (main + relief). Both outgoing edges are auto-detected inverse-mode; the inlet edge stays forward. Mass balance at the junction closes to ~1e-6. **⚠️ Not currently passing** — the LM solve fails to converge (status 3, residual stalls; the run is also very slow). Confirmed pre-existing on committed `HEAD` (independent of the Fanno-choke / two-phase-floor work), so it reflects a latent convergence issue in the multi-inverse-edge junction case. To be investigated.

(The component-level `test_dmdot_dT_choke_raises` in [examples.py](examples.py) covers the `ChokedFlowError` clamp in isolation. An end-to-end network test of the clamp through `walk_edge_inverse` was attempted but dropped — large vessel→atmosphere pressure ratios drive the choked-branch adiabatic expansion deep into two-phase territory, and the LM solver's trial trajectory at such ratios occasionally probes thermodynamically infeasible (P, T) combinations during early iterations, which the energy-balance `_safe_update_PT` rejects with `CoolProp PT update failed` rather than degrading gracefully. Wrapping the energy-balance updates in a penalty-fallback try/except (analogous to the `walk_edge` pattern) is the natural fix; it's open work tied to running the network solver in cryogenic / near-cricondentherm regimes more broadly.)

---

## Limitations

- **Compressible: uniform composition only.** All streams must share the same composition. Heterogeneous mixing would require per-edge composition variables and composition-mixing equations at junctions.
- **Compressible: no heat transfer.** Every edge runs adiabatically; the `q_wall` parameter exposed by `dP_dT()` is not plumbed through the network solver.
- **Finite-difference Jacobian.** Fine for networks of tens of edges; a sparse analytic Jacobian would be needed beyond that.
- **Asymmetric user-defined components.** Any component class not derived from `Base_Line_Segment` or `Base_Contraction_Expansion` is treated as symmetric under flow reversal by `_reversed_component()`. New asymmetric component types would need to be added to the duck-typing dispatch.
- **Boundary-condition well-posedness.** Both solvers require at least one P-spec node; over-specified or under-specified systems (e.g., everywhere Q-spec, or pressures inconsistent with a steady state) will fail to converge or produce physically odd results (e.g., a "high-pressure outlet" effectively becoming an inlet — which is what the math says, but probably not what the user meant). For the compressible solver, any inlet with positive `Q_ext` must also specify `T`.
- **Compressible solver sensitivity to initial guess on junction networks.** A uniform scalar `mdot_init_kgs` is mass-imbalanced at any junction with unequal in/out edge counts; the trust-region solver can stall at the trivial all-zero-flow minimum. Pass a per-edge `dict` for those cases.
- **Dead legs behind sealed check valves have indeterminate P/T (warned).** A free node connected to the P-spec'd part of the network only through sealed CV edges has no equation pinning its pressure (or, compressible, temperature); both solvers warn post-solve and the reported values there should not be trusted. Fix topologically (add a spec, or accept the warning). A junction whose only inflow paths seal off while a `Q_ext` demands flow is genuinely infeasible and produces the ordinary "did not converge" warning.
- **`parallel.py` does not model check-valve sealing.** The two-node parallel solvers assume every branch carries positive flow; a CV in a parallel branch acts as a plain K fitting there. Use the network solvers for topologies where a branch may need to seal.
- **Compressible solver: T_spec'd nodes with multiple edges + reversed flow.** A node with `T_spec_K` is not in `t_free_idx`, so its energy balance is *not* enforced — its T is pinned at the spec value. If such a node has more than one connected edge and one of those edges happens to carry flow outward (e.g. the node is overall acting as a sink but one connection reversed), the outward walk starts from `T_spec` rather than the energy-balanced mixed T of the actual incoming streams, producing a thermodynamically inconsistent solution. Mitigation: the GUI enforces a "Source/Sink touches exactly one edge" rule in `gui/screens/network.py::_build_network()` so this configuration cannot be built from the canvas; headless callers that put `T_spec` on a multi-edge node should expect spurious outward enthalpies if any of those edges reverses.
- **Inverse-mode edges are forward-flow only.** An edge auto-detected as inverse-mode (downstream P-spec) cannot reverse — `walk_edge_inverse` returns a penalty `mdot_solved = 0` plus a once-per-solve `UserWarning` if the LM trial flips its mdot negative. If a network is configured such that a P-spec-downstream edge legitimately needs to carry flow in the reverse direction, the inverse residual won't capture that and the solver will report `mdot ≈ 0` with non-zero residual norm. Workaround: change the topology so the edge's downstream is a junction rather than a P-spec node, which keeps it on the forward-residual path.
- **`Contraction_Expansion.dmdot_dT` does not handle expansions.** Raises `NotImplementedError` on the expansion direction (kinetic recovery typically dominates the K-loss for a sharp diffuser, reversing the residual sign convention). An inverse-mode edge containing only an expansion is unusable. Wrap the expansion in a downstream frictional component if inversion is required.
- **No opt-out for inverse-mode auto-detection.** Every edge with a P-spec downstream node is dispatched through `walk_edge_inverse`. Networks where a P-spec-downstream edge would benefit from the forward path (e.g. diagnostic comparison, or a topology where the brentq's per-iteration kernel cost dominates over LM's FD cost) cannot opt out short of patching `inverse_of` manually. A `force_forward_edges=[...]` kwarg on `solve()` is the obvious extension when needed.

---

## Persistence — `Network.save` / `Network.load`

The headless solver supports JSON save/load so a built network can be
round-tripped without rebuilding from code.  The file format
(`*.hydnet.json`, current `SAVE_FORMAT_VERSION` = 1) lives in `network.py`;
component-level (de)serialization lives in `component_classes.py` on each
`Base_*` parent class.  Both regimes share the same format — only the
component classes instantiated at load time differ.

```python
# Incompressible:
net = Network()
net.add_node("in",  P=ureg.Quantity(200, "psi"))
net.add_node("out", Q_ext=ureg.Quantity(-1000, "oil_bbl/day"))
net.add_edge("PIPE-1", "in", "out", [seg, bend, valve])
net.save("scenario.hydnet.json")

net2 = Network.load("scenario.hydnet.json")     # picks incompressible.* classes
result = net2.solve(fluid)

# Compressible (subclass selects compressible_flow.* component classes):
gnet = Compressible_Network.load("scenario.hydnet.json")
```

Each `Network` subclass carries a `REGIME` class attribute (`"incompressible"`
or `"compressible"`) that the file's `regime` field is matched against at
load time — calling `Network.load()` on a compressible file (or vice
versa) raises with a hint to use the matching class.  `from_dict()` accepts
an explicit `component_classes={"line_segment": cls, ...}` override for
callers who need to mix in custom subclasses.

**Components.**  Each `Base_*` class has `to_dict()` / `from_dict()`:

- `Base_Line_Segment`: stores `roughness_m`, `noncircular`, `k_wall`, and
  **either** the full `profile` point list **or** a `csv_path` (and the
  loader re-reads the CSV via `from_csv()` so edits to the CSV propagate
  without re-saving the network).  CSV origin is tracked via a
  `_csv_path` attribute set by `from_csv()`.  `from_csv()` also accepts
  `downsample` (bool or float — max step [m]) and `elev_tol` (float [m]) to
  pass through to `downsample_profile()` before constructing the segment.
- `Base_Bend`: `Di_m`, `ang_deg`, `bend_dias`, `name`.
- `Base_Valve` / `Base_CheckValve`: `Di_m`, `K`, `name`.  The K-factor is
  the resolved Crane value; the original valve "type" (globe/gate/...)
  is not preserved at the Network level.
- `Base_Contraction_Expansion`: `Di_US_m`, `Di_DS_m`, `name`.

**Nodes.**  Per node the file records `name`, `elevation_m`, `P_Pa`,
`Q_ext` (as `{"magnitude": float, "unit": str}` so the original pint
unit survives), and — when compressible — `T_K`.

**`gui_extras` block.**  A top-level `gui_extras` field is preserved
verbatim through save/load.  The headless solver ignores it; the GUI
uses it to round-trip canvas positions and the original-unit display
strings (so a pipe entered as `"4.026 in"` doesn't come back as
`"0.10226 m"` after a load).  Headless `Network.load()` produces a
working network regardless of whether the field is present.

## Result bundles

`NetworkResult.save_bundle(dir_path)` (incompressible) and
`save_compressible_result_bundle(dir_path, net, result, AS, isothermal=...)`
(compressible, module-level since the compressible solver currently
returns a flat dict) write a directory of files matching the
GUI's "Save Results..." action:

```
<dir_path>/
    summary.json              # converged flag, per-node P/T/Q_ext, per-edge mdot
    <edge>__<pos>__<comp>.csv # one per Line_Segment-like component
```

Profile CSVs use whatever columns each component's profile generator
returns — `distance_m, elevation_m, P_Pa, v_ms` for the incompressible
side; `distance_m, P_Pa, T_K, v_ms` for compressible.  The compressible
re-walk builds a fresh `FlowState` at each component's converged
flow-direction inlet `(P, T)` (with `A` seeded from the component's
own `inlet_area_si` and `z` left at 0 since the bundle does not feed
elevation back through the network) and runs `Line_Segment.dP_dT(fs)`.

## Open work

In rough priority order:

1. **Lift the `NetworkResult` / `EdgeResult` / `ComponentResult` accessors to also wrap `Compressible_Network` results.** Adds `T_K` to the dict-style keys and `pressure_and_temperature_profile()` (or similar) to `ComponentResult` so the same query patterns work across both solvers.  Also folds `save_compressible_result_bundle` into `NetworkResult.save_bundle`.
2. **Three calculation-mode wrappers for the compressible solver.** Same shape as the incompressible ones, with `T` as an additional inlet spec. Note that the underlying solver already handles all three patterns via the boundary-condition mix (P-spec inlet + P-spec outlet auto-routes the outlet edge through `walk_edge_inverse` for the "find Q" case); these would be ergonomic wrappers, not new physics.
3. **Path-stitching for plotting.** Walk a chosen inlet → outlet route through the network, assemble a continuous `(distance, elevation, P, v[, T])` curve by concatenating consecutive `ComponentResult.pressure_profile()` outputs. Then a matplotlib helper to plot pressure-vs-distance.
4. **GUI.** A tree/list view of the network with add/remove/reorder for components, fluid-property editor, mode selector, results pane (per-component readout + path plot). Likely PySide6 if a topology canvas is wanted later, or Tkinter if the simplest path matters more.
5. **Heterogeneous-composition compressible support.** Per-edge composition variables, composition-mixing equations at junctions, CoolProp updates with composition changes. Substantial work; only needed when the network actually has multiple gas sources.
6. **Heat transfer.** Plumb `q_wall` through `Compressible_Network.solve()` so adiabatic isn't the only mode.
