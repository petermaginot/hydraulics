# Pipe-network solver — status

A general directed-graph solver for pipe networks built on top of the existing component classes in [incompressible.py](incompressible.py) and [compressible_flow.py](compressible_flow.py) (and the base geometry / CSV-loading code in [component_classes.py](component_classes.py)). Two solvers live alongside the existing per-component physics:

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

An **`Edge`** is a directed connection between two existing nodes, carrying a list of components evaluated in series in the `from_node -> to_node` direction. The list may be empty (a zero-pressure-drop connector that pins `P_from == P_to`) or contain one or more `Line_Segment` / `Bend` / `Valve` / `Contraction_Expansion` instances.

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

The combined system is solved with `scipy.optimize.fsolve` (Powell hybrid) on the concatenated `(free P, edge mdot)` vector. The Jacobian is built by finite differences — adequate for networks up to a few dozen edges; a sparse analytic Jacobian would be needed for larger systems.

After convergence, the external flow at each P-spec node is recovered from its mass balance and reported alongside the spec'd / interior values.

**Initial guesses:**
- Free node pressures = mean of P-spec values.
- Edge mdots = the largest spec'd `|Q_ext|` (in kg/s) by default, or 10 kg/s if no Q_ext is spec'd. This is intentionally **scale-matched** to the boundary flow: a too-small uniform guess produces a noisy finite-difference Jacobian (friction goes like mdot², so its derivative near zero is tiny), which can leave `fsolve` stuck.
- Near `|mdot| / rho < 1e-9 m³/s` (effectively zero flow), friction is linearized through zero to avoid the `friction_factor(Re=0)` singularity and to keep the residual smooth.

**Convergence** is declared if `fsolve` reports `ier == 1` **or** the residual L2 norm falls below `1e-4` (catches the case where `fsolve` reports `ier == 5` because it has hit floating-point noise — equation magnitudes here are pressures in Pa and mdots in kg/s, so 1e-4 covers both with margin).

At least one P-spec node is required to anchor pressures.

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
- **Pipe equation per edge** — one residual: walk the components in flow direction starting from the flow-inlet node's `(P, T)` via `dP_dT()` (which mutates the AS in place); the resulting walked outlet P must equal the flow-outlet node's P. The walked outlet T is *not* directly matched (because at a mixer the node T differs from any single edge's walked T); it feeds into the energy balance instead.
- **Mass balance** at each non-P-spec node (kg/s).
- **Energy balance** at each non-T-spec node:
  - inflows (edges + supply): contribute `|mdot| * h_inflow` where `h_inflow` is the walked outlet enthalpy of an incoming edge, or `h(P_n, T_spec)` for a supply boundary;
  - outflows (edges + withdrawal): contribute `|mdot| * h(P_n, T_n)` at the node's mixed state;
  - **including the implicit withdrawal at a P-spec outlet** — `Q_ext` is derived from mass balance (`Q_ext = -net_signed_mdot_in`) and added as an outflow. Without this term the equation collapses to "inflow enthalpy = 0", which only the trivial all-zero-flow solution satisfies, producing a spurious zero-flow attractor. This was the v1 bug that prevented mixing tests from converging until fixed.

**Residual scaling.** Pipe / mass / energy balance residuals are normalized to O(1) magnitude (divided by `P_init`, `mdot_ref`, and `mdot_ref * 1e6` respectively) so the Jacobian is not dominated by any one equation class.

**Solver.** `scipy.optimize.least_squares` with the Trust Region Reflective algorithm. TRF accepts variable bounds, which keep trial pressures positive and temperatures above absolute zero — otherwise the ill-conditioned early Newton steps drive `T` deeply negative and the CoolProp PT update fails. `x_scale='jac'` normalizes unknown magnitudes automatically from the Jacobian.

**Phase envelope.** Built once per `solve()` via `_build_phase_limits()` and forwarded to every `dP_dT()` call (rebuilding it is expensive for HEOS mixtures).

**Per-edge initial guess.** `mdot_init_kgs` accepts either a scalar (broadcast to every edge) or a `dict {edge_name: kg/s}`. For networks with mixer/splitter junctions, a uniform scalar is mass-imbalanced at the junction and can trap the solver at the zero-flow minimum; pass a mass-balanced dict to seed properly.

**Boundary-condition rule** (validated at solve time): every node with positive `Q_ext` (a supply into the network) must specify `T` so the inflow enthalpy can be evaluated.

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
- everything else (Bend, Valve) is treated as symmetric and returned unchanged.

The reversed copies are cached by `id()` of the original so they're built at most once per solve. The originals are never mutated.

For the incompressible solver: the reversed shadow's `dP()` at `+|mdot|` gives the dP in flow direction directly; negating restores the forward inlet-to-outlet convention. For the compressible solver: walking the reversed shadow at `+|mdot|` from the new-flow-inlet's `(P, T)` produces the right outlet state directly.

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

`Compressible_Network.solve()` currently returns a plain dict with `P_Pa` / `T_K` / `mdot_kgs` / `Q_ext_mdot_kgs` / `converged` keys. Lifting the same `NetworkResult` / `EdgeResult` / `ComponentResult` accessors to also wrap compressible results (with an additional `T_K` / `pressure_and_temperature_profile()` surface) is on the open-work list.

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
- `_test_three_modes` — runs the same physical scenario through all three mode wrappers and verifies round-trip consistency: Mode 1 → Mode 2 recovers the original P_in to ~1e-10 psi; Mode 3 recovers the original Q to ~1e-12 relative error.

**Compressible** ([compressible_network.py](compressible_network.py)):

- `_test_single_segment_forward` — matches a direct `dP_dT()` walk to ~7e-8 psi, 3e-9 K, 4e-13 kg/s.
- `_test_parallel_two_branches` — matches `parallel.parallel_compressible()` on outlet P, T, and flow fractions to ~1e-4.
- `_test_mixing_junction` — two streams at different temperatures (350 K and 280 K) merge into one outlet; the solver finds the mixed `T_MIX = 309.5 K`, weighted toward the cold stream because cold gas has higher density (hence higher mass flow for the same pipe geometry and pressure drive). Mass balance closes to 3e-12 kg/s and the enthalpy mixing balance to 5e-12 relative error.

---

## Limitations

- **Compressible: uniform composition only.** All streams must share the same composition. Heterogeneous mixing would require per-edge composition variables and composition-mixing equations at junctions.
- **Compressible: no heat transfer.** Every edge runs adiabatically; the `q_wall` parameter exposed by `dP_dT()` is not plumbed through the network solver.
- **Finite-difference Jacobian.** Fine for networks of tens of edges; a sparse analytic Jacobian would be needed beyond that.
- **Asymmetric user-defined components.** Any component class not derived from `Base_Line_Segment` or `Base_Contraction_Expansion` is treated as symmetric under flow reversal by `_reversed_component()`. New asymmetric component types would need to be added to the duck-typing dispatch.
- **Boundary-condition well-posedness.** Both solvers require at least one P-spec node; over-specified or under-specified systems (e.g., everywhere Q-spec, or pressures inconsistent with a steady state) will fail to converge or produce physically odd results (e.g., a "high-pressure outlet" effectively becoming an inlet — which is what the math says, but probably not what the user meant). For the compressible solver, any inlet with positive `Q_ext` must also specify `T`.
- **Compressible solver sensitivity to initial guess on junction networks.** A uniform scalar `mdot_init_kgs` is mass-imbalanced at any junction with unequal in/out edge counts; the trust-region solver can stall at the trivial all-zero-flow minimum. Pass a per-edge `dict` for those cases.

---

## Open work

In rough priority order:

1. **Lift the `NetworkResult` / `EdgeResult` / `ComponentResult` accessors to also wrap `Compressible_Network` results.** Adds `T_K` to the dict-style keys and `pressure_and_temperature_profile()` (or similar) to `ComponentResult` so the same query patterns work across both solvers.
2. **Three calculation-mode wrappers for the compressible solver.** Same shape as the incompressible ones, with `T` as an additional inlet spec.
3. **Path-stitching for plotting.** Walk a chosen inlet → outlet route through the network, assemble a continuous `(distance, elevation, P, v[, T])` curve by concatenating consecutive `ComponentResult.pressure_profile()` outputs. Then a matplotlib helper to plot pressure-vs-distance.
4. **CSV / JSON project I/O.** Save and load a network definition so users don't rebuild networks from code each time. Decide whether CSV-profiled Line_Segments are stored by path reference or embedded snapshot.
5. **GUI.** A tree/list view of the network with add/remove/reorder for components, fluid-property editor, mode selector, results pane (per-component readout + path plot). Likely PySide6 if a topology canvas is wanted later, or Tkinter if the simplest path matters more.
6. **Heterogeneous-composition compressible support.** Per-edge composition variables, composition-mixing equations at junctions, CoolProp updates with composition changes. Substantial work; only needed when the network actually has multiple gas sources.
7. **Heat transfer.** Plumb `q_wall` through `Compressible_Network.solve()` so adiabatic isn't the only mode.
