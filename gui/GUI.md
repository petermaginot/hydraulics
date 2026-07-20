# GUI — status

A PySide6 desktop application wrapping the hydraulics back end. Launch with `python run_gui.py` from the repo root. Four workflows live behind a single Start screen:

1. **Point-to-point incompressible** — one pipe segment, one liquid, one calculation.
2. **Point-to-point compressible** — one pipe segment, one gas mixture, one calculation.
3. **Pipe network (incompressible)** — a P&ID-style canvas with multiple sources/sinks, junctions, pipes, fittings, valves, and check valves, calling `Network.solve()` from [network.py](network.py).
4. **Pipe network (compressible)** — same canvas, with a composition / EOS panel and per-Source/Sink temperature spec, calling `Compressible_Network.solve()` from [compressible_network.py](compressible_network.py).

Dependencies (install with pip): `pyside6`, `pyqtgraph`, `nodegraphqt`, plus the existing physics deps (`pint`, `fluids`, `coolprop`).

---

## File layout

```
run_gui.py               — entry point: calls gui.main.run()
gui/
    __init__.py
    _compat.py           — distutils.version.LooseVersion shim for NodeGraphQt on Py >= 3.12
    main.py              — MainWindow + QStackedWidget routing between screens
    state.py             — AppState dataclass shared between screens
    units.py             — per-category unit-dropdown lists; "BBL/D" -> "oil_bbl/day" rewrite
    persistence.py       — canvas Save / Load wrapper around network.py's Network.save / load
    screens/
        start.py                  — radio-button selector + Next
        segment.py                — point-to-point pipe-segment builder (Manual / CSV tabs)
        fluid.py                  — point-to-point fluid + inlet conditions (incompressible & compressible)
        composition.py            — compressible-network composition / EOS / mode builder
        results.py                — pressure-profile plot for the point-to-point flow
        network.py                — node-graph canvas + incompressible network solver
        compressible_network.py   — subclass of network.py wired to Compressible_Network
```

---

## Shared state — [`AppState`](gui/state.py)

A single `AppState` instance is owned by `MainWindow` and handed to every screen. Screens read what they need, write their results back, and `reset_for_flow_type_change()` drops every regime-dependent field when the user picks a different option on the Start screen (so an incompressible solver never gets handed a compressible segment).

Fields:

| Field | Source screen | Notes |
| --- | --- | --- |
| `flow_type` | Start | `"incompressible"` / `"compressible"` / `"network"` |
| `segment` | Segment | `Line_Segment` of the right subclass |
| `fluid` | Fluid | `Incompressible_Fluid` or pre-updated CoolProp `AbstractState` |
| `flow_rate` | Fluid | pint Quantity (volumetric/mass/molar/std-vol); `None` in inverse mode |
| `P_inlet_Pa` | Fluid | inlet pressure anchor (Pa) |
| `T_inlet_K` | Fluid | inlet T (compressible only) |
| `isothermal` | Fluid | compressible calculation mode |
| `solve_mode` | Fluid | `"forward"` (user provides flow rate) or `"inverse"` (user provides outlet pressure) |
| `P_outlet_Pa` | Fluid | target outlet pressure for inverse mode (Pa); `None` in forward mode |
| `flow_rate_display_unit` | Fluid | dropdown label (e.g. `"BBL/D"`) used to format `solved_mdot_kgs` on the Results summary; `None` in forward mode |
| `solved_mdot_kgs` | Results | mass flow rate returned by `dmdot` / `dmdot_dT` after a successful inverse solve; `None` otherwise |
| `results` | Results | list of `{distance_m, P_Pa, v_ms, [T_K]}` |

The **network screen does not use `AppState`** beyond reading `flow_type` — it holds its own per-node spec dicts (`node_specs`) and last-solve cache (`_last_net`, `_last_result`) because its data model differs structurally from the linear flow.

---

## Start screen — [`start.py`](gui/screens/start.py)

Four radio choices ("Point-to-point: Incompressible (liquid)" / "Point-to-point: Compressible (gas)" / "Pipe network (incompressible)" / "Pipe network (compressible)") plus a Next button. The point-to-point options route to the Segment screen → Fluid → Results pipeline; the incompressible network option jumps straight to the Network screen; the compressible network option routes through the Composition screen first (to build the AbstractState) and then to the CompressibleNetworkScreen. Switching regime clears stale state via `AppState.reset_for_flow_type_change()`.

---

## Segment screen — [`segment.py`](gui/screens/segment.py) (point-to-point only)

Two tabs:

- **Manual** — fields for ID (or OD + WT), length, elevation change, roughness, each with a unit dropdown. Builds a 2-point `Base_Line_Segment` via the regime-appropriate `Line_Segment` subclass.
- **Load CSV profile** — file picker plus a roughness input. Reuses `Line_Segment.from_csv()`; the CSV header determines geometry mode (`ID`/`OD`/`WT` or `D_h`/`flow_area`).  A **Downsample profile** checkbox (off by default) enables profile thinning before the segment is built: when checked, **Max step** (default 1000 m) caps the spacing between retained points, and **Elev. tol.** (default 0.1 m) suppresses slope-break points whose elevation hasn't moved enough from the last retained point — useful for QGIS polyline exports where the quantized elevation model produces many nearly-redundant step boundaries.  Internally the checkbox calls `downsample_profile()` from [component_classes.py](component_classes.py).

After a successful build, the screen draws an **ID-vs-distance** chart (pyqtgraph) on the left Y axis and an **elevation-vs-distance** overlay on a linked secondary right-hand Y axis. The X axis is in feet, ID in inches, elevation in feet by default — `pyqtgraph`'s SI-prefix scaler is intentionally bypassed for engineering English. A summary line reports points / length / net elevation change.

The Next button enables only after a segment is successfully built; `showEvent` re-syncs the UI if the user came back from Start with cleared state.

---

## Fluid screen — [`fluid.py`](gui/screens/fluid.py) (point-to-point only)

A `QStackedWidget` swaps two panels keyed on `state.flow_type`. Both panels carry an **Operating mode** radio group at the top with two choices:

- **"Solve for pressure profile (given flow rate)"** (default) — the existing forward solve. The Results screen calls `pressure_profile()` (incompressible) or `dP_dT()` (compressible).
- **"Solve for flow rate (given outlet pressure)"** — inverse mode. An **Outlet pressure** field becomes enabled and the **Flow rate** value entry is disabled (its unit dropdown stays enabled — that unit becomes the display unit for the solved mdot on the Results summary). The Results screen calls `Line_Segment.dmdot()` (incompressible) or `Line_Segment.dmdot_dT()` (compressible).

Disabled fields keep their text, so toggling the radio back and forth doesn't lose user input. `_apply_incompressible_mode()` and `_apply_compressible_mode()` are the per-panel enable/disable toggles, fired on radio change.

**Incompressible panel.** A radio toggle picks Density or API gravity (the other gets disabled). Below that: viscosity, **inlet pressure** (default `0 psi` — always visible, both modes), **outlet pressure** (inverse only), and flow rate, all with unit dropdowns. The flow-rate dropdown carries both **volumetric** and **mass** units (the `Network` / single-component solvers convert either to mass internally). On Calculate, an `Incompressible_Fluid` is built and stashed alongside the flow rate (forward) or the outlet pressure (inverse); `P_inlet_Pa` carries the user-supplied value (the default `0 psi` preserves the previous gauge-0 behavior when the field is left untouched). Inverse mode validates `P_inlet > P_outlet` and raises an error dialog on violation.

**Compressible panel.** A `QTableWidget` of `(Component, Mole fraction)` rows seeded with Methane = 1.0. `+ Component` / `- Remove` buttons and a `Load CSV...` reader (`composition.parse_composition_csv`). The per-row component combo filters out names already in use elsewhere, so a 20-row table can't produce a duplicate. EOS dropdown (HEOS / PR), inlet P and T fields (with unit dropdowns), **outlet pressure** (inverse only), and a flow-rate field whose dropdown carries mass, molar, and standard-volume units (`mmscf/day`, `mol/s`, etc.). A radio group below selects **Adiabatic (energy balance)** or **Isothermal (constant T)** mode. On Calculate, `composition.define_composition(**kwargs)` builds the `AbstractState`, `AS.update(PT_INPUTS, P, T)` anchors it at the inlet, and the AS + (flow rate OR outlet pressure) + inlet (P, T) + isothermal flag + `solve_mode` are stashed on `AppState`. Inverse mode validates `P_inlet > P_outlet`.

---

## Results screen — [`results.py`](gui/screens/results.py) (point-to-point only)

Runs the calculation on `showEvent` and again on the Re-run button. The Fluid screen stashes `state.solve_mode` (`"forward"` or `"inverse"`) and `_run()` branches on it for each regime:

- **Incompressible, forward**: `state.segment.pressure_profile(fluid, P0, flow_rate)` returns the per-point list directly.
- **Incompressible, inverse**: `state.segment.dmdot(fluid, P_inlet, P_outlet)` returns mass flow rate; that value is stashed on `state.solved_mdot_kgs` and fed back through `pressure_profile()` (as a `kg/s` Quantity) to produce the curve to plot.
- **Compressible, forward**: re-anchors the AS with `AS.update(PT_INPUTS, P_inlet, T_inlet)` before each run (because `dP_dT` mutates AS in place), constructs a `FlowState(AS, mdot, A=segment.inlet_area_si, z=0)` from that anchor — `mdot` from `_resolve_mdot(flow_rate, AS)` — then calls `segment.dP_dT(fs, isothermal=...)` and normalizes the tuple-of-tuples return into the same dict shape used downstream.
- **Compressible, inverse**: same re-anchor, builds a `FlowState` with a placeholder `mdot` (overwritten by the solver), and calls `segment.dmdot_dT(fs, P2=P_outlet, isothermal=...)`. The solver returns the profile directly (same tuple shape as `dP_dT`); `fs.mdot` at return is the solved mass flow rate, which is stashed on `state.solved_mdot_kgs`.

The plot is a pressure-vs-distance curve (left Y axis, blue) with an X-axis distance-unit selector and Y-axis pressure-unit selector. **Compressible adds a secondary right-hand axis** (linked ViewBox) carrying the temperature trace, with its own unit selector. Unit-selector changes re-render against `state.results` without re-running the solver — the underlying data is always SI.

A monospace summary block above the plot reports length, inlet/outlet P, total dP, and (compressible) inlet/outlet T. When `state.solved_mdot_kgs` is set (inverse mode), an additional **"Solved flow"** row is appended, formatted via the module-level `_format_solved_mdot()` helper. The helper inspects the requested unit's pint dimensionality and converts from `kg/s` directly (mass), via `AS.molar_mass()` (molar / std-vol units, which are registered as mole equivalents in the unit registry), or via `AS.rhomass()` / `Incompressible_Fluid.density_si` (actual-volume units). The display unit comes from `state.flow_rate_display_unit` — the value the user had selected on the (now-disabled) flow-rate dropdown when they pressed Calculate.

---

## Network screen — [`network.py`](gui/screens/network.py) (incompressible only)

A P&ID-style canvas driven by `NodeGraphQt`. Six node families plus connecting edges:

- **`SourceSinkNode`** — boundary node carrying a `P` or `Q_ext` spec (mutually exclusive). Multi-in and multi-out ports so one node can serve as the inlet/outlet for several chains (manifold). Result widgets: solved `P`, `T` (compressible only — left blank otherwise), and external `Q`.
- **`JunctionNode`** — interior splitter/merger with no boundary condition; multi-in and multi-out ports. Result widgets: solved `P` and (compressible only) `T`.
- **`PipeSegmentNode`** — inline pipe segment with geometry. Strict single-input + single-output ports (no branching) — branching happens at junction nodes.
- **`FittingNode`** — inline bend or sudden contraction/expansion.
- **`ValveNode`** — inline valve (globe / gate / plug / ball / butterfly), K-factor computed from Crane correlations at solve time.
- **`CheckValveNode`** — inline check valve (swing / lift / tilting-disk / angle-stop / globe-stop), again K from Crane. Seals perfectly on reverse flow: the network solvers pin a check-valved edge to exactly zero reverse flow (complementarity residual — see [network.md](network.md) § "Reverse-flow handling").

The four inline node types each carry the same set of result widgets, stacked top-to-bottom: a primary readout (`dP` for incompressible, outlet `P` for compressible — the field is reused rather than renamed to keep serialized graphs portable), `T` (compressible only), and signed flow.

A "pipe" in the underlying `Network` solver is **not** one canvas node — it's the chain of inline nodes (pipe / fitting / valve / check-valve) found by walking from a boundary node's output through inline nodes until another boundary node is reached. The chain's component instances become the `components=[...]` list on a single solver `Edge`. This means a long manifold-to-manifold run with multiple discrete blocks appears on the canvas as several visible nodes but resolves to one solver edge. Per-block results are recovered via `result.component(seg)` for the incompressible solver, or — for the compressible solver — by indexing `result["component_outlet_PT"][edge_name][pos]`, where `pos` is the inline node's 0-based position within its edge's chain (tracked by the screen in `self._inline_chain_pos`).

### Layout

```
[+ Source/Sink] [+ Junction] [+ Pipe] [- Delete selected]    [Save...] [Load...] [Save Results...] [Solve]
+----------------------------------------------+--------------------------+
|                                              | Fluid (incompressible)   |
|                                              |   density / viscosity    |
|                                              +--------------------------+
|                                              | Selected node            |
|        NodeGraphQt canvas (3 : 1 split)      |   (stacked editor)       |
|                                              +--------------------------+
|                                              | Result display units     |
|                                              |   pressure / flow        |  ← incompressible only
|                                              +--------------------------+
|                                              | Results (text panel)     |
+----------------------------------------------+--------------------------+
[<- Back]
```

For the **compressible** network screen the "Result display units" row is absent — those combos (pressure / flow / temperature) live on the Composition screen to the right of the EOS + mode selectors, freeing the side panel for a taller Results text area. `main.py` wires the combos to `_rerender_with_current_units` after both screens are constructed.

### Selected-node editor

A `QStackedWidget` with one page per node family, automatically swapped via `node_selected`. Each page shows a Name field, the appropriate spec fields (with unit dropdowns), and an Apply button that writes back to a `node_specs[node_id]` dict.

- **Source/Sink editor** — `P`, `Q_ext`, elevation, and an optional **Diameter** field. The P / Q_ext fields are mutually exclusive: typing in one disables the other (`_update_PQ_exclusivity`), matching the solver's `P xor Q_ext` rule per node. The compressible subclass adds a `T` row (required on every Source/Sink even when it's currently a withdrawal — a future solve could reverse the flow and need an inflow T). The Diameter field is optional; when blank, the compressible solver defaults the node area to the first connected component's diameter (see [CLAUDE.md](CLAUDE.md) for the resolution rule). Setting it explicitly makes the node's (P, T) the static state at that area, so velocity head is referenced to the right diameter when fluid enters / leaves the node. **A Source/Sink must connect to exactly one edge** — `_build_network()` raises a `ValueError` naming the offending node if it touches zero or more than one headless edge. All mixing / splitting must happen at Junctions, whose T is back-solved by the energy balance; this prevents a thermodynamic inconsistency where one of a multi-edge Source/Sink's connections carries reversed flow and the outward walk would otherwise leave at `T_spec` rather than the actual mixed inlet T.
- **Junction editor** — elevation and an optional **Diameter** field. Elevation is consistency-checked at solve time against the accumulated `net_elevation_change_m` of each connected edge's line segments (see "Solving" below); a mismatch surfaces as a warning. Diameter follows the same default-from-connected-component rule as Source/Sink; specify it explicitly when the junction joins pipes of different diameters and you want the junction's solved (P, T) to be the static state at a chosen reference area.
- **Pipe Segment editor** — same shape as the point-to-point Segment screen's Manual tab (ID/OD/WT/length/dz/roughness). Below the geometry fields, **Max step** and **Elev. tol.** control profile thinning (enabled only when the **Downsample profile** checkbox is checked). The bottom two rows of the editor are: `[Load CSV...]  [□ Downsample profile]` and `[Apply]  [Switch to manual]`. **Load CSV...** switches the pipe into CSV mode (manual geometry fields go read-only, showing CSV-derived summaries); **Switch to manual** reverts. The downsample settings round-trip through Save / Load so the in-memory profile matches what the user originally configured.
- **Fitting / Valve / Check Valve editors** — type-specific geometry (ID, angle, R/D, D1/D2, etc.) feeding the corresponding `fluids.fittings.K_*_Crane` correlation at solve time.

### Solving

The Solve button:

1. Flushes any pending in-editor changes into `node_specs` (so the user doesn't have to press Apply first).
2. Builds an `Incompressible_Fluid` from the side panel (or, for the compressible subclass, pulls the AbstractState that the Composition screen stashed on `AppState.fluid`).
3. Translates the canvas into a `Network`: each boundary node becomes a `net.add_node(...)`; each chain of inline nodes between two boundary nodes becomes one `net.add_edge(name, from, to, [component, ...])`. `_walk_chain` does the chain detection (raising if a chain doesn't terminate at a boundary node or if a node is shared between chains) and additionally records each inline node's 0-based position in its edge into `self._inline_chain_pos` so the compressible renderer can look up per-block results.
4. Runs `net.solve(...)` on whichever solver is wired in via `self.NETWORK_CLS`.
5. Renders the result.

The whole build + solve sequence runs inside `warnings.catch_warnings(record=True)`; any `UserWarning` emitted by the solver (e.g. `Network._validate_edge_elevations` flagging an edge whose endpoint-node Δz disagrees with the sum of `component.net_elevation_change_m` on its line segments) is collected and surfaced after the result renders via `dialogs.warning("Solve warnings", …)` — so messages that would otherwise vanish into stderr land in front of the user.

Component classes are also class-attribute overrides (`LINE_SEGMENT_CLS`, `BEND_CLS`, `VALVE_CLS`, `CHECKVALVE_CLS`, `CONTRACTION_EXP_CLS`, `NETWORK_CLS`), so the compressible subclass swaps in `compressible_flow.*` and `Compressible_Network` without touching the editor stack, chain walker, or canvas plumbing.

### Rendering results

After a successful solve, results land in two places:

- **On the canvas** — each `SourceSinkNode` gets `P=… psi`, `T=…` (compressible only), and `Q=…` (flow into / out of the network). Each `JunctionNode` gets `P=…` and (compressible only) `T=…`. Each inline node gets a primary readout (`dP=…` for incompressible, outlet `P=…` for compressible — same widget field reused), `T=…` (compressible only), and signed `Q=…`. The incompressible per-block dP comes from `result.component(seg)`; the compressible per-block (P, T) comes from `result["component_outlet_PT"][edge_name][pos]` (see [network.md](network.md) for the solver side).
- **In the side text panel** — a tab-aligned summary: convergence flag, every node's P (and T for compressible), every edge's signed flow, every boundary node's external Q. Currently flow magnitudes near zero are suppressed in the external-flow listing.

The **Result display units** combos re-render both the canvas annotations and the text panel without re-solving. For the incompressible network the combos (Pressure, Flow) live in the side panel's "Result display units" group box. For the compressible network the combos (Pressure, Flow, Temperature) live on the Composition screen — see that section — and are connected to the same `_rerender_with_current_units` handler by `main.py`. The flow combo lists both volumetric and mass units; `_convert_flow` picks the right source (`mdot_kgs` or `Q_m3s`) by checking the unit's pint dimensionality against cached mass-flow and volumetric-flow signatures.

### Per-block inspector

Each inline editor page (Pipe / Fitting / Valve / Check Valve) carries a **"Show solved details…"** button that becomes visible only when (a) a solve has completed and (b) that node was assigned a component during chain-walk. Clicking it opens a modal `NodeResultsDialog` ([gui/dialogs.py](gui/dialogs.py)) with formatted label/value rows:

- **Pipe** — mass flow, inlet/outlet P, dP, inlet/outlet velocity (plus inlet/outlet T for compressible). A **"Plot profile…"** button on the dialog opens a non-modal `PipeProfileWindow` showing P vs distance; for compressible a secondary right-hand axis carries the T trace, with its own unit selector — same layout as the point-to-point Results screen.
- **Valve / Check Valve** — K-factor, mass flow, dP, and velocity through the **smallest** characteristic diameter (seat bore `D1` for reducer types, single pipe `D` for butterfly / swing / tilting-disk). Compressible adds inlet/outlet P + T rows.
- **Fitting** — mass flow, dP, velocity. Bend reports velocity at the pipe ID; sudden contraction/expansion reports velocity at `min(Di_US, Di_DS)` (the point of maximum velocity).

For the incompressible case all rows come from `result.component(comp)` — `pressure_profile()` already supplies per-point `v_ms` so the profile window consumes it directly. For the compressible case, inlet/outlet (P, T) come from `_inline_inlet_outlet_PT()`, which reads `result["component_outlet_PT"]` and walks one step back in flow direction (handling reverse flow); inlet density for velocity comes from a save / restore around `AS.update(PT_INPUTS, P, T)`. The compressible pipe profile is generated lazily on **Plot profile…** click by re-anchoring the AS to the inlet (P, T), building a fresh `FlowState(AS, mdot, A=seg.inlet_area_si, z=0)`, and running `compressible_flow.Line_Segment.dP_dT(fs)` — the returned `(distance, P, T, v)` tuples feed `PipeProfileWindow` directly.

---

## Composition screen — [`composition.py`](gui/screens/composition.py)

The first stop for the compressible-network workflow, reached from Start before the canvas. Collects the gas composition, the equation of state, and the result display units, then stashes an unanchored `AbstractState` on `AppState.fluid`.

**Left column — composition + EOS + mode:**

- A `QTableWidget` of `(Component, Mole fraction)` rows seeded with Methane = 1.0. `+ Component` / `- Remove` / `Load CSV...` buttons manage the table; per-row combo boxes filter out names already in use so duplicates can't form.
- EOS dropdown (`HEOS` / `PR`).
- **Adiabatic / Isothermal** radio buttons. Isothermal is currently disabled (the network solver only supports adiabatic; the radio is shown greyed so the limitation is visible up front).

**Right column — Result display units:**

Three dropdowns — Pressure, Flow (mass / molar / standard-volume), and Temperature — that control how solved results are rendered on the compressible network canvas and in its text panel. These combos (`self.d_pressure`, `self.d_flow`, `self.d_temperature`) are stored as instance attributes; `main.py` assigns them to the `CompressibleNetworkScreen` after construction and connects their `currentTextChanged` signals to `_rerender_with_current_units`, so changing a unit here immediately re-renders the last solve.

**Build →** validates the composition table, calls `composition.define_composition(**kwargs)`, stashes the resulting `AbstractState` (and the isothermal flag) on `AppState`, clears any stale point-to-point solver state, and navigates to the compressible network canvas.

---

## Compressible Network screen — [`compressible_network.py`](gui/screens/compressible_network.py)

`CompressibleNetworkScreen` subclasses `NetworkScreen` and overrides the regime-specific knobs:

- **Component classes** — `LINE_SEGMENT_CLS`, `BEND_CLS`, `VALVE_CLS`, `CHECKVALVE_CLS`, `CONTRACTION_EXP_CLS` all swap to their `compressible_flow.*` equivalents; `NETWORK_CLS` swaps to `Compressible_Network`. Because the base class's `_build_*_component` helpers reach for these via `self`, the editor stack, chain walker, and canvas rendering are entirely inherited.
- **Fluid panel** — replaced with a read-only label summarising the composition / EOS / mode that `CompressibleCompositionScreen` stashed on `AppState.fluid`. The Composition screen runs first in the start-screen routing for this regime.
- **Source/Sink editor** — adds a `T` row (the required-on-every-Source/Sink rule lives in `_extra_kwargs_for_boundary`).
- **Display units** — the Pressure / Flow / Temperature combos live on the Composition screen (see below), not in the network side panel. `main.py` assigns those combo widgets to `d_pressure`, `d_flow`, and `d_temperature` on this screen after both screens are constructed, and connects their `currentTextChanged` signals to `_rerender_with_current_units`. The side panel's display-units group box is replaced with a hidden zero-height placeholder.
- **Result rendering** — the compressible solver returns a flat dict; `_annotate_results_on_canvas` fills `P` + `T` + external `Q` on Source/Sink, `P` + `T` on Junction, and outlet `P` + `T` + edge `mdot` on inline blocks. The per-block (P, T) come from `result["component_outlet_PT"]`, indexed by each inline node's `self._inline_chain_pos`. Per-component `dP` is not separately surfaced (the user can read consecutive P values off the canvas instead). **For a Source/Sink acting as a sink** (`Q_ext_mdot_kgs[name] < 0`), the displayed `T` is the *arriving fluid* temperature, not the user's backup `T_spec` — `_sink_arriving_T_K` reads the flow-direction-last `(P, T)` from `result["component_outlet_PT"]` on the node's single connected edge (the degree-1 invariant enforced in `_build_network()` makes this unambiguous). When that edge has no components (a topology-only zero-dP connector), `_sink_arriving_T_K` falls back to the solved `T_K` at the node on the other end of the edge — the fluid arrives unchanged across a connector. Source-acting and zero-flow Source/Sinks still show their pinned `T_spec` (which IS the supply T). The Results text panel marks overridden values with a trailing `(arriving)` so the reader can tell spec vs. solved at a glance.
- **Solver progress** — `Compressible_Network.solve()` accepts a `progress_callback(nfev, residual_norm)` that fires from inside its `residuals()` after each evaluation. `CompressibleNetworkScreen._solve_network` overrides the base no-op to seed `results_text` with "Solving..." and pass `_on_solve_progress` into the solver. The callback throttles to ~150 ms, rewrites the panel with `nfev` / residual norm / elapsed seconds, and calls `QApplication.processEvents()` so the UI repaints mid-solve. The base `NetworkScreen._solve` wraps the solve in a `try/finally` that disables `self._solve_btn` for the duration — necessary because `processEvents()` would otherwise let a second click re-enter `_solve`. The final `_render_results_text` overwrites the progress line with the normal result table.

The flow-rate dropdown is mass / molar / standard-volume only — actual-volumetric units are omitted because density varies along the network and there's no single conversion.

---

## Units module — [`units.py`](gui/units.py)

Centralizes the dropdown choices and the one rename pint can't handle out of the box:

- `BBL/D` — pint's `bbl` was deleted (see [component_classes.py](component_classes.py)) to prevent the 31.5-gal "US fluid barrel" footgun; the GUI shows the engineering-standard `BBL/D` label and `to_pint()` translates it back to `oil_bbl/day` (42 gal) before any pint call.
- `FLOW_RATE_INCOMPRESSIBLE` lists both volumetric (`m^3/s`, `m^3/h`, `gal/min`, `BBL/D`, `ft^3/s`) and mass (`kg/s`, `kg/h`, `lb/h`) units — both work because the network solver converts internally.
- `FLOW_RATE_COMPRESSIBLE` adds standard-volume (`mmscf/day`, `mscf/day`, `scf/min`, `scm/h`) and molar (`mol/s`, `mol/h`) units; actual volumetric is still listed for the point-to-point solver but is rejected by the compressible *network* solver (density varies, so there's no single conversion).

---

## Compatibility shim — [`_compat.py`](gui/_compat.py)

`NodeGraphQt` 0.6.44 still imports `distutils.version.LooseVersion`, which Python 3.12 dropped. `gui._compat` installs a stub backed by `packaging.version.Version` before any NodeGraphQt import resolves the name. The Network screen imports it at the top of `network.py` (`import gui._compat`); this needs to stay first.

---

## Save / Load / Save Results — [`gui/persistence.py`](gui/persistence.py)

Three toolbar buttons (**Save...**, **Load...**, **Save Results...**) sit
between the node-add buttons and **Solve** on both the incompressible and
compressible network screens.  The file format itself lives in
[network.py](network.py) (`Network.save` / `Network.load` /
`NetworkResult.save_bundle`) so headless code can produce and consume the
same files — see [network.md](network.md).  `gui/persistence.py` is the
thin canvas-side wrapper.

**Save** flushes any pending in-editor edits, then calls
`_build_network()` so a malformed canvas raises before anything hits
disk.  It then calls `Network.save(path, gui_extras=...)`.  The
`gui_extras` block carries the canvas-only state the headless format
does not represent: per-canvas-node position, the canvas id (used only to
wire connections within the file), the raw `node_specs` dict (which
preserves the user's input units and the OD/WT-vs-ID and valve-type
intent that gets flattened on the network side), and the canvas
connections by port name.  CSV-backed pipes have their `csv_path`
rewritten relative to the save file in `gui_extras` so the bundle is
portable as long as the CSVs move with it.  If the user had downsampling
enabled on a CSV pipe, `downsample_max_step_m` and `downsample_elev_tol_m`
are also stored in the spec dict so **Load** can re-apply the identical
`downsample_profile()` call and reproduce the same reduced profile.

**Load** reads the JSON, validates same-regime files by calling
`screen.NETWORK_CLS.from_dict(payload)` (catches malformed components,
unknown kinds, etc.), then rebuilds the canvas from `gui_extras`.
Cross-regime loads are supported: loading an incompressible save into the
compressible canvas merges in the compressible Source/Sink defaults (so
`T_str` / `T_unit` exist with blank values) and surfaces a warning
listing every Source/Sink that needs `T` filled in before the next
solve.  Files written without a `gui_extras` block (headless-only saves)
currently warn and leave the canvas empty — full canvas reconstruction
from the headless representation alone is on the open-work list.

**Save Results** is disabled until a successful solve, then opens a
directory picker and dispatches to `NetworkResult.save_bundle()` for the
incompressible side or to
`compressible_network.save_compressible_result_bundle()` for the
compressible side.  See [network.md](network.md) for the bundle layout
(`summary.json` plus one per-pipe profile CSV per Line_Segment-like
component on each edge).

The fluid / composition panel state is **not** included in save files by
design — the user re-enters density/viscosity or rebuilds the
AbstractState each session.  Loading a saved network does not touch the
side-panel fluid box.

---

## Things missing / next steps

In rough priority order:
1. **Three calculation-mode wrappers in the UI.** `Network.solve_for_outlet_pressure` / `_inlet_pressure` / `_flow_rate` aren't exposed as explicit modes; the user emulates them by choosing P-spec vs Q_ext-spec on individual boundary nodes. A mode selector with named labels would be more discoverable. Compressible analogues don't exist on the back end yet (see [network.md](network.md) open work).
2. **Multi-block path plot.** The per-block inspector plots a single pipe's profile; it doesn't yet stitch a continuous P-vs-distance (or P + T-vs-distance for compressible) curve across multiple inline blocks along a chosen inlet→outlet path through the graph.
3. **Fluid/composition autosave alongside the network.** Save/Load is topology-only by design; an opt-in "save fluid panel too" toggle would let users round-trip a full runnable scene without re-entering density/viscosity or rebuilding the AbstractState.
4. **Canvas reconstruction from headless saves.** A file produced by `Network.save()` headless (no `gui_extras`) currently loads to an empty canvas with a warning.  Auto-layout + sensible default unit choices would let the GUI open files written by code.

