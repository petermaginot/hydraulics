# GUI — status

A PySide6 desktop application wrapping the hydraulics back end. Launch with `python run_gui.py` from the repo root. Three workflows live behind a single Start screen:

1. **Point-to-point incompressible** — one pipe segment, one liquid, one calculation.
2. **Point-to-point compressible** — one pipe segment, one gas mixture, one calculation.
3. **Pipe network (incompressible)** — a P&ID-style canvas with multiple sources/sinks, junctions, and pipe segments, calling `Network.solve()` from [network.py](network.py).

Compressible *network* mode is the obvious next step; it isn't in the Start screen yet.

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
    screens/
        start.py         — radio-button selector + Next
        segment.py       — point-to-point pipe-segment builder (Manual / CSV tabs)
        fluid.py         — point-to-point fluid + inlet conditions (incompressible & compressible)
        results.py       — pressure-profile plot for the point-to-point flow
        network.py       — node-graph canvas + incompressible network solver
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
| `flow_rate` | Fluid | pint Quantity (volumetric/mass/molar/std-vol) |
| `P_inlet_Pa` | Fluid | inlet pressure anchor (Pa) |
| `T_inlet_K` | Fluid | inlet T (compressible only) |
| `isothermal` | Fluid | compressible calculation mode |
| `results` | Results | list of `{distance_m, P_Pa, v_ms, [T_K]}` |

The **network screen does not use `AppState`** beyond reading `flow_type` — it holds its own per-node spec dicts (`node_specs`) and last-solve cache (`_last_net`, `_last_result`) because its data model differs structurally from the linear flow.

---

## Start screen — [`start.py`](gui/screens/start.py)

Three radio choices ("Point-to-point: Incompressible (liquid)" / "Point-to-point: Compressible (gas)" / "Pipe network (incompressible)") plus a Next button. The point-to-point options route to the Segment screen → Fluid → Results pipeline; the network option jumps straight to the Network screen. Switching regime clears stale state via `AppState.reset_for_flow_type_change()`.

---

## Segment screen — [`segment.py`](gui/screens/segment.py) (point-to-point only)

Two tabs:

- **Manual** — fields for ID (or OD + WT), length, elevation change, roughness, each with a unit dropdown. Builds a 2-point `Base_Line_Segment` via the regime-appropriate `Line_Segment` subclass.
- **Load CSV profile** — file picker plus a roughness input. Reuses `Line_Segment.from_csv()`; the CSV header determines geometry mode (`ID`/`OD`/`WT` or `D_h`/`flow_area`).

After a successful build, the screen draws an **ID-vs-distance** chart (pyqtgraph) on the left Y axis and an **elevation-vs-distance** overlay on a linked secondary right-hand Y axis. The X axis is in feet, ID in inches, elevation in feet by default — `pyqtgraph`'s SI-prefix scaler is intentionally bypassed for engineering English. A summary line reports points / length / net elevation change.

The Next button enables only after a segment is successfully built; `showEvent` re-syncs the UI if the user came back from Start with cleared state.

---

## Fluid screen — [`fluid.py`](gui/screens/fluid.py) (point-to-point only)

A `QStackedWidget` swaps two panels keyed on `state.flow_type`.

**Incompressible panel.** A radio toggle picks Density or API gravity (the other gets disabled). Below that: viscosity and flow rate, both with unit dropdowns. The flow-rate dropdown carries both **volumetric** and **mass** units (the `Network` / single-component solvers convert either to mass internally). On Calculate, an `Incompressible_Fluid` is built and stashed alongside the flow rate; `P_inlet_Pa` is anchored at gauge 0.

**Compressible panel.** A `QTableWidget` of `(Component, Mole fraction)` rows seeded with Methane = 1.0. `+ Component` / `- Remove` buttons and a `Load CSV...` reader (`composition.parse_composition_csv`). The per-row component combo filters out names already in use elsewhere, so a 20-row table can't produce a duplicate. EOS dropdown (HEOS / PR), inlet P and T fields (with unit dropdowns), and a flow-rate field whose dropdown carries mass, molar, and standard-volume units (`mmscf/day`, `mol/s`, etc.). A radio group below selects **Adiabatic (energy balance)** or **Isothermal (constant T)** mode. On Calculate, `composition.define_composition(**kwargs)` builds the `AbstractState`, `AS.update(PT_INPUTS, P, T)` anchors it at the inlet, and the AS + flow rate + inlet (P, T) + isothermal flag are stashed on `AppState`.

---

## Results screen — [`results.py`](gui/screens/results.py) (point-to-point only)

Runs the calculation on `showEvent` and again on the Re-run button.

- **Incompressible** path: `state.segment.pressure_profile(fluid, P0, flow_rate)` returns the per-point list directly.
- **Compressible** path: re-anchors the AS with `AS.update(PT_INPUTS, P_inlet, T_inlet)` before each run (because `dP_dT` mutates AS in place), then calls `dP_dT(AS, flow_rate, isothermal=...)` and normalizes the tuple-of-tuples return into the same dict shape used downstream.

The plot is a pressure-vs-distance curve (left Y axis, blue) with an X-axis distance-unit selector and Y-axis pressure-unit selector. **Compressible adds a secondary right-hand axis** (linked ViewBox) carrying the temperature trace, with its own unit selector. Unit-selector changes re-render against `state.results` without re-running the solver — the underlying data is always SI.

A monospace summary block above the plot reports length, inlet/outlet P, total dP, and (compressible) inlet/outlet T.

---

## Network screen — [`network.py`](gui/screens/network.py) (incompressible only)

A P&ID-style canvas driven by `NodeGraphQt`. Three node families plus connecting edges:

- **`SourceSinkNode`** — boundary node carrying a `P` or `Q_ext` spec (mutually exclusive). Multi-in and multi-out ports so one node can serve as the inlet/outlet for several chains (manifold).
- **`JunctionNode`** — interior splitter/merger with no boundary condition; multi-in and multi-out ports.
- **`PipeSegmentNode`** — inline pipe segment with geometry. Strict single-input + single-output ports (no branching) — branching happens at junction nodes.

A "pipe" in the underlying `Network` solver is **not** one canvas node — it's the chain of `PipeSegmentNode`s found by walking from a boundary node's output through pipe nodes until another boundary node is reached. The chain's `Line_Segment` instances become the `components=[...]` list on a single solver `Edge`. This means a long manifold-to-manifold run with multiple discrete pipe sections appears on the canvas as several visible blocks but resolves to one solver edge, and the per-block dP / Q are recovered via `result.component(seg)` after the solve.

### Layout

```
[+ Source/Sink] [+ Junction] [+ Pipe] [- Delete selected]              [Solve]
+----------------------------------------------+--------------------------+
|                                              | Fluid (incompressible)   |
|                                              |   density / viscosity    |
|                                              +--------------------------+
|                                              | Selected node            |
|        NodeGraphQt canvas (3 : 1 split)      |   (stacked editor)       |
|                                              +--------------------------+
|                                              | Result display units     |
|                                              |   pressure / flow        |
|                                              +--------------------------+
|                                              | Results (text panel)     |
+----------------------------------------------+--------------------------+
[<- Back]
```

### Selected-node editor

A `QStackedWidget` with four pages keyed on node type, automatically swapped via `node_selected`. Each page shows a Name field, the appropriate spec fields (with unit dropdowns), and an Apply button that writes back to a `node_specs[node_id]` dict.

- **Source/Sink editor** — `P`, `Q_ext`, and elevation. The P / Q_ext fields are mutually exclusive: typing in one disables the other (`_update_PQ_exclusivity`), matching the solver's `P xor Q_ext` rule per node.
- **Junction editor** — just elevation (decorative).
- **Pipe Segment editor** — same shape as the point-to-point Segment screen's Manual tab (ID/OD/WT/length/dz/roughness) plus a **Load CSV...** button that switches the pipe into CSV mode. In CSV mode the manual geometry fields go read-only and display CSV-derived summaries (first/last ID, total length, net dz); a **Switch to manual** button reverts.

### Solving

The Solve button:

1. Flushes any pending in-editor changes into `node_specs` (so the user doesn't have to press Apply first).
2. Builds an `Incompressible_Fluid` from the side panel.
3. Translates the canvas into a `Network`: each boundary node becomes a `net.add_node(...)`; each chain of pipe-segment nodes between two boundary nodes becomes one `net.add_edge(name, from, to, [Line_Segment, ...])`. `_walk_chain` does the chain detection and raises if a pipe chain doesn't terminate at a boundary node or if a pipe is shared between chains.
4. Runs `net.solve(fluid)`.
5. Renders the result.

### Rendering results

After a successful solve, results land in two places:

- **On the canvas** — each `SourceSinkNode` gets `P=… psi` and `Q=… BBL/D` (flow into / out of the network). Each `JunctionNode` gets `P=… psi`. Each `PipeSegmentNode` gets `dP=… psi` and `Q=… BBL/D` (signed, per the edge's nominal direction). The latter comes from `result.component(seg)` — `ComponentResult` walks the edge in flow direction internally so each block reports its own dP, not the whole chain's.
- **In the side text panel** — a tab-aligned summary: convergence flag, every node's P, every edge's signed flow, every boundary node's external Q. Currently flow magnitudes near zero are suppressed in the external-flow listing.

The **Result display units** combos (pressure, flow) re-render both the canvas annotations and the text panel without re-running the solver. The flow combo lists both volumetric and mass units; `_convert_flow` picks the right source (`mdot_kgs` or `Q_m3s`) by checking the unit's pint dimensionality against cached mass-flow and volumetric-flow signatures.

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

## Things missing / next steps

In rough priority order:

1. **Compressible network on the canvas.** A `Compressible_Network` variant of the network screen would need a composition / EOS panel (same shape as the point-to-point fluid screen's compressible side), per-node `T` spec on the Source/Sink editor, and a mass-flow-aware display. The back end already exists in [compressible_network.py](compressible_network.py).
2. **Three calculation-mode wrappers in the UI.** `Network.solve_for_outlet_pressure` / `_inlet_pressure` / `_flow_rate` aren't exposed as explicit modes; the user emulates them by choosing P-spec vs Q_ext-spec on individual boundary nodes. A mode selector with named labels would be more discoverable.
4. **Path plot.** The Network screen reports per-pipe dP numerically; it doesn't yet plot a continuous P-vs-distance curve for a chosen inlet→outlet path through the graph (the natural compressible-network analogue would be P+T-vs-distance).
5. **Project save/load.** The canvas state isn't persisted; closing the app drops every node and connection. `NodeGraphQt` ships a session-save mechanism that can be paired with a JSON of the `node_specs` dicts.
6. **Validation tab / per-pipe inspector.** The canvas annotations are concise; a more detailed inspector showing the full pressure profile of a selected pipe (matching the point-to-point Results screen) would be useful for diagnostics.
