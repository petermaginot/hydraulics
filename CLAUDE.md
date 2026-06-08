# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Planning
Large changes to the code should be planned before any editing takes place. In plan mode, only describe what you would do. Never make actual file edits.

## Reference documents

The repo already carries dense design docs. Read them before making architectural changes — don't duplicate their content here.

- [README.md](README.md) — per-module overview, public API for each component class.
- [network.md](network.md) — `Network` / `Compressible_Network` solver internals: residual form, reverse-flow handling, check-valve treatment, save/load format, open work.
- [GUI.md](GUI.md) — PySide6 GUI architecture, screen routing, `AppState`, NodeGraphQt canvas, persistence.
- [troubleshooting.md](troubleshooting.md) — symptom-first diagnostic guide for solver/GUI runs.
- [improvements.md](improvements.md) - list if identified improvements to make

 When referencing external sources, do not use paywalled articles or sources. All sources must be openly accessible and directly referenced with links.

## Running things

- **GUI:** `python run_gui.py` from repo root (PySide6 + NodeGraphQt + pyqtgraph required).
- **Headless examples / textbook validation:** functions in [textbook_test_functions.py](textbook_test_functions.py) and [examples.py](examples.py) are run directly (no pytest harness — call the function from a script or REPL).
- **Solver self-tests:** the `if __name__ == "__main__":` blocks at the bottom of [network.py](network.py) and [compressible_network.py](compressible_network.py) run the suites described in `network.md` § Validation. Invoke with `python network.py` / `python compressible_network.py`.

There is no build step, no lint config, and no test runner — validation is by running the textbook/self-test functions and comparing to documented expected values.

## Architecture in one screen

Three layers, each depending only on the ones below it:

1. **Geometry + I/O** — [component_classes.py](component_classes.py). `Base_Line_Segment` / `Base_Bend` / `Base_Valve` / `Base_CheckValve` / `Base_Contraction_Expansion`. Pure dimension storage + CSV loading + `to_dict`/`from_dict`. No physics. Also registers custom pint units (`scm`, `scf`, `mscf`, `mmscf` as mole equivalents) and **deletes pint's `bbl`** (use `oil_bbl/day` = 42 gal).
2. **Per-component physics** — [incompressible.py](incompressible.py) (`dP()` methods) and [compressible_flow.py](compressible_flow.py) (`dP_dT()` methods). The compressible methods **mutate the passed `AbstractState` in place** as conditions evolve. Subclasses inherit geometry from the Base_* classes 1:1.
3. **Topology solvers** — [parallel.py](parallel.py) for two-node parallel branches, [network.py](network.py) / [compressible_network.py](compressible_network.py) for arbitrary directed graphs. The compressible network re-uses the incompressible `Node` / `Edge` data classes and most of the graph machinery; it swaps in a joint `(P, T, mdot)` Newton residual and the compressible component classes.

Side modules: [composition.py](composition.py) (build `AbstractState` from mole fractions), [parallel.py](parallel.py) (flow-split Newton), [heat.py](heat.py) (standalone Gnielinski calculator, not plumbed into the network solver), [utilities.py](utilities.py).

The GUI (`gui/`) is a thin shell over layers 2/3. Each screen reads/writes a shared `AppState` dataclass; the network screens hold their own `node_specs` dict. Component classes are class-attribute overrides on `NetworkScreen` so the compressible subclass swaps in `compressible_flow.*` without touching the canvas plumbing. NodeGraphQt 0.6.44 needs the `distutils.version.LooseVersion` shim in [gui/_compat.py](gui/_compat.py) on Python ≥ 3.12 — keep `import gui._compat` first in [gui/screens/network.py](gui/screens/network.py).

## Conventions that aren't obvious from the code

- **Reverse flow reverses geometry, never signs.** A component on an edge with negative mdot is evaluated against a *reversed shadow* of itself (profile flipped, `Di_US`/`Di_DS` swapped, check-valve `K → _SEALING_K`). Do **not** patch new asymmetric components with `sign(Q)*friction(|Q|)` shortcuts — extend `_reversed_component()` in [network.py](network.py) instead. See `network.md` § "Reverse-flow handling".
- **Prefer comprehensibility over micro-optimization.** When there's a trade-off, flag it explicitly and recommend the simpler option. Hot paths (e.g. `compressible_pipe_segment`, `_safe_update_PT`) are the documented exceptions.
- **Pint at boundaries, SI floats internally.** Public methods accept pint Quantities; internal storage and math is plain SI floats (`_si`-suffixed attributes). `_to_si()` in [component_classes.py](component_classes.py) does the conversion.
- **Compressible layer takes a `FlowState`, not a bare `AbstractState`.** Every public function in [compressible_flow.py](compressible_flow.py) (`compressible_K`, `compressible_changing_area_K`, `compressible_changing_area`, `compressible_pipe_segment`, `choked_mass_flux`) and every component `dP_dT(fs)` consumes a `FlowState` bundling `(AS at static, mdot, A_local, z, cached phase-envelope limits)`. `v`, `Ma`, and stagnation properties are derived from `fs` on access. AS is always at the **static** thermodynamic state by convention — code that reads `AS.hmass()` and labels it `h0` (stagnation) is a bug; use `fs.h_stagnation = h_static + 0.5*v²` instead.
- **Compressible `dP_dT` mutates the `FlowState` in place.** `fs.AS` advances to outlet static. `fs.A` is rewritten when the function changes the flow geometry (area-change boundaries, contraction/expansion). `fs.z` is advanced by `compressible_pipe_segment` so gravity bookkeeping stays current. Callers that re-invoke (parallel-branch iteration, GUI re-runs) must re-anchor `fs.AS` with `AS.update(CP.PT_INPUTS, P_in, T_in)` and rebuild the `FlowState` for each pass; the network solver does both automatically.
- **Components auto-absorb area discontinuities.** Each `dP_dT` calls `_area_match(fs, self.inlet_area_si)` at entry, which runs `compressible_changing_area_K(K=0)` if `fs.A` doesn't already match the component's inlet area. So a `4-in valve → 3-in pipe` chain doesn't need an explicit `Contraction_Expansion` between them. Side effect: every Base class exposes `inlet_area_si` / `outlet_area_si` properties; new asymmetric components must override both, and `_reversed_component` shadows pick them up automatically through swapped underlying geometry. The same `_area_match` also covers the node↔first-component transition: `Compressible_Network.solve` seeds each edge's `FlowState.A` from the inlet node's resolved area (see next bullet), so a node whose diameter differs from the first connected component just produces an isentropic area change at edge entry rather than crashing.
- **Source/Sink and Junction nodes carry an optional `area_m2`.** Pass `diameter=` or `area=` to `add_node` (Quantity or SI float). The compressible solver interprets each node's (P, T) as the static state at that area, so velocity head bookkeeping at the node is referenced to the right diameter. Default `None` → resolved at solve time: first connected edge's first component near-end area, with BFS through zero-component edges to propagate (so a Source connected by a bare connector edge to a Junction inherits the Junction's resolved area). A node with no resolvable area anywhere raises a clear error. Empty-component edges are explicitly legal — they're zero-dP connectors that pin `P_from == P_to` (see [network.py](network.py)'s `add_edge` docstring).
- **At least one P-spec node** is required to anchor any `Network.solve()`. Bare-float `Q_ext` defaults to **kg/s** (not whatever the GUI dropdown reads). Compressible solver also rejects actual-volumetric flow units and requires `T` on any node with positive `Q_ext`.
- **Phase envelope is expensive.** Build it once via `_build_phase_limits(AS)` and pass into the `FlowState` constructor; every `compressible_flow.*` function reads it back off `fs` (no more `T_cricondentherm=` / `P_cricondenbar=` kwargs threaded through every signature). Use `_safe_flowstate_update_PT(fs, P, T)` for in-place AS updates that need the supercritical phase hint. CoolProp's `build_phase_envelope` also corrupts the AS's solver state — do it on a temporary copy (`_build_phase_limits` already handles this).
- **`Network.save()` files (`*.hydnet.json`)** carry an optional `gui_extras` block for canvas positions / original unit strings. The headless solver ignores it; the GUI relies on it. CSV-backed pipes store paths relative to the save file in `gui_extras` and absolute in the top-level component dict — either is acceptable on load.

## When adding components or solver features

- New component class → derive from the appropriate `Base_*`, add `to_dict`/`from_dict` round-tripping, extend `_reversed_component()` if the component is asymmetric under flow reversal.
- New residual term in the network solver → keep it normalized to O(1) (`P_init`, `mdot_ref`, `mdot_ref * 1e6` are the existing scales), and add a self-test to the `__main__` block of the relevant network module.
- Don't introduce a pytest dependency just to wrap existing self-tests — the project deliberately runs validations as plain functions.
