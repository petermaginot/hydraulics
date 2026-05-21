# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reference documents

The repo already carries dense design docs. Read them before making architectural changes — don't duplicate their content here.

- [README.md](README.md) — per-module overview, public API for each component class.
- [network.md](network.md) — `Network` / `Compressible_Network` solver internals: residual form, reverse-flow handling, check-valve treatment, save/load format, open work.
- [GUI.md](GUI.md) — PySide6 GUI architecture, screen routing, `AppState`, NodeGraphQt canvas, persistence.
- [troubleshooting.md](troubleshooting.md) — symptom-first diagnostic guide for solver/GUI runs.

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
- **Compressible `dP_dT` mutates the `AbstractState`.** Callers that re-invoke (e.g. parallel-branch iteration, GUI re-runs) must re-anchor with `AS.update(CP.PT_INPUTS, P_in, T_in)` before each call. The network solver does this automatically.
- **At least one P-spec node** is required to anchor any `Network.solve()`. Bare-float `Q_ext` defaults to **kg/s** (not whatever the GUI dropdown reads). Compressible solver also rejects actual-volumetric flow units and requires `T` on any node with positive `Q_ext`.
- **Phase envelope is expensive.** Build it once via `_build_phase_limits(AS)` and forward to every `compressible_flow.*` call that takes `T_cricondentherm` / `P_cricondenbar` / `T_critical` / `P_critical` kwargs. CoolProp's `build_phase_envelope` also corrupts the AS's solver state — do it on a temporary copy (`_build_phase_limits` already handles this).
- **`Network.save()` files (`*.hydnet.json`)** carry an optional `gui_extras` block for canvas positions / original unit strings. The headless solver ignores it; the GUI relies on it. CSV-backed pipes store paths relative to the save file in `gui_extras` and absolute in the top-level component dict — either is acceptable on load.

## When adding components or solver features

- New component class → derive from the appropriate `Base_*`, add `to_dict`/`from_dict` round-tripping, extend `_reversed_component()` if the component is asymmetric under flow reversal.
- New residual term in the network solver → keep it normalized to O(1) (`P_init`, `mdot_ref`, `mdot_ref * 1e6` are the existing scales), and add a self-test to the `__main__` block of the relevant network module.
- Don't introduce a pytest dependency just to wrap existing self-tests — the project deliberately runs validations as plain functions.
