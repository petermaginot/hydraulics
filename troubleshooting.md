# Troubleshooting hydraulics network runs

A symptom-first reference for chasing down odd results.  Each section
starts with **what you can see in the saved files** and works back to the
likely cause.  When in doubt, re-load the saved network headless and
poke at it — that's almost always faster than guessing.

For solver internals and known limitations, see [network.md](network.md);
for the GUI surface, see [GUI.md](GUI.md).  This file deliberately
duplicates very little of either.

---

## Where to look first

When handed a `.hydnet.json` + a `<run>/` results directory, read them in
this order:

1. **`<run>/summary.json`** — the converged flag and the per-node /
   per-edge headline numbers.  Almost every diagnosis starts here.
2. **`<run>/<edge>__<pos>__<comp>.csv`** — per-pipe profile.  Open the
   first/last few rows to sanity-check monotonic P drop in flow direction.
3. **`<network>.hydnet.json`** — topology + boundary specs.  Check this
   when summary.json says something physically odd ("how is the flow going
   that way?"): the answer is usually in the BCs.

A successful run looks like:

- `summary.json` has `"converged": true`.
- Every junction satisfies `sum(signed edge mdots into node) + Q_ext_kgs ≈ 0`.
- Each pipe profile CSV shows P decreasing monotonically along the flow
  direction (modulo elevation gain on uphill segments).
- Compressible: T moves smoothly along each pipe; the sign of the change
  matches expansion (cooling) or compression.

---

## Symptom → likely cause

### `summary.json` says `"converged": false`

The solver gave up before the residual fell below `1e-4` (normalized).
Most common causes, in order:

- **Sealed-check-valve over-determination.**  A check valve is being
  forced to seal but the downstream pressure can't be satisfied at
  `mdot = 0`.  The sealed-edge residual swap drives `mdot → 0` correctly
  but leaves a non-zero residual elsewhere.  Cross-check: the offending
  edge has `mdot_kgs ≈ 0` and contains a `CheckValve` component.  The
  reported flows are still physically correct — the warning just reflects
  that no exact equilibrium exists.  See `network.md` (the "sealed-edge
  residual swap" subsection).
- **Over- or under-specified boundary conditions.**  At least one node
  must have `P` spec'd to anchor pressures.  Conversely, an all-Q-spec
  network has no anchor and the solver has nothing to converge to.
- **Compressible junction-network initial guess.**  A uniform scalar
  `mdot_init_kgs` is mass-imbalanced at any junction with unequal in/out
  edge counts; the solver can stall at the trivial all-zero-flow
  attractor.  Re-solve with a per-edge `dict {edge_name: kg/s}` guess
  that satisfies mass balance at every junction.
- **Genuine physical infeasibility.**  Trying to drive 1000 BBL/D
  through a fully-closed valve, or pumping uphill with insufficient
  inlet head.  If the residual is large *and* `mdot` values look
  pinned-to-zero everywhere, suspect this.
- **Compressible: phase envelope crossed.**  CoolProp can't do a PT
  update inside the dome; the slice splitter exhausts its budget and the
  walk returns the penalty `P = 1 Pa` sentinel.  Check the inlet (P, T):
  near the saturation line, the solver is fragile.

### A junction's mass balance doesn't close

For a junction `J` with no `Q_ext` spec, summary.json should give
`sum(mdot_kgs of incoming edges) - sum(mdot_kgs of outgoing edges) ≈ 0`
(within `1e-6 kg/s` for well-converged runs).  Larger imbalance points to:

- **A missing connection** in the canvas — an inline node never got
  wired to the next thing downstream and the chain walker dead-ended
  silently (rare; usually raises at solve time).
- **A duplicate node name** that got resolved by NodeGraphQt's
  auto-rename without the user noticing.  Open `.hydnet.json` and check
  for two nodes whose name field doesn't match what they should.
- **A wrong-sign `Q_ext`** at a boundary node.  Recall: `Q_ext > 0` is
  supply into the network, `Q_ext < 0` is withdrawal.

### An edge has unexpected reverse flow

`summary.json` shows `mdot_kgs` negative on an edge whose nominal
direction the user thought would be forward.  Almost always a real
result: the BCs make backflow physical (e.g. the downstream P is higher
than upstream).  If that's intentional, nothing to fix.  If not:

- Check inlet/outlet P specs in `.hydnet.json` — is one wrong by an
  order of magnitude (psi/bar confusion)?
- For check-valved edges, a backflow attempt should leave `mdot ≈ 0`
  (sealed).  If it doesn't, the CV component is mis-built — its `K`
  value or `check_valve = True` marker may be missing.

### A pipe profile CSV shows P *rising* in flow direction

Open the CSV: column `distance_m` runs 0 → length from the **flow
inlet**, not the nominal `from_node` end.  If `P_Pa` rises with
distance:

- **Flow is running in reverse** on this edge.  Check the corresponding
  `mdot_kgs` sign in summary.json; the profile is walked on the
  reversed shadow, so its inlet is the nominal `to_node`.
- **Uphill elevation gain** outpaces friction loss — physically valid;
  check the `elevation_m` column.
- **A pump / negative dP component** is on the edge — not supported by
  this solver yet, so this should not happen.  If it does, something
  is wrong with a custom component subclass.

### Compressible: outlet T = inlet T despite a long pipe

- **Isothermal mode is on** somewhere it shouldn't be.  The point-to-
  point screens have an Adiabatic / Isothermal radio; the network
  screens default to adiabatic.  Check `AppState.isothermal`.
- **Mass flow is near zero** — JT cooling and friction heating both go
  to zero as flow goes to zero.  If `mdot_kgs` is tiny, T is allowed to
  be flat.

### Compressible: `T_K` missing for some Source/Sink nodes

A cross-regime load (incompressible → compressible) merges in blank
`T_str` defaults but does not fill them.  The load dialog should have
flagged this; if it was dismissed, every Source/Sink needs a `T` set
in the editor before the next solve.  The `_extra_kwargs_for_boundary`
hook raises a clean error at solve time naming the offending node.

### Compressible: `RuntimeError: ... slice failed to converge after N recursive splits`

Raised by `compressible_pipe_segment` when a slice can't meet the split
tolerances within `max_split_depth` (default 8).  The message now reports
`corrected_energy_error` (the energy balance of the Heun+Newton **corrected**
outlet — what the function returns — not the Euler predictor), alongside
`dPdL_relchg` and `Ma_change`.  Read which metric is over tolerance:

- **`Ma_change` large and/or the choke hint fires** ("likely choking
  (Fanno)…"): the flow is genuinely at/near the choke inside that slice.
  Reduce `mdot`, or accept that this is the segment's Fanno limit.
- **`corrected_energy_error` and `dPdL_relchg` both just over tolerance, no
  choke hint**: the slice is *under-resolved*, not choked — typically a coarse
  single-segment profile (e.g. a 10 ft pipe entered as one ~3 m profile point
  pair) driven to a strongly-accelerating subsonic regime (Ma ≳ 0.6).  The
  near-exit slices want more than 8 splits.  Remedy: give the pipe a finer
  initial `profile` (more points), or raise `max_split_depth`.
- This is *not* the old spurious failure where the Euler predictor energy error
  tripped the gate while the corrected state was fine — that false split was
  fixed (improvements.md R12).  If you see a convergence failure now, treat it
  as real per the above.

### Compressible: `RuntimeError: ... could not bracket a subsonic root for A/A*=...`

Raised by `compressible_changing_area`.  For a **large expansion** (or an area
change probed at a tiny trial `mdot` by an inverse / mdot-solve, which makes
`A/A*` enormous) this is fixed — the outlet is treated as stagnation via a
deep-subsonic short-circuit (improvements.md R13), so a pipe→huge-expansion
discharge chain solves headlessly just like the GUI network walk.  If you still
see this, the `A/A*` in the message is **small (< 1)**, which means a
**contraction** whose throat would have to go supersonic — i.e. a real choke at
that area change.  Reduce `mdot` or open up the downstream area.

### Mass flow is orders of magnitude off expectation

Almost always a unit error:

- `Q_ext` entered in `gal/min` when the BC dropdown was on `BBL/D`.
- `m^3/s` confused with `m^3/h`.
- Density entered in `kg/m^3` while the unit dropdown said `lb/ft^3`
  (or vice versa).  Check the `gui_extras.canvas_nodes[*].spec` block
  in `.hydnet.json` for the original `_str` / `_unit` pair.
- Compressible: confusion between `mmscf/day` (standard volumetric)
  and an actual-volumetric flow attempted on the network solver
  (which rejects actual volumetric — that error fires at solve time).

### A pipe's CSV profile didn't load

Symptoms: load-time warning "could not load CSV ...", or after a load
the pipe's editor shows "manual mode" when the user expected CSV.

- **Relative path broke.**  The CSV was moved without the save file,
  or the save file was moved without the CSV.  Save files store
  `gui_extras.canvas_nodes[*].spec.csv_path` as a relative path; the
  top-level `edges[*].components[*].csv_path` stores the absolute path
  the file was loaded from.  Either is sufficient — check both before
  giving up.
- **CSV headers changed.**  `Line_Segment.from_csv()` accepts
  `ID`/`OD`/`WT` *or* `D_h`/`flow_area`.  Mixing the two, or renaming a
  column, makes the loader unable to determine geometry mode.
- **Zero diameters** in the CSV.  The loader treats `0` as "absent" so
  it can fall back to OD-2·WT, but if all of ID/OD/WT are zero on a
  row, the row fails.

---

## Diagnostic snippets

Copy-paste these into a scratch script next to the save file:

### 1. Re-solve verbose to see solver telemetry

```python
from network import Network
from incompressible import Incompressible_Fluid
from component_classes import ureg

net   = Network.load("scenario.hydnet.json")
fluid = Incompressible_Fluid(
    density=ureg.Quantity(50.0, "lb/ft^3"),
    viscosity=ureg.Quantity(1.0, "cP"),
)
result = net.solve(fluid, verbose=True)
print(f"converged: {result.converged}")
```

The verbose output prints status, `nfev`, and the final residual norm.
For compressible: replace with `Compressible_Network.load(...)` and pass
the AbstractState instead of a fluid.

### 2. Mass-balance closure at every node

```python
balance = {n: 0.0 for n in net._node_order}
for e in net._edges:
    m = result["mdot_kgs"][e.name]
    balance[e.from_node] -= m
    balance[e.to_node]   += m
for n in net._node_order:
    balance[n] += result["mdot_ext_kgs"].get(n, 0.0)
for n, b in balance.items():
    flag = "  OK" if abs(b) < 1e-6 else "  *** IMBALANCED ***"
    print(f"{n:<16} {b:+.3e} kg/s{flag}")
```

A node showing several kg/s of imbalance is almost always the
problem's epicentre.

### 3. List every reversed-flow edge

```python
for e in net._edges:
    m = result["mdot_kgs"][e.name]
    if m < 0.0:
        print(f"{e.name}: {m:+.3f} kg/s (nominal {e.from_node} -> {e.to_node})")
```

### 4. Walk a single edge component-by-component

```python
edge_name = "Source A->outlet"
for e in net._edges:
    if e.name == edge_name:
        for c in e.components:
            cr = result.component(c)
            print(f"  {type(c).__name__:<20} {getattr(c, 'name', ''):<10} "
                  f"P_in={cr.P_in_Pa:.0f} -> P_out={cr.P_out_Pa:.0f}  "
                  f"dP={cr.dP_Pa:+.0f} Pa")
        break
```

This breaks an edge's total dP into per-component contributions — the
single most useful diagnostic for "why is the pressure drop so high?".

### 5. Profile sanity check from the saved CSV

```python
import csv
with open("results/Source A->outlet__0__main_line.csv") as f:
    rows = list(csv.DictReader(f))
P = [float(r["P_Pa"]) for r in rows]
# Flag any internal point where P rose above its predecessor by > 100 Pa
# (small rises from elevation drop are OK -- adjust threshold per case).
for i in range(1, len(P)):
    if P[i] - P[i-1] > 100.0:
        print(f"row {i}: P rose by {P[i] - P[i-1]:.1f} Pa "
              f"(d={rows[i]['distance_m']}, elev={rows[i]['elevation_m']})")
```

---

## Known gotchas (cross-references)

These already live in [network.md](network.md) and aren't duplicated here:

- Reverse-flow handling reverses the **geometry**, not the friction sign
  — relevant if you've added a custom component subclass.
- Check valves use a tanh-blended K in the residual but the discrete
  `_SEALING_K` substitution in post-solve walks.
- `Q_ext` accepts pint Quantities with mass or volumetric dims; bare
  floats default to **kg/s** (not the units shown on the GUI dropdown).
- Compressible solver: an inlet with positive `Q_ext` *must* spec `T`,
  even if it's currently a withdrawal (a future solve might reverse
  it).  Enforced at solve time with a named-node error.
- Compressible: actual-volumetric flow units (`m^3/s`, `gal/min`) are
  rejected — density varies along the network.

---

## When the file format is the suspect

If `Network.load()` raises with a confusing error, the file format
spec lives in `network.py` (`Network.to_dict` docstring, around the
`SAVE_FORMAT_VERSION` constant) and is summarised in
[network.md](network.md) under "Persistence".  The top-level shape is:

```
{ "format_version": 1,
  "regime": "incompressible" | "compressible",
  "nodes":       [ {name, elevation_m, P_Pa, Q_ext, T_K?}, ... ],
  "edges":       [ {name, from_node, to_node, components: [...]}, ... ],
  "gui_extras":  { canvas_nodes, canvas_connections }   # optional
}
```

Hand-edits are supported — the format is plain JSON.  After hand-edits,
re-load via `Network.load()` to verify the file still parses cleanly
before opening it in the GUI.
