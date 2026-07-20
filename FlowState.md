# `FlowState` — quick reference

`FlowState` is the calling unit of the compressible layer. Every public
function in [compressible_flow.py](compressible_flow.py) and every component
`dP_dT(fs)` method takes one. This doc covers what it is, how to construct
one, and how to walk a network edge or a single segment with it.

For the design rationale, see the FlowState section in [README.md](README.md)
and the bug-fix note under "AbstractState does not carry velocity" in
[improvements.md](improvements.md).

---

## What a `FlowState` carries

```python
class FlowState:
    AS                  # CoolProp AbstractState, at STATIC (P, T)
    mdot                # mass flow rate [kg/s], magnitude (>= 0)
    A                   # local flow area [m^2]
    z                   # local elevation [m]
    T_cricondentherm    # cached phase-envelope limits, used by
    P_cricondenbar      # _safe_flowstate_update_PT so functions can apply
    T_critical          # a supercritical phase hint without callers
    P_critical          # forwarding the four kwargs themselves
```

Derived properties (always recomputed from `AS`, so they cannot desync
after an in-place mutation):

| property | value |
| --- | --- |
| `fs.P` | `AS.p()` — static pressure [Pa] |
| `fs.T` | `AS.T()` — static temperature [K] |
| `fs.rho` | `AS.rhomass()` |
| `fs.v` | `mdot / (rho * A)` |
| `fs.Ma` | `v / speed_sound` |
| `fs.h_stagnation` | `h_static + 0.5 * v**2` |
| `fs.h_total_with_g` | `h_stagnation + g * z` |
| `fs.s_static` | `AS.smass()` (static == stagnation for entropy) |

**Convention: `fs.AS` is at the *static* state, never stagnation.** Code
that reads `AS.hmass()` and labels it `h0` (stagnation) is a bug; use
`fs.h_stagnation` instead. The cross-cutting refactor that introduced
`FlowState` was driven by exactly this confusion inside
`choked_mass_flux`.

---

## Building one

```python
import CoolProp.CoolProp as CP
import composition
from compressible_flow import FlowState

AS = composition.define_composition(y_Methane=0.95, y_Ethane=0.05, eos="HEOS")
AS.update(CP.PT_INPUTS, P_in_Pa, T_in_K)             # caller anchors AS at static inlet

fs = FlowState(
    AS,
    mdot=2.0,                                         # kg/s, magnitude
    A=first_component.inlet_area_si,                  # m^2
    z=inlet_node.elevation_m,                         # m
)
```

A few details to know:

- **Phase envelope is built automatically** on the first FlowState you
  construct from a given `AS`. Subsequent FlowStates over the same solve
  should forward the cached limits so the (expensive for HEOS) envelope
  build isn't repeated:
  ```python
  T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)        # do this once
  fs = FlowState(AS, mdot, A, z,
                 T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                 T_critical=T_c, P_critical=P_c)
  ```
- **`mdot` is positive.** Reverse flow is handled by `_reversed_component`
  (see below), not by sign.
- **`A` seed.** When walking a chain, set `fs.A = first_component.inlet_area_si`
  so the first component's internal `_area_match` is a no-op. Every Base
  class exposes `inlet_area_si` / `outlet_area_si`.
- **`mdot` from a pint Quantity.** Use `_resolve_mdot(flow_rate_qty, AS)` to
  convert (accepts mass, molar, std-volume, and — for non-network callers —
  actual-volumetric units).

---

## The mutation contract

Every physics function and every component `dP_dT(fs)` mutates `fs` in
place:

| field | mutated by |
| --- | --- |
| `fs.AS` | every function — advances to the outlet **static** state |
| `fs.A` | `compressible_changing_area_K`, `Contraction_Expansion.dP_dT`, and `Line_Segment.dP_dT` (across internal profile area transitions) |
| `fs.z` | `compressible_pipe_segment` (advances by `dz` on each successful slice); `Line_Segment.dP_dT` (cumulatively across slices) |
| `fs.mdot` | never (mass flow is conserved within a single component) |
| cached envelope limits | never |

The cached envelope limits are read by `_safe_flowstate_update_PT(fs, P, T)`,
which every function uses for in-place AS updates that need the
supercritical phase hint.

---

## Walking a network edge — the canonical pattern

This is exactly what `Compressible_Network.walk_edge` does. Reuse the
shape verbatim wherever the GUI needs to walk a chain (Save Results,
"Plot profile…", per-block readouts, etc.):

```python
from compressible_flow import FlowState, _build_phase_limits, _safe_update_PT
from network import _reversed_component

# 1) Pick a flow direction and (possibly reversed) component list.
if mdot_e >= 0.0:
    comps   = edge.components
    inlet_node, outlet_node = edge.from_node, edge.to_node
else:
    comps   = [_reversed_component(c) for c in reversed(edge.components)]
    inlet_node, outlet_node = edge.to_node, edge.from_node

# 2) Anchor AS at the flow-inlet node's converged (P, T).
T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)        # cache up front
_safe_update_PT(AS, P_inlet, T_inlet, T_cric, P_bar, T_c, P_c)

# 3) Build ONE FlowState for the whole chain.
fs = FlowState(
    AS, mdot=abs(mdot_e),
    A=comps[0].inlet_area_si,
    z=inlet_node.elevation_m,
    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
    T_critical=T_c, P_critical=P_c,
)

# 4) Walk every component. Mixed-diameter chains "just work" -- each
#    dP_dT calls _area_match(fs, self.inlet_area_si) at entry, which
#    runs compressible_changing_area_K(K=0) if there's a discontinuity.
per_component_outlets_PT = []
for c in comps:
    c.dP_dT(fs)
    per_component_outlets_PT.append((fs.P, fs.T))

# At this point: fs.AS is at the edge outlet, fs.A == comps[-1].outlet_area_si,
# fs.z has been advanced through any line segments along the way.
```

### Reverse flow

`_reversed_component` returns a shallow-copied **shadow** of the original
that has its geometry flipped (profile reversed, `Di_US`/`Di_DS`
swapped). Its `inlet_area_si` / `outlet_area_si` then reflect the
swapped sides automatically — no extra logic needed at the FlowState
layer. **Always feed reversed shadows through the same `dP_dT(fs)`
interface; never write `sign(mdot) * forward_dp(|mdot|)` shortcuts.**

### Sealed check valves

Check valves seal perfectly: `Compressible_Network.walk_edge()` detects
a reverse trial flow (`mdot < 0`) on any CV-carrying edge **before
walking any component** and returns the inlet-node state as a
pass-through, flagging the edge `sealed` so the residual pins its mdot
to exactly zero — see [network.md](network.md) § "Reverse-flow
handling".  `CheckValve.dP_dT(fs)` itself models forward passage only.
GUI code that walks a chain by hand must apply the same rule: never
walk components under reverse conditions through a check valve — the
flow there is zero and the downstream side simply sits at its own
node's state.

---

## Walking a single segment for a profile plot

Same pattern, just shorter. This is the shape the GUI's per-block
"Plot profile…" button uses:

```python
AS.update(CP.PT_INPUTS, P_inlet, T_inlet)
fs = FlowState(AS, mdot=abs(mdot_e), A=seg.inlet_area_si, z=0.0)
profile_points = seg.dP_dT(fs, isothermal=False)
# profile_points: list of (distance_m, P_Pa, T_K, v_ms)
```

`Line_Segment.dP_dT` returns the per-profile-point tuples directly;
`Bend`, `Valve`, `CheckValve`, `Contraction_Expansion` return `None`
and just mutate `fs`.

---

## Per-component inlet / outlet readout

For a multi-component chain you typically want the `(P, T)` at the
boundary between each pair of adjacent components, not just the chain
endpoints.

**Cheap route (no extra walking).** When the chain was walked by the
network solver, `result["component_outlet_PT"][edge_name]` is already a
list of `(P_Pa, T_K)` tuples — one per component, indexed by the
*original* `edge.components` position regardless of which way the
converged flow actually ran. The boundary upstream of component `pos`
is the chain inlet (when `pos == 0`) or `component_outlet_PT[pos - 1]`
(when `pos > 0`); the downstream boundary is
`component_outlet_PT[pos]`. The compressible GUI canvas already uses
this; the inline node tracks its position via
`NetworkScreen._inline_chain_pos`.

**Live route (walking by hand).** Use the canonical pattern above and
append `(fs.P, fs.T)` after each `c.dP_dT(fs)`. The list you get is in
flow-direction order; if you want it back in the original
`edge.components` order under reverse flow, reverse it
(`list(reversed(...))`) — same trick `compressible_network.py` uses
when returning `component_outlet_PT`.

---

## Helpers worth knowing

- `_resolve_mdot(flow_rate_qty, AS)` — pint Quantity → kg/s float. Reads
  `AS.molar_mass()` for molar / standard-volume inputs and
  `AS.rhomass()` for actual-volumetric inputs (which the *network*
  solver rejects but the point-to-point path still accepts).
- `_build_phase_limits(AS)` — `(T_cricondentherm, P_cricondenbar, T_critical, P_critical)`.
  Builds on a temporary AS so the live one isn't corrupted. Cache the
  result; pass it into every `FlowState` you construct during the same
  solve.
- `_safe_flowstate_update_PT(fs, P, T)` — `AS.update(PT_INPUTS, P, T)`
  with the supercritical phase hint pulled off `fs`. Use this instead of
  `AS.update(...)` directly anywhere you need to re-anchor `fs.AS`
  mid-walk (`fs` itself isn't otherwise touched).
- `_area_match(fs, A_target)` — runs `compressible_changing_area_K(K=0)`
  iff `fs.A` differs from `A_target` beyond a 1e-6 fractional tolerance.
  Component `dP_dT` methods already call this at entry; you'll rarely
  need it from GUI code.

---

## Common pitfalls

- **Forgetting to re-anchor `fs.AS` between runs.** `dP_dT` mutates AS
  in place to the outlet state. If you walk the same chain twice (e.g.
  the GUI's "Re-run" button), call `AS.update(PT_INPUTS, P_in, T_in)`
  (or `_safe_flowstate_update_PT(fs_prev, P_in, T_in)`) and construct a
  fresh `FlowState` before the second walk.
- **Seeding `fs.A` from the wrong thing.** Use
  `comps[0].inlet_area_si`, not the node's elevation or the pipe ID.
  If you seed from a wrong area, the first component's `_area_match`
  will silently insert an isentropic area change that wasn't physically
  there.
- **Treating `fs.AS.hmass()` as stagnation enthalpy.** Use
  `fs.h_stagnation`. The old `choked_mass_flux` bug is the canonical
  example.
- **Forwarding the four phase-envelope kwargs by hand.** Cache them
  on the FlowState at construction; nothing downstream needs them as
  explicit kwargs anymore.
- **Walking a check-valve chain under reverse conditions.** A check
  valve seals perfectly: there is no reverse walk.  If the pressure
  drive across a CV-carrying chain is reverse, the flow is exactly zero
  — report the upstream blocks at the inlet state and the CV/downstream
  blocks at the downstream node's state, matching what
  `Compressible_Network` does for sealed edges.
