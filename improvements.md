# Improvements

A consolidated punch list from a review of the
compressible and incompressible hydraulics core. Organized by component, with
each item flagged as a **bug** (wrong answer in some regime), **gap** (missing
feature with real engineering consequence), or **polish** (cost / clarity / UX).
Priorities reflect engineering risk for realistic use cases (high-dP gas
service, cavitating liquid service, near-choke pipe flow), not implementation
effort.

---

## 1. Compressible valves: `compressible_K` and `Valve.dP_dT`

Files: [compressible_flow.py:1135-1242](compressible_flow.py#L1135-L1242),
[compressible_flow.py:378-436](compressible_flow.py#L378-L436),
[compressible_flow.py:474-541](compressible_flow.py#L474-L541).

### Core problem

`compressible_K` is a first-order Taylor expansion: every coefficient
(rho, v, Cp, partial derivatives) is evaluated at the **inlet** and the
resulting linear dP/drho relationship is applied over a *finite* step. The
docstring admits this ("fluid properties are roughly constant across the
fitting"). For a 50% pressure drop, none of those properties are roughly
constant — density nearly halves, velocity nearly doubles, the acceleration
feedback in the denominator strengthens nonlinearly as the gas expands.

### Specific issues

1. **[bug, high-dP]** The acceleration feedback term
   `1 - v^2*(drho/dP)_H / [1 - (v^2/rho)*(drho/dH)_P]` captures the runaway
   "expand -> speed up -> drop more P -> expand more" loop. Linearizing it at
   inlet conditions systematically *under-predicts* |dP| at high dP/P_1.

2. **[bug, choke detection]** The outlet Mach check ([compressible_flow.py:1238-1242](compressible_flow.py#L1238-L1242)) tests the body outlet, not the
   vena contracta inside the trim. Real valves choke at the vena contracta
   when the body outlet is still comfortably subsonic. ISA-75.01 / IEC-60534
   handles this with the `x_T` (terminal pressure-drop ratio) factor; we have
   no analogue. A control valve at 50% dP is almost certainly internally
   choked even when our Ma_out comes back ~0.4.

3. **[bug, dT accuracy]** The flow-work term uses `(1/rho_in)*dP` ([compressible_flow.py:1220](compressible_flow.py#L1220)). The correct integral is
   integral of dP/rho along the process. For a 50% expansion, mean 1/rho is
   ~1.5x inlet, so the JT cooling contribution is under-estimated by 30-50%.
   The one-step Newton correction at lines 1227-1234 polishes T_out *at the
   wrong P_out* and cannot fix P_out itself.

4. **[bug, entropy linearization]** Entropy generation is linearized at T_in.
   Compare to `compressible_changing_area_K` ([compressible_flow.py:1108](compressible_flow.py#L1108)) which uses T_avg. The author already knew this
   mattered.

5. **[bug, denominator singularity]** No guard on the `inner` term or the outer
   `1 - v^2*(drho/dP)_H/inner` going through zero. Both reflect the
   area-Mach singularity algebraically. At high Mach the function can return
   arbitrarily large |dP| from a near-zero denominator with no warning.

6. **[gap, K-factor validity]** K is referenced to inlet velocity head.
   Standard valve K correlations (Crane, Hooper, Idelchik) are calibrated
   against *incompressible* test data assuming rho*v^2 roughly constant
   across the fitting. Re-using the same K at high-dP gas service without an
   expansion-factor Y correction is precisely what the ISA / IEC valve-sizing
   standards exist to fix. Even with perfect numerics, the input model is
   wrong for the regime.

7. **[bug, propagation]** `CheckValve.dP_dT` ([compressible_flow.py:535-541](compressible_flow.py#L535-L541)) inherits all of the above for forward flow.

### Recommendations (ordered)

- **R1 [cheap fix] ✅ DONE.** In `compressible_K`, after computing the linearized dP,
  test |dP|/P_in > ~5-10% or denominator < ~0.5. If either, fall back to
  `compressible_changing_area_K` with A_in = A_out. The valve / check-valve
  `dP_dT` then becomes regime-aware automatically.

  *Implemented:* `compressible_K` now accepts a `dPmax` parameter (default 0.05).
  After computing the linearized `dP`, if `abs(dP)/P_in > dPmax` or
  `denominator < 0.5`, execution delegates to `compressible_changing_area_K`
  with `A_in = A_out = flow_area`. A sign bug in the original condition
  (`dP/P_in > dPmax` — always False for negative dP) was also fixed to
  `abs(dP)/P_in > dPmax`. Covered by `test_K()` in `examples.py`.

  *Subsequent fix (review of R1):* an initial refactor of the dP denominator
  was algebraically wrong — it dropped a term and put the Fanno singularity
  at Ma = 1/sqrt(gamma) (~0.85 for air) instead of Ma = 1. Restored the
  correct two-line form (`inner`/`denominator`). Items 1, 3, 4, 5, 7 below
  are now substantially mitigated by the R1 fallback: at high dP the
  rigorous `compressible_changing_area_K` is invoked, which uses T_avg in
  the entropy balance and a coupled (P, T) root-find that avoids the
  inlet-property linearization.

- **R1.5 [✅ DONE].** Real-gas choked-flow detection via `choked_mass_flux`
  isentropic march with mixture-safe (P,T)+entropy-root EOS
  updates). Both `compressible_K` and `compressible_changing_area_K` now
  carry a cheap ideal-gas `G_max` pre-screen; above the threshold the
  march runs, and a dedicated `ChokedFlowError(RuntimeError)` is raised when 
  `mdot > mdot_choked`. `_NoChokeBracketError` (sentinel) is
  raised when the grid scan finds no Mach=1 — caught and treated as "not
  choked" by the pre-screens; genuine CoolProp / numerical failures
  propagate. Substantially addresses item 5 (the denominator singularity
  is now caught before being evaluated). **Stage-1 caveat closed** by R2
  below: `Valve` / `CheckValve` now accept `minimum_diameter` and route
  it to `compressible_K(A_throat=...)`, which screens the trim throat
  with recovery to the body area.

- **R2 [✅ DONE for geometric throat; F_L recovery still open].** Added
  `minimum_diameter` (geometric throat) on `Base_Valve` / `Base_CheckValve`.
  When supplied, it flows through `Valve.dP_dT` / `CheckValve.dP_dT`
  into `compressible_K` as a new `A_throat=` kwarg.  The throat-area
  choke pre-screen is hoisted above the dP linearization so it fires
  regardless of whether the fast (linearized) or slow
  (`compressible_changing_area_K`) branch runs; `A_outlet` is set to
  the body area so post-throat recovery is modelled.  At construction,
  the implied `Cd_eff = (Di/D_min)^2 / sqrt(K)` is computed and a
  `UserWarning` fires if it falls outside [0.3, 1.05] (impossible
  throat at high end, F_L < 1 territory at low end).  Covered by
  `test_valve_minimum_diameter_choke` in
  [textbook_test_functions.py](textbook_test_functions.py): for a
  methane-rich mixture at 20 bar / 300 K with `D_pipe = 2"`,
  `D_min = 1"`, `K = 20`, `mdot = 4.5 kg/s`, the `minimum_diameter`-aware
  valve raises `ChokedFlowError` at the trim (~1.94 kg/s clamp) while
  the legacy pipe-area-only screen lets the flow pass silently.  **Open
  follow-on**: the geometric throat alone cannot model strong
  downstream pressure recovery (low `F_L`).  R3 (ISA-75.01 / IEC-60534
  with explicit `x_T` or `F_L`) remains the next refinement.

- **R3 [larger].** Switch the valve component to an ISA-75.01 / IEC-60534-2-1
  sizing model parameterized by Cv (or Kv) and x_T, with explicit choked-flow
  branch. Cv / Kv are already accepted per recent commits; the data is there.

- **R3.5 [follow-on to R1.5].** Have `compressible_network.walk_edge`
  catch `ChokedFlowError` specifically and consume `mdot_choked` from the
  exception to clamp the solver's `dmdot` step.  Today the exception is
  caught generically as a walk failure and the structured data on the
  exception is discarded, so the solver still recovers via repeated
  failed walks rather than a single targeted retreat.

---

## 2. Incompressible valves: cavitation handling

Files: [incompressible.py:412-486](incompressible.py#L412-L486),
[incompressible.py:75-119](incompressible.py#L75-L119),
[network.py:1340-1363](network.py#L1340-L1363).

### Core problem

The current `Valve.dP()` is end-to-end lumped via K-factor on pipe velocity
head. It cannot see the vena contracta where actual cavitation occurs.
`Incompressible_Fluid` stores only density and viscosity — no vapor pressure.
So today there is neither a way to detect cavitation nor anything to compare
against if a check were added. The following reference will be useful: (https://www.osti.gov/biblio/10155405)

### Three regimes worth distinguishing

Lumping them produces warnings that either fire too often or miss flashing
entirely:

1. **Flashing across the valve** (P_out < P_v): downstream is two-phase, model
   invalid. Should be a hard error.

2. **Choked cavitating flow**: dP_actual >= F_L^2 * (P_1 - F_F * P_v) with
   F_F approx 0.96 - 0.28*sqrt(P_v / P_c). Vena contracta pressure has
   reached P_v; further dP does not increase mass flow. The Newton-iteration
   in the network solver is being lied to about dQ/dP. Warn loudly and
   probably refuse the working point.

3. **Incipient / damaging cavitation**: sigma = (P_1 - P_v) / |dP| below
   ~1/F_L^2 but not yet choked. Bubbles form and collapse downstream,
   eroding the trim. Hydraulics still valid; equipment being destroyed. Warn.

### Recommendations (staged)

- **R4 [tier 1, cheapest, no API change].** In the network walk, check after
  each component that P_out is above a configurable floor (default 0 Pa
  absolute). Warn or raise (configurable). Catches negative absolute
  pressures and "user gave invalid inputs" cases that currently slip through
  silently. Does *not* catch cavitation specifically but catches the bug
  class you are worried about.

- **R5 [tier 2, cavitation check].** Add optional constructor args:
  - `Incompressible_Fluid(..., vapor_pressure=None, critical_pressure=None)`
  - `Valve(..., F_L=None)` and same on `CheckValve`.

  When both `fluid.vapor_pressure` and `valve.F_L` are present, run the
  three-regime check above per valve during the network walk, and surface
  results on the network result object so the GUI can flag offending
  components. Defaults of `None` preserve current behavior.

- **R6 [tier 3, contractions].** `Contraction_Expansion` does *not* need F_L —
  the throat static pressure follows from Bernoulli, since for a sharp
  contraction the throat is the downstream pipe cross-section. Compute
  throat P internally and warn if it falls below P_v.

### What not to do

- **Do not** check only `P_out > 0`. Passes for the most damaging cases
  (modest dP at moderate line pressure with high-P_v liquid like propane near
  ambient).
- **Do not** auto-estimate F_L from a Crane K. No defensible correlation.
  Published F_L comes from valve manufacturer testing and varies wildly
  between trim styles. Better to require F_L explicitly and skip the check
  when absent.
- **Do not** change `Valve.dP()` itself to limit dP at the choke point inside
  the component method. Breaks the Newton-derivative assumption the network
  solver depends on. The check belongs at the network walk.

---

## 3. Compressible line segment: `compressible_pipe_segment` and `Line_Segment.dP_dT`

Files: [compressible_flow.py:1245-1731](compressible_flow.py#L1245-L1731),
[compressible_flow.py:127-300](compressible_flow.py#L127-L300).

### What is right (preserve)

- The ODE derivation in the comments ([compressible_flow.py:1459-1484](compressible_flow.py#L1459-L1484)) is correct: combining energy,
  continuity, entropy-via-thermo-identity, and rho(H,P) gives the
  Fanno-with-heat-and-gravity dP/dL. Sanity check: for ideal gas the
  denominator `v^2*A + v^2*B/rho - 1` reduces algebraically to `Ma^2 - 1`,
  the classic Fanno denominator.
- The dT formula correctly splits into JT cooling
  (`-(T/rho^2)*(drho/dT)_P*dP` = `-mu_JT*dP`), friction heating, and external
  heat. Textbook-correct and a real strength.
- The two-metric splitter (energy residual + relative dP/dL change) is well
  motivated and catches most failure modes.

### Central limitation

**Forward Euler with inlet-only properties.** Every coefficient
(rho, v, Cp, A, B, f, T) is evaluated at the inlet and held constant across
the slice ([compressible_flow.py:1490-1497](compressible_flow.py#L1490-L1497)). Base scheme is first-order with O(dL) global accuracy. The
splitter compensates, but it is compensating for a low-order integrator.
A second-order scheme (midpoint, RK2, implicit trapezoidal) would cost one
extra property evaluation per step and let slices grow ~5-10x for the same
error. Implicit trapezoidal additionally buys A-stability near choke, where
the ODE becomes stiff.

### Issues for large dP / near-choke

8. **[gap, diagnostic]** Choke detection is reactive, not predictive. Fanno
   `L_max(Ma_in)` is closed-form for ideal gas and tractable for real gas.
   At each slice we could detect "the choke length for this mdot and inlet
   state is shorter than the remaining slice" and emit an actionable
   diagnostic. Today the user sees "failed to converge after 8 splits" with
   no way to distinguish numerical drift from physical impossibility.

9. **[polish]** The 0.98 Mach gate ([compressible_flow.py:1727](compressible_flow.py#L1727)) is a discontinuity. Ma_out = 0.97 returns
   silently as if all is well, although the inlet-property linearization is
   already poor. Either lower the warn threshold (warn at 0.7, hard-fail at
   0.95) or add a Ma-change splitter metric.

10. **[gap, splitter blind spot]** The splitter checks dP/dL and energy but
    *not* change in Mach number. A slice taking Ma from 0.3 to 0.7 can pass
    both metrics while having order-1 relative property change. Add
    `|Ma_out_trial - Ma_in| > Ma_change_tol` (default ~0.1) as a third
    splitter metric.

11. **[bug, error reporting]** Near choke, trial-outlet dP/dL recomputation
    can have `denom_out` ([compressible_flow.py:1551](compressible_flow.py#L1551)) cross zero. needs_split fires (good) but the
    diagnostic does not mention the choke origin — user sees only "failed to
    converge."

12. **[polish, energy correction]** The single Newton step on T_out
    ([compressible_flow.py:1625](compressible_flow.py#L1625)) projects onto
    the energy-conservation manifold at the *Euler-computed* P_out. P_out
    itself is not corrected. A midpoint / trapezoidal scheme would conserve
    energy to integrator order and this fixup would not be needed.

### Issues with the splitter machinery

13. **[polish, false splits]** `dPdL_avg = max(|dPdL_in|, |dPdL_out|, 1.0)`
    floors at 1 Pa/m. For low-velocity near-horizontal slices with true
    dP/dL ~ 0.1 Pa/m, the 5% relative tol becomes 0.05 Pa/m absolute -
    very tight, forces unnecessary splitting. Scale the floor to something
    like a fraction of `P_in / total_length` instead.

14. **[polish, tolerance calibration]** `energy_tol = 10 J/kg` is a fixed
    absolute value. Conservative for high-Cp fluids (water vapor), loose
    for low-Cp dense gases. Express as a fraction of v^2/2 or as DT*Cp_in.

15. **[polish, perf]** Each split costs one extra CoolProp update to restore
    AS to inlet ([compressible_flow.py:1594](compressible_flow.py#L1594)).
    Probably unavoidable but compounds: ~3 updates per slice baseline + 1
    per split level.

### `Line_Segment.dP_dT` issues

16. **[bug, tapered pipes]** Staircase representation of varying diameter
    ([compressible_flow.py:268](compressible_flow.py#L268)): each slice uses
    `flow_area=area_in` for the whole length, then a step-change at the
    boundary via `compressible_changing_area_K`
    ([compressible_flow.py:284](compressible_flow.py#L284)). For a
    profile representing a real tapered reducer this misrepresents the
    friction integral. Fine for typical pipelines (discrete diameter
    transitions), wrong for tapers. At minimum document this; ideally warn
    when a slice has both significant friction *and* significant area change.

17. **[polish, perf]** The boundary correction calls
    `compressible_changing_area_K` with K=0, which runs a root finder
    ([compressible_flow.py:1121](compressible_flow.py#L1121)). For genuinely
    isentropic boundaries `compressible_changing_area` (no K) gives the same
    answer without root-finding overhead. The unified call costs root-finder
    time on every slice boundary just past the 1e-6 area tolerance.

18. **[gap, ambient heat transfer]** Heat is distributed uniformly per unit
    length. Real environmental heat transfer is
    q = U * perim * (T_amb - T_fluid) * dL — proportional to local
    T_fluid - T_amb. A buried pipeline approaches soil temperature
    asymptotically, not linearly. Today the user must manually chunk and
    tune q per chunk. Feature gap for serious pipeline modeling.

19. **[polish, UX]** Mach check happens per slice but `Line_Segment.dP_dT`
    does not summarize max Ma along the segment. The user must compute v/a
    over the returned profile_points themselves. Return a `max_Ma_along_segment`
    or include local Ma in the profile tuples.

### Recommendations (ordered by value / effort)

- **R7 [✅ DONE].** Predictive ideal-gas choke diagnostic at
  `Line_Segment.dP_dT` entry. Two branches selected by the
  `isothermal` kwarg:

  - *Adiabatic:* closed-form Fanno `fLmax/D` ("Fluid Mechanics for
    Chemical Engineers, 2nd ed" §8.4.1 eq. 8.30, outlet `M=1`,
    Darcy convention):

        fLmax/D = (1 - M^2)/(k*M^2)
                + (k+1)/(2k) * ln[(k+1)*M^2 / (2 + (k-1)*M^2)]

    compared against the cumulative geometric integral
    `Σ f_i · dL_i / D_h_i` along the profile.

  - *Isothermal:* simplified long-pipeline form (§8.4.2 eqs. 8.31-8.33,
    kinetic-energy term dropped per the textbook's "long pipeline"
    assumption). Integrating `P·dP` at constant `T` with ideal-gas
    `ρ` and allowing for varying `D` and `A`:

        P1^2 - P2^2 = mdot^2 · (R_univ T / M_molar)
                      · Σ f_i · dL_i / (D_h_i · A_i^2)

    so a real `P2` requires `Σ f·dL/(D·A^2) < P1^2 · M / (mdot^2 RT)`.

  Both branches integrate along the profile, so stepped/tapered
  geometries are handled exactly under their respective ideal-gas
  assumptions. Inlet `ρ`/`μ` and constant `mdot` keep the per-slice
  Reynolds computation to a single `4·mdot/(π·D·μ)` evaluation; one
  `fluids.friction.friction_factor` call per profile slice — orders of
  magnitude lighter than a single `compressible_pipe_segment`
  invocation.

  Failure mode is `warnings.warn(UserWarning)`, never an exception:
  real-gas behavior near the critical point or under strong heat
  transfer can shift the actual choke point. Wrapped in a swallowing
  `try/except` so the diagnostic can't block real evaluation. Skip
  outside `0.01 < Ma_in < 0.98` (stagnant — no choke risk worth
  warning; or already in the reactive `compressible_pipe_segment`
  gate's territory). Helper lives at `_line_segment_choke_diagnostic`
  in [compressible_flow.py](compressible_flow.py), called once at the
  top of `Line_Segment.dP_dT` after `_area_match` and before the slice
  loop. Covered by `test_line_segment_choke_diagnostic` in
  [textbook_test_functions.py](textbook_test_functions.py):
  methane-rich mix at 5 bar / 300 K, 1" Sch 40, `Ma_in≈0.2` — the
  diagnostic fires on a 200 m segment (both adiabatic and isothermal)
  and stays silent on a 5 m segment.
- **R8 [cheap, high value].** Add Ma-change splitter metric (item 10). One
  extra divide per slice. Catches the dP/dL+energy blind spot.
- **R9 [medium, broad win].** Upgrade integrator to RK2 / midpoint. Doubles
  per-step cost, gains ~5-10x larger slices, removes need for the T-correction
  band-aid.
- **R10 [larger, conditional].** Implicit trapezoidal as an option for users
  routinely operating at Ma > 0.5. Bigger change; needs concrete demand.
- **R11 [cheap].** Document the staircase-diameter assumption in
  `Line_Segment.dP_dT` docstring; warn when slices combine friction and
  diameter change.

---

## Cross-cutting themes

- **Linearizations at inlet conditions are pervasive** (compressible_K,
  compressible_pipe_segment). Both are first-order in some quantity that can
  vary 50-100% across a single element. The remedies (higher-order
  integrator for pipes, fallback-to-rigorous-solver for valves) are
  well-scoped.

- **Choke / cavitation detection is reactive everywhere.** Compressible flow
  errors after the fact; incompressible flow does not detect cavitation at
  all. In both cases the cheap predictive check (closed-form Fanno choke
  length; sigma = (P_1 - P_v)/dP) is far more actionable than waiting for
  the numerics to fail.  *Status: real-gas choked-flow detection for
  fittings is implemented via `choked_mass_flux` (R1.5); internal-throat
  (vena contracta) choke check on valves is now wired through the new
  `minimum_diameter` parameter (R2); predictive ideal-gas
  Fanno / isothermal choke diagnostic for line segments now warns at
  `Line_Segment.dP_dT` entry (R7).*

- **The network walk is the natural enforcement point for both
  cavitation checks and the negative-pressure sanity floor.** Absolute
  pressures are known there. Cleanest place to surface warnings to the GUI.

- **Author already knew about most of the analytical limitations**
  (`compressible_K` docstring acknowledges the entropy / temperature
  approximation; `compressible_changing_area_K` exists as the rigorous
  alternative). The fixes are mostly about *invoking* the rigorous path when
  the linearized one is out of validity, not about new derivations.

- **[cross-cutting, architecture] AbstractState does not carry velocity. ✅ DONE.**
  Introduced [`FlowState`](compressible_flow.py) — a container bundling
  `(AS at static, mdot, A_local, z, cached phase-envelope limits)` — and
  refactored the compressible layer to take it as the calling unit
  throughout. `v`, `Ma`, stagnation enthalpy, and gravitational PE are
  derived properties on the container, so the static/stagnation
  distinction is now operative rather than implicit in docstrings.
  Bare functions (`compressible_K`, `compressible_changing_area_K`,
  `compressible_changing_area`, `compressible_pipe_segment`,
  `choked_mass_flux`) and component `dP_dT` methods now all take
  FlowState; the network walk in
  [compressible_network.py](compressible_network.py) constructs one
  FlowState at each edge inlet and passes it through the component
  chain.  Side benefits: area-change discontinuities between
  consecutive components are absorbed automatically (via `_area_match`
  → `compressible_changing_area_K(K=0)`), and the phase-envelope kwargs
  no longer thread through every call signature.

  *Bug fixed as a consequence:* `choked_mass_flux` now builds its
  stagnation reference from `fs.h_stagnation = h_static + v_in**2/2`
  instead of taking `AS.hmass()` literally — the old code claimed AS
  was at stagnation in the docstring while every caller passed static,
  silently undercounting `h0` by `v_in**2/2`.  Validation: at P=20 bar,
  T=300 K methane-rich mixture, the choked mass flow rises ~15% when
  v_in is raised from ~0 to ~a/2 — under the old code these would have
  returned the same value.  See
  `test_compressible_K_choke_roundtrip` in
  [textbook_test_functions.py](textbook_test_functions.py).
