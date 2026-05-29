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

- **R2.5 [partial — `compressible_dA` landed].** New
  [`compressible_dA(fs, A_throat, K, A2, P2)`](compressible_flow.py) — a
  single-entry-point constriction solver that splits the process into an
  isentropic acceleration from inlet to throat (`compressible_changing_area_K(K=0)`)
  followed by a K-dissipative recovery from throat to outlet area
  (`compressible_changing_area_K(e_loss=0.5·K·v_in²)`). The entropy balance on
  the recovery step uses the throat-to-outlet average temperature rather than
  the inlet-to-outlet average, which is more faithful to real geometry since
  most of the entropy generation lives in the post-vena-contracta turbulence.
  Two operating modes:

    - *Mode 1 — dictate `mdot`, solve for `P2`.* Mirrors the existing `dP_dT`
      calling convention.
    - *Mode 2 — dictate `P2`, solve for `mdot`.* Was previously a known gap
      (not yet on this list); now implemented. Runs `choked_mass_flux` to
      bound the answer, computes `P2_choked` via the same two-stage march,
      then either clamps to `mdot_choked` with adiabatic expansion to `P2`
      (`adiabatic_expansion_solver` helper) when `P2 < P2_choked`, or runs a
      1-D `brentq` on `mdot` for the subsonic case. Supersonic outlets on the
      choked branch raise `RuntimeError` (the same supersonic guard already
      present in `compressible_changing_area_K`).

  Substantially mitigates items 1, 3, 4 from §1 for any caller that switches
  from `compressible_K` to `compressible_dA` — properties are taken at the
  *throat*, not the inlet, for the dissipative leg, and the recovery step
  uses the rigorous coupled (P, T) solve. Covered by `test_compressible_dA`
  in [examples.py](examples.py).

  **Open follow-ons:**

    - **R2.6 [✅ DONE].** Switched `Valve.dP_dT` and `Orifice.dP_dT` off
      `compressible_K` / `compressible_orifice` so the production paths
      get the throat-then-recovery split for free.
      **Orifice**: `Orifice.dP_dT` resolves Cd (RHG or `Cd_override`),
      converts to K via `fluids.flow_meter.discharge_coefficient_to_K`, and
      calls `compressible_dA(fs, Cd*A_bore, K=K, A2=A_pipe)`. `compressible_orifice`
      is kept as a legacy function with a note. The K is flow-dependent (RHG
      depends on Re), so it is computed at call time rather than as a static
      property.
      **Valve / CheckValve**: both `Valve.dP_dT` and the forward-flow
      branch of `CheckValve.dP_dT` now dispatch on whether the trim
      carries a geometric constriction.  When `minimum_diameter` is
      `None` or equal to `Di` (no constriction), they delegate to
      `compressible_changing_area_K(fs, A_pipe, K=self.K)` — the
      rigorous coupled (P, T) solve with T_avg entropy balance is
      always used, replacing the inlet-property linearization in
      `compressible_K`.  When `minimum_diameter < Di`, they delegate to
      `compressible_dA(fs, A_throat, K=self.K, A2=A_pipe)` so the
      isentropic acceleration to the throat is separated from the
      K-dissipative recovery to the pipe.  Items 1, 3, 4 from §1 are
      now silently fixed on every valve call rather than only when the
      `dPmax`/denominator gate trips inside `compressible_K`.  The
      `CheckValve` reverse-flow (sealing K) short-circuit is unchanged.
      Covered by `test_valve_minimum_diameter_choke` in
      [textbook_test_functions.py](textbook_test_functions.py) and
      `test_K` in [examples.py](examples.py).
    - **R2.7 [✅ DONE — component-level `dmdot_dT` and network wiring
      both landed].** Every compressible component class now carries a
      `dmdot_dT(fs, P2)` inverse-solve method that mirrors the contract
      of `dP_dT` but takes the downstream pressure as input and mutates
      `fs` to the converged outlet state (with `fs.mdot` holding the
      solved value).  `Compressible_Network.solve` auto-detects edges
      whose downstream node is P-spec and dispatches them through an
      inverse walker (`walk_edge_inverse`) that swaps the standard
      `(walked_P - P_outlet)/P_ref` pipe residual for the local
      `(mdot - mdot_solved)/mdot_ref` form.  Implementation summary:

      The brentq-with-retreat machinery that originally lived inside
      `compressible_dA`'s Mode 2 non-choked branch was extracted into a
      shared private helper, `_solve_mdot_for_outlet_P(fs, P2,
      forward_at_mdot, mdot_choked, ...)`, in
      [compressible_flow.py](compressible_flow.py).  Each component's
      `dmdot_dT` builds an appropriate forward closure and delegates;
      `compressible_dA`'s Mode 2 was refactored to call the same helper
      (no behavior change — `test_compressible_dA` passes byte-identical).

      Per-class dispatch:

        * `Orifice.dmdot_dT`: Cd<->mdot fixed point (Cd from RHG depends
          on Re depends on mdot) wrapping `compressible_dA(..., P2=P2)`.
          Typically converges in 2-3 outer iterations; warns on
          non-convergence at iteration cap.  With `Cd_override` set,
          collapses to a single direct call.
        * `Valve.dmdot_dT` and `CheckValve.dmdot_dT` (forward flow):
          mirror the forward dispatch on `minimum_diameter`.  With a
          throat, forwards directly to `compressible_dA(..., P2=P2)`.
          Without a throat, drives `compressible_changing_area_K(fs,
          A_pipe, K)` via the shared helper.  `CheckValve` on the
          sealing-K shadow raises `ValueError` (the inverse is
          undefined: a sealed valve passes no flow at any P2).
        * `Bend.dmdot_dT`: K(Re)<->mdot fixed point wrapping
          `compressible_K(fs, K)` (Re depends only logarithmically on
          mdot via the friction-factor correlation, so 2-3 outer
          iterations suffice).
        * `Contraction_Expansion.dmdot_dT`: contraction direction uses a
          single helper drive of `compressible_changing_area_K(fs, A_DS,
          K)` (K is geometry-only).  Expansion direction raises
          `NotImplementedError` — for a sharp diffuser the kinetic-energy
          recovery typically dominates K-loss so P_out > P_in,
          reversing the residual sign convention the shared helper
          assumes; the use case (relief discharge to fixed downstream
          pressure) is invariably a constriction, so this was deferred.
        * `Line_Segment.dmdot_dT`: forward closure restores fs.AS, fs.A,
          and fs.z to the inlet snapshot on each call, then re-runs the
          full `dP_dT` slice loop.  Choke bound uses the ideal-gas G_max
          times the minimum profile area; the helper's retreat loop
          discovers the real-gas Fanno choke that friction induces well
          below the isentropic bound.  Returns `profile_points` from the
          final converged solve, matching the forward sibling.  The
          per-iteration ideal-gas Fanno diagnostic is suppressed during
          the inverse drive (it would otherwise fire dozens of times
          with slightly different cumulative integrals, defeating
          Python's dedupe).

      Failure mode across the suite: when the residual bracket cannot
      be established subsonically, components raise `ChokedFlowError`
      with `mdot_choked` populated — directly usable by the planned
      network-solver clamp (R3.5).  Helper widens its retreat-catch to
      include `ValueError` since CoolProp's internal Brent raises that
      class when constant-area K-loss inversion is pushed past the
      Fanno choke.

      Covered by three new tests in [examples.py](examples.py):
      `test_dmdot_dT_roundtrip` (every component recovers its forward
      mdot to <5e-3 relative error; observed errors range 1e-7 for
      constricted Valve to 2e-5 for Line_Segment),
      `test_dmdot_dT_choke_raises` (Plain Valve and Bend both raise
      `ChokedFlowError` with mdot_choked populated when P2 = 1 kPa
      against 100 psi inlet), and `test_orifice_dmdot_dT_vs_dA`
      (regression guard that the Cd fixed point reaches the same answer
      as a direct `compressible_dA` Mode 2 call when `Cd_override`
      eliminates the fixed point — observed drift 0 to machine
      precision).

      **Network-level wiring (`Compressible_Network.solve`).**
      Auto-detection is one line at the top of `solve()`:
      `inverse_of = [self._nodes[e.to_node].P_spec_Pa is not None for e
      in self._edges]`.  The residual function consults `inverse_of[e_idx]`
      twice -- once when choosing between `walk_edge` (forward) and
      `walk_edge_inverse`, and once when assembling the pipe-equation
      residual (`mdot - mdot_solved` for inverse edges, `walked_P -
      P_outlet` for forward).  Mass and energy balances are untouched
      -- they consume the unknown-vector `mdot[e_idx]` and the
      `walked_h_out[e_idx]` populated by both walkers.

      `walk_edge_inverse` itself dispatches on edge shape:

        * **Single-component edge** (the common relief-valve case):
          delegates directly to `component.dmdot_dT(fs, P2=P_target)`.
          This is essential for correctness on components with internal
          throats (Orifice, Valve / CheckValve with `minimum_diameter`):
          the component knows its own throat geometry and routes through
          `compressible_dA` Mode 2's exact bracketing.  An earlier
          implementation that did its own brentq with
          `mdot_choked = G_max * A_min` over component inlet areas
          consistently failed to bracket on constricted Valves and
          Orifices (throat is much smaller than the inlet area), then
          fell through to a penalty `mdot_solved = 0` which LM accepted
          as a trivial solution on single-edge two-P-spec networks.
        * **Multi-component edge**: builds a forward closure that walks
          the entire chain and drives mdot via the shared
          `_solve_mdot_for_outlet_P` helper.  Choke bound uses the cheap
          `_ideal_gas_G_max * A_min` estimate; the helper's retreat
          loop handles overestimates.
        * **Sealed CV on the edge** (highest priority): returns the
          clamped sealed-state outlet and `sealed=True`, identical to
          the forward `walk_edge` sealed-CV path.  The residual
          dispatcher then uses the sealed-edge `mdot / mdot_ref`
          residual rather than the inverse form.
        * **Reverse-flow trial** (`mdot_trial < 0`): not supported on
          inverse edges; returns `mdot_solved = 0` (driving the
          inverse residual toward zero, away from negative territory)
          plus a once-per-solve UserWarning via `_warn_once`.  Relief
          valves are physically forward-only so this is sufficient for
          the canonical use case.
        * **`P_target >= P_in_trial`** (transient infeasibility during
          LM): subsonic flow cannot reduce pressure to that target;
          returns `mdot_solved = 0` plus a once-per-solve UserWarning.
        * **ChokedFlowError from `component.dmdot_dT` or the inner
          brentq**: clamps `mdot_solved = exc.mdot_choked`, leaves AS at
          the choked outlet state, returns normally.  The Newton then
          drives `mdot[e_idx]` to the choke value.  This addresses
          R3.5's intent (consume `ChokedFlowError.mdot_choked` to clamp
          the solver) on the inverse path; the forward `walk_edge`
          ChokedFlowError handler still uses its own penalty-pressure
          smoothing, since the forward residual form doesn't have a
          natural place to consume the mdot value.

      **LM finite-difference step.**  When any edge is inverse-mode,
      `solve()` passes `diff_step=1e-5` to `least_squares` (vs the
      default `sqrt(machine_eps) ~ 1.5e-8`).  The cheap brentq tolerance
      inside `walk_edge_inverse` (`xtol_factor=1e-8`) has a noise floor
      that swamps the FD signal at the default step size, stalling LM
      on a non-zero residual.  1e-5 sits well above the brentq noise
      floor with negligible truncation-error penalty.  Forward-only
      networks keep the default step.

      **Energy-balance guard at P-spec outlets.**  An LM trial that
      flips an edge's mdot temporarily negative drives the implicit
      `Q_ext_eff` at a connected P-spec outlet positive (apparent
      "supply" into the network).  Previously the energy-balance loop
      read `node.T_spec_K` to evaluate the supply enthalpy, which crashes
      with `TypeError` on a P-spec-only outlet.  Fixed: the supply
      branch now `pass`-es when `T_spec_K is None`, treating that
      transient configuration as zero supply contribution so LM can
      step away.  At the converged solution the branch never fires
      (real outlets always have `Q_ext_eff < 0`).

      **Choke pre-screen optimization (the perf big win).**  The
      compressible kernels (`compressible_changing_area_K`,
      `compressible_K`) each run a `_choke_pre_screen` that calls
      `choked_mass_flux` -- a real-gas isentropic march that costs
      ~150-200 ms per call on HEOS mixtures (~80 CoolProp PT updates
      grid-scanning an isentrope).  This pre-screen is necessary for
      safety on arbitrary mdot inputs but is redundant inside any
      brentq driver that has already capped `mdot_hi ≤ 0.95 * mdot_choked`
      from a prior choke determination.  Both kernels now accept a
      `skip_choke_check=False` parameter; the brentq closures inside
      `compressible_dA` Mode 2, `Valve.dmdot_dT`, `Bend.dmdot_dT`, and
      `Contraction_Expansion.dmdot_dT` all pass `True`.  In tandem,
      Valve/Bend/Contraction_Expansion `dmdot_dT` switched from the
      expensive real-gas `choked_mass_flux` to a cheap
      `_ideal_gas_G_max * A` for the brentq upper bound (the
      real-gas call is now deferred to the bracket-failure branch
      where the structured `ChokedFlowError` payload is needed).
      Combined effect on a methane-mix benchmark (100 → 90 psi,
      `benchmark_dmdot.py`):

        * `Valve.dmdot_dT` (plain): 1697 ms → 11 ms (~150x)
        * `Bend.dmdot_dT`: 2922 ms → 18 ms (~160x)
        * Single-edge inverse network solve:
            - Valve plain:        22.2 s → 3.7 s (6.0x)
            - Bend:               12.6 s → 3.3 s (3.8x)
            - Line_Segment 200ft:  9.5 s → 8.9 s (1.1x)
            - Valve constricted:  28.0 s (wrong mdot) → 8.6 s (correct)
            - Orifice:            33.2 s (wrong mdot) → 10.7 s (correct)

      Constricted-throat components (Orifice, constricted Valve) still
      pay ~500-770 ms per forward `dP_dT` because `compressible_dA`
      Mode 1 retains its own upfront `choked_mass_flux` call -- that
      one is load-bearing (used both for the brentq bound and for the
      `P2_choked` branch decision).  Splitting Mode 2's choke logic to
      defer the real-gas call would remove the remaining bottleneck
      but is a deeper restructure.

      **Coverage.**  Three new tests in
      [examples.py](examples.py) at the component level:
      `test_dmdot_dT_roundtrip`, `test_dmdot_dT_choke_raises`,
      `test_orifice_dmdot_dT_vs_dA`.  Three at the network level in
      [compressible_network.py](compressible_network.py):
      `_test_inverse_single_relief_valve`,
      `_test_inverse_relief_from_junction`,
      `_test_inverse_choked_relief`.  All five pre-existing
      network self-tests
      (`_test_single_segment_forward`, `_test_parallel_two_branches`,
      `_test_mixing_junction`, `_test_orifice_subsonic`,
      `_test_orifice_choke`) still pass -- the kernel signature
      additions (`skip_choke_check`) default to the original behavior
      and the energy-balance guard only changes behavior on the
      previously-crashing path.

      **Limitations / known issues.**

        * No opt-out kwarg.  Auto-detection is hard-coded; if a user
          needs an edge with a P-spec downstream to run in forward
          mode anyway (e.g. for diagnostic comparison), they would need
          to patch `inverse_of` manually.  Add `force_forward_edges=[...]`
          to `solve()` if this comes up.
        * Reverse-flow on inverse edges is not modeled.  The penalty
          mdot pushes LM back toward positive flow; an edge that
          legitimately needs to reverse cannot be inverse-mode.  Use
          the forward path (downstream junction instead of P-spec) for
          those edges.
        * `Contraction_Expansion.dmdot_dT` raises `NotImplementedError`
          on the expansion direction (kinetic recovery typically
          dominates K-loss so the residual sign reverses).  An inverse
          edge containing only an expansion would crash.

- **R3 [larger].** Switch the valve component to an ISA-75.01 / IEC-60534-2-1
  sizing model parameterized by Cv (or Kv) and x_T, with explicit choked-flow
  branch. Cv / Kv are already accepted per recent commits; the data is there.
  `compressible_dA`'s two-stage structure is the natural place to host the
  expansion-factor `Y` correction once `x_T` is wired in.

- **R3.5 [partial — inverse-path covered by R2.7; forward path open].**
  Have `compressible_network.walk_edge` catch `ChokedFlowError`
  specifically and consume `mdot_choked` from the exception to clamp
  the solver's `dmdot` step.

  *Inverse-path status:* `walk_edge_inverse` already consumes the
  payload -- on `ChokedFlowError` it sets `mdot_solved = exc.mdot_choked`
  and the `(mdot - mdot_choked)/mdot_ref` residual drives the Newton
  to the clamp value in one step.

  *Forward-path status (still open):* `walk_edge` catches
  `ChokedFlowError` but only to compute a smoothed penalty-pressure
  (`(mdot_choked / abs_mdot)**2 * exc.P_outlet`) that gives LM a
  monotonic gradient back toward subsonic.  The `mdot_choked` value
  itself is discarded.  A more targeted retreat that snaps `mdot[e_idx]`
  to `0.99 * exc.mdot_choked` in the next residual call would converge
  in fewer LM iterations.

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

- **R5 [tier 2, cavitation check ✅ DONE — component-level; network surfacing
  still open].** New optional constructor args landed:
  - `Incompressible_Fluid(..., vapor_pressure=None, critical_pressure=None)`
    — `vapor_pressure_si` already existed; `critical_pressure_si` is the
    new slot, consumed only by the `F_F` factor in the choked threshold.
  - `Base_Valve(..., F_L=None)` and `Base_CheckValve(..., F_L=None)` in
    [component_classes.py](component_classes.py), validated to `(0, 1]`,
    round-tripped through `to_dict` / `from_dict`. The compressible side
    ignores `F_L` (it uses `minimum_diameter` for sonic choke); only the
    incompressible Valve / CheckValve read it.

  Implementation: shared module-level helper
  `_valve_cavitation_check(component, fluid, P_inlet, dP_perm)` in
  [incompressible.py](incompressible.py) runs the three-regime gate
  (flashing → `RuntimeError`, choked-cavitating → `RuntimeError`,
  incipient → `UserWarning`) when `F_L`, `vapor_pressure_si`, and
  `P_inlet` are all present. Silent no-op otherwise so existing call
  sites continue to behave identically. `Valve.dP` and `CheckValve.dP`
  gained an optional `P_inlet=None` kwarg (mirroring `Orifice.dP`) so
  the check can fire on the forward path; `Valve.dmdot` and
  `CheckValve.dmdot` (see R5.5) pass `P_inlet` automatically since they
  have it natively.

  `F_F = 0.96 - 0.28 * sqrt(Pv/Pc)` when `critical_pressure_si` is
  supplied; falls back to `F_F = 0.96` (the low-`Pv/Pc` asymptote)
  otherwise. The fallback is conservative — it yields a slightly higher
  choked-cavitation threshold than the true value, so spurious
  early-trips are not introduced when `Pc` is unknown.

  Reference: [LANL valve cavitation review](https://www.osti.gov/biblio/10155405),
  page 13 and following.

  Coverage: `test_incompressible_valve_cavitation` in
  [examples.py](examples.py) exercises the silent path (F_L=None),
  flashing, choked-cavitating, incipient warning, the `CheckValve`
  mirror, and round-trip preservation when F_L is set in the
  non-cavitating regime.

  **Open follow-on:** network-level surfacing. `Network.solve()` does
  not yet pass `P_inlet` into the per-component `dP` call sites, so the
  cavitation check is currently opportunistic — it fires when a caller
  invokes `component.dP(..., P_inlet=...)` or `component.dmdot(...)`
  directly, but not during a network solve. R4's negative-pressure
  floor is the natural place to thread `P_inlet` through; both are
  small additions to `_component_signed_dP` and the residual loop.

- **R5.5 [✅ DONE — `dmdot` on every incompressible component; Orifice.dP
  cavitation sign bug fixed].** Symmetric to the compressible R2.7
  work: every incompressible component class now carries a
  `dmdot(fluid, P_inlet, P_outlet) -> mdot_kg_s` method that mirrors
  the contract of `dP` but takes the downstream pressure as input.
  Most paths are analytic (closed-form rearrangement of the dP
  equation); a couple need a small inner loop:

  - **`Valve` / `CheckValve` / `Contraction_Expansion`:** analytic.
    `Q = A * sqrt(2 * dP_drop / (K * rho))` for the K-only components;
    `Contraction_Expansion` solves the quadratic
    `dP_static = 0.5 * rho * Q^2 * beta` where `beta` is geometry-only
    (handles both contraction and expansion directions — unlike the
    compressible analog which raises `NotImplementedError` on
    expansions for residual-sign reasons that don't apply here).
  - **`Bend`:** K(Re) fixed point — solve Q analytically from K, refresh
    Re and K, iterate. 2–3 iterations typical.
  - **`Orifice`:** Cd(Re) RHG fixed point + cavitation σ check (same
    thresholds the forward path uses).
  - **`Line_Segment`:** 1-D `brentq` on `mdot`, residual is
    `pressure_profile(P0=P_inlet, mdot)[-1]['P_Pa'] - P_outlet`.
    Closed form is infeasible because each slice carries its own
    f(Re), the staircase profile can change area between slices, and
    elevation `-rho*g*dz` is mdot-independent. Bracket: `[1e-9,
    mdot_hi]` with `mdot_hi` from a fully-turbulent `f~0.02` estimate
    using the segment's minimum `D_h` / `A`, expanded up to ~3 decades
    if the initial bracket fails.

  Returns `mdot` in kg/s (not pint, no `fs` to mutate — incompressible
  has no thermodynamic state container).

  Coverage: `test_incompressible_dmdot_roundtrip` in
  [examples.py](examples.py) round-trips every component (analytic
  paths to machine precision ~1e-16, K(Re) fixed point to ~1e-11,
  brentq Line_Segment to ~1e-13).

  **Network not wired** (deliberate, scope-limited): the incompressible
  network's existing residual `(P_from - P_to) + dP(Q) == 0` is already
  symmetric in which end is pinned — unlike the compressible case
  where the forward walk takes `mdot` as input and produces
  `walked_P`. No `walk_edge_inverse` analog is needed for correctness.
  The new `dmdot` methods are useful as standalone analytic helpers
  (post-solve diagnostics, GUI flow-rate-driven workflows, future
  control-loop work).

  **Orifice.dP cavitation sign bug fixed as part of this work.** The
  pre-existing block in `Orifice.dP` gated on `if dP_perm > 0.0`,
  which never fires since `dP_perm <= 0` for forward flow — so the
  Orifice cavitation check was effectively dead code in `dP`
  ([incompressible.py](incompressible.py#L856-L877), formerly L840-L864).
  Three concomitant bugs cleaned up: gating now reads `dP_perm < 0.0`;
  sigma uses `dP_abs = -dP_perm` (was dividing by negative dP_perm,
  giving a negative sigma that always passed the `< sigma_choked`
  test for the wrong reason); the dead `P_out = P_inlet - dP_perm`
  line is gone. Covered by `test_incompressible_orifice_cavitation`.

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
  `Line_Segment.dP_dT` entry (R7); incompressible Valve / CheckValve
  cavitation now flagged by the ISA-75.01 three-regime gate when `F_L`
  and `vapor_pressure_si` are supplied (R5); Orifice cavitation check
  on the forward `dP` path resurrected from a long-standing dead-code
  bug (R5.5).*

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
