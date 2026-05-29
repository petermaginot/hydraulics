"""compressible_network.py

Compressible-flow pipe-network solver.  Subclass of Network from network.py.

Reuses the Node / Edge data classes, the _reversed_component geometry
helper, and the general "node pressure + edge flow" formulation.  Replaces
the residual function with a joint Newton on (P_n, T_n, mdot_e) and the
incompressible per-edge dP() call with the compressible dP_dT() walk
through a CoolProp AbstractState.

Boundary conventions
--------------------
Each inlet (node with positive external supply) must specify BOTH P and T.
Outlets typically specify only P; their T emerges from the energy balance.
Q_ext is a mass flow rate [kg/s], or equivalently any pint Quantity with
mass or substance (molar / standard-volume) flow dimensions; actual
volumetric flow is rejected because the conversion depends on local
density.

Assumptions
-----------
-- Uniform composition.  All streams share the same composition, supplied
   as a single CoolProp AbstractState argument to solve().
-- No heat transfer.  q_wall is not exposed by the network solver; every
   edge runs adiabatic.
-- Mass flow is the per-edge solver variable (signed in nominal
   from->to direction).  Volumetric flow is not conserved along compressible
   pipes.

Solver formulation
------------------
Unknowns:
    -- P_n at each non-P-spec node;
    -- T_n at each non-T-spec node;
    -- signed mdot_e on each edge.

Equations:
    -- Per edge: walked outlet pressure (computed by reseeding AS at the
       flow-inlet node and stepping dP_dT through the components in flow
       order) equals the P at the flow-outlet node.  ONE residual per edge.
    -- Per non-P-spec node: mass balance.  Sum of signed edge flows + Q_ext
       = 0.
    -- Per non-T-spec node: energy balance.  Sum of (|mdot| * h) over
       inflows = sum over outflows, where inflow enthalpy is the walked
       outlet enthalpy of the contributing edge and outflow enthalpy is
       h(P_n, T_n) at the node's mixed state.
"""

import csv
import json
import math
import os
import warnings
import numpy as np
from scipy.optimize import root, least_squares

from component_classes import ureg
from network import (
    Node, Edge, Network,
    _reversed_component, _to_si_or_none, _to_pint_qty,
)
from compressible_flow import (
    _build_phase_limits, _safe_update_PT, _safe_flowstate_update_PT,
    _is_sealed_check_valve, _sealed_outlet_PT,
    _solve_mdot_for_outlet_P, _ideal_gas_G_max,
    FlowState, ChokedFlowError,
)


# Penalty walked-outlet pressure returned by walk_edge when a component
# dP_dT raises during an LM trial step.  Chosen well below any physical
# node pressure so the pipe-equation residual is strongly negative (~ -1
# after normalization by P_ref), signalling "infeasible trial" to the
# trust-region solver without crashing the whole solve.
_WALK_FAIL_P_PA = 1.0

# Inverse-mode penalty mdot returned by walk_edge_inverse when the trial
# state is infeasible (reverse flow on an inverse edge, P_target >= P_in,
# or the per-edge brentq kernel fails outright).  With mdot_solved = 0
# the pipe-equation residual reduces to mdot[e_idx] / mdot_ref, identical
# in form to the sealed-CV residual: zero only at mdot = 0, with positive
# slope so LM sees a clean gradient pulling the trial mdot back toward
# zero.  Once mdot crosses back into the feasible (positive) region the
# next trial re-enters the brentq branch and converges normally.
_INVERSE_PENALTY_MDOT_SOLVED = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qty_to_mdot_kgs_compressible(qty, abstract_state):
    """Convert a flow-rate pint Quantity to mass flow [kg/s] for a compressible
    network (uniform composition).

    Accepts mass ([mass]/[time]) or molar / standard-volume ([substance]/[time],
    e.g. mol/s or mmscf/day) flow.  Rejects actual volumetric flow because the
    density varies along the network -- there's no single rho to convert with.
    """
    if qty is None:
        return None
    dim = qty.dimensionality
    if dim == {"[mass]": 1, "[time]": -1}:
        return qty.to("kg/s").magnitude
    if dim == {"[substance]": 1, "[time]": -1}:
        mm_kg_per_mol = abstract_state.molar_mass()
        return qty.to("mol/s").magnitude * mm_kg_per_mol
    if dim == {"[length]": 3, "[time]": -1}:
        raise ValueError(
            "Compressible_Network: actual volumetric flow ([length]^3/[time]) "
            "is ambiguous because density varies along the network.  Use mass "
            "flow ([mass]/[time]) or molar / standard-volume flow "
            "([substance]/[time], e.g. kg/s, mmscf/day, mol/s)."
        )
    raise ValueError(f"Q_ext has unrecognized dimensions {dict(dim)}.")


# ---------------------------------------------------------------------------
# Compressible_Network
# ---------------------------------------------------------------------------

class Compressible_Network(Network):
    """Compressible-flow pipe-network solver.  See module docstring for
    formulation, boundary conventions, and assumptions.
    """

    REGIME = "compressible"

    @classmethod
    def _default_component_classes(cls):
        """Override Network._default_component_classes to plug in the
        compressible_flow.* component subclasses on load."""
        from compressible_flow import (
            Line_Segment, Bend, Valve, CheckValve, Contraction_Expansion,
            Orifice,
        )
        return {
            "line_segment":          Line_Segment,
            "bend":                  Bend,
            "valve":                 Valve,
            "check_valve":           CheckValve,
            "contraction_expansion": Contraction_Expansion,
            "orifice":               Orifice,
        }

    def _add_node_from_dict(self, node_payload):
        """Replay one serialized compressible node: forwards T_K to
        add_node() in addition to elevation / P / Q_ext."""
        from component_classes import ureg as _ureg
        kwargs = {"elevation": _ureg.Quantity(node_payload["elevation_m"], "m")}
        if node_payload.get("P_Pa") is not None:
            kwargs["P"] = _ureg.Quantity(node_payload["P_Pa"], "Pa")
        if node_payload.get("T_K") is not None:
            kwargs["T"] = _ureg.Quantity(node_payload["T_K"], "K")
        Q_dict = node_payload.get("Q_ext")
        if Q_dict is not None:
            kwargs["Q_ext"] = _ureg.Quantity(
                float(Q_dict["magnitude"]), Q_dict["unit"],
            )
        if node_payload.get("area_m2") is not None:
            kwargs["area"] = _ureg.Quantity(
                float(node_payload["area_m2"]), "m**2",
            )
        self.add_node(node_payload["name"], **kwargs)

    def add_node(self, name, *, elevation=0.0, P=None, T=None, Q_ext=None,
                 diameter=None, area=None, abstract_state=None):
        """Add a node.

        Args:
            name           : str, unique label.
            elevation      : pint Quantity or float (m if float).  Default 0.
            P              : pint Quantity or float (Pa if float).  Optional.
            T              : pint Quantity or float (K if float).  Optional.
            Q_ext          : pint Quantity (mass or molar/std-volume flow) or
                             float (kg/s if float).  Optional.  The Quantity is
                             stored as-is; conversion to kg/s is deferred to
                             solve() time, where it uses the abstract state's
                             molar mass for molar/std-volume inputs.
            diameter       : pint Quantity or float (m if float).  Diameter of
                             the cross-section associated with the node.
                             The node's (P, T) is interpreted as the static
                             state at this area.  Mutually exclusive with
                             `area`.  Optional; default None means "resolve
                             at solve() time from the first connected
                             component's near-end area".
            area           : pint Quantity or float (m^2 if float).  Direct
                             area form of `diameter`.  Optional.
            abstract_state : Kept for API stability; no longer needed at
                             add_node time because conversion is deferred.

        Specifying both P and Q_ext is rejected (over-constrains the node).
        Specifying both `diameter` and `area` is also rejected.
        """
        if name in self._nodes:
            raise ValueError(
                f"Compressible_Network.add_node: node {name!r} already exists."
            )

        elev_si = _to_si_or_none(elevation, "m")
        if elev_si is None:
            elev_si = 0.0

        P_si  = _to_si_or_none(P, "Pa")
        T_si  = _to_si_or_none(T, "K")
        Q_qty = _to_pint_qty(Q_ext, "kg/s")

        if P_si is not None and Q_qty is not None:
            raise ValueError(
                f"Compressible_Network.add_node: node {name!r} cannot specify "
                f"both P and Q_ext."
            )

        if diameter is not None and area is not None:
            raise ValueError(
                f"Compressible_Network.add_node: node {name!r} cannot specify "
                f"both diameter and area."
            )
        if area is not None:
            area_si = _to_si_or_none(area, "m**2")
        elif diameter is not None:
            D_si = _to_si_or_none(diameter, "m")
            area_si = math.pi / 4.0 * D_si * D_si if D_si is not None else None
        else:
            area_si = None
        if area_si is not None and area_si <= 0.0:
            raise ValueError(
                f"Compressible_Network.add_node: node {name!r} area must be "
                f"positive."
            )

        self._nodes[name] = Node(
            name=name, elevation_m=elev_si,
            P_spec_Pa=P_si, Q_ext_spec_qty=Q_qty, T_spec_K=T_si,
            area_m2=area_si,
        )
        self._node_order.append(name)

    def solve(self, abstract_state, *, P_init=None, T_init=None,
              mdot_init_kgs=1.0, verbose=False, xtol=1e-8, maxfev=5000,
              progress_callback=None):
        """Solve the compressible network.

        Args:
            abstract_state : CoolProp AbstractState for the (uniform) gas
                             composition.  Phase-envelope limits are
                             precomputed once and forwarded to every dP_dT
                             call.  AS is mutated repeatedly during the solve.
            P_init         : float (Pa) or None.  Initial guess for free
                             pressures.  Defaults to the mean of P-spec
                             values.
            T_init         : float (K) or None.  Initial guess for free
                             temperatures.  Defaults to the mean of T-spec
                             values, or 300 K if no T is spec'd.
            mdot_init_kgs  : float OR dict {edge_name: kg/s}.  Default 1.0
                             (uniform).  On networks with junctions a
                             scalar guess is mass-imbalanced at the
                             junction, which can leave the LM solver stuck
                             at the trivial "all zero" minimum -- pass a
                             dict that satisfies mass balance at every
                             junction to avoid this.
            verbose        : bool.
            xtol           : float, LM tolerance on unknowns / residual.
            maxfev         : int, max function evaluations.
            progress_callback : callable(nfev, residual_norm) or None.
                             Called from inside the residual function after
                             each evaluation.  Lets a GUI surface progress
                             during a long solve.  Exceptions raised by the
                             callback are swallowed so a buggy reporter
                             can't abort the solve.

        Returns:
            dict with keys "P_Pa", "T_K", "mdot_kgs", "Q_ext_mdot_kgs",
            "component_outlet_PT", and "converged".  The first four match
            the incompressible NetworkResult conventions (with mass flow
            and temperature added).  "component_outlet_PT" maps each
            edge name to a list of (P_Pa, T_K) tuples giving the flow-
            direction outlet conditions of each component, indexed by
            the original edge.components order (regardless of whether
            the converged flow ran forward or reverse on that edge).
        """
        node_names = list(self._node_order)
        n_nodes    = len(node_names)
        n_edges    = len(self._edges)

        if n_nodes == 0 or n_edges == 0:
            raise ValueError(
                "Compressible_Network.solve: network has no nodes or edges."
            )

        self._validate_edge_elevations()

        # Convert each node's Q_ext_spec_qty to mass flow [kg/s] now that
        # the AbstractState (and hence molar mass) is available.  Validate
        # that any positive supply also has T spec'd, so we can evaluate
        # the inflow enthalpy.
        mdot_ext_spec = {}   # node name -> kg/s or None
        for n in node_names:
            node = self._nodes[n]
            mdot_ext_spec[n] = _qty_to_mdot_kgs_compressible(
                node.Q_ext_spec_qty, abstract_state
            )
            if mdot_ext_spec[n] is not None and mdot_ext_spec[n] > 0.0 \
                    and node.T_spec_K is None:
                raise ValueError(
                    f"Compressible_Network.solve: node {n!r} has a positive "
                    f"Q_ext (supply into the network) but no T spec.  Supply "
                    f"streams must specify their temperature."
                )

        # Index nodes and classify by spec.
        idx_of = {n: i for i, n in enumerate(node_names)}
        p_free_idx = [i for i, n in enumerate(node_names)
                      if self._nodes[n].P_spec_Pa is None]
        p_spec_idx = [i for i, n in enumerate(node_names)
                      if self._nodes[n].P_spec_Pa is not None]
        t_free_idx = [i for i, n in enumerate(node_names)
                      if self._nodes[n].T_spec_K is None]
        t_spec_idx = [i for i, n in enumerate(node_names)
                      if self._nodes[n].T_spec_K is not None]

        if not p_spec_idx:
            raise ValueError(
                "Compressible_Network.solve: at least one node must have a "
                "specified pressure to anchor the solution."
            )

        # Auto-detect inverse-mode edges: any edge whose downstream node
        # (to_node) is P-spec.  Inverse-mode edges replace the standard
        # forward pipe-equation residual (walked_P_out - P_node) with a
        # locally solved (mdot - mdot_solved) residual driven by
        # _solve_mdot_for_outlet_P.  Advantages: analytic Jacobian column
        # for the edge's mdot (+1 / mdot_ref), brentq-bracketed local
        # solve robust near choke, and choke-aware via ChokedFlowError
        # clamp.  Edges feeding a junction keep the forward path
        # unchanged.  See network.md "Compressible solver" section.
        inverse_of = [
            self._nodes[edge.to_node].P_spec_Pa is not None
            for edge in self._edges
        ]

        # Build adjacency for mass / energy balance.
        # adj[i] = list of (edge_idx, sign) where sign=+1 means positive Q_e
        # carries fluid INTO this node, sign=-1 means it carries fluid OUT.
        adj = [[] for _ in range(n_nodes)]
        for e_idx, edge in enumerate(self._edges):
            adj[idx_of[edge.from_node]].append((e_idx, -1))
            adj[idx_of[edge.to_node]  ].append((e_idx, +1))

        # Initial guesses.
        if P_init is None:
            P_init = sum(self._nodes[node_names[i]].P_spec_Pa for i in p_spec_idx) \
                     / len(p_spec_idx)
        if T_init is None:
            if t_spec_idx:
                T_init = sum(self._nodes[node_names[i]].T_spec_K for i in t_spec_idx) \
                         / len(t_spec_idx)
            else:
                T_init = 300.0

        n_p_free = len(p_free_idx)
        n_t_free = len(t_free_idx)

        # Per-edge initial mdot.  Accept either a scalar (broadcast) or a
        # dict keyed by edge name.  A scalar broadcast guess is wrong at
        # any junction with unequal in/out edge counts, so passing a dict
        # is recommended for any network beyond single-path.
        if isinstance(mdot_init_kgs, dict):
            mdot0_per_edge = np.empty(n_edges)
            for e_idx, edge in enumerate(self._edges):
                if edge.name not in mdot_init_kgs:
                    raise KeyError(
                        f"Compressible_Network.solve: mdot_init_kgs dict "
                        f"missing entry for edge {edge.name!r}."
                    )
                mdot0_per_edge[e_idx] = float(mdot_init_kgs[edge.name])
            mdot_ref_scalar = max(
                abs(v) for v in mdot_init_kgs.values()
            ) or 1.0
        else:
            mdot0_per_edge = np.full(n_edges, float(mdot_init_kgs))
            mdot_ref_scalar = max(abs(float(mdot_init_kgs)), 1e-6)

        x0 = np.concatenate([
            np.full(n_p_free, float(P_init)),
            np.full(n_t_free, float(T_init)),
            mdot0_per_edge,
        ])

        # Precompute phase-envelope limits once -- _build_phase_limits is
        # expensive for HEOS mixtures and the composition is constant.
        T_cric, P_bar, T_c, P_c = _build_phase_limits(abstract_state)

        # Resolve the cross-section flow area at every node.  Used to seed
        # the FlowState at every edge walk so the node's (P, T) is
        # interpreted as the static state at the node's own area, not at
        # the first downstream component's inlet.  The first component's
        # _area_match then absorbs any area discontinuity between node and
        # component automatically.  Default order: explicit node area_m2,
        # else the first connected edge's near-end component area, else
        # BFS through zero-component edges to a node with a resolved area.
        node_area = [None] * n_nodes
        for i, name in enumerate(node_names):
            spec = self._nodes[name].area_m2
            if spec is not None:
                node_area[i] = float(spec)
        for i, name in enumerate(node_names):
            if node_area[i] is not None:
                continue
            for edge in self._edges:
                if not edge.components:
                    continue
                if edge.from_node == name:
                    node_area[i] = float(edge.components[0].inlet_area_si)
                    break
                if edge.to_node == name:
                    node_area[i] = float(edge.components[-1].outlet_area_si)
                    break
        changed = True
        while changed:
            changed = False
            for edge in self._edges:
                if edge.components:
                    continue
                fi = idx_of[edge.from_node]
                ti = idx_of[edge.to_node]
                if node_area[fi] is None and node_area[ti] is not None:
                    node_area[fi] = node_area[ti]
                    changed = True
                elif node_area[ti] is None and node_area[fi] is not None:
                    node_area[ti] = node_area[fi]
                    changed = True
        if any(a is None for a in node_area):
            unresolved = [node_names[i] for i, a in enumerate(node_area)
                          if a is None]
            raise ValueError(
                f"Compressible_Network.solve: cannot infer flow area for "
                f"node(s) {unresolved!r}; specify a diameter on at least "
                f"one of them."
            )

        # Cache reversed-shadow components, keyed by id(original).
        reversed_cache = {}

        def unpack(x):
            P = np.empty(n_nodes)
            T = np.empty(n_nodes)
            for i in p_spec_idx:
                P[i] = self._nodes[node_names[i]].P_spec_Pa
            for j, i in enumerate(p_free_idx):
                P[i] = x[j]
            for i in t_spec_idx:
                T[i] = self._nodes[node_names[i]].T_spec_K
            for j, i in enumerate(t_free_idx):
                T[i] = x[n_p_free + j]
            mdot = x[n_p_free + n_t_free:]
            return P, T, mdot

        def walk_edge(edge, mdot_e, P, T):
            """Walk the components on one edge in flow direction.

            Returns (P_outlet, T_outlet, h_outlet, inlet_i, outlet_i,
            sealed).  `sealed` is True iff the walked path contains a
            sealing-K check-valve shadow (i.e. mdot_e < 0 through a CV);
            the caller uses this to switch the pipe residual from the
            standard walked - P_outlet form to a mdot penalty, since a
            sealed CV's pipe equation is over-determined (no flow is
            permitted regardless of the pressure mismatch across it).
            """
            if mdot_e >= 0.0:
                inlet_i  = idx_of[edge.from_node]
                outlet_i = idx_of[edge.to_node]
                comps    = edge.components
            else:
                inlet_i  = idx_of[edge.to_node]
                outlet_i = idx_of[edge.from_node]
                rev_list = []
                for c in edge.components:
                    key = id(c)
                    if key not in reversed_cache:
                        reversed_cache[key] = _reversed_component(c)
                    rev_list.append(reversed_cache[key])
                comps = list(reversed(rev_list))

            P_in_e = P[inlet_i]
            T_in_e = T[inlet_i]
            z_in_e = self._nodes[node_names[inlet_i]].elevation_m
            abs_mdot = abs(mdot_e)

            # If any component in the walked path is a sealed check-valve
            # shadow, short-circuit the whole edge to the clamped sealed-state
            # outlet AND flag the edge as sealed so the residual function can
            # swap to a mdot-penalty form.  Walking downstream components
            # from the CV's clamped outlet drops them into a low-P /
            # high-velocity regime where compressible_pipe_segment can fail
            # to converge.
            if any(_is_sealed_check_valve(c) for c in comps):
                P_out, T_out = _sealed_outlet_PT(P_in_e, T_in_e)
                _safe_update_PT(abstract_state, P_out, T_out,
                                T_cric, P_bar, T_c, P_c)
                return (abstract_state.p(),
                        abstract_state.T(),
                        abstract_state.hmass(),
                        inlet_i, outlet_i, True)

            # Wrap the walk in a try/except so that an LM trial step into a
            # numerically infeasible region (e.g. compressible_pipe_segment
            # failing to converge its slice within the configured energy/dPdL
            # tolerances, or a CoolProp PT update failing at an extreme state)
            # does not abort the whole solve.  On failure, reset AS to inlet
            # conditions and return a penalty walked outlet at _WALK_FAIL_P_PA
            # (~ 1 Pa).  The pipe-equation residual then becomes ~ -1 after
            # normalization by P_ref, which the trust-region LM solver
            # interprets as "step rejected", shrinks its trust radius, and
            # retries from a safer point.
            try:
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                # Seed the FlowState's local area from the inlet node's
                # resolved area so the node's (P, T) is interpreted as
                # the static state at the node, not at the first
                # downstream component.  Any area discontinuity between
                # the node and the first component (or anywhere down the
                # chain) is absorbed automatically by _area_match inside
                # each component's dP_dT.  A zero-component edge falls
                # through the loop and returns the inlet state as the
                # outlet, matching the incompressible solver's implicit
                # zero-dP connector behavior.
                fs = FlowState(
                    abstract_state, abs_mdot,
                    A=node_area[inlet_i], z=z_in_e,
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )
                for c in comps:
                    c.dP_dT(fs)
            except ChokedFlowError as e:
                # mdot-dependent penalty so the LM solver has a gradient
                # to follow back below choke.  A flat _WALK_FAIL_P_PA wall
                # zeros the Jacobian column for mdot and the solver
                # terminates on gtol before it ever steps toward the
                # subsonic root.  Scaling the throat outlet pressure by
                # (mdot_choked / abs_mdot)**2 makes walked_P_out increase
                # monotonically as mdot shrinks back toward choke, so the
                # pipe-equation residual gives a clean step direction.
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                if e.mdot_choked > 0.0 and abs_mdot > e.mdot_choked:
                    ratio = e.mdot_choked / abs_mdot
                    penalty_P = max(_WALK_FAIL_P_PA, e.P_outlet * ratio * ratio)
                else:
                    penalty_P = max(_WALK_FAIL_P_PA, e.P_outlet)
                return (penalty_P,
                        T_in_e,
                        abstract_state.hmass(),
                        inlet_i, outlet_i, False)
            except (RuntimeError, ValueError):
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                return (_WALK_FAIL_P_PA,
                        T_in_e,
                        abstract_state.hmass(),
                        inlet_i, outlet_i, False)

            return (abstract_state.p(),
                    abstract_state.T(),
                    abstract_state.hmass(),
                    inlet_i, outlet_i, False)

        # Set of warning keys already emitted by walk_edge_inverse on
        # this solve.  A noisy LM trial can re-enter walk_edge_inverse
        # with the same infeasibility (reverse flow, P_target >= P_in)
        # dozens of times per residual call -- _warn_once dedupes so the
        # log shows one informational warning per condition per solve.
        warn_keys_seen = set()

        def _warn_once(key, message):
            if key in warn_keys_seen:
                return
            warn_keys_seen.add(key)
            warnings.warn(message, UserWarning)

        def walk_edge_inverse(edge, mdot_trial, P, T):
            """Inverse of walk_edge: solve for mdot such that a forward
            walk through edge.components from the upstream node lands at
            the downstream (P-spec) node's pressure.

            Returns the same 6-tuple as walk_edge, plus mdot_solved:
            (P_outlet, T_outlet, h_outlet, inlet_i, outlet_i, sealed,
            mdot_solved).  The caller stores mdot_solved separately and
            uses it as the (mdot - mdot_solved) pipe-equation residual.

            mdot_trial is consulted only as a sign hint and as a brentq
            seed -- the bracketed local solve is what determines
            magnitude.  Reverse flow (mdot_trial < 0) is not supported
            on inverse-mode edges; returns a penalty mdot_solved so the
            LM solver retreats to positive mdot.
            """
            inlet_i  = idx_of[edge.from_node]
            outlet_i = idx_of[edge.to_node]
            P_in_e   = P[inlet_i]
            T_in_e   = T[inlet_i]
            z_in_e   = self._nodes[node_names[inlet_i]].elevation_m

            mdot_penalty = _INVERSE_PENALTY_MDOT_SOLVED

            # Reverse-flow guard.  Inverse mode is forward-only by
            # design (relief valves, regulators); a trial mdot < 0 from
            # the LM solver is treated as an infeasible region.
            if mdot_trial < 0.0:
                _warn_once(
                    f"inverse_reverse_{edge.name}",
                    f"Compressible_Network.solve: inverse-mode edge "
                    f"{edge.name!r} produced a negative trial mdot; "
                    f"inverse mode does not support reverse flow.  "
                    f"Using penalty residual to drive mdot back positive.",
                )
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                return (_WALK_FAIL_P_PA, T_in_e, abstract_state.hmass(),
                        inlet_i, outlet_i, False, mdot_penalty)

            # Sealed-CV short-circuit takes priority over inverse mode:
            # a sealed CV blocks all flow regardless of the pressure
            # mismatch across it.  Same clamped sealed-state outlet as
            # walk_edge, and we still flag sealed=True so the
            # residual-function dispatcher uses the sealed-edge
            # mdot/mdot_ref residual.
            if any(_is_sealed_check_valve(c) for c in edge.components):
                P_out, T_out = _sealed_outlet_PT(P_in_e, T_in_e)
                _safe_update_PT(abstract_state, P_out, T_out,
                                T_cric, P_bar, T_c, P_c)
                return (abstract_state.p(),
                        abstract_state.T(),
                        abstract_state.hmass(),
                        inlet_i, outlet_i, True, 0.0)

            P_target = P[outlet_i]

            # P_target >= P_in: subsonically impossible (would require
            # negative friction).  Penalty mdot so LM retreats; same
            # once-per-solve warning as reverse flow.
            if P_target >= P_in_e:
                _warn_once(
                    f"inverse_P_invert_{edge.name}",
                    f"Compressible_Network.solve: inverse-mode edge "
                    f"{edge.name!r} has trial inlet P ({P_in_e:.4g} Pa) "
                    f"at or below outlet spec P ({P_target:.4g} Pa); "
                    f"subsonic flow is impossible.  Using penalty residual.",
                )
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                return (_WALK_FAIL_P_PA, T_in_e, abstract_state.hmass(),
                        inlet_i, outlet_i, False, mdot_penalty)

            # Zero-component edge: pure connector, no dP.  Trivially
            # mdot_solved equals whatever satisfies mass balance; with
            # no pressure drop, any mdot is consistent so we just match
            # the LM trial.
            if not edge.components:
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                return (P_in_e, T_in_e, abstract_state.hmass(),
                        inlet_i, outlet_i, False, mdot_trial)

            # Single-component edge (the common relief-valve case):
            # delegate to component.dmdot_dT(fs, P2).  This is much
            # faster than the multi-component brentq path -- the
            # component-level method knows its own internal throat
            # geometry (Cd*A_bore for Orifice, pi*D_min^2/4 for
            # constricted Valve / CheckValve) so it can dispatch to
            # compressible_dA Mode 2's exact bracketing instead of
            # our generic G_max * A_min estimate, which is way too
            # generous for components with an internal restriction and
            # otherwise exhausts the retreat budget.
            if len(edge.components) == 1:
                comp = edge.components[0]
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                fs = FlowState(
                    abstract_state, max(mdot_trial, 1.0e-9),
                    A=node_area[inlet_i], z=z_in_e,
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        comp.dmdot_dT(fs, P2=P_target)
                except ChokedFlowError as exc:
                    _safe_update_PT(abstract_state, exc.P_outlet,
                                    exc.T_outlet, T_cric, P_bar, T_c, P_c)
                    return (exc.P_outlet, exc.T_outlet,
                            abstract_state.hmass(),
                            inlet_i, outlet_i, False, exc.mdot_choked)
                except (RuntimeError, ValueError):
                    _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                    T_cric, P_bar, T_c, P_c)
                    return (_WALK_FAIL_P_PA, T_in_e,
                            abstract_state.hmass(),
                            inlet_i, outlet_i, False, mdot_penalty)
                return (fs.P, fs.T, fs.AS.hmass(),
                        inlet_i, outlet_i, False, fs.mdot)

            # Multi-component edge: drive an outer brentq over mdot
            # whose forward closure walks the whole chain.
            # Choke bound: ideal-gas isentropic sonic mass flux at the
            # smallest component inlet area along the edge.  True choke
            # under friction is lower; the helper's retreat handles it.
            _safe_update_PT(abstract_state, P_in_e, T_in_e,
                            T_cric, P_bar, T_c, P_c)
            G_max = _ideal_gas_G_max(abstract_state)
            A_min = min(c.inlet_area_si for c in edge.components)
            mdot_choked = G_max * A_min

            fs = FlowState(
                abstract_state, max(mdot_trial, 1.0e-9 * mdot_choked),
                A=node_area[inlet_i], z=z_in_e,
                T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                T_critical=T_c, P_critical=P_c,
            )

            def forward_at_mdot(mdot_t):
                _safe_flowstate_update_PT(fs, P_in_e, T_in_e)
                fs.A    = node_area[inlet_i]
                fs.z    = z_in_e
                fs.mdot = mdot_t
                for c in edge.components:
                    c.dP_dT(fs)

            # Brentq tolerance: tight enough that LM's FD perturbation of
            # the upstream-node P / T produces a clean signal in
            # mdot_solved, but not so tight that the inner brentq
            # dominates wall-time.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    mdot_solved = _solve_mdot_for_outlet_P(
                        fs, P_target,
                        forward_at_mdot=forward_at_mdot,
                        mdot_choked=mdot_choked,
                        mdot_guess=max(mdot_trial, 1.0e-3 * mdot_choked),
                        xtol_factor=1.0e-8,
                        rtol=1.0e-9,
                        caller_name=f"walk_edge_inverse[{edge.name}]",
                    )
            except ChokedFlowError as exc:
                _safe_update_PT(abstract_state, exc.P_outlet, exc.T_outlet,
                                T_cric, P_bar, T_c, P_c)
                return (exc.P_outlet, exc.T_outlet, abstract_state.hmass(),
                        inlet_i, outlet_i, False, exc.mdot_choked)
            except (RuntimeError, ValueError):
                _safe_update_PT(abstract_state, P_in_e, T_in_e,
                                T_cric, P_bar, T_c, P_c)
                return (_WALK_FAIL_P_PA, T_in_e, abstract_state.hmass(),
                        inlet_i, outlet_i, False, mdot_penalty)

            return (fs.P, fs.T, fs.AS.hmass(),
                    inlet_i, outlet_i, False, mdot_solved)

        # Reference scales used to non-dimensionalize each residual class.
        # All three are then O(1) at the initial guess and any further
        # imbalance is relative, so the Jacobian is not dominated by one
        # equation type.
        P_ref      = max(abs(P_init), 1.0)
        mdot_ref   = mdot_ref_scalar
        energy_ref = mdot_ref * 1.0e6   # J/s, rough enthalpy scale

        nfev_counter = [0]

        def residuals(x):
            P, T, mdot = unpack(x)
            res = np.empty(n_p_free + n_t_free + n_edges)

            walked_h_out   = np.empty(n_edges)
            walked_P_out   = np.empty(n_edges)
            mdot_solved_of = np.zeros(n_edges)
            outlet_i_of    = [None] * n_edges
            sealed_of      = [False] * n_edges

            for e_idx, edge in enumerate(self._edges):
                if inverse_of[e_idx]:
                    wP, wT, wh, _, outlet_i, sealed, m_solved = \
                        walk_edge_inverse(edge, mdot[e_idx], P, T)
                    mdot_solved_of[e_idx] = m_solved
                else:
                    wP, wT, wh, _, outlet_i, sealed = walk_edge(
                        edge, mdot[e_idx], P, T
                    )
                walked_P_out[e_idx] = wP
                walked_h_out[e_idx] = wh
                outlet_i_of[e_idx]  = outlet_i
                sealed_of[e_idx]    = sealed

            # Mass balance at each non-P-spec node (normalized by mdot_ref).
            for j, i in enumerate(p_free_idx):
                name = node_names[i]
                ext = mdot_ext_spec[name] if mdot_ext_spec[name] is not None else 0.0
                net_in = 0.0
                for e_idx, sign in adj[i]:
                    net_in += sign * mdot[e_idx]
                res[j] = (net_in + ext) / mdot_ref

            # Energy balance at each non-T-spec node.
            # Convention: inflows contribute |mdot| * h_walked_outlet (the
            # enthalpy of fluid arriving at this node); outflows contribute
            # |mdot| * h(P_node, T_node) (the enthalpy of the mixed fluid
            # leaving this node).
            #
            # External supply / withdrawal is part of the balance too:
            #   -- a Q-spec'd supply (Q_ext > 0) brings fluid at the node's
            #      spec'd T (validated above), contributing |Q_ext| * h_supply
            #      to inflows;
            #   -- a Q-spec'd withdrawal contributes |Q_ext| * h_node to
            #      outflows;
            #   -- a P-spec'd node has no Q_ext spec, but the mass balance
            #      derives one (Q_ext_implicit = -net_signed_in).  Including
            #      this implicit term is what makes a P-spec outlet behave
            #      like a real withdrawal.  Without it, the balance reduces
            #      to "inflow enthalpy = 0", which only the trivial mdot=0
            #      solution satisfies -- producing a spurious zero-flow
            #      attractor.
            for j, i in enumerate(t_free_idx):
                P_n = P[i]
                T_n = T[i]
                node = self._nodes[node_names[i]]

                _safe_update_PT(abstract_state, P_n, T_n,
                                T_cric, P_bar, T_c, P_c)
                h_node = abstract_state.hmass()

                sum_in  = 0.0
                sum_out = 0.0
                net_in_edges = 0.0
                for e_idx, sign in adj[i]:
                    m_e = mdot[e_idx]
                    net_in_edges += sign * m_e
                    if (sign * m_e) > 0.0:
                        sum_in  += abs(m_e) * walked_h_out[e_idx]
                    else:
                        sum_out += abs(m_e) * h_node

                # Effective external Q_ext: either an explicit spec or, for
                # P-spec'd nodes, the mass-balance-implied value.
                name = node_names[i]
                if mdot_ext_spec[name] is not None:
                    Q_ext_eff = mdot_ext_spec[name]
                elif node.P_spec_Pa is not None:
                    Q_ext_eff = -net_in_edges
                else:
                    Q_ext_eff = 0.0   # interior junction

                if Q_ext_eff > 0.0:
                    if node.T_spec_K is None:
                        # P-spec node with no T-spec acting as a transient
                        # implicit supply during an LM trial step (only
                        # reachable when a connected edge's trial mdot
                        # flipped sign).  No T to evaluate h_supply
                        # against; skip the supply contribution so the
                        # solver can step away rather than crash.  At the
                        # converged solution this branch never fires
                        # because a P-spec node without T-spec is a
                        # withdrawal, so Q_ext_eff is negative.
                        pass
                    else:
                        # External supply at T_spec.
                        _safe_update_PT(abstract_state, P_n, node.T_spec_K,
                                        T_cric, P_bar, T_c, P_c)
                        h_supply = abstract_state.hmass()
                        sum_in += Q_ext_eff * h_supply
                elif Q_ext_eff < 0.0:
                    sum_out += abs(Q_ext_eff) * h_node

                res[n_p_free + j] = (sum_in - sum_out) / energy_ref

            # Pipe equation per edge: normally walked outlet P == node P at
            # the flow-outlet end (normalized by P_ref).  When the walk
            # crossed a sealed check valve, the standard form is
            # over-determined (the CV blocks flow regardless of the
            # pressure mismatch across it), and a clamp-based walked_P_out
            # can collide with the natural P_outlet value, leaving zero
            # residual and inviting the solver to accept a spurious
            # backflow solution.  For sealed edges we therefore swap to a
            # direct mdot penalty: residual = mdot / mdot_ref, which is
            # zero iff mdot = 0 (the only physical solution through a
            # sealed CV).
            for e_idx in range(n_edges):
                if sealed_of[e_idx]:
                    res[n_p_free + n_t_free + e_idx] = mdot[e_idx] / mdot_ref
                elif inverse_of[e_idx]:
                    # Inverse mode: drive mdot toward the locally-solved
                    # value (brentq inside walk_edge_inverse).  Jacobian
                    # column for this edge's mdot is +1 / mdot_ref
                    # (analytic); only the upstream-node coupling needs
                    # finite differences.
                    res[n_p_free + n_t_free + e_idx] = (
                        mdot[e_idx] - mdot_solved_of[e_idx]
                    ) / mdot_ref
                else:
                    res[n_p_free + n_t_free + e_idx] = (
                        walked_P_out[e_idx] - P[outlet_i_of[e_idx]]
                    ) / P_ref

            nfev_counter[0] += 1
            if progress_callback is not None:
                # Swallow callback exceptions so a buggy reporter (GUI
                # progress hook etc.) cannot abort the solve.
                try:
                    progress_callback(
                        nfev_counter[0], float(np.linalg.norm(res)),
                    )
                except Exception:
                    pass

            return res

        # scipy.optimize.least_squares with the Trust Region Reflective
        # algorithm handles mixed-scale residuals robustly and (crucially)
        # accepts variable bounds, which keep trial pressures positive and
        # temperatures above absolute zero -- otherwise Newton steps in the
        # ill-conditioned early iterations send T deeply negative and the
        # CoolProp PT update fails.  x_scale='jac' lets it normalize
        # unknown magnitudes automatically from the Jacobian.
        n_total = n_p_free + n_t_free + n_edges
        lb = np.empty(n_total)
        ub = np.empty(n_total)
        # P bounds: positive only, capped by 10x P_init.
        lb[:n_p_free] = 1.0
        ub[:n_p_free] = 10.0 * abs(P_init)
        # T bounds: above absolute zero, capped at 5x T_init.
        lb[n_p_free:n_p_free + n_t_free] = 1.0
        ub[n_p_free:n_p_free + n_t_free] = 5.0 * abs(T_init)
        # mdot bounds: signed, bounded by 100x typical magnitude.
        lb[n_p_free + n_t_free:] = -100.0 * mdot_ref_scalar
        ub[n_p_free + n_t_free:] = +100.0 * mdot_ref_scalar

        # LM's default FD step is sqrt(machine_eps) ~ 1.5e-8 relative.
        # On networks with inverse-mode edges the per-edge brentq inside
        # walk_edge_inverse has a noise floor (set by xtol_factor below)
        # that swamps the FD signal at that step size, stalling LM on a
        # non-zero residual.  Bump the FD step to 1e-5 (relative) when
        # any edge is inverse-mode: ~700x larger, well above brentq's
        # 1e-8 noise floor, with negligible truncation-error penalty
        # since the residual is still O(1) at LM tolerances.  Forward
        # edges' walked_P_out has CoolProp precision (~1e-12 relative),
        # so the larger step is also fine for them.
        ls_kwargs = dict(
            bounds=(lb, ub),
            method="trf", x_scale="jac",
            xtol=xtol, ftol=xtol, gtol=xtol, max_nfev=maxfev,
        )
        if any(inverse_of):
            ls_kwargs["diff_step"] = 1.0e-5

        sol_obj = least_squares(residuals, x0, **ls_kwargs)
        residual_norm = float(np.linalg.norm(sol_obj.fun))
        converged = sol_obj.success and residual_norm < 1e-3
        if not converged:
            warnings.warn(
                f"Compressible_Network.solve: least_squares did not "
                f"converge to a root (status={sol_obj.status}, residual "
                f"norm = {residual_norm:.3e}): {sol_obj.message}",
                stacklevel=2,
            )

        if verbose:
            print(
                f"Compressible_Network.solve: status={sol_obj.status}, "
                f"nfev={sol_obj.nfev}, residual norm = {residual_norm:.3e}"
            )

        sol = sol_obj.x

        P_arr, T_arr, mdot_arr = unpack(sol)

        # Recover Q_ext at P-spec nodes from mass balance.
        Q_ext_out = {}
        for i, name in enumerate(node_names):
            node = self._nodes[name]
            if node.P_spec_Pa is not None:
                net_in = 0.0
                for e_idx, sign in adj[i]:
                    net_in += sign * mdot_arr[e_idx]
                Q_ext_out[name] = float(-net_in)
            else:
                Q_ext_out[name] = float(mdot_ext_spec[name] or 0.0)

        # Final per-component walk using the converged solution.  Records
        # the flow-direction outlet (P, T) of each component, indexed by
        # ORIGINAL edge.components order so callers can map results back to
        # their own per-component objects (e.g. GUI inline nodes) without
        # caring whether the converged flow ran forward or reverse.
        component_outlet_PT = {}
        for e_idx, edge in enumerate(self._edges):
            mdot_e = float(mdot_arr[e_idx])
            if mdot_e >= 0.0:
                inlet_i  = idx_of[edge.from_node]
                walk_comps = edge.components
                walk_in_orig_order = True
            else:
                inlet_i  = idx_of[edge.to_node]
                rev_list = []
                for c in edge.components:
                    key = id(c)
                    if key not in reversed_cache:
                        reversed_cache[key] = _reversed_component(c)
                    rev_list.append(reversed_cache[key])
                walk_comps = list(reversed(rev_list))
                walk_in_orig_order = False

            # Sealed-edge short-circuit (mirrors walk_edge above): if any
            # component on the walked path is a sealing-K check-valve shadow,
            # report the clamped sealed-state outlet for every component
            # rather than attempting the downstream walk, which would put
            # subsequent components into a near-vacuum state and fail.
            if any(_is_sealed_check_valve(c) for c in walk_comps):
                P_clamp, T_clamp = _sealed_outlet_PT(
                    float(P_arr[inlet_i]), float(T_arr[inlet_i])
                )
                _safe_update_PT(abstract_state, P_clamp, T_clamp,
                                T_cric, P_bar, T_c, P_c)
                walked_PT = [(P_clamp, T_clamp) for _ in walk_comps]
                if not walk_in_orig_order:
                    walked_PT = list(reversed(walked_PT))
                component_outlet_PT[edge.name] = walked_PT
                continue

            _safe_update_PT(abstract_state, P_arr[inlet_i], T_arr[inlet_i],
                            T_cric, P_bar, T_c, P_c)
            z_inlet_recon = self._nodes[node_names[inlet_i]].elevation_m
            fs_recon = FlowState(
                abstract_state, abs(mdot_e),
                A=node_area[inlet_i], z=z_inlet_recon,
                T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                T_critical=T_c, P_critical=P_c,
            )
            # The reconstruction walks each converged component once for
            # per-block reporting.  Wrap each step so a still-choked or
            # numerically-infeasible converged mdot does not crash result
            # assembly (the solver may have terminated on gtol without
            # actually reaching the subsonic root -- see walk_edge above).
            # On failure the remaining downstream components are stamped
            # with the throat / failure state so callers still get a
            # full-length list indexed by component position.
            walked_PT = []
            recon_failed = False
            for c in walk_comps:
                if recon_failed:
                    walked_PT.append((float(abstract_state.p()),
                                      float(abstract_state.T())))
                    continue
                try:
                    c.dP_dT(fs_recon)
                except ChokedFlowError as e:
                    _safe_update_PT(abstract_state,
                                    e.P_outlet, e.T_outlet,
                                    T_cric, P_bar, T_c, P_c)
                    recon_failed = True
                except (RuntimeError, ValueError):
                    _safe_update_PT(abstract_state,
                                    P_arr[inlet_i], T_arr[inlet_i],
                                    T_cric, P_bar, T_c, P_c)
                    recon_failed = True
                walked_PT.append((float(abstract_state.p()),
                                  float(abstract_state.T())))
            # For reverse flow, walked_PT[k] is the flow-direction outlet of
            # original edge.components[N-1-k]; reverse it so the list is
            # indexed by original component position.
            if not walk_in_orig_order:
                walked_PT = list(reversed(walked_PT))
            component_outlet_PT[edge.name] = walked_PT

        return {
            "P_Pa":               {n: float(P_arr[i]) for i, n in enumerate(node_names)},
            "T_K":                {n: float(T_arr[i]) for i, n in enumerate(node_names)},
            "mdot_kgs":           {self._edges[e].name: float(mdot_arr[e])
                                   for e in range(n_edges)},
            "Q_ext_mdot_kgs":     Q_ext_out,
            "component_outlet_PT": component_outlet_PT,
            "converged":          bool(converged),
        }


# ---------------------------------------------------------------------------
# Results bundle for the compressible solver.
#
# Compressible_Network.solve() currently returns a flat dict rather than a
# NetworkResult (see network.md open work), so the analogue of
# NetworkResult.save_bundle lives here as a module-level function rather
# than as a method.  Per-pipe profiles are generated by re-running
# Line_Segment.dP_dT on the converged inlet (P, T) -- the same path the
# GUI's per-block inspector uses for "Plot profile..." plots.
# ---------------------------------------------------------------------------

def save_compressible_result_bundle(dir_path, net, result, abstract_state,
                                    *, isothermal=False):
    """Write summary.json + per-pipe profile CSVs into `dir_path`.

    Args:
        dir_path        : str.  Directory to write into; created if absent.
        net             : Compressible_Network instance the result came from.
        result          : flat dict returned by net.solve().
        abstract_state  : the same AbstractState passed to net.solve() (the
                          AS is mutated and restored around each profile
                          walk so the caller sees no net change).
        isothermal      : bool.  Mirror of the flag passed to solve(); used
                          to re-walk per-pipe profiles consistently.

    Mutates `abstract_state` transiently but restores its (P, T) on exit.
    """
    import CoolProp.CoolProp as CP
    from network import _safe_csv_name

    os.makedirs(dir_path, exist_ok=True)

    summary = {
        "regime":    "compressible",
        "converged": bool(result["converged"]),
        "nodes": {
            name: {
                "P_Pa":      float(result["P_Pa"][name]),
                "T_K":       float(result["T_K"][name]),
                "Q_ext_kgs": float(result["Q_ext_mdot_kgs"].get(name, 0.0)),
            }
            for name in result["P_Pa"]
        },
        "edges": {
            edge_name: {"mdot_kgs": float(mdot)}
            for edge_name, mdot in result["mdot_kgs"].items()
        },
    }

    profile_errors = {}
    used = set()

    P_save = abstract_state.p()
    T_save = abstract_state.T()
    try:
        for edge in net._edges:
            mdot = result["mdot_kgs"][edge.name]
            pt_list = result.get("component_outlet_PT", {}).get(edge.name, [])
            # Components are indexed in the edge's nominal order; the
            # flow-direction inlet of position 0 is from_node (if forward)
            # or to_node (if reverse).  Subsequent inlets are the previous
            # position's outlet.  In reverse flow the components are
            # walked in reverse position order.
            from_P = result["P_Pa"][edge.from_node]
            from_T = result["T_K"][edge.from_node]
            to_P   = result["P_Pa"][edge.to_node]
            to_T   = result["T_K"][edge.to_node]

            for pos, comp in enumerate(edge.components):
                if not hasattr(comp, "dP_dT") or not hasattr(comp, "profile"):
                    continue
                # Flow-direction inlet (P, T) for this position.
                if mdot >= 0.0:
                    if pos == 0:
                        P_in, T_in = from_P, from_T
                    else:
                        P_in, T_in = pt_list[pos - 1]
                else:
                    if pos == len(pt_list) - 1:
                        P_in, T_in = to_P, to_T
                    else:
                        P_in, T_in = pt_list[pos + 1]
                try:
                    abstract_state.update(CP.PT_INPUTS, P_in, T_in)
                    # FlowState carries area+envelope.  inlet_area_si makes
                    # the component's internal _area_match a no-op; z is
                    # set to 0 here since this re-walk is only used to
                    # generate a profile and z doesn't feed back into it.
                    fs_profile = FlowState(
                        abstract_state, abs(mdot),
                        A=comp.inlet_area_si, z=0.0,
                    )
                    raw = comp.dP_dT(fs_profile, isothermal=isothermal)
                except Exception as exc:
                    profile_errors[f"{edge.name}#{pos}"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not raw:
                    continue
                comp_name = getattr(comp, "name", None) or f"comp{pos}"
                fname = _safe_csv_name(
                    f"{edge.name}__{pos}__{comp_name}", used,
                ) + ".csv"
                used.add(fname.lower())
                with open(os.path.join(dir_path, fname), "w",
                          newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["distance_m", "P_Pa", "T_K", "v_ms"])
                    for d, P, T, v in raw:
                        w.writerow([d, P, T, v])
    finally:
        import CoolProp.CoolProp as _CP
        try:
            abstract_state.update(_CP.PT_INPUTS, P_save, T_save)
        except Exception:
            pass

    if profile_errors:
        summary["profile_errors"] = profile_errors
    with open(os.path.join(dir_path, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_single_segment_forward():
    """One inlet (P, T spec) -> Line_Segment -> one outlet (Q_ext = -mdot).

    The network should converge to the same outlet state as a direct
    dP_dT walk on the segment at the same mass flow.
    """
    from compressible_flow import Line_Segment, _build_phase_limits, _safe_update_PT
    import composition

    P_in  = ureg.Quantity(1000.0, "psi").to("Pa").magnitude
    T_in  = 300.0   # K
    mdot_kgs = 2.0  # kg/s of gas

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )

    seg = Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="seg",
    )

    # ---- Direct reference walk.
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    _safe_update_PT(AS, P_in, T_in, T_cric, P_bar, T_c, P_c)
    fs_ref = FlowState(
        AS, mdot_kgs, A=seg.inlet_area_si, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
    seg.dP_dT(fs_ref)
    P_out_ref = AS.p()
    T_out_ref = AS.T()
    print(f"[single-fwd]  direct walk:    "
          f"P_out = {P_out_ref/6894.757:.4f} psi,  T_out = {T_out_ref:.4f} K")

    # ---- Network solve.
    net = Compressible_Network()
    net.add_node("in",  P=P_in, T=T_in)
    net.add_node("out", Q_ext=-mdot_kgs)
    net.add_edge("seg", "in", "out", seg)

    # Reseed AS to fresh inlet state because the reference walk mutated it.
    _safe_update_PT(AS, P_in, T_in, T_cric, P_bar, T_c, P_c)
    result = net.solve(AS, mdot_init_kgs=mdot_kgs, verbose=True)

    P_out_net = result["P_Pa"]["out"]
    T_out_net = result["T_K"]["out"]
    mdot_net  = result["mdot_kgs"]["seg"]
    print(f"[single-fwd]  network solve:  "
          f"P_out = {P_out_net/6894.757:.4f} psi,  "
          f"T_out = {T_out_net:.4f} K,  "
          f"mdot = {mdot_net:.4f} kg/s")

    errP = abs(P_out_net - P_out_ref) / 6894.757
    errT = abs(T_out_net - T_out_ref)
    errM = abs(mdot_net - mdot_kgs)
    print(f"[single-fwd]  errors: P {errP:.3e} psi, T {errT:.3e} K, "
          f"mdot {errM:.3e} kg/s")
    assert result["converged"]
    assert errP < 1e-2
    assert errT < 1e-2
    assert errM < 1e-6


def _test_parallel_two_branches():
    """Two pipes in parallel between a single inlet and a single outlet.

    Compares total mass flow and per-branch flow fractions against
    parallel.parallel_compressible() on the same network and same total
    mdot.  The reference returns per-branch outlet (P, T) and flow fractions
    after iterating to equalize per-branch outlet pressure; the network
    solver should reproduce these.
    """
    from compressible_flow import Line_Segment
    from parallel import parallel_compressible
    import composition

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )

    P_in_psi = 1000.0
    T_in_K   = 300.0
    P_in_Pa  = ureg.Quantity(P_in_psi, "psi").to("Pa").magnitude
    mdot_total = 2.0   # kg/s

    rgh = ureg.Quantity(0.00015, "ft")
    # Simple uniform pipes, different diameters so the flow split is
    # nontrivial.  Short lengths keep dP_dT walks cheap.
    seg_A = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="A",
    )
    seg_B = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(4.026, "inch"),
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="B",
    )

    # ---- Reference solve via parallel_compressible.
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    _safe_update_PT(AS, P_in_Pa, T_in_K, T_cric, P_bar, T_c, P_c)
    P_out_ref, T_out_ref, fractions_ref = parallel_compressible(
        [seg_A, seg_B], AS, ureg.Quantity(mdot_total, "kg/s"),
    )
    # parallel_compressible returns one P_out / T_out per branch (they
    # should agree to TOL).  Take the mean for comparison.
    P_out_ref_mean = sum(P_out_ref) / len(P_out_ref)
    T_out_ref_mean = sum(T_out_ref) / len(T_out_ref)

    # ---- Network solve.
    net = Compressible_Network()
    net.add_node("in",  P=P_in_Pa, T=T_in_K)
    net.add_node("out", Q_ext=-mdot_total)
    net.add_edge("A", "in", "out", seg_A)
    net.add_edge("B", "in", "out", seg_B)

    _safe_update_PT(AS, P_in_Pa, T_in_K, T_cric, P_bar, T_c, P_c)
    result = net.solve(AS, mdot_init_kgs=mdot_total / 2, verbose=True)

    P_out_net = result["P_Pa"]["out"]
    T_out_net = result["T_K"]["out"]
    mdot_A    = result["mdot_kgs"]["A"]
    mdot_B    = result["mdot_kgs"]["B"]
    frac_net  = [mdot_A / mdot_total, mdot_B / mdot_total]

    print(f"[parallel]  reference: P_out = {P_out_ref_mean/6894.757:.4f} psi, "
          f"T_out = {T_out_ref_mean:.4f} K, fractions = {fractions_ref}")
    print(f"[parallel]  network:   P_out = {P_out_net/6894.757:.4f} psi, "
          f"T_out = {T_out_net:.4f} K, fractions = {frac_net}")

    errP = abs(P_out_net - P_out_ref_mean) / 6894.757
    errT = abs(T_out_net - T_out_ref_mean)
    err_frac = max(abs(a - b) for a, b in zip(frac_net, fractions_ref))
    print(f"[parallel]  errors: P {errP:.3e} psi, T {errT:.3e} K, "
          f"max frac {err_frac:.3e}")
    assert result["converged"]
    assert errP < 0.5      # parallel_compressible's tol is 1e-6 relative
    assert errT < 0.5
    # The network solver interprets the inlet node's (P, T) as the static
    # state at the node's single resolved area (here = seg_A's inlet,
    # 3.068" -- the first connected component on the first edge).  Branch
    # B's first dP_dT then runs an isentropic area change from 3.068" to
    # 4.026" at entry, which slightly shifts the split versus
    # parallel_compressible (which treats each branch as starting at its
    # own diameter with no inlet area change).  The discrepancy is small
    # but real; tolerance is set to comfortably accommodate it while
    # still catching gross regressions.
    assert err_frac < 5e-2


def _test_mixing_junction():
    """Two inlets at different temperatures merge at a mixer; check that
    the mixed outgoing temperature satisfies the enthalpy-mixing balance.

    Topology:
        in_hot  (P_spec, T_hot)  --pipe_H-->  MIX  --pipe_out-->  out (P_spec)
        in_cold (P_spec, T_cold) --pipe_C-->  MIX

    With both inlets at the same P_spec and the outlet at a lower P_spec,
    flow rates split such that both pipes hit MIX at the same pressure.
    The energy balance at MIX determines its T.  We check that the
    sum(mdot * h) over inflows equals sum(mdot * h) over the outflow at
    the converged state.
    """
    import os
    from compressible_flow import Line_Segment
    import composition

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

    P_in_Pa = ureg.Quantity(1000.0, "psi").to("Pa").magnitude
    P_out_Pa = ureg.Quantity(900.0, "psi").to("Pa").magnitude
    T_hot  = 350.0   # K
    T_cold = 280.0   # K

    rgh = ureg.Quantity(0.00015, "ft")
    def short_pipe(name, dia_inch=3.068, L_ft=200.0):
        return Line_Segment(
            roughness=rgh, id_val=ureg.Quantity(dia_inch, "inch"),
            length=ureg.Quantity(L_ft, "ft"),
            elevation_change=ureg.Quantity(0.0, "ft"),
            name=name,
        )
    pipe_H   = short_pipe("hot")
    pipe_C   = short_pipe("cold")
    pipe_out = short_pipe("out", dia_inch=4.026)

    net = Compressible_Network()
    net.add_node("in_hot",  P=P_in_Pa, T=T_hot)
    net.add_node("in_cold", P=P_in_Pa, T=T_cold)
    net.add_node("MIX")
    net.add_node("out",     P=P_out_Pa)
    net.add_edge("pipe_H",   "in_hot",  "MIX", pipe_H)
    net.add_edge("pipe_C",   "in_cold", "MIX", pipe_C)
    net.add_edge("pipe_out", "MIX",     "out", pipe_out)

    # Mass-balanced initial guess at MIX: pipe_H + pipe_C feed pipe_out, so
    # init them as 0.5 + 0.5 = 1.0.  A uniform scalar init would imbalance
    # MIX by 1 kg/s and can trap LM at the trivial all-zero minimum.
    result = net.solve(
        AS,
        mdot_init_kgs={"pipe_H": 0.5, "pipe_C": 0.5, "pipe_out": 1.0},
        verbose=True,
    )
    assert result["converged"]

    T_MIX = result["T_K"]["MIX"]
    P_MIX = result["P_Pa"]["MIX"]
    mdot_H   = result["mdot_kgs"]["pipe_H"]
    mdot_C   = result["mdot_kgs"]["pipe_C"]
    mdot_out = result["mdot_kgs"]["pipe_out"]

    print(f"[mixing]  P_MIX = {P_MIX/6894.757:.4f} psi, T_MIX = {T_MIX:.4f} K")
    print(f"[mixing]  mdot_H = {mdot_H:.4f} kg/s, mdot_C = {mdot_C:.4f} kg/s, "
          f"mdot_out = {mdot_out:.4f} kg/s")

    # Mass balance at MIX:
    mass_err = (mdot_H + mdot_C) - mdot_out
    print(f"[mixing]  mass balance at MIX: in - out = {mass_err:.3e} kg/s")
    assert abs(mass_err) < 1e-6

    # Energy balance at MIX (verify the solver actually satisfied it):
    # Need the walked-outlet enthalpies of the hot and cold inflow edges.
    # Walk each from its inlet (P_in, T_in) at the solved mdot and read h.
    def walked_h_at_MIX(seg, T_inlet, mdot):
        _safe_update_PT(AS, P_in_Pa, T_inlet, T_cric, P_bar, T_c, P_c)
        fs_check = FlowState(
            AS, mdot, A=seg.inlet_area_si, z=0.0,
            T_cricondentherm=T_cric, P_cricondenbar=P_bar,
            T_critical=T_c, P_critical=P_c,
        )
        seg.dP_dT(fs_check)
        return AS.hmass()

    h_H_arriving = walked_h_at_MIX(pipe_H, T_hot,  mdot_H)
    h_C_arriving = walked_h_at_MIX(pipe_C, T_cold, mdot_C)
    _safe_update_PT(AS, P_MIX, T_MIX, T_cric, P_bar, T_c, P_c)
    h_MIX = AS.hmass()

    energy_err = mdot_H * h_H_arriving + mdot_C * h_C_arriving - mdot_out * h_MIX
    rel_err = abs(energy_err) / (mdot_out * abs(h_MIX) + 1.0)
    print(f"[mixing]  energy balance at MIX: net = {energy_err:.3e} W  "
          f"(relative {rel_err:.3e})")
    assert rel_err < 1e-4

    # Sanity: T_MIX should sit between T_cold and T_hot.
    assert T_cold < T_MIX < T_hot, f"T_MIX = {T_MIX} outside ({T_cold}, {T_hot})"


def _test_orifice_subsonic():
    """One inlet (P, T spec) -> Orifice -> one outlet (Q_ext = -mdot).

    Compares the network-solved outlet pressure to a direct Orifice.dP_dT
    walk at the same mdot.  Tests that an orifice round-trips through the
    network solver the same way a Valve / Bend does.
    """
    from compressible_flow import (
        Orifice, _build_phase_limits, _safe_update_PT,
    )
    import composition

    P_in     = ureg.Quantity(1000.0, "psi").to("Pa").magnitude
    T_in     = 300.0   # K
    mdot_kgs = 1.0     # kg/s -- comfortably subsonic for the geometry below

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )

    Di = ureg.Quantity(3.068, "inch").to("m").magnitude
    Do = ureg.Quantity(1.500, "inch").to("m").magnitude
    orf = Orifice(Di=Di, Do=Do, taps="corner", Cd_override=0.62, name="ORF")

    # ---- Direct reference walk.
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)
    _safe_update_PT(AS, P_in, T_in, T_cric, P_bar, T_c, P_c)
    fs_ref = FlowState(
        AS, mdot_kgs, A=orf.inlet_area_si, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
    orf.dP_dT(fs_ref)
    P_out_ref = AS.p()
    T_out_ref = AS.T()
    print(f"[orifice-sub] direct walk:    "
          f"P_out = {P_out_ref/6894.757:.4f} psi,  T_out = {T_out_ref:.4f} K")

    # ---- Network solve.
    net = Compressible_Network()
    net.add_node("in",  P=P_in, T=T_in, diameter=Di)
    net.add_node("out", Q_ext=-mdot_kgs)
    net.add_edge("orf", "in", "out", orf)

    _safe_update_PT(AS, P_in, T_in, T_cric, P_bar, T_c, P_c)
    result = net.solve(AS, mdot_init_kgs=mdot_kgs, verbose=True)

    P_out_net = result["P_Pa"]["out"]
    T_out_net = result["T_K"]["out"]
    mdot_net  = result["mdot_kgs"]["orf"]
    print(f"[orifice-sub] network solve:  "
          f"P_out = {P_out_net/6894.757:.4f} psi,  "
          f"T_out = {T_out_net:.4f} K,  "
          f"mdot = {mdot_net:.4f} kg/s")

    errP = abs(P_out_net - P_out_ref) / 6894.757
    errT = abs(T_out_net - T_out_ref)
    errM = abs(mdot_net - mdot_kgs)
    print(f"[orifice-sub] errors: P {errP:.3e} psi, T {errT:.3e} K, "
          f"mdot {errM:.3e} kg/s")
    assert result["converged"]
    assert errP < 1e-2
    assert errT < 1e-2
    assert errM < 1e-6


def _test_orifice_choke():
    """Orifice.dP_dT raises ChokedFlowError when fs.mdot exceeds the
    textbook ISO 5167 choked mass flow.

    The orifice's choke check is the standard critical-pressure-ratio
    test: P2/P1 < (2/(k+1))**(k/(k-1)).  Mass flow at that ratio is
    obtained from the ISO 5167 subsonic equation (with the expansibility
    factor) and is typically a few tens of percent above the pure ideal-
    gas G_max * A_bore estimate -- the expansibility correction Y and
    Cd interact non-trivially at the critical ratio.

    Drives a small orifice with methane at 5 MPa and a deliberately
    oversized mdot; checks that the raised ChokedFlowError carries a
    sensible throat pressure (close to the critical ratio of the gas).
    """
    from compressible_flow import (
        Orifice, ChokedFlowError, _build_phase_limits, _safe_update_PT,
    )
    import composition

    AS = composition.define_composition(y_Methane=1.0, eos="HEOS")
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

    P_in, T_in = 5.0e6, 300.0
    _safe_update_PT(AS, P_in, T_in, T_cric, P_bar, T_c, P_c)

    Di, Do = 0.1, 0.05
    A_pipe = math.pi * Di ** 2 / 4.0
    Cd     = 0.62

    orf = Orifice(Di=Di, Do=Do, Cd_override=Cd, name="ORF-choke")
    # Pick mdot deliberately oversized so the trip is unambiguous (well
    # above any sensible subsonic root for this geometry).
    fs = FlowState(
        AS, mdot=50.0, A=A_pipe, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )

    # Reference: ideal-gas critical pressure ratio for methane (k ~ 1.31).
    # The real-gas isentropic expansion finds the sonic point on the real-gas
    # isentrope, which for supercritical methane sits noticeably above
    # the ideal-gas ratio (Z != 1, the gas is denser and slower-speed-
    # of-sound than ideal).  Allow up to 25% departure from the ideal-
    # gas reference.
    k_ig          = AS.cpmass() / AS.cvmass()
    crit_ratio    = (2.0 / (k_ig + 1.0)) ** (k_ig / (k_ig - 1.0))
    P_throat_ref  = crit_ratio * P_in

    try:
        orf.dP_dT(fs)
    except ChokedFlowError as e:
        print(f"[orifice-choke] ChokedFlowError raised: "
              f"mdot_choked = {e.mdot_choked:.4f} kg/s")
        print(f"[orifice-choke] throat: P = {e.P_throat/1e6:.4f} MPa "
              f"(ideal-gas ref = {P_throat_ref/1e6:.4f} MPa), "
              f"T = {e.T_throat:.4f} K")
        # Throat pressure within an order-of-magnitude band centred on
        # the ideal-gas critical ratio.
        assert abs(e.P_throat - P_throat_ref) / P_throat_ref < 0.25
        # mdot_choked must be positive and not absurdly large.
        assert 1.0 < e.mdot_choked < 100.0
    else:
        raise AssertionError(
            "[orifice-choke] Expected ChokedFlowError, got subsonic solution."
        )


def _test_inverse_single_relief_valve():
    """Single-edge network with a Valve between two P-spec nodes
    (vessel -> atmosphere).  Auto-detection marks the edge inverse-mode
    because to_node is P-spec; walk_edge_inverse drives mdot locally.
    Recovers the same mdot as a direct forward Valve.dP_dT call.
    """
    from compressible_flow import Valve
    import composition

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )
    T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

    P_vessel_Pa = ureg.Quantity(150.0 + 14.7, "psi").to("Pa").magnitude
    P_atm_Pa    = ureg.Quantity(14.7, "psi").to("Pa").magnitude
    T_vessel    = 300.0

    relief = Valve(Di=ureg.Quantity(1.0, "inch"), K=20.0)

    # Forward ground truth: pick any mdot, walk Valve.dP_dT, measure
    # the outlet pressure, then ask the inverse to recover that mdot
    # from the same boundary conditions.
    _safe_update_PT(AS, P_vessel_Pa, T_vessel, T_cric, P_bar, T_c, P_c)
    A_pipe = math.pi * relief.Di_si ** 2 / 4.0
    fs_ref = FlowState(
        AS, mdot=0.3, A=A_pipe, z=0.0,
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
    relief.dP_dT(fs_ref)
    P_target = fs_ref.P
    mdot_ref = 0.3

    net = Compressible_Network()
    net.add_node("vessel", P=P_vessel_Pa, T=T_vessel)
    net.add_node("atm",    P=P_target)
    net.add_edge("relief", "vessel", "atm", relief)

    result = net.solve(AS, mdot_init_kgs=0.1, verbose=True)
    assert result["converged"]

    mdot_solved = result["mdot_kgs"]["relief"]
    rel_err = abs(mdot_solved - mdot_ref) / mdot_ref
    print(f"[inv-single]  ground-truth mdot = {mdot_ref:.6f} kg/s, "
          f"P_atm spec = {P_target/6894.757:.4f} psi")
    print(f"[inv-single]  network solve     = {mdot_solved:.6f} kg/s, "
          f"rel err = {rel_err:.2e}")
    assert rel_err < 1e-5


def _test_inverse_relief_from_junction():
    """Three-node network: an inlet feeds a junction which splits to a
    main outlet (P-spec) and a relief vent (P-spec, lower).  Both
    outgoing edges are auto-detected as inverse-mode; the inlet edge
    stays forward.  Sanity: mass balance closes at the junction.
    """
    from compressible_flow import Line_Segment, Valve
    import composition

    AS = composition.define_composition(
        y_Methane=0.9, y_Ethane=0.05, y_Propane=0.02,
        y_n_Butane=0.01, y_CarbonDioxide=0.02, eos="PR",
    )

    P_in_Pa     = ureg.Quantity(200.0 + 14.7, "psi").to("Pa").magnitude
    P_main_Pa   = ureg.Quantity(100.0 + 14.7, "psi").to("Pa").magnitude
    P_relief_Pa = ureg.Quantity(14.7,         "psi").to("Pa").magnitude

    feed = Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(50.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="feed",
    )
    main = Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="main",
    )
    relief = Valve(Di=ureg.Quantity(1.0, "inch"), K=20.0)

    net = Compressible_Network()
    net.add_node("inlet",  P=P_in_Pa, T=320.0)
    net.add_node("JCT")
    net.add_node("main_outlet",   P=P_main_Pa)
    net.add_node("relief_outlet", P=P_relief_Pa)
    net.add_edge("feed",   "inlet", "JCT",            feed)
    net.add_edge("main",   "JCT",   "main_outlet",    main)
    net.add_edge("relief", "JCT",   "relief_outlet",  relief)

    result = net.solve(
        AS,
        mdot_init_kgs={"feed": 1.0, "main": 0.7, "relief": 0.3},
        verbose=True,
    )
    assert result["converged"]

    mdot_feed   = result["mdot_kgs"]["feed"]
    mdot_main   = result["mdot_kgs"]["main"]
    mdot_relief = result["mdot_kgs"]["relief"]
    P_JCT       = result["P_Pa"]["JCT"]
    T_JCT       = result["T_K"]["JCT"]

    mass_err = mdot_feed - mdot_main - mdot_relief
    print(f"[inv-junction] P_JCT = {P_JCT/6894.757:.4f} psi, T_JCT = {T_JCT:.4f} K")
    print(f"[inv-junction] mdot_feed   = {mdot_feed:.6f} kg/s")
    print(f"[inv-junction] mdot_main   = {mdot_main:.6f} kg/s")
    print(f"[inv-junction] mdot_relief = {mdot_relief:.6f} kg/s")
    print(f"[inv-junction] mass balance at JCT: {mass_err:.3e} kg/s")
    assert abs(mass_err) < 1e-6
    assert P_main_Pa < P_JCT < P_in_Pa
    # Relief should choke (P_relief = atm well below choke pressure),
    # so mdot_relief takes whatever the local mdot_choked at JCT gives.
    # Just check it is positive and reasonable.
    assert 0.0 < mdot_relief < 5.0


if __name__ == "__main__":
    _test_single_segment_forward()
    print()
    _test_parallel_two_branches()
    print()
    _test_mixing_junction()
    print()
    _test_orifice_subsonic()
    print()
    _test_orifice_choke()
    print()
    _test_inverse_single_relief_valve()
    print()
    _test_inverse_relief_from_junction()
