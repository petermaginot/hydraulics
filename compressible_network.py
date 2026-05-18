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

import warnings
import numpy as np
from scipy.optimize import root, least_squares

from component_classes import ureg
from network import (
    Node, Edge, Network,
    _reversed_component, _to_si_or_none, _to_pint_qty,
)
from compressible_flow import _build_phase_limits, _safe_update_PT


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

    def add_node(self, name, *, elevation=0.0, P=None, T=None, Q_ext=None,
                 abstract_state=None):
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
            abstract_state : Kept for API stability; no longer needed at
                             add_node time because conversion is deferred.

        Specifying both P and Q_ext is rejected (over-constrains the node).
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

        self._nodes[name] = Node(
            name=name, elevation_m=elev_si,
            P_spec_Pa=P_si, Q_ext_spec_qty=Q_qty, T_spec_K=T_si,
        )
        self._node_order.append(name)

    def solve(self, abstract_state, *, P_init=None, T_init=None,
              mdot_init_kgs=1.0, verbose=False, xtol=1e-8, maxfev=5000):
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

        Returns:
            dict with keys "P_Pa", "T_K", "mdot_kgs", "Q_ext_mdot_kgs",
            and "converged".  The interpretation matches the incompressible
            NetworkResult but with mass flow and temperature included.
        """
        node_names = list(self._node_order)
        n_nodes    = len(node_names)
        n_edges    = len(self._edges)

        if n_nodes == 0 or n_edges == 0:
            raise ValueError(
                "Compressible_Network.solve: network has no nodes or edges."
            )

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

            Returns (P_outlet, T_outlet, h_outlet) at the flow outlet.
            Also returns the flow-inlet and flow-outlet node indices so
            the caller can match the pressure residual against the right
            node.
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
            abs_mdot = abs(mdot_e)

            _safe_update_PT(abstract_state, P_in_e, T_in_e,
                            T_cric, P_bar, T_c, P_c)
            for c in comps:
                c.dP_dT(
                    abstract_state,
                    ureg.Quantity(abs_mdot, "kg/s"),
                    T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                    T_critical=T_c, P_critical=P_c,
                )

            return (abstract_state.p(),
                    abstract_state.T(),
                    abstract_state.hmass(),
                    inlet_i, outlet_i)

        # Reference scales used to non-dimensionalize each residual class.
        # All three are then O(1) at the initial guess and any further
        # imbalance is relative, so the Jacobian is not dominated by one
        # equation type.
        P_ref      = max(abs(P_init), 1.0)
        mdot_ref   = mdot_ref_scalar
        energy_ref = mdot_ref * 1.0e6   # J/s, rough enthalpy scale

        def residuals(x):
            P, T, mdot = unpack(x)
            res = np.empty(n_p_free + n_t_free + n_edges)

            walked_h_out = np.empty(n_edges)
            walked_P_out = np.empty(n_edges)
            outlet_i_of  = [None] * n_edges

            for e_idx, edge in enumerate(self._edges):
                wP, wT, wh, _, outlet_i = walk_edge(edge, mdot[e_idx], P, T)
                walked_P_out[e_idx] = wP
                walked_h_out[e_idx] = wh
                outlet_i_of[e_idx]  = outlet_i

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
                    # External supply at T_spec (validated above to exist).
                    _safe_update_PT(abstract_state, P_n, node.T_spec_K,
                                    T_cric, P_bar, T_c, P_c)
                    h_supply = abstract_state.hmass()
                    sum_in += Q_ext_eff * h_supply
                elif Q_ext_eff < 0.0:
                    sum_out += abs(Q_ext_eff) * h_node

                res[n_p_free + j] = (sum_in - sum_out) / energy_ref

            # Pipe equation per edge: walked outlet P == node P at the
            # flow-outlet end (normalized by P_ref).
            for e_idx in range(n_edges):
                res[n_p_free + n_t_free + e_idx] = (
                    walked_P_out[e_idx] - P[outlet_i_of[e_idx]]
                ) / P_ref

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

        sol_obj = least_squares(
            residuals, x0, bounds=(lb, ub),
            method="trf", x_scale="jac",
            xtol=xtol, ftol=xtol, gtol=xtol, max_nfev=maxfev,
        )
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

        return {
            "P_Pa":           {n: float(P_arr[i]) for i, n in enumerate(node_names)},
            "T_K":            {n: float(T_arr[i]) for i, n in enumerate(node_names)},
            "mdot_kgs":       {self._edges[e].name: float(mdot_arr[e])
                               for e in range(n_edges)},
            "Q_ext_mdot_kgs": Q_ext_out,
            "converged":      bool(converged),
        }


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
    seg.dP_dT(
        AS, ureg.Quantity(mdot_kgs, "kg/s"),
        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
        T_critical=T_c, P_critical=P_c,
    )
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
    assert err_frac < 1e-3


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
        seg.dP_dT(
            AS, ureg.Quantity(mdot, "kg/s"),
            T_cricondentherm=T_cric, P_cricondenbar=P_bar,
            T_critical=T_c, P_critical=P_c,
        )
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


if __name__ == "__main__":
    _test_single_segment_forward()
    print()
    _test_parallel_two_branches()
    print()
    _test_mixing_junction()
