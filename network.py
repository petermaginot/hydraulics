"""network.py

General pipe-network evaluator for incompressible flow.

Handles an arbitrary directed graph of pipe components: multiple inlets,
multiple outlets, series, parallel, and edges whose flow direction is not
known in advance (the sign of the edge flow is part of the solution).

Three classes:
    Node     -- a junction, inlet, or outlet.  Holds elevation and at most
                one boundary condition (specified pressure or external flow).
    Edge     -- a directed path from one node to another.  Carries a list of
                pipe components evaluated in series in the from->to direction.
    Network  -- a collection of nodes and edges with a solve() method.

Sign conventions
----------------
For each Edge, a positive Q_e means flow in the nominal from->to direction;
a negative Q_e means flow runs to->from.  External flows (Q_ext) at a Node
are positive when supplied INTO the node from outside the network (an inlet)
and negative for a withdrawal (an outlet).

Solver formulation
------------------
Unknowns:
    -- pressure P_n at every node whose pressure is not specified;
    -- signed flow Q_e on every edge.

Equations:
    -- Pipe equation per edge:
           P_from - P_to = -dP_inlet_to_outlet(Q_e)
       where dP_inlet_to_outlet is the signed pressure change across the
       edge's components (negative for a friction loss in the forward
       direction).  See _component_signed_dP() for the sign treatment.

    -- Mass balance at each non-P-spec node:
           sum of (signed) edge flows entering the node + Q_ext_spec = 0.
       Q_ext_spec defaults to 0 for interior junctions.

The system is solved with scipy.optimize.fsolve (Powell hybrid).  After
convergence, the external flow at each P-spec node is recovered from its
mass balance.

Reverse-flow handling
---------------------
When an edge's solved flow is negative, each of its components is evaluated
against a reversed shadow copy of itself at flow magnitude |Q|, and the
returned dP is then negated to express it in the forward inlet-to-outlet
convention.  Reversing means:
    -- Line_Segment: the profile point list is rebuilt with distances
       flipped (new_dist = total_L - old_dist) and the elevation,
       hydraulic-diameter, and area entries reversed in order.  All
       segment properties (total_length_m, net_elevation_change_m,
       volume_m3) follow automatically from the new profile.
    -- Contraction_Expansion: Di_US and Di_DS are swapped, turning a
       contraction into an expansion and vice versa.
    -- Bend and Valve: symmetric, returned unchanged.
This reverses the actual geometry rather than applying a sign(Q) flip on
the friction term, so K-factor asymmetries (sharp contraction vs sharp
expansion) and along-pipe profile asymmetries are captured correctly.

Limitations
-----------
-- Incompressible flow only.  Constant density is assumed network-wide.
-- The Jacobian is computed by finite differences inside fsolve.  This is
   fine for networks up to a few dozen edges; larger networks would
   benefit from a sparse analytic Jacobian.
-- User-defined component types not derived from Base_Line_Segment or
   Base_Contraction_Expansion are assumed symmetric under flow reversal.
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import fsolve

from component_classes import ureg


# Standard gravity, used for hydrostatic elevation contributions.
_G = 9.8066   # m/s^2

# Below this absolute flow rate (m^3/s), friction is linearized through zero
# to avoid the friction_factor(Re=0) singularity and to keep the residual
# function smooth for Newton's method.
_Q_LIN_EPS = 1e-9


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_si_or_none(val, unit):
    """Convert a pint Quantity or plain float to SI float magnitude.

    None passes through unchanged.  A plain float is taken as already SI.
    """
    if val is None:
        return None
    if hasattr(val, "to"):
        return val.to(unit).magnitude
    return float(val)


def _to_pint_qty(val, default_unit):
    """Wrap a plain number as a pint Quantity with `default_unit`; pass
    through if already a Quantity; return None for None.

    Used by add_node() / _set_node_boundary() to defer flow-rate unit
    conversion until solve() time, when the fluid (and hence density) is
    available.
    """
    if val is None:
        return None
    if hasattr(val, "dimensionality"):
        return val
    return ureg.Quantity(float(val), default_unit)


def _qty_to_mdot_kgs_incompressible(qty, density_kg_per_m3):
    """Convert a flow-rate pint Quantity to mass flow [kg/s] for an
    incompressible fluid of constant density.

    Accepts mass ([mass]/[time]) or volumetric ([length]^3/[time]) Quantities.
    Compressible networks use a different helper that handles molar /
    standard-volume flow via molar mass.
    """
    if qty is None:
        return None
    dim = qty.dimensionality
    if dim == {"[mass]": 1, "[time]": -1}:
        return qty.to("kg/s").magnitude
    if dim == {"[length]": 3, "[time]": -1}:
        return qty.to("m^3/s").magnitude * float(density_kg_per_m3)
    raise ValueError(
        f"Incompressible Q_ext has unsupported dimensions {dict(dim)}.  "
        f"Expected [mass]/[time] or [length]^3/[time]."
    )


def _reversed_component(component):
    """Return a shadow copy of `component` representing the reverse flow
    direction.

    Dispatch is by duck typing so this module does not depend on importing
    every component class:
        -- a `.profile` attribute => Line_Segment-like.  The profile list is
           rebuilt with distance flipped and other per-point entries (elev,
           hydraulic diameter, flow area) reversed in order.
        -- `.Di_US_si` and `.Di_DS_si` attributes => Contraction_Expansion-like.
           The two diameters are swapped, so a contraction becomes an
           expansion and vice versa.
        -- otherwise the component is taken to be symmetric (Bend, Valve)
           and returned unchanged.

    Uses copy.copy (shallow) so the reversed shadow shares everything other
    than the rebuilt geometry with the original.
    """
    if hasattr(component, "profile") and component.profile:
        rev = copy.copy(component)
        total_L = component.profile[-1][0]
        rev.profile = [
            (total_L - dist, elev, dh, area)
            for (dist, elev, dh, area) in reversed(component.profile)
        ]
        return rev

    if hasattr(component, "Di_US_si") and hasattr(component, "Di_DS_si"):
        rev = copy.copy(component)
        rev.Di_US_si, rev.Di_DS_si = component.Di_DS_si, component.Di_US_si
        return rev

    return component   # symmetric -- Bend, Valve, anything else


def _component_signed_dP(component, fluid, Q_si, reversed_cache):
    """Signed inlet-to-outlet pressure change of a single component [Pa].

    For Q >= 0 the component is evaluated in its forward orientation.
    For Q <  0 the reversed shadow is evaluated at |Q|; the returned dP is
    the reversed component's inlet-to-outlet change, which is exactly the
    negative of the original component's inlet-to-outlet change at the same
    physical state.  Negating restores the answer to the forward
    inlet-to-outlet convention.

    For |Q| <= _Q_LIN_EPS, friction is linearized through zero to avoid the
    friction_factor(Re=0) singularity and to keep the residual smooth.  The
    linearization uses a forward-direction probe; this is exact in the
    sub-eps region because elevation is direction-independent and the tiny
    friction term swaps sign continuously through zero.

    Args:
        component       : a component instance (.dP method).
        fluid           : Incompressible_Fluid instance.
        Q_si            : signed volumetric flow [m^3/s].
        reversed_cache  : dict; the entry for this component is consulted
                          (and lazily filled) to avoid rebuilding the
                          reversed shadow every Newton step.

    Returns:
        float, signed inlet-to-outlet dP [Pa].
    """
    aQ = abs(Q_si)

    if aQ <= _Q_LIN_EPS:
        # Elevation contribution is symmetric; friction goes to zero.
        if hasattr(component, "net_elevation_change_m"):
            elev_part = -fluid.density_si * _G * component.net_elevation_change_m
        else:
            elev_part = 0.0
        if aQ == 0.0:
            return elev_part
        Q_probe = _Q_LIN_EPS
        c_dP_probe = component.dP(
            fluid, flow_rate=ureg.Quantity(Q_probe, "m^3/s")
        )
        friction_probe = c_dP_probe - elev_part
        friction_signed = friction_probe * (Q_si / Q_probe)
        return friction_signed + elev_part

    if Q_si > 0.0:
        return component.dP(fluid, flow_rate=ureg.Quantity(aQ, "m^3/s"))

    # Reverse flow: evaluate the reversed shadow at |Q| and negate.
    key = id(component)
    rev = reversed_cache.get(key)
    if rev is None:
        rev = _reversed_component(component)
        reversed_cache[key] = rev
    return -rev.dP(fluid, flow_rate=ureg.Quantity(aQ, "m^3/s"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A network junction or boundary.

    Attributes:
        name            : str, unique label.
        elevation_m     : float, elevation above datum [m].
        P_spec_Pa       : float or None, specified absolute pressure [Pa].
        Q_ext_spec_qty  : pint Quantity or None, specified external supply.
                          Stored as the raw Quantity so unit conversion can
                          be deferred until solve() time, when the fluid (and
                          hence density / molar mass) is available.  Positive
                          = flow into the node from outside.  Both solvers
                          convert to mass flow [kg/s] internally.  Bare
                          floats passed to add_node() are wrapped as kg/s.
        T_spec_K        : float or None, specified temperature [K].  Used
                          by compressible networks only; ignored by the
                          incompressible solver.  Inlets typically have it;
                          outlets typically don't (T emerges from the solve).
    """
    name: str
    elevation_m: float = 0.0
    P_spec_Pa: Optional[float] = None
    Q_ext_spec_qty: Optional[Any] = None
    T_spec_K: Optional[float] = None


@dataclass
class Edge:
    """A directed connection between two nodes carrying components in series.

    Attributes:
        name       : str, unique label.
        from_node  : str, name of the upstream node in the nominal direction.
        to_node    : str, name of the downstream node in the nominal direction.
        components : list of hydraulics components (Line_Segment, Bend,
                     Valve, Contraction_Expansion) traversed in order from
                     from_node to to_node.  May be empty (a zero-dP
                     connector that pins P_from == P_to).
    """
    name: str
    from_node: str
    to_node: str
    components: list


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """A directed pipe network for incompressible-flow analysis.

    Usage:
        net = Network()
        net.add_node("inlet",  P=ureg.Quantity(200, "psi"))
        net.add_node("outlet", P=ureg.Quantity(50,  "psi"))
        net.add_edge("PIPE-1", "inlet", "outlet", [some_line_segment])
        result = net.solve(fluid)

    See module docstring for sign conventions, boundary-condition rules,
    and solver formulation.
    """

    def __init__(self):
        self._node_order = []   # insertion-order list of node names
        self._nodes = {}        # name -> Node
        self._edges = []        # list of Edge in insertion order

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, name, *, elevation=0.0, P=None, Q_ext=None,
                 fluid_for_units=None):
        """Add a node.

        Args:
            name             : str, unique label.
            elevation        : pint Quantity or float (m if float).
                               Default 0.
            P                : pint Quantity or float (Pa if float).
                               Specified absolute pressure.  Optional.
            Q_ext            : pint Quantity (volumetric or mass) or float
                               (m^3/s if float).  Specified external supply
                               into the node (positive in, negative out).
                               Optional.  Bare floats are interpreted as
                               kg/s; pint Quantities can be mass or
                               volumetric (volumetric is converted to mass
                               at solve() time using the fluid density).
            fluid_for_units  : Kept for API stability; no longer needed
                               because conversion is deferred to solve().

        Specifying both P and Q_ext is an error.  Specifying neither makes
        the node interior with Q_ext = 0.
        """
        if name in self._nodes:
            raise ValueError(f"Network.add_node: node {name!r} already exists.")

        elev_si = _to_si_or_none(elevation, "m")
        if elev_si is None:
            elev_si = 0.0

        P_si  = _to_si_or_none(P, "Pa")
        Q_qty = _to_pint_qty(Q_ext, "kg/s")

        if P_si is not None and Q_qty is not None:
            raise ValueError(
                f"Network.add_node: node {name!r} cannot specify both P and "
                f"Q_ext."
            )

        self._nodes[name] = Node(name, elev_si, P_si, Q_qty)
        self._node_order.append(name)

    def add_edge(self, name, from_node, to_node, components):
        """Add a directed edge between two existing nodes.

        Args:
            name       : str, unique label.
            from_node  : str, must already exist as a node.
            to_node    : str, must already exist; cannot equal from_node.
            components : a single component or a list of components run in
                         series in the from->to direction.  An empty list
                         is allowed and represents a zero-dP connector
                         (pins P_from == P_to).
        """
        if from_node not in self._nodes:
            raise ValueError(
                f"Network.add_edge: unknown from_node {from_node!r} on edge "
                f"{name!r}."
            )
        if to_node not in self._nodes:
            raise ValueError(
                f"Network.add_edge: unknown to_node {to_node!r} on edge "
                f"{name!r}."
            )
        if from_node == to_node:
            raise ValueError(
                f"Network.add_edge: edge {name!r} would form a self-loop."
            )
        if not isinstance(components, list):
            components = [components]
        self._edges.append(Edge(name, from_node, to_node, list(components)))

    # ------------------------------------------------------------------
    # Boundary-condition helpers
    # ------------------------------------------------------------------

    def _set_node_boundary(self, name, *, P=None, Q_ext=None,
                           fluid_for_units=None):
        """Update a node's boundary spec in place.

        Setting one of P / Q_ext clears the other.  Pass both as None to
        make the node interior (Q_ext = 0).

        Used by the calculation-mode wrappers (solve_for_outlet_pressure,
        solve_for_inlet_pressure, solve_for_flow_rate) to repurpose existing
        nodes without forcing the user to rebuild the network.
        """
        if name not in self._nodes:
            raise KeyError(f"Network: no node named {name!r}.")
        if P is not None and Q_ext is not None:
            raise ValueError(
                f"Network._set_node_boundary: node {name!r} cannot have both "
                f"P and Q_ext specified."
            )
        node = self._nodes[name]
        if P is not None:
            node.P_spec_Pa      = _to_si_or_none(P, "Pa")
            node.Q_ext_spec_qty = None
        elif Q_ext is not None:
            node.Q_ext_spec_qty = _to_pint_qty(Q_ext, "kg/s")
            node.P_spec_Pa      = None
        else:
            node.P_spec_Pa      = None
            node.Q_ext_spec_qty = None

    # ------------------------------------------------------------------
    # Calculation-mode wrappers
    # ------------------------------------------------------------------
    #
    # All three operate on a pair of boundary nodes (inlet / outlet) and
    # delegate to solve().  They mutate the spec on the two named nodes and
    # leave any other boundary conditions in the network untouched, so the
    # same Network object can be reused across modes.

    def solve_for_outlet_pressure(self, fluid, *, inlet, outlet, P_inlet, Q,
                                  **solve_kwargs):
        """Mode 1: given inlet pressure and total flow, solve for outlet pressure.

        Sets inlet node to P_inlet, outlet node to Q_ext = -Q (a withdrawal
        of magnitude |Q| from the network), then runs solve().

        Args:
            fluid    : Incompressible_Fluid instance.
            inlet    : str, name of the node where pressure and flow enter.
            outlet   : str, name of the node where flow leaves.
            P_inlet  : pint Quantity or float (Pa if float).  Inlet pressure.
            Q        : pint Quantity (volumetric or mass) or float
                       (m^3/s if float).  Total flow rate.  Must be positive.
            solve_kwargs : forwarded to Network.solve() (P_init, verbose, etc.).

        Returns:
            NetworkResult.  The answer of interest is result["P_Pa"][outlet].
        """
        self._set_node_boundary(inlet,  P=P_inlet)
        self._set_node_boundary(outlet, Q_ext=-Q, fluid_for_units=fluid)
        return self.solve(fluid, **solve_kwargs)

    def solve_for_inlet_pressure(self, fluid, *, inlet, outlet, P_outlet, Q,
                                 **solve_kwargs):
        """Mode 2: given outlet pressure and total flow, solve for inlet pressure.

        Sets inlet node to Q_ext = +Q (supply into network), outlet node to
        P_outlet, then runs solve().

        Args:
            fluid    : Incompressible_Fluid instance.
            inlet    : str, name of the node where flow enters.
            outlet   : str, name of the node where pressure and flow leave.
            P_outlet : pint Quantity or float (Pa if float).  Outlet pressure.
            Q        : pint Quantity (volumetric or mass) or float
                       (m^3/s if float).  Total flow rate.  Must be positive.
            solve_kwargs : forwarded to Network.solve().

        Returns:
            NetworkResult.  The answer of interest is result["P_Pa"][inlet].
        """
        self._set_node_boundary(inlet,  Q_ext=Q, fluid_for_units=fluid)
        self._set_node_boundary(outlet, P=P_outlet)
        return self.solve(fluid, **solve_kwargs)

    def solve_for_flow_rate(self, fluid, *, inlet, outlet, P_inlet, P_outlet,
                            **solve_kwargs):
        """Mode 3: given inlet and outlet pressures, solve for total flow rate.

        Sets both endpoints to specified pressures; solve() recovers Q_ext at
        each from mass balance.

        Args:
            fluid     : Incompressible_Fluid instance.
            inlet     : str, name of the node where flow enters.
            outlet    : str, name of the node where flow leaves.
            P_inlet   : pint Quantity or float (Pa if float).  Inlet pressure.
            P_outlet  : pint Quantity or float (Pa if float).  Outlet pressure.
            solve_kwargs : forwarded to Network.solve().

        Returns:
            NetworkResult.  The answer of interest is
            result["Q_ext_m3s"][inlet] (positive = flow into network at inlet).
        """
        self._set_node_boundary(inlet,  P=P_inlet)
        self._set_node_boundary(outlet, P=P_outlet)
        return self.solve(fluid, **solve_kwargs)

    # ------------------------------------------------------------------
    # Solve helpers
    # ------------------------------------------------------------------

    def _mdot_ext_spec_kgs(self, rho):
        """External mass flow [kg/s] per node, with unspecified nodes
        defaulting to 0.0 so downstream lookups can be unconditional."""
        out = {}
        for name, node in self._nodes.items():
            mdot = _qty_to_mdot_kgs_incompressible(node.Q_ext_spec_qty, rho)
            out[name] = 0.0 if mdot is None else mdot
        return out

    def _partition_node_indices(self, node_names):
        """Split node indices into (free-pressure, P-spec) lists.

        Raises ValueError if no P-spec node exists -- without an anchor the
        pressure field is undetermined.
        """
        free_indices = [
            i for i, name in enumerate(node_names)
            if self._nodes[name].P_spec_Pa is None
        ]
        p_spec_indices = [
            i for i, name in enumerate(node_names)
            if self._nodes[name].P_spec_Pa is not None
        ]
        if not p_spec_indices:
            raise ValueError(
                "Network.solve: at least one node must have a specified "
                "pressure (P=...) to anchor the solution."
            )
        return free_indices, p_spec_indices

    def _build_adjacency(self, idx_of):
        """Per-node incidence list of (edge_index, sign) pairs.

        sign=+1 means a positive mdot enters this node; sign=-1 means it
        leaves.  Used by both the mass-balance residuals and the post-solve
        Q_ext recovery.
        """
        adj = [[] for _ in range(len(idx_of))]
        for e_idx, edge in enumerate(self._edges):
            adj[idx_of[edge.from_node]].append((e_idx, -1))
            adj[idx_of[edge.to_node]  ].append((e_idx, +1))
        return adj

    def _resolve_initial_guesses(self, P_init, mdot_init_kgs, rho):
        """Fill in missing initial guesses with reasonable defaults.

        P_init defaults to the mean of all spec'd pressures.  mdot_init_kgs
        defaults to the largest |Q_ext| actually spec'd at any node (in kg/s),
        falling back to 10 kg/s if nothing is spec'd.  Scaling the flow
        guess to the network's actual carrying capacity matters because a
        small initial mdot makes fsolve's FD Jacobian noisy on friction.
        """
        if P_init is None:
            P_specs = [n.P_spec_Pa for n in self._nodes.values()
                       if n.P_spec_Pa is not None]
            P_init = sum(P_specs) / len(P_specs)
        if mdot_init_kgs is None:
            spec_magnitudes = [
                abs(_qty_to_mdot_kgs_incompressible(n.Q_ext_spec_qty, rho))
                for n in self._nodes.values()
                if n.Q_ext_spec_qty is not None
            ]
            mdot_init_kgs = max(spec_magnitudes) if spec_magnitudes else 10.0
        return P_init, mdot_init_kgs

    def _report_convergence(self, info, ier, msg, verbose):
        """Decide whether fsolve actually converged and emit any warnings.

        fsolve sometimes reports ier=5 ("not making good progress") when the
        residual has already hit floating-point noise.  Cross-check the norm
        so a tiny residual still counts as success.
        """
        residual_norm = float(np.linalg.norm(info["fvec"]))
        converged = (ier == 1) or (residual_norm < 1e-4)
        if not converged:
            warnings.warn(
                f"Network.solve: fsolve did not converge (ier={ier}, "
                f"residual norm = {residual_norm:.3e}): {msg}",
                stacklevel=2,
            )
        if verbose:
            print(
                f"Network.solve: ier={ier}, nfev={info['nfev']}, "
                f"residual norm = {residual_norm:.3e}"
            )
        return converged

    def _recover_pspec_mdot_ext(self, mdot_arr, adj, mdot_ext_spec, node_names):
        """Back out external mass flow at every node after the solve.

        P-spec nodes: derived from local mass balance (sum of edge inflows
        equals -Q_ext).  All other nodes: copy the spec value, which has
        already been zero-defaulted for interior junctions.
        """
        out = {}
        for i, name in enumerate(node_names):
            if self._nodes[name].P_spec_Pa is not None:
                net_in = sum(sign * mdot_arr[e_idx] for e_idx, sign in adj[i])
                out[name] = float(-net_in)
            else:
                out[name] = float(mdot_ext_spec[name])
        return out

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, fluid, P_init=None, mdot_init_kgs=None, verbose=False,
              xtol=1e-8, maxfev=5000):
        """Solve the network for all nodal pressures and edge mass flows.

        The solver works internally on signed mass flow rates [kg/s] so the
        physics and bookkeeping match the compressible solver.  User inputs
        at add_node() may be either volumetric or mass pint Quantities;
        conversion to kg/s happens here using fluid.density_si.

        Args:
            fluid          : Incompressible_Fluid instance.
            P_init         : float (Pa) or None.  Initial guess for free-node
                             pressures.  If None, the mean of all P-spec
                             pressures is used.
            mdot_init_kgs  : float (kg/s) or None.  Initial guess for every
                             edge's signed mass flow.  If None, defaults to
                             the largest |Q_ext| spec'd at any node (in kg/s),
                             or 10 kg/s if no Q_ext is spec'd.  Nonzero to
                             avoid the sign(mdot) kink at zero.
            verbose        : bool, print solver progress.
            xtol           : float, fsolve absolute tolerance on unknowns.
            maxfev         : int, max function evaluations.

        Returns:
            NetworkResult with both mass-flow ("mdot_kgs", "Q_ext_mdot_kgs")
            and volumetric ("Q_m3s", "Q_ext_m3s") dicts; the latter are
            derived from the former using fluid.density_si and are kept
            for backwards compatibility.
        """
        node_names = list(self._node_order)
        n_nodes    = len(node_names)
        n_edges    = len(self._edges)
        if n_nodes == 0 or n_edges == 0:
            raise ValueError("Network.solve: network has no nodes or edges.")

        idx_of = {name: i for i, name in enumerate(node_names)}
        rho    = fluid.density_si

        mdot_ext_spec = self._mdot_ext_spec_kgs(rho)
        free_indices, p_spec_indices = self._partition_node_indices(node_names)
        adj = self._build_adjacency(idx_of)
        P_init, mdot_init_kgs = self._resolve_initial_guesses(
            P_init, mdot_init_kgs, rho,
        )

        n_free = len(free_indices)
        x0 = np.concatenate([
            np.full(n_free, float(P_init)),
            np.full(n_edges, float(mdot_init_kgs)),
        ])

        # P-spec values are loop-invariant; bake them into a template that
        # unpack() copies once per Newton step and then overwrites at the
        # free-node slots.
        P_template = np.empty(n_nodes)
        for i in p_spec_indices:
            P_template[i] = self._nodes[node_names[i]].P_spec_Pa

        # Lazy cache of reversed shadow components keyed by id(original).
        reversed_cache = {}

        def unpack(x):
            P = P_template.copy()
            P[free_indices] = x[:n_free]
            return P, x[n_free:]

        def residuals(x):
            P, mdot_e = unpack(x)
            res = np.empty(n_free + n_edges)

            # Mass balance at each non-P-spec node.
            for j, i in enumerate(free_indices):
                net_in = sum(sign * mdot_e[e_idx] for e_idx, sign in adj[i])
                res[j] = net_in + mdot_ext_spec[node_names[i]]

            # Pipe equation per edge.  _component_signed_dP accepts the SI
            # flow magnitude (currently typed as volumetric m^3/s by the
            # incompressible helper); convert mdot -> Q for the call.
            for e_idx, edge in enumerate(self._edges):
                i_from = idx_of[edge.from_node]
                i_to   = idx_of[edge.to_node]
                Q_si = mdot_e[e_idx] / rho     # signed volumetric for dP call
                dP_io = sum(
                    _component_signed_dP(c, fluid, Q_si, reversed_cache)
                    for c in edge.components
                )
                res[n_free + e_idx] = (P[i_from] - P[i_to]) + dP_io

            return res

        sol, info, ier, msg = fsolve(
            residuals, x0, full_output=True, xtol=xtol, maxfev=maxfev,
        )
        converged = self._report_convergence(info, ier, msg, verbose)

        P_arr, mdot_arr = unpack(sol)
        mdot_ext_out = self._recover_pspec_mdot_ext(
            mdot_arr, adj, mdot_ext_spec, node_names,
        )
        mdot_kgs = {self._edges[e].name: float(mdot_arr[e]) for e in range(n_edges)}
        return NetworkResult(
            network=self,
            fluid=fluid,
            P_Pa={name: float(P_arr[i]) for i, name in enumerate(node_names)},
            mdot_kgs=mdot_kgs,
            mdot_ext_kgs=mdot_ext_out,
            converged=bool(converged),
        )


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------

class NetworkResult:
    """Output of Network.solve().

    Supports two access styles:

    1. Dict-style (preserves the original return shape):
           result["P_Pa"]["MIX-1"]
           result["Q_m3s"]["PIPE-3"]
           result["Q_ext_m3s"]["inlet"]
           result["converged"]

    2. Per-object views (built on demand):
           result.edge("PIPE-3")          -> EdgeResult
           result.component(line_seg_obj) -> ComponentResult

    Components are looked up by object identity, not by name, since a name
    may be duplicated and identity is unambiguous.
    """

    def __init__(self, network, fluid, P_Pa, mdot_kgs, mdot_ext_kgs, converged):
        self._network = network
        self._fluid = fluid
        # The mass-flow dicts are primary; volumetric dicts are derived
        # from them via fluid.density_si and stored alongside so existing
        # callers that look up result["Q_m3s"] / result["Q_ext_m3s"]
        # keep working unchanged.
        rho = float(fluid.density_si)
        Q_m3s     = {k: v / rho for k, v in mdot_kgs.items()}
        Q_ext_m3s = {k: v / rho for k, v in mdot_ext_kgs.items()}
        self._data = {
            "P_Pa":             P_Pa,
            "mdot_kgs":         mdot_kgs,
            "mdot_ext_kgs":     mdot_ext_kgs,
            "Q_m3s":            Q_m3s,
            "Q_ext_m3s":        Q_ext_m3s,
            "converged":        converged,
        }

    # -- dict-style accessors --

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    # -- object-style accessors --

    @property
    def fluid(self):
        return self._fluid

    @property
    def converged(self):
        return self._data["converged"]

    def edge(self, name_or_edge):
        """Return an EdgeResult for the edge identified by name or by Edge
        instance."""
        if isinstance(name_or_edge, str):
            for e in self._network._edges:
                if e.name == name_or_edge:
                    return EdgeResult(self, e)
            raise KeyError(f"NetworkResult.edge: no edge named {name_or_edge!r}")
        return EdgeResult(self, name_or_edge)

    def component(self, comp):
        """Return a ComponentResult for the given component object.

        Dispatches by object identity.  Raises KeyError if the component
        does not appear on any edge, or ValueError if it appears on more
        than one (which would make P_in / P_out ambiguous).
        """
        hits = []
        for e in self._network._edges:
            for c in e.components:
                if c is comp:
                    hits.append(e)
                    break
        if not hits:
            raise KeyError(
                f"NetworkResult.component: component {comp!r} is not on any "
                f"edge in this network."
            )
        if len(hits) > 1:
            raise ValueError(
                f"NetworkResult.component: component {comp!r} appears on "
                f"multiple edges ({[e.name for e in hits]}); P_in/P_out would "
                f"be ambiguous."
            )
        return ComponentResult(self, hits[0], comp)


class EdgeResult:
    """View of one edge's solved state."""

    def __init__(self, network_result, edge):
        self._result = network_result
        self._edge = edge

    @property
    def name(self):
        return self._edge.name

    @property
    def edge(self):
        return self._edge

    @property
    def mdot_kgs(self):
        """Signed mass flow in the edge's nominal from->to direction."""
        return self._result["mdot_kgs"][self._edge.name]

    @property
    def mdot_abs_kgs(self):
        return abs(self.mdot_kgs)

    @property
    def Q_m3s(self):
        """Signed volumetric flow (derived from mass flow via fluid density)."""
        return self._result["Q_m3s"][self._edge.name]

    @property
    def Q_abs_m3s(self):
        return abs(self.Q_m3s)

    @property
    def flow_direction(self):
        """'forward' (mdot >= 0) or 'reverse' (mdot < 0)."""
        return "forward" if self.mdot_kgs >= 0.0 else "reverse"

    @property
    def P_from_Pa(self):
        """Pressure at the edge's nominal upstream end."""
        return self._result["P_Pa"][self._edge.from_node]

    @property
    def P_to_Pa(self):
        """Pressure at the edge's nominal downstream end."""
        return self._result["P_Pa"][self._edge.to_node]

    @property
    def P_inlet_Pa(self):
        """Pressure at the physical flow inlet (upstream in flow direction)."""
        return self.P_from_Pa if self.mdot_kgs >= 0.0 else self.P_to_Pa

    @property
    def P_outlet_Pa(self):
        """Pressure at the physical flow outlet (downstream in flow direction)."""
        return self.P_to_Pa if self.mdot_kgs >= 0.0 else self.P_from_Pa

    def __repr__(self):
        return (
            f"EdgeResult({self.name!r}, "
            f"{self.flow_direction}, "
            f"mdot={self.mdot_kgs:.6g} kg/s, "
            f"P_inlet={self.P_inlet_Pa:.6g} Pa, "
            f"P_outlet={self.P_outlet_Pa:.6g} Pa)"
        )


class ComponentResult:
    """View of one component's solved state on its edge.

    Conventions follow the flow direction, not the component's nominal
    orientation:

        P_in_Pa    pressure where the fluid enters this component.
        P_out_Pa   pressure where the fluid leaves this component.
        Q_m3s      signed volumetric flow in the edge's nominal direction
                   (so this matches the edge's Q_m3s; for the flow magnitude
                   use Q_abs_m3s).
        flow_direction  'forward' or 'reverse' relative to the edge's nominal
                        direction.

    For Line_Segment-like components, pressure_profile() returns a per-point
    pressure profile walked in flow direction.
    """

    def __init__(self, network_result, edge, component):
        self._result = network_result
        self._edge = edge
        self._component = component
        self._compute_endpoint_pressures()

    def _compute_endpoint_pressures(self):
        """Walk the edge's components in flow direction, accumulating P, and
        record P_in / P_out for this component."""
        mdot = self._result["mdot_kgs"][self._edge.name]
        fluid = self._result.fluid

        if mdot >= 0.0:
            comps_in_flow_order = list(self._edge.components)
            P_running = self._result["P_Pa"][self._edge.from_node]
        else:
            comps_in_flow_order = list(reversed(self._edge.components))
            P_running = self._result["P_Pa"][self._edge.to_node]

        abs_mdot_qty = ureg.Quantity(abs(mdot), "kg/s")
        for c in comps_in_flow_order:
            P_in_c = P_running
            if mdot >= 0.0:
                dP = c.dP(fluid, flow_rate=abs_mdot_qty)
            else:
                # Reverse flow: evaluate the component's reversed shadow at
                # +|mdot|, which gives the dP in flow direction directly.
                rev = _reversed_component(c)
                dP = rev.dP(fluid, flow_rate=abs_mdot_qty)
            P_out_c = P_in_c + dP

            if c is self._component:
                self.P_in_Pa = P_in_c
                self.P_out_Pa = P_out_c
                return
            P_running = P_out_c

        raise RuntimeError(
            "ComponentResult: walked past every component without finding the "
            "target -- internal inconsistency."
        )

    @property
    def component(self):
        return self._component

    @property
    def edge_name(self):
        return self._edge.name

    @property
    def mdot_kgs(self):
        """Signed mass flow in the edge's nominal direction."""
        return self._result["mdot_kgs"][self._edge.name]

    @property
    def mdot_abs_kgs(self):
        return abs(self.mdot_kgs)

    @property
    def Q_m3s(self):
        """Signed volumetric flow (derived from mass flow via fluid density)."""
        return self._result["Q_m3s"][self._edge.name]

    @property
    def Q_abs_m3s(self):
        return abs(self.Q_m3s)

    @property
    def flow_direction(self):
        return "forward" if self.mdot_kgs >= 0.0 else "reverse"

    @property
    def dP_Pa(self):
        """P_out - P_in (in flow direction).  Negative for a friction loss."""
        return self.P_out_Pa - self.P_in_Pa

    def pressure_profile(self):
        """Per-point pressure/velocity profile in flow direction.

        Only meaningful for components that expose a pressure_profile()
        method (Line_Segment-like).  Returns the same list-of-dicts shape
        as Line_Segment.pressure_profile():
            [{"distance_m":..., "elevation_m":..., "P_Pa":..., "v_ms":...}, ...]

        For reverse flow, the profile is computed on a reversed shadow, so
        distance_m runs from 0 at the flow inlet (= component's nominal
        outlet) to total_length_m at the flow outlet (= component's nominal
        inlet).
        """
        if not hasattr(self._component, "pressure_profile"):
            raise AttributeError(
                f"{type(self._component).__name__} has no pressure_profile() "
                f"method -- only Line_Segment-like components support "
                f"distance-resolved profiles."
            )
        mdot = self.mdot_kgs
        abs_mdot_qty = ureg.Quantity(abs(mdot), "kg/s")
        target = self._component if mdot >= 0.0 else _reversed_component(self._component)
        return target.pressure_profile(
            self._result.fluid,
            P0=self.P_in_Pa,
            flow_rate=abs_mdot_qty,
        )

    def __repr__(self):
        cname = type(self._component).__name__
        seg_name = getattr(self._component, "name", None)
        seg_str = f"{cname}({seg_name!r})" if seg_name else cname
        return (
            f"ComponentResult({seg_str} on {self._edge.name!r}, "
            f"{self.flow_direction}, "
            f"Q={self.Q_m3s:.6g} m^3/s, "
            f"P_in={self.P_in_Pa:.6g} Pa, "
            f"P_out={self.P_out_Pa:.6g} Pa)"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_single_segment():
    """One inlet, one outlet, one Line_Segment in between.

    Verifies that the network solver recovers the same dP as a direct call
    to Line_Segment.dP() when the flow rate is fixed by the Q_ext spec at
    the inlet.
    """
    from incompressible import Line_Segment, Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )
    seg = Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(2000.0, "ft"),
        elevation_change=ureg.Quantity(25.0, "ft"),
        name="single",
    )

    Q = ureg.Quantity(10000.0, "oil_bbl/day")
    P_out_spec = ureg.Quantity(50.0, "psi").to("Pa").magnitude

    direct_dP = seg.dP(fluid, Q)                          # negative for loss
    expected_P_in = P_out_spec - direct_dP                # P_in = P_out - dP

    net = Network()
    net.add_node("in",  Q_ext=Q)
    net.add_node("out", P=P_out_spec)
    net.add_edge("seg", "in", "out", seg)

    result = net.solve(fluid, verbose=True)

    P_in = result["P_Pa"]["in"]
    err_psi = abs(P_in - expected_P_in) / 6894.757
    print(
        f"[single-segment]  direct dP = {direct_dP/6894.757:.3f} psi,  "
        f"network P_in = {P_in/6894.757:.3f} psi,  "
        f"expected P_in = {expected_P_in/6894.757:.3f} psi,  "
        f"err = {err_psi:.2e} psi"
    )
    assert err_psi < 1e-3, "single-segment mismatch"


def _test_parallel_two_branches():
    """Two pipes in parallel from inlet to outlet.

    Compares against parallel.parallel_incompressible() on the same network.
    """
    from incompressible import Line_Segment, Incompressible_Fluid
    from parallel import parallel_incompressible

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0,
        viscosity=ureg.Quantity(1.0, "cP"),
    )
    rgh = ureg.Quantity(0.00015, "ft")
    seg1 = Line_Segment(roughness=rgh, id_val=ureg.Quantity(3.068, "inch"),
                        length=ureg.Quantity(2000.0, "ft"),
                        elevation_change=ureg.Quantity(0.0, "ft"), name="A")
    seg2 = Line_Segment(roughness=rgh, id_val=ureg.Quantity(4.026, "inch"),
                        length=ureg.Quantity(2000.0, "ft"),
                        elevation_change=ureg.Quantity(0.0, "ft"), name="B")

    Q_total = ureg.Quantity(10000.0, "oil_bbl/day")
    Q_total_si = Q_total.to("m^3/s").magnitude

    dP_ref, frac_ref = parallel_incompressible([seg1, seg2], fluid, Q_total)

    net = Network()
    net.add_node("in",  Q_ext=Q_total)
    net.add_node("out", P=0.0)
    net.add_edge("A", "in", "out", seg1)
    net.add_edge("B", "in", "out", seg2)
    result = net.solve(fluid)

    dP_net = result["P_Pa"]["out"] - result["P_Pa"]["in"]   # P_out - P_in
    frac_net = [result["Q_m3s"]["A"] / Q_total_si,
                result["Q_m3s"]["B"] / Q_total_si]

    print(
        f"[parallel-2]      reference dP = {dP_ref/6894.757:.4f} psi  "
        f"fractions {frac_ref}\n"
        f"                  network   dP = {dP_net/6894.757:.4f} psi  "
        f"fractions {frac_net}"
    )
    assert abs(dP_net - dP_ref) / abs(dP_ref) < 1e-3, "parallel dP mismatch"
    assert all(abs(a - b) < 1e-3 for a, b in zip(frac_net, frac_ref)), \
        "parallel fraction mismatch"


def _build_screenshot_network(*, inlet_1_spec, inlet_5_spec,
                              p_outlet_15_psi, p_outlet_16_psi):
    """Build the DWSIM-screenshot topology with configurable boundaries.

    Topology (mixers/splitters are interior nodes):

        Inlets : 1, 5      Outlets : 15, 16

        PIPE-1 :  1     -> MIX-1
        conn-5 :  5     -> MIX-1   (zero-length connector)
        PIPE-2 :  MIX-1 -> SPL-1
        PIPE-3 :  SPL-1 -> MIX-2
        PIPE-4 :  SPL-1 -> SPL-2
        PIPE-5 :  SPL-2 -> MIX-2   (flow direction emerges from the solve)
        conn-16:  SPL-2 -> 16
        conn-15:  MIX-2 -> 15

    Inlet boundary conditions are passed as add_node kwargs dicts so the
    same topology can be driven from either pressure-spec or flow-spec
    inlets without duplicating the builder, e.g.:
        inlet_1_spec = {"P":     ureg.Quantity(200.0, "psi")}
        inlet_1_spec = {"Q_ext": ureg.Quantity(10000.0, "oil_bbl/day")}
    """
    from incompressible import Line_Segment

    rgh = ureg.Quantity(0.00015, "ft")

    def pipe(name, id_in, L_ft, dz_ft=0.0):
        return Line_Segment(
            roughness=rgh,
            id_val=ureg.Quantity(id_in, "inch"),
            length=ureg.Quantity(L_ft, "ft"),
            elevation_change=ureg.Quantity(dz_ft, "ft"),
            name=name,
        )

    PIPE_1 = pipe("PIPE-1", 4.026,  500.0)
    PIPE_2 = pipe("PIPE-2", 4.026, 1000.0)
    PIPE_3 = pipe("PIPE-3", 3.068,  800.0)
    PIPE_4 = pipe("PIPE-4", 3.068,  800.0)
    PIPE_5 = pipe("PIPE-5", 2.067,  400.0)

    net = Network()
    net.add_node("1",  **inlet_1_spec)
    net.add_node("5",  **inlet_5_spec)
    net.add_node("15", P=ureg.Quantity(p_outlet_15_psi, "psi"))
    net.add_node("16", P=ureg.Quantity(p_outlet_16_psi, "psi"))
    net.add_node("MIX-1")
    net.add_node("SPL-1")
    net.add_node("MIX-2")
    net.add_node("SPL-2")

    net.add_edge("PIPE-1",  "1",     "MIX-1", PIPE_1)
    net.add_edge("conn-5",  "5",     "MIX-1", [])
    net.add_edge("PIPE-2",  "MIX-1", "SPL-1", PIPE_2)
    net.add_edge("PIPE-3",  "SPL-1", "MIX-2", PIPE_3)
    net.add_edge("PIPE-4",  "SPL-1", "SPL-2", PIPE_4)
    net.add_edge("PIPE-5",  "SPL-2", "MIX-2", PIPE_5)
    net.add_edge("conn-16", "SPL-2", "16",    [])
    net.add_edge("conn-15", "MIX-2", "15",    [])
    return net


def _print_screenshot_result(label, result):
    Q_to_bpd = ureg.Quantity(1.0, "m^3/s").to("oil_bbl/day").magnitude
    print(f"[{label}] node pressures (psi):")
    for name, P in result["P_Pa"].items():
        print(f"             {name:>6s}  P = {P/6894.757:8.3f} psi")
    print(f"[{label}] edge flows (BPD, signed in nominal direction):")
    for name, Q in result["Q_m3s"].items():
        arrow = " (REVERSED)" if Q < 0 else ""
        print(f"             {name:>7s}  Q = {Q * Q_to_bpd:10.1f} BPD{arrow}")
    print(f"[{label}] external flows at boundary nodes (BPD):")
    for name in ("1", "5", "15", "16"):
        Q = result["Q_ext_m3s"][name]
        print(f"             {name:>3s}     Q_ext = {Q * Q_to_bpd:10.1f} BPD")
    total = sum(result["Q_ext_m3s"].values())
    print(f"[{label}] total external flow sum = {total * Q_to_bpd:.3e} BPD")
    assert abs(total) < 1e-9, f"{label}: global mass balance failure"


def _test_screenshot_forward_PIPE5():
    """Both outlets open; outlet 16 at higher back-pressure than outlet 15.

    Lower back-pressure on 15 pulls flow toward MIX-2.  Some of the
    SPL-1 -> PIPE-4 -> SPL-2 lower branch then traverses PIPE-5 forward
    (SPL-2 -> MIX-2) to reach outlet 15 too.
    """
    from incompressible import Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=10.0, viscosity=ureg.Quantity(1.0, "cP"),
    )
    net = _build_screenshot_network(
        inlet_1_spec={"P": ureg.Quantity(200.0, "psi")},
        inlet_5_spec={"P": ureg.Quantity(180.0, "psi")},
        p_outlet_15_psi=40.0, p_outlet_16_psi=80.0,
    )
    result = net.solve(fluid, verbose=True)
    _print_screenshot_result("screenshot-FWD", result)
    assert result["Q_m3s"]["PIPE-5"] > 0.0, "expected PIPE-5 forward"


def _test_screenshot_reverse_PIPE5():
    """Outlet 15 throttled high, outlet 16 open.

    With outlet 15 nearly choked off, almost all flow must exit at 16.
    Whatever enters MIX-2 via PIPE-3 has only one way out -- backwards
    through PIPE-5 to SPL-2 and then to outlet 16.
    """
    from incompressible import Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0, viscosity=ureg.Quantity(1.0, "cP"),
    )
    net = _build_screenshot_network(
        inlet_1_spec={"P": ureg.Quantity(200.0, "psi")},
        inlet_5_spec={"P": ureg.Quantity(180.0, "psi")},
        p_outlet_15_psi=175.0, p_outlet_16_psi=40.0,
    )
    result = net.solve(fluid, verbose=True)
    _print_screenshot_result("screenshot-REV", result)
    assert result["Q_m3s"]["PIPE-5"] < 0.0, "expected PIPE-5 reversed"


def _test_specflow():
    """Spec'd-flow inlets driving spec'd-pressure outlets on the screenshot topology.

    Both inlets supply a fixed volumetric flow (10000 and 5000 BPD); both
    outlets are held at fixed back-pressures.  The solver finds all interior
    pressures and the routing across PIPE-3 / PIPE-4 / PIPE-5.  Exercises the
    Q_ext inlet path that the pressure-spec screenshot tests don't cover.
    """
    from incompressible import Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=10.0, viscosity=ureg.Quantity(1.0, "cP"),
    )
    net = _build_screenshot_network(
        inlet_1_spec={"Q_ext": ureg.Quantity(10000.0, "oil_bbl/day")},
        inlet_5_spec={"Q_ext": ureg.Quantity( 5000.0, "oil_bbl/day")},
        p_outlet_15_psi=170.0, p_outlet_16_psi=110.0,
    )
    result = net.solve(fluid, verbose=True)
    _print_screenshot_result("screenshot-SPECFLOW", result)


def _test_query_api():
    """Demo of the per-component / per-edge result accessors.

    Builds a small two-branch network with one branch in reverse flow,
    then walks each Line_Segment via result.component() and prints
    its flow-direction pressure profile.  Also queries a Bend.
    """
    from incompressible import Line_Segment, Bend, Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0, viscosity=ureg.Quantity(1.0, "cP"),
    )
    rgh = ureg.Quantity(0.00015, "ft")

    # Three pipes: upper branch (PIPE-A), lower branch (PIPE-B with a bend
    # in series), and a cross-link PIPE-X between the two branches'
    # midpoints.  Throttling the upper outlet pushes flow through PIPE-X
    # in reverse.
    PIPE_A = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(1500.0, "ft"),
        elevation_change=ureg.Quantity(20.0, "ft"),
        name="A",
    )
    PIPE_B = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(1500.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="B",
    )
    BEND_B = Bend(ureg.Quantity(3.068, "inch"), 90.0, 1.5)
    PIPE_X = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(2.067, "inch"),
        length=ureg.Quantity(300.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="X",
    )

    net = Network()
    net.add_node("in", Q_ext=ureg.Quantity(8000.0, "oil_bbl/day"))
    net.add_node("split")
    net.add_node("mid_upper")
    net.add_node("mid_lower")
    net.add_node("merge")
    net.add_node("mid_lower2")
    net.add_node("out_upper", P=ureg.Quantity(180.0, "psi"))   # throttled
    net.add_node("out_lower", P=ureg.Quantity( 80.0, "psi"))

    net.add_edge("feed",        "in",         "split",      [])
    net.add_edge("upper_head",  "split",      "mid_upper",  [])
    net.add_edge("PIPE-A",      "mid_upper",  "merge",      PIPE_A)
    net.add_edge("upper_tail",  "merge",      "out_upper",  [])
    net.add_edge("lower_head",  "split",      "mid_lower",  [])
    # bend + pipe in series on one edge -- demonstrates multi-component dP walk
    net.add_edge("PIPE-B",      "mid_lower",  "mid_lower2", [BEND_B, PIPE_B])
    net.add_edge("lower_tail",  "mid_lower2", "out_lower",  [])
    net.add_edge("PIPE-X",      "mid_upper",  "mid_lower",  PIPE_X)

    result = net.solve(fluid, verbose=True)
    assert result.converged

    print("\n[query-api] EdgeResult for PIPE-X (the cross-link):")
    er = result.edge("PIPE-X")
    print(f"   {er!r}")
    print(f"   flow direction = {er.flow_direction}")
    print(f"   P at flow inlet  = {er.P_inlet_Pa/6894.757:.3f} psi")
    print(f"   P at flow outlet = {er.P_outlet_Pa/6894.757:.3f} psi")

    print("\n[query-api] ComponentResult for PIPE_A (Line_Segment):")
    cr = result.component(PIPE_A)
    print(f"   {cr!r}")
    print(f"   dP across component = {cr.dP_Pa/6894.757:+.3f} psi  (in flow direction)")
    prof = cr.pressure_profile()
    print(f"   profile: {len(prof)} points, first / last:")
    print(f"     d={prof[ 0]['distance_m']:7.2f} m  "
          f"P={prof[ 0]['P_Pa']/6894.757:7.3f} psi  "
          f"v={prof[ 0]['v_ms']:5.3f} m/s  "
          f"z={prof[ 0]['elevation_m']:5.2f} m")
    print(f"     d={prof[-1]['distance_m']:7.2f} m  "
          f"P={prof[-1]['P_Pa']/6894.757:7.3f} psi  "
          f"v={prof[-1]['v_ms']:5.3f} m/s  "
          f"z={prof[-1]['elevation_m']:5.2f} m")

    print("\n[query-api] ComponentResult for PIPE_X (cross-link, direction emerges from solve):")
    cr = result.component(PIPE_X)
    print(f"   {cr!r}")
    print(f"   dP across component = {cr.dP_Pa/6894.757:+.3f} psi  (in flow direction)")
    prof = cr.pressure_profile()
    print(f"   profile first / last in flow direction:")
    print(f"     d={prof[ 0]['distance_m']:7.2f} m  "
          f"P={prof[ 0]['P_Pa']/6894.757:7.3f} psi  "
          f"z={prof[ 0]['elevation_m']:5.2f} m")
    print(f"     d={prof[-1]['distance_m']:7.2f} m  "
          f"P={prof[-1]['P_Pa']/6894.757:7.3f} psi  "
          f"z={prof[-1]['elevation_m']:5.2f} m")

    print("\n[query-api] ComponentResult for BEND_B (fitting; no profile expected):")
    cr = result.component(BEND_B)
    print(f"   {cr!r}")
    print(f"   dP across component = {cr.dP_Pa/6894.757:+.3f} psi")
    try:
        cr.pressure_profile()
        print("   (expected AttributeError, did not raise -- BUG)")
    except AttributeError as e:
        print(f"   pressure_profile() raises AttributeError as expected: {e}")


def _test_three_modes():
    """Round-trip consistency check for the three calculation-mode wrappers.

    Builds a simple two-pipe series network (with a 90 deg bend between the
    pipes) and runs the same physical scenario through all three modes:

        Mode 1 (find P_out)  given P_in, Q.
        Mode 2 (find P_in)   given P_out, Q.   (Should recover the original P_in.)
        Mode 3 (find Q)      given P_in, P_out.  (Should recover the original Q.)

    Same Network object is reused across all three calls -- the wrappers
    update boundary specs in place.
    """
    from incompressible import Line_Segment, Bend, Incompressible_Fluid

    fluid = Incompressible_Fluid.from_api_gravity(
        api_gravity=50.0, viscosity=ureg.Quantity(1.0, "cP"),
    )
    rgh = ureg.Quantity(0.00015, "ft")
    seg1 = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(3.068, "inch"),
        length=ureg.Quantity(1500.0, "ft"),
        elevation_change=ureg.Quantity(15.0, "ft"),
        name="seg1",
    )
    seg2 = Line_Segment(
        roughness=rgh, id_val=ureg.Quantity(4.026, "inch"),
        length=ureg.Quantity(2000.0, "ft"),
        elevation_change=ureg.Quantity(-5.0, "ft"),
        name="seg2",
    )
    bend = Bend(ureg.Quantity(3.068, "inch"), 90.0, 1.5)

    net = Network()
    net.add_node("in")
    net.add_node("mid")
    net.add_node("out")
    net.add_edge("E1", "in",  "mid", [seg1, bend])
    net.add_edge("E2", "mid", "out", seg2)

    P_in_known = ureg.Quantity(200.0, "psi")
    Q_known    = ureg.Quantity(10000.0, "oil_bbl/day")

    # ---- Mode 1: find P_out from (P_in, Q).
    r1 = net.solve_for_outlet_pressure(
        fluid, inlet="in", outlet="out", P_inlet=P_in_known, Q=Q_known,
    )
    P_out_solved = r1["P_Pa"]["out"]
    print(f"[three-modes] Mode 1: P_in = {P_in_known.to('psi'):.3f~P}, "
          f"Q = {Q_known.to('oil_bbl/day'):.0f~P}  ->  "
          f"P_out = {P_out_solved/6894.757:.4f} psi")

    # ---- Mode 2: find P_in from (P_out, Q).  Should recover P_in_known.
    P_out_quantity = ureg.Quantity(P_out_solved, "Pa")
    r2 = net.solve_for_inlet_pressure(
        fluid, inlet="in", outlet="out", P_outlet=P_out_quantity, Q=Q_known,
    )
    P_in_solved = r2["P_Pa"]["in"]
    err_psi = abs(P_in_solved - P_in_known.to("Pa").magnitude) / 6894.757
    print(f"[three-modes] Mode 2: P_out = {P_out_solved/6894.757:.4f} psi, "
          f"Q = {Q_known.to('oil_bbl/day'):.0f~P}  ->  "
          f"P_in = {P_in_solved/6894.757:.4f} psi  "
          f"(round-trip err = {err_psi:.2e} psi)")
    assert err_psi < 1e-4, "mode 2 round-trip failed"

    # ---- Mode 3: find Q from (P_in, P_out).  Should recover Q_known.
    r3 = net.solve_for_flow_rate(
        fluid, inlet="in", outlet="out",
        P_inlet=P_in_known, P_outlet=P_out_quantity,
    )
    Q_solved_si = r3["Q_ext_m3s"]["in"]
    Q_known_si  = Q_known.to("m^3/s").magnitude
    rel_err = abs(Q_solved_si - Q_known_si) / Q_known_si
    Q_to_bpd = ureg.Quantity(1.0, "m^3/s").to("oil_bbl/day").magnitude
    print(f"[three-modes] Mode 3: P_in = {P_in_known.to('psi'):.3f~P}, "
          f"P_out = {P_out_solved/6894.757:.4f} psi  ->  "
          f"Q = {Q_solved_si * Q_to_bpd:.2f} BPD  "
          f"(round-trip rel err = {rel_err:.2e})")
    assert rel_err < 1e-4, "mode 3 round-trip failed"

    print("[three-modes] All three modes self-consistent.")


if __name__ == "__main__":
    # _test_single_segment()
    # print()
    # _test_parallel_two_branches()
    # print()
    # _test_screenshot_forward_PIPE5()
    # print()
    # _test_screenshot_reverse_PIPE5()
    # _test_specflow()
    # print()
    # _test_query_api()
    _test_three_modes()