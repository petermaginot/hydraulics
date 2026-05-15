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
from typing import Optional

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


def _q_ext_to_si(Q_ext, fluid_density_si):
    """Convert an external flow rate to volumetric SI (m^3/s).

    Accepts:
        -- None (returns None).
        -- pint Quantity, volumetric ([length]^3/[time]) or mass
           ([mass]/[time]); mass flow is converted using fluid_density_si.
        -- plain float, taken as m^3/s.
    """
    if Q_ext is None:
        return None
    if hasattr(Q_ext, "dimensionality"):
        dim = Q_ext.dimensionality
        if dim == {"[length]": 3, "[time]": -1}:
            return Q_ext.to("m^3/s").magnitude
        if dim == {"[mass]": 1, "[time]": -1}:
            if fluid_density_si is None:
                raise ValueError(
                    "Mass-flow Q_ext requires a fluid density.  Pass the fluid "
                    "to add_node() (e.g. fluid_for_units=fluid) or supply a "
                    "volumetric flow rate instead."
                )
            return Q_ext.to("kg/s").magnitude / float(fluid_density_si)
        raise ValueError(
            f"Q_ext has unrecognized dimensions {dict(dim)}; expected "
            f"volumetric or mass flow."
        )
    return float(Q_ext)


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
        Q_ext_spec_si   : float or None, specified external supply [m^3/s].
                          Positive = flow into the node from outside the
                          network.  None on an interior node means 0.
    """
    name: str
    elevation_m: float = 0.0
    P_spec_Pa: Optional[float] = None
    Q_ext_spec_si: Optional[float] = None


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
                               Optional.
            fluid_for_units  : Incompressible_Fluid, required only when
                               Q_ext is a mass flow rate (used to convert
                               to volumetric).  Ignored otherwise.

        Specifying both P and Q_ext is an error.  Specifying neither makes
        the node interior with Q_ext = 0.
        """
        if name in self._nodes:
            raise ValueError(f"Network.add_node: node {name!r} already exists.")

        elev_si = _to_si_or_none(elevation, "m")
        if elev_si is None:
            elev_si = 0.0

        P_si = _to_si_or_none(P, "Pa")

        rho = fluid_for_units.density_si if fluid_for_units is not None else None
        Q_si = _q_ext_to_si(Q_ext, rho)

        if P_si is not None and Q_si is not None:
            raise ValueError(
                f"Network.add_node: node {name!r} cannot specify both P and "
                f"Q_ext."
            )

        self._nodes[name] = Node(name, elev_si, P_si, Q_si)
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
    # Solve
    # ------------------------------------------------------------------

    def solve(self, fluid, P_init=None, Q_init_si=0.01, verbose=False,
              xtol=1e-8, maxfev=5000):
        """Solve the network for all nodal pressures and edge flows.

        Args:
            fluid     : Incompressible_Fluid instance.
            P_init    : float (Pa) or None.  Initial guess for free-node
                        pressures.  If None, the mean of all P-spec
                        pressures is used.
            Q_init_si : float (m^3/s).  Initial guess for every edge flow.
                        A nonzero value avoids the sign(Q) kink at Q=0;
                        the default 0.01 m^3/s (~ 5400 BPD) is a reasonable
                        order of magnitude for typical pipeline cases.
            verbose   : bool, print solver progress.
            xtol      : float, fsolve absolute tolerance on the unknowns.
            maxfev    : int, max function evaluations.

        Returns:
            dict with keys:
                "P_Pa"      : {node_name: pressure [Pa]} for every node.
                "Q_m3s"     : {edge_name: signed volumetric flow [m^3/s]}
                              for every edge.  Positive = nominal direction.
                "Q_ext_m3s" : {node_name: external supply [m^3/s]} for every
                              node.  Recovered from mass balance at
                              P-spec nodes; equals the spec at Q-spec
                              nodes; equals 0 at interior nodes.
                "converged" : bool, True if fsolve reported success.
        """
        node_names = list(self._node_order)
        n_nodes    = len(node_names)
        n_edges    = len(self._edges)

        if n_nodes == 0 or n_edges == 0:
            raise ValueError("Network.solve: network has no nodes or edges.")

        # Index nodes.
        idx_of = {name: i for i, name in enumerate(node_names)}

        # Classify nodes by which boundary condition (if any) is specified.
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

        # Build node-edge adjacency.  For each node we record (edge_index,
        # sign), where sign=+1 means positive Q_e flows INTO this node and
        # sign=-1 means positive Q_e flows OUT of this node.
        adj = [[] for _ in range(n_nodes)]
        for e_idx, edge in enumerate(self._edges):
            adj[idx_of[edge.from_node]].append((e_idx, -1))
            adj[idx_of[edge.to_node]  ].append((e_idx, +1))

        # Default pressure guess.
        P_specs = [self._nodes[node_names[i]].P_spec_Pa for i in p_spec_indices]
        if P_init is None:
            P_init = sum(P_specs) / len(P_specs)

        n_free = len(free_indices)
        x0 = np.concatenate([
            np.full(n_free, float(P_init)),
            np.full(n_edges, float(Q_init_si)),
        ])

        # Lazily-filled cache of reversed shadow components, keyed by id() of
        # the original component.  One entry per asymmetric component, built
        # the first time the solver evaluates that component under reverse
        # flow; symmetric components never appear here.
        reversed_cache = {}

        def unpack(x):
            """Expand the solver vector into (P_all_nodes, Q_edges)."""
            P_free = x[:n_free]
            Q_e    = x[n_free:]
            P = np.empty(n_nodes)
            for i in p_spec_indices:
                P[i] = self._nodes[node_names[i]].P_spec_Pa
            for j, i in enumerate(free_indices):
                P[i] = P_free[j]
            return P, Q_e

        def residuals(x):
            P, Q_e = unpack(x)
            res = np.empty(n_free + n_edges)

            # Mass balance at each free (non-P-spec) node.
            for j, i in enumerate(free_indices):
                node = self._nodes[node_names[i]]
                Q_ext = node.Q_ext_spec_si if node.Q_ext_spec_si is not None else 0.0
                net_in = 0.0
                for e_idx, sign in adj[i]:
                    net_in += sign * Q_e[e_idx]
                # Steady-state: edges + external supply must net to zero.
                res[j] = net_in + Q_ext

            # Pipe equation on each edge.
            for e_idx, edge in enumerate(self._edges):
                i_from = idx_of[edge.from_node]
                i_to   = idx_of[edge.to_node]
                dP_io = 0.0
                for c in edge.components:
                    dP_io += _component_signed_dP(
                        c, fluid, Q_e[e_idx], reversed_cache
                    )
                # P_from - P_to = -dP_inlet_to_outlet
                res[n_free + e_idx] = (P[i_from] - P[i_to]) + dP_io

            return res

        sol, info, ier, msg = fsolve(
            residuals, x0, full_output=True, xtol=xtol, maxfev=maxfev,
        )
        # fsolve sometimes reports ier=5 ("not making good progress") when the
        # residual has already hit floating-point noise and there is nothing
        # left to improve.  Cross-check against the residual norm.  Equation
        # magnitudes here are pressures (~1e5 Pa) and flows (~1e-3 m^3/s), so
        # a 1e-4 absolute threshold covers both with margin.
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
                f"residual norm = {np.linalg.norm(info['fvec']):.3e}"
            )

        P_arr, Q_arr = unpack(sol)

        # Recover Q_ext at P-spec nodes from mass balance; copy spec or 0 at
        # the others.
        Q_ext_out = {}
        for i, name in enumerate(node_names):
            node = self._nodes[name]
            if node.P_spec_Pa is not None:
                net_in = 0.0
                for e_idx, sign in adj[i]:
                    net_in += sign * Q_arr[e_idx]
                # mass balance: net_in + Q_ext = 0  =>  Q_ext = -net_in
                Q_ext_out[name] = float(-net_in)
            else:
                Q_ext_out[name] = float(node.Q_ext_spec_si or 0.0)

        return {
            "P_Pa":      {name: float(P_arr[i]) for i, name in enumerate(node_names)},
            "Q_m3s":     {self._edges[e].name: float(Q_arr[e]) for e in range(n_edges)},
            "Q_ext_m3s": Q_ext_out,
            "converged": bool(converged),
        }


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


def _build_screenshot_network(p_outlet_15_psi, p_outlet_16_psi):
    """Build the DWSIM-screenshot topology with configurable outlet pressures.

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
    net.add_node("1",  P=ureg.Quantity(200.0, "psi"))
    net.add_node("5",  P=ureg.Quantity(180.0, "psi"))
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
    net = _build_screenshot_network(p_outlet_15_psi=40.0, p_outlet_16_psi=80.0)
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
    net = _build_screenshot_network(p_outlet_15_psi=175.0, p_outlet_16_psi=40.0)
    result = net.solve(fluid, verbose=True)
    _print_screenshot_result("screenshot-REV", result)
    assert result["Q_m3s"]["PIPE-5"] < 0.0, "expected PIPE-5 reversed"


if __name__ == "__main__":
    _test_single_segment()
    print()
    _test_parallel_two_branches()
    print()
    _test_screenshot_forward_PIPE5()
    print()
    _test_screenshot_reverse_PIPE5()
