"""
This benchmark is used when optimizing the various solvers used by dP_dT, dmdot_dT, and the network solver.
Benchmark the cost of dmdot_dT vs dP_dT at three levels per component:

  1. Forward dP_dT call (baseline cost of one walk).
  2. Component-level dmdot_dT call (brentq + forward closure).
  3. Single-edge network solve with auto-detected inverse mode
     (`Compressible_Network.solve` calling `walk_edge_inverse`).

For each level we record wall-clock time and the number of times the
component's dP_dT was actually invoked (instrumented via a monkeypatch).



Run: `python benchmark_dmdot.py`
"""

import math
import time
import os
import warnings

import composition
from component_classes import ureg
from compressible_flow import (
    Bend, Valve, Orifice, Line_Segment,
    FlowState, _build_phase_limits, _safe_update_PT, _resolve_mdot,
)
from compressible_network import Compressible_Network


def _build_AS():
    csv_path = os.path.join(os.path.dirname(__file__), "example_composition.csv")
    return composition.define_composition_from_csv(csv_path)


def _make_fs(AS, P, T, mdot, A, phase_limits):
    _safe_update_PT(AS, P, T, *phase_limits)
    return FlowState(
        AS, mdot, A=A, z=0.0,
        T_cricondentherm=phase_limits[0], P_cricondenbar=phase_limits[1],
        T_critical=phase_limits[2], P_critical=phase_limits[3],
    )


class _DPCounter:
    """Monkeypatch wrapper that counts dP_dT calls on a component instance."""
    def __init__(self, component):
        self.component = component
        self.calls = 0
        self._original = component.dP_dT
        component.dP_dT = self._wrapped

    def _wrapped(self, fs, *args, **kwargs):
        self.calls += 1
        return self._original(fs, *args, **kwargs)

    def restore(self):
        self.component.dP_dT = self._original


def benchmark_component(name, component, AS, P_in, T_in, mdot_target,
                        A_pipe, phase_limits, P_target,
                        component_class, di_for_node=None,
                        dmdot_dT_kwargs=None):
    """Run the three timing levels for one component type."""
    dmdot_dT_kwargs = dmdot_dT_kwargs or {}

    # Level 1: forward dP_dT.
    counter = _DPCounter(component)
    counter.calls = 0
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    t0 = time.perf_counter()
    component.dP_dT(fs, **dmdot_dT_kwargs)
    dt_fwd = time.perf_counter() - t0
    fwd_calls = counter.calls
    counter.restore()

    # Level 2: component-level dmdot_dT (brentq inside).
    counter = _DPCounter(component)
    counter.calls = 0
    fs = _make_fs(AS, P_in, T_in, 0.0, A_pipe, phase_limits)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        component.dmdot_dT(fs, P2=P_target, **dmdot_dT_kwargs)
    dt_dmdot = time.perf_counter() - t0
    dmdot_calls = counter.calls
    dmdot_mdot = fs.mdot
    counter.restore()

    # Level 3: 1-edge Compressible_Network.solve with inverse mode
    # (auto-detected because both endpoints are P-spec).
    net = Compressible_Network()
    # If the component carries a non-trivial diameter, use it as the
    # node area too (saves the area-mismatch _area_match call).
    if di_for_node is not None:
        net.add_node("inlet",  P=P_in, T=T_in, diameter=di_for_node)
        net.add_node("outlet", P=P_target,    diameter=di_for_node)
    else:
        net.add_node("inlet",  P=P_in, T=T_in)
        net.add_node("outlet", P=P_target)
    net.add_edge("edge", "inlet", "outlet", component)

    counter = _DPCounter(component)
    counter.calls = 0
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = net.solve(AS, mdot_init_kgs=max(mdot_target * 0.5, 0.1))
    dt_net = time.perf_counter() - t0
    net_calls = counter.calls
    net_mdot = result["mdot_kgs"]["edge"]
    net_converged = result["converged"]
    counter.restore()

    # Per-call cost: time per dP_dT invocation, for cross-component comparison.
    dt_per_fwd = (dt_fwd / fwd_calls) if fwd_calls else float("nan")

    return {
        "name": name,
        "dt_fwd_s":   dt_fwd,
        "dt_dmdot_s": dt_dmdot,
        "dt_net_s":   dt_net,
        "fwd_calls": fwd_calls,
        "dmdot_calls": dmdot_calls,
        "net_calls":   net_calls,
        "dt_per_fwd_s": dt_per_fwd,
        "dmdot_mdot":  dmdot_mdot,
        "net_mdot":    net_mdot,
        "mdot_target": mdot_target,
        "net_converged": net_converged,
    }


def main():
    AS = _build_AS()
    phase_limits = _build_phase_limits(AS)

    P_in = ureg.Quantity(100, "psi").to("Pa").magnitude
    T_in = ureg.Quantity(150, "degF").to("degK").magnitude

    ID    = ureg.Quantity(1.939, "inch")
    ID_si = ID.to("m").magnitude
    A_pipe = math.pi * ID_si ** 2 / 4.0
    D_trim = ureg.Quantity(0.25, "inch")

    _safe_update_PT(AS, P_in, T_in, *phase_limits)
    mdot_target = _resolve_mdot(ureg.Quantity(0.07, "mmscf/day"), AS)

    # For each component, first run forward to get the target P2, then
    # benchmark all three levels with that P2 as the target.
    cases = []

    # 1. Constricted Valve (forwards to compressible_dA Mode 2)
    v_throat = Valve(Di=ID, Cv=1.76, minimum_diameter=D_trim)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    v_throat.dP_dT(fs)
    P2 = fs.P
    cases.append(("Valve constricted", v_throat, P2, ID, None))

    # 2. Plain Valve (drives compressible_changing_area_K via helper)
    v_plain = Valve(Di=ID, K=10.0)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    v_plain.dP_dT(fs)
    P2 = fs.P
    cases.append(("Valve plain", v_plain, P2, ID, None))

    # 3. Orifice (compressible_dA + Cd<->mdot fixed point)
    orf = Orifice(Di=ID, Do=D_trim)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    orf.dP_dT(fs)
    P2 = fs.P
    cases.append(("Orifice", orf, P2, ID, None))

    # 4. Bend (compressible_K + K<->mdot fixed point)
    bend = Bend(Di=ID, ang_deg=90.0, bend_dias=3.0)
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    bend.dP_dT(fs)
    P2 = fs.P
    cases.append(("Bend", bend, P2, ID, None))

    # 5. Line_Segment 200 ft 2" (slice loop inside)
    seg = Line_Segment(
        roughness=ureg.Quantity(0.00015, "ft"),
        id_val=ID,
        length=ureg.Quantity(200.0, "ft"),
        elevation_change=ureg.Quantity(0.0, "ft"),
        name="seg",
    )
    fs = _make_fs(AS, P_in, T_in, mdot_target, A_pipe, phase_limits)
    seg.dP_dT(fs)
    P2 = fs.P
    cases.append(("Line_Segment 200ft", seg, P2, ID, None))

    rows = []
    for name, component, P2, di, kwargs in cases:
        print(f"\nBenchmarking {name} ...", flush=True)
        row = benchmark_component(
            name=name,
            component=component,
            AS=AS, P_in=P_in, T_in=T_in,
            mdot_target=mdot_target,
            A_pipe=A_pipe,
            phase_limits=phase_limits,
            P_target=P2,
            component_class=type(component),
            di_for_node=di,
        )
        rows.append(row)
        print(
            f"  fwd dP_dT      : {row['dt_fwd_s']*1000:8.2f} ms "
            f"({row['fwd_calls']:3d} dP_dT call{'s' if row['fwd_calls']!=1 else ''})",
            flush=True,
        )
        print(
            f"  dmdot_dT       : {row['dt_dmdot_s']*1000:8.2f} ms "
            f"({row['dmdot_calls']:3d} dP_dT calls) "
            f"-> mdot {row['dmdot_mdot']:.6g}  (target {row['mdot_target']:.6g})",
            flush=True,
        )
        print(
            f"  network solve  : {row['dt_net_s']*1000:8.2f} ms "
            f"({row['net_calls']:3d} dP_dT calls) "
            f"-> mdot {row['net_mdot']:.6g}  converged={row['net_converged']}",
            flush=True,
        )
        # Network amplification factor: dP_dT calls in network solve vs
        # in component-level dmdot_dT.  Tells us how much overhead LM +
        # FD adds on top of a single brentq.
        if row["dmdot_calls"]:
            amp = row["net_calls"] / row["dmdot_calls"]
            print(f"  amplification  : net/dmdot = {amp:.1f}x", flush=True)

    print("\n=== Summary ===")
    print(f"{'component':22s} {'fwd ms':>8s} {'dmdot ms':>10s} {'net ms':>10s} "
          f"{'dP_dT/dmdot':>12s} {'dP_dT/net':>10s} {'us/dP_dT':>10s}")
    for r in rows:
        print(
            f"{r['name']:22s} "
            f"{r['dt_fwd_s']*1000:8.2f} "
            f"{r['dt_dmdot_s']*1000:10.2f} "
            f"{r['dt_net_s']*1000:10.2f} "
            f"{r['dmdot_calls']:12d} "
            f"{r['net_calls']:10d} "
            f"{r['dt_per_fwd_s']*1e6:10.1f}"
        )


if __name__ == "__main__":
    main()
