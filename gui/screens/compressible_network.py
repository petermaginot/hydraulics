"""Compressible-network screen.

Subclasses the incompressible NetworkScreen and overrides:
  - LINE_SEGMENT_CLS / NETWORK_CLS -- compressible_flow.Line_Segment and
    Compressible_Network instead of incompressible.Line_Segment and Network.
  - The side-panel "fluid" box -- shows the composition / EOS / mode
    defined on the CompressibleCompositionScreen instead of a density /
    viscosity form.
  - Source/Sink editor -- adds a Temperature row.  Per the user's spec
    every Source/Sink carries a T spec even when it is acting as a
    withdrawal, so a flow reversal cannot strand a node without enough
    information to supply.
  - Result rendering -- the compressible solver returns a flat dict
    (not a NetworkResult) so per-component dP is not available; the
    canvas shows P and Q per Source/Sink, P and T per Junction (T is
    pinned at supply nodes and back-solved at the others), and edge
    mdot is duplicated onto every inline node in the chain.

The composition AbstractState lives on AppState.fluid; the network
screen reads it at solve time.  Per-edge mdot_init defaults to the
solver's scalar (1 kg/s) -- per network.md this can stall on junction
networks; a per-edge override is left for a later iteration.
"""

import gui._compat   # noqa: F401  -- must import before NodeGraphQt

import math
import time

import CoolProp.CoolProp as CP

from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import compressible_flow
from compressible_network import Compressible_Network
from component_classes import ureg
from gui import units as U
from gui.screens.network import (
    NetworkScreen,
    _fmt_pressure,
    _fmt_pressure_signed,
    _fmt_temperature,
    _fmt_velocity_at_D,
    _fmt_velocity,
)
from gui.screens.segment import _LabeledField


_TEMP_DIM = ureg.Quantity(1.0, "K").dimensionality


class CompressibleNetworkScreen(NetworkScreen):

    # Swap the regime-specific component classes and solver.  Everything
    # else (toolbar, canvas, editor stack, CSV loader, chain walker) is
    # inherited.  The base class's _build_*_component helpers reach for
    # these via self, so substituting them here is all that's needed.
    LINE_SEGMENT_CLS     = compressible_flow.Line_Segment
    BEND_CLS             = compressible_flow.Bend
    VALVE_CLS            = compressible_flow.Valve
    CHECKVALVE_CLS       = compressible_flow.CheckValve
    CONTRACTION_EXP_CLS  = compressible_flow.Contraction_Expansion
    NETWORK_CLS          = Compressible_Network
    DISPLAY_FLOW_UNITS   = [
        # Compressible network results are reported in mass [kg/s];
        # actual-volumetric units are omitted because density varies
        # along the network and no single conversion exists.
        "kg/s", "kg/h", "lb/h",
        "mol/s", "mol/h",
        "mmscf/day", "mscf/day", "scf/min", "scm/h",
    ]
    DISPLAY_FLOW_DEFAULT = "mmscf/day"
    FLUID_BOX_TITLE      = "Composition (compressible)"

    # ------------------------------------------------------------------
    # Fluid panel: shows the AS summary stashed by the composition screen.
    # ------------------------------------------------------------------

    def _build_fluid_box(self):
        self._composition_label = QLabel("(no composition defined)")
        self._composition_label.setWordWrap(True)
        self._composition_label.setStyleSheet("color: #444;")
        v = QVBoxLayout()
        v.addWidget(self._composition_label)
        self.fluid_box = QGroupBox(self.FLUID_BOX_TITLE)
        self.fluid_box.setLayout(v)

    def showEvent(self, event):
        # Refresh the composition summary every time the user lands here,
        # so a "Back" + redefine on the composition screen is reflected.
        self._refresh_composition_label()
        super().showEvent(event)

    def _refresh_composition_label(self):
        AS = getattr(self.state, "fluid", None)
        if AS is None:
            self._composition_label.setText("(no composition defined)")
            return
        try:
            backend = AS.backend_name()
        except Exception:
            backend = "?"
        try:
            n  = AS.get_mole_fractions()
            cs = [c.replace("-", "_") for c in AS.fluid_names()]
            parts = [f"{name}={frac:.3g}" for name, frac in zip(cs, n)]
            summary = ", ".join(parts)
        except Exception:
            summary = "(could not enumerate components)"
        mode = "isothermal" if getattr(self.state, "isothermal", False) \
                            else "adiabatic"
        self._composition_label.setText(
            f"EOS: {backend}\nMode: {mode}\n{summary}"
        )

    def _build_fluid(self):
        AS = getattr(self.state, "fluid", None)
        if AS is None:
            raise ValueError(
                "No composition defined.  Go back to the composition "
                "screen and press Build."
            )
        return AS

    # ------------------------------------------------------------------
    # Source/Sink editor: add a T row.
    # ------------------------------------------------------------------

    def _add_extra_source_sink_rows(self, form):
        self.ss_T = _LabeledField("", U.TEMPERATURE, "degF")
        self.ss_T.edit.setToolTip(
            "Temperature spec.  Required at every Source/Sink: even on a "
            "node currently acting as a sink, a future solve that "
            "reverses the flow will turn it into a supply, and the "
            "energy balance needs an inflow T to evaluate."
        )
        form.addRow("Temperature:", self.ss_T.widget())

    def _default_source_sink_extra(self):
        return {"T_str": "", "T_unit": "degF"}

    def _sync_source_sink_extra(self, spec):
        self.ss_T.edit.setText(spec.get("T_str", ""))
        self.ss_T.combo.setCurrentText(spec.get("T_unit", "degF"))

    def _apply_source_sink_extra(self, spec):
        spec["T_str"]  = self.ss_T.edit.text().strip()
        spec["T_unit"] = self.ss_T.combo.currentText()

    # ------------------------------------------------------------------
    # Display-units: combos live on the composition screen.
    # main.py assigns d_pressure / d_flow / d_temperature to this instance
    # after construction and connects their signals.
    # ------------------------------------------------------------------

    def _build_display_units_box(self):
        # Provide the attribute the base-class side-panel wiring expects,
        # but keep it invisible so it takes no layout space.
        self.display_box = QWidget()
        self.display_box.setVisible(False)

    def _extra_kwargs_for_boundary(self, spec, node_name):
        if spec.get("type") != "source_sink":
            return {}
        T_str = spec.get("T_str", "").strip()
        if not T_str:
            raise ValueError(
                f"Source/Sink '{node_name}': temperature is required on "
                f"every Source/Sink (a flow reversal could turn this "
                f"node into a supply)."
            )
        T_unit = spec.get("T_unit", "degF")
        return {"T": ureg.Quantity(float(T_str), U.to_pint(T_unit))}

    # ------------------------------------------------------------------
    # Solve: pump progress updates into the Results text panel while
    # Compressible_Network.solve() runs.  scipy.optimize.least_squares
    # has no native progress hook, so the solver counts residual
    # evaluations and calls back here every time; we throttle to
    # _PROGRESS_INTERVAL_S to avoid drowning the event loop.
    # ------------------------------------------------------------------

    _PROGRESS_INTERVAL_S = 0.15

    def _solve_network(self, net, fluid):
        self.results_text.setPlainText("Solving...")
        QApplication.processEvents()
        self._solve_t0           = time.monotonic()
        self._solve_last_update  = 0.0
        return net.solve(fluid, progress_callback=self._on_solve_progress)

    def _on_solve_progress(self, nfev, residual_norm):
        now = time.monotonic()
        # Always show the first eval; throttle the rest.
        if nfev > 1 and (now - self._solve_last_update) < self._PROGRESS_INTERVAL_S:
            return
        self._solve_last_update = now
        elapsed = now - self._solve_t0
        self.results_text.setPlainText(
            f"Solving... Iteration number {nfev}, "
            f"residual={residual_norm:.3e}, "
            f"elapsed={elapsed:.1f}s"
        )
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # Result rendering.  The compressible solver returns a flat dict, not
    # a NetworkResult, so per-component dP is unavailable in v1; we surface
    # P + T per node and edge mdot duplicated onto each inline block.
    # ------------------------------------------------------------------

    def _render_results(self, net, result):
        self._last_net    = net
        self._last_result = result
        self._render_canvas_and_panel(net, result)
        self._save_results_btn.setEnabled(True)

    def _render_canvas_and_panel(self, net, result):
        self._annotate_results_on_canvas(result)
        self._render_results_text(result)

    def _annotate_results_on_canvas(self, result):
        P_unit = self.d_pressure.currentText()
        T_unit = self.d_temperature.currentText()
        Q_unit = self.d_flow.currentText()

        P_dict        = result["P_Pa"]
        T_dict        = result["T_K"]
        mdot_dict     = result["mdot_kgs"]
        ext_mdot_dict = result["Q_ext_mdot_kgs"]
        # Per-component outlet (P_Pa, T_K) in flow direction, indexed by
        # the original edge.components order.  See Compressible_Network.solve.
        comp_PT_dict  = result.get("component_outlet_PT", {})

        for node in self.graph.all_nodes():
            spec = self.node_specs.get(node.id, {})
            t = spec.get("type")
            if t == "source_sink":
                name = node.name()
                P_Pa = P_dict.get(name)
                T_K  = T_dict.get(name)
                mass = ext_mdot_dict.get(name)
                self._set_widget(node, "P_result",
                                 self._format_pressure(P_Pa, P_unit))
                self._set_widget(node, "T_result",
                                 self._format_temperature(T_K, T_unit))
                self._set_widget(node, "Q_result",
                                 self._format_flow_mass(mass, Q_unit))
            elif t == "junction":
                name = node.name()
                P_Pa = P_dict.get(name)
                T_K  = T_dict.get(name)
                self._set_widget(node, "P_result",
                                 self._format_pressure(P_Pa, P_unit))
                self._set_widget(node, "T_result",
                                 self._format_temperature(T_K, T_unit))
            elif t in ("pipe", "fitting", "valve", "check_valve"):
                edge_name = self._pipe_edge_names.get(node.id)
                pos       = self._inline_chain_pos.get(node.id)
                mass      = mdot_dict.get(edge_name) if edge_name else None
                P_Pa = T_K = None
                if edge_name is not None and pos is not None:
                    pt_list = comp_PT_dict.get(edge_name, [])
                    if 0 <= pos < len(pt_list):
                        P_Pa, T_K = pt_list[pos]
                # dP_result holds the outlet pressure on the compressible
                # screen (the field is reused rather than renamed to keep
                # serialized graphs portable).
                self._set_widget(node, "dP_result",
                                 self._format_pressure(P_Pa, P_unit))
                self._set_widget(node, "T_result",
                                 self._format_temperature(T_K, T_unit))
                self._set_widget(node, "Q_result",
                                 self._format_flow_mass(mass, Q_unit))

    def _render_results_text(self, result):
        P_unit = self.d_pressure.currentText()
        T_unit = self.d_temperature.currentText()
        Q_unit = self.d_flow.currentText()

        lines = [f"Converged: {result['converged']}", ""]

        lines.append(f"Node pressures ({P_unit}):")
        for name, P_Pa in result["P_Pa"].items():
            P_val = ureg.Quantity(P_Pa, "Pa").to(U.to_pint(P_unit)).magnitude
            lines.append(f"  {name:<16} {P_val:>12.3f}")
        lines.append("")

        lines.append(f"Node temperatures ({T_unit}):")
        for name, T_K in result["T_K"].items():
            T_val = ureg.Quantity(T_K, "K").to(U.to_pint(T_unit)).magnitude
            lines.append(f"  {name:<16} {T_val:>12.3f}")
        lines.append("")

        lines.append(f"Edge mass flow ({Q_unit}, sign per nominal direction):")
        for edge_name, mass in result["mdot_kgs"].items():
            val = self._convert_mass_flow(mass, Q_unit)
            arrow = "->" if val >= 0 else "<-"
            lines.append(f"  {edge_name:<24} {arrow} {val:>+14.4f}")
        lines.append("")

        lines.append(f"External flow at boundary nodes ({Q_unit}):")
        for name, mass in result["Q_ext_mdot_kgs"].items():
            val = self._convert_mass_flow(mass, Q_unit)
            if val == 0:
                continue
            lines.append(f"  {name:<16} {val:>+14.4f}")

        self.results_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Compressible flow formatting helpers.  Mass-only because the
    # network has no single density to convert to actual volumetric;
    # mol/std-vol conversions go through CoolProp via pint's universal
    # context, anchored by the AbstractState's molar mass.
    # ------------------------------------------------------------------

    def _format_pressure(self, P_Pa, unit):
        if P_Pa is None:
            return ""
        val = ureg.Quantity(P_Pa, "Pa").to(U.to_pint(unit)).magnitude
        return f"P={val:.1f} {unit}"

    def _format_temperature(self, T_K, unit):
        if T_K is None:
            return ""
        val = ureg.Quantity(T_K, "K").to(U.to_pint(unit)).magnitude
        return f"T={val:.1f} {unit}"

    def _format_flow_mass(self, mass_kgs, unit):
        if mass_kgs is None:
            return ""
        val = self._convert_mass_flow(mass_kgs, unit)
        return f"Q={val:+.3f} {unit}"

    # ------------------------------------------------------------------
    # Per-node solved-details inspector overrides.
    #
    # The compressible solver returns a flat dict (no NetworkResult.
    # component()), and per-block (P, T) is sliced out of
    # result["component_outlet_PT"] using the chain-position bookkeeping
    # the base class already maintains.  For inlet (P, T) we walk one
    # step further: the predecessor in flow direction.
    #
    # Velocities and the pipe P/T profile both require an
    # AbstractState anchored at the inlet, so we save / restore
    # self.state.fluid's (P, T) around the calls.
    # ------------------------------------------------------------------

    def _have_result_for_inline(self, node):
        # Compressible: a node has a result if it ended up in the chain
        # position map AND the edge has component_outlet_PT for it.
        if self._last_result is None:
            return False
        edge_name = self._pipe_edge_names.get(node.id)
        pos       = self._inline_chain_pos.get(node.id)
        if edge_name is None or pos is None:
            return False
        pt_list = self._last_result.get("component_outlet_PT", {}).get(edge_name, [])
        return 0 <= pos < len(pt_list)

    def _inline_inlet_outlet_PT(self, node_or_id):
        """Return (P_in_Pa, T_in_K, P_out_Pa, T_out_K, mdot_signed) for an
        inline node, in flow direction.  Accepts a NodeGraphQt node or
        a node id string."""
        node_id   = node_or_id if isinstance(node_or_id, str) else node_or_id.id
        edge_name = self._pipe_edge_names[node_id]
        pos       = self._inline_chain_pos[node_id]
        pt_list   = self._last_result["component_outlet_PT"][edge_name]
        mdot      = self._last_result["mdot_kgs"][edge_name]
        edge      = self._find_edge_by_name(edge_name)

        # component_outlet_PT is always in the edge's ORIGINAL components
        # order, but each entry is the flow-direction outlet of that
        # original-position component (solver reverses-back if flow ran
        # backwards).  So pt_list[pos] is the flow-direction outlet of
        # the pos-th original component; the flow-direction predecessor
        # depends on the sign of mdot.
        P_out, T_out = pt_list[pos]
        if mdot >= 0.0:
            if pos == 0:
                P_in = self._last_result["P_Pa"][edge.from_node]
                T_in = self._last_result["T_K"][edge.from_node]
            else:
                P_in, T_in = pt_list[pos - 1]
        else:
            if pos == len(pt_list) - 1:
                P_in = self._last_result["P_Pa"][edge.to_node]
                T_in = self._last_result["T_K"][edge.to_node]
            else:
                P_in, T_in = pt_list[pos + 1]
        return P_in, T_in, P_out, T_out, mdot

    def _find_edge_by_name(self, edge_name):
        for e in self._last_net._edges:
            if e.name == edge_name:
                return e
        raise KeyError(f"edge {edge_name!r} not on last solved network")

    def _density_at(self, P_Pa, T_K):
        """Snapshot density at (P, T) without leaving AS perturbed."""
        AS = self.state.fluid
        P_save, T_save = AS.p(), AS.T()
        try:
            AS.update(CP.PT_INPUTS, P_Pa, T_K)
            return AS.rhomass()
        finally:
            AS.update(CP.PT_INPUTS, P_save, T_save)

    def _pipe_details(self, node):
        P_in, T_in, P_out, T_out, mdot = self._inline_inlet_outlet_PT(node)
        comp = self._pipe_components[node.id]
        # Velocity at inlet vs outlet: same mass flow, different density
        # AND (for the comp.dP_dT-walked profile) different cross-section
        # if the segment has varying ID.  For inlet/outlet we use the
        # first/last profile area, matching what dP_dT itself uses.
        A_in  = comp.profile[0][3]
        A_out = comp.profile[-1][3]
        rho_in  = self._density_at(P_in,  T_in)
        rho_out = self._density_at(P_out, T_out)
        v_in  = abs(mdot) / (rho_in  * A_in)
        v_out = abs(mdot) / (rho_out * A_out)

        P_unit = self.d_pressure.currentText()
        T_unit = self.d_temperature.currentText()
        Q_unit = self.d_flow.currentText()
        v_unit = "ft/s"

        rows = [
            ("Mass flow:",   f"{self._convert_mass_flow(abs(mdot), Q_unit):+.4g} {Q_unit}"),
            ("Inlet P:",     _fmt_pressure(P_in,  P_unit)),
            ("Outlet P:",    _fmt_pressure(P_out, P_unit)),
            ("dP:",          _fmt_pressure_signed(P_out - P_in, P_unit)),
            ("Inlet T:",     _fmt_temperature(T_in,  T_unit)),
            ("Outlet T:",    _fmt_temperature(T_out, T_unit)),
            ("Inlet vel.:",  _fmt_velocity(v_in,  v_unit)),
            ("Outlet vel.:", _fmt_velocity(v_out, v_unit)),
        ]
        # The profile is computed lazily inside the closure attached to
        # the dialog's plot button, so a user who never clicks Plot pays
        # nothing for it.  Pass a marker (True) so the base class wires
        # up the button; the actual generator lives on
        # _build_compressible_pipe_profile.
        return rows, ("compressible", node.id)

    def _open_pipe_profile_window(self, node_name, profile_points):
        # Override the base class's behavior: profile_points is actually
        # ("compressible", node_id) -- we build the real profile here so
        # the (potentially slow) re-solve happens at button-click time,
        # not when the details dialog opens.
        marker, node_id = profile_points
        if marker != "compressible":
            return super()._open_pipe_profile_window(node_name, profile_points)
        try:
            real_profile = self._build_compressible_pipe_profile(node_id)
        except Exception as exc:
            import gui.dialogs as dialogs
            import traceback
            dialogs.critical(
                self, "Profile generation failed",
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            )
            return
        super()._open_pipe_profile_window(node_name, real_profile)

    def _build_compressible_pipe_profile(self, node_id):
        """Re-run compressible Line_Segment.dP_dT on the solved inlet
        (P, T) and return the per-point profile in the dict-of-dicts shape
        PipeProfileWindow consumes.

        Mutates self.state.fluid in place; restored on exit.
        """
        comp = self._pipe_components[node_id]
        P_in, T_in, _P_out, _T_out, mdot = self._inline_inlet_outlet_PT(node_id)
        AS = self.state.fluid
        P_save, T_save = AS.p(), AS.T()
        try:
            AS.update(CP.PT_INPUTS, P_in, T_in)
            raw = comp.dP_dT(
                abstract_state = AS,
                flow_rate      = ureg.Quantity(abs(mdot), "kg/s"),
                isothermal     = bool(getattr(self.state, "isothermal", False)),
            )
        finally:
            AS.update(CP.PT_INPUTS, P_save, T_save)
        return [
            {"distance_m": d, "P_Pa": P, "T_K": T, "v_ms": v}
            for (d, P, T, v) in raw
        ]

    def _fitting_details(self, node):
        P_in, T_in, P_out, T_out, mdot = self._inline_inlet_outlet_PT(node)
        comp = self._pipe_components[node.id]
        D = (min(comp.Di_US_si, comp.Di_DS_si)
             if hasattr(comp, "Di_US_si") else comp.Di_si)
        # Use inlet density for the velocity reading at the smallest D
        # (the Crane K-factor correlations are defined relative to the
        # inlet stream).
        rho_in = self._density_at(P_in, T_in)
        v = abs(mdot) / (rho_in * math.pi * D * D / 4.0)

        P_unit = self.d_pressure.currentText()
        T_unit = self.d_temperature.currentText()
        Q_unit = self.d_flow.currentText()
        rows = [
            ("Mass flow:", f"{self._convert_mass_flow(abs(mdot), Q_unit):+.4g} {Q_unit}"),
            ("Inlet P:",   _fmt_pressure(P_in,  P_unit)),
            ("Outlet P:",  _fmt_pressure(P_out, P_unit)),
            ("dP:",        _fmt_pressure_signed(P_out - P_in, P_unit)),
            ("Inlet T:",   _fmt_temperature(T_in,  T_unit)),
            ("Outlet T:",  _fmt_temperature(T_out, T_unit)),
            ("Velocity:",  _fmt_velocity_at_D(v, D)),
        ]
        return rows

    def _valve_details(self, node):
        P_in, T_in, P_out, T_out, mdot = self._inline_inlet_outlet_PT(node)
        comp = self._pipe_components[node.id]
        D = self._valve_smallest_D_si(node)
        rho_in = self._density_at(P_in, T_in)
        v = abs(mdot) / (rho_in * math.pi * D * D / 4.0)

        P_unit = self.d_pressure.currentText()
        T_unit = self.d_temperature.currentText()
        Q_unit = self.d_flow.currentText()
        rows = [
            ("K-factor:",  f"{comp.K:.3f}"),
            ("Mass flow:", f"{self._convert_mass_flow(abs(mdot), Q_unit):+.4g} {Q_unit}"),
            ("Inlet P:",   _fmt_pressure(P_in,  P_unit)),
            ("Outlet P:",  _fmt_pressure(P_out, P_unit)),
            ("dP:",        _fmt_pressure_signed(P_out - P_in, P_unit)),
            ("Inlet T:",   _fmt_temperature(T_in,  T_unit)),
            ("Outlet T:",  _fmt_temperature(T_out, T_unit)),
            ("Velocity:",  _fmt_velocity_at_D(v, D)),
        ]
        return rows

    def _convert_mass_flow(self, mass_kgs, unit):
        """Convert a mass flow [kg/s] to the chosen display unit.

        Mass / molar / standard-volume units are all supported; the
        molar mass needed for the molar and std-vol conversions comes
        from the AbstractState that the user defined on the composition
        screen.
        """
        if mass_kgs is None:
            return 0.0
        pint_unit = U.to_pint(unit)
        target = ureg.Quantity(1.0, pint_unit)
        # Mass [M/T]: convert directly.
        if target.dimensionality == ureg.Quantity(1.0, "kg/s").dimensionality:
            return ureg.Quantity(mass_kgs, "kg/s").to(pint_unit).magnitude
        # Molar [N/T] or std-vol [L^3/T at standard conditions]: need MW
        # from the AbstractState.
        AS = self.state.fluid
        if AS is None:
            return 0.0
        mm_kg_per_mol = AS.molar_mass()
        mol_s = mass_kgs / mm_kg_per_mol
        if target.dimensionality == ureg.Quantity(1.0, "mol/s").dimensionality:
            return ureg.Quantity(mol_s, "mol/s").to(pint_unit).magnitude
        # Otherwise assume the unit is a standard-volume flow.  pint maps
        # mmscf/day / scf/min / scm/h to molar amounts via the same
        # universal context used in composition.py at solver build time.
        try:
            return ureg.Quantity(mol_s, "mol/s").to(pint_unit).magnitude
        except Exception:
            return 0.0
