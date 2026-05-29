"""Results screen: run the pressure-profile calculation and plot it."""

import traceback

import CoolProp.CoolProp as CP
import compressible_flow
import gui.dialogs as dialogs
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from component_classes import ureg
from gui import units as U


def _format_solved_mdot(mdot_kgs, display_unit, fluid):
    """Convert a mass flow rate [kg/s] into the user's display unit.

    Mass units (kg/s, lb/h, ...) are direct conversions.  Molar and
    standard-volume units (mol/s, mmscf/day, ...) are defined as mole
    equivalents in the unit registry (see component_classes.py), so they
    convert via molar mass.  Actual-volume units (m^3/s, gal/min, ...)
    convert via in-situ density.

    Args:
        mdot_kgs     : float, mass flow rate [kg/s].
        display_unit : str, the GUI dropdown label (may need to_pint
                       translation for engineering aliases like BBL/D).
        fluid        : Incompressible_Fluid or CoolProp AbstractState
                       (anchored at the inlet (P, T)).

    Returns:
        pint Quantity in the requested unit.
    """
    target = U.to_pint(display_unit)
    dim    = ureg.Quantity(1.0, target).dimensionality

    if dim == {"[mass]": 1, "[time]": -1}:
        return ureg.Quantity(mdot_kgs, "kg/s").to(target)

    if dim == {"[substance]": 1, "[time]": -1}:
        # AbstractState exposes molar_mass(); Incompressible_Fluid does
        # not (and these units are not offered for liquids), so this
        # branch is compressible-only.
        mw = fluid.molar_mass()
        return ureg.Quantity(mdot_kgs / mw, "mol/s").to(target)

    if dim == {"[length]": 3, "[time]": -1}:
        if hasattr(fluid, "rhomass"):
            rho = fluid.rhomass()
        else:
            rho = fluid.density_si
        return ureg.Quantity(mdot_kgs / rho, "m^3/s").to(target)

    # Unrecognized -- fall back to SI mass flow so the summary still
    # shows something useful.
    return ureg.Quantity(mdot_kgs, "kg/s")


class ResultsScreen(QWidget):
    back_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        self.summary = QLabel("(no results yet)")
        self.summary.setStyleSheet("font-family: monospace;")

        self.x_unit = QComboBox()
        self.x_unit.addItems(U.LENGTH)
        self.x_unit.setCurrentText("ft")
        self.x_unit.currentTextChanged.connect(self._rerender)

        self.y_unit = QComboBox()
        self.y_unit.addItems(U.PRESSURE)
        self.y_unit.setCurrentText("psi")
        self.y_unit.currentTextChanged.connect(self._rerender)

        self.t_unit = QComboBox()
        self.t_unit.addItems(U.TEMPERATURE)
        self.t_unit.setCurrentText("degF")
        self.t_unit.currentTextChanged.connect(self._rerender)
        self.t_unit_label = QLabel("Y2 (temperature):")

        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("X (distance):"))
        units_row.addWidget(self.x_unit)
        units_row.addSpacing(20)
        units_row.addWidget(QLabel("Y (pressure):"))
        units_row.addWidget(self.y_unit)
        units_row.addSpacing(20)
        units_row.addWidget(self.t_unit_label)
        units_row.addWidget(self.t_unit)
        units_row.addStretch()

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Secondary ViewBox for the temperature trace (compressible only).
        # pyqtgraph's PlotWidget exposes a single ViewBox; a right-hand axis
        # with its own scaling has to be added manually by linking a second
        # ViewBox to the main PlotItem's scene.  We always build it (so the
        # linkage exists from the start) and toggle visibility per flow type.
        plot_item = self.plot.getPlotItem()
        self.temp_vb = pg.ViewBox()
        plot_item.showAxis("right")
        plot_item.scene().addItem(self.temp_vb)
        plot_item.getAxis("right").linkToView(self.temp_vb)
        self.temp_vb.setXLink(plot_item)
        plot_item.vb.sigResized.connect(self._sync_temp_vb_geom)

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        rerun_btn = QPushButton("Re-run")
        rerun_btn.clicked.connect(self._run)

        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(rerun_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(units_row)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)

    # Auto-run whenever the screen is shown.  Re-show after a Back+forward
    # gets a fresh calculation against any updated inputs.
    def showEvent(self, event):
        compressible = (self.state.flow_type == "compressible")
        self.t_unit.setVisible(compressible)
        self.t_unit_label.setVisible(compressible)
        self.plot.getPlotItem().getAxis("right").setVisible(compressible)
        super().showEvent(event)
        self._run()

    def _sync_temp_vb_geom(self):
        """Keep the secondary ViewBox glued to the main plot area on resize."""
        plot_item = self.plot.getPlotItem()
        self.temp_vb.setGeometry(plot_item.vb.sceneBoundingRect())
        self.temp_vb.linkedViewChanged(plot_item.vb, self.temp_vb.XAxis)

    # ------------------------------------------------------------------
    # Run + render
    # ------------------------------------------------------------------

    def _run(self):
        s = self.state
        if s.segment is None or s.fluid is None:
            QMessageBox.warning(
                self, "Missing input",
                "Segment or fluid is not defined.  Go back and complete the "
                "previous screens.",
            )
            return
        if s.solve_mode == "forward" and s.flow_rate is None:
            QMessageBox.warning(
                self, "Missing input",
                "Flow rate is not defined.  Go back and complete the fluid screen.",
            )
            return
        if s.solve_mode == "inverse" and s.P_outlet_Pa is None:
            QMessageBox.warning(
                self, "Missing input",
                "Outlet pressure is not defined.  Go back and complete the fluid screen.",
            )
            return

        try:
            if s.flow_type == "incompressible":
                if s.solve_mode == "inverse":
                    # dmdot returns mass flow rate; feed it back through
                    # pressure_profile to get the curve to plot.  Both
                    # calls use the user-supplied P_inlet as the anchor.
                    mdot_si = s.segment.dmdot(
                        fluid    = s.fluid,
                        P_inlet  = s.P_inlet_Pa,
                        P_outlet = s.P_outlet_Pa,
                    )
                    s.solved_mdot_kgs = mdot_si
                    results = s.segment.pressure_profile(
                        fluid     = s.fluid,
                        P0        = s.P_inlet_Pa,
                        flow_rate = ureg.Quantity(mdot_si, "kg/s"),
                    )
                else:
                    s.solved_mdot_kgs = None
                    results = s.segment.pressure_profile(
                        fluid     = s.fluid,
                        P0        = s.P_inlet_Pa or 0.0,
                        flow_rate = s.flow_rate,
                    )
            else:
                # dP_dT and dmdot_dT both mutate the AbstractState (and the
                # FlowState it wraps) in place, so re-anchor to the user-
                # supplied inlet conditions and build a fresh FlowState
                # before every run.
                s.fluid.update(CP.PT_INPUTS, s.P_inlet_Pa, s.T_inlet_K)
                if s.solve_mode == "inverse":
                    # dmdot_dT seeds mdot internally; the value passed here
                    # is overwritten on the first forward evaluation.  Use
                    # a small positive placeholder to satisfy FlowState's
                    # construction-time invariants.
                    fs = compressible_flow.FlowState(
                        s.fluid, mdot=1e-3,
                        A=s.segment.inlet_area_si, z=0.0,
                    )
                    raw = s.segment.dmdot_dT(
                        fs, P2=s.P_outlet_Pa, isothermal=s.isothermal,
                    )
                    s.solved_mdot_kgs = fs.mdot
                else:
                    mdot = compressible_flow._resolve_mdot(s.flow_rate, s.fluid)
                    fs = compressible_flow.FlowState(
                        s.fluid, mdot=mdot,
                        A=s.segment.inlet_area_si, z=0.0,
                    )
                    raw = s.segment.dP_dT(fs, isothermal=s.isothermal)
                    s.solved_mdot_kgs = None
                results = [
                    {"distance_m": d, "P_Pa": P, "T_K": T, "v_ms": v}
                    for (d, P, T, v) in raw
                ]
        except Exception as e:
            dialogs.critical(
                self, "Calculation failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return

        s.results = results
        self._rerender()

    def _rerender(self):
        """Redraw the plot and summary using the currently chosen X/Y units.

        Pulls from state.results (already in SI) and converts via pint, so
        unit changes are free -- no recalculation.
        """
        results = self.state.results
        if not results:
            return

        x_unit = self.x_unit.currentText()
        y_unit = self.y_unit.currentText()

        dist_si = [r["distance_m"] for r in results]
        P_si    = [r["P_Pa"]       for r in results]

        dist = [ureg.Quantity(d, "m").to(x_unit).magnitude  for d in dist_si]
        P    = [ureg.Quantity(p, "Pa").to(y_unit).magnitude for p in P_si]

        self.plot.clear()
        self.temp_vb.clear()
        plot_item = self.plot.getPlotItem()
        plot_item.setLabel("bottom", f"Distance ({x_unit})")
        plot_item.setLabel("left",   f"Pressure ({y_unit})", color="#1f77b4")
        plot_item.plot(
            dist, P,
            pen=pg.mkPen("#1f77b4", width=2),
            symbol="o", symbolSize=4, symbolBrush="#1f77b4",
        )

        L_q     = ureg.Quantity(dist_si[-1] - dist_si[0], "m")
        P_in_q  = ureg.Quantity(P_si[0],  "Pa")
        P_out_q = ureg.Quantity(P_si[-1], "Pa")
        dP_q    = P_out_q - P_in_q

        lines = [
            f"Length       : {L_q.to(x_unit).magnitude:>12.2f} {x_unit}",
            f"Inlet  P     : {P_in_q.to(y_unit).magnitude:>12.2f} {y_unit}",
            f"Outlet P     : {P_out_q.to(y_unit).magnitude:>12.2f} {y_unit}",
            f"Total dP     : {dP_q.to(y_unit).magnitude:>12.2f} {y_unit}",
        ]

        if self.state.solved_mdot_kgs is not None:
            mdot_display = _format_solved_mdot(
                self.state.solved_mdot_kgs,
                self.state.flow_rate_display_unit or "kg/s",
                self.state.fluid,
            )
            lines.append(
                f"Solved flow  : {mdot_display.magnitude:>12.4g} "
                f"{self.state.flow_rate_display_unit or 'kg/s'}"
            )

        # Compressible only: overlay temperature on the right-hand axis.
        if self.state.flow_type == "compressible" and "T_K" in results[0]:
            t_unit = self.t_unit.currentText()
            T_si   = [r["T_K"] for r in results]
            T_conv = [ureg.Quantity(T, "K").to(t_unit).magnitude for T in T_si]

            t_curve = pg.PlotCurveItem(
                dist, T_conv,
                pen=pg.mkPen("#d62728", width=2, style=Qt.DashLine),
            )
            self.temp_vb.addItem(t_curve)
            plot_item.getAxis("right").setLabel(
                f"Temperature ({t_unit})", color="#d62728"
            )
            # Auto-range the secondary ViewBox so the dashed T line is visible
            # whether T spans a wide or narrow range.
            self.temp_vb.enableAutoRange(axis=self.temp_vb.YAxis, enable=True)
            self._sync_temp_vb_geom()

            T_in_q  = ureg.Quantity(T_si[0],  "K")
            T_out_q = ureg.Quantity(T_si[-1], "K")
            lines.append(
                f"Inlet  T     : {T_in_q.to(t_unit).magnitude:>12.2f} {t_unit}"
            )
            lines.append(
                f"Outlet T     : {T_out_q.to(t_unit).magnitude:>12.2f} {t_unit}"
            )

        self.summary.setText("\n".join(lines))
