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
        if s.segment is None or s.fluid is None or s.flow_rate is None:
            QMessageBox.warning(
                self, "Missing input",
                "Segment or fluid is not defined.  Go back and complete the "
                "previous screens.",
            )
            return

        try:
            if s.flow_type == "incompressible":
                results = s.segment.pressure_profile(
                    fluid     = s.fluid,
                    P0        = s.P_inlet_Pa or 0.0,
                    flow_rate = s.flow_rate,
                )
            else:
                # dP_dT mutates the AbstractState (and the FlowState it
                # wraps) in-place, so re-anchor to the user-supplied inlet
                # conditions and build a fresh FlowState before every run.
                s.fluid.update(CP.PT_INPUTS, s.P_inlet_Pa, s.T_inlet_K)
                mdot = compressible_flow._resolve_mdot(s.flow_rate, s.fluid)
                fs = compressible_flow.FlowState(
                    s.fluid, mdot=mdot,
                    A=s.segment.inlet_area_si, z=0.0,
                )
                raw = s.segment.dP_dT(fs, isothermal=s.isothermal)
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
