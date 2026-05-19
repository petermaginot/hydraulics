"""Segment screen: define a pipe segment manually or load a profile from CSV.

Two input modes (chosen via tab):
  - Manual: scalar fields for ID (or OD+WT), length, elevation change,
            roughness.  Builds a 2-point Base_Line_Segment.
  - CSV   : pick a profile CSV; geometry/elevation come from the file, the
            user still supplies the roughness.

After the segment is built it is stashed on state.segment and an ID-vs-distance
chart is drawn.
"""

import os
import traceback

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from component_classes import ureg
from gui import units as U

# Lazy import the physics modules so the GUI can still launch (and show a
# friendly error) if an EOS backend like CoolProp fails to load.
def _load_segment_class(flow_type):
    if flow_type == "incompressible":
        from incompressible import Line_Segment
    else:
        from compressible_flow import Line_Segment
    return Line_Segment


def _qty(value_str, unit):
    """Parse a text-field string into a pint Quantity, or None if blank.

    Applies units.to_pint() so display labels like 'BBL/D' are
    translated to the pint identifier ('oil_bbl/day') before pint sees
    them.
    """
    s = value_str.strip()
    if not s:
        return None
    return ureg.Quantity(float(s), U.to_pint(unit))


def _row(field_widget, unit_widget):
    """Pack (line-edit, unit-combo) side by side for a QFormLayout row."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(field_widget, 1)
    row.addWidget(unit_widget)
    wrap = QWidget()
    wrap.setLayout(row)
    return wrap


class _LabeledField:
    """A line-edit + unit-combo pair, with a helper to read it as a Quantity."""

    def __init__(self, default_text="", units=None, default_unit=None):
        self.edit  = QLineEdit(default_text)
        self.combo = QComboBox()
        self.combo.addItems(units or [])
        if default_unit is not None and default_unit in (units or []):
            self.combo.setCurrentText(default_unit)

    def widget(self):
        return _row(self.edit, self.combo)

    def quantity(self):
        return _qty(self.edit.text(), self.combo.currentText())


class SegmentScreen(QWidget):
    back_clicked = Signal()
    next_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_manual_tab(), "Manual")
        self.tabs.addTab(self._build_csv_tab(),    "Load CSV profile")

        self.plot = pg.PlotWidget()
        # Default to engineering English units for this preview chart:
        # distance in feet, hydraulic diameter in inches.  Axis label strings
        # carry the unit literally -- pyqtgraph's units= kwarg auto-applies
        # SI prefixes (m -> mm/km), which is wrong for ft/inch.
        self.plot.setLabel("bottom", "Distance (ft)")
        self.plot.setLabel("left",   "Hydraulic diameter (inch)", color="#1f77b4")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Secondary right-hand Y axis for elevation, same linked-ViewBox
        # pattern as the temperature axis on the results screen.
        plot_item = self.plot.getPlotItem()
        self.elev_vb = pg.ViewBox()
        plot_item.showAxis("right")
        plot_item.getAxis("right").setLabel("Elevation (ft)", color="#2ca02c")
        plot_item.scene().addItem(self.elev_vb)
        plot_item.getAxis("right").linkToView(self.elev_vb)
        self.elev_vb.setXLink(plot_item)
        plot_item.vb.sigResized.connect(self._sync_elev_vb_geom)

        self.summary = QLabel("No segment built yet.")
        self.summary.setStyleSheet("color: #555;")

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        self.next_btn = QPushButton("Next →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.next_clicked.emit)

        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(self.next_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.summary)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)

    # If the user goes back to Start and switches regime, AppState clears
    # state.segment.  Resync the UI on every show so a stale "Next" button
    # or stale plot doesn't survive the change.
    def showEvent(self, event):
        super().showEvent(event)
        if self.state.segment is None:
            self.plot.clear()
            self.elev_vb.clear()
            self.summary.setText("No segment built yet.")
            self.summary.setStyleSheet("color: #555;")
            self.next_btn.setEnabled(False)

    def _sync_elev_vb_geom(self):
        """Keep the elevation ViewBox glued to the main plot area on resize."""
        plot_item = self.plot.getPlotItem()
        self.elev_vb.setGeometry(plot_item.vb.sceneBoundingRect())
        self.elev_vb.linkedViewChanged(plot_item.vb, self.elev_vb.XAxis)

    # ------------------------------------------------------------------
    # Manual tab
    # ------------------------------------------------------------------

    def _build_manual_tab(self):
        self.m_id     = _LabeledField("4.026", U.DIAMETER,  "inch")
        self.m_od     = _LabeledField("",      U.DIAMETER,  "inch")
        self.m_wt     = _LabeledField("",      U.DIAMETER,  "inch")
        self.m_length = _LabeledField("1000",  U.LENGTH,    "ft")
        self.m_dz     = _LabeledField("0",     U.LENGTH,    "ft")
        self.m_rough  = _LabeledField("0.00015", U.ROUGHNESS, "ft")

        form = QFormLayout()
        form.addRow("Inner diameter (ID):",     self.m_id.widget())
        form.addRow("OR outer diameter (OD):",  self.m_od.widget())
        form.addRow("    wall thickness (WT):", self.m_wt.widget())
        form.addRow("Length:",                  self.m_length.widget())
        form.addRow("Elevation change (dz):",   self.m_dz.widget())
        form.addRow("Roughness:",               self.m_rough.widget())

        help_lbl = QLabel(
            "Supply ID, or both OD and WT.  Elevation change is positive uphill."
        )
        help_lbl.setStyleSheet("color: #777; font-style: italic;")

        build_btn = QPushButton("Build segment")
        build_btn.clicked.connect(self._build_from_manual)

        tab = QWidget()
        v = QVBoxLayout(tab)
        v.addLayout(form)
        v.addWidget(help_lbl)
        v.addWidget(build_btn)
        v.addStretch()
        return tab

    def _build_from_manual(self):
        try:
            Line_Segment = _load_segment_class(self.state.flow_type)
            seg = Line_Segment(
                roughness        = self.m_rough.quantity(),
                id_val           = self.m_id.quantity(),
                od_val           = self.m_od.quantity(),
                wt_val           = self.m_wt.quantity(),
                length           = self.m_length.quantity(),
                elevation_change = self.m_dz.quantity(),
            )
        except Exception as e:
            self._error("Could not build segment", e)
            return
        self._set_segment(seg)

    # ------------------------------------------------------------------
    # CSV tab
    # ------------------------------------------------------------------

    def _build_csv_tab(self):
        self.csv_path_lbl = QLabel("(no file selected)")
        self.csv_path_lbl.setStyleSheet("color: #777;")
        self.csv_path = None

        pick_btn = QPushButton("Choose CSV...")
        pick_btn.clicked.connect(self._pick_csv)

        self.c_rough = _LabeledField("0.00015", U.ROUGHNESS, "ft")

        form = QFormLayout()
        form.addRow("Profile CSV:",  self.csv_path_lbl)
        form.addRow("",              pick_btn)
        form.addRow("Roughness:",    self.c_rough.widget())

        help_lbl = QLabel(
            "CSV must have header columns: distance, elevation, and either "
            "ID (with optional OD/WT) or D_h+flow_area.  All values in SI."
        )
        help_lbl.setStyleSheet("color: #777; font-style: italic;")
        help_lbl.setWordWrap(True)

        build_btn = QPushButton("Load profile")
        build_btn.clicked.connect(self._build_from_csv)

        tab = QWidget()
        v = QVBoxLayout(tab)
        v.addLayout(form)
        v.addWidget(help_lbl)
        v.addWidget(build_btn)
        v.addStretch()
        return tab

    def _pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select profile CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.csv_path = path
            self.csv_path_lbl.setText(os.path.basename(path))
            self.csv_path_lbl.setStyleSheet("color: #000;")

    def _build_from_csv(self):
        if not self.csv_path:
            QMessageBox.warning(self, "No file", "Choose a CSV file first.")
            return
        try:
            Line_Segment = _load_segment_class(self.state.flow_type)
            seg = Line_Segment.from_csv(
                csv_path  = self.csv_path,
                roughness = self.c_rough.quantity(),
            )
        except Exception as e:
            self._error("Could not load profile", e)
            return
        self._set_segment(seg)

    # ------------------------------------------------------------------
    # Shared: stash, plot, summarize
    # ------------------------------------------------------------------

    def _set_segment(self, seg):
        self.state.segment = seg
        self._draw_profile(seg)
        L  = seg.total_length_m
        dz = seg.net_elevation_change_m
        self.summary.setText(
            f"Segment ready: {len(seg.profile)} points, length = {L:.2f} m, "
            f"net dz = {dz:+.2f} m."
        )
        self.summary.setStyleSheet("color: #006400;")
        self.next_btn.setEnabled(True)

    def _draw_profile(self, seg):
        dist_ft = [ureg.Quantity(p[0], "m").to("ft").magnitude   for p in seg.profile]
        d_h_in  = [ureg.Quantity(p[2], "m").to("inch").magnitude for p in seg.profile]
        elev_ft = [ureg.Quantity(p[1], "m").to("ft").magnitude   for p in seg.profile]

        self.plot.clear()
        self.plot.plot(
            dist_ft, d_h_in,
            pen=pg.mkPen("#1f77b4", width=2),
            symbol="o", symbolSize=4, symbolBrush="#1f77b4",
        )
        # Anchor the left Y axis at zero so the diameter is read against an
        # absolute baseline rather than auto-zoomed around a narrow band.
        # padding=0 suppresses pyqtgraph's default range padding, which
        # would otherwise push the view below zero.
        y_max = max(d_h_in) if d_h_in else 1.0
        self.plot.setYRange(0, y_max * 1.1, padding=0)

        # Elevation on the secondary right-hand axis.  Auto-ranged because
        # elevations come from an arbitrary survey datum and can be negative
        # (subsea pipelines) -- a forced zero baseline would just compress
        # the interesting variation.
        self.elev_vb.clear()
        elev_curve = pg.PlotCurveItem(
            dist_ft, elev_ft,
            pen=pg.mkPen("#2ca02c", width=2, style=Qt.DashLine),
        )
        self.elev_vb.addItem(elev_curve)
        self.elev_vb.enableAutoRange(axis=self.elev_vb.YAxis, enable=True)
        self._sync_elev_vb_geom()

    def _error(self, title, exc):
        QMessageBox.critical(
            self, title,
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )
