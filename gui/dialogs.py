"""Shared dialog helpers."""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from component_classes import ureg
from gui import units as U


def _scrollable_message(parent, title, text, icon):
    """Generic scrollable message dialog with a Copy button."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(600, 400)

    icon_label = QLabel()
    icon_label.setPixmap(dlg.style().standardPixmap(icon))
    icon_label.setFixedSize(32, 32)
    icon_label.setScaledContents(True)

    title_label = QLabel(f"<b>{title}</b>")

    header = QHBoxLayout()
    header.addWidget(icon_label)
    header.addWidget(title_label, stretch=1)

    text_box = QPlainTextEdit(text)
    text_box.setReadOnly(True)

    buttons = QDialogButtonBox()
    copy_btn = QPushButton("Copy")
    ok_btn = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
    buttons.addButton(copy_btn, QDialogButtonBox.ButtonRole.ActionRole)

    def _copy():
        QApplication.clipboard().setText(text)

    copy_btn.clicked.connect(_copy)
    ok_btn.clicked.connect(dlg.accept)

    layout = QVBoxLayout(dlg)
    layout.addLayout(header)
    layout.addWidget(text_box)
    layout.addWidget(buttons)

    dlg.exec()


def critical(parent, title, text):
    """Show a critical error dialog with a scrollable text area and Copy button."""
    _scrollable_message(
        parent, title, text, QStyle.StandardPixmap.SP_MessageBoxCritical,
    )


def warning(parent, title, text):
    """Show a non-fatal warning dialog with a scrollable text area and Copy button."""
    _scrollable_message(
        parent, title, text, QStyle.StandardPixmap.SP_MessageBoxWarning,
    )


# ---------------------------------------------------------------------------
# Node-results inspector dialog
# ---------------------------------------------------------------------------

class NodeResultsDialog(QDialog):
    """Modal dialog showing per-node solved details after a network solve.

    Rows is a list of (label, value_string) tuples that the caller has
    already formatted in display units.  An optional plot_callback, if
    given, adds a "Plot profile..." button that hands the dialog parent
    back to the caller so it can open a non-modal profile window.
    """

    def __init__(self, parent, title, rows, plot_callback=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._plot_callback = plot_callback

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label, value in rows:
            v = QLabel(value)
            v.setStyleSheet("font-family: monospace;")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(QLabel(label), v)

        button_row = QHBoxLayout()
        if plot_callback is not None:
            plot_btn = QPushButton("Plot profile...")
            plot_btn.clicked.connect(self._on_plot)
            button_row.addWidget(plot_btn)
        button_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(button_row)

    def _on_plot(self):
        # Close the modal first so the non-modal plot window isn't trapped
        # behind it on some window managers.
        self.accept()
        self._plot_callback()


# ---------------------------------------------------------------------------
# Pipe profile plot window
# ---------------------------------------------------------------------------

class PipeProfileWindow(QWidget):
    """Non-modal window plotting a pipe segment's solved profile.

    profile_points is a list of dicts with at least:
        {"distance_m": float, "P_Pa": float, "v_ms": float}
    and optionally "T_K": float for the compressible case.  When T_K is
    present, a secondary right-hand axis carries the temperature trace
    and a Temperature unit dropdown is shown -- mirrors ResultsScreen.

    The window owns its own X / Y / (T) unit dropdowns and re-renders
    against the SI-valued points dict on every selection change, so unit
    switches are free.
    """

    def __init__(
        self,
        title,
        profile_points,
        x_default="ft",
        p_default="psi",
        t_default="degF",
        parent=None,
    ):
        # Top-level Window flag so this stays independent of the modal dialog
        # that spawned it.
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.resize(800, 500)
        self._points = list(profile_points)
        self._has_T  = bool(self._points) and "T_K" in self._points[0]

        self.summary = QLabel("")
        self.summary.setStyleSheet("font-family: monospace;")

        self.x_unit = _make_unit_combo(U.LENGTH,      x_default, self._rerender)
        self.y_unit = _make_unit_combo(U.PRESSURE,    p_default, self._rerender)
        self.t_unit = _make_unit_combo(U.TEMPERATURE, t_default, self._rerender)
        self.t_unit_label = QLabel("Y2 (temperature):")

        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("X (distance):"))
        units_row.addWidget(self.x_unit)
        units_row.addSpacing(20)
        units_row.addWidget(QLabel("Y (pressure):"))
        units_row.addWidget(self.y_unit)
        if self._has_T:
            units_row.addSpacing(20)
            units_row.addWidget(self.t_unit_label)
            units_row.addWidget(self.t_unit)
        else:
            self.t_unit.setVisible(False)
            self.t_unit_label.setVisible(False)
        units_row.addStretch()

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Secondary ViewBox for the temperature trace (compressible only).
        # Same construction as ResultsScreen so both windows look identical.
        plot_item = self.plot.getPlotItem()
        self.temp_vb = pg.ViewBox()
        plot_item.showAxis("right")
        plot_item.scene().addItem(self.temp_vb)
        plot_item.getAxis("right").linkToView(self.temp_vb)
        self.temp_vb.setXLink(plot_item)
        plot_item.vb.sigResized.connect(self._sync_temp_vb_geom)
        plot_item.getAxis("right").setVisible(self._has_T)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(units_row)
        layout.addWidget(self.plot, 1)
        layout.addLayout(nav)

        self._rerender()

    def _sync_temp_vb_geom(self):
        plot_item = self.plot.getPlotItem()
        self.temp_vb.setGeometry(plot_item.vb.sceneBoundingRect())
        self.temp_vb.linkedViewChanged(plot_item.vb, self.temp_vb.XAxis)

    def _rerender(self):
        if not self._points:
            return
        x_unit = self.x_unit.currentText()
        y_unit = self.y_unit.currentText()

        dist_si = [r["distance_m"] for r in self._points]
        P_si    = [r["P_Pa"]       for r in self._points]
        dist = [ureg.Quantity(d, "m").to(U.to_pint(x_unit)).magnitude  for d in dist_si]
        P    = [ureg.Quantity(p, "Pa").to(U.to_pint(y_unit)).magnitude for p in P_si]

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
            f"Length       : {L_q.to(U.to_pint(x_unit)).magnitude:>12.2f} {x_unit}",
            f"Inlet  P     : {P_in_q.to(U.to_pint(y_unit)).magnitude:>12.2f} {y_unit}",
            f"Outlet P     : {P_out_q.to(U.to_pint(y_unit)).magnitude:>12.2f} {y_unit}",
            f"Total dP     : {dP_q.to(U.to_pint(y_unit)).magnitude:>12.2f} {y_unit}",
        ]

        if self._has_T:
            t_unit = self.t_unit.currentText()
            T_si   = [r["T_K"] for r in self._points]
            T_conv = [ureg.Quantity(T, "K").to(U.to_pint(t_unit)).magnitude for T in T_si]
            t_curve = pg.PlotCurveItem(
                dist, T_conv,
                pen=pg.mkPen("#d62728", width=2, style=Qt.PenStyle.DashLine),
            )
            self.temp_vb.addItem(t_curve)
            plot_item.getAxis("right").setLabel(
                f"Temperature ({t_unit})", color="#d62728"
            )
            self.temp_vb.enableAutoRange(axis=self.temp_vb.YAxis, enable=True)
            self._sync_temp_vb_geom()

            T_in_q  = ureg.Quantity(T_si[0],  "K")
            T_out_q = ureg.Quantity(T_si[-1], "K")
            lines.append(
                f"Inlet  T     : {T_in_q.to(U.to_pint(t_unit)).magnitude:>12.2f} {t_unit}"
            )
            lines.append(
                f"Outlet T     : {T_out_q.to(U.to_pint(t_unit)).magnitude:>12.2f} {t_unit}"
            )

        self.summary.setText("\n".join(lines))


def _make_unit_combo(items, default, on_change):
    from PySide6.QtWidgets import QComboBox
    cb = QComboBox()
    cb.addItems(items)
    cb.setCurrentText(default)
    cb.currentTextChanged.connect(lambda _t: on_change())
    return cb
