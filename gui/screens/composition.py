"""Composition screen for the compressible-network workflow.

A trimmed-down sibling of the compressible panel on the point-to-point
fluid screen: defines the gas composition, the equation of state, and
the per-edge calculation mode (adiabatic vs isothermal), but does NOT
collect inlet P/T or a single flow rate -- those are per-node in a
network and are entered on the canvas screen instead.

On Calculate the unanchored AbstractState plus the isothermal flag are
stashed on AppState; the network screen reads them at solve time.
"""

import traceback

import gui.dialogs as dialogs
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import composition


class CompressibleCompositionScreen(QWidget):
    back_clicked = Signal()
    next_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        # ---- Composition table ----
        self.comp_table = QTableWidget(0, 2)
        self.comp_table.setHorizontalHeaderLabels(["Component", "Mole fraction"])
        self.comp_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch,
        )
        self.comp_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents,
        )
        # Seed with one row so the shape is visible.
        self._add_component_row("Methane", "1.0")

        add_btn  = QPushButton("+ Component")
        add_btn.clicked.connect(lambda: self._add_component_row())
        rm_btn   = QPushButton("- Remove selected")
        rm_btn.clicked.connect(self._remove_selected_component_row)
        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self._load_composition_csv)

        comp_btns = QHBoxLayout()
        comp_btns.addWidget(add_btn)
        comp_btns.addWidget(rm_btn)
        comp_btns.addWidget(load_btn)
        comp_btns.addStretch()

        comp_box = QGroupBox("Composition (mole fractions, normalised internally)")
        comp_v = QVBoxLayout(comp_box)
        comp_v.addWidget(self.comp_table)
        comp_v.addLayout(comp_btns)

        # ---- EOS + calculation mode ----
        self.eos_combo = QComboBox()
        self.eos_combo.addItems(["HEOS", "PR"])

        eos_form = QFormLayout()
        eos_form.addRow("Equation of state:", self.eos_combo)

        self.rb_adiabatic  = QRadioButton("Adiabatic (energy balance)")
        self.rb_isothermal = QRadioButton("Isothermal (constant T)")
        self.rb_adiabatic.setChecked(True)
        # The network solver currently only supports adiabatic mode --
        # Compressible_Network builds energy-balance residuals at every
        # non-T-spec node and does not forward an isothermal flag to
        # dP_dT.  Surface that limitation up front so it isn't a surprise
        # at solve time.
        self.rb_isothermal.setEnabled(False)
        self.rb_isothermal.setToolTip(
            "Not yet supported by the compressible network solver: every "
            "edge runs adiabatic.  See network.md for the open-work item."
        )
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.rb_adiabatic)
        mode_group.addButton(self.rb_isothermal)

        mode_box = QGroupBox("Calculation mode (applied to every edge)")
        mode_v = QVBoxLayout(mode_box)
        mode_v.addWidget(self.rb_adiabatic)
        mode_v.addWidget(self.rb_isothermal)

        hint = QLabel(
            "Inlet pressure, temperature, and per-stream flow are specified "
            "per Source/Sink on the next screen — a network can have "
            "multiple sources and sinks at different conditions."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-style: italic;")

        # ---- Navigation ----
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        next_btn = QPushButton("Build →")
        next_btn.clicked.connect(self._on_build)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #555;")

        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(next_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(comp_box, 1)
        layout.addLayout(eos_form)
        layout.addWidget(mode_box)
        layout.addWidget(hint)
        layout.addWidget(self.status)
        layout.addLayout(nav)

    # ------------------------------------------------------------------
    # Composition table editing (shared shape with fluid.py's compressible
    # panel; intentionally duplicated rather than ripped out into a helper
    # because the two screens diverge on the surrounding context).
    # ------------------------------------------------------------------

    def _add_component_row(self, component=None, fraction=""):
        used = self._used_components()
        if component is None:
            available = [c for c in composition.KNOWN_COMPONENTS if c not in used]
            if not available:
                return
            component = available[0]

        row = self.comp_table.rowCount()
        self.comp_table.insertRow(row)

        combo = QComboBox()
        combo.addItems(composition.KNOWN_COMPONENTS)
        if component in composition.KNOWN_COMPONENTS:
            combo.setCurrentText(component)
        combo.currentTextChanged.connect(self._refresh_component_choices)
        self.comp_table.setCellWidget(row, 0, combo)

        item = QTableWidgetItem(str(fraction))
        self.comp_table.setItem(row, 1, item)

        self._refresh_component_choices()

    def _used_components(self):
        used = set()
        for r in range(self.comp_table.rowCount()):
            c = self.comp_table.cellWidget(r, 0)
            if c is not None:
                used.add(c.currentText())
        return used

    def _refresh_component_choices(self):
        n = self.comp_table.rowCount()
        combos = [self.comp_table.cellWidget(r, 0) for r in range(n)]
        selections = [c.currentText() if c is not None else "" for c in combos]
        for i, combo in enumerate(combos):
            if combo is None:
                continue
            own = selections[i]
            others = set(selections[:i]) | set(selections[i + 1:])
            allowed = [c for c in composition.KNOWN_COMPONENTS
                       if c == own or c not in others]
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(allowed)
            combo.setCurrentText(own)
            combo.blockSignals(False)

    def _remove_selected_component_row(self):
        rows = sorted({i.row() for i in self.comp_table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.comp_table.removeRow(r)
        self._refresh_component_choices()

    def _load_composition_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load composition CSV", "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            fractions = composition.parse_composition_csv(path)
        except Exception as e:
            dialogs.critical(
                self, "Could not load composition",
                f"{type(e).__name__}: {e}",
            )
            return
        self.comp_table.setRowCount(0)
        for name, val in fractions.items():
            self._add_component_row(name, f"{val:g}")
        self._refresh_component_choices()

    # ------------------------------------------------------------------
    # Build -> stash on AppState
    # ------------------------------------------------------------------

    def _on_build(self):
        try:
            AS = self._build_abstract_state()
        except Exception as e:
            dialogs.critical(
                self, "Invalid composition",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        self.state.fluid      = AS
        self.state.isothermal = self.rb_isothermal.isChecked()
        # Per-node P, T, mdot are set on the network screen; clear any
        # stale values stashed by an earlier point-to-point compressible
        # run so a later screen can't be tripped up by them.
        self.state.P_inlet_Pa = None
        self.state.T_inlet_K  = None
        self.state.flow_rate  = None
        self.next_clicked.emit()

    def _build_abstract_state(self):
        kwargs = {}
        seen = set()
        for row in range(self.comp_table.rowCount()):
            combo = self.comp_table.cellWidget(row, 0)
            item  = self.comp_table.item(row, 1)
            if combo is None or item is None:
                continue
            text = item.text().strip()
            if not text:
                continue
            val = float(text)
            if val <= 0:
                continue
            name = combo.currentText()
            # The per-row filter usually prevents duplicates but a CSV load
            # that runs before the filter rebuild could leave one behind.
            if name in seen:
                raise ValueError(
                    f"Duplicate component '{name}' in the composition "
                    f"table.  Remove the redundant row."
                )
            seen.add(name)
            kwargs[f"y_{name}"] = val
        if not kwargs:
            raise ValueError(
                "Add at least one component with a positive mole fraction."
            )
        kwargs["eos"] = self.eos_combo.currentText()
        return composition.define_composition(**kwargs)
