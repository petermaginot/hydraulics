"""Fluid screen: define the working fluid and flow rate.

Two layouts depending on state.flow_type:
  - incompressible: density OR API gravity, viscosity, flow rate.
  - compressible  : mole-fraction table, inlet P/T, flow rate.

A successful Calculate populates state.fluid, state.flow_rate, and inlet
P/T (compressible only) before emitting next_clicked.
"""

import traceback

import CoolProp.CoolProp as CP
import gui.dialogs as dialogs
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import composition
from component_classes import ureg
from gui import units as U
from gui.screens.segment import _LabeledField


class FluidScreen(QWidget):
    back_clicked = Signal()
    next_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_incompressible_panel())  # index 0
        self.stack.addWidget(self._build_compressible_panel())    # index 1

        self.status = QLabel("")
        self.status.setStyleSheet("color: #555;")

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        calc_btn = QPushButton("Calculate →")
        calc_btn.clicked.connect(self._on_calculate)

        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(calc_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.status)
        layout.addLayout(nav)

    # showEvent fires every time MainWindow flips to this screen; use it to
    # show the correct sub-panel for the flow type chosen on the Start screen.
    def showEvent(self, event):
        idx = 0 if self.state.flow_type == "incompressible" else 1
        self.stack.setCurrentIndex(idx)
        super().showEvent(event)

    # ------------------------------------------------------------------
    # Incompressible panel
    # ------------------------------------------------------------------

    def _build_incompressible_panel(self):
        self.rb_density = QRadioButton("Density")
        self.rb_api     = QRadioButton("API gravity")
        self.rb_density.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.rb_density)
        group.addButton(self.rb_api)

        self.i_density = _LabeledField("62.4", U.DENSITY,   "lb/ft^3")
        self.i_api     = _LabeledField("35.0", ["dimensionless"], "dimensionless")
        # The API combo is decorative -- API gravity is dimensionless.  Hide
        # the dropdown to avoid implying otherwise.
        self.i_api.combo.setVisible(False)

        self.i_visc    = _LabeledField("1.0",  U.VISCOSITY, "cP")
        self.i_flow    = _LabeledField("1000", U.FLOW_RATE_INCOMPRESSIBLE, "BBL/D")

        def _toggle():
            self.i_density.edit.setEnabled(self.rb_density.isChecked())
            self.i_density.combo.setEnabled(self.rb_density.isChecked())
            self.i_api.edit.setEnabled(self.rb_api.isChecked())
        self.rb_density.toggled.connect(_toggle)
        self.rb_api.toggled.connect(_toggle)
        _toggle()

        rho_box = QGroupBox("Liquid density")
        rho_form = QFormLayout(rho_box)
        rho_form.addRow(self.rb_density,  self.i_density.widget())
        rho_form.addRow(self.rb_api,      self.i_api.widget())

        other = QFormLayout()
        other.addRow("Viscosity:", self.i_visc.widget())
        other.addRow("Flow rate:", self.i_flow.widget())

        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(rho_box)
        v.addLayout(other)
        v.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Compressible panel
    # ------------------------------------------------------------------

    def _build_compressible_panel(self):
        self.comp_table = QTableWidget(0, 2)
        self.comp_table.setHorizontalHeaderLabels(["Component", "Mole fraction"])
        self.comp_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.comp_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        # Seed with a one-component default so the user sees the shape.
        self._add_component_row("Methane", "1.0")

        add_btn = QPushButton("+ Component")
        add_btn.clicked.connect(lambda: self._add_component_row())
        rm_btn  = QPushButton("- Remove selected")
        rm_btn.clicked.connect(self._remove_selected_component_row)
        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self._load_composition_csv)

        comp_btns = QHBoxLayout()
        comp_btns.addWidget(add_btn)
        comp_btns.addWidget(rm_btn)
        comp_btns.addWidget(load_btn)
        comp_btns.addStretch()

        self.eos_combo = QComboBox()
        self.eos_combo.addItems(["HEOS", "PR"])

        self.c_pressure = _LabeledField("1000", U.PRESSURE,    "psi")
        self.c_temp     = _LabeledField("80",   U.TEMPERATURE, "degF")
        self.c_flow     = _LabeledField("50",   U.FLOW_RATE_COMPRESSIBLE, "mmscf/day")

        self.rb_adiabatic  = QRadioButton("Adiabatic (energy balance)")
        self.rb_isothermal = QRadioButton("Isothermal (constant T)")
        self.rb_adiabatic.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.rb_adiabatic)
        mode_group.addButton(self.rb_isothermal)

        comp_box = QGroupBox("Composition (mole fractions, normalised internally)")
        comp_v = QVBoxLayout(comp_box)
        comp_v.addWidget(self.comp_table)
        comp_v.addLayout(comp_btns)

        cond = QFormLayout()
        cond.addRow("Equation of state:", self.eos_combo)
        cond.addRow("Inlet pressure:",    self.c_pressure.widget())
        cond.addRow("Inlet temperature:", self.c_temp.widget())
        cond.addRow("Flow rate:",         self.c_flow.widget())

        mode_box = QGroupBox("Calculation mode")
        mode_v = QVBoxLayout(mode_box)
        mode_v.addWidget(self.rb_adiabatic)
        mode_v.addWidget(self.rb_isothermal)

        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(comp_box, 1)
        v.addLayout(cond)
        v.addWidget(mode_box)
        return panel

    def _add_component_row(self, component=None, fraction=""):
        # If no explicit component is requested, default to the first
        # KNOWN_COMPONENTS entry not already in use, so '+ Component' never
        # produces an instant duplicate.
        used = self._used_components()
        if component is None:
            available = [c for c in composition.KNOWN_COMPONENTS if c not in used]
            if not available:
                return   # all 20 components already in the table; no-op
            component = available[0]

        row = self.comp_table.rowCount()
        self.comp_table.insertRow(row)

        combo = QComboBox()
        combo.addItems(composition.KNOWN_COMPONENTS)
        if component in composition.KNOWN_COMPONENTS:
            combo.setCurrentText(component)
        # Connect AFTER the initial selection so the refresh isn't triggered
        # during construction.
        combo.currentTextChanged.connect(self._refresh_component_choices)
        self.comp_table.setCellWidget(row, 0, combo)

        item = QTableWidgetItem(str(fraction))
        self.comp_table.setItem(row, 1, item)

        self._refresh_component_choices()

    def _used_components(self):
        """Set of component names currently selected across all table rows."""
        used = set()
        for r in range(self.comp_table.rowCount()):
            c = self.comp_table.cellWidget(r, 0)
            if c is not None:
                used.add(c.currentText())
        return used

    def _refresh_component_choices(self):
        """Rebuild each row's combo so already-picked components don't appear
        in other rows' dropdowns.  Each combo's own current selection is
        always preserved in its own list.  Signals are blocked during the
        rebuild so the connected currentTextChanged handler doesn't recurse.
        """
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
        """Replace the table contents with rows parsed from a CSV file.

        Uses composition.parse_composition_csv so the same loader works
        headless.  The user can still edit/add/remove rows after loading;
        the AbstractState is only built when Calculate is pressed.
        """
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
    # Calculate -> build fluid objects, stash on state
    # ------------------------------------------------------------------

    def _on_calculate(self):
        try:
            if self.state.flow_type == "incompressible":
                self._finalize_incompressible()
            else:
                self._finalize_compressible()
        except Exception as e:
            dialogs.critical(
                self, "Invalid fluid input",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        self.next_clicked.emit()

    def _finalize_incompressible(self):
        from incompressible import Incompressible_Fluid

        visc = self.i_visc.quantity()
        if visc is None:
            raise ValueError("Viscosity is required.")

        if self.rb_density.isChecked():
            rho = self.i_density.quantity()
            if rho is None:
                raise ValueError("Density is required.")
            fluid = Incompressible_Fluid(density=rho, viscosity=visc)
        else:
            api_str = self.i_api.edit.text().strip()
            if not api_str:
                raise ValueError("API gravity is required.")
            fluid = Incompressible_Fluid.from_api_gravity(
                api_gravity=float(api_str), viscosity=visc,
            )

        flow = self.i_flow.quantity()
        if flow is None:
            raise ValueError("Flow rate is required.")

        # For incompressible we use atmospheric-ish inlet P by default (the
        # pressure profile is anchored at P0 and only the *change* matters
        # for the friction/elevation calc).
        self.state.fluid      = fluid
        self.state.flow_rate  = flow
        self.state.P_inlet_Pa = ureg.Quantity(0.0, "psi").to("Pa").magnitude
        self.state.T_inlet_K  = None

    def _finalize_compressible(self):
        kwargs = {}
        seen  = set()
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
            # Backstop: the per-row combo filter should already prevent
            # duplicates, but a stale state (e.g. CSV load before the
            # filter rebuild ran) would silently overwrite via dict
            # semantics.  Fail loudly instead.
            if name in seen:
                raise ValueError(
                    f"Duplicate component '{name}' in the composition table. "
                    f"Remove the redundant row."
                )
            seen.add(name)
            kwargs[f"y_{name}"] = val
        if not kwargs:
            raise ValueError("Add at least one component with a positive mole fraction.")
        kwargs["eos"] = self.eos_combo.currentText()

        AS = composition.define_composition(**kwargs)

        P = self.c_pressure.quantity()
        T = self.c_temp.quantity()
        flow = self.c_flow.quantity()
        if P is None or T is None or flow is None:
            raise ValueError("Inlet pressure, temperature, and flow rate are required.")

        P_Pa = P.to("Pa").magnitude
        T_K  = T.to("K").magnitude
        AS.update(CP.PT_INPUTS, P_Pa, T_K)

        self.state.fluid      = AS
        self.state.flow_rate  = flow
        self.state.P_inlet_Pa = P_Pa
        self.state.T_inlet_K  = T_K
        self.state.isothermal = self.rb_isothermal.isChecked()
