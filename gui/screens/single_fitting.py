"""Single fitting/valve calculator -- incompressible (base class).

One-screen workflow: fitting type + geometry, fluid properties, operating
mode, and Calculate button all live on a single scrollable panel.  Results
render in-place in a text panel at the bottom (no separate Results screen).

Subclassed by CompressibleSingleFittingScreen for the compressible regime;
it overrides the component class attrs, the fluid block, and the solve logic.
"""

import math
import traceback
import warnings

import gui.dialogs as dialogs
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from component_classes import ureg
from gui import units as U
from gui.screens.segment import _LabeledField
from gui.widgets.fitting_editors import (
    FittingPropertiesEditor,
    ValvePropertiesEditor,
    CheckValvePropertiesEditor,
)
from gui.throat import incompressible_throat_state
from incompressible import (
    Incompressible_Fluid,
    Bend, Valve, CheckValve, Orifice, Contraction_Expansion,
)

# Pint dimensionality for unit dispatch in _format_mdot
_MASS_FLOW_DIM = ureg.Quantity(1.0, "kg/s").dimensionality
_MOLAR_FLOW_DIM = ureg.Quantity(1.0, "mol/s").dimensionality
_VOL_FLOW_DIM   = ureg.Quantity(1.0, "m^3/s").dimensionality


class SingleFittingScreen(QWidget):
    back_clicked = Signal()

    # Regime-specific component classes — subclass swaps these.
    BEND_CLS            = Bend
    VALVE_CLS           = Valve
    CHECKVALVE_CLS      = CheckValve
    ORIFICE_CLS         = Orifice
    CONTRACTION_EXP_CLS = Contraction_Expansion

    # Flow-rate unit list shown in the dropdown.
    FLOW_RATE_UNITS   = U.FLOW_RATE_INCOMPRESSIBLE
    FLOW_RATE_DEFAULT = "BBL/D"

    # Show the optional vapor-pressure field (incompressible cavitation check).
    SHOW_VAPOR_PRESSURE = True

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        # ---- Mode selector (forward / inverse) ----
        self.rb_forward = QRadioButton(
            "Solve for outlet pressure (given flow rate)"
        )
        self.rb_inverse = QRadioButton(
            "Solve for flow rate (given outlet pressure)"
        )
        self.rb_forward.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.rb_forward)
        mode_group.addButton(self.rb_inverse)
        self.rb_forward.toggled.connect(self._apply_mode)
        self.rb_inverse.toggled.connect(self._apply_mode)

        mode_box = QGroupBox("Operating mode")
        mode_v = QVBoxLayout(mode_box)
        mode_v.addWidget(self.rb_forward)
        mode_v.addWidget(self.rb_inverse)
        inner_layout.addWidget(mode_box)

        # ---- Fitting type selector + editor stack ----
        self.category_combo = QComboBox()
        self.category_combo.addItems([
            "Bend",
            "Sudden Contraction/Expansion",
            "Orifice plate",
            "Valve",
            "Check Valve",
        ])
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)

        self.fit_editor = FittingPropertiesEditor()
        self.fit_editor.type_combo.setVisible(False)  # outer combo controls type

        self.valve_editor = ValvePropertiesEditor()
        self.cv_editor    = CheckValvePropertiesEditor()

        self.editor_stack = QStackedWidget()
        self.editor_stack.addWidget(self.fit_editor)   # 0: bend/CE/orifice
        self.editor_stack.addWidget(self.valve_editor) # 1: valve
        self.editor_stack.addWidget(self.cv_editor)    # 2: check valve

        fit_box = QGroupBox("Fitting / valve")
        fit_v = QVBoxLayout(fit_box)
        fit_v.addWidget(self.category_combo)
        fit_v.addWidget(self.editor_stack)
        inner_layout.addWidget(fit_box)

        # ---- Fluid block (subclass may replace this) ----
        self.fluid_box = self._build_fluid_box()
        inner_layout.addWidget(self.fluid_box)

        # ---- Inlet / outlet / flow inputs ----
        self.p_inlet   = _LabeledField("100", U.PRESSURE, "psi")
        self.p_outlet  = _LabeledField("90",  U.PRESSURE, "psi")
        self.flow_rate = _LabeledField("1000", self.FLOW_RATE_UNITS,
                                       self.FLOW_RATE_DEFAULT)
        # vapor pressure (incompressible only, optional — enables cavitation check)
        if self.SHOW_VAPOR_PRESSURE:
            self.p_vapor   = _LabeledField("", U.PRESSURE, "psi")
            self.p_vapor.edit.setPlaceholderText("optional, for cavitation check")
        else:
            self.p_vapor   = None

        cond_form = QFormLayout()
        cond_form.addRow("Inlet pressure:",  self.p_inlet.widget())
        cond_form.addRow("Outlet pressure:", self.p_outlet.widget())
        cond_form.addRow("Flow rate:",       self.flow_rate.widget())
        self._add_extra_cond_rows(cond_form)
        if self.SHOW_VAPOR_PRESSURE:
            cond_form.addRow("Vapor pressure:",  self.p_vapor.widget())

        cond_box = QGroupBox("Operating conditions")
        cond_box.setLayout(cond_form)
        inner_layout.addWidget(cond_box)

        # ---- Calculate button ----
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self._on_calculate)
        inner_layout.addWidget(calc_btn)

        # ---- Results panel ----
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Results will appear here after Calculate.")
        self.results_text.setMinimumHeight(180)
        res_box = QGroupBox("Results")
        res_v = QVBoxLayout(res_box)
        res_v.addWidget(self.results_text)
        inner_layout.addWidget(res_box)

        inner_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()

        outer = QVBoxLayout(self)
        outer.addWidget(scroll, 1)
        outer.addLayout(nav)

        # Sync initial state
        self._on_category_changed(0)
        self._apply_mode()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_fluid_box(self):
        """Build and return the fluid-properties group box.

        Incompressible default: density / API radio + viscosity.
        Compressible subclass replaces this with a composition summary label.
        """
        self.rb_density = QRadioButton("Density")
        self.rb_api     = QRadioButton("API gravity")
        self.rb_density.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_density)
        grp.addButton(self.rb_api)

        self.i_density = _LabeledField("62.4", U.DENSITY, "lb/ft^3")
        self.i_api     = _LabeledField("35.0", ["dimensionless"], "dimensionless")
        self.i_api.combo.setVisible(False)
        self.i_visc    = _LabeledField("1.0", U.VISCOSITY, "cP")

        def _toggle():
            self.i_density.edit.setEnabled(self.rb_density.isChecked())
            self.i_density.combo.setEnabled(self.rb_density.isChecked())
            self.i_api.edit.setEnabled(self.rb_api.isChecked())
        self.rb_density.toggled.connect(_toggle)
        self.rb_api.toggled.connect(_toggle)
        _toggle()

        rho_box = QGroupBox("Liquid density")
        rho_form = QFormLayout(rho_box)
        rho_form.addRow(self.rb_density, self.i_density.widget())
        rho_form.addRow(self.rb_api,     self.i_api.widget())

        fluid_form = QFormLayout()
        fluid_form.addRow("Viscosity:", self.i_visc.widget())

        box = QGroupBox("Fluid")
        v = QVBoxLayout(box)
        v.addWidget(rho_box)
        v.addLayout(fluid_form)
        return box

    def _add_extra_cond_rows(self, form):
        """Hook: subclass adds inlet temperature row here."""
        pass

    def _build_fluid(self):
        """Parse the fluid panel and return an Incompressible_Fluid.

        Raises ValueError if required fields are missing.
        """
        visc = self.i_visc.quantity()
        if visc is None:
            raise ValueError("Viscosity is required.")
        if self.rb_density.isChecked():
            rho = self.i_density.quantity()
            if rho is None:
                raise ValueError("Density is required.")
            return Incompressible_Fluid(density=rho, viscosity=visc)
        api_str = self.i_api.edit.text().strip()
        if not api_str:
            raise ValueError("API gravity is required.")
        return Incompressible_Fluid.from_api_gravity(
            api_gravity=float(api_str), viscosity=visc,
        )

    def _get_vapor_pressure_pa(self):
        """Return vapor pressure [Pa] or None if field is blank."""
        pv = self.p_vapor.quantity()
        if pv is None:
            return None
        return pv.to("Pa").magnitude

    # ------------------------------------------------------------------
    # Category / mode management
    # ------------------------------------------------------------------

    # Maps outer combo index → (editor_stack page, fitting_editor type index or -1)
    _CAT_PAGE = [0, 0, 0, 1, 2]
    _CAT_FIT  = [0, 1, 2, -1, -1]   # FittingPropertiesEditor type index

    def _on_category_changed(self, idx):
        self.editor_stack.setCurrentIndex(self._CAT_PAGE[idx])
        fit_idx = self._CAT_FIT[idx]
        if fit_idx >= 0:
            self.fit_editor.type_combo.setCurrentIndex(fit_idx)
            self.fit_editor.input_stack.setCurrentIndex(fit_idx)

    def _apply_mode(self):
        inverse = self.rb_inverse.isChecked()
        self.p_outlet.edit.setEnabled(inverse)
        self.p_outlet.combo.setEnabled(inverse)
        self.flow_rate.edit.setEnabled(not inverse)

    # ------------------------------------------------------------------
    # Build fitting from current editor state
    # ------------------------------------------------------------------

    def _build_fitting(self):
        """Construct the component object from the current UI fields.

        Returns (component, fitting_kind_str).  Raises ValueError on
        missing / invalid input.
        """
        cat = self.category_combo.currentIndex()
        if cat == 3:   # Valve
            return (
                self.valve_editor.build_component(self.VALVE_CLS),
                "valve",
            )
        if cat == 4:   # Check Valve
            return (
                self.cv_editor.build_component(self.CHECKVALVE_CLS),
                "check_valve",
            )
        # Fitting (Bend / CE / Orifice) — delegate to FittingPropertiesEditor
        fit_idx = self._CAT_FIT[cat]
        kind_map = {0: "bend", 1: "contraction_expansion", 2: "orifice"}
        kind = kind_map[fit_idx]
        comp = self.fit_editor.build_component(
            self.BEND_CLS, self.ORIFICE_CLS, self.CONTRACTION_EXP_CLS,
        )
        return comp, kind

    # ------------------------------------------------------------------
    # Calculate
    # ------------------------------------------------------------------

    def _on_calculate(self):
        self.results_text.clear()
        try:
            self._run_calculate()
        except Exception as exc:
            self.results_text.setPlainText(
                f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

    def _run_calculate(self):
        fitting, kind = self._build_fitting()

        P_in_q = self.p_inlet.quantity()
        if P_in_q is None:
            raise ValueError("Inlet pressure is required.")
        P_inlet_Pa = P_in_q.to("Pa").magnitude

        inverse = self.rb_inverse.isChecked()
        if inverse:
            P_out_q = self.p_outlet.quantity()
            if P_out_q is None:
                raise ValueError("Outlet pressure is required in inverse mode.")
            P_outlet_Pa = P_out_q.to("Pa").magnitude
            if P_outlet_Pa >= P_inlet_Pa:
                raise ValueError(
                    "Inlet pressure must be greater than outlet pressure."
                )
        else:
            flow_q = self.flow_rate.quantity()
            if flow_q is None:
                raise ValueError("Flow rate is required in forward mode.")
            P_outlet_Pa = None

        result_lines, warn_lines = self._solve(
            fitting, kind, P_inlet_Pa, P_outlet_Pa,
            None if inverse else flow_q,
            inverse,
        )
        self._display_results(result_lines, warn_lines)

    def _solve(self, fitting, kind, P_inlet_Pa, P_outlet_Pa, flow_rate, inverse):
        """Run the incompressible solve and return (result_lines, warn_lines)."""
        fluid = self._build_fluid()
        # Attach vapor pressure if provided.
        pv = self._get_vapor_pressure_pa()
        if pv is not None:
            fluid.vapor_pressure_si = pv

        result_lines = []
        warn_lines   = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if inverse:
                mdot_kgs = fitting.dmdot(fluid, P_inlet_Pa, P_outlet_Pa)
                result_lines.append(
                    self._format_mdot_line(mdot_kgs, fluid)
                )
                # Forward pass to get outlet P (already known, but included for completeness)
                result_lines.append(
                    f"{'Outlet pressure:':28s} {self._fmt_P(P_outlet_Pa)}"
                )
                result_lines.append(
                    f"{'dP:':28s} {self._fmt_dP(P_outlet_Pa - P_inlet_Pa)}"
                )
                v_outlet = self._outlet_velocity(fitting, fluid, mdot_kgs)
                if v_outlet is not None:
                    result_lines.append(self._fmt_v_line("Outlet velocity:", v_outlet))
                # Throat state
                throat = incompressible_throat_state(fitting, fluid, mdot_kgs, P_inlet_Pa)
                if throat:
                    result_lines += self._format_throat_incompressible(throat)
            else:
                dP = fitting.dP(fluid, flow_rate, P_inlet=P_inlet_Pa)
                P_outlet_Pa = P_inlet_Pa + dP
                result_lines.append(
                    f"{'Outlet pressure:':28s} {self._fmt_P(P_outlet_Pa)}"
                )
                result_lines.append(
                    f"{'dP:':28s} {self._fmt_dP(dP)}"
                )
                # Inlet / outlet velocity
                mdot_kgs = self._mdot_from_flow(flow_rate, fluid)
                v_inlet  = self._inlet_velocity(fitting, fluid, mdot_kgs)
                if v_inlet is not None:
                    result_lines.append(self._fmt_v_line("Inlet velocity:", v_inlet))
                v_outlet = self._outlet_velocity(fitting, fluid, mdot_kgs)
                if v_outlet is not None:
                    result_lines.append(self._fmt_v_line("Outlet velocity:", v_outlet))
                # Throat state
                throat = incompressible_throat_state(fitting, fluid, mdot_kgs, P_inlet_Pa)
                if throat:
                    result_lines += self._format_throat_incompressible(throat)

            for w in caught:
                warn_lines.append(str(w.message))

        return result_lines, warn_lines

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _fmt_P(self, P_Pa):
        P_psi = ureg.Quantity(P_Pa, "Pa").to("psi").magnitude
        return f"{P_Pa/1e6:.4f} MPa  ({P_psi:.2f} psi)"

    def _fmt_dP(self, dP_Pa):
        dP_psi = ureg.Quantity(dP_Pa, "Pa").to("psi").magnitude
        sign = "+" if dP_Pa >= 0 else ""
        return f"{sign}{dP_Pa/1e6:.4f} MPa  ({sign}{dP_psi:.2f} psi)"

    def _format_mdot_line(self, mdot_kgs, fluid):
        unit_lbl = self.flow_rate.combo.currentText()
        q = self._convert_mdot(mdot_kgs, unit_lbl, fluid)
        return f"{'Solved flow rate:':28s} {q:.4g~P}"

    def _convert_mdot(self, mdot_kgs, display_unit, fluid):
        target = U.to_pint(display_unit)
        dim    = ureg.Quantity(1.0, target).dimensionality
        if dim == _MASS_FLOW_DIM:
            return ureg.Quantity(mdot_kgs, "kg/s").to(target)
        if dim == _MOLAR_FLOW_DIM:
            mw = getattr(fluid, "molar_mass", lambda: None)()
            if mw:
                return ureg.Quantity(mdot_kgs / mw, "mol/s").to(target)
        if dim == _VOL_FLOW_DIM:
            rho = getattr(fluid, "density_si", None) or getattr(fluid, "rhomass", lambda: 1.0)()
            return ureg.Quantity(mdot_kgs / rho, "m^3/s").to(target)
        return ureg.Quantity(mdot_kgs, "kg/s")

    def _mdot_from_flow(self, flow_rate, fluid):
        rho = fluid.density_si
        dim = flow_rate.dimensionality
        if dim == _MASS_FLOW_DIM:
            return flow_rate.to("kg/s").magnitude
        if dim == _VOL_FLOW_DIM:
            return flow_rate.to("m^3/s").magnitude * rho
        return flow_rate.to("kg/s").magnitude

    def _inlet_velocity(self, fitting, fluid, mdot_kgs):
        rho = fluid.density_si
        Q   = mdot_kgs / rho
        A   = getattr(fitting, "inlet_area_si", None)
        if A is None or A <= 0:
            Di = getattr(fitting, "Di_si", None) or getattr(fitting, "Di_US_si", None)
            if Di is None or Di <= 0:
                return None
            A = math.pi * Di ** 2 / 4.0
        return Q / A

    def _outlet_velocity(self, fitting, fluid, mdot_kgs):
        rho = fluid.density_si
        Q   = mdot_kgs / rho
        A   = getattr(fitting, "outlet_area_si", None)
        if A is None or A <= 0:
            Di = getattr(fitting, "Di_si", None) or getattr(fitting, "Di_DS_si", None)
            if Di is None or Di <= 0:
                return None
            A = math.pi * Di ** 2 / 4.0
        return Q / A

    def _fmt_v_line(self, label, v_ms):
        v_fps = ureg.Quantity(v_ms, "m/s").to("ft/s").magnitude
        return f"{label:28s} {v_ms:.3f} m/s  ({v_fps:.2f} ft/s)"

    def _format_throat_incompressible(self, throat):
        lines = ["", "--- Estimated vena contracta conditions ---"]
        v  = throat["v_ms"]
        P  = throat["P_Pa"]
        A  = throat["A_m2"]
        D_thr    = math.sqrt(4.0 * A / math.pi) if A > 0 else 0.0
        D_thr_in = ureg.Quantity(D_thr, "m").to("inch").magnitude
        A_in2    = ureg.Quantity(A, "m^2").to("inch^2").magnitude
        P_psi    = ureg.Quantity(P, "Pa").to("psi").magnitude
        v_fps    = ureg.Quantity(v, "m/s").to("ft/s").magnitude
        lines.append(f"{'Vena contracta area:':28s} {A*1e6:.2f} mm^2  ({A_in2:.4f} in^2)")
        lines.append(f"{'Vena contracta diameter:':28s} {D_thr*1000:.2f} mm  ({D_thr_in:.3f} in)")
        lines.append(f"{'Vena contracta pressure:':28s} {P/1e6:.4f} MPa  ({P_psi:.2f} psi)")
        lines.append(f"{'Vena contracta velocity:':28s} {v:.3f} m/s  ({v_fps:.2f} ft/s)")
        return lines

    def _display_results(self, result_lines, warn_lines):
        lines = []
        lines += result_lines
        if warn_lines:
            lines.append("")
            lines.append("--- Warnings ---")
            lines += warn_lines
        self.results_text.setPlainText("\n".join(lines))
