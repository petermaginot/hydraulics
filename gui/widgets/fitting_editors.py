"""Reusable fitting / valve / check-valve editor widgets.

Shared between the network node editor stack (gui/screens/network.py)
and the single-fitting calculator screen (gui/screens/single_fitting.py).

Each widget:
  - Owns its own sub-type combo and per-type stacked input pages.
  - Exposes get_spec() / set_spec(spec) for round-tripping through the
    network screen's node_specs dict (same key format as before).
  - Exposes build_component(cls_map, name) to construct the component
    object directly from the current field values, without needing a
    node_specs dict.
"""

import fluids.fittings
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from component_classes import ureg
from gui import units as U
from gui.screens.segment import _LabeledField


# ---------------------------------------------------------------------------
# FittingPropertiesEditor  (Bend / Contraction-Expansion / Orifice)
# ---------------------------------------------------------------------------

class FittingPropertiesEditor(QWidget):
    """Fitting type combo + per-type input stack for Bend / CE / Orifice."""

    _TYPES  = ["bend", "contraction_expansion", "orifice"]
    _LABELS = ["Bend", "Sudden Contraction/Expansion", "Orifice plate"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.type_combo = QComboBox()
        self.type_combo.addItems(self._LABELS)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        # --- Bend page (index 0) ---
        self.bend_Di    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.bend_angle = QLineEdit("90")
        self.bend_angle.setPlaceholderText("degrees")
        self.bend_dias  = QLineEdit("1.5")
        self.bend_dias.setPlaceholderText("R/D ratio")
        bend_form = QFormLayout()
        bend_form.addRow("Pipe ID:",     self.bend_Di.widget())
        bend_form.addRow("Angle (deg):", self.bend_angle)
        bend_form.addRow("Bend R/D:",    self.bend_dias)
        bend_hint = QLabel(
            "R/D = bend centerline radius ÷ pipe ID "
            "(e.g. 1.5 = standard long-radius elbow)."
        )
        bend_hint.setWordWrap(True)
        bend_hint.setStyleSheet("color: #777; font-style: italic;")
        bend_page = QWidget()
        bv = QVBoxLayout(bend_page)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addLayout(bend_form)
        bv.addWidget(bend_hint)

        # --- Contraction/Expansion page (index 1) ---
        self.ce_Di_US = _LabeledField("4.026", U.DIAMETER, "inch")
        self.ce_Di_DS = _LabeledField("3.0",   U.DIAMETER, "inch")
        ce_form = QFormLayout()
        ce_form.addRow("Upstream ID:",   self.ce_Di_US.widget())
        ce_form.addRow("Downstream ID:", self.ce_Di_DS.widget())
        ce_hint = QLabel(
            "Sharp-edged (abrupt) transition.  "
            "Contraction or expansion is inferred from the two diameters."
        )
        ce_hint.setWordWrap(True)
        ce_hint.setStyleSheet("color: #777; font-style: italic;")
        ce_page = QWidget()
        cev = QVBoxLayout(ce_page)
        cev.setContentsMargins(0, 0, 0, 0)
        cev.addLayout(ce_form)
        cev.addWidget(ce_hint)

        # --- Orifice page (index 2) ---
        self.orif_Di       = _LabeledField("4.026", U.DIAMETER, "inch")
        self.orif_Do       = _LabeledField("2.0",   U.DIAMETER, "inch")
        self.orif_taps     = QComboBox()
        self.orif_taps.addItems(["corner", "D and D/2", "flange"])
        self.orif_Cd       = QLineEdit("")
        self.orif_Cd.setPlaceholderText("blank = use RHG correlation")
        self.orif_beta_lbl = QLabel("β = -")
        self.orif_beta_lbl.setStyleSheet("color: #555;")
        orif_form = QFormLayout()
        orif_form.addRow("Pipe ID (Di):", self.orif_Di.widget())
        orif_form.addRow("Bore (Do):",    self.orif_Do.widget())
        orif_form.addRow("",              self.orif_beta_lbl)
        orif_form.addRow("Taps:",         self.orif_taps)
        orif_form.addRow("Cd override:",  self.orif_Cd)
        orif_hint = QLabel(
            "Square-edged concentric orifice plate.  "
            "Cd is computed via ISO 5167-2 (Reader-Harris-Gallagher) when "
            "the override is blank.  RHG valid range: β = Do/Di in [0.10, 0.75]."
        )
        orif_hint.setWordWrap(True)
        orif_hint.setStyleSheet("color: #777; font-style: italic;")
        orif_page = QWidget()
        ov = QVBoxLayout(orif_page)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.addLayout(orif_form)
        ov.addWidget(orif_hint)

        self.orif_Di.edit.textChanged.connect(self._update_beta_label)
        self.orif_Do.edit.textChanged.connect(self._update_beta_label)
        self.orif_Di.combo.currentTextChanged.connect(self._update_beta_label)
        self.orif_Do.combo.currentTextChanged.connect(self._update_beta_label)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(bend_page)   # 0
        self.input_stack.addWidget(ce_page)     # 1
        self.input_stack.addWidget(orif_page)   # 2

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.type_combo)
        layout.addWidget(self.input_stack)

    def _on_type_changed(self, idx):
        self.input_stack.setCurrentIndex(idx)

    def _update_beta_label(self, *_):
        try:
            Di = float(self.orif_Di.edit.text().strip())
            Do = float(self.orif_Do.edit.text().strip())
            Di_m = ureg.Quantity(
                Di, U.to_pint(self.orif_Di.combo.currentText())
            ).to("m").magnitude
            Do_m = ureg.Quantity(
                Do, U.to_pint(self.orif_Do.combo.currentText())
            ).to("m").magnitude
            if Di_m <= 0.0:
                raise ValueError
            beta = Do_m / Di_m
        except (ValueError, AttributeError):
            self.orif_beta_lbl.setText("β = -")
            self.orif_beta_lbl.setStyleSheet("color: #555;")
            return
        if 0.10 <= beta <= 0.75:
            self.orif_beta_lbl.setStyleSheet("color: #555;")
        else:
            self.orif_beta_lbl.setStyleSheet("color: #b07000;")
        self.orif_beta_lbl.setText(f"β = {beta:.4f}")

    def get_spec(self):
        """Return the spec dict in the same format network.py's node_specs uses."""
        ft = self._TYPES[self.type_combo.currentIndex()]
        return {
            "fitting_type":          ft,
            "Di_str":                self.bend_Di.edit.text().strip(),
            "Di_unit":               self.bend_Di.combo.currentText(),
            "angle_str":             self.bend_angle.text().strip(),
            "bend_dias_str":         self.bend_dias.text().strip(),
            "Di_US_str":             self.ce_Di_US.edit.text().strip(),
            "Di_US_unit":            self.ce_Di_US.combo.currentText(),
            "Di_DS_str":             self.ce_Di_DS.edit.text().strip(),
            "Di_DS_unit":            self.ce_Di_DS.combo.currentText(),
            "orif_Di_str":           self.orif_Di.edit.text().strip(),
            "orif_Di_unit":          self.orif_Di.combo.currentText(),
            "orif_Do_str":           self.orif_Do.edit.text().strip(),
            "orif_Do_unit":          self.orif_Do.combo.currentText(),
            "orif_taps":             self.orif_taps.currentText(),
            "orif_Cd_override_str":  self.orif_Cd.text().strip(),
        }

    def set_spec(self, spec):
        """Populate editor fields from a node_specs dict."""
        ft  = spec.get("fitting_type", "bend")
        idx = self._TYPES.index(ft) if ft in self._TYPES else 0
        self.type_combo.setCurrentIndex(idx)
        self.input_stack.setCurrentIndex(idx)

        self.bend_Di.edit.setText(spec.get("Di_str", "4.026"))
        self.bend_Di.combo.setCurrentText(spec.get("Di_unit", "inch"))
        self.bend_angle.setText(spec.get("angle_str", "90"))
        self.bend_dias.setText(spec.get("bend_dias_str", "1.5"))
        self.ce_Di_US.edit.setText(spec.get("Di_US_str", "4.026"))
        self.ce_Di_US.combo.setCurrentText(spec.get("Di_US_unit", "inch"))
        self.ce_Di_DS.edit.setText(spec.get("Di_DS_str", "3.0"))
        self.ce_Di_DS.combo.setCurrentText(spec.get("Di_DS_unit", "inch"))
        self.orif_Di.edit.setText(spec.get("orif_Di_str", "4.026"))
        self.orif_Di.combo.setCurrentText(spec.get("orif_Di_unit", "inch"))
        self.orif_Do.edit.setText(spec.get("orif_Do_str", "2.0"))
        self.orif_Do.combo.setCurrentText(spec.get("orif_Do_unit", "inch"))
        self.orif_taps.setCurrentText(spec.get("orif_taps", "corner"))
        self.orif_Cd.setText(spec.get("orif_Cd_override_str", ""))
        self._update_beta_label()

    def build_component(self, bend_cls, orifice_cls, ce_cls, name=None):
        """Build and return the component from the current field values."""
        ft = self._TYPES[self.type_combo.currentIndex()]
        nm = name or ""
        if ft == "bend":
            Di_str = self.bend_Di.edit.text().strip()
            if not Di_str:
                raise ValueError("Pipe ID is required for bend.")
            angle_str = self.bend_angle.text().strip()
            if not angle_str:
                raise ValueError("Bend angle is required.")
            bias_str = self.bend_dias.text().strip()
            if not bias_str:
                raise ValueError("Bend R/D ratio is required.")
            Di_q = ureg.Quantity(
                float(Di_str), U.to_pint(self.bend_Di.combo.currentText())
            )
            return bend_cls(
                Di=Di_q, ang_deg=float(angle_str),
                bend_dias=float(bias_str), name=nm,
            )
        if ft == "orifice":
            Di_str = self.orif_Di.edit.text().strip()
            Do_str = self.orif_Do.edit.text().strip()
            if not Di_str or not Do_str:
                raise ValueError("Orifice pipe ID and bore are required.")
            Di_q = ureg.Quantity(
                float(Di_str), U.to_pint(self.orif_Di.combo.currentText())
            )
            Do_q = ureg.Quantity(
                float(Do_str), U.to_pint(self.orif_Do.combo.currentText())
            )
            cd_str = self.orif_Cd.text().strip()
            Cd_override = float(cd_str) if cd_str else None
            return orifice_cls(
                Di=Di_q, Do=Do_q,
                taps=self.orif_taps.currentText(),
                Cd_override=Cd_override, name=nm,
            )
        # contraction_expansion
        Di_US_str = self.ce_Di_US.edit.text().strip()
        Di_DS_str = self.ce_Di_DS.edit.text().strip()
        if not Di_US_str or not Di_DS_str:
            raise ValueError("Upstream and downstream IDs are required.")
        Di_US_q = ureg.Quantity(
            float(Di_US_str), U.to_pint(self.ce_Di_US.combo.currentText())
        )
        Di_DS_q = ureg.Quantity(
            float(Di_DS_str), U.to_pint(self.ce_Di_DS.combo.currentText())
        )
        return ce_cls(Di_US=Di_US_q, Di_DS=Di_DS_q, name=nm)


# ---------------------------------------------------------------------------
# ValvePropertiesEditor
# ---------------------------------------------------------------------------

class ValvePropertiesEditor(QWidget):
    """Valve type combo + per-type input stack for the six Crane valve types."""

    _TYPES  = ["globe", "gate", "butterfly", "plug", "ball", "user_specified"]
    _LABELS = ["Globe", "Gate", "Butterfly", "Plug", "Ball", "User specified"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.type_combo = QComboBox()
        self.type_combo.addItems(self._LABELS)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        # Globe (index 0)
        self.globe_D1 = _LabeledField("4.026", U.DIAMETER, "inch")
        self.globe_D2 = _LabeledField("4.026", U.DIAMETER, "inch")
        globe_form = QFormLayout()
        globe_form.addRow("Seat bore D1:", self.globe_D1.widget())
        globe_form.addRow("Pipe ID D2:",   self.globe_D2.widget())
        globe_hint = QLabel(
            "D1 ≤ D2.  Set D1 = D2 for a full-bore globe valve "
            "(K ≈ 340×fₐ, typically 6–12)."
        )
        globe_hint.setWordWrap(True)
        globe_hint.setStyleSheet("color: #777; font-style: italic;")
        globe_page = QWidget()
        gv = QVBoxLayout(globe_page)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.addLayout(globe_form)
        gv.addWidget(globe_hint)

        # Gate (index 1)
        self.gate_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.gate_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.gate_angle = QLineEdit("0")
        self.gate_angle.setPlaceholderText("degrees")
        gate_form = QFormLayout()
        gate_form.addRow("Seat bore D1:",        self.gate_D1.widget())
        gate_form.addRow("Pipe ID D2:",          self.gate_D2.widget())
        gate_form.addRow("Reducer angle (deg):", self.gate_angle)
        gate_hint = QLabel(
            "D1 ≤ D2.  Angle is the cone half-angle of the port reducer. "
            "Use 0 for a full-bore gate valve (K ≈ 8×fₐ, typically 0.1–0.2)."
        )
        gate_hint.setWordWrap(True)
        gate_hint.setStyleSheet("color: #777; font-style: italic;")
        gate_page = QWidget()
        gtv = QVBoxLayout(gate_page)
        gtv.setContentsMargins(0, 0, 0, 0)
        gtv.addLayout(gate_form)
        gtv.addWidget(gate_hint)

        # Butterfly (index 2)
        self.butterfly_D     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.butterfly_style = QComboBox()
        self.butterfly_style.addItems(
            ["Centric (0)", "Double offset (1)", "Triple offset (2)"]
        )
        butterfly_form = QFormLayout()
        butterfly_form.addRow("Pipe ID:", self.butterfly_D.widget())
        butterfly_form.addRow("Style:",   self.butterfly_style)
        butterfly_hint = QLabel(
            "Single pipe diameter (no reducer).  "
            'N factor by size: 45/74/218 (2"-8"), '
            '35/52/96 (10"-14"), 25/43/55 (16"-24").'
        )
        butterfly_hint.setWordWrap(True)
        butterfly_hint.setStyleSheet("color: #777; font-style: italic;")
        butterfly_page = QWidget()
        bfv = QVBoxLayout(butterfly_page)
        bfv.setContentsMargins(0, 0, 0, 0)
        bfv.addLayout(butterfly_form)
        bfv.addWidget(butterfly_hint)

        # Plug (index 3)
        self.plug_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.plug_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.plug_angle = QLineEdit("0")
        self.plug_angle.setPlaceholderText("degrees")
        self.plug_style = QComboBox()
        self.plug_style.addItems(
            ["Straight-through (0)", "3-way, flow straight (1)", "3-way, flow 90° (2)"]
        )
        plug_form = QFormLayout()
        plug_form.addRow("Plug bore D1:",        self.plug_D1.widget())
        plug_form.addRow("Pipe ID D2:",          self.plug_D2.widget())
        plug_form.addRow("Reducer angle (deg):", self.plug_angle)
        plug_form.addRow("Style:",               self.plug_style)
        plug_hint = QLabel(
            "D1 ≤ D2.  Use angle = 0 and D1 = D2 for a full-bore "
            "straight-through plug valve (K ≈ 18×fₐ)."
        )
        plug_hint.setWordWrap(True)
        plug_hint.setStyleSheet("color: #777; font-style: italic;")
        plug_page = QWidget()
        plv = QVBoxLayout(plug_page)
        plv.setContentsMargins(0, 0, 0, 0)
        plv.addLayout(plug_form)
        plv.addWidget(plug_hint)

        # Ball (index 4)
        self.ball_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.ball_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.ball_angle = QLineEdit("0")
        self.ball_angle.setPlaceholderText("degrees")
        ball_form = QFormLayout()
        ball_form.addRow("Seat bore D1:",        self.ball_D1.widget())
        ball_form.addRow("Pipe ID D2:",          self.ball_D2.widget())
        ball_form.addRow("Reducer angle (deg):", self.ball_angle)
        ball_hint = QLabel(
            "D1 ≤ D2.  Use 0 for a full-bore ball valve "
            "(K ≈ 3×fₐ, typically 0.05–0.10)."
        )
        ball_hint.setWordWrap(True)
        ball_hint.setStyleSheet("color: #777; font-style: italic;")
        ball_page = QWidget()
        blv = QVBoxLayout(ball_page)
        blv.setContentsMargins(0, 0, 0, 0)
        blv.addLayout(ball_form)
        blv.addWidget(ball_hint)

        # User-specified (index 5)
        self.user_D     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.user_Dmin  = _LabeledField("", U.DIAMETER, "inch")
        self.user_Dmin.edit.setPlaceholderText("optional, for reduced trim")
        self.user_mode  = QComboBox()
        self.user_mode.addItems(["K", "Cv", "Kv"])
        self.user_mode.currentIndexChanged.connect(self._on_user_mode_changed)
        self.user_value = QLineEdit("1.0")
        self.user_value_label = QLabel("K value:")
        user_form = QFormLayout()
        user_form.addRow("Pipe ID:",      self.user_D.widget())
        user_form.addRow("Trim diameter:", self.user_Dmin.widget())
        user_form.addRow("Specify by:",   self.user_mode)
        user_form.addRow(self.user_value_label, self.user_value)
        user_hint = QLabel(
            "K is the dimensionless resistance coefficient referenced to "
            "the pipe velocity head.  Cv is the US flow coefficient "
            "[gpm/psi^0.5]; Kv is the metric flow coefficient "
            "[m^3/h/bar^0.5].  Cv and Kv are converted to K via "
            "K = 2.166e9 · D^4 / Cv^2 (D in m) and Cv = 1.156 · Kv. "
            "Trim diameter is the minimum (throat) diameter inside the "
            "valve; leave blank for a full-port valve.  Used to detect "
            "sonic choke at the trim."
        )
        user_hint.setWordWrap(True)
        user_hint.setStyleSheet("color: #777; font-style: italic;")
        user_page = QWidget()
        uv = QVBoxLayout(user_page)
        uv.setContentsMargins(0, 0, 0, 0)
        uv.addLayout(user_form)
        uv.addWidget(user_hint)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(globe_page)     # 0
        self.input_stack.addWidget(gate_page)      # 1
        self.input_stack.addWidget(butterfly_page) # 2
        self.input_stack.addWidget(plug_page)      # 3
        self.input_stack.addWidget(ball_page)      # 4
        self.input_stack.addWidget(user_page)      # 5

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.type_combo)
        layout.addWidget(self.input_stack)

    def _on_type_changed(self, idx):
        self.input_stack.setCurrentIndex(idx)

    def _on_user_mode_changed(self, idx):
        mode = ("K", "Cv", "Kv")[idx]
        self.user_value_label.setText(f"{mode} value:")

    def get_spec(self):
        vt = self._TYPES[self.type_combo.currentIndex()]
        if vt == "globe":
            return {
                "valve_type": vt,
                "D1_str":    self.globe_D1.edit.text().strip(),
                "D1_unit":   self.globe_D1.combo.currentText(),
                "D2_str":    self.globe_D2.edit.text().strip(),
                "D2_unit":   self.globe_D2.combo.currentText(),
                "angle_str": "", "D_str": "", "D_unit": "inch", "style_str": "",
            }
        if vt == "gate":
            return {
                "valve_type": vt,
                "D1_str":    self.gate_D1.edit.text().strip(),
                "D1_unit":   self.gate_D1.combo.currentText(),
                "D2_str":    self.gate_D2.edit.text().strip(),
                "D2_unit":   self.gate_D2.combo.currentText(),
                "angle_str": self.gate_angle.text().strip(),
                "D_str": "", "D_unit": "inch", "style_str": "",
            }
        if vt == "butterfly":
            return {
                "valve_type": vt,
                "D_str":     self.butterfly_D.edit.text().strip(),
                "D_unit":    self.butterfly_D.combo.currentText(),
                "style_str": str(self.butterfly_style.currentIndex()),
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch", "angle_str": "",
            }
        if vt == "plug":
            return {
                "valve_type": vt,
                "D1_str":    self.plug_D1.edit.text().strip(),
                "D1_unit":   self.plug_D1.combo.currentText(),
                "D2_str":    self.plug_D2.edit.text().strip(),
                "D2_unit":   self.plug_D2.combo.currentText(),
                "angle_str": self.plug_angle.text().strip(),
                "style_str": str(self.plug_style.currentIndex()),
                "D_str": "", "D_unit": "inch",
            }
        if vt == "ball":
            return {
                "valve_type": vt,
                "D1_str":    self.ball_D1.edit.text().strip(),
                "D1_unit":   self.ball_D1.combo.currentText(),
                "D2_str":    self.ball_D2.edit.text().strip(),
                "D2_unit":   self.ball_D2.combo.currentText(),
                "angle_str": self.ball_angle.text().strip(),
                "D_str": "", "D_unit": "inch", "style_str": "",
            }
        # user_specified
        return {
            "valve_type":    vt,
            "D_str":         self.user_D.edit.text().strip(),
            "D_unit":        self.user_D.combo.currentText(),
            "Dmin_str":      self.user_Dmin.edit.text().strip(),
            "Dmin_unit":     self.user_Dmin.combo.currentText(),
            "spec_mode":     ("K", "Cv", "Kv")[self.user_mode.currentIndex()],
            "spec_val_str":  self.user_value.text().strip(),
            "D1_str": "", "D1_unit": "inch",
            "D2_str": "", "D2_unit": "inch",
            "angle_str": "", "style_str": "",
        }

    def set_spec(self, spec):
        vt  = spec.get("valve_type", "globe")
        idx = self._TYPES.index(vt) if vt in self._TYPES else 0
        self.type_combo.setCurrentIndex(idx)
        self.input_stack.setCurrentIndex(idx)
        # D1/D2 pages
        for w1, w2 in [
            (self.globe_D1, self.globe_D2),
            (self.gate_D1,  self.gate_D2),
            (self.plug_D1,  self.plug_D2),
            (self.ball_D1,  self.ball_D2),
        ]:
            w1.edit.setText(spec.get("D1_str", "4.026"))
            w1.combo.setCurrentText(spec.get("D1_unit", "inch"))
            w2.edit.setText(spec.get("D2_str", "4.026"))
            w2.combo.setCurrentText(spec.get("D2_unit", "inch"))
        angle = spec.get("angle_str", "0")
        self.gate_angle.setText(angle)
        self.plug_angle.setText(angle)
        self.ball_angle.setText(angle)
        self.butterfly_D.edit.setText(spec.get("D_str", "4.026"))
        self.butterfly_D.combo.setCurrentText(spec.get("D_unit", "inch"))
        style_idx = int(spec.get("style_str", "0") or "0")
        self.butterfly_style.setCurrentIndex(style_idx)
        self.plug_style.setCurrentIndex(style_idx)
        self.user_D.edit.setText(spec.get("D_str", "4.026"))
        self.user_D.combo.setCurrentText(spec.get("D_unit", "inch"))
        self.user_Dmin.edit.setText(spec.get("Dmin_str", ""))
        self.user_Dmin.combo.setCurrentText(spec.get("Dmin_unit", "inch"))
        spec_mode = spec.get("spec_mode", "K")
        mode_idx  = {"K": 0, "Cv": 1, "Kv": 2}.get(spec_mode, 0)
        self.user_mode.setCurrentIndex(mode_idx)
        self.user_value_label.setText(f"{spec_mode} value:")
        self.user_value.setText(spec.get("spec_val_str", "1.0"))

    def build_component(self, valve_cls, name=None):
        """Build and return a Valve component from the current field values."""
        spec = self.get_spec()
        vt   = spec["valve_type"]
        nm   = name or ""

        def _qty(str_key, unit_key):
            s = spec.get(str_key, "").strip()
            if not s:
                raise ValueError(f"Valve: {str_key[:-4]} is required.")
            return ureg.Quantity(float(s), U.to_pint(spec.get(unit_key, "inch")))

        if vt == "butterfly":
            D_q   = _qty("D_str", "D_unit")
            style = int(spec.get("style_str", "0") or "0")
            K     = fluids.fittings.K_butterfly_valve_Crane(
                D=D_q.to("m").magnitude, style=style,
            )
            return valve_cls(Di=D_q, K=K, name=nm)

        if vt == "user_specified":
            D_q  = _qty("D_str", "D_unit")
            mode = spec.get("spec_mode", "K")
            s    = spec.get("spec_val_str", "").strip()
            if not s:
                raise ValueError(f"Valve: {mode} value is required.")
            val = float(s)
            dmin_s = spec.get("Dmin_str", "").strip()
            if dmin_s:
                Dmin_q = ureg.Quantity(
                    float(dmin_s),
                    U.to_pint(spec.get("Dmin_unit", "inch")),
                )
            else:
                Dmin_q = None
            if mode == "K":
                return valve_cls(Di=D_q, K=val, minimum_diameter=Dmin_q, name=nm)
            if mode == "Cv":
                return valve_cls(Di=D_q, Cv=val, minimum_diameter=Dmin_q, name=nm)
            return valve_cls(Di=D_q, Kv=val, minimum_diameter=Dmin_q, name=nm)

        D1_q  = _qty("D1_str", "D1_unit")
        D2_q  = _qty("D2_str", "D2_unit")
        D1_si = D1_q.to("m").magnitude
        D2_si = D2_q.to("m").magnitude
        angle = float(spec.get("angle_str", "0").strip() or "0")

        if vt == "globe":
            K = fluids.fittings.K_globe_valve_Crane(D1=D1_si, D2=D2_si)
        elif vt == "gate":
            K = fluids.fittings.K_gate_valve_Crane(D1=D1_si, D2=D2_si, angle=angle)
        elif vt == "plug":
            style = int(spec.get("style_str", "0") or "0")
            K = fluids.fittings.K_plug_valve_Crane(
                D1=D1_si, D2=D2_si, angle=angle, style=style,
            )
        else:  # ball
            K = fluids.fittings.K_ball_valve_Crane(D1=D1_si, D2=D2_si, angle=angle)

        return valve_cls(Di=D2_q, K=K, name=nm)


# ---------------------------------------------------------------------------
# CheckValvePropertiesEditor
# ---------------------------------------------------------------------------

class CheckValvePropertiesEditor(QWidget):
    """Check valve type combo + per-type input stack (five Crane CV types)."""

    _TYPES  = ["swing", "lift", "tilting_disk", "angle_stop", "globe_stop"]
    _LABELS = ["Swing", "Lift", "Tilting disk", "Angle stop", "Globe stop"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.type_combo = QComboBox()
        self.type_combo.addItems(self._LABELS)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        # Swing (index 0)
        self.swing_D      = _LabeledField("4.026", U.DIAMETER, "inch")
        self.swing_angled = QComboBox()
        self.swing_angled.addItems(["Angled body", "Straight body"])
        swing_form = QFormLayout()
        swing_form.addRow("Pipe ID:", self.swing_D.widget())
        swing_form.addRow("Body:",    self.swing_angled)
        swing_hint = QLabel(
            "Crane K = 100×fₐ (angled) or 600×fₐ (straight).  "
            "Typical angled K ≈ 2–3."
        )
        swing_hint.setWordWrap(True)
        swing_hint.setStyleSheet("color: #777; font-style: italic;")
        swing_page = QWidget()
        sv = QVBoxLayout(swing_page)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addLayout(swing_form)
        sv.addWidget(swing_hint)

        # Lift (index 1)
        self.lift_D1     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.lift_D2     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.lift_angled = QComboBox()
        self.lift_angled.addItems(["Angled body", "Straight body"])
        lift_form = QFormLayout()
        lift_form.addRow("Seat bore D1:", self.lift_D1.widget())
        lift_form.addRow("Pipe ID D2:",   self.lift_D2.widget())
        lift_form.addRow("Body:",         self.lift_angled)
        lift_hint = QLabel(
            "D1 ≤ D2.  Crane K = 55×fₐ (angled) or 600×fₐ (straight) "
            "at D1 = D2; scaled by (D2/D1)^4 for reducers."
        )
        lift_hint.setWordWrap(True)
        lift_hint.setStyleSheet("color: #777; font-style: italic;")
        lift_page = QWidget()
        lv = QVBoxLayout(lift_page)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(lift_form)
        lv.addWidget(lift_hint)

        # Tilting disk (index 2)
        self.tilt_D     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.tilt_angle = QLineEdit("5")
        self.tilt_angle.setPlaceholderText("degrees")
        tilt_form = QFormLayout()
        tilt_form.addRow("Pipe ID:",          self.tilt_D.widget())
        tilt_form.addRow("Disk angle (deg):", self.tilt_angle)
        tilt_hint = QLabel(
            "Disk angle from centerline (5°, 10°, or 15°).  "
            "Crane K ≈ 0.5–1.5 depending on size and angle."
        )
        tilt_hint.setWordWrap(True)
        tilt_hint.setStyleSheet("color: #777; font-style: italic;")
        tilt_page = QWidget()
        tv = QVBoxLayout(tilt_page)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addLayout(tilt_form)
        tv.addWidget(tilt_hint)

        # Angle stop (index 3)
        self.angstop_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.angstop_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.angstop_style = QComboBox()
        self.angstop_style.addItems(
            ["Style 0 (piston)", "Style 1 (no stem guide)"]
        )
        angstop_form = QFormLayout()
        angstop_form.addRow("Seat bore D1:", self.angstop_D1.widget())
        angstop_form.addRow("Pipe ID D2:",   self.angstop_D2.widget())
        angstop_form.addRow("Style:",        self.angstop_style)
        angstop_hint = QLabel(
            "D1 ≤ D2.  Crane K = 55×fₐ (style 0) or 150×fₐ (style 1) "
            "at D1 = D2; scaled by (D2/D1)^4 for reducers."
        )
        angstop_hint.setWordWrap(True)
        angstop_hint.setStyleSheet("color: #777; font-style: italic;")
        angstop_page = QWidget()
        av = QVBoxLayout(angstop_page)
        av.setContentsMargins(0, 0, 0, 0)
        av.addLayout(angstop_form)
        av.addWidget(angstop_hint)

        # Globe stop (index 4)
        self.globestop_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.globestop_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.globestop_style = QComboBox()
        self.globestop_style.addItems(
            ["Style 0 (piston)", "Style 1 (no stem guide)"]
        )
        globestop_form = QFormLayout()
        globestop_form.addRow("Seat bore D1:", self.globestop_D1.widget())
        globestop_form.addRow("Pipe ID D2:",   self.globestop_D2.widget())
        globestop_form.addRow("Style:",        self.globestop_style)
        globestop_hint = QLabel(
            "D1 ≤ D2.  Crane K = 340×fₐ (style 0) or 600×fₐ (style 1) "
            "at D1 = D2; scaled by (D2/D1)^4 for reducers."
        )
        globestop_hint.setWordWrap(True)
        globestop_hint.setStyleSheet("color: #777; font-style: italic;")
        globestop_page = QWidget()
        gv = QVBoxLayout(globestop_page)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.addLayout(globestop_form)
        gv.addWidget(globestop_hint)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(swing_page)     # 0
        self.input_stack.addWidget(lift_page)      # 1
        self.input_stack.addWidget(tilt_page)      # 2
        self.input_stack.addWidget(angstop_page)   # 3
        self.input_stack.addWidget(globestop_page) # 4

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.type_combo)
        layout.addWidget(self.input_stack)

    def _on_type_changed(self, idx):
        self.input_stack.setCurrentIndex(idx)

    def get_spec(self):
        cvt = self._TYPES[self.type_combo.currentIndex()]
        angled_str = "0" if (
            self.swing_angled.currentIndex() == 1
            or self.lift_angled.currentIndex() == 1
        ) else "1"
        if cvt == "swing":
            return {
                "cv_type":    cvt,
                "D_str":      self.swing_D.edit.text().strip(),
                "D_unit":     self.swing_D.combo.currentText(),
                "angled_str": angled_str,
                "angle_str": "", "style_str": "",
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch",
            }
        if cvt == "lift":
            return {
                "cv_type":    cvt,
                "D1_str":     self.lift_D1.edit.text().strip(),
                "D1_unit":    self.lift_D1.combo.currentText(),
                "D2_str":     self.lift_D2.edit.text().strip(),
                "D2_unit":    self.lift_D2.combo.currentText(),
                "angled_str": angled_str,
                "D_str": "", "D_unit": "inch",
                "angle_str": "", "style_str": "",
            }
        if cvt == "tilting_disk":
            return {
                "cv_type":    cvt,
                "D_str":      self.tilt_D.edit.text().strip(),
                "D_unit":     self.tilt_D.combo.currentText(),
                "angle_str":  self.tilt_angle.text().strip(),
                "angled_str": "", "style_str": "",
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch",
            }
        if cvt == "angle_stop":
            return {
                "cv_type":    cvt,
                "D1_str":     self.angstop_D1.edit.text().strip(),
                "D1_unit":    self.angstop_D1.combo.currentText(),
                "D2_str":     self.angstop_D2.edit.text().strip(),
                "D2_unit":    self.angstop_D2.combo.currentText(),
                "style_str":  str(self.angstop_style.currentIndex()),
                "D_str": "", "D_unit": "inch",
                "angle_str": "", "angled_str": "",
            }
        # globe_stop
        return {
            "cv_type":    cvt,
            "D1_str":     self.globestop_D1.edit.text().strip(),
            "D1_unit":    self.globestop_D1.combo.currentText(),
            "D2_str":     self.globestop_D2.edit.text().strip(),
            "D2_unit":    self.globestop_D2.combo.currentText(),
            "style_str":  str(self.globestop_style.currentIndex()),
            "D_str": "", "D_unit": "inch",
            "angle_str": "", "angled_str": "",
        }

    def set_spec(self, spec):
        cvt = spec.get("cv_type", "swing")
        idx = self._TYPES.index(cvt) if cvt in self._TYPES else 0
        self.type_combo.setCurrentIndex(idx)
        self.input_stack.setCurrentIndex(idx)
        self.swing_D.edit.setText(spec.get("D_str", "4.026"))
        self.swing_D.combo.setCurrentText(spec.get("D_unit", "inch"))
        self.tilt_D.edit.setText(spec.get("D_str", "4.026"))
        self.tilt_D.combo.setCurrentText(spec.get("D_unit", "inch"))
        self.tilt_angle.setText(spec.get("angle_str", "5"))
        angled_idx = 0 if spec.get("angled_str", "1") == "1" else 1
        self.swing_angled.setCurrentIndex(angled_idx)
        self.lift_angled.setCurrentIndex(angled_idx)
        for w1, w2 in [
            (self.lift_D1,      self.lift_D2),
            (self.angstop_D1,   self.angstop_D2),
            (self.globestop_D1, self.globestop_D2),
        ]:
            w1.edit.setText(spec.get("D1_str", "4.026"))
            w1.combo.setCurrentText(spec.get("D1_unit", "inch"))
            w2.edit.setText(spec.get("D2_str", "4.026"))
            w2.combo.setCurrentText(spec.get("D2_unit", "inch"))
        style_idx = int(spec.get("style_str", "0") or "0")
        self.angstop_style.setCurrentIndex(style_idx)
        self.globestop_style.setCurrentIndex(style_idx)

    def build_component(self, cv_cls, name=None):
        """Build and return a CheckValve component from the current field values."""
        spec = self.get_spec()
        cvt  = spec["cv_type"]
        nm   = name or ""

        def _qty(str_key, unit_key):
            s = spec.get(str_key, "").strip()
            if not s:
                raise ValueError(f"Check valve: {str_key[:-4]} is required.")
            return ureg.Quantity(float(s), U.to_pint(spec.get(unit_key, "inch")))

        if cvt == "swing":
            D_q    = _qty("D_str", "D_unit")
            angled = spec.get("angled_str", "1") != "0"
            K      = fluids.fittings.K_swing_check_valve_Crane(
                D=D_q.to("m").magnitude, angled=angled,
            )
            return cv_cls(Di=D_q, K=K, name=nm)
        if cvt == "lift":
            D1_q   = _qty("D1_str", "D1_unit")
            D2_q   = _qty("D2_str", "D2_unit")
            angled = spec.get("angled_str", "1") != "0"
            K      = fluids.fittings.K_lift_check_valve_Crane(
                D1=D1_q.to("m").magnitude, D2=D2_q.to("m").magnitude, angled=angled,
            )
            return cv_cls(Di=D2_q, K=K, name=nm)
        if cvt == "tilting_disk":
            D_q   = _qty("D_str", "D_unit")
            angle = float(spec.get("angle_str", "5").strip() or "5")
            K     = fluids.fittings.K_tilting_disk_check_valve_Crane(
                D=D_q.to("m").magnitude, angle=angle,
            )
            return cv_cls(Di=D_q, K=K, name=nm)
        if cvt == "angle_stop":
            D1_q  = _qty("D1_str", "D1_unit")
            D2_q  = _qty("D2_str", "D2_unit")
            style = int(spec.get("style_str", "0") or "0")
            K     = fluids.fittings.K_angle_stop_check_valve_Crane(
                D1=D1_q.to("m").magnitude, D2=D2_q.to("m").magnitude, style=style,
            )
            return cv_cls(Di=D2_q, K=K, name=nm)
        # globe_stop
        D1_q  = _qty("D1_str", "D1_unit")
        D2_q  = _qty("D2_str", "D2_unit")
        style = int(spec.get("style_str", "0") or "0")
        K     = fluids.fittings.K_globe_stop_check_valve_Crane(
            D1=D1_q.to("m").magnitude, D2=D2_q.to("m").magnitude, style=style,
        )
        return cv_cls(Di=D2_q, K=K, name=nm)
