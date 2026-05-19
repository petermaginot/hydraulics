"""Network screen: P&ID-style canvas.

The canvas carries three node types, plus zero-info connector edges:

  SourceSinkNode : boundary node carrying a P or Q_ext spec (mutually
                   exclusive); multi-in / multi-out ports.
  JunctionNode   : interior splitter/merger with no boundary condition;
                   multi-in / multi-out ports.
  PipeSegmentNode: inline pipe-with-geometry; exactly one input + one
                   output (no branching).  Holds the ID/OD/WT/length/dz/
                   roughness needed for a Line_Segment.

A "pipe" between two boundary nodes in the underlying Network solver is
assembled at solve time by walking the chain of PipeSegmentNodes between
them.  PipeSegmentNodes are not represented as solver nodes; their
components become the component list on the constructed solver Edge.

Layout:
    [+ Source/Sink] [+ Junction] [+ Pipe] [- Delete selected] [Solve]
    +------------------------------+--------------------+
    |                              | Fluid              |
    |                              | Selected node      |
    |    NodeGraphQt canvas        | Display units      |
    |                              | Results            |
    +------------------------------+--------------------+
    [<- Back]
"""

import gui._compat   # noqa: F401  -- must import before NodeGraphQt

import os
import traceback

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from NodeGraphQt import BaseNode, NodeGraph

from component_classes import ureg
from gui import units as U
from gui.screens.segment import _LabeledField
from incompressible import Incompressible_Fluid, Line_Segment
from network import Network


# NodeGraphQt node type identifiers.
SOURCE_SINK_TYPE_ID  = "pipe.SourceSinkNode"
JUNCTION_TYPE_ID     = "pipe.JunctionNode"
PIPE_SEGMENT_TYPE_ID = "pipe.PipeSegmentNode"

# Pint dimensionalities used to dispatch flow display units to either the
# mass-flow source (mdot_kgs) or the volumetric-flow source (Q_m3s).
_MASS_FLOW_DIM = ureg.Quantity(1.0, "kg/s").dimensionality
_VOL_FLOW_DIM  = ureg.Quantity(1.0, "m^3/s").dimensionality


# ---------------------------------------------------------------------------
# NodeGraphQt node subclasses
# ---------------------------------------------------------------------------

class SourceSinkNode(BaseNode):
    """Boundary node carrying a P or Q_ext spec.

    Multi-connection in and out ports so a single Source/Sink can serve
    as the start/end of multiple chains (e.g. a manifold).  Result widgets
    show the solved pressure and the (specified or derived) external flow.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Source/Sink"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=True)
        self.add_output("out", multi_output=True)
        self.add_text_input(
            "P_result", label="", text="",
            placeholder_text="(solve to see)",
        )
        self.add_text_input("Q_result", label="", text="")


class JunctionNode(BaseNode):
    """Interior splitter / merger.

    No boundary condition; the only spec the user can set is the
    (decorative) elevation.  Result widget shows the solved pressure.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Junction"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=True)
        self.add_output("out", multi_output=True)
        self.add_text_input(
            "P_result", label="", text="",
            placeholder_text="(solve to see)",
        )


class PipeSegmentNode(BaseNode):
    """Inline pipe segment with geometry.

    Single in + single out (no branching).  Stored geometry feeds a
    Line_Segment built at solve time and dropped into the solver Edge's
    component list.  Result widgets show the segment's dP and signed flow.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Pipe Segment"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=False)
        self.add_output("out", multi_output=False)
        self.add_text_input(
            "dP_result", label="", text="",
            placeholder_text="(solve to see)",
        )
        self.add_text_input("Q_result", label="", text="")


# ---------------------------------------------------------------------------
# NetworkScreen
# ---------------------------------------------------------------------------

class NetworkScreen(QWidget):
    back_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        # Per-node spec dicts.  Each value is one of _default_source_sink_fields
        # / _default_junction_fields / _default_pipe_fields, keyed by node id.
        # The "type" entry inside each dict tags which family the spec
        # belongs to, mirroring the NodeGraphQt class.
        self.node_specs = {}
        # Maps a PipeSegmentNode id to the Line_Segment instance built at the
        # last solve, so result.component(seg) can be looked up to render
        # per-pipe dP and flow on the canvas.
        self._pipe_components = {}
        # Last successful solve, retained so the display-unit selectors can
        # re-render the canvas + text panel without re-solving.
        self._last_net    = None
        self._last_result = None
        # Currently-edited node, tracked so Apply writes back to the right
        # spec dict.
        self._cur_node = None

        # ---- Graph engine ----
        self.graph = NodeGraph()
        self.graph.register_node(SourceSinkNode)
        self.graph.register_node(JunctionNode)
        self.graph.register_node(PipeSegmentNode)
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)

        # ---- Toolbar ----
        add_src_btn  = QPushButton("+ Source/Sink")
        add_src_btn.clicked.connect(self._add_source_sink)
        add_jct_btn  = QPushButton("+ Junction")
        add_jct_btn.clicked.connect(self._add_junction)
        add_pipe_btn = QPushButton("+ Pipe")
        add_pipe_btn.clicked.connect(self._add_pipe_segment)
        del_btn      = QPushButton("- Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        solve_btn    = QPushButton("Solve")
        solve_btn.clicked.connect(self._solve)

        toolbar = QHBoxLayout()
        toolbar.addWidget(add_src_btn)
        toolbar.addWidget(add_jct_btn)
        toolbar.addWidget(add_pipe_btn)
        toolbar.addWidget(del_btn)
        toolbar.addStretch()
        toolbar.addWidget(solve_btn)

        # ---- Side panel ----
        self._build_fluid_box()
        self._build_editor_stack()
        self._build_display_units_box()
        self._build_results_box()

        side = QVBoxLayout()
        side.addWidget(self.fluid_box)
        side.addWidget(self.editor_box)
        side.addWidget(self.display_box)
        side.addWidget(self.results_box, 1)

        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setMaximumWidth(420)

        center_row = QHBoxLayout()
        center_row.addWidget(self.graph.widget, 3)
        center_row.addWidget(side_wrap, 1)

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.back_clicked.emit)
        nav = QHBoxLayout()
        nav.addWidget(back_btn)
        nav.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addLayout(center_row, 1)
        layout.addLayout(nav)

        # No selection at startup -> show empty-state hint.
        self._sync_editor_to_selection()

    # ------------------------------------------------------------------
    # Default spec dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _default_source_sink_fields():
        return {
            "type":      "source_sink",
            "P_str":     "",   "P_unit":    "psi",
            "Q_str":     "",   "Q_unit":    "BBL/D",
            "elev_str":  "0",  "elev_unit": "ft",
        }

    @staticmethod
    def _default_junction_fields():
        return {
            "type":      "junction",
            "elev_str":  "0",  "elev_unit": "ft",
        }

    @staticmethod
    def _default_pipe_fields():
        # dz is a per-pipe input now (vs the old edges-with-Line_Segment
        # model, where it was derived from endpoint node elevations).
        return {
            "type":       "pipe",
            "mode":       "manual",
            "ID_str":     "4.026",   "ID_unit":    "inch",
            "OD_str":     "",        "OD_unit":    "inch",
            "WT_str":     "",        "WT_unit":    "inch",
            "L_str":      "1000",    "L_unit":     "ft",
            "dz_str":     "0",       "dz_unit":    "ft",
            "rough_str":  "0.00015", "rough_unit": "ft",
        }

    # ------------------------------------------------------------------
    # Side-panel builders
    # ------------------------------------------------------------------

    def _build_fluid_box(self):
        self.f_density   = _LabeledField("62.4", U.DENSITY,   "lb/ft^3")
        self.f_viscosity = _LabeledField("1.0",  U.VISCOSITY, "cP")
        form = QFormLayout()
        form.addRow("Density:",   self.f_density.widget())
        form.addRow("Viscosity:", self.f_viscosity.widget())
        self.fluid_box = QGroupBox("Fluid (incompressible)")
        self.fluid_box.setLayout(form)

    def _build_editor_stack(self):
        """Selection-aware editor.  Four pages keyed by node type, swapped
        via QStackedWidget when a node is selected.
        """
        self.editor_stack = QStackedWidget()
        self._build_empty_editor()      # index 0
        self._build_source_sink_editor()  # index 1
        self._build_junction_editor()     # index 2
        self._build_pipe_editor()         # index 3

        self.editor_box = QGroupBox("Selected node")
        v = QVBoxLayout(self.editor_box)
        v.addWidget(self.editor_stack)

    def _build_empty_editor(self):
        w = QLabel(
            "Click a node on the canvas to edit it.  Use the toolbar above "
            "to add a Source/Sink, Junction, or Pipe Segment, then drag "
            "between ports to connect them.  Solve when ready."
        )
        w.setWordWrap(True)
        w.setStyleSheet("color: #777; font-style: italic;")
        self.editor_stack.addWidget(w)

    def _build_source_sink_editor(self):
        self.ss_name = QLineEdit()
        self.ss_P    = _LabeledField("",  U.PRESSURE,                "psi")
        self.ss_Q    = _LabeledField("",  U.FLOW_RATE_INCOMPRESSIBLE, "BBL/D")
        self.ss_elev = _LabeledField("0", U.LENGTH,                  "ft")
        self.ss_apply = QPushButton("Apply")
        self.ss_apply.clicked.connect(self._apply_source_sink_edits)

        self.ss_P.edit.setToolTip(
            "Specified pressure boundary.  Mutually exclusive with Q_ext: "
            "each node has one mass-balance equation with one unknown."
        )
        self.ss_Q.edit.setToolTip(
            "Specified external supply (+) or withdrawal (-).  Mutually "
            "exclusive with P at this node."
        )
        self.ss_P.edit.textChanged.connect(self._update_PQ_exclusivity)
        self.ss_Q.edit.textChanged.connect(self._update_PQ_exclusivity)

        form = QFormLayout()
        form.addRow("Name:",       self.ss_name)
        form.addRow("P:",          self.ss_P.widget())
        form.addRow("Q_ext:",      self.ss_Q.widget())
        form.addRow("Elevation:",  self.ss_elev.widget())

        hint = QLabel(
            "Specify P OR Q_ext (never both) — the solver derives the "
            "other from mass balance.  Q_ext > 0 = supply into the "
            "network.  Elevation is informational."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-style: italic;")

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(form)
        v.addWidget(hint)
        v.addWidget(self.ss_apply)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_junction_editor(self):
        self.j_name  = QLineEdit()
        self.j_elev  = _LabeledField("0", U.LENGTH, "ft")
        self.j_apply = QPushButton("Apply")
        self.j_apply.clicked.connect(self._apply_junction_edits)

        form = QFormLayout()
        form.addRow("Name:",       self.j_name)
        form.addRow("Elevation:",  self.j_elev.widget())

        hint = QLabel(
            "Interior splitter / merger.  No boundary condition: the "
            "solver derives pressure from mass balance with neighbouring "
            "edges.  Elevation is informational."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-style: italic;")

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(form)
        v.addWidget(hint)
        v.addWidget(self.j_apply)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_pipe_editor(self):
        self.p_name = QLineEdit()
        self.p_csv_btn = QPushButton("Load CSV...")
        self.p_csv_btn.clicked.connect(self._on_load_pipe_csv)
        self.p_clear_csv_btn = QPushButton("Switch to manual")
        self.p_clear_csv_btn.clicked.connect(self._on_clear_pipe_csv)
        self.p_clear_csv_btn.setVisible(False)
        self.p_source_label = QLabel("(manual input)")
        self.p_source_label.setStyleSheet("color: #555; font-style: italic;")
        self.p_source_label.setWordWrap(True)

        self.p_id     = _LabeledField("4.026",   U.DIAMETER,  "inch")
        self.p_od     = _LabeledField("",        U.DIAMETER,  "inch")
        self.p_wt     = _LabeledField("",        U.DIAMETER,  "inch")
        self.p_length = _LabeledField("1000",    U.LENGTH,    "ft")
        self.p_dz     = _LabeledField("0",       U.LENGTH,    "ft")
        self.p_rough  = _LabeledField("0.00015", U.ROUGHNESS, "ft")
        self.p_apply  = QPushButton("Apply")
        self.p_apply.clicked.connect(self._apply_pipe_edits)

        csv_row = QHBoxLayout()
        csv_row.addWidget(self.p_csv_btn)
        csv_row.addWidget(self.p_clear_csv_btn)
        csv_row.addStretch()

        form = QFormLayout()
        form.addRow("Name:",     self.p_name)
        form.addRow("Source:",   self.p_source_label)
        form.addRow("ID:",       self.p_id.widget())
        form.addRow("OR OD:",    self.p_od.widget())
        form.addRow("WT:",       self.p_wt.widget())
        form.addRow("Length:",   self.p_length.widget())
        form.addRow("dz:",       self.p_dz.widget())
        form.addRow("Roughness:", self.p_rough.widget())

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(form)
        v.addLayout(csv_row)
        v.addWidget(self.p_apply)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_display_units_box(self):
        """Pressure + flow unit combos that re-render the last solve
        without re-running it (mirrors the point-to-point Results screen).
        """
        self.d_pressure = QComboBox()
        self.d_pressure.addItems(U.PRESSURE)
        self.d_pressure.setCurrentText("psi")
        self.d_pressure.currentTextChanged.connect(
            self._rerender_with_current_units
        )
        self.d_flow = QComboBox()
        self.d_flow.addItems(U.FLOW_RATE_INCOMPRESSIBLE)
        self.d_flow.setCurrentText("BBL/D")
        self.d_flow.currentTextChanged.connect(
            self._rerender_with_current_units
        )

        form = QFormLayout()
        form.addRow("Pressure:", self.d_pressure)
        form.addRow("Flow:",     self.d_flow)
        self.display_box = QGroupBox("Result display units")
        self.display_box.setLayout(form)

    def _build_results_box(self):
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("(no results yet)")
        self.results_text.setStyleSheet("font-family: monospace;")
        self.results_box = QGroupBox("Results")
        v = QVBoxLayout(self.results_box)
        v.addWidget(self.results_text)

    # ------------------------------------------------------------------
    # Graph mutation: add / delete nodes
    # ------------------------------------------------------------------

    def _add_source_sink(self):
        n = self.graph.create_node(SOURCE_SINK_TYPE_ID)
        self.node_specs[n.id] = self._default_source_sink_fields()

    def _add_junction(self):
        n = self.graph.create_node(JUNCTION_TYPE_ID)
        self.node_specs[n.id] = self._default_junction_fields()

    def _add_pipe_segment(self):
        n = self.graph.create_node(PIPE_SEGMENT_TYPE_ID)
        self.node_specs[n.id] = self._default_pipe_fields()

    def _delete_selected(self):
        nodes = self.graph.selected_nodes()
        if nodes:
            self.graph.delete_nodes(nodes)

    def _on_nodes_deleted(self, node_ids):
        node_ids = set(node_ids)
        for nid in node_ids:
            self.node_specs.pop(nid, None)
            self._pipe_components.pop(nid, None)
        if self._cur_node is not None and self._cur_node.id in node_ids:
            self._cur_node = None
        self._sync_editor_to_selection()

    # ------------------------------------------------------------------
    # Node-selection sync (drives the editor stack)
    # ------------------------------------------------------------------

    def _on_node_selected(self, node):
        self._cur_node = node
        self._sync_editor_to_selection()

    def _sync_editor_to_selection(self):
        """Switch the stacked editor to match the currently-selected
        node's type and populate its fields from node_specs.
        """
        n = self._cur_node
        if n is None or n.id not in self.node_specs:
            self.editor_stack.setCurrentIndex(0)
            self.editor_box.setTitle("Selected node: (none)")
            return
        spec = self.node_specs[n.id]
        t = spec.get("type")
        if t == "source_sink":
            self.editor_stack.setCurrentIndex(1)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Source/Sink]")
            self.ss_name.setText(n.name())
            self.ss_P.edit.setText(spec.get("P_str", ""))
            self.ss_P.combo.setCurrentText(spec.get("P_unit", "psi"))
            self.ss_Q.edit.setText(spec.get("Q_str", ""))
            self.ss_Q.combo.setCurrentText(spec.get("Q_unit", "BBL/D"))
            self.ss_elev.edit.setText(spec.get("elev_str", "0"))
            self.ss_elev.combo.setCurrentText(spec.get("elev_unit", "ft"))
            self._update_PQ_exclusivity()
        elif t == "junction":
            self.editor_stack.setCurrentIndex(2)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Junction]")
            self.j_name.setText(n.name())
            self.j_elev.edit.setText(spec.get("elev_str", "0"))
            self.j_elev.combo.setCurrentText(spec.get("elev_unit", "ft"))
        elif t == "pipe":
            self.editor_stack.setCurrentIndex(3)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Pipe Segment]")
            self.p_name.setText(n.name())
            self._sync_pipe_editor_to_spec(spec)
        else:
            self.editor_stack.setCurrentIndex(0)

    def _sync_pipe_editor_to_spec(self, spec):
        # Roughness is always user-editable, regardless of mode.
        self.p_rough.edit.setText(spec.get("rough_str", "0.00015"))
        self.p_rough.combo.setCurrentText(spec.get("rough_unit", "ft"))
        mode = spec.get("mode", "manual")
        if mode == "csv":
            self.p_clear_csv_btn.setVisible(True)
            path    = spec.get("csv_path", "")
            profile = spec.get("csv_profile", [])
            L_m  = (profile[-1][0] - profile[0][0]) if profile else 0.0
            L_ft = ureg.Quantity(L_m, "m").to("ft").magnitude
            self.p_source_label.setText(
                f"CSV: {os.path.basename(path)}  "
                f"({len(profile)} pts, {L_ft:.1f} ft)"
            )
            # Lock the manual geometry inputs and surface CSV-derived
            # summaries instead of stale manual values.
            for field in (self.p_id, self.p_od, self.p_wt, self.p_length, self.p_dz):
                self._set_field_editable(field, False)
            if profile:
                id_first_in = ureg.Quantity(profile[0][2], "m").to("inch").magnitude
                id_last_in  = ureg.Quantity(profile[-1][2], "m").to("inch").magnitude
                if abs(id_first_in - id_last_in) < 1e-6:
                    self.p_id.edit.setText(f"{id_first_in:g}")
                else:
                    self.p_id.edit.setText(f"{id_first_in:g} - {id_last_in:g}")
                self.p_id.combo.setCurrentText("inch")
                self.p_od.edit.setText("")
                self.p_wt.edit.setText("")
                self.p_length.edit.setText(f"{L_ft:g}")
                self.p_length.combo.setCurrentText("ft")
                dz_ft = ureg.Quantity(
                    profile[-1][1] - profile[0][1], "m",
                ).to("ft").magnitude
                self.p_dz.edit.setText(f"{dz_ft:g}")
                self.p_dz.combo.setCurrentText("ft")
        else:
            self.p_clear_csv_btn.setVisible(False)
            self.p_source_label.setText("(manual input)")
            for field, name, default_unit in (
                (self.p_id,     "ID",  "inch"),
                (self.p_od,     "OD",  "inch"),
                (self.p_wt,     "WT",  "inch"),
                (self.p_length, "L",   "ft"),
                (self.p_dz,     "dz",  "ft"),
            ):
                self._set_field_editable(field, True)
                field.edit.setText(spec.get(f"{name}_str", ""))
                field.combo.setCurrentText(spec.get(f"{name}_unit", default_unit))

    @staticmethod
    def _set_field_editable(field, editable):
        field.edit.setReadOnly(not editable)
        field.combo.setEnabled(editable)
        if editable:
            field.edit.setStyleSheet("")
        else:
            field.edit.setStyleSheet("background-color: #f0f0f0; color: #555;")

    # ------------------------------------------------------------------
    # Apply edits (write editor state back to node_specs)
    # ------------------------------------------------------------------

    def _apply_source_sink_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "source_sink":
            return
        new_name = self.ss_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        self.node_specs[n.id] = {
            "type":      "source_sink",
            "P_str":     self.ss_P.edit.text().strip(),
            "P_unit":    self.ss_P.combo.currentText(),
            "Q_str":     self.ss_Q.edit.text().strip(),
            "Q_unit":    self.ss_Q.combo.currentText(),
            "elev_str":  self.ss_elev.edit.text().strip() or "0",
            "elev_unit": self.ss_elev.combo.currentText(),
        }
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Source/Sink]")

    def _apply_junction_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "junction":
            return
        new_name = self.j_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        self.node_specs[n.id] = {
            "type":      "junction",
            "elev_str":  self.j_elev.edit.text().strip() or "0",
            "elev_unit": self.j_elev.combo.currentText(),
        }
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Junction]")

    def _apply_pipe_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "pipe":
            return
        new_name = self.p_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        spec = self.node_specs[n.id]
        # Roughness is always persistable.
        spec["rough_str"]  = self.p_rough.edit.text().strip()
        spec["rough_unit"] = self.p_rough.combo.currentText()
        # In CSV mode, the geometry fields show CSV-derived summaries that
        # would be lossy to write back into the manual storage; only the
        # manual mode persists them.
        if spec.get("mode", "manual") == "manual":
            for field, name in (
                (self.p_id,     "ID"),
                (self.p_od,     "OD"),
                (self.p_wt,     "WT"),
                (self.p_length, "L"),
                (self.p_dz,     "dz"),
            ):
                spec[f"{name}_str"]  = field.edit.text().strip()
                spec[f"{name}_unit"] = field.combo.currentText()
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Pipe Segment]")

    def _update_PQ_exclusivity(self):
        """Disable whichever of P / Q_ext on the Source/Sink editor the
        user isn't using.  Triggered by textChanged on either field and
        also called from _sync_editor_to_selection so the disabled state
        matches the loaded spec on every selection change.
        """
        has_P = bool(self.ss_P.edit.text().strip())
        has_Q = bool(self.ss_Q.edit.text().strip())
        if has_P and has_Q:
            self._set_labeled_field_enabled(self.ss_P, True)
            self._set_labeled_field_enabled(self.ss_Q, True)
        else:
            self._set_labeled_field_enabled(self.ss_P, not has_Q)
            self._set_labeled_field_enabled(self.ss_Q, not has_P)

    @staticmethod
    def _set_labeled_field_enabled(field, enabled):
        field.edit.setEnabled(enabled)
        field.combo.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Pipe CSV load / clear
    # ------------------------------------------------------------------

    def _on_load_pipe_csv(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "pipe":
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load pipe profile CSV", "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            # Reuse Line_Segment.from_csv for parsing/validation; the temp
            # roughness value is discarded -- only the parsed profile
            # matters.  The real roughness is read from the editor at
            # solve time.
            tmp_seg = Line_Segment.from_csv(path, roughness=1e-6)
        except Exception as e:
            QMessageBox.critical(
                self, "Could not load CSV", f"{type(e).__name__}: {e}",
            )
            return
        profile = list(tmp_seg.profile)
        if not profile:
            QMessageBox.critical(self, "Could not load CSV",
                                 "CSV produced an empty profile.")
            return
        spec = self.node_specs[n.id]
        spec["mode"]        = "csv"
        spec["csv_path"]    = path
        spec["csv_profile"] = profile
        self._sync_pipe_editor_to_spec(spec)

    def _on_clear_pipe_csv(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "pipe":
            return
        spec = self.node_specs[n.id]
        spec["mode"] = "manual"
        spec.pop("csv_path", None)
        spec.pop("csv_profile", None)
        self._sync_pipe_editor_to_spec(spec)

    # ------------------------------------------------------------------
    # Solve: translate canvas graph into solver Network and run.
    # ------------------------------------------------------------------

    def _solve(self):
        try:
            fluid  = self._build_fluid()
            net    = self._build_network()
            result = net.solve(fluid)
        except Exception as e:
            QMessageBox.critical(
                self, "Solve failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        self._render_results(net, result)

    def _build_fluid(self):
        rho = self.f_density.quantity()
        mu  = self.f_viscosity.quantity()
        if rho is None or mu is None:
            raise ValueError("Density and viscosity are required.")
        return Incompressible_Fluid(density=rho, viscosity=mu)

    def _build_network(self):
        # Push any pending in-editor changes into the spec dicts so a
        # Solve without a prior Apply still uses what the user sees.
        if self._cur_node is not None:
            t = self.node_specs.get(self._cur_node.id, {}).get("type")
            if t == "source_sink":
                self._apply_source_sink_edits()
            elif t == "junction":
                self._apply_junction_edits()
            elif t == "pipe":
                self._apply_pipe_edits()

        # Partition canvas nodes by type.
        boundary_nodes = []
        pipe_nodes     = []
        for node in self.graph.all_nodes():
            t = self.node_specs.get(node.id, {}).get("type")
            if t in ("source_sink", "junction"):
                boundary_nodes.append(node)
            elif t == "pipe":
                pipe_nodes.append(node)

        if not boundary_nodes:
            raise ValueError(
                "Add at least one Source/Sink or Junction node."
            )

        # Check for duplicate node names since the solver keys by name.
        seen_names = set()
        for node in boundary_nodes:
            name = node.name()
            if name in seen_names:
                raise ValueError(
                    f"Two nodes share the name '{name}'.  Rename one."
                )
            seen_names.add(name)

        net = Network()
        self._pipe_components = {}

        # Add boundary nodes to the solver.
        for node in boundary_nodes:
            spec = self.node_specs[node.id]
            kwargs = self._kwargs_for_boundary(spec, node.name())
            net.add_node(node.name(), **kwargs)

        # Walk chains from each boundary node's outgoing connections.
        visited_pipes = set()
        for start_node in boundary_nodes:
            for out_port, targets in start_node.connected_output_nodes().items():
                for target in targets:
                    components, end_node = self._walk_chain(
                        target, visited_pipes, start_node.name(),
                    )
                    if end_node is None:
                        # Chain didn't terminate at a boundary node.
                        raise ValueError(
                            f"Pipe chain starting from '{start_node.name()}' "
                            f"does not terminate at a Source/Sink or "
                            f"Junction node."
                        )
                    edge_name = self._unique_edge_name(
                        net, start_node.name(), end_node.name(),
                    )
                    net.add_edge(
                        edge_name, start_node.name(), end_node.name(),
                        components,
                    )
        return net

    def _kwargs_for_boundary(self, spec, node_name):
        t = spec.get("type")
        kwargs = {}
        if t == "source_sink":
            P_str = spec.get("P_str", "").strip()
            Q_str = spec.get("Q_str", "").strip()
            if P_str and Q_str:
                raise ValueError(
                    f"Source/Sink '{node_name}': specify P or Q_ext, not both."
                )
            if P_str:
                kwargs["P"] = ureg.Quantity(
                    float(P_str), U.to_pint(spec.get("P_unit", "psi")),
                )
            if Q_str:
                kwargs["Q_ext"] = ureg.Quantity(
                    float(Q_str), U.to_pint(spec.get("Q_unit", "BBL/D")),
                )
        elev_str = spec.get("elev_str", "0").strip() or "0"
        kwargs["elevation"] = ureg.Quantity(
            float(elev_str), U.to_pint(spec.get("elev_unit", "ft")),
        )
        return kwargs

    def _walk_chain(self, target_node, visited_pipes, source_name):
        """Walk forward through pipe-type nodes starting at target_node,
        collecting their Line_Segment components, until hitting a
        boundary node.  Returns (component_list, end_boundary_node) or
        (component_list, None) if no boundary node terminates the chain.
        """
        components = []
        cur = target_node
        while True:
            t = self.node_specs.get(cur.id, {}).get("type")
            if t in ("source_sink", "junction"):
                return components, cur
            if t != "pipe":
                return components, None
            if cur.id in visited_pipes:
                raise ValueError(
                    f"Pipe '{cur.name()}' is part of more than one chain "
                    f"(should be a single in/out node)."
                )
            visited_pipes.add(cur.id)
            seg = self._build_line_segment_for_pipe(cur)
            self._pipe_components[cur.id] = seg
            components.append(seg)
            # Pipe nodes are single-out: at most one outgoing connection.
            outs = []
            for port, targets in cur.connected_output_nodes().items():
                outs.extend(targets)
            if not outs:
                raise ValueError(
                    f"Pipe '{cur.name()}' has no outgoing connection."
                )
            cur = outs[0]

    def _build_line_segment_for_pipe(self, pipe_node):
        spec = self.node_specs[pipe_node.id]
        rough_str  = spec.get("rough_str", "").strip()
        rough_unit = spec.get("rough_unit", "ft")
        if not rough_str:
            raise ValueError(
                f"Pipe '{pipe_node.name()}': roughness is required."
            )
        rough_q = ureg.Quantity(float(rough_str), U.to_pint(rough_unit))

        if spec.get("mode") == "csv":
            profile = spec.get("csv_profile", [])
            if not profile:
                raise ValueError(
                    f"Pipe '{pipe_node.name()}': CSV mode but no profile "
                    f"loaded."
                )
            return Line_Segment(roughness=rough_q, profile=profile)

        def _opt(name):
            s = spec.get(f"{name}_str", "").strip()
            if not s:
                return None
            return ureg.Quantity(
                float(s), U.to_pint(spec.get(f"{name}_unit", "")),
            )

        return Line_Segment(
            roughness        = rough_q,
            id_val           = _opt("ID"),
            od_val           = _opt("OD"),
            wt_val           = _opt("WT"),
            length           = _opt("L"),
            elevation_change = _opt("dz"),
        )

    @staticmethod
    def _unique_edge_name(net, from_name, to_name):
        base = f"{from_name}->{to_name}"
        existing = {e.name for e in net._edges}
        if base not in existing:
            return base
        i = 2
        while f"{base}#{i}" in existing:
            i += 1
        return f"{base}#{i}"

    # ------------------------------------------------------------------
    # Result rendering
    # ------------------------------------------------------------------

    def _render_results(self, net, result):
        self._last_net    = net
        self._last_result = result
        self._render_canvas_and_panel(net, result)

    def _rerender_with_current_units(self):
        if self._last_result is None:
            return
        self._render_canvas_and_panel(self._last_net, self._last_result)

    def _render_canvas_and_panel(self, net, result):
        self._annotate_results_on_canvas(result)
        self._render_results_text(result)

    def _annotate_results_on_canvas(self, result):
        P_unit = self.d_pressure.currentText()
        Q_unit = self.d_flow.currentText()

        P_dict        = result["P_Pa"]
        ext_mass_dict = result["mdot_ext_kgs"]
        ext_vol_dict  = result["Q_ext_m3s"]

        # Boundary nodes: P (Source/Sink + Junction), Q (Source/Sink only).
        for node in self.graph.all_nodes():
            spec = self.node_specs.get(node.id, {})
            t = spec.get("type")
            if t == "source_sink":
                name = node.name()
                P_Pa = P_dict.get(name)
                mass = ext_mass_dict.get(name)
                vol  = ext_vol_dict.get(name)
                self._set_widget(node, "P_result",
                                 self._format_pressure(P_Pa, P_unit))
                self._set_widget(node, "Q_result",
                                 self._format_flow(mass, vol, Q_unit))
            elif t == "junction":
                P_Pa = P_dict.get(node.name())
                self._set_widget(node, "P_result",
                                 self._format_pressure(P_Pa, P_unit))
            elif t == "pipe":
                comp = self._pipe_components.get(node.id)
                if comp is None:
                    self._set_widget(node, "dP_result", "")
                    self._set_widget(node, "Q_result", "")
                    continue
                try:
                    cr = result.component(comp)
                except (KeyError, ValueError):
                    self._set_widget(node, "dP_result", "")
                    self._set_widget(node, "Q_result", "")
                    continue
                dP_txt = self._format_pressure_delta(cr.dP_Pa, P_unit)
                q_txt  = self._format_flow(cr.mdot_kgs, cr.Q_m3s, Q_unit)
                self._set_widget(node, "dP_result", dP_txt)
                self._set_widget(node, "Q_result", q_txt)

    @staticmethod
    def _set_widget(node, widget_name, text):
        w = node.get_widget(widget_name)
        if w is not None:
            w.set_value(text)

    @staticmethod
    def _format_pressure(P_Pa, unit):
        if P_Pa is None:
            return ""
        val = ureg.Quantity(P_Pa, "Pa").to(U.to_pint(unit)).magnitude
        return f"P={val:.1f} {unit}"

    @staticmethod
    def _format_pressure_delta(dP_Pa, unit):
        if dP_Pa is None:
            return ""
        val = ureg.Quantity(dP_Pa, "Pa").to(U.to_pint(unit)).magnitude
        return f"dP={val:+.2f} {unit}"

    @staticmethod
    def _format_flow(mass_kgs, vol_m3s, unit):
        if mass_kgs is None and vol_m3s is None:
            return ""
        val = NetworkScreen._convert_flow(mass_kgs, vol_m3s, unit)
        return f"Q={val:+.1f} {unit}"

    @staticmethod
    def _convert_flow(mass_kgs, vol_m3s, unit):
        """Convert mass [kg/s] or volumetric [m^3/s] flow to the chosen
        display unit, picking the source by dimensionality."""
        pint_unit = U.to_pint(unit)
        dim = ureg.Quantity(1.0, pint_unit).dimensionality
        if dim == _MASS_FLOW_DIM:
            if mass_kgs is None:
                return 0.0
            return ureg.Quantity(mass_kgs, "kg/s").to(pint_unit).magnitude
        if dim == _VOL_FLOW_DIM:
            if vol_m3s is None:
                return 0.0
            return ureg.Quantity(vol_m3s, "m^3/s").to(pint_unit).magnitude
        return 0.0

    def _render_results_text(self, result):
        P_unit = self.d_pressure.currentText()
        Q_unit = self.d_flow.currentText()

        lines = [f"Converged: {result['converged']}", ""]

        lines.append(f"Node pressures ({P_unit}):")
        for name, P_Pa in result["P_Pa"].items():
            P_val = ureg.Quantity(P_Pa, "Pa").to(U.to_pint(P_unit)).magnitude
            lines.append(f"  {name:<16} {P_val:>12.3f}")
        lines.append("")

        lines.append(f"Edge flow ({Q_unit}, sign per nominal direction):")
        mass_dict = result["mdot_kgs"]
        vol_dict  = result["Q_m3s"]
        for edge_name in mass_dict:
            val = self._convert_flow(
                mass_dict[edge_name], vol_dict[edge_name], Q_unit,
            )
            arrow = "->" if val >= 0 else "<-"
            lines.append(f"  {edge_name:<24} {arrow} {val:>+14.4f}")
        lines.append("")

        lines.append(f"External flow at boundary nodes ({Q_unit}):")
        ext_mass = result["mdot_ext_kgs"]
        ext_vol  = result["Q_ext_m3s"]
        for name in ext_mass:
            val = self._convert_flow(ext_mass[name], ext_vol[name], Q_unit)
            if val == 0:
                continue
            lines.append(f"  {name:<16} {val:>+14.4f}")

        self.results_text.setPlainText("\n".join(lines))
