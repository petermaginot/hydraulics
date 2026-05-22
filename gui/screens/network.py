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

import math
import os
import traceback

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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

import gui.dialogs as dialogs
from gui.dialogs import NodeResultsDialog, PipeProfileWindow

from NodeGraphQt import BaseNode, NodeGraph

from component_classes import ureg
from gui import units as U
from gui.screens.segment import _LabeledField
import fluids.fittings
from incompressible import (
    Incompressible_Fluid, Line_Segment, Bend, Contraction_Expansion, Valve,
    CheckValve,
)
from network import Network


# NodeGraphQt node type identifiers.
SOURCE_SINK_TYPE_ID  = "pipe.SourceSinkNode"
JUNCTION_TYPE_ID     = "pipe.JunctionNode"
PIPE_SEGMENT_TYPE_ID = "pipe.PipeSegmentNode"
FITTING_NODE_TYPE_ID     = "pipe.FittingNode"
VALVE_NODE_TYPE_ID       = "pipe.ValveNode"
CHECKVALVE_NODE_TYPE_ID  = "pipe.CheckValveNode"

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
    show the solved pressure, the solved temperature (compressible only;
    left blank otherwise), and the (specified or derived) external flow.
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
        self.add_text_input("T_result", label="", text="")
        self.add_text_input("Q_result", label="", text="")


class JunctionNode(BaseNode):
    """Interior splitter / merger.

    No boundary condition; the only spec the user can set is the
    (decorative) elevation.  Result widgets show the solved pressure
    and (compressible only) the solved temperature.
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
        self.add_text_input("T_result", label="", text="")


class PipeSegmentNode(BaseNode):
    """Inline pipe segment with geometry.

    Single in + single out (no branching).  Stored geometry feeds a
    Line_Segment built at solve time and dropped into the solver Edge's
    component list.  Result widgets show the segment's dP (incompressible)
    or outlet P (compressible), outlet T (compressible only), and signed
    flow.  Widgets the active regime doesn't populate are left blank.
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
        self.add_text_input("T_result", label="", text="")
        self.add_text_input("Q_result", label="", text="")


class FittingNode(BaseNode):
    """Inline fitting (bend or sudden contraction/expansion).

    Single in + single out.  Stored parameters feed a Bend or
    Contraction_Expansion built at solve time.  Result widget set matches
    PipeSegmentNode.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Fitting"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=False)
        self.add_output("out", multi_output=False)
        self.add_text_input(
            "dP_result", label="", text="",
            placeholder_text="(solve to see)",
        )
        self.add_text_input("T_result", label="", text="")
        self.add_text_input("Q_result", label="", text="")


class ValveNode(BaseNode):
    """Inline valve fitting (globe or gate).

    Single in + single out.  Stored parameters feed a Valve built at
    solve time via the Crane K-factor correlations.  Result widget set
    matches PipeSegmentNode.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Valve"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=False)
        self.add_output("out", multi_output=False)
        self.add_text_input(
            "dP_result", label="", text="",
            placeholder_text="(solve to see)",
        )
        self.add_text_input("T_result", label="", text="")
        self.add_text_input("Q_result", label="", text="")


class CheckValveNode(BaseNode):
    """Inline check valve that seals on reverse flow.

    Single in + single out.  Forward K from Crane correlations;
    reverse K = _SEALING_K (handled by _reversed_component in network.py).
    Result widget set matches PipeSegmentNode.
    """
    __identifier__ = "pipe"
    NODE_NAME = "Check Valve"

    def __init__(self):
        super().__init__()
        self.add_input("in",   multi_input=False)
        self.add_output("out", multi_output=False)
        self.add_text_input(
            "dP_result", label="", text="",
            placeholder_text="(solve to see)",
        )
        self.add_text_input("T_result", label="", text="")
        self.add_text_input("Q_result", label="", text="")


# ---------------------------------------------------------------------------
# NetworkScreen
# ---------------------------------------------------------------------------

class NetworkScreen(QWidget):
    back_clicked = Signal()

    # Class-level config knobs.  CompressibleNetworkScreen overrides these
    # to swap in the compressible Line_Segment, the Compressible_Network
    # solver, and the compressible flow-rate unit list.  Everything else
    # (canvas, node editors, chain walking, CSV loading) is shared.
    LINE_SEGMENT_CLS     = Line_Segment
    BEND_CLS             = Bend
    VALVE_CLS            = Valve
    CHECKVALVE_CLS       = CheckValve
    CONTRACTION_EXP_CLS  = Contraction_Expansion
    NETWORK_CLS          = Network
    DISPLAY_FLOW_UNITS   = U.FLOW_RATE_INCOMPRESSIBLE
    DISPLAY_FLOW_DEFAULT = "BBL/D"
    FLUID_BOX_TITLE      = "Fluid (incompressible)"

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
        # Maps an inline-node id (pipe/fitting/valve/check_valve) to the
        # name of the solver Edge its chain was assembled into.  Used by
        # the compressible subclass to render per-block flow when the
        # solver result is a flat dict (no NetworkResult.component()).
        self._pipe_edge_names = {}
        # Maps an inline-node id to its 0-based position within its edge's
        # ORIGINAL components list (i.e. the from_node -> to_node order
        # the chain was walked in at network-build time).  The compressible
        # screen indexes solver result["component_outlet_PT"][edge_name]
        # with this to recover the flow-direction outlet (P, T) of each
        # inline block.
        self._inline_chain_pos = {}
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
        self.graph.register_node(FittingNode)
        self.graph.register_node(ValveNode)
        self.graph.register_node(CheckValveNode)
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)

        # ---- Toolbar ----
        add_src_btn   = QPushButton("+ Source/Sink")
        add_src_btn.clicked.connect(self._add_source_sink)
        add_jct_btn   = QPushButton("+ Junction")
        add_jct_btn.clicked.connect(self._add_junction)
        add_pipe_btn  = QPushButton("+ Pipe")
        add_pipe_btn.clicked.connect(self._add_pipe_segment)
        add_fit_btn   = QPushButton("+ Fitting")
        add_fit_btn.clicked.connect(self._add_fitting)
        add_valve_btn = QPushButton("+ Valve")
        add_valve_btn.clicked.connect(self._add_valve)
        add_cv_btn    = QPushButton("+ Check Valve")
        add_cv_btn.clicked.connect(self._add_check_valve)
        del_btn       = QPushButton("- Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        save_btn      = QPushButton("Save...")
        save_btn.clicked.connect(self._on_save_network)
        load_btn      = QPushButton("Load...")
        load_btn.clicked.connect(self._on_load_network)
        # Save Results is enabled only after a successful solve; the
        # _last_result is None check inside the handler guards it but we
        # also disable the button visually until then.
        self._save_results_btn = QPushButton("Save Results...")
        self._save_results_btn.clicked.connect(self._on_save_results)
        self._save_results_btn.setEnabled(False)
        self._solve_btn = QPushButton("Solve")
        self._solve_btn.clicked.connect(self._solve)

        toolbar = QHBoxLayout()
        toolbar.addWidget(add_src_btn)
        toolbar.addWidget(add_jct_btn)
        toolbar.addWidget(add_pipe_btn)
        toolbar.addWidget(add_fit_btn)
        toolbar.addWidget(add_valve_btn)
        toolbar.addWidget(add_cv_btn)
        toolbar.addWidget(del_btn)
        toolbar.addStretch()
        toolbar.addWidget(save_btn)
        toolbar.addWidget(load_btn)
        toolbar.addWidget(self._save_results_btn)
        toolbar.addWidget(self._solve_btn)

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

    def _default_source_sink_fields(self):
        spec = {
            "type":      "source_sink",
            "P_str":     "",   "P_unit":    "psi",
            "Q_str":     "",   "Q_unit":    self.DISPLAY_FLOW_DEFAULT,
            "elev_str":  "0",  "elev_unit": "ft",
        }
        spec.update(self._default_source_sink_extra())
        return spec

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

    @staticmethod
    def _default_fitting_fields():
        return {
            "type":          "fitting",
            "fitting_type":  "bend",
            "Di_str":        "4.026",  "Di_unit":    "inch",
            "angle_str":     "90",
            "bend_dias_str": "1.5",
            "Di_US_str":     "4.026",  "Di_US_unit": "inch",
            "Di_DS_str":     "3.0",    "Di_DS_unit": "inch",
        }

    @staticmethod
    def _default_valve_fields():
        return {
            "type":       "valve",
            "valve_type": "globe",
            "D1_str":     "4.026",  "D1_unit": "inch",
            "D2_str":     "4.026",  "D2_unit": "inch",
            "angle_str":  "0",
            "D_str":      "4.026",  "D_unit":  "inch",
            "style_str":  "0",
            # User-specified K/Cv/Kv branch.
            "spec_mode":    "K",     # "K" | "Cv" | "Kv"
            "spec_val_str": "1.0",
        }

    @staticmethod
    def _default_check_valve_fields():
        return {
            "type":            "check_valve",
            "cv_type":         "swing",
            # swing / tilting_disk: single D
            "D_str":           "4.026",  "D_unit":  "inch",
            "angle_str":       "5",
            "angled_str":      "1",      # 1 = angled body, 0 = straight
            "style_str":       "0",
            # lift / angle_stop / globe_stop: D1 + D2
            "D1_str":          "4.026",  "D1_unit": "inch",
            "D2_str":          "4.026",  "D2_unit": "inch",
        }

    # ------------------------------------------------------------------
    # Side-panel builders
    # ------------------------------------------------------------------

    def _build_fluid_box(self):
        """Build the side-panel fluid box.

        Default (incompressible) layout has density / viscosity fields.
        CompressibleNetworkScreen overrides this to show a composition
        summary read from AppState instead.
        """
        self.f_density   = _LabeledField("62.4", U.DENSITY,   "lb/ft^3")
        self.f_viscosity = _LabeledField("1.0",  U.VISCOSITY, "cP")
        form = QFormLayout()
        form.addRow("Density:",   self.f_density.widget())
        form.addRow("Viscosity:", self.f_viscosity.widget())
        self.fluid_box = QGroupBox(self.FLUID_BOX_TITLE)
        self.fluid_box.setLayout(form)

    def _build_editor_stack(self):
        """Selection-aware editor.  Four pages keyed by node type, swapped
        via QStackedWidget when a node is selected.
        """
        self.editor_stack = QStackedWidget()
        self._build_empty_editor()        # index 0
        self._build_source_sink_editor()  # index 1
        self._build_junction_editor()     # index 2
        self._build_pipe_editor()         # index 3
        self._build_fitting_editor()       # index 4
        self._build_valve_editor()         # index 5
        self._build_check_valve_editor()   # index 6

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
        self.ss_P    = _LabeledField("",  U.PRESSURE,           "psi")
        self.ss_Q    = _LabeledField(
            "",  self.DISPLAY_FLOW_UNITS, self.DISPLAY_FLOW_DEFAULT,
        )
        self.ss_elev = _LabeledField("0", U.LENGTH,             "ft")
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
        form.addRow("Pressure:",          self.ss_P.widget())
        form.addRow("Flow Rate:",      self.ss_Q.widget())
        # Subclass hook: lets CompressibleNetworkScreen add a T row here.
        self._add_extra_source_sink_rows(form)
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

        self.p_id     = _LabeledField("4.026",   U.DIAMETER,  "inch")
        self.p_od     = _LabeledField("",        U.DIAMETER,  "inch")
        self.p_wt     = _LabeledField("",        U.DIAMETER,  "inch")
        self.p_length = _LabeledField("1000",    U.LENGTH,    "ft")
        self.p_dz     = _LabeledField("0",       U.LENGTH,    "ft")
        self.p_rough  = _LabeledField("0.00015", U.ROUGHNESS, "ft")
        self.p_apply  = QPushButton("Apply")
        self.p_apply.clicked.connect(self._apply_pipe_edits)
        self.p_details_btn = QPushButton("Show solved details...")
        self.p_details_btn.clicked.connect(self._show_current_node_details)
        self.p_details_btn.setVisible(False)

        self.p_downsample_chk = QCheckBox("Downsample profile")
        self.p_downsample_chk.setToolTip(
            "Reduce dense profiles to diameter-change points, elevation "
            "inflections, and a maximum step length.  Speeds up compressible "
            "flow calculations."
        )
        self.p_downsample_chk.toggled.connect(self._on_pipe_downsample_toggled)

        self.p_max_step = _LabeledField("1000", U.LENGTH, "m")
        self.p_max_step.edit.setEnabled(False)
        self.p_max_step.combo.setEnabled(False)

        self.p_elev_tol = _LabeledField("0.1", U.LENGTH, "m")
        self.p_elev_tol.edit.setEnabled(False)
        self.p_elev_tol.combo.setEnabled(False)

        form = QFormLayout()
        form.addRow("Name:",      self.p_name)
        form.addRow("Source:",    self.p_source_label)
        form.addRow("ID:",        self.p_id.widget())
        form.addRow("OR OD:",     self.p_od.widget())
        form.addRow("WT:",        self.p_wt.widget())
        form.addRow("Length:",    self.p_length.widget())
        form.addRow("dz:",        self.p_dz.widget())
        form.addRow("Roughness:", self.p_rough.widget())
        form.addRow("Max step:",   self.p_max_step.widget())
        form.addRow("Elev. tol.:", self.p_elev_tol.widget())

        load_row = QHBoxLayout()
        load_row.addWidget(self.p_csv_btn)
        load_row.addWidget(self.p_downsample_chk)
        load_row.addStretch()

        apply_row = QHBoxLayout()
        apply_row.addWidget(self.p_apply)
        apply_row.addWidget(self.p_clear_csv_btn)
        apply_row.addStretch()

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(form)
        v.addLayout(load_row)
        v.addLayout(apply_row)
        v.addWidget(self.p_details_btn)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_fitting_editor(self):
        self.fit_name = QLineEdit()
        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItems(["Bend", "Sudden Contraction/Expansion"])
        self.fit_type_combo.currentIndexChanged.connect(self._on_fitting_type_changed)

        # Bend sub-page
        self.fit_bend_Di    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.fit_bend_angle = QLineEdit("90")
        self.fit_bend_angle.setPlaceholderText("degrees")
        self.fit_bend_dias  = QLineEdit("1.5")
        self.fit_bend_dias.setPlaceholderText("R/D ratio")
        bend_form = QFormLayout()
        bend_form.addRow("Pipe ID:",     self.fit_bend_Di.widget())
        bend_form.addRow("Angle (deg):", self.fit_bend_angle)
        bend_form.addRow("Bend R/D:",    self.fit_bend_dias)
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

        # Contraction/expansion sub-page
        self.fit_ce_Di_US = _LabeledField("4.026", U.DIAMETER, "inch")
        self.fit_ce_Di_DS = _LabeledField("3.0",   U.DIAMETER, "inch")
        ce_form = QFormLayout()
        ce_form.addRow("Upstream ID:",   self.fit_ce_Di_US.widget())
        ce_form.addRow("Downstream ID:", self.fit_ce_Di_DS.widget())
        ce_hint = QLabel(
            "Sharp-edged (abrupt) transition.  "
            "Contraction or expansion is inferred from the two diameters."
        )
        ce_hint.setWordWrap(True)
        ce_hint.setStyleSheet("color: #777; font-style: italic;")
        ce_page = QWidget()
        cv = QVBoxLayout(ce_page)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addLayout(ce_form)
        cv.addWidget(ce_hint)

        self.fit_input_stack = QStackedWidget()
        self.fit_input_stack.addWidget(bend_page)   # index 0
        self.fit_input_stack.addWidget(ce_page)     # index 1

        self.fit_apply = QPushButton("Apply")
        self.fit_apply.clicked.connect(self._apply_fitting_edits)
        self.fit_details_btn = QPushButton("Show solved details...")
        self.fit_details_btn.clicked.connect(self._show_current_node_details)
        self.fit_details_btn.setVisible(False)

        top_form = QFormLayout()
        top_form.addRow("Name:", self.fit_name)
        top_form.addRow("Type:", self.fit_type_combo)

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(top_form)
        v.addWidget(self.fit_input_stack)
        v.addWidget(self.fit_apply)
        v.addWidget(self.fit_details_btn)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_valve_editor(self):
        self.valve_name = QLineEdit()
        self.valve_type_combo = QComboBox()
        self.valve_type_combo.addItems(
            ["Globe", "Gate", "Butterfly", "Plug", "Ball", "User specified"]
        )
        self.valve_type_combo.currentIndexChanged.connect(self._on_valve_type_changed)

        # Globe sub-page (index 0)
        self.valve_globe_D1 = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_globe_D2 = _LabeledField("4.026", U.DIAMETER, "inch")
        globe_form = QFormLayout()
        globe_form.addRow("Seat bore D1:", self.valve_globe_D1.widget())
        globe_form.addRow("Pipe ID D2:",   self.valve_globe_D2.widget())
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

        # Gate sub-page (index 1)
        self.valve_gate_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_gate_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_gate_angle = QLineEdit("0")
        self.valve_gate_angle.setPlaceholderText("degrees")
        gate_form = QFormLayout()
        gate_form.addRow("Seat bore D1:",        self.valve_gate_D1.widget())
        gate_form.addRow("Pipe ID D2:",          self.valve_gate_D2.widget())
        gate_form.addRow("Reducer angle (deg):", self.valve_gate_angle)
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

        # Butterfly sub-page (index 2)
        self.valve_butterfly_D     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_butterfly_style = QComboBox()
        self.valve_butterfly_style.addItems(
            ["Centric (0)", "Double offset (1)", "Triple offset (2)"]
        )
        butterfly_form = QFormLayout()
        butterfly_form.addRow("Pipe ID:", self.valve_butterfly_D.widget())
        butterfly_form.addRow("Style:",   self.valve_butterfly_style)
        butterfly_hint = QLabel(
            "Single pipe diameter (no reducer).  "
            "N factor by size: 45/74/218 (2“–8“), "
            "35/52/96 (10“–14“), 25/43/55 (16“–24“)."
        )
        butterfly_hint.setWordWrap(True)
        butterfly_hint.setStyleSheet("color: #777; font-style: italic;")
        butterfly_page = QWidget()
        bfv = QVBoxLayout(butterfly_page)
        bfv.setContentsMargins(0, 0, 0, 0)
        bfv.addLayout(butterfly_form)
        bfv.addWidget(butterfly_hint)

        # Plug sub-page (index 3)
        self.valve_plug_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_plug_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_plug_angle = QLineEdit("0")
        self.valve_plug_angle.setPlaceholderText("degrees")
        self.valve_plug_style = QComboBox()
        self.valve_plug_style.addItems(
            ["Straight-through (0)", "3-way, flow straight (1)", "3-way, flow 90° (2)"]
        )
        plug_form = QFormLayout()
        plug_form.addRow("Plug bore D1:",        self.valve_plug_D1.widget())
        plug_form.addRow("Pipe ID D2:",          self.valve_plug_D2.widget())
        plug_form.addRow("Reducer angle (deg):", self.valve_plug_angle)
        plug_form.addRow("Style:",               self.valve_plug_style)
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

        # Ball sub-page (index 4)
        self.valve_ball_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_ball_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_ball_angle = QLineEdit("0")
        self.valve_ball_angle.setPlaceholderText("degrees")
        ball_form = QFormLayout()
        ball_form.addRow("Seat bore D1:",        self.valve_ball_D1.widget())
        ball_form.addRow("Pipe ID D2:",          self.valve_ball_D2.widget())
        ball_form.addRow("Reducer angle (deg):", self.valve_ball_angle)
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

        # User-specified sub-page (index 5)
        self.valve_user_D       = _LabeledField("4.026", U.DIAMETER, "inch")
        self.valve_user_mode    = QComboBox()
        self.valve_user_mode.addItems(["K", "Cv", "Kv"])
        self.valve_user_mode.currentIndexChanged.connect(
            self._on_valve_user_mode_changed
        )
        self.valve_user_value   = QLineEdit("1.0")
        self.valve_user_value_label = QLabel("K value:")
        user_form = QFormLayout()
        user_form.addRow("Pipe ID:",      self.valve_user_D.widget())
        user_form.addRow("Specify by:",   self.valve_user_mode)
        user_form.addRow(self.valve_user_value_label, self.valve_user_value)
        user_hint = QLabel(
            "K is the dimensionless resistance coefficient referenced to "
            "the pipe velocity head.  Cv is the US flow coefficient "
            "[gpm/psi^0.5]; Kv is the metric flow coefficient "
            "[m^3/h/bar^0.5].  Cv and Kv are converted to K via "
            "K = 2.166e9 · D^4 / Cv^2 (D in m) and Cv = 1.156 · Kv."
        )
        user_hint.setWordWrap(True)
        user_hint.setStyleSheet("color: #777; font-style: italic;")
        user_page = QWidget()
        uv = QVBoxLayout(user_page)
        uv.setContentsMargins(0, 0, 0, 0)
        uv.addLayout(user_form)
        uv.addWidget(user_hint)

        self.valve_input_stack = QStackedWidget()
        self.valve_input_stack.addWidget(globe_page)      # index 0
        self.valve_input_stack.addWidget(gate_page)       # index 1
        self.valve_input_stack.addWidget(butterfly_page)  # index 2
        self.valve_input_stack.addWidget(plug_page)       # index 3
        self.valve_input_stack.addWidget(ball_page)       # index 4
        self.valve_input_stack.addWidget(user_page)       # index 5

        self.valve_apply = QPushButton("Apply")
        self.valve_apply.clicked.connect(self._apply_valve_edits)
        self.valve_details_btn = QPushButton("Show solved details...")
        self.valve_details_btn.clicked.connect(self._show_current_node_details)
        self.valve_details_btn.setVisible(False)

        top_form = QFormLayout()
        top_form.addRow("Name:", self.valve_name)
        top_form.addRow("Type:", self.valve_type_combo)

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(top_form)
        v.addWidget(self.valve_input_stack)
        v.addWidget(self.valve_apply)
        v.addWidget(self.valve_details_btn)
        v.addStretch()
        self.editor_stack.addWidget(page)

    def _build_check_valve_editor(self):
        self.cv_name = QLineEdit()
        self.cv_type_combo = QComboBox()
        self.cv_type_combo.addItems([
            "Swing", "Lift", "Tilting disk", "Angle stop", "Globe stop",
        ])
        self.cv_type_combo.currentIndexChanged.connect(self._on_cv_type_changed)

        # Swing sub-page (index 0) -- single D + angled
        self.cv_swing_D      = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_swing_angled = QComboBox()
        self.cv_swing_angled.addItems(["Angled body", "Straight body"])
        swing_form = QFormLayout()
        swing_form.addRow("Pipe ID:", self.cv_swing_D.widget())
        swing_form.addRow("Body:",    self.cv_swing_angled)
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

        # Lift sub-page (index 1) -- D1 + D2 + angled
        self.cv_lift_D1      = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_lift_D2      = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_lift_angled  = QComboBox()
        self.cv_lift_angled.addItems(["Angled body", "Straight body"])
        lift_form = QFormLayout()
        lift_form.addRow("Seat bore D1:", self.cv_lift_D1.widget())
        lift_form.addRow("Pipe ID D2:",   self.cv_lift_D2.widget())
        lift_form.addRow("Body:",         self.cv_lift_angled)
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

        # Tilting disk sub-page (index 2) -- single D + angle
        self.cv_tilt_D     = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_tilt_angle = QLineEdit("5")
        self.cv_tilt_angle.setPlaceholderText("degrees")
        tilt_form = QFormLayout()
        tilt_form.addRow("Pipe ID:",       self.cv_tilt_D.widget())
        tilt_form.addRow("Disk angle (deg):", self.cv_tilt_angle)
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

        # Angle stop sub-page (index 3) -- D1 + D2 + style
        self.cv_angstop_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_angstop_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_angstop_style = QComboBox()
        self.cv_angstop_style.addItems(["Style 0 (piston)", "Style 1 (no stem guide)"])
        angstop_form = QFormLayout()
        angstop_form.addRow("Seat bore D1:", self.cv_angstop_D1.widget())
        angstop_form.addRow("Pipe ID D2:",   self.cv_angstop_D2.widget())
        angstop_form.addRow("Style:",        self.cv_angstop_style)
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

        # Globe stop sub-page (index 4) -- D1 + D2 + style
        self.cv_globestop_D1    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_globestop_D2    = _LabeledField("4.026", U.DIAMETER, "inch")
        self.cv_globestop_style = QComboBox()
        self.cv_globestop_style.addItems(["Style 0 (piston)", "Style 1 (no stem guide)"])
        globestop_form = QFormLayout()
        globestop_form.addRow("Seat bore D1:", self.cv_globestop_D1.widget())
        globestop_form.addRow("Pipe ID D2:",   self.cv_globestop_D2.widget())
        globestop_form.addRow("Style:",        self.cv_globestop_style)
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

        self.cv_input_stack = QStackedWidget()
        self.cv_input_stack.addWidget(swing_page)     # index 0
        self.cv_input_stack.addWidget(lift_page)      # index 1
        self.cv_input_stack.addWidget(tilt_page)      # index 2
        self.cv_input_stack.addWidget(angstop_page)   # index 3
        self.cv_input_stack.addWidget(globestop_page) # index 4

        self.cv_apply = QPushButton("Apply")
        self.cv_apply.clicked.connect(self._apply_check_valve_edits)
        self.cv_details_btn = QPushButton("Show solved details...")
        self.cv_details_btn.clicked.connect(self._show_current_node_details)
        self.cv_details_btn.setVisible(False)

        top_form = QFormLayout()
        top_form.addRow("Name:", self.cv_name)
        top_form.addRow("Type:", self.cv_type_combo)

        page = QWidget()
        v = QVBoxLayout(page)
        v.addLayout(top_form)
        v.addWidget(self.cv_input_stack)
        v.addWidget(self.cv_apply)
        v.addWidget(self.cv_details_btn)
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
        self.d_flow.addItems(self.DISPLAY_FLOW_UNITS)
        self.d_flow.setCurrentText(self.DISPLAY_FLOW_DEFAULT)
        self.d_flow.currentTextChanged.connect(
            self._rerender_with_current_units
        )

        form = QFormLayout()
        form.addRow("Pressure:", self.d_pressure)
        form.addRow("Flow:",     self.d_flow)
        # Subclass hook: compressible adds a Temperature row here.
        self._add_extra_display_unit_rows(form)
        self.display_box = QGroupBox("Result display units")
        self.display_box.setLayout(form)

    def _add_extra_display_unit_rows(self, form):
        """Hook for subclasses to append rows to the display-units form.

        The default no-op keeps the incompressible screen at just
        Pressure + Flow.  CompressibleNetworkScreen overrides to add
        a Temperature combo.
        """
        return

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

    def _add_fitting(self):
        n = self.graph.create_node(FITTING_NODE_TYPE_ID)
        self.node_specs[n.id] = self._default_fitting_fields()

    def _add_valve(self):
        n = self.graph.create_node(VALVE_NODE_TYPE_ID)
        self.node_specs[n.id] = self._default_valve_fields()

    def _add_check_valve(self):
        n = self.graph.create_node(CHECKVALVE_NODE_TYPE_ID)
        self.node_specs[n.id] = self._default_check_valve_fields()

    def _delete_selected(self):
        nodes = self.graph.selected_nodes()
        if nodes:
            self.graph.delete_nodes(nodes)

    def _on_nodes_deleted(self, node_ids):
        node_ids = set(node_ids)
        for nid in node_ids:
            self.node_specs.pop(nid, None)
            self._pipe_components.pop(nid, None)
            self._pipe_edge_names.pop(nid, None)
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
            self._refresh_details_button_visibility()
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
            self.ss_Q.combo.setCurrentText(
                spec.get("Q_unit", self.DISPLAY_FLOW_DEFAULT)
            )
            self.ss_elev.edit.setText(spec.get("elev_str", "0"))
            self.ss_elev.combo.setCurrentText(spec.get("elev_unit", "ft"))
            # Subclass hook: populate any extra widgets (e.g. T for compressible).
            self._sync_source_sink_extra(spec)
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
        elif t == "fitting":
            self.editor_stack.setCurrentIndex(4)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Fitting]")
            self.fit_name.setText(n.name())
            ft  = spec.get("fitting_type", "bend")
            idx = 0 if ft == "bend" else 1
            self.fit_type_combo.setCurrentIndex(idx)
            self.fit_input_stack.setCurrentIndex(idx)
            self.fit_bend_Di.edit.setText(spec.get("Di_str", "4.026"))
            self.fit_bend_Di.combo.setCurrentText(spec.get("Di_unit", "inch"))
            self.fit_bend_angle.setText(spec.get("angle_str", "90"))
            self.fit_bend_dias.setText(spec.get("bend_dias_str", "1.5"))
            self.fit_ce_Di_US.edit.setText(spec.get("Di_US_str", "4.026"))
            self.fit_ce_Di_US.combo.setCurrentText(spec.get("Di_US_unit", "inch"))
            self.fit_ce_Di_DS.edit.setText(spec.get("Di_DS_str", "3.0"))
            self.fit_ce_Di_DS.combo.setCurrentText(spec.get("Di_DS_unit", "inch"))
        elif t == "valve":
            self.editor_stack.setCurrentIndex(5)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Valve]")
            self.valve_name.setText(n.name())
            vt   = spec.get("valve_type", "globe")
            _VT  = ["globe", "gate", "butterfly", "plug", "ball", "user_specified"]
            idx  = _VT.index(vt) if vt in _VT else 0
            self.valve_type_combo.setCurrentIndex(idx)
            self.valve_input_stack.setCurrentIndex(idx)
            # Populate D1/D2 on all pages that share those keys
            for w1, w2 in [
                (self.valve_globe_D1, self.valve_globe_D2),
                (self.valve_gate_D1,  self.valve_gate_D2),
                (self.valve_plug_D1,  self.valve_plug_D2),
                (self.valve_ball_D1,  self.valve_ball_D2),
            ]:
                w1.edit.setText(spec.get("D1_str", "4.026"))
                w1.combo.setCurrentText(spec.get("D1_unit", "inch"))
                w2.edit.setText(spec.get("D2_str", "4.026"))
                w2.combo.setCurrentText(spec.get("D2_unit", "inch"))
            angle = spec.get("angle_str", "0")
            self.valve_gate_angle.setText(angle)
            self.valve_plug_angle.setText(angle)
            self.valve_ball_angle.setText(angle)
            self.valve_butterfly_D.edit.setText(spec.get("D_str", "4.026"))
            self.valve_butterfly_D.combo.setCurrentText(spec.get("D_unit", "inch"))
            style_idx = int(spec.get("style_str", "0") or "0")
            self.valve_butterfly_style.setCurrentIndex(style_idx)
            self.valve_plug_style.setCurrentIndex(style_idx)
            # User-specified branch
            self.valve_user_D.edit.setText(spec.get("D_str", "4.026"))
            self.valve_user_D.combo.setCurrentText(spec.get("D_unit", "inch"))
            spec_mode = spec.get("spec_mode", "K")
            mode_idx  = {"K": 0, "Cv": 1, "Kv": 2}.get(spec_mode, 0)
            self.valve_user_mode.setCurrentIndex(mode_idx)
            self.valve_user_value_label.setText(f"{spec_mode} value:")
            self.valve_user_value.setText(spec.get("spec_val_str", "1.0"))
        elif t == "check_valve":
            self.editor_stack.setCurrentIndex(6)
            self.editor_box.setTitle(f"Selected node: {n.name()}  [Check Valve]")
            self.cv_name.setText(n.name())
            _CVT = ["swing", "lift", "tilting_disk", "angle_stop", "globe_stop"]
            cvt  = spec.get("cv_type", "swing")
            idx  = _CVT.index(cvt) if cvt in _CVT else 0
            self.cv_type_combo.setCurrentIndex(idx)
            self.cv_input_stack.setCurrentIndex(idx)
            # Single-D fields
            self.cv_swing_D.edit.setText(spec.get("D_str", "4.026"))
            self.cv_swing_D.combo.setCurrentText(spec.get("D_unit", "inch"))
            self.cv_tilt_D.edit.setText(spec.get("D_str", "4.026"))
            self.cv_tilt_D.combo.setCurrentText(spec.get("D_unit", "inch"))
            self.cv_tilt_angle.setText(spec.get("angle_str", "5"))
            angled_idx = 0 if spec.get("angled_str", "1") == "1" else 1
            self.cv_swing_angled.setCurrentIndex(angled_idx)
            self.cv_lift_angled.setCurrentIndex(angled_idx)
            # D1/D2 fields
            for w1, w2 in [
                (self.cv_lift_D1,      self.cv_lift_D2),
                (self.cv_angstop_D1,   self.cv_angstop_D2),
                (self.cv_globestop_D1, self.cv_globestop_D2),
            ]:
                w1.edit.setText(spec.get("D1_str", "4.026"))
                w1.combo.setCurrentText(spec.get("D1_unit", "inch"))
                w2.edit.setText(spec.get("D2_str", "4.026"))
                w2.combo.setCurrentText(spec.get("D2_unit", "inch"))
            style_idx = int(spec.get("style_str", "0") or "0")
            self.cv_angstop_style.setCurrentIndex(style_idx)
            self.cv_globestop_style.setCurrentIndex(style_idx)
        else:
            self.editor_stack.setCurrentIndex(0)
        self._refresh_details_button_visibility()

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
            # Restore downsample settings from spec.
            max_step_m = spec.get("downsample_max_step_m", 0.0)
            elev_tol_m = spec.get("downsample_elev_tol_m", 0.0)
            ds_enabled = max_step_m > 0.0
            self.p_downsample_chk.setChecked(ds_enabled)
            self._on_pipe_downsample_toggled(ds_enabled)
            self.p_max_step.edit.setText(f"{max_step_m:.6g}" if ds_enabled else "1000")
            self.p_max_step.combo.setCurrentText("m")
            self.p_elev_tol.edit.setText(f"{elev_tol_m:.6g}" if ds_enabled else "0.1")
            self.p_elev_tol.combo.setCurrentText("m")
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
            # No CSV in manual mode -- reset downsample UI to defaults.
            self.p_downsample_chk.setChecked(False)
            self._on_pipe_downsample_toggled(False)
            self.p_max_step.edit.setText("1000")
            self.p_max_step.combo.setCurrentText("m")
            self.p_elev_tol.edit.setText("0.1")
            self.p_elev_tol.combo.setCurrentText("m")

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
        spec = {
            "type":      "source_sink",
            "P_str":     self.ss_P.edit.text().strip(),
            "P_unit":    self.ss_P.combo.currentText(),
            "Q_str":     self.ss_Q.edit.text().strip(),
            "Q_unit":    self.ss_Q.combo.currentText(),
            "elev_str":  self.ss_elev.edit.text().strip() or "0",
            "elev_unit": self.ss_elev.combo.currentText(),
        }
        # Subclass hook: mutate spec to include extra fields (e.g. T).
        self._apply_source_sink_extra(spec)
        self.node_specs[n.id] = spec
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

    def _on_fitting_type_changed(self, index):
        self.fit_input_stack.setCurrentIndex(index)

    def _on_valve_type_changed(self, index):
        self.valve_input_stack.setCurrentIndex(index)

    def _on_valve_user_mode_changed(self, index):
        mode = ("K", "Cv", "Kv")[index]
        self.valve_user_value_label.setText(f"{mode} value:")

    def _on_cv_type_changed(self, index):
        self.cv_input_stack.setCurrentIndex(index)

    def _apply_fitting_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "fitting":
            return
        new_name = self.fit_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        ft = "bend" if self.fit_type_combo.currentIndex() == 0 else "contraction_expansion"
        self.node_specs[n.id] = {
            "type":          "fitting",
            "fitting_type":  ft,
            "Di_str":        self.fit_bend_Di.edit.text().strip(),
            "Di_unit":       self.fit_bend_Di.combo.currentText(),
            "angle_str":     self.fit_bend_angle.text().strip(),
            "bend_dias_str": self.fit_bend_dias.text().strip(),
            "Di_US_str":     self.fit_ce_Di_US.edit.text().strip(),
            "Di_US_unit":    self.fit_ce_Di_US.combo.currentText(),
            "Di_DS_str":     self.fit_ce_Di_DS.edit.text().strip(),
            "Di_DS_unit":    self.fit_ce_Di_DS.combo.currentText(),
        }
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Fitting]")

    def _apply_valve_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "valve":
            return
        new_name = self.valve_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        _VT = ["globe", "gate", "butterfly", "plug", "ball", "user_specified"]
        vt  = _VT[self.valve_type_combo.currentIndex()]
        if vt == "globe":
            spec = {
                "type": "valve", "valve_type": vt,
                "D1_str":  self.valve_globe_D1.edit.text().strip(),
                "D1_unit": self.valve_globe_D1.combo.currentText(),
                "D2_str":  self.valve_globe_D2.edit.text().strip(),
                "D2_unit": self.valve_globe_D2.combo.currentText(),
                "angle_str": "", "D_str": "", "D_unit": "inch", "style_str": "",
            }
        elif vt == "gate":
            spec = {
                "type": "valve", "valve_type": vt,
                "D1_str":  self.valve_gate_D1.edit.text().strip(),
                "D1_unit": self.valve_gate_D1.combo.currentText(),
                "D2_str":  self.valve_gate_D2.edit.text().strip(),
                "D2_unit": self.valve_gate_D2.combo.currentText(),
                "angle_str": self.valve_gate_angle.text().strip(),
                "D_str": "", "D_unit": "inch", "style_str": "",
            }
        elif vt == "butterfly":
            spec = {
                "type": "valve", "valve_type": vt,
                "D_str":  self.valve_butterfly_D.edit.text().strip(),
                "D_unit": self.valve_butterfly_D.combo.currentText(),
                "style_str": str(self.valve_butterfly_style.currentIndex()),
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch", "angle_str": "",
            }
        elif vt == "plug":
            spec = {
                "type": "valve", "valve_type": vt,
                "D1_str":  self.valve_plug_D1.edit.text().strip(),
                "D1_unit": self.valve_plug_D1.combo.currentText(),
                "D2_str":  self.valve_plug_D2.edit.text().strip(),
                "D2_unit": self.valve_plug_D2.combo.currentText(),
                "angle_str": self.valve_plug_angle.text().strip(),
                "style_str": str(self.valve_plug_style.currentIndex()),
                "D_str": "", "D_unit": "inch",
            }
        elif vt == "ball":
            spec = {
                "type": "valve", "valve_type": vt,
                "D1_str":  self.valve_ball_D1.edit.text().strip(),
                "D1_unit": self.valve_ball_D1.combo.currentText(),
                "D2_str":  self.valve_ball_D2.edit.text().strip(),
                "D2_unit": self.valve_ball_D2.combo.currentText(),
                "angle_str": self.valve_ball_angle.text().strip(),
                "D_str": "", "D_unit": "inch", "style_str": "",
            }
        else:  # user_specified
            spec = {
                "type": "valve", "valve_type": vt,
                "D_str":        self.valve_user_D.edit.text().strip(),
                "D_unit":       self.valve_user_D.combo.currentText(),
                "spec_mode":    ("K", "Cv", "Kv")[self.valve_user_mode.currentIndex()],
                "spec_val_str": self.valve_user_value.text().strip(),
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch",
                "angle_str": "", "style_str": "",
            }
        self.node_specs[n.id] = spec
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Valve]")

    def _apply_check_valve_edits(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "check_valve":
            return
        new_name = self.cv_name.text().strip()
        if new_name and new_name != n.name():
            n.set_name(new_name)
        _CVT = ["swing", "lift", "tilting_disk", "angle_stop", "globe_stop"]
        cvt  = _CVT[self.cv_type_combo.currentIndex()]
        angled_str = "0" if (
            self.cv_swing_angled.currentIndex() == 1
            or self.cv_lift_angled.currentIndex() == 1
        ) else "1"
        if cvt == "swing":
            spec = {
                "type": "check_valve", "cv_type": cvt,
                "D_str":  self.cv_swing_D.edit.text().strip(),
                "D_unit": self.cv_swing_D.combo.currentText(),
                "angled_str": angled_str,
                "angle_str": "", "style_str": "",
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch",
            }
        elif cvt == "lift":
            spec = {
                "type": "check_valve", "cv_type": cvt,
                "D1_str":  self.cv_lift_D1.edit.text().strip(),
                "D1_unit": self.cv_lift_D1.combo.currentText(),
                "D2_str":  self.cv_lift_D2.edit.text().strip(),
                "D2_unit": self.cv_lift_D2.combo.currentText(),
                "angled_str": angled_str,
                "D_str": "", "D_unit": "inch",
                "angle_str": "", "style_str": "",
            }
        elif cvt == "tilting_disk":
            spec = {
                "type": "check_valve", "cv_type": cvt,
                "D_str":    self.cv_tilt_D.edit.text().strip(),
                "D_unit":   self.cv_tilt_D.combo.currentText(),
                "angle_str": self.cv_tilt_angle.text().strip(),
                "angled_str": "", "style_str": "",
                "D1_str": "", "D1_unit": "inch",
                "D2_str": "", "D2_unit": "inch",
            }
        elif cvt == "angle_stop":
            spec = {
                "type": "check_valve", "cv_type": cvt,
                "D1_str":  self.cv_angstop_D1.edit.text().strip(),
                "D1_unit": self.cv_angstop_D1.combo.currentText(),
                "D2_str":  self.cv_angstop_D2.edit.text().strip(),
                "D2_unit": self.cv_angstop_D2.combo.currentText(),
                "style_str": str(self.cv_angstop_style.currentIndex()),
                "D_str": "", "D_unit": "inch",
                "angle_str": "", "angled_str": "",
            }
        else:  # globe_stop
            spec = {
                "type": "check_valve", "cv_type": cvt,
                "D1_str":  self.cv_globestop_D1.edit.text().strip(),
                "D1_unit": self.cv_globestop_D1.combo.currentText(),
                "D2_str":  self.cv_globestop_D2.edit.text().strip(),
                "D2_unit": self.cv_globestop_D2.combo.currentText(),
                "style_str": str(self.cv_globestop_style.currentIndex()),
                "D_str": "", "D_unit": "inch",
                "angle_str": "", "angled_str": "",
            }
        self.node_specs[n.id] = spec
        self.editor_box.setTitle(f"Selected node: {n.name()}  [Check Valve]")

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
            downsample = False
            elev_tol   = 0.0
            if self.p_downsample_chk.isChecked():
                qty = self.p_max_step.quantity()
                downsample = float(qty.to("m").magnitude) if qty is not None else 1000.0
                et_qty = self.p_elev_tol.quantity()
                elev_tol = float(et_qty.to("m").magnitude) if et_qty is not None else 0.0

            # Reuse Line_Segment.from_csv for parsing/validation; the temp
            # roughness value is discarded -- only the parsed profile
            # matters.  The real roughness is read from the editor at
            # solve time.
            tmp_seg = self.LINE_SEGMENT_CLS.from_csv(
                path, roughness=1e-6, downsample=downsample, elev_tol=elev_tol,
            )
        except Exception as e:
            dialogs.critical(
                self, "Could not load CSV", f"{type(e).__name__}: {e}",
            )
            return
        profile = list(tmp_seg.profile)
        if not profile:
            dialogs.critical(self, "Could not load CSV",
                             "CSV produced an empty profile.")
            return
        spec = self.node_specs[n.id]
        spec["mode"]                   = "csv"
        spec["csv_path"]               = path
        spec["csv_profile"]            = profile
        spec["downsample_max_step_m"]  = float(downsample) if downsample is not False else 0.0
        spec["downsample_elev_tol_m"]  = elev_tol
        self._sync_pipe_editor_to_spec(spec)

    def _on_clear_pipe_csv(self):
        n = self._cur_node
        if n is None or self.node_specs.get(n.id, {}).get("type") != "pipe":
            return
        spec = self.node_specs[n.id]
        spec["mode"] = "manual"
        spec.pop("csv_path",              None)
        spec.pop("csv_profile",           None)
        spec.pop("downsample_max_step_m", None)
        spec.pop("downsample_elev_tol_m", None)
        self._sync_pipe_editor_to_spec(spec)

    def _on_pipe_downsample_toggled(self, checked):
        self.p_max_step.edit.setEnabled(checked)
        self.p_max_step.combo.setEnabled(checked)
        self.p_elev_tol.edit.setEnabled(checked)
        self.p_elev_tol.combo.setEnabled(checked)

    # ------------------------------------------------------------------
    # Solve: translate canvas graph into solver Network and run.
    # ------------------------------------------------------------------

    def _solve(self):
        # Disable Solve while the solver is running.  The compressible
        # subclass pumps QApplication.processEvents() from its progress
        # callback so the UI can repaint mid-solve; without this guard a
        # second click would re-enter _solve.
        self._solve_btn.setEnabled(False)
        try:
            try:
                fluid  = self._build_fluid()
                net    = self._build_network()
                result = self._solve_network(net, fluid)
            except Exception as e:
                dialogs.critical(
                    self, "Solve failed",
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                )
                return
            self._render_results(net, result)
        finally:
            self._solve_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Save / Load network + Save Results.
    #
    # The file format lives in network.py (Network.to_dict / from_dict
    # and NetworkResult.save_bundle); this layer just wraps those calls
    # with a file picker, gui_extras capture, and a warning dialog for
    # cross-regime loads.  See gui/persistence.py for the actual work.
    # ------------------------------------------------------------------

    def _on_save_network(self):
        import gui.persistence as persistence
        # Flush any pending in-editor edits the same way Solve does so a
        # Save without a prior Apply still captures what the user sees.
        self._flush_pending_editor_edits()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save network", "",
            "Hydraulics network (*.hydnet.json);;JSON (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".hydnet.json"
        try:
            persistence.save_canvas(path, self)
        except Exception as e:
            dialogs.critical(
                self, "Save failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        QMessageBox.information(self, "Saved", f"Network saved to:\n{path}")

    def _on_load_network(self):
        import gui.persistence as persistence
        path, _ = QFileDialog.getOpenFileName(
            self, "Load network", "",
            "Hydraulics network (*.hydnet.json *.json);;All files (*)",
        )
        if not path:
            return
        try:
            warnings = persistence.load_canvas(path, self)
        except Exception as e:
            dialogs.critical(
                self, "Load failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        # Solve-related state was cleared by load_canvas; disable the
        # Save Results button until the next successful solve.
        self._save_results_btn.setEnabled(False)
        if warnings:
            QMessageBox.warning(
                self, "Network loaded with warnings",
                "\n\n".join(warnings),
            )

    def _on_save_results(self):
        import gui.persistence as persistence
        if self._last_net is None or self._last_result is None:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Choose directory for results bundle",
        )
        if not path:
            return
        try:
            persistence.save_canvas_results(
                path, self, self._last_net, self._last_result,
            )
        except Exception as e:
            dialogs.critical(
                self, "Save results failed",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return
        QMessageBox.information(
            self, "Saved", f"Results written to:\n{path}",
        )

    def _flush_pending_editor_edits(self):
        """Mirror of the Apply-flush at the top of _build_network: push
        whatever's in the active editor page back into node_specs so a
        Save (or anything else that reads node_specs) sees fresh state."""
        if self._cur_node is None:
            return
        t = self.node_specs.get(self._cur_node.id, {}).get("type")
        if t == "source_sink":
            self._apply_source_sink_edits()
        elif t == "junction":
            self._apply_junction_edits()
        elif t == "pipe":
            self._apply_pipe_edits()
        elif t == "fitting":
            self._apply_fitting_edits()
        elif t == "valve":
            self._apply_valve_edits()
        elif t == "check_valve":
            self._apply_check_valve_edits()

    def _build_fluid(self):
        """Return the fluid object passed to the solver.

        Default = Incompressible_Fluid built from the side-panel
        density/viscosity inputs.  CompressibleNetworkScreen overrides
        this to return the pre-built AbstractState from AppState.
        """
        rho = self.f_density.quantity()
        mu  = self.f_viscosity.quantity()
        if rho is None or mu is None:
            raise ValueError("Density and viscosity are required.")
        return Incompressible_Fluid(density=rho, viscosity=mu)

    def _solve_network(self, net, fluid):
        """Call the underlying solver.  The default works for the
        incompressible Network; the compressible subclass overrides to
        pass extra kwargs (P_init, T_init, mdot_init_kgs) supported by
        Compressible_Network.solve().
        """
        return net.solve(fluid)

    # ------------------------------------------------------------------
    # Subclass hooks (default no-ops, overridden by CompressibleNetworkScreen)
    # ------------------------------------------------------------------

    def _add_extra_source_sink_rows(self, form):
        """Optional extra QFormLayout rows on the Source/Sink editor.

        Called between the Q_ext row and the Elevation row.  Used by
        the compressible subclass to add a T row.
        """

    def _default_source_sink_extra(self):
        """Extra default-spec keys for new Source/Sink nodes.

        Returned dict is merged into the standard source/sink defaults.
        """
        return {}

    def _sync_source_sink_extra(self, spec):
        """Push extra fields (e.g. T) from the spec dict into editor widgets."""

    def _apply_source_sink_extra(self, spec):
        """Mutate the in-progress source/sink spec dict to capture extra
        editor fields (e.g. T) before it replaces node_specs[...]."""

    def _extra_kwargs_for_boundary(self, spec, node_name):
        """Optional extra kwargs to pass into net.add_node() for this node.

        Used by the compressible subclass to forward T as a pint Quantity.
        """
        return {}

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
            elif t == "fitting":
                self._apply_fitting_edits()
            elif t == "valve":
                self._apply_valve_edits()
            elif t == "check_valve":
                self._apply_check_valve_edits()

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

        net = self.NETWORK_CLS()
        self._pipe_components = {}
        self._pipe_edge_names = {}
        self._inline_chain_pos = {}

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
                    components, chain_node_ids, end_node = self._walk_chain(
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
                    for pos, nid in enumerate(chain_node_ids):
                        self._pipe_edge_names[nid] = edge_name
                        self._inline_chain_pos[nid] = pos

        # Source/Sink nodes must have exactly one connecting edge.  A
        # multi-edge Source/Sink hides a thermodynamic inconsistency in
        # the compressible solver: if one of those edges happens to carry
        # reversed flow, the outward walk leaves the node at T_spec
        # rather than the energy-balanced mixed T of the actual incoming
        # streams.  Mixing/splitting must be done at Junctions, whose T
        # is back-solved by the energy balance.
        for node in boundary_nodes:
            spec = self.node_specs.get(node.id, {})
            if spec.get("type") != "source_sink":
                continue
            ss_name = node.name()
            touching = [
                e for e in net._edges
                if e.from_node == ss_name or e.to_node == ss_name
            ]
            if len(touching) != 1:
                raise ValueError(
                    f"Source/Sink '{ss_name}' has {len(touching)} "
                    f"connected edges.  Source/Sink nodes must have "
                    f"exactly one connection -- add a Junction node and "
                    f"route mixing/splitting through it."
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
        # Subclass hook: e.g. compressible adds T to source/sink boundaries.
        kwargs.update(self._extra_kwargs_for_boundary(spec, node_name))
        return kwargs

    def _walk_chain(self, target_node, visited_pipes, source_name):
        """Walk forward through inline nodes (pipe/fitting/valve) starting at
        target_node, collecting components, until hitting a boundary node.
        Returns (component_list, chain_node_ids, end_boundary_node);
        end_boundary_node is None if the chain dead-ended at an unknown
        node type.
        """
        components     = []
        chain_node_ids = []
        cur = target_node
        while True:
            t = self.node_specs.get(cur.id, {}).get("type")
            if t in ("source_sink", "junction"):
                return components, chain_node_ids, cur
            if t not in ("pipe", "fitting", "valve", "check_valve"):
                return components, chain_node_ids, None
            if cur.id in visited_pipes:
                raise ValueError(
                    f"Node '{cur.name()}' is part of more than one chain "
                    f"(should be a single in/out node)."
                )
            visited_pipes.add(cur.id)
            if t == "pipe":
                comp = self._build_line_segment_for_pipe(cur)
            elif t == "fitting":
                comp = self._build_fitting_component(cur)
            elif t == "valve":
                comp = self._build_valve_component(cur)
            else:
                comp = self._build_check_valve_component(cur)
            self._pipe_components[cur.id] = comp
            chain_node_ids.append(cur.id)
            components.append(comp)
            outs = []
            for port, targets in cur.connected_output_nodes().items():
                outs.extend(targets)
            if not outs:
                raise ValueError(
                    f"Node '{cur.name()}' has no outgoing connection."
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
            seg = self.LINE_SEGMENT_CLS(
                roughness=rough_q, profile=profile, name=pipe_node.name(),
            )
            # Tag with csv source so net.save() can serialize it as a
            # csv_path reference rather than the full profile (matches
            # Line_Segment.from_csv() behavior).
            seg._csv_path = spec.get("csv_path")
            return seg

        def _opt(name):
            s = spec.get(f"{name}_str", "").strip()
            if not s:
                return None
            return ureg.Quantity(
                float(s), U.to_pint(spec.get(f"{name}_unit", "")),
            )

        return self.LINE_SEGMENT_CLS(
            roughness        = rough_q,
            id_val           = _opt("ID"),
            od_val           = _opt("OD"),
            wt_val           = _opt("WT"),
            length           = _opt("L"),
            elevation_change = _opt("dz"),
            name             = pipe_node.name(),
        )

    def _build_fitting_component(self, node):
        spec = self.node_specs[node.id]
        ft = spec.get("fitting_type", "bend")
        if ft == "bend":
            Di_str = spec.get("Di_str", "").strip()
            if not Di_str:
                raise ValueError(f"Fitting '{node.name()}': pipe ID is required.")
            angle_str = spec.get("angle_str", "").strip()
            if not angle_str:
                raise ValueError(f"Fitting '{node.name()}': bend angle is required.")
            bend_dias_str = spec.get("bend_dias_str", "").strip()
            if not bend_dias_str:
                raise ValueError(f"Fitting '{node.name()}': bend R/D ratio is required.")
            Di_q = ureg.Quantity(float(Di_str), U.to_pint(spec.get("Di_unit", "inch")))
            return self.BEND_CLS(
                Di=Di_q,
                ang_deg=float(angle_str),
                bend_dias=float(bend_dias_str),
                name=node.name(),
            )
        else:
            Di_US_str = spec.get("Di_US_str", "").strip()
            Di_DS_str = spec.get("Di_DS_str", "").strip()
            if not Di_US_str or not Di_DS_str:
                raise ValueError(
                    f"Fitting '{node.name()}': upstream and downstream IDs are required."
                )
            Di_US_q = ureg.Quantity(float(Di_US_str), U.to_pint(spec.get("Di_US_unit", "inch")))
            Di_DS_q = ureg.Quantity(float(Di_DS_str), U.to_pint(spec.get("Di_DS_unit", "inch")))
            return self.CONTRACTION_EXP_CLS(
                Di_US=Di_US_q,
                Di_DS=Di_DS_q,
                name=node.name(),
            )

    def _build_valve_component(self, node):
        spec = self.node_specs[node.id]
        vt   = spec.get("valve_type", "globe")

        def _qty(str_key, unit_key):
            s = spec.get(str_key, "").strip()
            if not s:
                raise ValueError(
                    f"Valve '{node.name()}': {str_key[:-4]} is required."
                )
            return ureg.Quantity(float(s), U.to_pint(spec.get(unit_key, "inch")))

        if vt == "butterfly":
            D_q   = _qty("D_str", "D_unit")
            style = int(spec.get("style_str", "0") or "0")
            K     = fluids.fittings.K_butterfly_valve_Crane(
                D=D_q.to("m").magnitude, style=style,
            )
            return self.VALVE_CLS(Di=D_q, K=K, name=node.name())

        if vt == "user_specified":
            D_q  = _qty("D_str", "D_unit")
            mode = spec.get("spec_mode", "K")
            s    = spec.get("spec_val_str", "").strip()
            if not s:
                raise ValueError(
                    f"Valve '{node.name()}': {mode} value is required."
                )
            val = float(s)
            if mode == "K":
                return self.VALVE_CLS(Di=D_q, K=val, name=node.name())
            if mode == "Cv":
                return self.VALVE_CLS(Di=D_q, Cv=val, name=node.name())
            if mode == "Kv":
                return self.VALVE_CLS(Di=D_q, Kv=val, name=node.name())
            raise ValueError(
                f"Valve '{node.name()}': unknown spec_mode {mode!r}."
            )

        D1_q      = _qty("D1_str", "D1_unit")
        D2_q      = _qty("D2_str", "D2_unit")
        D1_si     = D1_q.to("m").magnitude
        D2_si     = D2_q.to("m").magnitude
        angle     = float(spec.get("angle_str", "0").strip() or "0")

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

        return self.VALVE_CLS(Di=D2_q, K=K, name=node.name())

    def _build_check_valve_component(self, node):
        spec = self.node_specs[node.id]
        cvt  = spec.get("cv_type", "swing")

        def _qty(str_key, unit_key):
            s = spec.get(str_key, "").strip()
            if not s:
                raise ValueError(
                    f"Check valve '{node.name()}': {str_key[:-4]} is required."
                )
            return ureg.Quantity(float(s), U.to_pint(spec.get(unit_key, "inch")))

        if cvt == "swing":
            D_q    = _qty("D_str", "D_unit")
            angled = spec.get("angled_str", "1") != "0"
            K      = fluids.fittings.K_swing_check_valve_Crane(
                D=D_q.to("m").magnitude, angled=angled,
            )
            return self.CHECKVALVE_CLS(Di=D_q, K=K, name=node.name())

        if cvt == "lift":
            D1_q   = _qty("D1_str", "D1_unit")
            D2_q   = _qty("D2_str", "D2_unit")
            angled = spec.get("angled_str", "1") != "0"
            K      = fluids.fittings.K_lift_check_valve_Crane(
                D1=D1_q.to("m").magnitude,
                D2=D2_q.to("m").magnitude,
                angled=angled,
            )
            return self.CHECKVALVE_CLS(Di=D2_q, K=K, name=node.name())

        if cvt == "tilting_disk":
            D_q   = _qty("D_str", "D_unit")
            angle = float(spec.get("angle_str", "5").strip() or "5")
            K     = fluids.fittings.K_tilting_disk_check_valve_Crane(
                D=D_q.to("m").magnitude, angle=angle,
            )
            return self.CHECKVALVE_CLS(Di=D_q, K=K, name=node.name())

        if cvt == "angle_stop":
            D1_q  = _qty("D1_str", "D1_unit")
            D2_q  = _qty("D2_str", "D2_unit")
            style = int(spec.get("style_str", "0") or "0")
            K     = fluids.fittings.K_angle_stop_check_valve_Crane(
                D1=D1_q.to("m").magnitude,
                D2=D2_q.to("m").magnitude,
                style=style,
            )
            return self.CHECKVALVE_CLS(Di=D2_q, K=K, name=node.name())

        # globe_stop
        D1_q  = _qty("D1_str", "D1_unit")
        D2_q  = _qty("D2_str", "D2_unit")
        style = int(spec.get("style_str", "0") or "0")
        K     = fluids.fittings.K_globe_stop_check_valve_Crane(
            D1=D1_q.to("m").magnitude,
            D2=D2_q.to("m").magnitude,
            style=style,
        )
        return self.CHECKVALVE_CLS(Di=D2_q, K=K, name=node.name())

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
        self._refresh_details_button_visibility()
        self._save_results_btn.setEnabled(True)

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
            elif t in ("pipe", "fitting", "valve", "check_valve"):
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

    # ------------------------------------------------------------------
    # Per-node solved-details inspector (button on inline editor pages)
    # ------------------------------------------------------------------

    _DETAILS_BTNS_BY_TYPE = {
        "pipe":        "p_details_btn",
        "fitting":     "fit_details_btn",
        "valve":       "valve_details_btn",
        "check_valve": "cv_details_btn",
    }

    def _refresh_details_button_visibility(self):
        """Show the "Show solved details..." button on the active editor
        page only when (a) the current node is an inline node and (b) we
        have a usable solver result for it.  Called from selection-sync
        and from _render_results.
        """
        # Always hide everything first; the active page will be re-shown
        # below if eligible.
        for attr in self._DETAILS_BTNS_BY_TYPE.values():
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setVisible(False)
        n = self._cur_node
        if n is None or self._last_result is None:
            return
        spec = self.node_specs.get(n.id, {})
        t = spec.get("type")
        attr = self._DETAILS_BTNS_BY_TYPE.get(t)
        if attr is None:
            return
        if not self._have_result_for_inline(n):
            return
        btn = getattr(self, attr, None)
        if btn is not None:
            btn.setVisible(True)

    def _have_result_for_inline(self, node):
        """Default predicate: an inline node has a result if it was bound
        to a component during the last build_network.  Subclasses can
        override if they need stricter checks."""
        return node.id in self._pipe_components

    def _show_current_node_details(self):
        n = self._cur_node
        if n is None or self._last_result is None:
            return
        t = self.node_specs.get(n.id, {}).get("type")
        if t == "pipe":
            rows, profile = self._pipe_details(n)
            cb = (lambda points=profile, name=n.name():
                  self._open_pipe_profile_window(name, points)) if profile else None
            NodeResultsDialog(self, f"Pipe details: {n.name()}", rows, cb).exec()
        elif t == "fitting":
            rows = self._fitting_details(n)
            NodeResultsDialog(self, f"Fitting details: {n.name()}", rows).exec()
        elif t == "valve":
            rows = self._valve_details(n)
            NodeResultsDialog(self, f"Valve details: {n.name()}", rows).exec()
        elif t == "check_valve":
            rows = self._check_valve_details(n)
            NodeResultsDialog(self, f"Check valve details: {n.name()}", rows).exec()

    def _open_pipe_profile_window(self, node_name, profile_points):
        # Keep a reference on self so the non-modal window isn't garbage-
        # collected the moment _show_current_node_details returns.
        win = PipeProfileWindow(
            title=f"Profile: {node_name}",
            profile_points=profile_points,
            x_default="ft",
            p_default=self.d_pressure.currentText(),
            t_default=self._current_temp_unit_or_default(),
            parent=self,
        )
        self._profile_windows = getattr(self, "_profile_windows", [])
        self._profile_windows.append(win)
        win.show()

    def _current_temp_unit_or_default(self):
        """Compressible subclass exposes d_temperature; the base doesn't."""
        combo = getattr(self, "d_temperature", None)
        return combo.currentText() if combo is not None else "degF"

    # ------------------------------------------------------------------
    # Per-type incompressible detail builders.  The compressible subclass
    # overrides these because it sources data from the flat solver-result
    # dict + the AbstractState rather than NetworkResult.component().
    # ------------------------------------------------------------------

    def _pipe_details(self, node):
        """Return (rows, profile_points) for an incompressible pipe.

        profile_points is the per-point list returned by
        ComponentResult.pressure_profile() (already has distance_m, P_Pa,
        v_ms keys) -- PipeProfileWindow consumes it directly.
        """
        comp = self._pipe_components[node.id]
        cr   = self._last_result.component(comp)
        profile = cr.pressure_profile()

        P_unit = self.d_pressure.currentText()
        v_unit = "ft/s"

        v_in_str  = _fmt_velocity(profile[0]["v_ms"],  v_unit)
        v_out_str = _fmt_velocity(profile[-1]["v_ms"], v_unit)
        P_in_str  = _fmt_pressure(cr.P_in_Pa,  P_unit)
        P_out_str = _fmt_pressure(cr.P_out_Pa, P_unit)
        dP_str    = _fmt_pressure_signed(cr.dP_Pa, P_unit)
        Q_str     = self._fmt_flow_for_details(cr.mdot_kgs, cr.Q_m3s)

        rows = [
            ("Mass flow:",    Q_str),
            ("Inlet P:",      P_in_str),
            ("Outlet P:",     P_out_str),
            ("dP:",           dP_str),
            ("Inlet vel.:",   v_in_str),
            ("Outlet vel.:",  v_out_str),
        ]
        return rows, profile

    def _fitting_details(self, node):
        comp = self._pipe_components[node.id]
        cr   = self._last_result.component(comp)
        rho  = float(self._last_result.fluid.density_si)
        D    = _fitting_velocity_diameter(comp)
        v    = abs(cr.mdot_kgs) / (rho * math.pi * D * D / 4.0)

        P_unit = self.d_pressure.currentText()
        rows = [
            ("Mass flow:", self._fmt_flow_for_details(cr.mdot_kgs, cr.Q_m3s)),
            ("dP:",        _fmt_pressure_signed(cr.dP_Pa, P_unit)),
            ("Velocity:",  _fmt_velocity_at_D(v, D)),
        ]
        return rows

    def _valve_details(self, node):
        comp = self._pipe_components[node.id]
        cr   = self._last_result.component(comp)
        rho  = float(self._last_result.fluid.density_si)
        D    = self._valve_smallest_D_si(node)
        v    = abs(cr.mdot_kgs) / (rho * math.pi * D * D / 4.0)

        P_unit = self.d_pressure.currentText()
        rows = [
            ("K-factor:",   f"{comp.K:.3f}"),
            ("Mass flow:",  self._fmt_flow_for_details(cr.mdot_kgs, cr.Q_m3s)),
            ("dP:",         _fmt_pressure_signed(cr.dP_Pa, P_unit)),
            ("Velocity:",   _fmt_velocity_at_D(v, D)),
        ]
        return rows

    def _check_valve_details(self, node):
        # Same shape as a regular valve; reuse so subclasses only override once.
        return self._valve_details(node)

    def _fmt_flow_for_details(self, mdot_kgs, Q_m3s):
        """Format the edge mass flow using the screen's current flow-unit
        selection, picking mass / volumetric source by dimensionality."""
        unit = self.d_flow.currentText()
        val  = self._convert_flow(mdot_kgs, Q_m3s, unit)
        return f"{val:+.4g} {unit}"

    def _valve_smallest_D_si(self, node):
        """Smallest diameter in the flow path through this valve [m].

        Reads the spec rather than the built Valve, because for reducer
        types (globe/gate/plug/ball/lift/angle_stop/globe_stop) only D2
        is preserved on the Valve instance and D1 (the seat bore, often
        the smallest) lives only in the spec dict.
        """
        spec = self.node_specs[node.id]
        t    = spec.get("type")
        if t == "valve":
            vt = spec.get("valve_type", "globe")
            if vt in ("butterfly", "user_specified"):
                return _qty_to_m(spec.get("D_str"), spec.get("D_unit", "inch"))
            D1 = _qty_to_m(spec.get("D1_str"), spec.get("D1_unit", "inch"))
            D2 = _qty_to_m(spec.get("D2_str"), spec.get("D2_unit", "inch"))
            return min(D1, D2)
        if t == "check_valve":
            cvt = spec.get("cv_type", "swing")
            if cvt in ("swing", "tilting_disk"):
                return _qty_to_m(spec.get("D_str"), spec.get("D_unit", "inch"))
            D1 = _qty_to_m(spec.get("D1_str"), spec.get("D1_unit", "inch"))
            D2 = _qty_to_m(spec.get("D2_str"), spec.get("D2_unit", "inch"))
            return min(D1, D2)
        raise ValueError(f"_valve_smallest_D_si called on non-valve type {t!r}")


# ---------------------------------------------------------------------------
# Module-level formatters used by the details inspector (also reused by
# CompressibleNetworkScreen so they live here, not on the class).
# ---------------------------------------------------------------------------

def _fmt_pressure(P_Pa, unit):
    val = ureg.Quantity(P_Pa, "Pa").to(U.to_pint(unit)).magnitude
    return f"{val:.3f} {unit}"


def _fmt_pressure_signed(dP_Pa, unit):
    val = ureg.Quantity(dP_Pa, "Pa").to(U.to_pint(unit)).magnitude
    return f"{val:+.3f} {unit}"


def _fmt_velocity(v_ms, unit):
    val = ureg.Quantity(v_ms, "m/s").to(unit).magnitude
    return f"{val:.3f} {unit}"


def _fmt_velocity_at_D(v_ms, D_si):
    """Velocity line tagged with the reference diameter in inches."""
    D_in = ureg.Quantity(D_si, "m").to("inch").magnitude
    return f"{_fmt_velocity(v_ms, 'ft/s')}  (at D = {D_in:.3f} in)"


def _fmt_temperature(T_K, unit):
    val = ureg.Quantity(T_K, "K").to(U.to_pint(unit)).magnitude
    return f"{val:.3f} {unit}"


def _qty_to_m(value_str, unit):
    if value_str is None or value_str == "":
        raise ValueError("missing diameter spec for velocity calculation")
    return ureg.Quantity(float(value_str), U.to_pint(unit)).to("m").magnitude


def _fitting_velocity_diameter(comp):
    """Velocity-reference diameter for a fitting component [m].

    Bend: the single pipe ID.
    Contraction/Expansion: the smaller of upstream/downstream ID
    (= max velocity point through the abrupt area change).
    """
    if hasattr(comp, "Di_US_si") and hasattr(comp, "Di_DS_si"):
        return min(comp.Di_US_si, comp.Di_DS_si)
    return comp.Di_si
