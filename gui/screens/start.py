"""Start screen: pick incompressible (liquid) or compressible (gas) flow."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class StartScreen(QWidget):
    next_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        title = QLabel("Hydraulics — point-to-point pipe segment")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        prompt = QLabel("Choose the flow regime:")

        self.rb_incompressible = QRadioButton("Point-to-point: Incompressible (liquid)")
        self.rb_compressible   = QRadioButton("Point-to-point: Compressible (gas)")
        self.rb_network        = QRadioButton("Pipe network (incompressible)")
        self.rb_cnetwork       = QRadioButton("Pipe network (compressible)")
        self.rb_incompressible.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(self.rb_incompressible)
        group.addButton(self.rb_compressible)
        group.addButton(self.rb_network)
        group.addButton(self.rb_cnetwork)

        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self._on_next)

        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(next_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(prompt)
        layout.addWidget(self.rb_incompressible)
        layout.addWidget(self.rb_compressible)
        layout.addWidget(self.rb_network)
        layout.addWidget(self.rb_cnetwork)
        layout.addStretch()
        layout.addLayout(nav)

    def _on_next(self):
        if self.rb_incompressible.isChecked():
            new_type = "incompressible"
        elif self.rb_compressible.isChecked():
            new_type = "compressible"
        elif self.rb_network.isChecked():
            new_type = "network"
        else:
            new_type = "compressible_network"
        # Regime change invalidates the previously-built segment and fluid:
        # they are typed by regime and the wrong solver will be invoked
        # downstream.  Drop them so the segment screen forces a rebuild.
        if new_type != self.state.flow_type:
            self.state.reset_for_flow_type_change()
        self.state.flow_type = new_type
        self.next_clicked.emit()
