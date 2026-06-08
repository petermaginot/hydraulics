"""Start screen: pick flow type and workflow."""

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

        title = QLabel("Hydraulics Calculator")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        prompt = QLabel("Choose the workflow:")

        self.rb_incompressible        = QRadioButton("Single pipe segment: Incompressible (liquid)")
        self.rb_compressible          = QRadioButton("Single pipe segment: Compressible (gas)")
        self.rb_fitting_incompressible = QRadioButton("Single fitting/valve: Incompressible (liquid)")
        self.rb_fitting_compressible   = QRadioButton("Single fitting/valve: Compressible (gas)")
        self.rb_network               = QRadioButton("Pipe network (incompressible)")
        self.rb_cnetwork              = QRadioButton("Pipe network (compressible)")
        self.rb_incompressible.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(self.rb_incompressible)
        group.addButton(self.rb_compressible)
        group.addButton(self.rb_fitting_incompressible)
        group.addButton(self.rb_fitting_compressible)
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
        layout.addWidget(self.rb_fitting_incompressible)
        layout.addWidget(self.rb_fitting_compressible)
        layout.addWidget(self.rb_network)
        layout.addWidget(self.rb_cnetwork)
        layout.addStretch()
        layout.addLayout(nav)

    def _on_next(self):
        if self.rb_incompressible.isChecked():
            new_type = "incompressible"
        elif self.rb_compressible.isChecked():
            new_type = "compressible"
        elif self.rb_fitting_incompressible.isChecked():
            new_type = "single_fitting_incompressible"
        elif self.rb_fitting_compressible.isChecked():
            new_type = "single_fitting_compressible"
        elif self.rb_network.isChecked():
            new_type = "network"
        else:
            new_type = "compressible_network"
        if new_type != self.state.flow_type:
            self.state.reset_for_flow_type_change()
        self.state.flow_type = new_type
        self.next_clicked.emit()
