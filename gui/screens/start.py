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

        self.rb_incompressible = QRadioButton("Incompressible (liquid)")
        self.rb_compressible   = QRadioButton("Compressible (gas)")
        self.rb_incompressible.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(self.rb_incompressible)
        group.addButton(self.rb_compressible)

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
        layout.addStretch()
        layout.addLayout(nav)

    def _on_next(self):
        new_type = (
            "incompressible" if self.rb_incompressible.isChecked() else "compressible"
        )
        # Regime change invalidates the previously-built segment and fluid:
        # they are typed by regime and the wrong solver will be invoked
        # downstream.  Drop them so the segment screen forces a rebuild.
        if new_type != self.state.flow_type:
            self.state.reset_for_flow_type_change()
        self.state.flow_type = new_type
        self.next_clicked.emit()
