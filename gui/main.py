"""MainWindow + run() entry point.

A QStackedWidget switches between the three workflow screens.  Each screen
owns a reference to a single AppState; navigation signals are wired here.
"""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from gui.state import AppState
from gui.screens.start                import StartScreen
from gui.screens.segment              import SegmentScreen
from gui.screens.fluid                import FluidScreen
from gui.screens.results              import ResultsScreen
from gui.screens.network              import NetworkScreen
from gui.screens.composition          import CompressibleCompositionScreen
from gui.screens.compressible_network import CompressibleNetworkScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydraulics")
        self.resize(900, 700)

        self.state = AppState()

        self.start_screen        = StartScreen(self.state)
        self.segment_screen      = SegmentScreen(self.state)
        self.fluid_screen        = FluidScreen(self.state)
        self.results_screen      = ResultsScreen(self.state)
        self.network_screen      = NetworkScreen(self.state)
        self.composition_screen  = CompressibleCompositionScreen(self.state)
        self.cnetwork_screen     = CompressibleNetworkScreen(self.state)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.start_screen)
        self.stack.addWidget(self.segment_screen)
        self.stack.addWidget(self.fluid_screen)
        self.stack.addWidget(self.results_screen)
        self.stack.addWidget(self.network_screen)
        self.stack.addWidget(self.composition_screen)
        self.stack.addWidget(self.cnetwork_screen)
        self.setCentralWidget(self.stack)

        # Start.Next routes by flow_type.  Linear point-to-point flows
        # go to the segment screen; the two network flows go to their
        # canvas (incompressible) or to the composition definition step
        # first (compressible).
        self.start_screen.next_clicked.connect(self._on_start_next)
        self.network_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )
        self.segment_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )
        self.segment_screen.next_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.fluid_screen)
        )
        self.fluid_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.segment_screen)
        )
        self.fluid_screen.next_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.results_screen)
        )
        self.results_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.fluid_screen)
        )
        self.composition_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )
        self.composition_screen.next_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.cnetwork_screen)
        )
        self.cnetwork_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.composition_screen)
        )

    def _on_start_next(self):
        if self.state.flow_type == "network":
            self.stack.setCurrentWidget(self.network_screen)
        elif self.state.flow_type == "compressible_network":
            self.stack.setCurrentWidget(self.composition_screen)
        else:
            self.stack.setCurrentWidget(self.segment_screen)


def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
