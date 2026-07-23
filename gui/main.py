"""MainWindow + run() entry point.

A QStackedWidget switches between the workflow screens.  Each screen
owns a reference to a single AppState; navigation signals are wired here.
"""

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from gui.state import AppState
from gui.screens.start                       import StartScreen
from gui.screens.segment                     import SegmentScreen
from gui.screens.fluid                       import FluidScreen
from gui.screens.results                     import ResultsScreen
from gui.screens.network                     import NetworkScreen
from gui.screens.composition                 import CompressibleCompositionScreen
from gui.screens.compressible_network        import CompressibleNetworkScreen
from gui.screens.single_fitting              import SingleFittingScreen
from gui.screens.compressible_single_fitting import CompressibleSingleFittingScreen


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
        self.fitting_screen      = SingleFittingScreen(self.state)
        self.cfitting_screen     = CompressibleSingleFittingScreen(self.state)

        # Display-unit combos live on the composition screen; wire them to
        # both the compressible network screen and the compressible fitting screen.
        self.cnetwork_screen.d_pressure    = self.composition_screen.d_pressure
        self.cnetwork_screen.d_flow        = self.composition_screen.d_flow
        self.cnetwork_screen.d_temperature = self.composition_screen.d_temperature
        for _combo in (self.composition_screen.d_pressure,
                       self.composition_screen.d_flow,
                       self.composition_screen.d_temperature):
            _combo.currentTextChanged.connect(
                self.cnetwork_screen._rerender_with_current_units
            )

        self.stack = QStackedWidget()
        self.stack.addWidget(self.start_screen)
        self.stack.addWidget(self.segment_screen)
        self.stack.addWidget(self.fluid_screen)
        self.stack.addWidget(self.results_screen)
        self.stack.addWidget(self.network_screen)
        self.stack.addWidget(self.composition_screen)
        self.stack.addWidget(self.cnetwork_screen)
        self.stack.addWidget(self.fitting_screen)
        self.stack.addWidget(self.cfitting_screen)
        self.setCentralWidget(self.stack)

        # Start.Next routes by flow_type.
        self.start_screen.next_clicked.connect(self._on_start_next)

        # Network screens
        self.network_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )

        # Segment / Fluid / Results pipeline
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

        # Composition -> compressible network or compressible single fitting
        self.composition_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )
        self.composition_screen.next_clicked.connect(
            self._on_composition_next
        )
        self.cnetwork_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.composition_screen)
        )

        # Single fitting screens
        self.fitting_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_screen)
        )
        self.cfitting_screen.back_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.composition_screen)
        )

    def _on_start_next(self):
        ft = self.state.flow_type
        if ft == "network":
            self.stack.setCurrentWidget(self.network_screen)
        elif ft == "compressible_network":
            self.stack.setCurrentWidget(self.composition_screen)
        elif ft == "single_fitting_incompressible":
            self.stack.setCurrentWidget(self.fitting_screen)
        elif ft == "single_fitting_compressible":
            self.stack.setCurrentWidget(self.composition_screen)
        else:
            self.stack.setCurrentWidget(self.segment_screen)

    def _on_composition_next(self):
        ft = self.state.flow_type
        if ft == "single_fitting_compressible":
            self.stack.setCurrentWidget(self.cfitting_screen)
        else:
            self.stack.setCurrentWidget(self.cnetwork_screen)


def run():
    app = QApplication(sys.argv)
    # Consolas on Windows; monospace fallbacks for macOS / Linux so the app
    # stays monospaced everywhere and never errors on a missing family.  Only
    # the family changes -- the platform default point size is preserved.
    f = app.font()
    f.setFamilies(["Consolas", "Menlo", "DejaVu Sans Mono", "Courier New"])
    f.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(f)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
