"""Shared state passed between GUI screens.

A single AppState instance is owned by MainWindow and handed to every screen.
Each screen reads what it needs and writes its results back; the next screen
picks them up.  This keeps screens independent (no direct refs to each other)
without resorting to global module-level state.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class AppState:
    # Set on the Start screen.  Determines which Line_Segment / Fluid classes
    # the later screens instantiate.
    flow_type: str = "incompressible"   # "incompressible" or "compressible"

    # Set on the Segment screen.  A fully-constructed Line_Segment of the
    # appropriate subclass (incompressible.Line_Segment or
    # compressible_flow.Line_Segment), with geometry + roughness baked in.
    segment: Any = None

    # Set on the Fluid screen.
    # Incompressible: Incompressible_Fluid instance.
    # Compressible : CoolProp AbstractState, already updated to (P_inlet, T_inlet).
    fluid: Any = None

    # pint Quantity.  Volumetric/mass for incompressible; mass/molar/std-vol
    # for compressible.
    flow_rate: Any = None

    # Inlet pressure (Pa).  Used as P0 for the pressure-profile call.
    # For incompressible the user supplies it on the fluid screen.
    # For compressible it is the P used to update the AbstractState.
    P_inlet_Pa: Optional[float] = None

    # Inlet temperature (K).  Compressible only; None for incompressible.
    T_inlet_K: Optional[float] = None

    # Compressible-only calculation mode.  True = constant T through each
    # slice; False = adiabatic energy balance (default).  Ignored for
    # incompressible flow.
    isothermal: bool = False

    # Populated by the Results screen after a successful calculate().
    # Normalized to a list of dicts with keys: distance_m, P_Pa, v_ms,
    # and (compressible only) T_K.
    results: Optional[List[dict]] = field(default=None)

    def reset_for_flow_type_change(self):
        """Drop every field that depends on flow_type.

        Called by the Start screen when the user picks a different regime.
        The Line_Segment, fluid object, flow rate units, and inlet anchors
        are all regime-specific; reusing them after a switch caused a bug
        where an incompressible solver was handed a compressible segment
        (or vice versa).  Easier to force the user to redefine than to
        translate.
        """
        self.segment      = None
        self.fluid        = None
        self.flow_rate    = None
        self.P_inlet_Pa   = None
        self.T_inlet_K    = None
        self.isothermal   = False
        self.results      = None
