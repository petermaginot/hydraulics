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
    # the later screens instantiate, or whether to route the user into the
    # network-topology screen instead.
    # "incompressible" / "compressible" -> point-to-point pipeline.
    # "network" / "compressible_network" -> P&ID-style canvas.
    flow_type: str = "incompressible"

    # Set on the Segment screen.  A fully-constructed Line_Segment of the
    # appropriate subclass (incompressible.Line_Segment or
    # compressible_flow.Line_Segment), with geometry + roughness baked in.
    segment: Any = None

    # Set on the single-fitting screen.  A built component instance
    # (Bend / Valve / CheckValve / Orifice / Contraction_Expansion),
    # incompressible or compressible class depending on flow_type.
    fitting: Any = None
    # "bend" / "valve" / "check_valve" / "orifice" / "contraction_expansion"
    fitting_kind: Optional[str] = None

    # Set on the Fluid screen.
    # Incompressible: Incompressible_Fluid instance.
    # Compressible : CoolProp AbstractState, already updated to (P_inlet, T_inlet).
    fluid: Any = None

    # pint Quantity.  Volumetric/mass for incompressible; mass/molar/std-vol
    # for compressible.  None in inverse solve_mode (flow rate is the unknown).
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

    # Which inverse problem the Results screen solves.
    #   "forward" : user supplies flow_rate, solver returns pressure profile.
    #   "inverse" : user supplies P_outlet_Pa, solver returns mass flow rate
    #               plus the resulting pressure profile.
    solve_mode: str = "forward"

    # Target outlet pressure (Pa, absolute) for the inverse solve.  None
    # in forward mode.
    P_outlet_Pa: Optional[float] = None

    # Display unit (string label, e.g. "BBL/D") for the solved mdot in the
    # Results summary.  Captured from the flow-rate dropdown on the Fluid
    # screen so the inverse-mode result is shown in the unit the user was
    # already looking at.  None in forward mode.
    flow_rate_display_unit: Optional[str] = None

    # Populated by the Results screen after a successful inverse solve;
    # mass flow rate in kg/s.  None until then and in forward mode.
    solved_mdot_kgs: Optional[float] = None

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
        self.segment                = None
        self.fitting                = None
        self.fitting_kind           = None
        self.fluid                  = None
        self.flow_rate              = None
        self.P_inlet_Pa             = None
        self.T_inlet_K              = None
        self.isothermal             = False
        self.solve_mode             = "forward"
        self.P_outlet_Pa            = None
        self.flow_rate_display_unit = None
        self.solved_mdot_kgs        = None
        self.results                = None
