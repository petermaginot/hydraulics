"""Single fitting/valve calculator -- compressible (gas) regime.

Subclasses SingleFittingScreen and overrides:
  - Component classes -> compressible_flow equivalents
  - Fluid block -> read-only composition summary (AS from AppState.fluid)
  - Operating conditions -> adds inlet temperature field
  - Solve logic -> dP_dT / dmdot_dT with FlowState
  - Throat display -> adds T and Mach number
"""

import math
import traceback
import warnings

import CoolProp.CoolProp as CP

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

import compressible_flow
from component_classes import ureg
from compressible_flow import (
    FlowState,
    ChokedFlowError,
    _build_phase_limits,
    _resolve_mdot,
)
from gui import units as U
from gui.screens.segment import _LabeledField
from gui.screens.single_fitting import SingleFittingScreen
from gui.throat import compressible_throat_state
import compressible_flow as cf


class CompressibleSingleFittingScreen(SingleFittingScreen):

    # Swap to compressible component classes.
    BEND_CLS            = compressible_flow.Bend
    VALVE_CLS           = compressible_flow.Valve
    CHECKVALVE_CLS      = compressible_flow.CheckValve
    ORIFICE_CLS         = compressible_flow.Orifice
    CONTRACTION_EXP_CLS = compressible_flow.Contraction_Expansion

    # Compressible networks don't allow actual-volumetric units; same
    # restriction applies to a single compressible fitting (density varies).
    FLOW_RATE_UNITS   = [
        "kg/s", "kg/h", "lb/h",
        "mol/s", "mol/h",
        "mmscf/day", "mscf/day", "scf/min", "scm/h",
    ]
    FLOW_RATE_DEFAULT = "mmscf/day"

    # No vapor-pressure cavitation check for compressible (gas) fittings.
    SHOW_VAPOR_PRESSURE = False

    # ------------------------------------------------------------------
    # Fluid panel: composition summary from AppState
    # ------------------------------------------------------------------

    def _build_fluid_box(self):
        self._composition_label = QLabel("(no composition defined — go back and build one)")
        self._composition_label.setWordWrap(True)
        self._composition_label.setStyleSheet("color: #444;")
        v = QVBoxLayout()
        v.addWidget(self._composition_label)
        box = QGroupBox("Composition (compressible)")
        box.setLayout(v)
        return box

    def showEvent(self, event):
        self._refresh_composition_label()
        super().showEvent(event)

    def _refresh_composition_label(self):
        AS = getattr(self.state, "fluid", None)
        if AS is None:
            self._composition_label.setText(
                "(no composition defined — go back and build one)"
            )
            return
        try:
            fluids_list = AS.fluid_names()
            fracs = [AS.get_mole_fractions()[i] for i in range(len(fluids_list))]
            parts = [f"{n}: {f:.3g}" for n, f in zip(fluids_list, fracs)]
            eos   = AS.backend_name()
            self._composition_label.setText(
                f"{eos}  |  " + ",  ".join(parts)
            )
        except Exception:
            self._composition_label.setText("(composition defined)")

    # ------------------------------------------------------------------
    # Extra inlet temperature row
    # ------------------------------------------------------------------

    def _add_extra_cond_rows(self, form):
        self.t_inlet = _LabeledField("80", U.TEMPERATURE, "degF")
        form.addRow("Inlet temperature:", self.t_inlet.widget())

    # ------------------------------------------------------------------
    # Override solve
    # ------------------------------------------------------------------

    def _get_fluid_and_inlet_PT(self):
        """Return (AS, P_inlet_Pa, T_inlet_K) anchored to inlet conditions."""
        AS = getattr(self.state, "fluid", None)
        if AS is None:
            raise ValueError(
                "No gas composition defined.  Go back and build one on the "
                "Composition screen."
            )
        P_in_q = self.p_inlet.quantity()
        T_in_q = self.t_inlet.quantity()
        if P_in_q is None:
            raise ValueError("Inlet pressure is required.")
        if T_in_q is None:
            raise ValueError("Inlet temperature is required.")
        P_Pa = P_in_q.to("Pa").magnitude
        T_K  = T_in_q.to("K").magnitude
        return AS, P_Pa, T_K

    def _solve(self, fitting, kind, P_inlet_Pa, P_outlet_Pa, flow_rate, inverse):
        AS, P_inlet_Pa, T_inlet_K = self._get_fluid_and_inlet_PT()

        result_lines = []
        warn_lines   = []

        # Anchor AS to inlet.
        AS.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
        T_cric, P_bar, T_c, P_c = _build_phase_limits(AS)

        inlet_area = fitting.inlet_area_si

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if inverse:
                    if P_outlet_Pa >= P_inlet_Pa:
                        raise ValueError(
                            "Inlet pressure must be greater than outlet pressure."
                        )
                    # Re-anchor before building FlowState.
                    AS.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
                    fs = FlowState(
                        AS, mdot=1e-3, A=inlet_area, z=0.0,
                        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                        T_critical=T_c, P_critical=P_c,
                    )
                    fitting.dmdot_dT(fs, P2=P_outlet_Pa)
                    mdot_kgs   = fs.mdot
                    P_out_res  = fs.P
                    T_out_K    = fs.T
                    v_out      = fs.v
                    Ma_out     = fs.Ma

                    result_lines.append(
                        self._format_mdot_line_comp(mdot_kgs, AS)
                    )
                    result_lines.append(
                        f"{'Outlet pressure:':28s} {self._fmt_P(P_out_res)}"
                    )
                    result_lines.append(
                        f"{'Outlet temperature:':28s} {self._fmt_T(T_out_K)}"
                    )
                    result_lines.append(
                        f"{'dP:':28s} {self._fmt_dP(P_out_res - P_inlet_Pa)}"
                    )
                    result_lines.append(self._fmt_v_line("Outlet velocity:", v_out))
                    result_lines.append(f"{'Outlet Mach:':28s} {Ma_out:.4f}")

                    # Throat state (re-anchor for snapshot)
                    throat = compressible_throat_state(
                        fitting, AS, mdot_kgs,
                        P_inlet_Pa, T_inlet_K, (T_cric, P_bar, T_c, P_c),
                    )
                    if throat:
                        result_lines += self._format_throat_compressible(throat)

                else:
                    mdot = _resolve_mdot(flow_rate, AS)
                    AS.update(CP.PT_INPUTS, P_inlet_Pa, T_inlet_K)
                    fs = FlowState(
                        AS, mdot=mdot, A=inlet_area, z=0.0,
                        T_cricondentherm=T_cric, P_cricondenbar=P_bar,
                        T_critical=T_c, P_critical=P_c,
                    )
                    fitting.dP_dT(fs)
                    P_out_res = fs.P
                    T_out_K   = fs.T
                    v_out     = fs.v
                    Ma_out    = fs.Ma

                    result_lines.append(
                        f"{'Outlet pressure:':28s} {self._fmt_P(P_out_res)}"
                    )
                    result_lines.append(
                        f"{'Outlet temperature:':28s} {self._fmt_T(T_out_K)}"
                    )
                    result_lines.append(
                        f"{'dP:':28s} {self._fmt_dP(P_out_res - P_inlet_Pa)}"
                    )
                    result_lines.append(
                        f"{'dT:':28s} {self._fmt_dT(T_out_K - T_inlet_K)}"
                    )
                    result_lines.append(self._fmt_v_line("Outlet velocity:", v_out))
                    result_lines.append(f"{'Outlet Mach:':28s} {Ma_out:.4f}")

                    # Throat state
                    throat = compressible_throat_state(
                        fitting, AS, mdot,
                        P_inlet_Pa, T_inlet_K, (T_cric, P_bar, T_c, P_c),
                    )
                    if throat:
                        result_lines += self._format_throat_compressible(throat)

            except ChokedFlowError as choke:
                warn_lines.append(
                    f"CHOKED FLOW detected:\n"
                    f"  Max mass flow rate:  {choke.mdot_choked:.4g} kg/s\n"
                    f"  Throat pressure:     {self._fmt_P(choke.P_throat)}\n"
                    f"  Throat temperature:  {self._fmt_T(choke.T_throat)}"
                )
                # Choke exception carries outlet state too
                result_lines.append(f"{'Outlet pressure:':28s} {self._fmt_P(choke.P_outlet)}")
                result_lines.append(f"{'Outlet temperature:':28s} {self._fmt_T(choke.T_outlet)}")

            for w in caught:
                warn_lines.append(str(w.message))

        return result_lines, warn_lines

    # ------------------------------------------------------------------
    # Formatting helpers (compressible additions)
    # ------------------------------------------------------------------

    def _fmt_T(self, T_K):
        T_C = T_K - 273.15
        T_F = T_K * 9 / 5 - 459.67
        return f"{T_K:.2f} K  ({T_C:.2f} °C  /  {T_F:.2f} °F)"

    def _fmt_dT(self, dT_K):
        sign = "+" if dT_K >= 0 else ""
        return f"{sign}{dT_K:.4f} K"

    def _format_mdot_line_comp(self, mdot_kgs, AS):
        unit_lbl = self.flow_rate.combo.currentText()
        target   = U.to_pint(unit_lbl)
        dim      = ureg.Quantity(1.0, target).dimensionality

        _MASS  = ureg.Quantity(1.0, "kg/s").dimensionality
        _MOLAR = ureg.Quantity(1.0, "mol/s").dimensionality

        if dim == _MASS:
            q = ureg.Quantity(mdot_kgs, "kg/s").to(target)
        elif dim == _MOLAR:
            mw = AS.molar_mass()
            q  = ureg.Quantity(mdot_kgs / mw, "mol/s").to(target)
        else:
            # Standard-volume units are registered as mole equivalents
            mw = AS.molar_mass()
            q  = ureg.Quantity(mdot_kgs / mw, "mol/s").to(target)

        return f"{'Solved flow rate:':28s} {q:.4g~P}"

    def _format_throat_compressible(self, throat):
        lines = ["", "--- Estimated vena contracta conditions ---"]
        A    = throat["A_m2"]
        v    = throat["v_ms"]
        P    = throat["P_Pa"]
        T_K  = throat["T_K"]
        Ma   = throat["Ma"]
        D_thr    = math.sqrt(4.0 * A / math.pi) if A > 0 else 0.0
        D_thr_in = ureg.Quantity(D_thr, "m").to("inch").magnitude
        A_in2    = ureg.Quantity(A, "m^2").to("inch^2").magnitude
        P_psi    = ureg.Quantity(P, "Pa").to("psi").magnitude
        v_fps    = ureg.Quantity(v, "m/s").to("ft/s").magnitude
        T_C      = T_K - 273.15
        T_F      = T_K * 9 / 5 - 459.67
        lines.append(f"{'Vena contracta area:':28s} {A*1e6:.2f} mm^2  ({A_in2:.4f} in^2)")
        lines.append(f"{'Vena contracta diameter:':28s} {D_thr*1000:.2f} mm  ({D_thr_in:.3f} in)")
        lines.append(f"{'Vena contracta pressure:':28s} {P/1e6:.4f} MPa  ({P_psi:.2f} psi)")
        lines.append(
            f"{'Vena contracta T:':28s} {T_K:.2f} K  "
            f"({T_C:.2f} °C  /  {T_F:.2f} °F)"
        )
        lines.append(f"{'Vena contracta velocity:':28s} {v:.3f} m/s  ({v_fps:.2f} ft/s)")
        lines.append(f"{'Vena contracta Mach:':28s} {Ma:.4f}")
        return lines
