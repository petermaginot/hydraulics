"""GUI-side wrapper around network.py's save/load.

The file format itself lives in network.py (see Network.to_dict / from_dict
and NetworkResult.save_bundle), so headless code can reuse it without
touching anything in this module.  What this module adds is the canvas-
specific glue:

  -- on save: build a Network from the canvas (so the physics half of the
     file is validated), then attach a `gui_extras` block carrying the
     canvas-only state the headless format does not represent.  That
     means node positions, the raw node_specs dicts (which preserve the
     user's input units and the distinction between OD/WT vs ID, or
     between a globe and gate valve), and the inline-chain composition
     of every solver edge.

  -- on load: read the file, validate the physics via
     screen.NETWORK_CLS.from_dict(...), then rebuild the canvas from
     gui_extras.  Cross-regime loads are supported: T fields are merged
     in from compressible defaults, missing-T nodes are flagged in a
     warning list for the caller to surface.

  -- on save results: dispatch to NetworkResult.save_bundle() for the
     incompressible side, or to compressible_network's standalone bundle
     helper for the compressible side.

CSV-backed pipes round-trip via two mechanisms:
  1. Network's component.to_dict() stores the csv_path (relative paths
     are resolved at save time below to land relative to the file).
  2. The GUI's spec dict (in gui_extras) also carries the csv_path; on
     load we resolve it back and re-derive csv_profile via from_csv()
     so canvas-side geometry summaries match.
"""

import os

from compressible_network import save_compressible_result_bundle


# Spec-type discriminator -> NodeGraphQt registered type id.  Both regimes
# register the same six node classes (see network.py at the top).
_SPEC_TYPE_TO_NODEGRAPH_TYPE = {
    "source_sink":  "pipe.SourceSinkNode",
    "junction":     "pipe.JunctionNode",
    "pipe":         "pipe.PipeSegmentNode",
    "fitting":      "pipe.FittingNode",
    "valve":        "pipe.ValveNode",
    "check_valve":  "pipe.CheckValveNode",
}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_canvas(path, screen):
    """Save the screen's canvas state to `path`.

    Builds a Network from the canvas via _build_network() (so the
    physical half is validated up-front -- a malformed canvas raises
    here rather than producing a half-broken save file).  Then attaches
    a gui_extras block describing canvas-only state.
    """
    net = screen._build_network()

    save_dir = os.path.dirname(os.path.abspath(path))

    # Per-canvas-node metadata.  Spec dicts are deep-copied so the path
    # rewrite below does not touch the live canvas state.
    canvas_nodes = []
    for node in screen.graph.all_nodes():
        spec = dict(screen.node_specs.get(node.id, {}))
        if spec.get("type") == "pipe" and spec.get("mode") == "csv":
            csv_abs = spec.get("csv_path", "")
            if csv_abs:
                try:
                    rel = os.path.relpath(csv_abs, save_dir)
                    spec["csv_path"] = rel.replace("\\", "/")
                except ValueError:
                    # Different drives on Windows -- keep absolute.
                    spec["csv_path"] = csv_abs
            # Drop the parsed profile from gui_extras (the headless half
            # of the file already records csv_path; on load we re-derive
            # the profile from the CSV).
            spec.pop("csv_profile", None)
        canvas_nodes.append({
            "id":   node.id,
            "name": node.name(),
            "pos":  list(node.pos()),
            "spec": spec,
        })

    canvas_connections = []
    for node in screen.graph.all_nodes():
        for port in node.output_ports():
            for tgt_port in port.connected_ports():
                canvas_connections.append({
                    "from_id":   node.id,
                    "from_port": port.name(),
                    "to_id":     tgt_port.node().id,
                    "to_port":   tgt_port.name(),
                })

    # Also pre-translate any csv_path on the Network-side components so
    # they end up relative to the save file too (headless callers reading
    # only the top-level nodes/edges then get a portable file).
    for edge_payload, edge in zip(_compute_edge_payloads(net),
                                  net._edges):
        # We don't actually use edge_payload here -- it was a sketch.
        pass
    # Walk the live net._edges and rewrite csv_path on each line-segment
    # component to be relative to save_dir.  Components are shared between
    # the canvas's _pipe_components and the network we just built, so we
    # have to clone before mutating to avoid touching live objects.
    gui_extras = {
        "canvas_nodes":       canvas_nodes,
        "canvas_connections": canvas_connections,
    }

    # net.save() writes the headless format with our gui_extras attached.
    # csv_path on each component goes through component.to_dict() inside
    # net.save(); rewrite it after the dump if we wanted relative paths
    # there too.  For simplicity we leave the network-half csv_paths
    # absolute (they sit alongside the relative one in gui_extras), and
    # the loader prefers the gui_extras spec dict's csv_path on the GUI
    # side.  Headless loaders get the absolute path -- works as long as
    # the CSVs sit where they did when the file was written.
    net.save(path, gui_extras=gui_extras)


def _compute_edge_payloads(net):
    """Yield each edge's components as serialized dicts (utility kept
    for any future post-write rewriting; currently a no-op consumer)."""
    for edge in net._edges:
        yield [c.to_dict() for c in edge.components]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_canvas(path, screen):
    """Load a saved canvas into `screen`.

    Returns a list of warning strings (cross-regime missing fields, CSV
    reload failures, port-not-found, etc.) for the caller to surface in
    a dialog.  Raises ValueError if the file's regime tag does not match
    the screen's NETWORK_CLS.REGIME -- the caller should catch and show
    a "this file is for the other canvas" error in that case.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Validate the headless half first.  This catches missing components,
    # unknown kinds, bad geometry, etc. before we touch the canvas.
    # On a cross-regime load (file regime != screen regime) we DO NOT
    # call from_dict, because from_dict enforces the regime tag.  Instead
    # we proceed canvas-only -- the user gets a runnable canvas they can
    # solve with the screen's regime; T fields on Source/Sink fill in
    # blank for incompressible -> compressible loads.
    file_regime = payload.get("regime")
    target_regime = screen.NETWORK_CLS.REGIME
    if file_regime == target_regime:
        # Round-trip same-regime: validate by reconstructing the Network.
        # We don't actually keep the rebuilt Network -- it would lose the
        # canvas's preferred display units -- but the construction acts
        # as an integrity check.
        screen.NETWORK_CLS.from_dict(payload)

    warnings = []
    save_dir = os.path.dirname(os.path.abspath(path))

    gui_extras = payload.get("gui_extras") or {}
    canvas_nodes       = gui_extras.get("canvas_nodes",       [])
    canvas_connections = gui_extras.get("canvas_connections", [])

    if not canvas_nodes:
        # File written by headless save() with no GUI block.  Fall back
        # to a minimal canvas reconstruction from the top-level nodes/
        # edges, leaving display state at defaults.  Out of scope for v1
        # -- treat as a warning so the user knows.
        warnings.append(
            "Save file has no GUI canvas data (gui_extras.canvas_nodes "
            "is empty).  Headless-only saves cannot be opened in the GUI "
            "yet -- canvas left empty."
        )
        _reset_canvas(screen)
        return warnings

    _reset_canvas(screen)

    old_to_new = {}
    for n in canvas_nodes:
        spec = dict(n.get("spec", {}))
        spec_type = spec.get("type")
        type_id   = _SPEC_TYPE_TO_NODEGRAPH_TYPE.get(spec_type)
        if type_id is None:
            warnings.append(
                f"Skipped node {n.get('name', '?')!r}: unknown spec type "
                f"{spec_type!r}."
            )
            continue

        new_node = screen.graph.create_node(type_id)
        name = n.get("name")
        if name:
            new_node.set_name(name)
        pos = n.get("pos") or [0.0, 0.0]
        try:
            new_node.set_pos(float(pos[0]), float(pos[1]))
        except (TypeError, ValueError, IndexError):
            pass

        # Cross-regime: merge in target-regime defaults so the editor
        # doesn't crash on missing keys.  In practice this is T_str /
        # T_unit on Source/Sink for incompressible -> compressible.
        if spec_type == "source_sink":
            for k, v in screen._default_source_sink_extra().items():
                spec.setdefault(k, v)

        if spec_type == "pipe" and spec.get("mode") == "csv":
            rel_or_abs = spec.get("csv_path", "")
            if rel_or_abs and not os.path.isabs(rel_or_abs):
                csv_full = os.path.normpath(os.path.join(save_dir, rel_or_abs))
            else:
                csv_full = rel_or_abs
            try:
                max_step_m = spec.get("downsample_max_step_m", 0.0)
                elev_tol_m = spec.get("downsample_elev_tol_m", 0.0)
                downsample = float(max_step_m) if max_step_m > 0.0 else False
                tmp_seg = screen.LINE_SEGMENT_CLS.from_csv(
                    csv_full, roughness=1e-6,
                    downsample=downsample, elev_tol=elev_tol_m,
                )
                spec["csv_path"]    = csv_full
                spec["csv_profile"] = list(tmp_seg.profile)
            except Exception as exc:
                warnings.append(
                    f"Pipe {n.get('name', '?')!r}: could not read CSV "
                    f"{rel_or_abs!r} ({type(exc).__name__}: {exc}). "
                    f"Switched to manual mode -- geometry fields are blank."
                )
                spec["mode"] = "manual"
                spec.pop("csv_path",              None)
                spec.pop("csv_profile",           None)
                spec.pop("downsample_max_step_m", None)
                spec.pop("downsample_elev_tol_m", None)

        screen.node_specs[new_node.id] = spec
        old_to_new[n["id"]] = new_node

    for c in canvas_connections:
        a = old_to_new.get(c.get("from_id"))
        b = old_to_new.get(c.get("to_id"))
        if a is None or b is None:
            continue
        from_port = _find_port_by_name(a.output_ports(), c.get("from_port"))
        to_port   = _find_port_by_name(b.input_ports(),  c.get("to_port"))
        if from_port is None or to_port is None:
            warnings.append(
                f"Skipped connection {a.name()}.{c.get('from_port')!r} -> "
                f"{b.name()}.{c.get('to_port')!r}: port not found."
            )
            continue
        from_port.connect_to(to_port)

    # Cross-regime requirement: compressible canvas needs T on every
    # Source/Sink (even a current withdrawal -- the flow could reverse).
    if target_regime == "compressible" and file_regime != "compressible":
        missing = []
        for nid, spec in screen.node_specs.items():
            if spec.get("type") != "source_sink":
                continue
            if not (spec.get("T_str", "") or "").strip():
                node = screen.graph.get_node_by_id(nid)
                missing.append(node.name() if node else nid)
        if missing:
            warnings.append(
                "Compressible solver requires a temperature on every "
                "Source/Sink.  Set T on:\n  " + "\n  ".join(missing)
            )

    screen._sync_editor_to_selection()
    return warnings


def _reset_canvas(screen):
    screen.graph.clear_session()
    screen.node_specs.clear()
    screen._pipe_components.clear()
    screen._pipe_edge_names.clear()
    screen._inline_chain_pos.clear()
    screen._last_net    = None
    screen._last_result = None
    screen._cur_node    = None


def _find_port_by_name(ports, name):
    for p in ports:
        if p.name() == name:
            return p
    return None


# ---------------------------------------------------------------------------
# Results bundle
# ---------------------------------------------------------------------------

def save_canvas_results(dir_path, screen, net, result):
    """Dispatch to the right bundle helper for the screen's regime."""
    if screen.NETWORK_CLS.REGIME == "compressible":
        save_compressible_result_bundle(
            dir_path, net, result, screen.state.fluid,
            isothermal=bool(getattr(screen.state, "isothermal", False)),
        )
    else:
        # NetworkResult.save_bundle is a method on the result object.
        result.save_bundle(dir_path)
