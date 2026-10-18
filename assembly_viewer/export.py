"""Standalone replay, phase images, and streamed video exports."""

from __future__ import annotations

import json
import math
from pathlib import Path
import textwrap

from .recorder import TraceReader, json_text

# Video is a 960×720 screen preview, not the publication raster export.
VIDEO_FRAME_RESOLUTION = 100

def presentation_events(reader, detailed=False):
    from .recorder import TIMELINE_KINDS
    kinds = set(kind for row in TIMELINE_KINDS.values() for kind in row)
    for row in reader.connection.execute("SELECT * FROM events ORDER BY seq"):
        if detailed or row["kind"] in kinds:
            yield reader.event_dict(row)


def export_html(trace_dir, output, *, detailed=False):
    """Self-contained single-canvas replay with persistent phase starting states."""
    from .recorder import TIMELINE_KINDS
    web = Path(__file__).parent / "web"
    page = (web / "index.html").read_text(encoding="utf-8")
    page = page.replace('<link rel="stylesheet" href="/style.css">',
                        "<style>" + (web / "style.css").read_text(encoding="utf-8") + "</style>")
    page = page.replace('<script src="/app.js" defer></script>', "")
    prefix, suffix = page.rsplit("</body>", 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe = lambda item: json_text(item).replace("<", "\\u003c")
    with TraceReader(trace_dir) as reader, output.open("w", encoding="utf-8") as handle:
        handle.write(prefix + '<script type="application/json" id="embedded-trace">{"manifest":')
        handle.write(safe(reader.manifest()))
        handle.write(',"resources":' + safe(reader.resources()))
        handle.write(',"phases":' + safe(reader.phases()))
        handle.write(',"timeline_kinds":' + safe(TIMELINE_KINDS))
        handle.write(',"events":[')
        state_ids = set()
        for index, event in enumerate(presentation_events(reader, detailed)):
            if index:
                handle.write(",")
            handle.write(safe(event))
            state_ids.update(x for x in (event["state_id"], event["parent_state_id"]) if x is not None)
        for phase in reader.phases():
            state_ids.update(x for x in (phase["start_state_id"], phase["end_state_id"]) if x is not None)
        handle.write('],"states":{')
        for index, state_id in enumerate(sorted(state_ids)):
            if index:
                handle.write(",")
            handle.write(json.dumps(str(state_id)) + ":" + safe(reader.state(state_id)))
        handle.write('}}</script><script>' + (web / "app.js").read_text(encoding="utf-8") + "</script></body>" + suffix)
    return str(output)


def render_state(state, output, *, previous=None, title=None, main_only=True, dpi=300,
                 bounds=None, tight=True):
    """Render actual geometry; text and strokes scale with the node diameter."""
    return _render_outputs(state, [output], previous=previous, title=title,
                           main_only=main_only, dpi=dpi, bounds=bounds, tight=tight)[0]


def _render_outputs(state, outputs, *, previous=None, title=None, main_only=True,
                    dpi=300, bounds=None, tight=True):
    """Build one figure for all requested formats, without repeated audits.

    Target bond scale is 25 pt, with 3–20 inch viewports for raster memory.
    Editable SVG/PDF remain zoomable when distant staged components require
    smaller nodes. Video uses a fixed viewport with the same label proportion.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle
    from RL_MTCS.io.output import FONT_COLOR, _reference_node_color
    nodes = state["nodes"]
    if isinstance(nodes, list):
        nodes = {str(node["node_id"]): node for node in nodes}
    edges = state["edges"]
    if isinstance(edges, list):
        edges = {json_text([e["source_node_id"],e["target_node_id"],e["kind"],e["contract_id"]]): e
                 for e in edges}
    connected = set(state.get("completion_report", {}).get("main_component", ()))
    if not connected:
        connected = {node["unit_id"] for node in nodes.values()}
    visible = {gid: node for gid, node in nodes.items()
               if not main_only or node["unit_id"] in connected}
    if not visible:
        visible = nodes
    positions = {gid: (math.sqrt(3)/2*node["final_position"][0],
                       node["final_position"][1]-node["final_position"][0]/2)
                 for gid, node in visible.items()}
    if not visible:
        raise ValueError("cannot export an empty molecule")
    if bounds is None:
        xs, ys = zip(*positions.values())
        bounds = (min(xs)-.7, max(xs)+.7, min(ys)-.7, max(ys)+.7)
    xspan, yspan = bounds[1]-bounds[0], bounds[3]-bounds[2]
    if not (xspan > 0 and yspan > 0 and all(math.isfinite(v) for v in bounds)):
        raise ValueError("figure bounds must have positive finite spans")
    outputs = [Path(output) for output in outputs]
    if not outputs or any(output.suffix.lower() not in {".svg", ".pdf", ".png", ".tif", ".tiff"}
                          for output in outputs):
        raise ValueError("figure output must be SVG, PDF, PNG, or TIFF")
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be positive and finite")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    left, bottom, top = .22, .22, .48 if title else .22
    if tight:
        # The cap bounds raster allocation, including rigid snapshots whose
        # independent components still occupy distant staging coordinates.
        inches_per_unit = 25 / 72
        width = max(3., min(20., xspan * inches_per_unit + 2 * left))
        height = max(3., min(20., yspan * inches_per_unit + bottom + top))
    else:
        width, height = 8., 7.
    if title:
        # Wrap metadata on small figures instead of clipping it at the page.
        title = textwrap.fill(str(title), max(12, int((width - 2 * left) * 72 / 5.4)))
        extra = title.count("\n") * 11 / 72
        top += extra
        if tight:
            height = min(20., height + extra)
    figure = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(figure)
    axis = figure.add_axes((left/width, bottom/height,
                           1-2*left/width, 1-(bottom+top)/height))
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("equal")
    axis.apply_aspect()
    origin = axis.transData.transform((0, 0))
    unit = axis.transData.transform((1, 0))
    points_per_unit = abs(unit[0]-origin[0]) * 72 / figure.dpi
    radius = .24
    diameter_pt = 2 * radius * points_per_unit
    fontsize = .46 * diameter_pt
    edge_width = .052 * diameter_pt
    outline_width = .055 * diameter_pt
    old_nodes = previous["nodes"] if previous else nodes
    old_edges = previous["edges"] if previous else edges
    for key, edge in edges.items():
        a, b = str(edge["source_node_id"]), str(edge["target_node_id"])
        if a in positions and b in positions:
            axis.plot([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]],
                      color="#008f75" if key not in old_edges else "#64748b",
                      linewidth=edge_width, solid_capstyle="round", zorder=1)
    for gid, position in positions.items():
        node = nodes[gid]
        changed = gid not in old_nodes or old_nodes[gid] != node
        color = _reference_node_color(node["su_type"])
        circle = Circle(position, radius, facecolor=color,
                        edgecolor="#008f75" if changed else "white", linewidth=outline_width, zorder=2)
        circle.set_gid(f"node-{node['node_id']}")
        axis.add_patch(circle)
        label = axis.text(*position, str(node["su_type"]), ha="center", va="center",
                         color=FONT_COLOR,
                         fontsize=fontsize, fontfamily="DejaVu Sans", fontweight="bold",
                         clip_on=True, zorder=3)
        label.set_gid(f"label-{node['node_id']}")
    axis.axis("off")
    if title:
        axis.set_title(title, fontsize=9, fontfamily="DejaVu Sans", pad=9)
    # Use explicit family and editable text without changing other figures'
    # global matplotlib settings. PNG resolution does not affect SVG/PDF.
    try:
        with matplotlib.rc_context({"font.family": "sans-serif", "svg.fonttype": "none", "pdf.fonttype": 42}):
            for output in outputs:
                figure.savefig(output, dpi=dpi, facecolor="white", bbox_inches=None)
    finally:
        figure.clear()
    return [str(output) for output in outputs]


def export_molecule(molecule_json, output_dir=None, *, formats=("svg", "pdf", "png"), title=None, dpi=300):
    """Export a saved engine molecule, including runs without a process trace."""
    source = Path(molecule_json)
    state = json.loads(source.read_text(encoding="utf-8"))
    # Accept the engine's complete result or its molecule-only JSON.
    if "molecule" in state:
        state = state["molecule"]
    directory = Path(output_dir) if output_dir else source.parent / "publication_figures"
    return _render_outputs(state, _format_paths(directory / source.stem, formats),
                           main_only=False, title=title, dpi=dpi)


def _format_paths(stem, formats):
    formats = tuple(formats)
    if not formats or any(extension not in {"svg", "pdf", "png", "tif", "tiff"} for extension in formats):
        raise ValueError("formats must contain SVG, PDF, PNG, or TIFF")
    return [Path(str(stem) + "." + extension) for extension in dict.fromkeys(formats)]


def save_state_figures(state, output_path, *, title=None):
    """Instance-local replacement for the engine's PNG writer at stage/final output.

    Exporting happens only after a selected stage has finished, never during
    candidate enumeration or forward evaluation. All original data files stay
    under the engine's normal output interface.
    """
    paths = _format_paths(Path(output_path).with_suffix(""), ("svg", "pdf", "png"))
    _render_outputs(state.to_dict(), paths, main_only=False, title=title)
    return str(output_path)


def export_frames(trace_dir, output_dir, *, kind="stages", formats=("svg", "pdf", "png"), main_only=True):
    output = Path(output_dir)
    paths = []
    with TraceReader(trace_dir) as reader:
        prefix="SYNTHETIC DEMO | " if reader.manifest().get("synthetic") else ""
        query = "SELECT * FROM events WHERE kind=? ORDER BY seq"
        event_kind = "stage.complete" if kind == "stages" else "formal.connection"
        for row in reader.connection.execute(query, (event_kind,)):
            event = reader.event_dict(row)
            state = reader.state(event["state_id"])
            previous = reader.state(event["parent_state_id"]) if event["parent_state_id"] else None
            filenames = _format_paths(output / f"{event['seq']:06d}_{event['phase']}", formats)
            paths.extend(_render_outputs(state, filenames, previous=previous, main_only=main_only,
                         title=f"{prefix}{event['phase']} | step {event['seq']} | actual {event['elapsed']:.1f} s"))
    return paths


def export_video(trace_dir, output, *, duration_ms=350, scope="formal", main_only=True):
    """Spool frames using one fixed view, then encode GIF/MP4/WebM."""
    import tempfile
    import subprocess
    # Spool raster frames to disk rather than accumulate them in memory.
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("GIF/video export requires ffmpeg. Offline HTML and SVG/PNG need no encoder.")
    with tempfile.TemporaryDirectory(prefix="assembly-frames-") as temporary, TraceReader(trace_dir) as reader:
        prefix="SYNTHETIC DEMO | " if reader.manifest().get("synthetic") else ""
        from .recorder import TIMELINE_KINDS
        if scope not in TIMELINE_KINDS:
            raise ValueError("video scope must be formal, live, or forward")
        kinds = TIMELINE_KINDS[scope]
        query = "SELECT * FROM events WHERE state_id IS NOT NULL AND kind IN (" + ",".join("?" for _ in kinds) + ")"
        if scope == "live":
            # Match the UI: forward assessment updates status, never jumps the
            # displayed formal/trial molecule into later hypothetical stages.
            query += " AND scope != 'forward'"
        query += " ORDER BY seq"
        arguments = kinds
        low_x, high_x, low_y, high_y = math.inf,-math.inf,math.inf,-math.inf
        for row in reader.connection.execute(query,arguments):
            snapshot=reader.state(row["state_id"])
            main=set(snapshot["completion_report"]["main_component"])
            for node in snapshot["nodes"].values():
                if main_only and node["unit_id"] not in main:
                    continue
                q,r=node["final_position"]
                x,y=math.sqrt(3)/2*q,r-q/2
                low_x,high_x,low_y,high_y=min(low_x,x),max(high_x,x),min(low_y,y),max(high_y,y)
        bounds=(low_x-1,high_x+1,low_y-1,high_y+1) if math.isfinite(low_x) else None
        count = 0
        for row in reader.connection.execute(query, arguments):
            event = reader.event_dict(row)
            render_state(reader.state(event["state_id"]), Path(temporary) / f"{count:06d}.png",
                         main_only=main_only, dpi=VIDEO_FRAME_RESOLUTION,bounds=bounds,tight=False,
                         title=f"{prefix}{event['phase']} | {event['kind']} | frame {count+1} | actual {event['elapsed']:.1f} s")
            count += 1
        if not count:
            raise ValueError("trace has no initialized molecular state")
        output = Path(output).resolve()
        if output.suffix.lower() not in {".mp4",".gif",".webm"}:
            raise ValueError("video output must have .mp4, .gif, or .webm suffix")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-loglevel","error", "-framerate", str(1000/max(1, duration_ms)), "-i",
                        str(Path(temporary) / "%06d.png"), "-vf",
                        "scale=960:720:force_original_aspect_ratio=decrease,pad=960:720:(ow-iw)/2:(oh-ih)/2:white",
                        str(output)], check=True, capture_output=True)
    return str(output)
