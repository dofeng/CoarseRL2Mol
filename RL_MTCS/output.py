"""Coarse-molecule visualization and tabular export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .geometry import axial_to_cartesian


REFERENCE_NODE_RADIUS = 0.22
REFERENCE_FONT_SIZE = 7
REFERENCE_LABEL_ROTATION_DEGREES = 0.0


def _reference_node_color(su: int):
    """Return the MCTS palette used to generate ``data/demo/23823.png``."""
    color_map = {
        0: (0.6, 0.0, 0.3),
        1: (0.8, 0.0, 0.2),
        2: (0.7, 0.1, 0.4),
        3: (0.5, 0.0, 0.2),
        4: (0.4, 0.0, 0.4),
        5: (0.8, 0.3, 0.6),
        6: (0.6, 0.2, 0.8),
        7: (0.9, 0.8, 0.1),
        8: (0.5, 0.5, 0.5),
        9: (0.7, 0.1, 0.1),
        10: (0.2, 0.4, 0.9),
        11: (0.2, 0.8, 0.4),
        12: (0.9, 0.2, 0.2),
        13: (0.4, 0.7, 0.9),
        14: (0.0, 0.6, 0.8),
        15: (0.0, 0.7, 0.9),
        16: (0.1, 0.8, 1.0),
        17: (0.0, 0.4, 0.6),
        18: (0.0, 0.5, 0.7),
        19: (0.9, 0.4, 0.4),
        20: (0.7, 0.5, 0.9),
        21: (0.6, 0.6, 0.6),
        22: (0.8, 0.5, 0.1),
        23: (0.9, 0.6, 0.2),
        24: (0.9, 0.6, 0.2),
        25: (0.9, 0.6, 0.2),
        26: (0.3, 0.0, 0.9),
        27: (0.5, 0.0, 0.8),
        28: (1.0, 0.2, 0.2),
        29: (0.8, 0.0, 0.0),
        30: (0.9, 0.7, 0.0),
        31: (0.9, 0.9, 0.0),
        32: (0.4, 0.4, 0.4),
    }
    return color_map.get(int(su), (0.6, 0.6, 0.6))


def _cart(axial, scale: float = 1.0):
    # Final 5-membered substitutions may use mean (half-grid) coordinates.
    q, r = float(axial[0]), float(axial[1])
    import math
    return (
        float(scale) * math.sqrt(3.0) * 0.5 * q,
        float(scale) * (r - 0.5 * q),
    )


def _write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_state_png(
    state,
    output_path: str | Path,
    *,
    title: str | None = None,
    visible_node_ids: Iterable[int] | None = None,
) -> str:
    """Render an assembly state with the canonical coarse-graph style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    from model.shared.coarse_graph import (
        EDGE_COLOR,
        EDGE_LENGTH,
        EDGE_LW,
        FONT_COLOR,
        NODE_ALPHA,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    su_map = state.current_node_map()
    visible = (
        {int(node_id) for node_id in visible_node_ids}
        if visible_node_ids is not None
        else None
    )
    positions = {
        int(node_id): _cart(
            state.final_node_positions.get(int(node_id), axial), scale=EDGE_LENGTH
        )
        for node_id, axial in state.node_axial.items()
        if visible is None or int(node_id) in visible
    }
    if not positions:
        raise ValueError("cannot visualize an empty assembly state")

    center_x = 0.5 * (
        min(x for x, _ in positions.values())
        + max(x for x, _ in positions.values())
    )
    center_y = 0.5 * (
        min(y for _, y in positions.values())
        + max(y for _, y in positions.values())
    )
    positions = {
        node_id: (x - center_x, y - center_y)
        for node_id, (x, y) in positions.items()
    }
    all_x = [point[0] for point in positions.values()]
    all_y = [point[1] for point in positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    margin = REFERENCE_NODE_RADIUS * 3.0
    data_width = max(
        (max_x - min_x) + 2.0 * margin,
        float(EDGE_LENGTH) * 3.0,
    )
    data_height = max(
        (max_y - min_y) + 2.0 * margin,
        float(EDGE_LENGTH) * 3.0,
    )
    pixels_per_unit = 80.0
    fig_width = max(4.0, min(data_width * pixels_per_unit / 100.0, 20.0))
    fig_height = max(4.0, min(data_height * pixels_per_unit / 100.0, 20.0))

    fig = Figure(figsize=(fig_width, fig_height), dpi=100)
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    for edge in state.edges:
        left = positions.get(int(edge.source_node_id))
        right = positions.get(int(edge.target_node_id))
        if left is None or right is None:
            continue
        ax.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            zorder=1,
            solid_capstyle="round",
        )
    for node_id, (x, y) in positions.items():
        su = int(su_map.get(int(node_id), -1))
        circle = mpatches.Circle(
            (x, y),
            REFERENCE_NODE_RADIUS,
            facecolor=_reference_node_color(su),
            edgecolor="white",
            linewidth=0.8,
            alpha=NODE_ALPHA,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            str(su),
            ha="center",
            va="center",
            fontsize=REFERENCE_FONT_SIZE,
            fontweight="bold",
            color=FONT_COLOR,
            rotation=REFERENCE_LABEL_ROTATION_DEGREES,
            rotation_mode="anchor",
            zorder=4,
        )
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title or f"coarse graph | nodes={len(positions)}",
        fontsize=9,
        pad=8,
    )
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.1,
    )
    fig.clear()
    return str(path)


def export_final_state(state, output_dir: str | Path) -> Dict[str, str]:
    """Save PNG plus complete node/edge/audit files for a final state."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    molecule = state.to_dict()
    audit = state.audit()
    _write_json(molecule, directory / "final_molecule.json")
    _write_json(audit, directory / "final_audit.json")

    ownership = state.problem.ownership_map()
    su_map = state.current_node_map()
    planned_su_map = state.problem.node_map()
    with (directory / "final_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "node_id", "su_type", "planned_su_type", "unit_id",
            "q", "r", "final_q", "final_r", "x", "y",
        ])
        for node_id, axial in sorted(state.node_axial.items()):
            final_position = state.final_node_positions.get(
                int(node_id), (float(axial[0]), float(axial[1]))
            )
            x, y = _cart(final_position)
            writer.writerow([
                int(node_id), int(su_map.get(int(node_id), -1)),
                int(planned_su_map.get(int(node_id), -1)),
                ownership.get(int(node_id), ""),
                int(axial[0]), int(axial[1]),
                f"{state.final_node_positions.get(int(node_id), (float(axial[0]), float(axial[1])))[0]:.6f}",
                f"{state.final_node_positions.get(int(node_id), (float(axial[0]), float(axial[1])))[1]:.6f}",
                f"{x:.6f}", f"{y:.6f}",
            ])
    with (directory / "final_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_node_id", "target_node_id", "kind", "contract_id", "direction", "hex_neighbor"])
        for edge in state.edges:
            direction = "" if edge.direction is None else f"{edge.direction[0]},{edge.direction[1]}"
            writer.writerow([
                int(edge.source_node_id), int(edge.target_node_id), str(edge.kind),
                str(edge.contract_id), direction, bool(edge.direction is not None),
            ])

    residual_path = directory / "final_unassembled_resources.csv"
    with residual_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "node_id", "su_type", "state", "hop1_ids", "reason",
        ])
        for resource in state.problem.unassembled_closure_resources:
            writer.writerow([
                int(resource.node_id),
                int(resource.su_type),
                str(resource.state),
                json.dumps([int(value) for value in resource.hop1_ids]),
                str(resource.reason),
            ])

    png_path = directory / "final_molecule.png"
    try:
        save_state_png(
            state,
            png_path,
            title=(
                f"Inverse-driven hexagonal coarse molecule | nodes={len(state.node_axial)} "
                f"edges={len(state.edges)} components={audit.get('component_count')} "
                f"fallbacks={len(audit.get('geometry_fallback_contract_ids', []))}"
            ),
        )
        # Keep the historical consumer filename as a compatibility alias.
        subst_path = directory / "subst_result.png"
        subst_path.write_bytes(png_path.read_bytes())
    except Exception as exc:
        _write_json({"visualization_error": str(exc)}, directory / "visualization_error.json")

    return {
        "directory": str(directory),
        "png": str(png_path),
        "molecule_json": str(directory / "final_molecule.json"),
        "nodes_csv": str(directory / "final_nodes.csv"),
        "edges_csv": str(directory / "final_edges.csv"),
        "unassembled_resources_csv": str(residual_path),
        "audit_json": str(directory / "final_audit.json"),
        "subst_png": str(directory / "subst_result.png"),
    }


__all__ = ["export_final_state", "save_state_png"]
