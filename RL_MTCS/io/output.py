"""Coarse-molecule visualization and tabular export."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable
from RL_MTCS.geometry import hex_edge_valid

REFERENCE_NODE_RADIUS = 0.22

def contract_rows(contracts):
    return [
        {
            "contract_id": str(contract.contract_id),
            "kind": str(contract.kind),
            "source_unit_id": str(contract.source_unit_id),
            "source_node_id": int(contract.source_node_id),
            "source_port_id": str(contract.source_port_id),
            "target_unit_id": str(contract.target_unit_id),
            "target_node_id": int(contract.target_node_id),
            "target_port_id": str(contract.target_port_id),
            "mandatory": bool(contract.mandatory),
            "fixed": bool(contract.fixed),
        }
        for contract in contracts
    ]

REFERENCE_FONT_SIZE = 7
REFERENCE_LABEL_ROTATION_DEGREES = 0.0
# Keep the coarse-graph palette without importing its molecule/NMR stack
# (torch, PyG, RDKit) merely to render stage PNGs.
EDGE_COLOR = (0.5, 0.5, 0.5)
EDGE_LENGTH = 1.0
EDGE_LW = 1.5
FONT_COLOR = "white"
NODE_ALPHA = 0.9


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
    su_types: Dict[int, int] | None = None,
    show_node_ids: bool = False,
) -> str:
    """Render an assembly state with the canonical coarse-graph style."""
    # Jupyter/conda installations often make ``~/.config/matplotlib``
    # read-only.  Use a small writable process-local cache so stage rendering
    # does not emit warnings or fail before the first PNG is written.
    if not os.environ.get("MPLCONFIGDIR"):
        mpl_cache = Path("/tmp/graphnmr-mplconfig")
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    su_map = state.current_node_map()
    if su_types is not None:
        su_map.update({int(node_id): int(su) for node_id, su in su_types.items()})
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
        if show_node_ids:
            ax.text(
                x + REFERENCE_NODE_RADIUS * 1.15,
                y + REFERENCE_NODE_RADIUS * 1.15,
                str(node_id),
                ha="left", va="bottom", fontsize=5, color="#333333", zorder=5,
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
    try:
        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.1,
        )
    finally:
        # Stage snapshots can be rendered repeatedly during search.
        # Clear the figure even when writing the PNG fails so the renderer does
        # not retain the full graph until the notebook cell exits.
        fig.clear()
    return str(path)


def export_rigid_views(state, directory: str | Path, *, initial: bool = False) -> Dict[str, Any]:
    """Export exact aromatic subgraphs, one file pair per unit/component."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    aromatic = sorted(
        (unit.unit_id for unit in state.problem.units if unit.phase == "aromatic"),
        key=lambda value: int(value.split(":", 1)[1]),
    )
    groups = []
    if initial:
        groups = [(unit_id,) for unit_id in aromatic]
    else:
        remaining = set(aromatic)
        while remaining:
            unit_id = min(remaining, key=lambda value: int(value.split(":", 1)[1]))
            group = tuple(sorted(state.component_unit_ids(unit_id) & set(aromatic),
                                 key=lambda value: int(value.split(":", 1)[1])))
            groups.append(group)
            remaining.difference_update(group)
        groups.sort(key=lambda group: (-len(group), group))
    molecule = state.to_dict()
    owners = state.problem.ownership_map()
    planned = state.problem.node_map()
    rows = []
    for index, group in enumerate(groups, 1):
        name = f"aromatic_{index:02d}" if initial else f"rigid_cluster_{index:02d}"
        node_ids = {node_id for node_id in state.node_axial if owners.get(node_id) in group}
        nodes = [row for row in molecule["nodes"] if int(row["node_id"]) in node_ids]
        edges = [row for row in molecule["edges"] if int(row["source_node_id"]) in node_ids
                 and int(row["target_node_id"]) in node_ids]
        contracts = [contract for contract in state.problem.contracts if contract.kind == "rigid"
                     and contract.source_unit_id in group and contract.target_unit_id in group]
        payload = {
            "id": name,
            "unit_ids": list(group),
            "nodes": nodes,
            "edges": edges,
            "placements": {unit_id: molecule["placements"][unit_id] for unit_id in group},
            "su10_port_ids": sorted(node_id for node_id in node_ids if planned.get(node_id) == 10),
            "rigid_contract_ids": sorted(contract.contract_id for contract in contracts),
        }
        png = base / f"{name}.png"
        data = base / f"{name}.json"
        save_state_png(state, png, visible_node_ids=node_ids,
                       show_node_ids=False,
                       title=f"{name} | {len(group)} aromatic units")
        _write_json(payload, data)
        rows.append({"id": name, "unit_ids": list(group), "node_count": len(nodes),
                     "rigid_connection_count": len(contracts),
                     "png": str(png), "json": str(data)})
    manifest = base / "manifest.json"
    _write_json({"initial": initial, "count": len(rows), "items": rows}, manifest)
    return {"directory": str(base), "manifest": str(manifest), "count": len(rows)}


def composition_report(state, source_problem, pre_substitution=None, output_dir=None):
    """Reconcile input identities, intentional idle nodes and staging merges."""
    synthetic = {r.synthetic_node_id for r in source_problem.hetero_requests
                 if r.synthetic_node_id is not None}
    source = {gid: su for gid, su in source_problem.input_node_map().items()
              if gid not in synthetic}
    # The inverse identity already declares the final hetero SU, even though
    # the staging aromatic template displays one/two SU13 sites.
    source.update({r.target_node_id: r.target_su for r in source_problem.hetero_requests})
    final = {gid: su for gid, su in state.current_node_map().items() if gid in state.node_axial}
    idle = dict(source_problem.idle_aromatic_nodes)
    residual = source_problem.unassembled_resource_map()
    merged = {a['removed_node_id']: a['representative_node_id'] for a in state.action_history
              if a.get('kind') == 'substitution' and 'removed_node_id' in a}
    before_state = pre_substitution or state
    before = {gid: su for gid, su in before_state.current_node_map().items()
              if gid in before_state.node_axial}
    absent = set(source)-set(final)-set(idle)-set(residual)-set(merged)
    owner = state.problem.ownership_map()
    missing_required = {gid for gid in absent if owner.get(gid) in state.placements}
    unconnected = sorted(absent-missing_required)
    rows = []
    for gid in sorted(set(source) | set(final) | synthetic):
        disposition = ('assembled' if gid in final else 'idle_aromatic' if gid in idle
                       else 'unassembled_resource' if gid in residual
                       else 'substitution_merged' if gid in merged else 'missing_required' if gid in missing_required else 'unconnected')
        rows.append(dict(node_id=gid, source_su=source.get(gid), final_su=final.get(gid),
                         disposition=disposition, synthetic=gid in synthetic,
                         representative_node_id=merged.get(gid, gid if gid in final else None)))
    counts, placed, pre = Counter(source.values()), Counter(final.values()), Counter(before.values())
    idle_counts, residual_counts = Counter(idle.values()), Counter(residual.values())
    pending_ids = {r.target_node_id for r in source_problem.hetero_requests}
    pending_ids.update(n for c in state.mandatory_contracts() for n in (c.source_node_id,c.target_node_id))
    aromatic_ids = {n.node_id for u in state.problem.units if u.phase == 'aromatic' for n in u.nodes}
    pending_ids.update(n for n, su in state.problem.node_su_types
                       if n in aromatic_ids and (5 <= su <= 11 or su in {26, 30}))
    mismatches = sorted(gid for gid in set(final)&set(source) if final[gid]!=source[gid])
    pending_labels = {gid for gid in mismatches if final[gid]==13 and gid in pending_ids}
    pending_delta = Counter(final[gid] for gid in pending_labels)
    pending_delta.subtract(source[gid] for gid in pending_labels)
    pending_delta.update(final[gid] for gid in set(final)&synthetic)
    unconnected_counts = Counter(source[gid] for gid in unconnected)
    comparison = []
    for su in sorted(set(counts) | set(placed) | set(pre)):
        target = counts[su] - idle_counts[su] - residual_counts[su]
        comparison.append(dict(su_type=su, inverse_count=counts[su], assembly_target_count=target,
            pre_substitution_count=pre[su], final_count=placed[su],
            idle_aromatic_count=idle_counts[su], other_unassembled_count=residual_counts[su],
            substitution_delta=placed[su]-pre[su], unconnected_count=unconnected_counts[su],
            pending_label_delta=pending_delta[su],
            unexplained_delta=placed[su]-target+unconnected_counts[su]-pending_delta[su]))
    unexpected = sorted(set(final) - set(source) - synthetic)
    mismatches = sorted(gid for gid in set(final) & set(source) if final[gid] != source[gid])
    # Partial assembly is an accounted disposition, not lost input identity.
    reconciled = not (missing_required or unexpected or (set(mismatches)-pending_labels))
    summary = dict(assembly_complete=bool(state.is_complete()),
        identity_valid=not state.audit_identity_errors(), label_valid=not state.audit_label_errors(),
        input_valid=not state.problem.input_issues,
        input_errors=[json.loads(issue) for issue in state.problem.input_issues],
        all_input_nodes_used=set(source) <= set(final), composition_reconciled=reconciled,
        source_node_count=len(source), assembled_node_count=len(final),
        idle_aromatic_count=len(idle), other_unassembled_count=len(residual),
        missing_required_ids=sorted(missing_required), unconnected_node_ids=unconnected,
        pending_label_ids=sorted(pending_labels), unexpected_ids=unexpected, su_mismatch_ids=mismatches,
        synthetic_merged_ids=sorted(set(merged) & synthetic), su_comparison=comparison)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, records in [('final_su_comparison.csv', comparison),
                              ('final_node_disposition.csv', rows)]:
            with (directory / name).open('w', newline='', encoding='utf-8') as handle:
                if records:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        _write_json(summary, directory / 'final_composition_summary.json')
    return summary


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
            "logical_gid", "physical_slot_id", "source_unit_id", "original_su_type",
            "actual_degree", "label_state", "satisfied_contract_ids",
            "su25_center_parent_id",
        ])
        node_details = {row['node_id']: row for row in molecule['nodes']}
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
                node_id, node_details[node_id]['physical_slot_id'],
                node_details[node_id]['source_unit_id'], node_details[node_id]['original_su_type'],
                node_details[node_id]['actual_degree'], node_details[node_id]['label_state'],
                json.dumps(node_details[node_id]['satisfied_contract_ids']),
                node_details[node_id]['su25_center_parent_id'],
            ])
    with (directory / "final_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_node_id", "target_node_id", "kind", "contract_id", "direction",
                         "six_direction_neighbor", "honeycomb_edge_valid", "substitution_midpoint_edge",
                         "su25_center_branch_edge",
                         "actual_geometry_valid"])
        bad_pairs, bad_nodes = set(), set()
        for error in audit['actual_geometry_errors']:
            bad_nodes.update(error.get('nodes', ()))
            if 'node' in error:
                bad_nodes.add(error['node'])
            bad_pairs.update(frozenset(pair) for pair in error.get('edges', ()))
            if 'edge' in error:
                bad_pairs.add(frozenset(error['edge']))
        for edge in state.edges:
            pair = frozenset((edge.source_node_id, edge.target_node_id))
            direction = "" if edge.direction is None else f"{edge.direction[0]},{edge.direction[1]}"
            writer.writerow([
                int(edge.source_node_id), int(edge.target_node_id), str(edge.kind),
                str(edge.contract_id), direction,
                hex_edge_valid(state.node_axial[edge.source_node_id], state.node_axial[edge.target_node_id]),
                hex_edge_valid(state.node_axial[edge.source_node_id], state.node_axial[edge.target_node_id], honeycomb=True),
                pair in state.substitution_relaxed_edges,
                (state._su25_center_parents.get(edge.source_node_id) == edge.target_node_id
                 or state._su25_center_parents.get(edge.target_node_id) == edge.source_node_id),
                state._edge_geometry_valid(edge.source_node_id, edge.target_node_id,
                    state.node_axial[edge.source_node_id], state.node_axial[edge.target_node_id])
                    and pair not in bad_pairs and not pair.intersection(bad_nodes),
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

        already_reported = {r.node_id for r in state.problem.unassembled_closure_resources}
        merged = {a['removed_node_id'] for a in state.action_history
                  if a.get('kind') == 'substitution' and 'removed_node_id' in a}
        for node_id,su in state.problem.input_node_map().items():
            if node_id in state.node_axial or node_id in already_reported or node_id in merged:
                continue
            uid = ownership.get(node_id)
            reason = 'outside_main_component' if uid in state.excluded_unit_ids else 'no_legal_connection'
            if uid is not None:
                reason = dict(state.unit(uid).metadata).get('assembly_unavailable_reason') or reason
            writer.writerow([node_id,su,'unconnected','[]',reason])

    png_path = directory / "final_molecule.png"
    try:
        save_state_png(
            state,
            png_path,
            title=(
                f"Inverse-driven hexagonal coarse molecule | nodes={len(state.node_axial)} "
                f"edges={len(state.edges)} components={audit.get('component_count')}"
            ),
        )
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
    }


__all__ = ["export_final_state", "save_state_png", "export_rigid_views"]
