"""Extract and visualize k-hop subgraphs from coarse-grained molecule graphs.

This utility reads the coarse-grained `.pt` files produced by
model.shared.coarse_graph, extracts one local k-hop subgraph around each
eligible center node, exports subgraph composition tables, and draws topology
figures for paper-ready inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

try:
    from ..shared.coarse_graph import (
        NUM_SU_TYPES,
        SU_DEFS,
        SU_NAME_TO_ID,
        _assign_atoms_to_su,
        _compute_rdkit_su_positions,
        _layout_on_hex_grid,
        _node_color,
        _rotate_positions as _rotate_cg_positions,
    )
except ImportError:
    from model.shared.coarse_graph import (
        NUM_SU_TYPES,
        SU_DEFS,
        SU_NAME_TO_ID,
        _assign_atoms_to_su,
        _compute_rdkit_su_positions,
        _layout_on_hex_grid,
        _node_color,
        _rotate_positions as _rotate_cg_positions,
    )


EdgeRecord = Tuple[int, int, float, float]


def _load_graph(path: Path) -> Data:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        return Data.from_dict(obj)
    if hasattr(obj, "to_dict"):
        return Data.from_dict(obj.to_dict())
    raise TypeError(f"Unsupported graph object in {path}: {type(obj)!r}")


def _su_types(data: Data) -> torch.Tensor:
    if hasattr(data, "su_type_indices"):
        return data.su_type_indices.long().cpu()
    if not hasattr(data, "x") or data.x.numel() == 0:
        raise ValueError("Graph has no x features and no su_type_indices.")
    return data.x[:, :NUM_SU_TYPES].argmax(dim=-1).long().cpu()


def _is_carbon_mask(data: Data, su_types: torch.Tensor) -> torch.Tensor:
    if hasattr(data, "is_carbon"):
        return data.is_carbon.bool().cpu()
    return su_types < 26


def _edge_records(data: Data) -> List[EdgeRecord]:
    if not hasattr(data, "edge_index") or data.edge_index.numel() == 0:
        return []

    edge_index = data.edge_index.long().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.float().cpu()

    edge_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for ei in range(edge_index.size(1)):
        u = int(edge_index[0, ei])
        v = int(edge_index[1, ei])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if edge_attr is not None and edge_attr.numel() > 0 and ei < edge_attr.size(0):
            bond_order = float(edge_attr[ei, 0].item())
            is_ring = float(edge_attr[ei, 1].item()) if edge_attr.size(1) > 1 else 0.0
        else:
            bond_order = 1.0
            is_ring = 0.0
        edge_map.setdefault((a, b), (bond_order, is_ring))

    return [(u, v, attr[0], attr[1]) for (u, v), attr in sorted(edge_map.items())]


def _adjacency(num_nodes: int, records: Sequence[EdgeRecord]) -> List[List[int]]:
    adj = [[] for _ in range(num_nodes)]
    for u, v, _, _ in records:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    return [sorted(set(nbs)) for nbs in adj]


def _bfs_distances(center: int, adj: Sequence[Sequence[int]], k: int) -> Dict[int, int]:
    dist = {int(center): 0}
    q: deque[int] = deque([int(center)])
    while q:
        u = q.popleft()
        if dist[u] >= k:
            continue
        for v in adj[u]:
            if v in dist:
                continue
            dist[int(v)] = dist[u] + 1
            q.append(int(v))
    return dist


def _format_multiset(values: Iterable[int]) -> str:
    vals = sorted(int(v) for v in values)
    return "[" + ",".join(str(v) for v in vals) + "]"


def _histogram(values: Iterable[int], size: int = NUM_SU_TYPES) -> List[int]:
    hist = [0] * size
    for v in values:
        iv = int(v)
        if 0 <= iv < size:
            hist[iv] += 1
    return hist


def _rotate_pos(pos: Dict[int, Tuple[float, float]], degrees: float) -> Dict[int, Tuple[float, float]]:
    if not pos or abs(float(degrees)) < 1e-9:
        return dict(pos)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    theta = math.radians(float(degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    out = {}
    for node, (x, y) in pos.items():
        dx = x - cx
        dy = y - cy
        out[node] = (
            cx + dx * cos_t - dy * sin_t,
            cy + dx * sin_t + dy * cos_t,
        )
    return out


def _shell_positions(layers: Dict[int, List[int]], rotation_degrees: float) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    if layers.get(0):
        pos[layers[0][0]] = (0.0, 0.0)

    for hop in sorted(h for h in layers if h > 0):
        nodes = list(layers.get(hop, []))
        if not nodes:
            continue
        radius = float(hop)
        n = len(nodes)
        # A deterministic offset avoids placing every shell node on the same ray.
        offset = math.pi / 2.0 + (hop - 1) * math.pi / max(6, n)
        for idx, node in enumerate(nodes):
            theta = offset + 2.0 * math.pi * idx / max(1, n)
            pos[node] = (radius * math.cos(theta), radius * math.sin(theta))
    return _rotate_pos(pos, rotation_degrees)


def _coarse_graph_positions(
    data: Data,
    records: Sequence[EdgeRecord],
    su_types: torch.Tensor,
    rotation_degrees: float,
) -> Tuple[Optional[Dict[int, Tuple[float, float]]], str]:
    """Recompute the exact full-graph hex-grid coordinates used by coarse_graph.py."""
    smiles = getattr(data, "smiles", None)
    if isinstance(smiles, (list, tuple)):
        smiles = "".join(str(x) for x in smiles)
    if smiles is None or not str(smiles).strip():
        return None, "missing_smiles"

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        return None, f"rdkit_unavailable:{exc}"

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, "rdkit_parse_failed"
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        su_matches, _ = _assign_atoms_to_su(mol)
        order = list(range(len(su_matches)))
        order.sort(key=lambda i: (min(su_matches[i]["atoms"]), SU_NAME_TO_ID[su_matches[i]["name"]]))
        su_matches = [su_matches[i] for i in order]

        if len(su_matches) != int(data.num_nodes):
            return None, f"node_count_mismatch:pt={int(data.num_nodes)},recomputed={len(su_matches)}"

        recomputed_su = torch.tensor(
            [SU_NAME_TO_ID[su["name"]] for su in su_matches],
            dtype=torch.long,
        )
        if recomputed_su.numel() == su_types.numel() and not torch.equal(recomputed_su.cpu(), su_types.cpu()):
            return None, "su_sequence_mismatch"

        edge_list = [[int(u), int(v)] for u, v, _, _ in records]
        rdkit_pos = _compute_rdkit_su_positions(mol, su_matches)
        full_pos = _layout_on_hex_grid(
            edge_list,
            rdkit_pos,
            len(su_matches),
            mol=mol,
            su_matches=su_matches,
        )
        full_pos = _rotate_cg_positions(full_pos, rotation_degrees)
        return {int(k): (float(v[0]), float(v[1])) for k, v in full_pos.items()}, "coarse_graph_hex"
    except Exception as exc:
        return None, f"layout_failed:{exc}"


def _draw_subgraph(
    output_path: Path,
    center: int,
    nodes: Sequence[int],
    distances: Dict[int, int],
    records: Sequence[EdgeRecord],
    su_types: torch.Tensor,
    k: int,
    rotation_degrees: float = -90.0,
    show_su_id: bool = True,
    full_pos: Optional[Dict[int, Tuple[float, float]]] = None,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    node_set = set(int(n) for n in nodes)
    G = nx.Graph()
    for n in nodes:
        G.add_node(int(n), hop=int(distances[int(n)]), su=int(su_types[int(n)].item()))
    for u, v, bond_order, is_ring in records:
        if u in node_set and v in node_set:
            G.add_edge(int(u), int(v), bond_order=float(bond_order), is_ring=float(is_ring))

    if full_pos is not None and all(int(n) in full_pos for n in nodes):
        pos = {int(n): full_pos[int(n)] for n in nodes}
    else:
        layers: Dict[int, List[int]] = {}
        for n in nodes:
            layers.setdefault(int(distances[int(n)]), []).append(int(n))
        for hop in layers:
            layers[hop].sort(key=lambda n: (int(su_types[n].item()), n))
        pos = _shell_positions(layers, rotation_degrees)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for u, v, edge_data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ring = float(edge_data.get("is_ring", 0.0)) > 0.5
        order = float(edge_data.get("bond_order", 1.0))
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#666666",
            linewidth=2.0 if order >= 1.5 else 1.4,
            linestyle="-" if not ring else (0, (4, 3)),
            zorder=1,
            solid_capstyle="round",
        )

    for n in nodes:
        x, y = pos[int(n)]
        su = int(su_types[int(n)].item())
        radius = 0.20 if int(n) != int(center) else 0.27
        circle = plt.Circle(
            (x, y),
            radius,
            facecolor=_node_color(su),
            edgecolor="black" if int(n) == int(center) else "white",
            linewidth=1.2 if int(n) == int(center) else 0.7,
            zorder=3,
        )
        ax.add_patch(circle)
        if show_su_id:
            ax.text(
                x,
                y,
                str(su),
                ha="center",
                va="center",
                fontsize=15 if int(n) != int(center) else 15,
                color="white",
                fontweight="bold",
                zorder=4,
            )

    title_su = int(su_types[int(center)].item())
    ax.set_title(f"center {center} | SU {title_su} | {k}-hop", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.6
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


def extract_coarse_subgraphs(
    input_path: Path,
    output_dir: Path,
    k: int = 3,
    centers: Optional[Sequence[int]] = None,
    carbon_only: bool = False,
    draw: bool = True,
    max_plots: Optional[int] = None,
    save_subgraph_pt: bool = False,
    rotation_degrees: float = -90.0,
    show_su_id: bool = True,
) -> Dict[str, Any]:
    data = _load_graph(input_path)
    su_types = _su_types(data)
    carbon_mask = _is_carbon_mask(data, su_types)
    records = _edge_records(data)
    adj = _adjacency(int(data.num_nodes), records)
    full_pos, layout_mode = _coarse_graph_positions(data, records, su_types, rotation_degrees)

    if centers is None or len(centers) == 0:
        center_ids = list(range(int(data.num_nodes)))
    else:
        center_ids = sorted(set(int(c) for c in centers))
    if carbon_only:
        center_ids = [c for c in center_ids if 0 <= c < int(data.num_nodes) and bool(carbon_mask[c])]

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    pt_dir = output_dir / "subgraph_pt"

    summary_rows: List[Dict[str, Any]] = []
    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    template_buckets: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Dict[str, Any]] = {}

    plots_done = 0
    for center in center_ids:
        if center < 0 or center >= int(data.num_nodes):
            continue
        distances = _bfs_distances(center, adj, int(k))
        nodes = sorted(distances, key=lambda n: (distances[n], n))
        node_set = set(nodes)
        sub_edges = [rec for rec in records if rec[0] in node_set and rec[1] in node_set]

        hop_ms: Dict[int, Tuple[int, ...]] = {}
        for hop in range(1, int(k) + 1):
            hop_ms[hop] = tuple(
                sorted(int(su_types[n].item()) for n in nodes if int(distances[n]) == hop)
            )
        center_su = int(su_types[center].item())
        template_key = (center_su, hop_ms.get(1, tuple()), hop_ms.get(2, tuple()))
        su_hist = _histogram(int(su_types[n].item()) for n in nodes)

        summary_rows.append({
            "source": str(input_path),
            "center_node": int(center),
            "center_su": center_su,
            "center_su_name": SU_DEFS[center_su][0] if 0 <= center_su < len(SU_DEFS) else "Unknown",
            "k": int(k),
            "num_nodes": len(nodes),
            "num_edges": len(sub_edges),
            "hop1_multiset": _format_multiset(hop_ms.get(1, tuple())),
            "hop2_multiset": _format_multiset(hop_ms.get(2, tuple())),
            "hop3_multiset": _format_multiset(hop_ms.get(3, tuple())),
            "template_key": json.dumps({
                "center_su": center_su,
                "hop1_multiset": list(hop_ms.get(1, tuple())),
                "hop2_multiset": list(hop_ms.get(2, tuple())),
            }),
            "su_hist": json.dumps(su_hist),
        })

        for local_id, global_id in enumerate(nodes):
            su = int(su_types[global_id].item())
            node_rows.append({
                "center_node": int(center),
                "local_node": int(local_id),
                "global_node": int(global_id),
                "hop": int(distances[global_id]),
                "su": su,
                "su_name": SU_DEFS[su][0] if 0 <= su < len(SU_DEFS) else "Unknown",
            })

        global_to_local = {g: i for i, g in enumerate(nodes)}
        for u, v, bond_order, is_ring in sub_edges:
            edge_rows.append({
                "center_node": int(center),
                "src_local": int(global_to_local[u]),
                "dst_local": int(global_to_local[v]),
                "src_global": int(u),
                "dst_global": int(v),
                "src_su": int(su_types[u].item()),
                "dst_su": int(su_types[v].item()),
                "bond_order": float(bond_order),
                "is_ring": int(float(is_ring) > 0.5),
            })

        bucket = template_buckets.setdefault(template_key, {
            "center_su": center_su,
            "hop1_multiset": list(hop_ms.get(1, tuple())),
            "hop2_multiset": list(hop_ms.get(2, tuple())),
            "sample_count": 0,
            "centers": [],
            "hop3_multiset_counts": Counter(),
        })
        bucket["sample_count"] += 1
        bucket["centers"].append(int(center))
        bucket["hop3_multiset_counts"][_format_multiset(hop_ms.get(3, tuple()))] += 1

        if save_subgraph_pt:
            pt_dir.mkdir(parents=True, exist_ok=True)
            local_x = data.x[nodes].clone()
            local_edge_index = torch.tensor(
                [[global_to_local[u], global_to_local[v]] for u, v, _, _ in sub_edges],
                dtype=torch.long,
            ).t().contiguous() if sub_edges else torch.empty((2, 0), dtype=torch.long)
            local_edge_attr = torch.tensor(
                [[bo, ir] for _, _, bo, ir in sub_edges],
                dtype=torch.float,
            ) if sub_edges else torch.empty((0, 2), dtype=torch.float)
            sub_data = Data(x=local_x, edge_index=local_edge_index, edge_attr=local_edge_attr)
            sub_data.center_id = torch.tensor([global_to_local[int(center)]], dtype=torch.long)
            sub_data.original_center_id = torch.tensor([int(center)], dtype=torch.long)
            sub_data.su_type_indices = su_types[nodes].clone()
            sub_data.hop_distance = torch.tensor([distances[n] for n in nodes], dtype=torch.long)
            torch.save(sub_data.to_dict(), pt_dir / f"center_{center:03d}.pt")

        if draw and (max_plots is None or plots_done < int(max_plots)):
            _draw_subgraph(
                plot_dir / f"center_{center:03d}_su{center_su}_k{k}.png",
                center=center,
                nodes=nodes,
                distances=distances,
                records=sub_edges,
                su_types=su_types,
                k=int(k),
                rotation_degrees=rotation_degrees,
                show_su_id=show_su_id,
                full_pos=full_pos,
            )
            plots_done += 1

    _write_csv(output_dir / "subgraph_summary.csv", summary_rows)
    _write_csv(output_dir / "subgraph_nodes.csv", node_rows)
    _write_csv(output_dir / "subgraph_edges.csv", edge_rows)

    template_rows = []
    template_json = []
    for idx, (key, bucket) in enumerate(sorted(template_buckets.items(), key=lambda kv: kv[0])):
        hop3_counts = dict(sorted(bucket["hop3_multiset_counts"].items()))
        row = {
            "template_id": idx,
            "center_su": bucket["center_su"],
            "center_su_name": SU_DEFS[bucket["center_su"]][0] if 0 <= bucket["center_su"] < len(SU_DEFS) else "Unknown",
            "hop1_multiset": _format_multiset(bucket["hop1_multiset"]),
            "hop2_multiset": _format_multiset(bucket["hop2_multiset"]),
            "sample_count": bucket["sample_count"],
            "centers": _format_multiset(bucket["centers"]),
            "hop3_multiset_counts": json.dumps(hop3_counts, sort_keys=True),
        }
        template_rows.append(row)
        template_json.append({
            "template_id": idx,
            "template_key": {
                "center_su": bucket["center_su"],
                "hop1_multiset": bucket["hop1_multiset"],
                "hop2_multiset": bucket["hop2_multiset"],
            },
            "sample_count": bucket["sample_count"],
            "centers": bucket["centers"],
            "hop3_multiset_counts": hop3_counts,
        })
    _write_csv(output_dir / "subgraph_template_library.csv", template_rows)
    with (output_dir / "subgraph_template_library.json").open("w", encoding="utf-8") as f:
        json.dump(template_json, f, indent=2)

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "k": int(k),
        "num_centers": len(summary_rows),
        "num_templates": len(template_rows),
        "drawn_plots": plots_done,
        "carbon_only": bool(carbon_only),
        "layout_mode": layout_mode,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_centers(text: Optional[str]) -> Optional[List[int]]:
    if text is None or not str(text).strip():
        return None
    vals = []
    for part in str(text).replace(",", " ").split():
        vals.append(int(part))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and draw k-hop topology subgraphs from a coarse-grained molecule .pt file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input coarse-grained graph .pt file.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for CSV/JSON/PNG outputs.")
    parser.add_argument("--k", type=int, default=3, help="Subgraph radius in hops.")
    parser.add_argument("--centers", type=str, default=None, help="Optional center node ids, e.g. '0,1,9'.")
    parser.add_argument("--carbon_only", action="store_true", help="Only extract subgraphs centered on carbon-containing SUs.")
    parser.add_argument("--no_draw", action="store_true", help="Do not draw topology PNG files.")
    parser.add_argument("--max_plots", type=int, default=None, help="Maximum number of PNG subgraphs to draw.")
    parser.add_argument("--save_subgraph_pt", action="store_true", help="Also save each extracted subgraph as a .pt file.")
    parser.add_argument(
        "--rotation",
        type=float,
        default=-90.0,
        help="Topology plot rotation in degrees; -90 matches the default coarse_graph visualization.",
    )
    parser.add_argument("--hide_su_id", action="store_true", help="Hide SU ids inside topology nodes.")
    args = parser.parse_args()

    metadata = extract_coarse_subgraphs(
        input_path=args.input,
        output_dir=args.output_dir,
        k=args.k,
        centers=_parse_centers(args.centers),
        carbon_only=bool(args.carbon_only),
        draw=not bool(args.no_draw),
        max_plots=args.max_plots,
        save_subgraph_pt=bool(args.save_subgraph_pt),
        rotation_degrees=float(args.rotation),
        show_su_id=not bool(args.hide_su_id),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
