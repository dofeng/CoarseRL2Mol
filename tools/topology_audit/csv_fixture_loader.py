from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch

from model.shared.inverse_common import _NodeV3


TOPOLOGY_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
YM4_FINAL_NODES_CSV = TOPOLOGY_FIXTURE_DIR / "ym4_final_nodes_by_global_id.csv"


def _parse_int_list(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    if isinstance(parsed, (list, tuple)):
        return [int(x) for x in list(parsed)]
    return []


def _counter_from_su_list(values: Iterable[int]) -> Counter:
    counter: Counter = Counter()
    for value in list(values or []):
        counter[int(value)] += 1
    return counter


def _parse_optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return int(float(text))


def load_nodes_from_final_csv(path: str | Path, *, fixture_infer_fixed_connectors: bool = False) -> List[_NodeV3]:
    csv_path = Path(path)
    import csv

    nodes: List[_NodeV3] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            global_id = int(row["global_id"])
            su_type = int(row["su_type"])
            hop1_ms = _parse_int_list(row.get("hop1_ms", ""))
            hop2_ms = _parse_int_list(row.get("hop2_ms", ""))
            target_hop1_degree = row.get("target_hop1_degree", "")
            target_fixed_anchor_count = row.get("target_fixed_anchor_count", "")
            node = _NodeV3(
                global_id=global_id,
                su_type=su_type,
                hop1_su=_counter_from_su_list(hop1_ms),
                hop2_su=_counter_from_su_list(hop2_ms),
                z_vec=torch.zeros(16),
                mu=float(row.get("mu", 0.0) or 0.0),
                pi=float(row.get("pi", 1.0) or 1.0),
                target_hop1_degree=_parse_optional_int(target_hop1_degree),
                special_anchor_partition=str(row.get("special_anchor_partition", "") or "").strip() or None,
                target_fixed_anchor_count=_parse_optional_int(target_fixed_anchor_count),
                special_anchor_mode=str(row.get("special_anchor_mode", "") or "").strip() or None,
            )
            node.hop1_ids = _parse_int_list(row.get("hop1_ids", ""))
            node.fixed_hop1_ids = set()
            template_key = str(row.get("template_key", "") or "").strip()
            node.template_key = template_key if template_key else None
            nodes.append(node)
    nodes.sort(key=lambda item: int(item.global_id))
    if fixture_infer_fixed_connectors:
        nodes_by_id = {int(node.global_id): node for node in nodes}
        for node in nodes:
            want = int(getattr(node, "target_fixed_anchor_count", 0) or 0)
            if want <= 0:
                continue
            candidates: List[int] = []
            for neighbor_id in list(getattr(node, "hop1_ids", []) or []):
                neighbor = nodes_by_id.get(int(neighbor_id))
                if neighbor is None:
                    continue
                if int(getattr(neighbor, "su_type", -1)) in {0, 2, 27, 28, 29, 31}:
                    candidates.append(int(neighbor_id))
            if len(candidates) >= want:
                node.fixed_hop1_ids = set(sorted(candidates)[:want])
                score_components = dict(getattr(node, "score_components", {}) or {})
                score_components["fixture_inferred_fixed_connectors"] = True
                node.score_components = score_components
        for node in nodes:
            for neighbor_id in list(getattr(node, "fixed_hop1_ids", set()) or set()):
                neighbor = nodes_by_id.get(int(neighbor_id))
                if neighbor is not None:
                    neighbor.fixed_hop1_ids.add(int(node.global_id))
    return nodes
