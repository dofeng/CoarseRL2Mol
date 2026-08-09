from __future__ import annotations

import ast
import csv
from collections import Counter
from .csv_fixture_loader import YM4_FINAL_NODES_CSV, load_nodes_from_final_csv


def _parse_list(text: str) -> list[int]:
    value = ast.literal_eval(str(text or "[]"))
    return [int(x) for x in list(value or [])]


def test_csv_loader_restores_all_real_nodes() -> None:
    nodes = load_nodes_from_final_csv(YM4_FINAL_NODES_CSV)
    assert len(nodes) == 618
    assert nodes[0].global_id == 0
    assert nodes[-1].global_id == 617


def test_csv_loader_restores_hop1_degree_and_multisets() -> None:
    nodes = {int(node.global_id): node for node in load_nodes_from_final_csv(YM4_FINAL_NODES_CSV)}
    with YM4_FINAL_NODES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = int(row["global_id"])
            node = nodes[node_id]
            hop1_ids = _parse_list(row.get("hop1_ids", ""))
            hop1_ms = _parse_list(row.get("hop1_ms", ""))
            assert list(node.hop1_ids or []) == hop1_ids
            assert Counter(dict(node.hop1_su or {})) == Counter(hop1_ms)
            if str(row.get("target_hop1_degree", "")).strip():
                assert int(node.target_hop1_degree) == int(float(row["target_hop1_degree"]))
            if str(row.get("target_fixed_anchor_count", "")).strip():
                assert int(node.target_fixed_anchor_count) == int(float(row["target_fixed_anchor_count"]))
