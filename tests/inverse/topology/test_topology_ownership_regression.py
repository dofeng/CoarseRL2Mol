from __future__ import annotations

from pathlib import Path

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.topology.phase_eval import assemble_topology_units
from model.inverse.topology.classification.classifier import classify_topology_units
from model.shared.inverse_common import _NodeV3

from .helpers import connect, make_node


FIXTURE_DIR = Path(__file__).resolve().parents[3] / "test_results" / "YM-8" / "final_outputs"


def test_unresolved_open_fragment_is_exported_as_pending() -> None:
    center = make_node(1, 1)
    neighbor = make_node(2, 23)
    connect(center, neighbor)

    classification = classify_topology_units([center, neighbor])
    assembly = assemble_topology_units([center, neighbor], classification).to_dict()

    open_fragment_ids = {
        int(fragment.get("fragment_id"))
        for fragment in classification.to_dict().get("fragments", [])
        if fragment.get("open_slots") or fragment.get("branch_slots")
    }
    pending_fragment_ids = {
        int(fragment_id)
        for pending in assembly.get("pending_units", [])
        for fragment_id in pending.get("source_fragment_ids", [])
    }
    assert open_fragment_ids
    assert open_fragment_ids <= pending_fragment_ids


def test_pending_and_resource_pool_are_disjoint() -> None:
    center = make_node(1, 1)
    resource = make_node(2, 23)
    connect(center, resource)

    classification = classify_topology_units([center, resource])
    assembly = assemble_topology_units([center, resource], classification).to_dict()
    states = dict(
        dict(assembly.get("diagnostics", {}) or {}).get("ownership_ledger", {}).get("states", {})
        or {}
    )
    pending_ids = {int(node_id) for node_id in list(states.get("pending_topology", []) or [])}
    resource_ids = {
        int(node_id)
        for state in ("remaining_closure_resource", "remaining_aromatic_donor")
        for node_id in list(states.get(state, []) or [])
    }
    assert pending_ids.isdisjoint(resource_ids)


def test_ym8_final_products_do_not_claim_duplicate_nodes() -> None:
    node_csv = FIXTURE_DIR / "final_nodes_by_global_id.csv"
    if not node_csv.exists():
        return
    import ast
    import csv
    from collections import Counter

    nodes = []
    with node_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node = _NodeV3(int(row["global_id"]), int(row["su_type"]))
            node.hop1_ids = [int(value) for value in ast.literal_eval(row["hop1_ids"])]
            node.hop1_su = Counter(int(value) for value in ast.literal_eval(row["hop1_ms"]))
            node.hop2_su = Counter(int(value) for value in ast.literal_eval(row["hop2_ms"]))
            nodes.append(node)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()
    product_ids = [
        int(node_id)
        for product in assembly.get("products", [])
        for node_id in product.get("source_node_ids", [])
    ]
    assert len(product_ids) == len(set(product_ids))
