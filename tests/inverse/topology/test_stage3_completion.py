from __future__ import annotations

import torch

from model.inverse.topology import (
    apply_stage3_node_adjustments,
    assemble_topology_units,
    build_stage3_phase_evaluation,
    classify_topology_units,
    plan_stage3_resource_adjustments,
)
from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
from tests.inverse.topology.helpers import connect, make_node


def _histogram(nodes, size=40):
    out = torch.zeros(size, dtype=torch.long)
    for node in nodes:
        out[int(node.su_type)] += 1
    return out


def test_residual_self_closure_prefers_two_23_side_chain() -> None:
    nodes = [
        make_node(200, 11), make_node(201, 11), make_node(202, 11),
        make_node(203, 22),
        make_node(204, 23), make_node(205, 23), make_node(206, 23), make_node(207, 23),
    ]

    assembly = assemble_topology_units(nodes, classify_topology_units(nodes)).to_dict()

    residual = list(assembly.get("residual_closure_products", []) or [])
    assert residual
    assert str(residual[0].get("product_kind", "")) == "residual_side_chain"
    assert str(residual[0].get("actual_topology_su", "")) == "11-23-23-22"
    assert residual[0]["materialized_node_ids"] == residual[0]["source_node_ids"]
    assert residual[0]["claimed_resource_ids"]
    ledger = dict(assembly.get("resource_ledger", {}) or {})
    for su_type, initial in dict(ledger.get("initial_counts", {}) or {}).items():
        assert int(initial) == (
            int(dict(ledger.get("topology_consumed_counts", {}) or {}).get(str(su_type), 0))
            + int(dict(ledger.get("residual_consumed_counts", {}) or {}).get(str(su_type), 0))
            + int(dict(ledger.get("remaining_counts", {}) or {}).get(str(su_type), 0))
        )


def test_su25_cd_closes_with_real_f2_resources_when_available() -> None:
    center = make_node(250, 25)
    fixed22 = make_node(251, 22)
    pool23 = make_node(252, 23)
    pool22 = make_node(253, 22)
    pool22_b = make_node(254, 22)
    pool23_b = make_node(255, 23)
    pool22_c = make_node(256, 22)
    connect(center, fixed22)

    assembly = assemble_topology_units(
        [center, fixed22, pool23, pool22, pool22_b, pool23_b, pool22_c],
        classify_topology_units([center, fixed22, pool23, pool22, pool22_b, pool23_b, pool22_c]),
    ).to_dict()

    assert not [item for item in assembly.get("pending_units", []) if int(item.get("unit_id", -1)) == 250]
    product = next(item for item in assembly.get("products", []) if 250 in list(item.get("source_fragment_ids", []) or []))
    assert 250 in list(product.get("materialized_node_ids", []) or [])
    assert 252 in list(product.get("materialized_node_ids", []) or [])
    assert 253 in list(product.get("materialized_node_ids", []) or [])
    assert 254 in list(product.get("materialized_node_ids", []) or [])
    assert 251 in list(product.get("materialized_node_ids", []) or [])


def test_isolated_su24_finishes_with_residual_infeasibility_proof() -> None:
    nodes = [make_node(1, 24)]
    H = _histogram(nodes)
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert str(phase_eval.phase) == "residual_resource_self_closure"

    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    assert str(plan.phase) == "complete"
    proof = dict(plan.diagnostics.get("residual_infeasibility_proof", {}) or {})
    assert proof["reason"] == "no_executable_residual_template_fit"
    assert proof["remaining_counts"]["24"] == 1
    apply_result = apply_stage3_node_adjustments(nodes, H, plan)

    assert apply_result.changed is False
    assert int(apply_result.nodes[0].su_type) == 24
    assert int(apply_result.H[24].item()) == 1
    assert int(apply_result.H[23].item()) == 0
    assert int(dict(apply_result.conservation_after or {}).get("su_count", 0)) == 1


def test_isolated_su11_is_not_converted_without_executable_template() -> None:
    nodes = [make_node(1, 11)]
    H = _histogram(nodes)

    first_plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    first = apply_stage3_node_adjustments(nodes, H, first_plan)

    assert first.changed is False
    assert int(first.nodes[0].su_type) == 11
    assert int(first.H[11].item()) == 1
    assert int(first.H[13].item()) == 0
    assert first_plan.diagnostics["residual_infeasibility_proof"]["remaining_counts"]["11"] == 1


def test_isolated_residual_does_not_request_layer12_rebuild() -> None:
    nodes = [make_node(1, 24)]
    H = _histogram(nodes)
    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    applied = apply_stage3_node_adjustments(nodes, H, plan)

    assert plan.reason == "stage3_complete"
    assert applied.changed is False
    assert int(applied.nodes[0].su_type) == 24
    assert int(applied.H[24].item()) == 1
    assert int(applied.H[23].item()) == 0


def test_direct_residual_side_product_needs_no_histogram_rebuild() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 11),
        make_node(3, 22),
        make_node(4, 22),
        make_node(5, 23),
    ]
    H = _histogram(nodes)
    first_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert str(first_eval.phase) == "residual_resource_self_closure"

    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    applied = apply_stage3_node_adjustments(nodes, H, plan)
    assert plan.reason == "stage3_complete"
    assert applied.changed is False
    assert any(
        str(transaction.target_role) == "residual_side_chain"
        and bool(transaction.committed)
        for transaction in plan.phase_result.transactions
    )


def test_su24_resource_advances_to_residual_phase() -> None:
    nodes = [make_node(1, 24), make_node(2, 24)]
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert str(phase_eval.phase) == "residual_resource_self_closure"


def test_su24_and_aromatic_support_remain_resources_without_real_fragment() -> None:
    center = make_node(1, 24)
    aromatic = make_node(2, 11)
    support = make_node(3, 23)
    connect(center, aromatic)
    connect(center, support)

    classification = classify_topology_units([center, aromatic, support])
    pool = dict(classification.resource_pool or {})

    assert 1 in list(dict(pool.get("remaining_ids_by_su", {}) or {}).get("24", []) or [])
    assert 2 in list(dict(pool.get("remaining_ids_by_su", {}) or {}).get("11", []) or [])


def test_histogram_conversion_does_not_update_old_neighbor_histograms() -> None:
    center = make_node(1, 24)
    neighbor = make_node(2, 23)
    connect(center, neighbor)
    H = _histogram([center, neighbor])
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        planned_conversions=[{"op": "test_conversion", "from_su": 23, "to_su": 11}],
        runtime={"assembly_before_adjustment": {"resource_ledger": {"remaining_counts": {"23": 1}}}},
    )

    applied = apply_stage3_node_adjustments([center, neighbor], H, plan)

    rebuilt_center = next(node for node in applied.nodes if int(node.global_id) == 1)
    assert int(rebuilt_center.hop1_su.get(23, 0) or 0) == 1
    assert int(rebuilt_center.hop1_su.get(11, 0) or 0) == 0
    assert int(applied.H[23].item()) == 0
    assert int(applied.H[11].item()) == 1


def test_su24_resources_do_not_reserve_future_flex_resources() -> None:
    raw_left = make_node(100, 24)
    raw_left_11 = make_node(101, 11)
    raw_left_23 = make_node(102, 23)
    connect(raw_left, raw_left_11)
    connect(raw_left, raw_left_23)
    raw_right = make_node(200, 24)
    raw_right_11 = make_node(201, 11)
    raw_right_23 = make_node(202, 23)
    connect(raw_right, raw_right_11)
    connect(raw_right, raw_right_23)
    nodes = [raw_left, raw_left_11, raw_left_23, raw_right, raw_right_11, raw_right_23]

    result = assemble_topology_units(nodes, classify_topology_units(nodes)).to_dict()
    protected = dict(result.get("protected_resource_counts", {}) or {})

    assert protected == {}
    assert "flex_target_plan" in result
