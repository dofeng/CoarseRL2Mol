from __future__ import annotations

import torch

from model.inverse.topology import (
    audit_stage3_topology,
    apply_stage3_node_adjustments,
    assemble_topology_units,
    build_stage3_phase_evaluation,
    classify_topology_units,
    plan_stage3_resource_adjustments,
)
from model.inverse.topology.stage3_models import Stage3AdjustmentPlan
from model.inverse.topology.stage3_state_machine import _residual_contracts
from tests.inverse.topology.helpers import connect, make_node


def _histogram(nodes, size=40):
    out = torch.zeros(size, dtype=torch.long)
    for node in nodes:
        out[int(node.su_type)] += 1
    return out


def test_fresh_read_only_audit_completes_and_preserves_resource_ids() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    H = _histogram(nodes)
    H_before = H.detach().clone()
    node_types_before = [int(node.su_type) for node in nodes]

    audit = audit_stage3_topology(
        H=H,
        nodes=nodes,
        allow_conversions=False,
        enable_spectrum_rebalance=False,
    )

    assert audit["ok"] is True
    assert audit["reason"] == "stage3_complete"
    assert audit["requires_conversion"] is False
    assert audit["pending_contracts"] == []
    assert audit["products"][0]["topology_notation"] == "11-23-23-22"
    assert audit["resource_accounting"]["ok"] is True
    accounting = audit["resource_accounting"]
    aggregate_ids = {
        int(node_id)
        for values in accounting["aggregate_ids_after_by_su"].values()
        for node_id in values
    }
    partition_ids = {
        int(node_id)
        for field in ("final_remaining_ids_by_su", "committed_claimed_ids_by_su")
        for values in accounting[field].values()
        for node_id in values
    }
    assert aggregate_ids == partition_ids == {1, 2, 3, 4}
    assert torch.equal(H, H_before)
    assert [int(node.su_type) for node in nodes] == node_types_before


def test_read_only_audit_treats_terminalized_aromatics_as_final_support() -> None:
    nodes = [make_node(12, 12), make_node(13, 13)]

    audit = audit_stage3_topology(
        H=_histogram(nodes),
        nodes=nodes,
        allow_conversions=False,
        enable_spectrum_rebalance=False,
        protected_terminal_aromatic_node_ids=[12, 13],
    )

    assert audit["ok"] is True
    assert audit["reason"] == "stage3_complete"
    pool = audit["classification"]["resource_pool"]
    assert pool["su11_donor_ids_by_su"] == {"10": [], "12": [], "13": []}
    assert audit["resource_accounting"]["ok"] is True
    assert not any(
        audit["resource_accounting"]["aggregate_ids_before_by_su"].values()
    )
    assert audit["pending_contracts"] == []


def test_fresh_read_only_audit_requires_residual_su24_aromatization() -> None:
    nodes = [make_node(1, 24)]

    audit = audit_stage3_topology(
        H=_histogram(nodes),
        nodes=nodes,
        allow_conversions=False,
        enable_spectrum_rebalance=False,
    )

    assert audit["ok"] is False
    assert audit["reason"] == "phase_resource_infeasible"
    assert audit["requires_conversion"] is False
    assert audit["products"] == []
    assert audit["resource_accounting"]["final_remaining_ids_by_su"][24] == [1]
    assert audit["plan"]["phase"] == "residual_su24_aromatization"


def test_failed_fresh_audit_preserves_real_pending_and_resource_snapshot() -> None:
    center = make_node(10, 3)
    aromatic = make_node(11, 9)
    open_23 = make_node(12, 23)
    aromatic_donor = make_node(13, 12)
    connect(center, aromatic)
    connect(center, open_23)
    nodes = [center, aromatic, open_23, aromatic_donor]

    audit = audit_stage3_topology(
        H=_histogram(nodes),
        nodes=nodes,
        allow_conversions=False,
        enable_spectrum_rebalance=False,
    )

    assert audit["ok"] is False
    assert audit["reason"] == "fresh_audit_requires_conversion"
    assert audit["requires_conversion"] is True
    assert audit["classification"]["fragments"]
    assert audit["external_assembly"]["pending_units"]
    assert audit["resource_ledger"]["initial_ids_by_su"] == {"12": [13]}
    assert audit["resource_ledger"]["remaining_ids_by_su"] == {"12": [13]}
    assert audit["ownership_ledger"]["states"]["pending_topology"] == [10, 11, 12]
    assert audit["ownership_ledger"]["states"]["remaining_aromatic_donor"] == [13]
    assert audit["ownership_ledger"]["invariant_errors"] == []
    pending = audit["pending_contracts"][0]
    assert pending["real"] == "9-3-23-"
    assert pending["real_gid"] == "11-10-12-"
    assert pending["source_node_ids"] == [10, 11, 12]
    assert pending["required_counts"] == {"22": 1}
    assert pending["infeasibility_proof"]["missing_counts"] == {"22": 1}


def test_residual_planner_prefers_two_23_side_chain() -> None:
    contracts = _residual_contracts(
        {11: [200], 22: [203], 23: [204, 205], 24: []},
        max_flex_chains=0,
    )

    side = next(item for item in contracts if item.diagnostics.get("product_role") == "residual_side_chain")
    assert side.expected_topology == "11-23-23-22"
    assert side.required_counts == {11: 1, 22: 1, 23: 2}


def test_su25_cd_closes_with_real_f2_resources_when_available() -> None:
    center = make_node(250, 25, target_hop1_degree=4)
    fixed22 = make_node(251, 22)
    pool23 = make_node(252, 23)
    pool22 = make_node(253, 22)
    pool22_b = make_node(254, 22)
    pool23_b = make_node(255, 23)
    pool22_c = make_node(256, 22)
    pool11 = make_node(257, 11)
    pool23_c = make_node(258, 23)
    connect(center, fixed22)
    connect(center, pool23)
    connect(center, pool23_b)
    connect(center, pool23_c)

    nodes = [
        center, fixed22, pool23, pool22, pool22_b, pool23_b,
        pool22_c, pool11, pool23_c,
    ]
    classification = classify_topology_units(nodes)
    fragment = next(
        item for item in classification.fragments
        if int(item.fragment_id) == 250
    )
    assert [
        (slot.kind, slot.anchor_role, slot.required_su_sequence)
        for slot in fragment.open_mainline_slots
    ] == [
        ("F1", "left_mainline", [23, 11]),
        ("F2", "right_mainline", [23, 22]),
    ]
    assert [
        (slot.kind, slot.required_su_sequence)
        for slot in fragment.branch_contracts
        if not slot.is_complete()
    ] == [("H", [23, 22])]

    assembly = assemble_topology_units(
        nodes,
        classification,
    ).to_dict()

    assert not [item for item in assembly.get("pending_units", []) if int(item.get("unit_id", -1)) == 250]
    product = next(item for item in assembly.get("products", []) if 250 in list(item.get("source_fragment_ids", []) or []))
    assert product["actual_topology_su"] == "11-23-25(-23-22)(-22)-23-22"
    assert {250, 251, 252, 253, 254, 255, 257, 258}.issubset(
        set(product.get("symbolic_node_ids", []) or [])
    )


def test_su25_cd_pending_contract_counts_every_open_port() -> None:
    center = make_node(270, 25, target_hop1_degree=4)
    fixed22 = make_node(271, 22)
    pool23_a = make_node(272, 23)
    pool23_b = make_node(273, 23)
    pool23_c = make_node(274, 23)
    pool22_a = make_node(275, 22)
    pool22_b = make_node(276, 22)
    connect(center, fixed22)
    connect(center, pool23_a)
    connect(center, pool23_b)
    connect(center, pool23_c)

    nodes = [
        center, fixed22, pool23_a, pool23_b, pool23_c, pool22_a, pool22_b,
    ]
    assembly = assemble_topology_units(
        nodes,
        classify_topology_units(nodes),
    ).to_dict()

    pending = next(
        item for item in assembly.get("pending_units", [])
        if int(item.get("unit_id", -1)) == 270
    )
    assert pending["reason"] == "main_pool_shortage"
    assert pending["required_counts"] == {"11": 1, "22": 2, "23": 3}
    assert [slot["kind"] for slot in pending["open_mainline_slots"]] == [
        "F1",
        "F2",
    ]
    assert [slot["kind"] for slot in pending["branch_slots"]] == [
        "H",
        "CLOSED_TERMINAL",
    ]


def test_su25_requires_pool_fixed22_when_only_22_like_terminal_exists() -> None:
    center = make_node(260, 25)
    terminal1 = make_node(261, 1)
    open23 = make_node(262, 23)
    connect(center, terminal1)
    connect(center, open23)

    classification = classify_topology_units([center, terminal1, open23])
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 260)

    assert fragment.class_type == "25_CD"
    assert fragment.branch_slots == ["FIXED22", "H"]
    evidence = dict(fragment.evidence or {})
    assert evidence["fixed22_neighbor_ids"] == []
    assert evidence["fixed22_requires_pool"] is True
    assert [slot.kind for slot in fragment.branch_contracts[:2]] == ["FIXED22", "H"]
    assert fragment.branch_contracts[0].required_su_sequence == [22]


def test_atomic_fallback_fragment_is_projected_and_cannot_disappear_at_completion() -> None:
    left = make_node(1, 15)
    right = make_node(2, 15)
    connect(left, right)
    nodes = [left, right]

    classification = classify_topology_units(nodes)
    assert [(item.fragment_id, item.family) for item in classification.fragments] == [
        (1, "basic_unsaturated_pair")
    ]
    assert classification.fragments[0].class_type == "D"
    assert classification.fragments[0].evidence["flex_range_candidate_kind"] == "d_like_open"

    assembly = assemble_topology_units(nodes, classification).to_dict()
    covered_fragment_ids = {
        int(value)
        for row in list(assembly.get("products", []) or []) + list(assembly.get("pending_units", []) or [])
        for value in list(dict(row or {}).get("source_fragment_ids", []) or [])
    }
    assert covered_fragment_ids == {1}

    plan = plan_stage3_resource_adjustments(H=_histogram(nodes), nodes=nodes)
    assert plan.reason != "stage3_complete"


def test_external_flex_shortfall_is_not_materialized_without_real_fragments() -> None:
    nodes = [make_node(100 + index, 9) for index in range(10)]
    nodes.extend(make_node(200 + index, 10) for index in range(4))
    nodes.extend(make_node(300 + index, 11) for index in range(2))
    nodes.extend(make_node(400 + index, 23) for index in range(2))

    plan = plan_stage3_resource_adjustments(
        H=_histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
    )

    assert plan.reason == "stage3_complete"
    flex_transactions = [
        transaction
        for transaction in list(plan.phase_result.transactions or [])
        if int(transaction.contract_id) == -200000
    ]
    assert flex_transactions == []


def test_isolated_su24_is_aromatized_before_residual_tolerance() -> None:
    nodes = [make_node(1, 24)]
    H = _histogram(nodes)
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert 1 in list(phase_eval.closure_pool_ids_by_su.get(24, []) or [])

    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    assert str(plan.phase) == "residual_su24_aromatization"
    assert plan.reason == "stage3_rebuild_required"
    apply_result = apply_stage3_node_adjustments(nodes, H, plan)

    assert apply_result.changed is True
    assert int(apply_result.nodes[0].su_type) == 12
    assert int(apply_result.H[24].item()) == 0
    assert int(apply_result.H[12].item()) == 1
    assert int(apply_result.H[23].item()) == 0
    assert int(dict(apply_result.conservation_after or {}).get("su_count", 0)) == 1


def test_isolated_su11_is_left_untouched_within_residual_tolerance() -> None:
    nodes = [make_node(1, 11)]
    H = _histogram(nodes)

    first_plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    first = apply_stage3_node_adjustments(nodes, H, first_plan)

    assert first_plan.reason == "stage3_complete"
    assert first.changed is False
    assert int(first.nodes[0].su_type) == 11
    assert int(first.H[11].item()) == 1


def test_three_mixed_unassemblable_resources_are_left_untouched() -> None:
    nodes = [make_node(1, 24), make_node(2, 22), make_node(3, 22)]
    H = _histogram(nodes)
    aromatic_plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    applied = apply_stage3_node_adjustments(nodes, H, aromatic_plan)
    plan = plan_stage3_resource_adjustments(H=applied.H, nodes=applied.nodes)

    assert plan.reason == "stage3_complete"
    assert applied.changed is True
    assert [int(node.su_type) for node in applied.nodes] == [12, 22, 22]
    assert plan.diagnostics["tolerated_remaining_count"] == 2
    assert plan.diagnostics["tolerated_remaining_ids_by_su"] == {
        "22": [2, 3],
    }


def test_four_unassemblable_resources_still_require_conversion() -> None:
    nodes = [make_node(index, 22) for index in range(1, 5)]

    plan = plan_stage3_resource_adjustments(H=_histogram(nodes), nodes=nodes)

    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "residual_resource_self_closure"
    assert plan.histogram_adjustment_plan.primary_adjustment_count > 0


def test_direct_residual_side_products_include_11_22_without_rebuild() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 11),
        make_node(3, 22),
        make_node(4, 22),
        make_node(5, 23),
    ]
    H = _histogram(nodes)
    first_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert bool(first_eval.assembly_attempted) is False

    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)
    applied = apply_stage3_node_adjustments(nodes, H, plan)
    assert plan.reason == "stage3_complete"
    assert applied.changed is False
    residual_fits = list(plan.diagnostics.get("direct_residual_template_fits", []) or [])
    assert [item["chain_23_count"] for item in residual_fits] == [1]
    assert plan.diagnostics["tolerated_remaining_count"] == 2
    assert all(int(item["conversion_count"]) == 0 for item in residual_fits)


def test_rebuilt_state_is_reassembled_without_prior_residual_products() -> None:
    nodes = [
        make_node(12, 12),
        make_node(22, 22),
        make_node(23, 23),
        make_node(24, 23),
        make_node(25, 23),
    ]
    H = _histogram(nodes)

    first_plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
    )
    first_applied = apply_stage3_node_adjustments(nodes, H, first_plan)
    second_plan = plan_stage3_resource_adjustments(
        H=first_applied.H,
        nodes=first_applied.nodes,
        enable_post_external_spectrum=False,
    )

    assert first_plan.reason == "stage3_rebuild_required"
    assert second_plan.reason == "stage3_complete"
    assert second_plan.phase_result.infeasibility_proof == {}
    ledger = second_plan.runtime["iteration_topology_ledger"]
    assert ledger.conflicts == []
    assert ledger.conflict_node_ids == []
    assert all(
        int(node_id) in {int(node.global_id) for node in first_applied.nodes}
        for product in ledger.completed_products
        for values in dict(product.get("claimed_resource_ids_by_su", {}) or {}).values()
        for node_id in values
    )


def test_su24_resource_is_visible_in_round_closure_pool() -> None:
    nodes = [make_node(1, 24), make_node(2, 24)]
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    assert sorted(phase_eval.closure_pool_ids_by_su[24]) == [1, 2]


def test_connected_su24_remains_real_resource_with_other_resources() -> None:
    center = make_node(1, 24)
    aromatic = make_node(2, 11)
    support = make_node(3, 23)
    connect(center, aromatic)
    connect(center, support)

    classification = classify_topology_units([center, aromatic, support])
    pool = dict(classification.resource_pool or {})

    assert 1 in list(dict(pool.get("remaining_ids_by_su", {}) or {}).get("24", []) or [])
    assert 2 in list(dict(pool.get("remaining_ids_by_su", {}) or {}).get("11", []) or [])
    assert not any(
        int(item.get("node_id", -1)) == 1
        for item in list(classification.classification_rejects or [])
    )


def test_histogram_conversion_does_not_update_old_neighbor_histograms() -> None:
    center = make_node(1, 24)
    neighbor = make_node(2, 23)
    connect(center, neighbor)
    H = _histogram([center, neighbor])
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        planned_conversions=[{"op": "test_conversion", "from_su": 23, "to_su": 11, "node_id": 2}],
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
