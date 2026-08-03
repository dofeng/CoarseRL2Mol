from __future__ import annotations

import pytest
import torch

from model.inverse.topology import classify_topology_units, plan_stage3_resource_adjustments
from model.inverse.topology.assembly_external import (
    ExternalTopologyAssembler,
    assemble_pool_su24_rings,
)
from model.inverse.topology.iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_source_status,
    normalize_slot,
    seed_ledger_topology_state,
    slot_sequence,
)
from model.inverse.topology.resource_profiles import (
    alternative_closure_required_counts,
    closure_sequences,
    minimum_closure_required_counts,
)
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    TopologyClosureContract,
    TopologyClosureSlot,
)
from model.inverse.topology.stage3_state_machine import (
    _bind_real_node_ids,
    _pool_su24_ring_conversion_contract,
    _transaction_plan,
    _residual_contracts,
    build_su24_resource_role_profiles,
    plan_stage3_state_machine,
)
from model.inverse.topology.phase_eval import _canonical_stage3_contracts, build_stage3_phase_evaluation
from model.inverse.topology.round_context import (
    build_stage3_round_context,
    mark_round_pending_contracts,
)
from tests.inverse.topology.helpers import connect, make_node


def test_closure_profiles_define_short_defaults_and_long_alternatives() -> None:
    assert closure_sequences("H") == [[23, 22], [23, 23, 22]]
    assert closure_sequences("F1") == [[23, 11], [23, 23, 11]]
    assert closure_sequences("F2") == [[23, 22], [23, 23, 22]]
    assert closure_sequences("FIXED22") == [[22]]
    assert slot_sequence("H") == [23, 22]
    assert slot_sequence("F1") == [23, 11]
    assert minimum_closure_required_counts("H") == {23: 1, 22: 1}
    assert alternative_closure_required_counts("H") == [{23: 2, 22: 1}]


def test_zero_ids_survive_slot_and_round_context_normalization() -> None:
    slot = normalize_slot({
        "slot_id": "fragment:436:main:0:open_boundary:F1",
        "kind": "F1",
        "anchor_node_id": 0,
        "anchor_role": "fixed_path_open_boundary",
        "required_su_sequence": [23, 11],
    })
    assert slot.anchor_node_id == 0

    node = make_node(0, 15)
    classification = {
        "fragments": [{
            "fragment_id": 0,
            "source_node_ids": [0],
            "claimed_node_ids": [0],
            "context_node_ids": [],
        }],
        "resource_pool": {},
    }
    context = build_stage3_round_context(
        nodes=[node],
        classification=classification,
    )
    assert context.ownership_by_node_id[0].owner_fragment_id == 0

    marked = mark_round_pending_contracts(context, [{
        "contract_id": 0,
        "source_fragment_ids": [0],
        "claimed_source_node_ids": [0],
    }])
    assert marked.ownership_by_node_id[0].owner_contract_id == 0


def test_canonical_contract_flow_uses_actual_assembly_slots_for_selected_flex_owner() -> None:
    assembly_contract = {
        "contract_id": 22,
        "contract_kind": "mainline_closure",
        "target_role": "flex_chain",
        "source_fragment_ids": [22],
        "claimed_source_node_ids": [22, 487],
        "mainline_slots": [],
        "branch_slots": [],
        "required_counts": {"11": 2},
        "expected_topology": "-24-3-",
        "reason": "main_pool_shortage",
        "diagnostics": {"public_pending_kind": "d_like_open"},
    }
    flex_overlay = {
        **assembly_contract,
        "required_counts": {"11": 2, "23": 1},
        "expected_topology": "-23-23-23-",
        "reason": "supplemental_flex_candidate",
        "diagnostics": {"selected_for_target": True, "candidate_kind": "d_like_open"},
    }

    contracts = _canonical_stage3_contracts(
        flex_contracts=[flex_overlay],
        assembly_contracts=[assembly_contract],
    )

    assert len(contracts) == 1
    assert contracts[0]["required_counts"] == {"11": 2}
    assert contracts[0]["expected_topology"] == "-24-3-"
    assert contracts[0]["diagnostics"]["selected_for_target"] is True


def test_canonical_contract_preserves_structural_ring_blocker() -> None:
    assembly_blocker = {
        "contract_id": 22,
        "contract_kind": "structural_blocker",
        "target_role": "fatty_ring",
        "source_fragment_ids": [22],
        "claimed_source_node_ids": [22, 487],
        "mainline_slots": [],
        "branch_slots": [],
        "required_counts": {},
        "expected_topology": "-14-15-",
        "reason": "fused_bridgehead_missing",
        "blocker_kind": "fused_bridgehead_missing",
        "diagnostics": {},
    }
    flex_overlay = {
        **assembly_blocker,
        "contract_kind": "structural_blocker",
        "target_role": "flex_chain",
        "reason": "lower_priority_flex_overlay",
        "blocker_kind": "lower_priority_flex_overlay",
        "diagnostics": {"selected_for_target": False},
    }

    contracts = _canonical_stage3_contracts(
        flex_contracts=[flex_overlay],
        assembly_contracts=[assembly_blocker],
    )

    assert len(contracts) == 1
    assert contracts[0]["target_role"] == "fatty_ring"
    assert contracts[0]["reason"] == "fused_bridgehead_missing"
    assert contracts[0]["blocker_kind"] == "fused_bridgehead_missing"


@pytest.mark.parametrize(
    ("left_su", "right_su", "branch_count", "mainline_count"),
    [
        (14, 14, 2, 2),
        (14, 15, 1, 2),
        (14, 16, 1, 2),
        (15, 15, 0, 2),
        (15, 16, 0, 2),
        (17, 17, 0, 2),
        (17, 18, 0, 2),
    ],
)
def test_every_valid_unsaturated_pair_projects_one_atomic_fragment(
    left_su: int,
    right_su: int,
    branch_count: int,
    mainline_count: int,
) -> None:
    left = make_node(10, left_su)
    right = make_node(11, right_su)
    connect(left, right)

    result = classify_topology_units([left, right])
    fragments = [
        fragment
        for fragment in result.fragments
        if set(fragment.claimed_node_ids or []) == {10, 11}
    ]

    assert len(fragments) == 1
    fragment = fragments[0]
    assert len(fragment.branch_contracts) == branch_count
    assert len(fragment.open_mainline_slots) == mainline_count
    assert fragment.atomic_group_ids == [10]


@pytest.mark.parametrize(("left_su", "right_su"), [(16, 16), (18, 18)])
def test_invalid_unsaturated_pairs_are_explicitly_rejected(left_su: int, right_su: int) -> None:
    left = make_node(20, left_su)
    right = make_node(21, right_su)
    connect(left, right)

    result = classify_topology_units([left, right])

    assert not result.atomic_groups
    assert any(
        str(item.get("reason", "")) == "invalid_unsaturated_atomic_pair"
        for item in result.classification_rejects
    )


def test_slot_transaction_claims_h_and_h1_resources_once() -> None:
    nodes = [
        make_node(1, 5),
        make_node(2, 23),
        make_node(3, 22),
        make_node(4, 23),
        make_node(5, 22),
    ]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={23: [2, 4], 22: [3, 5]},
        claimed_source_ids=[1],
    )
    contract = TopologyClosureContract(
        contract_id=7,
        phase="external_side_chain_closure",
        claimed_source_node_ids=[1],
        source_node_ids=[1],
        branch_slots=[
            TopologyClosureSlot(slot_id="H", kind="H", anchor_node_id=1),
            TopologyClosureSlot(slot_id="H1", kind="H1", anchor_node_id=1),
        ],
    )

    updated, slots, claimed, proof = assign_contract_slots(ledger, contract)

    assert proof == {}
    assert updated is not None
    assert claimed == {23: [2, 4], 22: [3, 5]}
    assigned_ids = [value for slot in slots for value in slot.assigned_resource_ids]
    assert len(assigned_ids) == len(set(assigned_ids)) == 4
    assert all(slot.state == "FILLED" for slot in slots)


def test_slot_transaction_rolls_back_when_second_slot_is_short() -> None:
    nodes = [make_node(1, 5), make_node(2, 23), make_node(3, 22)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={23: [2], 22: [3]},
        claimed_source_ids=[1],
    )
    contract = TopologyClosureContract(
        contract_id=8,
        claimed_source_node_ids=[1],
        branch_slots=[
            TopologyClosureSlot(slot_id="H", kind="H", anchor_node_id=1),
            TopologyClosureSlot(slot_id="H1", kind="H1", anchor_node_id=1),
        ],
    )

    updated, _, claimed, proof = assign_contract_slots(ledger, contract)

    assert updated is None
    assert claimed == {}
    assert proof["reason"] == "slot_resource_deficit"
    assert ledger.available_ids_by_su[23] == [2]
    assert ledger.available_ids_by_su[22] == [3]
    assert all(slot.state == "OPEN" for slot in contract.mainline_slots)
    assert all(slot.assigned_resource_ids == [] for slot in contract.mainline_slots)


def test_su24_resource_profiles_are_exact_and_remain_node_bound() -> None:
    nodes = []
    fixtures = (
        (make_node(10, 24), (make_node(100, 11), make_node(101, 23), make_node(102, 0))),
        (make_node(11, 24), (make_node(110, 11), make_node(111, 22), make_node(112, 23))),
        (make_node(12, 24), (make_node(120, 23), make_node(121, 0), make_node(122, 2))),
        (make_node(13, 24), (make_node(130, 1), make_node(131, 23), make_node(132, 3))),
    )
    for center, neighbors in fixtures:
        nodes.append(center)
        nodes.extend(neighbors)
        for neighbor in neighbors:
            connect(center, neighbor)
    pool = {24: [10, 11, 12, 13]}

    profiles = build_su24_resource_role_profiles(nodes, pool)

    assert [profile.node_id for profile in profiles] == [10, 11, 12, 13]
    assert [profile.role for profile in profiles] == ["A", "B", "C", "D"]
    assert all(profile.allowed_ring_slots for profile in profiles)
    assert all(profile.roles == [profile.role] for profile in profiles)
    assert all(profile.available_roles == [profile.role] for profile in profiles)


def test_pool_su24_conversion_contract_binds_selected_sources_and_only_claims_generic_gap() -> None:
    attempt = {
        "template_priority": 1,
        "ring_template": "side_ring_plan",
        "selected_source_node_ids": [10, 11, 20, 21],
        "selected_su24_ids": [10, 11],
        "required_counts": {11: 2, 22: 3, 23: 4, 24: 2},
        "generic_required_sequence": [22, 23, 23],
        "missing_counts": {22: 1},
    }

    contract = _pool_su24_ring_conversion_contract(attempt, contract_id=901)

    assert contract.source_node_ids == [10, 11, 20, 21]
    assert contract.claimed_source_node_ids == [10, 11, 20, 21]
    assert contract.required_counts == {11: 2, 22: 3, 23: 4, 24: 2}
    assert len(contract.mainline_slots) == 1
    assert contract.mainline_slots[0].required_su_sequence == [22, 23, 23]
    assert contract.diagnostics["conversion_only"] is True
    assert contract.diagnostics["pool_su24_ring_attempt"]["selected_su24_ids"] == [10, 11]


def test_conversion_binding_never_uses_selected_ring_source_as_donor() -> None:
    conversion = HistogramConversion(from_su=24, to_su=22, count=1)
    histogram = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(primary_op=conversion)]
    )

    pool_after = _bind_real_node_ids(
        histogram,
        {24: [100, 101, 102, 103], 22: []},
        protected_node_ids=[100, 101, 102],
    )

    assert conversion.node_ids == [103]
    assert pool_after[24] == [100, 101, 102]
    assert pool_after[22] == [103]


def test_pool_su24_shortage_converts_only_unselected_real_su24() -> None:
    nodes = []
    shared_open = [make_node(node_id, 23) for node_id in (200, 201, 202)]
    for center_id, cap_id in ((100, 10), (101, 11)):
        center = make_node(center_id, 24)
        cap = make_node(cap_id, 11)
        for neighbor in (cap, shared_open[0], shared_open[1]):
            connect(center, neighbor)
        nodes.extend([center, cap])
    for center_id in (102, 103, 104, 105, 106):
        center = make_node(center_id, 24)
        for neighbor in shared_open:
            connect(center, neighbor)
        nodes.append(center)
    nodes.extend(shared_open)
    nodes.extend(make_node(node_id, 23) for node_id in (203, 204, 205))
    nodes.extend(make_node(node_id, 22) for node_id in (300, 301, 302))

    classification = classify_topology_units(nodes)
    pool = {
        int(key): list(values)
        for key, values in classification.resource_pool["remaining_ids_by_su"].items()
    }
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembler.pool_only_su24_ring_mode = True
    blocked = next(
        attempt
        for attempt, _ in assembler._pool_su24_ring_plan_candidates()
        if len(list(attempt.get("selected_su24_ids", []) or [])) == 6
        and dict(attempt.get("missing_counts", {}) or {}) == {22: 1}
    )
    assert blocked["selected_su24_ids"] == [100, 101, 102, 103, 104, 105]
    assert blocked["missing_counts"] == {22: 1}

    phase_eval = build_stage3_phase_evaluation(nodes, classification=classification)
    ledger = build_iteration_ledger(nodes, resource_ids_by_su=pool)
    H = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1
    plan = _transaction_plan(
        phase_eval=phase_eval,
        contract=_pool_su24_ring_conversion_contract(blocked),
        pool=pool,
        H=H,
        E_target=None,
        max_primary_adjustments=10,
        max_h_compensation_ops=0,
        h_tolerance=0.06,
        special_degree_meta=None,
        iteration_ledger=ledger,
        protected_resource_counts={24: 6},
        accounting_initial_pool=pool,
        reserve_contract_after_conversion=False,
    )

    assert plan.ok is True
    assert plan.reason == "stage3_rebuild_required"
    conversions = [
        group.primary_op for group in plan.histogram_adjustment_plan.groups
    ]
    assert [
        (item.from_su, item.to_su, item.node_ids)
        for item in conversions
    ] == [(24, 22, [106])]
    assert plan.runtime["iteration_topology_ledger"].completed_products == []


@pytest.mark.parametrize(
    ("preferred", "minimum", "expected_first_length"),
    [(3, 2, 3), (4, 2, 4), (3, 3, 3)],
)
def test_residual_flex_preferred_minimum_and_priority(
    preferred: int,
    minimum: int,
    expected_first_length: int,
) -> None:
    pool = {11: list(range(1, 20)), 22: [], 23: list(range(100, 140)), 24: []}

    contracts = _residual_contracts(
        pool,
        residual_flex_preferred_23=preferred,
        residual_flex_min_23=minimum,
        max_flex_chains=1,
    )
    flex = next(
        contract
        for contract in contracts
        if dict(contract.diagnostics or {}).get("product_role") == "residual_flex_chain"
    )

    assert flex.contract_kind == "residual_conversion"
    assert flex.target_role == "residual_resource_conversion"
    assert int(flex.required_counts[23]) == expected_first_length
    assert int(flex.required_counts[23]) >= minimum
    assert len(flex.mainline_slots) == 1
    assert flex.mainline_slots[0].required_su_sequence == (
        [11] + [23] * expected_first_length + [11]
    )


def test_residual_flex_uses_only_the_deferred_target_gap(monkeypatch) -> None:
    phase_eval = build_stage3_phase_evaluation(
        [], classification=classify_topology_units([])
    )
    phase_eval.flex_bounds = {"min": 2, "max": 5}
    phase_eval.flex_counts = {"final": 2}
    phase_eval.assembly_view["flex_target_plan"] = {
        "remaining_minimum_gap": 0,
        "remaining_preferred_gap": 0,
    }
    captured = {}

    def capture_next_residual(*, remaining_flex_chains, **kwargs):
        captured["remaining_flex_chains"] = remaining_flex_chains
        return {"mode": "none", "attempts": []}

    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._plan_next_residual_product",
        capture_next_residual,
    )
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **_kwargs: phase_eval,
    )

    plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=torch.zeros(40, dtype=torch.long),
        nodes=[],
        enable_post_external_spectrum=False,
    )

    assert captured["remaining_flex_chains"] == 0


def test_fresh_residual_target_uses_only_current_flex_and_inventory(monkeypatch) -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 11),
        make_node(3, 23),
        make_node(4, 23),
    ]
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    phase_eval.canonical_contracts = []
    phase_eval.flex_bounds = {"min": 2, "max": 5}
    phase_eval.flex_counts = {"final": 1}
    phase_eval.assembly_view["flex_target_plan"] = {
        "remaining_minimum_gap": 1,
        "remaining_preferred_gap": 1,
    }
    phase_eval.flex_audit["enforced"] = True
    captured = {}

    def capture_next_residual(*, remaining_flex_chains, iteration_ledger, **kwargs):
        captured["remaining_flex_chains"] = remaining_flex_chains
        captured["available_ids_by_su"] = {
            int(key): list(values)
            for key, values in iteration_ledger.available_ids_by_su.items()
        }
        return {"mode": "none", "attempts": []}

    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._plan_next_residual_product",
        capture_next_residual,
    )
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._evaluate_external_phases",
        lambda **_kwargs: phase_eval,
    )
    plan_stage3_state_machine(
        phase_eval=phase_eval,
        H=torch.tensor([0] * 40, dtype=torch.long),
        nodes=nodes,
        enable_post_external_spectrum=False,
    )

    assert captured["remaining_flex_chains"] == 1
    available_ids = {
        node_id
        for ids in captured["available_ids_by_su"].values()
        for node_id in ids
    }
    assert available_ids == {1, 2, 3, 4}


def test_completed_product_claim_type_mismatch_is_a_ledger_conflict() -> None:
    node = make_node(1, 23)
    ledger = build_iteration_ledger(
        [node],
        resource_ids_by_su={23: [1]},
    )

    seeded = seed_ledger_topology_state(
        ledger,
        completed_products=[{
            "product_id": "residual:2000000",
            "phase": "residual_resource_self_closure",
            "product_role": "residual_side_chain",
            "claimed_resource_ids_by_su": {"11": [1]},
        }],
    )

    assert seeded.available_ids_by_su[23] == [1]
    assert seeded.conflicts == [{
        "node_id": 1,
        "reason": "completed_product_su_type_mismatch",
        "product_id": "residual:2000000",
        "claimed_su_type": 11,
        "actual_su_type": 23,
    }]


def test_direct_stage3_ledger_closure_does_not_mutate_hop1() -> None:
    center = make_node(100, 0)
    aromatic_9 = make_node(9, 9)
    aromatic_6 = make_node(6, 6)
    connect(center, aromatic_9, fixed=True)
    connect(center, aromatic_6, fixed=True)
    nodes = [center, aromatic_9, aromatic_6]
    before_hop1 = {node.global_id: set(node.hop1_ids) for node in nodes}
    H = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1

    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)

    assert plan.reason == "stage3_complete"
    assert plan.phase_result.primary_adjustment_count == 0
    assert plan.phase_result.symbolic_topology_count >= 1
    assert all(set(node.hop1_ids) == before_hop1[node.global_id] for node in nodes)
    assert all(transaction.commit_scope == "iteration_ledger" for transaction in plan.phase_result.transactions)


def test_completed_alternative_contract_is_superseded_not_infeasible() -> None:
    nodes = [make_node(1, 15), make_node(2, 15), make_node(3, 23), make_node(4, 11)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={23: [3], 11: [4]},
        claimed_source_ids=[1, 2],
    )
    completed = {
        "product_id": "flex:1",
        "source_fragment_ids": [1],
        "source_node_ids": [1, 2, 3, 4],
    }
    pending = {
        "contract_id": 99,
        "source_fragment_ids": [1],
        "claimed_source_node_ids": [1, 2],
    }

    seeded = seed_ledger_topology_state(
        ledger,
        completed_products=[completed],
        pending_contracts=[pending],
    )
    status, proof = contract_source_status(seeded, [1, 2])

    assert status == "superseded"
    assert proof["owner_product_id"] == "flex:1"
    assert seeded.pending_contracts == []
    assert seeded.conflicts == []


def test_partial_completed_source_is_identity_conflict() -> None:
    nodes = [make_node(1, 15), make_node(2, 15)]
    ledger = build_iteration_ledger(nodes, resource_ids_by_su={}, claimed_source_ids=[1, 2])
    seeded = seed_ledger_topology_state(
        ledger,
        completed_products=[{
            "product_id": "flex:1",
            "source_fragment_ids": [1],
            "source_node_ids": [1],
        }],
        pending_contracts=[],
    )

    status, proof = contract_source_status(seeded, [1, 2])

    assert status == "conflict"
    assert proof["reason"] == "contract_identity_conflict"
    assert proof["completed_node_ids"] == [1]
    assert proof["available_node_ids"] == [2]


def test_blocked_contract_allows_later_direct_contract_but_prevents_false_completion(monkeypatch) -> None:
    nodes = [make_node(1, 5), make_node(2, 5), make_node(3, 11)]
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes))
    H = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1

    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._external_contracts",
        lambda phase_eval, nodes: [
            {
                "contract_id": 1,
                "contract_kind": "structural_blocker",
                "target_role": "flex_chain",
                "source_fragment_ids": [1],
                "claimed_source_node_ids": [1],
                "context_node_ids": [],
                "mainline_slots": [],
                "branch_slots": [],
                "required_counts": {},
                "expected_topology": "",
                "reason": "template_infeasible",
                "blocker_kind": "template_infeasible",
                "diagnostics": {},
            },
            {
                "contract_id": 2,
                "contract_kind": "mainline_closure",
                "target_role": "side_chain",
                "source_fragment_ids": [2],
                "claimed_source_node_ids": [2],
                "context_node_ids": [],
                "mainline_slots": [{
                    "slot_id": "side:2:F1",
                    "kind": "FIXED22",
                    "anchor_node_id": 2,
                    "required_su_sequence": [22],
                }],
                "branch_slots": [],
                "required_counts": {"22": 1},
                "expected_topology": "",
                "reason": "mainline_resource_closure",
                "blocker_kind": "",
                "diagnostics": {},
            },
        ],
    )
    monkeypatch.setattr(
        "model.inverse.topology.stage3_state_machine._residual_contracts",
        lambda pool, **kwargs: [],
    )

    plan = plan_stage3_state_machine(phase_eval=phase_eval, H=H, nodes=nodes)

    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "external_topology_assembly"
    assert plan.phase_result.phase_completed is False
    assert plan.diagnostics["external_rebuild_reason"] == "external_blocker_after_conversion"
    assert any(item.contract_id == 2 and item.committed for item in plan.phase_result.transactions)
    assert plan.phase_result.infeasibility_proof == {}
