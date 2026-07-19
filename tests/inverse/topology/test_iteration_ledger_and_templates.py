from __future__ import annotations

import pytest
import torch

from model.inverse.topology import classify_topology_units, plan_stage3_resource_adjustments
from model.inverse.topology.iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_source_status,
    seed_ledger_topology_state,
)
from model.inverse.topology.stage3_models import TopologyClosureContract, TopologyClosureSlot
from model.inverse.topology.stage3_state_machine import (
    _residual_contracts,
    _su24_ring_contracts,
    build_su24_resource_role_profiles,
    plan_stage3_state_machine,
)
from model.inverse.topology.phase_eval import _canonical_stage3_contracts, build_stage3_phase_evaluation
from tests.inverse.topology.helpers import connect, make_node


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
        "reason": "supplemental_flex_candidate_not_selected",
        "blocker_kind": "supplemental_flex_candidate_not_selected",
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
        (14, 14, 2, 0),
        (14, 15, 1, 0),
        (14, 16, 1, 0),
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


def test_su24_templates_are_role_matched_and_ordered() -> None:
    nodes = [make_node(index, 24) for index in range(10, 14)]
    fixed_cap = make_node(99, 11)
    connect(nodes[2], fixed_cap, fixed=True)
    nodes.append(fixed_cap)
    pool = {24: [10, 11, 12, 13]}

    profiles = build_su24_resource_role_profiles(nodes, pool)
    contracts = _su24_ring_contracts(profiles)

    assert profiles
    assert all(profile.allowed_ring_slots for profile in profiles)
    priorities = [contract.priority for contract in contracts]
    assert priorities == sorted(priorities)
    assert all(contract.contract_kind == "ring_partner" for contract in contracts)
    assert all(contract.target_role == "fatty_ring" for contract in contracts)
    assert {contract.expected_topology for contract in contracts} >= {
        "resource_su24_side_ring",
        "resource_su24_upper_lower_ring",
        "resource_su24_fused_ring",
    }


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
                    "required_su_sequence": [11],
                }],
                "branch_slots": [],
                "required_counts": {"11": 1},
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

    assert plan.reason == "phase_resource_infeasible"
    assert any(item.contract_id == 2 and item.committed for item in plan.phase_result.transactions)
    assert plan.phase_result.infeasibility_proof["reason"] == "stage3_completion_invariants_failed"
