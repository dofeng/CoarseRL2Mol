from types import SimpleNamespace

import pytest

from model.inverse.topology.sequential_resource import (
    build_ownership_ledger,
    claim_requirement,
    contract_required_counts,
    select_next_contract,
)
from model.inverse.topology.stage3_models import merge_stage3_contracts_by_owner


def _node(node_id, su_type):
    return SimpleNamespace(global_id=node_id, su_type=su_type)


def test_ownership_excludes_completed_and_pending_nodes_from_real_pool():
    nodes = [_node(1, 11), _node(2, 23), _node(3, 22), _node(4, 24)]
    ledger = build_ownership_ledger(
        nodes,
        completed_products=[{"unit_id": 10, "source_node_ids": [1]}],
        pending_topologies=[{"unit_id": 20, "source_node_ids": [2]}],
    )

    assert ledger.ids_by_su()[11] == []
    assert ledger.ids_by_su()[23] == []
    assert ledger.ids_by_su()[22] == [3]
    assert ledger.ids_by_su()[24] == [4]
    assert ledger.conflicts == []


def test_pending_topology_owns_all_explicit_core_nodes_not_only_resource_nodes():
    nodes = [_node(1, 5), _node(2, 29), _node(3, 19), _node(4, 11), _node(5, 23)]
    ledger = build_ownership_ledger(
        nodes,
        completed_products=[],
        pending_topologies=[{
            "unit_id": 3,
            "core_node_ids": [1, 2, 3],
            "source_node_ids": [3],
            "claimed_resource_ids_by_su": {"11": [4]},
        }],
    )

    assert {ledger.by_node_id[node_id].state for node_id in (1, 2, 3, 4)} == {"pending_topology"}
    assert ledger.ids_by_su()[11] == []
    assert ledger.ids_by_su()[23] == [5]
    assert ledger.conflicts == []


def test_sequential_contract_priority_is_flex_ring_side():
    contracts = [
        {"contract_id": 3, "contract_kind": "mainline_closure", "target_role": "side_chain", "source_fragment_ids": [3], "claimed_source_node_ids": [3]},
        {"contract_id": 2, "contract_kind": "ring_partner", "target_role": "fatty_ring", "source_fragment_ids": [2], "claimed_source_node_ids": [2]},
        {"contract_id": 1, "contract_kind": "fragment_pairing", "target_role": "flex_chain", "source_fragment_ids": [1], "claimed_source_node_ids": [1]},
    ]

    assert select_next_contract(contracts)["contract_id"] == 1


def test_single_contract_requirement_does_not_accumulate_later_contracts():
    current = {
        "contract_id": 1,
        "contract_kind": "mainline_closure",
        "target_role": "flex_chain",
        "source_fragment_ids": [1],
        "claimed_source_node_ids": [1],
        "required_counts": {11: 1, 23: 1},
    }
    later = {
        "contract_id": 2,
        "contract_kind": "mainline_closure",
        "target_role": "side_chain",
        "source_fragment_ids": [2],
        "claimed_source_node_ids": [2],
        "required_counts": {11: 1, 22: 1, 23: 2},
    }

    assert select_next_contract([later, current])["contract_id"] == 1
    assert contract_required_counts(current) == {11: 1, 23: 1}


def test_fatty_ring_requirement_claims_real_ids_atomically():
    nodes = [
        _node(1, 11), _node(2, 22), _node(3, 22),
        _node(4, 23), _node(5, 23), _node(6, 23), _node(7, 24),
    ]
    contract = {
        "contract_id": 30,
        "contract_kind": "ring_partner",
        "target_role": "fatty_ring",
        "required_counts": {11: 1, 22: 2, 23: 3, 24: 1},
    }
    ledger = build_ownership_ledger(nodes, completed_products=[], pending_topologies=[])

    result = claim_requirement(ledger, contract, contract_required_counts(contract))

    assert result["ok"] is True
    assert result["claimed_ids_by_su"] == {11: [1], 22: [2, 3], 23: [4, 5, 6], 24: [7]}
    assert all(item.state == "pending_topology" for item in ledger.by_node_id.values())


def test_contract_without_explicit_role_or_kind_is_rejected():
    with pytest.raises(ValueError, match="non-canonical topology contract"):
        contract_required_counts({"contract_id": 9, "required_counts": {11: 1}})


def test_conflicting_contracts_for_same_owner_are_rejected():
    base = {
        "contract_id": 10,
        "contract_kind": "mainline_closure",
        "target_role": "side_chain",
        "source_fragment_ids": [23],
        "claimed_source_node_ids": [23, 434],
        "required_counts": {11: 1},
    }
    with pytest.raises(ValueError, match="duplicate topology contract definitions"):
        merge_stage3_contracts_by_owner([base, {**base, "target_role": "flex_chain"}])


def test_failed_single_topology_claim_rolls_back_all_claims():
    nodes = [_node(1, 11), _node(2, 23)]
    ledger = build_ownership_ledger(nodes, completed_products=[], pending_topologies=[])

    result = claim_requirement(ledger, {"contract_id": 40}, {11: 1, 23: 2})

    assert result["ok"] is False
    assert ledger.ids_by_su()[11] == [1]
    assert ledger.ids_by_su()[23] == [2]


def test_claimed_nodes_are_removed_only_for_current_topology():
    nodes = [_node(1, 11), _node(2, 11), _node(3, 23), _node(4, 23)]
    ledger = build_ownership_ledger(nodes, completed_products=[], pending_topologies=[])
    first = claim_requirement(ledger, {"contract_id": 1}, {11: 1, 23: 1})

    assert first["claimed_ids_by_su"] == {11: [1], 23: [3]}
    assert ledger.ids_by_su()[11] == [2]
    assert ledger.ids_by_su()[23] == [4]


def test_materialized_stage3_nodes_are_excluded_from_remaining_pool():
    nodes = [_node(1, 11), _node(2, 23), _node(3, 22)]
    nodes[0].stage3_topology_owner_id = 77
    nodes[1].stage3_topology_owner_id = 77

    ledger = build_ownership_ledger(nodes, completed_products=[], pending_topologies=[])

    assert ledger.ids_by_su()[11] == []
    assert ledger.ids_by_su()[23] == []
    assert ledger.ids_by_su()[22] == [3]
