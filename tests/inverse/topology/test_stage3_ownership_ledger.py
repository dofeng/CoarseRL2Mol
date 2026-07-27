from types import SimpleNamespace

from model.inverse.topology.iteration_ledger import (
    assign_contract_slots,
    build_iteration_ledger,
    contract_priority,
    seed_ledger_topology_state,
)
from model.inverse.topology.round_context import (
    build_stage3_round_context,
    claim_round_product,
    round_context_invariants,
)
from model.inverse.topology.stage3_models import (
    Stage3OwnershipState,
    Stage3TopologyProduct,
    TopologyClosureContract,
)


def _node(node_id: int, su_type: int) -> SimpleNamespace:
    return SimpleNamespace(global_id=node_id, su_type=su_type)


def test_completed_and_pending_ids_leave_available_pool() -> None:
    nodes = [_node(1, 11), _node(2, 23), _node(3, 22), _node(4, 24)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1], 23: [2], 22: [3], 24: [4]},
    )
    ledger = seed_ledger_topology_state(
        ledger,
        completed_products=[{
            "product_id": "p1",
            "fixed_source_node_ids": [],
            "claimed_resource_ids_by_su": {"11": [1]},
        }],
        pending_contracts=[{
            "contract_id": 20,
            "source_fragment_ids": [20],
            "claimed_source_node_ids": [2],
        }],
    )

    assert ledger.available_ids_by_su[11] == []
    assert ledger.available_ids_by_su[23] == []
    assert ledger.available_ids_by_su[22] == [3]
    assert ledger.available_ids_by_su[24] == [4]
    assert ledger.conflicts == []


def test_contract_priority_is_flex_ring_side() -> None:
    contracts = [
        {"contract_kind": "mainline", "target_role": "side_chain", "source_fragment_ids": [3]},
        {"contract_kind": "ring", "target_role": "fatty_ring", "source_fragment_ids": [2]},
        {"contract_kind": "pair", "target_role": "flex_chain", "source_fragment_ids": [1]},
    ]
    ordered = sorted(contracts, key=contract_priority)
    assert [item["target_role"] for item in ordered] == [
        "flex_chain",
        "fatty_ring",
        "side_chain",
    ]


def test_single_contract_claim_is_atomic() -> None:
    nodes = [_node(1, 11), _node(2, 23), _node(3, 23)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1], 23: [2, 3]},
    )
    contract = TopologyClosureContract(
        contract_id=30,
        contract_kind="resource_path",
        phase="residual_resource_self_closure",
        target_role="residual_resource_conversion",
        required_counts={11: 1, 23: 2},
        expected_topology="11-23-23",
    )

    updated, _, claimed, proof = assign_contract_slots(ledger, contract)

    assert proof == {}
    assert claimed == {11: [1], 23: [2, 3]}
    assert updated is not None
    assert updated.available_ids_by_su[11] == []
    assert updated.available_ids_by_su[23] == []
    assert all(
        owner.state == Stage3OwnershipState.CLAIMED_CLOSURE_RESOURCE.value
        for owner in updated.ownership_by_node_id.values()
    )


def test_failed_contract_claim_leaves_input_ledger_unchanged() -> None:
    nodes = [_node(1, 11), _node(2, 23)]
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1], 23: [2]},
    )
    contract = TopologyClosureContract(
        contract_id=40,
        contract_kind="resource_path",
        phase="residual_resource_self_closure",
        target_role="residual_resource_conversion",
        required_counts={11: 1, 23: 2},
    )

    updated, _, claimed, proof = assign_contract_slots(ledger, contract)

    assert updated is None
    assert claimed == {}
    assert proof["reason"] == "slot_resource_deficit"
    assert ledger.available_ids_by_su[11] == [1]
    assert ledger.available_ids_by_su[23] == [2]


def test_persistent_node_owner_attributes_are_ignored() -> None:
    nodes = [_node(1, 11), _node(2, 23)]
    nodes[0].stage3_topology_owner_id = 77
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={11: [1], 23: [2]},
    )

    assert ledger.available_ids_by_su[11] == [1]
    assert ledger.available_ids_by_su[23] == [2]


def test_atomic_group_members_share_fixed_fragment_ownership() -> None:
    nodes = [_node(289, 15), _node(290, 15)]
    classification = SimpleNamespace(
        to_dict=lambda: {
            "fragments": [{
                "fragment_id": 289,
                "source_node_ids": [289, 290],
                "claimed_node_ids": [289, 290],
                "context_node_ids": [],
                "atomic_group_ids": [289],
                "evidence": {
                    "center_node_ids": [289],
                    "matched_neighbors": [{
                        "node_id": 290,
                        "is_fixed": False,
                    }],
                    "atomic_group_node_ids": [289, 290],
                    "atomic_groups": [{
                        "group_id": 289,
                        "kind": "double_bond_pair",
                        "node_ids": [289, 290],
                    }],
                },
            }],
            "resource_pool": {},
            "unclassified_nodes": [],
            "classification_rejects": [],
            "claim_conflicts": [],
            "support_units_only": [],
        },
    )

    context = build_stage3_round_context(
        nodes=nodes,
        classification=classification,
    )

    assert context.ownership_by_node_id[289].state == Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value
    assert context.ownership_by_node_id[290].state == Stage3OwnershipState.FIXED_TOPOLOGY_CORE.value
    assert context.ownership_by_node_id[290].owner_fragment_id == 289
    assert round_context_invariants(context) == []

    claimed, proof = claim_round_product(
        context,
        Stage3TopologyProduct(
            product_id="double-bond-flex",
            product_type="flex_chain",
            fixed_source_node_ids=[289, 290],
        ),
    )
    assert proof == {}
    assert claimed is not None
    assert claimed.ownership_by_node_id[289].state == Stage3OwnershipState.COMPLETED_TOPOLOGY.value
    assert claimed.ownership_by_node_id[290].state == Stage3OwnershipState.COMPLETED_TOPOLOGY.value


def test_released_evidence_neighbor_is_owned_only_by_its_explicit_fragment() -> None:
    nodes = [_node(1, 20), _node(2, 19)]
    classification = SimpleNamespace(
        to_dict=lambda: {
            "fragments": [
                {
                    "fragment_id": 1,
                    "source_node_ids": [1],
                    "claimed_node_ids": [1],
                    "context_node_ids": [2],
                    "evidence": {
                        "center_node_ids": [1],
                        "fixed_hop1_ids": [2],
                    },
                },
                {
                    "fragment_id": 2,
                    "source_node_ids": [2],
                    "claimed_node_ids": [2],
                    "context_node_ids": [],
                    "evidence": {"center_node_ids": [2]},
                },
            ],
            "resource_pool": {},
            "unclassified_nodes": [],
            "classification_rejects": [],
            "claim_conflicts": [],
            "support_units_only": [],
        },
    )

    context = build_stage3_round_context(
        nodes=nodes,
        classification=classification,
    )

    assert context.conflicts == []
    assert context.ownership_by_node_id[1].owner_fragment_id == 1
    assert context.ownership_by_node_id[2].owner_fragment_id == 2
    assert round_context_invariants(context) == []
