from __future__ import annotations

from model.inverse.topology.assembly.external import ExternalTopologyAssembler
from model.inverse.topology.assembly.state import ownership as ownership_module
from model.inverse.topology.assembly.closure import chain as chain_materializer_module
from model.inverse.topology.assembly.ring import segments as ring_segments_module
from model.inverse.topology.assembly.projection import project_fragment_views
from model.inverse.topology.assembly.state.round_context import build_stage3_round_context
from model.inverse.topology.classification.engine import classify_topology_units
from model.inverse.topology.resources.profiles import (
    build_raw_su24_resource_profile,
    raw_su24_segment_support,
)
from model.inverse.topology.stage3_models import Stage3OwnershipState
from tools.topology_audit.helpers import connect, make_node


def _raw_b_su24_fixture():
    center = make_node(100, 24, target_hop1_degree=3)
    cap = make_node(101, 11, target_hop1_degree=1)
    terminal = make_node(102, 22, target_hop1_degree=1)
    open_neighbor = make_node(103, 23, target_hop1_degree=1)
    connect(center, cap)
    connect(center, terminal)
    connect(center, open_neighbor)
    return [center, cap, terminal, open_neighbor]


def test_raw_b_su24_binds_real_cap_and_terminal_before_stage3_pooling():
    nodes = _raw_b_su24_fixture()
    classification = classify_topology_units(nodes)

    assert classification.su24_support_bindings == [{
        "su24_node_id": 100,
        "role": "B",
        "fixed_support_ids_by_su": {"11": [101], "22": [102]},
        "fixed_support_ids": [101, 102],
        "open_neighbor_ids": [103],
    }]
    assert classification.resource_pool["remaining_ids_by_su"]["11"] == []
    assert classification.resource_pool["remaining_ids_by_su"]["22"] == []
    assert classification.resource_pool["remaining_ids_by_su"]["24"] == [100]
    assert classification.resource_pool["remaining_ids_by_su"]["23"] == [103]


def test_raw_a_su24_open_neighbors_use_logical_h_not_real_branch_stubs():
    center = make_node(100, 24, target_hop1_degree=3)
    cap = make_node(101, 11, target_hop1_degree=1)
    open_23 = make_node(102, 23, target_hop1_degree=1)
    open_special = make_node(
        103,
        19,
        target_hop1_degree=3,
        target_fixed_anchor_count=2,
        special_anchor_mode="double",
    )
    for neighbor in [cap, open_23, open_special]:
        connect(center, neighbor)
    nodes = [center, cap, open_23, open_special]
    profile = build_raw_su24_resource_profile(
        center,
        {int(node.global_id): node for node in nodes},
    )

    assert profile is not None
    assert profile.role == "A"
    assert profile.cap_node_ids == [101]
    assert profile.open_neighbor_ids == [102, 103]
    support = raw_su24_segment_support(
        profile,
        skeleton_kind="side",
        slot_role="left_anchor",
    )
    assert support["support_required_counts"] == {23: 1, 22: 1}
    assert support["existing_branch_stubs"] == []


def test_bound_supports_are_not_generic_claimable_and_ring_claim_is_atomic():
    nodes = _raw_b_su24_fixture()
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification, flex_audit={})

    assert assembler.allocator.claim(11, 1) is None
    assert assembler.allocator.claim(22, 1) is None
    assert ownership_module.is_bound_su24_support(assembler, 100, 11, 101)
    assert ownership_module.is_bound_su24_support(assembler, 100, 22, 102)

    segment = ring_segments_module.claim_ring_pool_segment(
        assembler,
        24,
        mode="test",
        candidate_id="test:su24:100",
        node_id=100,
        skeleton_kind="side",
        role="left_anchor",
        profile_role="B",
        support_specific_ids_by_su={11: [101], 22: [102]},
    )
    assert segment is not None
    assert segment.fixed_source_node_ids == [101, 102]
    assert assembler.allocator.remaining_ids_by_su["24"] == []

    context = build_stage3_round_context(nodes=nodes, classification=classification)
    assert context.ownership_by_node_id[101].state == Stage3OwnershipState.SUPPORT_CONTEXT.value
    assert context.ownership_by_node_id[102].state == Stage3OwnershipState.SUPPORT_CONTEXT.value
    assert 101 not in context.generic_closure_pool_ids_by_su[11]
    assert 102 not in context.generic_closure_pool_ids_by_su[22]


def test_bound_support_owned_by_another_product_is_rejected_and_rolled_back():
    nodes = _raw_b_su24_fixture()
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={},
        occupied_node_ids={102},
    )
    before = {
        key: list(values)
        for key, values in assembler.allocator.remaining_ids_by_su.items()
    }

    segment = ring_segments_module.claim_ring_pool_segment(
        assembler,
        24,
        mode="test",
        candidate_id="test:su24:100",
        node_id=100,
        skeleton_kind="side",
        role="left_anchor",
        profile_role="B",
        support_specific_ids_by_su={11: [101], 22: [102]},
    )
    assert segment is None
    assert assembler.allocator.remaining_ids_by_su == before


def test_shared_support_is_reported_as_a_classification_conflict():
    first = make_node(100, 24, target_hop1_degree=3)
    second = make_node(104, 24, target_hop1_degree=3)
    shared_cap = make_node(101, 11, target_hop1_degree=2)
    first_terminal = make_node(102, 22, target_hop1_degree=1)
    second_terminal = make_node(105, 22, target_hop1_degree=1)
    first_open = make_node(103, 23, target_hop1_degree=1)
    second_open = make_node(106, 23, target_hop1_degree=1)
    connect(first, shared_cap)
    connect(first, first_terminal)
    connect(first, first_open)
    connect(second, shared_cap)
    connect(second, second_terminal)
    connect(second, second_open)

    classification = classify_topology_units([
        first,
        second,
        shared_cap,
        first_terminal,
        second_terminal,
        first_open,
        second_open,
    ])

    assert any(
        item.get("reason") == "su24_support_bound_to_multiple_centers"
        and item.get("support_node_id") == 101
        for item in classification.claim_conflicts
    )
    assembler = ExternalTopologyAssembler(
        [
            first,
            second,
            shared_cap,
            first_terminal,
            second_terminal,
            first_open,
            second_open,
        ],
        classification,
        flex_audit={},
    )
    assert not ownership_module.is_bound_su24_support(assembler, 100, 11, 101)
    assert not ownership_module.is_bound_su24_support(assembler, 104, 11, 101)


def test_single19d3_terminal_28_branch_is_explicitly_ring_capable() -> None:
    center = make_node(
        1,
        19,
        target_hop1_degree=3,
        target_fixed_anchor_count=1,
        special_anchor_mode="single",
        special_anchor_partition="ether",
    )
    connector_28 = make_node(2, 28, target_hop1_degree=1)
    aromatic_11 = make_node(3, 11, target_hop1_degree=1)
    open_23 = make_node(4, 23, target_hop1_degree=1)
    connect(center, connector_28, fixed=True)
    connect(center, aromatic_11)
    connect(center, open_23)

    classification = classify_topology_units(
        [center, connector_28, aromatic_11, open_23]
    )
    fragment = next(
        item for item in classification.fragments if int(item.fragment_id) == 1
    )
    assert fragment.class_type == "24_AB_N"
    assert "fixed_branch_double_open_flex" not in fragment.evidence
    assert "terminal_28_ring_only_branch" not in fragment.evidence
    assert fragment.evidence["connector_profiles"][0]["side_class"] == "N"
    assert fragment.evidence["assembly_capabilities"]["ring_allowed"] is True
    assert fragment.evidence["ring_allowed"] is True
    assert "side" in fragment.evidence["ring_allowed_skeletons"]
    assert fragment.evidence["fixed_connection_role"] == "branch"

    view = project_fragment_views(
        classification,
        {int(node.global_id): node for node in [center, connector_28, aromatic_11, open_23]},
    )[0]
    assert view.can_ring is True
    assert view.can_flex is False
    assert view.can_side is True
    assert view.preferred_product_type != "ring_candidate"
    assert set(view.target_roles) == {"fatty_ring", "side_chain"}
    capabilities = dict(fragment.evidence["assembly_capabilities"])
    assert capabilities["ring_allowed"] is True
    assert capabilities["side_allowed"] is True
    assert capabilities["flex_allowed"] is False
    assert capabilities["fixed_connection_role"] == "branch"

    assembler = ExternalTopologyAssembler(
        [center, connector_28, aromatic_11, open_23],
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    flex_views = chain_materializer_module.available_flex_reconcile_views(
        assembler,
        require_roles=["d_like_open", "double_end_flex_source"],
        require_materializable=False,
    )
    assert int(view.fragment_id) not in {
        int(item.fragment_id) for item in flex_views
    }
