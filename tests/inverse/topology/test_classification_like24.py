from __future__ import annotations

from model.inverse.topology.classification import classify_topology_units
from model.inverse.topology.classification.classifier import _ring_like24_units_for_fragment

from .helpers import connect, make_node


def _fragment_by_class(result: object, class_type: str):
    for fragment in list(getattr(result, "fragments", []) or []):
        if str(fragment.class_type) == str(class_type):
            return fragment
    return None


def test_classifier_creates_24_ab_m_for_single19d3_with_fixed_m_connector() -> None:
    center = make_node(300, 19, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(329, 29)
    aromatic_5 = make_node(5, 5)
    pool_11 = make_node(11, 11)
    open_23 = make_node(23, 23)
    connect(center, connector, fixed=True)
    connect(connector, aromatic_5)
    connect(center, pool_11)
    connect(center, open_23)

    result = classify_topology_units([center, connector, aromatic_5, pool_11, open_23])
    fragment = _fragment_by_class(result, "24_AB_M")
    assert fragment is not None
    assert str(fragment.family) == "like24_secondary"


def test_fixed_partition_meta_can_forbid_fixed_side_classification() -> None:
    center = make_node(1300, 19, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(1329, 29)
    aromatic = make_node(1305, 5)
    pool_11 = make_node(1311, 11)
    open_23 = make_node(1323, 23)
    connect(center, connector, fixed=True)
    connect(connector, aromatic)
    connect(center, pool_11)
    connect(center, open_23)

    result = classify_topology_units(
        [center, connector, aromatic, pool_11, open_23],
        fixed_partition_meta={"forbidden_fixed_connector_node_ids": [1300]},
    )

    assert _fragment_by_class(result, "24_AB_M") is None
    assert _fragment_by_class(result, "24_A") is None
    assert any(
        str(item.get("reason", "")) == "fixed_connector_forbidden_by_partition_meta"
        for item in list(result.classification_rejects or [])
    )


def test_su19_requires_explicit_valid_anchor_partition() -> None:
    center = make_node(
        1400,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        special_anchor_partition=None,
        target_fixed_anchor_count=1,
    )

    result = classify_topology_units([center])

    assert any(
        int(item.get("node_id", -1)) == 1400
        and str(item.get("reason", "")) == "su19_anchor_partition_missing"
        for item in list(result.classification_rejects or [])
    )


def test_su19_thio_partition_rejects_ether_connector() -> None:
    center = make_node(
        1401,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        special_anchor_partition="thio",
        target_fixed_anchor_count=1,
    )
    connector = make_node(1429, 29)
    aromatic = make_node(1405, 5)
    anchor = make_node(1411, 11)
    open_unit = make_node(1423, 23)
    connect(center, connector, fixed=True)
    connect(connector, aromatic)
    connect(center, anchor)
    connect(center, open_unit)

    result = classify_topology_units([center, connector, aromatic, anchor, open_unit])

    assert _fragment_by_class(result, "24_AB_M") is None
    assert any(
        str(item.get("reason", "")) == "su19_partition_connector_mismatch:thio:29"
        for item in list(result.classification_rejects or [])
    )


def test_su19_thio_partition_accepts_su31_connector() -> None:
    center = make_node(
        1402,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        special_anchor_partition="thio",
        target_fixed_anchor_count=1,
    )
    connector = make_node(1431, 31)
    aromatic = make_node(1407, 7)
    anchor = make_node(1412, 11)
    open_unit = make_node(1424, 23)
    connect(center, connector, fixed=True)
    connect(connector, aromatic)
    connect(center, anchor)
    connect(center, open_unit)

    result = classify_topology_units([center, connector, aromatic, anchor, open_unit])

    assert _fragment_by_class(result, "24_AB_M") is not None


def test_classifier_creates_24_ab_n_with_realization_modes() -> None:
    center = make_node(
        301,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        special_anchor_partition="thio",
        target_fixed_anchor_count=1,
    )
    connector = make_node(331, 31)
    side_single19d2 = make_node(401, 19, target_hop1_degree=2, special_anchor_mode="single")
    pool_11 = make_node(11, 11)
    terminal_28 = make_node(28, 28)
    connect(center, connector, fixed=True)
    connect(connector, side_single19d2)
    connect(center, pool_11)
    connect(center, terminal_28)

    result = classify_topology_units([center, connector, side_single19d2, pool_11, terminal_28])
    fragment = _fragment_by_class(result, "24_AB_N")
    assert fragment is not None
    assert sorted(list(fragment.evidence.get("realization_modes", []) or [])) == ["branch_open", "main_open"]


def test_classifier_prefers_m_primary_for_double19d3() -> None:
    center = make_node(302, 19, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    connector_2 = make_node(2, 2)
    connector_31 = make_node(31, 29)
    aromatic_9 = make_node(9, 9)
    side_single19d2 = make_node(402, 19, target_hop1_degree=2, special_anchor_mode="single")
    pool_11 = make_node(11, 11)
    connect(center, connector_2, fixed=True)
    connect(center, connector_31, fixed=True)
    connect(center, pool_11)
    connect(connector_2, aromatic_9)
    connect(connector_31, side_single19d2)

    result = classify_topology_units([center, connector_2, connector_31, aromatic_9, side_single19d2, pool_11])
    fragment = _fragment_by_class(result, "24_AB_M")
    assert fragment is not None
    assert str(dict(fragment.evidence or {}).get("selected_primary_connector", {}).get("side_class")) == "M"


def test_double19d3_preserves_su28_and_long_terminal_fixed_paths() -> None:
    aromatic_11 = make_node(180, 11)
    terminal_19 = make_node(317, 19, target_hop1_degree=1, special_anchor_mode="single")
    center = make_node(
        412,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    terminal_28 = make_node(505, 28)
    connector_29 = make_node(569, 29)
    connect(center, aromatic_11)
    connect(center, terminal_28, fixed=True)
    connect(center, connector_29, fixed=True)
    connect(connector_29, terminal_19)

    result = classify_topology_units([aromatic_11, terminal_19, center, terminal_28, connector_29])
    fragment = _fragment_by_class(result, "24_AB_P")

    assert fragment is not None
    assert sorted(fragment.claimed_node_ids) == [180, 317, 412, 505, 569]
    evidence = dict(fragment.evidence or {})
    assert int(dict(evidence.get("selected_primary_connector", {}) or {}).get("connector_node_id", -1)) == 569
    contracts = list(evidence.get("fixed_connection_contracts", []) or [])
    assert sorted(int(item["connector_node_id"]) for item in contracts) == [505, 569]
    assert _fragment_by_class(result, "24_B") is None


def test_double19d3_rejects_two_su28_fixed_connectors_without_base_fallback() -> None:
    center = make_node(
        600,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    aromatic_11 = make_node(601, 11)
    terminal_28_a = make_node(602, 28)
    terminal_28_b = make_node(603, 28)
    connect(center, aromatic_11)
    connect(center, terminal_28_a, fixed=True)
    connect(center, terminal_28_b, fixed=True)

    result = classify_topology_units([center, aromatic_11, terminal_28_a, terminal_28_b])

    assert not [fragment for fragment in result.fragments if int(fragment.fragment_id) == 600]
    assert any(
        str(item.get("reason", "")) == "double19_second_fixed_slot_disallows_su28"
        for item in list(result.classification_rejects or [])
    )


def test_double19d3_rejects_missing_second_fixed_connector_without_fallback() -> None:
    center = make_node(
        610,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    aromatic_11 = make_node(611, 11)
    connector_29 = make_node(612, 29)
    terminal_19 = make_node(613, 19, target_hop1_degree=1, special_anchor_mode="single")
    terminal_28 = make_node(614, 28)
    connect(center, aromatic_11)
    connect(center, connector_29, fixed=True)
    connect(connector_29, terminal_19)
    connect(center, terminal_28)

    result = classify_topology_units(
        [center, aromatic_11, connector_29, terminal_19, terminal_28]
    )

    assert not [fragment for fragment in result.fragments if int(fragment.fragment_id) == 610]
    assert any(
        str(item.get("reason", "")) == "fixed_connector_count_mismatch:1"
        for item in list(result.classification_rejects or [])
    )


def test_double19d3_rejects_target_fixed_count_mismatch() -> None:
    center = make_node(
        620,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=1,
    )
    aromatic_11 = make_node(621, 11)
    terminal_28 = make_node(622, 28)
    connector_29 = make_node(623, 29)
    terminal_19 = make_node(624, 19, target_hop1_degree=1, special_anchor_mode="single")
    connect(center, aromatic_11)
    connect(center, terminal_28, fixed=True)
    connect(center, connector_29, fixed=True)
    connect(connector_29, terminal_19)

    result = classify_topology_units(
        [center, aromatic_11, terminal_28, connector_29, terminal_19]
    )

    assert not [fragment for fragment in result.fragments if int(fragment.fragment_id) == 620]
    assert any(
        str(item.get("reason", "")) == "target_fixed_anchor_count_mismatch:1:2"
        for item in list(result.classification_rejects or [])
    )


def test_classifier_creates_24_cd_m_full_when_both_fixed_connectors_are_m() -> None:
    center = make_node(303, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    connector_0 = make_node(200, 0)
    connector_27 = make_node(227, 27)
    aromatic_9 = make_node(9, 9)
    aromatic_6 = make_node(6, 6)
    connect(center, connector_0, fixed=True)
    connect(center, connector_27, fixed=True)
    connect(connector_0, aromatic_9)
    connect(connector_27, aromatic_6)

    result = classify_topology_units([center, connector_0, connector_27, aromatic_9, aromatic_6])
    assert _fragment_by_class(result, "24_CD_M_FULL") is not None


def test_classifier_marks_non_primary_fixed_connector_as_branch_only_with_partner_spec() -> None:
    center = make_node(401, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    connector_0 = make_node(402, 0)
    aromatic_9 = make_node(409, 9)
    connector_27 = make_node(427, 27)
    side_single20d3 = make_node(520, 20, target_hop1_degree=3, special_anchor_mode="single")
    open_23 = make_node(523, 23)
    pool_11 = make_node(411, 11)
    connect(center, connector_0, fixed=True)
    connect(connector_0, aromatic_9)
    connect(center, connector_27, fixed=True)
    connect(connector_27, side_single20d3)
    connect(side_single20d3, open_23)
    connect(center, pool_11)

    result = classify_topology_units([center, connector_0, aromatic_9, connector_27, side_single20d3, open_23, pool_11])
    fragment = _fragment_by_class(result, "24_AB_M")
    assert fragment is not None
    contracts = list(dict(fragment.evidence or {}).get("fixed_connection_contracts", []) or [])
    partner_contract = next(item for item in contracts if int(dict(item or {}).get("connector_node_id", -1)) == 427)
    assert list(partner_contract.get("role_candidates", []) or []) == ["branch_only"]
    assert str(partner_contract.get("tail_closure_kind", "")) == "paired_like24"
    partner_spec = dict(partner_contract.get("partner_spec", {}) or {})
    assert str(partner_spec.get("partner_kind", "")) == "paired_like24"
    assert list(partner_spec.get("allowed_partner_connector_su_types", []) or []) == [27]


def test_classifier_keeps_21d3_as_base_like24_without_fixed_side_resolution() -> None:
    center = make_node(304, 21, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    pool_11 = make_node(11, 11)
    terminal_32 = make_node(32, 32)
    connect(center, pool_11)
    connect(center, terminal_32)

    result = classify_topology_units([center, pool_11, terminal_32])
    fragment = _fragment_by_class(result, "24_B")
    assert fragment is not None
    assert dict(fragment.evidence or {}).get("selected_primary_connector", {}) == {}


def test_classifier_covers_single19d3_with_fixed_28_and_open_neighbors() -> None:
    center = make_node(
        1903,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
        special_degree_source="19d3",
    )
    fixed_28 = make_node(1928, 28)
    unsaturated_15 = make_node(1915, 15)
    open_23 = make_node(1923, 23)
    connect(center, fixed_28, fixed=True)
    connect(center, unsaturated_15)
    connect(center, open_23)

    result = classify_topology_units([center, fixed_28, unsaturated_15, open_23])

    assert _fragment_by_class(result, "24_CD_P") is not None
    assert not [
        reject
        for reject in result.classification_rejects
        if int(reject.get("global_id", reject.get("node_id", -1))) == 1903
    ]


def test_classifier_keeps_connected_raw_24_as_real_resource() -> None:
    center = make_node(324, 24)
    pool_11 = make_node(11, 11)
    open_23 = make_node(23, 23)
    connect(center, pool_11)
    connect(center, open_23)

    result = classify_topology_units([center, pool_11, open_23])
    assert not any(str(fragment.class_type) in {"24_A", "24_B", "24_C", "24_D"} for fragment in result.fragments)
    assert result.resource_pool["remaining_ids_by_su"]["24"] == [324]
    assert not any(
        int(item.get("node_id", -1)) == 324
        for item in list(result.classification_rejects or [])
    )


def test_classifier_keeps_incomplete_raw_24_in_resource_pool() -> None:
    center = make_node(1324, 24)
    open_23 = make_node(1323, 23)
    connect(center, open_23)

    result = classify_topology_units([center, open_23])
    assert _fragment_by_class(result, "24_C") is None
    assert result.resource_pool["remaining_ids_by_su"]["24"] == [1324]


def test_classifier_projects_reciprocal_14_15_atomic_pair() -> None:
    center = make_node(1414, 14)
    partner = make_node(1415, 15)
    connect(center, partner)

    result = classify_topology_units([center, partner])
    fragment = _fragment_by_class(result, "24_C")
    assert fragment is not None
    assert fragment.claimed_node_ids == [1414, 1415]
    assert [slot.kind for slot in fragment.branch_contracts] == ["H"]


def test_like24_open_aliphatic_neighbors_remain_unowned_pool_resources() -> None:
    center = make_node(1514, 14)
    partner = make_node(1515, 15)
    left_open = make_node(1523, 23)
    right_open = make_node(1524, 23)
    connect(center, partner)
    connect(center, left_open)
    connect(partner, right_open)

    result = classify_topology_units([center, partner, left_open, right_open])
    fragment = _fragment_by_class(result, "24_C")

    assert fragment is not None
    assert fragment.claimed_node_ids == [1514, 1515]
    assert result.resource_pool["remaining_ids_by_su"]["23"] == [1523, 1524]


def test_classifier_keeps_fixed_14_15_pair_atomic() -> None:
    center = make_node(1414, 14)
    partner = make_node(1415, 15)
    connect(center, partner, fixed=True)

    result = classify_topology_units([center, partner])

    assert not [
        item
        for item in result.unclassified_nodes
        if int(item.get("node_id", -1)) == 1415
    ]
    fragment = _fragment_by_class(result, "24_C")
    assert fragment is not None
    assert fragment.claimed_node_ids == [1414, 1415]
    assert not [item for item in result.support_units_only if int(item.get("node_id", -1)) == 1415]


def test_classifier_does_not_demote_reciprocal_14_15_partner_to_support() -> None:
    center = make_node(1414, 14)
    partner = make_node(1415, 15)
    connect(center, partner)

    result = classify_topology_units([center, partner])

    assert not [item for item in result.unclassified_nodes if int(item.get("node_id", -1)) == 1415]
    fragment = _fragment_by_class(result, "24_C")
    assert fragment is not None
    assert fragment.claimed_node_ids == [1414, 1415]
    assert not [item for item in result.support_units_only if int(item.get("node_id", -1)) == 1415]


def test_classifier_includes_fixed_ether_connector_other_endpoint() -> None:
    center = make_node(
        433,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    connector = make_node(571, 29)
    endpoint = make_node(
        340,
        19,
        target_hop1_degree=1,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    terminal = make_node(505, 28)
    anchor = make_node(173, 11)
    connect(center, connector, fixed=True)
    connect(connector, endpoint, fixed=True)
    connect(center, terminal, fixed=True)
    connect(center, anchor)

    result = classify_topology_units([center, connector, endpoint, terminal, anchor])
    fragment = next(item for item in result.fragments if int(item.fragment_id) == 433)

    assert 340 in set(fragment.source_node_ids)
    assert 340 in set(fragment.claimed_node_ids)
    assert not [item for item in result.unclassified_nodes if int(item.get("node_id", -1)) == 340]


def test_su15_reciprocal_su14_partner_is_claimed_by_same_atomic_fragment() -> None:
    primary = make_node(341, 14)
    partner = make_node(343, 15)
    open_primary = make_node(467, 23)
    open_partner = make_node(475, 23)
    connect(primary, partner)
    connect(primary, open_primary)
    connect(partner, open_partner)

    result = classify_topology_units([primary, partner, open_primary, open_partner])

    assert all(int(item.get("node_id", -1)) != 343 for item in result.unclassified_nodes)
    fragment = next(item for item in result.fragments if int(item.fragment_id) == 341)
    assert {341, 343}.issubset(set(fragment.source_node_ids))
    assert {341, 343}.issubset(set(fragment.claimed_node_ids))
    assert dict(fragment.evidence or {}).get("atomic_group_node_ids") == [341, 343]



def test_unsaturated_atomic_groups_are_reciprocal_and_indivisible() -> None:
    left = make_node(150, 15)
    right = make_node(151, 16)
    open_23 = make_node(230, 23)
    connect(left, right)
    connect(left, open_23)

    result = classify_topology_units([left, right, open_23])

    assert result.atomic_groups == [{
        "group_id": 150,
        "kind": "double_bond_pair",
        "node_ids": [150, 151],
        "su_pair": [15, 16],
    }]
    fragment = next(item for item in result.fragments if 150 in set(item.claimed_node_ids))
    assert {150, 151}.issubset(set(fragment.claimed_node_ids))


def test_invalid_unsaturated_singleton_is_rejected_before_topology_classification() -> None:
    singleton = make_node(170, 17)
    open_23 = make_node(231, 23)
    connect(singleton, open_23)

    result = classify_topology_units([singleton, open_23])

    rejects = [item for item in result.classification_rejects if int(item.get("node_id", -1)) == 170]
    assert rejects
    assert str(rejects[0].get("reason", "")) == "unsaturated_pair_count_mismatch"
    assert not [fragment for fragment in result.fragments if 170 in set(fragment.claimed_node_ids)]

def test_classifier_rejects_incomplete_21d3_with_one_neighbor() -> None:
    center = make_node(1421, 21, target_hop1_degree=3, special_anchor_mode="single")
    terminal = make_node(1432, 32)
    connect(center, terminal)

    result = classify_topology_units([center, terminal])
    assert _fragment_by_class(result, "24_D") is None


def test_classifier_rejects_incomplete_single19d3_with_one_neighbor() -> None:
    center = make_node(1419, 19, target_hop1_degree=3, special_anchor_mode="single")
    terminal = make_node(1428, 28)
    connect(center, terminal)

    result = classify_topology_units([center, terminal])
    assert _fragment_by_class(result, "24_D") is None


def test_classifier_allows_14_14_pair_bridge() -> None:
    left = make_node(414, 14)
    right = make_node(415, 14)
    open_23 = make_node(23, 23)
    open_24 = make_node(24, 24)
    connect(left, right)
    connect(left, open_23)
    connect(right, open_24)

    result = classify_topology_units([left, right, open_23, open_24])
    fragment = _fragment_by_class(result, "24_CD_N_24_CD")
    assert fragment is not None
    evidence = dict(fragment.evidence or {})
    assert "ring_like24_span" not in evidence
    assert {"d_like_open", "double_end_flex_source", "like24_open"} <= set(list(evidence.get("flex_reconcile_roles", []) or []))
    assert str(evidence.get("flex_reconcile_main_kind", "")) == "d_like_open"
    assert str(evidence.get("flex_reconcile_preferred_product_type", "")) == "flex_chain"
    assert int(evidence.get("flex_reconcile_priority", 0) or 0) == 40
    assert int(evidence.get("ring_like24_units", 0) or 0) == 2
    ring_roles = set(list(evidence.get("ring_roles", []) or []))
    assert {"upper_core", "fused_core"} <= ring_roles
    assert "side_middle" not in ring_roles


def test_classifier_blocks_ring_eligibility_when_0_connector_is_mainline() -> None:
    center = make_node(500, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector_0 = make_node(501, 0)
    aromatic_11 = make_node(502, 11)
    side_single20d2 = make_node(503, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(center, connector_0, fixed=True)
    connect(center, aromatic_11)
    connect(connector_0, side_single20d2)

    result = classify_topology_units([center, connector_0, aromatic_11, side_single20d2])
    fragment = _fragment_by_class(result, "24_AB_N")
    assert fragment is not None
    evidence = dict(fragment.evidence or {})
    assert evidence.get("ring_block_reason") == "mainline_fixed_connector_0_2"
    assert list(evidence.get("ring_allowed_skeletons", []) or []) == []


def test_classifier_marks_24_ab_n_24_cd_as_side_path_capable() -> None:
    left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    left_connector = make_node(1001, 27)
    left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    left_11 = make_node(1003, 11)
    connect(left, left_connector, fixed=True)
    connect(left_connector, left_tail)
    connect(left, left_11)

    right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right_connector = make_node(1011, 27)
    right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(right, right_connector, fixed=True)
    connect(right_connector, right_tail)

    bridge = make_node(1020, 27)
    connect(bridge, left)
    connect(bridge, right)

    result = classify_topology_units([left, left_connector, left_tail, left_11, right, right_connector, right_tail, bridge])
    fragment = _fragment_by_class(result, "24_AB_N_24_CD")
    assert fragment is not None
    evidence = dict(fragment.evidence or {})
    assert "ring_like24_span" not in evidence
    assert list(evidence.get("ring_allowed_skeletons", []) or []) == ["side"]
    side_caps = list(dict(evidence.get("ring_path_capabilities", {}) or {}).get("side", []) or [])
    fused_caps = list(dict(evidence.get("ring_path_capabilities", {}) or {}).get("fused", []) or [])
    assert "terminal_core_pair" in side_caps
    assert fused_caps == []
    assert {"b_like_open", "single_end_flex_source", "like24_open"} <= set(list(evidence.get("flex_reconcile_roles", []) or []))
    assert str(evidence.get("flex_reconcile_main_kind", "")) == "b_like_open"
    assert str(evidence.get("flex_reconcile_preferred_product_type", "")) == "flex_chain"
    assert int(evidence.get("flex_reconcile_priority", 0) or 0) == 30
    assert int(evidence.get("ring_like24_units", 0) or 0) == 2
    assert {"side_anchor", "side_middle"} <= set(list(evidence.get("ring_roles", []) or []))
    contract = dict(dict(evidence.get("ring_contracts", {}) or {}).get("side_ring_plan", {}) or {})
    fused_contract = dict(dict(evidence.get("ring_contracts", {}) or {}).get("fused_ring_plan", {}) or {})
    assert bool(contract.get("eligible", False))
    assert not bool(fused_contract.get("eligible", False))


def test_classifier_allows_non_11_aromatic_terminal_path_capability() -> None:
    center = make_node(1200, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(1201, 27)
    aromatic_5 = make_node(1202, 6)
    aromatic_11 = make_node(1203, 11)
    connect(center, connector, fixed=True)
    connect(connector, aromatic_5)
    connect(center, aromatic_11)

    result = classify_topology_units([center, connector, aromatic_5, aromatic_11])
    fragment = _fragment_by_class(result, "24_AB_M")
    assert fragment is not None
    evidence = dict(fragment.evidence or {})
    assert "ring_like24_span" not in evidence
    assert str(evidence.get("selected_primary_connector_side_class", "")) == "M"
    assert bool(evidence.get("touches_aromatic_anchor", False))
    assert int(evidence.get("ring_like24_units", 0) or 0) == 1
    assert list(evidence.get("ring_roles", []) or []) == []


def test_classifier_does_not_create_external_ring_contract_for_connected_raw_24() -> None:
    center = make_node(1300, 24)
    pool_11 = make_node(1301, 11)
    open_23 = make_node(1302, 23)
    connect(center, pool_11)
    connect(center, open_23)

    result = classify_topology_units([center, pool_11, open_23])
    assert not any(str(fragment.class_type) in {"24_A", "24_B", "24_C", "24_D"} for fragment in result.fragments)
    assert result.resource_pool["remaining_ids_by_su"]["24"] == [1300]
    assert result.resource_pool["remaining_ids_by_su"]["11"] == [1301]
    assert result.resource_pool["remaining_ids_by_su"]["23"] == [1302]
    assert not any(
        int(item.get("node_id", -1)) == 1300
        for item in list(result.classification_rejects or [])
    )


def test_ring_like24_units_are_mapped_by_fragment_type_not_center_count() -> None:
    assert _ring_like24_units_for_fragment(
        family="like24_bridge",
        class_type="24_AB_N_24_CD",
        evidence={"center_node_ids": [1, 2, 3]},
    ) == 2
    assert _ring_like24_units_for_fragment(
        family="like24",
        class_type="24_A",
        evidence={"center_node_ids": [1, 2, 3]},
    ) == 1


def test_classifier_marks_only_closed_mainline_like24_as_explicit_flex() -> None:
    closed_center = make_node(300, 19, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    closed_connector = make_node(329, 29)
    aromatic_5 = make_node(5, 5)
    pool_11 = make_node(11, 11)
    open_23 = make_node(23, 23)
    connect(closed_center, closed_connector, fixed=True)
    connect(closed_connector, aromatic_5)
    connect(closed_center, pool_11)
    connect(closed_center, open_23)

    open_center = make_node(
        301,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        special_anchor_partition="thio",
        target_fixed_anchor_count=1,
    )
    open_connector = make_node(331, 31)
    side_single19d2 = make_node(401, 19, target_hop1_degree=2, special_anchor_mode="single")
    pool_11_b = make_node(111, 11)
    terminal_28 = make_node(28, 28)
    connect(open_center, open_connector, fixed=True)
    connect(open_connector, side_single19d2)
    connect(open_center, pool_11_b)
    connect(open_center, terminal_28)

    bridge_left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    bridge_left_connector = make_node(1001, 27)
    bridge_left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    bridge_left_11 = make_node(1003, 11)
    connect(bridge_left, bridge_left_connector, fixed=True)
    connect(bridge_left_connector, bridge_left_tail)
    connect(bridge_left, bridge_left_11)
    bridge_right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    bridge_right_connector = make_node(1011, 27)
    bridge_right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(bridge_right, bridge_right_connector, fixed=True)
    connect(bridge_right_connector, bridge_right_tail)
    bridge_mid = make_node(1020, 27)
    connect(bridge_mid, bridge_left)
    connect(bridge_mid, bridge_right)

    result = classify_topology_units([
        closed_center,
        closed_connector,
        aromatic_5,
        pool_11,
        open_23,
        open_center,
        open_connector,
        side_single19d2,
        pool_11_b,
        terminal_28,
        bridge_left,
        bridge_left_connector,
        bridge_left_tail,
        bridge_left_11,
        bridge_right,
        bridge_right_connector,
        bridge_right_tail,
        bridge_mid,
    ])

    closed_fragment = _fragment_by_class(result, "24_AB_M")
    assert closed_fragment is not None
    closed_evidence = dict(closed_fragment.evidence or {})
    assert bool(closed_evidence.get("flex_range_mainline_closed", False))
    assert str(closed_evidence.get("flex_range_candidate_kind", "")) == "explicit_flex"
    assert bool(closed_evidence.get("flex_range_counts_as_explicit", False))

    open_fragment = _fragment_by_class(result, "24_AB_N")
    assert open_fragment is not None
    open_evidence = dict(open_fragment.evidence or {})
    assert bool(open_evidence.get("flex_range_mainline_open", False))
    assert str(open_evidence.get("flex_range_candidate_kind", "")) == "b_like_open"
    assert not bool(open_evidence.get("flex_range_counts_as_explicit", False))

    bridge_fragment = _fragment_by_class(result, "24_AB_N_24_CD")
    assert bridge_fragment is not None
    bridge_evidence = dict(bridge_fragment.evidence or {})
    assert bool(bridge_evidence.get("flex_range_mainline_open", False))
    assert str(bridge_evidence.get("flex_range_candidate_kind", "")) == "b_like_open"
    assert not bool(bridge_evidence.get("flex_range_counts_as_explicit", False))


def test_classifier_allows_shared_nonfixed_15_between_like24_centers() -> None:
    left = make_node(314, 14)
    bridge_15 = make_node(315, 15)
    right = make_node(
        406,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    right_23 = make_node(447, 23)
    right_28 = make_node(518, 28)
    left_23 = make_node(401, 23)
    left_28 = make_node(405, 28)
    connect(left, bridge_15)
    connect(left, left_23)
    connect(left, left_28)
    connect(right, bridge_15)
    connect(right, right_23)
    connect(right, right_28, fixed=True)

    result = classify_topology_units([left, bridge_15, right, right_23, right_28, left_23, left_28])

    assert any(406 in fragment.source_node_ids for fragment in result.fragments)
    assert not [item for item in result.unclassified_nodes if int(item.get("node_id", -1)) in {406, 518}]
