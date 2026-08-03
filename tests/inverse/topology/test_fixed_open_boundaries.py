from __future__ import annotations

import pytest

from model.inverse.topology.assembly_external import (
    ExternalTopologyAssembler,
    _trace_real_branch_path,
    project_fragment_views,
)
from model.inverse.topology.classification import classify_topology_units

from .helpers import connect, make_node


def _fragment_by_class(result, class_type: str):
    return next(
        (fragment for fragment in result.fragments if fragment.class_type == class_type),
        None,
    )


def _fragment433_graph(offset: int = 0):
    left = make_node(433 + offset, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right = make_node(434 + offset, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    bridge = make_node(503 + offset, 27)
    open_19_a = make_node(418 + offset, 19, target_hop1_degree=3, special_anchor_mode="single")
    open_19_b = make_node(427 + offset, 19, target_hop1_degree=3, special_anchor_mode="single")
    open_23 = make_node(435 + offset, 23)
    connect(left, bridge, fixed=True)
    connect(right, bridge, fixed=True)
    connect(left, open_19_a)
    connect(left, open_19_b)
    connect(right, open_23)
    return [left, right, bridge, open_19_a, open_19_b, open_23]


def _fragment439_graph():
    center = make_node(
        439,
        20,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    connector = make_node(0, 0)
    structural_boundary = make_node(900, 9)
    open_11_a = make_node(418, 11)
    open_11_b = make_node(427, 11)
    connect(center, connector, fixed=True)
    connect(connector, structural_boundary)
    connect(center, open_11_a)
    connect(center, open_11_b)
    return [center, connector, structural_boundary, open_11_a, open_11_b]


def _fragment436_graph():
    center = make_node(
        436,
        20,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    connector = make_node(0, 0)
    connector_boundary = make_node(900, 9)
    center_boundary_a = make_node(901, 23)
    center_boundary_b = make_node(902, 23)
    connect(center, connector, fixed=True)
    connect(connector, connector_boundary)
    connect(center, center_boundary_a)
    connect(center, center_boundary_b)
    return [
        center,
        connector,
        connector_boundary,
        center_boundary_a,
        center_boundary_b,
    ]


@pytest.mark.parametrize(
    ("su_type", "kwargs"),
    [
        (20, {"target_hop1_degree": 3, "special_anchor_mode": "single"}),
        (7, {}),
        (22, {}),
        (0, {}),
        (2, {}),
        (3, {}),
    ],
)
def test_every_nonfixed_like24_neighbor_is_context_only(su_type: int, kwargs: dict) -> None:
    center = make_node(100, 19, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    fixed_28 = make_node(128, 28)
    boundary = make_node(200, su_type, **kwargs)
    connect(center, fixed_28, fixed=True)
    connect(center, boundary)

    result = classify_topology_units([center, fixed_28, boundary])
    fragment = next(item for item in result.fragments if item.fragment_id == 100)

    assert boundary.global_id not in fragment.claimed_node_ids
    assert boundary.global_id in fragment.context_node_ids
    assert any(
        item["boundary_neighbor_id"] == boundary.global_id
        for item in fragment.open_boundaries
    )


def test_m_path_requires_fixed_second_edge_and_n_open_path_stops_at_connector() -> None:
    def build(*, far_fixed: bool, far_su: int, far_kwargs: dict):
        center = make_node(
            300,
            19,
            target_hop1_degree=3,
            special_anchor_mode="single",
            special_anchor_partition="thio",
            target_fixed_anchor_count=1,
        )
        connector = make_node(331, 31)
        far = make_node(307, far_su, **far_kwargs)
        anchor = make_node(311, 11)
        open_23 = make_node(323, 23)
        connect(center, connector, fixed=True)
        connect(connector, far, fixed=far_fixed)
        connect(center, anchor)
        connect(center, open_23)
        return [center, connector, far, anchor, open_23]

    nonfixed_m = classify_topology_units(build(far_fixed=False, far_su=7, far_kwargs={}))
    assert _fragment_by_class(nonfixed_m, "24_AB_M") is None

    fixed_m = classify_topology_units(build(far_fixed=True, far_su=7, far_kwargs={}))
    m_fragment = _fragment_by_class(fixed_m, "24_AB_M")
    assert m_fragment is not None
    assert {300, 331, 307} <= set(m_fragment.claimed_node_ids)

    open_n = classify_topology_units(build(
        far_fixed=False,
        far_su=19,
        far_kwargs={"target_hop1_degree": 2, "special_anchor_mode": "single"},
    ))
    n_fragment = _fragment_by_class(open_n, "24_AB_N")
    assert n_fragment is not None
    contract = n_fragment.evidence["fixed_connection_contracts"][0]
    assert contract["base_path_node_ids"] == [331]
    assert contract["tail_closure_kind"] == "OPEN_BOUNDARY"
    assert 307 not in n_fragment.claimed_node_ids


def test_fixed_7_31_single19d2_path_stops_before_nonfixed_neighbor() -> None:
    aromatic = make_node(7, 7)
    connector = make_node(31, 31)
    d2 = make_node(19, 19, target_hop1_degree=2, special_anchor_mode="single")
    boundary = make_node(23, 23)
    connect(aromatic, connector, fixed=True)
    connect(connector, d2, fixed=True)
    connect(d2, boundary)
    nodes = [aromatic, connector, d2, boundary]

    trace = _trace_real_branch_path(
        center_id=7,
        first_node_id=31,
        nodes_by_id={node.global_id: node for node in nodes},
    )

    assert ["7", *trace["base_path_labels"]] == ["7", "31", "single19d2"]
    assert trace["terminated_kind"] == "open_boundary"
    assert trace["open_boundary"]["boundary_neighbor_id"] == 23


def test_su0_fixed_side_is_concrete_but_its_nonfixed_side_is_open() -> None:
    center = make_node(20, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(0, 0)
    aromatic = make_node(9, 9)
    nonfixed = make_node(3, 3)
    connect(center, connector, fixed=True)
    connect(connector, aromatic, fixed=True)
    connect(center, nonfixed)

    result = classify_topology_units([center, connector, aromatic, nonfixed])
    fragment = _fragment_by_class(result, "24_CD_M")
    assert fragment is not None
    assert {20, 0, 9} <= set(fragment.claimed_node_ids)
    assert 3 not in fragment.claimed_node_ids
    assert 3 in fragment.context_node_ids


def test_fragment433_pair_does_not_claim_nonfixed_special_neighbors() -> None:
    nodes = _fragment433_graph()
    result = classify_topology_units(nodes)
    fragment = _fragment_by_class(result, "24_CD_N_24_CD")

    assert fragment is not None
    assert fragment.claimed_node_ids == [433, 434, 503]
    assert {418, 427, 435} <= set(fragment.context_node_ids)
    assert not (set(fragment.claimed_node_ids) & set(fragment.context_node_ids))
    assert len(fragment.branch_contracts) == 2
    assert {slot.anchor_node_id for slot in fragment.branch_contracts} == {433, 434}
    assert all(
        contract.get("paired_like24_resolved") is True
        for contract in fragment.evidence["fixed_connection_contracts"]
    )


def test_fragment433_signature_is_invariant_to_node_id_permutation() -> None:
    def signature(offset: int):
        fragment = _fragment_by_class(
            classify_topology_units(_fragment433_graph(offset)),
            "24_CD_N_24_CD",
        )
        node_su = {node.global_id: node.su_type for node in _fragment433_graph(offset)}
        return (
            fragment.class_type,
            tuple(sorted(node_su[node_id] for node_id in fragment.claimed_node_ids)),
            tuple(sorted(item["neighbor_su_type"] for item in fragment.open_boundaries)),
            tuple(sorted(slot.kind for slot in fragment.branch_contracts)),
        )

    assert signature(0) == signature(1000)


def test_projector_never_traces_nonfixed_fragment433_edges(monkeypatch) -> None:
    nodes = _fragment433_graph()
    result = classify_topology_units(nodes)

    def fail_trace(**_kwargs):
        raise AssertionError("nonfixed edge reached path tracer")

    monkeypatch.setattr(
        "model.inverse.topology.assembly_external._trace_real_branch_path",
        fail_trace,
    )
    views = project_fragment_views(result, {node.global_id: node for node in nodes})
    assert views[0].main_node_ids == [433, 503, 434]


def test_fragment433_stage3_has_no_topology_materialization_error_with_resources() -> None:
    nodes = _fragment433_graph()
    for index in range(4):
        nodes.extend([
            make_node(600 + index, 11),
            make_node(700 + index, 22),
            make_node(800 + index, 23),
        ])
    classification = classify_topology_units(nodes)
    assembly = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()

    serialized = str(assembly)
    assert "stage3_topology_materialization_invalid" not in serialized
    assert "non_h_structure_unit_in_real_branch_path" not in serialized
    product = next(
        item
        for item in assembly["products"]
        if 433 in set(item["source_fragment_ids"])
    )
    assert product["product_kind"] == "flex_chain"
    assert product["actual_topology_su"] == (
        "11-23-single20d3(-23-22)-27-single20d3(-23-22)-23-11"
    )
    assert not [
        item
        for item in assembly["pending_units"]
        if 433 in set(item["source_fragment_ids"])
    ]


def test_fragment439_uses_two_open_mainline_slots_and_an_open_fixed_branch() -> None:
    nodes = _fragment439_graph()
    nodes.extend([make_node(600, 23), make_node(601, 23)])
    classification = classify_topology_units(nodes)
    fragment = next(
        item for item in classification.fragments if item.fragment_id == 439
    )

    assert fragment.class_type == "24_A"
    assert set(fragment.claimed_node_ids) == {0, 439}
    assert {418, 427, 900} <= set(fragment.context_node_ids)
    assert len(fragment.open_mainline_slots) == 2
    assert all(slot.kind == "F1" for slot in fragment.open_mainline_slots)
    assert {slot.anchor_node_id for slot in fragment.open_mainline_slots} == {439}
    assert "real_su11_pair_mainline" not in fragment.evidence

    contracts = list(fragment.evidence["fixed_connection_contracts"])
    assert len(contracts) == 1
    contract = contracts[0]
    assert contract["side_class"] == "N"
    assert contract["role_candidates"] == ["branch_only"]
    assert contract["base_path_node_ids"] == [0]
    assert contract["tail_closure_kind"] == "OPEN_BOUNDARY"
    assert contract["externally_closed"] is False

    view = next(
        item
        for item in project_fragment_views(
            classification,
            {node.global_id: node for node in nodes},
        )
        if item.fragment_id == 439
    )
    assert view.main_node_ids == [439]
    assert view.open_left is True
    assert view.open_right is True
    assert "d_like_open" in view.flex_reconcile_roles
    assert "explicit_flex" not in view.flex_reconcile_roles

    assembly = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()
    serialized = str(assembly)
    assert "external_branch_closure_pending" not in serialized
    product = next(
        item
        for item in assembly["products"]
        if 439 in set(item["source_fragment_ids"])
    )
    assert product["product_kind"] == "flex_chain"
    assert not [
        item
        for item in assembly["pending_units"]
        if 439 in set(item["source_fragment_ids"])
    ]


def test_fragment436_materializes_zero_id_double_open_mainline_and_h_branch() -> None:
    nodes = _fragment436_graph()
    nodes.extend([
        make_node(600, 11),
        make_node(604, 11),
        make_node(601, 22),
        make_node(602, 22),
        make_node(603, 23),
    ])
    classification = classify_topology_units(nodes)
    fragment = next(
        item for item in classification.fragments if item.fragment_id == 436
    )

    assert fragment.class_type == "24_CD_N"
    assert set(fragment.claimed_node_ids) == {0, 436}
    assert len(fragment.open_mainline_slots) == 2
    assert {slot.anchor_node_id for slot in fragment.open_mainline_slots} == {0, 436}
    assert [slot.kind for slot in fragment.branch_contracts] == ["H"]

    assembly = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()

    product = next(
        item
        for item in assembly["products"]
        if 436 in set(item["source_fragment_ids"])
    )
    assert product["product_kind"] == "flex_chain"
    assert product["actual_topology_su"] == (
        "11-23-single20d3(-23-22)-0-23-11"
    )
    assert not [
        item
        for item in assembly["pending_units"]
        if 436 in set(item["source_fragment_ids"])
    ]


def test_none_side_class_is_rejected_before_contract_generation(monkeypatch) -> None:
    nodes = _fragment439_graph()

    def invalid_profile(_center, connector, _tokens):
        return {
            "connector_node_id": int(connector.node_id),
            "connector_su_type": int(connector.su_type),
            "side_class": None,
            "reject_reason": "forced_invalid_profile",
        }

    monkeypatch.setattr(
        "model.inverse.topology.classification.classifier.classify_like24_fixed_side",
        invalid_profile,
    )
    classification = classify_topology_units(nodes)

    assert any(
        item.get("node_id") == 439
        and item.get("reason") == "connector_side_classification_failed"
        for item in classification.classification_rejects
    )
    assert not any(
        item.fragment_id == 439
        and item.evidence.get("fixed_connection_contracts")
        for item in classification.fragments
    )


def test_asymmetric_fixed_edge_is_audited_and_not_traversable() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    connect(center, connector)
    center.fixed_hop1_ids.add(2)

    result = classify_topology_units([center, connector])

    assert result.summary["fixed_edge_violation_count"] == 1
    assert any(
        item.get("reason") == "asymmetric_fixed_edge"
        for item in result.classification_rejects
    )
