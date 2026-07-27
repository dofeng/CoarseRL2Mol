from __future__ import annotations

import pytest

from model.inverse.topology.classification import build_node_token, normalize_subtype

from .helpers import connect, make_node


def test_normalize_subtype_maps_single_19d2() -> None:
    node = make_node(1, 19, target_hop1_degree=2, special_anchor_mode="single")
    assert normalize_subtype(node) == "single19d2"


def test_normalize_subtype_maps_double_19d2() -> None:
    node = make_node(1, 19, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    assert normalize_subtype(node) == "double19d2"


def test_normalize_subtype_maps_single_20d3() -> None:
    node = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single")
    assert normalize_subtype(node) == "single20d3"


def test_build_node_token_captures_structured_neighbors() -> None:
    center = make_node(
        7,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    connector = make_node(29, 29)
    aromatic = make_node(5, 5)
    pool_11 = make_node(11, 11)
    terminal = make_node(22, 22)

    connect(center, connector, fixed=True)
    connect(connector, aromatic)
    connect(center, pool_11)
    connect(center, terminal)

    token = build_node_token(center, nodes_by_id={node.global_id: node for node in [center, connector, aromatic, pool_11, terminal]})
    assert token.node_id == 7
    assert token.anchor_mode == "single"
    assert token.fixed_anchor_count == 1
    assert list(token.fixed_neighbor_ids) == [29]
    assert list(token.connector_neighbor_ids) == [29]
    assert sorted(token.terminal_neighbor_ids) == [22]
    assert len(list(token.neighbor_tokens or [])) == 3
    connector_neighbor = next(neighbor for neighbor in list(token.neighbor_tokens or []) if int(neighbor.node_id) == 29)
    assert connector_neighbor.is_fixed is True
    assert connector_neighbor.subtype == "29"


def test_build_node_token_rejects_dangling_neighbor_identity() -> None:
    center = make_node(7, 3)
    center.hop1_ids.append(999)

    with pytest.raises(ValueError, match=r"node 7 references missing hop1 node 999"):
        build_node_token(center, nodes_by_id={7: center})
