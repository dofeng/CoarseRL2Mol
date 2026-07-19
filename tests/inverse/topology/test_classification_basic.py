from __future__ import annotations

from model.inverse.topology.classification import classify_topology_units

from .helpers import connect, make_node


def _class_types(result: object) -> set[str]:
    return {str(fragment.class_type) for fragment in list(getattr(result, "fragments", []) or [])}


def test_classifier_creates_basic_a_fragment() -> None:
    center = make_node(100, 0)
    aromatic_9 = make_node(9, 9)
    aromatic_6 = make_node(6, 6)
    connect(center, aromatic_9)
    connect(center, aromatic_6)

    result = classify_topology_units([center, aromatic_9, aromatic_6])
    assert "A" in _class_types(result)


def test_classifier_creates_basic_c_fragment() -> None:
    center = make_node(129, 29)
    aromatic_5 = make_node(5, 5)
    terminal_19d1 = make_node(219, 19, target_hop1_degree=1, special_anchor_mode="single")
    connect(center, aromatic_5)
    connect(center, terminal_19d1)

    result = classify_topology_units([center, aromatic_5, terminal_19d1])
    assert "C" in _class_types(result)


def test_classifier_creates_basic_f_fragment() -> None:
    center = make_node(128, 28)
    aromatic_5 = make_node(5, 5)
    connect(center, aromatic_5)

    result = classify_topology_units([center, aromatic_5])
    assert "F" in _class_types(result)


def test_classifier_rejects_wide_basic_b_guess_for_invalid_center_0_pattern() -> None:
    center = make_node(100, 0)
    open_23 = make_node(223, 23)
    aromatic_9 = make_node(9, 9)
    connect(center, open_23)
    connect(center, aromatic_9)

    result = classify_topology_units([center, open_23, aromatic_9])
    assert "B" not in _class_types(result)
    assert int(result.summary.get("unclassified_non_resource_count", 0)) >= 1


def test_classifier_puts_unclaimed_pool_nodes_into_resource_pool() -> None:
    result = classify_topology_units([make_node(0, 11), make_node(1, 22), make_node(2, 23), make_node(3, 24)])
    pool = dict(result.resource_pool or {})
    assert dict(pool.get("remaining_counts", {}) or {}).get("11") == 1
    assert dict(pool.get("remaining_counts", {}) or {}).get("24") == 1


def test_classifier_keeps_nonfixed_open_su23_as_evidence_and_resource() -> None:
    center = make_node(10, 3)
    aromatic = make_node(11, 9)
    abstract_open = make_node(12, 23)
    connect(center, aromatic)
    connect(center, abstract_open)

    result = classify_topology_units([center, aromatic, abstract_open])
    pool = dict(result.resource_pool or {})

    assert "B" in _class_types(result)
    fragment = next(item for item in result.fragments if str(item.class_type) == "B")
    assert 12 in list(fragment.claimed_node_ids or [])
    assert list(dict(pool.get("remaining_ids_by_su", {}) or {}).get("23", []) or []) == [12]
