from __future__ import annotations

from model.inverse.topology.classification import classify_topology_units

from .csv_fixture_loader import load_nodes_from_final_csv
from .helpers import connect, make_node


CSV_PATH = "/home/dofengqi/Graphnmr/test_results/YM-4 copy/final_outputs/final_nodes_by_global_id.csv"


def _class_types(result: object) -> set[str]:
    return {str(fragment.class_type) for fragment in list(getattr(result, "fragments", []) or [])}


def test_double19d2_supports_all_major_component_combinations() -> None:
    # M + M -> A+
    center_a = make_node(500, 19, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    c2_a = make_node(502, 2)
    c29_a = make_node(529, 29)
    a9 = make_node(509, 9)
    a5 = make_node(505, 5)
    connect(center_a, c2_a, fixed=True)
    connect(center_a, c29_a, fixed=True)
    connect(c2_a, a9, fixed=True)
    connect(c29_a, a5, fixed=True)

    # M + P -> C+
    center_c = make_node(510, 19, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    c2_c = make_node(512, 2)
    c28_c = make_node(528, 28)
    c9 = make_node(519, 9)
    connect(center_c, c2_c, fixed=True)
    connect(center_c, c28_c, fixed=True)
    connect(c2_c, c9, fixed=True)

    # M + N -> B+
    center_b = make_node(520, 19, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    c2_b = make_node(522, 2)
    c31_b = make_node(531, 29)
    b9 = make_node(5299, 9)
    b_side = make_node(540, 19, target_hop1_degree=2, special_anchor_mode="single")
    connect(center_b, c2_b, fixed=True)
    connect(center_b, c31_b, fixed=True)
    connect(c2_b, b9, fixed=True)
    connect(c31_b, b_side)

    result = classify_topology_units([
        center_a, c2_a, c29_a, a9, a5,
        center_c, c2_c, c28_c, c9,
        center_b, c2_b, c31_b, b9, b_side,
    ])
    classes = _class_types(result)
    assert {"A+", "B+", "C+"}.issubset(classes)


def test_double20d2_supports_n_n_and_n_p() -> None:
    # N + N -> D+
    center_d = make_node(600, 20, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    c0_d = make_node(601, 0)
    c27_d = make_node(627, 27)
    open_23 = make_node(623, 23)
    side_single20d2 = make_node(640, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(center_d, c0_d, fixed=True)
    connect(center_d, c27_d, fixed=True)
    connect(c0_d, open_23)
    connect(c27_d, side_single20d2)

    # N + P -> E+
    center_e = make_node(610, 20, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    c0_e = make_node(611, 0)
    c27_e = make_node(637, 27)
    open_24 = make_node(624, 24)
    side_20d1 = make_node(641, 20, target_hop1_degree=1, special_anchor_mode="single")
    connect(center_e, c0_e, fixed=True)
    connect(center_e, c27_e, fixed=True)
    connect(c0_e, open_24)
    connect(c27_e, side_20d1, fixed=True)

    result = classify_topology_units([
        center_d, c0_d, c27_d, open_23, side_single20d2,
        center_e, c0_e, c27_e, open_24, side_20d1,
    ])
    classes = _class_types(result)
    assert {"D+", "E+"}.issubset(classes)


def test_classifier_reports_rejects_and_keeps_consumed_ids_unique() -> None:
    valid_center = make_node(700, 19, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    valid_c2 = make_node(702, 2)
    valid_c29 = make_node(729, 29)
    valid_9 = make_node(709, 9)
    valid_5 = make_node(705, 5)
    connect(valid_center, valid_c2, fixed=True)
    connect(valid_center, valid_c29, fixed=True)
    connect(valid_c2, valid_9)
    connect(valid_c29, valid_5)

    reject_center = make_node(710, 20, target_hop1_degree=2, special_anchor_mode="double", target_fixed_anchor_count=2)
    reject_c0 = make_node(711, 0)
    reject_c27 = make_node(737, 27)
    wrong_side = make_node(738, 8)
    side_20d1 = make_node(741, 20, target_hop1_degree=1, special_anchor_mode="single")
    connect(reject_center, reject_c0, fixed=True)
    connect(reject_center, reject_c27, fixed=True)
    connect(reject_c0, wrong_side, fixed=True)
    connect(reject_c27, side_20d1)

    result = classify_topology_units([
        valid_center, valid_c2, valid_c29, valid_9, valid_5,
        reject_center, reject_c0, reject_c27, wrong_side, side_20d1,
    ])

    consumed = list(result.consumed_node_ids or [])
    assert len(consumed) == len(set(consumed))
    assert int(result.summary.get("fragment_count", 0)) == len(list(result.fragments or []))
    assert int(result.summary.get("reject_count", 0)) >= 1
    assert len(list(result.classification_rejects or [])) >= 1


def test_classifier_allows_shared_open_support_between_b_and_g() -> None:
    center_b = make_node(1000, 3)
    aromatic = make_node(1001, 9)
    shared_open = make_node(1002, 23)
    center_g = make_node(1003, 1)
    connect(center_b, aromatic)
    connect(center_b, shared_open)
    connect(center_g, shared_open)

    result = classify_topology_units([center_b, aromatic, shared_open, center_g])
    classes = _class_types(result)
    print("shared-open classes", sorted(classes))
    print("shared-open claim_conflicts", list(getattr(result, "claim_conflicts", []) or []))
    print("shared-open true_no_rule", list(result.unclassified_nodes or []))

    assert "B" in classes
    assert "G" in classes
    assert not list(getattr(result, "claim_conflicts", []) or [])


def test_resource_pool_separates_general_resources_from_su11_donors() -> None:
    nodes = [
        make_node(11, 11),
        make_node(22, 22),
        make_node(23, 23),
        make_node(24, 24),
        make_node(10, 10),
        make_node(12, 12),
        make_node(13, 13),
    ]

    result = classify_topology_units(nodes)
    pool = dict(result.resource_pool or {})

    assert set(dict(pool.get("remaining_ids_by_su", {}) or {})) == {"11", "22", "23", "24"}
    assert pool["remaining_ids_by_su"] == {"11": [11], "22": [22], "23": [23], "24": [24]}
    assert pool["su11_donor_ids_by_su"] == {"10": [10], "12": [12], "13": [13]}


def test_classifier_csv_regression_key_nodes_print() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    result = classify_topology_units(nodes)
    fragments_by_center = {}
    for fragment in list(result.fragments or []):
        for node_id in list(dict(fragment.evidence or {}).get("center_node_ids", []) or []):
            fragments_by_center.setdefault(int(node_id), []).append(fragment)

    node_21_classes = [str(fragment.class_type) for fragment in fragments_by_center.get(21, [])]
    node_392_classes = [str(fragment.class_type) for fragment in fragments_by_center.get(392, [])]
    reject_392 = [item for item in list(result.classification_rejects or []) if int(item.get("node_id", -1)) == 392]
    no_rule_21 = [item for item in list(result.unclassified_nodes or []) if int(item.get("node_id", -1)) == 21]

    print("csv-node-21 classes", node_21_classes)
    print("csv-node-392 classes", node_392_classes)
    print("csv-node-392 rejects", reject_392)
    print("csv-node-21 true_no_rule", no_rule_21)
    print("csv-claim-conflicts", list(getattr(result, "claim_conflicts", []) or []))

    assert "B" in node_21_classes
    assert "24_D" in node_392_classes
    assert reject_392 == []
    assert no_rule_21 == []
