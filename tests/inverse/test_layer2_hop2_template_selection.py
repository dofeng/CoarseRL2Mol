from __future__ import annotations

from collections import Counter

from model.inverse.layer2.inverse_layer2 import Layer2Estimator


class _Node:
    def __init__(self, node_id: int, su_type: int, neighbors: list[int]) -> None:
        self.global_id = int(node_id)
        self.su_type = int(su_type)
        self.hop1_ids = list(neighbors)
        self.hop1_su = Counter()
        self.hop2_su = Counter()
        self.mu = 0.0


def test_hop2_is_derived_from_neighbors_neighbors_and_excludes_center() -> None:
    nodes = [
        _Node(0, 22, [1]),
        _Node(1, 23, [0, 2]),
        _Node(2, 11, [3, 1, 4]),
        _Node(3, 5, [2]),
        _Node(4, 13, [2, 5]),
        _Node(5, 13, [4, 6]),
        _Node(6, 13, [5]),
    ]

    Layer2Estimator.__new__(Layer2Estimator)._derive_hop2(nodes)

    assert nodes[0].hop2_ids == [2]
    assert nodes[0].hop2_su == Counter({11: 1})
    assert nodes[1].hop2_ids == [3, 4]
    assert nodes[1].hop2_su == Counter({5: 1, 13: 1})
    assert nodes[4].hop2_ids == [1, 3, 6]
    assert nodes[4].hop2_su == Counter({23: 1, 5: 1, 13: 1})


def test_hop2_ignores_asymmetric_hop1_edges() -> None:
    nodes = [
        _Node(0, 22, [1]),
        _Node(1, 23, [0, 2]),
        _Node(2, 11, []),
    ]

    Layer2Estimator.__new__(Layer2Estimator)._derive_hop2(nodes)

    assert nodes[0].hop2_ids == []
    assert nodes[0].hop2_su == Counter()


def test_approx_template_uses_hop2_difference_tier_then_mu_prior() -> None:
    estimator = Layer2Estimator.__new__(Layer2Estimator)
    exact_h1 = (1, 2, 4)
    target_h2 = (1, 1, 2, 3, 4)
    key_far_mu = (0, exact_h1, (1, 2, 2, 3, 4))
    key_near_mu = (0, exact_h1, (1, 1, 2, 4, 4))
    key_second_tier = (0, exact_h1, (1, 2, 2, 4, 4))
    templates = {
        key_far_mu: {"center_mu": 136.0, "sample_count": 20},
        key_near_mu: {"center_mu": 128.0, "sample_count": 10},
        key_second_tier: {"center_mu": 130.0, "sample_count": 100},
    }
    lib = {
        "templates": templates,
        "center_hop1_index": {
            (0, exact_h1): [
                (key_far_mu, exact_h1, key_far_mu[2], 20, 136.0),
                (key_near_mu, exact_h1, key_near_mu[2], 10, 128.0),
                (key_second_tier, exact_h1, key_second_tier[2], 100, 130.0),
            ]
        },
    }

    selected, mode, meta = estimator._select_template_key(
        0,
        exact_h1,
        target_h2,
        130.0,
        lib,
        max_hop2_diff_nodes=2,
    )

    assert selected == key_near_mu
    assert mode == "approx_h2"
    assert meta["hop2_match_tier"] == 1
    assert meta["mu_diff"] == 2.0


def test_layer1_node_mu_has_priority_over_library_aggregate_prior() -> None:
    estimator = Layer2Estimator.__new__(Layer2Estimator)
    node = _Node(0, 13, [])
    node.mu = 128.0
    lib = {
        "templates": {
            (13, (11, 13), (5, 13, 23)): {
                "center_mu": 136.0,
                "sample_count": 20,
            }
        }
    }

    prior = estimator._resolve_layer1_mu_prior(node, 13, (11, 13), lib)

    assert prior == 128.0
