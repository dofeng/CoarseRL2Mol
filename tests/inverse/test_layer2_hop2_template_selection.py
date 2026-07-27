from __future__ import annotations

from collections import Counter

import torch

from model.inverse.layer2.inverse_layer2 import Layer2Estimator
from model.shared.coarse_graph import PPM_AXIS
from model.shared.inverse_common import _NodeV3


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
    )

    assert selected == key_near_mu
    assert mode == "approx_h2"
    assert meta["hop2_match_tier"] == 1
    assert meta["mu_diff"] == 2.0


def test_approx_template_progresses_beyond_old_two_node_cap() -> None:
    estimator = Layer2Estimator.__new__(Layer2Estimator)
    exact_h1 = (11, 13)
    target_h2 = (5, 23, 13)
    farther_h2 = (5, 5, 5, 13, 13, 23)
    key = (13, exact_h1, farther_h2)
    lib = {
        "templates": {key: {"center_mu": 128.0, "sample_count": 10}},
        "center_hop1_index": {
            (13, exact_h1): [(key, exact_h1, farther_h2, 10, 128.0)]
        },
    }

    selected, mode, meta = estimator._select_template_key(
        13,
        exact_h1,
        target_h2,
        130.0,
        lib,
    )

    assert selected == key
    assert mode == "approx_h2"
    assert meta["hop2_match_tier"] > 2


def test_approx_template_never_changes_layer1_environment() -> None:
    estimator = Layer2Estimator.__new__(Layer2Estimator)
    lib = {
        "templates": {(13, (11, 12), (5,)): {"center_mu": 128.0}},
        "center_hop1_index": {},
    }

    selected, mode, _meta = estimator._select_template_key(
        13,
        (11, 13),
        (5,),
        130.0,
        lib,
    )

    assert selected is None
    assert mode == "missing"


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


def test_layer2_keeps_center_z_and_always_uses_template_center_mu() -> None:
    key = (22, (), ())
    template = {
        "center_z": torch.tensor([1.0]),
        "center_mu": 12.0,
        "center_sample_idx": 0,
        "sample_count": 1,
        "samples": {
            "z": torch.tensor([[2.0]]),
            "mu": torch.tensor([24.0]),
            "pi": torch.tensor([3.0]),
        },
    }
    estimator = Layer2Estimator(device=torch.device("cpu"))
    estimator._template_cache = {
        "templates": {key: template},
        "center_hop1_index": {(22, ()): [(key, (), (), 1, 12.0)]},
    }
    node = _NodeV3(global_id=0, su_type=22)
    node.mu = 12.0

    result = estimator.layer2_assign(
        [node],
        S_target=torch.zeros_like(PPM_AXIS),
        E_target=torch.zeros(6),
    )

    assert torch.equal(result[0].z_vec, torch.tensor([1.0]))
    assert result[0].mu == 12.0
    assert result[0].pi == 1.0
    assert result[0].score_components["layer2_spectral_source"] == "layer2_template_center_mu"
    assert result[0].score_components["spectral_source"] == "layer2_template_center_mu"
    assert estimator.stats["selected_spectral_source"] == "template"
    assert estimator.stats["spectral_contract"] == "template_center_mu"


def test_layer2_decoder_configuration_is_not_part_of_the_api() -> None:
    parameters = __import__('inspect').signature(Layer2Estimator.layer2_assign).parameters

    assert "layer2_decoder_max_mu_delta" not in parameters
    assert "layer2_use_g2s_decoder" not in parameters
