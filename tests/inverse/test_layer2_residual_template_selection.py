from __future__ import annotations

from collections import Counter

import pytest
import torch

from model.inverse.layer2.inverse_layer2 import Layer2Estimator
from model.shared.coarse_graph import PPM_AXIS
from model.shared.inverse_common import _NodeV3


def _template(mu: float, sample_count: int = 1) -> dict:
    return {
        "center_z": torch.tensor([float(mu)]),
        "center_mu": float(mu),
        "center_sample_idx": 0,
        "sample_count": int(sample_count),
    }


def _estimator(templates: dict) -> Layer2Estimator:
    estimator = Layer2Estimator(device=torch.device("cpu"))
    index = {}
    for key, template in templates.items():
        center_su, hop1, hop2 = key
        index.setdefault((center_su, hop1), []).append(
            (key, hop1, hop2, template["sample_count"], template["center_mu"])
        )
    estimator._template_cache = {
        "templates": templates,
        "center_hop1_index": index,
    }
    estimator._derive_hop2 = lambda _nodes: None
    return estimator


def _node(node_id: int,
          su_type: int,
          hop1: tuple[int, ...] = (),
          hop2: tuple[int, ...] = (),
          mu_prior: float = 80.0) -> _NodeV3:
    return _NodeV3(
        global_id=node_id,
        su_type=su_type,
        hop1_su=Counter(hop1),
        hop2_su=Counter(hop2),
        mu=mu_prior,
    )


def _peak(estimator: Layer2Estimator, mu: float) -> torch.Tensor:
    return estimator._unit_peak(mu, PPM_AXIS.double()).float()


def _assign(estimator: Layer2Estimator,
            nodes: list[_NodeV3],
            target: torch.Tensor,
            inventory_support: tuple[int, ...] = ()) -> list[_NodeV3]:
    node_count = len(nodes)
    next_id = max((int(node.global_id) for node in nodes), default=-1) + 1
    support_nodes = [
        _node(next_id + offset, su_type, mu_prior=0.0)
        for offset, su_type in enumerate(inventory_support)
    ]
    result = estimator.layer2_assign(
        nodes + support_nodes,
        S_target=target,
        E_target=torch.zeros(6),
    )
    return result[:node_count]


def test_exact_template_is_fixed_even_when_target_favors_an_approximate_key() -> None:
    exact_key = (0, (), ())
    approximate_key = (0, (), (1,))
    estimator = _estimator({
        exact_key: _template(40.0),
        approximate_key: _template(100.0),
    })

    result = _assign(estimator, [_node(0, 0)], _peak(estimator, 100.0))

    assert result[0].template_key == exact_key
    assert result[0].mu == 40.0
    assert result[0].score_components["layer2_match_mode"] == "exact"


def test_spectral_score_cannot_select_a_higher_hop2_distance_tier() -> None:
    minimum_tier_key = (0, (), (26,))
    higher_tier_key = (0, (), (26, 27))
    estimator = _estimator({
        minimum_tier_key: _template(40.0),
        higher_tier_key: _template(100.0),
    })

    result = _assign(
        estimator,
        [_node(0, 0)],
        _peak(estimator, 100.0),
        inventory_support=(26, 27),
    )

    assert result[0].template_key == minimum_tier_key
    assert result[0].score_components["layer2_match_mode"] == "forced_approx"
    assert result[0].score_components["layer2_hop2_match_tier"] == 1


def test_exact_and_forced_assignments_form_the_fixed_residual() -> None:
    exact_key = (0, (), ())
    forced_key = (1, (), (26,))
    ambiguous_60 = (2, (), (27,))
    ambiguous_100 = (2, (), (28,))
    estimator = _estimator({
        exact_key: _template(20.0),
        forced_key: _template(40.0),
        ambiguous_60: _template(60.0),
        ambiguous_100: _template(100.0),
    })
    target = (
        _peak(estimator, 20.0)
        + _peak(estimator, 40.0)
        + _peak(estimator, 100.0)
    )

    nodes = _assign(
        estimator,
        [_node(0, 0), _node(1, 1), _node(2, 2)],
        target,
        inventory_support=(26, 27, 28),
    )
    ambiguous = nodes[2]

    assert nodes[0].score_components["layer2_match_mode"] == "exact"
    assert nodes[1].score_components["layer2_match_mode"] == "forced_approx"
    assert ambiguous.template_key == ambiguous_100
    assert ambiguous.score_components["layer2_match_mode"] == "approx_residual_fill"
    expected_sse = float(torch.sum(_peak(estimator, 100.0).double() ** 2).item())
    assert ambiguous.score_components["layer2_residual_sse_before"] == pytest.approx(
        expected_sse,
        rel=1e-5,
    )


def test_residual_is_updated_after_each_ambiguous_node() -> None:
    key_60 = (0, (), (26,))
    key_100 = (0, (), (27,))
    estimator = _estimator({
        key_60: _template(60.0),
        key_100: _template(100.0),
    })
    target = 1.1 * _peak(estimator, 60.0) + _peak(estimator, 100.0)

    nodes = _assign(
        estimator,
        [_node(0, 0), _node(1, 0)],
        target,
        inventory_support=(26, 27),
    )

    assert nodes[0].template_key == key_60
    assert nodes[1].template_key == key_100
    assert (
        nodes[1].score_components["layer2_residual_sse_before"]
        == pytest.approx(nodes[0].score_components["layer2_residual_sse_after"])
    )


def test_ambiguous_candidate_that_reduces_sse_is_residual_fill() -> None:
    key_40 = (0, (), (26,))
    key_100 = (0, (), (27,))
    estimator = _estimator({
        key_40: _template(40.0),
        key_100: _template(100.0),
    })

    result = _assign(
        estimator,
        [_node(0, 0)],
        _peak(estimator, 100.0),
        inventory_support=(26, 27),
    )

    assert result[0].template_key == key_100
    assert result[0].score_components["layer2_match_mode"] == "approx_residual_fill"
    assert result[0].score_components["layer2_residual_sse_gain"] > 0.0


def test_all_worsening_candidates_choose_the_least_damage() -> None:
    edge_key = (0, (), (26,))
    center_key = (0, (), (27,))
    estimator = _estimator({
        edge_key: _template(0.2),
        center_key: _template(120.0),
    })

    result = _assign(
        estimator,
        [_node(0, 0)],
        torch.zeros_like(PPM_AXIS),
        inventory_support=(26, 27),
    )

    assert result[0].template_key == edge_key
    assert result[0].score_components["layer2_match_mode"] == "approx_least_damage"
    assert result[0].score_components["layer2_residual_sse_gain"] < 0.0


def test_residual_ties_use_mu_prior_sample_count_and_template_key() -> None:
    lower_count = (0, (), (26,))
    lower_key = (0, (), (27,))
    higher_key = (0, (), (28,))
    estimator = _estimator({
        lower_count: _template(100.0, sample_count=5),
        lower_key: _template(100.0, sample_count=10),
        higher_key: _template(100.0, sample_count=10),
    })

    result = _assign(
        estimator,
        [_node(0, 0, mu_prior=100.0)],
        _peak(estimator, 100.0),
        inventory_support=(26, 27, 28),
    )

    assert result[0].template_key == lower_key
    assert result[0].score_components["layer2_candidate_count"] == 3


def test_absent_su_candidate_is_filtered_before_hop2_tier_selection() -> None:
    absent_su32_key = (0, (), (32,))
    valid_next_tier_key = (0, (), (0, 0))
    estimator = _estimator({
        absent_su32_key: _template(100.0),
        valid_next_tier_key: _template(40.0),
    })

    result = _assign(estimator, [_node(0, 0)], _peak(estimator, 100.0))

    assert result[0].template_key == valid_next_tier_key
    assert result[0].score_components["layer2_hop2_match_tier"] == 2
    assert estimator.stats["inventory_filtered_candidates"] == 1
    assert estimator.stats["inventory_filtered_su_types"] == {32: 1}
