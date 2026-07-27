from __future__ import annotations

from collections import Counter, defaultdict
import inspect

import numpy as np
import pytest
import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.hop1_adjuster import Hop1Adjuster
from model.inverse.layer1.inverse_layer1 import Layer1Assigner
from model.inverse.layer1.layer1_refiner import Layer1Refiner
from model.inverse.layer1.layer1_eval import Layer1NmrEvaluator
from model.shared.coarse_graph import E_SU
from model.shared.inverse_common import _NodeV3


def _connect_test_nodes(nodes: list[_NodeV3], left_id: int, right_id: int) -> None:
    left = nodes[int(left_id)]
    right = nodes[int(right_id)]
    left.hop1_ids.append(int(right_id))
    right.hop1_ids.append(int(left_id))
    left.hop1_su[int(right.su_type)] += 1
    right.hop1_su[int(left.su_type)] += 1


def _topology_state(nodes: list[_NodeV3]) -> list[tuple[Counter, list[int], set[int]]]:
    return [
        (Counter(node.hop1_su), list(node.hop1_ids), set(node.fixed_hop1_ids))
        for node in nodes
    ]


def _unconstrained_hop1_adjuster() -> Hop1Adjuster:
    adjuster = Hop1Adjuster(
        port_combinations=None,
        validate_connection_fn=None,
        enforce_external_requirements=False,
    )
    adjuster.hop1_by_su = defaultdict(list)
    adjuster.template_by_key = {}
    return adjuster


def test_layer1_public_api_has_only_full_fresh_build_controls() -> None:
    pipeline_parameters = inspect.signature(InversePipelineV3.layer1_assign).parameters
    assigner_parameters = inspect.signature(Layer1Assigner.layer1_assign).parameters

    for obsolete in (
        "seed_nodes",
        "preserve_seed_node_ids",
        "layer1_repair_mode",
        "layer1_global_repair_max_attempts",
        "initial_enable_spectral_refine",
        "initial_hop1_adjust_iterations",
        "initial_hop1_adjust_max_candidate_trials",
        "profile",
        "eval_allow_approx",
    ):
        assert obsolete not in pipeline_parameters
        assert obsolete not in assigner_parameters

    for obsolete in ("local_swap_max_attempts",):
        assert obsolete not in pipeline_parameters
        assert obsolete not in assigner_parameters

    for required in (
        "hop1_adjust_iterations",
        "hop1_adjust_max_candidate_trials",
        "hop1_completion_max_backtracks",
        "hop1_completion_branch_limit",
        "enable_hop1_diversity_refine",
        "hop1_diversity_max_moves",
        "hop1_diversity_max_trials",
    ):
        assert required in pipeline_parameters
        assert required in assigner_parameters


@pytest.mark.parametrize(
    ("layer1_config", "layer2_config", "removed_key"),
    [
        ({"local_swap_max_attempts": 8}, None, "local_swap_max_attempts"),
        ({"layer1_total_time_budget_s": 30.0}, None, "layer1_total_time_budget_s"),
        ({"hop1_spectral_reserve_s": 5.0}, None, "hop1_spectral_reserve_s"),
        (None, {"layer2_decoder_max_mu_delta": 5.0}, "layer2_decoder_max_mu_delta"),
        (None, {"layer2_use_g2s_decoder": True}, "layer2_use_g2s_decoder"),
    ],
)
def test_removed_layer_configs_fail_explicitly(
    layer1_config,
    layer2_config,
    removed_key,
) -> None:
    pipeline = object.__new__(InversePipelineV3)

    with pytest.raises(ValueError, match=removed_key):
        pipeline._build_layer_runtime_configs(
            eval_hwhm=1.0,
            enable_hop1_adjust=True,
            hop1_adjust_iterations=4,
            hop1_neg_threshold=-0.7,
            hop1_pos_threshold=0.7,
            layer1_config=layer1_config,
            layer2_config=layer2_config,
            layer3_config=None,
        )


def test_layer1_group_uses_one_weighted_median_mu_for_all_nodes(monkeypatch) -> None:
    evaluator = Layer1NmrEvaluator("cpu", E_SU)
    lib_index = {
        "agg": {
            (13, (11, 13)): {
                "mu_sorted": np.asarray([110.0, 128.0]),
                "w_sorted": np.asarray([1.0, 3.0]),
                "cdf": np.asarray([0.25, 1.0]),
                "n_templates": 2,
                "mu_min": 110.0,
                "mu_max": 128.0,
            }
        },
        "center_to_hop1": {13: [(11, 13)]},
        "approx_hop1_cache": {},
    }
    monkeypatch.setattr(evaluator, "get_library_index", lambda _path: lib_index)
    nodes = [_NodeV3(global_id=index, su_type=13) for index in range(3)]
    for node in nodes:
        node.hop1_su.update({11: 1, 13: 1})

    assignments = evaluator.build_grouped_assignments(nodes, "unused.pt", allow_approx=False)

    assert [assignments[index]["mu"] for index in range(3)] == [128.0, 128.0, 128.0]
    assert [assignments[index]["pi"] for index in range(3)] == [1.0, 1.0, 1.0]


def test_layer1_library_does_not_require_or_read_template_pi(tmp_path) -> None:
    library_path = tmp_path / "mu_only_templates.pt"
    torch.save({
        "templates": {
            (13, (11, 13), (5,)): {
                "center_mu": 128.0,
                "sample_count": 3,
            }
        }
    }, library_path)
    evaluator = Layer1NmrEvaluator("cpu", E_SU)
    node = _NodeV3(global_id=0, su_type=13)
    node.hop1_su.update({11: 1, 13: 1})

    assignments = evaluator.build_grouped_assignments(
        [node],
        str(library_path),
        allow_approx=False,
    )

    assert assignments[0]["mu"] == 128.0
    assert assignments[0]["pi"] == 1.0


def test_hop1_alternative_resource_filter_excludes_center_and_missing_types() -> None:
    nodes_by_su = defaultdict(list, {
        25: [0],
        22: [1, 2],
    })

    shortage = Layer1Refiner._alternative_resource_shortage(
        nodes_by_su,
        center_su=25,
        hop1_tuple=(22, 25, 25, 30),
    )

    assert shortage == {25: 2, 30: 1}


def test_hop1_alternative_ranking_penalizes_overused_motif() -> None:
    adjuster = _unconstrained_hop1_adjuster()
    adjuster.hop1_by_su[13] = [
        {
            "hop1_tuple": (9, 10),
            "mu_median": 120.0,
            "mu_min": 118.0,
            "mu_max": 122.0,
        },
        {
            "hop1_tuple": (12, 12),
            "mu_median": 120.0,
            "mu_min": 118.0,
            "mu_max": 122.0,
        },
    ]
    motif_usage = Counter({(13, (9, 10)): 20})

    alternatives = adjuster.find_alternative_hop1(
        center_su=13,
        current_hop1=(13, 13),
        target_mu_range=(118.0, 122.0),
        motif_usage=motif_usage,
    )

    assert [entry["hop1_tuple"] for entry in alternatives] == [
        (12, 12),
        (9, 10),
    ]


def test_layer1_template_contract_checks_partial_and_exact_multisets() -> None:
    assigner = Layer1Assigner(device="cpu")
    assigner._enforce_hop1_template_contract = True
    assigner._hop1_templates_by_su = {13: ((5, 11),)}
    node = _NodeV3(global_id=0, su_type=13)
    node.target_hop1_degree = 2

    assert assigner._hop1_template_feasible(node, neighbor_su=5) is True
    assert assigner._hop1_template_feasible(node, neighbor_su=12) is False

    node.hop1_su.update({5: 1, 11: 1})
    assert assigner._hop1_template_feasible(node, require_exact=True) is True
    node.hop1_su.clear()
    node.hop1_su.update({5: 1, 12: 1})
    assert assigner._hop1_template_feasible(node, require_exact=True) is False


def test_can_add_cache_reuses_unrelated_state_and_invalidates_changed_endpoint(monkeypatch) -> None:
    assigner = Layer1Assigner(device="cpu")
    nodes = [_NodeV3(global_id=index, su_type=23) for index in range(4)]
    calls = []

    def _uncached(_nodes, left, right):
        calls.append((int(left.global_id), int(right.global_id)))
        return True

    monkeypatch.setattr(assigner, "_can_add_hop1_connection_uncached", _uncached)

    assert assigner._can_add_hop1_connection(nodes, nodes[0], nodes[1]) is True
    assert assigner._can_add_hop1_connection(nodes, nodes[0], nodes[1]) is True
    assert calls == [(0, 1)]

    # An unrelated topology edit must not invalidate the cached endpoint pair.
    _connect_test_nodes(nodes, 2, 3)
    assert assigner._can_add_hop1_connection(nodes, nodes[0], nodes[1]) is True
    assert calls == [(0, 1)]

    # Editing either endpoint changes its state signature and forces reevaluation.
    nodes[0].hop1_ids.append(2)
    nodes[0].hop1_su[int(nodes[2].su_type)] += 1
    assert assigner._can_add_hop1_connection(nodes, nodes[0], nodes[1]) is True
    assert calls == [(0, 1), (0, 1)]


def test_can_add_rejects_candidate_missing_exact_hop1_template(monkeypatch) -> None:
    assigner = Layer1Assigner(device="cpu")
    nodes = [_NodeV3(global_id=0, su_type=23), _NodeV3(global_id=1, su_type=24)]
    template_calls = []

    monkeypatch.setattr(assigner, "_edge_semantics_ok", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(assigner, "_is_su19_partition_compatible", lambda *_args: True)
    monkeypatch.setattr(assigner, "_violates_mode_specific_rule", lambda *_args: False)
    monkeypatch.setattr(assigner, "_violates_special_d3_terminal_rule", lambda *_args: False)
    monkeypatch.setattr(assigner, "_violates_terminal_neighbor_limit", lambda *_args: False)
    monkeypatch.setattr(assigner, "_violates_fixed_connector_special_end_rules", lambda *_args: False)
    monkeypatch.setattr(assigner, "_violates_forbidden_p_pair_connection", lambda *_args: False)
    monkeypatch.setattr(assigner, "_required_external_candidates_for_node", lambda *_args: [])
    monkeypatch.setattr(assigner, "_port_sets_for_node", lambda *_args: None)

    def _template_feasible(node, neighbor_su, require_exact=False):
        template_calls.append((int(node.global_id), int(neighbor_su), bool(require_exact)))
        return False

    monkeypatch.setattr(assigner, "_hop1_template_feasible", _template_feasible)

    assert assigner._can_add_hop1_connection_uncached(nodes, nodes[0], nodes[1]) is False
    assert template_calls == [(0, 24, False)]


def test_legacy_layer1_completion_paths_are_removed() -> None:
    for method_name in (
        "_complete_nodes_round_robin",
        "_complete_special_nodes_mrv",
        "_repair_exact_template_tails",
        "_complete_aromatic_hop1",
        "_complete_aliphatic_hetero_hop1",
        "_complete_unsaturated_saturated_end",
        "_complete_remaining_aliphatic_aromatic",
    ):
        assert not hasattr(Layer1Assigner, method_name)


def test_fixed_carbonyl_phase_only_assigns_su9_role_edge() -> None:
    assigner = Layer1Assigner(device="cpu")
    nodes = [
        _NodeV3(global_id=0, su_type=0, target_hop1_degree=1),
        _NodeV3(global_id=1, su_type=9, target_hop1_degree=3),
        _NodeV3(global_id=2, su_type=25, target_hop1_degree=4),
    ]

    assigner._assign_fixed_carbonyl(nodes)

    assert nodes[0].hop1_ids == [1]
    assert nodes[1].hop1_ids == [0]
    assert nodes[2].hop1_ids == []
    assert nodes[0].fixed_hop1_ids == set()
    assert nodes[1].fixed_hop1_ids == set()


def test_hop1_strict_swap_dry_run_restores_and_commit_returns_undo_state() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=13),
        _NodeV3(global_id=1, su_type=22),
        _NodeV3(global_id=2, su_type=24),
        _NodeV3(global_id=3, su_type=23),
    ]
    _connect_test_nodes(nodes, 0, 1)
    _connect_test_nodes(nodes, 2, 3)
    before = _topology_state(nodes)
    adjuster = _unconstrained_hop1_adjuster()
    nodes_by_su = defaultdict(list)
    for node in nodes:
        nodes_by_su[int(node.su_type)].append(int(node.global_id))

    probe = adjuster.execute_hop1_replacement(
        nodes,
        target_node_id=0,
        new_hop1_tuple=(24,),
        dry_run=True,
        nodes_by_su=nodes_by_su,
        return_undo_state=True,
    )

    assert probe["success"] is True
    assert "_undo_state" not in probe
    assert _topology_state(nodes) == before

    committed = adjuster.execute_hop1_replacement(
        nodes,
        target_node_id=0,
        new_hop1_tuple=(24,),
        nodes_by_su=nodes_by_su,
        return_undo_state=True,
    )

    assert committed["success"] is True
    assert committed["_undo_state"] == before
    assert [node.hop1_ids for node in nodes] == [[2], [3], [0], [1]]
    assert [Counter(node.hop1_su) for node in nodes] == [
        Counter({24: 1}),
        Counter({23: 1}),
        Counter({13: 1}),
        Counter({22: 1}),
    ]


def test_hop1_strict_swap_rolls_back_partial_multi_swap_failure() -> None:
    nodes = [
        _NodeV3(global_id=0, su_type=13),
        _NodeV3(global_id=1, su_type=22),
        _NodeV3(global_id=2, su_type=23),
        _NodeV3(global_id=3, su_type=24),
        _NodeV3(global_id=4, su_type=11),
        _NodeV3(global_id=5, su_type=25),
    ]
    _connect_test_nodes(nodes, 0, 1)
    _connect_test_nodes(nodes, 0, 2)
    _connect_test_nodes(nodes, 3, 4)
    before = _topology_state(nodes)
    adjuster = _unconstrained_hop1_adjuster()

    result = adjuster.execute_hop1_replacement(
        nodes,
        target_node_id=0,
        new_hop1_tuple=(24, 25),
        return_undo_state=True,
    )

    assert result["success"] is False
    assert result["error"] == "无法找到满足规则的换位方案"
    assert "_undo_state" not in result
    assert _topology_state(nodes) == before


def test_unsaturated_pairing_reserves_su15_for_su16() -> None:
    assigner = Layer1Assigner(device="cpu")
    nodes = [
        _NodeV3(global_id=0, su_type=16),
        _NodeV3(global_id=1, su_type=16),
        _NodeV3(global_id=2, su_type=15),
        _NodeV3(global_id=3, su_type=15),
    ]

    assigner._assign_unsaturated_pairs(nodes)

    assert {nodes[0].hop1_ids[0], nodes[1].hop1_ids[0]} == {2, 3}
    assert {nodes[2].hop1_ids[0], nodes[3].hop1_ids[0]} == {0, 1}
    assert all(node.fixed_hop1_ids == set(node.hop1_ids) for node in nodes)


def test_hop1_spectrum_regions_have_distinct_configurable_order() -> None:
    specs = Layer1Refiner._region_specs()

    assert [name for name, *_rest in specs] == [
        "aromatic",
        "carbonyl",
        "unsaturated",
        "aliphatic",
    ]
    priorities = {name: set(priority) for name, _lo, _hi, priority in specs}
    assert priorities["aromatic"].isdisjoint(priorities["unsaturated"])
