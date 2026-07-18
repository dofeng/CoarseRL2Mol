from __future__ import annotations

import inspect

import torch

from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer1.inverse_layer1 import Layer1Assigner
from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
from model.shared.inverse_common import _NodeV3
from model.inverse.layer4.inverse_layer4_block_c import (
    _tail_bundle_conservation_reason,
    _tail_h_compensation_at,
)


def test_layer1_public_api_has_single_cold_start_profile() -> None:
    pipeline_parameters = inspect.signature(InversePipelineV3.layer1_assign).parameters
    assigner_parameters = inspect.signature(Layer1Assigner.layer1_assign).parameters

    for obsolete in (
        "seed_nodes",
        "preserve_seed_node_ids",
        "layer1_repair_mode",
        "layer1_global_repair_max_attempts",
    ):
        assert obsolete not in pipeline_parameters
        assert obsolete not in assigner_parameters

    for required in (
        "profile",
        "local_swap_max_attempts",
        "initial_enable_spectral_refine",
        "initial_hop1_adjust_iterations",
        "initial_hop1_adjust_max_candidate_trials",
    ):
        assert required in pipeline_parameters
        assert required in assigner_parameters


def test_special_mrv_order_prefers_scarcity_then_gap_and_d3() -> None:
    rows = [
        {"candidate_count": 3, "degree_gap": 2, "target_degree": 2, "anchor_mode": "single", "node_id": 2},
        {"candidate_count": 2, "degree_gap": 1, "target_degree": 2, "anchor_mode": "single", "node_id": 3},
        {"candidate_count": 2, "degree_gap": 2, "target_degree": 3, "anchor_mode": "double", "node_id": 4},
    ]

    ordered = sorted(rows, key=Layer1Assigner._special_mrv_key)

    assert ordered[0]["node_id"] == 4
    assert ordered[-1]["node_id"] == 2


def test_remaining_completion_connects_terminal_su16_and_su18() -> None:
    assigner = Layer1Assigner(device="cpu")
    nodes = [
        _NodeV3(global_id=0, su_type=16),
        _NodeV3(global_id=1, su_type=15),
        _NodeV3(global_id=2, su_type=18),
        _NodeV3(global_id=3, su_type=17),
    ]

    assigner._complete_remaining_aliphatic_aromatic(nodes)

    assert nodes[0].hop1_ids == [1]
    assert nodes[1].hop1_ids == [0]
    assert nodes[2].hop1_ids == [3]
    assert nodes[3].hop1_ids == [2]


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


def test_tail_h_compensation_cycle_repeats_prescribed_order() -> None:
    assert [_tail_h_compensation_at(index) for index in range(4)] == [
        (23, 11),
        (23, 11),
        (22, 11),
        (23, 11),
    ]


def test_tail_bundle_audit_requires_total_and_fixed_element_conservation() -> None:
    adjuster = Layer4Adjuster(device=torch.device("cpu"))
    before = torch.zeros(33, dtype=torch.long)
    before[23] = 2

    valid = before.clone()
    valid[23] -= 1
    valid[11] += 1
    assert _tail_bundle_conservation_reason(adjuster, before, valid) == ""

    invalid_total = before.clone()
    invalid_total[11] += 1
    assert _tail_bundle_conservation_reason(adjuster, before, invalid_total) == "su_total_not_conserved"

    invalid_fixed_elements = before.clone()
    invalid_fixed_elements[23] -= 1
    invalid_fixed_elements[28] += 1
    assert _tail_bundle_conservation_reason(adjuster, before, invalid_fixed_elements) == "fixed_elements_not_conserved"
