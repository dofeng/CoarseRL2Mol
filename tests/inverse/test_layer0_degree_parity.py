from __future__ import annotations

import torch

from model.inverse.layer0.inverse_layer0 import Layer0Estimator
from model.shared.coarse_graph import E_SU


def test_layer0_adjusts_one_special_degree_when_target_sum_is_odd() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[0] = 1
    histogram[19] = 1
    degree_meta = {19: {1: 1, 2: 0, 3: 0}}
    partition_meta = {
        19: {
            "ether": {1: 1, 2: 0, 3: 0},
            "thio": {1: 0, 2: 0, 3: 0},
        }
    }
    anchor_mode_meta = {
        19: {
            "ether_single": {1: 1, 2: 0, 3: 0},
            "ether_double": {2: 0, 3: 0},
            "thio_single": {1: 0, 2: 0, 3: 0},
            "thio_double": {2: 0, 3: 0},
        }
    }

    result = estimator._enforce_even_target_degree_sum(
        histogram,
        degree_meta,
        partition_meta,
        anchor_mode_meta,
    )

    assert result == {
        "adjusted": True,
        "reason": "adjusted_one_special_degree",
        "target_degree_sum_before": 3,
        "target_degree_sum_after": 4,
        "transition": {
            "su_type": 19,
            "bucket": "ether_single",
            "partition": "ether",
            "source_degree": 1,
            "target_degree": 2,
        },
    }
    assert degree_meta[19] == {1: 0, 2: 1, 3: 0}
    assert partition_meta[19]["ether"] == {1: 0, 2: 1, 3: 0}
    assert anchor_mode_meta[19]["ether_single"] == {1: 0, 2: 1, 3: 0}


def test_layer0_preserves_special_degree_counts_when_target_sum_is_even() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[19] = 2
    degree_meta = {19: {1: 0, 2: 0, 3: 2}}
    partition_meta = {19: {"ether": {1: 0, 2: 0, 3: 2}}}
    anchor_mode_meta = {19: {"ether_single": {1: 0, 2: 0, 3: 2}}}

    result = estimator._enforce_even_target_degree_sum(
        histogram,
        degree_meta,
        partition_meta,
        anchor_mode_meta,
    )

    assert result["adjusted"] is False
    assert result["reason"] == "already_even"
    assert degree_meta[19] == {1: 0, 2: 0, 3: 2}


def test_layer0_repairs_odd_degree_sum_with_hydrogen_directed_aromatic_conversion() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[13] = 2
    elements = torch.tensor([2, 1, 0, 0, 0, 0], dtype=torch.float32)

    result_histogram, result = estimator._enforce_even_histogram_degree_sum(
        histogram,
        elements,
        {
            "target_degree_sum_before": 5,
            "target_degree_sum_after": 5,
        },
    )

    assert result["adjusted"] is True
    assert result["reason"] == "converted_histogram_for_degree_parity"
    assert result["target_degree_sum_after"] == 6
    assert result["transition"] == {
        "source_su": 13,
        "target_su": 12,
        "source_degree": 2,
        "target_degree": 3,
    }
    assert int(result_histogram[13].item()) == 1
    assert int(result_histogram[12].item()) == 1


def test_layer0_defers_to_special_degree_fallback_without_hydrogen_residual() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[13] = 1
    elements = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32)

    result_histogram, result = estimator._enforce_even_histogram_degree_sum(
        histogram,
        elements,
        {"target_degree_sum_before": 3},
    )

    assert result["adjusted"] is False
    assert result["reason"] == "no_hydrogen_directed_histogram_transition"
    assert torch.equal(result_histogram, histogram)


def test_layer0_thio_su19_initial_modes_only_use_single_d2_or_d3() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[19] = 6

    estimator._allocate_special_degree_meta(
        histogram,
        o_base_19=0,
        s_reserved_19=6,
        S_target=None,
    )

    thio = estimator.special_anchor_mode_meta[19]
    assert thio["thio_single"][1] == 0
    assert thio["thio_double"][2] == 0
    assert thio["thio_double"][3] == 0
    assert sum(thio["thio_single"].values()) == 6
    assert thio["thio_single"][2] + thio["thio_single"][3] == 6


def test_layer0_thio_su19_parity_adjustment_keeps_single_mode() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[19] = 1
    degree_meta = {19: {1: 0, 2: 0, 3: 1}}
    partition_meta = {
        19: {
            "ether": {1: 0, 2: 0, 3: 0},
            "thio": {1: 0, 2: 0, 3: 1},
        }
    }
    anchor_mode_meta = {
        19: {
            "ether_single": {1: 0, 2: 0, 3: 0},
            "ether_double": {2: 0, 3: 0},
            "thio_single": {1: 0, 2: 0, 3: 1},
            "thio_double": {2: 0, 3: 0},
        }
    }

    result = estimator._enforce_even_target_degree_sum(
        histogram,
        degree_meta,
        partition_meta,
        anchor_mode_meta,
    )

    assert result["adjusted"] is True
    assert result["transition"]["bucket"] == "thio_single"
    assert degree_meta[19] == {1: 0, 2: 1, 3: 0}
    assert partition_meta[19]["thio"] == {1: 0, 2: 1, 3: 0}
    assert anchor_mode_meta[19]["thio_single"] == {1: 0, 2: 1, 3: 0}
    assert anchor_mode_meta[19]["thio_double"] == {2: 0, 3: 0}


def test_layer0_ether_su19_mode_pool_remains_unchanged() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[19] = 6

    estimator._allocate_special_degree_meta(
        histogram,
        o_base_19=6,
        s_reserved_19=0,
        S_target=torch.ones(2400),
    )

    ether = estimator.special_anchor_mode_meta[19]
    assert sum(ether["ether_single"].values()) + sum(ether["ether_double"].values()) == 6
    assert sum(ether["ether_single"].values()) > 0
    assert sum(ether["ether_double"].values()) > 0


def test_layer0_thio_connection_budget_uses_single_mode_fixed_edges() -> None:
    estimator = Layer0Estimator(s2n_model=None, E_SU_tensor=E_SU, device="cpu")
    histogram = torch.zeros(33, dtype=torch.long)
    histogram[31] = 3

    _histogram_after, thio_meta = estimator._correct_thioether_connection(
        histogram,
        o_base_19=0,
        S_target=None,
    )

    assert thio_meta["s_reserved_19"] == thio_meta["s_fixed_edges_19"]
