from __future__ import annotations

import numpy as np
import torch

from model.inverse.layer4.inverse_layer4_block_b import adjust_special_anchor_count_impl
from model.inverse.special_anchor_evidence import (
    SU19_DOUBLE_EVIDENCE_WINDOWS,
    SU19_DOUBLE_SUPPORT_WEIGHT,
    split_su19_double_area,
    weighted_su19_double_area,
)

from .topology.helpers import connect, make_node


def _band_area(band_lo: float, band_hi: float):
    def area(lo: float, hi: float) -> float:
        return max(0.0, min(float(hi), float(band_hi)) - max(float(lo), float(band_lo)))

    return area


def test_su19_double_support_windows_exclude_global_extreme_tails() -> None:
    low_d2 = weighted_su19_double_area(_band_area(63.5, 65.5), "double_d2")
    high_d3 = weighted_su19_double_area(_band_area(105.2, 106.2), "double_d3")

    assert low_d2 == 0.0
    assert high_d3 == 0.0
    assert SU19_DOUBLE_SUPPORT_WEIGHT == 0.15


def test_su19_double_cores_are_disjoint_at_stats_midpoint() -> None:
    d2_core = SU19_DOUBLE_EVIDENCE_WINDOWS["double_d2"]["core"]
    d3_core = SU19_DOUBLE_EVIDENCE_WINDOWS["double_d3"]["core"]

    assert d2_core == (83.21875, 94.125)
    assert d3_core == (95.96875, 101.8125)
    assert float(d2_core[1]) < float(d3_core[0])


def test_su19_double_union_counts_each_exclusive_core_once() -> None:
    bands = [(88.0, 89.0), (98.0, 99.0)]

    def area(lo: float, hi: float) -> float:
        return sum(
            max(0.0, min(float(hi), band_hi) - max(float(lo), band_lo))
            for band_lo, band_hi in bands
        )

    split = split_su19_double_area(area)

    assert abs(sum(split.values()) - 2.0) < 1e-9
    assert split["double_d2"] > 0.0
    assert split["double_d3"] > 0.0


class _CountAdjuster:
    def __init__(self, nodes) -> None:
        self.nodes = list(nodes)
        self.fixed_partition_meta = {}
        self.layer0_estimator = None
        self._su_special_anchor_mode_stats_cache = {}

    def _current_neighbor_types(self, node, nodes):
        nodes_by_id = {int(item.global_id): item for item in nodes}
        return [int(nodes_by_id[node_id].su_type) for node_id in node.hop1_ids]

    def _get_special_anchor_mode_meta(self, H):
        del H
        return {
            19: {
                "ether_single": {1: 0, 2: 0, 3: 2},
                "ether_double": {2: 0, 3: 0},
                "thio_single": {1: 0, 2: 0, 3: 0},
                "thio_double": {2: 0, 3: 0},
            }
        }

    def _get_special_degree_meta(self, H):
        del H
        return {19: {1: 0, 2: 0, 3: 2}}

    def _get_su_common_window(self, su_type, fallback_mu, min_half_width):
        del fallback_mu, min_half_width
        return (20.0, 35.0, 29.0) if int(su_type) == 23 else (35.0, 45.0, 40.0)

    def _window_stats(self, ppm, diff, lo, hi):
        del ppm, diff
        if float(lo) >= 91.0:
            return {"pos": 12.0, "neg": 0.0, "abs": 12.0}
        if float(lo) >= 67.0 and float(hi) > 100.0:
            return {"pos": 14.0, "neg": 0.0, "abs": 14.0}
        if float(lo) >= 70.0:
            return {"pos": 0.0, "neg": 8.0, "abs": 8.0}
        if float(lo) >= 35.0:
            return {"pos": 4.0, "neg": 0.0, "abs": 4.0}
        return {"pos": 0.0, "neg": 0.0, "abs": 0.0}


def test_block_b_can_generate_double19d3_from_two_single19d3_nodes() -> None:
    left = make_node(1, 19, target_hop1_degree=3, special_anchor_mode="single", special_anchor_partition="ether")
    right = make_node(2, 19, target_hop1_degree=3, special_anchor_mode="single", special_anchor_partition="ether")
    left_28 = make_node(3, 28)
    right_28 = make_node(4, 28)
    connect(left, left_28, fixed=True)
    connect(right, right_28, fixed=True)
    nodes = [left, right, left_28, right_28]
    adjuster = _CountAdjuster(nodes)
    H = torch.zeros(33, dtype=torch.long)
    H[19] = 2
    H[28] = 2
    H[29] = 1

    H_after, moves, meta = adjust_special_anchor_count_impl(
        adjuster,
        H,
        ppm=np.asarray([95.0]),
        diff=np.asarray([1.0]),
        su_type=19,
        nodes=nodes,
        anchor_partition="ether",
        max_moves=1,
        peak_rel_threshold=0.0,
    )

    assert int(H_after[19].item()) == 1
    assert int(H_after[24].item()) == 1
    assert moves[0]["op"].startswith("forward_sd3_sd3_to_dd3[forward]")
    assert int(meta["counts_after"]["double_d3"]) == 1


class _FourSingleD3Adjuster(_CountAdjuster):
    def _get_special_anchor_mode_meta(self, H):
        del H
        return {
            19: {
                "ether_single": {1: 0, 2: 0, 3: 4},
                "ether_double": {2: 0, 3: 0},
                "thio_single": {1: 0, 2: 0, 3: 0},
                "thio_double": {2: 0, 3: 0},
            }
        }

    def _get_special_degree_meta(self, H):
        del H
        return {19: {1: 0, 2: 0, 3: 4}}


def test_block_b_adds_at_most_one_double19_before_rebuild() -> None:
    anchors = [
        make_node(
            index,
            19,
            target_hop1_degree=3,
            special_anchor_mode="single",
            special_anchor_partition="ether",
        )
        for index in range(1, 5)
    ]
    terminals = [make_node(index, 28) for index in range(10, 14)]
    for anchor, terminal in zip(anchors, terminals):
        connect(anchor, terminal, fixed=True)
    nodes = anchors + terminals
    adjuster = _FourSingleD3Adjuster(nodes)
    H = torch.zeros(33, dtype=torch.long)
    H[19] = 4
    H[28] = 4
    H[29] = 1

    H_after, moves, meta = adjust_special_anchor_count_impl(
        adjuster,
        H,
        ppm=np.asarray([98.0]),
        diff=np.asarray([1.0]),
        su_type=19,
        nodes=nodes,
        anchor_partition="ether",
        max_moves=4,
        peak_rel_threshold=0.0,
    )

    assert int(H_after[19].item()) == 3
    assert sum("[forward]" in move["op"] for move in moves) == 1
    assert int(meta["counts_after"]["double_d3"]) == 1


class _D2CountAdjuster(_CountAdjuster):
    def _get_special_anchor_mode_meta(self, H):
        del H
        return {
            19: {
                "ether_single": {1: 0, 2: 2, 3: 0},
                "ether_double": {2: 0, 3: 0},
                "thio_single": {1: 0, 2: 0, 3: 0},
                "thio_double": {2: 0, 3: 0},
            }
        }

    def _get_special_degree_meta(self, H):
        del H
        return {19: {1: 0, 2: 2, 3: 0}}

    def _window_stats(self, ppm, diff, lo, hi):
        del ppm, diff
        if 83.0 <= float(lo) <= 84.0 and 94.0 <= float(hi) <= 95.5:
            return {"pos": 20.0, "neg": 0.0, "abs": 20.0}
        if 59.0 <= float(lo) <= 61.0 and float(hi) <= 71.0:
            return {"pos": 0.0, "neg": 8.0, "abs": 8.0}
        if float(hi) <= 35.0:
            return {"pos": 10.0, "neg": 0.0, "abs": 10.0}
        return {"pos": 0.0, "neg": 0.0, "abs": 0.0}


def test_block_b_can_generate_double19d2_from_two_single19d2_nodes() -> None:
    left = make_node(
        11,
        19,
        target_hop1_degree=2,
        special_anchor_mode="single",
        special_anchor_partition="ether",
        target_fixed_anchor_count=1,
    )
    right = make_node(
        12,
        19,
        target_hop1_degree=2,
        special_anchor_mode="single",
        special_anchor_partition="ether",
        target_fixed_anchor_count=1,
    )
    connector_29 = make_node(29, 29)
    connect(left, connector_29, fixed=True)
    connect(right, connector_29, fixed=True)
    nodes = [left, right, connector_29]
    adjuster = _D2CountAdjuster(nodes)
    H = torch.zeros(33, dtype=torch.long)
    H[19] = 2
    H[29] = 1

    H_after, moves, meta = adjust_special_anchor_count_impl(
        adjuster,
        H,
        ppm=np.asarray([80.0]),
        diff=np.asarray([1.0]),
        su_type=19,
        nodes=nodes,
        anchor_partition="ether",
        max_moves=1,
        peak_rel_threshold=0.0,
    )

    assert int(H_after[19].item()) == 1
    assert int(H_after[23].item()) == 1
    assert int(H_after[29].item()) == 1
    assert int(H_after.sum().item()) == int(H.sum().item())
    assert moves[0]["op"].startswith("forward_sd2_sd2_to_dd2[forward]")
    assert int(meta["counts_after"]["double_d2"]) == 1
