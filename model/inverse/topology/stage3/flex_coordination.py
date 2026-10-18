"""Bounded, identity-aware aromatic candidates for flex-window coordination."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, islice
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch

from ....shared.aromatic_plan import (
    MAX_FLEX_WINDOW_WIDTH,
    exact_aromatic_cluster_plan,
    flex_count_window,
    staging_aromatic_counts,
    su12_transition_diagnostics,
)
from ..resources.profiles import MIN_STAGE3_AROMATIC_RESOURCE_COUNT
from ..stage3_models import HistogramConversion


def histogram_flex_window(H: torch.Tensor) -> Dict[str, Any]:
    counts = dict(enumerate(H.detach().cpu().long().tolist()))
    aromatic = exact_aromatic_cluster_plan(*staging_aromatic_counts(counts)[:2])
    return flex_count_window(aromatic["cluster_count"], counts.get(10, 0))


def flex_window_candidate_allowed(
    before: Mapping[str, Any], after: Mapping[str, Any], *, allow_intermediate: bool = False,
) -> bool:
    if not after.get("flex_window_ok", False):
        return False
    if after.get("flex_window_width_ok", False):
        return True
    return bool(
        allow_intermediate and before.get("flex_window_ok", False)
        and MAX_FLEX_WINDOW_WIDTH < int(after["flex_window_width"])
        < int(before.get("flex_window_width", 0))
    )


def available_su10_pairs(nodes: Sequence[Any]) -> List[Tuple[int, int]]:
    by_id = {int(node.global_id): node for node in nodes}
    su10_ids = {key for key, node in by_id.items() if int(node.su_type) == 10}
    neighbors = {
        key: set(map(int, getattr(by_id[key], "hop1_ids", ()) or ())) & su10_ids
        for key in su10_ids
    }
    return sorted(
        (left, right)
        for left in su10_ids
        for right in neighbors[left]
        if left < right and neighbors[left] == {right} and neighbors[right] == {left}
        and not any(
            getattr(by_id[key], "stage3_resource_claimed", False)
            or getattr(by_id[key], "stage3_locked_neighbor_ids", ())
            or any(
                int(getattr(by_id.get(int(neighbor)), "su_type", -1))
                not in {*range(5, 14), 26, 30}
                for neighbor in getattr(by_id[key], "hop1_ids", ())
            )
            for key in (left, right)
        )
    )


def aromatic_window_candidates(
    H: torch.Tensor,
    pool: Mapping[int, Sequence[int]],
    pairs: Sequence[Tuple[int, int]],
    *,
    max_primary_adjustments: int,
    fixed_flex_count: int,
    allow_intermediate: bool = False,
) -> List[Dict[str, Any]]:
    """Enumerate pair retirement/creation combined with SU12/SU13 exchange."""
    budget = max(0, int(max_primary_adjustments))
    counts = {index: int(value) for index, value in enumerate(H.tolist())}
    windows: Dict[Tuple[int, int], Any] = {}
    candidates: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    try:
        before_window = histogram_flex_window(H)
    except ValueError:
        before_window = {}
    create_limit = min(budget // 2, (len(pool.get(12, ())) + len(pool.get(13, ()))) // 2)
    retire_limit = min(budget // 2, len(pairs))
    for pair_delta in range(-retire_limit, create_limit + 1):
        pair_nodes = 2 * abs(pair_delta)
        for su12_nodes in range(pair_nodes + 1):
            su13_nodes = pair_nodes - su12_nodes
            for exchange in range(-(budget - pair_nodes), budget - pair_nodes + 1):
                specs = []
                if pair_delta < 0:
                    specs.extend(((10, 12, su12_nodes), (10, 13, su13_nodes)))
                elif pair_delta > 0:
                    specs.extend(((12, 10, su12_nodes), (13, 10, su13_nodes)))
                if exchange > 0:
                    specs.append((13, 12, exchange))
                elif exchange < 0:
                    specs.append((12, 13, -exchange))
                specs = [spec for spec in specs if spec[2] > 0]
                if not specs:
                    continue
                used = Counter()
                trial = dict(counts)
                for source, target, count in specs:
                    used[source] += count
                    trial[source] = trial.get(source, 0) - count
                    trial[target] = trial.get(target, 0) + count
                if any(used[source] > len(pool.get(source, ())) for source in (12, 13)):
                    continue
                if any(
                    used[source] and trial.get(source, 0) < MIN_STAGE3_AROMATIC_RESOURCE_COUNT
                    for source in (12, 13)
                ):
                    continue
                if trial.get(12, 0) % 2:
                    continue
                after = H.detach().clone()
                for source, target, count in specs:
                    after[source] -= count
                    after[target] += count
                if not su12_transition_diagnostics(H, after)["generation_allowed"]:
                    continue
                bridgeheads, perimeter, _five = staging_aromatic_counts(trial)
                key = (bridgeheads, perimeter)
                if key not in windows:
                    try:
                        windows[key] = exact_aromatic_cluster_plan(*key)
                    except ValueError:
                        windows[key] = None
                aromatic = windows[key]
                if aromatic is None:
                    continue
                window = flex_count_window(aromatic["cluster_count"], trial.get(10, 0))
                if not flex_window_candidate_allowed(
                    before_window, window, allow_intermediate=allow_intermediate,
                ):
                    continue
                if fixed_flex_count > window["flex_upper"]:
                    continue
                count = sum(spec[2] for spec in specs)
                intermediate = not window["flex_window_width_ok"]
                rank = (int(intermediate), window["flex_window_width"] if intermediate else 0,
                        max(0, window["flex_lower"] - fixed_flex_count), count,
                        window["flex_window_width"], abs(pair_delta), tuple(specs))
                state = tuple(trial.get(su_type, 0) for su_type in (10, 12, 13))
                if state not in candidates or rank < candidates[state]["rank"]:
                    candidates[state] = {
                        "specs": specs, "H_after": after, "window": window,
                        "aromatic": aromatic, "pair_delta": pair_delta,
                        "primary_count": count, "rank": rank,
                    }
    return sorted(candidates.values(), key=lambda candidate: candidate["rank"])


def bind_aromatic_window_candidate(
    candidate: Mapping[str, Any],
    pool: Mapping[int, Sequence[int]],
    pairs: Sequence[Tuple[int, int]],
    *,
    variant_limit: int = 1,
) -> Iterator[List[HistogramConversion]]:
    specs = list(candidate["specs"])
    required = Counter()
    for source, _target, count in specs:
        required[source] += count
    sources = sorted(required)

    def selections(index: int, chosen: Dict[int, List[int]]) -> Iterator[Dict[int, List[int]]]:
        if index == len(sources):
            yield chosen
            return
        source = sources[index]
        if source == 10:
            options = (
                [node_id for pair in selection for node_id in pair]
                for selection in combinations(pairs, required[source] // 2)
            )
        else:
            options = combinations(sorted(pool.get(source, ())), required[source])
        for selection in options:
            yield from selections(index + 1, {**chosen, source: list(selection)})

    for chosen in islice(selections(0, {}), max(1, int(variant_limit))):
        offsets = Counter()
        operations = []
        for source, target, count in specs:
            offset = offsets[source]
            operations.append(HistogramConversion(
                from_su=source, to_su=target, count=count,
                node_ids=chosen[source][offset:offset + count],
                reason="flex_window_coordination",
            ))
            offsets[source] += count
        yield operations
