"""Shared resource accounting for explicit aromatic-cluster plans.

The inverse pipeline is the authority for cluster selection.  RL_MTCS may
materialize the selected templates, but it must not independently choose a
different multiset of aromatic clusters.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

MAX_IDLE_AROMATIC_NODES = 5
MAX_FLEX_WINDOW_WIDTH = 6


def flex_count_window(cluster_count: int, su10_count: int) -> Dict[str, Any]:
    """One quantity rule for inverse planning and conversion scoring."""
    x, n10 = int(cluster_count), int(su10_count)
    y = n10 // 2
    lower, upper = x - y + 1, 4 * x // 5
    width = upper - lower
    return {
        "X": x, "Y": y, "Z": x - y,
        "flex_lower": lower, "flex_upper": upper,
        "flex_window_width": width,
        "flex_window_width_ok": 0 <= width <= MAX_FLEX_WINDOW_WIDTH,
        "flex_window_ok": n10 >= 0 and n10 % 2 == 0 and lower <= upper,
    }


AROMATIC_CLUSTER_RING_COUNTS: Dict[str, int] = {
    "benzene": 1,
    "naphthalene": 2,
    "phenanthrene": 3,
    "anthracene": 3,
    "tetracene": 4,
    "pyrene": 4,
    "benzo_pyrene": 5,
    "perylene": 5,
    "chrysene": 4,
    "triphenylene": 4,
    "coronene": 7,
}

AROMATIC_CLUSTER_COST: Dict[str, Tuple[int, int]] = {
    "coronene": (12, 12),
    "pyrene": (6, 10),
    "benzo_pyrene": (8, 12),
    "perylene": (8, 12),
    "chrysene": (6, 12),
    "triphenylene": (6, 12),
    "tetracene": (6, 12),
    "phenanthrene": (4, 10),
    "anthracene": (4, 10),
    "naphthalene": (2, 8),
    "benzene": (0, 6),
}

_KIND_ORDER: Tuple[str, ...] = (
    "coronene",
    "benzo_pyrene",
    "perylene",
    "pyrene",
    "chrysene",
    "triphenylene",
    "tetracene",
    "phenanthrene",
    "anthracene",
    "naphthalene",
    "benzene",
)

# SU12 is an aromatic bridgehead.  Its conversion budget is defined against
# the ordinary aromatic non-bridgehead pool only; staged SU26/SU30 sites are
# deliberately excluded from this ratio.
SU12_CAP_TYPES: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 13)


def _su_count(su_counts: Any, su_type: int) -> int:
    """Read one SU count from mappings, tensors, or integer sequences."""
    if isinstance(su_counts, Mapping):
        value = su_counts.get(int(su_type), su_counts.get(str(int(su_type)), 0))
    else:
        try:
            value = su_counts[int(su_type)]
        except (IndexError, KeyError, TypeError):
            value = 0
    try:
        if hasattr(value, "item"):
            value = value.item()
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def su12_generation_cap(su_counts: Any) -> Dict[str, int | bool]:
    """Return the dynamic SU12 upper bound for a histogram state.

    The requested 4:5 ratio is implemented with integer arithmetic.  SU12
    remains even because the aromatic bridgehead planner consumes paired
    bridgeheads, so an odd floor is rounded down to the previous even value.
    """
    non_bridgehead_count = int(
        sum(_su_count(su_counts, su_type) for su_type in SU12_CAP_TYPES)
    )
    raw_cap = int(4 * non_bridgehead_count // 5)
    even_cap = int(raw_cap - (raw_cap % 2))
    su12_count = int(_su_count(su_counts, 12))
    return {
        "non_bridgehead_count": non_bridgehead_count,
        "raw_cap": raw_cap,
        "even_cap": even_cap,
        "su12_count": su12_count,
        "headroom": int(even_cap - su12_count),
        "within_limit": bool(su12_count <= even_cap),
    }


def su12_transition_diagnostics(
    before: Any,
    after: Any,
) -> Dict[str, int | bool]:
    """Check the *final* SU12/ordinary-aromatic ratio for a transition.

    The cap is a state constraint, not merely a limit on positive SU12
    deltas.  A transition which consumes SU13/SU11 can lower the denominator
    and make an otherwise unchanged SU12 population invalid.  Such a
    transition must be rejected unless the input was already over the cap and
    the transition strictly reduces the overage (the recovery path used by
    ``SU12 -> SU13``).
    """
    before_info = su12_generation_cap(before)
    after_info = su12_generation_cap(after)
    generated = int(after_info["su12_count"]) - int(before_info["su12_count"])
    before_excess = max(
        0,
        int(before_info["su12_count"]) - int(before_info["even_cap"]),
    )
    after_excess = max(
        0,
        int(after_info["su12_count"]) - int(after_info["even_cap"]),
    )
    recovering_existing_overage = bool(
        before_excess > 0 and after_excess < before_excess
    )
    allowed = bool(after_excess == 0 or recovering_existing_overage)
    return {
        **{f"before_{key}": value for key, value in before_info.items()},
        **{f"after_{key}": value for key, value in after_info.items()},
        "generated_su12": int(generated),
        "before_excess": int(before_excess),
        "after_excess": int(after_excess),
        "recovering_existing_overage": bool(recovering_existing_overage),
        "generation_allowed": bool(allowed),
    }


def staging_aromatic_counts(su_counts: Mapping[int, int]) -> Tuple[int, int, int]:
    """Return bridgehead, non-bridgehead, and five-ring staging counts.

    Every final SU26/SU30 occupies one SU13 site during assembly.  One third
    of SU26 (integer floor) and every SU30 require one additional temporary
    SU13 site because final substitution collapses a 13-13 pair.
    """

    counts = Counter({int(key): int(value) for key, value in dict(su_counts).items()})
    n26 = int(counts.get(26, 0))
    n30 = int(counts.get(30, 0))
    five_member_count = int(n26 // 3 + n30)
    bridgeheads = int(counts.get(12, 0))
    ordinary = int(sum(counts.get(su, 0) for su in (5, 6, 7, 8, 9, 10, 11, 13)))
    non_bridgeheads = int(ordinary + n26 + n30 + five_member_count)
    return bridgeheads, non_bridgeheads, five_member_count


def _representative_kinds(
    preferred_kinds: Sequence[str],
) -> Tuple[str, ...]:
    preferred = Counter(str(value) for value in list(preferred_kinds or []))
    by_cost: Dict[Tuple[int, int], List[str]] = {}
    for kind in _KIND_ORDER:
        by_cost.setdefault(AROMATIC_CLUSTER_COST[kind], []).append(kind)
    representatives: List[str] = []
    for cost in sorted(by_cost, key=lambda value: (-value[0], -value[1])):
        choices = by_cost[cost]
        representatives.append(
            min(
                choices,
                key=lambda kind: (
                    -int(preferred.get(kind, 0)),
                    _KIND_ORDER.index(kind),
                ),
            )
        )
    return tuple(representatives)


def exact_aromatic_cluster_plan(
    bridgehead_count: int,
    non_bridgehead_count: int,
    *,
    preferred_kinds: Iterable[str] = (),
    max_idle_aromatic: int = MAX_IDLE_AROMATIC_NODES,
) -> Dict[str, object]:
    """Cover bridgeheads with complete templates and bounded idle perimeter nodes.

    Cluster costs are exact for SU12 bridgeheads.  A small non-bridgehead
    residual, up to five sites, remains outside the assembled graph.
    Six non-bridgehead sites form a complete benzene template, so a six-site
    residual is absorbed by the dynamic-programming plan. Larger residuals
    indicate that the cluster model cannot
    represent the current histogram and are rejected.
    """

    target_12 = int(bridgehead_count)
    available_13 = int(non_bridgehead_count)
    if target_12 < 0 or available_13 < 0:
        raise ValueError("aromatic staging counts must be non-negative")
    if target_12 % 2:
        raise ValueError(
            f"aromatic bridgehead count must be even, got {target_12}"
        )

    kinds = _representative_kinds(tuple(preferred_kinds or ()))
    # One shortest multiset per resource state is sufficient.  Selecting the
    # largest reachable SU13 total afterwards gives the smallest explicit
    # idle remainder; shortest paths favor compact fused templates.
    states: Dict[Tuple[int, int], Tuple[str, ...]] = {(0, 0): ()}
    for used_12 in range(target_12 + 1):
        for used_13 in range(available_13 + 1):
            current = states.get((used_12, used_13))
            if current is None:
                continue
            for kind in kinds:
                cost_12, cost_13 = AROMATIC_CLUSTER_COST[kind]
                next_state = (used_12 + cost_12, used_13 + cost_13)
                if next_state[0] > target_12 or next_state[1] > available_13:
                    continue
                candidate = current + (kind,)
                existing = states.get(next_state)
                if existing is None or (len(candidate), candidate) < (
                    len(existing), existing
                ):
                    states[next_state] = candidate

    candidates = [
        (used_13, cluster_kinds)
        for (used_12, used_13), cluster_kinds in states.items()
        if used_12 == target_12
    ]
    if not candidates:
        raise ValueError(
            "no aromatic cluster plan covers all SU12 bridgeheads: "
            f"SU12={target_12}, staging_non_bridgehead={available_13}"
        )
    used_13, selected = max(
        candidates,
        key=lambda item: (item[0], -len(item[1])),
    )
    idle_count = int(available_13 - used_13)
    if idle_count > int(max_idle_aromatic):
        raise ValueError(
            "aromatic cluster plan leaves too many non-bridgehead sites: "
            f"SU12={target_12}, available={available_13}, used={used_13}, "
            f"idle={idle_count}"
        )
    return {
        "kinds": list(selected),
        "bridgehead_count": target_12,
        "non_bridgehead_count": available_13,
        "used_bridgehead_count": target_12,
        "used_non_bridgehead_count": int(used_13),
        "idle_aromatic_count": idle_count,
        "cluster_count": int(len(selected)),
        "ring_count": int(sum(AROMATIC_CLUSTER_RING_COUNTS[kind] for kind in selected)),
        "complete": True,
    }


__all__ = [
    "MAX_IDLE_AROMATIC_NODES",
    "MAX_FLEX_WINDOW_WIDTH",
    "flex_count_window",
    "AROMATIC_CLUSTER_COST",
    "AROMATIC_CLUSTER_RING_COUNTS",
    "SU12_CAP_TYPES",
    "exact_aromatic_cluster_plan",
    "staging_aromatic_counts",
    "su12_generation_cap",
    "su12_transition_diagnostics",
]
