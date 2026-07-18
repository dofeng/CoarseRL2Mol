from __future__ import annotations

from typing import Callable, Dict, Tuple


SU19_DOUBLE_SUPPORT_WEIGHT = 0.35
SU19_DOUBLE_EVIDENCE_WINDOWS: Dict[str, Dict[str, Tuple[float, float] | float]] = {
    "double_d2": {
        "core": (66.25, 94.125),
        "support": (63.375, 98.25),
        "mu": 94.125,
    },
    "double_d3": {
        "core": (91.375, 101.8125),
        "support": (67.2572, 106.4375),
        "mu": 97.8125,
    },
}


def weighted_su19_double_area(
    area_fn: Callable[[float, float], float],
    mode_name: str,
) -> float:
    spec = dict(SU19_DOUBLE_EVIDENCE_WINDOWS[str(mode_name)])
    core_lo, core_hi = spec["core"]
    support_lo, support_hi = spec["support"]
    core_area = max(0.0, float(area_fn(float(core_lo), float(core_hi))))
    support_area = max(0.0, float(area_fn(float(support_lo), float(support_hi))))
    support_only_area = max(0.0, float(support_area) - float(core_area))
    return float(core_area + SU19_DOUBLE_SUPPORT_WEIGHT * support_only_area)


def split_su19_double_area(
    area_fn: Callable[[float, float], float],
) -> Dict[str, float]:
    raw = {
        mode_name: weighted_su19_double_area(area_fn, mode_name)
        for mode_name in SU19_DOUBLE_EVIDENCE_WINDOWS
    }
    raw_total = float(sum(raw.values()))
    if raw_total <= 1e-12:
        return {mode_name: 0.0 for mode_name in raw}
    core_lo = min(float(spec["core"][0]) for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values())
    core_hi = max(float(spec["core"][1]) for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values())
    support_lo = min(float(spec["support"][0]) for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values())
    support_hi = max(float(spec["support"][1]) for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values())
    core_union_area = max(0.0, float(area_fn(core_lo, core_hi)))
    support_union_area = max(0.0, float(area_fn(support_lo, support_hi)))
    combined_total = float(
        core_union_area
        + SU19_DOUBLE_SUPPORT_WEIGHT * max(0.0, support_union_area - core_union_area)
    )
    return {
        mode_name: float(combined_total * value / raw_total)
        for mode_name, value in raw.items()
    }
