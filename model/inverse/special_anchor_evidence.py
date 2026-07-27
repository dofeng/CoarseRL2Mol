from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, Tuple, Union


SU19_DOUBLE_SUPPORT_WEIGHT = 0.15
_SU19_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "z_library"
    / "su_special_anchor_mode_nmr_range_filtered.csv"
)
_SU19_REQUIRED_FIELDS = (
    "mu_median",
    "mu_common_min",
    "mu_common_max",
    "mu_q05",
    "mu_q95",
)
_DoubleWindowValue = Union[Tuple[float, float], float, str]


def _load_su19_mode_stats() -> Dict[str, Dict[str, float]]:
    required_modes = {"single_d3", "double_d2", "double_d3"}
    stats: Dict[str, Dict[str, float]] = {}
    try:
        with _SU19_STATS_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("center_su_idx", -1)) != 19:
                    continue
                mode = str(row.get("anchor_mode", "") or "")
                if mode not in required_modes:
                    continue
                stats[mode] = {
                    key: float(row[key])
                    for key in _SU19_REQUIRED_FIELDS
                }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"failed to load SU19 NMR statistics from {_SU19_STATS_PATH}: {exc}"
        ) from exc
    missing_modes = sorted(required_modes - set(stats))
    if missing_modes:
        raise RuntimeError(
            f"missing SU19 anchor modes in {_SU19_STATS_PATH}: {missing_modes}"
        )
    return stats


def _build_su19_double_windows() -> Dict[str, Dict[str, _DoubleWindowValue]]:
    stats = _load_su19_mode_stats()
    single_d3 = stats["single_d3"]
    double_d2 = stats["double_d2"]
    double_d3 = stats["double_d3"]

    d2_lower_boundary = 0.5 * (
        float(single_d3["mu_median"]) + float(double_d2["mu_median"])
    )
    double_boundary = 0.5 * (
        float(double_d2["mu_median"]) + float(double_d3["mu_median"])
    )
    d2_core = (
        max(float(double_d2["mu_q05"]), float(d2_lower_boundary)),
        min(float(double_d2["mu_q95"]), float(double_boundary)),
    )
    d3_core = (
        max(float(double_d3["mu_q05"]), float(double_boundary)),
        float(double_d3["mu_q95"]),
    )
    return {
        "double_d2": {
            "core": d2_core,
            "support": (
                max(float(double_d2["mu_common_min"]), float(single_d3["mu_q95"])),
                min(float(double_d2["mu_common_max"]), float(double_boundary)),
            ),
            "mu": float(double_d2["mu_median"]),
            "source": "stats_exclusive_core",
        },
        "double_d3": {
            "core": d3_core,
            "support": (
                max(float(double_d3["mu_common_min"]), float(double_d2["mu_q95"])),
                float(double_d3["mu_common_max"]),
            ),
            "mu": float(double_d3["mu_median"]),
            "source": "stats_exclusive_core",
        },
    }


SU19_DOUBLE_EVIDENCE_WINDOWS: Dict[str, Dict[str, _DoubleWindowValue]] = (
    _build_su19_double_windows()
)


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

    def _union_area(intervals: Tuple[Tuple[float, float], ...]) -> float:
        merged: list[list[float]] = []
        for lo, hi in sorted(intervals):
            if not merged or float(lo) > float(merged[-1][1]):
                merged.append([float(lo), float(hi)])
            else:
                merged[-1][1] = max(float(merged[-1][1]), float(hi))
        return float(sum(max(0.0, float(area_fn(lo, hi))) for lo, hi in merged))

    core_union_area = _union_area(tuple(
        (float(spec["core"][0]), float(spec["core"][1]))
        for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values()
    ))
    support_union_area = _union_area(tuple(
        (float(spec["support"][0]), float(spec["support"][1]))
        for spec in SU19_DOUBLE_EVIDENCE_WINDOWS.values()
    ))
    combined_total = float(
        core_union_area
        + SU19_DOUBLE_SUPPORT_WEIGHT * max(0.0, support_union_area - core_union_area)
    )
    return {
        mode_name: float(combined_total * value / raw_total)
        for mode_name, value in raw.items()
    }
