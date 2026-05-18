import math
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ...shared.inverse_common import SPECIAL_D3_TERMINAL_NEIGHBORS, violates_special_d3_terminal_limit


AROMATIC_ANCHOR_WINDOWS: Dict[int, Tuple[float, float, float]] = {
    5: (145.0, 165.0, 155.0),
    6: (135.0, 155.0, 145.0),
    7: (145.0, 160.0, 152.5),
    8: (126.0, 135.0, 130.5),
}


MANUAL_SPECIAL_ANCHOR_MODE_WINDOWS: Dict[int, Dict[str, Tuple[float, float, float]]] = {
    19: {
        'single_d1': (50.0, 60.0, 55.0),
        'single_d2': (60.0, 70.0, 65.0),
        'single_d3': (70.0, 90.0, 80.0),
        'double_d2': (90.0, 100.0, 95.0),
        'double_d3': (90.0, 100.0, 97.5),
    },
    20: {
        'single_d1': (38.0, 45.0, 41.5),
        'single_d2': (45.0, 52.0, 48.5),
        'single_d3': (52.0, 65.0, 58.5),
        'double_d2': (40.0, 50.0, 45.0),
        'double_d3': (50.0, 60.0, 55.0),
    },
    21: {
        'single_d2': (35.0, 42.0, 38.5),
        'single_d3': (55.0, 64.0, 59.5),
    },
}


def _window_score(adjuster: Any,
                  ppm_arr: np.ndarray,
                  diff_arr: np.ndarray,
                  lo: float,
                  hi: float,
                  mu: Optional[float] = None) -> Dict[str, float]:
    stats = adjuster._window_stats(ppm_arr, diff_arr, float(lo), float(hi))
    pos = float(stats.get('pos', 0.0))
    neg_abs = float(stats.get('neg', 0.0))
    return {
        'lo': float(lo),
        'hi': float(hi),
        'mu': float(mu if mu is not None else 0.5 * (float(lo) + float(hi))),
        'pos': float(pos),
        'neg': float(-neg_abs),
        'neg_abs': float(neg_abs),
        'score': float(pos - neg_abs),
        'abs': float(stats.get('abs', 0.0)),
    }


def _aromatic_anchor_score(adjuster: Any,
                           ppm_arr: np.ndarray,
                           diff_arr: np.ndarray,
                           su_type: int,
                           fallback_mu: Optional[float] = None) -> Dict[str, float]:
    su_i = int(su_type)
    if su_i in AROMATIC_ANCHOR_WINDOWS:
        lo, hi, mu = AROMATIC_ANCHOR_WINDOWS[int(su_i)]
    else:
        mu_f = float(fallback_mu if fallback_mu is not None else 0.0)
        lo, hi, mu = adjuster._get_su_common_window(
            int(su_i),
            fallback_mu=mu_f,
            pad=0.0,
            min_half_width=3.0,
        )
    return _window_score(adjuster, ppm_arr, diff_arr, float(lo), float(hi), float(mu))


def _count_degree_entries(counts: Dict[Any, Any], degrees: List[int]) -> int:
    return int(sum(int(dict(counts or {}).get(int(deg), dict(counts or {}).get(str(int(deg)), 0)) or 0) for deg in list(degrees or [])))


def _build_anchor_mode_meta_from_fixed_meta(H_cpu: torch.Tensor,
                                            fixed_partition_meta: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Dict[int, int]]]:
    fixed_meta = dict(fixed_partition_meta or {})
    raw_mode = _clone_special_anchor_mode_meta(fixed_meta.get('special_anchor_mode_meta'))
    raw_part = {
        int(su): {
            str(part): {int(deg): int(cnt) for deg, cnt in dict(part_counts or {}).items()}
            for part, part_counts in dict(parts or {}).items()
        }
        for su, parts in dict(fixed_meta.get('special_partition_meta', {}) or {}).items()
    }
    raw_degree = _clone_special_degree_meta(fixed_meta.get('special_degree_meta'))
    out: Dict[int, Dict[str, Dict[int, int]]] = {}

    total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
    if int(total_19) > 0:
        part_19 = dict(raw_part.get(19, raw_part.get('19', {})) or {})
        ether_counts = {int(deg): int(dict(part_19.get('ether', {}) or {}).get(int(deg), 0)) for deg in [1, 2, 3]}
        thio_counts = {int(deg): int(dict(part_19.get('thio', {}) or {}).get(int(deg), 0)) for deg in [1, 2, 3]}
        raw_19 = dict(raw_mode.get(19, raw_mode.get('19', {})) or {})
        ether_double = {
            int(deg): min(
                int(ether_counts.get(int(deg), 0)),
                max(0, int(dict(raw_19.get('ether_double', {}) or {}).get(int(deg), 0)))
            )
            for deg in [2, 3]
        }
        thio_double = {
            int(deg): min(
                int(thio_counts.get(int(deg), 0)),
                max(0, int(dict(raw_19.get('thio_double', {}) or {}).get(int(deg), 0)))
            )
            for deg in [2, 3]
        }
        ether_single = {
            int(deg): max(0, int(ether_counts.get(int(deg), 0)) - int(ether_double.get(int(deg), 0)))
            for deg in [1, 2, 3]
        }
        thio_single = {
            int(deg): max(0, int(thio_counts.get(int(deg), 0)) - int(thio_double.get(int(deg), 0)))
            for deg in [1, 2, 3]
        }
        out[19] = {
            'ether_single': {int(deg): int(ether_single.get(int(deg), 0)) for deg in [1, 2, 3]},
            'ether_double': {int(deg): int(ether_double.get(int(deg), 0)) for deg in [2, 3]},
            'thio_single': {int(deg): int(thio_single.get(int(deg), 0)) for deg in [1, 2, 3]},
            'thio_double': {int(deg): int(thio_double.get(int(deg), 0)) for deg in [2, 3]},
        }

    total_20 = int(H_cpu[20].item()) if int(H_cpu.numel()) > 20 else 0
    if int(total_20) > 0:
        deg_20 = dict(raw_degree.get(20, {}) or {})
        raw_20 = dict(raw_mode.get(20, raw_mode.get('20', {})) or {})
        double_counts = {
            int(deg): min(
                int(deg_20.get(int(deg), 0)),
                max(0, int(dict(raw_20.get('double', {}) or {}).get(int(deg), 0)))
            )
            for deg in [2, 3]
        }
        single_counts = {
            int(deg): max(0, int(deg_20.get(int(deg), 0)) - int(double_counts.get(int(deg), 0)))
            for deg in [1, 2, 3]
        }
        out[20] = {
            'single': {int(deg): int(single_counts.get(int(deg), 0)) for deg in [1, 2, 3]},
            'double': {int(deg): int(double_counts.get(int(deg), 0)) for deg in [2, 3]},
        }

    total_21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
    if int(total_21) > 0:
        deg_21 = dict(raw_degree.get(21, {}) or {})
        raw_21 = dict(raw_mode.get(21, raw_mode.get('21', {})) or {})
        raw_single_21 = dict(raw_21.get('single', {}) or {})
        raw_double_21 = dict(raw_21.get('double', {}) or {})
        if int(sum(int(deg_21.get(int(deg), 0)) for deg in [2, 3])) > 0:
            single_counts = {
                int(deg): max(0, int(deg_21.get(int(deg), 0)))
                for deg in [2, 3]
            }
        else:
            single_counts = {
                int(deg): (
                    max(0, int(raw_single_21.get(int(deg), raw_single_21.get(str(deg), 0)) or 0)) +
                    max(0, int(raw_double_21.get(int(deg), raw_double_21.get(str(deg), 0)) or 0))
                )
                for deg in [2, 3]
            }
            if int(sum(single_counts.values())) <= 0:
                single_counts[2] = int(total_21)
        excess = max(0, int(sum(single_counts.values())) - int(total_21))
        for deg in [2, 3]:
            if int(excess) <= 0:
                break
            take = min(int(excess), int(single_counts.get(int(deg), 0)))
            single_counts[int(deg)] = max(0, int(single_counts.get(int(deg), 0)) - int(take))
            excess -= int(take)
        deficit = max(0, int(total_21) - int(sum(single_counts.values())))
        if int(deficit) > 0:
            single_counts[2] = int(single_counts.get(2, 0)) + int(deficit)
        out[21] = {
            'single': {int(deg): int(single_counts.get(int(deg), 0)) for deg in [2, 3]},
            'double': {2: 0, 3: 0},
        }
    return out


def _fixed_anchor_edge_count(anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]],
                             su_type: int,
                             partition: Optional[str] = None) -> int:
    su_i = int(su_type)
    mode_meta = dict(anchor_mode_meta.get(int(su_i), {}) or {})
    if int(su_i) == 19:
        if str(partition or 'ether') == 'thio':
            single_counts = dict(mode_meta.get('thio_single', {}) or {})
            double_counts = dict(mode_meta.get('thio_double', {}) or {})
        else:
            single_counts = dict(mode_meta.get('ether_single', {}) or {})
            double_counts = dict(mode_meta.get('ether_double', {}) or {})
        return int(_count_degree_entries(single_counts, [1, 2, 3]) + 2 * _count_degree_entries(double_counts, [2, 3]))
    if int(su_i) == 20:
        single_counts = dict(mode_meta.get('single', {}) or {})
        double_counts = dict(mode_meta.get('double', {}) or {})
        return int(_count_degree_entries(single_counts, [1, 2, 3]) + 2 * _count_degree_entries(double_counts, [2, 3]))
    if int(su_i) == 21:
        single_counts = dict(mode_meta.get('single', {}) or {})
        return int(_count_degree_entries(single_counts, [2, 3]))
    return 0


def _summarize_block_b_balance(H: torch.Tensor,
                               fixed_partition_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    H_cpu = torch.clamp(H, min=0).long().detach().cpu()

    n0 = int(H_cpu[0].item()) if int(H_cpu.numel()) > 0 else 0
    n2 = int(H_cpu[2].item()) if int(H_cpu.numel()) > 2 else 0
    n5 = int(H_cpu[5].item()) if int(H_cpu.numel()) > 5 else 0
    n6 = int(H_cpu[6].item()) if int(H_cpu.numel()) > 6 else 0
    n7 = int(H_cpu[7].item()) if int(H_cpu.numel()) > 7 else 0
    n8 = int(H_cpu[8].item()) if int(H_cpu.numel()) > 8 else 0
    n19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
    n20 = int(H_cpu[20].item()) if int(H_cpu.numel()) > 20 else 0
    n21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
    n27 = int(H_cpu[27].item()) if int(H_cpu.numel()) > 27 else 0
    n28 = int(H_cpu[28].item()) if int(H_cpu.numel()) > 28 else 0
    n29 = int(H_cpu[29].item()) if int(H_cpu.numel()) > 29 else 0
    n31 = int(H_cpu[31].item()) if int(H_cpu.numel()) > 31 else 0
    n32 = int(H_cpu[32].item()) if int(H_cpu.numel()) > 32 else 0
    anchor_mode_meta = _build_anchor_mode_meta_from_fixed_meta(H_cpu, fixed_partition_meta=fixed_partition_meta)

    w_amine = int(n0 + 2 * n27)
    amine_20_edges = _fixed_anchor_edge_count(anchor_mode_meta, 20)
    if int(amine_20_edges) <= 0:
        amine_20_edges = int(n20)
    amine_total = int(n6 + amine_20_edges)
    amine_total_ok = bool(int(amine_total) == int(w_amine))

    w_thio = int(2 * n31)
    sulfur_required_19 = int((fixed_partition_meta or {}).get('s_reserved_19', max(0, int(w_thio - n7))))
    sulfur_required_19 = max(0, min(int(sulfur_required_19), int(n19)))
    sulfur_19_edges = _fixed_anchor_edge_count(anchor_mode_meta, 19, partition='thio')
    if int(sulfur_19_edges) <= 0:
        sulfur_19_edges = int(sulfur_required_19)
    sulfur_reserved_ok = bool(int(n19) >= int(sulfur_required_19) and int(n7 + sulfur_19_edges) == int(w_thio))

    w_ether = int(n2 + n28 + 2 * n29)
    ether_19_edges = _fixed_anchor_edge_count(anchor_mode_meta, 19)
    ether_19_available = max(0, int(n19 - sulfur_required_19))
    if int(ether_19_edges) <= 0:
        ether_19_edges = int(ether_19_available)
    ether_total = int(n5 + ether_19_edges)
    ether_total_ok = bool(int(ether_total) == int(w_ether))

    w_halogen = int(n32)
    halogen_21_edges = _fixed_anchor_edge_count(anchor_mode_meta, 21)
    if int(halogen_21_edges) <= 0:
        halogen_21_edges = int(n21)
    halogen_total = int(n8 + halogen_21_edges)
    halogen_total_ok = bool(int(halogen_total) == int(w_halogen))

    return {
        "amine_required": int(w_amine),
        "amine_total": int(amine_total),
        "amine_20_edges": int(amine_20_edges),
        "amine_total_ok": bool(amine_total_ok),
        "thio_required": int(w_thio),
        "sulfur_required_19": int(sulfur_required_19),
        "sulfur_19_edges": int(sulfur_19_edges),
        "sulfur_reserved_ok": bool(sulfur_reserved_ok),
        "ether_required": int(w_ether),
        "ether_total": int(ether_total),
        "ether_19_available": int(ether_19_edges),
        "ether_total_ok": bool(ether_total_ok),
        "halogen_required": int(w_halogen),
        "halogen_total": int(halogen_total),
        "halogen_21_edges": int(halogen_21_edges),
        "halogen_total_ok": bool(halogen_total_ok),
        "overall_ok": bool(amine_total_ok and sulfur_reserved_ok and ether_total_ok and halogen_total_ok),
    }


def _clone_special_degree_meta(meta: Optional[Dict[int, Dict[int, int]]]) -> Dict[int, Dict[int, int]]:
    return {
        int(su): {int(deg): int(cnt) for deg, cnt in dict(parts).items()}
        for su, parts in dict(meta or {}).items()
    }


def _clone_special_anchor_mode_meta(meta: Optional[Dict[int, Dict[str, Dict[int, int]]]]) -> Dict[int, Dict[str, Dict[int, int]]]:
    out: Dict[int, Dict[str, Dict[int, int]]] = {}
    for su, parts in dict(meta or {}).items():
        out[int(su)] = {}
        for mode_name, degree_map in dict(parts or {}).items():
            out[int(su)][str(mode_name)] = {
                int(deg): int(cnt) for deg, cnt in dict(degree_map or {}).items()
            }
    if 21 in out:
        single_counts = dict(out.get(21, {}).get('single', {}) or {})
        double_counts = dict(out.get(21, {}).get('double', {}) or {})
        for deg in [2, 3]:
            single_counts[int(deg)] = (
                int(single_counts.get(int(deg), single_counts.get(str(deg), 0)) or 0) +
                int(double_counts.get(int(deg), double_counts.get(str(deg), 0)) or 0)
            )
        out[21]['single'] = {int(deg): int(single_counts.get(int(deg), 0)) for deg in [2, 3]}
        out[21]['double'] = {2: 0, 3: 0}
    return out


def _format_degree_counts(counts: Dict[Any, Any], degrees: List[int]) -> str:
    return " ".join(
        f"d{int(deg)}={int(dict(counts or {}).get(int(deg), dict(counts or {}).get(str(int(deg)), 0)) or 0)}"
        for deg in list(degrees or [])
    )


def _anchor_mode_snapshot_rows(H: torch.Tensor,
                               fixed_partition_meta: Optional[Dict[str, Any]] = None) -> List[str]:
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    mode_meta = _build_anchor_mode_meta_from_fixed_meta(H_cpu, fixed_partition_meta=fixed_partition_meta)
    rows: List[str] = []

    mode_19 = dict(mode_meta.get(19, {}) or {})
    ether_single = dict(mode_19.get('ether_single', {}) or {})
    ether_double = dict(mode_19.get('ether_double', {}) or {})
    thio_single = dict(mode_19.get('thio_single', {}) or {})
    thio_double = dict(mode_19.get('thio_double', {}) or {})
    rows.append(
        "SU19 ether: "
        f"single[{_format_degree_counts(ether_single, [1, 2, 3])}] "
        f"double[{_format_degree_counts(ether_double, [2, 3])}] "
        f"nodes={int(_count_degree_entries(ether_single, [1, 2, 3]) + _count_degree_entries(ether_double, [2, 3]))} "
        f"fixed_edges={int(_fixed_anchor_edge_count(mode_meta, 19, partition='ether'))}"
    )
    rows.append(
        "SU19 thio:  "
        f"single[{_format_degree_counts(thio_single, [1, 2, 3])}] "
        f"double[{_format_degree_counts(thio_double, [2, 3])}] "
        f"nodes={int(_count_degree_entries(thio_single, [1, 2, 3]) + _count_degree_entries(thio_double, [2, 3]))} "
        f"fixed_edges={int(_fixed_anchor_edge_count(mode_meta, 19, partition='thio'))}"
    )

    mode_20 = dict(mode_meta.get(20, {}) or {})
    single_20 = dict(mode_20.get('single', {}) or {})
    double_20 = dict(mode_20.get('double', {}) or {})
    rows.append(
        "SU20:       "
        f"single[{_format_degree_counts(single_20, [1, 2, 3])}] "
        f"double[{_format_degree_counts(double_20, [2, 3])}] "
        f"nodes={int(_count_degree_entries(single_20, [1, 2, 3]) + _count_degree_entries(double_20, [2, 3]))} "
        f"fixed_edges={int(_fixed_anchor_edge_count(mode_meta, 20))}"
    )

    mode_21 = dict(mode_meta.get(21, {}) or {})
    single_21 = dict(mode_21.get('single', {}) or {})
    rows.append(
        "SU21:       "
        f"single[{_format_degree_counts(single_21, [2, 3])}] "
        f"nodes={int(_count_degree_entries(single_21, [2, 3]))} "
        f"fixed_edges={int(_fixed_anchor_edge_count(mode_meta, 21))}"
    )
    return rows


def _print_anchor_mode_snapshot(H: torch.Tensor,
                                fixed_partition_meta: Optional[Dict[str, Any]] = None,
                                header: Optional[str] = None,
                                indent: str = "  ") -> None:
    if header:
        print(f"{indent}{header}")
    for row in _anchor_mode_snapshot_rows(H, fixed_partition_meta=fixed_partition_meta):
        print(f"{indent}{row}")


def _parse_multiset_text(text: Any) -> Tuple[int, ...]:
    raw = str(text or '').strip().strip('"').strip("'")
    raw = raw.strip('[]')
    if not raw:
        return tuple()
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    try:
        return tuple(sorted(int(x) for x in parts))
    except Exception:
        return tuple()


def _fixed_anchor_types_for_special_su(su_type: int,
                                       partition: Optional[str] = None) -> Tuple[int, ...]:
    if int(su_type) == 19:
        if str(partition or '') == 'thio':
            return (31,)
        if str(partition or '') == 'ether':
            return (2, 28, 29)
        return (2, 28, 29, 31)
    if int(su_type) == 20:
        return (0, 27)
    if int(su_type) == 21:
        return (32,)
    return tuple()


def _classify_anchor_mode_from_hop1(su_type: int, hop1_multiset: Tuple[int, ...]) -> Optional[str]:
    su_i = int(su_type)
    degree_i = int(len(tuple(hop1_multiset)))
    if int(degree_i) not in {2, 3}:
        return None
    if violates_special_d3_terminal_limit(int(su_i), int(degree_i), tuple(hop1_multiset)):
        return None
    partition = 'thio' if 31 in set(int(x) for x in tuple(hop1_multiset)) else 'ether'
    fixed_types = set(int(x) for x in _fixed_anchor_types_for_special_su(int(su_i), partition=partition))
    if not fixed_types:
        return None
    fixed_count = int(sum(1 for x in tuple(hop1_multiset) if int(x) in fixed_types))
    if int(su_i) == 21:
        return f"single_d{int(degree_i)}" if int(fixed_count) == 1 else None
    if int(fixed_count) == 1:
        return f"single_d{int(degree_i)}"
    if int(fixed_count) == 2:
        return f"double_d{int(degree_i)}"
    return None


def _get_special_anchor_mode_stats(adjuster: Any) -> Dict[int, Dict[str, Dict[str, float]]]:
    cache = getattr(adjuster, '_su_special_anchor_mode_stats_cache', None)
    if isinstance(cache, dict):
        return cache

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    try:
        hop1_path = Path(getattr(adjuster, 'su_hop1_ranges_path', ''))
    except Exception:
        hop1_path = Path()
    mode_csv = hop1_path.parent / 'su_special_anchor_mode_nmr_range_filtered.csv'

    if mode_csv.exists():
        try:
            df_mode = pd.read_csv(mode_csv)
            for _, row in df_mode.iterrows():
                su_i = int(row.get('center_su_idx', row.get('center_su', -1)))
                mode = str(row.get('anchor_mode', '') or '')
                if not mode:
                    continue
                out.setdefault(int(su_i), {})[str(mode)] = {
                    'mu_median': float(row.get('mu_median', 0.0)),
                    'mu_common_min': float(row.get('mu_common_min', 0.0)),
                    'mu_common_max': float(row.get('mu_common_max', 0.0)),
                    'mu_q05': float(row.get('mu_q05', row.get('mu_common_min', 0.0))),
                    'mu_q95': float(row.get('mu_q95', row.get('mu_common_max', 0.0))),
                    'mu_global_min': float(row.get('mu_global_min', row.get('mu_common_min', 0.0))),
                    'mu_global_max': float(row.get('mu_global_max', row.get('mu_common_max', 0.0))),
                    'n_templates': int(row.get('n_templates', 0)),
                    'sample_count_total': int(row.get('sample_count_total', 0)),
                }
            setattr(adjuster, '_su_special_anchor_mode_stats_cache', out)
            return out
        except Exception:
            out = {}

    if hop1_path.exists():
        try:
            df_hop1 = pd.read_csv(hop1_path)
            grouped: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
            for _, row in df_hop1.iterrows():
                su_i = int(row.get('center_su_idx', row.get('center_su', -1)))
                if int(su_i) not in {19, 20, 21}:
                    continue
                hop1_tuple = _parse_multiset_text(row.get('hop1_multiset', ''))
                mode = _classify_anchor_mode_from_hop1(int(su_i), hop1_tuple)
                if not mode:
                    continue
                grouped.setdefault((int(su_i), str(mode)), []).append({
                    'mu': float(row.get('mu_median', 0.0)),
                    'mu_min': float(row.get('mu_common_min', row.get('mu_median', 0.0))),
                    'mu_max': float(row.get('mu_common_max', row.get('mu_median', 0.0))),
                    'sample_count': int(row.get('sample_count_total', row.get('sample_count', 1))),
                })
            for (su_i, mode), items in grouped.items():
                if not items:
                    continue
                mu_values = [float(it['mu']) for it in items]
                weights = [max(1, int(it['sample_count'])) for it in items]
                if sum(weights) <= 0:
                    continue
                order = np.argsort(mu_values)
                mu_sorted = [mu_values[int(i)] for i in order]
                w_sorted = [weights[int(i)] for i in order]
                total_w = float(sum(w_sorted))
                cumsum = np.cumsum(np.asarray(w_sorted, dtype=np.float64))
                q05_idx = int(np.searchsorted(cumsum, 0.05 * total_w, side='left'))
                q95_idx = int(np.searchsorted(cumsum, 0.95 * total_w, side='left'))
                weighted_mean = float(np.average(np.asarray(mu_values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
                out.setdefault(int(su_i), {})[str(mode)] = {
                    'mu_median': float(weighted_mean),
                    'mu_common_min': float(min(it['mu_min'] for it in items)),
                    'mu_common_max': float(max(it['mu_max'] for it in items)),
                    'mu_q05': float(mu_sorted[min(max(q05_idx, 0), len(mu_sorted) - 1)]),
                    'mu_q95': float(mu_sorted[min(max(q95_idx, 0), len(mu_sorted) - 1)]),
                    'mu_global_min': float(min(it['mu_min'] for it in items)),
                    'mu_global_max': float(max(it['mu_max'] for it in items)),
                    'n_templates': int(len(items)),
                    'sample_count_total': int(sum(weights)),
                }
        except Exception:
            out = {}

    setattr(adjuster, '_su_special_anchor_mode_stats_cache', out)
    return out

def _classify_anchor_mode_from_node(adjuster: Any,
                                    node: Any,
                                    nodes: List[Any]) -> Optional[str]:
    su_i = int(getattr(node, 'su_type', -1))
    if int(su_i) not in {19, 20, 21}:
        return None
    if int(su_i) == 19 and str(getattr(node, 'special_anchor_partition', None) or '') not in {'ether', 'thio'}:
        return None
    try:
        degree_i = int(getattr(node, 'target_hop1_degree', None) or max(0, len(list(getattr(node, 'hop1_ids', []) or []))))
    except Exception:
        degree_i = 0
    if int(degree_i) not in {2, 3}:
        return None
    part = str(getattr(node, 'special_anchor_partition', None) or '')
    fixed_types = set(int(x) for x in _fixed_anchor_types_for_special_su(int(su_i), partition=part))
    neighbor_types = [int(x) for x in list(adjuster._current_neighbor_types(node, nodes) or [])]
    if violates_special_d3_terminal_limit(int(su_i), int(degree_i), neighbor_types):
        return None
    fixed_count = int(sum(1 for x in neighbor_types if int(x) in fixed_types))
    if int(su_i) == 21:
        return f"single_d{int(degree_i)}" if int(fixed_count) == 1 else None
    if int(fixed_count) == 1:
        return f"single_d{int(degree_i)}"
    if int(fixed_count) == 2:
        return f"double_d{int(degree_i)}"
    return None


def _collect_anchor_mode_counts(adjuster: Any,
                                nodes: List[Any],
                                su_type: int,
                                anchor_partition: Optional[str] = None) -> Dict[str, int]:
    counts = {m: 0 for m in ('single_d2', 'single_d3', 'double_d2', 'double_d3')}
    for node in list(nodes or []):
        if int(getattr(node, 'su_type', -1)) != int(su_type):
            continue
        if int(su_type) == 19 and anchor_partition is not None:
            if str(getattr(node, 'special_anchor_partition', None) or '') != str(anchor_partition):
                continue
        mode = _classify_anchor_mode_from_node(adjuster, node, nodes)
        if mode in counts:
            counts[str(mode)] += 1
    return counts


def _build_anchor_mode_meta_from_nodes(adjuster: Any,
                                       nodes: List[Any]) -> Dict[int, Dict[str, Dict[int, int]]]:
    out: Dict[int, Dict[str, Dict[int, int]]] = {}

    counts_19 = {
        'ether_single': {1: 0, 2: 0, 3: 0},
        'ether_double': {2: 0, 3: 0},
        'thio_single': {1: 0, 2: 0, 3: 0},
        'thio_double': {2: 0, 3: 0},
    }
    counts_20 = {
        'single': {1: 0, 2: 0, 3: 0},
        'double': {2: 0, 3: 0},
    }
    counts_21 = {
        'single': {2: 0, 3: 0},
        'double': {2: 0, 3: 0},
    }
    for node in list(nodes or []):
        su_i = int(getattr(node, 'su_type', -1))
        deg_i = int(getattr(node, 'target_hop1_degree', None) or max(0, len(list(getattr(node, 'hop1_ids', []) or []))))
        if int(su_i) == 19:
            part = str(getattr(node, 'special_anchor_partition', None) or '')
            mode = _classify_anchor_mode_from_node(adjuster, node, nodes)
            if str(part) == 'thio':
                if str(mode) == 'double_d2':
                    counts_19['thio_double'][2] += 1
                elif str(mode) == 'double_d3':
                    counts_19['thio_double'][3] += 1
                elif int(deg_i) in {1, 2, 3}:
                    counts_19['thio_single'][int(deg_i)] += 1
                continue
            if str(mode) == 'double_d2':
                counts_19['ether_double'][2] += 1
            elif str(mode) == 'double_d3':
                counts_19['ether_double'][3] += 1
            elif int(deg_i) in {1, 2, 3}:
                counts_19['ether_single'][int(deg_i)] += 1
        elif int(su_i) == 20:
            mode = _classify_anchor_mode_from_node(adjuster, node, nodes)
            if str(mode) == 'double_d2':
                counts_20['double'][2] += 1
            elif str(mode) == 'double_d3':
                counts_20['double'][3] += 1
            elif int(deg_i) in {1, 2, 3}:
                counts_20['single'][int(deg_i)] += 1
        elif int(su_i) == 21:
            mode = _classify_anchor_mode_from_node(adjuster, node, nodes)
            if str(mode) in {'single_d2', 'single_d3'} and int(deg_i) in {2, 3}:
                counts_21['single'][int(deg_i)] += 1
    out[19] = {
        'ether_single': {int(deg): int(counts_19['ether_single'][int(deg)]) for deg in [1, 2, 3]},
        'ether_double': {int(deg): int(counts_19['ether_double'].get(int(deg), 0)) for deg in [2, 3]},
        'thio_single': {int(deg): int(counts_19['thio_single'][int(deg)]) for deg in [1, 2, 3]},
        'thio_double': {int(deg): int(counts_19['thio_double'].get(int(deg), 0)) for deg in [2, 3]},
    }
    out[20] = {
        'single': {int(deg): int(counts_20['single'][int(deg)]) for deg in [1, 2, 3]},
        'double': {int(deg): int(counts_20['double'].get(int(deg), 0)) for deg in [2, 3]},
    }
    out[21] = {
        'single': {int(deg): int(counts_21['single'].get(int(deg), 0)) for deg in [2, 3]},
        'double': {2: 0, 3: 0},
    }
    return out


def _special_anchor_mode_scores(adjuster: Any,
                                ppm_arr: np.ndarray,
                                diff_arr: np.ndarray,
                                su_type: int) -> Dict[str, Dict[str, float]]:
    stats_source = _get_special_anchor_mode_stats(adjuster)
    stats_by_mode = dict(stats_source.get(int(su_type), {}) or {})
    manual_by_mode = dict(MANUAL_SPECIAL_ANCHOR_MODE_WINDOWS.get(int(su_type), {}) or {})
    out: Dict[str, Dict[str, float]] = {}
    for mode_name in ('single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'):
        base = dict(stats_by_mode.get(str(mode_name), {}) or {})
        manual = manual_by_mode.get(str(mode_name))
        if manual is None and not base:
            continue
        if manual is not None:
            lo, hi, mu = (float(manual[0]), float(manual[1]), float(manual[2]))
        else:
            mu = float(base.get('mu_median', 0.0))
            lo = float(base.get('mu_common_min', base.get('mu_q05', mu)))
            hi = float(base.get('mu_common_max', base.get('mu_q95', mu)))
        if float(hi) <= float(lo):
            lo = float(mu - 3.0)
            hi = float(mu + 3.0)
        stats = _window_score(adjuster, ppm_arr, diff_arr, float(lo), float(hi), float(mu))
        out[str(mode_name)] = {
            'lo': float(lo),
            'hi': float(hi),
            'mu': float(mu),
            'pos': float(stats.get('pos', 0.0)),
            'neg': float(stats.get('neg', 0.0)),
            'neg_abs': float(stats.get('neg_abs', 0.0)),
            'score': float(stats.get('score', 0.0)),
            'abs': float(stats.get('abs', 0.0)),
            'source': 'manual' if manual is not None else 'csv',
            'csv_mu_median': float(base.get('mu_median', mu)),
            'csv_q05': float(base.get('mu_q05', lo)),
            'csv_q95': float(base.get('mu_q95', hi)),
        }
    return out


def _special_degree_core_scores(adjuster: Any,
                                ppm_arr: np.ndarray,
                                diff_arr: np.ndarray,
                                su_type: int,
                                degrees: List[int],
                                core_half_width: float = 3.5,
                                support_weight: float = 0.25) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for degree_i in [int(x) for x in list(degrees or [])]:
        lo_w, hi_w, mu = adjuster._get_su_special_degree_window(
            int(su_type),
            int(degree_i),
            fallback_mu=0.0,
            pad=0.0,
            min_half_width=float(core_half_width),
        )
        core_hw = float(core_half_width)
        if int(su_type) == 19 and int(degree_i) == 2:
            core_hi = min(float(mu + 2.5), float(hi_w))
            core_lo = max(float(mu - core_hw), float(lo_w))
            # d2 的高ppm上尾容易侵入 d3 区域，单独做弱化统计。
            tail_lo = max(float(core_hi), float(lo_w))
            tail_hi = float(hi_w)
        else:
            core_lo = max(float(mu - core_hw), float(lo_w))
            core_hi = min(float(mu + core_hw), float(hi_w))
            tail_lo = float(core_hi)
            tail_hi = float(core_hi)
        if float(core_hi) <= float(core_lo):
            core_lo = float(lo_w)
            core_hi = float(hi_w)
        stats_core = adjuster._window_stats(ppm_arr, diff_arr, float(core_lo), float(core_hi))
        stats_support = adjuster._window_stats(ppm_arr, diff_arr, float(lo_w), float(hi_w))
        if float(tail_hi) > float(tail_lo) + 1e-6:
            stats_tail = adjuster._window_stats(ppm_arr, diff_arr, float(tail_lo), float(tail_hi))
        else:
            stats_tail = {'pos': 0.0, 'neg': 0.0, 'abs': 0.0}
        core_score = float(stats_core.get('pos', 0.0) - stats_core.get('neg', 0.0))
        support_score = float(stats_support.get('pos', 0.0) - stats_support.get('neg', 0.0))
        tail_score = float(stats_tail.get('pos', 0.0) - stats_tail.get('neg', 0.0))
        # d2 的高ppm尾部只做弱支持，避免把 d3 信号误吞到 d2。
        if int(su_type) == 19 and int(degree_i) == 2 and float(tail_hi) > float(tail_lo) + 1e-6:
            final_score = float(core_score + float(support_weight) * support_score - 0.35 * max(0.0, float(tail_score)))
        else:
            final_score = float(core_score + float(support_weight) * support_score)
        out[int(degree_i)] = {
            'mu': float(mu),
            'lo': float(lo_w),
            'hi': float(hi_w),
            'core_lo': float(core_lo),
            'core_hi': float(core_hi),
            'tail_lo': float(tail_lo),
            'tail_hi': float(tail_hi),
            'core_pos': float(stats_core.get('pos', 0.0)),
            'core_neg': float(stats_core.get('neg', 0.0)),
            'support_pos': float(stats_support.get('pos', 0.0)),
            'support_neg': float(stats_support.get('neg', 0.0)),
            'tail_pos': float(stats_tail.get('pos', 0.0)),
            'tail_neg': float(stats_tail.get('neg', 0.0)),
            'core_score': float(core_score),
            'support_score': float(support_score),
            'tail_score': float(tail_score),
            'score': float(final_score),
            'abs': float(stats_core.get('abs', 0.0)),
        }
    return out


def _mode_tail_su(su_type: int, degree_i: int) -> Optional[int]:
    su_i = int(su_type)
    deg_i = int(degree_i)
    if su_i in {19, 20}:
        return {1: 22, 2: 23, 3: 24}.get(deg_i)
    if su_i == 21:
        return {2: 23, 3: 24}.get(deg_i)
    return None


def _special_mode_scores(adjuster: Any,
                         ppm_arr: np.ndarray,
                         diff_arr: np.ndarray,
                         su_type: int,
                         degrees: List[int],
                         window: float = 3.0) -> Dict[int, Dict[str, float]]:
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, int(su_type))
    if mode_scores:
        stats_by_degree: Dict[int, Dict[str, float]] = {}
        for degree_i in [int(x) for x in list(degrees or [])]:
            candidates: List[Dict[str, float]] = []
            for prefix in ('single', 'double'):
                key = f"{prefix}_d{int(degree_i)}"
                if key in mode_scores:
                    candidates.append(dict(mode_scores[key]))
            if not candidates:
                continue
            pos = float(sum(float(item.get('pos', 0.0)) for item in candidates))
            neg_abs = float(sum(float(item.get('neg_abs', abs(float(item.get('neg', 0.0))))) for item in candidates))
            abs_area = float(sum(float(item.get('abs', 0.0)) for item in candidates))
            best = max(candidates, key=lambda item: abs(float(item.get('score', 0.0))))
            stats_by_degree[int(degree_i)] = {
                'lo': float(min(float(item.get('lo', 0.0)) for item in candidates)),
                'hi': float(max(float(item.get('hi', 0.0)) for item in candidates)),
                'mu': float(best.get('mu', 0.0)),
                'pos': float(pos),
                'neg': float(-neg_abs),
                'neg_abs': float(neg_abs),
                'score': float(pos - neg_abs),
                'abs': float(abs_area),
                'core_lo': float(best.get('lo', 0.0)),
                'core_hi': float(best.get('hi', 0.0)),
                'mode_scores': {str(k): dict(v) for k, v in mode_scores.items() if str(k).endswith(f"d{int(degree_i)}")},
            }
        if stats_by_degree:
            return stats_by_degree
    stats_by_degree: Dict[int, Dict[str, float]] = {}
    common_fallback = {
        19: 66.6875,
        20: 49.375,
        21: 38.4141,
    }
    for degree_i in degrees:
        lo, hi, mu = adjuster._get_su_special_degree_window(
            int(su_type),
            int(degree_i),
            fallback_mu=common_fallback.get(int(su_type), 0.0),
            pad=0.20 * float(window),
            min_half_width=float(window),
        )
        stats = adjuster._window_stats(ppm_arr, diff_arr, lo, hi)
        dom = float(stats["pos"]) if float(stats["pos"]) >= float(stats["neg"]) else -float(stats["neg"])
        stats_by_degree[int(degree_i)] = {
            "lo": float(lo),
            "hi": float(hi),
            "mu": float(mu),
            "pos": float(stats["pos"]),
            "neg": -float(stats["neg"]),
            "neg_abs": float(stats["neg"]),
            "score": float(dom),
            "abs": float(stats["abs"]),
        }
    return stats_by_degree


def _pick_mode_increase_degree(scores: Dict[int, Dict[str, float]],
                               allowed_degrees: List[int],
                               donor_check_fn) -> Optional[int]:
    candidates = [int(d) for d in allowed_degrees if donor_check_fn(int(d))]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda d: (
            float(scores.get(int(d), {}).get("score", 0.0)),
            -int(d),
        ),
        reverse=True,
    )
    return int(ranked[0]) if ranked else None


def _pick_mode_decrease_degree(scores: Dict[int, Dict[str, float]],
                               counts: Dict[int, int],
                               allowed_degrees: List[int]) -> Optional[int]:
    candidates = [int(d) for d in allowed_degrees if int(counts.get(int(d), 0)) > 0]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda d: (
            float(-scores.get(int(d), {}).get("score", 0.0)),
            int(counts.get(int(d), 0)),
            int(d),
        ),
        reverse=True,
    )
    return int(ranked[0]) if ranked else None


def _mode_stats_meta(scores: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for degree_i, stats in scores.items():
        out[f"d{int(degree_i)}"] = dict(stats)
    return out


def _anchor_mode_stats_meta(scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    return {str(mode_name): dict(stats) for mode_name, stats in dict(scores or {}).items()}


def _anchor_mode_bucket_names(su_type: int) -> Tuple[str, str, List[int], List[int]]:
    su_i = int(su_type)
    if int(su_i) == 19:
        return 'ether_single', 'ether_double', [1, 2, 3], [2, 3]
    if int(su_i) == 20:
        return 'single', 'double', [1, 2, 3], [2, 3]
    if int(su_i) == 21:
        return 'single', 'double', [2, 3], []
    return 'single', 'double', [], []


def _mode_bucket_count(anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]],
                       su_type: int,
                       bucket: str,
                       degree_i: int) -> int:
    return int(dict(anchor_mode_meta.get(int(su_type), {}).get(str(bucket), {}) or {}).get(int(degree_i), 0))


def _count_nonterminal_fixed_anchor_capacity(H: torch.Tensor,
                                             anchor_types: List[int]) -> int:
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    slot_multipliers = {27: 2, 29: 2, 31: 2}
    return int(sum(
        int(H_cpu[int(su)].item()) * int(slot_multipliers.get(int(su), 1))
        for su in list(anchor_types or [])
        if int(su) not in set(int(x) for x in SPECIAL_D3_TERMINAL_NEIGHBORS)
        and 0 <= int(su) < int(H_cpu.numel())
    ))


def _special_mode_fixed_anchor_pool_violation(H: torch.Tensor,
                                              anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]],
                                              su_type: int,
                                              bucket: str,
                                              degree_i: int) -> Optional[str]:
    """Return a reason if a mode bucket exceeds Layer1's fixed-anchor pool."""
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    su_i = int(su_type)
    bucket_txt = str(bucket)
    deg_i = int(degree_i)
    count_i = int(_mode_bucket_count(anchor_mode_meta, int(su_i), bucket_txt, int(deg_i)))
    if int(count_i) <= 0:
        return None

    if int(su_i) == 19 and bucket_txt == 'ether_single' and int(deg_i) == 1:
        cap = int(H_cpu[2].item()) + int(H_cpu[29].item())
        if int(count_i) > int(cap):
            return f"ether_single_d1_pool_exceeded(count={int(count_i)},cap={int(cap)})"

    if int(su_i) == 19 and int(deg_i) == 3 and bucket_txt in {'ether_double', 'thio_double'}:
        anchor_types = [31] if bucket_txt == 'thio_double' else [2, 28, 29]
        nonterminal_cap = _count_nonterminal_fixed_anchor_capacity(H_cpu, anchor_types)
        if int(count_i) > int(nonterminal_cap):
            return (
                f"special_double_d3_nonterminal_pool_exceeded("
                f"su={int(su_i)},bucket={bucket_txt},count={int(count_i)},cap={int(nonterminal_cap)})"
            )

    if int(su_i) == 20 and bucket_txt == 'single' and int(deg_i) == 1:
        cap = int(H_cpu[0].item()) + 2 * int(H_cpu[27].item())
        if int(count_i) > int(cap):
            return f"amine_single_d1_pool_exceeded(count={int(count_i)},cap={int(cap)})"

    if int(su_i) == 20 and int(deg_i) == 3 and bucket_txt == 'double':
        nonterminal_cap = _count_nonterminal_fixed_anchor_capacity(H_cpu, [0, 27])
        if int(count_i) > int(nonterminal_cap):
            return (
                f"special_double_d3_nonterminal_pool_exceeded("
                f"su={int(su_i)},bucket={bucket_txt},count={int(count_i)},cap={int(nonterminal_cap)})"
            )

    if int(su_i) == 21 and bucket_txt == 'double':
        # SU21 fixed anchor is only X/SU32, which is terminal-like.  A SU21
        # node therefore always consumes exactly one X anchor; double modes are
        # not chemically valid for this type.
        return f"halogen_double_mode_not_allowed(count={int(count_i)})"

    return None


def _score_special_mode_total(mode_scores: Dict[str, Dict[str, float]],
                              mode_names: Optional[List[str]] = None) -> Dict[str, float]:
    names = list(mode_names or sorted(str(k) for k in dict(mode_scores or {}).keys()))
    items = [dict(mode_scores.get(str(name), {}) or {}) for name in names if str(name) in dict(mode_scores or {})]
    pos = float(sum(float(item.get('pos', 0.0)) for item in items))
    neg_abs = float(sum(float(item.get('neg_abs', abs(float(item.get('neg', 0.0))))) for item in items))
    abs_area = float(sum(float(item.get('abs', 0.0)) for item in items))
    return {
        'pos': float(pos),
        'neg': float(-neg_abs),
        'neg_abs': float(neg_abs),
        'score': float(pos - neg_abs),
        'abs': float(abs_area),
    }


def _mode_score_value(mode_scores: Dict[str, Dict[str, float]], mode_name: str) -> float:
    return float(dict(mode_scores.get(str(mode_name), {}) or {}).get('score', 0.0))


def _select_special_mode_for_increase(mode_scores: Dict[str, Dict[str, float]],
                                      H_like: torch.Tensor,
                                      anchor_su: int,
                                      special_su: int,
                                      single_bucket: str,
                                      double_bucket: str,
                                      allowed_single: List[int],
                                      allowed_double: List[int]) -> Optional[Tuple[str, str, int, int]]:
    _ = str(single_bucket), str(double_bucket)
    h_anchor = int(H_like[int(anchor_su)].item()) if int(H_like.numel()) > int(anchor_su) else 0
    candidates: List[Tuple[float, int, int, str, str, int, int]] = []
    for degree_i in [int(x) for x in list(allowed_single or [])]:
        mode_key = f"single_d{int(degree_i)}"
        tail_su = _mode_tail_su(int(special_su), int(degree_i))
        if mode_key not in mode_scores or tail_su is None:
            continue
        if int(h_anchor) < 1 or int(H_like[int(tail_su)].item()) <= 0:
            continue
        candidates.append((
            float(_mode_score_value(mode_scores, mode_key)),
            1,
            int(degree_i),
            'single',
            str(mode_key),
            int(tail_su),
            1,
        ))
    for degree_i in [int(x) for x in list(allowed_double or [])]:
        mode_key = f"double_d{int(degree_i)}"
        tail_su = _mode_tail_su(int(special_su), int(degree_i))
        if mode_key not in mode_scores or tail_su is None:
            continue
        if int(h_anchor) < 2 or int(H_like[int(tail_su)].item()) <= 0:
            continue
        candidates.append((
            float(_mode_score_value(mode_scores, mode_key)),
            2,
            int(degree_i),
            'double',
            str(mode_key),
            int(tail_su),
            2,
        ))
    if not candidates:
        return None
    _, _, degree_i, mode_kind, mode_key, _tail_su, cost = max(
        candidates,
        key=lambda item: (float(item[0]), int(item[1]), int(item[2])),
    )
    bucket = str(double_bucket) if str(mode_kind) == 'double' else str(single_bucket)
    return str(bucket), str(mode_kind), int(degree_i), int(cost)


def _select_special_mode_for_decrease(mode_scores: Dict[str, Dict[str, float]],
                                      mode_meta: Dict[int, Dict[str, Dict[int, int]]],
                                      special_su: int,
                                      single_bucket: str,
                                      double_bucket: str,
                                      allowed_single: List[int],
                                      allowed_double: List[int]) -> Optional[Tuple[str, str, int, int]]:
    candidates: List[Tuple[float, int, int, str, str, int]] = []
    for degree_i in [int(x) for x in list(allowed_single or [])]:
        if int(_mode_bucket_count(mode_meta, int(special_su), str(single_bucket), int(degree_i))) <= 0:
            continue
        mode_key = f"single_d{int(degree_i)}"
        candidates.append((
            float(-_mode_score_value(mode_scores, mode_key)),
            1,
            int(degree_i),
            str(single_bucket),
            'single',
            1,
        ))
    for degree_i in [int(x) for x in list(allowed_double or [])]:
        if int(_mode_bucket_count(mode_meta, int(special_su), str(double_bucket), int(degree_i))) <= 0:
            continue
        mode_key = f"double_d{int(degree_i)}"
        candidates.append((
            float(-_mode_score_value(mode_scores, mode_key)),
            2,
            int(degree_i),
            str(double_bucket),
            'double',
            2,
        ))
    if not candidates:
        return None
    _, _, degree_i, bucket, mode_kind, cost = max(
        candidates,
        key=lambda item: (float(item[0]), int(item[1]), int(item[2])),
    )
    return str(bucket), str(mode_kind), int(degree_i), int(cost)


def _apply_mode_bucket_delta(anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]],
                             su_type: int,
                             bucket: str,
                             degree_i: int,
                             delta: int) -> None:
    su_i = int(su_type)
    bucket_txt = str(bucket)
    cur = dict(anchor_mode_meta.get(int(su_i), {}).get(bucket_txt, {}) or {})
    cur[int(degree_i)] = max(0, int(cur.get(int(degree_i), 0)) + int(delta))
    anchor_mode_meta.setdefault(int(su_i), {})[bucket_txt] = cur


def _sync_adjuster_mode_meta(adjuster: Any,
                             H: torch.Tensor,
                             special_degree_meta: Dict[int, Dict[int, int]],
                             anchor_mode_meta: Optional[Dict[int, Dict[str, Dict[int, int]]]] = None) -> None:
    total_19 = 0
    fixed_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
    fixed_meta['special_degree_meta'] = _clone_special_degree_meta(special_degree_meta)
    if anchor_mode_meta is not None:
        fixed_meta['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(anchor_mode_meta)
        try:
            total_19 = int(torch.clamp(H.detach().cpu(), min=0).long()[19].item()) if H is not None and int(H.numel()) > 19 else 0
        except Exception:
            total_19 = 0
        if int(total_19) > 0:
            ether_single = dict(anchor_mode_meta.get(19, {}).get('ether_single', {}) or {})
            ether_double = dict(anchor_mode_meta.get(19, {}).get('ether_double', {}) or {})
            thio_single = dict(anchor_mode_meta.get(19, {}).get('thio_single', {}) or {})
            thio_double = dict(anchor_mode_meta.get(19, {}).get('thio_double', {}) or {})
            ether_total_by_degree = {
                int(deg): int(ether_single.get(int(deg), 0)) + int(ether_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            thio_total_by_degree = {
                int(deg): int(thio_single.get(int(deg), 0)) + int(thio_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
            fixed_meta['special_partition_meta'][19] = {
                'ether': dict(ether_total_by_degree),
                'thio': dict(thio_total_by_degree),
            }
            fixed_meta['o_base_19'] = int(sum(int(v) for v in ether_total_by_degree.values()))
            fixed_meta['s_reserved_19'] = int(sum(int(v) for v in thio_total_by_degree.values()))
            fixed_meta['n19_total'] = int(total_19)
    adjuster.fixed_partition_meta = dict(fixed_meta)
    adjuster._set_special_degree_meta(H, special_degree_meta)
    fixed_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
    if anchor_mode_meta is not None:
        fixed_meta['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(total_19) > 0:
            ether_single = dict(anchor_mode_meta.get(19, {}).get('ether_single', {}) or {})
            ether_double = dict(anchor_mode_meta.get(19, {}).get('ether_double', {}) or {})
            thio_single = dict(anchor_mode_meta.get(19, {}).get('thio_single', {}) or {})
            thio_double = dict(anchor_mode_meta.get(19, {}).get('thio_double', {}) or {})
            ether_total_by_degree = {
                int(deg): int(ether_single.get(int(deg), 0)) + int(ether_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            thio_total_by_degree = {
                int(deg): int(thio_single.get(int(deg), 0)) + int(thio_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
            fixed_meta['special_partition_meta'][19] = {
                'ether': dict(ether_total_by_degree),
                'thio': dict(thio_total_by_degree),
            }
            fixed_meta['o_base_19'] = int(sum(int(v) for v in ether_total_by_degree.values()))
            fixed_meta['s_reserved_19'] = int(sum(int(v) for v in thio_total_by_degree.values()))
            fixed_meta['n19_total'] = int(total_19)
    adjuster.fixed_partition_meta = dict(fixed_meta)
    try:
        if getattr(adjuster, 'layer0_estimator', None) is not None:
            layer0_meta = dict(getattr(adjuster.layer0_estimator, 'fixed_partition_meta', {}) or {})
            layer0_meta.update(dict(fixed_meta))
            adjuster.layer0_estimator.fixed_partition_meta = dict(layer0_meta)
            adjuster.layer0_estimator.special_partition_meta = dict(fixed_meta.get('special_partition_meta', {}) or {})
    except Exception:
        pass


def adjust_special_degree_mode_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    su_type: int,
    degrees: List[int],
    anchor_partition: Optional[str] = None,
    max_moves: int = 4,
    peak_rel_threshold: float = 0.01,
    window: float = 3.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    label = f"SU{int(su_type)}内部连接度"
    print(f"\n[{label}调整] 基于差谱分析")

    def _counts_view(counts_map: Dict[int, int]) -> str:
        return ", ".join(
            f"d{int(deg)}={int(counts_map.get(int(deg), 0))}"
            for deg in [int(x) for x in degrees]
        )

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    partition_name = str(anchor_partition or '') if int(su_type) == 19 else None
    if int(su_type) == 19 and partition_name not in {'ether', 'thio'}:
        return H, [], {
            "scores": {},
            "threshold": 0.0,
            "reason": "missing_partition",
        }

    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H))
    meta_cur = _clone_special_degree_meta(adjuster._get_special_degree_meta(H))
    single_bucket, double_bucket, single_degrees, double_degrees = _anchor_mode_bucket_names(int(su_type))
    if int(su_type) == 19 and str(partition_name) == 'thio':
        single_bucket = 'thio_single'
        double_bucket = 'thio_double'
    allowed_single = [int(x) for x in list(single_degrees or degrees)]
    allowed_double = [int(x) for x in list(double_degrees or [])]
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, int(su_type))
    active_mode_names = [f"single_d{int(deg)}" for deg in allowed_single]
    active_mode_names.extend([f"double_d{int(deg)}" for deg in allowed_double])
    active_scores = {
        str(name): dict(mode_scores.get(str(name), {}) or {})
        for name in active_mode_names
        if str(name) in mode_scores
    }
    total_abs = float(sum(float(v.get("abs", abs(float(v.get("score", 0.0))))) for v in active_scores.values()))
    thr = float(peak_rel_threshold) * max(1e-9, total_abs)

    for mode_name in active_mode_names:
        stats = active_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"  SU{int(su_type)} {str(mode_name)}@{float(stats.get('mu', 0.0)):.3f} "
            f"[{float(stats.get('lo', 0.0)):.3f},{float(stats.get('hi', 0.0)):.3f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    print(f"  threshold={thr:.3f}")

    single_counts = {
        int(deg): _mode_bucket_count(anchor_mode_meta, int(su_type), str(single_bucket), int(deg))
        for deg in allowed_single
    }
    double_counts = {
        int(deg): _mode_bucket_count(anchor_mode_meta, int(su_type), str(double_bucket), int(deg))
        for deg in allowed_double
    }
    if int(sum(single_counts.values()) + sum(double_counts.values())) <= 0:
        return H, [], {
            "scores": _anchor_mode_stats_meta(active_scores),
            "threshold": float(thr),
            "reason": "no_active_nodes",
            "special_degree_meta": dict(meta_cur),
        }
    print(f"  当前single数量: {_counts_view(single_counts)}")
    if allowed_double:
        print(f"  当前double数量: {_counts_view(double_counts)}")
    single_signal = max(
        abs(float(active_scores.get(f"single_d{int(deg)}", {}).get("score", 0.0)))
        for deg in allowed_single
    ) if allowed_single else 0.0
    double_signal = (
        max(
            abs(float(active_scores.get(f"double_d{int(deg)}", {}).get("score", 0.0)))
            for deg in allowed_double
        ) if allowed_double else 0.0
    )
    if max(float(single_signal), float(double_signal)) < float(thr):
        print("  峰强不足，跳过内部连接度调整")
        return H, [], {
            "scores": _anchor_mode_stats_meta(active_scores),
            "threshold": float(thr),
            "reason": "low_signal",
            "special_degree_meta": dict(meta_cur),
        }

    H_new = torch.clamp(H, min=0).long().clone()
    moves: List[Dict[str, Any]] = []
    meta_new = _clone_special_degree_meta(meta_cur)
    mode_new = _clone_special_anchor_mode_meta(anchor_mode_meta)

    def _apply_degree_delta(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        cur = dict(meta_local.get(int(su_type), {}) or {})
        cur[int(degree_i)] = max(0, int(cur.get(int(degree_i), 0)) + int(delta))
        meta_local[int(su_type)] = cur

    def _buffer_su_for_degree(degree_i: int) -> Optional[int]:
        return _mode_tail_su(int(su_type), int(degree_i))

    for _ in range(max(0, int(max_moves))):
        candidates: List[Tuple[float, str, int, int]] = []
        if int(sum(single_counts.values())) > 0:
            inc_single = max(
                allowed_single,
                key=lambda d: float(active_scores.get(f"single_d{int(d)}", {}).get("score", 0.0)),
            )
            dec_single_candidates = [int(d) for d in allowed_single if int(single_counts.get(int(d), 0)) > 0]
            dec_single = max(
                dec_single_candidates,
                key=lambda d: float(-active_scores.get(f"single_d{int(d)}", {}).get("score", 0.0)),
            ) if dec_single_candidates else None
            if inc_single is not None and dec_single is not None and int(inc_single) != int(dec_single):
                gain = float(
                    active_scores.get(f"single_d{int(inc_single)}", {}).get("score", 0.0) -
                    active_scores.get(f"single_d{int(dec_single)}", {}).get("score", 0.0)
                )
                candidates.append((float(gain), str(single_bucket), int(dec_single), int(inc_single)))
        if allowed_double and int(sum(double_counts.values())) > 0:
            inc_double = max(
                allowed_double,
                key=lambda d: float(active_scores.get(f"double_d{int(d)}", {}).get("score", 0.0)),
            )
            dec_double_candidates = [int(d) for d in allowed_double if int(double_counts.get(int(d), 0)) > 0]
            dec_double = max(
                dec_double_candidates,
                key=lambda d: float(-active_scores.get(f"double_d{int(d)}", {}).get("score", 0.0)),
            ) if dec_double_candidates else None
            if inc_double is not None and dec_double is not None and int(inc_double) != int(dec_double):
                gain = float(
                    active_scores.get(f"double_d{int(inc_double)}", {}).get("score", 0.0) -
                    active_scores.get(f"double_d{int(dec_double)}", {}).get("score", 0.0)
                )
                candidates.append((float(gain), str(double_bucket), int(dec_double), int(inc_double)))
        if not candidates:
            break
        _, bucket_name, dec_degree, inc_degree = max(candidates, key=lambda item: (float(item[0]), item[3], -item[2]))
        H_try = H_new.clone()
        meta_try = _clone_special_degree_meta(meta_new)
        mode_try = _clone_special_anchor_mode_meta(mode_new)
        src_tail = _buffer_su_for_degree(int(dec_degree))
        dst_tail = _buffer_su_for_degree(int(inc_degree))
        if src_tail is None or dst_tail is None:
            break
        if int(H_try[int(src_tail)].item()) <= 0:
            break
        H_try[int(src_tail)] -= 1
        H_try[int(dst_tail)] += 1
        if str(bucket_name) == str(single_bucket):
            counts_before = {int(deg): int(single_counts.get(int(deg), 0)) for deg in allowed_single}
            single_after = dict(single_counts)
            single_after[int(dec_degree)] = max(0, int(single_after.get(int(dec_degree), 0)) - 1)
            single_after[int(inc_degree)] = int(single_after.get(int(inc_degree), 0)) + 1
            counts_after = {int(deg): int(single_after.get(int(deg), 0)) for deg in allowed_single}
        else:
            counts_before = {int(deg): int(double_counts.get(int(deg), 0)) for deg in allowed_double}
            double_after = dict(double_counts)
            double_after[int(dec_degree)] = max(0, int(double_after.get(int(dec_degree), 0)) - 1)
            double_after[int(inc_degree)] = int(double_after.get(int(inc_degree), 0)) + 1
            counts_after = {int(deg): int(double_after.get(int(deg), 0)) for deg in allowed_double}
        _apply_mode_bucket_delta(mode_try, int(su_type), str(bucket_name), int(dec_degree), -1)
        _apply_mode_bucket_delta(mode_try, int(su_type), str(bucket_name), int(inc_degree), +1)
        _apply_degree_delta(meta_try, int(dec_degree), -1)
        _apply_degree_delta(meta_try, int(inc_degree), +1)
        violation = _special_mode_fixed_anchor_pool_violation(
            H_try,
            mode_try,
            int(su_type),
            str(bucket_name),
            int(inc_degree),
        )
        if violation is not None:
            active_scores[f"{'double' if str(bucket_name) == str(double_bucket) else 'single'}_d{int(inc_degree)}"]["score"] = -abs(
                float(active_scores.get(
                    f"{'double' if str(bucket_name) == str(double_bucket) else 'single'}_d{int(inc_degree)}",
                    {},
                ).get("score", 0.0))
            )
            continue
        H_new = H_try
        meta_new = meta_try
        mode_new = mode_try
        if str(bucket_name) == str(single_bucket):
            single_counts = dict(counts_after)
        else:
            double_counts = dict(counts_after)
        moves.append({
            "op": f"mode_su{int(su_type)}",
            "bucket": str(bucket_name),
            "from_degree": int(dec_degree),
            "to_degree": int(inc_degree),
            "tail_from": int(src_tail),
            "tail_to": int(dst_tail),
            "counts_before": dict(counts_before),
            "counts_after": dict(counts_after),
        })
        prefix = 'double' if str(bucket_name) == str(double_bucket) else 'single'
        inc_key = f"{prefix}_d{int(inc_degree)}"
        dec_key = f"{prefix}_d{int(dec_degree)}"
        if inc_key in active_scores:
            active_scores[inc_key]["score"] = float(active_scores[inc_key]["score"]) * 0.75
        if dec_key in active_scores:
            active_scores[dec_key]["score"] = float(active_scores[dec_key]["score"]) * 0.75

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        meta_new,
        anchor_mode_meta=mode_new,
    )
    print(f"  调整后single数量: {_counts_view(single_counts)}")
    if allowed_double:
        print(f"  调整后double数量: {_counts_view(double_counts)}")
    return H_new.clone(), moves, {
        "scores": _anchor_mode_stats_meta(active_scores),
        "threshold": float(thr),
        "special_degree_meta": dict(meta_new),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(mode_new),
        "counts_before": {
            "single": {int(deg): int(_mode_bucket_count(anchor_mode_meta, int(su_type), str(single_bucket), int(deg))) for deg in allowed_single},
            "double": {int(deg): int(_mode_bucket_count(anchor_mode_meta, int(su_type), str(double_bucket), int(deg))) for deg in allowed_double},
        },
        "counts_after": {
            "single": {int(deg): int(single_counts.get(int(deg), 0)) for deg in allowed_single},
            "double": {int(deg): int(double_counts.get(int(deg), 0)) for deg in allowed_double},
        },
    }


def adjust_special_anchor_count_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    su_type: int,
    nodes: Optional[List[Any]] = None,
    anchor_partition: Optional[str] = None,
    max_moves: int = 4,
    peak_rel_threshold: float = 0.01,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    label = f"SU{int(su_type)}单/双固定外接数量"
    print(f"\n[{label}调整] 基于真实1-hop与差谱分析")
    if int(su_type) == 21:
        print("  SU21 只能连接一个 SU32 端基，不存在 single/double 数量转换，跳过")
        return H, [], {'reason': 'su21_double_mode_not_applicable'}
    if nodes is None:
        print("  缺少 nodes，跳过调整")
        return H, [], {'reason': 'missing_nodes'}
    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {'reason': 'missing_diff'}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {'reason': 'empty_diff'}

    partition_name = str(anchor_partition or 'ether') if int(su_type) == 19 else None
    if int(su_type) == 19:
        has_active_nodes = any(
            int(getattr(n, 'su_type', -1)) == 19 and
            str(getattr(n, 'special_anchor_partition', None) or '') == str(partition_name)
            for n in list(nodes or [])
        )
    else:
        has_active_nodes = any(
            int(getattr(n, 'su_type', -1)) == int(su_type)
            for n in list(nodes or [])
        )
    if not bool(has_active_nodes):
        return H, [], {'reason': 'no_active_nodes'}

    counts = _collect_anchor_mode_counts(adjuster, nodes, int(su_type), anchor_partition=partition_name)
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, int(su_type))
    if not mode_scores:
        return H, [], {'reason': 'missing_mode_stats', 'counts_before': dict(counts)}

    lo23, hi23, _ = adjuster._get_su_common_window(23, fallback_mu=29.48, min_half_width=6.0)
    lo24, hi24, _ = adjuster._get_su_common_window(24, fallback_mu=39.97, min_half_width=6.0)
    tail_scores = {
        23: adjuster._window_stats(ppm_arr, diff_arr, lo23, hi23),
        24: adjuster._window_stats(ppm_arr, diff_arr, lo24, hi24),
    }
    tail_net = {23: float(tail_scores[23]['pos']) - float(tail_scores[23]['neg']),
                24: float(tail_scores[24]['pos']) - float(tail_scores[24]['neg'])}

    band_abs = float(sum(float(v.get('abs', 0.0)) for v in mode_scores.values())) + float(tail_scores[23]['abs']) + float(tail_scores[24]['abs'])
    thr = float(peak_rel_threshold) * max(1e-8, band_abs)
    print(f"  当前数量: {counts}")
    for mode_name in ('single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'):
        stats = mode_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"  {str(mode_name)}@{float(stats.get('mu', 0.0)):.3f} "
            f"[{float(stats.get('lo', 0.0)):.3f},{float(stats.get('hi', 0.0)):.3f}] "
            f"score={float(stats.get('score', 0.0)):.3f}"
        )
    print(f"  tail23={float(tail_net[23]):.3f}, tail24={float(tail_net[24]):.3f}, threshold={float(thr):.3f}")

    H_new = torch.clamp(H, min=0).long().clone()
    degree_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(_build_anchor_mode_meta_from_nodes(adjuster, nodes))
    fixed_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
    moves: List[Dict[str, Any]] = []

    move_specs = [
        ('single_d2', 'single_d2', 'double_d2', 23, 'forward_sd2_sd2_to_dd2'),
    ]

    def _mode_count(meta_local: Dict[int, Dict[str, Dict[int, int]]], mode_name: str) -> int:
        if int(su_type) == 19:
            single_bucket = 'thio_single' if str(partition_name) == 'thio' else 'ether_single'
            double_bucket = 'thio_double' if str(partition_name) == 'thio' else 'ether_double'
            if str(mode_name).startswith('single_'):
                deg = int(str(mode_name).split('d')[-1])
                return int(dict(meta_local.get(19, {}).get(single_bucket, {}) or {}).get(int(deg), 0))
            if str(mode_name).startswith('double_'):
                deg = int(str(mode_name).split('d')[-1])
                return int(dict(meta_local.get(19, {}).get(double_bucket, {}) or {}).get(int(deg), 0))
        if int(su_type) in {20, 21}:
            if str(mode_name).startswith('single_'):
                deg = int(str(mode_name).split('d')[-1])
                return int(dict(meta_local.get(int(su_type), {}).get('single', {}) or {}).get(int(deg), 0))
            if str(mode_name).startswith('double_'):
                deg = int(str(mode_name).split('d')[-1])
                return int(dict(meta_local.get(int(su_type), {}).get('double', {}) or {}).get(int(deg), 0))
        return 0

    def _apply_mode_delta(meta_local: Dict[int, Dict[str, Dict[int, int]]],
                          mode_name: str,
                          delta: int) -> None:
        deg = int(str(mode_name).split('d')[-1])
        if int(su_type) == 19:
            if str(partition_name) == 'thio':
                bucket = 'thio_double' if str(mode_name).startswith('double_') else 'thio_single'
            else:
                bucket = 'ether_double' if str(mode_name).startswith('double_') else 'ether_single'
            cur = dict(meta_local.get(19, {}).get(bucket, {}) or {})
            cur[int(deg)] = max(0, int(cur.get(int(deg), 0)) + int(delta))
            meta_local.setdefault(19, {})[str(bucket)] = cur
        elif int(su_type) in {20, 21}:
            bucket = 'double' if str(mode_name).startswith('double_') else 'single'
            cur = dict(meta_local.get(int(su_type), {}).get(bucket, {}) or {})
            cur[int(deg)] = max(0, int(cur.get(int(deg), 0)) + int(delta))
            meta_local.setdefault(int(su_type), {})[str(bucket)] = cur

    def _apply_degree_delta(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        cur = dict(meta_local.get(int(su_type), {}) or {})
        cur[int(degree_i)] = max(0, int(cur.get(int(degree_i), 0)) + int(delta))
        meta_local[int(su_type)] = cur

    def _sync_fixed_meta() -> None:
        nonlocal fixed_meta
        ether_single = {}
        ether_double = {}
        thio_single = {}
        thio_double = {}
        fixed_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
        fixed_meta['special_degree_meta'] = _clone_special_degree_meta(degree_meta)
        fixed_meta['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(su_type) == 19:
            ether_single = dict(anchor_mode_meta.get(19, {}).get('ether_single', {}) or {})
            ether_double = dict(anchor_mode_meta.get(19, {}).get('ether_double', {}) or {})
            thio_single = dict(anchor_mode_meta.get(19, {}).get('thio_single', {}) or {})
            thio_double = dict(anchor_mode_meta.get(19, {}).get('thio_double', {}) or {})
            ether_total_by_degree = {
                int(deg): int(ether_single.get(int(deg), 0)) + int(ether_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            thio_total_by_degree = {
                int(deg): int(thio_single.get(int(deg), 0)) + int(thio_double.get(int(deg), 0))
                for deg in [1, 2, 3]
            }
            fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
            fixed_meta['special_partition_meta'][19] = {
                'ether': {int(deg): int(ether_total_by_degree.get(int(deg), 0)) for deg in [1, 2, 3]},
                'thio': {int(deg): int(thio_total_by_degree.get(int(deg), 0)) for deg in [1, 2, 3]},
            }
            fixed_meta['o_base_19'] = int(sum(int(v) for v in ether_total_by_degree.values()))
            fixed_meta['n19_total'] = int(H_new[19].item())
        adjuster.fixed_partition_meta = dict(fixed_meta)
        try:
            adjuster._set_special_degree_meta(H_new, degree_meta)
        except Exception:
            pass
        fixed_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
        fixed_meta['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(su_type) == 19:
            fixed_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
            fixed_meta['special_partition_meta'][19] = {
                'ether': {
                    int(deg): int(ether_single.get(int(deg), 0)) + int(ether_double.get(int(deg), 0))
                    for deg in [1, 2, 3]
                },
                'thio': {
                    int(deg): int(thio_single.get(int(deg), 0)) + int(thio_double.get(int(deg), 0))
                    for deg in [1, 2, 3]
                },
            }
            fixed_meta['o_base_19'] = int(sum(
                int(v) for v in dict(anchor_mode_meta.get(19, {}).get('ether_single', {}) or {}).values()
            ) + sum(
                int(v) for v in dict(anchor_mode_meta.get(19, {}).get('ether_double', {}) or {}).values()
            ))
            fixed_meta['s_reserved_19'] = int(sum(
                int(v) for v in dict(anchor_mode_meta.get(19, {}).get('thio_single', {}) or {}).values()
            ) + sum(
                int(v) for v in dict(anchor_mode_meta.get(19, {}).get('thio_double', {}) or {}).values()
            ))
        adjuster.fixed_partition_meta = dict(fixed_meta)
        try:
            if getattr(adjuster, 'layer0_estimator', None) is not None:
                layer0_meta = dict(getattr(adjuster.layer0_estimator, 'fixed_partition_meta', {}) or {})
                layer0_meta.update(dict(fixed_meta))
                adjuster.layer0_estimator.fixed_partition_meta = dict(layer0_meta)
                adjuster.layer0_estimator.special_partition_meta = dict(fixed_meta.get('special_partition_meta', {}) or {})
        except Exception:
            pass

    def _current_counts_view() -> Dict[str, int]:
        return {
            'single_d2': int(_mode_count(anchor_mode_meta, 'single_d2')),
            'single_d3': int(_mode_count(anchor_mode_meta, 'single_d3')),
            'double_d2': int(_mode_count(anchor_mode_meta, 'double_d2')),
            'double_d3': int(_mode_count(anchor_mode_meta, 'double_d3')),
        }

    for _ in range(max(0, int(max_moves))):
        best: Optional[Tuple[float, str, Tuple[str, str, str, int, str]]] = None
        cur_counts = _current_counts_view()
        for mode_a, mode_b, mode_t, tail_su, op_name in move_specs:
            cnt_a = int(cur_counts.get(str(mode_a), 0))
            cnt_b = int(cur_counts.get(str(mode_b), 0))
            cnt_t = int(cur_counts.get(str(mode_t), 0))
            raw_a = float(mode_scores.get(str(mode_a), {}).get('score', 0.0))
            raw_b = float(mode_scores.get(str(mode_b), {}).get('score', 0.0))
            raw_t = float(mode_scores.get(str(mode_t), {}).get('score', 0.0))
            tail_raw = float(tail_net.get(int(tail_su), 0.0))

            avail_fwd = int(cnt_a // 2) if str(mode_a) == str(mode_b) else int(min(cnt_a, cnt_b))
            if int(avail_fwd) > 0:
                desirability = (
                    max(0.0, -float(raw_a)) +
                    max(0.0, -float(raw_b)) +
                    max(0.0, float(raw_t)) +
                    0.35 * max(0.0, float(tail_raw))
                )
                opposition = (
                    max(0.0, float(raw_a)) +
                    max(0.0, float(raw_b)) +
                    max(0.0, -float(raw_t)) +
                    0.35 * max(0.0, -float(tail_raw))
                )
                net = float(desirability - opposition)
                if best is None or float(net) > float(best[0]):
                    best = (float(net), 'forward', (mode_a, mode_b, mode_t, int(tail_su), str(op_name)))

            if int(cnt_t) > 0 and int(H_new[int(tail_su)].item()) > 0:
                desirability = (
                    max(0.0, float(raw_a)) +
                    max(0.0, float(raw_b)) +
                    max(0.0, -float(raw_t)) +
                    0.35 * max(0.0, -float(tail_raw))
                )
                opposition = (
                    max(0.0, -float(raw_a)) +
                    max(0.0, -float(raw_b)) +
                    max(0.0, float(raw_t)) +
                    0.35 * max(0.0, float(tail_raw))
                )
                net = float(desirability - opposition)
                if best is None or float(net) > float(best[0]):
                    best = (float(net), 'reverse', (mode_a, mode_b, mode_t, int(tail_su), str(op_name)))

        if best is None or float(best[0]) <= float(thr):
            break

        _, direction, (mode_a, mode_b, mode_t, tail_su, op_name) = best
        counts_before = dict(_current_counts_view())
        degree_before = {int(k): int(v) for k, v in dict(degree_meta.get(int(su_type), {}) or {}).items()}

        if str(direction) == 'forward':
            _apply_mode_delta(anchor_mode_meta, str(mode_a), -1)
            _apply_mode_delta(anchor_mode_meta, str(mode_b), -1)
            _apply_mode_delta(anchor_mode_meta, str(mode_t), +1)
            H_new[int(su_type)] -= 1
            H_new[int(tail_su)] += 1
            if str(mode_a) == 'single_d2' and str(mode_b) == 'single_d2':
                _apply_degree_delta(degree_meta, 2, -1)
            elif str(mode_a) == 'single_d3' and str(mode_b) == 'single_d3':
                _apply_degree_delta(degree_meta, 3, -1)
        else:
            _apply_mode_delta(anchor_mode_meta, str(mode_a), +1)
            _apply_mode_delta(anchor_mode_meta, str(mode_b), +1)
            _apply_mode_delta(anchor_mode_meta, str(mode_t), -1)
            H_new[int(su_type)] += 1
            H_new[int(tail_su)] -= 1
            if str(mode_a) == 'single_d2' and str(mode_b) == 'single_d2':
                _apply_degree_delta(degree_meta, 2, +1)
            elif str(mode_a) == 'single_d3' and str(mode_b) == 'single_d3':
                _apply_degree_delta(degree_meta, 3, +1)

        _sync_fixed_meta()
        counts_after = dict(_current_counts_view())
        moves.append({
            'op': f"{str(op_name)}[{str(direction)}]",
            'stage': f"anchor_count_su{int(su_type)}",
            'delta': {int(su_type): (-1 if str(direction) == 'forward' else +1), int(tail_su): (+1 if str(direction) == 'forward' else -1)},
            'counts_before': dict(counts_before),
            'counts_after': dict(counts_after),
            'degree_before': dict(degree_before),
            'degree_after': {int(k): int(v) for k, v in dict(degree_meta.get(int(su_type), {}) or {}).items()},
        })
        mode_scores[str(mode_a)]['score'] = float(mode_scores[str(mode_a)]['score']) * 0.85
        mode_scores[str(mode_b)]['score'] = float(mode_scores[str(mode_b)]['score']) * 0.85
        mode_scores[str(mode_t)]['score'] = float(mode_scores[str(mode_t)]['score']) * 0.85

    return H_new, moves, {
        'scores': {str(k): dict(v) for k, v in mode_scores.items()},
        'tail_scores': {int(k): float(v) for k, v in tail_net.items()},
        'threshold': float(thr),
        'counts_before': dict(counts),
        'counts_after': dict(_current_counts_view()),
        'special_degree_meta': _clone_special_degree_meta(degree_meta),
        'special_anchor_mode_meta': _clone_special_anchor_mode_meta(anchor_mode_meta),
    }


def adjust_block_b_hetero_anchor_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    max_moves_each: int = 3,
    max_moves_total: Optional[int] = None,
    max_moves_count: Optional[int] = None,
    max_moves_mode: Optional[int] = None,
    peak_rel_threshold: float = 0.01,
    substage: Optional[str] = None,
    nodes: Optional[List[Any]] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[Block B] 异原子锚点联合调整")
    H_work = torch.clamp(H, min=0).long().clone()
    all_moves: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    max_moves_each = min(int(max_moves_each), 6)
    total_stage_moves = min(int(max_moves_total if max_moves_total is not None else max_moves_each), 6)
    count_stage_moves = min(int(max_moves_count if max_moves_count is not None else max_moves_each), 4)
    mode_stage_moves = min(int(max_moves_mode if max_moves_mode is not None else max_moves_each), 4)
    substage_name = str(substage).strip().lower() if substage is not None else None

    subcalls = [
        ("ether_2829", adjust_ether_2829_conversion_impl, {
            "max_moves": int(total_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
            "min_keep_28": 0,
            "min_keep_29": 0,
            "min_keep_5": 0,
        }),
        ("ether", adjust_ether_519_by_difference_impl, {
            "max_moves": int(total_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
            "min_keep": 1,
            "reserved_19": max(0, int(2 * H_work[31].item()) - int(H_work[7].item())),
        }),
        ("ether_count19", adjust_special_anchor_count_impl, {
            "su_type": 19,
            "nodes": nodes,
            "max_moves": int(count_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("ether_mode19", adjust_special_degree_mode_impl, {
            "su_type": 19,
            "anchor_partition": "ether",
            "degrees": [1, 2, 3],
            "max_moves": int(mode_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("amine", adjust_amine_620_by_difference_impl, {
            "max_moves": int(total_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
            "min_keep": 0,
        }),
        ("amine_count20", adjust_special_anchor_count_impl, {
            "su_type": 20,
            "nodes": nodes,
            "max_moves": int(count_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("amine_mode20", adjust_special_degree_mode_impl, {
            "su_type": 20,
            "degrees": [1, 2, 3],
            "max_moves": int(mode_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("thioether", adjust_thioether_719_by_difference_impl, {
            "max_moves": int(total_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
            "min_keep": 0,
        }),
        ("thio_count19", adjust_special_anchor_count_impl, {
            "su_type": 19,
            "anchor_partition": "thio",
            "nodes": nodes,
            "max_moves": int(count_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("thio_mode19", adjust_special_degree_mode_impl, {
            "su_type": 19,
            "anchor_partition": "thio",
            "degrees": [1, 2, 3],
            "max_moves": int(mode_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
        ("halogen", adjust_halogen_821_by_difference_impl, {
            "max_moves": int(total_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
            "min_keep": 0,
        }),
        ("halogen_mode21", adjust_special_degree_mode_impl, {
            "su_type": 21,
            "degrees": [2, 3],
            "max_moves": int(mode_stage_moves),
            "peak_rel_threshold": float(peak_rel_threshold),
        }),
    ]
    for name, fn, kwargs in subcalls:
        if substage_name is not None and str(name) != str(substage_name):
            continue
        partition_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
        before_balance = _summarize_block_b_balance(H_work, fixed_partition_meta=partition_meta)
        _print_anchor_mode_snapshot(
            H_work,
            fixed_partition_meta=partition_meta,
            header=f"[Block B/{name}] 模式快照(调整前)",
            indent="  ",
        )
        H_work, moves, submeta = fn(
            adjuster,
            H_work,
            ppm,
            diff,
            **kwargs,
        )
        # Substage helpers may already have synchronized derived metadata
        # (especially special_anchor_mode_meta / special_partition_meta) onto
        # the adjuster. Start from that fresh state instead of the pre-call
        # snapshot, then overlay explicit fields returned in submeta.
        after_partition_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or partition_meta)
        if isinstance(submeta, dict):
            if 's_reserved_19' in submeta:
                after_partition_meta['s_reserved_19'] = int(submeta.get('s_reserved_19', 0))
            if 'o_base_19' in submeta:
                after_partition_meta['o_base_19'] = int(submeta.get('o_base_19', 0))
            if 'special_anchor_mode_meta' in submeta:
                after_partition_meta['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(
                    submeta.get('special_anchor_mode_meta')
                )
            if 'special_partition_meta' in submeta:
                after_partition_meta['special_partition_meta'] = {
                    int(su): {
                        str(part): {int(deg): int(cnt) for deg, cnt in dict(part_counts or {}).items()}
                        for part, part_counts in dict(parts or {}).items()
                    }
                    for su, parts in dict(submeta.get('special_partition_meta') or {}).items()
                }
            thio_meta = dict(submeta.get('thio_meta', {}) or {})
            ether_meta = dict(submeta.get('ether_meta', {}) or {})
            if 's_reserved_19' in thio_meta:
                after_partition_meta['s_reserved_19'] = int(thio_meta.get('s_reserved_19', 0))
            if 'o_base_19' in ether_meta:
                after_partition_meta['o_base_19'] = int(ether_meta.get('o_base_19', 0))
        if 's_reserved_19' in after_partition_meta:
            after_partition_meta['o_base_19'] = max(
                0,
                int(H_work[19].item()) - int(after_partition_meta.get('s_reserved_19', 0)),
            )
        if isinstance(submeta, dict) and 'special_degree_meta' in submeta:
            after_partition_meta['special_degree_meta'] = _clone_special_degree_meta(submeta.get('special_degree_meta'))
        else:
            after_partition_meta['special_degree_meta'] = _clone_special_degree_meta(
                adjuster._get_special_degree_meta(H_work)
            )
        after_partition_meta['n19_total'] = int(H_work[19].item())
        adjuster.fixed_partition_meta = dict(after_partition_meta)
        try:
            adjuster._set_special_degree_meta(H_work, after_partition_meta.get('special_degree_meta', {}))
        except Exception:
            pass
        try:
            fixed_meta_after_degree = dict(getattr(adjuster, 'fixed_partition_meta', {}) or {})
            if 'special_anchor_mode_meta' in after_partition_meta:
                fixed_meta_after_degree['special_anchor_mode_meta'] = _clone_special_anchor_mode_meta(
                    after_partition_meta.get('special_anchor_mode_meta')
                )
            if 'special_partition_meta' in after_partition_meta:
                fixed_meta_after_degree['special_partition_meta'] = after_partition_meta.get('special_partition_meta')
            adjuster.fixed_partition_meta = dict(fixed_meta_after_degree)
            if getattr(adjuster, 'layer0_estimator', None) is not None:
                layer0_meta = dict(getattr(adjuster.layer0_estimator, 'fixed_partition_meta', {}) or {})
                layer0_meta.update(dict(fixed_meta_after_degree))
                adjuster.layer0_estimator.fixed_partition_meta = dict(layer0_meta)
                adjuster.layer0_estimator.special_partition_meta = dict(
                    fixed_meta_after_degree.get('special_partition_meta', {}) or {}
                )
        except Exception:
            pass
        after_partition_meta = dict(getattr(adjuster, 'fixed_partition_meta', {}) or after_partition_meta)
        after_balance = _summarize_block_b_balance(H_work, fixed_partition_meta=after_partition_meta)
        _print_anchor_mode_snapshot(
            H_work,
            fixed_partition_meta=after_partition_meta,
            header=f"[Block B/{name}] 模式快照(调整后)",
            indent="  ",
        )
        meta[name] = submeta
        if isinstance(meta[name], dict):
            meta[name]["balance_before"] = dict(before_balance)
            meta[name]["balance_after"] = dict(after_balance)
        for mv in moves:
            tagged = dict(mv)
            tagged["block"] = "B"
            tagged["substage"] = name
            all_moves.append(tagged)
    return H_work, all_moves, meta


def adjust_ether_2829_conversion_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    peak_rel_threshold: float = 0.01,
    max_moves: int = 3,
    min_keep_28: int = 0,
    min_keep_29: int = 0,
    min_keep_5: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[Ether预调节(28/29)] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    s5 = _aromatic_anchor_score(adjuster, ppm_arr, diff_arr, 5, fallback_mu=154.75)
    mode19_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, 19)
    s19_total = _score_special_mode_total(
        mode19_scores,
        ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'],
    )

    band_abs = float(s5["abs"]) + float(s19_total["abs"])
    thr = float(peak_rel_threshold) * max(1e-8, band_abs)

    need_5 = float(s5["pos"])
    excess_5 = float(s5["neg_abs"])
    need_19 = float(s19_total["pos"])
    excess_19 = float(s19_total["neg_abs"])
    need_total = float(need_5 + need_19)
    excess_total = float(excess_5 + excess_19)
    degree_scores = _special_mode_scores(adjuster, ppm_arr, diff_arr, 19, [1, 2, 3])
    need_d = {int(d): max(0.0, float(stats.get('score', 0.0))) for d, stats in degree_scores.items()}
    excess_d = {int(d): max(0.0, -float(stats.get('score', 0.0))) for d, stats in degree_scores.items()}

    print(
        f"  combined need={need_total:.3f}, excess={excess_total:.3f}, threshold={thr:.3f}"
    )
    print(
        f"  SU5 band {float(s5['lo']):.1f}-{float(s5['hi']):.1f}: "
        f"need={need_5:.3f}, excess={excess_5:.3f}"
    )
    print(
        f"  SU19 modes total: need={need_19:.3f}, excess={excess_19:.3f}"
    )
    for mode_name in ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3']:
        stats = mode19_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"    {mode_name} [{float(stats.get('lo', 0.0)):.1f},{float(stats.get('hi', 0.0)):.1f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    for degree_i in [1, 2, 3]:
        stats = degree_scores.get(int(degree_i), {})
        print(
            f"  SU19(d{int(degree_i)}) core@{float(stats.get('mu', 0.0)):.3f} "
            f"[{float(stats.get('core_lo', 0.0)):.3f},{float(stats.get('core_hi', 0.0)):.3f}] "
            f"score={float(stats.get('score', 0.0)):.3f}"
        )

    if max(float(need_total), float(excess_total)) < float(thr):
        print("  峰强不足，跳过调整")
        return H, [], {
            "scores": {
                "need_total": float(need_total),
                "excess_total": float(excess_total),
                "5_need": float(need_5),
                "5_excess": float(excess_5),
                "19_need": float(need_19),
                "19_excess": float(excess_19),
            },
            "threshold": float(thr),
        }

    H_new = torch.clamp(H, min=0).long().clone()
    moves: List[Dict[str, Any]] = []
    special_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H_new))

    virtual_need_5 = float(need_5)
    virtual_need_19 = float(need_19)
    virtual_excess_5 = float(excess_5)
    virtual_excess_19 = float(excess_19)

    def _pick_5_source(hh: torch.Tensor) -> Optional[int]:
        if int(hh[11].item()) > 0:
            return 11
        if int(hh[13].item()) > 0:
            return 13
        return None

    def _single19_ether_count(mode_meta_local: Dict[int, Dict[str, Dict[int, int]]], degree_i: int) -> int:
        return int(_mode_bucket_count(mode_meta_local, 19, 'ether_single', int(degree_i)))

    def _dec_single19_ether_to_tail(hh: torch.Tensor,
                                    degree_meta_local: Dict[int, Dict[int, int]],
                                    mode_meta_local: Dict[int, Dict[str, Dict[int, int]]],
                                    degree_i: int) -> List[Dict[str, Any]]:
        deg_i = int(degree_i)
        if int(_single19_ether_count(mode_meta_local, int(deg_i))) <= 0:
            return []
        tail_su = _mode_tail_su(19, int(deg_i))
        if tail_su is None:
            return []
        hh[19] -= 1
        hh[int(tail_su)] += 1
        counts_19 = dict(degree_meta_local.get(19, {}) or {})
        counts_19[int(deg_i)] = max(0, int(counts_19.get(int(deg_i), 0)) - 1)
        degree_meta_local[19] = counts_19
        _apply_mode_bucket_delta(mode_meta_local, 19, 'ether_single', int(deg_i), -1)
        return [{
            "op": f"dec_19_ether_single_d{int(deg_i)}",
            "from": 19,
            "to": int(tail_su),
            "degree": int(deg_i),
            "anchor_mode": "ether_single",
            "fixed_edges": 1,
        }]

    def _inc_single19_ether_from_tail(hh: torch.Tensor,
                                      degree_meta_local: Dict[int, Dict[int, int]],
                                      mode_meta_local: Dict[int, Dict[str, Dict[int, int]]],
                                      degree_i: int) -> List[Dict[str, Any]]:
        deg_i = int(degree_i)
        tail_su = _mode_tail_su(19, int(deg_i))
        if tail_su is None:
            return []
        if int(deg_i) == 3:
            donor_su = 24 if int(hh[24].item()) > 0 else (23 if int(hh[23].item()) > 0 else None)
        elif int(deg_i) == 1:
            donor_su = 22 if int(hh[22].item()) > 0 else None
        else:
            donor_su = 23 if int(hh[23].item()) > 0 else None
        if donor_su is None:
            return []
        hh[int(donor_su)] -= 1
        hh[19] += 1
        counts_19 = dict(degree_meta_local.get(19, {}) or {})
        counts_19[int(deg_i)] = max(0, int(counts_19.get(int(deg_i), 0)) + 1)
        degree_meta_local[19] = counts_19
        _apply_mode_bucket_delta(mode_meta_local, 19, 'ether_single', int(deg_i), +1)
        return [{
            "op": f"inc_19_ether_single_d{int(deg_i)}",
            "from": int(donor_su),
            "to": 19,
            "degree": int(deg_i),
            "anchor_mode": "ether_single",
            "fixed_edges": 1,
        }]

    def _dec_5_to_13(hh: torch.Tensor) -> List[Dict[str, Any]]:
        if int(hh[5].item()) <= int(min_keep_5):
            return []
        hh[5] -= 1
        hh[13] += 1
        return [{
            "op": "dec_5",
            "from": 5,
            "to": 13,
            "fixed_edges": 1,
        }]

    def _inc_5_from_11_or_13(hh: torch.Tensor) -> List[Dict[str, Any]]:
        src_5 = _pick_5_source(hh)
        if src_5 is None:
            return []
        hh[int(src_5)] -= 1
        hh[5] += 1
        return [{
            "op": "inc_5",
            "from": int(src_5),
            "to": 5,
            "fixed_edges": 1,
        }]

    def _pick_need_degree() -> int:
        ranked = sorted([1, 2, 3], key=lambda d: (float(need_d.get(int(d), 0.0)), -int(d)), reverse=True)
        return int(ranked[0]) if ranked else 2

    def _pick_excess_degree(mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> Optional[int]:
        ranked = sorted([1, 2, 3], key=lambda d: (float(excess_d.get(int(d), 0.0)), int(_single19_ether_count(mode_meta_local, int(d))), -int(d)), reverse=True)
        for deg_i in ranked:
            if int(_single19_ether_count(mode_meta_local, int(deg_i))) > 0:
                return int(deg_i)
        return None

    for _ in range(max(0, int(max_moves))):
        virtual_need_total = float(virtual_need_5 + virtual_need_19)
        virtual_excess_total = float(virtual_excess_5 + virtual_excess_19)
        if float(virtual_excess_total) <= float(thr) and float(virtual_need_total) <= float(thr):
            break

        reduce_direction = bool(float(virtual_excess_total) >= float(virtual_need_total))
        branch_order: List[str]
        if reduce_direction:
            branch_order = ['19', '5'] if float(virtual_excess_19) >= float(virtual_excess_5) else ['5', '19']
        else:
            branch_order = ['19', '5'] if float(virtual_need_19) >= float(virtual_need_5) else ['5', '19']

        step_done = False
        for branch in branch_order:
            H_try = H_new.clone()
            degree_try = _clone_special_degree_meta(special_meta)
            mode_try = _clone_special_anchor_mode_meta(anchor_mode_meta)
            step_moves: List[Dict[str, Any]] = []

            if reduce_direction:
                if int(H_try[29].item()) <= int(min_keep_29):
                    continue
                H_try[29] -= 1
                H_try[28] += 1
                step_moves.append({
                    "op": "29_to_28",
                    "from": 29,
                    "to": 28,
                    "ether_required_delta": -1,
                })
                if str(branch) == '19':
                    excess_deg = _pick_excess_degree(mode_try)
                    if excess_deg is None:
                        continue
                    branch_moves = _dec_single19_ether_to_tail(H_try, degree_try, mode_try, int(excess_deg))
                else:
                    branch_moves = _dec_5_to_13(H_try)
                if not branch_moves:
                    continue
                step_moves.extend(branch_moves)
            else:
                if int(H_try[28].item()) <= int(min_keep_28):
                    continue
                H_try[28] -= 1
                H_try[29] += 1
                step_moves.append({
                    "op": "28_to_29",
                    "from": 28,
                    "to": 29,
                    "ether_required_delta": +1,
                })
                if str(branch) == '19':
                    need_deg = _pick_need_degree()
                    branch_moves = _inc_single19_ether_from_tail(H_try, degree_try, mode_try, int(need_deg))
                else:
                    branch_moves = _inc_5_from_11_or_13(H_try)
                if not branch_moves:
                    continue
                step_moves.extend(branch_moves)

            H_new = H_try
            special_meta = degree_try
            anchor_mode_meta = mode_try
            moves.extend(step_moves)
            if reduce_direction:
                if str(branch) == '19':
                    virtual_excess_19 *= 0.72
                else:
                    virtual_excess_5 *= 0.72
            else:
                if str(branch) == '19':
                    virtual_need_19 *= 0.72
                else:
                    virtual_need_5 *= 0.72
            step_done = True
            break

        if not step_done:
            break

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        special_meta,
        anchor_mode_meta=anchor_mode_meta,
    )

    meta = {
        "scores": {
            "need_total": float(need_total),
            "excess_total": float(excess_total),
            "5_need": float(need_5),
            "5_excess": float(excess_5),
            "19_need": float(need_19),
            "19_excess": float(excess_19),
        },
        "windows": {
            "5": s5,
            "19_modes": _anchor_mode_stats_meta(mode19_scores),
        },
        "threshold": float(thr),
        "n_moves": int(len(moves)),
        "counts_before": {
            "H5": int(H[5].item()),
            "H19": int(H[19].item()),
            "H23": int(H[23].item()),
            "H13": int(H[13].item()),
            "H11": int(H[11].item()),
            "H28": int(H[28].item()),
            "H29": int(H[29].item()),
            "ether_single_d1": int(_single19_ether_count(_clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H)), 1)),
            "ether_single_d2": int(_single19_ether_count(_clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H)), 2)),
            "ether_single_d3": int(_single19_ether_count(_clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H)), 3)),
        },
        "counts_after": {
            "H5": int(H_new[5].item()),
            "H19": int(H_new[19].item()),
            "H23": int(H_new[23].item()),
            "H24": int(H_new[24].item()),
            "H22": int(H_new[22].item()),
            "H13": int(H_new[13].item()),
            "H11": int(H_new[11].item()),
            "H28": int(H_new[28].item()),
            "H29": int(H_new[29].item()),
            "ether_single_d1": int(_single19_ether_count(anchor_mode_meta, 1)),
            "ether_single_d2": int(_single19_ether_count(anchor_mode_meta, 2)),
            "ether_single_d3": int(_single19_ether_count(anchor_mode_meta, 3)),
        },
        "special_degree_meta": dict(special_meta),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(anchor_mode_meta),
        "degree_scores": {f"d{int(k)}": dict(v) for k, v in degree_scores.items()},
    }

    print(
        f"  完成 {len(moves)} 条变更记录 | "
        f"H[28]={int(H[28].item())}->{int(H_new[28].item())}, "
        f"H[29]={int(H[29].item())}->{int(H_new[29].item())}, "
        f"H[5]={int(H[5].item())}->{int(H_new[5].item())}, "
        f"H[19]={int(H[19].item())}->{int(H_new[19].item())}, "
        f"H[23]={int(H[23].item())}->{int(H_new[23].item())}, "
        f"ether_single_d1={int(meta['counts_before']['ether_single_d1'])}->{int(meta['counts_after']['ether_single_d1'])}, "
        f"ether_single_d2={int(meta['counts_before']['ether_single_d2'])}->{int(meta['counts_after']['ether_single_d2'])}, "
        f"ether_single_d3={int(meta['counts_before']['ether_single_d3'])}->{int(meta['counts_after']['ether_single_d3'])}"
    )
    return H_new, moves, meta


def adjust_ether_519_by_difference_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    window_19: float = 3.0,
    peak_rel_threshold: float = 0.01,
    max_moves: int = 5,
    min_keep: int = 1,
    reserved_19: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[O连接(5/19)调整] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}
    s5 = _aromatic_anchor_score(adjuster, ppm_arr, diff_arr, 5, fallback_mu=154.75)
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, 19)
    s19_total = _score_special_mode_total(
        mode_scores,
        ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'],
    )
    s23_tail = adjuster._window_stats(ppm_arr, diff_arr, 18.0, 35.0)
    s24_tail = adjuster._window_stats(ppm_arr, diff_arr, 32.0, 50.0)
    s25_tail = adjuster._window_stats(ppm_arr, diff_arr, 45.0, 65.0)

    band_abs = float(s5["abs"]) + float(s19_total["abs"])
    thr = float(peak_rel_threshold) * max(1e-8, band_abs)

    need_5 = float(s5["pos"])
    excess_5 = float(s5["neg_abs"])
    need_19 = float(s19_total["pos"])
    excess_19 = float(s19_total["neg_abs"])

    balance_before = _summarize_block_b_balance(H, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))
    W_ether = int(balance_before["ether_required"])
    reserved_19 = max(int(reserved_19), int(balance_before["sulfur_required_19"]))
    soft_floor_5 = max(int(min_keep), int(math.ceil(0.50 * float(W_ether))))
    soft_cap_19_edges = max(int(min_keep), int(math.ceil(0.35 * float(W_ether))))

    print(
        f"  SU5 band {float(s5['lo']):.1f}-{float(s5['hi']):.1f}: "
        f"need={need_5:.3f}, excess={excess_5:.3f}"
    )
    print(
        f"  SU19 modes total: need={need_19:.3f}, excess={excess_19:.3f}"
    )
    for mode_name in ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3']:
        stats = mode_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"    {mode_name} [{float(stats.get('lo', 0.0)):.1f},{float(stats.get('hi', 0.0)):.1f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    print(
        f"  threshold={thr:.3f}, W_ether={int(W_ether)}, "
        f"soft_floor_5={int(soft_floor_5)}, soft_cap_19_edges={int(soft_cap_19_edges)} (reserved_19={int(reserved_19)})"
    )

    if max(float(need_5), float(excess_5), float(need_19), float(excess_19)) < float(thr):
        print("  峰强不足，跳过调整")
        return H, [], {
            "scores": {
                "5_need": float(need_5), "5_excess": float(excess_5),
                "19_need": float(need_19), "19_excess": float(excess_19),
            },
            "windows": {
                "5": s5,
                "19_modes": _anchor_mode_stats_meta(mode_scores),
            },
            "threshold": float(thr),
            "reserved_19": int(reserved_19),
            "soft_floor_5": int(soft_floor_5),
            "soft_cap_19": int(soft_cap_19_edges),
        }

    H_new = torch.clamp(H, min=0).long().clone()
    moves: List[Dict[str, Any]] = []
    special_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H_new))
    single_bucket, double_bucket, _, _ = _anchor_mode_bucket_names(19)

    def _pick_5_donors(hh: torch.Tensor, num: int) -> Optional[List[int]]:
        donors: List[int] = []
        for _ in range(max(0, int(num))):
            if int(hh[13].item()) > 0:
                hh[13] -= 1
                donors.append(13)
            elif int(hh[11].item()) > 0:
                hh[11] -= 1
                donors.append(11)
            else:
                for su_idx in donors:
                    hh[int(su_idx)] += 1
                return None
        return donors

    def _restore_5_donors(hh: torch.Tensor, donors: List[int]) -> None:
        for su_idx in list(donors or []):
            hh[int(su_idx)] += 1

    def _add_degree_count(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        counts = dict(meta_local.get(19, {}) or {})
        counts[int(degree_i)] = max(0, int(counts.get(int(degree_i), 0)) + int(delta))
        meta_local[19] = counts

    def _ether_edge_total(mode_meta_local: Dict[int, Dict[str, Dict[int, int]]], hh: torch.Tensor) -> int:
        return int(hh[5].item()) + int(_fixed_anchor_edge_count(mode_meta_local, 19))
    soft_floor_5 = max(int(min_keep), int(math.ceil(0.50 * float(W_ether))))

    def _apply_shift_19_to_5(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_decrease(
            mode_scores,
            mode_meta_local,
            19,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, dec_degree, donor_num = picked
        donors = _pick_5_donors(hh, donor_num)
        if donors is None:
            return []
        tail_su = _mode_tail_su(19, int(dec_degree))
        if tail_su is None or int(hh[19].item()) <= 0:
            _restore_5_donors(hh, donors)
            return []
        hh[19] -= 1
        hh[int(tail_su)] += 1
        hh[5] += int(donor_num)
        _add_degree_count(degree_meta_local, int(dec_degree), -1)
        _apply_mode_bucket_delta(mode_meta_local, 19, bucket, int(dec_degree), -1)
        step_moves = [{
            "op": f"dec_19_{mode_name}",
            "from": 19,
            "to": int(tail_su),
            "degree": int(dec_degree),
            "fixed_edges": int(donor_num),
            "reserved_19": int(reserved_19),
        }]
        for donor_su in list(donors or []):
            step_moves.append({
                "op": "inc_5",
                "from": int(donor_su),
                "to": 5,
            })
        return step_moves

    def _apply_shift_5_to_19(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_increase(
            mode_scores,
            hh,
            5,
            19,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, inc_degree, h5_cost = picked
        tail_su = _mode_tail_su(19, int(inc_degree))
        if tail_su is None or int(hh[int(tail_su)].item()) <= 0 or int(hh[5].item()) < int(h5_cost):
            return []
        hh[int(tail_su)] -= 1
        hh[19] += 1
        hh[5] -= int(h5_cost)
        hh[13] += int(h5_cost)
        _add_degree_count(degree_meta_local, int(inc_degree), +1)
        _apply_mode_bucket_delta(mode_meta_local, 19, bucket, int(inc_degree), +1)
        step_moves = [{
            "op": f"inc_19_{mode_name}",
            "from": int(tail_su),
            "to": 19,
            "degree": int(inc_degree),
            "fixed_edges": int(h5_cost),
            "reserved_19": int(reserved_19),
        }]
        for _ in range(int(h5_cost)):
            step_moves.append({
                "op": "dec_5",
                "from": 5,
                "to": 13,
            })
        return step_moves

    v_need_5 = float(need_5)
    v_excess_5 = float(excess_5)
    v_need_19 = float(need_19)
    v_excess_19 = float(excess_19)

    for _ in range(max(0, int(max_moves))):
        edge19 = int(_fixed_anchor_edge_count(anchor_mode_meta, 19))
        cur5 = int(H_new[5].item())
        over_19 = max(0, int(edge19 - int(soft_cap_19_edges)))
        under_5 = max(0, int(int(soft_floor_5) - int(cur5)))

        if (float(v_need_5) > float(thr) or int(under_5) > 0) and (float(v_excess_19) > 0.6 * float(thr) or int(over_19) > 0):
            direction = '19_to_5'
        elif float(v_need_19) > 1.4 * float(thr) and float(v_excess_5) > 0.6 * float(thr):
            direction = '5_to_19'
        elif float(v_need_5) > float(thr):
            direction = '19_to_5'
        elif float(v_need_19) > float(thr):
            direction = '5_to_19'
        else:
            break

        H_try = H_new.clone()
        degree_try = _clone_special_degree_meta(special_meta)
        mode_try = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if str(direction) == '19_to_5':
            step_moves = _apply_shift_19_to_5(H_try, degree_try, mode_try)
            if step_moves:
                v_need_5 *= 0.75
                v_excess_19 *= 0.75
        else:
            step_moves = _apply_shift_5_to_19(H_try, degree_try, mode_try)
            if step_moves:
                v_need_19 *= 0.75
                v_excess_5 *= 0.75

        if not step_moves:
            break
        if int(_ether_edge_total(mode_try, H_try)) != int(W_ether):
            break

        H_new = H_try
        special_meta = degree_try
        anchor_mode_meta = mode_try
        moves.extend(step_moves)

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        special_meta,
        anchor_mode_meta=anchor_mode_meta,
    )

    meta = {
        "scores": {
            "5_need": float(need_5),
            "5_excess": float(excess_5),
            "19_need": float(need_19),
            "19_excess": float(excess_19),
        },
        "windows": {
            "5": s5,
            "19_modes": _anchor_mode_stats_meta(mode_scores),
        },
        "threshold": float(thr),
        "reserved_19": int(reserved_19),
        "soft_floor_5": int(soft_floor_5),
        "soft_cap_19": int(soft_cap_19_edges),
        "W_ether": int(W_ether),
        "ether_total_final": int(_ether_edge_total(anchor_mode_meta, H_new)),
        "mode_scores": _anchor_mode_stats_meta(mode_scores),
        "special_degree_meta": dict(special_meta),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(anchor_mode_meta),
        "balance_before": dict(balance_before),
        "balance_after": dict(_summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))),
    }

    print(f"  完成 {len(moves)} 条变更记录")
    print(f"  H[5]={int(H[5].item())} -> {int(H_new[5].item())}, H[19]={int(H[19].item())} -> {int(H_new[19].item())} (reserved_19={int(reserved_19)})")
    return H_new, moves, meta


def adjust_amine_620_by_difference_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    window_6: float = 3.0,
    window_20: float = 3.0,
    peak_rel_threshold: float = 0.01,
    max_moves: int = 5,
    min_keep: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[N连接(6/20)调整] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    s6 = _aromatic_anchor_score(adjuster, ppm_arr, diff_arr, 6, fallback_mu=146.375)
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, 20)
    s20 = _score_special_mode_total(
        mode_scores,
        ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'],
    )
    score_6 = float(s6.get("score", 0.0))
    score_20 = float(s20.get("score", 0.0))

    total_abs = float(s6.get('abs', 0.0)) + float(s20.get('abs', 0.0))
    thr = float(peak_rel_threshold) * max(1e-9, total_abs)

    print(f"  SU6@{float(s6.get('mu', 0.0)):.3f} [{float(s6.get('lo', 0.0)):.3f},{float(s6.get('hi', 0.0)):.3f}] score={score_6:.3f} (pos={float(s6.get('pos', 0.0)):.3f}, neg={float(s6.get('neg', 0.0)):.3f})")
    print(f"  SU20 modes total score={score_20:.3f} (pos={float(s20.get('pos', 0.0)):.3f}, neg={float(s20.get('neg', 0.0)):.3f})")
    for mode_name in ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3']:
        stats = mode_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"    {mode_name} [{float(stats.get('lo', 0.0)):.1f},{float(stats.get('hi', 0.0)):.1f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

    if max(abs(score_6), abs(score_20)) < thr:
        print("  峰强不足，跳过调整")
        return H, [], {
            "scores": {"6": s6, "20": s20},
            "centers": {"6": float(s6.get('mu', 0.0)), "20": float(s20.get('mu', 0.0))},
            "threshold": thr,
        }

    def _sgn(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    dir_6 = _sgn(float(score_6))
    dir_20 = _sgn(float(score_20))

    if dir_6 != 0 and dir_20 != 0 and dir_6 != dir_20:
        inc = 6 if dir_6 > 0 else 20
        dec = 20 if int(inc) == 6 else 6
    else:
        priority = 6 if abs(float(score_6)) >= abs(float(score_20)) else 20
        priority_dir = dir_6 if int(priority) == 6 else dir_20
        if int(priority_dir) >= 0:
            inc = int(priority)
            dec = 20 if int(inc) == 6 else 6
        else:
            dec = int(priority)
            inc = 20 if int(dec) == 6 else 6

    H_new = H.clone()
    moves: List[Dict[str, Any]] = []
    special_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H_new))
    balance_before = _summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))
    W = int(balance_before["amine_required"])
    total_anchor = int(balance_before["amine_total"])
    single_bucket, double_bucket, _, _ = _anchor_mode_bucket_names(20)

    if int(W) <= 0 and int(total_anchor) <= 0:
        print("  H[0]+2*H[27]=0，当前无可用胺锚点需求，跳过调整")
        return H, [], {
            "scores": {"6": s6, "20": s20},
            "centers": {"6": float(s6.get('mu', 0.0)), "20": float(s20.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
            "total_anchor": int(total_anchor),
        }

    if int(W) <= 0 and int(total_anchor) > 0:
        for _ in range(max(0, int(max_moves))):
            if int(H_new[20].item()) > int(min_keep):
                H_new[20] -= 1
                H_new[23] += 1
                moves.append({"op": "cleanup_20", "from": 20, "to": 23})
                continue
            if int(H_new[6].item()) > int(min_keep):
                H_new[6] -= 1
                H_new[13] += 1
                moves.append({"op": "cleanup_6", "from": 6, "to": 13})
                continue
            break

        meta = {
            "scores": {"6": s6, "20": s20},
            "centers": {"6": float(s6.get('mu', 0.0)), "20": float(s20.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
            "total_anchor": int(total_anchor),
            "cleanup_only": True,
        }

        print(f"  完成 {len(moves)} 条清理记录")
        print(f"  H[6]={int(H[6].item())} -> {int(H_new[6].item())}, H[20]={int(H[20].item())} -> {int(H_new[20].item())} (W={int(W)})")
        return H_new, moves, meta

    if int(total_anchor) != int(W):
        print(f"  警告: 当前(6+20)={int(total_anchor)} 与需求 W={int(W)} 不一致，Block B 仅做比例调整，跳过本轮胺调整")
        return H, [], {
            "scores": {"6": s6, "20": s20},
            "centers": {"6": float(s6.get('mu', 0.0)), "20": float(s20.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
            "total_anchor": int(total_anchor),
            "reason": "amine_total_mismatch",
        }

    def _pick_aromatic_donors(hh: torch.Tensor, num: int) -> Optional[List[int]]:
        donors: List[int] = []
        for _ in range(max(0, int(num))):
            if int(hh[13].item()) > 0:
                hh[13] -= 1
                donors.append(13)
            elif int(hh[11].item()) > 0:
                hh[11] -= 1
                donors.append(11)
            else:
                for su_idx in donors:
                    hh[int(su_idx)] += 1
                return None
        return donors

    def _restore_aromatic_donors(hh: torch.Tensor, donors: List[int]) -> None:
        for su_idx in list(donors or []):
            hh[int(su_idx)] += 1

    def _add_degree_count(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        counts = dict(meta_local.get(20, {}) or {})
        counts[int(degree_i)] = max(0, int(counts.get(int(degree_i), 0)) + int(delta))
        meta_local[20] = counts

    def _apply_shift_20_to_6(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_decrease(
            mode_scores,
            mode_meta_local,
            20,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, dec_degree, donor_num = picked
        donors = _pick_aromatic_donors(hh, donor_num)
        if donors is None:
            return []
        tail_su = _mode_tail_su(20, int(dec_degree))
        if tail_su is None or int(hh[20].item()) <= 0:
            _restore_aromatic_donors(hh, donors)
            return []
        hh[20] -= 1
        hh[int(tail_su)] += 1
        hh[6] += int(donor_num)
        _add_degree_count(degree_meta_local, int(dec_degree), -1)
        _apply_mode_bucket_delta(mode_meta_local, 20, bucket, int(dec_degree), -1)
        step_moves = [{
            "op": f"dec_20_{mode_name}",
            "from": 20,
            "to": int(tail_su),
            "degree": int(dec_degree),
            "fixed_edges": int(donor_num),
        }]
        for donor_su in list(donors or []):
            step_moves.append({
                "op": "inc_6",
                "from": int(donor_su),
                "to": 6,
            })
        return step_moves

    def _apply_shift_6_to_20(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_increase(
            mode_scores,
            hh,
            6,
            20,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, inc_degree, h6_cost = picked
        tail_su = _mode_tail_su(20, int(inc_degree))
        if tail_su is None or int(hh[int(tail_su)].item()) <= 0 or int(hh[6].item()) < int(h6_cost):
            return []
        hh[int(tail_su)] -= 1
        hh[20] += 1
        hh[6] -= int(h6_cost)
        hh[13] += int(h6_cost)
        _add_degree_count(degree_meta_local, int(inc_degree), +1)
        _apply_mode_bucket_delta(mode_meta_local, 20, bucket, int(inc_degree), +1)
        step_moves = [{
            "op": f"inc_20_{mode_name}",
            "from": int(tail_su),
            "to": 20,
            "degree": int(inc_degree),
            "fixed_edges": int(h6_cost),
        }]
        for _ in range(int(h6_cost)):
            step_moves.append({
                "op": "dec_6",
                "from": 6,
                "to": 13,
            })
        return step_moves

    for _ in range(max(0, int(max_moves))):
        H_try = H_new.clone()
        degree_try = _clone_special_degree_meta(special_meta)
        mode_try = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(inc) == 6:
            step_moves = _apply_shift_20_to_6(H_try, degree_try, mode_try)
        else:
            step_moves = _apply_shift_6_to_20(H_try, degree_try, mode_try)
        if not step_moves:
            break
        if int(H_try[6].item()) + int(_fixed_anchor_edge_count(mode_try, 20)) != int(W):
            break
        H_new = H_try
        special_meta = degree_try
        anchor_mode_meta = mode_try
        moves.extend(step_moves)

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        special_meta,
        anchor_mode_meta=anchor_mode_meta,
    )

    meta = {
        "scores": {"6": s6, "20": s20},
        "centers": {"6": float(s6.get('mu', 0.0)), "20": float(s20.get('mu', 0.0))},
        "threshold": thr,
        "direction": {"inc": int(inc), "dec": int(dec)},
        "W": int(W),
        "total_anchor": int(total_anchor),
        "mode_scores": _anchor_mode_stats_meta(mode_scores),
        "special_degree_meta": dict(special_meta),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(anchor_mode_meta),
        "balance_before": dict(balance_before),
        "balance_after": dict(_summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))),
    }

    print(f"  完成 {len(moves)} 条变更记录")
    print(f"  H[6]={int(H[6].item())} -> {int(H_new[6].item())}, H[20]={int(H[20].item())} -> {int(H_new[20].item())}")
    return H_new, moves, meta


def adjust_thioether_719_by_difference_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    window_7: float = 3.0,
    window_19: float = 3.0,
    peak_rel_threshold: float = 0.01,
    max_moves: int = 5,
    min_keep: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[S连接(7/19)调整] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    s7 = _aromatic_anchor_score(adjuster, ppm_arr, diff_arr, 7, fallback_mu=152.875)
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, 19)
    s19 = _score_special_mode_total(
        mode_scores,
        ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3'],
    )
    score_7 = float(s7.get("score", 0.0))
    score_19 = float(s19.get("score", 0.0))

    total_abs = float(s7.get('abs', 0.0)) + float(s19.get('abs', 0.0))
    thr = float(peak_rel_threshold) * max(1e-9, total_abs)

    print(f"  SU7@{float(s7.get('mu', 0.0)):.3f} [{float(s7.get('lo', 0.0)):.3f},{float(s7.get('hi', 0.0)):.3f}] score={score_7:.3f} (pos={float(s7.get('pos', 0.0)):.3f}, neg={float(s7.get('neg', 0.0)):.3f})")
    print(f"  SU19 thio modes total score={score_19:.3f} (pos={float(s19.get('pos', 0.0)):.3f}, neg={float(s19.get('neg', 0.0)):.3f})")
    for mode_name in ['single_d1', 'single_d2', 'single_d3', 'double_d2', 'double_d3']:
        stats = mode_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"    {mode_name} [{float(stats.get('lo', 0.0)):.1f},{float(stats.get('hi', 0.0)):.1f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

    if max(abs(score_7), abs(score_19)) < thr:
        print("  峰强不足，跳过调整")
        return H, [], {
            "scores": {"7": s7, "19": s19},
            "centers": {"7": float(s7.get('mu', 0.0)), "19": float(s19.get('mu', 0.0))},
            "threshold": thr,
        }

    def _sgn(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    dir_7 = _sgn(float(score_7))
    dir_19 = _sgn(float(score_19))

    if dir_7 != 0 and dir_19 != 0 and dir_7 != dir_19:
        inc = 7 if dir_7 > 0 else 19
        dec = 19 if int(inc) == 7 else 7
    else:
        priority = 7 if abs(float(score_7)) >= abs(float(score_19)) else 19
        priority_dir = dir_7 if int(priority) == 7 else dir_19
        if int(priority_dir) >= 0:
            inc = int(priority)
            dec = 19 if int(inc) == 7 else 7
        else:
            dec = int(priority)
            inc = 19 if int(dec) == 7 else 7

    H_new = H.clone()
    moves: List[Dict[str, Any]] = []
    special_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H_new))
    balance_before = _summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))
    W = int(balance_before["thio_required"])
    total_anchor = int(H_new[7].item()) + int(_fixed_anchor_edge_count(anchor_mode_meta, 19, partition='thio'))
    single_bucket = 'thio_single'
    double_bucket = 'thio_double'

    if int(W) <= 0:
        print("  H[31]=0，无硫醚连接需求，跳过调整")
        return H, [], {
            "scores": {"7": s7, "19": s19},
            "centers": {"7": float(s7.get('mu', 0.0)), "19": float(s19.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
        }

    if int(total_anchor) != int(W):
        print(f"  警告: 当前(SU7+thio19_edges)={int(total_anchor)} 与需求 W={int(W)} 不一致，跳过本轮硫调整")
        return H, [], {
            "scores": {"7": s7, "19": s19},
            "centers": {"7": float(s7.get('mu', 0.0)), "19": float(s19.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
            "total_anchor": int(total_anchor),
            "reason": "thio_total_mismatch",
        }

    def _pick_aromatic_donors(hh: torch.Tensor, num: int) -> Optional[List[int]]:
        donors: List[int] = []
        for _ in range(max(0, int(num))):
            if int(hh[13].item()) > 0:
                hh[13] -= 1
                donors.append(13)
            elif int(hh[11].item()) > 0:
                hh[11] -= 1
                donors.append(11)
            else:
                for su_idx in donors:
                    hh[int(su_idx)] += 1
                return None
        return donors

    def _restore_aromatic_donors(hh: torch.Tensor, donors: List[int]) -> None:
        for su_idx in list(donors or []):
            hh[int(su_idx)] += 1

    def _add_degree_count(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        counts = dict(meta_local.get(19, {}) or {})
        counts[int(degree_i)] = max(0, int(counts.get(int(degree_i), 0)) + int(delta))
        meta_local[19] = counts

    def _thio_edge_total(mode_meta_local: Dict[int, Dict[str, Dict[int, int]]], hh: torch.Tensor) -> int:
        return int(hh[7].item()) + int(_fixed_anchor_edge_count(mode_meta_local, 19, partition='thio'))

    def _apply_shift_19_to_7(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_decrease(
            mode_scores,
            mode_meta_local,
            19,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, dec_degree, donor_num = picked
        donors = _pick_aromatic_donors(hh, donor_num)
        if donors is None:
            return []
        tail_su = _mode_tail_su(19, int(dec_degree))
        if tail_su is None or int(hh[19].item()) <= 0:
            _restore_aromatic_donors(hh, donors)
            return []
        hh[19] -= 1
        hh[int(tail_su)] += 1
        hh[7] += int(donor_num)
        _add_degree_count(degree_meta_local, int(dec_degree), -1)
        _apply_mode_bucket_delta(mode_meta_local, 19, bucket, int(dec_degree), -1)
        step_moves = [{
            "op": f"dec_19_thio_{mode_name}",
            "from": 19,
            "to": int(tail_su),
            "degree": int(dec_degree),
            "fixed_edges": int(donor_num),
        }]
        for donor_su in list(donors or []):
            step_moves.append({
                "op": "inc_7",
                "from": int(donor_su),
                "to": 7,
            })
        return step_moves

    def _apply_shift_7_to_19(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_increase(
            mode_scores,
            hh,
            7,
            19,
            str(single_bucket),
            str(double_bucket),
            [1, 2, 3],
            [2, 3],
        )
        if picked is None:
            return []
        bucket, mode_name, inc_degree, h7_cost = picked
        tail_su = _mode_tail_su(19, int(inc_degree))
        if tail_su is None or int(hh[int(tail_su)].item()) <= 0 or int(hh[7].item()) < int(h7_cost):
            return []
        hh[int(tail_su)] -= 1
        hh[19] += 1
        hh[7] -= int(h7_cost)
        hh[13] += int(h7_cost)
        _add_degree_count(degree_meta_local, int(inc_degree), +1)
        _apply_mode_bucket_delta(mode_meta_local, 19, bucket, int(inc_degree), +1)
        step_moves = [{
            "op": f"inc_19_thio_{mode_name}",
            "from": int(tail_su),
            "to": 19,
            "degree": int(inc_degree),
            "fixed_edges": int(h7_cost),
        }]
        for _ in range(int(h7_cost)):
            step_moves.append({
                "op": "dec_7",
                "from": 7,
                "to": 13,
            })
        return step_moves

    for _ in range(max(0, int(max_moves))):
        H_try = H_new.clone()
        degree_try = _clone_special_degree_meta(special_meta)
        mode_try = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(inc) == 7:
            step_moves = _apply_shift_19_to_7(H_try, degree_try, mode_try)
        else:
            step_moves = _apply_shift_7_to_19(H_try, degree_try, mode_try)
        if not step_moves:
            break
        if int(_thio_edge_total(mode_try, H_try)) != int(W):
            break
        H_new = H_try
        special_meta = degree_try
        anchor_mode_meta = mode_try
        moves.extend(step_moves)

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        special_meta,
        anchor_mode_meta=anchor_mode_meta,
    )

    meta = {
        "scores": {"7": s7, "19": s19},
        "centers": {"7": float(s7.get('mu', 0.0)), "19": float(s19.get('mu', 0.0))},
        "threshold": thr,
        "direction": {"inc": int(inc), "dec": int(dec)},
        "W": int(W),
        "total_anchor": int(total_anchor),
        "mode_scores": _anchor_mode_stats_meta(mode_scores),
        "special_degree_meta": dict(special_meta),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(anchor_mode_meta),
        "balance_before": dict(balance_before),
        "balance_after": dict(_summarize_block_b_balance(
            H_new,
            fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None),
        )),
    }

    print(f"  完成 {len(moves)} 条变更记录")
    print(f"  H[7]={int(H[7].item())} -> {int(H_new[7].item())}, H[19]={int(H[19].item())} -> {int(H_new[19].item())}")
    return H_new, moves, meta


def adjust_halogen_821_by_difference_impl(
    adjuster: Any,
    H: torch.Tensor,
    ppm: Optional[np.ndarray],
    diff: Optional[np.ndarray],
    window_8: float = 3.0,
    window_21: float = 3.0,
    peak_rel_threshold: float = 0.01,
    max_moves: int = 5,
    min_keep: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[X连接(8/21)调整] 基于差谱分析")

    if ppm is None or diff is None:
        print("  无差谱数据，跳过调整")
        return H, [], {}

    ppm_arr = np.asarray(ppm, dtype=np.float64)
    diff_arr = np.asarray(diff, dtype=np.float64)
    if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
        print("  差谱为空，跳过调整")
        return H, [], {}

    s8 = _aromatic_anchor_score(adjuster, ppm_arr, diff_arr, 8, fallback_mu=131.4244)
    mode_scores = _special_anchor_mode_scores(adjuster, ppm_arr, diff_arr, 21)
    s21 = _score_special_mode_total(
        mode_scores,
        ['single_d2', 'single_d3'],
    )
    score_8 = float(s8.get("score", 0.0))
    score_21 = float(s21.get("score", 0.0))

    total_abs = float(s8.get('abs', 0.0)) + float(s21.get('abs', 0.0))
    thr = float(peak_rel_threshold) * max(1e-9, total_abs)

    print(f"  SU8@{float(s8.get('mu', 0.0)):.3f} [{float(s8.get('lo', 0.0)):.3f},{float(s8.get('hi', 0.0)):.3f}] score={score_8:.3f} (pos={float(s8.get('pos', 0.0)):.3f}, neg={float(s8.get('neg', 0.0)):.3f})")
    print(f"  SU21 modes total score={score_21:.3f} (pos={float(s21.get('pos', 0.0)):.3f}, neg={float(s21.get('neg', 0.0)):.3f})")
    for mode_name in ['single_d2', 'single_d3']:
        stats = mode_scores.get(str(mode_name), {})
        if not stats:
            continue
        print(
            f"    {mode_name} [{float(stats.get('lo', 0.0)):.1f},{float(stats.get('hi', 0.0)):.1f}] "
            f"score={float(stats.get('score', 0.0)):.3f} "
            f"pos={float(stats.get('pos', 0.0)):.3f} neg={float(stats.get('neg', 0.0)):.3f}"
        )
    print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

    if max(abs(score_8), abs(score_21)) < thr:
        print("  峰强不足，跳过调整")
        return H, [], {
            "scores": {"8": s8, "21": s21},
            "centers": {"8": float(s8.get('mu', 0.0)), "21": float(s21.get('mu', 0.0))},
            "threshold": thr,
        }

    def _sgn(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    dir_8 = _sgn(float(score_8))
    dir_21 = _sgn(float(score_21))

    if dir_8 != 0 and dir_21 != 0 and dir_8 != dir_21:
        inc = 8 if dir_8 > 0 else 21
        dec = 21 if int(inc) == 8 else 8
    else:
        priority = 8 if abs(float(score_8)) >= abs(float(score_21)) else 21
        priority_dir = dir_8 if int(priority) == 8 else dir_21
        if int(priority_dir) >= 0:
            inc = int(priority)
            dec = 21 if int(inc) == 8 else 8
        else:
            dec = int(priority)
            inc = 21 if int(dec) == 8 else 8

    H_new = H.clone()
    moves: List[Dict[str, Any]] = []
    special_meta = _clone_special_degree_meta(adjuster._get_special_degree_meta(H_new))
    anchor_mode_meta = _clone_special_anchor_mode_meta(adjuster._get_special_anchor_mode_meta(H_new))
    balance_before = _summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))
    W = int(balance_before["halogen_required"])
    total_x = int(balance_before["halogen_total"])
    single_bucket, double_bucket, _, _ = _anchor_mode_bucket_names(21)
    if int(W) != int(total_x):
        print(f"  警告: 当前(8+21)={total_x} 与 H[32]={W} 不一致，跳过调整")
        return H, [], {
            "scores": {"8": s8, "21": s21},
            "centers": {"8": float(s8.get('mu', 0.0)), "21": float(s21.get('mu', 0.0))},
            "threshold": thr,
            "direction": {"inc": int(inc), "dec": int(dec)},
            "W": int(W),
            "total_x": int(total_x),
        }

    def _pick_aromatic_donors(hh: torch.Tensor, num: int) -> Optional[List[int]]:
        donors: List[int] = []
        for _ in range(max(0, int(num))):
            if int(hh[13].item()) > 0:
                hh[13] -= 1
                donors.append(13)
            elif int(hh[11].item()) > 0:
                hh[11] -= 1
                donors.append(11)
            else:
                for su_idx in donors:
                    hh[int(su_idx)] += 1
                return None
        return donors

    def _restore_aromatic_donors(hh: torch.Tensor, donors: List[int]) -> None:
        for su_idx in list(donors or []):
            hh[int(su_idx)] += 1

    def _add_degree_count(meta_local: Dict[int, Dict[int, int]], degree_i: int, delta: int) -> None:
        counts = dict(meta_local.get(21, {}) or {})
        counts[int(degree_i)] = max(0, int(counts.get(int(degree_i), 0)) + int(delta))
        meta_local[21] = counts

    def _apply_shift_21_to_8(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_decrease(
            mode_scores,
            mode_meta_local,
            21,
            str(single_bucket),
            str(double_bucket),
            [2, 3],
            [],
        )
        if picked is None:
            return []
        bucket, mode_name, dec_degree, donor_num = picked
        donors = _pick_aromatic_donors(hh, donor_num)
        if donors is None:
            return []
        tail_su = _mode_tail_su(21, int(dec_degree))
        if tail_su is None or int(hh[21].item()) <= int(min_keep):
            _restore_aromatic_donors(hh, donors)
            return []
        hh[21] -= 1
        hh[int(tail_su)] += 1
        hh[8] += int(donor_num)
        _add_degree_count(degree_meta_local, int(dec_degree), -1)
        _apply_mode_bucket_delta(mode_meta_local, 21, bucket, int(dec_degree), -1)
        step_moves = [{
            "op": f"dec_21_{mode_name}",
            "from": 21,
            "to": int(tail_su),
            "degree": int(dec_degree),
            "fixed_edges": int(donor_num),
        }]
        for donor_su in list(donors or []):
            step_moves.append({
                "op": "inc_8",
                "from": int(donor_su),
                "to": 8,
            })
        return step_moves

    def _apply_shift_8_to_21(hh: torch.Tensor,
                             degree_meta_local: Dict[int, Dict[int, int]],
                             mode_meta_local: Dict[int, Dict[str, Dict[int, int]]]) -> List[Dict[str, Any]]:
        picked = _select_special_mode_for_increase(
            mode_scores,
            hh,
            8,
            21,
            str(single_bucket),
            str(double_bucket),
            [2, 3],
            [],
        )
        if picked is None:
            return []
        bucket, mode_name, inc_degree, h8_cost = picked
        tail_su = _mode_tail_su(21, int(inc_degree))
        if tail_su is None or int(hh[int(tail_su)].item()) <= 0 or int(hh[8].item()) < int(h8_cost):
            return []
        hh[int(tail_su)] -= 1
        hh[21] += 1
        hh[8] -= int(h8_cost)
        hh[13] += int(h8_cost)
        _add_degree_count(degree_meta_local, int(inc_degree), +1)
        _apply_mode_bucket_delta(mode_meta_local, 21, bucket, int(inc_degree), +1)
        step_moves = [{
            "op": f"inc_21_{mode_name}",
            "from": int(tail_su),
            "to": 21,
            "degree": int(inc_degree),
            "fixed_edges": int(h8_cost),
        }]
        for _ in range(int(h8_cost)):
            step_moves.append({
                "op": "dec_8",
                "from": 8,
                "to": 13,
            })
        return step_moves

    for _ in range(max(0, int(max_moves))):
        H_try = H_new.clone()
        degree_try = _clone_special_degree_meta(special_meta)
        mode_try = _clone_special_anchor_mode_meta(anchor_mode_meta)
        if int(inc) == 8:
            step_moves = _apply_shift_21_to_8(H_try, degree_try, mode_try)
        else:
            step_moves = _apply_shift_8_to_21(H_try, degree_try, mode_try)
        if not step_moves:
            break
        if int(H_try[8].item()) + int(_fixed_anchor_edge_count(mode_try, 21)) != int(W):
            break
        H_new = H_try
        special_meta = degree_try
        anchor_mode_meta = mode_try
        moves.extend(step_moves)

    _sync_adjuster_mode_meta(
        adjuster,
        H_new,
        special_meta,
        anchor_mode_meta=anchor_mode_meta,
    )

    meta = {
        "scores": {"8": s8, "21": s21},
        "centers": {"8": float(s8.get('mu', 0.0)), "21": float(s21.get('mu', 0.0))},
        "threshold": thr,
        "direction": {"inc": int(inc), "dec": int(dec)},
        "W": int(W),
        "total_x": int(W),
        "mode_scores": _anchor_mode_stats_meta(mode_scores),
        "special_degree_meta": dict(special_meta),
        "special_anchor_mode_meta": _clone_special_anchor_mode_meta(anchor_mode_meta),
        "balance_before": dict(balance_before),
        "balance_after": dict(_summarize_block_b_balance(H_new, fixed_partition_meta=getattr(adjuster, 'fixed_partition_meta', None))),
    }

    print(f"  完成 {len(moves)} 条变更记录")
    print(f"  H[8]={int(H[8].item())} -> {int(H_new[8].item())}, H[21]={int(H[21].item())} -> {int(H_new[21].item())} (8+21={int(H_new[8].item()) + int(H_new[21].item())} == {int(W)})")
    return H_new, moves, meta
