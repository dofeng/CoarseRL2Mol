import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
from collections import Counter
import math

from ...paths import Z_LIBRARY_DIR
from ...shared.inverse_common import (
    E_SU, _NodeV3,
    TERMINAL_PORT_BASE_SU,
    SPECIAL_DEGREE_PRIORS,
    get_aliphatic_carbon_policy,
    get_effective_hist_element_vector,
    estimate_region_carbon_budgets,
    normalize_special_degree_meta,
    rebuild_su19_partition_meta,
)
from ..layer0.inverse_layer0 import Layer0Estimator
from ..topology import (
    apply_stage3_node_adjustments,
    assemble_topology_units,
    classify_topology_units,
    plan_stage3_resource_adjustments,
    Stage3AdjustmentApplyResult,
    Stage3AdjustmentPlan,
    Stage3ProgressPhase,
)
from .inverse_layer4_block_a import (
    adjust_block_a_carbonyl_anchor_impl,
)
from .inverse_layer4_block_b import (
    adjust_block_b_hetero_anchor_impl,
)
from .inverse_layer4_block_c import (
    run_block_c_resource_adjustment_loop_impl,
)

class Layer4Adjuster:

    def __init__(self, 
                 device: torch.device = None,
                 layer0_estimator: Optional[Layer0Estimator] = None,
                 su_common_ranges_path: Optional[str] = None,
                 su_special_degree_ranges_path: Optional[str] = None):
        self.device = device or torch.device('cpu')
        self.E_SU = E_SU.to(self.device)
        self.layer0_estimator = layer0_estimator
        self.fixed_partition_meta: Dict[str, Any] = {}
        
        if su_common_ranges_path is None:
            default_common = Z_LIBRARY_DIR / 'su_nmr_common_range_filtered.csv'
            self.su_common_ranges_path = str(default_common) if default_common.exists() else None
        else:
            self.su_common_ranges_path = su_common_ranges_path
        if su_special_degree_ranges_path is None:
            default_special = Z_LIBRARY_DIR / 'su_special_degree_nmr_common_range_filtered.csv'
            self.su_special_degree_ranges_path = str(default_special) if default_special.exists() else None
        else:
            self.su_special_degree_ranges_path = su_special_degree_ranges_path
        self._su_common_stats_cache = None
        self._su_special_degree_stats_cache = None
        self._spectrum_hop1_template_options: Dict[int, List[Dict[str, Any]]] = {}
        self._stage3_su11_rotation_index = 0
        self._stage3_parity_balance_cursor = 0
        self._spectrum_search_memory: Dict[str, Dict[str, Any]] = {}
        self.fixed_assignment_precheck = None
        self._h_tolerance: float = 0.08
        self._current_S_target: Optional[torch.Tensor] = None

    @staticmethod
    def _build_node_lookup(nodes: List[_NodeV3]) -> Dict[int, _NodeV3]:
        lookup: Dict[int, _NodeV3] = {}
        for node in nodes:
            try:
                lookup[int(node.global_id)] = node
            except Exception:
                continue
        return lookup

    def _current_neighbor_types(self,
                                node: _NodeV3,
                                nodes: List[_NodeV3],
                                node_lookup: Optional[Dict[int, _NodeV3]] = None) -> List[int]:
        if node_lookup is None:
            node_lookup = self._build_node_lookup(nodes)
        out: List[int] = []
        hop1_ids = list(getattr(node, 'hop1_ids', []) or [])
        for nid in hop1_ids:
            try:
                nb = node_lookup.get(int(nid))
            except Exception:
                nb = None
            if nb is None:
                continue
            try:
                out.append(int(nb.su_type))
            except Exception:
                continue
        if out:
            return out

        hop1_counter = getattr(node, 'hop1_su', None)
        if isinstance(hop1_counter, Counter):
            restored: List[int] = []
            for su_type, count in hop1_counter.items():
                try:
                    restored.extend([int(su_type)] * int(count))
                except Exception:
                    continue
            return restored
        return []

    def _current_hop2_counter(self,
                              node: _NodeV3,
                              nodes: List[_NodeV3],
                              node_lookup: Optional[Dict[int, _NodeV3]] = None) -> Counter:
        if node_lookup is None:
            node_lookup = self._build_node_lookup(nodes)

        hop2 = Counter()
        hop1_ids = list(getattr(node, 'hop1_ids', []) or [])
        if hop1_ids:
            center_id = int(getattr(node, 'global_id', -1))
            for nb_id in hop1_ids:
                try:
                    nb = node_lookup.get(int(nb_id))
                except Exception:
                    nb = None
                if nb is None:
                    continue
                for nb2_id in list(getattr(nb, 'hop1_ids', []) or []):
                    try:
                        nb2_id_i = int(nb2_id)
                    except Exception:
                        continue
                    if nb2_id_i == center_id:
                        continue
                    nb2 = node_lookup.get(nb2_id_i)
                    if nb2 is None:
                        continue
                    try:
                        hop2[int(nb2.su_type)] += 1
                    except Exception:
                        continue
            return hop2

        hop2_counter = getattr(node, 'hop2_su', None)
        if isinstance(hop2_counter, Counter):
            return Counter({int(k): int(v) for k, v in hop2_counter.items()})
        return hop2

    def _refresh_node_counters(self, nodes: List[_NodeV3]) -> None:
        """Refresh hop1_su/hop2_su after temporary SU-type conversions."""
        node_lookup = self._build_node_lookup(nodes)
        for node in nodes:
            hop1_counter = Counter()
            for su_type in self._current_neighbor_types(node, nodes, node_lookup=node_lookup):
                hop1_counter[int(su_type)] += 1
            node.hop1_su = hop1_counter

        for node in nodes:
            node.hop2_su = self._current_hop2_counter(node, nodes, node_lookup=node_lookup)

    def _apply_final_structure_constraints(self, H: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, int]], Dict[str, Any]]:
        """
        Final discrete constraints applied after Layer4 finishes.

        Current rules:
        1. SU10 count must be even; if odd, convert one SU10 to SU11.
        2. SU14+SU15+SU16 total must be even; if odd, use count-preserving conversion:
           prefer 15->13, else 16->23, else 14->11.
        """
        H_work = torch.clamp(H, min=0).long().clone()
        moves: List[Dict[str, int]] = []

        try:
            unsat_total = int(H_work[14].item()) + int(H_work[15].item()) + int(H_work[16].item())
        except Exception:
            unsat_total = 0

        if int(unsat_total) % 2 != 0:
            for src_idx, dst_idx in ((15, 13), (16, 23), (14, 11)):
                if int(H_work[src_idx].item()) > 0:
                    H_work[src_idx] -= 1
                    H_work[dst_idx] += 1
                    moves.append({'op': 'final_even_14_15_16', 'from': int(src_idx), 'to': int(dst_idx)})
                    break

        try:
            if int(H_work[10].item()) % 2 != 0 and int(H_work[10].item()) > 0:
                H_work[10] -= 1
                H_work[11] += 1
                moves.append({'op': 'final_even_10', 'from': 10, 'to': 11})
        except Exception:
            pass

        meta = {
            'final_su10': int(H_work[10].item()) if int(H_work.numel()) > 10 else 0,
            'final_unsat_141516': int(H_work[14].item()) + int(H_work[15].item()) + int(H_work[16].item()),
        }
        return H_work, moves, meta

    @staticmethod
    def _window_stats(ppm_arr: np.ndarray, diff_arr: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
        mask = (ppm_arr >= float(lo)) & (ppm_arr <= float(hi))
        if not bool(mask.any()):
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0, 'dom': 0.0}
        seg = diff_arr[mask]
        if int(seg.size) <= 0:
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0, 'dom': 0.0}
        pos = float(np.sum(seg[seg > 0])) if np.any(seg > 0) else 0.0
        neg = float(-np.sum(seg[seg < 0])) if np.any(seg < 0) else 0.0
        dom = float(pos) if float(pos) >= float(neg) else -float(neg)
        return {
            'pos': float(pos),
            'neg': float(neg),
            'net': float(pos - neg),
            'abs': float(np.sum(np.abs(seg))),
            'dom': float(dom),
        }

    def _get_su_common_stats(self) -> Dict[int, Dict[str, float]]:
        if self._su_common_stats_cache is not None:
            return self._su_common_stats_cache

        stats: Dict[int, Dict[str, float]] = {}
        path = self.su_common_ranges_path
        if path is None or not Path(path).exists():
            self._su_common_stats_cache = stats
            return stats

        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                su_idx = int(row['center_su_idx'])
                stats[su_idx] = {
                    'mu_median': float(row['mu_median']),
                    'mu_common_min': float(row['mu_common_min']),
                    'mu_common_max': float(row['mu_common_max']),
                    'mu_q05': float(row.get('mu_q05', row.get('mu_common_min', row.get('mu_median', 0.0)))),
                    'mu_q95': float(row.get('mu_q95', row.get('mu_common_max', row.get('mu_median', 0.0)))),
                }
        except Exception:
            stats = {}

        self._su_common_stats_cache = stats
        return stats

    def set_spectrum_hop1_template_options(
        self,
        options_by_su: Optional[Dict[int, List[Dict[str, Any]]]],
    ) -> None:
        """Install the filtered-library 1-hop peaks used only for spectrum trials."""
        self._spectrum_hop1_template_options = {
            int(su_type): [dict(option or {}) for option in list(options or [])]
            for su_type, options in dict(options_by_su or {}).items()
        }

    def _get_su_common_window(self,
                              su_idx: int,
                              fallback_mu: Optional[float] = None,
                              pad: float = 0.0,
                              min_half_width: float = 0.0) -> Tuple[float, float, float]:
        stats = self._get_su_common_stats().get(int(su_idx))
        if stats is not None:
            mu = float(stats['mu_median'])
            lo = float(stats['mu_common_min'])
            hi = float(stats['mu_common_max'])
        else:
            mu = float(fallback_mu or 0.0)
            width = max(float(min_half_width), float(pad))
            lo = float(mu - width)
            hi = float(mu + width)

        lo = float(lo) - float(pad)
        hi = float(hi) + float(pad)
        if float(min_half_width) > 0.0:
            lo = min(float(lo), float(mu - float(min_half_width)))
            hi = max(float(hi), float(mu + float(min_half_width)))
        return float(lo), float(hi), float(mu)

    def _get_su_tail_core_window(self,
                                 su_idx: int,
                                 fallback_mu: Optional[float] = None,
                                 pad: float = 0.0,
                                 max_half_width: float = 8.0) -> Tuple[float, float, float]:
        """
        Tail-region core window for 22/23/24-like decisions.

        We intentionally prefer q05-q95 over the broader common range and
        then clamp around the median, because the full common ranges of
        22/23/24 overlap heavily and can cause SU24 to be misread from
        nearby 23/25 or 50-60 ppm features.
        """
        stats = self._get_su_common_stats().get(int(su_idx))
        if stats is not None:
            mu = float(stats['mu_median'])
            lo = float(stats.get('mu_q05', stats.get('mu_common_min', mu)))
            hi = float(stats.get('mu_q95', stats.get('mu_common_max', mu)))
        else:
            mu = float(fallback_mu or 0.0)
            width = max(2.0, float(max_half_width))
            lo = float(mu - width)
            hi = float(mu + width)

        if float(max_half_width) > 0.0:
            lo = max(float(lo), float(mu - float(max_half_width)))
            hi = min(float(hi), float(mu + float(max_half_width)))

        lo = float(lo) - float(pad)
        hi = float(hi) + float(pad)
        if float(hi) <= float(lo):
            width = max(2.0, float(max_half_width))
            lo = float(mu - width)
            hi = float(mu + width)
        return float(lo), float(hi), float(mu)

    def _get_su_special_degree_stats(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        if self._su_special_degree_stats_cache is not None:
            return self._su_special_degree_stats_cache

        stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        path = self.su_special_degree_ranges_path
        if path is None or not Path(path).exists():
            self._su_special_degree_stats_cache = stats
            return stats

        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                su_idx = int(row['center_su_idx'])
                degree_i = int(row['target_degree'])
                stats[(su_idx, degree_i)] = {
                    'mu_median': float(row['mu_median']),
                    'mu_common_min': float(row['mu_common_min']),
                    'mu_common_max': float(row['mu_common_max']),
                }
        except Exception:
            stats = {}

        self._su_special_degree_stats_cache = stats
        return stats

    def _get_su_special_degree_window(self,
                                      su_idx: int,
                                      degree_i: int,
                                      fallback_mu: Optional[float] = None,
                                      pad: float = 0.0,
                                      min_half_width: float = 0.0) -> Tuple[float, float, float]:
        stats = self._get_su_special_degree_stats().get((int(su_idx), int(degree_i)))
        if stats is not None:
            mu = float(stats['mu_median'])
            lo = float(stats['mu_common_min'])
            hi = float(stats['mu_common_max'])
        else:
            return self._get_su_common_window(
                su_idx=int(su_idx),
                fallback_mu=fallback_mu,
                pad=pad,
                min_half_width=min_half_width,
            )

        lo = float(lo) - float(pad)
        hi = float(hi) + float(pad)
        if float(min_half_width) > 0.0:
            lo = min(float(lo), float(mu - float(min_half_width)))
            hi = max(float(hi), float(mu + float(min_half_width)))
        return float(lo), float(hi), float(mu)

    def _get_special_degree_meta(self, H: Optional[torch.Tensor] = None) -> Dict[int, Dict[int, int]]:
        total_counts = {}
        if H is not None:
            H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
            total_counts = {
                int(su): int(H_cpu[int(su)].item())
                for su in SPECIAL_DEGREE_PRIORS.keys()
                if int(H_cpu.numel()) > int(su)
            }
        src = dict(getattr(self, 'fixed_partition_meta', {}) or {})
        raw_meta = dict(src.get('special_degree_meta', {}) or {})
        return normalize_special_degree_meta(total_counts, special_degree_meta=raw_meta)

    def _set_special_degree_meta(self, H: Optional[torch.Tensor], special_degree_meta: Dict[int, Dict[int, int]]) -> Dict[int, Dict[int, int]]:
        meta_norm = self._get_special_degree_meta(H if H is not None else torch.zeros(33, dtype=torch.long))
        if special_degree_meta:
            total_counts = {
                int(su): int((torch.clamp(H.detach().cpu(), min=0).long()[int(su)].item() if H is not None and int(H.numel()) > int(su) else 0))
                for su in SPECIAL_DEGREE_PRIORS.keys()
            }
            meta_norm = normalize_special_degree_meta(total_counts, special_degree_meta=special_degree_meta)
        fixed_meta = dict(getattr(self, 'fixed_partition_meta', {}) or {})
        fixed_meta['special_degree_meta'] = {
            int(su): {int(deg): int(cnt) for deg, cnt in dict(parts).items()}
            for su, parts in dict(meta_norm).items()
        }
        total_19 = 0
        if H is not None:
            try:
                total_19 = int(torch.clamp(H.detach().cpu(), min=0).long()[19].item())
            except Exception:
                total_19 = 0
        o_base_19 = max(0, min(int(fixed_meta.get('o_base_19', total_19)), int(total_19)))
        s_reserved_19 = max(0, min(int(fixed_meta.get('s_reserved_19', max(0, total_19 - o_base_19))), int(total_19 - o_base_19)))
        if int(o_base_19 + s_reserved_19) < int(total_19):
            o_base_19 += int(total_19 - (o_base_19 + s_reserved_19))
        prev_part_meta_19 = dict((fixed_meta.get('special_partition_meta', {}) or {}).get(19, (fixed_meta.get('special_partition_meta', {}) or {}).get('19', {})) or {})
        fixed_meta['special_partition_meta'] = dict(getattr(self, 'fixed_partition_meta', {}) or {}).get('special_partition_meta', {}) or {}
        fixed_meta['special_partition_meta'] = {
            int(su): {
                str(part): {int(deg): int(cnt) for deg, cnt in dict(part_counts).items()}
                for part, part_counts in dict(parts).items()
            }
            for su, parts in dict(fixed_meta.get('special_partition_meta', {}) or {}).items()
        }
        fixed_meta['special_partition_meta'][19] = rebuild_su19_partition_meta(
            total_19=int(total_19),
            o_base_19=int(o_base_19),
            s_reserved_19=int(s_reserved_19),
            special_degree_meta_19=dict(meta_norm.get(19, {}) or {}),
            existing_partition_meta_19=prev_part_meta_19,
        )
        self.fixed_partition_meta = fixed_meta
        try:
            if self.layer0_estimator is not None:
                layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
                layer0_meta['special_degree_meta'] = dict(fixed_meta['special_degree_meta'])
                layer0_meta['special_partition_meta'] = dict(fixed_meta.get('special_partition_meta', {}) or {})
                if 'special_anchor_mode_meta' in fixed_meta:
                    layer0_meta['special_anchor_mode_meta'] = dict(fixed_meta.get('special_anchor_mode_meta', {}) or {})
                self.layer0_estimator.fixed_partition_meta = layer0_meta
                self.layer0_estimator.special_degree_meta = dict(fixed_meta['special_degree_meta'])
                self.layer0_estimator.special_partition_meta = dict(fixed_meta.get('special_partition_meta', {}) or {})
        except Exception:
            pass
        return dict(fixed_meta['special_degree_meta'])

    def _get_special_anchor_mode_meta(self, H: Optional[torch.Tensor] = None) -> Dict[int, Dict[str, Dict[int, int]]]:
        total_19 = 0
        total_20 = 0
        total_21 = 0
        if H is not None:
            try:
                H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
                total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
                total_20 = int(H_cpu[20].item()) if int(H_cpu.numel()) > 20 else 0
                total_21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
            except Exception:
                total_19 = total_20 = total_21 = 0
        src = dict(getattr(self, 'fixed_partition_meta', {}) or {})
        raw_mode = dict(src.get('special_anchor_mode_meta', {}) or {})
        raw_part = dict(src.get('special_partition_meta', {}) or {})
        out: Dict[int, Dict[str, Dict[int, int]]] = {}

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

        if int(total_20) > 0:
            deg_20 = dict(self._get_special_degree_meta(H).get(20, {}) or {})
            raw_20 = dict(raw_mode.get(20, raw_mode.get('20', {})) or {})
            total_counts_20 = {
                int(deg): max(0, int(deg_20.get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            double_counts = {
                int(deg): min(
                    int(total_counts_20.get(int(deg), 0)),
                    max(0, int(dict(raw_20.get('double', {}) or {}).get(int(deg), 0)))
                )
                for deg in [2, 3]
            }
            required_20_edges = 0
            try:
                required_20_edges = int(max(
                    0,
                    int(H_cpu[0].item()) + 2 * int(H_cpu[27].item()) - int(H_cpu[6].item()),
                ))
            except Exception:
                required_20_edges = int(total_20)
            total_nodes_20 = int(sum(int(v) for v in total_counts_20.values()))
            if int(total_nodes_20) <= 0:
                total_nodes_20 = int(total_20)
            target_double_total = int(max(0, int(required_20_edges) - int(total_nodes_20)))
            target_double_total = min(
                int(target_double_total),
                int(total_counts_20.get(2, 0)) + int(total_counts_20.get(3, 0)),
            )
            current_double_total = int(sum(int(v) for v in double_counts.values()))
            if int(current_double_total) < int(target_double_total):
                deficit = int(target_double_total) - int(current_double_total)
                for deg in [3, 2]:
                    room = max(0, int(total_counts_20.get(int(deg), 0)) - int(double_counts.get(int(deg), 0)))
                    take = min(int(deficit), int(room))
                    if int(take) > 0:
                        double_counts[int(deg)] = int(double_counts.get(int(deg), 0)) + int(take)
                        deficit -= int(take)
                    if int(deficit) <= 0:
                        break
            elif int(current_double_total) > int(target_double_total):
                excess = int(current_double_total) - int(target_double_total)
                for deg in [2, 3]:
                    take = min(int(excess), int(double_counts.get(int(deg), 0)))
                    if int(take) > 0:
                        double_counts[int(deg)] = int(double_counts.get(int(deg), 0)) - int(take)
                        excess -= int(take)
                    if int(excess) <= 0:
                        break
            single_counts = {
                int(deg): max(0, int(total_counts_20.get(int(deg), 0)) - int(double_counts.get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            out[20] = {
                'single': {int(deg): int(single_counts.get(int(deg), 0)) for deg in [1, 2, 3]},
                'double': {int(deg): int(double_counts.get(int(deg), 0)) for deg in [2, 3]},
            }
        if int(total_21) > 0:
            deg_21 = dict(self._get_special_degree_meta(H).get(21, {}) or {})
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

    def _get_carb_joint_windows(self) -> Dict[str, Tuple[float, float]]:
        lo_0, hi_0, mu_0 = self._get_su_common_window(0, fallback_mu=167.125, min_half_width=6.0)
        lo_1, hi_1, mu_1 = self._get_su_common_window(1, fallback_mu=174.875, min_half_width=6.0)
        lo_2, hi_2, mu_2 = self._get_su_common_window(2, fallback_mu=169.6288, min_half_width=6.0)
        lo_3, hi_3, mu_3 = self._get_su_common_window(3, fallback_mu=195.8284, min_half_width=8.0)

        split_12 = float((mu_1 + mu_2) * 0.5)
        low_lo = float(min(lo_0, lo_2))
        low_hi = float(split_12)
        mid_lo = float(split_12)
        mid_hi = float(max(hi_1, hi_2))
        high_lo = float(lo_3)
        high_hi = float(hi_3)
        overall_lo = float(min(low_lo, mid_lo, high_lo))
        overall_hi = float(max(low_hi, mid_hi, high_hi))

        return {
            'low': (low_lo, low_hi),
            'mid': (mid_lo, mid_hi),
            'high': (high_lo, high_hi),
            'overall': (overall_lo, overall_hi),
        }

    @staticmethod
    def _apply_count_delta(H: torch.Tensor,
                           delta: Dict[int, int],
                           min_keep: Optional[Dict[int, int]] = None) -> Optional[torch.Tensor]:
        H_new = torch.clamp(H, min=0).long().clone()
        keep = dict(min_keep or {})
        for su_idx, change in delta.items():
            idx = int(su_idx)
            nxt = int(H_new[idx].item()) + int(change)
            if nxt < int(keep.get(idx, 0)):
                return None
            if nxt < 0:
                return None
            H_new[idx] = int(nxt)
        return H_new

    @staticmethod
    def _summarize_hist_changes(H_before: torch.Tensor, H_after: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        rows: List[Tuple[int, int, int, int]] = []
        n = int(min(H_before.numel(), H_after.numel()))
        for idx in range(n):
            before = int(H_before[idx].item())
            after = int(H_after[idx].item())
            if before != after:
                rows.append((int(idx), int(before), int(after), int(after - before)))
        return rows

    def _print_hist_change_summary(self,
                                   stage: str,
                                   H_before: torch.Tensor,
                                   H_after: torch.Tensor,
                                   limit: int = 12) -> None:
        rows = self._summarize_hist_changes(H_before, H_after)
        if not rows:
            print(f"  [{stage}] H无变化")
            return
        print(f"  [{stage}] 结构单元调整:")
        for idx, before, after, delta in rows[:limit]:
            print(f"    SU{idx:02d} {str(self._su_name(idx)):25s}: {before} -> {after} ({delta:+d})")
        if len(rows) > int(limit):
            print(f"    ... 其余 {len(rows) - int(limit)} 项略")

    @staticmethod
    def _su_name(idx: int) -> str:
        try:
            from ...shared.coarse_graph import SU_DEFS
            if 0 <= int(idx) < len(SU_DEFS):
                return str(SU_DEFS[int(idx)][0])
        except Exception:
            pass
        return f"SU{int(idx)}"

    def _format_nonzero_histogram(self, H: torch.Tensor) -> str:
        vals = torch.clamp(H.detach().cpu(), min=0).long().tolist()
        parts = [f"SU{idx}:{int(cnt)}" for idx, cnt in enumerate(vals) if int(cnt) > 0]
        return ", ".join(parts) if parts else "(empty)"

    def _print_move_summary(self,
                            stage: str,
                            moves: List[Dict[str, Any]],
                            limit: int = 10) -> None:
        if not moves:
            return
        print(f"  [{stage}] 调整动作:")

        def _fmt_degree_counts(counts: Dict[str, Any]) -> str:
            if not isinstance(counts, dict):
                return ""
            parts = []
            for degree_i in (1, 2, 3):
                val = counts.get(int(degree_i), counts.get(str(int(degree_i)), None))
                if val is None:
                    continue
                parts.append(f"d{int(degree_i)}={int(val)}")
            return ", ".join(parts)

        for mv in moves[:limit]:
            if isinstance(mv, dict):
                parts: List[str] = []
                if 'stage' in mv:
                    parts.append(f"stage={mv['stage']}")
                if 'op' in mv:
                    parts.append(str(mv['op']))
                if 'from' in mv and 'to' in mv:
                    parts.append(f"{mv['from']}->{mv['to']}")
                if 'from_degree' in mv and 'to_degree' in mv:
                    parts.append(f"d{int(mv['from_degree'])}->d{int(mv['to_degree'])}")
                if 'delta' in mv:
                    delta = mv.get('delta', {}) or {}
                    delta_txt = ", ".join(f"{int(k)}:{int(v):+d}" for k, v in delta.items())
                    if delta_txt:
                        parts.append(f"delta[{delta_txt}]")
                if 'counts_before' in mv:
                    before_txt = _fmt_degree_counts(dict(mv.get('counts_before', {}) or {}))
                    if before_txt:
                        parts.append(f"before[{before_txt}]")
                if 'counts_after' in mv:
                    after_txt = _fmt_degree_counts(dict(mv.get('counts_after', {}) or {}))
                    if after_txt:
                        parts.append(f"after[{after_txt}]")
                print(f"    - {' | '.join(parts) if parts else str(mv)}")
            else:
                print(f"    - {mv}")
        if len(moves) > int(limit):
            print(f"    ... 其余 {len(moves) - int(limit)} 条略")

    def _format_effective_resource_summary(self,
                                           H: torch.Tensor) -> str:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        out = {
            '11': 0,
            '22': 0,
            '23': 0,
            '24': 0,
            '25': 0,
        }
        for idx in range(int(H_cpu.numel())):
            cnt = int(H_cpu[idx].item())
            if cnt <= 0:
                continue
            kind = None
            if int(idx) in {5, 6, 7, 8, 9, 11}:
                kind = '11'
            elif int(idx) in {1, 4, 16, 18, 22, 28, 32}:
                kind = '22'
            elif int(idx) in {14, 24}:
                kind = '24'
            elif int(idx) == 25:
                kind = '25'
            elif int(idx) in {0, 2, 3, 15, 17, 19, 20, 21, 23, 27, 29, 31}:
                kind = '23'
            if kind is not None:
                out[str(kind)] += int(cnt)
        return (
            f"effective[11={int(out['11'])} 22={int(out['22'])} "
            f"23={int(out['23'])} 24={int(out['24'])} 25={int(out['25'])}]"
        )

    def _print_stage_distribution_summary(self,
                                          label: str,
                                          H: torch.Tensor) -> None:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        print(f"  [{label}] literal_H = {self._format_nonzero_histogram(H_cpu)}")
        print(f"  [{label}] {self._format_effective_resource_summary(H_cpu)}")

    def _rank_tail_targets(self, ppm_arr: np.ndarray, diff_arr: np.ndarray) -> List[int]:
        scores = {}
        defaults = {
            23: 29.48,
            24: 39.97,
            25: 39.63,
        }
        for su_idx in (23, 24, 25):
            lo, hi, _ = self._get_su_common_window(
                su_idx,
                fallback_mu=defaults[su_idx],
                pad=0.0,
                min_half_width=6.0,
            )
            scores[su_idx] = self._window_stats(ppm_arr, diff_arr, lo, hi)
        ranked = sorted(
            scores.keys(),
            key=lambda su: (float(scores[su]['net']), float(scores[su]['pos']), -float(scores[su]['neg'])),
            reverse=True,
        )
        return [int(su) for su in ranked]

    def adjust_block_a_carbonyl_anchor(self,
                                       H: torch.Tensor,
                                       ppm: Optional[np.ndarray],
                                       diff: Optional[np.ndarray],
                                       max_moves: int = 6,
                                       carbonyl_max_moves: int = 2,
                                       score_rel_threshold: float = 0.02,
                                       peak_rel_threshold: float = 0.01,
                                       min_keep: int = 0) -> Tuple[torch.Tensor, List[Dict], Dict]:
        return adjust_block_a_carbonyl_anchor_impl(
            self,
            H,
            ppm,
            diff,
            max_moves=max_moves,
            carbonyl_max_moves=carbonyl_max_moves,
            score_rel_threshold=score_rel_threshold,
            peak_rel_threshold=peak_rel_threshold,
            min_keep=min_keep,
        )

    def adjust_block_b_hetero_anchor(self,
                                     H: torch.Tensor,
                                     ppm: Optional[np.ndarray],
                                     diff: Optional[np.ndarray],
                                     max_moves_each: int = 3,
                                     max_moves_total: Optional[int] = None,
                                     max_moves_count: Optional[int] = None,
                                     max_moves_mode: Optional[int] = None,
                                     peak_rel_threshold: float = 0.01,
                                     substage: Optional[str] = None,
                                     nodes: Optional[List[_NodeV3]] = None,
                                     verbose: bool = False) -> Tuple[torch.Tensor, List[Dict], Dict]:
        return adjust_block_b_hetero_anchor_impl(
            self,
            H,
            ppm,
            diff,
            max_moves_each=max_moves_each,
            max_moves_total=max_moves_total,
            max_moves_count=max_moves_count,
            max_moves_mode=max_moves_mode,
            peak_rel_threshold=peak_rel_threshold,
            substage=substage,
            nodes=nodes,
            verbose=verbose,
        )

    # ========================================================================
    # 分阶段调整接口
    # ========================================================================
    
    def _estimate_aliphatic_upper_bound(self,
                                        S_target: Optional[torch.Tensor],
                                        E_target: Optional[torch.Tensor]) -> Optional[int]:
        if S_target is None or E_target is None:
            return None
        try:
            budgets = estimate_region_carbon_budgets(S_target, E_target)
            x = float(budgets.get('x', 0.33))
            target_c = float(budgets.get('N', 0.0))
            policy = get_aliphatic_carbon_policy(E_target)
            upper_scale = float(policy.get('layer4_aliphatic_upper_scale', 0.90))
            return max(0, int(math.floor(float(upper_scale) * float(x) * float(target_c))))
        except Exception:
            return None

    def _estimate_aliphatic_region_bounds(self,
                                          S_target: Optional[torch.Tensor],
                                          E_target: Optional[torch.Tensor]) -> Dict[str, int]:
        if S_target is None or E_target is None:
            return {
                'ordinary_min': 0,
                'ordinary_max': 10**9,
                'oxygenated_min': 0,
                'oxygenated_max': 10**9,
                'total_min': 0,
                'total_max': 10**9,
            }
        try:
            budgets = estimate_region_carbon_budgets(S_target, E_target)
            target_c = float(budgets.get('N', 0.0))
            ordinary = float(budgets.get('ordinary_aliphatic_C', 0.0))
            oxygenated = float(budgets.get('oxygenated_aliphatic_C', 0.0))
            total = float(budgets.get('xN', 0.0))
            policy = get_aliphatic_carbon_policy(E_target)
            total_upper_scale = float(policy.get('layer4_aliphatic_upper_scale', 1.0))
            return {
                'ordinary_min': max(0, int(math.floor(0.50 * ordinary))),
                'ordinary_max': max(0, int(math.ceil(1.20 * ordinary + 2.0))),
                'oxygenated_min': max(0, int(math.floor(0.55 * oxygenated))),
                'oxygenated_max': max(0, int(math.ceil(1.35 * oxygenated + 2.0))),
                'total_min': max(0, int(math.floor(0.55 * total))),
                'total_max': max(0, int(math.floor(float(total_upper_scale) * max(total, ordinary + oxygenated, 0.0) + 2.0))),
                'target_c': int(round(target_c)),
                'ordinary_target': int(round(ordinary)),
                'oxygenated_target': int(round(oxygenated)),
            }
        except Exception:
            return {
                'ordinary_min': 0,
                'ordinary_max': 10**9,
                'oxygenated_min': 0,
                'oxygenated_max': 10**9,
                'total_min': 0,
                'total_max': 10**9,
            }

    @staticmethod
    def _count_map_entry(src: Dict[str, Any], key: int) -> int:
        return int(src.get(int(key), src.get(str(int(key)), 0)) or 0)

    @staticmethod
    def _mode_generic_slots(single_counts: Dict[str, Any],
                            double_counts: Dict[str, Any]) -> Tuple[int, int]:
        donor_nodes = 0
        donor_slots = 0
        for deg in [1, 2, 3]:
            cnt_single = Layer4Adjuster._count_map_entry(single_counts, int(deg))
            generic_single = max(0, int(deg) - 1)
            if int(generic_single) > 0:
                donor_nodes += int(cnt_single)
                donor_slots += int(cnt_single) * int(generic_single)
        for deg in [2, 3]:
            cnt_double = Layer4Adjuster._count_map_entry(double_counts, int(deg))
            generic_double = max(0, int(deg) - 2)
            if int(generic_double) > 0:
                donor_nodes += int(cnt_double)
                donor_slots += int(cnt_double) * int(generic_double)
        return int(donor_nodes), int(donor_slots)

    def _estimate_layer1_hist_feasibility(self,
                                          H: torch.Tensor) -> Dict[str, Any]:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        special_degree_meta = self._get_special_degree_meta(H_cpu)
        anchor_mode_meta = self._get_special_anchor_mode_meta(H_cpu)

        n11 = int(H_cpu[11].item()) if int(H_cpu.numel()) > 11 else 0
        n15 = int(H_cpu[15].item()) if int(H_cpu.numel()) > 15 else 0
        n17 = int(H_cpu[17].item()) if int(H_cpu.numel()) > 17 else 0
        n22 = int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0
        n23 = int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0
        n24 = int(H_cpu[24].item()) if int(H_cpu.numel()) > 24 else 0
        n25 = int(H_cpu[25].item()) if int(H_cpu.numel()) > 25 else 0
        n0 = int(H_cpu[0].item()) if int(H_cpu.numel()) > 0 else 0
        n1 = int(H_cpu[1].item()) if int(H_cpu.numel()) > 1 else 0
        n2 = int(H_cpu[2].item()) if int(H_cpu.numel()) > 2 else 0
        n3 = int(H_cpu[3].item()) if int(H_cpu.numel()) > 3 else 0

        mode_19 = dict(anchor_mode_meta.get(19, {}) or {})
        mode_20 = dict(anchor_mode_meta.get(20, {}) or {})
        mode_21 = dict(anchor_mode_meta.get(21, {}) or {})
        thio_single_19 = dict(mode_19.get('thio_single', {}) or {})
        thio_double_19 = dict(mode_19.get('thio_double', {}) or {})
        ether_single_19 = dict(mode_19.get('ether_single', {}) or {})
        ether_double_19 = dict(mode_19.get('ether_double', {}) or {})
        single_20 = dict(mode_20.get('single', {}) or {})
        double_20 = dict(mode_20.get('double', {}) or {})
        single_21 = dict(mode_21.get('single', {}) or {})

        donor_nodes_19a, donor_slots_19a = self._mode_generic_slots(ether_single_19, ether_double_19)
        donor_nodes_19b, donor_slots_19b = self._mode_generic_slots(thio_single_19, thio_double_19)
        donor_nodes_20, donor_slots_20 = self._mode_generic_slots(single_20, double_20)
        donor_nodes_21, donor_slots_21 = self._mode_generic_slots(single_21, {})

        su11_external_nodes = int(n15 + n17 + n22 + n23 + n24 + n25 + donor_nodes_19a + donor_nodes_19b + donor_nodes_20 + donor_nodes_21)
        su11_external_slots = int(
            n15 + n17 + n22 + 2 * n23 + 3 * n24 + 4 * n25 +
            donor_slots_19a + donor_slots_19b + donor_slots_20 + donor_slots_21
        )
        max_supportable_11 = int(su11_external_slots)
        additional_11_headroom = int(max_supportable_11 - int(n11))
        su11_external_ok = bool(int(max_supportable_11) >= int(n11))

        special_generic_demand = int(donor_slots_19a + donor_slots_19b + donor_slots_20 + donor_slots_21)
        n14 = int(H_cpu[14].item()) if int(H_cpu.numel()) > 14 else 0
        generic_partner_slots = int(
            n11 +
            n0 + n1 + n2 + n3 +
            2 * int(n14)
        )
        generic_partner_slots += int(n15 + n17 + n22 + 2 * n23 + 3 * n24 + 4 * n25)
        special_generic_ok = bool(int(special_generic_demand) <= int(generic_partner_slots) + 4)

        return {
            'su11_required': int(n11),
            'su11_external_nodes': int(su11_external_nodes),
            'su11_external_slots': int(su11_external_slots),
            'max_supportable_11': int(max_supportable_11),
            'projected_n11_after_delta': int(n11),
            'additional_11_headroom': int(additional_11_headroom),
            'su11_external_ok': bool(su11_external_ok),
            'special_generic_demand': int(special_generic_demand),
            'special_generic_partner_slots': int(generic_partner_slots),
            'special_generic_ok': bool(special_generic_ok),
        }

    def _evaluate_runtime_layer1_proxy(self,
                                       nodes: List[_NodeV3]) -> Dict[str, Any]:
        node_lookup = self._build_node_lookup(nodes)

        def _neighbor_types(node: _NodeV3) -> List[int]:
            return [int(x) for x in self._current_neighbor_types(node, nodes, node_lookup=node_lookup)]

        su11_missing_external = 0
        special_missing_external = 0
        special_fixed_anchor_gap = 0
        special_degree_gap = 0

        for node in list(nodes or []):
            su_i = int(getattr(node, 'su_type', -1))
            nb_types = _neighbor_types(node)
            current_deg = int(len(nb_types))
            try:
                target_deg = getattr(node, 'target_hop1_degree', None)
                target_deg_i = int(target_deg) if target_deg is not None else None
            except Exception:
                target_deg_i = None

            if int(su_i) == 11:
                if not any(int(x) in {15, 17, 19, 20, 21, 22, 23, 24, 25} for x in nb_types):
                    su11_missing_external += 1

            required_external: List[int] = []
            if int(su_i) == 19:
                part = str(getattr(node, 'special_anchor_partition', None) or '')
                if str(part) == 'thio':
                    required_external = [31]
                elif str(part) == 'ether':
                    if int(target_deg_i or 0) == 1:
                        required_external = [2, 29]
                    else:
                        required_external = [2, 28, 29]
            elif int(su_i) == 20:
                required_external = [0, 27]
            elif int(su_i) == 21:
                required_external = [32]

            target_fixed_cnt = 1
            try:
                raw_fixed_cnt = getattr(node, 'target_fixed_anchor_count', None)
                if raw_fixed_cnt is not None:
                    target_fixed_cnt = max(1, int(raw_fixed_cnt))
            except Exception:
                target_fixed_cnt = 1
            fixed_cnt = int(sum(1 for x in nb_types if int(x) in set(required_external)))
            if required_external and int(fixed_cnt) < int(target_fixed_cnt):
                special_missing_external += 1
                special_fixed_anchor_gap += int(target_fixed_cnt - fixed_cnt)
            if int(su_i) in {19, 20, 21} and target_deg_i is not None and int(current_deg) < int(target_deg_i):
                special_degree_gap += int(target_deg_i - int(current_deg))

        runtime_ok = bool(int(su11_missing_external) == 0 and int(special_missing_external) == 0 and int(special_degree_gap) == 0)
        return {
            'runtime_layer1_ok': bool(runtime_ok),
            'su11_missing_external': int(su11_missing_external),
            'special_missing_external': int(special_missing_external),
            'special_fixed_anchor_gap': int(special_fixed_anchor_gap),
            'special_degree_gap': int(special_degree_gap),
        }

    def _evaluate_required_hist_constraints(self,
                                            H: torch.Tensor,
                                            E_target: Optional[torch.Tensor],
                                            S_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        H_cpu = torch.clamp(H, min=0).long().detach().cpu()
        if E_target is None:
            return {
                'ok': True,
                'h_ok': True,
                'h_constraint_mode': 'soft',
                'h_topology_gate': False,
                'h_rel': 0.0,
                'h_tol': float(getattr(self, '_h_tolerance', 0.08)),
                'su22_ok': True,
                'req22': 0,
                'n22': int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0,
                'n23': int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0,
                'even10_ok': True,
                'unsat_even_ok': True,
                'unsat_total': 0,
                'reason': 'ok',
                'reasons': [],
            }

        E_target_cpu = E_target.detach().cpu().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float)
        special_degree_meta = self._get_special_degree_meta(H_cpu)
        E_pred = get_effective_hist_element_vector(H_cpu, special_degree_meta=special_degree_meta, E_SU_tensor=self.E_SU.cpu())
        current_h = float(E_pred[1].item()) if int(E_pred.numel()) > 1 else 0.0
        target_h = float(E_target_cpu[1].item()) if int(E_target_cpu.numel()) > 1 else 0.0
        h_tol = float(getattr(self, '_h_tolerance', 0.08))
        h_rel = float(abs(float(current_h) - float(target_h)) / max(abs(float(target_h)), 1.0))
        h_ok = bool(float(h_rel) <= float(h_tol))

        n22 = int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0
        n23 = int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0
        req22 = 0
        su22_ok = True

        even10_ok = True
        if int(H_cpu.numel()) > 10:
            even10_ok = bool(int(H_cpu[10].item()) % 2 == 0)

        if int(H_cpu.numel()) > 16:
            unsat_total = int(H_cpu[14].item()) + int(H_cpu[15].item()) + int(H_cpu[16].item())
            unsat_even_ok = bool(int(unsat_total) % 2 == 0)
        else:
            unsat_total = 0
            unsat_even_ok = True

        n5 = int(H_cpu[5].item()) if int(H_cpu.numel()) > 5 else 0
        n6 = int(H_cpu[6].item()) if int(H_cpu.numel()) > 6 else 0
        n7 = int(H_cpu[7].item()) if int(H_cpu.numel()) > 7 else 0
        n8 = int(H_cpu[8].item()) if int(H_cpu.numel()) > 8 else 0
        n19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
        n20 = int(H_cpu[20].item()) if int(H_cpu.numel()) > 20 else 0
        n21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
        n29 = int(H_cpu[29].item()) if int(H_cpu.numel()) > 29 else 0
        n31 = int(H_cpu[31].item()) if int(H_cpu.numel()) > 31 else 0
        n32 = int(H_cpu[32].item()) if int(H_cpu.numel()) > 32 else 0
        n0 = int(H_cpu[0].item()) if int(H_cpu.numel()) > 0 else 0
        n2 = int(H_cpu[2].item()) if int(H_cpu.numel()) > 2 else 0
        n27 = int(H_cpu[27].item()) if int(H_cpu.numel()) > 27 else 0
        n28 = int(H_cpu[28].item()) if int(H_cpu.numel()) > 28 else 0
        mode_meta = self._get_special_anchor_mode_meta(H_cpu)
        w_amine = int(n0 + 2 * n27)
        amine_mode = dict(mode_meta.get(20, {}) or {})
        amine_20_edges = int(sum(
            int(dict(amine_mode.get('single', {}) or {}).get(int(deg), 0)) +
            2 * int(dict(amine_mode.get('double', {}) or {}).get(int(deg), 0))
            for deg in [1, 2, 3]
        ))
        if int(amine_20_edges) <= 0:
            amine_20_edges = int(n20)
        amine_total = int(n6 + amine_20_edges)
        amine_ok = bool(int(amine_total) == int(w_amine))

        w_halogen = int(n32)
        halogen_mode = dict(mode_meta.get(21, {}) or {})
        halogen_21_edges = int(sum(
            int(dict(halogen_mode.get('single', {}) or {}).get(int(deg), 0))
            for deg in [2, 3]
        ))
        if int(halogen_21_edges) <= 0:
            halogen_21_edges = int(n21)
        halogen_total = int(n8 + halogen_21_edges)
        halogen_ok = bool(int(halogen_total) == int(w_halogen))

        mode_19 = dict(mode_meta.get(19, {}) or {})
        ether_single_19 = dict(mode_19.get('ether_single', {}) or {})
        ether_double_19 = dict(mode_19.get('ether_double', {}) or {})
        thio_double_19 = dict(mode_19.get('thio_double', {}) or {})
        w_thio = int(2 * n31)
        reserved_meta = dict(getattr(self, 'fixed_partition_meta', {}) or {})
        sulfur_reserved_19 = int(reserved_meta.get('s_reserved_19', max(0, int(w_thio - n7))))
        sulfur_reserved_19 = max(0, min(int(sulfur_reserved_19), int(n19)))
        sulfur_19_edges = int(sum(
            int(dict(mode_19.get('thio_single', {}) or {}).get(int(deg), 0)) +
            2 * int(dict(mode_19.get('thio_double', {}) or {}).get(int(deg), 0))
            for deg in [1, 2, 3]
        ))
        if int(sulfur_19_edges) <= 0:
            sulfur_19_edges = int(sulfur_reserved_19)
        sulfur_ok = bool(
            int(n19) >= int(sulfur_reserved_19) and
            int(n7 + sulfur_19_edges) == int(w_thio)
        )

        w_ether = int(n2 + n28 + 2 * n29)
        ether_19_edges = int(sum(
            int(dict(mode_19.get('ether_single', {}) or {}).get(int(deg), 0)) +
            2 * int(dict(mode_19.get('ether_double', {}) or {}).get(int(deg), 0))
            for deg in [1, 2, 3]
        ))
        if int(ether_19_edges) <= 0:
            ether_19_edges = max(0, int(n19 - sulfur_reserved_19))
        ether_total = int(n5 + max(0, ether_19_edges))
        ether_ok = bool(int(ether_total) == int(w_ether))

        ether_single_d1 = int(ether_single_19.get(1, ether_single_19.get('1', 0)) or 0)
        ether_double_d2 = int(ether_double_19.get(2, ether_double_19.get('2', 0)) or 0)
        ether_single_d1_raw_cap = int(n2 + n29)
        ether_single_d1_cap = max(
            0,
            int(ether_single_d1_raw_cap) - min(max(0, int(ether_single_d1_raw_cap)), max(0, int(ether_double_d2))),
        )
        ether_d1_pool_ok = bool(int(ether_single_d1) <= int(ether_single_d1_cap))
        def _nonterminal_anchor_capacity(anchor_types: List[int]) -> int:
            terminal_set = set(int(x) for x in TERMINAL_PORT_BASE_SU)
            slot_multipliers = {27: 2, 29: 2, 31: 2}
            return int(sum(
                int(H_cpu[int(su)].item()) * int(slot_multipliers.get(int(su), 1))
                for su in list(anchor_types or [])
                if int(su) not in terminal_set and 0 <= int(su) < int(H_cpu.numel())
            ))

        ether_double_d3 = int(ether_double_19.get(3, ether_double_19.get('3', 0)) or 0)
        ether_double_d3_cap = int(_nonterminal_anchor_capacity([2, 28, 29]))
        ether_double_d3_ok = bool(int(ether_double_d3) <= int(ether_double_d3_cap))
        thio_double_d3 = int(thio_double_19.get(3, thio_double_19.get('3', 0)) or 0)
        thio_double_d3_cap = int(_nonterminal_anchor_capacity([31]))
        thio_double_d3_ok = bool(int(thio_double_d3) <= int(thio_double_d3_cap))

        amine_single_20 = dict(amine_mode.get('single', {}) or {})
        amine_double_20 = dict(amine_mode.get('double', {}) or {})
        amine_single_d1 = int(amine_single_20.get(1, amine_single_20.get('1', 0)) or 0)
        amine_double_d2 = int(amine_double_20.get(2, amine_double_20.get('2', 0)) or 0)
        amine_single_d1_raw_cap = int(n0 + 2 * n27)
        amine_single_d1_cap = max(
            0,
            int(amine_single_d1_raw_cap) - min(max(0, int(amine_single_d1_raw_cap)), max(0, int(amine_double_d2))),
        )
        amine_d1_pool_ok = bool(int(amine_single_d1) <= int(amine_single_d1_cap))
        amine_double_d3 = int(amine_double_20.get(3, amine_double_20.get('3', 0)) or 0)
        amine_double_d3_cap = int(_nonterminal_anchor_capacity([0, 27]))
        amine_double_d3_ok = bool(int(amine_double_d3) <= int(amine_double_d3_cap))

        halogen_double_21 = dict(halogen_mode.get('double', {}) or {})
        halogen_double_total = int(sum(
            int(halogen_double_21.get(int(deg), halogen_double_21.get(str(deg), 0)) or 0)
            for deg in [2, 3]
        ))
        halogen_double_ok = bool(int(halogen_double_total) <= 0)

        fixed_connection_ok = bool(
            amine_ok and halogen_ok and sulfur_ok and ether_ok and
            ether_d1_pool_ok and ether_double_d3_ok and thio_double_d3_ok and
            amine_d1_pool_ok and amine_double_d3_ok and halogen_double_ok
        )
        layer1_hist_proxy = self._estimate_layer1_hist_feasibility(H_cpu)
        su11_external_ok = bool(layer1_hist_proxy.get('su11_external_ok', True))
        special_generic_ok = bool(layer1_hist_proxy.get('special_generic_ok', True))

        reasons: List[str] = []
        if not bool(even10_ok):
            reasons.append("su10_odd")
        if not bool(unsat_even_ok):
            reasons.append(f"unsat_141516_odd(total={int(unsat_total)})")
        if not bool(amine_ok):
            reasons.append(f"amine_fixed_pool_mismatch(total={int(amine_total)},required={int(w_amine)},su20_edges={int(amine_20_edges)})")
        if not bool(ether_ok):
            reasons.append(f"ether_fixed_pool_mismatch(total={int(ether_total)},required={int(w_ether)},su19_edges={int(ether_19_edges)})")
        if not bool(ether_d1_pool_ok):
            reasons.append(
                f"ether_single_d1_pool_exceeded(count={int(ether_single_d1)},cap={int(ether_single_d1_cap)},raw_cap={int(ether_single_d1_raw_cap)},double_d2={int(ether_double_d2)})"
            )
        if not bool(ether_double_d3_ok):
            reasons.append(
                f"ether_double_d3_nonterminal_pool_exceeded(count={int(ether_double_d3)},cap={int(ether_double_d3_cap)})"
            )
        if not bool(sulfur_ok):
            reasons.append(f"sulfur_fixed_pool_mismatch(reserved19={int(sulfur_reserved_19)},required={int(w_thio)},n7={int(n7)},n19={int(n19)},su19_edges={int(sulfur_19_edges)})")
        if not bool(thio_double_d3_ok):
            reasons.append(
                f"thio_double_d3_nonterminal_pool_exceeded(count={int(thio_double_d3)},cap={int(thio_double_d3_cap)})"
            )
        if not bool(halogen_ok):
            reasons.append(f"halogen_fixed_pool_mismatch(total={int(halogen_total)},required={int(w_halogen)},su21_edges={int(halogen_21_edges)})")
        if not bool(amine_d1_pool_ok):
            reasons.append(
                f"amine_single_d1_pool_exceeded(count={int(amine_single_d1)},cap={int(amine_single_d1_cap)},raw_cap={int(amine_single_d1_raw_cap)},double_d2={int(amine_double_d2)})"
            )
        if not bool(amine_double_d3_ok):
            reasons.append(
                f"amine_double_d3_nonterminal_pool_exceeded(count={int(amine_double_d3)},cap={int(amine_double_d3_cap)})"
            )
        if not bool(halogen_double_ok):
            reasons.append(f"halogen_double_mode_not_allowed(count={int(halogen_double_total)})")
        if not bool(su11_external_ok):
            reasons.append(
                f"su11_external_capacity_short(required={int(layer1_hist_proxy.get('su11_required', 0))},"
                f"slots={int(layer1_hist_proxy.get('su11_external_slots', 0))},"
                f"max_supportable_11={int(layer1_hist_proxy.get('max_supportable_11', 0))},"
                f"projected_n11_after_delta={int(layer1_hist_proxy.get('projected_n11_after_delta', 0))})"
            )
        if not bool(special_generic_ok):
            reasons.append(
                f"special_generic_capacity_short(demand={int(layer1_hist_proxy.get('special_generic_demand', 0))},"
                f"slots={int(layer1_hist_proxy.get('special_generic_partner_slots', 0))})"
            )
        reason_txt = "ok" if not reasons else "; ".join(str(x) for x in reasons)

        return {
            'ok': bool(
                even10_ok and unsat_even_ok and fixed_connection_ok and
                su11_external_ok and special_generic_ok
            ),
            'h_ok': bool(h_ok),
            'h_constraint_mode': 'soft',
            'h_topology_gate': False,
            'h_rel_exceeds_soft_tolerance': bool(not bool(h_ok)),
            'h_rel': float(h_rel),
            'h_tol': float(h_tol),
            'current_h': float(current_h),
            'target_h': float(target_h),
            'su22_ok': bool(su22_ok),
            'req22': int(req22),
            'n22': int(n22),
            'n23': int(n23),
            'even10_ok': bool(even10_ok),
            'unsat_even_ok': bool(unsat_even_ok),
            'unsat_total': int(unsat_total),
            'fixed_connection_ok': bool(fixed_connection_ok),
            'amine_ok': bool(amine_ok),
            'amine_total': int(amine_total),
            'amine_required': int(w_amine),
            'amine_20_edges': int(amine_20_edges),
            'ether_ok': bool(ether_ok),
            'ether_total': int(ether_total),
            'ether_required': int(w_ether),
            'ether_19_edges': int(ether_19_edges),
            'ether_single_d1_pool_ok': bool(ether_d1_pool_ok),
            'ether_single_d1': int(ether_single_d1),
            'ether_single_d1_cap': int(ether_single_d1_cap),
            'ether_single_d1_raw_cap': int(ether_single_d1_raw_cap),
            'ether_double_d2': int(ether_double_d2),
            'ether_double_d3_ok': bool(ether_double_d3_ok),
            'ether_double_d3': int(ether_double_d3),
            'ether_double_d3_cap': int(ether_double_d3_cap),
            'thio_double_d3_ok': bool(thio_double_d3_ok),
            'thio_double_d3': int(thio_double_d3),
            'thio_double_d3_cap': int(thio_double_d3_cap),
            'sulfur_ok': bool(sulfur_ok),
            'sulfur_reserved_19': int(sulfur_reserved_19),
            'sulfur_required': int(w_thio),
            'sulfur_19_edges': int(sulfur_19_edges),
            'halogen_ok': bool(halogen_ok),
            'halogen_total': int(halogen_total),
            'halogen_required': int(w_halogen),
            'halogen_21_edges': int(halogen_21_edges),
            'amine_single_d1_pool_ok': bool(amine_d1_pool_ok),
            'amine_single_d1': int(amine_single_d1),
            'amine_single_d1_cap': int(amine_single_d1_cap),
            'amine_single_d1_raw_cap': int(amine_single_d1_raw_cap),
            'amine_double_d2': int(amine_double_d2),
            'amine_double_d3_ok': bool(amine_double_d3_ok),
            'amine_double_d3': int(amine_double_d3),
            'amine_double_d3_cap': int(amine_double_d3_cap),
            'halogen_double_ok': bool(halogen_double_ok),
            'halogen_double_total': int(halogen_double_total),
            'layer1_hist_proxy_ok': bool(su11_external_ok and special_generic_ok),
            'su11_external_ok': bool(su11_external_ok),
            'su11_required': int(layer1_hist_proxy.get('su11_required', 0)),
            'su11_external_nodes': int(layer1_hist_proxy.get('su11_external_nodes', 0)),
            'su11_external_slots': int(layer1_hist_proxy.get('su11_external_slots', 0)),
            'max_supportable_11': int(layer1_hist_proxy.get('max_supportable_11', 0)),
            'projected_n11_after_delta': int(layer1_hist_proxy.get('projected_n11_after_delta', 0)),
            'su11_headroom': int(layer1_hist_proxy.get('additional_11_headroom', 0)),
            'special_generic_ok': bool(special_generic_ok),
            'special_generic_demand': int(layer1_hist_proxy.get('special_generic_demand', 0)),
            'special_generic_partner_slots': int(layer1_hist_proxy.get('special_generic_partner_slots', 0)),
            'reason': str(reason_txt),
            'reasons': list(reasons),
        }

    def _print_allocation_details(self, alloc_res: Any, header: str = "候选完整资源分配结果（未验收）") -> None:
        print(f"\n  [Topology-Alloc] {header}: 详见 final_topology_summary/specs/pending/resource_pool；完整诊断位于 debug/")

    def _evaluate_full_allocation_balance(self,
                                          nodes: List[_NodeV3],
                                          S_target: Optional[torch.Tensor] = None,
                                          E_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        nodes_local = list(nodes or [])
        classification = classify_topology_units(
            nodes_local,
            fixed_partition_meta=getattr(self, "fixed_partition_meta", {}) or {},
        )
        assembly = assemble_topology_units(
            nodes=nodes_local,
            classification=classification,
        )
        classification_dict = classification.to_dict()
        assembly_dict = assembly.to_dict()
        assembly_public = assembly.to_public_dict() if hasattr(assembly, "to_public_dict") else dict(assembly_dict)
        cls_summary = dict(classification_dict.get("summary", {}) or {})
        resource_pool = dict(classification_dict.get("resource_pool", {}) or {})
        assembly_ledger = dict(assembly_public.get("resource_ledger", {}) or {})
        assembly_remaining = {
            str(k): int(v)
            for k, v in dict(assembly_ledger.get("remaining_counts", {}) or {}).items()
        }
        assembly_shortage = {
            str(k): int(v)
            for k, v in dict(assembly_ledger.get("shortage_counts", {}) or {}).items()
        }
        remaining_counts = {
            str(k): int(v)
            for k, v in dict(resource_pool.get("remaining_counts", {}) or {}).items()
        }
        shortages = list(resource_pool.get("shortages", []) or [])

        def _shortage_count(su_type: int) -> int:
            return int(sum(1 for item in shortages if int(dict(item).get("su_type", -1)) == int(su_type)))

        unclassified_nodes = list(classification_dict.get("unclassified_nodes", []) or [])
        unclassified_non_resource = [
            rec for rec in unclassified_nodes
            if not bool(dict(rec).get("resource_pool_member", False))
        ]
        pending_units = list(assembly_dict.get("pending_units", []) or [])
        assembly_diag = dict(assembly_dict.get("diagnostics", {}) or {})
        return {
            "ok": bool(assembly.ok),
            "reason": str(assembly.reason),
            "warnings": [] if bool(assembly.ok) else [str(assembly.reason)],
            "topology_assembly_ok": bool(assembly.ok),
            "topology_plan_feasible": bool(assembly.feasible),
            "unrecognized_topology_node_count": int(cls_summary.get("unclassified_non_resource_count", len(unclassified_non_resource)) or 0),
            "unrecognized_topology_nodes": list(unclassified_non_resource),
            "pending_closure_total_count": int(len(pending_units)),
            "required_extra_11": int(assembly_shortage.get("11", _shortage_count(11))),
            "required_extra_22": int(assembly_shortage.get("22", _shortage_count(22))),
            "required_extra_23": int(assembly_shortage.get("23", _shortage_count(23))),
            "required_extra_24": 0,
            "required_extra_25": 0,
            "remaining": {
                "11": int(assembly_remaining.get("11", remaining_counts.get("11", 0))),
                "22": int(assembly_remaining.get("22", remaining_counts.get("22", 0))),
                "23": int(assembly_remaining.get("23", remaining_counts.get("23", 0))),
                "24": int(assembly_remaining.get("24", 0)),
                "25": 0,
            },
            "resource_ledger": assembly_ledger,
            "topology_assembly_diagnostics": {
                "fragment_count": int(cls_summary.get("fragment_count", 0) or 0),
                "closed_fragment_count": int(cls_summary.get("closed_fragment_count", 0) or 0),
                "open_fragment_count": int(cls_summary.get("open_fragment_count", 0) or 0),
                "unrecognized_topology_node_count": int(cls_summary.get("unclassified_non_resource_count", len(unclassified_non_resource)) or 0),
                "class_counts": dict(cls_summary.get("class_counts", {}) or {}),
                "family_counts": dict(cls_summary.get("family_counts", {}) or {}),
                "resource_pool": resource_pool,
                "assembly_reason": str(assembly.reason),
                "assembly_feasible": bool(assembly.feasible),
                "flex_bounds": dict(assembly_dict.get("flex_bounds", {}) or {}),
                "flex_counts": dict(assembly_dict.get("flex_counts", {}) or {}),
                "product_count": int(len(assembly_dict.get("products", []) or [])),
                "pending_unit_count": int(len(pending_units)),
                "assembly": assembly_diag,
            },
            "planned_conversion_count": int(len(assembly_dict.get("planned_conversions", []) or [])),
            "closed_topology_count": int(len(assembly_dict.get("products", []) or [])),
        }

    def plan_topology_resource_adjustments(self,
                                           H: torch.Tensor,
                                           nodes: List[_NodeV3],
                                           S_target: Optional[torch.Tensor] = None,
                                           E_target: Optional[torch.Tensor] = None,
                                           flex_max_primary_adjustments: int = 10,
                                           external_max_primary_adjustments: int = 10,
                                           external_primary_adjustments_used: int = 0,
                                           internal_max_primary_adjustments: int = 30,
                                           max_h_compensation_ops_per_adjustment: int = 8,
                                           residual_flex_preferred_23: int = 3,
                                           residual_flex_min_23: int = 2,
                                           residual_side_preferred_23: int = 2,
                                           residual_side_min_23: int = 1,
                                           h_tolerance: float = 0.08,
                                           generation_id: int = 0,
                                           difference_spectrum: Optional[Dict[str, Any]] = None,
                                           enable_post_external_spectrum: bool = True,
                                           progress_phase: Stage3ProgressPhase = Stage3ProgressPhase.FLEX_RECONCILIATION,
                                           allow_ring_recovery: bool = False,
                                           external_assembler: Any = None,
                                           spectrum_candidate_limit: int = 8,
                                           spectrum_max_candidate_trials: int = 4,
                                           spectrum_max_primary_adjustments: int = 10,
                                           rejected_spectrum_histogram_signatures: Optional[Any] = None,
                                           rejected_spectrum_bundle_signatures: Optional[Any] = None,
                                           forbidden_spectrum_inverse_signatures: Optional[Any] = None,
                                           forbidden_resource_conversion_signatures: Optional[Any] = None) -> Any:
        nodes_local = list(nodes or [])
        plan = plan_stage3_resource_adjustments(
            H=H,
            nodes=nodes_local,
            fixed_partition_meta=getattr(self, "fixed_partition_meta", {}) or {},
            S_target=S_target,
            E_target=E_target,
            flex_max_primary_adjustments=flex_max_primary_adjustments,
            external_max_primary_adjustments=external_max_primary_adjustments,
            external_primary_adjustments_used=external_primary_adjustments_used,
            allow_ring_recovery=bool(allow_ring_recovery),
            external_assembler=external_assembler,
            internal_max_primary_adjustments=internal_max_primary_adjustments,
            max_h_compensation_ops_per_adjustment=max_h_compensation_ops_per_adjustment,
            residual_flex_preferred_23=residual_flex_preferred_23,
            residual_flex_min_23=residual_flex_min_23,
            residual_side_preferred_23=residual_side_preferred_23,
            residual_side_min_23=residual_side_min_23,
            h_tolerance=h_tolerance,
            special_degree_meta=self._get_special_degree_meta(H.detach().cpu() if hasattr(H, "detach") else None),
            su11_rotation_index=int(getattr(self, "_stage3_su11_rotation_index", 0) or 0),
            generation_id=int(generation_id),
            difference_spectrum=difference_spectrum,
            enable_post_external_spectrum=bool(enable_post_external_spectrum),
            progress_phase=progress_phase,
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_candidate_trials=int(spectrum_max_candidate_trials),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            spectrum_profile_stats=self._get_su_common_stats(),
            spectrum_hop1_template_options=self._spectrum_hop1_template_options,
            rejected_spectrum_histogram_signatures=rejected_spectrum_histogram_signatures,
            rejected_spectrum_bundle_signatures=rejected_spectrum_bundle_signatures,
            forbidden_spectrum_inverse_signatures=forbidden_spectrum_inverse_signatures,
            forbidden_resource_conversion_signatures=forbidden_resource_conversion_signatures,
        )
        return plan

    def apply_topology_resource_adjustments(self,
                                            nodes: List[_NodeV3],
                                            H: torch.Tensor,
                                            plan: Stage3AdjustmentPlan) -> Stage3AdjustmentApplyResult:
        if not isinstance(plan, Stage3AdjustmentPlan):
            raise TypeError("plan must be a Stage3AdjustmentPlan")
        result = apply_stage3_node_adjustments(nodes=nodes, H=H, plan=plan)
        if bool(getattr(result, "changed", False)) and list(getattr(result, "applied_ops", []) or []):
            diagnostics = dict(plan.diagnostics or {})
            self._stage3_su11_rotation_index = int(
                diagnostics.get(
                    "su11_rotation_index_after",
                    getattr(self, "_stage3_su11_rotation_index", 0),
                ) or 0
            )
        return result

    def adjust_by_stage(self,
                       H: torch.Tensor,
                       ppm: Optional[np.ndarray],
                       diff: Optional[np.ndarray],
                       E_target: Optional[torch.Tensor],
                       S_target: Optional[torch.Tensor] = None,
                       stage: str = 'block_a',
                       **kwargs) -> Tuple[torch.Tensor, List[Dict], Dict]:
        """
        Layer4 阶段路由入口。
        
        Args:
            H: SU直方图 [33]
            ppm: PPM轴
            diff: 差谱 (target - reconstructed)
            E_target: 目标元素组成 [6]
            S_target: 目标谱图（用于碳骨架修正）
            stage:
                正式主流程推荐使用:
                - 'block_a'
                - 'block_b'
                - 'block_c_resource' -> Stage3 resource-pool conversion plan
            **kwargs:
                各阶段附加参数。
                阶段结果是否采纳由调用侧决定；Layer4 这里只负责生成候选 H 与诊断信息。
        
        Returns:
            H_adjusted: 调整后的SU直方图
            moves: 调整记录
            meta: 调整元数据
        """
        verbose = bool(kwargs.get('verbose', False))
        if verbose:
            print(f"\n{'='*80}")
            print(f"Layer4 [{stage.upper()}阶段] 调整")
            print(f"{'='*80}")
        
        H_input = torch.clamp(H, min=0).long().clone().cpu()
        H_work = H.clone()
        moves = []
        meta = {}
        self.E_target = E_target.detach().clone() if hasattr(E_target, 'detach') else E_target
        self._current_S_target = S_target.detach().cpu() if hasattr(S_target, 'detach') else S_target
        if self.layer0_estimator is not None:
            try:
                self.layer0_estimator._current_S_target = self._current_S_target
            except Exception:
                pass
        region_bounds = self._estimate_aliphatic_region_bounds(S_target, E_target)
        
        # 1. 执行对应阶段的调整
        if stage == 'block_a':
            H_work, moves, meta = self.adjust_block_a_carbonyl_anchor(
                H_work, ppm, diff,
                max_moves=kwargs.get('max_moves', 6),
                carbonyl_max_moves=kwargs.get('carbonyl_max_moves', 2),
                score_rel_threshold=kwargs.get('score_rel_threshold', 0.02),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                min_keep=kwargs.get('min_keep', 0),
            )
        elif stage == 'block_b':
            H_work, moves, meta = self.adjust_block_b_hetero_anchor(
                H_work, ppm, diff,
                max_moves_each=kwargs.get('max_moves_each', 3),
                max_moves_total=kwargs.get('max_moves_total'),
                max_moves_count=kwargs.get('max_moves_count'),
                max_moves_mode=kwargs.get('max_moves_mode'),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                substage=kwargs.get('block_b_substage'),
                nodes=kwargs.get('nodes'),
                verbose=verbose,
            )
        elif stage == 'block_c_resource':
            supported_block_c_keys = {
                'nodes',
                'rebuild_fn',
                'validate_fn',
                'output_dir',
                'flex_max_primary_adjustments',
                'external_max_primary_adjustments',
                'internal_max_primary_adjustments',
                'max_h_compensation_ops_per_adjustment',
                'residual_flex_preferred_23',
                'residual_flex_min_23',
                'residual_side_preferred_23',
                'residual_side_min_23',
                'h_tolerance',
                'outer_cycle',
                'iteration_offset',
                'context_snapshot_fn',
                'context_restore_fn',
                'difference_spectrum',
                'enable_post_external_spectrum',
                'allow_ring_recovery',
                'spectrum_active_through_cycle',
                'spectrum_candidate_limit',
                'spectrum_max_candidate_trials',
                'spectrum_max_primary_adjustments',
                'spectrum_min_r2_gain',
                'verbose',
            }
            unknown_block_c_keys = set(kwargs) - supported_block_c_keys
            if unknown_block_c_keys:
                raise ValueError(
                    'block_c_resource contains unknown keys: '
                    + ', '.join(
                        sorted(str(key) for key in unknown_block_c_keys)
                    )
                )
            workflow_result = run_block_c_resource_adjustment_loop_impl(
                self,
                H_work,
                nodes=kwargs.get('nodes'),
                E_target=E_target,
                S_target=S_target,
                rebuild_fn=kwargs.get('rebuild_fn'),
                validate_fn=kwargs.get('validate_fn'),
                output_dir=kwargs.get('output_dir'),
                flex_max_primary_adjustments=kwargs.get('flex_max_primary_adjustments', 10),
                external_max_primary_adjustments=kwargs.get('external_max_primary_adjustments', 10),
                internal_max_primary_adjustments=kwargs.get('internal_max_primary_adjustments', 30),
                max_h_compensation_ops_per_adjustment=kwargs.get(
                    'max_h_compensation_ops_per_adjustment', 8
                ),
                residual_flex_preferred_23=kwargs.get('residual_flex_preferred_23', 3),
                residual_flex_min_23=kwargs.get('residual_flex_min_23', 2),
                residual_side_preferred_23=kwargs.get('residual_side_preferred_23', 2),
                residual_side_min_23=kwargs.get('residual_side_min_23', 1),
                h_tolerance=kwargs.get('h_tolerance', 0.08),
                outer_cycle=kwargs.get('outer_cycle', 0),
                # Each Outer Cycle owns an independent Stage3 cycle counter;
                # legacy offsets must not leak into spectrum cutoff routing.
                iteration_offset=0,
                context_snapshot_fn=kwargs.get('context_snapshot_fn'),
                context_restore_fn=kwargs.get('context_restore_fn'),
                difference_spectrum=kwargs.get('difference_spectrum'),
                enable_post_external_spectrum=kwargs.get('enable_post_external_spectrum', True),
                allow_ring_recovery=kwargs.get('allow_ring_recovery', False),
                spectrum_active_through_cycle=kwargs.get('spectrum_active_through_cycle', 20),
                spectrum_candidate_limit=kwargs.get('spectrum_candidate_limit', 8),
                spectrum_max_candidate_trials=kwargs.get('spectrum_max_candidate_trials', 4),
                spectrum_max_primary_adjustments=kwargs.get('spectrum_max_primary_adjustments', 10),
                spectrum_min_r2_gain=kwargs.get('spectrum_min_r2_gain', 0.0005),
                verbose=kwargs.get('verbose', False),
            )
            H_work = torch.clamp(getattr(workflow_result, 'H', H_work).detach().clone(), min=0).long()
            meta = workflow_result.to_dict()
            moves = []
            for cycle in list(getattr(workflow_result, 'cycles', []) or []):
                diag = dict(getattr(cycle, 'diagnostics', {}) or {})
                moves.extend(list(diag.get('planned_conversions', []) or []))
            meta['h_rotation_meta'] = {
                'applied': False,
                'reason': 'stage3_resource_adjustment_only',
            }
            if verbose:
                self._print_hist_change_summary('BLOCK_C-RESOURCE', H_input, torch.clamp(H_work, min=0).long().cpu())
                self._print_move_summary('BLOCK_C-RESOURCE', moves)
                self._print_stage_distribution_summary('BLOCK_C-RESOURCE final_literal_vs_effective', torch.clamp(H_work, min=0).long().cpu())
                print(f"\nBLOCK_C-RESOURCE候选调整完成: 共{len(moves)}次变更")
            return H_work, moves, meta

        else:
            print(f"  [警告] 未知的调整阶段: {stage}")
            return H_work, moves, meta

        H_final = torch.clamp(H_work, min=0).long().cpu()
        if verbose:
            self._print_hist_change_summary(stage.upper(), H_input, H_final)
            self._print_move_summary(stage.upper(), moves)
            self._print_stage_distribution_summary(
                f"{stage.upper()} final_literal_vs_effective", H_final
            )
            print(f"\n{stage.upper()}阶段调整完成: 共{len(moves)}次变更")
        elif stage == 'block_b' and moves and bool(kwargs.get('emit_summary', True)):
            changes = ", ".join(
                f"SU{idx}:{before}->{after}"
                for idx, before, after, _delta in self._summarize_hist_changes(
                    H_input, H_final
                )
            ) or "mode-only"
            substage = str(kwargs.get('block_b_substage', 'all') or 'all')
            print(
                f"[Block B/{substage}] changes={len(moves)} {changes}",
                flush=True,
            )
        return H_work, moves, meta

    def run_block_c_resource_adjustment_loop(
        self,
        H: torch.Tensor,
        nodes: Optional[List[_NodeV3]],
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        rebuild_fn: Any = None,
        validate_fn: Any = None,
        output_dir: Optional[str] = None,
        flex_max_primary_adjustments: int = 10,
        external_max_primary_adjustments: int = 10,
        internal_max_primary_adjustments: int = 30,
        max_h_compensation_ops_per_adjustment: int = 8,
        residual_flex_preferred_23: int = 3,
        residual_flex_min_23: int = 2,
        residual_side_preferred_23: int = 2,
        residual_side_min_23: int = 1,
        h_tolerance: float = 0.08,
        outer_cycle: int = 0,
        iteration_offset: int = 0,
        context_snapshot_fn: Any = None,
        context_restore_fn: Any = None,
        difference_spectrum: Optional[Dict[str, Any]] = None,
        enable_post_external_spectrum: bool = True,
        spectrum_active_through_cycle: int = 20,
        spectrum_candidate_limit: int = 8,
        spectrum_max_candidate_trials: int = 4,
        spectrum_max_primary_adjustments: int = 10,
        spectrum_min_r2_gain: float = 0.0005,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise ValueError(
                'block_c_resource contains unknown keys: '
                + ', '.join(sorted(str(key) for key in kwargs))
            )
        return run_block_c_resource_adjustment_loop_impl(
            self,
            H=H,
            nodes=nodes,
            E_target=E_target,
            S_target=S_target,
            rebuild_fn=rebuild_fn,
            validate_fn=validate_fn,
            output_dir=output_dir,
            flex_max_primary_adjustments=int(flex_max_primary_adjustments),
            external_max_primary_adjustments=int(external_max_primary_adjustments),
            internal_max_primary_adjustments=int(internal_max_primary_adjustments),
            max_h_compensation_ops_per_adjustment=int(max_h_compensation_ops_per_adjustment),
            residual_flex_preferred_23=int(residual_flex_preferred_23),
            residual_flex_min_23=int(residual_flex_min_23),
            residual_side_preferred_23=int(residual_side_preferred_23),
            residual_side_min_23=int(residual_side_min_23),
            h_tolerance=float(h_tolerance),
            outer_cycle=int(outer_cycle),
            # ``iteration_offset`` is retained for old callers only.  Stage3
            # always starts at local Cycle 1 for each Outer Cycle invocation.
            iteration_offset=0,
            context_snapshot_fn=context_snapshot_fn,
            context_restore_fn=context_restore_fn,
            difference_spectrum=difference_spectrum,
            enable_post_external_spectrum=bool(enable_post_external_spectrum),
            spectrum_active_through_cycle=int(spectrum_active_through_cycle),
            spectrum_candidate_limit=int(spectrum_candidate_limit),
            spectrum_max_candidate_trials=int(spectrum_max_candidate_trials),
            spectrum_max_primary_adjustments=int(spectrum_max_primary_adjustments),
            spectrum_min_r2_gain=float(spectrum_min_r2_gain),
            verbose=bool(verbose),
        )
    
