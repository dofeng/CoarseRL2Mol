import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
from collections import Counter
import math
import io
from contextlib import redirect_stdout

from ...paths import Z_LIBRARY_DIR
from ...shared.inverse_common import (
    SU_ALIPHATIC, E_SU, _NodeV3,
    SPECIAL_D3_TERMINAL_NEIGHBORS,
    SPECIAL_DEGREE_PRIORS,
    get_aliphatic_carbon_policy,
    get_effective_hist_element_vector,
    estimate_region_carbon_budgets,
    normalize_special_degree_meta,
    rebuild_su19_partition_meta,
)
from RL_MTCS.RL_allocator import FlexAllocator
from RL_MTCS.RL_init import ClusterGenerator
from ..layer0.inverse_layer0 import Layer0Estimator
from .inverse_layer4_block_a import (
    adjust_block_a_carbonyl_anchor_impl,
)
from .inverse_layer4_block_b import (
    adjust_block_b_hetero_anchor_impl,
)
from .inverse_layer4_block_c import (
    adjust_block_c_aliphatic_tail_impl,
    adjust_block_c_branch_phase_impl,
    adjust_block_c_extra_phase_impl,
)

class Layer4Adjuster:
    """
    Layer4: 基于差谱的多阶段SU直方图调整器

    当前正式主流程只保留以下阶段：
    1. block_a: 羰基/锚点修正
    2. block_b: 异原子锚点修正
    3. block_c:
       - tail_diff
       - branch_topology
       - extra_global

    Layer4 只负责基于差谱/资源分配做 SU 计数与拓扑修正；
    Layer0 修正仅用于初始化阶段，不在 Layer4 中重复调用。

    调用侧约定（见 InversePipelineV3.infer）：
    - block_c / tail_diff 作为候选微调阶段，是否采纳由调用侧的
      Layer2 improvement gate 决定。
    - block_c_branch / block_c_extra 是结构与资源修复阶段；
      只要调用侧完成重跑，就直接采纳该阶段结果，不与 A/B/tail
      共用 improvement gate。
    """
    
    def __init__(self, 
                 device: torch.device = None,
                 layer0_estimator: Optional[Layer0Estimator] = None,
                 su_hop1_ranges_path: Optional[str] = None,
                 su_common_ranges_path: Optional[str] = None,
                 su_special_degree_ranges_path: Optional[str] = None):
        self.device = device or torch.device('cpu')
        self.E_SU = E_SU.to(self.device)
        self.layer0_estimator = layer0_estimator
        self.fixed_partition_meta: Dict[str, Any] = {}
        
        # 加载 hop1 NMR范围数据
        if su_hop1_ranges_path is None:
            default_path = Z_LIBRARY_DIR / 'su_hop1_nmr_range_filtered.csv'
            self.su_hop1_ranges_path = str(default_path) if default_path.exists() else None
        else:
            self.su_hop1_ranges_path = su_hop1_ranges_path
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
        # Persist H-adjust rotation state across repeated block_c branch/extra calls.
        self._h_rotation_state = 0
        self._rigid10_rotation_state = 0
        self._h_rotation_aliphatic_cap: Optional[int] = None
        self._h_rotation_ordinary_aliphatic_cap: Optional[int] = None
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
                                     nodes: Optional[List[_NodeV3]] = None) -> Tuple[torch.Tensor, List[Dict], Dict]:
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
        )

    def adjust_block_c_aliphatic_tail(self,
                                      H: torch.Tensor,
                                      ppm: Optional[np.ndarray],
                                      diff: Optional[np.ndarray],
                                      E_target: Optional[torch.Tensor] = None,
                                      max_moves: int = 6,
                                      peak_rel_threshold: float = 0.01,
                                      min_keep_22: int = 1,
                                      min_keep_23: int = 0,
                                      min_keep_24: int = 0,
                                      carbonyl_couple: bool = True,
                                      h_tolerance: float = 0.08) -> Tuple[torch.Tensor, List[Dict], Dict]:
        return adjust_block_c_aliphatic_tail_impl(
            self,
            H,
            ppm,
            diff,
            E_target=E_target,
            max_moves=max_moves,
            peak_rel_threshold=peak_rel_threshold,
            min_keep_22=min_keep_22,
            min_keep_23=min_keep_23,
            min_keep_24=min_keep_24,
            carbonyl_couple=carbonyl_couple,
            h_tolerance=h_tolerance,
        )

    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    # ========================================================================
    # 分阶段调整接口
    # ========================================================================
    
    def _make_h_helpers(self):
        def _current_h(tmp: torch.Tensor):
            meta = self._get_special_degree_meta(tmp)
            eff = get_effective_hist_element_vector(tmp, special_degree_meta=meta, E_SU_tensor=self.E_SU.cpu())
            return float(eff[1].item())
            
        def _h_ratio(tmp: torch.Tensor):
            target_H = float(self.E_target[1].item())
            if target_H <= 0: return 0.0
            return (_current_h(tmp) - target_H) / target_H
            
        def _check_h(tmp: torch.Tensor, tol: float = 0.08):
            return abs(_h_ratio(tmp)) <= tol
            
        def _ali_total(tmp: torch.Tensor):
            return int(sum(tmp[i].item() for i in SU_ALIPHATIC))
            
        return _current_h, _h_ratio, _check_h, _ali_total

    @staticmethod
    def _can_increase_su12(H_work: torch.Tensor, inc12: int = 1, dec13: int = 0) -> bool:
        try:
            n12 = int(H_work[12].item())
            n13 = int(H_work[13].item())
        except Exception:
            return True
        n12_new = int(n12 + int(inc12))
        n13_new = int(n13 - int(dec13))
        return int(n12_new) <= int(max(n13_new, 0))

    @staticmethod
    def _h_rotation_adjust(tmp_nodes,
                           H_work,
                           h_ratio_fn,
                           rot_idx,
                           max_ops: Optional[int] = None,
                           max_aliphatic_total: Optional[int] = None,
                           max_ordinary_aliphatic_total: Optional[int] = None,
                           h_tolerance: float = 0.08):
        ops = []
        failed_steps = 0
        max_ops_i = None if max_ops is None else max(0, int(max_ops))

        def _aliphatic_total() -> int:
            return int(sum(int(H_work[i].item()) for i in SU_ALIPHATIC))

        def _ordinary_aliphatic_total() -> int:
            return int(sum(int(H_work[i].item()) for i in (22, 23, 24, 25)))
        
        tol = max(0.0, float(h_tolerance))
        while abs(h_ratio_fn(H_work)) > float(tol):
            if max_ops_i is not None and len(ops) >= int(max_ops_i):
                break
            ratio = h_ratio_fn(H_work)
            step_type = rot_idx % 5
            success = False
            
            if ratio > float(tol):
                if step_type in [0, 2]:
                    if not Layer4Adjuster._can_increase_su12(H_work, inc12=1, dec13=1):
                        rot_idx += 1
                        failed_steps += 1
                        if failed_steps >= 5:
                            print("    [H调整] 连续5步轮转失败，无法继续调H")
                            break
                        continue
                    for n in tmp_nodes:
                        if n.su_type == 13:
                            n.su_type = 12
                            H_work[13] -= 1; H_work[12] += 1
                            ops.append('H:13->12')
                            success = True
                            break
                elif step_type in [1, 3]:
                    for n in tmp_nodes:
                        if n.su_type == 23:
                            n.su_type = 13
                            H_work[23] -= 1; H_work[13] += 1
                            ops.append('H:23->13')
                            success = True
                            break
                elif step_type == 4:
                    n13 = int(H_work[13].item())
                    n14 = int(H_work[14].item())
                    n15 = int(H_work[15].item())
                    n16 = int(H_work[16].item())
                    total_unsat = int(n14 + n15 + n16)
                    min_unsat_pool = 0.05 * float(max(n13, 1))
                    if float(total_unsat) >= float(min_unsat_pool) and Layer4Adjuster._can_increase_su12(H_work, inc12=1, dec13=0):
                        pairs = []
                        if int(n15) >= 1 and int(n16) >= 1 and int(n16 - 1) <= int(n14 + n15 - 1):
                            pairs.append((15, 16))
                        if int(n15) >= 2:
                            pairs.append((15, 15))
                        if int(n14) >= 1 and int(n15) >= 1 and int(n14) <= int(n15):
                            pairs.append((14, 15))
                        for p in pairs:
                            n_a, n_b = None, None
                            for n in tmp_nodes:
                                if n.su_type == p[0] and n_a is None: n_a = n
                                elif n.su_type == p[1] and n_a != n and n_b is None: n_b = n
                            if n_a and n_b:
                                n_a.su_type = 12
                                n_b.su_type = 13
                                H_work[p[0]] -= 1; H_work[12] += 1
                                H_work[p[1]] -= 1; H_work[13] += 1
                                ops.append(f'H:{p[0]}+{p[1]}->12+13')
                                success = True
                                break

            elif ratio < -float(tol):
                if step_type in [0, 2]:
                    for n in tmp_nodes:
                        if n.su_type == 12:
                            n.su_type = 13
                            H_work[12] -= 1; H_work[13] += 1
                            ops.append('H:12->13')
                            success = True
                            break
                elif step_type in [1, 3]:
                    can_grow_aliphatic = True
                    if max_aliphatic_total is not None:
                        can_grow_aliphatic = bool(int(_aliphatic_total()) < int(max_aliphatic_total))
                    if max_ordinary_aliphatic_total is not None:
                        can_grow_aliphatic = bool(
                            can_grow_aliphatic and
                            int(_ordinary_aliphatic_total()) < int(max_ordinary_aliphatic_total)
                        )
                    if can_grow_aliphatic:
                        for n in tmp_nodes:
                            if n.su_type == 13:
                                n.su_type = 23
                                H_work[13] -= 1; H_work[23] += 1
                                ops.append('H:13->23')
                                success = True
                                break
                elif step_type == 4:
                    n12 = int(H_work[12].item())
                    n13 = int(H_work[13].item())
                    n14 = int(H_work[14].item())
                    n15 = int(H_work[15].item())
                    n16 = int(H_work[16].item())
                    aro_5_13 = int(sum(int(H_work[i].item()) for i in range(5, 14)))
                    next_aro_5_13 = max(0, int(aro_5_13 - 2))
                    unsat_cap = max(0, int(math.floor(0.07 * float(max(next_aro_5_13, 1)))))

                    def _pair_allowed(add14: int, add15: int, add16: int) -> bool:
                        new14 = int(n14 + add14)
                        new15 = int(n15 + add15)
                        new16 = int(n16 + add16)
                        new_total = int(new14 + new15 + new16)
                        if int(new_total) > int(unsat_cap):
                            return False
                        if int(new14) > int(new15):
                            return False
                        if int(new16) > int(new14 + new15):
                            return False
                        return True

                    if int(n12) >= 1 and int(n13) >= 1:
                        pairs = []
                        if _pair_allowed(0, 2, 0):
                            pairs.append(((15, 15), 'H:12+13->15+15'))
                        if _pair_allowed(0, 1, 1):
                            pairs.append(((15, 16), 'H:12+13->15+16'))
                        if _pair_allowed(1, 1, 0):
                            pairs.append(((14, 15), 'H:12+13->14+15'))

                        for (dst_a, dst_b), op_name in pairs:
                            n_12, n_13 = None, None
                            for n in tmp_nodes:
                                if n.su_type == 12 and n_12 is None:
                                    n_12 = n
                                elif n.su_type == 13 and n_13 is None:
                                    n_13 = n
                                if n_12 is not None and n_13 is not None:
                                    break
                            if n_12 is None or n_13 is None:
                                break
                            n_12.su_type = int(dst_a)
                            n_13.su_type = int(dst_b)
                            H_work[12] -= 1; H_work[int(dst_a)] += 1
                            H_work[13] -= 1; H_work[int(dst_b)] += 1
                            ops.append(str(op_name))
                            success = True
                            break
                    
            rot_idx += 1
            if success:
                failed_steps = 0
            else:
                failed_steps += 1
                if failed_steps >= 5:
                    print("    [H调整] 连续5步轮转失败，无法继续调H")
                    break
        return ops, rot_idx

    def _apply_h_rotation_to_counts(self,
                                    H: torch.Tensor,
                                    E_target: Optional[torch.Tensor],
                                    max_ops: Optional[int] = None) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
        H_work = torch.clamp(H, min=0).long().clone().cpu()
        if E_target is None:
            return H_work.to(H.device), [], {'applied': False, 'reason': 'missing_E_target'}

        self.E_target = E_target.detach().cpu() if hasattr(E_target, 'detach') else E_target
        tmp_nodes: List[_NodeV3] = []
        gid = 0
        for su_idx in range(int(H_work.numel())):
            count = int(H_work[su_idx].item())
            for _ in range(max(0, count)):
                tmp_nodes.append(_NodeV3(gid, int(su_idx)))
                gid += 1

        _, h_ratio_fn, _, _ = self._make_h_helpers()
        before_ratio = float(h_ratio_fn(H_work))
        rot_ops, rot_idx = self._h_rotation_adjust(
            tmp_nodes,
            H_work,
            h_ratio_fn,
            int(self._h_rotation_state),
            max_ops=max_ops,
            max_aliphatic_total=getattr(self, '_h_rotation_aliphatic_cap', None),
            max_ordinary_aliphatic_total=getattr(self, '_h_rotation_ordinary_aliphatic_cap', None),
            h_tolerance=float(getattr(self, '_h_tolerance', 0.08)),
        )
        self._h_rotation_state = int(rot_idx)
        after_ratio = float(h_ratio_fn(H_work))

        moves = [{'stage': 'h_rotation', 'op': str(op)} for op in rot_ops]
        meta = {
            'applied': bool(rot_ops),
            'ops': list(rot_ops),
            'before_ratio': float(before_ratio),
            'after_ratio': float(after_ratio),
            'rotation_state': int(self._h_rotation_state),
            'max_ops': (None if max_ops is None else int(max_ops)),
        }
        return H_work.to(H.device), moves, meta

    def _derive_bridgehead_info_from_nodes(self, nodes: List[_NodeV3]) -> Tuple[int, int, int]:
        node_lookup = self._build_node_lookup(nodes)
        m, n, p = 0, 0, 0
        any_hop2 = False
        for node in nodes:
            if int(getattr(node, 'su_type', -1)) != 12:
                continue
            hop2_counter = self._current_hop2_counter(node, nodes, node_lookup=node_lookup)
            count_12 = int(hop2_counter.get(12, 0))
            if int(sum(hop2_counter.values())) > 0:
                any_hop2 = True
            if count_12 >= 2:
                m += 1
            elif count_12 == 1:
                n += 1
            else:
                p += 1
        if not any_hop2:
            p = sum(1 for node in nodes if int(getattr(node, 'su_type', -1)) == 12)
        return int(m), int(n), int(p)

    def _compute_aromatic_cluster_metrics(self, nodes: List[_NodeV3]) -> Dict[str, Any]:
        su_counts = Counter()
        for node in nodes:
            try:
                su_counts[int(node.su_type)] += 1
            except Exception:
                continue
        bridgehead_info = self._derive_bridgehead_info_from_nodes(nodes)
        gen = ClusterGenerator(dict(su_counts), bridgehead_info=bridgehead_info)
        clusters = gen.generate()
        kind_counts = Counter(getattr(c, 'kind', 'unknown') for c in clusters)
        return {
            'cluster_count': int(len(clusters)),
            'bridgehead_info': tuple(int(x) for x in bridgehead_info),
            'converted_13': float(gen.n13),
            'converted_12': int(gen.n12),
            'remaining_12': int(gen.remaining_12),
            'remaining_13': float(gen.remaining_13),
            'original_12': int(getattr(gen, 'original_12', 0)),
            'used_12_to_13': int(getattr(gen, 'used_12_to_13', 0)),
            'used_13_to_12': int(getattr(gen, 'used_13_to_12', 0)),
            'synthetic_13_topup_used': int(getattr(gen, 'synthetic_13_topup_used', 0)),
            'cluster_kind_counts': {str(k): int(v) for k, v in sorted(kind_counts.items())},
        }

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

    def _estimate_aromatic_ch_target(self,
                                     S_target: Optional[torch.Tensor],
                                     E_target: Optional[torch.Tensor],
                                     H: Optional[torch.Tensor] = None) -> int:
        if S_target is not None and E_target is not None:
            try:
                spec = S_target.detach().cpu().float().flatten()
                target_c = float(E_target.detach().cpu().flatten()[0].item())
                total_area = float(spec.sum().item()) * 0.1
                if int(spec.numel()) > 0 and float(total_area) > 1e-8 and float(target_c) > 0.0:
                    lo_i = max(0, int(math.floor(115.0 / 0.1)))
                    hi_i = min(int(spec.numel()), int(math.ceil(135.0 / 0.1)))
                    if hi_i > lo_i:
                        area = float(spec[lo_i:hi_i].sum().item()) * 0.1
                        return max(0, int(round(float(area / total_area) * float(target_c))))
            except Exception:
                pass

        if H is not None:
            try:
                H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
                aromatic_pool = int(sum(
                    int(H_cpu[int(su)].item())
                    for su in [10, 11, 12, 13]
                    if int(su) < int(H_cpu.numel())
                ))
                return max(0, int(round(0.35 * float(aromatic_pool))))
            except Exception:
                pass
        return 0

    def _evaluate_aromatic_balance_constraints(self,
                                               H: torch.Tensor,
                                               S_target: Optional[torch.Tensor] = None,
                                               E_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
        n12 = int(H_cpu[12].item()) if int(H_cpu.numel()) > 12 else 0
        n13 = int(H_cpu[13].item()) if int(H_cpu.numel()) > 13 else 0
        aromatic_ch_target = int(self._estimate_aromatic_ch_target(S_target, E_target, H_cpu))
        min13 = max(1 if int(n12 + n13) > 0 else 0, int(math.ceil(0.20 * float(aromatic_ch_target))))
        max12 = int(math.floor(2.5 * float(max(1, int(n13)))))
        su13_ok = bool(int(n13) >= int(min13))
        su12_ok = bool(int(n12) <= int(max12))
        reasons: List[str] = []
        if not bool(su13_ok):
            reasons.append(f"su13_below_min(count={int(n13)},min={int(min13)},target={int(aromatic_ch_target)})")
        if not bool(su12_ok):
            reasons.append(f"su12_exceeds_ratio(count={int(n12)},max={int(max12)},su13={int(n13)})")
        return {
            'ok': bool(su13_ok and su12_ok),
            'su13_ok': bool(su13_ok),
            'su12_ok': bool(su12_ok),
            'su12': int(n12),
            'su13': int(n13),
            'su13_min': int(min13),
            'su12_max': int(max12),
            'aromatic_ch_target': int(aromatic_ch_target),
            'reasons': list(reasons),
            'reason': 'ok' if not reasons else '; '.join(str(x) for x in reasons),
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
        su11_external_ok = bool(int(su11_external_slots) >= int(n11))

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
                'h_rel': 0.0,
                'h_tol': float(getattr(self, '_h_tolerance', 0.08)),
                'su22_ok': True,
                'req22': 0,
                'n22': int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0,
                'n23': int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0,
                'even10_ok': True,
                'unsat_even_ok': True,
                'unsat_total': 0,
                'aromatic_balance_ok': True,
                'reason': 'ok',
                'reasons': [],
            }

        E_target_cpu = E_target.detach().cpu().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float)
        special_degree_meta = self._get_special_degree_meta(H_cpu)
        E_pred = get_effective_hist_element_vector(H_cpu, special_degree_meta=special_degree_meta, E_SU_tensor=self.E_SU.cpu())
        target_h = float(E_target_cpu[1].item()) if int(E_target_cpu.numel()) > 1 else 0.0
        current_h = float(E_pred[1].item()) if int(E_pred.numel()) > 1 else 0.0
        h_tol = float(getattr(self, '_h_tolerance', 0.08))
        if target_h > 1e-8:
            h_rel = abs(float(current_h - target_h)) / float(target_h)
            h_ok = bool(h_rel <= float(h_tol) + 1e-9)
        else:
            h_rel = 0.0
            h_ok = True

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
        aromatic_balance = self._evaluate_aromatic_balance_constraints(
            H_cpu,
            S_target=S_target,
            E_target=E_target_cpu,
        )

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
        ether_single_d1_cap = int(n2 + n29)
        ether_d1_pool_ok = bool(int(ether_single_d1) <= int(ether_single_d1_cap))
        def _nonterminal_anchor_capacity(anchor_types: List[int]) -> int:
            terminal_set = set(int(x) for x in SPECIAL_D3_TERMINAL_NEIGHBORS)
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
        amine_single_d1_cap = int(n0 + 2 * n27)
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
        if not bool(h_ok):
            reasons.append(f"h_rel_exceeds_tol({float(h_rel):.4f}>{float(h_tol):.4f})")
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
                f"ether_single_d1_pool_exceeded(count={int(ether_single_d1)},cap={int(ether_single_d1_cap)})"
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
                f"amine_single_d1_pool_exceeded(count={int(amine_single_d1)},cap={int(amine_single_d1_cap)})"
            )
        if not bool(amine_double_d3_ok):
            reasons.append(
                f"amine_double_d3_nonterminal_pool_exceeded(count={int(amine_double_d3)},cap={int(amine_double_d3_cap)})"
            )
        if not bool(halogen_double_ok):
            reasons.append(f"halogen_double_mode_not_allowed(count={int(halogen_double_total)})")
        if not bool(aromatic_balance.get('ok', True)):
            reasons.extend(str(x) for x in list(aromatic_balance.get('reasons', []) or []))
        if not bool(su11_external_ok):
            reasons.append(
                f"su11_external_capacity_short(required={int(layer1_hist_proxy.get('su11_required', 0))},"
                f"slots={int(layer1_hist_proxy.get('su11_external_slots', 0))})"
            )
        if not bool(special_generic_ok):
            reasons.append(
                f"special_generic_capacity_short(demand={int(layer1_hist_proxy.get('special_generic_demand', 0))},"
                f"slots={int(layer1_hist_proxy.get('special_generic_partner_slots', 0))})"
            )
        reason_txt = "ok" if not reasons else "; ".join(str(x) for x in reasons)

        return {
            'ok': bool(
                h_ok and even10_ok and unsat_even_ok and fixed_connection_ok and
                su11_external_ok and special_generic_ok and bool(aromatic_balance.get('ok', True))
            ),
            'h_ok': bool(h_ok),
            'h_rel': float(h_rel),
            'h_tol': float(h_tol),
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
            'amine_double_d3_ok': bool(amine_double_d3_ok),
            'amine_double_d3': int(amine_double_d3),
            'amine_double_d3_cap': int(amine_double_d3_cap),
            'halogen_double_ok': bool(halogen_double_ok),
            'halogen_double_total': int(halogen_double_total),
            'aromatic_balance_ok': bool(aromatic_balance.get('ok', True)),
            'aromatic_balance': dict(aromatic_balance),
            'layer1_hist_proxy_ok': bool(su11_external_ok and special_generic_ok),
            'su11_external_ok': bool(su11_external_ok),
            'su11_required': int(layer1_hist_proxy.get('su11_required', 0)),
            'su11_external_nodes': int(layer1_hist_proxy.get('su11_external_nodes', 0)),
            'su11_external_slots': int(layer1_hist_proxy.get('su11_external_slots', 0)),
            'special_generic_ok': bool(special_generic_ok),
            'special_generic_demand': int(layer1_hist_proxy.get('special_generic_demand', 0)),
            'special_generic_partner_slots': int(layer1_hist_proxy.get('special_generic_partner_slots', 0)),
            'reason': str(reason_txt),
            'reasons': list(reasons),
        }

    @staticmethod
    def _pick_nodes_by_type(nodes: List[_NodeV3], su_type: int, count: int) -> List[_NodeV3]:
        picked = [n for n in nodes if int(getattr(n, 'su_type', -1)) == int(su_type)]
        picked.sort(key=lambda n: (int(getattr(n, 'global_id', 0))))
        return picked[:max(0, int(count))]

    def _apply_node_type_conversion(self,
                                    nodes: List[_NodeV3],
                                    src_type: int,
                                    dst_type: int,
                                    count: int = 1) -> int:
        if int(dst_type) < 0:
            return 0
        picked = self._pick_nodes_by_type(nodes, int(src_type), int(count))
        for node in picked:
            node.su_type = int(dst_type)
        if picked:
            self._refresh_node_counters(nodes)
        return int(len(picked))

    def _apply_post_moves_to_nodes(self,
                                   nodes: Optional[List[_NodeV3]],
                                   moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if nodes is None:
            return []
        synced: List[Dict[str, Any]] = []
        for mv in list(moves or []):
            applied = 0
            if isinstance(mv, dict) and 'from' in mv and 'to' in mv:
                try:
                    src = int(mv['from'])
                    dst = int(mv['to'])
                except Exception:
                    src = dst = -999
                if int(dst) >= 0:
                    applied = self._apply_node_type_conversion(nodes, src, dst, 1)
            elif isinstance(mv, dict) and 'op' in mv:
                op = str(mv.get('op', ''))
                if op.startswith('H:') and '->' in op:
                    left, right = op[2:].split('->', 1)
                    src_parts = [p for p in left.split('+') if p]
                    dst_parts = [p for p in right.split('+') if p]
                    if len(src_parts) == len(dst_parts):
                        ok = True
                        total = 0
                        for src_txt, dst_txt in zip(src_parts, dst_parts):
                            try:
                                src = int(src_txt)
                                dst = int(dst_txt)
                            except Exception:
                                ok = False
                                break
                            n_applied = self._apply_node_type_conversion(nodes, src, dst, 1)
                            if n_applied <= 0:
                                ok = False
                                break
                            total += int(n_applied)
                        applied = int(total) if ok else 0
            if applied > 0:
                rec = dict(mv)
                rec['node_sync_applied'] = int(applied)
                synced.append(rec)
        return synced

    def _apply_aromatic_cluster_alignment(self,
                                          nodes: Optional[List[_NodeV3]],
                                          H_work: torch.Tensor,
                                          protect_11: bool = False) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Materialize ClusterGenerator's 12<->13 balancing on the current node list.

        This keeps the block_c third-phase SU counts closer to the aromatic-cluster model,
        reducing the need for RL_init.py to silently perform the same conversion later.
        """
        H_new = torch.clamp(H_work, min=0).long().clone().cpu()
        if nodes is None:
            return H_new.to(H_work.device), [], {'applied': False, 'reason': 'missing_nodes'}

        before = self._compute_aromatic_cluster_metrics(nodes)
        need_to13 = max(0, int(before.get('used_12_to_13', 0)))
        need_to12 = max(0, int(before.get('used_13_to_12', 0)))
        if need_to13 <= 0 and need_to12 <= 0:
            return H_new.to(H_work.device), [], {
                'applied': False,
                'before': before,
                'after': before,
                'requested_12_to_13': 0,
                'requested_13_to_12': 0,
                'applied_12_to_13': 0,
                'applied_13_to_12': 0,
            }

        moves: List[Dict[str, Any]] = []

        def _pick_nodes(src_types: List[int], count: int) -> List[_NodeV3]:
            picked: List[_NodeV3] = []
            seen: set[int] = set()
            for src in src_types:
                cands = [n for n in nodes if int(getattr(n, 'su_type', -1)) == int(src)]
                cands.sort(key=lambda n: int(getattr(n, 'global_id', 0)))
                for node in cands:
                    gid = int(getattr(node, 'global_id', -1))
                    if gid in seen:
                        continue
                    picked.append(node)
                    seen.add(gid)
                    if len(picked) >= int(count):
                        return picked
            return picked

        def _convert(node: _NodeV3, dst_su: int) -> bool:
            src_su = int(getattr(node, 'su_type', -1))
            dst_su_i = int(dst_su)
            if int(src_su) == int(dst_su_i):
                return False
            if not (0 <= int(src_su) < int(H_new.numel()) and 0 <= int(dst_su_i) < int(H_new.numel())):
                return False
            if int(H_new[src_su].item()) <= 0:
                return False
            if int(dst_su_i) == 12:
                dec13 = 1 if int(src_su) == 13 else 0
                if not self._can_increase_su12(H_new, inc12=1, dec13=int(dec13)):
                    return False
            H_new[src_su] -= 1
            node.su_type = int(dst_su_i)
            H_new[dst_su_i] += 1
            return True

        applied_to13 = 0
        for node in _pick_nodes([12], need_to13):
            if _convert(node, 13):
                applied_to13 += 1
                moves.append({
                    'stage': 'skeleton_align',
                    'op': 'ALIGN_12->13',
                    'from': 12,
                    'to': 13,
                    'global_id': int(getattr(node, 'global_id', -1)),
                })

        applied_to12 = 0
        src_pool = [13] if bool(protect_11) else [13, 11]
        for node in _pick_nodes(src_pool, need_to12):
            src_su = int(getattr(node, 'su_type', -1))
            if _convert(node, 12):
                applied_to12 += 1
                moves.append({
                    'stage': 'skeleton_align',
                    'op': f'ALIGN_{src_su}->12',
                    'from': int(src_su),
                    'to': 12,
                    'global_id': int(getattr(node, 'global_id', -1)),
                })

        self._refresh_node_counters(nodes)
        after = self._compute_aromatic_cluster_metrics(nodes)
        return H_new.to(H_work.device), moves, {
            'applied': bool(moves),
            'before': before,
            'after': after,
            'requested_12_to_13': int(need_to13),
            'requested_13_to_12': int(need_to12),
            'applied_12_to_13': int(applied_to13),
            'applied_13_to_12': int(applied_to12),
            'protect_11': bool(protect_11),
        }

    @staticmethod
    def _compute_flexible_window(cluster_count: int,
                                 rigid_pairs: int,
                                 flex_ratio: float,
                                 flex_lower_extra: int = 1) -> Tuple[int, int, int]:
        cluster_count_i = max(0, int(cluster_count))
        rigid_pairs_i = max(0, int(rigid_pairs))
        z_clusters = max(1, int(cluster_count_i - rigid_pairs_i))
        flex_lower = max(0, int(z_clusters + int(flex_lower_extra)))
        flex_upper = max(int(flex_lower), int(math.floor(float(cluster_count_i) * float(flex_ratio))))
        return int(z_clusters), int(flex_lower), int(flex_upper)

    @staticmethod
    def _format_chain_spec(chain: Any) -> str:
        comp = "-".join(str(int(x)) for x in list(getattr(chain, 'composition', []) or []))
        ctype = str(getattr(chain, 'chain_type', '?'))
        origin = str(getattr(chain, 'origin_type', '?'))
        src = list(getattr(chain, 'source_ids', []) or [])
        meta = getattr(chain, 'metadata', {}) or {}
        meta_brief = []
        src_su = list(meta.get('source_su_types', []) or [])
        src_hop1 = list(meta.get('source_hop1', []) or [])
        src_eff = list(meta.get('source_effective_kinds', []) or [])
        src_deg = list(meta.get('source_target_degrees', []) or [])
        src_modes = list(meta.get('source_anchor_modes', []) or [])

        def _raw_node_label(idx: int, su_type: Any, hop1_vals: Any) -> str:
            try:
                su_i = int(su_type)
            except Exception:
                return str(su_type)
            try:
                neighbors = [int(x) for x in list(hop1_vals or [])]
            except Exception:
                neighbors = []
            degree = None
            if int(idx) < len(src_deg):
                try:
                    degree = int(src_deg[int(idx)]) if src_deg[int(idx)] is not None else None
                except Exception:
                    degree = None
            mode = None
            if int(idx) < len(src_modes):
                mode_raw = src_modes[int(idx)]
                mode = str(mode_raw) if mode_raw is not None and str(mode_raw) != 'None' else None
            center = f"{su_i}d{degree}" if su_i in {19, 20, 21} and degree is not None else str(su_i)
            if mode:
                center = f"{center}:{mode}"
            if len(neighbors) == 2 and su_i in {0, 2, 3, 27, 29, 31}:
                return f"{neighbors[0]}-{su_i}-{neighbors[1]}"
            if len(neighbors) == 1 and su_i in {1, 4, 16, 18, 28, 32}:
                return f"{neighbors[0]}-{su_i}"
            if neighbors:
                return f"{center}({','.join(str(x) for x in neighbors)})"
            return center

        raw_topology = str(meta.get('raw_topology', '') or meta.get('display_topology', '') or '')
        if not raw_topology and src_su:
            raw_parts = []
            for i, su_i in enumerate(src_su):
                hop1_i = src_hop1[i] if i < len(src_hop1) else []
                raw_parts.append(_raw_node_label(i, su_i, hop1_i))
            raw_topology = " + ".join(raw_parts)
        if not raw_topology:
            raw_topology = "?"
        if meta:
            if 'branch_type' in meta:
                meta_brief.append(f"branch_type={meta['branch_type']}")
            if 'tail_source' in meta:
                meta_brief.append(f"tail={meta['tail_source']}")
            if 'tail_sources' in meta:
                meta_brief.append(f"tails={meta['tail_sources']}")
            if src_su:
                meta_brief.append(f"src_su={src_su}")
            if src_eff:
                meta_brief.append(f"src_eff={src_eff}")
            if src_deg:
                meta_brief.append(f"src_deg={src_deg}")
            if src_modes:
                meta_brief.append(f"src_mode={src_modes}")
            if src_hop1:
                meta_brief.append(f"src_hop1={src_hop1}")
        meta_txt = f" | {', '.join(meta_brief)}" if meta_brief else ""
        return f"{ctype}/{origin}: effective={comp} | raw={raw_topology} | src={src}{meta_txt}"

    def _extract_allocation_details(self, alloc_res: Any) -> Dict[str, Any]:
        def _rows(chains: List[Any]) -> List[str]:
            return [self._format_chain_spec(ch) for ch in list(chains or [])]

        bridge_rows = _rows(getattr(alloc_res, 'bridge_chains', []))
        side_rows = _rows(getattr(alloc_res, 'side_chains', []))
        branch_rows = _rows(getattr(alloc_res, 'branch_chains', []))
        return {
            'bridge_count': int(len(bridge_rows)),
            'side_count': int(len(side_rows)),
            'branch_count': int(len(branch_rows)),
            'bridge_rows': bridge_rows,
            'side_rows': side_rows,
            'branch_rows': branch_rows,
        }

    def _print_allocation_details(self, alloc_res: Any, header: str = "候选完整资源分配结果（未验收）") -> None:
        details = self._extract_allocation_details(alloc_res)
        print(f"\n  [Skeleton-Alloc] {header}")
        print(
            f"    Bridge chains: {details['bridge_count']} | "
            f"Side chains: {details['side_count']} | "
            f"Branch structures: {details['branch_count']}"
        )
        if details['bridge_rows']:
            print("    [Bridge]")
            for idx, row in enumerate(details['bridge_rows']):
                print(f"      [{idx}] {row}")
        if details['side_rows']:
            print("    [Side]")
            for idx, row in enumerate(details['side_rows']):
                print(f"      [{idx}] {row}")
        if details['branch_rows']:
            print("    [Branch]")
            for idx, row in enumerate(details['branch_rows']):
                print(f"      [{idx}] {row}")

    @staticmethod
    def _extract_resource_ledger(alloc_res: Any) -> Dict[str, Dict[str, int]]:
        keys = ('11', '22', '23', '24', '25')

        def _get_map(attr_name: str) -> Dict[str, int]:
            raw = getattr(alloc_res, attr_name, {}) or {}
            return {
                str(k): int(raw.get(k, 0))
                for k in keys
            }

        return {
            'native_total': _get_map('native_total_by_kind'),
            'proxy_total': _get_map('proxy_total_by_kind'),
            'native_consumed': _get_map('native_consumed_by_kind'),
            'proxy_consumed': _get_map('proxy_consumed_by_kind'),
            'native_remaining': _get_map('native_remaining_by_kind'),
            'proxy_remaining': _get_map('proxy_remaining_by_kind'),
        }

    def _evaluate_full_allocation_balance(self,
                                          nodes: List[_NodeV3],
                                          flex_ratio: float = 0.80,
                                          flex_lower_extra: int = 1,
                                          S_target: Optional[torch.Tensor] = None,
                                          E_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        allocator = FlexAllocator(nodes=nodes)
        with redirect_stdout(io.StringIO()):
            alloc_res = allocator.allocate()

        cluster_meta = self._compute_aromatic_cluster_metrics(nodes)
        cluster_count = int(cluster_meta.get('cluster_count', 0))
        rigid_pairs = max(0, sum(1 for n in nodes if int(getattr(n, 'su_type', -1)) == 10) // 2)
        z_clusters, flex_lower, flex_upper = self._compute_flexible_window(
            cluster_count=cluster_count,
            rigid_pairs=rigid_pairs,
            flex_ratio=float(flex_ratio),
            flex_lower_extra=int(flex_lower_extra),
        )
        def _is_flexible_bridge(ch) -> bool:
            comp = list(getattr(ch, 'composition', []) or [])
            return len(comp) >= 3 and int(comp[0]) == 11 and int(comp[-1]) == 11 and int(getattr(ch, 'n_23', 0)) >= 1

        def _is_aliphatic_side_to_22(ch) -> bool:
            comp = list(getattr(ch, 'composition', []) or [])
            if len(comp) < 3:
                return False
            if int(getattr(ch, 'n_23', 0)) < 1:
                return False
            return {int(comp[0]), int(comp[-1])} == {11, 22}

        flexible_bridge_chains = [ch for ch in getattr(alloc_res, 'bridge_chains', []) if _is_flexible_bridge(ch)]
        flexible_bridge_count = int(len(flexible_bridge_chains))
        extra_flexible_bridge_count = int(sum(1 for ch in flexible_bridge_chains if str(getattr(ch, 'origin_type', '')) == 'extra'))
        fixed_flexible_bridge_count = int(max(0, flexible_bridge_count - extra_flexible_bridge_count))
        side_to_22_count = sum(1 for ch in getattr(alloc_res, 'side_chains', []) if _is_aliphatic_side_to_22(ch))
        aliphatic_total = int(sum(1 for n in nodes if 19 <= int(getattr(n, 'su_type', -1)) <= 25))
        oxygenated_aliphatic_total = int(sum(1 for n in nodes if int(getattr(n, 'su_type', -1)) in {19, 20, 21}))
        ordinary_aliphatic_total = int(sum(1 for n in nodes if int(getattr(n, 'su_type', -1)) in {22, 23, 24, 25}))
        effective_cluster_count = max(1, int(cluster_count))
        unallocated_bridge = int(getattr(alloc_res, 'unallocated_bridge', 0))
        unallocated_branch = int(getattr(alloc_res, 'unallocated_branch', 0))
        required_extra_11 = int(getattr(alloc_res, 'required_extra_11', 0))
        required_extra_22 = int(getattr(alloc_res, 'required_extra_22', 0))
        required_extra_23 = int(getattr(alloc_res, 'required_extra_23', 0))
        resource_ledger = self._extract_resource_ledger(alloc_res)

        bounds = self._estimate_aliphatic_region_bounds(S_target, E_target)
        aliphatic_min = int(bounds.get('total_min', 0))
        aliphatic_max = int(bounds.get('total_max', 10**9))
        ordinary_aliphatic_min = int(bounds.get('ordinary_min', 0))
        ordinary_aliphatic_max = int(bounds.get('ordinary_max', 10**9))
        oxygenated_aliphatic_min = int(bounds.get('oxygenated_min', 0))
        oxygenated_aliphatic_max = int(bounds.get('oxygenated_max', 10**9))

        rigid_ok = int(rigid_pairs) < int(effective_cluster_count)
        flex_hi_ok = int(flexible_bridge_count) <= int(flex_upper)
        flex_lo_ok = int(flexible_bridge_count) >= int(flex_lower)
        aliphatic_ok = int(aliphatic_total) >= int(aliphatic_min)
        aliphatic_hi_ok = int(aliphatic_total) <= int(aliphatic_max)
        ordinary_aliphatic_ok = int(ordinary_aliphatic_total) >= int(ordinary_aliphatic_min)
        ordinary_aliphatic_hi_ok = int(ordinary_aliphatic_total) <= int(ordinary_aliphatic_max)
        oxygenated_aliphatic_ok = int(oxygenated_aliphatic_total) >= int(oxygenated_aliphatic_min)
        oxygenated_aliphatic_hi_ok = int(oxygenated_aliphatic_total) <= int(oxygenated_aliphatic_max)
        branch_alloc_ok = int(unallocated_branch) == 0
        bridge_alloc_ok = int(unallocated_bridge) == 0
        extra_resource_ok = (
            int(required_extra_11) == 0 and
            int(required_extra_22) == 0 and
            int(required_extra_23) == 0
        )

        reasons = []
        warnings = []
        if not bridge_alloc_ok:
            reasons.append('bridge_unallocated')
        if not branch_alloc_ok:
            reasons.append('branch_unallocated')
        if not extra_resource_ok:
            reasons.append('resource_shortage')
        if not rigid_ok:
            reasons.append('rigid_excess')
        if not flex_lo_ok:
            reasons.append('flex_shortage')
        if not aliphatic_ok:
            reasons.append('aliphatic_shortage')
        if not aliphatic_hi_ok:
            reasons.append('aliphatic_excess')
        if not ordinary_aliphatic_ok:
            reasons.append('ordinary_aliphatic_shortage')
        if not ordinary_aliphatic_hi_ok:
            reasons.append('ordinary_aliphatic_excess')
        if not oxygenated_aliphatic_ok:
            reasons.append('oxygenated_aliphatic_shortage')
        if not oxygenated_aliphatic_hi_ok:
            reasons.append('oxygenated_aliphatic_excess')
        if not flex_hi_ok:
            reasons.append('flex_excess')
            warnings.append('flex_excess')
        reason = 'ok' if not reasons else '+'.join(reasons)

        return {
            'ok': bool(
                bridge_alloc_ok and
                branch_alloc_ok and
                extra_resource_ok and
                rigid_ok and
                flex_hi_ok and
                flex_lo_ok and
                aliphatic_ok and
                aliphatic_hi_ok and
                ordinary_aliphatic_ok and
                ordinary_aliphatic_hi_ok and
                oxygenated_aliphatic_ok and
                oxygenated_aliphatic_hi_ok
            ),
            'reason': str(reason),
            'warnings': [str(x) for x in warnings],
            'cluster_count': int(cluster_count),
            'effective_cluster_count': int(effective_cluster_count),
            'rigid_pairs': int(rigid_pairs),
            'rigid_cluster_count': int(z_clusters),
            'rigid_min_flex': int(flex_lower),
            'flexible_bridge_min': int(flex_lower),
            'flex_ratio': float(flex_ratio),
            'flex_lower_extra': int(flex_lower_extra),
            'flexible_bridge_count': int(flexible_bridge_count),
            'flex_hi_ok': bool(flex_hi_ok),
            'flex_lo_ok': bool(flex_lo_ok),
            'fixed_flexible_bridge_count': int(fixed_flexible_bridge_count),
            'extra_flexible_bridge_count': int(extra_flexible_bridge_count),
            'flexible_bridge_limit': int(flex_upper),
            'side_to_22_count': int(side_to_22_count),
            'aliphatic_total': int(aliphatic_total),
            'aliphatic_min_total': int(aliphatic_min),
            'aliphatic_max_total': int(aliphatic_max),
            'ordinary_aliphatic_total': int(ordinary_aliphatic_total),
            'ordinary_aliphatic_min_total': int(ordinary_aliphatic_min),
            'ordinary_aliphatic_max_total': int(ordinary_aliphatic_max),
            'oxygenated_aliphatic_total': int(oxygenated_aliphatic_total),
            'oxygenated_aliphatic_min_total': int(oxygenated_aliphatic_min),
            'oxygenated_aliphatic_max_total': int(oxygenated_aliphatic_max),
            'aliphatic_region_bounds': dict(bounds),
            'cluster_meta': cluster_meta,
            'allocation_result': alloc_res,
            'allocation_details': self._extract_allocation_details(alloc_res),
            'resource_ledger': resource_ledger,
            'unallocated_bridge': int(unallocated_bridge),
            'unallocated_branch': int(unallocated_branch),
            'required_extra_11': int(required_extra_11),
            'required_extra_22': int(required_extra_22),
            'required_extra_23': int(required_extra_23),
            'remaining': {
                '11': int(getattr(alloc_res, 'remaining_11', 0)),
                '22': int(getattr(alloc_res, 'remaining_22', 0)),
                '23': int(getattr(alloc_res, 'remaining_23', 0)),
                '24': int(getattr(alloc_res, 'remaining_24', 0)),
                '25': int(getattr(alloc_res, 'remaining_25', 0)),
            },
            'native_remaining': dict(resource_ledger.get('native_remaining', {}) or {}),
            'proxy_remaining': dict(resource_ledger.get('proxy_remaining', {}) or {}),
            'native_total': dict(resource_ledger.get('native_total', {}) or {}),
            'proxy_total': dict(resource_ledger.get('proxy_total', {}) or {}),
            'native_consumed': dict(resource_ledger.get('native_consumed', {}) or {}),
            'proxy_consumed': dict(resource_ledger.get('proxy_consumed', {}) or {}),
        }

    def _adjust_skeleton_branch_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        ppm: Optional[np.ndarray] = None,
        diff: Optional[np.ndarray] = None,
        max_steps: int = 50,
        nodes: Optional[List[_NodeV3]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        return adjust_block_c_branch_phase_impl(
            self,
            H,
            E_target,
            S_target=S_target,
            ppm=ppm,
            diff=diff,
            max_steps=max_steps,
            nodes=nodes,
            **kwargs,
        )

    def _adjust_skeleton_extra_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        ppm: Optional[np.ndarray] = None,
        diff: Optional[np.ndarray] = None,
        guided_max_steps: int = 150,
        relaxed_flexible_ratio: float = 0.82,
        nodes: Optional[List[_NodeV3]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        return adjust_block_c_extra_phase_impl(
            self,
            H,
            E_target,
            S_target=S_target,
            ppm=ppm,
            diff=diff,
            guided_max_steps=guided_max_steps,
            relaxed_flexible_ratio=relaxed_flexible_ratio,
            nodes=nodes,
            **kwargs,
        )

    def _adjust_skeleton_by_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        ppm: Optional[np.ndarray] = None,
        diff: Optional[np.ndarray] = None,
        max_steps: int = 50,
        nodes: Optional[List[_NodeV3]] = None,
        phase: str = 'full',
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        phase_name = str(phase or 'full').strip().lower()
        H_input = torch.clamp(H, min=0).long().clone().cpu()
        H_work = H_input.detach().clone()
        branch_moves: List[Dict[str, Any]] = []
        branch_meta: Dict[str, Any] = {
            'n_moves': 0,
            'ok': True if phase_name == 'extra' else False,
            'records': [],
            'phase': 'branch',
            'skipped': bool(phase_name == 'extra'),
            'reason': 'phase_extra_only' if phase_name == 'extra' else 'not_run',
            'final_diag': {},
            'final_scenario': 'phase_extra_only' if phase_name == 'extra' else 'not_run',
        }

        if phase_name in ('full', 'branch'):
            H_work, branch_moves, branch_meta = self._adjust_skeleton_branch_allocation(
                H=H,
                E_target=E_target,
                S_target=S_target,
                ppm=ppm,
                diff=diff,
                max_steps=max_steps,
                nodes=nodes,
                **kwargs,
            )

        all_moves = list(branch_moves)
        phase_moves: Dict[str, List[Dict[str, Any]]] = {
            'branch': list(branch_moves),
            'extra': [],
            'align': [],
            'post': [],
        }
        H_after_branch = H_work.detach().clone().cpu()
        final_h_ratio = float(branch_meta.get('final_h_ratio', 0.0))

        if phase_name == 'branch':
            return H_work, all_moves, {
                'n_moves': len(all_moves),
                'ok': bool(branch_meta.get('ok', False)),
                'branch_ok': bool(branch_meta.get('ok', False)),
                'extra_ok': True,
                'final_h_ratio': float(final_h_ratio),
                'records': all_moves,
                'branch_meta': branch_meta,
                'extra_meta': {
                    'n_moves': 0,
                    'ok': True,
                    'records': [],
                    'phase': 'extra',
                    'skipped': True,
                    'reason': 'phase_branch_only',
                },
                'align_meta': {'applied': False, 'reason': 'phase_branch_only'},
                'final_diag': dict((branch_meta or {}).get('final_diag', {}) or {}),
                'final_allocation': {},
                'post_meta': {'post_changed': False, 'recheck_completed': False},
                'phase_hists': {
                    'input': H_input,
                    'after_branch': H_after_branch,
                },
                'phase_moves': phase_moves,
                'recheck_required': False,
                'post_changed': False,
                'phase': 'branch',
                'final_scenario': str(branch_meta.get('final_scenario', 'branch')),
            }

        extra_meta: Dict[str, Any] = {
            'n_moves': 0,
            'ok': False,
            'records': [],
            'phase': 'extra',
            'skipped': True,
            'reason': 'branch_not_ok',
        }

        can_run_extra = bool(phase_name == 'extra' or bool(branch_meta.get('ok', False)))
        if can_run_extra:
            extra_kwargs = dict(kwargs)
            guided_max_steps = int(extra_kwargs.pop('guided_max_steps', max(12, int(max_steps) * 3)))
            relaxed_flexible_ratio = float(extra_kwargs.pop('relaxed_flexible_ratio', 0.82))
            H_work, extra_moves, extra_meta = self._adjust_skeleton_extra_allocation(
                H=H_work,
                E_target=E_target,
                S_target=S_target,
                ppm=ppm,
                diff=diff,
                guided_max_steps=guided_max_steps,
                relaxed_flexible_ratio=relaxed_flexible_ratio,
                nodes=nodes,
                **extra_kwargs,
            )
            all_moves.extend(extra_moves)
            extra_phase_moves = dict(extra_meta.get('phase_moves', {}) or {})
            if extra_phase_moves:
                phase_moves['extra'].extend(list(extra_phase_moves.get('extra', []) or []))
                phase_moves['align'].extend(list(extra_phase_moves.get('align', []) or []))
                phase_moves['post'].extend(list(extra_phase_moves.get('post', []) or []))
            else:
                phase_moves['extra'].extend(list(extra_moves))
        else:
            print("  [Skeleton-Alloc] 分支资源分配尚未通过，跳过 extra 阶段")
        extra_phase_hists = dict(extra_meta.get('phase_hists', {}) or {})
        H_after_extra = extra_phase_hists.get('after_extra', H_work.detach().clone().cpu())
        H_after_align = extra_phase_hists.get('after_align', H_after_extra)
        H_after_post = extra_phase_hists.get('after_post', H_after_align)

        if 'final_h_ratio' in extra_meta:
            final_h_ratio = float(extra_meta['final_h_ratio'])

        branch_ok_for_phase = True if phase_name == 'extra' else bool(branch_meta.get('ok', False))
        overall_ok = bool(branch_ok_for_phase) and bool(extra_meta.get('ok', False))
        final_diag = extra_meta.get('final_diag') if bool(can_run_extra) else branch_meta.get('final_diag')
        align_meta: Dict[str, Any] = dict(extra_meta.get('align_meta', {}) or {})
        post_meta: Dict[str, Any] = dict(extra_meta.get('post_meta', {}) or {})
        final_alloc_diag = dict(extra_meta.get('final_allocation', {}) or {})
        recheck_required = bool(extra_meta.get('recheck_required', False))
        post_changed = bool(extra_meta.get('post_changed', False))
        recheck_completed = bool(post_meta.get('recheck_completed', False))
        final_alloc_ok = bool(final_alloc_diag.get('ok', not bool(post_changed)))
        overall_ok = bool(overall_ok) and (not bool(recheck_required)) and bool(final_alloc_ok)
        return H_work, all_moves, {
            'n_moves': len(all_moves),
            'ok': overall_ok,
            'branch_ok': bool(branch_ok_for_phase),
            'extra_ok': bool(extra_meta.get('ok', False)),
            'final_h_ratio': final_h_ratio,
            'records': all_moves,
            'branch_meta': branch_meta,
            'extra_meta': extra_meta,
            'align_meta': align_meta,
            'final_diag': final_diag,
            'final_allocation': final_alloc_diag,
            'post_meta': post_meta,
            'phase_hists': {
                'input': H_input,
                'after_branch': H_after_branch,
                'after_extra': H_after_extra,
                'after_align': H_after_align,
                'after_post': H_after_post,
            },
            'phase_moves': phase_moves,
            'recheck_required': bool(recheck_required),
            'post_changed': bool(post_changed),
            'phase': str(phase_name),
            'final_scenario': (
                'ok'
                if overall_ok else (
                    'branch_not_ok'
                    if not bool(branch_ok_for_phase)
                    else str(final_alloc_diag.get('reason', extra_meta.get('reason', 'extra_not_ok')))
                )
            ),
        }

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
                - 'block_c'        -> block_c_tail / tail_diff
                - 'block_c_branch' -> branch_topology
                - 'block_c_extra'  -> extra_global
            **kwargs:
                各阶段附加参数。对于 block_c:
                - block_c_tail: max_moves, peak_rel_threshold, carbonyl_couple, h_tolerance
                - block_c_branch: max_steps
                - block_c_extra: guided_max_steps, relaxed_flexible_ratio
                阶段结果是否采纳由调用侧决定；Layer4 这里只负责生成候选 H 与诊断信息。
        
        Returns:
            H_adjusted: 调整后的SU直方图
            moves: 调整记录
            meta: 调整元数据
        """
        print(f"\n{'='*80}")
        print(f"Layer4 [{stage.upper()}阶段] 调整")
        print(f"{'='*80}")
        
        H_input = torch.clamp(H, min=0).long().clone().cpu()
        H_work = H.clone()
        moves = []
        meta = {}
        self.E_target = E_target.detach().cpu() if hasattr(E_target, 'detach') else E_target
        self._current_S_target = S_target.detach().cpu() if hasattr(S_target, 'detach') else S_target
        if self.layer0_estimator is not None:
            try:
                self.layer0_estimator._current_S_target = self._current_S_target
            except Exception:
                pass
        self._h_rotation_aliphatic_cap = self._estimate_aliphatic_upper_bound(S_target, E_target)
        region_bounds = self._estimate_aliphatic_region_bounds(S_target, E_target)
        self._h_rotation_ordinary_aliphatic_cap = int(region_bounds.get('ordinary_max', 10**9))
        
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
            )
        elif stage == 'block_c':
            H_work, moves, meta = self.adjust_block_c_aliphatic_tail(
                H_work, ppm, diff,
                E_target=E_target,
                max_moves=kwargs.get('max_moves', 6),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                min_keep_22=kwargs.get('min_keep_22', 1),
                min_keep_23=kwargs.get('min_keep_23', 0),
                min_keep_24=kwargs.get('min_keep_24', 0),
                carbonyl_couple=kwargs.get('carbonyl_couple', True),
                h_tolerance=kwargs.get('h_tolerance', float(getattr(self, '_h_tolerance', 0.08))),
            )
        elif stage in ('block_c_branch', 'block_c_extra'):
            phase_name = 'branch' if stage == 'block_c_branch' else 'extra'
            skeleton_kwargs = dict(kwargs)
            skeleton_max_steps = skeleton_kwargs.pop('max_steps', 40)
            skeleton_nodes = skeleton_kwargs.pop('nodes', None)
            H_work, moves, meta = self._adjust_skeleton_by_allocation(
                H_work,
                E_target=E_target,
                S_target=S_target,
                ppm=ppm,
                diff=diff,
                max_steps=skeleton_max_steps,
                nodes=skeleton_nodes,
                phase=phase_name,
                **skeleton_kwargs,
            )
            meta['h_rotation_meta'] = {
                'applied': False,
                'reason': 'handled_inside_block_c_skeleton',
            }
            summary_label = 'BLOCK_C-BRANCH' if phase_name == 'branch' else 'BLOCK_C-EXTRA'
            self._print_hist_change_summary(summary_label, H_input, torch.clamp(H_work, min=0).long().cpu())
            self._print_move_summary(summary_label, moves)
            self._print_stage_distribution_summary(f"{summary_label} final_literal_vs_effective", torch.clamp(H_work, min=0).long().cpu())
            phase_hists = meta.get('phase_hists', {}) or {}
            phase_moves = meta.get('phase_moves', {}) or {}
            if phase_hists:
                for label, before_key, after_key in (
                    ('BLOCK_C-BRANCH', 'input', 'after_branch'),
                    ('BLOCK_C-EXTRA', 'after_branch', 'after_extra'),
                    ('BLOCK_C-ALIGN', 'after_extra', 'after_align'),
                    ('BLOCK_C-POST', 'after_align', 'after_post'),
                ):
                    before = phase_hists.get(before_key)
                    after = phase_hists.get(after_key)
                    if before is None or after is None:
                        continue
                    self._print_hist_change_summary(str(label), before, after)
                    self._print_stage_distribution_summary(f"{str(label)} after_literal_vs_effective", after)
            for label, key in (
                ('BLOCK_C-BRANCH', 'branch'),
                ('BLOCK_C-EXTRA', 'extra'),
                ('BLOCK_C-ALIGN', 'align'),
                ('BLOCK_C-POST', 'post'),
            ):
                self._print_move_summary(str(label), list(phase_moves.get(key, []) or []))
            print(f"\n{summary_label}候选调整完成: 共{len(moves)}次变更")
            return H_work, moves, meta

        else:
            print(f"  [警告] 未知的调整阶段: {stage}")
            return H_work, moves, meta

        try:
            H_work, hrot_moves, hrot_meta = self._apply_h_rotation_to_counts(H_work, E_target, max_ops=None)
            if hrot_moves:
                moves.extend(hrot_moves)
            meta['h_rotation_meta'] = hrot_meta
        except Exception:
            pass
        
        self._print_hist_change_summary(stage.upper(), H_input, torch.clamp(H_work, min=0).long().cpu())
        self._print_move_summary(stage.upper(), moves)
        self._print_stage_distribution_summary(f"{stage.upper()} final_literal_vs_effective", torch.clamp(H_work, min=0).long().cpu())
        print(f"\n{stage.upper()}阶段调整完成: 共{len(moves)}次变更")
        return H_work, moves, meta
    
