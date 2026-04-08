import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
from collections import Counter
import math
import io
from contextlib import redirect_stdout

from .inverse_common import SU_ALIPHATIC, SU_AROMATIC, E_SU, _NodeV3
from RL_MTCS.RL_allocator import FlexAllocator
from RL_MTCS.RL_init import ClusterGenerator
from .inverse_layer0 import Layer0Estimator


class Layer4Adjuster:
    """
    Layer4: 基于差谱的多阶段SU直方图调整器
    
    集成所有SU调整逻辑：
    1. 羰基类型调整 (1/2/3 互转)
    2. SU9数量调整 (9 ↔ 10/11/12/13)
    3. O连接调整 (5/19)
    4. N连接调整 (6/20)
    5. S连接调整 (7/19)
    6. X连接调整 (8/21)
    
    Layer4 只负责基于差谱/资源分配做 SU 计数调整；
    Layer0 修正仅用于初始化阶段，不在 Layer4 中重复调用。
    """
    
    def __init__(self, 
                 device: torch.device = None,
                 layer0_estimator: Optional[Layer0Estimator] = None,
                 su_hop1_ranges_path: Optional[str] = None,
                 su_common_ranges_path: Optional[str] = None):
        self.device = device or torch.device('cpu')
        self.E_SU = E_SU.to(self.device)
        
        # 加载 hop1 NMR范围数据
        if su_hop1_ranges_path is None:
            default_path = Path(__file__).resolve().parents[1] / 'z_library' / 'su_hop1_nmr_range_filtered.csv'
            self.su_hop1_ranges_path = str(default_path) if default_path.exists() else None
        else:
            self.su_hop1_ranges_path = su_hop1_ranges_path
        if su_common_ranges_path is None:
            default_common = Path(__file__).resolve().parents[1] / 'z_library' / 'su_nmr_common_range_filtered.csv'
            self.su_common_ranges_path = str(default_common) if default_common.exists() else None
        else:
            self.su_common_ranges_path = su_common_ranges_path
        self._su_hop1_mu_median_cache = None
        self._su_common_stats_cache = None
        # Persist H-adjust rotation state across repeated skeleton adjustment calls.
        self._h_rotation_state = 0
        self._rigid10_rotation_state = 0

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

    def _enforce_su22_ratio_and_h(self,
                                 H: torch.Tensor,
                                 E_target: Optional[torch.Tensor],
                                 enable: bool = True,
                                 ratio: float = 0.1,
                                 h_tol: float = 0.03) -> Tuple[torch.Tensor, List[Dict], Dict]:
        H_work = torch.clamp(H, min=0).long().clone()
        moves: List[Dict] = []

        if not bool(enable):
            return H_work, moves, {}

        try:
            n22 = int(H_work[22].item())
            n23 = int(H_work[23].item())
        except Exception:
            return H_work, moves, {}

        try:
            ratio_f = float(ratio)
        except Exception:
            ratio_f = 0.1
        ratio_f = max(0.0, float(ratio_f))

        req22 = int(math.ceil(float(ratio_f) * float(n23)))
        req22 = max(1, int(req22))

        while int(n22) < int(req22):
            if int(H_work[23].item()) <= 0:
                break
            H_work[23] -= 1
            H_work[22] += 1
            moves.append({'op': 'enforce_22_ratio', 'from': 23, 'to': 22})
            try:
                n22 = int(H_work[22].item())
                n23 = int(H_work[23].item())
                req22 = int(math.ceil(float(ratio_f) * float(n23)))
                req22 = max(1, int(req22))
            except Exception:
                break

        if int(H_work[22].item()) <= 0:
            if int(H_work[23].item()) > 0:
                H_work[23] -= 1
                H_work[22] += 1
                moves.append({'op': 'enforce_22_nonzero', 'from': 23, 'to': 22})

        meta: Dict[str, Any] = {
            'req22': int(req22),
            'final_22': int(H_work[22].item()),
            'final_23': int(H_work[23].item()),
        }

        if E_target is not None:
            try:
                E_curr = torch.matmul(H_work.float(), self.E_SU)
                curr_H = float(E_curr[1].item())
                tgt_H = float(E_target.to(E_curr.device)[1].item())
                meta['curr_H'] = float(curr_H)
                meta['tgt_H'] = float(tgt_H)
            except Exception:
                curr_H = None
                tgt_H = None

            if curr_H is not None and tgt_H is not None and float(tgt_H) > 1e-9:
                try:
                    h_tol_f = float(h_tol)
                except Exception:
                    h_tol_f = 0.03
                h_tol_f = max(0.0, float(h_tol_f))
                h_cap = float(tgt_H) * (1.0 + float(h_tol_f))
                while float(curr_H) > float(h_cap) + 1e-9:
                    if int(H_work[13].item()) <= 0:
                        break
                    H_work[13] -= 1
                    H_work[10] += 1
                    moves.append({'op': 'reduce_H_13_to_10', 'from': 13, 'to': 10})
                    curr_H = float(curr_H) - 1.0
                meta['final_H_after_cap'] = float(curr_H)
                meta['h_cap'] = float(h_cap)

        H_work = torch.clamp(H_work, min=0).long()
        return H_work, moves, meta

    def _apply_final_structure_constraints(self, H: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, int]], Dict[str, Any]]:
        """
        Final discrete constraints applied after Layer4 finishes.

        Current rules:
        1. SU10 count must be even; if odd, remove one SU10.
        2. SU14+SU15+SU16 total must be even; if odd, remove one SU15 first,
           otherwise SU16, otherwise SU14.
        """
        H_work = torch.clamp(H, min=0).long().clone()
        moves: List[Dict[str, int]] = []

        try:
            unsat_total = int(H_work[14].item()) + int(H_work[15].item()) + int(H_work[16].item())
        except Exception:
            unsat_total = 0

        if int(unsat_total) % 2 != 0:
            for su_idx in (15, 16, 14):
                if int(H_work[su_idx].item()) > 0:
                    H_work[su_idx] -= 1
                    moves.append({'op': 'final_even_14_15_16', 'from': int(su_idx), 'to': -1})
                    break

        try:
            if int(H_work[10].item()) % 2 != 0 and int(H_work[10].item()) > 0:
                H_work[10] -= 1
                moves.append({'op': 'final_even_10', 'from': 10, 'to': -1})
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
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0}
        seg = diff_arr[mask]
        if int(seg.size) <= 0:
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0}
        pos = float(np.sum(seg[seg > 0])) if np.any(seg > 0) else 0.0
        neg = float(-np.sum(seg[seg < 0])) if np.any(seg < 0) else 0.0
        return {
            'pos': float(pos),
            'neg': float(neg),
            'net': float(pos - neg),
            'abs': float(np.sum(np.abs(seg))),
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
            from .coarse_graph import SU_DEFS
            if 0 <= int(idx) < len(SU_DEFS):
                return str(SU_DEFS[int(idx)][0])
        except Exception:
            pass
        return f"SU{int(idx)}"

    def _print_move_summary(self,
                            stage: str,
                            moves: List[Dict[str, Any]],
                            limit: int = 10) -> None:
        if not moves:
            return
        print(f"  [{stage}] 调整动作:")
        for mv in moves[:limit]:
            if isinstance(mv, dict):
                parts: List[str] = []
                if 'stage' in mv:
                    parts.append(f"stage={mv['stage']}")
                if 'op' in mv:
                    parts.append(str(mv['op']))
                if 'from' in mv and 'to' in mv:
                    parts.append(f"{mv['from']}->{mv['to']}")
                if 'delta' in mv:
                    delta = mv.get('delta', {}) or {}
                    delta_txt = ", ".join(f"{int(k)}:{int(v):+d}" for k, v in delta.items())
                    if delta_txt:
                        parts.append(f"delta[{delta_txt}]")
                print(f"    - {' | '.join(parts) if parts else str(mv)}")
            else:
                print(f"    - {mv}")
        if len(moves) > int(limit):
            print(f"    ... 其余 {len(moves) - int(limit)} 条略")

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
        print("\n[Block A] 羰基-锚点联合调整")

        H_work = torch.clamp(H, min=0).long().clone()
        all_moves: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        # 先做少量羰基中心类型修正，再做 9/11 与 22/23/24/25 的联合迁移
        if int(max_moves) > 0 and int(carbonyl_max_moves) > 0:
            H_work, center_moves, center_meta = self.adjust_carbonyl_by_difference(
                H_work,
                ppm,
                diff,
                score_rel_threshold=float(score_rel_threshold),
                max_moves=min(int(max_moves), int(carbonyl_max_moves)),
                min_keep=int(min_keep),
            )
            for mv in center_moves:
                tagged = dict(mv)
                tagged['block'] = 'A_center'
                all_moves.append(tagged)
            meta['center_meta'] = center_meta

        if ppm is None or diff is None:
            meta['joint_direction'] = None
            return H_work, all_moves, meta

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            meta['joint_direction'] = None
            return H_work, all_moves, meta

        carb_windows = self._get_carb_joint_windows()
        low_lo, low_hi = carb_windows['low']
        mid_lo, mid_hi = carb_windows['mid']
        high_lo, high_hi = carb_windows['high']
        overall_lo, overall_hi = carb_windows['overall']
        low = self._window_stats(ppm_arr, diff_arr, low_lo, low_hi)
        mid = self._window_stats(ppm_arr, diff_arr, mid_lo, mid_hi)
        high = self._window_stats(ppm_arr, diff_arr, high_lo, high_hi)
        carbonyl_mask = (ppm_arr >= overall_lo) & (ppm_arr <= overall_hi)
        carbonyl_abs = float(np.sum(np.abs(diff_arr[carbonyl_mask]))) if np.any(carbonyl_mask) else float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-8, carbonyl_abs)

        direction = None
        if float(low['neg']) > float(thr) and float(mid['pos']) > float(thr):
            direction = 'to_aliphatic'
        elif float(low['pos']) > float(thr) and float(mid['neg']) > float(thr):
            direction = 'to_aryl9'

        meta['joint_direction'] = direction
        meta['joint_windows'] = {
            f'{low_lo:.1f}_{low_hi:.1f}': low,
            f'{mid_lo:.1f}_{mid_hi:.1f}': mid,
            f'{high_lo:.1f}_{high_hi:.1f}': high,
        }
        meta['joint_threshold'] = float(thr)

        remain_moves = max(0, int(max_moves) - len(center_moves) if 'center_moves' in locals() else int(max_moves))
        if direction is None or int(remain_moves) <= 0:
            return H_work, all_moves, meta

        tail_rank = self._rank_tail_targets(ppm_arr, diff_arr)
        joint_candidates: List[Tuple[str, Dict[int, int]]] = []
        if direction == 'to_aliphatic':
            for su in tail_rank:
                if su == 23:
                    joint_candidates.append(('A_9to11__22to23', {9: -1, 11: +1, 22: -1, 23: +1}))
                    joint_candidates.append(('A_9to11__13to23', {9: -1, 11: +1, 13: -1, 23: +1}))
                elif su == 24:
                    joint_candidates.append(('A_9to11__23to24', {9: -1, 11: +1, 23: -1, 24: +1}))
            joint_candidates.extend([
                ('A_9to11', {9: -1, 11: +1}),
                ('A_2to1__9to11__22to23', {2: -1, 1: +1, 9: -1, 11: +1, 22: -1, 23: +1}),
                ('A_3to2__9to11__23to24', {3: -1, 2: +1, 9: -1, 11: +1, 23: -1, 24: +1}),
            ])
        else:
            joint_candidates.extend([
                ('A_11to9__23to22', {11: -1, 9: +1, 23: -1, 22: +1}),
                ('A_11to9__24to23', {11: -1, 9: +1, 24: -1, 23: +1}),
                ('A_23to13__11to9', {23: -1, 13: +1, 11: -1, 9: +1}),
                ('A_1to2__11to9__23to22', {1: -1, 2: +1, 11: -1, 9: +1, 23: -1, 22: +1}),
                ('A_2to3__11to9__24to23', {2: -1, 3: +1, 11: -1, 9: +1, 24: -1, 23: +1}),
            ])

        keep = {1: int(min_keep), 2: int(min_keep), 3: int(min_keep), 22: 1}
        for _ in range(int(remain_moves)):
            applied = False
            for name, delta in joint_candidates:
                H_try = self._apply_count_delta(H_work, delta, min_keep=keep)
                if H_try is None:
                    continue
                H_work = H_try
                all_moves.append({'block': 'A_joint', 'op': name, 'delta': dict(delta)})
                applied = True
                break
            if not applied:
                break

        return H_work, all_moves, meta

    def adjust_block_b_hetero_anchor(self,
                                     H: torch.Tensor,
                                     ppm: Optional[np.ndarray],
                                     diff: Optional[np.ndarray],
                                     max_moves_each: int = 3,
                                     peak_rel_threshold: float = 0.01) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[Block B] 异原子锚点联合调整")
        H_work = torch.clamp(H, min=0).long().clone()
        all_moves: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        subcalls = [
            ('ether', self.adjust_ether_519_by_difference, {
                'max_moves': int(max_moves_each),
                'peak_rel_threshold': float(peak_rel_threshold),
                'min_keep': 1,
                'reserved_19': max(0, int(2 * H_work[31].item()) - int(H_work[7].item())),
            }),
            ('amine', self.adjust_amine_620_by_difference, {'max_moves': int(max_moves_each), 'peak_rel_threshold': float(peak_rel_threshold), 'min_keep': 0}),
            ('thioether', self.adjust_thioether_719_by_difference, {'max_moves': int(max_moves_each), 'peak_rel_threshold': float(peak_rel_threshold), 'min_keep': 0}),
            ('halogen', self.adjust_halogen_821_by_difference, {'max_moves': int(max_moves_each), 'peak_rel_threshold': float(peak_rel_threshold), 'min_keep': 0}),
        ]
        for name, fn, kwargs in subcalls:
            H_work, moves, submeta = fn(H_work, ppm, diff, **kwargs)
            meta[name] = submeta
            for mv in moves:
                tagged = dict(mv)
                tagged['block'] = 'B'
                tagged['substage'] = name
                all_moves.append(tagged)
        return H_work, all_moves, meta

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
                                      min_keep_25: int = 0,
                                      carbonyl_couple: bool = True,
                                      h_tolerance: float = 0.04) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[Block C] 脂肪尾部 22/23/24/25 联合调整")

        if ppm is None or diff is None:
            return H, [], {'reason': 'missing_diff'}

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            return H, [], {'reason': 'empty_diff'}

        lo22, hi22, _ = self._get_su_common_window(22, fallback_mu=19.81, min_half_width=6.0)
        lo23, hi23, _ = self._get_su_common_window(23, fallback_mu=29.48, min_half_width=6.0)
        lo24, hi24, _ = self._get_su_common_window(24, fallback_mu=39.97, min_half_width=6.0)
        lo25, hi25, _ = self._get_su_common_window(25, fallback_mu=39.63, min_half_width=6.0)
        s22 = self._window_stats(ppm_arr, diff_arr, lo22, hi22)
        s23 = self._window_stats(ppm_arr, diff_arr, lo23, hi23)
        s24 = self._window_stats(ppm_arr, diff_arr, lo24, hi24)
        s25 = self._window_stats(ppm_arr, diff_arr, lo25, hi25)
        s23_wide = self._window_stats(ppm_arr, diff_arr, 15.0, 45.0)
        s12_13 = self._window_stats(ppm_arr, diff_arr, 115.0, 135.0)
        carb_windows = self._get_carb_joint_windows()
        low_lo, low_hi = carb_windows['low']
        mid_lo, mid_hi = carb_windows['mid']
        low = self._window_stats(ppm_arr, diff_arr, low_lo, low_hi)
        mid = self._window_stats(ppm_arr, diff_arr, mid_lo, mid_hi)

        tail_mask = (ppm_arr >= 8.0) & (ppm_arr <= 65.0)
        tail_abs = float(np.sum(np.abs(diff_arr[tail_mask]))) if np.any(tail_mask) else float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-8, tail_abs)

        move_order: List[Tuple[str, Dict[int, int]]] = []
        # 新策略：以23为中峰，优先把23分流到22/24两侧，而不是做22→23→24→25链式迁移。
        if float(s23['neg']) > float(thr) and float(s22['pos']) > float(thr) and float(s24['pos']) > float(thr):
            move_order.append(('C_2x23_to_22_24', {23: -2, 22: +1, 24: +1}))
        if float(s23['neg']) > float(thr) and float(s22['pos']) > float(thr):
            move_order.append(('C_23to22', {23: -1, 22: +1}))
        if float(s23['neg']) > float(thr) and float(s24['pos']) > float(thr):
            move_order.append(('C_23to24', {23: -1, 24: +1}))

        # 两侧峰之间的直接平衡：22 ↔ 24
        if float(s22['neg']) > float(thr) and float(s24['pos']) > float(thr):
            move_order.append(('C_22to24', {22: -1, 24: +1}))
        if float(s24['neg']) > float(thr) and float(s22['pos']) > float(thr):
            move_order.append(('C_24to22', {24: -1, 22: +1}))

        # 当羰基区提示低场端偏弱、高场端偏强时，优先推动 23 -> 24 / 22 -> 24 的组合
        if bool(carbonyl_couple) and float(mid['pos']) > float(thr) and float(low['neg']) > float(thr):
            coupled = []
            if float(s23['neg']) > float(thr) and float(s24['pos']) > float(thr):
                coupled.append(('C_couple_23to24', {23: -1, 24: +1}))
            if float(s22['neg']) > float(thr) and float(s24['pos']) > float(thr):
                coupled.append(('C_couple_22to24', {22: -1, 24: +1}))
            if float(s23['neg']) > float(thr) and float(s22['pos']) > float(thr) and float(s24['pos']) > float(thr):
                coupled.insert(0, ('C_couple_2x23_to_22_24', {23: -2, 22: +1, 24: +1}))
            move_order = coupled + move_order

        H_work = torch.clamp(H, min=0).long().clone()
        all_moves: List[Dict[str, Any]] = []
        keep = {22: int(min_keep_22), 23: int(min_keep_23), 24: int(min_keep_24), 25: int(min_keep_25)}
        aro23_rotation = int(getattr(self, '_block_c_aro23_rotation', 0))

        def _current_h(hh: torch.Tensor) -> float:
            return float(torch.matmul(hh.float(), self.E_SU)[1].item())

        def _h_within_tolerance(hh_before: torch.Tensor, hh_after: torch.Tensor) -> bool:
            if E_target is None:
                return True
            try:
                target_h = float(E_target.to(hh_after.device)[1].item())
            except Exception:
                return True
            if target_h <= 1e-8:
                return True
            before = float(_current_h(hh_before))
            after = float(_current_h(hh_after))
            tol = max(0.0, float(h_tolerance))
            after_rel = abs(after - target_h) / target_h
            before_rel = abs(before - target_h) / target_h
            return bool(after_rel <= tol or after_rel <= before_rel + 1e-9)

        def _apply_aro23_transfer(hh: torch.Tensor, src: int, dst: int) -> Optional[Dict[str, Any]]:
            H_try = hh.clone()
            if int(H_try[src].item()) <= 0:
                return None
            H_try[src] -= 1
            H_try[dst] += 1
            if not _h_within_tolerance(hh, H_try):
                return None
            hh.copy_(H_try)
            return {'block': 'C', 'op': f'C_{src}to{dst}', 'delta': {int(src): -1, int(dst): +1}}

        def _maybe_apply_aro23_coupling(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            nonlocal aro23_rotation
            # 20-45 区域正峰 => 欠缺 23；115-130 区域负峰 => 12/13 过多。
            need_23 = float(s23_wide['pos'])
            excess_1213 = float(s12_13['neg'])
            # 若用户观察到 diff 符号与这里相反，也允许在 |neg| 主导时触发一次弱判据。
            alt_need_23 = float(s23_wide['neg'])
            if not (
                (need_23 > float(thr) and excess_1213 > float(thr)) or
                (alt_need_23 > 1.4 * float(thr) and excess_1213 > float(thr))
            ):
                return None

            # 当24或22本身明显缺峰时，优先先做脂肪尾部平衡，不让12/13->23耦合抢占动作。
            if float(s23['neg']) > float(thr) and (float(s24['pos']) > float(thr) or float(s22['pos']) > float(thr)):
                return None

            order = [12, 13] if int(aro23_rotation) % 2 == 0 else [13, 12]
            for src in order:
                mv = _apply_aro23_transfer(hh, int(src), 23)
                if mv is not None:
                    aro23_rotation += 1
                    return mv
            return None

        def _maybe_apply_23aro_reverse(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            nonlocal aro23_rotation
            # 若 20-45 过强而 115-130 欠缺，则允许 23 -> 13/12 回补。
            if not (float(s23_wide['neg']) > float(thr) and float(s12_13['pos']) > float(thr)):
                return None
            order = [13, 12] if int(aro23_rotation) % 2 == 0 else [12, 13]
            for dst in order:
                mv = _apply_aro23_transfer(hh, 23, int(dst))
                if mv is not None:
                    aro23_rotation += 1
                    return mv
            return None

        for _ in range(max(0, int(max_moves))):
            special_move = _maybe_apply_aro23_coupling(H_work)
            if special_move is None:
                special_move = _maybe_apply_23aro_reverse(H_work)
            if special_move is not None:
                all_moves.append(special_move)
                continue

            applied = False
            for name, delta in move_order:
                H_try = self._apply_count_delta(H_work, delta, min_keep=keep)
                if H_try is None:
                    continue
                H_work = H_try
                all_moves.append({'block': 'C', 'op': name, 'delta': dict(delta)})
                applied = True
                break
            if not applied:
                break

        self._block_c_aro23_rotation = int(aro23_rotation)

        meta = {
            'threshold': float(thr),
            'windows': {
                '22': s22, '23': s23, '24': s24, '25': s25,
                '23_20_45': s23_wide, '12_13_115_130': s12_13,
                f'{low_lo:.1f}_{low_hi:.1f}': low,
                f'{mid_lo:.1f}_{mid_hi:.1f}': mid,
            },
            'move_order': [name for name, _ in move_order],
            'h_tolerance': float(h_tolerance),
        }
        return H_work, all_moves, meta

    # ========================================================================
    # 羰基调整 (1/2/3)
    # ========================================================================
    
    def adjust_carbonyl_by_difference(self,
                                     H: torch.Tensor,
                                     ppm: Optional[np.ndarray],
                                     diff: Optional[np.ndarray],
                                     window_12: float = 5.0,
                                     window_3: float = 10.0,
                                     score_rel_threshold: float = 0.15,
                                     max_moves: int = 5,
                                     min_keep: int = 1) -> Tuple[torch.Tensor, List[Dict], Dict]:
        """
        基于差谱调整羰基类型（1/2/3 互转）
        
        策略：
        - 1号（羧酸）~174.8ppm ±window_12
        - 2号（酯）~169.6ppm ±window_12
        - 3号（醛酮）~195.8ppm ±window_3
        - 正峰 -> 增加该类型
        - 负峰 -> 减少该类型
        - 守恒互转：优先 3↔1/2，允许 1↔2
        """
        print("\n[羰基调整] 基于差谱分析")
        
        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}
 
        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            print("  差谱为空，跳过调整")
            return H, [], {}

        lo_0, hi_0, mu_0 = self._get_su_common_window(0, fallback_mu=167.125, pad=0.25 * float(window_12), min_half_width=float(window_12))
        lo_1, hi_1, mu_1 = self._get_su_common_window(1, fallback_mu=174.8, pad=0.25 * float(window_12), min_half_width=float(window_12))
        lo_2, hi_2, mu_2 = self._get_su_common_window(2, fallback_mu=169.6, pad=0.25 * float(window_12), min_half_width=float(window_12))
        lo_3, hi_3, mu_3 = self._get_su_common_window(3, fallback_mu=195.8, pad=0.25 * float(window_3), min_half_width=float(window_3))

        carbonyl_mask = (ppm_arr >= 160.0) & (ppm_arr <= 240.0)
        carbonyl_abs = float(np.sum(np.abs(diff_arr[carbonyl_mask]))) if bool(carbonyl_mask.any()) else float(np.sum(np.abs(diff_arr)))
        thr = float(score_rel_threshold) * max(1e-9, float(carbonyl_abs))

        s0 = self._window_stats(ppm_arr, diff_arr, lo_0, hi_0)
        s1 = self._window_stats(ppm_arr, diff_arr, lo_1, hi_1)
        s2 = self._window_stats(ppm_arr, diff_arr, lo_2, hi_2)
        s3 = self._window_stats(ppm_arr, diff_arr, lo_3, hi_3)

        print(f"  0号@{mu_0:.3f} [{lo_0:.3f},{hi_0:.3f}] pos={float(s0['pos']):.3f}, neg={float(s0['neg']):.3f}, net={float(s0['net']):.3f} (固定不调整)")
        print(f"  1号@{mu_1:.3f} [{lo_1:.3f},{hi_1:.3f}] pos={float(s1['pos']):.3f}, neg={float(s1['neg']):.3f}, net={float(s1['net']):.3f}")
        print(f"  2号@{mu_2:.3f} [{lo_2:.3f},{hi_2:.3f}] pos={float(s2['pos']):.3f}, neg={float(s2['neg']):.3f}, net={float(s2['net']):.3f}")
        print(f"  3号@{mu_3:.3f} [{lo_3:.3f},{hi_3:.3f}] pos={float(s3['pos']):.3f}, neg={float(s3['neg']):.3f}, net={float(s3['net']):.3f}")
        print(f"  threshold={thr:.3f} (score_rel_threshold={float(score_rel_threshold):.4f}, carbonyl_abs={carbonyl_abs:.3f})")

        def _need(stats: Dict[str, float]) -> int:
            pos = float(stats.get('pos', 0.0))
            neg_abs = abs(float(stats.get('neg', 0.0)))
            net_abs = abs(float(stats.get('net', 0.0)))
            if pos > thr and neg_abs > thr and net_abs < 0.25 * (pos + neg_abs):
                return 0
            dom = float(stats.get('dom', 0.0))
            if dom > thr:
                return 1
            if dom < -thr:
                return -1
            return 0

        stats_map = {1: s1, 2: s2, 3: s3}
        needs = {k: _need(v) for k, v in stats_map.items()}
        print(f"  需求判断(正=缺乏/需增加, 负=过量/需减少): {needs}")

        H_new = H.clone()
        moves: List[Dict] = []

        def _count(k: int) -> int:
            return int(H_new[k].item())

        for _ in range(int(max_moves)):
            inc_candidates = [k for k, v in needs.items() if int(v) > 0]
            if not inc_candidates:
                break

            receiver = max(inc_candidates, key=lambda k: abs(float(stats_map[k].get('dom', 0.0))))

            donor = None
            if int(receiver) in (1, 2) and _count(3) > int(min_keep) and int(receiver) != 3:
                if int(needs.get(3, 0)) <= 0:
                    if int(needs.get(3, 0)) < 0 or not any(int(needs.get(k, 0)) < 0 for k in (1, 2, 3)):
                        donor = 3

            if donor is None:
                dec_candidates = [k for k, v in needs.items() if int(v) < 0 and int(k) != int(receiver) and _count(int(k)) > int(min_keep)]
                if dec_candidates:
                    donor = min(dec_candidates, key=lambda k: float(stats_map[k].get('dom', 0.0)))

            if donor is None:
                fallback = [k for k in (3, 1, 2) if int(k) != int(receiver) and _count(int(k)) > int(min_keep)]
                if fallback:
                    donor = min(fallback, key=lambda k: float(stats_map[int(k)].get('dom', 0.0)))

            if donor is None:
                break

            if _count(int(donor)) <= int(min_keep):
                break

            H_new[int(donor)] -= 1
            H_new[int(receiver)] += 1
            moves.append({'from': int(donor), 'to': int(receiver)})
            print(f"    {int(donor)} -> {int(receiver)}")

        meta = {
            'n_moves': int(len(moves)),
            'threshold': float(thr),
            'carbonyl_abs': float(carbonyl_abs),
            'scores': {'0': s0, '1': s1, '2': s2, '3': s3},
            'needs': needs,
        }

        print(f"  完成 {len(moves)} 次羰基互转")
        return H_new, moves, meta
    
    # ========================================================================
    # SU9调整
    # ========================================================================
    
    def adjust_su9_by_difference(self,
                                H: torch.Tensor,
                                ppm: Optional[np.ndarray],
                                diff: Optional[np.ndarray],
                                window: float = 10.0,
                                score_rel_threshold: float = 0.15,
                                max_moves: int = 5,
                                min_keep: int = 1) -> Tuple[torch.Tensor, List[Dict], Dict]:
        """
        基于差谱调整 SU9 数量（9 ↔ 10/11/12/13）
        
        策略：
        - 对每个候选 hop1 模式（9, 19, 20, 21, 23, 24, 25）计算得分
        - 如果非9候选得分 > SU9得分 -> 减少9号
        - 减少9号时转换为11号（芳香烷基取代碳）
        - 增加9号时从10/11/12/13中选择
        """
        print("\n[SU9调整] 基于差谱分析")
        
        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}
        
        mu_map = self._get_su_hop1_mu_median()
        
        # 候选 hop1 singleton 类型
        candidate_hop1_types = [9, 19, 20, 21, 23, 24, 25]
        
        # 计算每个候选的得分
        scores = {}
        for hop1_type in candidate_hop1_types:
            key_1 = (1, (hop1_type,))
            mu_1 = mu_map.get(key_1, None)
            if mu_1 is None:
                scores[hop1_type] = -999.0
                continue
            
            mask = (ppm >= mu_1 - window) & (ppm <= mu_1 + window)
            score = float(np.sum(diff[mask])) if mask.any() else 0.0
            scores[hop1_type] = score
        
        print(f"  候选得分: {scores}")
        
        # 找最佳非9候选
        non9_scores = {k: v for k, v in scores.items() if k != 9}
        if not non9_scores:
            print("  无有效非9候选，跳过调整")
            return H, [], {}
        
        best_non9 = max(non9_scores, key=lambda k: non9_scores[k])
        best_non9_score = non9_scores[best_non9]
        su9_score = scores.get(9, -999.0)
        
        print(f"  最佳非9候选: {best_non9} (score={best_non9_score:.3f})")
        print(f"  SU9得分: {su9_score:.3f}")
        
        H_new = H.clone()
        moves = []
        
        # 决定调整方向
        if best_non9_score > su9_score + score_rel_threshold:
            # 减少9号
            current_9 = int(H_new[9].item())
            n_reduce = min(max_moves, max(0, current_9 - min_keep))
            
            for _ in range(n_reduce):
                H_new[9] -= 1
                H_new[11] += 1
                moves.append({'type': 'reduce', 'from': 9, 'to': 11})
            
            print(f"  减少9号: {current_9} -> {int(H_new[9].item())} (转为11号)")
        
        elif su9_score > best_non9_score + score_rel_threshold:
            # 增加9号
            aromatic_pool = [10, 11, 12, 13]
            available = [su for su in aromatic_pool if int(H_new[su].item()) > min_keep]
            
            if available:
                n_increase = min(max_moves, len(available))
                for _ in range(n_increase):
                    donor = available[0]
                    H_new[donor] -= 1
                    H_new[9] += 1
                    moves.append({'type': 'increase', 'from': donor, 'to': 9})
                    if int(H_new[donor].item()) <= min_keep:
                        available.remove(donor)
                    if not available:
                        break
                
                print(f"  增加9号: 从芳香碳池转入 {len(moves)} 个")
        
        meta = {
            'n_moves': len(moves),
            'best_non9': best_non9,
            'best_non9_score': best_non9_score,
            'su9_score': su9_score,
        }
        
        print(f"  完成 {len(moves)} 次SU9调整")
        return H_new, moves, meta
    
    # ========================================================================
    # O连接调整 (5/19)
    # ========================================================================
    
    def adjust_ether_519_by_difference(self,
                                      H: torch.Tensor,
                                      ppm: Optional[np.ndarray],
                                      diff: Optional[np.ndarray],
                                      window_5: float = 3.0,
                                      window_19: float = 3.0,
                                      peak_rel_threshold: float = 0.01,
                                      max_moves: int = 5,
                                      min_keep: int = 1,
                                      reserved_19: int = 0) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[O连接(5/19)调整] 基于差谱分析")

        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            print("  差谱为空，跳过调整")
            return H, [], {}
        lo_5, hi_5, _ = self._get_su_common_window(5, fallback_mu=154.75)
        edge_5 = float(min(max(lo_5 + 0.5 * (hi_5 - lo_5), lo_5 + 1e-6), hi_5 - 1e-6))
        s5_lo = self._window_stats(ppm_arr, diff_arr, lo_5, edge_5)
        s5_hi = self._window_stats(ppm_arr, diff_arr, edge_5, hi_5)

        lo_19, hi_19, _ = self._get_su_common_window(19, fallback_mu=66.6875)
        edges_19 = np.linspace(float(lo_19), float(hi_19), 4)
        s19_lo = self._window_stats(ppm_arr, diff_arr, float(edges_19[0]), float(edges_19[1]))
        s19_mid = self._window_stats(ppm_arr, diff_arr, float(edges_19[1]), float(edges_19[2]))
        s19_hi = self._window_stats(ppm_arr, diff_arr, float(edges_19[2]), float(edges_19[3]))
        s23_tail = self._window_stats(ppm_arr, diff_arr, 18.0, 35.0)
        s24_tail = self._window_stats(ppm_arr, diff_arr, 32.0, 50.0)
        s25_tail = self._window_stats(ppm_arr, diff_arr, 45.0, 65.0)

        band_abs = (
            float(s5_lo['abs']) + float(s5_hi['abs']) +
            float(s19_lo['abs']) + float(s19_mid['abs']) + float(s19_hi['abs'])
        )
        thr = float(peak_rel_threshold) * max(1e-8, band_abs)

        need_5 = 0.8 * float(s5_lo['pos']) + 1.2 * float(s5_hi['pos'])
        excess_5 = 0.8 * float(s5_lo['neg']) + 1.2 * float(s5_hi['neg'])
        # 19 的增量证据必须主要来自 63-72 核心窗口，72-80 只给较小权重，避免 shoulder 误增。
        need_19 = 0.35 * float(s19_lo['pos']) + 1.00 * float(s19_mid['pos']) + 0.20 * float(s19_hi['pos'])
        # 对 19 的减少更敏感，尤其当 72-80 形成宽肩时。
        excess_19 = 0.45 * float(s19_lo['neg']) + 1.00 * float(s19_mid['neg']) + 0.65 * float(s19_hi['neg'])

        W_ether = int(H[2].item()) + int(H[28].item()) + 2 * int(H[29].item())

        def _effective_19_count(hh: torch.Tensor) -> int:
            return max(0, int(hh[19].item()) - int(reserved_19))

        soft_cap_19 = int(reserved_19) + max(int(min_keep), int(math.ceil(0.35 * float(W_ether))))
        soft_floor_5 = max(int(min_keep), int(math.ceil(0.50 * float(W_ether))))

        print(
            f"  SU5 band 145-160: need={need_5:.3f}, excess={excess_5:.3f} | "
            f"sub=[145-151 pos={float(s5_lo['pos']):.3f} neg={float(s5_lo['neg']):.3f}, "
            f"151-160 pos={float(s5_hi['pos']):.3f} neg={float(s5_hi['neg']):.3f}]"
        )
        print(
            f"  SU19 band 55-80: need={need_19:.3f}, excess={excess_19:.3f} | "
            f"sub=[55-63 pos={float(s19_lo['pos']):.3f} neg={float(s19_lo['neg']):.3f}, "
            f"63-72 pos={float(s19_mid['pos']):.3f} neg={float(s19_mid['neg']):.3f}, "
            f"72-80 pos={float(s19_hi['pos']):.3f} neg={float(s19_hi['neg']):.3f}]"
        )
        print(
            f"  threshold={thr:.3f}, W_ether={int(W_ether)}, "
            f"soft_floor_5={int(soft_floor_5)}, soft_cap_19={int(soft_cap_19)} (reserved_19={int(reserved_19)})"
        )

        if max(float(need_5), float(excess_5), float(need_19), float(excess_19)) < float(thr):
            print("  峰强不足，跳过调整")
            return H, [], {
                'scores': {
                    '5_need': float(need_5), '5_excess': float(excess_5),
                    '19_need': float(need_19), '19_excess': float(excess_19),
                },
                'windows': {
                    '5_145_151': s5_lo, '5_151_160': s5_hi,
                    '19_55_63': s19_lo, '19_63_72': s19_mid, '19_72_80': s19_hi,
                },
                'threshold': float(thr),
                'reserved_19': int(reserved_19),
                'soft_floor_5': int(soft_floor_5),
                'soft_cap_19': int(soft_cap_19),
            }

        H_new = torch.clamp(H, min=0).long().clone()
        moves: List[Dict[str, Any]] = []

        def _pick_5_donor(hh: torch.Tensor) -> Optional[int]:
            cands = []
            for su in (11, 13):
                cnt = int(hh[su].item())
                if cnt > 0:
                    cands.append((cnt, su))
            if not cands:
                return None
            cands.sort(reverse=True)
            return int(cands[0][1])

        def _pick_tail_acceptor() -> int:
            tail_scores = {
                23: float(s23_tail['pos']) - 0.4 * float(s23_tail['neg']),
                24: float(s24_tail['pos']) - 0.4 * float(s24_tail['neg']),
                25: float(s25_tail['pos']) - 0.4 * float(s25_tail['neg']),
            }
            return int(max(tail_scores.keys(), key=lambda k: tail_scores[k]))

        def _apply_inc_5(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            donor = _pick_5_donor(hh)
            if donor is None:
                return None
            hh[donor] -= 1
            hh[5] += 1
            return {'op': 'inc_5', 'from': int(donor), 'to': 5}

        def _apply_dec_5(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            if int(hh[5].item()) <= int(min_keep):
                return None
            hh[5] -= 1
            dst = 11 if int(hh[11].item()) <= int(hh[13].item()) else 13
            hh[dst] += 1
            return {'op': 'dec_5', 'from': 5, 'to': int(dst)}

        def _apply_inc_19(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            if int(_effective_19_count(hh)) >= int(soft_cap_19):
                return None
            donor = None
            if int(hh[23].item()) > 0:
                donor = 23
            elif int(hh[24].item()) > 0:
                donor = 24
            if donor is None:
                return None
            hh[donor] -= 1
            hh[19] += 1
            return {'op': 'inc_19', 'from': int(donor), 'to': 19}

        def _apply_dec_19(hh: torch.Tensor) -> Optional[Dict[str, Any]]:
            if int(_effective_19_count(hh)) <= int(min_keep):
                return None
            hh[19] -= 1
            dst = _pick_tail_acceptor()
            hh[dst] += 1
            return {'op': 'dec_19', 'from': 19, 'to': int(dst), 'reserved_19': int(reserved_19)}

        # 微循环：每步基于宽带窗口证据和 soft cap/floor 决策。
        v_need_5 = float(need_5)
        v_excess_5 = float(excess_5)
        v_need_19 = float(need_19)
        v_excess_19 = float(excess_19)

        for _ in range(max(0, int(max_moves))):
            eff19 = int(_effective_19_count(H_new))
            cur5 = int(H_new[5].item())
            over_19 = max(0, int(eff19 - int(soft_cap_19)))
            under_5 = max(0, int(int(soft_floor_5) - int(cur5)))

            op_sequence: List[str] = []
            # 5 缺且 19 过：优先做 dec19 + inc5
            if (float(v_need_5) > float(thr) or int(under_5) > 0) and (float(v_excess_19) > 0.6 * float(thr) or int(over_19) > 0):
                op_sequence = ['dec19_inc5', 'inc5_only', 'dec19_only']
            elif float(v_need_5) > float(thr) or int(under_5) > 0:
                op_sequence = ['inc5_only', 'dec19_inc5']
            elif float(v_excess_19) > float(thr) or int(over_19) > 0:
                op_sequence = ['dec19_only', 'dec19_inc5']
            elif float(v_need_19) > 1.5 * float(thr) and float(v_need_5) < 0.6 * float(thr) and int(eff19) < int(soft_cap_19):
                op_sequence = ['inc19_only']
            elif float(v_excess_5) > 1.6 * float(thr) and float(v_need_19) > 1.8 * float(thr) and int(cur5) > int(soft_floor_5):
                op_sequence = ['dec5_inc19']
            else:
                break

            applied = False
            for op_name in op_sequence:
                H_try = H_new.clone()
                step_moves: List[Dict[str, Any]] = []

                if op_name == 'dec19_inc5':
                    mv1 = _apply_dec_19(H_try)
                    mv2 = _apply_inc_5(H_try)
                    if mv1 and mv2:
                        step_moves.extend([mv1, mv2])
                elif op_name == 'inc5_only':
                    mv = _apply_inc_5(H_try)
                    if mv:
                        step_moves.append(mv)
                elif op_name == 'dec19_only':
                    mv = _apply_dec_19(H_try)
                    if mv:
                        step_moves.append(mv)
                elif op_name == 'inc19_only':
                    mv = _apply_inc_19(H_try)
                    if mv:
                        step_moves.append(mv)
                elif op_name == 'dec5_inc19':
                    mv1 = _apply_dec_5(H_try)
                    mv2 = _apply_inc_19(H_try)
                    if mv1 and mv2:
                        step_moves.extend([mv1, mv2])

                if not step_moves:
                    continue

                H_new = H_try
                moves.extend(step_moves)
                applied = True

                # 轻量虚拟更新，避免在同一个方向上过冲。
                if op_name in ('dec19_inc5', 'inc5_only'):
                    v_need_5 *= 0.55
                if op_name in ('dec19_inc5', 'dec19_only'):
                    v_excess_19 *= 0.55
                if op_name == 'inc19_only':
                    v_need_19 *= 0.60
                if op_name == 'dec5_inc19':
                    v_excess_5 *= 0.60
                    v_need_19 *= 0.60
                break

            if not applied:
                break

        meta = {
            'scores': {
                '5_need': float(need_5),
                '5_excess': float(excess_5),
                '19_need': float(need_19),
                '19_excess': float(excess_19),
            },
            'windows': {
                '5_145_151': s5_lo,
                '5_151_160': s5_hi,
                '19_55_63': s19_lo,
                '19_63_72': s19_mid,
                '19_72_80': s19_hi,
            },
            'threshold': float(thr),
            'reserved_19': int(reserved_19),
            'soft_floor_5': int(soft_floor_5),
            'soft_cap_19': int(soft_cap_19),
        }

        print(f"  完成 {len(moves)} 条变更记录")
        print(f"  H[5]={int(H[5].item())} -> {int(H_new[5].item())}, H[19]={int(H[19].item())} -> {int(H_new[19].item())} (reserved_19={int(reserved_19)})")
        return H_new, moves, meta
    
    # ========================================================================
    # N连接调整 (6/20)
    # ========================================================================
    
    def adjust_amine_620_by_difference(self,
                                      H: torch.Tensor,
                                      ppm: Optional[np.ndarray],
                                      diff: Optional[np.ndarray],
                                      window_6: float = 3.0,
                                      window_20: float = 3.0,
                                      peak_rel_threshold: float = 0.01,
                                      max_moves: int = 5,
                                      min_keep: int = 0) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[N连接(6/20)调整] 基于差谱分析")

        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            print("  差谱为空，跳过调整")
            return H, [], {}

        lo_6, hi_6, mu_6 = self._get_su_common_window(6, fallback_mu=146.375, pad=0.20 * float(window_6), min_half_width=float(window_6))
        lo_20, hi_20, mu_20 = self._get_su_common_window(20, fallback_mu=49.375, pad=0.20 * float(window_20), min_half_width=float(window_20))

        def _window_score(lo: float, hi: float) -> Dict[str, float]:
            stats = self._window_stats(ppm_arr, diff_arr, lo, hi)
            dom = float(stats['pos']) if float(stats['pos']) >= float(stats['neg']) else -float(stats['neg'])
            return {'pos': float(stats['pos']), 'neg': -float(stats['neg']), 'score': float(dom)}

        s6 = _window_score(lo_6, hi_6)
        s20 = _window_score(lo_20, hi_20)
        score_6 = float(s6.get('score', 0.0))
        score_20 = float(s20.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU6@{mu_6:.3f} [{lo_6:.3f},{hi_6:.3f}] score={score_6:.3f} (pos={float(s6.get('pos', 0.0)):.3f}, neg={float(s6.get('neg', 0.0)):.3f})")
        print(f"  SU20@{mu_20:.3f} [{lo_20:.3f},{hi_20:.3f}] score={score_20:.3f} (pos={float(s20.get('pos', 0.0)):.3f}, neg={float(s20.get('neg', 0.0)):.3f})")
        print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

        if max(abs(score_6), abs(score_20)) < thr:
            print("  峰强不足，跳过调整")
            return H, [], {
                'scores': {'6': s6, '20': s20},
                'centers': {'6': mu_6, '20': mu_20},
                'threshold': thr,
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
        moves: List[Dict] = []

        for _ in range(max(0, int(max_moves))):
            ok_inc = True
            ok_dec = True

            if int(inc) == 6:
                donor = None
                try:
                    d13 = int(H_new[13].item())
                    d11 = int(H_new[11].item())
                    cand = [(d13, 13), (d11, 11)]
                    cand.sort(reverse=True)
                    for cnt, su in cand:
                        if int(cnt) > 0:
                            donor = int(su)
                            break
                except Exception:
                    donor = 13 if int(H_new[13].item()) > 0 else (11 if int(H_new[11].item()) > 0 else None)

                if donor is None:
                    ok_inc = False
                else:
                    H_new[donor] -= 1
                    H_new[6] += 1
                    moves.append({'op': 'inc_6', 'from': donor, 'to': 6})
            else:
                if int(H_new[23].item()) <= 0:
                    ok_inc = False
                else:
                    H_new[23] -= 1
                    H_new[20] += 1
                    moves.append({'op': 'inc_20', 'from': 23, 'to': 20})

            if int(dec) == 20:
                if int(H_new[20].item()) <= int(min_keep):
                    ok_dec = False
                else:
                    H_new[20] -= 1
                    H_new[23] += 1
                    moves.append({'op': 'dec_20', 'from': 20, 'to': 23})
            else:
                if int(H_new[6].item()) <= int(min_keep):
                    ok_dec = False
                else:
                    H_new[6] -= 1
                    dst = 13 if int(H_new[13].item()) <= int(H_new[11].item()) else 11
                    H_new[int(dst)] += 1
                    moves.append({'op': 'dec_6', 'from': 6, 'to': int(dst)})

            if not (ok_inc and ok_dec):
                break

        meta = {
            'scores': {'6': s6, '20': s20},
            'centers': {'6': mu_6, '20': mu_20},
            'threshold': thr,
            'direction': {'inc': int(inc), 'dec': int(dec)},
        }

        print(f"  完成 {len(moves)} 条变更记录")
        print(f"  H[6]={int(H[6].item())} -> {int(H_new[6].item())}, H[20]={int(H[20].item())} -> {int(H_new[20].item())}")
        return H_new, moves, meta
    
    # ========================================================================
    # S连接调整 (7/19)
    # ========================================================================
    
    def adjust_thioether_719_by_difference(self,
                                          H: torch.Tensor,
                                          ppm: Optional[np.ndarray],
                                          diff: Optional[np.ndarray],
                                          window_7: float = 3.0,
                                          window_19: float = 3.0,
                                          peak_rel_threshold: float = 0.01,
                                          max_moves: int = 5,
                                          min_keep: int = 0) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[S连接(7/19)调整] 基于差谱分析")

        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            print("  差谱为空，跳过调整")
            return H, [], {}

        lo_7, hi_7, mu_7 = self._get_su_common_window(7, fallback_mu=152.875, pad=0.20 * float(window_7), min_half_width=float(window_7))
        lo_19, hi_19, mu_19 = self._get_su_common_window(19, fallback_mu=66.6875, pad=0.20 * float(window_19), min_half_width=float(window_19))

        def _window_score(lo: float, hi: float) -> Dict[str, float]:
            stats = self._window_stats(ppm_arr, diff_arr, lo, hi)
            dom = float(stats['pos']) if float(stats['pos']) >= float(stats['neg']) else -float(stats['neg'])
            return {'pos': float(stats['pos']), 'neg': -float(stats['neg']), 'score': float(dom)}

        s7 = _window_score(lo_7, hi_7)
        s19 = _window_score(lo_19, hi_19)
        score_7 = float(s7.get('score', 0.0))
        score_19 = float(s19.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU7@{mu_7:.3f} [{lo_7:.3f},{hi_7:.3f}] score={score_7:.3f} (pos={float(s7.get('pos', 0.0)):.3f}, neg={float(s7.get('neg', 0.0)):.3f})")
        print(f"  SU19@{mu_19:.3f} [{lo_19:.3f},{hi_19:.3f}] score={score_19:.3f} (pos={float(s19.get('pos', 0.0)):.3f}, neg={float(s19.get('neg', 0.0)):.3f})")
        print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

        if max(abs(score_7), abs(score_19)) < thr:
            print("  峰强不足，跳过调整")
            return H, [], {
                'scores': {'7': s7, '19': s19},
                'centers': {'7': mu_7, '19': mu_19},
                'threshold': thr,
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
        moves: List[Dict] = []

        W = int(2 * H_new[31].item())
        if int(W) <= 0:
            print("  H[31]=0，无硫醚连接需求，跳过调整")
            return H, [], {
                'scores': {'7': s7, '19': s19},
                'centers': {'7': mu_7, '19': mu_19},
                'threshold': thr,
                'direction': {'inc': int(inc), 'dec': int(dec)},
                'W': int(W),
            }

        reserved_19_init = max(0, int(W) - int(H_new[7].item()))
        o_base_19 = int(H_new[19].item()) - int(reserved_19_init)
        if int(o_base_19) < 0:
            o_base_19 = 0

        def _target_19(hh: torch.Tensor) -> int:
            need_reserved = max(0, int(W) - int(hh[7].item()))
            return int(o_base_19) + int(need_reserved)

        tgt0 = _target_19(H_new)
        while int(H_new[19].item()) < int(tgt0) and int(H_new[23].item()) > 0:
            H_new[23] -= 1
            H_new[19] += 1
            moves.append({'op': 'topup_reserved_19', 'from': 23, 'to': 19})

        for _ in range(max(0, int(max_moves))):
            ok = True

            if int(inc) == 7:
                if int(H_new[7].item()) >= int(W):
                    ok = False
                else:
                    donor = None
                    try:
                        d13 = int(H_new[13].item())
                        d11 = int(H_new[11].item())
                        cand = [(d13, 13), (d11, 11)]
                        cand.sort(reverse=True)
                        for cnt, su in cand:
                            if int(cnt) > 0:
                                donor = int(su)
                                break
                    except Exception:
                        donor = 13 if int(H_new[13].item()) > 0 else (11 if int(H_new[11].item()) > 0 else None)
                    if donor is None:
                        ok = False
                    else:
                        H_new[donor] -= 1
                        H_new[7] += 1
                        moves.append({'op': 'inc_7', 'from': donor, 'to': 7})
            else:
                if int(H_new[7].item()) <= int(min_keep) or int(H_new[23].item()) <= 0:
                    ok = False
                else:
                    H_new[7] -= 1
                    dst = 13 if int(H_new[13].item()) <= int(H_new[11].item()) else 11
                    H_new[int(dst)] += 1
                    moves.append({'op': 'dec_7', 'from': 7, 'to': int(dst)})

            if not bool(ok):
                break

            tgt = _target_19(H_new)
            while int(H_new[19].item()) > int(tgt):
                H_new[19] -= 1
                H_new[23] += 1
                moves.append({'op': 'trim_reserved_19', 'from': 19, 'to': 23, 'o_base_19': int(o_base_19)})

            while int(H_new[19].item()) < int(tgt) and int(H_new[23].item()) > 0:
                H_new[23] -= 1
                H_new[19] += 1
                moves.append({'op': 'topup_reserved_19', 'from': 23, 'to': 19, 'o_base_19': int(o_base_19)})

            if int(H_new[19].item()) != int(tgt):
                break

        meta = {
            'scores': {'7': s7, '19': s19},
            'centers': {'7': mu_7, '19': mu_19},
            'threshold': thr,
            'direction': {'inc': int(inc), 'dec': int(dec)},
            'W': int(W),
            'o_base_19': int(o_base_19),
        }

        print(f"  完成 {len(moves)} 条变更记录")
        print(f"  H[7]={int(H[7].item())} -> {int(H_new[7].item())}, H[19]={int(H[19].item())} -> {int(H_new[19].item())} (o_base_19={int(o_base_19)}, W={int(W)})")
        return H_new, moves, meta
    
    # ========================================================================
    # X连接调整 (8/21)
    # ========================================================================
    
    def adjust_halogen_821_by_difference(self,
                                        H: torch.Tensor,
                                        ppm: Optional[np.ndarray],
                                        diff: Optional[np.ndarray],
                                        window_8: float = 3.0,
                                        window_21: float = 3.0,
                                        peak_rel_threshold: float = 0.01,
                                        max_moves: int = 5,
                                        min_keep: int = 0) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print("\n[X连接(8/21)调整] 基于差谱分析")

        if ppm is None or diff is None:
            print("  无差谱数据，跳过调整")
            return H, [], {}

        ppm_arr = np.asarray(ppm, dtype=np.float64)
        diff_arr = np.asarray(diff, dtype=np.float64)
        if int(ppm_arr.size) == 0 or int(diff_arr.size) == 0:
            print("  差谱为空，跳过调整")
            return H, [], {}

        lo_8, hi_8, mu_8 = self._get_su_common_window(8, fallback_mu=131.4244, pad=0.20 * float(window_8), min_half_width=float(window_8))
        lo_21, hi_21, mu_21 = self._get_su_common_window(21, fallback_mu=38.4141, pad=0.20 * float(window_21), min_half_width=float(window_21))

        def _window_score(lo: float, hi: float) -> Dict[str, float]:
            stats = self._window_stats(ppm_arr, diff_arr, lo, hi)
            dom = float(stats['pos']) if float(stats['pos']) >= float(stats['neg']) else -float(stats['neg'])
            return {'pos': float(stats['pos']), 'neg': -float(stats['neg']), 'score': float(dom)}

        s8 = _window_score(lo_8, hi_8)
        s21 = _window_score(lo_21, hi_21)
        score_8 = float(s8.get('score', 0.0))
        score_21 = float(s21.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU8@{mu_8:.3f} [{lo_8:.3f},{hi_8:.3f}] score={score_8:.3f} (pos={float(s8.get('pos', 0.0)):.3f}, neg={float(s8.get('neg', 0.0)):.3f})")
        print(f"  SU21@{mu_21:.3f} [{lo_21:.3f},{hi_21:.3f}] score={score_21:.3f} (pos={float(s21.get('pos', 0.0)):.3f}, neg={float(s21.get('neg', 0.0)):.3f})")
        print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

        if max(abs(score_8), abs(score_21)) < thr:
            print("  峰强不足，跳过调整")
            return H, [], {
                'scores': {'8': s8, '21': s21},
                'centers': {'8': mu_8, '21': mu_21},
                'threshold': thr,
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
        moves: List[Dict] = []

        W = int(H_new[32].item())
        total_x = int(H_new[8].item()) + int(H_new[21].item())
        if int(W) != int(total_x):
            print(f"  警告: 当前(8+21)={total_x} 与 H[32]={W} 不一致，跳过调整")
            return H, [], {
                'scores': {'8': s8, '21': s21},
                'centers': {'8': mu_8, '21': mu_21},
                'threshold': thr,
                'direction': {'inc': int(inc), 'dec': int(dec)},
                'W': int(W),
                'total_x': int(total_x),
            }

        for _ in range(max(0, int(max_moves))):
            ok = True

            if int(inc) == 8:
                if int(H_new[13].item()) <= 0 or int(H_new[21].item()) <= int(min_keep):
                    ok = False
                else:
                    H_new[13] -= 1
                    H_new[8] += 1
                    H_new[21] -= 1
                    H_new[23] += 1
                    moves.append({'op': 'inc_8_dec_21', '13_to_8': 1, '21_to_23': 1})
            else:
                if int(H_new[23].item()) <= 0 or int(H_new[8].item()) <= int(min_keep):
                    ok = False
                else:
                    H_new[23] -= 1
                    H_new[21] += 1
                    H_new[8] -= 1
                    H_new[13] += 1
                    moves.append({'op': 'inc_21_dec_8', '23_to_21': 1, '8_to_13': 1})

            if not bool(ok):
                break

        meta = {
            'scores': {'8': s8, '21': s21},
            'centers': {'8': mu_8, '21': mu_21},
            'threshold': thr,
            'direction': {'inc': int(inc), 'dec': int(dec)},
            'W': int(W),
            'total_x': int(W),
        }

        print(f"  完成 {len(moves)} 条变更记录")
        print(f"  H[8]={int(H[8].item())} -> {int(H_new[8].item())}, H[21]={int(H[21].item())} -> {int(H_new[21].item())} (8+21={int(H_new[8].item()) + int(H_new[21].item())} == {int(W)})")
        return H_new, moves, meta
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _get_su_hop1_mu_median(self) -> Dict[Tuple, float]:
        """加载 su_hop1_nmr_range_filtered.csv 中的 mu 中位数"""
        if self._su_hop1_mu_median_cache is not None:
            return self._su_hop1_mu_median_cache
        
        if self.su_hop1_ranges_path is None or not Path(self.su_hop1_ranges_path).exists():
            print(f"[警告] su_hop1_ranges_path 不存在: {self.su_hop1_ranges_path}")
            self._su_hop1_mu_median_cache = {}
            return self._su_hop1_mu_median_cache

        df = pd.read_csv(self.su_hop1_ranges_path)
        mu_map: Dict[Tuple, float] = {}

        for _, row in df.iterrows():
            if 'center_su_idx' in row:
                center_su = int(row['center_su_idx'])
            else:
                center_su = int(row.get('center_su', 0))

            hop1_str = str(row.get('hop1_multiset', ''))
            mu_median = float(row.get('mu_median', 0.0))

            hop1_str = hop1_str.strip().strip('"').strip("'")
            hop1_str = hop1_str.strip('[]')
            if not hop1_str:
                hop1_tuple = ()
            else:
                parts = [p.strip() for p in hop1_str.split(',') if p.strip()]
                hop1_tuple = tuple(sorted(int(x) for x in parts))

            key = (int(center_su), hop1_tuple)
            mu_map[key] = mu_median

        self._su_hop1_mu_median_cache = mu_map
        print(f"[SU调整器] 加载 {len(mu_map)} 条 hop1 mu 中位数")
        return mu_map
    
    # ========================================================================
    # 分阶段调整接口
    # ========================================================================
    
    def _make_h_helpers(self):
        def _current_h(tmp: torch.Tensor):
            return float(torch.matmul(tmp.float(), self.E_SU.cpu())[1].item())
            
        def _h_ratio(tmp: torch.Tensor):
            target_H = float(self.E_target[1].item())
            if target_H <= 0: return 0.0
            return (_current_h(tmp) - target_H) / target_H
            
        def _check_h(tmp: torch.Tensor, tol: float = 0.04):
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
    def _h_rotation_adjust(tmp_nodes, H_work, h_ratio_fn, rot_idx):
        ops = []
        failed_steps = 0
        
        while abs(h_ratio_fn(H_work)) > 0.04:
            ratio = h_ratio_fn(H_work)
            step_type = rot_idx % 5
            success = False
            
            if ratio > 0.04:
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

            elif ratio < -0.04:
                if step_type in [0, 2]:
                    for n in tmp_nodes:
                        if n.su_type == 12:
                            n.su_type = 13
                            H_work[12] -= 1; H_work[13] += 1
                            ops.append('H:12->13')
                            success = True
                            break
                elif step_type in [1, 3]:
                    for n in tmp_nodes:
                        if n.su_type == 13:
                            n.su_type = 23
                            H_work[13] -= 1; H_work[23] += 1
                            ops.append('H:13->23')
                            success = True
                            break
                elif step_type == 4:
                    n_12, n_13 = None, None
                    for n in tmp_nodes:
                        if n.su_type == 12 and n_12 is None: n_12 = n
                        elif n.su_type == 13 and n_13 is None: n_13 = n
                    if n_12 and n_13:
                        n_12.su_type = 15
                        n_13.su_type = 15
                        H_work[12] -= 1; H_work[15] += 1
                        H_work[13] -= 1; H_work[15] += 1
                        ops.append('H:12+13->15+15')
                        success = True
                    
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
                                    E_target: Optional[torch.Tensor]) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
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
        rot_ops, rot_idx = self._h_rotation_adjust(tmp_nodes, H_work, h_ratio_fn, int(self._h_rotation_state))
        self._h_rotation_state = int(rot_idx)
        after_ratio = float(h_ratio_fn(H_work))

        moves = [{'stage': 'h_rotation', 'op': str(op)} for op in rot_ops]
        meta = {
            'applied': bool(rot_ops),
            'ops': list(rot_ops),
            'before_ratio': float(before_ratio),
            'after_ratio': float(after_ratio),
            'rotation_state': int(self._h_rotation_state),
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

    def _evaluate_required_hist_constraints(self,
                                            H: torch.Tensor,
                                            E_target: Optional[torch.Tensor],
                                            su22_ratio: float = 0.1,
                                            su22_h_tol: float = 0.03) -> Dict[str, Any]:
        H_cpu = torch.clamp(H, min=0).long().detach().cpu()
        if E_target is None:
            return {
                'ok': True,
                'h_ok': True,
                'h_rel': 0.0,
                'h_tol': float(max(0.04, float(su22_h_tol))),
                'su22_ok': True,
                'req22': 0,
                'n22': int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0,
                'n23': int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0,
                'even10_ok': True,
                'unsat_even_ok': True,
                'unsat_total': 0,
            }

        E_target_cpu = E_target.detach().cpu().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float)
        E_pred = torch.matmul(H_cpu.float(), self.E_SU.cpu())
        target_h = float(E_target_cpu[1].item()) if int(E_target_cpu.numel()) > 1 else 0.0
        current_h = float(E_pred[1].item()) if int(E_pred.numel()) > 1 else 0.0
        h_tol = max(0.04, float(su22_h_tol))
        if target_h > 1e-8:
            h_rel = abs(float(current_h - target_h)) / float(target_h)
            h_ok = bool(h_rel <= float(h_tol) + 1e-9)
        else:
            h_rel = 0.0
            h_ok = True

        n22 = int(H_cpu[22].item()) if int(H_cpu.numel()) > 22 else 0
        n23 = int(H_cpu[23].item()) if int(H_cpu.numel()) > 23 else 0
        req22 = max(1, int(math.ceil(float(max(0.0, su22_ratio)) * float(n23)))) if int(n23) > 0 else 0
        su22_ok = True if int(n23) <= 0 else bool(int(n22) >= int(req22))

        even10_ok = True
        if int(H_cpu.numel()) > 10:
            even10_ok = bool(int(H_cpu[10].item()) % 2 == 0)

        if int(H_cpu.numel()) > 16:
            unsat_total = int(H_cpu[14].item()) + int(H_cpu[15].item()) + int(H_cpu[16].item())
            unsat_even_ok = bool(int(unsat_total) % 2 == 0)
        else:
            unsat_total = 0
            unsat_even_ok = True

        return {
            'ok': bool(h_ok and su22_ok and even10_ok and unsat_even_ok),
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

        This keeps the skeleton-stage SU counts closer to the aromatic-cluster model,
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
        if meta:
            if 'branch_type' in meta:
                meta_brief.append(f"branch_type={meta['branch_type']}")
            if 'tail_source' in meta:
                meta_brief.append(f"tail={meta['tail_source']}")
            if 'tail_sources' in meta:
                meta_brief.append(f"tails={meta['tail_sources']}")
            if src_su:
                meta_brief.append(f"src_su={src_su}")
            if src_hop1:
                meta_brief.append(f"src_hop1={src_hop1}")
        meta_txt = f" | {', '.join(meta_brief)}" if meta_brief else ""
        return f"{ctype}/{origin}: {comp} src={src}{meta_txt}"

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

        flexible_bridge_count = sum(1 for ch in getattr(alloc_res, 'bridge_chains', []) if _is_flexible_bridge(ch))
        side_to_22_count = sum(1 for ch in getattr(alloc_res, 'side_chains', []) if _is_aliphatic_side_to_22(ch))
        aliphatic_total = int(sum(1 for n in nodes if 19 <= int(getattr(n, 'su_type', -1)) <= 25))
        effective_cluster_count = max(1, int(cluster_count))
        unallocated_bridge = int(getattr(alloc_res, 'unallocated_bridge', 0))
        unallocated_branch = int(getattr(alloc_res, 'unallocated_branch', 0))
        required_extra_11 = int(getattr(alloc_res, 'required_extra_11', 0))
        required_extra_22 = int(getattr(alloc_res, 'required_extra_22', 0))
        required_extra_23 = int(getattr(alloc_res, 'required_extra_23', 0))

        if S_target is not None and E_target is not None:
            try:
                spec = S_target.detach().cpu().flatten().numpy()
                total_area = float(np.sum(spec) * 0.1)
                ali_area = float(np.sum(spec[:900]) * 0.1)
                x = float(ali_area / total_area) if total_area > 1e-9 else 0.33
                x_pct = float(x) * 100.0
                target_c = float(E_target.detach().cpu().flatten()[0].item())
                aliphatic_min = int(math.ceil(((4.425 + 0.123 * x_pct + 0.00754 * x_pct * x_pct) / 100.0) * target_c))
                aliphatic_max = int(math.floor(0.90 * float(x) * float(target_c)))
            except Exception:
                aliphatic_min = 0
                aliphatic_max = 10**9
        else:
            aliphatic_min = 0
            aliphatic_max = 10**9

        rigid_ok = int(rigid_pairs) < int(effective_cluster_count)
        flex_hi_ok = int(flexible_bridge_count) <= int(flex_upper)
        flex_lo_ok = int(flexible_bridge_count) >= int(flex_lower)
        aliphatic_ok = int(aliphatic_total) >= int(aliphatic_min)
        aliphatic_hi_ok = int(aliphatic_total) <= int(aliphatic_max)
        branch_alloc_ok = int(unallocated_branch) == 0
        bridge_alloc_ok = int(unallocated_bridge) == 0
        extra_resource_ok = (
            int(required_extra_11) == 0 and
            int(required_extra_22) == 0 and
            int(required_extra_23) == 0
        )

        reasons = []
        if not bridge_alloc_ok:
            reasons.append('bridge_unallocated')
        if not branch_alloc_ok:
            reasons.append('branch_unallocated')
        if not extra_resource_ok:
            reasons.append('resource_shortage')
        if not rigid_ok:
            reasons.append('rigid_excess')
        if not flex_hi_ok:
            reasons.append('flex_excess')
        if not flex_lo_ok:
            reasons.append('flex_shortage')
        if not aliphatic_ok:
            reasons.append('aliphatic_shortage')
        if not aliphatic_hi_ok:
            reasons.append('aliphatic_excess')
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
                aliphatic_hi_ok
            ),
            'reason': str(reason),
            'cluster_count': int(cluster_count),
            'effective_cluster_count': int(effective_cluster_count),
            'rigid_pairs': int(rigid_pairs),
            'rigid_cluster_count': int(z_clusters),
            'rigid_min_flex': int(flex_lower),
            'flexible_bridge_min': int(flex_lower),
            'flex_ratio': float(flex_ratio),
            'flex_lower_extra': int(flex_lower_extra),
            'flexible_bridge_count': int(flexible_bridge_count),
            'flexible_bridge_limit': int(flex_upper),
            'side_to_22_count': int(side_to_22_count),
            'aliphatic_total': int(aliphatic_total),
            'aliphatic_min_total': int(aliphatic_min),
            'aliphatic_max_total': int(aliphatic_max),
            'cluster_meta': cluster_meta,
            'allocation_result': alloc_res,
            'allocation_details': self._extract_allocation_details(alloc_res),
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
        }

    def _adjust_skeleton_branch_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        ppm: Optional[np.ndarray] = None,
        diff: Optional[np.ndarray] = None,
        max_steps: int = 15,
        nodes: Optional[List[_NodeV3]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        
        print(f"\n[Skeleton-Alloc Adjust] 开始")
        if nodes is None:
            print("  [ERROR] nodes 必须提供以进行真实拓扑分配评估！")
            return H, [], {'n_moves': 0, 'ok': False}
            
        import copy
        H_work = H.cpu().clone()
        self.E_target = E_target
        tmp_nodes = copy.deepcopy(nodes)
        self._refresh_node_counters(tmp_nodes)

        tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}
        if ppm is not None and diff is not None:
            try:
                ppm_arr = np.asarray(ppm, dtype=np.float64)
                diff_arr = np.asarray(diff, dtype=np.float64)
                if int(ppm_arr.size) > 0 and int(diff_arr.size) > 0:
                    lo22, hi22, _ = self._get_su_common_window(22, fallback_mu=19.81, min_half_width=6.0)
                    lo23, hi23, _ = self._get_su_common_window(23, fallback_mu=29.48, min_half_width=6.0)
                    lo24, hi24, _ = self._get_su_common_window(24, fallback_mu=39.97, min_half_width=6.0)
                    s22 = self._window_stats(ppm_arr, diff_arr, lo22, hi22)
                    s23 = self._window_stats(ppm_arr, diff_arr, lo23, hi23)
                    s24 = self._window_stats(ppm_arr, diff_arr, lo24, hi24)
                    tail_bias = {
                        'need_22': float(s22['pos']) - float(s22['neg']),
                        'need_23': float(s23['pos']) - float(s23['neg']),
                        'need_24': float(s24['pos']) - float(s24['neg']),
                        'prefer_preserve_24': float(s24['pos']) > float(s24['neg']) + 1e-9,
                    }
            except Exception:
                tail_bias = {'need_22': 0.0, 'need_23': 0.0, 'need_24': 0.0, 'prefer_preserve_24': False}
        
        _, _h_ratio, _check_h, _ali_total = self._make_h_helpers()
        rot_idx = int(getattr(self, '_h_rotation_state', 0))
        moves = []
        
        def _log_move(op_desc, stage):
            moves.append({
                'op': op_desc,
                'stage': stage,
                'h_ratio_before_h_adjust': float(_h_ratio(H_work)),
            })
            print(f"    -> {op_desc} (H偏差: {_h_ratio(H_work)*100:.1f}%)")

        def _chain_text(ch) -> str:
            comp = '-'.join(str(int(x)) for x in getattr(ch, 'composition', []))
            ctype = getattr(ch, 'chain_type', '?')
            origin = getattr(ch, 'origin_type', '?')
            src = getattr(ch, 'source_ids', [])
            return f"{ctype}/{origin}: {comp} src={src}"

        def _print_alloc_snapshot(tag: str, res: Dict[str, Any], chain_limit: int = 8):
            closed = res.get('closed_consumed', {})
            opened = res.get('open_consumed', {})
            pre = res.get('pre_branch_available', {})
            rem = res.get('remaining', {})
            chains = res.get('branch_chains', []) or []
            if closed or opened or pre:
                print(
                    f"    [{tag}资源] closed消耗 11×{closed.get('11', 0)} 23×{closed.get('23', 0)} 22×{closed.get('22', 0)} | "
                    f"open消耗 11×{opened.get('11', 0)} 23×{opened.get('23', 0)} 22×{opened.get('22', 0)} | "
                    f"branch前剩余 11×{pre.get('11', 0)} 23×{pre.get('23', 0)} 22×{pre.get('22', 0)}"
                )
            if chains:
                print(f"    [{tag}已分配结构] {len(chains)}个")
                for idx, ch in enumerate(chains[:chain_limit]):
                    print(f"      [{idx}] {_chain_text(ch)}")
                if len(chains) > chain_limit:
                    print(f"      ... 其余 {len(chains) - chain_limit} 个结构未展开")
            if rem:
                print(
                    f"    [{tag}分支后剩余] 11×{rem.get('11', 0)} 23×{rem.get('23', 0)} 22×{rem.get('22', 0)} "
                    f"24×{rem.get('24', 0)} 25×{rem.get('25', 0)}"
                )

        def _classify_24_like(node: _NodeV3) -> Optional[str]:
            aro = set(int(x) for x in SU_AROMATIC)
            if int(node.su_type) not in (14, 24):
                return None
            hop1 = set(self._current_neighbor_types(node, tmp_nodes))
            has_aro = any(h in aro for h in hop1)
            has_22 = 22 in hop1
            if has_aro and not has_22:
                return '24_A'
            if has_aro and has_22:
                return '24_B'
            if (not has_aro) and (not has_22):
                return '24_C'
            return '24_D'

        def _pick_24_node_for_conversion(target: int) -> Optional[Tuple[_NodeV3, str]]:
            buckets: Dict[str, List[_NodeV3]] = {'24_A': [], '24_B': [], '24_C': [], '24_D': []}
            for n in tmp_nodes:
                label = _classify_24_like(n)
                if label is None:
                    continue
                buckets[label].append(n)
            total_24_like = sum(len(v) for v in buckets.values())
            if int(total_24_like) <= 2:
                return None
            for label in ['24_B', '24_D', '24_A', '24_C']:
                native = [n for n in buckets[label] if int(n.su_type) == 24]
                if native:
                    return native[0], label
                derived = [n for n in buckets[label] if int(n.su_type) == 14]
                if derived:
                    return derived[0], label
            return None

        def _can_convert_25_to_24(min_ratio: float = 0.01) -> bool:
            try:
                keep_25 = max(1, int(math.ceil(float(min_ratio) * float(_ali_total(H_work)))))
                return int(H_work[25].item()) > int(keep_25)
            except Exception:
                return int(H_work[25].item()) > 1

        def _convert_node(node: _NodeV3, dst_su: int) -> None:
            src_su = int(node.su_type)
            H_work[src_su] -= 1
            node.su_type = int(dst_su)
            H_work[int(dst_su)] += 1

        def _apply_2x13_to_23_12() -> bool:
            picked = [n for n in tmp_nodes if int(n.su_type) == 13]
            picked.sort(key=lambda n: int(n.global_id))
            if len(picked) < 2:
                return False
            _convert_node(picked[0], 23)
            _convert_node(picked[1], 12)
            return True
            
        allocator = FlexAllocator(nodes=tmp_nodes)

        # ==========================================================
        # Step 0: SU 25 优先分配
        # ==========================================================
        print(f"\n  [Step 0] SU 25 优先分配")
        s0_iter = 0
        while s0_iter < max_steps:
            s0_iter += 1
            res_25 = allocator.evaluate_su25_only(tmp_nodes)
            print(
                f"    [Step 0诊断] shortage={res_25.get('shortage_type', 'none')} "
                f"req22={res_25.get('req_22', 0)} req11={res_25.get('req_11', 0)} req23={res_25.get('req_23', 0)}"
            )
            _print_alloc_snapshot('Step 0', res_25)
            
            if res_25['ok']:
                print("    [Step 0] 25号分配通过")
                break
                
            shortage = res_25['shortage_type']
            req_22 = res_25.get('req_22', 0)
            req_11 = res_25.get('req_11', 0)
            req_23 = res_25.get('req_23', 0)
            
            op = ''
            if shortage == '22_shortage':
                if int(req_22) < 3:
                    for n in tmp_nodes:
                        if n.su_type == 23:
                            _convert_node(n, 22)
                            op = 'S0_23->22'
                            break
                else:
                    if _can_convert_25_to_24(min_ratio=0.01):
                        for n in tmp_nodes:
                            if n.su_type == 25:
                                _convert_node(n, 24)
                                op = 'S0_25->24'
                                break
            elif shortage == '11_shortage' or req_11 > 0:
                for src in (12, 13):
                    for n in tmp_nodes:
                        if int(n.su_type) == int(src):
                            _convert_node(n, 11)
                            op = f'S0_{src}->11'
                            break
                    if op:
                        break
            elif shortage == '23_shortage' or req_23 > 0:
                if _apply_2x13_to_23_12():
                    op = 'S0_2x13->23+12'
            
            if not op:
                print(f"    [Step 0] 无法处理短缺: {shortage}")
                break
                
            _log_move(op, 'S0')
            moves[-1]['diagnostic_before'] = {
                'shortage_type': str(shortage),
                'req_22': int(res_25.get('req_22', 0)),
                'req_11': int(res_25.get('req_11', 0)),
                'req_23': int(res_25.get('req_23', 0)),
            }
            h_ops, rot_idx = self._h_rotation_adjust(tmp_nodes, H_work, _h_ratio, rot_idx)
            if h_ops:
                print(f"      H调整: {' + '.join(h_ops)}")
                moves[-1]['h_ops'] = list(h_ops)
            moves[-1]['h_ratio_after_h_adjust'] = float(_h_ratio(H_work))
            self._refresh_node_counters(tmp_nodes)
                
            allocator = FlexAllocator(nodes=tmp_nodes)

        # ==========================================================
        # Step 1: SU 24 分支全面分配
        # ==========================================================
        print(f"\n  [Step 1] SU 24 分支全面分配")
        s1_iter = 0
        toggle_11 = 0
        
        while s1_iter < max_steps:
            s1_iter += 1
            res_24 = allocator.evaluate_su24_branches(tmp_nodes)
            print(
                f"    [Step 1诊断] shortage={res_24.get('shortage_type', 'none')} "
                f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} req23={res_24.get('req_23', 0)}"
            )
            _print_alloc_snapshot('Step 1', res_24)
            
            if res_24['ok']:
                print("    [Step 1] 24号分支分配通过")
                break
                
            shortage = res_24['shortage_type']
            req_22 = res_24.get('req_22', 0)
            req_11 = res_24.get('req_11', 0)
            req_23 = res_24.get('req_23', 0)
            op = ''
            
            if shortage == '22_shortage':
                if _can_convert_25_to_24(min_ratio=0.01):
                    for n in tmp_nodes:
                        if int(n.su_type) == 25:
                            _convert_node(n, 24)
                            op = 'S1_25->24'
                            break
                elif int(req_22) < 4:
                    for n in tmp_nodes:
                        if int(n.su_type) == 23:
                            _convert_node(n, 22)
                            op = 'S1_23->22'
                            break
                elif int(req_22) <= 8:
                    picked = None if bool(tail_bias.get('prefer_preserve_24', False)) else _pick_24_node_for_conversion(22)
                    if picked:
                        chosen_node, p = picked
                        _convert_node(chosen_node, 22)
                        op = f'S1_24({p})->22'
                    else:
                        for n in tmp_nodes:
                            if int(n.su_type) == 23:
                                _convert_node(n, 22)
                                op = 'S1_fallback_23->22'
                                break
                else:
                    picked = None if bool(tail_bias.get('prefer_preserve_24', False)) else _pick_24_node_for_conversion(23)
                    if picked:
                        chosen_node, p = picked
                        _convert_node(chosen_node, 23)
                        op = f'S1_24({p})->23'
                    else:
                        for n in tmp_nodes:
                            if int(n.su_type) == 23:
                                _convert_node(n, 22)
                                op = 'S1_fallback_23->22'
                                break
                        
            elif shortage == '11_shortage' or req_11 > 0:
                strategies = [12, 13] if toggle_11 % 2 == 0 else [13, 12]
                for src in strategies:
                    if int(H_work[src].item()) > 0:
                        for n in tmp_nodes:
                            if n.su_type == src:
                                _convert_node(n, 11)
                                op = f'S1_{src}->11'
                                break
                        if op: break
                toggle_11 += 1
                
            elif shortage == '23_shortage' or req_23 > 0:
                if _apply_2x13_to_23_12():
                    op = 'S1_2x13->23+12'
            
            if not op:
                print(f"    [Step 1] 无法处理短缺: {shortage}")
                break
                
            _log_move(op, 'S1')
            moves[-1]['diagnostic_before'] = {
                'shortage_type': str(shortage),
                'req_22': int(res_24.get('req_22', 0)),
                'req_11': int(res_24.get('req_11', 0)),
                'req_23': int(res_24.get('req_23', 0)),
            }
            h_ops, rot_idx = self._h_rotation_adjust(tmp_nodes, H_work, _h_ratio, rot_idx)
            if h_ops:
                print(f"      H调整: {' + '.join(h_ops)}")
                moves[-1]['h_ops'] = list(h_ops)
            moves[-1]['h_ratio_after_h_adjust'] = float(_h_ratio(H_work))
            self._refresh_node_counters(tmp_nodes)
                
            allocator = FlexAllocator(nodes=tmp_nodes)

        self._h_rotation_state = int(rot_idx)
        self._refresh_node_counters(tmp_nodes)
        for i, tn in enumerate(tmp_nodes):
            nodes[i].su_type = tn.su_type
            nodes[i].hop1_su = Counter(tn.hop1_su)
            nodes[i].hop2_su = Counter(tn.hop2_su)
            
        if not res_24['ok']:
            print(
                f"  [Skeleton-Alloc] 分支仍未全部分配: "
                f"unallocated_branch={res_24.get('unallocated_branch', 0)} "
                f"req22={res_24.get('req_22', 0)} req11={res_24.get('req_11', 0)} req23={res_24.get('req_23', 0)}"
            )
        print(f"  [Skeleton-Alloc] 最终H偏差: {_h_ratio(H_work)*100:.2f}%")
        return H_work, moves, {
            'n_moves': len(moves),
            'ok': res_24['ok'],
            'final_h_ratio': float(_h_ratio(H_work)),
            'records': moves,
            'final_diag': res_24,
            'phase': 'branch',
            'final_scenario': 'ok' if bool(res_24['ok']) else 'branch_not_ok',
        }

    def _adjust_skeleton_extra_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        max_steps: int = 45,
        nodes: Optional[List[_NodeV3]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        print(f"\n  [Step 2] EXTRA 柔性链/侧链资源修复")
        if nodes is None:
            print("    [Step 2] 缺少 nodes，无法进行 extra 资源评估")
            return H, [], {'n_moves': 0, 'ok': False, 'phase': 'extra', 'reason': 'missing_nodes'}

        import copy
        H_work = H.cpu().clone()
        self.E_target = E_target
        tmp_nodes = copy.deepcopy(nodes)
        self._refresh_node_counters(tmp_nodes)
        _current_h, _h_ratio, _, _ = self._make_h_helpers()
        moves: List[Dict[str, Any]] = []
        strict_flex_ratio = float(kwargs.get('extra_flexible_ratio', 0.80))
        strict_flex_lower_extra = int(kwargs.get('extra_flexible_lower_extra', 1))
        relaxed_flex_ratio = float(kwargs.get('extra_relaxed_flexible_ratio', 0.82))
        relaxed_flex_lower_extra = int(kwargs.get('extra_relaxed_lower_extra', 0))

        def _log_extra_move(op_desc: str, diag: Dict[str, Any]):
            moves.append({
                'op': op_desc,
                'stage': 'S2',
                'h_ratio_before_h_adjust': float(_h_ratio(H_work)),
                'diagnostic_before': {
                    'reason': str(diag.get('reason', 'ok')),
                    'evaluation_mode': str(diag.get('evaluation_mode', 'strict')),
                    'cluster_count': int(diag.get('cluster_count', 0)),
                    'flexible_bridge_count': int(diag.get('flexible_bridge_count', 0)),
                    'flexible_bridge_min': int(diag.get('flexible_bridge_min', 0)),
                    'flexible_bridge_limit': int(diag.get('flexible_bridge_limit', 0)),
                    'side_to_22_count': int(diag.get('side_to_22_count', 0)),
                },
            })
            print(f"    -> {op_desc} (H偏差: {_h_ratio(H_work)*100:.1f}%)")

        def _select_nodes(src_type: int, count: int) -> List[_NodeV3]:
            picked = [n for n in tmp_nodes if int(n.su_type) == int(src_type)]
            picked.sort(key=lambda n: int(n.global_id))
            return picked[:count] if len(picked) >= count else []

        def _apply_bulk_convert(src_type: int, dst_types: List[int]) -> bool:
            picked = _select_nodes(src_type, len(dst_types))
            if len(picked) != len(dst_types):
                return False
            for n, dst in zip(picked, dst_types):
                H_work[int(n.su_type)] -= 1
                n.su_type = int(dst)
                H_work[int(dst)] += 1
            return True

        def _h_delta(src_type: int, dst_type: int) -> float:
            e_su_cpu = self.E_SU.cpu()
            return float(e_su_cpu[int(dst_type), 1].item() - e_su_cpu[int(src_type), 1].item())

        def _apply_combo_17_18() -> bool:
            if int(H_work[23].item()) >= 2 and int(H_work[11].item()) >= 3:
                _apply_bulk_convert(23, [17, 18])
                _apply_bulk_convert(11, [13, 13, 13])
                return True
            return False

        def _apply_combo_2x15_balance() -> bool:
            if int(H_work[23].item()) >= 2 and int(H_work[11].item()) >= 2:
                _apply_bulk_convert(23, [15, 15])
                _apply_bulk_convert(11, [13, 13])
                return True
            return False

        def _apply_combo_2x15_with_22() -> bool:
            if int(H_work[23].item()) >= 3 and int(H_work[11].item()) >= 1:
                _apply_bulk_convert(23, [15, 15])
                _apply_bulk_convert(11, [13])
                _apply_bulk_convert(23, [22])
                return True
            return False

        def _apply_combo_11_23_to_2x13() -> bool:
            if int(H_work[11].item()) >= 1 and int(H_work[23].item()) >= 1:
                _apply_bulk_convert(11, [13])
                _apply_bulk_convert(23, [13])
                return True
            return False

        def _apply_combo_2x23_to_13_22() -> bool:
            if int(H_work[23].item()) >= 2:
                _apply_bulk_convert(23, [13, 22])
                return True
            return False

        def _apply_2x13_to_23_12() -> bool:
            if int(H_work[13].item()) >= 2 and _can_increase_aliphatic(1, res_extra):
                _apply_bulk_convert(13, [23, 12])
                return True
            return False

        def _apply_rigid10_rebalance() -> Optional[str]:
            if int(H_work[10].item()) <= 0:
                return None
            order = [11, 12, 13]
            rot = int(getattr(self, '_rigid10_rotation_state', 0))
            ordered = order[rot % len(order):] + order[:rot % len(order)]
            for dst in ordered:
                if int(dst) == 12 and not self._can_increase_su12(H_work, inc12=1, dec13=0):
                    continue
                _apply_bulk_convert(10, [int(dst)])
                self._rigid10_rotation_state = int(rot + 1)
                return f'S2_10->{int(dst)}'
            return None

        def _aromatic_total() -> int:
            return int(sum(int(H_work[i].item()) for i in range(5, 14)))

        def _aliphatic_total_now() -> int:
            return int(sum(int(H_work[i].item()) for i in range(19, 26)))

        def _can_reduce_aliphatic(by_count: int, diag: Dict[str, Any]) -> bool:
            try:
                current = int(_aliphatic_total_now())
                lower = int(diag.get('aliphatic_min_total', 0))
                return int(current - int(by_count)) >= int(lower)
            except Exception:
                return True

        def _can_increase_aliphatic(by_count: int, diag: Dict[str, Any]) -> bool:
            try:
                current = int(_aliphatic_total_now())
                upper = int(diag.get('aliphatic_max_total', 10**9))
                return int(current + int(by_count)) <= int(upper)
            except Exception:
                return True

        def _evaluate_balance(relaxed: bool = False) -> Dict[str, Any]:
            ratio = float(relaxed_flex_ratio if relaxed else strict_flex_ratio)
            lower_extra = int(relaxed_flex_lower_extra if relaxed else strict_flex_lower_extra)
            diag = self._evaluate_full_allocation_balance(
                tmp_nodes,
                flex_ratio=float(ratio),
                flex_lower_extra=int(lower_extra),
                S_target=S_target,
                E_target=E_target,
            )
            diag['evaluation_mode'] = 'relaxed' if bool(relaxed) else 'strict'
            return diag

        eval_relaxed_mode = False
        res_extra = _evaluate_balance(relaxed=bool(eval_relaxed_mode))
        cycle_idx = 0
        fallback_11_cycle = 0
        max_reentry_rounds = max(0, int(kwargs.get('extra_refill_reentry_rounds', 3)))
        reentry_round = 0
        step2_initial_dumped = False
        step2_final_dumped = False

        def _has_h_headroom(src_type: int, dst_type: int, max_ratio: float = 0.04) -> bool:
            try:
                target_H = float(self.E_target[1].item()) if self.E_target is not None else 0.0
            except Exception:
                target_H = 0.0
            if target_H <= 0.0:
                return True
            delta_h = float(_h_delta(src_type, dst_type))
            if delta_h <= 0.0:
                return True
            curr_h = float(_current_h(H_work))
            h_cap = float(target_H) * (1.0 + float(max_ratio))
            return float(curr_h + delta_h) <= float(h_cap) + 1e-9

        def _h_move_within_limit(src_type: int, dst_type: int, max_ratio: float = 0.04) -> bool:
            try:
                target_H = float(self.E_target[1].item()) if self.E_target is not None else 0.0
            except Exception:
                target_H = 0.0
            if target_H <= 0.0:
                return True
            curr_h = float(_current_h(H_work))
            delta_h = float(_h_delta(src_type, dst_type))
            next_h = float(curr_h + delta_h)
            h_lo = float(target_H) * (1.0 - float(max_ratio))
            h_hi = float(target_H) * (1.0 + float(max_ratio))
            return float(h_lo) - 1e-9 <= float(next_h) <= float(h_hi) + 1e-9

        while True:
            step = 0
            while step < max_steps:
                print(
                    f"    [Step 2诊断] reason={res_extra.get('reason', 'ok')} "
                    f"mode={res_extra.get('evaluation_mode', 'strict')} "
                    f"clusters={res_extra.get('cluster_count', 0)} "
                    f"rigid10={res_extra.get('rigid_pairs', 0)} "
                    f"flex={res_extra.get('flexible_bridge_count', 0)}/"
                    f"[{res_extra.get('flexible_bridge_min', 0)},{res_extra.get('flexible_bridge_limit', 0)}] "
                    f"side22={res_extra.get('side_to_22_count', 0)} "
                    f"ali={res_extra.get('aliphatic_total', 0)}/"
                    f"[{res_extra.get('aliphatic_min_total', 0)},{res_extra.get('aliphatic_max_total', 0)}]"
                )
                alloc_res_now = res_extra.get('allocation_result', None)
                if alloc_res_now is not None and not bool(step2_initial_dumped):
                    self._print_allocation_details(alloc_res_now, header="Step 2 当前完整资源分配结果")
                    step2_initial_dumped = True
                if res_extra.get('ok'):
                    if alloc_res_now is not None and not bool(step2_final_dumped):
                        self._print_allocation_details(alloc_res_now, header="Step 2 通过时完整资源分配结果")
                        step2_final_dumped = True
                    print("    [Step 2] 完整资源分配评估通过")
                    break

                aromatic_total = max(1, _aromatic_total())
                op = ''
                rigid_excess = 'rigid_excess' in str(res_extra.get('reason', ''))
                flex_excess = 'flex_excess' in str(res_extra.get('reason', ''))
                flex_short = 'flex_shortage' in str(res_extra.get('reason', ''))
                aliphatic_excess = 'aliphatic_excess' in str(res_extra.get('reason', ''))
                req11 = int(res_extra.get('required_extra_11', 0))
                req22 = int(res_extra.get('required_extra_22', 0))
                req23 = int(res_extra.get('required_extra_23', 0))
                branch_short = bool(int(res_extra.get('unallocated_branch', 0)) > 0 or int(req11) > 0 or int(req22) > 0 or int(req23) > 0)

                if not op and branch_short and int(req11) > 0:
                    if _apply_bulk_convert(12, [11]):
                        op = 'S2_req11_12->11'
                    elif _h_move_within_limit(13, 11, max_ratio=0.04) and _apply_bulk_convert(13, [11]):
                        op = 'S2_req11_13->11'

                if not op and branch_short and int(req22) > 0:
                    if _apply_bulk_convert(23, [22]):
                        op = 'S2_req22_23->22'

                if not op and branch_short and int(req23) > 0:
                    if _can_increase_aliphatic(1, res_extra) and _has_h_headroom(13, 23, max_ratio=0.04) and _apply_bulk_convert(13, [23]):
                        op = 'S2_req23_13->23'
                    elif _can_increase_aliphatic(1, res_extra) and _has_h_headroom(12, 23, max_ratio=0.04) and _apply_bulk_convert(12, [23]):
                        op = 'S2_req23_12->23'

                rigid_op = _apply_rigid10_rebalance() if (not op and rigid_excess) else None
                if rigid_op:
                    op = str(rigid_op)
                elif aliphatic_excess and int(H_work[23].item()) > 0:
                    _apply_bulk_convert(23, [13])
                    op = 'S2_23->13_cap'
                elif flex_excess and not bool(eval_relaxed_mode):
                    next_1718 = int(H_work[17].item()) + int(H_work[18].item()) + 2
                    if float(next_1718) <= 0.03 * float(aromatic_total) and _can_reduce_aliphatic(2, res_extra) and _apply_combo_17_18():
                        op = 'S2_2x23+3x11->17+18+3x13'
                    else:
                        next_141516 = int(H_work[14].item()) + int(H_work[15].item()) + int(H_work[16].item()) + 2
                        if float(next_141516) <= 0.06 * float(aromatic_total):
                            if _can_reduce_aliphatic(2, res_extra) and _apply_combo_2x15_balance():
                                op = 'S2_2x23+2x11->2x15+2x13'
                            elif _can_reduce_aliphatic(2, res_extra) and _apply_combo_2x15_with_22():
                                op = 'S2_2x23->2x15, 11->13, 23->22'

                if not op and flex_excess and not bool(eval_relaxed_mode):
                    mode = cycle_idx % 3
                    if mode in (0, 1):
                        if _can_reduce_aliphatic(1, res_extra) and _apply_combo_11_23_to_2x13():
                            op = 'S2_cycle_11+23->2x13'
                        elif _can_reduce_aliphatic(1, res_extra) and _apply_combo_2x23_to_13_22():
                            op = 'S2_cycle_2x23->13+22'
                    else:
                        if _can_reduce_aliphatic(1, res_extra) and _apply_combo_2x23_to_13_22():
                            op = 'S2_cycle_2x23->13+22'
                        elif _can_reduce_aliphatic(1, res_extra) and _apply_combo_11_23_to_2x13():
                            op = 'S2_cycle_11+23->2x13'
                    cycle_idx += 1

                if not op and flex_short and not bool(eval_relaxed_mode):
                    if _apply_2x13_to_23_12():
                        op = 'S2_flexlow_2x13->23+12'

                # 当23已经触底（受脂肪碳最低数量约束）但 flex 仍不满足时，
                # 转而通过降低11号端点数来压缩柔性链数量。
                if not op and flex_excess and not _can_reduce_aliphatic(1, res_extra):
                    eval_relaxed_mode = True
                    mode = fallback_11_cycle % 2
                    if mode == 0:
                        if _has_h_headroom(11, 13, max_ratio=0.04) and _apply_bulk_convert(11, [13]):
                            op = 'S2_flexcap_11->13'
                    else:
                        if _apply_bulk_convert(11, [12]):
                            op = 'S2_flexcap_11->12'
                    fallback_11_cycle += 1

                if not op:
                    print("    [Step 2] 没有可执行的11/23调整组合，停止")
                    break

                step += 1
                _log_extra_move(op, res_extra)
                moves[-1]['h_ratio_after_h_adjust'] = float(_h_ratio(H_work))
                self._refresh_node_counters(tmp_nodes)
                res_extra = _evaluate_balance(relaxed=bool(eval_relaxed_mode))

            if not bool(res_extra.get('ok', False)):
                break

            target_H = 0.0
            try:
                if self.E_target is not None:
                    target_H = float(self.E_target[1].item())
            except Exception:
                target_H = 0.0

            current_h_ratio = float(_h_ratio(H_work))
            if target_H > 0.0 and current_h_ratio <= 0.04:
                # Refill may increase H by at most +2% relative to the current ratio,
                # but the absolute final cap must not exceed +4%.
                # Alternate 13->23 (+1H) and 12->23 (+2H) so the refill order is
                # deterministic and matches the desired gradual H increase pattern.
                refill_cap_ratio = min(current_h_ratio + 0.02, 0.04)
                if refill_cap_ratio > current_h_ratio + 1e-9:
                    h_cap = float(target_H) * (1.0 + refill_cap_ratio)
                    refill_order = [
                        (13, 23, 'S2_hcap_13->23'),
                        (12, 23, 'S2_hcap_12->23'),
                    ]
                    refill_idx = 0
                    stalled = 0
                    while stalled < len(refill_order):
                        src_type, dst_type, op = refill_order[refill_idx % len(refill_order)]
                        refill_idx += 1
                        curr_h = float(_current_h(H_work))
                        delta_h = float(_h_delta(src_type, dst_type))
                        if delta_h <= 0.0:
                            stalled += 1
                            continue
                        if curr_h + delta_h > h_cap + 1e-9:
                            stalled += 1
                            continue
                        if not _can_increase_aliphatic(1, res_extra):
                            stalled += 1
                            continue
                        if not _apply_bulk_convert(src_type, [dst_type]):
                            stalled += 1
                            continue
                        _log_extra_move(op, res_extra)
                        moves[-1]['h_ratio_after_h_adjust'] = float(_h_ratio(H_work))
                        self._refresh_node_counters(tmp_nodes)
                        stalled = 0

                    self._refresh_node_counters(tmp_nodes)
                    res_extra = _evaluate_balance(relaxed=bool(eval_relaxed_mode))

            refill_reason = str(res_extra.get('reason', 'ok'))
            if (
                not bool(res_extra.get('ok', False))
                and ('flex_excess' in refill_reason or 'rigid_excess' in refill_reason or 'flex_shortage' in refill_reason)
                and int(reentry_round) < int(max_reentry_rounds)
            ):
                reentry_round += 1
                print(
                    f"    [Step 2] H回填后退化为 {refill_reason}，"
                    f"重新进入extra压缩循环 ({reentry_round}/{max_reentry_rounds})"
                )
                continue
            break

        self._refresh_node_counters(tmp_nodes)
        for i, tn in enumerate(tmp_nodes):
            nodes[i].su_type = tn.su_type
            nodes[i].hop1_su = Counter(tn.hop1_su)
            nodes[i].hop2_su = Counter(tn.hop2_su)

        if not res_extra.get('ok'):
            print(
                f"  [Skeleton-Extra] 仍不满足: "
                f"reason={res_extra.get('reason', 'unknown')} "
                f"rigid10={res_extra.get('rigid_pairs', 0)} "
                f"flex={res_extra.get('flexible_bridge_count', 0)}/"
                f"[{res_extra.get('flexible_bridge_min', 0)},{res_extra.get('flexible_bridge_limit', 0)}] "
                f"side22={res_extra.get('side_to_22_count', 0)} "
                f"ali={res_extra.get('aliphatic_total', 0)}/"
                f"[{res_extra.get('aliphatic_min_total', 0)},{res_extra.get('aliphatic_max_total', 0)}]"
            )
        print(f"  [Skeleton-Extra] 最终H偏差: {_h_ratio(H_work)*100:.2f}%")
        return H_work, moves, {
            'n_moves': len(moves),
            'ok': bool(res_extra.get('ok', False)),
            'final_h_ratio': float(_h_ratio(H_work)),
            'records': moves,
            'final_diag': res_extra,
            'phase': 'extra',
            'relaxed_mode': bool(eval_relaxed_mode),
            'final_scenario': 'ok' if bool(res_extra.get('ok', False)) else str(res_extra.get('reason', 'extra_not_ok')),
        }

    def _adjust_skeleton_by_allocation(
        self,
        H: torch.Tensor,
        E_target: Optional[torch.Tensor],
        S_target: Optional[torch.Tensor] = None,
        ppm: Optional[np.ndarray] = None,
        diff: Optional[np.ndarray] = None,
        max_steps: int = 15,
        nodes: Optional[List[_NodeV3]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        H_input = torch.clamp(H, min=0).long().clone().cpu()
        H_work, branch_moves, branch_meta = self._adjust_skeleton_branch_allocation(
            H=H,
            E_target=E_target,
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
        extra_meta: Dict[str, Any] = {
            'n_moves': 0,
            'ok': False,
            'records': [],
            'phase': 'extra',
            'skipped': True,
            'reason': 'branch_not_ok',
        }

        if bool(branch_meta.get('ok', False)):
            extra_max_steps = int(kwargs.get('extra_max_steps', max(12, int(max_steps) * 3)))
            H_work, extra_moves, extra_meta = self._adjust_skeleton_extra_allocation(
                H=H_work,
                E_target=E_target,
                S_target=S_target,
                max_steps=extra_max_steps,
                nodes=nodes,
                **kwargs,
            )
            all_moves.extend(extra_moves)
            phase_moves['extra'].extend(list(extra_moves))
        else:
            print("  [Skeleton-Alloc] 分支资源分配尚未通过，跳过 extra 阶段")
        H_after_extra = H_work.detach().clone().cpu()

        align_meta: Dict[str, Any] = {
            'applied': False,
            'reason': 'not_run',
        }
        protect_11_count = 0
        try:
            protect_11_count = max(
                int((branch_meta.get('final_diag', {}) or {}).get('req_11', 0)),
                int((extra_meta.get('final_diag', {}) or {}).get('required_extra_11', 0)),
            )
        except Exception:
            protect_11_count = 0
        try:
            H_aligned, align_moves, align_meta = self._apply_aromatic_cluster_alignment(
                nodes,
                H_work,
                protect_11=bool(int(protect_11_count) > 0),
            )
            H_work = H_aligned.detach().clone().cpu()
            if align_moves:
                all_moves.extend(align_moves)
                phase_moves['align'].extend(list(align_moves))
                print(f"  [Skeleton-Align] 12/13 芳香预对齐: {len(align_moves)} 次转换")
        except Exception as e:
            print(f"  [Skeleton-Align] 失败: {e}")
            align_meta = {
                'applied': False,
                'reason': 'error',
                'error': str(e),
            }
        H_after_align = H_work.detach().clone().cpu()

        final_h_ratio = float(branch_meta.get('final_h_ratio', 0.0))
        if 'final_h_ratio' in extra_meta:
            final_h_ratio = float(extra_meta['final_h_ratio'])

        overall_ok = bool(branch_meta.get('ok', False)) and bool(extra_meta.get('ok', False))
        final_diag = extra_meta.get('final_diag') if bool(branch_meta.get('ok', False)) else branch_meta.get('final_diag')
        post_meta: Dict[str, Any] = {}
        recheck_required = False
        post_moves: List[Dict[str, Any]] = []

        try:
            H_post, su22_moves, su22_meta = self._enforce_su22_ratio_and_h(
                H_work,
                E_target,
                enable=bool(kwargs.get('enable_su22_adjust', True)),
                ratio=float(kwargs.get('su22_ratio', 0.1)),
                h_tol=float(kwargs.get('su22_h_tol', 0.03)),
            )
            if su22_moves:
                recheck_required = True
                H_work = H_post
                for mv in su22_moves:
                    tagged = dict(mv)
                    tagged['stage'] = 'skeleton_post'
                    all_moves.append(tagged)
                    post_moves.append(dict(tagged))
            post_meta['su22_meta'] = su22_meta
        except Exception:
            pass

        try:
            if nodes is not None and E_target is not None:
                import copy
                tmp_nodes = copy.deepcopy(nodes)
                self._refresh_node_counters(tmp_nodes)
                self.E_target = E_target
                _, h_ratio_fn, _, _ = self._make_h_helpers()
                rot_ops, rot_idx = self._h_rotation_adjust(tmp_nodes, H_work, h_ratio_fn, int(self._h_rotation_state))
                self._h_rotation_state = int(rot_idx)
                if rot_ops:
                    recheck_required = True
                    for op in rot_ops:
                        tagged = {'stage': 'skeleton_h_rotation', 'op': str(op)}
                        all_moves.append(tagged)
                        post_moves.append(dict(tagged))
                    self._refresh_node_counters(tmp_nodes)
                    for i, tn in enumerate(tmp_nodes):
                        nodes[i].su_type = tn.su_type
                        nodes[i].hop1_su = Counter(tn.hop1_su)
                        nodes[i].hop2_su = Counter(tn.hop2_su)
                post_meta['h_rotation_ops'] = list(rot_ops)
        except Exception:
            pass

        try:
            H_post, final_moves, final_meta = self._apply_final_structure_constraints(H_work)
            if final_moves:
                recheck_required = True
                H_work = H_post
                for mv in final_moves:
                    tagged = dict(mv)
                    tagged['stage'] = 'skeleton_final_constraints'
                    all_moves.append(tagged)
                    post_moves.append(dict(tagged))
            post_meta['final_structure_constraints'] = final_meta
        except Exception:
            pass

        try:
            H_post, hrot_moves, hrot_meta = self._apply_h_rotation_to_counts(H_work, E_target)
            if hrot_moves:
                recheck_required = True
                H_work = H_post
                all_moves.extend(hrot_moves)
                for mv in hrot_moves:
                    post_moves.append(dict(mv))
            post_meta['post_constraint_h_rotation'] = hrot_meta
        except Exception:
            pass

        try:
            synced_post_moves = self._apply_post_moves_to_nodes(nodes, post_moves)
            post_meta['node_sync_post_moves'] = list(synced_post_moves)
        except Exception as e:
            post_meta['node_sync_error'] = str(e)

        phase_moves['post'].extend(list(post_moves))
        H_after_post = H_work.detach().clone().cpu()

        final_alloc_diag = {}
        try:
            strict_balance_diag = self._evaluate_full_allocation_balance(
                nodes,
                flex_ratio=float(kwargs.get('extra_flexible_ratio', 0.80)),
                flex_lower_extra=int(kwargs.get('extra_flexible_lower_extra', 1)),
                S_target=S_target,
                E_target=E_target,
            )
            relaxed_balance_diag = self._evaluate_full_allocation_balance(
                nodes,
                flex_ratio=float(kwargs.get('extra_relaxed_flexible_ratio', 0.82)),
                flex_lower_extra=int(kwargs.get('extra_relaxed_lower_extra', 0)),
                S_target=S_target,
                E_target=E_target,
            )
            balance_diag = relaxed_balance_diag if bool(extra_meta.get('relaxed_mode', False)) else strict_balance_diag
            alloc_res = balance_diag.get('allocation_result', None)
            if alloc_res is not None:
                self._print_allocation_details(alloc_res, header="最终完整资源分配结果")
            final_alloc_diag = {
                'ok': bool(balance_diag.get('ok', False)),
                'reason': str(balance_diag.get('reason', 'unknown')),
                'selected_mode': 'relaxed' if bool(extra_meta.get('relaxed_mode', False)) else 'strict',
                'strict_ok': bool(strict_balance_diag.get('ok', False)),
                'strict_reason': str(strict_balance_diag.get('reason', 'unknown')),
                'relaxed_ok': bool(relaxed_balance_diag.get('ok', False)),
                'relaxed_reason': str(relaxed_balance_diag.get('reason', 'unknown')),
                'cluster_count': int(balance_diag.get('cluster_count', 0)),
                'effective_cluster_count': int(balance_diag.get('effective_cluster_count', 0)),
                'rigid_cluster_count': int(balance_diag.get('rigid_cluster_count', 0)),
                'flexible_bridge_count': int(balance_diag.get('flexible_bridge_count', 0)),
                'flexible_bridge_min': int(balance_diag.get('flexible_bridge_min', 0)),
                'flexible_bridge_limit': int(balance_diag.get('flexible_bridge_limit', 0)),
                'rigid_pairs': int(balance_diag.get('rigid_pairs', 0)),
                'side_to_22_count': int(balance_diag.get('side_to_22_count', 0)),
                'aliphatic_total': int(balance_diag.get('aliphatic_total', 0)),
                'aliphatic_min_total': int(balance_diag.get('aliphatic_min_total', 0)),
                'aliphatic_max_total': int(balance_diag.get('aliphatic_max_total', 0)),
                'unallocated_bridge': int(balance_diag.get('unallocated_bridge', 0)),
                'unallocated_branch': int(balance_diag.get('unallocated_branch', 0)),
                'required_extra_11': int(balance_diag.get('required_extra_11', 0)),
                'required_extra_22': int(balance_diag.get('required_extra_22', 0)),
                'required_extra_23': int(balance_diag.get('required_extra_23', 0)),
                'remaining_11': int((balance_diag.get('remaining', {}) or {}).get('11', 0)),
                'remaining_22': int((balance_diag.get('remaining', {}) or {}).get('22', 0)),
                'remaining_23': int((balance_diag.get('remaining', {}) or {}).get('23', 0)),
                'remaining_24': int((balance_diag.get('remaining', {}) or {}).get('24', 0)),
                'remaining_25': int((balance_diag.get('remaining', {}) or {}).get('25', 0)),
                'allocation_details': dict(balance_diag.get('allocation_details', {}) or {}),
                'cluster_meta': dict(balance_diag.get('cluster_meta', {}) or {}),
            }
        except Exception as e:
            print(f"  [Skeleton-Alloc] 最终完整资源分配输出失败: {e}")

        try:
            if E_target is not None:
                self.E_target = E_target.detach().cpu() if hasattr(E_target, 'detach') else E_target
                _, h_ratio_fn, _, _ = self._make_h_helpers()
                final_h_ratio = float(h_ratio_fn(H_work.detach().cpu()))
        except Exception:
            pass

        overall_ok = bool(overall_ok) and (not bool(recheck_required)) and bool(final_alloc_diag.get('ok', True))
        return H_work, all_moves, {
            'n_moves': len(all_moves),
            'ok': overall_ok,
            'branch_ok': bool(branch_meta.get('ok', False)),
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
            'final_scenario': (
                'ok'
                if overall_ok else (
                    'branch_not_ok'
                    if not bool(branch_meta.get('ok', False))
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
                       stage: str = 'carbonyl',
                       **kwargs) -> Tuple[torch.Tensor, List[Dict], Dict]:
        """
        按阶段执行单个类型的SU调整，并执行完整的Layer0修正
        
        Args:
            H: SU直方图 [33]
            ppm: PPM轴
            diff: 差谱 (target - reconstructed)
            E_target: 目标元素组成 [6]
            S_target: 目标谱图（用于碳骨架修正）
            stage: 调整阶段，可选 'carbonyl', 'su9', 'ether', 'amine', 'thioether', 'halogen'
            **kwargs: 各调整方法的额外参数
        
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
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
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
                min_keep_25=kwargs.get('min_keep_25', 0),
                carbonyl_couple=kwargs.get('carbonyl_couple', True),
                h_tolerance=kwargs.get('h_tolerance', kwargs.get('su22_h_tol', 0.04)),
            )
        elif stage == 'carbonyl':
            H_work, moves, meta = self.adjust_carbonyl_by_difference(
                H_work, ppm, diff,
                window_12=kwargs.get('window_12', 5.0),
                window_3=kwargs.get('window_3', 10.0),
                score_rel_threshold=kwargs.get('score_rel_threshold', 0.15),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 1)
            )
        elif stage == 'su9':
            H_work, moves, meta = self.adjust_su9_by_difference(
                H_work, ppm, diff,
                window=kwargs.get('window', 10.0),
                score_rel_threshold=kwargs.get('score_rel_threshold', 0.15),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 1)
            )
        elif stage == 'ether':
            H_work, moves, meta = self.adjust_ether_519_by_difference(
                H_work, ppm, diff,
                window_5=kwargs.get('window_5', 3.0),
                window_19=kwargs.get('window_19', 3.0),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 1),
                reserved_19=kwargs.get('reserved_19', 0)
            )
        elif stage == 'amine':
            H_work, moves, meta = self.adjust_amine_620_by_difference(
                H_work, ppm, diff,
                window_6=kwargs.get('window_6', 3.0),
                window_20=kwargs.get('window_20', 3.0),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 0)
            )
        elif stage == 'thioether':
            H_work, moves, meta = self.adjust_thioether_719_by_difference(
                H_work, ppm, diff,
                window_7=kwargs.get('window_7', 3.0),
                window_19=kwargs.get('window_19', 3.0),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 0)
            )
        elif stage == 'halogen':
            H_work, moves, meta = self.adjust_halogen_821_by_difference(
                H_work, ppm, diff,
                window_8=kwargs.get('window_8', 3.0),
                window_21=kwargs.get('window_21', 3.0),
                peak_rel_threshold=kwargs.get('peak_rel_threshold', 0.01),
                max_moves=kwargs.get('max_moves', 5),
                min_keep=kwargs.get('min_keep', 0)
            )
        elif stage == 'skeleton':
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
                **skeleton_kwargs,
            )
            if meta.get('ok'):
                status = '资源分配全部通过'
            elif meta.get('branch_ok', False):
                status = '分支通过但柔性/侧链阶段未通过'
            else:
                status = '分支阶段未通过'
            self._print_hist_change_summary(stage.upper(), H_input, torch.clamp(H_work, min=0).long().cpu())
            self._print_move_summary(stage.upper(), moves)
            phase_hists = meta.get('phase_hists', {}) or {}
            phase_moves = meta.get('phase_moves', {}) or {}
            if phase_hists:
                for label, before_key, after_key in (
                    ('SKELETON-BRANCH', 'input', 'after_branch'),
                    ('SKELETON-EXTRA', 'after_branch', 'after_extra'),
                    ('SKELETON-ALIGN', 'after_extra', 'after_align'),
                    ('SKELETON-POST', 'after_align', 'after_post'),
                ):
                    before = phase_hists.get(before_key)
                    after = phase_hists.get(after_key)
                    if before is None or after is None:
                        continue
                    self._print_hist_change_summary(str(label), before, after)
            for label, key in (
                ('SKELETON-BRANCH', 'branch'),
                ('SKELETON-EXTRA', 'extra'),
                ('SKELETON-ALIGN', 'align'),
                ('SKELETON-POST', 'post'),
            ):
                self._print_move_summary(str(label), list(phase_moves.get(key, []) or []))
            print(f"\n{stage.upper()}候选调整完成: 共{len(moves)}次变更, {status}")
            return H_work, moves, meta

        else:
            print(f"  [警告] 未知的调整阶段: {stage}")
            return H_work, moves, meta

        try:
            H_work, hrot_moves, hrot_meta = self._apply_h_rotation_to_counts(H_work, E_target)
            if hrot_moves:
                moves.extend(hrot_moves)
            meta['h_rotation_meta'] = hrot_meta
        except Exception:
            pass
        
        self._print_hist_change_summary(stage.upper(), H_input, torch.clamp(H_work, min=0).long().cpu())
        self._print_move_summary(stage.upper(), moves)
        print(f"\n{stage.upper()}阶段调整完成: 共{len(moves)}次变更")
        return H_work, moves, meta
    
