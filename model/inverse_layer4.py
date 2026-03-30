import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
from collections import Counter
import math
import io
from contextlib import redirect_stdout

from .inverse_common import SU_ALIPHATIC, SU_AROMATIC, SU_DEFS, E_SU, PPM_AXIS, NUM_SU_TYPES, _NodeV3
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
    
    每次调整后都执行Layer0完整修正流程。
    """
    
    def __init__(self, 
                 device: torch.device = None,
                 layer0_estimator: Optional[Layer0Estimator] = None,
                 su_hop1_ranges_path: Optional[str] = None):
        self.device = device or torch.device('cpu')
        self.layer0 = layer0_estimator
        self.E_SU = E_SU.to(self.device)
        
        # 加载 hop1 NMR范围数据
        if su_hop1_ranges_path is None:
            default_path = Path(__file__).resolve().parents[1] / 'z_library' / 'su_hop1_nmr_range_filtered.csv'
            self.su_hop1_ranges_path = str(default_path) if default_path.exists() else None
        else:
            self.su_hop1_ranges_path = su_hop1_ranges_path
        self._su_hop1_mu_median_cache = None
        # Persist H-adjust rotation state across repeated skeleton adjustment calls.
        self._h_rotation_state = 0

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

        mu_0 = 167.125
        mu_1 = 174.8
        mu_2 = 169.6
        mu_3 = 195.8

        def _window_stats(mu: float, window: float) -> Dict[str, float]:
            mask = (ppm_arr >= float(mu) - float(window)) & (ppm_arr <= float(mu) + float(window))
            if not bool(mask.any()):
                return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'dom': 0.0, 'abs': 0.0}
            seg = diff_arr[mask]
            if int(seg.size) == 0:
                return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'dom': 0.0, 'abs': 0.0}
            pos = float(np.sum(seg[seg > 0])) if bool((seg > 0).any()) else 0.0
            neg = float(np.sum(seg[seg < 0])) if bool((seg < 0).any()) else 0.0
            net = float(pos + neg)
            dom = float(pos) if float(pos) >= abs(float(neg)) else float(neg)
            abs_sum = float(np.sum(np.abs(seg)))
            return {'pos': pos, 'neg': neg, 'net': net, 'dom': dom, 'abs': abs_sum}

        carbonyl_mask = (ppm_arr >= 160.0) & (ppm_arr <= 240.0)
        carbonyl_abs = float(np.sum(np.abs(diff_arr[carbonyl_mask]))) if bool(carbonyl_mask.any()) else float(np.sum(np.abs(diff_arr)))
        thr = float(score_rel_threshold) * max(1e-9, float(carbonyl_abs))

        s0 = _window_stats(mu_0, float(window_12))
        s1 = _window_stats(mu_1, float(window_12))
        s2 = _window_stats(mu_2, float(window_12))
        s3 = _window_stats(mu_3, float(window_3))

        print(f"  0号@{mu_0:.3f} pos={float(s0['pos']):.3f}, neg={float(s0['neg']):.3f}, net={float(s0['net']):.3f} (固定不调整)")
        print(f"  1号@{mu_1:.3f} pos={float(s1['pos']):.3f}, neg={float(s1['neg']):.3f}, net={float(s1['net']):.3f}")
        print(f"  2号@{mu_2:.3f} pos={float(s2['pos']):.3f}, neg={float(s2['neg']):.3f}, net={float(s2['net']):.3f}")
        print(f"  3号@{mu_3:.3f} pos={float(s3['pos']):.3f}, neg={float(s3['neg']):.3f}, net={float(s3['net']):.3f}")
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

        mu_5 = 154.75
        mu_19 = 66.6

        def _window_score(mu: float, window: float) -> Dict[str, float]:
            mask = (ppm_arr >= float(mu) - float(window)) & (ppm_arr <= float(mu) + float(window))
            if not bool(mask.any()):
                return {'pos': 0.0, 'neg': 0.0, 'score': 0.0}
            seg = diff_arr[mask]
            pos = float(np.sum(seg[seg > 0])) if int(seg.size) > 0 else 0.0
            neg = float(np.sum(seg[seg < 0])) if int(seg.size) > 0 else 0.0
            dom = float(pos) if float(pos) >= abs(float(neg)) else float(neg)
            return {'pos': pos, 'neg': neg, 'score': dom}

        s5 = _window_score(mu_5, float(window_5))
        s19 = _window_score(mu_19, float(window_19))
        score_5 = float(s5.get('score', 0.0))
        score_19 = float(s19.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU5@{mu_5:.2f} score={score_5:.3f} (pos={float(s5.get('pos', 0.0)):.3f}, neg={float(s5.get('neg', 0.0)):.3f})")
        print(f"  SU19@{mu_19:.2f} score={score_19:.3f} (pos={float(s19.get('pos', 0.0)):.3f}, neg={float(s19.get('neg', 0.0)):.3f})")
        print(f"  threshold={thr:.3f} (peak_rel_threshold={float(peak_rel_threshold):.4f}, total_abs={total_abs:.3f})")

        if max(abs(score_5), abs(score_19)) < thr:
            print("  峰强不足，跳过调整")
            return H, [], {
                'scores': {'5': s5, '19': s19},
                'centers': {'5': mu_5, '19': mu_19},
                'threshold': thr,
                'reserved_19': int(reserved_19),
            }

        def _sgn(x: float) -> int:
            if x > 0:
                return 1
            if x < 0:
                return -1
            return 0

        dir_5 = _sgn(score_5)
        dir_19 = _sgn(score_19)

        if dir_5 != 0 and dir_19 != 0 and dir_5 != dir_19:
            inc = 5 if dir_5 > 0 else 19
            dec = 19 if int(inc) == 5 else 5
        else:
            priority = 5 if abs(float(score_5)) >= abs(float(score_19)) else 19
            priority_dir = dir_5 if int(priority) == 5 else dir_19
            if int(priority_dir) >= 0:
                inc = int(priority)
                dec = 19 if int(inc) == 5 else 5
            else:
                dec = int(priority)
                inc = 19 if int(dec) == 5 else 5

        H_new = H.clone()
        moves: List[Dict] = []

        def _effective_19_count(hh: torch.Tensor) -> int:
            return max(0, int(hh[19].item()) - int(reserved_19))

        for _ in range(max(0, int(max_moves))):
            if int(H_new[5].item()) < int(min_keep):
                need_5 = int(min_keep) - int(H_new[5].item())
                for __ in range(max(0, int(need_5))):
                    donor_5 = None
                    if int(H_new[13].item()) > 0:
                        donor_5 = 13
                    elif int(H_new[11].item()) > 0:
                        donor_5 = 11
                    if donor_5 is None:
                        break
                    H_new[donor_5] -= 1
                    H_new[5] += 1
                    moves.append({'op': 'fill_min_keep_5', 'from': donor_5, 'to': 5})

            if _effective_19_count(H_new) < int(min_keep):
                need_19 = int(min_keep) - _effective_19_count(H_new)
                for __ in range(max(0, int(need_19))):
                    if int(H_new[23].item()) <= 0:
                        break
                    H_new[23] -= 1
                    H_new[19] += 1
                    moves.append({'op': 'fill_min_keep_19', 'from': 23, 'to': 19})

            H_try = H_new.clone()
            iter_moves: List[Dict] = []

            ok_inc = True
            ok_dec = True

            if int(inc) == 5:
                donor = None
                if int(H_try[13].item()) > 0:
                    donor = 13
                elif int(H_try[11].item()) > 0:
                    donor = 11
                if donor is None:
                    ok_inc = False
                else:
                    H_try[donor] -= 1
                    H_try[5] += 1
                    iter_moves.append({'op': 'inc_5', 'from': donor, 'to': 5})
            else:
                if int(H_try[23].item()) <= 0:
                    ok_inc = False
                else:
                    H_try[23] -= 1
                    H_try[19] += 1
                    iter_moves.append({'op': 'inc_19', 'from': 23, 'to': 19})

            if int(dec) == 19:
                if _effective_19_count(H_try) <= int(min_keep):
                    ok_dec = False
                else:
                    H_try[19] -= 1
                    H_try[23] += 1
                    iter_moves.append({'op': 'dec_19', 'from': 19, 'to': 23, 'reserved_19': int(reserved_19)})
            else:
                if int(H_try[5].item()) <= int(min_keep):
                    ok_dec = False
                else:
                    H_try[5] -= 1
                    dst = 13 if int(H_try[13].item()) <= int(H_try[11].item()) else 11
                    H_try[dst] += 1
                    iter_moves.append({'op': 'dec_5', 'from': 5, 'to': int(dst)})

            if not (ok_inc and ok_dec):
                break

            H_new = H_try
            moves.extend(iter_moves)

        meta = {
            'scores': {'5': s5, '19': s19},
            'centers': {'5': mu_5, '19': mu_19},
            'threshold': thr,
            'reserved_19': int(reserved_19),
            'direction': {'inc': int(inc), 'dec': int(dec)},
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

        mu_6 = 146.375
        mu_20 = 49.375

        def _window_score(mu: float, window: float) -> Dict[str, float]:
            mask = (ppm_arr >= float(mu) - float(window)) & (ppm_arr <= float(mu) + float(window))
            if not bool(mask.any()):
                return {'pos': 0.0, 'neg': 0.0, 'score': 0.0}
            seg = diff_arr[mask]
            pos = float(np.sum(seg[seg > 0])) if int(seg.size) > 0 else 0.0
            neg = float(np.sum(seg[seg < 0])) if int(seg.size) > 0 else 0.0
            dom = float(pos) if float(pos) >= abs(float(neg)) else float(neg)
            return {'pos': pos, 'neg': neg, 'score': dom}

        s6 = _window_score(mu_6, float(window_6))
        s20 = _window_score(mu_20, float(window_20))
        score_6 = float(s6.get('score', 0.0))
        score_20 = float(s20.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU6@{mu_6:.3f} score={score_6:.3f} (pos={float(s6.get('pos', 0.0)):.3f}, neg={float(s6.get('neg', 0.0)):.3f})")
        print(f"  SU20@{mu_20:.3f} score={score_20:.3f} (pos={float(s20.get('pos', 0.0)):.3f}, neg={float(s20.get('neg', 0.0)):.3f})")
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

        mu_7 = 152.875
        mu_19 = 66.6

        def _window_score(mu: float, window: float) -> Dict[str, float]:
            mask = (ppm_arr >= float(mu) - float(window)) & (ppm_arr <= float(mu) + float(window))
            if not bool(mask.any()):
                return {'pos': 0.0, 'neg': 0.0, 'score': 0.0}
            seg = diff_arr[mask]
            pos = float(np.sum(seg[seg > 0])) if int(seg.size) > 0 else 0.0
            neg = float(np.sum(seg[seg < 0])) if int(seg.size) > 0 else 0.0
            dom = float(pos) if float(pos) >= abs(float(neg)) else float(neg)
            return {'pos': pos, 'neg': neg, 'score': dom}

        s7 = _window_score(mu_7, float(window_7))
        s19 = _window_score(mu_19, float(window_19))
        score_7 = float(s7.get('score', 0.0))
        score_19 = float(s19.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU7@{mu_7:.3f} score={score_7:.3f} (pos={float(s7.get('pos', 0.0)):.3f}, neg={float(s7.get('neg', 0.0)):.3f})")
        print(f"  SU19@{mu_19:.3f} score={score_19:.3f} (pos={float(s19.get('pos', 0.0)):.3f}, neg={float(s19.get('neg', 0.0)):.3f})")
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

        mu_8 = 131.4
        mu_21 = 38.4

        def _window_score(mu: float, window: float) -> Dict[str, float]:
            mask = (ppm_arr >= float(mu) - float(window)) & (ppm_arr <= float(mu) + float(window))
            if not bool(mask.any()):
                return {'pos': 0.0, 'neg': 0.0, 'score': 0.0}
            seg = diff_arr[mask]
            pos = float(np.sum(seg[seg > 0])) if int(seg.size) > 0 else 0.0
            neg = float(np.sum(seg[seg < 0])) if int(seg.size) > 0 else 0.0
            dom = float(pos) if float(pos) >= abs(float(neg)) else float(neg)
            return {'pos': pos, 'neg': neg, 'score': dom}

        s8 = _window_score(mu_8, float(window_8))
        s21 = _window_score(mu_21, float(window_21))
        score_8 = float(s8.get('score', 0.0))
        score_21 = float(s21.get('score', 0.0))

        total_abs = float(np.sum(np.abs(diff_arr)))
        thr = float(peak_rel_threshold) * max(1e-9, total_abs)

        print(f"  SU8@{mu_8:.3f} score={score_8:.3f} (pos={float(s8.get('pos', 0.0)):.3f}, neg={float(s8.get('neg', 0.0)):.3f})")
        print(f"  SU21@{mu_21:.3f} score={score_21:.3f} (pos={float(s21.get('pos', 0.0)):.3f}, neg={float(s21.get('neg', 0.0)):.3f})")
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
    def _h_rotation_adjust(tmp_nodes, H_work, h_ratio_fn, rot_idx):
        ops = []
        failed_steps = 0
        
        while abs(h_ratio_fn(H_work)) > 0.04:
            ratio = h_ratio_fn(H_work)
            step_type = rot_idx % 5
            success = False
            
            if ratio > 0.04:
                if step_type in [0, 2]:
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
                    if float(total_unsat) >= float(min_unsat_pool):
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
        return {
            'cluster_count': int(len(clusters)),
            'bridgehead_info': tuple(int(x) for x in bridgehead_info),
            'converted_13': int(gen.n13),
            'converted_12': int(gen.n12),
            'remaining_12': int(gen.remaining_12),
            'remaining_13': int(gen.remaining_13),
        }

    @staticmethod
    def _compute_flexible_window(cluster_count: int,
                                 rigid_pairs: int,
                                 flex_ratio: float,
                                 flex_lower_extra: int = 1) -> Tuple[int, int, int]:
        cluster_count_i = max(0, int(cluster_count))
        rigid_pairs_i = max(0, int(rigid_pairs))
        if cluster_count_i > 0:
            z_clusters = max(1, int(cluster_count_i - rigid_pairs_i))
        else:
            z_clusters = 0
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
        if meta:
            if 'branch_type' in meta:
                meta_brief.append(f"branch_type={meta['branch_type']}")
            if 'tail_source' in meta:
                meta_brief.append(f"tail={meta['tail_source']}")
            if 'tail_sources' in meta:
                meta_brief.append(f"tails={meta['tail_sources']}")
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

    def _print_allocation_details(self, alloc_res: Any, header: str = "最终完整资源分配结果") -> None:
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

        if S_target is not None and E_target is not None:
            try:
                spec = S_target.detach().cpu().flatten().numpy()
                total_area = float(np.sum(spec) * 0.1)
                ali_area = float(np.sum(spec[:900]) * 0.1)
                x = float(ali_area / total_area) if total_area > 1e-9 else 0.33
                x_pct = float(x) * 100.0
                target_c = float(E_target.detach().cpu().flatten()[0].item())
                aliphatic_min = int(math.ceil(((4.425 + 0.123 * x_pct + 0.00754 * x_pct * x_pct) / 100.0) * target_c))
            except Exception:
                aliphatic_min = 0
        else:
            aliphatic_min = 0

        rigid_ok = (int(rigid_pairs) < int(cluster_count)) if int(cluster_count) > 0 else True
        flex_hi_ok = int(flexible_bridge_count) <= int(flex_upper)
        flex_lo_ok = int(flexible_bridge_count) >= int(flex_lower)
        aliphatic_ok = int(aliphatic_total) >= int(aliphatic_min)

        reasons = []
        if not rigid_ok:
            reasons.append('rigid_excess')
        if not flex_hi_ok:
            reasons.append('flex_excess')
        if not flex_lo_ok:
            reasons.append('flex_shortage')
        if not aliphatic_ok:
            reasons.append('aliphatic_shortage')
        reason = 'ok' if not reasons else '+'.join(reasons)

        return {
            'ok': bool(rigid_ok and flex_hi_ok and flex_lo_ok and aliphatic_ok),
            'reason': str(reason),
            'cluster_count': int(cluster_count),
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
            'cluster_meta': cluster_meta,
            'allocation_result': alloc_res,
            'allocation_details': self._extract_allocation_details(alloc_res),
            'unallocated_bridge': int(getattr(alloc_res, 'unallocated_bridge', 0)),
            'unallocated_branch': int(getattr(alloc_res, 'unallocated_branch', 0)),
            'required_extra_11': int(getattr(alloc_res, 'required_extra_11', 0)),
            'required_extra_22': int(getattr(alloc_res, 'required_extra_22', 0)),
            'required_extra_23': int(getattr(alloc_res, 'required_extra_23', 0)),
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
                    picked = _pick_24_node_for_conversion(22)
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
                    picked = _pick_24_node_for_conversion(23)
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
            if int(H_work[13].item()) >= 2:
                _apply_bulk_convert(13, [23, 12])
                return True
            return False

        def _apply_10_to_12() -> bool:
            if int(H_work[10].item()) <= 0:
                return False
            _apply_bulk_convert(10, [12])
            return True

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
                    f"ali={res_extra.get('aliphatic_total', 0)}/{res_extra.get('aliphatic_min_total', 0)}"
                )
                if res_extra.get('ok'):
                    print("    [Step 2] 完整资源分配评估通过")
                    break

                aromatic_total = max(1, _aromatic_total())
                op = ''
                rigid_excess = 'rigid_excess' in str(res_extra.get('reason', ''))
                flex_excess = 'flex_excess' in str(res_extra.get('reason', ''))
                flex_short = 'flex_shortage' in str(res_extra.get('reason', ''))

                if rigid_excess and _apply_10_to_12():
                    op = 'S2_10->12'
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
                refill_cap_ratio = min(current_h_ratio + 0.02, 0.04)
                if refill_cap_ratio > current_h_ratio + 1e-9:
                    h_cap = float(target_H) * (1.0 + refill_cap_ratio)
                    refill_order = [
                        (13, 23, 'S2_hcap_13->23'),
                        (11, 23, 'S2_hcap_11->23'),
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
                f"ali={res_extra.get('aliphatic_total', 0)}/{res_extra.get('aliphatic_min_total', 0)}"
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
        else:
            print("  [Skeleton-Alloc] 分支资源分配尚未通过，跳过 extra 阶段")

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
                'rigid_cluster_count': int(balance_diag.get('rigid_cluster_count', 0)),
                'flexible_bridge_count': int(balance_diag.get('flexible_bridge_count', 0)),
                'flexible_bridge_min': int(balance_diag.get('flexible_bridge_min', 0)),
                'flexible_bridge_limit': int(balance_diag.get('flexible_bridge_limit', 0)),
                'rigid_pairs': int(balance_diag.get('rigid_pairs', 0)),
                'side_to_22_count': int(balance_diag.get('side_to_22_count', 0)),
                'aliphatic_total': int(balance_diag.get('aliphatic_total', 0)),
                'aliphatic_min_total': int(balance_diag.get('aliphatic_min_total', 0)),
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
            }
        except Exception as e:
            print(f"  [Skeleton-Alloc] 最终完整资源分配输出失败: {e}")

        final_h_ratio = float(branch_meta.get('final_h_ratio', 0.0))
        if 'final_h_ratio' in extra_meta:
            final_h_ratio = float(extra_meta['final_h_ratio'])

        overall_ok = bool(branch_meta.get('ok', False)) and bool(extra_meta.get('ok', False))
        final_diag = extra_meta.get('final_diag') if bool(branch_meta.get('ok', False)) else branch_meta.get('final_diag')
        return H_work, all_moves, {
            'n_moves': len(all_moves),
            'ok': overall_ok,
            'branch_ok': bool(branch_meta.get('ok', False)),
            'extra_ok': bool(extra_meta.get('ok', False)),
            'final_h_ratio': final_h_ratio,
            'records': all_moves,
            'branch_meta': branch_meta,
            'extra_meta': extra_meta,
            'final_diag': final_diag,
            'final_allocation': final_alloc_diag,
            'final_scenario': (
                'ok'
                if overall_ok else (
                    'branch_not_ok'
                    if not bool(branch_meta.get('ok', False))
                    else str(extra_meta.get('reason', 'extra_not_ok'))
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
        
        H_work = H.clone()
        moves = []
        meta = {}
        
        # 1. 执行对应阶段的调整
        if stage == 'carbonyl':
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
            try:
                H_work, final_moves, final_meta = self._apply_final_structure_constraints(H_work)
                if final_moves:
                    moves.extend(final_moves)
                meta['final_structure_constraints'] = final_meta
            except Exception:
                pass
            if meta.get('ok'):
                status = '资源分配全部通过'
            elif meta.get('branch_ok', False):
                status = '分支通过但柔性/侧链阶段未通过'
            else:
                status = '分支阶段未通过'
            print(f"\n{stage.upper()}阶段调整完成: 共{len(moves)}次变更, {status}")
            return H_work, moves, meta

        else:
            print(f"  [警告] 未知的调整阶段: {stage}")
            return H_work, moves, meta
        
        # 2. 执行完整的Layer0修正流程
        if self.layer0 and E_target is not None:
            print(f"\n[Layer0完整修正] {stage}调整后执行")
            
            # O元素修正 (28, 29号)
            H_work = self.layer0._correct_oxygen_O(H_work, E_target)
            
            # 羰基连接修正 (9号)
            H_work = self.layer0._correct_carbonyl_connection(H_work)
            
            # 醚连接修正 (5, 19号)
            H_work, ether_meta = self.layer0._correct_ether_connection(H_work)
            o_base_19 = ether_meta.get('o_base_19', 0)
            
            # 硫醚连接修正 (7, 19号)
            H_work, _ = self.layer0._correct_thioether_connection(H_work, locals().get('o_base_19', 0))
            
            # 氨基连接修正 (6, 20号)
            H_work = self.layer0._correct_amine_connection(H_work)
            
            # 卤素连接修正 (8, 21号)
            H_work = self.layer0._correct_halogen_connection(H_work)
            
            # 最终碳元素守恒修正，避免阶段性连接修正引入C总量漂移
            if S_target is not None:
                H_work = self.layer0._reconcile_carbon_total(H_work, S_target, E_target)
            
            # H元素调整
            H_work = self.layer0._adjust_hydrogen(H_work, E_target)

        try:
            enable_su22_adjust = bool(kwargs.get('enable_su22_adjust', True))
            su22_ratio = float(kwargs.get('su22_ratio', 0.1))
            su22_h_tol = float(kwargs.get('su22_h_tol', 0.03))
            H_work, su22_moves, su22_meta = self._enforce_su22_ratio_and_h(
                H_work,
                E_target,
                enable=bool(enable_su22_adjust),
                ratio=float(su22_ratio),
                h_tol=float(su22_h_tol),
            )
            if su22_moves:
                moves.extend(su22_moves)
            meta['su22_meta'] = su22_meta
        except Exception:
            pass

        try:
            H_work, final_moves, final_meta = self._apply_final_structure_constraints(H_work)
            if final_moves:
                moves.extend(final_moves)
            meta['final_structure_constraints'] = final_meta
        except Exception:
            pass
        
        print(f"\n{stage.upper()}阶段调整完成: 共{len(moves)}次变更")
        return H_work, moves, meta
    
