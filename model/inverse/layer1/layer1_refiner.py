import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict

from ...paths import Z_LIBRARY_DIR
from ..hop1_adjuster import (
    Hop1Adjuster,
    build_motif_usage_from_pairs,
    motif_penalty_alpha,
    multiset_l1_distance,
)
from ...shared.coarse_graph import get_su_ppm_range
from ...shared.inverse_common import (
    HOP1_PORT_COMBINATIONS,
    _NodeV3,
    check_external_connection_requirement,
    required_external_candidates_for_node,
    validate_connection,
)
from .layer1_eval import Layer1NmrEvaluator


class Layer1Refiner:
    def __init__(self,
                 device: str | torch.device,
                 evaluator: Layer1NmrEvaluator):
        self.device = device
        self.evaluator = evaluator
        self._su_common_range_path = str(Z_LIBRARY_DIR / 'su_nmr_common_range_filtered.csv')
        self._su_common_ranges_cache: Optional[Dict[int, Dict[str, float]]] = None

    @staticmethod
    def _capture_hop1_state(nodes: List[_NodeV3]) -> List[Tuple[Counter, List[int], set]]:
        return [
            (
                Counter(getattr(node, 'hop1_su', {}) or {}),
                list(getattr(node, 'hop1_ids', []) or []),
                set(getattr(node, 'fixed_hop1_ids', set()) or set()),
            )
            for node in nodes
        ]

    @staticmethod
    def _restore_hop1_state(nodes: List[_NodeV3], state: List[Tuple[Counter, List[int], set]]) -> None:
        for node, (hop1_su, hop1_ids, fixed_ids) in zip(nodes, state):
            node.hop1_su = Counter(hop1_su)
            node.hop1_ids = list(hop1_ids)
            try:
                node.fixed_hop1_ids = set(fixed_ids)
            except Exception:
                pass

    @staticmethod
    def _required_external_candidates_for_node(node: _NodeV3) -> Optional[List[int]]:
        return required_external_candidates_for_node(node, missing_su19_as_none=True)

    def _partition_aware_external_requirement(self,
                                              node: _NodeV3,
                                              hop1_counter: Counter) -> Tuple[bool, str]:
        required = self._required_external_candidates_for_node(node)
        center_su = int(getattr(node, 'su_type', -1))
        if required is None:
            ext_present = sorted(int(x) for x in hop1_counter.keys() if int(x) in {2, 28, 29, 31})
            if ext_present:
                return False, f"SU19 missing partition but uses fixed anchors {ext_present}"
            return False, "SU19 missing partition"
        if not required:
            return check_external_connection_requirement(center_su, hop1_counter)

        target_count = 1
        try:
            val = getattr(node, 'target_fixed_anchor_count', None)
            if val is not None:
                target_count = max(1, int(val))
        except Exception:
            target_count = 1

        ext_cnt = int(sum(int(v) for k, v in hop1_counter.items() if int(k) in set(int(x) for x in required)))
        if int(ext_cnt) < int(target_count):
            return False, (
                f"SU {center_su} requires external connection to "
                f"{sorted(int(x) for x in required)} count={int(ext_cnt)}/{int(target_count)}"
            )
        if int(ext_cnt) > int(target_count):
            return False, (
                f"SU {center_su} has too many external anchors: "
                f"required={sorted(int(x) for x in required)} count={int(ext_cnt)}/{int(target_count)}"
            )
        return True, ""

    def _get_su_common_ranges(self) -> Dict[int, Dict[str, float]]:
        if isinstance(self._su_common_ranges_cache, dict):
            return self._su_common_ranges_cache
        out: Dict[int, Dict[str, float]] = {}
        try:
            df = pd.read_csv(self._su_common_range_path)
            for _, row in df.iterrows():
                try:
                    su = int(row['center_su_idx'])
                    out[su] = {
                        'mu_median': float(row.get('mu_median', 0.0) or 0.0),
                        'mu_min': float(row.get('mu_common_min', 0.0) or 0.0),
                        'mu_max': float(row.get('mu_common_max', 0.0) or 0.0),
                    }
                except Exception:
                    continue
        except Exception:
            out = {}
        for su_type in (11, 12, 13, 22, 23, 24):
            coarse_lo, coarse_hi = get_su_ppm_range(int(su_type))
            out[int(su_type)] = {
                'mu_median': 0.5 * (float(coarse_lo) + float(coarse_hi)),
                'mu_min': float(coarse_lo),
                'mu_max': float(coarse_hi),
            }
        self._su_common_ranges_cache = out
        return out

    @staticmethod
    def _region_specs() -> List[Tuple[str, float, float, List[int]]]:
        return [
            ('aromatic', 90.0, 160.0, [13, 12, 11, 10, 9, 5, 6, 7, 8]),
            ('carbonyl', 160.0, 240.0, [1, 2, 3, 0]),
            ('unsaturated', 90.0, 160.0, [14, 15, 16, 17, 18, 4]),
            ('aliphatic', 0.0, 90.0, [22, 23, 24, 25, 19, 20, 21]),
        ]

    @staticmethod
    def _ranges_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
        return not (float(hi1) < float(lo2) or float(hi2) < float(lo1))

    @staticmethod
    def _alternative_resource_shortage(nodes_by_su: Dict[int, List[int]],
                                       center_su: int,
                                       hop1_tuple: Tuple[int, ...]) -> Dict[int, int]:
        needed = Counter(int(x) for x in tuple(hop1_tuple))
        shortages: Dict[int, int] = {}
        for su_type, count in needed.items():
            available = int(len(nodes_by_su.get(int(su_type), [])))
            if int(su_type) == int(center_su):
                available = max(0, int(available) - 1)
            if int(count) > int(available):
                shortages[int(su_type)] = int(count) - int(available)
        return shortages

    @staticmethod
    def _motif_concentration(usage: Counter) -> float:
        return float(sum(
            float(motif_penalty_alpha(int(center_su))) * float(count * max(0, count - 1))
            for (center_su, _hop1), count in usage.items()
        ))

    def diversify_hop1_motifs(self,
                               nodes: List[_NodeV3],
                               E_target: torch.Tensor,
                               output_dir: Optional[str] = None,
                               max_moves: int = 12,
                               max_trials: int = 96,
                               deadline_monotonic: Optional[float] = None) -> Tuple[List[_NodeV3], Dict[str, object]]:
        adjuster = Hop1Adjuster(
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            external_requirement_fn=check_external_connection_requirement,
            external_requirement_node_fn=self._partition_aware_external_requirement,
        )
        nodes_by_su: Dict[int, List[int]] = defaultdict(list)
        for node in nodes:
            nodes_by_su[int(node.su_type)].append(int(node.global_id))

        eligible_sus = {5, 10, 11, 12, 13, 22, 23, 24, 25}
        accepted: List[Dict[str, object]] = []
        trials = 0
        stop_reason = 'no_improving_swap'
        while len(accepted) < max(0, int(max_moves)) and trials < max(0, int(max_trials)):
            if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                stop_reason = 'deadline'
                break
            usage = build_motif_usage_from_pairs(
                (
                    int(node.su_type),
                    tuple(sorted(int(x) for x in Counter(getattr(node, 'hop1_su', {}) or {}).elements())),
                )
                for node in nodes
            )
            concentration_before = self._motif_concentration(usage)
            hotspots = sorted(
                (
                    (int(count), int(center_su), tuple(hop1_tuple))
                    for (center_su, hop1_tuple), count in usage.items()
                    if int(center_su) in eligible_sus and int(count) > 1 and tuple(hop1_tuple)
                ),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            improved = False
            for hotspot_count, center_su, current_hop1 in hotspots:
                alternatives = [
                    dict(entry)
                    for entry in adjuster.hop1_by_su.get(int(center_su), [])
                    if tuple(entry.get('hop1_tuple', ())) != tuple(current_hop1)
                    and len(tuple(entry.get('hop1_tuple', ()))) == len(tuple(current_hop1))
                    and not self._alternative_resource_shortage(
                        nodes_by_su,
                        int(center_su),
                        tuple(int(x) for x in tuple(entry.get('hop1_tuple', ()))),
                    )
                ]
                alternatives.sort(key=lambda entry: (
                    int(usage.get((int(center_su), tuple(entry['hop1_tuple'])), 0)),
                    int(multiset_l1_distance(tuple(current_hop1), tuple(entry['hop1_tuple']))),
                    abs(float(entry.get('mu_median', 0.0))),
                    tuple(entry['hop1_tuple']),
                ))
                if not alternatives:
                    continue
                center_nodes = [
                    node for node in nodes
                    if int(node.su_type) == int(center_su)
                    and tuple(sorted(int(x) for x in Counter(node.hop1_su).elements())) == tuple(current_hop1)
                ]
                center_nodes.sort(key=lambda node: -int(node.global_id))
                for center_node in center_nodes:
                    for alternative in alternatives:
                        if trials >= int(max_trials):
                            stop_reason = 'trial_budget'
                            break
                        if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                            stop_reason = 'deadline'
                            break
                        trials += 1
                        result = adjuster.execute_hop1_replacement(
                            nodes=nodes,
                            target_node_id=int(center_node.global_id),
                            new_hop1_tuple=tuple(int(x) for x in tuple(alternative['hop1_tuple'])),
                            E_target=E_target,
                            dry_run=False,
                            nodes_by_su=nodes_by_su,
                            return_undo_state=True,
                        )
                        if not bool(result.get('success', False)):
                            continue
                        undo_state = result.pop('_undo_state', None)
                        if undo_state is None:
                            raise RuntimeError('successful diversity swap did not provide undo state')
                        usage_after = build_motif_usage_from_pairs(
                            (
                                int(node.su_type),
                                tuple(sorted(int(x) for x in Counter(getattr(node, 'hop1_su', {}) or {}).elements())),
                            )
                            for node in nodes
                        )
                        concentration_after = self._motif_concentration(usage_after)
                        if float(concentration_after) < float(concentration_before) - 1e-9:
                            accepted.append({
                                'move': int(len(accepted) + 1),
                                'global_id': int(center_node.global_id),
                                'center_su': int(center_su),
                                'old_hop1': tuple(current_hop1),
                                'new_hop1': tuple(alternative['hop1_tuple']),
                                'hotspot_count_before': int(hotspot_count),
                                'concentration_before': float(concentration_before),
                                'concentration_after': float(concentration_after),
                                'affected_nodes': list(result.get('affected_nodes', []) or []),
                            })
                            improved = True
                            break
                        self._restore_hop1_state(nodes, undo_state)
                    if improved or trials >= int(max_trials) or stop_reason == 'deadline':
                        break
                if improved or trials >= int(max_trials) or stop_reason == 'deadline':
                    break
            if not improved:
                if trials >= int(max_trials):
                    stop_reason = 'trial_budget'
                break

        if len(accepted) >= max(0, int(max_moves)) and int(max_moves) > 0:
            stop_reason = 'move_budget'
        if output_dir and accepted:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(accepted).to_csv(out / 'layer1_hop1_diversity_moves.csv', index=False)
        return nodes, {
            'adjustments': int(len(accepted)),
            'trials': int(trials),
            'stop_reason': str(stop_reason),
            'details': accepted,
        }

    def _rank_su_types_for_positive_peaks(self,
                                          priority_list: List[int],
                                          positive_peaks: List,
                                          fallback_window: Optional[Tuple[float, float]] = None) -> List[int]:
        common_ranges = self._get_su_common_ranges()
        ranked: List[int] = []
        for su in priority_list:
            meta = common_ranges.get(int(su), None)
            if not isinstance(meta, dict):
                continue
            mu_lo = float(meta.get('mu_min', meta.get('mu_median', 0.0)))
            mu_hi = float(meta.get('mu_max', meta.get('mu_median', 0.0)))
            ok = False
            for pk in positive_peaks:
                if self._ranges_overlap(mu_lo, mu_hi, float(pk.ppm_min), float(pk.ppm_max)):
                    ok = True
                    break
            if not ok and fallback_window is not None:
                ok = self._ranges_overlap(mu_lo, mu_hi, float(fallback_window[0]), float(fallback_window[1]))
            if ok:
                ranked.append(int(su))
        return ranked if ranked else [int(x) for x in priority_list]

    def _pick_positive_peak_for_node(self,
                                     center_su: int,
                                     positive_peaks: List,
                                     neg_peak,
                                     target_window_ppm: float = 24.0) -> Optional[object]:
        common_ranges = self._get_su_common_ranges()
        meta = common_ranges.get(int(center_su), None)
        mu_lo = float(meta.get('mu_min', 0.0)) if isinstance(meta, dict) else -1e9
        mu_hi = float(meta.get('mu_max', 0.0)) if isinstance(meta, dict) else 1e9
        chosen = None
        best_key = None
        neg_center = float(getattr(neg_peak, 'center_ppm', 0.0))
        for pk in positive_peaks:
            pos_lo = float(getattr(pk, 'ppm_min', neg_center))
            pos_hi = float(getattr(pk, 'ppm_max', neg_center))
            pos_center = float(getattr(pk, 'center_ppm', 0.5 * (pos_lo + pos_hi)))
            if abs(float(pos_center) - float(neg_center)) > float(target_window_ppm):
                continue
            if not self._ranges_overlap(mu_lo, mu_hi, pos_lo, pos_hi):
                continue
            key = (
                -float(getattr(pk, 'intensity', 0.0)),
                abs(float(pos_center) - float(neg_center)),
                abs(0.5 * (pos_lo + pos_hi) - float((mu_lo + mu_hi) * 0.5)),
            )
            if best_key is None or key < best_key:
                best_key = key
                chosen = pk
        return chosen

    @staticmethod
    def _region_abs_loss(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> float:
        mask = (ppm >= float(lo)) & (ppm <= float(hi))
        if not np.any(mask):
            return float('inf')
        return float(np.sum(np.abs(diff[mask])))

    def _get_carbonyl_anchor_ids(self, nodes: List[_NodeV3], node: _NodeV3) -> List[int]:
        center_su = int(node.su_type)
        hop1_ids = [int(nid) for nid in list(getattr(node, 'hop1_ids', []) or []) if 0 <= int(nid) < len(nodes)]
        if center_su == 1:
            return hop1_ids[:1]
        if center_su == 2:
            return [nid for nid in hop1_ids if int(nodes[nid].su_type) not in {5, 19}]
        return []

    @staticmethod
    def _joint_window_score(ppm: np.ndarray, diff: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
        mask = (ppm >= float(lo)) & (ppm <= float(hi))
        if not mask.any():
            return {'pos': 0.0, 'neg': 0.0, 'net': 0.0, 'abs': 0.0}
        seg = diff[mask]
        pos = float(np.sum(seg[seg > 0])) if np.any(seg > 0) else 0.0
        neg = float(-np.sum(seg[seg < 0])) if np.any(seg < 0) else 0.0
        return {
            'pos': pos,
            'neg': neg,
            'net': float(pos - neg),
            'abs': float(np.sum(np.abs(seg))),
        }

    def _decide_carbonyl_joint_direction(self,
                                         ppm: np.ndarray,
                                         diff: np.ndarray,
                                         pos_rel_threshold: float = 0.08,
                                         neg_rel_threshold: float = 0.08) -> Dict[str, object]:
        low = self._joint_window_score(ppm, diff, 160.0, 170.0)
        mid = self._joint_window_score(ppm, diff, 172.0, 180.0)
        ket = self._joint_window_score(ppm, diff, 186.0, 205.0)

        carbonyl_mask = (ppm >= 160.0) & (ppm <= 205.0)
        carbonyl_abs = float(np.sum(np.abs(diff[carbonyl_mask]))) if np.any(carbonyl_mask) else float(np.sum(np.abs(diff)))
        pos_thr = float(pos_rel_threshold) * max(1e-8, carbonyl_abs)
        neg_thr = float(neg_rel_threshold) * max(1e-8, carbonyl_abs)

        direction = None
        if float(low['neg']) > float(neg_thr) and float(mid['pos']) > float(pos_thr):
            direction = 'to_aliphatic'
        elif float(low['pos']) > float(pos_thr) and float(mid['neg']) > float(neg_thr):
            direction = 'to_aryl9'
        return {
            'direction': direction,
            'thresholds': {'pos': float(pos_thr), 'neg': float(neg_thr), 'carbonyl_abs': float(carbonyl_abs)},
            'windows': {
                '160_170': low,
                '172_180': mid,
                '186_205': ket,
            },
        }

    def _rank_joint_target_anchor_types(self,
                                        ppm: np.ndarray,
                                        diff: np.ndarray,
                                        direction: str) -> List[int]:
        if direction == 'to_aryl9':
            return [9]
        scores = {
            23: self._joint_window_score(ppm, diff, 18.0, 35.0),
            24: self._joint_window_score(ppm, diff, 32.0, 50.0),
            25: self._joint_window_score(ppm, diff, 40.0, 60.0),
        }
        ranked = sorted(
            scores.keys(),
            key=lambda su: (float(scores[su]['net']), float(scores[su]['pos']), -float(scores[su]['neg'])),
            reverse=True,
        )
        return [int(su) for su in ranked]

    def _try_joint_carbonyl_swap(self,
                                 nodes: List[_NodeV3],
                                 center_id: int,
                                 old_anchor_id: int,
                                 target_anchor_types: List[int],
                                 E_target: torch.Tensor,
                                 swap_helper: Hop1Adjuster) -> Optional[Dict[str, object]]:
        center_node = nodes[int(center_id)]
        old_anchor = nodes[int(old_anchor_id)]
        for target_type in target_anchor_types:
            candidates = [
                n for n in nodes
                if int(n.su_type) == int(target_type)
                and int(n.global_id) not in center_node.hop1_ids
                and int(n.global_id) != int(center_id)
            ]
            candidates.sort(key=lambda n: (int(n.remaining_hop1_slots()) > 0, -int(n.remaining_hop1_slots()), -int(n.global_id)), reverse=True)
            for target in candidates:
                success, swap_tail_id, affected = swap_helper._try_two_edge_swap(
                    nodes,
                    t=int(center_id),
                    u=int(old_anchor_id),
                    v=int(target.global_id),
                    E_target=E_target,
                )
                if not success:
                    continue
                swap_helper._remove_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                swap_helper._remove_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                if not swap_helper._add_hop1_edge(nodes, int(center_id), int(target.global_id)):
                    swap_helper._add_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                    swap_helper._add_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                    continue
                if not swap_helper._add_hop1_edge(nodes, int(old_anchor_id), int(swap_tail_id)):
                    swap_helper._remove_hop1_edge(nodes, int(center_id), int(target.global_id))
                    swap_helper._add_hop1_edge(nodes, int(center_id), int(old_anchor_id))
                    swap_helper._add_hop1_edge(nodes, int(target.global_id), int(swap_tail_id))
                    continue
                return {
                    'center_id': int(center_id),
                    'center_su': int(center_node.su_type),
                    'old_anchor_id': int(old_anchor_id),
                    'old_anchor_su': int(old_anchor.su_type),
                    'new_anchor_id': int(target.global_id),
                    'new_anchor_su': int(target.su_type),
                    'swap_tail_id': int(swap_tail_id),
                    'affected_nodes': list(sorted(set(int(x) for x in affected))),
                }
        return None

    def adjust_carbonyl_anchor_jointly(self,
                                       nodes: List[_NodeV3],
                                       S_target: torch.Tensor,
                                       E_target: torch.Tensor,
                                       lib_path: Optional[str] = None,
                                       hwhm: float = 1.0,
                                       allow_approx: bool = True,
                                       max_iterations: int = 3,
                                       max_adjustments_per_iter: int = 3,
                                       pos_rel_threshold: float = 0.08,
                                       neg_rel_threshold: float = 0.08,
                                       deadline_monotonic: Optional[float] = None) -> Tuple[List[_NodeV3], Dict[str, object]]:
        if lib_path is None:
            return nodes, {'adjustments': 0, 'iterations': 0, 'details': [], 'reason': 'missing_lib'}

        swap_helper = Hop1Adjuster(
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            external_requirement_fn=check_external_connection_requirement,
            external_requirement_node_fn=self._partition_aware_external_requirement,
        )

        all_moves: List[Dict[str, object]] = []
        diagnostics: List[Dict[str, object]] = []
        deadline_exhausted = False
        for iter_idx in range(max(1, int(max_iterations))):
            if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                deadline_exhausted = True
                break
            diff_info = self.evaluator.compute_difference_spectrum(
                nodes=nodes,
                S_target=S_target,
                lib_path=lib_path,
                hwhm=hwhm,
                allow_approx=allow_approx,
            )
            ppm = np.asarray(diff_info.get('ppm', []), dtype=np.float64)
            diff = np.asarray(diff_info.get('diff', []), dtype=np.float64)
            if ppm.size == 0 or diff.size == 0:
                break

            decision = self._decide_carbonyl_joint_direction(
                ppm,
                diff,
                pos_rel_threshold=float(pos_rel_threshold),
                neg_rel_threshold=float(neg_rel_threshold),
            )
            direction = decision.get('direction')
            diagnostics.append({
                'iteration': int(iter_idx + 1),
                'direction': direction,
                'decision': decision,
                'r2_before': float(diff_info.get('r2', 0.0)),
            })
            if direction is None:
                break

            if direction == 'to_aliphatic':
                source_anchor_types = {9}
                target_anchor_types = self._rank_joint_target_anchor_types(ppm, diff, direction)
            else:
                source_anchor_types = {23, 24, 25}
                target_anchor_types = [9]

            grouped_assignments = self.evaluator.build_grouped_assignments(nodes, lib_path, allow_approx)
            candidates: List[Dict[str, object]] = []
            for node in nodes:
                center_su = int(node.su_type)
                if center_su not in {1, 2}:
                    continue
                anchor_ids = self._get_carbonyl_anchor_ids(nodes, node)
                if not anchor_ids:
                    continue
                mu_pred = grouped_assignments.get(int(node.global_id), {}).get('mu', 0.0)
                for anchor_id in anchor_ids:
                    anchor_su = int(nodes[int(anchor_id)].su_type)
                    if anchor_su not in source_anchor_types:
                        continue
                    candidates.append({
                        'center_id': int(node.global_id),
                        'center_su': center_su,
                        'mu_pred': float(mu_pred or 0.0),
                        'old_anchor_id': int(anchor_id),
                        'old_anchor_su': int(anchor_su),
                    })

            if direction == 'to_aliphatic':
                candidates.sort(key=lambda c: (0 if int(c['center_su']) == 1 else 1, abs(float(c['mu_pred']) - 165.0), int(c['center_id'])))
            else:
                candidates.sort(key=lambda c: (0 if int(c['center_su']) == 2 else 1, abs(float(c['mu_pred']) - 176.0), int(c['center_id'])))

            iter_moves: List[Dict[str, object]] = []
            used_centers = set()
            for cand in candidates:
                if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                    deadline_exhausted = True
                    break
                if len(iter_moves) >= int(max_adjustments_per_iter):
                    break
                if int(cand['center_id']) in used_centers:
                    continue
                move = self._try_joint_carbonyl_swap(
                    nodes=nodes,
                    center_id=int(cand['center_id']),
                    old_anchor_id=int(cand['old_anchor_id']),
                    target_anchor_types=target_anchor_types,
                    E_target=E_target,
                    swap_helper=swap_helper,
                )
                if move is None:
                    continue
                move['iteration'] = int(iter_idx + 1)
                move['direction'] = str(direction)
                iter_moves.append(move)
                used_centers.add(int(cand['center_id']))

            if not iter_moves:
                break
            all_moves.extend(iter_moves)

        return nodes, {
            'adjustments': int(len(all_moves)),
            'iterations': int(len(diagnostics)),
            'details': all_moves,
            'diagnostics': diagnostics,
            'deadline_exhausted': bool(deadline_exhausted),
            'reason': 'deadline' if deadline_exhausted else 'complete',
        }

    def adjust_hop1_based_on_spectrum(self,
                                      nodes: List[_NodeV3],
                                      S_target: torch.Tensor,
                                      E_target: torch.Tensor,
                                      lib_path: Optional[str] = None,
                                      output_dir: Optional[str] = None,
                                      hwhm: float = 1.0,
                                      allow_approx: bool = True,
                                      neg_threshold: float = -0.5,
                                      pos_threshold: float = 0.5,
                                      max_iterations: int = 3,
                                      adjustment_groups: Optional[List[str]] = None,
                                      max_candidate_trials: int = 240,
                                      max_candidate_nodes_per_su: int = 24,
                                      max_alternatives_per_node: int = 4,
                                      max_moves_per_region: int = 8,
                                      max_full_evals: int = 32,
                                      min_r2_gain: float = 1e-4,
                                      deadline_monotonic: Optional[float] = None) -> Tuple[List[_NodeV3], Dict[str, object]]:
        device = self.device
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        if adjustment_groups is None:
            adjustment_groups = ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic']
        wanted_groups = [str(x).strip().lower() for x in list(adjustment_groups or []) if str(x).strip()]

        adjuster = Hop1Adjuster(
            port_combinations=HOP1_PORT_COMBINATIONS,
            validate_connection_fn=validate_connection,
            external_requirement_fn=check_external_connection_requirement,
            external_requirement_node_fn=self._partition_aware_external_requirement,
        )

        E_target = E_target.to(device).flatten()
        S_target, _ = self.evaluator.resolve_inputs(S_target)

        total_adjustments = 0
        total_candidate_trials = 0
        raw_candidate_records = 0
        unique_candidate_records = 0
        cached_structural_failures = 0
        candidate_trial_budget = int(max_candidate_trials or 0)
        if candidate_trial_budget <= 0:
            candidate_trial_budget = 10**9
        candidate_cap = max(1, int(max_candidate_nodes_per_su or 1))
        alternative_cap = max(1, int(max_alternatives_per_node or 1))
        moves_per_region_cap = max(1, int(max_moves_per_region or 1))
        full_eval_budget = int(max_full_evals or 0)
        if full_eval_budget <= 0:
            full_eval_budget = 10**9
        full_eval_count = 0
        iteration_summaries: List[Dict[str, object]] = []
        reject_summaries: List[Dict[str, object]] = []
        best_r2 = -1e9
        best_nodes = copy.deepcopy(nodes)
        region_summaries: List[Dict[str, object]] = []
        budget_exhausted = False
        deadline_exhausted = False
        graph_revision = 0
        structural_failure_cache: Set[Tuple[int, int, Tuple[int, ...]]] = set()
        nodes_by_su: Dict[int, List[int]] = defaultdict(list)
        for node in nodes:
            nodes_by_su[int(node.su_type)].append(int(node.global_id))

        region_specs = list(self._region_specs())
        if wanted_groups:
            specs_by_name = {
                str(spec[0]).lower(): spec
                for spec in region_specs
            }
            region_specs = [
                specs_by_name[name]
                for name in wanted_groups
                if name in specs_by_name
            ]

        region_count = max(1, int(len(region_specs)))
        candidate_quota_base, candidate_quota_extra = divmod(int(candidate_trial_budget), region_count)
        full_eval_quota_base, full_eval_quota_extra = divmod(int(full_eval_budget), region_count)
        candidate_quota_carry = 0
        full_eval_quota_carry = 0

        for region_index, (region_name, region_lo, region_hi, priority_list) in enumerate(region_specs):
            if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                budget_exhausted = True
                deadline_exhausted = True
                break
            candidate_quota = int(candidate_quota_base + (1 if region_index < candidate_quota_extra else 0))
            full_eval_quota = int(full_eval_quota_base + (1 if region_index < full_eval_quota_extra else 0))
            region_candidate_limit = min(
                int(candidate_trial_budget),
                int(total_candidate_trials + candidate_quota + candidate_quota_carry),
            )
            region_full_eval_limit = min(
                int(full_eval_budget),
                int(full_eval_count + full_eval_quota + full_eval_quota_carry),
            )
            region_candidate_start = int(total_candidate_trials)
            region_full_eval_start = int(full_eval_count)
            region_adjustments = 0
            region_details: List[Dict[str, object]] = []
            stop_reason = 'max_iterations'
            region_budget_exhausted = False
            for region_iter in range(max(1, int(max_iterations))):
                if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                    stop_reason = 'layer1_deadline_exhausted'
                    budget_exhausted = True
                    deadline_exhausted = True
                    break
                if int(region_adjustments) >= int(moves_per_region_cap):
                    stop_reason = 'region_move_budget_exhausted'
                    break
                if int(total_candidate_trials) >= int(candidate_trial_budget):
                    stop_reason = 'candidate_trial_budget_exhausted'
                    budget_exhausted = True
                    break
                if int(total_candidate_trials) >= int(region_candidate_limit):
                    stop_reason = 'region_candidate_trial_budget_exhausted'
                    region_budget_exhausted = True
                    break
                if int(full_eval_count) >= int(full_eval_budget):
                    stop_reason = 'full_eval_budget_exhausted'
                    budget_exhausted = True
                    break
                if int(full_eval_count) >= int(region_full_eval_limit):
                    stop_reason = 'region_full_eval_budget_exhausted'
                    region_budget_exhausted = True
                    break
                snapshot = self.evaluator.build_eval_snapshot(
                    nodes=nodes,
                    S_target=S_target,
                    lib_path=lib_path,
                    hwhm=hwhm,
                    allow_approx=bool(allow_approx),
                )
                full_eval_count += 1
                if int(snapshot.get('n_peaks', 0)) <= 0:
                    stop_reason = 'no_valid_peaks'
                    break

                r2 = float(snapshot.get('r2', 0.0))
                diff = np.asarray(snapshot.get('diff', []), dtype=np.float64)
                ppm_np = np.asarray(snapshot.get('ppm', []), dtype=np.float64)
                node_peak_rows = list(snapshot.get('rows', []))
                loss_before = self._region_abs_loss(ppm_np, diff, float(region_lo), float(region_hi))
                if float(r2) > float(best_r2):
                    best_r2 = float(r2)
                    best_nodes = copy.deepcopy(nodes)

                region_mask = (ppm_np >= float(region_lo)) & (ppm_np <= float(region_hi))
                neg_peaks, pos_peaks = adjuster.analyze_difference_spectrum(
                    diff=diff[region_mask],
                    ppm=ppm_np[region_mask],
                    neg_threshold=float(neg_threshold),
                    pos_threshold=float(pos_threshold),
                )
                if not neg_peaks:
                    stop_reason = 'no_negative_peaks'
                    break
                if not pos_peaks:
                    stop_reason = 'no_positive_peaks'
                    break

                su_priority = self._rank_su_types_for_positive_peaks(
                    priority_list=priority_list,
                    positive_peaks=pos_peaks,
                    fallback_window=(float(region_lo), float(region_hi)),
                )
                motif_usage = build_motif_usage_from_pairs(
                    (
                        int(node.su_type),
                        tuple(sorted(int(x) for x in Counter(getattr(node, 'hop1_su', {}) or {}).elements())),
                    )
                    for node in nodes
                )

                accepted_move: Optional[Dict[str, object]] = None
                candidate_records_by_key: Dict[Tuple[int, Tuple[int, ...]], Dict[str, object]] = {}
                for neg_peak in list(neg_peaks[: max(1, min(5, len(neg_peaks)))]):
                    neg_nodes_all = adjuster.find_nodes_in_peak_region(nodes, node_peak_rows, neg_peak)
                    for su_type in su_priority:
                        cand_nodes = [x for x in neg_nodes_all if int(x['center_su']) == int(su_type)]
                        cand_nodes.sort(key=lambda x: (
                            -int(motif_usage.get((int(x['center_su']), tuple(x['hop1_ms'])), 0)),
                            int(x['global_id']),
                        ))
                        cand_nodes = cand_nodes[:candidate_cap]
                        for node_info in cand_nodes:
                            for pos_peak in list(pos_peaks[: max(1, min(5, len(pos_peaks)))]):
                                alternatives = adjuster.find_alternative_hop1(
                                    int(node_info['center_su']),
                                    tuple(node_info['hop1_ms']),
                                    (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                                    motif_usage=motif_usage,
                                )
                                if not alternatives:
                                    reject_summaries.append({
                                        'region': str(region_name),
                                        'iteration': int(region_iter + 1),
                                        'global_id': int(node_info['global_id']),
                                        'center_su': int(node_info['center_su']),
                                        'old_hop1': tuple(node_info['hop1_ms']),
                                        'neg_window': (float(neg_peak.ppm_min), float(neg_peak.ppm_max)),
                                        'pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                                        'reason': 'no_alternative_hop1_template',
                                    })
                                    continue
                                feasible_alternatives: List[Dict[str, object]] = []
                                for alt in alternatives:
                                    resource_shortage = self._alternative_resource_shortage(
                                        nodes_by_su,
                                        int(node_info['center_su']),
                                        tuple(int(x) for x in tuple(alt['hop1_tuple'])),
                                    )
                                    if resource_shortage:
                                        reject_summaries.append({
                                            'region': str(region_name),
                                            'iteration': int(region_iter + 1),
                                            'global_id': int(node_info['global_id']),
                                            'center_su': int(node_info['center_su']),
                                            'old_hop1': tuple(node_info['hop1_ms']),
                                            'new_hop1': tuple(alt['hop1_tuple']),
                                            'neg_window': (float(neg_peak.ppm_min), float(neg_peak.ppm_max)),
                                            'pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                                            'resource_shortage': dict(sorted(resource_shortage.items())),
                                            'reason': 'alternative_resource_unavailable',
                                        })
                                        continue
                                    feasible_alternatives.append(alt)
                                    if len(feasible_alternatives) >= int(alternative_cap):
                                        break
                                for alt in feasible_alternatives:
                                    pos_center = 0.5 * (float(pos_peak.ppm_min) + float(pos_peak.ppm_max))
                                    target_usage = int(motif_usage.get(
                                        (int(node_info['center_su']), tuple(int(x) for x in tuple(alt['hop1_tuple']))),
                                        0,
                                    ))
                                    candidate_record = {
                                        'node_info': node_info,
                                        'neg_peak': neg_peak,
                                        'pos_peak': pos_peak,
                                        'alt': alt,
                                        'rank': (
                                            -abs(float(neg_peak.intensity)),
                                            abs(float(alt.get('mu_median', pos_center)) - float(pos_center)),
                                            int(target_usage),
                                            -float(alt.get('score', 0.0)),
                                            int(node_info['global_id']),
                                        ),
                                    }
                                    raw_candidate_records += 1
                                    candidate_key = (
                                        int(node_info['global_id']),
                                        tuple(int(x) for x in tuple(alt['hop1_tuple'])),
                                    )
                                    previous = candidate_records_by_key.get(candidate_key)
                                    if previous is None or candidate_record['rank'] < previous['rank']:
                                        candidate_records_by_key[candidate_key] = candidate_record
                candidate_records = list(candidate_records_by_key.values())
                unique_candidate_records += int(len(candidate_records))
                candidate_records.sort(key=lambda x: x['rank'])

                for rec in candidate_records:
                    if deadline_monotonic is not None and time.perf_counter() >= float(deadline_monotonic):
                        budget_exhausted = True
                        deadline_exhausted = True
                        break
                    if int(total_candidate_trials) >= int(candidate_trial_budget):
                        budget_exhausted = True
                        break
                    if int(total_candidate_trials) >= int(region_candidate_limit):
                        region_budget_exhausted = True
                        break
                    node_info = rec['node_info']
                    neg_peak = rec['neg_peak']
                    pos_peak = rec['pos_peak']
                    alt = rec['alt']
                    replacement_key = (
                        int(graph_revision),
                        int(node_info['global_id']),
                        tuple(int(x) for x in tuple(alt['hop1_tuple'])),
                    )
                    if replacement_key in structural_failure_cache:
                        cached_structural_failures += 1
                        continue
                    total_candidate_trials += 1
                    result = adjuster.execute_hop1_replacement(
                        nodes=nodes,
                        target_node_id=int(node_info['global_id']),
                        new_hop1_tuple=tuple(alt['hop1_tuple']),
                        E_target=E_target,
                        dry_run=False,
                        nodes_by_su=nodes_by_su,
                        return_undo_state=True,
                    )
                    if not bool(result.get('success', False)):
                        structural_failure_cache.add(replacement_key)
                        reject_summaries.append({
                            'region': str(region_name),
                            'iteration': int(region_iter + 1),
                            'global_id': int(node_info['global_id']),
                            'center_su': int(node_info['center_su']),
                            'old_hop1': tuple(node_info['hop1_ms']),
                            'new_hop1': tuple(alt['hop1_tuple']),
                            'neg_window': (float(neg_peak.ppm_min), float(neg_peak.ppm_max)),
                            'pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                            'reason': str(result.get('error', 'strict_swap_failed')),
                        })
                        continue
                    prev_state = result.pop('_undo_state', None)
                    if prev_state is None:
                        raise RuntimeError("successful 1-hop replacement did not provide undo state")
                    if int(full_eval_count) >= int(full_eval_budget):
                        self._restore_hop1_state(nodes, prev_state)
                        budget_exhausted = True
                        break
                    if int(full_eval_count) >= int(region_full_eval_limit):
                        self._restore_hop1_state(nodes, prev_state)
                        region_budget_exhausted = True
                        break
                    try:
                        snapshot_try = self.evaluator.build_eval_snapshot(
                            nodes=nodes,
                            S_target=S_target,
                            lib_path=lib_path,
                            hwhm=hwhm,
                            allow_approx=bool(allow_approx),
                        )
                    except Exception:
                        self._restore_hop1_state(nodes, prev_state)
                        raise
                    full_eval_count += 1
                    diff_try = np.asarray(snapshot_try.get('diff', []), dtype=np.float64)
                    ppm_try = np.asarray(snapshot_try.get('ppm', []), dtype=np.float64)
                    loss_after = self._region_abs_loss(ppm_try, diff_try, float(region_lo), float(region_hi))
                    r2_after = float(snapshot_try.get('r2', r2))
                    r2_gain = float(r2_after) - float(r2)
                    if (
                        float(loss_after) < float(loss_before) - 1e-9
                        and float(r2_gain) >= max(0.0, float(min_r2_gain))
                    ):
                        accepted_move = {
                            'region': str(region_name),
                            'iteration': int(region_iter + 1),
                            'global_id': int(node_info['global_id']),
                            'center_su': int(node_info['center_su']),
                            'old_hop1': tuple(node_info['hop1_ms']),
                            'new_hop1': tuple(alt['hop1_tuple']),
                            'old_mu': float(node_info['mu']),
                            'target_pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                            'target_pos_center': float(pos_peak.center_ppm),
                            'target_mu_median': float(alt.get('mu_median', 0.0)),
                            'loss_before': float(loss_before),
                            'loss_after': float(loss_after),
                            'r2_before': float(r2),
                            'r2_after': float(r2_after),
                            'r2_gain': float(r2_gain),
                            'neg_window': (float(neg_peak.ppm_min), float(neg_peak.ppm_max)),
                            'pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                            'affected_nodes': list(result.get('affected_nodes', []) or []),
                        }
                        break
                    self._restore_hop1_state(nodes, prev_state)
                    reject_summaries.append({
                        'region': str(region_name),
                        'iteration': int(region_iter + 1),
                        'global_id': int(node_info['global_id']),
                        'center_su': int(node_info['center_su']),
                        'old_hop1': tuple(node_info['hop1_ms']),
                        'new_hop1': tuple(alt['hop1_tuple']),
                        'neg_window': (float(neg_peak.ppm_min), float(neg_peak.ppm_max)),
                        'pos_window': (float(pos_peak.ppm_min), float(pos_peak.ppm_max)),
                        'loss_before': float(loss_before),
                        'loss_after': float(loss_after),
                        'r2_before': float(r2),
                        'r2_after': float(r2_after),
                        'r2_gain': float(r2_gain),
                        'required_r2_gain': max(0.0, float(min_r2_gain)),
                        'reason': (
                            'r2_gain_below_threshold'
                            if float(r2_gain) < max(0.0, float(min_r2_gain))
                            else 'region_loss_not_improved'
                        ),
                    })

                if accepted_move is None:
                    if bool(budget_exhausted):
                        stop_reason = 'candidate_trial_budget_exhausted'
                    elif bool(region_budget_exhausted):
                        stop_reason = 'region_budget_exhausted'
                    else:
                        stop_reason = 'no_valid_structural_swap'
                    break

                region_adjustments += 1
                total_adjustments += 1
                graph_revision += 1
                region_details.append(dict(accepted_move))
                iteration_summaries.append(dict(accepted_move))

            region_summaries.append({
                'region': str(region_name),
                'window': (float(region_lo), float(region_hi)),
                'adjustments': int(region_adjustments),
                'stop_reason': str(stop_reason),
                'candidate_trials': int(total_candidate_trials - region_candidate_start),
                'candidate_trial_quota': int(region_candidate_limit - region_candidate_start),
                'full_evals': int(full_eval_count - region_full_eval_start),
                'full_eval_quota': int(region_full_eval_limit - region_full_eval_start),
                'details': region_details,
            })
            candidate_quota_carry = max(0, int(region_candidate_limit - total_candidate_trials))
            full_eval_quota_carry = max(0, int(region_full_eval_limit - full_eval_count))
            if bool(budget_exhausted):
                break

        if output_dir:
            try:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                rows = []
                for move in iteration_summaries:
                    rows.append({
                        'region': str(move.get('region', '')),
                        'iteration': int(move.get('iteration', 0)),
                        'global_id': int(move.get('global_id', -1)),
                        'center_su': int(move.get('center_su', -1)),
                        'old_hop1': str(tuple(move.get('old_hop1', ()) or ())),
                        'new_hop1': str(tuple(move.get('new_hop1', ()) or ())),
                        'affected_nodes': ','.join(str(x) for x in list(move.get('affected_nodes', []) or [])),
                        'neg_window': str(tuple(move.get('neg_window', ()) or ())),
                        'pos_window': str(tuple(move.get('pos_window', ()) or ())),
                        'old_mu': float(move.get('old_mu', 0.0)),
                        'target_mu_median': float(move.get('target_mu_median', 0.0)),
                        'loss_before': float(move.get('loss_before', 0.0)),
                        'loss_after': float(move.get('loss_after', 0.0)),
                        'r2_before': float(move.get('r2_before', 0.0)),
                        'r2_after': float(move.get('r2_after', 0.0)),
                        'r2_gain': float(move.get('r2_gain', 0.0)),
                    })
                pd.DataFrame(rows).to_csv(str(out / 'layer1_hop1_adjust_moves.csv'), index=False)
                reject_rows = []
                for rec in reject_summaries:
                    reject_rows.append({
                        'region': str(rec.get('region', '')),
                        'iteration': int(rec.get('iteration', 0)),
                        'global_id': int(rec.get('global_id', -1)),
                        'center_su': int(rec.get('center_su', -1)),
                        'old_hop1': str(tuple(rec.get('old_hop1', ()) or ())),
                        'new_hop1': str(tuple(rec.get('new_hop1', ()) or ())),
                        'neg_window': str(tuple(rec.get('neg_window', ()) or ())),
                        'pos_window': str(tuple(rec.get('pos_window', ()) or ())),
                        'loss_before': rec.get('loss_before', ''),
                        'loss_after': rec.get('loss_after', ''),
                        'r2_before': rec.get('r2_before', ''),
                        'r2_after': rec.get('r2_after', ''),
                        'r2_gain': rec.get('r2_gain', ''),
                        'required_r2_gain': rec.get('required_r2_gain', ''),
                        'reason': str(rec.get('reason', '')),
                    })
                pd.DataFrame(reject_rows).to_csv(str(out / 'layer1_hop1_adjust_rejects.csv'), index=False)
            except Exception as e:
                print(f"  [Hop1调整] 保存move表失败: {e}")

        final_metrics = self.evaluator.evaluate_with_library(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path,
            output_dir=output_dir,
            hwhm=hwhm,
            allow_approx=bool(allow_approx),
        )
        final_attempt_r2 = float(final_metrics.get('r2', 0.0))
        selected_nodes = nodes
        selected_metrics = final_metrics
        best_source = 'final_attempt'

        if float(final_attempt_r2) >= float(best_r2):
            best_r2 = float(final_attempt_r2)
            best_nodes = copy.deepcopy(nodes)
        else:
            selected_nodes = copy.deepcopy(best_nodes)
            selected_metrics = self.evaluator.evaluate_with_library(
                nodes=selected_nodes,
                S_target=S_target,
                lib_path=lib_path,
                output_dir=output_dir,
                hwhm=hwhm,
                allow_approx=bool(allow_approx),
            )
            best_r2 = float(selected_metrics.get('r2', best_r2))
            best_source = 'best_iteration'

        summary = {
            'total_adjustments': int(total_adjustments),
            'iterations': int(len(iteration_summaries)),
            'iteration_details': iteration_summaries,
            'region_summaries': region_summaries,
            'best_r2': float(best_r2),
            'final_r2': float(selected_metrics.get('r2', 0.0)),
            'final_attempt_r2': float(final_attempt_r2),
            'selected_source': str(best_source),
            'adjuster_stats': dict(adjuster.stats),
            'candidate_trials': int(total_candidate_trials),
            'candidate_trial_budget': int(candidate_trial_budget),
            'candidate_records_raw': int(raw_candidate_records),
            'candidate_records_unique': int(unique_candidate_records),
            'candidate_records_deduplicated': int(raw_candidate_records - unique_candidate_records),
            'cached_structural_failures': int(cached_structural_failures),
            'rejects': int(len(reject_summaries)),
            'full_eval_count': int(full_eval_count),
            'full_eval_budget': int(full_eval_budget),
            'moves_per_region_budget': int(moves_per_region_cap),
            'min_r2_gain': max(0.0, float(min_r2_gain)),
            'budget_exhausted': bool(budget_exhausted),
            'deadline_exhausted': bool(deadline_exhausted),
        }
        return selected_nodes, summary
