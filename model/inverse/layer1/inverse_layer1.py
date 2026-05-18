import copy
import numpy as np
import torch
import random
from typing import Any, List, Optional, Dict, Tuple, Set
from collections import Counter, defaultdict

from ...shared.coarse_graph import E_SU, NUM_SU_TYPES
from ...shared.inverse_common import (
    _NodeV3, HOP1_PORT_COMBINATIONS, SU_FIXED_CONNECTIONS,
    SPECIAL_DEGREE_PRIORS,
    allocate_special_degree_counts,
    rebuild_su19_partition_meta,
    SU_EXTERNAL_CONNECTIONS,
    format_port_patterns_debug,
    get_effective_nodes_element_vector,
    is_mode_specific_neighbor_forbidden,
    select_port_patterns_for_degree,
    validate_connection,
    violates_special_d3_terminal_limit,
)
from ..hop1_adjuster import (
    can_match_ports_exact,
    can_match_ports_partial,
    hop1_counter_to_multiset,
    motif_penalty_alpha,
    build_motif_usage_from_pairs,
)
from .layer1_eval import Layer1NmrEvaluator
from .layer1_refiner import Layer1Refiner


DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS = 3


class Layer1Assigner:
    """Layer1的1-hop分配器"""
    
    def __init__(self,
                 device: str = 'cpu',
                 E_SU_tensor: torch.Tensor = None,
                 layer0_estimator=None,
                 intensity_scale: float = 1.0,
                 unit_peak_intensity: bool = True,
                 deterministic: bool = True):
        """
        初始化Layer1分配器
        """
        self.device = device
        self.E_SU = E_SU_tensor.to(device) if E_SU_tensor is not None else E_SU.to(device)
        self.intensity_scale = float(intensity_scale)
        self.unit_peak_intensity = bool(unit_peak_intensity)
        self.deterministic = bool(deterministic)
        self._build_variant = 0
        self._current_E_target: Optional[torch.Tensor] = None
        self.layer0_estimator = layer0_estimator
        self._nmr_evaluator = Layer1NmrEvaluator(
            device=self.device,
            E_SU_tensor=self.E_SU,
            intensity_scale=self.intensity_scale,
            unit_peak_intensity=self.unit_peak_intensity,
        )
        self._refiner = Layer1Refiner(
            device=self.device,
            evaluator=self._nmr_evaluator,
        )
        self._runtime_nodes_ref: Optional[List[_NodeV3]] = None
        self._node_ids_by_su: Dict[int, List[int]] = {}
        self._empty_ids_by_su: Dict[int, Set[int]] = {}
        self._incomplete_ids_by_su: Dict[int, Set[int]] = {}
        self._available_ids_by_su: Dict[int, Set[int]] = {}
        self._port_pattern_cache: Dict[Tuple[int, Optional[int]], Any] = {}
        self._edge_semantics_cache: Dict[Tuple[int, int, Optional[int]], bool] = {}
        self._can_add_hop1_cache: Dict[Tuple[Any, Any], bool] = {}

    def _reset_layer1_search_caches(self) -> None:
        self._edge_semantics_cache = {}
        self._can_add_hop1_cache = {}

    def _runtime_index_matches(self, nodes: List[_NodeV3]) -> bool:
        return nodes is self._runtime_nodes_ref

    def _refresh_runtime_node_entry(self, nodes: List[_NodeV3], node_id: int) -> None:
        if not self._runtime_index_matches(nodes):
            return
        if int(node_id) < 0 or int(node_id) >= len(nodes):
            return
        node = nodes[int(node_id)]
        su = int(getattr(node, 'su_type', -1))
        if su < 0:
            return

        empty_bucket = self._empty_ids_by_su.setdefault(int(su), set())
        incomplete_bucket = self._incomplete_ids_by_su.setdefault(int(su), set())
        available_bucket = self._available_ids_by_su.setdefault(int(su), set())
        empty_bucket.discard(int(node_id))
        incomplete_bucket.discard(int(node_id))
        available_bucket.discard(int(node_id))

        if bool(node.is_hop1_empty()):
            empty_bucket.add(int(node_id))
        if (not bool(node.is_hop1_empty())) and (not bool(node.is_hop1_complete())):
            incomplete_bucket.add(int(node_id))
        if int(node.remaining_hop1_slots()) > 0:
            available_bucket.add(int(node_id))

    def _init_runtime_node_index(self, nodes: List[_NodeV3]) -> None:
        self._reset_layer1_search_caches()
        self._runtime_nodes_ref = nodes
        ids_by_su: Dict[int, List[int]] = defaultdict(list)
        self._empty_ids_by_su = defaultdict(set)
        self._incomplete_ids_by_su = defaultdict(set)
        self._available_ids_by_su = defaultdict(set)
        for node in nodes:
            nid = int(getattr(node, 'global_id', -1))
            su = int(getattr(node, 'su_type', -1))
            if nid < 0 or su < 0:
                continue
            ids_by_su[int(su)].append(int(nid))
        self._node_ids_by_su = {
            int(su): sorted(int(x) for x in ids)
            for su, ids in ids_by_su.items()
        }
        for nid in range(len(nodes)):
            self._refresh_runtime_node_entry(nodes, int(nid))

    def _stable_node_order(self, nodes: List[_NodeV3]) -> List[_NodeV3]:
        ordered = list(nodes)
        ordered.sort(key=lambda n: (int(getattr(n, 'global_id', -1)), int(getattr(n, 'su_type', -1))))
        return ordered

    def _maybe_shuffle_nodes(self, nodes: List[_NodeV3], salt: int = 0) -> List[_NodeV3]:
        ordered = list(nodes)
        if bool(self.deterministic):
            ordered = self._stable_node_order(ordered)
            if ordered:
                offset = int(self._build_variant + int(salt)) % len(ordered)
                if offset > 0:
                    ordered = ordered[offset:] + ordered[:offset]
            return ordered
        random.shuffle(ordered)
        return ordered
    
    def _histogram_from_nodes(self, nodes: List[_NodeV3]) -> torch.Tensor:
        """从节点列表构建SU直方图"""
        H = torch.zeros(NUM_SU_TYPES, dtype=torch.float, device=self.device)
        for n in nodes:
            try:
                su = int(n.su_type)
                if 0 <= su < NUM_SU_TYPES:
                    H[su] += 1
            except Exception as e:
                import logging
                logging.warning(f"Failed to count SU for node {getattr(n, 'global_id', '?')}: {e}")
        return H

    def _get_special_degree_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[int, int]]:
        raw_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_meta = dict(layer0_meta.get('special_degree_meta', {}) or {})
        except Exception:
            raw_meta = {}

        H_cpu = H_init.detach().cpu().long()
        meta: Dict[int, Dict[int, int]] = {}
        for su_type, priors in SPECIAL_DEGREE_PRIORS.items():
            total = int(H_cpu[int(su_type)].item()) if int(H_cpu.numel()) > int(su_type) else 0
            src = raw_meta.get(int(su_type), raw_meta.get(str(int(su_type)), {}))
            degree_counts: Dict[int, int] = {}
            if isinstance(src, dict):
                for degree in priors.keys():
                    degree_counts[int(degree)] = max(
                        0,
                        int(src.get(int(degree), src.get(str(int(degree)), 0)) or 0),
                    )
            if sum(degree_counts.values()) != int(total):
                degree_counts = allocate_special_degree_counts(int(total), priors)
            meta[int(su_type)] = {
                int(degree): int(cnt)
                for degree, cnt in degree_counts.items()
            }
        return meta

    def _get_special_partition_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[str, Dict[int, int]]]:
        raw_meta = {}
        layer0_meta = {}
        raw_degree_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_meta = dict(layer0_meta.get('special_partition_meta', {}) or {})
            raw_degree_meta = dict(layer0_meta.get('special_degree_meta', {}) or {})
        except Exception:
            raw_meta = {}
            layer0_meta = {}
            raw_degree_meta = {}

        H_cpu = H_init.detach().cpu().long()
        total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
        if int(total_19) <= 0:
            return {}

        src_19 = dict(raw_meta.get(19, raw_meta.get('19', {})) or {})
        raw_degree_19 = dict(raw_degree_meta.get(19, raw_degree_meta.get('19', {})) or {})
        if int(sum(int(raw_degree_19.get(int(deg), raw_degree_19.get(str(int(deg)), 0)) or 0) for deg in [1, 2, 3])) != int(total_19):
            raw_degree_19 = allocate_special_degree_counts(int(total_19), SPECIAL_DEGREE_PRIORS[19])
        o_base_19 = max(0, min(int(layer0_meta.get('o_base_19', total_19)), int(total_19)))
        s_reserved_19 = max(0, min(int(layer0_meta.get('s_reserved_19', max(0, total_19 - o_base_19))), int(total_19 - o_base_19)))
        if int(o_base_19 + s_reserved_19) < int(total_19):
            o_base_19 += int(total_19 - (o_base_19 + s_reserved_19))
        rebuilt = rebuild_su19_partition_meta(
            total_19=int(total_19),
            o_base_19=int(o_base_19),
            s_reserved_19=int(s_reserved_19),
            special_degree_meta_19={
                int(deg): max(0, int(raw_degree_19.get(int(deg), raw_degree_19.get(str(int(deg)), 0)) or 0))
                for deg in [1, 2, 3]
            },
            existing_partition_meta_19=src_19,
        )
        return {
            19: {
                'ether': {int(deg): max(0, int(rebuilt.get('ether', {}).get(int(deg), 0) or 0)) for deg in [1, 2, 3]},
                'thio': {int(deg): max(0, int(rebuilt.get('thio', {}).get(int(deg), 0) or 0)) for deg in [1, 2, 3]},
            }
        }

    def _get_special_anchor_mode_meta(self, H_init: torch.Tensor) -> Dict[int, Dict[str, Dict[int, int]]]:
        layer0_meta = {}
        raw_mode_meta = {}
        try:
            layer0_meta = dict(getattr(self.layer0_estimator, 'fixed_partition_meta', {}) or {})
            raw_mode_meta = dict(layer0_meta.get('special_anchor_mode_meta', {}) or {})
        except Exception:
            layer0_meta = {}
            raw_mode_meta = {}

        H_cpu = H_init.detach().cpu().long()
        degree_meta = self._get_special_degree_meta(H_init)
        partition_meta = self._get_special_partition_meta(H_init)
        out: Dict[int, Dict[str, Dict[int, int]]] = {}

        total_19 = int(H_cpu[19].item()) if int(H_cpu.numel()) > 19 else 0
        if int(total_19) > 0:
            ether_counts = dict((partition_meta.get(19, {}) or {}).get('ether', {}) or {})
            thio_counts = dict((partition_meta.get(19, {}) or {}).get('thio', {}) or {})
            raw_19 = dict(raw_mode_meta.get(19, raw_mode_meta.get('19', {})) or {})
            ether_double_src = dict(raw_19.get('ether_double', {}) or {})
            thio_double_src = dict(raw_19.get('thio_double', {}) or {})
            ether_double = {
                int(deg): max(0, int(ether_double_src.get(int(deg), ether_double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            thio_double = {
                int(deg): max(0, int(thio_double_src.get(int(deg), thio_double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            for deg in [2, 3]:
                ether_double[int(deg)] = min(int(ether_double[int(deg)]), int(ether_counts.get(int(deg), 0)))
                thio_double[int(deg)] = min(int(thio_double[int(deg)]), int(thio_counts.get(int(deg), 0)))
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
            raw_20 = dict(raw_mode_meta.get(20, raw_mode_meta.get('20', {})) or {})
            total_counts_20 = {
                int(deg): max(0, int(dict(degree_meta.get(20, {}) or {}).get(int(deg), 0)))
                for deg in [1, 2, 3]
            }
            double_src = dict(raw_20.get('double', {}) or {})
            double_counts = {
                int(deg): max(0, int(double_src.get(int(deg), double_src.get(str(int(deg)), 0)) or 0))
                for deg in [2, 3]
            }
            for deg in [2, 3]:
                double_counts[int(deg)] = min(int(double_counts[int(deg)]), int(total_counts_20.get(int(deg), 0)))
            required_20_edges = int(max(
                0,
                int(H_cpu[0].item()) + 2 * int(H_cpu[27].item()) - int(H_cpu[6].item()),
            )) if int(H_cpu.numel()) > 27 else int(total_20)
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
        total_21 = int(H_cpu[21].item()) if int(H_cpu.numel()) > 21 else 0
        if int(total_21) > 0:
            raw_21 = dict(raw_mode_meta.get(21, raw_mode_meta.get('21', {})) or {})
            total_counts_21 = dict(degree_meta.get(21, {}) or {})
            raw_single_21 = dict(raw_21.get('single', {}) or {})
            raw_double_21 = dict(raw_21.get('double', {}) or {})
            if int(sum(int(total_counts_21.get(int(deg), 0)) for deg in [2, 3])) > 0:
                single_counts = {
                    int(deg): max(0, int(total_counts_21.get(int(deg), 0)))
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

    @staticmethod
    def _build_special_degree_sequence(su_type: int,
                                       count: int,
                                       special_degree_meta: Dict[int, Dict[int, int]]) -> List[int]:
        if int(su_type) not in SPECIAL_DEGREE_PRIORS:
            return []
        seq: List[int] = []
        degree_map = dict(special_degree_meta.get(int(su_type), {}) or {})
        for degree in sorted(degree_map.keys()):
            seq.extend([int(degree)] * max(0, int(degree_map.get(int(degree), 0))))
        count_i = max(0, int(count))
        if len(seq) < count_i:
            max_degree = max([int(k) for k in degree_map.keys()] or [0])
            if int(max_degree) <= 0:
                return []
            seq.extend([int(max_degree)] * max(0, count_i - len(seq)))
        return seq[:count_i]

    @staticmethod
    def _build_special_partition_sequence_19(count: int,
                                             partition_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, Optional[str]]]:
        seq: List[Tuple[int, Optional[str]]] = []
        part_meta = dict(partition_meta.get(19, {}) or {})
        for part_name in ('thio', 'ether'):
            degree_map = dict(part_meta.get(str(part_name), {}) or {})
            for degree in sorted(degree_map.keys()):
                seq.extend([(int(degree), str(part_name))] * max(0, int(degree_map.get(int(degree), 0))))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_19(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        seq: List[Tuple[int, Optional[str], int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(19, {}) or {})
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('thio_single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'thio', 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('thio_double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'thio', 2, 'double')] * max(0, cnt))
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('ether_single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'ether', 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('ether_double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 'ether', 2, 'double')] * max(0, cnt))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_20(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, int, Optional[str]]]:
        seq: List[Tuple[int, int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(20, {}) or {})
        for degree in [1, 2, 3]:
            cnt = int(dict(mode_meta.get('single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 1, 'single')] * max(0, cnt))
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('double', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 2, 'double')] * max(0, cnt))
        return seq[: max(0, int(count))]

    @staticmethod
    def _build_special_anchor_sequence_21(count: int,
                                          anchor_mode_meta: Dict[int, Dict[str, Dict[int, int]]]) -> List[Tuple[int, int, Optional[str]]]:
        seq: List[Tuple[int, int, Optional[str]]] = []
        mode_meta = dict(anchor_mode_meta.get(21, {}) or {})
        for degree in [2, 3]:
            cnt = int(dict(mode_meta.get('single', {}) or {}).get(int(degree), 0))
            seq.extend([(int(degree), 1, 'single')] * max(0, cnt))
        return seq[: max(0, int(count))]

    def _current_neighbor_types(self, nodes: List[_NodeV3], node: _NodeV3) -> List[int]:
        return [int(nodes[nid].su_type) for nid in node.hop1_ids if 0 <= int(nid) < len(nodes)]

    @staticmethod
    def _node_target_degree(node: _NodeV3) -> Optional[int]:
        try:
            val = getattr(node, 'target_hop1_degree', None)
            return int(val) if val is not None else None
        except Exception:
            return None

    def _port_sets_for_node(self, node: _NodeV3):
        su_i = int(getattr(node, 'su_type', -1))
        target_degree = self._node_target_degree(node)
        key = (int(su_i), target_degree)
        if key not in self._port_pattern_cache:
            port_sets = HOP1_PORT_COMBINATIONS.get(int(su_i))
            self._port_pattern_cache[key] = select_port_patterns_for_degree(port_sets, target_degree)
        return self._port_pattern_cache.get(key)

    def _can_add_state_signature(self, nodes: List[_NodeV3], node: _NodeV3) -> Tuple[Any, ...]:
        neighbor_ids = tuple(sorted(int(x) for x in list(getattr(node, 'hop1_ids', []) or [])))
        neighbor_types = []
        for nid in neighbor_ids:
            if 0 <= int(nid) < len(nodes):
                neighbor_types.append(int(getattr(nodes[int(nid)], 'su_type', -1)))
        return (
            int(getattr(node, 'global_id', -1)),
            int(getattr(node, 'su_type', -1)),
            self._node_target_degree(node),
            self._node_anchor_partition(node),
            self._node_fixed_anchor_target(node),
            self._node_anchor_mode(node),
            int(node.remaining_hop1_slots()),
            neighbor_ids,
            tuple(sorted(neighbor_types)),
        )

    @staticmethod
    def _node_anchor_partition(node: _NodeV3) -> Optional[str]:
        try:
            val = getattr(node, 'special_anchor_partition', None)
            return str(val) if val is not None else None
        except Exception:
            return None

    @staticmethod
    def _node_anchor_mode(node: _NodeV3) -> Optional[str]:
        try:
            val = getattr(node, 'special_anchor_mode', None)
            if val is None:
                return None
            mode = str(val)
            if str(mode).startswith('double'):
                return 'double'
            if str(mode).startswith('single'):
                return 'single'
            if str(mode).startswith('thio'):
                return 'thio'
            return mode
        except Exception:
            return None

    @staticmethod
    def _node_fixed_anchor_target(node: _NodeV3) -> int:
        try:
            val = getattr(node, 'target_fixed_anchor_count', None)
            if val is not None:
                return max(0, int(val))
        except Exception:
            pass
        try:
            mode = str(getattr(node, 'special_anchor_mode', None) or '')
            if str(mode).startswith('double'):
                return 2
            if str(mode) in {'single', 'thio'}:
                return 1
        except Exception:
            pass
        su_i = int(getattr(node, 'su_type', -1))
        if int(su_i) == 19:
            if Layer1Assigner._node_anchor_partition(node) in {'ether', 'thio'}:
                return 1
        if int(su_i) == 20:
            return 1
        if int(su_i) == 21:
            return 1
        if int(su_i) in set(int(x) for x in SU_EXTERNAL_CONNECTIONS.keys()):
            return 1
        return 0

    def _is_su19_partition_compatible(self, node: _NodeV3, neighbor_su: int) -> bool:
        su_i = int(getattr(node, 'su_type', -1))
        nb_i = int(neighbor_su)
        if int(su_i) != 19:
            return True
        if int(nb_i) not in {2, 28, 29, 31}:
            return True
        part = self._node_anchor_partition(node)
        if str(part) == 'thio':
            return int(nb_i) == 31
        if str(part) == 'ether':
            return int(nb_i) in {2, 28, 29}
        return False

    def _count_partition_nodes(self, nodes: List[_NodeV3], su_type: int, partition: Optional[str]) -> int:
        part_txt = str(partition) if partition is not None else None
        total = 0
        for node in nodes:
            if int(getattr(node, 'su_type', -1)) != int(su_type):
                continue
            if part_txt is None:
                if self._node_anchor_partition(node) is None:
                    total += 1
                continue
            if self._node_anchor_partition(node) == str(part_txt):
                total += 1
        return int(total)

    def _sort_special_degree_nodes(self, nodes: List[_NodeV3]) -> List[_NodeV3]:
        return sorted(
            list(nodes),
            key=lambda n: (
                0 if self._node_anchor_partition(n) == 'thio' else 1,
                self._node_target_degree(n) if self._node_target_degree(n) is not None else 99,
                int(getattr(n, 'global_id', 0)),
            ),
        )

    def _promote_fixed_anchor_count(self,
                                    nodes: List[_NodeV3],
                                    su_type: int,
                                    node_ids: List[int]) -> None:
        promote_ids = set(int(x) for x in list(node_ids or []))
        for node in list(nodes or []):
            if int(getattr(node, 'global_id', -1)) not in promote_ids:
                continue
            if int(getattr(node, 'su_type', -1)) != int(su_type):
                continue
            try:
                node.target_fixed_anchor_count = 2
                node.init_target_fixed_anchor_count = 2
                deg_i = int(self._node_target_degree(node) or 0)
                if int(su_type) == 19:
                    node.special_anchor_mode = f"double_d{int(deg_i)}" if int(deg_i) in {2, 3} else 'double'
                elif int(su_type) == 20:
                    node.special_anchor_mode = f"double_d{int(deg_i)}" if int(deg_i) in {2, 3} else 'double'
            except Exception:
                continue

    def _required_external_candidates_for_node(self, node: _NodeV3) -> List[int]:
        su_i = int(getattr(node, 'su_type', -1))
        if int(su_i) == 19:
            part = self._node_anchor_partition(node)
            deg = self._node_target_degree(node)
            if str(part) == 'thio':
                return [31]
            if str(part) == 'ether':
                if int(deg or 0) == 1:
                    return [2, 29]
                return [2, 28, 29]
            return []
        return [int(x) for x in list(SU_EXTERNAL_CONNECTIONS.get(int(su_i), []) or [])]

    def _has_required_external_for_node(self, node: _NodeV3, nodes: List[_NodeV3]) -> bool:
        required = set(int(x) for x in self._required_external_candidates_for_node(node))
        if not required:
            return True
        target_count = max(1, int(self._node_fixed_anchor_target(node)))
        current_count = int(sum(1 for x in self._current_neighbor_types(nodes, node) if int(x) in required))
        return int(current_count) >= int(target_count)

    def _has_pending_required_external(self, node: _NodeV3, nodes: List[_NodeV3]) -> bool:
        required = self._required_external_candidates_for_node(node)
        if not required:
            return False
        return not self._has_required_external_for_node(node, nodes)

    def _required_external_count_for_node(self, node: _NodeV3, nodes: List[_NodeV3]) -> int:
        required = set(int(x) for x in self._required_external_candidates_for_node(node))
        if not required:
            return 0
        return int(sum(1 for x in self._current_neighbor_types(nodes, node) if int(x) in required))

    def _fixed_target_rank(self,
                           center: _NodeV3,
                           candidate: _NodeV3,
                           priority_list: List[int]) -> Tuple[int, int, int, int]:
        center_su = int(getattr(center, 'su_type', -1))
        cand_su = int(getattr(candidate, 'su_type', -1))
        cand_deg = self._node_target_degree(candidate)
        try:
            pri_idx = int(priority_list.index(int(cand_su)))
        except ValueError:
            pri_idx = int(len(priority_list))

        fixed_pref = 50
        if center_su in {2, 29}:
            if cand_su == 19:
                if self._node_anchor_partition(candidate) != 'ether':
                    fixed_pref = 98
                else:
                    fixed_pref = {1: 0, 2: 2, 3: 3}.get(int(cand_deg or 0), 9)
            elif cand_su == 5:
                fixed_pref = 1
        elif center_su == 28:
            if cand_su == 5:
                fixed_pref = 0
            elif cand_su == 19:
                if self._node_anchor_partition(candidate) != 'ether':
                    fixed_pref = 98
                else:
                    fixed_pref = {2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su == 31 and cand_su == 19:
            if self._node_anchor_partition(candidate) != 'thio':
                fixed_pref = 98
            else:
                fixed_pref = {1: 0, 2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su in {0, 27} and cand_su == 20:
            fixed_pref = {1: 0, 2: 1, 3: 2}.get(int(cand_deg or 0), 9)
        elif center_su == 32 and cand_su == 21:
            fixed_pref = {2: 0, 3: 1}.get(int(cand_deg or 0), 9)

        return (
            int(fixed_pref),
            int(pri_idx),
            int(candidate.remaining_hop1_slots()),
            int(getattr(candidate, 'global_id', 0)),
        )

    def _choose_fixed_candidate(self,
                                center: _NodeV3,
                                candidates: List[_NodeV3],
                                priority_list: List[int],
                                nodes: Optional[List[_NodeV3]] = None) -> Optional[_NodeV3]:
        filtered = [
            n for n in candidates
            if n.global_id != center.global_id
            and n.remaining_hop1_slots() > 0
            and n.global_id not in center.hop1_ids
        ]
        if not filtered:
            return None
        if nodes is not None:
            filtered = [n for n in filtered if self._can_add_hop1_connection(nodes, center, n)]
            if not filtered:
                return None
        ranked = sorted(
            filtered,
            key=lambda n: self._fixed_target_rank(center, n, priority_list),
        )
        return ranked[0] if ranked else None

    def _violates_mode_specific_rule(self, node: _NodeV3, neighbor_su: int) -> bool:
        return bool(
            is_mode_specific_neighbor_forbidden(
                int(getattr(node, 'su_type', -1)),
                self._node_target_degree(node),
                int(neighbor_su),
            )
        )

    def _violates_special_d3_terminal_rule(self,
                                           nodes: List[_NodeV3],
                                           node: _NodeV3,
                                           proposed_neighbor_su: Optional[int] = None) -> bool:
        neighbor_types = [int(x) for x in self._current_neighbor_types(nodes, node)]
        if proposed_neighbor_su is not None:
            neighbor_types.append(int(proposed_neighbor_su))
        return bool(
            violates_special_d3_terminal_limit(
                int(getattr(node, 'su_type', -1)),
                self._node_target_degree(node),
                neighbor_types,
            )
        )

    def _collect_fixed_connection_markers(self, nodes: List[_NodeV3]) -> Dict[str, Set[int]]:
        markers: Dict[str, Set[Any]] = {
            'carbonyl_9_ids': set(),
            'ether_5_ids': set(),
            'ether_19_ids': set(),
            'ether_5_edges': set(),
            'ether_19_edges': set(),
            'sulfur_7_ids': set(),
            'sulfur_19_ids': set(),
            'sulfur_7_edges': set(),
            'sulfur_19_edges': set(),
            'amine_6_ids': set(),
            'amine_20_ids': set(),
            'amine_6_edges': set(),
            'amine_20_edges': set(),
            'halogen_8_ids': set(),
            'halogen_21_ids': set(),
            'halogen_8_edges': set(),
            'halogen_21_edges': set(),
            'invalid_su19_31_ids': set(),
            'invalid_su19_o_ids': set(),
            'unpartitioned_su19_external_ids': set(),
        }
        for node in nodes:
            gid = int(getattr(node, 'global_id', -1))
            su = int(getattr(node, 'su_type', -1))
            neighbor_ids = [int(nid) for nid in list(getattr(node, 'hop1_ids', []) or []) if 0 <= int(nid) < len(nodes)]
            neighbor_types = [int(nodes[int(nid)].su_type) for nid in neighbor_ids]
            neigh = set(int(x) for x in neighbor_types)
            if su == 9 and any(int(x) in {0, 1, 2, 3} for x in neigh):
                markers['carbonyl_9_ids'].add(gid)
            elif su == 5:
                used_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {2, 28, 29}]
                if used_nb_ids:
                    markers['ether_5_ids'].add(gid)
                for nb_id in used_nb_ids:
                    markers['ether_5_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 19:
                part = self._node_anchor_partition(node)
                sulfur_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 31]
                if sulfur_nb_ids:
                    if str(part) == 'thio':
                        markers['sulfur_19_ids'].add(gid)
                        for nb_id in sulfur_nb_ids:
                            markers['sulfur_19_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
                    elif str(part) == 'ether':
                        markers['invalid_su19_31_ids'].add(gid)
                    else:
                        markers['invalid_su19_31_ids'].add(gid)
                        markers['unpartitioned_su19_external_ids'].add(gid)
                oxygen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {2, 28, 29}]
                if oxygen_nb_ids:
                    if str(part) == 'ether':
                        markers['ether_19_ids'].add(gid)
                        for nb_id in oxygen_nb_ids:
                            markers['ether_19_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
                    elif str(part) == 'thio':
                        markers['invalid_su19_o_ids'].add(gid)
                    else:
                        markers['invalid_su19_o_ids'].add(gid)
                        markers['unpartitioned_su19_external_ids'].add(gid)
            elif su == 7:
                sulfur_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 31]
                if sulfur_nb_ids:
                    markers['sulfur_7_ids'].add(gid)
                for nb_id in sulfur_nb_ids:
                    markers['sulfur_7_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 6:
                amine_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {0, 27}]
                if amine_nb_ids:
                    markers['amine_6_ids'].add(gid)
                for nb_id in amine_nb_ids:
                    markers['amine_6_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 20:
                amine_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) in {0, 27}]
                if amine_nb_ids:
                    markers['amine_20_ids'].add(gid)
                for nb_id in amine_nb_ids:
                    markers['amine_20_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 8:
                halogen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 32]
                if halogen_nb_ids:
                    markers['halogen_8_ids'].add(gid)
                for nb_id in halogen_nb_ids:
                    markers['halogen_8_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
            elif su == 21:
                halogen_nb_ids = [int(nid) for nid in neighbor_ids if int(nodes[int(nid)].su_type) == 32]
                if halogen_nb_ids:
                    markers['halogen_21_ids'].add(gid)
                for nb_id in halogen_nb_ids:
                    markers['halogen_21_edges'].add(tuple(sorted((int(gid), int(nb_id)))))
        return markers

    def _fixed_anchor_invariant_errors(self,
                                       nodes: List[_NodeV3],
                                       check_thio: bool = True,
                                       check_ether: bool = True,
                                       check_amine: bool = True,
                                       check_halogen: bool = True) -> List[str]:
        errors: List[str] = []
        H_actual = self._histogram_from_nodes(nodes).detach().cpu().long()
        markers = self._collect_fixed_connection_markers(nodes)

        invalid_31 = sorted(int(x) for x in set(markers.get('invalid_su19_31_ids', set()) or set()))
        invalid_o = sorted(int(x) for x in set(markers.get('invalid_su19_o_ids', set()) or set()))
        unpartitioned = sorted(int(x) for x in set(markers.get('unpartitioned_su19_external_ids', set()) or set()))
        if invalid_31:
            errors.append(f"存在非thio的SU19连接到SU31: ids={invalid_31}")
        if invalid_o:
            errors.append(f"存在非ether的SU19连接到O锚点(2/28/29): ids={invalid_o}")
        if unpartitioned:
            errors.append(f"存在未分区的SU19参与固定外接锚点: ids={unpartitioned}")

        if bool(check_thio):
            n31 = int(H_actual[31].item()) if int(H_actual.numel()) > 31 else 0
            n7 = int(H_actual[7].item()) if int(H_actual.numel()) > 7 else 0
            n19_thio = self._count_partition_nodes(nodes, 19, 'thio')
            expected_thio = int(2 * n31)
            used_7 = int(len(markers.get('sulfur_7_edges', set()) or set()))
            used_19 = int(len(markers.get('sulfur_19_edges', set()) or set()))
            target_19 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 19 and self._node_anchor_partition(node) == 'thio'
            ))
            if int(n7 + target_19) != int(expected_thio):
                errors.append(
                    f"thio固定池数量不匹配: SU7({n7}) + SU19(thio_edges)({target_19}) != 2*SU31({expected_thio})"
                )
            if int(used_7) != int(n7):
                errors.append(f"SU7外接31边数不匹配: used={used_7} vs total={n7}")
            if int(used_19) != int(target_19):
                errors.append(f"SU19(thio)外接31边数不匹配: used={used_19} vs target={target_19}")
            if int(used_7 + used_19) != int(expected_thio):
                errors.append(
                    f"固定连接总量不匹配: SU7(连31)={used_7}, SU19(thio)(连31)={used_19}, SU31需求={expected_thio}"
                )

        if bool(check_ether):
            n2 = int(H_actual[2].item()) if int(H_actual.numel()) > 2 else 0
            n28 = int(H_actual[28].item()) if int(H_actual.numel()) > 28 else 0
            n29 = int(H_actual[29].item()) if int(H_actual.numel()) > 29 else 0
            n5 = int(H_actual[5].item()) if int(H_actual.numel()) > 5 else 0
            n19_ether = self._count_partition_nodes(nodes, 19, 'ether')
            expected_ether = int(n2 + n28 + 2 * n29)
            used_5 = int(len(markers.get('ether_5_edges', set()) or set()))
            used_19 = int(len(markers.get('ether_19_edges', set()) or set()))
            target_19 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 19 and self._node_anchor_partition(node) == 'ether'
            ))
            if int(used_5) != int(n5):
                errors.append(
                    f"SU5外接O锚点边数不匹配: used={used_5} vs total={n5}"
                )
            if int(used_19) != int(target_19):
                errors.append(f"SU19(ether)外接O锚点边数不匹配: used={used_19} vs target={target_19}")
            if int(used_5 + used_19) != int(expected_ether):
                errors.append(
                    f"固定连接总量不匹配: SU5(连O)={used_5}, SU19(ether)(连O)={used_19}, SU2+SU28+2*SU29={expected_ether}"
                )

        if bool(check_amine):
            n0 = int(H_actual[0].item()) if int(H_actual.numel()) > 0 else 0
            n27 = int(H_actual[27].item()) if int(H_actual.numel()) > 27 else 0
            n6 = int(H_actual[6].item()) if int(H_actual.numel()) > 6 else 0
            expected_amine = int(n0 + 2 * n27)
            used_6 = int(len(markers.get('amine_6_edges', set()) or set()))
            used_20 = int(len(markers.get('amine_20_edges', set()) or set()))
            target_20 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 20
            ))
            if int(used_6) != int(n6):
                errors.append(f"SU6外接N锚点边数不匹配: used={used_6} vs total={n6}")
            if int(used_20) != int(target_20):
                errors.append(f"SU20外接N锚点边数不匹配: used={used_20} vs target={target_20}")
            if int(used_6 + used_20) != int(expected_amine):
                errors.append(
                    f"固定连接总量不匹配: SU6(连N)={used_6}, SU20(连N)={used_20}, SU0+2*SU27={expected_amine}"
                )

        if bool(check_halogen):
            n32 = int(H_actual[32].item()) if int(H_actual.numel()) > 32 else 0
            n8 = int(H_actual[8].item()) if int(H_actual.numel()) > 8 else 0
            expected_halogen = int(n32)
            used_8 = int(len(markers.get('halogen_8_edges', set()) or set()))
            used_21 = int(len(markers.get('halogen_21_edges', set()) or set()))
            target_21 = int(sum(
                int(self._node_fixed_anchor_target(node))
                for node in nodes
                if int(getattr(node, 'su_type', -1)) == 21
            ))
            if int(used_8) != int(n8):
                errors.append(f"SU8外接X锚点边数不匹配: used={used_8} vs total={n8}")
            if int(used_21) != int(target_21):
                errors.append(f"SU21外接X锚点边数不匹配: used={used_21} vs target={target_21}")
            if int(used_8 + used_21) != int(expected_halogen):
                errors.append(
                    f"固定连接总量不匹配: SU8(连X)={used_8}, SU21(连X)={used_21}, SU32={expected_halogen}"
                )

        return errors

    def _raise_if_fixed_anchor_invariants_fail(self,
                                               nodes: List[_NodeV3],
                                               stage: str,
                                               check_thio: bool = True,
                                               check_ether: bool = True,
                                               check_amine: bool = True,
                                               check_halogen: bool = True) -> None:
        errors = self._fixed_anchor_invariant_errors(
            nodes,
            check_thio=bool(check_thio),
            check_ether=bool(check_ether),
            check_amine=bool(check_amine),
            check_halogen=bool(check_halogen),
        )
        if errors:
            msg = "; ".join(str(x) for x in errors[:6])
            raise RuntimeError(f"[{stage}] 固定锚点约束失败: {msg}")

    def _clear_all_edges_incident_to_set(self,
                                         nodes: List[_NodeV3],
                                         node_ids: List[int]) -> None:
        node_set = set(int(x) for x in list(node_ids or []))
        if not node_set:
            return
        seen_edges: Set[Tuple[int, int]] = set()
        for nid in sorted(node_set):
            if int(nid) < 0 or int(nid) >= len(nodes):
                continue
            node = nodes[int(nid)]
            for nb_id in list(getattr(node, 'hop1_ids', []) or []):
                nb_i = int(nb_id)
                edge = tuple(sorted((int(nid), int(nb_i))))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                self._remove_bidirectional_hop1_with_force(
                    nodes,
                    int(edge[0]),
                    int(edge[1]),
                    force=True,
                )

    def _solve_exact_fixed_anchor_matching(self,
                                           nodes: List[_NodeV3],
                                           owner_slots: List[Dict[str, int]],
                                           targets: List[_NodeV3],
                                           compat_fn) -> Optional[List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        used_slots = [False] * len(owner_slots)
        owner_d1_usage: Dict[int, int] = defaultdict(int)
        target_ids = [int(getattr(n, 'global_id', -1)) for n in list(targets or [])]

        def _target_priority(node: _NodeV3) -> Tuple[int, int, int, int]:
            su_i = int(getattr(node, 'su_type', -1))
            deg_i = int(self._node_target_degree(node) or 0)
            if int(su_i) == 19 and int(deg_i) == 1:
                rank = 0
            elif int(su_i) == 7:
                rank = 1
            elif int(su_i) == 19:
                rank = 2
            elif int(su_i) == 5:
                rank = 3
            else:
                rank = 4
            return (int(rank), int(su_i), int(deg_i), int(getattr(node, 'global_id', 0)))

        def _slot_priority(slot: Dict[str, int]) -> Tuple[int, int, int]:
            owner = work_nodes[int(slot['owner_id'])]
            owner_su = int(getattr(owner, 'su_type', -1))
            if int(owner_su) == 28:
                rank = 0
            elif int(owner_su) == 29:
                rank = 1
            elif int(owner_su) == 2:
                rank = 2
            else:
                rank = 3
            return (int(rank), int(slot['owner_id']), int(slot.get('slot_index', 0)))

        def _candidate_slots(target_id: int) -> List[int]:
            target = work_nodes[int(target_id)]
            out: List[int] = []
            for slot_idx, slot in enumerate(owner_slots):
                if bool(used_slots[slot_idx]):
                    continue
                owner = work_nodes[int(slot['owner_id'])]
                if not compat_fn(work_nodes, owner, target, slot, owner_d1_usage):
                    continue
                out.append(int(slot_idx))
            out.sort(key=lambda idx: _slot_priority(owner_slots[int(idx)]))
            return out

        def _dfs(remaining_target_ids: List[int]) -> bool:
            if not remaining_target_ids:
                return True

            best_target_id: Optional[int] = None
            best_options: Optional[List[int]] = None
            for tid in sorted(remaining_target_ids, key=lambda x: _target_priority(work_nodes[int(x)])):
                options = _candidate_slots(int(tid))
                if not options:
                    return False
                if best_options is None or len(options) < len(best_options):
                    best_target_id = int(tid)
                    best_options = list(options)
                    if len(best_options) <= 1:
                        break

            if best_target_id is None or best_options is None:
                return False

            next_remaining = [int(tid) for tid in remaining_target_ids if int(tid) != int(best_target_id)]
            target = work_nodes[int(best_target_id)]
            target_deg = int(self._node_target_degree(target) or 0)
            for slot_idx in list(best_options):
                slot = owner_slots[int(slot_idx)]
                owner_id = int(slot['owner_id'])
                owner = work_nodes[int(owner_id)]
                if not self._can_add_hop1_connection(work_nodes, owner, target):
                    continue
                self._add_bidirectional_hop1(work_nodes, int(owner_id), int(best_target_id), lock=True)
                if int(best_target_id) not in set(int(x) for x in list(getattr(work_nodes[int(owner_id)], 'hop1_ids', []) or [])):
                    continue
                used_slots[int(slot_idx)] = True
                inc_d1 = (
                    int(getattr(owner, 'su_type', -1)) == 29 and
                    int(getattr(target, 'su_type', -1)) == 19 and
                    int(target_deg) == 1
                )
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] += 1
                if _dfs(next_remaining):
                    return True
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] -= 1
                    if int(owner_d1_usage[int(owner_id)]) <= 0:
                        owner_d1_usage.pop(int(owner_id), None)
                self._remove_bidirectional_hop1_with_force(
                    work_nodes,
                    int(owner_id),
                    int(best_target_id),
                    force=True,
                )
                used_slots[int(slot_idx)] = False
            return False

        ok = _dfs([int(x) for x in target_ids])
        if not ok:
            return None

        target_id_set = set(int(x) for x in target_ids)
        owner_id_set = set(int(slot['owner_id']) for slot in list(owner_slots or []))
        edges: Set[Tuple[int, int]] = set()
        for owner_id in sorted(owner_id_set):
            owner = work_nodes[int(owner_id)]
            for nb_id in list(getattr(owner, 'hop1_ids', []) or []):
                nb_i = int(nb_id)
                if nb_i not in target_id_set:
                    continue
                edges.add(tuple(sorted((int(owner_id), int(nb_i)))))
        return list(sorted(edges))

    def _solve_exact_fixed_anchor_matching_demands(self,
                                                   nodes: List[_NodeV3],
                                                   owner_slots: List[Dict[str, int]],
                                                   target_demands: List[Dict[str, Any]],
                                                   compat_fn) -> Optional[List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        used_slots = [False] * len(owner_slots)
        owner_d1_usage: Dict[int, int] = defaultdict(int)
        demand_indices = [int(i) for i in range(len(target_demands))]

        def _target_priority(demand: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
            node = work_nodes[int(demand['target_id'])]
            su_i = int(getattr(node, 'su_type', -1))
            deg_i = int(self._node_target_degree(node) or 0)
            fixed_i = int(self._node_fixed_anchor_target(node))
            mode_i = self._node_anchor_mode(node)
            if int(su_i) == 19 and str(mode_i) == 'double':
                rank = 0
            elif int(su_i) == 20 and str(mode_i) == 'double':
                rank = 1
            elif int(su_i) == 19 and int(deg_i) == 1:
                rank = 2
            elif int(su_i) == 7:
                rank = 3
            elif int(su_i) == 19:
                rank = 4
            elif int(su_i) == 5:
                rank = 5
            elif int(su_i) == 20:
                rank = 6
            elif int(su_i) == 6:
                rank = 7
            else:
                rank = 8
            return (
                int(rank),
                -int(fixed_i),
                int(su_i),
                int(deg_i),
                int(demand.get('demand_index', 0)),
            )

        def _slot_priority(slot: Dict[str, int]) -> Tuple[int, int, int]:
            owner = work_nodes[int(slot['owner_id'])]
            owner_su = int(getattr(owner, 'su_type', -1))
            if int(owner_su) == 28:
                rank = 0
            elif int(owner_su) == 29:
                rank = 1
            elif int(owner_su) == 2:
                rank = 2
            elif int(owner_su) == 27:
                rank = 3
            elif int(owner_su) == 0:
                rank = 4
            else:
                rank = 5
            return (int(rank), int(slot['owner_id']), int(slot.get('slot_index', 0)))

        def _candidate_slots(demand_idx: int) -> List[int]:
            demand = dict(target_demands[int(demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            out: List[int] = []
            for slot_idx, slot in enumerate(owner_slots):
                if bool(used_slots[slot_idx]):
                    continue
                owner = work_nodes[int(slot['owner_id'])]
                if not compat_fn(work_nodes, owner, target, slot, owner_d1_usage, demand):
                    continue
                out.append(int(slot_idx))
            out.sort(key=lambda idx: _slot_priority(owner_slots[int(idx)]))
            return out

        def _dfs(remaining_demand_indices: List[int]) -> bool:
            if not remaining_demand_indices:
                return True

            best_demand_idx: Optional[int] = None
            best_options: Optional[List[int]] = None
            for idx in sorted(remaining_demand_indices, key=lambda x: _target_priority(target_demands[int(x)])):
                options = _candidate_slots(int(idx))
                if not options:
                    return False
                if best_options is None or len(options) < len(best_options):
                    best_demand_idx = int(idx)
                    best_options = list(options)
                    if len(best_options) <= 1:
                        break

            if best_demand_idx is None or best_options is None:
                return False

            next_remaining = [int(idx) for idx in remaining_demand_indices if int(idx) != int(best_demand_idx)]
            demand = dict(target_demands[int(best_demand_idx)] or {})
            target = work_nodes[int(demand['target_id'])]
            target_deg = int(self._node_target_degree(target) or 0)
            for slot_idx in list(best_options):
                slot = owner_slots[int(slot_idx)]
                owner_id = int(slot['owner_id'])
                owner = work_nodes[int(owner_id)]
                if not self._can_add_hop1_connection(work_nodes, owner, target):
                    continue
                self._add_bidirectional_hop1(work_nodes, int(owner_id), int(demand['target_id']), lock=True)
                if int(demand['target_id']) not in set(int(x) for x in list(getattr(work_nodes[int(owner_id)], 'hop1_ids', []) or [])):
                    continue
                used_slots[int(slot_idx)] = True
                inc_d1 = (
                    int(getattr(owner, 'su_type', -1)) == 29 and
                    int(getattr(target, 'su_type', -1)) == 19 and
                    int(target_deg) == 1
                )
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] += 1
                if _dfs(next_remaining):
                    return True
                if bool(inc_d1):
                    owner_d1_usage[int(owner_id)] -= 1
                    if int(owner_d1_usage[int(owner_id)]) <= 0:
                        owner_d1_usage.pop(int(owner_id), None)
                self._remove_bidirectional_hop1_with_force(
                    work_nodes,
                    int(owner_id),
                    int(demand['target_id']),
                    force=True,
                )
                used_slots[int(slot_idx)] = False
            return False

        ok = _dfs(list(demand_indices))
        if not ok:
            return None

        target_id_set = set(int(d.get('target_id', -1)) for d in list(target_demands or []))
        owner_id_set = set(int(slot['owner_id']) for slot in list(owner_slots or []))
        edges: Set[Tuple[int, int]] = set()
        for owner_id in sorted(owner_id_set):
            owner = work_nodes[int(owner_id)]
            for nb_id in list(getattr(owner, 'hop1_ids', []) or []):
                nb_i = int(nb_id)
                if nb_i not in target_id_set:
                    continue
                edges.add(tuple(sorted((int(owner_id), int(nb_i)))))
        return list(sorted(edges))

    def _bridge_candidate_bias(self, center_su: int, current_neighbors: List[int], cand_su: int) -> float:
        center_su = int(center_su)
        cand_su = int(cand_su)
        current = [int(x) for x in current_neighbors]
        cnt = Counter(current)

        if center_su == 27:
            if cnt[6] > 0 and cand_su == 20:
                return 4.0
            if cnt[20] > 0 and cand_su == 6:
                return 4.0
            if cnt[cand_su] > 0:
                return 0.2

        if center_su == 29:
            if cnt[5] > 0 and cand_su == 19:
                return 4.0
            if cnt[19] > 0 and cand_su == 5:
                return 4.0
            if cnt[cand_su] > 0:
                return 0.15

        if center_su == 31:
            if cand_su == 7:
                if cnt[7] == 0:
                    return 5.0
                return 3.5
            if cand_su == 19:
                if cnt[7] == 0:
                    return 0.25
                return 1.0

        if center_su == 3:
            if cand_su == 9 and cnt[9] > 0:
                return 0.05
            if cnt[9] > 0 and cand_su != 9:
                return 2.5
            if cand_su == 9 and cnt[9] == 0:
                return 1.5

        if center_su == 2:
            if cnt[9] > 0 and cand_su == 19:
                return 3.0
            if cnt[9] > 0 and cand_su == 5:
                return 0.35
            if cand_su == 19 and cnt[19] == 0:
                return 1.75
            if cand_su == 5 and cnt[19] == 0:
                return 0.85

        return 1.0

    def _filter_valid_fixed_targets(self,
                                    nodes: List[_NodeV3],
                                    center: _NodeV3,
                                    targets: List[_NodeV3]) -> List[_NodeV3]:
        valid = []
        for n in targets:
            if int(n.global_id) == int(center.global_id):
                continue
            if int(n.global_id) in center.hop1_ids:
                continue
            if int(n.remaining_hop1_slots()) <= 0:
                continue
            if not self._can_add_hop1_connection(nodes, center, n):
                continue
            valid.append(n)
        return valid

    def _pick_fixed_target(self,
                           nodes: List[_NodeV3],
                           center: _NodeV3,
                           target_pool: List[_NodeV3],
                           priority_list: List[int]) -> Optional[_NodeV3]:
        candidates = self._filter_valid_fixed_targets(nodes, center, target_pool)
        if not candidates:
            return None
        return self._choose_fixed_candidate(center, candidates, priority_list, nodes=nodes)

    def _complete_required_external_anchors(self,
                                            nodes: List[_NodeV3],
                                            su_types: List[int]) -> None:
        self._refiner.complete_required_external_anchors(
            assigner=self,
            nodes=nodes,
            su_types=su_types,
        )

    @staticmethod
    def _print_consistency_errors(errors: List[str], prefix: str, limit: int = 8) -> None:
        if not errors:
            return
        max_show = max(1, int(limit))
        shown = list(errors[:max_show])
        for idx, msg in enumerate(shown, start=1):
            print(f"    [{prefix} #{idx}] {msg}")
        if len(errors) > len(shown):
            print(f"    [{prefix}] ... 其余 {len(errors) - len(shown)} 条省略")

    @staticmethod
    def _is_soft_consistency_error(msg: str) -> bool:
        text = str(msg)
        if ('实际连接度' in text and '目标连接度' in text):
            # 19/20/21 encode mode-specific anchor degrees.  A missing edge
            # changes d1/d2/d3 semantics, so it must not be accepted as a soft
            # tail-gap.
            for special_su in (19, 20, 21):
                if f"(SU{int(special_su)})" in text:
                    return False
            return True
        if '未优先连接SU9' in text:
            return True
        if '(SU11): 缺少必需外接锚点' in text:
            return True
        if '(SU11): 固定锚点边数不足' in text:
            return True
        if '(SU11): 固定锚点边数超过目标上限' in text:
            return True
        return False

    def assess_consistency_acceptance(self,
                                      errors: List[str],
                                      max_soft_errors: int = DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS) -> Tuple[bool, Dict[str, Any]]:
        soft_errors: List[str] = []
        hard_errors: List[str] = []
        for msg in list(errors or []):
            if self._is_soft_consistency_error(str(msg)):
                soft_errors.append(str(msg))
            else:
                hard_errors.append(str(msg))
        accepted = bool(len(hard_errors) == 0 and len(soft_errors) <= int(max_soft_errors))
        return accepted, {
            'accepted': bool(accepted),
            'max_soft_errors': int(max_soft_errors),
            'soft_errors': list(soft_errors),
            'hard_errors': list(hard_errors),
            'n_soft': int(len(soft_errors)),
            'n_hard': int(len(hard_errors)),
        }
    
    def validate_graph_consistency(self, nodes: List[_NodeV3], 
                                    H: torch.Tensor, 
                                    E_target: Optional[torch.Tensor] = None,
                                    verbose: bool = False) -> Tuple[bool, List[str]]:
        """
        验证图的全局一致性
        """
        errors = []
        
        # 1. 检查节点数量
        expected_total = int(H.sum().item())
        actual_total = len(nodes)
        if actual_total != expected_total:
            errors.append(f"节点数量不匹配: 实际{actual_total} vs 预期{expected_total}")
        
        # 2. 检查SU类型分布
        H_actual = self._histogram_from_nodes(nodes)
        for su_type in range(NUM_SU_TYPES):
            expected = int(H[su_type].item())
            actual = int(H_actual[su_type].item())
            if expected != actual:
                errors.append(f"SU{su_type}数量不匹配: 实际{actual} vs 预期{expected}")
        
        # 3. 检查互为1-hop对称性
        for n in nodes:
            for neighbor_id in n.hop1_ids:
                if neighbor_id >= len(nodes):
                    errors.append(f"节点{n.global_id}: hop1_ids包含越界ID {neighbor_id}")
                    continue
                neighbor = nodes[neighbor_id]
                if n.global_id not in neighbor.hop1_ids:
                    errors.append(f"1-hop不对称: {n.global_id}->{neighbor_id} 但反向缺失")
        
        # 4. 检查hop1_su与hop1_ids一致性
        for n in nodes:
            actual_counter = Counter()
            for nid in n.hop1_ids:
                if nid < len(nodes):
                    actual_counter[nodes[nid].su_type] += 1
            
            if actual_counter != n.hop1_su:
                errors.append(f"节点{n.global_id}(SU{n.su_type}): hop1_su={dict(n.hop1_su)} != 实际={dict(actual_counter)}")
        
        # 5. 检查每个节点的内部一致性
        for n in nodes:
            is_valid, node_errors = n.validate_hop1_consistency()
            if not is_valid:
                errors.extend(node_errors)
            target_degree = self._node_target_degree(n)
            if target_degree is not None and int(n.get_hop1_degree()) != int(target_degree):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): 实际连接度{n.get_hop1_degree()} != 目标连接度{target_degree}"
                )
        
        # 6. 检查 per-port 连接规则合规性
        for n in nodes:
            port_sets = self._port_sets_for_node(n)
            if not port_sets:
                continue
            target_degree = self._node_target_degree(n)
            neighbor_types = [int(nodes[nid].su_type) for nid in n.hop1_ids if nid < len(nodes)]
            if n.is_hop1_complete() and not can_match_ports_exact(neighbor_types, port_sets):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): hop1={sorted(neighbor_types)} 不满足端口规则 "
                    f"{format_port_patterns_debug(port_sets)}"
                )
            if int(n.su_type) == 19 and int(target_degree or 0) == 1 and 28 in set(neighbor_types):
                errors.append(
                    f"节点{n.global_id}(SU19, d1): 不允许连接SU28, hop1={sorted(neighbor_types)}"
                )
            if violates_special_d3_terminal_limit(int(n.su_type), target_degree, neighbor_types):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}, d3): 端基类邻居(1/22/28/32)数量超过1, "
                    f"hop1={sorted(neighbor_types)}"
                )

        # 6.5 检查外接结构要求（即使节点未满度，也不能缺失必须存在的外接锚点）
        for n in nodes:
            if int(getattr(n, 'su_type', -1)) == 19 and self._node_anchor_partition(n) not in {'thio', 'ether'}:
                errors.append(
                    f"节点{n.global_id}(SU19): 缺少严格分区标记 special_anchor_partition, part={self._node_anchor_partition(n)}"
                )
            required_external = self._required_external_candidates_for_node(n)
            neighbor_types = [int(nodes[nid].su_type) for nid in n.hop1_ids if nid < len(nodes)]
            if required_external and not self._has_required_external_for_node(n, nodes):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): 缺少必需外接锚点, hop1={sorted(neighbor_types)} | "
                    f"required={sorted(required_external)} part={self._node_anchor_partition(n)}"
                )
            ext_cnt = int(sum(1 for x in neighbor_types if int(x) in set(required_external)))
            target_ext_cnt = int(self._node_fixed_anchor_target(n))
            if required_external and ext_cnt < int(target_ext_cnt):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): 固定锚点边数不足, hop1={sorted(neighbor_types)} | "
                    f"required={sorted(required_external)} part={self._node_anchor_partition(n)} count={ext_cnt} target={target_ext_cnt}"
                )
            if required_external and ext_cnt > int(target_ext_cnt):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): 固定锚点边数超过目标上限, hop1={sorted(neighbor_types)} | "
                    f"required={sorted(required_external)} part={self._node_anchor_partition(n)} count={ext_cnt} target={target_ext_cnt}"
                )

        # 6.6 检查固定连接类型的全局数量关系（按真实图上的固定锚点使用统计）
        try:
            n0 = int(H_actual[0].item()) if int(H_actual.numel()) > 0 else 0
            n27 = int(H_actual[27].item()) if int(H_actual.numel()) > 27 else 0
            n32 = int(H_actual[32].item()) if int(H_actual.numel()) > 32 else 0
            markers = self._collect_fixed_connection_markers(nodes)

            w_amine = int(n0 + 2 * n27)
            amine_total = int(len(markers.get('amine_6_edges', set()) or set()) + len(markers.get('amine_20_edges', set()) or set()))
            if int(amine_total) != int(w_amine):
                errors.append(f"固定连接总量不匹配: SU6+20={amine_total} vs SU0+2*SU27={w_amine}")

            w_halogen = int(n32)
            halogen_total = int(len(markers.get('halogen_8_edges', set()) or set()) + len(markers.get('halogen_21_edges', set()) or set()))
            if int(halogen_total) != int(w_halogen):
                errors.append(f"固定连接总量不匹配: SU8+21={halogen_total} vs SU32={w_halogen}")
        except Exception:
            pass
        errors.extend(
            self._fixed_anchor_invariant_errors(
                nodes,
                check_thio=True,
                check_ether=True,
                check_amine=True,
                check_halogen=True,
            )
        )

        # 7. 检查羰基结构单元优先连接SU9的规则
        carbonyl_sus = [0, 1, 2, 3]
        for n in nodes:
            if int(n.su_type) in carbonyl_sus and n.is_hop1_complete():
                neighbor_types = [int(nodes[nid].su_type) for nid in n.hop1_ids if nid < len(nodes)]
                # 检查是否优先连接了SU9（仅当存在真实可接入 carbonyl port3 的SU9时）
                su9_available = any(
                    int(getattr(cand, 'su_type', -1)) == 9 and
                    int(getattr(cand, 'global_id', -1)) != int(n.global_id) and
                    int(getattr(cand, 'global_id', -1)) not in n.hop1_ids and
                    int(cand.remaining_hop1_slots()) > 0 and
                    self._can_add_hop1_connection(nodes, n, cand)
                    for cand in nodes
                )
                if su9_available and 9 not in neighbor_types:
                    errors.append(
                        f"羰基节点{n.global_id}(SU{n.su_type}): hop1={sorted(neighbor_types)} 未优先连接SU9"
                    )

        # 7.5 检查逐边语义合法性
        seen_edges: Set[Tuple[int, int]] = set()
        for n in nodes:
            for neighbor_id in n.hop1_ids:
                if neighbor_id < 0 or neighbor_id >= len(nodes):
                    continue
                edge = tuple(sorted((int(n.global_id), int(neighbor_id))))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                neighbor = nodes[neighbor_id]
                if not self._edge_semantics_ok(int(n.su_type), int(neighbor.su_type), E_target=E_target):
                    errors.append(
                        f"非法1-hop边: {int(n.global_id)}(SU{int(n.su_type)}) <-> "
                        f"{int(neighbor.global_id)}(SU{int(neighbor.su_type)})"
                    )
        
        # 8. 检查元素组成（可选）
        if E_target is not None:
            E_pred = get_effective_nodes_element_vector(nodes, self.E_SU, device=torch.device(self.device))
            E_target = E_target.to(self.device)
            E_diff = torch.abs(E_pred - E_target)
            rel_err = E_diff / (E_target + 1e-6)
            
            for i, elem_name in enumerate(['C', 'H', 'O', 'N', 'S', 'X']):
                tol = 0.10 if str(elem_name) == 'H' else 0.05
                if rel_err[i] > float(tol):
                    errors.append(f"元素{elem_name}误差过大: 预测{E_pred[i]:.1f} vs 目标{E_target[i]:.1f} (相对误差{rel_err[i]:.2%})")

        if bool(verbose) and errors:
            self._print_consistency_errors(errors, prefix='Validate')
        
        return len(errors) == 0, errors
    
    # ========================================================================
    # Layer1: 1-hop分配辅助方法
    # ========================================================================
    
    def _initialize_node_pool(self, H_init: torch.Tensor) -> List[_NodeV3]:
        """初始化全局节点池，为每个SU实例创建节点对象
        """
        nodes = []
        global_id = 0
        special_degree_meta = self._get_special_degree_meta(H_init)
        special_partition_meta = self._get_special_partition_meta(H_init)
        special_anchor_mode_meta = self._get_special_anchor_mode_meta(H_init)
        
        for su_type in range(NUM_SU_TYPES):
            count = int(H_init[su_type].item())
            degree_sequence = self._build_special_degree_sequence(int(su_type), int(count), special_degree_meta)
            partition_sequence_19 = self._build_special_partition_sequence_19(int(count), special_partition_meta) if int(su_type) == 19 else []
            anchor_sequence_19 = self._build_special_anchor_sequence_19(int(count), special_anchor_mode_meta) if int(su_type) == 19 else []
            anchor_sequence_20 = self._build_special_anchor_sequence_20(int(count), special_anchor_mode_meta) if int(su_type) == 20 else []
            anchor_sequence_21 = self._build_special_anchor_sequence_21(int(count), special_anchor_mode_meta) if int(su_type) == 21 else []
            for idx in range(count):
                target_fixed_anchor_count = None
                anchor_mode = None
                if int(su_type) == 19 and idx < len(anchor_sequence_19):
                    target_degree, partition, target_fixed_anchor_count, anchor_mode = anchor_sequence_19[idx]
                elif int(su_type) == 19 and idx < len(partition_sequence_19):
                    target_degree, partition = partition_sequence_19[idx]
                elif int(su_type) == 20 and idx < len(anchor_sequence_20):
                    target_degree, target_fixed_anchor_count, anchor_mode = anchor_sequence_20[idx]
                    partition = None
                elif int(su_type) == 21 and idx < len(anchor_sequence_21):
                    target_degree, target_fixed_anchor_count, anchor_mode = anchor_sequence_21[idx]
                    partition = None
                else:
                    target_degree = degree_sequence[idx] if idx < len(degree_sequence) else None
                    partition = None
                    if int(su_type) in {19, 20, 21} and target_degree is not None:
                        target_fixed_anchor_count = 1
                special_source = 'layer0_special_degree_meta' if target_degree is not None else None
                node = _NodeV3(
                    global_id=global_id,
                    su_type=su_type,
                    target_hop1_degree=target_degree,
                    special_degree_source=special_source,
                    special_anchor_partition=partition,
                    target_fixed_anchor_count=target_fixed_anchor_count,
                    special_anchor_mode=anchor_mode,
                )
                nodes.append(node)
                global_id += 1
        
        return nodes

    def _restore_seed_topology(self, nodes: List[_NodeV3], seed_nodes: Optional[List[_NodeV3]]) -> None:
        """Best-effort warm start from a previous node list with a nearby histogram.

        Mapping is done within stable structural buckets instead of raw SU type only.
        This avoids assigning an old SU19(d2/ether) topology onto a new SU19(d3/ether)
        node after Layer4 changes special-degree metadata while keeping the histogram
        unchanged.

        Edges are restored only when both endpoints still exist after remapping and the
        edge still satisfies the incremental per-port validation on both sides.
        """
        if not seed_nodes:
            return

        def _seed_bucket_key(node: _NodeV3) -> Tuple[int, Optional[str], Optional[int], int, Optional[str]]:
            try:
                su_type = int(getattr(node, 'su_type', -1))
            except Exception:
                su_type = -1
            part = self._node_anchor_partition(node)
            degree = self._node_target_degree(node)
            fixed_cnt = self._node_fixed_anchor_target(node)
            mode = self._node_anchor_mode(node)
            return (
                int(su_type),
                (str(part) if part is not None else None),
                (int(degree) if degree is not None else None),
                int(fixed_cnt),
                (str(mode) if mode is not None else None),
            )

        new_by_bucket: Dict[Tuple[int, Optional[str], Optional[int]], List[_NodeV3]] = defaultdict(list)
        for node in nodes:
            new_by_bucket[_seed_bucket_key(node)].append(node)
        for bucket in list(new_by_bucket.keys()):
            new_by_bucket[bucket].sort(key=lambda n: int(n.global_id))

        seed_by_bucket: Dict[Tuple[int, Optional[str], Optional[int]], List[_NodeV3]] = defaultdict(list)
        seed_lookup: Dict[int, _NodeV3] = {}
        for node in seed_nodes:
            try:
                bucket = _seed_bucket_key(node)
                seed_by_bucket[bucket].append(node)
                seed_lookup[int(node.global_id)] = node
            except Exception:
                continue
        for bucket in list(seed_by_bucket.keys()):
            seed_by_bucket[bucket].sort(key=lambda n: int(n.global_id))

        old_to_new: Dict[int, _NodeV3] = {}
        for bucket, new_group in new_by_bucket.items():
            old_group = seed_by_bucket.get(bucket, [])
            for old_node, new_node in zip(old_group, new_group):
                try:
                    old_to_new[int(old_node.global_id)] = new_node
                except Exception:
                    continue
                try:
                    new_node.mu = float(getattr(old_node, 'mu', 0.0))
                    new_node.pi = float(getattr(old_node, 'pi', 1.0))
                except Exception:
                    pass
                try:
                    old_z = getattr(old_node, 'z_vec', None)
                    if isinstance(old_z, torch.Tensor):
                        new_node.z_vec = old_z.detach().clone()
                except Exception:
                    pass
                try:
                    old_hist = getattr(old_node, 'z_history', None)
                    if isinstance(old_hist, list):
                        new_node.z_history = [
                            z.detach().clone() if isinstance(z, torch.Tensor) else z
                            for z in old_hist
                        ]
                except Exception:
                    pass
                try:
                    old_sc = getattr(old_node, 'score_components', None)
                    if isinstance(old_sc, dict):
                        new_node.score_components = dict(old_sc)
                except Exception:
                    pass
                try:
                    new_node.template_key = getattr(old_node, 'template_key', None)
                except Exception:
                    pass

        seen_edges: Set[Tuple[int, int]] = set()
        for old_id, new_u in old_to_new.items():
            old_u = seed_lookup.get(int(old_id))
            if old_u is None:
                continue
            for old_v_id in list(getattr(old_u, 'hop1_ids', []) or []):
                try:
                    old_v_id_i = int(old_v_id)
                except Exception:
                    continue
                new_v = old_to_new.get(old_v_id_i)
                if new_v is None:
                    continue
                edge = tuple(sorted((int(new_u.global_id), int(new_v.global_id))))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if int(new_v.global_id) in new_u.hop1_ids:
                    continue
                if int(new_u.remaining_hop1_slots()) <= 0 or int(new_v.remaining_hop1_slots()) <= 0:
                    continue
                if not self._can_add_hop1_connection(nodes, new_u, new_v):
                    continue
                is_fixed = False
                try:
                    is_fixed = int(old_v_id_i) in set(getattr(old_u, 'fixed_hop1_ids', set()) or set())
                except Exception:
                    is_fixed = False
                self._add_bidirectional_hop1(
                    nodes,
                    int(new_u.global_id),
                    int(new_v.global_id),
                    lock=bool(is_fixed),
                )

    @staticmethod
    def _edge_is_fixed(node1: _NodeV3, node2: _NodeV3) -> bool:
        return bool(
            int(node2.global_id) in set(getattr(node1, 'fixed_hop1_ids', set()) or set())
            or int(node1.global_id) in set(getattr(node2, 'fixed_hop1_ids', set()) or set())
        )

    def _add_bidirectional_hop1(self,
                                nodes: List[_NodeV3],
                                id1: int,
                                id2: int,
                                lock: bool = False) -> bool:
        """添加双向1-hop连接
        """
        node1 = nodes[id1]
        node2 = nodes[id2]
        
        if id1 == id2:
            return False
        
        if id2 in node1.hop1_ids or id1 in node2.hop1_ids:
            return False
        if self._violates_special_d3_terminal_rule(nodes, node1, int(node2.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node2, int(node1.su_type)):
            return False
        
        # 添加互为1-hop（SU类型计数）
        node1.hop1_su[node2.su_type] += 1
        node2.hop1_su[node1.su_type] += 1
        
        # 记录全局ID（用于追踪具体连接）
        node1.hop1_ids.append(id2)
        node2.hop1_ids.append(id1)
        if bool(lock):
            node1.fixed_hop1_ids.add(int(id2))
            node2.fixed_hop1_ids.add(int(id1))
        self._refresh_runtime_node_entry(nodes, int(id1))
        self._refresh_runtime_node_entry(nodes, int(id2))
        # Cache keys include each endpoint's neighbor ids/types and remaining
        # slots, so successful edge edits naturally produce new signatures.
        # Avoid clearing the whole cache on every add; Layer1 calls this path
        # many thousands of times on large graphs.
        return True

    def _remove_bidirectional_hop1(self, nodes: List[_NodeV3], id1: int, id2: int) -> bool:
        """移除一条双向1-hop连接（仅移除一条，多重边会移除一次）"""
        return self._remove_bidirectional_hop1_with_force(nodes, id1, id2, force=False)

    def _remove_bidirectional_hop1_with_force(self,
                                              nodes: List[_NodeV3],
                                              id1: int,
                                              id2: int,
                                              force: bool = False) -> bool:
        """移除一条双向1-hop连接；force=True 时允许拆除 fixed 边。"""
        node1 = nodes[id1]
        node2 = nodes[id2]

        # 先确认边存在
        if id2 not in node1.hop1_ids or id1 not in node2.hop1_ids:
            return False
        if (not bool(force)) and self._edge_is_fixed(node1, node2):
            return False

        # 更新SU类型计数
        node1.hop1_su[node2.su_type] -= 1
        if node1.hop1_su[node2.su_type] <= 0:
            del node1.hop1_su[node2.su_type]

        node2.hop1_su[node1.su_type] -= 1
        if node2.hop1_su[node1.su_type] <= 0:
            del node2.hop1_su[node1.su_type]

        # 更新全局ID列表（移除一次）
        node1.hop1_ids.remove(id2)
        node2.hop1_ids.remove(id1)
        node1.fixed_hop1_ids.discard(int(id2))
        node2.fixed_hop1_ids.discard(int(id1))
        self._refresh_runtime_node_entry(nodes, int(id1))
        self._refresh_runtime_node_entry(nodes, int(id2))
        return True

    def _get_allowed_neighbor_types(self, su_type: int) -> List[int]:
        allowed = SU_FIXED_CONNECTIONS.get(su_type)
        if allowed is None:
            return list(range(NUM_SU_TYPES))
        return list(allowed)

    def _edge_semantics_ok(self,
                           su1: int,
                           su2: int,
                           E_target: Optional[torch.Tensor] = None) -> bool:
        E_eval = E_target if E_target is not None else getattr(self, '_current_E_target', None)
        cache_key = (int(su1), int(su2), (id(E_eval) if E_eval is not None else None))
        cached = self._edge_semantics_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        try:
            ok = bool(validate_connection(int(su1), int(su2), E_eval)) and bool(
                validate_connection(int(su2), int(su1), E_eval)
            )
            self._edge_semantics_cache[cache_key] = bool(ok)
            return bool(ok)
        except Exception:
            self._edge_semantics_cache[cache_key] = False
            return False

    def _can_add_hop1_connection(self, nodes: List[_NodeV3], node1: _NodeV3, node2: _NodeV3) -> bool:
        key = (
            self._can_add_state_signature(nodes, node1),
            self._can_add_state_signature(nodes, node2),
        )
        cached = self._can_add_hop1_cache.get(key)
        if cached is not None:
            return bool(cached)
        ok = self._can_add_hop1_connection_uncached(nodes, node1, node2)
        if len(self._can_add_hop1_cache) > 200000:
            self._can_add_hop1_cache.clear()
        self._can_add_hop1_cache[key] = bool(ok)
        return bool(ok)

    def _can_add_hop1_connection_uncached(self, nodes: List[_NodeV3], node1: _NodeV3, node2: _NodeV3) -> bool:
        """检查在 node1 和 node2 之间添加1-hop连接是否满足双向 per-port 规则
        
        对两侧分别检查：将新邻居加入当前邻居列表后，是否仍可分配到端口中。
        """
        if node1.global_id == node2.global_id:
            return False

        # Fixed-anchor matching may reason over multiple owner slots per node.
        # A repeated edge between the same two nodes must never count as filling
        # an extra slot/demand, otherwise the matcher can "use up" supply
        # without creating a real second connection.
        if int(node2.global_id) in set(getattr(node1, 'hop1_ids', []) or []):
            return False
        if int(node1.global_id) in set(getattr(node2, 'hop1_ids', []) or []):
            return False

        if not self._edge_semantics_ok(int(node1.su_type), int(node2.su_type)):
            return False

        if int(node1.remaining_hop1_slots()) <= 0 or int(node2.remaining_hop1_slots()) <= 0:
            return False

        if not self._is_su19_partition_compatible(node1, int(node2.su_type)):
            return False
        if not self._is_su19_partition_compatible(node2, int(node1.su_type)):
            return False

        if self._violates_mode_specific_rule(node1, int(node2.su_type)):
            return False
        if self._violates_mode_specific_rule(node2, int(node1.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node1, int(node2.su_type)):
            return False
        if self._violates_special_d3_terminal_rule(nodes, node2, int(node1.su_type)):
            return False

        required1 = set(int(x) for x in self._required_external_candidates_for_node(node1))
        target_req1 = int(self._node_fixed_anchor_target(node1))
        current_req1 = int(self._required_external_count_for_node(node1, nodes))
        if required1 and int(current_req1) < int(target_req1) and int(node2.su_type) not in required1:
            return False
        if required1 and int(node2.su_type) in required1 and int(current_req1) >= int(target_req1):
            return False
        required2 = set(int(x) for x in self._required_external_candidates_for_node(node2))
        target_req2 = int(self._node_fixed_anchor_target(node2))
        current_req2 = int(self._required_external_count_for_node(node2, nodes))
        if required2 and int(current_req2) < int(target_req2) and int(node1.su_type) not in required2:
            return False
        if required2 and int(node1.su_type) in required2 and int(current_req2) >= int(target_req2):
            return False

        # 检查 node1 侧
        port_sets1 = self._port_sets_for_node(node1)
        if port_sets1:
            current_neighbors1 = [int(nodes[nid].su_type) for nid in node1.hop1_ids]
            proposed1 = current_neighbors1 + [int(node2.su_type)]
            if not can_match_ports_partial(proposed1, port_sets1):
                return False

        # 检查 node2 侧
        port_sets2 = self._port_sets_for_node(node2)
        if port_sets2:
            current_neighbors2 = [int(nodes[nid].su_type) for nid in node2.hop1_ids]
            proposed2 = current_neighbors2 + [int(node1.su_type)]
            if not can_match_ports_partial(proposed2, port_sets2):
                return False

        return True

    def _repair_remaining_hop1_slots(self, nodes: List[_NodeV3]) -> None:
        self._refiner.repair_remaining_hop1_slots(
            assigner=self,
            nodes=nodes,
        )
    
    def _get_nodes_by_su_type(self, nodes: List[_NodeV3], su_type: int) -> List[_NodeV3]:
        """获取指定SU类型的所有节点"""
        if self._runtime_index_matches(nodes):
            ids = list(self._node_ids_by_su.get(int(su_type), []))
            return [nodes[int(i)] for i in ids]
        return [n for n in nodes if n.su_type == su_type]
    
    def _get_empty_hop1_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中1-hop为空的节点"""
        if self._runtime_index_matches(nodes):
            result: List[_NodeV3] = []
            for su_type in su_types:
                ids = sorted(int(i) for i in self._empty_ids_by_su.get(int(su_type), set()))
                result.extend(nodes[int(i)] for i in ids)
            return result
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and n.is_hop1_empty():
                    result.append(n)
        return result
    
    def _get_incomplete_hop1_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中1-hop不完整（有分配但未满）的节点"""
        if self._runtime_index_matches(nodes):
            result: List[_NodeV3] = []
            for su_type in su_types:
                ids = sorted(int(i) for i in self._incomplete_ids_by_su.get(int(su_type), set()))
                result.extend(nodes[int(i)] for i in ids)
            return result
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and not n.is_hop1_empty() and not n.is_hop1_complete():
                    result.append(n)
        return result
    
    def _get_available_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中还有空闲1-hop槽位的节点"""
        if self._runtime_index_matches(nodes):
            result: List[_NodeV3] = []
            for su_type in su_types:
                ids = sorted(int(i) for i in self._available_ids_by_su.get(int(su_type), set()))
                result.extend(nodes[int(i)] for i in ids)
            return result
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and n.remaining_hop1_slots() > 0:
                    result.append(n)
        return result

    def _complete_nodes_round_robin(self,
                                    nodes: List[_NodeV3],
                                    target_nodes: List[_NodeV3],
                                    candidate_types: List[int],
                                    priority_list: List[int],
                                    salt: int = 0) -> None:
        """
        每轮每个节点最多补一条边，避免前排节点过早吃光高优先级候选，导致局部 motif 坍缩。
        """
        round_idx = 0
        while True:
            active_nodes = [n for n in target_nodes if not n.is_hop1_complete()]
            if not active_nodes:
                break

            progressed = False
            motif_usage = build_motif_usage_from_pairs(
                (int(node.su_type), hop1_counter_to_multiset(getattr(node, 'hop1_su', {}) or {}))
                for node in nodes
            )
            ordered_nodes = self._maybe_shuffle_nodes(active_nodes, salt=int(salt) + int(round_idx))
            for node in ordered_nodes:
                if node.is_hop1_complete():
                    continue

                candidate_types_eff = list(candidate_types)
                priority_eff = list(priority_list)
                required_external = self._required_external_candidates_for_node(node)
                if required_external:
                    if not self._has_required_external_for_node(node, nodes):
                        candidate_types_eff = [t for t in candidate_types_eff if int(t) in set(required_external)]
                        priority_eff = [t for t in priority_eff if int(t) in set(required_external)]
                        if not candidate_types_eff:
                            candidate_types_eff = list(required_external)
                        if not priority_eff:
                            priority_eff = list(required_external)

                empty_candidates = self._get_empty_hop1_nodes(nodes, candidate_types_eff)
                empty_candidates = [
                    n for n in empty_candidates
                    if n.global_id != node.global_id and n.remaining_hop1_slots() > 0
                ]
                if empty_candidates:
                    target = self._choose_weighted_candidate(
                        node,
                        empty_candidates,
                        priority_eff,
                        nodes=nodes,
                        motif_usage=motif_usage,
                    )
                else:
                    available_candidates = self._get_available_nodes(nodes, candidate_types_eff)
                    available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                    target = self._choose_weighted_candidate(node, available_candidates, priority_eff, nodes=nodes, motif_usage=motif_usage) \
                        if available_candidates else None

                if target is None:
                    continue

                old_center_ms = hop1_counter_to_multiset(getattr(node, 'hop1_su', {}) or {})
                old_target_ms = hop1_counter_to_multiset(getattr(target, 'hop1_su', {}) or {})
                self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
                new_center_ms = hop1_counter_to_multiset(getattr(node, 'hop1_su', {}) or {})
                new_target_ms = hop1_counter_to_multiset(getattr(target, 'hop1_su', {}) or {})

                center_key_old = (int(node.su_type), tuple(old_center_ms))
                center_key_new = (int(node.su_type), tuple(new_center_ms))
                target_key_old = (int(target.su_type), tuple(old_target_ms))
                target_key_new = (int(target.su_type), tuple(new_target_ms))
                if old_center_ms:
                    motif_usage[center_key_old] -= 1
                    if int(motif_usage[center_key_old]) <= 0:
                        motif_usage.pop(center_key_old, None)
                if old_target_ms:
                    motif_usage[target_key_old] -= 1
                    if int(motif_usage[target_key_old]) <= 0:
                        motif_usage.pop(target_key_old, None)
                if new_center_ms:
                    motif_usage[center_key_new] += 1
                if new_target_ms:
                    motif_usage[target_key_new] += 1
                progressed = True

            if not progressed:
                break
            round_idx += 1
    
    def _choose_weighted_candidate(self,
                                   center: _NodeV3,
                                   candidates: List[_NodeV3],
                                   priority_list: List[int],
                                   nodes: Optional[List[_NodeV3]] = None,
                                   motif_usage: Optional[Counter] = None) -> Optional[_NodeV3]:
        filtered = [
            n for n in candidates
            if n.global_id != center.global_id
            and n.remaining_hop1_slots() > 0
            and n.global_id not in center.hop1_ids
        ]
        if not filtered:
            return None

        # per-port 双向验证：确保添加连接后两侧端口规则均满足
        if nodes is not None:
            filtered = [n for n in filtered if self._can_add_hop1_connection(nodes, center, n)]
            if not filtered:
                return None

        motif_usage_local = motif_usage if motif_usage is not None else (
            build_motif_usage_from_pairs(
                (int(node.su_type), hop1_counter_to_multiset(getattr(node, 'hop1_su', {}) or {}))
                for node in nodes
            ) if nodes is not None else Counter()
        )

        weights = []
        current_neighbors = self._current_neighbor_types(nodes, center) if nodes is not None else []
        priority_rank = {int(su): int(idx) for idx, su in enumerate(priority_list)}
        required_external_center = set(int(x) for x in self._required_external_candidates_for_node(center))
        center_missing_external = bool(required_external_center) and not self._has_required_external_for_node(center, nodes) if nodes is not None else False
        for n in filtered:
            idx = priority_rank.get(int(n.su_type), len(priority_list))
            base = 1.0 / (1 + idx)
            w = base
            w /= (1 + center.hop1_su.get(n.su_type, 0))
            w /= (1 + n.hop1_su.get(center.su_type, 0))
            w *= self._bridge_candidate_bias(int(center.su_type), current_neighbors, int(n.su_type))
            if nodes is not None and int(center.su_type) == 11:
                cand_su = int(n.su_type)
                if bool(center_missing_external):
                    if cand_su in {22, 23, 24, 25, 15, 17}:
                        w *= 2.50
                    elif cand_su in {19, 20, 21}:
                        w *= 0.35
                    if cand_su in {19, 20, 21} and self._has_pending_required_external(n, nodes):
                        w *= 0.10
                elif cand_su in {19, 20, 21} and self._has_pending_required_external(n, nodes):
                    w *= 0.10
            if nodes is not None:
                center_after = tuple(sorted(current_neighbors + [int(n.su_type)]))
                target_neighbors = self._current_neighbor_types(nodes, n)
                target_after = tuple(sorted(target_neighbors + [int(center.su_type)]))
                center_usage = int(motif_usage_local.get((int(center.su_type), tuple(center_after)), 0))
                target_usage = int(motif_usage_local.get((int(n.su_type), tuple(target_after)), 0))
                w /= (1.0 + float(motif_penalty_alpha(int(center.su_type))) * float(center_usage))
                w /= (1.0 + 0.5 * float(motif_penalty_alpha(int(n.su_type))) * float(target_usage))
            weights.append(w)

        if bool(self.deterministic):
            ranked = sorted(
                range(len(filtered)),
                key=lambda i: (
                    float(weights[i]),
                    int(filtered[i].remaining_hop1_slots()),
                    -int(filtered[i].global_id),
                ),
                reverse=True,
            )
            if not ranked:
                return None
            top_weight = float(weights[int(ranked[0])])
            if top_weight > 0.0:
                window = [idx for idx in ranked if float(weights[int(idx)]) >= 0.85 * float(top_weight)]
            else:
                window = list(ranked)
            window = window[: max(1, min(4, len(window)))]
            choose_pos = int(self._build_variant + int(center.global_id)) % len(window)
            return filtered[int(window[int(choose_pos)])]
        return random.choices(filtered, weights=weights, k=1)[0]
    
    # ========================================================================
    # Layer1: 固定连接分配方法（a-f）
    # ========================================================================
    
    def _assign_fixed_halogen_X(self, nodes: List[_NodeV3]):
        """a) 32号X -> 8号/21号"""
        owner_ids, target_ids_all, edge_pairs = self._build_exact_halogen_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Halogen Fixed Assign',
            check_thio=False,
            check_ether=False,
            check_amine=False,
            check_halogen=True,
        )

    def _build_exact_halogen_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        x_nodes = self._get_nodes_by_su_type(work_nodes, 32)
        su8_nodes = self._get_nodes_by_su_type(work_nodes, 8)
        su21_nodes = self._sort_special_degree_nodes(self._get_nodes_by_su_type(work_nodes, 21))
        owner_ids = sorted(int(n.global_id) for n in list(x_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su8_nodes) + list(su21_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(x_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 32,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su21_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '21_halogen',
                })
        for node in list(su8_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '8_halogen',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 21 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Halogen Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) not in {8, 21}:
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Halogen Fixed Assign] 无法找到满足严格 single/double 约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _build_exact_thio_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        s_nodes = self._get_nodes_by_su_type(work_nodes, 31)
        su7_nodes = self._get_nodes_by_su_type(work_nodes, 7)
        su19_nodes = [
            n for n in self._get_nodes_by_su_type(work_nodes, 19)
            if self._node_anchor_partition(n) == 'thio'
        ]
        owner_ids = sorted(int(n.global_id) for n in list(s_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su7_nodes) + list(su19_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for s_node in self._stable_node_order(s_nodes):
            for slot_idx in range(max(0, int(s_node.get_max_degree()))):
                owner_slots.append({
                    'owner_id': int(s_node.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 31,
                })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su19_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '19_thio',
                })
        for node in list(su7_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '7_thio',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 19 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Thio Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            if int(getattr(target, 'su_type', -1)) == 19 and self._node_anchor_partition(target) != 'thio':
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Thio Fixed Assign] 无法找到满足严格分区约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _build_exact_ether_fixed_edges(self, nodes: List[_NodeV3]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        work_nodes = copy.deepcopy(nodes)
        su2_nodes = self._get_nodes_by_su_type(work_nodes, 2)
        su28_nodes = self._get_nodes_by_su_type(work_nodes, 28)
        su29_nodes = self._get_nodes_by_su_type(work_nodes, 29)
        su5_nodes = self._get_nodes_by_su_type(work_nodes, 5)
        su19_nodes = [
            n for n in self._get_nodes_by_su_type(work_nodes, 19)
            if self._node_anchor_partition(n) == 'ether'
        ]
        owner_ids = sorted(int(n.global_id) for n in list(su2_nodes) + list(su28_nodes) + list(su29_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su5_nodes) + list(su19_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(su28_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 28,
            })
        for owner in self._stable_node_order(su29_nodes):
            for slot_idx in range(max(0, int(owner.get_max_degree()))):
                owner_slots.append({
                    'owner_id': int(owner.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 29,
                })
        for owner in self._stable_node_order(su2_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 2,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su19_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '19_ether',
                })
        for node in list(su5_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '5_ether',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 19 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Ether Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            owner_su = int(getattr(owner, 'su_type', -1))
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) == 5:
                return bool(self._can_add_hop1_connection(work_local, owner, target))
            if int(target_su) != 19 or self._node_anchor_partition(target) != 'ether':
                return False
            target_deg = int(self._node_target_degree(target) or 0)
            if int(target_deg) == 1:
                if int(owner_su) == 28:
                    return False
                if int(owner_su) == 29 and int(owner_d1_usage.get(int(owner.global_id), 0)) >= 1:
                    return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        if edge_pairs is None:
            raise RuntimeError("[Ether Fixed Assign] 无法找到满足严格分区约束的精确匹配")
        return owner_ids, target_ids_all, list(edge_pairs)

    def _assign_fixed_thioether_S(self, nodes: List[_NodeV3]):
        """b) 31号硫醚 -> 7号/19号"""
        owner_ids, target_ids_all, edge_pairs = self._build_exact_thio_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Thio Fixed Assign',
            check_thio=True,
            check_ether=False,
            check_amine=False,
            check_halogen=False,
        )
    
    def _assign_fixed_amine_N(self, nodes: List[_NodeV3]):
        """c) 0号氨基端、27号 -> 6号/20号"""
        work_nodes = copy.deepcopy(nodes)
        su0_nodes = self._get_nodes_by_su_type(work_nodes, 0)
        su27_nodes = self._get_nodes_by_su_type(work_nodes, 27)
        su6_nodes = self._get_nodes_by_su_type(work_nodes, 6)
        su20_nodes = self._sort_special_degree_nodes(self._get_nodes_by_su_type(work_nodes, 20))

        owner_ids = sorted(int(n.global_id) for n in list(su0_nodes) + list(su27_nodes))
        target_ids_all = sorted(int(n.global_id) for n in list(su6_nodes) + list(su20_nodes))
        self._clear_all_edges_incident_to_set(work_nodes, owner_ids + target_ids_all)

        owner_slots: List[Dict[str, int]] = []
        for owner in self._stable_node_order(su27_nodes):
            for slot_idx in range(2):
                owner_slots.append({
                    'owner_id': int(owner.global_id),
                    'slot_index': int(slot_idx),
                    'kind': 27,
                })
        for owner in self._stable_node_order(su0_nodes):
            owner_slots.append({
                'owner_id': int(owner.global_id),
                'slot_index': 0,
                'kind': 0,
            })

        target_demands: List[Dict[str, Any]] = []
        for node in list(su20_nodes):
            demand_cnt = max(1, int(self._node_fixed_anchor_target(node)))
            for demand_idx in range(max(0, int(demand_cnt))):
                target_demands.append({
                    'target_id': int(node.global_id),
                    'demand_index': int(demand_idx),
                    'kind': '20_amine',
                })
        for node in list(su6_nodes):
            target_demands.append({
                'target_id': int(node.global_id),
                'demand_index': 0,
                'kind': '6_amine',
            })
        target_demands.sort(
            key=lambda d: (
                0 if int(work_nodes[int(d['target_id'])].su_type) == 20 else 1,
                -(self._node_fixed_anchor_target(work_nodes[int(d['target_id'])])),
                int(self._node_target_degree(work_nodes[int(d['target_id'])]) or 0),
                int(d.get('demand_index', 0)),
                int(d['target_id']),
            )
        )
        if int(len(owner_slots)) != int(len(target_demands)):
            raise RuntimeError(
                f"[Amine Fixed Assign] target_demand={len(target_demands)} != owner_slots={len(owner_slots)}"
            )

        def _compat(work_local: List[_NodeV3],
                    owner: _NodeV3,
                    target: _NodeV3,
                    slot: Dict[str, int],
                    owner_d1_usage: Dict[int, int],
                    demand: Dict[str, Any]) -> bool:
            _ = slot
            _ = owner_d1_usage
            _ = demand
            target_su = int(getattr(target, 'su_type', -1))
            if int(target_su) not in {6, 20}:
                return False
            return bool(self._can_add_hop1_connection(work_local, owner, target))

        edge_pairs = self._solve_exact_fixed_anchor_matching_demands(
            work_nodes,
            owner_slots,
            target_demands,
            _compat,
        )
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        if edge_pairs is None:
            raise RuntimeError("[Amine Fixed Assign] 无法找到满足严格 single/double 约束的精确匹配")
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Amine Fixed Assign',
            check_thio=True,
            check_ether=False,
            check_amine=True,
            check_halogen=False,
        )
    
    def _assign_fixed_ether_O(self, nodes: List[_NodeV3]):
        """d) 2号醚端、28号、29号 -> 5号/19号
        """
        owner_ids, target_ids_all, edge_pairs = self._build_exact_ether_fixed_edges(nodes)
        self._clear_all_edges_incident_to_set(nodes, owner_ids + target_ids_all)
        for id1, id2 in edge_pairs:
            self._add_bidirectional_hop1(nodes, int(id1), int(id2), lock=True)
        self._raise_if_fixed_anchor_invariants_fail(
            nodes,
            stage='Ether Fixed Assign',
            check_thio=True,
            check_ether=True,
            check_amine=False,
            check_halogen=False,
        )
    
    def _assign_fixed_carbonyl(self, nodes: List[_NodeV3]):
        """e) 0/1/2/3羰基端 -> 必须先消耗所有9号，再分配其他类型
        
        核心规则：每个SU9的port3={0,1,2,3}，必须恰好有一个羰基邻居。
        因此羰基节点必须先强制消耗所有SU9，然后再用加权随机填充剩余槽位。
        """
        su0_nodes = self._get_nodes_by_su_type(nodes, 0)
        su1_nodes = self._get_nodes_by_su_type(nodes, 1)
        su2_nodes = self._get_nodes_by_su_type(nodes, 2)
        su3_nodes = self._get_nodes_by_su_type(nodes, 3)
        su9_nodes = [n for n in self._get_nodes_by_su_type(nodes, 9) if n.remaining_hop1_slots() > 0]
        
        def _can_pair(center_su: int, neigh_su: int) -> bool:
            try:
                c_allowed = set(self._get_allowed_neighbor_types(int(center_su)))
                n_allowed = set(self._get_allowed_neighbor_types(int(neigh_su)))
                return (int(neigh_su) in c_allowed) and (int(center_su) in n_allowed)
            except Exception as e:
                import logging
                logging.debug(f"Exception in _both_allowed check: {e}")
                return True

        def _available_by_types(center: _NodeV3, types: List[int]) -> List[_NodeV3]:
            out = []
            for n in nodes:
                if int(n.su_type) not in set(int(x) for x in types):
                    continue
                if int(n.global_id) == int(center.global_id):
                    continue
                if int(n.global_id) in center.hop1_ids:
                    continue
                if int(n.remaining_hop1_slots()) <= 0:
                    continue
                if not _can_pair(int(center.su_type), int(n.su_type)):
                    continue
                if not self._can_add_hop1_connection(nodes, center, n):
                    continue
                out.append(n)
            return out

        # ---- 阶段1：先给每个0/1/2/3尽量分一个9号 ----
        first_round = self._maybe_shuffle_nodes(list(su0_nodes) + list(su1_nodes) + list(su2_nodes) + list(su3_nodes), salt=401)
        for carb in first_round:
            if carb.remaining_hop1_slots() <= 0:
                continue
            su9_cands = self._filter_valid_fixed_targets(nodes, carb, su9_nodes)
            if not su9_cands:
                continue
            su9 = self._choose_weighted_candidate(carb, su9_cands, [9], nodes=nodes)
            if su9 is None:
                continue
            self._add_bidirectional_hop1(nodes, carb.global_id, su9.global_id)

        # ---- 阶段1.5：如果9号还有剩余，再继续给3号分配第二个9 ----
        for n3 in su3_nodes:
            while n3.remaining_hop1_slots() > 0:
                su9_cands = self._filter_valid_fixed_targets(nodes, n3, su9_nodes)
                if not su9_cands:
                    break
                su9 = self._choose_weighted_candidate(n3, su9_cands, [9], nodes=nodes)
                if su9 is None:
                    break
                self._add_bidirectional_hop1(nodes, n3.global_id, su9.global_id)

        # ---- 阶段2：填充羰基节点的剩余槽位（非SU9类型） ----
        pri_su3 = [23, 24, 25, 22, 19, 20, 21, 14, 15, 17]
        pri_su0 = [23, 24, 25, 22, 14, 15, 17]
        pri_su2 = [23, 24, 25, 22, 19, 20, 21, 14, 15, 17]
        pri_su1 = [23, 24, 25, 19, 20, 21, 14, 15, 17]

        for n3 in su3_nodes:
            needed = n3.remaining_hop1_slots()
            for _ in range(max(0, int(needed))):
                cands = _available_by_types(n3, pri_su3)
                t = self._choose_weighted_candidate(n3, cands, pri_su3, nodes=nodes)
                if t is None:
                    break
                self._add_bidirectional_hop1(nodes, n3.global_id, t.global_id)

        for n0 in su0_nodes:
            needed = n0.remaining_hop1_slots()
            for _ in range(max(0, int(needed))):
                current = self._current_neighbor_types(nodes, n0)
                if not any(int(x) in {6, 20} for x in current):
                    cands = _available_by_types(n0, [6, 20])
                    t = self._choose_weighted_candidate(n0, cands, [6, 20], nodes=nodes)
                else:
                    cands = _available_by_types(n0, pri_su0)
                    t = self._choose_weighted_candidate(n0, cands, pri_su0, nodes=nodes)
                if t is None:
                    break
                self._add_bidirectional_hop1(nodes, n0.global_id, t.global_id)

        for n2 in su2_nodes:
            needed = n2.remaining_hop1_slots()
            for _ in range(max(0, int(needed))):
                current = self._current_neighbor_types(nodes, n2)
                if not any(int(x) in {5, 19} for x in current):
                    cands = _available_by_types(n2, [5, 19])
                    t = self._choose_weighted_candidate(n2, cands, [5, 19], nodes=nodes)
                else:
                    cands = _available_by_types(n2, pri_su2)
                    t = self._choose_weighted_candidate(n2, cands, pri_su2, nodes=nodes)
                if t is None:
                    break
                self._add_bidirectional_hop1(nodes, n2.global_id, t.global_id)

        for n1 in su1_nodes:
            needed = n1.remaining_hop1_slots()
            for _ in range(max(0, int(needed))):
                cands = _available_by_types(n1, pri_su1)
                t = self._choose_weighted_candidate(n1, cands, pri_su1, nodes=nodes)
                if t is None:
                    break
                self._add_bidirectional_hop1(nodes, n1.global_id, t.global_id)

    def _assign_unsaturated_pairs(self, nodes: List[_NodeV3]):
        """f) 14/15/16双键配对，17/18三键配对
        """
        # 收集所有需要配对的双键节点（不饱和端连接度=1）
        su14_nodes = self._get_nodes_by_su_type(nodes, 14)
        su15_nodes = self._get_nodes_by_su_type(nodes, 15)
        su16_nodes = self._get_nodes_by_su_type(nodes, 16)
        
        # 按优先级15>16>14构建池，并随机打乱
        double_bond_pool = self._maybe_shuffle_nodes(list(su15_nodes) + list(su16_nodes) + list(su14_nodes), salt=503)
        
        # 两两配对：每个节点的不饱和端连接度=1，只能配对一次
        paired = set()
        while len(double_bond_pool) >= 2:
            # 取出第一个未配对的节点
            node1 = None
            for n in double_bond_pool:
                if n.global_id not in paired:
                    node1 = n
                    break
            if node1 is None:
                break
            
            # 找一个可配对的伙伴（优先选择15>16>14，且未配对）
            partner = None
            priority_order = [15, 16, 14]
            for su_type in priority_order:
                for n in double_bond_pool:
                    if n.global_id != node1.global_id and n.global_id not in paired and n.su_type == su_type:
                        if not self._can_add_hop1_connection(nodes, node1, n):
                            continue
                        partner = n
                        break
                if partner:
                    break
            
            if partner:
                paired.add(node1.global_id)
                paired.add(partner.global_id)
                self._add_bidirectional_hop1(nodes, node1.global_id, partner.global_id)
            else:
                # 没有可配对的伙伴，跳过
                paired.add(node1.global_id)
        
        # 三键配对 (17, 18) - 17的不饱和端连接度=1，18的连接度=1
        su17_nodes = self._get_nodes_by_su_type(nodes, 17)
        su18_nodes = list(self._get_nodes_by_su_type(nodes, 18))
        
        # 先17-18配对
        for n17 in su17_nodes:
            if su18_nodes and n17.remaining_hop1_slots() > 0:
                partner_idx = None
                for idx, n18 in enumerate(su18_nodes):
                    if self._can_add_hop1_connection(nodes, n17, n18):
                        partner_idx = idx
                        break
                if partner_idx is not None:
                    n18 = su18_nodes.pop(partner_idx)
                    self._add_bidirectional_hop1(nodes, n17.global_id, n18.global_id)
        
        # 如果还有剩余的17且port2未填充（即没有17/18邻居），才做17-17配对
        # SU17端口规则: [{芳香}, {17, 18}]，port2只能放17或18
        # 已有SU18邻居的SU17的port2已满，剩余槽位是port1(芳香)，不能再放17
        remaining_17_no_triple = [
            n for n in su17_nodes 
            if n.remaining_hop1_slots() > 0 
            and 18 not in n.hop1_su and 17 not in n.hop1_su
        ]
        while len(remaining_17_no_triple) >= 2:
            n1 = remaining_17_no_triple.pop(0)
            partner_idx = None
            for idx, n2 in enumerate(remaining_17_no_triple):
                if self._can_add_hop1_connection(nodes, n1, n2):
                    partner_idx = idx
                    break
            if partner_idx is None:
                continue
            n2 = remaining_17_no_triple.pop(partner_idx)
            self._add_bidirectional_hop1(nodes, n1.global_id, n2.global_id)
    
    def _assign_heterocyclic_NS(self, nodes: List[_NodeV3]):
        """f2) 26号杂环N和30号杂环S的1-hop分配（连接度=2）"""
        aromatic_priority = [13, 11, 12, 10, 5, 6, 7, 8, 9]
        
        # 处理26号杂环N（连接度=2）
        su26_nodes = self._get_nodes_by_su_type(nodes, 26)
        for node in su26_nodes:
            needed = node.remaining_hop1_slots()
            for _ in range(needed):
                # 优先选择空1-hop且有剩余槽位的芳香节点
                candidates = self._get_empty_hop1_nodes(nodes, aromatic_priority)
                candidates = [n for n in candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                
                if not candidates:
                    # 没有空节点，选择有剩余槽位的不完整节点
                    candidates = self._get_available_nodes(nodes, aromatic_priority)
                    candidates = [n for n in candidates if n.global_id != node.global_id]
                
                if candidates:
                    target = self._choose_weighted_candidate(node, candidates, aromatic_priority, nodes=nodes)
                    if target is None:
                        break
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
                else:
                    break
        
        # 处理30号杂环S（连接度=2）
        su30_nodes = self._get_nodes_by_su_type(nodes, 30)
        for node in su30_nodes:
            needed = node.remaining_hop1_slots()
            for _ in range(needed):
                # 优先选择空1-hop且有剩余槽位的芳香节点
                candidates = self._get_empty_hop1_nodes(nodes, aromatic_priority)
                candidates = [n for n in candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                
                if not candidates:
                    candidates = self._get_available_nodes(nodes, aromatic_priority)
                    candidates = [n for n in candidates if n.global_id != node.global_id]
                
                if candidates:
                    target = self._choose_weighted_candidate(node, candidates, aromatic_priority, nodes=nodes)
                    if target is None:
                        break
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
                else:
                    break
    
    def _assign_aryl_connections(self, nodes: List[_NodeV3]):
        """f3) 10号芳基取代碳的port3={4,10}分配
        
        规则：
        - SU10的port3只能放SU4或SU10
        - 优先让SU4连接SU10（SU4端口={23,24,25,10}，port1只含10）
        - 剩余SU10之间10-10配对，要求剩余数量为偶数
        - 若无SU4且SU10为奇数，则有一个SU10的port3无法填充
        """
        su4_nodes = [n for n in self._get_nodes_by_su_type(nodes, 4) if n.remaining_hop1_slots() > 0]
        su10_nodes = self._get_nodes_by_su_type(nodes, 10)
        
        # 阶段1：SU4 -> SU10 配对（SU4的唯一端口允许{23,24,25,10}，优先连10）
        for n4 in su4_nodes:
            # 找一个有空闲槽位的SU10来连接
            available_10 = [n for n in su10_nodes 
                          if n.remaining_hop1_slots() > 0 
                          and n.global_id not in n4.hop1_ids
                          and n4.global_id not in n.hop1_ids]
            if available_10:
                target = available_10[0]
                self._add_bidirectional_hop1(nodes, n4.global_id, target.global_id)
        
        # 阶段2：剩余SU10之间10-10配对
        # 收集port3尚未填充的SU10（即还没有SU4或SU10邻居占据port3）
        su10_need_port3 = [n for n in su10_nodes 
                          if n.remaining_hop1_slots() > 0
                          and 4 not in n.hop1_su and 10 not in n.hop1_su]
        
        su10_need_port3 = self._maybe_shuffle_nodes(su10_need_port3, salt=601)
        while len(su10_need_port3) >= 2:
            n1 = su10_need_port3.pop(0)
            n2 = su10_need_port3.pop(0)
            self._add_bidirectional_hop1(nodes, n1.global_id, n2.global_id)
        
    # ========================================================================
    # Layer1: 完成1-hop分配方法（g-i）
    # ========================================================================
    
    def _complete_aromatic_hop1(self, nodes: List[_NodeV3]):
        """g) 完成芳香结构的互为1-hop（5/6/7/8/9需要完成剩余连接）"""
        aromatic_types = [5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30]
        priority_list = [13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30]

        # 先补齐必需外接锚点，避免7/5/6/8/9先被普通芳香位填满。
        self._complete_required_external_anchors(nodes, [5, 6, 7, 8, 9])
        
        # 处理5/6/7/8/9（这些与杂原子互为1-hop后需要补全）
        for su_type in [8, 7, 6, 5, 9]:
            incomplete_nodes = self._get_incomplete_hop1_nodes(nodes, [su_type])
            self._complete_nodes_round_robin(
                nodes,
                incomplete_nodes,
                aromatic_types,
                priority_list,
                salt=700 + int(su_type),
            )
    
    def _complete_aliphatic_hetero_hop1(self, nodes: List[_NodeV3]):
        """g2) 完成19/20/21号脂肪杂原子碳的剩余1-hop

        X/N/S/O 固定锚点已经在前面的专用阶段分配完成；这里仅补全这些
        节点的非固定锚点端口。
        """
        # 非固定锚点补全优先级（不含 SU16 和 SU18）
        priority_list = [23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17]

        # 再次补齐必需固定锚点，避免 19/20/21 被普通脂肪/芳香连接提前占满。
        self._complete_required_external_anchors(nodes, [19, 20, 21])
        
        # 处理 21/20/19 号未满节点。
        # 这些节点的 fixed-anchor single/double 模式由 metadata 决定：
        #   target_hop1_degree
        #   target_fixed_anchor_count
        #   special_anchor_mode
        for su_type in [21, 20, 19]:
            incomplete_nodes = self._get_incomplete_hop1_nodes(nodes, [su_type])
            self._complete_nodes_round_robin(
                nodes,
                incomplete_nodes,
                priority_list,
                priority_list,
                salt=760 + int(su_type),
            )
    
    def _complete_unsaturated_saturated_end(self, nodes: List[_NodeV3]):
        """h) 完成不饱和结构的饱和端和4号腈"""
        # SU14 端口1/2: {23,24,25,22,19,20,21,2,1,0,3,4} (脂肪碳优先级)
        # SU15 端口1:   {23,24,25,22,19,20,21,2,1,0,3,4} (脂肪碳优先级)
        # SU17 端口1:   {23,24,25,19,20,21,2,0,3}        (饱和/羰基优先级)
        pri_14_15 = [23, 24, 25, 22, 19, 20, 21, 2, 1, 0, 3, 4]
        pri_17 = [23, 24, 25, 19, 20, 21, 2, 0, 3]
        
        for su_type in [17, 14, 15]:
            incomplete_nodes = self._get_incomplete_hop1_nodes(nodes, [su_type])
            priority_list = pri_17 if su_type == 17 else pri_14_15
            
            for node in incomplete_nodes:
                needed = node.remaining_hop1_slots()
                
                for _ in range(needed):
                    # 优先选择空1-hop且有剩余槽位的候选
                    empty_candidates = self._get_empty_hop1_nodes(nodes, priority_list)
                    empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                    
                    if empty_candidates:
                        target = self._choose_weighted_candidate(node, empty_candidates, priority_list, nodes=nodes)
                        if target is None:
                            break
                    else:
                        # 选择有剩余槽位的节点
                        available_candidates = self._get_available_nodes(nodes, priority_list)
                        available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                        if available_candidates:
                            target = self._choose_weighted_candidate(node, available_candidates, priority_list, nodes=nodes)
                            if target is None:
                                break
                        else:
                            break
                    
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
        
        # 4号腈（连接度=1）端口: [{23,24,25,10}]
        su4_nodes = self._get_nodes_by_su_type(nodes, 4)
        priority_list = [23, 24, 25, 10]
        
        for node in su4_nodes:
            if node.is_hop1_empty():
                # 优先选择空1-hop且有剩余槽位的候选
                empty_candidates = self._get_empty_hop1_nodes(nodes, priority_list)
                empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                
                if empty_candidates:
                    target = self._choose_weighted_candidate(node, empty_candidates, priority_list, nodes=nodes)
                else:
                    # 选择有剩余槽位的节点
                    available_candidates = self._get_available_nodes(nodes, priority_list)
                    available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                    if available_candidates:
                        target = self._choose_weighted_candidate(node, available_candidates, priority_list, nodes=nodes)
                    else:
                        continue

                if target is None:
                    continue
                
                self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
    
    def _complete_remaining_aliphatic_aromatic(self, nodes: List[_NodeV3]):
        """i) 完成剩余脂肪碳(22-25)和芳香碳(10-13)"""
        # 先强制补全 11 号的外接脂肪/不饱和端口，避免 19/20/21 与 22/23/24/25
        # 在后续普通补全中被早早占满，导致 SU11 长期停在无外接锚点状态。
        self._complete_required_external_anchors(nodes, [11])
        
        # 先完成22号（末端，连接度=1）
        # SU22 端口: [{25,24,19,20,21,23,11,2,3,1,0,14,15,17}]
        su22_empty = self._get_empty_hop1_nodes(nodes, [22])
        priority_list = [23, 24, 25, 11, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17]
        
        for node in su22_empty:
            empty_candidates = self._get_empty_hop1_nodes(nodes, priority_list)
            empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
            
            if empty_candidates:
                target = self._choose_weighted_candidate(node, empty_candidates, priority_list, nodes=nodes)
            else:
                available_candidates = self._get_available_nodes(nodes, priority_list)
                available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                if available_candidates:
                    target = self._choose_weighted_candidate(node, available_candidates, priority_list, nodes=nodes)
                else:
                    continue

            if target is None:
                continue
            
            self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
        
        # 完成23号（连接度=2）- 包括空节点和不完整节点
        su23_nodes = [n for n in nodes if n.su_type == 23 and not n.is_hop1_complete()]
        priority_list = [23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17]
        self._complete_nodes_round_robin(
            nodes,
            su23_nodes,
            priority_list,
            priority_list,
            salt=821,
        )
        
        # 完成24号和25号（连接度=3/4）
        for su_type in [24, 25]:
            incomplete_nodes = [n for n in nodes if n.su_type == su_type and not n.is_hop1_complete()]
            self._complete_nodes_round_robin(
                nodes,
                incomplete_nodes,
                priority_list,
                priority_list,
                salt=840 + int(su_type),
            )

        # 22-25/15/17 的占用在前面已经变化，这里再补一次 11 号必需外接锚点，
        # 让后续芳香互补阶段只处理其余两个芳香端口。
        self._complete_required_external_anchors(nodes, [11])

        # 完成芳香结构（10-13互相补全，候选池包含全部芳香类型以满足SU10等端口规则）
        aromatic_types = [5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30]
        aromatic_priority = [13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30]
        
        for su_type in [11, 10, 12, 13]:
            incomplete_nodes = [n for n in nodes if n.su_type == su_type and not n.is_hop1_complete()]
            self._complete_nodes_round_robin(
                nodes,
                incomplete_nodes,
                aromatic_types,
                aromatic_priority,
                salt=900 + int(su_type),
            )
 
    def layer1_assign(self, H_init: torch.Tensor, S_target: torch.Tensor,
                      E_target: torch.Tensor,
                      eval_nmr: bool = True,
                      eval_output_dir: Optional[str] = None,
                      eval_lib_path: Optional[str] = None,
                      eval_hwhm: float = 1.0,
                      eval_allow_approx: bool = True,
                      build_variant: int = 0,
                      seed_nodes: Optional[List[_NodeV3]] = None,
                      enable_carbonyl_joint_adjust: bool = True,
                      carbonyl_joint_iterations: int = 3,
                      carbonyl_joint_max_adjustments: int = 3,
                      carbonyl_joint_pos_threshold: float = 0.08,
                      carbonyl_joint_neg_threshold: float = 0.08,
                      enable_hop1_adjust: bool = True,
                      hop1_adjust_iterations: int = 3,
                      hop1_neg_threshold: float = -0.5,
                      hop1_pos_threshold: float = 0.5) -> List[_NodeV3]:
        """
        Layer1: 为每个中心SU分配1-hop邻居
        
        职责（仅限1-hop分配，不涉及SU直方图调整）：
        1. 初始化分配：按a-i顺序进行固定连接的初始分配
        2. 互为1-hop：维护双向连接关系
        3. 合理分散：优先使用空1-hop节点，避免少数节点相互连接
        4. 兜底修复：补全所有剩余的1-hop槽位
        5. 可选的1-hop调整：基于差谱分析优化1-hop分配（enable_hop1_adjust=True）
        
        SU直方图的调整（羰基互转、SU9、O/N/S/X等）由Layer4负责。
        
        Args:
            H_init: 初始SU直方图
            S_target: 目标谱图
            E_target: 目标元素组成
            eval_nmr: 是否评估NMR
            eval_output_dir: 评估结果输出目录
            eval_lib_path: 子图库路径
            eval_hwhm: 谱峰半高宽
            eval_allow_approx: 是否允许近似匹配
            enable_carbonyl_joint_adjust: 是否启用羰基-锚点联合调整
            carbonyl_joint_iterations: 联合调整最大迭代次数
            carbonyl_joint_max_adjustments: 每轮最大换边次数
            carbonyl_joint_pos_threshold: 联合调整正峰相对阈值
            carbonyl_joint_neg_threshold: 联合调整负峰相对阈值
            enable_hop1_adjust: 是否启用1-hop调整
            hop1_adjust_iterations: 1-hop调整最大迭代次数
            hop1_neg_threshold: 负峰阈值
            hop1_pos_threshold: 正峰阈值
        """
        device = self.device
        H_init = H_init.to(device)
        S_target = S_target.to(device)
        E_target = E_target.to(device)
        self._current_E_target = E_target
        self._build_variant = int(build_variant)
    
        # 初始化全局节点池
        nodes = self._initialize_node_pool(H_init)
        self._init_runtime_node_index(nodes)
        print("\n" + "="*60)
        print(f"Layer1: 1-hop分配 ({len(nodes)}个节点)")
        print("="*60)
        self._restore_seed_topology(nodes, seed_nodes)
    
        # a) 32号X -> 8号/21号
        self._assign_fixed_halogen_X(nodes)
    
        # b) 31号硫醚 -> 7号/19号
        self._assign_fixed_thioether_S(nodes)
    
        # c) 0号氨基端、27号 -> 6号/20号
        self._assign_fixed_amine_N(nodes)
    
        # d) 2号醚端、28号、29号 -> 5号/19号
        self._assign_fixed_ether_O(nodes)
    
        # e) 0/1/2/3羰基端 -> 9/22/23/24/25/14/15/17
        self._assign_fixed_carbonyl(nodes)
    
        # f) 14/15/16双键配对，17/18三键配对
        self._assign_unsaturated_pairs(nodes)
    
        # f2) 26号杂环N和30号杂环S分配
        self._assign_heterocyclic_NS(nodes)
    
        # f3) 10号芳基取代碳的port3={4,10}分配
        self._assign_aryl_connections(nodes)
    
        # g) 完成固定连接结构的互为1-hop（芳香结构）
        self._complete_aromatic_hop1(nodes)
    
        # g2) 完成19/20/21号脂肪杂原子碳的剩余1-hop
        self._complete_aliphatic_hetero_hop1(nodes)
    
        # h) 完成不饱和结构的饱和端
        self._complete_unsaturated_saturated_end(nodes)
    
        # i) 完成剩余脂肪碳和芳香碳
        self._complete_remaining_aliphatic_aromatic(nodes)

        # i2) 兜底修复：尽量补全所有剩余的1-hop槽位
        self._repair_remaining_hop1_slots(nodes)
    
        # 一致性验证
        is_valid, errors = self.validate_graph_consistency(
            nodes=nodes, 
            H=H_init, 
            E_target=E_target,
            verbose=False
        )
        accept_ok, accept_meta = self.assess_consistency_acceptance(
            errors,
            max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
        )
        if not is_valid:
            print(f"  ⚠ 图结构存在{len(errors)}个不一致")
            self._print_consistency_errors(errors, prefix='Layer1')
        if bool(accept_ok) and int(accept_meta.get('n_soft', 0)) > 0:
            print(
                "  [Layer1 Soft Acceptance] "
                f"soft={int(accept_meta.get('n_soft', 0))}/{int(accept_meta.get('max_soft_errors', 0))}"
            )
        hard_structure_invalid = bool(int(accept_meta.get('n_hard', 0)) > 0)

        if eval_lib_path and bool(enable_carbonyl_joint_adjust) and not bool(hard_structure_invalid):
            try:
                pre_joint_nodes = copy.deepcopy(nodes)
                nodes, joint_summary = self.adjust_carbonyl_anchor_jointly(
                    nodes=nodes,
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    hwhm=eval_hwhm,
                    allow_approx=eval_allow_approx,
                    max_iterations=int(carbonyl_joint_iterations),
                    max_adjustments_per_iter=int(carbonyl_joint_max_adjustments),
                    pos_rel_threshold=float(carbonyl_joint_pos_threshold),
                    neg_rel_threshold=float(carbonyl_joint_neg_threshold),
                )
                n_joint = int(joint_summary.get('adjustments', 0))
                if n_joint > 0:
                    print(f"  Carbonyl联合调整: {n_joint}次")
                joint_ok, joint_errors = self.validate_graph_consistency(
                    nodes=nodes,
                    H=H_init,
                    E_target=E_target,
                    verbose=False,
                )
                joint_accept_ok, joint_accept_meta = self.assess_consistency_acceptance(
                    joint_errors,
                    max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
                )
                if not joint_accept_ok:
                    print(f"  [Carbonyl联合调整后校验失败] 回退到调整前状态，errors={len(joint_errors)}")
                    self._print_consistency_errors(joint_errors, prefix='Carbonyl')
                    nodes = pre_joint_nodes
                elif (not joint_ok) and int(joint_accept_meta.get('n_soft', 0)) > 0:
                    print(
                        "  [Carbonyl联合调整后软缺口容忍] "
                        f"soft={int(joint_accept_meta.get('n_soft', 0))}/{int(joint_accept_meta.get('max_soft_errors', 0))}"
                    )
            except Exception as e:
                print(f"  [Carbonyl联合调整失败] {e}")
        elif bool(enable_carbonyl_joint_adjust) and bool(hard_structure_invalid):
            print("  [Carbonyl联合调整跳过] 当前Layer1图仍存在硬结构约束缺口")
        
        # NMR评估
        if eval_nmr and eval_lib_path:
            round_metrics = {
                'r2': 0.0,
                'r2_carbonyl': 0.0,
                'r2_aromatic': 0.0,
                'r2_aliphatic': 0.0,
                'matched_ratio': 0.0,
            }
            try:
                round_metrics = self.evaluate_layer1_nmr_with_library(
                    nodes=nodes,
                    S_target=S_target,
                    lib_path=eval_lib_path,
                    output_dir=eval_output_dir,
                    hwhm=eval_hwhm,
                    allow_approx=eval_allow_approx,
                )
            except Exception as e:
                print(f"  [NMR评估失败] {e}")
        # 可选的1-hop调整（基于差谱分析，Layer1.5）
        if enable_hop1_adjust and not bool(hard_structure_invalid):
            try:
                pre_adjust_nodes = copy.deepcopy(nodes)
                nodes, adjust_summary = self.adjust_hop1_based_on_spectrum(
                    nodes=nodes,
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    output_dir=eval_output_dir,
                    hwhm=eval_hwhm,
                    allow_approx=eval_allow_approx,
                    neg_threshold=hop1_neg_threshold,
                    pos_threshold=hop1_pos_threshold,
                    max_iterations=hop1_adjust_iterations,
                )
                total_adj = adjust_summary.get('total_adjustments', 0)
                if total_adj > 0:
                    print(f"  Hop1调整: {total_adj}次")
                adjust_ok, adjust_errors = self.validate_graph_consistency(
                    nodes=nodes,
                    H=H_init,
                    E_target=E_target,
                    verbose=False,
                )
                adjust_accept_ok, adjust_accept_meta = self.assess_consistency_acceptance(
                    adjust_errors,
                    max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
                )
                if not adjust_accept_ok:
                    print(f"  [Hop1调整后校验失败] 回退到调整前状态，errors={len(adjust_errors)}")
                    self._print_consistency_errors(adjust_errors, prefix='Hop1')
                    nodes = pre_adjust_nodes
                elif (not adjust_ok) and int(adjust_accept_meta.get('n_soft', 0)) > 0:
                    print(
                        "  [Hop1调整后软缺口容忍] "
                        f"soft={int(adjust_accept_meta.get('n_soft', 0))}/{int(adjust_accept_meta.get('max_soft_errors', 0))}"
                    )
            except Exception as e:
                print(f"  [Hop1调整失败] {e}")
        elif bool(enable_hop1_adjust) and bool(hard_structure_invalid):
            print("  [Hop1调整跳过] 当前Layer1图仍存在硬结构约束缺口")

        final_ok, final_errors = self.validate_graph_consistency(
            nodes=nodes,
            H=H_init,
            E_target=E_target,
            verbose=False,
        )
        final_accept_ok, final_accept_meta = self.assess_consistency_acceptance(
            final_errors,
            max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
        )
        final_hard_gap = bool(int(final_accept_meta.get('n_hard', 0)) > 0)
        final_soft_overflow = bool(int(final_accept_meta.get('n_soft', 0)) > int(final_accept_meta.get('max_soft_errors', 0)))
        if (not bool(final_accept_ok)) and (bool(final_hard_gap) or bool(final_soft_overflow)):
            try:
                print("  [Layer1 Final Repair] 检测到剩余1-hop缺口，执行最终结构修复")
                self._complete_required_external_anchors(nodes, [11, 19, 20, 21, 5, 6, 7, 8, 9])
                self._repair_remaining_hop1_slots(nodes)
                final_ok, final_errors = self.validate_graph_consistency(
                    nodes=nodes,
                    H=H_init,
                    E_target=E_target,
                    verbose=False,
                )
                final_accept_ok, final_accept_meta = self.assess_consistency_acceptance(
                    final_errors,
                    max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
                )
            except Exception as e:
                print(f"  [Layer1 Final Repair失败] {e}")
        if not final_accept_ok:
            print(f"  [Layer1 Final Check] 仍存在{len(final_errors)}个不一致")
            self._print_consistency_errors(final_errors, prefix='Final')
        elif (not final_ok) and int(final_accept_meta.get('n_soft', 0)) > 0:
            print(
                "  [Layer1 Final Check] 允许少量1-hop软缺口: "
                f"soft={int(final_accept_meta.get('n_soft', 0))}/{int(final_accept_meta.get('max_soft_errors', 0))}"
            )

        print("Layer1 完成\n")
        
        return nodes

    def _compute_layer1_difference_spectrum(self,
                                           nodes: List[_NodeV3],
                                           S_target: torch.Tensor,
                                           lib_path: Optional[str],
                                           hwhm: float,
                                           allow_approx: bool) -> Dict[str, object]:
        try:
            return self._nmr_evaluator.compute_difference_spectrum(
                nodes=nodes,
                S_target=S_target,
                lib_path=lib_path,
                hwhm=hwhm,
                allow_approx=allow_approx,
            )
        except Exception as e:
            _, ppm_axis = self._nmr_evaluator.resolve_inputs(S_target)
            return {
                'ppm': ppm_axis.detach().cpu().numpy(),
                'diff': np.zeros(int(ppm_axis.numel()), dtype=np.float64),
                'r2': 0.0,
                'alpha': 0.0,
                'n_peaks': 0,
                'error': str(e),
            }

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
                                       neg_rel_threshold: float = 0.08) -> Tuple[List[_NodeV3], Dict[str, object]]:
        """
        羰基类型 + SU9/脂肪锚点联合调整。

        操作表:
          - SU1/SU2: 9 -> 23/24/25
          - SU1/SU2: 23/24/25 -> 9

        判据表:
          - 160-170 负峰显著 && 172-180 正峰显著: 往脂肪锚点迁移
          - 160-170 正峰显著 && 172-180 负峰显著: 往 9 迁回

        该阶段只重连 hop1，不直接修改 H 直方图。
        """
        return self._refiner.adjust_carbonyl_anchor_jointly(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            lib_path=lib_path,
            hwhm=hwhm,
            allow_approx=allow_approx,
            max_iterations=max_iterations,
            max_adjustments_per_iter=max_adjustments_per_iter,
            pos_rel_threshold=pos_rel_threshold,
            neg_rel_threshold=neg_rel_threshold,
        )

    def evaluate_layer1_nmr_with_library(self,
                                         nodes: List[_NodeV3],
                                         S_target: torch.Tensor,
                                         lib_path: Optional[str] = None,
                                         output_dir: str = 'inverse_result',
                                         hwhm: float = 1.0,
                                         allow_approx: bool = True) -> Dict[str, float]:
        return self._nmr_evaluator.evaluate_with_library(
            nodes=nodes,
            S_target=S_target,
            lib_path=lib_path,
            output_dir=output_dir,
            hwhm=hwhm,
            allow_approx=allow_approx,
        )
    
    # ========================================================================
    # Layer1.5: 1-hop调整（基于差谱分析）
    # ========================================================================
    def adjust_hop1_based_on_spectrum(self,
                                       nodes: List[_NodeV3],
                                       S_target: torch.Tensor,
                                       E_target: torch.Tensor,
                                       lib_path: Optional[str] = None,
                                       output_dir: str = 'inverse_result',
                                       hwhm: float = 1.0,
                                       allow_approx: bool = True,
                                       neg_threshold: float = -0.5,
                                       pos_threshold: float = 0.5,
                                       max_iterations: int = 3,
                                       adjustment_groups: Optional[List[str]] = None) -> Tuple[List[_NodeV3], Dict]:
        """
        基于差谱分析调整1-hop连接
        
        流程：
        1. 计算当前差谱（目标-重建）
        2. 识别负峰（过度集中区域）和正峰（缺失结构区域）
        3. 对负峰区域的节点寻找替代1-hop组合（指向正峰区域）
        4. 执行替换并处理互为1-hop的级联更新
        5. 迭代调整直到收敛或达到最大迭代次数
        
        Args:
            nodes: 节点列表
            S_target: 目标谱图
            E_target: 目标元素组成
            lib_path: 子图库路径
            output_dir: 输出目录
            hwhm: 谱峰半高宽
            neg_threshold: 负峰阈值
            pos_threshold: 正峰阈值
            max_iterations: 最大迭代次数
            adjustment_groups: 调整组顺序
        
        Returns:
            (adjusted_nodes, summary): 调整后的节点和调整摘要
        """
        return self._refiner.adjust_hop1_based_on_spectrum(
            nodes=nodes,
            S_target=S_target,
            E_target=E_target,
            lib_path=lib_path,
            output_dir=output_dir,
            hwhm=hwhm,
            allow_approx=bool(allow_approx),
            neg_threshold=neg_threshold,
            pos_threshold=pos_threshold,
            max_iterations=max_iterations,
            adjustment_groups=adjustment_groups,
        )
