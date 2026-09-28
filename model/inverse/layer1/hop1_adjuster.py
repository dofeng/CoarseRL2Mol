import copy
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Dict, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass

try:
    from .terminal_paths import terminal_path_reason
    from ...paths import Z_LIBRARY_DIR
    from ...shared.coarse_graph import PPM_AXIS, SU_DEFS
    from ...shared.inverse_common import (
        evaluate_mu_pi_assignments,
        get_port_pattern_degrees,
        iter_port_signature_objects,
        has_forbidden_fixed_connector_special_combo,
        multiset_from_counter as _common_multiset_from_counter,
        multiset_l1_distance as _common_multiset_l1_distance,
        port_entry_matches,
        signature_matches_center,
        save_node_peak_rows,
        save_spectrum_comparison,
        save_spectrum_figure,
        terminal_neighbor_count,
        reconstruction_peak_scale,
    )
except ImportError:
    from model.inverse.layer1.terminal_paths import terminal_path_reason
    from model.paths import Z_LIBRARY_DIR
    from model.shared.coarse_graph import PPM_AXIS, SU_DEFS
    from model.shared.inverse_common import (
        evaluate_mu_pi_assignments,
        get_port_pattern_degrees,
        iter_port_signature_objects,
        has_forbidden_fixed_connector_special_combo,
        multiset_from_counter as _common_multiset_from_counter,
        multiset_l1_distance as _common_multiset_l1_distance,
        port_entry_matches,
        signature_matches_center,
        save_node_peak_rows,
        save_spectrum_comparison,
        save_spectrum_figure,
        terminal_neighbor_count,
        reconstruction_peak_scale,
    )

def _normalize_hop1_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in values))


def can_match_ports_exact(neighbors: Sequence[Any],
                          port_sets: Sequence[Set[Any]],
                          center_node: Optional[Any] = None) -> bool:
    return _can_match_ports_impl(neighbors, port_sets, require_full=True, center_node=center_node)


def can_match_ports_partial(neighbors: Sequence[Any],
                            port_sets: Sequence[Set[Any]],
                            center_node: Optional[Any] = None) -> bool:
    return _can_match_ports_impl(neighbors, port_sets, require_full=False, center_node=center_node)


def _can_match_ports_impl(neighbors: Sequence[Any],
                          port_sets: Sequence[Set[Any]],
                          require_full: bool,
                          center_node: Optional[Any] = None) -> bool:
    neighbors_i = list(neighbors or [])
    if has_forbidden_fixed_connector_special_combo(center_node, neighbors_i):
        return False
    signatures = iter_port_signature_objects(port_sets)

    if not signatures:
        return False

    def _match_single_signature(signature) -> bool:
        if not signature_matches_center(signature, center_node):
            return False
        ports = list(signature.ports)
        if require_full:
            if len(neighbors_i) != len(ports):
                return False
        elif len(neighbors_i) > len(ports):
            return False

        if not neighbors_i:
            return True

        options_by_neighbor = []
        for idx, n in enumerate(neighbors_i):
            options = [
                pi for pi, allowed in enumerate(ports)
                if any(port_entry_matches(entry, n) for entry in allowed)
            ]
            if not options:
                return False
            options_by_neighbor.append((idx, options))

        options_by_neighbor.sort(key=lambda item: (len(item[1]), item[0]))
        used_ports = [False] * len(ports)

        def dfs(pos: int) -> bool:
            if pos >= len(options_by_neighbor):
                return True
            _, options = options_by_neighbor[pos]
            for port_idx in options:
                if used_ports[port_idx]:
                    continue
                used_ports[port_idx] = True
                if dfs(pos + 1):
                    return True
                used_ports[port_idx] = False
            return False

        return dfs(0)

    return any(_match_single_signature(sig) for sig in signatures)


def hop1_counter_to_multiset(hop1_su: Mapping[int, int]) -> Tuple[int, ...]:
    return tuple(_common_multiset_from_counter(Counter(hop1_su)))


def multiset_l1_distance(ms1: Sequence[int], ms2: Sequence[int]) -> int:
    return int(_common_multiset_l1_distance(
        tuple(int(x) for x in ms1),
        tuple(int(x) for x in ms2),
    ))


def build_motif_usage_from_pairs(pairs: Iterable[Tuple[int, Sequence[int]]]) -> Counter:
    usage = Counter()
    for center_su, hop1_values in pairs:
        hop1_ms = _normalize_hop1_tuple(hop1_values)
        if hop1_ms:
            usage[(int(center_su), hop1_ms)] += 1
    return usage


@dataclass
class PeakRegion:
    """峰区域描述"""
    ppm_min: float
    ppm_max: float
    intensity: float  
    center_ppm: float
    area: float = 0.0


class Hop1Adjuster:
    """Exact-template Layer1 spectrum service and transactional adjuster."""

    INVERSE_HWHM = 1.0
    REGION_ORDER: Tuple[Tuple[str, float, float], ...] = (
        ('0-60', 0.0, 60.0),
        ('30-90', 30.0, 90.0),
        ('60-120', 60.0, 120.0),
        ('90-150', 90.0, 150.0),
        ('120-180', 120.0, 180.0),
        ('150-210', 150.0, 210.0),
        ('180-240', 180.0, 240.0),
    )
    _library_index_cache: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    _common_range_cache: Dict[Tuple[str, int, int], Dict[int, Tuple[float, float]]] = {}

    def __init__(self,
                 template_lib_path: Optional[str] = None,
                 common_ranges_path: Optional[str] = None,
                 port_combinations: Optional[Dict[int, List[Set[int]]]] = None,
                 validate_connection_fn: Optional[Callable[[int, int, torch.Tensor], bool]] = None,
                 external_requirement_fn: Optional[Callable[[int, Counter], Tuple[bool, str]]] = None,
                 external_requirement_node_fn: Optional[Callable[[Any, Counter], Tuple[bool, str]]] = None,
                 enforce_external_requirements: bool = True,
                 exact_hop1_signatures: Optional[Set[Tuple[int, Tuple[int, ...]]]] = None,
                 exact_hop1_required_centers: Optional[Set[int]] = None,
                 carbon_su_set: Optional[Set[int]] = None,
                 device: str | torch.device = 'cpu',
                 strict_swap_branch_limit: int = 12,
                 aromatic_response_scale: float = 1.0):
        self.device = device
        self.aromatic_response_scale = float(aromatic_response_scale)
        self.carbon_su_set = set(int(x) for x in set(carbon_su_set or set()))
        self.template_lib_path = str(template_lib_path) if template_lib_path else None
        self.library_index = self._load_template_library_index(self.template_lib_path)
        self.shift_options_by_su = dict(
            self.library_index.get('center_options', {}) or {}
        )
        if common_ranges_path is None:
            common_ranges_path = str(Z_LIBRARY_DIR / 'su_nmr_common_range_filtered.csv')
        self.common_ranges = self._load_common_ranges(str(common_ranges_path))

        self.port_combinations = port_combinations
        self.validate_connection_fn = validate_connection_fn
        self.external_requirement_fn = external_requirement_fn
        self.external_requirement_node_fn = external_requirement_node_fn
        self.enforce_external_requirements = bool(enforce_external_requirements)
        supplied_signatures = {
            (
                int(center_su),
                tuple(sorted(int(x) for x in tuple(hop1_ms))),
            )
            for center_su, hop1_ms in set(exact_hop1_signatures or set())
        }
        library_signatures = set(
            self.library_index.get('median_by_signature', {}) or {}
        )
        self.exact_hop1_signatures = supplied_signatures or library_signatures
        self.exact_hop1_required_centers = set(
            int(x) for x in set(exact_hop1_required_centers or set())
        )
        if not self.exact_hop1_required_centers and self.carbon_su_set:
            self.exact_hop1_required_centers = set(self.carbon_su_set)
        self.strict_swap_branch_limit = max(1, int(strict_swap_branch_limit))
        
        # 统计
        self.stats = {
            'adjustments_attempted': 0,
            'adjustments_successful': 0,
            'cascade_updates': 0,
        }

    @staticmethod
    def _edge_is_fixed(nodes: List, id1: int, id2: int) -> bool:
        try:
            node1 = nodes[int(id1)]
            node2 = nodes[int(id2)]
        except Exception:
            return False
        return bool(
            int(id2) in set(getattr(node1, 'fixed_hop1_ids', set()) or set())
            or int(id1) in set(getattr(node2, 'fixed_hop1_ids', set()) or set())
        )

    @classmethod
    def _load_template_library_index(cls, path: Optional[str]) -> Dict[str, object]:
        if not path:
            return {'median_by_signature': {}, 'center_options': {}}
        path_obj = Path(path)
        stat = path_obj.stat()
        cache_key = (
            str(path_obj.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
        cached = cls._library_index_cache.get(cache_key)
        if cached is not None:
            return cached

        raw = torch.load(str(path_obj), map_location='cpu', weights_only=False)
        templates = raw.get('templates', {}) if isinstance(raw, dict) else {}
        aggregated: Dict[Tuple[int, Tuple[int, ...]], Dict[str, object]] = {}
        for template_key, template in templates.items():
            if not isinstance(template_key, tuple) or len(template_key) != 3:
                continue
            try:
                center_su = int(template_key[0])
                hop1_tuple = _normalize_hop1_tuple(template_key[1])
                center_mu = float(template.get('center_mu', 0.0))
            except Exception:
                continue
            if not np.isfinite(center_mu) or center_mu == 0.0:
                continue
            sample_count = int(template.get('sample_count', 0) or 0)
            if sample_count <= 0:
                samples_mu = dict(template.get('samples', {}) or {}).get('mu')
                sample_count = int(samples_mu.numel()) if torch.is_tensor(samples_mu) else 1
            entry = aggregated.setdefault(
                (center_su, hop1_tuple),
                {
                    'mu_values': [],
                    'weights': [],
                    'mu_min': center_mu,
                    'mu_max': center_mu,
                    'n_templates': 0,
                },
            )
            entry['mu_values'].append(center_mu)
            entry['weights'].append(max(1, sample_count))
            entry['mu_min'] = min(
                float(entry['mu_min']),
                float(template.get('mu_min', center_mu)),
            )
            entry['mu_max'] = max(
                float(entry['mu_max']),
                float(template.get('mu_max', center_mu)),
            )
            entry['n_templates'] = int(entry['n_templates']) + 1

        medians: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        center_options: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for signature, entry in aggregated.items():
            values = np.asarray(entry['mu_values'], dtype=np.float64)
            weights = np.asarray(entry['weights'], dtype=np.float64)
            order = np.argsort(values, kind='stable')
            values = values[order]
            weights = weights[order]
            pick = int(np.searchsorted(
                np.cumsum(weights),
                0.5 * float(weights.sum()),
                side='left',
            ))
            pick = min(max(pick, 0), int(values.size) - 1)
            median = float(values[pick])
            medians[signature] = median
            center_options[int(signature[0])].append({
                'hop1_tuple': tuple(signature[1]),
                'mu_median': median,
                'mu_min': float(entry['mu_min']),
                'mu_max': float(entry['mu_max']),
                'n_templates': int(entry['n_templates']),
                'sample_weight_sum': float(weights.sum()),
            })
        for options in center_options.values():
            options.sort(
                key=lambda item: (
                    tuple(item['hop1_tuple']),
                    float(item['mu_median']),
                )
            )
        result = {
            'path': str(path_obj.resolve()),
            'median_by_signature': medians,
            'details_by_signature': aggregated,
            'center_options': dict(center_options),
        }
        cls._library_index_cache.clear()
        cls._library_index_cache[cache_key] = result
        return result

    @classmethod
    def _load_common_ranges(cls, path: str) -> Dict[int, Tuple[float, float]]:
        path_obj = Path(path)
        stat = path_obj.stat()
        cache_key = (
            str(path_obj.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
        cached = cls._common_range_cache.get(cache_key)
        if cached is not None:
            return cached
        frame = pd.read_csv(path_obj)
        ranges = {
            int(row.center_su_idx): (
                float(row.mu_common_min),
                float(row.mu_common_max),
            )
            for row in frame.itertuples(index=False)
        }
        cls._common_range_cache.clear()
        cls._common_range_cache[cache_key] = ranges
        return ranges

    def get_exact_hop1_median_index(self) -> Dict[Tuple[int, Tuple[int, ...]], float]:
        return dict(self.library_index.get('median_by_signature', {}) or {})

    def build_assignments(self, nodes: Sequence[Any]) -> Dict[int, Dict[str, object]]:
        medians = self.get_exact_hop1_median_index()
        details = dict(self.library_index.get('details_by_signature', {}) or {})
        assignments: Dict[int, Dict[str, object]] = {}
        for node in nodes:
            center_su = int(getattr(node, 'su_type', -1))
            if center_su not in self.carbon_su_set:
                continue
            hop1_tuple = hop1_counter_to_multiset(
                getattr(node, 'hop1_su', {}) or {}
            )
            signature = (center_su, hop1_tuple)
            mu = medians.get(signature)
            info = dict(details.get(signature, {}) or {})
            assignments[int(node.global_id)] = {
                'mu': None if mu is None else float(mu),
                'pi': None if mu is None else 1.0,
                'matched': mu is not None,
                'hop1_tuple': hop1_tuple,
                'n_templates': int(info.get('n_templates', 0)),
                'sample_weight_sum': float(sum(info.get('weights', []) or [])),
                'mu_min': float(info.get('mu_min', mu or 0.0)),
                'mu_max': float(info.get('mu_max', mu or 0.0)),
            }
        return assignments

    def build_peak_rows(
        self,
        nodes: Sequence[Any],
        assignments: Mapping[int, Dict[str, object]],
    ) -> Dict[str, object]:
        su_names = [name for name, _definition in SU_DEFS]
        mus: List[float] = []
        pis: List[float] = []
        peak_scales: List[float] = []
        rows: List[Dict[str, object]] = []
        matched_count = 0
        carbon_count = 0
        for node in nodes:
            center_su = int(getattr(node, 'su_type', -1))
            hop1_tuple = hop1_counter_to_multiset(
                getattr(node, 'hop1_su', {}) or {}
            )
            is_carbon = center_su in self.carbon_su_set
            record = dict(assignments.get(int(node.global_id), {}) or {})
            matched = bool(record.get('matched', False))
            mu = record.get('mu')
            pi = record.get('pi')
            if is_carbon:
                carbon_count += 1
                if matched:
                    matched_count += 1
                if mu is not None:
                    mus.append(float(mu))
                    pis.append(1.0)
                    peak_scales.append(
                        reconstruction_peak_scale(
                            center_su,
                            self.aromatic_response_scale,
                        )
                    )
            rows.append({
                'global_id': int(node.global_id),
                'center_su_idx': center_su,
                'center_su': (
                    su_names[center_su]
                    if 0 <= center_su < len(su_names)
                    else str(center_su)
                ),
                'hop1_ms': '[' + ' '.join(str(x) for x in hop1_tuple) + ']',
                'hop1_tuple': hop1_tuple,
                'matched': matched,
                'n_templates': int(record.get('n_templates', 0)),
                'sample_weight_sum': float(record.get('sample_weight_sum', 0.0)),
                'mu': float(mu) if mu is not None else np.nan,
                'pi': float(pi) if pi is not None else np.nan,
                'mu_min': float(record.get('mu_min', 0.0)),
                'mu_max': float(record.get('mu_max', 0.0)),
            })
        return {
            'rows': rows,
            'mus': mus,
            'pis': pis,
            'peak_scales': peak_scales,
            'matched_count': matched_count,
            'carbon_count': carbon_count,
        }

    def build_eval_snapshot(
        self,
        nodes: Sequence[Any],
        S_target: torch.Tensor,
    ) -> Dict[str, object]:
        S_eval = S_target.to(self.device).flatten()
        ppm_axis = PPM_AXIS.to(self.device).flatten()
        n = min(int(S_eval.numel()), int(ppm_axis.numel()))
        S_eval = S_eval[:n]
        ppm_axis = ppm_axis[:n]
        assignments = self.build_assignments(nodes)
        peak_data = self.build_peak_rows(nodes, assignments)
        snapshot: Dict[str, object] = {
            'ppm': ppm_axis.detach().cpu().numpy(),
            'rows': list(peak_data['rows']),
            'assignments': assignments,
            'matched_cnt': int(peak_data['matched_count']),
            'carbon_cnt': int(peak_data['carbon_count']),
            'matched_ratio': float(peak_data['matched_count']) / max(
                1.0,
                float(peak_data['carbon_count']),
            ),
            'n_peaks': len(peak_data['mus']),
        }
        snapshot.update(evaluate_mu_pi_assignments(
            S_target=S_eval,
            ppm_axis=ppm_axis,
            mus=peak_data['mus'],
            pis=peak_data['pis'],
            hwhm=self.INVERSE_HWHM,
            intensity_scale=1.0,
            device=self.device,
            unit_peak_intensity=True,
            peak_scales=peak_data['peak_scales'],
        ))
        return snapshot

    def apply_assignments_to_nodes(
        self,
        nodes: Sequence[Any],
        assignments: Mapping[int, Dict[str, object]],
    ) -> None:
        for node in nodes:
            record = dict(assignments.get(int(node.global_id), {}) or {})
            if record.get('mu') is None:
                continue
            node.mu = float(record['mu'])
            node.pi = 1.0

    def evaluate_with_library(
        self,
        nodes: Sequence[Any],
        S_target: torch.Tensor,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        snapshot = self.build_eval_snapshot(nodes, S_target)
        self.apply_assignments_to_nodes(nodes, snapshot['assignments'])
        if output_dir:
            save_node_peak_rows(snapshot['rows'], output_dir)
            if int(snapshot.get('n_peaks', 0)) > 0:
                save_spectrum_comparison(
                    S_target=snapshot['S_target'],
                    ppm=np.asarray(snapshot['ppm'], dtype=np.float64),
                    S_recon=snapshot['S_recon'].detach().cpu().numpy(),
                    diff=np.asarray(snapshot['diff'], dtype=np.float64),
                    output_dir=output_dir,
                    prefix='layer1_library',
                )
                save_spectrum_figure(
                    S_target=snapshot['S_target'],
                    S_recon=snapshot['S_recon'],
                    ppm=np.asarray(snapshot['ppm'], dtype=np.float64),
                    output_dir=output_dir,
                    layer_name='Layer1-Library',
                )
        metrics = {
            'r2': float(snapshot.get('r2', 0.0)),
            'r2_carbonyl': float(snapshot.get('r2_carbonyl', 0.0)),
            'r2_aromatic': float(snapshot.get('r2_aromatic', 0.0)),
            'r2_aliphatic': float(snapshot.get('r2_aliphatic', 0.0)),
            'matched_ratio': float(snapshot.get('matched_ratio', 0.0)),
        }
        print(
            '[Layer1-NMR-Eval] '
            f"R2={metrics['r2']:.4f} "
            f"(carbonyl={metrics['r2_carbonyl']:.4f}, "
            f"aromatic={metrics['r2_aromatic']:.4f}, "
            f"aliphatic={metrics['r2_aliphatic']:.4f})"
        )
        return metrics
    
    def _required_degree(self, center_su: int) -> Optional[Tuple[int, ...]]:
        if not isinstance(self.port_combinations, dict):
            return None
        port_sets = self.port_combinations.get(int(center_su))
        if not port_sets:
            return None
        degrees = get_port_pattern_degrees(port_sets)
        return tuple(int(x) for x in degrees) if degrees else None

    def _get_neighbor_types(self, nodes: List, node) -> List[int]:
        return [int(nodes[int(nid)].su_type) for nid in list(node.hop1_ids)]

    def _get_neighbor_nodes(self, nodes: List, node) -> List[Any]:
        return [
            nodes[int(nid)]
            for nid in list(getattr(node, 'hop1_ids', []) or [])
            if 0 <= int(nid) < len(nodes)
        ]

    @staticmethod
    def _node_target_degree(node) -> Optional[int]:
        try:
            val = getattr(node, 'target_hop1_degree', None)
            return int(val) if val is not None else None
        except Exception:
            return None

    def _terminal_neighbor_cap_ok(self,
                                  nodes: List,
                                  node,
                                  proposed_neighbor: Optional[Any] = None) -> bool:
        neighbor_nodes = self._get_neighbor_nodes(nodes, node)
        if proposed_neighbor is not None:
            neighbor_nodes.append(proposed_neighbor)
        return int(terminal_neighbor_count(neighbor_nodes)) <= 1

    def _is_hop1_valid_types(self,
                             center_su: int,
                             neighbor_types: List[int],
                             E_target: Optional[torch.Tensor],
                             center_node: Optional[Any] = None,
                             neighbor_nodes: Optional[List[Any]] = None) -> bool:
        center_su = int(center_su)
        match_neighbors = list(neighbor_nodes) if neighbor_nodes is not None else list(neighbor_types)
        if int(terminal_neighbor_count(match_neighbors)) > 1:
            return False
        if isinstance(self.port_combinations, dict):
            port_sets = self.port_combinations.get(center_su)
            if not port_sets:
                return False
            if not can_match_ports_exact(match_neighbors, port_sets, center_node=center_node):
                return False

        if center_su in self.exact_hop1_required_centers:
            exact_key = (center_su, tuple(sorted(int(x) for x in neighbor_types)))
            if exact_key not in self.exact_hop1_signatures:
                return False

        if callable(self.validate_connection_fn) and E_target is not None:
            for nb in neighbor_types:
                if not bool(self.validate_connection_fn(center_su, int(nb), E_target)):
                    return False

        if bool(self.enforce_external_requirements):
            if callable(self.external_requirement_node_fn) and center_node is not None:
                ok, _msg = self.external_requirement_node_fn(center_node, Counter(neighbor_types))
                if not bool(ok):
                    return False
            elif callable(self.external_requirement_fn):
                ok, _msg = self.external_requirement_fn(center_su, Counter(neighbor_types))
                if not bool(ok):
                    return False

        return True

    def analyze_difference_spectrum(self,
                                    diff: np.ndarray,
                                    ppm: np.ndarray,
                                    neg_threshold: float = -0.3,
                                    pos_threshold: float = 0.3,
                                    min_width_ppm: float = 2.0) -> Tuple[List[PeakRegion], List[PeakRegion]]:
        """
        分析差谱，识别负峰和正峰区域
        
        Args:
            diff: 差谱数组 (target - reconstructed)
            ppm: ppm轴
            neg_threshold: 负峰阈值（小于此值视为负峰）
            pos_threshold: 正峰阈值（大于此值视为正峰）
            min_width_ppm: 最小峰宽度(ppm)
        
        Returns:
            (negative_peaks, positive_peaks): 负峰和正峰区域列表
        """
        negative_peaks = []
        positive_peaks = []
        
        # 找负峰区域（差谱<neg_threshold）
        neg_regions = self._find_continuous_regions(diff, ppm, lambda x: x < neg_threshold)
        for region in neg_regions:
            if region['ppm_max'] - region['ppm_min'] >= min_width_ppm:
                negative_peaks.append(PeakRegion(
                    ppm_min=region['ppm_min'],
                    ppm_max=region['ppm_max'],
                    intensity=region['min_val'],  # 最强负峰强度
                    center_ppm=region['center_ppm'],
                    area=region['abs_area'],
                ))
        
        # 找正峰区域（差谱>pos_threshold）
        pos_regions = self._find_continuous_regions(diff, ppm, lambda x: x > pos_threshold)
        for region in pos_regions:
            if region['ppm_max'] - region['ppm_min'] >= min_width_ppm:
                positive_peaks.append(PeakRegion(
                    ppm_min=region['ppm_min'],
                    ppm_max=region['ppm_max'],
                    intensity=region['max_val'],  # 最强正峰强度
                    center_ppm=region['center_ppm'],
                    area=region['abs_area'],
                ))
        
        # 按强度排序（负峰按绝对值降序，正峰按值降序）
        negative_peaks.sort(key=lambda p: (-float(p.area), float(p.intensity)))
        positive_peaks.sort(key=lambda p: (-float(p.area), -float(p.intensity)))
        
        return negative_peaks, positive_peaks
    
    def _find_continuous_regions(self, 
                                  arr: np.ndarray, 
                                  ppm: np.ndarray,
                                  condition_fn) -> List[Dict]:
        """找连续满足条件的区域"""
        regions = []
        in_region = False
        start_idx = 0
        
        for i, val in enumerate(arr):
            if condition_fn(val):
                if not in_region:
                    in_region = True
                    start_idx = i
            else:
                if in_region:
                    in_region = False
                    regions.append(self._extract_region_info(arr, ppm, start_idx, i-1))
        
        # 处理末尾
        if in_region:
            regions.append(self._extract_region_info(arr, ppm, start_idx, len(arr)-1))
        
        return regions
    
    def _extract_region_info(self, arr: np.ndarray, ppm: np.ndarray, 
                              start_idx: int, end_idx: int) -> Dict:
        """提取区域信息"""
        region_arr = arr[start_idx:end_idx+1]
        region_ppm = ppm[start_idx:end_idx+1]
        
        # 找极值位置
        min_idx = np.argmin(region_arr)
        max_idx = np.argmax(region_arr)
        
        return {
            'ppm_min': float(region_ppm.min()),
            'ppm_max': float(region_ppm.max()),
            'min_val': float(region_arr.min()),
            'max_val': float(region_arr.max()),
            'center_ppm': float(region_ppm[min_idx if abs(region_arr.min()) > abs(region_arr.max()) else max_idx]),
            'abs_area': float(np.trapz(np.abs(region_arr), region_ppm)) if len(region_arr) > 1 else float(abs(region_arr[0])),
        }
    
    # ========================================================================
    # 节点-峰区域匹配
    # ========================================================================
    
    def find_nodes_in_peak_region(self,
                                   nodes: List,  # List[_NodeV3]
                                   node_peaks: Any,
                                   peak_region: PeakRegion) -> List[Dict]:
        """
        找出mu落在指定峰区域内的节点
        
        Args:
            nodes: 节点列表
            node_peaks: layer1_library_node_peaks.csv数据
            peak_region: 峰区域
        
        Returns:
            匹配的节点信息列表
        """
        matched = []
        node_by_id = {int(getattr(n, 'global_id', -1)): n for n in nodes}

        def _row_get(row: Any, key: str, default: Any = None) -> Any:
            if isinstance(row, dict):
                return row.get(key, default)
            return getattr(row, key, default)

        if isinstance(node_peaks, pd.DataFrame):
            if node_peaks.empty or 'mu' not in node_peaks.columns:
                return matched
            mu_values = pd.to_numeric(node_peaks['mu'], errors='coerce')
            mask = (
                mu_values.notna()
                & (mu_values >= float(peak_region.ppm_min))
                & (mu_values <= float(peak_region.ppm_max))
            )
            rows_iter = node_peaks.loc[mask].to_dict('records')
        else:
            rows_iter = list(node_peaks or [])

        for row in rows_iter:
            mu = _row_get(row, 'mu', np.nan)
            if pd.isna(mu):
                continue
            mu_f = float(mu)
            if not (float(peak_region.ppm_min) <= mu_f <= float(peak_region.ppm_max)):
                continue
            try:
                global_id = int(_row_get(row, 'global_id'))
            except Exception:
                continue
            node = node_by_id.get(int(global_id))
            if node is None:
                continue
            hop1_ms = _row_get(row, 'hop1_tuple', None)
            if hop1_ms is None:
                hop1_ms = self._parse_hop1_string(_row_get(row, 'hop1_ms', ''))
            else:
                hop1_ms = tuple(sorted(int(x) for x in tuple(hop1_ms)))
            matched.append({
                'node': node,
                'global_id': int(global_id),
                'center_su': int(_row_get(row, 'center_su_idx')),
                'hop1_ms': hop1_ms,
                'mu': float(mu_f),
            })

        return matched
    
    def _parse_hop1_string(self, s: str) -> Tuple[int, ...]:
        """解析hop1字符串如'[10 11]'为元组"""
        if pd.isna(s):
            return ()
        s = str(s).strip('[]')
        if not s:
            return ()
        return tuple(sorted(int(x) for x in s.split()))

    # ========================================================================
    # 候选1-hop查找
    # ========================================================================
    
    def find_shift_guided_hop1_targets(self,
                                       center_su: int,
                                       current_hop1: Tuple[int, ...],
                                       target_mu_range: Tuple[float, float],
                                       exclude_hop1: Optional[Set[Tuple[int, ...]]] = None,
                                       motif_usage: Optional[Counter] = None) -> List[Dict]:
        """Return exact same-degree templates that can cover a positive peak."""
        candidates = []
        exclude_hop1 = exclude_hop1 or set()
        target_min, target_max = target_mu_range
        available = self.shift_options_by_su.get(center_su, [])
        for entry in available:
            hop1_tuple = tuple(entry['hop1_tuple'])
            if hop1_tuple == current_hop1 or hop1_tuple in exclude_hop1:
                continue
            if len(hop1_tuple) != len(current_hop1):
                continue
            mu_median = entry['mu_median']
            mu_min = entry['mu_min']
            mu_max = entry['mu_max']
            median_in_target = bool(float(target_min) <= float(mu_median) <= float(target_max))
            range_overlaps_target = not bool(mu_max < target_min or mu_min > target_max)
            if not median_in_target and not range_overlaps_target:
                continue
            target_center = (target_min + target_max) / 2
            distance = abs(mu_median - target_center)
            candidates.append({
                'hop1_tuple': hop1_tuple,
                'mu_median': mu_median,
                'mu_min': mu_min,
                'mu_max': mu_max,
                'median_in_target': bool(median_in_target),
                'motif_usage': int((motif_usage or Counter()).get(
                    (int(center_su), tuple(hop1_tuple)),
                    0,
                )),
                'distance': float(distance),
            })
        direct = [item for item in candidates if bool(item['median_in_target'])]
        if direct:
            candidates = direct
        candidates.sort(key=lambda x: (
            int(x['motif_usage']),
            float(x['distance']),
            tuple(x['hop1_tuple']),
        ))
        return candidates
    
    # ========================================================================
    # 1-hop替换执行
    # ========================================================================
    
    def execute_hop1_replacement(self,
                                  nodes: List,
                                  target_node_id: int,
                                  new_hop1_tuple: Tuple[int, ...],
                                  E_target: Optional[torch.Tensor] = None,
                                  dry_run: bool = False,
                                  nodes_by_su: Optional[Mapping[int, Sequence[int]]] = None,
                                  return_undo_state: bool = False) -> Dict:
        """
        执行1-hop替换，处理互为1-hop的级联更新
        
        Args:
            nodes: 节点列表
            target_node_id: 目标节点ID
            new_hop1_tuple: 新的1-hop组合
            E_target: 目标边集（可选）
            dry_run: 是否仅模拟执行
            nodes_by_su: 可复用的 SU -> 节点ID 索引
            return_undo_state: 成功提交时是否返回轻量回滚状态
        
        Returns:
            执行结果字典
        """
        result = {
            'success': False,
            'target_node_id': target_node_id,
            'old_hop1': None,
            'new_hop1': new_hop1_tuple,
            'affected_nodes': [],
            'error': None,
        }
        
        # 找目标节点；常规节点池 global_id 与列表下标一致，保留回退以兼容外部构造。
        target_node = None
        if 0 <= int(target_node_id) < len(nodes):
            cand = nodes[int(target_node_id)]
            if int(getattr(cand, 'global_id', -1)) == int(target_node_id):
                target_node = cand
        if target_node is None:
            for n in nodes:
                if int(getattr(n, 'global_id', -1)) == int(target_node_id):
                    target_node = n
                    break
        
        if target_node is None:
            result['error'] = f"未找到节点ID={target_node_id}"
            return result
        
        old_hop1 = tuple(sorted(target_node.hop1_su.elements()))
        result['old_hop1'] = old_hop1

        if len(new_hop1_tuple) != len(old_hop1):
            result['error'] = f"中心节点1-hop度数必须保持不变: old={len(old_hop1)}, new={len(new_hop1_tuple)}"
            return result

        req_deg = self._required_degree(int(target_node.su_type))
        if req_deg is not None and int(len(old_hop1)) not in set(int(x) for x in req_deg):
            result['error'] = f"目标节点度数与端口规则不一致: node_deg={len(old_hop1)}, required={list(req_deg)}"
            return result
        if isinstance(self.port_combinations, dict):
            port_sets = self.port_combinations.get(int(target_node.su_type))
            if port_sets and not can_match_ports_exact(list(new_hop1_tuple), port_sets, center_node=target_node):
                result['error'] = f"新的1-hop组合不满足状态感知端口规则: {tuple(new_hop1_tuple)}"
                return result
        
        if nodes_by_su is None:
            nodes_by_su_local: Mapping[int, Sequence[int]] = defaultdict(list)
            for n in nodes:
                nodes_by_su_local[int(n.su_type)].append(int(n.global_id))
        else:
            nodes_by_su_local = nodes_by_su

        # 检查新的1-hop是否可行
        # 1. 检查SU类型是否存在于当前节点池
        needed_sus = Counter(new_hop1_tuple)
        
        for su_type, count in needed_sus.items():
            available_count = int(len(nodes_by_su_local.get(int(su_type), [])))
            if available_count < int(count):
                result['error'] = f"SU类型{su_type}数量不足（需要{count}，可用{available_count}）"
                return result
        
        # 2. 通过“2-edge swap”执行严格换位：
        #    remove (t-u) + (v-w), add (t-v) + (u-w)
        try:
            ok, affected, undo_state = self._execute_strict_swaps(
                nodes,
                target_node,
                new_hop1_tuple,
                E_target,
                dry_run=dry_run,
                nodes_by_su=nodes_by_su_local,
            )
            if not ok:
                result['error'] = '无法找到满足规则的换位方案'
                return result
            result['success'] = True
            result['affected_nodes'] = affected
            if bool(return_undo_state) and undo_state is not None:
                result['_undo_state'] = undo_state
            if not dry_run:
                self.stats['adjustments_successful'] += 1
        except Exception as e:
            result['error'] = str(e)
        
        self.stats['adjustments_attempted'] += 1
        return result

    def replaceable_resource_shortage(
        self,
        nodes: Sequence[Any],
        target_node_id: int,
        new_hop1_tuple: Sequence[int],
        nodes_by_su: Optional[Mapping[int, Sequence[int]]] = None,
    ) -> Dict[int, int]:
        """Return shortages in real, unlocked donor edges for a target motif."""
        target = nodes[int(target_node_id)]
        old_types = Counter(self._get_neighbor_types(list(nodes), target))
        additions = Counter(int(x) for x in new_hop1_tuple) - old_types
        if not additions:
            return {}
        if nodes_by_su is None:
            grouped: Mapping[int, Sequence[int]] = defaultdict(list)
            for node in nodes:
                grouped[int(node.su_type)].append(int(node.global_id))
            nodes_by_su = grouped

        existing = set(int(x) for x in list(target.hop1_ids))
        shortages: Dict[int, int] = {}
        for su_type, required in additions.items():
            available_edges = 0
            for donor_id in nodes_by_su.get(int(su_type), []):
                donor_i = int(donor_id)
                if donor_i == int(target_node_id) or donor_i in existing:
                    continue
                donor = nodes[donor_i]
                for neighbor_id in list(getattr(donor, 'hop1_ids', []) or []):
                    neighbor_i = int(neighbor_id)
                    if neighbor_i == int(target_node_id):
                        continue
                    if self._edge_is_fixed(list(nodes), donor_i, neighbor_i):
                        continue
                    available_edges += 1
            if int(available_edges) < int(required):
                shortages[int(su_type)] = int(required) - int(available_edges)
        return shortages
    
    @staticmethod
    def _capture_topology_state(nodes: Sequence[Any]) -> List[Tuple[Counter, List[int], Set[int]]]:
        return [
            (
                Counter(getattr(node, 'hop1_su', {}) or {}),
                list(getattr(node, 'hop1_ids', []) or []),
                set(getattr(node, 'fixed_hop1_ids', set()) or set()),
            )
            for node in nodes
        ]

    @staticmethod
    def _restore_topology_state(nodes: Sequence[Any],
                                state: Sequence[Tuple[Counter, List[int], Set[int]]]) -> None:
        for node, (hop1_su, hop1_ids, fixed_hop1_ids) in zip(nodes, state):
            node.hop1_su = Counter(hop1_su)
            node.hop1_ids = list(hop1_ids)
            try:
                node.fixed_hop1_ids = set(fixed_hop1_ids)
            except Exception:
                pass

    def _execute_strict_swaps(self,
                               nodes: List,
                               target_node,
                               new_hop1_tuple: Tuple[int, ...],
                               E_target: Optional[torch.Tensor],
                               dry_run: bool,
                               nodes_by_su: Optional[Mapping[int, Sequence[int]]] = None
                               ) -> Tuple[
                                   bool,
                                   List[int],
                                   Optional[List[Tuple[Counter, List[int], Set[int]]]],
                               ]:
        t = int(target_node.global_id)
        topology_state = self._capture_topology_state(nodes)
        work_nodes = nodes
        work_target = work_nodes[t]
        committed = False
        try:
            old_types = self._get_neighbor_types(work_nodes, work_target)
            new_types = list(int(x) for x in list(new_hop1_tuple))

            rem_counter = Counter(old_types) - Counter(new_types)
            add_counter = Counter(new_types) - Counter(old_types)
            if sum(rem_counter.values()) != sum(add_counter.values()):
                return False, [], None

            if nodes_by_su is None:
                nodes_by_su = defaultdict(list)
                for n in work_nodes:
                    nodes_by_su[int(n.su_type)].append(int(n.global_id))

            affected_all: Set[int] = {t}
            branch_limit = int(self.strict_swap_branch_limit)

            def _affected_valid() -> bool:
                if terminal_path_reason(work_nodes, start_ids=sorted(affected_all)):
                    return False
                if Counter(self._get_neighbor_types(work_nodes, work_nodes[t])) != Counter(new_types):
                    return False
                for node_id in sorted(affected_all):
                    node = work_nodes[int(node_id)]
                    neighbor_types = self._get_neighbor_types(work_nodes, node)
                    if not self._is_hop1_valid_types(
                        int(node.su_type),
                        neighbor_types,
                        E_target,
                        center_node=node,
                        neighbor_nodes=self._get_neighbor_nodes(work_nodes, node),
                    ):
                        return False
                return True

            def _search(remaining_remove: Counter, remaining_add: Counter) -> bool:
                if not remaining_add:
                    return _affected_valid()

                add_su = min(
                    (int(su) for su, count in remaining_add.items() if int(count) > 0),
                    key=lambda su: (len(nodes_by_su.get(int(su), [])), int(su)),
                )
                removable_ids = [
                    int(node_id)
                    for node_id in list(work_nodes[t].hop1_ids)
                    if int(remaining_remove.get(int(work_nodes[int(node_id)].su_type), 0)) > 0
                    and not self._edge_is_fixed(work_nodes, t, int(node_id))
                ]
                explored = 0
                for u in sorted(removable_ids):
                    removed_su = int(work_nodes[u].su_type)
                    for v_raw in nodes_by_su.get(int(add_su), []):
                        v = int(v_raw)
                        if v in {t, u} or v in work_nodes[t].hop1_ids:
                            continue
                        for w_raw in list(work_nodes[v].hop1_ids):
                            w = int(w_raw)
                            if w in {t, u, v}:
                                continue
                            if self._edge_is_fixed(work_nodes, v, w):
                                continue
                            if w in work_nodes[u].hop1_ids:
                                continue
                            if w in work_nodes[t].hop1_ids and w != u:
                                continue
                            explored += 1
                            if explored > int(branch_limit):
                                return False
                            branch_state = self._capture_topology_state(work_nodes)
                            previous_affected = set(affected_all)
                            self._remove_hop1_edge(work_nodes, t, u)
                            self._remove_hop1_edge(work_nodes, v, w)
                            if not self._add_hop1_edge_raw(work_nodes, t, v):
                                self._restore_topology_state(work_nodes, branch_state)
                                continue
                            if not self._add_hop1_edge_raw(work_nodes, u, w):
                                self._restore_topology_state(work_nodes, branch_state)
                                continue
                            affected_all.update({t, u, v, w})
                            next_remove = Counter(remaining_remove)
                            next_add = Counter(remaining_add)
                            next_remove[removed_su] -= 1
                            next_add[int(add_su)] -= 1
                            if next_remove[removed_su] <= 0:
                                del next_remove[removed_su]
                            if next_add[int(add_su)] <= 0:
                                del next_add[int(add_su)]
                            if _search(next_remove, next_add):
                                return True
                            self._restore_topology_state(work_nodes, branch_state)
                            affected_all.clear()
                            affected_all.update(previous_affected)
                return False

            if not _search(Counter(rem_counter), Counter(add_counter)):
                return False, [], None

            committed = not bool(dry_run)
            if committed:
                self.stats['cascade_updates'] += max(0, int(len(affected_all) - 1))
            undo_state = topology_state if committed else None
            return True, list(sorted(affected_all)), undo_state
        finally:
            if bool(dry_run) or not bool(committed):
                self._restore_topology_state(nodes, topology_state)

    def _try_two_edge_swap(self,
                            nodes: List,
                            t: int,
                            u: int,
                            v: int,
                            E_target: Optional[torch.Tensor]) -> Tuple[bool, int, List[int]]:
        t_node = nodes[int(t)]
        u_node = nodes[int(u)]
        v_node = nodes[int(v)]
        if int(u) not in t_node.hop1_ids:
            return False, -1, []
        if int(v) in t_node.hop1_ids:
            return False, -1, []
        if self._edge_is_fixed(nodes, int(t), int(u)):
            return False, -1, []

        for w in list(v_node.hop1_ids):
            w = int(w)
            if w == t or w == u or w == v:
                continue
            if self._edge_is_fixed(nodes, int(v), int(w)):
                continue
            w_node = nodes[int(w)]
            if int(w) in u_node.hop1_ids:
                continue
            if int(w) in t_node.hop1_ids and int(w) != u:
                continue

            t_new_ids = [int(x) for x in list(t_node.hop1_ids) if int(x) != int(u)] + [int(v)]
            u_new_ids = [int(x) for x in list(u_node.hop1_ids) if int(x) != int(t)] + [int(w)]
            v_new_ids = [int(x) for x in list(v_node.hop1_ids) if int(x) != int(w)] + [int(t)]
            w_new_ids = [int(x) for x in list(w_node.hop1_ids) if int(x) != int(v)] + [int(u)]

            if len(set(t_new_ids)) != len(t_new_ids):
                continue
            if len(set(u_new_ids)) != len(u_new_ids):
                continue
            if len(set(v_new_ids)) != len(v_new_ids):
                continue
            if len(set(w_new_ids)) != len(w_new_ids):
                continue

            t_types = [int(nodes[i].su_type) for i in t_new_ids]
            u_types = [int(nodes[i].su_type) for i in u_new_ids]
            v_types = [int(nodes[i].su_type) for i in v_new_ids]
            w_types = [int(nodes[i].su_type) for i in w_new_ids]

            t_neighbors = [nodes[int(i)] for i in t_new_ids]
            u_neighbors = [nodes[int(i)] for i in u_new_ids]
            v_neighbors = [nodes[int(i)] for i in v_new_ids]
            w_neighbors = [nodes[int(i)] for i in w_new_ids]

            if terminal_path_reason(nodes, neighbor_overrides={
                t: t_new_ids, u: u_new_ids, v: v_new_ids, w: w_new_ids,
            }, start_ids=[t, u, v, w]):
                continue

            if not self._is_hop1_valid_types(int(t_node.su_type), t_types, E_target, center_node=t_node, neighbor_nodes=t_neighbors):
                continue
            if not self._is_hop1_valid_types(int(u_node.su_type), u_types, E_target, center_node=u_node, neighbor_nodes=u_neighbors):
                continue
            if not self._is_hop1_valid_types(int(v_node.su_type), v_types, E_target, center_node=v_node, neighbor_nodes=v_neighbors):
                continue
            if not self._is_hop1_valid_types(int(w_node.su_type), w_types, E_target, center_node=w_node, neighbor_nodes=w_neighbors):
                continue

            return True, int(w), [int(t), int(u), int(v), int(w)]

        return False, -1, []
    
    def _remove_hop1_edge(self, nodes: List, id1: int, id2: int):
        """移除一条双向1-hop边"""
        node1 = nodes[id1] if id1 < len(nodes) else None
        node2 = nodes[id2] if id2 < len(nodes) else None
        
        if node1 is None or node2 is None:
            return
        if self._edge_is_fixed(nodes, int(id1), int(id2)):
            return
        
        # 更新hop1_su计数
        if node2.su_type in node1.hop1_su and node1.hop1_su[node2.su_type] > 0:
            node1.hop1_su[node2.su_type] -= 1
            if node1.hop1_su[node2.su_type] == 0:
                del node1.hop1_su[node2.su_type]
        
        if node1.su_type in node2.hop1_su and node2.hop1_su[node1.su_type] > 0:
            node2.hop1_su[node1.su_type] -= 1
            if node2.hop1_su[node1.su_type] == 0:
                del node2.hop1_su[node1.su_type]
        
        # 更新hop1_ids
        if id2 in node1.hop1_ids:
            node1.hop1_ids.remove(id2)
        if id1 in node2.hop1_ids:
            node2.hop1_ids.remove(id1)
        try:
            node1.fixed_hop1_ids.discard(int(id2))
            node2.fixed_hop1_ids.discard(int(id1))
        except Exception:
            pass

    @staticmethod
    def _add_hop1_edge_raw(nodes: List, id1: int, id2: int) -> bool:
        """Add a degree-preserving transaction edge; validate after the full bundle."""
        if int(id1) == int(id2):
            return False
        node1 = nodes[int(id1)]
        node2 = nodes[int(id2)]
        if int(id2) in node1.hop1_ids or int(id1) in node2.hop1_ids:
            return False
        node1.hop1_ids.append(int(id2))
        node2.hop1_ids.append(int(id1))
        node1.hop1_su[int(node2.su_type)] += 1
        node2.hop1_su[int(node1.su_type)] += 1
        return True
    
    def _add_hop1_edge(self, nodes: List, id1: int, id2: int) -> bool:
        """添加一条双向1-hop边"""
        if terminal_path_reason(nodes, proposed_edge=(int(id1), int(id2))):
            return False
        node1 = nodes[id1] if id1 < len(nodes) else None
        node2 = nodes[id2] if id2 < len(nodes) else None
        
        if node1 is None or node2 is None:
            return False
        
        if id1 == id2:
            return False
        
        # 检查是否已存在
        if id2 in node1.hop1_ids:
            return False
        if id1 in node2.hop1_ids:
            return False
        if not self._terminal_neighbor_cap_ok(nodes, node1, node2):
            return False
        if not self._terminal_neighbor_cap_ok(nodes, node2, node1):
            return False
        if not self._is_hop1_valid_types(
            int(node1.su_type),
            [int(nodes[int(nid)].su_type) for nid in list(node1.hop1_ids)] + [int(node2.su_type)],
            None,
            center_node=node1,
            neighbor_nodes=self._get_neighbor_nodes(nodes, node1) + [node2],
        ):
            return False
        if not self._is_hop1_valid_types(
            int(node2.su_type),
            [int(nodes[int(nid)].su_type) for nid in list(node2.hop1_ids)] + [int(node1.su_type)],
            None,
            center_node=node2,
            neighbor_nodes=self._get_neighbor_nodes(nodes, node2) + [node1],
        ):
            return False
        
        # 添加连接
        node1.hop1_su[node2.su_type] += 1
        node2.hop1_su[node1.su_type] += 1
        node1.hop1_ids.append(id2)
        node2.hop1_ids.append(id1)
        old_enforce_external = bool(self.enforce_external_requirements)
        self.enforce_external_requirements = False
        try:
            ok1 = self._is_hop1_valid_types(
                int(node1.su_type),
                [int(nodes[int(nid)].su_type) for nid in list(node1.hop1_ids)],
                None,
                center_node=node1,
                neighbor_nodes=self._get_neighbor_nodes(nodes, node1),
            )
            ok2 = self._is_hop1_valid_types(
                int(node2.su_type),
                [int(nodes[int(nid)].su_type) for nid in list(node2.hop1_ids)],
                None,
                center_node=node2,
                neighbor_nodes=self._get_neighbor_nodes(nodes, node2),
            )
        finally:
            self.enforce_external_requirements = old_enforce_external
        if not (bool(ok1) and bool(ok2)):
            self._remove_hop1_edge(nodes, id1, id2)
            return False
        return True

    @staticmethod
    def _unit_peak(
        ppm: np.ndarray,
        mu: Optional[float],
        *,
        peak_scale: float = 1.0,
    ) -> np.ndarray:
        if mu is None:
            return np.zeros_like(ppm, dtype=np.float64)
        delta = np.asarray(ppm, dtype=np.float64) - float(mu)
        return float(peak_scale) / (delta * delta + 1.0)

    @staticmethod
    def _region_abs_error(diff: np.ndarray, mask: np.ndarray) -> float:
        return float(np.abs(np.asarray(diff, dtype=np.float64)[mask]).sum())

    def _motif_usage(self, nodes: Sequence[Any]) -> Counter:
        return build_motif_usage_from_pairs(
            (
                int(node.su_type),
                hop1_counter_to_multiset(node.hop1_su),
            )
            for node in nodes
            if int(node.su_type) in self.carbon_su_set
        )

    def _positive_regions_for_su(
        self,
        center_su: int,
        negative_peak: PeakRegion,
        positive_peaks: Sequence[PeakRegion],
    ) -> List[PeakRegion]:
        common = self.common_ranges.get(int(center_su))
        if common is None:
            return []
        common_min, common_max = common
        candidates = [
            peak
            for peak in positive_peaks
            if not (
                float(peak.ppm_max) < float(common_min)
                or float(peak.ppm_min) > float(common_max)
            )
        ]
        return sorted(
            candidates,
            key=lambda peak: (
                abs(float(peak.center_ppm) - float(negative_peak.center_ppm)),
                -float(peak.area),
                float(peak.center_ppm),
            ),
        )

    def adjust_regions(
        self,
        nodes: List[Any],
        S_target: torch.Tensor,
        E_target: Optional[torch.Tensor],
        validator: Callable[[List[Any]], Tuple[bool, List[str]]],
        neg_peak_threshold: float = -0.8,
        pos_peak_threshold: float = 0.8,
        max_candidate_trials_per_region: int = 40,
        max_moves_per_region: int = 6,
        region_adjustment_cycles: int = 2,
    ) -> Tuple[List[Any], Dict[str, object]]:
        """Run overlapping regional adjustment passes until the cycle limit or convergence."""
        configured_cycles = max(1, int(region_adjustment_cycles))
        initial_snapshot = self.build_eval_snapshot(nodes, S_target)
        all_moves: List[Dict[str, object]] = []
        all_regions: List[Dict[str, object]] = []
        cycle_summaries: List[Dict[str, object]] = []
        converged = False

        for cycle_index in range(configured_cycles):
            cycle_number = int(cycle_index) + 1
            cycle_before = self.build_eval_snapshot(nodes, S_target)
            nodes, pass_summary = self._adjust_regions_once(
                nodes=nodes,
                S_target=S_target,
                E_target=E_target,
                validator=validator,
                neg_peak_threshold=float(neg_peak_threshold),
                pos_peak_threshold=float(pos_peak_threshold),
                max_candidate_trials_per_region=int(max_candidate_trials_per_region),
                max_moves_per_region=int(max_moves_per_region),
            )
            cycle_moves = [
                {'cycle': cycle_number, **dict(row)}
                for row in list(pass_summary.get('moves', []) or [])
            ]
            cycle_regions = [
                {'cycle': cycle_number, **dict(row)}
                for row in list(pass_summary.get('regions', []) or [])
            ]
            all_moves.extend(cycle_moves)
            all_regions.extend(cycle_regions)

            committed_moves = int(len(cycle_moves))
            cycle_after_r2 = float(pass_summary.get('final_r2', 0.0))
            cycle_summaries.append({
                'cycle': cycle_number,
                'committed_moves': committed_moves,
                'r2_before': float(cycle_before.get('r2', 0.0)),
                'r2_after': cycle_after_r2,
            })
            if committed_moves == 0:
                converged = True
                break

        final_snapshot = self.build_eval_snapshot(nodes, S_target)
        self.apply_assignments_to_nodes(nodes, final_snapshot['assignments'])
        return nodes, {
            'region_order': [name for name, _lo, _hi in self.REGION_ORDER],
            'configured_cycles': configured_cycles,
            'completed_cycles': len(cycle_summaries),
            'converged': bool(converged),
            'stop_reason': 'no_committed_moves' if converged else 'cycle_limit',
            'cycles': cycle_summaries,
            'regions': all_regions,
            'moves': all_moves,
            'total_adjustments': len(all_moves),
            'initial_r2': float(initial_snapshot.get('r2', 0.0)),
            'final_r2': float(final_snapshot.get('r2', 0.0)),
        }

    def _adjust_regions_once(
        self,
        nodes: List[Any],
        S_target: torch.Tensor,
        E_target: Optional[torch.Tensor],
        validator: Callable[[List[Any]], Tuple[bool, List[str]]],
        neg_peak_threshold: float = -0.8,
        pos_peak_threshold: float = 0.8,
        max_candidate_trials_per_region: int = 40,
        max_moves_per_region: int = 6,
    ) -> Tuple[List[Any], Dict[str, object]]:
        """Apply one ordered pass of regional batches with whole-batch rollback."""
        initial_snapshot = self.build_eval_snapshot(nodes, S_target)
        if int(initial_snapshot.get('matched_cnt', 0)) != int(
            initial_snapshot.get('carbon_cnt', 0)
        ):
            raise RuntimeError(
                'layer1_adjust_requires_exact_template_coverage: '
                f"matched={initial_snapshot.get('matched_cnt', 0)}/"
                f"{initial_snapshot.get('carbon_cnt', 0)}"
            )
        ppm = np.asarray(initial_snapshot['ppm'], dtype=np.float64)
        target = initial_snapshot['S_target'].detach().cpu().numpy().astype(np.float64)
        current_snapshot = initial_snapshot
        details: List[Dict[str, object]] = []
        region_summaries: List[Dict[str, object]] = []

        for region_name, region_min, region_max in self.REGION_ORDER:
            region_state = self._capture_topology_state(nodes)
            nodes_by_global_id = {
                int(getattr(item, 'global_id', -1)): item
                for item in nodes
            }
            batch_before = self.build_eval_snapshot(nodes, S_target)
            batch_before_r2 = float(batch_before.get('r2', 0.0))
            recon = batch_before['S_recon'].detach().cpu().numpy().astype(np.float64)
            diff = target - recon
            mask = (ppm >= float(region_min)) & (ppm <= float(region_max))
            loss_before = self._region_abs_error(diff, mask)
            assignments = dict(batch_before.get('assignments', {}) or {})
            attempts_seen: Set[Tuple[int, Tuple[int, ...], float, float]] = set()
            trials = 0
            tentative_moves = 0
            batch_details: List[Dict[str, object]] = []

            while (
                trials < int(max_candidate_trials_per_region)
                and tentative_moves < int(max_moves_per_region)
            ):
                region_diff = diff[mask]
                region_ppm = ppm[mask]
                negative_peaks, positive_peaks = self.analyze_difference_spectrum(
                    region_diff,
                    region_ppm,
                    neg_threshold=float(neg_peak_threshold),
                    pos_threshold=float(pos_peak_threshold),
                    min_width_ppm=0.0,
                )
                if not negative_peaks or not positive_peaks:
                    break

                motif_usage = self._motif_usage(nodes)
                node_candidates: List[Tuple[Tuple[object, ...], Any, PeakRegion]] = []
                for negative_peak in negative_peaks:
                    for node_id, assignment in assignments.items():
                        mu = assignment.get('mu')
                        if mu is None or not (
                            float(negative_peak.ppm_min)
                            <= float(mu)
                            <= float(negative_peak.ppm_max)
                        ):
                            continue
                        node = nodes[int(node_id)]
                        motif = hop1_counter_to_multiset(node.hop1_su)
                        node_candidates.append((
                            (
                                -int(motif_usage.get((int(node.su_type), motif), 0)),
                                -float(negative_peak.area),
                                int(node.global_id),
                            ),
                            node,
                            negative_peak,
                        ))
                node_candidates.sort(key=lambda item: item[0])

                moved = False
                nodes_by_su: Dict[int, List[int]] = defaultdict(list)
                for node in nodes:
                    nodes_by_su[int(node.su_type)].append(int(node.global_id))
                for _key, node, negative_peak in node_candidates:
                    current_hop1 = hop1_counter_to_multiset(node.hop1_su)
                    positive_for_su = self._positive_regions_for_su(
                        int(node.su_type),
                        negative_peak,
                        positive_peaks,
                    )
                    for positive_peak in positive_for_su:
                        alternatives = self.find_shift_guided_hop1_targets(
                            center_su=int(node.su_type),
                            current_hop1=current_hop1,
                            target_mu_range=(
                                float(positive_peak.ppm_min),
                                float(positive_peak.ppm_max),
                            ),
                            motif_usage=motif_usage,
                        )
                        for alternative in alternatives:
                            signature = (
                                int(node.global_id),
                                tuple(alternative['hop1_tuple']),
                                float(positive_peak.ppm_min),
                                float(positive_peak.ppm_max),
                            )
                            if signature in attempts_seen:
                                continue
                            shortage = self.replaceable_resource_shortage(
                                nodes,
                                int(node.global_id),
                                alternative['hop1_tuple'],
                                nodes_by_su=nodes_by_su,
                            )
                            alternative['resource_shortage'] = int(sum(shortage.values()))
                        alternatives.sort(key=lambda item: (
                            int(item.get('motif_usage', 0)),
                            float(item.get('distance', 0.0)),
                            int(item.get('resource_shortage', 0)),
                            tuple(item['hop1_tuple']),
                        ))
                        for alternative in alternatives:
                            signature = (
                                int(node.global_id),
                                tuple(alternative['hop1_tuple']),
                                float(positive_peak.ppm_min),
                                float(positive_peak.ppm_max),
                            )
                            if signature in attempts_seen:
                                continue
                            attempts_seen.add(signature)
                            trials += 1
                            old_assignments = assignments
                            result = self.execute_hop1_replacement(
                                nodes=nodes,
                                target_node_id=int(node.global_id),
                                new_hop1_tuple=tuple(alternative['hop1_tuple']),
                                E_target=E_target,
                                dry_run=False,
                                nodes_by_su=nodes_by_su,
                                return_undo_state=True,
                            )
                            if not bool(result.get('success', False)):
                                if trials >= int(max_candidate_trials_per_region):
                                    break
                                continue

                            step_valid, step_errors = validator(nodes)
                            if not step_valid or step_errors:
                                undo_state = result.get('_undo_state')
                                if undo_state is not None:
                                    self._restore_topology_state(nodes, undo_state)
                                if trials >= int(max_candidate_trials_per_region):
                                    break
                                continue

                            new_assignments = self.build_assignments(nodes)
                            for affected_id in result.get('affected_nodes', []):
                                old_mu = dict(old_assignments.get(int(affected_id), {}) or {}).get('mu')
                                new_mu = dict(new_assignments.get(int(affected_id), {}) or {}).get('mu')
                                affected_node = nodes_by_global_id.get(
                                    int(affected_id)
                                )
                                affected_su = int(
                                    getattr(affected_node, 'su_type', -1)
                                    if affected_node is not None else -1
                                )
                                recon -= self._unit_peak(
                                    ppm,
                                    old_mu,
                                    peak_scale=reconstruction_peak_scale(
                                        affected_su,
                                        self.aromatic_response_scale,
                                    ),
                                )
                                recon += self._unit_peak(
                                    ppm,
                                    new_mu,
                                    peak_scale=reconstruction_peak_scale(
                                        affected_su,
                                        self.aromatic_response_scale,
                                    ),
                                )
                            assignments = new_assignments
                            diff = target - recon
                            tentative_moves += 1
                            moved = True
                            batch_details.append({
                                'region': region_name,
                                'move': tentative_moves,
                                'trial': trials,
                                'global_id': int(node.global_id),
                                'center_su': int(node.su_type),
                                'old_hop1': list(current_hop1),
                                'new_hop1': list(alternative['hop1_tuple']),
                                'negative_peak_min': float(negative_peak.ppm_min),
                                'negative_peak_max': float(negative_peak.ppm_max),
                                'positive_peak_min': float(positive_peak.ppm_min),
                                'positive_peak_max': float(positive_peak.ppm_max),
                                'affected_nodes': list(result.get('affected_nodes', [])),
                            })
                            break
                        if moved or trials >= int(max_candidate_trials_per_region):
                            break
                    if moved or trials >= int(max_candidate_trials_per_region):
                        break
                if not moved:
                    break

            valid, validation_errors = validator(nodes)
            batch_after = self.build_eval_snapshot(nodes, S_target)
            loss_after = self._region_abs_error(
                np.asarray(batch_after.get('diff', []), dtype=np.float64),
                mask,
            )
            batch_after_r2 = float(batch_after.get('r2', 0.0))
            committed = bool(
                tentative_moves > 0
                and valid
                and not validation_errors
                and loss_before - loss_after > 1e-9
                and batch_after_r2 - batch_before_r2 > 1e-12
            )
            if not committed:
                self._restore_topology_state(nodes, region_state)
                current_snapshot = self.build_eval_snapshot(nodes, S_target)
            else:
                current_snapshot = batch_after
                details.extend(batch_details)
            region_summaries.append({
                'region': region_name,
                'trials': trials,
                'tentative_moves': tentative_moves,
                'committed_moves': tentative_moves if committed else 0,
                'committed': committed,
                'loss_before': loss_before,
                'loss_after': loss_after,
                'r2_before': batch_before_r2,
                'r2_after': batch_after_r2,
                'validation_errors': list(validation_errors or []),
            })

        final_snapshot = self.build_eval_snapshot(nodes, S_target)
        self.apply_assignments_to_nodes(nodes, final_snapshot['assignments'])
        return nodes, {
            'region_order': [name for name, _lo, _hi in self.REGION_ORDER],
            'regions': region_summaries,
            'moves': details,
            'total_adjustments': len(details),
            'initial_r2': float(initial_snapshot.get('r2', 0.0)),
            'final_r2': float(final_snapshot.get('r2', 0.0)),
        }
    
