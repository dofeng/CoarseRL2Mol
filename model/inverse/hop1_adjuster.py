import copy
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Dict, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass

try:
    from ..paths import Z_LIBRARY_DIR
    from ..shared.inverse_common import (
        get_port_pattern_degrees,
        iter_port_signatures,
        multiset_from_counter as _common_multiset_from_counter,
        multiset_l1_distance as _common_multiset_l1_distance,
    )
except ImportError:
    from model.paths import Z_LIBRARY_DIR
    from model.shared.inverse_common import (
        get_port_pattern_degrees,
        iter_port_signatures,
        multiset_from_counter as _common_multiset_from_counter,
        multiset_l1_distance as _common_multiset_l1_distance,
    )

def _normalize_hop1_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in values))


def can_match_ports_exact(neighbors: Sequence[int], port_sets: Sequence[Set[int]]) -> bool:
    return _can_match_ports_impl(neighbors, port_sets, require_full=True)


def can_match_ports_partial(neighbors: Sequence[int], port_sets: Sequence[Set[int]]) -> bool:
    return _can_match_ports_impl(neighbors, port_sets, require_full=False)


def _can_match_ports_impl(neighbors: Sequence[int],
                          port_sets: Sequence[Set[int]],
                          require_full: bool) -> bool:
    neighbors_i = [int(x) for x in neighbors]
    signatures = iter_port_signatures(port_sets)

    if not signatures:
        return False

    def _match_single_signature(ports: List[Set[int]]) -> bool:
        if require_full:
            if len(neighbors_i) != len(ports):
                return False
        elif len(neighbors_i) > len(ports):
            return False

        if not neighbors_i:
            return True

        options_by_neighbor = []
        for idx, n in enumerate(neighbors_i):
            options = [pi for pi, allowed in enumerate(ports) if int(n) in allowed]
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


def motif_penalty_alpha(su_type: int) -> float:
    su_i = int(su_type)
    if su_i in {10, 11, 12, 13}:
        return 0.70
    if su_i in {23, 24, 25}:
        return 0.40
    if su_i in {22, 26, 29, 30}:
        return 0.25
    return 0.15


def motif_preference_score(center_su: int, hop1_tuple: Sequence[int]) -> float:
    center_i = int(center_su)
    cnt = Counter(_normalize_hop1_tuple(hop1_tuple))

    if center_i == 27:
        if cnt[6] == 1 and cnt[20] == 1:
            return 2.5
        if cnt[6] == 2 or cnt[20] == 2:
            return -2.0

    if center_i == 29:
        if cnt[5] == 1 and cnt[19] == 1:
            return 2.5
        if cnt[5] == 2 or cnt[19] == 2:
            return -2.0

    if center_i == 31:
        if cnt[7] == 1 and cnt[19] == 1:
            return 2.5
        if cnt[7] == 2:
            return 1.5
        if cnt[19] == 2:
            return -3.0

    if center_i == 3:
        if cnt[9] >= 2:
            return -3.0
        if cnt[9] == 1 and len(hop1_tuple) == 2:
            return 2.0

    if center_i == 2:
        if cnt[9] == 1 and cnt[19] == 1:
            return 2.5
        if cnt[9] == 1 and cnt[5] == 1:
            return -2.0
        if cnt[19] >= 1 and cnt[5] == 0:
            return 0.8

    return 0.0


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


class Hop1Adjuster:
    """
    1-hop调整器
    
    基于差谱分析自动调整结构单元的1-hop连接，以优化NMR谱图匹配。
    """
    _hop1_template_df_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    
    def __init__(self, 
                 su_hop1_ranges_path: Optional[str] = None,
                 port_combinations: Optional[Dict[int, List[Set[int]]]] = None,
                 validate_connection_fn: Optional[Callable[[int, int, torch.Tensor], bool]] = None,
                 external_requirement_fn: Optional[Callable[[int, Counter], Tuple[bool, str]]] = None,
                 external_requirement_node_fn: Optional[Callable[[Any, Counter], Tuple[bool, str]]] = None):
        """
        初始化调整器
        
        Args:
            su_hop1_ranges_path: su_hop1_nmr_range_filtered.csv路径
            port_combinations: SU端口组合字典（可选）
            validate_connection_fn: 验证连接函数（可选）
            external_requirement_fn: 外部要求函数（可选）
        """
        # 默认路径
        base_dir = Z_LIBRARY_DIR
        if su_hop1_ranges_path is None:
            su_hop1_ranges_path = str(base_dir / 'su_hop1_nmr_range_filtered.csv')
        
        # 加载1-hop模板NMR范围
        self.hop1_templates = self._load_hop1_templates(su_hop1_ranges_path)
        
        # 构建快速查找索引
        self._build_template_index()

        self.port_combinations = port_combinations
        self.validate_connection_fn = validate_connection_fn
        self.external_requirement_fn = external_requirement_fn
        self.external_requirement_node_fn = external_requirement_node_fn
        
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

    def _motif_preference_score(self, center_su: int, hop1_tuple: Tuple[int, ...]) -> float:
        return float(motif_preference_score(center_su, hop1_tuple))
    
    def _load_hop1_templates(self, path: str) -> pd.DataFrame:
        """加载1-hop模板NMR范围数据"""
        try:
            path_obj = Path(path)
            stat = path_obj.stat()
            cache_key = (str(path_obj.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
            cached = self._hop1_template_df_cache.get(cache_key)
            if cached is not None:
                return cached

            df = pd.read_csv(path_obj)
            # 解析hop1_multiset字符串为元组
            def parse_multiset(s):
                if pd.isna(s):
                    return ()
                s = str(s).strip('[]"')
                if not s:
                    return ()
                return tuple(sorted(int(x) for x in s.split(',')))
            
            df['hop1_tuple'] = df['hop1_multiset'].apply(parse_multiset)
            self._hop1_template_df_cache.clear()
            self._hop1_template_df_cache[cache_key] = df
            return df
        except Exception:
            return pd.DataFrame()
    
    def _build_template_index(self):
        """构建模板快速查找索引"""
        # 按(center_su, hop1_tuple)索引
        self.template_by_key = {}
        # 按center_su索引所有可用的hop1组合
        self.hop1_by_su = defaultdict(list)
        
        if self.hop1_templates.empty:
            return
        
        for _, row in self.hop1_templates.iterrows():
            su_idx = int(row['center_su_idx'])
            hop1_tuple = row['hop1_tuple']
            mu_median = float(row['mu_median'])
            mu_min = float(row.get('mu_common_min', mu_median - 5))
            mu_max = float(row.get('mu_common_max', mu_median + 5))
            
            key = (su_idx, hop1_tuple)
            self.template_by_key[key] = {
                'mu_median': mu_median,
                'mu_min': mu_min,
                'mu_max': mu_max,
                'n_templates': int(row.get('n_templates', 1)),
                'sample_count': int(row.get('sample_count_total', 1)),
            }
            
            self.hop1_by_su[su_idx].append({
                'hop1_tuple': hop1_tuple,
                'mu_median': mu_median,
                'mu_min': mu_min,
                'mu_max': mu_max,
            })
    
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

    def _is_hop1_valid_types(self,
                             center_su: int,
                             neighbor_types: List[int],
                             E_target: Optional[torch.Tensor],
                             center_node: Optional[Any] = None) -> bool:
        center_su = int(center_su)
        if isinstance(self.port_combinations, dict):
            port_sets = self.port_combinations.get(center_su)
            if not port_sets:
                return False
            if not can_match_ports_exact(list(neighbor_types), port_sets):
                return False

        if callable(self.validate_connection_fn) and E_target is not None:
            for nb in neighbor_types:
                if not bool(self.validate_connection_fn(center_su, int(nb), E_target)):
                    return False

        if callable(self.external_requirement_node_fn) and center_node is not None:
            ok, _msg = self.external_requirement_node_fn(center_node, Counter(neighbor_types))
            if not bool(ok):
                return False
        elif callable(self.external_requirement_fn):
            ok, _msg = self.external_requirement_fn(center_su, Counter(neighbor_types))
            if not bool(ok):
                return False

        # SU31 must prioritize consuming SU7 when available; allow 31:[7,7]
        # even if the current hop1 template table does not contain that motif.
        if int(center_su) == 31:
            cnt = Counter(int(x) for x in neighbor_types)
            if int(cnt.get(7, 0)) >= 1:
                return True

        # 严格模式：如果该SU在模板范围表中出现过，则要求新hop1 multiset必须存在于模板表
        # 这样可保证级联更新后的节点仍然是“数据库里出现过的1-hop模式”
        if self.hop1_by_su.get(center_su):
            hop1_ms = tuple(sorted(int(x) for x in neighbor_types))
            if (center_su, hop1_ms) not in self.template_by_key:
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
                    center_ppm=region['center_ppm']
                ))
        
        # 找正峰区域（差谱>pos_threshold）
        pos_regions = self._find_continuous_regions(diff, ppm, lambda x: x > pos_threshold)
        for region in pos_regions:
            if region['ppm_max'] - region['ppm_min'] >= min_width_ppm:
                positive_peaks.append(PeakRegion(
                    ppm_min=region['ppm_min'],
                    ppm_max=region['ppm_max'],
                    intensity=region['max_val'],  # 最强正峰强度
                    center_ppm=region['center_ppm']
                ))
        
        # 按强度排序（负峰按绝对值降序，正峰按值降序）
        negative_peaks.sort(key=lambda p: p.intensity)  # 最负的排前面
        positive_peaks.sort(key=lambda p: -p.intensity)  # 最正的排前面
        
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
    
    def find_alternative_hop1(self,
                               center_su: int,
                               current_hop1: Tuple[int, ...],
                               target_mu_range: Tuple[float, float],
                               exclude_hop1: Optional[Set[Tuple[int, ...]]] = None,
                               motif_usage: Optional[Counter] = None) -> List[Dict]:
        """
        为指定SU查找能产生目标mu范围的替代1-hop组合
        
        Args:
            center_su: 中心SU类型
            current_hop1: 当前1-hop组合
            target_mu_range: 目标mu范围(min, max)
            exclude_hop1: 排除的1-hop组合集合
        
        Returns:
            候选1-hop组合列表
        """
        candidates = []
        exclude_hop1 = exclude_hop1 or set()
        target_min, target_max = target_mu_range
        
        # 获取该SU的所有可用1-hop组合
        available = self.hop1_by_su.get(center_su, [])
        
        for entry in available:
            hop1_tuple = entry['hop1_tuple']
            
            # 排除当前组合和已排除的
            if hop1_tuple == current_hop1 or hop1_tuple in exclude_hop1:
                continue
            
            mu_median = entry['mu_median']
            mu_min = entry['mu_min']
            mu_max = entry['mu_max']
            
            # 检查mu范围是否与目标重叠
            if mu_max < target_min or mu_min > target_max:
                continue
            
            # 计算匹配得分（mu_median越接近目标中心越好）
            target_center = (target_min + target_max) / 2
            distance = abs(mu_median - target_center)
            score = 1.0 / (1.0 + distance)
            hotspot_count = int((motif_usage or Counter()).get((int(center_su), tuple(hop1_tuple)), 0))
            hotspot_alpha = float(motif_penalty_alpha(int(center_su)))
            score /= (1.0 + float(hotspot_alpha) * float(hotspot_count))
            
            # 严格模式：中心节点1-hop度数必须保持不变
            len_diff = abs(len(hop1_tuple) - len(current_hop1))
            if len_diff != 0:
                continue
            
            candidates.append({
                'hop1_tuple': hop1_tuple,
                'mu_median': mu_median,
                'mu_min': mu_min,
                'mu_max': mu_max,
                'score': score,
            })
        
        # 按得分排序
        for entry in candidates:
            entry['score'] = float(entry['score']) + self._motif_preference_score(center_su, entry['hop1_tuple'])
        candidates.sort(key=lambda x: -x['score'])
        
        # 对羰基结构单元(SU0,1,2,3)，优先选择包含SU9的组合
        if center_su in [0, 1, 2, 3]:
            su9_candidates = [c for c in candidates if 9 in c['hop1_tuple']]
            other_candidates = [c for c in candidates if 9 not in c['hop1_tuple']]
            candidates = su9_candidates + other_candidates
        
        return candidates
    
    # ========================================================================
    # 1-hop替换执行
    # ========================================================================
    
    def execute_hop1_replacement(self,
                                  nodes: List,
                                  target_node_id: int,
                                  new_hop1_tuple: Tuple[int, ...],
                                  E_target: Optional[torch.Tensor] = None,
                                  dry_run: bool = False) -> Dict:
        """
        执行1-hop替换，处理互为1-hop的级联更新
        
        Args:
            nodes: 节点列表
            target_node_id: 目标节点ID
            new_hop1_tuple: 新的1-hop组合
            E_target: 目标边集（可选）
            dry_run: 是否仅模拟执行
        
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
        
        nodes_by_su = defaultdict(list)
        for n in nodes:
            nodes_by_su[int(n.su_type)].append(int(n.global_id))

        # 检查新的1-hop是否可行
        # 1. 检查SU类型是否存在于当前节点池
        needed_sus = Counter(new_hop1_tuple)
        
        for su_type, count in needed_sus.items():
            available_count = int(len(nodes_by_su.get(int(su_type), [])))
            if available_count < int(count):
                result['error'] = f"SU类型{su_type}数量不足（需要{count}，可用{available_count}）"
                return result
        
        # 2. 通过“2-edge swap”执行严格换位：
        #    remove (t-u) + (v-w), add (t-v) + (u-w)
        try:
            ok, affected = self._execute_strict_swaps(
                nodes,
                target_node,
                new_hop1_tuple,
                E_target,
                dry_run=dry_run,
                nodes_by_su=nodes_by_su,
            )
            if not ok:
                result['error'] = '无法找到满足规则的换位方案'
                return result
            result['success'] = True
            result['affected_nodes'] = affected
            if not dry_run:
                self.stats['adjustments_successful'] += 1
        except Exception as e:
            result['error'] = str(e)
        
        self.stats['adjustments_attempted'] += 1
        return result
    
    def _execute_strict_swaps(self,
                               nodes: List,
                               target_node,
                               new_hop1_tuple: Tuple[int, ...],
                               E_target: Optional[torch.Tensor],
                               dry_run: bool,
                               nodes_by_su: Optional[Dict[int, List[int]]] = None) -> Tuple[bool, List[int]]:
        t = int(target_node.global_id)
        work_nodes = copy.deepcopy(nodes)
        work_target = work_nodes[t]
        old_types = self._get_neighbor_types(work_nodes, work_target)
        new_types = list(int(x) for x in list(new_hop1_tuple))

        rem_counter = Counter(old_types) - Counter(new_types)
        add_counter = Counter(new_types) - Counter(old_types)
        if sum(rem_counter.values()) != sum(add_counter.values()):
            return False, []

        removals: List[int] = []
        for su_type, cnt in rem_counter.items():
            cnt = int(cnt)
            if cnt <= 0:
                continue
            cand_ids = [int(nid) for nid in list(work_target.hop1_ids) if int(work_nodes[int(nid)].su_type) == int(su_type)]
            if len(cand_ids) < cnt:
                return False, []
            removals.extend(cand_ids[:cnt])

        additions: List[int] = []
        for su_type, cnt in add_counter.items():
            additions.extend([int(su_type)] * int(cnt))

        affected_all: Set[int] = set([t])

        if nodes_by_su is None:
            nodes_by_su = defaultdict(list)
            for n in work_nodes:
                nodes_by_su[int(n.su_type)].append(int(n.global_id))

        cascade_updates = 0
        for add_su in additions:
            if not removals:
                return False, []
            u = int(removals.pop(0))

            ok = False
            for v in nodes_by_su.get(int(add_su), []):
                v = int(v)
                if v == t or v == u:
                    continue
                if v in work_target.hop1_ids:
                    continue

                success, w, aff = self._try_two_edge_swap(work_nodes, t=t, u=u, v=v, E_target=E_target)
                if not success:
                    continue

                ok = True
                affected_all.update(set(aff))
                self._remove_hop1_edge(work_nodes, t, u)
                self._remove_hop1_edge(work_nodes, v, w)
                self._add_hop1_edge(work_nodes, t, v)
                self._add_hop1_edge(work_nodes, u, w)
                cascade_updates += int(len(set(aff)))
                break

            if not ok:
                return False, []

        final_types = self._get_neighbor_types(work_nodes, work_nodes[t])
        if Counter(final_types) != Counter(new_types):
            return False, []
        if not self._is_hop1_valid_types(
            int(work_nodes[t].su_type),
            final_types,
            E_target,
            center_node=work_nodes[t],
        ):
            return False, []

        if not dry_run:
            for orig, updated in zip(nodes, work_nodes):
                orig.hop1_su = Counter(updated.hop1_su)
                orig.hop1_ids = list(updated.hop1_ids)
                try:
                    orig.fixed_hop1_ids = set(getattr(updated, 'fixed_hop1_ids', set()) or set())
                except Exception:
                    pass
            self.stats['cascade_updates'] += int(cascade_updates)

        return True, list(sorted(affected_all))

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

            if not self._is_hop1_valid_types(int(t_node.su_type), t_types, E_target, center_node=t_node):
                continue
            if not self._is_hop1_valid_types(int(u_node.su_type), u_types, E_target, center_node=u_node):
                continue
            if not self._is_hop1_valid_types(int(v_node.su_type), v_types, E_target, center_node=v_node):
                continue
            if not self._is_hop1_valid_types(int(w_node.su_type), w_types, E_target, center_node=w_node):
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
    
    def _add_hop1_edge(self, nodes: List, id1: int, id2: int):
        """添加一条双向1-hop边"""
        node1 = nodes[id1] if id1 < len(nodes) else None
        node2 = nodes[id2] if id2 < len(nodes) else None
        
        if node1 is None or node2 is None:
            return
        
        if id1 == id2:
            return
        
        # 检查是否已存在
        if id2 in node1.hop1_ids:
            return
        
        # 添加连接
        node1.hop1_su[node2.su_type] += 1
        node2.hop1_su[node1.su_type] += 1
        node1.hop1_ids.append(id2)
        node2.hop1_ids.append(id1)
    
