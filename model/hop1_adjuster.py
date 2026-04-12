import copy
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass

# SU分类
SU_CARBONYL = [0, 1, 2, 3, 4]
SU_AROMATIC = [5, 6, 7, 8, 9, 10, 11, 12, 13]
SU_UNSATURATED = [14, 15, 16, 17, 18]
SU_ALIPHATIC = [19, 20, 21, 22, 23, 24, 25]
SU_HETERO = [26, 27, 28, 29, 30, 31, 32]

# 调整优先级分组（按化学特性分组调整）
ADJUSTMENT_GROUPS = {
    'aromatic': SU_AROMATIC,
    'carbonyl': SU_CARBONYL,
    'unsaturated': SU_UNSATURATED,
    'aliphatic': SU_ALIPHATIC,
}


def _can_match_ports(neighbors: List[int], port_sets: List[Set[int]]) -> bool:
    if len(neighbors) != len(port_sets):
        return False
    if not neighbors:
        return True

    used = [False] * len(neighbors)
    order = sorted(range(len(port_sets)), key=lambda i: len(port_sets[i]))

    def dfs(port_pos: int) -> bool:
        if port_pos >= len(order):
            return True
        p = order[port_pos]
        allowed = port_sets[p]
        for i, n in enumerate(neighbors):
            if used[i]:
                continue
            if n not in allowed:
                continue
            used[i] = True
            if dfs(port_pos + 1):
                return True
            used[i] = False
        return False

    return dfs(0)


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
    
    def __init__(self, 
                 su_hop1_ranges_path: Optional[str] = None,
                 su_nmr_ranges_path: Optional[str] = None,
                 connection_rules: Optional[Dict] = None,
                 port_combinations: Optional[Dict[int, List[Set[int]]]] = None,
                 validate_connection_fn: Optional[Callable[[int, int, torch.Tensor], bool]] = None,
                 external_requirement_fn: Optional[Callable[[int, Counter], Tuple[bool, str]]] = None):
        """
        初始化调整器
        
        Args:
            su_hop1_ranges_path: su_hop1_nmr_range_filtered.csv路径
            su_nmr_ranges_path: su_nmr_common_range_filtered.csv路径  
            connection_rules: SU连接规则字典（可选）
            port_combinations: SU端口组合字典（可选）
            validate_connection_fn: 验证连接函数（可选）
            external_requirement_fn: 外部要求函数（可选）
        """
        # 默认路径
        base_dir = Path(__file__).resolve().parents[1] / 'z_library'
        if su_hop1_ranges_path is None:
            su_hop1_ranges_path = str(base_dir / 'su_hop1_nmr_range_filtered.csv')
        if su_nmr_ranges_path is None:
            su_nmr_ranges_path = str(base_dir / 'su_nmr_common_range_filtered.csv')
        
        # 加载1-hop模板NMR范围
        self.hop1_templates = self._load_hop1_templates(su_hop1_ranges_path)
        self.su_nmr_ranges = self._load_su_nmr_ranges(su_nmr_ranges_path)
        
        # 构建快速查找索引
        self._build_template_index()
        
        # 连接规则
        self.connection_rules = connection_rules or self._default_connection_rules()

        self.port_combinations = port_combinations
        self.validate_connection_fn = validate_connection_fn
        self.external_requirement_fn = external_requirement_fn
        
        # 统计
        self.stats = {
            'adjustments_attempted': 0,
            'adjustments_successful': 0,
            'cascade_updates': 0,
        }

    def _motif_preference_score(self, center_su: int, hop1_tuple: Tuple[int, ...]) -> float:
        center_su = int(center_su)
        ms = tuple(sorted(int(x) for x in hop1_tuple))
        cnt = Counter(ms)

        if center_su == 27:
            if cnt[6] == 1 and cnt[20] == 1:
                return 2.5
            if cnt[6] == 2 or cnt[20] == 2:
                return -2.0

        if center_su == 29:
            if cnt[5] == 1 and cnt[19] == 1:
                return 2.5
            if cnt[5] == 2 or cnt[19] == 2:
                return -2.0

        if center_su == 31:
            if cnt[7] == 1 and cnt[19] == 1:
                return 2.5
            if cnt[7] == 2 or cnt[19] == 2:
                return -2.0

        if center_su == 3:
            if cnt[9] >= 2:
                return -3.0
            if cnt[9] == 1 and len(ms) == 2:
                return 2.0

        if center_su == 2:
            if cnt[9] == 1 and cnt[19] == 1:
                return 2.5
            if cnt[9] == 1 and cnt[5] == 1:
                return -2.0
            if cnt[19] >= 1 and cnt[5] == 0:
                return 0.8

        return 0.0
    
    def _load_hop1_templates(self, path: str) -> pd.DataFrame:
        """加载1-hop模板NMR范围数据"""
        try:
            df = pd.read_csv(path)
            # 解析hop1_multiset字符串为元组
            def parse_multiset(s):
                if pd.isna(s):
                    return ()
                s = str(s).strip('[]"')
                if not s:
                    return ()
                return tuple(sorted(int(x) for x in s.split(',')))
            
            df['hop1_tuple'] = df['hop1_multiset'].apply(parse_multiset)
            return df
        except Exception:
            return pd.DataFrame()
    
    def _load_su_nmr_ranges(self, path: str) -> Dict[int, Dict]:
        """加载SU的NMR范围"""
        ranges = {}
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                su_idx = int(row['center_su_idx'])
                ranges[su_idx] = {
                    'mu_median': float(row['mu_median']),
                    'mu_common_min': float(row['mu_common_min']),
                    'mu_common_max': float(row['mu_common_max']),
                }
        except Exception:
            return ranges
        return ranges
    
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
    
    def _default_connection_rules(self) -> Dict:
        """默认连接规则（从inverse_pipeline_v3.py导入）"""
        # 简化版连接规则
        return {
            'max_degree': {i: 3 for i in range(33)},  # 默认最大连接度
        }

    def _required_degree(self, center_su: int) -> Optional[int]:
        if not isinstance(self.port_combinations, dict):
            return None
        port_sets = self.port_combinations.get(int(center_su))
        if not port_sets:
            return None
        return int(len(port_sets))

    def _get_neighbor_types(self, nodes: List, node) -> List[int]:
        return [int(nodes[int(nid)].su_type) for nid in list(node.hop1_ids)]

    def _is_hop1_valid_types(self, center_su: int, neighbor_types: List[int], E_target: Optional[torch.Tensor]) -> bool:
        center_su = int(center_su)
        if isinstance(self.port_combinations, dict):
            port_sets = self.port_combinations.get(center_su)
            if not port_sets:
                return False
            if int(len(neighbor_types)) != int(len(port_sets)):
                return False
            if not _can_match_ports(list(neighbor_types), port_sets):
                return False

        if callable(self.validate_connection_fn) and E_target is not None:
            for nb in neighbor_types:
                if not bool(self.validate_connection_fn(center_su, int(nb), E_target)):
                    return False

        if callable(self.external_requirement_fn):
            ok, _msg = self.external_requirement_fn(center_su, Counter(neighbor_types))
            if not bool(ok):
                return False

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
                                   node_peaks: pd.DataFrame,
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
        
        for _, row in node_peaks.iterrows():
            mu = row.get('mu', np.nan)
            if pd.isna(mu):
                continue
            
            # 检查mu是否在峰区域内
            if peak_region.ppm_min <= mu <= peak_region.ppm_max:
                global_id = int(row['global_id'])
                # 找对应节点
                node = None
                for n in nodes:
                    if n.global_id == global_id:
                        node = n
                        break
                
                if node is not None:
                    matched.append({
                        'node': node,
                        'global_id': global_id,
                        'center_su': int(row['center_su_idx']),
                        'hop1_ms': self._parse_hop1_string(row['hop1_ms']),
                        'mu': float(mu),
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
                               exclude_hop1: Optional[Set[Tuple[int, ...]]] = None) -> List[Dict]:
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
        
        # 找目标节点
        target_node = None
        for n in nodes:
            if n.global_id == target_node_id:
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
        if req_deg is not None and int(req_deg) != len(old_hop1):
            result['error'] = f"目标节点度数与端口规则不一致: node_deg={len(old_hop1)}, required={req_deg}"
            return result
        
        # 检查新的1-hop是否可行
        # 1. 检查SU类型是否存在于当前节点池
        available_sus = Counter(n.su_type for n in nodes)
        needed_sus = Counter(new_hop1_tuple)
        
        for su_type, count in needed_sus.items():
            if available_sus.get(su_type, 0) < count:
                result['error'] = f"SU类型{su_type}数量不足（需要{count}，可用{available_sus.get(su_type, 0)}）"
                return result
        
        # 2. 通过“2-edge swap”执行严格换位：
        #    remove (t-u) + (v-w), add (t-v) + (u-w)
        try:
            ok, affected = self._execute_strict_swaps(nodes, target_node, new_hop1_tuple, E_target, dry_run=dry_run)
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
                               dry_run: bool) -> Tuple[bool, List[int]]:
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
        if not self._is_hop1_valid_types(int(work_nodes[t].su_type), final_types, E_target):
            return False, []

        if not dry_run:
            for orig, updated in zip(nodes, work_nodes):
                orig.hop1_su = Counter(updated.hop1_su)
                orig.hop1_ids = list(updated.hop1_ids)
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

        for w in list(v_node.hop1_ids):
            w = int(w)
            if w == t or w == u or w == v:
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

            if not self._is_hop1_valid_types(int(t_node.su_type), t_types, E_target):
                continue
            if not self._is_hop1_valid_types(int(u_node.su_type), u_types, E_target):
                continue
            if not self._is_hop1_valid_types(int(v_node.su_type), v_types, E_target):
                continue
            if not self._is_hop1_valid_types(int(w_node.su_type), w_types, E_target):
                continue

            return True, int(w), [int(t), int(u), int(v), int(w)]

        return False, -1, []
    
    def _remove_hop1_edge(self, nodes: List, id1: int, id2: int):
        """移除一条双向1-hop边"""
        node1 = nodes[id1] if id1 < len(nodes) else None
        node2 = nodes[id2] if id2 < len(nodes) else None
        
        if node1 is None or node2 is None:
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
    
    # ========================================================================
    # 主调整流程
    # ========================================================================
    
    def adjust(self,
               nodes: List,
               node_peaks: pd.DataFrame,
               diff_spectrum: np.ndarray,
               ppm_axis: np.ndarray,
               E_target: Optional[torch.Tensor] = None,
               neg_threshold: float = -0.5,
               pos_threshold: float = 0.5,
               max_adjustments_per_group: int = 10,
               adjustment_groups: Optional[List[str]] = None) -> Tuple[List, Dict]:
        """
        主调整入口
        
        Args:
            nodes: 节点列表
            node_peaks: layer1_library_node_peaks.csv数据
            diff_spectrum: 差谱 (target - reconstructed)
            ppm_axis: ppm轴
            E_target: 目标边集（可选）
            neg_threshold: 负峰阈值
            pos_threshold: 正峰阈值
            max_adjustments_per_group: 每组最大调整数
            adjustment_groups: 调整组顺序 ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic']
        
        Returns:
            (adjusted_nodes, summary): 调整后的节点和摘要
        """
        if adjustment_groups is None:
            adjustment_groups = ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic']
        
        # 1. 分析差谱
        negative_peaks, positive_peaks = self.analyze_difference_spectrum(
            diff_spectrum, ppm_axis, neg_threshold, pos_threshold
        )
        
        if not negative_peaks or not positive_peaks:
            return nodes, {'adjustments': 0, 'negative_peaks': [], 'positive_peaks': []}
        
        total_adjustments = 0
        adjustment_details = []
        
        # 2. 按组进行调整
        for group_name in adjustment_groups:
            su_types = ADJUSTMENT_GROUPS.get(group_name, [])
            if not su_types:
                continue
            
            group_adjustments = self._adjust_group(
                nodes, node_peaks, 
                negative_peaks, positive_peaks,
                su_types, max_adjustments_per_group,
                E_target
            )
            
            total_adjustments += len(group_adjustments)
            adjustment_details.extend(group_adjustments)
        
        if total_adjustments > 0:
            print(f"[Hop1Adjuster] 完成{total_adjustments}次调整")
        
        summary = {
            'adjustments': total_adjustments,
            'negative_peaks': [(p.ppm_min, p.ppm_max, p.intensity) for p in negative_peaks],
            'positive_peaks': [(p.ppm_min, p.ppm_max, p.intensity) for p in positive_peaks],
            'details': adjustment_details,
            'stats': dict(self.stats),
        }
        
        return nodes, summary
    
    def _adjust_group(self,
                       nodes: List,
                       node_peaks: pd.DataFrame,
                       negative_peaks: List[PeakRegion],
                       positive_peaks: List[PeakRegion],
                       su_types: List[int],
                       max_adjustments: int,
                       E_target: Optional[torch.Tensor]) -> List[Dict]:
        """
        调整特定SU类型组
        """
        adjustments = []
        
        for neg_peak in negative_peaks:
            if len(adjustments) >= max_adjustments:
                break
            
            # 找落在负峰区域的节点
            nodes_in_region = self.find_nodes_in_peak_region(nodes, node_peaks, neg_peak)
            
            # 筛选属于当前调整组的节点
            group_nodes = [n for n in nodes_in_region if n['center_su'] in su_types]
            
            if not group_nodes:
                continue
            
            # 找相邻的正峰区域作为目标
            target_pos_peak = self._find_adjacent_positive_peak(neg_peak, positive_peaks)
            if target_pos_peak is None:
                continue
            
            # 尝试调整每个节点
            for node_info in group_nodes:
                if len(adjustments) >= max_adjustments:
                    break
                
                # 查找替代1-hop
                alternatives = self.find_alternative_hop1(
                    node_info['center_su'],
                    node_info['hop1_ms'],
                    (target_pos_peak.ppm_min, target_pos_peak.ppm_max)
                )
                
                if not alternatives:
                    continue
                
                # 依次尝试前几个候选，避免首个候选因严格swap失败而直接放弃该节点
                for best_alt in alternatives[: min(6, len(alternatives))]:
                    result = self.execute_hop1_replacement(
                        nodes, node_info['global_id'], 
                        best_alt['hop1_tuple'],
                        E_target=E_target,
                        dry_run=False
                    )
                    
                    if result['success']:
                        adjustments.append({
                            'node_id': node_info['global_id'],
                            'center_su': node_info['center_su'],
                            'old_hop1': node_info['hop1_ms'],
                            'new_hop1': best_alt['hop1_tuple'],
                            'old_mu': node_info['mu'],
                            'new_mu': best_alt['mu_median'],
                            'from_region': (neg_peak.ppm_min, neg_peak.ppm_max),
                            'to_region': (target_pos_peak.ppm_min, target_pos_peak.ppm_max),
                        })
                        break
        
        return adjustments
    
    def _find_adjacent_positive_peak(self,
                                      neg_peak: PeakRegion,
                                      positive_peaks: List[PeakRegion],
                                      max_distance_ppm: float = 20.0) -> Optional[PeakRegion]:
        """
        找与负峰相邻的正峰
        """
        best = None
        best_distance = float('inf')
        
        for pos_peak in positive_peaks:
            # 计算距离
            if pos_peak.ppm_max < neg_peak.ppm_min:
                distance = neg_peak.ppm_min - pos_peak.ppm_max
            elif pos_peak.ppm_min > neg_peak.ppm_max:
                distance = pos_peak.ppm_min - neg_peak.ppm_max
            else:
                distance = 0  # 重叠
            
            if distance < best_distance and distance <= max_distance_ppm:
                best_distance = distance
                best = pos_peak
        
        return best


# ============================================================================
# 便捷函数
# ============================================================================

def adjust_hop1_connections(nodes: List,
                            node_peaks_path: str,
                            spectrum_comparison_path: str,
                            E_target: Optional[torch.Tensor] = None,
                            su_hop1_ranges_path: Optional[str] = None,
                            neg_threshold: float = -0.5,
                            pos_threshold: float = 0.5) -> Tuple[List, Dict]:
    """
    便捷函数：调整1-hop连接
    
    Args:
        nodes: 节点列表
        node_peaks_path: layer1_library_node_peaks.csv路径
        spectrum_comparison_path: layer1_library_spectrum_comparison.csv路径
        E_target: 目标边集（可选）
        su_hop1_ranges_path: su_hop1_nmr_range_filtered.csv路径
        neg_threshold: 负峰阈值
        pos_threshold: 正峰阈值
    
    Returns:
        (adjusted_nodes, summary)
    """
    # 加载数据
    node_peaks = pd.read_csv(node_peaks_path)
    spec_df = pd.read_csv(spectrum_comparison_path)
    
    diff_spectrum = spec_df['difference'].values
    ppm_axis = spec_df['ppm'].values
    
    # 创建调整器
    adjuster = Hop1Adjuster(su_hop1_ranges_path=su_hop1_ranges_path)
    
    # 执行调整
    return adjuster.adjust(
        nodes, node_peaks, diff_spectrum, ppm_axis,
        E_target=E_target,
        neg_threshold=neg_threshold, pos_threshold=pos_threshold
    )


if __name__ == '__main__':
    adjuster = Hop1Adjuster()
    test_diff = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.8
    test_ppm = np.linspace(0, 200, 100)
    
    adjuster.analyze_difference_spectrum(
        test_diff, test_ppm, neg_threshold=-0.3, pos_threshold=0.3
    )
