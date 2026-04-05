import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict

from .coarse_graph import SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS
from .inverse_common import (
    _NodeV3, SU_CONNECTION_DEGREE, HOP1_PORT_COMBINATIONS, SU_FIXED_CONNECTIONS,
    lorentzian_spectrum, compute_r2_score, compute_segment_r2,
    visualize_spectrum_comparison, validate_connection, check_external_connection_requirement,
    evaluate_spectrum_reconstruction,
)
from .hop1_adjuster import Hop1Adjuster

def _can_partial_match_ports(neighbors: List[int], port_sets: List[set]) -> bool:
    """检查当前已分配的邻居是否可以被分配到端口集合的某个子集中
    
    与 hop1_adjuster._can_match_ports 不同，此函数允许部分填充
    （len(neighbors) <= len(port_sets)），用于构建过程中的增量验证。
    """
    if len(neighbors) > len(port_sets):
        return False
    if not neighbors:
        return True

    used_ports = [False] * len(port_sets)

    def dfs(ni: int) -> bool:
        if ni >= len(neighbors):
            return True
        n = neighbors[ni]
        for pi in range(len(port_sets)):
            if used_ports[pi]:
                continue
            if n not in port_sets[pi]:
                continue
            used_ports[pi] = True
            if dfs(ni + 1):
                return True
            used_ports[pi] = False
        return False

    return dfs(0)


class Layer1Assigner:
    """Layer1的1-hop分配器"""
    
    def __init__(self, device: str = 'cpu', vae_model=None, E_SU_tensor: torch.Tensor = None, layer0_estimator=None):
        """
        初始化Layer1分配器
        """
        self.device = device
        self.vae = vae_model
        self.E_SU = E_SU_tensor.to(device) if E_SU_tensor is not None else E_SU.to(device)
        self.layer0 = layer0_estimator
    
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

    def _current_neighbor_types(self, nodes: List[_NodeV3], node: _NodeV3) -> List[int]:
        return [int(nodes[nid].su_type) for nid in node.hop1_ids if 0 <= int(nid) < len(nodes)]

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
            if cnt[7] > 0 and cand_su == 19:
                return 4.0
            if cnt[19] > 0 and cand_su == 7:
                return 4.0
            if cnt[cand_su] > 0:
                return 0.15

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
        return self._choose_weighted_candidate(center, candidates, priority_list, nodes=nodes)
    
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
        
        # 6. 检查 per-port 连接规则合规性
        for n in nodes:
            port_sets = HOP1_PORT_COMBINATIONS.get(int(n.su_type))
            if not port_sets:
                continue
            neighbor_types = [int(nodes[nid].su_type) for nid in n.hop1_ids if nid < len(nodes)]
            if n.is_hop1_complete() and not _can_partial_match_ports(neighbor_types, port_sets):
                errors.append(
                    f"节点{n.global_id}(SU{n.su_type}): hop1={sorted(neighbor_types)} 不满足端口规则 "
                    f"{[sorted(s) for s in port_sets]}"
                )
        
        # 7. 检查羰基结构单元优先连接SU9的规则
        carbonyl_sus = [0, 1, 2, 3]
        for n in nodes:
            if int(n.su_type) in carbonyl_sus and n.is_hop1_complete():
                neighbor_types = [int(nodes[nid].su_type) for nid in n.hop1_ids if nid < len(nodes)]
                # 检查是否优先连接了SU9（如果SU9可用）
                su9_available = any(int(nodes[i].su_type) == 9 for i in range(len(nodes)) 
                                   if i != int(n.global_id) and nodes[i].remaining_hop1_slots() > 0)
                if su9_available and 9 not in neighbor_types:
                    errors.append(
                        f"羰基节点{n.global_id}(SU{n.su_type}): hop1={sorted(neighbor_types)} 未优先连接SU9"
                    )
        
        # 8. 检查元素组成（可选）
        if E_target is not None:
            E_pred = torch.matmul(H_actual.float().to(self.device), self.E_SU)
            E_target = E_target.to(self.device)
            E_diff = torch.abs(E_pred - E_target)
            rel_err = E_diff / (E_target + 1e-6)
            
            for i, elem_name in enumerate(['C', 'H', 'O', 'N', 'S', 'X']):
                if rel_err[i] > 0.05:  # 5%误差容忍
                    errors.append(f"元素{elem_name}误差过大: 预测{E_pred[i]:.1f} vs 目标{E_target[i]:.1f} (相对误差{rel_err[i]:.2%})")
        
        if verbose and errors:
            pass # pass # print("\n" + "=" * 60)
            pass # pass # print("⚠️  图一致性验证失败:")
            for i, err in enumerate(errors[:10], 1):  # 只显示前10个错误
                pass # pass # print(f"  {i}. {err}")
            if len(errors) > 10:
                pass # pass # print(f"  ... 还有 {len(errors) - 10} 个错误")
            pass # pass # print("=" * 60)
        
        return len(errors) == 0, errors
    
    # ========================================================================
    # Layer1: 1-hop分配辅助方法
    # ========================================================================
    
    def _initialize_node_pool(self, H_init: torch.Tensor) -> List[_NodeV3]:
        """初始化全局节点池，为每个SU实例创建节点对象
        """
        nodes = []
        global_id = 0
        
        for su_type in range(NUM_SU_TYPES):
            count = int(H_init[su_type].item())
            for _ in range(count):
                node = _NodeV3(global_id=global_id, su_type=su_type)
                nodes.append(node)
                global_id += 1
        
        return nodes
    
    def _add_bidirectional_hop1(self, nodes: List[_NodeV3], id1: int, id2: int):
        """添加双向1-hop连接
        """
        node1 = nodes[id1]
        node2 = nodes[id2]
        
        if id1 == id2:
            return
        
        if id2 in node1.hop1_ids or id1 in node2.hop1_ids:
            return
        
        # 添加互为1-hop（SU类型计数）
        node1.hop1_su[node2.su_type] += 1
        node2.hop1_su[node1.su_type] += 1
        
        # 记录全局ID（用于追踪具体连接）
        node1.hop1_ids.append(id2)
        node2.hop1_ids.append(id1)

    def _remove_bidirectional_hop1(self, nodes: List[_NodeV3], id1: int, id2: int) -> bool:
        """移除一条双向1-hop连接（仅移除一条，多重边会移除一次）"""
        node1 = nodes[id1]
        node2 = nodes[id2]

        # 先确认边存在
        if id2 not in node1.hop1_ids or id1 not in node2.hop1_ids:
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
        return True

    def _get_allowed_neighbor_types(self, su_type: int) -> List[int]:
        allowed = SU_FIXED_CONNECTIONS.get(su_type)
        if allowed is None:
            return list(range(NUM_SU_TYPES))
        if isinstance(allowed, dict):
            merged = []
            for v in allowed.values():
                merged.extend(list(v))
            return merged
        return list(allowed)

    def _can_add_hop1_connection(self, nodes: List[_NodeV3], node1: _NodeV3, node2: _NodeV3) -> bool:
        """检查在 node1 和 node2 之间添加1-hop连接是否满足双向 per-port 规则
        
        对两侧分别检查：将新邻居加入当前邻居列表后，是否仍可分配到端口中。
        """
        if node1.global_id == node2.global_id:
            return False

        # 检查 node1 侧
        port_sets1 = HOP1_PORT_COMBINATIONS.get(int(node1.su_type))
        if port_sets1:
            current_neighbors1 = [int(nodes[nid].su_type) for nid in node1.hop1_ids]
            proposed1 = current_neighbors1 + [int(node2.su_type)]
            if not _can_partial_match_ports(proposed1, port_sets1):
                return False

        # 检查 node2 侧
        port_sets2 = HOP1_PORT_COMBINATIONS.get(int(node2.su_type))
        if port_sets2:
            current_neighbors2 = [int(nodes[nid].su_type) for nid in node2.hop1_ids]
            proposed2 = current_neighbors2 + [int(node1.su_type)]
            if not _can_partial_match_ports(proposed2, port_sets2):
                return False

        return True

    def _repair_remaining_hop1_slots(self, nodes: List[_NodeV3]) -> None:
        """在主要步骤后做一次兜底修复：尽量让所有节点达到连接度上限
        
        使用 per-port 验证（_can_add_hop1_connection）确保所有操作合规。
        """

        remaining = [n for n in nodes if n.remaining_hop1_slots() > 0]
        if not remaining:
            return

        max_iters = len(remaining) * 20
        iters = 0

        def _can_direct_connect(u: _NodeV3, v: _NodeV3) -> bool:
            if u.global_id == v.global_id:
                return False
            if v.global_id in u.hop1_ids:
                return False
            if u.remaining_hop1_slots() <= 0 or v.remaining_hop1_slots() <= 0:
                return False
            return self._can_add_hop1_connection(nodes, u, v)

        def _is_locked(n: _NodeV3) -> bool:
            return n.su_type in {27, 28, 29, 31, 32}

        def _check_port_valid_after(node: _NodeV3) -> bool:
            """检查节点当前的邻居是否满足 per-port 规则"""
            ps = HOP1_PORT_COMBINATIONS.get(int(node.su_type))
            if not ps:
                return True
            cur = [int(nodes[nid].su_type) for nid in node.hop1_ids]
            return _can_partial_match_ports(cur, ps)

        while iters < max_iters:
            iters += 1
            remaining = [n for n in nodes if n.remaining_hop1_slots() > 0]
            if not remaining:
                break

            remaining.sort(key=lambda n: n.remaining_hop1_slots(), reverse=True)
            u = remaining[0]

            # 1) 尝试直接连接（per-port 验证）
            direct_candidates = [v for v in remaining[1:] if _can_direct_connect(u, v)]
            if direct_candidates:
                v = direct_candidates[0]
                self._add_bidirectional_hop1(nodes, u.global_id, v.global_id)
                continue

            # 1.5) 尝试"拆边"：(a-b) -> (u-a) + (u-b)
            if u.remaining_hop1_slots() >= 2:
                found_split = False
                preferred_edge_types = {10, 11, 12, 13}
                edge_endpoints = []
                for a in nodes:
                    if _is_locked(a):
                        continue
                    for b_id in a.hop1_ids:
                        if b_id <= a.global_id:
                            continue
                        b = nodes[b_id]
                        if _is_locked(b):
                            continue
                        edge_endpoints.append((a, b))

                for only_preferred in (True, False):
                    for a, b in edge_endpoints:
                        if only_preferred and (a.su_type not in preferred_edge_types or b.su_type not in preferred_edge_types):
                            continue
                        if a.global_id in u.hop1_ids or b.global_id in u.hop1_ids:
                            continue

                        # 模拟拆边后验证 per-port
                        # 移除 (a-b) 后 a 和 b 各少一个邻居，然后 u 连 a 和 b
                        # 先检查 u 能否连 a 和 b（增量）
                        # 还要检查 a 失去 b 后 + 得到 u 仍合规，b 同理

                        # 暂时模拟
                        a_neighbors_after = [int(nodes[nid].su_type) for nid in a.hop1_ids if nid != b.global_id] + [int(u.su_type)]
                        b_neighbors_after = [int(nodes[nid].su_type) for nid in b.hop1_ids if nid != a.global_id] + [int(u.su_type)]
                        u_neighbors_after = [int(nodes[nid].su_type) for nid in u.hop1_ids] + [int(a.su_type), int(b.su_type)]

                        ps_a = HOP1_PORT_COMBINATIONS.get(int(a.su_type))
                        ps_b = HOP1_PORT_COMBINATIONS.get(int(b.su_type))
                        ps_u = HOP1_PORT_COMBINATIONS.get(int(u.su_type))

                        if ps_a and not _can_partial_match_ports(a_neighbors_after, ps_a):
                            continue
                        if ps_b and not _can_partial_match_ports(b_neighbors_after, ps_b):
                            continue
                        if ps_u and not _can_partial_match_ports(u_neighbors_after, ps_u):
                            continue

                        # 执行拆边
                        ok = self._remove_bidirectional_hop1(nodes, a.global_id, b.global_id)
                        if not ok:
                            continue
                        self._add_bidirectional_hop1(nodes, u.global_id, a.global_id)
                        self._add_bidirectional_hop1(nodes, u.global_id, b.global_id)
                        found_split = True
                        break
                    if found_split:
                        break

                if found_split:
                    continue

            # 2) 尝试边交换：移除(a-b)，添加(u-a),(v-b)
            found_swap = False
            for v in remaining[1:]:
                if u.global_id == v.global_id:
                    continue
                if u.remaining_hop1_slots() <= 0 or v.remaining_hop1_slots() <= 0:
                    continue

                for a in nodes:
                    if a.remaining_hop1_slots() != 0:
                        continue
                    if _is_locked(a):
                        continue
                    if a.global_id == u.global_id or a.global_id == v.global_id:
                        continue
                    if a.global_id in u.hop1_ids:
                        continue

                    for b_id in list(a.hop1_ids):
                        b = nodes[b_id]
                        if b.remaining_hop1_slots() != 0:
                            continue
                        if _is_locked(b):
                            continue
                        if b.global_id == u.global_id or b.global_id == v.global_id:
                            continue
                        if b.global_id in v.hop1_ids:
                            continue

                        # 模拟交换后验证 per-port
                        u_nb_after = [int(nodes[nid].su_type) for nid in u.hop1_ids] + [int(a.su_type)]
                        v_nb_after = [int(nodes[nid].su_type) for nid in v.hop1_ids] + [int(b.su_type)]
                        a_nb_after = [int(nodes[nid].su_type) for nid in a.hop1_ids if nid != b.global_id] + [int(u.su_type)]
                        b_nb_after = [int(nodes[nid].su_type) for nid in b.hop1_ids if nid != a.global_id] + [int(v.su_type)]

                        ps_u = HOP1_PORT_COMBINATIONS.get(int(u.su_type))
                        ps_v = HOP1_PORT_COMBINATIONS.get(int(v.su_type))
                        ps_a = HOP1_PORT_COMBINATIONS.get(int(a.su_type))
                        ps_b = HOP1_PORT_COMBINATIONS.get(int(b.su_type))

                        if ps_u and not _can_partial_match_ports(u_nb_after, ps_u):
                            continue
                        if ps_v and not _can_partial_match_ports(v_nb_after, ps_v):
                            continue
                        if ps_a and not _can_partial_match_ports(a_nb_after, ps_a):
                            continue
                        if ps_b and not _can_partial_match_ports(b_nb_after, ps_b):
                            continue

                        # 执行交换
                        ok = self._remove_bidirectional_hop1(nodes, a.global_id, b.global_id)
                        if not ok:
                            continue
                        self._add_bidirectional_hop1(nodes, u.global_id, a.global_id)
                        self._add_bidirectional_hop1(nodes, v.global_id, b.global_id)
                        found_swap = True
                        break
                    if found_swap:
                        break
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    def _get_nodes_by_su_type(self, nodes: List[_NodeV3], su_type: int) -> List[_NodeV3]:
        """获取指定SU类型的所有节点"""
        return [n for n in nodes if n.su_type == su_type]
    
    def _get_empty_hop1_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中1-hop为空的节点"""
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and n.is_hop1_empty():
                    result.append(n)
        return result
    
    def _get_incomplete_hop1_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中1-hop不完整（有分配但未满）的节点"""
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and not n.is_hop1_empty() and not n.is_hop1_complete():
                    result.append(n)
        return result
    
    def _get_available_nodes(self, nodes: List[_NodeV3], su_types: List[int]) -> List[_NodeV3]:
        """获取指定SU类型中还有空闲1-hop槽位的节点"""
        result = []
        for su_type in su_types:
            for n in nodes:
                if n.su_type == su_type and n.remaining_hop1_slots() > 0:
                    result.append(n)
        return result
    
    def _choose_weighted_candidate(self, center: _NodeV3, candidates: List[_NodeV3], priority_list: List[int], nodes: Optional[List[_NodeV3]] = None) -> Optional[_NodeV3]:
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

        weights = []
        current_neighbors = self._current_neighbor_types(nodes, center) if nodes is not None else []
        for n in filtered:
            try:
                idx = priority_list.index(n.su_type)
            except ValueError:
                idx = len(priority_list)
            base = 1.0 / (1 + idx)
            w = base
            w /= (1 + center.hop1_su.get(n.su_type, 0))
            w /= (1 + n.hop1_su.get(center.su_type, 0))
            w *= self._bridge_candidate_bias(int(center.su_type), current_neighbors, int(n.su_type))
            weights.append(w)

        return random.choices(filtered, weights=weights, k=1)[0]
    
    # ========================================================================
    # Layer1: 固定连接分配方法（a-f）
    # ========================================================================
    
    def _assign_fixed_halogen_X(self, nodes: List[_NodeV3]):
        """a) 32号X -> 8号/21号"""
        x_nodes = self._get_nodes_by_su_type(nodes, 32)
        su8_nodes = self._get_nodes_by_su_type(nodes, 8)
        su21_nodes = self._get_nodes_by_su_type(nodes, 21)
        
        pass # pass # print(f"\n[a] 分配32号X (共{len(x_nodes)}个)")
        
        for x_node in x_nodes:
            target = self._pick_fixed_target(nodes, x_node, su8_nodes, [8])
            if target is None:
                target = self._pick_fixed_target(nodes, x_node, su21_nodes, [21])
            if target is not None:
                self._add_bidirectional_hop1(nodes, x_node.global_id, target.global_id)
    
    def _assign_fixed_thioether_S(self, nodes: List[_NodeV3]):
        """b) 31号硫醚 -> 7号/19号"""
        s_nodes = self._get_nodes_by_su_type(nodes, 31)
        su7_nodes = self._get_nodes_by_su_type(nodes, 7)
        su19_nodes = self._get_nodes_by_su_type(nodes, 19)
        
        pass # pass # print(f"\n[b] 分配31号S (共{len(s_nodes)}个)")
        
        random.shuffle(s_nodes)

        # 第一轮：每个31尽量先分一个7
        for s_node in s_nodes:
            target = self._pick_fixed_target(nodes, s_node, su7_nodes, [7])
            if target is not None:
                self._add_bidirectional_hop1(nodes, s_node.global_id, target.global_id)

        # 第二轮：如果7仍有剩余，再继续给31分第二个7
        for s_node in s_nodes:
            if s_node.remaining_hop1_slots() <= 0:
                continue
            target = self._pick_fixed_target(nodes, s_node, su7_nodes, [7])
            if target is not None:
                self._add_bidirectional_hop1(nodes, s_node.global_id, target.global_id)

        # 最后用19补足剩余端口
        for s_node in s_nodes:
            while s_node.remaining_hop1_slots() > 0:
                target = self._pick_fixed_target(nodes, s_node, su19_nodes, [19])
                if target is None:
                    break
                self._add_bidirectional_hop1(nodes, s_node.global_id, target.global_id)
    
    def _assign_fixed_amine_N(self, nodes: List[_NodeV3]):
        """c) 0号氨基端、27号 -> 6号/20号"""
        su0_nodes = self._get_nodes_by_su_type(nodes, 0)
        su27_nodes = self._get_nodes_by_su_type(nodes, 27)
        su6_nodes = self._get_nodes_by_su_type(nodes, 6)
        su20_nodes = self._get_nodes_by_su_type(nodes, 20)
        
        pass # pass # print(f"\n[c] 分配0号氨基端和27号 (0: {len(su0_nodes)}个, 27: {len(su27_nodes)}个)")
        
        random.shuffle(su27_nodes)

        # 先给每个27分一个6
        for n27 in su27_nodes:
            target = self._pick_fixed_target(nodes, n27, su6_nodes, [6])
            if target is not None:
                self._add_bidirectional_hop1(nodes, n27.global_id, target.global_id)

        # 如果6还有剩余，再继续给27分第二个6
        for n27 in su27_nodes:
            if n27.remaining_hop1_slots() <= 0:
                continue
            target = self._pick_fixed_target(nodes, n27, su6_nodes, [6])
            if target is not None:
                self._add_bidirectional_hop1(nodes, n27.global_id, target.global_id)

        # 剩余端口用20补足
        for n27 in su27_nodes:
            while n27.remaining_hop1_slots() > 0:
                target = self._pick_fixed_target(nodes, n27, su20_nodes, [20])
                if target is None:
                    break
                self._add_bidirectional_hop1(nodes, n27.global_id, target.global_id)
        
        # 再分配0号氨基端（需要1个邻居，羰基端后续补充）
        for n0 in su0_nodes:
            t = self._pick_fixed_target(nodes, n0, su6_nodes, [6])
            if t is None:
                t = self._pick_fixed_target(nodes, n0, su20_nodes, [20])
            if t is not None:
                self._add_bidirectional_hop1(nodes, n0.global_id, t.global_id)
    
    def _assign_fixed_ether_O(self, nodes: List[_NodeV3]):
        """d) 2号醚端、28号、29号 -> 5号/19号
        """
        su2_nodes = self._get_nodes_by_su_type(nodes, 2)
        su28_nodes = self._get_nodes_by_su_type(nodes, 28)
        su29_nodes = self._get_nodes_by_su_type(nodes, 29)
        su5_nodes_all = self._get_nodes_by_su_type(nodes, 5)
        su19_nodes_all = self._get_nodes_by_su_type(nodes, 19)
        
        # 过滤出有空闲槽位的5号（理论上此时都是空的，但以防万一）
        su5_nodes = [n for n in su5_nodes_all if n.remaining_hop1_slots() > 0]
        
        # 过滤出O专用的19号：有空闲槽位 且 未被31号(硫醚)占用
        # 31号在hop1_su中表示该19号已经是S专用的
        su19_nodes_o = [n for n in su19_nodes_all 
                        if n.remaining_hop1_slots() > 0 
                        and 31 not in n.hop1_su]
        
        # 统计S专用的19号数量（用于调试）
        su19_nodes_s = [n for n in su19_nodes_all if 31 in n.hop1_su]
        
        pass # pass # print(f"\n[d] 分配2号醚端、28号、29号 (2: {len(su2_nodes)}个, 28: {len(su28_nodes)}个, 29: {len(su29_nodes)}个)")
        pass # pass # print(f"    5号总计: {len(su5_nodes_all)}个, 可用: {len(su5_nodes)}个")
        pass # pass # print(f"    19号总计: {len(su19_nodes_all)}个, O专用(可用): {len(su19_nodes_o)}个, S专用: {len(su19_nodes_s)}个")
        
        # 计算需求量
        W_needed = len(su2_nodes) * 1 + len(su28_nodes) * 1 + len(su29_nodes) * 2
        W_available = len(su5_nodes) + len(su19_nodes_o)
        pass # pass # print(f"    需求量W={W_needed}, 可用量={W_available}")
        
        # 守恒约束检查
        if W_available < W_needed:
            pass # pass # print(f"    ⚠️  守恒约束违反! 需求{W_needed} > 可用{W_available}，差{W_needed - W_available}个")
            pass # pass # print(f"       请检查: H[5]+H[19](O专用) 应该等于 2*1+28*1+29*2={W_needed}")
        
        # 第一轮：29/28/2 都尽量先分一个5
        first_round_nodes = list(su29_nodes) + list(su28_nodes) + list(su2_nodes)
        random.shuffle(first_round_nodes)
        for node in first_round_nodes:
            if node.remaining_hop1_slots() <= 0:
                continue
            t = self._pick_fixed_target(nodes, node, su5_nodes, [5])
            if t is not None:
                self._add_bidirectional_hop1(nodes, node.global_id, t.global_id)

        # 第二轮：29如果还有空位且5还有剩余，再继续给29分第二个5
        for n29 in su29_nodes:
            if n29.remaining_hop1_slots() <= 0:
                continue
            t = self._pick_fixed_target(nodes, n29, su5_nodes, [5])
            if t is not None:
                self._add_bidirectional_hop1(nodes, n29.global_id, t.global_id)
        
        # 最后用19补足剩余端口
        for n29 in su29_nodes:
            while n29.remaining_hop1_slots() > 0:
                t = self._pick_fixed_target(nodes, n29, su19_nodes_o, [19])
                if t is None:
                    break
                self._add_bidirectional_hop1(nodes, n29.global_id, t.global_id)

        for n28 in su28_nodes:
            if n28.remaining_hop1_slots() <= 0:
                continue
            t = self._pick_fixed_target(nodes, n28, su19_nodes_o, [19])
            if t is not None:
                self._add_bidirectional_hop1(nodes, n28.global_id, t.global_id)
            else:
                pass # pass # print(f"  28(id={n28.global_id}) -> [警告:无法分配!]")
        
        # 最后分配2号醚端（需要1个邻居，羰基端后续补充）
        for n2 in su2_nodes:
            if n2.remaining_hop1_slots() <= 0:
                continue
            t = self._pick_fixed_target(nodes, n2, su19_nodes_o, [19])
            if t is not None:
                self._add_bidirectional_hop1(nodes, n2.global_id, t.global_id)
    
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
        
        pass # pass # print(f"\n[e] 分配羰基端 (0: {len(su0_nodes)}个, 1: {len(su1_nodes)}个, 2: {len(su2_nodes)}个, 3: {len(su3_nodes)}个)")
        pass # pass # print(f"    SU9可用: {len(su9_nodes)}个（必须全部被羰基消耗）")

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
        first_round = list(su0_nodes) + list(su1_nodes) + list(su2_nodes) + list(su3_nodes)
        random.shuffle(first_round)
        su9_assigned = 0
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
            su9_assigned += 1

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
                su9_assigned += 1

        pass # pass # print(f"    阶段1完成: {su9_assigned}/{len(su9_nodes)}个SU9已分配给羰基")

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

        try:
            su1_cnt = Counter()
            for n1 in su1_nodes:
                for k, v in n1.hop1_su.items():
                    su1_cnt[int(k)] += int(v)
            su3_cnt = Counter()
            for n3 in su3_nodes:
                for k, v in n3.hop1_su.items():
                    su3_cnt[int(k)] += int(v)
            pass # pass # print("\n[e] 羰基端分配统计:")
            pass # pass # print(f"  SU1 hop1 neighbors: {dict(su1_cnt)}")
            pass # pass # print(f"  SU3 hop1 neighbors: {dict(su3_cnt)}")
        except Exception as e:
            import logging
            logging.debug(f"Exception in carbonyl stats: {e}")
    
    def _assign_unsaturated_pairs(self, nodes: List[_NodeV3]):
        """f) 14/15/16双键配对，17/18三键配对
        """
        pass # pass # print(f"\n[f] 分配不饱和键配对")
        
        # 收集所有需要配对的双键节点（不饱和端连接度=1）
        su14_nodes = self._get_nodes_by_su_type(nodes, 14)
        su15_nodes = self._get_nodes_by_su_type(nodes, 15)
        su16_nodes = self._get_nodes_by_su_type(nodes, 16)
        
        # 按优先级15>16>14构建池，并随机打乱
        double_bond_pool = list(su15_nodes) + list(su16_nodes) + list(su14_nodes)
        random.shuffle(double_bond_pool)
        
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
                pass # pass # print(f"  警告：{node1.su_type}(id={node1.global_id})找不到配对伙伴")
        
        # 三键配对 (17, 18) - 17的不饱和端连接度=1，18的连接度=1
        su17_nodes = self._get_nodes_by_su_type(nodes, 17)
        su18_nodes = list(self._get_nodes_by_su_type(nodes, 18))
        
        # 先17-18配对
        for n17 in su17_nodes:
            if su18_nodes and n17.remaining_hop1_slots() > 0:
                n18 = su18_nodes.pop(0)
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
            n2 = remaining_17_no_triple.pop(0)
            self._add_bidirectional_hop1(nodes, n1.global_id, n2.global_id)
    
    def _assign_heterocyclic_NS(self, nodes: List[_NodeV3]):
        """f2) 26号杂环N和30号杂环S的1-hop分配（连接度=2）"""
        pass # pass # print(f"\n[f2] 26号杂环N和30号杂环S分配")
        
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
                    pass # pass # print(f"  警告：26号(id={node.global_id})找不到可用的芳香结构")
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
                    pass # pass # print(f"  30(id={node.global_id}) -> {target.su_type}(id={target.global_id})")
                else:
                    pass # pass # print(f"  警告：30号(id={node.global_id})找不到可用的芳香结构")
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
        
        pass # pass # print(f"\n[f3] 分配10号芳基取代的port3 (SU4: {len(su4_nodes)}个, SU10: {len(su10_nodes)}个)")
        
        # 阶段1：SU4 -> SU10 配对（SU4的唯一端口允许{23,24,25,10}，优先连10）
        su4_assigned = 0
        for n4 in su4_nodes:
            # 找一个有空闲槽位的SU10来连接
            available_10 = [n for n in su10_nodes 
                          if n.remaining_hop1_slots() > 0 
                          and n.global_id not in n4.hop1_ids
                          and n4.global_id not in n.hop1_ids]
            if available_10:
                target = available_10[0]
                self._add_bidirectional_hop1(nodes, n4.global_id, target.global_id)
                su4_assigned += 1
        
        pass # pass # print(f"    阶段1: {su4_assigned}/{len(su4_nodes)}个SU4已连接SU10")
        
        # 阶段2：剩余SU10之间10-10配对
        # 收集port3尚未填充的SU10（即还没有SU4或SU10邻居占据port3）
        su10_need_port3 = [n for n in su10_nodes 
                          if n.remaining_hop1_slots() > 0
                          and 4 not in n.hop1_su and 10 not in n.hop1_su]
        
        if len(su10_need_port3) % 2 != 0:
            pass # pass # print(f"    ⚠️ 需要10-10配对的SU10数量为奇数({len(su10_need_port3)})，"
        
        random.shuffle(su10_need_port3)
        su10_paired = 0
        while len(su10_need_port3) >= 2:
            n1 = su10_need_port3.pop(0)
            n2 = su10_need_port3.pop(0)
            self._add_bidirectional_hop1(nodes, n1.global_id, n2.global_id)
            su10_paired += 1
        
        pass # pass # print(f"    阶段2: {su10_paired}对SU10已完成10-10配对")

    # ========================================================================
    # Layer1: 完成1-hop分配方法（g-i）
    # ========================================================================
    
    def _complete_aromatic_hop1(self, nodes: List[_NodeV3]):
        """g) 完成芳香结构的互为1-hop（5/6/7/8/9需要完成剩余连接）"""
        pass # pass # print(f"\n[g] 完成芳香结构1-hop")
        
        aromatic_types = [5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30]
        priority_list = [13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30]
        
        # 处理5/6/7/8/9（这些与杂原子互为1-hop后需要补全）
        for su_type in [8, 7, 6, 5, 9]:
            incomplete_nodes = self._get_incomplete_hop1_nodes(nodes, [su_type])
            
            for node in incomplete_nodes:
                needed = node.remaining_hop1_slots()
                
                for _ in range(needed):
                    # 优先选择空1-hop且有剩余槽位的芳香节点
                    empty_candidates = self._get_empty_hop1_nodes(nodes, aromatic_types)
                    empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                    
                    if empty_candidates:
                        target = self._choose_weighted_candidate(node, empty_candidates, priority_list, nodes=nodes)
                        if target is None:
                            break
                    else:
                        # 没有空节点，选择有剩余槽位的不完整节点
                        incomplete_candidates = self._get_available_nodes(nodes, aromatic_types)
                        incomplete_candidates = [n for n in incomplete_candidates if n.global_id != node.global_id]
                        
                        if incomplete_candidates:
                            target = self._choose_weighted_candidate(node, incomplete_candidates, priority_list, nodes=nodes)
                            if target is None:
                                break
                        else:
                            break
                    
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
    
    def _complete_aliphatic_hetero_hop1(self, nodes: List[_NodeV3]):
        """g2) 完成19/20/21号脂肪杂原子碳的1-hop（与X/N/S/O互为1-hop后补全剩余）
        """
        pass # pass # print(f"\n[g2] 完成19/20/21号1-hop")
        
        # SU19/20/21 端口1: {23,11,22,24,25,19,20,21,2,3,1,0,14,15,17}
        # 不含 SU16 和 SU18
        priority_list = [23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17]
        
        # 处理21/20/19号（连接度=2）
        for su_type in [21, 20, 19]:
            incomplete_nodes = self._get_incomplete_hop1_nodes(nodes, [su_type])
            
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
                            pass # pass # print(f"  警告：{su_type}号(id={node.global_id})找不到可用的候选")
                            break
                    
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
    
    def _complete_unsaturated_saturated_end(self, nodes: List[_NodeV3]):
        """h) 完成不饱和结构的饱和端和4号腈"""
        pass # pass # print(f"\n[h] 完成不饱和结构的饱和端")
        
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
                        pass # pass # print(f"  警告：4号(id={node.global_id})找不到可用的候选")
                        continue

                if target is None:
                    pass # pass # print(f"  警告：4号(id={node.global_id})找不到可用的候选")
                    continue
                
                self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
    
    def _complete_remaining_aliphatic_aromatic(self, nodes: List[_NodeV3]):
        """i) 完成剩余脂肪碳(22-25)和芳香碳(10-13)"""
        pass # pass # print(f"\n[i] 完成剩余脂肪碳和芳香碳")

        # 先定向补全11号的脂肪端口，避免22/23/24/25被先耗尽后11长期停在芳香半成品状态
        su11_aliphatic_priority = [23, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17]
        for node in [n for n in nodes if n.su_type == 11 and not n.is_hop1_complete()]:
            current = self._current_neighbor_types(nodes, node)
            has_aliphatic_like = any(int(x) in set(su11_aliphatic_priority) for x in current)
            if has_aliphatic_like:
                continue
            empty_candidates = self._get_empty_hop1_nodes(nodes, su11_aliphatic_priority)
            empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
            if empty_candidates:
                target = self._choose_weighted_candidate(node, empty_candidates, su11_aliphatic_priority, nodes=nodes)
            else:
                available_candidates = self._get_available_nodes(nodes, su11_aliphatic_priority)
                available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                target = self._choose_weighted_candidate(node, available_candidates, su11_aliphatic_priority, nodes=nodes) if available_candidates else None
            if target is not None:
                self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
        
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
        
        for node in su23_nodes:
            needed = node.remaining_hop1_slots()
            for _ in range(needed):
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
                        break

                if target is None:
                    break
                
                self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
        
        # 完成24号和25号（连接度=3/4）
        for su_type in [24, 25]:
            incomplete_nodes = [n for n in nodes if n.su_type == su_type and not n.is_hop1_complete()]
            
            for node in incomplete_nodes:
                needed = node.remaining_hop1_slots()
                
                for _ in range(needed):
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
                            break

                    if target is None:
                        break
                    
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
                    pass # pass # print(f"  {su_type}(id={node.global_id}) -> {target.su_type}(id={target.global_id})")

        # 完成芳香结构（10-13互相补全，候选池包含全部芳香类型以满足SU10等端口规则）
        aromatic_types = [5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30]
        aromatic_priority = [13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30]
        
        for su_type in [11, 10, 12, 13]:
            incomplete_nodes = [n for n in nodes if n.su_type == su_type and not n.is_hop1_complete()]
            
            for node in incomplete_nodes:
                needed = node.remaining_hop1_slots()
                
                for _ in range(needed):
                    empty_candidates = self._get_empty_hop1_nodes(nodes, aromatic_types)
                    empty_candidates = [n for n in empty_candidates if n.global_id != node.global_id and n.remaining_hop1_slots() > 0]
                    
                    if empty_candidates:
                        target = self._choose_weighted_candidate(node, empty_candidates, aromatic_priority, nodes=nodes)
                    else:
                        available_candidates = self._get_available_nodes(nodes, aromatic_types)
                        available_candidates = [n for n in available_candidates if n.global_id != node.global_id]
                        if available_candidates:
                            target = self._choose_weighted_candidate(node, available_candidates, aromatic_priority, nodes=nodes)
                        else:
                            break

                    if target is None:
                        break
                    
                    self._add_bidirectional_hop1(nodes, node.global_id, target.global_id)
 
    def _print_hop1_summary(self, nodes: List[_NodeV3]):
        """打印1-hop分配总结"""
        pass # pass # print(f"\n总节点数: {len(nodes)}")
        
        # 统计每种SU的1-hop完成情况
        su_stats = {}
        for su_type in range(NUM_SU_TYPES):
            su_nodes = self._get_nodes_by_su_type(nodes, su_type)
            if su_nodes:
                complete_count = sum(1 for n in su_nodes if n.is_hop1_complete())
                incomplete_count = sum(1 for n in su_nodes if not n.is_hop1_empty() and not n.is_hop1_complete())
                empty_count = sum(1 for n in su_nodes if n.is_hop1_empty())
                
                su_stats[su_type] = {
                    'total': len(su_nodes),
                    'complete': complete_count,
                    'incomplete': incomplete_count,
                    'empty': empty_count,
                    'max_degree': su_nodes[0].get_max_degree()
                }
        
        # 打印统计信息
        incomplete_sus = [su for su, stats in su_stats.items() if stats['incomplete'] > 0 or stats['empty'] > 0]
        if incomplete_sus:
            pass # pass # print(f"\n⚠️ 存在不完整的1-hop分配:")
            for su_type in sorted(incomplete_sus):
                stats = su_stats[su_type]
                pass # pass # print(f"  SU {su_type:2d}: 完整={stats['complete']:3d}, 不完整={stats['incomplete']:3d}, 空={stats['empty']:3d} (度数≤{stats['max_degree']})")
        else:
            pass # pass # print(f"\n✓ 所有节点的1-hop分配完成")
        
        # 打印已完成的SU类型
        complete_sus = [su for su, stats in su_stats.items() if stats['incomplete'] == 0 and stats['empty'] == 0]
        if complete_sus:
            pass # pass # print(f"\n✓ 已完成1-hop分配的SU类型: {sorted(complete_sus)}")

    def layer1_assign(self, H_init: torch.Tensor, S_target: torch.Tensor,
                      E_target: torch.Tensor,
                      eval_nmr: bool = True,
                      eval_output_dir: str = 'inverse_result',
                      eval_lib_path: Optional[str] = None,
                      eval_hwhm: float = 1.0,
                      eval_allow_approx: bool = True,
                      enable_hop1_adjust: bool = False,
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
            enable_hop1_adjust: 是否启用1-hop调整
            hop1_adjust_iterations: 1-hop调整最大迭代次数
            hop1_neg_threshold: 负峰阈值
            hop1_pos_threshold: 正峰阈值
        """
        device = self.device
        H_init = H_init.to(device)
        S_target = S_target.to(device)
        E_target = E_target.to(device)
    
        print("\n" + "="*60)
        print(f"Layer1: 1-hop分配 ({len(self._initialize_node_pool(H_init))}个节点)")
        print("="*60)
    
        # 初始化全局节点池
        nodes = self._initialize_node_pool(H_init)
    
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
    
        # g2) 完成19/20/21号脂肪杂原子碳的1-hop
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
        if not is_valid:
            print(f"  ⚠ 图结构存在{len(errors)}个不一致")
        
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
            r2 = round_metrics.get('r2', 0.0)
            print(f"  R²={r2:.4f}")
    
        # 可选的1-hop调整（基于差谱分析，Layer1.5）
        if enable_hop1_adjust:
            try:
                nodes, adjust_summary = self.adjust_hop1_based_on_spectrum(
                    nodes=nodes,
                    S_target=S_target,
                    E_target=E_target,
                    lib_path=eval_lib_path,
                    output_dir=eval_output_dir,
                    hwhm=eval_hwhm,
                    neg_threshold=hop1_neg_threshold,
                    pos_threshold=hop1_pos_threshold,
                    max_iterations=hop1_adjust_iterations,
                )
                total_adj = adjust_summary.get('total_adjustments', 0)
                if total_adj > 0:
                    print(f"  Hop1调整: {total_adj}次")
            except Exception as e:
                print(f"  [Hop1调整失败] {e}")

        print("Layer1 完成\n")
        
        return nodes

    def _compute_layer1_difference_spectrum(self,
                                           nodes: List[_NodeV3],
                                           S_target: torch.Tensor,
                                           lib_path: Optional[str],
                                           hwhm: float,
                                           allow_approx: bool) -> Dict[str, object]:
        device = self.device
        S_target = S_target.to(device).flatten()
        if int(S_target.numel()) != int(PPM_AXIS.numel()):
            n = int(min(S_target.numel(), PPM_AXIS.numel()))
            S_target = S_target[:n]
            ppm_axis = PPM_AXIS.to(device)[:n]
        else:
            ppm_axis = PPM_AXIS.to(device)

        try:
            lib_index = self._get_layer1_library_index(lib_path)
        except Exception as e:
            return {
                'ppm': ppm_axis.detach().cpu().numpy(),
                'diff': np.zeros(int(ppm_axis.numel()), dtype=np.float64),
                'r2': 0.0,
                'alpha': 0.0,
                'n_peaks': 0,
                'error': str(e),
            }
        grouped_assignments = self._assign_grouped_mu_pi_by_hop1(nodes, lib_index, allow_approx)
        mus = []
        pis = []
        for n in nodes:
            center_su = int(n.su_type)
            if float(self.E_SU[center_su, 0].detach().cpu().item()) <= 0:
                continue
            assigned = grouped_assignments.get(int(n.global_id), None)
            if assigned is None:
                continue
            mu = assigned.get('mu')
            pi = assigned.get('pi')
            if mu is None or pi is None:
                continue
            mus.append(float(mu))
            pis.append(float(max(0.0, pi)))

        if not mus:
            return {
                'ppm': ppm_axis.detach().cpu().numpy(),
                'diff': np.zeros(int(ppm_axis.numel()), dtype=np.float64),
                'r2': 0.0,
                'alpha': 0.0,
                'n_peaks': 0,
            }

        mu_t = torch.tensor(mus, dtype=torch.float, device=device)
        pi_t = torch.tensor(pis, dtype=torch.float, device=device)
        S_recon = lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=float(hwhm))
        eval_info = evaluate_spectrum_reconstruction(
            S_target,
            S_recon,
            ppm_axis=ppm_axis,
            fit_scale=True,
            nonnegative_alpha=True,
        )
        S_fit = eval_info['S_fit']
        diff = (eval_info['S_target'] - S_fit).detach().cpu().numpy()
        return {
            'ppm': ppm_axis.detach().cpu().numpy(),
            'diff': diff,
            'r2': float(eval_info.get('r2', 0.0)),
            'alpha': float(eval_info.get('alpha', 1.0)),
            'n_peaks': int(len(mus)),
        }

    def _hop1_to_multiset(self, hop1_su: Dict[int, int]) -> Tuple[int, ...]:
        ms = []
        for su, v in hop1_su.items():
            ms.extend([int(su)] * int(v))
        ms.sort()
        return tuple(ms)

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray, q: float) -> float:
        """计算加权分位数（与z_library.py保持一致）"""
        idx = np.argsort(values)
        v = values[idx]
        w = weights[idx]
        cw = np.cumsum(w)
        cutoff = float(q) * float(cw[-1])
        pos = int(np.searchsorted(cw, cutoff, side='left'))
        pos = min(max(pos, 0), len(v) - 1)
        return float(v[pos])

    def _get_layer1_library_index(self, lib_path: Optional[str]):
        if lib_path is None:
            lib_path = str(Path(__file__).resolve().parents[1] / 'z_library' / 'subgraph_library.pt')

        cache = getattr(self, '_layer1_lib_index_cache', None)
        if isinstance(cache, dict) and cache.get('path') == lib_path:
            return cache

        lib = torch.load(lib_path, map_location='cpu',weights_only=False)
        templates = lib.get('templates', {}) if isinstance(lib, dict) else {}

        agg = {}
        center_to_hop1 = defaultdict(list)

        for k, tpl in templates.items():
            if not isinstance(k, tuple) or len(k) != 3:
                continue
            c, h1, _h2 = k
            try:
                c = int(c)
                h1_ms = tuple(int(x) for x in tuple(h1))
            except Exception:
                continue

            mu_s = tpl.get('samples', {}).get('mu', None)
            pi_s = tpl.get('samples', {}).get('pi', None)
            if mu_s is None or pi_s is None:
                continue
            if not torch.is_tensor(mu_s) or not torch.is_tensor(pi_s):
                continue
            if int(mu_s.numel()) <= 0 or int(pi_s.numel()) <= 0:
                continue

            sc = int(tpl.get('sample_count', 0))
            cmu = float(tpl.get('center_mu', 0.0))
            if sc <= 0:
                sc = int(mu_s.numel()) if int(mu_s.numel()) > 0 else 1
            if cmu == 0.0:
                continue

            key = (c, h1_ms)
            if key not in agg:
                agg[key] = {
                    'mu_values': [],
                    'pi_values': [],
                    'weights': [],
                    'mu_min': float(tpl.get('mu_min', cmu)),
                    'mu_max': float(tpl.get('mu_max', cmu)),
                    'n_templates': 0,
                }
                center_to_hop1[c].append(h1_ms)

            a = agg[key]
            a['mu_values'].append(float(cmu))
            cpi = None
            try:
                cpi = float(tpl.get('center_pi', 0.0))
            except Exception:
                cpi = None
            if cpi is None or cpi <= 0.0:
                try:
                    cpi = float(torch.median(pi_s.detach().float()).item())
                except Exception:
                    cpi = 1.0
            a['pi_values'].append(float(max(1e-6, cpi)))
            a['weights'].append(float(sc))
            a['mu_min'] = float(min(a['mu_min'], float(tpl.get('mu_min', cmu))))
            a['mu_max'] = float(max(a['mu_max'], float(tpl.get('mu_max', cmu))))
            a['n_templates'] += 1

        new_cache = {
            'path': lib_path,
            'agg': agg,
            'center_to_hop1': dict(center_to_hop1),
            'n_templates_total': len(templates),
        }
        setattr(self, '_layer1_lib_index_cache', new_cache)
        return new_cache

    def _multiset_l1_distance(self, a_ms: Tuple[int, ...], b_ms: Tuple[int, ...]) -> int:
        ca = Counter(int(x) for x in tuple(a_ms))
        cb = Counter(int(x) for x in tuple(b_ms))
        keys = set(ca.keys()) | set(cb.keys())
        return int(sum(abs(int(ca.get(k, 0)) - int(cb.get(k, 0))) for k in keys))

    def _get_template_library(self, lib_path: Optional[str]) -> Dict[str, object]:
        if lib_path is None:
            lib_path = str(Path(__file__).resolve().parents[1] / 'z_library' / 'subgraph_library.pt')

        cache = getattr(self, '_template_lib_cache', None)
        if isinstance(cache, dict) and cache.get('path') == lib_path:
            return cache

        lib = torch.load(lib_path, map_location='cpu')
        templates = lib.get('templates', {}) if isinstance(lib, dict) else {}
        center_index = lib.get('center_index', {}) if isinstance(lib, dict) else {}

        new_cache = {
            'path': lib_path,
            'templates': templates,
            'center_index': center_index,
        }
        setattr(self, '_template_lib_cache', new_cache)
        return new_cache

    def _multiset_from_counter(self, cnt: Counter) -> Tuple[int, ...]:
        vec = []
        for k, v in cnt.items():
            vec.extend([int(k)] * int(v))
        vec.sort()
        return tuple(vec)

    def _select_template_key_layer2(self,
                                   center_su: int,
                                   hop1_ms: Tuple[int, ...],
                                   hop2_ms: Tuple[int, ...],
                                   lib: Dict[str, object]) -> Tuple[Optional[Tuple], str]:
        templates = lib.get('templates', {}) if isinstance(lib, dict) else {}
        center_index = lib.get('center_index', {}) if isinstance(lib, dict) else {}

        key_exact = (int(center_su), tuple(hop1_ms), tuple(hop2_ms))
        if key_exact in templates:
            return key_exact, 'exact'

        cand_keys = center_index.get(int(center_su), [])
        pool = []
        for k in cand_keys:
            kt = tuple(k) if not isinstance(k, tuple) else k
            if len(kt) != 3:
                continue
            try:
                c, h1, h2 = kt
                c = int(c)
                h1_t = tuple(h1) if not isinstance(h1, tuple) else h1
                h2_t = tuple(h2) if not isinstance(h2, tuple) else h2
            except Exception:
                continue
            if int(c) != int(center_su):
                continue
            pool.append((kt, h1_t, h2_t))

        if not pool:
            return None, 'missing'

        same_h1 = [it for it in pool if tuple(it[1]) == tuple(hop1_ms)]
        mode = 'approx_h2' if same_h1 else 'approx_h1h2'
        use_pool = same_h1 if same_h1 else pool

        best = None
        for kt, h1_t, h2_t in use_pool:
            d_h2 = self._multiset_l1_distance(tuple(hop2_ms), tuple(h2_t))
            d_h1 = 0 if tuple(h1_t) == tuple(hop1_ms) else self._multiset_l1_distance(tuple(hop1_ms), tuple(h1_t))
            d = int(d_h1 + d_h2)
            tpl = templates.get(kt, {}) if isinstance(templates, dict) else {}
            sc = int(tpl.get('sample_count', 0)) if isinstance(tpl, dict) else 0
            cand = (d, -sc, kt)
            if best is None or cand < best:
                best = cand
        return best[2], mode

    def _global_embed_from_elements(self, E_target: torch.Tensor) -> torch.Tensor:
        device = self.device
        e = E_target.to(device).view(1, -1)
        if float(e.max().item()) > 1.1:
            s = e.sum(dim=1, keepdim=True).clamp(min=1.0)
            e = e / s
        try:
            g = self.vae.global_mlp(e)
            g = 0.02 * g
            return g
        except Exception as e:
            import logging
            logging.warning(f"Failed to compute global embedding: {e}")
            return torch.zeros((1, 2), dtype=torch.float, device=device)

    def _decode_mu_pi_from_z(self,
                             center_su: int,
                             z_vec: torch.Tensor,
                             g_embed: torch.Tensor) -> Optional[Tuple[float, float]]:
        device = self.device
        try:
            su_feat = F.one_hot(
                torch.tensor([int(center_su)], dtype=torch.long, device=device),
                num_classes=NUM_SU_TYPES,
            ).float()
            z = z_vec.to(device).view(1, -1)
            pred = self.vae.decoder(su_feat, z, g_embed)
            mu_pred = float(pred[:, 0].detach().item())
            pi_pred = float(F.softplus(pred[:, 1]).detach().item())
            return mu_pred, pi_pred
        except Exception as e:
            import logging
            logging.warning(f"Failed to decode mu/pi from z: {e}")
            return None

    def _lookup_mu_pi_by_hop1(self,
                              center_su: int,
                              hop1_ms: Tuple[int, ...],
                              lib_index: dict,
                              allow_approx: bool) -> Tuple[Optional[float], Optional[float], dict]:
        agg = lib_index.get('agg', {})
        key = (int(center_su), tuple(hop1_ms))
        info = agg.get(key)
        chosen_hop1 = tuple(hop1_ms)
        approx_used = False

        if info is None and allow_approx:
            hop1_keys = lib_index.get('center_to_hop1', {}).get(int(center_su), [])
            if hop1_keys:
                best = min(
                    hop1_keys,
                    key=lambda ms: (self._multiset_l1_distance(ms, hop1_ms), abs(len(ms) - len(hop1_ms)))
                )
                info = agg.get((int(center_su), tuple(best)))
                chosen_hop1 = tuple(best)
                approx_used = True

        if info is None:
            return None, None, {
                'matched': False,
                'approx_used': False,
                'chosen_hop1_ms': chosen_hop1,
                'n_templates': 0,
                'w_sum': 0.0,
                'mu_min': 0.0,
                'mu_max': 0.0,
            }

        mu_values = info.get('mu_values', [])
        pi_values = info.get('pi_values', [])
        weights = info.get('weights', [])
        if not mu_values or not pi_values or not weights:
            return None, None, {
                'matched': False,
                'approx_used': approx_used,
                'chosen_hop1_ms': chosen_hop1,
                'n_templates': int(info.get('n_templates', 0)),
                'w_sum': 0.0,
                'mu_min': float(info.get('mu_min', 0.0)),
                'mu_max': float(info.get('mu_max', 0.0)),
            }

        v = np.asarray(mu_values, dtype=np.float64)
        p = np.asarray(pi_values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w_sum = float(w.sum())
        mu = self._weighted_quantile(v, w, 0.5)
        pi = self._weighted_quantile(p, w, 0.5)
        return mu, pi, {
            'matched': True,
            'approx_used': approx_used,
            'chosen_hop1_ms': chosen_hop1,
            'n_templates': int(info.get('n_templates', 0)),
            'w_sum': w_sum,
            'mu_min': float(info.get('mu_min', 0.0)),
            'mu_max': float(info.get('mu_max', 0.0)),
        }

    def _assign_grouped_mu_pi_by_hop1(self,
                                      nodes: List[_NodeV3],
                                      lib_index: dict,
                                      allow_approx: bool) -> Dict[int, Dict[str, object]]:
        """
        对具有相同 (center_su, hop1_ms) 的节点做组内展开，避免所有节点都落到同一中位 ppm。
        """
        grouped = defaultdict(list)
        for node in nodes:
            try:
                center_su = int(node.su_type)
                is_carbon = float(self.E_SU[center_su, 0].detach().cpu().item()) > 0
            except Exception:
                is_carbon = False
            if not bool(is_carbon):
                continue
            hop1_ms = self._hop1_to_multiset(node.hop1_su)
            grouped[(int(node.su_type), tuple(hop1_ms))].append(node)

        assignments: Dict[int, Dict[str, object]] = {}
        agg = lib_index.get('agg', {})
        center_to_hop1 = lib_index.get('center_to_hop1', {})

        for (center_su, hop1_ms), group_nodes in grouped.items():
            info = agg.get((int(center_su), tuple(hop1_ms)))
            chosen_hop1 = tuple(hop1_ms)
            approx_used = False

            if info is None and allow_approx:
                hop1_keys = center_to_hop1.get(int(center_su), [])
                if hop1_keys:
                    best = min(
                        hop1_keys,
                        key=lambda ms: (self._multiset_l1_distance(ms, hop1_ms), abs(len(ms) - len(hop1_ms)))
                    )
                    info = agg.get((int(center_su), tuple(best)))
                    chosen_hop1 = tuple(best)
                    approx_used = True

            if info is None:
                for node in group_nodes:
                    assignments[int(node.global_id)] = {
                        'mu': None,
                        'pi': None,
                        'meta': {
                            'matched': False,
                            'approx_used': False,
                            'chosen_hop1_ms': chosen_hop1,
                            'n_templates': 0,
                            'w_sum': 0.0,
                            'mu_min': 0.0,
                            'mu_max': 0.0,
                        },
                    }
                continue

            mu_values = np.asarray(info.get('mu_values', []), dtype=np.float64)
            pi_values = np.asarray(info.get('pi_values', []), dtype=np.float64)
            weights = np.asarray(info.get('weights', []), dtype=np.float64)

            if mu_values.size == 0 or pi_values.size == 0 or weights.size == 0:
                for node in group_nodes:
                    assignments[int(node.global_id)] = {
                        'mu': None,
                        'pi': None,
                        'meta': {
                            'matched': False,
                            'approx_used': approx_used,
                            'chosen_hop1_ms': chosen_hop1,
                            'n_templates': int(info.get('n_templates', 0)),
                            'w_sum': 0.0,
                            'mu_min': float(info.get('mu_min', 0.0)),
                            'mu_max': float(info.get('mu_max', 0.0)),
                        },
                    }
                continue

            order = np.argsort(mu_values)
            mu_sorted = mu_values[order]
            pi_sorted = pi_values[order]
            w_sorted = np.maximum(weights[order], 1e-8)
            cdf = np.cumsum(w_sorted) / float(np.sum(w_sorted))

            ordered_nodes = sorted(group_nodes, key=lambda n: int(getattr(n, 'global_id', 0)))
            n_nodes = len(ordered_nodes)
            for idx, node in enumerate(ordered_nodes):
                q = (idx + 0.5) / max(1.0, float(n_nodes))
                pick = int(np.searchsorted(cdf, q, side='left'))
                pick = min(max(pick, 0), len(mu_sorted) - 1)
                assignments[int(node.global_id)] = {
                    'mu': float(mu_sorted[pick]),
                    'pi': float(max(1e-6, pi_sorted[pick])),
                    'meta': {
                        'matched': True,
                        'approx_used': approx_used,
                        'chosen_hop1_ms': chosen_hop1,
                        'n_templates': int(info.get('n_templates', 0)),
                        'w_sum': float(np.sum(w_sorted)),
                        'mu_min': float(info.get('mu_min', 0.0)),
                        'mu_max': float(info.get('mu_max', 0.0)),
                    },
                }

        return assignments

    def evaluate_layer1_nmr_with_library(self,
                                         nodes: List[_NodeV3],
                                         S_target: torch.Tensor,
                                         lib_path: Optional[str] = None,
                                         output_dir: str = 'inverse_result',
                                         hwhm: float = 1.0,
                                         allow_approx: bool = True) -> Dict[str, float]:
        device = self.device
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        S_target = S_target.to(device).flatten()
        if int(S_target.numel()) != int(PPM_AXIS.numel()):
            n = int(min(S_target.numel(), PPM_AXIS.numel()))
            S_target = S_target[:n]
            ppm_axis = PPM_AXIS.to(device)[:n]
        else:
            ppm_axis = PPM_AXIS.to(device)

        lib_index = self._get_layer1_library_index(lib_path)
        grouped_assignments = self._assign_grouped_mu_pi_by_hop1(nodes, lib_index, allow_approx)
        su_names = [name for name, _ in SU_DEFS]

        mus = []
        pis = []
        rows = []
        matched_cnt = 0
        carbon_cnt = 0

        for n in nodes:
            center_su = int(n.su_type)
            is_carbon = float(self.E_SU[center_su, 0].detach().cpu().item()) > 0

            hop1_ms = self._hop1_to_multiset(n.hop1_su)
            if is_carbon:
                carbon_cnt += 1
                assigned = grouped_assignments.get(int(n.global_id), None)
                if assigned is None:
                    mu, pi, meta = None, None, {
                        'matched': False,
                        'approx_used': False,
                        'chosen_hop1_ms': tuple(hop1_ms),
                        'n_templates': 0,
                        'w_sum': 0.0,
                        'mu_min': 0.0,
                        'mu_max': 0.0,
                    }
                else:
                    mu = assigned.get('mu')
                    pi = assigned.get('pi')
                    meta = assigned.get('meta', {})
                if bool(meta.get('matched')):
                    matched_cnt += 1

                if mu is not None and pi is not None:
                    mus.append(float(mu))
                    pis.append(float(max(0.0, pi)))

                rows.append({
                    'global_id': int(n.global_id),
                    'center_su_idx': center_su,
                    'center_su': su_names[center_su] if 0 <= center_su < len(su_names) else str(center_su),
                    'hop1_ms': '[' + ' '.join(str(x) for x in hop1_ms) + ']',
                    'matched': bool(meta.get('matched')),
                    'approx_used': bool(meta.get('approx_used')),
                    'chosen_hop1_ms': '[' + ' '.join(str(x) for x in meta.get('chosen_hop1_ms', ())) + ']',
                    'n_templates': int(meta.get('n_templates', 0)),
                    'sample_weight_sum': float(meta.get('w_sum', 0.0)),
                    'mu': float(mu) if mu is not None else np.nan,
                    'pi': float(pi) if pi is not None else np.nan,
                    'mu_min': float(meta.get('mu_min', 0.0)),
                    'mu_max': float(meta.get('mu_max', 0.0)),
                })
            else:
                rows.append({
                    'global_id': int(n.global_id),
                    'center_su_idx': center_su,
                    'center_su': su_names[center_su] if 0 <= center_su < len(su_names) else str(center_su),
                    'hop1_ms': '[' + ' '.join(str(x) for x in hop1_ms) + ']',
                    'matched': False,
                    'approx_used': False,
                    'chosen_hop1_ms': '[' + ' '.join(str(x) for x in hop1_ms) + ']',
                    'n_templates': 0,
                    'sample_weight_sum': 0.0,
                    'mu': 0.0,
                    'pi': 0.0,
                    'mu_min': 0.0,
                    'mu_max': 0.0,
                })

        pd.DataFrame(rows).to_csv(str(Path(output_dir) / 'layer1_library_node_peaks.csv'), index=False)

        matched_ratio = float(matched_cnt) / max(1.0, float(carbon_cnt))
        print(f"[Layer1-NMR-Eval] carbon_nodes={carbon_cnt}, matched={matched_cnt}, matched_ratio={matched_ratio:.3f}")

        if not mus:
            print("[Layer1-NMR-Eval] 未找到可用模板峰，跳过谱图重构")
            return {
                'r2': 0.0,
                'r2_carbonyl': 0.0,
                'r2_aromatic': 0.0,
                'r2_aliphatic': 0.0,
                'matched_ratio': matched_ratio,
            }

        mu_t = torch.tensor(mus, dtype=torch.float, device=device)
        pi_t = torch.tensor(pis, dtype=torch.float, device=device)
        S_recon = lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=float(hwhm))

        eval_info = evaluate_spectrum_reconstruction(
            S_target,
            S_recon,
            ppm_axis=ppm_axis,
            fit_scale=True,
            nonnegative_alpha=True,
        )
        S_fit = eval_info['S_fit']
        r2 = float(eval_info.get('r2', 0.0))
        r2_carb = float(eval_info.get('r2_carbonyl', 0.0))
        r2_aro = float(eval_info.get('r2_aromatic', 0.0))
        r2_ali = float(eval_info.get('r2_aliphatic', 0.0))
        alpha = float(eval_info.get('alpha', 1.0))

        df_spec = pd.DataFrame({
            'ppm': ppm_axis.detach().cpu().numpy(),
            'target': S_target.detach().cpu().numpy(),
            'reconstructed_raw': S_recon.detach().cpu().numpy(),
            'reconstructed': S_fit.detach().cpu().numpy(),
            'difference': (S_target - S_fit).detach().cpu().numpy(),
        })
        df_spec.to_csv(str(Path(output_dir) / 'layer1_library_spectrum_comparison.csv'), index=False)

        try:
            visualize_spectrum_comparison(S_target.detach().cpu(), S_fit.detach().cpu(), ppm_axis.detach().cpu(), 'Layer1-Library', save_dir=output_dir)
        except Exception as e:
            print(f"[Layer1-NMR-Eval] 绘图失败: {e}")

        print(f"[Layer1-NMR-Eval] R2={r2:.4f} (carbonyl={r2_carb:.4f}, aromatic={r2_aro:.4f}, aliphatic={r2_ali:.4f}), alpha={alpha:.4f}")
        return {
            'r2': float(r2),
            'r2_carbonyl': float(r2_carb),
            'r2_aromatic': float(r2_aro),
            'r2_aliphatic': float(r2_ali),
            'matched_ratio': matched_ratio,
        }
    
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
        device = self.device
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if adjustment_groups is None:
            adjustment_groups = ['aromatic', 'carbonyl', 'unsaturated', 'aliphatic']
        
        # Layer1.5调整静默进行
        
        # 初始化调整器
        adjuster = Hop1Adjuster(
             port_combinations=HOP1_PORT_COMBINATIONS,
             validate_connection_fn=validate_connection,
             external_requirement_fn=check_external_connection_requirement,
         )
        
        S_target = S_target.to(device).flatten()
        E_target = E_target.to(device).flatten()
        ppm_axis = PPM_AXIS.to(device)
        if int(S_target.numel()) != int(ppm_axis.numel()):
            n = int(min(S_target.numel(), ppm_axis.numel()))
            S_target = S_target[:n]
            ppm_axis = ppm_axis[:n]
        
        lib_index = self._get_layer1_library_index(lib_path)
        su_names = [name for name, _ in SU_DEFS]
        
        total_adjustments = 0
        iteration_summaries = []
        best_r2 = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n--- 调整迭代 {iteration + 1}/{max_iterations} ---")
            
            # 1. 重新计算当前谱图
            mus, pis, rows = [], [], []
            for n in nodes:
                center_su = int(n.su_type)
                if float(self.E_SU[center_su, 0].detach().cpu().item()) <= 0:
                    continue
                
                hop1_ms = self._hop1_to_multiset(n.hop1_su)
                mu, pi, meta = self._lookup_mu_pi_by_hop1(center_su, hop1_ms, lib_index, allow_approx=True)
                
                if mu is not None and pi is not None:
                    mus.append(float(mu))
                    pis.append(float(max(0.0, pi)))
                
                rows.append({
                    'global_id': int(n.global_id),
                    'center_su_idx': center_su,
                    'center_su': su_names[center_su] if 0 <= center_su < len(su_names) else str(center_su),
                    'hop1_ms': '[' + ' '.join(str(x) for x in hop1_ms) + ']',
                    'matched': bool(meta.get('matched')),
                    'approx_used': bool(meta.get('approx_used')),
                    'chosen_hop1_ms': '[' + ' '.join(str(x) for x in meta.get('chosen_hop1_ms', ())) + ']',
                    'n_templates': int(meta.get('n_templates', 0)),
                    'sample_weight_sum': float(meta.get('w_sum', 0.0)),
                    'mu': float(mu) if mu is not None else np.nan,
                    'pi': float(pi) if pi is not None else np.nan,
                    'mu_min': float(meta.get('mu_min', 0.0)),
                    'mu_max': float(meta.get('mu_max', 0.0)),
                })
            
            if not mus:
                print("  无有效峰，跳过调整")
                break
            
            # 重建谱图
            mu_t = torch.tensor(mus, dtype=torch.float, device=device)
            pi_t = torch.tensor(pis, dtype=torch.float, device=device)
            S_recon = lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=float(hwhm))
            
            # 缩放
            denom = (S_recon * S_recon).sum().clamp(min=1e-8)
            alpha = (S_target * S_recon).sum() / denom
            S_fit = alpha * S_recon
            
            # 计算R²和差谱
            r2 = compute_r2_score(S_target, S_fit)
            diff = (S_target - S_fit).detach().cpu().numpy()
            ppm_np = ppm_axis.detach().cpu().numpy()
            
            print(f"  当前R² = {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
            
            # 保存当前node_peaks用于调整
            node_peaks_df = pd.DataFrame(rows)
            
            # 2. 执行调整
            _, adjust_summary = adjuster.adjust(
                 nodes=nodes,
                 node_peaks=node_peaks_df,
                 diff_spectrum=diff,
                 ppm_axis=ppm_np,
                 E_target=E_target,
                 neg_threshold=neg_threshold,
                 pos_threshold=pos_threshold,
                 max_adjustments_per_group=5,
                 adjustment_groups=adjustment_groups
             )
            
            iter_adjustments = adjust_summary.get('adjustments', 0)
            total_adjustments += iter_adjustments
            iteration_summaries.append({
                'iteration': iteration + 1,
                'r2_before': float(r2),
                'adjustments': iter_adjustments,
            })
            
            # 检查是否有调整
            if iter_adjustments == 0:
                print("  无可行调整，停止迭代")
                break
        
        # 最终评估
        print("\n--- 调整完成，最终评估 ---")
        final_metrics = self.evaluate_layer1_nmr_with_library(
            nodes=nodes, S_target=S_target, lib_path=lib_path,
            output_dir=output_dir, hwhm=hwhm, allow_approx=True
        )
        
        summary = {
            'total_adjustments': total_adjustments,
            'iterations': len(iteration_summaries),
            'iteration_details': iteration_summaries,
            'best_r2': best_r2,
            'final_r2': final_metrics.get('r2', 0.0),
            'adjuster_stats': adjuster.stats,
        }
        
        pass # pass # print(f"总调整次数: {total_adjustments}")
        pass # pass # print(f"最终R²: {final_metrics.get('r2', 0.0):.4f}")
        
        return nodes, summary
