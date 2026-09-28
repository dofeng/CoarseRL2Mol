from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Set, Tuple

import torch

from ...shared.coarse_graph import NUM_SU_TYPES
from ...shared.inverse_common import (
    _NodeV3,
    format_port_patterns_debug,
    has_forbidden_fixed_connector_special_combo,
)
from .hop1_adjuster import can_match_ports_exact, can_match_ports_partial
from .terminal_paths import terminal_path_reason


class Layer1GraphValidator:
    """Validate only Layer1 topology and exact 1-hop contracts."""

    def __init__(self, assigner: Any) -> None:
        self.assigner = assigner

    def validate(
        self,
        nodes: List[_NodeV3],
        H: torch.Tensor,
        E_target: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[bool, List[str]]:
        a = self.assigner
        errors: List[str] = []
        terminal_reason = terminal_path_reason(nodes)
        if terminal_reason:
            errors.append(terminal_reason)

        expected_total = int(H.sum().item())
        if len(nodes) != expected_total:
            errors.append(f"节点数量不匹配: 实际{len(nodes)} vs 预期{expected_total}")

        actual_histogram = a._histogram_from_nodes(nodes)
        for su_type in range(NUM_SU_TYPES):
            expected = int(H[su_type].item())
            actual = int(actual_histogram[su_type].item())
            if expected != actual:
                errors.append(f"SU{su_type}数量不匹配: 实际{actual} vs 预期{expected}")

        for node in nodes:
            node_id = int(node.global_id)
            ids = [int(x) for x in list(node.hop1_ids or [])]
            if len(ids) != len(set(ids)):
                errors.append(f"节点{node_id}(SU{node.su_type}): 存在重复1-hop节点ID")
            for neighbor_id in ids:
                if neighbor_id < 0 or neighbor_id >= len(nodes):
                    errors.append(f"节点{node_id}: hop1_ids包含越界ID {neighbor_id}")
                    continue
                neighbor = nodes[neighbor_id]
                if node_id not in neighbor.hop1_ids:
                    errors.append(f"1-hop不对称: {node_id}->{neighbor_id} 但反向缺失")

            actual_counter = Counter(
                int(nodes[neighbor_id].su_type)
                for neighbor_id in ids
                if 0 <= neighbor_id < len(nodes)
            )
            if actual_counter != Counter(node.hop1_su):
                errors.append(
                    f"节点{node_id}(SU{node.su_type}): hop1_su={dict(node.hop1_su)} "
                    f"!= 实际={dict(actual_counter)}"
                )

            valid, node_errors = node.validate_hop1_consistency()
            if not valid:
                errors.extend(str(x) for x in node_errors)
            target_degree = a._node_target_degree(node)
            if target_degree is None or int(node.get_hop1_degree()) != int(target_degree):
                errors.append(
                    f"节点{node_id}(SU{node.su_type}): 实际连接度{node.get_hop1_degree()} "
                    f"!= 目标连接度{target_degree}"
                )

            fixed_ids = set(int(x) for x in set(node.fixed_hop1_ids or set()))
            for fixed_id in sorted(fixed_ids):
                if fixed_id not in ids:
                    errors.append(f"节点{node_id}: fixed边{fixed_id}不在hop1_ids中")
                    continue
                if not (0 <= fixed_id < len(nodes)):
                    continue
                if node_id not in set(nodes[fixed_id].fixed_hop1_ids or set()):
                    errors.append(f"fixed边不对称: {node_id}<->{fixed_id}")

        errors.extend(a._su11_external_role_errors(nodes))

        for node in nodes:
            node_id = int(node.global_id)
            ids = [int(x) for x in list(node.hop1_ids or [])]
            fixed_ids = set(int(x) for x in set(node.fixed_hop1_ids or set()))
            neighbor_nodes = a._current_neighbor_nodes(nodes, node)
            neighbor_types = [int(x.su_type) for x in neighbor_nodes]
            port_sets = a._port_sets_for_node(node)
            if port_sets:
                if not can_match_ports_partial(neighbor_nodes, port_sets, center_node=node):
                    errors.append(
                        f"节点{node_id}(SU{node.su_type}): partial hop1={sorted(neighbor_types)} "
                        f"不满足端口规则 {format_port_patterns_debug(port_sets)}"
                    )
                elif not can_match_ports_exact(neighbor_nodes, port_sets, center_node=node):
                    errors.append(
                        f"节点{node_id}(SU{node.su_type}): hop1={sorted(neighbor_types)} "
                        f"不满足完整端口规则 {format_port_patterns_debug(port_sets)}"
                    )

            su_type = int(node.su_type)
            if su_type in {14, 15, 16}:
                paired = [x for x in neighbor_types if x in {14, 15, 16}]
                if len(paired) != 1:
                    errors.append(
                        f"节点{node_id}(SU{su_type}): 14/15/16固定配对必须恰好一个, "
                        f"pair_neighbors={sorted(paired)}"
                    )
                if su_type == 16 and 16 in paired:
                    errors.append(f"节点{node_id}(SU16): 禁止16-16固定配对")
                aromatic = [
                    x for x in neighbor_types
                    if 5 <= x <= 13 and not (su_type == 15 and x == 11)
                ]
                if aromatic:
                    errors.append(
                        f"节点{node_id}(SU{su_type}): 非法芳香邻居={sorted(aromatic)}"
                    )
            if su_type in {17, 18}:
                paired = [x for x in neighbor_types if x in {17, 18}]
                if len(paired) != 1:
                    errors.append(
                        f"节点{node_id}(SU{su_type}): 17/18固定配对必须恰好一个, "
                        f"pair_neighbors={sorted(paired)}"
                    )
                if su_type == 18 and 18 in paired:
                    errors.append(f"节点{node_id}(SU18): 禁止18-18固定配对")

            if su_type == 10:
                su10_ids = [
                    int(neighbor_id)
                    for neighbor_id in ids
                    if 0 <= int(neighbor_id) < len(nodes)
                    and int(nodes[int(neighbor_id)].su_type) == 10
                ]
                if len(su10_ids) != 1:
                    errors.append(
                        f"节点{node_id}(SU10): 外接SU10角色边必须恰好一个, "
                        f"su10_neighbors={su10_ids}"
                    )
                elif (
                    su10_ids[0] not in fixed_ids
                    or node_id not in set(nodes[su10_ids[0]].fixed_hop1_ids or set())
                ):
                    errors.append(
                        f"节点{node_id}(SU10): 外接SU10边{su10_ids[0]}未双向锁定"
                    )

            if su_type == 12 and 12 not in neighbor_types:
                errors.append(
                    f"节点{node_id}(SU12): 完整1-hop必须至少包含一个SU12邻居"
                )

            if su_type in set(a._carbon_su_set) and not a._node_exact_hop1_shift_state_ok(
                node,
                require_full=True,
            ):
                errors.append(
                    f"节点{node_id}(SU{su_type}): 完整1-hop={sorted(neighbor_types)} "
                    "不存在于活动Layer1模板库"
                )

        for node in nodes:
            if int(node.su_type) not in {27, 29, 31}:
                continue
            neighbor_nodes = a._current_neighbor_nodes(nodes, node)
            if has_forbidden_fixed_connector_special_combo(node, neighbor_nodes):
                errors.append(
                    f"节点{node.global_id}(SU{node.su_type}): fixed connector "
                    "不能同时连接两个double d3 special"
                )
            if a._violates_fixed_connector_special_end_rules(nodes, node):
                errors.append(
                    f"节点{node.global_id}(SU{node.su_type}): fixed connector特殊端基组合非法"
                )

        for node in nodes:
            if a._violates_terminal_neighbor_limit(nodes, node):
                errors.append(
                    f"节点{node.global_id}(SU{node.su_type}): 结构单元最多只能连接一个端基结构"
                )

        for row in a._forbidden_p_pair_rows(nodes):
            errors.append(
                "P+P固定锚点非法连接: "
                f"double_id={int(row.get('double_id', -1))} "
                f"p_connectors={list(row.get('p_connector_ids', []) or [])}"
            )

        for node in nodes:
            if int(node.su_type) == 19 and a._node_anchor_partition(node) not in {'thio', 'ether'}:
                errors.append(
                    f"节点{node.global_id}(SU19): 缺少严格分区标记"
                )
            required = set(int(x) for x in a._required_external_candidates_for_node(node))
            neighbor_types = [
                int(nodes[int(neighbor_id)].su_type)
                for neighbor_id in list(node.hop1_ids or [])
                if 0 <= int(neighbor_id) < len(nodes)
            ]
            external_count = sum(1 for su_type in neighbor_types if su_type in required)
            target_count = int(a._node_fixed_anchor_target(node))
            if required and not a._has_required_external_for_node(node, nodes):
                errors.append(
                    f"节点{node.global_id}(SU{node.su_type}): 缺少必需外接锚点"
                )
            if required and target_count > 0 and external_count != target_count:
                errors.append(
                    f"节点{node.global_id}(SU{node.su_type}): 固定锚点边数"
                    f"{external_count} != {target_count}"
                )

        errors.extend(
            a._fixed_anchor_invariant_errors(
                nodes,
                check_thio=True,
                check_ether=True,
                check_amine=True,
                check_halogen=True,
            )
        )

        for node in nodes:
            if int(node.su_type) != 9:
                continue
            carbonyl_ids = [
                int(neighbor_id)
                for neighbor_id in list(node.hop1_ids or [])
                if 0 <= int(neighbor_id) < len(nodes)
                and int(nodes[int(neighbor_id)].su_type) in {0, 1, 2, 3}
            ]
            if len(carbonyl_ids) != 1:
                errors.append(
                    f"节点{node.global_id}(SU9): 羰基固定角色边必须恰好一个, "
                    f"carbonyl_neighbors={carbonyl_ids}"
                )
            elif (
                carbonyl_ids[0] not in set(node.fixed_hop1_ids or set())
                or int(node.global_id) not in set(nodes[carbonyl_ids[0]].fixed_hop1_ids or set())
            ):
                errors.append(
                    f"节点{node.global_id}(SU9): 羰基边{carbonyl_ids[0]}未双向锁定"
                )

        seen_edges: Set[Tuple[int, int]] = set()
        for node in nodes:
            for neighbor_id in list(node.hop1_ids or []):
                neighbor_id = int(neighbor_id)
                if not (0 <= neighbor_id < len(nodes)):
                    continue
                edge = tuple(sorted((int(node.global_id), neighbor_id)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if not a._node_edge_semantics_ok(
                    nodes,
                    node,
                    nodes[neighbor_id],
                    E_target=E_target,
                ):
                    errors.append(
                        f"非法1-hop边: {node.global_id}(SU{node.su_type}) <-> "
                        f"{neighbor_id}(SU{nodes[neighbor_id].su_type})"
                    )

        if verbose and errors:
            a._print_consistency_errors(errors, prefix='Validate')
        return not errors, errors
