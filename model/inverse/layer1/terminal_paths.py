"""Whole-path guard for joining two terminal-bearing linear half-chains."""

from typing import Mapping, Optional, Sequence, Tuple

from ...shared.inverse_common import is_terminal_port_node, SU_CONNECTION_DEGREE


def terminal_path_reason(
    nodes,
    *,
    proposed_edge: Optional[Tuple[int, int]] = None,
    neighbor_overrides: Optional[Mapping[int, Sequence[int]]] = None,
    start_ids: Optional[Sequence[int]] = None,
) -> Optional[str]:
    overrides = neighbor_overrides or {}

    def node_at(gid):
        try:
            node = nodes[gid]
        except (KeyError, IndexError):
            return None
        return node if int(node.global_id) == gid else None

    def neighbors(gid, node):
        ids = set(int(x) for x in overrides.get(gid, node.hop1_ids))
        if proposed_edge is not None:
            a, b = proposed_edge
            if gid == a:
                ids.add(b)
            elif gid == b:
                ids.add(a)
        return ids

    seeds = start_ids
    if seeds is None:
        seeds = proposed_edge if proposed_edge is not None else [
            int(n.global_id) for n in (nodes.values() if isinstance(nodes, Mapping) else nodes)
            if is_terminal_port_node(n)
        ]
    checked = set()
    for seed in seeds:
        pending, component, terminals = [int(seed)], set(), []
        closed_linear = True
        while pending:
            gid = pending.pop()
            if gid in component:
                continue
            if gid in checked:
                closed_linear = False
                break
            node = node_at(gid)
            if node is None:
                closed_linear = False
                break
            adjacent = neighbors(gid, node)
            degree = getattr(node, 'target_hop1_degree', None)
            degree = SU_CONNECTION_DEGREE.get(int(node.su_type), 0) if degree is None else int(degree)
            # A branch, aromatic endpoint or unfilled port still connects to
            # the larger skeleton; it is not a closed terminal-to-terminal path.
            if int(node.su_type) in {5, 6, 7, 8, 9, 10, 11, 12, 13} or degree > 2 or len(adjacent) != degree:
                closed_linear = False
                break
            component.add(gid)
            if is_terminal_port_node(node):
                terminals.append(gid)
            pending.extend(adjacent - component)
        checked.update(component)
        if closed_linear and len(terminals) >= 2:
            return f"terminal_halfchains_joined:{sorted(component)}"
    return None
