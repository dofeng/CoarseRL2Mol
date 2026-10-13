"""Shared state-aware SU port rules, without model or plotting dependencies."""

from dataclasses import dataclass
import re
from typing import Any, FrozenSet, Iterable, List, Tuple, Dict, Optional, Set, Union, Sequence
from model.shared.topology_contract import (AROMATIC_RING_SU_TYPES,
    AROMATIC_RING_NEIGHBOR_SU_TYPES, CARBONYL_BODY_SU_TYPES)


@dataclass(frozen=True)
class PortRuleNode:
    su_type: int
    target_hop1_degree: int
    special_anchor_mode: Optional[str] = None
    special_anchor_partition: Optional[str] = None


def port_rule_nodes(labels, adjacency, metadata=None):
    """Use actual degrees and explicit modes, inferring unspecified anchors."""
    nodes = {}
    for gid, su in labels.items():
        row = (metadata or {}).get(gid, {})
        label = row.get('su_label', '')
        match = re.match(r'(single|double)', label)
        mode = row.get('special_anchor_mode') or (match.group(1) if match else None)
        partition = row.get('special_anchor_partition') or None
        kinds = [labels.get(n) for n in adjacency.get(gid, ())]
        if su in {19, 20}:
            anchors = {0, 27} if su == 20 else {2, 28, 29, 31}
            if mode is None:
                mode = 'double' if sum(k in anchors for k in kinds) >= 2 else 'single'
            if su == 19 and partition is None:
                partition = 'thio' if 31 in kinds else 'ether'
        nodes[gid] = PortRuleNode(su, len(adjacency.get(gid, ())), mode, partition)
    return nodes

SPECIAL_DEGREE_PRIORS: Dict[int, Dict[int, float]] = {
    19: {1: 0.35, 2: 0.60, 3: 0.05},
    20: {1: 0.35, 2: 0.60, 3: 0.05},
    21: {2: 0.90, 3: 0.10},
}

SPECIAL_MODE_FORBIDDEN_NEIGHBORS: Dict[Tuple[int, int], Set[int]] = {
    (19, 1): {28},
}

TERMINAL_PORT_BASE_SU: Set[int] = {1, 4, 16, 18, 22, 28, 32}


@dataclass(frozen=True)
class Hop1PortToken:
    """State-aware neighbor token used by 1-hop port rules."""

    su_type: int
    degrees: Tuple[int, ...] = ()
    modes: Tuple[str, ...] = ()
    partitions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Hop1PortSignature:
    """A complete ordered set of ports, with optional center-node constraints."""

    ports: Tuple[FrozenSet[Union[int, Hop1PortToken]], ...]
    center_modes: Tuple[str, ...] = ()
    center_partitions: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.ports)

    def __iter__(self):
        return iter(self.ports)

    def __getitem__(self, index: int) -> FrozenSet[Union[int, Hop1PortToken]]:
        return self.ports[index]


def hop1_token(su_type: int,
               degrees: Optional[Iterable[int]] = None,
               modes: Optional[Iterable[str]] = None,
               partitions: Optional[Iterable[str]] = None) -> Hop1PortToken:
    return Hop1PortToken(
        su_type=int(su_type),
        degrees=tuple(sorted(set(int(x) for x in list(degrees or [])))),
        modes=tuple(sorted(set(str(x) for x in list(modes or [])))),
        partitions=tuple(sorted(set(str(x) for x in list(partitions or [])))),
    )


def hop1_signature(*ports: Iterable[Union[int, Hop1PortToken]],
                   center_modes: Optional[Iterable[str]] = None,
                   center_partitions: Optional[Iterable[str]] = None) -> Hop1PortSignature:
    return Hop1PortSignature(
        ports=tuple(
            frozenset(_normalize_port_entry(entry) for entry in list(port or []))
            for port in ports
        ),
        center_modes=tuple(sorted(set(str(x) for x in list(center_modes or [])))),
        center_partitions=tuple(sorted(set(str(x) for x in list(center_partitions or [])))),
    )


def _normalize_special_mode(mode: Optional[Any]) -> Optional[str]:
    if mode is None:
        return None
    txt = str(mode)
    if txt.startswith('double'):
        return 'double'
    if txt.startswith('single'):
        return 'single'
    return txt


def _node_su_type(node_or_su: Any) -> Optional[int]:
    try:
        if isinstance(node_or_su, Hop1PortToken):
            return int(node_or_su.su_type)
        if hasattr(node_or_su, 'su_type'):
            return int(getattr(node_or_su, 'su_type'))
        return int(node_or_su)
    except Exception:
        return None


def _node_target_degree(node_or_su: Any) -> Optional[int]:
    try:
        val = getattr(node_or_su, 'target_hop1_degree', None)
        return int(val) if val is not None else None
    except Exception:
        return None


def _node_anchor_mode(node_or_su: Any) -> Optional[str]:
    try:
        mode = _normalize_special_mode(getattr(node_or_su, 'special_anchor_mode', None))
        if mode is not None:
            return mode
        fixed = getattr(node_or_su, 'target_fixed_anchor_count', None)
        if fixed is not None:
            fixed_i = int(fixed)
            if fixed_i >= 2:
                return 'double'
            if fixed_i == 1:
                return 'single'
        return None
    except Exception:
        return None


def _node_anchor_partition(node_or_su: Any) -> Optional[str]:
    try:
        val = getattr(node_or_su, 'special_anchor_partition', None)
        return str(val) if val is not None else None
    except Exception:
        return None


def _normalize_port_entry(entry: Union[int, Hop1PortToken]) -> Union[int, Hop1PortToken]:
    if isinstance(entry, Hop1PortToken):
        return entry
    return int(entry)


def port_entry_base_su(entry: Union[int, Hop1PortToken]) -> int:
    if isinstance(entry, Hop1PortToken):
        return int(entry.su_type)
    return int(entry)


def port_entry_matches(entry: Union[int, Hop1PortToken], neighbor: Any) -> bool:
    neighbor_su = _node_su_type(neighbor)
    if neighbor_su is None:
        return False
    if isinstance(entry, Hop1PortToken):
        if int(neighbor_su) != int(entry.su_type):
            return False

        # Integer-only callers are legacy/template validators.  They can only
        # validate the coarse SU type; node-state constraints are enforced by
        # Layer1/adjusters when actual nodes are available.
        if not hasattr(neighbor, 'su_type'):
            return True

        if entry.degrees:
            degree = _node_target_degree(neighbor)
            if degree is None or int(degree) not in set(int(x) for x in entry.degrees):
                return False
        if entry.modes:
            mode = _node_anchor_mode(neighbor)
            if mode is None or str(mode) not in set(str(x) for x in entry.modes):
                return False
        if entry.partitions:
            partition = _node_anchor_partition(neighbor)
            if partition is None or str(partition) not in set(str(x) for x in entry.partitions):
                return False
        return True
    return int(neighbor_su) == int(entry)


def port_entry_debug(entry: Union[int, Hop1PortToken]) -> str:
    if not isinstance(entry, Hop1PortToken):
        return str(int(entry))
    suffix = []
    if entry.degrees:
        suffix.append('d' + ''.join(str(int(x)) for x in entry.degrees))
    if entry.modes:
        suffix.append('mode=' + '/'.join(str(x) for x in entry.modes))
    if entry.partitions:
        suffix.append('part=' + '/'.join(str(x) for x in entry.partitions))
    return f"{int(entry.su_type)}" + (f"({';'.join(suffix)})" if suffix else "")


def signature_matches_center(signature: Hop1PortSignature, center_node: Optional[Any]) -> bool:
    if center_node is None:
        return True
    if signature.center_modes:
        mode = _node_anchor_mode(center_node)
        if mode is not None and str(mode) not in set(str(x) for x in signature.center_modes):
            return False
    if signature.center_partitions:
        partition = _node_anchor_partition(center_node)
        if partition is not None and str(partition) not in set(str(x) for x in signature.center_partitions):
            return False
    return True


def is_terminal_port_node(node_or_su: Any) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None:
        return False
    if int(su_i) in TERMINAL_PORT_BASE_SU:
        return True
    if int(su_i) in {19, 20}:
        degree = _node_target_degree(node_or_su)
        return bool(degree is not None and int(degree) == 1)
    return False


def terminal_neighbor_count(neighbors: Iterable[Any]) -> int:
    return int(sum(1 for nb in list(neighbors or []) if is_terminal_port_node(nb)))


def violates_special_d3_terminal_limit(su_type: int,
                                       target_degree: Optional[int],
                                       neighbors: Iterable[Any]) -> bool:
    """Return whether a d3 special center would be capped by too many terminals."""
    try:
        if int(su_type) not in {19, 20}:
            return False
        if target_degree is None or int(target_degree) != 3:
            return False
    except Exception:
        return False
    return bool(terminal_neighbor_count(neighbors) > 1)


def is_single_special_d3_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) != 3:
        return False
    mode = _node_anchor_mode(node_or_su)
    return str(mode) != 'double'


def is_double_special_d23_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) not in {2, 3}:
        return False
    return str(_node_anchor_mode(node_or_su)) == 'double'


def is_double_special_d3_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) != 3:
        return False
    return str(_node_anchor_mode(node_or_su)) == 'double'


def fixed_connector_special_pair_ok(center_node: Any, neighbors: Iterable[Any]) -> bool:
    center_su = _node_su_type(center_node)
    if center_su not in {27, 29, 31}:
        return True
    special_su = 20 if int(center_su) == 27 else 19
    relevant = [
        nb for nb in list(neighbors or [])
        if _node_su_type(nb) == int(special_su)
    ]
    if len(relevant) < 2:
        return True
    double_d3_count = int(sum(
        1 for nb in relevant
        if is_double_special_d3_port_node(nb, int(special_su))
    ))
    return bool(double_d3_count < 2)


def has_forbidden_fixed_connector_special_combo(center_node: Any, neighbors: Iterable[Any]) -> bool:
    """Return whether a fixed connector has an impossible special pairing."""
    return not fixed_connector_special_pair_ok(center_node, neighbors)


D23 = (2, 3)
D123 = (1, 2, 3)
SINGLE = ('single',)
DOUBLE = ('double',)
ETHER = ('ether',)
THIO = ('thio',)

SU19_D123 = hop1_token(19, degrees=D123)
SU19_D23 = hop1_token(19, degrees=D23)
SU19_D1 = hop1_token(19, degrees=(1,))
SU19_SINGLE_D23 = hop1_token(19, degrees=D23, modes=SINGLE)
SU20_D123 = hop1_token(20, degrees=D123)
SU20_D23 = hop1_token(20, degrees=D23)
SU20_D1 = hop1_token(20, degrees=(1,))
SU20_SINGLE_D23 = hop1_token(20, degrees=D23, modes=SINGLE)
SU21_D23 = hop1_token(21, degrees=D23)

ARO = set(AROMATIC_RING_SU_TYPES)
ARO_NO9 = set(AROMATIC_RING_NEIGHBOR_SU_TYPES[9])
ARO_NO10 = set(AROMATIC_RING_NEIGHBOR_SU_TYPES[10])
ARO_HET_RING = set(AROMATIC_RING_NEIGHBOR_SU_TYPES[26])
CARBONYL_BODY_PORTS = set(CARBONYL_BODY_SU_TYPES - {19, 20, 21}) | {SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_1 = {23, 11, 22, 24, 25, 2, 29, 31, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2 = {23, 11, 24, 25, 2, 29, 31, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_1_19_ETHER = {23, 11, 22, 24, 25, 2, 29, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2_19_ETHER = {23, 11, 24, 25, 2, 29, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_1_19_THIO = {23, 11, 22, 24, 25, 31, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2_19_THIO = {23, 11, 24, 25, 31, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
O_ANCHORS = {2, 28, 29}
O_ANCHORS_NO28 = {2, 29}
S_ANCHORS = {31}
HAL_EXT = {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}

HOP1_PORT_COMBINATIONS: Dict[int, Any] = {
    0: [
        {9, 23, 24, 25, 22, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {6, SU20_D123},
    ],
    1: [{9, 23, 24, 25, 14, 15, SU19_D23, SU20_D23, SU21_D23}],
    2: [
        CARBONYL_BODY_PORTS,
        {5, SU19_D123},
    ],
    3: [
        CARBONYL_BODY_PORTS,
        CARBONYL_BODY_PORTS - {22},
    ],
    4: [{23, 24, 25}],
    5: [ARO, ARO, {2, 28, 29}],
    6: [ARO, ARO, {0, 27}],
    7: [ARO, ARO, {31}],
    8: [ARO, ARO, {32}],
    9: [ARO_NO9, ARO_NO9, {0, 1, 2, 3}],
    10: [ARO_NO10, ARO_NO10, {10}],
    11: [ARO, ARO, {23, 24, 25, 22, 15, 17, SU19_D23, SU20_D23, SU21_D23}],
    12: [ARO, ARO, {12}],
    13: [ARO, ARO],
    14: [
        {23, 24, 25, 22, 2, 1, 0, 3, SU19_D23, SU20_D23, SU21_D23},
        {23, 24, 25, 22, 2, 1, 0, 3, 4, SU19_D23, SU20_D23, SU21_D23},
        {14, 15, 16},
    ],
    15: [{23, 24, 25, 22, 11, 2, 1, 0, 3, SU19_D23, SU20_D23, SU21_D23}, {14, 15, 16}],
    16: [{14, 15}],
    17: [{23, 24, 25, 11, 19, 20, 21, 2, 0, 3}, {17, 18}],
    18: [{17}],
    19: [
        hop1_signature(O_ANCHORS_NO28, center_modes=SINGLE, center_partitions=ETHER),
        hop1_signature(S_ANCHORS, center_modes=SINGLE, center_partitions=THIO),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            O_ANCHORS,
            center_modes=SINGLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            S_ANCHORS,
            center_modes=SINGLE,
            center_partitions=THIO,
        ),
        hop1_signature(O_ANCHORS, O_ANCHORS_NO28, center_modes=DOUBLE, center_partitions=ETHER),
        hop1_signature(S_ANCHORS, S_ANCHORS, center_modes=DOUBLE, center_partitions=THIO),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            SAT_EXT_2_19_ETHER,
            O_ANCHORS,
            center_modes=SINGLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            SAT_EXT_2_19_THIO,
            S_ANCHORS,
            center_modes=SINGLE,
            center_partitions=THIO,
        ),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            O_ANCHORS,
            O_ANCHORS_NO28,
            center_modes=DOUBLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            S_ANCHORS,
            S_ANCHORS,
            center_modes=DOUBLE,
            center_partitions=THIO,
        ),
    ],
    20: [
        hop1_signature({0, 27}, center_modes=SINGLE),
        hop1_signature({23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}, {0, 27}, center_modes=SINGLE),
        hop1_signature({0, 27}, {0, 27}, center_modes=DOUBLE),
        hop1_signature(
            {23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {23, 11, 24, 25, 27, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {0, 27},
            center_modes=SINGLE,
        ),
        hop1_signature(
            {23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {0, 27},
            {0, 27},
            center_modes=DOUBLE,
        ),
    ],
    21: [
        hop1_signature(HAL_EXT, {32}),
        hop1_signature(HAL_EXT, HAL_EXT, {32}),
    ],
    22: [{23, 11, 24, 25, 2, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}],
    23: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    24: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    25: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    26: [ARO_HET_RING, ARO_HET_RING],
    27: [
        hop1_signature({6, SU20_D123}, {6, SU20_D23}),
        hop1_signature({SU20_D123}, {SU20_D23}),
    ],
    28: [{5, SU19_D23}],
    29: [
        hop1_signature({5, SU19_D123}, {5, SU19_D23}),
        hop1_signature({5}, {SU19_D23}),
        hop1_signature({SU19_D123}, {SU19_D23}),
    ],
    30: [ARO_HET_RING, ARO_HET_RING],
    31: [
        hop1_signature({7, SU19_D123}, {7, SU19_D23}),
        hop1_signature({SU19_D123}, {SU19_D23}),
    ],
    32: [{8, SU21_D23}],
}

def iter_port_signature_objects(port_patterns: Any) -> List[Hop1PortSignature]:
    """返回带中心状态约束的签名对象。"""
    if not port_patterns:
        return []

    if isinstance(port_patterns, Hop1PortSignature):
        return [port_patterns]

    if isinstance(port_patterns, (list, tuple)) and port_patterns:
        first = port_patterns[0]
        if isinstance(first, set) and all(isinstance(item, set) for item in port_patterns):
            return [
                hop1_signature(*[
                    {_normalize_port_entry(v) for v in allowed}
                    for allowed in port_patterns
                ])
            ]

    signatures: List[Hop1PortSignature] = []
    if isinstance(port_patterns, (list, tuple)):
        for signature in port_patterns:
            if isinstance(signature, Hop1PortSignature):
                signatures.append(signature)
                continue
            if not isinstance(signature, (list, tuple)):
                continue
            sig_sets: List[Set[Union[int, Hop1PortToken]]] = []
            ok = True
            for allowed in signature:
                if not isinstance(allowed, set):
                    ok = False
                    break
                sig_sets.append(set(_normalize_port_entry(v) for v in allowed))
            if ok and sig_sets:
                signatures.append(hop1_signature(*sig_sets))
    return signatures


def get_port_pattern_degrees(port_patterns: Any) -> Tuple[int, ...]:
    """返回该端口规则允许的所有合法连接度。"""
    return tuple(sorted(set(len(sig.ports) for sig in iter_port_signature_objects(port_patterns) if sig.ports)))


def format_port_patterns_debug(port_patterns: Any) -> List[List[List[str]]]:
    """将端口规则格式化为便于打印/调试的整数列表。"""
    return [
        [
            sorted(port_entry_debug(v) for v in allowed)
            for allowed in sig.ports
        ]
        for sig in iter_port_signature_objects(port_patterns)
    ]


# 每个SU的所有端口允许的邻居类型的并集
SU_FIXED_CONNECTIONS = {
    su: sorted({
        int(port_entry_base_su(v))
        for sig in iter_port_signature_objects(port_sets)
        for allowed in sig.ports
        for v in allowed
    })
    for su, port_sets in HOP1_PORT_COMBINATIONS.items()
}

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
