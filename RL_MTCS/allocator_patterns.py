from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


AROMATIC_SET = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
TERMINAL_SU = {1, 4, 16, 18, 22, 28, 32}


@dataclass(frozen=True)
class SpecialCarbonProfile:
    su_type: int
    name: str
    connector_su: Tuple[int, ...]
    terminal_anchor_su: Tuple[int, ...]
    allow_double_d2: bool = True


SPECIAL_CARBON_PROFILES: Dict[int, SpecialCarbonProfile] = {
    19: SpecialCarbonProfile(
        su_type=19,
        name='su19',
        connector_su=(29, 31),
        terminal_anchor_su=(28,),
        allow_double_d2=True,
    ),
    20: SpecialCarbonProfile(
        su_type=20,
        name='su20',
        connector_su=(27,),
        terminal_anchor_su=(),
        allow_double_d2=True,
    ),
    21: SpecialCarbonProfile(
        su_type=21,
        name='su21',
        connector_su=(),
        terminal_anchor_su=(32,),
        allow_double_d2=False,
    ),
}


@dataclass
class SpecialChainPattern:
    chain_type: str
    composition: List[int]
    origin_type: str
    phase: str = 'closed'
    source_ids: List[int] = field(default_factory=list)
    consumed_ids: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def node_id(node: Any) -> int:
    return _safe_int(getattr(node, 'global_id', -1), -1)


def node_su(node: Any) -> int:
    return _safe_int(getattr(node, 'su_type', -1), -1)


def node_degree(node: Any) -> Optional[int]:
    raw = getattr(node, 'target_degree', None)
    if raw is None:
        raw = getattr(node, 'target_hop1_degree', None)
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            return None
    hop1_ids = list(getattr(node, 'hop1_ids', ()) or ())
    if hop1_ids:
        return int(len(hop1_ids))
    hop1 = list(getattr(node, 'hop1', ()) or ())
    if hop1:
        return int(len(hop1))
    return None


def node_fixed_anchor_count(node: Any) -> int:
    raw = getattr(node, 'target_fixed_anchor_count', None)
    if raw is not None:
        try:
            return max(0, int(raw))
        except Exception:
            return 0
    return 0


def node_anchor_partition(node: Any) -> Optional[str]:
    raw = getattr(node, 'anchor_partition', None)
    if raw is None:
        raw = getattr(node, 'special_anchor_partition', None)
    if raw is None:
        return None
    try:
        return str(raw)
    except Exception:
        return None


def node_anchor_mode(node: Any) -> Optional[str]:
    raw = getattr(node, 'anchor_mode', None)
    if raw is None:
        raw = getattr(node, 'special_anchor_mode', None)
    if raw is None:
        return None
    try:
        return str(raw)
    except Exception:
        return None


def is_aromatic_node(node: Any) -> bool:
    return int(node_su(node)) in set(int(x) for x in AROMATIC_SET)


def is_special_carbon(node: Any) -> bool:
    return int(node_su(node)) in set(int(x) for x in SPECIAL_CARBON_PROFILES.keys())


def is_double_special(node: Any) -> bool:
    mode = str(node_anchor_mode(node) or '')
    if str(mode).startswith('double'):
        return True
    return int(node_fixed_anchor_count(node)) >= 2


def endpoint_class_for_node(node: Any) -> str:
    su_i = int(node_su(node))
    deg_i = int(node_degree(node) or 0)
    if su_i in set(int(x) for x in AROMATIC_SET):
        return 'aromatic'
    if su_i in set(int(x) for x in TERMINAL_SU):
        return 'terminal'
    if su_i in {19, 20} and int(deg_i) == 1:
        return 'terminal'
    return 'aliphatic'


def neighbor_nodes(node: Any, lookup: Dict[int, Any]) -> List[Any]:
    out: List[Any] = []
    for nid in list(getattr(node, 'hop1_ids', ()) or ()):
        nb = lookup.get(int(nid))
        if nb is not None:
            out.append(nb)
    return out


def other_endpoints_for_connector(connector: Any, center_gid: int, lookup: Dict[int, Any]) -> List[Any]:
    out: List[Any] = []
    for nid in list(getattr(connector, 'hop1_ids', ()) or ()):
        nid_i = int(nid)
        if int(nid_i) == int(center_gid):
            continue
        nb = lookup.get(int(nid_i))
        if nb is not None:
            out.append(nb)
    return out


def _sort_nodes(nodes: Sequence[Any]) -> List[Any]:
    return sorted(list(nodes or []), key=lambda n: (int(node_id(n)), int(node_su(n))))


def _connector_records(node: Any,
                       profile: SpecialCarbonProfile,
                       lookup: Dict[int, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for conn in _sort_nodes([
        nb for nb in neighbor_nodes(node, lookup)
        if int(node_su(nb)) in set(int(x) for x in profile.connector_su)
    ]):
        endpoints = _sort_nodes(other_endpoints_for_connector(conn, int(node_id(node)), lookup))
        records.append({
            'connector': conn,
            'endpoints': list(endpoints),
        })
    return records


def _endpoint_signature(endpoint: Optional[Any],
                        profile: SpecialCarbonProfile) -> str:
    if endpoint is None:
        return 'none'
    su_i = int(node_su(endpoint))
    if bool(is_aromatic_node(endpoint)):
        return 'aromatic'
    if su_i in set(int(x) for x in profile.terminal_anchor_su):
        return 'terminal'
    if su_i == int(profile.su_type):
        return 'special_double' if bool(is_double_special(endpoint)) else 'special_single'
    if bool(is_special_carbon(endpoint)):
        return 'special_other'
    if str(endpoint_class_for_node(endpoint)) == 'terminal':
        return 'terminal'
    return 'aliphatic'


def _pick_primary_endpoint(endpoints: Sequence[Any],
                           profile: SpecialCarbonProfile) -> Optional[Any]:
    if not endpoints:
        return None
    order = {
        'aromatic': 0,
        'special_single': 1,
        'special_double': 2,
        'terminal': 3,
        'special_other': 4,
        'aliphatic': 5,
        'none': 6,
    }
    return min(
        list(endpoints),
        key=lambda ep: (
            int(order.get(_endpoint_signature(ep, profile), 99)),
            int(node_degree(ep) or 99),
            int(node_id(ep)),
        ),
    )


def _pattern_metadata(pattern_name: str,
                      node: Any,
                      profile: Optional[SpecialCarbonProfile] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
        'special_pattern': str(pattern_name),
        'source_center_su': int(node_su(node)),
        'source_center_degree': int(node_degree(node) or -1),
        'source_center_partition': node_anchor_partition(node),
        'source_center_anchor_mode': node_anchor_mode(node),
    }
    if profile is not None:
        meta['special_family'] = str(profile.name)
    if extra:
        meta.update(dict(extra))
    return meta


def _get_profile(node: Any) -> Optional[SpecialCarbonProfile]:
    return SPECIAL_CARBON_PROFILES.get(int(node_su(node)))


def _has_invalid_capped_su19_path(node: Any, lookup: Dict[int, Any]) -> bool:
    if int(node_su(node)) != 19:
        return False
    nbs = neighbor_nodes(node, lookup)
    if len([nb for nb in nbs if int(node_su(nb)) == 28]) >= 2:
        return True
    if any(int(node_su(nb)) == 28 for nb in nbs):
        for conn in [nb for nb in nbs if int(node_su(nb)) in {29, 31}]:
            for ep in other_endpoints_for_connector(conn, int(node_id(node)), lookup):
                if int(node_su(ep)) != 19:
                    continue
                if any(int(node_su(ep2)) == 28 for ep2 in neighbor_nodes(ep, lookup)):
                    return True
    return False


def _build_single_d1_connector_pattern(node: Any,
                                       profile: SpecialCarbonProfile,
                                       lookup: Dict[int, Any]) -> Optional[SpecialChainPattern]:
    if int(node_degree(node) or 0) != 1:
        return None
    nbs = neighbor_nodes(node, lookup)
    connectors = [nb for nb in nbs if int(node_su(nb)) in set(int(x) for x in profile.connector_su)]
    fixeds = [nb for nb in nbs if int(node_su(nb)) in set(int(x) for x in profile.terminal_anchor_su)]
    aro = [nb for nb in nbs if endpoint_class_for_node(nb) == 'aromatic']
    ali = [nb for nb in nbs if endpoint_class_for_node(nb) == 'aliphatic']
    consumed = {int(node_id(node))} | {int(node_id(x)) for x in connectors + fixeds}
    meta_base = {
        'connector_ids': [int(node_id(x)) for x in connectors],
        'connector_types': [int(node_su(x)) for x in connectors],
        'fixed_ids': [int(node_id(x)) for x in fixeds],
        'fixed_types': [int(node_su(x)) for x in fixeds],
        'neighbor_su': [int(node_su(x)) for x in nbs],
    }
    if aro:
        return SpecialChainPattern(
            chain_type='side',
            composition=[11, 23, 22],
            origin_type=f'C{int(profile.su_type)}d1',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + fixeds + aro[:1]],
            consumed_ids=set(consumed) | {int(node_id(x)) for x in aro[:1]},
            metadata=_pattern_metadata(
                f'connector_d1_{profile.name}_aromatic',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 1, '22': 1, '23': 1, '24': 0, '25': 0}},
            ),
            priority=30,
        )
    if ali:
        return SpecialChainPattern(
            chain_type='side',
            composition=[11, 23, 23, 22],
            origin_type=f'E{int(profile.su_type)}d1',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + fixeds + ali[:1]],
            consumed_ids=set(consumed) | {int(node_id(x)) for x in ali[:1]},
            metadata=_pattern_metadata(
                f'connector_d1_{profile.name}_aliphatic',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 1, '22': 1, '23': 2, '24': 0, '25': 0}},
            ),
            priority=32,
        )
    return None


def _build_single_d2_bridge_pattern(node: Any,
                                    profile: SpecialCarbonProfile,
                                    lookup: Dict[int, Any]) -> Optional[SpecialChainPattern]:
    if int(node_degree(node) or 0) != 2 or bool(is_double_special(node)):
        return None
    nbs = neighbor_nodes(node, lookup)
    aro = [nb for nb in nbs if endpoint_class_for_node(nb) == 'aromatic']
    term = [nb for nb in nbs if endpoint_class_for_node(nb) == 'terminal']
    ali = [nb for nb in nbs if endpoint_class_for_node(nb) == 'aliphatic']
    connectors = [nb for nb in nbs if int(node_su(nb)) in set(int(x) for x in profile.connector_su)]
    consumed = {int(node_id(node))} | {int(node_id(x)) for x in connectors}
    meta_base = {
        'connector_ids': [int(node_id(x)) for x in connectors],
        'connector_types': [int(node_su(x)) for x in connectors],
        'neighbor_su': [int(node_su(x)) for x in nbs],
    }
    if aro:
        return SpecialChainPattern(
            chain_type='bridge',
            composition=[11, 23, 23, 11],
            origin_type=f'B{int(profile.su_type)}d2',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + aro[:1]],
            consumed_ids=set(consumed) | {int(node_id(x)) for x in aro[:1]},
            metadata=_pattern_metadata(
                f'single_d2_{profile.name}_bridge',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 2, '22': 0, '23': 2, '24': 0, '25': 0}},
            ),
            priority=40,
        )
    if term:
        return SpecialChainPattern(
            chain_type='side',
            composition=[22, 23, 23, 11],
            origin_type=f'E{int(profile.su_type)}d2',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + term[:1]],
            consumed_ids=set(consumed) | {int(node_id(x)) for x in term[:1]},
            metadata=_pattern_metadata(
                f'single_d2_{profile.name}_terminal',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 1, '22': 1, '23': 2, '24': 0, '25': 0}},
            ),
            priority=41,
        )
    if ali:
        return SpecialChainPattern(
            chain_type='bridge',
            composition=[11, 23, 23, 23, 11],
            origin_type=f'D{int(profile.su_type)}d2',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + ali[:1]],
            consumed_ids=set(consumed) | {int(node_id(x)) for x in ali[:1]},
            metadata=_pattern_metadata(
                f'single_d2_{profile.name}_aliphatic',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 2, '22': 0, '23': 3, '24': 0, '25': 0}},
            ),
            priority=42,
        )
    return None


def _build_double_d2_bridge_pattern(node: Any,
                                    profile: SpecialCarbonProfile,
                                    lookup: Dict[int, Any]) -> Optional[SpecialChainPattern]:
    if not bool(profile.allow_double_d2):
        return None
    if int(node_degree(node) or 0) != 2 or not bool(is_double_special(node)):
        return None
    nbs = neighbor_nodes(node, lookup)
    connector_recs = _connector_records(node, profile, lookup)
    connectors = [rec['connector'] for rec in connector_recs]
    if not connectors:
        return None
    primary_eps = [_pick_primary_endpoint(rec.get('endpoints', []), profile) for rec in connector_recs]
    connector_endpoints = [ep for ep in primary_eps if ep is not None]
    aro_eps = [nb for nb in connector_endpoints if _endpoint_signature(nb, profile) == 'aromatic']
    single_eps = [nb for nb in connector_endpoints if _endpoint_signature(nb, profile) == 'special_single']
    direct_term = [nb for nb in nbs if int(node_su(nb)) in set(int(x) for x in profile.terminal_anchor_su)]
    meta_base = {
        'connector_ids': [int(node_id(x)) for x in connectors],
        'connector_types': [int(node_su(x)) for x in connectors],
        'connector_endpoint_su': [int(node_su(x)) for x in connector_endpoints],
        'connector_endpoint_signatures': [_endpoint_signature(x, profile) for x in connector_endpoints],
    }
    consumed = {int(node_id(node))} | {int(node_id(x)) for x in connectors + direct_term + connector_endpoints}
    if direct_term and aro_eps:
        return SpecialChainPattern(
            chain_type='side',
            composition=[11, 23, 23, 22],
            origin_type=f'C+{int(profile.su_type)}',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + direct_term + aro_eps[:1]],
            consumed_ids=set(consumed),
            metadata=_pattern_metadata(
                f'double_d2_{profile.name}_to_11232322',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 1, '22': 1, '23': 2, '24': 0, '25': 0}},
            ),
            priority=20,
        )
    if len(aro_eps) >= 2:
        return SpecialChainPattern(
            chain_type='bridge',
            composition=[11, 23, 23, 23, 11],
            origin_type=f'B{int(profile.su_type)}+',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + aro_eps[:2]],
            consumed_ids=set(consumed),
            metadata=_pattern_metadata(
                f'double_d2_{profile.name}_to_1123232311',
                node,
                profile,
                extra={**meta_base, 'resource_requirements_override': {'11': 2, '22': 0, '23': 3, '24': 0, '25': 0}},
            ),
            priority=21,
        )
    if len(aro_eps) >= 1 and single_eps:
        return SpecialChainPattern(
            chain_type='bridge',
            composition=[11, 23, 23, 23, 23, 11],
            origin_type=f'B{int(profile.su_type)}++',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + single_eps],
            consumed_ids=set(consumed),
            metadata=_pattern_metadata(
                f'double_d2_{profile.name}_to_112323232311',
                node,
                profile,
                extra={
                    **meta_base,
                    'resource_requirements_override': {'11': 2, '22': 0, '23': 4, '24': 0, '25': 0},
                    'partner_special_ids': [int(node_id(x)) for x in single_eps[:1]],
                },
            ),
            priority=22,
        )
    if len(single_eps) >= 2:
        return SpecialChainPattern(
            chain_type='bridge',
            composition=[11, 23, 23, 23, 23, 23, 11],
            origin_type=f'D{int(profile.su_type)}+',
            phase='closed',
            source_ids=[int(node_id(node))] + [int(node_id(x)) for x in connectors + single_eps],
            consumed_ids=set(consumed),
            metadata=_pattern_metadata(
                f'double_d2_{profile.name}_to_11232323232311',
                node,
                profile,
                extra={
                    **meta_base,
                    'resource_requirements_override': {'11': 2, '22': 0, '23': 5, '24': 0, '25': 0},
                    'partner_special_ids': [int(node_id(x)) for x in single_eps[:2]],
                },
            ),
            priority=23,
        )
    return None


def _pattern_sort_key(pat: SpecialChainPattern) -> Tuple[int, int, int, str]:
    return (
        int(getattr(pat, 'priority', 100)),
        -int(len(list(getattr(pat, 'source_ids', []) or []))),
        int(len(set(getattr(pat, 'consumed_ids', set()) or set()))),
        -int(len(list(getattr(pat, 'composition', []) or []))),
        str(getattr(pat, 'origin_type', '')),
    )


def build_special_patterns(nodes: Sequence[Any],
                           lookup: Dict[int, Any]) -> Tuple[List[SpecialChainPattern], Set[int]]:
    candidates: List[SpecialChainPattern] = []
    ordered = sorted(list(nodes or []), key=lambda n: int(node_id(n)))
    for node in ordered:
        profile = _get_profile(node)
        if profile is None:
            continue
        if int(node_degree(node) or 0) == 1:
            pat = _build_single_d1_connector_pattern(node, profile, lookup)
            if pat is not None:
                candidates.append(pat)
        elif int(node_degree(node) or 0) == 2:
            pat = _build_single_d2_bridge_pattern(node, profile, lookup)
            if pat is not None:
                candidates.append(pat)
            pat2 = _build_double_d2_bridge_pattern(node, profile, lookup)
            if pat2 is not None:
                candidates.append(pat2)
        elif int(node_degree(node) or 0) == 3:
            continue

    patterns: List[SpecialChainPattern] = []
    consumed_ids: Set[int] = set()
    for pat in sorted(candidates, key=_pattern_sort_key):
        cids = set(int(x) for x in list(getattr(pat, 'consumed_ids', set()) or set()))
        if cids & consumed_ids:
            continue
        patterns.append(pat)
        consumed_ids.update(cids)
    return patterns, consumed_ids


def split_special_patterns(patterns: Sequence[SpecialChainPattern]) -> Tuple[List[SpecialChainPattern], List[SpecialChainPattern]]:
    closed: List[SpecialChainPattern] = []
    branch: List[SpecialChainPattern] = []
    for pat in list(patterns or []):
        if str(getattr(pat, 'phase', 'closed')) == 'branch':
            branch.append(pat)
        else:
            closed.append(pat)
    return closed, branch


def branch24_priority(node: Any) -> Tuple[int, int]:
    profile = getattr(node, 'branch24_profile', None)
    source_kind = str((profile or {}).get('source_kind', ''))
    fixed_usage = str((profile or {}).get('fixed_usage', ''))
    abcd_type = str((profile or {}).get('abcd_type', ''))
    if source_kind.startswith(('su19d3', 'su20d3', 'su21d3')):
        if abcd_type in {'24_A', '24_B'}:
            return (0, int(node_id(node)))
        if fixed_usage in {'ring', 'branch'}:
            return (1, int(node_id(node)))
        return (2, int(node_id(node)))
    return (3, int(node_id(node)))
