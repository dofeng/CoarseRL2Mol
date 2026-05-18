import sys
import io
import ast
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass, field
from collections import Counter
from contextlib import redirect_stdout

from .RL_init import _parse_template_key
from .allocator_patterns import (
    _has_invalid_capped_su19_path,
    branch24_priority,
    build_special_patterns,
    is_double_special,
    node_degree,
    split_special_patterns,
)

AROMATIC_SET = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
TO_11 = {5, 6, 7, 8, 9, 11}
TO_23 = {0, 2, 3, 15, 17, 19, 20, 21, 23, 27, 29, 31}
TO_24 = {14, 24}
TO_22 = {1, 4, 16, 18, 22, 28, 32}
BRIDGE_SU = {0, 2, 3, 27, 29, 31}
TERMINAL_SU = {1, 4, 28}
BRANCH_TERMINAL_SU = {1, 22, 28, 32}
SPECIAL_CONNECTOR_SU = {19: {29, 31}, 20: {27}, 21: set()}
SPECIAL_FIXED_BRANCH_LABEL = {19: 'OS', 20: 'N', 21: 'X'}
MAX_23_PER_CHAIN = 6
RAW_NATIVE_KIND = {
    '11': {11},
    '22': {22},
    '23': {23},
    '24': {24},
    '25': {25},
}


def _resource_count(values: List[int], allowed: set[int]) -> int:
    return int(sum(1 for x in values if int(x) in allowed))


def _effective_resource_kind_from_parts(su_type: int,
                                        target_degree: Optional[int]) -> Optional[str]:
    su_i = int(su_type)
    deg_i = int(target_degree) if target_degree is not None else None

    if su_i in {5, 6, 7, 8, 9, 11}:
        return '11'
    if su_i in {0, 2, 3, 15, 17, 23, 27, 29, 31}:
        return '23'
    if su_i in {1, 4, 16, 18, 22, 28, 32}:
        return '22'
    if su_i in {14, 24}:
        return '24'
    if su_i == 25:
        return '25'
    if su_i in {19, 20}:
        if deg_i == 1:
            return '22'
        if deg_i == 3:
            return '24'
        return '23'
    if su_i == 21:
        if deg_i == 3:
            return '24'
        return '23'
    return None


def _bridge_endpoint_class_from_parts(su_type: int,
                                      target_degree: Optional[int]) -> str:
    su_i = int(su_type)
    deg_i = int(target_degree) if target_degree is not None else None

    if su_i in AROMATIC_SET:
        return 'aromatic'
    if su_i in {1, 4, 16, 18, 22, 28, 32}:
        return 'terminal'
    if su_i in {0, 2, 3, 14, 15, 17, 23, 24, 25, 27, 29, 31}:
        return 'aliphatic'
    if su_i in {19, 20}:
        if deg_i == 1:
            return 'terminal'
        return 'aliphatic'
    if su_i == 21:
        return 'aliphatic'
    return 'aliphatic'


def _resource_kind_for_node(node: Any) -> Optional[str]:
    return _effective_resource_kind_from_parts(
        int(getattr(node, 'su_type', -1)),
        getattr(node, 'target_degree', None),
    )


def _endpoint_class_for_node(node: Any) -> str:
    return _bridge_endpoint_class_from_parts(
        int(getattr(node, 'su_type', -1)),
        getattr(node, 'target_degree', None),
    )


def _is_native_resource_kind(su_type: int, kind: Optional[str]) -> bool:
    if kind is None:
        return False
    return int(su_type) in set(int(x) for x in RAW_NATIVE_KIND.get(str(kind), set()))


def _su3_role_from_neighbors(neighbor_su: Sequence[int]) -> Optional[str]:
    vals = [int(x) for x in list(neighbor_su or [])]
    if not vals:
        return None
    n9 = sum(1 for x in vals if int(x) == 9)
    has_non9 = any(int(x) != 9 for x in vals)
    if int(n9) == 1 and bool(has_non9):
        return 'b_like'
    if int(n9) == 0:
        return 'd_like'
    return 'mixed'

# ==================== Data Structures ====================

@dataclass
class SUNode:
    global_id: int
    su_type: int
    hop1: Tuple[int, ...]
    hop2: Tuple[int, ...]
    target_degree: Optional[int] = None
    target_fixed_anchor_count: Optional[int] = None
    anchor_partition: Optional[str] = None
    anchor_mode: Optional[str] = None
    hop1_ids: Tuple[int, ...] = field(default_factory=tuple)
    effective_kind: Optional[str] = None
    endpoint_class: Optional[str] = None
    su3_role: Optional[str] = None
    branch24_profile: Optional[Dict[str, Any]] = None

@dataclass
class Branch24Profile:
    node_id: int
    su_type: int
    abcd_type: str
    source_kind: str
    fixed_usage: str = 'none'        # none / branch / ring / invalid
    fixed_label: Optional[str] = None
    fixed_connector_ids: List[int] = field(default_factory=list)
    fixed_connector_types: List[int] = field(default_factory=list)
    partner_special_ids: List[int] = field(default_factory=list)
    partner_special_degrees: List[int] = field(default_factory=list)
    fixed_path_kind: Optional[str] = None
    is_double_special: bool = False
    allowed_tail_modes: List[str] = field(default_factory=list)
    allowed_slots: List[str] = field(default_factory=list)
    can_be_fixed_fused_c_bridgehead: bool = True
    valid: bool = True
    invalid_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': int(self.node_id),
            'su_type': int(self.su_type),
            'abcd_type': str(self.abcd_type),
            'source_kind': str(self.source_kind),
            'fixed_usage': str(self.fixed_usage),
            'fixed_label': self.fixed_label,
            'fixed_connector_ids': [int(x) for x in self.fixed_connector_ids],
            'fixed_connector_types': [int(x) for x in self.fixed_connector_types],
            'partner_special_ids': [int(x) for x in self.partner_special_ids],
            'partner_special_degrees': [int(x) for x in self.partner_special_degrees],
            'fixed_path_kind': self.fixed_path_kind,
            'is_double_special': bool(self.is_double_special),
            'allowed_tail_modes': list(self.allowed_tail_modes),
            'allowed_slots': list(self.allowed_slots),
            'can_be_fixed_fused_c_bridgehead': bool(self.can_be_fixed_fused_c_bridgehead),
            'valid': bool(self.valid),
            'invalid_reason': self.invalid_reason,
        }

@dataclass
class ChainSpec:
    chain_type: str           # 'bridge' or 'side'
    composition: List[int]    # e.g. [11,23,23,11]
    origin_type: str          # 'A','B','C',... 'extra'
    source_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_11: int = 0
    n_23: int = 0
    n_22: int = 0
    n_24: int = 0
    n_25: int = 0

    def __post_init__(self):
        self.n_11 = _resource_count(self.composition, TO_11)
        self.n_23 = _resource_count(self.composition, TO_23)
        self.n_22 = _resource_count(self.composition, TO_22)
        self.n_24 = _resource_count(self.composition, TO_24)
        self.n_25 = _resource_count(self.composition, {25})

@dataclass
class AllocationResult:
    bridge_chains: List[ChainSpec] = field(default_factory=list)
    side_chains: List[ChainSpec] = field(default_factory=list)
    branch_chains: List[ChainSpec] = field(default_factory=list)
    total_11: int = 0
    total_23: int = 0
    total_22: int = 0
    total_24: int = 0
    total_25: int = 0
    consumed_11: int = 0
    consumed_23: int = 0
    consumed_22: int = 0
    consumed_24: int = 0
    consumed_25: int = 0
    remaining_11: int = 0
    remaining_23: int = 0
    remaining_22: int = 0
    remaining_24: int = 0
    remaining_25: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    unallocated_bridge: int = 0
    unallocated_branch: int = 0
    required_extra_22: int = 0
    required_extra_11: int = 0
    required_extra_23: int = 0
    unsupported_special_count: int = 0
    unsupported_special_blocked_count: int = 0
    unsupported_special_nodes: List[int] = field(default_factory=list)
    unsupported_special_details: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_special_reasons: Dict[str, int] = field(default_factory=dict)
    
    extra_11_23_11_count: int = 0   
    extra_long_23_chains: int = 0   
    extra_11_22_count: int = 0
    extra_short_bridge_count: int = 0
    extra_side_to_22_count: int = 0
    extra_bridge_avg_23: float = 0.0
    post_flex_23_requested: int = 0
    post_flex_23_applied: int = 0
    post_flex_23_to_side: int = 0
    post_flex_23_to_branch: int = 0
    native_total_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_total_by_kind: Dict[str, int] = field(default_factory=dict)
    native_consumed_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_consumed_by_kind: Dict[str, int] = field(default_factory=dict)
    native_remaining_by_kind: Dict[str, int] = field(default_factory=dict)
    proxy_remaining_by_kind: Dict[str, int] = field(default_factory=dict)

# ==================== Helper ====================


def _chain_required_resource_counts(chain: ChainSpec) -> Dict[str, int]:
    meta = dict(getattr(chain, 'metadata', {}) or {})
    override = dict(meta.get('resource_requirements_override', {}) or {})
    if override:
        return {
            '11': int(override.get('11', 0)),
            '22': int(override.get('22', 0)),
            '23': int(override.get('23', 0)),
            '24': int(override.get('24', 0)),
            '25': int(override.get('25', 0)),
        }
    return {
        '11': int(getattr(chain, 'n_11', 0)),
        '22': int(getattr(chain, 'n_22', 0)),
        '23': int(getattr(chain, 'n_23', 0)),
        '24': int(getattr(chain, 'n_24', 0)),
        '25': int(getattr(chain, 'n_25', 0)),
    }


def _sum_chain_required_resource_counts(chains: Sequence[ChainSpec]) -> Dict[str, int]:
    total = {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0}
    for chain in list(chains or []):
        req = _chain_required_resource_counts(chain)
        for kind in total.keys():
            total[kind] += int(req.get(kind, 0))
    return total


def chain_spec_counts_match(chain_spec: ChainSpec, candidate_sus: List[int]) -> bool:
    target = _chain_required_resource_counts(chain_spec)
    got = {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0}
    for su in list(candidate_sus or []):
        su_i = int(su)
        if su_i in TO_11:
            got['11'] += 1
        if su_i in TO_22:
            got['22'] += 1
        if su_i in TO_23:
            got['23'] += 1
        if su_i in TO_24:
            got['24'] += 1
        if su_i == 25:
            got['25'] += 1
    return all(int(got[k]) == int(target.get(k, 0)) for k in ('11', '22', '23', '24', '25'))


# ==================== FlexAllocator ====================

class FlexAllocator:

    def __init__(self, csv_path: Optional[str] = None, su_counts: Optional[Dict[int, int]] = None, nodes: Optional[List] = None):
        self.csv_path = csv_path
        self._nodes: List[SUNode] = []
        self._node_lookup: Dict[int, SUNode] = {}
        self._type_lists: Dict[str, List[SUNode]] = {}
        self._result = AllocationResult()
        self._su_counts = su_counts.copy() if su_counts is not None else {}
        self._input_nodes = nodes
        self._special_bridge_patterns = []
        self._special_branch_patterns = []
        self._consumed_special_ids: set[int] = set()
        self._consumed_special_closed_ids: set[int] = set()
        self._consumed_special_branch_ids: set[int] = set()
        self._unsupported_special_ids: set[int] = set()
        self._unsupported_special_blocked_ids: set[int] = set()
        self._unsupported_special_details: List[Dict[str, Any]] = []
        self._branch24_profiles: Dict[int, Branch24Profile] = {}

    def _reset_runtime_state(self):
        self._nodes = []
        self._node_lookup = {}
        self._type_lists = {}
        self._result = AllocationResult()
        self._special_bridge_patterns = []
        self._special_branch_patterns = []
        self._consumed_special_ids = set()
        self._consumed_special_closed_ids = set()
        self._consumed_special_branch_ids = set()
        self._unsupported_special_ids = set()
        self._unsupported_special_blocked_ids = set()
        self._unsupported_special_details = []
        self._branch24_profiles = {}

    @staticmethod
    def _init_chain_resource_usage(chain: ChainSpec) -> Dict[str, Any]:
        meta = dict(chain.metadata or {})
        if 'resource_usage' not in meta or not isinstance(meta.get('resource_usage'), dict):
            meta['resource_usage'] = {
                kind: {'native': [], 'proxy': []}
                for kind in ('11', '22', '23', '24', '25')
            }
        if 'resource_usage_counts' not in meta or not isinstance(meta.get('resource_usage_counts'), dict):
            meta['resource_usage_counts'] = {
                kind: {'native': 0, 'proxy': 0}
                for kind in ('11', '22', '23', '24', '25')
            }
        chain.metadata = meta
        return meta

    def _build_resource_inventory(self) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
        inventory = {
            kind: {'native': [], 'proxy': []}
            for kind in ('11', '22', '23', '24', '25')
        }
        for node in self._nodes:
            kind = _resource_kind_for_node(node)
            if kind is None:
                continue
            rec = {
                'id': int(node.global_id),
                'su_type': int(node.su_type),
                'target_degree': int(node.target_degree) if getattr(node, 'target_degree', None) is not None else -1,
                'effective_kind': str(kind),
                'endpoint_class': str(getattr(node, 'endpoint_class', _endpoint_class_for_node(node))),
            }
            bucket = 'native' if _is_native_resource_kind(int(node.su_type), str(kind)) else 'proxy'
            inventory[str(kind)][bucket].append(rec)

        for kind in inventory:
            inventory[kind]['native'].sort(key=lambda rec: (int(rec['su_type']), int(rec['id'])))
            inventory[kind]['proxy'].sort(key=lambda rec: (int(rec['su_type']), int(rec['id'])))
        return inventory

    @staticmethod
    def _take_specific_resource(
        inventory: Dict[str, Dict[str, List[Dict[str, int]]]],
        kind: str,
        gid: int,
    ) -> Optional[Tuple[str, Dict[str, int]]]:
        for bucket in ('native', 'proxy'):
            items = inventory.get(str(kind), {}).get(bucket, [])
            for idx, rec in enumerate(items):
                if int(rec.get('id', -1)) != int(gid):
                    continue
                return bucket, items.pop(idx)
        return None

    @staticmethod
    def _take_any_resource(
        inventory: Dict[str, Dict[str, List[Dict[str, int]]]],
        kind: str,
    ) -> Optional[Tuple[str, Dict[str, int]]]:
        # Prefer proxy contributors first so native block_c-adjustable units are
        # only consumed when the abstract allocation really needs them.
        for bucket in ('proxy', 'native'):
            items = inventory.get(str(kind), {}).get(bucket, [])
            if items:
                return bucket, items.pop(0)
        return None

    def _record_chain_resource(
        self,
        chain: ChainSpec,
        kind: str,
        bucket: str,
        rec: Dict[str, int],
    ) -> None:
        meta = self._init_chain_resource_usage(chain)
        usage = meta['resource_usage'][str(kind)][str(bucket)]
        if any(int(item.get('id', -1)) == int(rec.get('id', -1)) for item in usage):
            return
        usage.append({'id': int(rec['id']), 'su_type': int(rec['su_type'])})
        meta['resource_usage_counts'][str(kind)][str(bucket)] = int(len(usage))
        chain.metadata = meta

    @staticmethod
    def _collect_chain_trace_roles(chain: ChainSpec) -> Dict[int, List[str]]:
        trace_roles: Dict[int, List[str]] = {}
        src_ids = [int(x) for x in list(getattr(chain, 'source_ids', []) or [])]
        for gid in src_ids:
            trace_roles.setdefault(int(gid), []).append('source')
        meta = dict(getattr(chain, 'metadata', {}) or {})
        for key, value in meta.items():
            if not str(key).endswith('_ids'):
                continue
            if not isinstance(value, (list, tuple, set)):
                continue
            role = str(key)
            for raw_id in list(value or []):
                try:
                    gid = int(raw_id)
                except Exception:
                    continue
                roles = trace_roles.setdefault(int(gid), [])
                if role not in roles:
                    roles.append(role)
        return trace_roles

    @staticmethod
    def _collect_chain_trace_ids(chain: ChainSpec) -> List[int]:
        trace_roles = FlexAllocator._collect_chain_trace_roles(chain)
        return sorted(int(gid) for gid in trace_roles.keys())

    def _annotate_chain_resource_usage(self, chains: List[ChainSpec]) -> None:
        inventory = self._build_resource_inventory()
        native_total = {kind: int(len(inventory[kind]['native'])) for kind in inventory}
        proxy_total = {kind: int(len(inventory[kind]['proxy'])) for kind in inventory}

        for chain in list(chains or []):
            meta = self._init_chain_resource_usage(chain)
            required = _chain_required_resource_counts(chain)

            trace_ids = self._collect_chain_trace_ids(chain)
            for gid in list(trace_ids or []):
                node = self._node_lookup.get(int(gid))
                if node is None:
                    continue
                kind = _resource_kind_for_node(node)
                if kind is None:
                    continue
                taken = self._take_specific_resource(inventory, str(kind), int(gid))
                if taken is None:
                    continue
                bucket, rec = taken
                self._record_chain_resource(chain, str(kind), str(bucket), rec)

            for kind, need in required.items():
                current = meta['resource_usage_counts'][str(kind)]
                allocated = int(current.get('native', 0)) + int(current.get('proxy', 0))
                while allocated < int(need):
                    taken = self._take_any_resource(inventory, str(kind))
                    if taken is None:
                        break
                    bucket, rec = taken
                    self._record_chain_resource(chain, str(kind), str(bucket), rec)
                    allocated += 1

            usage_counts = meta['resource_usage_counts']
            meta['native_consumed_counts'] = {
                kind: int((usage_counts.get(kind, {}) or {}).get('native', 0))
                for kind in ('11', '22', '23', '24', '25')
                if int((usage_counts.get(kind, {}) or {}).get('native', 0)) > 0
            }
            meta['proxy_consumed_counts'] = {
                kind: int((usage_counts.get(kind, {}) or {}).get('proxy', 0))
                for kind in ('11', '22', '23', '24', '25')
                if int((usage_counts.get(kind, {}) or {}).get('proxy', 0)) > 0
            }
            meta['consumed_resource_ids'] = {
                kind: [int(item['id']) for bucket in ('native', 'proxy') for item in meta['resource_usage'][kind][bucket]]
                for kind in ('11', '22', '23', '24', '25')
                if any(meta['resource_usage'][kind][bucket] for bucket in ('native', 'proxy'))
            }
            chain.metadata = meta

        native_remaining = {kind: int(len(inventory[kind]['native'])) for kind in inventory}
        proxy_remaining = {kind: int(len(inventory[kind]['proxy'])) for kind in inventory}
        native_consumed = {
            kind: int(native_total[kind] - native_remaining[kind])
            for kind in native_total
        }
        proxy_consumed = {
            kind: int(proxy_total[kind] - proxy_remaining[kind])
            for kind in proxy_total
        }

        self._result.native_total_by_kind = dict(native_total)
        self._result.proxy_total_by_kind = dict(proxy_total)
        self._result.native_consumed_by_kind = dict(native_consumed)
        self._result.proxy_consumed_by_kind = dict(proxy_consumed)
        self._result.native_remaining_by_kind = dict(native_remaining)
        self._result.proxy_remaining_by_kind = dict(proxy_remaining)

    @staticmethod
    def _has_resources(avail_11: int,
                       avail_23: int,
                       avail_22: int,
                       need_11: int = 0,
                       need_23: int = 0,
                       need_22: int = 0) -> bool:
        return (
            int(avail_11) >= int(need_11) and
            int(avail_23) >= int(need_23) and
            int(avail_22) >= int(need_22)
        )

    @staticmethod
    def _refresh_chain_counts(chain: ChainSpec):
        chain.n_11 = _resource_count(chain.composition, TO_11)
        chain.n_23 = _resource_count(chain.composition, TO_23)
        chain.n_22 = _resource_count(chain.composition, TO_22)
        chain.n_24 = _resource_count(chain.composition, TO_24)
        chain.n_25 = _resource_count(chain.composition, {25})

    @staticmethod
    def _bump_chain_resource_override(chain: ChainSpec, kind: str, delta: int = 1) -> None:
        meta = dict(getattr(chain, 'metadata', {}) or {})
        override = dict(meta.get('resource_requirements_override', {}) or {})
        if not override:
            return
        key = str(kind)
        override[key] = int(override.get(key, 0)) + int(delta)
        meta['resource_requirements_override'] = override
        chain.metadata = meta

    @staticmethod
    def _side_branch_insert_index(chain: ChainSpec) -> int:
        meta = chain.metadata or {}
        branch_22 = int(meta.get('branch_22_count', 0))
        extra_22 = int(meta.get('extra_22_count', 0))
        idx = len(chain.composition) - (branch_22 + extra_22)
        return max(1, idx)

    @staticmethod
    def _normalize_branch_tail_lengths(chain: ChainSpec) -> Dict[str, int]:
        meta = dict(chain.metadata or {})
        tail_lengths = {
            str(k): int(v)
            for k, v in (meta.get('branch_tail_lengths', {}) or {}).items()
            if int(v) > 0
        }
        if tail_lengths:
            meta['branch_tail_lengths'] = tail_lengths
            chain.metadata = meta
        return tail_lengths

    @staticmethod
    def _pick_shortest_branch_slot(chain: ChainSpec) -> Optional[str]:
        tail_lengths = FlexAllocator._normalize_branch_tail_lengths(chain)
        if not tail_lengths:
            return None
        return min(tail_lengths.keys(), key=lambda key: (tail_lengths[key], key))

    @staticmethod
    def _can_expand_with_one_23(chain: ChainSpec) -> bool:
        chain_type = str(getattr(chain, 'chain_type', '') or '')
        if chain_type in ('branch_side', 'branch_bridge', 'side', 'bridge'):
            return True
        if chain_type in ('vertical_ring', 'side_ring', 'fused_side_ring'):
            return FlexAllocator._pick_shortest_branch_slot(chain) is not None
        return False

    @staticmethod
    def _add_one_23_to_chain(chain: ChainSpec, branch_slot: Optional[str] = None) -> bool:
        meta = dict(chain.metadata or {})

        if chain.chain_type in ('branch_side', 'branch_bridge'):
            meta['branch_23_count'] = int(meta.get('branch_23_count', 0)) + 1
            insert_idx = FlexAllocator._side_branch_insert_index(chain)
            chain.composition.insert(insert_idx, 23)
            chain.metadata = meta
            FlexAllocator._bump_chain_resource_override(chain, '23', 1)
            FlexAllocator._refresh_chain_counts(chain)
            return True

        if chain.chain_type in ('vertical_ring', 'side_ring', 'fused_side_ring'):
            tail_lengths = {
                str(k): int(v)
                for k, v in (meta.get('branch_tail_lengths', {}) or {}).items()
                if int(v) > 0
            }
            if not tail_lengths:
                return False
            slot = branch_slot if branch_slot in tail_lengths else min(
                tail_lengths.keys(), key=lambda key: (tail_lengths[key], key)
            )
            tail_lengths[slot] += 1
            meta['branch_tail_lengths'] = tail_lengths
            chain.composition.append(23)
            chain.metadata = meta
            FlexAllocator._bump_chain_resource_override(chain, '23', 1)
            FlexAllocator._refresh_chain_counts(chain)
            return True

        if chain.chain_type in ('side', 'bridge'):
            chain.composition.insert(max(1, len(chain.composition) - 1), 23)
            FlexAllocator._bump_chain_resource_override(chain, '23', 1)
            FlexAllocator._refresh_chain_counts(chain)
            return True

        return False

    def redistribute_remaining_flex_23(self, excess_23: int) -> Dict[str, int]:
        """Re-inject all leftover flex-body SU23 into side chains and branch structures.

        The redistribution is intentionally category-balanced: when both side and
        branch recipients exist, we seed both sides first and then alternate.
        """
        summary = {
            'requested_23': int(max(0, excess_23)),
            'applied_23': 0,
            'to_side': 0,
            'to_branch': 0,
            'remaining_23': 0,
        }
        if excess_23 <= 0:
            return summary

        side_recipients = [
            ch for ch in self._result.side_chains
            if ch.chain_type in ('side', 'branch_side')
        ]
        branch_recipients = [
            ch for ch in self._result.branch_chains
            if FlexAllocator._pick_shortest_branch_slot(ch) is not None
        ]

        def pick_side_target() -> Optional[ChainSpec]:
            if not side_recipients:
                return None
            return min(
                side_recipients,
                key=lambda ch: (
                    ch.n_23,
                    1 if ch.chain_type == 'branch_side' else 0,
                    len(ch.composition),
                ),
            )

        def pick_branch_target() -> Optional[Tuple[ChainSpec, str]]:
            best = None
            for ch in branch_recipients:
                slot = FlexAllocator._pick_shortest_branch_slot(ch)
                if slot is None:
                    continue
                lengths = ch.metadata.get('branch_tail_lengths', {}) if ch.metadata else {}
                slot_len = int(lengths.get(slot, 0))
                key = (slot_len, ch.n_23, slot)
                if best is None or key < best[0]:
                    best = (key, ch, slot)
            if best is None:
                return None
            return best[1], best[2]

        # Seed both categories first when possible.
        if excess_23 >= 2:
            seeded = False
            side_target = pick_side_target()
            if side_target is not None and FlexAllocator._add_one_23_to_chain(side_target):
                summary['applied_23'] += 1
                summary['to_side'] += 1
                excess_23 -= 1
                seeded = True
            branch_target = pick_branch_target()
            if branch_target is not None:
                branch_chain, branch_slot = branch_target
                if FlexAllocator._add_one_23_to_chain(branch_chain, branch_slot):
                    summary['applied_23'] += 1
                    summary['to_branch'] += 1
                    excess_23 -= 1
                    seeded = True
            if not seeded and side_target is None and branch_target is None:
                summary['remaining_23'] = int(excess_23)
                return summary

        prefer_side = summary['to_side'] <= summary['to_branch']
        while excess_23 > 0:
            used = False
            order = ('side', 'branch') if prefer_side else ('branch', 'side')
            for category in order:
                if category == 'side':
                    target = pick_side_target()
                    if target is None:
                        continue
                    if FlexAllocator._add_one_23_to_chain(target):
                        summary['applied_23'] += 1
                        summary['to_side'] += 1
                        excess_23 -= 1
                        used = True
                        break
                else:
                    target = pick_branch_target()
                    if target is None:
                        continue
                    branch_chain, branch_slot = target
                    if FlexAllocator._add_one_23_to_chain(branch_chain, branch_slot):
                        summary['applied_23'] += 1
                        summary['to_branch'] += 1
                        excess_23 -= 1
                        used = True
                        break
            if not used:
                break
            prefer_side = summary['to_side'] <= summary['to_branch']

        summary['remaining_23'] = int(excess_23)
        self._result.post_flex_23_requested += int(summary['requested_23'])
        self._result.post_flex_23_applied += int(summary['applied_23'])
        self._result.post_flex_23_to_side += int(summary['to_side'])
        self._result.post_flex_23_to_branch += int(summary['to_branch'])
        return summary

    # ---------- Phase 1: Parse Input ----------
    def _parse_input(self):
        if self._input_nodes is not None:
            # Parse from Layer1-2-3 _NodeV3 objects
            self._su_counts = {}
            node_lookup: Dict[int, Any] = {}
            for n in self._input_nodes:
                try:
                    node_lookup[int(n.global_id)] = n
                except Exception:
                    continue

            for n in self._input_nodes:
                su_type = int(n.su_type)
                self._su_counts[su_type] = self._su_counts.get(su_type, 0) + 1
                target_degree = getattr(n, 'target_hop1_degree', None)
                try:
                    target_degree_i = int(target_degree) if target_degree is not None else None
                except Exception:
                    target_degree_i = None
                target_fixed_anchor_count = getattr(n, 'target_fixed_anchor_count', None)
                try:
                    target_fixed_anchor_count_i = (
                        max(0, int(target_fixed_anchor_count))
                        if target_fixed_anchor_count is not None else None
                    )
                except Exception:
                    target_fixed_anchor_count_i = None
                anchor_partition = getattr(n, 'special_anchor_partition', None)
                anchor_partition_s = str(anchor_partition) if anchor_partition is not None else None
                anchor_mode = getattr(n, 'special_anchor_mode', None)
                anchor_mode_s = str(anchor_mode) if anchor_mode is not None else None
                
                hop1 = []
                hop2 = []
                hop1_ids = list(getattr(n, 'hop1_ids', []) or [])
                if hop1_ids:
                    for nid in hop1_ids:
                        try:
                            nb = node_lookup.get(int(nid))
                        except Exception:
                            nb = None
                        if nb is None:
                            continue
                        try:
                            hop1.append(int(nb.su_type))
                        except Exception:
                            continue

                    center_id = int(n.global_id)
                    for nid in hop1_ids:
                        try:
                            nb = node_lookup.get(int(nid))
                        except Exception:
                            nb = None
                        if nb is None:
                            continue
                        for nid2 in list(getattr(nb, 'hop1_ids', []) or []):
                            try:
                                nid2_i = int(nid2)
                            except Exception:
                                continue
                            if nid2_i == center_id:
                                continue
                            nb2 = node_lookup.get(nid2_i)
                            if nb2 is None:
                                continue
                            try:
                                hop2.append(int(nb2.su_type))
                            except Exception:
                                continue
                else:
                    if hasattr(n, 'hop1_su') and n.hop1_su:
                        for k, v in n.hop1_su.items():
                            hop1.extend([int(k)] * int(v))
                    if hasattr(n, 'hop2_su') and n.hop2_su:
                        for k, v in n.hop2_su.items():
                            hop2.extend([int(k)] * int(v))

                su_node = SUNode(
                    int(n.global_id),
                    su_type,
                    tuple(hop1),
                    tuple(hop2),
                    target_degree=target_degree_i,
                    target_fixed_anchor_count=target_fixed_anchor_count_i,
                    anchor_partition=anchor_partition_s,
                    anchor_mode=anchor_mode_s,
                    hop1_ids=tuple(int(x) for x in hop1_ids),
                )
                su_node.effective_kind = _resource_kind_for_node(su_node)
                su_node.endpoint_class = _endpoint_class_for_node(su_node)
                if int(su_type) == 3:
                    su_node.su3_role = _su3_role_from_neighbors(hop1)
                self._nodes.append(su_node)
                self._node_lookup[int(su_node.global_id)] = su_node
            return

        # su_counts-only 模式：无节点拓扑信息，仅基于粗粒化计数进行分配评估
        if self._su_counts:
            print("  [FlexAllocator] su_counts-only mode: no node topology, classification will be empty")
            return

        if not self.csv_path:
            raise ValueError("Must provide either csv_path, nodes, or su_counts")

        df = pd.read_csv(self.csv_path)
        su_col = None
        for col in ['center_su_idx', 'su_type', 'su_idx', 'type']:
            if col in df.columns:
                su_col = col
                break
        if su_col is None:
            raise ValueError(f"Cannot find SU type column in {self.csv_path}")

        if not self._su_counts:
            self._su_counts = dict(Counter(df[su_col].values))

        if 'template_key' not in df.columns:
            if 'hop1_ms' not in df.columns and 'hop2_ms' not in df.columns:
                raise ValueError("CSV missing template_key / hop1_ms / hop2_ms columns")

        def _parse_int(value: Any) -> Optional[int]:
            if value is None or pd.isna(value):
                return None
            try:
                text = str(value).strip()
                if not text or text.lower() == 'none':
                    return None
                return int(float(text))
            except Exception:
                return None

        def _parse_str(value: Any) -> Optional[str]:
            if value is None or pd.isna(value):
                return None
            text = str(value).strip()
            if not text or text.lower() == 'none':
                return None
            return text

        def _parse_int_list(value: Any) -> Tuple[int, ...]:
            if value is None or pd.isna(value):
                return tuple()
            if isinstance(value, (list, tuple, set)):
                out = []
                for item in list(value):
                    try:
                        out.append(int(item))
                    except Exception:
                        continue
                return tuple(out)
            text = str(value).strip()
            if not text or text.lower() == 'none':
                return tuple()
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                out = []
                for item in list(parsed):
                    try:
                        out.append(int(item))
                    except Exception:
                        continue
                return tuple(out)
            nums = []
            for token in text.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    nums.append(int(token))
                except Exception:
                    continue
            return tuple(nums)

        def _parse_multiset_column(value: Any) -> List[int]:
            if value is None or pd.isna(value):
                return []
            if isinstance(value, (list, tuple, set)):
                out = []
                for item in list(value):
                    try:
                        out.append(int(item))
                    except Exception:
                        continue
                return out
            text = str(value).strip()
            if not text or text.lower() == 'none':
                return []
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                out = []
                for item in list(parsed):
                    try:
                        out.append(int(item))
                    except Exception:
                        continue
                return out
            return [int(x) for x in re.findall(r'-?\d+', text)]

        for _, row in df.iterrows():
            gid = int(row.get('global_id', row.name))
            su_type = int(row[su_col])
            hop1: List[int] = []
            hop2: List[int] = []
            if 'template_key' in df.columns and not pd.isna(row.get('template_key', None)):
                _, hop1, hop2 = _parse_template_key(str(row['template_key']))
            if not hop1 and 'hop1_ms' in df.columns:
                hop1 = _parse_multiset_column(row.get('hop1_ms'))
            if not hop2 and 'hop2_ms' in df.columns:
                hop2 = _parse_multiset_column(row.get('hop2_ms'))

            hop1_ids = tuple(_parse_int_list(row.get('hop1_ids', None)))
            target_degree = (
                _parse_int(row.get('target_hop1_degree', None))
                if 'target_hop1_degree' in df.columns else None
            )
            if target_degree is None and 'target_degree' in df.columns:
                target_degree = _parse_int(row.get('target_degree', None))
            if target_degree is None and 'actual_hop1_degree' in df.columns:
                target_degree = _parse_int(row.get('actual_hop1_degree', None))
            if target_degree is None and hop1:
                target_degree = int(len(hop1))
            if target_degree is None and hop1_ids:
                target_degree = int(len(hop1_ids))

            target_fixed_anchor_count = (
                _parse_int(row.get('target_fixed_anchor_count', None))
                if 'target_fixed_anchor_count' in df.columns else None
            )
            anchor_partition = None
            for col in ('special_anchor_partition', 'anchor_partition'):
                if col in df.columns:
                    anchor_partition = _parse_str(row.get(col, None))
                    if anchor_partition is not None:
                        break
            anchor_mode = None
            for col in ('special_anchor_mode', 'anchor_mode'):
                if col in df.columns:
                    anchor_mode = _parse_str(row.get(col, None))
                    if anchor_mode is not None:
                        break

            su_node = SUNode(
                gid,
                su_type,
                tuple(hop1),
                tuple(hop2),
                target_degree=target_degree,
                target_fixed_anchor_count=target_fixed_anchor_count,
                anchor_partition=anchor_partition,
                anchor_mode=anchor_mode,
                hop1_ids=hop1_ids,
            )
            su_node.effective_kind = _resource_kind_for_node(su_node)
            su_node.endpoint_class = _endpoint_class_for_node(su_node)
            if int(su_type) == 3:
                su_node.su3_role = _su3_role_from_neighbors(hop1)
            self._nodes.append(su_node)
            self._node_lookup[int(su_node.global_id)] = su_node

    def _record_unsupported_special(self,
                                    node: SUNode,
                                    reason: str,
                                    blocked: bool = False,
                                    extra: Optional[Dict[str, Any]] = None) -> None:
        gid = int(getattr(node, 'global_id', -1))
        if int(gid) < 0:
            return
        detail = {
            'global_id': int(gid),
            'su_type': int(getattr(node, 'su_type', -1)),
            'target_degree': (
                int(getattr(node, 'target_degree'))
                if getattr(node, 'target_degree', None) is not None else None
            ),
            'target_fixed_anchor_count': (
                int(getattr(node, 'target_fixed_anchor_count'))
                if getattr(node, 'target_fixed_anchor_count', None) is not None else None
            ),
            'anchor_partition': getattr(node, 'anchor_partition', None),
            'anchor_mode': getattr(node, 'anchor_mode', None),
            'reason': str(reason),
            'blocked': bool(blocked),
        }
        if extra:
            detail.update(dict(extra))
        if not any(int(rec.get('global_id', -999)) == int(gid) for rec in self._unsupported_special_details):
            self._unsupported_special_details.append(detail)
        self._unsupported_special_ids.add(int(gid))
        if bool(blocked):
            self._unsupported_special_blocked_ids.add(int(gid))

    def _detect_unsupported_special_topologies(self) -> None:
        for node in list(self._nodes):
            su_i = int(getattr(node, 'su_type', -1))
            if int(su_i) not in {19, 20, 21}:
                continue
            deg_i = node_degree(node)
            if deg_i is None:
                continue
            is_double = bool(is_double_special(node))
            if int(su_i) == 19 and bool(_has_invalid_capped_su19_path(node, self._node_lookup)):
                self._record_unsupported_special(
                    node,
                    reason='invalid_capped_su19_path',
                    blocked=True,
                )
                continue
            if bool(is_double) and int(deg_i) not in {2, 3}:
                self._record_unsupported_special(
                    node,
                    reason='double_special_invalid_degree',
                    blocked=True,
                )
                continue
            if int(su_i) == 21 and bool(is_double):
                self._record_unsupported_special(
                    node,
                    reason='double_su21_not_supported',
                    blocked=True,
                )

    def _populate_result_special_diagnostics(self) -> None:
        reasons: Dict[str, int] = {}
        for detail in list(self._unsupported_special_details):
            reason = str(detail.get('reason', 'unknown'))
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        self._result.unsupported_special_count = int(len(self._unsupported_special_ids))
        self._result.unsupported_special_blocked_count = int(len(self._unsupported_special_blocked_ids))
        self._result.unsupported_special_nodes = sorted(int(x) for x in set(self._unsupported_special_ids))
        self._result.unsupported_special_details = [dict(x) for x in list(self._unsupported_special_details)]
        self._result.unsupported_special_reasons = dict(sorted(reasons.items()))

    def _prepare_allocation_state(self, emit_logs: bool = False) -> Dict[str, Any]:
        self._reset_runtime_state()
        self._parse_input()
        self._detect_unsupported_special_topologies()
        all_special_patterns, _all_consumed_ids = build_special_patterns(self._nodes, self._node_lookup)
        self._special_bridge_patterns, self._special_branch_patterns = split_special_patterns(all_special_patterns)
        self._consumed_special_closed_ids = {
            int(x)
            for sp in list(self._special_bridge_patterns or [])
            for x in list(getattr(sp, 'consumed_ids', set()) or set())
        }
        self._consumed_special_branch_ids = {
            int(x)
            for sp in list(self._special_branch_patterns or [])
            for x in list(getattr(sp, 'consumed_ids', set()) or set())
        }
        self._consumed_special_ids = set(self._consumed_special_closed_ids) | set(self._consumed_special_branch_ids)
        self._convert_and_count()
        self._classify_all()
        self._classify_branch_24()
        self._classify_branch_25()
        self._populate_result_special_diagnostics()
        if bool(emit_logs):
            print(f"  Parsed {len(self._nodes)} SU nodes from CSV")
            if self._branch24_profiles:
                profile_counts = Counter(
                    (
                        str(profile.source_kind),
                        str(profile.abcd_type),
                        str(profile.fixed_usage),
                    )
                    for profile in self._branch24_profiles.values()
                )
                parts = [
                    f"{kind}/{abcd}/{fixed}={count}"
                    for (kind, abcd, fixed), count in sorted(profile_counts.items())
                ]
                print(f"  [FlexAllocator] Branch24 profiles: {', '.join(parts)}")
                self._print_branch24_profile_details(limit=None)
            if self._result.unsupported_special_count > 0:
                print(
                    "  [FlexAllocator] Unsupported special topologies: "
                    f"{self._result.unsupported_special_count}"
                )
                if self._result.unsupported_special_reasons:
                    parts = [
                        f"{str(k)}={int(v)}"
                        for k, v in dict(self._result.unsupported_special_reasons).items()
                    ]
                    print(f"    reasons: {', '.join(parts)}")
        return {
            'special_bridge_patterns': list(self._special_bridge_patterns),
            'special_branch_patterns': list(self._special_branch_patterns),
            'unsupported_special_count': int(self._result.unsupported_special_count),
            'unsupported_special_blocked_count': int(self._result.unsupported_special_blocked_count),
            'unsupported_special_nodes': list(self._result.unsupported_special_nodes),
        }

    def _annotate_chain_sources(self, chain: ChainSpec) -> ChainSpec:
        meta = dict(chain.metadata or {})
        trace_roles = self._collect_chain_trace_roles(chain)
        src_ids = sorted(int(gid) for gid in trace_roles.keys())
        src_su = []
        src_hop1 = []
        src_resource_kinds = []
        src_target_degrees = []
        src_anchor_partitions = []
        src_endpoint_classes = []
        src_anchor_modes = []
        src_su3_roles = []
        src_records = []
        converted_records = []
        for gid in src_ids:
            node = self._node_lookup.get(int(gid))
            if node is None:
                continue
            su_i = int(node.su_type)
            eff_kind = _resource_kind_for_node(node)
            hop1_vals = list(int(x) for x in tuple(node.hop1))
            target_deg = int(node.target_degree) if getattr(node, 'target_degree', None) is not None else None
            anchor_part = str(node.anchor_partition) if getattr(node, 'anchor_partition', None) is not None else None
            anchor_mode = str(getattr(node, 'anchor_mode', None)) if getattr(node, 'anchor_mode', None) is not None else None
            endpoint_class = str(getattr(node, 'endpoint_class', 'aliphatic'))
            su3_role = str(getattr(node, 'su3_role', None)) if getattr(node, 'su3_role', None) is not None else None
            roles = list(trace_roles.get(int(gid), []))
            src_su.append(int(su_i))
            src_hop1.append(list(hop1_vals))
            src_resource_kinds.append(eff_kind)
            src_target_degrees.append(
                target_deg
            )
            src_anchor_partitions.append(anchor_part)
            src_anchor_modes.append(anchor_mode)
            src_endpoint_classes.append(endpoint_class)
            src_su3_roles.append(su3_role)
            bucket_guess = 'native' if _is_native_resource_kind(int(su_i), eff_kind) else 'proxy'
            rec = {
                'id': int(gid),
                'original_su_type': int(su_i),
                'effective_kind': str(eff_kind) if eff_kind is not None else None,
                'target_degree': target_deg,
                'anchor_partition': anchor_part,
                'anchor_mode': anchor_mode,
                'endpoint_class': endpoint_class,
                'su3_role': su3_role,
                'roles': list(roles),
                'bucket_guess': str(bucket_guess),
            }
            branch24_profile = getattr(node, 'branch24_profile', None)
            if isinstance(branch24_profile, dict):
                rec['branch24_profile'] = dict(branch24_profile)
            src_records.append(dict(rec))
            if str(bucket_guess) == 'proxy' or any(str(role) != 'source' for role in roles):
                converted_records.append(dict(rec))
        meta['source_su_types'] = list(src_su)
        meta['source_hop1'] = list(src_hop1)
        meta['source_resource_kinds'] = list(src_resource_kinds)
        meta['source_effective_kinds'] = list(src_resource_kinds)
        meta['source_target_degrees'] = list(src_target_degrees)
        meta['source_anchor_partitions'] = list(src_anchor_partitions)
        meta['source_anchor_modes'] = list(src_anchor_modes)
        meta['source_endpoint_classes'] = list(src_endpoint_classes)
        meta['source_su3_roles'] = list(src_su3_roles)
        meta['trace_ids'] = list(src_ids)
        meta['trace_roles'] = {int(gid): list(trace_roles.get(int(gid), [])) for gid in src_ids}
        meta['source_node_records'] = list(src_records)
        meta['converted_source_records'] = list(converted_records)
        chain.metadata = meta
        return chain

    # ---------- Phase 2: Convert & Count ----------
    def _convert_and_count(self):
        r = self._result
        if not self._nodes and self._su_counts:
            r.total_11 = sum(self._su_counts.get(k, 0) for k in TO_11)
            r.total_23 = sum(self._su_counts.get(k, 0) for k in TO_23)
            r.total_22 = sum(self._su_counts.get(k, 0) for k in TO_22)
            r.total_24 = sum(self._su_counts.get(k, 0) for k in TO_24)
            r.total_25 = self._su_counts.get(25, 0)
            return
        r.total_11 = int(sum(1 for node in self._nodes if str(getattr(node, 'effective_kind', '')) == '11'))
        r.total_23 = int(sum(1 for node in self._nodes if str(getattr(node, 'effective_kind', '')) == '23'))
        r.total_22 = int(sum(1 for node in self._nodes if str(getattr(node, 'effective_kind', '')) == '22'))
        r.total_24 = int(sum(1 for node in self._nodes if str(getattr(node, 'effective_kind', '')) == '24'))
        r.total_25 = int(sum(1 for node in self._nodes if str(getattr(node, 'effective_kind', '')) == '25'))

    # ---------- Phase 3: Classify ----------
    def _classify_all(self):
        for t in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            self._type_lists[t] = []

        for node in self._nodes:
            if int(getattr(node, 'global_id', -1)) in set(self._consumed_special_closed_ids):
                continue
            if node.su_type in BRIDGE_SU:
                self._classify_bridge(node)
            elif node.su_type in TERMINAL_SU:
                self._classify_terminal(node)

    def _neighbor_nodes(self, node: SUNode) -> List[SUNode]:
        neighbors: List[SUNode] = []
        hop1_ids = list(getattr(node, 'hop1_ids', ()) or ())
        if hop1_ids:
            for nid in hop1_ids:
                nb = self._node_lookup.get(int(nid))
                if nb is not None:
                    neighbors.append(nb)
        if neighbors:
            return neighbors

        for su_type in list(getattr(node, 'hop1', ()) or ()):
            proxy = SUNode(int(-1), int(su_type), (), ())
            proxy.effective_kind = _resource_kind_for_node(proxy)
            proxy.endpoint_class = _endpoint_class_for_node(proxy)
            neighbors.append(proxy)
        return neighbors

    @staticmethod
    def _is_branch_terminal_node(node: SUNode) -> bool:
        su_i = int(getattr(node, 'su_type', -1))
        deg_i = getattr(node, 'target_degree', None)
        if su_i in BRANCH_TERMINAL_SU:
            return True
        if su_i in {19, 20} and deg_i is not None and int(deg_i) == 1:
            return True
        return False

    def _other_connector_endpoints(self, connector: SUNode, center_id: int) -> List[SUNode]:
        out: List[SUNode] = []
        for nid in list(getattr(connector, 'hop1_ids', ()) or ()):
            try:
                nid_i = int(nid)
            except Exception:
                continue
            if int(nid_i) == int(center_id):
                continue
            nb = self._node_lookup.get(int(nid_i))
            if nb is not None:
                out.append(nb)
        return out

    @staticmethod
    def _source_kind_for_24_like(node: SUNode) -> str:
        su_i = int(getattr(node, 'su_type', -1))
        if su_i == 24:
            return 'native_24'
        if su_i == 14:
            return 'su14_as_24'
        if su_i in {19, 20, 21}:
            deg_i = getattr(node, 'target_degree', None)
            deg_s = str(int(deg_i)) if deg_i is not None else 'x'
            return f'su{su_i}d{deg_s}'
        return f'su{su_i}_as_24'

    @staticmethod
    def _allowed_slots_for_24_type(abcd_type: str,
                                   source_kind: str,
                                   fixed_usage: str,
                                   fixed_path_kind: Optional[str] = None) -> List[str]:
        slots: List[str] = []
        if abcd_type in {'24_A', '24_B'}:
            slots.extend(['side_ab', 'fused_ab'])
        if abcd_type in {'24_C', '24_D'}:
            slots.extend(['side_cd', 'vertical_inter', 'fused_outer_cd'])
        if abcd_type == '24_A':
            slots.append('vertical_fixed_a')
        if abcd_type == '24_C' and (
            source_kind == 'native_24'
            or fixed_path_kind == 'connector_anchor_edge'
        ):
            slots.append('fused_fixed_c_bridgehead')
        if str(source_kind).startswith(('su19d3', 'su20d3', 'su21d3')):
            if fixed_path_kind != 'connector_anchor_edge':
                slots = [s for s in slots if s not in {'fused_fixed_c_bridgehead'}]
            if fixed_usage != 'branch':
                slots = [s for s in slots if s != 'vertical_fixed_a']
        return sorted(set(slots))

    def _build_branch24_profile(self, node: SUNode) -> Branch24Profile:
        neighbors = self._neighbor_nodes(node)
        has_aro = any(str(getattr(nb, 'endpoint_class', _endpoint_class_for_node(nb))) == 'aromatic' for nb in neighbors)
        has_terminal = any(self._is_branch_terminal_node(nb) for nb in neighbors)
        if has_aro and not has_terminal:
            abcd_type = '24_A'
        elif has_aro and has_terminal:
            abcd_type = '24_B'
        elif not has_aro and not has_terminal:
            abcd_type = '24_C'
        else:
            abcd_type = '24_D'

        source_kind = self._source_kind_for_24_like(node)
        valid = True
        invalid_reason = None
        su_i = int(getattr(node, 'su_type', -1))
        deg_i = getattr(node, 'target_degree', None)
        fixed_usage = 'none'
        fixed_label = None
        fixed_connector_ids: List[int] = []
        fixed_connector_types: List[int] = []
        partner_special_ids: List[int] = []
        partner_special_degrees: List[int] = []
        fixed_path_kind: Optional[str] = None
        is_double = bool(is_double_special(node)) if su_i in {19, 20, 21} else False

        if su_i == 14 and has_aro:
            valid = False
            invalid_reason = 'su14_cannot_attach_aromatic'
            abcd_type = '24_D' if has_terminal else '24_C'

        if su_i in {19, 20, 21} and deg_i is not None and int(deg_i) == 3:
            fixed_label = SPECIAL_FIXED_BRANCH_LABEL.get(int(su_i))
            connector_sus = set(int(x) for x in SPECIAL_CONNECTOR_SU.get(int(su_i), set()))
            connector_nodes = [nb for nb in neighbors if int(getattr(nb, 'su_type', -1)) in connector_sus]
            direct_fixed_nodes: List[SUNode] = []
            if su_i == 19:
                direct_fixed_nodes = [nb for nb in neighbors if int(getattr(nb, 'su_type', -1)) == 28]
            elif su_i == 21:
                direct_fixed_nodes = [nb for nb in neighbors if int(getattr(nb, 'su_type', -1)) == 32]

            for nb in connector_nodes + direct_fixed_nodes:
                fixed_connector_ids.append(int(getattr(nb, 'global_id', -1)))
                fixed_connector_types.append(int(getattr(nb, 'su_type', -1)))

            if su_i == 21:
                if not direct_fixed_nodes:
                    valid = False
                    invalid_reason = 'su21d3_requires_32_fixed_branch'
                    fixed_usage = 'invalid'
                    fixed_path_kind = 'invalid_missing_x'
                else:
                    fixed_usage = 'branch'
                    fixed_path_kind = 'direct_x'
            elif any(int(getattr(nb, 'su_type', -1)) in {0, 2} for nb in neighbors):
                fixed_usage = 'branch'
                has_branch_terminal_neighbor = any(
                    self._is_branch_terminal_node(ep)
                    for nb in neighbors
                    if int(getattr(nb, 'su_type', -1)) in {0, 2}
                    for ep in self._other_connector_endpoints(nb, int(getattr(node, 'global_id', -1)))
                )
                fixed_path_kind = 'hetero_short_branch' if bool(has_branch_terminal_neighbor) else 'hetero_long_branch'
            elif direct_fixed_nodes:
                fixed_usage = 'branch'
                fixed_path_kind = 'direct_terminal'
            elif connector_nodes:
                ring_partner_found = False
                branch_partner_found = False
                anchor_connector_found = False
                branch_partner_degrees: List[int] = []
                for conn in connector_nodes:
                    endpoints = self._other_connector_endpoints(conn, int(getattr(node, 'global_id', -1)))
                    for ep in endpoints:
                        ep_su = int(getattr(ep, 'su_type', -1))
                        ep_deg = getattr(ep, 'target_degree', None)
                        if ep_su == su_i and ep_deg is not None and int(ep_deg) == 3:
                            ring_partner_found = True
                            partner_special_ids.append(int(getattr(ep, 'global_id', -1)))
                            partner_special_degrees.append(3)
                        elif ep_su == su_i and ep_deg is not None and int(ep_deg) in {1, 2}:
                            branch_partner_found = True
                            partner_special_ids.append(int(getattr(ep, 'global_id', -1)))
                            partner_special_degrees.append(int(ep_deg))
                            branch_partner_degrees.append(int(ep_deg))
                        elif ep_su in {5, 6, 7}:
                            anchor_connector_found = True
                        elif self._is_branch_terminal_node(ep):
                            branch_partner_found = True
                if bool(anchor_connector_found) and abcd_type == '24_C':
                    fixed_usage = 'ring'
                    fixed_path_kind = 'connector_anchor_edge'
                elif bool(ring_partner_found):
                    fixed_usage = 'ring'
                    fixed_path_kind = 'connector_ring_pair'
                elif bool(branch_partner_found):
                    fixed_usage = 'branch'
                    fixed_path_kind = 'connector_d1_branch' if 1 in branch_partner_degrees else 'connector_d2_branch'
                else:
                    fixed_usage = 'branch'
                    fixed_path_kind = 'connector_unresolved_branch'
            else:
                fixed_usage = 'branch' if su_i in {19, 20} else 'none'
                fixed_path_kind = 'implicit_branch' if su_i in {19, 20} else None

        if abcd_type in {'24_A', '24_C'}:
            allowed_tail_modes = ['H_long', 'H_short']
        else:
            allowed_tail_modes = ['terminal']
        if su_i == 21:
            allowed_tail_modes = ['X']
        elif su_i == 19 and fixed_usage == 'branch':
            allowed_tail_modes = ['OS_family']
        elif su_i == 20 and fixed_usage == 'branch':
            allowed_tail_modes = ['N_family']
        elif su_i in {19, 20} and fixed_usage == 'ring' and bool(is_double):
            allowed_tail_modes = [f"{SPECIAL_FIXED_BRANCH_LABEL.get(int(su_i), 'fixed_branch')}_family"]

        can_fixed_c = bool(
            (su_i == 24 and abcd_type == '24_C')
            or (
                su_i in {19, 20}
                and abcd_type == '24_C'
                and fixed_path_kind == 'connector_anchor_edge'
            )
        )
        allowed_slots = self._allowed_slots_for_24_type(abcd_type, source_kind, fixed_usage, fixed_path_kind)
        if not bool(can_fixed_c):
            allowed_slots = [slot for slot in allowed_slots if slot != 'fused_fixed_c_bridgehead']
        if fixed_path_kind in {'connector_unresolved_branch', 'invalid_missing_x'}:
            allowed_slots = []
            if fixed_path_kind == 'connector_unresolved_branch':
                valid = False
                invalid_reason = 'unresolved_special_connector_path'

        return Branch24Profile(
            node_id=int(getattr(node, 'global_id', -1)),
            su_type=int(su_i),
            abcd_type=str(abcd_type),
            source_kind=str(source_kind),
            fixed_usage=str(fixed_usage),
            fixed_label=fixed_label,
            fixed_connector_ids=[int(x) for x in fixed_connector_ids if int(x) >= 0],
            fixed_connector_types=[int(x) for x in fixed_connector_types if int(x) >= 0],
            partner_special_ids=sorted(set(int(x) for x in partner_special_ids if int(x) >= 0)),
            partner_special_degrees=sorted(set(int(x) for x in partner_special_degrees if int(x) >= 0)),
            fixed_path_kind=fixed_path_kind,
            is_double_special=bool(is_double),
            allowed_tail_modes=list(allowed_tail_modes),
            allowed_slots=list(allowed_slots),
            can_be_fixed_fused_c_bridgehead=bool(can_fixed_c),
            valid=bool(valid),
            invalid_reason=invalid_reason,
        )

    def _classify_bridge(self, node: SUNode):
        neighbors = self._neighbor_nodes(node)
        if len(neighbors) < 2:
            return

        classes = sorted(str(getattr(nb, 'endpoint_class', _endpoint_class_for_node(nb))) for nb in neighbors[:2])
        key = tuple(classes)
        if key == ('aromatic', 'aromatic'):
            self._type_lists['A'].append(node)
        elif key == ('aliphatic', 'aromatic'):
            self._type_lists['B'].append(node)
        elif key == ('aromatic', 'terminal'):
            self._type_lists['C'].append(node)
        elif key == ('aliphatic', 'aliphatic'):
            self._type_lists['D'].append(node)
        elif key == ('aliphatic', 'terminal'):
            self._type_lists['E'].append(node)
        elif key == ('terminal', 'terminal'):
            # Invalid: 22-bridge-22, treat as warning
            print(f"  [WARN] Bridge SU {node.global_id} (type {node.su_type}) has two terminal neighbors, skip")

    def _classify_terminal(self, node: SUNode):
        neighbors = self._neighbor_nodes(node)
        if len(neighbors) < 1:
            return
        neighbor = neighbors[0]
        if str(getattr(neighbor, 'endpoint_class', _endpoint_class_for_node(neighbor))) == 'aromatic':
            self._type_lists['F'].append(node)
        else:
            self._type_lists['G'].append(node)

    # ---------- Phase 3b: Classify branch 24/25 ----------
    def _classify_branch_24(self):
        """Classify effective-24 nodes with explicit 24-like branch semantics."""
        for t in ['24_A', '24_B', '24_C', '24_D']:
            self._type_lists[t] = []
        self._branch24_profiles = {}
        for node in self._nodes:
            if int(getattr(node, 'global_id', -1)) in set(self._consumed_special_closed_ids):
                continue
            if int(getattr(node, 'global_id', -1)) in set(self._unsupported_special_blocked_ids):
                continue
            if str(getattr(node, 'effective_kind', '')) != '24':
                continue
            profile = self._build_branch24_profile(node)
            self._branch24_profiles[int(node.global_id)] = profile
            node.branch24_profile = profile.to_dict()
            if not bool(profile.valid):
                self._record_unsupported_special(
                    node,
                    reason=str(profile.invalid_reason or 'invalid_branch24_profile'),
                    blocked=False,
                    extra={'branch24_profile': profile.to_dict()},
                )
                continue
            self._type_lists[str(profile.abcd_type)].append(node)
        for t in ['24_A', '24_B', '24_C', '24_D']:
            self._type_lists[t].sort(key=branch24_priority)

    def _classify_branch_25(self):
        """Classify 25-type nodes based on aromatic endpoint presence."""
        for t in ['25_aro', '25_ali']:
            self._type_lists[t] = []
        for node in self._nodes:
            if int(getattr(node, 'global_id', -1)) in set(self._consumed_special_closed_ids):
                continue
            if int(getattr(node, 'global_id', -1)) in set(self._unsupported_special_blocked_ids):
                continue
            if node.su_type != 25:
                continue
            neighbors = self._neighbor_nodes(node)
            has_aro = any(
                str(getattr(nb, 'endpoint_class', _endpoint_class_for_node(nb))) == 'aromatic'
                for nb in neighbors
            )
            if has_aro:
                self._type_lists['25_aro'].append(node)
            else:
                self._type_lists['25_ali'].append(node)

    def _print_branch24_profile_details(self, limit: Optional[int] = None) -> None:
        profiles = sorted(
            list(self._branch24_profiles.values()),
            key=lambda p: (str(p.abcd_type), int(p.su_type), int(p.node_id)),
        )
        if not profiles:
            return
        shown = profiles if limit is None else profiles[:max(0, int(limit))]
        print("  [FlexAllocator] Branch24 profile details:")
        for profile in shown:
            node = self._node_lookup.get(int(profile.node_id))
            hop1_vals = list(getattr(node, 'hop1', ()) or ()) if node is not None else []
            print(
                "    "
                f"id={int(profile.node_id)} su={int(profile.su_type)} "
                f"hop1={hop1_vals} abcd={profile.abcd_type} "
                f"fixed={profile.fixed_usage}"
                f"{('/' + str(profile.fixed_label)) if profile.fixed_label else ''} "
                f"slots={list(profile.allowed_slots)} tails={list(profile.allowed_tail_modes)}"
            )
        if limit is not None and len(profiles) > int(limit):
            print(f"    ... {len(profiles) - int(limit)} more 24-like profiles")

    @staticmethod
    def _branch_cost(btype: str) -> Tuple[int, int]:
        """Return (n_23, n_22) for a 24-node's side branch.
        Uses user-specified resources: A/C -> -23-22 (1, 1). B/D -> -22 (0, 1)."""
        if btype in ('24_A', '24_C', '25_aro', '25_ali'):
            return (1, 1)  # -23-22
        else:  # 24_B, 24_D
            return (0, 1)  # -22

    @staticmethod
    def _is_long_tail_preferred(btype: str) -> bool:
        return btype in ('24_A', '24_C', '25_aro', '25_ali')

    @staticmethod
    def _branch_role_for_type(btype: str) -> str:
        if btype == '24_A':
            return 'bridgehead_anchor'
        if btype == '24_B':
            return 'anchor_with_terminal'
        if btype == '24_C':
            return 'aliphatic_branch'
        if btype == '24_D':
            return 'terminal_branch'
        if btype in ('25_aro', '25_ali'):
            return 'quaternary_branch'
        return 'generic_branch'

    @staticmethod
    def _tail_cost_options(btype: str) -> List[Tuple[int, int, str]]:
        if btype in ('24_A', '24_C'):
            return [(2, 1, 'raw_long'), (1, 1, 'raw_short')]
        if btype in ('25_aro', '25_ali'):
            return [(2, 1, 'raw_long'), (1, 1, 'raw_short')]
        return [(0, 1, 'raw')]

    def _consume_reserved_branch_tail(self,
                                      btype: str,
                                      remaining_E: List[SUNode],
                                      remaining_G: List[SUNode]) -> Tuple[int, int, str]:
        """Prefer reserved terminal chains for branch-side ending.
        Returns (n_23, n_22, source_tag).
        """
        prefer_long = self._is_long_tail_preferred(btype)
        if prefer_long:
            if remaining_E:
                remaining_E.pop(0)
                return 2, 1, 'E'
            if remaining_G:
                remaining_G.pop(0)
                return 1, 1, 'G'
        else:
            if remaining_G:
                remaining_G.pop(0)
                return 1, 1, 'G'
            if remaining_E:
                remaining_E.pop(0)
                return 2, 1, 'E'
        costs = self._tail_cost_options(btype)
        if costs:
            b23, b22, tag = costs[0]
            return int(b23), int(b22), str(tag)
        b23, b22 = self._branch_cost(btype)
        return int(b23), int(b22), 'raw'

    def _consume_branch_tail_for_node(self,
                                      node: SUNode,
                                      btype: str,
                                      remaining_E: List[SUNode],
                                      remaining_G: List[SUNode]) -> Tuple[int, int, str]:
        profile = getattr(node, 'branch24_profile', None)
        if isinstance(profile, dict):
            modes = set(str(x) for x in list(profile.get('allowed_tail_modes', []) or []))
            fixed_usage = str(profile.get('fixed_usage', 'none'))
            if 'X' in modes:
                return 0, 1, 'X'
            if 'OS_family' in modes or 'N_family' in modes:
                path_kind = str(profile.get('fixed_path_kind', 'fixed_branch'))
                if path_kind in {'hetero_long_branch', 'connector_d2_branch'}:
                    return 1, 1, path_kind
                return 0, 1, path_kind
            if 'terminal' in modes and not ({'H_long', 'H_short'} & modes):
                return 0, 1, 'terminal'
            if fixed_usage == 'ring' and ({'H_long', 'H_short'} & modes):
                if remaining_G:
                    remaining_G.pop(0)
                    return 1, 1, 'G'
                return 1, 1, 'raw_short'
        return self._consume_reserved_branch_tail(btype, remaining_E, remaining_G)

    def _consume_branch_tail_for_node_with_special_reuse(self,
                                                         node: SUNode,
                                                         btype: str,
                                                         remaining_E: List[SUNode],
                                                         remaining_G: List[SUNode]) -> Tuple[int, int, str]:
        profile = getattr(node, 'branch24_profile', None)
        if isinstance(profile, dict):
            fixed_usage = str(profile.get('fixed_usage', 'none'))
            modes = set(str(x) for x in list(profile.get('allowed_tail_modes', []) or []))
            if fixed_usage == 'ring' and ({'H_long', 'H_short'} & modes):
                # Type F side chains include converted d2-terminal fragments such
                # as -19d2-28 / -21d2-32, which are valid replacements for the
                # short -23-22 H tail.
                for idx, tail_node in enumerate(list(remaining_G)):
                    tail_su = int(getattr(tail_node, 'su_type', -1))
                    tail_deg = getattr(tail_node, 'target_degree', None)
                    if tail_su in {19, 21} and tail_deg is not None and int(tail_deg) == 2:
                        remaining_G.pop(idx)
                        return 1, 1, 'special_d2_terminal_tail'
        return self._consume_branch_tail_for_node(node, btype, remaining_E, remaining_G)

    @staticmethod
    def _branch24_profiles_for_items(items: Sequence[Tuple[SUNode, str]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node, role in list(items or []):
            profile = getattr(node, 'branch24_profile', None)
            if not isinstance(profile, dict):
                continue
            rec = dict(profile)
            rec['allocator_role'] = str(role)
            out.append(rec)
        return out

    @staticmethod
    def _branch24_profile(node: SUNode) -> Dict[str, Any]:
        profile = getattr(node, 'branch24_profile', None)
        return dict(profile) if isinstance(profile, dict) else {}

    @staticmethod
    def _profile_allows_slot(node: SUNode, slot: str) -> bool:
        profile = FlexAllocator._branch24_profile(node)
        return str(slot) in set(str(x) for x in list(profile.get('allowed_slots', []) or []))

    @staticmethod
    def _side_ring_comp_for_types(type1: str, type2: str) -> List[int]:
        is_ab_1 = str(type1) in ('24_A', '24_B')
        is_ab_2 = str(type2) in ('24_A', '24_B')
        if is_ab_1 and is_ab_2:
            return [11, 24, 23, 23, 24, 11]
        if not is_ab_1 and not is_ab_2:
            return [11, 23, 24, 24, 23, 11]
        if is_ab_1 and not is_ab_2:
            return [11, 24, 23, 24, 23, 11]
        return [11, 23, 24, 23, 24, 11]

    @staticmethod
    def _side_ring_body_from_slots(slots: Sequence[Optional[Tuple[SUNode, str]]]) -> List[int]:
        """Return side-ring body order [pos1, pos3, pos4, pos2].

        Each position may be a real 24-like slot or a 23-like placeholder. This
        matches the user grammar 11-(24/23)-(24/23)-(24/23)-(24/23)-11 with
        two to four 24-like positions.
        """
        out: List[int] = []
        for slot in list(slots or [])[:4]:
            out.append(24 if slot is not None else 23)
        while len(out) < 4:
            out.append(23)
        return out

    @staticmethod
    def _vertical_ring_comp_for_inter(inter_types: Sequence[Optional[str]]) -> List[int]:
        inter = list(inter_types or [])[:2]
        while len(inter) < 2:
            inter.append(None)
        return [
            11,
            24,
            23,
            24 if inter[0] is not None else 23,
            23,
            24 if inter[1] is not None else 23,
            23,
        ]

    @staticmethod
    def _resource_override_for_stage_ring(comp: Sequence[int], tail_23: int, tail_22: int) -> Dict[str, int]:
        body = list(int(x) for x in list(comp or []))
        inner = body[1:-1] if len(body) >= 2 and body[0] == 11 and body[-1] == 11 else body
        counts = {'11': 0, '22': 0, '23': 0, '24': 0, '25': 0}
        for su in inner:
            if su in TO_22:
                counts['22'] += 1
            elif su in TO_23:
                counts['23'] += 1
            elif su in TO_24:
                counts['24'] += 1
            elif su == 25:
                counts['25'] += 1
        counts['23'] += int(tail_23)
        counts['22'] += int(tail_22)
        return counts

    @staticmethod
    def _ring_comp_from_body(body_su: Sequence[int], include_right_anchor: bool = True) -> List[int]:
        body = [int(x) for x in list(body_su or [])]
        if bool(include_right_anchor):
            return [11] + body + [11]
        return [11] + body

    @staticmethod
    def _connector_pair_metadata(connector_pair: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if connector_pair is None:
            return None
        return {
            'connector_id': int(connector_pair['connector_id']),
            'connector_su': int(connector_pair['connector_su']),
            'special_ids': [int(x) for x in connector_pair.get('special_ids', [])],
            'special_su': [int(x) for x in connector_pair.get('special_su', [])],
        }

    @staticmethod
    def _fused_ring_body_su(base_upper_type: Optional[str],
                            base_lower_type: Optional[str],
                            outer_upper_type: Optional[str],
                            outer_lower_type: Optional[str]) -> Tuple[List[int], List[int]]:
        base_ring = [
            24 if base_upper_type is not None else 23,
            24,
            24,
            24 if base_lower_type is not None else 23,
        ]
        outer_ring = [
            23,
            24 if outer_upper_type is not None else 23,
            24 if outer_lower_type is not None else 23,
            23,
        ]
        return base_ring, outer_ring

    def _apply_single_ring_body_substitute(
        self,
        body_su: Sequence[int],
        connectors: Sequence[SUNode],
        slot_nodes: Optional[Sequence[Optional[Tuple[SUNode, str]]]] = None,
        allow_su3_edge: bool = True,
    ) -> Tuple[List[int], List[int], List[int], Optional[Dict[str, Any]], Optional[SUNode]]:
        return self._apply_contextual_ring_substitute(
            body_su,
            list(slot_nodes or []),
            connectors,
            allow_su3_edge=bool(allow_su3_edge),
        )

    def _find_su3_b_like_substitute(self,
                                    connectors: Sequence[SUNode],
                                    used_ids: Optional[set[int]] = None) -> Optional[SUNode]:
        used = set(int(x) for x in set(used_ids or set()))
        for node in sorted(list(connectors or []), key=lambda n: int(getattr(n, 'global_id', -1))):
            if int(getattr(node, 'global_id', -1)) in used:
                continue
            if int(getattr(node, 'su_type', -1)) != 3:
                continue
            if str(getattr(node, 'su3_role', None)) == 'b_like':
                return node
        return None

    @staticmethod
    def _profile_su(node: SUNode) -> int:
        profile = FlexAllocator._branch24_profile(node)
        return int(profile.get('su_type', getattr(node, 'su_type', -1)))

    @staticmethod
    def _profile_is_ring_special(node: SUNode) -> bool:
        profile = FlexAllocator._branch24_profile(node)
        return (
            str(profile.get('fixed_usage', 'none')) == 'ring'
            and int(profile.get('su_type', getattr(node, 'su_type', -1))) in {19, 20}
        )

    @staticmethod
    def _valid_connector_su_for_special_pair(conn_su: int, left_su: int, right_su: int) -> bool:
        conn_i = int(conn_su)
        return (
            (conn_i == 27 and int(left_su) == 20 and int(right_su) == 20)
            or (conn_i in {29, 31} and int(left_su) == 19 and int(right_su) == 19)
        )

    def _connector_between_nodes(self,
                                 left_node: SUNode,
                                 right_node: SUNode,
                                 connectors: Sequence[SUNode]) -> Optional[Dict[str, Any]]:
        left_id = int(getattr(left_node, 'global_id', -1))
        right_id = int(getattr(right_node, 'global_id', -1))
        left_su = self._profile_su(left_node)
        right_su = self._profile_su(right_node)
        for conn in sorted(list(connectors or []), key=lambda n: int(getattr(n, 'global_id', -1))):
            conn_su = int(getattr(conn, 'su_type', -1))
            if not self._valid_connector_su_for_special_pair(conn_su, left_su, right_su):
                continue
            endpoint_ids = {
                int(getattr(ep, 'global_id', -1))
                for ep in self._other_connector_endpoints(conn, -999999)
            }
            if left_id in endpoint_ids and right_id in endpoint_ids:
                return {
                    'connector': conn,
                    'connector_id': int(getattr(conn, 'global_id', -1)),
                    'connector_su': int(conn_su),
                    'special_nodes': [left_node, right_node],
                    'special_ids': [int(left_id), int(right_id)],
                    'special_su': [int(left_su), int(right_su)],
                }
        return None

    def _apply_contextual_ring_substitute(
        self,
        body_su: Sequence[int],
        slot_nodes: Sequence[Optional[Tuple[SUNode, str]]],
        connectors: Sequence[SUNode],
        allow_su3_edge: bool,
    ) -> Tuple[List[int], List[int], List[int], Optional[Dict[str, Any]], Optional[SUNode]]:
        comp = [int(x) for x in list(body_su or [])]
        slots = list(slot_nodes or [])
        while len(slots) < len(comp):
            slots.append(None)

        # Connector-pair replacement is only legal for a local special-23-special
        # fragment, i.e. the connector replaces the middle 23 between two ring
        # 19/20 d3 slots that are explicitly in ring-connector mode.
        for idx, su in enumerate(comp):
            if int(su) != 23 or idx <= 0 or idx >= len(comp) - 1:
                continue
            left = slots[idx - 1]
            right = slots[idx + 1]
            if left is None or right is None:
                continue
            left_node, _left_role = left
            right_node, _right_role = right
            if not self._profile_is_ring_special(left_node) or not self._profile_is_ring_special(right_node):
                continue
            connector_pair = self._connector_between_nodes(left_node, right_node, connectors)
            if connector_pair is None:
                continue
            comp[idx] = int(connector_pair['connector_su'])
            ids = [int(connector_pair['connector_id'])] + [int(x) for x in connector_pair.get('special_ids', [])]
            types = [int(connector_pair['connector_su'])] + [int(x) for x in connector_pair.get('special_su', [])]
            return comp, ids, types, connector_pair, None

        if bool(allow_su3_edge) and comp:
            su3_node = self._find_su3_b_like_substitute(connectors)
            if su3_node is not None:
                # SU3 B-like replacement is restricted to an anchor-adjacent
                # 23 slot (11-23-...), matching the 9-3-... user rule.
                edge_indices = [0]
                if len(comp) > 1:
                    edge_indices.append(len(comp) - 1)
                for idx in edge_indices:
                    if int(comp[idx]) == 23:
                        comp[idx] = 3
                        return comp, [int(su3_node.global_id)], [3], None, su3_node

        return comp, [], [], None, None

    @staticmethod
    def _consume_ring_substitute_from_pools(connector_pair: Optional[Dict[str, Any]],
                                            su3_node: Optional[SUNode],
                                            remaining_B: List[SUNode],
                                            remaining_D: List[SUNode]) -> None:
        if connector_pair is not None:
            conn = connector_pair['connector']
            if conn in remaining_B:
                remaining_B.remove(conn)
            if conn in remaining_D:
                remaining_D.remove(conn)
            return
        if su3_node is not None:
            if su3_node in remaining_B:
                remaining_B.remove(su3_node)
            if su3_node in remaining_D:
                remaining_D.remove(su3_node)

    def _allocate_reserved_terminal_sides(self,
                                          remaining_E: List[SUNode],
                                          remaining_G: List[SUNode],
                                          avail_11: int,
                                          avail_23: int,
                                          avail_22: int) -> Tuple[List[ChainSpec], int, int, int]:
        """Allocate remaining reserved Type E/G after branch sealing."""
        chains = []

        for e in remaining_E:
            if avail_11 >= 1 and avail_23 >= 3 and avail_22 >= 1:
                comp = [22, 23, 23, 23, 11]
                chains.append(ChainSpec('side', comp, 'E', [e.global_id]))
                avail_11 -= 1; avail_23 -= 3; avail_22 -= 1
            elif avail_11 >= 1 and avail_23 >= 2 and avail_22 >= 1:
                comp = [22, 23, 23, 11]
                chains.append(ChainSpec('side', comp, 'E', [e.global_id]))
                avail_11 -= 1; avail_23 -= 2; avail_22 -= 1
            else:
                print(f"  [WARN] Cannot close remaining Type E node {e.global_id}")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 1

        for g in remaining_G:
            if avail_11 >= 1 and avail_23 >= 2 and avail_22 >= 1:
                comp = [22, 23, 23, 11]
                chains.append(ChainSpec('side', comp, 'G', [g.global_id]))
                avail_11 -= 1; avail_23 -= 2; avail_22 -= 1
            elif avail_11 >= 1 and avail_23 >= 1 and avail_22 >= 1:
                comp = [22, 23, 11]
                chains.append(ChainSpec('side', comp, 'G', [g.global_id]))
                avail_11 -= 1; avail_23 -= 1; avail_22 -= 1
            else:
                print(f"  [WARN] Cannot close remaining Type G node {g.global_id}")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 1

        return chains, avail_11, avail_23, avail_22

    def _get_branch_terminals(self, count: int, avail_11: int, avail_22: int) -> Tuple[List[int], int, int]:
        """Return a list of terminal SU types (strictly 22) and updated avail counts."""
        terms = []
        for _ in range(count):
            if avail_22 > 0:
                terms.append(22)
                avail_22 -= 1
            else:
                return [], avail_11, avail_22 # Failed (no 11 fallback allowed for branches)
        return terms, avail_11, avail_22

    def _accumulate_shortage(self, need_11: int = 0, need_23: int = 0, need_22: int = 0):
        self._result.required_extra_11 += max(0, int(need_11))
        self._result.required_extra_23 += max(0, int(need_23))
        self._result.required_extra_22 += max(0, int(need_22))

    def _allocate_su25_only(self,
                            avail_11: int,
                            avail_23: int,
                            avail_22: int,
                            avail_25: Optional[int] = None,
                            excluded_ids: Optional[set[int]] = None):
        chains = []
        avail_25 = int(self._result.total_25 if avail_25 is None else avail_25)
        excluded = set(int(x) for x in set(excluded_ids or set()))
        for t_key in ['25_aro', '25_ali']:
            is_aro = (t_key == '25_aro')
            for n in self._type_lists.get(t_key, []):
                if int(getattr(n, 'global_id', -1)) in excluded:
                    continue
                if avail_25 < 1:
                    self._result.unallocated_branch += 1
                    continue

                tail23, tail22, tail_src = self._consume_reserved_branch_tail(t_key, [], [])
                extra_22_count = 1
                total_23_needed = 1 + int(tail23)
                allocated = False
                branch_meta = {
                    'branch_type': t_key,
                    'branch_23_count': int(tail23),
                    'branch_22_count': int(tail22),
                    'extra_22_count': int(extra_22_count),
                    'tail_source': str(tail_src),
                }

                if avail_11 >= 1 and avail_23 >= total_23_needed:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    side_terminals = int(tail22 + extra_22_count + 1)
                    terms, temp_11, temp_22 = self._get_branch_terminals(side_terminals, temp_11, temp_22)
                    if len(terms) == int(side_terminals):
                        if is_aro:
                            comp = [11, 25]
                        else:
                            comp = [11, 23, 25]
                        comp += [23] * int(tail23)
                        comp += [terms[0]]
                        comp += [23]
                        comp += list(terms[1:])
                        desc = f"Br-25({'aro' if is_aro else 'ali'})"
                        chains.append(ChainSpec('branch_side', comp, desc, [n.global_id], metadata=branch_meta.copy()))
                        avail_11 = temp_11
                        avail_22 = temp_22
                        avail_23 -= total_23_needed
                        avail_25 -= 1
                        allocated = True

                if not allocated and avail_11 >= 2 and avail_23 >= total_23_needed:
                    temp_11 = avail_11 - 2
                    temp_22 = avail_22
                    bridge_terminals = int(tail22 + extra_22_count)
                    terms, temp_11, temp_22 = self._get_branch_terminals(bridge_terminals, temp_11, temp_22)
                    if len(terms) == int(bridge_terminals):
                        if is_aro:
                            comp = [11, 25]
                        else:
                            comp = [11, 23, 25]
                        comp += [23] * int(tail23)
                        comp += [22] * int(tail22)
                        comp += [23, 11]
                        comp += list(terms)
                        desc = f"Br-25({'aro' if is_aro else 'ali'})"
                        chains.append(ChainSpec('branch_bridge', comp, desc, [n.global_id], metadata=branch_meta.copy()))
                        avail_11 = temp_11
                        avail_22 = temp_22
                        avail_23 -= total_23_needed
                        avail_25 -= 1
                        allocated = True

                if not allocated:
                    print(f"  [WARN] Cannot allocate branch for 25 node {n.global_id}")
                    self._result.unallocated_branch += 1
                    side_need_11 = max(0, 1 - avail_11)
                    side_need_23 = max(0, total_23_needed - avail_23)
                    side_need_22 = max(0, int(tail22 + extra_22_count + 1) - avail_22)
                    bridge_need_11 = max(0, 2 - avail_11)
                    bridge_need_23 = max(0, total_23_needed - avail_23)
                    bridge_need_22 = max(0, int(tail22 + extra_22_count) - avail_22)
                    side_gap = side_need_11 + side_need_23 + side_need_22
                    bridge_gap = bridge_need_11 + bridge_need_23 + bridge_need_22
                    if side_gap <= bridge_gap:
                        self._accumulate_shortage(side_need_11, side_need_23, side_need_22)
                    else:
                        self._accumulate_shortage(bridge_need_11, bridge_need_23, bridge_need_22)

        return chains, avail_11, avail_23, avail_22, avail_25

    # ---------- Phase 4.5: Allocate branches (24/25) ----------
    
    def _allocate_branches(self, avail_11: int, avail_23: int, avail_22: int,
                           remaining_E: Optional[List[SUNode]] = None,
                           remaining_G: Optional[List[SUNode]] = None,
                           remaining_B: Optional[List[SUNode]] = None,
                           remaining_D: Optional[List[SUNode]] = None):
        """Allocate branch structures for SU 24 and 25.
        Returns: (chains, avail_11, avail_23, avail_22, avail_24, avail_25)
        """
        remaining_E = list(remaining_E or [])
        remaining_G = list(remaining_G or [])
        remaining_B = list(remaining_B if remaining_B is not None else self._type_lists.get('B', []))
        remaining_D = list(remaining_D if remaining_D is not None else self._type_lists.get('D', []))
        avail_24 = self._result.total_24
        special_candidates = self._build_special_branch_patterns()
        avail_25 = int(self._result.total_25)
        chains, avail_11, avail_23, avail_22, avail_24, avail_25 = self._select_feasible_special_branch_patterns(
            special_candidates,
            avail_11,
            avail_23,
            avail_22,
            avail_24,
            avail_25,
        )
        selected_special_ids = set()
        for ch in list(chains or []):
            selected_special_ids.update(int(x) for x in self._collect_chain_trace_ids(ch))
        A = [(n, '24_A') for n in self._type_lists.get('24_A', []) if int(getattr(n, 'global_id', -1)) not in selected_special_ids]
        B = [(n, '24_B') for n in self._type_lists.get('24_B', []) if int(getattr(n, 'global_id', -1)) not in selected_special_ids]
        C = [(n, '24_C') for n in self._type_lists.get('24_C', []) if int(getattr(n, 'global_id', -1)) not in selected_special_ids]
        D = [(n, '24_D') for n in self._type_lists.get('24_D', []) if int(getattr(n, 'global_id', -1)) not in selected_special_ids]

        AB = A + B

        # ===== Step 1: C-dominant fused side rings (脂肪并环, 最高优先级) =====
        # 优先消耗 2 个固定 C 类 + 2 个 AB/CD 外层位，构建最完整的并环。
        c_pool = [n for n in C if self._profile_allows_slot(n[0], 'fused_fixed_c_bridgehead')]
        mixed_cd_pool = [n for n in (C + D) if n not in c_pool[:2]]
        while len(c_pool) >= 2 and len(mixed_cd_pool) >= 2 and len(AB) >= 2:
            fixed_c1 = c_pool.pop(0)
            fixed_c2 = c_pool.pop(0)
            base1 = AB.pop(0)
            base2 = AB.pop(0)
            node1 = mixed_cd_pool.pop(0)
            node2 = mixed_cd_pool.pop(0)
            if (
                not self._profile_allows_slot(base1[0], 'fused_ab')
                or not self._profile_allows_slot(base2[0], 'fused_ab')
                or not self._profile_allows_slot(node1[0], 'fused_outer_cd')
                or not self._profile_allows_slot(node2[0], 'fused_outer_cd')
            ):
                AB.insert(0, base2)
                AB.insert(0, base1)
                for candidate in (node2, node1, fixed_c2, fixed_c1):
                    if candidate[1] == '24_C':
                        C.insert(0, candidate)
                    else:
                        D.insert(0, candidate)
                break
            for candidate in (fixed_c1, fixed_c2, node1, node2):
                if candidate in C:
                    C.remove(candidate)
                if candidate in D:
                    D.remove(candidate)

            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            b1_23, b1_22, src1 = self._consume_branch_tail_for_node_with_special_reuse(node1[0], node1[1], remaining_E, remaining_G)
            b2_23, b2_22, src2 = self._consume_branch_tail_for_node_with_special_reuse(node2[0], node2[1], remaining_E, remaining_G)
            base1_23, base1_22, base_src1 = self._consume_branch_tail_for_node_with_special_reuse(base1[0], base1[1], remaining_E, remaining_G)
            base2_23, base2_22, base_src2 = self._consume_branch_tail_for_node_with_special_reuse(base2[0], base2[1], remaining_E, remaining_G)
            branch_23 = b1_23 + b2_23 + base1_23 + base2_23
            branch_22 = b1_22 + b2_22 + base1_22 + base2_22
            base_ring_su, outer_ring_su = self._fused_ring_body_su(
                base1[1],
                base2[1],
                node1[1],
                node2[1],
            )
            ring_body_su = list(base_ring_su) + list(outer_ring_su)
            ring_slots = [
                base1,
                fixed_c1,
                fixed_c2,
                base2,
                None,
                node1,
                node2,
                None,
            ]
            ring_23_needed = sum(1 for su in ring_body_su if int(su) == 23)
            total_23_needed = ring_23_needed + branch_23

            if avail_11 >= 2 and avail_23 >= total_23_needed and avail_22 >= branch_22 and avail_24 >= 6:
                ring_body_su, substitute_ids, substitute_types, connector_pair, su3_substitute = self._apply_single_ring_body_substitute(
                    ring_body_su,
                    list(remaining_B) + list(remaining_D),
                    slot_nodes=ring_slots,
                    allow_su3_edge=True,
                )
                base_ring_su = list(ring_body_su[:4])
                outer_ring_su = list(ring_body_su[4:])
                ring_comp = self._ring_comp_from_body(ring_body_su)
                total_23_needed = sum(1 for su in ring_body_su if int(su) == 23) + branch_23
                if not (avail_23 >= total_23_needed):
                    remaining_E = prev_E
                    remaining_G = prev_G
                    AB.insert(0, base2)
                    AB.insert(0, base1)
                    for candidate in (node2, node1, fixed_c2, fixed_c1):
                        if candidate[1] == '24_C':
                            C.insert(0, candidate)
                        else:
                            D.insert(0, candidate)
                    break
                comp = list(ring_comp)
                comp += [23] * branch_23 + [22] * branch_22
                ids = [
                    base1[0].global_id, base2[0].global_id,
                    fixed_c1[0].global_id, fixed_c2[0].global_id,
                    node1[0].global_id, node2[0].global_id,
                ]
                ids = sorted(set(int(x) for x in ids + list(substitute_ids)))
                desc = f"Fused-S-ring(Base:{base1[1][-1]}{base2[1][-1]}+Br:CC)+Out:{node1[1][-1]}{node2[1][-1]}"
                fused_meta = {
                    'fused_priority_path': True,
                    'fixed_c_ids': [int(fixed_c1[0].global_id), int(fixed_c2[0].global_id)],
                    'fixed_c_source_kinds': [
                        self._branch24_profile(fixed_c1[0]).get('source_kind'),
                        self._branch24_profile(fixed_c2[0]).get('source_kind'),
                    ],
                    'fixed_c_path_kinds': [
                        self._branch24_profile(fixed_c1[0]).get('fixed_path_kind'),
                        self._branch24_profile(fixed_c2[0]).get('fixed_path_kind'),
                    ],
                    'base_ring_ids': [int(base1[0].global_id), int(base2[0].global_id)],
                    'base_ring_roles': [
                        self._branch_role_for_type(base1[1]),
                        self._branch_role_for_type(base2[1]),
                    ],
                    'tail_sources': [base_src1, base_src2, src1, src2],
                    'branch24_profiles': self._branch24_profiles_for_items([
                        (base1[0], 'fused_base'),
                        (base2[0], 'fused_base'),
                        (fixed_c1[0], 'fused_fixed_c'),
                        (fixed_c2[0], 'fused_fixed_c'),
                        (node1[0], 'fused_outer'),
                        (node2[0], 'fused_outer'),
                    ]),
                    'resource_requirements_override': self._resource_override_for_stage_ring(
                        ring_comp,
                        branch_23,
                        branch_22,
                    ),
                    'base_ring_su': list(base_ring_su),
                    'outer_ring_su': list(outer_ring_su),
                    'ring_substitute_ids': list(substitute_ids),
                    'ring_substitute_types': list(substitute_types),
                    'connector_pair': self._connector_pair_metadata(connector_pair),
                    'branch_tail_lengths': {
                        'base_upper': base1_23 + base1_22,
                        'base_lower': base2_23 + base2_22,
                        'outer_upper': b1_23 + b1_22,
                        'outer_lower': b2_23 + b2_22,
                    },
                }
                chains.append(ChainSpec('fused_side_ring', comp, desc, ids, metadata=fused_meta))
                avail_11 -= 2
                avail_23 -= total_23_needed
                avail_22 -= branch_22
                avail_24 -= 6
                self._consume_ring_substitute_from_pools(connector_pair, su3_substitute, remaining_B, remaining_D)
                continue

            remaining_E = prev_E
            remaining_G = prev_G
            AB.insert(0, base2)
            AB.insert(0, base1)
            for candidate in (node2, node1, fixed_c2, fixed_c1):
                if candidate[1] == '24_C':
                    C.insert(0, candidate)
                else:
                    D.insert(0, candidate)
            break

        # ===== Step 1.5: Generic fused side rings (脂肪并环) =====
        C_fixed = [n for n in C if self._profile_allows_slot(n[0], 'fused_fixed_c_bridgehead')]
        C_nonfixed = [n for n in C if n not in C_fixed]
        C = list(C_fixed) + list(C_nonfixed)
        while len(C_fixed) >= 2:
            bridge1 = C_fixed.pop(0)
            bridge2 = C_fixed.pop(0)
            if bridge1 in C:
                C.remove(bridge1)
            if bridge2 in C:
                C.remove(bridge2)

            base_nodes = []
            while AB and len(base_nodes) < 2:
                candidate = AB.pop(0)
                if self._profile_allows_slot(candidate[0], 'fused_ab'):
                    base_nodes.append(candidate)

            outer_pool = sorted(
                [x for x in (C + D) if self._profile_allows_slot(x[0], 'fused_outer_cd')],
                key=lambda x: 0 if x[1] == '24_C' else 1,
            )
            outer_nodes = []
            while outer_pool and len(outer_nodes) < 2:
                node = outer_pool.pop(0)
                outer_nodes.append(node)

            for s in outer_nodes:
                if s in C:
                    C.remove(s)
                elif s in D:
                    D.remove(s)

            n_extra_24 = len(base_nodes) + len(outer_nodes)
            if n_extra_24 < 2:
                C_fixed.insert(0, bridge2)
                C_fixed.insert(0, bridge1)
                C.insert(0, bridge2)
                C.insert(0, bridge1)
                for s in reversed(base_nodes):
                    AB.insert(0, s)
                for s in reversed(outer_nodes):
                    if s[1] == '24_C':
                        C.insert(0, s)
                    else:
                        D.insert(0, s)
                break

            base_str = "".join(n[1][-1] for n in base_nodes).ljust(2, 'X')
            out_str = "".join(n[1][-1] for n in outer_nodes).ljust(2, 'X')
            nodes_to_consume = 2 + n_extra_24

            base_upper_type = base_nodes[0][1] if len(base_nodes) > 0 else None
            base_lower_type = base_nodes[1][1] if len(base_nodes) > 1 else None
            outer_upper_type = outer_nodes[0][1] if len(outer_nodes) > 0 else None
            outer_lower_type = outer_nodes[1][1] if len(outer_nodes) > 1 else None

            base_ring_su, outer_ring_su = self._fused_ring_body_su(
                base_upper_type,
                base_lower_type,
                outer_upper_type,
                outer_lower_type,
            )
            ring_body_su = list(base_ring_su) + list(outer_ring_su)
            ring_slots = [
                base_nodes[0] if len(base_nodes) > 0 else None,
                bridge1,
                bridge2,
                base_nodes[1] if len(base_nodes) > 1 else None,
                None,
                outer_nodes[0] if len(outer_nodes) > 0 else None,
                outer_nodes[1] if len(outer_nodes) > 1 else None,
                None,
            ]
            ring_23_needed = sum(1 for su in ring_body_su if int(su) == 23)

            branch_23 = 0
            branch_22 = 0
            tail_sources = []
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            prev_D = list(remaining_D)
            for n in base_nodes + outer_nodes:
                b23, b22, tail_src = self._consume_branch_tail_for_node_with_special_reuse(n[0], n[1], remaining_E, remaining_G)
                branch_23 += b23
                branch_22 += b22
                tail_sources.append(tail_src)

            total_23_needed = ring_23_needed + branch_23

            if avail_11 >= 2 and avail_23 >= total_23_needed and avail_22 >= branch_22 and avail_24 >= nodes_to_consume:
                ring_body_su, substitute_ids, substitute_types, connector_pair, su3_substitute = self._apply_single_ring_body_substitute(
                    ring_body_su,
                    list(remaining_B) + list(remaining_D),
                    slot_nodes=ring_slots,
                    allow_su3_edge=True,
                )
                base_ring_su = list(ring_body_su[:4])
                outer_ring_su = list(ring_body_su[4:])
                ring_comp = self._ring_comp_from_body(ring_body_su)
                total_23_needed = sum(1 for su in ring_body_su if int(su) == 23) + branch_23
                if not (avail_23 >= total_23_needed):
                    remaining_E = prev_E
                    remaining_G = prev_G
                    remaining_D = prev_D
                    C_fixed.insert(0, bridge2)
                    C_fixed.insert(0, bridge1)
                    C.insert(0, bridge2)
                    C.insert(0, bridge1)
                    for s in reversed(base_nodes):
                        AB.insert(0, s)
                    for s in reversed(outer_nodes):
                        if s[1] == '24_C':
                            C.insert(0, s)
                        else:
                            D.insert(0, s)
                    break
                comp = ring_comp[:-1] + [23] * branch_23 + [22] * branch_22 + [ring_comp[-1]]
                ids = [bridge1[0].global_id, bridge2[0].global_id] + [n[0].global_id for n in base_nodes + outer_nodes]
                ids = sorted(set(int(x) for x in ids + list(substitute_ids)))
                desc = f"Fused-S-ring(Base:{base_str}+Br:CC)+Out:{out_str}"

                base_upper_len = sum(self._branch_cost(base_upper_type)) if base_upper_type else 0
                base_lower_len = sum(self._branch_cost(base_lower_type)) if base_lower_type else 0
                outer_upper_len = sum(self._branch_cost(outer_upper_type)) if outer_upper_type else 0
                outer_lower_len = sum(self._branch_cost(outer_lower_type)) if outer_lower_type else 0

                fused_meta = {
                    'fixed_c_ids': [int(bridge1[0].global_id), int(bridge2[0].global_id)],
                    'fixed_c_source_kinds': [
                        self._branch24_profile(bridge1[0]).get('source_kind'),
                        self._branch24_profile(bridge2[0]).get('source_kind'),
                    ],
                    'fixed_c_path_kinds': [
                        self._branch24_profile(bridge1[0]).get('fixed_path_kind'),
                        self._branch24_profile(bridge2[0]).get('fixed_path_kind'),
                    ],
                    'tail_sources': tail_sources,
                    'base_ring_roles': [self._branch_role_for_type(x[1]) for x in base_nodes],
                    'outer_ring_roles': [self._branch_role_for_type(x[1]) for x in outer_nodes],
                    'branch24_profiles': self._branch24_profiles_for_items(
                        [(bridge1[0], 'fused_fixed_c'), (bridge2[0], 'fused_fixed_c')]
                        + [(x[0], 'fused_base') for x in base_nodes]
                        + [(x[0], 'fused_outer') for x in outer_nodes]
                    ),
                    'resource_requirements_override': self._resource_override_for_stage_ring(
                        ring_comp,
                        branch_23,
                        branch_22,
                    ),
                    'base_ring_su': list(base_ring_su),
                    'outer_ring_su': list(outer_ring_su),
                    'ring_substitute_ids': list(substitute_ids),
                    'ring_substitute_types': list(substitute_types),
                    'connector_pair': self._connector_pair_metadata(connector_pair),
                    'branch_tail_lengths': {
                        'base_upper': int(base_upper_len),
                        'base_lower': int(base_lower_len),
                        'outer_upper': int(outer_upper_len),
                        'outer_lower': int(outer_lower_len),
                    },
                }
                chains.append(ChainSpec('fused_side_ring', comp, desc, ids, metadata=fused_meta))
                avail_11 = max(0, avail_11 - 2)
                avail_23 -= total_23_needed
                avail_22 -= branch_22
                avail_24 -= nodes_to_consume
                self._consume_ring_substitute_from_pools(connector_pair, su3_substitute, remaining_B, remaining_D)
                continue

            remaining_E = prev_E
            remaining_G = prev_G
            remaining_D = prev_D
            C_fixed.insert(0, bridge2)
            C_fixed.insert(0, bridge1)
            C.insert(0, bridge2)
            C.insert(0, bridge1)
            for s in reversed(base_nodes):
                AB.insert(0, s)
            for s in reversed(outer_nodes):
                if s[1] == '24_C':
                    C.insert(0, s)
                else:
                    D.insert(0, s)
            break

        # Re-sync A, B from AB after fused-ring allocation
        A = [x for x in AB if x[1] == '24_A']
        B = [x for x in AB if x[1] == '24_B']
        CD = C + D

        # ===== Step 2: Vertical Rings (上下脂肪环, 第二优先级) =====
        a_idx = 0
        while a_idx < len(A) and CD:
            a_node = A[a_idx]
            if not self._profile_allows_slot(a_node[0], 'vertical_fixed_a'):
                a_idx += 1
                continue

            if len(CD) >= 2:
                cd1 = CD.pop(0)
                cd2 = CD.pop(0)
                if not self._profile_allows_slot(cd1[0], 'vertical_inter') or not self._profile_allows_slot(cd2[0], 'vertical_inter'):
                    CD.insert(0, cd2)
                    CD.insert(0, cd1)
                    a_idx += 1
                    continue
                ring_comp = self._vertical_ring_comp_for_inter([cd1[1], cd2[1]])
                ring_body_su, substitute_ids, substitute_types, connector_pair, su3_substitute = self._apply_single_ring_body_substitute(
                    list(ring_comp[1:]),
                    list(remaining_B) + list(remaining_D),
                    slot_nodes=[a_node, None, cd1, None, cd2, None],
                    allow_su3_edge=False,
                )
                ring_comp = [11] + list(ring_body_su)
                base_ring_23 = sum(1 for su in ring_comp[1:] if int(su) == 23)
                ring_23 = int(base_ring_23)
                ring_terms = 0
                prev_E = list(remaining_E)
                prev_G = list(remaining_G)
                b23_1, bterms_1, src1 = self._consume_branch_tail_for_node_with_special_reuse(cd1[0], cd1[1], remaining_E, remaining_G)
                b23_2, bterms_2, src2 = self._consume_branch_tail_for_node_with_special_reuse(cd2[0], cd2[1], remaining_E, remaining_G)
                ring_23 += b23_1 + b23_2
                ring_terms += bterms_1 + bterms_2
                
                if avail_11 >= 1 and avail_23 >= ring_23 and avail_24 >= 3:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    terms, temp_11, temp_22 = self._get_branch_terminals(ring_terms, temp_11, temp_22)
                    
                    if len(terms) == ring_terms:
                        comp = list(ring_comp)
                        comp += [23] * (b23_1 + b23_2) + terms
                        ids = [a_node[0].global_id, cd1[0].global_id, cd2[0].global_id]
                        ids = sorted(set(int(x) for x in ids + list(substitute_ids)))
                        desc = f"V-ring(A+{cd1[1][-1]}+{cd2[1][-1]})"
                        vr_meta = {
                            'tail_sources': [src1, src2],
                            'vertical_inter_types': [cd1[1], cd2[1]],
                            'vertical_inter_roles': [
                                self._branch_role_for_type(cd1[1]),
                                self._branch_role_for_type(cd2[1]),
                            ],
                            'vertical_ring_su': list(ring_comp[1:]),
                            'branch24_profiles': self._branch24_profiles_for_items([
                                (a_node[0], 'vertical_fixed_a'),
                                (cd1[0], 'vertical_inter'),
                                (cd2[0], 'vertical_inter'),
                            ]),
                            'resource_requirements_override': self._resource_override_for_stage_ring(
                                ring_comp,
                                b23_1 + b23_2,
                                ring_terms,
                            ),
                            'ring_substitute_ids': list(substitute_ids),
                            'ring_substitute_types': list(substitute_types),
                            'connector_pair': self._connector_pair_metadata(connector_pair),
                            'branch_tail_lengths': {
                                'right': b23_1 + bterms_1,
                                'left': b23_2 + bterms_2,
                            },
                        }
                        chains.append(ChainSpec('vertical_ring', comp, desc, ids, metadata=vr_meta))
                        avail_11 = temp_11; avail_22 = temp_22
                        avail_23 -= ring_23; avail_24 -= 3
                        self._consume_ring_substitute_from_pools(connector_pair, su3_substitute, remaining_B, remaining_D)
                        
                        A.pop(a_idx)
                        C = [x for x in CD if x[1] == '24_C']
                        D = [x for x in CD if x[1] == '24_D']
                        continue
                remaining_E = prev_E
                remaining_G = prev_G
                CD.insert(0, cd2)
                CD.insert(0, cd1)

            if len(CD) >= 1:
                cd1 = CD.pop(0)
                if not self._profile_allows_slot(cd1[0], 'vertical_inter'):
                    CD.insert(0, cd1)
                    a_idx += 1
                    continue
                ring_comp = self._vertical_ring_comp_for_inter([cd1[1], None])
                ring_body_su, substitute_ids, substitute_types, connector_pair, su3_substitute = self._apply_single_ring_body_substitute(
                    list(ring_comp[1:]),
                    list(remaining_B) + list(remaining_D),
                    slot_nodes=[a_node, None, cd1, None, None, None],
                    allow_su3_edge=False,
                )
                ring_comp = [11] + list(ring_body_su)
                base_ring_23 = sum(1 for su in ring_comp[1:] if int(su) == 23)
                ring_23 = int(base_ring_23)
                ring_terms = 0
                prev_E = list(remaining_E)
                prev_G = list(remaining_G)
                b23, bterms, src1 = self._consume_branch_tail_for_node_with_special_reuse(cd1[0], cd1[1], remaining_E, remaining_G)
                ring_23 += b23
                ring_terms += bterms
                
                if avail_11 >= 1 and avail_23 >= ring_23 and avail_24 >= 2:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    terms, temp_11, temp_22 = self._get_branch_terminals(ring_terms, temp_11, temp_22)
                    
                    if len(terms) == ring_terms:
                        comp = list(ring_comp)
                        comp += [23] * b23 + terms
                        ids = [a_node[0].global_id, cd1[0].global_id]
                        ids = sorted(set(int(x) for x in ids + list(substitute_ids)))
                        desc = f"V-ring(A+{cd1[1][-1]})"
                        vr_meta = {
                            'tail_sources': [src1],
                            'vertical_inter_types': [cd1[1], None],
                            'vertical_inter_roles': [self._branch_role_for_type(cd1[1]), None],
                            'vertical_ring_su': list(ring_comp[1:]),
                            'branch24_profiles': self._branch24_profiles_for_items([
                                (a_node[0], 'vertical_fixed_a'),
                                (cd1[0], 'vertical_inter'),
                            ]),
                            'resource_requirements_override': self._resource_override_for_stage_ring(
                                ring_comp,
                                b23,
                                ring_terms,
                            ),
                            'ring_substitute_ids': list(substitute_ids),
                            'ring_substitute_types': list(substitute_types),
                            'connector_pair': self._connector_pair_metadata(connector_pair),
                            'branch_tail_lengths': {
                                'right': b23 + bterms,
                            },
                        }
                        chains.append(ChainSpec('vertical_ring', comp, desc, ids, metadata=vr_meta))
                        avail_11 = temp_11; avail_22 = temp_22
                        avail_23 -= ring_23; avail_24 -= 2
                        self._consume_ring_substitute_from_pools(connector_pair, su3_substitute, remaining_B, remaining_D)
                        
                        A.pop(a_idx)
                        C = [x for x in CD if x[1] == '24_C']
                        D = [x for x in CD if x[1] == '24_D']
                        continue
                remaining_E = prev_E
                remaining_G = prev_G
                CD.insert(0, cd1)
            a_idx += 1

        # ===== Step 3: Side Rings (侧边脂肪环, 第三优先级) =====
        # Four independent ring slots are available in traversal order
        # [pos1, pos3, pos4, pos2].  pos1/pos2 prefer AB-type 24-like nodes
        # while pos3/pos4 prefer CD-type nodes, but any slot may remain as a
        # 23-like placeholder.  Legal side rings therefore contain 2-4 actual
        # 24-like positions, matching 11-(24/23)^4-11.
        AB = A + B
        remaining: List[Tuple[SUNode, str]] = AB + CD

        while len(remaining) >= 2:
            prev_remaining = list(remaining)
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)

            slots: List[Optional[Tuple[SUNode, str]]] = [None, None, None, None]

            def _take_for_slot(slot_name: str, preferred_types: Tuple[str, ...]) -> Optional[Tuple[SUNode, str]]:
                for prefer in (True, False):
                    for idx, cand in enumerate(list(remaining)):
                        ctype = str(cand[1])
                        if bool(prefer) and ctype not in set(preferred_types):
                            continue
                        if not bool(prefer) and ctype in set(preferred_types):
                            continue
                        if self._profile_allows_slot(cand[0], slot_name):
                            return remaining.pop(idx)
                return None

            slots[0] = _take_for_slot('side_ab', ('24_A', '24_B'))
            slots[3] = _take_for_slot('side_ab', ('24_A', '24_B'))
            slots[1] = _take_for_slot('side_cd', ('24_C', '24_D'))
            slots[2] = _take_for_slot('side_cd', ('24_C', '24_D'))

            slot_items = [slot for slot in slots if slot is not None]
            if len(slot_items) < 2:
                remaining = prev_remaining
                break

            branch_23 = 0
            ring_22 = 0
            tail_sources: List[Optional[str]] = []
            branch_tail_lengths: Dict[str, int] = {}
            slot_names = ['pos1', 'pos3', 'pos4', 'pos2']
            for idx, slot in enumerate(slots):
                if slot is None:
                    tail_sources.append(None)
                    continue
                b23, b22, src = self._consume_branch_tail_for_node_with_special_reuse(
                    slot[0],
                    slot[1],
                    remaining_E,
                    remaining_G,
                )
                branch_23 += int(b23)
                ring_22 += int(b22)
                tail_sources.append(src)
                branch_tail_lengths[slot_names[idx]] = int(b23 + b22)

            ring_body_su = self._side_ring_body_from_slots(slots)
            ring_body_su, substitute_ids, substitute_types, connector_pair, su3_substitute = self._apply_single_ring_body_substitute(
                list(ring_body_su),
                list(remaining_B) + list(remaining_D),
                slot_nodes=slots,
                allow_su3_edge=True,
            )
            ring_comp = self._ring_comp_from_body(ring_body_su)
            ring_body_23 = sum(1 for su in ring_comp[1:-1] if int(su) == 23)
            total_23 = int(ring_body_23 + branch_23)

            if avail_11 >= 2 and avail_23 >= total_23 and avail_22 >= ring_22 and avail_24 >= len(slot_items):
                comp = list(ring_comp)
                comp += [23] * int(branch_23) + [22] * int(ring_22)
                ids = [int(slot[0].global_id) for slot in slot_items]
                ids = sorted(set(int(x) for x in ids + list(substitute_ids)))
                type_desc = ''.join(str(slot[1][-1]) if slot is not None else 'X' for slot in slots)
                desc = f"S-ring({type_desc})"
                sr_meta = {
                    'tail_sources': list(tail_sources),
                    'side_ring_node_types': [slot[1] if slot is not None else None for slot in slots],
                    'side_ring_node_roles': [
                        self._branch_role_for_type(slot[1]) if slot is not None else None
                        for slot in slots
                    ],
                    'side_ring_slot_names': list(slot_names),
                    'branch24_profiles': self._branch24_profiles_for_items([
                        (slot[0], 'side_slot') for slot in slots if slot is not None
                    ]),
                    'resource_requirements_override': self._resource_override_for_stage_ring(
                        ring_comp,
                        branch_23,
                        ring_22,
                    ),
                    'side_ring_su': list(ring_comp[1:-1]),
                    'ring_substitute_ids': list(substitute_ids),
                    'ring_substitute_types': list(substitute_types),
                    'connector_pair': self._connector_pair_metadata(connector_pair),
                    'branch_tail_lengths': branch_tail_lengths,
                }
                chains.append(ChainSpec('side_ring', comp, desc, ids, metadata=sr_meta))

                avail_11 -= 2
                avail_23 -= total_23
                avail_22 -= ring_22
                avail_24 -= len(slot_items)
                self._consume_ring_substitute_from_pools(connector_pair, su3_substitute, remaining_B, remaining_D)
            else:
                remaining = prev_remaining
                remaining_E = prev_E
                remaining_G = prev_G
                break

        # ===== Step 4: Chain branches for remaining single 24 (最低优先级中的 24) =====
        for n in remaining:
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            b23, b22, tail_src = self._consume_branch_tail_for_node_with_special_reuse(n[0], n[1], remaining_E, remaining_G)
            is_ab = n[1] in ('24_A', '24_B')
            base_23 = 2
            
            if avail_24 < 1:
                print(f"  [WARN] No 24 left for branch node {n[0].global_id}")
                self._result.unallocated_branch += 1
                continue
                
            total_23_needed = base_23 + b23
            
            if avail_11 >= 1 and avail_23 >= total_23_needed and avail_22 >= (b22 + 1):
                if is_ab: comp = [11, 24, 23, 23, 22]
                else:     comp = [11, 23, 24, 23, 22]
                comp += [23] * b23 + [22] * b22
                branch_meta = {
                    'branch_type': n[1],
                    'branch_role': self._branch_role_for_type(n[1]),
                    'branch_23_count': b23,
                    'branch_22_count': b22,
                    'extra_22_count': 0,
                    'tail_source': tail_src,
                    'branch24_profiles': self._branch24_profiles_for_items([(n[0], 'branch_single')]),
                }
                chains.append(ChainSpec('branch_side', comp, f"Br-chain({n[1][-1]})", [n[0].global_id], metadata=branch_meta.copy()))
                avail_11 -= 1; avail_23 -= total_23_needed; avail_22 -= (b22 + 1); avail_24 -= 1
            elif avail_11 >= 2 and avail_23 >= total_23_needed and avail_22 >= b22:
                if is_ab: comp = [11, 24, 23, 23, 11]
                else:     comp = [11, 23, 24, 23, 11]
                comp += [23] * b23 + [22] * b22
                branch_meta = {
                    'branch_type': n[1],
                    'branch_role': self._branch_role_for_type(n[1]),
                    'branch_23_count': b23,
                    'branch_22_count': b22,
                    'extra_22_count': 0,
                    'tail_source': tail_src,
                    'branch24_profiles': self._branch24_profiles_for_items([(n[0], 'branch_single')]),
                }
                chains.append(ChainSpec('branch_bridge', comp, f"Br-chain({n[1][-1]})", [n[0].global_id], metadata=branch_meta.copy()))
                avail_11 -= 2; avail_23 -= total_23_needed; avail_22 -= b22; avail_24 -= 1
            else:
                print(f"  [WARN] Cannot allocate branch for 24 node {n[0].global_id} ({n[1]})")
                self._result.unallocated_branch += 1
                side_need_11 = max(0, 1 - avail_11)
                side_need_23 = max(0, total_23_needed - avail_23)
                side_need_22 = max(0, (b22 + 1) - avail_22)
                bridge_need_11 = max(0, 2 - avail_11)
                bridge_need_23 = max(0, total_23_needed - avail_23)
                bridge_need_22 = max(0, b22 - avail_22)
                side_gap = side_need_11 + side_need_23 + side_need_22
                bridge_gap = bridge_need_11 + bridge_need_23 + bridge_need_22
                if side_gap <= bridge_gap:
                    self._accumulate_shortage(side_need_11, side_need_23, side_need_22)
                else:
                    self._accumulate_shortage(bridge_need_11, bridge_need_23, bridge_need_22)
                remaining_E = prev_E
                remaining_G = prev_G

        # ===== Step 5: Single-branch allocation for 25 (放在 24 环/支链之后) =====
        su25_chains, avail_11, avail_23, avail_22, avail_25 = self._allocate_su25_only(
            avail_11, avail_23, avail_22, avail_25=avail_25, excluded_ids=selected_special_ids
        )
        chains.extend(list(su25_chains))


        return chains, avail_11, avail_23, avail_22, avail_24, avail_25, remaining_E, remaining_G, remaining_B, remaining_D

    # ---------- Phase 4: Build chains ----------
    def _build_closed_chains(self) -> List[ChainSpec]:
        chains = []
        for sp in list(getattr(self, '_special_bridge_patterns', []) or []):
            chains.append(ChainSpec(
                str(sp.chain_type),
                list(sp.composition),
                str(sp.origin_type),
                [int(x) for x in list(sp.source_ids or [])],
                metadata=dict(sp.metadata or {}),
            ))
        # Type A: 11-23-11 (bridge)
        for n in self._type_lists['A']:
            chains.append(ChainSpec('bridge', [11, 23, 11], 'A', [n.global_id]))
        # Type C: 11-23-22 (side)
        for n in self._type_lists['C']:
            chains.append(ChainSpec('side', [11, 23, 22], 'C', [n.global_id]))
        # Type F: 11-22 (side)
        for n in self._type_lists['F']:
            chains.append(ChainSpec('side', [11, 22], 'F', [n.global_id]))
        return chains

    def _build_special_branch_patterns(self) -> List[ChainSpec]:
        chains = []
        for sp in list(getattr(self, '_special_branch_patterns', []) or []):
            chains.append(ChainSpec(
                str(sp.chain_type),
                list(sp.composition),
                str(sp.origin_type),
                [int(x) for x in list(sp.source_ids or [])],
                metadata=dict(sp.metadata or {}),
            ))
        return chains

    @staticmethod
    def _select_feasible_special_branch_patterns(chains: List[ChainSpec],
                                                 avail_11: int,
                                                 avail_23: int,
                                                 avail_22: int,
                                                 avail_24: int,
                                                 avail_25: int) -> Tuple[List[ChainSpec], int, int, int, int, int]:
        selected: List[ChainSpec] = []
        a11 = int(avail_11)
        a23 = int(avail_23)
        a22 = int(avail_22)
        a24 = int(avail_24)
        a25 = int(avail_25)
        for ch in list(chains or []):
            req = _chain_required_resource_counts(ch)
            need11 = int(req.get('11', 0))
            need23 = int(req.get('23', 0))
            need22 = int(req.get('22', 0))
            need24 = int(req.get('24', 0))
            need25 = int(req.get('25', 0))
            if (
                int(a11) >= int(need11) and
                int(a23) >= int(need23) and
                int(a22) >= int(need22) and
                int(a24) >= int(need24) and
                int(a25) >= int(need25)
            ):
                selected.append(ch)
                a11 -= int(need11)
                a23 -= int(need23)
                a22 -= int(need22)
                a24 -= int(need24)
                a25 -= int(need25)
        return selected, a11, a23, a22, a24, a25

    def _allocate_open_chains(self,
                              avail_11: int,
                              avail_23: int,
                              avail_22: int,
                              remaining_E: Optional[List[SUNode]] = None,
                              remaining_G: Optional[List[SUNode]] = None,
                              remaining_B: Optional[List[SUNode]] = None,
                              remaining_D: Optional[List[SUNode]] = None):
        chains = []
        remaining_E = list(remaining_E if remaining_E is not None else self._type_lists['E'])
        remaining_G = list(remaining_G if remaining_G is not None else self._type_lists['G'])
        remaining_B = list(remaining_B if remaining_B is not None else self._type_lists['B'])
        remaining_D = list(remaining_D if remaining_D is not None else self._type_lists['D'])

        # ----- Type B: 11-23-... -----
        for n in remaining_B:
            if remaining_E and self._has_resources(avail_11, avail_23, avail_22, need_11=1, need_23=3, need_22=1):
                e = remaining_E.pop(0)
                # B(11-23-...) + E(22-23-23-...) = 11-23-23-23-22
                comp = [11, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'B+E', [n.global_id, e.global_id]))
                avail_11 -= 1; avail_23 -= 3; avail_22 -= 1
            elif remaining_G and self._has_resources(avail_11, avail_23, avail_22, need_11=1, need_23=2, need_22=1):
                g = remaining_G.pop(0)
                # B(11-23-...) + G(22-23-...) = 11-23-23-22
                comp = [11, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'B+G', [n.global_id, g.global_id]))
                avail_11 -= 1; avail_23 -= 2; avail_22 -= 1
            elif avail_11 >= 2 and avail_23 >= 2:
                # Prefer the shorter bridge closure first to preserve 23 for
                # later branch/flex growth phases.
                comp = [11, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'B', [n.global_id]))
                avail_11 -= 2; avail_23 -= 2
            else:
                print(f"  [WARN] Cannot close Type B node {n.global_id}, insufficient resources")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 1

        # ----- Type D: ...-23-23-23-... -----
        for n in remaining_D:
            if remaining_E and self._has_resources(avail_11, avail_23, avail_22, need_11=1, need_23=5, need_22=1):
                e = remaining_E.pop(0)
                comp = [11, 23, 23, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'D+E', [n.global_id, e.global_id]))
                avail_11 -= 1; avail_23 -= 5; avail_22 -= 1
            elif remaining_G and self._has_resources(avail_11, avail_23, avail_22, need_11=1, need_23=4, need_22=1):
                g = remaining_G.pop(0)
                # 11 + D(23-23-23) + G(23-22) = 11-23-23-23-23-22
                comp = [11, 23, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'D+G', [n.global_id, g.global_id]))
                avail_11 -= 1; avail_23 -= 4; avail_22 -= 1
            elif avail_11 >= 2 and avail_23 >= 3:
                # Prefer the shorter D bridge closure first to preserve 23 for
                # later water-filling/branch reuse.
                comp = [11, 23, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'D', [n.global_id]))
                avail_11 -= 2; avail_23 -= 3
            else:
                print(f"  [WARN] Cannot close Type D node {n.global_id}")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 2

        if min(int(avail_11), int(avail_23), int(avail_22)) < 0:
            raise RuntimeError(
                f"Open-chain allocation underflow: 11={avail_11}, 23={avail_23}, 22={avail_22}"
            )

        # Remaining Type E/G are reserved for branch terminal sealing in Phase 4.5.
        return chains, avail_11, avail_23, avail_22, remaining_E, remaining_G

    # ---------- Phase 5: Allocate remaining (balanced) ----------
    def _allocate_remaining(self, avail_11: int, avail_23: int, avail_22: int) -> Tuple[List[ChainSpec], int]:
        chains = []
        avail_11 = max(0, int(avail_11))
        avail_23 = max(0, int(avail_23))
        avail_22 = max(0, int(avail_22))
        # 先把 side/bridge 两类 extra 都“种出来”，再把剩余 23 交给水填充做均衡注水。
        # 这样可避免一开始就把绝大多数 23 吃进长 bridge extra，而 side extra 只剩 11-22。

        # 1) 先创建全部可行的 extra side 基础链：11-22
        side_count = min(avail_11, avail_22)
        for _ in range(int(side_count)):
            chains.append(ChainSpec('side', [11, 22], 'extra'))
        avail_11 -= int(side_count)
        avail_22 -= int(side_count)

        # 2) 再创建 extra bridge 的“种子”链，但只给最小种子长度，保留更多 23 给 Phase 5b 均衡注水
        bridge_slots = int(avail_11 // 2)
        if bridge_slots > 0 and avail_23 > 0:
            if avail_23 >= 2:
                seed_len = 2
                actual_bridges = min(int(bridge_slots), int(avail_23 // 2))
            else:
                seed_len = 1
                actual_bridges = min(int(bridge_slots), int(avail_23))

            for _ in range(int(actual_bridges)):
                comp = [11] + [23] * int(seed_len) + [11]
                chains.append(ChainSpec('bridge', comp, 'extra'))
            avail_11 -= int(actual_bridges) * 2
            avail_23 -= int(actual_bridges) * int(seed_len)

        # 3) 若还没有任何 bridge extra，但又还有桥资源，则补最短 bridge 种子
        if not any(ch.origin_type == 'extra' and ch.chain_type == 'bridge' for ch in chains):
            while avail_11 >= 2 and avail_23 >= 1:
                chains.append(ChainSpec('bridge', [11, 23, 11], 'extra'))
                avail_11 -= 2
                avail_23 -= 1

        return chains, avail_23

    def _prepare_branch_phase_resources(self) -> Dict[str, Any]:
        prep_diag = self._prepare_allocation_state(emit_logs=False)

        r = self._result
        closed = self._build_closed_chains()
        c11 = sum(c.n_11 for c in closed)
        c23 = sum(c.n_23 for c in closed)
        c22 = sum(c.n_22 for c in closed)

        avail_11 = r.total_11 - c11
        avail_23 = r.total_23 - c23
        avail_22 = r.total_22 - c22
        remaining_E = list(self._type_lists['E'])
        remaining_G = list(self._type_lists['G'])

        r.unallocated_bridge = 0
        r.unallocated_branch = 0
        r.required_extra_11 = 0
        r.required_extra_22 = 0
        r.required_extra_23 = 0

        return {
            'closed_chains': closed,
            'open_chains': [],
            'special_bridge_patterns': list(self._special_bridge_patterns),
            'special_branch_patterns': list(self._special_branch_patterns),
            'unsupported_special_count': int(prep_diag.get('unsupported_special_count', 0)),
            'unsupported_special_blocked_count': int(prep_diag.get('unsupported_special_blocked_count', 0)),
            'unsupported_special_nodes': list(prep_diag.get('unsupported_special_nodes', [])),
            'remaining_E': remaining_E,
            'remaining_G': remaining_G,
            'remaining_B': list(self._type_lists['B']),
            'remaining_D': list(self._type_lists['D']),
            'closed_consumed': {'11': c11, '23': c23, '22': c22},
            'open_consumed': {'11': 0, '23': 0, '22': 0},
            'pre_branch_available': {'11': avail_11, '23': avail_23, '22': avail_22},
            'pre_branch_bridge_diag': {
                'unallocated_bridge': 0,
                'req_11': 0,
                'req_22': 0,
                'req_23': 0,
            }
        }

    # ---------- Phase 5b: Redistribute excess 23 into existing chains ----------
    @staticmethod
    def _redistribute_excess_23(all_chains: List[ChainSpec], excess_23: int) -> int:
        """Push excess 23s into existing B, D, B+G, branch, and extra type chains using water-filling.
        Preserves most type B chains for vertical connections.
        Returns remaining excess that could not be placed."""
        if excess_23 <= 0:
            return 0
        cap = MAX_23_PER_CHAIN
        # Eligible chains: origin_type in B, D, B+G, extra, and linear branches
        expandable_types = {'B', 'D', 'B+G', 'extra'}
        expandable_chain_types = {
            'branch_side', 'branch_bridge',
            'vertical_ring', 'side_ring', 'fused_side_ring',
        }
        
        eligible = []
        b_chains = []
        for ch in all_chains:
            if ch.origin_type in expandable_types or ch.chain_type in expandable_chain_types:
                if ch.origin_type == 'B' and ch.chain_type == 'bridge':
                    b_chains.append(ch)
                else:
                    eligible.append(ch)
        
        # Keep ~75% of B chains at their original length for vertical
        # connections. Use a stable order to keep allocation deterministic.
        b_chains.sort(key=lambda ch: (
            int(getattr(ch, 'n_23', 0)),
            len(list(getattr(ch, 'source_ids', []) or [])),
            tuple(int(x) for x in list(getattr(ch, 'source_ids', []) or [])),
            str(getattr(ch, 'origin_type', '')),
        ))
        num_b_to_expand = int(len(b_chains) * 0.25)
        eligible.extend(b_chains[:num_b_to_expand])

        if not eligible:
            return excess_23

        def priority_key(ch: ChainSpec) -> Tuple[int, int, int]:
            # extra bridge / extra side 采用统一的“当前 n23 更短者优先”策略，
            # tie 时轻微偏向 side，避免 bridge 独占所有剩余 23。
            if ch.origin_type == 'extra' and ch.chain_type == 'side':
                return (0, ch.n_23, 0)
            if ch.origin_type == 'extra' and ch.chain_type == 'bridge':
                return (0, ch.n_23, 1)
            return (4, ch.n_23, 2)

        # Water-filling: iteratively add 1 to the shortest fillable chain
        while excess_23 > 0:
            fillable = [
                ch for ch in eligible
                if ch.n_23 < cap and FlexAllocator._can_expand_with_one_23(ch)
            ]
            if not fillable:
                break

            best_key = min(priority_key(ch) for ch in fillable)
            targets = [ch for ch in fillable if priority_key(ch) == best_key]
            
            if excess_23 < len(targets):
                targets = sorted(
                    targets,
                    key=lambda ch: (
                        int(getattr(ch, 'n_23', 0)),
                        len(list(getattr(ch, 'source_ids', []) or [])),
                        tuple(int(x) for x in list(getattr(ch, 'source_ids', []) or [])),
                        str(getattr(ch, 'origin_type', '')),
                    ),
                )[:int(excess_23)]

            progressed = False
            for ch in targets:
                if FlexAllocator._add_one_23_to_chain(ch):
                    excess_23 -= 1
                    progressed = True
            if not progressed:
                break
        return excess_23

    @staticmethod
    def _collect_extra_chain_metrics(extra: List[ChainSpec]) -> Dict[str, Any]:
        extra_bridges = [c for c in extra if c.origin_type == 'extra' and c.chain_type == 'bridge']
        extra_sides = [c for c in extra if c.origin_type == 'extra' and c.chain_type == 'side']
        short_bridge_count = sum(
            1 for c in extra_bridges
            if c.n_11 == 2 and c.n_23 in (1, 2)
        )
        exact_11_22_count = sum(
            1 for c in extra_sides
            if c.n_11 == 1 and c.n_22 == 1 and c.n_23 == 0
        )
        side_to_22_count = sum(
            1 for c in extra_sides
            if c.n_11 == 1 and c.n_22 == 1
        )
        bridge_avg_23 = 0.0
        if extra_bridges:
            bridge_avg_23 = float(sum(c.n_23 for c in extra_bridges)) / float(len(extra_bridges))
        long_23_chains = sum(
            1 for c in extra
            if c.origin_type == 'extra' and c.n_23 >= 6
        )
        return {
            'extra_short_bridge_count': int(short_bridge_count),
            'extra_11_23_11_count': int(short_bridge_count),
            'extra_11_22_count': int(exact_11_22_count),
            'extra_side_to_22_count': int(side_to_22_count),
            'extra_bridge_avg_23': float(bridge_avg_23),
            'extra_long_23_chains': int(long_23_chains),
        }

    # ---------- Incremental evaluation for Layer4 skeleton adjustment ----------

    def evaluate_su25_only(self, nodes: List) -> Dict[str, Any]:
        """
        Evaluate resource allocation focusing on SU25 allocation (Step 0).
        Evaluates based on the actual node list to accurately reflect resource needs based on aromatic/aliphatic types.
        """
        result = {
            'ok': True,
            'unallocated_25': 0,
            'shortage_type': 'none',
            'avail_after_25': {'11': 0, '22': 0, '23': 0},
            'pre_branch_available': {'11': 0, '22': 0, '23': 0},
            'closed_consumed': {'11': 0, '22': 0, '23': 0},
            'open_consumed': {'11': 0, '22': 0, '23': 0},
            'pre_branch_bridge_diag': {'unallocated_bridge': 0, 'req_11': 0, 'req_22': 0, 'req_23': 0},
            'total_25': 0,
            'consumed_25': 0,
            'req_22': 0,
            'req_11': 0,
            'req_23': 0,
            'unsupported_special_count': 0,
            'unsupported_special_blocked_count': 0,
            'unsupported_special_nodes': [],
            'unsupported_special_reasons': {},
            'branch_chains': [],
        }

        temp_allocator = FlexAllocator(nodes=nodes)
        prep = temp_allocator._prepare_branch_phase_resources()
        result['closed_consumed'] = dict(prep['closed_consumed'])
        result['open_consumed'] = dict(prep['open_consumed'])
        result['pre_branch_available'] = dict(prep['pre_branch_available'])
        result['pre_branch_bridge_diag'] = dict(prep['pre_branch_bridge_diag'])
        result['unsupported_special_count'] = int(prep.get('unsupported_special_count', 0))
        result['unsupported_special_blocked_count'] = int(prep.get('unsupported_special_blocked_count', 0))
        result['unsupported_special_nodes'] = list(prep.get('unsupported_special_nodes', []))
        result['unsupported_special_reasons'] = dict(temp_allocator._result.unsupported_special_reasons or {})
        result['total_25'] = int(temp_allocator._result.total_25)

        if temp_allocator._result.total_25 == 0:
            result['avail_after_25'] = dict(prep['pre_branch_available'])
            return result

        avail = prep['pre_branch_available']
        chains, rem_11, rem_23, rem_22, _ = temp_allocator._allocate_su25_only(
            avail['11'], avail['23'], avail['22']
        )
        res = temp_allocator._result
        result['branch_chains'] = list(chains)
        result['consumed_25'] = sum(c.n_25 for c in chains)
        result['unallocated_25'] = int(res.unallocated_branch)
        result['avail_after_25'] = {'11': rem_11, '22': rem_22, '23': rem_23}
        result['req_22'] = int(res.required_extra_22)
        result['req_11'] = int(res.required_extra_11)
        result['req_23'] = int(res.required_extra_23)

        if result['unallocated_25'] > 0:
            result['ok'] = False
            if int(result.get('unsupported_special_blocked_count', 0)) > 0:
                result['shortage_type'] = 'unsupported_special_topology'
            elif result['req_22'] > 0:
                result['shortage_type'] = '22_shortage'
            elif result['req_11'] > 0:
                result['shortage_type'] = '11_shortage'
            elif result['req_23'] > 0:
                result['shortage_type'] = '23_shortage'
            else:
                result['shortage_type'] = 'general_shortage'
        
        return result

    def evaluate_su24_branches(self, nodes: List, quiet: bool = False) -> Dict[str, Any]:
        """
        Evaluate full resource allocation focusing on SU24 branch allocation (Steps 1-4).
        
        Runs the complete allocate() pipeline and returns diagnostics about
        24 allocation success, branch type breakdown, and resource bottlenecks.
        
        Args:
            nodes: _NodeV3 list for topology-aware classification
            
        Returns dict with:
          - 'ok': bool, True if all branches allocated
          - 'unallocated_branch': int
          - 'unallocated_bridge': int  
          - 'shortage_type': str
          - 'req_22': int, extra 22 needed
          - 'req_11': int, extra 11 needed
          - 'req_23': int, extra 23 needed
          - 'remaining': dict with remaining 11/22/23/24/25
          - 'type_counts': dict of ABCD type counts for 24
          - 'branch_chains': list of chain specs
          - 'extra_11_23_11_count': int
          - 'extra_11_22_count': int
          - 'extra_long_23_chains': int
          - 'alloc_result': AllocationResult
        """
        result = {
            'ok': True,
            'unallocated_branch': 0,
            'unallocated_bridge': 0,
            'shortage_type': 'none',
            'req_22': 0,
            'req_11': 0,
            'req_23': 0,
            'remaining': {},
            'type_counts': {},
            'branch_chains': [],
            'extra_11_23_11_count': 0,
            'extra_11_22_count': 0,
            'extra_long_23_chains': 0,
            'unsupported_special_count': 0,
            'unsupported_special_blocked_count': 0,
            'unsupported_special_nodes': [],
            'unsupported_special_reasons': {},
            'alloc_result': None,
        }
        
        try:
            allocator = FlexAllocator(nodes=nodes)
            prep = allocator._prepare_branch_phase_resources()

            result['type_counts'] = {
                '24_A': len(allocator._type_lists.get('24_A', [])),
                '24_B': len(allocator._type_lists.get('24_B', [])),
                '24_C': len(allocator._type_lists.get('24_C', [])),
                '24_D': len(allocator._type_lists.get('24_D', [])),
                '25_aro': len(allocator._type_lists.get('25_aro', [])),
                '25_ali': len(allocator._type_lists.get('25_ali', [])),
            }
            result['closed_consumed'] = dict(prep['closed_consumed'])
            result['open_consumed'] = dict(prep['open_consumed'])
            result['pre_branch_available'] = dict(prep['pre_branch_available'])
            result['pre_branch_bridge_diag'] = dict(prep['pre_branch_bridge_diag'])
            result['unsupported_special_count'] = int(prep.get('unsupported_special_count', 0))
            result['unsupported_special_blocked_count'] = int(prep.get('unsupported_special_blocked_count', 0))
            result['unsupported_special_nodes'] = list(prep.get('unsupported_special_nodes', []))
            result['unsupported_special_reasons'] = dict(allocator._result.unsupported_special_reasons or {})

            avail_11 = prep['pre_branch_available']['11']
            avail_23 = prep['pre_branch_available']['23']
            avail_22 = prep['pre_branch_available']['22']

            if bool(quiet):
                with redirect_stdout(io.StringIO()):
                    branch_chains, rem_11, rem_23, rem_22, rem_24, rem_25, _rem_E, _rem_G, _rem_B, _rem_D = allocator._allocate_branches(
                        avail_11, avail_23, avail_22,
                        prep.get('remaining_E', []),
                        prep.get('remaining_G', []),
                        prep.get('remaining_B', []),
                        prep.get('remaining_D', []),
                    )
            else:
                branch_chains, rem_11, rem_23, rem_22, rem_24, rem_25, _rem_E, _rem_G, _rem_B, _rem_D = allocator._allocate_branches(
                    avail_11, avail_23, avail_22,
                    prep.get('remaining_E', []),
                    prep.get('remaining_G', []),
                    prep.get('remaining_B', []),
                    prep.get('remaining_D', []),
                )

            res = allocator._result
            result['alloc_result'] = res
            result['unallocated_branch'] = res.unallocated_branch
            result['unallocated_bridge'] = res.unallocated_bridge
            result['req_22'] = res.required_extra_22
            result['req_11'] = res.required_extra_11
            result['req_23'] = res.required_extra_23
            result['remaining'] = {
                '11': rem_11,
                '22': rem_22,
                '23': rem_23,
                '24': rem_24,
                '25': rem_25,
            }
            result['branch_chains'] = list(branch_chains)
            
            if res.unallocated_branch > 0:
                result['ok'] = False
                if int(result.get('unsupported_special_blocked_count', 0)) > 0:
                    result['shortage_type'] = 'unsupported_special_topology'
                elif res.required_extra_22 > 0 or res.required_extra_11 > 0 or res.required_extra_23 > 0:
                    max_req = max(res.required_extra_22, res.required_extra_11, res.required_extra_23)
                    if res.required_extra_22 == max_req:
                        result['shortage_type'] = '22_shortage'
                    elif res.required_extra_11 == max_req:
                        result['shortage_type'] = '11_shortage'
                    else:
                        result['shortage_type'] = '23_shortage'
                elif res.required_extra_22 > 0:
                    result['shortage_type'] = '22_shortage'
                elif rem_11 <= 0 and rem_23 > 0:
                    result['shortage_type'] = '11_shortage'
                    result['req_11'] = 1
                elif rem_23 <= 0:
                    result['shortage_type'] = '23_shortage'
                    result['req_23'] = 1
                else:
                    result['shortage_type'] = 'general_shortage'
                
        except Exception as e:
            print(f"  [evaluate_su24_branches Error] {e}")
            import traceback
            traceback.print_exc()
            result['ok'] = False
            result['shortage_type'] = '11_shortage' if 'underflow: 11=' in str(e) else 'error'
            result['error'] = str(e)
            if 'underflow: 11=' in str(e):
                result['req_11'] = max(int(result.get('req_11', 0)), 1)
            if 'underflow: 23=' in str(e):
                result['req_23'] = max(int(result.get('req_23', 0)), 1)
            if 'underflow: 22=' in str(e):
                result['req_22'] = max(int(result.get('req_22', 0)), 1)
        
        return result

    def evaluate_extra_allocation(
        self,
        nodes: List,
        short_bridge_threshold: int = 8,
        min_side_to_22: int = 5,
    ) -> Dict[str, Any]:
        result = {
            'ok': True,
            'short_bridge_threshold': int(short_bridge_threshold),
            'min_side_to_22': int(min_side_to_22),
            'extra_short_bridge_count': 0,
            'extra_11_23_11_count': 0,
            'extra_11_22_count': 0,
            'extra_side_to_22_count': 0,
            'extra_bridge_avg_23': 0.0,
            'extra_long_23_chains': 0,
            'unallocated_bridge': 0,
            'unallocated_branch': 0,
            'required_extra_11': 0,
            'required_extra_22': 0,
            'required_extra_23': 0,
            'unsupported_special_count': 0,
            'unsupported_special_blocked_count': 0,
            'unsupported_special_nodes': [],
            'unsupported_special_reasons': {},
            'remaining': {},
            'bridge_chains': [],
            'side_chains': [],
            'branch_chains': [],
            'alloc_result': None,
            'reason': 'ok',
        }
        try:
            allocator = FlexAllocator(nodes=nodes)
            with redirect_stdout(io.StringIO()):
                alloc_res = allocator.allocate()
            result['alloc_result'] = alloc_res
            result['bridge_chains'] = list(alloc_res.bridge_chains)
            result['side_chains'] = list(alloc_res.side_chains)
            result['branch_chains'] = list(alloc_res.branch_chains)
            result['remaining'] = {
                '11': int(alloc_res.remaining_11),
                '22': int(alloc_res.remaining_22),
                '23': int(alloc_res.remaining_23),
                '24': int(alloc_res.remaining_24),
                '25': int(alloc_res.remaining_25),
            }
            result['unallocated_bridge'] = int(alloc_res.unallocated_bridge)
            result['unallocated_branch'] = int(alloc_res.unallocated_branch)
            result['required_extra_11'] = int(alloc_res.required_extra_11)
            result['required_extra_22'] = int(alloc_res.required_extra_22)
            result['required_extra_23'] = int(alloc_res.required_extra_23)
            result['unsupported_special_count'] = int(getattr(alloc_res, 'unsupported_special_count', 0))
            result['unsupported_special_blocked_count'] = int(getattr(alloc_res, 'unsupported_special_blocked_count', 0))
            result['unsupported_special_nodes'] = list(getattr(alloc_res, 'unsupported_special_nodes', []) or [])
            result['unsupported_special_reasons'] = dict(getattr(alloc_res, 'unsupported_special_reasons', {}) or {})
            result['extra_short_bridge_count'] = int(getattr(alloc_res, 'extra_short_bridge_count', 0))
            result['extra_11_23_11_count'] = int(getattr(alloc_res, 'extra_11_23_11_count', 0))
            result['extra_11_22_count'] = int(getattr(alloc_res, 'extra_11_22_count', 0))
            result['extra_side_to_22_count'] = int(getattr(alloc_res, 'extra_side_to_22_count', 0))
            result['extra_bridge_avg_23'] = float(getattr(alloc_res, 'extra_bridge_avg_23', 0.0))
            result['extra_long_23_chains'] = int(getattr(alloc_res, 'extra_long_23_chains', 0))
            bad_short_bridge = int(result['extra_short_bridge_count']) >= int(short_bridge_threshold)
            bad_side_count = int(result['extra_side_to_22_count']) < int(min_side_to_22)
            if bad_short_bridge and bad_side_count:
                result['ok'] = False
                result['reason'] = 'short_extra_bridges_and_few_side_22'
        except Exception as e:
            result['ok'] = False
            result['reason'] = 'error'
            result['error'] = str(e)
        return result

    # ---------- Main entry ----------
    def allocate(self) -> AllocationResult:
        print("=" * 60)
        print("[FlexAllocator] Starting resource allocation")
        print("=" * 60)

        # Phase 1
        self._prepare_allocation_state(emit_logs=True)

        # Phase 2
        r = self._result
        print(f"\n  [Phase 2] SU Conversion Totals:")
        print(f"    11 (effective aromatic endpoints): {r.total_11}")
        print(f"    23 (effective chain body):        {r.total_23}")
        print(f"    22 (effective terminals):         {r.total_22}")
        print(f"    24 (effective branch CH):         {r.total_24}")
        print(f"    25 (effective branch Cq):         {r.total_25}")

        # Phase 3a: Classify bridge/terminal SUs
        self._classify_all()
        print(f"\n  [Phase 3a] Bridge/Terminal Classification:")
        for t in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            lst = self._type_lists[t]
            if lst:
                ids = [n.global_id for n in lst]
                print(f"    Type {t:>3}: {len(lst):>3} nodes  ids={ids[:10]}{'...' if len(ids) > 10 else ''}")
            r.type_counts[t] = len(lst)

        # Phase 3b: Classify branch 24/25
        self._classify_branch_24()
        self._classify_branch_25()
        print(f"\n  [Phase 3b] Branch 24/25 Classification:")
        for t in ['24_A', '24_B', '24_C', '24_D', '25_aro', '25_ali']:
            lst = self._type_lists.get(t, [])
            if lst:
                ids = [n.global_id for n in lst]
                print(f"    Type {t:>5}: {len(lst):>3} nodes  ids={ids[:10]}{'...' if len(ids) > 10 else ''}")
            r.type_counts[t] = len(lst)
        if self._special_branch_patterns:
            print(f"    Special branch/ring seeds: {len(self._special_branch_patterns)}")

        # Phase 4a: closed chains
        closed = self._build_closed_chains()
        closed_req = _sum_chain_required_resource_counts(closed)
        c11 = int(closed_req['11'])
        c23 = int(closed_req['23'])
        c22 = int(closed_req['22'])
        print(f"\n  [Phase 4a] Closed chains: {len(closed)}")
        print(f"    Consumed: 11×{c11}, 23×{c23}, 22×{c22}")

        avail_11 = r.total_11 - c11
        avail_23 = r.total_23 - c23
        avail_22 = r.total_22 - c22
        print(f"    Available after closed: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        remaining_E = list(self._type_lists['E'])
        remaining_G = list(self._type_lists['G'])

        # Phase 4b: Branch allocation (24/25) first
        branch_chains, avail_11, avail_23, avail_22, avail_24, avail_25, remaining_E, remaining_G, remaining_B, remaining_D = \
            self._allocate_branches(
                avail_11, avail_23, avail_22,
                remaining_E, remaining_G,
                list(self._type_lists['B']),
                list(self._type_lists['D']),
            )
        branch_req = _sum_chain_required_resource_counts(branch_chains)
        br11 = int(branch_req['11'])
        br23 = int(branch_req['23'])
        br22 = int(branch_req['22'])
        br24 = int(branch_req['24'])
        br25 = int(branch_req['25'])
        print(f"\n  [Phase 4b] Branch allocation (24/25): {len(branch_chains)}")
        print(f"    Consumed: 11×{br11}, 23×{br23}, 22×{br22}, 24×{br24}, 25×{br25}")
        print(f"    Available after branch: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}, 24×{avail_24}, 25×{avail_25}")

        # Phase 4c: open chains using leftover E/G after branches
        open_chains, avail_11, avail_23, avail_22, remaining_E, remaining_G = self._allocate_open_chains(
            avail_11, avail_23, avail_22, remaining_E, remaining_G, remaining_B, remaining_D
        )
        all_chains = closed + open_chains
        open_req = _sum_chain_required_resource_counts(open_chains)
        o11 = int(open_req['11'])
        o23 = int(open_req['23'])
        o22 = int(open_req['22'])
        print(f"\n  [Phase 4c] Open chain allocation: {len(open_chains)}")
        print(f"    Consumed: 11×{o11}, 23×{o23}, 22×{o22}")
        print(f"    Available after open: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        reserved_sides, avail_11, avail_23, avail_22 = self._allocate_reserved_terminal_sides(
            remaining_E, remaining_G, avail_11, avail_23, avail_22
        )
        all_chains += reserved_sides
        reserved_req = _sum_chain_required_resource_counts(reserved_sides)
        rs11 = int(reserved_req['11'])
        rs23 = int(reserved_req['23'])
        rs22 = int(reserved_req['22'])
        print(f"\n  [Phase 4d] Reserved E/G side chains: {len(reserved_sides)}")
        print(f"    Consumed: 11×{rs11}, 23×{rs23}, 22×{rs22}")
        print(f"    Available after reserved tails: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        # Phase 5: remaining (create base extra chains)
        extra, leftover_23 = self._allocate_remaining(avail_11, avail_23, avail_22)
        all_chains += extra
        extra_req = _sum_chain_required_resource_counts(extra)
        e11 = int(extra_req['11'])
        e23 = int(extra_req['23'])
        e22 = int(extra_req['22'])
        print(f"\n  [Phase 5] Extra chains from remaining: {len(extra)}")
        print(f"    Consumed: 11×{e11}, 23×{e23}, 22×{e22}")
        if leftover_23 > 0:
            print(f"    Leftover 23: {leftover_23}")
            
        pre_waterfill_extra_metrics = self._collect_extra_chain_metrics(extra)
        print(
            f"    [Phase 5 初始extra] short={pre_waterfill_extra_metrics['extra_short_bridge_count']} "
            f"side22={pre_waterfill_extra_metrics['extra_side_to_22_count']} "
            f"avg23={pre_waterfill_extra_metrics['extra_bridge_avg_23']:.2f}"
        )

        # Phase 5b: Redistribute excess 23 into existing expandable chains using water-filling
        if leftover_23 > 0:
            # Include branch_chains so branches can grow too
            still_left = self._redistribute_excess_23(all_chains + branch_chains, leftover_23)
            redistributed = leftover_23 - still_left
            print(f"\n  [Phase 5b] Redistributed {redistributed} excess 23s into expandable chains (water-filling)")
            if still_left > 0:
                print(f"    Still unplaced 23s: {still_left}")
            
            # Recalculate consumed 23s for all categories since they might have changed
            c23 = int(_sum_chain_required_resource_counts(closed)['23'])
            o23 = int(_sum_chain_required_resource_counts(open_chains)['23'])
            e23 = int(_sum_chain_required_resource_counts(extra)['23'])
            br23 = int(_sum_chain_required_resource_counts(branch_chains)['23'])
            
            leftover_23 = still_left

        # initial Phase 5 seed chains.
        final_extra_metrics = self._collect_extra_chain_metrics(all_chains)
        r.extra_11_23_11_count = int(final_extra_metrics['extra_11_23_11_count'])
        r.extra_11_22_count = int(final_extra_metrics['extra_11_22_count'])
        r.extra_short_bridge_count = int(final_extra_metrics['extra_short_bridge_count'])
        r.extra_side_to_22_count = int(final_extra_metrics['extra_side_to_22_count'])
        r.extra_bridge_avg_23 = float(final_extra_metrics['extra_bridge_avg_23'])
        r.extra_long_23_chains = int(final_extra_metrics['extra_long_23_chains'])

        # Build result
        final_chains = list(all_chains) + list(branch_chains)
        final_req = _sum_chain_required_resource_counts(final_chains)
        r.consumed_11 = int(final_req['11'])
        r.consumed_23 = int(final_req['23'])
        r.consumed_22 = int(final_req['22'])
        r.consumed_24 = int(final_req['24'])
        r.consumed_25 = int(final_req['25'])
        r.remaining_11 = r.total_11 - r.consumed_11
        r.remaining_23 = r.total_23 - r.consumed_23
        r.remaining_22 = r.total_22 - r.consumed_22
        r.remaining_24 = r.total_24 - r.consumed_24
        r.remaining_25 = r.total_25 - r.consumed_25

        for ch in final_chains:
            self._annotate_chain_sources(ch)
        self._annotate_chain_resource_usage(final_chains)

        for ch in all_chains:
            if ch.chain_type == 'bridge':
                r.bridge_chains.append(ch)
            else:
                r.side_chains.append(ch)
        
        # Integrate branch_bridge and branch_side into flex/side stages
        for ch in branch_chains:
            if ch.chain_type == 'branch_bridge':
                r.bridge_chains.append(ch)
            elif ch.chain_type == 'branch_side':
                r.side_chains.append(ch)
            else:
                # vertical_ring, side_ring stay in branch_chains
                r.branch_chains.append(ch)

        self._print_summary()
        return r

    def _print_summary(self):
        r = self._result
        print("\n" + "=" * 60)
        print("[FlexAllocator] Allocation Summary")
        print("=" * 60)

        def _source_text(ch: ChainSpec) -> str:
            meta = getattr(ch, 'metadata', {}) or {}
            src_su = list(meta.get('source_su_types', []) or [])
            src_hop1 = list(meta.get('source_hop1', []) or [])
            special_pattern = meta.get('special_pattern', None)
            native_counts = dict(meta.get('native_consumed_counts', {}) or {})
            proxy_counts = dict(meta.get('proxy_consumed_counts', {}) or {})
            if not src_su:
                extra = ""
            else:
                extra = f" src_su={src_su}"
                if src_hop1:
                    extra += f" src_hop1={src_hop1}"
            if special_pattern:
                extra += f" pattern={special_pattern}"
            usage_parts = []
            if native_counts:
                usage_parts.append(f"native={native_counts}")
            if proxy_counts:
                usage_parts.append(f"proxy={proxy_counts}")
            if usage_parts:
                extra += f" use[{', '.join(usage_parts)}]"
            return extra

        print(f"  Bridge chains: {len(r.bridge_chains)}")
        for i, ch in enumerate(r.bridge_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, len_23={ch.n_23}){_source_text(ch)}")

        print(f"\n  Side chains: {len(r.side_chains)}")
        for i, ch in enumerate(r.side_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, len_23={ch.n_23}){_source_text(ch)}")

        print(f"\n  Branch structures (24/25): {len(r.branch_chains)}")
        for i, ch in enumerate(r.branch_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            extra = f", 24×{ch.n_24}" if ch.n_24 else ""
            extra += f", 25×{ch.n_25}" if ch.n_25 else ""
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, 23×{ch.n_23}, 22×{ch.n_22}{extra}){_source_text(ch)}")

        print(f"\n  Resource Usage:")
        print(f"    {'':>10} {'Total':>8} {'Consumed':>10} {'Remaining':>10}")
        print(f"    {'SU 11':>10} {r.total_11:>8} {r.consumed_11:>10} {r.remaining_11:>10}")
        print(f"    {'SU 23':>10} {r.total_23:>8} {r.consumed_23:>10} {r.remaining_23:>10}")
        print(f"    {'SU 22':>10} {r.total_22:>8} {r.consumed_22:>10} {r.remaining_22:>10}")
        print(f"    {'SU 24':>10} {r.total_24:>8} {r.consumed_24:>10} {r.remaining_24:>10}")
        print(f"    {'SU 25':>10} {r.total_25:>8} {r.consumed_25:>10} {r.remaining_25:>10}")
        if r.native_total_by_kind or r.proxy_total_by_kind:
            print(f"\n  Native/Proxy Breakdown:")
            for kind in ('11', '22', '23', '24', '25'):
                nt = int((r.native_total_by_kind or {}).get(kind, 0))
                pt = int((r.proxy_total_by_kind or {}).get(kind, 0))
                nc = int((r.native_consumed_by_kind or {}).get(kind, 0))
                pc = int((r.proxy_consumed_by_kind or {}).get(kind, 0))
                nr = int((r.native_remaining_by_kind or {}).get(kind, 0))
                pr = int((r.proxy_remaining_by_kind or {}).get(kind, 0))
                print(
                    f"    kind {kind}: native {nc}/{nt} used, {nr} left | "
                    f"proxy {pc}/{pt} used, {pr} left"
                )

        print(f"\n  Type Distribution:")
        for t, cnt in sorted(r.type_counts.items()):
            if cnt > 0:
                print(f"    Type {t}: {cnt}")

        n_bridge_23 = sum(ch.n_23 for ch in r.bridge_chains)
        n_side_23 = sum(ch.n_23 for ch in r.side_chains)
        n_branch_23 = sum(ch.n_23 for ch in r.branch_chains)
        print(f"\n  Chain Length Stats:")
        print(f"    Bridge:  {len(r.bridge_chains)} chains, total 23s = {n_bridge_23}")
        if r.bridge_chains:
            lens = [ch.n_23 for ch in r.bridge_chains]
            print(f"      lengths(23): {sorted(lens)}")
        print(f"    Side:    {len(r.side_chains)} chains, total 23s = {n_side_23}")
        if r.side_chains:
            lens = [ch.n_23 for ch in r.side_chains]
            print(f"      lengths(23): {sorted(lens)}")
        print(f"    Branch:  {len(r.branch_chains)} structures, total 23s = {n_branch_23}")
        if r.branch_chains:
            lens = [ch.n_23 for ch in r.branch_chains]
            print(f"      lengths(23): {sorted(lens)}")
        print(f"\n  Extra diagnostics:")
        print(f"    extra short bridges (11-23-11 / 11-23-23-11): {r.extra_short_bridge_count}")
        print(f"    extra exact 11-22 sides: {r.extra_11_22_count}")
        print(f"    extra side-to-22 chains (11-...-22): {r.extra_side_to_22_count}")
        print(f"    extra bridge avg 23 length: {r.extra_bridge_avg_23:.2f}")
        print(f"    extra long chains (n_23>=6): {r.extra_long_23_chains}")
        print("=" * 60)


# ==================== Standalone test ====================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        csv_path = 'test_results/1-4/final_outputs/final_nodes.csv'
        print(f"No CSV path given, using default: {csv_path}")
    else:
        csv_path = sys.argv[1]

    allocator = FlexAllocator(csv_path)
    result = allocator.allocate()
