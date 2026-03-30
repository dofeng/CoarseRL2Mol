import sys
import io
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
from contextlib import redirect_stdout

from .RL_init import _parse_template_key

AROMATIC_SET = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
TO_11 = {5, 6, 7, 8, 9, 11}
TO_23 = {0, 2, 3, 15, 17, 19, 20, 21, 23, 27, 29, 31}
TO_24 = {14, 24}
STRUCTURAL_PLACEHOLDER_TO_23 = {14, 24, 25}
TO_22 = {1, 4, 16, 18, 22, 28, 32}
AROMATIC_5_13 = {5, 6, 7, 8, 9, 10, 11, 12, 13}
BRIDGE_SU = {0, 2, 3, 27, 29, 31}
TERMINAL_SU = {1, 4, 28}
MAX_23_PER_CHAIN = 6

# ==================== Data Structures ====================

@dataclass
class SUNode:
    global_id: int
    su_type: int
    hop1: Tuple[int, ...]
    hop2: Tuple[int, ...]

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
        self.n_11 = self.composition.count(11)
        self.n_23 = self.composition.count(23)
        self.n_22 = self.composition.count(22)
        self.n_24 = self.composition.count(24)
        self.n_25 = self.composition.count(25)

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

# ==================== Helper ====================

def _categorize(su_type: int) -> str:
    """Categorize a neighbor SU type."""
    if su_type in AROMATIC_SET:
        return 'aromatic'
    if su_type == 22:
        return 'terminal'
    if su_type in TO_23:
        return 'aliphatic'
    if su_type in TO_22:
        return 'terminal'
    return 'aliphatic'


def count_su_values(values: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for value in values:
        su = int(value)
        counts[su] = counts.get(su, 0) + 1
    return counts


def chain_spec_counts_match(spec: ChainSpec, values: List[int]) -> bool:
    counts = count_su_values(values)
    return (
        int(counts.get(22, 0)) == int(spec.n_22) and
        int(counts.get(23, 0)) == int(spec.n_23) and
        int(counts.get(24, 0)) == int(spec.n_24) and
        int(counts.get(25, 0)) == int(spec.n_25)
    )

# ==================== FlexAllocator ====================

class FlexAllocator:

    def __init__(self, csv_path: Optional[str] = None, su_counts: Optional[Dict[int, int]] = None, nodes: Optional[List] = None):
        self.csv_path = csv_path
        self._nodes: List[SUNode] = []
        self._type_lists: Dict[str, List[SUNode]] = {}
        self._result = AllocationResult()
        self._su_counts = su_counts.copy() if su_counts is not None else {}
        self._input_nodes = nodes

    @staticmethod
    def _refresh_chain_counts(chain: ChainSpec):
        chain.n_11 = chain.composition.count(11)
        chain.n_23 = chain.composition.count(23)
        chain.n_22 = chain.composition.count(22)
        chain.n_24 = chain.composition.count(24)
        chain.n_25 = chain.composition.count(25)

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
    def _add_one_23_to_chain(chain: ChainSpec, branch_slot: Optional[str] = None) -> bool:
        meta = dict(chain.metadata or {})

        if chain.chain_type in ('branch_side', 'branch_bridge'):
            meta['branch_23_count'] = int(meta.get('branch_23_count', 0)) + 1
            insert_idx = FlexAllocator._side_branch_insert_index(chain)
            chain.composition.insert(insert_idx, 23)
            chain.metadata = meta
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
            FlexAllocator._refresh_chain_counts(chain)
            return True

        if chain.chain_type in ('side', 'bridge'):
            chain.composition.insert(max(1, len(chain.composition) - 1), 23)
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
                        
                self._nodes.append(SUNode(int(n.global_id), su_type, tuple(hop1), tuple(hop2)))
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
            raise ValueError("CSV missing template_key column")

        for _, row in df.iterrows():
            gid = int(row.get('global_id', row.name))
            su_type = int(row[su_col])
            _, hop1, hop2 = _parse_template_key(str(row['template_key']))
            self._nodes.append(SUNode(gid, su_type, tuple(hop1), tuple(hop2)))

    # ---------- Phase 2: Convert & Count ----------
    def _convert_and_count(self):
        r = self._result
        r.total_11 = sum(self._su_counts.get(k, 0) for k in TO_11)
        r.total_23 = sum(self._su_counts.get(k, 0) for k in TO_23)
        r.total_22 = sum(self._su_counts.get(k, 0) for k in TO_22)
        r.total_24 = sum(self._su_counts.get(k, 0) for k in TO_24)
        r.total_25 = self._su_counts.get(25, 0)

    # ---------- Phase 3: Classify ----------
    def _classify_all(self):
        for t in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            self._type_lists[t] = []

        for node in self._nodes:
            if node.su_type in BRIDGE_SU:
                self._classify_bridge(node)
            elif node.su_type in TERMINAL_SU:
                self._classify_terminal(node)

    def _classify_bridge(self, node: SUNode):
        if len(node.hop1) < 2:
            return
        cat1 = _categorize(node.hop1[0])
        cat2 = _categorize(node.hop1[1])
        cats = sorted([cat1, cat2])

        key = tuple(cats)
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
        if len(node.hop1) < 1:
            return
        neighbor = node.hop1[0]
        if neighbor in AROMATIC_SET:
            self._type_lists['F'].append(node)
        else:
            self._type_lists['G'].append(node)

    # ---------- Phase 3b: Classify branch 24/25 ----------
    def _classify_branch_24(self):
        """Classify 24-type nodes (from SU 14, 24) into A/B/C/D based on 1-hop."""
        for t in ['24_A', '24_B', '24_C', '24_D']:
            self._type_lists[t] = []
        for node in self._nodes:
            if node.su_type not in (14, 24):
                continue
            has_aro = any(h in AROMATIC_5_13 for h in node.hop1)
            has_22 = 22 in node.hop1
            if has_aro and not has_22:
                self._type_lists['24_A'].append(node)
            elif has_aro and has_22:
                self._type_lists['24_B'].append(node)
            elif not has_aro and not has_22:
                self._type_lists['24_C'].append(node)
            else:
                self._type_lists['24_D'].append(node)

    def _classify_branch_25(self):
        """Classify 25-type nodes based on 1-hop aromatic presence."""
        for t in ['25_aro', '25_ali']:
            self._type_lists[t] = []
        for node in self._nodes:
            if node.su_type != 25:
                continue
            has_aro = any(h in AROMATIC_5_13 for h in node.hop1)
            if has_aro:
                self._type_lists['25_aro'].append(node)
            else:
                self._type_lists['25_ali'].append(node)

    @staticmethod
    def _branch_cost(btype: str) -> Tuple[int, int]:
        """Return (n_23, n_22) for a 24-node's side branch.
        Uses user-specified resources: A/C -> -23-22 (1, 1). B/D -> -22 (0, 1)."""
        if btype in ('24_A', '24_C', '25_aro', '25_ali'):
            return (1, 1)  # -23-22
        else:  # 24_B, 24_D
            return (0, 1)  # -22

    def _consume_reserved_branch_tail(self,
                                      btype: str,
                                      remaining_E: List[SUNode],
                                      remaining_G: List[SUNode]) -> Tuple[int, int, str]:
        """Prefer reserved terminal chains for branch-side ending.
        Returns (n_23, n_22, source_tag).
        """
        if btype in ('24_A', '24_C'):
            if remaining_G:
                remaining_G.pop(0)
                return 1, 1, 'G'
            if remaining_E:
                remaining_E.pop(0)
                return 2, 1, 'E'
        b23, b22 = self._branch_cost(btype)
        return int(b23), int(b22), 'raw'

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

    def _allocate_su25_only(self, avail_11: int, avail_23: int, avail_22: int):
        chains = []
        avail_25 = self._result.total_25
        for t_key in ['25_aro', '25_ali']:
            is_aro = (t_key == '25_aro')
            for n in self._type_lists.get(t_key, []):
                if avail_25 < 1:
                    self._result.unallocated_branch += 1
                    continue

                total_23_needed = 3
                allocated = False
                branch_meta = {
                    'branch_type': t_key,
                    'branch_23_count': 1,
                    'branch_22_count': 1,
                    'extra_22_count': 1,
                }

                if avail_11 >= 1 and avail_23 >= total_23_needed:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    terms, temp_11, temp_22 = self._get_branch_terminals(3, temp_11, temp_22)
                    if len(terms) == 3:
                        if is_aro:
                            comp = [11, 25, 23, 23, terms[0]]
                        else:
                            comp = [11, 23, 25, 23, terms[0]]
                        comp += [23, terms[1], terms[2]]
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
                    terms, temp_11, temp_22 = self._get_branch_terminals(2, temp_11, temp_22)
                    if len(terms) == 2:
                        if is_aro:
                            comp = [11, 25, 23, 23, 11]
                        else:
                            comp = [11, 23, 25, 23, 11]
                        comp += [23] + terms
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
                    side_need_22 = max(0, 3 - avail_22)
                    bridge_need_11 = max(0, 2 - avail_11)
                    bridge_need_23 = max(0, total_23_needed - avail_23)
                    bridge_need_22 = max(0, 2 - avail_22)
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
                           remaining_G: Optional[List[SUNode]] = None):
        """Allocate branch structures for SU 24 and 25.
        Returns: (chains, avail_11, avail_23, avail_22, avail_24, avail_25)
        """
        remaining_E = list(remaining_E or [])
        remaining_G = list(remaining_G or [])
        avail_24 = self._result.total_24
        chains, avail_11, avail_23, avail_22, avail_25 = self._allocate_su25_only(
            avail_11, avail_23, avail_22
        )
        A = [(n, '24_A') for n in self._type_lists.get('24_A', [])]
        B = [(n, '24_B') for n in self._type_lists.get('24_B', [])]
        C = [(n, '24_C') for n in self._type_lists.get('24_C', [])]
        D = [(n, '24_D') for n in self._type_lists.get('24_D', [])]

        AB = A + B

        # ===== Step 1: Fused Side Rings (脂肪并环) =====
        # 脂肪并环中的24号最多6个，最少4个。
        # 需要2个固定C类作为桥头。其他的（至少2个，最多4个）分布在内环和外环的上下位置。
        
        while len(C) >= 2:
            bridge1 = C.pop(0)
            bridge2 = C.pop(0)

            base_nodes = []
            while AB and len(base_nodes) < 2:
                base_nodes.append(AB.pop(0))

            outer_pool = sorted(C + D, key=lambda x: 0 if x[1] == '24_C' else 1)
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

            # Fused side ring always contains:
            #   2 fixed bridgehead 24(C) +
            #   2 outer-inner 23 closure nodes +
            #   4 optional slots (base upper/lower, outer upper/lower)
            # Missing optional 24 slots are filled by 23.
            ring_23_needed = 6 - n_extra_24

            branch_23 = 0
            branch_22 = 0
            tail_sources = []
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            for n in base_nodes + outer_nodes:
                b23, b22, tail_src = self._consume_reserved_branch_tail(n[1], remaining_E, remaining_G)
                branch_23 += b23
                branch_22 += b22
                tail_sources.append(tail_src)

            total_23_needed = ring_23_needed + branch_23

            if avail_11 >= 2 and avail_23 >= total_23_needed and avail_22 >= branch_22 and avail_24 >= nodes_to_consume:
                comp = [11] + [24] * nodes_to_consume + [23] * total_23_needed + [22] * branch_22 + [11]
                ids = [bridge1[0].global_id, bridge2[0].global_id] + [n[0].global_id for n in base_nodes + outer_nodes]
                desc = f"Fused-S-ring(Base:{base_str}+Br:CC)+Out:{out_str}"

                base_upper_type = base_nodes[0][1] if len(base_nodes) > 0 else None
                base_lower_type = base_nodes[1][1] if len(base_nodes) > 1 else None
                outer_upper_type = outer_nodes[0][1] if len(outer_nodes) > 0 else None
                outer_lower_type = outer_nodes[1][1] if len(outer_nodes) > 1 else None

                base_upper_len = sum(self._branch_cost(base_upper_type)) if base_upper_type else 0
                base_lower_len = sum(self._branch_cost(base_lower_type)) if base_lower_type else 0
                outer_upper_len = sum(self._branch_cost(outer_upper_type)) if outer_upper_type else 0
                outer_lower_len = sum(self._branch_cost(outer_lower_type)) if outer_lower_type else 0

                fused_meta = {
                    'tail_sources': tail_sources,
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
                continue

            remaining_E = prev_E
            remaining_G = prev_G
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

        # Re-sync A, B from AB
        A = [x for x in AB if x[1] == '24_A']
        B = [x for x in AB if x[1] == '24_B']
        CD = C + D

        # ===== Step 2: Vertical Rings (上下脂肪环) =====
        a_idx = 0
        while a_idx < len(A) and CD:
            a_node = A[a_idx]
            
            if len(CD) >= 2:
                cd1 = CD.pop(0)
                cd2 = CD.pop(0)
                ring_23 = 3 
                ring_terms = 0
                prev_E = list(remaining_E)
                prev_G = list(remaining_G)
                b23_1, bterms_1, src1 = self._consume_reserved_branch_tail(cd1[1], remaining_E, remaining_G)
                b23_2, bterms_2, src2 = self._consume_reserved_branch_tail(cd2[1], remaining_E, remaining_G)
                ring_23 += b23_1 + b23_2
                ring_terms += bterms_1 + bterms_2
                
                if avail_11 >= 1 and avail_23 >= ring_23 and avail_24 >= 3:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    terms, temp_11, temp_22 = self._get_branch_terminals(ring_terms, temp_11, temp_22)
                    
                    if len(terms) == ring_terms:
                        comp = [11, 24, 23, 24, 23, 24, 23]
                        comp += [23] * (b23_1 + b23_2) + terms
                        ids = [a_node[0].global_id, cd1[0].global_id, cd2[0].global_id]
                        desc = f"V-ring(A+{cd1[1][-1]}+{cd2[1][-1]})"
                        vr_meta = {
                            'tail_sources': [src1, src2],
                            'vertical_inter_types': [cd1[1], cd2[1]],
                            'branch_tail_lengths': {
                                'right': b23_1 + bterms_1,
                                'left': b23_2 + bterms_2,
                            },
                        }
                        chains.append(ChainSpec('vertical_ring', comp, desc, ids, metadata=vr_meta))
                        avail_11 = temp_11; avail_22 = temp_22
                        avail_23 -= ring_23; avail_24 -= 3
                        
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
                ring_23 = 4
                ring_terms = 0
                prev_E = list(remaining_E)
                prev_G = list(remaining_G)
                b23, bterms, src1 = self._consume_reserved_branch_tail(cd1[1], remaining_E, remaining_G)
                ring_23 += b23
                ring_terms += bterms
                
                if avail_11 >= 1 and avail_23 >= ring_23 and avail_24 >= 2:
                    temp_11 = avail_11 - 1
                    temp_22 = avail_22
                    terms, temp_11, temp_22 = self._get_branch_terminals(ring_terms, temp_11, temp_22)
                    
                    if len(terms) == ring_terms:
                        comp = [11, 24, 23, 24, 23, 23, 23]
                        comp += [23] * b23 + terms
                        ids = [a_node[0].global_id, cd1[0].global_id]
                        desc = f"V-ring(A+{cd1[1][-1]})"
                        vr_meta = {
                            'tail_sources': [src1],
                            'vertical_inter_types': [cd1[1], None],
                            'branch_tail_lengths': {
                                'right': b23 + bterms,
                            },
                        }
                        chains.append(ChainSpec('vertical_ring', comp, desc, ids, metadata=vr_meta))
                        avail_11 = temp_11; avail_22 = temp_22
                        avail_23 -= ring_23; avail_24 -= 2
                        
                        A.pop(a_idx)
                        C = [x for x in CD if x[1] == '24_C']
                        D = [x for x in CD if x[1] == '24_D']
                        continue
                remaining_E = prev_E
                remaining_G = prev_G
                CD.insert(0, cd1)
            a_idx += 1

        # ===== Step 3: Side Rings (侧边脂肪环) =====
        AB = A + B
        remaining = AB + CD
        
        while len(remaining) >= 2:
            n1 = remaining.pop(0)
            n2 = remaining.pop(0)
            
            is_ab_1 = n1[1] in ('24_A', '24_B')
            is_ab_2 = n2[1] in ('24_A', '24_B')
            
            ring_body_23 = 2 
            if not is_ab_1: ring_body_23 += 1
            if not is_ab_2: ring_body_23 += 1
            
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            b1_23, b1_22, src1 = self._consume_reserved_branch_tail(n1[1], remaining_E, remaining_G)
            b2_23, b2_22, src2 = self._consume_reserved_branch_tail(n2[1], remaining_E, remaining_G)
            branch_23 = b1_23 + b2_23
            ring_22 = b1_22 + b2_22
            
            total_23 = ring_body_23 + branch_23
            
            if avail_11 >= 2 and avail_23 >= total_23 and avail_22 >= ring_22 and avail_24 >= 2:
                if is_ab_1 and is_ab_2:
                    comp = [11, 24, 23, 23, 24, 11]
                elif not is_ab_1 and not is_ab_2:
                    comp = [11, 23, 24, 24, 23, 11]
                else:
                    comp = [11, 24, 23, 24, 23, 11]
                    
                comp += [23]*branch_23 + [22]*ring_22
                ids = [n1[0].global_id, n2[0].global_id]
                desc = f"S-ring({n1[1][-1]}+{n2[1][-1]})"
                sr_meta = {
                    'tail_sources': [src1, src2],
                    'side_ring_node_types': [n1[1], n2[1]],
                    'branch_tail_lengths': {
                        'upper': b1_23 + b1_22,
                        'lower': b2_23 + b2_22,
                    },
                }
                chains.append(ChainSpec('side_ring', comp, desc, ids, metadata=sr_meta))
                
                avail_11 -= 2; avail_23 -= total_23; avail_22 -= ring_22; avail_24 -= 2
            else:
                remaining_E = prev_E
                remaining_G = prev_G
                remaining.insert(0, n2)
                remaining.insert(0, n1)
                break

        # ===== Step 4: Chain branches for remaining single 24 =====
        for n in remaining:
            prev_E = list(remaining_E)
            prev_G = list(remaining_G)
            b23, b22, tail_src = self._consume_reserved_branch_tail(n[1], remaining_E, remaining_G)
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
                    'branch_23_count': b23,
                    'branch_22_count': b22,
                    'extra_22_count': 0,
                    'tail_source': tail_src,
                }
                chains.append(ChainSpec('branch_side', comp, f"Br-chain({n[1][-1]})", [n[0].global_id], metadata=branch_meta.copy()))
                avail_11 -= 1; avail_23 -= total_23_needed; avail_22 -= (b22 + 1); avail_24 -= 1
            elif avail_11 >= 2 and avail_23 >= total_23_needed and avail_22 >= b22:
                if is_ab: comp = [11, 24, 23, 23, 11]
                else:     comp = [11, 23, 24, 23, 11]
                comp += [23] * b23 + [22] * b22
                branch_meta = {
                    'branch_type': n[1],
                    'branch_23_count': b23,
                    'branch_22_count': b22,
                    'extra_22_count': 0,
                    'tail_source': tail_src,
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


        return chains, avail_11, avail_23, avail_22, avail_24, avail_25, remaining_E, remaining_G

    # ---------- Phase 4: Build chains ----------
    def _build_closed_chains(self) -> List[ChainSpec]:
        chains = []
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

    def _allocate_open_chains(self, avail_11: int, avail_23: int, avail_22: int):
        chains = []
        remaining_E = list(self._type_lists['E'])
        remaining_G = list(self._type_lists['G'])

        # ----- Type B: 11-23-23-... -----
        for n in self._type_lists['B']:
            if remaining_E and avail_23 >= 2:
                e = remaining_E.pop(0)
                # B(11-23-23) + E(23-23-22) = 11-23-23-23-23-22
                comp = [11, 23, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'B+E', [n.global_id, e.global_id]))
                avail_11 -= 1; avail_23 -= 4; avail_22 -= 1
            elif remaining_G and avail_23 >= 1:
                g = remaining_G.pop(0)
                # B(11-23-23) + G(23-22) = 11-23-23-23-22
                comp = [11, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'B+G', [n.global_id, g.global_id]))
                avail_11 -= 1; avail_23 -= 3; avail_22 -= 1
            elif avail_11 >= 2 and avail_23 >= 2:
                # Close with 11: 11-23-23-11
                comp = [11, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'B', [n.global_id]))
                avail_11 -= 2; avail_23 -= 2
            elif avail_11 >= 2 and avail_23 >= 3:
                # Close with 23+11: 11-23-23-23-11
                comp = [11, 23, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'B', [n.global_id]))
                avail_11 -= 2; avail_23 -= 3
            else:
                print(f"  [WARN] Cannot close Type B node {n.global_id}, insufficient resources")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 1

        # ----- Type D: ...-23-23-23-... -----
        for n in self._type_lists['D']:
            if remaining_G and avail_11 >= 1 and avail_23 >= 1:
                g = remaining_G.pop(0)
                # 11 + D(23-23-23) + G(23-22) = 11-23-23-23-23-22
                comp = [11, 23, 23, 23, 23, 22]
                chains.append(ChainSpec('side', comp, 'D+G', [n.global_id, g.global_id]))
                avail_11 -= 1; avail_23 -= 4; avail_22 -= 1
            elif avail_11 >= 2 and avail_23 >= 3:
                # 11 + D(23-23-23) + 11 = 11-23-23-23-11
                comp = [11, 23, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'D', [n.global_id]))
                avail_11 -= 2; avail_23 -= 3
            elif avail_11 >= 2 and avail_23 >= 4:
                # 11 + D(23-23-23) + 23 + 11 = 11-23-23-23-23-11
                comp = [11, 23, 23, 23, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'D', [n.global_id]))
                avail_11 -= 2; avail_23 -= 4
            else:
                print(f"  [WARN] Cannot close Type D node {n.global_id}")
                self._result.unallocated_bridge += 1
                self._result.required_extra_11 += 2

        # Remaining Type E/G are reserved for branch terminal sealing in Phase 4.5.
        return chains, avail_11, avail_23, avail_22, remaining_E, remaining_G

    # ---------- Phase 5: Allocate remaining (balanced) ----------
    def _allocate_remaining(self, avail_11: int, avail_23: int, avail_22: int) -> Tuple[List[ChainSpec], int]:
        chains = []
        cap = MAX_23_PER_CHAIN
        # Reserve 11 for side chains: each 22 needs 1×11
        reserved_11_for_side = min(avail_22, avail_11)
        bridge_11 = avail_11 - reserved_11_for_side

        # Bridges: use bridge_11 pool, but distribute 23s as evenly as possible
        # to avoid creating too many dense 11-23-11 extras.
        n_bridges = bridge_11 // 2
        if n_bridges > 0 and avail_23 > 0:
            actual_bridges = min(n_bridges, avail_23)
            base_len = max(1, min(cap, avail_23 // actual_bridges))
            if avail_23 >= actual_bridges * 2:
                base_len = max(base_len, 2)

            bridge_lengths = [base_len] * actual_bridges
            consumed_23 = base_len * actual_bridges
            rem_23 = avail_23 - consumed_23
            idx = 0
            while rem_23 > 0 and bridge_lengths:
                if bridge_lengths[idx] < cap:
                    bridge_lengths[idx] += 1
                    rem_23 -= 1
                idx = (idx + 1) % len(bridge_lengths)
                if idx == 0 and all(bl >= cap for bl in bridge_lengths):
                    break

            for n_23 in sorted(bridge_lengths, reverse=True):
                comp = [11] + [23] * n_23 + [11]
                chains.append(ChainSpec('bridge', comp, 'extra'))

            avail_11 -= actual_bridges * 2
            avail_23 -= sum(bridge_lengths)

        # Side chains: 11-22 (needs 0x23)
        while avail_11 >= 1 and avail_22 >= 1:
            comp = [11, 22]
            chains.append(ChainSpec('side', comp, 'extra'))
            avail_11 -= 1
            avail_22 -= 1

        # If bridge resources remain, prefer len>=2 extras before falling back to len=1.
        while avail_11 >= 2 and avail_23 >= 2:
            comp = [11, 23, 23, 11]
            chains.append(ChainSpec('bridge', comp, 'extra'))
            avail_11 -= 2
            avail_23 -= 2

        if not any(ch.origin_type == 'extra' and ch.chain_type == 'bridge' for ch in chains):
            while avail_11 >= 2 and avail_23 >= 1:
                comp = [11, 23, 11]
                chains.append(ChainSpec('bridge', comp, 'extra'))
                avail_11 -= 2
                avail_23 -= 1
            
        return chains, avail_23

    def _prepare_branch_phase_resources(self) -> Dict[str, Any]:
        self._parse_input()
        self._convert_and_count()
        self._classify_all()
        self._classify_branch_24()
        self._classify_branch_25()

        r = self._result
        closed = self._build_closed_chains()
        c11 = sum(c.n_11 for c in closed)
        c23 = sum(c.n_23 for c in closed)
        c22 = sum(c.n_22 for c in closed)

        avail_11 = r.total_11 - c11
        avail_23 = r.total_23 - c23
        avail_22 = r.total_22 - c22

        open_chains, avail_11, avail_23, avail_22, remaining_E, remaining_G = self._allocate_open_chains(
            avail_11, avail_23, avail_22
        )
        o11 = sum(c.n_11 for c in open_chains)
        o23 = sum(c.n_23 for c in open_chains)
        o22 = sum(c.n_22 for c in open_chains)

        pre_branch_unallocated_bridge = int(r.unallocated_bridge)
        pre_branch_req_11 = int(r.required_extra_11)
        pre_branch_req_22 = int(r.required_extra_22)
        pre_branch_req_23 = int(r.required_extra_23)

        r.unallocated_bridge = 0
        r.unallocated_branch = 0
        r.required_extra_11 = 0
        r.required_extra_22 = 0
        r.required_extra_23 = 0

        return {
            'closed_chains': closed,
            'open_chains': open_chains,
            'remaining_E': remaining_E,
            'remaining_G': remaining_G,
            'closed_consumed': {'11': c11, '23': c23, '22': c22},
            'open_consumed': {'11': o11, '23': o23, '22': o22},
            'pre_branch_available': {'11': avail_11, '23': avail_23, '22': avail_22},
            'pre_branch_bridge_diag': {
                'unallocated_bridge': pre_branch_unallocated_bridge,
                'req_11': pre_branch_req_11,
                'req_22': pre_branch_req_22,
                'req_23': pre_branch_req_23,
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
        
        # Keep ~75% of B chains at their original length for vertical connections (allow 25% to expand)
        import random
        random.shuffle(b_chains)
        num_b_to_expand = int(len(b_chains) * 0.25)
        eligible.extend(b_chains[:num_b_to_expand])

        if not eligible:
            return excess_23

        def priority_key(ch: ChainSpec) -> Tuple[int, int, int]:
            if ch.origin_type == 'extra' and ch.chain_type == 'bridge':
                if ch.n_23 < 2:
                    return (0, ch.n_23, 0)
                if ch.n_23 < 3:
                    return (1, ch.n_23, 0)
                return (3, ch.n_23, 0)
            if ch.origin_type == 'extra' and ch.chain_type == 'side':
                return (2, ch.n_23, 1)
            return (4, ch.n_23, 2)

        # Water-filling: iteratively add 1 to the shortest fillable chain
        while excess_23 > 0:
            fillable = [ch for ch in eligible if ch.n_23 < cap]
            if not fillable:
                break

            best_key = min(priority_key(ch) for ch in fillable)
            targets = [ch for ch in fillable if priority_key(ch) == best_key]
            
            if excess_23 < len(targets):
                targets = random.sample(targets, excess_23)
                
            for ch in targets:
                if FlexAllocator._add_one_23_to_chain(ch):
                    excess_23 -= 1
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
            'branch_chains': [],
        }

        temp_allocator = FlexAllocator(nodes=nodes)
        prep = temp_allocator._prepare_branch_phase_resources()
        result['closed_consumed'] = dict(prep['closed_consumed'])
        result['open_consumed'] = dict(prep['open_consumed'])
        result['pre_branch_available'] = dict(prep['pre_branch_available'])
        result['pre_branch_bridge_diag'] = dict(prep['pre_branch_bridge_diag'])
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
            if result['req_22'] > 0:
                result['shortage_type'] = '22_shortage'
            elif result['req_11'] > 0:
                result['shortage_type'] = '11_shortage'
            elif result['req_23'] > 0:
                result['shortage_type'] = '23_shortage'
            else:
                result['shortage_type'] = 'general_shortage'
        
        return result

    def evaluate_su24_branches(self, nodes: List) -> Dict[str, Any]:
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

            avail_11 = prep['pre_branch_available']['11']
            avail_23 = prep['pre_branch_available']['23']
            avail_22 = prep['pre_branch_available']['22']

            branch_chains, rem_11, rem_23, rem_22, rem_24, rem_25, _rem_E, _rem_G = allocator._allocate_branches(
                avail_11, avail_23, avail_22,
                prep.get('remaining_E', []),
                prep.get('remaining_G', []),
            )

            res = allocator._result
            result['alloc_result'] = res
            result['unallocated_branch'] = res.unallocated_branch
            result['unallocated_bridge'] = 0
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
                if res.required_extra_22 > 0 or res.required_extra_11 > 0 or res.required_extra_23 > 0:
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
            result['shortage_type'] = 'error'
        
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
        self._parse_input()
        print(f"  Parsed {len(self._nodes)} SU nodes from CSV")

        # Phase 2
        self._convert_and_count()
        r = self._result
        print(f"\n  [Phase 2] SU Conversion Totals:")
        print(f"    11 (aromatic endpoints): {r.total_11}  (from SU {sorted(TO_11)})")
        print(f"    23 (chain body):         {r.total_23}  (from SU {sorted(TO_23)})")
        print(f"    22 (terminals):           {r.total_22}  (from SU {sorted(TO_22)})")
        print(f"    24 (branch CH):           {r.total_24}  (from SU {sorted(TO_24)})")
        print(f"    25 (branch Cq):           {r.total_25}  (from SU [25])")

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

        # Phase 4a: closed chains
        closed = self._build_closed_chains()
        c11 = sum(c.n_11 for c in closed)
        c23 = sum(c.n_23 for c in closed)
        c22 = sum(c.n_22 for c in closed)
        print(f"\n  [Phase 4a] Closed chains: {len(closed)}")
        print(f"    Consumed: 11×{c11}, 23×{c23}, 22×{c22}")

        avail_11 = r.total_11 - c11
        avail_23 = r.total_23 - c23
        avail_22 = r.total_22 - c22
        print(f"    Available after closed: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        # Phase 4b: open chains
        open_chains, avail_11, avail_23, avail_22, remaining_E, remaining_G = self._allocate_open_chains(
            avail_11, avail_23, avail_22
        )
        all_chains = closed + open_chains
        o11 = sum(c.n_11 for c in open_chains)
        o23 = sum(c.n_23 for c in open_chains)
        o22 = sum(c.n_22 for c in open_chains)
        print(f"\n  [Phase 4b] Open chain allocation: {len(open_chains)}")
        print(f"    Consumed: 11×{o11}, 23×{o23}, 22×{o22}")
        print(f"    Available after open: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        # Phase 4.5: Branch allocation (24/25)
        branch_chains, avail_11, avail_23, avail_22, avail_24, avail_25, remaining_E, remaining_G = \
            self._allocate_branches(avail_11, avail_23, avail_22, remaining_E, remaining_G)
        br11 = sum(c.n_11 for c in branch_chains)
        br23 = sum(c.n_23 for c in branch_chains)
        br22 = sum(c.n_22 for c in branch_chains)
        br24 = sum(c.n_24 for c in branch_chains)
        br25 = sum(c.n_25 for c in branch_chains)
        print(f"\n  [Phase 4.5] Branch allocation (24/25): {len(branch_chains)}")
        print(f"    Consumed: 11×{br11}, 23×{br23}, 22×{br22}, 24×{br24}, 25×{br25}")
        print(f"    Available after branch: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}, 24×{avail_24}, 25×{avail_25}")

        reserved_sides, avail_11, avail_23, avail_22 = self._allocate_reserved_terminal_sides(
            remaining_E, remaining_G, avail_11, avail_23, avail_22
        )
        all_chains += reserved_sides
        rs11 = sum(c.n_11 for c in reserved_sides)
        rs23 = sum(c.n_23 for c in reserved_sides)
        rs22 = sum(c.n_22 for c in reserved_sides)
        print(f"\n  [Phase 4.6] Reserved E/G side chains: {len(reserved_sides)}")
        print(f"    Consumed: 11×{rs11}, 23×{rs23}, 22×{rs22}")
        print(f"    Available after reserved tails: 11×{avail_11}, 23×{avail_23}, 22×{avail_22}")

        # Phase 5: remaining (create base extra chains)
        extra, leftover_23 = self._allocate_remaining(avail_11, avail_23, avail_22)
        all_chains += extra
        e11 = sum(c.n_11 for c in extra)
        e23 = sum(c.n_23 for c in extra)
        e22 = sum(c.n_22 for c in extra)
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
            c23 = sum(c.n_23 for c in closed)
            o23 = sum(c.n_23 for c in open_chains)
            e23 = sum(c.n_23 for c in extra)
            br23 = sum(c.n_23 for c in branch_chains)
            
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
        total_consumed_11 = c11 + o11 + br11 + e11
        total_consumed_23 = c23 + o23 + br23 + e23
        total_consumed_22 = c22 + o22 + br22 + e22
        r.consumed_11 = total_consumed_11
        r.consumed_23 = total_consumed_23
        r.consumed_22 = total_consumed_22
        r.consumed_24 = br24
        r.consumed_25 = br25
        r.remaining_11 = r.total_11 - r.consumed_11
        r.remaining_23 = r.total_23 - r.consumed_23
        r.remaining_22 = r.total_22 - r.consumed_22
        r.remaining_24 = r.total_24 - r.consumed_24
        r.remaining_25 = r.total_25 - r.consumed_25

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
        print(f"  Bridge chains: {len(r.bridge_chains)}")
        for i, ch in enumerate(r.bridge_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, len_23={ch.n_23})")

        print(f"\n  Side chains: {len(r.side_chains)}")
        for i, ch in enumerate(r.side_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, len_23={ch.n_23})")

        print(f"\n  Branch structures (24/25): {len(r.branch_chains)}")
        for i, ch in enumerate(r.branch_chains):
            comp_str = '-'.join(str(x) for x in ch.composition)
            extra = f", 24×{ch.n_24}" if ch.n_24 else ""
            extra += f", 25×{ch.n_25}" if ch.n_25 else ""
            print(f"    [{i}] {comp_str}  (type={ch.origin_type}, 23×{ch.n_23}, 22×{ch.n_22}{extra})")

        print(f"\n  Resource Usage:")
        print(f"    {'':>10} {'Total':>8} {'Consumed':>10} {'Remaining':>10}")
        print(f"    {'SU 11':>10} {r.total_11:>8} {r.consumed_11:>10} {r.remaining_11:>10}")
        print(f"    {'SU 23':>10} {r.total_23:>8} {r.consumed_23:>10} {r.remaining_23:>10}")
        print(f"    {'SU 22':>10} {r.total_22:>8} {r.consumed_22:>10} {r.remaining_22:>10}")
        print(f"    {'SU 24':>10} {r.total_24:>8} {r.consumed_24:>10} {r.remaining_24:>10}")
        print(f"    {'SU 25':>10} {r.total_25:>8} {r.consumed_25:>10} {r.remaining_25:>10}")

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
