from typing import Any, List, Dict, Optional, Tuple, Set
import copy
import math
import random

from .RL_state import (
    MCTSState, ChainNode, EdgeBranch, HexGrid, HEX_VERTEX_OFFSETS,
    RU, RD, LU, LD, UP, DN, OPPOSITE,
    qr_shape_score_from_points, spatial_uniformity_score_from_points,
)
from .RL_allocator import FlexAllocator, ChainSpec, chain_spec_counts_match


def _build_alternating_path(start_q, start_r,
                            first_step: Tuple[int, int],
                            second_step: Tuple[int, int],
                            length: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    cur_q, cur_r = start_q, start_r
    for i in range(max(0, int(length))):
        dq, dr = first_step if i % 2 == 0 else second_step
        cur_q += dq
        cur_r += dr
        coords.append((cur_q, cur_r))
    return coords

def _vertical_ring_coords(anchor_q, anchor_r, direction='DN'):
    """Compute coordinates for a 6-node vertical aliphatic ring.

    anchor = aromatic SU13 site (a, b) that gets converted to SU11.
    direction = 'DN' (downward) or 'UP' (upward).
    
    Coordinates as per user spec:
    DN from (a, b):
      first_24:    (a,   b-1)
      right_23:    (a+1, b-1)
      left_23:     (a-1, b-2)
      inter_right: (a+1, b-2)
      inter_left:  (a-1, b-3)
      closing_23:  (a,   b-3)

    UP from (a, b) (flipping - to +, + to - for r-deltas):
      first_24:    (a,   b+1)
      right_23:    (a+1, b+1)
      left_23:     (a-1, b+2)
      inter_right: (a+1, b+2)
      inter_left:  (a-1, b+3)
      closing_23:  (a,   b+3)
    """
    a, b = anchor_q, anchor_r
    if direction == 'DN':
        return {
            'anchor': (a, b),
            'first_24': (a, b - 1),
            'right_23': (a + 1, b - 1),
            'inter_right': (a + 1, b - 2),
            'closing_23': (a, b - 3),
            'inter_left': (a - 1, b - 3),
            'left_23': (a - 1, b - 2),
        }
    else:  # UP
        return {
            'anchor': (a, b),
            'first_24': (a, b + 1),
            'right_23': (a + 1, b + 2),
            'inter_right': (a + 1, b + 3),
            'closing_23': (a, b + 3),
            'inter_left': (a - 1, b + 2),
            'left_23': (a - 1, b + 1),
        }


def _branch_coords_for_inter(inter_q, inter_r, side='right', direction='DN',
                             branch_type='C', branch_len: Optional[int] = None):
    """Compute branch coordinates hanging off an inter-position 24 node.

    For C/A-type: -23-23-22 branch (3 nodes)
    For D/B-type: -22 only (1 node)
    
    Based on user exact spec:
    DN right from (a+1, b-2): (a+2, b-2), (a+3, b-1), (a+4, b-1)
    DN left from (a-1, b-3): (a-2, b-4), (a-3, b-4), (a-4, b-5)
    
    UP right: flip b-deltas
    UP left: flip b-deltas
    """
    n23, n22 = FlexAllocator._branch_cost(branch_type)
    length = int(branch_len) if branch_len is not None else int(n23 + n22)
    if direction == 'DN':
        first_step, second_step = (RD, RU) if side == 'right' else (LD, LU)
    else:
        first_step, second_step = (RU, RD) if side == 'right' else (LU, LD)
    return _build_alternating_path(inter_q, inter_r, first_step, second_step, length)


def _side_ring_coords_right(upper_q, upper_r, lower_q, lower_r):
    """Compute coordinates for a side (horizontal) aliphatic ring, right side.

    User specification:
      upper site (a, b) → upper_24 or 23 (a+1, b+1)
      lower site (a, b-1) → lower_24 or 23 (a+1, b-1)
      upper bridge (a+2, b+1)
      lower bridge (a+2, b)
    """
    a_u, b_u = upper_q, upper_r
    a_l, b_l = lower_q, lower_r

    return {
        'upper_site': (a_u, b_u),
        'lower_site': (a_l, b_l),
        'upper_24': (a_u + 1, b_u + 1),
        'lower_24': (a_l + 1, b_l),
        'upper_bridge_23': (a_u + 2, b_u + 1),
        'lower_bridge_23': (a_l + 2, b_l + 1),  # b_l + 1 is equivalent to b since b_l = b_u - 1
    }


def _side_ring_coords_left(upper_q, upper_r, lower_q, lower_r):
    """Compute coordinates for a side (horizontal) aliphatic ring, left side.

    User specification:
      lower site (a, b) → lower_24 or 23 (a-1, b-1)
      upper site (a, b+1) → upper_24 or 23 (a-1, b+1)
      lower bridge (a-2, b-1)
      upper bridge (a-2, b)
    """
    a_l, b_l = lower_q, lower_r  # User denoted lower as (a, b)
    a_u, b_u = upper_q, upper_r  # User denoted upper as (a, b+1)

    return {
        'upper_site': (a_u, b_u),
        'lower_site': (a_l, b_l),
        'upper_24': (a_u - 1, b_u),       # (a-1, b+1)
        'lower_24': (a_l - 1, b_l - 1),   # (a-1, b-1)
        'upper_bridge_23': (a_u - 2, b_u - 1), # (a-2, b)
        'lower_bridge_23': (a_l - 2, b_l - 1), # (a-2, b-1)
    }



def _fused_side_ring_coords_right(upper_q, upper_r, lower_q, lower_r):
    a_u, b_u = upper_q, upper_r
    a_l, b_l = lower_q, lower_r
    return {
        'upper_site': (a_u, b_u),
        'lower_site': (a_l, b_l),
        'base_upper_24': (a_u + 1, b_u + 1),
        'base_lower_24': (a_l + 1, b_l),
        'bridge_upper_24': (a_u + 2, b_u + 1),
        'bridge_lower_24': (a_l + 2, b_l + 1),
        'outer_upper_23': (a_u + 3, b_u + 2),
        'outer_upper_24': (a_u + 4, b_u + 2),
        'outer_lower_24': (a_l + 4, b_l + 2),
        'outer_lower_23': (a_l + 3, b_l + 1),
        'outer_inner_upper': (a_u + 3, b_u + 2),
        'outer_outer_upper': (a_u + 4, b_u + 2),
        'outer_outer_lower': (a_l + 4, b_l + 2),
        'outer_inner_lower': (a_l + 3, b_l + 1),
    }

def _fused_side_ring_coords_left(upper_q, upper_r, lower_q, lower_r):
    """Fused side ring coords for LEFT side.

    Derived by mirroring the right-side ring: RU↔LU, RD↔LD in hex directions.
    Left bridge_upper = (a_u-2, b_u-1), bridge_lower = (a_l-2, b_l-1).
    Outer ring traverses: LU → LD → DN → RD → RU → UP (mirror of right).
    """
    a_u, b_u = upper_q, upper_r
    a_l, b_l = lower_q, lower_r
    return {
        'upper_site': (a_u, b_u),
        'lower_site': (a_l, b_l),
        'base_upper_24': (a_u - 1, b_u),
        'base_lower_24': (a_l - 1, b_l - 1),
        'bridge_upper_24': (a_u - 2, b_u - 1),
        'bridge_lower_24': (a_l - 2, b_l - 1),
        'outer_upper_23': (a_u - 3, b_u - 1),
        'outer_upper_24': (a_u - 4, b_u - 2),
        'outer_lower_24': (a_l - 4, b_l - 2),
        'outer_lower_23': (a_l - 3, b_l - 2),
        'outer_inner_upper': (a_u - 3, b_u - 1),
        'outer_outer_upper': (a_u - 4, b_u - 2),
        'outer_outer_lower': (a_l - 4, b_l - 2),
        'outer_inner_lower': (a_l - 3, b_l - 2),
    }

def _fused_outer_branch_coords(q, r, side, is_upper, branch_len: int = 3):
    """Branch coords from an outer-ring 24 node (max 3 nodes: -23-23-22)."""
    if side == 'right':
        first_step, second_step = (RU, RD) if is_upper else (RD, RU)
    else:
        first_step, second_step = (LU, LD) if is_upper else (LD, LU)
    return _build_alternating_path(q, r, first_step, second_step, branch_len)


def _side_ring_branch_coords(q, r, side, is_upper, branch_type,
                             branch_len: Optional[int] = None):
    """Branch coords from a 24 node in a side ring.

    A/B-type 24 nodes are the outer-adjacent ring vertices:
      right-upper: UP -> RU
      right-lower: DN -> RD
      left-upper:  UP -> LU
      left-lower:  DN -> LD

    C/D-type 24 nodes are the bridge-side ring vertices:
      right-upper: RU -> RD
      right-lower: RD -> RU
      left-upper:  LU -> LD
      left-lower:  LD -> LU
    """
    n23, n22 = FlexAllocator._branch_cost(branch_type)
    length = int(branch_len) if branch_len is not None else int(n23 + n22)
    is_ab = str(branch_type) in ('24_A', '24_B', '25_aro')
    if side == 'right':
        if is_ab:
            first_step, second_step = (UP, RU) if is_upper else (DN, RD)
        else:
            first_step, second_step = (RU, RD) if is_upper else (RD, RU)
    else:
        if is_ab:
            first_step, second_step = (UP, LU) if is_upper else (DN, LD)
        else:
            first_step, second_step = (LU, LD) if is_upper else (LD, LU)
    return _build_alternating_path(q, r, first_step, second_step, length)


def _chain_branch_family(position_idx: int, branch_type: str) -> str:
    if branch_type in ('24_A', '24_B', '25_aro'):
        return 'AB'
    if branch_type in ('24_C', '24_D', '25_ali'):
        return 'CD'
    return 'AB' if position_idx <= 0 else 'CD'


def _chain_branch_pair(outward_dir, position_idx: int, branch_type: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    family = _chain_branch_family(position_idx, branch_type)
    if outward_dir == RU:
        return (UP, RU) if family == 'AB' else (DN, RD)
    if outward_dir == RD:
        return (DN, RD) if family == 'AB' else (UP, RU)
    if outward_dir == LU:
        return (UP, LU) if family == 'AB' else (DN, LD)
    if outward_dir == LD:
        return (DN, LD) if family == 'AB' else (UP, LU)
    return None


def _side_family(outward: Optional[Tuple[int, int]]) -> Optional[str]:
    if outward in (RU, RD):
        return 'right'
    if outward in (LU, LD):
        return 'left'
    return None


# ==================== Exported helpers (used by stage_flex / stage_side) ====================

def get_branch_info_from_chain_spec(chain_spec: ChainSpec):
    """Extract branch (24/25) info from a ChainSpec.

    Returns dict with keys: position_idx, su_type, branch_type,
    branch_23_count, branch_22_count.  Returns None if no 24/25.
    """
    body = chain_spec.composition[1:-1]  # strip endpoints
    for i, su in enumerate(body):
        if su not in (24, 25):
            continue
        meta = getattr(chain_spec, 'metadata', {}) or {}
        # Determine branch type from origin_type string
        btype = '24_A'
        if meta.get('branch_type'):
            btype = str(meta['branch_type'])
        desc = chain_spec.origin_type or ''
        if btype == '24_A' and 'Br-chain(' in desc:
            letter = desc.split('Br-chain(')[-1].rstrip(')')
            if letter in ('A', 'B', 'C', 'D'):
                btype = '24_' + letter
        elif btype == '24_A' and 'Br-25(' in desc:
            subtype = desc.split('Br-25(')[-1].rstrip(')')
            if subtype in ('aro', 'ali'):
                btype = '25_' + subtype
            else:
                btype = '25_aro'

        if su == 25:
            branch_23 = int(meta.get('branch_23_count', 2))
            branch_22 = int(meta.get('branch_22_count', 1))
            extra_22 = int(meta.get('extra_22_count', 1))
            profile = _branch24_profile_for_role(chain_spec, 'branch_single', 0)
            return {
                'position_idx': i,
                'su_type': 25,
                'branch_type': btype,
                'branch_23_count': branch_23,
                'branch_22_count': branch_22,
                'extra_22_count': extra_22,
                'tail_source': meta.get('tail_source'),
                'branch24_profile': profile,
            }
        else:
            if 'branch_23_count' in meta or 'branch_22_count' in meta:
                b23 = int(meta.get('branch_23_count', 0))
                b22 = int(meta.get('branch_22_count', 1))
            elif btype in ('24_A', '24_C'):
                b23, b22 = 2, 1
            else:
                b23, b22 = 0, 1
            profile = _branch24_profile_for_role(chain_spec, 'branch_single', 0)
            return {
                'position_idx': i,
                'su_type': 24,
                'branch_type': btype,
                'branch_23_count': b23,
                'branch_22_count': b22,
                'extra_22_count': 0,
                'tail_source': meta.get('tail_source'),
                'branch24_profile': profile,
            }
    return None


_SEMANTIC_METADATA_KEYS = (
    'base_ring_su',
    'outer_ring_su',
    'side_ring_su',
    'vertical_ring_su',
    'ring_substitute_ids',
    'ring_substitute_types',
    'connector_pair',
    'tail_sources',
    'tail_source',
    'branch_tail_lengths',
    'fixed_c_ids',
    'fixed_c_source_kinds',
    'fixed_c_path_kinds',
)


def _chain_spec_semantic_summary(chain_spec: ChainSpec) -> Dict[str, Any]:
    """Small immutable-ish summary copied into ChainNode.meta.

    The allocator now carries topology semantics in ChainSpec.metadata.  Stages
    should preserve those semantics on the placed nodes so later substitution
    logic does not have to rediscover special 19/20/21d3 context from geometry.
    """
    meta = dict(getattr(chain_spec, 'metadata', {}) or {})
    out: Dict[str, Any] = {
        'chain_type': getattr(chain_spec, 'chain_type', None),
        'origin_type': getattr(chain_spec, 'origin_type', None),
        'source_ids': [int(x) for x in list(getattr(chain_spec, 'source_ids', []) or [])],
    }
    for key in _SEMANTIC_METADATA_KEYS:
        if key in meta:
            out[key] = copy.deepcopy(meta.get(key))
    return out


def _branch24_profiles(chain_spec: ChainSpec) -> List[Dict[str, Any]]:
    meta = getattr(chain_spec, 'metadata', {}) or {}
    profiles = meta.get('branch24_profiles', []) or []
    return [dict(p) for p in profiles if isinstance(p, dict)]


def _branch24_profile_for_role(chain_spec: ChainSpec,
                               role: str,
                               occurrence: int = 0) -> Optional[Dict[str, Any]]:
    matches = [
        p for p in _branch24_profiles(chain_spec)
        if str(p.get('allocator_role')) == str(role)
    ]
    idx = int(max(0, occurrence))
    if idx >= len(matches):
        return None
    return dict(matches[idx])


def _branch_type_from_profile(profile: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(profile, dict):
        return None
    btype = profile.get('abcd_type')
    if isinstance(btype, str) and btype in ('24_A', '24_B', '24_C', '24_D'):
        return btype
    return None


def _branch_types_for_role(chain_spec: ChainSpec,
                           role: str,
                           count: int) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for idx in range(int(count)):
        out.append(_branch_type_from_profile(
            _branch24_profile_for_role(chain_spec, role, idx)
        ))
    return out


def _tail_source_for_index(chain_spec: ChainSpec, index: int) -> Optional[str]:
    meta = getattr(chain_spec, 'metadata', {}) or {}
    sources = list(meta.get('tail_sources', []) or [])
    if not sources and meta.get('tail_source') is not None:
        sources = [meta.get('tail_source')]
    idx = int(index)
    if idx < 0 or idx >= len(sources):
        return None
    return str(sources[idx])


def _tail_source_for_side_slot(chain_spec: ChainSpec, slot_name: str) -> Optional[str]:
    meta = getattr(chain_spec, 'metadata', {}) or {}
    slot_names = [str(x) for x in list(meta.get('side_ring_slot_names', []) or [])]
    sources = list(meta.get('tail_sources', []) or [])
    try:
        idx = slot_names.index(str(slot_name))
    except ValueError:
        return None
    if idx < 0 or idx >= len(sources) or sources[idx] is None:
        return None
    return str(sources[idx])


def _side_profile_index_by_slot(chain_spec: ChainSpec) -> Dict[str, int]:
    meta = getattr(chain_spec, 'metadata', {}) or {}
    slot_names = list(meta.get('side_ring_slot_names', []) or [])
    out: Dict[str, int] = {}
    profile_idx = 0
    for idx, slot_name in enumerate(slot_names):
        if slot_name is None:
            continue
        btype = None
        node_types = list(meta.get('side_ring_node_types', []) or [])
        if idx < len(node_types):
            btype = node_types[idx]
        if btype is None:
            continue
        out[str(slot_name)] = int(profile_idx)
        profile_idx += 1
    return out


def build_chain_node_meta(chain_spec: ChainSpec,
                          *,
                          stage: str,
                          ring_role: Optional[str] = None,
                          branch_type: Optional[str] = None,
                          branch_kind: Optional[str] = None,
                          position_idx: Optional[int] = None,
                          profile_role: Optional[str] = None,
                          profile_index: int = 0,
                          tail_source: Optional[str] = None,
                          su_type: Optional[int] = None) -> Dict[str, Any]:
    """Build metadata for a placed ChainNode, preserving allocator semantics."""
    meta: Dict[str, Any] = {
        'stage': str(stage),
        'origin_type': getattr(chain_spec, 'origin_type', None),
        'chain_type': getattr(chain_spec, 'chain_type', None),
        'chain_spec_semantics': _chain_spec_semantic_summary(chain_spec),
    }
    if ring_role is not None:
        meta['ring_role'] = str(ring_role)
    if branch_type is not None:
        meta['branch_type'] = branch_type
    if branch_kind is not None:
        meta['branch_kind'] = str(branch_kind)
    if position_idx is not None:
        meta['position_idx'] = int(position_idx)
    if tail_source is not None:
        meta['tail_source'] = str(tail_source)

    profile = _branch24_profile_for_role(chain_spec, profile_role, profile_index) if profile_role else None
    if isinstance(profile, dict):
        meta['branch24_profile'] = dict(profile)
        meta['source_node_id'] = int(profile.get('node_id')) if profile.get('node_id') is not None else None
        meta['source_su_type'] = int(profile.get('su_type')) if profile.get('su_type') is not None else None
        meta['source_kind'] = profile.get('source_kind')
        meta['fixed_usage'] = profile.get('fixed_usage')
        meta['fixed_label'] = profile.get('fixed_label')
        meta['fixed_path_kind'] = profile.get('fixed_path_kind')
        meta['fixed_connector_ids'] = list(profile.get('fixed_connector_ids', []) or [])
        meta['fixed_connector_types'] = list(profile.get('fixed_connector_types', []) or [])
        meta['partner_special_ids'] = list(profile.get('partner_special_ids', []) or [])
        meta['partner_special_degrees'] = list(profile.get('partner_special_degrees', []) or [])
        meta['is_double_special'] = bool(profile.get('is_double_special', False))
        meta['allowed_tail_modes'] = list(profile.get('allowed_tail_modes', []) or [])
        meta['allowed_slots'] = list(profile.get('allowed_slots', []) or [])
        if branch_type is None and profile.get('abcd_type') is not None:
            meta['branch_type'] = str(profile.get('abcd_type'))

    spec_meta = getattr(chain_spec, 'metadata', {}) or {}
    substitute_types = [int(x) for x in list(spec_meta.get('ring_substitute_types', []) or [])]
    if su_type is not None and int(su_type) in substitute_types:
        meta['ring_substitute'] = True
        meta['ring_substitute_types'] = substitute_types
        meta['ring_substitute_ids'] = [int(x) for x in list(spec_meta.get('ring_substitute_ids', []) or [])]
        if spec_meta.get('connector_pair') is not None:
            meta['connector_pair'] = copy.deepcopy(spec_meta.get('connector_pair'))

    return meta


def horizontal_branch_coords(chain_coords, position_idx, outward_dir, branch_type, branch_len=None):
    """Compute branch coordinates for a 24 node inside a horizontal flex chain.

    Args:
        chain_coords: body coordinates of the flex chain (list of (q, r))
        position_idx: index of the 24 node in chain_coords
        outward_dir: outward hex direction of the chain's source site
        branch_type: '24_A', '24_B', '24_C', '24_D'

    Returns list of (q, r) for the branch nodes.
    """
    if position_idx >= len(chain_coords):
        return []

    bq, br = chain_coords[position_idx]

    if branch_len is not None:
        n = max(0, int(branch_len))
    elif str(branch_type).startswith('25') or branch_type in ('24_A', '24_C', 'A', 'C'):
        n = 3  # -23-23-22
    else:
        n = 1  # -22

    pair = _chain_branch_pair(outward_dir, position_idx, branch_type)
    if pair is None:
        return []
    return _build_alternating_path(bq, br, pair[0], pair[1], n)


def su25_extra_branch_coord(chain_coords, position_idx, outward_dir, branch_type):
    """Compute the extra -22 branch coord for an SU25 node (degree-4 vertex).

    The main branch goes in one perpendicular direction (via
    horizontal_branch_coords); this returns the single coord for
    the *other* perpendicular direction.
    """
    if position_idx >= len(chain_coords):
        return None

    bq, br = chain_coords[position_idx]

    pair = _chain_branch_pair(outward_dir, position_idx, branch_type)
    if pair is None:
        return None

    first_dir = pair[0]
    extra_dir = OPPOSITE.get(first_dir)
    if extra_dir is None:
        return None
    return (bq + extra_dir[0], br + extra_dir[1])


# ==================== Outward-direction helper (shared with stage_flex) ====================

def _get_site_outward(cluster, site):
    """Return the outward hex direction for *site* on *cluster*."""
    sq, sr = site.axial
    ring_offsets = {}
    for ri, (cq, cr) in enumerate(cluster.centers):
        dq, dr = sq - cq, sr - cr
        if (dq, dr) in HEX_VERTEX_OFFSETS:
            ring_offsets[ri] = (dq, dr)
    if not ring_offsets:
        return None
    indices = sorted(ring_offsets.keys())
    if len(indices) == 1:
        return ring_offsets[indices[0]]
    best, best_s = None, -1
    for ri in [indices[0], indices[-1]]:
        o = ring_offsets[ri]
        s = abs(o[0]) + abs(o[1])
        if s > best_s:
            best_s = s
            best = o
    return best


# ==================== BranchStage ====================

class BranchStage:
    """Stage that places aliphatic ring structures (vertical, side, fused side)
    using pre-allocated ChainSpec entries from the FlexAllocator."""

    def __init__(self, state: MCTSState, branch_specs: List[ChainSpec]):
        self.state = state
        self._specs: List[ChainSpec] = list(branch_specs)
        self._n_total = len(self._specs)
        self._done = 0

        # Placed vertex coordinates (collision tracking)
        self._placed: Set[Tuple[int, int]] = set()
        for c in state.graph.clusters:
            if c.placed:
                for s in c.sites:
                    self._placed.add(s.axial)
                for ctr in c.centers:
                    self._placed.add(ctr)
        for cn in state.graph.chains:
            self._placed.add(cn.axial)

    def clone(self):
        new = copy.deepcopy(self)
        return new

    def is_done(self) -> bool:
        return self._done >= self._n_total

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def get_candidates(self, k: int = 5) -> List[Dict]:
        if self.is_done():
            return []
        spec = self._specs[self._done]
        ctype = spec.chain_type
        if ctype == 'vertical_ring':
            return self._gen_vertical_ring_candidates(spec, k)
        elif ctype == 'fused_side_ring':
            return self._gen_fused_side_ring_candidates(spec, k)
        elif ctype == 'side_ring':
            return self._gen_side_ring_candidates(spec, k)
        return []

    # ------------------------------------------------------------------
    # Site finding helpers
    # ------------------------------------------------------------------
    def _find_vertical_sites(self) -> List[Dict]:
        """Find unoccupied SU13 sites at UP or DN positions."""
        sites = []
        for cluster in self.state.graph.clusters:
            if not cluster.placed:
                continue
            for si, site in enumerate(cluster.sites):
                if site.su_type != 13 or site.occupied:
                    continue
                outward = _get_site_outward(cluster, site)
                if outward in (UP, DN):
                    direction = 'UP' if outward == UP else 'DN'
                    sites.append({
                        'cluster': cluster,
                        'site_idx': si,
                        'site': site,
                        'direction': direction,
                    })
        return sites

    def _find_side_edge_pairs(self) -> List[Dict]:
        """Find adjacent unoccupied SU13 pairs on the same cluster edge."""
        pairs = []
        for cluster in self.state.graph.clusters:
            if not cluster.placed:
                continue
            sites = cluster.sites
            for i in range(len(sites)):
                si = sites[i]
                if si.su_type != 13 or si.occupied:
                    continue
                oi = _get_site_outward(cluster, si)
                side_i = _side_family(oi)
                if side_i is None:
                    continue
                for j in range(i + 1, len(sites)):
                    sj = sites[j]
                    if sj.su_type != 13 or sj.occupied:
                        continue
                    oj = _get_site_outward(cluster, sj)
                    side_j = _side_family(oj)
                    if side_j is None:
                        continue
                    # Must belong to the same left/right side family.
                    # Real aromatic edge pairs are typically RU+RD (right)
                    # or LU+LD (left), not identical outward vectors.
                    if side_i != side_j:
                        continue
                    # Must be vertically adjacent (same q, |delta_r| == 1)
                    qi, ri = si.axial
                    qj, rj = sj.axial
                    if qi != qj or abs(ri - rj) != 1:
                        continue
                    side = side_i
                    upper_idx = i if ri > rj else j
                    lower_idx = j if ri > rj else i
                    pairs.append({
                        'cluster': cluster,
                        'pair': (upper_idx, lower_idx),
                        'upper_site': sites[upper_idx],
                        'lower_site': sites[lower_idx],
                        'side': side,
                    })
        return pairs

    def _check_collision(self, coords: List[Tuple[int, int]]) -> bool:
        """Return True if any coord collides with an already-placed vertex."""
        if len(set(coords)) != len(coords):
            return True
        for c in coords:
            if c in self._placed:
                return True
        return False

    def _branch_tail_lengths(self, spec: ChainSpec) -> Dict[str, int]:
        meta = getattr(spec, 'metadata', {}) or {}
        return {
            str(k): int(v)
            for k, v in (meta.get('branch_tail_lengths', {}) or {}).items()
            if int(v) > 0
        }

    @staticmethod
    def _metadata_su_list(spec: ChainSpec, key: str, expected_len: int) -> List[int]:
        meta = getattr(spec, 'metadata', {}) or {}
        raw = list(meta.get(str(key), []) or [])
        out: List[int] = []
        for value in raw[:int(expected_len)]:
            try:
                out.append(int(value))
            except Exception:
                return []
        if len(out) != int(expected_len):
            return []
        return out

    def _preview_qr_shape_score(self, coords: List[Tuple[int, int]]) -> float:
        points = set((int(q), int(r)) for q, r in self._placed)
        points.update((int(q), int(r)) for q, r in coords)
        return qr_shape_score_from_points(points, 0.9, 2.3)

    def _preview_uniformity_score(self, coords: List[Tuple[int, int]], bins: int = 3) -> float:
        points = set((int(q), int(r)) for q, r in self._placed)
        points.update((int(q), int(r)) for q, r in coords)
        return spatial_uniformity_score_from_points(points, bins)

    @staticmethod
    def _vertical_action_sus(action: Dict) -> List[int]:
        out = list(int(x) for x in list(action.get('ring_su', []) or []))
        if not out:
            ir_is_24 = bool(action.get('ir_is_24', False))
            il_is_24 = bool(action.get('il_is_24', False))
            out = [24, 23, 24 if ir_is_24 else 23, 23, 24 if il_is_24 else 23, 23]
        right_branch = list(action.get('right_branch_coords', []) or [])
        left_branch = list(action.get('left_branch_coords', []) or [])
        out += [23] * max(0, len(right_branch) - 1)
        if right_branch:
            out += [22]
        out += [23] * max(0, len(left_branch) - 1)
        if left_branch:
            out += [22]
        return out

    @staticmethod
    def _side_action_sus(action: Dict) -> List[int]:
        out = list(int(x) for x in list(action.get('ring_su', []) or []))
        if action.get('side_branches') is not None:
            for br in list(action.get('side_branches', []) or []):
                out += list(int(x) for x in list(br.get('branch_su', []) or []))
            return out
        out += list(int(x) for x in list(action.get('upper_branch_su', []) or []))
        out += list(int(x) for x in list(action.get('lower_branch_su', []) or []))
        return out

    @staticmethod
    def _fused_action_sus(action: Dict) -> List[int]:
        out = list(int(x) for x in list(action.get('ring_su', []) or []))
        out += list(int(x) for x in list(action.get('outer_ring_su', []) or []))
        out += list(int(x) for x in list(action.get('upper_branch_su', []) or []))
        out += list(int(x) for x in list(action.get('lower_branch_su', []) or []))
        out += list(int(x) for x in list(action.get('outer_upper_branch_su', []) or []))
        out += list(int(x) for x in list(action.get('outer_lower_branch_su', []) or []))
        return out

    # ------------------------------------------------------------------
    # Candidate generators (existing methods follow)
    # ------------------------------------------------------------------
    def _gen_fused_side_ring_candidates(self, spec: ChainSpec, k: int) -> List[Dict]:
        """Generate candidates for fused side (horizontal) aliphatic ring placement."""
        candidates = []
        pairs = self._find_side_edge_pairs()
        
        upper_base_type, lower_base_type = _branch_types_for_role(spec, 'fused_base', 2)
        upper_bridge_type, lower_bridge_type = _branch_types_for_role(spec, 'fused_fixed_c', 2)
        outer_upper_type, outer_lower_type = _branch_types_for_role(spec, 'fused_outer', 2)
        planned_base_su = self._metadata_su_list(spec, 'base_ring_su', 4)
        planned_outer_su = self._metadata_su_list(spec, 'outer_ring_su', 4)

        if not planned_base_su or not any((upper_bridge_type, lower_bridge_type)):
            # Allocator-produced fused rings must carry a complete slot plan.
            # Silently parsing origin_type would discard connector/tail metadata.
            return []

        is_upper_ab = upper_base_type in ('24_A', '24_B')
        is_lower_ab = lower_base_type in ('24_A', '24_B')
        has_outer = bool(planned_outer_su)
            
        for pair in pairs:
            uq, ur = pair['upper_site'].axial
            lq, lr = pair['lower_site'].axial
            side = pair['side']

            if side == 'right':
                ring = _fused_side_ring_coords_right(uq, ur, lq, lr)
            else:
                ring = _fused_side_ring_coords_left(uq, ur, lq, lr)

            # Base ring coords
            pos1 = ring['base_upper_24']
            pos2 = ring['base_lower_24']
            pos3 = ring['bridge_upper_24']
            pos4 = ring['bridge_lower_24']
            
            ring_coords = [pos1, pos2, pos3, pos4]
            
            # Determine SU types for base ring. Prefer allocator slot-plan
            # metadata so connector/SU3 substitutions survive the stage boundary.
            pos1_su, pos3_su, pos4_su, pos2_su = planned_base_su

            branch_tail_lengths = self._branch_tail_lengths(spec)
            default_upper_len = sum(FlexAllocator._branch_cost(upper_base_type)) if (upper_base_type and is_upper_ab) else 0
            default_lower_len = sum(FlexAllocator._branch_cost(lower_base_type)) if (lower_base_type and is_lower_ab) else 0
            upper_len = int(branch_tail_lengths.get('base_upper', default_upper_len))
            lower_len = int(branch_tail_lengths.get('base_lower', default_lower_len))

            upper_branch = _side_ring_branch_coords(
                pos1[0], pos1[1], side, True, upper_base_type, branch_len=upper_len
            ) if is_upper_ab and upper_len > 0 else []
            lower_branch = _side_ring_branch_coords(
                pos2[0], pos2[1], side, False, lower_base_type, branch_len=lower_len
            ) if is_lower_ab and lower_len > 0 else []
            
            all_coords = ring_coords + upper_branch + lower_branch
            upper_branch_su = ([23] * max(0, upper_len - 1) + [22]) if upper_len > 0 else []
            lower_branch_su = ([23] * max(0, lower_len - 1) + [22]) if lower_len > 0 else []
            
            outer_ring_su = []
            outer_upper_branch = []
            outer_lower_branch = []
            outer_upper_branch_su = []
            outer_lower_branch_su = []

            if has_outer:
                opos_iu = ring['outer_upper_23']
                opos_ou = ring['outer_upper_24']
                opos_ol = ring['outer_lower_24']
                opos_il = ring['outer_lower_23']
                
                outer_coords = [opos_iu, opos_ou, opos_ol, opos_il]
                all_coords += outer_coords

                outer_ring_su = list(planned_outer_su)
                
                default_outer_upper = sum(FlexAllocator._branch_cost(outer_upper_type)) if outer_upper_type else 0
                default_outer_lower = sum(FlexAllocator._branch_cost(outer_lower_type)) if outer_lower_type else 0
                outer_upper_len = int(branch_tail_lengths.get('outer_upper', default_outer_upper))
                outer_lower_len = int(branch_tail_lengths.get('outer_lower', default_outer_lower))

                outer_upper_branch = _fused_outer_branch_coords(
                    opos_ou[0], opos_ou[1], side, True, branch_len=outer_upper_len
                ) if outer_upper_type and outer_upper_len > 0 else []
                outer_lower_branch = _fused_outer_branch_coords(
                    opos_ol[0], opos_ol[1], side, False, branch_len=outer_lower_len
                ) if outer_lower_type and outer_lower_len > 0 else []
                
                all_coords += outer_upper_branch + outer_lower_branch
                outer_upper_branch_su = ([23] * max(0, outer_upper_len - 1) + [22]) if outer_upper_len > 0 else []
                outer_lower_branch_su = ([23] * max(0, outer_lower_len - 1) + [22]) if outer_lower_len > 0 else []

            if self._check_collision(all_coords):
                continue

            candidate_sus = [pos1_su, pos3_su, pos4_su, pos2_su]
            candidate_sus += list(upper_branch_su) + list(lower_branch_su)
            candidate_sus += list(outer_ring_su) + list(outer_upper_branch_su) + list(outer_lower_branch_su)
            if not chain_spec_counts_match(spec, candidate_sus):
                continue

            cx, cy = self._global_centroid()
            sx, sy = pair['upper_site'].pos2d
            dist = math.hypot(sx - cx, sy - cy)
            qr_bonus = self._preview_qr_shape_score(all_coords)
            uniform_bonus = self._preview_uniformity_score(all_coords)
            score = dist / 5.0 + 2.5 * qr_bonus + 1.5 * uniform_bonus

            candidates.append({
                'type': 'fused_side_ring',
                'cluster_id': pair['cluster'].id,
                'upper_idx': pair['pair'][0],
                'lower_idx': pair['pair'][1],
                'side': pair['side'],
                'ring': ring,
                'has_outer': has_outer,
                'base_node_types': [upper_base_type, lower_base_type],
                'bridge_node_types': [upper_bridge_type, lower_bridge_type],
                'outer_node_types': [outer_upper_type, outer_lower_type],
                'ring_su': [pos1_su, pos3_su, pos4_su, pos2_su],
                'outer_ring_su': outer_ring_su,
                'upper_branch': upper_branch,
                'lower_branch': lower_branch,
                'upper_branch_su': upper_branch_su,
                'lower_branch_su': lower_branch_su,
                'outer_upper_branch': outer_upper_branch,
                'outer_lower_branch': outer_lower_branch,
                'outer_upper_branch_su': outer_upper_branch_su,
                'outer_lower_branch_su': outer_lower_branch_su,
                'all_coords': all_coords,
                'spec': spec,
                'score': -(score + random.uniform(0, 0.2)),
            })

        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    def _gen_vertical_ring_candidates(self, spec: ChainSpec, k: int) -> List[Dict]:
        """Generate candidates for vertical aliphatic ring placement.

        Ring always has 6 positions. Inter positions are 24 (with branch)
        or 23 (no branch) depending on the allocator spec.
        """
        candidates = []
        v_sites = self._find_vertical_sites()

        inter_profiles = [
            _branch24_profile_for_role(spec, 'vertical_inter', 0),
            _branch24_profile_for_role(spec, 'vertical_inter', 1),
        ]
        inter_types = [
            _branch_type_from_profile(inter_profiles[0]),
            _branch_type_from_profile(inter_profiles[1]),
        ]
        planned_vertical_su = self._metadata_su_list(spec, 'vertical_ring_su', 6)
        if not planned_vertical_su:
            # Vertical rings are allocator slot-plan driven.  Do not fall back
            # to origin_type parsing because it loses special 24-like metadata.
            return []
        branch_tail_lengths = self._branch_tail_lengths(spec)

        for vs in v_sites:
            site = vs['site']
            direction = vs['direction']
            a, b = site.axial

            ring = _vertical_ring_coords(a, b, direction)

            # All 6 ring body coords (always present)
            ring_coords = [
                ring['first_24'], ring['right_23'], ring['inter_right'],
                ring['closing_23'], ring['inter_left'], ring['left_23'],
            ]

            # Determine SU types for inter positions. Metadata keeps allocator
            # slot constraints authoritative across candidate generation.
            ring_su = list(planned_vertical_su)
            ir_is_24 = int(ring_su[2]) == 24
            il_is_24 = int(ring_su[4]) == 24

            # Compute branch coords for 24-type inter positions
            right_branch_coords = []
            left_branch_coords = []
            if ir_is_24:
                btype = inter_types[0]
                right_len = int(branch_tail_lengths.get('right', sum(FlexAllocator._branch_cost(btype))))
                right_branch_coords = _branch_coords_for_inter(
                    *ring['inter_right'], 'right', direction, btype,
                    branch_len=right_len)
            if il_is_24:
                btype = inter_types[1]
                left_len = int(branch_tail_lengths.get('left', sum(FlexAllocator._branch_cost(btype))))
                left_branch_coords = _branch_coords_for_inter(
                    *ring['inter_left'], 'left', direction, btype,
                    branch_len=left_len)

            all_coords = ring_coords + right_branch_coords + left_branch_coords
            if self._check_collision(all_coords):
                continue

            candidate_sus = list(ring_su)
            candidate_sus += [23] * max(0, len(right_branch_coords) - 1)
            if right_branch_coords:
                candidate_sus += [22]
            candidate_sus += [23] * max(0, len(left_branch_coords) - 1)
            if left_branch_coords:
                candidate_sus += [22]
            if not chain_spec_counts_match(spec, candidate_sus):
                continue

            # Score: prefer peripheral placement (secondary to NMR)
            cx, cy = self._global_centroid()
            sx, sy = site.pos2d
            dist = math.hypot(sx - cx, sy - cy)
            qr_bonus = self._preview_qr_shape_score(all_coords)
            uniform_bonus = self._preview_uniformity_score(all_coords)
            score = dist / 5.0 + 2.5 * qr_bonus + 1.5 * uniform_bonus

            candidates.append({
                'type': 'vertical_ring',
                'cluster_id': vs['cluster'].id,
                'site_idx': vs['site_idx'],
                'direction': direction,
                'ring': ring,
                'ir_is_24': ir_is_24,
                'il_is_24': il_is_24,
                'ring_su': list(ring_su),
                'inter_types': inter_types,
                'right_branch_coords': right_branch_coords,
                'left_branch_coords': left_branch_coords,
                'all_coords': all_coords,
                'spec': spec,
                'score': -(score + random.uniform(0, 0.2)),
            })

        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    def _gen_side_ring_candidates(self, spec: ChainSpec, k: int) -> List[Dict]:
        """Generate candidates for side (horizontal) aliphatic ring placement.

        Handles all type combinations (A+A, C+C, A+C, C+A) by dynamically
        determining which ring positions are SU 24 vs 23, and computing
        branch directions perpendicular to ring edges at the 24 positions.
        """
        candidates = []
        pairs = self._find_side_edge_pairs()

        meta = getattr(spec, 'metadata', {}) or {}
        node_types = list(meta.get('side_ring_node_types', []) or [])
        planned_ring_su = self._metadata_su_list(spec, 'side_ring_su', 4)
        if not node_types or not planned_ring_su:
            # Side rings now require allocator slot metadata. Origin string
            # fallback cannot represent 4-slot rings or special connector state.
            return []
        node_types = list(node_types[:4])
        while len(node_types) < 4:
            node_types.append(None)
        branch_tail_lengths = self._branch_tail_lengths(spec)

        for pair in pairs:
            uq, ur = pair['upper_site'].axial
            lq, lr = pair['lower_site'].axial
            side = pair['side']

            if side == 'right':
                ring = _side_ring_coords_right(uq, ur, lq, lr)
            else:
                ring = _side_ring_coords_left(uq, ur, lq, lr)

            # 4 ring body positions (same for all type combos)
            pos1 = ring['upper_24']        # adjacent to upper site
            pos3 = ring['upper_bridge_23'] # bridge upper
            pos4 = ring['lower_bridge_23'] # bridge lower
            pos2 = ring['lower_24']        # adjacent to lower site

            ring_coords = [pos1, pos2, pos3, pos4]

            # Determine SU types for ring positions. Prefer allocator
            # slot-plan metadata so AB/CD order and substitutions stay intact.
            pos1_su, pos3_su, pos4_su, pos2_su = planned_ring_su

            slot_defs = [
                ('pos1', pos1, True),
                ('pos3', pos3, True),
                ('pos4', pos4, False),
                ('pos2', pos2, False),
            ]
            side_branches = []
            all_coords = list(ring_coords)
            for idx, (slot_name, slot_pos, is_upper) in enumerate(slot_defs):
                btype = node_types[idx] if idx < len(node_types) else None
                if btype is None or int([pos1_su, pos3_su, pos4_su, pos2_su][idx]) != 24:
                    continue
                default_len = sum(FlexAllocator._branch_cost(btype))
                branch_len = int(branch_tail_lengths.get(slot_name, default_len))
                branch_coords = _side_ring_branch_coords(
                    slot_pos[0], slot_pos[1], side, bool(is_upper), btype, branch_len=branch_len
                ) if branch_len > 0 else []
                branch_su = ([23] * max(0, branch_len - 1) + [22]) if branch_len > 0 else []
                if branch_coords:
                    side_branches.append({
                        'slot_name': slot_name,
                        'slot_index': idx,
                        'branch_type': btype,
                        'is_upper': bool(is_upper),
                        'base_pos': slot_pos,
                        'branch_coords': branch_coords,
                        'branch_su': branch_su,
                    })
                    all_coords += branch_coords
            if self._check_collision(all_coords):
                continue

            candidate_sus = [pos1_su, pos3_su, pos4_su, pos2_su]
            for br in side_branches:
                candidate_sus += list(br.get('branch_su', []) or [])
            if not chain_spec_counts_match(spec, candidate_sus):
                continue

            cx, cy = self._global_centroid()
            sx, sy = pair['upper_site'].pos2d
            dist = math.hypot(sx - cx, sy - cy)
            qr_bonus = self._preview_qr_shape_score(all_coords)
            uniform_bonus = self._preview_uniformity_score(all_coords)
            score = dist / 5.0 + 2.5 * qr_bonus + 1.5 * uniform_bonus

            candidates.append({
                'type': 'side_ring',
                'cluster_id': pair['cluster'].id,
                'upper_idx': pair['pair'][0],
                'lower_idx': pair['pair'][1],
                'side': pair['side'],
                'ring': ring,
                'node_types': node_types,
                'ring_su': [pos1_su, pos3_su, pos4_su, pos2_su],
                'side_branches': side_branches,
                'upper_branch': side_branches[0]['branch_coords'] if len(side_branches) > 0 else [],
                'lower_branch': side_branches[1]['branch_coords'] if len(side_branches) > 1 else [],
                'upper_branch_su': side_branches[0]['branch_su'] if len(side_branches) > 0 else [],
                'lower_branch_su': side_branches[1]['branch_su'] if len(side_branches) > 1 else [],
                'all_coords': all_coords,
                'spec': spec,
                'score': -(score + random.uniform(0, 0.2)),
            })

        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------
    def step(self, action: Dict) -> bool:
        atype = action.get('type')
        if atype == 'vertical_ring':
            return self._step_vertical_ring(action)
        elif atype == 'fused_side_ring':
            return self._step_fused_side_ring(action)
        elif atype == 'side_ring':
            return self._step_side_ring(action)
        return False

    def _step_vertical_ring(self, action: Dict) -> bool:
        """Execute vertical ring placement.

        Always places 6 ring nodes. Inter positions are SU24 (with branch)
        or SU23 (no branch) depending on the allocator spec.

        Ring body order: first_24 → right_23 → inter_right → closing_23
                         → inter_left → left_23 → (back to first_24)
        Branches off inter nodes are stored as separate EdgeBranch entries.
        """
        cluster = self.state.graph.clusters[action['cluster_id']]
        site = cluster.sites[action['site_idx']]
        ring = action['ring']
        spec = action['spec']
        if not chain_spec_counts_match(spec, self._vertical_action_sus(action)):
            return False
        ir_is_24 = action['ir_is_24']
        il_is_24 = action['il_is_24']
        inter_types = action['inter_types']
        ring_su = list(int(x) for x in list(action.get('ring_su', []) or []))
        if len(ring_su) != 6:
            ring_su = [24, 23, 24 if ir_is_24 else 23, 23, 24 if il_is_24 else 23, 23]

        # Mark anchor SU13 → SU11
        site.occupied = True
        site.su_type = 11

        uid_prefix = f"VR-{cluster.id}-{action['site_idx']}"

        # --- Create all 6 ring body nodes ---
        fq, fr = ring['first_24']
        n_first = ChainNode(uid=f"{uid_prefix}-24a", su_type=ring_su[0], axial=(fq, fr),
                           pos2d=HexGrid.axial_to_cart(fq, fr),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               branch_type='24_A',
                               ring_role='first_24',
                               profile_role='vertical_fixed_a',
                               su_type=ring_su[0],
                           ))
        self._placed.add((fq, fr))

        rq, rr = ring['right_23']
        n_right = ChainNode(uid=f"{uid_prefix}-23R", su_type=ring_su[1], axial=(rq, rr),
                           pos2d=HexGrid.axial_to_cart(rq, rr),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               ring_role='right_23',
                               su_type=ring_su[1],
                           ))
        self._placed.add((rq, rr))

        iq_r, ir_r = ring['inter_right']
        ir_su = int(ring_su[2])
        n_ir = ChainNode(uid=f"{uid_prefix}-ir", su_type=ir_su, axial=(iq_r, ir_r),
                        pos2d=HexGrid.axial_to_cart(iq_r, ir_r),
                        meta=build_chain_node_meta(
                            spec,
                            stage='branch',
                            branch_type=inter_types[0],
                            ring_role='inter_right',
                            profile_role='vertical_inter' if ir_is_24 else None,
                            profile_index=0,
                            su_type=ir_su,
                        ))
        self._placed.add((iq_r, ir_r))

        cq, cr = ring['closing_23']
        n_close = ChainNode(uid=f"{uid_prefix}-23C", su_type=ring_su[3], axial=(cq, cr),
                           pos2d=HexGrid.axial_to_cart(cq, cr),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               ring_role='closing_23',
                               su_type=ring_su[3],
                           ))
        self._placed.add((cq, cr))

        iq_l, ir_l = ring['inter_left']
        il_su = int(ring_su[4])
        n_il = ChainNode(uid=f"{uid_prefix}-il", su_type=il_su, axial=(iq_l, ir_l),
                        pos2d=HexGrid.axial_to_cart(iq_l, ir_l),
                        meta=build_chain_node_meta(
                            spec,
                            stage='branch',
                            branch_type=inter_types[1],
                            ring_role='inter_left',
                            profile_role='vertical_inter' if il_is_24 else None,
                            profile_index=1 if ir_is_24 else 0,
                            su_type=il_su,
                        ))
        self._placed.add((iq_l, ir_l))

        lq, lr = ring['left_23']
        n_left = ChainNode(uid=f"{uid_prefix}-23L", su_type=ring_su[5], axial=(lq, lr),
                          pos2d=HexGrid.axial_to_cart(lq, lr),
                          meta=build_chain_node_meta(
                              spec,
                              stage='branch',
                              ring_role='left_23',
                              su_type=ring_su[5],
                          ))
        self._placed.add((lq, lr))

        # --- Ring body: all 6 nodes in hex-adjacent traversal order ---
        ring_body = [n_first, n_right, n_ir, n_close, n_il, n_left]
        edge_ring = EdgeBranch(base=site.uid, chain=ring_body, target=n_first.uid)
        self.state.graph.branch.append(edge_ring)
        self.state.graph.chains.extend(ring_body)

        # --- Branches off inter nodes (separate edges) ---
        right_branch_coords = action.get('right_branch_coords', [])
        if ir_is_24 and right_branch_coords:
            br_nodes = []
            for bi, (bq, br) in enumerate(right_branch_coords):
                su = 22 if bi == len(right_branch_coords) - 1 else 23
                bn = ChainNode(uid=f"{uid_prefix}-br-ir-{bi}", su_type=su,
                              axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br),
                              meta=build_chain_node_meta(
                                  spec,
                                  stage='branch',
                                  branch_type=inter_types[0],
                                  branch_kind='tail',
                                  position_idx=bi,
                                  profile_role='vertical_inter',
                                  profile_index=0,
                                  tail_source=_tail_source_for_index(spec, 0),
                                  su_type=su,
                              ))
                br_nodes.append(bn)
                self._placed.add((bq, br))
            edge_br = EdgeBranch(base=n_ir.uid, chain=br_nodes)
            self.state.graph.branch.append(edge_br)
            self.state.graph.chains.extend(br_nodes)

        left_branch_coords = action.get('left_branch_coords', [])
        if il_is_24 and left_branch_coords:
            br_nodes = []
            for bi, (bq, br) in enumerate(left_branch_coords):
                su = 22 if bi == len(left_branch_coords) - 1 else 23
                bn = ChainNode(uid=f"{uid_prefix}-br-il-{bi}", su_type=su,
                              axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br),
                              meta=build_chain_node_meta(
                                  spec,
                                  stage='branch',
                                  branch_type=inter_types[1],
                                  branch_kind='tail',
                                  position_idx=bi,
                                  profile_role='vertical_inter',
                                  profile_index=1 if ir_is_24 else 0,
                                  tail_source=_tail_source_for_index(spec, 1 if ir_is_24 else 0),
                                  su_type=su,
                              ))
                br_nodes.append(bn)
                self._placed.add((bq, br))
            edge_br = EdgeBranch(base=n_il.uid, chain=br_nodes)
            self.state.graph.branch.append(edge_br)
            self.state.graph.chains.extend(br_nodes)

        self._done += 1
        self.state.stage_step += 1
        return True

    def _step_side_ring(self, action: Dict) -> bool:
        """Execute side ring placement with correct SU types for all type combos.

        Creates separate EdgeBranch entries for the ring body and each branch
        so that the sequential chain traversal produces correct graph edges.
        Ring body order: pos1 → pos3 → pos4 → pos2 (all hex-adjacent).
        """
        cluster = self.state.graph.clusters[action['cluster_id']]
        upper_site = cluster.sites[action['upper_idx']]
        lower_site = cluster.sites[action['lower_idx']]
        ring = action['ring']
        spec = action['spec']
        if not chain_spec_counts_match(spec, self._side_action_sus(action)):
            return False
        ring_su = action['ring_su']  # [pos1_su, pos3_su, pos4_su, pos2_su]
        node_types = list(action['node_types'])
        while len(node_types) < 4:
            node_types.append(None)
        profile_index_by_slot = _side_profile_index_by_slot(spec)

        # Mark both sites as occupied, convert to 11
        upper_site.occupied = True
        upper_site.su_type = 11
        lower_site.occupied = True
        lower_site.su_type = 11

        uid_prefix = f"SR-{cluster.id}-{action['upper_idx']}-{action['lower_idx']}"

        # --- Ring body (4 positions in correct traversal order) ---
        p1q, p1r = ring['upper_24']
        n_pos1 = ChainNode(uid=f"{uid_prefix}-p1", su_type=ring_su[0],
                           axial=(p1q, p1r), pos2d=HexGrid.axial_to_cart(p1q, p1r),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               branch_type=node_types[0],
                               ring_role='upper_outer',
                               profile_role='side_slot' if node_types[0] is not None else None,
                               profile_index=profile_index_by_slot.get('pos1', 0),
                               su_type=ring_su[0],
                           ))
        self._placed.add((p1q, p1r))

        p3q, p3r = ring['upper_bridge_23']
        n_pos3 = ChainNode(uid=f"{uid_prefix}-p3", su_type=ring_su[1],
                           axial=(p3q, p3r), pos2d=HexGrid.axial_to_cart(p3q, p3r),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               branch_type=node_types[1],
                               ring_role='upper_bridge',
                               profile_role='side_slot' if node_types[1] is not None else None,
                               profile_index=profile_index_by_slot.get('pos3', 0),
                               su_type=ring_su[1],
                           ))
        self._placed.add((p3q, p3r))

        p4q, p4r = ring['lower_bridge_23']
        n_pos4 = ChainNode(uid=f"{uid_prefix}-p4", su_type=ring_su[2],
                           axial=(p4q, p4r), pos2d=HexGrid.axial_to_cart(p4q, p4r),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               branch_type=node_types[2],
                               ring_role='lower_bridge',
                               profile_role='side_slot' if node_types[2] is not None else None,
                               profile_index=profile_index_by_slot.get('pos4', 0),
                               su_type=ring_su[2],
                           ))
        self._placed.add((p4q, p4r))

        p2q, p2r = ring['lower_24']
        n_pos2 = ChainNode(uid=f"{uid_prefix}-p2", su_type=ring_su[3],
                           axial=(p2q, p2r), pos2d=HexGrid.axial_to_cart(p2q, p2r),
                           meta=build_chain_node_meta(
                               spec,
                               stage='branch',
                               branch_type=node_types[3],
                               ring_role='lower_outer',
                               profile_role='side_slot' if node_types[3] is not None else None,
                               profile_index=profile_index_by_slot.get('pos2', 0),
                               su_type=ring_su[3],
                           ))
        self._placed.add((p2q, p2r))

        ring_body = [n_pos1, n_pos3, n_pos4, n_pos2]
        edge_ring = EdgeBranch(base=upper_site.uid, chain=ring_body, target=lower_site.uid)
        self.state.graph.branch.append(edge_ring)
        self.state.graph.chains.extend(ring_body)

        ring_nodes_by_slot = {
            'pos1': n_pos1,
            'pos3': n_pos3,
            'pos4': n_pos4,
            'pos2': n_pos2,
        }

        # --- Branches from every 24-like ring position (separate edges) ---
        for br_action in list(action.get('side_branches', []) or []):
            slot_name = str(br_action.get('slot_name'))
            base_node = ring_nodes_by_slot.get(slot_name)
            if base_node is None:
                continue
            branch_coords = list(br_action.get('branch_coords', []) or [])
            branch_su = list(br_action.get('branch_su', []) or [])
            if not branch_coords:
                continue
            btype = br_action.get('branch_type')
            profile_idx = profile_index_by_slot.get(slot_name, 0)
            br_nodes = []
            for bi, (bq, br) in enumerate(branch_coords):
                su = branch_su[bi] if bi < len(branch_su) else 22
                bn = ChainNode(uid=f"{uid_prefix}-br-{slot_name}-{bi}", su_type=su,
                              axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br),
                              meta=build_chain_node_meta(
                                  spec,
                                  stage='branch',
                                  branch_type=btype,
                                  branch_kind='tail',
                                  position_idx=bi,
                                  profile_role='side_slot',
                                  profile_index=profile_idx,
                                  tail_source=_tail_source_for_side_slot(spec, slot_name),
                                  su_type=su,
                              ))
                br_nodes.append(bn)
                self._placed.add((bq, br))
            edge_br = EdgeBranch(base=base_node.uid, chain=br_nodes)
            self.state.graph.branch.append(edge_br)
            self.state.graph.chains.extend(br_nodes)

        self._done += 1
        self.state.stage_step += 1
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _global_centroid(self) -> Tuple[float, float]:
        xs, ys, n = 0.0, 0.0, 0
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            for s in c.sites:
                xs += s.pos2d[0]; ys += s.pos2d[1]; n += 1
        for cn in self.state.graph.chains:
            xs += cn.pos2d[0]; ys += cn.pos2d[1]; n += 1
        if n == 0: return (0.0, 0.0)
        return (xs / n, ys / n)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self) -> float:
        missing = self._n_total - self._done
        return -missing * 30.0 + self._done * 5.0

    def get_result(self) -> Dict:
        return {
            'branches_placed': self._done,
            'branches_total': self._n_total,
            'branch_edges': len(self.state.graph.branch),
            'branch_nodes': sum(len(e.chain) for e in self.state.graph.branch),
        }

    def _step_fused_side_ring(self, action: Dict) -> bool:
        """Execute fused side ring placement."""
        cluster = self.state.graph.clusters[action['cluster_id']]
        upper_site = cluster.sites[action['upper_idx']]
        lower_site = cluster.sites[action['lower_idx']]
        ring = action['ring']
        spec = action['spec']
        if not chain_spec_counts_match(spec, self._fused_action_sus(action)):
            return False
        ring_su = action['ring_su']
        has_outer = action['has_outer']
        base_node_types = list(action.get('base_node_types', [None, None]) or [None, None])
        bridge_node_types = list(action.get('bridge_node_types', ['24_C', '24_C']) or ['24_C', '24_C'])
        outer_node_types = list(action.get('outer_node_types', [None, None]) or [None, None])
        
        upper_site.occupied = True
        upper_site.su_type = 11
        lower_site.occupied = True
        lower_site.su_type = 11

        uid_prefix = f"FSR-{cluster.id}-{action['upper_idx']}-{action['lower_idx']}"

        # Base Ring
        p1q, p1r = ring['base_upper_24']
        n_pos1 = ChainNode(uid=f"{uid_prefix}-b-pu", su_type=ring_su[0], axial=(p1q, p1r), pos2d=HexGrid.axial_to_cart(p1q, p1r))
        n_pos1.meta = build_chain_node_meta(
            spec,
            stage='branch',
            branch_type=base_node_types[0] if int(ring_su[0]) == 24 else None,
            ring_role='base_upper',
            profile_role='fused_base',
            profile_index=0,
            su_type=ring_su[0],
        )
        self._placed.add((p1q, p1r))

        p3q, p3r = ring['bridge_upper_24']
        n_pos3 = ChainNode(uid=f"{uid_prefix}-b-bru", su_type=ring_su[1], axial=(p3q, p3r), pos2d=HexGrid.axial_to_cart(p3q, p3r))
        n_pos3.meta = build_chain_node_meta(
            spec,
            stage='branch',
            branch_type=bridge_node_types[0] if int(ring_su[1]) == 24 else None,
            ring_role='bridge_upper',
            profile_role='fused_fixed_c',
            profile_index=0,
            su_type=ring_su[1],
        )
        self._placed.add((p3q, p3r))

        p4q, p4r = ring['bridge_lower_24']
        n_pos4 = ChainNode(uid=f"{uid_prefix}-b-brl", su_type=ring_su[2], axial=(p4q, p4r), pos2d=HexGrid.axial_to_cart(p4q, p4r))
        n_pos4.meta = build_chain_node_meta(
            spec,
            stage='branch',
            branch_type=bridge_node_types[1] if int(ring_su[2]) == 24 else None,
            ring_role='bridge_lower',
            profile_role='fused_fixed_c',
            profile_index=1,
            su_type=ring_su[2],
        )
        self._placed.add((p4q, p4r))

        p2q, p2r = ring['base_lower_24']
        n_pos2 = ChainNode(uid=f"{uid_prefix}-b-pl", su_type=ring_su[3], axial=(p2q, p2r), pos2d=HexGrid.axial_to_cart(p2q, p2r))
        n_pos2.meta = build_chain_node_meta(
            spec,
            stage='branch',
            branch_type=base_node_types[1] if int(ring_su[3]) == 24 else None,
            ring_role='base_lower',
            profile_role='fused_base',
            profile_index=1,
            su_type=ring_su[3],
        )
        self._placed.add((p2q, p2r))

        ring_body = [n_pos1, n_pos3, n_pos4, n_pos2]
        edge_ring = EdgeBranch(base=upper_site.uid, chain=ring_body, target=lower_site.uid)
        self.state.graph.branch.append(edge_ring)
        self.state.graph.chains.extend(ring_body)
        
        # Base Branches
        upper_branch = action.get('upper_branch', [])
        upper_branch_su = action.get('upper_branch_su', [])
        if upper_branch:
            ub_nodes = []
            for bi, (bq, br) in enumerate(upper_branch):
                su = upper_branch_su[bi] if bi < len(upper_branch_su) else 22
                bn = ChainNode(uid=f"{uid_prefix}-b-u-{bi}", su_type=su, axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br))
                bn.meta = build_chain_node_meta(
                    spec,
                    stage='branch',
                    branch_type=base_node_types[0],
                    branch_kind='tail',
                    position_idx=bi,
                    profile_role='fused_base',
                    profile_index=0,
                    tail_source=_tail_source_for_index(spec, 0),
                    su_type=su,
                )
                ub_nodes.append(bn)
                self._placed.add((bq, br))
            self.state.graph.branch.append(EdgeBranch(base=n_pos1.uid, chain=ub_nodes))
            self.state.graph.chains.extend(ub_nodes)

        lower_branch = action.get('lower_branch', [])
        lower_branch_su = action.get('lower_branch_su', [])
        if lower_branch:
            lb_nodes = []
            for bi, (bq, br) in enumerate(lower_branch):
                su = lower_branch_su[bi] if bi < len(lower_branch_su) else 22
                bn = ChainNode(uid=f"{uid_prefix}-b-l-{bi}", su_type=su, axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br))
                bn.meta = build_chain_node_meta(
                    spec,
                    stage='branch',
                    branch_type=base_node_types[1],
                    branch_kind='tail',
                    position_idx=bi,
                    profile_role='fused_base',
                    profile_index=1,
                    tail_source=_tail_source_for_index(spec, 1),
                    su_type=su,
                )
                lb_nodes.append(bn)
                self._placed.add((bq, br))
            self.state.graph.branch.append(EdgeBranch(base=n_pos2.uid, chain=lb_nodes))
            self.state.graph.chains.extend(lb_nodes)

        # Outer Ring
        if has_outer:
            outer_ring_su = action['outer_ring_su']
            
            ou23q, ou23r = ring['outer_upper_23']
            on_pos_u23 = ChainNode(uid=f"{uid_prefix}-o-u23", su_type=outer_ring_su[0], axial=(ou23q, ou23r), pos2d=HexGrid.axial_to_cart(ou23q, ou23r))
            on_pos_u23.meta = build_chain_node_meta(
                spec,
                stage='branch',
                ring_role='outer_upper_23',
                su_type=outer_ring_su[0],
            )
            self._placed.add((ou23q, ou23r))
            
            ou24q, ou24r = ring['outer_upper_24']
            on_pos_u24 = ChainNode(uid=f"{uid_prefix}-o-u24", su_type=outer_ring_su[1], axial=(ou24q, ou24r), pos2d=HexGrid.axial_to_cart(ou24q, ou24r))
            on_pos_u24.meta = build_chain_node_meta(
                spec,
                stage='branch',
                branch_type=outer_node_types[0] if int(outer_ring_su[1]) == 24 else None,
                ring_role='outer_upper_24',
                profile_role='fused_outer' if int(outer_ring_su[1]) == 24 else None,
                profile_index=0,
                su_type=outer_ring_su[1],
            )
            self._placed.add((ou24q, ou24r))
            
            ol24q, ol24r = ring['outer_lower_24']
            on_pos_l24 = ChainNode(uid=f"{uid_prefix}-o-l24", su_type=outer_ring_su[2], axial=(ol24q, ol24r), pos2d=HexGrid.axial_to_cart(ol24q, ol24r))
            on_pos_l24.meta = build_chain_node_meta(
                spec,
                stage='branch',
                branch_type=outer_node_types[1] if int(outer_ring_su[2]) == 24 else None,
                ring_role='outer_lower_24',
                profile_role='fused_outer' if int(outer_ring_su[2]) == 24 else None,
                profile_index=1,
                su_type=outer_ring_su[2],
            )
            self._placed.add((ol24q, ol24r))
            
            ol23q, ol23r = ring['outer_lower_23']
            on_pos_l23 = ChainNode(uid=f"{uid_prefix}-o-l23", su_type=outer_ring_su[3], axial=(ol23q, ol23r), pos2d=HexGrid.axial_to_cart(ol23q, ol23r))
            on_pos_l23.meta = build_chain_node_meta(
                spec,
                stage='branch',
                ring_role='outer_lower_23',
                su_type=outer_ring_su[3],
            )
            self._placed.add((ol23q, ol23r))
            
            outer_ring_body = [on_pos_u23, on_pos_u24, on_pos_l24, on_pos_l23]
            # Outer ring attached to the bridgehead 24s of the base ring (n_pos3 and n_pos4)
            oedge_ring = EdgeBranch(base=n_pos3.uid, chain=outer_ring_body, target=n_pos4.uid)
            self.state.graph.branch.append(oedge_ring)
            self.state.graph.chains.extend(outer_ring_body)
            
            outer_upper_branch = action.get('outer_upper_branch', [])
            outer_upper_branch_su = action.get('outer_upper_branch_su', [])
            if outer_upper_branch:
                oub_nodes = []
                for bi, (bq, br) in enumerate(outer_upper_branch):
                    su = outer_upper_branch_su[bi] if bi < len(outer_upper_branch_su) else 22
                    bn = ChainNode(uid=f"{uid_prefix}-o-u-{bi}", su_type=su, axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br))
                    bn.meta = build_chain_node_meta(
                        spec,
                        stage='branch',
                        branch_type=outer_node_types[0],
                        branch_kind='tail',
                        position_idx=bi,
                        profile_role='fused_outer',
                        profile_index=0,
                        tail_source=_tail_source_for_index(spec, 2),
                        su_type=su,
                    )
                    oub_nodes.append(bn)
                    self._placed.add((bq, br))
                self.state.graph.branch.append(EdgeBranch(base=on_pos_u24.uid, chain=oub_nodes))
                self.state.graph.chains.extend(oub_nodes)

            outer_lower_branch = action.get('outer_lower_branch', [])
            outer_lower_branch_su = action.get('outer_lower_branch_su', [])
            if outer_lower_branch:
                olb_nodes = []
                for bi, (bq, br) in enumerate(outer_lower_branch):
                    su = outer_lower_branch_su[bi] if bi < len(outer_lower_branch_su) else 22
                    bn = ChainNode(uid=f"{uid_prefix}-o-l-{bi}", su_type=su, axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br))
                    bn.meta = build_chain_node_meta(
                        spec,
                        stage='branch',
                        branch_type=outer_node_types[1],
                        branch_kind='tail',
                        position_idx=bi,
                        profile_role='fused_outer',
                        profile_index=1,
                        tail_source=_tail_source_for_index(spec, 3),
                        su_type=su,
                    )
                    olb_nodes.append(bn)
                    self._placed.add((bq, br))
                self.state.graph.branch.append(EdgeBranch(base=on_pos_l24.uid, chain=olb_nodes))
                self.state.graph.chains.extend(olb_nodes)

        self._done += 1
        self.state.stage_step += 1
        return True
