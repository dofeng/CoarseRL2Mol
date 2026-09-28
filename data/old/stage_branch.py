from typing import List, Dict, Optional, Tuple, Set
import copy
import math
import random

from .RL_state import (
    MCTSState, ChainNode, EdgeBranch, HexGrid, HEX_VERTEX_OFFSETS,
    RU, RD, LU, LD, UP, DN, OPPOSITE,
    qr_shape_score_from_points, spatial_uniformity_score_from_points,
)
from .RL_allocator import FlexAllocator, ChainSpec, chain_spec_counts_match


def _branch_type_from_letter(letter: str) -> Optional[str]:
    if letter in ('A', 'B', 'C', 'D'):
        return '24_' + letter
    return None


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
            return {
                'position_idx': i,
                'su_type': 25,
                'branch_type': btype,
                'branch_23_count': branch_23,
                'branch_22_count': branch_22,
                'extra_22_count': extra_22,
            }
        else:
            if 'branch_23_count' in meta or 'branch_22_count' in meta:
                b23 = int(meta.get('branch_23_count', 0))
                b22 = int(meta.get('branch_22_count', 1))
            elif btype in ('24_A', '24_C'):
                b23, b22 = 2, 1
            else:
                b23, b22 = 0, 1
            return {
                'position_idx': i,
                'su_type': 24,
                'branch_type': btype,
                'branch_23_count': b23,
                'branch_22_count': b22,
                'extra_22_count': 0,
            }
    return None


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
        ir_is_24 = bool(action.get('ir_is_24', False))
        il_is_24 = bool(action.get('il_is_24', False))
        out: List[int] = [24, 23, 24 if ir_is_24 else 23, 23, 24 if il_is_24 else 23, 23]
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
        
        desc = spec.origin_type
        import re
        m_base = re.search(r'Base:([ABCDX]+)\+Br:([ABCDX]+)', desc)
        if not m_base:
            return []
            
        base_str = m_base.group(1)
        bridge_str = m_base.group(2)
        
        m_out = re.search(r'\+Out:([ABCDX]+)', desc)
        out_str = m_out.group(1) if m_out else ""
        
        upper_base_type = _branch_type_from_letter(base_str[0]) if len(base_str) > 0 else None
        lower_base_type = _branch_type_from_letter(base_str[1]) if len(base_str) > 1 else None
        upper_bridge_type = _branch_type_from_letter(bridge_str[0]) if len(bridge_str) > 0 else '24_C'
        lower_bridge_type = _branch_type_from_letter(bridge_str[1]) if len(bridge_str) > 1 else '24_C'
        is_upper_ab = upper_base_type in ('24_A', '24_B')
        is_lower_ab = lower_base_type in ('24_A', '24_B')
        
        has_outer = len(out_str) > 0
        if has_outer:
            outer_upper_type = _branch_type_from_letter(out_str[0]) if len(out_str) > 0 else None
            outer_lower_type = _branch_type_from_letter(out_str[1]) if len(out_str) > 1 else None
        else:
            outer_upper_type = None
            outer_lower_type = None
            
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
            
            # Determine SU types for base ring
            pos1_su = 24 if is_upper_ab else 23
            pos2_su = 24 if is_lower_ab else 23
            
            pos3_su = 24 
            pos4_su = 24

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

                opos_iu_su = 23
                opos_ou_su = 24 if outer_upper_type else 23
                opos_ol_su = 24 if outer_lower_type else 23
                opos_il_su = 23
                outer_ring_su = [opos_iu_su, opos_ou_su, opos_ol_su, opos_il_su]
                
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

            candidate_sus = [pos1_su, pos2_su, pos3_su, pos4_su]
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

        # Parse inter types from description like "V-ring(A+C+D)" or "V-ring(A+C)" or "V-ring(A)"
        # First type = first_24 (always 24, no branch needed — all 3 neighbours are ring nodes)
        # Subsequent types = inter_right, inter_left
        desc = spec.origin_type
        inter_types = self._parse_inter_types(desc)
        branch_tail_lengths = self._branch_tail_lengths(spec)
        # inter_types[0] → inter_right type (or None if 23)
        # inter_types[1] → inter_left type (or None if 23)

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

            # Determine SU types for inter positions
            ir_is_24 = inter_types[0] is not None
            il_is_24 = inter_types[1] is not None

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

            candidate_sus = [24, 23, 24 if ir_is_24 else 23, 23, 24 if il_is_24 else 23, 23]
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

        desc = spec.origin_type
        node_types = self._parse_side_ring_types(desc)
        branch_tail_lengths = self._branch_tail_lengths(spec)

        is_upper_ab = node_types[0] in ('24_A', '24_B')
        is_lower_ab = node_types[1] in ('24_A', '24_B')

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

            # Determine SU types for ring positions
            pos1_su = 24 if is_upper_ab else 23
            pos3_su = 23 if is_upper_ab else 24
            pos4_su = 23 if is_lower_ab else 24
            pos2_su = 24 if is_lower_ab else 23

            # Determine which positions are 24 (for branches)
            # Upper 24: pos1 if AB, pos3 if CD
            if is_upper_ab:
                u24_pos = pos1
            else:
                u24_pos = pos3

            # Lower 24: pos2 if AB, pos4 if CD
            if is_lower_ab:
                l24_pos = pos2
            else:
                l24_pos = pos4

            # Compute branch paths from each 24 node based on user rules
            upper_len = int(branch_tail_lengths.get('upper', sum(FlexAllocator._branch_cost(node_types[0]))))
            lower_len = int(branch_tail_lengths.get('lower', sum(FlexAllocator._branch_cost(node_types[1]))))

            upper_branch = _side_ring_branch_coords(
                u24_pos[0], u24_pos[1], side, True, node_types[0], branch_len=upper_len
            ) if upper_len > 0 else []
            lower_branch = _side_ring_branch_coords(
                l24_pos[0], l24_pos[1], side, False, node_types[1], branch_len=lower_len
            ) if lower_len > 0 else []

            upper_branch_su = ([23] * max(0, upper_len - 1) + [22]) if upper_len > 0 else []
            lower_branch_su = ([23] * max(0, lower_len - 1) + [22]) if lower_len > 0 else []

            all_coords = ring_coords + upper_branch + lower_branch
            if self._check_collision(all_coords):
                continue

            candidate_sus = [pos1_su, pos2_su, pos3_su, pos4_su]
            candidate_sus += list(upper_branch_su) + list(lower_branch_su)
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
                'upper_branch': upper_branch,
                'lower_branch': lower_branch,
                'upper_branch_su': upper_branch_su,
                'lower_branch_su': lower_branch_su,
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

        # Mark anchor SU13 → SU11
        site.occupied = True
        site.su_type = 11

        uid_prefix = f"VR-{cluster.id}-{action['site_idx']}"

        # --- Create all 6 ring body nodes ---
        fq, fr = ring['first_24']
        n_first = ChainNode(uid=f"{uid_prefix}-24a", su_type=24, axial=(fq, fr),
                           pos2d=HexGrid.axial_to_cart(fq, fr),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': '24_A', 'ring_role': 'first_24'})
        self._placed.add((fq, fr))

        rq, rr = ring['right_23']
        n_right = ChainNode(uid=f"{uid_prefix}-23R", su_type=23, axial=(rq, rr),
                           pos2d=HexGrid.axial_to_cart(rq, rr),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'ring_role': 'right_23'})
        self._placed.add((rq, rr))

        iq_r, ir_r = ring['inter_right']
        ir_su = 24 if ir_is_24 else 23
        n_ir = ChainNode(uid=f"{uid_prefix}-ir", su_type=ir_su, axial=(iq_r, ir_r),
                        pos2d=HexGrid.axial_to_cart(iq_r, ir_r),
                        meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': inter_types[0], 'ring_role': 'inter_right'})
        self._placed.add((iq_r, ir_r))

        cq, cr = ring['closing_23']
        n_close = ChainNode(uid=f"{uid_prefix}-23C", su_type=23, axial=(cq, cr),
                           pos2d=HexGrid.axial_to_cart(cq, cr),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'ring_role': 'closing_23'})
        self._placed.add((cq, cr))

        iq_l, ir_l = ring['inter_left']
        il_su = 24 if il_is_24 else 23
        n_il = ChainNode(uid=f"{uid_prefix}-il", su_type=il_su, axial=(iq_l, ir_l),
                        pos2d=HexGrid.axial_to_cart(iq_l, ir_l),
                        meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': inter_types[1], 'ring_role': 'inter_left'})
        self._placed.add((iq_l, ir_l))

        lq, lr = ring['left_23']
        n_left = ChainNode(uid=f"{uid_prefix}-23L", su_type=23, axial=(lq, lr),
                          pos2d=HexGrid.axial_to_cart(lq, lr),
                          meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'ring_role': 'left_23'})
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
                              meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': inter_types[0], 'branch_kind': 'tail', 'position_idx': int(bi)})
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
                              meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': inter_types[1], 'branch_kind': 'tail', 'position_idx': int(bi)})
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
        node_types = action['node_types']

        # Mark both sites as occupied, convert to 11
        upper_site.occupied = True
        upper_site.su_type = 11
        lower_site.occupied = True
        lower_site.su_type = 11

        uid_prefix = f"SR-{cluster.id}-{action['upper_idx']}-{action['lower_idx']}"

        # --- Identify 24 role types before creating nodes ---
        is_upper_ab = node_types[0] in ('24_A', '24_B')
        is_lower_ab = node_types[1] in ('24_A', '24_B')

        # --- Ring body (4 positions in correct traversal order) ---
        p1q, p1r = ring['upper_24']
        n_pos1 = ChainNode(uid=f"{uid_prefix}-p1", su_type=ring_su[0],
                           axial=(p1q, p1r), pos2d=HexGrid.axial_to_cart(p1q, p1r),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[0], 'ring_role': 'upper_outer'})
        self._placed.add((p1q, p1r))

        p3q, p3r = ring['upper_bridge_23']
        n_pos3 = ChainNode(uid=f"{uid_prefix}-p3", su_type=ring_su[1],
                           axial=(p3q, p3r), pos2d=HexGrid.axial_to_cart(p3q, p3r),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[0] if not is_upper_ab else None, 'ring_role': 'upper_bridge'})
        self._placed.add((p3q, p3r))

        p4q, p4r = ring['lower_bridge_23']
        n_pos4 = ChainNode(uid=f"{uid_prefix}-p4", su_type=ring_su[2],
                           axial=(p4q, p4r), pos2d=HexGrid.axial_to_cart(p4q, p4r),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[1] if not is_lower_ab else None, 'ring_role': 'lower_bridge'})
        self._placed.add((p4q, p4r))

        p2q, p2r = ring['lower_24']
        n_pos2 = ChainNode(uid=f"{uid_prefix}-p2", su_type=ring_su[3],
                           axial=(p2q, p2r), pos2d=HexGrid.axial_to_cart(p2q, p2r),
                           meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[1], 'ring_role': 'lower_outer'})
        self._placed.add((p2q, p2r))

        ring_body = [n_pos1, n_pos3, n_pos4, n_pos2]
        edge_ring = EdgeBranch(base=upper_site.uid, chain=ring_body, target=lower_site.uid)
        self.state.graph.branch.append(edge_ring)
        self.state.graph.chains.extend(ring_body)

        # --- Identify 24 nodes for branch attachment ---
        upper_24_node = n_pos1 if is_upper_ab else n_pos3
        lower_24_node = n_pos2 if is_lower_ab else n_pos4

        # --- Upper branch (separate edge) ---
        upper_branch = action.get('upper_branch', [])
        upper_branch_su = action.get('upper_branch_su', [])
        if upper_branch:
            ub_nodes = []
            for bi, (bq, br) in enumerate(upper_branch):
                su = upper_branch_su[bi] if bi < len(upper_branch_su) else 22
                bn = ChainNode(uid=f"{uid_prefix}-br-u-{bi}", su_type=su,
                              axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br),
                              meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[0], 'branch_kind': 'tail', 'position_idx': int(bi)})
                ub_nodes.append(bn)
                self._placed.add((bq, br))
            edge_ub = EdgeBranch(base=upper_24_node.uid, chain=ub_nodes)
            self.state.graph.branch.append(edge_ub)
            self.state.graph.chains.extend(ub_nodes)

        # --- Lower branch (separate edge) ---
        lower_branch = action.get('lower_branch', [])
        lower_branch_su = action.get('lower_branch_su', [])
        if lower_branch:
            lb_nodes = []
            for bi, (bq, br) in enumerate(lower_branch):
                su = lower_branch_su[bi] if bi < len(lower_branch_su) else 22
                bn = ChainNode(uid=f"{uid_prefix}-br-l-{bi}", su_type=su,
                              axial=(bq, br), pos2d=HexGrid.axial_to_cart(bq, br),
                              meta={'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'branch_type': node_types[1], 'branch_kind': 'tail', 'position_idx': int(bi)})
                lb_nodes.append(bn)
                self._placed.add((bq, br))
            edge_lb = EdgeBranch(base=lower_24_node.uid, chain=lb_nodes)
            self.state.graph.branch.append(edge_lb)
            self.state.graph.chains.extend(lb_nodes)

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

    def _parse_inter_types(self, desc: str) -> List[Optional[str]]:
        """Parse inter types from description like 'V-ring(A+C+D)'.

        Returns [inter_right_type, inter_left_type].
        Each is '24_X' string if the position is SU24, or None if SU23.

        Examples:
          'V-ring(A+C+D)' → ['24_C', '24_D']  (both inter are 24)
          'V-ring(A+C)'   → ['24_C', None]     (right=24, left=23)
          'V-ring(A)'     → [None, None]        (both inter are 23)
        """
        types: List[Optional[str]] = [None, None]
        if '+' in desc:
            parts = desc.replace('V-ring(', '').replace(')', '').split('+')
            # Skip the first part (A-type first_24 anchor)
            inter_parts = parts[1:]
            for i, p in enumerate(inter_parts[:2]):
                p = p.strip()
                if p in ('A', 'B', 'C', 'D'):
                    types[i] = '24_' + p
        return types

    def _parse_side_ring_types(self, desc: str) -> List[str]:
        """Parse types from description like 'S-ring(A+B)' or 'Fused-S-ring(Base:AA+Br:CC)'."""
        types = []
        if desc.startswith('Fused-S-ring'):
            import re
            m = re.search(r'Base:([ABCD]+)\+Br:([ABCD]+)', desc)
            if m:
                base_str = m.group(1)
                # Just take the first two bases for standard side ring fallback geometry
                for char in base_str:
                    types.append('24_' + char)
        elif '+' in desc:
            parts = desc.replace('S-ring(', '').replace(')', '').split('+')
            for p in parts:
                p = p.strip()
                if p in ('A', 'B', 'C', 'D'):
                    types.append('24_' + p)
        while len(types) < 2:
            types.append('24_A')
        return types[:2]

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
        n_pos1.meta = {
            'stage': 'branch',
            'origin_type': getattr(spec, 'origin_type', None),
            'branch_type': base_node_types[0] if int(ring_su[0]) == 24 else None,
            'ring_role': 'base_upper',
        }
        self._placed.add((p1q, p1r))

        p3q, p3r = ring['bridge_upper_24']
        n_pos3 = ChainNode(uid=f"{uid_prefix}-b-bru", su_type=ring_su[1], axial=(p3q, p3r), pos2d=HexGrid.axial_to_cart(p3q, p3r))
        n_pos3.meta = {
            'stage': 'branch',
            'origin_type': getattr(spec, 'origin_type', None),
            'branch_type': bridge_node_types[0] if int(ring_su[1]) == 24 else None,
            'ring_role': 'bridge_upper',
        }
        self._placed.add((p3q, p3r))

        p4q, p4r = ring['bridge_lower_24']
        n_pos4 = ChainNode(uid=f"{uid_prefix}-b-brl", su_type=ring_su[2], axial=(p4q, p4r), pos2d=HexGrid.axial_to_cart(p4q, p4r))
        n_pos4.meta = {
            'stage': 'branch',
            'origin_type': getattr(spec, 'origin_type', None),
            'branch_type': bridge_node_types[1] if int(ring_su[2]) == 24 else None,
            'ring_role': 'bridge_lower',
        }
        self._placed.add((p4q, p4r))

        p2q, p2r = ring['base_lower_24']
        n_pos2 = ChainNode(uid=f"{uid_prefix}-b-pl", su_type=ring_su[3], axial=(p2q, p2r), pos2d=HexGrid.axial_to_cart(p2q, p2r))
        n_pos2.meta = {
            'stage': 'branch',
            'origin_type': getattr(spec, 'origin_type', None),
            'branch_type': base_node_types[1] if int(ring_su[3]) == 24 else None,
            'ring_role': 'base_lower',
        }
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
                bn.meta = {
                    'stage': 'branch',
                    'origin_type': getattr(spec, 'origin_type', None),
                    'branch_type': base_node_types[0],
                    'branch_kind': 'tail',
                    'position_idx': int(bi),
                }
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
                bn.meta = {
                    'stage': 'branch',
                    'origin_type': getattr(spec, 'origin_type', None),
                    'branch_type': base_node_types[1],
                    'branch_kind': 'tail',
                    'position_idx': int(bi),
                }
                lb_nodes.append(bn)
                self._placed.add((bq, br))
            self.state.graph.branch.append(EdgeBranch(base=n_pos2.uid, chain=lb_nodes))
            self.state.graph.chains.extend(lb_nodes)

        # Outer Ring
        if has_outer:
            outer_ring_su = action['outer_ring_su']
            
            ou23q, ou23r = ring['outer_upper_23']
            on_pos_u23 = ChainNode(uid=f"{uid_prefix}-o-u23", su_type=outer_ring_su[0], axial=(ou23q, ou23r), pos2d=HexGrid.axial_to_cart(ou23q, ou23r))
            on_pos_u23.meta = {'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'ring_role': 'outer_upper_23'}
            self._placed.add((ou23q, ou23r))
            
            ou24q, ou24r = ring['outer_upper_24']
            on_pos_u24 = ChainNode(uid=f"{uid_prefix}-o-u24", su_type=outer_ring_su[1], axial=(ou24q, ou24r), pos2d=HexGrid.axial_to_cart(ou24q, ou24r))
            on_pos_u24.meta = {
                'stage': 'branch',
                'origin_type': getattr(spec, 'origin_type', None),
                'branch_type': outer_node_types[0] if int(outer_ring_su[1]) == 24 else None,
                'ring_role': 'outer_upper_24',
            }
            self._placed.add((ou24q, ou24r))
            
            ol24q, ol24r = ring['outer_lower_24']
            on_pos_l24 = ChainNode(uid=f"{uid_prefix}-o-l24", su_type=outer_ring_su[2], axial=(ol24q, ol24r), pos2d=HexGrid.axial_to_cart(ol24q, ol24r))
            on_pos_l24.meta = {
                'stage': 'branch',
                'origin_type': getattr(spec, 'origin_type', None),
                'branch_type': outer_node_types[1] if int(outer_ring_su[2]) == 24 else None,
                'ring_role': 'outer_lower_24',
            }
            self._placed.add((ol24q, ol24r))
            
            ol23q, ol23r = ring['outer_lower_23']
            on_pos_l23 = ChainNode(uid=f"{uid_prefix}-o-l23", su_type=outer_ring_su[3], axial=(ol23q, ol23r), pos2d=HexGrid.axial_to_cart(ol23q, ol23r))
            on_pos_l23.meta = {'stage': 'branch', 'origin_type': getattr(spec, 'origin_type', None), 'ring_role': 'outer_lower_23'}
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
                    bn.meta = {
                        'stage': 'branch',
                        'origin_type': getattr(spec, 'origin_type', None),
                        'branch_type': outer_node_types[0],
                        'branch_kind': 'tail',
                        'position_idx': int(bi),
                    }
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
                    bn.meta = {
                        'stage': 'branch',
                        'origin_type': getattr(spec, 'origin_type', None),
                        'branch_type': outer_node_types[1],
                        'branch_kind': 'tail',
                        'position_idx': int(bi),
                    }
                    olb_nodes.append(bn)
                    self._placed.add((bq, br))
                self.state.graph.branch.append(EdgeBranch(base=on_pos_l24.uid, chain=olb_nodes))
                self.state.graph.chains.extend(olb_nodes)

        self._done += 1
        self.state.stage_step += 1
        return True
