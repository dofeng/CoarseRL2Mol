from typing import List, Dict, Optional, Tuple, Set
import copy
import random
import math

from .RL_state import (
    MCTSState, ChainNode, EdgeSide, EdgeBranch, HexGrid, HEX_VERTEX_OFFSETS,
    RU, RD, LU, LD, UP, DN, OPPOSITE,
)
from .RL_allocator import FlexAllocator, ChainSpec
from .stage_branch import (
    horizontal_branch_coords, su25_extra_branch_coord, get_branch_info_from_chain_spec
)

# ---------------------------------------------------------------
# Hex vertex connectivity rules (honeycomb lattice)
# ---------------------------------------------------------------
_TYPE_A_DIRS = (RD, UP, LD)   # (1,0), (0,1), (-1,-1)
_TYPE_B_DIRS = (LU, DN, RU)   # (-1,0), (0,-1), (1,1)


def _valid_dirs(q: int, r: int) -> Tuple[Tuple[int, int], ...]:
    """Return the 3 valid neighbour directions for hex vertex (q, r)."""
    mod = (q + r) % 3
    if mod == 1:
        return _TYPE_A_DIRS
    elif mod == 2:
        return _TYPE_B_DIRS
    return ()


# ---------------------------------------------------------------
# Scoring weights (configurable)
# ---------------------------------------------------------------
SIDE_SCORE_WEIGHTS = {
    'peripheral': 3.0,    # reward ports far from global centroid
    'repulsion':  4.0,    # reward ports far from existing side chains
    'completion': 50.0,   # penalty per missing side chain
}


# ---------------------------------------------------------------
# MCTSState → raw_mol adapter (for NMR prediction)
# ---------------------------------------------------------------
def mcts_state_to_raw_mol(
    state: MCTSState,
    global_elem_ratio=None,
) -> Dict:
    """Convert an MCTSState to the raw_mol dict expected by NMRPredictor.

    This mirrors the local connection-graph adapter logic but works
    with the RL_MTCS data structures (Site.uid, ChainNode.uid, etc.).
    """
    try:
        import torch
        from model.coarse_graph import NUM_SU_TYPES, PPM_AXIS
    except ImportError:
        return {}

    if global_elem_ratio is None:
        global_elem_ratio = torch.zeros(6)
    else:
        global_elem_ratio = global_elem_ratio.float()
        if global_elem_ratio.sum() > 1.1:
            global_elem_ratio = global_elem_ratio / global_elem_ratio.sum().clamp(min=1.0)

    uid_to_idx: Dict[str, int] = {}
    x_rows = []
    su_type_list: List[int] = []

    def _add(uid: str, su_type: int):
        if uid in uid_to_idx:
            return uid_to_idx[uid]
        idx = len(uid_to_idx)
        uid_to_idx[uid] = idx
        oh = torch.zeros(NUM_SU_TYPES)
        if 0 <= su_type < NUM_SU_TYPES:
            oh[su_type] = 1.0
        x_rows.append(torch.cat([oh, global_elem_ratio]))
        su_type_list.append(su_type)
        return idx

    # 1. Cluster sites
    for c in state.graph.clusters:
        if not c.placed:
            continue
        for s in c.sites:
            _add(s.uid, s.su_type)

    # 2. Chain nodes (flex + side + branch)
    for cn in state.graph.chains:
        _add(cn.uid, cn.su_type)

    edges: Set[Tuple[int, int]] = set()

    # Intra-cluster edges
    for c in state.graph.clusters:
        if not c.placed:
            continue
        for a, b in c.edges:
            ia = uid_to_idx.get(c.sites[a].uid)
            ib = uid_to_idx.get(c.sites[b].uid)
            if ia is not None and ib is not None:
                edges.add((min(ia, ib), max(ia, ib)))

    # Rigid edges
    for e in state.graph.rigid:
        u = uid_to_idx.get(e.u)
        v = uid_to_idx.get(e.v)
        if u is not None and v is not None:
            edges.add((min(u, v), max(u, v)))

    # Flex edges
    for e in state.graph.flex:
        prev = uid_to_idx.get(e.u)
        for cn in e.chain:
            cur = uid_to_idx.get(cn.uid)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur
        v = uid_to_idx.get(e.v)
        if prev is not None and v is not None:
            edges.add((min(prev, v), max(prev, v)))

    # Side edges
    for e in state.graph.side:
        prev = uid_to_idx.get(e.u)
        for cn in e.chain:
            cur = uid_to_idx.get(cn.uid)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur

    # Branch edges
    for e in state.graph.branch:
        prev = uid_to_idx.get(e.base)
        for cn in e.chain:
            cur = uid_to_idx.get(cn.uid)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur
        if e.target is not None:
            t = uid_to_idx.get(e.target)
            if prev is not None and t is not None:
                edges.add((min(prev, t), max(prev, t)))

    if not x_rows:
        return {}

    x = torch.stack(x_rows)
    su_t = torch.tensor(su_type_list, dtype=torch.long)
    is_carbon = (su_t <= 25)

    su_hist = torch.zeros(NUM_SU_TYPES)
    for st in su_type_list:
        if 0 <= st < NUM_SU_TYPES:
            su_hist[st] += 1

    from .utils import SU_ELEMENT_COUNTS
    total_atom_counts = torch.zeros(6)
    for st in su_type_list:
        if st in SU_ELEMENT_COUNTS:
            for i in range(6):
                total_atom_counts[i] += SU_ELEMENT_COUNTS[st][i]

    if edges:
        ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        ea = torch.zeros((ei.shape[1], 2), dtype=torch.float)
    else:
        ei = torch.empty(2, 0, dtype=torch.long)
        ea = torch.empty(0, 2)

    return {
        "edge_index": ei,
        "edge_attr": ea,
        "x": x,
        "is_carbon": is_carbon,
        "su_hist": su_hist,
        "total_atom_counts": total_atom_counts,
        "y_spectrum": torch.zeros_like(PPM_AXIS),
    }


# ===============================================================
# SideStage
# ===============================================================

class SideStage:
    """MCTS Stage for placing side chains.

    Path generation strictly obeys hex-vertex connectivity.
    Scoring uses spatial distribution (peripheral + repulsion).
    """

    def __init__(self, state: MCTSState, allocator: FlexAllocator):
        self.state = state
        self.allocator = allocator

        alloc_result = allocator._result

        self._side_pool: List[ChainSpec] = []
        self._extra_side_pool: List[ChainSpec] = []
        if alloc_result:
            for chain in alloc_result.side_chains:
                # Accept 'side' type chains (allocator uses 'side', not 'side_chain')
                # Also accept 'branch_side' type for chains with 24/25 branches
                if chain.chain_type in ('side', 'branch_side'):
                    self._side_pool.append(chain)
                    if 'extra' in chain.origin_type:
                        self._extra_side_pool.append(chain)

        # Sort: longer chains first
        self._side_pool.sort(key=lambda c: len(c.composition), reverse=True)

        self._n_sides_total = len(self._side_pool)
        self._sides_done = 0

        # Collision tracking
        self._placed_vertices: Set[Tuple[int, int]] = set(
            self.state.get_placed_vertices()
        )

        # uid → site lookup
        self._uid_to_site: Dict[str, 'Site'] = {}
        for c in self.state.graph.clusters:
            for s in c.sites:
                self._uid_to_site[s.uid] = s

        # Per-cluster axial sets
        self._cluster_axials: Dict[int, Set[Tuple[int, int]]] = {}
        for c in self.state.graph.clusters:
            self._cluster_axials[c.id] = {s.axial for s in c.sites}

        # Pre-compute global centroid of the backbone (clusters + flex chains)
        self._global_centroid = self._compute_global_centroid()

        # Track side chain attachment positions (Cartesian) for repulsion
        self._side_attach_positions: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------
    def clone(self) -> 'SideStage':
        """Deep copy for beam expansion."""
        new = SideStage.__new__(SideStage)
        new.state = self.state.copy()
        new.allocator = self.allocator
        new._side_pool = list(self._side_pool)  # ChainSpec are read-only
        new._extra_side_pool = list(getattr(self, '_extra_side_pool', []))
        new._n_sides_total = self._n_sides_total
        new._sides_done = self._sides_done
        new._placed_vertices = set(self._placed_vertices)
        new._uid_to_site = self._uid_to_site
        new._cluster_axials = self._cluster_axials
        new._global_centroid = self._global_centroid
        new._side_attach_positions = list(self._side_attach_positions)
        return new

    def is_done(self) -> bool:
        return self._sides_done >= self._n_sides_total

    def _get_active_chain(self) -> Optional[ChainSpec]:
        if self._sides_done >= self._n_sides_total:
            return None
        return self._side_pool[self._sides_done]

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------
    def _compute_global_centroid(self) -> Tuple[float, float]:
        """Centroid of all backbone atoms (cluster sites + flex chain nodes)."""
        xs, ys, n = 0.0, 0.0, 0
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            for s in c.sites:
                xs += s.pos2d[0]
                ys += s.pos2d[1]
                n += 1
        for cn in self.state.graph.chains:
            xs += cn.pos2d[0]
            ys += cn.pos2d[1]
            n += 1
        if n == 0:
            return (0.0, 0.0)
        return (xs / n, ys / n)

    def _peripheral_score(self, site_pos: Tuple[float, float]) -> float:
        """Score ∈ [0,1].  Higher = site farther from global centroid."""
        cx, cy = self._global_centroid
        dx = site_pos[0] - cx
        dy = site_pos[1] - cy
        dist = math.sqrt(dx * dx + dy * dy)
        # Normalise: score = 1 - 1/(1 + dist/scale)
        return 1.0 - 1.0 / (1.0 + dist / 3.0)

    def _repulsion_score(self, site_pos: Tuple[float, float]) -> float:
        """Score ∈ [0,1].  Higher = site farther from existing side chains."""
        if not self._side_attach_positions:
            return 1.0  # first side chain — no penalty
        min_dist = float('inf')
        for sx, sy in self._side_attach_positions:
            d = math.sqrt((site_pos[0] - sx) ** 2 + (site_pos[1] - sy) ** 2)
            if d < min_dist:
                min_dist = d
        # Sigmoidal: score = 1 - exp(-min_dist / scale)
        return 1.0 - math.exp(-min_dist / 2.5)

    def _spatial_score(self, site_pos: Tuple[float, float]) -> float:
        """Combined spatial score for candidate ranking."""
        w = SIDE_SCORE_WEIGHTS
        p = self._peripheral_score(site_pos) * w['peripheral']
        r = self._repulsion_score(site_pos) * w['repulsion']
        return p + r

    @staticmethod
    def _qr_shape_score_from_points(points: Set[Tuple[int, int]],
                                    min_ratio: float = 0.9,
                                    max_ratio: float = 2.3) -> float:
        if not points:
            return 1.0
        qs = [int(q) for q, _ in points]
        rs = [int(r) for _, r in points]
        q_span = float(max(qs) - min(qs))
        r_span = float(max(rs) - min(rs))
        if q_span < 1e-6 and r_span < 1e-6:
            return 1.0
        if r_span < 1e-6:
            return -1.0
        ratio = float(q_span / r_span)
        if min_ratio <= ratio <= max_ratio:
            mid = math.sqrt(min_ratio * max_ratio)
            return max(0.0, 1.0 - abs(ratio - mid) / max(mid, 1e-6))
        if ratio < min_ratio:
            return -min(1.0, (min_ratio - ratio) / max(min_ratio, 1e-6))
        return -min(1.0, (ratio - max_ratio) / max(max_ratio, 1e-6))

    def _preview_qr_shape_score(self, coords: List[Tuple[int, int]]) -> float:
        points = set((int(q), int(r)) for q, r in self._placed_vertices)
        points.update((int(q), int(r)) for q, r in coords)
        return self._qr_shape_score_from_points(points, 0.9, 2.3)

    def _preview_uniformity_score(self, coords: List[Tuple[int, int]], bins: int = 3) -> float:
        points = set((int(q), int(r)) for q, r in self._placed_vertices)
        points.update((int(q), int(r)) for q, r in coords)
        if len(points) <= 1:
            return 1.0
        qs = [q for q, _ in points]
        rs = [r for _, r in points]
        q0, q1 = min(qs), max(qs)
        r0, r1 = min(rs), max(rs)
        q_span = max(1.0, float(q1 - q0))
        r_span = max(1.0, float(r1 - r0))
        n_bins = max(2, int(bins))
        counts = [[0 for _ in range(n_bins)] for _ in range(n_bins)]
        for q, r in points:
            qi = min(n_bins - 1, int(((float(q) - float(q0)) / q_span) * n_bins))
            ri = min(n_bins - 1, int(((float(r) - float(r0)) / r_span) * n_bins))
            counts[qi][ri] += 1
        flat = [c for row in counts for c in row]
        occupied = sum(1 for c in flat if c > 0)
        avg = float(sum(flat)) / float(len(flat))
        variance = float(sum((c - avg) ** 2 for c in flat)) / float(len(flat))
        occupied_score = float(occupied) / float(len(flat))
        variance_penalty = min(1.0, variance / max(avg * avg + 1e-6, 1.0))
        return max(0.0, min(1.0, 0.65 * occupied_score + 0.35 * (1.0 - variance_penalty)))

    # ------------------------------------------------------------------
    # Outward direction
    # ------------------------------------------------------------------
    def _get_outward_dir(self, cluster, site) -> Optional[Tuple[int, int]]:
        """Return the unique outward direction for *site* on *cluster*."""
        q, r = site.axial
        valid = _valid_dirs(q, r)
        if not valid:
            return None

        ring_axials = self._cluster_axials.get(cluster.id, set())
        outward = [d for d in valid
                   if (q + d[0], r + d[1]) not in ring_axials]

        if len(outward) == 1:
            return outward[0]
        if len(outward) > 1:
            cx, cy = cluster.centroid()
            px, py = site.pos2d
            vx, vy = px - cx, py - cy
            best_d, best_dot = outward[0], -1e9
            for d in outward:
                dx, dy = HexGrid.axial_to_cart(*d)
                dot = vx * dx + vy * dy
                if dot > best_dot:
                    best_dot = dot
                    best_d = d
            return best_d
        return None

    # ------------------------------------------------------------------
    # Candidate generation (with spatial scoring)
    # ------------------------------------------------------------------
    def get_candidates(self, k: int = 10) -> List[Dict]:
        """Generate valid placement candidates scored by spatial quality."""
        candidates: List[Dict] = []
        chain_spec = self._get_active_chain()
        if not chain_spec:
            return []
        su_types = list(chain_spec.composition[1:])

        # Check if chain has 24/25 branch
        branch_info = get_branch_info_from_chain_spec(chain_spec)
        
        if branch_info:
            branch_nodes_count = branch_info['branch_23_count'] + branch_info['branch_22_count']
            branch_nodes_count += branch_info.get('extra_22_count', 0)
            main_chain_len = len(chain_spec.composition) - 1 - branch_nodes_count
        else:
            main_chain_len = len(chain_spec.composition) - 1

        if main_chain_len <= 0:
            return []
            
        # Helper logic for geometry heuristics
        is_long = main_chain_len >= 5  # e.g. >= 3 CH2 groups
        is_short = main_chain_len <= 1 # e.g. 11-22

        # 1. Collect available ports
        available_ports = []
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            for i, site in enumerate(c.sites):
                if not site.occupied and site.su_type in (11, 13, 10):
                    outward = self._get_outward_dir(c, site)
                    if outward is not None:
                        available_ports.append((c, i, site, outward))

        if not available_ports:
            return []
        
        # 3. Generate paths and score with spatial heuristic
        cx, cy = self._global_centroid
        
        for cluster, site_idx, site, outward_dir in available_ports:
            # Only horizontal directions support branches (chain length > 2)
            if branch_info and outward_dir in (UP, DN) and main_chain_len > 2:
                continue  # Skip vertical directions for chains with branches
                
            paths = self._generate_linear_paths(
                site.axial, outward_dir, main_chain_len
            )
            
            # Compute outward alignment (dot product with vector from center)
            px, py = site.pos2d
            vx, vy = px - cx, py - cy
            dist_to_center = math.hypot(vx, vy)
            dx, dy = HexGrid.axial_to_cart(*outward_dir)
            dir_len = math.hypot(dx, dy)
            outward_alignment = (vx * dx + vy * dy) / (dist_to_center * dir_len + 1e-9)
            
            for path in paths:
                if self._check_collision(path):
                    continue
                
                # Generate branch coordinates if chain has 24/25
                branch_coords = []
                extra_22_coord = None
                if branch_info and outward_dir in (RU, RD, LU, LD):
                    branch_coords = horizontal_branch_coords(
                        path, branch_info['position_idx'],
                        outward_dir, branch_info['branch_type'],
                        branch_info['branch_23_count'] + branch_info['branch_22_count']
                    )
                    # Check branch collision
                    if self._check_collision(branch_coords):
                        continue
                    
                    # For SU25, get extra -22 branch coordinate
                    if branch_info.get('extra_22_count', 0) > 0:
                        extra_22_coord = su25_extra_branch_coord(
                            path, branch_info['position_idx'],
                            outward_dir, branch_info['branch_type']
                        )
                        if extra_22_coord and extra_22_coord in self._placed_vertices:
                            continue
                
                candidate_sus = [int(su_types[idx] if idx < len(su_types) else 23) for idx in range(len(path))]
                if branch_info:
                    candidate_sus += [23] * int(branch_info.get('branch_23_count', 0))
                    candidate_sus += [22] * int(branch_info.get('branch_22_count', 0))
                    if branch_info.get('su_type') == 25 and extra_22_coord is not None:
                        candidate_sus += [22]
                if not self._spec_counts_match(chain_spec, candidate_sus):
                    continue

                spatial = self._spatial_score(site.pos2d)
                
                # Base candidate score is spatial
                cand_score = spatial
                
                # Bonus for pointing outwards away from the global centroid
                cand_score += outward_alignment * 1.5

                preview_coords = list(path) + list(branch_coords)
                if extra_22_coord is not None:
                    preview_coords.append(extra_22_coord)
                cand_score += 3.0 * self._preview_qr_shape_score(preview_coords)
                cand_score += 2.0 * self._preview_uniformity_score(preview_coords)
                
                # Penalty for long chains going vertically
                is_vertical = outward_dir in (UP, DN)
                if is_long and is_vertical:
                    cand_score -= 2.0
                    
                # Strict penalty for ultra-short chains pointing inwards or not being peripheral enough
                if is_short:
                    if outward_alignment < 0.2:
                        cand_score -= 4.0  # huge penalty for pointing inwards/tangentially
                    cand_score += self._peripheral_score(site.pos2d) * 5.0  # extra weight for peripheral
                
                candidates.append({
                    'type': 'side_placement',
                    'cluster_id': cluster.id,
                    'site_idx': site_idx,
                    'path': path,
                    'chain_spec': chain_spec,
                    'direction': outward_dir,
                    'spatial_score': cand_score,
                    'score': -(cand_score + random.uniform(0, 0.5)),
                    'branch_info': branch_info,
                    'branch_coords': branch_coords,
                    'extra_22_coord': extra_22_coord,
                })

        # Lower score = better (sorted ascending, then truncated)
        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    # ------------------------------------------------------------------
    # Hex-correct linear path generation
    # ------------------------------------------------------------------
    def _generate_linear_paths(
        self,
        start: Tuple[int, int],
        outward_dir: Tuple[int, int],
        length: int,
    ) -> List[List[Tuple[int, int]]]:
        """Generate strictly linear zigzag paths extending outward."""
        if length <= 0:
            return []

        if outward_dir in (RU, RD, LU, LD):
            pair_map = {RU: RD, RD: RU, LU: LD, LD: LU}
            step1, step2 = outward_dir, pair_map[outward_dir]
            path, curr = [], start
            for i in range(length):
                d = step1 if i % 2 == 0 else step2
                curr = (curr[0] + d[0], curr[1] + d[1])
                path.append(curr)
            return [path]

        elif outward_dir in (UP, DN):
            caps = (RU, LU) if outward_dir == UP else (RD, LD)
            paths = []
            for cap in caps:
                path, curr = [], start
                for i in range(length):
                    d = outward_dir if i % 2 == 0 else cap
                    curr = (curr[0] + d[0], curr[1] + d[1])
                    path.append(curr)
                paths.append(path)
            return paths

        return []

    def _check_collision(self, path: List[Tuple[int, int]]) -> bool:
        """Return True if any coordinate in *path* overlaps placed vertices."""
        for p in path:
            if p in self._placed_vertices:
                return True
        return False

    @staticmethod
    def _count_su_values(values: List[int]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for v in values:
            iv = int(v)
            counts[iv] = counts.get(iv, 0) + 1
        return counts

    @staticmethod
    def _spec_counts_match(chain_spec: ChainSpec, values: List[int]) -> bool:
        counts = SideStage._count_su_values(values)
        return (
            int(counts.get(22, 0)) == int(chain_spec.n_22) and
            int(counts.get(23, 0)) == int(chain_spec.n_23) and
            int(counts.get(24, 0)) == int(chain_spec.n_24) and
            int(counts.get(25, 0)) == int(chain_spec.n_25)
        )

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------
    def step(self, action: Dict) -> bool:
        """Execute one side-chain placement."""
        if action.get('type') != 'side_placement':
            return False

        cluster_id = action['cluster_id']
        site_idx = action['site_idx']
        path = action['path']
        chain_spec = action['chain_spec']
        branch_info = action.get('branch_info')
        branch_coords = action.get('branch_coords', [])
        extra_22_coord = action.get('extra_22_coord')
        su_types = list(chain_spec.composition[1:])

        expected_sus = [int(su_types[idx] if idx < len(su_types) else 23) for idx in range(len(path))]
        if branch_info:
            expected_sus += [23] * int(branch_info.get('branch_23_count', 0))
            expected_sus += [22] * int(branch_info.get('branch_22_count', 0))
            if branch_info.get('su_type') == 25 and extra_22_coord is not None:
                expected_sus += [22]
        if not self._spec_counts_match(chain_spec, expected_sus):
            return False

        cluster = self.state.graph.clusters[cluster_id]
        site = cluster.sites[site_idx]

        # Mark attachment site as occupied
        site.occupied = True
        site.su_type = 11

        # Record attachment position for repulsion scoring
        self._side_attach_positions.append(site.pos2d)

        # Build ChainNode objects for main chain
        chain_nodes: List[ChainNode] = []
        
        # Get the length of main chain body (excluding branch parts from composition)
        main_chain_len = len(path)

        for idx, p in enumerate(path):
            self._placed_vertices.add(p)
            pos2d = HexGrid.axial_to_cart(*p)
            su = su_types[idx] if idx < len(su_types) else 23
            node_meta = {
                'stage': 'side',
                'origin_type': getattr(chain_spec, 'origin_type', None),
                'position_idx': int(idx),
            }
            if branch_info and int(idx) == int(branch_info.get('position_idx', -1)) and int(su) in (24, 25):
                node_meta['branch_type'] = branch_info.get('branch_type')
                node_meta['branch_kind'] = 'main'
            node = ChainNode(
                uid=f"Side-{cluster.id}-{site_idx}-{idx}",
                su_type=su,
                axial=p,
                pos2d=pos2d,
                meta=node_meta,
            )
            chain_nodes.append(node)

        edge = EdgeSide(u=site.uid, chain=chain_nodes)
        self.state.graph.side.append(edge)
        self.state.graph.chains.extend(chain_nodes)

        branch_base_node = chain_nodes[branch_info['position_idx']] if branch_info else None

        # Create branch nodes if chain has 24/25
        if branch_info and branch_coords:
            # Build branch chain nodes
            branch_nodes: List[ChainNode] = []
            branch_23_count = branch_info['branch_23_count']
            branch_22_count = branch_info['branch_22_count']
            
            branch_su_list = [23] * branch_23_count + [22] * branch_22_count
            
            for bi, bc in enumerate(branch_coords):
                self._placed_vertices.add(bc)
                pos2d = HexGrid.axial_to_cart(*bc)
                su = branch_su_list[bi] if bi < len(branch_su_list) else 22
                bn = ChainNode(
                    uid=f"Side-{cluster.id}-{site_idx}-br-{bi}",
                    su_type=su,
                    axial=bc,
                    pos2d=pos2d,
                    meta={
                        'stage': 'side_branch',
                        'origin_type': getattr(chain_spec, 'origin_type', None),
                        'branch_type': branch_info.get('branch_type'),
                        'branch_kind': 'tail',
                        'position_idx': int(bi),
                    },
                )
                branch_nodes.append(bn)
            
            if branch_nodes:
                branch_edge = EdgeBranch(base=branch_base_node.uid, chain=branch_nodes)
                self.state.graph.branch.append(branch_edge)
                self.state.graph.chains.extend(branch_nodes)
            
            # Handle SU25 extra -22 branch
        if branch_info and branch_base_node is not None and branch_info.get('su_type') == 25 and extra_22_coord:
            self._placed_vertices.add(extra_22_coord)
            pos2d = HexGrid.axial_to_cart(*extra_22_coord)
            extra_node = ChainNode(
                uid=f"Side-{cluster.id}-{site_idx}-br-extra",
                su_type=22,
                axial=extra_22_coord,
                pos2d=pos2d,
                meta={
                    'stage': 'side_branch',
                    'origin_type': getattr(chain_spec, 'origin_type', None),
                    'branch_type': branch_info.get('branch_type'),
                    'branch_kind': 'extra_22',
                },
            )
            extra_edge = EdgeBranch(base=branch_base_node.uid, chain=[extra_node])
            self.state.graph.branch.append(extra_edge)
            self.state.graph.chains.append(extra_node)

        self._sides_done += 1
        self.state.stage_step += 1
        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self) -> float:
        """Combined score: completion + spatial uniformity.

        Higher = better.
        """
        w = SIDE_SCORE_WEIGHTS
        # Completion penalty
        missing = self._n_sides_total - self._sides_done
        completion = -missing * w['completion']

        # Spatial uniformity bonus (average repulsion of placed chains)
        if len(self._side_attach_positions) >= 2:
            total_min_dist = 0.0
            for i, (ax, ay) in enumerate(self._side_attach_positions):
                min_d = float('inf')
                for j, (bx, by) in enumerate(self._side_attach_positions):
                    if i == j:
                        continue
                    d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if d < min_d:
                        min_d = d
                total_min_dist += min_d
            avg_min_dist = total_min_dist / len(self._side_attach_positions)
            uniformity = min(1.0, avg_min_dist / 4.0)
        else:
            uniformity = 1.0

        return completion + uniformity * 10.0

    def get_result(self) -> Dict:
        """Return a summary dict."""
        return {
            'sides_placed': self._sides_done,
            'sides_total': self._n_sides_total,
            'side_edges': len(self.state.graph.side),
            'side_chain_nodes': sum(len(e.chain) for e in self.state.graph.side),
        }
        
