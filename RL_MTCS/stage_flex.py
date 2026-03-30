import math
import random
import copy
from typing import List, Dict, Tuple, Optional, Set

from .RL_state import (
    MCTSState, AromaticCluster, ChainNode, EdgeFlex, EdgeBranch, HexGrid,
    HEX_VERTEX_OFFSETS, OPPOSITE,
    RU, RD, LU, LD, UP, DN,
)
from .RL_allocator import ChainSpec, MAX_23_PER_CHAIN
from .stage_branch import (
    horizontal_branch_coords, su25_extra_branch_coord, get_branch_info_from_chain_spec
)

# Direction name mapping
DIR_NAME = {
    RU: 'right_up', RD: 'right_down',
    LU: 'left_up',  LD: 'left_down',
    UP: 'up',        DN: 'down',
}

V_MAX_23 = 4

# ==================== Chain Geometry ====================

def horizontal_chain(start_11: Tuple[int,int], main: Tuple[int,int], m: int = 1):
    """
    Build horizontal zigzag chain on hex grid.
    main must be one of RU, RD, LU, LD.
    Returns: (coords_23, end11, last_step)
    """
    assert main in (RU, RD, LU, LD), f"main must be horizontal direction, got {main}"
    
    pair_map = {RU: RD, RD: RU, LU: LD, LD: LU}
    step1 = main
    step_pair = pair_map[main]
    
    coords_23 = []
    cur = start_11
    
    if m == 0:
        end11 = (cur[0] + step1[0], cur[1] + step1[1])
        return coords_23, end11, step1
    
    # First 23 along main direction
    cur = (cur[0] + step1[0], cur[1] + step1[1])
    coords_23.append(cur)
    
    # Subsequent: alternate step1 and step_pair
    for i in range(2, m + 1):
        step = step_pair if i % 2 == 0 else step1
        cur = (cur[0] + step[0], cur[1] + step[1])
        coords_23.append(cur)
    
    last_step = step1 if m % 2 == 0 else step_pair
    end11 = (cur[0] + last_step[0], cur[1] + last_step[1])
    
    return coords_23, end11, last_step


def vertical_chain(start_11: Tuple[int,int], vertical: Tuple[int,int],
                    m: int = 1, cap: Optional[Tuple[int,int]] = None):
    """
    Build vertical zigzag chain on hex grid.
    vertical must be UP or DN.
    Returns: (coords_23, end11, last_step)
    """
    assert vertical in (UP, DN)
    if cap is None:
        cap = RU if vertical == UP else RD
    
    coords_23 = []
    cur = start_11
    
    if m == 0:
        end11 = (cur[0] + vertical[0], cur[1] + vertical[1])
        return coords_23, end11, vertical
    
    # Alternate vertical and cap
    directions = []
    for i in range(m):
        directions.append(vertical if i % 2 == 0 else cap)
    
    for d in directions:
        cur = (cur[0] + d[0], cur[1] + d[1])
        coords_23.append(cur)
    
    # End direction: opposite type of last step
    last_dir = directions[-1]
    end_dir = cap if last_dir == vertical else vertical
    end11 = (cur[0] + end_dir[0], cur[1] + end_dir[1])
    
    return coords_23, end11, end_dir


def _build_displacement_table(max_h: int = MAX_23_PER_CHAIN, max_v: int = V_MAX_23):
    """
    Pre-compute reverse lookup: displacement (dq, dr) -> [(dir_class, main_dir, m, cap), ...]
    For each chain direction and length, compute start→end displacement and index it.
    """
    table = {}  # (dq, dr) -> list of (dir_class, main_dir, m, cap)
    origin = (0, 0)

    # Horizontal chains: RU, RD, LU, LD
    for main_dir in [RU, RD, LU, LD]:
        for m in range(1, max_h + 1):
            _, end11, _ = horizontal_chain(origin, main_dir, m)
            table.setdefault(end11, []).append(('H', main_dir, m, None))

    # Vertical chains: UP, DN with various caps
    v_caps = {UP: [RU, LU], DN: [RD, LD]}
    for vert_dir in [UP, DN]:
        for cap in v_caps[vert_dir]:
            for m in range(1, max_v + 1):
                _, end11, _ = vertical_chain(origin, vert_dir, m, cap)
                table.setdefault(end11, []).append(('V', vert_dir, m, cap))

    return table

# Module-level displacement table (computed once)
_DISP_TABLE = _build_displacement_table()


def get_site_outward_direction(cluster: AromaticCluster,
                                site_idx: int) -> Optional[Tuple[int,int]]:
    """
    Get the outward direction of a site relative to its ring center(s).
    For single-ring sites: direct offset from ring center.
    For bridgehead (multi-ring): pick the outermost ring's offset.
    """
    site = cluster.sites[site_idx]
    sq, sr = site.axial
    
    # Build ring_offsets: ring_idx -> (dq, dr) offset from that ring center
    ring_offsets = {}
    for ring_idx, (cq, cr) in enumerate(cluster.centers):
        dq, dr = sq - cq, sr - cr
        if (dq, dr) in HEX_VERTEX_OFFSETS:
            ring_offsets[ring_idx] = (dq, dr)
    
    if not ring_offsets:
        return None
    
    ring_indices = sorted(ring_offsets.keys())
    
    if len(ring_indices) == 1:
        offset = ring_offsets[ring_indices[0]]
    else:
        # Multi-ring (bridgehead): pick outermost
        edge_rings = [ring_indices[0], ring_indices[-1]]
        best_offset, best_score = None, -1
        for ri in edge_rings:
            o = ring_offsets[ri]
            score = abs(o[0]) + abs(o[1])
            if score > best_score:
                best_score = score
                best_offset = o
        offset = best_offset
    
    if offset not in HEX_VERTEX_OFFSETS:
        return None
    return offset


# ==================== Flex Stage ====================

class FlexStage:
    """
    Flexible connection stage: connect all rigid clusters via pre-allocated
    bridge chains from RL_allocator. Cross-component placement is prioritized
    until the rigid scaffold is connected; secondary/intra-component closures
    are considered afterwards.
    """
    
    def __init__(self, state: MCTSState, original_su_counts: Dict[int, int],
                 allocation_result=None, skip_init: bool = False):
        """
        Args:
            state: MCTSState after rigid stage
            original_su_counts: original SU histogram
            allocation_result: AllocationResult from RL_allocator (bridge_chains, side_chains, etc.)
            skip_init: lightweight init for MCTS copies
        """
        self.state = state
        self.original_su_counts = original_su_counts.copy()
        self.allocation_result = allocation_result  # Store for _adjust_bridge_pool
        self._cluster_map: Dict[int, AromaticCluster] = {
            c.id: c for c in state.graph.clusters
        }
        
        # Track placed vertices (sites + chain nodes)
        self._placed_vertices: Set[Tuple[int, int]] = set()
        self._placed_centers: Set[Tuple[int, int]] = set()
        self._flex_used_sites: Dict[int, Set[int]] = {}  # cluster_id -> set of used site indices
        self._seed_root: Optional[int] = None
        
        # Build uid → cluster_id cache for O(1) lookup
        self._uid_to_cluster: Dict[str, int] = {}
        for c in state.graph.clusters:
            for s in c.sites:
                self._uid_to_cluster[s.uid] = c.id
        
        self._init_placed_vertices()
        
        # Build rigid cluster map. Clusters inside the same rigid cluster may
        # also be connected by flex if the bridge is sufficiently long.
        self._rigid_cluster_map: Dict[int, int] = self._build_rigid_cluster_map()
        
        if skip_init:
            self._bridge_pool: List[int] = []
            self._bridge_specs: List[ChainSpec] = []
            self._n_bridges_total = 0
            self._n_connections_needed = 0
            self._11_consumed = 0
            self._23_consumed = 0
            self._bridges_done = 0
            self._last_direction: Optional[str] = None
            self._connections_made = 0
            # rigid cluster map already built above
            return
        
        # --- Resource initialization from allocator ---
        alloc = allocation_result
        
        all_bridges = list(alloc.bridge_chains) if alloc else []
        
        # Preserve allocator intent: keep mandatory bridges ahead of extras.
        self._bridge_specs: List[ChainSpec] = self._prioritize_bridge_specs(all_bridges)
        # Also keep length pool for backward compatibility
        self._bridge_pool = [ch.n_23 for ch in self._bridge_specs]
        self._n_bridges_total = len(all_bridges)
        
        # Connectivity info
        all_roots = set(state._find(c.id) for c in state.graph.clusters)
        n_components = len(all_roots)
        self._n_connections_needed = max(0, n_components - 1)
        
        # Track consumed resources
        self._11_consumed = 0
        self._23_consumed = 0
        self._bridges_done = 0
        self._last_direction: Optional[str] = None
        self._connections_made = 0
        
        # KEY: Unplace non-seed clusters so flex can translate them into position
        self._unplace_non_seed_clusters()
        
        self._print_init_info()
    
    def _print_init_info(self):
        n = len(self.state.graph.clusters)
        n_placed = sum(1 for c in self.state.graph.clusters if c.placed)
        n_unplaced = n - n_placed
        print(f"[FlexStage] Init: {n} clusters ({n_placed} placed, {n_unplaced} unplaced), "
              f"{self._n_connections_needed+1} components, {self._n_bridges_total} bridges")

    def clone(self) -> 'FlexStage':
        """Deep-copy the current flex stage for unified MCTS expansion."""
        return self._clone_for_action_validation()

    def _prioritize_bridge_specs(self, bridge_specs: List[ChainSpec]) -> List[ChainSpec]:
        """Keep non-extra bridges in allocator order and postpone extras."""
        ordered = list(bridge_specs or [])
        mandatory = [spec for spec in ordered if getattr(spec, 'origin_type', None) != 'extra']
        extras = [spec for spec in ordered if getattr(spec, 'origin_type', None) == 'extra']
        return mandatory + extras
    
    def _build_rigid_cluster_map(self) -> Dict[int, int]:
        """Build a mapping: aromatic_cluster_id -> rigid_cluster_root.
        Two aromatic clusters are in the same rigid cluster iff they are
        connected by a path of rigid (10-10) edges. Uses a local union-find
        independent of the main state UF (which also includes flex unions).
        """
        parent: Dict[int, int] = {}
        for c in self.state.graph.clusters:
            parent[c.id] = c.id

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in self.state.graph.rigid:
            ca, cb = edge.cluster_a, edge.cluster_b
            if ca >= 0 and cb >= 0:
                _union(ca, cb)

        return {cid: _find(cid) for cid in parent}

    def _same_rigid_cluster_flex_allowed(self, cluster_a_id: int, cluster_b_id: int,
                                         chain_len: int) -> bool:
        if self._rigid_cluster_map.get(cluster_a_id) != self._rigid_cluster_map.get(cluster_b_id):
            return True
        if cluster_a_id == cluster_b_id:
            return False
        return chain_len >= 4

    def _init_placed_vertices(self):
        """Collect all currently placed vertices and centers."""
        for c in self.state.graph.clusters:
            if c.placed:
                for s in c.sites:
                    self._placed_vertices.add(s.axial)
                for center in c.centers:
                    self._placed_centers.add(center)
        # Chain nodes from rigid stage (if any)
        for cn in self.state.graph.chains:
            self._placed_vertices.add(cn.axial)
    
    def _unplace_non_seed_clusters(self):
        """
        Remove non-seed-component clusters from placed sets.
        This is CRITICAL: after rigid stage's place_all_remaining(),
        all clusters are scattered. The flex stage should grow outward
        from the seed component, translating target clusters into position.
        Unplacing them prevents collision with scattered clusters.
        """
        # Find the largest connected component (seed)
        comp_sizes = {}  # root -> total_rings
        for c in self.state.graph.clusters:
            if c.placed:
                root = self.state._find(c.id)
                comp_sizes[root] = comp_sizes.get(root, 0) + c.rings
        
        if not comp_sizes:
            return
        
        self._seed_root = max(comp_sizes.keys(), key=lambda r: comp_sizes[r])
        
        # Unplace all clusters NOT in the seed component
        unplaced_count = 0
        for c in self.state.graph.clusters:
            if c.placed and self.state._find(c.id) != self._seed_root:
                # Remove from placed sets
                for s in c.sites:
                    self._placed_vertices.discard(s.axial)
                for center in c.centers:
                    self._placed_centers.discard(center)
                c.placed = False
                unplaced_count += 1
        
        # Silently track unplaced count (no verbose print)
        self._unplaced_count = unplaced_count
    
    # ==================== Component Helpers ====================
    
    def _get_cluster(self, cid: int) -> Optional[AromaticCluster]:
        return self._cluster_map.get(cid)
    
    def _get_component_roots(self) -> Dict[int, List[int]]:
        """Get component map: root -> [cluster_ids]"""
        comp = {}
        for c in self.state.graph.clusters:
            root = self.state._find(c.id)
            comp.setdefault(root, []).append(c.id)
        return comp
    
    def _get_component_size(self, root: int) -> int:
        """Get total ring count for a component."""
        total = 0
        for c in self.state.graph.clusters:
            if self.state._find(c.id) == root:
                total += c.rings
        return total
    
    def _get_seed_root(self) -> Optional[int]:
        """Get the root of the largest (seed) component. Updated after each union."""
        if self._seed_root is not None:
            # Return current seed root (follows union-find)
            return self.state._find(self._seed_root)
        comp = self._get_component_roots()
        if not comp:
            return None
        self._seed_root = max(comp.keys(), key=lambda r: self._get_component_size(r))
        return self._seed_root
    
    def _is_site_available(self, cluster: AromaticCluster, site_idx: int,
                            min_gap: int = 2) -> bool:
        """Check if a site is available for flex connection.
        Enforces minimum gap between flex-used sites on the same cluster.
        """
        site = cluster.sites[site_idx]
        if site.occupied:
            return False
        if site.su_type not in (13,):
            return False
        # Check min_gap to previously used flex sites on this cluster
        used = self._flex_used_sites.get(cluster.id, set())
        if site_idx in used:
            return False
        for used_idx in used:
            used_site = cluster.sites[used_idx]
            dist = HexGrid.distance(site.axial, used_site.axial)
            if dist < min_gap:
                return False
        return True
    
    # ==================== Collision Detection ====================
    
    def _check_chain_collision(self, coords_23: List[Tuple[int,int]],
                                end11: Tuple[int,int],
                                skip_end11: bool = False) -> bool:
        """Check if chain coordinates collide with existing placements.
        skip_end11: for secondary (intra-component) connections where end11
                    is an already-placed target site.
        """
        for p in coords_23:
            if p in self._placed_vertices or p in self._placed_centers:
                return True
        if not skip_end11:
            if end11 in self._placed_vertices or end11 in self._placed_centers:
                return True
        return False

    def _uid_to_root(self, uid: str) -> Optional[int]:
        cid = self._site_uid_to_cluster_id(uid)
        if cid is not None:
            return self.state._find(cid)
        for edge in self.state.graph.flex:
            if any(cn.uid == uid for cn in edge.chain):
                edge_cid = self._site_uid_to_cluster_id(edge.u)
                if edge_cid is not None:
                    return self.state._find(edge_cid)
                edge_cid = self._site_uid_to_cluster_id(edge.v)
                if edge_cid is not None:
                    return self.state._find(edge_cid)
        for edge in self.state.graph.side:
            if any(cn.uid == uid for cn in edge.chain):
                return self._uid_to_root(edge.u)
        for edge in self.state.graph.branch:
            if any(cn.uid == uid for cn in edge.chain):
                return self._uid_to_root(edge.base)
        return None
    
    def _component_chain_nodes(self, root: int) -> List[ChainNode]:
        """Get all chain nodes belonging to a component (from flex/side/branch edges)."""
        nodes: List[ChainNode] = []
        seen: Set[str] = set()
        for edge in self.state.graph.flex:
            u_cid = self._site_uid_to_cluster_id(edge.u)
            v_cid = self._site_uid_to_cluster_id(edge.v)
            if u_cid is not None and self.state._find(u_cid) == root:
                for cn in edge.chain:
                    if cn.uid not in seen:
                        nodes.append(cn)
                        seen.add(cn.uid)
            elif v_cid is not None and self.state._find(v_cid) == root:
                for cn in edge.chain:
                    if cn.uid not in seen:
                        nodes.append(cn)
                        seen.add(cn.uid)
        for edge in self.state.graph.side:
            if self._uid_to_root(edge.u) == root:
                for cn in edge.chain:
                    if cn.uid not in seen:
                        nodes.append(cn)
                        seen.add(cn.uid)
        for edge in self.state.graph.branch:
            base_root = self._uid_to_root(edge.base)
            target_root = self._uid_to_root(edge.target) if edge.target else None
            if base_root == root or target_root == root:
                for cn in edge.chain:
                    if cn.uid not in seen:
                        nodes.append(cn)
                        seen.add(cn.uid)
        return nodes

    def _build_uid_axial_map(self) -> Dict[str, Tuple[int, int]]:
        mapping: Dict[str, Tuple[int, int]] = {}
        for c in self.state.graph.clusters:
            for s in c.sites:
                mapping[s.uid] = s.axial
        for cn in self.state.graph.chains:
            mapping[cn.uid] = cn.axial
        return mapping

    def _axials_to_cart(self, axials: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        return [HexGrid.axial_to_cart(*p) for p in axials]

    def _segments_intersect(self, ax, ay, bx, by, cx, cy, dx, dy) -> bool:
        def ccw(px, py, qx, qy, rx, ry):
            return (ry - py) * (qx - px) > (qy - py) * (rx - px)

        eps = 1e-6
        if (abs(ax - cx) < eps and abs(ay - cy) < eps) or \
           (abs(ax - dx) < eps and abs(ay - dy) < eps) or \
           (abs(bx - cx) < eps and abs(by - cy) < eps) or \
           (abs(bx - dx) < eps and abs(by - dy) < eps):
            return False

        if ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy):
            return True

        return False

    def _polylines_intersect(self, line_a: List[Tuple[float, float]],
                              line_b: List[Tuple[float, float]]) -> bool:
        if len(line_a) < 2 or len(line_b) < 2:
            return False
        for i in range(len(line_a) - 1):
            ax, ay = line_a[i]
            bx, by = line_a[i + 1]
            for j in range(len(line_b) - 1):
                cx, cy = line_b[j]
                dx, dy = line_b[j + 1]
                if self._segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    return True
        return False

    def _collect_existing_edge_lines(self, exclude_root: Optional[int] = None) -> List[List[Tuple[float, float]]]:
        uid_to_axial = self._build_uid_axial_map()
        lines: List[List[Tuple[float, float]]] = []
        for edge in self.state.graph.rigid:
            ru = self._uid_to_root(edge.u)
            rv = self._uid_to_root(edge.v)
            if exclude_root is not None and (ru == exclude_root or rv == exclude_root):
                continue
            if edge.line and len(edge.line) >= 2:
                lines.append(list(edge.line))
                continue
            a = uid_to_axial.get(edge.u)
            b = uid_to_axial.get(edge.v)
            if a is not None and b is not None:
                lines.append(self._axials_to_cart([a, b]))
        for edge in self.state.graph.flex:
            ru = self._uid_to_root(edge.u)
            rv = self._uid_to_root(edge.v)
            if exclude_root is not None and (ru == exclude_root or rv == exclude_root):
                continue
            a = uid_to_axial.get(edge.u)
            b = uid_to_axial.get(edge.v)
            if a is None or b is None:
                continue
            axials = [a] + [cn.axial for cn in edge.chain] + [b]
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        for edge in self.state.graph.side:
            ru = self._uid_to_root(edge.u)
            if exclude_root is not None and ru == exclude_root:
                continue
            a = uid_to_axial.get(edge.u)
            if a is None:
                continue
            axials = [a] + [cn.axial for cn in edge.chain]
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        for edge in self.state.graph.branch:
            base_root = self._uid_to_root(edge.base)
            target_root = self._uid_to_root(edge.target) if edge.target else None
            if exclude_root is not None and (base_root == exclude_root or target_root == exclude_root):
                continue
            base = uid_to_axial.get(edge.base)
            if base is None:
                continue
            axials = [base] + [cn.axial for cn in edge.chain]
            if edge.target:
                target = uid_to_axial.get(edge.target)
                if target is not None:
                    axials.append(target)
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        return lines

    def _collect_component_edge_lines(self, root: int, dq: int, dr: int) -> List[List[Tuple[float, float]]]:
        uid_to_axial = self._build_uid_axial_map()
        lines: List[List[Tuple[float, float]]] = []

        def shift(axial: Tuple[int, int]) -> Tuple[int, int]:
            return (axial[0] + dq, axial[1] + dr)

        for edge in self.state.graph.rigid:
            ru = self._uid_to_root(edge.u)
            rv = self._uid_to_root(edge.v)
            if ru != root or rv != root:
                continue
            a = uid_to_axial.get(edge.u)
            b = uid_to_axial.get(edge.v)
            if a is not None and b is not None:
                lines.append(self._axials_to_cart([shift(a), shift(b)]))
        for edge in self.state.graph.flex:
            ru = self._uid_to_root(edge.u)
            rv = self._uid_to_root(edge.v)
            if ru != root or rv != root:
                continue
            a = uid_to_axial.get(edge.u)
            b = uid_to_axial.get(edge.v)
            if a is None or b is None:
                continue
            axials = [shift(a)] + [shift(cn.axial) for cn in edge.chain] + [shift(b)]
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        for edge in self.state.graph.side:
            ru = self._uid_to_root(edge.u)
            if ru != root:
                continue
            a = uid_to_axial.get(edge.u)
            if a is None:
                continue
            axials = [shift(a)] + [shift(cn.axial) for cn in edge.chain]
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        for edge in self.state.graph.branch:
            base_root = self._uid_to_root(edge.base)
            target_root = self._uid_to_root(edge.target) if edge.target else None
            if base_root != root and target_root != root:
                continue
            base = uid_to_axial.get(edge.base)
            if base is None:
                continue
            axials = [shift(base)] + [shift(cn.axial) for cn in edge.chain]
            if edge.target:
                target = uid_to_axial.get(edge.target)
                if target is not None:
                    axials.append(shift(target))
            if len(axials) >= 2:
                lines.append(self._axials_to_cart(axials))
        return lines

    def _preview_branch_geometry(self, coords_23: List[Tuple[int, int]],
                                  direction: str,
                                  outward_dir: Optional[Tuple[int, int]],
                                  chain_spec: Optional[ChainSpec]) -> Tuple[List[Tuple[int, int]], List[List[Tuple[float, float]]]]:
        if chain_spec is None or direction != 'H' or len(coords_23) <= 2:
            return [], []
        branch_info = get_branch_info_from_chain_spec(chain_spec)
        if not branch_info or outward_dir not in (RU, RD, LU, LD):
            return [], []
        position_idx = branch_info['position_idx']
        if position_idx >= len(coords_23):
            return [], []
        branch_coords = horizontal_branch_coords(
            coords_23,
            position_idx,
            outward_dir,
            branch_info['branch_type'],
            branch_info['branch_23_count'] + branch_info['branch_22_count'],
        )
        all_coords = list(branch_coords)
        base_coord = coords_23[position_idx]
        lines: List[List[Tuple[float, float]]] = []
        if branch_coords:
            lines.append(self._axials_to_cart([base_coord] + branch_coords))
        if branch_info.get('extra_22_count', 0) > 0:
            extra_coord = su25_extra_branch_coord(
                coords_23,
                position_idx,
                outward_dir,
                branch_info['branch_type'],
            )
            if extra_coord is not None:
                all_coords.append(extra_coord)
                lines.append(self._axials_to_cart([base_coord, extra_coord]))
        return all_coords, lines

    def _validate_flex_proposal(self,
                                 cluster_a: AromaticCluster,
                                 site_a_idx: int,
                                 cluster_b: AromaticCluster,
                                 site_b_idx: int,
                                 coords_23: List[Tuple[int, int]],
                                 end11: Tuple[int, int],
                                 translation: Tuple[int, int],
                                 direction: str,
                                 outward_dir: Optional[Tuple[int, int]],
                                 chain_spec: Optional[ChainSpec]) -> bool:
        if not coords_23:
            return False
        if len(set(coords_23)) != len(coords_23):
            return False
        if end11 in set(coords_23):
            return False

        site_a = cluster_a.sites[site_a_idx]
        site_b = cluster_b.sites[site_b_idx]
        src_root = self.state._find(cluster_a.id)
        tgt_root = self.state._find(cluster_b.id)
        is_cross = src_root != tgt_root
        tq, tr = translation

        stationary_verts = set(self._placed_vertices)
        stationary_centers = set(self._placed_centers)
        moved_verts: Set[Tuple[int, int]] = set()
        moved_centers: Set[Tuple[int, int]] = set()
        moving_root: Optional[int] = None

        if is_cross:
            moving_root = tgt_root
            for c in self.state.graph.clusters:
                if self.state._find(c.id) != moving_root:
                    continue
                for s in c.sites:
                    stationary_verts.discard(s.axial)
                    moved_verts.add((s.axial[0] + tq, s.axial[1] + tr))
                for center in c.centers:
                    stationary_centers.discard(center)
                    moved_centers.add((center[0] + tq, center[1] + tr))
            for cn in self._component_chain_nodes(moving_root):
                stationary_verts.discard(cn.axial)
                moved_verts.add((cn.axial[0] + tq, cn.axial[1] + tr))
        elif cluster_b.placed:
            if tq != 0 or tr != 0:
                return False
        else:
            for s in cluster_b.sites:
                moved_verts.add((s.axial[0] + tq, s.axial[1] + tr))
            for center in cluster_b.centers:
                moved_centers.add((center[0] + tq, center[1] + tr))

        translated_target = (site_b.axial[0] + tq, site_b.axial[1] + tr)
        if translated_target != end11:
            return False

        branch_coords, branch_lines = self._preview_branch_geometry(
            coords_23,
            direction,
            outward_dir,
            chain_spec,
        )
        proposal_verts = list(coords_23) + list(branch_coords)
        if len(set(proposal_verts)) != len(proposal_verts):
            return False
        if any(p == end11 for p in branch_coords):
            return False

        for p in proposal_verts:
            if p in stationary_verts or p in stationary_centers:
                return False
            if p in moved_centers:
                return False
            if p in moved_verts and p != end11:
                return False

        if moved_verts:
            overlap = {v for v in moved_verts if v in stationary_verts}
            if overlap - {end11}:
                return False
        if moved_centers & stationary_centers:
            return False

        main_line = self._axials_to_cart([site_a.axial] + list(coords_23) + [end11])
        proposal_lines = [main_line] + branch_lines

        for i in range(len(proposal_lines)):
            for j in range(i + 1, len(proposal_lines)):
                if self._polylines_intersect(proposal_lines[i], proposal_lines[j]):
                    return False

        existing_lines = self._collect_existing_edge_lines(exclude_root=moving_root if is_cross else None)
        for line in proposal_lines:
            for other in existing_lines:
                if self._polylines_intersect(line, other):
                    return False

        if moving_root is not None and (tq != 0 or tr != 0):
            moved_lines = self._collect_component_edge_lines(moving_root, tq, tr)
            for moved_line in moved_lines:
                for other in existing_lines:
                    if self._polylines_intersect(moved_line, other):
                        return False
                for line in proposal_lines:
                    if self._polylines_intersect(moved_line, line):
                        return False

        return True
    
    def _site_uid_to_cluster_id(self, uid: str) -> Optional[int]:
        """Extract cluster_id from site uid. Uses pre-built cache for O(1) lookup."""
        return self._uid_to_cluster.get(uid)
    
    def _would_collide_component(self, root: int, dq: int, dr: int,
                                  allowed: Optional[Set[Tuple[int,int]]] = None) -> bool:
        """Check if translating a component would collide.
        NOTE: includes UNPLACED clusters in the component (they have coords but aren't on grid).
        """
        allowed = allowed or set()
        comp_verts = set()
        comp_centers = set()
        # Include ALL clusters in this component (placed or not)
        for c in self.state.graph.clusters:
            if self.state._find(c.id) == root:
                for s in c.sites:
                    comp_verts.add(s.axial)
                for center in c.centers:
                    comp_centers.add(center)
        # Include chain nodes
        for cn in self._component_chain_nodes(root):
            comp_verts.add(cn.axial)
        
        # Occupied = currently placed minus this component's current positions
        occupied_verts = self._placed_vertices - comp_verts
        occupied_centers = self._placed_centers - comp_centers
        
        for v in comp_verts:
            new_v = (v[0] + dq, v[1] + dr)
            if new_v in occupied_verts and new_v not in allowed:
                return True
        for c in comp_centers:
            new_c = (c[0] + dq, c[1] + dr)
            if new_c in occupied_centers:
                return True
        return False

    def _would_collide_single_cluster(self, cluster: AromaticCluster, dq: int, dr: int,
                                      allowed: Optional[Set[Tuple[int, int]]] = None) -> bool:
        """Check collision for translating a single cluster."""
        allowed = allowed or set()
        cluster_verts = {s.axial for s in cluster.sites}
        cluster_centers = set(cluster.centers)
        occupied_verts = set(self._placed_vertices)
        occupied_centers = set(self._placed_centers)
        if cluster.placed:
            occupied_verts -= cluster_verts
            occupied_centers -= cluster_centers
        for v in cluster_verts:
            new_v = (v[0] + dq, v[1] + dr)
            if new_v in occupied_verts and new_v not in allowed:
                return True
        for c in cluster_centers:
            new_c = (c[0] + dq, c[1] + dr)
            if new_c in occupied_centers:
                return True
        return False
    
    def _translate_component(self, root: int, dq: int, dr: int):
        """Translate all clusters AND chain nodes in a component.
        Handles both placed and unplaced clusters. After translation,
        all clusters in the component are marked as placed.
        """
        # Gather chain nodes belonging to this component
        chain_nodes = self._component_chain_nodes(root)
        
        # Remove old positions of PLACED clusters
        for c in self.state.graph.clusters:
            if self.state._find(c.id) == root and c.placed:
                for s in c.sites:
                    self._placed_vertices.discard(s.axial)
                for center in c.centers:
                    self._placed_centers.discard(center)
        for cn in chain_nodes:
            self._placed_vertices.discard(cn.axial)
        
        # Translate ALL clusters in the component (placed or not)
        for c in self.state.graph.clusters:
            if self.state._find(c.id) == root:
                if dq != 0 or dr != 0:
                    c.translate(dq, dr)  # This also sets c.placed = True
                else:
                    c.placed = True
        
        # Translate chain nodes
        if dq != 0 or dr != 0:
            for cn in chain_nodes:
                cn.axial = (cn.axial[0] + dq, cn.axial[1] + dr)
                cn.pos2d = HexGrid.axial_to_cart(*cn.axial)
        
        # Re-register ALL positions (now all are placed)
        for c in self.state.graph.clusters:
            if self.state._find(c.id) == root:
                for s in c.sites:
                    self._placed_vertices.add(s.axial)
                for center in c.centers:
                    self._placed_centers.add(center)
        for cn in chain_nodes:
            self._placed_vertices.add(cn.axial)

    def _translate_single_cluster(self, cluster: AromaticCluster, dq: int, dr: int):
        """Translate a single cluster and mark it placed."""
        if cluster.placed:
            for s in cluster.sites:
                self._placed_vertices.discard(s.axial)
            for center in cluster.centers:
                self._placed_centers.discard(center)
        if dq != 0 or dr != 0:
            cluster.translate(dq, dr)
        else:
            cluster.placed = True
        for s in cluster.sites:
            self._placed_vertices.add(s.axial)
        for center in cluster.centers:
            self._placed_centers.add(center)
    
    # ==================== Target Finding ====================
    
    def _find_targets_for_end(self, end11: Tuple[int,int],
                               final_vec: Tuple[int,int],
                               src_root: int,
                               forbidden_clusters: Optional[Set[int]] = None,
                               allow_intra: bool = False,
                               candidate_ok = None,
                               max_results: int = 32,
                               ) -> List[Tuple[AromaticCluster, int, Tuple[int,int], int]]:
        """
        Find a target cluster for the chain endpoint.
        The target's component will be TRANSLATED to align its site with end11.
        Accepts both placed and unplaced clusters (unplaced = not yet on grid).
        Returns a bounded list of feasible targets ordered by placement preference.
        """
        if final_vec not in DIR_NAME:
            return []
        opposite = OPPOSITE.get(final_vec)
        if opposite is None:
            return []
        
        forbidden_clusters = forbidden_clusters or set()
        results: List[Tuple[AromaticCluster, int, Tuple[int, int], int]] = []
        seen: Set[Tuple[int, int, int, int, int]] = set()
        
        # Collect cross-component and intra-component candidates
        cross_candidates = []
        intra_candidates = []
        for c in self.state.graph.clusters:
            if c.id in forbidden_clusters:
                continue
            site_indices = self._candidate_sites_for_direction(c, opposite)
            if not site_indices:
                continue
            if self.state._find(c.id) != src_root:
                cross_candidates.append((c, site_indices))
            elif allow_intra:
                intra_candidates.append((c, site_indices))
        
        # Sort: prefer unplaced (easy to move), then larger clusters
        cross_candidates.sort(
            key=lambda item: (0 if not item[0].placed else 1, -item[0].rings, item[0].id)
        )
        intra_candidates.sort(key=lambda item: (-item[0].rings, item[0].id))
        
        # Try cross first, then intra
        for candidates in [cross_candidates, intra_candidates]:
            for cluster, site_indices in candidates:
                for si in site_indices:
                    if len(results) >= max_results:
                        return results
                    if not self._is_site_available(cluster, si):
                        continue
                    site = cluster.sites[si]
                    tq = end11[0] - site.axial[0]
                    tr = end11[1] - site.axial[1]
                    target_root = self.state._find(cluster.id)
                    
                    # Intra-component handling
                    if target_root == src_root:
                        if tq == 0 and tr == 0:
                            if candidate_ok is None or candidate_ok(cluster, si, (0, 0), target_root):
                                key = (cluster.id, si, 0, 0, target_root)
                                if key not in seen:
                                    seen.add(key)
                                    results.append((cluster, si, (0, 0), target_root))
                        if not cluster.placed and not self._would_collide_single_cluster(cluster, tq, tr, {end11}):
                            if candidate_ok is None or candidate_ok(cluster, si, (tq, tr), target_root):
                                key = (cluster.id, si, tq, tr, target_root)
                                if key not in seen:
                                    seen.add(key)
                                    results.append((cluster, si, (tq, tr), target_root))
                        continue
                    
                    # Cross-component: check translation feasibility
                    allowed = {end11}
                    if not self._would_collide_component(target_root, tq, tr, allowed):
                        if candidate_ok is None or candidate_ok(cluster, si, (tq, tr), target_root):
                            key = (cluster.id, si, tq, tr, target_root)
                            if key not in seen:
                                seen.add(key)
                                results.append((cluster, si, (tq, tr), target_root))
        
        return results
    
    def _candidate_sites_for_direction(self, cluster: AromaticCluster,
                                        direction_vec: Tuple[int,int]) -> List[int]:
        """Find site indices whose outward direction matches direction_vec."""
        result = []
        for idx, site in enumerate(cluster.sites):
            if site.su_type != 13 or site.occupied:
                continue
            outward = get_site_outward_direction(cluster, idx)
            if outward == direction_vec:
                result.append(idx)
        return result
    
    # ==================== Bridge Selection ====================

    def _bridge_spec_is_extra(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._bridge_specs):
            return False
        return getattr(self._bridge_specs[idx], 'origin_type', None) == 'extra'

    def _remaining_bridge_indices(self) -> List[int]:
        return list(range(self._bridges_done, len(self._bridge_specs)))

    def _activate_bridge_index(self, bridge_idx: int) -> bool:
        """Swap a selected remaining bridge into the active front position."""
        if bridge_idx < self._bridges_done or bridge_idx >= len(self._bridge_pool):
            return False
        current_idx = self._bridges_done
        if bridge_idx == current_idx:
            return True
        self._bridge_pool[current_idx], self._bridge_pool[bridge_idx] = (
            self._bridge_pool[bridge_idx],
            self._bridge_pool[current_idx],
        )
        self._bridge_specs[current_idx], self._bridge_specs[bridge_idx] = (
            self._bridge_specs[bridge_idx],
            self._bridge_specs[current_idx],
        )
        return True

    def _candidate_bridge_indices(self, max_bridges: int = 8) -> List[int]:
        """Select a bounded set of remaining bridges to branch on."""
        remaining = self._remaining_bridge_indices()
        if len(remaining) <= max_bridges:
            return remaining

        head = remaining[:max_bridges]
        mandatory = [idx for idx in remaining if not self._bridge_spec_is_extra(idx)]
        selected = []
        for idx in head + mandatory:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_bridges:
                break
        return selected

    def _build_fallback_lengths(self, planned_len: int,
                                is_extra: bool = False,
                                max_len: int = MAX_23_PER_CHAIN) -> List[int]:
        """
        Build list of chain lengths to try.
        Used only for extra bridges during dynamic rescue search.
        """
        lengths = [planned_len]
        min_len = 1
        if is_extra and planned_len <= 2:
            min_len = 2

        for l in range(planned_len + 1, min(planned_len + 5, max_len + 1)):
            if l not in lengths:
                lengths.append(l)

        for l in range(planned_len - 1, min_len - 1, -1):
            if l not in lengths:
                lengths.append(l)

        if min_len > 1 and 1 not in lengths:
            lengths.append(1)

        return lengths
    
    def _get_preferred_direction(self) -> Optional[str]:
        """
        Determine preferred direction with enhanced alternation.
        Returns 'H', 'V', or None (no preference).
        """
        ratio = self.state.get_qr_ratio()

        if ratio == float('inf') or ratio > 2.3:
            return 'V'  # Too wide, prefer vertical
        if ratio < 0.9:
            return 'H'  # Too tall, prefer horizontal
        
        # Enhanced alternation (80% probability)
        if self._last_direction is not None:
            if random.random() < 0.80:
                return 'V' if self._last_direction == 'H' else 'H'
        
        return None  # No preference
    
    # ==================== Candidate Generation ====================

    def _has_remaining_chains(self) -> bool:
        """Check if there are remaining bridge chains to place."""
        return self._bridges_done < len(self._bridge_pool)
    
    def _get_remaining_lengths(self, exclude_idx: Optional[int] = None) -> List[int]:
        """Get remaining chain lengths."""
        return [
            self._bridge_pool[i]
            for i in range(self._bridges_done, len(self._bridge_pool))
            if exclude_idx is None or i != exclude_idx
        ]

    def _all_clusters_placed_and_connected(self) -> bool:
        """Check if ALL clusters (not just placed ones) are placed and in one component.
        This is the true completion condition for the primary phase.
        """
        unplaced = [c for c in self.state.graph.clusters if not c.placed]
        if unplaced:
            return False
        return self.state.is_all_connected()

    def get_candidates(self, k: int = 40) -> List[Dict]:
        """Generate explicit bridge-placement actions for the remaining search tree."""
        if self.is_done():
            return []
        if not self._has_remaining_chains():
            return []
        k = min(10, max(1, int(k)))
        remaining = self._remaining_bridge_indices()
        bridge_windows = []
        connected = bool(self.state.is_all_connected())
        window_plan = (4, 8, len(remaining)) if not connected else (3, 6, len(remaining))
        for width in window_plan:
            if width <= 0:
                continue
            window = tuple(self._candidate_bridge_indices(max_bridges=width))
            if window and window not in bridge_windows:
                bridge_windows.append(window)

        for bridge_indices in bridge_windows:
            strict_cands = self._generate_all_candidates(k, strict_lengths=True, bridge_indices=list(bridge_indices))
            if strict_cands:
                self.state.stage_mode = 'exact'
                return strict_cands[:k]

        extra_indices = [idx for idx in remaining if self._bridge_spec_is_extra(idx)]
        if not extra_indices:
            return []

        dynamic_windows = []
        for width in ((4, len(extra_indices)) if not connected else (3, len(extra_indices))):
            if width <= 0:
                continue
            window = tuple(extra_indices[:width])
            if window and window not in dynamic_windows:
                dynamic_windows.append(window)

        for bridge_indices in dynamic_windows:
            dynamic_cands = self._generate_all_candidates(k, strict_lengths=False, bridge_indices=list(bridge_indices))
            if dynamic_cands:
                self.state.stage_mode = 'adjust_length'
                return dynamic_cands[:k]
        return []

    def _candidate_sort_key(self, cand: Dict, needs_connectivity: bool) -> Tuple:
        is_cross = bool(cand.get('is_cross_component', False))
        bridge_idx = int(cand.get('bridge_idx', self._bridges_done))
        is_extra = bool(cand.get('bridge_is_extra', False))
        return (
            0 if (not needs_connectivity or is_cross) else 1,
            1 if is_extra else 0,
            bridge_idx - self._bridges_done,
            cand.get('score', float('inf')),
        )

    def _generate_all_candidates(self, k: int, strict_lengths: bool,
                                 bridge_indices: Optional[List[int]] = None) -> List[Dict]:
        """Generate bounded actions over a selected subset of remaining bridges."""
        cands: List[Dict] = []
        has_unplaced = any(not c.placed for c in self.state.graph.clusters)
        needs_connectivity = has_unplaced or not self.state.is_all_connected()

        bridge_indices = bridge_indices or self._remaining_bridge_indices()
        if not bridge_indices:
            return []

        per_bridge_k = max(2, min(4 if needs_connectivity else 3, k))
        limit = max(k * 2, 20 if needs_connectivity else 12)
        for bridge_idx in bridge_indices:
            primary_cross = self._get_primary_candidates(
                k=per_bridge_k,
                bridge_idx=bridge_idx,
                allow_intra=False,
                strict_lengths=strict_lengths,
            )
            cands.extend(primary_cross)

            if not needs_connectivity:
                cands.extend(self._get_secondary_candidates(
                    k=per_bridge_k,
                    bridge_idx=bridge_idx,
                    strict_lengths=strict_lengths,
                ))
                cands.extend(self._get_primary_candidates(
                    k=per_bridge_k,
                    bridge_idx=bridge_idx,
                    allow_intra=True,
                    strict_lengths=strict_lengths,
                ))

            if len(cands) >= limit:
                break

        if not cands:
            return []

        seen = set()
        unique_cands = []
        for cand in cands:
            sig = (
                cand['bridge_idx'],
                cand['cluster_a_id'],
                cand['site_a_idx'],
                cand['cluster_b_id'],
                cand['site_b_idx'],
                cand['chain_len'],
                tuple(cand.get('coords_23', [])),
                cand.get('end11'),
                tuple(cand.get('translation', (0, 0))),
                cand.get('direction'),
            )
            if sig not in seen:
                seen.add(sig)
                unique_cands.append(cand)

        unique_cands = [cand for cand in unique_cands if self._action_resource_feasible(cand)]
        if not unique_cands:
            return []

        unique_cands.sort(key=lambda cand: self._candidate_sort_key(cand, needs_connectivity))
        validation_budget = max(k, min(limit, k * 2))
        executable: List[Dict] = []
        for cand in unique_cands[:validation_budget]:
            if self._action_step_feasible(cand):
                executable.append(cand)
            if len(executable) >= limit:
                break
        return executable[:limit]

    # ---------- Phase 1: Primary (cross-component) ----------

    def _get_primary_candidates(self, k: int = 40, bridge_idx: Optional[int] = None,
                                allow_intra: bool = False,
                                strict_lengths: bool = True) -> List[Dict]:
        """Generate flex connection candidates using outward chain growth.
        When allow_intra=True, also finds targets within the same component.
        """
        prefer_dir = self._get_preferred_direction()
        seed_root = self._get_seed_root()
        if bridge_idx is None:
            bridge_idx = self._bridges_done
        if bridge_idx < self._bridges_done or bridge_idx >= len(self._bridge_specs):
            return []

        current_chain_spec = self._bridge_specs[bridge_idx]
        planned_len = self._bridge_pool[bridge_idx]
        bridge_is_extra = self._bridge_spec_is_extra(bridge_idx)

        if seed_root is not None:
            all_sources = [c for c in self.state.graph.clusters
                           if c.placed and self.state._find(c.id) == seed_root]
        else:
            all_sources = [c for c in self.state.graph.clusters if c.placed]
        global_sources = [c for c in self.state.graph.clusters if c.placed]

        candidates = []
        source_passes = [
            (all_sources[:8], 4, prefer_dir),
            (all_sources, None, None),
            (global_sources, None, prefer_dir),
        ]

        for sources, site_limit, pass_prefer in source_passes:
            if not sources:
                continue
            random.shuffle(sources)

            for A in sources:
                free_sites = [(i, s) for i, s in enumerate(A.sites)
                              if s.su_type == 13 and not s.occupied]
                random.shuffle(free_sites)
                if site_limit is not None:
                    free_sites = free_sites[:site_limit]

                for si, site in free_sites:
                    if not self._is_site_available(A, si):
                        continue

                    axy = site.axial
                    outward = get_site_outward_direction(A, si)
                    if outward is None:
                        continue

                    src_root = self.state._find(A.id)

                    if outward in (RU, RD, LU, LD):
                        dir_class = 'H'
                        if pass_prefer == 'V':
                            continue
                        if strict_lengths or not bridge_is_extra:
                            try_lengths = [planned_len]
                        else:
                            try_lengths = self._build_fallback_lengths(planned_len, is_extra=True)

                    elif outward in (UP, DN):
                        dir_class = 'V'
                        if pass_prefer == 'H':
                            continue
                        if planned_len > V_MAX_23 and strict_lengths:
                            continue
                        if strict_lengths or not bridge_is_extra:
                            try_lengths = [planned_len]
                        else:
                            try_lengths = [
                                l for l in self._build_fallback_lengths(planned_len, is_extra=True)
                                if l <= V_MAX_23
                            ]
                    else:
                        continue

                    for m in try_lengths:
                        if m < 1:
                            continue
                        if dir_class == 'V' and m > V_MAX_23:
                            continue

                        # For V chains, try both cap directions
                        if dir_class == 'V':
                            caps_to_try = [RU, LU] if outward == UP else [RD, LD]
                        else:
                            caps_to_try = [None]  # H chains don't use cap

                        for v_cap in caps_to_try:
                            try:
                                if dir_class == 'H':
                                    coords_23, end11, _ = horizontal_chain(axy, outward, m)
                                else:
                                    coords_23, end11, _ = vertical_chain(axy, outward, m, v_cap)
                            except Exception:
                                continue

                            if not coords_23:
                                continue
                            # For intra-component, end11 may be an existing placed site
                            # — skip end11 collision if it's already a known vertex
                            skip_end = allow_intra and end11 in self._placed_vertices
                            if self._check_chain_collision(coords_23, end11, skip_end11=skip_end):
                                continue

                            final_vec = (end11[0] - coords_23[-1][0], end11[1] - coords_23[-1][1])
                            if final_vec not in DIR_NAME:
                                continue

                            def candidate_ok(Bcand, tgt_idx, trans, target_root):
                                if allow_intra and target_root == src_root:
                                    if not self._same_rigid_cluster_flex_allowed(A.id, Bcand.id, m):
                                        return False
                                return self._validate_flex_proposal(
                                    A,
                                    si,
                                    Bcand,
                                    tgt_idx,
                                    coords_23,
                                    end11,
                                    trans,
                                    dir_class,
                                    outward,
                                    current_chain_spec,
                                )

                            tgt_infos = self._find_targets_for_end(
                                end11, final_vec, src_root,
                                forbidden_clusters={A.id},
                                allow_intra=allow_intra,
                                candidate_ok=candidate_ok,
                                max_results=min(8, max(4, k // 6 + 2)),
                            )
                            if not tgt_infos:
                                continue

                            for B, tgt_idx, trans, target_root in tgt_infos:
                                remaining_lengths_after = self._get_remaining_lengths(exclude_idx=bridge_idx)
                                score = self._score_candidate(
                                    A, B, src_root, target_root,
                                    dir_class, m, planned_len or m,
                                    end11=end11,
                                    translation=trans,
                                    coords_23=coords_23,
                                    chain_spec=current_chain_spec,
                                    bridge_idx=bridge_idx,
                                    remaining_lengths_after=remaining_lengths_after,
                                )

                                candidates.append({
                                    'type': 'flex',
                                    'bridge_idx': bridge_idx,
                                    'bridge_origin_type': getattr(current_chain_spec, 'origin_type', ''),
                                    'bridge_is_extra': bridge_is_extra,
                                    'bridge_n23': int(getattr(current_chain_spec, 'n_23', planned_len)),
                                    'is_cross_component': src_root != target_root,
                                    'cluster_a_id': A.id,
                                    'site_a_idx': si,
                                    'cluster_b_id': B.id,
                                    'site_b_idx': tgt_idx,
                                    'chain_len': m,
                                    'coords_23': coords_23,
                                    'end11': end11,
                                    'translation': trans,
                                    'direction': dir_class,
                                    'outward_dir': outward,  # For branch coordinate generation
                                    'score': score,
                                })

                                if len(candidates) >= max(k * 2, 20):
                                    candidates.sort(key=lambda x: x['score'])
                                    return candidates[:max(k * 2, 20)]

        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    # ---------- Phase 2: Secondary (intra-component) ----------

    def _get_secondary_candidates(self, k: int = 40, bridge_idx: Optional[int] = None,
                                  strict_lengths: bool = True) -> List[Dict]:
        """Generate intra-component flex connection candidates via endpoint-pair
        displacement matching.  For every pair of placed clusters, compute
        displacement between free SU-13 sites and look up in _DISP_TABLE to
        find geometrically valid chain connections.

        When strict_lengths=True, only accept exact planned lengths.
        When strict_lengths=False, allow dynamic length adjustment (fallback mode).
        Length=1 chains are always strict (never adjusted).
        """
        if bridge_idx is None:
            bridge_idx = self._bridges_done
        if bridge_idx < self._bridges_done or bridge_idx >= len(self._bridge_specs):
            return []

        planned_len = self._bridge_pool[bridge_idx]
        bridge_is_extra = self._bridge_spec_is_extra(bridge_idx)
        current_chain_spec = self._bridge_specs[bridge_idx]

        if strict_lengths:
            all_acceptable = {planned_len}
        else:
            if bridge_is_extra:
                all_acceptable = set(
                    self._build_fallback_lengths(planned_len, is_extra=True)
                )
            else:
                all_acceptable = {planned_len}

        # Collect placed clusters with free SU13 sites and their outward directions
        # Each entry: (cluster, [(site_idx, site, outward_dir), ...])
        placed_with_sites = []
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            free_with_dir = []
            for i, s in enumerate(c.sites):
                if s.su_type != 13 or s.occupied:
                    continue
                if not self._is_site_available(c, i):
                    continue
                outward = get_site_outward_direction(c, i)
                if outward is None:
                    continue
                free_with_dir.append((i, s, outward))
            if free_with_dir:
                placed_with_sites.append((c, free_with_dir))

        candidates: List[Dict] = []

        for ai in range(len(placed_with_sites)):
            A, a_sites = placed_with_sites[ai]
            for bi in range(ai + 1, len(placed_with_sites)):
                B, b_sites = placed_with_sites[bi]

                if not self._same_rigid_cluster_flex_allowed(
                    A.id, B.id, max(all_acceptable) if all_acceptable else planned_len
                ):
                    continue

                for si_a, site_a, outward_a in a_sites:
                    axy = site_a.axial
                    for si_b, site_b, outward_b in b_sites:
                        bxy = site_b.axial
                        # --- direction A → B ---
                        disp_ab = (bxy[0] - axy[0], bxy[1] - axy[1])
                        self._try_secondary_match(
                            A, si_a, axy, outward_a,
                            B, si_b, bxy, outward_b,
                            disp_ab, all_acceptable, candidates,
                            bridge_idx, planned_len, bridge_is_extra, current_chain_spec)
                        # --- direction B → A ---
                        disp_ba = (-disp_ab[0], -disp_ab[1])
                        self._try_secondary_match(
                            B, si_b, bxy, outward_b,
                            A, si_a, axy, outward_a,
                            disp_ba, all_acceptable, candidates,
                            bridge_idx, planned_len, bridge_is_extra, current_chain_spec)
                        if len(candidates) >= max(k * 2, 20):
                            candidates.sort(key=lambda x: x['score'])
                            return candidates[:max(k * 2, 20)]

        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]

    def _try_secondary_match(self, A, si_a, axy, outward_a,
                              B, si_b, bxy, outward_b,
                              disp, available_lengths, out_list,
                              bridge_idx: int, planned_len: int,
                              bridge_is_extra: bool,
                              current_chain_spec: Optional[ChainSpec]):
        """Check displacement table for a valid chain from A-site to B-site,
        enforcing hexagonal direction rules, and append candidate(s) to out_list.

        Rules enforced:
        - Chain start direction must match source site A's outward direction.
        - Chain arrival direction at end11 must satisfy:
          OPPOSITE[arrival_dir] == target site B's outward direction.
        """
        matches = _DISP_TABLE.get(disp)
        if not matches:
            return
        for dir_class, main_dir, m, cap in matches:
            if m not in available_lengths:
                continue
            if not self._same_rigid_cluster_flex_allowed(A.id, B.id, m):
                continue

            # Rule 2a: Chain start direction must match source outward
            if dir_class == 'H':
                chain_start_dir = main_dir
            else:  # V
                chain_start_dir = main_dir  # main_dir is vertical dir (UP/DN) for V entries
            if chain_start_dir != outward_a:
                continue

            # Build chain geometry and verify endpoint
            try:
                if dir_class == 'H':
                    coords_23, end11, _ = horizontal_chain(axy, main_dir, m)
                else:
                    coords_23, end11, _ = vertical_chain(axy, main_dir, m, cap)
            except Exception:
                continue

            # Safety: end11 must equal target site position
            if end11 != bxy:
                continue
            if not coords_23:
                continue

            # Rule 2b: Chain arrival direction must match target outward
            # arrival_dir = direction from last_23 to end11
            arrival_dir = (end11[0] - coords_23[-1][0], end11[1] - coords_23[-1][1])
            # Target site B faces outward in outward_b; chain arrives from that direction
            # so arrival_dir must equal OPPOSITE[outward_b]
            expected_arrival = OPPOSITE.get(outward_b)
            if expected_arrival is None or arrival_dir != expected_arrival:
                continue

            # Collision check — skip end11 because it is the existing target site
            if self._check_chain_collision(coords_23, end11, skip_end11=True):
                continue

            if not self._validate_flex_proposal(
                A,
                si_a,
                B,
                si_b,
                coords_23,
                end11,
                (0, 0),
                dir_class,
                outward_a,
                current_chain_spec,
            ):
                continue

            src_root = self.state._find(A.id)
            tgt_root = self.state._find(B.id)

            remaining_lengths_after = self._get_remaining_lengths(exclude_idx=bridge_idx)
            score = self._score_candidate(
                A, B, src_root, tgt_root,
                dir_class, m, m, end11=end11,
                translation=(0, 0),
                coords_23=coords_23,
                chain_spec=current_chain_spec,
                bridge_idx=bridge_idx,
                remaining_lengths_after=remaining_lengths_after,
            )
            if m == planned_len:
                score -= 20
            else:
                score += 12 if m <= 2 else 5

            out_list.append({
                'type': 'flex',
                'bridge_idx': bridge_idx,
                'bridge_origin_type': getattr(current_chain_spec, 'origin_type', ''),
                'bridge_is_extra': bridge_is_extra,
                'bridge_n23': int(getattr(current_chain_spec, 'n_23', planned_len)),
                'is_cross_component': src_root != tgt_root,
                'cluster_a_id': A.id,
                'site_a_idx': si_a,
                'cluster_b_id': B.id,
                'site_b_idx': si_b,
                'chain_len': m,
                'coords_23': coords_23,
                'end11': end11,
                'translation': (0, 0),
                'direction': dir_class,
                'outward_dir': outward_a,  # For branch coordinate generation
                'score': score,
            })
    
    def _compute_center_of_mass(self) -> Tuple[float, float]:
        """Compute center of mass (in 2D cartesian) of all placed clusters."""
        sx, sy, cnt = 0.0, 0.0, 0
        for c in self.state.graph.clusters:
            if c.placed:
                for s in c.sites:
                    sx += s.pos2d[0]
                    sy += s.pos2d[1]
                    cnt += 1
        for cn in self.state.graph.chains:
            sx += cn.pos2d[0]
            sy += cn.pos2d[1]
            cnt += 1
        if cnt == 0:
            return (0.0, 0.0)
        return (sx / cnt, sy / cnt)
    
    def _get_sector_counts(self, cx: float, cy: float,
                            n_sectors: int = 8) -> List[int]:
        """Count placed clusters in each angular sector around (cx, cy).
        Returns list of counts per sector.
        """
        counts = [0] * n_sectors
        sector_angle = 2 * math.pi / n_sectors
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            # Use first center as cluster representative
            if c.centers:
                px, py = HexGrid.axial_to_cart(*c.centers[0])
            else:
                px, py = c.sites[0].pos2d if c.sites else (0, 0)
            dx, dy = px - cx, py - cy
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            angle = math.atan2(dy, dx)  # [-pi, pi]
            if angle < 0:
                angle += 2 * math.pi
            sector = min(int(angle / sector_angle), n_sectors - 1)
            counts[sector] += 1
        return counts
    
    def _count_local_density(self, target_pos: Tuple[float, float],
                              radius: float = 5.0) -> int:
        """Count placed vertices within radius of target_pos (2D cartesian)."""
        tx, ty = target_pos
        count = 0
        r2 = radius * radius
        for v in self._placed_vertices:
            vx, vy = HexGrid.axial_to_cart(*v)
            dx, dy = vx - tx, vy - ty
            if dx * dx + dy * dy <= r2:
                count += 1
        return count

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

    def _preview_qr_shape_score(self,
                                target_cluster: AromaticCluster,
                                src_root: int,
                                tgt_root: int,
                                translation: Tuple[int, int],
                                coords_23: List[Tuple[int, int]],
                                end11: Optional[Tuple[int, int]] = None) -> float:
        points: Set[Tuple[int, int]] = set()
        points.update((int(q), int(r)) for q, r in self._placed_vertices)
        points.update((int(q), int(r)) for q, r in self._placed_centers)

        tq, tr = translation
        if src_root != tgt_root:
            current_component: Set[Tuple[int, int]] = set()
            shifted_component: Set[Tuple[int, int]] = set()
            for c in self.state.graph.clusters:
                if self.state._find(c.id) != tgt_root:
                    continue
                for s in c.sites:
                    current_component.add((int(s.axial[0]), int(s.axial[1])))
                    shifted_component.add((int(s.axial[0] + tq), int(s.axial[1] + tr)))
                for center in c.centers:
                    current_component.add((int(center[0]), int(center[1])))
                    shifted_component.add((int(center[0] + tq), int(center[1] + tr)))
            for cn in self._component_chain_nodes(tgt_root):
                current_component.add((int(cn.axial[0]), int(cn.axial[1])))
                shifted_component.add((int(cn.axial[0] + tq), int(cn.axial[1] + tr)))
            points.difference_update(current_component)
            points.update(shifted_component)
        elif not target_cluster.placed and (tq != 0 or tr != 0):
            for s in target_cluster.sites:
                points.add((int(s.axial[0] + tq), int(s.axial[1] + tr)))
            for center in target_cluster.centers:
                points.add((int(center[0] + tq), int(center[1] + tr)))

        points.update((int(q), int(r)) for q, r in coords_23)
        if end11 is not None:
            points.add((int(end11[0]), int(end11[1])))
        return self._qr_shape_score_from_points(points, 0.9, 2.3)
    
    def _secondary_match_weight(self, outward_a, outward_b,
                                axy: Tuple[int, int], bxy: Tuple[int, int],
                                remaining_lengths: List[int],
                                source_cluster: Optional[AromaticCluster] = None,
                                source_idx: Optional[int] = None,
                                target_cluster: Optional[AromaticCluster] = None,
                                target_idx: Optional[int] = None,
                                target_translation: Tuple[int, int] = (0, 0)) -> float:
        disp = (bxy[0] - axy[0], bxy[1] - axy[1])
        matches = _DISP_TABLE.get(disp)
        if not matches:
            return 0.0
        best = 0.0
        for dir_class, main_dir, m, cap in matches:
            if m not in remaining_lengths:
                continue
            if dir_class == 'H':
                chain_start_dir = main_dir
                _, end11, _ = horizontal_chain(axy, main_dir, m)
            else:
                chain_start_dir = main_dir
                _, end11, _ = vertical_chain(axy, main_dir, m, cap)
            if chain_start_dir != outward_a or end11 != bxy:
                continue
            expected_arrival = OPPOSITE.get(outward_b)
            if expected_arrival is None:
                continue
            if dir_class == 'H':
                coords_23, _, _ = horizontal_chain(axy, main_dir, m)
            else:
                coords_23, _, _ = vertical_chain(axy, main_dir, m, cap)
            if not coords_23:
                continue
            arrival_dir = (bxy[0] - coords_23[-1][0], bxy[1] - coords_23[-1][1])
            if arrival_dir != expected_arrival:
                continue
            if source_cluster is not None and source_idx is not None and target_cluster is not None and target_idx is not None:
                if not self._validate_flex_proposal(
                    source_cluster,
                    source_idx,
                    target_cluster,
                    target_idx,
                    coords_23,
                    bxy,
                    target_translation,
                    dir_class,
                    outward_a,
                    None,
                ):
                    continue
            elif self._check_chain_collision(coords_23, bxy, skip_end11=True):
                continue
            weight = 120.0 if m == 1 else 90.0 if m == 2 else max(35.0, 80.0 - 8.0 * (m - 2))
            if weight > best:
                best = weight
        return best

    def _future_closure_reward(self, B: AromaticCluster, tgt_root: int,
                               translation: Tuple[int, int],
                               remaining_lengths: List[int]) -> float:
        if not remaining_lengths:
            return 0.0
        dq, dr = translation
        translated_sites = []
        for c in self.state.graph.clusters:
            if self.state._find(c.id) != tgt_root:
                continue
            for i, s in enumerate(c.sites):
                if s.su_type != 13 or s.occupied:
                    continue
                outward = get_site_outward_direction(c, i)
                if outward is None:
                    continue
                translated_sites.append(((s.axial[0] + dq, s.axial[1] + dr), outward, c, i))
        active_ports = []
        rigid_ids_in_target = set(
            self._rigid_cluster_map.get(c.id)
            for c in self.state.graph.clusters
            if self.state._find(c.id) == tgt_root
        )
        for c in self.state.graph.clusters:
            if not c.placed:
                continue
            if self._rigid_cluster_map.get(c.id) in rigid_ids_in_target:
                continue
            for i, s in enumerate(c.sites):
                if s.su_type != 13 or s.occupied:
                    continue
                outward = get_site_outward_direction(c, i)
                if outward is None:
                    continue
                active_ports.append((s.axial, outward, c, i))
        rewards = []
        for bxy, outward_b, active_cluster, active_idx in active_ports:
            for axy, outward_a, translated_cluster, translated_idx in translated_sites:
                weight = self._secondary_match_weight(
                    outward_b,
                    outward_a,
                    bxy,
                    axy,
                    remaining_lengths,
                    source_cluster=active_cluster,
                    source_idx=active_idx,
                    target_cluster=translated_cluster,
                    target_idx=translated_idx,
                    target_translation=translation,
                )
                if weight > 0:
                    rewards.append(weight)
        rewards.sort(reverse=True)
        return sum(rewards[:4])

    def _score_candidate(self, A: AromaticCluster, B: AromaticCluster,
                          src_root: int, tgt_root: int,
                          direction: str, chain_len: int,
                          planned_len: int,
                          end11: Optional[Tuple[int, int]] = None,
                          translation: Tuple[int, int] = (0, 0),
                          coords_23: Optional[List[Tuple[int, int]]] = None,
                          chain_spec: Optional[ChainSpec] = None,
                          bridge_idx: Optional[int] = None,
                          remaining_lengths_after: Optional[List[int]] = None) -> float:
        """Score a flex candidate (lower is better).
        Includes spatial uniformity terms for balanced growth,
        and lightweight potential energy reward for future loop closures.
        """
        score = 0.0
        
        # 1. Cross-component bonus (strongest signal)
        if src_root != tgt_root:
            score -= 200
        
        # 2. Chain length: prefer planned length
        length_diff = abs(chain_len - planned_len)
        score += length_diff * 10
        score -= min(chain_len, 5) * 3
        if chain_len <= 2:
            score += 18 if chain_len == 1 else 8
        if chain_spec is not None and getattr(chain_spec, 'origin_type', None) == 'extra':
            if chain_len == 1:
                score += 45
            elif chain_len == 2:
                score += 18
            elif chain_len >= 4:
                score -= 10
        if bridge_idx is not None:
            score += max(0, bridge_idx - self._bridges_done) * 2
        
        # 3. Direction alternation bonus
        if self._last_direction is not None and direction != self._last_direction:
            score -= 30
        
        # 4. Cluster kind diversity
        if A.kind != B.kind:
            score -= 20

        preview_qr_score = self._preview_qr_shape_score(
            B, src_root, tgt_root, translation, coords_23=list(coords_23 or []), end11=end11
        )
        score -= 35.0 * float(preview_qr_score)
        
        # --- Lightweight Potential Energy Reward ---
        # If this is a primary action (placing a new cluster B), we want to pull B
        # into a position that makes future secondary (loop-closing) connections easy.
        if src_root != tgt_root and end11 is not None:
            remaining_lengths = list(remaining_lengths_after or [])
            if remaining_lengths:
                score -= self._future_closure_reward(B, tgt_root, translation, remaining_lengths)

        # 5. Spatial uniformity: angular sector balance
        if end11 is not None:
            cx, cy = self._compute_center_of_mass()
            sector_counts = self._get_sector_counts(cx, cy, n_sectors=8)
            
            # Which sector does the target (end11) fall into?
            tx, ty = HexGrid.axial_to_cart(*end11)
            dx, dy = tx - cx, ty - cy
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                sector = min(int(angle / (2 * math.pi / 8)), 7)
                
                avg_count = sum(sector_counts) / max(len(sector_counts), 1)
                sector_count = sector_counts[sector]
                
                # Reward placing in under-populated sectors, penalize crowded ones
                if sector_count < avg_count:
                    score -= 40  # under-populated → good
                elif sector_count > avg_count * 1.5:
                    score += 35  # over-crowded → bad
            
            # 6. Local density penalty
            local_density = self._count_local_density((tx, ty), radius=4.0)
            score += local_density * 3  # penalize dense areas
        
        # 7. Noise
        score += random.uniform(-15, 15)
        
        return score
    
    # ==================== Step Execution ====================
    
    def _adjust_bridge_pool(self, current_idx: int, target_len: int) -> bool:
        return self._adjust_bridge_pool_on(
            self._bridge_pool,
            self._bridge_specs,
            self.allocation_result,
            current_idx,
            target_len,
        )

    def _adjust_bridge_pool_on(
        self,
        bridge_pool: List[int],
        bridge_specs: List[ChainSpec],
        allocation_result,
        current_idx: int,
        target_len: int,
    ) -> bool:
        """
        Dynamically adjust chain lengths to satisfy target_len for the current bridge chain.
        
        Adjustment order (ensures 23 conservation):
        1. FIRST: Try to balance WITHIN remaining flex chains in the pool
           - Borrow from longer chains (keep min length 1)
           - Give surplus to other flex chains
        2. ONLY IF internal adjustment insufficient: borrow from/give to side chains
        
        Returns True if successful, False otherwise.
        """
        planned_len = bridge_pool[current_idx]
        diff = target_len - planned_len  # > 0 means need more 23s, < 0 means surplus
        cap = MAX_23_PER_CHAIN
        
        if target_len < 1 or target_len > cap:
            return False

        if diff == 0:
            return True

        if current_idx < 0 or current_idx >= len(bridge_specs):
            return False
        if getattr(bridge_specs[current_idx], 'origin_type', None) != 'extra':
            return False
        
        # =================================================================
        # Phase 1: Balance WITHIN remaining flex chains first
        # =================================================================
        # Sort remaining chains by length (descending for borrowing, ascending for giving)
        remaining_indices = [
            i for i in range(current_idx + 1, len(bridge_pool))
            if getattr(bridge_specs[i], 'origin_type', None) == 'extra'
        ]
        
        if diff > 0:
            # Need more 23s - borrow from longer flex chains first
            remaining_indices.sort(key=lambda i: bridge_pool[i], reverse=True)
            for i in remaining_indices:
                if diff == 0:
                    break
                min_keep = 2
                if bridge_pool[i] > min_keep:
                    borrow = min(diff, bridge_pool[i] - min_keep)
                    bridge_pool[i] -= borrow
                    # Update spec
                    if i < len(bridge_specs):
                        bridge_specs[i].n_23 = bridge_pool[i]
                        bridge_specs[i].composition = [11] + [23] * bridge_pool[i] + [11]
                    diff -= borrow
        else:
            # Have surplus 23s - give to other flex chains (spread evenly)
            give_total = -diff
            # Give to chains that are >= 1 (prefer shorter ones first to balance)
            remaining_indices.sort(key=lambda i: bridge_pool[i])
            while give_total > 0 and remaining_indices:
                for i in remaining_indices:
                    if give_total == 0:
                        break
                    if bridge_pool[i] >= cap:
                        continue
                    bridge_pool[i] += 1
                    if i < len(bridge_specs):
                        bridge_specs[i].n_23 = bridge_pool[i]
                        bridge_specs[i].composition = [11] + [23] * bridge_pool[i] + [11]
                    give_total -= 1
                if all(bridge_pool[i] >= cap for i in remaining_indices):
                    break
            diff = -give_total  # Remaining surplus
        
        # Update current chain if internal adjustment succeeded
        if diff == 0:
            bridge_pool[current_idx] = target_len
            if current_idx < len(bridge_specs):
                bridge_specs[current_idx].n_23 = target_len
                bridge_specs[current_idx].composition = [11] + [23] * target_len + [11]
            return True
        
        # =================================================================
        # Phase 2: Internal adjustment insufficient - use side chains
        # =================================================================
        alloc_result = allocation_result
        if not alloc_result:
            return False
        
        # Only touch 'extra' type side chains
        extra_side_chains = [c for c in alloc_result.side_chains if 'extra' in c.origin_type]
        if not extra_side_chains:
            return False
        
        if diff > 0:
            # Need more 23s - borrow from side chains with most 23s
            extra_side_chains.sort(key=lambda c: c.n_23, reverse=True)
            for c in extra_side_chains:
                if diff == 0:
                    break
                if c.n_23 > 0:
                    borrow = min(diff, c.n_23)
                    c.n_23 -= borrow
                    c.composition = [11] + [23] * c.n_23 + [22]
                    diff -= borrow
        else:
            # Have surplus 23s - give to side chains (spread evenly to shortest first)
            give = -diff
            while give > 0:
                extra_side_chains.sort(key=lambda c: c.n_23)
                c = extra_side_chains[0]
                c.n_23 += 1
                c.composition = [11] + [23] * c.n_23 + [22]
                give -= 1
            diff = 0
        
        # Final update
        if diff == 0:
            bridge_pool[current_idx] = target_len
            if current_idx < len(bridge_specs):
                bridge_specs[current_idx].n_23 = target_len
                bridge_specs[current_idx].composition = [11] + [23] * target_len + [11]
            return True
        
        return False

    def _action_resource_feasible(self, action: Dict) -> bool:
        """Check whether a candidate can satisfy bridge-length conservation at step time."""
        bridge_idx = int(action.get('bridge_idx', self._bridges_done))
        target_len = int(action.get('chain_len', 0))
        if bridge_idx < self._bridges_done or bridge_idx >= len(self._bridge_pool):
            return False

        pool_copy = list(self._bridge_pool)
        specs_copy = copy.deepcopy(self._bridge_specs)
        alloc_copy = copy.deepcopy(self.allocation_result)
        current_idx = self._bridges_done

        if bridge_idx != current_idx:
            pool_copy[current_idx], pool_copy[bridge_idx] = pool_copy[bridge_idx], pool_copy[current_idx]
            specs_copy[current_idx], specs_copy[bridge_idx] = specs_copy[bridge_idx], specs_copy[current_idx]

        planned_len = pool_copy[current_idx]
        if target_len == planned_len:
            return True

        return self._adjust_bridge_pool_on(
            pool_copy,
            specs_copy,
            alloc_copy,
            current_idx,
            target_len,
        )

    def _clone_for_action_validation(self) -> 'FlexStage':
        sim_state = self.state.copy()
        sim_stage = FlexStage(sim_state, self.original_su_counts, skip_init=True)
        sim_stage._bridge_pool = list(self._bridge_pool)
        sim_stage._bridge_specs = copy.deepcopy(self._bridge_specs)
        sim_stage._n_bridges_total = self._n_bridges_total
        sim_stage._n_connections_needed = self._n_connections_needed
        sim_stage._11_consumed = self._11_consumed
        sim_stage._23_consumed = self._23_consumed
        sim_stage._bridges_done = self._bridges_done
        sim_stage._last_direction = self._last_direction
        sim_stage._connections_made = self._connections_made
        sim_stage._placed_vertices = set(self._placed_vertices)
        sim_stage._placed_centers = set(self._placed_centers)
        sim_stage._flex_used_sites = {
            cid: set(indices) for cid, indices in self._flex_used_sites.items()
        }
        sim_stage._seed_root = self._seed_root
        sim_stage._rigid_cluster_map = dict(self._rigid_cluster_map)
        sim_stage._uid_to_cluster = dict(self._uid_to_cluster)
        sim_stage.allocation_result = copy.deepcopy(self.allocation_result)
        sim_stage._cluster_map = {c.id: c for c in sim_state.graph.clusters}
        return sim_stage

    def _action_step_feasible(self, action: Dict) -> bool:
        sim_stage = self._clone_for_action_validation()
        return sim_stage.step(copy.deepcopy(action))
    
    def step(self, action: Dict) -> bool:
        """Execute a flex connection action."""
        if action.get('type') != 'flex':
            return False
        
        cluster_a_id = action['cluster_a_id']
        cluster_b_id = action['cluster_b_id']
        site_a_idx = action['site_a_idx']
        site_b_idx = action['site_b_idx']
        coords_23 = action['coords_23']
        end11 = action['end11']
        translation = action['translation']
        direction = action.get('direction', 'H')
        chain_len = action['chain_len']
        bridge_idx = int(action.get('bridge_idx', self._bridges_done))
        if chain_len > MAX_23_PER_CHAIN:
            return False
        
        cluster_a = self._get_cluster(cluster_a_id)
        cluster_b = self._get_cluster(cluster_b_id)
        if not cluster_a or not cluster_b:
            return False
        target_was_placed = cluster_b.placed
        
        src_root = self.state._find(cluster_a_id)
        tgt_root = self.state._find(cluster_b_id)
        is_cross = src_root != tgt_root
        
        # Guard: must have remaining bridge chains in the pool
        if not self._has_remaining_chains():
            return False

        if not self._activate_bridge_index(bridge_idx):
            return False
            
        planned_len = self._bridge_pool[self._bridges_done]
        chain_spec = self._bridge_specs[self._bridges_done] if self._bridges_done < len(self._bridge_specs) else None
        
        if not self._validate_flex_proposal(
            cluster_a,
            site_a_idx,
            cluster_b,
            site_b_idx,
            coords_23,
            end11,
            translation,
            direction,
            action.get('outward_dir'),
            chain_spec,
        ):
            return False
                
        # 2. 所有碰撞检查通过后，再进行动态调整（转移23号）
        if chain_len != planned_len:
            # We need to borrow/transfer length within flex pool or from side chains
            success = self._adjust_bridge_pool(self._bridges_done, chain_len)
            if not success:
                return False
            chain_spec = self._bridge_specs[self._bridges_done] if self._bridges_done < len(self._bridge_specs) else chain_spec
                
        # 3. 执行平移
        tq, tr = translation
        if is_cross:
            self._translate_component(tgt_root, tq, tr)
        elif not target_was_placed:
            self._translate_single_cluster(cluster_b, tq, tr)
        
        # 4. 更新节点状态
        # Mark source and target sites as 11
        site_a = cluster_a.sites[site_a_idx]
        site_b = cluster_b.sites[site_b_idx]
        site_a.su_type = 11
        site_a.occupied = True
        site_b.su_type = 11
        site_b.occupied = True
        
        # Track flex-used sites for min_gap enforcement
        self._flex_used_sites.setdefault(cluster_a_id, set()).add(site_a_idx)
        self._flex_used_sites.setdefault(cluster_b_id, set()).add(site_b_idx)
        
        # Get current chain spec for branch info
        branch_info = None
        if chain_spec is not None:
            branch_info = get_branch_info_from_chain_spec(chain_spec)
        
        # Create chain nodes with correct SU types
        chain_nodes = []
        su_types = chain_spec.composition[1:-1] if chain_spec else [23] * len(coords_23)
        for i, p in enumerate(coords_23):
            self._placed_vertices.add(p)
            su = su_types[i] if i < len(su_types) else 23
            node_meta = {
                'stage': 'flex',
                'origin_type': getattr(chain_spec, 'origin_type', None),
                'bridge_idx': int(bridge_idx),
                'position_idx': int(i),
            }
            if branch_info and int(i) == int(branch_info.get('position_idx', -1)) and int(su) in (24, 25):
                node_meta['branch_type'] = branch_info.get('branch_type')
                node_meta['branch_kind'] = 'main'
            node = ChainNode(
                uid=f"Flex-{site_a.uid}-{site_b.uid}-{i}",
                su_type=su,
                axial=p,
                pos2d=HexGrid.axial_to_cart(*p),
                meta=node_meta,
            )
            chain_nodes.append(node)
        
        # Add flex edge
        edge = EdgeFlex(
            u=site_a.uid,
            v=site_b.uid,
            chain=chain_nodes,
        )
        self.state.graph.flex.append(edge)
        self.state.graph.chains.extend(chain_nodes)
        
        # Handle 24/25 branch if present
        if branch_info and direction in ('H',) and len(coords_23) > 2:
            # Get the outward direction for branch coordinate generation
            outward = action.get('outward_dir')
            if outward and outward in (RU, RD, LU, LD):
                # Generate branch coordinates
                branch_coords = horizontal_branch_coords(
                    coords_23, branch_info['position_idx'],
                    outward, branch_info['branch_type'],
                    branch_info['branch_23_count'] + branch_info['branch_22_count']
                )
                
                # Check collision for branch
                branch_collision = (
                    len(set(branch_coords)) != len(branch_coords)
                    or any(bc in self._placed_vertices or bc in self._placed_centers for bc in branch_coords)
                )
                if not branch_collision and branch_coords:
                    # Find the 24/25 node in chain_nodes
                    branch_base_node = chain_nodes[branch_info['position_idx']]
                    
                    # Build branch chain nodes
                    branch_nodes = []
                    branch_su_list = [23] * branch_info['branch_23_count'] + [22] * branch_info['branch_22_count']
                    
                    for bi, bc in enumerate(branch_coords):
                        self._placed_vertices.add(bc)
                        su = branch_su_list[bi] if bi < len(branch_su_list) else 22
                        bn = ChainNode(
                            uid=f"Flex-{site_a.uid}-{site_b.uid}-br-{bi}",
                            su_type=su,
                            axial=bc,
                            pos2d=HexGrid.axial_to_cart(*bc),
                            meta={
                                'stage': 'flex_branch',
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
                    if branch_info.get('extra_22_count', 0) > 0:
                        extra_22_coord = su25_extra_branch_coord(
                            coords_23, branch_info['position_idx'],
                            outward, branch_info['branch_type']
                        )
                        if extra_22_coord and extra_22_coord not in self._placed_vertices and extra_22_coord not in self._placed_centers:
                            self._placed_vertices.add(extra_22_coord)
                            extra_node = ChainNode(
                                uid=f"Flex-{site_a.uid}-{site_b.uid}-br-extra",
                                su_type=22,
                                axial=extra_22_coord,
                                pos2d=HexGrid.axial_to_cart(*extra_22_coord),
                                meta={
                                    'stage': 'flex_branch',
                                    'origin_type': getattr(chain_spec, 'origin_type', None),
                                    'branch_type': branch_info.get('branch_type'),
                                    'branch_kind': 'extra_22',
                                },
                            )
                            extra_edge = EdgeBranch(base=branch_base_node.uid, chain=[extra_node])
                            self.state.graph.branch.append(extra_edge)
                            self.state.graph.chains.append(extra_node)
        
        # Update connectivity
        cluster_a.connected.add(cluster_b_id)
        cluster_b.connected.add(cluster_a_id)
        if is_cross:
            self.state._union(cluster_a_id, cluster_b_id)
            # Update seed root to follow the union
            self._seed_root = self.state._find(cluster_a_id)
        
        # Consume from allocator chain pool
        self._11_consumed += 2
        self._23_consumed += chain_len
        
        # Update direction tracking: consume from the appropriate pool
        self._last_direction = direction
        self._bridges_done += 1
        self._connections_made += 1
        
        self.state.step_count += 1
        self.state.stage_step += 1
        return True
    
    # ==================== Done Check ====================
    
    def is_done(self) -> bool:
        """Check if flex stage is complete.
        Done when all remaining bridge specs have been consumed.
        """
        if not self._has_remaining_chains():
            return True
        
        return False
    
    def get_state(self) -> MCTSState:
        return self.state
    
    def get_result(self) -> Dict:
        """Get flex stage result summary."""
        all_roots = set(self.state._find(c.id) for c in self.state.graph.clusters)
        remaining_flex_23 = int(sum(
            int(spec.n_23) for spec in self._bridge_specs[self._bridges_done:]
        )) if self._bridge_specs else 0
        
        su_remaining = dict(self.state.su_counts)
        su_original = dict(self.original_su_counts)
        
        return {
            'stage': 'flex',
            'flex_edges': len(self.state.graph.flex),
            'connections_made': self._connections_made,
            'connections_needed': self._n_connections_needed,
            'bridges_total': self._n_bridges_total,
            'all_bridges_placed': bool(self._bridges_done >= self._n_bridges_total),
            'all_connected': len(all_roots) <= 1,
            'components': len(all_roots),
            '23_consumed': self._23_consumed,
            '11_consumed': self._11_consumed,
            'bridges_done': self._bridges_done,
            'remaining_flex_23': int(remaining_flex_23),
            'aspect_ratio': self.state.get_aspect_ratio(),
            'su_original': su_original,
            'su_remaining': su_remaining,
        }
    
    def print_result(self):
        r = self.get_result()
        print(f"\n[FlexStage] Result:")
        print(f"  Flex edges: {r['flex_edges']}, Connections: {r['connections_made']}/{r['connections_needed']} "
              f"(bridges available: {r['bridges_total']})")
        print(f"  All connected: {r['all_connected']}, Components: {r['components']}")
        print(f"  23 consumed: {r['23_consumed']}, 11 consumed: {r['11_consumed']}")
        print(f"  Bridges done: {r['bridges_done']}/{r['bridges_total']}")
        print(f"  Aspect ratio: {r['aspect_ratio']:.2f}")
        
        # Resource table
        su_orig = r.get('su_original', {})
        su_rem = r.get('su_remaining', {})
        all_su = sorted(set(su_orig) | set(su_rem))
        
        print(f"\n  {'SU':>4} | {'Original':>8} | {'Remaining':>9}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*9}")
        for su in all_su:
            orig = su_orig.get(su, 0)
            rem = su_rem.get(su, 0)
            if orig == 0 and rem == 0:
                continue
            print(f"  {su:>4} | {orig:>8} | {rem:>9}")
        
        total_orig = sum(su_orig.values())
        total_rem = sum(su_rem.values())
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*9}")
        print(f"  {'Sum':>4} | {total_orig:>8} | {total_rem:>9}")
