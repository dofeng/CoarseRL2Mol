import math
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from .RL_state import (
    MCTSState, Site, AromaticCluster, EdgeRigid, HexGrid,
    RU, RD, LU, LD, UP, DN, HEX_VERTEX_OFFSETS
)

# Direction labels for connection sites
DIRECTION_LABELS = {
    'right_up': RU,
    'right_down': RD,
    'left_up': LU,
    'left_down': LD,
    'up': UP,
    'down': DN,
}
OFFSET_TO_DIRECTION = {v: k for k, v in DIRECTION_LABELS.items()}

# Para-position connection pairs (opposite directions)
PARA_PAIRS = [
    ('right_up', 'left_down'),
    ('right_down', 'left_up'),
    ('up', 'down'),
]


@dataclass
class ConnectionSite:
    """Connection site information"""
    site_idx: int
    site: Site
    direction: str
    cluster_id: int


class RigidConnectionMatcher:
    """Rigid connection site matcher for aromatic clusters"""
    
    @staticmethod
    def get_connection_sites(cluster: AromaticCluster) -> Dict[str, List[ConnectionSite]]:
        """
        Get available SU13 connection sites classified by outward direction.
        Works for both linear and non-linear aromatic clusters.
        """
        result = {d: [] for d in DIRECTION_LABELS}
        
        if not cluster.centers:
            return result

        for idx, site in enumerate(cluster.sites):
            if site.su_type != 13 or site.occupied:
                continue

            outward_dir = None
            vq, vr = site.axial
            for qc, rc in cluster.centers:
                dq, dr = vq - qc, vr - rc
                if (dq, dr) in OFFSET_TO_DIRECTION:
                    outward_dir = OFFSET_TO_DIRECTION[(dq, dr)]
                    break

            if outward_dir is None:
                continue

            result[outward_dir].append(ConnectionSite(
                site_idx=idx,
                site=site,
                direction=outward_dir,
                cluster_id=cluster.id,
            ))
        
        return result
    
    @staticmethod
    def find_para_connection_pairs(
        cluster_a: AromaticCluster,
        cluster_b: AromaticCluster,
    ) -> List[Tuple[ConnectionSite, ConnectionSite, Tuple[int, int]]]:
        """
        Find para-position connection pairs between two clusters.
        """
        sites_a = RigidConnectionMatcher.get_connection_sites(cluster_a)
        sites_b = RigidConnectionMatcher.get_connection_sites(cluster_b)
        
        # Priority groups (higher priority first)
        priority_groups = [
            [('right_up', 'left_down'), ('right_down', 'left_up')],  # Horizontal
            [('down', 'up'), ('up', 'down')],  # Vertical
        ]
        
        pairs = []
        
        for group in priority_groups:
            for dir_a, dir_b in group:
                for sa in sites_a.get(dir_a, []):
                    for sb in sites_b.get(dir_b, []):
                        # Calculate outward vertex from sa
                        target_q = sa.site.axial[0] + DIRECTION_LABELS[dir_a][0]
                        target_r = sa.site.axial[1] + DIRECTION_LABELS[dir_a][1]
                        
                        # Translation to move sb to target position
                        dq = target_q - sb.site.axial[0]
                        dr = target_r - sb.site.axial[1]
                        
                        pairs.append((sa, sb, (dq, dr)))
        
        return pairs


# ==================== Rigid Stage ====================

# Aromatic cluster pairing preferences (from original mcts_types.py)
AROMATIC_PAIR_PREF = {
    "coronene": ("perylene", "benzo_pyrene", "pyrene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "coronene"),
    "perylene": ("benzo_pyrene", "pyrene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "perylene", "coronene"),
    "benzo_pyrene": ("perylene", "pyrene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "benzo_pyrene", "coronene"),
    "pyrene": ("chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "pyrene", "perylene", "benzo_pyrene", "coronene"),
    "chrysene": ("triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "chrysene", "pyrene", "perylene", "benzo_pyrene", "coronene"),
    "triphenylene": ("chrysene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene", "triphenylene", "pyrene", "perylene", "benzo_pyrene", "coronene"),
    "tetracene": ("chrysene", "triphenylene", "phenanthrene", "anthracene", "naphthalene", "benzene", "pyrene", "tetracene", "perylene", "benzo_pyrene", "coronene"),
    "phenanthrene": ("anthracene", "naphthalene", "benzene", "phenanthrene", "tetracene", "chrysene", "triphenylene", "pyrene", "perylene", "benzo_pyrene", "coronene"),
    "anthracene": ("phenanthrene", "naphthalene", "benzene", "anthracene", "tetracene", "chrysene", "triphenylene", "pyrene", "perylene", "benzo_pyrene", "coronene"),
    "naphthalene": ("benzene", "naphthalene", "phenanthrene", "anthracene", "pyrene", "tetracene", "chrysene", "triphenylene", "perylene", "benzo_pyrene", "coronene"),
    "benzene": ("benzene", "naphthalene", "phenanthrene", "anthracene", "pyrene", "tetracene", "chrysene", "triphenylene", "perylene", "benzo_pyrene", "coronene"),
}
AROMATIC_PRIORITY = (
    "coronene",
    "perylene",
    "benzo_pyrene",
    "pyrene",
    "chrysene",
    "triphenylene",
    "tetracene",
    "phenanthrene",
    "anthracene",
    "naphthalene",
    "benzene",
)


class RigidStage:
    """
    Rigid connection stage: 10-10 connections between aromatic clusters.
    """
    
    DEFAULT_MAX_CLUSTER_SIZE = 4
    
    def __init__(self, state: MCTSState,
                 skip_init: bool = False, max_cluster_size: int = None):
        self.state = state
        self.matcher = RigidConnectionMatcher()
        self.max_cluster_size = max_cluster_size or self.DEFAULT_MAX_CLUSTER_SIZE
        self._cluster_map: Dict[int, AromaticCluster] = {
            c.id: c for c in self.state.graph.clusters
        }
        
        if skip_init:
            return
        
        # Place first cluster at origin if none placed
        self._init_placement()
    
    def _init_placement(self):
        """Place first cluster at origin if none placed"""
        clusters = self.state.graph.clusters
        if not clusters:
            return
        if any(c.placed for c in clusters):
            return
        largest = max(clusters, key=lambda c: len(c.sites))
        largest.placed = True
        print(f"[RigidStage] Placed cluster {largest.id} ({largest.kind}) at origin")

    def clone(self) -> 'RigidStage':
        """Deep-copy rigid stage for search expansion."""
        return RigidStage(
            self.state.copy(),
            skip_init=True,
            max_cluster_size=self.max_cluster_size,
        )
    
    # ==================== RC (Rigid Cluster) Management ====================
    
    def get_rc_map(self) -> Dict[int, List[int]]:
        """Get current rigid clusters via Union-Find.
        Returns: root_id -> [member_cluster_ids]"""
        rc_map: Dict[int, List[int]] = {}
        for c in self.state.graph.clusters:
            root = self.state._find(c.id)
            rc_map.setdefault(root, []).append(c.id)
        return rc_map
    
    def _get_cluster(self, cluster_id: int) -> Optional[AromaticCluster]:
        """Get cluster by ID (O(1) dict lookup)."""
        return self._cluster_map.get(cluster_id)
    
    def _rc_ring_count(self, root: int, rc_map: Dict[int, List[int]]) -> int:
        """Get total ring count for an RC."""
        total = 0
        for cid in rc_map.get(root, []):
            c = self._get_cluster(cid)
            if c:
                total += c.rings
        return total
    
    def _rc_has_rigid_edges(self, root: int, rc_map: Dict[int, List[int]]) -> bool:
        """Check if an RC has any rigid edges."""
        member_set = set(rc_map.get(root, []))
        for edge in self.state.graph.rigid:
            if edge.cluster_a in member_set or edge.cluster_b in member_set:
                return True
        return False
    
    def _has_available_sites(self, cluster: AromaticCluster) -> bool:
        """Check if cluster has free SU 13 sites for connection."""
        return any(s.su_type == 13 and not s.occupied for s in cluster.sites)
    
    # ==================== Candidate Generation (RC-pair based) ====================
    
    def get_candidates(self, k: int = 40) -> List[Dict]:
        """
        Generate 10-10 connection candidates between DIFFERENT Rigid Clusters.
        """
        if self.state.su_counts.get(10, 0) < 2:
            return []
        
        rc_map = self.get_rc_map()
        roots = list(rc_map.keys())
        
        if len(roots) < 2:
            return []
        
        # Pre-compute scoring context (once, not per-candidate)
        used_kinds = self._get_used_kinds_in_rigid()
        used_dirs = self._get_used_directions()
        rc_sizes = {r: self._rc_ring_count(r, rc_map) for r in roots}
        rc_standalone = {r: (len(rc_map[r]) == 1 and not self._rc_has_rigid_edges(r, rc_map))
                         for r in roots}
        n_standalone = sum(1 for v in rc_standalone.values() if v)
        
        candidates = []
        
        # For each pair of different RCs
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                root_i, root_j = roots[i], roots[j]
                
                # Get member clusters with available sites
                avail_i = [self._get_cluster(cid) for cid in rc_map[root_i]
                           if self._has_available_sites(self._get_cluster(cid))]
                avail_j = [self._get_cluster(cid) for cid in rc_map[root_j]
                           if self._has_available_sites(self._get_cluster(cid))]
                
                avail_i = [c for c in avail_i if c is not None]
                avail_j = [c for c in avail_j if c is not None]
                
                if not avail_i or not avail_j:
                    continue
                
                # Limit per RC pair to avoid explosion
                random.shuffle(avail_i)
                random.shuffle(avail_j)
                
                for cluster_a in avail_i[:4]:
                    for cluster_b in avail_j[:4]:
                        try:
                            pairs = self.matcher.find_para_connection_pairs(
                                cluster_a, cluster_b)
                        except Exception:
                            continue
                        
                        # Filter by unused directions
                        valid_pairs = [
                            (sa, sb, trans) for sa, sb, trans in pairs
                            if sa.direction not in cluster_a.used_directions
                            and sb.direction not in cluster_b.used_directions
                        ]
                        if not valid_pairs:
                            continue
                        
                        for sa, sb, trans in valid_pairs[:4]:
                            if self._clusters_would_overlap(cluster_a, (0, 0), cluster_b, trans):
                                continue
                            # Collision check for unplaced cluster_b
                            if cluster_a.placed and not cluster_b.placed:
                                if self._would_collide(cluster_b, trans):
                                    continue
                            
                            # Edge crossing check (only if both have positions)
                            if cluster_a.placed:
                                sa_pos = sa.site.pos2d
                                new_sb_axial = (
                                    cluster_b.sites[sb.site_idx].axial[0] + trans[0],
                                    cluster_b.sites[sb.site_idx].axial[1] + trans[1],
                                )
                                sb_pos = HexGrid.axial_to_cart(*new_sb_axial)
                                if self._check_edge_crossing(sa_pos, sb_pos):
                                    continue
                            
                            score = self._score_candidate_rc(
                                root_i, root_j, rc_map, rc_sizes,
                                rc_standalone, n_standalone,
                                cluster_a, cluster_b, sa, sb, trans,
                                used_kinds, used_dirs,
                            )
                            
                            candidates.append({
                                'type': 'rigid',
                                'cluster_a_id': cluster_a.id,
                                'cluster_b_id': cluster_b.id,
                                'site_a_idx': sa.site_idx,
                                'site_b_idx': sb.site_idx,
                                'dir_a': sa.direction,
                                'dir_b': sb.direction,
                                'translation': trans,
                                'score': score,
                            })
        
        candidates.sort(key=lambda x: x['score'])
        return candidates[:k]
    
    def _score_candidate_rc(
        self,
        root_a: int, root_b: int,
        rc_map: Dict[int, List[int]],
        rc_sizes: Dict[int, int],
        rc_standalone: Dict[int, bool],
        n_standalone: int,
        cluster_a: AromaticCluster,
        cluster_b: AromaticCluster,
        sa: ConnectionSite, sb: ConnectionSite,
        trans: Tuple[int, int],
        used_kinds: set,
        used_dirs: set,
    ) -> float:
        """
        Score a connection candidate (LOWER = BETTER).
        """
        score = 0.0
        
        size_a = rc_sizes.get(root_a, 1)
        size_b = rc_sizes.get(root_b, 1)
        merged = size_a + size_b
        a_solo = rc_standalone.get(root_a, True)
        b_solo = rc_standalone.get(root_b, True)
        
        # ------- 1. Coverage bonus -------
        if a_solo and b_solo:
            score -= 120
        elif a_solo or b_solo:
            score -= 60
        else:
            score += 30
        
        if n_standalone <= 2 and not a_solo and not b_solo:
            score += 40
        
        # ------- 2. Size penalty (avoid giant RC) -------
        if merged > self.max_cluster_size:
            excess = merged - self.max_cluster_size
            score += 60 * (excess ** 1.5)
        
        # ------- 3. Kind diversity -------
        if cluster_a.kind != cluster_b.kind:
            score -= 80
        if cluster_a.kind not in used_kinds:
            score -= 50
        if cluster_b.kind not in used_kinds:
            score -= 50
        
        # ------- 4. Direction diversity -------
        if sa.direction not in used_dirs:
            score -= 80
        if sb.direction not in used_dirs:
            score -= 80
        
        # ------- 5. Compactness (minor) -------
        dist = abs(trans[0]) + abs(trans[1])
        score += dist * 0.3
        
        # ------- 6. Exploration noise -------
        score += random.uniform(-50, 50)
        
        return score
    
    def _get_used_kinds_in_rigid(self) -> set:
        """Get kinds of clusters already connected by rigid edges."""
        used_kinds = set()
        for edge in self.state.graph.rigid:
            ca = self._get_cluster(edge.cluster_a)
            cb = self._get_cluster(edge.cluster_b)
            if ca:
                used_kinds.add(ca.kind)
            if cb:
                used_kinds.add(cb.kind)
        return used_kinds
    
    def _get_used_directions(self) -> set:
        """Get directions already used in rigid connections."""
        used_dirs = set()
        for edge in self.state.graph.rigid:
            if edge.dir_a:
                used_dirs.add(edge.dir_a)
            if edge.dir_b:
                used_dirs.add(edge.dir_b)
        return used_dirs

    def _ring_polygon(self, center: Tuple[int, int]) -> List[Tuple[float, float]]:
        cq, cr = center
        return [
            HexGrid.axial_to_cart(cq + dq, cr + dr)
            for dq, dr in HEX_VERTEX_OFFSETS
        ]

    @staticmethod
    def _polygon_axes(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        axes: List[Tuple[float, float]] = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            norm = math.hypot(nx, ny)
            if norm < 1e-9:
                continue
            axes.append((nx / norm, ny / norm))
        return axes

    @staticmethod
    def _project_polygon(poly: List[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
        ax, ay = axis
        vals = [x * ax + y * ay for x, y in poly]
        return min(vals), max(vals)

    def _polygons_overlap(self, poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]], eps: float = 1e-6) -> bool:
        for axis in self._polygon_axes(poly_a) + self._polygon_axes(poly_b):
            min_a, max_a = self._project_polygon(poly_a, axis)
            min_b, max_b = self._project_polygon(poly_b, axis)
            if max_a < min_b - eps or max_b < min_a - eps:
                return False
        return True

    def _cluster_ring_polygons(self, cluster: AromaticCluster,
                               translation: Tuple[int, int] = (0, 0)) -> List[List[Tuple[float, float]]]:
        dq, dr = translation
        return [
            self._ring_polygon((cq + dq, cr + dr))
            for cq, cr in cluster.centers
        ]

    def _clusters_would_overlap(self,
                                cluster_a: AromaticCluster,
                                translation_a: Tuple[int, int],
                                cluster_b: AromaticCluster,
                                translation_b: Tuple[int, int]) -> bool:
        polys_a = self._cluster_ring_polygons(cluster_a, translation_a)
        polys_b = self._cluster_ring_polygons(cluster_b, translation_b)
        for poly_a in polys_a:
            for poly_b in polys_b:
                if self._polygons_overlap(poly_a, poly_b):
                    return True
        return False
    
    # ==================== Collision & Geometry Checks ====================
    
    def _would_collide(self, cluster: AromaticCluster, translation: Tuple[int, int],
                        exclude_cluster_id: Optional[int] = None,
                        exclude_cluster_ids: Optional[Set[int]] = None) -> bool:
        """
        Check if translating cluster would cause collision.
        exclude_cluster_ids: set of cluster IDs to exclude from collision check
                             (e.g., all members of the partner's RC).
        exclude_cluster_id: single ID shortcut (for backward compatibility).
        """
        dq, dr = translation
        
        # Build exclusion set
        skip_ids = set()
        skip_ids.add(cluster.id)
        if exclude_cluster_id is not None:
            skip_ids.add(exclude_cluster_id)
        if exclude_cluster_ids is not None:
            skip_ids.update(exclude_cluster_ids)
        
        # 1. Collect all occupied positions from placed clusters
        occupied_sites = set()
        occupied_centers = set()
        occupied_polygons: List[List[Tuple[float, float]]] = []
        for c in self.state.graph.clusters:
            if c.placed and c.id not in skip_ids:
                for s in c.sites:
                    occupied_sites.add(s.axial)
                for center in c.centers:
                    occupied_centers.add(center)
                occupied_polygons.extend(self._cluster_ring_polygons(c))
        
        # 2. Check translated site positions
        for s in cluster.sites:
            new_pos = (s.axial[0] + dq, s.axial[1] + dr)
            if new_pos in occupied_sites:
                return True
        
        # 3. Check translated ring centers
        for center in cluster.centers:
            new_center = (center[0] + dq, center[1] + dr)
            if new_center in occupied_centers:
                return True

        # 4. Check actual aromatic ring polygon overlap/touching.
        for poly in self._cluster_ring_polygons(cluster, translation):
            for occ in occupied_polygons:
                if self._polygons_overlap(poly, occ):
                    return True
        
        return False
    
    def _check_edge_crossing(self, site_a_pos: Tuple[float, float], 
                              site_b_pos: Tuple[float, float]) -> bool:
        """
        Check if a new rigid edge would cross existing rigid edges.
        
        This prevents 10-10 connections from creating overlapping structures.
        """
        if not self.state.graph.rigid:
            return False
        
        # New edge line segment
        ax, ay = site_a_pos
        bx, by = site_b_pos
        
        for edge in self.state.graph.rigid:
            if not hasattr(edge, 'line') or not edge.line or len(edge.line) < 2:
                continue
            
            # Existing edge line segment
            cx, cy = edge.line[0]
            dx, dy = edge.line[1]
            
            # Check line segment intersection
            if self._segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
        
        return False
    
    def _segments_intersect(self, ax, ay, bx, by, cx, cy, dx, dy) -> bool:
        """
        Check if line segment AB intersects with line segment CD.
        Uses cross product method.
        """
        def ccw(px, py, qx, qy, rx, ry):
            return (ry - py) * (qx - px) > (qy - py) * (rx - px)
        
        # Check if segments share an endpoint (allowed)
        eps = 1e-6
        if (abs(ax - cx) < eps and abs(ay - cy) < eps) or \
           (abs(ax - dx) < eps and abs(ay - dy) < eps) or \
           (abs(bx - cx) < eps and abs(by - cy) < eps) or \
           (abs(bx - dx) < eps and abs(by - dy) < eps):
            return False
        
        # Standard intersection check
        if ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy):
            return True
        
        return False
    
    # ==================== Step Execution ====================
    
    def step(self, action: Dict) -> bool:
        """
        Execute a 10-10 connection between two clusters from different RCs.
        """
        if action.get('type') != 'rigid':
            return False
        
        cluster_a_id = action['cluster_a_id']
        cluster_b_id = action['cluster_b_id']
        site_a_idx = action['site_a_idx']
        site_b_idx = action['site_b_idx']
        
        cluster_a = self._get_cluster(cluster_a_id)
        cluster_b = self._get_cluster(cluster_b_id)
        if not cluster_a or not cluster_b:
            return False
        
        # Must be in different RCs
        if self.state._find(cluster_a_id) == self.state._find(cluster_b_id):
            return False
        
        if self.state.su_counts.get(10, 0) < 2:
            return False
        
        # --- Placement & Connection ---
        a_placed = cluster_a.placed
        b_placed = cluster_b.placed
        
        site_a = cluster_a.sites[site_a_idx]
        site_b = cluster_b.sites[site_b_idx]
        dir_a = action.get('dir_a', '')
        dir_vec = DIRECTION_LABELS.get(dir_a, (0, 0))
        
        if not a_placed and not b_placed:
            target_q = site_a.axial[0] + dir_vec[0]
            target_r = site_a.axial[1] + dir_vec[1]
            rel_trans = (target_q - site_b.axial[0], target_r - site_b.axial[1])
            if self._clusters_would_overlap(cluster_a, (0, 0), cluster_b, rel_trans):
                return False
            cluster_b.translate(*rel_trans)
            
            # Step 2: find a free position for the pair
            if not self._place_pair_at_free_position(cluster_a, cluster_b):
                # Undo
                cluster_b.translate(-rel_trans[0], -rel_trans[1])
                return False
        
        elif a_placed and not b_placed:
            # CASE 2: cluster_a placed, cluster_b not → translate b to connection
            target_q = site_a.axial[0] + dir_vec[0]
            target_r = site_a.axial[1] + dir_vec[1]
            translation = (target_q - site_b.axial[0], target_r - site_b.axial[1])
            
            if self._would_collide(cluster_b, translation):
                return False
            
            sa_pos = site_a.pos2d
            new_sb_axial = (site_b.axial[0] + translation[0],
                            site_b.axial[1] + translation[1])
            sb_pos = HexGrid.axial_to_cart(*new_sb_axial)
            if self._check_edge_crossing(sa_pos, sb_pos):
                return False
            
            cluster_b.translate(*translation)
        
        elif not a_placed and b_placed:
            # CASE 3: cluster_b placed, cluster_a not → swap and translate a
            target_q = site_b.axial[0] + DIRECTION_LABELS.get(
                action.get('dir_b', ''), (0, 0))[0]
            target_r = site_b.axial[1] + DIRECTION_LABELS.get(
                action.get('dir_b', ''), (0, 0))[1]
            translation = (target_q - site_a.axial[0], target_r - site_a.axial[1])
            
            if self._would_collide(cluster_a, translation):
                return False
            
            cluster_a.translate(*translation)
        
        else:
            # CASE 4: Both placed → translate entire RC of cluster_b
            target_q = site_a.axial[0] + dir_vec[0]
            target_r = site_a.axial[1] + dir_vec[1]
            translation = (target_q - site_b.axial[0], target_r - site_b.axial[1])
            if translation != (0, 0):
                if not self._translate_rc_component(
                    cluster_b_id, translation):
                    return False
        
        # --- Mark sites occupied, convert SU 13 → SU 10 ---
        site_a = cluster_a.sites[site_a_idx]
        site_b = cluster_b.sites[site_b_idx]
        
        if site_a.su_type not in (10, 13) or site_b.su_type not in (10, 13):
            return False
        
        site_a.occupied = True
        site_b.occupied = True
        site_a.su_type = 10
        site_b.su_type = 10
        
        # Record used directions
        if action.get('dir_a'):
            cluster_a.used_directions.add(action['dir_a'])
        if action.get('dir_b'):
            cluster_b.used_directions.add(action['dir_b'])
        
        # Consume SU 10
        self.state.su_counts[10] = self.state.su_counts.get(10, 0) - 2
        
        # Add rigid edge
        edge = EdgeRigid(
            u=site_a.uid,
            v=site_b.uid,
            cluster_a=cluster_a_id,
            cluster_b=cluster_b_id,
            dir_a=action.get('dir_a', ''),
            dir_b=action.get('dir_b', ''),
            line=[site_a.pos2d, site_b.pos2d],
        )
        self.state.graph.rigid.append(edge)
        
        # Update connectivity
        cluster_a.connected.add(cluster_b_id)
        cluster_b.connected.add(cluster_a_id)
        self.state._union(cluster_a_id, cluster_b_id)
        
        self.state.step_count += 1
        self.state.stage_step += 1
        return True
    
    def _find_free_offset(
        self,
        clusters: List[AromaticCluster],
        exclude_ids: Optional[Set[int]] = None,
        max_radius: int = 30,
        step: int = 6,
    ) -> Optional[Tuple[int, int]]:
        """
        Spiral-search for a (dq, dr) offset where none of *clusters* collide.
        """
        # IDs to exclude: the clusters being placed + any extra
        group_ids = {c.id for c in clusters} | (exclude_ids or set())
        
        # Pre-compute occupied positions from all OTHER placed clusters
        occupied_sites: Set[Tuple[int, int]] = set()
        occupied_centers: Set[Tuple[int, int]] = set()
        occupied_polygons: List[List[Tuple[float, float]]] = []
        for c in self.state.graph.clusters:
            if c.placed and c.id not in group_ids:
                for s in c.sites:
                    occupied_sites.add(s.axial)
                for center in c.centers:
                    occupied_centers.add(center)
                occupied_polygons.extend(self._cluster_ring_polygons(c))
        
        def collides_at(dq: int, dr: int) -> bool:
            translated_polygons: List[List[Tuple[float, float]]] = []
            for cl in clusters:
                for s in cl.sites:
                    if (s.axial[0] + dq, s.axial[1] + dr) in occupied_sites:
                        return True
                for center in cl.centers:
                    if (center[0] + dq, center[1] + dr) in occupied_centers:
                        return True
                translated_polygons.extend(self._cluster_ring_polygons(cl, (dq, dr)))
            for poly in translated_polygons:
                for occ in occupied_polygons:
                    if self._polygons_overlap(poly, occ):
                        return True
            return False
        
        for offset_mult in range(max_radius):
            offset = offset_mult * step
            positions = [(offset, 0), (-offset, 0), (0, offset), (0, -offset),
                         (offset, offset), (-offset, -offset),
                         (offset, -offset), (-offset, offset)]
            if offset_mult == 0:
                positions = [(0, 0)]
            for tq, tr in positions:
                if not collides_at(tq, tr):
                    return (tq, tr)
        return None
    
    def _place_at_free_position(self, cluster: AromaticCluster) -> bool:
        """Place an unplaced cluster at a non-colliding position."""
        offset = self._find_free_offset([cluster])
        if offset is None:
            return False
        tq, tr = offset
        if tq != 0 or tr != 0:
            cluster.translate(tq, tr)
        cluster.placed = True
        return True
    
    def _place_pair_at_free_position(
        self,
        cluster_a: AromaticCluster,
        cluster_b: AromaticCluster,
    ) -> bool:
        """Place two pre-connected (relative-positioned) clusters as a unit."""
        offset = self._find_free_offset(
            [cluster_a, cluster_b],
            exclude_ids={cluster_a.id, cluster_b.id},
        )
        if offset is None:
            return False
        tq, tr = offset
        if tq != 0 or tr != 0:
            cluster_a.translate(tq, tr)
            cluster_b.translate(tq, tr)
        cluster_a.placed = True
        cluster_b.placed = True
        return True
    
    def _translate_rc_component(self, cluster_id: int,
                                translation: Tuple[int, int],
                                exclude_cluster_id: Optional[int] = None,
                                exclude_cluster_ids: Optional[Set[int]] = None) -> bool:
        """Translate all clusters in the same RC as cluster_id.
        exclude_cluster_ids: all cluster IDs to exclude from collision checks
                             (e.g., all members of the partner's RC).
        """
        root = self.state._find(cluster_id)
        dq, dr = translation
        
        members = [c for c in self.state.graph.clusters
                    if self.state._find(c.id) == root and c.placed]
        
        # Build exclusion set: partner RC + own RC members
        member_ids = {c.id for c in members}
        all_exclude = set(member_ids)
        if exclude_cluster_id is not None:
            all_exclude.add(exclude_cluster_id)
        if exclude_cluster_ids is not None:
            all_exclude.update(exclude_cluster_ids)
        
        # Check collision for ALL members
        for c in members:
            if self._would_collide(c, translation,
                                   exclude_cluster_ids=all_exclude):
                return False
        
        # Translate all members
        for c in members:
            c.translate(dq, dr)
        return True
    
    def is_done(self) -> bool:
        """
        Check if rigid stage is complete.
        Uses lightweight checks first, only falls back to get_candidates()
        when the cheap checks are inconclusive.
        """
        # Check available SU10 (accounting for reserved)
        available_10 = self.state.get_available(10)
        if available_10 < 2:
            return True
        
        # Need at least 2 different RCs to connect
        rc_map = self.get_rc_map()
        if len(rc_map) < 2:
            return True
        
        # Lightweight check: are there any free SU13 sites on at least 2 different RCs?
        rcs_with_free_sites = set()
        for root, members in rc_map.items():
            for cid in members:
                cluster = self._get_cluster(cid)
                if cluster and any(s.su_type == 13 and not s.occupied for s in cluster.sites):
                    rcs_with_free_sites.add(root)
                    break
        if len(rcs_with_free_sites) < 2:
            return True
        
        # Expensive final check only if cheap checks pass
        candidates = self.get_candidates(k=1)
        return len(candidates) == 0
    
    def get_state(self) -> MCTSState:
        """Get current state"""
        return self.state
    
    def get_result(self) -> Dict:
        """Get stage result summary"""
        rigid_clusters = self.get_rigid_cluster_distribution()
        return {
            'stage': 'rigid',
            'rigid_edges': len(self.state.graph.rigid),
            'placed_clusters': sum(1 for c in self.state.graph.clusters if c.placed),
            'total_clusters': len(self.state.graph.clusters),
            'remaining_su10': self.state.su_counts.get(10, 0),
            'rigid_cluster_count': len(rigid_clusters),
            'rigid_clusters': rigid_clusters,
            'aspect_ratio': self.state.get_aspect_ratio(),
        }
    
    def place_all_remaining(self):
        """
        Place ALL unplaced clusters after rigid stage completes.
        """
        unplaced = [c for c in self.state.graph.clusters if not c.placed]
        
        if not unplaced:
            return
        
        for cluster in unplaced:
            if not self._place_at_free_position(cluster):
                # Fallback: place far away
                far_offset = (len(self.state.graph.clusters) + cluster.id) * 8
                cluster.translate(far_offset, 0)
                cluster.placed = True
        
        print(f"[RigidStage] Placed {len(unplaced)} remaining clusters (all clusters now positioned)")

    def consume_all_possible_connections(self, max_iters: int = 4096, candidate_k: int = 256) -> int:
        """Exhaustively consume all remaining feasible 10-10 rigid connections."""
        applied = 0
        for _ in range(max(1, int(max_iters))):
            if self.state.get_available(10) < 2:
                break
            candidates = self.get_candidates(k=max(8, int(candidate_k)))
            if not candidates:
                break
            success = False
            for action in candidates:
                if self.step(action):
                    applied += 1
                    success = True
                    break
            if not success:
                break
        return int(applied)
    
    def get_rigid_cluster_distribution(self) -> List[Dict]:
        """
        Get distribution of rigid clusters after rigid connection stage.
        """
        clusters = self.state.graph.clusters
        rigid_clusters = []
        
        # Reuse Union-Find based RC map
        component_map = self.get_rc_map()
        
        # Create rigid cluster info with NEW sequential IDs
        for root, member_ids in component_map.items():
            members = [c for c in clusters if c.id in member_ids]
            total_rings = sum(c.rings for c in members)
            total_sites = sum(len(c.sites) for c in members)
            kinds = [c.kind for c in members]
            
            # Check if this component has rigid edges
            has_rigid_edges = any(
                e.cluster_a in member_ids and e.cluster_b in member_ids
                for e in self.state.graph.rigid
            )
            
            rc_id = len(rigid_clusters)  # NEW sequential ID
            rigid_clusters.append({
                'id': rc_id,
                'root_id': root,
                'member_ids': member_ids,
                'num_clusters': len(members),
                'total_rings': total_rings,
                'total_sites': total_sites,
                'kinds': kinds,
                'has_rigid_edges': has_rigid_edges,
            })
        
        return rigid_clusters
    
    def print_rigid_cluster_distribution(self):
        """Print rigid cluster distribution for debugging."""
        dist = self.get_rigid_cluster_distribution()
        n_with_edges = sum(1 for d in dist if d.get('has_rigid_edges'))
        n_standalone = sum(1 for d in dist if not d.get('has_rigid_edges'))
        
        print(f"\n[RigidStage] Rigid Cluster Distribution ({len(dist)} total):")
        print(f"  With rigid edges: {n_with_edges}, Standalone: {n_standalone}")
        
        for rc in dist[:10]:
            label = "10-10" if rc.get('has_rigid_edges') else "standalone"
            kinds_str = '+'.join(rc['kinds'])
            print(f"    RC#{rc['id']}: {rc['total_rings']} rings, "
                  f"{rc['num_clusters']} clusters ({label}) [{kinds_str}]")
        if len(dist) > 10:
            print(f"    ... and {len(dist) - 10} more")
    
    def create_rigid_cluster_copies(self) -> List[Dict]:
        """
        Create independent deep copies of all rigid clusters for visualization/output.
        """
        import copy
        
        rigid_edges = self.state.graph.rigid
        rc_map = self.get_rc_map()  # root_id -> [member_ids]
        rigid_cluster_copies = []
        
        def _deep_copy_members(members):
            """Deep-copy a list of clusters with coordinate normalization."""
            ref_q, ref_r = 0, 0
            if members and members[0].centers:
                ref_q, ref_r = members[0].centers[0]
            copied = []
            for cl in members:
                c = copy.deepcopy(cl)
                for site in c.sites:
                    site.axial = (site.axial[0] - ref_q, site.axial[1] - ref_r)
                    site.pos2d = HexGrid.axial_to_cart(*site.axial)
                c.centers = [(cq - ref_q, cr - ref_r) for cq, cr in c.centers]
                copied.append(c)
            return copied
        
        for _root, member_ids in rc_map.items():
            member_set = set(member_ids)
            members = [self._get_cluster(cid) for cid in member_ids]
            members = [m for m in members if m is not None]
            
            copied_clusters = _deep_copy_members(members)
            
            # Internal rigid edges for this component
            internal_edges = [
                e for e in rigid_edges
                if e.cluster_a in member_set and e.cluster_b in member_set
            ]
            
            # Composition string
            kind_counts: Dict[str, int] = {}
            for m in members:
                kind_counts[m.kind] = kind_counts.get(m.kind, 0) + 1
            composition_parts = []
            for kind, count in sorted(kind_counts.items()):
                composition_parts.append(f"{count}×{kind}" if count > 1 else kind)
            composition = " + ".join(composition_parts)
            
            rigid_cluster_copies.append({
                'id': len(rigid_cluster_copies),
                'aromatic_clusters': copied_clusters,
                'rigid_edges': internal_edges,
                'size': sum(m.rings for m in members),
                'composition': composition,
                'type': 'connected' if len(members) > 1 else 'individual',
                'member_ids': member_ids,
            })
        
        return rigid_cluster_copies
