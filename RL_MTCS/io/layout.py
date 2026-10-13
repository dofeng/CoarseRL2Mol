"""Topology embeddings from explicit CSV connectivity."""
from __future__ import annotations

from collections import defaultdict
import itertools
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Set, Tuple
from RL_MTCS.geometry import (Axial, HEX_DIRECTIONS, HEX_VERTEX_OFFSETS, hex_edge_valid,
    axial_to_cartesian, valid_hex_directions, topology_site_valid, topology_edge_valid)
from RL_MTCS.mdp.types import InverseInputError
if TYPE_CHECKING:
    from RL_MTCS.io.input import ParsedTopology

RU, RD, LU, LD, UP, DN = HEX_DIRECTIONS


def _su25_center_branches(parsed, su_by_id, source_neighbors=None):
    """Bind one fixed singleton SU22 branch before any geometry search."""
    adjacency = {gid: set() for gid in parsed.nodes}
    for a, b in parsed.edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    pairs = []
    for parent in parsed.nodes:
        if su_by_id[parent] != 25:
            continue
        if len(adjacency[parent]) != 4:
            raise InverseInputError(f'SU25 GID {parent} must retain four topology connections')
        candidates = [path[0] for p, path in parsed.branches
                      if p == parent and len(path) == 1 and su_by_id[path[0]] == 22
                      and adjacency[path[0]] == {parent}]
        if not candidates:
            raise InverseInputError(f'SU25 GID {parent} has no fixed terminal SU22 branch')
        # Original neighbors identify the fixed branch only. Parsed edges remain
        # authoritative; a newly supplied branch falls back to notation order.
        fixed = next((gid for gid in (source_neighbors or {}).get(parent, ())
                      if gid in candidates), candidates[0])
        pairs.append((parent, fixed))
    return tuple(pairs)


def _three_aromatic_hub_layout(
    parsed: ParsedTopology,
    su_by_id: Mapping[int, int],
) -> Optional[Dict[int, Axial]]:
    """Return the fixed honeycomb pose for the 9/6/6 three-aromatic hub.

    ``9-0-single20d3(-15-15-22)-double20d3(-27-6)-27-6``
    contains two adjacent degree-three centers.  Treating its branches as
    ordinary alternating chains folds its aromatic termini into a pose where
    three complete aromatic components cannot be attached without collision.
    This domain topology therefore has its own immutable local pose, expressed
    by structural roles rather than YM-8-specific GIDs.
    """
    main = tuple(parsed.main_path)
    if tuple(int(su_by_id.get(node_id, -1)) for node_id in main) != (
        9, 0, 20, 20, 27, 6,
    ):
        return None
    branches = {
        int(parent): tuple(path)
        for parent, path in parsed.branches
        if path
    }
    left_branch = branches.get(int(main[2]), ())
    right_branch = branches.get(int(main[3]), ())
    if (
        tuple(int(su_by_id.get(node_id, -1)) for node_id in left_branch)
        != (15, 15, 22)
        or tuple(int(su_by_id.get(node_id, -1)) for node_id in right_branch)
        != (27, 6)
        or set(branches) != {int(main[2]), int(main[3])}
    ):
        return None

    # Keep every ordinary path on the honeycomb sublattice.  In particular,
    # both SU27 exits must turn before reaching their SU6 aromatic sites;
    # continuing the SU20-SU27 direction would create an impossible straight
    # three-node path on the hexagonal carbon skeleton.  The global offset
    # keeps the full GID template on the valid modulo-3 lattice.  Re-basing at
    # GID 0 below recovers the immutable local pose used by placement search.
    coordinates = {
        int(main[0]): (2, 0),
        int(main[1]): (1, 0),
        int(main[2]): (0, -1),
        int(main[3]): (-1, -1),
        int(main[4]): (-2, -2),
        int(main[5]): (-3, -2),
        int(left_branch[0]): (0, -2),
        int(left_branch[1]): (-1, -3),
        int(left_branch[2]): (-1, -4),
        int(right_branch[0]): (-1, 0),
        int(right_branch[1]): (0, 1),
    }
    edge_keys = {frozenset((int(left), int(right))) for left, right in parsed.edges}
    for left, right in itertools.combinations(sorted(coordinates), 2):
        delta = (
            coordinates[right][0] - coordinates[left][0],
            coordinates[right][1] - coordinates[left][1],
        )
        if delta in HEX_DIRECTIONS or (-delta[0], -delta[1]) in HEX_DIRECTIONS:
            pair = frozenset((left, right))
            if pair not in edge_keys:
                raise InverseInputError(
                    "three-aromatic hub contains a non-bonded neighboring pair: "
                    f"{left}-{right}"
                )
    return coordinates


def _validate_topology_layout(
    parsed: ParsedTopology,
    su_by_id: Mapping[int, int],
    coords: Mapping[int, Axial],
    center_branches=(),
) -> None:
    center_parents = {child: parent for parent, child in center_branches}
    if any(not topology_site_valid(gid, point, center_parents) for gid, point in coords.items()):
        raise InverseInputError('topology node violates its honeycomb geometry role')
    for left, right in parsed.edges:
        if not hex_edge_valid(coords[left], coords[right]):
            raise InverseInputError(f"topology edge {left}-{right} is not a hex neighbor")
        if not topology_edge_valid(left, right, coords[left], coords[right], center_parents):
            raise InverseInputError(f"topology edge {left}-{right} violates honeycomb parity")


def _closure_cycle_nodes(
    parsed: ParsedTopology,
) -> Tuple[Tuple[int, ...], ...]:
    """Return each closure cycle using the rooted non-closure tree path."""
    closure_keys = {
        frozenset((int(left), int(right)))
        for left, right in parsed.closure_edges
    }
    tree_adjacency: Dict[int, Set[int]] = {
        int(node_id): set() for node_id in parsed.nodes
    }
    for left, right in parsed.edges:
        if frozenset((int(left), int(right))) in closure_keys:
            continue
        tree_adjacency[int(left)].add(int(right))
        tree_adjacency[int(right)].add(int(left))

    cycles: List[Tuple[int, ...]] = []
    for source, target in parsed.closure_edges:
        queue: List[Tuple[int, Tuple[int, ...]]] = [
            (int(target), (int(target),))
        ]
        seen = {int(target)}
        selected: Optional[Tuple[int, ...]] = None
        while queue:
            current, path = queue.pop(0)
            if current == int(source):
                selected = path
                break
            for neighbor in sorted(tree_adjacency.get(current, set())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
        if selected is None:
            raise InverseInputError(
                f"ring closure {source}-{target} has no rooted path"
            )
        cycles.append(tuple(selected))
    return tuple(cycles)


def _layout_topology_with_closures(
    parsed: ParsedTopology,
    su_by_id: Mapping[int, int],
    center_branches=(),
) -> Dict[int, Axial]:
    """Embed exported six-membered aliphatic closures on the hex grid."""
    cycles = _closure_cycle_nodes(parsed)
    six_cycles = [cycle for cycle in cycles if len(cycle) == 6]
    if not six_cycles:
        raise InverseInputError(
            "closed topology needs a six-membered seed cycle; "
            f"cycles={[len(cycle) for cycle in cycles]}"
        )

    cycle = six_cycles[0]
    coords: Dict[int, Axial] = {
        int(node_id): tuple(HEX_VERTEX_OFFSETS[index])
        for index, node_id in enumerate(cycle)
    }
    occupied: Set[Axial] = set(coords.values())
    adjacency: Dict[int, Set[int]] = {
        int(node_id): set() for node_id in parsed.nodes
    }
    edge_keys: Set[frozenset[int]] = set()
    for left, right in parsed.edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
        edge_keys.add(frozenset((int(left), int(right))))

    remaining = set(int(node_id) for node_id in parsed.nodes) - set(coords)
    center_parents = {child: parent for parent, child in center_branches}

    def edge_allowed(
        left: int,
        right: int,
        left_coord: Axial,
        right_coord: Axial,
    ) -> bool:
        return topology_edge_valid(left, right, left_coord, right_coord, center_parents)

    def place_remaining() -> bool:
        if not remaining:
            return True
        frontier = [
            node_id for node_id in remaining
            if adjacency[node_id] & set(coords)
        ]
        if not frontier:
            return False
        node_id = min(
            frontier,
            key=lambda value: (
                -len(adjacency[value] & set(coords)),
                -len(adjacency[value]),
                value,
            ),
        )
        assigned_neighbors = sorted(adjacency[node_id] & set(coords))
        possible: Optional[Set[Axial]] = None
        for neighbor in assigned_neighbors:
            around = {
                (
                    coords[neighbor][0] + direction[0],
                    coords[neighbor][1] + direction[1],
                )
                for direction in HEX_DIRECTIONS
            }
            possible = around if possible is None else possible & around
        for candidate in sorted(possible or set()):
            if candidate in occupied:
                continue
            if any(
                not edge_allowed(
                    node_id, neighbor, candidate, coords[neighbor]
                )
                for neighbor in assigned_neighbors
            ):
                continue
            # A neighboring coordinate represents a bond in this model.  Do
            # not create an implicit extra closure while attaching prefixes or
            # fixed branches to the canonical ring.
            if any(
                frozenset((node_id, other_id)) not in edge_keys
                and hex_edge_valid(candidate, coords[other_id], honeycomb=True)
                for other_id in coords
            ):
                continue
            coords[node_id] = candidate
            occupied.add(candidate)
            remaining.remove(node_id)
            if place_remaining():
                return True
            remaining.add(node_id)
            occupied.remove(candidate)
            coords.pop(node_id, None)
        return False

    if not place_remaining():
        raise InverseInputError(
            "cannot embed closed topology on the hex grid: "
            f"{parsed.nodes}"
        )
    _validate_topology_layout(parsed, su_by_id, coords, center_branches)
    return coords


def _layout_topology_tree(parsed: ParsedTopology, su_by_id: Mapping[int, int],
                          center_branches=None) -> Dict[int, Axial]:
    if center_branches is None:
        center_branches = _su25_center_branches(parsed, su_by_id)
    center_parents = {child: parent for parent, child in center_branches}
    if parsed.closure_edges:
        return _layout_topology_with_closures(parsed, su_by_id, center_branches)
    special = _three_aromatic_hub_layout(parsed, su_by_id)
    if special is not None:
        _validate_topology_layout(parsed, su_by_id, special)
        return special

    coords: Dict[int, Axial] = {}
    occupied: Set[Axial] = set()
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for left, right in parsed.edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))

    # Canonical horizontal chains follow the legacy right-up/right-down
    # sequence: 11(a,b)-23(a+1,b+1)-23(a+2,b+1)-... .  Starting on a
    # phase-B vertex keeps every ordinary edge on the honeycomb sublattice.
    # Connection search may rotate or reflect this whole immutable geometry,
    # but it must never change an individual step.
    for index, node_id in enumerate(parsed.main_path):
        if index == 0:
            coord = (1, 1)
        else:
            previous = coords[parsed.main_path[index - 1]]
            step = RU if (index - 1) % 2 == 0 else RD
            coord = (previous[0] + step[0], previous[1] + step[1])
        coords[node_id] = coord
        occupied.add(coord)

    grouped: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
    for order, (parent, path) in enumerate(parsed.branches):
        if path:
            grouped[int(parent)].append((order, tuple(path)))

    branch_rows: List[Tuple[int, int, Tuple[int, ...]]] = []
    for parent, rows in grouped.items():
        branch_rows.extend(
            (int(parent), int(order), tuple(path))
            for order, path in rows
        )
    pending_branch_rows = list(branch_rows)
    branch_rows = []
    reachable_ids = {int(node_id) for node_id in parsed.main_path}
    while pending_branch_rows:
        ready_index = next(
            (
                index
                for index, (parent, _order, _path) in enumerate(
                    pending_branch_rows
                )
                if int(parent) in reachable_ids
            ),
            None,
        )
        if ready_index is None:
            raise InverseInputError(
                "topology branches are disconnected from the main path"
            )
        row = pending_branch_rows.pop(int(ready_index))
        branch_rows.append(row)
        reachable_ids.update(int(node_id) for node_id in row[2])

    def allowed_edge_directions(left: int, right: int) -> List[Axial]:
        left_coord = coords[int(left)]
        return [d for d in HEX_DIRECTIONS if topology_edge_valid(
            left, right, left_coord, (left_coord[0] + d[0], left_coord[1] + d[1]), center_parents)]

    branch_forward_by_parent: Dict[int, Optional[Tuple[float, float]]] = {}
    for parent, _order, _path in branch_rows:
        branch_forward: Optional[Tuple[float, float]] = None
        if parent in parsed.main_path:
            parent_index = parsed.main_path.index(parent)
            if 0 < parent_index < len(parsed.main_path) - 1:
                before = coords[parsed.main_path[parent_index - 1]]
                after = coords[parsed.main_path[parent_index + 1]]
                branch_forward = axial_to_cartesian((
                    after[0] - before[0], after[1] - before[1]
                ))
            elif parent_index + 1 < len(parsed.main_path):
                after = coords[parsed.main_path[parent_index + 1]]
                branch_forward = axial_to_cartesian((
                    after[0] - coords[parent][0],
                    after[1] - coords[parent][1],
                ))
            elif parent_index > 0:
                before = coords[parsed.main_path[parent_index - 1]]
                branch_forward = axial_to_cartesian((
                    coords[parent][0] - before[0],
                    coords[parent][1] - before[1],
                ))
        branch_forward_by_parent[parent] = branch_forward

    last_blocked_gid: List[int] = []

    def place_branch_path(row_index: int, path_index: int) -> bool:
        if row_index >= len(branch_rows):
            return True
        parent, _order, path = branch_rows[row_index]
        if path_index >= len(path):
            return place_branch_path(row_index + 1, 0)

        node_id = int(path[path_index])
        previous_id = int(parent if path_index == 0 else path[path_index - 1])
        previous_previous_id = (
            None
            if path_index == 0
            else int(parent if path_index == 1 else path[path_index - 2])
        )
        if previous_id not in coords:
            return False
        parent_coord = coords[previous_id]
        used = {
            (
                coords[neighbor][0] - parent_coord[0],
                coords[neighbor][1] - parent_coord[1],
            )
            for neighbor in adjacency[previous_id]
            if neighbor in coords
        }
        preferred = [
            direction
            for direction in allowed_edge_directions(previous_id, node_id)
            if direction not in used
        ]
        if previous_previous_id is not None:
            movement = (
                parent_coord[0] - coords[previous_previous_id][0],
                parent_coord[1] - coords[previous_previous_id][1],
            )
            back = (-movement[0], -movement[1])
            forward = [
                direction for direction in preferred
                if direction != back
            ]
            mx, my = axial_to_cartesian(movement)
            branch_forward = branch_forward_by_parent.get(parent)
            preferred = sorted(
                forward,
                key=lambda direction: (
                    (
                        branch_forward[0] * axial_to_cartesian(direction)[0]
                        + branch_forward[1] * axial_to_cartesian(direction)[1]
                        if branch_forward is not None else 0.0
                    ),
                    mx * axial_to_cartesian(direction)[0]
                    + my * axial_to_cartesian(direction)[1],
                    direction,
                ),
                reverse=True,
            )

        for direction in preferred:
            candidate = (
                parent_coord[0] + direction[0],
                parent_coord[1] + direction[1],
            )
            if candidate in occupied:
                continue
            if any(other not in adjacency[node_id]
                   and hex_edge_valid(candidate, point, honeycomb=True)
                   for other, point in coords.items()):
                continue
            coords[node_id] = candidate
            occupied.add(candidate)
            if place_branch_path(row_index, path_index + 1):
                return True
            occupied.remove(candidate)
            coords.pop(node_id, None)
        last_blocked_gid[:] = [previous_id]
        return False

    if not place_branch_path(0, 0):
        blocked = last_blocked_gid[0] if last_blocked_gid else parsed.main_path[0]
        raise InverseInputError(
            f"cannot embed topology branch at GID {blocked} on the honeycomb grid"
        )

    if set(coords) != set(parsed.nodes):
        missing = sorted(set(parsed.nodes) - set(coords))
        raise InverseInputError(f"topology layout missed GIDs: {missing}")

    # Tree endpoints remain independent. Only explicit ring products use
    # the aromatic-edge closure embedding; proximity never implies a ring.

    _validate_topology_layout(parsed, su_by_id, coords, center_branches)
    return coords
