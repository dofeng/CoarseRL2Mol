"""Build immutable hex-grid assembly contracts from inverse CSV exports.

The node CSV supplies the final node identities/SU distribution and the
aromatic bridgehead evidence.  The topology CSV supplies the authoritative
flex/side trees.  In particular, topology edges are parsed from ``gid`` and
are never reconstructed from the older ``hop1_ids`` cache.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, is_dataclass
import ast
import csv
import json
import math
import itertools
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .geometry import (
    Axial,
    HEX_DIRECTIONS,
    HEX_VERTEX_OFFSETS,
    axial_to_cartesian,
    valid_hex_directions,
)
from .models import (
    ExternalContract,
    InverseAssemblyInput,
    InverseInputError,
    InputSourceKind,
    TopologyUnit,
    UnitEdge,
    UnitNode,
    UnitPort,
)
from .state import HexAssemblyState


AROMATIC_SU_TYPES = frozenset(range(5, 14)) | {26, 30}
RU, RD, LU, LD, UP, DN = (1, 1), (1, 0), (-1, 0), (-1, -1), (0, 1), (0, -1)

AROMATIC_CLUSTER_CENTERS: Dict[str, Tuple[Axial, ...]] = {
    "benzene": ((0, 0),),
    "naphthalene": ((0, 0), (2, 1)),
    "phenanthrene": ((-2, -1), (0, 0), (1, -1)),
    "anthracene": ((0, 0), (2, 1), (4, 2)),
    "tetracene": ((0, 0), (2, 1), (4, 2), (6, 3)),
    "pyrene": ((0, 0), (2, 1), (1, -1), (3, 0)),
    "benzo_pyrene": ((-2, -1), (0, 0), (2, 1), (1, -1), (3, 0)),
    "perylene": ((0, 0), (-1, 1), (1, 2), (1, -1), (-1, -2)),
    "chrysene": ((-2, -1), (0, 0), (1, -1), (3, 0)),
    "triphenylene": ((0, 0), (-2, -1), (1, 2), (1, -1)),
    "coronene": ((0, 0), (-1, 1), (1, 2), (2, 1), (1, -1), (-1, -2), (-2, -1)),
}

CLUSTER_COST = {
    "coronene": (12, 12),
    "pyrene": (6, 10),
    "benzo_pyrene": (8, 12),
    "perylene": (8, 12),
    "chrysene": (6, 12),
    "triphenylene": (6, 12),
    "tetracene": (6, 12),
    "phenanthrene": (4, 10),
    "anthracene": (4, 10),
    "naphthalene": (2, 8),
    "benzene": (0, 6),
}


@dataclass(frozen=True)
class NodeRow:
    node_id: int
    su_type: int
    su_name: str
    hop1_ids: Tuple[int, ...]
    template_key: str


@dataclass(frozen=True)
class ParsedTopology:
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    main_path: Tuple[int, ...]
    branches: Tuple[Tuple[int, Tuple[int, ...]], ...]


@dataclass
class _ClusterTemplate:
    kind: str
    centers: List[Axial]
    coords: List[Axial]
    shared: List[bool]
    edges: List[Tuple[int, int]]
    rings: List[Tuple[Axial, ...]]


class _DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _literal_ints(value: object) -> Tuple[int, ...]:
    try:
        parsed = ast.literal_eval(str(value or "[]"))
    except (SyntaxError, ValueError):
        parsed = []
    if isinstance(parsed, (list, tuple, set)):
        return tuple(int(item) for item in parsed)
    return ()


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise InverseInputError(f"cannot read inverse CSV: {path}") from exc


def load_node_rows(path: str | Path) -> Dict[int, NodeRow]:
    target = Path(path)
    rows = _csv_rows(target)
    required = {"global_id", "su_type", "hop1_ids", "template_key"}
    if not rows or not required.issubset(rows[0]):
        raise InverseInputError(
            f"node CSV must contain {sorted(required)}: {target}"
        )
    result: Dict[int, NodeRow] = {}
    for raw in rows:
        node_id = int(raw["global_id"])
        if node_id in result:
            raise InverseInputError(f"duplicate global_id in node CSV: {node_id}")
        result[node_id] = NodeRow(
            node_id=node_id,
            su_type=int(raw["su_type"]),
            su_name=str(raw.get("su_name", "") or ""),
            hop1_ids=_literal_ints(raw.get("hop1_ids", "[]")),
            template_key=str(raw.get("template_key", "") or ""),
        )
    missing_neighbors = sorted({
        neighbor
        for row in result.values()
        for neighbor in row.hop1_ids
        if neighbor not in result
    })
    if missing_neighbors:
        raise InverseInputError(
            f"node CSV hop1_ids reference unknown GIDs: {missing_neighbors[:12]}"
        )
    return result


def parse_topology_notation(value: str) -> ParsedTopology:
    """Parse ``1-2-3(-4)(-5-6)-7`` into a rooted tree.

    Branch parentheses attach to the node immediately before ``(``.  GID
    notation contains only integer atoms, so labels in ``actual_su`` cannot
    accidentally create fake node IDs.
    """
    tokens = re.findall(r"\d+|[()]|-", str(value or ""))
    nodes: List[int] = []
    edges: List[Tuple[int, int]] = []
    branches: List[Tuple[int, Tuple[int, ...]]] = []

    def parse_sequence(index: int, parent: Optional[int] = None) -> Tuple[int, List[int]]:
        previous = parent
        path: List[int] = []
        while index < len(tokens):
            token = tokens[index]
            if token == ")":
                return index + 1, path
            if token == "-":
                index += 1
                continue
            if token == "(":
                if previous is None:
                    raise InverseInputError(f"branch without parent in topology notation: {value}")
                branch_parent = previous
                index, branch_path = parse_sequence(index + 1, branch_parent)
                branches.append((branch_parent, tuple(branch_path)))
                continue
            node_id = int(token)
            if node_id in nodes:
                raise InverseInputError(
                    f"topology notation repeats GID {node_id}; ring-closure syntax is unsupported: {value}"
                )
            nodes.append(node_id)
            path.append(node_id)
            if previous is not None:
                edges.append((int(previous), node_id))
            previous = node_id
            index += 1
        return index, path

    _index, main_path = parse_sequence(0)
    if not nodes:
        raise InverseInputError(f"empty topology GID notation: {value!r}")
    edge_keys = {tuple(sorted(edge)) for edge in edges}
    if len(edge_keys) != len(edges) or len(edges) != len(nodes) - 1:
        raise InverseInputError(f"topology notation is not a tree: {value}")
    return ParsedTopology(
        nodes=tuple(nodes),
        edges=tuple((int(left), int(right)) for left, right in edges),
        main_path=tuple(main_path),
        branches=tuple(branches),
    )


def _parse_fragment_ids(value: object) -> Tuple[int, ...]:
    return tuple(sorted(set(_literal_ints(value))))


def load_topology_rows(path: str | Path) -> List[Dict[str, object]]:
    target = Path(path)
    rows = _csv_rows(target)
    required = {"actual_su", "gid", "product_type", "required_slots_satisfied"}
    if not rows or not required.issubset(rows[0]):
        raise InverseInputError(
            f"topology CSV must contain {sorted(required)}: {target}"
        )
    result: List[Dict[str, object]] = []
    for index, raw in enumerate(rows):
        satisfied = str(raw.get("required_slots_satisfied", "")).strip().lower()
        if satisfied not in {"true", "1", "yes"}:
            raise InverseInputError(f"topology row {index} is not complete")
        result.append({
            **raw,
            "row_index": index,
            "parsed": parse_topology_notation(str(raw.get("gid", ""))),
            "source_fragment_ids_parsed": _parse_fragment_ids(raw.get("source_fragment_ids", "[]")),
        })
    return result


def _bridgehead_counts(nodes: Mapping[int, NodeRow]) -> Tuple[int, int, int, int, int, int, int]:
    x = y = z = m = n = p = q = 0
    pattern = re.compile(r"\((\d+),\s*\(([^)]*)\),\s*\(([^)]*)\)\)")
    for row in nodes.values():
        if row.su_type != 12:
            continue
        match = pattern.match(row.template_key)
        if not match:
            q += 1
            continue
        hop1 = [int(value) for value in re.findall(r"\d+", match.group(2))]
        hop2 = [int(value) for value in re.findall(r"\d+", match.group(3))]
        count1, count2 = hop1.count(12), hop2.count(12)
        if len(hop1) == 3 and count1 == 3:
            x += 1
        elif count1 == 2 and count2 >= 3:
            y += 1
        elif count1 == 2 and count2 == 2:
            z += 1
        elif count2 >= 2:
            m += 1
        elif count1 == 2 and count2 == 1:
            n += 1
        elif count2 == 1:
            p += 1
        else:
            q += 1
    return x, y, z, m, n, p, q


class _AromaticPlanner:
    """Legacy aromatic-cluster accounting without SU allocation downstream."""

    DOWNGRADE = {
        "coronene": ["pyrene", "benzo_pyrene", "perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "pyrene": ["benzo_pyrene", "perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "benzo_pyrene": ["perylene", "chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "perylene": ["chrysene", "triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "chrysene": ["triphenylene", "tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "triphenylene": ["tetracene", "phenanthrene", "anthracene", "naphthalene", "benzene"],
        "tetracene": ["phenanthrene", "anthracene", "naphthalene", "benzene"],
        "phenanthrene": ["anthracene", "naphthalene", "benzene"],
        "anthracene": ["naphthalene", "benzene"],
        "naphthalene": ["benzene"],
        "benzene": [],
    }

    def __init__(self, counts: Mapping[int, int], bridgeheads: Sequence[int]):
        self.x, self.y, self.z, self.m, self.n, self.p, self.q = [int(v) for v in bridgeheads]
        self.remaining12 = int(counts.get(12, 0))
        self.remaining12 += self.remaining12 % 2
        self.remaining13 = float(sum(counts.get(su, 0) for su in (5, 6, 7, 8, 9, 10, 11, 13)))
        self.remaining13 += 1.5 * float(counts.get(26, 0)) + 2.0 * float(counts.get(30, 0))
        self.max12to13 = 4
        self.max13to12 = max(0, 2 - (self.remaining12 - int(counts.get(12, 0))))
        self.used12to13 = 0
        self.used13to12 = 0
        self.kinds: List[str] = []

    def _conversion(self, need12: int, need13: int) -> Optional[Tuple[int, int]]:
        choices = []
        for to13 in range(min(4, self.max12to13 - self.used12to13, self.remaining12) + 1):
            for to12 in range(min(4, self.max13to12 - self.used13to12, int(self.remaining13)) + 1):
                if to13 + to12 > 4:
                    continue
                if self.remaining12 - to13 + to12 >= need12 and self.remaining13 + to13 - to12 >= need13:
                    choices.append(((to13 + to12, abs(to13 - to12), to13, to12), (to13, to12)))
        return min(choices)[1] if choices else None

    def _make(self, kind: str) -> bool:
        need12, need13 = CLUSTER_COST[kind]
        conversion = self._conversion(need12, need13)
        if conversion is None:
            return False
        to13, to12 = conversion
        self.remaining12 += -to13 + to12
        self.remaining13 += to13 - to12
        self.used12to13 += to13
        self.used13to12 += to12
        self.remaining12 -= need12
        self.remaining13 -= need13
        self.kinds.append(kind)
        return True

    def _downgrade(self, kind: str) -> bool:
        return any(self._make(candidate) for candidate in self.DOWNGRADE[kind])

    def plan(self) -> List[str]:
        poly5 = (self.y + 1) // 2
        poly4 = (self.z + 1) // 2
        targets = [
            ("coronene", self.x // 6),
            ("pyrene", (max(0, self.x - (self.x // 6) * 6) + 1) // 2),
            ("benzo_pyrene", (poly5 + 1) // 2),
            ("perylene", poly5 // 2),
            ("tetracene", self.m // 2),
            ("chrysene", (2 * poly4 + 2) // 3),
            ("triphenylene", poly4 - (2 * poly4 + 2) // 3),
            ("phenanthrene", self.n // 2),
            ("anthracene", self.p // 2),
            ("naphthalene", self.q // 2),
        ]
        for kind, amount in targets:
            for _ in range(amount):
                if not self._make(kind) and not self._downgrade(kind):
                    break
        progress = True
        while progress:
            progress = any(self._make(kind) for kind in (
                "tetracene", "chrysene", "triphenylene", "phenanthrene", "anthracene", "naphthalene"
            ))
        while self.remaining13 >= 6:
            self._make("benzene")
        if 4 <= self.remaining13 < 6:
            # The legacy initializer permits a final 1-2 site top-up.
            self.remaining13 = 6
            self._make("benzene")
        return list(self.kinds)


def _make_cluster_template(kind: str) -> _ClusterTemplate:
    centers = list(AROMATIC_CLUSTER_CENTERS[kind])
    memberships: Dict[Axial, List[int]] = defaultdict(list)
    rings: List[Tuple[Axial, ...]] = []
    for ring_index, center in enumerate(centers):
        ring = tuple((center[0] + dq, center[1] + dr) for dq, dr in HEX_VERTEX_OFFSETS)
        rings.append(ring)
        for coord in ring:
            memberships[coord].append(ring_index)
    coords = sorted(memberships)
    lookup = {coord: index for index, coord in enumerate(coords)}
    edge_set: Set[Tuple[int, int]] = set()
    for ring in rings:
        indices = [lookup[coord] for coord in ring]
        for index in range(6):
            edge_set.add(tuple(sorted((indices[index], indices[(index + 1) % 6]))))
    return _ClusterTemplate(
        kind=kind,
        centers=centers,
        coords=coords,
        shared=[len(memberships[coord]) >= 2 for coord in coords],
        edges=sorted(edge_set),
        rings=rings,
    )


def _cart(coord: Axial) -> Tuple[float, float]:
    return axial_to_cartesian(coord)


def _outward_direction(template: _ClusterTemplate, slot_index: int) -> Axial:
    coord = template.coords[slot_index]
    used = set()
    for left, right in template.edges:
        if left == slot_index:
            other = template.coords[right]
        elif right == slot_index:
            other = template.coords[left]
        else:
            continue
        used.add((other[0] - coord[0], other[1] - coord[1]))
    blocked = set(template.coords) | set(template.centers)
    choices = [
        direction for direction in valid_hex_directions(coord)
        if direction not in used
        and (coord[0] + direction[0], coord[1] + direction[1]) not in blocked
    ]
    if not choices:
        return RD
    cx = sum(_cart(center)[0] for center in template.centers) / max(1, len(template.centers))
    cy = sum(_cart(center)[1] for center in template.centers) / max(1, len(template.centers))
    px, py = _cart(coord)
    radial = (px - cx, py - cy)
    return max(choices, key=lambda direction: radial[0] * _cart(direction)[0] + radial[1] * _cart(direction)[1])


def _extend_templates_to_node_count(templates: List[_ClusterTemplate], target_count: int) -> None:
    current = sum(len(template.coords) for template in templates)
    while current < target_count:
        template = max(templates, key=lambda item: (len(item.coords), item.kind))
        occupied = set(template.coords) | set(template.centers)
        added = False
        for base_index in range(len(template.coords)):
            direction = _outward_direction(template, base_index)
            coord = (
                template.coords[base_index][0] + direction[0],
                template.coords[base_index][1] + direction[1],
            )
            if coord in occupied:
                continue
            template.coords.append(coord)
            template.shared.append(False)
            template.edges.append((base_index, len(template.coords) - 1))
            current += 1
            added = True
            break
        if not added:
            raise InverseInputError("cannot extend aromatic template to cover all inverse nodes")
    if current != target_count:
        raise InverseInputError(
            f"aromatic templates contain {current} sites for {target_count} inverse aromatic nodes"
        )


def _rigid_pairs(nodes: Mapping[int, NodeRow]) -> List[Tuple[int, int]]:
    pairs = {
        tuple(sorted((row.node_id, neighbor)))
        for row in nodes.values()
        if row.su_type == 10
        for neighbor in row.hop1_ids
        if nodes[neighbor].su_type == 10
    }
    degree = Counter(node_id for pair in pairs for node_id in pair)
    su10_ids = {row.node_id for row in nodes.values() if row.su_type == 10}
    if set(degree) != su10_ids or any(value != 1 for value in degree.values()):
        raise InverseInputError(
            "SU10 rigid initialization requires exactly one SU10 partner per SU10 node"
        )
    return sorted(pairs)


def _phase_for_product(product_type: str, boundary_count: int) -> str:
    kind = str(product_type or "").lower()
    if kind == "flex_chain":
        return "flex"
    if kind == "side_chain":
        return "side"
    if kind == "residual_resource_conversion":
        return "flex" if boundary_count >= 2 else "side"
    raise InverseInputError(f"unsupported topology product_type: {product_type!r}")


def _turn_directions(direction: Axial) -> List[Axial]:
    index = HEX_VERTEX_OFFSETS.index(direction)
    return [HEX_VERTEX_OFFSETS[(index - 1) % 6], HEX_VERTEX_OFFSETS[(index + 1) % 6]]


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
) -> None:
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for left, right in parsed.edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    for left, right in parsed.edges:
        delta = (
            coords[right][0] - coords[left][0],
            coords[right][1] - coords[left][1],
        )
        if delta not in HEX_DIRECTIONS:
            raise InverseInputError(f"topology edge {left}-{right} is not a hex neighbor")
        if (
            25 not in {int(su_by_id.get(left, -1)), int(su_by_id.get(right, -1))}
            and int(su_by_id.get(left, -1)) <= 25
            and int(su_by_id.get(right, -1)) <= 25
            and len(adjacency[int(left)]) <= 2
            and len(adjacency[int(right)]) <= 2
            and delta not in valid_hex_directions(coords[left])
        ):
            raise InverseInputError(f"topology edge {left}-{right} violates honeycomb parity")


def _layout_topology_tree(parsed: ParsedTopology, su_by_id: Mapping[int, int]) -> Dict[int, Axial]:
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

    su25_standard_direction: Dict[int, Axial] = {}
    for parent, rows in grouped.items():
        # SU25's long branch uses the ordinary third honeycomb direction.  Its
        # extra one-node substituent uses the opposite, fourth connection site.
        if int(su_by_id.get(parent, -1)) == 25:
            rows = sorted(rows, key=lambda item: (-len(item[1]), item[0]))
        for _order, path in rows:
            previous_id = parent
            previous_previous_id: Optional[int] = None
            branch_forward: Optional[Tuple[float, float]] = None
            if parent in parsed.main_path:
                parent_index = parsed.main_path.index(parent)
                if 0 < parent_index < len(parsed.main_path) - 1:
                    before = coords[parsed.main_path[parent_index - 1]]
                    after = coords[parsed.main_path[parent_index + 1]]
                    branch_forward = _cart((
                        after[0] - before[0], after[1] - before[1]
                    ))
                elif parent_index + 1 < len(parsed.main_path):
                    after = coords[parsed.main_path[parent_index + 1]]
                    branch_forward = _cart((
                        after[0] - coords[parent][0],
                        after[1] - coords[parent][1],
                    ))
                elif parent_index > 0:
                    before = coords[parsed.main_path[parent_index - 1]]
                    branch_forward = _cart((
                        coords[parent][0] - before[0],
                        coords[parent][1] - before[1],
                    ))
            for path_index, node_id in enumerate(path):
                parent_coord = coords[previous_id]
                used = {
                    (coords[neighbor][0] - parent_coord[0], coords[neighbor][1] - parent_coord[1])
                    for neighbor in adjacency[previous_id]
                    if neighbor in coords
                }
                preferred: List[Axial]
                if path_index == 0:
                    preferred = [
                        direction for direction in valid_hex_directions(parent_coord)
                        if direction not in used
                    ]
                    if int(su_by_id.get(parent, -1)) == 25:
                        if preferred and parent not in su25_standard_direction:
                            su25_standard_direction[parent] = preferred[0]
                        elif parent in su25_standard_direction:
                            standard = su25_standard_direction[parent]
                            opposite = (-standard[0], -standard[1])
                            preferred = [opposite] + preferred
                else:
                    assert previous_previous_id is not None
                    movement = (
                        parent_coord[0] - coords[previous_previous_id][0],
                        parent_coord[1] - coords[previous_previous_id][1],
                    )
                    back = (-movement[0], -movement[1])
                    forward = [
                        direction for direction in valid_hex_directions(parent_coord)
                        if direction != back and direction not in used
                    ]
                    mx, my = _cart(movement)
                    preferred = sorted(
                        forward,
                        key=lambda direction: (
                            (
                                branch_forward[0] * _cart(direction)[0]
                                + branch_forward[1] * _cart(direction)[1]
                                if branch_forward is not None else 0.0
                            ),
                            mx * _cart(direction)[0]
                            + my * _cart(direction)[1],
                            direction,
                        ),
                        reverse=True,
                    )
                chosen = None
                for direction in preferred:
                    candidate = (parent_coord[0] + direction[0], parent_coord[1] + direction[1])
                    if candidate not in occupied:
                        chosen = candidate
                        break
                if chosen is None:
                    raise InverseInputError(
                        f"cannot embed topology branch at GID {previous_id} on the honeycomb grid"
                    )
                coords[node_id] = chosen
                occupied.add(chosen)
                previous_previous_id, previous_id = previous_id, node_id

    if set(coords) != set(parsed.nodes):
        missing = sorted(set(parsed.nodes) - set(coords))
        raise InverseInputError(f"topology layout missed GIDs: {missing}")

    # A degree-3 ring connector can terminate at two adjacent aromatic sites:
    # parent-inner-mainBoundary and parent-inner-branchBoundary form the two
    # sides of the same attachment edge.  A plain tree walk opens these paths
    # by two lattice steps and destroys the ring closure.  Fold only this
    # explicitly identifiable terminal motif; all node identities and edges
    # remain unchanged.
    for parent, path in parsed.branches:
        if (
            len(path) != 2
            or int(su_by_id.get(path[-1], -1)) not in AROMATIC_SU_TYPES
        ):
            continue
        if parent not in parsed.main_path:
            continue
        parent_index = parsed.main_path.index(parent)
        main_tail = parsed.main_path[parent_index + 1:]
        if (
            len(main_tail) != 2
            or int(su_by_id.get(main_tail[-1], -1)) not in AROMATIC_SU_TYPES
        ):
            continue
        main_inner, main_boundary = map(int, main_tail)
        branch_inner, branch_boundary = map(int, path)
        for node_id in (main_inner, main_boundary, branch_inner, branch_boundary):
            occupied.discard(coords[node_id])
        parent_coord = coords[int(parent)]
        coords[main_inner] = (parent_coord[0], parent_coord[1] - 1)
        coords[main_boundary] = (parent_coord[0] + 1, parent_coord[1])
        coords[branch_inner] = (parent_coord[0], parent_coord[1] + 1)
        coords[branch_boundary] = (parent_coord[0] + 1, parent_coord[1] + 1)
        new_values = {
            coords[main_inner], coords[main_boundary],
            coords[branch_inner], coords[branch_boundary],
        }
        if len(new_values) != 4 or new_values & occupied:
            raise InverseInputError(
                f"two-site aromatic attachment collides at topology parent GID {parent}"
            )
        occupied.update(new_values)

    # A branch which terminates in a second aromatic boundary beside the
    # main-path boundary represents a two-site attachment to one aromatic
    # perimeter (for example the double20d3 unit in YM-8).  The topology CSV
    # omits the aromatic ring edge because that edge belongs to the aromatic
    # unit, but the two virtual boundary coordinates must still be adjacent.
    aromatic_boundaries = {
        int(node_id)
        for node_id in parsed.nodes
        if int(su_by_id.get(node_id, -1)) in AROMATIC_SU_TYPES
        and len(adjacency[node_id]) == 1
    }
    main_boundaries = [
        node_id for node_id in (parsed.main_path[0], parsed.main_path[-1])
        if node_id in aromatic_boundaries
    ]
    for _parent, path in parsed.branches:
        if len(path) < 2 or path[-1] not in aromatic_boundaries:
            continue
        branch_boundary = int(path[-1])
        branch_inner = int(path[-2])
        old_coordinate = coords[branch_boundary]
        candidates = []
        for main_boundary in main_boundaries:
            if main_boundary == branch_boundary:
                continue
            common = (
                {
                    (
                        coords[branch_inner][0] + direction[0],
                        coords[branch_inner][1] + direction[1],
                    )
                    for direction in HEX_DIRECTIONS
                }
                & {
                    (
                        coords[main_boundary][0] + direction[0],
                        coords[main_boundary][1] + direction[1],
                    )
                    for direction in HEX_DIRECTIONS
                }
            )
            for candidate in common:
                if candidate == old_coordinate or candidate not in occupied:
                    dq = coords[main_boundary][0] - coords[branch_inner][0]
                    dr = coords[main_boundary][1] - coords[branch_inner][1]
                    candidates.append((
                        (abs(dq) + abs(dr) + abs(dq - dr)) // 2,
                        main_boundary,
                        candidate,
                    ))
        if candidates:
            _distance, _main_boundary, selected = min(candidates)
            occupied.discard(old_coordinate)
            coords[branch_boundary] = selected
            occupied.add(selected)

    _validate_topology_layout(parsed, su_by_id, coords)
    return coords


def _assign_aromatic_nodes(
    nodes: Mapping[int, NodeRow],
    topology_rows: Sequence[Mapping[str, object]],
    templates: List[_ClusterTemplate],
    rigid_pairs: Sequence[Tuple[int, int]],
    aromatic_ids: Optional[Set[int]] = None,
    spread_flex_slots: bool = True,
) -> Tuple[Dict[int, Tuple[int, int]], Dict[str, object]]:
    aromatic_ids = (
        set(aromatic_ids)
        if aromatic_ids is not None
        else {
            node_id
            for node_id, row in nodes.items()
            if row.su_type in AROMATIC_SU_TYPES
        }
    )
    boundary_by_product: List[List[int]] = []
    boundary_ids: Set[int] = set()
    for raw in topology_rows:
        parsed = raw["parsed"]
        assert isinstance(parsed, ParsedTopology)
        degrees = Counter(node_id for edge in parsed.edges for node_id in edge)
        boundaries = [
            node_id for node_id in parsed.nodes
            if node_id in aromatic_ids and degrees[node_id] == 1
        ]
        internal_aromatic = [node_id for node_id in parsed.nodes if node_id in aromatic_ids and degrees[node_id] != 1]
        if internal_aromatic:
            raise InverseInputError(
                f"topology row {raw['row_index']} contains non-boundary aromatic GIDs: {internal_aromatic}"
            )
        boundary_by_product.append(boundaries)
        boundary_ids.update(boundaries)

    exterior_slots = [
        (cluster_index, slot_index)
        for cluster_index, template in enumerate(templates)
        for slot_index, shared in enumerate(template.shared)
        if not shared
    ]
    shared_slots = [
        (cluster_index, slot_index)
        for cluster_index, template in enumerate(templates)
        for slot_index, shared in enumerate(template.shared)
        if shared
    ]
    assignment: Dict[int, Tuple[int, int]] = {}
    used_slots: Set[Tuple[int, int]] = set()
    fixed_flex_coordinates: Dict[int, Axial] = {}
    rigid_endpoint_ids = {
        int(node_id) for pair in rigid_pairs for node_id in pair
    }
    cluster_load = Counter()
    dsu = _DSU(len(templates))

    def rotate(value: Axial, turns: int) -> Axial:
        q, r = int(value[0]), int(value[1])
        for _ in range(int(turns) % 6):
            q, r = q - r, q
        return q, r

    def rigid_pair_slots(
        left_cluster: int,
        right_cluster: int,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find two exterior sites that can form a collision-free rigid bond."""
        left_choices = [
            slot for slot in exterior_slots
            if slot not in used_slots and slot[0] == left_cluster
        ]
        right_choices = [
            slot for slot in exterior_slots
            if slot not in used_slots and slot[0] == right_cluster
        ]
        left_template = templates[left_cluster]
        right_template = templates[right_cluster]
        left_nodes = set(left_template.coords)
        right_nodes = set(right_template.coords)
        left_centers = set(left_template.centers)
        right_centers = set(right_template.centers)
        for left_slot in left_choices:
            left_coord = left_template.coords[left_slot[1]]
            left_direction = _outward_direction(left_template, left_slot[1])
            for right_slot in right_choices:
                right_coord = right_template.coords[right_slot[1]]
                right_direction = _outward_direction(right_template, right_slot[1])
                for turns in range(6):
                    if rotate(right_direction, turns) != (-left_direction[0], -left_direction[1]):
                        continue
                    rotated_right = rotate(right_coord, turns)
                    translation = (
                        left_coord[0] + left_direction[0] - rotated_right[0],
                        left_coord[1] + left_direction[1] - rotated_right[1],
                    )
                    moved_nodes = {
                        (rotate(coord, turns)[0] + translation[0], rotate(coord, turns)[1] + translation[1])
                        for coord in right_nodes
                    }
                    moved_centers = {
                        (rotate(coord, turns)[0] + translation[0], rotate(coord, turns)[1] + translation[1])
                        for coord in right_centers
                    }
                    if moved_nodes.isdisjoint(left_nodes) and moved_centers.isdisjoint(left_centers):
                        return left_slot, right_slot
        return None

    def allocate(node_id: int, preferred_cluster: Optional[int] = None, excluded_roots: Set[int] | None = None) -> int:
        excluded_roots = excluded_roots or set()
        choices = [
            slot for slot in exterior_slots
            if slot not in used_slots
            and (preferred_cluster is None or slot[0] == preferred_cluster)
            and dsu.find(slot[0]) not in excluded_roots
        ]
        if not choices and preferred_cluster is not None:
            return allocate(node_id, None, excluded_roots)
        if not choices and excluded_roots:
            return allocate(node_id, preferred_cluster, set())
        if not choices:
            raise InverseInputError(f"no exterior aromatic slot available for GID {node_id}")
        slot = min(choices, key=lambda item: (cluster_load[item[0]], item[0], item[1]))
        assignment[node_id] = slot
        used_slots.add(slot)
        cluster_load[slot[0]] += 1
        return slot[0]

    # Rigid SU10 endpoints are fixed by the node CSV.  Spreading the pairs
    # over separate templates reproduces the legacy rigid-cluster start.
    for pair_index, (left, right) in enumerate(rigid_pairs):
        preferred_left = (2 * pair_index) % len(templates)
        preferred_right = (2 * pair_index + 1) % len(templates)
        pair_slots = rigid_pair_slots(preferred_left, preferred_right)
        if pair_slots is None:
            # Preserve the legacy pairing order where possible, but search for
            # another unused pair when a particular template orientation has
            # no non-overlapping para connection.
            pair_slots = next(
                (
                    candidate
                    for left_cluster in range(len(templates))
                    for right_cluster in range(len(templates))
                    if left_cluster != right_cluster
                    and (candidate := rigid_pair_slots(left_cluster, right_cluster)) is not None
                ),
                None,
            )
        if pair_slots is None:
            raise InverseInputError(f"no collision-free aromatic rigid slot pair for SU10 GIDs {left}-{right}")
        left_slot, right_slot = pair_slots
        left_cluster, right_cluster = left_slot[0], right_slot[0]
        assignment[left] = left_slot
        assignment[right] = right_slot
        used_slots.update({left_slot, right_slot})
        cluster_load[left_cluster] += 1
        cluster_load[right_cluster] += 1
        dsu.union(left_cluster, right_cluster)

    # Flex products are assigned first so their prescribed bridges connect
    # the rigid components instead of accidentally becoming intra-component
    # decorations.  Side anchors are balanced after the skeleton is fixed.
    phases = [
        _phase_for_product(str(raw.get("product_type", "")), len(boundaries))
        for raw, boundaries in zip(topology_rows, boundary_by_product)
    ]
    # Multi-ended hubs must claim distinct, compact aromatic components before
    # ordinary two-ended bridges consume those components.  The CSV row order
    # is a serialization detail; it must not decide which rigid components are
    # left for the only topology that can merge more than two of them.
    flex_product_indices = sorted(
        (
            index for index, phase in enumerate(phases)
            if phase == "flex"
        ),
        key=lambda index: (
            -len(boundary_by_product[index]),
            int(topology_rows[index]["row_index"]),
        ),
    )
    for product_index in flex_product_indices:
        boundaries = boundary_by_product[product_index]
        selected: List[int] = []
        selected_roots: Set[int] = set()
        for node_id in boundaries:
            cluster_index = allocate(node_id, excluded_roots=selected_roots)
            selected.append(cluster_index)
            selected_roots.add(dsu.find(cluster_index))

        parsed = topology_rows[product_index]["parsed"]
        assert isinstance(parsed, ParsedTopology)
        main_su = tuple(nodes[node_id].su_type for node_id in parsed.main_path)
        if len(boundaries) == 3 and main_su == (9, 0, 20, 20, 27, 6):
            branch_boundaries = [
                node_id for node_id in boundaries
                if node_id not in {parsed.main_path[0], parsed.main_path[-1]}
            ]
            if len(branch_boundaries) != 1:
                raise InverseInputError(
                    "three-aromatic hub must expose one branch boundary"
                )
            preferred_coordinates = {
                int(parsed.main_path[0]): (-1, 0),
                int(parsed.main_path[-1]): (1, 0),
                int(branch_boundaries[0]): (-1, -1),
            }
            for node_id, coordinate in preferred_coordinates.items():
                old_slot = assignment[node_id]
                cluster_index = old_slot[0]
                template = templates[cluster_index]
                try:
                    slot_index = template.coords.index(coordinate)
                except ValueError as exc:
                    raise InverseInputError(
                        f"three-aromatic hub slot {coordinate} is unavailable on "
                        f"cluster {cluster_index}"
                    ) from exc
                preferred_slot = (cluster_index, slot_index)
                if preferred_slot in used_slots and preferred_slot != old_slot:
                    raise InverseInputError(
                        "three-aromatic hub preferred slot was consumed before "
                        f"hub initialization: {preferred_slot}"
                    )
                used_slots.discard(old_slot)
                used_slots.add(preferred_slot)
                assignment[node_id] = preferred_slot
                fixed_flex_coordinates[int(node_id)] = coordinate
        for cluster_index in selected[1:]:
            dsu.union(selected[0], cluster_index)

    # The CSV fixes which GID belongs to each flex product but does not carry
    # an aromatic-ring slot coordinate.  Within an already selected aromatic
    # cluster, match long bridge endpoints to lateral slots and reserve
    # vertical slots for one/two-body bridges.  This is geometry assignment,
    # not resource allocation: product identity, cluster ownership and every
    # topology node remain unchanged.
    body_length_by_boundary: Dict[int, int] = {}
    flex_boundary_ids: Set[int] = set()
    for raw, boundaries, phase in zip(
        topology_rows, boundary_by_product, phases
    ):
        if phase != "flex":
            continue
        parsed = raw["parsed"]
        assert isinstance(parsed, ParsedTopology)
        main_body_length = sum(
            node_id not in aromatic_ids for node_id in parsed.main_path
        )
        for node_id in boundaries:
            body_length_by_boundary[int(node_id)] = int(main_body_length)
            flex_boundary_ids.add(int(node_id))

    for cluster_index, template in enumerate(templates):
        cluster_boundaries = sorted(
            node_id for node_id in flex_boundary_ids
            if assignment.get(node_id, (-1, -1))[0] == cluster_index
        )
        if not cluster_boundaries:
            continue
        fixed_boundaries = [
            node_id for node_id in cluster_boundaries
            if node_id in fixed_flex_coordinates
        ]
        movable_boundaries = [
            node_id for node_id in cluster_boundaries
            if node_id not in fixed_flex_coordinates
        ]

        # A three-way hub fixes one attachment direction on each of its three
        # aromatic rings.  Keep that direction immutable and reserve the
        # diametrically opposite site for the next flex bridge on the same
        # ring.  This avoids folding the following bridge into the compact hub
        # while preserving every CSV GID and aromatic-cluster ownership.
        preferred_by_boundary: Dict[int, Tuple[int, int]] = {}
        if len(fixed_boundaries) == 1 and movable_boundaries:
            fixed_coordinate = fixed_flex_coordinates[fixed_boundaries[0]]
            opposite_coordinate = (
                -int(fixed_coordinate[0]), -int(fixed_coordinate[1])
            )
            if opposite_coordinate in template.coords:
                opposite_slot = (
                    cluster_index, template.coords.index(opposite_coordinate)
                )
                if (
                    opposite_slot not in used_slots
                    or opposite_slot
                    in {assignment[node_id] for node_id in movable_boundaries}
                ):
                    preferred_by_boundary[movable_boundaries[0]] = opposite_slot

        current_slots = {
            assignment[node_id] for node_id in movable_boundaries
        }
        available_slots = (
            sorted(
                slot for slot in exterior_slots
                if slot[0] == cluster_index
                and (
                    slot not in used_slots
                    or slot in current_slots
                )
            )
            if spread_flex_slots or fixed_boundaries
            else sorted(current_slots)
        )
        if len(movable_boundaries) > len(available_slots):
            raise InverseInputError(
                f"not enough flex slots on aromatic cluster {cluster_index}"
            )

        def slot_is_vertical(slot: Tuple[int, int]) -> bool:
            direction = _outward_direction(template, slot[1])
            x, y = _cart(direction)
            return abs(y) > abs(x)

        def pair_cost(node_id: int, slot: Tuple[int, int]) -> Tuple[int, int]:
            length = int(body_length_by_boundary.get(node_id, 0))
            vertical = slot_is_vertical(slot)
            orientation_penalty = (
                1000 if length > 2 and vertical
                else 20 if length <= 2 and not vertical
                else 0
            )
            return orientation_penalty, slot[1]

        fixed_slots = [assignment[node_id] for node_id in fixed_boundaries]
        rigid_slots = [
            assignment[node_id]
            for node_id in rigid_endpoint_ids
            if assignment.get(node_id, (-1, -1))[0] == cluster_index
        ]

        def separation_rank(
            candidate: Tuple[Tuple[int, int], ...],
        ) -> Tuple[float, float]:
            coordinates = [
                template.coords[slot[1]]
                for slot in rigid_slots + fixed_slots + list(candidate)
            ]
            distances = [
                sum(
                    value * value
                    for value in (
                        _cart(left)[0] - _cart(right)[0],
                        _cart(left)[1] - _cart(right)[1],
                    )
                )
                for left, right in itertools.combinations(coordinates, 2)
            ]
            if not distances:
                return 0.0, 0.0
            return -min(distances), -sum(distances)

        best = min(
            itertools.permutations(
                available_slots, len(movable_boundaries)
            ),
            key=lambda candidate: (
                sum(
                    100000
                    for node_id, slot in zip(movable_boundaries, candidate)
                    if node_id in preferred_by_boundary
                    and slot != preferred_by_boundary[node_id]
                ),
                separation_rank(candidate),
                sum(
                    pair_cost(node_id, slot)[0]
                    for node_id, slot in zip(movable_boundaries, candidate)
                ),
                tuple(
                    pair_cost(node_id, slot)[1]
                    for node_id, slot in zip(movable_boundaries, candidate)
                ),
            ),
        )
        old_movable_slots = {
            assignment[node_id] for node_id in movable_boundaries
        }
        used_slots.difference_update(old_movable_slots)
        for node_id, slot in zip(movable_boundaries, best):
            assignment[node_id] = slot
            used_slots.add(slot)

    for phase, boundaries in zip(phases, boundary_by_product):
        if phase != "side":
            continue
        for node_id in boundaries:
            allocate(node_id)

    peripheral = sorted(
        aromatic_ids - set(assignment),
        key=lambda node_id: (nodes[node_id].su_type == 12, nodes[node_id].su_type, node_id),
    )
    shared_nodes = [node_id for node_id in peripheral if nodes[node_id].su_type == 12]
    nonshared_nodes = [node_id for node_id in peripheral if nodes[node_id].su_type != 12]
    if len(shared_nodes) > len(shared_slots):
        raise InverseInputError("more SU12 nodes than fused aromatic bridgehead slots")
    while len(shared_nodes) < len(shared_slots):
        if not nonshared_nodes:
            raise InverseInputError("cannot fill all fused aromatic bridgehead slots")
        shared_nodes.append(nonshared_nodes.pop())
    for node_id, slot in zip(shared_nodes, shared_slots):
        assignment[node_id] = slot
        used_slots.add(slot)
    remaining_exterior = [slot for slot in exterior_slots if slot not in used_slots]
    if len(nonshared_nodes) != len(remaining_exterior):
        raise InverseInputError(
            f"aromatic node/slot mismatch after fixed anchors: nodes={len(nonshared_nodes)} slots={len(remaining_exterior)}"
        )
    for node_id, slot in zip(nonshared_nodes, remaining_exterior):
        assignment[node_id] = slot
        used_slots.add(slot)
    if set(assignment) != aromatic_ids:
        raise InverseInputError("aromatic assignment did not cover every inverse aromatic node")
    return assignment, {
        "boundary_ids": boundary_ids,
        "boundary_by_product": boundary_by_product,
        "phases": phases,
        "bridgehead_slot_count": len(shared_slots),
    }


def build_csv_inverse_input(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
) -> InverseAssemblyInput:
    nodes = load_node_rows(nodes_csv)
    topology_rows = load_topology_rows(topology_specs_csv)
    counts = Counter(row.su_type for row in nodes.values())
    bridgeheads = _bridgehead_counts(nodes)
    kinds = _AromaticPlanner(counts, bridgeheads).plan()
    if not kinds:
        raise InverseInputError("node CSV does not produce any aromatic cluster")
    templates = [_make_cluster_template(kind) for kind in kinds]
    rigid_pairs = _rigid_pairs(nodes)
    source_aromatic_ids = {
        node_id for node_id, row in nodes.items()
        if row.su_type in AROMATIC_SU_TYPES
    }
    ring_slot_count = sum(len(template.coords) for template in templates)
    surplus_count = len(source_aromatic_ids) - ring_slot_count
    if surplus_count < 0:
        raise InverseInputError(
            f"aromatic templates contain {ring_slot_count} ring sites for "
            f"{len(source_aromatic_ids)} inverse aromatic nodes"
        )

    excluded_aromatic_ids: Set[int] = set()
    if surplus_count:
        # Reproduce the old assignment only to identify which GIDs occupied the
        # synthetic, non-ring slots.  The real templates below are never
        # extended, so those coordinates cannot enter the assembly state.
        provisional_templates = [_make_cluster_template(kind) for kind in kinds]
        base_lengths = [len(template.coords) for template in provisional_templates]
        _extend_templates_to_node_count(
            provisional_templates, len(source_aromatic_ids)
        )
        provisional_assignment, _ = _assign_aromatic_nodes(
            nodes,
            topology_rows,
            provisional_templates,
            rigid_pairs,
            aromatic_ids=source_aromatic_ids,
            spread_flex_slots=False,
        )
        excluded_aromatic_ids = {
            node_id
            for node_id, (cluster_index, slot_index) in provisional_assignment.items()
            if slot_index >= base_lengths[cluster_index]
        }
        topology_referenced_ids = {
            int(node_id)
            for row in topology_rows
            for node_id in row["parsed"].nodes
        }
        rigid_endpoint_ids = {
            int(node_id) for pair in rigid_pairs for node_id in pair
        }
        invalid_exclusions = sorted(
            node_id
            for node_id in excluded_aromatic_ids
            if nodes[node_id].su_type != 13
            or node_id in topology_referenced_ids
            or node_id in rigid_endpoint_ids
        )
        if len(excluded_aromatic_ids) != surplus_count or invalid_exclusions:
            raise InverseInputError(
                "surplus aromatic nodes cannot be safely excluded; "
                f"expected={surplus_count} identified={sorted(excluded_aromatic_ids)} "
                f"invalid={invalid_exclusions}"
            )

    aromatic_ids = source_aromatic_ids - excluded_aromatic_ids
    assignment, assignment_info = _assign_aromatic_nodes(
        nodes,
        topology_rows,
        templates,
        rigid_pairs,
        aromatic_ids=aromatic_ids,
    )

    topology_boundaries = assignment_info["boundary_by_product"]
    topology_phases = assignment_info["phases"]
    assert isinstance(topology_boundaries, list)
    assert isinstance(topology_phases, list)

    aromatic_port_requests: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)
    topology_units: List[TopologyUnit] = []
    contracts: List[ExternalContract] = []
    ownership: Dict[int, str] = {}

    for pair_index, (left, right) in enumerate(rigid_pairs):
        left_cluster, _left_slot = assignment[left]
        right_cluster, _right_slot = assignment[right]
        left_unit, right_unit = f"aromatic:{left_cluster}", f"aromatic:{right_cluster}"
        left_port = f"{left_unit}:rigid:{left}:{right}"
        right_port = f"{right_unit}:rigid:{right}:{left}"
        aromatic_port_requests[left].append((right, left_port, "rigid"))
        aromatic_port_requests[right].append((left, right_port, "rigid"))
        contracts.append(ExternalContract(
            contract_id=f"rigid:{pair_index}:{left}-{right}",
            source_unit_id=left_unit,
            source_node_id=left,
            target_unit_id=right_unit,
            target_node_id=right,
            kind="rigid",
            fixed=True,
            source_port_id=left_port,
            target_port_id=right_port,
        ))

    su_by_id = {
        node_id: row.su_type
        for node_id, row in nodes.items()
        if node_id not in excluded_aromatic_ids
    }
    su25_nodes: Set[int] = set()
    for raw, boundaries, phase in zip(topology_rows, topology_boundaries, topology_phases):
        parsed = raw["parsed"]
        assert isinstance(parsed, ParsedTopology)
        layout = _layout_topology_tree(parsed, su_by_id)
        internal_ids = [node_id for node_id in parsed.nodes if node_id not in aromatic_ids]
        if not internal_ids:
            raise InverseInputError(f"topology row {raw['row_index']} has no non-aromatic body")
        unit_id = f"topology:{raw['row_index']}"
        origin = layout[internal_ids[0]]
        local = {
            node_id: (layout[node_id][0] - origin[0], layout[node_id][1] - origin[1])
            for node_id in internal_ids
        }
        internal_set = set(internal_ids)
        internal_edges = tuple(
            UnitEdge(left, right, phase)
            for left, right in parsed.edges
            if left in internal_set and right in internal_set
        )
        degree = Counter(node_id for edge in parsed.edges for node_id in edge)
        for node_id in internal_ids:
            if su_by_id[node_id] == 25:
                su25_nodes.add(node_id)
                neighbors = [other for edge in parsed.edges if node_id in edge for other in edge if other != node_id]
                neighbor_coords = {layout[neighbor] for neighbor in neighbors}
                if degree[node_id] != 4 or len(neighbor_coords) != 4:
                    raise InverseInputError(
                        f"SU25 GID {node_id} must have four distinct hex connection sites"
                    )

        ports: List[UnitPort] = []
        boundary_set = set(boundaries)
        for boundary_id in boundaries:
            cross_edges = [edge for edge in parsed.edges if boundary_id in edge]
            if len(cross_edges) != 1:
                raise InverseInputError(
                    f"aromatic boundary GID {boundary_id} must have exactly one topology edge"
                )
            edge = cross_edges[0]
            internal_id = edge[1] if edge[0] == boundary_id else edge[0]
            direction = (
                layout[boundary_id][0] - layout[internal_id][0],
                layout[boundary_id][1] - layout[internal_id][1],
            )
            port_id = f"{unit_id}:port:{internal_id}:{boundary_id}"
            ports.append(UnitPort(
                port_id=port_id,
                node_id=internal_id,
                role="topology_boundary",
                phase=phase,
                direction=direction,
                contract_id=f"{phase}:{raw['row_index']}:{boundary_id}-{internal_id}",
            ))
            aromatic_unit_index, _slot = assignment[boundary_id]
            aromatic_unit_id = f"aromatic:{aromatic_unit_index}"
            aromatic_port_id = f"{aromatic_unit_id}:port:{boundary_id}:{internal_id}"
            aromatic_port_requests[boundary_id].append((internal_id, aromatic_port_id, phase))
            contracts.append(ExternalContract(
                contract_id=f"{phase}:{raw['row_index']}:{boundary_id}-{internal_id}",
                source_unit_id=unit_id,
                source_node_id=internal_id,
                target_unit_id=aromatic_unit_id,
                target_node_id=boundary_id,
                kind=phase,
                fixed=True,
                source_port_id=port_id,
                target_port_id=aromatic_port_id,
            ))

        topology_unit = TopologyUnit(
            unit_id=unit_id,
            product_type=f"{phase}_chain",
            nodes=tuple(UnitNode(
                node_id=node_id,
                su_type=su_by_id[node_id],
                local_axial=local[node_id],
                role="topology_node",
                metadata=(("source_product_type", str(raw.get("product_type", ""))),),
            ) for node_id in internal_ids),
            internal_edges=internal_edges,
            ports=tuple(ports),
            source_fragment_ids=tuple(raw["source_fragment_ids_parsed"]),
            source_node_ids=tuple(internal_ids),
            topology_notation=str(raw.get("actual_su", "")),
            node_id_notation=str(raw.get("gid", "")),
            closed=True,
            metadata=(
                ("csv_row", str(raw["row_index"])),
                ("source_product_type", str(raw.get("product_type", ""))),
            ),
        )
        topology_units.append(topology_unit)
        for node_id in internal_ids:
            if node_id in ownership:
                raise InverseInputError(f"topology GID {node_id} is claimed more than once")
            ownership[node_id] = unit_id

    aromatic_units: List[TopologyUnit] = []
    by_cluster: Dict[int, Dict[int, int]] = defaultdict(dict)
    for node_id, (cluster_index, slot_index) in assignment.items():
        by_cluster[cluster_index][slot_index] = node_id
    for cluster_index, template in enumerate(templates):
        unit_id = f"aromatic:{cluster_index}"
        slot_to_node = by_cluster[cluster_index]
        if set(slot_to_node) != set(range(len(template.coords))):
            raise InverseInputError(f"aromatic template {cluster_index} has unassigned sites")
        unit_ports: List[UnitPort] = []
        for slot_index, node_id in sorted(slot_to_node.items()):
            direction = _outward_direction(template, slot_index)
            for _neighbor, port_id, phase in aromatic_port_requests.get(node_id, []):
                unit_ports.append(UnitPort(
                    port_id=port_id,
                    node_id=node_id,
                    role="aromatic_boundary",
                    phase=phase,
                    direction=direction,
                ))
        aromatic_units.append(TopologyUnit(
            unit_id=unit_id,
            product_type=template.kind,
            nodes=tuple(UnitNode(
                node_id=slot_to_node[slot_index],
                su_type=12 if template.shared[slot_index] else 13,
                local_axial=coord,
                role="aromatic_site",
                metadata=(
                    ("planned_su_type", str(su_by_id[slot_to_node[slot_index]])),
                    ("template_bridgehead", str(template.shared[slot_index])),
                ),
            ) for slot_index, coord in enumerate(template.coords)),
            internal_edges=tuple(UnitEdge(
                slot_to_node[left], slot_to_node[right], "aromatic"
            ) for left, right in template.edges),
            ports=tuple(unit_ports),
            source_node_ids=tuple(slot_to_node[index] for index in range(len(template.coords))),
            topology_notation=template.kind,
            closed=True,
            local_rings=tuple(template.rings),
            metadata=(("ring_count", str(len(template.centers))),),
        ))
        for node_id in slot_to_node.values():
            ownership[node_id] = unit_id

    expected_ids = set(nodes) - excluded_aromatic_ids
    if set(ownership) != expected_ids:
        raise InverseInputError(
            f"CSV initialization ownership mismatch; missing={sorted(expected_ids-set(ownership))[:12]} "
            f"unexpected={sorted(set(ownership)-expected_ids)[:12]}"
        )
    units = aromatic_units + topology_units
    contracts.sort(key=lambda item: ({"rigid": 0, "flex": 1, "side": 2}[item.kind], item.contract_id))
    return InverseAssemblyInput(
        units=tuple(units),
        contracts=tuple(contracts),
        node_su_types=tuple(sorted(su_by_id.items())),
        ownership=tuple(sorted(ownership.items())),
        diagnostics=tuple(sorted((
            ("initializer", "inverse_csv_pair"),
            ("nodes_csv", str(Path(nodes_csv))),
            ("topology_specs_csv", str(Path(topology_specs_csv))),
            ("node_count", str(len(su_by_id))),
            ("source_node_count", str(len(nodes))),
            ("assembly_node_count", str(len(su_by_id))),
            (
                "excluded_surplus_aromatic_node_ids",
                ",".join(str(value) for value in sorted(excluded_aromatic_ids)),
            ),
            ("aromatic_unit_count", str(len(aromatic_units))),
            ("topology_unit_count", str(len(topology_units))),
            ("rigid_contract_count", str(sum(c.kind == "rigid" for c in contracts))),
            ("flex_contract_count", str(sum(c.kind == "flex" for c in contracts))),
            ("side_contract_count", str(sum(c.kind == "side" for c in contracts))),
            ("su25_nodes", ",".join(str(value) for value in sorted(su25_nodes))),
            ("bridgehead_counts", ",".join(str(value) for value in bridgeheads)),
        ))),
        source_kind=InputSourceKind.CSV_PAIR,
    )


def _dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_public_dict"):
        try:
            return dict(value.to_public_dict() or {})
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict() or {})
        except Exception:
            pass
    if is_dataclass(value):
        return {
            str(name): getattr(value, name)
            for name in getattr(value, "__dataclass_fields__", {})
        }
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _axial(value: Any, default: Optional[Axial] = None) -> Optional[Axial]:
    if isinstance(value, Mapping):
        value = value.get("local_axial", value.get("axial", value.get("coord")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except Exception:
            return default
    return default


def _unique_ints(values: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(sorted({_int(value) for value in values if _int(value) >= 0}))


def _node_id_from_row(row: Any) -> int:
    data = _dict(row)
    return _int(data.get("node_id", data.get("global_id", getattr(row, "global_id", -1))))


def _extract_products(payload: Any) -> List[Any]:
    root = _dict(payload)
    for key in ("topology_products", "topology_units", "planned_topology", "products"):
        values = root.get(key)
        if isinstance(values, Mapping):
            values = list(values.values())
        if isinstance(values, (list, tuple)) and values:
            return list(values)
    for key in ("external_assembly", "assembly_view", "assembly"):
        nested = root.get(key)
        if nested is not None:
            found = _extract_products(nested)
            if found:
                return found
    return []


def _extract_aromatic_units(payload: Any) -> List[Any]:
    root = _dict(payload)
    for key in ("aromatic_units", "aromatic_clusters", "aromatic_unit_geometry", "clusters"):
        values = root.get(key)
        if isinstance(values, Mapping):
            values = list(values.values())
        if isinstance(values, (list, tuple)) and values:
            return list(values)
    for key in ("external_assembly", "assembly_view", "assembly"):
        nested = root.get(key)
        if nested is not None:
            found = _extract_aromatic_units(nested)
            if found:
                return found
    return []


def _extract_nodes(payload: Any) -> Dict[int, Dict[str, Any]]:
    root = _dict(payload)
    values = root.get(
        "nodes",
        root.get("node_table", root.get("final_nodes", root.get("node_rows", []))),
    )
    if isinstance(values, Mapping):
        values = list(values.values())
    result: Dict[int, Dict[str, Any]] = {}
    for row in list(values or []):
        data = _dict(row)
        node_id = _node_id_from_row(row)
        if node_id < 0:
            continue
        result[node_id] = {
            **data,
            "node_id": node_id,
            "su_type": _int(data.get("su_type", getattr(row, "su_type", -1))),
            "hop1_ids": list(data.get("hop1_ids", getattr(row, "hop1_ids", [])) or []),
            "fixed_hop1_ids": list(data.get("fixed_hop1_ids", getattr(row, "fixed_hop1_ids", [])) or []),
        }
    if not result:
        for key in ("external_assembly", "assembly_view", "assembly", "inverse_result"):
            nested = root.get(key)
            if nested is not None:
                result = _extract_nodes(nested)
                if result:
                    break
    return result


def _parse_ids(text: Any) -> List[int]:
    return [_int(value) for value in re.findall(r"\d+", str(text or ""))]


def _parse_su_tokens(text: Any) -> List[int]:
    values: List[int] = []
    for token in re.findall(r"\d+", str(text or "")):
        value = _int(token)
        if 0 <= value <= 32:
            values.append(value)
    return values


def _node_rows_for_unit(raw: Any, global_nodes: Dict[int, Dict[str, Any]], unit_id: str) -> List[Dict[str, Any]]:
    data = _dict(raw)
    rows = data.get("nodes", data.get("sites", data.get("local_nodes", [])))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    rows = list(rows or [])
    ids = list(data.get("source_node_ids", data.get("node_ids", [])) or [])
    if not rows and ids:
        su_tokens = _parse_su_tokens(data.get("topology_notation", data.get("actual_topology_su", "")))
        rows = [
            {
                "node_id": _int(node_id),
                "su_type": global_nodes.get(_int(node_id), {}).get("su_type", su_tokens[index] if index < len(su_tokens) else -1),
            }
            for index, node_id in enumerate(ids)
        ]
    # Product/fragment exports use ``main_path`` and ``branches`` as the
    # authoritative ordered topology.  Preserve those node IDs and roles at
    # the boundary instead of parsing labels or assigning fresh IDs.
    if not rows:
        path_rows: List[Any] = []
        path_rows.extend(list(data.get("main_path", []) or []))
        for branch in list(data.get("branches", []) or []):
            path_rows.extend(list(branch or []))
        for path_row in path_rows:
            item = _dict(path_row)
            if str(item.get("kind", "node")) != "node":
                continue
            node_id = _node_id_from_row(item)
            if node_id < 0:
                continue
            rows.append({
                "node_id": node_id,
                "su_type": item.get("su_type", global_nodes.get(node_id, {}).get("su_type", -1)),
                "role": item.get("role", ""),
            })
    if not rows:
        gid_ids = _parse_ids(data.get("node_id_notation", data.get("actual_topology_gid", "")))
        su_tokens = _parse_su_tokens(data.get("topology_notation", data.get("actual_topology_su", "")))
        rows = [
            {"node_id": node_id, "su_type": global_nodes.get(node_id, {}).get("su_type", su_tokens[index] if index < len(su_tokens) else -1)}
            for index, node_id in enumerate(gid_ids)
        ]

    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = _dict(row)
        node_id = _node_id_from_row(row)
        if node_id < 0:
            continue
        global_row = global_nodes.get(node_id, {})
        item["node_id"] = node_id
        item["su_type"] = _int(item.get("su_type", global_row.get("su_type", -1)))
        item["role"] = str(item.get("role", ""))
        item["local_axial"] = _axial(
            item.get(
                "local_axial",
                item.get("axial", item.get("coord", global_row.get("local_axial", global_row.get("axial")))),
            )
        )
        normalized.append(item)
    return normalized


def _normalize_edges(raw: Any, node_ids: Sequence[int]) -> List[UnitEdge]:
    data = _dict(raw)
    values = data.get("internal_edges", data.get("edges", []))
    if isinstance(values, Mapping):
        values = list(values.values())
    edges: List[UnitEdge] = []
    node_set = {int(node_id) for node_id in node_ids}
    for edge in list(values or []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            left, right = _int(edge[0]), _int(edge[1])
            kind = "internal"
        else:
            row = _dict(edge)
            left = _int(row.get("source_node_id", row.get("u", row.get("source", -1))))
            right = _int(row.get("target_node_id", row.get("v", row.get("target", -1))))
            kind = str(row.get("kind", "internal"))
        # Legacy aromatic cluster exports used site indices in ``edges``;
        # convert those indices to the inverse node IDs at the boundary.
        if left not in node_set and 0 <= left < len(node_ids):
            left = int(node_ids[left])
        if right not in node_set and 0 <= right < len(node_ids):
            right = int(node_ids[right])
        if left >= 0 and right >= 0 and left != right:
            edges.append(UnitEdge(left, right, kind))
    if not edges:
        # Preserve the topology path semantics when an export omits an
        # explicit edge list.  Branches are independent paths; do not connect
        # the last main-path node to the first branch node accidentally.
        paths: List[List[int]] = []
        main_path = [
            _node_id_from_row(item)
            for item in list(data.get("main_path", []) or [])
            if str(_dict(item).get("kind", "node")) == "node"
        ]
        if len(main_path) > 1:
            paths.append(main_path)
        for branch in list(data.get("branches", []) or []):
            branch_ids = [
                _node_id_from_row(item)
                for item in list(branch or [])
                if str(_dict(item).get("kind", "node")) == "node"
            ]
            if len(branch_ids) > 1:
                paths.append(branch_ids)
        if not paths and len(node_ids) > 1:
            paths = [list(node_ids)]
        seen: set[Tuple[int, int, str]] = set()
        for path in paths:
            for left, right in zip(path, path[1:]):
                if left < 0 or right < 0 or left == right:
                    continue
                key = (min(int(left), int(right)), max(int(left), int(right)), "internal")
                if key not in seen:
                    edges.append(UnitEdge(int(left), int(right)))
                    seen.add(key)
    return edges


def _normalize_ports(raw: Any, unit_id: str, node_ids: Sequence[int], phase: str, edges: Sequence[UnitEdge]) -> List[UnitPort]:
    data = _dict(raw)
    values = data.get("ports", data.get("open_ports", []))
    # Stage3 names open boundaries as mainline/branch slots.  A slot may
    # carry an explicit node ID, or a list of source/anchor IDs; retain the
    # first valid endpoint and preserve the slot identity for contract lookup.
    if not values:
        values = []
        for slot_index, slot in enumerate(
            list(data.get("mainline_slots", []) or [])
            + list(data.get("branch_slots", []) or [])
        ):
            row = _dict(slot)
            node_id = _int(
                row.get(
                    "node_id",
                    row.get(
                        "anchor_node_id",
                        row.get("source_node_id", (list(row.get("source_node_ids", []) or []) or [-1])[0]),
                    ),
                )
            )
            if node_id < 0:
                continue
            values.append({
                **row,
                "node_id": node_id,
                "port_id": row.get("port_id", row.get("slot_id", f"{unit_id}:slot:{slot_index}")),
                "role": row.get("role", row.get("target_role", row.get("slot_kind", "open"))),
                "phase": row.get("phase", phase),
                "contract_id": row.get("contract_id", row.get("slot_id", "")),
            })
    if isinstance(values, Mapping):
        values = list(values.values())
    ports: List[UnitPort] = []
    for index, value in enumerate(list(values or [])):
        row = _dict(value)
        node_id = _int(row.get("node_id", row.get("anchor_node_id", -1)))
        if node_id < 0:
            continue
        required = tuple(sorted({_int(v) for v in list(row.get("required_su_types", row.get("required_su_sequence", [])) or []) if _int(v) >= 0}))
        ports.append(UnitPort(
            port_id=str(row.get("port_id", row.get("slot_id", f"{unit_id}:port:{index}"))),
            node_id=node_id,
            role=str(row.get("role", row.get("anchor_role", "open"))),
            phase=str(row.get("phase", phase)),
            direction=_axial(row.get("direction", row.get("outward_direction"))),
            contract_id=str(row.get("contract_id", row.get("slot_id", ""))),
            required_su_types=required,
            metadata=tuple(sorted((str(k), str(v)) for k, v in row.items() if k not in {"node_id", "anchor_node_id", "port_id", "slot_id", "role", "anchor_role", "phase", "direction", "outward_direction", "contract_id", "required_su_types", "required_su_sequence"})),
        ))
    if ports:
        return ports
    degree: Dict[int, int] = {int(node_id): 0 for node_id in node_ids}
    for edge in edges:
        degree[int(edge.source_node_id)] = degree.get(int(edge.source_node_id), 0) + 1
        degree[int(edge.target_node_id)] = degree.get(int(edge.target_node_id), 0) + 1
    endpoint_ids = [int(node_id) for node_id in node_ids if degree.get(int(node_id), 0) <= 1]
    if len(endpoint_ids) == 0 and node_ids:
        endpoint_ids = [int(node_ids[0])]
    return [UnitPort(
        port_id=f"{unit_id}:port:{index}",
        node_id=node_id,
        role="open",
        phase=phase,
    ) for index, node_id in enumerate(endpoint_ids)]


def _normalize_unit(raw: Any, index: int, global_nodes: Dict[int, Dict[str, Any]], aromatic: bool = False) -> TopologyUnit:
    data = _dict(raw)
    kind = str(data.get("product_type", data.get("product_kind", data.get("kind", "aromatic_cluster" if aromatic else ""))) or "")
    unit_id = str(data.get("unit_id", data.get("cluster_id", data.get("id", f"{'aromatic' if aromatic else 'product'}:{index}"))))
    rows = _node_rows_for_unit(data, global_nodes, unit_id)
    if not rows:
        raise InverseInputError(f"inverse unit {unit_id!r} has no node geometry")
    if any(_int(row.get("su_type", -1)) < 0 for row in rows):
        raise InverseInputError(f"inverse unit {unit_id!r} has nodes without su_type")
    missing_geometry = [
        _int(row.get("node_id", -1))
        for row in rows
        if _axial(row.get("local_axial")) is None
    ]
    if missing_geometry:
        raise InverseInputError(
            f"inverse unit {unit_id!r} has nodes without local_axial geometry: {missing_geometry[:12]}"
        )
    nodes = tuple(UnitNode(
        node_id=_int(row["node_id"]),
        su_type=_int(row["su_type"]),
        local_axial=_axial(row.get("local_axial")) or (0, 0),
        role=str(row.get("role", "")),
        metadata=tuple(sorted((str(k), str(v)) for k, v in row.items() if k not in {"node_id", "su_type", "local_axial", "axial", "coord", "role"})),
    ) for row in rows)
    node_ids = tuple(node.node_id for node in nodes)
    edges = tuple(_normalize_edges(data, node_ids))
    ports = tuple(_normalize_ports(data, unit_id, node_ids, "aromatic" if aromatic else "side", edges))
    source_fragment_ids = _unique_ints(data.get("source_fragment_ids", []))
    source_node_ids = _unique_ints(data.get("source_node_ids", data.get("node_ids", node_ids)))
    rings_raw = data.get("rings", data.get("local_rings", []))
    rings: List[Tuple[Axial, ...]] = []
    for ring in list(rings_raw or []):
        points = [_axial(point) for point in list(ring or [])]
        points = [point for point in points if point is not None]
        if points:
            rings.append(tuple(points))
    return TopologyUnit(
        unit_id=unit_id,
        product_type=kind,
        nodes=nodes,
        internal_edges=edges,
        ports=ports,
        source_fragment_ids=source_fragment_ids,
        source_node_ids=source_node_ids,
        topology_notation=str(data.get("topology_notation", data.get("actual_topology_su", "")) or ""),
        node_id_notation=str(data.get("node_id_notation", data.get("actual_topology_gid", "")) or ""),
        closed=bool(data.get("closed", True)),
        local_rings=tuple(rings),
        metadata=tuple(sorted((str(k), str(v)) for k, v in data.items() if k not in {"unit_id", "cluster_id", "id", "product_type", "product_kind", "kind", "nodes", "sites", "local_nodes", "node_ids", "source_node_ids", "internal_edges", "edges", "ports", "open_ports", "rings", "local_rings", "topology_notation", "actual_topology_su", "node_id_notation", "actual_topology_gid", "closed", "source_fragment_ids"})),
    )


def _infer_contracts(payload: Any, units: Sequence[TopologyUnit], nodes: Dict[int, Dict[str, Any]]) -> List[ExternalContract]:
    root = _dict(payload)
    explicit = root.get("external_contracts", root.get("connection_contracts", []))
    if not explicit:
        explicit = dict(root.get("diagnostics", {}) or {}).get("external_contracts", [])
    if not explicit:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = _dict(root.get(key))
            explicit = nested.get("external_contracts", nested.get("connection_contracts", []))
            if explicit:
                break
    ownership = {int(node.node_id): unit.unit_id for unit in units for node in unit.nodes}
    ports_by_node = {
        int(port.node_id): (unit.unit_id, port.port_id)
        for unit in units
        for port in unit.ports
    }
    contracts: List[ExternalContract] = []
    if isinstance(explicit, Mapping):
        explicit = list(explicit.values())
    for index, value in enumerate(list(explicit or [])):
        row = _dict(value)
        source_node = _int(row.get("source_node_id", row.get("u", row.get("left_node_id", -1))))
        target_node = _int(row.get("target_node_id", row.get("v", row.get("right_node_id", -1))))
        source_unit = str(row.get("source_unit_id", ownership.get(source_node, "")))
        target_unit = str(row.get("target_unit_id", ownership.get(target_node, "")))
        if source_node < 0 or target_node < 0 or not source_unit or not target_unit or source_unit == target_unit:
            continue
        kind = str(row.get("kind", row.get("connection_kind", row.get("phase", "side"))))
        kind_lower = kind.lower()
        if "rigid" in kind_lower or kind_lower in {"aromatic", "aromatic_aromatic"}:
            kind = "rigid"
        elif "flex" in kind_lower or "ring" in kind_lower:
            kind = "flex"
        elif "side" in kind_lower:
            kind = "side"
        elif kind not in {"rigid", "flex", "side"}:
            kind = "side"
        contracts.append(ExternalContract(
            contract_id=str(row.get("contract_id", f"external:{index}")),
            source_unit_id=source_unit,
            source_node_id=source_node,
            target_unit_id=target_unit,
            target_node_id=target_node,
            kind=kind,
            mandatory=bool(row.get("mandatory", True)),
            fixed=bool(row.get("fixed", row.get("connection_semantics", "") == "fixed")),
            source_port_id=str(row.get("source_port_id", ports_by_node.get(source_node, ("", ""))[1])),
            target_port_id=str(row.get("target_port_id", ports_by_node.get(target_node, ("", ""))[1])),
        ))
    if contracts:
        return contracts
    seen: set[Tuple[int, int]] = set()
    for source_node, row in sorted(nodes.items()):
        for target_node in list(row.get("hop1_ids", []) or []):
            target_node = _int(target_node)
            pair = tuple(sorted((int(source_node), int(target_node))))
            if target_node < 0 or pair in seen or source_node not in ownership or target_node not in ownership:
                continue
            seen.add(pair)
            source_unit, target_unit = ownership[source_node], ownership[target_node]
            if source_unit == target_unit:
                continue
            source_phase = next(unit.phase for unit in units if unit.unit_id == source_unit)
            target_phase = next(unit.phase for unit in units if unit.unit_id == target_unit)
            if source_phase == target_phase == "aromatic":
                kind = "rigid"
            elif "flex" in {source_phase, target_phase}:
                kind = "flex"
            else:
                kind = "side"
            fixed_ids = {_int(value) for value in list(row.get("fixed_hop1_ids", []) or [])}
            target_row = nodes.get(target_node, {})
            fixed = target_node in fixed_ids or source_node in {_int(value) for value in list(target_row.get("fixed_hop1_ids", []) or [])}
            contracts.append(ExternalContract(
                contract_id=f"hop1:{pair[0]}-{pair[1]}",
                source_unit_id=source_unit,
                source_node_id=source_node,
                target_unit_id=target_unit,
                target_node_id=target_node,
                kind=kind,
                fixed=fixed,
                source_port_id=ports_by_node.get(source_node, ("", ""))[1],
                target_port_id=ports_by_node.get(target_node, ("", ""))[1],
            ))
    return contracts


def build_inverse_input(payload: Any) -> InverseAssemblyInput:
    """Normalize a complete inverse result for the new MCTS initializer."""
    if isinstance(payload, InverseAssemblyInput):
        return payload
    root = _dict(payload)
    global_nodes = _extract_nodes(payload)
    raw_aromatic = _extract_aromatic_units(payload)
    raw_products = _extract_products(payload)
    if not raw_aromatic:
        raise InverseInputError(
            "inverse result has no aromatic_units; export complete aromatic geometry "
            "with node_id/local_axial before starting RL_MTCS"
        )
    units: List[TopologyUnit] = []
    for index, raw in enumerate(raw_aromatic):
        units.append(_normalize_unit(raw, index, global_nodes, aromatic=True))
    for index, raw in enumerate(raw_products):
        units.append(_normalize_unit(raw, index, global_nodes, aromatic=False))
    unit_ids = [unit.unit_id for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        duplicates = sorted({unit_id for unit_id in unit_ids if unit_ids.count(unit_id) > 1})
        raise InverseInputError(f"inverse unit IDs are not unique: {duplicates[:12]}")
    by_node: Dict[int, str] = {}
    for unit in units:
        for node in unit.nodes:
            if node.node_id in by_node and by_node[node.node_id] != unit.unit_id:
                raise InverseInputError(
                    f"node {node.node_id} is claimed by both {by_node[node.node_id]!r} and {unit.unit_id!r}"
                )
            by_node[node.node_id] = unit.unit_id
    if global_nodes:
        missing = sorted(set(global_nodes) - set(by_node))
        ownership_ledger = dict(root.get("ownership_ledger", {}) or {})
        support = {
            _int(value)
            for value in list(ownership_ledger.get("support_context_node_ids", []) or [])
        }
        support.update(
            _int(value)
            for value in list(dict(ownership_ledger.get("states", {}) or {}).get("support_context", []) or [])
        )
        missing = [node_id for node_id in missing if node_id not in support]
        if missing:
            raise InverseInputError(f"inverse nodes are not covered by any complete unit: {missing[:12]}")
    contracts = _infer_contracts(payload, units, global_nodes)
    known_units = set(unit_ids)
    bad_contracts = [
        contract.contract_id
        for contract in contracts
        if contract.source_unit_id not in known_units or contract.target_unit_id not in known_units
    ]
    if bad_contracts:
        raise InverseInputError(
            f"inverse connection contracts reference unknown units: {bad_contracts[:12]}"
        )
    node_su = {
        int(node.node_id): int(node.su_type)
        for unit in units
        for node in unit.nodes
    }
    return InverseAssemblyInput(
        units=tuple(units),
        contracts=tuple(contracts),
        node_su_types=tuple(sorted(node_su.items())),
        ownership=tuple(sorted(by_node.items())),
        diagnostics=tuple(sorted((str(k), str(v)) for k, v in {
            "unit_count": len(units),
            "contract_count": len(contracts),
            "aromatic_unit_count": sum(1 for unit in units if unit.phase == "aromatic"),
        }.items())),
    )


def _load_inverse_payload(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    if path.suffix.lower() != ".json":
        raise InverseInputError(
            "initialize expects an inverse JSON export or the two final CSV files"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InverseInputError(f"cannot read inverse JSON export: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InverseInputError(f"invalid inverse JSON export: {path}") from exc


def _validate_inverse_completion(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key in ("ok", "topology_assembly_ok"):
        if key in payload and not bool(payload.get(key)):
            raise InverseInputError(
                f"inverse topology audit is not complete ({key}=False)"
            )
    pending = payload.get("pending_contracts") or payload.get("pending_units") or []
    if not pending:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = dict(payload.get(key, {}) or {})
            pending = nested.get("pending_contracts") or nested.get("pending_units") or []
            if pending:
                break
    if pending:
        raise InverseInputError(
            f"inverse topology has {len(list(pending))} pending units/contracts"
        )
    ownership = dict(payload.get("ownership_ledger", {}) or {})
    if not ownership:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = dict(payload.get(key, {}) or {})
            ownership = dict(nested.get("ownership_ledger", {}) or {})
            if ownership:
                break
    bad = list(dict(ownership.get("states", {}) or {}).get(
        "unclassified_or_conflict", []
    ) or [])
    if bad:
        raise InverseInputError(
            f"inverse ownership contains unclassified/conflicting nodes: {bad[:12]}"
        )


def _choose_anchor(problem: InverseAssemblyInput) -> Optional[str]:
    aromatic = [unit for unit in problem.units if unit.phase == "aromatic"]
    if not aromatic:
        return None
    return max(
        aromatic,
        key=lambda unit: (len(unit.nodes), len(unit.local_rings), unit.unit_id),
    ).unit_id


def _place_csv_aromatic_staging(
    state: HexAssemblyState,
    anchor_id: str,
) -> None:
    if not state.place_unit(anchor_id, rotation=0, translation=(0, 0)):
        raise InverseInputError(
            f"failed to place initial aromatic unit {anchor_id!r}"
        )
    cursor = 12
    for unit in sorted(
        (
            item for item in state.problem.units
            if item.phase == "aromatic" and item.unit_id != anchor_id
        ),
        key=lambda item: item.unit_id,
    ):
        placed = False
        for _attempt in range(2048):
            translation = (cursor, 0)
            cursor += 12
            if state.place_unit(unit.unit_id, rotation=0, translation=translation):
                placed = True
                break
        if not placed:
            raise InverseInputError(
                f"cannot create staging placement for {unit.unit_id!r}"
            )


def initialize(
    inverse_result: Any,
    topology_specs_csv: Any = None,
) -> Dict[str, Any]:
    """Build an immutable problem and its initial aromatic staging state."""
    csv_mode = (
        isinstance(inverse_result, (str, Path))
        and Path(inverse_result).suffix.lower() == ".csv"
    )
    if csv_mode:
        if topology_specs_csv is None:
            raise InverseInputError(
                "CSV/SU-histogram initialization requires both "
                "final_nodes_by_global_id.csv and final_topology_specs.csv"
            )
        problem = build_csv_inverse_input(inverse_result, topology_specs_csv)
    else:
        if topology_specs_csv is not None:
            raise InverseInputError(
                "topology_specs_csv is only valid with a node CSV input"
            )
        payload = _load_inverse_payload(inverse_result)
        _validate_inverse_completion(payload)
        problem = build_inverse_input(payload)

    state = HexAssemblyState(problem=problem)
    anchor_id = _choose_anchor(problem)
    if anchor_id is None:
        raise InverseInputError("inverse result contains no aromatic unit")
    if csv_mode:
        _place_csv_aromatic_staging(state, anchor_id)
    elif not state.place_unit(anchor_id, rotation=0, translation=(0, 0)):
        raise InverseInputError(
            f"failed to place initial aromatic unit {anchor_id!r}"
        )
    state.stage = "rigid"

    aromatic_units = [unit for unit in problem.units if unit.phase == "aromatic"]
    topology_units = [unit for unit in problem.units if unit.phase != "aromatic"]
    return {
        "problem": problem,
        "units": list(problem.units),
        "aromatic_units": aromatic_units,
        "topology_units": topology_units,
        "planned_topology": topology_units,
        "unit_registry": {unit.unit_id: unit for unit in problem.units},
        "contracts": list(problem.contracts),
        "ownership": problem.ownership_map(),
        "node_su_types": problem.node_map(),
        "current_node_su_types": state.current_node_map(),
        "anchor_unit_id": anchor_id,
        "state": state,
        "diagnostics": {
            "initializer": "inverse_csv_pair" if csv_mode else "inverse_topology_units",
            "allocation": False,
            "conversion": False,
            "substitution": False,
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "external_contract_count": len(problem.contracts),
            "problem_diagnostics": dict(problem.diagnostics),
        },
    }


def initialize_from_inverse(
    inverse_result: Any,
    topology_specs_csv: Any = None,
) -> Dict[str, Any]:
    return initialize(inverse_result, topology_specs_csv)


def _metadata(unit: TopologyUnit) -> Dict[str, str]:
    return {str(key): str(value) for key, value in unit.metadata}


def _count_rows(values: Iterable[Any], field_name: str) -> List[Dict[str, Any]]:
    counts = Counter(str(value) for value in values)
    return [
        {field_name: key, "count": int(value)}
        for key, value in sorted(counts.items(), key=lambda item: item[0])
    ]


def _su_distribution(node_su: Mapping[int, int], names: Mapping[int, str]) -> List[Dict[str, Any]]:
    counts = Counter(int(value) for value in node_su.values())
    return [
        {
            "su_type": int(su_type),
            "su_name": str(names.get(int(su_type), "")),
            "count": int(count),
        }
        for su_type, count in sorted(counts.items())
    ]


def _unit_su_counts(unit: TopologyUnit) -> Dict[str, int]:
    return {
        str(su_type): int(count)
        for su_type, count in sorted(Counter(int(node.su_type) for node in unit.nodes).items())
    }


def _bbox(unit: TopologyUnit) -> Dict[str, int]:
    q_values = [int(node.local_axial[0]) for node in unit.nodes]
    r_values = [int(node.local_axial[1]) for node in unit.nodes]
    return {
        "q_min": min(q_values),
        "q_max": max(q_values),
        "r_min": min(r_values),
        "r_max": max(r_values),
    }


def _contract_rows(contracts: Sequence[ExternalContract]) -> List[Dict[str, Any]]:
    return [
        {
            "contract_id": str(contract.contract_id),
            "kind": str(contract.kind),
            "source_unit_id": str(contract.source_unit_id),
            "source_node_id": int(contract.source_node_id),
            "source_port_id": str(contract.source_port_id),
            "target_unit_id": str(contract.target_unit_id),
            "target_node_id": int(contract.target_node_id),
            "target_port_id": str(contract.target_port_id),
            "mandatory": bool(contract.mandatory),
            "fixed": bool(contract.fixed),
        }
        for contract in contracts
    ]


def _external_boundaries(
    unit_id: str,
    contracts: Sequence[ExternalContract],
) -> List[int]:
    result: List[int] = []
    for contract in contracts:
        if contract.source_unit_id == unit_id:
            result.append(int(contract.target_node_id))
        elif contract.target_unit_id == unit_id:
            result.append(int(contract.source_node_id))
    return sorted(set(result))


def _aromatic_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: item.unit_id):
        port_counts = Counter(str(port.phase or "unspecified") for port in unit.ports)
        rows.append({
            "unit_id": str(unit.unit_id),
            "cluster_kind": str(unit.product_type),
            "ring_count": int(len(unit.local_rings)),
            "node_count": int(len(unit.nodes)),
            "internal_edge_count": int(len(unit.internal_edges)),
            "port_count": int(len(unit.ports)),
            "rigid_port_count": int(port_counts.get("rigid", 0)),
            "flex_port_count": int(port_counts.get("flex", 0)),
            "side_port_count": int(port_counts.get("side", 0)),
            "node_ids": [int(node.node_id) for node in unit.nodes],
            "external_boundary_ids": _external_boundaries(unit.unit_id, contracts),
            "su_distribution": _unit_su_counts(unit),
            "local_bbox": _bbox(unit),
        })
    return rows


def _topology_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: int(_metadata(item).get("csv_row", -1))):
        meta = _metadata(unit)
        related = [
            contract for contract in contracts
            if unit.unit_id in {contract.source_unit_id, contract.target_unit_id}
        ]
        rows.append({
            "csv_row": int(meta.get("csv_row", -1)),
            "unit_id": str(unit.unit_id),
            "source_product_type": str(meta.get("source_product_type", "")),
            "assembly_phase": str(unit.phase),
            "su_notation": str(unit.topology_notation),
            "gid_notation": str(unit.node_id_notation),
            "node_count": int(len(unit.nodes)),
            "internal_edge_count": int(len(unit.internal_edges)),
            "boundary_port_count": int(len(unit.ports)),
            "boundary_node_ids": _external_boundaries(unit.unit_id, contracts),
            "contract_ids": [str(contract.contract_id) for contract in related],
            "source_fragment_ids": [int(value) for value in unit.source_fragment_ids],
            "node_ids": [int(node.node_id) for node in unit.nodes],
            "su_distribution": _unit_su_counts(unit),
            "local_bbox": _bbox(unit),
        })
    return rows


def _su25_rows(
    units: Sequence[TopologyUnit],
    contracts: Sequence[ExternalContract],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in units:
        node_map = unit.node_map()
        for node in unit.nodes:
            if int(node.su_type) != 25:
                continue
            neighbors: List[Dict[str, Any]] = []
            directions = []
            for edge in unit.internal_edges:
                if int(node.node_id) not in {int(edge.source_node_id), int(edge.target_node_id)}:
                    continue
                other_id = (
                    int(edge.target_node_id)
                    if int(edge.source_node_id) == int(node.node_id)
                    else int(edge.source_node_id)
                )
                other = node_map[other_id]
                direction = (
                    int(other.local_axial[0]) - int(node.local_axial[0]),
                    int(other.local_axial[1]) - int(node.local_axial[1]),
                )
                directions.append(direction)
                neighbors.append({
                    "node_id": other_id,
                    "su_type": int(other.su_type),
                    "direction": [int(direction[0]), int(direction[1])],
                    "kind": "internal",
                })
            external = [
                contract for contract in contracts
                if (
                    contract.source_unit_id == unit.unit_id
                    and int(contract.source_node_id) == int(node.node_id)
                ) or (
                    contract.target_unit_id == unit.unit_id
                    and int(contract.target_node_id) == int(node.node_id)
                )
            ]
            rows.append({
                "unit_id": str(unit.unit_id),
                "node_id": int(node.node_id),
                "local_axial": [int(node.local_axial[0]), int(node.local_axial[1])],
                "degree": int(len(neighbors) + len(external)),
                "distinct_internal_direction_count": int(len(set(directions))),
                "neighbors": neighbors,
                "external_contract_ids": [str(item.contract_id) for item in external],
            })
    return rows


def _validate_units(
    units: Sequence[TopologyUnit],
    source_node_ids: set[int],
    ownership: Mapping[int, str],
    su25_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    errors: List[str] = []
    if set(int(value) for value in ownership) != source_node_ids:
        missing = sorted(source_node_ids - set(int(value) for value in ownership))
        unexpected = sorted(set(int(value) for value in ownership) - source_node_ids)
        errors.append(f"ownership_mismatch:missing={missing[:12]}:unexpected={unexpected[:12]}")
    for unit in units:
        coords = {int(node.node_id): tuple(node.local_axial) for node in unit.nodes}
        if len(set(coords.values())) != len(coords):
            errors.append(f"duplicate_local_axial:{unit.unit_id}")
        for edge in unit.internal_edges:
            left = coords[int(edge.source_node_id)]
            right = coords[int(edge.target_node_id)]
            delta = (int(right[0]) - int(left[0]), int(right[1]) - int(left[1]))
            if delta not in HEX_DIRECTIONS:
                errors.append(
                    f"non_hex_internal_edge:{unit.unit_id}:{edge.source_node_id}-{edge.target_node_id}"
                )
    for row in su25_rows:
        if int(row.get("degree", 0)) != 4:
            errors.append(f"su25_degree_not_four:{row.get('node_id')}:{row.get('degree')}")
        if int(row.get("distinct_internal_direction_count", 0)) + len(row.get("external_contract_ids", [])) != 4:
            errors.append(f"su25_connection_sites_not_four:{row.get('node_id')}")
    return errors


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key, "")) for key in fieldnames})


def inspect_csv_initialization(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Initialize the CSV pair, validate it, and return a serializable report."""
    nodes_path = Path(nodes_csv).resolve()
    topology_path = Path(topology_specs_csv).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else nodes_path.parent / "rl_mtcs_initialization"
    )

    node_rows = load_node_rows(nodes_path)
    source_topologies = load_topology_rows(topology_path)
    initialized = initialize(nodes_path, topology_path)
    problem = initialized["problem"]
    state = initialized["state"]
    aromatic_units = list(initialized["aromatic_units"])
    topology_units = list(initialized["topology_units"])
    contracts = list(problem.contracts)
    names = {int(row.su_type): str(row.su_name) for row in node_rows.values()}
    problem_diagnostics = {
        str(key): str(value) for key, value in problem.diagnostics
    }
    excluded_node_ids = sorted(
        int(value)
        for value in problem_diagnostics.get(
            "excluded_surplus_aromatic_node_ids", ""
        ).split(",")
        if value.strip()
    )

    aromatic_rows = _aromatic_rows(aromatic_units, contracts)
    topology_rows = _topology_rows(topology_units, contracts)
    contract_rows = _contract_rows(contracts)
    su25_rows = _su25_rows(topology_units, contracts)
    validation_errors = _validate_units(
        list(problem.units), set(problem.node_map()), problem.ownership_map(), su25_rows
    )
    initial_audit = state.audit()
    if initial_audit.get("duplicate_axials"):
        validation_errors.append("initial_aromatic_staging_has_duplicate_axials")
    if initial_audit.get("non_hex_edges"):
        validation_errors.append("initial_aromatic_staging_has_non_hex_edges")
    placed_initial_su = {
        int(node_id): int(state.current_su_types[node_id])
        for node_id in state.node_axial
    }
    if set(placed_initial_su.values()) - {12, 13}:
        validation_errors.append(
            "initial_aromatic_nodes_are_not_only_su12_su13"
        )
    for unit in aromatic_units:
        ring_coordinates = {
            tuple(coord) for ring in unit.local_rings for coord in ring
        }
        outside_ring = sorted(
            int(node.node_id)
            for node in unit.nodes
            if tuple(node.local_axial) not in ring_coordinates
        )
        if outside_ring:
            validation_errors.append(
                f"aromatic_nodes_outside_rings:{unit.unit_id}:{outside_ring}"
            )

    report: Dict[str, Any] = {
        "ok": not validation_errors,
        "inputs": {
            "nodes_csv": str(nodes_path),
            "topology_specs_csv": str(topology_path),
            "node_csv_row_count": len(node_rows),
            "topology_csv_row_count": len(source_topologies),
        },
        "summary": {
            "node_count": len(problem.node_su_types),
            "source_node_count": len(node_rows),
            "assembly_node_count": len(problem.node_su_types),
            "excluded_surplus_aromatic_node_ids": excluded_node_ids,
            "unit_count": len(problem.units),
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "contract_count": len(contracts),
            "initially_placed_unit_count": len(state.placements),
            "initially_placed_node_count": len(state.node_axial),
            "initial_component_count": state.component_count(),
            "anchor_unit_id": str(initialized["anchor_unit_id"]),
            "su25_count": len(su25_rows),
        },
        "distributions": {
            "su_types": _su_distribution(state.current_node_map(), names),
            "planned_su_types": _su_distribution(problem.node_map(), names),
            "initially_placed_aromatic_su_types": _su_distribution(
                placed_initial_su, names
            ),
            "aromatic_kinds": _count_rows(
                (unit.product_type for unit in aromatic_units), "cluster_kind"
            ),
            "topology_phases": _count_rows(
                (unit.phase for unit in topology_units), "assembly_phase"
            ),
            "source_product_types": _count_rows(
                (_metadata(unit).get("source_product_type", "") for unit in topology_units),
                "source_product_type",
            ),
            "topology_node_counts": _count_rows(
                (len(unit.nodes) for unit in topology_units), "node_count"
            ),
            "topology_boundary_counts": _count_rows(
                (len(unit.ports) for unit in topology_units), "boundary_port_count"
            ),
            "contract_kinds": _count_rows(
                (contract.kind for contract in contracts), "kind"
            ),
        },
        "aromatic_clusters": aromatic_rows,
        "topology_units": topology_rows,
        "contracts": contract_rows,
        "su25_sites": su25_rows,
        "initial_state_audit": initial_audit,
        "validation": {
            "ok": not validation_errors,
            "errors": validation_errors,
            "owned_node_count": len(problem.ownership),
            "source_node_count": len(node_rows),
            "assembly_node_count": len(problem.node_su_types),
            "excluded_surplus_aromatic_node_ids": excluded_node_ids,
        },
        "outputs": {
            "directory": str(directory),
            "report_json": str(directory / "initialization_report.json"),
            "su_distribution_csv": str(directory / "su_distribution.csv"),
            "aromatic_clusters_csv": str(directory / "aromatic_clusters.csv"),
            "topology_units_csv": str(directory / "topology_units.csv"),
            "contracts_csv": str(directory / "contracts.csv"),
            "su25_sites_csv": str(directory / "su25_sites.csv"),
        },
    }

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "initialization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(directory / "su_distribution.csv", report["distributions"]["su_types"])
    _write_csv(directory / "aromatic_clusters.csv", aromatic_rows)
    _write_csv(directory / "topology_units.csv", topology_rows)
    _write_csv(directory / "contracts.csv", contract_rows)
    _write_csv(directory / "su25_sites.csv", su25_rows)
    return report


def print_initialization_report(report: Mapping[str, Any]) -> None:
    """Print the high-signal distributions; full rows remain in CSV/JSON."""
    summary = dict(report.get("summary", {}))
    print("[RL_MTCS initialization]")
    print(f"ok={bool(report.get('ok', False))}")
    for key in (
        "source_node_count", "assembly_node_count",
        "excluded_surplus_aromatic_node_ids",
        "node_count", "unit_count", "aromatic_unit_count", "topology_unit_count",
        "contract_count", "initially_placed_unit_count", "initially_placed_node_count",
        "initial_component_count", "anchor_unit_id", "su25_count",
    ):
        print(f"{key}={summary.get(key)}")
    for name, rows in dict(report.get("distributions", {})).items():
        print(f"\n[{name}]")
        for row in rows:
            print("  " + ", ".join(f"{key}={value}" for key, value in row.items()))
    validation = dict(report.get("validation", {}))
    print("\n[validation]")
    print(f"ok={validation.get('ok')}")
    for error in validation.get("errors", []):
        print(f"  ERROR: {error}")
    print(f"\noutputs={dict(report.get('outputs', {})).get('directory', '')}")
__all__ = [
    "AROMATIC_SU_TYPES",
    "ParsedTopology",
    "build_csv_inverse_input",
    "build_inverse_input",
    "initialize",
    "initialize_from_inverse",
    "inspect_csv_initialization",
    "print_initialization_report",
    "load_node_rows",
    "load_topology_rows",
    "parse_topology_notation",
]
