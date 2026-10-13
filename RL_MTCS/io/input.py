"""Build hex-grid units and fixed non-rigid topology from inverse CSV exports.

The node CSV supplies the final node identities/SU distribution and the
aromatic bridgehead evidence.  The topology CSV supplies the authoritative
flex/side trees.  In particular, topology edges are parsed from ``gid`` and
are never reconstructed from the older ``hop1_ids`` cache.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import ast
import csv
import hashlib
import json
import math
import itertools
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from model.shared.aromatic_plan import MAX_IDLE_AROMATIC_NODES, exact_aromatic_cluster_plan
from model.shared.topology_contract import (
    MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES,
    TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES,
    AROMATIC_EXTERNAL_SU_TYPES,
)

from RL_MTCS.geometry import (
    Axial,
    HEX_DIRECTIONS,
    HEX_VERTEX_OFFSETS,
    axial_distance,
    axial_to_cartesian,
    valid_hex_directions,
)
from RL_MTCS.mdp.types import (
    ExternalContract,
    InverseAssemblyInput,
    InverseInputError,
    TopologyUnit,
    UnitEdge,
    UnitNode,
    UnitPort,
    HeteroAromaticRequest,
    UnassembledClosureResource,
)
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.sites import aromatic_template_errors
from RL_MTCS.io.layout import _layout_topology_tree, _closure_cycle_nodes, _su25_center_branches
from RL_MTCS.io.output import contract_rows


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

@dataclass(frozen=True)
class NodeRow:
    node_id: int
    su_type: int
    su_name: str
    hop1_ids: Tuple[int, ...]
    template_key: str
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ParsedTopology:
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    main_path: Tuple[int, ...]
    branches: Tuple[Tuple[int, Tuple[int, ...]], ...]
    closure_edges: Tuple[Tuple[int, int], ...] = ()


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


def _csv_bool(value: object, *, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "y"}


def _load_resource_pool_rows(nodes_csv: str | Path) -> Dict[int, Dict[str, Any]]:
    """Load the inverse resource manifest colocated with the node CSV."""
    path = Path(nodes_csv).resolve().parent / "final_topology_resource_pool.csv"
    if not path.is_file():
        return {}
    rows = _csv_rows(path)
    result: Dict[int, Dict[str, Any]] = {}
    for raw in rows:
        if str(raw.get("node_id", "")).strip() == "":
            continue
        node_id = int(raw["node_id"])
        if node_id in result:
            raise InverseInputError(
                f"duplicate node_id in topology resource pool: {node_id}"
            )
        state = str(raw.get("state", "") or "")
        eligible = _csv_bool(
            raw.get("completion_tolerated", ""),
            default=False,
        )
        result[node_id] = {
            "node_id": node_id,
            "su_type": int(raw.get("su_type", -1)),
            "state": state,
            "completion_tolerated": bool(eligible),
            "tolerance_reason": str(raw.get("tolerance_reason", "") or ""),
        }
    return result


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
            metadata=tuple((key, str(raw.get(key, '') or '')) for key in
                ('special_anchor_mode', 'special_anchor_partition', 'target_hop1_degree', 'target_fixed_anchor_count')),
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
    """Parse a rooted topology, including ``@(gid)`` ring closures.

    Branch parentheses attach to the node immediately before ``(``.  GID
    notation contains only integer atoms, so labels in ``actual_su`` cannot
    accidentally create fake node IDs.  ``@(gid)`` adds an edge from the
    preceding node to an already declared node without declaring that node a
    second time.
    """
    source = re.sub(r"\s+", "", str(value or ""))
    tokens = re.findall(r"@\(\d+\)|\d+|[()\[\]-]", source)
    if "".join(tokens) != source:
        raise InverseInputError(
            f"unsupported token in topology notation: {value}"
        )
    nodes: List[int] = []
    edges: List[Tuple[int, int]] = []
    closure_edges: List[Tuple[int, int]] = []
    branches: List[Tuple[int, Tuple[int, ...]]] = []
    bracket_stack: List[int] = []
    pending_bridgehead: Optional[int] = None

    def parse_sequence(index: int, parent: Optional[int] = None) -> Tuple[int, List[int]]:
        nonlocal pending_bridgehead
        previous = parent
        path: List[int] = []
        while index < len(tokens):
            token = tokens[index]
            if token == ")":
                return index + 1, path
            if token == "[":
                if previous is None or bracket_stack:
                    raise InverseInputError("invalid fused ring bracket")
                bracket_stack.append(previous)
                index += 1
                continue
            if token == "]":
                if not bracket_stack:
                    raise InverseInputError("unmatched fused ring bracket")
                pending_bridgehead = bracket_stack.pop()
                index += 1
                continue
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
            if token.startswith("@("):
                if previous is None:
                    raise InverseInputError(
                        f"ring closure without source in topology notation: {value}"
                    )
                target_id = int(token[2:-1])
                if target_id not in nodes:
                    raise InverseInputError(
                        f"ring closure references undeclared GID {target_id}: {value}"
                    )
                if int(previous) == target_id:
                    raise InverseInputError(
                        f"ring closure self-edge at GID {target_id}: {value}"
                    )
                edge_key = tuple(sorted((int(previous), target_id)))
                if edge_key in {
                    tuple(sorted((int(left), int(right))))
                    for left, right in edges
                }:
                    raise InverseInputError(
                        f"duplicate topology edge {previous}-{target_id}: {value}"
                    )
                closure_edge = (int(previous), target_id)
                edges.append(closure_edge)
                closure_edges.append(closure_edge)
                index += 1
                continue
            node_id = int(token)
            if node_id in nodes:
                raise InverseInputError(
                    f"topology notation repeats GID {node_id}; use @({node_id}) for a ring closure: {value}"
                )
            nodes.append(node_id)
            path.append(node_id)
            if previous is not None:
                edges.append((int(previous), node_id))
            if pending_bridgehead is not None:
                shared = (pending_bridgehead, node_id)
                if pending_bridgehead == node_id or frozenset(shared) in {frozenset(e) for e in edges}:
                    raise InverseInputError("invalid fused ring shared edge")
                edges.append(shared)
                closure_edges.append(shared)
                pending_bridgehead = None
            previous = node_id
            index += 1
        return index, path

    _index, main_path = parse_sequence(0)
    if bracket_stack or pending_bridgehead is not None:
        raise InverseInputError("incomplete fused ring bracket")
    if not nodes:
        raise InverseInputError(f"empty topology GID notation: {value!r}")
    closure_keys = {tuple(sorted(edge)) for edge in closure_edges}
    edge_keys = {tuple(sorted(edge)) for edge in edges}
    ordinary_edge_count = len(edges) - len(closure_edges)
    if (
        len(edge_keys) != len(edges)
        or len(closure_keys) != len(closure_edges)
        or ordinary_edge_count != len(nodes) - 1
    ):
        raise InverseInputError(f"topology notation has an invalid rooted graph: {value}")
    return ParsedTopology(
        nodes=tuple(nodes),
        edges=tuple((int(left), int(right)) for left, right in edges),
        main_path=tuple(main_path),
        branches=tuple(branches),
        closure_edges=tuple(closure_edges),
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


class _AromaticPlanner:
    """Materialize one exact aromatic plan for CSV fallback inputs.

    The inverse pipeline normally writes ``final_aromatic_clusters.json`` and
    RL consumes that file directly.  Older CSV-only callers still need a
    deterministic fallback, but they must use the same exact planner rather
    than the removed heuristic/downgrade pass.
    """

    def __init__(self, counts: Mapping[int, int]):
        self.counts = {int(key): int(value) for key, value in dict(counts).items()}
        self.plan_diagnostics: Dict[str, object] = {}

    def plan(self) -> List[str]:
        bridgeheads = int(self.counts.get(12, 0))
        # The CSV contains final SU26/SU30 nodes, not the temporary duplicated
        # SU13 staging sites used while the inverse aromatic plan is built.
        # RL materialization therefore plans against actual CSV node count.
        staging_non_bridgeheads = int(sum(
            self.counts.get(su, 0)
            for su in (5, 6, 7, 8, 9, 10, 11, 13, 26, 30)
        ))
        preferred_kinds: List[str] = []
        exact_plan = exact_aromatic_cluster_plan(
            int(bridgeheads),
            int(staging_non_bridgeheads),
            preferred_kinds=preferred_kinds,
        )
        self.plan_diagnostics = dict(exact_plan)
        return [str(kind) for kind in exact_plan["kinds"]]


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


def _rigid_pairs(nodes: Mapping[int, NodeRow]) -> List[Tuple[int, int]]:
    pairs = {
        tuple(sorted((row.node_id, neighbor)))
        for row in nodes.values()
        if row.su_type == 10
        for neighbor in row.hop1_ids
        if neighbor != row.node_id and nodes[neighbor].su_type == 10
    }
    # These are source provenance only.  Assembly may choose different
    # partners, and a CSV with unpaired SU10 ports is still a valid search
    # input if its aromatic template exposes those ports.
    return sorted(pairs)


def _phase_for_product(product_type: str, boundary_count: int) -> str:
    kind = str(product_type or "").lower()
    if "ring" in kind:
        return "ring"
    if kind in {"flex_chain", "bridge", "chain"}:
        return "flex"
    if kind == "side_chain":
        return "side"
    raise InverseInputError(f"unsupported topology product_type: {product_type!r}")


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
        int(node_id) for node_id in aromatic_ids if nodes[node_id].su_type == 10
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

    # Source SU10 pairs are staging guidance for aromatic slot assignment;
    # the later rigid MCTS is free to choose different partners.
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

    for node_id in sorted(rigid_endpoint_ids - set(assignment)):
        allocate(node_id)

    # A rigid aromatic unit may expose two (or more) SU10 endpoints.  The
    # historical allocator picked the first available perimeter sites, which
    # can put every endpoint on the same side of a fused template.  Such a
    # placement has no collision-free embedding when the rigid graph is grown
    # as a chain.  Reassign endpoints within each template to well-separated
    # exterior slots before flex/side boundaries consume the remaining sites.
    if spread_flex_slots:
        for cluster_index, template in enumerate(templates):
            endpoint_nodes = sorted(
                int(node_id)
                for node_id, slot in assignment.items()
                if int(node_id) in rigid_endpoint_ids
                and int(slot[0]) == int(cluster_index)
            )
            if len(endpoint_nodes) < 2:
                continue
            available_slots = [
                slot for slot in exterior_slots
                if slot[0] == cluster_index
                and (slot not in used_slots or slot in {
                    assignment[node_id] for node_id in endpoint_nodes
                })
            ]
            if len(available_slots) < len(endpoint_nodes):
                continue
            old_slots = {assignment[node_id] for node_id in endpoint_nodes}
            # Do not maximize endpoint distance here.  For a linear aromatic
            # chain that policy deliberately chooses the two opposite sides
            # of every fused unit, so every rigid contract is propagated in
            # the same direction and the complete molecule becomes a rod.
            # Prefer a 120-degree angle between the *outward* port vectors;
            # after two ports are joined this produces a 60-degree turn in
            # the backbone while retaining the immutable aromatic topology.
            # Only the selected slot set affects the pairwise angle objective;
            # assign its sorted members to the stable endpoint order.  Using
            # combinations avoids factorial growth on higher-degree units.
            def slot_angle(left: Tuple[int, int], right: Tuple[int, int]) -> float:
                left_vec = _cart(_outward_direction(template, left[1]))
                right_vec = _cart(_outward_direction(template, right[1]))
                left_norm = math.hypot(*left_vec)
                right_norm = math.hypot(*right_vec)
                if left_norm <= 1e-12 or right_norm <= 1e-12:
                    return 180.0
                cosine = (
                    left_vec[0] * right_vec[0]
                    + left_vec[1] * right_vec[1]
                ) / (left_norm * right_norm)
                return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

            best: Optional[Tuple[Tuple[int, int], ...]] = None
            best_rank: Optional[Tuple[Any, ...]] = None
            for candidate in itertools.combinations(
                sorted(available_slots), len(endpoint_nodes)
            ):
                angles = [
                    slot_angle(candidate[left], candidate[right])
                    for left, right in itertools.combinations(
                        range(len(candidate)), 2
                    )
                ]
                angle_penalty = sum(
                    abs(angle - 120.0) for angle in angles
                )
                distances = [
                    axial_distance(
                        template.coords[candidate[left][1]],
                        template.coords[candidate[right][1]],
                    )
                    for left, right in itertools.combinations(
                        range(len(candidate)), 2
                    )
                ]
                # Angle is the primary objective.  Distance remains a weak
                # tie-breaker so two equally angled choices do not collapse
                # onto adjacent slots when a wider pair is available.
                rank = (
                    float(angle_penalty),
                    -float(min(distances) if distances else 0.0),
                    -float(sum(distances)),
                    tuple(candidate),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = tuple(candidate)
            if best is None:
                continue
            used_slots.difference_update(old_slots)
            for node_id, slot in zip(endpoint_nodes, best):
                assignment[node_id] = slot
                used_slots.add(slot)

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
    # This is topology ownership, not a runtime allocation: product identity,
    # cluster ownership and every
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
    if len(shared_nodes) != len(shared_slots):
        raise InverseInputError("SU12 count does not match fused aromatic bridgehead slots")
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


def _topology_assignment_info(
    topology_rows: Sequence[Mapping[str, Any]],
    aromatic_ids: Set[int],
    templates: Sequence[_ClusterTemplate],
) -> Dict[str, object]:
    boundary_by_product: List[List[int]] = []
    boundary_ids: Set[int] = set()
    phases: List[str] = []
    for raw in topology_rows:
        parsed = raw["parsed"]
        assert isinstance(parsed, ParsedTopology)
        degrees = Counter(node_id for edge in parsed.edges for node_id in edge)
        boundaries = [
            int(node_id)
            for node_id in parsed.nodes
            if int(node_id) in aromatic_ids and int(degrees[node_id]) == 1
        ]
        internal_aromatic = [
            int(node_id)
            for node_id in parsed.nodes
            if int(node_id) in aromatic_ids and int(degrees[node_id]) != 1
        ]
        if internal_aromatic:
            raise InverseInputError(
                f"topology row {raw['row_index']} contains non-boundary "
                f"aromatic GIDs: {internal_aromatic}"
            )
        boundary_by_product.append(boundaries)
        boundary_ids.update(boundaries)
        phases.append(
            _phase_for_product(
                str(raw.get("product_type", "")), len(boundaries)
            )
        )
    return {
        "boundary_ids": boundary_ids,
        "boundary_by_product": boundary_by_product,
        "phases": phases,
        "bridgehead_slot_count": int(sum(
            sum(bool(value) for value in template.shared)
            for template in templates
        )),
    }


def _load_authoritative_aromatic_plan(
    nodes_csv: str | Path,
) -> Optional[Dict[str, Any]]:
    path = Path(nodes_csv).resolve().parent / "final_aromatic_clusters.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InverseInputError(
            f"cannot read authoritative aromatic plan {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not bool(payload.get("complete", False)):
        raise InverseInputError(
            f"authoritative aromatic plan is incomplete: {path}"
        )
    if int(payload.get("schema_version", -1)) != 1:
        raise InverseInputError(
            "unsupported authoritative aromatic plan schema: "
            f"{payload.get('schema_version')!r}"
        )
    return dict(payload)


def _templates_from_authoritative_plan(
    plan: Mapping[str, Any],
    nodes: Mapping[int, NodeRow],
    aromatic_ids: Set[int],
) -> Tuple[List[_ClusterTemplate], Dict[int, Tuple[int, int]]]:
    clusters = sorted(
        [dict(item or {}) for item in list(plan.get("clusters", []) or [])],
        key=lambda item: int(item.get("cluster_id", -1)),
    )
    if not clusters:
        raise InverseInputError("authoritative aromatic plan has no clusters")
    templates: List[_ClusterTemplate] = []
    assignment: Dict[int, Tuple[int, int]] = {}
    planned_su_by_id: Dict[int, int] = {}
    for expected_index, cluster in enumerate(clusters):
        cluster_index = int(cluster.get("cluster_id", -1))
        if cluster_index != expected_index:
            raise InverseInputError(
                "authoritative aromatic cluster IDs must be contiguous: "
                f"expected={expected_index} actual={cluster_index}"
            )
        node_rows = sorted(
            [dict(item or {}) for item in list(cluster.get("nodes", []) or [])],
            key=lambda item: int(item.get("slot_index", -1)),
        )
        coords = [
            tuple(int(value) for value in list(item.get("local_axial", []) or []))
            for item in node_rows
        ]
        if any(len(coord) != 2 for coord in coords):
            raise InverseInputError(
                f"aromatic cluster {cluster_index} has invalid coordinates"
            )
        shared = [
            (
                str(item.get("template_bridgehead", False)).strip().lower()
                in {"1", "true", "yes"}
            )
            for item in node_rows
        ]
        edges = [
            tuple(int(value) for value in list(edge or []))
            for edge in list(cluster.get("edges_by_slot", []) or [])
        ]
        invalid_edges = [
            edge for edge in edges
            if len(edge) != 2
            or edge[0] == edge[1]
            or min(edge) < 0
            or max(edge) >= len(node_rows)
        ]
        if invalid_edges:
            raise InverseInputError(
                f"aromatic cluster {cluster_index} has invalid slot edges: "
                f"{invalid_edges[:6]}"
            )
        rings = [
            tuple(
                tuple(int(value) for value in list(coord or []))
                for coord in list(ring or [])
            )
            for ring in list(cluster.get("local_rings", []) or [])
        ]
        centers = [
            tuple(int(value) for value in list(center or []))
            for center in list(cluster.get("ring_centers", []) or [])
        ]
        if not centers:
            centers = [
                (
                    int(round(sum(coord[0] for coord in ring) / len(ring))),
                    int(round(sum(coord[1] for coord in ring) / len(ring))),
                )
                for ring in rings
                if ring
            ]
        template = _ClusterTemplate(
            kind=str(cluster.get("kind", "") or ""),
            centers=list(centers),
            coords=list(coords),
            shared=list(shared),
            edges=[tuple(sorted(edge)) for edge in edges],
            rings=list(rings),
        )
        if len(template.coords) != len(node_rows):
            raise InverseInputError(
                f"aromatic cluster {cluster_index} node/slot mismatch"
            )
        templates.append(template)
        for slot_index, item in enumerate(node_rows):
            if int(item.get("slot_index", -1)) != slot_index:
                raise InverseInputError(
                    f"aromatic cluster {cluster_index} has non-contiguous slots"
                )
            node_id = int(item.get("node_id", -1))
            if node_id in assignment:
                raise InverseInputError(
                    f"aromatic GID {node_id} occurs in more than one cluster"
                )
            if node_id not in nodes:
                raise InverseInputError(
                    f"aromatic plan references missing GID {node_id}"
                )
            planned_su = int(item.get("planned_su_type", nodes[node_id].su_type))
            if int(nodes[node_id].su_type) != planned_su:
                raise InverseInputError(
                    f"aromatic plan SU mismatch for GID {node_id}: "
                    f"plan={planned_su} nodes={nodes[node_id].su_type}"
                )
            staging_su = int(item.get(
                "staging_su_type", 12 if shared[slot_index] else 13
            ))
            expected_staging_su = 12 if shared[slot_index] else 13
            if staging_su != expected_staging_su:
                raise InverseInputError(
                    f"aromatic plan staging SU mismatch for GID {node_id}: "
                    f"staging={staging_su} bridgehead={shared[slot_index]}"
                )
            assignment[node_id] = (cluster_index, slot_index)
            planned_su_by_id[node_id] = planned_su

    if set(assignment) != set(aromatic_ids):
        raise InverseInputError(
            "authoritative aromatic plan node coverage mismatch: "
            f"missing={sorted(aromatic_ids-set(assignment))[:12]} "
            f"unexpected={sorted(set(assignment)-aromatic_ids)[:12]}"
        )
    planned_bridgeheads = {
        node_id
        for node_id, (cluster_index, slot_index) in assignment.items()
        if bool(templates[cluster_index].shared[slot_index])
    }
    actual_bridgeheads = {
        node_id for node_id in aromatic_ids if int(nodes[node_id].su_type) == 12
    }
    if planned_bridgeheads != actual_bridgeheads:
        raise InverseInputError(
            "authoritative aromatic bridgehead identity mismatch: "
            f"missing={sorted(actual_bridgeheads-planned_bridgeheads)[:12]} "
            f"unexpected={sorted(planned_bridgeheads-actual_bridgeheads)[:12]}"
        )
    if int(plan.get("cluster_count", len(templates))) != len(templates):
        raise InverseInputError("authoritative aromatic cluster count mismatch")
    ring_count = int(sum(len(template.rings) for template in templates))
    if int(plan.get("ring_count", ring_count)) != ring_count:
        raise InverseInputError("authoritative aromatic ring count mismatch")
    if int(plan.get("aromatic_node_count", len(assignment))) != len(assignment):
        raise InverseInputError("authoritative aromatic node count mismatch")
    signature_rows = [
        f"{node_id}:{planned_su_by_id[node_id]}"
        for node_id in sorted(planned_su_by_id)
    ]
    actual_signature = hashlib.sha256(
        "|".join(signature_rows).encode("utf-8")
    ).hexdigest()
    expected_signature = str(plan.get("node_signature", "") or "")
    if expected_signature and expected_signature != actual_signature:
        raise InverseInputError(
            "authoritative aromatic node signature mismatch"
        )
    return templates, assignment


def build_csv_inverse_input(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    *,
    use_authoritative_plan: bool = True,
) -> InverseAssemblyInput:
    nodes = load_node_rows(nodes_csv)
    original_node_count = len(nodes)
    resource_pool_rows = _load_resource_pool_rows(nodes_csv)
    topology_rows = load_topology_rows(topology_specs_csv)
    counts = Counter(row.su_type for row in nodes.values())
    n26 = sum(1 for row in nodes.values() if row.su_type == 26)
    n30 = sum(1 for row in nodes.values() if row.su_type == 30)
    n26_five = n26 // 3
    hetero_five_targets = [
        node_id for node_id, row in sorted(nodes.items()) if row.su_type == 26
    ][:n26_five]
    hetero_five_targets.extend(
        node_id for node_id, row in sorted(nodes.items()) if row.su_type == 30
    )
    synthetic_by_target: Dict[int, int] = {}
    next_synthetic = -1000000
    for target_id in hetero_five_targets:
        while next_synthetic in nodes:
            next_synthetic -= 1
        synthetic_by_target[int(target_id)] = int(next_synthetic)
        nodes[int(next_synthetic)] = NodeRow(
            node_id=int(next_synthetic), su_type=13, su_name="synthetic_staging_13",
            hop1_ids=(), template_key="",
        )
        next_synthetic -= 1
    counts[13] += len(synthetic_by_target)
    rigid_pairs = _rigid_pairs(nodes)
    source_aromatic_ids = {
        node_id for node_id, row in nodes.items()
        if row.su_type in AROMATIC_SU_TYPES
    }
    excluded_aromatic_ids: Set[int] = set()
    # Only uncommitted real SU13 may be left idle. In particular, do not
    # discard a rigid port or a temporary site reserved for substitution.
    protected_ids = {node_id for raw in topology_rows for node_id in raw["parsed"].nodes}
    protected_ids.update(hetero_five_targets)
    for node_id, row in nodes.items():
        if row.su_type in {26, 30}:
            protected_ids.update(row.hop1_ids)
    idle_candidates = {
        node_id for node_id in source_aromatic_ids
        if nodes[node_id].su_type == 13 and node_id >= 0
        and node_id not in protected_ids
    }
    authoritative_plan = (
        _load_authoritative_aromatic_plan(nodes_csv) if use_authoritative_plan else None
    )
    if authoritative_plan is not None:
        idle_rows = list(authoritative_plan.get("idle_aromatic_nodes", []))
        excluded_aromatic_ids = {int(row["node_id"]) for row in idle_rows}
        if (len(excluded_aromatic_ids) != len(idle_rows)
                or len(idle_rows) > MAX_IDLE_AROMATIC_NODES
                or not excluded_aromatic_ids <= idle_candidates
                or any(int(row["su_type"]) != nodes[int(row["node_id"])].su_type
                       for row in idle_rows)):
            raise InverseInputError("invalid idle aromatic node declaration")
        if int(authoritative_plan.get(
            "source_node_count", original_node_count
        )) != int(original_node_count):
            raise InverseInputError(
                "authoritative aromatic plan source node count mismatch"
            )
        if int(authoritative_plan.get(
            "assembly_node_count", len(nodes) - len(excluded_aromatic_ids)
        )) != int(len(nodes) - len(excluded_aromatic_ids)):
            raise InverseInputError(
                "authoritative aromatic plan assembly node count mismatch"
            )
        templates, assignment = _templates_from_authoritative_plan(
            authoritative_plan,
            nodes,
            source_aromatic_ids - excluded_aromatic_ids,
        )
        kinds = [str(template.kind) for template in templates]
        aromatic_ids = source_aromatic_ids - excluded_aromatic_ids
        assignment_info = _topology_assignment_info(
            topology_rows,
            aromatic_ids,
            templates,
        )
    else:
        kinds = _AromaticPlanner(counts).plan()
        if not kinds:
            raise InverseInputError("node CSV does not produce any aromatic cluster")
        templates = [_make_cluster_template(kind) for kind in kinds]
        ring_slot_count = sum(len(template.coords) for template in templates)
        surplus_count = len(source_aromatic_ids) - ring_slot_count
        if surplus_count < 0:
            raise InverseInputError(
                f"aromatic templates contain {ring_slot_count} ring sites for "
                f"{len(source_aromatic_ids)} inverse aromatic nodes"
            )
        if surplus_count:
            shared_slot_count = sum(
                sum(bool(value) for value in template.shared)
                for template in templates
            )
            if shared_slot_count != int(counts.get(12, 0)):
                raise InverseInputError(
                    "aromatic plan does not cover every SU12 bridgehead: "
                    f"nodes={int(counts.get(12, 0))} slots={shared_slot_count}"
                )
            if surplus_count > MAX_IDLE_AROMATIC_NODES or surplus_count > len(idle_candidates):
                raise InverseInputError(
                    f"aromatic_idle_capacity_exceeded: surplus={surplus_count}, "
                    f"eligible={len(idle_candidates)}, limit={MAX_IDLE_AROMATIC_NODES}"
                )
            excluded_aromatic_ids = set(sorted(idle_candidates)[-surplus_count:])

        aromatic_ids = source_aromatic_ids - excluded_aromatic_ids
        assignment, assignment_info = _assign_aromatic_nodes(
            nodes,
            topology_rows,
            templates,
            rigid_pairs,
            aromatic_ids=aromatic_ids,
        )

    # Stage each five-membered request as two adjacent, non-bridgehead SU13
    # sites.  The inverse CSV identifies the final target GID, but does not
    # identify a physical aromatic slot; this deterministic permutation makes
    # the pair available to the final substitution search while preserving
    # every GID and every unit's ownership.
    protected_ids = {int(node_id) for pair in rigid_pairs for node_id in pair}
    reserved_ids = set(int(value) for value in synthetic_by_target) | {
        int(value) for value in synthetic_by_target.values()
    }
    slot_to_node = {slot: int(node_id) for node_id, slot in assignment.items()}
    for target_id, synthetic_id in synthetic_by_target.items():
        target_id, synthetic_id = int(target_id), int(synthetic_id)
        target_slot = assignment.get(target_id)
        synthetic_slot = assignment.get(synthetic_id)
        if target_slot is None or synthetic_slot is None:
            raise InverseInputError(
                f"five-member staging IDs are not assigned: {target_id}, {synthetic_id}"
            )
        candidate_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for cluster_index, template in enumerate(templates):
            for left, right in template.edges:
                if template.shared[left] or template.shared[right]:
                    continue
                slot_left, slot_right = (cluster_index, left), (cluster_index, right)
                occupants = {
                    slot_to_node.get(slot_left, -1), slot_to_node.get(slot_right, -1)
                }
                if occupants & protected_ids:
                    continue
                if occupants & (reserved_ids - {target_id, synthetic_id}):
                    continue
                candidate_pairs.append((slot_left, slot_right))
        if not candidate_pairs:
            raise InverseInputError(
                f"no adjacent aromatic SU13 staging slots for five-member request {target_id}"
            )
        pair = min(
            candidate_pairs,
            key=lambda value: (
                -sum(slot in value for slot in (target_slot, synthetic_slot)),
                value[0][0], value[0][1], value[1][1],
            ),
        )
        slot_a, slot_b = pair
        if {
            target_slot, synthetic_slot
        } == {slot_a, slot_b}:
            continue
        if authoritative_plan is not None:
            raise InverseInputError(
                "authoritative aromatic plan does not place five-member "
                f"staging pair {target_id}/{synthetic_id} on adjacent "
                "non-bridgehead sites"
            )
        # Remove both requested IDs, capture the occupants displaced from the
        # destination pair, then refill the vacated source slots.
        old_slots = {target_slot, synthetic_slot}
        displaced = [
            slot_to_node[slot]
            for slot in (slot_a, slot_b)
            if slot_to_node.get(slot, -1) not in {target_id, synthetic_id}
        ]
        for slot, node_id in list(slot_to_node.items()):
            if node_id in {target_id, synthetic_id}:
                slot_to_node.pop(slot, None)
        slot_to_node[slot_a] = target_id
        slot_to_node[slot_b] = synthetic_id
        empty_slots = sorted(old_slots - {slot_a, slot_b})
        if len(empty_slots) != len(displaced):
            raise InverseInputError(
                f"invalid staging permutation for five-member request {target_id}"
            )
        for slot, node_id in zip(empty_slots, displaced):
            slot_to_node[slot] = node_id
        assignment = {int(node_id): slot for slot, node_id in slot_to_node.items()}
        protected_ids.update({target_id, synthetic_id})

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

    source_rigid_contracts = tuple(contracts)
    contracts.clear()
    su10_ids = sorted(node_id for node_id in source_aromatic_ids if nodes[node_id].su_type == 10)
    if len(su10_ids) % 2:
        raise InverseInputError("input_geometry_invalid: odd number of SU10 rigid ports")
    for node_id in su10_ids:
        cluster_index, slot_index = assignment[node_id]
        if templates[cluster_index].shared[slot_index]:
            raise InverseInputError(
                f"input_geometry_invalid: SU10 GID {node_id} is not an exterior site"
            )
        if _outward_direction(templates[cluster_index], slot_index) not in HEX_DIRECTIONS:
            raise InverseInputError(
                f"input_geometry_invalid: SU10 GID {node_id} has no hex port direction"
            )
        unit_id = f"aromatic:{cluster_index}"
        aromatic_port_requests[node_id].append(
            (-1, f"{unit_id}:rigid:{node_id}", "rigid")
        )

    su_by_id = {
        node_id: row.su_type
        for node_id, row in nodes.items()
        if node_id not in excluded_aromatic_ids
    }
    su25_nodes: Set[int] = set()
    input_issues = []
    for raw, boundaries, phase in zip(topology_rows, topology_boundaries, topology_phases):
        parsed = raw["parsed"]
        assert isinstance(parsed, ParsedTopology)
        source_product_type = str(raw.get("product_type", "")).strip().lower()
        attached_ring = source_product_type in {
            "aliphatic_ring", "fatty_ring", "fatty_ring_fused",
            "fatty_ring_side", "fatty_ring_upper_lower",
        }
        layout_graph = parsed
        if attached_ring:
            if len(boundaries) != 2:
                raise InverseInputError(
                    f"ring_label_geometry_mismatch: {raw['row_index']} requires two aromatic anchors"
                )
            # This temporary edge is the EXISTING aromatic edge the unit must
            # share. It constrains embedding, never adds a topology contract.
            aromatic_edge = tuple(boundaries)
            layout_graph = replace(parsed,
                edges=(*parsed.edges, aromatic_edge),
                closure_edges=(*parsed.closure_edges, aromatic_edge))
        labels = re.findall(r"(?:double|single)?\d+(?:d\d+)?", re.sub(
            r"@\(\d+\)", "", str(raw.get("actual_su", ""))))
        if len(labels) != len(parsed.nodes):
            raise InverseInputError(f"SU/GID node count mismatch in topology row {raw['row_index']}")
        label_by_id = dict(zip(parsed.nodes, labels))
        gids = iter(parsed.nodes)
        translated = re.sub(r"@\(\d+\)|(?:double|single)?\d+(?:d\d+)?",
            lambda m: m.group() if m.group().startswith('@') else str(next(gids)),
            str(raw.get('actual_su', '')))
        if parse_topology_notation(translated) != parsed:
            raise InverseInputError(f"SU/GID connectivity mismatch in topology row {raw['row_index']}")
        degree = Counter(node_id for edge in parsed.edges for node_id in edge)
        center_branches = _su25_center_branches(
            parsed, su_by_id, {gid: nodes[gid].hop1_ids for gid in parsed.nodes})
        center_parents = {parent for parent, _child in center_branches}
        unavailable = ''
        for gid, label in label_by_id.items():
            if int(re.search(r'\d+', label).group()) != su_by_id[gid]:
                raise InverseInputError(f"SU/GID type mismatch at GID {gid}")
            target = re.search(r'd(\d+)$', label)
            if target and int(target.group(1)) != degree[gid]:
                raise InverseInputError(f"topology target degree mismatch at GID {gid}")
            declared = dict(nodes[gid].metadata).get('target_hop1_degree')
            if gid not in boundaries and declared and int(declared) != degree[gid]:
                raise InverseInputError(f"CSV target degree mismatch at GID {gid}")
            declared_mode = dict(nodes[gid].metadata).get('special_anchor_mode')
            label_mode = re.match(r'(single|double)', label)
            if declared_mode and label_mode and declared_mode != label_mode.group(1):
                raise InverseInputError(f"CSV/topology anchor mode mismatch at GID {gid}")
            if degree[gid] > (4 if gid in center_parents else 3):
                unavailable = 'degree_exceeds_honeycomb_capacity'
        if unavailable:
            input_issues.append(json.dumps(dict(reason=unavailable,
                unit_id=f"topology:{raw['row_index']}", node_ids=list(parsed.nodes))))
            # These are provenance coordinates only; this resource never enters
            # placement. Keep its identities and edges in the original inventory.
            layout = {gid: (index, 0) for index, gid in enumerate(parsed.nodes)}
        else:
            layout = _layout_topology_tree(layout_graph, su_by_id, center_branches)
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
        closure_cycles = _closure_cycle_nodes(parsed) if parsed.closure_edges else ()
        local_rings = tuple(
            tuple(local[int(node_id)] for node_id in cycle)
            for cycle in closure_cycles
            if set(int(node_id) for node_id in cycle).issubset(internal_set)
        )
        internal_edges = tuple(
            UnitEdge(left, right, phase)
            for left, right in parsed.edges
            if left in internal_set and right in internal_set
        )
        for node_id in internal_ids:
            if su_by_id[node_id] == 25:
                su25_nodes.add(node_id)

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

        source_product_type = str(raw.get("product_type", "")).strip().lower()
        if source_product_type in {"flex_chain", "bridge", "chain"} and local_rings:
            raise InverseInputError(
                "ring_label_geometry_mismatch: topology row "
                f"{raw['row_index']} is labelled {source_product_type!r} "
                "but contains an explicit internal cycle"
            )
        if source_product_type in {
            "aliphatic_ring", "fatty_ring", "fatty_ring_fused",
            "fatty_ring_side", "fatty_ring_upper_lower",
        }:
            product_type = source_product_type
        elif parsed.closure_edges:
            product_type = "aliphatic_ring"
        else:
            product_type = f"{phase}_chain"
        ring_geometry_mode = (
            "internal_cycle"
            if local_rings
            else "external_contract_ring"
            if product_type in {
                "aliphatic_ring", "fatty_ring", "fatty_ring_fused",
                "fatty_ring_side", "fatty_ring_upper_lower",
            }
            else "tree"
        )
        topology_unit = TopologyUnit(
            unit_id=unit_id,
            product_type=product_type,
            nodes=tuple(UnitNode(
                node_id=node_id,
                su_type=su_by_id[node_id],
                local_axial=local[node_id],
                role="topology_node",
                metadata=(("source_product_type", str(raw.get("product_type", ""))),
                          ("su_label", label_by_id[node_id]), *nodes[node_id].metadata),
            ) for node_id in internal_ids),
            internal_edges=internal_edges,
            ports=tuple(ports),
            source_fragment_ids=tuple(raw["source_fragment_ids_parsed"]),
            source_node_ids=tuple(internal_ids),
            topology_notation=str(raw.get("actual_su", "")),
            node_id_notation=str(raw.get("gid", "")),
            closed=True,
            local_rings=local_rings,
            su25_center_branches=center_branches,
            metadata=(
                ("csv_row", str(raw["row_index"])),
                ("source_product_type", str(raw.get("product_type", ""))),
                ("ring_geometry_mode", ring_geometry_mode),
                ("aromatic_edge_attachment", "true" if attached_ring else "false"),
                ("geometry_source", "generated_from_connectivity"),
                ("assembly_unavailable_reason", unavailable),
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
                slot_id=f"{unit_id}:slot:{slot_index}",
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
    owned_ids = set(ownership)
    unexpected_ids = owned_ids - expected_ids
    if unexpected_ids:
        raise InverseInputError(
            "CSV initialization ownership mismatch; "
            f"missing={sorted(expected_ids-owned_ids)[:12]} "
            f"unexpected={sorted(unexpected_ids)[:12]}"
        )
    missing_ids = expected_ids - owned_ids
    declared_tolerated_ids = {
        int(node_id)
        for node_id, row in resource_pool_rows.items()
        if str(row.get("state", "")) == "remaining_closure_resource"
        and bool(row.get("completion_tolerated", False))
    }
    # The Stage3 resource ledger is a historical view.  A residual donor can
    # be re-used by the final aromatic assignment after that ledger was
    # written, so stale declarations must not override the final ownership
    # graph.  Only genuinely unowned nodes are completion-tolerated resources.
    unknown_declared = declared_tolerated_ids - set(nodes)
    # The final ownership graph is authoritative.  A node which was declared
    # as a residual in an older Stage3 manifest may have been consumed by the
    # final aromatic plan; that declaration is stale metadata, not an
    # ownership conflict.  Only IDs that are absent from the node CSV are
    # invalid declarations.
    if unknown_declared:
        raise InverseInputError(
            "CSV initialization ownership mismatch; invalid completion-tolerated "
            "resource manifest; "
            f"unknown={sorted(unknown_declared)[:12]}"
        )
    undeclared_missing = missing_ids - declared_tolerated_ids
    if undeclared_missing:
        raise InverseInputError(
            "CSV initialization ownership mismatch; unowned nodes are not "
            "declared as completion-tolerated resources; "
            f"missing={sorted(undeclared_missing)[:12]}"
        )
    if len(missing_ids) > MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES:
        raise InverseInputError(
            "too many completion-tolerated unassembled closure resources: "
            f"count={len(missing_ids)} "
            f"limit={MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES}"
        )
    unassembled_resources: List[UnassembledClosureResource] = []
    for node_id in sorted(missing_ids):
        node = nodes[int(node_id)]
        manifest = resource_pool_rows[int(node_id)]
        if int(manifest.get("su_type", -1)) != int(node.su_type):
            raise InverseInputError(
                f"resource-pool SU mismatch for GID {node_id}: "
                f"nodes={node.su_type} manifest={manifest.get('su_type')}"
            )
        if int(node.su_type) not in TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES:
            raise InverseInputError(
                "completion-tolerated unassembled resource has unsupported "
                f"SU type: GID={node_id} SU={node.su_type}"
            )
        unassembled_resources.append(UnassembledClosureResource(
            node_id=int(node_id),
            su_type=int(node.su_type),
            state=str(manifest.get("state", "")),
            hop1_ids=tuple(int(value) for value in node.hop1_ids),
            reason=str(
                manifest.get("tolerance_reason", "")
                or "stage3_tolerated_unassembled_closure_resource"
            ),
        ))
    if authoritative_plan is not None:
        expected_unassembled = {
            int(dict(row or {}).get("node_id", -1))
            for row in list(
                authoritative_plan.get("unassembled_closure_resources", []) or []
            )
        }
        if expected_unassembled and expected_unassembled != missing_ids:
            raise InverseInputError(
                "authoritative aromatic plan unassembled-resource mismatch; "
                f"plan={sorted(expected_unassembled)} current={sorted(missing_ids)}"
            )
    units = aromatic_units + topology_units
    unit_by_id = {str(unit.unit_id): unit for unit in units}
    node_owner = {
        int(node.node_id): str(unit.unit_id)
        for unit in units
        for node in unit.nodes
    }
    input_errors: List[str] = []
    if len(unit_by_id) != len(units):
        input_errors.append("duplicate_unit_id")
    contract_ids = [str(contract.contract_id) for contract in contracts]
    if len(set(contract_ids)) != len(contract_ids):
        input_errors.append("duplicate_contract_id")
    for contract in contracts:
        source_unit = unit_by_id.get(str(contract.source_unit_id))
        target_unit = unit_by_id.get(str(contract.target_unit_id))
        if source_unit is None or target_unit is None:
            input_errors.append(f"unknown_contract_unit:{contract.contract_id}")
            continue
        if node_owner.get(int(contract.source_node_id)) != str(source_unit.unit_id):
            input_errors.append(f"source_contract_node_ownership:{contract.contract_id}")
        if node_owner.get(int(contract.target_node_id)) != str(target_unit.unit_id):
            input_errors.append(f"target_contract_node_ownership:{contract.contract_id}")
        if contract.kind == "rigid" and not (
            source_unit.phase == target_unit.phase == "aromatic"
        ):
            input_errors.append(f"rigid_contract_phase:{contract.contract_id}")
        if contract.kind == "side" and not (
            {source_unit.phase, target_unit.phase} == {"aromatic", "side"}
        ):
            input_errors.append(f"side_contract_phase:{contract.contract_id}")
        if contract.kind in {"flex", "ring"} and not (
            {source_unit.phase, target_unit.phase} == {"aromatic", contract.kind}
        ):
            input_errors.append(f"flex_contract_phase:{contract.contract_id}")
    if input_errors:
        raise InverseInputError(
            "input_geometry_invalid: " + ", ".join(input_errors[:16])
        )
    contracts.sort(key=lambda item: ({"rigid": 0, "flex": 1, "ring": 2, "side": 3}[item.kind], item.contract_id))
    if authoritative_plan is not None:
        expected_contracts = list(
            authoritative_plan.get("aromatic_boundary_contracts", []) or []
        )
        # Earlier inverse exports called ring attachment contracts "flex".
        # Migrate only that label; still compare every identity and endpoint.
        expected_contracts = [dict(row) for row in expected_contracts]
        for row in expected_contracts:
            if row.get('kind') == 'flex' and any(
                unit_by_id.get(row.get(key)) is not None
                and unit_by_id[row[key]].phase == 'ring'
                for key in ('source_unit_id', 'target_unit_id')
            ):
                row['kind'] = 'ring'
                if row['contract_id'].startswith('flex:'):
                    row['contract_id'] = 'ring:' + row['contract_id'][5:]
        actual_contracts = contract_rows(sorted(
            (
                contract
                for contract in (*source_rigid_contracts, *contracts)
                if str(contract.source_unit_id).startswith("aromatic:")
                or str(contract.target_unit_id).startswith("aromatic:")
            ),
            key=lambda item: ({"rigid": 0, "flex": 1, "ring": 2, "side": 3}[item.kind], item.contract_id),
        ))
        if expected_contracts and sorted(expected_contracts, key=lambda row: row["contract_id"]) != sorted(
            actual_contracts, key=lambda row: row["contract_id"]
        ):
            raise InverseInputError(
                "authoritative aromatic boundary contract mismatch; the node, "
                "topology, and aromatic-plan files are not from one snapshot"
            )
    covered = {n for c in contracts for n in (c.source_node_id, c.target_node_id)}
    for gid, su in su_by_id.items():
        if su in AROMATIC_EXTERNAL_SU_TYPES and su != 10 and gid not in covered:
            input_issues.append(json.dumps(dict(reason='input_task_coverage_error',
                                               node_id=gid, planned_su_type=su)))
    return InverseAssemblyInput(
        units=tuple(units),
        contracts=tuple(contracts),
        input_issues=tuple(input_issues),
        node_su_types=tuple(sorted(
            (node_id, su_type)
            for node_id, su_type in su_by_id.items()
            if int(node_id) not in missing_ids
        )),
        ownership=tuple(sorted(ownership.items())),
        hetero_requests=tuple(
            HeteroAromaticRequest(
                target_su=int(nodes[target_id].su_type),
                ring_size=5,
                target_node_id=int(target_id),
                synthetic_node_id=int(synthetic_id),
                hop1_ids=tuple(nodes[target_id].hop1_ids),
            )
            for target_id, synthetic_id in sorted(synthetic_by_target.items())
        ) + tuple(
            HeteroAromaticRequest(26, 6, int(node_id), hop1_ids=tuple(row.hop1_ids))
            for node_id, row in sorted(nodes.items())
            if row.su_type == 26 and int(node_id) not in synthetic_by_target
        ) + tuple(
            HeteroAromaticRequest(30, 6, int(node_id), hop1_ids=tuple(row.hop1_ids))
            for node_id, row in sorted(nodes.items())
            if row.su_type == 30 and int(node_id) not in synthetic_by_target
        ),
        diagnostics=tuple(sorted((
            ("initializer", "inverse_csv_pair"),
            ("nodes_csv", str(Path(nodes_csv))),
            ("topology_specs_csv", str(Path(topology_specs_csv))),
            ("node_count", str(len(su_by_id))),
            ("source_node_count", str(original_node_count)),
            ("assembly_node_count", str(len(su_by_id))),
            ("graph_node_count", str(len(su_by_id) - len(missing_ids))),
            ("unassembled_closure_resource_count", str(len(missing_ids))),
            (
                "unassembled_closure_resource_ids",
                ",".join(str(value) for value in sorted(missing_ids)),
            ),
            (
                "excluded_surplus_aromatic_node_ids",
                ",".join(str(value) for value in sorted(excluded_aromatic_ids)),
            ),
            ("aromatic_unit_count", str(len(aromatic_units))),
            ("topology_unit_count", str(len(topology_units))),
            ("su26_total", str(n26)),
            ("su26_five_member_count", str(n26_five)),
            ("su26_six_member_count", str(n26 - n26_five)),
            ("su30_five_member_count", str(n30)),
            ("synthetic_staging_node_ids", ",".join(str(value) for value in sorted(synthetic_by_target.values()))),
            ("rigid_contract_count", str(len(source_rigid_contracts))),
            ("flex_contract_count", str(sum(c.kind == "flex" for c in contracts))),
            ("ring_contract_count", str(sum(c.kind == "ring" for c in contracts))),
            ("side_contract_count", str(sum(c.kind == "side" for c in contracts))),
            ("flex_chain_unit_count", str(sum(
                unit.phase == "flex" and dict(unit.metadata).get("source_product_type", "").lower()
                == "flex_chain" for unit in topology_units
            ))),
            ("aliphatic_ring_unit_count", str(sum(
                unit.phase == "ring" and dict(unit.metadata).get("source_product_type", "").lower()
                in {"aliphatic_ring", "fatty_ring", "fatty_ring_fused", "fatty_ring_side", "fatty_ring_upper_lower"}
                for unit in topology_units
            ))),
            ("internal_cycle_count", str(sum(
                dict(unit.metadata).get("ring_geometry_mode") == "internal_cycle"
                for unit in topology_units
            ))),
            ("external_contract_ring_count", str(sum(
                dict(unit.metadata).get("ring_geometry_mode") == "external_contract_ring"
                for unit in topology_units
            ))),
            ("su25_nodes", ",".join(str(value) for value in sorted(su25_nodes))),
        ))),
        unassembled_closure_resources=tuple(unassembled_resources),
        source_rigid_contracts=source_rigid_contracts,
        idle_aromatic_nodes=tuple((node_id, nodes[node_id].su_type)
                                 for node_id in sorted(excluded_aromatic_ids)),
    )


def serialize_aromatic_plan(
    problem: InverseAssemblyInput,
    *,
    generation_id: int = 0,
) -> Dict[str, Any]:
    """Serialize the exact aromatic assignment consumed by RL_MTCS.

    The payload records both the template staging identity (SU12/SU13) and
    the inverse node's planned final SU identity.  Synthetic negative GIDs
    are therefore explicit and reproducible for five-membered SU26/SU30
    substitutions instead of being reconstructed by a second planner.
    """

    aromatic_units = [unit for unit in problem.units if unit.phase == "aromatic"]
    errors = aromatic_template_errors(problem)
    if errors:
        raise InverseInputError(f"invalid aromatic template: {errors}")

    def cluster_index(unit: TopologyUnit) -> int:
        match = re.fullmatch(r"aromatic:(\d+)", str(unit.unit_id))
        if match is None:
            raise InverseInputError(
                f"aromatic unit has non-canonical ID: {unit.unit_id!r}"
            )
        return int(match.group(1))

    aromatic_units.sort(key=cluster_index)
    actual_indices = [cluster_index(unit) for unit in aromatic_units]
    if actual_indices != list(range(len(aromatic_units))):
        raise InverseInputError(
            "aromatic unit IDs must be contiguous before serialization: "
            f"{actual_indices}"
        )

    planned_su_by_id = problem.node_map()
    clusters: List[Dict[str, Any]] = []
    serialized_ids: Set[int] = set()
    for unit in aromatic_units:
        index = cluster_index(unit)
        node_to_slot = {
            int(node.node_id): int(slot_index)
            for slot_index, node in enumerate(unit.nodes)
        }
        if len(node_to_slot) != len(unit.nodes):
            raise InverseInputError(
                f"aromatic unit {unit.unit_id} contains duplicate node IDs"
            )
        node_rows: List[Dict[str, Any]] = []
        for slot_index, node in enumerate(unit.nodes):
            node_id = int(node.node_id)
            if node_id in serialized_ids:
                raise InverseInputError(
                    f"aromatic GID {node_id} belongs to more than one cluster"
                )
            serialized_ids.add(node_id)
            metadata = {str(key): str(value) for key, value in node.metadata}
            planned_su = int(metadata.get(
                "planned_su_type",
                planned_su_by_id.get(node_id, node.su_type),
            ))
            bridgehead_text = str(
                metadata.get("template_bridgehead", int(node.su_type) == 12)
            ).strip().lower()
            template_bridgehead = bridgehead_text in {"1", "true", "yes"}
            node_rows.append({
                "slot_index": int(slot_index),
                "node_id": node_id,
                "planned_su_type": planned_su,
                "staging_su_type": int(node.su_type),
                "local_axial": [
                    int(node.local_axial[0]), int(node.local_axial[1])
                ],
                "template_bridgehead": bool(template_bridgehead),
            })

        edges_by_slot: List[List[int]] = []
        internal_edges_by_gid: List[Dict[str, Any]] = []
        for edge in unit.internal_edges:
            left = int(edge.source_node_id)
            right = int(edge.target_node_id)
            if left not in node_to_slot or right not in node_to_slot:
                raise InverseInputError(
                    f"aromatic edge {left}-{right} escapes {unit.unit_id}"
                )
            edges_by_slot.append(sorted([
                int(node_to_slot[left]), int(node_to_slot[right])
            ]))
            internal_edges_by_gid.append({
                "source_node_id": left,
                "target_node_id": right,
                "kind": str(edge.kind),
            })
        edges_by_slot.sort(key=lambda edge: (edge[0], edge[1]))
        internal_edges_by_gid.sort(key=lambda edge: (
            min(edge["source_node_id"], edge["target_node_id"]),
            max(edge["source_node_id"], edge["target_node_id"]),
        ))

        local_rings = [
            [[int(coord[0]), int(coord[1])] for coord in ring]
            for ring in unit.local_rings
        ]
        ring_centers = [
            [
                int(round(sum(coord[0] for coord in ring) / len(ring))),
                int(round(sum(coord[1] for coord in ring) / len(ring))),
            ]
            for ring in unit.local_rings
            if ring
        ]
        ports = [{
            "port_id": str(port.port_id),
            "node_id": int(port.node_id),
            "role": str(port.role),
            "phase": str(port.phase),
            "direction": (
                [int(port.direction[0]), int(port.direction[1])]
                if port.direction is not None else None
            ),
            "contract_id": str(port.contract_id),
            "used": bool(port.used),
        } for port in unit.ports]
        ports.sort(key=lambda port: (
            port["phase"], port["node_id"], port["port_id"]
        ))
        clusters.append({
            "cluster_id": int(index),
            "unit_id": str(unit.unit_id),
            "kind": str(unit.product_type),
            "nodes": node_rows,
            "edges_by_slot": edges_by_slot,
            "internal_edges_by_gid": internal_edges_by_gid,
            "local_rings": local_rings,
            "ring_centers": ring_centers,
            "ports": ports,
        })

    aromatic_contracts = [
        contract
        for contract in problem.contracts
        if str(contract.source_unit_id).startswith("aromatic:")
        or str(contract.target_unit_id).startswith("aromatic:")
    ]
    rigid_contracts = contract_rows(problem.source_rigid_contracts)
    boundary_contracts = contract_rows([
        *problem.source_rigid_contracts, *aromatic_contracts
    ])
    hetero_requests = [{
        "target_su": int(request.target_su),
        "ring_size": int(request.ring_size),
        "target_node_id": int(request.target_node_id),
        "synthetic_node_id": (
            int(request.synthetic_node_id)
            if request.synthetic_node_id is not None else None
        ),
        "hop1_ids": [int(value) for value in request.hop1_ids],
    } for request in problem.hetero_requests]

    diagnostics = {str(key): str(value) for key, value in problem.diagnostics}
    source_node_count = int(diagnostics.get(
        "source_node_count", len(problem.node_su_types)
    ))
    assembly_node_count = int(diagnostics.get(
        "assembly_node_count", len(problem.node_su_types)
    ))
    signature_rows = [
        f"{node_id}:{planned_su_by_id.get(node_id, -1)}"
        for node_id in sorted(serialized_ids)
    ]
    node_signature = hashlib.sha256(
        "|".join(signature_rows).encode("utf-8")
    ).hexdigest()
    unassembled_resources = [{
        "node_id": int(resource.node_id),
        "su_type": int(resource.su_type),
        "state": str(resource.state),
        "hop1_ids": [int(value) for value in resource.hop1_ids],
        "reason": str(resource.reason),
    } for resource in problem.unassembled_closure_resources]
    idle_nodes = [{"node_id": node_id, "su_type": su_type,
                   "state": "idle_aromatic", "reason": "bounded_template_residual"}
                  for node_id, su_type in problem.idle_aromatic_nodes]
    return {
        "schema_version": 1,
        "complete": True,
        "generation_id": int(generation_id),
        "cluster_count": int(len(clusters)),
        "ring_count": int(sum(len(unit.local_rings) for unit in aromatic_units)),
        "source_node_count": source_node_count,
        "assembly_node_count": assembly_node_count,
        "aromatic_node_count": int(len(serialized_ids)),
        "node_signature": node_signature,
        "clusters": clusters,
        "rigid_contracts": rigid_contracts,
        "aromatic_boundary_contracts": boundary_contracts,
        "hetero_requests": hetero_requests,
        "unassembled_closure_resources": unassembled_resources,
        "unassembled_closure_resource_count": len(unassembled_resources),
        "idle_aromatic_nodes": idle_nodes,
        "idle_aromatic_count": len(idle_nodes),
        "completion_mode": (
            "bounded_idle_aromatic" if idle_nodes else
            "tolerated_unassembled_closure_resources"
            if unassembled_resources else "strict_all_nodes_assembled"
        ),
        "diagnostics": diagnostics,
    }


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
    # Keep staging components well separated.  They are temporary poses used
    # only to materialize every aromatic unit before the rigid search; a
    # spacing of 66 (a phase-preserving multiple of three) is comfortably larger than the largest supported cluster
    # footprint and prevents one component from masking another candidate.
    cursor = 66
    for unit in sorted(
        (
            item for item in state.problem.units
            if item.phase == "aromatic" and item.unit_id != anchor_id
        ),
        key=lambda item: int(str(item.unit_id).rsplit(":", 1)[-1]),
    ):
        if not state.place_unit(unit.unit_id, rotation=0, translation=(cursor, 0)):
            raise InverseInputError(
                f"cannot create staging placement for {unit.unit_id!r}"
            )
        cursor += 66


def initialize(
    inverse_result: Any,
    topology_specs_csv: Any = None,
) -> Dict[str, Any]:
    """Build an immutable problem from the authoritative inverse CSV pair."""
    csv_mode = (
        isinstance(inverse_result, (str, Path))
        and Path(inverse_result).suffix.lower() == ".csv"
    )
    if not csv_mode or topology_specs_csv is None:
        raise InverseInputError(
            "RL_MTCS requires final_nodes_by_global_id.csv and "
            "final_topology_specs.csv; JSON input is no longer supported"
        )
    problem = build_csv_inverse_input(inverse_result, topology_specs_csv)

    errors = aromatic_template_errors(problem)
    if errors:
        raise InverseInputError(
            "invalid aromatic template: " + str(errors),
            {"errors": errors, "source": "aromatic_template", "status": "input_geometry_invalid"},
        )

    state = HexAssemblyState(problem=problem)
    anchor_id = _choose_anchor(problem)
    if anchor_id is None:
        raise InverseInputError("inverse result contains no aromatic unit")
    _place_csv_aromatic_staging(state, anchor_id)
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
        "unassembled_closure_resources": list(
            problem.unassembled_closure_resources
        ),
        "idle_aromatic_nodes": list(problem.idle_aromatic_nodes),
        "current_node_su_types": state.current_node_map(),
        "anchor_unit_id": anchor_id,
        "state": state,
        "diagnostics": {
            "input_errors": [json.loads(issue) for issue in problem.input_issues],
            "initializer": "inverse_csv_pair",
            "conversion": False,
            "substitution": bool(problem.hetero_requests),
            "hetero_request_count": len(problem.hetero_requests),
            "unassembled_closure_resource_count": len(
                problem.unassembled_closure_resources
            ),
            "completion_mode": (
                "bounded_idle_aromatic" if problem.idle_aromatic_nodes else
                "tolerated_unassembled_closure_resources"
                if problem.unassembled_closure_resources
                else "strict_all_nodes_assembled"
            ),
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "external_contract_count": len(problem.contracts),
            "problem_diagnostics": dict(problem.diagnostics),
        },
    }


__all__ = [
    "AROMATIC_SU_TYPES",
    "ParsedTopology",
    "build_csv_inverse_input",
    "serialize_aromatic_plan",
    "initialize",
    "load_node_rows",
    "load_topology_rows",
    "parse_topology_notation",
]
