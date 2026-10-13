"""Final SU26/SU30 substitution on a completed RL_MTCS assembly.

The placement stages deliberately work on a carbon-only aromatic staging
graph.  This module is the only place where heteroaromatic atoms are exposed:
six-membered requests are label substitutions, while five-membered requests
collapse two adjacent staging SU13 nodes into one final SU26/SU30 node.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from RL_MTCS.mdp.types import HeteroAromaticRequest, InverseAssemblyInput, InverseInputError
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.mdp.types import AssemblyEdge
from RL_MTCS.reporting.diagnostics import note_unit
from RL_MTCS.mdp.transition import relabel_aromatic_sites


def iter_substitution_outcomes(state):
    done = {a.get('request_target') for a in state.action_history if a.get('kind') == 'substitution'}
    for request in _request_rows(state):
        if request.target_node_id in done or request.target_node_id not in state.node_axial:
            continue
        note_unit(f'substitution:{request.target_node_id}')
        for row in _candidate_rows(state,request):
            child = apply_candidate(state,row)
            child.action_history[-1]['request_target'] = request.target_node_id
            child.stage = 'substitution'
            yield child


@dataclass(frozen=True)
class SubstitutionCandidate:
    target_su: int
    ring_size: int
    unit_id: str
    site_ids: Tuple[int, ...]
    neighbor_types: Tuple[int, ...] = ()
    representative_node_id: Optional[int] = None
    synthetic_node_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_su": int(self.target_su),
            "ring_size": int(self.ring_size),
            "unit_id": str(self.unit_id),
            "site_ids": list(self.site_ids),
            "neighbor_types": list(self.neighbor_types),
            "representative_node_id": self.representative_node_id,
            "synthetic_node_id": self.synthetic_node_id,
        }


def _edge_key(left: int, right: int) -> frozenset[int]:
    return frozenset((int(left), int(right)))


def _adjacency(unit: Any) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = {int(node.node_id): set() for node in unit.nodes}
    for edge in unit.internal_edges:
        left, right = int(edge.source_node_id), int(edge.target_node_id)
        result.setdefault(left, set()).add(right)
        result.setdefault(right, set()).add(left)
    return result


def _aromatic_units(state: HexAssemblyState) -> Sequence[Any]:
    return tuple(unit for unit in state.problem.units if unit.phase == "aromatic"
                 and unit.unit_id in state.placements)


def _pair_neighbor_types(unit: Any, left: int, right: int, su_map: Mapping[int, int]) -> Tuple[int, ...]:
    adjacency = _adjacency(unit)
    values: List[int] = []
    for node_id in (int(left), int(right)):
        for neighbor in sorted(adjacency.get(node_id, ())):
            if neighbor in {int(left), int(right)}:
                continue
            value = int(su_map.get(neighbor, -1))
            if 5 <= value <= 13:
                values.append(value)
    return tuple(sorted(values))


def _single_neighbor_types(unit: Any, node_id: int, su_map: Mapping[int, int]) -> Tuple[int, ...]:
    """Return aromatic SU labels around one staging site.

    These labels are retained in candidate diagnostics; candidate ordering
    does not score hop-1 similarity.
    """
    values = [
        int(su_map[neighbor])
        for neighbor in sorted(_adjacency(unit).get(int(node_id), ()))
        if neighbor in su_map and 5 <= int(su_map[neighbor]) <= 13
    ]
    return tuple(sorted(values))


def _request_rows(state: HexAssemblyState) -> List[HeteroAromaticRequest]:
    explicit = state.problem.hetero_requests
    if explicit:
        rows: List[HeteroAromaticRequest] = []
        node_units = {
            int(node.node_id): node
            for unit in state.problem.units if unit.phase == "aromatic"
            for node in unit.nodes
        }
        for request in explicit:
            target = int(request.target_node_id)
            node = node_units.get(target)
            if int(request.ring_size) == 5 and (
                request.synthetic_node_id is None
                or int(request.synthetic_node_id) not in state.node_axial
            ):
                continue
            # A complete inverse export may retain a request audit record while
            # already materializing SU26/SU30 in its aromatic unit.  Do not
            # attempt to substitute such a finalized node again.
            if int(request.ring_size) == 6 and node is not None and int(node.su_type) in {26, 30}:
                continue
            rows.append(request)
        return rows
    return []


def _candidate_rows(state: HexAssemblyState, request: HeteroAromaticRequest) -> List[SubstitutionCandidate]:
    su_map = state.current_node_map()
    reserved = {
        int(value)
        for other in _request_rows(state)
        for value in (other.target_node_id, other.synthetic_node_id)
        if value is not None
    }
    reserved -= {int(request.target_node_id)}
    if request.synthetic_node_id is not None:
        reserved.discard(int(request.synthetic_node_id))
    contract_nodes = {
        int(node_id)
        for contract in state.problem.contracts if contract.contract_id in state.satisfied_contracts
        for node_id in (contract.source_node_id, contract.target_node_id)
    }
    target_id = int(request.target_node_id)
    # Heteroaromatic atoms are interior ring sites.  Moving an endpoint that
    # already participates in an external contract would silently change the
    # searched topology, so such malformed requests are rejected.
    if target_id in contract_nodes:
        return []
    rows: List[SubstitutionCandidate] = []
    for unit in _aromatic_units(state):
        node_ids = {int(node.node_id) for node in unit.nodes}
        node_by_id = {int(node.node_id): node for node in unit.nodes}
        adjacency = _adjacency(unit)
        if int(request.ring_size) == 6:
            for node in unit.nodes:
                site_id = int(node.node_id)
                if int(node.su_type) != 13 or int(su_map.get(site_id, -1)) != 13:
                    continue
                if site_id in reserved:
                    continue
                if site_id in contract_nodes and site_id != target_id:
                    continue
                rows.append(SubstitutionCandidate(
                    int(request.target_su), 6, str(unit.unit_id), (site_id,),
                    _single_neighbor_types(unit, site_id, su_map),
                    int(target_id),
                ))
            continue
        # Five-membered ring: two adjacent staging carbons in one aromatic
        # unit.  Prefer the inverse target GID as the representative when it
        # occupies a staging SU13 slot; otherwise a synthetic target may be
        # represented by the first site of the pair.
        pairs: Set[Tuple[int, int]] = set()
        for left, neighbors in adjacency.items():
            for right in neighbors:
                pair = tuple(sorted((int(left), int(right))))
                if pair in pairs:
                    continue
                pairs.add(pair)
                if any(int(node_by_id.get(node_id).su_type) != 13 for node_id in pair if node_id in node_by_id):
                    continue
                if int(su_map.get(pair[0], -1)) != 13 or int(su_map.get(pair[1], -1)) != 13:
                    continue
                if any(value in reserved or (value in contract_nodes and value not in {target_id, int(request.synthetic_node_id or -1)}) for value in pair):
                    continue
                ordered = pair
                rows.append(SubstitutionCandidate(
                    int(request.target_su), 5, str(unit.unit_id), ordered,
                    _pair_neighbor_types(unit, pair[0], pair[1], su_map),
                    int(request.target_node_id),
                    int(request.synthetic_node_id) if request.synthetic_node_id is not None else None,
                ))
    return sorted(rows, key=lambda row: (row.unit_id, row.site_ids))


def _remap_problem(
    state: HexAssemblyState,
    unit_id: str,
    representative: int,
    removed: int,
    target_su: int,
) -> InverseAssemblyInput:
    units = []
    for unit in state.problem.units:
        if str(unit.unit_id) != str(unit_id):
            units.append(unit)
            continue
        nodes = []
        for node in unit.nodes:
            node_id = int(node.node_id)
            if node_id == removed:
                continue
            if node_id == representative:
                nodes.append(replace(node, su_type=int(target_su)))
            else:
                nodes.append(node)
        edges = []
        seen_edges: Set[Tuple[int, int, str]] = set()
        for edge in unit.internal_edges:
            left = representative if int(edge.source_node_id) == removed else int(edge.source_node_id)
            right = representative if int(edge.target_node_id) == removed else int(edge.target_node_id)
            if left == right:
                continue
            key = (min(left, right), max(left, right), str(edge.kind))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(replace(edge, source_node_id=left, target_node_id=right))
        ports = []
        for port in unit.ports:
            node_id = representative if int(port.node_id) == removed else int(port.node_id)
            if not any(int(node.node_id) == node_id for node in nodes):
                continue
            ports.append(replace(port, node_id=node_id))
        removed_coordinate = unit.node_map()[removed].local_axial
        rings = tuple(tuple(p for p in ring if p != removed_coordinate) for ring in unit.local_rings)
        units.append(replace(unit, nodes=tuple(nodes), internal_edges=tuple(edges), ports=tuple(ports),
                             local_rings=rings))

    node_su = state.problem.node_map()
    node_su.pop(int(removed), None)
    node_su[int(representative)] = int(target_su)
    ownership = state.problem.ownership_map()
    ownership.pop(int(removed), None)
    ownership[int(representative)] = str(unit_id)
    return replace(
        state.problem,
        units=tuple(units),
        node_su_types=tuple(sorted(node_su.items())),
        ownership=tuple(sorted(ownership.items())),
    )


def _rebind_substitution_sites(state: HexAssemblyState, mapping: Mapping[int, int]) -> HexAssemblyState:
    """Move logical identities without retargeting pending input contracts."""
    mapping = {int(old): int(new) for old, new in mapping.items()}
    if not mapping:
        return state.clone()
    affected = set(mapping) | set(mapping.values())
    if set(mapping) != set(mapping.values()) or len(mapping) != len(affected):
        raise InverseInputError("aromatic identity permutation is not bijective")
    if any(node_id not in state.node_axial for node_id in affected):
        raise InverseInputError("aromatic identity permutation contains an unplaced site")
    if all(old == new for old, new in mapping.items()):
        return state.clone()
    protected = {n for e in state.edges if e.contract_id
                 for n in (e.source_node_id, e.target_node_id)}
    return relabel_aromatic_sites(
        state, {new: old for old, new in mapping.items() if old != new},
        state.action_history, protected_aromatic_ids=protected,
        movable_hetero_ids=affected, replay_contract_kinds=('rigid', 'flex', 'ring', 'side'),
        stage='substitution', validate_geometry=False,
    )


def _pair_identity_mapping(left: int, right: int, target: int, synthetic: int) -> Dict[int, int]:
    """Build a bijection that assigns target/synthetic IDs to a physical pair."""
    desired = {int(left): int(target), int(right): int(synthetic)}
    affected = set(desired) | set(desired.values())
    available = set(affected)
    mapping: Dict[int, int] = {}
    for old, new in desired.items():
        mapping[int(old)] = int(new)
        available.discard(int(new))
    for old, new in zip(sorted(affected - set(mapping)), sorted(available)):
        mapping[int(old)] = int(new)
    return mapping


def apply_candidate(state: HexAssemblyState, candidate: SubstitutionCandidate) -> HexAssemblyState:
    """Apply one candidate transactionally and return a new state."""
    if candidate.ring_size not in {5, 6} or candidate.target_su not in {26, 30}:
        raise InverseInputError('unsupported substitution request')
    unit = state.unit(candidate.unit_id)
    if unit.phase != 'aromatic' or any(gid not in unit.node_map() for gid in candidate.site_ids):
        raise InverseInputError('substitution sites do not belong to the selected aromatic unit')
    if any(state.current_su_types.get(gid) != 13 for gid in candidate.site_ids):
        raise InverseInputError('substitution requires available staging SU13 sites')
    target = candidate.representative_node_id
    requests = [r for r in state.problem.hetero_requests
                if (r.target_node_id, r.target_su, r.ring_size, r.synthetic_node_id) ==
                (target, candidate.target_su, candidate.ring_size, candidate.synthetic_node_id)]
    if len(requests) != 1 or state.source_problem.node_map().get(target) != candidate.target_su:
        raise InverseInputError('substitution candidate does not match an original request')
    if len(candidate.site_ids) != (2 if candidate.ring_size == 5 else 1):
        raise InverseInputError('substitution site count mismatch')
    reserved = {gid for r in state.problem.hetero_requests if r.target_node_id != target
                for gid in (r.target_node_id, r.synthetic_node_id) if gid is not None}
    if reserved.intersection(candidate.site_ids):
        raise InverseInputError('substitution uses another request identity')
    adjacency = _adjacency(unit)
    if any(len(adjacency.get(gid, ())) != 2 for gid in candidate.site_ids):
        raise InverseInputError('substitution requires degree-two ring perimeter sites')
    moved = set(candidate.site_ids) | {target}
    if candidate.ring_size == 5:
        synthetic = candidate.synthetic_node_id
        if (synthetic is None or synthetic == target
                or state.source_problem.node_map().get(synthetic) != 13
                or any(synthetic in {c.source_node_id, c.target_node_id} for c in state.problem.contracts)
                or candidate.site_ids[1] not in adjacency.get(candidate.site_ids[0], ())):
            raise InverseInputError('five-member substitution requires its adjacent synthetic staging site')
        moved.add(synthetic)
    if any(moved.intersection((e.source_node_id, e.target_node_id)) for e in state.edges if e.contract_id):
        raise InverseInputError('substitution cannot move a committed external endpoint')
    if candidate.ring_size == 6:
        trial = state.clone()
        target = int(candidate.representative_node_id if candidate.representative_node_id is not None else candidate.site_ids[0])
        site = int(candidate.site_ids[0])
        if site != target:
            # Swap the selected physical site with the inverse target GID;
            # only uncontracted aromatic sites are admitted as destinations.
            trial = _rebind_substitution_sites(trial, {site: target, target: site})
        trial.current_su_types[target] = int(candidate.target_su)
        trial.action_history.append({
            "kind": "substitution", "target_su": int(candidate.target_su),
            "ring_size": 6, "site_ids": [target], "unit_id": candidate.unit_id,
            "request_target": target,
            "selected_physical_site_id": site,
        })
        return trial
    left, right = (int(value) for value in candidate.site_ids)
    representative = int(candidate.representative_node_id if candidate.representative_node_id is not None else left)
    synthetic = int(candidate.synthetic_node_id if candidate.synthetic_node_id is not None else right)
    if representative not in state.node_axial or synthetic not in state.node_axial:
        raise InverseInputError("substitution candidate contains an unplaced site")
    trial = state.clone()
    # Place the requested target and synthetic IDs at the selected physical
    # pair.  This permits NMR-guided selection among every eligible 13-13 edge
    # instead of forcing the pair chosen by initialization.
    trial = _rebind_substitution_sites(
        trial, _pair_identity_mapping(left, right, representative, synthetic)
    )
    representative, removed = int(candidate.representative_node_id if candidate.representative_node_id is not None else representative), synthetic
    unit = trial.unit(candidate.unit_id)
    if representative not in trial.node_axial or removed not in trial.node_axial:
        raise InverseInputError("substitution candidate permutation lost a site")
    trial.problem = _remap_problem(trial, candidate.unit_id, representative, removed, candidate.target_su)
    trial._units = {unit.unit_id: unit for unit in trial.problem.units}
    trial._ports = {f"{unit.unit_id}:{port.port_id}": port for unit in trial.problem.units for port in unit.ports}
    removed_axial = trial.node_axial[removed]
    removed_local = unit.node_map()[removed].local_axial
    trial.node_axial.pop(removed, None)
    if trial.occupied_axial.get(removed_axial) == removed:
        trial.occupied_axial.pop(removed_axial, None)
    left_pos = trial.final_node_positions.get(representative)
    if left_pos is None:
        axial = trial.node_axial[representative]
        left_pos = (float(axial[0]), float(axial[1]))
    right_pos = trial.final_node_positions.get(removed)
    if right_pos is None:
        right_pos = (float(removed_axial[0]), float(removed_axial[1]))
    trial.final_node_positions.pop(removed, None)
    mean_position = ((left_pos[0] + right_pos[0]) / 2.0, (left_pos[1] + right_pos[1]) / 2.0)
    if any(
        abs(float(position[0]) - mean_position[0]) < 1e-9
        and abs(float(position[1]) - mean_position[1]) < 1e-9
        for node_id, position in trial.final_node_positions.items()
        if int(node_id) != representative
    ):
        raise InverseInputError("five-member substitution creates a duplicate final coordinate")
    trial.final_node_positions[representative] = mean_position
    trial.current_su_types = {
        (representative if int(node_id) == removed else int(node_id)): int(su)
        for node_id, su in trial.current_su_types.items()
        if int(node_id) != removed
    }
    trial.current_su_types[representative] = int(candidate.target_su)
    rewritten: List[AssemblyEdge] = []
    seen: Set[Tuple[int, int, str, str]] = set()
    for edge in trial.edges:
        source = representative if int(edge.source_node_id) == removed else int(edge.source_node_id)
        target = representative if int(edge.target_node_id) == removed else int(edge.target_node_id)
        if source == target:
            continue
        key = (min(source, target), max(source, target), str(edge.kind), str(edge.contract_id))
        if key in seen:
            continue
        seen.add(key)
        direction = trial._edge_direction(trial.node_axial.get(source, (0, 0)), trial.node_axial.get(target, (0, 0)))
        rewritten.append(AssemblyEdge(source, target, edge.kind, edge.contract_id, direction))
        if representative in {source, target}:
            trial.substitution_relaxed_edges.add(_edge_key(source, target))
    trial.edges = rewritten
    trial.action_history.append({
        "kind": "substitution", "target_su": int(candidate.target_su), "ring_size": 5,
        "unit_id": candidate.unit_id,
        "request_target": representative,
        "affected_edge_pairs": sorted(sorted(pair) for pair in trial.substitution_relaxed_edges
                                      if representative in pair),
        "site_ids": [representative, removed], "selected_physical_site_ids": [left, right], "representative_node_id": representative,
        "removed_node_id": removed, "mean_position": list(trial.final_node_positions[representative]),
        "removed_local_axial": list(removed_local),
        "source_positions": [list(left_pos), list(right_pos)],
    })
    return trial
