"""Final SU26/SU30 substitution on a completed RL_MTCS assembly.

The placement stages deliberately work on a carbon-only aromatic staging
graph.  This module is the only place where heteroaromatic atoms are exposed:
six-membered requests are label substitutions, while five-membered requests
collapse two adjacent staging SU13 nodes into one final SU26/SU30 node.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .models import HeteroAromaticRequest, InverseAssemblyInput, InverseInputError
from .state import AssemblyEdge, HexAssemblyState


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
    return tuple(unit for unit in state.problem.units if unit.phase == "aromatic")


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

    ``hop1_ids`` in the inverse CSV is a list of global IDs, whereas the
    legacy substitutor compared the corresponding SU labels.  Candidate
    discovery therefore keeps the graph-ID layer and scoring uses this
    normalized SU-label layer.
    """
    values = [
        int(su_map[neighbor])
        for neighbor in sorted(_adjacency(unit).get(int(node_id), ()))
        if neighbor in su_map and 5 <= int(su_map[neighbor]) <= 13
    ]
    return tuple(sorted(values))


def _hop1_match(neighbor_types: Sequence[int], request_hop1: Sequence[int]) -> int:
    """Return a multiset mismatch count (zero means an exact containment)."""
    pool = Counter(int(value) for value in neighbor_types)
    wanted = Counter(int(value) for value in request_hop1 if 5 <= int(value) <= 13)
    return int(sum(max(0, count - pool.get(value, 0)) for value, count in wanted.items()))


def _request_hop1_types(state: HexAssemblyState, request: HeteroAromaticRequest) -> Tuple[int, ...]:
    """Resolve request hop-1 global IDs to SU labels.

    A few programmatic callers historically supplied SU labels directly, so
    values which are not present as node IDs are retained when they already
    look like aromatic SU types.
    """
    planned = state.problem.node_map()
    result: List[int] = []
    for value in request.hop1_ids:
        value = int(value)
        if value in planned:
            su = int(planned[value])
            if 5 <= su <= 13:
                result.append(su)
        elif 5 <= value <= 13:
            result.append(value)
    return tuple(result)


def _request_rows(state: HexAssemblyState) -> List[HeteroAromaticRequest]:
    explicit = tuple(getattr(state.problem, "hetero_requests", ()) or ())
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
    # CSV staging is the only supported runtime input and requires this final
    # reconstruction pass for SU26/SU30 requests.
    planned = state.problem.node_map()
    n26 = sum(value == 26 for value in planned.values())
    n30 = sum(value == 30 for value in planned.values())
    # Integer quota: floor(n26/3) is five-membered, with all SU30 assigned to
    # thiophene.  A request is tied to its inverse GID so six-member labels
    # remain stable in exports.
    n26_five = n26 // 3
    rows: List[HeteroAromaticRequest] = []
    for index, node_id in enumerate(sorted(node_id for node_id, su in planned.items() if su == 26)):
        rows.append(HeteroAromaticRequest(26, 5 if index < n26_five else 6, int(node_id)))
    rows.extend(
        HeteroAromaticRequest(30, 5, int(node_id))
        for node_id in sorted(node_id for node_id, su in planned.items() if su == 30)
    )
    return rows


def _candidate_rows(state: HexAssemblyState, request: HeteroAromaticRequest) -> List[SubstitutionCandidate]:
    su_map = state.current_node_map()
    # Side completion restores the planned 26/30 label before this stage.
    # Those atoms still occupy a carbon staging site, so treat the requested
    # target GID as SU13 while discovering a five-membered pair.
    if int(request.ring_size) in {5, 6}:
        su_map = dict(su_map)
        su_map[int(request.target_node_id)] = 13
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
        for contract in state.problem.contracts
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
        units.append(replace(unit, nodes=tuple(nodes), internal_edges=tuple(edges), ports=tuple(ports)))

    contracts = []
    for contract in state.problem.contracts:
        source = representative if int(contract.source_node_id) == removed else int(contract.source_node_id)
        target = representative if int(contract.target_node_id) == removed else int(contract.target_node_id)
        if source == target:
            continue
        contracts.append(replace(contract, source_node_id=source, target_node_id=target))
    node_su = state.problem.node_map()
    node_su.pop(int(removed), None)
    node_su[int(representative)] = int(target_su)
    ownership = state.problem.ownership_map()
    ownership.pop(int(removed), None)
    ownership[int(representative)] = str(unit_id)
    return replace(
        state.problem,
        units=tuple(units),
        contracts=tuple(contracts),
        node_su_types=tuple(sorted(node_su.items())),
        ownership=tuple(sorted(ownership.items())),
    )


def _permute_aromatic_ids(state: HexAssemblyState, mapping: Mapping[int, int]) -> None:
    """Permute aromatic GID identities between already placed physical slots."""
    mapping = {int(old): int(new) for old, new in mapping.items()}
    if not mapping:
        return
    affected = set(mapping) | set(mapping.values())
    if set(mapping) != set(mapping.values()) or len(mapping) != len(affected):
        raise InverseInputError("aromatic identity permutation is not bijective")
    if any(node_id not in state.node_axial for node_id in affected):
        raise InverseInputError("aromatic identity permutation contains an unplaced site")
    if all(old == new for old, new in mapping.items()):
        return

    def remap(value: int) -> int:
        return int(mapping.get(int(value), int(value)))

    units = []
    for unit in state.problem.units:
        if unit.phase != "aromatic":
            units.append(unit)
            continue
        units.append(replace(
            unit,
            nodes=tuple(replace(node, node_id=remap(node.node_id)) for node in unit.nodes),
            internal_edges=tuple(replace(edge, source_node_id=remap(edge.source_node_id), target_node_id=remap(edge.target_node_id)) for edge in unit.internal_edges),
            ports=tuple(replace(port, node_id=remap(port.node_id)) for port in unit.ports),
            source_node_ids=tuple(remap(node_id) for node_id in unit.source_node_ids),
        ))
    owner = {remap(node_id): unit_id for node_id, unit_id in state.problem.ownership_map().items()}
    contracts = tuple(replace(
        contract,
        source_node_id=remap(contract.source_node_id),
        target_node_id=remap(contract.target_node_id),
        source_unit_id=owner.get(remap(contract.source_node_id), contract.source_unit_id),
        target_unit_id=owner.get(remap(contract.target_node_id), contract.target_unit_id),
    ) for contract in state.problem.contracts)
    state.problem = replace(state.problem, units=tuple(units), contracts=contracts, ownership=tuple(sorted(owner.items())))
    state._units = {unit.unit_id: unit for unit in state.problem.units}
    state._ports = {f"{unit.unit_id}:{port.port_id}": port for unit in state.problem.units for port in unit.ports}
    state.node_axial = {remap(node_id): axial for node_id, axial in state.node_axial.items()}
    state.occupied_axial = {axial: remap(node_id) for axial, node_id in state.occupied_axial.items()}
    state.final_node_positions = {remap(node_id): position for node_id, position in state.final_node_positions.items()}
    # Restore assembly-time labels from the logical inverse GIDs.  The
    # permutation only changes which physical slot carries a GID; it must not
    # carry a hetero label over to another logical node.
    previous_current = dict(state.current_su_types)
    planned = state.problem.node_map()
    state.current_su_types = {
        remap(node_id): int(planned.get(remap(node_id), su)) if node_id in affected else int(su)
        for node_id, su in previous_current.items()
    }
    state.edges = [AssemblyEdge(
        remap(edge.source_node_id), remap(edge.target_node_id), edge.kind, edge.contract_id,
        state._edge_direction(state.node_axial[remap(edge.source_node_id)], state.node_axial[remap(edge.target_node_id)]),
    ) for edge in state.edges]
    state.allowed_unbonded_contacts = {frozenset(remap(node_id) for node_id in pair) for pair in state.allowed_unbonded_contacts}
    state.substitution_relaxed_edges = {frozenset(remap(node_id) for node_id in pair) for pair in state.substitution_relaxed_edges}


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
    if candidate.ring_size == 6:
        trial = state.clone()
        target = int(candidate.representative_node_id if candidate.representative_node_id is not None else candidate.site_ids[0])
        site = int(candidate.site_ids[0])
        if site != target:
            # Swap the selected physical site with the inverse target GID;
            # only uncontracted aromatic sites are admitted as destinations.
            _permute_aromatic_ids(trial, {site: target, target: site})
        trial.current_su_types[target] = int(candidate.target_su)
        trial.action_history.append({
            "kind": "substitution", "target_su": int(candidate.target_su),
            "ring_size": 6, "site_ids": [target],
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
    _permute_aromatic_ids(
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
        touched_removed = removed in {int(edge.source_node_id), int(edge.target_node_id)}
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
        if touched_removed:
            trial.substitution_relaxed_edges.add(_edge_key(source, target))
            if edge.contract_id:
                trial.substitution_relaxed_contracts.add(str(edge.contract_id))
    for contract in state.problem.contracts:
        if removed in {int(contract.source_node_id), int(contract.target_node_id)}:
            trial.substitution_relaxed_contracts.add(str(contract.contract_id))
    trial.edges = rewritten
    trial.allowed_unbonded_contacts = {
        frozenset(representative if int(node_id) == removed else int(node_id) for node_id in pair)
        for pair in trial.allowed_unbonded_contacts
        if removed not in pair or representative not in pair
    }
    trial.action_history.append({
        "kind": "substitution", "target_su": int(candidate.target_su), "ring_size": 5,
        "site_ids": [representative, removed], "selected_physical_site_ids": [left, right], "representative_node_id": representative,
        "removed_node_id": removed, "mean_position": list(trial.final_node_positions[representative]),
        "source_positions": [list(left_pos), list(right_pos)],
    })
    return trial


def run_substitution_stage(
    state: HexAssemblyState,
    *,
    nmr_scorer: Any = None,
    candidate_k: int = 32,
    beam_width: int = 4,
    nmr_weight: float = 1.0,
) -> Tuple[HexAssemblyState, Dict[str, Any]]:
    """Apply all final SU26/SU30 requests with a bounded beam.

    Each beam item is a complete state after the requests processed so far.
    This matters for five-membered substitutions because a pair consumes two
    staging sites and can make a later pair unavailable.  NMR scoring is
    performed in one batch per request layer whenever the scorer exposes the
    batch API; a small compatibility fallback keeps lightweight test scorers
    working as well.
    """
    requests = _request_rows(state)
    beam: List[Dict[str, Any]] = [{
        "state": state.clone(),
        "objective": 0.0,
        "mismatch": 0,
        "selected": [],
        "unresolved": [],
    }]
    candidate_records: List[Dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        expansions: List[Dict[str, Any]] = []
        for item in beam:
            current = item["state"]
            target_id = int(request.target_node_id)
            # Make the stage idempotent for callers which resume from a final
            # JSON/state export.
            if any(
                action.get("kind") == "substitution"
                and int(action.get("ring_size", 0)) == int(request.ring_size)
                and int(action.get("target_su", -1)) == int(request.target_su)
                and target_id in {int(value) for value in action.get("site_ids", [])}
                for action in current.action_history
            ):
                trial = current.clone()
                selected = list(item["selected"])
                selected.append({"request": request.__dict__, "mode": "already_substituted"})
                expansions.append({**item, "state": trial, "selected": selected})
                continue

            all_candidates = _candidate_rows(current, request)
            hop_types = _request_hop1_types(current, request)
            candidates = sorted(
                all_candidates,
                key=lambda value: (
                    _hop1_match(value.neighbor_types, hop_types),
                    value.unit_id, value.site_ids,
                ),
            )[: max(1, int(candidate_k))]
            if not candidates:
                unresolved = list(item["unresolved"])
                unresolved.append({"request": request.__dict__, "reason": "no_candidate"})
                expansions.append({
                    **item,
                    "unresolved": unresolved,
                    "mismatch": int(item["mismatch"]) + 1,
                    "objective": float(item["objective"]) + 1e6,
                })
                continue

            trials: List[HexAssemblyState] = []
            valid_candidates: List[SubstitutionCandidate] = []
            errors: List[Optional[str]] = []
            for candidate in candidates:
                try:
                    trials.append(apply_candidate(current, candidate))
                    valid_candidates.append(candidate)
                    errors.append(None)
                except Exception as exc:
                    errors.append(str(exc))

            if not valid_candidates:
                unresolved = list(item["unresolved"])
                unresolved.append({
                    "request": request.__dict__,
                    "reason": "candidate_application_failed",
                    "errors": [value for value in errors if value],
                })
                expansions.append({
                    **item,
                    "unresolved": unresolved,
                    "mismatch": int(item["mismatch"]) + 1,
                    "objective": float(item["objective"]) + 1e6,
                })
                continue

            scores: List[Dict[str, Any]] = []
            if trials and nmr_scorer is not None:
                try:
                    if hasattr(nmr_scorer, "score_candidates"):
                        scores = [dict(value) for value in nmr_scorer.score_candidates(
                            trials, stage="substitution", depth=int(request_index)
                        )]
                    else:
                        scores = [dict(nmr_scorer.score_state(value, stage="substitution")) for value in trials]
                except Exception as exc:
                    scores = [{"loss": float("inf"), "error": str(exc)} for _ in trials]
            else:
                scores = [{"loss": 0.0} for _ in trials]

            for index, candidate in enumerate(valid_candidates):
                score = scores[index] if index < len(scores) else {"loss": float("inf")}
                mismatch = _hop1_match(candidate.neighbor_types, hop_types)
                record = {
                    "request": request.__dict__,
                    "candidate": candidate.to_dict(),
                    "score": score,
                    "hop1_mismatch": int(mismatch),
                }
                candidate_records.append(record)
                selected = list(item["selected"])
                selected.append({"request": request.__dict__, "candidate": candidate.to_dict(), "score": score, "hop1_mismatch": int(mismatch)})
                expansions.append({
                    "state": trials[index],
                    "objective": float(item["objective"]) + float(nmr_weight) * float(score.get("loss", float("inf"))) + 1000.0 * float(mismatch),
                    "mismatch": int(item["mismatch"]) + int(mismatch),
                    "selected": selected,
                    "unresolved": list(item["unresolved"]),
                })
            # Invalid candidates should not silently erase an otherwise valid
            # beam.  They are recorded for diagnostics but are not expanded.
            for candidate, error in zip(candidates, errors):
                if error is not None:
                    candidate_records.append({
                        "request": request.__dict__,
                        "candidate": candidate.to_dict(),
                        "score": {"loss": float("inf"), "error": error},
                        "hop1_mismatch": None,
                    })
        if not expansions:
            break
        expansions.sort(key=lambda item: (
            len(item["unresolved"]), int(item["mismatch"]),
            float(item["objective"]),
            item["state"].state_signature(),
        ))
        beam = expansions[: max(1, int(beam_width))]

    best = min(
        beam,
        key=lambda item: (
            len(item["unresolved"]), int(item["mismatch"]),
            float(item["objective"]), item["state"].state_signature(),
        ),
    )
    current = best["state"]
    selected = best["selected"]
    unresolved = best["unresolved"]
    current.stage = "substitution"
    current.action_history.append({"kind": "substitution_stage", "request_count": len(requests), "selected_count": len(selected), "unresolved_count": len(unresolved), "beam_width": int(beam_width)})
    applied = sum(
        1 for row in selected
        if "candidate" in row
    )
    return current, {
        "requests": [request.__dict__ for request in requests],
        "selected": selected,
        "candidates": candidate_records,
        "unresolved": unresolved,
        "selected_count": len(selected),
        "applied_count": int(applied),
        "complete": not unresolved,
    }


__all__ = ["SubstitutionCandidate", "apply_candidate", "run_substitution_stage"]
