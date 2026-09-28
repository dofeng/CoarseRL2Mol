"""MCTS placement of complete inverse topology units on a hexagonal lattice."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import (
    HEX_DIRECTIONS,
    axial_add,
    axial_distance,
    axial_neg,
    aromatic_slot_direction,
    coordinate_aspect,
    layout_metrics,
    transform_axial,
    unit_sort_key,
)
from .state import AssemblyEdge, HexAssemblyState, UnitPlacement
from .models import ExternalContract, TopologyUnit, UnitPort


def _action_key(action: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in action.items() if key not in {"score"}))


def _port_for_node(state: HexAssemblyState, unit: TopologyUnit, node_id: int, preferred: str = "") -> Optional[UnitPort]:
    if preferred:
        for port in unit.ports:
            if str(port.port_id) == str(preferred) and int(port.node_id) == int(node_id):
                return port
    return next((port for port in unit.ports if int(port.node_id) == int(node_id)), None)


def _port_compatible(source: UnitPort, target: UnitPort, source_su: int, target_su: int) -> bool:
    if source.required_su_types and int(target_su) not in set(source.required_su_types):
        return False
    if target.required_su_types and int(source_su) not in set(target.required_su_types):
        return False
    return True


def _unit_matches_search_phase(unit: TopologyUnit, phase: str) -> bool:
    if str(phase) == "rigid":
        return unit.phase == "aromatic"
    return unit.phase == str(phase)


class TopologyPlacementStage:
    """Stage adapter consumed by the generic MCTS search."""

    def __init__(
        self,
        state: HexAssemblyState,
        phase: str,
        contract_order: Optional[Sequence[str]] = None,
    ):
        self.state = state
        self.phase = str(phase)
        self.contract_order = (
            tuple(str(value) for value in contract_order)
            if contract_order is not None
            else None
        )
        self.state.stage = str(phase)

    def clone(self) -> "TopologyPlacementStage":
        return TopologyPlacementStage(
            self.state.clone(), self.phase, self.contract_order
        )

    def is_done(self) -> bool:
        if self.contract_order is not None:
            return all(
                contract_id in self.state.satisfied_contracts
                for contract_id in self.contract_order
            )
        return bool(self.state.phase_done(self.phase))

    def _contract_candidates(self) -> List[ExternalContract]:
        pending = self.state.mandatory_contracts(self.phase)
        if self.contract_order is None:
            return list(pending)
        by_id = {str(contract.contract_id): contract for contract in pending}
        for contract_id in self.contract_order:
            if contract_id in by_id:
                return [by_id[contract_id]]
        return []

    def _candidate_for_pair(
        self,
        source_unit: TopologyUnit,
        source_port: UnitPort,
        target_unit: TopologyUnit,
        target_port: UnitPort,
        *,
        contract: Optional[ExternalContract] = None,
    ) -> List[Dict[str, Any]]:
        if source_unit.unit_id == target_unit.unit_id:
            return []
        if not _unit_matches_search_phase(source_unit, self.phase) and not _unit_matches_search_phase(target_unit, self.phase):
            return []
        if source_unit.unit_id in self.state.placements and target_unit.unit_id in self.state.placements:
            source_axial = self.state.node_axial.get(int(source_port.node_id))
            target_axial = self.state.node_axial.get(int(target_port.node_id))
            if source_axial is None or target_axial is None:
                return []
            if axial_distance(source_axial, target_axial) != 1:
                if target_unit.unit_id in self.state.component_unit_ids(source_unit.unit_id):
                    return []
                actions: List[Dict[str, Any]] = []
                actions.extend(self._component_transform_candidates(
                    source_unit,
                    source_port,
                    target_unit,
                    target_port,
                    contract=contract,
                ))
                actions.extend(self._component_transform_candidates(
                    target_unit,
                    target_port,
                    source_unit,
                    source_port,
                    contract=contract,
                ))
                return actions
            return [self._make_action(source_unit, source_port, target_unit, target_port, contract=contract)]

        if source_unit.unit_id in self.state.placements:
            source_unit, source_port, target_unit, target_port = target_unit, target_port, source_unit, source_port
        if target_unit.unit_id not in self.state.placements or source_unit.unit_id in self.state.placements:
            return []
        target_axial = self.state.node_axial.get(int(target_port.node_id))
        if target_axial is None:
            return []
        source_su = source_unit.node_map().get(int(source_port.node_id))
        target_su_node = target_unit.node_map().get(int(target_port.node_id))
        if source_su is None or target_su_node is None:
            return []
        if not _port_compatible(source_port, target_port, int(source_su.su_type), int(target_su_node.su_type)):
            return []

        actions: List[Dict[str, Any]] = []
        target_direction = self.state.global_port_direction(target_unit.unit_id, target_port)
        directions = [target_direction] if target_direction is not None else list(HEX_DIRECTIONS)
        for reflection in (False, True):
            for rotation in range(6):
                source_direction = self.state.global_port_direction(source_unit.unit_id, source_port, rotation=rotation, reflection=reflection)
                for direction in directions:
                    if direction is None:
                        continue
                    direction = tuple(int(value) for value in direction)
                    if source_direction is not None and source_direction != axial_neg(direction):
                        continue
                    source_target = axial_add(target_axial, direction)
                    source_local = self.state.global_node_axial(
                        source_unit.unit_id,
                        int(source_port.node_id),
                        rotation=rotation,
                        translation=(0, 0),
                        reflection=reflection,
                    )
                    translation = (source_target[0] - source_local[0], source_target[1] - source_local[1])
                    ok, _reason = self.state.can_place(
                        source_unit.unit_id,
                        rotation,
                        translation,
                        allowed_contact=(int(source_port.node_id), int(target_port.node_id)),
                        reflection=reflection,
                    )
                    if not ok:
                        continue
                    action = self._make_action(
                        source_unit,
                        source_port,
                        target_unit,
                        target_port,
                        contract=contract,
                        source_rotation=rotation,
                        source_translation=translation,
                    )
                    action["source_reflection"] = bool(reflection)
                    actions.append(action)
        return actions

    def _component_transform_candidates(
        self,
        movable_unit: TopologyUnit,
        movable_port: UnitPort,
        stationary_unit: TopologyUnit,
        stationary_port: UnitPort,
        *,
        contract: Optional[ExternalContract],
    ) -> List[Dict[str, Any]]:
        movable_axial = self.state.node_axial.get(int(movable_port.node_id))
        stationary_axial = self.state.node_axial.get(int(stationary_port.node_id))
        if movable_axial is None or stationary_axial is None:
            return []
        stationary_direction = self.state.global_port_direction(
            stationary_unit.unit_id, stationary_port
        )
        movable_direction = self.state.global_port_direction(
            movable_unit.unit_id, movable_port
        )
        directions = [stationary_direction] if stationary_direction is not None else list(HEX_DIRECTIONS)
        component_nodes = self.state.component_node_ids(movable_unit.unit_id)
        actions: List[Dict[str, Any]] = []
        for reflection in (False, True):
            for rotation in range(6):
                transformed_direction = (
                    transform_axial(movable_direction, rotation, reflection)
                    if movable_direction is not None else None
                )
                rotated_axial = transform_axial(
                    movable_axial, rotation, reflection
                )
                for direction in directions:
                    if direction is None:
                        continue
                    direction = tuple(int(value) for value in direction)
                    if transformed_direction is not None and transformed_direction != axial_neg(direction):
                        continue
                    desired = axial_add(stationary_axial, direction)
                    translation = (
                        desired[0] - rotated_axial[0],
                        desired[1] - rotated_axial[1],
                    )
                    ok, _reason = self.state.can_transform_component(
                        movable_unit.unit_id,
                        rotation,
                        translation,
                        allowed_contact=(int(movable_port.node_id), int(stationary_port.node_id)),
                        reflection=reflection,
                    )
                    if not ok:
                        continue
                    action = self._make_action(
                        movable_unit,
                        movable_port,
                        stationary_unit,
                        stationary_port,
                        contract=contract,
                    )
                    action.update({
                        "move_component_unit_id": movable_unit.unit_id,
                        "component_rotation": int(rotation),
                        "component_reflection": bool(reflection),
                        "component_translation": [int(translation[0]), int(translation[1])],
                        "moved_node_count": len(component_nodes),
                    })
                    actions.append(action)
        return actions

    @staticmethod
    def _make_action(
        source_unit: TopologyUnit,
        source_port: UnitPort,
        target_unit: TopologyUnit,
        target_port: UnitPort,
        *,
        contract: Optional[ExternalContract] = None,
        source_rotation: int = 0,
        source_translation: Tuple[int, int] = (0, 0),
    ) -> Dict[str, Any]:
        action_kind = str(contract.kind) if contract is not None else (
            "rigid" if source_unit.phase == "aromatic" else str(source_unit.phase)
        )
        return {
            "kind": action_kind,
            "contract_id": str(contract.contract_id if contract is not None else ""),
            "source_unit_id": str(source_unit.unit_id),
            "source_node_id": int(source_port.node_id),
            "source_port_id": str(source_port.port_id),
            "target_unit_id": str(target_unit.unit_id),
            "target_node_id": int(target_port.node_id),
            "target_port_id": str(target_port.port_id),
            "source_rotation": int(source_rotation) % 6,
            "source_translation": [int(source_translation[0]), int(source_translation[1])],
        }

    def get_candidates(self, k: int = 16) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()

        contracts = self._contract_candidates()
        for contract in contracts:
            source_unit = self.state.unit(contract.source_unit_id)
            target_unit = self.state.unit(contract.target_unit_id)
            source_port = _port_for_node(self.state, source_unit, contract.source_node_id, contract.source_port_id)
            target_port = _port_for_node(self.state, target_unit, contract.target_node_id, contract.target_port_id)
            if source_port is None or target_port is None:
                continue
            for action in self._candidate_for_pair(source_unit, source_port, target_unit, target_port, contract=contract):
                key = _action_key(action)
                if key not in seen:
                    action["score"] = self._placement_score(action) - 100.0
                    candidates.append(action)
                    seen.add(key)

        candidates.sort(key=lambda action: (float(action.get("score", 0.0)), _action_key(action)))
        return candidates[: max(1, int(k))]

    def _placement_score(self, action: Dict[str, Any]) -> float:
        score = 0.0
        if action.get("contract_id"):
            score -= 80.0
        score += 0.25 * float(action.get("moved_node_count", 0))
        score += float(self.state.component_count()) * 2.0
        translation = action.get("source_translation", (0, 0))
        score += 0.02 * (abs(int(translation[0])) + abs(int(translation[1])))
        return score

    def step(self, action: Dict[str, Any]) -> bool:
        if str(action.get("kind", "")) not in {self.phase, "rigid", "flex", "side"}:
            return False
        contract_id = str(action.get("contract_id", ""))
        if contract_id:
            contract = next((item for item in self.state.problem.contracts if item.contract_id == contract_id), None)
            if contract is None or contract.kind != self.phase:
                return False
        ok, _reason = self.state.connect(action)
        return bool(ok)

    def score(self) -> float:
        done_units = sum(1 for unit in self.state.problem.units if self.state.is_placed(unit.unit_id))
        done_contracts = len(self.state.satisfied_contracts)
        if self.contract_order is None:
            pending = len(self.state.mandatory_contracts(self.phase))
        else:
            pending = sum(
                contract_id not in self.state.satisfied_contracts
                for contract_id in self.contract_order
            )
        score = 4.0 * done_units + 8.0 * done_contracts - 20.0 * pending
        score -= 2.0 * max(0, self.state.component_count() - 1)
        if self.is_done():
            score += 80.0
        return score

    def get_result(self) -> Dict[str, Any]:
        phase_unit = "aromatic" if self.phase == "rigid" else self.phase
        return {
            "phase": self.phase,
            "done": bool(self.is_done()),
            "placed_units": sum(
                self.state.is_placed(unit.unit_id)
                for unit in self.state.problem.units if unit.phase == phase_unit
            ),
            "phase_units_total": sum(1 for unit in self.state.problem.units if unit.phase == phase_unit),
            "contracts_done": sum(1 for contract in self.state.problem.contracts if contract.kind == self.phase and contract.contract_id in self.state.satisfied_contracts),
            "contracts_total": sum(1 for contract in self.state.problem.contracts if contract.kind == self.phase),
            "components": self.state.component_count(),
            "coordinate_aspect": coordinate_aspect(
                self.state.node_axial.values()
            ),
            "score": self.score(),
        }


@dataclass
class SearchResult:
    state: HexAssemblyState
    score: float
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    candidate_states: List[HexAssemblyState] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "actions": copy.deepcopy(self.action_sequence),
            "info": copy.deepcopy(self.info),
            "molecule": self.state.to_dict(),
        }


@dataclass
class SearchNode:
    stage: TopologyPlacementStage
    action: Optional[Dict[str, Any]] = None
    parent: Optional["SearchNode"] = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Tuple[Any, ...], "SearchNode"] = field(default_factory=dict)
    untried: Optional[List[Dict[str, Any]]] = None

    def candidates(self, k: int) -> List[Dict[str, Any]]:
        if self.untried is None:
            self.untried = list(self.stage.get_candidates(k=k))
        return self.untried

    def terminal(self) -> bool:
        return bool(self.stage.is_done())

    def best_child(self, c_puct: float) -> Optional["SearchNode"]:
        if not self.children:
            return None
        best = None
        best_score = -float("inf")
        for child in self.children.values():
            if child.visits == 0:
                value = float("inf")
            else:
                value = child.value_sum / child.visits + c_puct * child.prior * math.sqrt(max(1, self.visits)) / (1 + child.visits)
            if value > best_score:
                best_score, best = value, child
        return best


class TopologyMCTS:
    def __init__(self, iterations: int = 32, c_puct: float = 1.414, candidate_k: int = 16, rollout_depth: int = 4, seed: Optional[int] = None):
        self.iterations = max(1, int(iterations))
        self.c_puct = float(c_puct)
        self.candidate_k = max(1, int(candidate_k))
        self.rollout_depth = max(1, int(rollout_depth))
        self.random = random.Random(seed)

    def choose_actions(self, stage: TopologyPlacementStage, branch_width: int = 1) -> List[Dict[str, Any]]:
        root = SearchNode(stage=stage.clone())
        if not root.candidates(self.candidate_k):
            return []
        for _ in range(self.iterations):
            self._iterate(root)
        ranked = sorted(root.children.values(), key=lambda child: (child.visits, child.value_sum / max(1, child.visits), child.prior), reverse=True)
        return [copy.deepcopy(child.action) for child in ranked[:max(1, int(branch_width))] if child.action is not None]

    def _iterate(self, root: SearchNode) -> None:
        node = root
        while not node.terminal():
            if node.candidates(self.candidate_k):
                break
            child = node.best_child(self.c_puct)
            if child is None:
                break
            node = child
        if not node.terminal():
            actions = node.candidates(self.candidate_k)
            child = None
            while actions:
                action = actions.pop(0)
                next_stage = node.stage.clone()
                if not next_stage.step(copy.deepcopy(action)):
                    continue
                child = SearchNode(stage=next_stage, action=copy.deepcopy(action), parent=node, prior=self._prior(action))
                node.children[_action_key(action)] = child
                break
            if child is not None:
                node = child
        value = self._rollout(node.stage.clone())
        while node is not None:
            node.visits += 1
            node.value_sum += float(value)
            node = node.parent

    def _rollout(self, stage: TopologyPlacementStage) -> float:
        for _ in range(self.rollout_depth):
            if stage.is_done():
                break
            actions = stage.get_candidates(self.candidate_k)
            if not actions:
                break
            action = actions[0]
            if not stage.step(copy.deepcopy(action)):
                break
        return float(stage.score())

    @staticmethod
    def _prior(action: Dict[str, Any]) -> float:
        raw = float(action.get("score", 0.0))
        return 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, raw / 10.0))))


def search_phase(stage: TopologyPlacementStage, *, iterations: int, candidate_k: int, max_steps: int, beam_width: int = 1, c_puct: float = 1.414, seed: Optional[int] = None, search_guidance=None, max_tree_nodes: int = 128) -> SearchResult:
    if str(stage.phase) == "rigid":
        if not any(contract.kind == "rigid" for contract in stage.state.problem.contracts):
            return _search_dynamic_rigid(
                stage, candidate_k=candidate_k, beam_width=beam_width,
                max_steps=max_steps, iterations=iterations, c_puct=c_puct,
                seed=seed, search_guidance=search_guidance,
                max_tree_nodes=max_tree_nodes,
            )
        return _search_rigid_components(
            stage,
            candidate_k=int(candidate_k),
            beam_width=int(beam_width),
            max_steps=int(max_steps),
            root_trials=int(iterations),
            c_puct=c_puct,
            seed=seed,
            search_guidance=search_guidance,
            max_tree_nodes=max_tree_nodes,
        )
    # Rollouts are only a value estimate.  Scaling their depth with the full
    # phase length makes a CSV with hundreds of contracts needlessly clone a
    # 600-node state tens of thousands of times.  Search itself still honors
    # ``max_steps`` below.
    rollout_depth = min(6, max(1, int(max_steps) // 3))
    engine = TopologyMCTS(iterations=iterations, candidate_k=candidate_k, rollout_depth=rollout_depth, c_puct=c_puct, seed=seed)
    current = stage.clone()
    actions: List[Dict[str, Any]] = []
    for _ in range(max(1, int(max_steps))):
        if current.is_done():
            break
        choices = engine.choose_actions(current, branch_width=max(1, int(beam_width)))
        if not choices:
            break
        selected = choices[0]
        if not current.step(copy.deepcopy(selected)):
            break
        actions.append(copy.deepcopy(selected))
    return SearchResult(
        state=current.state,
        score=float(current.score()),
        action_sequence=actions,
        info={"phase": stage.phase, "result": current.get_result()},
    )


from .rigid_sites import search_dynamic_rigid as _search_dynamic_rigid


def _rigid_unit_components(
    state: HexAssemblyState,
) -> List[Tuple[set[str], List[Any]]]:
    """Return connected components of the prescribed aromatic rigid graph."""
    rigid_contracts = [
        contract for contract in state.problem.contracts
        if contract.kind == "rigid"
    ]
    aromatic_ids = {
        str(unit.unit_id)
        for unit in state.problem.units
        if unit.phase == "aromatic"
    }
    adjacency: Dict[str, set[str]] = {
        unit_id: set() for unit_id in aromatic_ids
    }
    for contract in rigid_contracts:
        left, right = str(contract.source_unit_id), str(contract.target_unit_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    components: List[Tuple[set[str], List[Any]]] = []
    unseen = set(aromatic_ids)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        units = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    units.add(neighbor)
                    stack.append(neighbor)
        contracts = [
            contract for contract in rigid_contracts
            if str(contract.source_unit_id) in units
            and str(contract.target_unit_id) in units
        ]
        components.append((units, contracts))
    components.sort(key=lambda item: min(item[0]))
    return components


def _rigid_boundary_actions(
    state: HexAssemblyState,
    contract: Any,
    active_units: set[str],
    candidate_k: int,
) -> List[Dict[str, Any]]:
    """Enumerate actions that grow ``active_units`` across one rigid edge."""
    source_unit = state.unit(contract.source_unit_id)
    target_unit = state.unit(contract.target_unit_id)
    source_port = _port_for_node(
        state, source_unit, contract.source_node_id, contract.source_port_id
    )
    target_port = _port_for_node(
        state, target_unit, contract.target_node_id, contract.target_port_id
    )
    if source_port is None or target_port is None:
        return []
    actions = TopologyPlacementStage(state, "rigid")._candidate_for_pair(
        source_unit,
        source_port,
        target_unit,
        target_port,
        contract=contract,
    )
    result: List[Dict[str, Any]] = []
    for action in actions:
        move_id = str(action.get("move_component_unit_id", ""))
        if move_id:
            moved_units = state.component_unit_ids(move_id)
            # Keep the rooted component fixed while growing the tree.  An
            # action that moves an already active unit is deferred to cycle
            # handling and is not a valid tree expansion.
            if moved_units & set(active_units):
                continue
        result.append(action)
    ranked: List[Tuple[Tuple[Any, ...], Tuple[int, int], Dict[str, Any]]] = []
    prospective_units = set(active_units) | {
        str(contract.source_unit_id), str(contract.target_unit_id)
    }
    for action in result:
        trial = state.clone()
        ok, _reason = trial.connect(copy.deepcopy(action))
        if not ok:
            continue
        source = trial.node_axial[int(action["source_node_id"])]
        target = trial.node_axial[int(action["target_node_id"])]
        direction = (target[0] - source[0], target[1] - source[1])
        axis = min(direction, axial_neg(direction))
        ranked.append((
            _rigid_component_shape_rank(trial, prospective_units, complete=False),
            axis, action,
        ))
    ranked.sort(key=lambda row: (row[0], _action_key(row[2])))
    selected = []
    axes = set()
    limit = max(1, int(candidate_k))
    for _rank, axis, action in ranked:
        if axis not in axes:
            selected.append(action)
            axes.add(axis)
        if len(selected) >= limit:
            return selected
    selected_keys = {_action_key(action) for action in selected}
    for _rank, _axis, action in ranked:
        if _action_key(action) not in selected_keys:
            selected.append(action)
            selected_keys.add(_action_key(action))
        if len(selected) >= limit:
            break
    return selected


def _rigid_connection_concentration(
    state: HexAssemblyState,
    units: Iterable[str],
    contracts: Sequence[Any],
) -> Tuple[float, int, float]:
    """Return a soft penalty for overly concentrated rigid joins.

    The realized rigid graph is authoritative: a unit may have several rigid
    neighbours and this function never rejects that topology.  It only ranks
    candidate poses that spread joins more evenly when the prescribed graph
    offers alternatives.  A degree of two is the normal aromatic backbone;
    degrees three and four are allowed; only higher loads receive a penalty.
    """
    unit_ids = {str(value) for value in units}
    degrees = {unit_id: 0 for unit_id in unit_ids}
    satisfied = set(state.satisfied_contracts)
    for contract in contracts:
        if (
            str(getattr(contract, "kind", "")) != "rigid"
            or str(contract.contract_id) not in satisfied
        ):
            continue
        left = str(contract.source_unit_id)
        right = str(contract.target_unit_id)
        if left in degrees and right in degrees and left != right:
            degrees[left] += 1
            degrees[right] += 1
    if not degrees:
        return 0.0, 0, 0.0
    values = list(degrees.values())
    max_degree = max(values)
    overload = sum(max(0, degree - 4) ** 2 for degree in values)
    # A small variance term distinguishes a balanced branch from a layout that
    # repeatedly spends the same hub's perimeter ports, without penalising the
    # common degree-1/degree-2 aromatic backbone pattern.
    mean = sum(values) / len(values)
    variance = sum((degree - mean) ** 2 for degree in values) / len(values)
    penalty = 3.0 * float(overload) + 0.08 * float(variance)
    return penalty, int(max_degree), float(variance)


def _grow_rigid_component(
    state: HexAssemblyState,
    units: set[str],
    contracts: Sequence[Any],
    root: str,
    *,
    candidate_k: int,
    budget: List[int],
) -> Optional[HexAssemblyState]:
    """Backtracking tree growth for one aromatic rigid component."""
    if budget[0] <= 0:
        return None
    active = state.component_unit_ids(root)
    active &= set(units)
    pending = [
        contract for contract in contracts
        if contract.contract_id not in state.satisfied_contracts
    ]
    if not pending:
        return state

    boundary: List[Tuple[int, Any, List[Dict[str, Any]]]] = []
    for contract in pending:
        left, right = str(contract.source_unit_id), str(contract.target_unit_id)
        left_active, right_active = left in active, right in active
        if left_active == right_active:
            continue
        actions = _rigid_boundary_actions(
            state, contract, active, candidate_k
        )
        boundary.append((len(actions), contract, actions))
    boundary.sort(key=lambda item: (item[0], str(item[1].contract_id)))
    # A cyclic rigid specification can leave no unprocessed boundary edge;
    # only an already-adjacent closure is allowed in that case.
    if not boundary:
        for contract in pending:
            source_unit = state.unit(contract.source_unit_id)
            target_unit = state.unit(contract.target_unit_id)
            source_port = _port_for_node(
                state, source_unit, contract.source_node_id, contract.source_port_id
            )
            target_port = _port_for_node(
                state, target_unit, contract.target_node_id, contract.target_port_id
            )
            if source_port is None or target_port is None:
                continue
            actions = TopologyPlacementStage(state, "rigid")._candidate_for_pair(
                source_unit, source_port, target_unit, target_port, contract=contract
            )
            for action in actions:
                if action.get("move_component_unit_id"):
                    continue
                budget[0] -= 1
                trial = state.clone()
                ok, _reason = trial.connect(copy.deepcopy(action))
                if ok:
                    result = _grow_rigid_component(
                        trial,
                        units,
                        contracts,
                        root,
                        candidate_k=candidate_k,
                        budget=budget,
                    )
                    if result is not None:
                        return result
        return None

    # Prefer the most constrained edge, but try every boundary edge when the
    # first one leads to a later geometric dead end.
    for _count, contract, actions in boundary:
        if not actions:
            continue
        for action in actions:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            trial = state.clone()
            ok, _reason = trial.connect(copy.deepcopy(action))
            if not ok:
                continue
            result = _grow_rigid_component(
                trial,
                units,
                contracts,
                root,
                candidate_k=candidate_k,
                budget=budget,
            )
            if result is not None:
                return result
    return None


def _rigid_component_shape_rank(
    state: HexAssemblyState,
    units: Iterable[str],
    *,
    complete: bool = False,
) -> Tuple[Any, ...]:
    """Rank one physical rigid component independently of staging poses."""
    node_ids = {
        int(node.node_id)
        for unit_id in units
        if state.is_placed(str(unit_id))
        for node in state.unit(str(unit_id)).nodes
        if int(node.node_id) in state.node_axial
    }
    component_values = [state.node_axial[node_id] for node_id in node_ids]
    unit_groups = {
        str(unit_id): [
            state.node_axial[int(node.node_id)]
            for node in state.unit(str(unit_id)).nodes
            if int(node.node_id) in node_ids
        ]
        for unit_id in units
    }
    placed_unit_count = len([values for values in unit_groups.values() if values])
    shape = layout_metrics(
        component_values,
        unit_groups,
        full=bool(complete),
        include_void=True,
    )
    q_span = int(shape["q_span"])
    r_span = int(shape["r_span"])
    principal_ratio = float(shape["principal_axis_ratio"])
    # One or two aromatic units are intrinsically elongated.  Tightening the
    # target only after a component has enough joints prevents early beam
    # search from rejecting every legitimate start while strongly favouring a
    # folded, space-filling pose for long rigid chains.
    target_ratio = (
        math.inf if placed_unit_count <= 2
        else 3.0 if placed_unit_count <= 4
        else 2.2
    )
    axis_excess = (
        0.0 if math.isinf(target_ratio)
        else max(0.0, principal_ratio - target_ratio)
    )
    compactness_energy = (
        3.0 * axis_excess * axis_excess
        + 1.5 * float(shape["normalized_radius_of_gyration"])
        + 1.0 * float(shape["unit_density_cv"])
        + 0.5 * float(shape["node_density_cv"])
        + 1.5 * max(0.0, float(shape["largest_void_radius"]) - 4.0) ** 2
        + 0.5 * max(0.0, 0.45 - float(shape["hull_fill"])) ** 2
    )
    concentration_penalty, max_degree, degree_variance = (
        _rigid_connection_concentration(state, units, state.problem.contracts)
    )
    # Component size is deliberately a soft term.  The CSV rigid graph may
    # legitimately contain a large prescribed component; this must not become
    # a hidden maximum-size feasibility rule.
    size_penalty = 0.02 * max(0, placed_unit_count - 4) ** 2
    return (
        float(concentration_penalty + size_penalty + compactness_energy),
        float(concentration_penalty),
        int(max_degree),
        float(degree_variance),
        float(axis_excess),
        float(shape["largest_void_radius"]),
        float(shape["normalized_radius_of_gyration"]),
        float(shape["unit_density_cv"]),
        float(shape["node_density_cv"]),
        principal_ratio,
        float(shape["linearity"]),
        max(q_span, r_span),
        abs(q_span - r_span),
        int(shape["total_span"]),
        abs(float(shape["width_over_height"]) - 1.0),
    )


def _rigid_component_signature(
    state: HexAssemblyState,
    units: Iterable[str],
    contracts: Sequence[Any],
) -> Tuple[Any, ...]:
    unit_ids = set(str(value) for value in units)
    contract_ids = {
        str(contract.contract_id) for contract in contracts
    }
    return (
        tuple(sorted(
            (
                unit_id,
                int(placement.rotation),
                tuple(placement.translation),
                bool(placement.reflection),
            )
            for unit_id, placement in state.placements.items()
            if unit_id in unit_ids
        )),
        tuple(sorted(contract_ids & state.satisfied_contracts)),
    )


def _rigid_component_beam_rank(
    state: HexAssemblyState,
    units: Iterable[str],
    contracts: Sequence[Any],
) -> Tuple[Any, ...]:
    unit_ids = set(str(value) for value in units)
    contract_ids = {
        str(contract.contract_id) for contract in contracts
    }
    completed = len(contract_ids & state.satisfied_contracts)
    placed = sum(state.is_placed(unit_id) for unit_id in unit_ids)
    return (
        -int(completed),
        -int(placed),
        *_rigid_component_shape_rank(state, unit_ids, complete=False),
        _rigid_component_signature(state, unit_ids, contracts),
    )


def _rigid_root_candidates(
    units: Iterable[str],
    contracts: Sequence[Any],
) -> List[str]:
    """Return graph-centre, branch and leaf roots in a bounded order."""
    unit_ids = set(str(value) for value in units)
    adjacency: Dict[str, set[str]] = {unit_id: set() for unit_id in unit_ids}
    for contract in contracts:
        left = str(contract.source_unit_id)
        right = str(contract.target_unit_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    eccentricity: Dict[str, int] = {}
    for root in unit_ids:
        distance = {root: 0}
        queue = [root]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        eccentricity[root] = max(distance.values(), default=0)

    centers = sorted(
        unit_ids,
        key=lambda unit_id: (
            eccentricity[unit_id],
            -len(adjacency[unit_id]),
            unit_sort_key(unit_id),
        ),
    )[:2]
    branches = sorted(
        (unit_id for unit_id in unit_ids if len(adjacency[unit_id]) > 2),
        key=lambda unit_id: (-len(adjacency[unit_id]), unit_sort_key(unit_id)),
    )
    leaves = sorted(
        (unit_id for unit_id in unit_ids if len(adjacency[unit_id]) <= 1),
        key=lambda unit_id: unit_sort_key(unit_id),
    )
    ordered: List[str] = []
    for unit_id in centers + branches + leaves:
        if unit_id not in ordered:
            ordered.append(unit_id)
    return ordered[:6] or sorted(unit_ids, key=unit_sort_key)[:1]


def _grow_rigid_component_mcts(
    state: HexAssemblyState,
    units: set[str],
    contracts: Sequence[Any],
    root: str,
    *,
    candidate_k: int,
    beam_width: int,
    max_steps: int,
    iterations: int,
    c_puct: float,
    seed: Optional[int],
    diagnostics: list,
    search_guidance=None,
    max_tree_nodes: int = 128,
) -> List[HexAssemblyState]:
    """Keep a persistent PUCT tree over rigid connection actions."""
    from .search import PersistentMCTS, SearchBudget, SearchObservation

    contract_ids = {
        str(contract.contract_id) for contract in contracts
    }
    def expand(candidate):
        pending = [c for c in contracts if c.contract_id not in candidate.satisfied_contracts]
        if len(contract_ids & candidate.satisfied_contracts) >= max_steps:
            return
        active = candidate.component_unit_ids(root) & units
        boundary = []
        for contract in pending:
            if (contract.source_unit_id in active) == (contract.target_unit_id in active):
                continue
            actions = _rigid_boundary_actions(candidate, contract, active, max(12, 3 * candidate_k))
            if actions:
                boundary.append((len(actions), contract.contract_id, actions))
        actions = []
        if boundary:
            for _count, _cid, choices in sorted(boundary):
                actions.extend(choices)
        else:
            # A cycle may only be closed without moving an already joined unit.
            for contract in pending:
                left, right = candidate.unit(contract.source_unit_id), candidate.unit(contract.target_unit_id)
                lp = _port_for_node(candidate, left, contract.source_node_id, contract.source_port_id)
                rp = _port_for_node(candidate, right, contract.target_node_id, contract.target_port_id)
                if lp is not None and rp is not None:
                    actions.extend(a for a in TopologyPlacementStage(candidate, 'rigid')._candidate_for_pair(
                        left, lp, right, rp, contract=contract) if not a.get('move_component_unit_id'))
        trials = []
        for action in actions:
            trial = candidate.clone()
            ok, _reason = trial.connect(copy.deepcopy(action))
            if ok:
                trials.append(trial)
        yield from sorted(trials, key=lambda value: _rigid_component_beam_rank(value, units, contracts))

    budget = SearchBudget(max(1, iterations) * max(1, len(contracts)))
    tree = PersistentMCTS(state, expand=expand,
        accept=lambda s: contract_ids <= s.satisfied_contracts,
        signature=lambda s: _rigid_component_signature(s, units, contracts),
        solution_signature=lambda s: _rigid_layout_signature(s, units),
        evaluate=lambda s: 0.8 * len(contract_ids & s.satisfied_contracts) / max(1, len(contract_ids)),
        action=lambda p, c: c.action_history[-1] if c.action_history else {},
        c_puct=c_puct, seed=seed,
        max_nodes=max(1, int(max_tree_nodes)),
        guidance=(search_guidance.adapted(lambda s: SearchObservation('rigid', s,
            dict(root_unit_id=root, component_unit_ids=tuple(sorted(units)))))
            if search_guidance is not None else None))
    tree.run(budget, solution_limit=max(1, beam_width, candidate_k))
    tree.record()
    diagnostics.append(dict(root=root, simulations=budget.simulations,
                            budget=budget.limit, **tree.diagnostics()))
    ordered = sorted(
        tree.solutions,
        key=lambda value: (
            _rigid_component_shape_rank(value, units, complete=True),
            _rigid_component_signature(value, units, contracts),
        ),
    )
    unique = {}
    for value in ordered:
        unique.setdefault(_rigid_layout_signature(value, units), value)
    return list(unique.values())[:max(1, beam_width, candidate_k)]


def _rigid_layout_signature(state, units):
    """Ignore whole-component translations, rotations and reflections."""
    ids = sorted(n.node_id for uid in units for n in state.unit(uid).nodes
                 if n.node_id in state.node_axial)
    if not ids:
        return ()
    origin = state.node_axial[ids[0]]
    relative = [(state.node_axial[n][0] - origin[0],
                 state.node_axial[n][1] - origin[1]) for n in ids]
    return min(tuple(transform_axial(p, rotation, reflection) for p in relative)
               for reflection in (False, True) for rotation in range(6))


def _rigid_component_metrics(
    state: HexAssemblyState,
    components: Sequence[Tuple[set[str], Sequence[Any]]],
) -> List[Dict[str, Any]]:
    """Audit physical shape per rigid component, excluding staging gaps."""
    rows: List[Dict[str, Any]] = []
    for index, (units, contracts) in enumerate(components):
        node_ids = {
            int(node.node_id)
            for unit_id in units
            if state.is_placed(str(unit_id))
            for node in state.unit(str(unit_id)).nodes
            if int(node.node_id) in state.node_axial
        }
        groups = {
            str(unit_id): [
                state.node_axial[int(node.node_id)]
                for node in state.unit(str(unit_id)).nodes
                if int(node.node_id) in node_ids
            ]
            for unit_id in units
        }
        metrics = layout_metrics(
            [state.node_axial[node_id] for node_id in node_ids],
            groups,
            full=True,
            include_void=True,
        )
        unit_count = len(units)
        ratio_limit = math.inf if unit_count <= 2 else 3.0 if unit_count <= 4 else 2.2
        rows.append({
            "component_index": int(index),
            "unit_ids": sorted(units, key=unit_sort_key),
            "unit_count": int(unit_count),
            "node_count": len(node_ids),
            "contract_count": len(contracts),
            "principal_axis_ratio_limit": (
                None if math.isinf(ratio_limit) else float(ratio_limit)
            ),
            **metrics,
            "component_layout_valid": bool(
                float(metrics["principal_axis_ratio"]) <= ratio_limit
                and float(metrics["largest_void_radius"]) <= 8.0
            ),
        })
    return rows


def _pack_rigid_components(
    state: HexAssemblyState,
    components: Sequence[Tuple[set[str], Sequence[Any]]],
    *,
    clearance: int = 3,
) -> None:
    """Translate disconnected rigid components into compact staging rows."""
    records: List[Dict[str, Any]] = []
    for units, _contracts in components:
        node_ids = {
            int(node.node_id)
            for unit_id in units
            if state.is_placed(unit_id)
            for node in state.unit(unit_id).nodes
            if int(node.node_id) in state.node_axial
        }
        if not node_ids:
            continue
        q_values = [int(state.node_axial[node_id][0]) for node_id in node_ids]
        r_values = [int(state.node_axial[node_id][1]) for node_id in node_ids]
        records.append({
            "units": set(units),
            "nodes": node_ids,
            "min_q": min(q_values),
            "min_r": min(r_values),
            "width": max(q_values) - min(q_values) + 1,
            "height": max(r_values) - min(r_values) + 1,
        })
    if not records:
        return

    gap = max(2, int(clearance))
    padded_area = sum(
        (int(record["width"]) + gap) * (int(record["height"]) + gap)
        for record in records
    )
    records.sort(key=lambda record: (
        -int(record["height"]) * int(record["width"]),
        -int(record["height"]),
        tuple(sorted(record["units"], key=unit_sort_key)),
    ))

    max_width = max(int(record["width"]) for record in records)
    square_width = int(math.ceil(math.sqrt(float(padded_area))))
    cumulative_widths: List[int] = []
    running_width = 0
    for record in records:
        running_width += int(record["width"])
        cumulative_widths.append(running_width + gap * len(cumulative_widths))
    candidate_widths = sorted({
        max_width,
        square_width,
        int(math.ceil(square_width * 1.25)),
        int(math.ceil(square_width * 1.5)),
        *cumulative_widths,
    })

    def build_layout(target_width: int) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        cursor_q = 0
        cursor_r = 0
        row_height = 0
        deltas: List[Tuple[int, int]] = []
        coordinates: List[Tuple[int, int]] = []
        for record in records:
            width = int(record["width"])
            height = int(record["height"])
            if cursor_q and cursor_q + width > int(target_width):
                cursor_q = 0
                cursor_r += row_height + gap
                row_height = 0
            delta = (
                int(cursor_q - int(record["min_q"])),
                int(cursor_r - int(record["min_r"])),
            )
            deltas.append(delta)
            coordinates.extend(
                axial_add(state.node_axial[int(node_id)], delta)
                for node_id in record["nodes"]
            )
            cursor_q += width + gap
            row_height = max(row_height, height)
        return deltas, coordinate_aspect(coordinates)

    layout_candidates = [
        (*build_layout(width), width) for width in candidate_widths
    ]
    deltas, _shape, _width = min(
        layout_candidates,
        key=lambda item: (
            float(item[1]["violation"]),
            float(item[1]["principal_axis_ratio"]),
            float(item[1]["linearity"]),
            int(item[1]["total_span"]),
            int(item[2]),
        ),
    )

    for record, delta in zip(records, deltas):
        for unit_id in record["units"]:
            placement = state.placements.get(str(unit_id))
            if placement is None:
                continue
            state.placements[str(unit_id)] = UnitPlacement(
                unit_id=str(unit_id),
                rotation=int(placement.rotation),
                translation=axial_add(placement.translation, delta),
                reflection=bool(placement.reflection),
            )
        for node_id in record["nodes"]:
            state.node_axial[int(node_id)] = axial_add(
                state.node_axial[int(node_id)], delta
            )
            old_position = state.final_node_positions.get(int(node_id))
            if old_position is None:
                axial = state.node_axial[int(node_id)]
                state.final_node_positions[int(node_id)] = (
                    float(axial[0]), float(axial[1])
                )
            else:
                state.final_node_positions[int(node_id)] = (
                    float(old_position[0]) + delta[0],
                    float(old_position[1]) + delta[1],
                )

    state.occupied_axial = {
        axial: int(node_id) for node_id, axial in state.node_axial.items()
    }
    state.edges = [
        AssemblyEdge(
            edge.source_node_id,
            edge.target_node_id,
            edge.kind,
            edge.contract_id,
            state._edge_direction(
                state.node_axial[edge.source_node_id],
                state.node_axial[edge.target_node_id],
            ),
        )
        for edge in state.edges
    ]


def _search_rigid_components(
    stage, *, candidate_k, beam_width=4, max_steps, root_trials=16,
    c_puct=1.414, seed=None,
    search_guidance=None,
    max_tree_nodes: int = 128,
):
    """Retain distinct solved component layouts for the downstream Flex tree."""
    initial = stage.clone().state
    components = _rigid_unit_components(initial)
    width = max(1, int(candidate_k), int(beam_width))
    beam = [initial]
    diagnostics = []
    for units, contracts in components:
        if not contracts:
            continue
        expanded = []
        for current in beam:
            for root in _rigid_root_candidates(units, contracts)[:max(1, root_trials)]:
                trial = current.clone()
                removed = {n.node_id for uid in units if uid != root
                           for n in trial.unit(uid).nodes}
                for uid in units - {root}:
                    trial.placements.pop(uid, None)
                for node in removed:
                    trial.node_axial.pop(node, None)
                    trial.final_node_positions.pop(node, None)
                root_nodes = {n.node_id for n in trial.unit(root).nodes}
                trial.occupied_axial = {p: n for n, p in trial.node_axial.items()
                                        if n in root_nodes}
                expanded.extend(_grow_rigid_component_mcts(
                    trial, units, contracts, root, candidate_k=candidate_k,
                    beam_width=beam_width, max_steps=max_steps,
                    iterations=root_trials, c_puct=c_puct, seed=seed, diagnostics=diagnostics,
                    search_guidance=search_guidance,
                    max_tree_nodes=max_tree_nodes,
                ))
        if not expanded:
            break
        unique = {}
        for candidate in sorted(expanded, key=lambda s: (
            _rigid_component_shape_rank(s, units, complete=True), s.state_signature(),
        )):
            signature = tuple(_rigid_layout_signature(candidate, group)
                              for group, _contracts in components)
            unique.setdefault(signature, candidate)
        beam = list(unique.values())[:width]
    for candidate in beam:
        _pack_rigid_components(candidate, components)
    current = beam[0]
    result = TopologyPlacementStage(current, 'rigid').get_result()
    result['rigid_component_metrics'] = _rigid_component_metrics(current, components)
    result['retained_candidate_count'] = len(beam)
    result['search_algorithm'] = 'component_puct'
    result['mcts_searches'] = diagnostics
    result['status'] = 'complete' if result['done'] else 'budget_exhausted'
    result['failure_scope'] = 'none' if result['done'] else 'rigid_search_budget'
    result['candidate_space_complete'] = False
    result['completed_contract_count'] = int(result['contracts_done'])
    result['zero_domain_units'] = []
    result['best_partial_state'] = {
        'placed_unit_ids': sorted(str(uid) for uid in current.placements),
        'satisfied_contract_ids': sorted(str(cid) for cid in current.satisfied_contracts),
        'completed_contract_count': int(result['contracts_done']),
        'component_count': int(current.component_count()),
    }
    return SearchResult(
        state=current, score=float(TopologyPlacementStage(current, 'rigid').score()),
        action_sequence=copy.deepcopy(current.action_history),
        info={'phase': 'rigid', 'result': result}, candidate_states=beam,
    )
