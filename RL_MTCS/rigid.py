"""MCTS placement of complete inverse topology units on a hexagonal lattice."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import (
    HEX_DIRECTIONS,
    axial_add,
    axial_distance,
    axial_neg,
    rotate_axial,
    transform_direction,
)
from .state import HexAssemblyState
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


def _phase_for_unit(unit: TopologyUnit) -> str:
    return str(unit.phase)


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
        for rotation in range(6):
            source_direction = self.state.global_port_direction(source_unit.unit_id, source_port, rotation=rotation)
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
                )
                translation = (source_target[0] - source_local[0], source_target[1] - source_local[1])
                ok, _reason = self.state.can_place(
                    source_unit.unit_id,
                    rotation,
                    translation,
                    allowed_contact=(int(source_port.node_id), int(target_port.node_id)),
                )
                if not ok:
                    continue
                actions.append(self._make_action(
                    source_unit,
                    source_port,
                    target_unit,
                    target_port,
                    contract=contract,
                    source_rotation=rotation,
                    source_translation=translation,
                ))
                if len(actions) >= 12:
                    return actions
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
        for rotation in range(6):
            transformed_direction = (
                transform_direction(movable_direction, rotation)
                if movable_direction is not None else None
            )
            rotated_axial = rotate_axial(movable_axial, rotation)
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
                    "component_translation": [int(translation[0]), int(translation[1])],
                    "moved_node_count": len(component_nodes),
                })
                actions.append(action)
                if len(actions) >= 12:
                    return actions
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
            "placed_units": len(self.state.placements),
            "phase_units_total": sum(1 for unit in self.state.problem.units if unit.phase == phase_unit),
            "contracts_done": sum(1 for contract in self.state.problem.contracts if contract.kind == self.phase and contract.contract_id in self.state.satisfied_contracts),
            "contracts_total": sum(1 for contract in self.state.problem.contracts if contract.kind == self.phase),
            "components": self.state.component_count(),
            "score": self.score(),
        }


@dataclass
class SearchResult:
    state: HexAssemblyState
    score: float
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

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


def search_phase(stage: TopologyPlacementStage, *, iterations: int, candidate_k: int, max_steps: int, beam_width: int = 1, c_puct: float = 1.414, seed: Optional[int] = None) -> SearchResult:
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


def assemble_inverse_topology(init_data: Dict[str, Any], config: Any = None) -> SearchResult:
    """Run rigid -> flex -> side placement using the inverse initializer."""
    cfg = config
    state = init_data["state"].clone()
    all_actions: List[Dict[str, Any]] = []
    phase_results: Dict[str, Any] = {}
    phase_specs = (
        ("rigid", int(getattr(cfg, "iterations", 32) if cfg is not None else 32), int(getattr(cfg, "candidate_k", 16) if cfg is not None else 16), int(getattr(cfg, "rigid_max_steps", 128) if cfg is not None else 128)),
        ("flex", int(getattr(cfg, "flex_iterations", 32) if cfg is not None else 32), int(getattr(cfg, "candidate_k", 16) if cfg is not None else 16), int(getattr(cfg, "flex_max_steps", 128) if cfg is not None else 128)),
        ("side", int(getattr(cfg, "side_iterations", 32) if cfg is not None else 32), int(getattr(cfg, "candidate_k", 16) if cfg is not None else 16), int(getattr(cfg, "side_max_steps", 128) if cfg is not None else 128)),
    )
    for phase, iterations, candidate_k, max_steps in phase_specs:
        stage = TopologyPlacementStage(state, phase)
        result = search_phase(stage, iterations=iterations, candidate_k=candidate_k, max_steps=max_steps, beam_width=int(getattr(cfg, "beam_width", 1) if cfg is not None else 1), c_puct=float(getattr(cfg, "c_puct", 1.414) if cfg is not None else 1.414), seed=getattr(cfg, "seed", None) if cfg is not None else None)
        state = result.state
        all_actions.extend(result.action_sequence)
        phase_results[phase] = result.info.get("result", {})
        if not bool(result.info.get("result", {}).get("done", False)):
            break
    return SearchResult(
        state=state,
        score=float(sum(float(row.get("score", 0.0)) for row in phase_results.values())),
        action_sequence=all_actions,
        info={"phase_results": phase_results, "audit": state.audit(), "complete": state.is_complete()},
    )
