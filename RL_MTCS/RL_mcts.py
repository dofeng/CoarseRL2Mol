"""Inverse-driven Monte-Carlo topology assembly entry point.

Only placement is searched here.  Aromatic clusters and topology products are
immutable input units supplied by ``RL_init.initialize``; no allocator,
structure synthesis, SU conversion, branch allocation, or substitution is
performed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .RL_init import initialize
from .hex_assembly_state import HexAssemblyState
from .inverse_contract import InverseInputError
from .inverse_mcts import (
    SearchResult,
    TopologyPlacementStage,
    assemble_inverse_topology,
    search_phase,
)


@dataclass
class MCTSConfig:
    """Configuration for inverse topology placement."""

    c_puct: float = 1.414
    iterations: int = 64
    flex_iterations: int = 64
    side_iterations: int = 64
    candidate_k: int = 16
    beam_width: int = 1
    max_depth: int = 128
    rigid_max_steps: int = 128
    flex_max_steps: int = 128
    side_max_steps: int = 128
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.iterations = max(1, int(self.iterations))
        self.flex_iterations = max(1, int(self.flex_iterations))
        self.side_iterations = max(1, int(self.side_iterations))
        self.candidate_k = max(1, int(self.candidate_k))
        self.beam_width = max(1, int(self.beam_width))
        self.max_depth = max(1, int(self.max_depth))
        self.rigid_max_steps = max(1, int(self.rigid_max_steps))
        self.flex_max_steps = max(1, int(self.flex_max_steps))
        self.side_max_steps = max(1, int(self.side_max_steps))


DEFAULT_CONFIG = MCTSConfig()


@dataclass
class BeamCandidate:
    """Compatibility wrapper for callers that consumed the old beam result."""

    state: HexAssemblyState
    score: float
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _candidate_from_result(result: SearchResult) -> BeamCandidate:
    return BeamCandidate(
        state=result.state,
        score=float(result.score),
        action_sequence=list(result.action_sequence),
        info=copy.deepcopy(result.info),
    )


class MCTSRunner:
    """Run inverse topology placement: rigid -> flex -> side."""

    def __init__(self, inverse_result: Any, config: Optional[MCTSConfig] = None):
        self.inverse_result = inverse_result
        self.config = config or DEFAULT_CONFIG
        self.init_data = initialize(inverse_result)
        self.initial_state: HexAssemblyState = self.init_data["state"].clone()

    def _run_phase(self, state: HexAssemblyState, phase: str) -> SearchResult:
        if phase == "rigid":
            iterations = self.config.iterations
            max_steps = self.config.rigid_max_steps
        elif phase == "flex":
            iterations = self.config.flex_iterations
            max_steps = self.config.flex_max_steps
        else:
            iterations = self.config.side_iterations
            max_steps = self.config.side_max_steps
        return search_phase(
            TopologyPlacementStage(state, phase),
            iterations=iterations,
            candidate_k=self.config.candidate_k,
            max_steps=max_steps,
            beam_width=self.config.beam_width,
            c_puct=self.config.c_puct,
            seed=self.config.seed,
        )

    @staticmethod
    def _stage_payload(result: SearchResult) -> Dict[str, Any]:
        return {
            "state": result.state,
            "best_candidate": _candidate_from_result(result),
            "beam_candidates": [_candidate_from_result(result)],
            "summaries": [copy.deepcopy(result.info.get("result", {}))],
            "result": copy.deepcopy(result.info.get("result", {})),
            "molecule": result.state.to_dict(),
        }

    def run_rigid_stage(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        result = self._run_phase(self.initial_state.clone(), "rigid")
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "rigid")
        return payload

    def run_flex_stage(self, rigid_result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        state = self._state_from_result(rigid_result)
        result = self._run_phase(state, "flex")
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "flex")
        return payload

    def run_side_stage(self, flex_result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        state = self._state_from_result(flex_result)
        result = self._run_phase(state, "side")
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "side")
        return payload

    def run_all_stages(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        base = Path(output_dir) if output_dir else None
        rigid = self.run_rigid_stage(str(base / "rigid") if base else None)
        if not rigid["result"].get("done", False):
            return self._failed_result("rigid", rigid)
        flex = self.run_flex_stage(rigid, str(base / "flex") if base else None)
        if not flex["result"].get("done", False):
            return self._failed_result("flex", flex)
        side = self.run_side_stage(flex, str(base / "side") if base else None)
        final_state = self._state_from_result(side)
        result = {
            "status": "complete" if final_state.is_complete() else "incomplete",
            "stage": "side",
            "init": self.init_data["diagnostics"],
            "rigid": rigid["result"],
            "flex": flex["result"],
            "side": side["result"],
            "best_candidate": side["best_candidate"],
            "beam_candidates": side["beam_candidates"],
            "molecule": final_state.to_dict(),
            "audit": final_state.audit(),
        }
        self._write_json(result, str(base / "final.json") if base else None)
        return result

    @staticmethod
    def _failed_result(stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state")
        return {
            "status": "failed",
            "stage": str(stage),
            "best_candidate": payload.get("best_candidate"),
            "beam_candidates": payload.get("beam_candidates", []),
            "molecule": state.to_dict() if isinstance(state, HexAssemblyState) else {},
            "audit": state.audit() if isinstance(state, HexAssemblyState) else {},
        }

    @staticmethod
    def _state_from_result(result: Dict[str, Any]) -> HexAssemblyState:
        state = result.get("state")
        if isinstance(state, HexAssemblyState):
            return state.clone()
        candidate = result.get("best_candidate")
        if isinstance(candidate, BeamCandidate):
            return candidate.state.clone()
        raise InverseInputError("stage result does not contain a HexAssemblyState")

    @staticmethod
    def _write_json(value: Any, path: Optional[str]) -> None:
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _write_stage_output(self, payload: Dict[str, Any], output_dir: Optional[str], stage: str) -> None:
        if not output_dir:
            return
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(payload["molecule"], str(directory / f"{stage}_molecule.json"))
        self._write_json(payload["result"], str(directory / f"{stage}_summary.json"))


def assemble_from_inverse(inverse_result: Any, config: Optional[MCTSConfig] = None) -> Dict[str, Any]:
    """Functional API for inverse pipeline integration."""
    runner = MCTSRunner(inverse_result, config=config)
    return runner.run_all_stages()


def main() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Inverse topology hex-grid MCTS assembly")
    parser.add_argument("--inverse-json", required=True, help="Complete inverse topology JSON export")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--flex-iterations", type=int, default=64)
    parser.add_argument("--side-iterations", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    config = MCTSConfig(
        iterations=args.iterations,
        flex_iterations=args.flex_iterations,
        side_iterations=args.side_iterations,
        candidate_k=args.candidate_k,
        seed=args.seed,
    )
    result = MCTSRunner(args.inverse_json, config=config).run_all_stages(args.output)
    print(json.dumps({"status": result.get("status"), "audit": result.get("audit", {})}, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
