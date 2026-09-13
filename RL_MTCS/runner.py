"""Inverse-driven Monte-Carlo topology assembly entry point.

Only placement is searched here.  Aromatic clusters and topology products are
immutable input units supplied by :func:`RL_MTCS.input.initialize`; no allocator,
structure synthesis, SU conversion, branch allocation, or substitution is
performed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .input import initialize, inspect_csv_initialization
from .state import HexAssemblyState
from .models import InverseInputError, MCTSConfig
from .rigid import (
    SearchResult,
    TopologyPlacementStage,
    search_phase,
)
from .output import export_final_state
from .flex import run_planned_flex_search
from .side import run_planned_side_search
from .substitution import run_substitution_stage
from .geometry import HEX_DIRECTIONS
from .output import save_state_png


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


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
    """Run inverse topology placement: rigid -> flex -> side -> substitution."""

    def __init__(
        self,
        inverse_result: Any,
        config: Optional[MCTSConfig] = None,
        topology_specs_csv: Any = None,
    ):
        self.inverse_result = inverse_result
        self.topology_specs_csv = topology_specs_csv
        self.config = config or DEFAULT_CONFIG
        self.init_data = initialize(inverse_result, topology_specs_csv)
        self.initial_state: HexAssemblyState = self.init_data["state"].clone()
        self.nmr_scorer = self._initialize_nmr_scorer()

    def _initialize_nmr_scorer(self):
        spectrum_csv = getattr(self.config, "spectrum_csv", None)
        if not spectrum_csv:
            return None
        elements = getattr(self.config, "elements", None)
        composition_csv = getattr(self.config, "element_composition_csv", None)
        if elements is None and composition_csv:
            elements = composition_csv
        if elements is None and isinstance(self.inverse_result, (str, Path)):
            nodes_path = Path(self.inverse_result)
            inferred = nodes_path.resolve().parent / "final_element_composition.csv"
            if inferred.is_file():
                elements = str(inferred)
        if elements is None:
            raise InverseInputError(
                "NMR guidance requires target elements or element_composition_csv"
            )
        checkpoint = getattr(self.config, "g2s_checkpoint", None)
        if not checkpoint:
            inferred_checkpoint = (
                Path(__file__).resolve().parents[1]
                / "checkpoints_g2s_k3"
                / "best_g2s_model.pt"
            )
            if inferred_checkpoint.is_file():
                checkpoint = str(inferred_checkpoint)
        if not checkpoint or not Path(checkpoint).is_file():
            raise InverseInputError(
                "NMR guidance requires a valid g2s_checkpoint"
            )
        from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
        return NMRGuidanceScorer(
            spectrum_csv=spectrum_csv,
            elements=elements,
            model_checkpoint=checkpoint,
            device=str(getattr(self.config, "nmr_device", "auto")),
            batch_size=int(getattr(self.config, "nmr_batch_size", 512)),
            hwhm=float(getattr(self.config, "nmr_hwhm", 1.0)),
        )

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
        result, plan = run_planned_flex_search(
            state,
            iterations=self.config.flex_iterations,
            candidate_k=self.config.candidate_k,
            max_steps=self.config.flex_max_steps,
            beam_width=self.config.beam_width,
            c_puct=self.config.c_puct,
            seed=self.config.seed,
            strict=bool(getattr(self.config, "strict_geometry", False)),
            nmr_scorer=self.nmr_scorer,
            nmr_weight=float(getattr(self.config, "nmr_weight", 1.0)),
            geometry_weight=float(getattr(self.config, "geometry_weight", 0.05)),
            contact_weight=float(getattr(self.config, "contact_weight", 0.02)),
        )
        payload = self._stage_payload(result)
        payload["flex_plan"] = plan.to_dict()
        self._write_stage_output(payload, output_dir, "flex")
        return payload

    def run_side_stage(self, flex_result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        state = self._state_from_result(flex_result)
        result = run_planned_side_search(
            state,
            iterations=self.config.side_iterations,
            candidate_k=self.config.candidate_k,
            max_steps=self.config.side_max_steps,
            beam_width=self.config.beam_width,
            c_puct=self.config.c_puct,
            seed=self.config.seed,
            strict=bool(getattr(self.config, "strict_geometry", False)),
            nmr_scorer=self.nmr_scorer,
            nmr_weight=float(getattr(self.config, "nmr_weight", 1.0)),
            geometry_weight=float(getattr(self.config, "geometry_weight", 0.05)),
            contact_weight=float(getattr(self.config, "contact_weight", 0.02)),
        )
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "side")
        return payload

    def run_substitution_stage(self, side_result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        state = self._state_from_result(side_result)
        final_state, summary = run_substitution_stage(
            state,
            nmr_scorer=self.nmr_scorer,
            candidate_k=int(getattr(self.config, "substitution_candidate_k", 32)),
            beam_width=int(getattr(self.config, "substitution_beam_width", 4)),
            nmr_weight=float(getattr(self.config, "substitution_nmr_weight", 1.0)),
        )
        payload = {
            "state": final_state,
            "best_candidate": BeamCandidate(final_state, 0.0, [], summary),
            "beam_candidates": [BeamCandidate(final_state, 0.0, [], summary)],
            "summaries": [summary],
            "result": summary,
            "molecule": final_state.to_dict(),
        }
        self._write_stage_output(payload, output_dir, "substitution")
        return payload

    def run_all_stages(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        # CSV runs are user-facing exports.  Keep their artefacts beside the
        # inverse CSV pair even when the caller does not provide ``--output``.
        if output_dir:
            base = Path(output_dir)
        elif isinstance(self.inverse_result, (str, Path)) and Path(self.inverse_result).suffix.lower() == ".csv":
            base = Path(self.inverse_result).resolve().parent / "rl_mtcs_outputs"
        else:
            base = None
        rigid = self.run_rigid_stage(str(base / "rigid") if base else None)
        if not rigid["result"].get("done", False):
            return self._failed_result("rigid", rigid)
        flex = self.run_flex_stage(rigid, str(base / "flex") if base else None)
        if not flex["result"].get("done", False):
            return self._failed_result("flex", flex)
        side = self.run_side_stage(flex, str(base / "side") if base else None)
        if not side["result"].get("done", False):
            return self._failed_result("side", side)
        substitution = self.run_substitution_stage(
            side, str(base / "substitution") if base else None
        )
        if not substitution["result"].get("complete", False):
            return self._failed_result("substitution", substitution)
        final_state = substitution["state"]
        outputs = export_final_state(final_state, base) if base else {}
        nmr_result: Dict[str, Any] = {"enabled": False}
        if self.nmr_scorer is not None:
            final_nmr_score = self.nmr_scorer.score_state(
                final_state, stage="final_selection"
            )
            nmr_result = {
                "enabled": True,
                "final_score": final_nmr_score,
                "statistics": self.nmr_scorer.statistics(),
            }
            if base:
                outputs.update(self.nmr_scorer.export_outputs(base, final_state))
                nmr_result["statistics"] = self.nmr_scorer.statistics()
        audit = final_state.audit()
        complete = bool(final_state.is_complete() and audit.get("complete", False))
        result = {
            "ok": complete,
            "status": "complete" if complete else "incomplete",
            "stage": "substitution",
            "init": self.init_data["diagnostics"],
            "rigid": rigid["result"],
            "flex": flex["result"],
            "side": side["result"],
            "substitution": substitution["result"],
            "nmr": nmr_result,
            "best_candidate": substitution["best_candidate"],
            "beam_candidates": substitution["beam_candidates"],
            "molecule": final_state.to_dict(),
            "audit": audit,
            "outputs": outputs,
        }
        self._write_json(result, str(base / "final.json") if base else None)
        return result

    @staticmethod
    def _failed_result(stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state")
        return {
            "ok": False,
            "status": "failed",
            "stage": str(stage),
            "best_candidate": payload.get("best_candidate"),
            "beam_candidates": payload.get("beam_candidates", []),
            "result": payload.get("result", {}),
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


def assemble_from_csv(
    nodes_csv: Any,
    topology_specs_csv: Any,
    config: Optional[MCTSConfig] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble directly from the two final inverse CSV exports."""
    runner = MCTSRunner(
        nodes_csv,
        config=config,
        topology_specs_csv=topology_specs_csv,
    )
    return runner.run_all_stages(output_dir)


def _connection_row(step: int, action: Mapping[str, Any], before, after) -> Dict[str, Any]:
    source_node_id = int(action["source_node_id"])
    target_node_id = int(action["target_node_id"])
    source_axial = after.node_axial[source_node_id]
    target_axial = after.node_axial[target_node_id]
    direction = (
        int(target_axial[0]) - int(source_axial[0]),
        int(target_axial[1]) - int(source_axial[1]),
    )
    moved_ids = sorted(
        node_id
        for node_id, axial in before.node_axial.items()
        if after.node_axial.get(int(node_id)) != axial
    )
    component_translation = action.get("component_translation", (0, 0))
    source_translation = action.get("source_translation", (0, 0))
    return {
        "step": int(step),
        "contract_id": str(action.get("contract_id", "")),
        "source_unit_id": str(action.get("source_unit_id", "")),
        "source_node_id": source_node_id,
        "source_port_id": str(action.get("source_port_id", "")),
        "target_unit_id": str(action.get("target_unit_id", "")),
        "target_node_id": target_node_id,
        "target_port_id": str(action.get("target_port_id", "")),
        "move_component_unit_id": str(action.get("move_component_unit_id", "")),
        "component_rotation": int(action.get("component_rotation", 0)),
        "component_translation_q": int(component_translation[0]),
        "component_translation_r": int(component_translation[1]),
        "source_rotation": int(action.get("source_rotation", 0)),
        "source_translation_q": int(source_translation[0]),
        "source_translation_r": int(source_translation[1]),
        "reported_moved_node_count": int(action.get("moved_node_count", 0)),
        "actual_moved_node_count": len(moved_ids),
        "source_q": int(source_axial[0]),
        "source_r": int(source_axial[1]),
        "target_q": int(target_axial[0]),
        "target_r": int(target_axial[1]),
        "direction_q": int(direction[0]),
        "direction_r": int(direction[1]),
        "hex_neighbor": direction in HEX_DIRECTIONS,
        "component_count_after_step": int(after.component_count()),
        "rigid_contracts_done_after_step": sum(
            1
            for contract in after.problem.contracts
            if contract.kind == "rigid"
            and contract.contract_id in after.satisfied_contracts
        ),
    }


def _final_component_rows(state, output_dir: Path) -> tuple[list[Dict[str, Any]], list[str]]:
    """Save one readable image for every final rigid-stage component."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ownership = state.problem.ownership_map()
    su_map = state.current_node_map()
    remaining_unit_ids = set(state.placements)
    components = []
    while remaining_unit_ids:
        seed_unit_id = min(remaining_unit_ids)
        unit_ids = state.component_unit_ids(seed_unit_id)
        remaining_unit_ids.difference_update(unit_ids)
        node_ids = sorted(
            int(node_id)
            for node_id in state.node_axial
            if ownership.get(int(node_id), "") in unit_ids
        )
        node_id_set = set(node_ids)
        edges = [
            edge for edge in state.edges
            if int(edge.source_node_id) in node_id_set
            and int(edge.target_node_id) in node_id_set
        ]
        rigid_edges = [edge for edge in edges if str(edge.kind) == "rigid"]
        rigid_contract_ids = sorted({
            str(edge.contract_id) for edge in rigid_edges if edge.contract_id
        })
        components.append({
            "unit_ids": sorted(unit_ids),
            "node_ids": node_ids,
            "su10_node_ids": sorted(
                node_id for node_id in node_ids
                if int(su_map.get(node_id, -1)) == 10
            ),
            "edge_count": len(edges),
            "aromatic_edge_count": sum(
                1 for edge in edges if str(edge.kind) == "aromatic"
            ),
            "rigid_connection_count": len(rigid_edges),
            "rigid_contract_ids": rigid_contract_ids,
        })

    def contract_sort_key(contract_id: str) -> tuple[int, str]:
        parts = str(contract_id).split(":", 2)
        try:
            return int(parts[1]), str(contract_id)
        except (IndexError, ValueError):
            return 10**9, str(contract_id)

    components.sort(key=lambda item: (
        not bool(item["rigid_contract_ids"]),
        tuple(contract_sort_key(value) for value in item["rigid_contract_ids"]),
        item["node_ids"],
    ))
    rows: list[Dict[str, Any]] = []
    pngs: list[str] = []
    for index, component in enumerate(components, start=1):
        cluster_id = f"rigid_cluster_{index:02d}"
        kind = (
            "rigid_connected_aromatic_cluster"
            if component["rigid_contract_ids"]
            else "standalone_aromatic_cluster"
        )
        display_kind = (
            "rigid-connected"
            if component["rigid_contract_ids"]
            else "standalone aromatic"
        )
        png_path = output_dir / f"{cluster_id}.png"
        contract_text = (
            ", ".join(component["rigid_contract_ids"])
            if component["rigid_contract_ids"]
            else "none"
        )
        save_state_png(
            state,
            png_path,
            title=f"{cluster_id} | {display_kind} | {contract_text}",
            visible_node_ids=component["node_ids"],
        )
        row = {
            "cluster_index": index,
            "cluster_id": cluster_id,
            "cluster_kind": kind,
            "unit_count": len(component["unit_ids"]),
            "node_count": len(component["node_ids"]),
            "edge_count": component["edge_count"],
            "aromatic_edge_count": component["aromatic_edge_count"],
            "rigid_connection_count": component["rigid_connection_count"],
            "unit_ids": component["unit_ids"],
            "node_ids": component["node_ids"],
            "su10_node_ids": component["su10_node_ids"],
            "rigid_contract_ids": component["rigid_contract_ids"],
            "png": str(png_path),
        }
        rows.append(row)
        pngs.append(str(png_path))
    return rows, pngs


def run_rigid_mcts(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    iterations: int = 16,
    candidate_k: int = 16,
    max_steps: int = 128,
    beam_width: int = 1,
    c_puct: float = 1.414,
    seed: int | None = 7,
) -> Dict[str, Any]:
    """Run and audit only the prescribed aromatic rigid connections."""
    nodes_path = Path(nodes_csv).resolve()
    topology_path = Path(topology_specs_csv).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else nodes_path.parent / "rl_mtcs_rigid"
    )
    steps_dir = directory / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    config = MCTSConfig(
        iterations=iterations,
        candidate_k=candidate_k,
        rigid_max_steps=max_steps,
        beam_width=beam_width,
        c_puct=c_puct,
        seed=seed,
    )
    runner = MCTSRunner(nodes_path, config=config, topology_specs_csv=topology_path)
    initial_state = runner.initial_state.clone()
    rigid_contracts = [
        contract for contract in initial_state.problem.contracts
        if contract.kind == "rigid"
    ]
    aromatic_node_ids = {
        int(node.node_id)
        for unit in initial_state.problem.units
        if unit.phase == "aromatic"
        for node in unit.nodes
    }
    initial_png = directory / "rigid_initial.png"
    save_state_png(
        initial_state,
        initial_png,
        title=(
            f"Rigid MCTS initial state | aromatic clusters={len(initial_state.placements)} "
            f"nodes={len(initial_state.node_axial)} components={initial_state.component_count()}"
        ),
    )

    payload = runner.run_rigid_stage()
    actions = list(payload["best_candidate"].action_sequence)
    replay_state = initial_state.clone()
    connection_rows = []
    step_pngs = []
    replay_errors = []
    for step, action in enumerate(actions, start=1):
        before = replay_state.clone()
        ok, reason = replay_state.connect(copy.deepcopy(action))
        if not ok:
            replay_errors.append(
                f"replay_failed:step={step}:contract={action.get('contract_id', '')}:reason={reason}"
            )
            break
        row = _connection_row(step, action, before, replay_state)
        connection_rows.append(row)
        step_png = steps_dir / f"rigid_step_{step:02d}.png"
        connected_node_ids = replay_state.component_node_ids(
            str(action["source_unit_id"])
        )
        save_state_png(
            replay_state,
            step_png,
            title=(
                f"Rigid MCTS step {step}/{len(rigid_contracts)} | "
                f"contract={row['contract_id']} | "
                f"SU10 GIDs {row['source_node_id']}-{row['target_node_id']} | "
                f"components={row['component_count_after_step']}"
            ),
            visible_node_ids=connected_node_ids,
        )
        step_pngs.append(str(step_png))

    final_state = replay_state
    final_audit = final_state.audit()
    final_png = directory / "rigid_final.png"
    save_state_png(
        final_state,
        final_png,
        title=(
            f"Rigid MCTS final state | contracts={len(final_state.satisfied_contracts & {contract.contract_id for contract in rigid_contracts})}"
            f"/{len(rigid_contracts)} nodes={len(final_state.node_axial)} "
            f"components={final_state.component_count()}"
        ),
    )
    cluster_rows, cluster_pngs = _final_component_rows(
        final_state, directory / "rigid_clusters"
    )

    expected_contract_ids = {str(contract.contract_id) for contract in rigid_contracts}
    completed_contract_ids = expected_contract_ids & final_state.satisfied_contracts
    pending_contract_ids = expected_contract_ids - final_state.satisfied_contracts
    expected_component_count = initial_state.component_count() - len(rigid_contracts)
    rigid_endpoint_ids = {
        int(node_id)
        for contract in rigid_contracts
        for node_id in (contract.source_node_id, contract.target_node_id)
    }
    initial_su_map = initial_state.current_node_map()
    final_su_map = final_state.current_node_map()
    placed_non_aromatic = sorted(
        unit_id
        for unit_id in final_state.placements
        if final_state.unit(unit_id).phase != "aromatic"
    )
    validation_errors = list(replay_errors)
    initial_aromatic_types = {
        int(initial_su_map[node_id]) for node_id in aromatic_node_ids
    }
    if initial_aromatic_types - {12, 13}:
        validation_errors.append(
            f"initial_aromatic_su_types:{sorted(initial_aromatic_types)}:expected_only_12_13"
        )
    activated_rigid_ids = {
        node_id for node_id in aromatic_node_ids
        if int(final_su_map[node_id]) == 10
    }
    if activated_rigid_ids != rigid_endpoint_ids:
        validation_errors.append(
            "activated_su10_nodes_do_not_match_rigid_endpoints:"
            f"actual={sorted(activated_rigid_ids)}:expected={sorted(rigid_endpoint_ids)}"
        )
    unexpected_rigid_stage_types = sorted(
        {
            int(final_su_map[node_id])
            for node_id in aromatic_node_ids
        } - {10, 12, 13}
    )
    if unexpected_rigid_stage_types:
        validation_errors.append(
            "premature_aromatic_substitution_types:"
            + ",".join(str(value) for value in unexpected_rigid_stage_types)
        )
    if pending_contract_ids:
        validation_errors.append(
            "pending_rigid_contracts:" + ",".join(sorted(pending_contract_ids))
        )
    if len(actions) != len(rigid_contracts):
        validation_errors.append(
            f"rigid_action_count:{len(actions)}:expected={len(rigid_contracts)}"
        )
    if set(final_state.node_axial) != aromatic_node_ids:
        validation_errors.append(
            f"placed_node_set_is_not_aromatic_only:{len(final_state.node_axial)}:expected={len(aromatic_node_ids)}"
        )
    if placed_non_aromatic:
        validation_errors.append(
            "non_aromatic_units_placed:" + ",".join(placed_non_aromatic)
        )
    if final_audit.get("duplicate_axials"):
        validation_errors.append("duplicate_axial_coordinates")
    if final_audit.get("non_hex_edges"):
        validation_errors.append("non_hexagonal_edges")
    if final_state.component_count() != expected_component_count:
        validation_errors.append(
            f"component_count:{final_state.component_count()}:expected={expected_component_count}"
        )
    if final_state.state_signature() != payload["state"].state_signature():
        validation_errors.append("replayed_state_differs_from_mcts_result")
    if len(cluster_rows) != final_state.component_count():
        validation_errors.append(
            f"rigid_cluster_image_count:{len(cluster_rows)}:expected={final_state.component_count()}"
        )
    if final_audit.get("geometry_fallback_contract_ids"):
        validation_errors.append("geometry_fallback_used")
    if any(not row["hex_neighbor"] for row in connection_rows):
        validation_errors.append("rigid_endpoints_are_not_hex_neighbors")

    report: Dict[str, Any] = {
        "ok": not validation_errors,
        "stage": "rigid",
        "scope": "rigid_only_no_flex_no_side_no_force_finish",
        "inputs": {
            "nodes_csv": str(nodes_path),
            "topology_specs_csv": str(topology_path),
        },
        "config": {
            "iterations": config.iterations,
            "candidate_k": config.candidate_k,
            "max_steps": config.rigid_max_steps,
            "beam_width": config.beam_width,
            "c_puct": config.c_puct,
            "seed": config.seed,
        },
        "summary": {
            "aromatic_cluster_count": sum(
                1 for unit in initial_state.problem.units if unit.phase == "aromatic"
            ),
            "placed_node_count": len(final_state.node_axial),
            "rigid_contract_count": len(rigid_contracts),
            "rigid_action_count": len(actions),
            "rigid_contracts_done": len(completed_contract_ids),
            "rigid_contracts_pending": len(pending_contract_ids),
            "initial_component_count": initial_state.component_count(),
            "final_component_count": final_state.component_count(),
            "expected_final_component_count": expected_component_count,
            "final_rigid_cluster_count": len(cluster_rows),
            "rigid_connected_cluster_count": sum(
                row["cluster_kind"] == "rigid_connected_aromatic_cluster"
                for row in cluster_rows
            ),
            "standalone_aromatic_cluster_count": sum(
                row["cluster_kind"] == "standalone_aromatic_cluster"
                for row in cluster_rows
            ),
            "geometry_valid": bool(final_audit.get("geometry_valid", False)),
            "activated_su10_node_count": len(activated_rigid_ids),
            "initial_aromatic_su_types": sorted(initial_aromatic_types),
            "final_aromatic_su_types": sorted({
                int(final_su_map[node_id]) for node_id in aromatic_node_ids
            }),
        },
        "mcts_result": dict(payload.get("result", {})),
        "connections": connection_rows,
        "action_sequence": copy.deepcopy(actions),
        "final_audit": final_audit,
        "validation": {
            "ok": not validation_errors,
            "errors": validation_errors,
            "completed_rigid_contract_ids": sorted(completed_contract_ids),
            "pending_rigid_contract_ids": sorted(pending_contract_ids),
            "placed_non_aromatic_unit_ids": placed_non_aromatic,
        },
        "outputs": {
            "directory": str(directory),
            "initial_png": str(initial_png),
            "step_pngs": step_pngs,
            "final_png": str(final_png),
            "clusters_directory": str(directory / "rigid_clusters"),
            "cluster_pngs": cluster_pngs,
            "clusters_csv": str(directory / "rigid_clusters.csv"),
            "connections_csv": str(directory / "rigid_connections.csv"),
            "molecule_json": str(directory / "rigid_molecule.json"),
            "audit_json": str(directory / "rigid_audit.json"),
            "report_json": str(directory / "rigid_report.json"),
        },
    }
    _write_csv(directory / "rigid_connections.csv", connection_rows)
    _write_csv(directory / "rigid_clusters.csv", cluster_rows)
    _write_json(directory / "rigid_molecule.json", final_state.to_dict())
    _write_json(directory / "rigid_audit.json", final_audit)
    _write_json(directory / "rigid_report.json", report)
    return report


def print_rigid_report(report: Mapping[str, Any]) -> None:
    """Print a compact rigid-stage report for scripts and notebooks."""
    summary = dict(report.get("summary", {}))
    print("[RL_MTCS rigid-only MCTS]")
    print(f"ok={bool(report.get('ok', False))}")
    for key in (
        "aromatic_cluster_count",
        "placed_node_count",
        "rigid_contract_count",
        "rigid_action_count",
        "rigid_contracts_done",
        "rigid_contracts_pending",
        "initial_component_count",
        "final_component_count",
        "expected_final_component_count",
        "final_rigid_cluster_count",
        "rigid_connected_cluster_count",
        "standalone_aromatic_cluster_count",
        "geometry_valid",
        "activated_su10_node_count",
        "initial_aromatic_su_types",
        "final_aromatic_su_types",
    ):
        print(f"{key}={summary.get(key)}")
    print("\n[connections]")
    for row in report.get("connections", []):
        print(
            f"  step={row['step']:02d} contract={row['contract_id']} "
            f"nodes={row['source_node_id']}-{row['target_node_id']} "
            f"direction=({row['direction_q']},{row['direction_r']}) "
            f"components={row['component_count_after_step']}"
        )
    for error in report.get("validation", {}).get("errors", []):
        print(f"ERROR: {error}")
    outputs = dict(report.get("outputs", {}))
    print(f"\noutputs={outputs.get('directory', '')}")
    print(f"rigid_clusters={outputs.get('clusters_directory', '')}")
    print(f"rigid_clusters_csv={outputs.get('clusters_csv', '')}")




def run_flex_mcts(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    rigid_iterations: int = 16,
    flex_iterations: int = 4,
    candidate_k: int = 8,
    max_steps: int = 128,
    beam_width: int = 1,
    c_puct: float = 1.414,
    seed: int | None = 7,
) -> Dict[str, Any]:
    """Run through flex only and save an auditable partial or complete state."""
    nodes_path = Path(nodes_csv).resolve()
    topology_path = Path(topology_specs_csv).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else nodes_path.parent / "rl_mtcs_flex"
    )
    directory.mkdir(parents=True, exist_ok=True)

    runner = MCTSRunner(
        nodes_path,
        config=MCTSConfig(
            iterations=rigid_iterations,
            flex_iterations=flex_iterations,
            candidate_k=candidate_k,
            flex_max_steps=max_steps,
            beam_width=beam_width,
            c_puct=c_puct,
            seed=seed,
        ),
        topology_specs_csv=topology_path,
    )
    rigid = runner.run_rigid_stage()
    flex = runner.run_flex_stage(rigid)
    state = flex["state"]
    summary = dict(flex["result"])
    audit = state.audit()
    plan = dict(flex["flex_plan"])

    png_path = directory / "flex_result.png"
    save_state_png(
        state,
        png_path,
        title=(
            "Flex MCTS result | "
            f"backbone={summary.get('backbone_contracts_done', 0)}/"
            f"{summary.get('backbone_contracts', 0)} "
            f"interior={summary.get('interior_contracts_done', 0)}/"
            f"{summary.get('interior_contracts', 0)} "
            f"components={audit.get('component_count')}"
        ),
    )

    ownership = state.problem.ownership_map()
    slot_assignments = {
        int(boundary_id): int(site_id)
        for boundary_id, site_id in dict(
            summary.get("slot_geometry", {}).get(
                "boundary_site_assignments", {}
            )
        ).items()
    }
    contract_rows = []
    for contract in state.problem.contracts:
        if contract.kind != "flex":
            continue
        if state.unit(contract.source_unit_id).phase == "aromatic":
            boundary_node_id = int(contract.source_node_id)
        else:
            boundary_node_id = int(contract.target_node_id)
        contract_rows.append({
            "contract_id": str(contract.contract_id),
            "satisfied": str(contract.contract_id) in state.satisfied_contracts,
            "source_unit_id": str(contract.source_unit_id),
            "source_node_id": int(contract.source_node_id),
            "target_unit_id": str(contract.target_unit_id),
            "target_node_id": int(contract.target_node_id),
            "source_owner": ownership.get(int(contract.source_node_id), ""),
            "target_owner": ownership.get(int(contract.target_node_id), ""),
            "aromatic_boundary_node_id": boundary_node_id,
            "selected_physical_site_id": slot_assignments.get(
                boundary_node_id, ""
            ),
            "selected_aromatic_unit_id": ownership.get(
                boundary_node_id, ""
            ),
            "boundary_axial": list(
                state.node_axial.get(boundary_node_id, ())
            ),
            "backbone_contract": str(contract.contract_id)
            in set(plan.get("backbone_contract_ids", [])),
            "interior_contract": str(contract.contract_id)
            in set(plan.get("interior_contract_ids", [])),
        })

    action_rows = []
    for step, action in enumerate(flex["best_candidate"].action_sequence, start=1):
        contract_ids = [
            str(value) for value in action.get("contract_ids", [])
        ]
        if not contract_ids and action.get("contract_id"):
            contract_ids = [str(action["contract_id"])]
        action_rows.append({
            "step": int(step),
            "contract_id": contract_ids[0] if len(contract_ids) == 1 else "",
            "contract_ids": contract_ids,
            "topology_unit_id": str(action.get("topology_unit_id", "")),
            "move_component_unit_id": str(
                action.get("move_component_unit_id", "")
            ),
            "topology_reembedded": bool(action.get("topology_reembedded", False)),
            "planned_backbone_merge": bool(
                action.get("planned_flex_backbone_merge", False)
            ),
            "planned_internal_closure": bool(
                action.get("planned_internal_closure", False)
            ),
            "attachment_contract_id": str(
                action.get("attachment_contract_id", "")
            ),
            "merge_contract_ids": list(
                action.get("merge_contract_ids", [])
            ),
            "boundary_site_assignments": dict(
                action.get("boundary_site_assignments", {})
            ),
            "topology_rotation": action.get("topology_rotation", ""),
            "topology_reflection": action.get("topology_reflection", ""),
            "topology_translation": action.get("topology_translation", ""),
            "component_moves": list(action.get("component_moves", [])),
            "action": dict(action),
        })

    _write_json(directory / "flex_summary.json", summary)
    _write_json(directory / "flex_plan.json", plan)
    _write_json(directory / "flex_audit.json", audit)
    _write_json(directory / "flex_molecule.json", state.to_dict())
    _write_csv(directory / "flex_contracts.csv", contract_rows)
    _write_csv(directory / "flex_actions.csv", action_rows)

    result = {
        "ok": bool(summary.get("done", False)),
        "state": state,
        "rigid": dict(rigid["result"]),
        "flex": summary,
        "audit": audit,
        "plan": plan,
        "outputs": {
            "directory": str(directory),
            "png": str(png_path),
            "summary_json": str(directory / "flex_summary.json"),
            "plan_json": str(directory / "flex_plan.json"),
            "audit_json": str(directory / "flex_audit.json"),
            "molecule_json": str(directory / "flex_molecule.json"),
            "contracts_csv": str(directory / "flex_contracts.csv"),
            "actions_csv": str(directory / "flex_actions.csv"),
        },
    }
    return result




def run_side_mcts(
    nodes_csv: str | Path,
    topology_specs_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    rigid_iterations: int = 16,
    flex_iterations: int = 4,
    side_iterations: int = 4,
    candidate_k: int = 2,
    max_steps: int = 128,
    beam_width: int = 1,
    c_puct: float = 1.414,
    seed: int | None = 7,
) -> Dict[str, Any]:
    """Run all connection stages and save the completed side-chain state."""
    nodes_path = Path(nodes_csv).resolve()
    topology_path = Path(topology_specs_csv).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else nodes_path.parent / "rl_mtcs_side"
    )
    directory.mkdir(parents=True, exist_ok=True)

    runner = MCTSRunner(
        nodes_path,
        config=MCTSConfig(
            iterations=rigid_iterations,
            flex_iterations=flex_iterations,
            side_iterations=side_iterations,
            candidate_k=candidate_k,
            flex_max_steps=max_steps,
            side_max_steps=max_steps,
            beam_width=beam_width,
            c_puct=c_puct,
            seed=seed,
        ),
        topology_specs_csv=topology_path,
    )
    rigid = runner.run_rigid_stage()
    flex = runner.run_flex_stage(rigid)
    side = runner.run_side_stage(flex)
    state = side["state"]
    summary = dict(side["result"])
    audit = state.audit()

    png_path = directory / "side_result.png"
    save_state_png(
        state,
        png_path,
        title=(
            "Complete side-chain result | "
            f"side={summary.get('contracts_done', 0)}/"
            f"{summary.get('contracts_total', 0)} "
            f"nodes={audit.get('placed_node_count')}/"
            f"{audit.get('expected_node_count')}"
        ),
    )

    assignments = {
        int(boundary_id): int(site_id)
        for boundary_id, site_id in dict(
            summary.get("slot_geometry", {}).get(
                "boundary_site_assignments", {}
            )
        ).items()
    }
    ownership = state.problem.ownership_map()
    contract_rows = []
    for contract in state.problem.contracts:
        if contract.kind != "side":
            continue
        source = state.unit(contract.source_unit_id)
        if source.phase == "aromatic":
            boundary_id = int(contract.source_node_id)
            topology_id = int(contract.target_node_id)
        else:
            boundary_id = int(contract.target_node_id)
            topology_id = int(contract.source_node_id)
        contract_rows.append({
            "contract_id": str(contract.contract_id),
            "satisfied": str(contract.contract_id) in state.satisfied_contracts,
            "topology_node_id": topology_id,
            "aromatic_boundary_node_id": boundary_id,
            "selected_physical_site_id": assignments.get(boundary_id, ""),
            "selected_aromatic_unit_id": ownership.get(boundary_id, ""),
            "boundary_axial": list(state.node_axial.get(boundary_id, ())),
            "topology_axial": list(state.node_axial.get(topology_id, ())),
        })

    action_rows = []
    for step, action in enumerate(
        side["best_candidate"].action_sequence, start=1
    ):
        boundary_id = int(action.get("boundary_node_id", -1))
        action_rows.append({
            "step": int(step),
            "contract_id": str(action.get("contract_id", "")),
            "topology_unit_id": str(action.get("topology_unit_id", "")),
            "topology_node_id": int(action.get("topology_node_id", -1)),
            "aromatic_boundary_node_id": boundary_id,
            "selected_physical_site_id": int(
                action.get("physical_site_id", -1)
            ),
            "final_boundary_axial": list(
                state.node_axial.get(boundary_id, ())
            ),
            "topology_rotation": int(action.get("topology_rotation", 0)),
            "topology_reflection": bool(
                action.get("topology_reflection", False)
            ),
            "topology_translation": list(
                action.get("topology_translation", ())
            ),
            "allowed_nonbonded_contact_count": int(
                action.get("allowed_nonbonded_contact_count", 0)
            ),
            "action": dict(action),
        })

    _write_json(directory / "side_summary.json", summary)
    _write_json(directory / "side_audit.json", audit)
    _write_json(directory / "side_molecule.json", state.to_dict())
    _write_csv(directory / "side_contracts.csv", contract_rows)
    _write_csv(directory / "side_actions.csv", action_rows)
    final_outputs = export_final_state(state, directory)

    return {
        "ok": bool(summary.get("done", False) and audit.get("complete", False)),
        "state": state,
        "rigid": dict(rigid["result"]),
        "flex": dict(flex["result"]),
        "side": summary,
        "audit": audit,
        "outputs": {
            "directory": str(directory),
            "side_png": str(png_path),
            "side_summary_json": str(directory / "side_summary.json"),
            "side_audit_json": str(directory / "side_audit.json"),
            "side_molecule_json": str(directory / "side_molecule.json"),
            "side_contracts_csv": str(directory / "side_contracts.csv"),
            "side_actions_csv": str(directory / "side_actions.csv"),
            "summary_json": str(directory / "side_summary.json"),
            "contracts_csv": str(directory / "side_contracts.csv"),
            "actions_csv": str(directory / "side_actions.csv"),
            **final_outputs,
        },
    }




def main() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fixed-topology hex-grid coarse-molecule assembly"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inverse-json", help="Complete inverse topology JSON export")
    source.add_argument("--nodes-csv", help="final_nodes_by_global_id.csv")
    parser.add_argument("--topology-specs-csv", help="final_topology_specs.csv (required with --nodes-csv)")
    parser.add_argument(
        "--stage",
        choices=("init", "rigid", "flex", "all"),
        default="all",
    )
    parser.add_argument("--rigid-iterations", "--iterations", type=int, default=16)
    parser.add_argument("--flex-iterations", type=int, default=4)
    parser.add_argument("--side-iterations", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--substitution-beam-width", type=int, default=4)
    parser.add_argument("--substitution-candidate-k", type=int, default=32)
    parser.add_argument("--substitution-nmr-weight", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.414)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.nodes_csv and not args.topology_specs_csv:
        parser.error("--topology-specs-csv is required with --nodes-csv")
    if args.inverse_json:
        if args.stage != "all":
            parser.error("--inverse-json currently supports only --stage all")
        result = MCTSRunner(
            args.inverse_json,
            config=MCTSConfig(
                iterations=args.rigid_iterations,
                flex_iterations=args.flex_iterations,
                side_iterations=args.side_iterations,
                candidate_k=args.candidate_k,
                beam_width=args.beam_width,
                rigid_max_steps=args.max_steps,
                flex_max_steps=args.max_steps,
                side_max_steps=args.max_steps,
                c_puct=args.c_puct,
                seed=args.seed,
                substitution_beam_width=args.substitution_beam_width,
                substitution_candidate_k=args.substitution_candidate_k,
                substitution_nmr_weight=args.substitution_nmr_weight,
            ),
        ).run_all_stages(args.output)
    elif args.stage == "init":
        result = inspect_csv_initialization(
            args.nodes_csv, args.topology_specs_csv, args.output
        )
    elif args.stage == "rigid":
        result = run_rigid_mcts(
            args.nodes_csv,
            args.topology_specs_csv,
            args.output,
            iterations=args.rigid_iterations,
            candidate_k=args.candidate_k,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            c_puct=args.c_puct,
            seed=args.seed,
        )
    elif args.stage == "flex":
        result = run_flex_mcts(
            args.nodes_csv,
            args.topology_specs_csv,
            args.output,
            rigid_iterations=args.rigid_iterations,
            flex_iterations=args.flex_iterations,
            candidate_k=args.candidate_k,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            c_puct=args.c_puct,
            seed=args.seed,
        )
    elif args.stage == "all":
        result = MCTSRunner(
            args.nodes_csv,
            config=MCTSConfig(
                iterations=args.rigid_iterations,
                flex_iterations=args.flex_iterations,
                side_iterations=args.side_iterations,
                candidate_k=args.candidate_k,
                beam_width=args.beam_width,
                rigid_max_steps=args.max_steps,
                flex_max_steps=args.max_steps,
                side_max_steps=args.max_steps,
                c_puct=args.c_puct,
                seed=args.seed,
                substitution_beam_width=args.substitution_beam_width,
                substitution_candidate_k=args.substitution_candidate_k,
                substitution_nmr_weight=args.substitution_nmr_weight,
            ),
            topology_specs_csv=args.topology_specs_csv,
        ).run_all_stages(args.output)
    else:
        result = run_side_mcts(
            args.nodes_csv,
            args.topology_specs_csv,
            args.output,
            rigid_iterations=args.rigid_iterations,
            flex_iterations=args.flex_iterations,
            side_iterations=args.side_iterations,
            candidate_k=args.candidate_k,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            c_puct=args.c_puct,
            seed=args.seed,
        )
    print(json.dumps(
        {
            "ok": result.get("ok", result.get("status") == "complete"),
            "audit": result.get("audit", {}),
            "outputs": result.get("outputs", {}),
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    return result


if __name__ == "__main__":
    main()
