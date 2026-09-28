"""Inverse-driven Monte-Carlo topology assembly entry point.

Placement uses immutable aromatic clusters and topology products supplied by
:func:`RL_MTCS.input.initialize`; this module does not allocate or synthesize
topology units.  Rigid SU10 partners are chosen dynamically. Final
heteroaromatic SU26/SU30 substitution is applied only
after rigid, flex, and side placement have completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .input import initialize, inspect_csv_initialization
from .state import HexAssemblyState, stage_snapshot
from .models import AssemblyConfig, AssemblySearchError, InverseInputError
from .search import SearchGuidance
from .rigid import (
    SearchResult,
    TopologyPlacementStage,
    search_phase,
    _rigid_unit_components,
    _rigid_component_metrics,
)
from .output import export_final_state
from .flex import run_planned_flex_search
from .side import run_planned_side_search
from .substitution import run_substitution_stage
from .output import export_rigid_views, save_state_png


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


@dataclass
class BeamCandidate:
    """Compatibility wrapper for callers that consumed the old beam result."""

    state: HexAssemblyState
    score: float
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _json_default(value):
    # Molecules already have a dedicated export. Avoid serializing full state
    # reprs once per beam candidate inside final.json.
    if isinstance(value, BeamCandidate):
        return {"score": value.score, "action_count": len(value.action_sequence),
                "contracts_done": len(value.state.satisfied_contracts)}
    return str(value)


def _candidate_from_result(result: SearchResult) -> BeamCandidate:
    return BeamCandidate(
        state=result.state,
        score=float(result.score),
        action_sequence=list(result.action_sequence),
        info=copy.deepcopy({k: v for k, v in result.info.items() if k != 'stage_states'}),
    )


class MCTSRunner:
    """Run inverse topology placement: rigid -> flex -> side -> substitution."""

    def __init__(
        self,
        inverse_result: Any,
        config: AssemblyConfig | None = None,
        topology_specs_csv: Any = None,
        search_guidance: SearchGuidance | None = None,
    ):
        self.inverse_result = inverse_result
        self.topology_specs_csv = topology_specs_csv
        self.config = config if config is not None else AssemblyConfig()
        self.search_guidance = search_guidance
        self.init_data = initialize(inverse_result, topology_specs_csv)
        self.initial_state: HexAssemblyState = self.init_data["state"].clone()
        self._print_input_summary()
        self.nmr_scorer = self._initialize_nmr_scorer()

    def _print_input_summary(self) -> None:
        """Print the immutable inverse topology inventory once per run.

        This is deliberately based on the parsed CSV objects rather than on
        search results, so the report remains useful when the first stage
        stops before making any progress.
        """
        if not bool(getattr(self.config, "verbose", True)):
            return
        units = list(self.init_data.get("units", ()))
        contracts = list(self.init_data.get("contracts", ()))
        unit_counts: Dict[str, int] = {}
        for unit in units:
            key = str(getattr(unit, "product_type", "aromatic"))
            unit_counts[key] = unit_counts.get(key, 0) + 1
        contract_counts: Dict[str, int] = {}
        for contract in contracts:
            key = str(getattr(contract, "kind", "unknown"))
            contract_counts[key] = contract_counts.get(key, 0) + 1
        print(
            "[RL_MTCS][input] "
            f"units={len(units)} "
            f"aromatic={sum(getattr(unit, 'phase', '') == 'aromatic' for unit in units)} "
            f"flex_chain={unit_counts.get('flex_chain', 0)} "
            f"aliphatic_ring={unit_counts.get('aliphatic_ring', 0)} "
            f"side_chain={unit_counts.get('side_chain', 0)}",
            flush=True,
        )
        print(
            "[RL_MTCS][input] "
            f"rigid_mode=dynamic_su10_sites "
            f"rigid_ports={sum(su == 10 for _node_id, su in self.initial_state.problem.node_su_types)} "
            f"source_pairs={len(self.initial_state.problem.source_rigid_contracts)} "
            f"flex:{contract_counts.get('flex', 0)} "
            f"side:{contract_counts.get('side', 0)}",
            flush=True,
        )

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
        try:
            from model.forward.g2s.rl_mtcs_utils import NMRGuidanceScorer
            return NMRGuidanceScorer(
                spectrum_csv=spectrum_csv,
                elements=elements,
                model_checkpoint=checkpoint,
                device=str(getattr(self.config, "nmr_device", "auto")),
                batch_size=int(getattr(self.config, "nmr_batch_size", 512)),
                hwhm=float(getattr(self.config, "nmr_hwhm", 1.0)),
                graph_cache_limit=int(getattr(self.config, "nmr_graph_cache_limit", 256)),
                node_cache_limit=int(getattr(self.config, "nmr_node_cache_limit", 2048)),
                record_limit=int(getattr(self.config, "nmr_record_limit", 5000)),
                model_call_limit=int(getattr(self.config, "nmr_model_call_limit", 1)),
                center_limit=int(getattr(self.config, "nmr_center_limit", 1)),
            )
        except ImportError as exc:
            raise InverseInputError(
                f"requested NMR guidance could not load its dependencies: {exc}"
            ) from exc

    def _run_phase(self, state: HexAssemblyState, phase: str) -> SearchResult:
        if phase == "rigid":
            iterations = self.config.rigid_iterations
            max_steps = self.config.max_steps
        elif phase == "flex":
            iterations = 1
            max_steps = self.config.max_steps
        else:
            iterations = 1
            max_steps = self.config.max_steps
        return search_phase(
            TopologyPlacementStage(state, phase),
            iterations=iterations,
            candidate_k=self.config.candidate_k,
            max_steps=max_steps,
            beam_width=self.config.beam_width,
            c_puct=self.config.c_puct,
            seed=self.config.seed,
            search_guidance=getattr(self, 'search_guidance', None),
            max_tree_nodes=self.config.max_tree_nodes,
        )

    @staticmethod
    def _stage_payload(result: SearchResult) -> Dict[str, Any]:
        candidates = result.candidate_states or [result.state]
        return {
            "state": result.state,
            "best_candidate": _candidate_from_result(result),
            "beam_candidates": [BeamCandidate(
                state, float(TopologyPlacementStage(state, result.info.get("phase", state.stage)).score()),
                list(state.action_history), {},
            ) for state in candidates],
            "summaries": [copy.deepcopy(result.info.get("result", {}))],
            "result": copy.deepcopy(result.info.get("result", {})),
            "molecule": result.state.to_dict(),
        }

    def run_rigid_stage(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        initial_views = None
        if output_dir and self.config.save_stage_images:
            initial_views = export_rigid_views(
                self.initial_state, Path(output_dir) / "aromatic_initial", initial=True
            )
        result = self._run_phase(self.initial_state.clone(), "rigid")
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "rigid")
        if output_dir:
            self._write_rigid_detail(payload, output_dir, initial_views)
        self._print_stage_summary("rigid", payload)
        return payload

    def _write_rigid_detail(self, payload: Dict[str, Any], output_dir: str,
                            initial_views: Optional[Dict[str, Any]] = None) -> None:
        state = payload["state"]
        directory = Path(output_dir)
        actual = [contract for contract in state.problem.contracts if contract.kind == "rigid"
                  and contract.contract_id in state.satisfied_contracts]
        source = state.problem.source_rigid_contracts
        source_pairs = {tuple(sorted((c.source_node_id, c.target_node_id))) for c in source}
        realized_pairs = {tuple(sorted((c.source_node_id, c.target_node_id))) for c in actual}
        directions: Dict[str, int] = {}
        axes: Dict[str, int] = {}
        for edge in state.edges:
            if edge.kind != "rigid" or edge.direction is None:
                continue
            direction = tuple(int(value) for value in edge.direction)
            axis = min(direction, (-direction[0], -direction[1]))
            directions[str(direction)] = directions.get(str(direction), 0) + 1
            axes[str(axis)] = axes.get(str(axis), 0) + 1
        metrics = _rigid_component_metrics(state, _rigid_unit_components(state))
        payload['result']['rigid_component_metrics'] = metrics
        sizes = sorted((len(metric['unit_ids']) for metric in metrics), reverse=True)
        target_sizes = sorted((sum(sizes) // len(sizes) + int(i < sum(sizes) % len(sizes))
                               for i in range(len(sizes))), reverse=True) if sizes else []
        comparison = {
            "pairing_mode": "dynamic_su10_sites",
            "source_pairs": [list(pair) for pair in sorted(source_pairs)],
            "realized_pairs": [list(pair) for pair in sorted(realized_pairs)],
            "retained_source_pairs": [list(pair) for pair in sorted(source_pairs & realized_pairs)],
            "replaced_source_pairs": [list(pair) for pair in sorted(source_pairs - realized_pairs)],
            "new_pairs": [list(pair) for pair in sorted(realized_pairs - source_pairs)],
            "source_is_constraint": False,
            "component_sizes": sizes,
            "target_component_sizes": target_sizes,
            "balance_deviation": sum(abs(a - b) for a, b in zip(sizes, target_sizes)),
            "actual_site_bindings": state.aromatic_site_bindings(),
            "balance_deviation_from_6_6_5": (
                sum(abs(actual_size - ideal) for actual_size, ideal in zip(sizes, (6, 6, 5)))
                if len(sizes) == 3 else None
            ),
            "direction_counts": directions,
            "axis_counts": axes,
        }
        comparison_path = directory / "rigid_pairing_comparison.json"
        self._write_json(comparison, str(comparison_path))
        outputs = payload.get("stage_outputs", {})
        outputs["pairing_comparison_json"] = str(comparison_path)
        if initial_views:
            outputs["aromatic_initial"] = initial_views
        if self.config.save_stage_images and actual:
            outputs["rigid_clusters"] = export_rigid_views(
                state, directory / "rigid_clusters"
            )
        payload["result"]["stage_outputs"] = outputs
        self._write_json(payload["result"], str(directory / "rigid_summary.json"))

    def run_flex_stage(self, rigid_result: Dict[str, Any], output_dir: Optional[str] = None,
                       *, search_budget: Optional[int] = None) -> Dict[str, Any]:
        if not bool(dict(rigid_result.get("result", {})).get("done", False)):
            return self._finish_failed_stage("flex", rigid_result, output_dir, None)
        candidates = [c.state for c in rigid_result.get("beam_candidates", [])]
        if not candidates:
            candidates = [self._state_from_result(rigid_result)]
        def search(state, budget, roots=None):
            return run_planned_flex_search(
                state, iterations=1,
                candidate_k=self.config.candidate_k, max_steps=self.config.max_steps,
                beam_width=self.config.beam_width, c_puct=self.config.c_puct,
                seed=self.config.seed, strict=False, nmr_scorer=self.nmr_scorer,
                nmr_weight=self.config.nmr_weight, geometry_weight=self.config.geometry_weight,
                contact_weight=self.config.contact_weight, search_budget=budget,
                conformer_limit=self.config.conformer_limit,
                rigid_candidates=roots,
                search_guidance=getattr(self, 'search_guidance', None),
                max_tree_nodes=self.config.max_tree_nodes,
                domain_check_limit=self.config.domain_check_limit,
                domain_pose_limit=self.config.domain_pose_limit,
                max_domain_check_limit=self.config.max_domain_check_limit,
                max_domain_pose_limit=self.config.max_domain_pose_limit,
            )
        result, plan = search(candidates[0], search_budget or self.config.search_budget, candidates)
        self._write_flex_checkpoints(result.info.get('stage_states', {}), result.info['result'], output_dir)
        index = result.info['result'].get('selected_rigid_candidate_index', 0)
        payload = self._stage_payload(result)
        payload["flex_plan"] = plan.to_dict()
        payload['selected_rigid_state'] = candidates[index]
        self._write_stage_output(payload, output_dir, "flex")
        if output_dir:
            selected_dir = Path(output_dir).parent / "rigid" / "selected_by_flex"
            selected = payload['selected_rigid_state']
            if self.config.save_stage_images:
                payload['result']['selected_rigid_views'] = export_rigid_views(
                    selected, selected_dir
                )
            else:
                selected_dir.mkdir(parents=True, exist_ok=True)
            self._write_json({
                "selected_rigid_candidate_index": index,
                "flex_complete": bool(payload['result'].get('done')),
                "realized_pairs": sorted([
                    sorted((c.source_node_id, c.target_node_id))
                    for c in selected.problem.contracts
                    if c.kind == "rigid" and c.contract_id in selected.satisfied_contracts
                ]),
            }, str(selected_dir / "selection.json"))
            self._write_json(payload['result'], str(Path(output_dir) / "flex_summary.json"))
        if not payload['result'].get('done'):
            payload['result']['partial'] = True
        self._print_stage_summary("flex", payload)
        return payload

    def _write_flex_checkpoints(self, states, summary_row, output_dir):
        for name, snapshot in states.items():
            units = [u for u in snapshot.problem.units if u.product_type == name]
            ids = {u.unit_id for u in units}
            contracts = [c for c in snapshot.problem.contracts
                         if c.source_unit_id in ids or c.target_unit_id in ids]
            done = (all(snapshot.is_placed(u.unit_id) for u in units)
                    and all(c.contract_id in snapshot.satisfied_contracts for c in contracts))
            summary = {'phase': name, 'done': done, 'partial': not done,
                       'status': 'complete' if done else summary_row['status'],
                       'placed_units': sum(snapshot.is_placed(u.unit_id) for u in units),
                       'phase_units_total': len(units),
                       'contracts_done': sum(c.contract_id in snapshot.satisfied_contracts for c in contracts),
                       'contracts_total': len(contracts),
                       'selected_rigid_candidate_index': summary_row.get('selected_rigid_candidate_index', 0)}
            checkpoint = {'state': snapshot, 'molecule': snapshot.to_dict(), 'result': summary}
            self._write_stage_output(checkpoint, str(Path(output_dir).parent / name) if output_dir else None, name)
            self._print_stage_summary(name, checkpoint)

    def run_side_stage(self, flex_result: Dict[str, Any], output_dir: Optional[str] = None,
                       *, search_budget: Optional[int] = None) -> Dict[str, Any]:
        if not bool(dict(flex_result.get("result", {})).get("done", False)):
            return self._finish_failed_stage("side", flex_result, output_dir, None)
        state = self._state_from_result(flex_result)
        result = run_planned_side_search(
            state,
            iterations=1,
            candidate_k=self.config.candidate_k,
            max_steps=self.config.max_steps,
            beam_width=self.config.beam_width,
            c_puct=self.config.c_puct,
            seed=self.config.seed,
            strict=False,
            nmr_scorer=self.nmr_scorer,
            nmr_weight=float(getattr(self.config, "nmr_weight", 1.0)),
            geometry_weight=float(getattr(self.config, "geometry_weight", 0.05)),
            contact_weight=float(getattr(self.config, "contact_weight", 0.02)),
            search_budget=search_budget or self.config.search_budget,
            search_guidance=getattr(self, 'search_guidance', None),
            max_tree_nodes=self.config.max_tree_nodes,
        )
        payload = self._stage_payload(result)
        self._write_stage_output(payload, output_dir, "side")
        if not payload['result'].get('done'):
            payload['result']['partial'] = True
        self._print_stage_summary("side", payload)
        return payload

    def run_substitution_stage(self, side_result: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        if not bool(dict(side_result.get("result", {})).get("done", False)):
            return self._finish_failed_stage(
                "substitution", side_result, output_dir, None
            )
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
        self._print_stage_summary("substitution", payload)
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
        try:
            rigid = self.run_rigid_stage(str(base / "rigid") if base else None)
        except AssemblySearchError as exc:
            return self._finish_failed_stage("rigid", {"result": {
                "status": "search_failed", "failure_scope": "rigid",
                "planning_errors": [str(exc)], "diagnostics": exc.diagnostics,
            }}, str(base / "rigid") if base else None, base)
        if not rigid["result"].get("done", False):
            return self._finish_failed_stage(
                "rigid", rigid, str(base / "rigid") if base else None, base
            )
        try:
            flex = self.run_flex_stage(rigid, str(base / "flex") if base else None)
        except (AssemblySearchError, InverseInputError) as exc:
            return self._finish_failed_stage("flex", {
                "state": rigid.get("state"),
                "result": {
                    "status": "candidate_space_exhausted",
                    "failure_scope": "flex",
                    "planning_errors": [str(exc)],
                    "partial": True,
                },
            }, str(base / "flex") if base else None, base)
        if not flex["result"].get("done", False):
            return self._finish_failed_stage(
                "flex", flex, str(base / "flex") if base else None, base
            )
        retained_rigid = list(rigid.get('beam_candidates', []))
        selected_rigid = flex.get('selected_rigid_state', rigid['state'])
        if selected_rigid is not rigid['state']:
            rigid = self._stage_payload(SearchResult(
                selected_rigid, float(TopologyPlacementStage(selected_rigid, 'rigid').score()),
                list(selected_rigid.action_history),
                {'phase': 'rigid', 'result': TopologyPlacementStage(selected_rigid, 'rigid').get_result()},
            ))
            self._write_stage_output(rigid, str(base / 'rigid') if base else None, 'rigid')
            if base:
                self._write_rigid_detail(rigid, str(base / 'rigid'))
        try:
            side = self.run_side_stage(flex, str(base / "side") if base else None)
        except (AssemblySearchError, InverseInputError) as exc:
            return self._finish_failed_stage("side", {
                "state": flex.get("state"),
                "result": {
                    "status": "candidate_space_exhausted",
                    "failure_scope": "side",
                    "planning_errors": [str(exc)],
                    "partial": True,
                },
            }, str(base / "side") if base else None, base)
        if not side["result"].get("done", False):
            # A side dead end can be caused by early ring/chain placement.
            # Retry bounded retained rigid alternatives; never reallocate topology.
            feedback = []
            alternatives = [c for c in retained_rigid
                            if c.state.state_signature() != selected_rigid.state_signature()]
            for index, candidate in enumerate(alternatives[:min(self.config.stage_retry_limit, self.config.search_budget)]):
                retry_dir = base / 'retries' / f'candidate_{index + 1:02d}' if base else None
                retry_rigid = {'state': candidate.state, 'beam_candidates': [candidate],
                               'result': {'done': True}}
                retry_budget = max(1, self.config.search_budget // max(1, self.config.stage_retry_limit))
                trial_flex = self.run_flex_stage(retry_rigid, str(retry_dir / 'flex') if retry_dir else None,
                                                 search_budget=retry_budget)
                if not trial_flex['result'].get('done'):
                    feedback.append({'candidate': index + 1, 'stage': 'flex', 'status': trial_flex['result']['status']})
                    continue
                trial_side = self.run_side_stage(trial_flex, str(retry_dir / 'side') if retry_dir else None,
                                                 search_budget=retry_budget)
                feedback.append({'candidate': index + 1, 'stage': 'side', 'status': trial_side['result']['status']})
                if trial_side['result'].get('done') or trial_side['result'].get('contracts_done', 0) > side['result'].get('contracts_done', 0):
                    flex, side, selected_rigid = trial_flex, trial_side, candidate.state
                    rigid = self._stage_payload(SearchResult(
                        selected_rigid, 0.0, list(selected_rigid.action_history),
                        {'phase': 'rigid', 'result': TopologyPlacementStage(selected_rigid, 'rigid').get_result()}))
                    self._write_stage_output(rigid, str(base / 'rigid') if base else None, 'rigid')
                    self._write_stage_output(flex, str(base / 'flex') if base else None, 'flex')
                    self._write_stage_output(side, str(base / 'side') if base else None, 'side')
                    if base:
                        self._write_rigid_detail(rigid, str(base / 'rigid'))
                        self._write_flex_checkpoints(
                            {name: stage_snapshot(flex['state'], name)
                             for name in ('flex_chain', 'aliphatic_ring')},
                            flex['result'], str(base / 'flex'))
                if side['result'].get('done'):
                    break
            if base:
                self._write_json(feedback, str(base / 'stage_backtracking.json'))
        if not side["result"].get("done", False):
            return self._finish_failed_stage(
                "side", side, str(base / "side") if base else None, base
            )
        try:
            substitution = self.run_substitution_stage(
                side, str(base / "substitution") if base else None
            )
        except (AssemblySearchError, InverseInputError) as exc:
            return self._finish_failed_stage("substitution", {
                "state": side.get("state"),
                "result": {
                    "status": "candidate_space_exhausted",
                    "failure_scope": "substitution",
                    "planning_errors": [str(exc)],
                    "partial": True,
                },
            }, str(base / "substitution") if base else None, base)
        if not substitution["result"].get("complete", False):
            return self._finish_failed_stage(
                "substitution", substitution,
                str(base / "substitution") if base else None, base
            )
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
        topology_complete = bool(
            final_state.is_complete() and audit.get("complete", False)
        )
        complete = bool(
            topology_complete
            and audit.get("geometry_valid", False)
        )
        result = {
            "ok": complete,
            "status": (
                "complete" if complete
                else "layout_invalid" if topology_complete
                else "incomplete"
            ),
            "topology_complete": topology_complete,
            "failure_scope": "none" if complete else "assembly",
            "candidate_space_complete": bool(
                flex["result"].get("candidate_space_complete", False)
            ),
            "feasibility_audit": flex["result"].get(
                "search_diagnostics", {}
            ).get("feasibility_audit", {}),
            "best_partial_state": flex["result"].get(
                "best_partial_state", {}
            ),
            "zero_domain_units": list(
                flex["result"].get("zero_domain_units", [])
            ),
            "completed_contract_count": int(
                len(final_state.satisfied_contracts)
            ),
            "stage": "substitution",
            "init": self.init_data["diagnostics"],
            "rigid": rigid["result"],
            "flex": flex["result"],
            "side": side["result"],
            "substitution": substitution["result"],
            "stage_outputs": {
                "rigid": dict(rigid.get("stage_outputs", {}) or {}),
                "flex": dict(flex.get("stage_outputs", {}) or {}),
                "side": dict(side.get("stage_outputs", {}) or {}),
                "substitution": dict(
                    substitution.get("stage_outputs", {}) or {}
                ),
            },
            "nmr": nmr_result,
            "best_candidate": substitution["best_candidate"],
            "beam_candidates": substitution["beam_candidates"],
            "molecule": final_state.to_dict(),
            "audit": audit,
            "outputs": outputs,
        }
        self._write_json(result, str(base / "final.json") if base else None)
        self._print_final_summary(result)
        return result

    def _finish_failed_stage(
        self,
        stage: str,
        payload: Dict[str, Any],
        stage_output_dir: Optional[str],
        base: Optional[Path],
    ) -> Dict[str, Any]:
        """Persist and report a failed/partial stage before returning.

        Search failures are expected outcomes of a bounded MCTS run.  They
        must therefore leave the same molecule JSON, PNG and summary as a
        successful stage, plus a top-level ``final.json`` describing why the
        pipeline stopped.
        """
        failed = self._failed_result(stage, payload)
        state = payload.get("state")
        stage_payload = dict(payload)
        stage_payload["result"] = dict(failed.get("result", {}))
        if isinstance(state, HexAssemblyState):
            stage_payload.setdefault("molecule", state.to_dict())
            if not stage_payload.get("best_candidate"):
                stage_payload["best_candidate"] = BeamCandidate(state, 0.0, [], {})
            if stage_output_dir and not stage_payload.get("stage_outputs"):
                self._write_stage_output(stage_payload, stage_output_dir, stage)
            failed["stage_outputs"] = dict(stage_payload.get("stage_outputs", {}))
            if not payload.get("stage_outputs"):
                self._print_stage_summary(stage, stage_payload)
        elif bool(getattr(self.config, "verbose", True)):
            print(
                f"[RL_MTCS][{stage}] status={failed.get('status', 'incomplete')} "
                f"done=False (no state available)",
                flush=True,
            )
        if base is not None:
            self._write_json(failed, str(base / "final.json"))
        return failed

    def _print_final_summary(self, result: Mapping[str, Any]) -> None:
        if not bool(getattr(self.config, "verbose", True)):
            return
        outputs = dict(result.get("outputs", {}) or {})
        audit = dict(result.get("audit", {}) or {})
        print(
            "[RL_MTCS][final] "
            f"status={result.get('status', 'unknown')} "
            f"ok={bool(result.get('ok', False))} "
            f"contracts={result.get('completed_contract_count', 0)} "
            f"components={audit.get('component_count', '?')}",
            flush=True,
        )
        if outputs.get("png"):
            print(f"[RL_MTCS][final] image={outputs['png']}", flush=True)

    @staticmethod
    def _failed_result(stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state")
        summary = dict(payload.get("result", {}) or {})
        status = str(summary.get("status", "incomplete"))
        summary.setdefault("partial", True)
        return {
            "ok": False,
            "status": status,
            "stage": str(stage),
            "failure_scope": summary.get("failure_scope", str(stage)),
            "best_candidate": payload.get("best_candidate"),
            "beam_candidates": payload.get("beam_candidates", []),
            "result": summary,
            "stage_result": summary,
            "candidate_space_complete": bool(
                summary.get("candidate_space_complete", False)
            ),
            "feasibility_audit": summary.get("feasibility_audit", {}),
            "best_partial_state": summary.get("best_partial_state", {}),
            "zero_domain_units": list(summary.get("zero_domain_units", [])),
            "completed_contract_count": int(
                summary.get("completed_contract_count", 0)
            ),
            "stage_outputs": dict(
                payload.get("stage_outputs", summary.get("stage_outputs", {}))
                or {}
            ),
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
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    def _print_stage_summary(self, stage: str, payload: Mapping[str, Any]) -> None:
        """Print a compact, stable report after every assembly stage."""
        if not bool(getattr(self.config, "verbose", True)):
            return
        summary = dict(payload.get("result", {}) or {})
        state = payload.get("state")
        audit = state.audit() if isinstance(state, HexAssemblyState) else {}
        done = summary.get("done", summary.get("complete", False))
        status = summary.get("status", "complete" if done else "incomplete")
        contracts_done = summary.get(
            "contracts_done",
            summary.get("completed_contract_count", len(
                state.satisfied_contracts if isinstance(state, HexAssemblyState) else ()
            )),
        )
        contracts_total = summary.get("contracts_total", "?")
        placed = summary.get("placed_units", "?")
        total = summary.get("phase_units_total", "?")
        components = summary.get("components", audit.get("component_count", "?"))
        outputs = dict(payload.get("stage_outputs", {}) or {})
        image = outputs.get("png", "")
        print(
            f"[RL_MTCS][{stage}] status={status} done={bool(done)} "
            f"units={placed}/{total} contracts={contracts_done}/{contracts_total} "
            f"components={components}",
            flush=True,
        )
        if summary.get("flex_chain_units_total") is not None:
            print(
                f"[RL_MTCS][{stage}] flex_chain="
                f"{summary.get('flex_chain_units_placed', 0)}/"
                f"{summary.get('flex_chain_units_total', 0)} "
                f"aliphatic_ring={summary.get('aliphatic_ring_units_placed', 0)}/"
                f"{summary.get('aliphatic_ring_units_total', 0)}",
                flush=True,
            )
        if summary.get("failure_scope") and summary.get("failure_scope") != "none":
            print(
                f"[RL_MTCS][{stage}] failure_scope="
                f"{summary.get('failure_scope')} "
                f"zero_domain_units={summary.get('zero_domain_units', [])}",
                flush=True,
            )
        if image:
            print(f"[RL_MTCS][{stage}] image={image}", flush=True)
        summary_path = outputs.get("summary_json", "")
        if summary_path:
            print(
                f"[RL_MTCS][{stage}] summary={summary_path}",
                flush=True,
            )

    def _write_stage_output(self, payload: Dict[str, Any], output_dir: Optional[str], stage: str) -> None:
        if not output_dir:
            return
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(payload["molecule"], str(directory / f"{stage}_molecule.json"))
        stage_outputs: Dict[str, str] = {
            "directory": str(directory),
            "molecule_json": str(directory / f"{stage}_molecule.json"),
            "summary_json": str(directory / f"{stage}_summary.json"),
        }
        if bool(getattr(self.config, "save_stage_images", True)):
            state = payload.get("state")
            if isinstance(state, HexAssemblyState) and state.node_axial:
                image_path = directory / f"{stage}_molecule.png"
                try:
                    audit = state.audit()
                    save_state_png(
                        state,
                        image_path,
                        title=(
                            f"RL_MTCS {stage} | nodes={len(state.node_axial)} "
                            f"edges={len(state.edges)} "
                            f"components={audit.get('component_count', '?')}"
                        ),
                    )
                    stage_outputs["png"] = str(image_path)
                except Exception as exc:
                    stage_outputs["visualization_error"] = str(exc)
                    self._write_json(
                        {"visualization_error": str(exc)},
                        str(directory / f"{stage}_visualization_error.json"),
                    )
        payload["stage_outputs"] = stage_outputs
        payload["result"]["stage_outputs"] = dict(stage_outputs)
        # Write the summary after visualization so the PNG path and any
        # visualization error are present in the persisted stage report.
        self._write_json(payload["result"], str(directory / f"{stage}_summary.json"))
        if 'search_diagnostics' in payload['result']:
            self._write_json({
                'actions': payload['best_candidate'].action_sequence,
                'plan': payload.get('flex_plan', {}),
                'diagnostics': payload['result']['search_diagnostics'],
                'rigid_search_attempts': payload['result'].get('rigid_search_attempts', []),
            }, str(directory / f'{stage}_search_trace.json'))


def assemble_from_csv(
    nodes_csv: Any,
    topology_specs_csv: Any,
    config: AssemblyConfig | None = None,
    output_dir: Optional[str] = None,
    search_guidance: SearchGuidance | None = None,
) -> Dict[str, Any]:
    """Assemble directly from the two final inverse CSV exports."""
    try:
        runner = MCTSRunner(
            nodes_csv,
            config=config,
            topology_specs_csv=topology_specs_csv,
            search_guidance=search_guidance,
        )
    except InverseInputError as exc:
        payload = {
            "ok": False,
            "status": "input_geometry_invalid",
            "failure_scope": "input",
            "candidate_space_complete": False,
            "feasibility_audit": {},
            "best_partial_state": {},
            "zero_domain_units": [],
            "completed_contract_count": 0,
            "errors": [str(exc)],
            "diagnostics": exc.diagnostics,
        }
        print(
            f"[RL_MTCS][input] status=input_geometry_invalid error={exc}",
            flush=True,
        )
        if output_dir:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            _write_json(target / "final.json", payload)
        return payload
    return runner.run_all_stages(output_dir)


def main() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fixed-topology hex-grid coarse-molecule assembly"
    )
    parser.add_argument("--nodes-csv", required=True, help="final_nodes_by_global_id.csv")
    parser.add_argument("--topology-specs-csv", help="final_topology_specs.csv (required with --nodes-csv)")
    parser.add_argument(
        "--stage",
        choices=("init", "rigid", "flex", "side", "all"),
        default="all",
    )
    parser.add_argument("--rigid-iterations", "--iterations", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--search-budget", type=int, default=2000)
    parser.add_argument("--conformer-limit", type=int, default=12)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--domain-check-limit", type=int, default=64)
    parser.add_argument("--domain-pose-limit", type=int, default=16)
    parser.add_argument("--max-domain-check-limit", type=int, default=512)
    parser.add_argument("--max-domain-pose-limit", type=int, default=64)
    parser.add_argument(
        "--strict-geometry", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--substitution-beam-width", type=int, default=4)
    parser.add_argument("--substitution-candidate-k", type=int, default=32)
    parser.add_argument("--substitution-nmr-weight", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.414)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if not args.topology_specs_csv:
        parser.error("--topology-specs-csv is required with --nodes-csv")
    config = AssemblyConfig(
        rigid_iterations=args.rigid_iterations,
        candidate_k=args.candidate_k,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        search_budget=args.search_budget,
        conformer_limit=args.conformer_limit,
        max_tree_nodes=args.max_tree_nodes,
        domain_check_limit=args.domain_check_limit,
        domain_pose_limit=args.domain_pose_limit,
        max_domain_check_limit=args.max_domain_check_limit,
        max_domain_pose_limit=args.max_domain_pose_limit,
        strict_geometry=args.strict_geometry,
        c_puct=args.c_puct,
        seed=args.seed,
        substitution_beam_width=args.substitution_beam_width,
        substitution_candidate_k=args.substitution_candidate_k,
        substitution_nmr_weight=args.substitution_nmr_weight,
    )
    if args.stage == "init":
        result = inspect_csv_initialization(
            args.nodes_csv, args.topology_specs_csv, args.output
        )
    else:
        runner = MCTSRunner(
            args.nodes_csv, config=config,
            topology_specs_csv=args.topology_specs_csv,
        )
        if args.stage == "all":
            result = runner.run_all_stages(args.output)
        elif args.stage == "rigid":
            result = runner.run_rigid_stage(args.output)
        elif args.stage == "flex":
            rigid = runner.run_rigid_stage()
            result = runner.run_flex_stage(rigid, args.output)
        else:
            rigid = runner.run_rigid_stage()
            flex = runner.run_flex_stage(rigid)
            result = runner.run_side_stage(flex, args.output)
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
