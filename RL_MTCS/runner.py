"""Inverse-driven Monte-Carlo topology assembly entry point.

Placement uses immutable aromatic clusters and topology products supplied by
:func:`RL_MTCS.io.input.initialize`; this module does not allocate or synthesize
topology units.  Rigid SU10 partners are chosen dynamically. Final
heteroaromatic SU26/SU30 substitution is applied only
after rigid, flex, ring, and side placement have completed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from RL_MTCS.io.input import initialize
from RL_MTCS.reporting.initialization import inspect_csv_initialization
from RL_MTCS.mdp.state import HexAssemblyState
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.mdp.types import InverseInputError
from RL_MTCS.policy.mcts import SearchGuidance
from RL_MTCS.io.output import export_final_state, composition_report
from RL_MTCS.io.output import save_state_png


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


class MCTSRunner:
    """Run rigid -> flex -> ring -> side -> substitution."""

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
        if not bool(self.config.verbose):
            return
        units = list(self.init_data.get("units", ()))
        contracts = list(self.init_data.get("contracts", ()))
        from RL_MTCS.mdp.components import phase_of
        unit_counts = {phase:sum(phase_of(u)==phase for u in units)
                       for phase in ('rigid','flex','ring','side')}
        owners = {u.unit_id:u for u in units}
        contract_counts = {}
        for contract in contracts:
            phase = 'rigid' if contract.kind=='rigid' else next(
                phase_of(owners[uid]) for uid in (contract.source_unit_id,contract.target_unit_id)
                if owners[uid].phase!='aromatic')
            contract_counts[phase]=contract_counts.get(phase,0)+1
        print(
            "[RL_MTCS][input] "
            f"units={len(units)} "
            f"aromatic={sum(getattr(unit, 'phase', '') == 'aromatic' for unit in units)} "
            f"flex_chain={unit_counts.get('flex', 0)} "
            f"aliphatic_ring={unit_counts.get('ring', 0)} "
            f"side_chain={unit_counts.get('side', 0)}",
            flush=True,
        )
        for issue in self.initial_state.problem.input_issues:
            row = json.loads(issue)
            print(f"[RL_MTCS][input][issue] {row}", flush=True)
        print(
            "[RL_MTCS][input] "
            f"rigid_mode=dynamic_su10_sites "
            f"rigid_ports={sum(su == 10 for _node_id, su in self.initial_state.problem.node_su_types)} "
            f"source_pairs={len(self.initial_state.problem.source_rigid_contracts)} "
            f"flex_contracts={contract_counts.get('flex', 0)} "
            f"ring_contracts={contract_counts.get('ring', 0)} "
            f"side_contracts={contract_counts.get('side', 0)}",
            flush=True,
        )

    def _initialize_nmr_scorer(self):
        spectrum_csv = self.config.spectrum_csv
        if not spectrum_csv:
            return None
        elements = self.config.elements
        composition_csv = self.config.element_composition_csv
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
        checkpoint = self.config.g2s_checkpoint
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
            scorer = NMRGuidanceScorer(
                spectrum_csv=spectrum_csv,
                elements=elements,
                model_checkpoint=checkpoint,
                device=str(self.config.nmr_device),
                batch_size=self.config.nmr_batch_size,
                hwhm=float(self.config.nmr_hwhm),
                graph_cache_limit=int(self.config.nmr_graph_cache_limit),
                node_cache_limit=int(self.config.nmr_node_cache_limit),
                record_limit=int(self.config.nmr_record_limit),
                center_limit=self.config.nmr_center_limit,
                aromatic_response_scale=self.config.nmr_aromatic_response_scale,
            )
            return scorer
        except ImportError as exc:
            raise InverseInputError(
                f"requested NMR guidance could not load its dependencies: {exc}"
            ) from exc

    def _controller(self):
        from RL_MTCS.policy.checkpoints import StageController
        if not hasattr(self, '_stage_controller'):
            self._stage_controller = StageController(self.config, self.nmr_scorer,
                                                     getattr(self, 'search_guidance', None))
        return self._stage_controller


    @staticmethod
    def _write_json(value, path):
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_json(target, value)

    def _run_stage(self, state, phase, output_dir=None):
        current, summary = self._controller().run(state, phase)
        payload = dict(state=current, result=summary, molecule=current.to_dict(),
                       ok=summary['legal'], status=summary['status'])
        self._write_stage_output(payload, output_dir, phase)
        if self.config.verbose:
            print(f"[RL_MTCS][{phase}] status={summary['status']} "
                  f"units={summary['units_done']}/{summary['units_total']} "
                  f"legal={summary['legal']}", flush=True)
        return payload

    def run_rigid_stage(self, output_dir=None):
        return self._run_stage(self.initial_state, 'rigid', output_dir)

    def run_flex_stage(self, rigid_result, output_dir=None):
        return self._run_stage(rigid_result['state'], 'flex', output_dir)

    def run_ring_stage(self, flex_result, output_dir=None):
        return self._run_stage(flex_result['state'], 'ring', output_dir)

    def run_side_stage(self, ring_result, output_dir=None):
        return self._run_stage(ring_result['state'], 'side', output_dir)

    def run_substitution_stage(self, side_result, output_dir=None):
        return self._run_stage(side_result['state'], 'substitution', output_dir)

    def run_all_stages(self, output_dir=None):
        from RL_MTCS.reporting.completion import completion_report
        stages = {}
        current = self.initial_state
        pre_substitution = None
        for phase in ('rigid', 'flex', 'ring', 'side', 'substitution'):
            if phase == 'substitution':
                pre_substitution = current.clone()
            directory = str(Path(output_dir) / phase) if output_dir else None
            payload = self._run_stage(current, phase, directory)
            stages[phase] = payload
            current = payload['state']
        audit = current.audit()
        report = completion_report(current)
        result = dict(ok=audit['legal'],
                      status='complete' if report['all_complete'] else 'partial',
                      all_complete=report['all_complete'], legal=audit['legal'],
                      input_valid=not current.problem.input_issues,
                      input_errors=[json.loads(issue) for issue in current.problem.input_issues],
                      state=current, molecule=current.to_dict(), audit=audit,
                      completion_report=report, stages=stages,
                      checkpoints=self._controller().checkpoints,
                      composition=composition_report(current,self.initial_state.problem,
                                                     pre_substitution,output_dir))
        if output_dir:
            result['outputs'] = export_final_state(current,output_dir)
            if self.nmr_scorer is not None:
                result['outputs'].update(self.nmr_scorer.export_outputs(output_dir,current))
                result['nmr_guidance'] = self.nmr_scorer.statistics()
            serial = {k:v for k,v in result.items() if k not in ('state','stages')}
            serial['stages'] = {p:v['result'] for p,v in stages.items()}
            self._write_json(serial,str(Path(output_dir)/'final.json'))
        return result

    def _write_stage_output(self, payload: Dict[str, Any], output_dir: Optional[str], stage: str) -> None:
        if not output_dir:
            return
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(payload["molecule"], str(directory / f"{stage}_molecule.json"))
        stage_outputs: Dict[str, str] = dict(payload.get("stage_outputs", {}) or {})
        stage_outputs.update({
            "directory": str(directory),
            "molecule_json": str(directory / f"{stage}_molecule.json"),
            "summary_json": str(directory / f"{stage}_summary.json"),
        })
        if bool(self.config.save_stage_images):
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
        summary = payload['result']
        trace = {'actions': payload['state'].action_history}
        if 'search_diagnostics' in summary:
            trace['diagnostics'] = summary['search_diagnostics']
        if 'checkpoints' in summary:
            trace['checkpoints'] = summary['checkpoints']
        elif 'retry_selection' in summary:
            trace['checkpoints'] = [summary['retry_selection']]
        self._write_json(trace, str(directory / f'{stage}_search_trace.json'))


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
            "legal": False,
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
        choices=("init", "rigid", "flex", "ring", "side", "all"),
        default="all",
    )
    parser.add_argument('--connection-candidate-limit', type=int, default=10)
    parser.add_argument('--forward-checkpoint-interval', type=int, default=4)
    parser.add_argument('--forward-min-completion', type=float, default=.80)
    parser.add_argument('--forward-max-rollbacks', type=int, default=4)
    parser.add_argument('--clearance-radius', type=int, default=3)
    parser.add_argument("--c-puct", type=float, default=1.414)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if not args.topology_specs_csv:
        parser.error("--topology-specs-csv is required with --nodes-csv")
    config = AssemblyConfig(
        connection_candidate_limit=args.connection_candidate_limit,
        forward_checkpoint_interval=args.forward_checkpoint_interval,
        forward_min_completion=args.forward_min_completion,
        forward_max_rollbacks=args.forward_max_rollbacks,
        clearance_radius=args.clearance_radius, c_puct=args.c_puct, seed=args.seed,
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
            ring = runner.run_ring_stage(flex, args.output if args.stage == "ring" else None)
            result = ring if args.stage == "ring" else runner.run_side_stage(ring, args.output)
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
