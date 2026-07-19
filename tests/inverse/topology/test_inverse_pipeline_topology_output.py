from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import model.inverse.inverse_pipeline as inverse_pipeline_module
import model.inverse.layer4.inverse_layer4 as inverse_layer4_module
from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
from model.inverse.topology.stage3_models import Stage3WorkflowCycleRecord, Stage3WorkflowResult
from model.inverse.topology.workflow import plan_stage3_resource_adjustments

from .csv_fixture_loader import load_nodes_from_final_csv
from .helpers import make_node


CSV_PATH = "/home/dofengqi/Graphnmr/test_results/YM-4 copy/final_outputs/final_nodes_by_global_id.csv"


def _build_pipeline_stub() -> InversePipelineV3:
    pipeline = InversePipelineV3.__new__(InversePipelineV3)
    pipeline.device = torch.device("cpu")
    pipeline.E_SU = inverse_pipeline_module.E_SU.detach().cpu()
    pipeline.layer4_adjuster = SimpleNamespace(fixed_partition_meta={})
    return pipeline


def test_inverse_pipeline_final_topology_rows_use_public_contract() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    pipeline = _build_pipeline_stub()

    rows, summary = pipeline._final_topology_rows(nodes)

    assert rows
    assert bool(summary.get("ok", True)) is False
    assert int(summary.get("closed_topology_count", 0) or 0) == len(rows)
    assert str(dict(summary.get("diagnostics", {}) or {}).get("reason", "")) == "external_ring_candidates_pending"
    assembly_diag = dict(dict(summary.get("diagnostics", {}) or {}).get("assembly", {}) or {})
    assert str(assembly_diag.get("stage3_phase", "")) in {
        "external_flex_closure",
        "external_fatty_ring_closure",
        "external_side_chain_closure",
    }
    flex_counts = dict(dict(summary.get("diagnostics", {}) or {}).get("flex_counts", {}) or {})
    assert int(flex_counts.get("explicit_total", 0) or 0) == 13
    assert int(flex_counts.get("final", 0) or 0) >= int(flex_counts.get("explicit_total", 0) or 0)

    pending = list(summary.get("unclosed_candidates", []) or [])
    assert pending
    assert set(pending[0].keys()) == {
        "contract_id", "real", "real_gid", "pending_kind", "phase", "subphase",
        "source_fragment_ids", "source_fragment_classes", "source_node_ids", "claimed_source_node_ids",
        "core_node_ids", "context_node_ids", "open_mainline_slots", "branch_slots", "required_counts",
        "available_counts", "infeasibility_proof", "reason", "attempt_reasons", "validation_errors",
    }
    assert str(pending[0].get("reason", ""))
    assert "source_ids" not in pending[0]
    assert not any(str(item.get("real", "")) == "-1" for item in pending)
    assert all(int(item.get("contract_id", -1)) >= 0 for item in pending)

    resource_ledger = dict(dict(summary.get("diagnostics", {}) or {}).get("resource_ledger", {}) or {})
    assert set(resource_ledger.keys()) == {
        "initial_counts", "consumed_counts", "topology_consumed_counts",
        "residual_consumed_counts", "remaining_counts", "shortage_counts", "snapshot_stage",
    }
    assert "classification_only" not in summary
    assert "assembly_implemented" not in summary
    assert "optionless_pending_group_count" not in summary
    assert "materialization_required" not in summary
    assert "unallocated_branch" not in summary
    assert "disabled" not in dict(summary.get("diagnostics", {}) or {})


def test_final_topology_export_forwards_stage3_targets(monkeypatch) -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    pipeline = _build_pipeline_stub()
    spectrum_marker = torch.ones(2, dtype=torch.float32)
    element_marker = torch.ones(6, dtype=torch.float32)
    captured = {}
    real_assemble = inverse_pipeline_module.assemble_topology_units

    def capture_targets(*args, **kwargs):
        captured["S_target"] = kwargs.get("S_target")
        captured["E_target"] = kwargs.get("E_target")
        kwargs["S_target"] = None
        kwargs["E_target"] = None
        return real_assemble(*args, **kwargs)

    monkeypatch.setattr(inverse_pipeline_module, "assemble_topology_units", capture_targets)
    pipeline._final_topology_rows(
        nodes,
        S_target=spectrum_marker,
        E_target=element_marker,
    )

    assert captured["S_target"] is spectrum_marker
    assert captured["E_target"] is element_marker


def test_final_topology_export_accepts_stage3_proven_residual_terminal(monkeypatch) -> None:
    pipeline = _build_pipeline_stub()
    pipeline._final_stage3_completion_diagnostics = {
        "residual_infeasibility_proof": {
            "reason": "no_executable_residual_template_fit",
            "remaining_counts": {"11": 4, "23": 1},
            "executable": False,
        }
    }

    class ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "classification_rejects": [],
                "unclassified_nodes": [],
                "resource_pool": {},
            }

    class AssemblyStub:
        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "residual_resource_self_closure_pending",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "resource_ledger": {"remaining_counts": {"11": 4, "23": 1}},
            }

        def to_dict(self):
            return {
                "diagnostics": {},
                "flex_bounds": {},
                "flex_counts": {},
                "flex_target_plan": {},
                "resource_demand": {},
                "protected_resource_counts": {},
                "referenced_resource_counts": {},
            }

    monkeypatch.setattr(inverse_pipeline_module, "classify_topology_units", lambda *args, **kwargs: ClassificationStub())
    monkeypatch.setattr(inverse_pipeline_module, "assemble_topology_units", lambda *args, **kwargs: AssemblyStub())

    rows, summary = pipeline._final_topology_rows([])

    assert rows == []
    assert summary["ok"] is True
    assert summary["pending_closure_total_count"] == 0
    assert summary["diagnostics"]["solver_reason"] == "stage3_complete"
    assert summary["diagnostics"]["residual_infeasibility_proof"]["executable"] is False


def test_final_topology_export_uses_matching_complete_generation_products(monkeypatch) -> None:
    pipeline = _build_pipeline_stub()
    nodes = [
        SimpleNamespace(
            global_id=node_id,
            su_type=su_type,
            hop1_ids=[],
            fixed_hop1_ids=set(),
            target_hop1_degree=None,
            target_fixed_anchor_count=None,
        )
        for node_id, su_type in ((1, 11), (2, 22), (3, 23), (4, 23))
    ]
    product = {
        "product_id": "11-23-23-22",
        "generation_id": 9,
        "product_type": "residual_resource_conversion",
        "source_fragment_ids": [],
        "fixed_source_node_ids": [],
        "claimed_resource_ids_by_su": {"11": [1], "22": [2], "23": [3, 4]},
        "mainline_slots": [{
            "slot_id": "residual:1:mainline",
            "assigned_resource_ids": [1, 3, 4, 2],
            "state": "FILLED",
        }],
        "branch_slots": [],
        "topology_notation": "11-23-23-22",
        "node_id_notation": "1-3-4-2",
        "closed": True,
    }
    pipeline._final_stage3_completion_diagnostics = {
        "stage3_complete": True,
        "generation_id": 9,
        "products": [product],
        "node_state_signature": pipeline._stage3_node_state_signature(nodes),
        "ownership_ledger": {
            "states": {
                "claimed_closure_resource": [1, 2, 3, 4],
                "remaining_closure_resource": [],
                "remaining_aromatic_donor": [],
            },
            "closure_pool_ids_by_su": {"11": [], "22": [], "23": [], "24": []},
            "aromatic_donor_ids_by_su": {"10": [], "12": [], "13": []},
            "invariant_errors": [],
        },
    }

    class ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "classification_rejects": [],
                "unclassified_nodes": [],
                "atomic_groups": [],
                "resource_pool": {
                    "initial_ids_by_su": {"11": [1], "22": [2], "23": [3, 4]},
                    "su11_donor_ids_by_su": {},
                },
            }

    class AssemblyStub:
        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "remaining_pool_spectrum_rebalance_ready",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "resource_ledger": {},
            }

        def to_dict(self):
            return {"diagnostics": {}}

    monkeypatch.setattr(inverse_pipeline_module, "classify_topology_units", lambda *args, **kwargs: ClassificationStub())
    monkeypatch.setattr(inverse_pipeline_module, "assemble_topology_units", lambda *args, **kwargs: AssemblyStub())

    rows, summary = pipeline._final_topology_rows(nodes)

    assert rows == [product]
    assert summary["ok"] is True
    assert summary["diagnostics"]["solver_reason"] == "stage3_complete"
    assert summary["diagnostics"]["stage3_generation_id"] == 9
    ledger = summary["diagnostics"]["resource_ledger"]
    assert ledger["remaining_counts"]["11"] == 0
    assert ledger["residual_consumed_counts"]["23"] == 2

    nodes[0].su_type = 12
    stale_rows, stale_summary = pipeline._final_topology_rows(nodes)
    assert stale_rows == []
    assert stale_summary["diagnostics"]["solver_reason"] != "stage3_complete"


def test_pipeline_captures_complete_generation_products_from_state_machine_ledger() -> None:
    nodes = [make_node(1, 11), make_node(2, 22), make_node(3, 23), make_node(4, 23)]
    histogram = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        histogram[int(node.su_type)] += 1
    plan = plan_stage3_resource_adjustments(
        histogram,
        nodes,
        enable_post_external_spectrum=False,
        generation_id=7,
    )
    assert plan.reason == "stage3_complete"
    workflow = Stage3WorkflowResult(
        nodes=nodes,
        H=histogram,
        diff={"r2": 0.5},
        ok=True,
        stop_reason="stage3_complete",
        cycles=[Stage3WorkflowCycleRecord(
            cycle=7,
            phase="complete",
            stop_reason="stage3_complete",
            plan=plan.to_dict(),
        )],
    )
    pipeline = _build_pipeline_stub()
    pipeline.layer4_adjuster.run_block_c_resource_adjustment_loop = lambda **_kwargs: workflow

    _, _, _, meta = pipeline._run_topology_stage3_resource_cycles(
        nodes=nodes,
        H=histogram,
        current_diff={"r2": 0.5},
        S_target=torch.zeros(2401),
        E_target=torch.zeros(6),
        eval_lib_path=None,
        layer1_runtime_cfg={},
        layer2_runtime_cfg={},
        layer2_eval_hwhm=1.0,
        outer_idx=0,
        enable_post_external_spectrum=False,
    )

    completion = meta["completion_diagnostics"]
    assert completion["stage3_complete"] is True
    assert completion["generation_id"] == 7
    assert completion["products"][0]["product_type"] == "residual_resource_conversion"
    assert completion["products"][0]["topology_notation"] == "11-23-23-22"
    assert completion["products"][0]["node_id_notation"] == "1-3-4-2"


def test_pipeline_captures_and_flattens_terminal_infeasibility_proof(tmp_path) -> None:
    nodes = [make_node(1, 24)]
    histogram = torch.zeros(40, dtype=torch.long)
    histogram[24] = 1
    proof = {
        "phase": "residual_resource_self_closure",
        "reason": "stage3_iteration_budget_exhausted",
        "required_counts": {"23": 1},
        "available_counts": {"24": 1},
        "missing_counts": {"23": 1},
    }
    workflow = Stage3WorkflowResult(
        nodes=nodes,
        H=histogram,
        diff={"r2": 0.5},
        ok=False,
        stop_reason="phase_resource_infeasible",
        cycles=[Stage3WorkflowCycleRecord(
            cycle=11,
            phase="residual_resource_self_closure",
            stop_reason="phase_resource_infeasible",
            plan={
                "phase": "residual_resource_self_closure",
                "phase_result": {"infeasibility_proof": proof},
            },
        )],
    )
    pipeline = _build_pipeline_stub()
    pipeline.layer4_adjuster.run_block_c_resource_adjustment_loop = lambda **_kwargs: workflow

    _, _, _, meta = pipeline._run_topology_stage3_resource_cycles(
        nodes=nodes,
        H=histogram,
        current_diff={"r2": 0.5},
        S_target=torch.zeros(2401),
        E_target=torch.zeros(6),
        eval_lib_path=None,
        layer1_runtime_cfg={},
        layer2_runtime_cfg={},
        layer2_eval_hwhm=1.0,
        outer_idx=0,
        enable_post_external_spectrum=False,
    )

    terminal = meta["terminal_diagnostics"]
    assert terminal["stop_reason"] == "phase_resource_infeasible"
    assert terminal["infeasibility_proof"] == proof
    rows = pipeline._flatten_stage3_resource_cycles(meta, outer_idx=0)
    assert 'stage3_iteration_budget_exhausted' in rows[-1]["infeasibility_proof"]
    pipeline._save_final_stage3_resource_summary(rows, str(tmp_path))
    summary = pd.read_json(tmp_path / "final_stage3_resource_summary.json", typ="series")
    assert summary["summary"]["infeasibility_proof"] == proof
    cycles = pd.read_csv(tmp_path / "final_stage3_resource_cycles.csv")
    assert "transactions" not in cycles.columns
    assert "infeasibility_reason" in cycles.columns
    assert (
        tmp_path / "debug" / "final_stage3_resource_cycle_details.jsonl"
    ).exists()
    assert (
        tmp_path / "debug" / "final_stage3_topology_transaction_details.jsonl"
    ).exists()


def test_pipeline_flattens_spectrum_candidate_scores() -> None:
    pipeline = _build_pipeline_stub()
    candidate = {
        "candidate_id": "spectrum:23:24:2",
        "primary_adjustment_count": 2,
    }
    scores = [{
        "candidate": candidate,
        "accepted": True,
        "r2_gain": 0.012,
        "window_loss_gain": 0.4,
    }]
    meta = {
        "accepted_cycles": 1,
        "cycles": [{
            "cycle": 3,
            "phase": "remaining_pool_spectrum_rebalance",
            "applied": True,
            "stop_reason": "accepted",
            "layer2_r2": 0.61,
            "diagnostics": {
                "selected_candidate": candidate,
                "r2_gain": 0.012,
                "window_loss_gain": 0.4,
                "evaluated_candidates": scores,
            },
            "plan": {"phase_result": {}},
        }],
    }

    row = pipeline._flatten_stage3_resource_cycles(meta, outer_idx=0)[0]

    assert row["spectrum_candidate_id"] == "spectrum:23:24:2"
    assert row["spectrum_r2_gain"] == 0.012
    assert row["spectrum_window_loss_gain"] == 0.4
    assert "spectrum:23:24:2" in row["spectrum_candidate_scores"]


def test_final_topology_export_uses_only_matching_terminal_infeasibility_proof() -> None:
    pipeline = _build_pipeline_stub()
    nodes = [make_node(1, 24)]
    proof = {
        "phase": "residual_resource_self_closure",
        "reason": "stage3_iteration_budget_exhausted",
    }
    pipeline._final_stage3_terminal_diagnostics = {
        "stop_reason": "phase_resource_infeasible",
        "generation_id": 11,
        "infeasibility_proof": proof,
        "node_state_signature": pipeline._stage3_node_state_signature(nodes),
    }

    _, summary = pipeline._final_topology_rows(nodes)

    assert summary["ok"] is False
    assert summary["diagnostics"]["solver_reason"] == "phase_resource_infeasible"
    assert summary["diagnostics"]["infeasibility_proof"] == proof
    assert summary["diagnostics"]["stage3_generation_id"] == 11
    assert summary["diagnostics"]["stage3_terminal_snapshot_match"] is True

    nodes[0].su_type = 23
    _, stale_summary = pipeline._final_topology_rows(nodes)
    assert stale_summary["diagnostics"]["solver_reason"] != "phase_resource_infeasible"
    assert stale_summary["diagnostics"]["infeasibility_proof"] == {}


def test_inverse_pipeline_final_topology_print_is_minimal(capsys) -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    pipeline = _build_pipeline_stub()

    pipeline._print_final_topology_specs(nodes, limit=200)
    output = capsys.readouterr().out

    assert "[Final Topology]" in output
    assert "[Resource Ledger]" in output
    assert "initial_total=" in output
    assert "consumed_total=" in output
    assert "remaining_total=" in output
    assert "shortage_total=" in output
    assert "initial_counts=" in output
    assert "consumed_counts=" in output
    assert "remaining_counts=" in output
    assert "shortage_counts=" in output
    assert "[Final Topology Pending]" in output
    assert "] real=-1 real_gid=" not in output
    assert "topology_consumed_counts=" in output
    assert "residual_consumed_counts=" in output
    assert "snapshot_stage=" in output
    assert "real_gid=" in output
    assert "topology=" in output
    assert "node_ids=" in output

    for token in [
        "type=",
        "origin=",
        "comp=",
        "src=",
        "candidate_group_id",
        "preferred_mode",
        "branch_ops",
        "main_ops",
        "reason=",
        "optionless=",
        "unallocated_branch=",
    ]:
        assert token not in output


def test_inverse_pipeline_final_topology_csv_exports_are_public(tmp_path: Path) -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    pipeline = _build_pipeline_stub()
    stale_duplicate = tmp_path / "final_topology_unclosed_candidates.csv"
    stale_duplicate.write_text("obsolete\n", encoding="utf-8")

    pipeline._save_final_result_summary(
        nodes,
        E_target=torch.zeros(6, dtype=torch.float32),
        S_target=None,
        output_dir=str(tmp_path),
    )

    specs = pd.read_csv(tmp_path / "final_topology_specs.csv")
    assert list(specs.columns) == [
        "actual_su",
        "gid",
        "product_type",
        "required_slots_satisfied",
        "source_fragment_ids",
        "fixed_source_node_ids",
        "claimed_resource_ids",
    ]
    assert not specs.empty
    assert "mainline_slots" not in specs.columns
    assert "branch_slots" not in specs.columns

    pending = pd.read_csv(tmp_path / "final_topology_pending.csv")
    assert list(pending.columns) == [
        "actual_su",
        "gid",
        "pending_kind",
        "phase",
        "contract_id",
        "source_fragment_ids",
        "fixed_source_node_ids",
        "required_counts",
        "available_counts",
        "reason",
    ]
    assert not any(str(value) == "-1" for value in list(pending["actual_su"]))
    assert not stale_duplicate.exists()

    resource_pool = pd.read_csv(tmp_path / "final_topology_resource_pool.csv")
    assert list(resource_pool.columns) == ["node_id", "su_type", "state"]
    summary = pd.read_csv(tmp_path / "final_topology_summary.csv").iloc[0]
    assert "initial_resource_pool_count" not in summary.index
    assert int(summary["initial_closure_resource_count"]) == (
        int(summary["topology_consumed_closure_resource_count"])
        + int(summary["residual_consumed_closure_resource_count"])
        + int(summary["remaining_closure_resource_count"])
    )
    assert int(summary["remaining_resource_pool_count"]) == (
        int(summary["remaining_closure_resource_count"])
        + int(summary["remaining_aromatic_donor_count"])
    )
    assert (tmp_path / "debug" / "final_topology_slot_details.jsonl").exists()
    assert (tmp_path / "debug" / "final_topology_pending_details.jsonl").exists()
    assert (tmp_path / "debug" / "final_topology_diagnostics.json").exists()


def test_inverse_pipeline_final_topology_summary_keeps_bad_topologies(monkeypatch) -> None:
    pipeline = _build_pipeline_stub()
    bad_topology = "11-single20d3-27-double20d3(-27)-14(-23-22)-15-single19d3(-28)-11"

    class _ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "classification_rejects": [],
                "unclassified_nodes": [],
                "resource_pool": {},
            }

    class _AssemblyStub:
        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "external_side_chain_closure_pending",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "resource_ledger": {
                    "consumed_counts": {},
                    "remaining_counts": {},
                    "shortage_counts": {},
                },
            }

        def to_dict(self):
            return {
                "ok": False,
                "reason": "external_side_chain_closure_pending",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "flex_bounds": {"min": 0, "max": 0, "target": 0, "enforced": False},
                "flex_counts": {"explicit_total": 0, "final": 0},
                "resource_ledger": {
                    "consumed_counts": {},
                    "remaining_counts": {},
                    "shortage_counts": {},
                },
                "diagnostics": {
                    "stage3_phase": "external_side_chain_closure",
                    "bad_topologies": [
                        {
                            "product_kind": "fatty_ring_side",
                            "actual_topology_su": bad_topology,
                            "actual_topology_gid": "1-2-3-4-5-6-7",
                            "reason": "side_ring_backbone_node_count:7",
                        }
                    ],
                },
                "planned_conversions": [],
            }

    monkeypatch.setattr(inverse_pipeline_module, "classify_topology_units", lambda *args, **kwargs: _ClassificationStub())
    monkeypatch.setattr(inverse_pipeline_module, "assemble_topology_units", lambda *args, **kwargs: _AssemblyStub())

    rows, summary = pipeline._final_topology_rows([])

    assert rows == []
    assembly_diag = dict(dict(summary.get("diagnostics", {}) or {}).get("assembly", {}) or {})
    assert list(assembly_diag.get("bad_topologies", []) or []) == [
        {
            "product_kind": "fatty_ring_side",
            "actual_topology_su": bad_topology,
            "actual_topology_gid": "1-2-3-4-5-6-7",
            "reason": "side_ring_backbone_node_count:7",
        }
    ]


def test_inverse_pipeline_pending_gid_invariant_allows_suppressed_support_nodes(monkeypatch) -> None:
    pipeline = _build_pipeline_stub()
    nodes = [SimpleNamespace(global_id=node_id, su_type=su_type) for node_id, su_type in [
        (182, 11), (310, 19), (405, 19), (505, 28), (569, 29),
    ]]
    pending = {
        "contract_id": 405,
        "real": "11-double19d3(-28)-",
        "real_gid": "182-405(-505)-",
        "pending_kind": "pending_ring_candidate",
        "phase": "external_fatty_ring_closure",
        "subphase": "real_like24_ring_assembly",
        "source_fragment_ids": [405],
        "source_fragment_classes": ["24_B"],
        "source_node_ids": [182, 310, 405, 505, 569],
        "claimed_source_node_ids": [182, 310, 405, 505, 569],
        "core_node_ids": [182, 310, 405, 505, 569],
        "context_node_ids": [],
        "open_mainline_slots": [],
        "branch_slots": [],
        "required_counts": {},
        "available_counts": {"11": 13, "23": 8, "24": 1},
        "infeasibility_proof": {
            "reason": "fused_bridgehead_missing",
            "suppressed_support_node_ids": [310, 569],
            "validation_errors": [],
        },
        "reason": "fused_bridgehead_missing",
        "attempt_reasons": ["fused_bridgehead_missing"],
        "validation_errors": [],
    }

    class _ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "classification_rejects": [],
                "unclassified_nodes": [],
                "resource_pool": {},
                "atomic_groups": [],
            }

    class _AssemblyStub:
        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "external_ring_candidates_pending",
                "feasible": True,
                "products": [],
                "pending_units": [pending],
                "resource_ledger": {
                    "initial_counts": {},
                    "consumed_counts": {},
                    "topology_consumed_counts": {},
                    "residual_consumed_counts": {},
                    "remaining_counts": {},
                    "shortage_counts": {},
                    "snapshot_stage": "final_ownership_reconciled",
                },
            }

        def to_dict(self):
            return {
                "ok": False,
                "reason": "external_ring_candidates_pending",
                "feasible": True,
                "products": [],
                "pending_units": [pending],
                "flex_bounds": {"min": 0, "max": 0, "target": 0, "enforced": False},
                "flex_counts": {"explicit_total": 0, "final": 0},
                "resource_ledger": self.to_public_dict()["resource_ledger"],
                "diagnostics": {
                    "stage3_phase": "external_fatty_ring_closure",
                    "node_coverage": {
                        "total_node_count": 5,
                        "completed_node_count": 0,
                        "pending_node_count": 5,
                        "remaining_pool_node_count": 0,
                        "uncovered_node_ids": [],
                    },
                },
                "planned_conversions": [],
            }

    monkeypatch.setattr(inverse_pipeline_module, "classify_topology_units", lambda *args, **kwargs: _ClassificationStub())
    monkeypatch.setattr(inverse_pipeline_module, "assemble_topology_units", lambda *args, **kwargs: _AssemblyStub())

    rows, summary = pipeline._final_topology_rows(nodes)

    assert rows == []
    assert summary["coverage_valid"] is True
    conflicts = list(dict(summary.get("diagnostics", {}) or {}).get("topology_invariant_conflicts", []) or [])
    assert conflicts == []
    assert str(dict(summary.get("diagnostics", {}) or {}).get("solver_reason", "")) != "topology_invariant_violation"


def test_final_topology_allows_adjacent_slots_to_share_a_path_node(monkeypatch) -> None:
    pipeline = _build_pipeline_stub()
    nodes = [
        SimpleNamespace(global_id=node_id, su_type=su_type)
        for node_id, su_type in ((1, 11), (2, 23), (3, 23), (4, 22))
    ]
    product = {
        "product_type": "residual_resource_conversion",
        "source_fragment_ids": [],
        "fixed_source_node_ids": [1],
        "claimed_resource_ids_by_su": {"23": [2, 3], "22": [4]},
        "mainline_slots": [
            {"slot_id": "path:0", "assigned_resource_ids": [2]},
            {"slot_id": "path:1", "assigned_resource_ids": [2, 3]},
            {"slot_id": "path:2", "assigned_resource_ids": [3, 4]},
        ],
        "branch_slots": [],
        "topology_notation": "11-23-23-22",
        "node_id_notation": "1-2-3-4",
        "closed": True,
    }

    class _ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "classification_rejects": [],
                "unclassified_nodes": [],
                "resource_pool": {},
                "atomic_groups": [],
            }

    class _AssemblyStub:
        def to_public_dict(self):
            return {
                "ok": True,
                "reason": "stage3_complete",
                "feasible": True,
                "products": [product],
                "pending_units": [],
                "resource_ledger": {},
            }

        def to_dict(self):
            return {
                **self.to_public_dict(),
                "diagnostics": {},
                "flex_bounds": {},
                "flex_counts": {},
                "flex_target_plan": {},
                "resource_demand": {},
                "protected_resource_counts": {},
                "referenced_resource_counts": {},
            }

    monkeypatch.setattr(
        inverse_pipeline_module,
        "classify_topology_units",
        lambda *args, **kwargs: _ClassificationStub(),
    )
    monkeypatch.setattr(
        inverse_pipeline_module,
        "assemble_topology_units",
        lambda *args, **kwargs: _AssemblyStub(),
    )

    rows, summary = pipeline._final_topology_rows(nodes)

    assert rows == [product]
    assert summary["coverage_valid"] is True
    assert list(
        dict(summary.get("diagnostics", {}) or {}).get("topology_invariant_conflicts", []) or []
    ) == []


def test_layer4_allocation_balance_default_payload_is_slim() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    adjuster = Layer4Adjuster(device=torch.device("cpu"), layer0_estimator=None)
    adjuster.fixed_partition_meta = {}

    diag = adjuster._evaluate_full_allocation_balance(nodes)

    required_keys = {
        "ok",
        "reason",
        "topology_assembly_ok",
        "topology_plan_feasible",
        "pending_closure_total_count",
        "closed_topology_count",
        "remaining",
        "resource_ledger",
    }
    assert required_keys.issubset(set(diag.keys()))

    removed_keys = {
        "disabled",
        "allocation_details",
        "allocation_result",
        "topology_classification",
        "topology_assembly",
        "topology_materialized_ok",
        "trace_exact_ok",
        "pending_closure_count",
        "source_su_ledger",
        "native_remaining",
        "proxy_remaining",
        "recommended_adjustments",
        "planned_conversions",
        "allocation_usage_ledgers",
        "pending_closure_blocking_count",
        "pending_closure_deferred_count",
        "unallocated_bridge",
        "unallocated_branch",
        "materialization_required",
    }
    assert not (removed_keys & set(diag.keys()))

    resource_ledger = dict(diag.get("resource_ledger", {}) or {})
    assert set(resource_ledger.keys()) == {
        "initial_counts", "consumed_counts", "topology_consumed_counts",
        "residual_consumed_counts", "remaining_counts", "shortage_counts", "snapshot_stage",
    }
    topology_diag = dict(diag.get("topology_assembly_diagnostics", {}) or {})
    assert "classification_only" not in topology_diag
    assert "assembly_implemented" not in topology_diag


def test_layer4_allocation_balance_keeps_bad_topologies_without_legacy_flags(monkeypatch) -> None:
    bad_topology = "11-single20d3-27-double20d3(-27)-14(-23-22)-15-single19d3(-28)-11"

    class _ClassificationStub:
        def to_dict(self):
            return {
                "summary": {},
                "unclassified_nodes": [],
                "resource_pool": {},
            }

    class _AssemblyStub:
        ok = False
        reason = "external_side_chain_closure_pending"
        feasible = True

        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "external_side_chain_closure_pending",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "resource_ledger": {
                    "consumed_counts": {},
                    "remaining_counts": {},
                    "shortage_counts": {},
                },
            }

        def to_dict(self):
            return {
                "ok": False,
                "reason": "external_side_chain_closure_pending",
                "feasible": True,
                "products": [],
                "pending_units": [],
                "flex_bounds": {"min": 0, "max": 0, "target": 0, "enforced": False},
                "flex_counts": {"explicit_total": 0, "final": 0},
                "resource_ledger": {
                    "consumed_counts": {},
                    "remaining_counts": {},
                    "shortage_counts": {},
                },
                "diagnostics": {
                    "stage3_phase": "external_side_chain_closure",
                    "bad_topologies": [
                        {
                            "product_kind": "fatty_ring_side",
                            "actual_topology_su": bad_topology,
                            "actual_topology_gid": "1-2-3-4-5-6-7",
                            "reason": "side_ring_backbone_node_count:7",
                        }
                    ],
                },
                "planned_conversions": [],
            }

    monkeypatch.setattr(inverse_layer4_module, "classify_topology_units", lambda *args, **kwargs: _ClassificationStub())
    monkeypatch.setattr(inverse_layer4_module, "assemble_topology_units", lambda *args, **kwargs: _AssemblyStub())

    adjuster = Layer4Adjuster(device=torch.device("cpu"), layer0_estimator=None)
    adjuster.fixed_partition_meta = {}
    diag = adjuster._evaluate_full_allocation_balance([])

    topology_diag = dict(diag.get("topology_assembly_diagnostics", {}) or {})
    assert "classification_only" not in topology_diag
    assert "assembly_implemented" not in topology_diag
    assert list(dict(topology_diag.get("assembly", {}) or {}).get("bad_topologies", []) or []) == [
        {
            "product_kind": "fatty_ring_side",
            "actual_topology_su": bad_topology,
            "actual_topology_gid": "1-2-3-4-5-6-7",
            "reason": "side_ring_backbone_node_count:7",
        }
    ]
