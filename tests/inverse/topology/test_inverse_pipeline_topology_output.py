from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import model.inverse.inverse_pipeline as inverse_pipeline_module
import model.inverse.layer4.inverse_layer4 as inverse_layer4_module
from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse.layer4.inverse_layer4 import Layer4Adjuster

from .csv_fixture_loader import load_nodes_from_final_csv


CSV_PATH = "/home/dofengqi/Graphnmr/test_results/YM-4 copy/final_outputs/final_nodes_by_global_id.csv"


def _build_pipeline_stub() -> InversePipelineV3:
    pipeline = InversePipelineV3.__new__(InversePipelineV3)
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
    assert str(assembly_diag.get("stage3_phase", "")) == "external_structural_closure"
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
    assert "actual_su=" in output

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

    topology_rows, topology_summary = pipeline._final_topology_rows(nodes)

    pd.DataFrame(
        topology_rows,
        columns=[
            "actual_su",
            "gid",
            "source_fragment_ids",
            "source_fragment_classes",
            "source_node_ids",
            "mainline_node_ids",
            "branch_node_ids",
            "claimed_resource_ids",
            "slot_assignments",
            "edge_plan",
        ],
    ).to_csv(tmp_path / "final_topology_specs.csv", index=False)

    unclosed_rows = list(topology_summary.get("unclosed_candidates", []) or [])
    pd.DataFrame(unclosed_rows, columns=["real", "real_gid"]).to_csv(
        tmp_path / "final_topology_unclosed_candidates.csv",
        index=False,
    )

    specs = pd.read_csv(tmp_path / "final_topology_specs.csv")
    assert list(specs.columns) == [
        "actual_su",
        "gid",
        "source_fragment_ids",
        "source_fragment_classes",
        "source_node_ids",
        "mainline_node_ids",
        "branch_node_ids",
        "claimed_resource_ids",
        "slot_assignments",
        "edge_plan",
    ]
    assert not specs.empty

    pending = pd.read_csv(tmp_path / "final_topology_unclosed_candidates.csv")
    assert list(pending.columns) == ["real", "real_gid"]
    assert not any(str(value) == "-1" for value in list(pending["real"]))


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
                "reason": "external_structural_closure_pending",
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
                "reason": "external_structural_closure_pending",
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
                    "stage3_phase": "external_structural_closure",
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
        "phase": "external_structural_closure",
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
                    "stage3_phase": "external_structural_closure",
                    "node_coverage": {
                        "total_node_count": 5,
                        "completed_node_count": 0,
                        "pending_node_count": 5,
                        "remaining_resource_node_count": 0,
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
        reason = "external_structural_closure_pending"
        feasible = True

        def to_public_dict(self):
            return {
                "ok": False,
                "reason": "external_structural_closure_pending",
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
                "reason": "external_structural_closure_pending",
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
                    "stage3_phase": "external_structural_closure",
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
