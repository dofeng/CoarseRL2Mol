from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.shared.coarse_graph import get_su_ppm_range
from model.inverse.inverse_pipeline import InversePipelineV3
from model.inverse import inverse_pipeline as inverse_pipeline_module
from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
from model.inverse.layer4 import inverse_layer4_block_c as block_c_module
from model.inverse.layer4.inverse_layer4_block_c import (
    _append_stage3_parity_balance,
    _append_stage3_cycle,
    _clear_stage3_node_runtime_state,
    _compact_stage3_plan_dict,
    _print_stage3_histogram_adjustments,
    _print_stage3_plan_summary,
    _prepare_spectrum_candidate_for_rebuild,
    _register_rejected_resource_candidate,
    _stage3_chemical_state_signature,
    _stage3_physical_state_signature,
    run_block_c_resource_adjustment_loop_impl,
)
from model.inverse.topology import (
    audit_stage3_topology,
    apply_stage3_node_adjustments,
    classify_topology_units,
    plan_stage3_resource_adjustments,
)
from model.inverse.topology.assembly.external import (
    ExternalTopologyAssembler,
)
from model.inverse.topology.assembly.projection import (
    project_fragment_views,
    project_like24_fragment,
)
from model.inverse.topology.assembly.closure.chain import (
    is_chain_materializable_view,
    remove_materialized_bridge_from_payload,
)
from model.inverse.topology.assembly.closure import chain as chain_materializer_module
from model.inverse.topology.assembly.workflow.coordinator import run_external_assembly_phases
from model.inverse.topology.assembly.state import pending as pending_module
from model.inverse.topology.assembly.state import ownership as ownership_module
from model.inverse.topology.assembly.closure import branch as branch_closure_module
from model.inverse.topology.assembly.ring import materializer as ring_materializer_module
from model.inverse.topology.assembly.ring import policy as ring_policy_module
from model.inverse.topology.assembly.ring import planner as ring_planner_module
from model.inverse.topology.assembly.ring import segments as ring_segments_module
from model.inverse.topology.assembly.state.pending import record_pending
from model.inverse.topology.assembly.core.fragment import AssemblyFragmentView
from model.inverse.topology.assembly.core.models import BranchClosureResult
from model.inverse.topology.assembly.projection.validation import (
    validate_fragment_view_projection,
    view_render_route_node_ids,
)
from model.inverse.topology.assembly.ring.models import RingCandidatePlan
from model.inverse.topology.assembly.ring import topology as ring_topology_module
from model.inverse.topology.assembly.core.rules import TERMINAL_ENDPOINT_LABELS
from model.inverse.topology.assembly.core.validation import (
    nested_path_edges,
    validate_chain_product_shape,
)
from model.inverse.topology.assembly.core.pool import PoolAllocator
from model.inverse.topology.assembly.state.round_context import (
    build_stage3_round_context,
    normalize_stage3_product,
    ownership_public_view,
)
from model.inverse.topology.stage3 import phase_eval as phase_eval_module
from model.inverse.topology.stage3 import state_machine as state_machine_module
from model.inverse.topology.stage3.phase_eval import PHASE_SEQUENCE
from model.inverse.topology.classification.classification_models import (
    TopologyClassificationResult,
    TopologyFragment,
)
from model.inverse.topology.classification.classification_tokens import (
    build_node_tokens,
)
from model.inverse.topology.classification.engine import (
    _build_resource_pool,
    _finalize_fragment_edge_semantics,
    _normalize_fragment_contract,
)
from model.inverse.topology.stage3.state_machine import (
    _accounting_remaining_pool,
    _all_physical_su24_ids,
    _fixed_flex_capacity_candidate,
    _fixed_flex_capacity_expansion_plan,
    _finalize_phase_plan,
    _internal_conversion_is_allowed,
    _iteration_ledger_for_phase_eval,
    _phase_infeasibility_labels,
    _restore_uncommitted_ring_pool_nodes,
    _runtime_with_pool,
    _round_resource_ids,
    _spectrum_trial_node_ids,
    _uncommitted_physical_su24_ids,
)
from model.inverse.topology.assembly.state.iteration_ledger import (
    build_iteration_ledger,
    seed_ledger_topology_state,
)
from model.inverse.topology.fixed_routes import (
    fragment_structure_signature,
    stage3_fixed_route_preflight,
)


def test_final_topology_success_reports_terminal_snapshot_match() -> None:
    pipeline = object.__new__(InversePipelineV3)
    pipeline._final_stage3_snapshot_authoritative = True
    pipeline._final_stage3_snapshot_node_ids = [1]
    pipeline._final_stage3_completion_diagnostics = {
        "stage3_complete": True,
        "node_ids": [1],
        "products": [],
        "classification": {
            "summary": {},
            "classification_rejects": [],
            "unclassified_nodes": [],
            "atomic_groups": [],
            "resource_pool": {},
        },
        "external_assembly": {},
        "ownership_ledger": {
            "states": {"remaining_aromatic_donor": [1]},
        },
        "resource_ledger": {},
        "resource_accounting": {"ok": True, "errors": []},
    }

    _, summary = pipeline._final_topology_rows(
        [SimpleNamespace(global_id=1)],
    )

    assert summary["ok"] is True
    assert summary["diagnostics"]["solver_reason"] == "stage3_complete"
    assert summary["diagnostics"]["stage3_terminal_snapshot_match"] is True
from model.inverse.topology.resources.accounting import (
    build_resource_accounting_snapshot,
    preflight_histogram_accounting,
)
from model.inverse.topology.resources.conversion_scoring import (
    ConversionContext,
    score_conversion_bundle,
)
from model.inverse.topology.resources.histogram import (
    _h_total,
    _phase_inventory,
    _stateful_conversion_bundle_signature,
    apply_histogram_adjustment_plan,
    build_histogram_adjustment_plan,
)
from model.inverse.topology.resources.spectrum_rebalance import (
    DEFAULT_SPECTRUM_WINDOWS,
    SPECTRUM_RESOURCE_TYPES,
    _atomic_score_state,
    _lorentzian_peak,
    _ranked_atomics_for_direction,
    _spectrum_profile_weights,
    _validated_spectrum_profiles,
    apply_spectrum_candidate,
    generate_spectrum_candidates,
)
from model.inverse.topology.resources.profiles import classify_raw_su24_semantics
from model.inverse.topology.assembly.ring.topology import audit_ring_topology_contract
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    RingTopologyContract,
    Stage3AdjustmentOp,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    Stage3ProgressPhase,
    Stage3OwnershipState,
    Stage3SpectrumCandidate,
    Stage3WorkflowCycleRecord,
    Stage3WorkflowResult,
    HistogramConversion,
    TopologyAssemblyProduct,
    TopologyAssemblyResult,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyPendingUnit,
)
from tools.topology_audit.helpers import connect, histogram, make_node


def _spectrum_profile_fixture() -> dict[int, dict[str, float]]:
    windows = {
        11: (100.0, 102.0, 110.0, 118.0, 120.0),
        12: (125.0, 127.0, 135.0, 143.0, 145.0),
        13: (150.0, 152.0, 160.0, 168.0, 170.0),
        22: (5.0, 7.0, 10.0, 13.0, 15.0),
        23: (30.0, 32.0, 35.0, 38.0, 40.0),
        24: (50.0, 52.0, 55.0, 58.0, 60.0),
    }
    return {
        su_type: {
            "mu_common_min": lo,
            "mu_q05": q05,
            "mu_median": median,
            "mu_q95": q95,
            "mu_common_max": hi,
        }
        for su_type, (lo, q05, median, q95, hi) in windows.items()
    }


def _spectrum_hop1_fixture() -> dict[int, list[dict[str, object]]]:
    profiles = _spectrum_profile_fixture()
    return {
        su_type: [{
            "hop1_tuple": (su_type,),
            "mu_median": float(profile["mu_median"]),
            "sample_weight_sum": 10.0,
            "n_templates": 1,
        }]
        for su_type, profile in profiles.items()
    }


def _node_with_mu(node_id: int, su_type: int, mu: float):
    node = make_node(node_id, su_type)
    node.mu = float(mu)
    return node


def test_outer_selected_state_signature_detects_true_convergence() -> None:
    left = _node_with_mu(0, 22, 10.0)
    right = _node_with_mu(1, 23, 35.0)
    left.pi = right.pi = 1.0
    left.hop1_ids = [1]
    right.hop1_ids = [0]
    nodes = [left, right]
    H = histogram(nodes)

    baseline = InversePipelineV3._outer_selected_state_signature(nodes, H)
    same = InversePipelineV3._outer_selected_state_signature(
        [right, left],
        H.detach().clone(),
    )
    assert same == baseline

    right.mu = 35.1
    assert InversePipelineV3._outer_selected_state_signature(nodes, H) != baseline
    right.mu = 35.0
    right.hop1_ids = []
    assert InversePipelineV3._outer_selected_state_signature(nodes, H) != baseline


def test_stage3_console_summary_is_concise_by_default(capsys) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 4
    op = SimpleNamespace(
        from_su=12,
        to_su=13,
        via_su=None,
        count=2,
        node_ids=[101, 102],
        reason="internal_scoring_reason",
    )
    group = SimpleNamespace(primary_op=op, h_compensation_ops=[])
    histogram_plan = SimpleNamespace(
        groups=[group],
        H_after=H,
        parity_balance_op=None,
        port_role_balance_op=None,
    )
    plan = SimpleNamespace(
        histogram_adjustment_plan=histogram_plan,
        diagnostics={},
        phase="flex_range_reconciliation",
        plan_kind="fixed_flex_capacity_expansion",
        reason="stage3_rebuild_required",
    )

    _print_stage3_plan_summary(1, plan, H, None)
    _print_stage3_histogram_adjustments(1, histogram_plan)
    concise = capsys.readouterr().out
    assert "[Stage3 Plan]" in concise
    assert "SU12->SU13 x2" in concise
    assert "[Stage3 H]" not in concise
    assert "[Stage3 Histogram]" not in concise
    assert "nodes=" not in concise
    assert "internal_scoring_reason" not in concise

    _print_stage3_plan_summary(1, plan, H, None, verbose=True)
    _print_stage3_histogram_adjustments(1, histogram_plan, verbose=True)
    detailed = capsys.readouterr().out
    assert "[Stage3 H]" in detailed
    assert "[Stage3 Histogram]" in detailed
    assert "nodes=[101, 102]" in detailed
    assert "internal_scoring_reason" in detailed


def test_stage3_cycle_reporter_aggregates_attempts_and_concrete_recovery(capsys) -> None:
    reporter = block_c_module._Stage3CycleConsoleReporter(
        verbose=False,
        flex_limit=14,
        external_limit=12,
        internal_limit=30,
    )
    reporter.begin(
        cycle=19,
        attempt=1,
        phase="external_topology_assembly",
        external_used=0,
    )
    recovery_op = SimpleNamespace(
        from_su=24,
        to_su=23,
        via_su=None,
        count=2,
        node_ids=[101, 102],
    )
    reporter.record_histogram(
        SimpleNamespace(groups=[SimpleNamespace(
            primary_op=recovery_op,
            h_compensation_ops=[],
        )], parity_balance_op=None, port_role_balance_op=None),
        external_budget_after=2,
        recovery=True,
    )
    reporter.begin(
        cycle=19,
        attempt=2,
        phase="external_topology_assembly",
        external_used=2,
    )
    ordinary_op = SimpleNamespace(
        from_su=13,
        to_su=22,
        via_su=11,
        count=1,
        node_ids=[103],
    )
    reporter.record_histogram(
        SimpleNamespace(groups=[SimpleNamespace(
            primary_op=ordinary_op,
            h_compensation_ops=[SimpleNamespace()],
        )], parity_balance_op=None, port_role_balance_op=None),
        external_budget_after=3,
    )
    reporter.finish(boundary="external_generation_complete")

    output = capsys.readouterr().out
    assert output.count("[Stage3 Cycle 19]") == 1
    assert "attempts=2 primary=3/12 compensation=1" in output
    assert "SU24->SU23 x2" in output
    assert "SU13->SU11->SU22 x1" in output
    assert "recovery: SU24->SU23 x2" in output


def test_outer_cycle_snapshot_publishes_complete_and_terminal_state(
    tmp_path,
    monkeypatch,
) -> None:
    pipeline = object.__new__(InversePipelineV3)
    pipeline.device = torch.device("cpu")
    nodes = [make_node(1, 23), make_node(2, 24)]
    H = histogram(nodes)

    monkeypatch.setattr(
        pipeline,
        "_save_final_node_exports",
        lambda **_kwargs: None,
    )

    def _write_topology(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "final_topology_specs.csv").write_text(
            "actual_su,gid\n24,2\n", encoding="utf-8"
        )
        (output / "final_pending_topology_specs.csv").write_text(
            "actual_su,gid\n", encoding="utf-8"
        )

    monkeypatch.setattr(pipeline, "_save_final_result_summary", _write_topology)
    result = pipeline._save_outer_cycle_snapshot(
        outer_idx=0,
        nodes=nodes,
        H=H,
        E_target=torch.zeros(6),
        S_target=None,
        output_root=tmp_path,
        run_id="snapshot-test",
        stage3_complete=True,
        stage3_meta={
            "stop_reason": "stage3_complete",
            "completion_diagnostics": {
                "stage3_complete": True,
                "products": [{"product_id": "ring-1"}],
                "ownership_ledger": {"states": {"completed_product": [2]}},
            },
        },
        selected_diff={"r2": 0.7, "spectral_contract": "test"},
        pre_layer3_r2=0.6,
        post_layer3_r2=0.7,
        layer3_committed=True,
    )

    final_dir = tmp_path / "outer_cycle_1" / "final_outputs"
    assert result["published"] is True
    assert (final_dir / "final_su_distribution.csv").exists()
    assert (final_dir / "final_topology_specs.csv").exists()
    assert (final_dir / "final_pending_topology_specs.csv").exists()
    status = json.loads(
        (final_dir / "outer_cycle_status.json").read_text(encoding="utf-8")
    )
    assert status["complete"] is True
    assert status["snapshot_ok"] is True
    assert status["su_identity"]["node_su24_count"] == 1
    assert status["su_identity"]["stage_h_su24_count"] == 1
    assert status["stage3_snapshot"]["products"] == [{"product_id": "ring-1"}]


def test_outer_cycle_snapshot_keeps_su_csv_when_topology_export_fails(
    tmp_path,
    monkeypatch,
) -> None:
    pipeline = object.__new__(InversePipelineV3)
    pipeline.device = torch.device("cpu")
    nodes = [make_node(1, 23)]
    H = histogram(nodes)
    old_dir = tmp_path / "outer_cycle_1" / "final_outputs"
    old_dir.mkdir(parents=True)
    (old_dir / "stale.csv").write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "_save_final_node_exports",
        lambda **_kwargs: None,
    )

    def _fail_topology(**_kwargs):
        raise RuntimeError("simulated topology export failure")

    monkeypatch.setattr(pipeline, "_save_final_result_summary", _fail_topology)
    result = pipeline._save_outer_cycle_snapshot(
        outer_idx=0,
        nodes=nodes,
        H=H,
        E_target=torch.zeros(6),
        S_target=None,
        output_root=tmp_path,
        run_id="snapshot-failure-test",
        stage3_complete=False,
        stage3_meta={
            "stop_reason": "phase_resource_infeasible",
            "terminal_diagnostics": {
                "phase": "external_topology_assembly",
                "stop_reason": "phase_resource_infeasible",
                "infeasibility_proof": {"missing_counts": {"11": 1}},
                "products": [{"product_id": "partial-1"}],
                "pending_contracts": [{"contract_id": 406}],
                "ownership_ledger": {"states": {}},
            },
        },
        selected_diff={"r2": 0.5},
        pre_layer3_r2=0.5,
        post_layer3_r2=float("nan"),
        layer3_committed=False,
    )

    final_dir = tmp_path / "outer_cycle_1" / "final_outputs"
    assert result["published"] is True
    assert (final_dir / "final_su_distribution.csv").exists()
    assert not (final_dir / "stale.csv").exists()
    assert (final_dir / "final_pending_topology_specs.csv").exists()
    assert (tmp_path / "outer_cycle_1" / "snapshot_failure.json").exists()
    assert list((tmp_path / "outer_cycle_1" / "archive").glob("*/stale.csv"))
    status = json.loads(
        (final_dir / "outer_cycle_status.json").read_text(encoding="utf-8")
    )
    assert status["complete"] is False
    assert status["snapshot_ok"] is False
    assert status["terminal_proof"]["missing_counts"] == {"11": 1}
    assert status["stage3_snapshot"]["pending_contracts"] == [
        {"contract_id": 406}
    ]


def test_final_stage3_resource_summary_includes_final_generation_metrics(tmp_path) -> None:
    pipeline = object.__new__(InversePipelineV3)
    pipeline._save_final_stage3_resource_summary(
        rows=[{
            "outer_cycle": 1,
            "cycle": 4,
            "phase": "complete",
            "stage3_stop_reason": "stage3_complete",
        }],
        output_dir=str(tmp_path),
        completion={
            "generation_count": 4,
            "internal_physical_conversion_count": 7,
            "internal_ring_count": 3,
            "remaining_resource_counts": {11: 2, 23: 1},
            "generation_metrics": [
                {
                    "generation_id": 1000002,
                    "internal_physical_conversion_count": 30,
                    "internal_ring_count": 2,
                },
                {
                    "generation_id": 1000003,
                    "internal_physical_conversion_count": 7,
                    "internal_ring_count": 3,
                },
            ],
        },
    )

    payload = json.loads(
        (tmp_path / "final_stage3_resource_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = payload["summary"]
    assert summary["generation_count"] == 4
    assert summary["final_generation_internal_physical_conversion_count"] == 7
    assert summary["final_generation_internal_ring_count"] == 3
    assert summary["remaining_resource_counts"] == {"11": 2, "23": 1}
    assert summary["generation_metrics"] == [
        {
            "generation_id": 1000002,
            "internal_physical_conversion_count": 30,
            "internal_ring_count": 2,
        },
        {
            "generation_id": 1000003,
            "internal_physical_conversion_count": 7,
            "internal_ring_count": 3,
        },
    ]


def test_aliphatic_resource_spectrum_windows_use_updated_authoritative_ranges() -> None:
    expected = {
        22: (0.0, 25.0),
        23: (18.0, 45.0),
        24: (28.0, 60.0),
    }
    for su_type, ppm_range in expected.items():
        assert get_su_ppm_range(su_type) == ppm_range
        assert DEFAULT_SPECTRUM_WINDOWS[su_type] == (*ppm_range, *ppm_range)


def _raw_su24_pool_fixture(roles):
    nodes = []
    pool = {11: [], 22: [], 23: [], 24: []}
    next_id = 1

    def add(su_type, *, resource=False):
        nonlocal next_id
        node = make_node(next_id, su_type)
        next_id += 1
        nodes.append(node)
        if resource and int(su_type) in pool:
            pool[int(su_type)].append(int(node.global_id))
        return node

    for role in roles:
        center = add(24, resource=True)
        if role == "A":
            neighbors = [add(11, resource=True), add(23, resource=True), add(23, resource=True)]
        elif role == "B":
            neighbors = [add(11, resource=True), add(22, resource=True), add(23, resource=True)]
        elif role == "C":
            neighbors = [add(0), add(2), add(3)]
        elif role == "D":
            neighbors = [add(22, resource=True), add(0), add(2)]
        else:
            raise ValueError(role)
        for neighbor in neighbors:
            connect(center, neighbor)

    # Generic ring endpoints, middle positions and H branches all claim real
    # closure resources from this same pool.
    for _ in range(8):
        add(11, resource=True)
    for _ in range(16):
        add(22, resource=True)
    for _ in range(24):
        add(23, resource=True)
    return nodes, pool


def test_basic_b_and_d_views_expose_existing_paths_to_ring_templates() -> None:
    nodes = [
        make_node(1, 9),
        make_node(2, 3),
        make_node(3, 15),
        make_node(4, 15),
    ]
    classification = TopologyClassificationResult(fragments=[
        TopologyFragment(
            fragment_id=2,
            family="basic",
            class_type="B",
            topology_template="9-3-OPEN",
            source_node_ids=[1, 2],
            claimed_node_ids=[1, 2],
            open_slots=["OPEN"],
            evidence={
                "center_node_id": 2,
                "matched_neighbors": [
                    {"node_id": 1, "su_type": 9, "subtype": "9"},
                ],
            },
        ),
        TopologyFragment(
            fragment_id=3,
            family="basic",
            class_type="D",
            topology_template="OPEN-15-15-OPEN",
            source_node_ids=[3, 4],
            claimed_node_ids=[3, 4],
            open_slots=["OPEN", "OPEN"],
            evidence={
                "center_node_id": 3,
                "matched_neighbors": [
                    {"node_id": 4, "su_type": 15, "subtype": "15"},
                ],
            },
        ),
    ])

    views = {
        view.class_type: view
        for view in project_fragment_views(
            classification,
            {int(node.global_id): node for node in nodes},
        )
    }

    assert views["B"].can_ring is True
    assert views["B"].ring_backbone_node_ids["side"] == [1, 2]
    assert views["B"].evidence["ring_path_capabilities"]["side"] == [
        "terminal_pair"
    ]
    assert views["D"].can_ring is True
    assert views["D"].ring_backbone_node_ids["side"] == [3, 4]
    assert views["D"].evidence["ring_path_capabilities"] == {
        "side": ["inner_pair"],
        "upper_lower": ["path_pair"],
    }


def test_e_plus_is_an_existing_branch_donor_without_becoming_ring_source() -> None:
    donor = AssemblyFragmentView(
        unit_id=1,
        fragment_id=1,
        family="double_d2_combo",
        class_type="E+",
        topology_template="terminal-connector-center-OPEN",
        source_node_ids=[1],
        main_labels=["6", "19"],
        main_node_ids=[1, 2],
        canonical_labels=["6", "19"],
        open_left=False,
        open_right=True,
        can_side=True,
        can_ring=False,
    )
    closed_side = AssemblyFragmentView(
        unit_id=2,
        fragment_id=2,
        family="basic",
        class_type="C",
        topology_template="11-center-terminal",
        source_node_ids=[3],
        main_labels=["11", "3", "22"],
        main_node_ids=[3, 4, 5],
        canonical_labels=["11", "3", "22"],
        open_left=False,
        open_right=False,
        can_side=True,
        can_ring=False,
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [donor, closed_side]
    assembler.fragment_state_by_id = {1: "available", 2: "available"}

    assert [view.fragment_id for view in chain_materializer_module.available_branch_donors(assembler, )] == [1]


def test_branch_source_collection_preserves_transaction_local_view() -> None:
    stale = AssemblyFragmentView(
        unit_id=1,
        fragment_id=1,
        family="test",
        class_type="test",
        topology_template="11-23",
        source_node_ids=[1, 2],
        main_labels=["11", "23"],
        main_node_ids=[1, 2],
        canonical_labels=["11", "23"],
        open_left=False,
        open_right=False,
    )
    prepared = AssemblyFragmentView(
        unit_id=1,
        fragment_id=1,
        family="test",
        class_type="test",
        topology_template="11-23",
        source_node_ids=[1, 3],
        main_labels=["11", "23"],
        main_node_ids=[1, 3],
        canonical_labels=["11", "23"],
        open_left=False,
        open_right=False,
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.view_by_id = {1: stale}

    collected = chain_materializer_module.collect_source_views_from_branches(assembler,
        [prepared],
        [BranchClosureResult(source_fragment_ids=[1])],
    )

    assert collected == [prepared]
    assert collected[0] is prepared


def test_product_identity_is_derived_from_rendered_paths(monkeypatch) -> None:
    view = AssemblyFragmentView(
        unit_id=23,
        fragment_id=23,
        family="test",
        class_type="test",
        topology_template="11-23",
        source_node_ids=[23, 322, 323, 450],
        main_labels=["11", "23"],
        main_node_ids=[23, 450],
        canonical_labels=["11", "23"],
        open_left=False,
        open_right=False,
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes = []
    monkeypatch.setattr(
        chain_materializer_module,
        "physical_product_reason",
        lambda _assembler, **_kwargs: None,
    )

    product = chain_materializer_module.make_product(assembler,
        product_kind="flex_chain",
        reason="test",
        source_views=[view],
        labels=["11", "23"],
        node_ids=[23, 450],
        branch_map={},
        branch_id_map={},
        suppressed_support_node_ids=[322, 323],
    )

    assert product is not None
    assert product.fixed_source_node_ids == [23, 450]
    assert product.rendered_node_ids() == [23, 450]
    assert product.owned_node_ids() == [23, 450]
    assert "source_node_ids" not in product.to_dict()
    assert "symbolic_node_ids" not in product.to_dict()


def test_rendered_base_node_overrides_foreign_suppressed_context(monkeypatch) -> None:
    base = AssemblyFragmentView(
        unit_id=497,
        fragment_id=497,
        family="test",
        class_type="test",
        topology_template="24",
        source_node_ids=[497],
        main_labels=["24"],
        main_node_ids=[497],
        canonical_labels=["24"],
        open_left=False,
        open_right=False,
    )
    donor_context = AssemblyFragmentView(
        unit_id=430,
        fragment_id=430,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[430],
        main_labels=["double19d3"],
        main_node_ids=[430],
        canonical_labels=["double19d3"],
        open_left=True,
        open_right=True,
        suppressed_support_node_ids=[497],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes = []
    monkeypatch.setattr(
        chain_materializer_module,
        "physical_product_reason",
        lambda _assembler, **_kwargs: None,
    )

    product = chain_materializer_module.make_product(assembler,
        product_kind="flex_chain",
        reason="test_cross_fragment_suppression",
        source_views=[base, donor_context],
        labels=["24"],
        node_ids=[497],
        branch_map={},
        branch_id_map={},
        suppressed_support_node_ids=[497],
    )

    assert product is not None
    assert product.rendered_node_ids() == [497]
    assert product.fixed_source_node_ids == [497]


def test_product_identity_mismatch_rolls_back_allocator() -> None:
    nodes = [make_node(1, 0), make_node(10, 11)]
    classification = TopologyClassificationResult(
        resource_pool={"remaining_ids_by_su": {"11": [10]}}
    )
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    snapshot = assembler.allocator.snapshot()
    assert assembler.allocator.claim(11, 1) == [10]
    product = TopologyAssemblyProduct(
        product_kind="flex_chain",
        actual_topology_su="0",
        actual_topology_gid="1",
        fixed_source_node_ids=[1],
        mainline_node_ids=[1],
    )

    assert ownership_module.commit_product(
        assembler,
        product,
        allocator_snapshot=snapshot,
    ) is False
    assert assembler.allocator.remaining_ids_by_su["11"] == [10]
    assert assembler.products == []
    assert assembler.last_invalid_topology["reason"] == (
        "rendered_product_ownership_mismatch"
    )


def test_pending_contract_has_one_canonical_node_identity_field() -> None:
    pending = TopologyPendingUnit(
        unit_id=23,
        claimed_source_node_ids=[23, 450],
    )

    payload = pending.to_dict()
    public = TopologyAssemblyResult(pending_units=[payload]).to_public_dict()[
        "pending_units"
    ][0]

    assert payload["claimed_source_node_ids"] == [23, 450]
    assert public["claimed_source_node_ids"] == [23, 450]
    for old_field in ("source_node_ids", "core_node_ids"):
        assert old_field not in payload
        assert old_field not in public


def test_external_priority_ring_can_start_from_only_real_su24_pool() -> None:
    nodes, _ = _raw_su24_pool_fixture(["C", "C"])
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )

    result = assembler.assemble().to_dict()
    rings = [
        product
        for product in result["products"]
        if str(product["product_kind"]).startswith("fatty_ring_")
    ]

    assert len(rings) == 1
    assert len(rings[0]["ring_audit"]["cycles"][0]) == 6
    assert result["resource_ledger"]["remaining_counts"]["24"] == 0


def test_external_side_ring_combines_basic_b_endpoint_with_real_su24() -> None:
    nodes = []
    next_id = 1

    def add(su_type: int):
        nonlocal next_id
        node = make_node(next_id, su_type)
        next_id += 1
        nodes.append(node)
        return node

    aromatic = add(9)
    center = add(3)
    open_path = add(23)
    connect(center, aromatic)
    connect(center, open_path)
    for _ in range(2):
        raw_su24 = add(24)
        for su_type in (0, 2, 3):
            connect(raw_su24, add(su_type))
    for su_type, count in ((11, 8), (22, 16), (23, 24)):
        for _ in range(count):
            add(su_type)

    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={
            "enforced": True,
            "scenario": "in_range",
            "flex_lower": 0,
            "flex_upper": 0,
            "target_flex_count": 0,
        },
    )
    result = assembler.assemble().to_dict()
    ring = next(
        product
        for product in result["products"]
        if product["product_kind"] == "fatty_ring_side"
    )

    assert ring["source_fragment_ids"] == [int(center.global_id)]
    assert len(ring["ring_audit"]["cycles"][0]) == 6
    assert int(center.global_id) not in set(ring["ring_audit"]["like24_node_ids"])
    assert result["diagnostics"]["invalid_topology_count"] == 0


def test_external_histogram_authorization_rejects_non_adjustment_types() -> None:
    cases = [
        ({10: 1}, {13: 1}, 10, 13),
        ({24: 1}, {11: 1}, 24, 11),
        ({23: 1}, {24: 1}, 23, 24),
    ]
    for inventory, demand, source, target in cases:
        H = torch.zeros(33, dtype=torch.long)
        H[int(source)] = 1
        plan = Stage3AdjustmentPlan(
            phase="external_fatty_ring_closure",
            hard_required_counts={str(key): value for key, value in demand.items()},
            runtime={
                "assembly_before_adjustment": {
                    "resource_ledger": {
                        "convertible_counts": {
                            str(key): value for key, value in inventory.items()
                        },
                    },
                },
            },
            diagnostics={"forbidden_conversion_su_types": [24]},
        )
        histogram_plan = build_histogram_adjustment_plan(
            plan=plan,
            H=H,
            max_primary_adjustments=2,
            max_h_compensation_ops=2,
        )

        assert histogram_plan.groups == []


def test_spectrum_su24_view_includes_every_physical_node() -> None:
    nodes = [make_node(1, 24), make_node(2, 24), make_node(3, 23)]
    canonical_pool = {11: [], 22: [], 23: [3], 24: [2]}

    spectrum_pool = _spectrum_trial_node_ids(nodes, canonical_pool)

    assert _all_physical_su24_ids(nodes) == [1, 2]
    assert spectrum_pool[24] == [1, 2]
    assert canonical_pool[24] == [2]


def test_internal_stage_converts_su24_without_stale_committed_product_cache() -> None:
    nodes = [make_node(1, 24), make_node(2, 24)]
    H = histogram(nodes)
    committed = {
        "product_id": "external:ring:1",
        "product_kind": "fatty_ring_side",
        "phase": "external_fatty_ring_closure",
        "fixed_source_node_ids": [],
        "claimed_resource_ids_by_su": {"24": [1]},
        "closed": True,
    }

    plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
        progress_phase=Stage3ProgressPhase.INTERNAL_TOPOLOGY,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.plan_kind == "sequential_phase_transactions"
    assert plan.histogram_adjustment_plan is not None
    assert sorted(
        int(op.node_id) for op in plan.planned_conversions
    ) == [1, 2]
    assert all(int(op.from_su) == 24 for op in plan.planned_conversions)
    assert all(int(op.to_su) in {12, 13} for op in plan.planned_conversions)
    assert plan.diagnostics["su24_remaining_count"] == 0
    assert plan.diagnostics["internal_physical_conversion_count"] == 2
    assert plan.diagnostics["resource_invariant_errors"] == []
    assert committed["claimed_resource_ids_by_su"] == {"24": [1]}


def test_paired_like24_open_mainline_roles_are_left_and_right() -> None:
    left = make_node(428, 14, target_hop1_degree=3)
    right = make_node(429, 14, target_hop1_degree=3)
    bridge = make_node(503, 27, target_hop1_degree=2)
    left_open = make_node(600, 11)
    right_open = make_node(601, 11)
    connect(left, bridge, fixed=True)
    connect(bridge, right, fixed=True)
    connect(left, left_open)
    connect(right, right_open)
    nodes = [left, right, bridge, left_open, right_open]
    tokens_by_id = {
        int(token.node_id): token for token in build_node_tokens(nodes)
    }
    fragment = TopologyFragment(
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        claimed_node_ids=[428, 429, 503],
        branch_contracts=[
            TopologyClosureSlot(
                slot_id="pair:428:branch:left:428",
                kind="H",
                anchor_node_id=428,
                anchor_role="left_like24_branch",
                required_su_sequence=[23, 22],
            ),
            TopologyClosureSlot(
                slot_id="pair:428:branch:right:429",
                kind="H",
                anchor_node_id=429,
                anchor_role="right_like24_branch",
                required_su_sequence=[23, 22],
            ),
        ],
        evidence={"center_node_ids": [428, 429]},
    )

    finalized = _finalize_fragment_edge_semantics(fragment, tokens_by_id)

    assert [
        (int(slot.anchor_node_id), str(slot.anchor_role))
        for slot in finalized.open_mainline_slots
    ] == [
        (428, "left_mainline"),
        (429, "right_mainline"),
    ]


def _like24_suppressed_support_physical_fixture():
    center = make_node(428, 20, target_hop1_degree=3)
    left = make_node(600, 11)
    right = make_node(601, 11)
    support = make_node(602, 11)
    connect(center, left)
    connect(center, right)
    connect(center, support)
    view = AssemblyFragmentView(
        unit_id=428,
        fragment_id=428,
        family="like24",
        class_type="24_A",
        topology_template="-24(?)-",
        source_node_ids=[428, 600, 601, 602],
        main_labels=["11", "single20d3", "11"],
        main_node_ids=[600, 428, 601],
        canonical_labels=["11", "24", "11"],
        open_left=False,
        open_right=False,
        evidence={"center_node_ids": [428]},
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes = [center, left, right, support]
    assembler.nodes_by_id = {
        int(node.global_id): node for node in assembler.nodes
    }
    return assembler, view


def test_physical_product_accepts_real_suppressed_center_neighbor() -> None:
    assembler, view = _like24_suppressed_support_physical_fixture()
    diagnostics = {}

    reason = chain_materializer_module.physical_product_reason(assembler,
        product_kind="flex_chain",
        source_views=[view],
        node_ids=[600, 428, 601],
        branch_id_map={},
        suppressed_support_node_ids=[602],
        diagnostics=diagnostics,
    )

    assert reason is None
    assert diagnostics["center_degree_audit"] == [{
        "center_id": 428,
        "hop1_ids": [600, 601, 602],
        "rendered_degree": 2,
        "suppressed_neighbor_ids": [602],
        "credited_suppressed_neighbor_ids": [602],
        "effective_degree": 3,
        "target_degree": 3,
    }]
    product = chain_materializer_module.make_product(assembler,
        product_kind="flex_chain",
        reason="suppressed_support_regression",
        source_views=[view],
        labels=["11", "single20d3", "11"],
        node_ids=[600, 428, 601],
        branch_map={},
        branch_id_map={},
        suppressed_support_node_ids=[602],
    )
    assert product is not None
    assert product.actual_topology_gid == "600-428-601"


@pytest.mark.parametrize("suppressed_ids", [[], [999]])
def test_physical_product_rejects_missing_or_unrelated_suppressed_neighbor(
    suppressed_ids,
) -> None:
    assembler, view = _like24_suppressed_support_physical_fixture()

    reason = chain_materializer_module.physical_product_reason(assembler,
        product_kind="flex_chain",
        source_views=[view],
        node_ids=[600, 428, 601],
        branch_id_map={},
        suppressed_support_node_ids=suppressed_ids,
    )

    assert reason == "center_degree_mismatch:428:rendered=2:target=3"


def test_physical_product_ignores_unrendered_fixed_support_context() -> None:
    """A donor view may mention a fixed support node omitted from the route.

    The support is credited as suppressed context; it must not be reported as
    a missing physical edge for the product being rendered.
    """
    center = make_node(411, 20, target_hop1_degree=1)
    support = make_node(529, 28)
    connect(center, support, fixed=True)
    view = AssemblyFragmentView(
        unit_id=411,
        fragment_id=411,
        family="like24",
        class_type="24_A",
        topology_template="-24(?)-",
        source_node_ids=[411, 529],
        main_labels=["single20d3"],
        main_node_ids=[411],
        canonical_labels=["24"],
        open_left=False,
        open_right=False,
        evidence={"center_node_ids": [411]},
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes = [center, support]
    assembler.nodes_by_id = {int(node.global_id): node for node in assembler.nodes}

    reason = chain_materializer_module.physical_product_reason(assembler,
        product_kind="flex_chain",
        source_views=[view],
        node_ids=[411],
        branch_id_map={},
        suppressed_support_node_ids=[529],
    )

    assert reason is None


def test_basic_d_routes_terminal_atomic_pair_as_flat_branch() -> None:
    """A SU17-SU18 pair attached through SU17 cannot enter the mainline."""
    center = make_node(25, 3, target_hop1_degree=2)
    branch_root = make_node(333, 17, target_hop1_degree=2)
    terminal = make_node(335, 18, target_hop1_degree=1)
    mainline_open = make_node(337, 17, target_hop1_degree=1)
    connect(center, branch_root)
    connect(branch_root, terminal, fixed=True)
    connect(center, mainline_open)
    nodes = [center, branch_root, terminal, mainline_open]

    classification = classify_topology_units(nodes)
    view = project_fragment_views(
        classification,
        {int(node.global_id): node for node in nodes},
    )[0]

    assert view.main_labels == ["3", "17"]
    assert view.main_node_ids == [25, 337]
    assert view.open_left is False
    assert view.open_right is True
    assert [
        stub.node_id_notation
        for stub in view.real_branch_stubs_by_anchor[0]
    ] == ["-333-335"]
    assert view.diagnostics["atomic_terminal_branch_paths"] == [[333, 335]]

    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "atomic_terminal_branch"},
    )
    assert chain_materializer_module.physical_product_reason(assembler,
        product_kind="flex_chain",
        source_views=[view],
        node_ids=[25, 337],
        branch_id_map={0: ["-333-335"]},
    ) is None


def test_fixed_contract_rejects_cross_fragment_terminal_pair() -> None:
    left = make_node(404, 19, target_hop1_degree=2, target_fixed_anchor_count=1)
    left_terminal = make_node(523, 28, target_hop1_degree=1)
    right = make_node(405, 19, target_hop1_degree=2, target_fixed_anchor_count=1)
    right_terminal = make_node(524, 28, target_hop1_degree=1)
    connect(left, left_terminal, fixed=True)
    connect(right, right_terminal, fixed=True)
    nodes = [left, left_terminal, right, right_terminal]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "cross_fragment_pair"},
    )
    assembler.nodes_by_id = {int(node.global_id): node for node in nodes}
    assembler.fixed_node_owner_by_id = {
        404: 404,
        523: 404,
        405: 405,
        524: 405,
    }
    reason = branch_closure_module.fixed_contract_flat_route_reason(
        assembler,
        {
            "base_path_node_ids": [404, 524],
            "base_path_labels": ["single19d3", "28"],
        },
        None,
    )
    assert reason == "fixed_route_edge_missing:404-524"


def test_fixed_contract_rejects_one_sided_fixed_edge_metadata() -> None:
    left = make_node(530, 19, target_hop1_degree=2)
    right = make_node(531, 28, target_hop1_degree=1)
    left.hop1_ids.append(531)
    right.hop1_ids.append(530)
    left.fixed_hop1_ids.add(531)
    assembler = ExternalTopologyAssembler(
        [left, right],
        TopologyClassificationResult(),
        flex_audit={"enforced": False, "scenario": "one_sided_fixed_edge"},
    )

    reason = branch_closure_module.fixed_contract_flat_route_reason(
        assembler,
        {
            "base_path_node_ids": [530, 531],
            "base_path_labels": ["single19d3", "28"],
        },
        None,
    )

    assert reason == "fixed_route_edge_missing:530-531"


def test_fixed_contract_does_not_expand_to_sibling_branch_owned_by_same_fragment() -> None:
    center = make_node(496, 24, target_hop1_degree=3)
    left_root = make_node(222, 19, target_hop1_degree=2)
    left_terminal = make_node(223, 28, target_hop1_degree=1)
    right_root = make_node(240, 19, target_hop1_degree=2)
    right_terminal = make_node(241, 28, target_hop1_degree=1)
    connect(center, left_root, fixed=True)
    connect(left_root, left_terminal, fixed=True)
    connect(center, right_root, fixed=True)
    connect(right_root, right_terminal, fixed=True)
    nodes = [center, left_root, left_terminal, right_root, right_terminal]
    assembler = ExternalTopologyAssembler(
        nodes,
        TopologyClassificationResult(
            fixed_node_owner_by_id={node.global_id: 496 for node in nodes},
        ),
        flex_audit={"enforced": False, "scenario": "sibling_fixed_branches"},
    )
    owner_view = AssemblyFragmentView(
        unit_id=496,
        fragment_id=496,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[node.global_id for node in nodes],
        main_labels=["24"],
        main_node_ids=[496],
        canonical_labels=["24"],
        open_left=True,
        open_right=True,
    )
    contract = {
        "center_node_id": 496,
        "base_path_node_ids": [222, 223],
        "base_path_labels": ["single19d3", "28"],
        "externally_closed": True,
        "tail_closure_kind": "none",
    }

    assert branch_closure_module.fixed_contract_flat_route_reason(
        assembler, contract, owner_view
    ) is None
    result, required_counts, blocker = branch_closure_module.resolve_fixed_branch_contract(
        assembler,
        contract,
        [],
        anchor_index=0,
        owner_view=owner_view,
    )

    assert blocker == ""
    assert required_counts == {}
    assert result is not None
    assert result.branch_topology == "-single19d3-28"
    assert result.branch_node_ids == "-222-223"
    assert result.rendered_node_ids == [222, 223]


def test_fixed_connector_with_two_children_becomes_structural_boundary() -> None:
    center = make_node(600, 24, target_hop1_degree=3)
    connector = make_node(601, 19, target_hop1_degree=3)
    first_terminal = make_node(602, 28, target_hop1_degree=1)
    second_terminal = make_node(603, 28, target_hop1_degree=1)
    connect(center, connector, fixed=True)
    connect(connector, first_terminal, fixed=True)
    connect(connector, second_terminal, fixed=True)
    tokens_by_id = {
        int(token.node_id): token
        for token in build_node_tokens(
            [center, connector, first_terminal, second_terminal]
        )
    }
    fragment = TopologyFragment(
        fragment_id=600,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[600, 601, 602, 603],
        claimed_node_ids=[600, 601, 602, 603],
        mainline_node_ids=[600],
        mainline_labels=["24"],
        evidence={
            "center_node_ids": [600],
            "fixed_connection_contracts": [{
                "center_node_id": 600,
                "connector_node_id": 601,
                "base_path_node_ids": [601, 602, 603],
                "base_path_labels": ["single19d3", "28", "28"],
                "role_candidates": ["branch_only"],
                "tail_closure_kind": "none",
                "externally_closed": True,
            }],
        },
    )

    finalized = _finalize_fragment_edge_semantics(fragment, tokens_by_id)
    contract = finalized.evidence["fixed_connection_contracts"][0]

    assert contract["base_path_node_ids"] == [601]
    assert contract["tail_closure_kind"] == "STRUCTURAL_BOUNDARY"
    assert contract["structural_boundary"]["fixed_child_node_ids"] == [602, 603]


def test_two_center_connectors_remain_sibling_fixed_contracts() -> None:
    center = make_node(620, 25, target_hop1_degree=4)
    left = make_node(621, 23, target_hop1_degree=2)
    left_terminal = make_node(622, 22, target_hop1_degree=1)
    right = make_node(623, 19, target_hop1_degree=2)
    right_terminal = make_node(624, 28, target_hop1_degree=1)
    connect(center, left, fixed=True)
    connect(left, left_terminal, fixed=True)
    connect(center, right, fixed=True)
    connect(right, right_terminal, fixed=True)
    tokens_by_id = {
        int(token.node_id): token
        for token in build_node_tokens(
            [center, left, left_terminal, right, right_terminal]
        )
    }
    fragment = TopologyFragment(
        fragment_id=620,
        family="su25",
        class_type="25_CD",
        topology_template="-25(-H)(-H)-",
        source_node_ids=[620, 621, 622, 623, 624],
        claimed_node_ids=[620, 621, 622, 623, 624],
        mainline_node_ids=[620],
        mainline_labels=["25"],
        evidence={
            "center_node_ids": [620],
            "fixed_connection_contracts": [
                {
                    "center_node_id": 620,
                    "connector_node_id": 621,
                    "role_candidates": ["branch_only"],
                    "base_path_node_ids": [621, 622],
                    "base_path_labels": ["23", "22"],
                    "tail_closure_kind": "none",
                    "externally_closed": True,
                },
                {
                    "center_node_id": 620,
                    "connector_node_id": 623,
                    "role_candidates": ["branch_only"],
                    "base_path_node_ids": [623, 624],
                    "base_path_labels": ["single19d3", "28"],
                    "tail_closure_kind": "none",
                    "externally_closed": True,
                },
            ],
        },
    )

    finalized = _finalize_fragment_edge_semantics(fragment, tokens_by_id)
    contracts = finalized.evidence["fixed_connection_contracts"]

    assert [item["base_path_node_ids"] for item in contracts] == [
        [621, 622],
        [623, 624],
    ]
    assert all(
        item["tail_closure_kind"] != "STRUCTURAL_BOUNDARY"
        for item in contracts
    )


def test_stage3_fixed_route_preflight_checks_selected_flex_only() -> None:
    center = make_node(610, 24, target_hop1_degree=3)
    connector = make_node(611, 19, target_hop1_degree=3)
    first_terminal = make_node(612, 28, target_hop1_degree=1)
    second_terminal = make_node(613, 28, target_hop1_degree=1)
    connect(center, connector, fixed=True)
    connect(connector, first_terminal, fixed=True)
    connect(connector, second_terminal, fixed=True)
    nodes = [center, connector, first_terminal, second_terminal]
    fragment = TopologyFragment(
        fragment_id=610,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[610, 611, 612, 613],
        claimed_node_ids=[610, 611, 612, 613],
        mainline_node_ids=[610],
        mainline_labels=["24"],
        evidence={
            "fixed_connection_contracts": [{
                "center_node_id": 610,
                "connector_node_id": 611,
                "base_path_node_ids": [611],
                "base_path_labels": ["single19d3"],
                "role_candidates": ["branch_only"],
                "tail_closure_kind": "STRUCTURAL_BOUNDARY",
                "externally_closed": False,
            }],
        },
    )
    classification = TopologyClassificationResult(
        fragments=[fragment],
        fixed_node_owner_by_id={node.global_id: 610 for node in nodes},
    )

    deferred = stage3_fixed_route_preflight(
        nodes,
        classification,
        {"selected_candidate_fragment_ids": []},
    )
    selected = stage3_fixed_route_preflight(
        nodes,
        classification,
        {"selected_candidate_fragment_ids": [610]},
    )

    assert deferred["ok"] is True
    assert selected["ok"] is False
    assert selected["failure_domain"] == "topology_materialization"


def test_su25_is_not_classified_as_terminal_endpoint() -> None:
    assert "25" not in TERMINAL_ENDPOINT_LABELS


def test_degree_complete_like24_does_not_restore_legacy_mainline_slot() -> None:
    center = make_node(
        427,
        20,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    fixed_anchor = make_node(0, 0, target_hop1_degree=1)
    open_left = make_node(112, 11, target_hop1_degree=2)
    open_right = make_node(113, 11, target_hop1_degree=2)
    connect(center, fixed_anchor, fixed=True)
    connect(center, open_left)
    connect(center, open_right)
    tokens_by_id = {
        int(token.node_id): token
        for token in build_node_tokens(
            [center, fixed_anchor, open_left, open_right]
        )
    }
    fragment = TopologyFragment(
        fragment_id=427,
        family="like24",
        class_type="24_A",
        topology_template="-24(?)-",
        source_node_ids=[0, 427],
        claimed_node_ids=[0, 427],
        open_slots=["OPEN"],
        branch_slots=["H", "H"],
        mainline_node_ids=[427],
        branch_contracts=[
            TopologyClosureSlot(
                slot_id=f"fragment:427:branch:427:{index}:H",
                kind="H",
                anchor_node_id=427,
                anchor_role="center_branch",
                required_su_sequence=[23, 22],
            )
            for index in range(2)
        ],
        evidence={"center_node_ids": [427]},
    )

    finalized = _finalize_fragment_edge_semantics(fragment, tokens_by_id)
    normalized = _normalize_fragment_contract(finalized)

    assert normalized.evidence["edge_semantics_finalized"] is True
    assert normalized.open_mainline_slots == []
    assert normalized.open_slots == []

    view = project_like24_fragment(
        normalized,
        {
            int(node.global_id): node
            for node in [center, fixed_anchor, open_left, open_right]
        },
    )
    assert view.open_left is False
    assert view.open_right is False
    assert is_chain_materializable_view(view) is False
    assert (
        validate_chain_product_shape(
            "side_chain",
            ["single20d3", "9"],
            [view],
            node_ids=[427, 0],
        )
        is None
    )


def test_degree_complete_like24_uses_branch_only_side_finalization(
    monkeypatch,
) -> None:
    view = AssemblyFragmentView(
        unit_id=427,
        fragment_id=427,
        family="like24",
        class_type="24_A",
        topology_template="-24(?)-",
        source_node_ids=[0, 112, 427],
        main_labels=["11", "20"],
        main_node_ids=[112, 427],
        canonical_labels=["11", "20"],
        open_left=False,
        open_right=False,
        unresolved_branch_kinds=["H", "H"],
        flex_reconcile_roles=["b_like_open"],
        can_ring=False,
        evidence={"edge_semantics_finalized": True},
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.fragment_state_by_id = {427: "available"}
    assembler.selected_flex_candidate_ids = set()
    assembler.ring_exhausted_fragment_ids = set()
    finalized = []

    def _finalize(_assembler, candidate, **kwargs):
        finalized.append((candidate.fragment_id, kwargs["reason"]))
        assembler.fragment_state_by_id[int(candidate.fragment_id)] = "completed"
        return True

    monkeypatch.setattr(
        chain_materializer_module, "finalize_single_view_product", _finalize
    )

    assert chain_materializer_module.assemble_remaining_b_d_as_side(assembler, []) is True
    assert finalized == [(427, "canonical_branch_only_closure")]


def _d_like_open_fixture(fragment_id: int = 610) -> AssemblyFragmentView:
    return AssemblyFragmentView(
        unit_id=fragment_id,
        fragment_id=fragment_id,
        family="chain",
        class_type="D",
        topology_template="11-23(-22)-11",
        source_node_ids=[fragment_id, fragment_id + 1],
        main_labels=["11", "23", "11"],
        main_node_ids=[fragment_id, fragment_id + 1, fragment_id + 2],
        canonical_labels=["11", "23", "11"],
        open_left=True,
        open_right=True,
        flex_reconcile_roles=["d_like_open"],
        preferred_product_type="flex_chain",
        can_flex=True,
        can_side=True,
        target_roles=["flex_chain", "side_chain"],
    )


def test_ring_exhausted_d_candidate_is_eligible_for_side_closure(
    monkeypatch,
) -> None:
    view = _d_like_open_fixture()
    view.can_ring = True
    view.target_roles.append("fatty_ring")
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {int(view.fragment_id): view}
    assembler.fragment_state_by_id = {int(view.fragment_id): "available"}
    assembler.selected_flex_candidate_ids = set()
    assembler.ring_exhausted_fragment_ids = {int(view.fragment_id)}
    assembler.materialization_failure_by_fragment_id = {}
    finalized = []

    def _finalize(_assembler, candidate, **kwargs):
        finalized.append((candidate.fragment_id, kwargs["product_kind"]))
        assembler.fragment_state_by_id[int(candidate.fragment_id)] = "completed"
        return True

    monkeypatch.setattr(
        chain_materializer_module, "finalize_with_two_end_closures", _finalize
    )

    assert chain_materializer_module.assemble_remaining_b_d_as_side(assembler, []) is True
    assert finalized == [(int(view.fragment_id), "side_chain")]


def test_selected_ring_capable_d_candidate_materializes_as_flex(
    monkeypatch,
) -> None:
    view = _d_like_open_fixture(611)
    view.can_ring = True
    view.target_roles.append("fatty_ring")
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {int(view.fragment_id): view}
    assembler.fragment_state_by_id = {int(view.fragment_id): "available"}
    assembler.selected_flex_candidate_ids = {int(view.fragment_id)}
    assembler.selected_flex_candidate_groups = [{
        "strategy": "d_double_closure",
        "fragment_ids": [int(view.fragment_id)],
    }]
    assembler.supplemental_flex_candidate_ids = {int(view.fragment_id)}
    assembler.ring_exhausted_fragment_ids = set()
    assembler.flex_bounds = {"enforced": True, "min": 1, "max": 1, "target": 1}
    assembler.flex_audit = {"scenario": "below_range"}
    assembler.products = []
    calls = []

    def _finalize(_assembler, candidate, **kwargs):
        calls.append((kwargs["product_kind"], kwargs.get("left_mode"), kwargs.get("right_mode")))
        assembler.fragment_state_by_id[int(candidate.fragment_id)] = "completed"
        assembler.products.append(SimpleNamespace(product_kind="flex_chain"))
        return True

    monkeypatch.setattr(
        chain_materializer_module, "finalize_with_two_end_closures", _finalize
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_b_plus_b_flex",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_d_double_flex",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_single_b_closures_for_flex",
        lambda *a, **k: False,
    )

    assert chain_materializer_module.assemble_selected_supplemental_flex(assembler, []) is True
    assert calls == [("flex_chain", "F1", "F1")]


def test_explicit_empty_contract_missing_does_not_fall_back_to_old_claim() -> None:
    contract = {
        "missing_counts": {},
        "diagnostics": {"claim_failure_missing_counts": {"11": 1}},
    }

    assert state_machine_module._contract_reported_missing_counts(contract) == {}


def test_terminal_proof_keeps_current_empty_missing_and_compact_live_state() -> None:
    allocator = PoolAllocator({"11": [1, 2], "23": [3]})
    assembler = SimpleNamespace(
        generation_id=17,
        current_phase="external_flex_closure",
        completed_phases={"explicit_branch_closure"},
        fragment_state_by_id={406: "available"},
        selected_flex_candidate_ids={406},
        selected_flex_candidate_groups=[{
            "strategy": "d_double_closure",
            "fragment_ids": [406],
        }],
        allocator=allocator,
        pending_contracts=[{
            "contract_id": 406,
            "target_role": "flex_chain",
            "continuation_ready": True,
            "claim_trace": [{"mode": "generic", "su_type": 11}],
        }],
        ring_recovery_attempts=[],
    )
    plan = SimpleNamespace(
        phase="external_topology",
        plan_kind="single_topology_transaction",
        blocked_reason="resource_contract_claimability_inconsistent",
        unresolved_counts={"11": 1},
        pending_contracts=[],
        planned_conversions=[],
        spectrum_candidates=[],
        histogram_adjustment_plan=None,
        phase_result=SimpleNamespace(
            rebuild_required=False,
            infeasibility_proof={
                "reason": "resource_contract_claimability_inconsistent",
                "required_counts": {"11": 2},
                "available_counts": {"11": 2},
                "missing_counts": {},
                "unresolved_counts": {"11": 1},
            },
        ),
        diagnostics={
            "ring_recovery_capacity_proof": {
                "recovery_allowed": False,
                "blocked_reason": "ordinary_conversion_not_exhausted",
            },
        },
        runtime={"external_assembler": assembler},
    )

    proof = block_c_module._stage3_terminal_infeasibility_proof(plan)

    assert proof["missing_counts"] == {}
    assert proof["claim_trace"] == [{"mode": "generic", "su_type": 11}]
    assert proof["recovery_proof"]["recovery_allowed"] is False
    live = proof["live_assembler_summary"]
    assert live["generation_id"] == 17
    assert live["current_phase"] == "external_flex_closure"
    assert live["continuation_ready_contract_ids"] == [406]
    assert live["allocator"]["raw_available_ids_by_su"] == {
        "11": [1, 2],
        "23": [3],
    }


def test_zero_conversion_transaction_stays_uncommitted_for_live_assembler() -> None:
    contract = TopologyClosureContract(
        contract_id=41,
        contract_kind="mainline_closure",
        phase="external_flex_closure",
        target_role="flex_chain",
    )

    plan = state_machine_module._transaction_plan(
        phase_eval=SimpleNamespace(assembly_view={}),
        contract=contract,
        pool={},
        H=torch.zeros(33, dtype=torch.long),
        E_target=torch.zeros(6, dtype=torch.float32),
        max_primary_adjustments=0,
        max_h_compensation_ops=0,
        h_tolerance=0.08,
        special_degree_meta=None,
    )

    transaction = plan.phase_result.transactions[0]
    assert plan.ok is False
    assert transaction.committed is False
    assert transaction.commit_scope == "uncommitted_live_assembler"
    assert transaction.rollback_reason == "live_assembler_materialization_required"


def test_conversion_sync_refreshes_pending_contract_for_live_continuation() -> None:
    donor_12 = make_node(158, 11)
    donor_13 = make_node(232, 11)
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes_by_id = {158: donor_12, 232: donor_13}
    assembler.allocator = PoolAllocator({
        "12": [158],
        "13": [232],
        "23": [467, 468],
    })
    assembler.pending_recycled_resource_conversion_ops = []
    assembler.pending_contracts = [{
        "contract_id": 406,
        "target_role": "flex_chain",
        "reason": "main_pool_shortage",
        "failure_domain": "",
        "required_counts": {"11": 2, "23": 2},
        "diagnostics": {
            "failure_domain": "",
            "missing_counts": {"11": 2},
            "claim_failure_missing_counts": {"11": 1},
            "claim_trace": [{
                "mode": "generic",
                "su_type": 11,
                "requested_count": 1,
                "success": False,
            }],
        },
    }]
    assembler.pending_units = [TopologyPendingUnit(
        unit_id=406,
        target_role="flex_chain",
        reason="main_pool_shortage",
        required_counts={"11": 2, "23": 2},
        claim_trace=[{"mode": "generic", "su_type": 11}],
        infeasibility_proof={
            "failure_domain": "resource_capacity",
            "missing_counts": {"11": 2},
        },
    )]

    assembler.sync_after_resource_adjustments(
        [donor_12, donor_13],
        H=torch.zeros(33, dtype=torch.long),
        applied_ops=[
            {"from_su": 12, "to_su": 11, "node_ids": [158]},
            {"from_su": 13, "to_su": 11, "node_ids": [232]},
        ],
    )

    contract = assembler.pending_contracts[0]
    assert contract["continuation_ready"] is True
    assert contract["missing_counts"] == {}
    assert contract["diagnostics"]["claim_trace"] == []
    assert contract["diagnostics"]["claim_failure_missing_counts"] == {}
    assert assembler.allocator.remaining_ids_by_su["11"] == [158, 232]
    assert assembler.pending_units[0].claim_trace == []

    # The next planner pass refreshes H/nodes before calling continuation.
    # That passive synchronization must not erase the accepted retry token.
    assembler.sync_after_resource_adjustments(
        [donor_12, donor_13],
        H=torch.zeros(33, dtype=torch.long),
    )
    assert assembler.pending_contracts[0]["continuation_ready"] is True


def test_in_range_flex_pending_still_runs_live_continuation(monkeypatch) -> None:
    assembler = SimpleNamespace(
        flex_audit={"enforced": True, "scenario": "in_range"},
        pending_contracts=[{
            "contract_id": 406,
            "target_role": "flex_chain",
            "continuation_ready": True,
        }],
        pending_units=[],
        completed_phases={
            "explicit_branch_closure",
            "real_like24_ring_assembly",
            "remaining_external_halfclosed_closure",
        },
        current_phase="external_flex_closure",
    )
    calls = []

    monkeypatch.setattr(
        chain_materializer_module, "available_branch_donors", lambda _: []
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "has_pending_contracts_for_subphase",
        lambda *_args, **_kwargs: False,
    )

    def retry(current, _donors, **kwargs):
        calls.append(dict(kwargs))
        current.pending_contracts = []
        return True

    monkeypatch.setattr(
        chain_materializer_module, "retry_ready_pending_contracts", retry
    )
    monkeypatch.setattr(
        chain_materializer_module, "flex_range_violation_reason", lambda _: None
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "available_remaining_side_views",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        ring_planner_module, "ring_resource_blockers", lambda _: {}
    )
    monkeypatch.setattr(
        ring_planner_module, "ring_materialization_blockers", lambda _: {}
    )
    monkeypatch.setattr(
        ownership_module, "available_views", lambda *_args, **_kwargs: []
    )
    run_external_assembly_phases(assembler)

    assert calls == [{"target_role": "flex_chain"}]
    assert assembler.current_phase == "complete"


def test_selected_flex_structure_signature_mismatch_is_structural_blocker(
    monkeypatch,
) -> None:
    view = _d_like_open_fixture(614)
    fragment = TopologyFragment(
        fragment_id=614,
        family="basic",
        class_type="D",
        topology_template="-23-",
        source_node_ids=[614],
        claimed_node_ids=[614],
        mainline_node_ids=[614],
        mainline_labels=["23"],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {614: view}
    assembler.fragment_by_id = {614: fragment}
    assembler.fragment_state_by_id = {614: "available"}
    assembler.selected_flex_candidate_ids = {614}
    assembler.selected_flex_candidate_groups = [{
        "group_id": "d_single:614",
        "strategy": "d_double_closure",
        "fragment_ids": [614],
        "fragment_structure_signatures": {"614": "stale-signature"},
        "fragment_structure_snapshots": {"614": {}},
    }]
    assembler.supplemental_flex_candidate_ids = {614}
    assembler.flex_bounds = {"enforced": True, "min": 1, "max": 1, "target": 1}
    assembler.flex_audit = {"scenario": "below_range"}
    assembler.products = []
    assembler.allocator = PoolAllocator({})
    assembler.materialization_failure_by_fragment_id = {}
    pending_calls = []

    monkeypatch.setattr(
        pending_module,
        "record_pending",
        lambda _assembler, **kwargs: pending_calls.append(kwargs),
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_b_plus_b_flex",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_d_double_flex",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_single_b_closures_for_flex",
        lambda *args, **kwargs: False,
    )

    assert fragment_structure_signature(fragment) != "stale-signature"
    assert chain_materializer_module.assemble_selected_supplemental_flex(assembler, []) is False
    assert assembler.fragment_state_by_id[614] == "blocked"
    assert pending_calls[0]["reason"] == "selected_flex_structure_signature_mismatch"
    failure = assembler.materialization_failure_by_fragment_id[614]
    assert failure["failure_domain"] == "topology_materialization"


def test_structural_flex_pending_blocks_ring_and_side_phases(monkeypatch) -> None:
    calls = []
    assembler = SimpleNamespace(
        flex_audit={"enforced": True, "scenario": "below_range"},
        pending_contracts=[],
    )
    monkeypatch.setattr(
        chain_materializer_module, "available_branch_donors", lambda _: []
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "close_explicit_flex_branches_if_needed",
        lambda _, donors: calls.append("explicit_flex"),
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "has_pending_contracts_for_subphase",
        lambda _, subphase: False,
    )

    def _selected_flex(donors):
        calls.append("supplemental_flex")
        assembler.pending_contracts.append({
            "target_role": "flex_chain",
            "failure_domain": "topology_materialization",
            "required_counts": {},
        })

    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_selected_supplemental_flex",
        lambda _, donors: _selected_flex(donors),
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "release_satisfied_flex_selection",
        lambda _: calls.append("release"),
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_flex",
        lambda _, donors: calls.append("unbounded_flex"),
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "flex_range_violation_reason",
        lambda _: None,
    )
    monkeypatch.setattr(
        ring_planner_module,
        "assemble_priority_rings",
        lambda _, donors: calls.append("ring"),
    )
    monkeypatch.setattr(
        ring_planner_module, "ring_resource_blockers", lambda _: {}
    )
    monkeypatch.setattr(
        ring_planner_module, "ring_materialization_blockers", lambda _: {}
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_side_products",
        lambda _, donors: calls.append("side"),
    )
    assembler.views = []
    assembler.fragment_state_by_id = {}
    monkeypatch.setattr(
        chain_materializer_module,
        "record_remaining_external_pending",
        lambda _: calls.append("pending"),
    )
    assembler.pending_units = []
    assembler.contract_definition_conflict_count = 0
    assembler.last_contract_definition_conflict = {}

    state = run_external_assembly_phases(assembler)

    assert calls == [
        "explicit_flex",
        "supplemental_flex",
        "release",
    ]
    assert state.external_subphases == [
        "explicit_branch_closure",
        "supplemental_flex_closure",
    ]


def test_spectrum_trial_rejects_fragment_projection_failure(monkeypatch) -> None:
    pipeline = object.__new__(InversePipelineV3)
    pipeline.layer1_assigner = SimpleNamespace(
        validate_graph_consistency=lambda **kwargs: (True, []),
    )
    pipeline.layer4_adjuster = SimpleNamespace(
        fixed_partition_meta={},
        _evaluate_required_hist_constraints=lambda *args, **kwargs: {"ok": True},
    )
    pipeline._sync_fixed_partition_meta_from_nodes = lambda nodes: None
    restored = []
    pipeline._set_fixed_partition_meta = lambda value: restored.append(value)
    monkeypatch.setattr(
        inverse_pipeline_module,
        "classify_topology_units",
        lambda nodes: TopologyClassificationResult(),
    )
    monkeypatch.setattr(
        inverse_pipeline_module,
        "preflight_fragment_view_projection",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "stage3_fragment_projection_invalid",
            "failure_domain": "topology_materialization",
        },
    )

    ok, reason = pipeline._validate_stage3_candidate(
        [],
        torch.zeros(33, dtype=torch.long),
        S_target=torch.zeros(1),
        E_target=torch.zeros(6),
        tag="Spectrum Trial",
    )

    assert ok is False
    assert reason == "stage3_fragment_projection_invalid"
    assert restored == [{}]


def test_selected_flex_failure_is_not_demoted_to_side(
    monkeypatch,
) -> None:
    view = _d_like_open_fixture(612)
    view.can_ring = True
    view.target_roles.append("fatty_ring")
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {int(view.fragment_id): view}
    assembler.fragment_state_by_id = {int(view.fragment_id): "available"}
    assembler.selected_flex_candidate_ids = {int(view.fragment_id)}
    assembler.selected_flex_candidate_groups = [{
        "strategy": "d_double_closure",
        "fragment_ids": [int(view.fragment_id)],
    }]
    assembler.supplemental_flex_candidate_ids = {int(view.fragment_id)}
    assembler.flex_bounds = {"enforced": True, "min": 1, "max": 1, "target": 1}
    assembler.flex_audit = {"scenario": "below_range"}
    assembler.products = []
    calls = []

    def _finalize(_assembler, candidate, **kwargs):
        calls.append((kwargs["product_kind"], kwargs.get("left_mode"), kwargs.get("right_mode")))
        return False

    monkeypatch.setattr(
        chain_materializer_module, "finalize_with_two_end_closures", _finalize
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_b_plus_b_flex",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_d_double_flex",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        chain_materializer_module,
        "assemble_remaining_external_single_b_closures_for_flex",
        lambda *a, **k: False,
    )

    assert chain_materializer_module.assemble_selected_supplemental_flex(assembler, []) is False
    assert calls == [("flex_chain", "F1", "F1")]
    assert assembler.fragment_state_by_id[int(view.fragment_id)] == "available"


def test_satisfied_flex_target_releases_unused_ring_candidate() -> None:
    view = _d_like_open_fixture(613)
    view.can_ring = True
    view.target_roles.append("fatty_ring")
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.view_by_id = {int(view.fragment_id): view}
    assembler.fragment_state_by_id = {int(view.fragment_id): "available"}
    assembler.selected_flex_candidate_ids = {int(view.fragment_id)}
    assembler.selected_flex_candidate_groups = [{
        "strategy": "d_double_closure",
        "fragment_ids": [int(view.fragment_id)],
    }]
    assembler.flex_bounds = {"enforced": True, "min": 1, "max": 1, "target": 1}
    assembler.flex_audit = {}
    assembler.products = [SimpleNamespace(product_kind="flex_chain")]
    assembler.pending_units = [SimpleNamespace(
        source_fragment_ids=[int(view.fragment_id)],
        claimed_source_node_ids=list(view.source_node_ids),
    )]
    assembler.pending_contracts = [{
        "source_fragment_ids": [int(view.fragment_id)],
        "claimed_source_node_ids": list(view.source_node_ids),
    }]
    assembler.ring_exhausted_fragment_ids = set()

    assert chain_materializer_module.release_satisfied_flex_selection(assembler, ) == [int(view.fragment_id)]
    assert assembler.selected_flex_candidate_ids == set()
    assert assembler.selected_flex_candidate_groups == []
    assert assembler.pending_units == []
    assert assembler.pending_contracts == []
    assert ring_policy_module.ring_candidate_pending_for_ring_stage(
        assembler, view
    ) is True


def test_attempt_slots_count_only_endcaps_for_existing_su23_bridges() -> None:
    view = _d_like_open_fixture(614)
    view.main_labels = ["23", "15", "15", "23"]
    view.main_node_ids = [700, 614, 615, 701]
    assembler = object.__new__(ExternalTopologyAssembler)

    slots = chain_materializer_module.attempt_mainline_slots(assembler,
        view,
        [("left", "F1"), ("right", "F1")],
    )

    assert [slot["required_su_sequence"] for slot in slots] == [[11], [11]]


def test_two_end_closure_drops_only_shared_materialized_bridge() -> None:
    payload = (["23", "11"], [483, 245])

    labels, node_ids = remove_materialized_bridge_from_payload(
        payload,
        bridge=("23", 483),
        materialized_node_ids=[245, 483, 423],
    )

    assert labels == ["11"]
    assert node_ids == [245]

    # A bridge not present in the transaction must remain a real part of the
    # closure payload; this helper is not a general route de-duplicator.
    labels, node_ids = remove_materialized_bridge_from_payload(
        payload,
        bridge=("23", 999),
        materialized_node_ids=[245, 483, 423],
    )
    assert labels == ["23", "11"]
    assert node_ids == [483, 245]


def test_branch_map_drops_mainline_context_from_donor_route() -> None:
    view = _d_like_open_fixture(615)
    view.main_labels = ["23", "29", "19d1"]
    view.main_node_ids = [423, 565, 339]
    assembler = object.__new__(ExternalTopologyAssembler)

    branch = BranchClosureResult(
        branch_topology="-28",
        branch_node_ids="-505-423-565-339",
        anchor_index=0,
    )
    _, branch_id_map = chain_materializer_module.build_branch_maps(assembler, view, [branch])

    assert branch_id_map == {0: ["-505-"]}


def test_ring_capable_view_isolated_from_side_fallback_failures(
    monkeypatch,
) -> None:
    view = AssemblyFragmentView(
        unit_id=428,
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        main_labels=["24", "23", "24"],
        main_node_ids=[428, 503, 429],
        canonical_labels=["24", "23", "24"],
        open_left=True,
        open_right=True,
        preferred_product_type="side_chain",
        flex_reconcile_roles=["b_like_open"],
        can_side=True,
        can_ring=True,
        target_roles=["side_chain", "fatty_ring"],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.fragment_state_by_id = {428: "available"}
    assembler.selected_flex_candidate_ids = set()
    assembler.materialization_failure_by_fragment_id = {}
    assembler.composition_attempt_failure_count = 0
    assembler.last_composition_attempt_failure = {}

    assert chain_materializer_module.available_remaining_side_views(assembler,
        require_generic_join=False
    ) == []
    assert chain_materializer_module.available_remaining_b_open_views(assembler,
        require_generic_join=False
    ) == []

    ring_failure = {
        "attempted_product_kind": "fatty_ring_side",
        "source_fragment_ids": [428],
        "validation_reason": "center_degree_mismatch:428:rendered=2:target=3",
    }
    ownership_module.record_composition_failure_entry(assembler, ring_failure)
    ownership_module.record_composition_failure_entry(assembler, {
        "attempted_product_kind": "side_chain",
        "source_fragment_ids": [23, 428],
        "validation_reason": "side_failure_must_not_replace_ring_failure",
    })

    assert assembler.materialization_failure_by_fragment_id[428] == ring_failure
    # This fragment is dual-use.  A failed ring attempt keeps it reserved
    # while ring alternatives are still being explored, but it must not turn
    # a side/flex failure into a hard ring blocker.
    assert ring_planner_module.ring_materialization_blockers(assembler) == {}
    assert chain_materializer_module.available_remaining_side_views(assembler,
        require_generic_join=False
    ) == []

    assembler.ring_exhausted_fragment_ids = {428}
    assert chain_materializer_module.available_remaining_side_views(assembler,
        require_generic_join=False
    ) == [view]
    assert chain_materializer_module.available_remaining_b_open_views(assembler,
        require_generic_join=False
    ) == [view]

    captured = {}
    assembler.pending_units = []
    assembler.ring_pending_reasons = {
        428: "center_degree_mismatch:428:rendered=2:target=3"
    }
    assembler.ring_infeasible_contracts = {
        428: {"required_counts": {"11": 2, "22": 2, "23": 4}}
    }
    assembler.materialization_failure_by_fragment_id[428] = {
        "attempted_product_kind": "side_chain",
        "source_fragment_ids": [23, 428],
        "validation_reason": "side_failure_must_be_ignored",
        "attempted_resource_counts": {"11": 1, "22": 1, "23": 2},
        "attempted_mainline_slots": [{"slot_id": "wrong_side_slot"}],
    }
    assembler.ring_exhausted_fragment_ids = set()
    assembler.allocator = SimpleNamespace(
        available_counts=lambda: {
            "11": 4,
            "22": 8,
            "23": 21,
        }
    )
    monkeypatch.setattr(
        pending_module,
        "record_pending",
        lambda _assembler, **kwargs: captured.update(kwargs),
    )

    chain_materializer_module.record_remaining_external_pending(assembler, )

    assert captured["pending_kind"] == "b_like_open"
    assert captured["required_counts"] == {"11": 1, "22": 1, "23": 2}
    assert captured["diagnostics"]["materialization_failure"][
        "validation_reason"
    ] == "side_failure_must_be_ignored"
    assert captured["mainline_slots_override"] == [
        {"slot_id": "wrong_side_slot"}
    ]


def test_pair_core_external_side_ring_uses_real_pool_endpoints() -> None:
    view = AssemblyFragmentView(
        unit_id=428,
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        main_labels=["24", "23", "24"],
        main_node_ids=[428, 503, 429],
        canonical_labels=["24", "23", "24"],
        open_left=True,
        open_right=True,
        preferred_product_type="flex_chain",
        can_ring=True,
        target_roles=["fatty_ring"],
        ring_backbone_labels={"side": ["24", "23", "24"]},
        ring_backbone_node_ids={"side": [428, 503, 429]},
        ring_backbone_span={"side": 3},
        ring_like24_units=2,
        unresolved_branch_kinds_by_anchor={0: ["H"], 2: ["H"]},
        evidence={
            "ring_path_capabilities": {"side": ["inner_pair"]},
            "center_node_ids": [428, 429],
            # Classification diagnostics retain physical center IDs, while
            # the projected view above is authoritative for path indices.
            "required_branch_kinds_by_anchor": {
                "428": ["H"],
                "429": ["H"],
            },
            "required_branch_closures_by_anchor_in_ring": {
                "side": {},
            },
            "ring_contracts": {},
        },
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {428: view}
    assembler.fragment_state_by_id = {428: "available"}
    assembler._ring_plan_excluded_pool_node_ids = set()
    assembler.nodes = [
        make_node(428, 14, target_hop1_degree=3),
        make_node(429, 14, target_hop1_degree=3),
        make_node(503, 27, target_hop1_degree=2),
        make_node(600, 11),
        make_node(601, 11),
        make_node(602, 23),
        make_node(603, 23),
        make_node(604, 23),
        make_node(605, 22),
        make_node(606, 22),
    ]
    assembler.nodes_by_id = {
        int(node.global_id): node for node in assembler.nodes
    }
    assembler.allocator = PoolAllocator({
        "11": [600, 601],
        "22": [605, 606],
        "23": [602, 603, 604],
        "24": [],
    })
    assembler.last_ring_materialization_reason = ""
    assembler.last_ring_missing_counts = {}
    assembler.generation_id = 1

    plan = ring_planner_module.plan_side_ring_templates(assembler, [view])

    assert plan.template_complete is True
    assert plan.missing_slots == []
    assert sum(
        int(segment.get("backbone_span", 0) or 0)
        for segment in plan.ordered_segments
    ) == 6
    assert [
        list(segment.get("backbone_labels", []) or [])
        for segment in plan.ordered_segments
    ] == [["11", "23"], ["24", "23", "24"], ["11"]]

    materialization = ring_materializer_module.resolve_ring_candidate_topology(
        assembler,
        plan,
        skeleton_kind="side",
        branch_donors=[],
    )
    product = ring_materializer_module.product_from_ring_topology_contract(
        assembler,
        product_kind="fatty_ring_side",
        reason="ring:side_template",
        materialization=materialization,
        diagnostics={
            "ring_template_kind": "side",
            "ring_has_fixed_anchor": False,
        },
    )

    assert product is not None
    assert product.actual_topology_su == "11-23-24(-23-22)-23-24(-23-22)-11"
    assert product.actual_topology_gid == "600-602-428(-603-605)-503-429(-604-606)-601"
    assert product.ring_contract is not None
    assert product.ring_contract.complete is True
    assert product.ring_contract.audit_reason == ""
    assert product.ring_contract.required_resource_counts == {
        "11": 2,
        "22": 2,
        "23": 3,
    }
    assert {
        (int(path["anchor_node_id"]), tuple(path["node_ids"]))
        for path in product.ring_contract.branch_paths
    } == {
        (428, (603, 605)),
        (429, (604, 606)),
    }


def test_ring_candidate_claims_specific_support_before_generic_branch() -> None:
    view = AssemblyFragmentView(
        unit_id=1,
        fragment_id=1,
        family="like24",
        class_type="24_C",
        topology_template="24",
        source_node_ids=[1],
        main_labels=["24"],
        main_node_ids=[1],
        canonical_labels=["24"],
        open_left=False,
        open_right=False,
        branch_anchor_index=0,
        unresolved_branch_kinds_by_anchor={0: ["H"]},
        ring_backbone_labels={"side": ["24"]},
        ring_backbone_node_ids={"side": [1]},
        ring_backbone_span={"side": 1},
        ring_like24_units=1,
        can_ring=True,
        evidence={
            "center_node_ids": [1],
            "required_branch_closures_by_anchor_in_ring": {
                "side": {"center": ["H"]},
            },
        },
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.view_by_id = {1: view}
    assembler.nodes = [
        make_node(1, 24),
        make_node(100, 24),
        make_node(200, 22),
        make_node(201, 22),
        make_node(300, 23),
    ]
    assembler.nodes_by_id = {
        int(node.global_id): node for node in assembler.nodes
    }
    assembler.allocator = PoolAllocator({
        "22": [200, 201],
        "23": [300],
        "24": [100],
    })
    assembler.last_ring_materialization_reason = ""
    assembler.last_ring_missing_counts = {}

    plan = RingCandidatePlan(
        ring_kind="fatty_ring_side",
        skeleton_kind="side_ring_plan",
        source_views=[view],
        ordered_segments=[
            {
                "kind": "view",
                "role": "middle",
                "candidate_id": "view:side:middle:1",
                "fragment_id": 1,
                "view": view,
                "reverse": False,
            },
            {
                "kind": "pool",
                "role": "middle",
                "candidate_id": "pool:side:middle:24:100",
                "pool_su": 24,
                "pool_node_id": 100,
                "support_specific_ids_by_su": {22: [200]},
            },
        ],
        template_complete=True,
    )

    materialization = ring_materializer_module.resolve_ring_candidate_topology(
        assembler,
        plan,
        skeleton_kind="side",
        branch_donors=[],
    )

    assert materialization.topology_notation == "24(-23-22)-24"
    assert materialization.node_id_notation == "1(-300-201)-100"
    assert materialization.pool_node_ids_by_su == {
        "22": [200, 201],
        "23": [300],
        "24": [100],
    }
    assert assembler.last_ring_materialization_reason == ""


def test_ring_candidate_rejects_duplicate_specific_support_identity() -> None:
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.allocator = PoolAllocator({
        "11": [200],
        "24": [100, 101],
    })
    assembler.last_ring_materialization_reason = ""
    assembler.last_ring_missing_counts = {}
    plan = RingCandidatePlan(
        ring_kind="fatty_ring_side",
        skeleton_kind="side_ring_plan",
        ordered_segments=[
            {
                "kind": "pool",
                "role": "left_anchor",
                "candidate_id": "pool:side:left_anchor:24:100",
                "pool_su": 24,
                "pool_node_id": 100,
                "support_specific_ids_by_su": {11: [200]},
            },
            {
                "kind": "pool",
                "role": "right_anchor",
                "candidate_id": "pool:side:right_anchor:24:101",
                "pool_su": 24,
                "pool_node_id": 101,
                "support_specific_ids_by_su": {11: [200]},
            },
        ],
        template_complete=True,
    )

    with pytest.raises(
        ValueError,
        match="ring_candidate_specific_id_collision:200",
    ):
        ring_materializer_module.resolve_ring_candidate_topology(
            assembler,
            plan,
            skeleton_kind="side",
            branch_donors=[],
        )

    assert assembler.allocator.available_counts() == {"11": 1, "24": 2}


def test_structurally_missing_ring_skeleton_has_no_generic_slot_demand(
    monkeypatch,
) -> None:
    view = AssemblyFragmentView(
        unit_id=428,
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        main_labels=["24", "23", "24"],
        main_node_ids=[428, 503, 429],
        canonical_labels=["24", "23", "24"],
        open_left=True,
        open_right=True,
        can_ring=True,
        target_roles=["fatty_ring"],
    )
    fragment = TopologyFragment(
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        claimed_node_ids=[428, 429, 503],
        open_mainline_slots=[
            TopologyClosureSlot(
                slot_id="left",
                kind="F1",
                anchor_node_id=428,
                required_su_sequence=[23, 11],
            ),
            TopologyClosureSlot(
                slot_id="right",
                kind="F1",
                anchor_node_id=429,
                required_su_sequence=[23, 11],
            ),
        ],
        branch_contracts=[
            TopologyClosureSlot(
                slot_id="left-h",
                kind="H",
                anchor_node_id=428,
                required_su_sequence=[23, 22],
            ),
            TopologyClosureSlot(
                slot_id="right-h",
                kind="H",
                anchor_node_id=429,
                required_su_sequence=[23, 22],
            ),
        ],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.views = [view]
    assembler.classification = SimpleNamespace(fragments=[fragment])
    assembler.fragment_state_by_id = {428: "available"}
    assembler.products = []
    assembler.pending_units = []
    assembler.ring_pending_reasons = {428: "fixed_top_anchor_missing"}
    assembler.ring_infeasible_contracts = {}
    assembler.materialization_failure_by_fragment_id = {}
    assembler.allocator = SimpleNamespace(
        available_counts=lambda: {
            "11": 2,
            "22": 3,
            "23": 6,
        }
    )
    captured = {}
    monkeypatch.setattr(
        pending_module,
        "record_pending_contract",
        lambda _assembler, **kwargs: captured.setdefault("contract", kwargs),
    )

    record_pending(
        assembler,
        unit_id=428,
        pending_kind="pending_ring_candidate",
        views=[view],
        required_counts={},
        available_counts={"11": 2, "22": 3, "23": 6},
        reason="fixed_top_anchor_missing",
        partial_render=view.partial_render(),
        actual_render=view.pending_render(),
    )

    assert assembler.pending_units[0].required_counts == {}
    assert captured["contract"]["required_counts"] == {}
    assert captured["contract"]["contract_kind"] == "structural_blocker"
    assert captured["contract"]["reason"] == "fixed_top_anchor_missing"


def test_multicap_raw_su24_uses_one_cap_and_remains_ring_resource() -> None:
    fixtures = [
        ("A", [11, 11, 23], [2], [3, 4], []),
        ("B", [11, 11, 22], [2], [3], [4]),
        ("A", [11, 11, 11], [2], [3, 4], []),
    ]
    for expected_role, neighbor_types, cap_ids, open_ids, terminal_ids in fixtures:
        center = make_node(1, 24)
        neighbors = [
            make_node(index + 2, su_type)
            for index, su_type in enumerate(neighbor_types)
        ]
        for neighbor in neighbors:
            connect(center, neighbor)
        nodes = [center, *neighbors]
        role, details = classify_raw_su24_semantics(
            su_type=24,
            neighbors=[
                {
                    "node_id": int(node.global_id),
                    "su_type": int(node.su_type),
                    "subtype": str(int(node.su_type)),
                }
                for node in neighbors
            ],
        )
        assert role == expected_role
        assert details["cap_node_ids"] == cap_ids
        assert details["open_neighbor_ids"] == open_ids
        assert details["terminal_node_ids"] == terminal_ids

        result = classify_topology_units(nodes)
        binding = next(
            item for item in result.su24_support_bindings
            if int(item["su24_node_id"]) == int(center.global_id)
        )
        assert binding["role"] == expected_role
        assert binding["fixed_support_ids_by_su"].get("11", []) == cap_ids
        assert binding["open_neighbor_ids"] == open_ids
        assert int(center.global_id) in result.resource_pool["remaining_ids_by_su"]["24"]
        assert all(
            int(fragment.fragment_id) != int(center.global_id)
            for fragment in result.fragments
        )


def test_raw_su24_open_neighbor_is_not_claimed_as_fixed_branch() -> None:
    center = make_node(494, 24, target_hop1_degree=3)
    left_cap = make_node(174, 11)
    right_cap = make_node(229, 11)
    branch_partner = make_node(326, 15)
    branch_attachment = make_node(327, 15)
    connect(center, left_cap)
    connect(center, right_cap)
    connect(center, branch_attachment, fixed=True)
    connect(branch_attachment, branch_partner, fixed=True)
    nodes = [center, left_cap, right_cap, branch_partner, branch_attachment]

    classification = classify_topology_units(nodes)
    binding = next(
        item for item in classification.su24_support_bindings
        if int(item["su24_node_id"]) == int(center.global_id)
    )
    assert binding == {
        "su24_node_id": 494,
        "role": "A",
        "fixed_support_ids_by_su": {"11": [174]},
        "fixed_support_ids": [174],
        "open_neighbor_ids": [229, 327],
    }
    assert 327 not in binding["fixed_support_ids"]
    assert 326 not in binding["fixed_support_ids"]
    assert all(
        not ({494, 326, 327} <= set(fragment.claimed_node_ids))
        for fragment in classification.fragments
    )
    assert 494 in classification.resource_pool["remaining_ids_by_su"]["24"]


def test_double19d3_neighbor_is_independent_from_raw_su24() -> None:
    center = make_node(497, 24, target_hop1_degree=3)
    cap = make_node(166, 11, target_hop1_degree=1)
    open_neighbor = make_node(448, 23, target_hop1_degree=1)
    branch_root = make_node(
        430,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    fixed_support = make_node(512, 28, target_hop1_degree=1)
    fixed_connector = make_node(572, 29, target_hop1_degree=2)
    fixed_tail = make_node(346, 19, target_hop1_degree=1)
    for left, right in [
        (branch_root, fixed_support),
        (branch_root, fixed_connector),
        (fixed_connector, fixed_tail),
    ]:
        connect(left, right, fixed=True)
    connect(center, cap)
    connect(center, open_neighbor)
    connect(center, branch_root)
    nodes = [
        center,
        cap,
        open_neighbor,
        branch_root,
        fixed_support,
        fixed_connector,
        fixed_tail,
    ]
    classification = classify_topology_units(nodes)
    binding = next(
        item for item in classification.su24_support_bindings
        if int(item["su24_node_id"]) == 497
    )
    assert binding == {
        "su24_node_id": 497,
        "role": "A",
        "fixed_support_ids_by_su": {"11": [166]},
        "fixed_support_ids": [166],
        "open_neighbor_ids": [430, 448],
    }
    assert all(
        not (
            int(dict(contract or {}).get("center_node_id", -1)) == 497
            and int(dict(contract or {}).get("connector_node_id", -1)) == 430
        )
        for fragment in classification.fragments
        for contract in list(
            dict(fragment.evidence or {}).get("fixed_connection_contracts", [])
            or []
        )
    )
    assert 430 not in binding["fixed_support_ids"]
    assert 497 in classification.resource_pool["remaining_ids_by_su"]["24"]
    assert stage3_fixed_route_preflight(
        nodes,
        classification,
        {"hard_explicit_flex_fragment_ids": []},
    )["ok"] is True


def test_topology_materialization_failure_uses_specific_stage3_labels() -> None:
    assert _phase_infeasibility_labels("topology_materialization") == (
        "stage3_topology_materialization_invalid",
        "sequential_phase_topology_materialization_invalid",
    )
    assert _phase_infeasibility_labels("") == (
        "stage3_completion_invariants_failed",
        "sequential_phase_infeasible_after_progress",
    )


def test_topology_materialization_failure_is_not_resource_infeasible_plan() -> None:
    H = torch.zeros(33, dtype=torch.long)
    phase_eval = SimpleNamespace(
        assembly_view={"resource_ledger": {}},
        classification_view={},
        round_context=None,
    )
    plan = _finalize_phase_plan(
        phase_eval=phase_eval,
        phase="external_topology_assembly",
        H_before=H,
        H_after=H,
        initial_pool={},
        final_pool={},
        groups=[],
        transactions=[],
        rotation_before=0,
        rotation_after=0,
        h_tolerance=0.08,
        infeasible_contracts=[{
            "contract_id": 496,
            "target_role": "flex_chain",
            "required_counts": {},
            "proof": {
                "failure_domain": "topology_materialization",
                "reason": "nested_branch_topology_forbidden",
            },
        }],
    )

    assert plan.reason == "phase_topology_materialization_invalid"
    assert plan.blocked_reason == "stage3_topology_materialization_invalid"
    assert plan.phase_result.stop_reason == (
        "phase_topology_materialization_invalid"
    )
    assert plan.requested_counts == {}


@pytest.mark.parametrize(
    (
        "role",
        "neighbor_types",
        "profile",
        "claimed_ids",
        "context_ids",
        "expected_main_ids",
        "expected_branch_ids",
    ),
    [
        (
            "A",
            [11, 23, 25],
            {"cap_node_ids": [1], "terminal_node_ids": [], "open_neighbor_ids": [2, 3]},
            [100, 1],
            [2, 3],
            [1, 100],
            [],
        ),
        (
            "B",
            [11, 22, 23],
            {"cap_node_ids": [1], "terminal_node_ids": [2], "open_neighbor_ids": [3]},
            [100, 1, 2],
            [3],
            [1, 100],
            [2],
        ),
        (
            "C",
            [0, 2, 3],
            {"cap_node_ids": [], "terminal_node_ids": [], "open_neighbor_ids": [1, 2, 3]},
            [100],
            [1, 2, 3],
            [100],
            [],
        ),
        (
            "D",
            [22, 0, 2],
            {"cap_node_ids": [], "terminal_node_ids": [1], "open_neighbor_ids": [2, 3]},
            [100, 1],
            [2, 3],
            [100],
            [1],
        ),
    ],
)
def test_raw_su24_projection_keeps_open_neighbors_as_context_only(
    role,
    neighbor_types,
    profile,
    claimed_ids,
    context_ids,
    expected_main_ids,
    expected_branch_ids,
) -> None:
    center = make_node(100, 24, target_hop1_degree=3)
    neighbors = [
        make_node(index, su_type)
        for index, su_type in enumerate(neighbor_types, start=1)
    ]
    for neighbor in neighbors:
        connect(center, neighbor, fixed=True)
    fragment = TopologyFragment(
        fragment_id=100,
        family="like24",
        class_type=f"24_{role}",
        topology_template="-24(?)-",
        source_node_ids=claimed_ids,
        claimed_node_ids=claimed_ids,
        context_node_ids=context_ids,
        evidence={
            "center_node_ids": [100],
            "raw_su24_role": role,
            "raw_su24_profile": profile,
            "raw_su24_real_neighbors": True,
        },
    )
    nodes_by_id = {
        int(node.global_id): node for node in [center, *neighbors]
    }

    view = project_like24_fragment(fragment, nodes_by_id)
    validate_fragment_view_projection(fragment, view)

    assert view.main_node_ids == expected_main_ids
    assert [
        int(node_id)
        for stub in view.real_branch_stubs_by_anchor.get(
            view.branch_anchor_index, []
        )
        for node_id in stub.source_node_ids
    ] == expected_branch_ids
    assert set(context_ids).isdisjoint(view_render_route_node_ids(view))
    assert set(fragment.claimed_node_ids) <= view_render_route_node_ids(view)


def test_raw_su24_open_atomic_pair_is_not_rendered_as_a_branch() -> None:
    center = make_node(494, 24, target_hop1_degree=3)
    left = make_node(321, 0)
    right = make_node(328, 2)
    branch = make_node(326, 15)
    branch_partner = make_node(327, 15)
    for neighbor in [left, right, branch]:
        connect(center, neighbor, fixed=True)
    connect(branch, branch_partner, fixed=True)
    nodes = [center, left, right, branch, branch_partner]
    fragment = TopologyFragment(
        fragment_id=494,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[494],
        claimed_node_ids=[494],
        context_node_ids=[321, 328, 326, 327],
        evidence={
            "center_node_ids": [494],
            "raw_su24_role": "C",
            "raw_su24_profile": {
                "cap_node_ids": [],
                "terminal_node_ids": [],
                "open_neighbor_ids": [321, 328, 326],
            },
            "raw_su24_real_neighbors": True,
            "atomic_groups": [{
                "group_id": 326,
                "kind": "double_bond_pair",
                "node_ids": [326, 327],
                "su_pair": [15, 15],
            }],
            "atomic_group_node_ids": [326, 327],
        },
    )
    nodes_by_id = {int(node.global_id): node for node in nodes}

    view = project_like24_fragment(fragment, nodes_by_id)
    validate_fragment_view_projection(fragment, view)

    rendered_branch_ids = [
        int(node_id)
        for stub in view.real_branch_stubs_by_anchor.get(
            view.branch_anchor_index, []
        )
        for node_id in stub.source_node_ids
    ]
    assert view.main_node_ids == [494]
    assert rendered_branch_ids == []
    assert {321, 328, 326, 327}.isdisjoint(
        view_render_route_node_ids(view)
    )


def test_like24_projection_routes_owned_fixed_component_missing_from_family_mainline() -> None:
    center = make_node(494, 20, target_hop1_degree=3)
    open_left = make_node(600, 11)
    open_right = make_node(601, 11)
    fixed_branch = make_node(326, 15)
    fixed_partner = make_node(327, 15)
    connect(center, open_left)
    connect(center, open_right)
    connect(center, fixed_branch, fixed=True)
    connect(fixed_branch, fixed_partner, fixed=True)
    nodes = [center, open_left, open_right, fixed_branch, fixed_partner]
    fragment = TopologyFragment(
        fragment_id=494,
        family="like24",
        class_type="24_C",
        topology_template="-24(?)-",
        source_node_ids=[494, 326, 327],
        claimed_node_ids=[494, 326, 327],
        evidence={
            "center_node_ids": [494],
            "atomic_groups": [{
                "group_id": 326,
                "kind": "double_bond_pair",
                "node_ids": [326, 327],
                "su_pair": [15, 15],
            }],
            "atomic_group_node_ids": [326, 327],
        },
    )
    classification = TopologyClassificationResult(fragments=[fragment])

    view = project_fragment_views(
        classification,
        {int(node.global_id): node for node in nodes},
    )[0]

    assert view.diagnostics["auto_routed_claimed_fixed_components"] == [{
        "anchor_node_id": 494,
        "anchor_index": 0,
        "node_ids": [326, 327],
    }]
    assert [
        int(node_id)
        for stub in view.real_branch_stubs_by_anchor[0]
        for node_id in stub.source_node_ids
    ] == [326, 327]
    assert set(fragment.claimed_node_ids) <= view_render_route_node_ids(view)


def test_donor_branch_keeps_direct_fixed_support_route() -> None:
    edges, parsed = nested_path_edges("-411(529)-447-163")
    assert parsed == {411, 529, 447, 163}
    assert edges == {(411, 529), (411, 447), (163, 447)}


def test_spectrum_candidate_rejects_duplicate_node_and_aromatic_floor() -> None:
    duplicate_node = make_node(1, 11)
    duplicate = Stage3SpectrumCandidate(operations=[
        {"node_id": 1, "from_su": 11, "to_su": 12},
        {"node_id": 1, "from_su": 11, "to_su": 12},
    ])
    nodes_out, H_out, proof = apply_spectrum_candidate(
        nodes=[duplicate_node],
        H=histogram([duplicate_node]),
        candidate=duplicate,
    )
    assert nodes_out is None and H_out is None
    assert proof["reason"] == "spectrum_candidate_real_id_mismatch"

    aromatic_nodes = [make_node(node_id, 12) for node_id in range(1, 4)]
    at_floor = Stage3SpectrumCandidate(operations=[
        {"node_id": 1, "from_su": 12, "to_su": 13},
    ])
    nodes_out, H_out, proof = apply_spectrum_candidate(
        nodes=aromatic_nodes,
        H=histogram(aromatic_nodes),
        candidate=at_floor,
    )
    assert nodes_out is None and H_out is None
    assert proof["reason"] == "spectrum_candidate_aromatic_minimum_preserved"


def test_spectrum_profile_weights_taper_and_normalize_only_overlap() -> None:
    profiles = _spectrum_profile_fixture()
    validated, audit = _validated_spectrum_profiles(profiles)
    assert audit["reason"] == "ok"
    assert validated is not None
    ppm = np.asarray([5.0, 6.0, 7.0, 13.0, 14.0, 15.0])
    weights = _spectrum_profile_weights(ppm, validated)
    su22_row = SPECTRUM_RESOURCE_TYPES.index(22)

    assert weights[su22_row, 0] == pytest.approx(0.0)
    assert weights[su22_row, 1] == pytest.approx(0.5)
    assert weights[su22_row, 2] == pytest.approx(1.0)
    assert weights[su22_row, 4] == pytest.approx(0.5)
    assert weights[su22_row, 5] == pytest.approx(0.0)

    profiles[23] = {
        "mu_common_min": 10.0,
        "mu_q05": 12.0,
        "mu_median": 15.0,
        "mu_q95": 18.0,
        "mu_common_max": 20.0,
    }
    validated, audit = _validated_spectrum_profiles(profiles)
    assert audit["reason"] == "ok"
    assert validated is not None

    ppm = np.asarray([11.0])
    weights = _spectrum_profile_weights(ppm, validated)
    su23_row = SPECTRUM_RESOURCE_TYPES.index(23)

    assert weights[su22_row, 0] > 0.0
    assert weights[su23_row, 0] > 0.0
    assert float(weights[:, 0].sum()) == pytest.approx(1.0)


def test_spectrum_profile_missing_type_has_no_coarse_fallback() -> None:
    profiles = _spectrum_profile_fixture()
    profiles.pop(24)
    node = _node_with_mu(1, 22, 10.0)

    candidates, audit = generate_spectrum_candidates(
        H=histogram([node]),
        nodes=[node],
        difference_spectrum={
            "ppm": [10.0, 35.0],
            "diff": [-10.0, 10.0],
        },
        profile_stats=profiles,
        hop1_template_options=_spectrum_hop1_fixture(),
        spectrum_node_ids_by_su={22: [1]},
        aromatic_donor_ids_by_su={},
    )

    assert candidates == []
    assert audit["reason"] == "spectrum_profile_unavailable"
    assert audit["missing_su_types"] == [24]


def test_spectrum_hop1_profile_missing_type_has_no_peak_fallback() -> None:
    node = _node_with_mu(1, 22, 10.0)
    hop1_options = _spectrum_hop1_fixture()
    hop1_options.pop(24)

    candidates, audit = generate_spectrum_candidates(
        H=histogram([node]),
        nodes=[node],
        difference_spectrum={"ppm": [10.0, 35.0], "diff": [-10.0, 10.0]},
        profile_stats=_spectrum_profile_fixture(),
        hop1_template_options=hop1_options,
        spectrum_node_ids_by_su={22: [1]},
        aromatic_donor_ids_by_su={},
    )

    assert candidates == []
    assert audit["reason"] == "spectrum_hop1_profile_unavailable"
    assert audit["hop1_missing_su_types"] == [24]


def test_spectrum_candidate_uses_layer2_mu_donor_and_positive_target_peak() -> None:
    ppm = np.linspace(0.0, 180.0, 1801)
    profiles = _spectrum_profile_fixture()
    donor_off_peak = _node_with_mu(1, 22, 7.0)
    donor_on_peak = _node_with_mu(2, 22, 10.0)
    diff = (
        -8.0 / ((ppm - 10.0) ** 2 + 1.0)
        + 8.0 / ((ppm - 35.0) ** 2 + 1.0)
    )

    candidates, audit = generate_spectrum_candidates(
        H=histogram([donor_off_peak, donor_on_peak]),
        nodes=[donor_off_peak, donor_on_peak],
        difference_spectrum={"ppm": ppm, "diff": diff},
        profile_stats=profiles,
        hop1_template_options=_spectrum_hop1_fixture(),
        spectrum_node_ids_by_su={22: [1, 2]},
        aromatic_donor_ids_by_su={},
        candidate_limit=8,
        max_primary_adjustments=1,
    )

    candidate = next(
        item
        for item in candidates
        if item.diagnostics["seed_direction"] == [22, 23]
    )
    operation = candidate.operations[0]
    assert operation["node_id"] == 2
    assert operation["source_mu"] == pytest.approx(10.0)
    assert operation["target_mu_goal"] == pytest.approx(35.0, abs=0.11)
    assert operation["predicted_sse_gain"] > 0.0
    assert audit["profile_source"] == "su_nmr_common_range_filtered.csv"


def test_spectrum_target_peak_is_discrete_filtered_hop1_template() -> None:
    ppm = np.linspace(0.0, 180.0, 1801)
    donor = _node_with_mu(1, 22, 10.0)
    hop1_options = _spectrum_hop1_fixture()
    hop1_options[23] = [
        {
            "hop1_tuple": (11, 22),
            "mu_median": 34.0,
            "sample_weight_sum": 100.0,
            "n_templates": 4,
        },
        {
            "hop1_tuple": (12, 23),
            "mu_median": 37.0,
            "sample_weight_sum": 20.0,
            "n_templates": 2,
        },
    ]
    diff = (
        -8.0 / ((ppm - 10.0) ** 2 + 1.0)
        + 8.0 / ((ppm - 36.8) ** 2 + 1.0)
    )

    candidates, _audit = generate_spectrum_candidates(
        H=histogram([donor]),
        nodes=[donor],
        difference_spectrum={"ppm": ppm, "diff": diff},
        profile_stats=_spectrum_profile_fixture(),
        hop1_template_options=hop1_options,
        spectrum_node_ids_by_su={22: [1]},
        aromatic_donor_ids_by_su={},
        max_primary_adjustments=1,
    )

    operation = next(
        candidate.operations[0]
        for candidate in candidates
        if candidate.diagnostics["seed_direction"] == [22, 23]
    )
    assert operation["target_mu_goal"] == pytest.approx(37.0)
    assert operation["target_mu_goal"] != pytest.approx(36.8)
    assert operation["simulated_target_hop1"] == [12, 23]


def test_spectrum_multistep_updates_residual_and_never_reuses_node() -> None:
    ppm = np.linspace(0.0, 180.0, 1801)
    nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 24, 55.0),
    ]
    diff = (
        -8.0 / ((ppm - 10.0) ** 2 + 1.0)
        + 8.0 / ((ppm - 35.0) ** 2 + 1.0)
        - 7.0 / ((ppm - 55.0) ** 2 + 1.0)
        + 7.0 / ((ppm - 110.0) ** 2 + 1.0)
    )

    candidates, _audit = generate_spectrum_candidates(
        H=histogram(nodes),
        nodes=nodes,
        difference_spectrum={"ppm": ppm, "diff": diff},
        profile_stats=_spectrum_profile_fixture(),
        hop1_template_options=_spectrum_hop1_fixture(),
        spectrum_node_ids_by_su={22: [1], 24: [2]},
        aromatic_donor_ids_by_su={},
        max_primary_adjustments=10,
    )

    candidate = next(item for item in candidates if len(item.operations) == 2)
    node_ids = [int(operation["node_id"]) for operation in candidate.operations]
    directions = {
        (int(operation["from_su"]), int(operation["to_su"]))
        for operation in candidate.operations
    }
    assert node_ids == [1, 2]
    assert len(node_ids) == len(set(node_ids))
    assert len(directions) == 2
    assert {source for source, _target in directions} == {22, 24}
    assert all(
        float(operation["predicted_sse_gain"]) > 0.0
        for operation in candidate.operations
    )


def test_spectrum_seed_enumeration_uses_exact_donor_template_triples() -> None:
    ppm = np.linspace(0.0, 50.0, 501)
    donor_rows = [
        {"node_id": 1, "su_type": 22, "mu": 10.0},
        {"node_id": 2, "su_type": 22, "mu": 12.0},
    ]
    target_rows = [
        {
            "su_type": 23,
            "hop1_tuple": (11, 22),
            "mu_median": 35.0,
            "sample_weight_sum": 20.0,
            "n_templates": 2,
            "profile_support": 1.0,
        },
        {
            "su_type": 23,
            "hop1_tuple": (12, 23),
            "mu_median": 37.0,
            "sample_weight_sum": 10.0,
            "n_templates": 1,
            "profile_support": 1.0,
        },
    ]
    donor_peaks = np.stack([
        _lorentzian_peak(ppm, row["mu"]) for row in donor_rows
    ])
    target_peaks = np.stack([
        _lorentzian_peak(ppm, row["mu_median"]) for row in target_rows
    ])
    residual = (
        -8.0 * donor_peaks[0]
        -7.0 * donor_peaks[1]
        +8.0 * target_peaks[0]
        +7.0 * target_peaks[1]
    )
    score_state = _atomic_score_state(
        residual=residual,
        donor_rows=donor_rows,
        donor_peaks=donor_peaks,
        target_rows=target_rows,
        target_peaks=target_peaks,
    )
    H = torch.zeros(33, dtype=torch.long)
    H[22] = 2

    operations = _ranked_atomics_for_direction(
        residual=residual,
        source_su=22,
        target_su=23,
        donor_rows=donor_rows,
        donor_peaks=donor_peaks,
        donor_peak_norms=np.sum(donor_peaks * donor_peaks, axis=1),
        target_rows=target_rows,
        target_peaks=target_peaks,
        target_peak_norms=np.sum(target_peaks * target_peaks, axis=1),
        donor_target_dot=donor_peaks @ target_peaks.T,
        score_state=score_state,
        used_node_ids=set(),
        H_state=H,
        max_results=10,
    )

    exact_triples = {
        (
            operation["node_id"],
            operation["to_su"],
            tuple(operation["simulated_target_hop1"]),
        )
        for operation in operations
    }
    assert exact_triples == {
        (1, 23, (11, 22)),
        (1, 23, (12, 23)),
        (2, 23, (11, 22)),
        (2, 23, (12, 23)),
    }
    assert all(operation["predicted_sse_gain"] > 0.0 for operation in operations)


def test_spectrum_candidate_generation_is_bounded_unique_and_deterministic() -> None:
    profiles = _spectrum_profile_fixture()
    nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 23, 35.0),
        _node_with_mu(3, 24, 55.0),
        _node_with_mu(4, 11, 110.0),
    ]
    for node_id in range(5, 9):
        nodes.append(_node_with_mu(node_id, 12, 135.0))
    for node_id in range(9, 13):
        nodes.append(_node_with_mu(node_id, 13, 160.0))
    ppm = np.linspace(0.0, 180.0, 1801)
    diff = np.zeros_like(ppm)
    for mu in (10.0, 55.0, 135.0):
        diff -= 7.0 / ((ppm - mu) ** 2 + 1.0)
    for mu in (35.0, 110.0, 160.0):
        diff += 7.0 / ((ppm - mu) ** 2 + 1.0)
    kwargs = {
        "H": histogram(nodes),
        "nodes": nodes,
        "difference_spectrum": {"ppm": ppm, "diff": diff},
        "profile_stats": profiles,
        "hop1_template_options": _spectrum_hop1_fixture(),
        "spectrum_node_ids_by_su": {
            11: [4], 22: [1], 23: [2], 24: [3],
        },
        "aromatic_donor_ids_by_su": {
            12: list(range(5, 9)), 13: list(range(9, 13)),
        },
        "candidate_limit": 8,
        "max_primary_adjustments": 10,
    }

    first, first_audit = generate_spectrum_candidates(**kwargs)
    second, _ = generate_spectrum_candidates(**kwargs)
    first_payload = [candidate.to_dict() for candidate in first]
    second_payload = [candidate.to_dict() for candidate in second]

    assert first_payload == second_payload
    assert 1 <= len(first) <= 8
    assert first_audit["seed_atomic_count"] > first_audit["seed_direction_count"]
    assert all(1 <= candidate.primary_adjustment_count <= 10 for candidate in first)
    assert len({candidate.histogram_signature for candidate in first}) == len(first)
    assert all(
        candidate.diagnostics["h_compensation_count"] == 0
        and candidate.diagnostics["final_balance_operations"] == []
        and all(
            operation.get("operation_kind") != "h_compensation"
            for operation in candidate.operations
        )
        for candidate in first
    )
    assert all(
        len({operation["node_id"] for operation in candidate.operations})
        == len(candidate.operations)
        for candidate in first
    )


def test_spectrum_candidate_generation_rejects_nonpositive_sse_gain() -> None:
    node = _node_with_mu(1, 22, 10.0)
    ppm = np.linspace(0.0, 180.0, 1801)

    candidates, audit = generate_spectrum_candidates(
        H=histogram([node]),
        nodes=[node],
        difference_spectrum={"ppm": ppm, "diff": np.zeros_like(ppm)},
        profile_stats=_spectrum_profile_fixture(),
        hop1_template_options=_spectrum_hop1_fixture(),
        spectrum_node_ids_by_su={22: [1]},
        aromatic_donor_ids_by_su={},
    )

    assert candidates == []
    assert audit["reason"] == "no_ranked_spectrum_candidate"


def test_spectrum_candidate_runs_one_final_two_part_balance(monkeypatch) -> None:
    nodes = [_node_with_mu(1, 22, 10.0)]
    nodes.extend(_node_with_mu(node_id, 12, 135.0) for node_id in range(2, 6))
    nodes.append(_node_with_mu(6, 11, 110.0))
    H = histogram(nodes)
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={
            11: [6], 12: list(range(2, 6)), 13: [],
            22: [1], 23: [], 24: [],
        },
    )
    plan = Stage3AdjustmentPlan(runtime={"iteration_topology_ledger": ledger})
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:test-final-balance",
        inverse_signature="test-primary-inverse",
        operations=[{"node_id": 1, "from_su": 22, "to_su": 23}],
        primary_adjustment_count=1,
        diagnostics={"bundle_signature": "test-final-balance"},
    )
    balance_calls = 0
    original_balance = block_c_module._append_stage3_parity_balance

    def count_balance_calls(**kwargs):
        nonlocal balance_calls
        balance_calls += 1
        return original_balance(**kwargs)

    monkeypatch.setattr(
        block_c_module,
        "_append_stage3_parity_balance",
        count_balance_calls,
    )

    prepared, candidate_nodes, candidate_H, diagnostics = (
        _prepare_spectrum_candidate_for_rebuild(
            plan=plan,
            candidate=candidate,
            nodes=nodes,
            H=H,
            special_degree_meta=None,
            balance_cursor=0,
        )
    )

    balance_operations = prepared.diagnostics["final_balance_operations"]
    assert balance_calls == 1
    assert prepared.diagnostics["primary_inverse_signature"] == (
        "test-primary-inverse"
    )
    assert prepared.inverse_signature != "test-primary-inverse"
    assert diagnostics["ok"] is True
    assert candidate_nodes is not None and candidate_H is not None
    assert [operation["operation_kind"] for operation in balance_operations] == [
        "parity_balance", "port_role_balance",
    ]
    assert len({operation["node_id"] for operation in balance_operations}) == 2
    assert 1 not in {operation["node_id"] for operation in balance_operations}
    assert sum(
        operation["operation_kind"] == "parity_balance"
        for operation in balance_operations
    ) == 1
    assert sum(
        operation["operation_kind"] == "port_role_balance"
        for operation in balance_operations
    ) == 1


@pytest.mark.parametrize(
    ("actual_gain", "expected_accepted"),
    [(0.0009, True), (0.0005, True), (0.0004, False)],
)
def test_spectrum_trial_accepts_only_actual_layer2_threshold(
    actual_gain: float,
    expected_accepted: bool,
) -> None:
    nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 22, 10.0),
    ]
    H = histogram(nodes)
    baseline_r2 = 0.5
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([-1.0, 1.0]),
        "r2": baseline_r2,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:threshold-test",
        histogram_signature="threshold-primary-h",
        inverse_signature="threshold-inverse",
        operations=[{"node_id": 1, "from_su": 22, "to_su": 24}],
        primary_adjustment_count=1,
        diagnostics={
            "bundle_signature": "threshold-bundle",
            "primary_histogram_signature": "threshold-primary-h",
            "predicted_global_sse_gain": 1.0,
            "final_balance_operations": [],
        },
    )

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        def __init__(self):
            self.calls = 0
            self._spectrum_search_memory = {}

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    plan_kind="remaining_pool_spectrum_candidate_set",
                    spectrum_candidates=[candidate],
                    diagnostics={"external_stage_completed": True},
                    runtime={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
                runtime={},
            )

    def rebuild_fn(*, H_seed, nodes_seed, **_kwargs):
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = baseline_r2 + float(actual_gain)
        return list(nodes_seed), H_seed.detach().clone(), candidate_diff, candidate_diff["r2"]

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_candidate_limit=8,
        spectrum_min_r2_gain=0.0005,
    )

    assert result.ok is True
    assert result.accepted_cycles == (1 if expected_accepted else 0)
    assert float(result.diff["r2"]) == pytest.approx(
        baseline_r2 + actual_gain if expected_accepted else baseline_r2
    )
    assert int(result.H[24].item()) == (1 if expected_accepted else 0)
    assert int(result.H[22].item()) == (1 if expected_accepted else 2)


def test_replanned_spectrum_candidate_batch_reenters_evaluation() -> None:
    nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 22, 10.0),
    ]
    H = histogram(nodes)
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([-1.0, 1.0]),
        "r2": 0.5,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }

    def candidate(candidate_id: str, node_id: int) -> Stage3SpectrumCandidate:
        return Stage3SpectrumCandidate(
            candidate_id=candidate_id,
            histogram_signature=f"{candidate_id}-histogram",
            inverse_signature=f"{candidate_id}-inverse",
            operations=[{"node_id": node_id, "from_su": 22, "to_su": 24}],
            primary_adjustment_count=1,
            diagnostics={
                "bundle_signature": f"{candidate_id}-bundle",
                "primary_histogram_signature": f"{candidate_id}-histogram",
                "predicted_global_sse_gain": 1.0,
                "final_balance_operations": [],
            },
        )

    first_candidate = candidate("spectrum:first-invalid", 1)
    second_candidate = candidate("spectrum:second-valid", 2)

    class ReplanningAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        def __init__(self):
            self.calls = 0
            self._spectrum_search_memory = {}

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                candidates = [first_candidate]
            elif self.calls in {2, 3}:
                candidates = [second_candidate]
            else:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_complete",
                    phase="complete",
                    plan_kind="stage3_state_machine_complete",
                    phase_result=Stage3PhaseResult(
                        active_phase="complete",
                        phase_completed=True,
                        rebuild_required=False,
                    ),
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_spectrum_candidates_ready",
                phase="remaining_pool_spectrum_rebalance",
                plan_kind="remaining_pool_spectrum_candidate_set",
                spectrum_candidates=candidates,
                diagnostics={"external_stage_completed": True},
            )

    adjuster = ReplanningAdjuster()

    def rebuild_fn(*, H_seed, nodes_seed, **_kwargs):
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = 0.51
        return list(nodes_seed), H_seed.detach().clone(), candidate_diff, 0.51

    def validate_fn(*, nodes, **_kwargs):
        invalid_first = any(
            int(node.global_id) == 1 and int(node.su_type) == 24
            for node in nodes
        )
        return (not invalid_first), (
            "first_candidate_invalid" if invalid_first else "ok"
        )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=adjuster,
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rebuild_fn,
        validate_fn=validate_fn,
        spectrum_min_r2_gain=0.0005,
    )

    assert result.ok is True
    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert adjuster.calls == 4
    assert int(result.nodes[0].su_type) == 22
    assert int(result.nodes[1].su_type) == 24


def test_spectrum_trial_r2_tie_ignores_predicted_sse() -> None:
    nodes = [_node_with_mu(1, 22, 10.0), _node_with_mu(2, 22, 10.0)]
    H = histogram(nodes)
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([-1.0, 1.0]),
        "r2": 0.5,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }

    def candidate(candidate_id: str, node_id: int, predicted_gain: float):
        return Stage3SpectrumCandidate(
            candidate_id=candidate_id,
            histogram_signature=f"{candidate_id}-h",
            inverse_signature=f"{candidate_id}-inverse",
            operations=[{
                "node_id": node_id,
                "from_su": 22,
                "to_su": 24,
            }],
            primary_adjustment_count=1,
            diagnostics={
                "bundle_signature": f"{candidate_id}-bundle",
                "primary_histogram_signature": f"{candidate_id}-h",
                "predicted_global_sse_gain": predicted_gain,
                "final_balance_operations": [],
            },
        )

    candidates = [
        candidate("a-lower-prediction", 1, 1.0),
        candidate("z-higher-prediction", 2, 100.0),
    ]

    class TieAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        def __init__(self):
            self.calls = 0
            self._spectrum_search_memory = {}

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    spectrum_candidates=candidates,
                    diagnostics={},
                    runtime={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                ),
                diagnostics={},
                runtime={},
            )

    def rebuild_fn(*, H_seed, nodes_seed, **_kwargs):
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = 0.51
        return list(nodes_seed), H_seed.clone(), candidate_diff, 0.51

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=TieAdjuster(),
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert int(result.nodes[0].su_type) == 24
    assert int(result.nodes[1].su_type) == 22


def test_spectrum_trial_uses_configured_cycle_window_and_topology_continues(
    capsys,
) -> None:
    initial_nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 22, 10.0),
    ]
    H = histogram(initial_nodes)
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([-1.0, 1.0]),
        "r2": 0.5,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }
    spectrum_candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:cycle-twenty-final",
        histogram_signature="cycle-twenty-final-h",
        inverse_signature="cycle-twenty-final-inverse",
        operations=[{"node_id": 1, "from_su": 22, "to_su": 24}],
        primary_adjustment_count=1,
        diagnostics={
            "bundle_signature": "cycle-twenty-final-bundle",
            "primary_histogram_signature": "cycle-twenty-final-h",
            "predicted_global_sse_gain": 1.0,
            "final_balance_operations": [],
        },
    )
    spectrum_flags = []

    class FinalizingAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0

        def __init__(self):
            self.calls = 0
            self._spectrum_search_memory = {}

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **kwargs):
            self.calls += 1
            spectrum_flags.append(bool(kwargs["enable_post_external_spectrum"]))
            if self.calls == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    plan_kind="remaining_pool_spectrum_candidate_set",
                    spectrum_candidates=[spectrum_candidate],
                    diagnostics={"external_stage_completed": True},
                    runtime={},
                )
            if self.calls == 2:
                # The accepted spectrum candidate performed a fresh rebuild,
                # so the next planner call is the next Stage3 cycle.
                assert kwargs["enable_post_external_spectrum"] is False
                H_after = kwargs["H"].clone()
                histogram_plan = HistogramAdjustmentPlan(
                    groups=[
                        AtomicAdjustmentGroup(
                            primary_op=HistogramConversion(
                                from_su=24,
                                to_su=22,
                                node_ids=[1],
                                reason="complete_topology_after_spectrum_window",
                            ),
                            h_compensation_ops=[HistogramConversion(
                                from_su=22,
                                to_su=24,
                                node_ids=[2],
                                reason="complete_topology_after_spectrum_window",
                            )],
                            combined_delta={},
                        ),
                    ],
                    H_before=kwargs["H"],
                    H_after=H_after,
                    inventory_before={22: 1, 24: 1},
                )
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_rebuild_required",
                    phase="internal_topology",
                    plan_kind="sequential_phase_transactions",
                    histogram_adjustment_plan=histogram_plan,
                    phase_result=Stage3PhaseResult(
                        active_phase="internal_topology",
                        phase_completed=False,
                        rebuild_required=True,
                    ),
                    diagnostics={},
                    runtime={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
                runtime={},
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, plan, **_kwargs):
            return SimpleNamespace(
                changed=True,
                nodes=[make_node(1, 22), make_node(2, 24)],
                H=plan.histogram_adjustment_plan.H_after.clone(),
                requested_counts={},
                unresolved_counts={},
                applied_ops=[
                    {"node_id": 1, "from_su": 24, "to_su": 22},
                    {"node_id": 2, "from_su": 22, "to_su": 24},
                ],
                request_ops=[],
                rejected_ops=[],
            )

    def rebuild_fn(*, H_seed, phase, nodes_seed, **_kwargs):
        r2 = 0.6 if phase == "remaining_pool_spectrum_rebalance" else 0.55
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = r2
        return list(nodes_seed), H_seed.clone(), candidate_diff, r2

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FinalizingAdjuster(),
        H=H,
        nodes=initial_nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
        iteration_offset=2,
        enable_post_external_spectrum=True,
        spectrum_active_through_cycle=3,
    )

    assert result.stop_reason == "stage3_complete"
    assert result.ok is True
    assert spectrum_flags == [True, False, False]
    assert int(result.H[22].item()) == 1
    assert int(result.H[24].item()) == 1
    assert int(result.nodes[0].su_type) == 22
    assert int(result.nodes[1].su_type) == 24
    assert float(result.diff["r2"]) == pytest.approx(0.55)
    assert result.diagnostics["spectrum_rebalance_finalized"] is True
    assert result.diagnostics["spectrum_finalized_on_cycle"] == 4
    assert result.diagnostics["spectrum_active_through_cycle"] == 3
    assert "[Spectrum Trial Finalized]" in capsys.readouterr().out


def test_spectrum_trial_rejects_negative_active_cycle_window() -> None:
    with pytest.raises(
        ValueError,
        match="spectrum_active_through_cycle must be >= 0",
    ):
        run_block_c_resource_adjustment_loop_impl(
            adjuster=SimpleNamespace(),
            H=torch.zeros(30, dtype=torch.long),
            nodes=[],
            E_target=None,
            spectrum_active_through_cycle=-1,
        )


@pytest.mark.parametrize(
    ("actual_gain", "expected_accepted"),
    [(0.0004, False), (0.0008, True)],
)
def test_product_owned_su24_spectrum_trial_is_transactional(
    actual_gain: float,
    expected_accepted: bool,
) -> None:
    nodes = [_node_with_mu(0, 24, 55.0), _node_with_mu(1, 24, 55.0)]
    connect(nodes[0], nodes[1])
    H = histogram(nodes)
    baseline_r2 = 0.5
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([1.0, -1.0]),
        "r2": baseline_r2,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }
    committed = {
        "product_id": "external:ring:product-owned-su24",
        "product_kind": "fatty_ring_side",
        "phase": "external_fatty_ring_closure",
        "fixed_source_node_ids": [],
        "claimed_resource_ids_by_su": {"24": [0]},
        "closed": True,
        "provisional": False,
    }
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:product-owned-su24",
        histogram_signature="product-owned-su24-h",
        inverse_signature="product-owned-su24-inverse",
        operations=[{
            "node_id": 0,
            "from_su": 24,
            "to_su": 22,
            "source_mu": 55.0,
            "simulated_target_hop1": [11, 23],
            "target_mu_goal": 10.0,
            "predicted_sse_gain": 1.0,
        }],
        primary_adjustment_count=1,
        diagnostics={
            "bundle_signature": "product-owned-su24-bundle",
            "primary_histogram_signature": "product-owned-su24-h",
            "predicted_global_sse_gain": 1.0,
            "final_balance_operations": [],
        },
    )

    class ProductAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        def __init__(self):
            self.calls = []
            self._spectrum_search_memory = {}

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **kwargs):
            self.calls.append({
                "progress_phase": kwargs["progress_phase"],
                "generation_id": kwargs["generation_id"],
            })
            if len(self.calls) == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    plan_kind="remaining_pool_spectrum_candidate_set",
                    spectrum_candidates=[candidate],
                    diagnostics={"external_stage_completed": True},
                    runtime={
                        "iteration_topology_ledger": SimpleNamespace(
                            completed_products=[committed]
                        ),
                    },
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
                runtime={},
            )

    adjuster = ProductAdjuster()

    def rebuild_fn(*, H_seed, nodes_seed, **_kwargs):
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = baseline_r2 + float(actual_gain)
        return (
            list(nodes_seed),
            H_seed.detach().clone(),
            candidate_diff,
            candidate_diff["r2"],
        )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=adjuster,
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.0005,
        outer_cycle=4,
    )

    assert result.ok is True
    assert result.accepted_cycles == (1 if expected_accepted else 0)
    assert len(adjuster.calls) == 2
    if expected_accepted:
        assert adjuster.calls[1]["progress_phase"] is (
            Stage3ProgressPhase.FLEX_RECONCILIATION
        )
        assert adjuster.calls[1]["generation_id"] == 4_000_001
        assert int(result.H[22].item()) == 1
        assert int(result.H[24].item()) == 1
        assert int(result.nodes[0].su_type) == 22
        assert not hasattr(result.nodes[0], "simulated_target_hop1")
    else:
        assert int(result.H[22].item()) == 0
        assert int(result.H[24].item()) == 2
        assert float(result.diff["r2"]) == pytest.approx(baseline_r2)
        assert int(result.nodes[0].su_type) == 24
        assert result.nodes[0].hop1_ids == [1]
        assert result.nodes[1].hop1_ids == [0]


def test_spectrum_search_memory_is_reused_for_same_layer2_baseline() -> None:
    nodes = [
        _node_with_mu(1, 22, 10.0),
        _node_with_mu(2, 22, 10.0),
    ]
    H = histogram(nodes)
    baseline_diff = {
        "ppm": np.asarray([10.0, 55.0]),
        "diff": np.asarray([-1.0, 1.0]),
        "r2": 0.5,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }
    candidate = Stage3SpectrumCandidate(
        candidate_id="spectrum:memory-test",
        histogram_signature="memory-final-h",
        inverse_signature="memory-inverse",
        operations=[{"node_id": 1, "from_su": 22, "to_su": 24}],
        primary_adjustment_count=1,
        diagnostics={
            "bundle_signature": "memory-bundle",
            "primary_histogram_signature": "memory-primary-h",
            "predicted_global_sse_gain": 1.0,
        },
    )

    class MemoryAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        def __init__(self):
            self._spectrum_search_memory = {}
            self.run_number = 1
            self.calls_this_run = 0
            self.second_run_kwargs = None

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        @staticmethod
        def _get_su_common_stats():
            return _spectrum_profile_fixture()

        def plan_topology_resource_adjustments(self, **kwargs):
            self.calls_this_run += 1
            if self.run_number == 1 and self.calls_this_run == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_spectrum_candidates_ready",
                    phase="remaining_pool_spectrum_rebalance",
                    spectrum_candidates=[candidate],
                    diagnostics={},
                    runtime={},
                )
            if self.run_number == 2 and self.calls_this_run == 1:
                self.second_run_kwargs = dict(kwargs)
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                ),
                diagnostics={},
                runtime={},
            )

    adjuster = MemoryAdjuster()

    def rejected_rebuild(*, H_seed, nodes_seed, **_kwargs):
        candidate_diff = dict(baseline_diff)
        candidate_diff["r2"] = 0.5004
        return list(nodes_seed), H_seed.detach().clone(), candidate_diff, 0.5004

    first = run_block_c_resource_adjustment_loop_impl(
        adjuster=adjuster,
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
        rebuild_fn=rejected_rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        spectrum_min_r2_gain=0.0005,
    )
    assert first.accepted_cycles == 0

    adjuster.run_number = 2
    adjuster.calls_this_run = 0
    second = run_block_c_resource_adjustment_loop_impl(
        adjuster=adjuster,
        H=H,
        nodes=nodes,
        E_target=None,
        difference_spectrum=baseline_diff,
    )
    assert second.ok is True
    assert adjuster.second_run_kwargs is not None
    assert {
        "memory-primary-h", "memory-final-h",
    } <= set(
        adjuster.second_run_kwargs["rejected_spectrum_histogram_signatures"]
    )
    assert "memory-bundle" in set(
        adjuster.second_run_kwargs["rejected_spectrum_bundle_signatures"]
    )
    assert "memory-inverse" in set(
        adjuster.second_run_kwargs["forbidden_spectrum_inverse_signatures"]
    )


def test_stage3_chemical_signature_ignores_rebuilt_edges_only() -> None:
    first = [make_node(1, 22), make_node(2, 23), make_node(3, 11)]
    second = [make_node(1, 22), make_node(2, 23), make_node(3, 11)]
    connect(first[0], first[1])
    connect(second[1], second[2])
    H = histogram(first)

    assert _stage3_physical_state_signature(first, H) != (
        _stage3_physical_state_signature(second, H)
    )
    assert _stage3_chemical_state_signature(first, H) == (
        _stage3_chemical_state_signature(second, H)
    )


def test_spectrum_inverse_guard_filters_only_the_exact_next_bundle() -> None:
    E_target = torch.tensor([100.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    H_before = torch.zeros(33, dtype=torch.long)
    H_before[22] = 1
    forward_node = _node_with_mu(1, 22, 10.0)
    forward, _ = generate_spectrum_candidates(
        H=H_before,
        nodes=[forward_node],
        difference_spectrum={
            "ppm": [10.0, 35.0],
            "diff": [-10.0, 10.0],
            "r2": 0.1,
        },
        profile_stats=_spectrum_profile_fixture(),
        hop1_template_options=_spectrum_hop1_fixture(),
        spectrum_node_ids_by_su={22: [1]},
        aromatic_donor_ids_by_su={},
        candidate_limit=1,
        max_primary_adjustments=1,
        E_target=E_target,
    )
    assert len(forward) == 1
    operation = dict(forward[0].operations[0])

    H_after = torch.zeros(33, dtype=torch.long)
    H_after[int(operation["to_su"])] = 1
    inverse_node = _node_with_mu(1, int(operation["to_su"]), 35.0)
    inverse_kwargs = {
        "H": H_after,
        "nodes": [inverse_node],
        "difference_spectrum": {
            "ppm": [10.0, 35.0],
            "diff": [10.0, -10.0],
            "r2": 0.1,
        },
        "profile_stats": _spectrum_profile_fixture(),
        "hop1_template_options": _spectrum_hop1_fixture(),
        "spectrum_node_ids_by_su": {int(operation["to_su"]): [1]},
        "aromatic_donor_ids_by_su": {},
        "candidate_limit": 1,
        "max_primary_adjustments": 1,
        "E_target": E_target,
    }
    inverse, _ = generate_spectrum_candidates(**inverse_kwargs)
    blocked, audit = generate_spectrum_candidates(
        **inverse_kwargs,
        forbidden_inverse_signatures=[forward[0].inverse_signature],
    )

    assert len(inverse) == 1
    assert blocked == []
    assert audit["reason"] == "no_ranked_spectrum_candidate"


def test_histogram_batch_allows_contiguous_real_conversion_path() -> None:
    nodes = [make_node(1, 22), make_node(2, 23)]
    H = histogram(nodes)
    H_after = H.clone()
    H_after[22] -= 1
    H_after[11] += 1
    plan = HistogramAdjustmentPlan(
        groups=[
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(
                    from_su=22, to_su=23, node_ids=[1]
                ),
                combined_delta={22: -1, 23: 1},
            ),
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(
                    from_su=23, to_su=11, node_ids=[1]
                ),
                combined_delta={23: -1, 11: 1},
            ),
        ],
        H_before=H,
        H_after=H_after,
        inventory_before={22: 1, 23: 1},
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes)

    assert applied.changed is True
    assert applied.rejected_ops == []
    assert int(applied.nodes[0].su_type) == 11


def test_histogram_batch_rejects_non_contiguous_real_conversion_path() -> None:
    nodes = [make_node(1, 22)]
    H = histogram(nodes)
    plan = HistogramAdjustmentPlan(
        groups=[
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(
                    from_su=22, to_su=23, node_ids=[1]
                ),
                combined_delta={22: -1, 23: 1},
            ),
            AtomicAdjustmentGroup(
                primary_op=HistogramConversion(
                    from_su=22, to_su=11, node_ids=[1]
                ),
                combined_delta={22: -1, 11: 1},
            ),
        ],
        H_before=H,
        H_after=H,
        inventory_before={22: 1},
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes)

    assert applied.changed is False
    assert applied.rejected_ops[0]["reason"] == "non_contiguous_real_conversion_path"
    assert int(applied.nodes[0].su_type) == 22


def test_histogram_batch_rejects_primary_node_as_balance_donor() -> None:
    nodes = [make_node(1, 11)]
    H = histogram(nodes)
    plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=11, to_su=12, node_ids=[1]
            ),
            combined_delta={11: -1, 12: 1},
        )],
        port_role_balance_op=HistogramConversion(
            from_su=12, to_su=11, node_ids=[1]
        ),
        H_before=H,
        H_after=H,
        inventory_before={11: 1},
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes)

    assert applied.changed is False
    assert applied.rejected_ops[0]["reason"] == "duplicate_real_conversion_node_id"
    assert applied.rejected_ops[0]["operation"] == "port_role_balance_op"


def test_histogram_via_su_is_one_atomic_real_node_conversion() -> None:
    nodes = [make_node(1, 22)]
    H = histogram(nodes)
    H_after = H.clone()
    H_after[22] -= 1
    H_after[11] += 1
    plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=22,
                via_su=23,
                to_su=11,
                node_ids=[1],
            ),
            combined_delta={22: -1, 11: 1},
        )],
        H_before=H,
        H_after=H_after,
        inventory_before={22: 1},
    )

    applied = apply_histogram_adjustment_plan(H, plan, nodes)

    assert applied.changed is True
    assert int(applied.nodes[0].su_type) == 11


def test_histogram_aromatic_floor_blocks_only_further_donation() -> None:
    at_floor_nodes = [make_node(node_id, 12) for node_id in range(1, 4)]
    at_floor_H = histogram(at_floor_nodes)
    blocked_H = at_floor_H.clone()
    blocked_H[12] -= 1
    blocked_H[13] += 1
    blocked_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=12, to_su=13, node_ids=[1]
            ),
            combined_delta={12: -1, 13: 1},
        )],
        H_before=at_floor_H,
        H_after=blocked_H,
        inventory_before={12: 3},
    )
    blocked = apply_histogram_adjustment_plan(
        at_floor_H, blocked_plan, at_floor_nodes
    )
    assert blocked.changed is False
    assert blocked.rejected_ops[0]["reason"] == "donor_not_in_authorized_stage3_pool"

    above_floor_nodes = [make_node(node_id, 12) for node_id in range(1, 5)]
    above_floor_H = histogram(above_floor_nodes)
    allowed_H = above_floor_H.clone()
    allowed_H[12] -= 1
    allowed_H[13] += 1
    allowed_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=12, to_su=13, node_ids=[1]
            ),
            combined_delta={12: -1, 13: 1},
        )],
        H_before=above_floor_H,
        H_after=allowed_H,
        inventory_before={12: 4},
    )
    allowed = apply_histogram_adjustment_plan(
        above_floor_H, allowed_plan, above_floor_nodes
    )
    assert allowed.changed is True
    assert int(allowed.H[12].item()) == 3

    below_floor_nodes = [make_node(1, 11), make_node(2, 12)]
    below_floor_H = histogram(below_floor_nodes)
    generated_H = below_floor_H.clone()
    generated_H[11] -= 1
    generated_H[13] += 1
    generated_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=11, to_su=13, node_ids=[1]
            ),
            combined_delta={11: -1, 13: 1},
        )],
        H_before=below_floor_H,
        H_after=generated_H,
        inventory_before={11: 1, 12: 1},
    )
    generated = apply_histogram_adjustment_plan(
        below_floor_H, generated_plan, below_floor_nodes
    )
    assert generated.changed is True
    assert int(generated.H[12].item()) == 1
    assert int(generated.H[13].item()) == 1


def _ring_contract(
    nodes,
    *,
    ring_kind,
    cycles,
    like24_node_ids,
    role_node_ids=None,
    node_profiles_by_id=None,
    bridgehead_kind="",
    external_attachments=None,
    branch_paths=None,
):
    by_id = {int(node.global_id): node for node in nodes}
    declared = {int(value) for cycle in cycles for value in cycle}
    for edge in list(external_attachments or []):
        declared.update(int(value) for value in edge)
    for path in list(branch_paths or []):
        declared.add(int(path["anchor_node_id"]))
        declared.update(int(value) for value in path["node_ids"])
    return RingTopologyContract(
        ring_kind=str(ring_kind),
        cycles=[[int(value) for value in cycle] for cycle in cycles],
        node_labels_by_id={node_id: str(by_id[node_id].su_type) for node_id in declared},
        node_su_by_id={node_id: int(by_id[node_id].su_type) for node_id in declared},
        node_profiles_by_id={
            int(key): str(value) for key, value in dict(node_profiles_by_id or {}).items()
        },
        role_node_ids=dict(role_node_ids or {}),
        external_attachments=list(external_attachments or []),
        branch_paths=list(branch_paths or []),
        branch_requirements=[
            {
                "anchor_node_id": int(path.get("anchor_node_id", -1)),
                "node_ids": [int(value) for value in list(path.get("node_ids", []) or [])],
                "kind": path.get("kind", "H"),
                "state": path.get("state", "FILLED"),
            }
            for path in list(branch_paths or [])
        ],
        topology_notation="ring",
        node_id_notation="node-ring",
        like24_node_ids=list(like24_node_ids),
        bridgehead_kind=str(bridgehead_kind),
    )


def test_stage3_phase_sequence_is_one_way() -> None:
    assert PHASE_SEQUENCE == (
        "flex_range_reconciliation",
        "external_flex_closure",
        "external_fatty_ring_closure",
        "external_side_chain_closure",
        "remaining_pool_spectrum_rebalance",
        "residual_resource_self_closure",
        "complete",
    )


def test_side_upper_and_fused_ring_contracts_use_real_cycles() -> None:
    side_nodes = [
        make_node(node_id, su_type)
        for node_id, su_type in zip(range(1, 7), [11, 24, 23, 23, 24, 11])
    ]
    side = _ring_contract(
        side_nodes,
        ring_kind="side",
        cycles=[[1, 2, 3, 4, 5, 6]],
        like24_node_ids=[2, 5],
        node_profiles_by_id={2: "A", 5: "A"},
    )
    assert audit_ring_topology_contract(side, nodes=side_nodes) is None

    upper_nodes = [make_node(10, 11)] + [
        make_node(node_id, su_type)
        for node_id, su_type in zip(range(1, 7), [24, 23, 24, 23, 23, 23])
    ]
    upper = _ring_contract(
        upper_nodes,
        ring_kind="upper_lower",
        cycles=[[1, 2, 3, 4, 5, 6]],
        like24_node_ids=[1, 3],
        node_profiles_by_id={1: "A", 3: "C"},
        role_node_ids={"fixed_anchor": [1]},
        external_attachments=[[10, 1]],
    )
    assert audit_ring_topology_contract(upper, nodes=upper_nodes) is None

    fused_nodes = [
        make_node(
            node_id,
            11 if node_id in {1, 10} else 24 if node_id in {2, 3, 6, 8, 9} else 23,
        )
        for node_id in range(1, 11)
    ]
    fused = _ring_contract(
        fused_nodes,
        ring_kind="fused",
        cycles=[[1, 2, 3, 8, 9, 10], [3, 4, 5, 6, 7, 8]],
        like24_node_ids=[2, 3, 6, 8, 9],
        node_profiles_by_id={2: "A", 3: "C", 6: "C", 8: "C", 9: "A"},
        role_node_ids={"bridgeheads": [3, 8]},
        bridgehead_kind="real_24c",
    )
    assert audit_ring_topology_contract(fused, nodes=fused_nodes) is None


def test_external_ring_serialization_keeps_read_only_graph_audit() -> None:
    nodes = [
        make_node(node_id, su_type)
        for node_id, su_type in zip(range(1, 7), [11, 23, 24, 24, 23, 11])
    ]
    contract = _ring_contract(
        nodes,
        ring_kind="side",
        cycles=[[1, 2, 3, 4, 5, 6]],
        like24_node_ids=[3, 4],
        node_profiles_by_id={3: "C", 4: "D"},
    )
    contract.complete = True
    contract.diagnostics["generation_id"] = 17
    product = TopologyAssemblyProduct(
        product_kind="fatty_ring_side",
        product_type="aliphatic_ring",
        ring_contract=contract,
    )

    payload = product.to_dict()

    assert "ring_contract" not in payload
    assert payload["ring_audit"]["generation_id"] == 17
    assert payload["ring_audit"]["cycles"] == [[1, 2, 3, 4, 5, 6]]
    assert payload["ring_audit"]["complete"] is True

    normalized = normalize_stage3_product(payload, generation_id=17)
    assert normalized.phase == "external_fatty_ring_closure"
    assert normalized.ring_audit == payload["ring_audit"]


def test_known_xlt_five_node_ring_is_rejected() -> None:
    su_by_id = {
        214: 11,
        584: 24,
        572: 24,
        571: 24,
        42: 4,
        463: 23,
        427: 22,
        464: 23,
        502: 23,
    }
    nodes = [make_node(node_id, su_type) for node_id, su_type in su_by_id.items()]
    contract = _ring_contract(
        nodes,
        ring_kind="upper_lower",
        cycles=[[584, 572, 571, 464, 502]],
        like24_node_ids=[584, 572, 571],
        node_profiles_by_id={584: "A", 572: "C", 571: "C"},
        role_node_ids={"fixed_anchor": [584]},
        external_attachments=[[214, 584]],
        branch_paths=[
            {
                "anchor_node_id": 572,
                "node_ids": [463, 427],
                "kind": "H",
                "state": "FILLED",
            },
            {
                "anchor_node_id": 571,
                "node_ids": [42],
                "kind": "FIXED_TERMINAL",
                "state": "FILLED",
            },
        ],
    )
    assert audit_ring_topology_contract(contract, nodes=nodes) == (
        "ring_cycle_node_count:upper_lower:0:5"
    )


def test_large_ring_pool_uses_bounded_logical_resource_candidates() -> None:
    nodes, _ = _raw_su24_pool_fixture(["C"] * 36)
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )

    side_23 = ring_segments_module.build_pool_ring_template_segments(
        assembler,
        skeleton_kind="side",
        role="middle",
        allowed_su=[23],
    )
    upper_23 = ring_segments_module.build_pool_ring_template_segments(
        assembler,
        skeleton_kind="upper_lower",
        role="path",
        allowed_su=[23],
    )
    fused_23 = ring_segments_module.build_pool_ring_template_segments(
        assembler,
        skeleton_kind="fused",
        role="inner_core",
        allowed_su=[23],
    )
    fused_24 = ring_segments_module.build_pool_ring_template_segments(
        assembler,
        skeleton_kind="fused",
        role="inner_core",
        allowed_su=[24],
    )

    assert len(side_23) <= 4
    assert len(upper_23) <= 5
    assert len(fused_23) <= 1
    assert sum(item["kind"] == "pool" for item in fused_24) <= 8
    assert sum(item["kind"] == "pool_group" for item in fused_24) <= 24


def test_upper_ring_pool_candidates_exclude_nonphysical_su_types() -> None:
    nodes, _ = _raw_su24_pool_fixture(["C"] * 8)
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )

    candidates = ring_segments_module.build_pool_ring_template_segments(
        assembler,
        skeleton_kind="upper_lower",
        role="path",
        allowed_su=[3, 23, 24],
    )

    assert candidates
    assert {int(item["pool_su"]) for item in candidates} <= {23, 24}
    assert all(
        int(key) in {11, 22, 23, 24}
        for item in candidates
        for key in dict(item.get("support_required_counts", {}) or {})
    )


def test_large_raw_su24_external_search_terminates_with_real_audited_rings() -> None:
    nodes, _ = _raw_su24_pool_fixture(["A", "B", "C", "D"] * 9)
    result = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()
    rings = [
        product
        for product in result["products"]
        if str(product["product_kind"]).startswith("fatty_ring_")
    ]

    assert rings
    assert result["diagnostics"]["invalid_topology_count"] == 0
    assert all(product["ring_audit"]["complete"] for product in rings)
    assert all(
        len(cycle) == 6
        for product in rings
        for cycle in product["ring_audit"]["cycles"]
    )


def test_stage3_history_serialization_excludes_live_runtime_and_graphs() -> None:
    plan = Stage3AdjustmentPlan(
        ok=True,
        phase="external_fatty_ring_closure",
        reason="external_conversion_continuation",
        runtime={
            "phase_evaluation": {"large_live_graph": [object()]},
            "assembly_before_adjustment": {"products": [{"node_ids": [1, 2]}]},
        },
    )

    assert "runtime" not in plan.to_dict()
    compact = _compact_stage3_plan_dict(plan)
    assert "runtime" not in compact
    assert "phase_evaluation" not in repr(compact)

    workflow = Stage3WorkflowResult(
        nodes=[make_node(1, 24)],
        H=torch.ones(33, dtype=torch.long),
        diff={"spectrum": list(range(1000))},
        ok=True,
        diagnostics={
            "retained_cycle_limit": 8,
            "completion_snapshot": {"large_graph_copy": [object()]},
            "terminal_snapshot": {"large_plan_copy": [object()]},
        },
    )
    payload = workflow.to_dict()
    assert "nodes" not in payload
    assert "H" not in payload
    assert "diff" not in payload
    assert "completion_snapshot" not in payload["diagnostics"]
    assert "terminal_snapshot" not in payload["diagnostics"]


def test_completion_snapshot_keeps_live_products_without_phase_field() -> None:
    product = TopologyAssemblyProduct(
        product_kind="fatty_ring_side",
        product_type="aliphatic_ring",
        actual_topology_su="11-24-23-24-11",
        actual_topology_gid="1-2-3-4-5",
        fixed_source_node_ids=[1, 5],
        claimed_resource_ids_by_su={"23": [3], "24": [2, 4]},
        expected_su_type_by_node_id={1: 11, 2: 24, 3: 23, 4: 24, 5: 11},
        closed=True,
    )
    assembler = SimpleNamespace(
        products=[product],
        pending_units=[],
        pending_contracts=[],
        allocator=SimpleNamespace(remaining_ids_by_su={}),
    )
    plan = SimpleNamespace(
        phase="complete",
        active_subphase="",
        diagnostics={},
        runtime={"external_assembler": assembler},
        pending_contracts=[],
    )

    snapshot = block_c_module._stage3_runtime_snapshot(plan)

    assert len(snapshot["products"]) == 1
    assert snapshot["products"][0]["actual_topology_su"] == "11-24-23-24-11"
    assert snapshot["products"][0]["claimed_resource_ids_by_su"]["24"] == [2, 4]


def test_stage3_cycle_history_is_bounded_to_recent_summaries() -> None:
    cycles = []
    for cycle_number in range(1, 13):
        _append_stage3_cycle(
            cycles,
            Stage3WorkflowCycleRecord(cycle=cycle_number),
        )

    assert [record.cycle for record in cycles] == list(range(5, 13))


def test_stage3_candidate_attempt_diagnostics_keep_first_and_last_samples() -> None:
    attempts = [{"index": index} for index in range(20)]

    samples = state_machine_module._diagnostic_attempt_samples(attempts)

    assert [item["index"] for item in samples] == [
        0, 1, 2, 3, 16, 17, 18, 19,
    ]


def test_assembly_failure_history_keeps_count_latest_and_last_per_fragment() -> None:
    assembler = ExternalTopologyAssembler.__new__(ExternalTopologyAssembler)
    assembler.composition_attempt_failure_count = 0
    assembler.last_composition_attempt_failure = {}
    assembler.materialization_failure_by_fragment_id = {}

    for attempt_index in range(12):
        ownership_module.record_composition_failure_entry(assembler, {
            "source_fragment_ids": [attempt_index % 3],
            "attempt_index": attempt_index,
        })

    assert assembler.composition_attempt_failure_count == 12
    assert assembler.last_composition_attempt_failure["attempt_index"] == 11
    assert {
        fragment_id: failure["attempt_index"]
        for fragment_id, failure in (
            assembler.materialization_failure_by_fragment_id.items()
        )
    } == {0: 9, 1: 10, 2: 11}
    assert not hasattr(assembler, "composition_attempt_failures")


def test_stage3_cycle_serialization_contains_only_compact_fields() -> None:
    record = Stage3WorkflowCycleRecord(cycle=1, phase="complete")

    assert set(record.to_dict()) == {
        "cycle",
        "attempt",
        "phase",
        "applied",
        "stop_reason",
        "accepted_state_retained",
        "rebuild_valid",
        "layer2_r2",
        "blocked_reason",
        "accepted",
        "acceptance_reason",
        "primary_adjustment_count",
        "physical_conversion_count",
        "h_compensation_count",
        "parity_balance_count",
        "fresh_rebuild",
    }


def test_stage3_node_runtime_ownership_is_cleared_for_fresh_graph() -> None:
    node = make_node(1, 24)
    node.stage3_topology_owner_id = "old:ring"
    node.stage3_topology_role = "bridgehead"
    node.stage3_locked_neighbor_ids = {2, 3}
    node.stage3_resource_claimed = True

    _clear_stage3_node_runtime_state([node])

    assert node.stage3_topology_owner_id is None
    assert node.stage3_topology_role is None
    assert node.stage3_locked_neighbor_ids == set()
    assert node.stage3_resource_claimed is False


def test_conversion_scoring_rejects_su24_ring_resource_as_donor() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23] = 2
    H[24] = 2
    context = ConversionContext(
        inventory={11: 0, 22: 0, 23: 2, 24: 2},
        demand={11: 1},
        current_H=H,
        phase="external_fatty_ring_closure",
    )
    donor_23 = score_conversion_bundle(
        context,
        [HistogramConversion(23, 11, 1, "test")],
        h_total_fn=lambda hist, _meta: float(hist.sum().item()),
    )
    donor_24 = score_conversion_bundle(
        context,
        [HistogramConversion(24, 11, 1, "test")],
        h_total_fn=lambda hist, _meta: float(hist.sum().item()),
    )

    assert donor_23.feasible is True
    assert donor_24.feasible is False
    assert donor_24.diagnostics["reason"] == (
        "unauthorized_external_adjustment_su_type"
    )
    assert donor_24.diagnostics["unauthorized_su_types"] == [24]


@pytest.mark.parametrize("source,target", [(23, 24), (24, 23)])
def test_accounting_preflight_rejects_su24_parity_balance(
    source: int,
    target: int,
) -> None:
    plan = HistogramAdjustmentPlan(
        parity_balance_op=HistogramConversion(
            from_su=source,
            to_su=target,
            count=1,
        )
    )

    ok, proof = preflight_histogram_accounting(plan)

    assert ok is False
    assert proof["errors"][0]["reason"] == (
        "invalid_parity_balance_contract"
    )


def test_external_ring_retries_next_candidate_after_atomic_rollback(
    monkeypatch,
) -> None:
    view = AssemblyFragmentView(
        unit_id=428,
        fragment_id=428,
        family="like24_bridge",
        class_type="24_CD_N_24_CD",
        topology_template="-24(?)-23-24(?)-",
        source_node_ids=[428, 429, 503],
        main_labels=["24", "23", "24"],
        main_node_ids=[428, 503, 429],
        canonical_labels=["24", "23", "24"],
        open_left=True,
        open_right=True,
        can_ring=True,
        target_roles=["fatty_ring"],
    )

    def make_plan(candidate_id: str):
        return RingCandidatePlan(
            ring_kind="fatty_ring_side",
            skeleton_kind="side_ring_plan",
            source_views=[view],
            ordered_segments=[{
                "candidate_id": candidate_id,
                "fragment_id": 428,
                "view": view,
            }],
            template_complete=True,
        )

    first_plan = make_plan("candidate-a")
    second_plan = make_plan("candidate-b")
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.allocator = PoolAllocator({"23": [201, 202]})
    assembler.view_by_id = {428: view}
    assembler.last_ring_candidate_required_counts = {}
    assembler.last_ring_missing_counts = {}
    assembler.last_ring_materialization_reason = ""
    attempts = []

    monkeypatch.setattr(
        ring_planner_module,
        "plan_side_ring_templates",
        lambda _, _views: first_plan,
    )
    monkeypatch.setattr(
        ring_planner_module,
        "enumerate_ring_plans",
        lambda _, _builder, **_kwargs: [first_plan, second_plan],
    )
    monkeypatch.setattr(
        ring_planner_module,
        "pool_ring_attempt",
        lambda *_args, **_kwargs: {
            "required_counts": {"23": 1},
            "missing_counts": {},
        },
    )

    def commit(candidate, _branch_donors, **_kwargs):
        snapshot = assembler.allocator.snapshot()
        claimed = assembler.allocator.claim(23, 1)
        candidate_id = candidate.ordered_segments[0]["candidate_id"]
        attempts.append((candidate_id, claimed))
        if candidate_id == "candidate-a":
            assembler.last_ring_materialization_reason = "synthetic_audit_failure"
            assembler.allocator.rollback(snapshot)
            return False
        return True

    monkeypatch.setattr(
        ring_materializer_module,
        "commit_single_cycle_ring_candidate",
        lambda _, candidate, donors, **kwargs: commit(
            candidate, donors, **kwargs
        ),
    )

    assert ring_materializer_module.assemble_ring(
        assembler,
        product_kind="fatty_ring_side",
        preferred_views=[view],
        branch_donors=[],
    ) is True
    assert attempts == [
        ("candidate-a", [201]),
        ("candidate-b", [201]),
    ]
    assert assembler.allocator.remaining_ids_by_su["23"] == [202]
    assert assembler.allocator.consumed_ids_by_su["23"] == [201]


def test_allocator_reclassifies_recycled_su24_atomically() -> None:
    allocator = PoolAllocator({"24": [1], "23": [2]})

    moved = allocator.reclassify_ids([1], from_su=24, to_su=23)

    assert moved == [1]
    assert allocator.remaining_ids_by_su["24"] == []
    assert allocator.remaining_ids_by_su["23"] == [1, 2]
    assert allocator.su_key_for_node_id(1) == "23"


def test_allocator_reclassify_rollback_restores_pool_state() -> None:
    allocator = PoolAllocator({"24": [1], "23": [2]})
    allocator.protected_counts_by_su["24"] = 1
    allocator._generic_claim_excluded_node_ids.add(1)
    snapshot = allocator.snapshot()

    assert allocator.reclassify_ids([1], from_su=24, to_su=23) == [1]
    allocator.rollback(snapshot)

    assert allocator.remaining_ids_by_su == {"24": [1], "23": [2]}
    assert allocator.su_key_for_node_id(1) == "24"
    assert allocator.protected_counts_by_su == {"24": 1}
    assert allocator._generic_claim_excluded_node_ids == {1}


def test_allocator_reports_raw_claimable_specific_and_rollback_trace() -> None:
    allocator = PoolAllocator({"23": [1, 2, 3]})
    allocator.protected_counts_by_su["23"] = 1
    allocator._generic_claim_excluded_node_ids.add(2)

    snapshot = allocator.claimability_snapshot()
    assert snapshot["raw_available_counts"] == {"23": 3}
    assert snapshot["claimable_available_counts"] == {"23": 1}
    assert snapshot["protected_counts"] == {"23": 1}
    assert snapshot["excluded_node_ids"] == [2]

    request = allocator.claimability_for_request(
        23,
        count=2,
        specific_node_ids=[2, 3, 99],
    )
    assert request["generic_claim_satisfied"] is False
    assert request["specific_claim_satisfied_by_node_id"] == {
        "2": True,
        "3": True,
        "99": False,
    }

    transaction = allocator.snapshot()
    checkpoint = allocator.trace_checkpoint()
    with allocator.claim_trace_scope("right_closure", "right:F1:23"):
        assert allocator.claim(23, 2) is None
        assert allocator.claim_specific(23, 2) == [2]
        allocator.rollback(transaction)

    trace = allocator.claim_trace_since(checkpoint)
    assert [row["mode"] for row in trace] == ["generic", "specific", "rollback"]
    assert trace[0]["success"] is False
    assert trace[0]["claimable_available_count"] == 1
    assert trace[1]["selected_node_ids"] == [2]
    assert trace[2]["rolled_back_ids_by_su"] == {"23": [2]}
    assert allocator.remaining_ids_by_su == {"23": [1, 2, 3]}


def test_round_context_normalizes_legacy_released_su24_to_su23() -> None:
    nodes = [make_node(1, 24)]
    classification = TopologyClassificationResult(
        resource_pool={
            "remaining_ids_by_su": {"11": [], "22": [], "23": [], "24": [1]},
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        }
    )

    context = build_stage3_round_context(
        nodes=nodes,
        classification=classification,
        released_resource_ids_by_su={"24": [1]},
    )

    assert context.su24_ring_resource_ids == []
    assert context.generic_closure_pool_ids_by_su[23] == [1]
    assert context.recycled_su24_to_23_ids == [1]
    assert context.ownership_by_node_id[1].su_type == 23


def test_recycling_pure_resource_ring_converts_su24_nodes_to_su23() -> None:
    nodes, _pool = _raw_su24_pool_fixture(["C", "C"])
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembler.assemble()
    ring = assembler.products[0]
    su24_ids = sorted(
        int(node_id)
        for node_id, su_type in ring.ring_contract.node_su_by_id.items()
        if int(su_type) == 24
    )

    released = ownership_module.recycle_resource_ring(assembler, ring)

    assert set(su24_ids).issubset(set(released))
    assert all(
        int(assembler.nodes_by_id[node_id].su_type) == 23
        for node_id in su24_ids
    )
    assert set(su24_ids).issubset(
        set(assembler.allocator.remaining_ids_by_su["23"])
    )
    assert not set(su24_ids) & set(assembler.allocator.remaining_ids_by_su["24"])
    assert all(
        item["from_su"] == 24 and item["to_su"] == 23
        for item in assembler.recycled_resource_conversion_ops
    )


def test_ring_recovery_recycles_only_one_ring_per_authorization(
    monkeypatch,
) -> None:
    rings = [
        SimpleNamespace(
            product_kind="fatty_ring_side",
            ring_contract=SimpleNamespace(
                contract_id="ring-1", node_su_by_id={1: 24}
            ),
        ),
        SimpleNamespace(
            product_kind="fatty_ring_side",
            ring_contract=SimpleNamespace(
                contract_id="ring-2", node_su_by_id={2: 24}
            ),
        ),
    ]
    assembler = SimpleNamespace(
        products=list(rings),
        allocator=SimpleNamespace(remaining_ids_by_su={"23": []}),
        last_recycled_ring={},
        ring_recovery_attempts=[],
        allow_ring_recovery=True,
        ring_recovery_max_su24_cost=1,
    )
    recycled_contracts = []

    monkeypatch.setattr(
        chain_materializer_module,
        "_recyclable_ring_candidates",
        lambda current: list(current.products),
    )

    def recycle(current, ring):
        contract_id = str(ring.ring_contract.contract_id)
        recycled_contracts.append(contract_id)
        current.products.remove(ring)
        released_id = 100 + len(recycled_contracts)
        current.allocator.remaining_ids_by_su["23"].append(released_id)
        current.last_recycled_ring = {"contract_id": contract_id}
        return [released_id]

    monkeypatch.setattr(ownership_module, "recycle_resource_ring", recycle)

    assert chain_materializer_module.recover_one_pure_resource_ring(assembler) is True

    assert recycled_contracts == ["ring-1"]
    assert assembler.allocator.remaining_ids_by_su["23"] == [101]
    assert len(assembler.ring_recovery_attempts) == 1
    assert assembler.allow_ring_recovery is False
    assert assembler.ring_recovery_max_su24_cost is None


def _ring_recovery_gate_plan(
    *,
    role: str = "side_chain",
    failure_domain: str = "resource_capacity",
    ordinary_conversion_feasible: bool = False,
    missing_counts: dict | None = None,
) -> SimpleNamespace:
    missing = dict(missing_counts or {"11": 1})
    required = {str(key): int(value) for key, value in missing.items()}
    proof = {
        "reason": "external_resource_deficit",
        "failure_domain": str(failure_domain),
        "required_counts": required,
        "available_counts": {},
        "missing_counts": missing,
        "conversion_capacity_proof": {
            "search_exhausted": True,
            "ordinary_conversion_feasible": bool(ordinary_conversion_feasible),
            "missing_counts": missing,
            "remaining_primary_budget": 4,
            "allowed_donor_su_types": [11, 12, 13, 22, 23],
            "failure_domain": str(failure_domain),
        },
    }
    return SimpleNamespace(
        phase="external_topology_assembly",
        reason="phase_resource_infeasible",
        pending_contracts=[{
            "target_role": str(role),
            "required_counts": required,
            "proof": proof,
        }],
        phase_result=SimpleNamespace(infeasibility_proof={}),
        diagnostics={},
    )


def test_recovery_gate_rejects_feasible_ordinary_conversion() -> None:
    plan = _ring_recovery_gate_plan(ordinary_conversion_feasible=True)

    assert block_c_module._external_ring_recovery_authorized(plan) is False


@pytest.mark.parametrize(
    "failure_domain",
    ["h_constraint", "conversion_scoring", "protected_resource", "future_topology_reservation"],
)
def test_recovery_gate_rejects_non_capacity_failures(failure_domain: str) -> None:
    plan = _ring_recovery_gate_plan(failure_domain=failure_domain)

    assert block_c_module._external_ring_recovery_authorized(plan) is False


@pytest.mark.parametrize("role", ["fatty_ring", "side_chain"])
def test_recovery_gate_accepts_exhausted_conversion_proof(role: str) -> None:
    plan = _ring_recovery_gate_plan(role=role)

    assert block_c_module._external_ring_recovery_authorized(plan) is True
    capacity = plan.diagnostics["ring_recovery_capacity_proof"]
    assert capacity["recovery_allowed"] is True
    assert capacity["conversion_capacity_proof"]["search_exhausted"] is True


def test_recovery_gate_rejects_true_su24_deficit() -> None:
    plan = _ring_recovery_gate_plan(missing_counts={"24": 1})

    assert block_c_module._external_ring_recovery_authorized(plan) is False
    capacity = plan.diagnostics["ring_recovery_capacity_proof"]
    assert capacity["blocked_reason"] == "su24_deficit_cannot_be_filled_by_recovery"


def test_recovery_gate_requires_explicit_capacity_proof() -> None:
    plan = SimpleNamespace(
        phase="external_side_chain_closure",
        reason="phase_resource_infeasible",
        pending_contracts=[{
            "target_role": "side_chain",
            "required_counts": {"11": 3, "23": 3},
            "proof": {
                "reason": "external_resource_deficit",
                "failure_domain": "resource_capacity",
                "available_counts": {"11": 0, "22": 2, "23": 0},
            },
        }],
        phase_result=SimpleNamespace(infeasibility_proof={}),
        diagnostics={},
    )

    assert block_c_module._external_ring_recovery_authorized(plan) is False


def test_selected_ring_atomically_claims_its_concrete_su24_support_nodes() -> None:
    nodes, pool = _raw_su24_pool_fixture(["C", "C", "D", "D"])
    by_id = {int(node.global_id): node for node in nodes}
    terminal_22_ids = sorted({
        int(neighbor_id)
        for node in nodes
        if int(node.su_type) == 24
        for neighbor_id in list(node.hop1_ids or [])
        if int(by_id[int(neighbor_id)].su_type) == 22
    })
    generic_22_ids = [
        int(node_id) for node_id in pool[22]
        if int(node_id) not in set(terminal_22_ids)
    ]
    pool[22] = list(terminal_22_ids) + list(generic_22_ids[:2])

    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembler.allocator = PoolAllocator({str(key): list(values) for key, values in pool.items()})
    result = assembler.assemble().to_dict()

    assert result["products"]
    claimed_22 = {
        int(node_id)
        for product in result["products"]
        for node_id in product["ring_audit"]["claimed_resource_ids_by_su"].get(
            "22", []
        )
    }
    # Bound SU22 terminals are fixed sources of the selected SU24 ring, not
    # generic allocator claims.
    assert not (set(terminal_22_ids) & claimed_22)
    fixed_terminal_ids = {
        int(node_id)
        for product in result["products"]
        for path in product["ring_audit"]["branch_paths"]
        if path.get("kind") == "FIXED22"
        for node_id in path.get("node_ids", [])
    }
    assert set(terminal_22_ids) <= fixed_terminal_ids
    owned_22 = claimed_22 | {
        int(node_id)
        for product in result["products"]
        for node_id in product.get("fixed_source_node_ids", [])
        if int(by_id[int(node_id)].su_type) == 22
    }
    assert set(terminal_22_ids) <= owned_22


def test_ring_planner_keeps_raw_su24_with_classified_fixed_support() -> None:
    nodes, pool = _raw_su24_pool_fixture(["A", "A", "C", "C"])
    by_id = {int(node.global_id): node for node in nodes}
    cap_ids = {
        int(neighbor_id)
        for node in nodes
        if int(node.su_type) == 24
        for neighbor_id in list(node.hop1_ids or [])
        if int(by_id[int(neighbor_id)].su_type) == 11
    }
    pool[11] = [
        int(node_id) for node_id in pool[11]
        if int(node_id) not in cap_ids
    ]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembler.allocator = PoolAllocator({
        str(key): list(values) for key, values in pool.items()
    })
    left, right, middle = ring_planner_module.build_side_ring_template_candidates(
        assembler, []
    )

    assert any(
        str(candidate.get("raw_su24_role", "")) == "A"
        for candidate in [*left, *right, *middle]
    )
    assert any(
        str(candidate.get("candidate_id", ""))
        == "pool:side:left_anchor:endpoint:11-23"
        for candidate in left
    )
    assert any(
        str(candidate.get("candidate_id", ""))
        == "pool:side:right_anchor:endpoint:23-11"
        for candidate in right
    )


def test_external_ring_does_not_reuse_completed_product_fixed_support() -> None:
    nodes, pool = _raw_su24_pool_fixture(["A", "A", "C", "C"])
    by_id = {int(node.global_id): node for node in nodes}
    cap_ids = sorted({
        int(neighbor_id)
        for node in nodes
        if int(node.su_type) == 24
        for neighbor_id in list(node.hop1_ids or [])
        if int(by_id[int(neighbor_id)].su_type) == 11
    })
    assert cap_ids
    occupied_support = int(cap_ids[0])

    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
        occupied_node_ids=[occupied_support],
    )
    assembler.allocator = PoolAllocator({str(key): list(values) for key, values in pool.items()})
    result = assembler.assemble().to_dict()

    assert result["products"]
    owned_ids = set()
    for product in result["products"]:
        audit = dict(product.get("ring_audit", {}) or {})
        owned_ids.update(
            int(value)
            for cycle in list(audit.get("cycles", []) or [])
            for value in list(cycle or [])
        )
        owned_ids.update(
            int(value)
            for value in list(product.get("fixed_source_node_ids", []) or [])
        )
        owned_ids.update(
            int(value)
            for path in list(audit.get("branch_paths", []) or [])
            for value in list(dict(path or {}).get("node_ids", []) or [])
        )
    assert occupied_support not in owned_ids


def test_pool_ring_attempt_does_not_protect_unused_su24_open_neighbors() -> None:
    nodes, pool = _raw_su24_pool_fixture(["C", "C"])
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembler.allocator = PoolAllocator({
        str(key): list(values) for key, values in pool.items()
    })
    plan = ring_planner_module.plan_side_ring_templates(assembler, [])

    attempt = ring_planner_module.pool_ring_attempt(
        assembler,
        plan,
        product_kind="fatty_ring_side",
        skeleton_kind="side",
    )

    protected = {
        int(key): list(values)
        for key, values in dict(
            attempt.get("protected_source_ids_by_su", {}) or {}
        ).items()
    }
    assert not ({0, 2, 3} & set(protected))


def test_external_allocator_excludes_raw_su24_support_from_generic_claims() -> None:
    nodes, pool = _raw_su24_pool_fixture(["D", "D"])
    by_id = {int(node.global_id): node for node in nodes}
    terminal_22_ids = sorted({
        int(neighbor_id)
        for node in nodes
        if int(node.su_type) == 24
        for neighbor_id in list(node.hop1_ids or [])
        if int(by_id[int(neighbor_id)].su_type) == 22
    })
    generic_22_ids = [
        int(node_id) for node_id in pool[22]
        if int(node_id) not in set(terminal_22_ids)
    ]
    classification = classify_topology_units(nodes)
    classification.resource_pool["remaining_ids_by_su"]["22"] = (
        list(terminal_22_ids) + list(generic_22_ids)
    )

    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    claimed = assembler.allocator.claim(22, 2)

    assert set(claimed).isdisjoint(set(terminal_22_ids))


def test_specific_claim_uses_the_same_path_for_every_transaction() -> None:
    allocator = PoolAllocator({"22": [101, 102]})

    assert allocator.claim_specific(22, 101) == [101]
    assert allocator.claim_specific(22, 102) == [102]


def test_candidate_specific_claim_scope_does_not_become_global_reservation() -> None:
    allocator = PoolAllocator({"22": [101, 102, 103]})

    with allocator.candidate_specific_claim_scope([101]):
        assert allocator.claim(22, 1) == [102]
        assert allocator.claim_specific(22, 101) == [101]

    assert allocator.claim(22, 1) == [103]


def test_ordinary_external_view_cannot_claim_su24_fixed_support_id() -> None:
    nodes, pool = _raw_su24_pool_fixture(["D", "D"])
    by_id = {int(node.global_id): node for node in nodes}
    support_22_ids = sorted({
        int(neighbor_id)
        for node in nodes
        if int(node.su_type) == 24
        for neighbor_id in list(node.hop1_ids or [])
        if int(by_id[int(neighbor_id)].su_type) == 22
    })
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    view = AssemblyFragmentView(
        unit_id=999,
        fragment_id=999,
        family="test",
        class_type="test",
        topology_template="22",
        source_node_ids=[],
        main_labels=["22"],
        main_node_ids=[support_22_ids[0]],
        canonical_labels=["22"],
        open_left=False,
        open_right=False,
    )

    rebound, shortage = chain_materializer_module.claim_view_context_resources(assembler, view)

    assert shortage == {"22": 1}
    assert rebound is None
    assert support_22_ids[0] not in assembler.allocator.consumed_ids_by_su["22"]


def test_round_context_models_su24_as_ring_resource_not_generic_closure() -> None:
    nodes = [make_node(1, 11), make_node(2, 24)]
    classification = TopologyClassificationResult(
        resource_pool={
            "remaining_ids_by_su": {
                "11": [1],
                "22": [],
                "23": [],
                "24": [2],
            },
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        }
    )

    context = build_stage3_round_context(
        nodes=nodes,
        classification=classification,
    )
    public = ownership_public_view(context)

    assert context.generic_closure_pool_ids_by_su == {
        11: [1],
        22: [],
        23: [],
    }
    assert context.su24_ring_resource_ids == [2]
    assert context.ownership_by_node_id[2].state == (
        Stage3OwnershipState.REMAINING_SU24_RING_RESOURCE.value
    )
    assert public["generic_closure_pool_ids_by_su"] == {
        "11": [1],
        "22": [],
        "23": [],
    }
    assert public["su24_ring_resource_ids"] == [2]
    assert "closure_pool_ids_by_su" not in public


def test_topology_ledger_never_marks_su24_as_convertible() -> None:
    nodes = [make_node(1, 11), make_node(2, 24)]
    classification = TopologyClassificationResult(
        resource_pool={
            "remaining_ids_by_su": {
                "11": [1],
                "22": [],
                "23": [],
                "24": [2],
            },
            "initial_ids_by_su": {
                "11": [1],
                "22": [],
                "23": [],
                "24": [2],
            },
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        }
    )
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )

    ledger = ownership_module.reconcile_resource_ownership(
        assembler,
        assembler.allocator.build_ledger({})
    )

    assert ledger.remaining_ids_by_su["24"] == [2]
    assert ledger.convertible_ids_by_su == {
        "11": [1],
        "12": [],
        "13": [],
        "22": [],
        "23": [],
    }
    assert "24" not in ledger.convertible_counts


def test_recycled_su24_is_exposed_only_as_ordinary_su23() -> None:
    phase_eval = SimpleNamespace(
        assembly_view={
            "resource_ledger": {
                "released_su24_ids": [2],
                "released_resource_ids_by_su": {"24": [2]},
            }
        }
    )
    runtime = _runtime_with_pool(
        phase_eval,
        {11: [], 12: [], 13: [], 22: [], 23: [], 24: [1, 2]},
    )
    ledger = runtime["assembly_before_adjustment"]["resource_ledger"]

    assert "24" not in ledger["convertible_ids_by_su"]
    assert "24" not in ledger["convertible_counts"]
    assert ledger["remaining_ids_by_su"]["24"] == [1]
    assert ledger["remaining_ids_by_su"]["23"] == [2]


def test_stale_convertible_su24_field_does_not_authorize_ring_conversion() -> None:
    phase_eval = SimpleNamespace(
        assembly_view={
            "resource_ledger": {
                "convertible_ids_by_su": {"24": [1]},
            }
        }
    )
    runtime = _runtime_with_pool(
        phase_eval,
        {11: [], 12: [], 13: [], 22: [], 23: [], 24: [1]},
    )
    ledger = runtime["assembly_before_adjustment"]["resource_ledger"]

    assert "24" not in ledger["convertible_ids_by_su"]
    assert "24" not in ledger["convertible_counts"]


def test_recycled_su24_is_not_planned_as_a_24_donor() -> None:
    phase_eval = SimpleNamespace(
        assembly_view={
            "resource_ledger": {
                "released_su24_ids": [2],
                "released_resource_ids_by_su": {"24": [2]},
            }
        }
    )
    runtime = _runtime_with_pool(
        phase_eval,
        {11: [], 12: [], 13: [], 22: [], 23: [], 24: [2]},
    )
    plan = Stage3AdjustmentPlan(
        phase="residual_resource_self_closure",
        hard_required_counts={"11": 1},
        minimum_resource_request={"11": 1},
        runtime=runtime,
    )
    H = torch.zeros(40, dtype=torch.long)
    H[24] = 1
    histogram_plan = build_histogram_adjustment_plan(
        plan=plan,
        H=H,
        max_primary_adjustments=1,
        max_h_compensation_ops=0,
    )

    assert _phase_inventory(plan, H).get(24, 0) == 0
    assert len(histogram_plan.groups) == 0


def test_internal_conversion_rejects_su24_even_when_released() -> None:
    released = HistogramAdjustmentPlan(
        accounting_diagnostics={"released_su24_ids": [2]},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=24,
                to_su=11,
                node_ids=[2],
            )
        )],
    )
    ordinary = HistogramAdjustmentPlan(
        accounting_diagnostics={"released_su24_ids": [2]},
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=24,
                to_su=11,
                node_ids=[1],
            )
        )],
    )

    assert _internal_conversion_is_allowed(released) is False
    assert _internal_conversion_is_allowed(ordinary) is False


def test_iteration_ledger_keeps_all_unclaimed_resources_in_generic_pool() -> None:
    nodes = [make_node(1, 22), make_node(2, 22), make_node(3, 23)]
    phase_eval = SimpleNamespace(
        classification_view={"support_units_only": []},
        assembly_view={"diagnostics": {}},
        round_context=None,
    )

    ledger = _iteration_ledger_for_phase_eval(
        phase_eval,
        nodes,
        {22: [1, 2], 23: [3]},
        [],
    )

    assert ledger.available_ids_by_su[22] == [1, 2]
    assert ledger.available_ids_by_su[23] == [3]
    assert 1 not in ledger.support_context_node_ids


def test_physical_accounting_uses_only_the_actual_remaining_pool() -> None:
    remaining = _accounting_remaining_pool(
        {11: [4], 22: [5]},
    )

    assert remaining == {11: [4], 22: [5]}


def test_explicit_ring_resource_restore_does_not_rediscover_su24() -> None:
    nodes = [make_node(1, 24), make_node(2, 24), make_node(3, 22)]
    phase_eval = SimpleNamespace(
        classification_view={"support_units_only": []},
        assembly_view={"diagnostics": {}},
        round_context=None,
    )
    ledger = _iteration_ledger_for_phase_eval(
        phase_eval,
        nodes,
        {24: [1, 2], 22: [3]},
        [],
    )
    ledger = seed_ledger_topology_state(
        ledger,
        completed_products=[{
            "product_id": "external:1",
            "fixed_source_node_ids": [2],
            "claimed_resource_ids_by_su": {},
            "closed": True,
        }],
    )

    released, audit = _restore_uncommitted_ring_pool_nodes(
        ledger,
        nodes,
        restore_resource_ids_by_su={22: [3]},
    )

    assert audit["ok"] is True
    assert released.available_ids_by_su[24] == [1]
    assert released.available_ids_by_su[22] == [3]
    assert 2 not in released.available_ids_by_su[24]
    assert _uncommitted_physical_su24_ids(released, nodes) == [1]


def test_ring_pool_uses_fresh_round_ownership_over_stale_assembly_ledger() -> None:
    phase_eval = SimpleNamespace(
        round_context=SimpleNamespace(
            generic_closure_pool_ids_by_su={11: [1], 23: [3]},
            su24_ring_resource_ids=[4, 5],
            aromatic_donor_ids_by_su={12: [6]},
        ),
        assembly_view={
            "resource_ledger": {
                "remaining_ids_by_su": {
                    "11": [1],
                    "23": [2, 3],
                    "24": [4, 5],
                }
            }
        },
        classification_view={},
    )

    pool = _round_resource_ids(phase_eval)

    assert pool == {11: [1], 12: [6], 23: [3], 24: [4, 5]}
    assert 2 not in pool[23]


def test_stage3_production_path_materializes_pool_su24_ring(monkeypatch) -> None:
    nodes, _pool = _raw_su24_pool_fixture(["C", "C"])
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    assembly = assembler.assemble().to_dict()
    ring_products = [
        product
        for product in assembly["products"]
        if product.get("product_kind", "").startswith("fatty_ring_")
    ]
    assert ring_products
    assert all(product.get("closed") is True for product in ring_products)
    assert all(product.get("ring_audit", {}).get("complete") for product in ring_products)
    assert "real_like24_ring_assembly" in set(
        assembly.get("diagnostics", {}).get("external_subphases", [])
    )


def test_optional_ring_view_does_not_shadow_pool_only_ring_fallback() -> None:
    nodes, _pool = _raw_su24_pool_fixture(["C", "C"])
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    optional_view = AssemblyFragmentView(
        unit_id=9999,
        fragment_id=9999,
        family="test_optional_ring",
        class_type="test_optional_ring",
        topology_template="-23-",
        source_node_ids=[],
        main_labels=["23"],
        main_node_ids=[None],
        canonical_labels=["23"],
        open_left=True,
        open_right=True,
        can_side=True,
        can_ring=True,
        target_roles=["fatty_ring", "side_chain"],
    )
    assembler.views.append(optional_view)
    assembler.view_by_id[int(optional_view.fragment_id)] = optional_view
    assembler.fragment_state_by_id[int(optional_view.fragment_id)] = "available"

    created = ring_planner_module.assemble_priority_rings(assembler, [])

    rings = [
        product
        for product in assembler.products
        if str(product.product_kind).startswith("fatty_ring_")
    ]
    assert created is True
    assert rings
    assert int(optional_view.fragment_id) in assembler.ring_exhausted_fragment_ids
    assert any(
        not list(product.source_fragment_ids or [])
        for product in rings
    )


def test_ring_candidate_does_not_block_pool_only_ring_fallback() -> None:
    nodes, _pool = _raw_su24_pool_fixture(["C", "C"])
    assembler = ExternalTopologyAssembler(
        nodes,
        classify_topology_units(nodes),
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    required_view = AssemblyFragmentView(
        unit_id=9998,
        fragment_id=9998,
        family="test_required_ring",
        class_type="test_required_ring",
        topology_template="-23-",
        source_node_ids=[],
        main_labels=["23"],
        main_node_ids=[None],
        canonical_labels=["23"],
        open_left=True,
        open_right=True,
        can_ring=True,
        target_roles=["fatty_ring"],
    )
    assembler.views.append(required_view)
    assembler.view_by_id[int(required_view.fragment_id)] = required_view
    assembler.fragment_state_by_id[int(required_view.fragment_id)] = "available"

    created = ring_planner_module.assemble_priority_rings(assembler, [])

    assert created is True
    rings = [
        product
        for product in assembler.products
        if str(product.product_kind).startswith("fatty_ring_")
    ]
    assert rings
    assert int(required_view.fragment_id) in assembler.ring_exhausted_fragment_ids


def test_ring_audit_survives_stage3_product_normalization() -> None:
    raw = {
        "product_id": "ring:1",
        "product_type": "aliphatic_ring",
        "phase": "external_fatty_ring_closure",
        "fixed_source_node_ids": [],
        "claimed_resource_ids_by_su": {24: [1, 2]},
        "topology_notation": "ring",
        "node_id_notation": "1-2",
        "closed": True,
        "ring_audit": {
            "ring_kind": "side",
            "cycles": [[1, 2, 3, 4, 5, 6]],
            "complete": True,
        },
    }
    normalized = normalize_stage3_product(raw, generation_id=3)

    assert normalized.ring_audit == raw["ring_audit"]


def test_internal_branch_resource_is_materialized_at_center_anchor() -> None:
    center = make_node(
        445,
        20,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    left_connector = make_node(606, 27)
    left_terminal = make_node(144, 6)
    right_connector = make_node(607, 27)
    right_terminal = make_node(145, 6)
    open_boundary = make_node(42, 4)
    pool_23 = make_node(514, 23)
    pool_22 = make_node(464, 22)
    connect(center, left_connector, fixed=True)
    connect(left_connector, left_terminal, fixed=True)
    connect(center, right_connector, fixed=True)
    connect(right_connector, right_terminal, fixed=True)
    connect(center, open_boundary)
    nodes = [
        center,
        left_connector,
        left_terminal,
        right_connector,
        right_terminal,
        open_boundary,
        pool_23,
        pool_22,
    ]

    classification = classify_topology_units(nodes)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 445)
    assert [slot.anchor_node_id for slot in fragment.branch_contracts] == [445]

    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    )
    product = next(
        item
        for item in assembler.assemble().to_dict()["products"]
        if 445 in set(item["source_fragment_ids"])
    )
    assert product["actual_topology_gid"] == "144-606-445(-514-464)-607-145"
    assert product["claimed_resource_ids_by_su"] == {"22": [464], "23": [514]}


def test_single_unpaired_su24_is_converted_before_stage3_completion() -> None:
    nodes = [make_node(1, 24)]
    H = histogram(nodes)
    plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
    )
    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "internal_topology"
    assert plan.plan_kind == "sequential_phase_transactions"
    assert plan.blocked_reason == ""
    assert plan.diagnostics["su24_remaining_count"] == 0
    assert plan.diagnostics["internal_physical_conversion_count"] == 1
    assert len(plan.planned_conversions) == 1
    assert int(plan.planned_conversions[0].node_id) == 1
    assert int(plan.planned_conversions[0].from_su) == 24
    assert int(plan.planned_conversions[0].to_su) in {12, 13}
    assert int(H[24].item()) == 1


def test_multiple_unassembled_su24_convert_within_internal_budget() -> None:
    nodes = [make_node(node_id, 24) for node_id in range(1, 4)]
    H = histogram(nodes)

    plan = plan_stage3_resource_adjustments(
        H=H,
        nodes=nodes,
        enable_post_external_spectrum=False,
        internal_max_primary_adjustments=2,
    )

    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase == "internal_topology"
    assert plan.plan_kind == "sequential_phase_transactions"
    assert plan.histogram_adjustment_plan is not None
    ledger = plan.runtime["iteration_topology_ledger"]
    assert sorted(ledger.available_ids_by_su[24]) == [3]
    assert len(plan.planned_conversions) == 2
    assert sorted(int(op.node_id) for op in plan.planned_conversions) == [1, 2]
    assert plan.diagnostics["su24_remaining_count"] == 1
    assert plan.diagnostics["internal_physical_conversion_count"] == 2


def test_removed_generic_stage3_budget_key_is_rejected() -> None:
    adjuster = Layer4Adjuster()
    H = torch.zeros(33, dtype=torch.long)

    with pytest.raises(ValueError, match="max_primary_adjustments_per_stage"):
        adjuster.run_block_c_resource_adjustment_loop(
            H=H,
            nodes=[],
            E_target=None,
            max_primary_adjustments_per_stage=10,
        )


def _fixed_flex_phase_eval(
    *,
    explicit_flex_count: int,
    flex_upper: int,
    closure_ids=None,
    aromatic_ids=None,
    bridgehead_info=(0, 0, 0, 0, 0, 0, 0),
    histogram_su12_count: int = 0,
):
    closure_pool = {11: [], 22: [], 23: [], 24: []}
    closure_pool.update(dict(closure_ids or {}))
    aromatic_pool = {10: [], 12: [], 13: []}
    aromatic_pool.update(dict(aromatic_ids or {}))
    return SimpleNamespace(
        flex_audit={
            "scenario": "above_range",
            "above_range_excess": max(
                0, int(explicit_flex_count) - int(flex_upper)
            ),
            "enforced": True,
            "bridgehead_info": tuple(bridgehead_info),
            "histogram_su12_count": int(histogram_su12_count),
            "explicit_flex_count": int(explicit_flex_count),
            "flex_lower": 0,
            "flex_upper": int(flex_upper),
        },
        round_context=SimpleNamespace(
            generic_closure_pool_ids_by_su={
                int(key): list(values or [])
                for key, values in closure_pool.items()
                if int(key) != 24
            },
            su24_ring_resource_ids=list(closure_pool.get(24, []) or []),
            aromatic_donor_ids_by_su=aromatic_pool,
        ),
        flex_reconciliation_ids_by_su={
            int(key): list(values or [])
            for key, values in {**closure_pool, **aromatic_pool}.items()
        },
        assembly_view={},
        classification_view={},
    )


def _fixed_flex_difference(*, aliphatic: float, su13: float, su12: float = 0.0):
    ppm = list(range(201))
    diff = []
    for value in ppm:
        signal = 0.0
        if 10 <= value <= 80:
            signal = float(aliphatic)
        if 100 <= value <= 130:
            signal = float(su13)
        if 135 <= value <= 140:
            signal = float(su12)
        diff.append(signal)
    return {
        "ppm": ppm,
        "diff": diff,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
    }


def _plan_fixed_flex_capacity(phase_eval, H, difference_spectrum):
    return _fixed_flex_capacity_expansion_plan(
        phase_eval=phase_eval,
        H=H,
        E_target=torch.tensor([float(H.sum().item()), 100.0, 0.0, 0.0, 0.0, 0.0]),
        difference_spectrum=difference_spectrum,
        max_primary_adjustments=10,
        max_h_compensation_ops=0,
        h_tolerance=1.0,
        special_degree_meta=None,
        su11_rotation_index=0,
    )


def test_fixed_flex_phase_disables_generic_h_compensation(monkeypatch) -> None:
    donor_ids = list(range(101, 111))
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[23] = len(donor_ids)
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=1,
        flex_upper=0,
        closure_ids={23: donor_ids},
        aromatic_ids={10: list(range(201, 205)), 12: [301, 302]},
        histogram_su12_count=2,
    )
    observed = {}
    original = state_machine_module.build_histogram_adjustment_plan

    def capture(**kwargs):
        observed["max_h_compensation_ops"] = kwargs["max_h_compensation_ops"]
        return original(**kwargs)

    monkeypatch.setattr(
        state_machine_module,
        "build_histogram_adjustment_plan",
        capture,
    )
    _fixed_flex_capacity_expansion_plan(
        phase_eval=phase_eval,
        H=H,
        E_target=torch.tensor([20.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        difference_spectrum=_fixed_flex_difference(
            aliphatic=-1.0,
            su13=1.0,
            su12=-1.0,
        ),
        max_primary_adjustments=10,
        max_h_compensation_ops=8,
        h_tolerance=0.08,
        special_degree_meta=None,
        su11_rotation_index=0,
    )

    assert observed["max_h_compensation_ops"] == 0


def test_flex_reconciliation_uses_physical_nodes_before_topology_ownership(
    monkeypatch,
) -> None:
    nodes = [
        make_node(1, 12),
        make_node(2, 12),
        make_node(3, 11),
        make_node(4, 23),
        make_node(5, 13),
        make_node(6, 22),
        make_node(7, 24),
    ]
    classification = TopologyClassificationResult(
        fragments=[
            TopologyFragment(
                fragment_id=100,
                family="temporary",
                class_type="temporary_aromatic_view",
                topology_template="12",
                source_node_ids=[1],
                claimed_node_ids=[1],
            ),
            TopologyFragment(
                fragment_id=200,
                family="basic",
                class_type="fixed_flex",
                topology_template="11-23",
                source_node_ids=[2],
                claimed_node_ids=[2],
            ),
        ],
        resource_pool={
            "remaining_ids_by_su": {
                "11": [], "22": [], "23": [], "24": [],
            },
            "initial_ids_by_su": {
                "11": [], "22": [], "23": [], "24": [],
            },
            "su11_donor_ids_by_su": {
                "10": [], "12": [], "13": [],
            },
        },
        unclassified_nodes=[{"node_id": 4}],
        summary={},
    )
    flex_audit = {
        "scenario": "above_range",
        "enforced": True,
        "X": 2,
        "Y": 0,
        "Z": 2,
        "flex_lower": 0,
        "flex_upper": 1,
        "target_flex_count": 2,
        "explicit_flex_count": 2,
        "above_range_excess": 1,
        "hard_explicit_flex_fragment_ids": [200],
        "required_resource_counts": {},
    }
    monkeypatch.setattr(
        phase_eval_module,
        "_compute_flex_audit",
        lambda *args, **kwargs: dict(flex_audit),
    )

    phase_eval = phase_eval_module.build_stage3_phase_evaluation(
        nodes=nodes,
        classification=classification,
        generation_id=17,
    )

    assert phase_eval.phase == "flex_range_reconciliation"
    assert phase_eval.ownership_initialized is False
    assert phase_eval.round_context.ownership_by_node_id == {}
    assert phase_eval.flex_reconciliation_ids_by_su == {
        11: [3],
        12: [1, 2],
        13: [5],
        22: [6],
        23: [],
    }
    assert phase_eval.ownership_ledger["deferred_reason"] == (
        "flex_range_reconciliation_pending"
    )
    pool_diagnostics = phase_eval.diagnostics["flex_reconciliation_pool"]
    assert pool_diagnostics["protected_hard_flex_node_ids"] == [2]
    assert 4 in pool_diagnostics["excluded_unsafe_node_ids"]


def test_fixed_flex_capacity_prefers_preassembly_physical_pool() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = 2
    H[13] = 14
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=1,
        aromatic_ids={},
        bridgehead_info=(0, 0, 0, 0, 0, 0, 2),
        histogram_su12_count=2,
    )
    phase_eval.flex_reconciliation_ids_by_su = {
        10: [],
        11: [],
        12: [51, 52],
        23: [],
    }

    plan = _plan_fixed_flex_capacity(phase_eval, H, None)

    assert plan.ok is False
    assert plan.planned_conversions == []
    selection = plan.phase_result.infeasibility_proof["candidate_selection"]
    assert selection["candidate_attempts"][1]["reason"] == "aromatic_minimum_preserved"


def test_cycle60_like_flex_reconciliation_does_not_lock_classified_su12(
    monkeypatch,
) -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 43)]
    classification = TopologyClassificationResult(
        fragments=[
            TopologyFragment(
                fragment_id=1000 + node_id,
                family="temporary",
                class_type="temporary_aromatic_view",
                topology_template="12",
                source_node_ids=[node_id],
                claimed_node_ids=[node_id],
            )
            for node_id in range(1, 43)
        ],
        resource_pool={
            "remaining_ids_by_su": {"11": [], "22": [], "23": [], "24": []},
            "initial_ids_by_su": {"11": [], "22": [], "23": [], "24": []},
            "su11_donor_ids_by_su": {"10": [], "12": [], "13": []},
        },
        summary={},
    )
    flex_audit = {
        "scenario": "above_range",
        "above_range_excess": 2,
        "enforced": True,
        "X": 42,
        "Y": 17,
        "Z": 25,
        "bridgehead_info": (0, 0, 0, 0, 0, 0, 42),
        "histogram_su12_count": 42,
        "explicit_flex_count": 35,
        "flex_lower": 26,
        "flex_upper": 33,
        "target_flex_count": 26,
        "hard_explicit_flex_fragment_ids": [],
        "required_resource_counts": {},
    }
    monkeypatch.setattr(
        phase_eval_module,
        "_compute_flex_audit",
        lambda *args, **kwargs: dict(flex_audit),
    )
    phase_eval = phase_eval_module.build_stage3_phase_evaluation(
        nodes=nodes,
        classification=classification,
        generation_id=60,
    )
    H = torch.zeros(33, dtype=torch.long)
    for su_type, count in {
        5: 154,
        10: 34,
        11: 42,
        12: 42,
        13: 66,
        22: 48,
        23: 87,
        24: 20,
    }.items():
        H[su_type] = count

    plan = _fixed_flex_capacity_expansion_plan(
        phase_eval=phase_eval,
        H=H,
        E_target=None,
        difference_spectrum=None,
        max_primary_adjustments=10,
        max_h_compensation_ops=0,
        h_tolerance=1.0,
        special_degree_meta=None,
        su11_rotation_index=0,
    )

    assert phase_eval.ownership_initialized is False
    assert phase_eval.round_context.ownership_by_node_id == {}
    assert phase_eval.flex_reconciliation_ids_by_su[12] == list(range(1, 43))
    assert plan.ok is True
    assert plan.reason == "stage3_rebuild_required"
    assert [(op.from_su, op.to_su) for op in plan.planned_conversions] == [
        (12, 13)
    ] * 5
    assert [op.node_id for op in plan.planned_conversions] == list(range(1, 6))
    assert plan.diagnostics["fixed_flex_capacity_expansion"][
        "selection_stop_reason"
    ] == "fixed_flex_count_entered_valid_range"


def test_fixed_flex_capacity_uses_aromatic_fallback_without_layer2_spectrum() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = 2
    H[13] = 14
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=1,
        aromatic_ids={
            10: list(range(20, 24)),
            12: [51, 52],
            13: list(range(60, 74)),
        },
        bridgehead_info=(0, 0, 0, 0, 0, 0, 2),
        histogram_su12_count=2,
    )

    missing = _plan_fixed_flex_capacity(phase_eval, H, None)
    stale = _fixed_flex_difference(aliphatic=-1.0, su13=1.0)
    stale["source_stage"] = "post_layer3"
    wrong_source = _plan_fixed_flex_capacity(phase_eval, H, stale)

    assert missing.ok is False
    assert wrong_source.ok is False
    assert missing.planned_conversions == []
    assert wrong_source.planned_conversions == []
    assert missing.phase_result.infeasibility_proof["candidate_selection"][
        "su23_spectrum_eligible"
    ] is False


def test_fixed_flex_capacity_skips_su23_but_uses_aromatic_fallback() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = 2
    H[13] = 14
    H[23] = 10
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=1,
        closure_ids={23: list(range(1, 11))},
        aromatic_ids={
            10: list(range(20, 24)),
            12: [51, 52],
            13: list(range(60, 74)),
        },
        bridgehead_info=(0, 0, 0, 0, 0, 0, 2),
        histogram_su12_count=2,
    )

    plan = _plan_fixed_flex_capacity(
        phase_eval,
        H,
        _fixed_flex_difference(aliphatic=1.0, su13=1.0),
    )

    assert plan.ok is False
    assert plan.planned_conversions == []
    attempts = plan.phase_result.infeasibility_proof["candidate_selection"][
        "candidate_attempts"
    ]
    assert attempts[0]["source_su"] == 23
    assert attempts[0]["reason"] == (
        "su23_requires_layer2_aliphatic_surplus_and_su13_deficit"
    )


def test_fixed_flex_capacity_prefers_real_su23_to_su13_nodes_and_rebuilds() -> None:
    donor_ids = list(range(101, 111))
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[23] = len(donor_ids)
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=1,
        flex_upper=0,
        closure_ids={23: donor_ids},
        aromatic_ids={10: list(range(201, 205)), 12: [301, 302]},
        histogram_su12_count=2,
    )

    plan = _plan_fixed_flex_capacity(
        phase_eval,
        H,
        _fixed_flex_difference(aliphatic=-1.0, su13=1.0, su12=-1.0),
    )

    assert plan.ok is True
    assert plan.reason == "stage3_rebuild_required"
    assert plan.phase_result.rebuild_required is True
    assert [op.from_su for op in plan.planned_conversions] == [23] * 10
    assert [op.to_su for op in plan.planned_conversions] == [13] * 10
    assert [op.node_id for op in plan.planned_conversions] == donor_ids
    assert all(op.to_su != 11 for op in plan.planned_conversions)
    assert plan.diagnostics["restart_stage3_after_rebuild"] is True


def test_fixed_flex_capacity_uses_su12_without_donor_window_surplus() -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = 2
    H[13] = 14
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=1,
        aromatic_ids={10: list(range(20, 24)), 12: [51, 52], 13: list(range(60, 74))},
        bridgehead_info=(0, 0, 0, 0, 0, 0, 2),
        histogram_su12_count=2,
    )

    selected = _plan_fixed_flex_capacity(
        phase_eval,
        H,
        _fixed_flex_difference(aliphatic=1.0, su13=-1.0, su12=0.0),
    )

    assert selected.ok is False
    assert selected.planned_conversions == []
    attempts = selected.phase_result.infeasibility_proof["candidate_selection"][
        "candidate_attempts"
    ]
    assert attempts[0]["source_su"] == 23
    assert attempts[0]["reason"] == (
        "su23_requires_layer2_aliphatic_surplus_and_su13_deficit"
    )


def test_fixed_flex_capacity_uses_full_primary_budget_when_still_above_range() -> None:
    donor_ids = list(range(51, 61))
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = len(donor_ids)
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=0,
        aromatic_ids={10: list(range(20, 24)), 12: donor_ids},
        bridgehead_info=(0, 0, 0, 0, 0, 0, len(donor_ids)),
        histogram_su12_count=len(donor_ids),
    )

    plan = _plan_fixed_flex_capacity(phase_eval, H, None)

    assert plan.ok is True
    # Seven primaries would leave SU12 at its minimum with no distinct donor
    # for the required parity conversion. Six is the largest physical batch.
    assert len(plan.planned_conversions) == 6
    assert [op.node_id for op in plan.planned_conversions] == donor_ids[:6]
    assert all((op.from_su, op.to_su) == (12, 13) for op in plan.planned_conversions)
    selection = plan.diagnostics["fixed_flex_capacity_expansion"]
    assert selection["selected_completed_flex_range"] is False
    assert selection["selection_stop_reason"] == "available_valid_donors_exhausted"
    assert selection["predicted_flex_after"]["scenario"] == "above_range"
    assert selection["predicted_excess_after"] < selection["predicted_excess_before"]
    assert plan.diagnostics["candidate_alternatives_enabled"] is True


def test_aromatic_donor_pool_uses_physical_type_despite_topology_exclusions() -> None:
    nodes = [
        make_node(1, 10),
        make_node(2, 12),
        make_node(3, 13),
    ]

    pool = _build_resource_pool(
        build_node_tokens(nodes),
        consumed={1, 2, 3},
        excluded_ids={1, 2, 3},
    )

    assert pool["su11_donor_ids_by_su"] == {
        "10": [1],
        "12": [2],
        "13": [3],
    }


def test_rebuilt_aromatized_su12_nodes_all_enter_flex_pool() -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 37)]
    classification = TopologyClassificationResult(
        fragments=[
            TopologyFragment(
                fragment_id=1000 + node_id,
                family="temporary",
                class_type="temporary_aromatic_view",
                topology_template="12",
                source_node_ids=[node_id],
                claimed_node_ids=[node_id],
            )
            for node_id in range(1, 37)
        ],
        resource_pool={},
        classification_rejects=[{"node_id": node_id} for node_id in range(1, 37)],
    )

    pool, _diagnostics = phase_eval_module._build_flex_reconciliation_pool(
        nodes,
        classification,
        {
            "hard_explicit_flex_fragment_ids": [
                1000 + node_id for node_id in range(1, 37)
            ],
        },
    )

    assert pool[12] == list(range(1, 37))


def test_fixed_flex_candidate_binds_distinct_real_parity_donor(
    monkeypatch,
) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 5
    H[13] = 3

    def fake_flex_impact(H_before, H_after, _audit):
        gained = int(H_after[13].item()) - int(H_before[13].item())
        return SimpleNamespace(
            before={"scenario": "above_range", "flex_lower": 0, "flex_upper": 0},
            after={"scenario": "in_range", "flex_lower": 0, "flex_upper": gained},
        )

    monkeypatch.setattr(state_machine_module, "evaluate_flex_impact", fake_flex_impact)
    monkeypatch.setattr(
        state_machine_module,
        "audit_layer1_port_roles",
        lambda *_args, **_kwargs: {
            "ok": True,
            "total_even": True,
            "role_parity_consistent": True,
        },
    )

    operations, selection = _fixed_flex_capacity_candidate(
        H=H,
        audit={"explicit_flex_count": 1, "flex_lower": 0, "flex_upper": 0},
        pool={10: [], 11: [], 12: [101, 102, 103, 104, 105], 13: [201, 202, 203], 23: []},
        spectrum_audit={"eligible": False},
        max_primary_adjustments=10,
        special_degree_meta=None,
    )

    assert [op.node_id for op in operations] == [101]
    assert selection["predicted_parity_balance_node_ids"] == [102]
    assert set(selection["selected_primary_node_ids"]).isdisjoint(
        selection["predicted_parity_balance_node_ids"]
    )


def test_fixed_flex_candidate_ignores_su10_and_su24_donors(
    monkeypatch,
) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 2
    H[11] = 2
    H[13] = 3
    H[24] = 2

    def fake_flex_impact(H_before, H_after, _audit):
        gained = int(H_after[13].item()) - int(H_before[13].item())
        return SimpleNamespace(
            before={"scenario": "above_range", "flex_lower": 0, "flex_upper": 0},
            after={"scenario": "in_range", "flex_lower": 0, "flex_upper": gained},
        )

    monkeypatch.setattr(state_machine_module, "evaluate_flex_impact", fake_flex_impact)
    monkeypatch.setattr(
        state_machine_module,
        "audit_layer1_port_roles",
        lambda *_args, **_kwargs: {
            "ok": True,
            "total_even": True,
            "role_parity_consistent": True,
        },
    )
    operations, selection = _fixed_flex_capacity_candidate(
        H=H,
        audit={"explicit_flex_count": 1, "flex_lower": 0, "flex_upper": 0},
        pool={10: [201, 202], 11: [101, 102], 24: [301, 302]},
        spectrum_audit={"eligible": False},
        max_primary_adjustments=10,
        special_degree_meta=None,
    )

    assert operations
    assert all(
        (op.from_su, op.to_su) == (11, 13)
        and op.node_id in {101, 102}
        for op in operations
    )
    assert selection["selected_source_sus"] == [11]
    assert selection["authorized_adjustment_su_types"] == [11, 12, 13, 22, 23]
    assert 10 not in selection["selected_counts_by_su"]
    assert 24 not in selection["selected_counts_by_su"]


def test_fixed_flex_retries_without_repeating_rejected_final_step() -> None:
    donor_ids = list(range(51, 61))
    H = torch.zeros(33, dtype=torch.long)
    H[10] = 4
    H[12] = len(donor_ids)
    phase_eval = _fixed_flex_phase_eval(
        explicit_flex_count=2,
        flex_upper=0,
        aromatic_ids={10: list(range(20, 24)), 12: donor_ids},
        bridgehead_info=(0, 0, 0, 0, 0, 0, len(donor_ids)),
        histogram_su12_count=len(donor_ids),
    )
    first = _plan_fixed_flex_capacity(phase_eval, H, None)
    rejected = set()
    rejected_signature = _register_rejected_resource_candidate(
        first,
        first.histogram_adjustment_plan,
        rejected,
    )

    assert rejected_signature
    retry = _fixed_flex_capacity_expansion_plan(
        phase_eval=phase_eval,
        H=H,
        E_target=torch.tensor([float(H.sum().item()), 100.0, 0.0, 0.0, 0.0, 0.0]),
        difference_spectrum=None,
        max_primary_adjustments=10,
        max_h_compensation_ops=0,
        h_tolerance=1.0,
        special_degree_meta=None,
        su11_rotation_index=0,
        forbidden_primary_conversion_signatures=sorted(rejected),
    )

    assert retry.ok is True
    assert rejected_signature in retry.diagnostics[
        "forbidden_primary_conversion_signatures"
    ]
    retry_conversions = [
        HistogramConversion(
            from_su=int(op.from_su),
            to_su=int(op.to_su),
            node_ids=[int(op.node_id)],
        )
        for op in retry.planned_conversions
    ]
    assert _stateful_conversion_bundle_signature(
        H, retry_conversions
    ) != rejected_signature


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (HistogramConversion(from_su=24, to_su=12), True),
        (HistogramConversion(from_su=23, to_su=24), False),
        (HistogramConversion(from_su=23, to_su=22, via_su=24), False),
    ],
)
def test_internal_topology_only_allows_su24_as_aromatic_donor(op, expected) -> None:
    histogram_plan = HistogramAdjustmentPlan(groups=[AtomicAdjustmentGroup(
        primary_op=op,
    )])

    assert _internal_conversion_is_allowed(histogram_plan) is expected


def test_pre_rebuild_parity_uses_reverse_direction_when_preferred_donor_is_empty() -> None:
    nodes = [make_node(1, 13), make_node(2, 13)] + [
        make_node(node_id, 24) for node_id in range(3, 13)
    ]
    H = histogram(nodes)
    targets = [12, 12, 13, 12, 12, 13, 12, 12, 13, 12]
    groups = []
    H_after = H.clone()
    for node, target in zip(nodes[2:], targets):
        op = HistogramConversion(
            from_su=24,
            to_su=int(target),
            count=1,
            reason="residual_su24_aromatic_rotation",
            node_ids=[int(node.global_id)],
        )
        groups.append(AtomicAdjustmentGroup(
            primary_op=op,
            combined_delta={24: -1, int(target): 1},
        ))
        H_after[24] -= 1
        H_after[int(target)] += 1
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={
            10: [], 11: [], 12: [], 13: [1, 2],
            22: [], 23: [], 24: list(range(3, 13)),
        },
    )
    plan = Stage3AdjustmentPlan(runtime={"iteration_topology_ledger": ledger})
    histogram_plan = HistogramAdjustmentPlan(
        groups=groups,
        H_before=H,
        H_after=H_after,
        inventory_before={13: 2, 24: 10},
    )

    ok, diagnostics = _append_stage3_parity_balance(
        plan=plan,
        histogram_plan=histogram_plan,
        nodes=nodes,
        special_degree_meta=None,
        balance_cursor=1,
    )

    assert ok is True, repr(diagnostics)
    assert diagnostics["required_direction"] == [12, 13]
    assert diagnostics["applied_direction"] == [13, 12]
    assert histogram_plan.parity_balance_op.node_ids == [1]


def test_pre_rebuild_parity_uses_real_aromatic_nodes_omitted_by_accounting() -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 7)]
    H = histogram(nodes)
    H_after = H.clone()
    H_after[12] -= 1
    H_after[13] += 1
    primary = HistogramConversion(
        from_su=12,
        to_su=13,
        node_ids=[1],
        reason="test_primary_conversion",
    )
    histogram_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=primary,
            combined_delta={12: -1, 13: 1},
        )],
        H_before=H,
        H_after=H_after,
        inventory_before={12: 6, 13: 0},
        accounting_diagnostics={
            "final_remaining_ids_by_su": {23: []},
            "authorized_ids_before_by_su": {23: []},
        },
    )

    ok, diagnostics = _append_stage3_parity_balance(
        plan=Stage3AdjustmentPlan(),
        histogram_plan=histogram_plan,
        nodes=nodes,
        special_degree_meta=None,
        balance_cursor=1,
    )

    assert ok is True, diagnostics
    assert diagnostics["applied_direction"] == [12, 13]
    assert histogram_plan.parity_balance_op.node_ids == [2]


def test_pre_rebuild_port_role_balance_alternates_11_and_12_directions() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 12),
        make_node(3, 24),
        make_node(4, 22),
    ]
    H = histogram(nodes)
    H_after = H.clone()
    H_after[22] -= 1
    H_after[24] += 1
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={
            10: [], 11: [1], 12: [2], 13: [],
            22: [4], 23: [], 24: [3],
        },
    )

    def run(cursor):
        op = HistogramConversion(
            from_su=22,
            to_su=24,
            count=1,
            reason="test_role_balance",
            node_ids=[4],
        )
        plan = Stage3AdjustmentPlan(runtime={"iteration_topology_ledger": ledger})
        histogram_plan = HistogramAdjustmentPlan(
            groups=[AtomicAdjustmentGroup(
                primary_op=op,
                combined_delta={22: -1, 24: 1},
            )],
            H_before=H,
            H_after=H_after,
            inventory_before={11: 1, 12: 1, 22: 1, 24: 1},
        )
        ok, diagnostics = _append_stage3_parity_balance(
            plan=plan,
            histogram_plan=histogram_plan,
            nodes=nodes,
            special_degree_meta=None,
            balance_cursor=int(cursor),
        )
        return ok, diagnostics, histogram_plan

    first_ok, first_diagnostics, first_plan = run(0)
    second_ok, second_diagnostics, second_plan = run(1)

    assert first_ok is True, repr(first_diagnostics)
    assert second_ok is True
    assert (first_plan.port_role_balance_op.from_su, first_plan.port_role_balance_op.to_su) == (11, 12)
    # SU12 is already below the protected minimum, so the alternating
    # preference falls back to the safe SU11 donor direction.
    assert (second_plan.port_role_balance_op.from_su, second_plan.port_role_balance_op.to_su) == (11, 12)
    assert first_diagnostics["next_cursor"] == 1
    assert second_diagnostics["next_cursor"] == 2


def test_pre_rebuild_port_balance_keeps_topology_candidate_outside_h_soft_window() -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 6)]
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 5
    H[23] = 300
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={
            10: [], 11: [], 12: list(range(1, 6)), 13: [],
            22: [], 23: [], 24: [],
        },
    )
    plan = Stage3AdjustmentPlan(runtime={"iteration_topology_ledger": ledger})
    histogram_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=12,
                to_su=12,
                node_ids=[99],
                reason="test_existing_topology_conversion",
            ),
            combined_delta={},
        )],
        H_before=H,
        H_after=H.clone(),
        inventory_before={12: 5},
    )
    target_h = float(_h_total(H, None)) / 1.08

    ok, diagnostics = _append_stage3_parity_balance(
        plan=plan,
        histogram_plan=histogram_plan,
        nodes=nodes,
        special_degree_meta=None,
        balance_cursor=1,
        E_target=torch.tensor([0.0, target_h, 0.0, 0.0, 0.0, 0.0]),
        h_tolerance=0.08,
    )

    assert ok is True
    assert diagnostics["reason"] == "pre_rebuild_ports_balanced"
    assert diagnostics["applied_direction"] == [12, 13]
    assert float(diagnostics["h_after"]) > float(diagnostics["h_soft_bounds"][1])
    assert histogram_plan.parity_balance_op.node_ids == [1]


def test_deferred_boundary_balance_uses_aromatic_donor_with_empty_external_ledger() -> None:
    nodes = [make_node(node_id, 12) for node_id in range(1, 6)]
    H = torch.zeros(33, dtype=torch.long)
    H[12] = 5
    H[23] = 300
    ledger = build_iteration_ledger(
        nodes,
        resource_ids_by_su={
            10: [], 11: [], 12: [], 13: [], 22: [], 23: [], 24: [],
        },
    )
    plan = Stage3AdjustmentPlan(
        runtime={"iteration_topology_ledger": ledger},
    )

    balanced_nodes, balanced_H, diagnostics = (
        block_c_module._apply_deferred_port_balance(
            plan=plan,
            nodes=nodes,
            H=H,
            special_degree_meta=None,
            balance_cursor=1,
            E_target=None,
            h_tolerance=0.08,
        )
    )

    assert diagnostics["reason"] == "pre_rebuild_ports_balanced"
    assert diagnostics["balance_applied"] is True
    assert diagnostics["applied_direction"] == [12, 13]
    assert int(balanced_H[12].item()) == 4
    assert int(balanced_H[13].item()) == 1
    assert int(balanced_nodes[0].su_type) == 13


def test_histogram_audit_treats_h_tolerance_as_soft_only() -> None:
    adjuster = Layer4Adjuster()
    H = torch.zeros(33, dtype=torch.long)
    E_target = torch.tensor([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])

    diagnostics = adjuster._evaluate_required_hist_constraints(H, E_target)

    assert diagnostics["ok"] is True
    assert diagnostics["h_ok"] is False
    assert diagnostics["h_constraint_mode"] == "soft"
    assert diagnostics["h_topology_gate"] is False
    assert diagnostics["h_rel_exceeds_soft_tolerance"] is True


def test_external_conversion_continues_same_generation_before_boundary(
    monkeypatch,
) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[23] = 1
    calls = []

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        def plan_topology_resource_adjustments(self, **kwargs):
            calls.append({
                "progress_phase": kwargs["progress_phase"],
                "generation_id": kwargs["generation_id"],
            })
            if len(calls) == 1:
                H_after = kwargs["H"].clone()
                H_after[23] -= 1
                H_after[22] += 1
                histogram_plan = HistogramAdjustmentPlan(
                    groups=[AtomicAdjustmentGroup(
                        primary_op=HistogramConversion(
                            from_su=23,
                            to_su=22,
                            node_ids=[77],
                            reason="test_rejected_su24_resource_variant",
                        ),
                        combined_delta={23: -1, 22: 1},
                    )],
                    H_before=kwargs["H"],
                    H_after=H_after,
                    inventory_before={23: 1},
                )
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="external_conversion_continuation",
                    phase="external_topology_assembly",
                    plan_kind="external_conversion_continuation",
                    histogram_adjustment_plan=histogram_plan,
                    phase_result=Stage3PhaseResult(
                        active_phase="external_topology_assembly",
                        phase_completed=False,
                        rebuild_required=False,
                    ),
                    diagnostics={"candidate_alternatives_enabled": True},
                    runtime={},
                )
            if len(calls) == 2:
                assert kwargs["progress_phase"] is Stage3ProgressPhase.EXTERNAL_TOPOLOGY
                assert kwargs["external_primary_adjustments_used"] == 1
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_complete",
                    phase="complete",
                    plan_kind="stage3_state_machine_complete",
                    phase_result=Stage3PhaseResult(
                        active_phase="complete",
                        phase_completed=True,
                        rebuild_required=False,
                    ),
                    diagnostics={},
                    runtime={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
                runtime={},
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, nodes, H, plan):
            return apply_stage3_node_adjustments(nodes=nodes, H=H, plan=plan)

    monkeypatch.setattr(
        block_c_module,
        "_apply_deferred_port_balance",
        lambda **kwargs: (
            list(kwargs["nodes"]),
            kwargs["H"].detach().clone(),
            {"reason": "pre_rebuild_ports_balanced", "balance_applied": False},
        ),
    )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H,
        nodes=[make_node(77, 23)],
        E_target=None,
        enable_post_external_spectrum=False,
        outer_cycle=2,
        rebuild_fn=lambda **kwargs: (
            list(kwargs["nodes_seed"]),
            kwargs["H_seed"].detach().clone(),
            {"r2": 0.5, "source_stage": "layer2", "spectral_contract": "template_center_mu"},
            0.5,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert len(calls) == 3
    assert calls[0]["progress_phase"] is Stage3ProgressPhase.FLEX_RECONCILIATION
    assert calls[1]["progress_phase"] is Stage3ProgressPhase.EXTERNAL_TOPOLOGY
    assert calls[2]["progress_phase"] is Stage3ProgressPhase.FLEX_RECONCILIATION
    assert [call["generation_id"] for call in calls] == [
        2_000_000,
        2_000_000,
        2_000_001,
    ]
    assert result.nodes[0].su_type == 22
    assert result.accepted_cycles == 1
    assert [record.cycle for record in result.cycles] == [1, 1, 2]
    assert [record.attempt for record in result.cycles] == [1, 2, 1]
    assert result.diagnostics["total_cycle_count"] == 2
    assert result.diagnostics["total_planner_attempt_count"] == 3


def test_recovery_continues_then_rebuilds_once_at_external_completion(
    monkeypatch,
) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[24] = 1
    nodes = [make_node(1, 24)]
    calls = []
    rebuild_calls = []
    balance_calls = []

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        def plan_topology_resource_adjustments(self, **kwargs):
            calls.append({
                "allow_recovery": bool(kwargs.get("allow_ring_recovery", False)),
                "generation_id": kwargs["generation_id"],
                "progress_phase": kwargs["progress_phase"],
            })
            if len(calls) == 1:
                H_after = H.clone()
                H_after[24] = 0
                H_after[23] = 1
                histogram_plan = HistogramAdjustmentPlan(
                    phase="external_side_chain_closure",
                    groups=[AtomicAdjustmentGroup(
                        primary_op=HistogramConversion(
                            from_su=24,
                            to_su=23,
                            node_ids=[1],
                            reason="recycle_pure_resource_ring",
                        ),
                        combined_delta={24: -1, 23: 1},
                    )],
                    H_before=H,
                    H_after=H_after,
                    inventory_before={24: 1},
                    inventory_after={23: 1},
                )
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="external_recovery_continuation",
                    phase="external_topology_assembly",
                    plan_kind="external_recovery_continuation",
                    histogram_adjustment_plan=histogram_plan,
                    phase_result=Stage3PhaseResult(
                        active_phase="external_topology",
                        rebuild_required=False,
                    ),
                    diagnostics={
                        "external_recovery_continuation": True,
                    },
                )
            if len(calls) == 2:
                assert kwargs["external_primary_adjustments_used"] == 1
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_complete",
                    phase="complete",
                    phase_result=Stage3PhaseResult(
                        active_phase="complete",
                        phase_completed=True,
                        rebuild_required=False,
                    ),
                    diagnostics={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, nodes, H, plan):
            return apply_stage3_node_adjustments(nodes=nodes, H=H, plan=plan)

    def rebuild_fn(**_kwargs):
        rebuild_calls.append(True)
        return (
            list(_kwargs["nodes_seed"]),
            _kwargs["H_seed"].detach().clone(),
            {"r2": 0.5, "source_stage": "layer2", "spectral_contract": "template_center_mu"},
            0.5,
        )

    def record_boundary_balance(**kwargs):
        balance_calls.append(True)
        return (
            list(kwargs["nodes"]),
            kwargs["H"].detach().clone(),
            {"reason": "pre_rebuild_ports_balanced", "balance_applied": False},
        )

    monkeypatch.setattr(
        block_c_module,
        "_apply_deferred_port_balance",
        record_boundary_balance,
    )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H,
        nodes=nodes,
        E_target=None,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
        enable_post_external_spectrum=False,
    )

    assert result.ok is True
    assert result.H[24].item() == 0
    assert result.H[23].item() == 1
    assert result.nodes[0].su_type == 23
    assert rebuild_calls == [True]
    assert balance_calls == [True]
    assert [call["allow_recovery"] for call in calls] == [False, False, False]
    assert [call["generation_id"] for call in calls] == [0, 0, 1]
    assert result.cycles[0].fresh_rebuild is False


def test_side_resource_failure_opens_one_shot_recovery_gate(
    monkeypatch,
) -> None:
    H = torch.zeros(33, dtype=torch.long)
    H[24] = 1
    nodes = [make_node(1, 24)]
    calls = []
    rebuild_calls = []

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        def plan_topology_resource_adjustments(self, **kwargs):
            allow_recovery = bool(kwargs.get("allow_ring_recovery", False))
            calls.append({
                "allow_recovery": allow_recovery,
                "generation_id": kwargs["generation_id"],
                "external_used": kwargs["external_primary_adjustments_used"],
            })
            if len(calls) == 1:
                self._stage3_external_assembler = SimpleNamespace()
                histogram_plan = HistogramAdjustmentPlan(
                    phase="external_side_chain_closure",
                    groups=[],
                    H_before=H,
                    H_after=H.clone(),
                )
                return Stage3AdjustmentPlan(
                    ok=False,
                    reason="phase_resource_infeasible",
                    phase="external_topology_assembly",
                    histogram_adjustment_plan=histogram_plan,
                    pending_contracts=[{
                        "target_role": "side_chain",
                        "required_counts": {"11": 1},
                        "proof": {
                            "reason": "external_resource_deficit",
                            "failure_domain": "resource_capacity",
                            "required_counts": {"11": 1},
                            "available_counts": {},
                            "missing_counts": {"11": 1},
                            "conversion_capacity_proof": {
                                "search_exhausted": True,
                                "ordinary_conversion_feasible": False,
                                "missing_counts": {"11": 1},
                                "remaining_primary_budget": 10,
                                "allowed_donor_su_types": [11, 12, 13, 22, 23],
                                "failure_domain": "resource_capacity",
                            },
                        },
                    }],
                    phase_result=Stage3PhaseResult(
                        active_phase="external_topology_assembly",
                        infeasibility_proof={
                            "failed_contract": {
                                "target_role": "side_chain",
                                "required_counts": {"11": 1},
                                "proof": {
                                    "failure_domain": "resource_capacity",
                                    "missing_counts": {"11": 1},
                                    "conversion_capacity_proof": {
                                        "search_exhausted": True,
                                        "ordinary_conversion_feasible": False,
                                        "missing_counts": {"11": 1},
                                        "remaining_primary_budget": 10,
                                        "allowed_donor_su_types": [11, 12, 13, 22, 23],
                                        "failure_domain": "resource_capacity",
                                    },
                                },
                            },
                            "reason": "external_resource_deficit",
                        },
                    ),
                )
            if len(calls) == 2:
                H_after = H.clone()
                H_after[24] = 0
                H_after[23] = 1
                histogram_plan = HistogramAdjustmentPlan(
                    phase="external_side_chain_closure",
                    groups=[AtomicAdjustmentGroup(
                        primary_op=HistogramConversion(
                            from_su=24,
                            to_su=23,
                            node_ids=[1],
                            reason="recycle_pure_resource_ring",
                        ),
                        combined_delta={24: -1, 23: 1},
                    )],
                    H_before=H,
                    H_after=H_after,
                    inventory_before={24: 1},
                    inventory_after={23: 1},
                )
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="external_recovery_continuation",
                    phase="external_topology_assembly",
                    plan_kind="external_recovery_continuation",
                    histogram_adjustment_plan=histogram_plan,
                    phase_result=Stage3PhaseResult(
                        active_phase="external_topology_assembly",
                        rebuild_required=False,
                    ),
                    diagnostics={
                        "external_recovery_continuation": True,
                    },
                )
            if len(calls) == 3:
                assert kwargs["external_primary_adjustments_used"] == 1
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_complete",
                    phase="complete",
                    phase_result=Stage3PhaseResult(
                        active_phase="complete",
                        phase_completed=True,
                    ),
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                ),
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, nodes, H, plan):
            return apply_stage3_node_adjustments(nodes=nodes, H=H, plan=plan)

    monkeypatch.setattr(
        block_c_module,
        "ring_recovery_candidate_costs",
        lambda _assembler: [{
            "product_kind": "fatty_ring_side",
            "contract_id": "ring-1",
            "su24_cost": 1,
        }],
    )
    monkeypatch.setattr(
        block_c_module,
        "_apply_deferred_port_balance",
        lambda **kwargs: (
            list(kwargs["nodes"]),
            kwargs["H"].detach().clone(),
            {"reason": "pre_rebuild_ports_balanced", "balance_applied": False},
        ),
    )

    def rebuild_fn(**kwargs):
        rebuild_calls.append(True)
        return (
            list(kwargs["nodes_seed"]),
            kwargs["H_seed"].detach().clone(),
            {"r2": 0.5, "source_stage": "layer2", "spectral_contract": "template_center_mu"},
            0.5,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H,
        nodes=nodes,
        E_target=None,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
        enable_post_external_spectrum=False,
    )

    assert result.ok is True
    assert [call["allow_recovery"] for call in calls] == [
        False,
        True,
        False,
        False,
    ]
    assert [call["generation_id"] for call in calls] == [0, 0, 0, 1]
    assert [call["external_used"] for call in calls] == [0, 0, 1, 0]
    assert result.nodes[0].su_type == 23
    assert rebuild_calls == [True]


def test_accepted_rebuild_restarts_stage3_with_new_graph_generation() -> None:
    initial_nodes = [
        make_node(1, 23),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 22),
    ]
    H = histogram(initial_nodes)
    calls = []

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0
        _stage3_su24_nonfused_cursor = 0

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        def plan_topology_resource_adjustments(self, **kwargs):
            calls.append({
                "progress_phase": kwargs["progress_phase"],
                "generation_id": kwargs["generation_id"],
            })
            if len(calls) == 1:
                H_after = kwargs["H"].clone()
                histogram_plan = HistogramAdjustmentPlan(
                    groups=[AtomicAdjustmentGroup(
                        primary_op=HistogramConversion(
                            from_su=23,
                            to_su=22,
                            node_ids=[1],
                            reason="test_accepted_rebuild",
                        ),
                        h_compensation_ops=[HistogramConversion(
                            from_su=22,
                            to_su=23,
                            node_ids=[2],
                            reason="test_accepted_rebuild_balance",
                        )],
                        combined_delta={},
                    )],
                    H_before=kwargs["H"],
                    H_after=H_after,
                        inventory_before={22: 2, 23: 2},
                )
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="external_conversion_continuation",
                    phase="external_fatty_ring_closure",
                    plan_kind="external_conversion_continuation",
                    histogram_adjustment_plan=histogram_plan,
                    phase_result=Stage3PhaseResult(
                        active_phase="external_fatty_ring_closure",
                        phase_completed=False,
                        rebuild_required=False,
                    ),
                    diagnostics={"su24_rebuild_reason": "support_conversion"},
                    runtime={},
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
                diagnostics={},
                runtime={},
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, nodes, H, plan):
            rebuilt_nodes = [
                make_node(1, 22),
                make_node(2, 23),
                make_node(3, 23),
                make_node(4, 22),
            ]
            return SimpleNamespace(
                changed=True,
                nodes=rebuilt_nodes,
                H=plan.histogram_adjustment_plan.H_after.clone(),
                requested_counts={},
                unresolved_counts={},
                applied_ops=[
                    {"node_id": 1, "from_su": 23, "to_su": 22},
                    {"node_id": 2, "from_su": 22, "to_su": 23},
                ],
                request_ops=[],
                rejected_ops=[],
            )

    def rebuild_fn(*, H_seed, phase, cycle_idx, nodes_seed):
        return (
            list(nodes_seed),
            H_seed.clone(),
            {"r2": 0.5, "source_stage": "layer2", "spectral_contract": "template_center_mu"},
            0.5,
        )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H,
        nodes=initial_nodes,
        E_target=None,
        enable_post_external_spectrum=False,
        outer_cycle=3,
        rebuild_fn=rebuild_fn,
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert len(calls) == 3
    assert calls[0]["generation_id"] == 3_000_000
    assert calls[1]["generation_id"] == 3_000_000
    assert calls[1]["progress_phase"] is Stage3ProgressPhase.EXTERNAL_TOPOLOGY
    assert calls[2]["generation_id"] == 3_000_001
    assert calls[2]["progress_phase"] is Stage3ProgressPhase.FLEX_RECONCILIATION


def _su24_conversion_plan(
    H_before: torch.Tensor,
    candidate: dict,
) -> Stage3AdjustmentPlan:
    H_after = H_before.detach().clone()
    H_after[23] -= 1
    H_after[22] += 1
    histogram_plan = HistogramAdjustmentPlan(
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(
                from_su=23,
                to_su=22,
                node_ids=[1],
                reason="test_pending_su24_candidate_conversion",
            ),
            combined_delta={23: -1, 22: 1},
        )],
        H_before=H_before.detach().clone(),
        H_after=H_after,
        inventory_before={22: int(H_before[22]), 23: int(H_before[23])},
    )
    return Stage3AdjustmentPlan(
        ok=True,
        reason="external_conversion_continuation",
        phase="external_fatty_ring_closure",
        plan_kind="external_conversion_continuation",
        histogram_adjustment_plan=histogram_plan,
        phase_result=Stage3PhaseResult(
            active_phase="external_fatty_ring_closure",
            phase_completed=False,
            rebuild_required=False,
        ),
        diagnostics={
            "candidate_alternatives_enabled": True,
            "su24_conversion_candidate": dict(candidate),
        },
    )


def test_candidate_conversion_continues_then_discards_state_at_boundary() -> None:
    initial_nodes = [
        make_node(1, 23),
        make_node(2, 22),
        make_node(100, 24),
        make_node(101, 24),
    ]
    H_before = histogram(initial_nodes)
    diff_before = {
        "r2": 0.41,
        "source_stage": "layer2",
        "spectral_contract": "template_center_mu",
        "marker": "baseline",
    }
    candidate = {
        "candidate_signature": "side:test-pending-ring",
        "selected_su24_ids": [100, 101],
        "product_kind": "fatty_ring_side",
        "skeleton_kind": "side",
        "required_counts": {11: 2, 23: 4},
        "converted_node_ids": [1],
    }
    planner_calls = []

    class FakeAdjuster:
        _stage3_su11_rotation_index = 0
        _stage3_parity_balance_cursor = 0

        @staticmethod
        def _get_special_degree_meta(_H):
            return None

        def plan_topology_resource_adjustments(self, **kwargs):
            planner_calls.append(kwargs)
            if len(planner_calls) == 1:
                return _su24_conversion_plan(kwargs["H"], candidate)
            assert "pending_su24_ring_candidate" not in kwargs
            if len(planner_calls) == 2:
                assert kwargs["progress_phase"] is Stage3ProgressPhase.EXTERNAL_TOPOLOGY
                assert kwargs["external_primary_adjustments_used"] == 1
            else:
                assert kwargs["progress_phase"] is Stage3ProgressPhase.FLEX_RECONCILIATION
                assert kwargs["external_primary_adjustments_used"] == 0
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                plan_kind="stage3_state_machine_complete",
                phase_result=Stage3PhaseResult(
                    active_phase="complete",
                    phase_completed=True,
                    rebuild_required=False,
                ),
            )

        @staticmethod
        def apply_topology_resource_adjustments(*, nodes, H, plan):
            histogram_plan = plan.histogram_adjustment_plan
            if histogram_plan is None:
                return SimpleNamespace(
                    changed=False,
                    nodes=list(nodes),
                    H=H.detach().clone(),
                    requested_counts={},
                    unresolved_counts={},
                    applied_ops=[],
                    request_ops=[],
                    rejected_ops=[],
                )
            converted_nodes = [
                make_node(int(node.global_id), 22 if int(node.global_id) == 1 else int(node.su_type))
                for node in nodes
            ]
            return SimpleNamespace(
                changed=True,
                nodes=converted_nodes,
                H=histogram_plan.H_after.detach().clone(),
                requested_counts={},
                unresolved_counts={},
                applied_ops=[{"node_id": 1, "from_su": 23, "to_su": 22}],
                request_ops=[],
                rejected_ops=[],
            )

    result = run_block_c_resource_adjustment_loop_impl(
        adjuster=FakeAdjuster(),
        H=H_before,
        nodes=initial_nodes,
        E_target=None,
        difference_spectrum=diff_before,
        enable_post_external_spectrum=False,
        rebuild_fn=lambda **kwargs: (
            list(kwargs["nodes_seed"]),
            kwargs["H_seed"].detach().clone(),
            {
                **diff_before,
                "r2": 0.52,
                "marker": "converted-candidate",
            },
            0.52,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
    )

    assert result.ok is True
    assert result.accepted_cycles == 1
    assert int(result.H[23].item()) == int(H_before[23].item()) - 1
    assert int(result.H[22].item()) == int(H_before[22].item()) + 1
    assert [(int(node.global_id), int(node.su_type)) for node in result.nodes] == [
        (1, 22),
        (2, 22),
        (100, 24),
        (101, 24),
    ]
    assert result.diff["marker"] == "converted-candidate"
    assert len(planner_calls) == 3
    assert planner_calls[0]["generation_id"] == planner_calls[1]["generation_id"]
    assert planner_calls[2]["generation_id"] == planner_calls[1]["generation_id"] + 1
    assert result.cycles[0].accepted is True
    assert result.cycles[0].acceptance_reason == "external_conversion_applied"
    assert sum(1 for cycle in result.cycles if cycle.fresh_rebuild) == 1


def test_residual_side_chain_converts_one_su22_to_su11() -> None:
    nodes = [
        make_node(1, 22),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    plan = plan_stage3_resource_adjustments(
        H=histogram(nodes),
        nodes=nodes,
        enable_post_external_spectrum=False,
    )

    groups = plan.histogram_adjustment_plan.groups
    assert plan.reason == "stage3_rebuild_required"
    assert [(group.primary_op.from_su, group.primary_op.to_su) for group in groups] == [
        (22, 11)
    ]
    assert groups[0].primary_op.node_ids == [1]
    transaction = plan.phase_result.transactions[0]
    assert transaction.claimed_resource_ids == {11: [1], 22: [2], 23: [3, 4]}


def test_read_only_stage3_audit_preserves_real_resource_ids() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 22),
        make_node(3, 23),
        make_node(4, 23),
    ]
    H = histogram(nodes)
    audit = audit_stage3_topology(
        H=H,
        nodes=nodes,
        allow_conversions=False,
        enable_spectrum_rebalance=False,
    )

    assert audit["ok"] is True
    assert audit["products"][0]["topology_notation"] == "11-23-23-22"
    accounting = audit["resource_accounting"]
    aggregate_ids = {
        int(node_id)
        for values in accounting["aggregate_ids_after_by_su"].values()
        for node_id in values
    }
    partition_ids = {
        int(node_id)
        for field in ("final_remaining_ids_by_su", "committed_claimed_ids_by_su")
        for values in accounting[field].values()
        for node_id in values
    }
    assert aggregate_ids == partition_ids == {1, 2, 3, 4}


def test_resource_accounting_rejects_duplicate_product_ownership() -> None:
    snapshot = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={11: [1]},
        authorized_ids_before_by_su={},
        final_remaining_ids_by_su={},
        committed_claimed_ids_by_su={11: [1, 1]},
        groups=[],
    )

    assert snapshot["ok"] is False
    assert any(
        item["reason"] == "resource_claimed_by_multiple_products"
        for item in snapshot["errors"]
    )


def test_stage3_product_merge_deduplicates_same_concrete_product() -> None:
    original = {
        "product_id": "generation:ring:18",
        "product_type": "aliphatic_ring",
        "fixed_source_node_ids": [179, 182],
        "claimed_resource_ids_by_su": {
            "22": [212, 220],
            "23": [270, 271, 275, 276],
            "24": [483, 484],
        },
    }
    duplicate_projection = {
        **original,
        "product_id": "generation:ring:161",
        "topology_notation": "11-24-23-23-24-11",
    }

    merged = block_c_module._merge_stage3_committed_products(
        [original],
        [duplicate_projection],
    )

    assert len(merged) == 1
    assert merged[0]["product_id"] == "generation:ring:161"
    assert merged[0]["topology_notation"] == "11-24-23-23-24-11"


def test_resource_accounting_allows_ordered_reconversion_of_one_real_node() -> None:
    first = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(
            from_su=22,
            to_su=23,
            node_ids=[424],
            reason="test_chain_step_1",
        ),
        combined_delta={22: -1, 23: 1},
    )
    second = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(
            from_su=23,
            to_su=11,
            node_ids=[424],
            reason="test_chain_step_2",
        ),
        combined_delta={23: -1, 11: 1},
    )
    snapshot = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={22: [424], 23: [500]},
        authorized_ids_before_by_su={22: [424], 23: [500]},
        final_remaining_ids_by_su={11: [424], 23: [500]},
        committed_claimed_ids_by_su={},
        groups=[first, second],
    )

    assert snapshot["ok"] is True
    assert snapshot["aggregate_ids_after_by_su"] == {11: [424], 23: [500]}


def test_resource_accounting_rejects_discontinuous_reconversion_chain() -> None:
    first = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(from_su=22, to_su=23, node_ids=[424]),
        combined_delta={22: -1, 23: 1},
    )
    invalid_second = AtomicAdjustmentGroup(
        primary_op=HistogramConversion(from_su=22, to_su=11, node_ids=[424]),
        combined_delta={22: -1, 11: 1},
    )
    snapshot = build_resource_accounting_snapshot(
        aggregate_ids_before_by_su={22: [424]},
        authorized_ids_before_by_su={22: [424]},
        final_remaining_ids_by_su={11: [424]},
        committed_claimed_ids_by_su={},
        groups=[first, invalid_second],
    )

    assert snapshot["ok"] is False
    assert any(
        item["reason"] == "conversion_donor_not_authorized"
        and item["node_id"] == 424
        and item["authorized_su"] == 23
        for item in snapshot["errors"]
    )


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return str(node.func.attr)
    if isinstance(node.func, ast.Name):
        return str(node.func.id)
    return ""


def test_pipeline_runs_one_block_c_between_layer12_and_layer3() -> None:
    source = textwrap.dedent(inspect.getsource(InversePipelineV3.infer))
    tree = ast.parse(source)
    outer_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "outer_idx"
    )
    calls = [
        (_call_name(node), int(node.lineno))
        for node in ast.walk(outer_loop)
        if isinstance(node, ast.Call)
    ]
    block_c = [line for name, line in calls if name == "_run_topology_stage3_resource_cycles"]
    layer12 = [line for name, line in calls if name == "_run_layer12_cycle"]
    layer3 = [line for name, line in calls if name == "layer3_adjust_templates"]

    assert len(block_c) == 1
    assert max(layer12) < block_c[0] < max(layer3)
