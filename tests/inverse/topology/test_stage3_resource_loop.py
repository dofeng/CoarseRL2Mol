from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from model.inverse.layer4.inverse_layer4 import Layer4Adjuster
import model.inverse.layer4.inverse_layer4 as inverse_layer4_module
from model.inverse.layer4.inverse_layer4_block_c import run_block_c_resource_adjustment_loop_impl
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.histogram_adjustment import _h_total, apply_histogram_adjustment_plan
from model.inverse.topology.iteration_ledger import build_iteration_ledger
from model.inverse.topology.stage3_models import (
    AtomicAdjustmentGroup,
    HistogramAdjustmentPlan,
    HistogramConversion,
    Stage3AdjustmentBundle,
    Stage3AdjustmentPlan,
    Stage3PhaseResult,
    TopologyClosureContract,
    TopologyClosureSlot,
    TopologyResourceTransaction,
)
from model.inverse.topology.stage3_state_machine import (
    _best_residual_template_fit,
    _residual_contracts,
    _transaction_plan,
    build_su24_resource_role_profiles,
    plan_stage3_state_machine,
)
from model.inverse.topology.phase_eval import (
    PHASE_SEQUENCE,
    build_stage3_phase_evaluation,
    classify_topology_units,
)
from model.inverse.topology.workflow import plan_stage3_resource_adjustments

from .helpers import make_node


def _histogram(nodes):
    H = torch.zeros(40, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1
    return H


def test_stage3_default_budget_and_spectrum_limits_are_consistent() -> None:
    layer4_defaults = inspect.signature(
        Layer4Adjuster.run_block_c_resource_adjustment_loop
    ).parameters
    assert layer4_defaults["max_stage3_iterations"].default == 20
    assert layer4_defaults["spectrum_candidate_limit"].default == 10
    assert layer4_defaults["spectrum_max_primary_adjustments"].default == 8

    for function in (plan_stage3_resource_adjustments, plan_stage3_state_machine):
        defaults = inspect.signature(function).parameters
        assert defaults["spectrum_candidate_limit"].default == 10
        assert defaults["spectrum_max_primary_adjustments"].default == 8


def test_layer4_wrapper_forwards_block_c_resource_configuration(monkeypatch) -> None:
    captured = {}

    def fake_impl(adjuster, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(inverse_layer4_module, "run_block_c_resource_adjustment_loop_impl", fake_impl)
    Layer4Adjuster().run_block_c_resource_adjustment_loop(
        H=torch.zeros(40, dtype=torch.long),
        nodes=[],
        E_target=torch.zeros(6),
        max_stage3_iterations=16,
        max_primary_adjustments_per_stage=6,
        max_h_compensation_ops_per_adjustment=8,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        h_tolerance=0.06,
        iteration_offset=10,
    )
    assert captured["max_stage3_iterations"] == 16
    assert captured["max_primary_adjustments_per_stage"] == 6
    assert captured["max_h_compensation_ops_per_adjustment"] == 8
    assert captured["residual_flex_preferred_23"] == 3
    assert captured["residual_flex_min_23"] == 2
    assert captured["residual_side_preferred_23"] == 2
    assert captured["residual_side_min_23"] == 1
    assert captured["h_tolerance"] == 0.06
    assert captured["iteration_offset"] == 10


def test_layer4_stage_router_forwards_complete_block_c_configuration(monkeypatch) -> None:
    captured = {}

    def fake_impl(adjuster, H, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            H=H,
            cycles=[],
            to_dict=lambda: {},
        )

    monkeypatch.setattr(inverse_layer4_module, "run_block_c_resource_adjustment_loop_impl", fake_impl)
    Layer4Adjuster().adjust_by_stage(
        H=torch.zeros(40, dtype=torch.long),
        ppm=None,
        diff=None,
        E_target=torch.zeros(6),
        stage="block_c_resource",
        nodes=[],
        max_h_compensation_ops_per_adjustment=7,
        residual_flex_preferred_23=4,
        residual_flex_min_23=3,
        residual_side_preferred_23=3,
        residual_side_min_23=2,
        h_tolerance=0.05,
        outer_cycle=6,
        iteration_offset=9,
    )

    assert captured["max_h_compensation_ops_per_adjustment"] == 7
    assert captured["residual_flex_preferred_23"] == 4
    assert captured["residual_flex_min_23"] == 3
    assert captured["residual_side_preferred_23"] == 3
    assert captured["residual_side_min_23"] == 2
    assert captured["h_tolerance"] == 0.05
    assert captured["outer_cycle"] == 6
    assert captured["iteration_offset"] == 9


def test_layer4_does_not_forward_removed_block_b_tail_configuration(monkeypatch) -> None:
    del monkeypatch
    source = inspect.getsource(inverse_layer4_module.adjust_block_b_hetero_anchor_impl)
    assert "tail_spectrum" not in source


def test_stage3_phase_sequence_is_complete_and_ordered() -> None:
    assert PHASE_SEQUENCE == (
        "flex_range_reconciliation",
        "external_flex_closure",
        "external_fatty_ring_closure",
        "external_side_chain_closure",
        "remaining_pool_spectrum_rebalance",
        "resource_su24_ring_closure",
        "residual_resource_self_closure",
        "complete",
    )


def test_residual_side_contracts_wait_for_remaining_flex_target() -> None:
    pool = {11: [1], 12: [], 13: [], 22: [2], 23: [3, 4], 24: []}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)

    direct = _residual_contracts(pool, max_flex_chains=1)
    fitted = _best_residual_template_fit(
        pool=pool,
        H=H,
        E_target=None,
        h_tolerance=0.06,
        special_degree_meta=None,
        su11_rotation_index=0,
        remaining_flex_chains=1,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        max_conversions=4,
        max_h_compensation_ops=4,
    )

    assert direct == []
    assert fitted is not None
    assert str(fitted["product_role"]) == "residual_flex_chain"


def test_symbolic_product_operation_is_not_applied_to_node_graph() -> None:
    nodes = [make_node(1, 19), make_node(2, 23), make_node(3, 11)]
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        adjustment_bundles=[Stage3AdjustmentBundle(bundle_ops=[{
            "op": "symbolic_topology_product",
            "unit_id": 1,
            "target_role": "flex_chain",
            "source_node_ids": [1],
            "claimed_resource_ids_by_su": {"23": [2], "11": [3]},
        }])],
    )
    result = apply_stage3_node_adjustments(nodes, _histogram(nodes), plan)
    assert result.changed is False
    assert not any(item.get("op") == "symbolic_topology_product" for item in result.applied_ops)
    assert all(not list(getattr(node, "hop1_ids", []) or []) for node in result.nodes)
    assert all(not set(getattr(node, "fixed_hop1_ids", set()) or set()) for node in result.nodes)
    assert all(getattr(node, "stage3_topology_owner_id", None) is None for node in result.nodes)


def test_histogram_adjustment_cannot_consume_fixed_global_resource() -> None:
    nodes = [make_node(index, 24) for index in range(1, 6)]
    H = _histogram(nodes)
    plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(24, 23, 1, "test"),
            combined_delta={24: -1, 23: 1},
        )],
        inventory_before={24: 0, 23: 0},
    )
    result = apply_histogram_adjustment_plan(H, plan, nodes=nodes)
    assert result.changed is False
    assert result.rejected_ops[0]["reason"] == "donor_not_in_authorized_stage3_pool"


def test_residual_su24_resources_are_rebalanced_into_complete_side_requirements() -> None:
    nodes = [make_node(7, 24), make_node(3, 24), make_node(9, 24)]
    plan = plan_stage3_resource_adjustments(H=_histogram(nodes), nodes=nodes)
    groups = plan.histogram_adjustment_plan.groups
    assert len(groups) == 3
    conversions = [group.primary_op for group in groups]
    assert all(conversion.from_su == 24 for conversion in conversions)
    assert [conversion.to_su for conversion in conversions] == [23, 23, 23]
    assert [conversion.node_ids[0] for conversion in conversions] == [3, 7, 9]
    assert plan.phase_result.primary_adjustment_count == 3
    assert plan.phase_result.primary_adjustment_count <= 10


def test_residual_template_fit_accepts_only_final_h_of_atomic_bundle() -> None:
    pool = {11: [1, 2], 12: [], 13: [], 22: [], 23: [3, 4, 5], 24: [6]}
    H = torch.zeros(40, dtype=torch.long)
    for su_type, node_ids in pool.items():
        H[int(su_type)] = len(node_ids)
    E_target = torch.tensor([6, 7, 0, 0, 0, 0], dtype=torch.float32)

    fitted = _best_residual_template_fit(
        pool=pool,
        H=H,
        E_target=E_target,
        h_tolerance=0.0,
        special_degree_meta=None,
        su11_rotation_index=0,
        remaining_flex_chains=0,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        max_conversions=4,
        max_h_compensation_ops=4,
    )

    assert fitted is not None
    conversions = list(fitted["conversions"])
    assert [(item.from_su, item.to_su) for item in conversions] == [(23, 22), (24, 11)]
    assert float(fitted["h_after"]) == 7.0


def test_ym9_post_rebuild_pool_has_executable_atomic_side_fit() -> None:
    H = torch.zeros(40, dtype=torch.long)
    for su_type, count in {
        0: 1, 1: 10, 2: 11, 3: 12, 5: 64, 6: 2, 9: 23, 10: 30,
        11: 115, 14: 1, 15: 8, 16: 3, 17: 2, 18: 2, 19: 105,
        20: 2, 22: 18, 23: 87, 24: 3, 25: 1, 26: 3, 27: 2,
        28: 50, 29: 61,
    }.items():
        H[int(su_type)] = int(count)
    pool = {
        11: list(range(240, 268)),
        12: [],
        13: [],
        22: [],
        23: list(range(460, 475)),
        24: [],
    }

    fitted = _best_residual_template_fit(
        pool=pool,
        H=H,
        E_target=torch.tensor([500, 520, 166, 6, 0, 0], dtype=torch.float32),
        h_tolerance=0.06,
        special_degree_meta=None,
        su11_rotation_index=0,
        remaining_flex_chains=0,
        residual_flex_preferred_23=3,
        residual_flex_min_23=2,
        residual_side_preferred_23=2,
        residual_side_min_23=1,
        max_conversions=12,
        max_h_compensation_ops=8,
    )

    assert fitted is not None
    assert fitted["required_counts"] == {11: 4, 22: 4, 23: 6}
    assert int(fitted["product_count"]) == 4
    assert [(item.from_su, item.to_su) for item in fitted["conversions"]] == (
        [(23, 22)] * 4 + [(23, 11)] * 5
    )
    assert float(fitted["h_after"]) == 551.0


def test_stage3_transaction_records_effective_h_not_histogram_node_count() -> None:
    nodes = [make_node(1, 22), make_node(7, 24), make_node(3, 24), make_node(9, 24)]
    H = _histogram(nodes)
    plan = plan_stage3_resource_adjustments(H=H, nodes=nodes)

    assert plan.phase_result.transactions
    first = plan.phase_result.transactions[0]
    assert first.h_before == _h_total(H, None)
    assert first.h_before != float(H.sum().item())
    assert first.h_after == _h_total(plan.histogram_adjustment_plan.H_after, None)


def test_su24_profile_does_not_claim_or_consume_resource() -> None:
    nodes = [make_node(4, 24), make_node(2, 24)]
    pool = {24: [2, 4]}
    profiles = build_su24_resource_role_profiles(nodes, pool)
    assert [item.node_id for item in profiles] == [2, 4]
    assert pool == {24: [2, 4]}
    assert all(item.roles for item in profiles)


def test_block_c_stops_with_specific_phase_resource_infeasible() -> None:
    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason="current_contract_resource_infeasible",
                phase="external_fatty_ring_closure",
                unresolved_counts={"24": 1},
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        max_stage3_iterations=2,
    )
    assert result.stop_reason == "phase_resource_infeasible"
    assert result.accepted_cycles == 0
    proof = result.cycles[-1].plan["phase_result"]["infeasibility_proof"]
    assert proof["phase"] == "external_fatty_ring_closure"
    assert proof["reason"] == "current_contract_resource_infeasible"
    assert proof["missing_counts"] == {"24": 1}
    assert "failed_contract" in proof
    assert "attempted_conversions" in proof
    assert "h_or_element_rejections" in proof
    assert "template_checks" in proof


def test_zero_conversion_fallback_cannot_commit_without_iteration_ledger() -> None:
    nodes = [
        make_node(1, 11),
        make_node(2, 11),
        make_node(3, 23),
        make_node(4, 23),
        make_node(5, 23),
    ]
    H = _histogram(nodes)
    pool = {11: [1, 2], 22: [], 23: [3, 4, 5], 24: []}
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    contract = _residual_contracts(pool, max_flex_chains=1)[0]

    plan = _transaction_plan(
        phase_eval=phase_eval,
        contract=contract,
        pool=pool,
        H=H,
        E_target=None,
        max_primary_adjustments=10,
        max_h_compensation_ops=8,
        h_tolerance=0.06,
        special_degree_meta=None,
        iteration_ledger=None,
    )

    transaction = plan.phase_result.transactions[-1]
    assert plan.reason == "phase_resource_infeasible"
    assert plan.phase_result.symbolic_topology_count == 0
    assert transaction.committed is False
    assert transaction.claimed_resource_ids == {}
    assert transaction.product_node_ids == []
    assert transaction.infeasibility_proof["reason"] == (
        "zero_conversion_transaction_must_use_iteration_ledger"
    )
    assert transaction.infeasibility_proof["attempted_claimed_resource_ids"] == {
        11: [1, 2],
        23: [3, 4, 5],
    }


def test_transaction_plan_never_reintroduces_ids_absent_from_ledger_pool() -> None:
    nodes = [make_node(1, 11), make_node(2, 11)]
    H = _histogram(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classify_topology_units(nodes),
    )
    ledger = build_iteration_ledger(nodes, resource_ids_by_su={11: [1]})
    contract = TopologyClosureContract(
        contract_id=77,
        contract_kind="residual_conversion",
        phase="residual_resource_self_closure",
        target_role="residual_resource_conversion",
        mainline_slots=[TopologyClosureSlot(
            slot_id="residual:77:mainline",
            kind="RESOURCE_PATH",
            required_su_sequence=[11, 11],
        )],
        required_counts={11: 2},
        expected_topology="11-11",
    )

    plan = _transaction_plan(
        phase_eval=phase_eval,
        contract=contract,
        pool={11: [1, 2]},
        H=H,
        E_target=None,
        max_primary_adjustments=0,
        max_h_compensation_ops=0,
        h_tolerance=0.06,
        special_degree_meta=None,
        iteration_ledger=ledger,
    )

    proof = plan.phase_result.infeasibility_proof
    transaction = plan.phase_result.transactions[-1]
    assert plan.reason == "phase_resource_infeasible"
    assert proof["available_counts"][11] == 1
    assert proof["slot_assignment_proof"]["reason"] == "slot_resource_deficit"
    assert transaction.committed is False
    assert transaction.claimed_resource_ids == {}
    assert 2 not in plan.runtime["iteration_topology_ledger"].available_ids_by_su[11]


def test_block_c_iteration_offset_keeps_post_rebuild_audit_ids_unique() -> None:
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=99,
        target_role="residual_resource_conversion",
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(transactions=[transaction]),
            )

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        max_stage3_iterations=1,
        outer_cycle=4,
        iteration_offset=10,
    )

    exported = result.cycles[0].plan["phase_result"]["transactions"][0]
    assert result.cycles[0].cycle == 11
    assert exported["outer_cycle"] == 4
    assert exported["iteration"] == 11


def test_block_c_generation_id_is_unique_across_outer_cycles() -> None:
    captured = []

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            captured.append(int(kwargs["generation_id"]))
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_complete",
                phase="complete",
                phase_result=Stage3PhaseResult(stop_reason="stage3_complete"),
            )

    run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        torch.zeros(40, dtype=torch.long),
        [],
        E_target=None,
        max_stage3_iterations=1,
        outer_cycle=4,
        iteration_offset=10,
    )

    assert captured == [4_000_011]


def test_block_c_final_read_only_audit_promotes_complete_after_last_rebuild() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            if int(kwargs["H"][11].item()) == 1:
                return Stage3AdjustmentPlan(
                    ok=True,
                    reason="stage3_complete",
                    phase="complete",
                    phase_result=Stage3PhaseResult(
                        active_phase="complete",
                        phase_completed=True,
                        stop_reason="stage3_complete",
                    ),
                )
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                histogram_adjustment_plan=histogram_plan,
                phase_result=Stage3PhaseResult(rebuild_required=True),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            kwargs["identity_seed_nodes"],
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
        max_stage3_iterations=1,
    )

    assert result.ok is True
    assert result.stop_reason == "stage3_complete"
    assert result.accepted_cycles == 1
    assert len(result.cycles) == 2
    assert result.cycles[-1].applied is False
    assert result.cycles[-1].acceptance_reason == "final_read_only_audit_complete"


def test_block_c_rolls_back_valid_rebuild_without_topology_progress() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    before_metrics = {
        "subphase_rank": 5,
        "pending_count": 0,
        "unresolved_total": 0,
        "shortage_total": 0,
        "flex_gap": 1,
        "product_count": 5,
    }
    after_metrics = {
        **before_metrics,
        "flex_gap": 2,
        "product_count": 4,
    }
    histogram_plan = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
        su11_rotation_index_before=4,
        su11_rotation_index_after=5,
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=7,
        target_role="residual_resource_conversion",
        converted_node_ids=[7],
        conversion_path=[{"from_su": 12, "to_su": 11, "node_ids": [7]}],
        committed=True,
    )
    before_plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        histogram_adjustment_plan=histogram_plan,
        phase_result=Stage3PhaseResult(transactions=[transaction], rebuild_required=True),
        runtime={"phase_evaluation": SimpleNamespace(topology_metrics=before_metrics)},
    )
    after_plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        runtime={"phase_evaluation": SimpleNamespace(topology_metrics=after_metrics)},
    )

    class Adjuster:
        _stage3_su11_rotation_index = 4

        def plan_topology_resource_adjustments(self, **kwargs):
            return after_plan if int(kwargs["H"][11].item()) == 1 else before_plan

        def apply_topology_resource_adjustments(self, **kwargs):
            self._stage3_su11_rotation_index = 5
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    context = {"generation": "before"}

    def rebuild(**kwargs):
        context["generation"] = "candidate"
        return kwargs["identity_seed_nodes"], kwargs["H_seed"], {"r2": 0.7}, 0.7

    def restore(snapshot):
        context.clear()
        context.update(snapshot)

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        rebuild_fn=rebuild,
        validate_fn=lambda **_kwargs: (True, "ok"),
        context_snapshot_fn=lambda: dict(context),
        context_restore_fn=restore,
        max_stage3_iterations=2,
    )

    assert result.stop_reason == "phase_resource_infeasible"
    assert result.accepted_cycles == 0
    assert int(result.H[12].item()) == 1
    assert int(result.H[11].item()) == 0
    assert int(result.nodes[0].su_type) == 12
    assert context == {"generation": "before"}
    assert adjuster._stage3_su11_rotation_index == 4
    cycle = result.cycles[0]
    assert cycle.accepted is False
    assert cycle.accepted_state_retained is True
    assert cycle.blocked_reason == "topology_progress_not_improved"
    proof = cycle.plan["phase_result"]["infeasibility_proof"]
    assert proof["reason"] == "topology_progress_not_improved"
    exported = cycle.plan["phase_result"]["transactions"][0]
    assert exported["committed"] is False
    assert exported["rollback_reason"] == "topology_progress_not_improved"


def test_block_c_iteration_budget_exhaustion_exports_infeasibility_proof() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 2},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **_kwargs):
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                plan_kind="external_flex_resource_rebalance",
                hard_required_counts={"11": 2},
                planned_conversions=[
                    {"op": "convert", "from_su": 12, "to_su": 11, "node_id": 7}
                ],
                histogram_adjustment_plan=histogram_plan,
                phase_result=Stage3PhaseResult(rebuild_required=True),
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (
            kwargs["identity_seed_nodes"],
            kwargs["H_seed"],
            {},
            0.0,
        ),
        validate_fn=lambda **_kwargs: (True, "ok"),
        max_stage3_iterations=1,
    )

    assert result.ok is False
    assert result.stop_reason == "phase_resource_infeasible"
    assert result.accepted_cycles == 1
    assert len(result.cycles) == 2
    terminal = result.cycles[-1]
    proof = terminal.plan["phase_result"]["infeasibility_proof"]
    assert terminal.applied is False
    assert terminal.blocked_reason == "stage3_iteration_budget_exhausted"
    assert proof["reason"] == "stage3_iteration_budget_exhausted"
    assert proof["phase"] == "external_flex_closure"
    assert proof["required_counts"] == {"11": 2}
    assert proof["available_counts"] == {"11": 0, "12": 1}
    assert proof["missing_counts"] == {"11": 2}
    assert proof["attempted_conversions"][0]["node_id"] == 7
    assert proof["max_stage3_iterations"] == 1


def test_block_c_does_not_rebuild_an_attached_empty_histogram_plan() -> None:
    node = make_node(7, 22)
    H = _histogram([node])
    empty_histogram = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        H_before=H.detach().clone(),
        H_after=H.detach().clone(),
        blocked_reason="hydrogen_soft_direction_incompatible_with_resource_demand",
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=False,
                reason="phase_resource_infeasible",
                blocked_reason="hydrogen_soft_direction_incompatible_with_resource_demand",
                phase="residual_resource_self_closure",
                planned_conversions=[{"from_su": 22, "to_su": 11, "reason": "blocked"}],
                histogram_adjustment_plan=empty_histogram,
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    rebuild_calls = []
    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=torch.tensor([0.0, 100.0]),
        rebuild_fn=lambda **kwargs: rebuild_calls.append(kwargs),
        validate_fn=lambda **kwargs: (True, "ok"),
        max_stage3_iterations=1,
    )

    assert rebuild_calls == []
    assert result.accepted_cycles == 0
    assert result.stop_reason == "phase_resource_infeasible"


def test_block_c_rejects_cold_rebuild_histogram_drift() -> None:
    node = make_node(7, 12)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[12] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="external_flex_closure",
        inventory_before={12: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(12, 11, 1, "test", node_ids=[7]),
            combined_delta={12: -1, 11: 1},
        )],
    )

    class Adjuster:
        def plan_topology_resource_adjustments(self, **kwargs):
            return Stage3AdjustmentPlan(
                ok=True,
                reason="stage3_rebuild_required",
                phase="external_flex_closure",
                histogram_adjustment_plan=histogram_plan,
            )

        def apply_topology_resource_adjustments(self, **kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    validation_calls = []
    result = run_block_c_resource_adjustment_loop_impl(
        Adjuster(),
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (kwargs["identity_seed_nodes"], H.detach().clone(), {}, 0.0),
        validate_fn=lambda **kwargs: validation_calls.append(kwargs) or (True, "ok"),
        max_stage3_iterations=1,
    )

    assert validation_calls == []
    assert result.accepted_cycles == 0
    assert result.stop_reason == "all_histogram_adjustment_variants_invalid"
    assert result.cycles[0].blocked_reason == "cold_rebuild_histogram_mismatch"


def test_block_c_exports_invalid_cold_rebuild_transactions_as_rolled_back() -> None:
    node = make_node(7, 13)
    H = _histogram([node])
    H_after = H.detach().clone()
    H_after[13] = 0
    H_after[11] = 1
    histogram_plan = HistogramAdjustmentPlan(
        phase="residual_resource_self_closure",
        inventory_before={13: 1, 11: 0},
        demand_before={11: 1},
        H_before=H.detach().clone(),
        H_after=H_after,
        primary_adjustment_count=1,
        physical_conversion_count=1,
        groups=[AtomicAdjustmentGroup(
            primary_op=HistogramConversion(13, 11, 1, "test", node_ids=[7]),
            combined_delta={13: -1, 11: 1},
        )],
        su11_rotation_index_before=4,
        su11_rotation_index_after=5,
    )
    transaction = TopologyResourceTransaction(
        phase="residual_resource_self_closure",
        contract_id=3_000_007,
        target_role="residual_resource_conversion",
        converted_node_ids=[7],
        conversion_path=[{"from_su": 13, "to_su": 11, "node_ids": [7]}],
        claimed_resource_ids={11: [7]},
        product_node_ids=[7],
        commit_scope="physical_histogram",
        committed=True,
    )
    plan = Stage3AdjustmentPlan(
        ok=True,
        reason="stage3_rebuild_required",
        phase="residual_resource_self_closure",
        histogram_adjustment_plan=histogram_plan,
        diagnostics={"su11_rotation_index_after": 5},
        phase_result=Stage3PhaseResult(transactions=[transaction]),
    )

    class Adjuster:
        _stage3_su11_rotation_index = 4

        def plan_topology_resource_adjustments(self, **kwargs):
            return plan

        def apply_topology_resource_adjustments(self, **kwargs):
            self._stage3_su11_rotation_index = 5
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    adjuster = Adjuster()
    result = run_block_c_resource_adjustment_loop_impl(
        adjuster,
        H,
        [node],
        E_target=None,
        rebuild_fn=lambda **kwargs: (kwargs["identity_seed_nodes"], H.detach().clone(), {}, 0.0),
        validate_fn=lambda **kwargs: (True, "ok"),
        max_stage3_iterations=1,
    )

    exported = result.cycles[0].plan["phase_result"]["transactions"][0]
    assert exported["committed"] is False
    assert exported["rollback_reason"] == "cold_rebuild_histogram_mismatch"
    assert exported["converted_node_ids"] == []
    assert exported["claimed_resource_ids"] == {}
    attempted = exported["infeasibility_proof"]["attempted_transaction"]
    assert attempted["converted_node_ids"] == [7]
    assert attempted["claimed_resource_ids"] == {"11": [7]}
    assert adjuster._stage3_su11_rotation_index == 4


def test_user_supply_demand_example_converts_lowest_su24_to_su23_first() -> None:
    nodes = [
        make_node(10, 11),
        make_node(20, 23), make_node(21, 23), make_node(22, 23),
        make_node(30, 22), make_node(31, 22), make_node(32, 22),
        make_node(40, 24),
    ]
    plan = Stage3AdjustmentPlan(
        phase="external_flex_closure",
        hard_required_counts={"11": 4, "23": 7, "22": 3, "24": 0},
        runtime={
            "assembly_before_adjustment": {
                "resource_ledger": {
                    "remaining_ids_by_su": {
                        "11": [10], "23": [20, 21, 22], "22": [30, 31, 32], "24": [40],
                    },
                    "remaining_counts": {"11": 1, "23": 3, "22": 3, "24": 1},
                },
            },
        },
    )
    from model.inverse.topology.histogram_adjustment import build_histogram_adjustment_plan

    histogram_plan = build_histogram_adjustment_plan(
        plan=plan,
        H=_histogram(nodes),
        max_primary_adjustments=1,
    )
    assert histogram_plan.primary_adjustment_count == 1
    primary = histogram_plan.groups[0].primary_op
    assert (primary.from_su, primary.to_su) == (24, 23)
    assert histogram_plan.groups[0].demand_after == {11: 3, 22: 0, 23: 3, 24: 0}
    assert histogram_plan.physical_conversion_count >= histogram_plan.primary_adjustment_count
