from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

warnings.filterwarnings(
    "ignore",
    message=r"`torch_geometric\.distributed` has been deprecated.*",
    category=DeprecationWarning,
)

from model.inverse.topology.classification.classifier import classify_topology_units
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.assembly_external import (
    AssemblyFragmentView,
    BranchStub,
    ExternalTopologyAssembler,
    _validate_ring_product_shape,
    assemble_external_topologies,
)
from model.inverse.topology.phase_eval import assemble_topology_units, build_stage3_phase_evaluation
from model.inverse.topology.stage3_models import merge_stage3_contracts_by_owner
from model.inverse.topology.stage3_models import TopologyAssemblyProduct, TopologyPathRender, TopologyPoolLedger
from model.inverse.topology.workflow import plan_stage3_resource_adjustments
from model.inverse.inverse_pipeline import InversePipelineV3

from .assembly_report import build_external_assembly_audit_report
from .csv_fixture_loader import load_nodes_from_final_csv
from .helpers import connect, make_node


CSV_PATH = "/home/dofengqi/Graphnmr/test_results/YM-4 copy/final_outputs/final_nodes_by_global_id.csv"

PRODUCT_KEYS = {
    "product_kind",
    "product_type",
    "actual_topology_su",
    "actual_topology_gid",
    "source_fragment_ids",
    "source_fragment_classes",
    "source_node_ids",
    "materialized_node_ids",
    "mainline_node_ids",
    "branch_node_ids",
    "claimed_resource_ids",
    "slot_assignments",
    "edge_plan",
    "closed",
    "reason",
}


def test_resource_ownership_reconciliation_does_not_inflate_convertible_pool() -> None:
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.nodes = [make_node(1, 11), make_node(2, 11)]
    assembler.nodes_by_id = {int(node.global_id): node for node in assembler.nodes}
    assembler.products = []
    assembler.classification = SimpleNamespace(fragments=[])
    ledger = TopologyPoolLedger(
        remaining_ids_by_su={"11": [1], "10": [], "12": [], "13": [], "22": [], "23": [], "24": []},
    )

    reconciled = assembler._reconcile_resource_ownership(ledger)

    assert reconciled.diagnostic_remaining_ids_by_su["11"] == [1, 2]
    assert reconciled.remaining_ids_by_su["11"] == [1]
    assert reconciled.convertible_ids_by_su["11"] == [1]
    assert reconciled.convertible_counts["11"] == 1


def test_contract_merge_rejects_conflicting_fragment_owner_definitions() -> None:
    base = {
        "contract_id": 10,
        "contract_kind": "branch_closure",
        "target_role": "flex_chain",
        "source_fragment_ids": [7, 3],
        "claimed_source_node_ids": [101, 102],
        "required_counts": {"11": 1, "22": 1},
    }

    with pytest.raises(ValueError, match="duplicate topology contract definitions"):
        merge_stage3_contracts_by_owner([base, {**base, "required_counts": {"11": 2, "23": 1}}])


def test_pipeline_shares_layer2_template_cache_with_layer3() -> None:
    pipeline = object.__new__(InversePipelineV3)
    shared_cache = {"templates": {"key": "value"}, "center_index": {}}
    pipeline.default_template_lib_path = "/tmp/templates.pt"
    pipeline.layer2_estimator = SimpleNamespace(
        lib_path="/tmp/templates.pt",
        _template_cache=shared_cache,
    )
    pipeline.layer3_estimator = SimpleNamespace(
        lib_path="/tmp/templates.pt",
        _template_cache=None,
    )

    assert pipeline._share_layer2_template_cache_with_layer3("/tmp/templates.pt") is True
    assert pipeline.layer3_estimator._template_cache is shared_cache


def test_layer3_impl_reuses_layer2_cache_before_estimator_call() -> None:
    pipeline = object.__new__(InversePipelineV3)
    shared_cache = {"templates": {"key": "value"}, "center_index": {}}
    pipeline.default_template_lib_path = "/tmp/templates.pt"
    pipeline.layer2_estimator = SimpleNamespace(
        lib_path="/tmp/templates.pt",
        _template_cache=shared_cache,
    )
    observed = {}

    def layer3_adjust_templates(**kwargs):
        observed["cache"] = pipeline.layer3_estimator._template_cache
        observed["kwargs"] = dict(kwargs)
        return list(kwargs["nodes"]), 0.0

    pipeline.layer3_estimator = SimpleNamespace(
        lib_path="/tmp/templates.pt",
        _template_cache=None,
        layer3_adjust_templates=layer3_adjust_templates,
    )

    nodes, score = pipeline.layer3_adjust_templates_impl(
        nodes=[],
        S_target=torch.zeros(1),
        E_target=torch.zeros(6),
        lib_path="/tmp/templates.pt",
        verbose=False,
    )

    assert nodes == []
    assert score == 0.0
    assert observed["cache"] is shared_cache
PENDING_KEYS = {
    "unit_id",
    "pending_kind",
    "contract_kind",
    "target_role",
    "real_topology_partial",
    "real_node_partial",
    "source_fragment_ids",
    "source_fragment_classes",
    "source_node_ids",
    "phase",
    "subphase",
    "required_counts",
    "available_counts",
    "reason",
    "context_node_ids",
    "open_mainline_slots",
    "branch_slots",
    "infeasibility_proof",
    "attempt_reasons",
    "validation_errors",
}
ASSEMBLY_KEYS = {
    "ok",
    "reason",
    "feasible",
    "products",
    "pending_units",
    "resource_ledger",
    "flex_bounds",
    "flex_counts",
    "diagnostics",
    "planned_conversions",
    "flex_decision",
    "topology_metrics",
    "topology_signature",
    "requested_counts",
    "unresolved_counts",
    "blocked_reason",
    "active_subphase",
    "active_phase",
    "completed_phases",
    "residual_closure_products",
    "terminal_resource_counts",
    "terminal_converted_counts",
    "terminal_unresolved_counts",
    "phase_metrics",
    "phase_signature",
    "flex_target_plan",
    "resource_demand",
    "protected_resource_counts",
    "referenced_resource_counts",
    "external_candidate_capacity_exhausted",
    "ownership_ledger",
}
PUBLIC_PRODUCT_KEYS = {
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
}
PUBLIC_PENDING_KEYS = {
    "contract_id",
    "real",
    "real_gid",
    "pending_kind",
    "phase",
    "subphase",
    "source_fragment_ids",
    "source_fragment_classes",
    "source_node_ids",
    "claimed_source_node_ids",
    "core_node_ids",
    "context_node_ids",
    "open_mainline_slots",
    "branch_slots",
    "infeasibility_proof",
    "required_counts",
    "available_counts",
    "reason",
    "attempt_reasons",
    "validation_errors",
}
PUBLIC_ASSEMBLY_KEYS = {
    "ok",
    "reason",
    "feasible",
    "products",
    "pending_units",
    "resource_ledger",
    "flex_decision",
    "topology_metrics",
    "topology_signature",
    "requested_counts",
    "unresolved_counts",
    "blocked_reason",
    "active_subphase",
    "active_phase",
    "completed_phases",
    "residual_closure_products",
    "terminal_resource_counts",
    "terminal_converted_counts",
    "terminal_unresolved_counts",
    "phase_metrics",
    "phase_signature",
    "flex_target_plan",
    "resource_demand",
    "protected_resource_counts",
    "referenced_resource_counts",
    "external_candidate_capacity_exhausted",
    "ownership_ledger",
}
PUBLIC_LEDGER_KEYS = {
    "initial_counts",
    "consumed_counts",
    "topology_consumed_counts",
    "residual_consumed_counts",
    "remaining_counts",
    "shortage_counts",
    "snapshot_stage",
}


def _assembly_dict(nodes):
    classification = classify_topology_units(list(nodes or []))
    result = assemble_topology_units(list(nodes or []), classification)
    return classification.to_dict(), result.to_dict()


def _assert_minimal_product(product):
    assert set(product.keys()) == PRODUCT_KEYS
    assert isinstance(product["source_fragment_ids"], list)
    assert isinstance(product["source_fragment_classes"], list)
    assert isinstance(product["source_node_ids"], list)
    assert "diagnostics" not in product


def _assert_minimal_pending(item):
    assert set(item.keys()) == PENDING_KEYS
    assert isinstance(item["source_fragment_ids"], list)
    assert isinstance(item["source_fragment_classes"], list)
    assert isinstance(item["source_node_ids"], list)
    assert "diagnostics" not in item


def _assert_public_product(item):
    assert set(item.keys()) == PUBLIC_PRODUCT_KEYS
    assert isinstance(item["source_fragment_ids"], list)
    assert isinstance(item["source_fragment_classes"], list)
    assert isinstance(item["source_node_ids"], list)


def _assert_public_pending(item):
    assert set(item.keys()) == PUBLIC_PENDING_KEYS


def test_external_assembly_minimal_product_contract_for_explicit_a() -> None:
    center = make_node(10, 3)
    left = make_node(11, 9)
    right = make_node(12, 9)
    connect(center, left)
    connect(center, right)

    _, assembly = _assembly_dict([center, left, right])

    assert set(assembly.keys()) == ASSEMBLY_KEYS
    assert "units" not in assembly
    assert len(assembly["products"]) == 1
    product = assembly["products"][0]
    _assert_minimal_product(product)
    assert product["product_kind"] == "flex_chain"
    assert product["actual_topology_su"] == "9-3-9"
    assert product["actual_topology_gid"] == "11-10-12"
    assert product["source_fragment_classes"] == ["A"]
    assert assembly["pending_units"] == []


def test_external_assembly_closes_b_plus_g_without_legacy_fields() -> None:
    b_center = make_node(45, 3)
    b_aro = make_node(46, 9)
    b_open = make_node(47, 23)
    connect(b_center, b_aro)
    connect(b_center, b_open)

    g_center = make_node(55, 1)
    g_open = make_node(56, 23)
    connect(g_center, g_open)

    _, assembly = _assembly_dict([b_center, b_aro, b_open, g_center, g_open])

    assert len(assembly["products"]) == 1
    product = assembly["products"][0]
    _assert_minimal_product(product)
    assert product["product_kind"] == "side_chain"
    assert product["actual_topology_su"] == "1-23-23-3-9"
    assert product["source_fragment_classes"] == ["G", "B"]


def test_like24_b_plus_e_joins_real_open_end_terminal_to_aromatic() -> None:
    b_center = make_node(100, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    b_connector = make_node(101, 0)
    b_aromatic = make_node(102, 9)
    branch_23 = make_node(103, 23)
    branch_22 = make_node(104, 22)
    connect(b_center, b_connector, fixed=True)
    connect(b_connector, b_aromatic)
    connect(b_center, branch_23)
    connect(branch_23, branch_22)

    e_center = make_node(200, 2)
    e_terminal = make_node(201, 19, target_hop1_degree=1, special_anchor_mode="single")
    e_open = make_node(202, 23)
    connect(e_center, e_terminal)
    connect(e_center, e_open)

    nodes = [
        b_center,
        b_connector,
        b_aromatic,
        branch_23,
        branch_22,
        e_center,
        e_terminal,
        e_open,
    ]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    b_view = assembler.view_by_id[100]
    e_view = assembler.view_by_id[200]

    assert b_view.class_type == "24_CD_M"
    assert b_view.endpoint("left").role == "b_like_open_end"
    assert b_view.endpoint("right").role == "aromatic_closed_end"
    assert e_view.endpoint("left").role == "terminal_closed_end"
    assert e_view.endpoint("right").role == "e_like_open_end"

    assembly = assembler.assemble().to_dict()
    assert len(assembly["products"]) == 1
    product = assembly["products"][0]
    assert product["actual_topology_su"] == "19d1-2-23-single20d3(-23-22)-0-9"
    assert "-9-19d1-" not in product["actual_topology_su"]
    assert set(product["source_fragment_ids"]) == {100, 200}
    assert assembly["pending_units"] == []
    assert "topology_notation" not in product
    assert "node_id_notation" not in product
    assert "main_closure_mode" not in product
    assert "resource_cost" not in product


def test_double19d3_su28_su29_terminal_path_materializes_without_pending() -> None:
    aromatic_11 = make_node(180, 11)
    terminal_19 = make_node(317, 19, target_hop1_degree=1, special_anchor_mode="single")
    center = make_node(
        412,
        19,
        target_hop1_degree=3,
        special_anchor_mode="double",
        target_fixed_anchor_count=2,
    )
    terminal_28 = make_node(505, 28)
    connector_29 = make_node(569, 29)
    connect(center, aromatic_11)
    connect(center, terminal_28, fixed=True)
    connect(center, connector_29, fixed=True)
    connect(connector_29, terminal_19)

    nodes = [aromatic_11, terminal_19, center, terminal_28, connector_29]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    products = list(assembly.get("products", []) or [])
    assert len(products) == 1
    assert products[0]["actual_topology_su"] == "19d1-29-double19d3(-28)-11"
    assert sorted(products[0]["materialized_node_ids"]) == [180, 317, 412, 505, 569]
    assert assembly["pending_units"] == []


def test_remaining_stage_prioritizes_b_plus_b_flex_before_b_plus_g_side() -> None:
    b1_center = make_node(10, 3)
    b1_aro = make_node(11, 9)
    b1_open = make_node(12, 23)
    connect(b1_center, b1_aro)
    connect(b1_center, b1_open)

    b2_center = make_node(20, 3)
    b2_aro = make_node(21, 9)
    b2_open = make_node(22, 23)
    connect(b2_center, b2_aro)
    connect(b2_center, b2_open)

    g_center = make_node(30, 1)
    g_open = make_node(31, 23)
    connect(g_center, g_open)

    _, assembly = _assembly_dict([b1_center, b1_aro, b1_open, b2_center, b2_aro, b2_open, g_center, g_open])

    products = list(assembly.get("products", []) or [])
    assert len(products) == 1
    product = products[0]
    _assert_minimal_product(product)
    assert product["product_kind"] == "flex_chain"
    assert product["actual_topology_su"] == "9-3-23-23-3-9"
    assert product["source_fragment_classes"] == ["B", "B"]
    pending = [item for item in list(assembly.get("pending_units", []) or []) if item["pending_kind"] == "e_like_open"]
    assert len(pending) == 1
    assert pending[0]["reason"] == "F1_pool_shortage"


def test_external_assembly_closes_d_with_two_end_f1_segments() -> None:
    center = make_node(60, 3)
    open_left = make_node(61, 23)
    open_right = make_node(62, 23)
    pool_11_left = make_node(63, 11)
    pool_11_right = make_node(64, 11)
    connect(center, open_left)
    connect(center, open_right)

    _, assembly = _assembly_dict([center, open_left, open_right, pool_11_left, pool_11_right])

    assert len(assembly["products"]) == 1
    product = assembly["products"][0]
    _assert_minimal_product(product)
    assert product["product_kind"] == "flex_chain"
    assert product["actual_topology_su"] == "11-23-3-23-11"
    assert product["source_fragment_classes"] == ["D"]


def test_external_assembly_branch_pool_shortage_pending_uses_minimal_contract() -> None:
    center = make_node(60, 19, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(61, 29, target_hop1_degree=2)
    aromatic = make_node(62, 5)
    h11 = make_node(63, 11)
    connect(center, connector, fixed=True)
    connect(connector, aromatic)
    connect(center, h11)

    _, assembly = _assembly_dict([center, connector, aromatic, h11])

    assert assembly["products"] == []
    assert assembly["pending_units"]
    for item in assembly["pending_units"]:
        _assert_minimal_pending(item)
    pending = next(item for item in assembly["pending_units"] if item["reason"] == "branch_pool_shortage")
    assert pending["pending_kind"] == "explicit_flex"
    assert pending["real_topology_partial"] == "11-single19d3-29-5"
    assert pending["real_node_partial"] == "63-60-61-62"
    assert pending["source_fragment_classes"] == ["24_AB_M"]
    assert "missing_counts" not in pending
    assert "topology_template" not in pending


def test_fixed_branch_secondary_connector_can_pending_as_h1_shortage_without_role_conflict() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    connector_0 = make_node(2, 0)
    aromatic_9 = make_node(3, 9)
    connector_27 = make_node(4, 27)
    side_single20d2 = make_node(5, 20, target_hop1_degree=2, special_anchor_mode="single")
    pool_11 = make_node(6, 11)
    pool_23 = make_node(7, 23)
    pool_22 = make_node(8, 22)
    connect(center, connector_0, fixed=True)
    connect(connector_0, aromatic_9)
    connect(center, connector_27, fixed=True)
    connect(connector_27, side_single20d2)
    connect(center, pool_11)

    classification = classify_topology_units([center, connector_0, aromatic_9, connector_27, side_single20d2, pool_11, pool_23, pool_22])
    assembly = assemble_topology_units([center, connector_0, aromatic_9, connector_27, side_single20d2, pool_11, pool_23, pool_22], classification).to_dict()

    pending = next(item for item in list(assembly.get("pending_units", []) or []) if int(item.get("unit_id", 0)) == 1)
    assert str(pending.get("reason", "")) == "branch_pool_shortage"
    diagnostics = dict(assembly.get("diagnostics", {}) or {})
    pending_contracts = list(diagnostics.get("pending_contracts", []) or [])
    assert any(
        str(item.get("contract_kind", "")) == "branch_closure"
        and str(item.get("reason", "")) == "branch_pool_shortage"
        for item in pending_contracts
    )
    assert not any(str(item.get("reason", "")) == "fixed_branch_role_conflict" for item in pending_contracts)


def test_fixed_branch_secondary_connector_with_paired_like24_becomes_partner_pending() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    connector_0 = make_node(2, 0)
    aromatic_9 = make_node(3, 9)
    connector_27 = make_node(4, 27)
    side_single20d3 = make_node(5, 20, target_hop1_degree=3, special_anchor_mode="single")
    open_23 = make_node(6, 23)
    pool_11 = make_node(7, 11)
    connect(center, connector_0, fixed=True)
    connect(connector_0, aromatic_9)
    connect(center, connector_27, fixed=True)
    connect(connector_27, side_single20d3)
    connect(side_single20d3, open_23)
    connect(center, pool_11)

    classification = classify_topology_units([center, connector_0, aromatic_9, connector_27, side_single20d3, open_23, pool_11])
    assembly = assemble_topology_units([center, connector_0, aromatic_9, connector_27, side_single20d3, open_23, pool_11], classification).to_dict()

    pending = next(item for item in list(assembly.get("pending_units", []) or []) if int(item.get("unit_id", 0)) == 1)
    assert str(pending.get("reason", "")) == "fixed_branch_partner_missing"
    diagnostics = dict(assembly.get("diagnostics", {}) or {})
    pending_contracts = list(diagnostics.get("pending_contracts", []) or [])
    partner_contract = next(item for item in pending_contracts if str(item.get("contract_kind", "")) == "ring_partner")
    assert dict(partner_contract.get("required_counts", {}) or {}) == {}
    assert str(partner_contract.get("reason", "")) == "fixed_branch_partner_missing"


def test_external_assembly_side_ring_candidate_keeps_source_fragment_classes_top_level() -> None:
    left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    left_connector = make_node(1001, 27)
    left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    left_11 = make_node(1003, 11)
    connect(left, left_connector, fixed=True)
    connect(left_connector, left_tail)
    connect(left, left_11)

    right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right_connector = make_node(1011, 27)
    right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(right, right_connector, fixed=True)
    connect(right_connector, right_tail)

    bridge = make_node(1020, 27)
    connect(bridge, left)
    connect(bridge, right)

    _, assembly = _assembly_dict([left, left_connector, left_tail, left_11, right, right_connector, right_tail, bridge])

    pending = [
        item
        for item in assembly["pending_units"]
        if "24_AB_N_24_CD" in item["source_fragment_classes"]
    ]
    assert pending
    assert pending[0]["reason"] == "fixed_branch_H1_missing"
    for item in pending:
        _assert_minimal_pending(item)


def test_ring_validator_enforces_strict_backbone_and_bridgehead_rules() -> None:
    assert (
        _validate_ring_product_shape(
            "fatty_ring_side",
            labels=["11", "single19d3", "14", "15", "single19d3", "11"],
            topology_notation="11-single19d3-14-15-single19d3-11",
            actual_topology_su="11-single19d3-14-15-single19d3-11",
            diagnostics={
                "ring_backbone_labels": ["11", "single19d3", "14", "15", "single19d3"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5],
                "ring_like24_count": 2,
                "ring_template_kind": "side",
            },
        )
        == "side_ring_backbone_node_count:5"
    )

    assert (
        _validate_ring_product_shape(
            "fatty_ring_upper_lower",
            labels=["single19d3"] * 6,
            topology_notation="single19d3-23-single19d3-23-single19d3-24",
            actual_topology_su="single19d3-23-single19d3-23-single19d3-24",
            diagnostics={
                "ring_backbone_labels": ["24", "23", "single19d3", "23", "single19d3", "23"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6],
                "ring_like24_count": 2,
                "ring_template_kind": "upper_lower",
                "ring_has_fixed_anchor": False,
                "closure_target_fragment_id": 10,
            },
        )
        == "upper_ring_missing_fixed_anchor"
    )

    assert (
        _validate_ring_product_shape(
            "fatty_ring_fused",
            labels=["11", "single19d3", "14", "23", "14", "single19d3", "11"],
            topology_notation="11-single19d3-14-[-23-14-23-]-14-single19d3-11",
            actual_topology_su="11-single19d3-14-[-23-14-23-]-14-single19d3-11",
            diagnostics={
                "ring_backbone_labels": ["11", "single19d3", "14", "14", "single19d3", "11"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6],
                "ring_outer_backbone_labels": ["11", "single19d3", "14", "14", "single19d3", "11"],
                "ring_outer_backbone_node_ids": [1, 2, 3, 4, 5, 6],
                "ring_inner_backbone_labels": ["14", "23", "14", "23", "14", "14"],
                "ring_inner_backbone_node_ids": [3, 7, 8, 9, 10, 4],
                "ring_like24_count": 4,
                "ring_template_kind": "fused",
                "ring_bridgehead_ids": [3, 4],
                "ring_pair_bridgehead_mode": True,
                "ring_inner_bridgehead_capable": True,
            },
        )
        == "fused_bridgehead_template_conflict"
    )


def test_ring_validator_accepts_strict_valid_ring_diagnostics() -> None:
    assert (
        _validate_ring_product_shape(
            "fatty_ring_side",
            labels=["11", "24", "23", "23", "24", "11"],
            topology_notation="11-24-23-23-24-11",
            actual_topology_su="11-24-23-23-24-11",
            diagnostics={
                "ring_backbone_labels": ["11", "24", "23", "23", "24", "11"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6],
                "ring_like24_count": 2,
                "ring_template_kind": "side",
            },
        )
        is None
    )

    assert (
        _validate_ring_product_shape(
            "fatty_ring_upper_lower",
            labels=["24", "23", "24", "23", "23", "23"],
            topology_notation="24-23-24-23-23-23",
            actual_topology_su="24-23-24-23-23-23",
            diagnostics={
                "ring_backbone_labels": ["24", "23", "24", "23", "23", "23"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6],
                "ring_like24_count": 2,
                "ring_template_kind": "upper_lower",
                "ring_has_fixed_anchor": True,
                "closure_target_fragment_id": 10,
                "closure_target_node_id": 1,
            },
        )
        is None
    )

    assert (
        _validate_ring_product_shape(
            "fatty_ring_fused",
            labels=["11", "24", "14", "23", "23", "24", "23", "14", "24", "11"],
            topology_notation="11-24-14-[-23-23-24-23-]-14-24-11",
            actual_topology_su="11-24-14-[-23-23-24-23-]-14-24-11",
            diagnostics={
                "ring_backbone_labels": ["11", "24", "14", "14", "24", "11"],
                "ring_backbone_node_ids": [1, 2, 3, 8, 9, 10],
                "ring_outer_backbone_labels": ["11", "24", "14", "14", "24", "11"],
                "ring_outer_backbone_node_ids": [1, 2, 3, 8, 9, 10],
                "ring_inner_backbone_labels": ["14", "23", "23", "24", "23", "14"],
                "ring_inner_backbone_node_ids": [3, 4, 5, 6, 7, 8],
                "ring_like24_count": 4,
                "ring_template_kind": "fused",
                "ring_bridgehead_ids": [3, 8],
                "ring_pair_bridgehead_mode": False,
                "ring_inner_bridgehead_capable": False,
            },
        )
        is None
    )


def test_ring_validator_rejects_known_side_ring_backbone_overcount() -> None:
    bad_topology = "11-single20d3-27-double20d3(-27)-14(-23-22)-15-single19d3(-28)-11"
    assert (
        _validate_ring_product_shape(
            "fatty_ring_side",
            labels=["11", "single20d3", "27", "14", "15", "11"],
            topology_notation=bad_topology,
            actual_topology_su=bad_topology,
            diagnostics={
                "ring_backbone_labels": ["11", "single20d3", "27", "14", "15", "single19d3", "11"],
                "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6, 7],
                "ring_like24_count": 3,
                "ring_template_kind": "side",
            },
        )
        == "side_ring_backbone_node_count:7"
    )


def test_invalid_ring_attempts_are_reported_as_bad_topologies() -> None:
    bad_topology = "11-single20d3-27-double20d3(-27)-14(-23-22)-15-single19d3(-28)-11"
    classification = classify_topology_units([])
    assembler = ExternalTopologyAssembler([], classification)

    product = assembler._make_product(
        product_kind="fatty_ring_side",
        reason="ring:side_template",
        source_views=[],
        labels=["11", "single20d3", "27", "14", "15", "11"],
        node_ids=[1, 2, 3, 4, 5, 6],
        branch_map={},
        branch_id_map={},
        diagnostics={
            "topology_notation": bad_topology,
            "node_id_notation": "1-2-3-4-5-6-7",
            "actual_topology_su": bad_topology,
            "actual_topology_gid": "1-2-3-4-5-6-7",
            "ring_backbone_labels": ["11", "single20d3", "27", "14", "15", "single19d3", "11"],
            "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6, 7],
            "ring_like24_count": 3,
            "ring_template_kind": "side",
        },
    )

    assert product is None
    assert assembler.invalid_topologies == [
        {
            "product_kind": "fatty_ring_side",
            "actual_topology_su": bad_topology,
            "actual_topology_gid": "1-2-3-4-5-6-7",
            "topology_notation": bad_topology,
            "reason": "side_ring_backbone_node_count:7",
            "ring_template_kind": "side",
            "ring_like24_count": 3,
            "ring_backbone_labels": ["11", "single20d3", "27", "14", "15", "single19d3", "11"],
            "ring_backbone_node_ids": [1, 2, 3, 4, 5, 6, 7],
        }
    ]

    report = build_external_assembly_audit_report(
        {},
        {
            "ok": False,
            "reason": "external_structural_closure_pending",
            "products": [],
            "pending_units": [],
            "diagnostics": {"bad_topologies": list(assembler.invalid_topologies)},
        },
    )
    assert "=== Bad Topologies ===" in report
    assert f"actual_su={bad_topology}" in report
    assert "gid=1-2-3-4-5-6-7" in report
    assert "reason=side_ring_backbone_node_count:7" in report


def test_external_assembly_report_includes_empty_unrecognized_section() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification)

    report = build_external_assembly_audit_report(classification.to_dict(), assembly.to_dict())

    assert "=== Unrecognized Topologies ===" in report
    assert "=== Unrecognized Topologies ===\n(none)" in report


def test_external_assembly_report_lists_unrecognized_topologies() -> None:
    report = build_external_assembly_audit_report(
        {
            "classification_rejects": [
                {
                    "global_id": 392,
                    "node_id": 392,
                    "su_type": 24,
                    "reason": "claim_conflict",
                }
            ],
            "unclassified_nodes": [
                {
                    "global_id": 777,
                    "node_id": 777,
                    "su_type": 14,
                    "resource_pool_member": False,
                    "reason": "unclassified_non_resource",
                }
            ],
        },
        {
            "ok": False,
            "reason": "external_structural_closure_pending",
            "products": [],
            "pending_units": [],
            "diagnostics": {},
        },
    )

    assert "=== Unrecognized Topologies ===" in report
    assert "1. [unrecognized_topology]" in report
    assert "gid=392" in report
    assert "su_type=24" in report
    assert "reason=claim_conflict" in report
    assert "gid=777" in report
    assert "su_type=14" in report
    assert "reason=unclassified_non_resource" in report


def test_external_assembly_flex_range_enforced_with_targets_caps_flex_output() -> None:
    center_a = make_node(10, 3)
    left_a = make_node(11, 9)
    right_a = make_node(12, 9)
    connect(center_a, left_a)
    connect(center_a, right_a)

    center_b = make_node(20, 3)
    left_b = make_node(21, 9)
    right_b = make_node(22, 9)
    connect(center_b, left_b)
    connect(center_b, right_b)

    center_c = make_node(30, 3)
    left_c = make_node(31, 9)
    right_c = make_node(32, 9)
    connect(center_c, left_c)
    connect(center_c, right_c)

    classification = classify_topology_units([center_a, left_a, right_a, center_b, left_b, right_b, center_c, left_c, right_c])
    assembly = assemble_topology_units(
        [center_a, left_a, right_a, center_b, left_b, right_b, center_c, left_c, right_c],
        classification,
        S_target=None,
        E_target=None,
    ).to_dict()
    flex_bounds = dict(assembly.get("flex_bounds", {}) or {})
    flex_counts = dict(assembly.get("flex_counts", {}) or {})
    assert bool(flex_bounds.get("enforced", False))
    assert int(flex_counts.get("final", 0) or 0) == 0
    assert int(flex_counts.get("explicit_total", 0) or 0) == 3
    assert str(assembly.get("reason", "")) == "external_flex_range_exceeded"
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "flex_range_reconciliation"
    pending_units = list(assembly.get("pending_units", []) or [])
    assert len(pending_units) == 3
    assert all(str(item.get("pending_kind", "")) == "explicit_flex" for item in pending_units)
    assert all(str(item.get("reason", "")) == "flex_range_reconciliation_pending" for item in pending_units)

    import torch

    targeted = assemble_topology_units(
        [center_a, left_a, right_a, center_b, left_b, right_b, center_c, left_c, right_c],
        classification,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ).to_dict()
    flex_bounds = dict(targeted.get("flex_bounds", {}) or {})
    flex_counts = dict(targeted.get("flex_counts", {}) or {})
    assert bool(flex_bounds.get("enforced", False))
    assert int(flex_counts.get("final", 0) or 0) == 0
    assert int(flex_counts.get("explicit_total", 0) or 0) == 3
    assert str(targeted.get("reason", "")) == "external_flex_range_exceeded"
    assert str(dict(targeted.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "flex_range_reconciliation"
    pending_units = list(targeted.get("pending_units", []) or [])
    assert len(pending_units) == 3
    assert all(str(item.get("pending_kind", "")) == "explicit_flex" for item in pending_units)
    assert all(str(item.get("reason", "")) == "flex_range_reconciliation_pending" for item in pending_units)


def test_reduced_flex_target_completes_external_then_residual_flex() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    side = make_node(3, 20, target_hop1_degree=2, special_anchor_mode="single")
    open23 = make_node(4, 23)
    connect(center, connector, fixed=True)
    connect(connector, side)
    connect(center, open23)

    anchor24 = make_node(500, 24)
    anchor11 = make_node(501, 11)
    anchor23 = make_node(502, 23)
    connect(anchor24, anchor11)
    connect(anchor24, anchor23)

    import torch

    nodes = [
        center,
        connector,
        side,
        open23,
        anchor24,
        anchor11,
        anchor23,
        make_node(600, 11),
        make_node(601, 11),
        make_node(605, 11),
        make_node(602, 23),
        make_node(603, 23),
        make_node(604, 23),
        make_node(700, 22),
    ]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(
        nodes,
        classification,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ).to_dict()
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ).to_dict()
    plan = plan_stage3_resource_adjustments(
        H=None,
        nodes=nodes,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    products = list(assembly.get("products", []) or [])
    assert [str(item.get("product_kind", "")) for item in products] == [
        "flex_chain",
        "residual_flex_chain",
    ]
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "residual_resource_self_closure"
    assert dict(assembly.get("flex_counts", {}) or {}).get("final") == 2
    assert dict(assembly.get("flex_target_plan", {}) or {}).get("remaining_preferred_gap") == 0
    assert int(dict(phase_eval.get("flex_audit", {}) or {}).get("target_flex_count", 0)) == 2
    assert str(phase_eval.get("phase", "")) == "residual_resource_self_closure"
    assert str(plan.phase) == "complete"
    assert str(plan.reason) == "stage3_complete"
    assert not any(
        int(transaction.contract_id) >= 2_000_000
        for transaction in list(plan.phase_result.transactions or [])
    )


def test_assemble_external_topologies_honors_flex_preassembly_gate() -> None:
    center_a = make_node(10, 3)
    left_a = make_node(11, 9)
    right_a = make_node(12, 9)
    connect(center_a, left_a)
    connect(center_a, right_a)

    center_b = make_node(20, 3)
    left_b = make_node(21, 9)
    right_b = make_node(22, 9)
    connect(center_b, left_b)
    connect(center_b, right_b)

    center_c = make_node(30, 3)
    left_c = make_node(31, 9)
    right_c = make_node(32, 9)
    connect(center_c, left_c)
    connect(center_c, right_c)

    nodes = [center_a, left_a, right_a, center_b, left_b, right_b, center_c, left_c, right_c]
    classification = classify_topology_units(nodes)
    assembly = assemble_external_topologies(nodes, classification).to_dict()

    assert list(assembly.get("products", []) or []) == []
    assert str(assembly.get("reason", "")) == "external_flex_range_exceeded"
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "flex_range_reconciliation"
    pending_units = list(assembly.get("pending_units", []) or [])
    assert len(pending_units) == 3
    assert all(str(item.get("pending_kind", "")) == "explicit_flex" for item in pending_units)
    assert all(str(item.get("reason", "")) == "flex_range_reconciliation_pending" for item in pending_units)


def test_direct_external_topology_assembler_assemble_honors_flex_preassembly_gate() -> None:
    center_a = make_node(10, 3)
    left_a = make_node(11, 9)
    right_a = make_node(12, 9)
    connect(center_a, left_a)
    connect(center_a, right_a)

    center_b = make_node(20, 3)
    left_b = make_node(21, 9)
    right_b = make_node(22, 9)
    connect(center_b, left_b)
    connect(center_b, right_b)

    center_c = make_node(30, 3)
    left_c = make_node(31, 9)
    right_c = make_node(32, 9)
    connect(center_c, left_c)
    connect(center_c, right_c)

    nodes = [center_a, left_a, right_a, center_b, left_b, right_b, center_c, left_c, right_c]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    assembly = assembler.assemble().to_dict()

    assert list(assembly.get("products", []) or []) == []
    assert str(assembly.get("reason", "")) == "external_flex_range_exceeded"
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "flex_range_reconciliation"
    pending_units = list(assembly.get("pending_units", []) or [])
    assert len(pending_units) == 3
    assert all(str(item.get("pending_kind", "")) == "explicit_flex" for item in pending_units)
    assert all(str(item.get("reason", "")) == "flex_range_reconciliation_pending" for item in pending_units)


def test_below_range_only_selects_b_like_candidates_for_supplemental_flex() -> None:
    b_center = make_node(10, 3)
    b_aro = make_node(11, 9)
    b_open = make_node(12, 23)
    connect(b_center, b_aro)
    connect(b_center, b_open)

    extra_aromatics = [make_node(100 + idx, 9) for idx in range(9)]
    rigid10_nodes = [make_node(200 + idx, 10) for idx in range(4)]
    nodes = [b_center, b_aro, b_open] + extra_aromatics + rigid10_nodes
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    ).to_dict()

    assert str(phase_eval.get("phase", "")) == "external_structural_closure"
    assert str(phase_eval.get("flex_range_action_class", "")) == "below_range"
    assert bool(phase_eval.get("assembly_attempted", False)) is True
    decision = dict(phase_eval.get("flex_decision", {}) or {})
    assert list(decision.get("selected_candidate_fragment_ids", []) or []) == [10]
    flex_audit = dict(phase_eval.get("flex_audit", {}) or {})
    assert int(flex_audit.get("X_before_trim", 0) or 0) == 2
    assert int(flex_audit.get("X_after_trim", 0) or 0) == 2
    assert int(flex_audit.get("Y", 0) or 0) == 2
    assert int(flex_audit.get("flex_lower", 0) or 0) == 1
    assert int(flex_audit.get("flex_upper", 0) or 0) == 1
    assembly_view = dict(phase_eval.get("assembly_view", {}) or {})
    assert list(assembly_view.get("products", []) or []) == []
    pending_units = list(assembly_view.get("pending_units", []) or [])
    assert [str(item.get("pending_kind", "")) for item in pending_units] == ["b_like_open"]
    assert str(pending_units[0].get("reason", "")) == "F1_pool_shortage"
    contracts = list(phase_eval.get("flex_reconciliation_contracts", []) or [])
    candidate_contracts = [
        item for item in contracts
        if str(item.get("reason", "")) == "supplemental_flex_candidate"
    ]
    assert [int(item.get("contract_id", -1)) for item in candidate_contracts] == [10]
    assert all(
        str(dict(item.get("diagnostics", {}) or {}).get("candidate_kind", "")) == "b_like_open"
        for item in candidate_contracts
    )


def test_below_range_includes_d_like_open_in_supplemental_candidate_pool() -> None:
    left = make_node(414, 14)
    right = make_node(415, 14)
    open_23 = make_node(23, 23)
    open_24 = make_node(24, 24)
    connect(left, right)
    connect(left, open_23)
    connect(right, open_24)

    extra_aromatics = [make_node(100 + idx, 9) for idx in range(9)]
    rigid10_nodes = [make_node(200 + idx, 10) for idx in range(4)]
    nodes = [left, right, open_23, open_24] + extra_aromatics + rigid10_nodes
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(
        nodes,
        classification=classification,
    ).to_dict()

    assert str(phase_eval.get("phase", "")) == "external_structural_closure"
    flex_audit = dict(phase_eval.get("flex_audit", {}) or {})
    candidate_profiles = list(flex_audit.get("supplemental_flex_candidate_profiles", []) or [])
    assert any(str(item.get("candidate_kind", "")) == "d_like_open" for item in candidate_profiles)


def test_flex_audit_only_counts_strict_explicit_mainline_fragments() -> None:
    explicit_center = make_node(900, 3)
    explicit_left = make_node(901, 9)
    explicit_right = make_node(902, 9)
    connect(explicit_center, explicit_left)
    connect(explicit_center, explicit_right)

    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    side = make_node(3, 20, target_hop1_degree=2, special_anchor_mode="single")
    open23 = make_node(4, 23)
    connect(center, connector, fixed=True)
    connect(connector, side)
    connect(center, open23)

    left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    left_connector = make_node(1001, 27)
    left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    left_11 = make_node(1003, 11)
    connect(left, left_connector, fixed=True)
    connect(left_connector, left_tail)
    connect(left, left_11)

    right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right_connector = make_node(1011, 27)
    right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(right, right_connector, fixed=True)
    connect(right_connector, right_tail)

    bridge = make_node(1020, 27)
    connect(bridge, left)
    connect(bridge, right)

    nodes = [
        explicit_center,
        explicit_left,
        explicit_right,
        center,
        connector,
        side,
        open23,
        left,
        left_connector,
        left_tail,
        left_11,
        right,
        right_connector,
        right_tail,
        bridge,
    ]
    classification = classify_topology_units(nodes)
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classification).to_dict()

    flex_audit = dict(phase_eval.get("flex_audit", {}) or {})
    assert int(flex_audit.get("explicit_flex_count", 0) or 0) == 1
    assert list(flex_audit.get("hard_explicit_flex_fragment_ids", []) or []) == [900]


def test_like24_open_views_expose_structured_flex_reconcile_roles() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    side = make_node(3, 20, target_hop1_degree=2, special_anchor_mode="single")
    open23 = make_node(4, 23)
    connect(center, connector, fixed=True)
    connect(connector, side)
    connect(center, open23)

    left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    left_connector = make_node(1001, 27)
    left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    left_11 = make_node(1003, 11)
    connect(left, left_connector, fixed=True)
    connect(left_connector, left_tail)
    connect(left, left_11)
    right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right_connector = make_node(1011, 27)
    right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(right, right_connector, fixed=True)
    connect(right_connector, right_tail)
    bridge = make_node(1020, 27)
    connect(bridge, left)
    connect(bridge, right)

    pair14a = make_node(410, 14)
    pair14b = make_node(414, 14)
    pair23 = make_node(412, 23)
    pair24 = make_node(424, 24)
    connect(pair14a, pair14b)
    connect(pair14a, pair23)
    connect(pair14b, pair24)

    nodes = [
        center,
        connector,
        side,
        open23,
        left,
        left_connector,
        left_tail,
        left_11,
        right,
        right_connector,
        right_tail,
        bridge,
        pair14a,
        pair14b,
        pair23,
        pair24,
    ]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    views_by_class = {str(view.class_type): view for view in list(assembler.views or [])}

    cd_n_view = views_by_class["24_CD_N"]
    assert {"d_like_open", "double_end_flex_source", "like24_open"} <= set(cd_n_view.flex_reconcile_roles)
    assert int(cd_n_view.flex_reconcile_priority) == 40
    assert set(cd_n_view.ring_roles) == set(dict(cd_n_view.evidence or {}).get("ring_roles", []) or [])
    assert int(cd_n_view.ring_like24_units) == int(dict(cd_n_view.evidence or {}).get("ring_like24_units", 0) or 0)

    ab_cd_view = views_by_class["24_AB_N_24_CD"]
    assert {"b_like_open", "single_end_flex_source", "like24_open"} <= set(ab_cd_view.flex_reconcile_roles)
    assert int(ab_cd_view.flex_reconcile_priority) == 30
    assert set(ab_cd_view.ring_roles) == set(dict(ab_cd_view.evidence or {}).get("ring_roles", []) or [])
    assert int(ab_cd_view.ring_like24_units) == int(dict(ab_cd_view.evidence or {}).get("ring_like24_units", 0) or 0)

    cd_cd_view = views_by_class["24_CD_N_24_CD"]
    assert {"d_like_open", "double_end_flex_source", "like24_open"} <= set(cd_cd_view.flex_reconcile_roles)
    assert int(cd_cd_view.flex_reconcile_priority) == 40
    assert set(cd_cd_view.ring_roles) == set(dict(cd_cd_view.evidence or {}).get("ring_roles", []) or [])
    assert int(cd_cd_view.ring_like24_units) == int(dict(cd_cd_view.evidence or {}).get("ring_like24_units", 0) or 0)

    assert cd_n_view.ring_backbone_labels == {
        "fused": ["single20d3", "27", "single20d2"],
        "side": ["single20d3", "27", "single20d2"],
        "upper_lower": ["single20d3", "27", "single20d2"],
    }
    assert cd_n_view.ring_backbone_span == {"fused": 3, "side": 3, "upper_lower": 3}
    assert cd_n_view.ring_backbone_kind == {
        "fused": "fused_path",
        "side": "side_path",
        "upper_lower": "upper_path",
    }
    assert cd_n_view.ring_template_roles == {
        "side": ["inner_single"],
        "upper_lower": ["path_single"],
        "fused": ["inner_single"],
    }
    assert dict(cd_n_view.ring_anchor_profile or {}).get("fixed_anchor_eligible", False) is False

    assert ab_cd_view.ring_backbone_labels == {
        "side": ["11", "single20d3", "27", "single20d3"],
    }
    assert ab_cd_view.ring_backbone_node_ids == {
        "side": [1003, 1000, 1020, 1010],
    }
    assert ab_cd_view.ring_backbone_span == {"side": 4}
    assert ab_cd_view.ring_backbone_kind == {"side": "side_path"}
    assert ab_cd_view.ring_template_roles == {
        "side": ["terminal_core_pair"],
        "upper_lower": [],
        "fused": [],
    }
    assert dict(ab_cd_view.ring_anchor_profile or {}).get("left_aromatic", False) is True
    assert dict(ab_cd_view.ring_anchor_profile or {}).get("touches_aromatic_anchor", False) is True

    assert cd_cd_view.ring_backbone_labels == {
        "fused": ["14", "14"],
        "side": ["14", "14"],
        "upper_lower": ["14", "14"],
    }
    assert cd_cd_view.ring_backbone_span == {"fused": 2, "side": 2, "upper_lower": 2}
    assert cd_cd_view.ring_backbone_kind == {
        "fused": "fused_path",
        "side": "side_path",
        "upper_lower": "upper_path",
    }
    assert cd_cd_view.ring_template_roles == {
        "side": ["inner_pair"],
        "upper_lower": ["path_pair"],
        "fused": ["inner_pair"],
    }
    assert dict(cd_cd_view.ring_anchor_profile or {}).get("bridgehead_role", "") == "fused_bridgehead_substitute_pair"


def test_priority_ring_scheduler_dispatches_fused_side_upper_in_order() -> None:
    anchor24 = make_node(500, 24)
    anchor11 = make_node(501, 11)
    anchor23 = make_node(502, 23)
    connect(anchor24, anchor11)
    connect(anchor24, anchor23)

    classification = classify_topology_units([anchor24, anchor11, anchor23])
    assembler = ExternalTopologyAssembler([anchor24, anchor11, anchor23], classification)
    call_order = []

    def _capture_assemble_ring(*, product_kind, preferred_views, branch_donors, like24_fill_views=None):
        del preferred_views, branch_donors, like24_fill_views
        call_order.append(str(product_kind))
        return False

    assembler._assemble_ring = _capture_assemble_ring
    assert assembler._assemble_priority_rings([]) is False
    assert call_order == []


def test_non_applicable_flex_window_does_not_force_flex_range_gate() -> None:
    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    side = make_node(3, 20, target_hop1_degree=2, special_anchor_mode="single")
    open23 = make_node(4, 23)
    connect(center, connector, fixed=True)
    connect(connector, side)
    connect(center, open23)

    import torch

    nodes = [center, connector, side, open23]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(
        nodes,
        classification,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ).to_dict()

    assert not bool(assembly.get("ok", True))
    assert str(assembly.get("reason", "")) == "external_structural_closure_pending"
    assert list(assembly.get("pending_units", []) or [])
    assert all("flex_range" not in str(item.get("reason", "")) for item in assembly["pending_units"])
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "external_structural_closure"
    flex_audit = dict(dict(assembly.get("diagnostics", {}) or {}).get("flex_audit", {}) or {})
    assert bool(flex_audit.get("flex_window_applicable", True)) is False
    assert str(flex_audit.get("scenario", "")) == "not_applicable"






















def test_external_h_branch_does_not_consume_an_unrelated_fragment() -> None:
    b_center = make_node(10, 3)
    b_aro = make_node(11, 9)
    b_open = make_node(12, 23)
    connect(b_center, b_aro)
    connect(b_center, b_open)

    g_center = make_node(20, 1)
    g_open = make_node(21, 23)
    connect(g_center, g_open)

    classification = classify_topology_units([b_center, b_aro, b_open, g_center, g_open])
    assembler = ExternalTopologyAssembler([b_center, b_aro, b_open, g_center, g_open], classification)
    branch_donors = assembler._available_views(assembler._is_side_pending_view)
    result = assembler._resolve_branch_closure("H", branch_donors, anchor_index=0)

    assert result is None
    assert len(branch_donors) == 1


def test_external_assembly_stage3_phase_tracks_only_front_half_state() -> None:
    center_complete = make_node(50, 3)
    left_complete = make_node(51, 9)
    right_complete = make_node(52, 9)
    connect(center_complete, left_complete)
    connect(center_complete, right_complete)
    nodes_complete = [center_complete, left_complete, right_complete]
    classification_complete = classify_topology_units(nodes_complete)
    assembly_complete = assemble_topology_units(nodes_complete, classification_complete).to_dict()
    assert str(dict(assembly_complete.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "complete"

    center = make_node(1, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    connector = make_node(2, 27)
    side = make_node(3, 20, target_hop1_degree=2, special_anchor_mode="single")
    open23 = make_node(4, 23)
    connect(center, connector, fixed=True)
    connect(connector, side)
    connect(center, open23)

    import torch

    nodes_flex = [center, connector, side, open23]
    classification_flex = classify_topology_units(nodes_flex)
    assembly_flex = assemble_topology_units(
        nodes_flex,
        classification_flex,
        S_target=torch.zeros(2400),
        E_target=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ).to_dict()
    assert str(dict(assembly_flex.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "external_structural_closure"

    left = make_node(1000, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    left_connector = make_node(1001, 27)
    left_tail = make_node(1002, 20, target_hop1_degree=2, special_anchor_mode="single")
    left_11 = make_node(1003, 11)
    connect(left, left_connector, fixed=True)
    connect(left_connector, left_tail)
    connect(left, left_11)
    right = make_node(1010, 20, target_hop1_degree=3, special_anchor_mode="single", target_fixed_anchor_count=1)
    right_connector = make_node(1011, 27)
    right_tail = make_node(1012, 20, target_hop1_degree=2, special_anchor_mode="single")
    connect(right, right_connector, fixed=True)
    connect(right_connector, right_tail)
    bridge = make_node(1020, 27)
    connect(bridge, left)
    connect(bridge, right)
    anchor24 = make_node(500, 24)
    anchor11 = make_node(501, 11)
    anchor23 = make_node(502, 23)
    connect(anchor24, anchor11)
    connect(anchor24, anchor23)
    nodes_pending = [
        left,
        left_connector,
        left_tail,
        left_11,
        right,
        right_connector,
        right_tail,
        bridge,
        anchor24,
        anchor11,
        anchor23,
        make_node(600, 23),
        make_node(601, 23),
        make_node(602, 23),
        make_node(603, 23),
    ]
    classification_pending = classify_topology_units(nodes_pending)
    assembly_pending = assemble_topology_units(nodes_pending, classification_pending).to_dict()
    assert str(dict(assembly_pending.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "external_structural_closure"


def test_unclassified_su24_candidate_remains_residual_resource() -> None:
    center = make_node(324, 24)
    pool11 = make_node(11, 11)
    open23 = make_node(23, 23)
    connect(center, pool11)
    connect(center, open23)

    nodes = [center, pool11, open23]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()
    phase_eval = build_stage3_phase_evaluation(nodes, classification=classification).to_dict()

    assert list(assembly.get("products", []) or []) == []
    assert list(assembly.get("pending_units", []) or []) == []
    assert str(assembly.get("reason", "")) == "residual_resource_self_closure_pending"
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "residual_resource_self_closure"
    assert str(phase_eval.get("phase", "")) == "residual_resource_self_closure"
    phase_diag = dict(phase_eval.get("diagnostics", {}) or {})
    assert int(phase_diag.get("external_pending_unit_count", 0) or 0) == 0
    assert int(phase_diag.get("resource_pool_only_pending_unit_count", 0) or 0) == 0


def test_residual_self_closure_consumes_resource_pool_side_chain() -> None:
    nodes = [
        make_node(200, 11),
        make_node(201, 11),
        make_node(202, 11),
        make_node(203, 22),
        make_node(204, 23),
        make_node(205, 23),
        make_node(206, 23),
        make_node(207, 23),
    ]

    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    products = list(assembly.get("products", []) or [])
    assert len(products) == 1
    assert str(products[0].get("product_kind", "")) == "residual_side_chain"
    assert str(products[0].get("actual_topology_su", "")) == "11-23-23-22"
    assert assembly["pending_units"] == []
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "residual_resource_self_closure"
    remaining_counts = dict(dict(assembly.get("resource_ledger", {}) or {}).get("remaining_counts", {}) or {})
    assert int(remaining_counts.get("11", 0) or 0) == 2
    assert int(remaining_counts.get("22", 0) or 0) == 0
    assert int(remaining_counts.get("23", 0) or 0) == 2


def test_residual_self_closure_keeps_leftover_24_for_conversion_cycle() -> None:
    nodes = [
        make_node(300, 11),
        make_node(301, 11),
        make_node(302, 22),
        make_node(303, 23),
        make_node(304, 23),
        make_node(305, 24),
    ]

    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    products = list(assembly.get("products", []) or [])
    assert len(products) == 1
    assert str(products[0].get("actual_topology_su", "")) == "11-23-23-22"
    remaining_counts = dict(dict(assembly.get("resource_ledger", {}) or {}).get("remaining_counts", {}) or {})
    assert int(remaining_counts.get("24", 0) or 0) == 1
    assert int(remaining_counts.get("23", 0) or 0) == 0
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "residual_resource_self_closure"


def test_external_assembly_csv_regression_contract_and_topologies() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    assert set(assembly.keys()) == ASSEMBLY_KEYS
    assert "units" not in assembly
    assert "complete_topologies" not in dict(assembly.get("diagnostics", {}) or {})
    assert "pending_topologies" not in dict(assembly.get("diagnostics", {}) or {})

    ledger = dict(assembly.get("resource_ledger", {}) or {})
    initial_counts = dict(ledger.get("initial_counts", {}) or {})
    consumed_counts = dict(ledger.get("consumed_counts", {}) or {})
    remaining_counts = dict(ledger.get("remaining_counts", {}) or {})
    initial_ids_by_su = dict(ledger.get("initial_ids_by_su", {}) or {})

    assert int(initial_counts.get("11", 0)) == int(consumed_counts.get("11", 0)) + int(remaining_counts.get("11", 0))
    assert int(initial_counts.get("22", 0)) == int(consumed_counts.get("22", 0)) + int(remaining_counts.get("22", 0))
    assert int(initial_counts.get("23", 0)) == int(consumed_counts.get("23", 0)) + int(remaining_counts.get("23", 0))
    assert int(initial_counts.get("24", 0)) == int(consumed_counts.get("24", 0)) + int(remaining_counts.get("24", 0))

    ownership_states = dict(ledger.get("allocation_state_by_node", {}) or {})
    assert set(ownership_states.values()).issubset({"fixed", "fragment_owned", "remaining", "remaining"})

    assert int(dict(classification.summary or {}).get("claim_conflict_count", 0) or 0) == 0
    assert not [item for item in list(classification.classification_rejects or []) if int(item.get("node_id", -1)) == 392]
    assert not [item for item in list(classification.unclassified_nodes or []) if int(item.get("node_id", -1)) == 21]

    for item in list(assembly.get("products", []) or [])[:20]:
        _assert_minimal_product(item)
    for item in list(assembly.get("pending_units", []) or [])[:20]:
        _assert_minimal_pending(item)

    actual_topologies = {str(item.get("actual_topology_su", "")) for item in list(assembly.get("products", []) or [])}
    bad_actual_topologies = {
        "23-11-23-3-23",
        "23-11-23-1",
        "23-11-17-18",
        "11-17-17-18",
        "9-3-18",
        "23(14)(23)(23)-11-14-14",
        "single20d3(11)(23)(27)-27-double20d3-1",
        "11-single20d3-27-double20d3(-27)-14(-23-22)-15-single19d3(-28)-11",
        "11-single19d3(-28)-14-[-23-14(-23-22)-15-23-]-14-single19d3(-28)-11",
        "11-single19d3(-28)-[23-14(-22)-15-23]-single19d3(-28)-11",
        "11-single19d3(-28)-[23-single20d3(-22)-27-single20d3(-22)-23]-single19d3(-28)-11",
        "11-single19d3(-28)-[23-14(-23-22)-15-14(-22)-15-23]-single19d3(-28)-11",
    }
    assert not (bad_actual_topologies & actual_topologies)
    assert actual_topologies
    assert str(assembly.get("reason", "")) == "external_ring_candidates_pending"
    assert str(dict(assembly.get("diagnostics", {}) or {}).get("stage3_phase", "")) == "external_structural_closure"

    assert any(
        str(fragment.class_type) in {"24_AB_N_24_CD", "24_CD_N_24_CD"}
        for fragment in list(classification.fragments or [])
    )

    side_products = [
        item for item in list(assembly.get("products", []) or [])
        if str(item.get("product_kind", "")) == "fatty_ring_side"
    ]
    assert all("[" not in str(item.get("actual_topology_su", "")) for item in side_products)
    assert all("]" not in str(item.get("actual_topology_su", "")) for item in side_products)

    ring_pending = [
        item for item in list(assembly.get("pending_units", []) or [])
        if str(item.get("pending_kind", "")) == "pending_ring_candidate"
    ]
    assert ring_pending
    assert all(str(item.get("phase", "")) == "external_structural_closure" for item in ring_pending)
    assert all(item.get("source_node_ids") for item in ring_pending)
    assert any(str(item.get("pending_kind", "")) == "e_like_open" for item in list(assembly.get("pending_units", []) or []))
    assert not any(str(item.get("real_topology_partial", "")) == "-1" for item in list(assembly.get("pending_units", []) or []))
    assert all(item.get("source_fragment_ids") for item in list(assembly.get("pending_units", []) or []))


def test_external_assembly_report_is_minimal() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification)

    report = build_external_assembly_audit_report(classification.to_dict(), assembly.to_dict())

    assert "=== External Topology Assembly Audit ===" in report
    assert "Products=" in report
    assert "Pending=" in report
    assert "=== Resource Ledger ===" in report
    assert "initial_total=" in report
    assert "consumed_total=" in report
    assert "remaining_total=" in report
    assert "shortage_total=" in report
    assert "=== Front-Half Phase ===" in report
    assert "stage3_phase=external_structural_closure" in report
    assert "initial_counts=" in report
    assert "consumed_counts=" in report
    assert "remaining_counts=" in report
    assert "shortage_counts=" in report
    assert "=== Complete Flex Chains ===" in report
    assert "=== Complete Side Chains ===" in report
    assert "=== Complete Fatty Rings ===" in report
    assert "=== Pending Topologies ===" in report
    assert "Products=0 | Pending=" not in report
    assert "ok=False | reason=external_ring_candidates_pending" in report
    assert "\n   real=-1\n" not in report
    assert "real=-25(-22)-" not in report
    assert "real_gid=" in report
    assert "=== Bad Topologies ===" not in report

    removed_tokens = [
        "suppressed_support_node_ids=",
        "ring_plan_kind=",
        "cycle_kind=",
        "seat_assignments=",
        "closure_target_fragment_id=",
        "closure_target_node_id=",
        "ring_contract:",
        "ring_contract_coverage:",
        "missing_slots=",
        "all_missing_slots=",
        "missing_slots_by_skeleton=",
        "Resource Pool:",
        "Consumed By Product:",
        "Pending And Unclassified:",
        "Classification Rejects:",
        "Claim Conflicts:",
        "True No-Rule Matched:",
        "Debug Appendix",
    ]
    for token in removed_tokens:
        assert token not in report


def test_external_assembly_public_contract_is_minimal() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification)

    public = assembly.to_public_dict()

    assert set(public.keys()) == PUBLIC_ASSEMBLY_KEYS
    assert set(dict(public.get("resource_ledger", {}) or {}).keys()) == PUBLIC_LEDGER_KEYS
    assert "initial_ids_by_su" not in dict(public.get("resource_ledger", {}) or {})
    assert "remaining_ids_by_su" not in dict(public.get("resource_ledger", {}) or {})
    assert "consumed_ids_by_su" not in dict(public.get("resource_ledger", {}) or {})

    for item in list(public.get("products", []) or [])[:20]:
        _assert_public_product(item)
    for item in list(public.get("pending_units", []) or [])[:20]:
        _assert_public_pending(item)

    assert list(public.get("products", []) or [])
    assert str(public.get("reason", "")) == "external_ring_candidates_pending"
    assert not any(str(item.get("real", "")) == "-1" for item in list(public.get("pending_units", []) or []))
    assert all(int(item.get("contract_id", -1)) >= 0 for item in list(public.get("pending_units", []) or []))


def test_external_assembly_diagnostics_keep_only_live_keys() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    diagnostics = dict(assembly.get("diagnostics", {}) or {})
    assert set(diagnostics.keys()) == {
        "explicit_flex_count",
        "external_subphases",
        "active_external_subphase",
        "external_subphase_status",
        "flex_audit",
            "pending_contracts",
            "composition_attempt_failures",
            "contract_definition_conflicts",
        "total_shortage",
        "stage3_phase",
        "residual_closure_product_count",
        "completed_phases",
        "protected_resource_counts",
        "remaining_minimum_flex_gap",
        "remaining_preferred_flex_gap",
        "claimable_counts",
        "allocator_remaining_counts",
        "diagnostic_remaining_counts",
        "unusable_counts",
        "ledger_invariant_errors",
        "ownership_ledger",
        "product_ownership_conflicts",
        "node_coverage",
    }
    assert "topology_rules_implemented" not in diagnostics
    assert "flex_policy_implemented" not in diagnostics
    assert "flex_bounds_enforced" not in diagnostics
    assert "classification_reject_count" not in diagnostics
    assert "classification_claim_conflict_count" not in diagnostics
    assert "classification_support_only_count" not in diagnostics
    assert "classification_true_no_rule_count" not in diagnostics
    assert "classification_open_fragment_count" not in diagnostics
    assert isinstance(list(diagnostics.get("pending_contracts", []) or []), list)
    assert isinstance(dict(diagnostics.get("external_subphase_status", {}) or {}), dict)
    flex_bounds = dict(assembly.get("flex_bounds", {}) or {})
    assert set(flex_bounds.keys()) >= {"min", "max", "target", "enforced"}
    assert int(flex_bounds.get("min", 0) or 0) >= 0
    assert int(flex_bounds.get("max", 0) or 0) >= int(flex_bounds.get("min", 0) or 0)


def test_external_assembly_product_and_pending_exports_have_no_legacy_fields() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    assert list(assembly.get("products", []) or [])
    pending = next(item for item in assembly["pending_units"] if item["pending_kind"] == "e_like_open")

    for key in [
        "topology_template",
        "abstract_topology_partial",
        "abstract_node_id_partial",
        "topology_notation_partial",
        "node_id_notation_partial",
        "actual_topology_su",
        "actual_topology_gid",
        "missing_counts",
        "required_resources",
        "required_su_counts",
        "preferred_product_type",
        "main_kind",
        "diagnostics",
    ]:
        assert key not in pending


def test_flex_target_uses_lower_bound_and_carries_external_shortfall() -> None:
    b_center = make_node(10, 3)
    b_aro = make_node(11, 9)
    b_open = make_node(12, 23)
    connect(b_center, b_aro)
    connect(b_center, b_open)
    nodes = [b_center, b_aro, b_open]
    nodes += [make_node(100 + idx, 9) for idx in range(14)]
    nodes += [make_node(200 + idx, 10) for idx in range(4)]

    phase = build_stage3_phase_evaluation(nodes, classification=classify_topology_units(nodes)).to_dict()
    audit = dict(phase.get("flex_audit", {}) or {})
    assembly = dict(phase.get("assembly_view", {}) or {})
    target = dict(assembly.get("flex_target_plan", {}) or {})

    assert int(audit.get("flex_target", 0)) == int(audit["flex_lower"])
    assert int(target.get("external_candidate_capacity", 0)) == 1
    assert bool(target.get("external_capacity_exhausted", False)) is bool(
        audit.get("external_capacity_exhausted", False)
    )
    assert int(target.get("remaining_minimum_gap", 0)) >= 0


def test_pending_public_contract_exposes_resource_traceability() -> None:
    center = make_node(10, 3)
    aromatic = make_node(11, 9)
    open_node = make_node(12, 23)
    connect(center, aromatic)
    connect(center, open_node)

    result = assemble_topology_units([center, aromatic, open_node], classify_topology_units([center, aromatic, open_node]))
    pending = list(result.to_public_dict().get("pending_units", []) or [])

    assert pending
    assert pending[0]["phase"] == "external_structural_closure"
    assert pending[0]["source_node_ids"]
    assert isinstance(pending[0]["required_counts"], dict)
    assert pending[0]["reason"]


def test_like24_single19d3_with_multiple_11_neighbors_selects_one_anchor() -> None:
    center = make_node(
        414,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    left_11 = make_node(168, 11)
    right_11 = make_node(184, 11)
    terminal_28 = make_node(522, 28)
    connect(center, left_11)
    connect(center, right_11)
    connect(center, terminal_28, fixed=True)

    classification = classify_topology_units([center, left_11, right_11, terminal_28])
    assembler = ExternalTopologyAssembler([center, left_11, right_11, terminal_28], classification)
    view = next(item for item in assembler.views if int(item.fragment_id) == 414)

    assert view.main_labels == ["28", "single19d3", "11"]
    assert view.main_node_ids == [522, 414, 168]
    assert view.main_closed is True
    assert view.actual_branch_stubs == []
    assert all("-11" not in stub.topology_notation for stub in view.actual_branch_stubs)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 414)
    assert 168 in set(fragment.source_node_ids)
    assert 184 not in set(fragment.source_node_ids)
    assert 184 in set(classification.resource_pool["remaining_ids_by_su"]["11"])


def test_like24_extra_11_neighbors_remain_open_resources_not_side_branches() -> None:
    for su_type, terminal_su in ((19, 28),):
        center = make_node(
            1000 + su_type,
            su_type,
            target_hop1_degree=3,
            special_anchor_mode="single",
            target_fixed_anchor_count=1,
        )
        left_11 = make_node(1100 + su_type, 11)
        right_11 = make_node(1200 + su_type, 11)
        terminal = make_node(1300 + su_type, terminal_su)
        connect(center, left_11)
        connect(center, right_11)
        connect(center, terminal, fixed=True)

        classification = classify_topology_units([center, left_11, right_11, terminal])
        assembler = ExternalTopologyAssembler([center, left_11, right_11, terminal], classification)
        view = next(item for item in assembler.views if int(item.fragment_id) == int(center.global_id))

        assert view.main_labels[0] == "28"
        assert view.main_labels == ["28", "single19d3", "11"]
        assert view.main_closed is True
        assert all("11" not in stub.topology_notation for stub in view.actual_branch_stubs)
        fragment = next(item for item in classification.fragments if int(item.fragment_id) == int(center.global_id))
        assert int(right_11.global_id) not in set(fragment.source_node_ids)
        assert int(right_11.global_id) in set(classification.resource_pool["remaining_ids_by_su"]["11"])


def test_flex_candidate_keeps_nonfixed_su28_branch_on_single19d3() -> None:
    left_center = make_node(
        414,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    left_11 = make_node(168, 11)
    left_28 = make_node(522, 28)
    left_connector = make_node(523, 29)
    left_tail = make_node(525, 19, target_hop1_degree=2, special_anchor_mode="single")
    connect(left_center, left_11)
    connect(left_center, left_28)
    connect(left_center, left_connector, fixed=True)
    connect(left_connector, left_tail)

    right_center = make_node(
        416,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    right_11 = make_node(170, 11)
    right_28 = make_node(524, 28)
    right_connector = make_node(527, 29)
    right_tail = make_node(529, 19, target_hop1_degree=2, special_anchor_mode="single")
    connect(right_center, right_11)
    connect(right_center, right_28)
    connect(right_center, right_connector, fixed=True)
    connect(right_connector, right_tail)

    middle_23_a = make_node(474, 23)
    middle_23_b = make_node(475, 23)
    nodes = [
        left_center,
        left_11,
        left_28,
        left_connector,
        left_tail,
        right_center,
        right_11,
        right_28,
        right_connector,
        right_tail,
        middle_23_a,
        middle_23_b,
    ]

    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={
            "enforced": False,
            "scenario": "in_range",
            "selected_candidate_fragment_ids": [],
        },
    )
    flex_views = assembler._available_views(
        lambda item: str(item.preferred_product_type) == "flex_chain"
    )

    assert [view.fragment_id for view in flex_views] == [414, 416]
    assert assembler._available_views(lambda item: assembler._is_like24_ring_fill_view(item)) == []
    assert [stub.topology_notation for stub in flex_views[0].actual_branch_stubs] == ["-28"]
    assert [stub.topology_notation for stub in flex_views[1].actual_branch_stubs] == ["-28"]


def test_public_assemble_prioritizes_b_like_flex_before_ring() -> None:
    left_center = make_node(
        414,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    left_11 = make_node(168, 11)
    left_28 = make_node(522, 28)
    left_connector = make_node(523, 29)
    left_tail = make_node(525, 19, target_hop1_degree=2, special_anchor_mode="single")
    connect(left_center, left_11)
    connect(left_center, left_28)
    connect(left_center, left_connector, fixed=True)
    connect(left_connector, left_tail)

    right_center = make_node(
        416,
        19,
        target_hop1_degree=3,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    right_11 = make_node(170, 11)
    right_28 = make_node(524, 28)
    right_connector = make_node(527, 29)
    right_tail = make_node(529, 19, target_hop1_degree=2, special_anchor_mode="single")
    connect(right_center, right_11)
    connect(right_center, right_28)
    connect(right_center, right_connector, fixed=True)
    connect(right_connector, right_tail)

    nodes = [
        left_center, left_11, left_28, left_connector, left_tail,
        right_center, right_11, right_28, right_connector, right_tail,
        make_node(474, 23), make_node(475, 23),
    ]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range", "selected_candidate_fragment_ids": []},
    )

    assembly = assembler.assemble().to_dict()
    product = next(item for item in assembly["products"] if item["product_kind"] == "flex_chain")

    assert product["actual_topology_su"] == (
        "11-single19d3(-28)-29-single19d2-single19d2-29-single19d3(-28)-11"
    )
    assert len(product["mainline_node_ids"]) == 8
    assert len(product["source_node_ids"]) == len(set(product["source_node_ids"]))
    assert assembly["pending_units"] == []


def test_raw_su24_only_fills_ring_until_like24_minimum() -> None:
    assembler = object.__new__(ExternalTopologyAssembler)
    candidates = [
        {
            "kind": "pool",
            "candidate_id": "raw24",
            "fragment_id": None,
            "pool_su": 24,
            "backbone_span": 1,
            "like24_units": 1,
            "priority": (0,),
        },
        {
            "kind": "pool",
            "candidate_id": "pool23",
            "fragment_id": None,
            "pool_su": 23,
            "backbone_span": 1,
            "like24_units": 0,
            "priority": (1,),
        },
    ]

    selected = assembler._search_ring_middle_segments(
        candidates,
        remaining_span=1,
        claimed_fragment_ids=set(),
        used_candidate_ids=set(),
        current_like24=2,
        min_like24=2,
        max_like24=4,
    )

    assert selected is not None
    assert selected[0]["pool_su"] == 23


def test_double20d3_closed_secondary_fixed_branch_does_not_add_h() -> None:
    center = make_node(430, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    anchor_11 = make_node(217, 11)
    main_connector = make_node(503, 27)
    main_terminal = make_node(116, 6)
    branch_connector = make_node(504, 27)
    branch_terminal = make_node(117, 6)
    connect(center, anchor_11)
    connect(center, main_connector, fixed=True)
    connect(main_connector, main_terminal)
    connect(center, branch_connector, fixed=True)
    connect(branch_connector, branch_terminal)
    nodes = [center, anchor_11, main_connector, main_terminal, branch_connector, branch_terminal]

    classification = classify_topology_units(nodes)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 430)
    assert fragment.class_type == "24_AB_M"
    assert dict(fragment.evidence.get("required_branch_kinds_by_anchor", {}) or {}) == {}

    assembly = assemble_topology_units(nodes, classification).to_dict()
    product = next(item for item in assembly["products"] if 430 in item["source_fragment_ids"])
    assert product["actual_topology_su"] == "11-double20d3(-27-6)-27-6"
    assert "-23-22" not in product["actual_topology_su"]


def test_double20d3_full_mainline_does_not_materialize_untyped_third_branch() -> None:
    center = make_node(438, 20, target_hop1_degree=3, special_anchor_mode="double", target_fixed_anchor_count=2)
    left_connector = make_node(503, 27)
    left_terminal = make_node(128, 6)
    right_connector = make_node(504, 27)
    right_terminal = make_node(129, 6)
    branch_center = make_node(33, 3)
    branch_middle = make_node(455, 23)
    branch_terminal = make_node(217, 11)
    connect(center, left_connector, fixed=True)
    connect(left_connector, left_terminal)
    connect(center, right_connector, fixed=True)
    connect(right_connector, right_terminal)
    connect(center, branch_center)
    connect(branch_center, branch_middle)
    connect(branch_middle, branch_terminal)
    nodes = [
        center, left_connector, left_terminal, right_connector, right_terminal,
        branch_center, branch_middle, branch_terminal,
    ]

    classification = classify_topology_units(nodes)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 438)
    assert fragment.class_type == "24_CD_M_FULL"

    assembly = assemble_topology_units(nodes, classification).to_dict()
    product = next(item for item in assembly["products"] if 438 in item["source_fragment_ids"])
    assert product["actual_topology_su"] == "6-27-double20d3-27-6"
    assert product["actual_topology_gid"] == "128-503-438-504-129"
    assert set(product["source_fragment_ids"]) == {438}
    assert {33, 455, 217}.isdisjoint(set(product["source_node_ids"]))




def test_assembler_rejects_duplicate_global_id_across_products() -> None:
    nodes = [make_node(1, 11), make_node(2, 23), make_node(3, 19)]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    first = TopologyAssemblyProduct(
        product_kind="flex_chain",
        actual_topology_su="11-23",
        actual_topology_gid="1-2",
        source_node_ids=[1, 2],
    )
    second = TopologyAssemblyProduct(
        product_kind="side_chain",
        actual_topology_su="23-19",
        actual_topology_gid="2-3",
        source_node_ids=[2, 3],
    )

    assert assembler._commit_product(first) is True
    assert assembler._commit_product(second) is False
    assert len(assembler.products) == 1
    assert assembler.product_ownership_conflicts[0]["reason"] == "duplicate_topology_ownership"
    assert assembler.product_ownership_conflicts[0]["node_ids"] == [2]


def test_committed_product_resources_are_removed_from_allocator_pool() -> None:
    nodes = [make_node(1, 11), make_node(2, 23), make_node(3, 22), make_node(4, 24)]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    product = TopologyAssemblyProduct(
        product_kind="side_chain",
        actual_topology_su="11-23-22",
        actual_topology_gid="1-2-3",
        source_node_ids=[1, 2, 3],
    )

    assert assembler._commit_product(product) is True
    remaining = assembler.allocator.remaining_ids_by_su
    assert 1 not in list(remaining.get("11", []) or [])
    assert 2 not in list(remaining.get("23", []) or [])
    assert 3 not in list(remaining.get("22", []) or [])
    assert 4 in list(remaining.get("24", []) or [])


def test_committed_product_resource_claims_are_unique() -> None:
    nodes = [make_node(1, 11), make_node(2, 11), make_node(3, 23)]
    classification = classify_topology_units(nodes)
    assembler = ExternalTopologyAssembler(nodes, classification)
    product = TopologyAssemblyProduct(
        product_kind="residual_side_chain",
        actual_topology_su="11-11-23",
        actual_topology_gid="1-2-3",
        source_node_ids=[1, 2, 3],
        claimed_resource_ids={"11": [1, 2, 1, 2], "23": [3, 3]},
    )

    assert assembler._commit_product(product) is True
    assert product.claimed_resource_ids == {"11": [1, 2], "23": [3]}
    assert product.to_public_dict()["claimed_resource_ids"] == {"11": [1, 2], "23": [3]}


def test_pending_render_allows_declared_suppressed_support_nodes() -> None:
    core_node_ids = [182, 310, 405, 505, 569]
    nodes = [
        make_node(182, 11),
        make_node(310, 19),
        make_node(405, 19),
        make_node(505, 28),
        make_node(569, 29),
    ]
    fragment = SimpleNamespace(
        fragment_id=405,
        class_type="24_B",
        claimed_node_ids=list(core_node_ids),
        open_mainline_slots=[],
        branch_contracts=[],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.classification = SimpleNamespace(fragments=[fragment])
    assembler.products = []
    assembler.pending_units = []
    assembler.pending_contracts = []
    assembler.contract_definition_conflicts = []
    view = AssemblyFragmentView(
        unit_id=405,
        fragment_id=405,
        family="like24",
        class_type="24_B",
        topology_template="11-double19d3(-28)-",
        source_node_ids=list(core_node_ids),
        main_labels=["11", "double19d3"],
        main_node_ids=[182, 405],
        canonical_labels=["11", "double19d3"],
        open_left=False,
        open_right=True,
        suppressed_support_node_ids=[310, 569],
        can_ring=True,
        target_roles=["fatty_ring"],
    )
    render = TopologyPathRender(
        topology_notation="11-double19d3(-28)-",
        node_id_notation="182-405(-505)-",
        suppressed_support_node_ids=[310, 569],
    )

    assembler._record_pending(
        unit_id=405,
        pending_kind="pending_ring_candidate",
        views=[view],
        required_counts={},
        available_counts={"11": 13, "23": 8, "24": 1},
        reason="fused_bridgehead_missing",
        partial_render=render,
        actual_render=render,
    )

    pending = assembler.pending_units[0]
    assert pending.core_node_ids == core_node_ids
    assert pending.validation_errors == []
    assert pending.infeasibility_proof["suppressed_support_node_ids"] == [310, 569]


def test_pending_render_claims_visible_branch_stub_nodes() -> None:
    fragment = SimpleNamespace(
        fragment_id=405,
        class_type="24_B",
        claimed_node_ids=[182, 405, 505],
        open_mainline_slots=[],
        branch_contracts=[],
    )
    assembler = object.__new__(ExternalTopologyAssembler)
    assembler.classification = SimpleNamespace(fragments=[fragment])
    assembler.products = []
    assembler.pending_units = []
    assembler.pending_contracts = []
    assembler.contract_definition_conflicts = []
    visible_branch = BranchStub(
        topology_notation="-29",
        node_id_notation="-569",
        source_node_ids=[569],
        branch_kind="actual",
        mode="visible_branch",
    )
    view = AssemblyFragmentView(
        unit_id=405,
        fragment_id=405,
        family="like24",
        class_type="24_B",
        topology_template="11-double19d3(-28)(-29)-",
        source_node_ids=[182, 405, 505],
        main_labels=["11", "double19d3"],
        main_node_ids=[182, 405],
        canonical_labels=["11", "double19d3"],
        open_left=False,
        open_right=True,
        branch_stubs_by_anchor={1: [visible_branch]},
        real_branch_stubs_by_anchor={1: [visible_branch]},
        can_ring=True,
        target_roles=["fatty_ring"],
    )
    render = TopologyPathRender(
        topology_notation="11-double19d3(-28)(-29)-",
        node_id_notation="182-405(-505)(-569)-",
    )

    assembler._record_pending(
        unit_id=405,
        pending_kind="pending_ring_candidate",
        views=[view],
        required_counts={},
        available_counts={},
        reason="fused_bridgehead_missing",
        partial_render=render,
        actual_render=render,
    )

    pending = assembler.pending_units[0]
    assert pending.core_node_ids == [182, 405, 505, 569]
    assert pending.validation_errors == []


def test_resource_ledger_invariant_uses_full_ownership_partition() -> None:
    nodes = [make_node(1, 10), make_node(2, 12), make_node(3, 13), make_node(4, 11)]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    assert str(assembly.get("reason", "")) != "final_resource_ledger_invalid"


def test_ownership_remaining_resource_prevents_false_stage3_completion() -> None:
    nodes = [make_node(1, 11)]
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification).to_dict()

    assert assembly["ok"] is False
    assert assembly["reason"] == "residual_resource_self_closure_pending"
    assert assembly["active_phase"] == "residual_resource_self_closure"


def test_fixture_active_products_have_unique_global_id_ownership() -> None:
    nodes = load_nodes_from_final_csv(CSV_PATH, fixture_infer_fixed_connectors=True)
    assembly = assemble_topology_units(nodes, classify_topology_units(nodes)).to_dict()
    owner_by_node_id = {}

    for product_index, product in enumerate(list(assembly.get("products", []) or [])):
        for node_id in list(product.get("source_node_ids", []) or []):
            assert int(node_id) not in owner_by_node_id, (
                int(node_id),
                owner_by_node_id.get(int(node_id)),
                int(product_index),
            )
            owner_by_node_id[int(node_id)] = int(product_index)

    ownership = dict(assembly.get("ownership_ledger", {}) or {})
    assert list(ownership.get("conflicts", []) or []) == []


def test_su15_d_pair_uses_fixed_partner_once_between_two_f1_caps() -> None:
    center = make_node(332, 15)
    fixed_partner = make_node(336, 15)
    existing_open_23 = make_node(468, 23)
    pool_23 = make_node(472, 23)
    left_11 = make_node(225, 11)
    right_11 = make_node(226, 11)
    connect(center, fixed_partner, fixed=True)
    connect(center, existing_open_23)
    nodes = [center, fixed_partner, existing_open_23, pool_23, left_11, right_11]

    classification = classify_topology_units(nodes)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 332)
    assert fragment.class_type == "D"

    assembly = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()
    product = next(item for item in assembly["products"] if item["source_fragment_ids"] == [332])

    assert product["actual_topology_su"] == "11-23-15-15-23-11"
    gid_tokens = [int(value) for value in product["actual_topology_gid"].split("-")]
    assert len(gid_tokens) == len(set(gid_tokens))
    assert gid_tokens.count(336) == 1


def test_su15_d_pair_nonfixed_partner_is_not_reused_as_open_bridge() -> None:
    center = make_node(326, 15)
    partner = make_node(327, 15)
    existing_open_23 = make_node(452, 23)
    pool_23 = make_node(453, 23)
    left_11 = make_node(223, 11)
    right_11 = make_node(224, 11)
    connect(center, partner)
    connect(center, existing_open_23)
    nodes = [center, partner, existing_open_23, pool_23, left_11, right_11]

    classification = classify_topology_units(nodes)
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 326)
    assert fragment.class_type == "D"

    assembly = ExternalTopologyAssembler(
        nodes,
        classification,
        flex_audit={"enforced": False, "scenario": "in_range"},
    ).assemble().to_dict()
    product = next(item for item in assembly["products"] if item["source_fragment_ids"] == [326])

    assert product["actual_topology_su"] == "11-23-15-15-23-11"
    gid_tokens = [int(value) for value in product["actual_topology_gid"].split("-")]
    assert len(gid_tokens) == len(set(gid_tokens))
    assert gid_tokens.count(327) == 1


def test_su28_g_fragment_keeps_fixed_single19d2_as_existing_mainline() -> None:
    terminal_28 = make_node(519, 28)
    existing_19 = make_node(
        411,
        19,
        target_hop1_degree=2,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    cap_11 = make_node(220, 11)
    connect(existing_19, terminal_28, fixed=True)

    classification = classify_topology_units([existing_19, terminal_28, cap_11])
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 519)
    assert fragment.class_type == "G"
    assert set(fragment.claimed_node_ids) == {411, 519}

    assembly = ExternalTopologyAssembler(
        [existing_19, terminal_28, cap_11],
        classification,
        flex_audit={"enforced": False, "scenario": "not_applicable"},
    ).assemble().to_dict()
    product = next(item for item in assembly["products"] if item["source_fragment_ids"] == [519])

    assert product["actual_topology_su"] == "11-single19d2-28"
    assert product["actual_topology_gid"] == "220-411-519"
    assert set(product["source_node_ids"]) == {220, 411, 519}


def test_su28_g_pending_renders_existing_single19d2_with_terminal() -> None:
    terminal_28 = make_node(520, 28)
    existing_19 = make_node(
        412,
        19,
        target_hop1_degree=2,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    connect(existing_19, terminal_28, fixed=True)

    classification = classify_topology_units([existing_19, terminal_28])
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 520)
    assert fragment.mainline_node_ids == [412, 520]
    assert fragment.open_mainline_slots[0].required_su_sequence == [11]
    assert fragment.open_mainline_slots[0].anchor_node_id == 412
    assembly = ExternalTopologyAssembler(
        [existing_19, terminal_28],
        classification,
        flex_audit={"enforced": False, "scenario": "not_applicable"},
    ).assemble().to_dict()
    pending = next(item for item in assembly["pending_units"] if item["source_fragment_ids"] == [520])

    assert pending["real_topology_partial"] == "-single19d2-28"
    assert pending["real_node_partial"] == "-412-520"
    assert set(pending["source_node_ids"]) == {412, 520}
    assert pending["required_counts"] == {"11": 1}


def test_su32_f_fragment_keeps_fixed_aromatic_8_as_complete_topology() -> None:
    terminal_32 = make_node(620, 32)
    aromatic_8 = make_node(120, 8)
    connect(aromatic_8, terminal_32, fixed=True)

    classification = classify_topology_units([aromatic_8, terminal_32])
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 620)
    assert fragment.class_type == "F"
    assert set(fragment.claimed_node_ids) == {120, 620}

    assembly = ExternalTopologyAssembler(
        [aromatic_8, terminal_32],
        classification,
        flex_audit={"enforced": False, "scenario": "not_applicable"},
    ).assemble().to_dict()
    product = next(item for item in assembly["products"] if item["source_fragment_ids"] == [620])

    assert product["actual_topology_su"] == "8-32"
    assert product["actual_topology_gid"] == "120-620"


def test_su32_g_fragment_keeps_fixed_21d2_as_existing_mainline() -> None:
    terminal_32 = make_node(621, 32)
    existing_21 = make_node(
        421,
        21,
        target_hop1_degree=2,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    cap_11 = make_node(221, 11)
    connect(existing_21, terminal_32, fixed=True)

    classification = classify_topology_units([existing_21, terminal_32, cap_11])
    fragment = next(item for item in classification.fragments if int(item.fragment_id) == 621)
    assert fragment.class_type == "G"
    assert fragment.mainline_node_ids == [421, 621]
    assert fragment.mainline_labels == ["21d2", "32"]
    assert fragment.open_mainline_slots[0].required_su_sequence == [11]
    assert fragment.open_mainline_slots[0].anchor_node_id == 421

    assembly = ExternalTopologyAssembler(
        [existing_21, terminal_32, cap_11],
        classification,
        flex_audit={"enforced": False, "scenario": "not_applicable"},
    ).assemble().to_dict()
    product = next(item for item in assembly["products"] if item["source_fragment_ids"] == [621])

    assert product["actual_topology_su"] == "11-21d2-32"
    assert product["actual_topology_gid"] == "221-421-621"
    assert set(product["source_node_ids"]) == {221, 421, 621}


def test_su32_g_pending_renders_existing_21d2_with_terminal() -> None:
    terminal_32 = make_node(622, 32)
    existing_21 = make_node(
        422,
        21,
        target_hop1_degree=2,
        special_anchor_mode="single",
        target_fixed_anchor_count=1,
    )
    connect(existing_21, terminal_32, fixed=True)

    classification = classify_topology_units([existing_21, terminal_32])
    assembly = ExternalTopologyAssembler(
        [existing_21, terminal_32],
        classification,
        flex_audit={"enforced": False, "scenario": "not_applicable"},
    ).assemble().to_dict()
    pending = next(item for item in assembly["pending_units"] if item["source_fragment_ids"] == [622])

    assert pending["real_topology_partial"] == "-21d2-32"
    assert pending["real_node_partial"] == "-422-622"
    assert set(pending["source_node_ids"]) == {422, 622}
    assert pending["required_counts"] == {"11": 1}


def test_chain_validator_rejects_duplicate_mainline_global_ids() -> None:
    from model.inverse.topology.assembly_external import _validate_chain_product_shape

    reason = _validate_chain_product_shape(
        "flex_chain",
        ["11", "23", "15", "15", "15", "23", "11"],
        [],
        node_ids=[225, 472, 336, 332, 336, 468, 226],
    )

    assert reason == "duplicate_mainline_node_ids:[336]"
