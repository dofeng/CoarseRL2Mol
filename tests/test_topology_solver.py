from RL_MTCS.allocator_models import ChainSpec, chain_23_count
from RL_MTCS.branch_assembler import BranchTopologyAssembler


def _constraints():
    return {"flex_lower": 0, "flex_upper": 99, "flex_window_ok": True}


def test_residual_conversion_plan_is_not_materialized_until_applied():
    result = BranchTopologyAssembler(
        [],
        [],
        {"11": 3, "22": 2, "23": 0, "24": 0, "25": 0},
        constraints=_constraints(),
    ).assemble()

    diag = result.diagnostics
    assert diag["topology_plan_feasible"] is True
    assert diag["topology_materialized_ok"] is False
    assert diag["materialization_required"] is True
    assert diag["resource_ledger"]["remaining"] == {
        "11": 0,
        "22": 0,
        "23": 0,
        "24": 0,
        "25": 0,
    }
    assert [row["op"] for row in diag["planned_conversions"]] == ["22->23", "11->23"]
    assert [chain.composition for chain in result.closed_topology_specs] == [
        [11, 23, 23, 23, 22],
    ]


def test_residual_growth_replaces_source_chain_without_double_counting():
    ordinary = [
        ChainSpec(
            chain_type="side",
            composition=[11, 23, 22],
            origin_type="base",
            metadata={"actual_topology_su": [11, 23, 22]},
        )
    ]
    result = BranchTopologyAssembler(
        [],
        ordinary,
        {"11": 2, "22": 2, "23": 6, "24": 0, "25": 0},
        constraints=_constraints(),
    ).assemble()

    assert result.diagnostics["topology_materialized_ok"] is True
    assert result.diagnostics["materialization_required"] is False
    assert result.ordinary_chains[0].composition == [11, 23, 23, 23, 23, 23, 22]
    assert [chain.composition for chain in result.closed_topology_specs] == [
        [11, 23, 22],
    ]
    assert result.diagnostics["resource_ledger"]["consumed"] == {
        "11": 2,
        "22": 2,
        "23": 6,
        "24": 0,
        "25": 0,
    }
    assert result.diagnostics["resource_ledger"]["remaining"] == {
        "11": 0,
        "22": 0,
        "23": 0,
        "24": 0,
        "25": 0,
    }


def test_residual_chains_respect_six_su23_limit():
    result = BranchTopologyAssembler(
        [],
        [],
        {"11": 0, "22": 5, "23": 20, "24": 0, "25": 0},
        constraints=_constraints(),
    ).assemble()

    diag = result.diagnostics
    assert diag["topology_plan_feasible"] is True
    assert diag["chain_23_limit_violations"] == []
    assert all(chain_23_count(chain) <= 6 for chain in result.closed_topology_specs)
    assert diag["resource_ledger"]["remaining"] == {
        "11": 0,
        "22": 0,
        "23": 0,
        "24": 0,
        "25": 0,
    }
