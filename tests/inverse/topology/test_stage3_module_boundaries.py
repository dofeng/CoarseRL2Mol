from __future__ import annotations

import inspect
from pathlib import Path

from model.inverse.topology import assembly_external, workflow
from model.inverse.topology.stage3_state_machine import plan_stage3_state_machine


def test_external_assembler_uses_single_completion_path() -> None:
    source = inspect.getsource(assembly_external.ExternalTopologyAssembler)
    assert "raw_su24_ring_assembly" not in source
    assert "plan_raw_su24_phase" not in source
    assert "raw_su24_role" in source
    assert "def _assemble_residual_resource" not in source
    topology_root = Path(assembly_external.__file__).resolve().parent
    assert not (topology_root / "assembly_residual.py").exists()
    assert not (topology_root / "sequential_resource.py").exists()
    assert "materialized_node_ids" not in source
    assert "edge_plan" not in source


def test_workflow_delegates_to_single_stage3_state_machine() -> None:
    source = inspect.getsource(workflow)
    assert "raw_su24_ring_assembly" not in source
    assert "plan_raw_su24_phase" not in source
    assert callable(plan_stage3_state_machine)


def test_topology_package_has_no_rl_mtcs_dependency() -> None:
    topology_root = Path(assembly_external.__file__).resolve().parent
    sources = "\n".join(path.read_text() for path in topology_root.rglob("*.py"))
    assert "RL_MTCS" not in sources
