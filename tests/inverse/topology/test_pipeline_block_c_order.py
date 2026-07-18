from __future__ import annotations

import ast
import inspect
import textwrap

from model.inverse.inverse_pipeline import InversePipelineV3


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return str(func.attr)
    if isinstance(func, ast.Name):
        return str(func.id)
    return ""


def test_final_outer_cycle_rechecks_block_c_after_mandatory_layer12_rebuild() -> None:
    source = textwrap.dedent(inspect.getsource(InversePipelineV3.infer))
    tree = ast.parse(source)
    outer_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "outer_idx"
    )
    loop_calls = [
        (_call_name(node), int(node.lineno))
        for node in ast.walk(outer_loop)
        if isinstance(node, ast.Call)
    ]
    block_c_lines = [line for name, line in loop_calls if name == "_run_topology_stage3_resource_cycles"]
    layer12_lines = [line for name, line in loop_calls if name == "_run_layer12_cycle"]
    layer3_lines = [line for name, line in loop_calls if name == "layer3_adjust_templates"]

    block_c_lines = sorted(block_c_lines)
    layer12_lines = sorted(layer12_lines)
    layer3_lines = sorted(layer3_lines)

    assert len(block_c_lines) == 2
    assert layer12_lines
    assert layer3_lines
    assert block_c_lines[0] < max(layer12_lines) < block_c_lines[1] < max(layer3_lines)
    assert "int(outer_idx) + 1 == int(outer_cycle_count)" in source
    assert "iteration_offset=int(resource_cfg.get('max_stage3_iterations', 20))" in source

    all_block_c_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_run_topology_stage3_resource_cycles"
    ]
    assert len(all_block_c_calls) == 2


def test_pipeline_does_not_rebuild_layer12_after_final_outer_layer3() -> None:
    source = textwrap.dedent(inspect.getsource(InversePipelineV3.infer))
    tree = ast.parse(source)
    outer_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "outer_idx"
    )
    post_outer_layer12_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_run_layer12_cycle"
        and int(node.lineno) > int(outer_loop.end_lineno)
    ]

    assert post_outer_layer12_calls == []
    assert "using final Post-Layer3 node state" in source
