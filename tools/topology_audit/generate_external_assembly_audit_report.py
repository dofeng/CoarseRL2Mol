from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.inverse.topology.classification.classifier import classify_topology_units
from model.inverse.layer4.inverse_layer4_block_c import run_block_c_resource_adjustment_loop_impl
from model.inverse.topology.conversion_solver import apply_stage3_node_adjustments
from model.inverse.topology.phase_eval import assemble_topology_units
from model.inverse.topology.workflow import plan_stage3_resource_adjustments
from tests.inverse.topology.csv_fixture_loader import YM4_FINAL_NODES_CSV, load_nodes_from_final_csv
from tools.topology_audit.assembly_report import build_external_assembly_audit_report


DEFAULT_FIXTURE_CSV = YM4_FINAL_NODES_CSV

PYTEST_COMMANDS: list[list[str]] = [
    [sys.executable, "-m", "pytest", "tests/inverse/topology/test_classification_like24.py", "-q"],
    [sys.executable, "-m", "pytest", "tests/inverse/topology/test_external_assembly.py", "-q"],
    [sys.executable, "-m", "pytest", "tests/inverse/topology/test_inverse_pipeline_topology_output.py", "-q"],
    [sys.executable, "-m", "pytest", "tests/inverse/topology", "-q"],
]


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an external topology assembly audit report with pytest results.",
    )
    parser.add_argument(
        "--fixture-csv",
        type=Path,
        default=DEFAULT_FIXTURE_CSV,
        help="Path to final_nodes_by_global_id.csv fixture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test_results" / f"external_assembly_audit_report_{datetime.now().strftime('%Y%m%d')}.txt",
        help="Output report path.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Generate the audit section only and skip pytest execution.",
    )
    return parser.parse_args()


def _run_command(command: Sequence[str]) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration_seconds = time.perf_counter() - start
    return CommandResult(
        command=list(command),
        returncode=int(completed.returncode),
        duration_seconds=duration_seconds,
        stdout=str(completed.stdout or "").strip(),
        stderr=str(completed.stderr or "").strip(),
    )


def _command_results_lines(results: Sequence[CommandResult]) -> list[str]:
    lines = ["=== Pytest Results ==="]
    if not list(results or []):
        lines.append("(skipped)")
        return lines

    for index, result in enumerate(results, start=1):
        status = "passed" if int(result.returncode) == 0 else "failed"
        lines.append(f"{index}. status={status}")
        lines.append(f"   command={' '.join(result.command)}")
        lines.append(f"   exit_code={result.returncode}")
        lines.append(f"   duration_seconds={result.duration_seconds:.2f}")
        if result.stdout:
            lines.append("   stdout:")
            for line in result.stdout.splitlines():
                lines.append(f"   {line}")
        else:
            lines.append("   stdout=(empty)")
        if result.stderr:
            lines.append("   stderr:")
            for line in result.stderr.splitlines():
                lines.append(f"   {line}")
    return lines


def _metadata_lines(fixture_csv: Path, output: Path) -> list[str]:
    return [
        "=== Report Metadata ===",
        f"generated_at={datetime.now().astimezone().isoformat()}",
        f"repo_root={REPO_ROOT}",
        f"output_path={output.resolve()}",
        f"fixture_csv={fixture_csv.resolve()}",
        f"python={sys.executable}",
        f"python_version={platform.python_version()}",
        f"conda_env={os.environ.get('CONDA_DEFAULT_ENV', '')}",
    ]


def _build_audit_section(fixture_csv: Path) -> str:
    nodes = load_nodes_from_final_csv(str(fixture_csv), fixture_infer_fixed_connectors=True)
    classification = classify_topology_units(nodes)
    assembly = assemble_topology_units(nodes, classification)
    static_report = build_external_assembly_audit_report(classification.to_dict(), assembly.to_dict())

    class _AuditAdjuster:
        @staticmethod
        def plan_topology_resource_adjustments(**kwargs):
            return plan_stage3_resource_adjustments(**kwargs)

        @staticmethod
        def apply_topology_resource_adjustments(**kwargs):
            return apply_stage3_node_adjustments(kwargs["nodes"], kwargs["H"], kwargs["plan"])

    H = torch.zeros(33, dtype=torch.long)
    for node in nodes:
        H[int(node.su_type)] += 1
    workflow = run_block_c_resource_adjustment_loop_impl(
        _AuditAdjuster(),
        H,
        nodes,
        E_target=None,
        rebuild_fn=lambda **kwargs: (nodes, kwargs["H_seed"], {}, 0.0),
        validate_fn=lambda **kwargs: (True, "ok"),
        max_primary_adjustments_per_stage=10,
    )
    workflow_lines = [
        "=== Block-C Resource Loop Audit ===",
        f"accepted_cycles={int(workflow.accepted_cycles)} | stop_reason={str(workflow.stop_reason)} | ok={bool(workflow.ok)}",
    ]
    for cycle in list(workflow.cycles or []):
        workflow_lines.append(
            f"cycle={int(cycle.cycle)} | phase={str(cycle.phase)} | applied={bool(cycle.applied)} | "
            f"accepted={bool(cycle.accepted)} | acceptance_reason={str(cycle.acceptance_reason)} | "
            f"stop_reason={str(cycle.stop_reason)}"
        )
    return static_report + "\n\n" + "\n".join(workflow_lines)


def main() -> int:
    args = _parse_args()
    fixture_csv = Path(args.fixture_csv).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command_results: list[CommandResult] = []
    if not bool(args.skip_pytest):
        for command in PYTEST_COMMANDS:
            command_results.append(_run_command(command))

    audit_section = _build_audit_section(fixture_csv)

    lines: list[str] = []
    lines.extend(_metadata_lines(fixture_csv, output))
    lines.append("")
    lines.extend(_command_results_lines(command_results))
    lines.append("")
    lines.append(audit_section)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0 if all(int(item.returncode) == 0 for item in command_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
