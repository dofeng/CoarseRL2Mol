"""Separate input preparation and assembly, reusing original RL entry points."""
from dataclasses import asdict
import json
from pathlib import Path

from .recorder import TraceRecorder, json_text
from .resources import checksum, provenance, register_input, resource_catalog


def load_spec(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(path):
    from RL_MTCS import AssemblyConfig
    from RL_MTCS.io.input import initialize
    spec = load_spec(path)
    config = AssemblyConfig(**spec["config"])
    meta = provenance(spec["nodes_csv"], spec["topology_csv"], config)
    meta.update(status="preparing", sample=spec["sample"], run_id=spec["id"])
    with TraceRecorder(Path(spec["output_dir"]) / "process_trace", meta) as recorder:
        recorder.emit("input.loading", scope="input", phase="input")
        try:
            initialized = initialize(spec["nodes_csv"], spec["topology_csv"])
            register_input(recorder, initialized["state"])
            recorder.update_manifest(status="ready", input_diagnostics=initialized["diagnostics"])
            print("输入已读取，等待点击开始组装。", flush=True)
        except BaseException as exc:
            recorder.emit("run.error", data=dict(type=type(exc).__name__, message=str(exc)))
            recorder.finish("error", error=dict(type=type(exc).__name__, message=str(exc)))
            raise


def run(path):
    from RL_MTCS import AssemblyConfig
    from RL_MTCS.runner import MCTSRunner
    from .adapter import _run_observed_runner
    spec = load_spec(path)
    with TraceRecorder.begin_prepared(Path(spec["output_dir"]) / "process_trace") as recorder:
        try:
            config = AssemblyConfig(**spec["config"])
            current = provenance(spec["nodes_csv"], spec["topology_csv"], config)
            if current["inputs"] != recorder.info["inputs"] or current["engine_sha256"] != recorder.info["engine_sha256"]:
                raise ValueError("输入文件或装配代码在读取后发生变化，请重新读取后开始")
            recorder.update_manifest(config=asdict(config), command=spec["command"])
            recorder.emit("run.started")
            runner = MCTSRunner(spec["nodes_csv"], topology_specs_csv=spec["topology_csv"], config=config)
            if json_text(resource_catalog(runner.initial_state)) != (recorder.directory / "resources.json").read_text(encoding="utf-8"):
                raise ValueError("运行时的拓扑资源与读取预览不一致，请重新读取")
            return _run_observed_runner(runner, recorder, spec["output_dir"])
        except BaseException as exc:
            recorder.emit("run.error", data=dict(type=type(exc).__name__, message=str(exc)))
            recorder.finish("error", error=dict(type=type(exc).__name__, message=str(exc)))
            raise
