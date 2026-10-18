"""Persistent runs with an explicit prepare → ready → start lifecycle."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
from uuid import uuid4

from .recorder import TraceReader, json_text

ENGINE_PROJECT = Path(__file__).resolve().parents[1]


class RunConflict(RuntimeError):
    pass


class RunManager:
    def __init__(self, project=None, defaults=None):
        self.project = Path(project or ENGINE_PROJECT).resolve()
        self.registry = self.project / "assembly_viewer" / "runs" / "registry.sqlite3"
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, spec TEXT NOT NULL)")
        self.defaults = defaults or {}
        self.processes = {}
        self.lock = threading.RLock()
        self.export_lock = threading.Lock()

    def db(self):
        return sqlite3.connect(self.registry)

    def metadata(self):
        from RL_MTCS import AssemblyConfig
        directory = self.project / "test_results"
        samples = [p.name for p in sorted(directory.iterdir()) if
                   (p / "final_outputs" / "final_nodes_by_global_id.csv").is_file() and
                   (p / "final_outputs" / "final_topology_specs.csv").is_file()] if directory.is_dir() else []
        return dict(project=str(self.project), samples=samples, python=sys.executable,
                    defaults={**asdict(AssemblyConfig(forward_max_rollbacks=3, seed=27)), **self.defaults})

    def get(self, run_id):
        with self.db() as db:
            row = db.execute("SELECT spec FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("运行记录不存在")
        return json.loads(row[0])

    @staticmethod
    def trace(spec):
        return Path(spec["output_dir"]) / "process_trace"

    def save(self, spec):
        path = Path(spec["output_dir"]) / "run.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json_text(spec), encoding="utf-8")
        temporary.replace(path)
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES(?,?)", (spec["id"], json_text(spec)))

    @staticmethod
    def runtime(value):
        executable = shutil.which(str(value))
        if not executable or not Path(executable).is_file():
            raise ValueError("Python 可执行文件不存在")
        # Preserve virtualenv symlink spelling: resolving it can change sys.prefix.
        return str(Path(executable).absolute())

    def spawn(self, spec, command):
        args = [spec["python"], "-u", "-m", "assembly_viewer", command, "--spec",
                str(Path(spec["output_dir"]) / "run.json")]
        spec["command"] = shlex.join(args)
        spec["worker_kind"] = command
        self.save(spec)
        with (Path(spec["output_dir"]) / "run.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=ENGINE_PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.processes[spec["id"]] = process
        spec["pid"] = process.pid
        self.save(spec)

    def prepare(self, payload):
        from RL_MTCS import AssemblyConfig
        sample = payload.get("sample", "YM-3")
        if not isinstance(sample, str) or not sample or Path(sample).name != sample or sample in {".", ".."}:
            raise ValueError("样品名称必须是单个文件夹名称")
        config = asdict(AssemblyConfig(**payload.get("config", self.metadata()["defaults"])))
        root = self.project / "test_results" / sample
        nodes = Path(payload.get("nodes_csv") or root / "final_outputs" / "final_nodes_by_global_id.csv").expanduser().resolve()
        topology = Path(payload.get("topology_csv") or root / "final_outputs" / "final_topology_specs.csv").expanduser().resolve()
        for path in (nodes, topology):
            if not path.is_file():
                raise FileNotFoundError(f"输入文件不存在: {path}")
        runtime = self.runtime(payload.get("python") or sys.executable)
        for key in ("spectrum_csv", "element_composition_csv", "g2s_checkpoint"):
            if config.get(key):
                config[key] = str(Path(config[key]).expanduser().resolve())
        with self.lock:
            output_root = Path(payload.get("output_root") or root / "rl_mtcs_nmr").expanduser().resolve()
            output = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output.mkdir(parents=True, exist_ok=False)
            spec = dict(id=uuid4().hex[:16], sample=sample, created_at=datetime.now(timezone.utc).isoformat(),
                        nodes_csv=str(nodes), topology_csv=str(topology), output_dir=str(output),
                        python=runtime, config=config)
            self.spawn(spec, "prepare")
            return self.info(spec["id"])

    def start(self, run_id, payload):
        from RL_MTCS import AssemblyConfig
        from .resources import provenance
        with self.lock:
            spec = self.get(run_id)
            with TraceReader(self.trace(spec)) as reader:
                manifest = reader.manifest()
            if manifest["status"] != "ready" or spec["worker_kind"] == "run":
                raise RunConflict("仅可启动已读取、未开始的记录；不能重复启动")
            config = AssemblyConfig(**payload.get("config", spec["config"]))
            current = provenance(spec["nodes_csv"], spec["topology_csv"], config)
            if current["inputs"] != manifest["inputs"] or current["engine_sha256"] != manifest["engine_sha256"]:
                raise RunConflict("输入文件或装配代码已变化，请重新读取拓扑")
            spec["config"] = asdict(config)
            if payload.get("python"):
                spec["python"] = self.runtime(payload["python"])
            self.spawn(spec, "run")
            return self.info(run_id)

    def info(self, run_id):
        spec = self.get(run_id)
        info = dict(spec, status="preparing", ready=False)
        trace = self.trace(spec)
        if (trace / "manifest.json").is_file():
            with TraceReader(trace) as reader:
                manifest = reader.manifest()
                info.update(status=manifest["status"], ready=True, latest_sequence=reader.latest_sequence())
                for key in ("completion_report", "legal", "all_complete", "error", "current_phase", "resource_counts"):
                    if key in manifest:
                        info[key] = manifest[key]
        process = self.processes.get(run_id)
        if process is not None:
            code = process.poll()
            info["exit_code"] = code
            if code is not None and code != 0:
                info.update(status="error", error=info.get("error") or dict(message="工作进程异常退出，请查看运行日志", exit_code=code))
            elif spec.get("worker_kind") == "run" and code is None and info["status"] == "ready":
                info["status"] = "starting"
        elif info["status"] in {"running", "preparing"} or (info["status"] == "ready" and spec.get("worker_kind") == "run"):
            # A restarted server can observe a detached worker. Validate its PID
            # against this exact specification; never treat a reused PID as ours.
            pid = spec.get("pid")
            command_line = Path(f"/proc/{pid}/cmdline") if pid else None
            expected = str(Path(spec["output_dir"]) / "run.json").encode()
            try:
                alive = command_line is not None and expected in command_line.read_bytes()
            except OSError:
                alive = False
            if alive and info["status"] == "ready":
                info["status"] = "starting"
            if not alive:
                info.update(status="interrupted", error=dict(message="工作进程已不存在；保留已记录过程，不自动重新开始"))
        # Never resume or launch a new search when opening history after restart.
        return info

    def list(self):
        with self.db() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM runs ORDER BY rowid DESC")]
        return [self.info(run_id) for run_id in ids]

    def reopen(self, path):
        trace = Path(path).expanduser().resolve()
        if trace.name != "process_trace" and (trace / "process_trace").is_dir():
            trace /= "process_trace"
        with TraceReader(trace) as reader:
            manifest = reader.manifest()
        output = trace.parent
        if (output / "run.json").is_file():
            spec = json.loads((output / "run.json").read_text(encoding="utf-8"))
            spec["output_dir"] = str(output)
        else:
            spec = dict(id=uuid4().hex[:16], output_dir=str(output), sample=manifest.get("sample", "已有记录"),
                        created_at=manifest["created_at"], config=manifest.get("config", {}), worker_kind="imported")
        # Imported files are read-only; keep their specification in the registry.
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES(?,?)", (spec["id"], json_text(spec)))
        return self.info(spec["id"])

    def log_tail(self, run_id):
        path = Path(self.get(run_id)["output_dir"]) / "run.log"
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 16384))
            return handle.read().decode("utf-8", errors="replace")
