"""Local-only dashboard with bounded queries and one supervised analysis at a time."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dashboard_queries import db_connect, db_summary, fragment_detail, network, records, table_exists, timeseries
from coarse_network.queries import coarse_network, diagnostics
from pathway.tasks import PathwayTasks
from pipeline_runtime import atomic_json, database_lock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
STATIC_DIR = SRC_DIR / "web"
TASK_ROOT = PROJECT_DIR / "results/dashboard_tasks"
PIPELINE = SRC_DIR / "trajectory_pipeline.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(value: str) -> Path:
    text = str(value).strip()
    if text.startswith(("vscode-remote://", "file://")):
        return Path(unquote(urlparse(text).path)).expanduser()
    return Path(text).expanduser()


def absolute_path(value: str) -> Path:
    path = normalize_path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def validate_database(value: str) -> Path:
    if not value:
        raise ValueError("Database path is required")
    path = absolute_path(value)
    if not path.is_file():
        raise ValueError(f"Database does not exist: {path}")
    with db_connect(path) as connection:
        if not table_exists(connection, "frames"):
            raise ValueError("Not a trajectory database")
    return path


def read_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if not 0 < length <= 65536:
        raise ValueError("Request body must be 1..65536 bytes")
    result = json.loads(handler.rfile.read(length))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CoalReactionNetworkDashboard/3.0"

    def send_json(self, payload, status=200):
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def task_state(self, task_id):
        if not re.fullmatch(r"[a-f0-9]{12}", task_id):
            raise ValueError("Invalid task ID")
        with self.server.task_lock:
            state = dict(self.server.tasks.get(task_id, {}))
        directory = TASK_ROOT / task_id
        if not state:
            for name in ("status.json", "request.json"):
                if (directory / name).is_file():
                    state = json.loads((directory / name).read_text())
                    break
        if not state:
            raise ValueError("Task not found")
        if (directory / "progress.json").is_file():
            state["progress"] = json.loads((directory / "progress.json").read_text())
        if state.get("status") in {"running", "queued"} and task_id not in self.server.processes:
            pid = state.get("pid") or state.get("progress", {}).get("pid")
            try:
                command = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
                alive = str(PIPELINE).encode() in command and str(directory / "progress.json").encode() in command
            except (OSError, TypeError, ValueError):
                alive = False
            if not alive:
                state["status"] = state.get("progress", {}).get("status", "interrupted")
                if state["status"] == "running":
                    state["status"] = "interrupted"
        return state

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                      "/style.css": ("style.css", "text/css"),
                      "/coarse-network.js": ("coarse-network.js", "text/javascript")}
            assets["/lucide.min.js"] = ("lucide.min.js", "text/javascript")
            assets.update({"/pathway-view.js": ("pathway-view.js", "text/javascript"),
                           "/pathway-layout.js": ("pathway-layout.js", "text/javascript"),
                           "/pathway-camera.js": ("pathway-camera.js", "text/javascript"),
                           "/pathway-scene.js": ("pathway-scene.js", "text/javascript"),
                           "/graph-ui.js": ("graph-ui.js", "text/javascript"),
                           "/graph-ui.css": ("graph-ui.css", "text/css")})
            if parsed.path in assets:
                filename, content_type = assets[parsed.path]
                raw = (STATIC_DIR / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; object-src 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(raw)
                return
            query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            if parsed.path in {"/api/pathway/status", "/api/pathway/result"}:
                method = self.server.pathways.status if parsed.path.endswith("status") else self.server.pathways.result
                self.send_json(method(query.get("id", "")))
                return
            if parsed.path == "/api/task":
                self.send_json(self.task_state(query.get("id", "")))
                return
            database = validate_database(query.get("db", ""))
            with self.server.query_slots:
                if parsed.path == "/api/summary":
                    payload = db_summary(database)
                elif parsed.path == "/api/network":
                    payload = network(database, query)
                elif parsed.path == "/api/coarse-network":
                    payload = coarse_network(database, query)
                elif parsed.path == "/api/coarse-diagnostics":
                    payload = diagnostics(database, query)
                elif parsed.path == "/api/fragment":
                    payload = fragment_detail(database, query)
                elif parsed.path == "/api/timeseries":
                    payload = timeseries(database, query)
                elif parsed.path == "/api/species":
                    with db_connect(database) as connection:
                        payload = {"rows": [dict(row) for row in connection.execute("SELECT * FROM species ORDER BY species_id LIMIT 1000")]}
                elif parsed.path.removeprefix("/api/") in {"frames", "events", "tracks", "species-states", "fragments", "lineage"}:
                    payload = records(database, parsed.path.removeprefix("/api/"), query)
                else:
                    self.send_json({"error": "Not found"}, 404)
                    return
            self.send_json(payload)
        except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                raise ValueError("Cross-origin task requests are not allowed")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            payload = read_json_body(self)
            route = urlparse(self.path).path
            if route == "/api/pathway/start":
                self.send_json(self.server.pathways.start(validate_database(payload.get("db", "")), payload), 202)
                return
            if route == "/api/pathway/cancel":
                self.server.pathways.cancel(payload.get("id", ""))
                self.send_json({"status": "cancelling"})
                return
            if route == "/api/cancel":
                task_id = payload.get("task_id", "")
                with self.server.task_lock:
                    process = self.server.processes.get(task_id)
                    if not process or process.poll() is not None:
                        raise ValueError("No running task owned by this server")
                    self.server.tasks[task_id]["status"] = "cancelling"
                    os.killpg(process.pid, signal.SIGTERM)
                self.send_json({"status": "cancelling"})
                return
            if route != "/api/run":
                self.send_json({"error": "Not found"}, 404)
                return
            self.start_task(payload)
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
            self.send_json({"error": str(exc)}, 400)

    def start_task(self, payload):
        raw_input = str(payload.get("input", "")).strip()
        if not raw_input:
            raise ValueError("Input path is required")
        source = absolute_path(raw_input)
        if not source.exists() and not any(character in str(source) for character in "*?["):
            raise ValueError("Input path does not exist")
        database = absolute_path(payload.get("database") or "results/reaction_network_v3/network.sqlite")
        workers = int(payload.get("workers", 2))
        inflight = int(payload.get("max_inflight", 2 * workers))
        budget = int(payload.get("memory_limit_mib", 3072))
        if not 1 <= workers <= 8 or not workers <= inflight <= 16 or not 128 <= budget <= 4096:
            raise ValueError("6GB profile: workers 1..8, workers <= inflight <=16, memory 128..4096 MiB")
        stage = payload.get("stage", "all")
        if stage not in {"all", "frames", "transitions", "postprocess"}:
            raise ValueError("Unknown processing stage")
        with self.server.task_lock:
            if any(process.poll() is None for process in self.server.processes.values()):
                raise ValueError("One analysis is already running; wait or cancel it before starting another")
            with database_lock(database):
                if database.is_file():
                    with db_connect(database) as connection:
                        version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                        if not version or version[0] != "3":
                            raise ValueError("Choose a NEW database path: legacy parser results cannot be resumed")
            task_id = uuid.uuid4().hex[:12]
            directory = TASK_ROOT / task_id
            directory.mkdir(parents=True)
            args = [sys.executable, "-u", str(PIPELINE), str(source), "--database", str(database),
                    "--workers", str(workers), "--max-inflight", str(inflight),
                    "--stage", stage,
                    "--memory-limit-mib", str(budget), "--progress-file", str(directory / "progress.json")]
            options = {"pattern": "--pattern", "frame_order": "--frame-order", "cutoff": "--cutoff",
                       "temporal_mode": "--temporal-mode", "minimum_lifetime_frames": "--minimum-lifetime-frames",
                       "reversal_window_frames": "--reversal-window-frames", "expected_frame_step": "--expected-frame-step"}
            for key, flag in options.items():
                if payload.get(key) not in (None, ""):
                    args.extend([flag, str(payload[key])])
            for key, flag in {"retry_failed": "--retry-failed", "build_tracks": "--build-tracks",
                              "export_states": "--export-states"}.items():
                if payload.get(key):
                    args.append(flag)
            if payload.get("output_dir"):
                args.extend(["--output-dir", str(absolute_path(payload["output_dir"]))])
            if payload.get("config"):
                args.extend(["--config", str(absolute_path(payload["config"]))])
            state = dict(task_id=task_id, status="running", database=str(database), started_at=now(),
                         request=payload, command=args)
            with (directory / "pipeline.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(args, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONPATH": str(SRC_DIR)}, start_new_session=True)
            state["pid"] = process.pid
            atomic_json(directory / "request.json", state)
            self.server.tasks[task_id] = state
            self.server.processes[task_id] = process
        def watch():
            code = process.wait()
            with self.server.task_lock:
                cancelled = state["status"] == "cancelling"
                status = "cancelled" if cancelled else "completed" if code == 0 else "partial" if code == 2 else "failed"
                state.update(status=status, return_code=code, finished_at=now())
                atomic_json(directory / "status.json", state)
                self.server.processes.pop(task_id, None)
        threading.Thread(target=watch, daemon=True).start()
        self.send_json({"task_id": task_id, "database": str(database), "status": "running"}, 202)

    def log_message(self, *args):
        pass


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler):
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("Dashboard must listen on a loopback address; use port forwarding")
        super().__init__(address, handler)
        self.tasks = {}
        self.processes = {}
        self.task_lock = threading.Lock()
        self.query_slots = threading.BoundedSemaphore(2)
        self.pathways = PathwayTasks(PROJECT_DIR / "results/pathway_tasks")

    def server_close(self):
        self.pathways.stop.set()
        super().server_close()


def start_dashboard(host="127.0.0.1", port=8765):
    server = DashboardServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, name="reaction-dashboard", daemon=True)
    thread.start()
    return server, thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = DashboardServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
