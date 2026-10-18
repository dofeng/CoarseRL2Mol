"""Disk-backed, branch-aware display states and ordered observation events."""

from __future__ import annotations

from collections import Counter, OrderedDict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

SCHEMA_VERSION = 2

TIMELINE_KINDS = {
    "formal": ("input.ready", "formal.initialized", "stage.start", "formal.connection",
               "checkpoint.accepted", "stage.complete", "run.finished", "run.error"),
    "live": ("input.ready", "formal.initialized", "stage.start", "search.start", "search.visit",
             "search.result", "search.selected", "checkpoint.assessment", "checkpoint.rollback",
             "checkpoint.accepted", "formal.connection", "stage.complete", "forward.start",
             "forward.complete", "run.finished", "run.error"),
    "forward": ("forward.start", "forward.connection", "forward.flex_finished", "forward.phase.start",
                "forward.phase.no_actions", "forward.complete"),
}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def digest(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def difference(before, after, path=()):
    """JSON-only edits, including deletions and changes of existing identities."""
    if isinstance(before, dict) and isinstance(after, dict):
        edits = [[list((*path, key)), "delete"] for key in before.keys() - after.keys()]
        for key, value in after.items():
            if key not in before:
                edits.append([list((*path, key)), "set", value])
            else:
                edits.extend(difference(before[key], value, (*path, key)))
        return edits
    return [[list(path), "set", after]] if before != after else []


def _apply_difference_owned(result, edits):
    """Apply decoded edits to a private projection without copying each ancestor."""
    for path, operation, *values in edits:
        if not path:
            result = values[0]
            continue
        target = result
        for key in path[:-1]:
            target = target[key]
        if operation == "delete":
            del target[path[-1]]
        elif operation == "set":
            target[path[-1]] = values[0]
        else:
            raise ValueError(f"unknown display delta: {operation}")
    return result


def apply_difference(before, edits):
    # Public helper must not mutate either caller-owned argument.
    return _apply_difference_owned(copy.deepcopy(before), copy.deepcopy(edits))


def project_state(state):
    """Read-only geometry and legality, without repeating full shape diagnostics.

    Original stage/final outputs still run their full audit. Intermediate
    frames need no hull/void scan of widely separated staging components.
    """
    from RL_MTCS.reporting.completion import completion_report
    from RL_MTCS.mdp.sites import aromatic_perimeter_ids, hetero_staging_ids
    display = copy.copy(state)
    display.action_history = ()
    # Audit the original history (including substitutions), not the empty
    # display history. This private view never mutates the search state.
    display.audit = lambda: state.audit(diagnostics=False)
    raw = display.to_dict()
    raw["audit"].update(diagnostics_scope="legality_only", shape_valid=None)
    # Action history is recorded on events, rather than copied into every frame.
    raw.pop("action_history", None)
    raw["nodes"] = {str(node["node_id"]): node for node in raw["nodes"]}
    raw["edges"] = {json_text([edge["source_node_id"], edge["target_node_id"],
                                edge["kind"], edge["contract_id"]]): edge
                    for edge in raw["edges"]}
    raw["units"] = {unit["unit_id"]: unit for unit in raw["units"]}
    raw.update(stage=state.stage, used_ports=sorted(state.used_ports),
               satisfied_contracts=sorted(state.satisfied_contracts),
               backbone_hull=[list(point) for point in state.backbone_hull],
               task_inventory=[list(row) for row in state.task_inventory],
               completion_report=completion_report(state))
    raw["occupied_axial"]={f"{q},{r}":gid for (q,r),gid in state.occupied_axial.items()}
    protected=set(hetero_staging_ids(state))
    for edge in state.edges:
        if edge.contract_id:
            protected.update((edge.source_node_id,edge.target_node_id))
    perimeter={gid for unit in state.problem.units if unit.phase=="aromatic"
               and state.is_placed(unit.unit_id) for gid in aromatic_perimeter_ids(unit)}
    raw["clearance_site_ids"]=sorted(gid for gid in perimeter if state.current_su_types.get(gid)==13
        and gid not in protected and f"aromatic_site:{gid}" not in state.used_ports
        and state.node_axial.get(gid) in state.occupied_axial)
    raw["counts"] = {
        "nodes": len(state.node_axial), "edges": len(state.edges),
        "components": state.component_count(),
        "placed_units": dict(Counter(unit["stage"] for unit in raw["units"].values()
                                     if unit["placed"])),
        "su_types": dict(Counter(str(node["su_type"]) for node in raw["nodes"].values())),
    }
    return raw


class TraceReader:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "trace.sqlite3"
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        # Each HTTP request gets its own read-only connection; no writer locks.
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        if self.manifest().get("schema_version") != SCHEMA_VERSION:
            self.connection.close()
            raise ValueError("此记录不是当前单画布流程格式，请使用当前记录入口重新生成；不启用旧格式兼容")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def manifest(self):
        row = self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        return json.loads(row[0]) if row else {}

    def state(self, state_id):
        chain = []
        current = int(state_id)
        while current is not None:
            row = self.connection.execute("SELECT * FROM states WHERE id=?", (current,)).fetchone()
            if row is None:
                raise KeyError(f"state {current} is absent")
            chain.append(row)
            if row["kind"] == "snapshot":
                result = json.loads(row["payload"])
                break
            current = row["parent_id"]
        else:
            raise ValueError("state delta has no snapshot ancestor")
        for row in reversed(chain[:-1]):
            result = _apply_difference_owned(result, json.loads(row["payload"]))
        if digest(result) != chain[0]["digest"]:
            raise ValueError(f"display state {state_id} failed its checksum")
        return result

    def events(self, after=0, limit=250, scope=None, before=None, mode=None):
        conditions, arguments = ["seq > ?"], [int(after)]
        if scope and scope != "all":
            conditions.append("scope=?")
            arguments.append(scope)
        if before is not None:
            conditions.append("seq<=?")
            arguments.append(int(before))
        if mode:
            if mode not in TIMELINE_KINDS:
                raise ValueError("未知过程范围")
            kinds = TIMELINE_KINDS[mode]
            conditions.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            arguments.extend(kinds)
        arguments.append(max(1, min(int(limit), 1000)))
        rows = self.connection.execute(
            f"SELECT * FROM events WHERE {' AND '.join(conditions)} ORDER BY seq LIMIT ?", arguments)
        return [self.event_dict(row) for row in rows]

    @staticmethod
    def event_dict(row):
        result = dict(row)
        result["data"] = json.loads(result.pop("payload"))
        return result

    def event(self, seq):
        row = self.connection.execute("SELECT * FROM events WHERE seq=?", (int(seq),)).fetchone()
        if row is None:
            raise KeyError(seq)
        return self.event_dict(row)

    def latest_sequence(self):
        return self.connection.execute("SELECT coalesce(max(seq),0) FROM events").fetchone()[0]

    def phases(self):
        return [dict(row) for row in self.connection.execute("SELECT * FROM phases ORDER BY start_seq")]

    def position(self, at, mode):
        if mode not in TIMELINE_KINDS:
            raise ValueError("未知过程范围")
        kinds = TIMELINE_KINDS[mode]
        row = self.connection.execute(
            "SELECT * FROM events WHERE seq<=? AND kind IN (" + ",".join("?" for _ in kinds)
            + ") ORDER BY seq DESC LIMIT 1", (int(at), *kinds)).fetchone()
        return self.event_dict(row) if row else None

    def resources(self):
        return json.loads((self.directory / "resources.json").read_text(encoding="utf-8"))


class TraceRecorder(TraceReader):
    """Batched durable events with a bounded cache of display projections.

    Cache entries contain JSON only, never live molecular states/search nodes.
    Every event is kept. Publication happens at boundaries and at most every
    200 ms while events are produced; closing always commits remaining rows.
    """

    projection_cache_limit = 4
    publish_interval = 0.2
    publish_events = frozenset({"run.started", "formal.initialized", "search.start",
        "search.selected", "checkpoint.assessment", "checkpoint.rollback", "checkpoint.accepted",
        "forward.start", "forward.complete", "run.error", "run.finished"})

    def _setup_buffers(self):
        self._projections = OrderedDict()
        self._last_publish = time.monotonic()

    def _write(self, sql, arguments=()):
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN")
        return self.connection.execute(sql, arguments)

    def flush(self):
        """Atomically expose states, transitions, and ordered events to readers."""
        if self.connection.in_transaction:
            self.connection.commit()
        self._last_publish = time.monotonic()

    def close(self):
        try:
            self.flush()
        finally:
            self._projections.clear()
            super().close()

    def _remember(self, state_id, projection):
        self._projections[state_id] = projection
        self._projections.move_to_end(state_id)
        while len(self._projections) > self.projection_cache_limit:
            self._projections.popitem(last=False)

    def _projection(self, state_id):
        cached = self._projections.get(state_id)
        if cached is None:
            cached = super().state(state_id)
            self._remember(state_id, cached)
        else:
            self._projections.move_to_end(state_id)
        return cached

    def state(self, state_id):
        # Returning a cache entry would let callers corrupt subsequent deltas.
        return copy.deepcopy(self._projection(int(state_id)))

    def __init__(self, directory, manifest=None, snapshot_interval=24):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "trace.sqlite3"
        # Never silently overwrite an earlier experiment.
        with self.path.open("xb"):
            pass
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE states(id INTEGER PRIMARY KEY,digest TEXT UNIQUE NOT NULL,
                parent_id INTEGER REFERENCES states(id),kind TEXT NOT NULL,depth INTEGER NOT NULL,
                payload TEXT NOT NULL);
            CREATE TABLE events(seq INTEGER PRIMARY KEY,elapsed REAL NOT NULL,scope TEXT NOT NULL,
                kind TEXT NOT NULL,phase TEXT,round_id TEXT,attempt INTEGER,state_id INTEGER REFERENCES states(id),
                parent_state_id INTEGER REFERENCES states(id),payload TEXT NOT NULL);
            CREATE INDEX event_scope ON events(scope,seq);
            CREATE INDEX event_kind ON events(kind,seq);
            CREATE TABLE transitions(key TEXT PRIMARY KEY,parent_id INTEGER NOT NULL,
                state_id INTEGER NOT NULL,action_json TEXT NOT NULL);
            CREATE TABLE phases(phase TEXT PRIMARY KEY,start_seq INTEGER NOT NULL,start_state_id INTEGER,
                end_seq INTEGER,end_state_id INTEGER);
        """)
        self.started = time.monotonic()
        self._setup_buffers()
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.info = {"schema_version": SCHEMA_VERSION, "status": "running",
                     "created_at": datetime.now(timezone.utc).isoformat(), "time_origin":time.time(), **(manifest or {})}
        self.update_manifest()

    @classmethod
    def begin_prepared(cls, directory):
        """Continue a validated, ready input record; this is not search resumption."""
        writer = cls.__new__(cls)
        writer.directory = Path(directory).resolve()
        writer.path = writer.directory / "trace.sqlite3"
        writer.connection = sqlite3.connect(writer.path, isolation_level=None)
        writer.connection.row_factory = sqlite3.Row
        writer._setup_buffers()
        writer.connection.execute("PRAGMA foreign_keys=ON")
        writer.info = writer.manifest()
        if writer.info.get("schema_version") != SCHEMA_VERSION or writer.info.get("status") != "ready":
            writer.close()
            raise ValueError("仅可从当前格式的已读取、未开始资源启动装配")
        writer.started = time.monotonic() - max(0, time.time() - writer.info["time_origin"])
        writer.snapshot_interval = 24
        writer.update_manifest(status="running", compute_started_at=datetime.now(timezone.utc).isoformat())
        return writer

    def phase_start(self, phase, seq, state_id):
        self._write("INSERT INTO phases(phase,start_seq,start_state_id) VALUES(?,?,?)",
                                (phase,seq,state_id))
        self.flush()

    def phase_end(self, phase, seq, state_id):
        self._write("UPDATE phases SET end_seq=?,end_state_id=? WHERE phase=?",
                                (seq,state_id,phase))
        self.flush()

    def update_manifest(self, **changes):
        self.info.update(changes)
        text = self._store_manifest()
        self.flush()
        self._export_manifest(text)

    def _store_manifest(self):
        text = json_text(self.info)
        self._write("INSERT OR REPLACE INTO metadata VALUES('manifest',?)", (text,))
        return text

    def _export_manifest(self, text):
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")

    def capture(self, state, parent_id=None, snapshot=False):
        return self._capture_owned(project_state(state), parent_id, snapshot)

    def capture_projection(self, projection, parent_id=None, snapshot=False):
        # Public JSON may be reused/mutated after capture; own the cached copy.
        return self._capture_owned(copy.deepcopy(projection), parent_id, snapshot)

    def _capture_owned(self, projection, parent_id=None, snapshot=False):
        key = digest(projection)
        existing = self.connection.execute("SELECT id,kind FROM states WHERE digest=?", (key,)).fetchone()
        if existing:
            if snapshot and existing["kind"]!="snapshot":
                self._write("UPDATE states SET kind='snapshot',depth=0,payload=? WHERE id=?",
                                        (json_text(projection),existing["id"]))
            self._remember(existing[0], projection)
            return existing[0]
        depth = 0
        if parent_id is not None:
            row = self.connection.execute("SELECT depth FROM states WHERE id=?", (parent_id,)).fetchone()
            if row is None:
                raise KeyError(parent_id)
            depth = row[0] + 1
        if snapshot or parent_id is None or depth >= self.snapshot_interval:
            kind, payload, depth = "snapshot", projection, 0
        else:
            kind, payload = "delta", difference(self._projection(parent_id), projection)
        cursor = self._write(
            "INSERT INTO states(digest,parent_id,kind,depth,payload) VALUES(?,?,?,?,?)",
            (key, parent_id, kind, depth, json_text(payload)))
        self._remember(cursor.lastrowid, projection)
        return cursor.lastrowid

    def emit(self, kind, *, scope="system", state_id=None, parent_state_id=None,
             phase=None, round_id=None, attempt=None, data=None):
        cursor = self._write(
            "INSERT INTO events(elapsed,scope,kind,phase,round_id,attempt,state_id,parent_state_id,payload)"
            " VALUES(?,?,?,?,?,?,?,?,?)", (time.monotonic()-self.started, scope, kind, phase,
            round_id, attempt, state_id, parent_state_id, json_text(data or {})))
        if kind in self.publish_events or time.monotonic() - self._last_publish >= self.publish_interval:
            self.flush()
        return cursor.lastrowid

    def save_transition(self, encoded, parent_id, state_id, action):
        key = hashlib.sha256(encoded).hexdigest() + f":{parent_id}"
        self._write("INSERT OR IGNORE INTO transitions VALUES(?,?,?,?)",
                                (key, parent_id, state_id, json_text(action)))
        return key

    def transition(self, encoded, parent_id):
        key = hashlib.sha256(encoded).hexdigest() + f":{parent_id}"
        row = self.connection.execute("SELECT * FROM transitions WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError("search transition was not observed")
        return row["state_id"], json.loads(row["action_json"])

    def finish(self, status="complete", **metadata):
        self.info.update(status=status, elapsed_seconds=time.monotonic()-self.started, **metadata)
        text = self._store_manifest()
        # SSE readers must never observe a completed manifest before its final
        # event. Commit both together, including any remaining frame deltas.
        self.emit("run.finished", data={"status": status, **metadata})
        self._export_manifest(text)
        self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
