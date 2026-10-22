"""One cancellable background query; artifacts never modify the source database."""

import json
import threading
import uuid

from pathway.tracing import trace
from pipeline_runtime import atomic_json


class PathwayTasks:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.current = None
        self.stop = threading.Event()

    def start(self, path, query):
        with self.lock:
            if self.current and self.current["status"] == "running":
                raise ValueError("已有路径任务运行，请等待或取消")
            identifier = uuid.uuid4().hex[:12]
            self.stop = threading.Event()
            stop = self.stop
            state = dict(id=identifier, status="running", query=query, database=str(path))
            self.current = state
            self.root.mkdir(parents=True, exist_ok=True)
            def run():
                try:
                    def progress(value):
                        with self.lock:
                            state["progress"] = value
                    result = trace(path, query, stop.is_set, progress)
                    if stop.is_set():
                        raise InterruptedError("路径任务已取消")
                    atomic_json(self.root / f"{identifier}.json", result)
                    with self.lock:
                        state["status"] = "completed"
                except Exception as exc:
                    with self.lock:
                        state.update(status="cancelled" if stop.is_set() else "failed", error=str(exc))
            threading.Thread(target=run, daemon=True, name="pathway-query").start()
            return dict(state)

    def status(self, identifier):
        with self.lock:
            if not self.current or self.current["id"] != identifier:
                raise ValueError("路径任务不存在或服务已重启，请重新查询")
            return dict(self.current)

    def result(self, identifier):
        if self.status(identifier)["status"] != "completed":
            raise ValueError("路径尚未完成")
        return json.loads((self.root / f"{identifier}.json").read_text())

    def cancel(self, identifier):
        self.status(identifier)
        self.stop.set()
