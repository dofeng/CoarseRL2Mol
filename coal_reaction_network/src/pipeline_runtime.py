"""Bounded execution, process accounting and crash-safe progress reporting."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path


def process_tree_rss_mib(pid: int | None = None) -> float:
    pending = [pid or os.getpid()]
    visited = set()
    total = 0
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                pending.extend(map(int, children.read_text().split()))
        except (OSError, ValueError):
            continue
    return total / 1024


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def database_lock(database: Path):
    database.parent.mkdir(parents=True, exist_ok=True)
    path = database.with_suffix(database.suffix + ".lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another analysis is writing {database}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def bounded_results(worker, tasks, workers: int, max_inflight: int, memory_limit_mib: int = 3072):
    if workers < 1 or max_inflight < workers:
        raise ValueError("workers >= 1 and max_inflight >= workers are required")
    iterator = iter(tasks)
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    pending = {}
    exhausted = False
    try:
        while pending or not exhausted:
            usage = process_tree_rss_mib()
            if memory_limit_mib and usage > memory_limit_mib:
                raise MemoryError(
                    f"Process-tree RSS {usage:.0f} MiB exceeds budget {memory_limit_mib} MiB; "
                    "resume with fewer workers or smaller max-inflight"
                )
            while not exhausted and len(pending) < max_inflight:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(worker, task)] = task
            if not pending:
                break
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            while ready:
                future = ready.pop()
                task = pending.pop(future)
                yield task, future
                del future, task
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class Progress:
    def __init__(self, path: Path, memory_limit_mib: int = 3072):
        self.path = path
        self.memory_limit_mib = memory_limit_mib
        self.started = time.monotonic()
        self.last_write = 0.0
        self.peak_rss = 0.0
        self.state = {"status": "running", "stage": "discovery", "pid": os.getpid()}

    def update(self, force: bool = False, **values) -> None:
        self.state.update(values)
        current = time.monotonic()
        if not force and current - self.last_write < 0.5:
            return
        usage = process_tree_rss_mib()
        self.peak_rss = max(self.peak_rss, usage)
        self.state.update(rss_mib=round(usage, 1), peak_rss_mib=round(self.peak_rss, 1),
                          elapsed_seconds=round(current - self.started, 1),
                          updated_at=time.time())
        atomic_json(self.path, self.state)
        self.last_write = current
        if self.memory_limit_mib and usage > self.memory_limit_mib:
            raise MemoryError(f"Process-tree RSS {usage:.0f} MiB exceeds configured memory budget")
