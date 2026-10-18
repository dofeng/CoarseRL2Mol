"""Explicit preparation, isolated workers, restartable history and one interface."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from assembly_viewer import start_ui, open_workbench
from assembly_viewer.recorder import TraceReader
from assembly_viewer.tests.test_trace import minimal_csv_inputs, settings
from assembly_viewer.workflow import RunManager, RunConflict
from assembly_viewer.worker import prepare


def request(base, path, data=None, origin=None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if origin:
        headers["Origin"] = origin
    req = Request(base + path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    with urlopen(req, timeout=20) as response:
        content = response.read()
        return response.status, json.loads(content) if "application/json" in response.headers.get("Content-Type", "") else content.decode()


def wait_worker(manager, identifier):
    manager.processes[identifier].wait(timeout=20)
    info = manager.info(identifier)
    assert info["status"] != "error", manager.log_tail(identifier)
    return info


def test_prepare_never_calls_search_or_nmr(tmp_path, monkeypatch):
    nodes, topology = minimal_csv_inputs(tmp_path / "input")
    output = tmp_path / "output"
    output.mkdir()
    spec = dict(id="test", sample="test", nodes_csv=str(nodes), topology_csv=str(topology),
                output_dir=str(output), config=asdict(settings()))
    path = output / "run.json"
    path.write_text(json.dumps(spec))
    def forbidden(*args, **kwargs):
        raise AssertionError("input preview invoked search/NMR")
    from RL_MTCS.runner import MCTSRunner
    monkeypatch.setattr(MCTSRunner, "__init__", forbidden)
    from RL_MTCS.policy.mcts import MCTS
    monkeypatch.setattr(MCTS, "__init__", forbidden)
    prepare(path)
    with TraceReader(output / "process_trace") as reader:
        assert reader.manifest()["status"] == "ready"
        assert reader.resources()["counts"] == {"rigid": 1, "side": 1}
        assert not any(e["kind"].startswith(("search.", "forward.")) for e in reader.events())
        assert not (output / "final.json").exists()


def test_real_workflow_start_and_restart_history_and_exports(tmp_path):
    for sample in ("YM-3", "LY-2"):
        minimal_csv_inputs(tmp_path / "test_results" / sample / "final_outputs")
    server = start_ui(tmp_path, port=0)
    base = server.url
    try:
        _, page = request(base, "/")
        assert page.count('<svg ') == 1 and '<iframe' not in page
        _, meta = request(base, "/api/context")
        assert set(meta["samples"]) == {"YM-3", "LY-2"}
        status, info = request(base, "/api/prepare", dict(sample="YM-3", config=asdict(settings())))
        assert status == 201
        identifier = info["id"]
        ready = wait_worker(server.manager, identifier)
        assert ready["status"] == "ready" and " prepare " in ready["command"]
        _, resources = request(base, f"/api/trace/resources?run={identifier}")
        assert resources["node_count"] == 7
        with TraceReader(server.manager.trace(server.manager.get(identifier))) as reader:
            assert all(e["scope"] not in {"search", "forward"} for e in reader.events())
        _, started = request(base, f"/api/runs/{identifier}/start", {"config": asdict(settings())})
        assert " run " in started["command"]
        final = wait_worker(server.manager, identifier)
        assert final["legal"] and final["all_complete"]
        with pytest.raises(HTTPError) as err:
            request(base, f"/api/runs/{identifier}/start", {})
        assert err.value.code == 409
        _, phases = request(base, f"/api/trace/phases?run={identifier}")
        assert len(phases) == 6
        _, events = request(base, f"/api/trace/events?run={identifier}&mode=live&limit=2")
        _, next_events = request(base, f"/api/trace/events?run={identifier}&mode=live&after={events['events'][-1]['seq']}&limit=2")
        assert next_events["events"][0]["seq"] > events["events"][-1]["seq"]
        _, exported = request(base, f"/api/runs/{identifier}/export", {})
        assert exported["figure_count"] == 15
        _, svg = request(base, exported["svg_url"])
        assert '<svg ' in svg and '<text ' in svg and '<image ' not in svg
        with urlopen(base + exported["pdf_url"]) as response:
            assert response.headers.get_content_type() == "application/pdf"
            assert response.read().startswith(b"%PDF-")
        assert len(exported["final_figures"]) == 3
        _, offline = request(base, exported["html_url"])
        assert 'id="embedded-trace"' in offline
        _, logs = request(base, f"/api/runs/{identifier}/log")
        assert isinstance(logs["log"], str)
        with pytest.raises(HTTPError) as err:
            request(base, "/api/tree")
        assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
    restarted = RunManager(tmp_path)
    assert restarted.list()[0]["id"] == identifier
    assert restarted.info(identifier)["all_complete"]
    assert restarted.reopen(final["output_dir"])["id"] == identifier
    assert not restarted.processes


def test_changed_inputs_rejected_at_start(tmp_path):
    nodes, topology = minimal_csv_inputs(tmp_path / "test_results" / "YM-3" / "final_outputs")
    manager = RunManager(tmp_path)
    ready = manager.prepare(dict(sample="YM-3", config=asdict(settings())))
    wait_worker(manager, ready["id"])
    nodes.write_text(nodes.read_text() + "\n")
    with pytest.raises(RunConflict, match="已变化"):
        manager.start(ready["id"], {})
    assert manager.get(ready["id"])["worker_kind"] == "prepare"


def test_bad_inputs_and_settings_never_start_worker(tmp_path):
    minimal_csv_inputs(tmp_path / "test_results" / "YM-3" / "final_outputs")
    server = start_ui(tmp_path, port=0)
    try:
        for case in [dict(sample="../other"), dict(sample="missing"), dict(sample="YM-3", config={"connection_candidate_limit": 0}), dict(sample="YM-3", python="/missing/python")]:
            with pytest.raises(HTTPError) as err:
                request(server.url, "/api/prepare", case)
            assert err.value.code == 400
        with pytest.raises(HTTPError) as err:
            request(server.url, "/api/prepare", dict(sample="YM-3"), origin="http://other.invalid")
        assert err.value.code == 403
        assert not server.manager.list()
    finally:
        server.shutdown()
        server.server_close()


def test_notebook_entry_opens_ui_without_computation(tmp_path, capsys):
    server = open_workbench(project=tmp_path, config=settings(), port=0)
    try:
        assert not server.manager.list()
        assert server.manager.metadata()["defaults"]["seed"] == 17
        assert server.url in capsys.readouterr().out
    finally:
        server.shutdown()
        server.server_close()


def test_restart_detects_missing_start_worker_without_resuming(tmp_path):
    minimal_csv_inputs(tmp_path / "test_results" / "YM-3" / "final_outputs")
    manager = RunManager(tmp_path)
    info = manager.prepare(dict(sample="YM-3", config=asdict(settings())))
    wait_worker(manager, info["id"])
    spec = manager.get(info["id"])
    spec.update(worker_kind="run", pid=-1)
    manager.save(spec)
    restarted = RunManager(tmp_path)
    assert restarted.info(info["id"])["status"] == "interrupted"
    assert not restarted.processes
