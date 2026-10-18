"""Behavioral checks: recording is observational and replay preserves identity."""
from dataclasses import replace
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest

from RL_MTCS import AssemblyConfig
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.runner import MCTSRunner
from RL_MTCS.mdp.types import InverseAssemblyInput, HeteroAromaticRequest
from RL_MTCS.mdp.state import HexAssemblyState

from assembly_viewer.adapter import ObservedController, record_state
from assembly_viewer.demo import demo_state
from assembly_viewer.export import export_html, export_frames
from assembly_viewer.recorder import TraceRecorder, TraceReader, difference, apply_difference, project_state
from assembly_viewer.server import start_server


def settings(**kwargs):
    return AssemblyConfig(connection_candidate_limit=2, forward_max_rollbacks=0,
                          seed=17, verbose=False, save_stage_images=False, **kwargs)


def baseline(state, config):
    runner = MCTSRunner.__new__(MCTSRunner)
    runner.initial_state = state.clone()
    runner.config = config
    runner.nmr_scorer = None
    return runner.run_all_stages()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    output = tmp_path_factory.mktemp("observed-demo")
    initial = demo_state()
    result = record_state(initial, output_dir=output, config=settings())
    return initial, result, output


def test_recording_preserves_selected_structure_and_original_input(recorded):
    initial, observed, _ = recorded
    original_signature = initial.state_signature()
    normal = baseline(initial, settings())
    assert observed["state"].state_signature() == normal["state"].state_signature()
    assert observed["state"].problem == normal["state"].problem
    assert observed["state"].action_history == normal["state"].action_history
    assert observed["completion_report"] == normal["completion_report"]
    assert observed["audit"] == normal["audit"]
    assert initial.state_signature() == original_signature
    assert observed["legal"] and observed["all_complete"]


def test_all_recorded_frames_reconstruct_with_checksum_and_gids(recorded):
    _, result, _ = recorded
    with TraceReader(result["process_trace"]) as reader:
        for row in reader.connection.execute("SELECT id FROM states"):
            frame = reader.state(row[0])
            assert all(int(gid) == node["logical_gid"] == node["node_id"]
                       for gid, node in frame["nodes"].items())
            assert all(str(edge["source_node_id"]) in frame["nodes"]
                       and str(edge["target_node_id"]) in frame["nodes"] for edge in frame["edges"].values())
        final = reader.state(reader.manifest()["final_state"])
        assert final == project_state(result["state"])
        assert reader.connection.execute("SELECT count(*) FROM states WHERE kind='delta'").fetchone()[0] > 0


def test_formal_connections_follow_selected_path_only(recorded):
    _, result, _ = recorded
    with TraceReader(result["process_trace"]) as reader:
        events = reader.events(limit=1000)
        formal = [e for e in events if e["kind"] == "formal.connection"]
        forward = [e for e in events if e["kind"] == "forward.connection"]
        assert {e["phase"] for e in formal} == {"rigid", "flex", "ring", "side", "substitution"}
        assert {e["phase"] for e in forward} >= {"flex", "ring", "side", "substitution"}
        assert len(formal) == len(result["state"].action_history)
        for phase in ("rigid", "flex"):
            accepted = next(e for e in events if e["kind"] == "checkpoint.accepted" and e["phase"] == phase)
            matching = [e for e in formal if e["phase"] == phase]
            assert all(e["attempt"] == accepted["data"]["selected_attempt"] for e in matching)
        assert any(e["kind"] == "search.backup" for e in events)
        assert len([e for e in events if e["kind"] == "forward.start"]) == 2


def test_original_modules_are_not_patched(recorded):
    import RL_MTCS.policy.checkpoints as checkpoint_module
    import RL_MTCS.policy.mcts as mcts_module
    import RL_MTCS.policy.rollout as rollout_module
    assert checkpoint_module.MCTS is mcts_module.MCTS
    assert checkpoint_module.fast_rollout_remaining is rollout_module.fast_rollout_remaining
    assert type(StageController(settings())) is StageController


def test_observer_does_not_add_nmr_evaluations(tmp_path):
    class Scorer:
        def __init__(self): self.calls = []
        def score_state(self,state,stage):
            self.calls.append((stage,state.state_signature()))
            return {"loss": .3, "scored": True}
    initial = demo_state()
    scorer1, scorer2 = Scorer(), Scorer()
    baseline_controller = StageController(settings(),scorer1)
    rigid, _ = baseline_controller.run(initial,"rigid")
    expected, _ = baseline_controller.run(rigid,"flex")
    with TraceRecorder(tmp_path/"trace") as recorder:
        traced = ObservedController(recorder,settings(),scorer2)
        other_rigid,_ = traced.run(initial,"rigid")
        actual,_ = traced.run(other_rigid,"flex")
    assert scorer1.calls == scorer2.calls
    assert expected.state_signature() == actual.state_signature()


def test_rollback_and_partial_fallback_keep_unconnected_resources(tmp_path):
    state = demo_state()
    isolated = replace(state.unit("aromatic:3"),unit_id="aromatic:4",ports=(),
        nodes=tuple(replace(n,node_id=n.node_id+1000) for n in state.unit("aromatic:3").nodes),
        internal_edges=tuple(replace(e,source_node_id=e.source_node_id+1000,
            target_node_id=e.target_node_id+1000) for e in state.unit("aromatic:3").internal_edges))
    extra = tuple((n.node_id,13) for n in isolated.nodes)
    problem = replace(state.problem,units=(*state.problem.units,isolated),
        node_su_types=(*state.problem.node_su_types,*extra),
        ownership=(*state.problem.ownership,*((n.node_id,isolated.unit_id) for n in isolated.nodes)))
    partial = HexAssemblyState(problem)
    for i,unit in enumerate(u for u in problem.units if u.phase=="aromatic"):
        assert partial.place_unit(unit.unit_id,0,(30*i,0))
    cfg = replace(settings(),forward_max_rollbacks=1,forward_min_completion=1.)
    normal = baseline(partial,cfg)
    result = record_state(partial,output_dir=tmp_path/"partial",config=cfg)
    assert result["state"].state_signature()==normal["state"].state_signature()
    assert result["legal"] and not result["all_complete"]
    assert result["completion_report"]["excluded_units"]
    assert result["completion_report"]["total"] == 9
    with TraceReader(result["process_trace"]) as reader:
        rows=reader.events(limit=1000)
        assert any(e["kind"]=="checkpoint.rollback" for e in rows)
        assert any(e["kind"]=="checkpoint.accepted" and e["data"]["fallback"] for e in rows)
        assert all(e["attempt"]<=1 for e in rows if e["attempt"] is not None)


def test_zero_flex_cache_reuse_is_visible(tmp_path):
    source = demo_state().unit("aromatic:0")
    aromatic = replace(source,ports=())
    problem=InverseAssemblyInput((aromatic,),(),tuple((n.node_id,13) for n in aromatic.nodes),
                                 tuple((n.node_id,aromatic.unit_id) for n in aromatic.nodes))
    initial=HexAssemblyState(problem)
    assert initial.place_unit(aromatic.unit_id,0,(0,0))
    result=record_state(initial,output_dir=tmp_path,config=settings())
    with TraceReader(result["process_trace"]) as reader:
        events=reader.events(limit=1000)
        assert len([e for e in events if e["kind"]=="forward.start"])==1
        assert any(e["kind"]=="checkpoint.assessment" and e["data"]["cached"] for e in events)
        assessments=[e for e in events if e["kind"]=="checkpoint.assessment"]
        assert assessments[0]["data"]["forward_state_id"]==assessments[1]["data"]["forward_state_id"]
        assert reader.state(assessments[1]["data"]["forward_state_id"])["completion_report"]["all_complete"]


def test_five_member_substitution_records_removed_gid_and_actual_midpoint(tmp_path):
    template=replace(demo_state().unit("aromatic:0"),ports=())
    target,synthetic=template.nodes[0].node_id,template.nodes[1].node_id
    problem=InverseAssemblyInput((template,),(),tuple((n.node_id,30 if n.node_id==target else 13) for n in template.nodes),
        tuple((n.node_id,template.unit_id) for n in template.nodes),
        hetero_requests=(HeteroAromaticRequest(30,5,target,synthetic),))
    initial=HexAssemblyState(problem)
    assert initial.place_unit(template.unit_id,0,(0,0))
    result=record_state(initial,output_dir=tmp_path,config=settings())
    with TraceReader(result["process_trace"]) as reader:
        events=reader.events(limit=1000)
        event=next(e for e in events if e["kind"]=="formal.connection" and e["phase"]=="substitution")
        before,after=reader.state(event["parent_state_id"]),reader.state(event["state_id"])
        assert str(synthetic) in before["nodes"] and str(synthetic) not in after["nodes"]
        assert after["nodes"][str(target)]["su_type"]==30
        assert after["nodes"][str(target)]["final_position"]==list(result["state"].final_node_positions[target])
        assert after["nodes"][str(target)]["physical_slot_id"]


def test_branch_deltas_and_deletions_and_snapshot_depth(tmp_path):
    before={"nodes":{"1":{"su":13,"xy":[1,0]},"2":{"su":13}},"edges":{"1-2":{}}}
    after={"nodes":{"1":{"su":30,"xy":[.5,.5]}},"edges":{}}
    assert apply_difference(before,difference(before,after))==after
    assert before["nodes"]["1"]["su"]==13
    with TraceRecorder(tmp_path,snapshot_interval=2) as writer:
        root=writer.capture_projection(before)
        left=writer.capture_projection(after,root)
        right=writer.capture_projection({**before,"stage":"flex"},root)
        leaf=writer.capture_projection({**after,"stage":"final"},left)
        assert writer.state(left)==after
        assert writer.state(right)["nodes"]["1"]["su"]==13
        assert writer.state(leaf)["stage"]=="final"
        assert writer.connection.execute("SELECT kind FROM states WHERE id=?",(leaf,)).fetchone()[0]=="snapshot"
        assert writer.capture_projection(before)==root
        assert writer.capture_projection(after,snapshot=True)==left
        assert writer.connection.execute("SELECT kind FROM states WHERE id=?",(left,)).fetchone()[0]=="snapshot"
        assert writer.state(leaf)["stage"]=="final"
    with pytest.raises(FileExistsError):
        TraceRecorder(tmp_path)


def test_display_projection_does_not_repeat_full_void_diagnostics(monkeypatch):
    import RL_MTCS.validation as validation
    state = demo_state()
    original_audit = state.audit(diagnostics=False)
    history = state.action_history
    before = state.state_signature()
    def unnecessary_diagnostics(*args, **kwargs):
        raise AssertionError("display frame invoked full spatial diagnostics")
    monkeypatch.setattr(validation, "state_layout_metrics", unnecessary_diagnostics)
    frame = project_state(state)
    assert frame["audit"] == {**original_audit, "diagnostics_scope": "legality_only", "shape_valid": None}
    assert "action_history" not in frame
    assert state.action_history is history and state.state_signature() == before


def test_batched_events_publish_atomically_and_close_keeps_pending_rows(tmp_path):
    with TraceRecorder(tmp_path) as writer:
        # Keep this test independent of clock speed; an explicit flush is the
        # public publication boundary, not a change to search or event count.
        writer.publish_interval = float("inf")
        commits = []
        writer.connection.set_trace_callback(lambda sql: commits.append(sql) if sql == "COMMIT" else None)
        with TraceReader(tmp_path) as reader:
            root = writer.capture_projection({"nodes": {"1": {"su": 13}}})
            for i in range(20):
                child = writer.capture_projection({"nodes": {"1": {"su": 13}}, "step": i}, root)
                writer.emit("search.visit", scope="search", state_id=child, parent_state_id=root)
            assert reader.latest_sequence() == 0 and not commits
            writer.flush()
            assert len(commits) == 1
            events = reader.events(limit=100)
            assert len(events) == 20
            assert all(reader.state(event["state_id"])["step"] == i for i, event in enumerate(events))
            seq = writer.emit("stage.complete", phase="rigid", state_id=child)
            writer.phase_start("rigid", seq, root)
            writer.phase_end("rigid", seq, child)
            assert reader.phases()[0]["end_state_id"] == child
            writer.emit("search.visit", state_id=child)
            assert reader.latest_sequence() == 21
    with TraceReader(tmp_path) as reader:
        assert reader.latest_sequence() == 22


def test_timed_batch_publication_preserves_all_event_sequences(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import assembly_viewer.recorder as module
    with TraceRecorder(tmp_path) as writer:
        now = [writer._last_publish]
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
        with TraceReader(tmp_path) as reader:
            frame = writer.capture_projection({"nodes": {}})
            now[0] += 0.19
            writer.emit("search.visit", state_id=frame)
            assert reader.latest_sequence() == 0
            now[0] += 0.02
            writer.emit("search.visit", state_id=frame)
            assert [row["seq"] for row in reader.events()] == [1, 2]


def test_pending_rows_survive_exception_and_boundaries_publish(tmp_path):
    with pytest.raises(RuntimeError, match="test exception"):
        with TraceRecorder(tmp_path) as writer:
            writer.publish_interval = float("inf")
            frame = writer.capture_projection({"nodes": {}})
            writer.emit("search.visit", state_id=frame)
            with TraceReader(tmp_path) as reader:
                assert reader.latest_sequence() == 0
                writer.emit("search.selected", state_id=frame)
                assert reader.latest_sequence() == 2
                assert reader.state(frame) == {"nodes": {}}
            writer.emit("search.visit", state_id=frame)
            raise RuntimeError("test exception")
    with TraceReader(tmp_path) as reader:
        assert reader.latest_sequence() == 3


def test_finish_publishes_status_and_final_event_in_one_transaction(tmp_path):
    with TraceRecorder(tmp_path) as writer:
        writer.publish_interval = float("inf")
        with TraceReader(tmp_path) as reader:
            before_commit = []
            def inspect_transaction(sql):
                if sql == "COMMIT":
                    before_commit.append((reader.manifest()["status"], reader.latest_sequence()))
            writer.connection.set_trace_callback(inspect_transaction)
            frame = writer.capture_projection({"nodes": {}})
            writer.emit("search.visit", state_id=frame)
            writer.finish("complete", final_state=frame)
            assert before_commit == [("running", 0)]
            assert reader.manifest()["status"] == "complete"
            assert reader.events()[-1]["kind"] == "run.finished"


def test_normal_stage_files_images_and_full_audits_are_preserved(tmp_path):
    config = replace(settings(), save_stage_images=True)
    result = record_state(demo_state(), output_dir=tmp_path, config=config)
    for phase in ("rigid", "flex", "ring", "side", "substitution"):
        for suffix in ("molecule.json", "summary.json", "search_trace.json", "molecule.png"):
            assert (tmp_path / phase / f"{phase}_{suffix}").is_file()
        molecule = json.loads((tmp_path / phase / f"{phase}_molecule.json").read_text())
        assert molecule["audit"]["shape_metrics"]
        assert isinstance(molecule["audit"]["shape_valid"], bool)
        assert "diagnostics_scope" not in molecule["audit"]
    assert (tmp_path / "final.json").is_file()
    assert result["audit"]["shape_metrics"] and result["legal"]


def test_projection_cache_is_bounded_and_isolated_from_callers(tmp_path):
    with TraceRecorder(tmp_path, snapshot_interval=3) as writer:
        source = {"nodes": {"1": {"su": 13}}}
        root = writer.capture_projection(source)
        source["nodes"]["1"]["su"] = -1
        returned = writer.state(root)
        returned["nodes"]["1"]["su"] = -2
        assert writer.state(root)["nodes"]["1"]["su"] == 13
        parent = root
        for i in range(20):
            parent = writer.capture_projection({"nodes": {"1": {"su": i}}}, parent)
            assert len(writer._projections) <= writer.projection_cache_limit
        # Evicted ancestors and a different branch still reconstruct exactly.
        branch = writer.capture_projection({"nodes": {"2": {"su": 30}}}, root)
        assert writer.state(root)["nodes"]["1"]["su"] == 13
        assert writer.state(parent)["nodes"]["1"]["su"] == 19
        assert writer.state(branch) == {"nodes": {"2": {"su": 30}}}
    assert not writer._projections


def test_search_reuses_captured_frames_without_reprojecting_rewards_or_replay(tmp_path, monkeypatch):
    import inspect
    callers = []
    capture = TraceRecorder.capture
    def measured(self, *args, **kwargs):
        callers.append(inspect.currentframe().f_back.f_code.co_name)
        return capture(self, *args, **kwargs)
    monkeypatch.setattr(TraceRecorder, "capture", measured)
    with TraceRecorder(tmp_path) as writer:
        state, info = ObservedController(writer, settings()).search(demo_state(), "rigid")
        transition_count = writer.connection.execute("SELECT count(*) FROM transitions").fetchone()[0]
        assert callers.count("observed_encode") == transition_count
        assert not {"observed_evaluate", "observed_replay", "frame_for"}.intersection(callers)
        assert info["result_candidate_count"] and state.audit(diagnostics=False)["legal"]


def test_stage_index_and_live_visits_exist_before_selection(recorded):
    _, result, _ = recorded
    with TraceReader(result["process_trace"]) as reader:
        phases = reader.phases()
        assert [p["phase"] for p in phases] == ["input", "rigid", "flex", "ring", "side", "substitution"]
        assert all(p["start_state_id"] and p["end_state_id"] for p in phases)
        events = reader.events(limit=1000, mode="live")
        visit = next(e for e in events if e["kind"] == "search.visit")
        selected = next(e for e in events if e["kind"] == "search.selected")
        assert visit["seq"] < selected["seq"]
        assert reader.position(visit["seq"], "live") == visit
        assert reader.connection.execute("SELECT name FROM sqlite_master WHERE name LIKE 'tree_%'").fetchall() == []


def test_offline_export_parses_and_contains_no_external_assets(recorded,tmp_path):
    from html.parser import HTMLParser
    _,result,_=recorded
    path=export_html(result["process_trace"],tmp_path/"replay.html")
    text=Path(path).read_text()
    marker='<script type="application/json" id="embedded-trace">'
    payload=json.loads(text.split(marker,1)[1].split('</script>',1)[0])
    assert payload["manifest"]["legal"]
    assert payload["events"] and payload["states"] and payload["phases"] and payload["resources"]
    assert '<script src=' not in text and '<link rel="stylesheet"' not in text
    assert all(str(e["state_id"]) in payload["states"] for e in payload["events"] if e["state_id"])
    HTMLParser().feed(text)


def test_http_state_pagination_and_sse_reconnection(recorded):
    _,result,_=recorded
    server=start_server(result["process_trace"],port=0)
    address=f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(address+'/') as response: assert b'GRAPHNMR / ASSEMBLY' in response.read()
        manifest=json.load(urlopen(address+'/api/trace/manifest'))
        frame=json.load(urlopen(address+f"/api/trace/state/{manifest['final_state']}"))
        assert frame["completion_report"]["all_complete"]
        page=json.load(urlopen(address+'/api/trace/events?limit=2'))["events"]
        next_page=json.load(urlopen(address+f"/api/trace/events?after={page[-1]['seq']}&limit=2"))["events"]
        assert next_page[0]["seq"]>page[-1]["seq"]
        with urlopen(address+f"/api/trace/stream?after={manifest['latest_sequence']-1}",timeout=3) as response:
            lines=response.read().decode()
            assert f"id: {manifest['latest_sequence']}" in lines and 'event: finished' in lines
    finally:
        server.shutdown();server.server_close()


def test_publication_stage_frames_are_exportable(recorded,tmp_path):
    _,result,_=recorded
    paths=export_frames(result["process_trace"],tmp_path,formats=("svg",))
    assert len(paths)==5
    assert all(Path(path).is_file() and '<svg' in Path(path).read_text() for path in paths)


def minimal_csv_inputs(directory):
    import csv
    directory.mkdir(parents=True,exist_ok=True)
    unit=replace(demo_state().unit("aromatic:0"),ports=())
    neighbors={n.node_id:[] for n in unit.nodes}
    for edge in unit.internal_edges:
        neighbors[edge.source_node_id].append(edge.target_node_id)
        neighbors[edge.target_node_id].append(edge.source_node_id)
    neighbors[0].append(10)
    nodes_path=directory/"final_nodes_by_global_id.csv"
    with nodes_path.open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=["global_id","su_type","su_name","hop1_ids","template_key"])
        writer.writeheader()
        for node in unit.nodes:
            writer.writerow(dict(global_id=node.node_id,su_type=11 if node.node_id==0 else 13,su_name="SU",
                                 hop1_ids=json.dumps(neighbors[node.node_id]),template_key="(13, (13, 13), ())"))
        writer.writerow(dict(global_id=10,su_type=22,su_name="SU22",hop1_ids="[0]",template_key="(22, (11,), ())"))
    topology_path=directory/"final_topology_specs.csv"
    topology_path.write_text("actual_su,gid,product_type,required_slots_satisfied\n11-22,0-10,side_chain,True\n")
    plan=dict(schema_version=1,complete=True,cluster_count=1,ring_count=1,aromatic_node_count=6,
        source_node_count=7,assembly_node_count=7,clusters=[dict(cluster_id=0,kind="benzene",
        nodes=[dict(slot_index=i,node_id=n.node_id,planned_su_type=11 if n.node_id==0 else 13,staging_su_type=13,
                    template_bridgehead=False,local_axial=n.local_axial) for i,n in enumerate(unit.nodes)],
        edges_by_slot=[[e.source_node_id,e.target_node_id] for e in unit.internal_edges],local_rings=unit.local_rings)])
    (directory/"final_aromatic_clusters.json").write_text(json.dumps(plan))
    return nodes_path,topology_path


def test_public_csv_entrypoint_and_provenance(tmp_path):
    from assembly_viewer import assemble_with_trace
    from RL_MTCS import assemble_from_csv
    nodes,topology=minimal_csv_inputs(tmp_path/"inputs")
    expected=assemble_from_csv(nodes,topology,config=settings())
    result=assemble_with_trace(nodes,topology,output_dir=tmp_path/"run",config=settings())
    assert result["state"].state_signature()==expected["state"].state_signature()
    with TraceReader(result["process_trace"]) as reader:
        manifest=reader.manifest()
        assert set(manifest["inputs"])=={"nodes","topology","aromatic_clusters"}
        assert len(manifest["engine_sha256"])>10
        assert manifest["legal"]



def test_failure_has_persisted_report_and_does_not_patch_modules(tmp_path,monkeypatch):
    import assembly_viewer.adapter as adapter
    import RL_MTCS.policy.checkpoints as module
    original=module.MCTS
    def fail(*args,**kwargs):
        raise RuntimeError("intentional observation failure")
    monkeypatch.setattr(adapter,"ObservedMCTS",fail)
    with pytest.raises(RuntimeError,match="intentional"):
        record_state(demo_state(),output_dir=tmp_path,config=settings())
    with TraceReader(tmp_path/"process_trace") as reader:
        assert reader.manifest()["status"]=="error"
        assert any(e["kind"]=="run.error" for e in reader.events())
    assert module.MCTS is original


def test_observer_trees_release_without_a_gc_cycle(tmp_path,monkeypatch):
    import weakref
    import assembly_viewer.adapter as adapter
    original=adapter.ObservedMCTS
    references=[]
    class Observed(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            references.append(weakref.ref(self))
    monkeypatch.setattr(adapter,"ObservedMCTS",Observed)
    record_state(demo_state(),output_dir=tmp_path,config=settings())
    assert references and all(ref() is None for ref in references)


def test_sse_delivers_events_while_writer_is_running(tmp_path):
    received=[]
    begun=threading.Event()
    with TraceRecorder(tmp_path) as writer:
        writer.emit("run.started")
        server=start_server(tmp_path,port=0)
        address=f"http://127.0.0.1:{server.server_address[1]}"
        def listen():
            with urlopen(address+'/api/trace/stream?after=1',timeout=5) as response:
                begun.set()
                received.append(response.read().decode())
        listener=threading.Thread(target=listen)
        listener.start()
        try:
            assert begun.wait(3)
            writer.emit("formal.connection",scope="formal",data={"action":{"kind":"test"}})
            writer.finish("complete")
            listener.join(5)
            assert not listener.is_alive()
            assert 'id: 3' in received[0] and 'id: 1\n' not in received[0]
        finally:
            server.shutdown();server.server_close()


def test_zero_connection_video_has_initial_and_stage_frames(tmp_path, monkeypatch):
    from assembly_viewer.export import export_video
    import shutil
    import subprocess
    source = replace(demo_state().unit("aromatic:0"), ports=())
    problem = InverseAssemblyInput((source,), (), tuple((n.node_id, 13) for n in source.nodes),
                                  tuple((n.node_id, source.unit_id) for n in source.nodes))
    state = HexAssemblyState(problem)
    assert state.place_unit(source.unit_id, 0, (0, 0))
    result = record_state(state, output_dir=tmp_path / "run", config=settings())
    with TraceReader(result["process_trace"]) as reader:
        assert not any(e["kind"] == "formal.connection" for e in reader.events(limit=1000))
    calls = []
    def encoder(args, **kwargs):
        frame_pattern = Path(args[args.index("-i") + 1])
        calls.append(len(list(frame_pattern.parent.glob("*.png"))))
        Path(args[-1]).write_bytes(b"test video")
    monkeypatch.setattr(shutil, "which", lambda _: "/test/ffmpeg")
    monkeypatch.setattr(subprocess, "run", encoder)
    video = export_video(result["process_trace"], tmp_path / "zero.mp4")
    assert calls[0] >= 11  # initial state plus starts/ends for all five stages
    assert Path(video).is_file()


def test_pagination_and_stage_seek_independent_of_loaded_page(tmp_path):
    state = project_state(demo_state())
    with TraceRecorder(tmp_path) as writer:
        frame = writer.capture_projection(state)
        for i in range(320):
            writer.emit("search.visit", scope="search", state_id=frame, phase="rigid", data=dict(simulation=i))
        start = writer.emit("stage.start", scope="formal", state_id=frame, phase="side")
        writer.phase_start("side", start, frame)
        end = writer.emit("stage.complete", scope="formal", state_id=frame, phase="side")
        writer.phase_end("side", end, frame)
    with TraceReader(tmp_path) as reader:
        first = reader.events(limit=250, mode="live")
        second = reader.events(after=first[-1]["seq"], limit=250, mode="live")
        assert len(first) == 250 and len(second) == 72
        assert [e["seq"] for e in first + second] == list(range(1, 323))
        assert reader.phases()[0]["start_seq"] == 321
        assert reader.position(321, "formal")["state_id"] == frame
        with pytest.raises(ValueError, match="未知"):
            reader.events(mode="obsolete")
