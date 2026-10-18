"""External observation adapters: reuse original function bodies and decisions.

Function globals are rebound in private function objects, never monkey-patched
on RL_MTCS modules. No candidate, reward, RNG, or stopping rule is changed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import FunctionType, MethodType
import weakref

from RL_MTCS.config import AssemblyConfig
from RL_MTCS.policy.mcts import MCTS
from RL_MTCS.policy.checkpoints import StageController
from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.runner import MCTSRunner

from .recorder import TraceRecorder


def call_with_globals(function, replacements, *args, **kwargs):
    """Invoke unchanged bytecode with instance-local observer dependencies."""
    adapted = FunctionType(function.__code__, {**function.__globals__, **replacements},
                           function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = function.__kwdefaults__
    return adapted(*args, **kwargs)


def _write_stage_output(runner, payload, output_dir, stage):
    """Reuse the engine's data output, replacing only this instance's drawing."""
    from .export import save_state_figures
    call_with_globals(MCTSRunner._write_stage_output, {"save_state_png": save_state_figures},
                      runner, payload, output_dir, stage)
    if not output_dir or not runner.config.save_stage_images:
        return
    outputs = payload["stage_outputs"]
    for extension in ("svg", "pdf"):
        path = Path(output_dir) / f"{stage}_molecule.{extension}"
        if path.is_file():
            outputs[extension] = str(path)
    payload["result"]["stage_outputs"] = dict(outputs)
    runner._write_json(payload["result"], str(Path(output_dir) / f"{stage}_summary.json"))


def _export_final_state(state, output_dir):
    from RL_MTCS.io.output import export_final_state
    from .export import save_state_figures
    outputs = call_with_globals(export_final_state, {"save_state_png": save_state_figures},
                                state, output_dir)
    for extension in ("svg", "pdf"):
        path = Path(output_dir) / f"final_molecule.{extension}"
        if path.is_file():
            outputs[extension] = str(path)
    return outputs


class ObservedMCTS(MCTS):
    def __init__(self, initial, *, observer, context, **kwargs):
        self.recorder = observer.recorder
        self.controller = observer
        self.context = context
        self.nodes = {}  # Live tree IDs only; never hold extra node/state objects.
        self.node_counter = 0
        self.best_path = []
        self.releasing = False
        self.root_frame = self.recorder.capture(initial, observer.origin_id)
        self._frame_ref = weakref.ref(initial)
        self._frame_id = self.root_frame
        self._prepared_ref = None
        self._prepared_frame = None
        encode = kwargs["encode"]
        replay = kwargs["replay"]
        evaluate = kwargs["evaluate"]
        action_function = kwargs["action"]

        def observed_encode(parent, child):
            encoded = encode(parent, child)
            parent_id = self.frame_for(parent)
            child_id = self.recorder.capture(child, parent_id)
            self.bind_frame(child, child_id)
            action = action_function(parent, child)
            self.recorder.save_transition(encoded, parent_id, child_id, action)
            self.event("search.candidate", child_id, parent_id,
                       action=action, root_candidate=parent_id == self.root_frame)
            return encoded

        def observed_replay(parent, encoded):
            child = replay(parent, encoded)
            if not self.releasing:
                parent_id = self.frame_for(parent)
                child_id, action = self.recorder.transition(encoded, parent_id)
                self.bind_frame(child, child_id)
                self.event("search.visit", child_id, parent_id, action=action,
                           simulation=self.simulations)
            return child

        def observed_evaluate(state):
            frame = self.frame_for(state)
            previous = observer.reward_frame
            observer.reward_frame = frame
            try:
                return evaluate(state)
            finally:
                observer.reward_frame = previous

        kwargs.update(encode=observed_encode, replay=observed_replay, evaluate=observed_evaluate)
        super().__init__(initial, **kwargs)
        self.node_info(self.root)

    def bind_frame(self, state, frame):
        self._frame_ref, self._frame_id = weakref.ref(state), frame

    def frame_for(self, state):
        """Reuse frames only during the engine's read-only search lifecycle.

        Two weak references suffice for the enumerated parent and the current
        child. No persistent object-ID cache can reuse a mutated stage state.
        """
        if self._prepared_ref is not None and self._prepared_ref() is state:
            return self._prepared_frame
        if self._frame_ref is not None and self._frame_ref() is state:
            return self._frame_id
        if state is self.initial:
            return self.root_frame
        frame = self.recorder.capture(state)
        self.bind_frame(state, frame)
        return frame

    def event(self, kind, state_id=None, parent_state_id=None, **data):
        return self.recorder.emit(kind, scope="search", state_id=state_id,
                                  parent_state_id=parent_state_id, data=data, **self.context)

    def node_info(self, node):
        cached = self.nodes.get(id(node))
        if cached:
            return cached
        if node.parent is None:
            parent, state_id, action = None, self.root_frame, {}
        else:
            parent, parent_frame = self.node_info(node.parent)
            state_id, action = self.recorder.transition(node.transition, parent_frame)
        node_id = f"{self.context['round_id']}/n{self.node_counter}"
        ordinal = self.node_counter
        self.node_counter += 1
        self.nodes[id(node)] = (node_id, state_id)
        self.event("search.node", state_id, node_id=node_id, parent_node_id=parent,
                         action=action, ordinal=ordinal)
        return node_id, state_id

    def _prepare(self, node, state):
        node_id, frame = self.node_info(node)
        self._prepared_ref, self._prepared_frame = weakref.ref(state), frame
        self.event("search.expand", frame, node_id=node_id)
        from RL_MTCS.reporting.diagnostics import CandidateDiagnostics, collect_candidates
        scan=CandidateDiagnostics(lambda row:self.event("search.scan",frame,node_id=node_id,**row))
        try:
            with collect_candidates(scan):
                return super()._prepare(node, state)
        finally:
            self.event("search.scan_summary",frame,node_id=node_id,units=scan.report())
            # The engine replays one of these candidates immediately after
            # enumeration. Parent/child frame IDs are already on disk.
            self._prepared_ref = self._prepared_frame = None
            self.bind_frame(state, frame)
            self.recorder.flush()

    def _backup(self, node, value):
        path = []
        current = node
        while current is not None:
            self.node_info(current)
            path.append(current)
            current = current.parent
        super()._backup(node, value)
        summaries = []
        for current in path:
            node_id, frame = self.node_info(current)
            summaries.append(dict(node_id=node_id, state_id=frame, visits=current.visits,
                                  value=current.value, prior=current.prior, closed=current.closed))
        self.event("search.backup", self.node_info(node)[1], value=value, path=summaries)

    def _terminal(self, node, state):
        node_id, frame = self.node_info(node)
        self.bind_frame(state, frame)
        value = super()._terminal(node, state)
        self.event("search.result", frame, node_id=node_id, value=value,
                   eligible=value is not None, selected_so_far=self.best_state is state)
        if self.best_state is state:
            path = []
            current = node
            while current.parent is not None:
                _, parent_frame = self.node_info(current.parent)
                _, child_frame = self.node_info(current)
                _, action = self.recorder.transition(current.transition, parent_frame)
                path.append(dict(source=parent_frame, target=child_frame, action=action))
                current = current.parent
            self.best_path = list(reversed(path))
        return value

    def _retire(self, node, state):
        node_id, frame = self.node_info(node)
        children = [dict(node_id=self.node_info(child)[0], visits=child.visits,
                         value=child.value, prior=child.prior, closed=child.closed)
                    for child in node.children]
        self.event("search.close", frame, node_id=node_id, children=children)
        child_keys = [id(child) for child in node.children]
        result = super()._retire(node, state)
        for key in child_keys:
            self.nodes.pop(key, None)
        return result

    def _release_round(self):
        self.releasing = True
        try:
            return super()._release_round()
        finally:
            self.nodes.clear()
            self.releasing = False

    def dispose(self):
        """Break observer closures immediately after the round, without waiting for GC."""
        self.encode = self.replay = self.evaluate = self.expand = None
        self.accept = self.signature = self.action = self.observe = None
        self.guidance = self.progress = self.tie_break = None
        self.initial = self.best_state = self.controller = None
        self.nodes.clear()
        self.best_path = []
        self._frame_ref = self._prepared_ref = None
        self._frame_id = self._prepared_frame = None


class ObservedController(StageController):
    def __init__(self, recorder, config, nmr_scorer=None, guidance=None):
        super().__init__(config, nmr_scorer, guidance)
        self.recorder = recorder
        self.round_number = 0
        self.context = {}
        self.origin_id = None
        self.formal_id = None
        self.reward_frame = None
        self.in_search = False
        self.last_forward = None
        self.last_forward_frame = None
        self.forward_frames = {}
        self.cached_forward_frame = None
        self.paths = {}
        self.last_path = []

    def reward(self, state, forward=None, *, demand=None):
        result = super().reward(state, forward, demand=demand)
        if self.in_search:
            self.recorder.emit("search.reward", scope="search", state_id=self.reward_frame,
                               data=result, **self.context)
        elif self.context and self.context["phase"] in {"rigid", "flex"}:
            report = self.last_forward
            cached = False
            if report is None:
                report = self.forward_cache.get(state.forward_signature())
                cached = report is not None
            if report is not None:
                frame = self.recorder.capture(state)
                valid = bool(report["audit"]["legal"])
                accepted = valid and report["forward_completion"] >= self.config.forward_min_completion
                self.recorder.emit("checkpoint.assessment", scope="formal", state_id=frame,
                    data=dict(rewards=result, forward_completion=report["forward_completion"],
                              forward_valid=valid, accepted=accepted, cached=cached,
                              threshold=self.config.forward_min_completion,
                              forward_state_id=self.cached_forward_frame if cached else self.last_forward_frame,
                              completion_report={k: report[k] for k in ("completed", "total", "stages")}),
                    **self.context)
        return result

    def search(self, initial, phase, excluded=(), *, demand=None):
        self.round_number += 1
        self.context = dict(phase=phase, round_id=f"r{self.round_number}", attempt=len(excluded))
        self.origin_id = self.recorder.capture(initial, self.formal_id)
        self.last_forward = None
        self.last_forward_frame = None
        if excluded:
            self.recorder.emit("checkpoint.rollback", scope="formal", state_id=self.origin_id,
                data=dict(excluded_result_count=len(excluded),
                          reason="restore checkpoint and search a different whole result"), **self.context)
        self.recorder.emit("search.start", scope="search", state_id=self.origin_id,
                           data=dict(candidate_limit=self.config.connection_candidate_limit), **self.context)
        trees = []

        def factory(*args, **kwargs):
            tree = ObservedMCTS(*args, observer=self, context=dict(self.context), **kwargs)
            trees.append(tree)
            return tree

        self.in_search = True
        try:
            selected, info = call_with_globals(StageController.search, {"MCTS": factory},
                                               self, initial, phase, excluded, demand=demand)
        except BaseException:
            for tree in trees:
                tree.dispose()
            trees.clear()
            raise
        finally:
            self.in_search = False
        path = trees[0].best_path if trees else []
        self.last_path = path
        self.paths[self.context["attempt"]] = (list(path), dict(self.context))
        frame = self.recorder.capture(selected, self.origin_id) if selected is not None else None
        self.recorder.emit("search.selected", scope="search", state_id=frame,
                           data=dict(diagnostics=info, path=path), **self.context)
        for tree in trees:
            tree.dispose()
        trees.clear()
        return selected, info

    def forward(self, state, *, progress=None):
        context = dict(self.context)
        current_id = self.recorder.capture(state)
        current_ref = weakref.ref(state)
        self.recorder.emit("forward.start", scope="forward", state_id=current_id, **context)
        original = fast_rollout_remaining.__globals__["connection_states"]
        finish = fast_rollout_remaining.__globals__["finish_flex"]

        def connections(parent, phase, *args, **kwargs):
            nonlocal current_id, current_ref
            parent_id = (current_id if current_ref() is parent
                         else self.recorder.capture(parent, current_id))
            stream = iter(original(parent, phase, *args, **kwargs))
            try:
                for child in stream:
                    current_id = self.recorder.capture(child, parent_id)
                    current_ref = weakref.ref(child)
                    self.recorder.emit("forward.connection", scope="forward", phase=phase,
                        round_id=context["round_id"], attempt=context["attempt"],
                        state_id=current_id, parent_state_id=parent_id,
                        data=dict(action=child.action_history[-1] if child.action_history else {}))
                    yield child
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()

        def finish_flex(parent):
            nonlocal current_id, current_ref
            child = finish(parent)
            current_id = self.recorder.capture(child, current_id)
            current_ref = weakref.ref(child)
            self.recorder.emit("forward.flex_finished", scope="forward", state_id=current_id,
                               phase="flex", round_id=context["round_id"], attempt=context["attempt"])
            return child

        def on_progress(row):
            self.recorder.emit("forward.phase." + row["event"], scope="forward", state_id=current_id,
                phase=row["phase"], round_id=context["round_id"], attempt=context["attempt"], data=row)
            if progress:
                progress(row)

        result = call_with_globals(fast_rollout_remaining,
            {"connection_states": connections, "finish_flex": finish_flex}, state, progress=on_progress)
        self.last_forward = result.to_dict()
        if current_ref() is not result.state:
            current_id = self.recorder.capture(result.state, current_id)
        self.last_forward_frame = current_id
        self.forward_frames[context["attempt"]] = current_id
        self.recorder.emit("forward.complete", scope="forward", state_id=current_id,
                           data=self.last_forward, **context)
        return result

    def promote(self, path, context):
        for step in path:
            self.recorder.emit("formal.connection", scope="formal", state_id=step["target"],
                parent_state_id=step["source"], data=dict(action=step["action"]), **context)
            self.formal_id = step["target"]

    def checkpoint(self, initial, phase):
        self.paths = {}
        self.forward_frames = {}
        selected, info = call_with_globals(StageController.checkpoint,
            {"fast_rollout_remaining": self.forward}, self, initial, phase)
        trace = info["retry_selection"]
        self.cached_forward_frame = self.forward_frames.get(trace["selected_attempt"], self.cached_forward_frame)
        path, context = self.paths.get(trace["selected_attempt"], ([], dict(self.context)))
        self.promote(path, context)
        self.formal_id = self.recorder.capture(selected, self.formal_id, snapshot=True)
        self.recorder.emit("checkpoint.accepted", scope="formal", state_id=self.formal_id,
            data=dict(selected_attempt=trace["selected_attempt"], fallback=trace["fallback"],
                      start_count=trace["start_count"], end_count=trace["end_count"],
                      rewards=info["rewards"], forward_completion=info["fast_rollout"]["forward_completion"],
                      forward_state_id=self.cached_forward_frame),
            **context)
        self.paths = {}
        return selected, info

    def run(self, state, phase):
        self.formal_id = self.recorder.capture(state, self.formal_id, snapshot=True)
        seq = self.recorder.emit("stage.start", scope="formal", state_id=self.formal_id, phase=phase)
        self.recorder.phase_start(phase, seq, self.formal_id)
        self.recorder.update_manifest(current_phase=phase)
        selected, info = super().run(state, phase)
        if phase not in {"rigid", "flex"}:
            self.promote(self.last_path, self.context)
        self.formal_id = self.recorder.capture(selected, self.formal_id, snapshot=True)
        seq = self.recorder.emit("stage.complete", scope="formal", state_id=self.formal_id, phase=phase, data=info)
        self.recorder.phase_end(phase, seq, self.formal_id)
        self.recorder.update_manifest(current_phase=phase, latest_formal_state=self.formal_id)
        return selected, info


def assemble_with_trace(nodes_csv, topology_specs_csv, *, output_dir, config=None,
                        trace_dir=None, search_guidance=None):
    """Same assembly result as MCTSRunner, with an independent process trace.

    Use a fresh output directory. The CLI can launch this in a separate worker;
    the synchronous API is also suitable for a notebook. No global hooks remain.
    """
    config = config or AssemblyConfig()
    output_dir = Path(output_dir).resolve()
    trace_dir = Path(trace_dir).resolve() if trace_dir else output_dir / "process_trace"
    from .resources import provenance
    with TraceRecorder(trace_dir, provenance(nodes_csv, topology_specs_csv, config)) as recorder:
        recorder.emit("run.started")
        try:
            runner = MCTSRunner(nodes_csv, config=config, topology_specs_csv=topology_specs_csv,
                                search_guidance=search_guidance)
            return _run_observed_runner(runner, recorder, output_dir, search_guidance)
        except BaseException as exc:
            recorder.emit("run.error", data=dict(type=type(exc).__name__, message=str(exc)))
            recorder.finish("error", error=dict(type=type(exc).__name__, message=str(exc)))
            raise


def _run_observed_runner(runner, recorder, output_dir, guidance=None):
    from .resources import register_input
    controller = ObservedController(recorder, runner.config, runner.nmr_scorer, guidance)
    # Runner already supports an instance-owned controller; its code is unchanged.
    runner._stage_controller = controller
    initial_id = recorder.info.get("initial_state")
    if initial_id is None:
        initial_id = register_input(recorder, runner.initial_state)
    else:
        from .recorder import digest, project_state
        if digest(project_state(runner.initial_state)) != digest(recorder.state(initial_id)):
            raise ValueError("读取预览与运行初始化状态不一致，请重新读取输入")
    controller.formal_id = initial_id
    recorder.emit("formal.initialized", scope="formal", state_id=initial_id)
    runner._write_stage_output = MethodType(_write_stage_output, runner)
    result = call_with_globals(MCTSRunner.run_all_stages, {"export_final_state": _export_final_state},
                               runner, str(output_dir))
    recorder.finish(result["status"], legal=result["legal"], all_complete=result["all_complete"],
                    final_state=controller.formal_id, completion_report=result["completion_report"])
    result["process_trace"] = str(recorder.directory)
    return result


def record_state(initial_state, *, output_dir, config=None, nmr_scorer=None, search_guidance=None,
                 synthetic=False):
    """Record an already initialized state, including explicitly marked demos."""
    config = config or AssemblyConfig()
    with TraceRecorder(Path(output_dir) / "process_trace",
                       dict(config=asdict(config), source="initialized_state", synthetic=bool(synthetic), inputs={})) as recorder:
        recorder.emit("run.started")
        runner = MCTSRunner.__new__(MCTSRunner)
        runner.initial_state = initial_state.clone()
        runner.config = config
        runner.nmr_scorer = nmr_scorer
        runner.search_guidance = search_guidance
        try:
            return _run_observed_runner(runner, recorder, output_dir, search_guidance)
        except BaseException as exc:
            recorder.emit("run.error", data=dict(type=type(exc).__name__, message=str(exc)))
            recorder.finish("error", error=dict(type=type(exc).__name__, message=str(exc)))
            raise
