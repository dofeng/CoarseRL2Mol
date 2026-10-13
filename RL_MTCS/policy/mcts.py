"""Single-round PUCT with compact transitions and streamed statistics."""

from dataclasses import dataclass, field
from collections import deque
import pickle
import zlib
import math
import random
import time
import os
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class SearchObservation:
    """Graph state and stage-specific resources supplied to learning hooks.

    ``state`` is a HexAssemblyState snapshot; inspect ``state.problem`` for
    prescribed graphs and ``state.to_dict()`` for placed geometry. Do not mutate.
    """

    phase: str
    state: Any
    context: dict = field(default_factory=dict)


@dataclass
class SearchGuidance:
    """Optional learning hooks; callbacks must treat states as read-only.

    policy(state, action, default_prior) returns a finite positive weight.
    value(state, default_value) returns a finite value in [-1, 1]. Partial and complete terminals use the same reward scale.
    record(state, children, backed_up_value) receives final action visit counts
    when each node closes, or current counts when its selection round finishes.
    Graph encoding and model training remain outside the search engine.
    Callbacks must not retain full state snapshots.
    """

    policy: Callable | None = None
    value: Callable | None = None
    record: Callable | None = None

    def adapted(self, observation):
        return SearchGuidance(
            policy=(lambda s, a, p: self.policy(observation(s), a, p)) if self.policy else None,
            value=(lambda s, v: self.value(observation(s), v)) if self.value else None,
            record=(lambda s, c, v: self.record(observation(s), c, v)) if self.record else None,
        )


@dataclass(slots=True)
class _Node:
    # A transition contains no live molecular state or suspended generator.
    transition: Any = None
    parent: Any = None
    action: dict = field(default_factory=dict)
    prior: float = 1.0
    visits: int = 0
    reward_visits: int = 0
    value_sum: float = 0.0
    children: list = field(default_factory=list)
    pending: Any = None
    fully_expanded: bool = False
    closed: bool = False
    root_action_index: int | None = None

    @property
    def value(self):
        return self.value_sum / self.reward_visits if self.reward_visits else 0.0


class MCTS:
    """PUCT with compact transitions and retirement of exhausted subtrees.

    With result_limit, each simulation descends to a round endpoint and the
    best of that many distinct result candidates is returned. Endpoints may
    be legal partial states with no actions; completeness is never required.
    Without result_limit, the generic engine explores the admitted finite tree.
    Molecular adapters supply exact delta encoding/replay. The generic fallback
    serializes a child, never retaining the caller's live mutable object.
    Record callbacks are streamed at node closure or round selection.
    """
    def __init__(self, initial, *, expand, accept, signature, evaluate,
                 action=None, observe=None, guidance=None, c_puct=1.414,
                 seed=None, excluded=(), encode=None, replay=None, progress=None,
                 result_limit=None, tie_break=None, order_by_value=False):
        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError('c_puct must be finite and positive')
        if (encode is None) != (replay is None):
            raise ValueError('encode and replay must be supplied together')
        if result_limit is not None and (isinstance(result_limit, bool)
                or not isinstance(result_limit, int) or result_limit < 1):
            raise ValueError('result_limit must be a positive integer')
        self.initial = initial
        self.root = _Node()
        self.expand, self.accept = expand, accept
        self.signature, self.evaluate = signature, evaluate
        self.action = action or (lambda p, c: {})
        self.observe = observe
        self.guidance = guidance or SearchGuidance()
        self.encode = encode or (lambda p, c: zlib.compress(pickle.dumps(c, protocol=5), 1))
        self.replay = replay or (lambda p, delta: pickle.loads(zlib.decompress(delta)))
        self.c_puct = c_puct
        self.random = random.Random(seed)
        self.excluded = set(excluded)
        self.result_limit = result_limit
        self.result_signatures = set()
        self.result_signatures_in_order = []
        self.result_rewards = []
        self.root_action_count = None
        self.root_evaluated_actions = set()
        self.stop_reason = None
        self.best_state = None
        self.best_value = -math.inf
        self.tie_break = tie_break
        self.order_by_value = order_by_value
        self.best_tie = None
        self.simulations = self.backup_updates = self.max_depth = self.terminal_count = 0
        self.visited_states = 1
        self.retained_nodes = self.peak_retained_nodes = 1
        self.transition_bytes = self.peak_transition_bytes = 0
        self.progress = progress
        self.started_at = self.last_report = time.monotonic()

    def _report_progress(self):
        now = time.monotonic()
        if self.progress is not None and now - self.last_report >= 15:
            self.last_report = now
            self.progress(self.diagnostics())

    def _value(self, state):
        value = float(self.evaluate(state))
        if self.guidance.value is not None:
            value = float(self.guidance.value(state, value))
        if not math.isfinite(value):
            raise ValueError('MCTS value must be finite')
        return value

    def _backup(self, node, value):
        while node is not None:
            node.visits += 1
            if value is not None:
                node.reward_visits += 1
                node.value_sum += value
            self.backup_updates += 1
            node = node.parent

    def _terminal(self, node, state):
        value = float(self.evaluate(state))
        if not math.isfinite(value):
            raise ValueError('MCTS terminal reward must be finite')
        node.closed = True
        self.terminal_count += 1
        branch = node
        while branch.parent is not None and branch.parent is not self.root:
            branch = branch.parent
        if branch.root_action_index is not None:
            self.root_evaluated_actions.add(branch.root_action_index)
        key = self.signature(state)
        if key in self.excluded or key in self.result_signatures:
            # A rejected whole result was visited, but is not a selectable
            # value sample. Its individual prefix actions remain admissible.
            return None
        if key not in self.excluded and (self.result_limit is None
                                        or key not in self.result_signatures):
            if self.result_limit is not None:
                # At most result_limit entries, never all intermediate states.
                if len(self.result_signatures) < self.result_limit:
                    self.result_signatures.add(key)
                    self.result_rewards.append(value)
                else:
                    # Root coverage can finish after the representative pool
                    # fills. Retain only its best results, including that root.
                    worst = min(range(len(self.result_rewards)), key=self.result_rewards.__getitem__)
                    if value > self.result_rewards[worst]:
                        previous = self.result_signatures_in_order[worst]
                        self.result_signatures.remove(previous)
                        self.result_signatures.add(key)
                        self.result_rewards[worst] = value
                        self.result_signatures_in_order[worst] = key
                if len(self.result_signatures_in_order) < len(self.result_rewards):
                    self.result_signatures_in_order.append(key)
            tie = self.tie_break(state) if self.tie_break is not None else None
            if (self.best_state is None or value > self.best_value
                    or (value == self.best_value and tie is not None and tie > self.best_tie)):
                self.best_state, self.best_value = state, value
                self.best_tie = tie
        return value

    def _prepare(self, node, state):
        # The assembly adapter yields at most the configured LEGAL candidates.
        # Consume once, retaining compact transitions rather than generators
        # which would keep every temporary clone in their frames alive.
        pending, seen = deque(), set()
        ranked = []
        stream = iter(self.expand(state))
        try:
            for child in stream:
                key = self.signature(child)
                if key in seen:
                    continue
                seen.add(key)
                delta = self.encode(state, child)
                action = dict(self.action(state, child))
                if self.order_by_value:
                    ranked.append((self._value(child), delta, action))
                else:
                    pending.append((delta, action))
                self.transition_bytes += len(delta)
                self.peak_transition_bytes = max(self.peak_transition_bytes, self.transition_bytes)
                self._report_progress()
        finally:
            if hasattr(stream, 'close'):
                stream.close()
        if self.order_by_value:
            ranked.sort(key=lambda row: row[0], reverse=True)
            pending.extend((delta, action) for _, delta, action in ranked)
        node.pending = pending
        if node is self.root:
            self.root_action_count = len(pending)

    def _root_covered(self):
        return self.root.closed or (self.root_action_count is not None
            and len(self.root_evaluated_actions) == self.root_action_count)

    def _retire(self, node, state):
        if self.guidance.record:
            self.guidance.record(state, [dict(action=c.action, visits=c.visits,
                value=c.value, prior=c.prior, closed=c.closed) for c in node.children], node.value)
        # All children must have closed; never discard a live frontier.
        assert all(c.closed for c in node.children)
        assert not node.pending
        self.retained_nodes -= len(node.children)
        node.children.clear()
        node.pending = None
        if node.transition is not None:
            self.transition_bytes -= len(node.transition)
            node.transition = None

    def run(self):
        if self.stop_reason is not None:
            return self.best_state
        if self.observe:
            self.observe(self.initial)
        while not self.root.closed and (self.result_limit is None
                or len(self.result_signatures) < self.result_limit or not self._root_covered()):
            self._report_progress()
            self.simulations += 1
            node, state, depth = self.root, self.initial, 0
            while True:
                if self.accept(state):
                    self._backup(node, self._terminal(node, state))
                    self._retire(node, state)
                    break
                live = [child for child in node.children if not child.closed]
                width = max(1, math.ceil(math.sqrt(node.visits + 1)))
                root_pending = node is self.root and self.result_limit is not None
                if not node.fully_expanded and (root_pending or len(node.children) < width or not live):
                    if node.pending is None:
                        self._prepare(node, state)
                    if node.pending:
                        delta, action = node.pending.popleft()
                        child_state = self.replay(state, delta)
                        prior = float(self.guidance.policy(state, action, 1.0)) if self.guidance.policy else 1.0
                        if not math.isfinite(prior) or prior <= 0:
                            raise ValueError('MCTS policy weight must be finite and positive')
                        child = _Node(transition=delta, parent=node, action=action, prior=prior)
                        if node is self.root:
                            child.root_action_index = len(node.children)
                        node.children.append(child)
                        self.visited_states += 1
                        self.retained_nodes += 1
                        self.peak_retained_nodes = max(self.peak_retained_nodes, self.retained_nodes)
                        self.max_depth = max(self.max_depth, depth + 1)
                        if self.observe:
                            self.observe(child_state)
                        terminal = self.accept(child_state)
                        if self.result_limit is not None and not terminal:
                            # Complete a simulation now, rather than growing a
                            # broad frontier of one-step value estimates before
                            # any usable result reaches the checkpoint.
                            node, state = child, child_state
                            depth += 1
                            continue
                        value = self._terminal(child, child_state) if terminal else self._value(child_state)
                        self._backup(child, value)
                        if terminal:
                            self._retire(child, child_state)
                        break
                    node.fully_expanded = True
                live = [child for child in node.children if not child.closed]
                if not live and node.fully_expanded:
                    if not node.children:
                        value = self._terminal(node, state)
                    else:
                        node.closed = True
                        eligible = [child.value for child in node.children if child.reward_visits]
                        value = max(eligible) if eligible else None
                    self._backup(node, value)
                    self._retire(node, state)
                    break
                total = sum(child.prior for child in live)
                scores = [(child.value + self.c_puct * child.prior / total
                           * math.sqrt(max(1, node.visits)) / (1 + child.visits), child)
                          for child in live]
                maximum = max(value for value, _ in scores)
                chosen = self.random.choice([child for value, child in scores
                                             if abs(value - maximum) < 1e-12])
                state = self.replay(state, chosen.transition)
                node = chosen
                depth += 1
        self.stop_reason = 'tree_exhausted' if self.root.closed else 'result_candidates_ready'
        if not self.root.closed:
            self._release_round()
        return self.best_state

    def _release_round(self):
        """The round has selected its result; release its remaining work tree.

        This is a candidate-selection stop, not a claim that unvisited branches
        were exhausted. Emit current learning statistics for open nodes once.
        """
        stack = [(self.root, self.initial)]
        while stack:
            node, state = stack.pop()
            if self.guidance.record and not node.closed:
                self.guidance.record(state, [dict(action=c.action, visits=c.visits,
                    value=c.value, prior=c.prior, closed=c.closed) for c in node.children], node.value)
            for child in node.children:
                if not child.closed:
                    child_state = self.replay(state, child.transition) if self.guidance.record else None
                    stack.append((child, child_state))
                child.parent = None
            node.children.clear()
            node.pending = None
            node.transition = None
        self.retained_nodes = 1
        self.transition_bytes = 0

    def diagnostics(self):
        values = dict(simulations=self.simulations, visited_states=self.visited_states,
                    max_depth=self.max_depth, terminal_count=self.terminal_count,
                    backup_updates=self.backup_updates, search_exhausted=self.root.closed,
                    stop_reason=self.stop_reason,
                    result_candidate_count=len(self.result_signatures),
                    result_candidate_limit=self.result_limit,
                    result_candidate_rewards=list(self.result_rewards),
                    selected_reward=self.best_value if self.best_state is not None else None,
                    selected_tie_break=self.best_tie,
                    root_legal_action_count=self.root_action_count or 0,
                    root_evaluated_action_count=len(self.root_evaluated_actions),
                    root_actions_covered=self._root_covered(),
                    retained_nodes=self.retained_nodes, peak_retained_nodes=self.peak_retained_nodes,
                    transition_bytes=self.transition_bytes, peak_transition_bytes=self.peak_transition_bytes,
                    elapsed_seconds=time.monotonic()-self.started_at)
        try:
            values['rss_mib'] = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1024**2
        except (OSError, ValueError, IndexError):
            pass
        return values

__all__ = ["SearchObservation", "SearchGuidance", "MCTS"]
