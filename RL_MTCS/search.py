"""Persistent single-player PUCT over immutable-by-convention state snapshots."""

from dataclasses import dataclass, field
import math
import random
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
    value(state, default_value) returns a finite value in [-1, 1]. Terminal
    success/failure cannot be overridden by a network.
    record(state, children, backed_up_value) receives action visit counts after
    search. Graph encoding and model training remain outside the search engine.
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


@dataclass
class SearchBudget:
    limit: int
    visited: int = 0
    backtracks: int = 0
    exhausted: bool = False
    simulations: int = 0


@dataclass
class _Node:
    state: Any
    parent: Any = None
    action: dict = field(default_factory=dict)
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list = field(default_factory=list)
    iterator: Any = None
    seen: set = field(default_factory=set)
    fully_expanded: bool = False
    closed: bool = False
    solved: bool = False

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


class PersistentMCTS:
    """Select by PUCT, expand lazily, evaluate a leaf, then back up to root.

    The tree persists across simulations; no action is irrevocably committed.
    Progressive widening exposes additional successors as a node is revisited.
    Exhaustion refers only to the action space offered by the adapter.
    """

    def __init__(self, initial, *, expand, accept, signature, evaluate,
                 action=None, observe=None, guidance=None, c_puct=1.414, seed=None,
                 solution_signature=None, max_nodes=256, root_child_limit=None):
        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError('c_puct must be finite and positive')
        self.root = _Node(initial)
        self.expand, self.accept = expand, accept
        self.signature, self.evaluate = signature, evaluate
        self.solution_signature = solution_signature or signature
        self._solution_keys = set()
        self.action = action or (lambda parent, child: {})
        self.observe = observe
        self.guidance = guidance or SearchGuidance()
        self.c_puct = c_puct
        self.random = random.Random(seed)
        self.nodes = [self.root]
        self.solutions = []
        self.dead_ends = 0
        self.max_depth = 0
        self.backup_updates = 0
        self.pruned_subtrees = 0
        self._started = False
        # A persistent tree stores a full HexAssemblyState at every node.
        # Bound the retained frontier so a large simulation budget cannot turn
        # into unbounded process memory.  The search remains bounded and is
        # reported as such by the caller when this limit is reached.
        self.max_nodes = max(1, int(max_nodes))
        self.root_child_limit = (
            None if root_child_limit is None else max(1, int(root_child_limit))
        )
        self.node_limit_reached = False

    @staticmethod
    def _close_subtree(node):
        """Release generator-owned candidate lists before dropping a branch."""
        close = getattr(node.iterator, "close", None)
        if close is not None:
            close()
        node.iterator = None
        for child in node.children:
            PersistentMCTS._close_subtree(child)
        node.children.clear()

    def _make_room(self, protected):
        """Drop one low-value branch while preserving the active path.

        A node contains a complete assembly snapshot, so retaining every
        explored branch defeats a large simulation budget.  Pruning a closed
        or low-visit subtree turns ``max_nodes`` into a hard memory bound while
        preserving live ancestors needed for backtracking.
        """
        protected = set(protected)
        candidates = [
            node for node in self.nodes
            if node is not self.root and id(node) not in protected
            and not node.solved
        ]
        if not candidates:
            return False
        victim = min(
            candidates,
            key=lambda node: (
                0 if node.closed else 1,
                int(node.visits),
                -len(node.children),
            ),
        )
        parent = victim.parent
        if parent is None:
            return False
        try:
            parent.children.remove(victim)
        except ValueError:
            return False
        removed = set()
        stack = [victim]
        while stack:
            current = stack.pop()
            removed.add(id(current))
            stack.extend(current.children)
        self._close_subtree(victim)
        self.nodes = [node for node in self.nodes if id(node) not in removed]
        self.pruned_subtrees += 1
        return True

    def _leaf_value(self, node):
        value = float(self.evaluate(node.state))
        if self.guidance.value is not None:
            value = float(self.guidance.value(node.state, value))
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('MCTS value must be finite and in [-1, 1]')
        return value

    def _backup(self, node, value, *, solved=False):
        while node is not None:
            node.solved = node.solved or solved
            node.visits += 1
            node.value_sum += value
            self.backup_updates += 1
            node = node.parent

    def _solution(self, state):
        key = self.solution_signature(state)
        if key not in self._solution_keys:
            self._solution_keys.add(key)
            self.solutions.append(state)

    def _next_child(self, node):
        if node.iterator is None:
            node.iterator = iter(self.expand(node.state))
        for state in node.iterator:
            key = self.signature(state)
            if key in node.seen:
                continue
            node.seen.add(key)
            if len(self.nodes) >= self.max_nodes:
                protected = set()
                current = node
                while current is not None:
                    protected.add(id(current))
                    current = current.parent
                if not self._make_room(protected):
                    self.node_limit_reached = True
                    # ``expand`` is commonly a generator backed by a list of
                    # full state clones.  Closing it here releases that list
                    # as soon as the bounded tree is full.
                    close = getattr(node.iterator, "close", None)
                    if close is not None:
                        close()
                    node.iterator = None
                    return None
            action = dict(self.action(node.state, state))
            prior = 1.0 / math.sqrt(1 + len(node.children))
            if self.guidance.policy is not None:
                prior = float(self.guidance.policy(node.state, action, prior))
            if not math.isfinite(prior) or prior <= 0:
                raise ValueError('MCTS policy weight must be finite and positive')
            child = _Node(state, parent=node, action=action, prior=prior)
            node.children.append(child)
            self.nodes.append(child)
            if self.observe is not None:
                self.observe(state)
            return child
        node.fully_expanded = True
        return None

    def run(self, budget, *, solution_limit=1):
        if not self._started:
            self._started = True
            budget.visited += 1
            if self.observe is not None:
                self.observe(self.root.state)
            if self.accept(self.root.state):
                self.root.closed = True
                self._solution(self.root.state)
                self._backup(self.root, 1.0, solved=True)
        while (budget.simulations < budget.limit and not self.root.closed
               and len(self.solutions) < solution_limit
               and not self.node_limit_reached):
            budget.simulations += 1
            node = self.root
            depth = 0
            while True:
                live = [child for child in node.children if not child.closed]
                width = max(1, math.ceil(math.sqrt(node.visits + 1)))
                if node is self.root and self.root_child_limit is not None:
                    width = min(width, self.root_child_limit)
                if not node.fully_expanded and (len(node.children) < width or not live):
                    child = self._next_child(node)
                    if child is not None:
                        budget.visited += 1
                        self.max_depth = max(self.max_depth, depth + 1)
                        if self.accept(child.state):
                            child.closed = True
                            self._solution(child.state)
                            value = 1.0
                        else:
                            value = self._leaf_value(child)
                        self._backup(child, value, solved=child.closed)
                        break
                    if self.node_limit_reached:
                        self._backup(node, self._leaf_value(node))
                        break
                if not live:
                    node.closed = True
                    self.dead_ends += int(not node.solved)
                    budget.backtracks += 1
                    # Solved subtrees are exhausted, not unsuccessful.
                    value = 1.0 if node.solved else -1.0
                    self._backup(node, value)
                    break
                total_prior = sum(child.prior for child in live)
                scores = [(child.value + self.c_puct * child.prior / total_prior
                           * math.sqrt(max(1, node.visits)) / (1 + child.visits), child)
                          for child in live]
                best = max(score for score, _child in scores)
                node = self.random.choice([child for score, child in scores
                                           if abs(score - best) < 1e-12])
                depth += 1
        budget.exhausted = (
            (budget.simulations >= budget.limit or self.node_limit_reached)
            and not self.root.closed
            and len(self.solutions) < solution_limit
        )
        return self.solutions[0] if self.solutions else None

    def seed_path(self, states):
        """Retain a validated bounded probe path, including every parent.

        The adapter owns legality checks. This only seeds the same tree used
        by PUCT, so a rollout never commits early actions irreversibly.
        """
        node = self.root
        for state in states:
            if len(self.nodes) >= self.max_nodes:
                break
            key = self.signature(state)
            child = next((c for c in node.children if self.signature(c.state) == key), None)
            if child is None:
                child = _Node(state, parent=node, action=dict(self.action(node.state, state)))
                node.children.append(child)
                node.seen.add(key)
                self.nodes.append(child)
            if self.observe:
                self.observe(state)
            node = child
        if self.accept(node.state):
            node.closed = True
            self._solution(node.state)
            self._backup(node, 1.0, solved=True)
        else:
            self._backup(node, self._leaf_value(node))

    @staticmethod
    def child_statistics(node):
        return [dict(action=child.action, visits=child.visits, value=child.value,
                     prior=child.prior, closed=child.closed) for child in node.children]

    def diagnostics(self):
        return dict(algorithm='persistent_puct', tree_nodes=len(self.nodes),
                    root_visits=self.root.visits, backup_updates=self.backup_updates,
                    dead_ends=self.dead_ends, max_depth=self.max_depth,
                    max_nodes=self.max_nodes,
                    root_child_limit=self.root_child_limit,
                    node_limit_reached=bool(self.node_limit_reached),
                    pruned_subtrees=self.pruned_subtrees,
                    root_children=self.child_statistics(self.root))

    def record(self):
        if self.guidance.record is not None:
            for node in self.nodes:
                if node.state is not None and node.children:
                    self.guidance.record(node.state, self.child_statistics(node), node.value)


def domain_conflict_check(domains, *, complete=True):
    """Run necessary joint checks over retained placement domains.

    ``complete`` states whether every legal option is represented in
    ``domains``.  A conflict found in a bounded pool is useful search risk,
    but it is not a proof that the full geometry has no solution.  The
    returned record therefore carries ``hard``/``bounded`` metadata so
    callers can choose soft scoring for incomplete pools.
    """
    def result(**values):
        values.update(hard=bool(complete), bounded=not bool(complete))
        return values

    forced = {}
    unions = {}
    demand = {}
    for uid, options in domains.items():
        sets = [set(option) for option in options]
        if not sets:
            return result(reason='empty_domain', topologies=[uid])
        unions[uid] = set.union(*sets)
        demand[uid] = min(map(len, sets))
        for site in sorted(set.intersection(*sets)):
            if site in forced:
                return result(reason='shared_required_site', site=site,
                              topologies=[forced[site], uid])
            forced[site] = uid
    remaining = set(unions)
    while remaining:
        group = {min(remaining)}
        remaining -= group
        sites = set(unions[next(iter(group))])
        while True:
            joined = {uid for uid in remaining if unions[uid] & sites}
            if not joined:
                break
            remaining -= joined
            group |= joined
            for uid in joined:
                sites |= unions[uid]
        required = sum(demand[uid] for uid in group)
        if len(sites) < required:
            return result(reason='site_capacity', topologies=sorted(group),
                          sites=sorted(sites), required=required, available=len(sites))
    return None


__all__ = [
    "SearchObservation",
    "SearchGuidance",
    "SearchBudget",
    "PersistentMCTS",
    "domain_conflict_check",
]
