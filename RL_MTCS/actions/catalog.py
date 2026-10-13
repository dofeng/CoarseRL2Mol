"""Stage action scheduling for formal search and quick continuation.

The cap is applied after committing and auditing a complete connection. Failed
sites, conformers and partial multi-port assignments never consume that cap.
"""
from RL_MTCS.geometry import unit_sort_key
from RL_MTCS.reporting.diagnostics import note_candidate
from RL_MTCS.mdp.signatures import search_signature
from RL_MTCS.actions.ordering import preferred_axis, ordered, round_robin
from RL_MTCS.mdp.components import phase_of, choose_anchor, stage_disconnected_components
from RL_MTCS.actions.flex import iter_flex_outcomes, _topology_complexity_key
from RL_MTCS.actions.ring import iter_ring_outcomes
from RL_MTCS.actions.side import iter_side_outcomes, iter_side_repair_outcomes, is_complex_side
from RL_MTCS.actions.substitution import iter_substitution_outcomes
from RL_MTCS.mdp.transition import validate_proposal, prepare_action, apply_action, state_key


def legal_actions(state, phase, limit=10, *, seed=None, allow_side_repacking=False):
    """At most limit audited complete actions; failed scans do not consume it."""
    if phase not in ('rigid', 'flex', 'ring', 'side', 'substitution'):
        raise ValueError(f'unknown assembly phase: {phase}')
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('legal candidate limit must be a positive integer')
    source_key = state_key(state)
    current = state.clone()
    if phase != 'rigid':
        choose_anchor(current)
    if phase == 'flex':
        stage_disconnected_components(current)
    units = sorted((u for u in current.problem.units if phase_of(u) == phase
                    and u.unit_id not in current.placements and u.unit_id not in current.excluded_unit_ids),
                   key=lambda u: (_topology_complexity_key(u),unit_sort_key(u.unit_id)))
    # Shuffle equally complex units only; geometry preferences remain primary.
    from itertools import groupby
    units = [u for _, group in groupby(units, _topology_complexity_key)
             for u in ordered(group, seed, 'topology-units')]
    def with_axes(factory):
        first = preferred_axis(current, phase)
        found = False
        for child in factory(first):
            child = validate_proposal(child, phase)
            if child is None:
                continue
            found = True
            yield child
        if not found:
            for child in factory('V' if first == 'H' else 'H'):
                child = validate_proposal(child, phase)
                if child is not None:
                    yield child
    def raw():
        if phase == 'rigid':
            from RL_MTCS.actions.rigid import iter_rigid_outcomes
            yield from with_axes(lambda axis: iter_rigid_outcomes(current, axis=axis, seed=seed))
        elif phase == 'substitution':
            yield from iter_substitution_outcomes(current)
        elif phase == 'flex':
            found = False
            def outcomes(merge, axis):
                yield from round_robin(iter_flex_outcomes(current,u,merge,axis=axis,seed=seed) for u in units)
            for child in with_axes(lambda axis: outcomes(True, axis)):
                found = True
                yield child
            if not found:
                yield from with_axes(lambda axis: outcomes(False, axis))
        elif phase == 'side':
            blockers = {u.unit_id: set() for u in units}
            found = False
            for complex_side in (True, False):
                group = [u for u in units if is_complex_side(u) == complex_side]
                regions = ('outside','crossing','inside') if complex_side else ('inside','crossing','outside')
                for region in regions:
                    for child in round_robin(iter_side_outcomes(current,u,region=region,seed=seed,
                                              blocked_units=blockers[u.unit_id]) for u in group):
                        found = True
                        yield child
            if allow_side_repacking and not found:
                for complex_side in (True, False):
                    group = [u for u in units if is_complex_side(u) == complex_side]
                    regions = ('outside','crossing','inside') if complex_side else ('inside','crossing','outside')
                    for region in regions:
                        yield from round_robin(iter_side_repair_outcomes(current,u,blockers[u.unit_id],
                                              region=region,seed=seed) for u in group)
        else:
            for region in ('outside', 'crossing', 'inside'):
                yield from round_robin(iter_ring_outcomes(current,u,region=region,seed=seed) for u in units)
    seen = set()
    for child in raw():
        if phase not in ('rigid', 'flex'):
            child = validate_proposal(child, phase)
            if child is None:
                continue
        key = search_signature(child, phase)
        if key in seen:
            continue
        seen.add(key)
        note_candidate(child.action_history[-1].get('topology_unit_id', phase))
        yield prepare_action(state, child, phase, source_key)
        if len(seen) >= limit:
            return


def connection_states(state, phase, limit=10, *, seed=None, allow_side_repacking=False):
    """MCTS state adapter; actions and all transitions use the same interface."""
    for action in legal_actions(state, phase, limit, seed=seed, allow_side_repacking=allow_side_repacking):
        yield apply_action(state, action)
