"""Exact storage differences; molecular transitions belong to mdp.transition.

The encoding includes every dataclass field, including identity permutations,
contracts, conformers and final substitution coordinates. It is a local internal
format, not a loader for untrusted files. Replay never enumerates candidates.
"""
from dataclasses import fields, is_dataclass, replace
import pickle
import zlib


def _difference(before, after):
    if before is after:
        return None
    if type(before) is not type(after):
        return ('value', after)
    if is_dataclass(after):
        edits = {f.name: d for f in fields(after)
                 if (d := _difference(getattr(before, f.name), getattr(after, f.name))) is not None}
        return ('object', edits) if edits else None
    if isinstance(after, dict):
        removed = tuple(k for k in before if k not in after)
        edits = {}
        for key, value in after.items():
            delta = _difference(before[key], value) if key in before else ('value', value)
            if delta is not None:
                edits[key] = delta
        order = tuple(after) if tuple(before) != tuple(after) else None
        return ('mapping', removed, edits, order) if removed or edits or order is not None else None
    if isinstance(after, (list, tuple)):
        edits = {}
        for index, value in enumerate(after):
            delta = _difference(before[index], value) if index < len(before) else ('value', value)
            if delta is not None:
                edits[index] = delta
        return (('sequence', len(after), edits, isinstance(after, tuple))
                if edits or len(before) != len(after) else None)
    if isinstance(after, (set, frozenset)):
        return ('set', before - after, after - before) if before != after else None
    return ('value', after) if before != after else None


def _apply(before, delta):
    if delta is None:
        return before
    kind, *data = delta
    if kind == 'value':
        return data[0]
    if kind == 'object':
        return replace(before, **{k: _apply(getattr(before, k), v) for k, v in data[0].items()})
    if kind == 'mapping':
        removed, edits, order = data
        result = {k: v for k, v in before.items() if k not in removed}
        for k, v in edits.items():
            result[k] = _apply(before.get(k), v)
        return {k: result[k] for k in order} if order is not None else result
    if kind == 'sequence':
        length, edits, as_tuple = data
        result = list(before[:length]) + [None] * max(0, length - len(before))
        for i, v in edits.items():
            result[i] = _apply(result[i], v)
        return tuple(result) if as_tuple else result
    if kind == 'set':
        return (before - data[0]) | data[1]
    raise ValueError(f'unknown state delta operation: {kind}')


def encode_transition(parent, child):
    edits = {f.name: d for f in fields(child)
             if (d := _difference(getattr(parent, f.name), getattr(child, f.name))) is not None}
    return zlib.compress(pickle.dumps(edits, protocol=5), 1)


def replay_transition(parent, encoded):
    state = parent.clone()
    for name, delta in pickle.loads(zlib.decompress(encoded)).items():
        setattr(state, name, _apply(getattr(state, name), delta))
    state.__post_init__()
    return state
