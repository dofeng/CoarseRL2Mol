"""Deterministic scheduling preferences; never restrict failed-position scans."""
from collections import deque
from hashlib import sha256
import random


def preferred_axis(state, phase):
    previous = getattr(state, f'last_{phase}_axis')
    return 'V' if previous == 'H' else 'H'


def ordered(rows, seed=None, namespace=''):
    rows = list(rows)
    if seed is not None:
        key = int.from_bytes(sha256(f'{seed}:{namespace}'.encode('ascii')).digest()[:8], 'big')
        random.Random(key).shuffle(rows)
    return rows


def round_robin(streams):
    pending = deque(iter(stream) for stream in streams)
    try:
        while pending:
            stream = pending[0]
            try:
                value = next(stream)
            except StopIteration:
                pending.popleft()
                continue
            yield value
            pending.rotate(-1)
    finally:
        for stream in pending:
            if hasattr(stream, 'close'):
                stream.close()
