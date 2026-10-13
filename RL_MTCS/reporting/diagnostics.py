"""Count-only candidate failures and actual SU13 attachment entrances."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_scan = ContextVar('rl_mtcs_candidate_scan', default=None)


class CandidateDiagnostics:
    def __init__(self, progress=None):
        self.units = {}
        self.current_unit = None
        self.progress = progress
        self.last_report = time.monotonic()

    def tick(self):
        now = time.monotonic()
        if self.progress is not None and now-self.last_report >= 15:
            self.last_report = now
            row = self.units.get(self.current_unit, {})
            self.progress(dict(unit_id=self.current_unit, checks=row.get('checks', 0),
                               rejection_counts=dict(row.get('rejection_counts', {}))))

    def unit(self, uid):
        self.current_unit = uid
        return self.units.setdefault(uid, dict(checks=0, legal_candidates=0, rejection_counts=Counter()))

    def report(self):
        return {uid: dict(checks=row['checks'], legal_candidates=row['legal_candidates'],
                          rejection_counts=dict(row['rejection_counts']),
                          status='legal_candidate_found' if row['legal_candidates'] else 'no_legal_candidate')
                for uid, row in self.units.items()}


@contextmanager
def collect_candidates(diagnostics):
    token = _scan.set(diagnostics)
    try:
        yield
    finally:
        _scan.reset(token)


def note_unit(uid):
    recorder = _scan.get()
    if recorder is not None:
        recorder.unit(uid)


def note_rejection(reason, uid=None):
    recorder = _scan.get()
    if recorder is not None and (uid is not None or recorder.current_unit is not None):
        row = recorder.unit(uid or recorder.current_unit)
        row['rejection_counts'][reason.split(':', 1)[0]] += 1
        recorder.tick()


def note_candidate(uid):
    recorder = _scan.get()
    if recorder is not None:
        recorder.unit(uid)['legal_candidates'] += 1


def note_audit_failure(audit):
    if _scan.get() is None:
        return
    note_rejection('commit_audit_failure')
    for field in ('actual_geometry_errors', 'non_hex_edges', 'su25_center_branch_errors',
                  'port_direction_errors', 'topology_conformation_errors', 'substitution_errors',
                  'ring_attachment_errors', 'flex_endpoint_su_errors', 'aromatic_template_errors'):
        for error in audit.get(field, ()):
            note_rejection(error.get('reason', field) if isinstance(error, dict) else field)


def record_feasibility(method):
    @wraps(method)
    def checked(self, unit_id, *args, **kwargs):
        result = method(self, unit_id, *args, **kwargs)
        recorder = _scan.get()
        if recorder is not None:
            row = recorder.unit(recorder.current_unit or unit_id)
            row['checks'] += 1
            if not result[0]:
                row['rejection_counts'][result[1].split(':', 1)[0]] += 1
            recorder.tick()
        return result
    return checked


def entry_profile(state, *, main_only=False, include_sites=False):
    """Axial first-step availability for an ordinary one-node side chain.

    This is a geometry diagnostic, not a guarantee that a larger topology fits.
    Check all currently placed aromatic components; excluded units are absent.
    Clearance radius/counting and reward weights are untouched.
    """
    from RL_MTCS.actions.flex import _component_slot_records
    from RL_MTCS.geometry import axial_add, hex_edge_valid
    if main_only:
        from RL_MTCS.mdp.components import main_units
        placed = main_units(state)
    else:
        placed = state.placements
    active = [uid for uid in placed if state.unit(uid).phase == 'aromatic']
    slots = _component_slot_records(state, active, ())
    reasons = Counter()
    rows = []
    for uid, node, point, direction in slots:
        target = axial_add(point, direction)
        blockers = []
        if not hex_edge_valid(point, target, honeycomb=True):
            reason = 'invalid_honeycomb_direction'
        elif target in state.occupied_axial:
            reason = 'coordinate_collision'
            blockers.append(state.occupied_axial[target])
        else:
            reason = 'available'
        reasons[reason] += 1
        if include_sites:
            rows.append(dict(unit_id=uid, site_id=node, target=list(target),
                             status=reason, blocking_nodes=sorted(blockers)))
    result = dict(free_su13=len(slots), available_entries=reasons['available'],
                  blocked_entries=len(slots)-reasons['available'], reasons=dict(reasons),
                  scope='main_component' if main_only else 'placed_components',
                  probe='ordinary_single_node_axial_attachment')
    if include_sites:
        result['sites'] = rows
    return result
