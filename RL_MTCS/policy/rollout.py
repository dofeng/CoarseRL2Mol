"""One greedy legal continuation of all stages at rigid/flex checkpoints."""
from dataclasses import dataclass, field
import time
from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.mdp.components import finish_flex
from RL_MTCS.reporting.completion import ensure_inventory, completion_report
from RL_MTCS.reporting.diagnostics import CandidateDiagnostics, collect_candidates, entry_profile




@dataclass
class FastRolloutResult:
    state: object
    report: dict
    audit: dict
    diagnostics: dict = field(default_factory=dict)

    @property
    def forward_completion(self):
        return self.report['completion']

    @property
    def unplaced_unit_count(self):
        return sum(len(r['unconnected']) for p,r in self.report['stages'].items() if p != 'substitution')

    @property
    def status(self):
        return 'complete' if self.report['all_complete'] else 'partial'

    def to_dict(self):
        rows = self.report['stages']
        return dict(self.report,forward_completion=self.forward_completion,status=self.status,
                    placed_unit_count=sum(r['completed'] for p,r in rows.items() if p != 'substitution'),
                    unplaced_unit_count=self.unplaced_unit_count,
                    unplaced_units={p:list(r['unconnected']) for p,r in rows.items()},
                    unplaced_contracts=[c.contract_id for c in self.state.mandatory_contracts()],
                    audit=self.audit,stage_no_actions=True, diagnostics=self.diagnostics)


def fast_rollout_remaining(state, *, progress=None):
    current = state.clone()
    ensure_inventory(current)
    started = time.monotonic()
    diagnostics = {}
    # Do not isolate disconnected clusters before finishing remaining chains.
    for phase in ('flex','ring','side','substitution'):
        added = 0
        before = entry_profile(current) if phase != 'substitution' else None
        last_report = time.monotonic()
        if progress:
            progress(dict(phase=phase, event='start', connected=added,
                          elapsed_seconds=last_report-started))
        while True:
            def scanning(row):
                progress(dict(row, phase=phase, event='scan', connected=added,
                              elapsed_seconds=time.monotonic()-started))
            scan = CandidateDiagnostics(scanning if progress is not None else None)
            with collect_candidates(scan):
                child = next(connection_states(current,phase,limit=1),None)
            if child is None:
                break
            current = child
            added += 1
            now = time.monotonic()
            if progress and now-last_report >= 15:
                progress(dict(phase=phase, event='progress', connected=added,
                              elapsed_seconds=now-started))
                last_report = now
        if phase == 'flex':
            current = finish_flex(current)
        diagnostics[phase] = dict(connected=added, no_actions=True,
            entries_before=before, entries_after=entry_profile(current) if phase != 'substitution' else None,
            unconnected=scan.report())
        if progress:
            progress(dict(phase=phase, event='no_actions', connected=added,
                          elapsed_seconds=time.monotonic()-started,
                          entries=diagnostics[phase]['entries_after']))
    audit = current.audit()
    return FastRolloutResult(current,completion_report(current),audit,diagnostics)


__all__ = ['fast_rollout_remaining','FastRolloutResult']
