"""Selected-result checkpoints and local rollback, without global budgets."""
from hashlib import sha256
from RL_MTCS.config import AssemblyConfig
from RL_MTCS.policy.mcts import MCTS, SearchObservation
from RL_MTCS.actions.catalog import connection_states
from RL_MTCS.mdp.components import finish_flex, phase_of, choose_anchor
from RL_MTCS.policy.rollout import fast_rollout_remaining
from RL_MTCS.reporting.completion import completion_report, ensure_inventory
from RL_MTCS.rewards.scoring import score_state, su13_clearance_profile, nmr_quality
from RL_MTCS.rewards.demand import spatial_demand
from RL_MTCS.storage.state_delta import replay_transition
from RL_MTCS.mdp.signatures import search_signature
from RL_MTCS.reporting.diagnostics import entry_profile


class StageController:
    def __init__(self,config=None,nmr_scorer=None,guidance=None):
        self.config = config or AssemblyConfig()
        self.nmr_scorer,self.guidance = nmr_scorer,guidance
        self.forward_cache = {}
        self.checkpoints = []

    def reward(self,state,forward=None,*,demand=None):
        nmr = None
        if self.nmr_scorer is not None and state.stage != 'rigid':
            phase = dict(flex='flex_chain',ring='aliphatic_ring').get(state.stage,state.stage)
            row = self.nmr_scorer.score_state(state,stage=phase)
            if row.get('scored',True):
                nmr = nmr_quality(row.get('loss'))
        return score_state(state,completion=completion_report(state)['completion'],
            nmr=nmr,forward=forward,weights=self.config.reward_weights,
            clearance=su13_clearance_profile(state,self.config.clearance_radius)['quality'],
            demand=demand)

    @staticmethod
    def entrance_tie_break(state,demand):
        # More entrances than the frozen remaining port demand bring no extra
        # tie advantage. Finished attachment tasks need no entrances at all.
        if not spatial_demand(state,demand.phase).remaining_ports:
            return 0
        return min(demand.remaining_ports,
                   entry_profile(state,main_only=demand.phase!='rigid')['available_entries'])

    @staticmethod
    def count(state,phase):
        if phase == 'substitution':
            return sum(a.get('kind')=='substitution' for a in state.action_history)
        return sum(phase_of(state.unit(uid)) == phase for uid in state.placements)

    def search(self,initial,phase,excluded=(),*,demand=None):
        current = initial.clone()
        current.stage = phase
        ensure_inventory(current)
        if phase != 'rigid':
            choose_anchor(current)
        demand = spatial_demand(current,phase) if demand is None else demand
        start = self.count(current,phase)
        interval = self.config.forward_checkpoint_interval
        seed = (None if self.config.seed is None else int.from_bytes(sha256(
            f'{self.config.seed}:{phase}:{len(self.checkpoints)}:{len(excluded)}'.encode('ascii')
        ).digest()[:8], 'big'))
        def progress(row):
            print(f"[RL_MTCS][{phase}][search] "
                  f"candidates={row['result_candidate_count']}/{row['result_candidate_limit']} "
                  f"root_actions={row['root_evaluated_action_count']}/{row['root_legal_action_count']} "
                  f"visited={row['visited_states']} "
                  f"retained={row['retained_nodes']} "
                  f"delta_mib={row['transition_bytes']/1024**2:.2f} "
                  f"rss_mib={row.get('rss_mib', 0):.1f} "
                  f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        if self.config.verbose:
            print(f"[RL_MTCS][{phase}][search] start_count={start} "
                  f"candidate_limit={self.config.connection_candidate_limit} "
                  f"space_pressure={demand.pressure:.3f} "
                  f"pending_nodes={demand.remaining_nodes} "
                  f"pending_ports={demand.remaining_ports}", flush=True)
        # No forward simulations in tree evaluation. Cached forward values are
        # supplied only when comparing the selected checkpoint attempts.
        tree = MCTS(current,
            expand=lambda s:connection_states(s,phase,self.config.connection_candidate_limit,seed=seed,
                                              allow_side_repacking=phase=='side'),
            accept=lambda s: phase == 'flex' and self.count(s,phase)-start >= interval,
            signature=lambda s:search_signature(s,phase),
            evaluate=lambda s:self.reward(s,demand=demand)['total'],
            action=lambda p,c:c.action_history[-1] if c.action_history else {},
            encode=lambda p,c:c._connection_action.payload, replay=replay_transition,
            progress=progress if self.config.verbose else None,
            result_limit=self.config.connection_candidate_limit, order_by_value=True,
            # Only exact reward ties compare entrances, up to actual demand.
            # This diagnostic never invokes forward or changes the weights.
            tie_break=(lambda s:self.entrance_tie_break(s,demand))
                      if demand.remaining_ports else None,
            c_puct=self.config.c_puct,seed=seed,excluded=excluded,
            guidance=self.guidance.adapted(lambda s:SearchObservation(
                phase,s,context={'spatial_demand':demand.to_dict()})) if self.guidance else None)
        chosen = tree.run()
        if chosen is not None:
            chosen.__dict__.pop('_connection_action', None)
        if self.config.verbose:
            row = tree.diagnostics()
            print(f"[RL_MTCS][{phase}][search] selected "
                  f"candidates={row['result_candidate_count']}/{row['result_candidate_limit']} "
                  f"root_actions={row['root_evaluated_action_count']}/{row['root_legal_action_count']} "
                  f"reward={row['selected_reward']} stop={row['stop_reason']} "
                  f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        return chosen,dict(tree.diagnostics(), search_seed=seed,
                           spatial_demand=demand.to_dict())

    def checkpoint(self,initial,phase):
        excluded, attempts = set(), []
        demand = spatial_demand(initial,phase)
        best = None
        accepted = False
        for retry in range(self.config.forward_max_rollbacks+1):
            selected, search = self.search(initial, phase, excluded, demand=demand)
            if selected is None:
                break
            signature = search_signature(selected,phase)
            forward_key = selected.forward_signature()
            forward = self.forward_cache.get(forward_key)
            if forward is None:
                if self.config.verbose:
                    print(f"[RL_MTCS][{phase}][forward] start attempt={retry}", flush=True)
                    def progress(row):
                        entries = row.get('entries')
                        extra = (f" entries={entries['available_entries']}/{entries['free_su13']}"
                                 if entries is not None else '')
                        if row.get('unit_id') is not None:
                            extra += f" unit={row['unit_id']} checks={row['checks']}"
                        print(f"[RL_MTCS][{phase}][forward:{row['phase']}] "
                              f"{row['event']} connected={row['connected']} "
                              f"elapsed={row['elapsed_seconds']:.1f}s{extra}", flush=True)
                    forward = fast_rollout_remaining(selected, progress=progress).to_dict()
                else:
                    forward = fast_rollout_remaining(selected).to_dict()
            valid = bool(forward['audit']['legal'])
            reward = self.reward(selected, forward['forward_completion'] if valid else None,
                                 demand=demand)
            record = dict(attempt=retry, search=search, forward=forward, rewards=reward,
                          forward_valid=valid)
            attempts.append(record)
            accepted = valid and forward['forward_completion'] >= self.config.forward_min_completion
            if valid and (best is None or accepted or reward['total'] > best[1]['total']):
                best = (selected, reward, record, forward_key)
            if self.config.verbose:
                print(f"[RL_MTCS][{phase}][checkpoint={self.count(selected,phase)}] "
                      f"attempt={retry} forward_completion={forward['forward_completion']:.3f} "
                      f"forward_valid={valid}", flush=True)
            if accepted:
                break
            excluded.add(signature)
            # Attempts retain reports only; losing molecular snapshots die here.
            del selected
        if best is None:
            raise RuntimeError('checkpoint has no result with a valid forward audit; '
                               'this is an audit failure, not exhausted legal connections')
        selected, reward, record, forward_key = best
        # Only the accepted endpoint can be reused by the next checkpoint.
        # Historical reports remain in traces, not in a full-state-key cache.
        self.forward_cache = {forward_key: record['forward']}
        trace = dict(phase=phase, start_count=self.count(initial,phase),
                     end_count=self.count(selected,phase), selected_attempt=record['attempt'],
                     attempts=attempts, fallback=not accepted,
                     spatial_demand=demand.to_dict())
        self.checkpoints.append(trace)
        return selected, dict(rewards=reward, fast_rollout=record['forward'], retry_selection=trace)

    def run(self,state,phase):
        initial = state.clone()
        ensure_inventory(initial)
        if phase == 'rigid':
            current,info = self.checkpoint(initial,phase)
        elif phase == 'flex':
            current,info = initial,{}
            first = len(self.checkpoints)
            while True:
                current,info = self.checkpoint(current,'flex')
                if next(connection_states(current,'flex',limit=1),None) is None:
                    # The selected state was already evaluated, including a
                    # zero-progress or short last segment. Never evaluate twice.
                    current = finish_flex(current)
                    break
            info['checkpoints'] = self.checkpoints[first:]
        else:
            demand = spatial_demand(initial,phase)
            current,search = self.search(initial,phase,demand=demand)
            info = dict(search_diagnostics=search,rewards=self.reward(current,demand=demand))
        current.stage = phase
        report = completion_report(current)
        row = report['stages'][phase]
        units_done = row['completed']
        stage_complete = units_done == row['total']
        if phase == 'rigid':
            # Rigid assembly produces several clusters. Joining those clusters
            # is a flex task, not a prerequisite for rigid-stage completion.
            units_done = self.count(current, phase)
            stage_complete = current.phase_done('rigid')
            info['contracts_done'] = sum(
                c.kind == 'rigid' and c.contract_id in current.satisfied_contracts
                for c in current.problem.contracts)
            info['contracts_total'] = sum(
                su == 10 for _, su in current.problem.node_su_types) // 2
        info.update(status='complete' if stage_complete else 'partial',
                    done=True,stage_no_actions=True,all_complete=report['all_complete'],
                    units_done=units_done,units_total=row['total'],completion_report=report,
                    legal=current.audit()['legal'],main_anchor=current.main_anchor,
                    excluded_units=sorted(current.excluded_unit_ids),optimality='best_in_searched_candidates')
        info['connection_policy'] = 'direct_then_single_side_repacking' if phase=='side' else 'direct'
        next_phase = dict(rigid='flex',flex='ring',ring='side',side='substitution',
                          substitution='final')[phase]
        info['remaining_spatial_demand'] = spatial_demand(current,next_phase).to_dict()
        return current,info
