"""Fixed original task inventory and per-stage completion."""
from RL_MTCS.mdp.components import main_units, phase_of, anchor_for


def inventory_for(state):
    if state.task_inventory:
        return state.task_inventory
    from RL_MTCS.actions.substitution import _request_rows
    return tuple((phase_of(u),u.unit_id) for u in state.problem.units) + tuple(
        ('substitution',r.target_node_id) for r in (state.problem.hetero_requests or _request_rows(state)))


def ensure_inventory(state):
    state.task_inventory = inventory_for(state)
    return state.task_inventory


def completion_report(state):
    inventory = inventory_for(state)
    connected = main_units(state)
    substitutions = {a.get('request_target',a.get('representative_node_id',a.get('site_ids',[None])[0]))
                     for a in state.action_history if a.get('kind') == 'substitution'}
    for request in state.problem.hetero_requests:
        owner = state.problem.ownership_map().get(request.target_node_id)
        if owner not in connected:
            continue
        node = state.unit(owner).node_map().get(request.target_node_id)
        if (node is not None and node.su_type == request.target_su
                and state.current_su_types.get(request.target_node_id) == request.target_su
                and (request.ring_size == 6 or request.synthetic_node_id not in state.node_axial)):
            substitutions.add(request.target_node_id)
    stages = {p:dict(total=0,completed=0,unconnected=[]) for p in ('rigid','flex','ring','side','substitution')}
    for phase, uid in inventory:
        row = stages[phase]
        row['total'] += 1
        done = uid in (substitutions if phase == 'substitution' else connected)
        row['completed'] += int(done)
        if not done:
            row['unconnected'].append(uid)
    total = sum(r['total'] for r in stages.values())
    completed = sum(r['completed'] for r in stages.values())
    return dict(total=total,completed=completed,completion=completed/total if total else 1.0,
                all_complete=completed==total and state.phase_done("rigid") and not state.mandatory_contracts()
                             and not state.problem.input_issues and not state.problem.unassembled_closure_resources
                             and not state.problem.idle_aromatic_nodes,
                stages=stages,main_component=sorted(connected),
                main_anchor=anchor_for(state),excluded_units=sorted(state.excluded_unit_ids))
