"""Exterior-preferred ring attachment with legal spatial fallback."""
from RL_MTCS.geometry import hull_region
from RL_MTCS.actions.conformers import iter_conformer_states
from RL_MTCS.actions.flex import _probe, _interior_site_poses, _place_interior_pose
from RL_MTCS.mdp.transition import commit_attachment
from RL_MTCS.reporting.diagnostics import note_unit


def iter_ring_outcomes(state, unit, *, region=None, seed=None):
    note_unit(unit.unit_id)
    preference = (region,) if region is not None else ('outside', 'crossing', 'inside')
    for region in preference:
        for trial in iter_conformer_states(state, unit.unit_id):
            candidate, components = _probe(trial)
            if candidate is None:
                return
            for pose in _interior_site_poses(candidate, unit.unit_id, components, seed=seed):
                coordinates = trial._placement_coordinates(trial.unit(unit.unit_id), pose[0], pose[1], pose[2])
                if hull_region(state.backbone_hull, coordinates.values()) != region:
                    continue
                result = _place_interior_pose(candidate, unit.unit_id, pose)
                if result is not None:
                    committed = commit_attachment(result, state, 'ring')
                    if committed is not None:
                        yield committed
