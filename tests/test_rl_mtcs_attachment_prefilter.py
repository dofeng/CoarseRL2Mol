"""Necessary endpoint occupancy rejects impossible conformers before products."""
from dataclasses import replace

from RL_MTCS.actions import catalog as connections, flex
from RL_MTCS.geometry import transform_axial, axial_add
from test_rl_mtcs_three_terminal import three_terminal_fixture


def test_footprint_is_required_by_every_free_aromatic_slot():
    state, _, _ = three_terminal_fixture()
    footprint = flex._mandatory_aromatic_footprint(state)
    assert len(footprint) == 6
    for uid, _, point, outward in flex._component_slot_records(
            state, sorted(state.placements), ()):
        for reflected in (False, True):
            for rotation in range(6):
                if transform_axial(outward, rotation, reflected) != (1, 0):
                    continue
                occupied = {transform_axial((state.node_axial[n.node_id][0]-point[0],
                    state.node_axial[n.node_id][1]-point[1]), rotation, reflected)
                    for n in state.unit(uid).nodes}
                assert footprint <= occupied


def test_impossible_initial_conformer_skipped_and_next_legal_conformer_committed(monkeypatch):
    state, _, _ = three_terminal_fixture()
    initial = state.unit('topology:0')
    footprint = flex._mandatory_aromatic_footprint(state)
    assert not flex._attachment_footprints_clear(initial, footprint)
    seen = []
    original = flex._backbone_site_outcomes

    def checked(candidate, uid, *args, **kwargs):
        variant = candidate.state.unit(uid)
        assert flex._attachment_footprints_clear(variant, footprint)
        seen.append(variant)
        yield from original(candidate, uid, *args, **kwargs)

    monkeypatch.setattr(flex, '_backbone_site_outcomes', checked)
    signature = state.state_signature()
    result = next(connections.connection_states(state, 'flex', 1))
    assert seen and seen[0] != initial
    assert result.component_count() == 1
    assert result.audit()['geometry_valid']
    assert state.state_signature() == signature
    # Feasibility is independent of global translation, reflection and rotation.
    for variant in (initial, seen[0]):
        expected = flex._attachment_footprints_clear(variant, footprint)
        for reflected in (False, True):
            for rotation in range(6):
                transformed = replace(variant,
                    nodes=tuple(replace(n, local_axial=axial_add((12, -6),
                        transform_axial(n.local_axial, rotation, reflected))) for n in variant.nodes),
                    ports=tuple(replace(p, direction=transform_axial(p.direction, rotation, reflected))
                                for p in variant.ports))
                assert flex._attachment_footprints_clear(transformed, footprint) == expected


def test_no_remaining_clusters_does_not_enumerate_merge_conformers(monkeypatch):
    state, _, _ = three_terminal_fixture()
    connected = next(connections.connection_states(state, 'flex', 1))
    assert connected.component_count() == 1

    def unexpected(*args, **kwargs):
        raise AssertionError('no outside cluster: merge conformers must not be enumerated')

    monkeypatch.setattr(flex, 'iter_conformer_states', unexpected)
    assert list(flex.iter_flex_outcomes(connected, connected.unit('topology:0'), True)) == []
