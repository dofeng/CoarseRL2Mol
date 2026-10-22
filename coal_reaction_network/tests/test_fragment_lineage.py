import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fragment_lineage import aggregate_transitions, build_transition_events, match_fragments
from reaxff_fragments import FragmentRecord


def _fragment(fragment_id, atom_ids, node_id="C1-5_R0_O0"):
    return FragmentRecord(
        fragment_id=fragment_id,
        atom_ids=atom_ids,
        atom_types={},
        element_counts={"C": len(atom_ids)},
        molecular_mass=12.011 * len(atom_ids),
        ring_count=0,
        ring_sizes=[],
        radical_sites=0,
        radical_atom_ids=[],
        valence_deficit=0.0,
        edge_count=max(0, len(atom_ids) - 1),
        node_id=node_id,
        carbon_bin="C1-5",
        ring_bin="R0",
        oxygen_bin="O0",
        atom_details=[],
        topology_hash="synthetic-unchanged-topology",
    )


def test_fragmentation_is_one_conserved_bipartite_event():
    parents = [_fragment(1, [1, 2, 3, 4])]
    children = [_fragment(1, [1, 2]), _fragment(2, [3, 4])]
    events = build_transition_events(parents, children, match_fragments(parents, children))
    assert len(events) == 1
    assert events[0].event_type == "fragmentation"
    assert events[0].resolution == "resolved"
    assert events[0].atom_conservation_ratio == 1.0


def test_fusion_is_one_conserved_bipartite_event():
    parents = [_fragment(1, [1, 2]), _fragment(2, [3, 4])]
    children = [_fragment(1, [1, 2, 3, 4])]
    events = build_transition_events(parents, children, match_fragments(parents, children))
    assert len(events) == 1
    assert events[0].event_type == "fusion"
    assert events[0].atom_conservation_ratio == 1.0


def test_unmatched_fragments_become_birth_and_death_events():
    parents = [_fragment(1, [1])]
    children = [_fragment(1, [2])]
    events = build_transition_events(parents, children, match_fragments(parents, children))
    assert {event.event_type for event in events} == {"birth", "death"}
    assert all(event.resolution == "unresolved" for event in events)


def test_unchanged_fragment_is_persistence_not_reaction():
    parents = [_fragment(1, [1, 2])]
    children = [_fragment(1, [1, 2])]
    events = build_transition_events(parents, children, match_fragments(parents, children))
    assert len(events) == 1
    assert events[0].event_type == "persistence"
    assert events[0].confidence == 1.0


def test_aggregator_retains_event_type_breakdown():
    parents=[_fragment(1,[1,2,3,4])]
    children=[_fragment(1,[1,2]),_fragment(2,[3,4])]
    event=build_transition_events(parents,children,match_fragments(parents,children))[0]
    channel=aggregate_transitions([event,replace(event,event_id=2,event_type="many_to_many")])[0]
    assert channel["event_type"] == "mixed"
    assert channel["event_count"] == 2
    assert channel["event_type_counts"] == {"fragmentation":1,"many_to_many":1}
