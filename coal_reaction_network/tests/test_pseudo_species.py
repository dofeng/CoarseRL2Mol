import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pseudo_species import FragmentProperties, ThreeDimensionalClassifier


def test_class_uses_three_descriptors_and_keeps_internal_attributes():
    classifier = ThreeDimensionalClassifier()
    first = classifier.classify(FragmentProperties(41, 1, 0, 2, 512.4))
    second = classifier.classify(FragmentProperties(41, 1, 0, 0, 498.1))

    assert first.node_id == second.node_id == "C40+_R1_O0"
    assert first.descriptor == "C=C40+;R=R1;O=O0"
    assert first.fragment.radical_sites == 2
    assert second.fragment.molecular_mass == 498.1


def test_distinct_ring_class_gets_distinct_node():
    classifier = ThreeDimensionalClassifier()
    one_ring = classifier.classify(FragmentProperties(41, 1, 0))
    many_rings = classifier.classify(FragmentProperties(41, 11, 0))

    assert one_ring.node_id != many_rings.node_id
    assert many_rings.ring_bin == "R10+"


def test_node_ids_are_stable_across_discovery_order():
    first = ThreeDimensionalClassifier()
    second = ThreeDimensionalClassifier()
    expected = first.classify(FragmentProperties(41, 1, 0)).node_id
    second.classify(FragmentProperties(1, 0, 0))
    actual = second.classify(FragmentProperties(41, 1, 0)).node_id
    assert expected == actual == "C40+_R1_O0"
