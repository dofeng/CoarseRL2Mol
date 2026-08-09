from RL_MTCS.visualization import (
    REFERENCE_FONT_SIZE,
    REFERENCE_LABEL_ROTATION_DEGREES,
    REFERENCE_NODE_RADIUS,
    _cart,
    _reference_node_color,
)


def test_rigid_visualization_matches_demo_23823_style():
    assert REFERENCE_NODE_RADIUS == 0.22
    assert REFERENCE_FONT_SIZE == 7
    assert REFERENCE_LABEL_ROTATION_DEGREES == 0.0
    assert _reference_node_color(5) == (0.8, 0.3, 0.6)
    assert _reference_node_color(11) == (0.2, 0.8, 0.4)
    assert _reference_node_color(13) == (0.4, 0.7, 0.9)
    assert _reference_node_color(22) == (0.8, 0.5, 0.1)
    assert _reference_node_color(25) == (0.9, 0.6, 0.2)


def test_plot_projection_keeps_positive_r_vertical():
    origin = _cart((0, 0))
    positive_r = _cart((0, 1))
    assert positive_r[0] == origin[0]
    assert positive_r[1] > origin[1]
