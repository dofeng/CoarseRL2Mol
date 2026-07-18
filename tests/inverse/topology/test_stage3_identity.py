from __future__ import annotations

from model.inverse.inverse_pipeline import InversePipelineV3


def test_stage3_cold_rebuild_has_no_cross_iteration_identity_restore() -> None:
    assert not hasattr(InversePipelineV3, "_restore_stage3_node_identity")
