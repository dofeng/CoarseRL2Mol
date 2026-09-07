"""Stable public entry points for external Stage3 topology assembly."""

from .workflow.runtime import (
    ExternalTopologyAssembler,
    assemble_external_topologies,
)
from .projection import project_fragment_views
from .core.fragment import AssemblyFragmentView

__all__ = [
    "AssemblyFragmentView",
    "ExternalTopologyAssembler",
    "assemble_external_topologies",
    "project_fragment_views",
]
