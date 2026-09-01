"""External topology assembly.

Internal modules are grouped by responsibility: ``core`` for shared value
objects and rules, ``projection`` for fragment views, ``state`` for ownership
and pending contracts, ``closure`` for chain materialization, ``ring`` for
aliphatic rings, and ``workflow`` for phase coordination. Public callers
should use :mod:`model.inverse.topology.assembly.external`.
"""

from .external import (
    AssemblyFragmentView,
    ExternalTopologyAssembler,
    assemble_external_topologies,
    project_fragment_views,
)

__all__ = [
    "AssemblyFragmentView",
    "ExternalTopologyAssembler",
    "assemble_external_topologies",
    "project_fragment_views",
]
