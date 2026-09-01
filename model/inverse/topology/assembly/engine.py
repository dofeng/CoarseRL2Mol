"""Compatibility exports for the external topology assembly API.

Implementation lives in focused runtime, coordination, and materialization
modules. New internal code should import those modules directly.
"""

from .workflow.runtime import (
    ExternalTopologyAssembler,
    assemble_external_topologies,
)

__all__ = [
    "ExternalTopologyAssembler",
    "assemble_external_topologies",
]
