"""Authoritative Stage3 resource-type boundaries."""

from __future__ import annotations


# Ordinary external topology closures may consume these resource types.
GENERIC_CLOSURE_RESOURCE_TYPES = (11, 22, 23)

# SU24 is a physical ring skeleton resource. It is deliberately excluded from
# ordinary generic closure conversion; recovery reclassifies released nodes
# to SU23 before they can enter a conversion plan.
SU24_RING_RESOURCE_TYPE = 24
PHYSICAL_TOPOLOGY_RESOURCE_TYPES = (
    *GENERIC_CLOSURE_RESOURCE_TYPES,
    SU24_RING_RESOURCE_TYPE,
)

# The SU types authorized for ordinary Stage3 histogram adjustment. A SU24
# released from a pure ring is reclassified to SU23 before entering this pool.
EXTERNAL_ADJUSTMENT_RESOURCE_TYPES = (11, 12, 13, 22, 23)

# Internal conversion consumes only ordinary resources. SU24 is never an
# internal conversion donor; it is converted to SU23 at ring recovery time.
INTERNAL_CONVERSION_SOURCE_TYPES = EXTERNAL_ADJUSTMENT_RESOURCE_TYPES

# Accounting also tracks SU10 aromatic construction context and physical SU24,
# neither of which is authorized for ordinary external histogram conversion.
STAGE3_ACCOUNTING_SU_TYPES = (10, 11, 12, 13, 22, 23, 24)
