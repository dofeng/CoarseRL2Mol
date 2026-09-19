"""Shared inverse-to-assembly completion contract."""

MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES = 3
# A bounded terminal remainder may contain any ordinary closure resource.
# Stage3 and RL_MTCS must use this same set; otherwise inverse completion can
# publish a snapshot that the placement stage immediately rejects.
TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES = frozenset({11, 22, 23})
UNASSEMBLED_CLOSURE_RESOURCE_STATE = "remaining_closure_resource"
