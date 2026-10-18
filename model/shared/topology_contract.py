"""Shared inverse-to-assembly completion contract."""

MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES = 3
# A bounded terminal remainder may contain any ordinary closure resource.
# Stage3 and RL_MTCS must use this same set; otherwise inverse completion can
# publish a snapshot that the placement stage immediately rejects.
TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES = frozenset({11, 22, 23})

# Chemical identities used by both inverse exports and runtime assembly.
AROMATIC_EXTERNAL_SU_TYPES = {
    5: (2, 28, 29), 6: (0, 27), 7: (31,), 8: (32,),
    9: (0, 1, 2, 3), 10: (10,),
    11: (23, 24, 25, 22, 19, 20, 21, 15, 17),
}
CARBONYL_BODY_SU_TYPES = frozenset({9, 23, 24, 25, 22, 14, 15, 17, 19, 20, 21})
AROMATIC_RING_SU_TYPES = frozenset({5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30})
AROMATIC_RING_NEIGHBOR_SU_TYPES = {
    su: AROMATIC_RING_SU_TYPES - ({9} if su == 9 else {10} if su == 10
                                 else {26, 30} if su in {26, 30} else set())
    for su in AROMATIC_RING_SU_TYPES
}
