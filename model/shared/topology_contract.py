"""Shared inverse-to-assembly completion contract."""

MAX_TOLERATED_UNASSEMBLED_CLOSURE_RESOURCES = 3
# A bounded terminal remainder may contain any ordinary closure resource.
# Stage3 and RL_MTCS must use this same set; otherwise inverse completion can
# publish a snapshot that the placement stage immediately rejects.
TOLERATED_UNASSEMBLED_CLOSURE_SU_TYPES = frozenset({11, 22, 23})
UNASSEMBLED_CLOSURE_RESOURCE_STATE = "remaining_closure_resource"

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


def aromatic_external_partner_valid(su, partner_su, partner_degree):
    return (partner_su in AROMATIC_EXTERNAL_SU_TYPES.get(su, ())
            and (su != 11 or partner_su not in {19, 20, 21} or partner_degree in {2, 3}))


def carbonyl_neighbors_valid(su, neighbors):
    """Shared SU2/SU3 rules. Neighbors contain (SU, actual target degree)."""
    if len(neighbors) != 2:
        return False
    def body(value, exclude_methyl=False):
        kind, degree = value
        return (kind in CARBONYL_BODY_SU_TYPES - ({22} if exclude_methyl else set())
                and (kind not in {19, 20, 21} or degree in {2, 3}))
    if su == 2:
        return any(body(neighbors[i]) and
                   (neighbors[1-i][0] == 5 or neighbors[1-i][0] == 19
                    and neighbors[1-i][1] in {1, 2, 3}) for i in (0, 1))
    if su == 3:
        return any(body(neighbors[i]) and body(neighbors[1-i], True) for i in (0, 1))
    raise ValueError(f'unsupported carbonyl SU: {su}')
