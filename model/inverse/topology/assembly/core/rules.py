"""Static class groups and closure operators for topology assembly."""

from __future__ import annotations

from .models import ClosureOperatorSpec


EXPLICIT_FLEX_CLASSES = {
    "A", "A+", "24_AB_M", "24_CD_M_FULL",
}
EXPLICIT_SIDE_CLASSES = {"C", "C+", "F", "24_AB_P"}
B_LIKE_OPEN_CLASSES = {
    "B", "B+", "24_A", "24_B", "24_AB_N", "24_CD_M",
    "24_AB_N_24_CD", "25_AB",
}
D_LIKE_OPEN_CLASSES = {
    "D", "D+", "24_C", "24_D", "24_CD_N", "24_CD_N_24_CD",
    "24_AB_N_24_AB", "25_CD",
}
E_LIKE_OPEN_CLASSES = {"E", "E+", "G", "24_CD_P"}
RING_ONLY_LIKE24_CLASSES = set()
CHAIN_MATERIALIZABLE_LIKE24_CLASSES = {"24_AB_M", "24_AB_P", "24_CD_M_FULL"}
CHAIN_MATERIALIZABLE_OPEN_CLASSES = {
    "B", "B+", "C", "C+", "D", "D+", "E", "E+", "F", "G",
    "24_A", "24_B", "24_C", "24_D", "24_AB_N", "24_CD_M",
    "24_CD_N", "24_CD_P", "24_AB_N_24_AB", "24_AB_N_24_CD", "24_CD_N_24_CD",
}

TERMINAL_ENDPOINT_LABELS = {
    "1", "4", "16", "18", "22", "28", "32",
    "19d1", "20d1", "21d2",
}
RING_PRODUCT_KIND_TO_TYPE = {
    "fatty_ring_fused": "aliphatic_ring",
    "fatty_ring_side": "aliphatic_ring",
    "fatty_ring_upper_lower": "aliphatic_ring",
}
CANONICAL_LABEL_BY_CLASS = {
    "A": ["11", "23", "11"],
    "B": ["11", "23"],
    "C": ["11", "23", "22"],
    "D": ["23"],
    "E": ["23", "22"],
    "F": ["11", "22"],
    "G": ["23", "22"],
    "A+": ["11", "23", "23", "23", "11"],
    "B+": ["11", "23", "23", "23"],
    "C+": ["11", "23", "23", "22"],
    "D+": ["23", "23", "23"],
    "E+": ["23", "23", "22"],
    "25_AB": ["11", "25"],
    "25_CD": ["25"],
}

BRANCH_CLOSURE_OPERATOR_SPECS = {
    "H": ClosureOperatorSpec(name="H", scope="branch"),
    "H1": ClosureOperatorSpec(
        name="H1", scope="branch", fixed_side_compatible_only=True,
    ),
    "FIXED22": ClosureOperatorSpec(
        name="FIXED22", scope="branch", fixed_side_compatible_only=True,
    ),
}
MAINLINE_CLOSURE_OPERATOR_SPECS = {
    "F1": ClosureOperatorSpec(name="F1", scope="mainline"),
    "F2": ClosureOperatorSpec(name="F2", scope="mainline"),
}


def external_phase_for_role(target_role: str) -> str:
    return {
        "flex_chain": "external_flex_closure",
        "fatty_ring": "external_fatty_ring_closure",
        "side_chain": "external_side_chain_closure",
    }.get(str(target_role or ""), "external_side_chain_closure")


def main_kind_for_class(class_type: str) -> str:
    if class_type in EXPLICIT_FLEX_CLASSES:
        return "explicit_flex"
    if class_type in EXPLICIT_SIDE_CLASSES:
        return "explicit_side"
    if class_type in B_LIKE_OPEN_CLASSES:
        return "b_like_open"
    if class_type in D_LIKE_OPEN_CLASSES:
        return "d_like_open"
    if class_type in E_LIKE_OPEN_CLASSES:
        return "e_like_open"
    if class_type in RING_ONLY_LIKE24_CLASSES:
        return "ring_only_like24"
    return "other"
