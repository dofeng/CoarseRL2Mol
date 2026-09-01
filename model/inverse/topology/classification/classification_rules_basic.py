from __future__ import annotations

from typing import Any, Dict, List, Set


ALIPHATIC_OPEN_NEIGHBORS: Set[int] = {14, 15, 17, 19, 20, 21, 23, 24, 25}
AROMATIC_NEIGHBORS: Set[int] = {5, 6, 7, 8, 9, 11}
TERMINAL_NEIGHBORS: Set[int] = {1, 4, 16, 18, 22, 28, 32}
SPECIAL_CONNECTOR_NEIGHBORS: Set[int] = {0, 2, 27, 29, 31}

GENERIC_OPEN_BASE_SU: Set[int] = {0, 2, 3, 14, 15, 17, 23, 24, 25}
GENERIC_OPEN_SUBTYPES: Set[str] = {
    "single19d2",
    "single20d2",
    "double19d2",
    "double20d2",
    "single19d3",
    "single20d3",
    "double19d3",
    "double20d3",
    "21d3",
}
TERMINAL_SUBTYPES: Set[str] = {"19d1", "20d1", "21d2"}

BASIC_A_AROMATIC_BY_CENTER: Dict[int, List[int]] = {
    0: [6, 9],
    2: [5, 9],
    3: [9, 9],
    27: [6, 6],
    29: [5, 5],
    31: [7, 7],
}

BASIC_B_SPECIAL_OPEN_BY_CENTER: Dict[int, Set[str]] = {
    27: {"single20d2"},
    29: {"single19d2"},
    31: {"single19d2"},
}

BASIC_C_SPECIAL_TERMINAL_BY_CENTER: Dict[int, Set[str]] = {
    27: {"20d1"},
    29: {"19d1"},
    31: {"19d1"},
}

BASIC_D_SPECIAL_OPEN_BY_CENTER: Dict[int, Set[str]] = {
    27: {"single20d2"},
    29: {"single19d2"},
    31: {"single19d2"},
}

BASIC_E_SPECIAL_SIDE_BY_CENTER: Dict[int, Set[str]] = {
    27: {"20d1"},
    29: {"19d1"},
    31: {"19d1"},
}

BASIC_F_AROMATIC_BY_CENTER: Dict[int, Set[int]] = {
    1: {9},
    28: {5},
    32: {8},
}

BASIC_G_SIDE_BY_CENTER: Dict[int, Dict[str, Any]] = {
    1: {"allow_generic_open": True},
    4: {"exact_su_types": {23, 24, 25}},
    16: {"exact_su_types": {15}},
    18: {"exact_su_types": {17}},
    28: {"exact_subtypes": {"single19d2"}},
    32: {"exact_subtypes": {"21d2"}},
}

BASIC_CHAIN_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "A",
        "family": "basic",
        "class_type": "A",
        "topology_template": "11-23-11",
        "center_su": {0, 2, 3, 27, 29, 31},
        "closed_main": True,
    },
    {
        "rule_id": "B",
        "family": "basic",
        "class_type": "B",
        "topology_template": "11-23-23-",
        "center_su": {0, 2, 3, 27, 29, 31},
        "closed_main": False,
    },
    {
        "rule_id": "C",
        "family": "basic",
        "class_type": "C",
        "topology_template": "11-23-22",
        "center_su": {0, 2, 3, 27, 29, 31},
        "closed_main": True,
    },
    {
        "rule_id": "D",
        "family": "basic",
        "class_type": "D",
        "topology_template": "-23-23-23-",
        "center_su": {0, 2, 3, 15, 27, 29, 31},
        "closed_main": False,
    },
    {
        "rule_id": "E",
        "family": "basic",
        "class_type": "E",
        "topology_template": "22-23-23-",
        "center_su": {0, 2, 3, 15, 27, 29, 31},
        "closed_main": False,
    },
    {
        "rule_id": "F",
        "family": "basic",
        "class_type": "F",
        "topology_template": "11-22",
        "center_su": {1, 28, 32},
        "closed_main": True,
    },
    {
        "rule_id": "G",
        "family": "basic",
        "class_type": "G",
        "topology_template": "-23-22",
        "center_su": {1, 4, 16, 18, 28, 32},
        "closed_main": False,
    },
]

DOUBLE_D2_ALLOWED_CONNECTORS: Dict[str, Set[int]] = {
    "double19d2": {2, 28, 29, 31},
    "double20d2": {0, 27},
}

DOUBLE_D2_SIDE_RULES: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {
    "double19d2": {
        2: {
            "M": {"exact_su_types": {9}},
            "N": {"allow_generic_open": True},
            "P": {"exact_su_types": {22}},
        },
        28: {
            # Open neighbours belong to the Like24 ring context; only an
            # actually isolated SU28 is the closed P-side terminus.
            "N": {"allow_generic_open": True},
            "P": {"exact_neighbor_count": 0},
        },
        29: {
            "M": {"exact_su_types": {5}},
            "N": {"exact_subtypes": {"single19d2"}},
            "P": {"exact_subtypes": {"19d1"}},
        },
        31: {
            "M": {"exact_su_types": {7}},
            "N": {"exact_subtypes": {"single19d2"}},
            "P": {"exact_subtypes": {"19d1"}},
        },
    },
    "double20d2": {
        0: {
            "M": {"exact_su_types": {9}},
            "N": {"allow_generic_open": True},
            "P": {"exact_su_types": {22}},
        },
        27: {
            "M": {"exact_su_types": {6}},
            "N": {"exact_subtypes": {"single20d2"}},
            "P": {"exact_subtypes": {"20d1"}},
        },
    },
}

LIKE24_FIXED_ELIGIBLE_SUBTYPES: Set[str] = {
    "single19d3",
    "double19d3",
    "single20d3",
    "double20d3",
}

LIKE24_ALLOWED_CONNECTORS: Dict[int, Set[int]] = {
    19: {2, 28, 29, 31},
    20: {0, 27},
}

LIKE24_FIXED_SIDE_RULES: Dict[int, Dict[int, Dict[str, Dict[str, Any]]]] = {
    19: {
        2: {
            "M": {"exact_su_types": {9}},
            "N": {"allow_generic_open": True},
            "P": {"exact_su_types": {22}},
        },
        28: {
            # SU28 is a branch terminus when isolated, but an open 23/24
            # boundary on the Like24 centre makes the fixed edge an N-side
            # ring context.  The classifier supplies that centre context.
            "N": {"allow_generic_open": True},
            "P": {"exact_neighbor_count": 0},
        },
        29: {
            "M": {"exact_su_types": {5}},
            "N": {"exact_subtypes": {"single19d2", "single19d3", "double19d3"}},
            "P": {"exact_subtypes": {"19d1"}},
        },
        31: {
            "M": {"exact_su_types": {7}},
            "N": {"exact_subtypes": {"single19d2", "single19d3", "double19d3"}},
            "P": {"exact_subtypes": {"19d1"}},
        },
    },
    20: {
        0: {
            "M": {"exact_su_types": {9}},
            "N": {"allow_generic_open": True},
            "P": {"exact_su_types": {22}},
        },
        27: {
            "M": {"exact_su_types": {6}},
            "N": {"exact_subtypes": {"single20d2", "single20d3", "double20d3"}},
            "P": {"exact_subtypes": {"20d1"}},
        },
    },
}

DOUBLE_D2_COMBINATION_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "A+",
        "family": "double_d2_combo",
        "class_type": "A+",
        "topology_template": "11-23-23-23-11",
        "components": ("M", "M"),
        "closed_main": True,
    },
    {
        "rule_id": "B+",
        "family": "double_d2_combo",
        "class_type": "B+",
        "topology_template": "11-23-23-23-23-",
        "components": ("M", "N"),
        "closed_main": False,
    },
    {
        "rule_id": "C+",
        "family": "double_d2_combo",
        "class_type": "C+",
        "topology_template": "11-23-23-22",
        "components": ("M", "P"),
        "closed_main": True,
    },
    {
        "rule_id": "D+",
        "family": "double_d2_combo",
        "class_type": "D+",
        "topology_template": "-23-23-23-23-",
        "components": ("N", "N"),
        "closed_main": False,
    },
    {
        "rule_id": "E+",
        "family": "double_d2_combo",
        "class_type": "E+",
        "topology_template": "-23-23-23-22",
        "components": ("N", "P"),
        "closed_main": False,
    },
]

SU25_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "25_AB",
        "family": "su25",
        "class_type": "25_AB",
        "topology_template": "11-25(-22)(-H)-",
        "requires_11_neighbor": True,
        "requires_branch_22_like": True,
    },
    {
        "rule_id": "25_CD",
        "family": "su25",
        "class_type": "25_CD",
        "topology_template": "-25(-22)(-H)-",
        "requires_11_neighbor": False,
        "requires_branch_22_like": True,
    },
]
