"""Pre-Layer1 port-role accounting and SU11/SU12 balancing."""

from typing import Any, Dict, Optional, Tuple

import torch

from ...shared.inverse_common import (
    SPECIAL_DEGREE_PRIORS,
    SU_CONNECTION_DEGREE,
    normalize_special_degree_meta,
)


LAYER1_AROMATIC_RING_TYPES = frozenset(
    {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
)
AROMATIC_EXTERNAL_ROLE_TYPES = (5, 6, 7, 8, 9, 10, 11)
NON_AROMATIC_EXTERNAL_PARTNER_TYPES = (5, 6, 7, 8, 9, 11)
PORT_ROLE_BALANCE_DIRECTIONS = ((12, 11), (11, 12))


def _histogram(H: torch.Tensor) -> torch.Tensor:
    if not isinstance(H, torch.Tensor):
        raise TypeError("Layer1 port-role audit requires a torch.Tensor histogram")
    return torch.clamp(H.detach().cpu(), min=0).long()


def _count(histogram: torch.Tensor, su_type: int) -> int:
    if int(su_type) < 0 or int(su_type) >= int(histogram.numel()):
        return 0
    return int(histogram[int(su_type)].item())


def _degree_ports_by_su(
    histogram: torch.Tensor,
    special_degree_meta: Optional[Dict[Any, Any]],
) -> Dict[int, int]:
    special_counts = {
        int(su_type): _count(histogram, int(su_type))
        for su_type in SPECIAL_DEGREE_PRIORS
    }
    normalized_special = normalize_special_degree_meta(
        special_counts,
        special_degree_meta=special_degree_meta,
    )
    degree_ports: Dict[int, int] = {}
    for su_type, count in enumerate(histogram.tolist()):
        if int(su_type) in normalized_special:
            degree_ports[int(su_type)] = int(sum(
                int(degree) * int(degree_count)
                for degree, degree_count in normalized_special[int(su_type)].items()
            ))
            continue
        degree = SU_CONNECTION_DEGREE.get(int(su_type), 0)
        if isinstance(degree, tuple):
            degree = degree[1]
        degree_ports[int(su_type)] = int(count) * int(degree)
    return degree_ports


def audit_layer1_port_roles(
    H: torch.Tensor,
    special_degree_meta: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Audit the two port parities required before a Layer1 rebuild."""
    histogram = _histogram(H)
    degree_ports = _degree_ports_by_su(histogram, special_degree_meta)
    aromatic_degree_ports = int(sum(
        int(degree_ports.get(int(su_type), 0))
        for su_type in LAYER1_AROMATIC_RING_TYPES
    ))
    total_ports = int(sum(degree_ports.values()))
    non_aromatic_degree_ports = int(total_ports - aromatic_degree_ports)
    aromatic_external_role_ports = int(sum(
        _count(histogram, int(su_type))
        for su_type in AROMATIC_EXTERNAL_ROLE_TYPES
    ))
    non_aromatic_external_partner_ports = int(sum(
        _count(histogram, int(su_type))
        for su_type in NON_AROMATIC_EXTERNAL_PARTNER_TYPES
    ))
    aromatic_internal_ports = int(
        aromatic_degree_ports - aromatic_external_role_ports
    )
    non_aromatic_internal_ports = int(
        non_aromatic_degree_ports - non_aromatic_external_partner_ports
    )
    total_even = bool(total_ports % 2 == 0)
    aromatic_even = bool(
        aromatic_internal_ports >= 0 and aromatic_internal_ports % 2 == 0
    )
    non_aromatic_even = bool(
        non_aromatic_internal_ports >= 0
        and non_aromatic_internal_ports % 2 == 0
    )
    nonnegative = bool(
        aromatic_internal_ports >= 0 and non_aromatic_internal_ports >= 0
    )
    return {
        "ok": bool(total_even and aromatic_even and non_aromatic_even and nonnegative),
        "total_ports": int(total_ports),
        "aromatic_degree_ports": int(aromatic_degree_ports),
        "aromatic_external_role_ports": int(aromatic_external_role_ports),
        "aromatic_internal_ports": int(aromatic_internal_ports),
        "non_aromatic_degree_ports": int(non_aromatic_degree_ports),
        "non_aromatic_external_partner_ports": int(
            non_aromatic_external_partner_ports
        ),
        "non_aromatic_internal_ports": int(non_aromatic_internal_ports),
        "total_even": bool(total_even),
        "aromatic_internal_even": bool(aromatic_even),
        "non_aromatic_internal_even": bool(non_aromatic_even),
        "role_balance_required": bool(
            total_even and not aromatic_even and not non_aromatic_even
        ),
        "role_parity_consistent": bool(aromatic_even == non_aromatic_even),
        "scope": "pre_layer1_port_roles",
    }


def select_layer1_port_balance(
    H: torch.Tensor,
    audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select at most one deterministic SU11/SU12 role-balance conversion."""
    histogram = _histogram(H)
    audit_local = dict(audit or audit_layer1_port_roles(histogram))
    if not bool(audit_local.get("total_even", False)):
        return {
            "required": False,
            "possible": False,
            "reason": "total_port_degree_must_be_balanced_first",
        }
    aromatic_even = bool(audit_local.get("aromatic_internal_even", False))
    non_aromatic_even = bool(
        audit_local.get("non_aromatic_internal_even", False)
    )
    if aromatic_even and non_aromatic_even:
        return {
            "required": False,
            "possible": True,
            "reason": "layer1_port_roles_already_balanced",
        }
    if aromatic_even != non_aromatic_even:
        return {
            "required": True,
            "possible": False,
            "reason": "layer1_port_role_parities_disagree",
        }
    for source, target in PORT_ROLE_BALANCE_DIRECTIONS:
        if _count(histogram, int(source)) > 0:
            return {
                "required": True,
                "possible": True,
                "from_su": int(source),
                "to_su": int(target),
                "count": 1,
                "reason": "pre_layer1_port_role_balance",
            }
    return {
        "required": True,
        "possible": False,
        "reason": "layer1_port_role_balance_donor_unavailable",
    }


def apply_layer1_port_balance(
    H: torch.Tensor,
    conversion: Dict[str, Any],
    special_degree_meta: Optional[Dict[Any, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply one selected SU11/SU12 conversion and verify the result."""
    H_before = _histogram(H)
    before = audit_layer1_port_roles(H_before, special_degree_meta)
    if not bool(conversion.get("required", False)):
        return H_before.to(H.device), {
            "adjusted": False,
            "reason": str(conversion.get("reason", "not_required")),
            "before": before,
            "after": before,
        }
    if not bool(conversion.get("possible", False)):
        raise RuntimeError(str(conversion.get(
            "reason", "layer1_port_role_balance_impossible"
        )))
    source = int(conversion.get("from_su", -1))
    target = int(conversion.get("to_su", -1))
    count = int(conversion.get("count", 0))
    if (source, target) not in set(PORT_ROLE_BALANCE_DIRECTIONS) or count != 1:
        raise ValueError(
            "Layer1 port-role balance only supports one SU11/SU12 conversion"
        )
    if _count(H_before, source) < count:
        raise RuntimeError("layer1_port_role_balance_donor_unavailable")
    H_after = H_before.clone()
    H_after[source] -= count
    H_after[target] += count
    after = audit_layer1_port_roles(H_after, special_degree_meta)
    if not bool(after.get("ok", False)):
        raise RuntimeError(
            "layer1_port_role_balance_failed: "
            f"aromatic={after['aromatic_internal_ports']} "
            f"non_aromatic={after['non_aromatic_internal_ports']}"
        )
    return H_after.to(H.device), {
        "adjusted": True,
        "reason": "layer1_port_roles_balanced_before_rebuild",
        "from_su": int(source),
        "to_su": int(target),
        "count": int(count),
        "before": before,
        "after": after,
    }
