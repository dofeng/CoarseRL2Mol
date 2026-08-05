"""Initialization for inverse-driven topology assembly.

This module intentionally has no SU histogram generation, resource allocation,
cluster downgrading, or substitution logic.  The inverse pipeline supplies
complete units; initialization only validates and registers them for hex-grid
placement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .hex_assembly_state import HexAssemblyState
from .inverse_contract import InverseAssemblyInput, InverseInputError, build_inverse_input


def _load_payload(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() != ".json":
            raise InverseInputError(
                "RL_MTCS.initialize now accepts an inverse result or JSON export; "
                "CSV/SU-histogram initialization was removed"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise InverseInputError(f"cannot read inverse JSON export: {path}") from exc
        except json.JSONDecodeError as exc:
            raise InverseInputError(f"invalid inverse JSON export: {path}") from exc
    return value


def _validate_inverse_completion(payload: Any) -> None:
    """Reject incomplete inverse output before MCTS can place anything."""
    if not isinstance(payload, dict):
        return
    for key in ("ok", "topology_assembly_ok"):
        if key in payload and not bool(payload.get(key)):
            raise InverseInputError(
                f"inverse topology audit is not complete ({key}=False); "
                "RL_MTCS will not allocate missing structures"
            )
    pending = payload.get("pending_contracts", payload.get("pending_units", []))
    if not pending:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = dict(payload.get(key, {}) or {})
            pending = nested.get("pending_contracts", nested.get("pending_units", []))
            if pending:
                break
    if pending:
        raise InverseInputError(
            f"inverse topology has {len(list(pending or []))} pending units/contracts; "
            "complete them in inverse before MCTS assembly"
        )
    ownership = dict(payload.get("ownership_ledger", {}) or {})
    if not ownership:
        for key in ("external_assembly", "assembly_view", "assembly"):
            nested = dict(payload.get(key, {}) or {})
            ownership = dict(nested.get("ownership_ledger", {}) or {})
            if ownership:
                break
    states = dict(ownership.get("states", {}) or {})
    bad = list(states.get("unclassified_or_conflict", []) or [])
    if bad:
        raise InverseInputError(
            f"inverse ownership contains unclassified/conflicting nodes: {bad[:12]}"
        )


def _choose_anchor(problem: InverseAssemblyInput) -> Optional[str]:
    aromatic = [unit for unit in problem.units if unit.phase == "aromatic"]
    if not aromatic:
        return None
    return max(aromatic, key=lambda unit: (len(unit.nodes), len(unit.local_rings), unit.unit_id)).unit_id


def initialize(inverse_result: Any) -> Dict[str, Any]:
    """Build the immutable inverse-driven assembly state.

    Parameters
    ----------
    inverse_result:
        The result returned by the inverse topology audit, or a JSON export of
        that result.  It must contain complete aromatic unit geometry and
        complete topology products.  No histogram or node CSV is accepted.
    """
    payload = _load_payload(inverse_result)
    _validate_inverse_completion(payload)
    problem = build_inverse_input(payload)
    state = HexAssemblyState(problem=problem)

    anchor_id = _choose_anchor(problem)
    if anchor_id is None:
        raise InverseInputError("inverse result contains no aromatic unit to use as the hex-grid anchor")
    if not state.place_unit(anchor_id, rotation=0, translation=(0, 0)):
        raise InverseInputError(f"failed to place initial aromatic unit {anchor_id!r} at the hex-grid origin")
    state.stage = "rigid"

    aromatic_units = [unit for unit in problem.units if unit.phase == "aromatic"]
    topology_units = [unit for unit in problem.units if unit.phase != "aromatic"]
    return {
        "problem": problem,
        "units": list(problem.units),
        "aromatic_units": aromatic_units,
        "topology_units": topology_units,
        "contracts": list(problem.contracts),
        "ownership": problem.ownership_map(),
        "node_su_types": problem.node_map(),
        "anchor_unit_id": anchor_id,
        "state": state,
        "diagnostics": {
            "initializer": "inverse_topology_units",
            "allocation": False,
            "conversion": False,
            "substitution": False,
            "aromatic_unit_count": len(aromatic_units),
            "topology_unit_count": len(topology_units),
            "external_contract_count": len(problem.contracts),
        },
    }


def initialize_from_inverse(inverse_result: Any) -> Dict[str, Any]:
    """Explicit alias for callers migrating from the legacy initializer."""
    return initialize(inverse_result)


__all__ = [
    "HexAssemblyState",
    "InverseAssemblyInput",
    "InverseInputError",
    "initialize",
    "initialize_from_inverse",
]
