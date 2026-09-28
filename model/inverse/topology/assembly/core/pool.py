"""Transaction-local allocator for real Stage3 resource node IDs."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from ...stage3_models import TopologyPoolLedger


def _su_key(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


class PoolAllocator:
    def __init__(self, initial_ids_by_su: Dict[str, Iterable[int]]) -> None:
        normalized = {
            _su_key(key): [int(value) for value in list(values or [])]
            for key, values in dict(initial_ids_by_su or {}).items()
        }
        self.initial_ids_by_su = {
            str(key): list(dict.fromkeys(int(value) for value in list(values or [])))
            for key, values in normalized.items()
        }
        self.remaining_ids_by_su = deepcopy(self.initial_ids_by_su)
        self.consumed_ids_by_su = {
            str(key): [] for key in self.initial_ids_by_su
        }
        self.su_key_by_node_id: Dict[int, str] = {}
        for su_key, node_ids in self.initial_ids_by_su.items():
            for node_id in node_ids:
                existing_key = self.su_key_by_node_id.get(int(node_id))
                if existing_key is not None and str(existing_key) != str(su_key):
                    raise ValueError(
                        "pool resource id assigned to multiple SU types: "
                        f"node={int(node_id)} su={existing_key},{su_key}"
                    )
                self.su_key_by_node_id[int(node_id)] = str(su_key)
        self.protected_counts_by_su: Dict[str, int] = {}
        self._generic_claim_excluded_node_ids: Set[int] = set()
        self._claim_trace: List[Dict[str, Any]] = []
        self._claim_trace_context: Dict[str, str] = {}

    def trace_checkpoint(self) -> int:
        return len(self._claim_trace)

    @contextmanager
    def claim_trace_scope(
        self,
        stage: str,
        slot_id: str = "",
    ) -> Iterator[None]:
        previous = dict(self._claim_trace_context)
        self._claim_trace_context = {
            "stage": str(stage or "resource_claim"),
            "slot_id": str(slot_id or ""),
        }
        try:
            yield
        finally:
            self._claim_trace_context = previous

    def claim_trace_since(self, checkpoint: int) -> List[Dict[str, Any]]:
        start = max(0, int(checkpoint))
        return [deepcopy(item) for item in self._claim_trace[start:]]

    def claimable_ids_by_su(self) -> Dict[str, List[int]]:
        """Return the IDs reachable through generic ``claim`` calls.

        ``available_counts`` intentionally remains the raw allocator inventory.
        Pending-contract planning uses this request-aware projection so protected
        and candidate-specific resources are not reported as freely claimable.
        """
        result: Dict[str, List[int]] = {}
        for raw_key, raw_values in self.remaining_ids_by_su.items():
            key = str(raw_key)
            generic = [
                int(value)
                for value in list(raw_values or [])
                if int(value) not in self._generic_claim_excluded_node_ids
            ]
            protected = max(
                0, int(self.protected_counts_by_su.get(key, 0) or 0)
            )
            keep = max(0, len(generic) - protected)
            result[key] = generic[:keep]
        return result

    def claimable_counts(self) -> Dict[str, int]:
        return {
            str(key): len(list(values or []))
            for key, values in self.claimable_ids_by_su().items()
        }

    def claimability_snapshot(self) -> Dict[str, Any]:
        raw_ids = {
            str(key): [int(value) for value in list(values or [])]
            for key, values in self.remaining_ids_by_su.items()
        }
        claimable_ids = self.claimable_ids_by_su()
        return {
            "raw_available_ids_by_su": raw_ids,
            "raw_available_counts": {
                str(key): len(values) for key, values in raw_ids.items()
            },
            "claimable_ids_by_su": claimable_ids,
            "claimable_available_counts": {
                str(key): len(values) for key, values in claimable_ids.items()
            },
            "protected_counts": {
                str(key): max(0, int(value))
                for key, value in self.protected_counts_by_su.items()
                if int(value) > 0
            },
            "excluded_node_ids": sorted(
                int(value) for value in self._generic_claim_excluded_node_ids
            ),
            "excluded_node_ids_by_su": {
                str(key): sorted(
                    int(value)
                    for value in list(raw_ids.get(str(key), []) or [])
                    if int(value) in self._generic_claim_excluded_node_ids
                )
                for key in raw_ids
                if any(
                    int(value) in self._generic_claim_excluded_node_ids
                    for value in list(raw_ids.get(str(key), []) or [])
                )
            },
        }

    def claimability_for_request(
        self,
        su_type: int,
        *,
        count: int = 1,
        specific_node_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        """Describe exactly what generic and specific claim calls can take."""
        su_key = _su_key(su_type)
        want = max(0, int(count))
        raw_ids = [
            int(value)
            for value in list(self.remaining_ids_by_su.get(su_key, []) or [])
        ]
        generic_ids = [
            int(value)
            for value in self.claimable_ids_by_su().get(su_key, [])
        ]
        protected = max(
            0, int(self.protected_counts_by_su.get(su_key, 0) or 0)
        )
        specific = sorted({
            int(value) for value in list(specific_node_ids or [])
        })
        specific_result = {
            str(node_id): bool(
                int(node_id) in set(raw_ids)
                and len(raw_ids) - int(protected) >= 1
            )
            for node_id in specific
        }
        return {
            "su_type": int(su_type),
            "requested_count": int(want),
            "raw_available_ids": raw_ids,
            "generic_claimable_ids": generic_ids,
            "protected_count": int(protected),
            "excluded_node_ids": sorted(
                int(value)
                for value in self._generic_claim_excluded_node_ids
                if int(value) in set(raw_ids)
            ),
            "generic_claim_satisfied": len(generic_ids) >= int(want),
            "specific_claim_satisfied_by_node_id": specific_result,
        }

    def _record_claim(
        self,
        *,
        su_key: str,
        mode: str,
        requested_count: int,
        selected_ids: Iterable[int],
        specific_node_id: Optional[int] = None,
    ) -> None:
        snapshot = self.claimability_snapshot()
        self._claim_trace.append({
            **dict(self._claim_trace_context),
            "mode": str(mode),
            "su_type": int(su_key) if str(su_key).lstrip("-").isdigit() else str(su_key),
            "requested_count": max(0, int(requested_count)),
            "specific_node_id": (
                None if specific_node_id is None else int(specific_node_id)
            ),
            "selected_node_ids": [int(value) for value in list(selected_ids or [])],
            "success": len(list(selected_ids or [])) >= max(0, int(requested_count)),
            "raw_available_count": int(
                dict(snapshot["raw_available_counts"]).get(str(su_key), 0)
            ),
            "claimable_available_count": int(
                dict(snapshot["claimable_available_counts"]).get(str(su_key), 0)
            ),
            "protected_count": int(
                self.protected_counts_by_su.get(str(su_key), 0) or 0
            ),
            "excluded_node_ids": list(
                dict(snapshot.get("excluded_node_ids_by_su", {}) or {}).get(
                    str(su_key), []
                )
            ),
        })

    def su_key_for_node_id(self, node_id: int) -> Optional[str]:
        return self.su_key_by_node_id.get(int(node_id))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "initial": deepcopy(self.initial_ids_by_su),
            "remaining": deepcopy(self.remaining_ids_by_su),
            "consumed": deepcopy(self.consumed_ids_by_su),
            "su_key_by_node_id": deepcopy(self.su_key_by_node_id),
            "protected_counts": deepcopy(self.protected_counts_by_su),
            "generic_claim_excluded": sorted(
                int(value) for value in self._generic_claim_excluded_node_ids
            ),
        }

    def rollback(self, snapshot: Dict[str, Any]) -> None:
        snapshot = dict(snapshot or {})
        # Transaction snapshots can come from JSON/legacy callers where SU
        # bucket keys were serialized as integers.  The allocator contract is
        # string-keyed everywhere else; restoring the raw mapping would make
        # subsequent ``get(str(su))`` calls silently miss valid resources.
        def normalize_ids(raw: Any) -> Dict[str, List[int]]:
            normalized: Dict[str, List[int]] = {}
            for raw_key, raw_values in dict(raw or {}).items():
                key = _su_key(raw_key)
                normalized.setdefault(key, []).extend(
                    int(value) for value in list(raw_values or [])
                )
            return {
                key: sorted(set(values))
                for key, values in normalized.items()
            }

        before_remaining = {
            str(key): {int(value) for value in list(values or [])}
            for key, values in self.remaining_ids_by_su.items()
        }
        restored_remaining = {
            key: set(values)
            for key, values in normalize_ids(snapshot.get("remaining", {})).items()
        }
        rolled_back_ids_by_su = {
            str(key): sorted(
                restored_remaining.get(str(key), set())
                - before_remaining.get(str(key), set())
            )
            for key in set(before_remaining) | set(restored_remaining)
            if restored_remaining.get(str(key), set())
            - before_remaining.get(str(key), set())
        }
        self.initial_ids_by_su = normalize_ids(
            snapshot.get("initial", self.initial_ids_by_su)
        )
        self.remaining_ids_by_su = normalize_ids(
            snapshot.get("remaining", self.remaining_ids_by_su)
        )
        self.consumed_ids_by_su = normalize_ids(
            snapshot.get("consumed", self.consumed_ids_by_su)
        )
        self.su_key_by_node_id = {
            int(node_id): _su_key(su_key)
            for node_id, su_key in dict(
                snapshot.get("su_key_by_node_id", self.su_key_by_node_id)
                or {}
            ).items()
        }
        self.protected_counts_by_su = {
            _su_key(key): max(0, int(value))
            for key, value in dict(
                snapshot.get("protected_counts", self.protected_counts_by_su)
                or {}
            ).items()
        }
        self._generic_claim_excluded_node_ids = {
            int(value)
            for value in list(
                snapshot.get("generic_claim_excluded", []) or []
            )
        }
        if rolled_back_ids_by_su:
            self._claim_trace.append({
                **dict(self._claim_trace_context),
                "mode": "rollback",
                "success": True,
                "rolled_back_ids_by_su": rolled_back_ids_by_su,
            })

    def reclassify_ids(
        self,
        node_ids: Iterable[int],
        *,
        from_su: int,
        to_su: int,
    ) -> List[int]:
        """Move available concrete resources to another SU bucket atomically.

        This is used for the one permitted post-recycle reclassification:
        SU24 nodes released from a pure resource ring become ordinary SU23
        resources.  The node must already be available in ``from_su``; a
        consumed or unknown ID is rejected rather than silently duplicated.
        """
        source_key = _su_key(from_su)
        target_key = _su_key(to_su)
        selected = sorted({int(node_id) for node_id in list(node_ids or [])})
        if not selected:
            return []
        if source_key == target_key:
            return selected
        source_available = set(
            int(value)
            for value in list(self.remaining_ids_by_su.get(source_key, []) or [])
        )
        if not set(selected).issubset(source_available):
            return []
        for node_id in selected:
            current_key = self.su_key_by_node_id.get(int(node_id))
            if current_key is not None and str(current_key) != source_key:
                return []

        self.initial_ids_by_su.setdefault(source_key, [])
        self.initial_ids_by_su.setdefault(target_key, [])
        self.remaining_ids_by_su.setdefault(source_key, [])
        self.remaining_ids_by_su.setdefault(target_key, [])
        self.consumed_ids_by_su.setdefault(source_key, [])
        self.consumed_ids_by_su.setdefault(target_key, [])
        selected_set = set(selected)
        self.initial_ids_by_su[source_key] = [
            int(value)
            for value in self.initial_ids_by_su[source_key]
            if int(value) not in selected_set
        ]
        self.initial_ids_by_su[target_key] = sorted(
            set(self.initial_ids_by_su[target_key]) | selected_set
        )
        self.remaining_ids_by_su[source_key] = [
            int(value)
            for value in self.remaining_ids_by_su[source_key]
            if int(value) not in selected_set
        ]
        self.remaining_ids_by_su[target_key] = sorted(
            set(self.remaining_ids_by_su[target_key]) | selected_set
        )
        self.consumed_ids_by_su[source_key] = [
            int(value)
            for value in self.consumed_ids_by_su[source_key]
            if int(value) not in selected_set
        ]
        self.consumed_ids_by_su[target_key] = [
            int(value)
            for value in self.consumed_ids_by_su[target_key]
            if int(value) not in selected_set
        ]
        for node_id in selected:
            self.su_key_by_node_id[int(node_id)] = target_key
        return selected

    def claimed_ids_since(self, snapshot: Dict[str, Any]) -> Dict[str, List[int]]:
        before = {
            str(key): set(int(value) for value in list(values or []))
            for key, values in dict(snapshot or {}).get("remaining", {}).items()
        }
        return {
            str(key): sorted(
                before.get(str(key), set())
                - set(int(value) for value in list(values or []))
            )
            for key, values in self.remaining_ids_by_su.items()
            if before.get(str(key), set())
            - set(int(value) for value in list(values or []))
        }

    @contextmanager
    def candidate_specific_claim_scope(
        self,
        node_ids: Iterable[int],
    ) -> Iterator[None]:
        """Keep candidate-bound IDs available only to ``claim_specific``."""
        previous = set(self._generic_claim_excluded_node_ids)
        self._generic_claim_excluded_node_ids.update(
            int(node_id) for node_id in list(node_ids or [])
        )
        try:
            yield
        finally:
            self._generic_claim_excluded_node_ids = previous

    def claim(self, su_type: int, count: int = 1) -> Optional[List[int]]:
        su_key = _su_key(su_type)
        want = max(0, int(count))
        available = list(self.remaining_ids_by_su.get(su_key, []) or [])
        claimable = [
            int(value)
            for value in available
            if int(value) not in self._generic_claim_excluded_node_ids
        ]
        if want <= 0:
            self._record_claim(
                su_key=su_key,
                mode="generic",
                requested_count=want,
                selected_ids=[],
            )
            return []
        protected = max(0, int(self.protected_counts_by_su.get(su_key, 0) or 0))
        if len(claimable) - protected < want:
            self._record_claim(
                su_key=su_key,
                mode="generic",
                requested_count=want,
                selected_ids=[],
            )
            return None
        claimed = [int(value) for value in claimable[:want]]
        self._record_claim(
            su_key=su_key,
            mode="generic",
            requested_count=want,
            selected_ids=claimed,
        )
        claimed_set = set(claimed)
        available = [value for value in available if int(value) not in claimed_set]
        self.remaining_ids_by_su[su_key] = available
        self.consumed_ids_by_su.setdefault(su_key, []).extend(claimed)
        return claimed

    def claim_specific(
        self,
        su_type: int,
        node_id: int,
    ) -> Optional[List[int]]:
        """Claim one concrete resource ID inside the current transaction."""
        su_key = _su_key(su_type)
        selected = int(node_id)
        available = list(self.remaining_ids_by_su.get(su_key, []) or [])
        if selected not in available:
            self._record_claim(
                su_key=su_key,
                mode="specific",
                requested_count=1,
                selected_ids=[],
                specific_node_id=selected,
            )
            return None
        protected = max(0, int(self.protected_counts_by_su.get(su_key, 0) or 0))
        if len(available) - protected < 1:
            self._record_claim(
                su_key=su_key,
                mode="specific",
                requested_count=1,
                selected_ids=[],
                specific_node_id=selected,
            )
            return None
        self._record_claim(
            su_key=su_key,
            mode="specific",
            requested_count=1,
            selected_ids=[selected],
            specific_node_id=selected,
        )
        available.remove(selected)
        self.remaining_ids_by_su[su_key] = available
        self.consumed_ids_by_su.setdefault(su_key, []).append(selected)
        return [selected]

    def consume_existing_ids(self, ids_by_su: Dict[Any, Iterable[int]]) -> None:
        for raw_su, raw_ids in dict(ids_by_su or {}).items():
            su_key = _su_key(raw_su)
            selected = {int(value) for value in list(raw_ids or [])}
            if not selected:
                continue
            available = list(self.remaining_ids_by_su.get(su_key, []) or [])
            consumed_now = [value for value in available if int(value) in selected]
            self.remaining_ids_by_su[su_key] = [
                int(value) for value in available if int(value) not in selected
            ]
            self.consumed_ids_by_su.setdefault(su_key, []).extend(consumed_now)
            self.consumed_ids_by_su[su_key] = sorted(
                set(self.consumed_ids_by_su[su_key])
            )

    def release_ids(self, ids_by_su: Dict[Any, Iterable[int]]) -> None:
        """Return concrete nodes to the allocator after a product is dismantled.

        A classifier-fixed resource node may not have been present in the
        allocator at construction time.  Releasing it therefore registers the
        node in ``initial_ids_by_su`` as well as ``remaining_ids_by_su``.  The
        operation is idempotent and preserves the allocator's SU identity map.
        """
        for raw_su, raw_ids in dict(ids_by_su or {}).items():
            su_key = _su_key(raw_su)
            selected = sorted({int(value) for value in list(raw_ids or [])})
            if not selected:
                continue
            self.initial_ids_by_su.setdefault(su_key, [])
            self.remaining_ids_by_su.setdefault(su_key, [])
            self.consumed_ids_by_su.setdefault(su_key, [])
            for node_id in selected:
                existing_key = self.su_key_by_node_id.get(int(node_id))
                if existing_key is not None and str(existing_key) != str(su_key):
                    raise ValueError(
                        "released resource id assigned to multiple SU types: "
                        f"node={int(node_id)} su={existing_key},{su_key}"
                    )
                self.su_key_by_node_id[int(node_id)] = str(su_key)
                if int(node_id) not in self.initial_ids_by_su[su_key]:
                    self.initial_ids_by_su[su_key].append(int(node_id))
                if int(node_id) not in self.remaining_ids_by_su[su_key]:
                    self.remaining_ids_by_su[su_key].append(int(node_id))
                self.consumed_ids_by_su[su_key] = [
                    int(value)
                    for value in self.consumed_ids_by_su[su_key]
                    if int(value) != int(node_id)
                ]
                self._generic_claim_excluded_node_ids.discard(int(node_id))
            self.initial_ids_by_su[su_key] = sorted(set(self.initial_ids_by_su[su_key]))
            self.remaining_ids_by_su[su_key] = sorted(set(self.remaining_ids_by_su[su_key]))
            self.consumed_ids_by_su[su_key] = sorted(set(self.consumed_ids_by_su[su_key]))

    def build_ledger(self, shortage_counts: Dict[str, int]) -> TopologyPoolLedger:
        return TopologyPoolLedger(
            initial_ids_by_su={
                str(key): list(values or [])
                for key, values in self.initial_ids_by_su.items()
            },
            remaining_ids_by_su={
                str(key): list(values or [])
                for key, values in self.remaining_ids_by_su.items()
            },
            consumed_ids_by_su={
                str(key): list(values or [])
                for key, values in self.consumed_ids_by_su.items()
            },
            initial_counts={
                str(key): len(list(values or []))
                for key, values in self.initial_ids_by_su.items()
            },
            consumed_counts={
                str(key): len(list(values or []))
                for key, values in self.consumed_ids_by_su.items()
            },
            remaining_counts={
                str(key): len(list(values or []))
                for key, values in self.remaining_ids_by_su.items()
            },
            shortage_counts={
                str(key): int(value)
                for key, value in dict(shortage_counts or {}).items()
            },
        )

    def available_counts(self) -> Dict[str, int]:
        return {
            str(key): len(list(values or []))
            for key, values in self.remaining_ids_by_su.items()
        }
