from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from .stage3_models import TopologyAssemblyProduct


def _append_pool_chain_product(
    products: List[Any],
    *,
    product_kind: str,
    labels: Sequence[int],
    node_ids: Sequence[int],
    reason: str,
    commit_product: Optional[Callable[[TopologyAssemblyProduct], bool]] = None,
) -> bool:
    claimed_resource_ids: Dict[str, List[int]] = defaultdict(list)
    for label, node_id in zip(list(labels or []), list(node_ids or [])):
        claimed_resource_ids[str(int(label))].append(int(node_id))
    product = TopologyAssemblyProduct(
        product_kind=str(product_kind),
        product_type=str(product_kind),
        actual_topology_su="-".join(str(int(value)) for value in list(labels or [])),
        actual_topology_gid="-".join(str(int(value)) for value in list(node_ids or [])),
        source_fragment_ids=[],
        source_fragment_classes=["resource_pool"],
        source_node_ids=[int(value) for value in list(node_ids or [])],
        materialized_node_ids=[int(value) for value in list(node_ids or [])],
        claimed_resource_ids={
            str(key): list(values) for key, values in claimed_resource_ids.items()
        },
        closed=True,
        reason=str(reason),
    )
    if commit_product is not None:
        return bool(commit_product(product))
    products.append(product)
    return True


def assemble_residual_resource_chains(
    *,
    allocator: Any,
    products: List[Any],
    current_flex_count: int,
    flex_lower: int,
    flex_target: int,
    flex_upper: int,
    commit_product: Optional[Callable[[TopologyAssemblyProduct], bool]] = None,
) -> bool:
    progress = False
    target = max(int(flex_lower), min(int(flex_upper), int(flex_target)))
    flex_count = int(current_flex_count)
    while True:
        if int(flex_count) < int(target):
            snapshot = allocator.snapshot()
            left = allocator.claim(11, 1)
            middle = allocator.claim(23, 3) if left is not None else None
            right = allocator.claim(11, 1) if middle is not None else None
            if left is None or middle is None or right is None:
                allocator.rollback(snapshot)
                left = allocator.claim(11, 1)
                middle = allocator.claim(23, 2) if left is not None else None
                right = allocator.claim(11, 1) if middle is not None else None
            if left is not None and middle is not None and right is not None:
                committed = _append_pool_chain_product(
                    products,
                    product_kind="residual_flex_chain",
                    labels=[11] + [23] * len(middle) + [11],
                    node_ids=list(left) + list(middle) + list(right),
                    reason="residual_resource_self_closure:flex",
                    commit_product=commit_product,
                )
                if not committed:
                    allocator.rollback(snapshot)
                    break
                progress = True
                flex_count += 1
                continue
            allocator.rollback(snapshot)

        snapshot = allocator.snapshot()
        left = allocator.claim(11, 1)
        middle = allocator.claim(23, 2) if left is not None else None
        right = allocator.claim(22, 1) if middle is not None else None
        if left is None or middle is None or right is None:
            allocator.rollback(snapshot)
            left = allocator.claim(11, 1)
            middle = allocator.claim(23, 1) if left is not None else None
            right = allocator.claim(22, 1) if middle is not None else None
        if left is not None and middle is not None and right is not None:
            committed = _append_pool_chain_product(
                products,
                product_kind="residual_side_chain",
                labels=[11] + [23] * len(middle) + [22],
                node_ids=list(left) + list(middle) + list(right),
                reason="residual_resource_self_closure:side",
                commit_product=commit_product,
            )
            if not committed:
                allocator.rollback(snapshot)
                break
            progress = True
            continue
        allocator.rollback(snapshot)
        break
    return bool(progress)
