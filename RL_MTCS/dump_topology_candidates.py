from __future__ import annotations

import argparse
import ast
import copy
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .RL_allocator import FlexAllocator
    from .allocator_patterns import state_for_node
    from .allocator_models import _chain_required_resource_counts
except ImportError:  # Allows direct execution: python RL_MTCS/dump_topology_candidates.py ...
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from RL_MTCS.RL_allocator import FlexAllocator
    from RL_MTCS.allocator_patterns import state_for_node
    from RL_MTCS.allocator_models import _chain_required_resource_counts


BASE_BACKBONE_RESOURCE_SU = {10, 11, 12, 13, 22, 23, 26, 30}
IGNORED_TOPOLOGY_SU = {10, 11, 12, 13, 22, 23, 26, 30}
RESOURCE_KEYS = ("11", "22", "23", "24", "25")
FLEX_TOPOLOGY_TYPES = {"bridge", "branch_bridge"}
SIDE_TOPOLOGY_TYPES = {"side", "branch_side"}
RING_TOPOLOGY_TYPES = {"side_ring", "vertical_ring", "fused_side_ring"}
SU_MATRIX_VALUES = tuple(range(33))
SU_MATRIX_COLS = 11


def _read_block_c_targets_from_final_outputs(csv_path: str) -> Tuple[Any, Any, Optional[str]]:
    try:
        import torch
    except ModuleNotFoundError:
        return None, None, "missing_torch"
    base_dir = Path(csv_path).expanduser().resolve().parent
    spectrum_path = base_dir / "final_nmr_comparison.csv"
    element_path = base_dir / "final_element_composition.csv"
    S_target = None
    E_target = None
    if spectrum_path.exists():
        try:
            ppm_vals: List[float] = []
            target_vals: List[float] = []
            with open(spectrum_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        ppm_vals.append(float(row.get("ppm", 0.0)))
                        target_vals.append(float(row.get("target", 0.0)))
                    except Exception:
                        continue
            if target_vals:
                S_target = torch.tensor(target_vals, dtype=torch.float)
        except Exception as exc:
            return None, None, f"spectrum_read_failed:{exc}"
    if element_path.exists():
        try:
            targets: Dict[str, float] = {}
            with open(element_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    elem = str(row.get("element", "") or "").strip().upper()
                    if elem:
                        targets[elem] = float(row.get("target", 0.0) or 0.0)
            if targets:
                E_target = torch.tensor(
                    [float(targets.get(sym, 0.0)) for sym in ("C", "H", "O", "N", "S", "X")],
                    dtype=torch.float,
                )
        except Exception as exc:
            return S_target, None, f"element_read_failed:{exc}"
    if S_target is None or E_target is None:
        missing = []
        if S_target is None:
            missing.append(str(spectrum_path))
        if E_target is None:
            missing.append(str(element_path))
        return S_target, E_target, "missing_targets:" + ",".join(missing)
    return S_target, E_target, None


def _parse_optional_int(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _parse_int_tuple(value: Any) -> Tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return tuple()
    if not isinstance(parsed, (list, tuple)):
        return tuple()
    out: List[int] = []
    for item in parsed:
        try:
            out.append(int(item))
        except Exception:
            continue
    return tuple(out)


def _read_nodes(csv_path: str) -> List[Any]:
    nodes: List[Any] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = _parse_optional_int(row.get("global_id"))
            su_type = _parse_optional_int(row.get("su_type"))
            if gid is None or su_type is None:
                continue
            degree = _parse_optional_int(row.get("target_hop1_degree"))
            if degree is None:
                degree = _parse_optional_int(row.get("target_degree"))
            if degree is None:
                degree = _parse_optional_int(row.get("actual_hop1_degree"))
            fixed_anchor_count = _parse_optional_int(row.get("target_fixed_anchor_count")) or 0
            anchor_mode = str(row.get("special_anchor_mode") or row.get("anchor_mode") or "").strip() or None
            anchor_partition = (
                str(row.get("special_anchor_partition") or row.get("anchor_partition") or "").strip() or None
            )
            nodes.append(
                SimpleNamespace(
                    global_id=gid,
                    su_type=su_type,
                    su_name=str(row.get("su_name") or "").strip(),
                    hop1_ids=_parse_int_tuple(row.get("hop1_ids")),
                    hop1_ms=_parse_int_tuple(row.get("hop1_ms")),
                    hop2_ms=_parse_int_tuple(row.get("hop2_ms")),
                    target_degree=degree,
                    target_hop1_degree=degree,
                    actual_hop1_degree=_parse_optional_int(row.get("actual_hop1_degree")),
                    target_fixed_anchor_count=fixed_anchor_count,
                    anchor_mode=anchor_mode,
                    special_anchor_mode=anchor_mode,
                    anchor_partition=anchor_partition,
                    special_anchor_partition=anchor_partition,
                    raw_row=dict(row),
                )
            )
    return nodes


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _nodes_to_histogram(nodes: Sequence[Any]) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        hist = [0 for _ in range(33)]
        for node in list(nodes or []):
            try:
                su_i = int(getattr(node, "su_type", -1))
            except Exception:
                su_i = -1
            if 0 <= int(su_i) < len(hist):
                hist[int(su_i)] += 1
        return hist

    hist = torch.zeros(33, dtype=torch.long)
    for node in list(nodes or []):
        try:
            su_i = int(getattr(node, "su_type", -1))
        except Exception:
            su_i = -1
        if 0 <= int(su_i) < int(hist.numel()):
            hist[int(su_i)] += 1
    return hist


def _hist_delta(before: Any, after: Any) -> Dict[int, int]:
    out: Dict[int, int] = {}
    try:
        n_before = int(before.numel()) if hasattr(before, "numel") else len(before)
        n_after = int(after.numel()) if hasattr(after, "numel") else len(after)
        n = min(n_before, n_after)
    except Exception:
        return out
    for idx in range(n):
        b_raw = before[int(idx)]
        a_raw = after[int(idx)]
        b_val = int(b_raw.item()) if hasattr(b_raw, "item") else int(b_raw)
        a_val = int(a_raw.item()) if hasattr(a_raw, "item") else int(a_raw)
        delta = int(a_val) - int(b_val)
        if int(delta) != 0:
            out[int(idx)] = int(delta)
    return out


def _allocate_nodes(nodes: Sequence[Any], show_allocator_log: bool = False) -> Any:
    allocator = FlexAllocator(nodes=list(nodes or []))
    if bool(show_allocator_log):
        return allocator.allocate()
    with redirect_stdout(io.StringIO()):
        return allocator.allocate()


def _node_gid(node: Any) -> int:
    try:
        return int(getattr(node, "global_id", -1))
    except Exception:
        return -1


def _node_su(node: Any) -> int:
    try:
        return int(getattr(node, "su_type", -1))
    except Exception:
        return -1


def _node_sort_key(node: Any) -> Tuple[int, int]:
    gid = _node_gid(node)
    return (gid if gid >= 0 else 10**12, _node_su(node))


def _node_su_counter(nodes: Sequence[Any]) -> Counter[int]:
    return Counter(_node_su(node) for node in list(nodes or []) if _node_su(node) >= 0)


def _clone_nodes(nodes: Sequence[Any]) -> List[Any]:
    return copy.deepcopy(list(nodes or []))


def _counter_delta(before: Counter[int], after: Counter[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for key in sorted(set(before) | set(after)):
        delta = int(after.get(int(key), 0)) - int(before.get(int(key), 0))
        if delta:
            out[int(key)] = int(delta)
    return out


def _set_node_su(node: Any, su_type: int) -> None:
    setattr(node, "su_type", int(su_type))
    raw_row = getattr(node, "raw_row", None)
    if isinstance(raw_row, dict):
        for key in ("su_type", "center_su_idx", "su_idx", "type"):
            if key in raw_row:
                raw_row[key] = str(int(su_type))


def _refresh_node_counters_local(nodes: Sequence[Any]) -> None:
    node_lookup = {
        _node_gid(node): node
        for node in list(nodes or [])
        if _node_gid(node) >= 0
    }
    for node in list(nodes or []):
        hop1_counter: Counter[int] = Counter()
        hop1_ids = [int(x) for x in list(getattr(node, "hop1_ids", ()) or ())]
        if hop1_ids:
            for nid in hop1_ids:
                nb = node_lookup.get(int(nid))
                if nb is not None and _node_su(nb) >= 0:
                    hop1_counter[_node_su(nb)] += 1
        else:
            for su_type in list(getattr(node, "hop1_ms", ()) or ()):
                try:
                    hop1_counter[int(su_type)] += 1
                except Exception:
                    continue
        setattr(node, "hop1_su", hop1_counter)

    for node in list(nodes or []):
        hop2_counter: Counter[int] = Counter()
        hop1_ids = [int(x) for x in list(getattr(node, "hop1_ids", ()) or ())]
        center_id = _node_gid(node)
        if hop1_ids:
            for nid in hop1_ids:
                nb = node_lookup.get(int(nid))
                if nb is None:
                    continue
                for nb2_id in list(getattr(nb, "hop1_ids", ()) or ()):
                    try:
                        nb2_id_i = int(nb2_id)
                    except Exception:
                        continue
                    if int(nb2_id_i) == int(center_id):
                        continue
                    nb2 = node_lookup.get(int(nb2_id_i))
                    if nb2 is not None and _node_su(nb2) >= 0:
                        hop2_counter[_node_su(nb2)] += 1
        else:
            for su_type in list(getattr(node, "hop2_ms", ()) or ()):
                try:
                    hop2_counter[int(su_type)] += 1
                except Exception:
                    continue
        setattr(node, "hop2_su", hop2_counter)


def _nodes_to_layer_nodes(nodes: Sequence[Any]) -> List[Any]:
    import torch
    from model.shared.inverse_common import _NodeV3

    out: List[Any] = []
    for node in list(nodes or []):
        gid = _node_gid(node)
        su = _node_su(node)
        if gid < 0 or su < 0:
            continue
        layer_node = _NodeV3(
            int(gid),
            int(su),
            target_hop1_degree=getattr(node, "target_hop1_degree", None),
            special_anchor_partition=(
                getattr(node, "special_anchor_partition", None)
                or getattr(node, "anchor_partition", None)
            ),
            target_fixed_anchor_count=getattr(node, "target_fixed_anchor_count", None),
            special_anchor_mode=(
                getattr(node, "special_anchor_mode", None)
                or getattr(node, "anchor_mode", None)
            ),
        )
        layer_node.hop1_ids = [int(x) for x in list(getattr(node, "hop1_ids", []) or [])]
        layer_node.fixed_hop1_ids = set(int(x) for x in list(getattr(node, "fixed_hop1_ids", set()) or set()))
        layer_node.hop1_su = Counter(int(k) for k in list(getattr(node, "hop1_ms", []) or []))
        try:
            hop1_counter = getattr(node, "hop1_su", None)
            if isinstance(hop1_counter, Counter):
                layer_node.hop1_su = Counter({int(k): int(v) for k, v in hop1_counter.items()})
        except Exception:
            pass
        layer_node.hop2_su = Counter(int(k) for k in list(getattr(node, "hop2_ms", []) or []))
        try:
            hop2_counter = getattr(node, "hop2_su", None)
            if isinstance(hop2_counter, Counter):
                layer_node.hop2_su = Counter({int(k): int(v) for k, v in hop2_counter.items()})
        except Exception:
            pass
        try:
            layer_node.z_vec = torch.zeros(16, dtype=torch.float)
        except Exception:
            pass
        out.append(layer_node)
    out.sort(key=lambda n: int(getattr(n, "global_id", 0)))
    return out


def _sync_nodes_from_layer_nodes(nodes: List[Any], layer_nodes: Sequence[Any]) -> None:
    by_gid = {
        _node_gid(node): node
        for node in list(nodes or [])
        if _node_gid(node) >= 0
    }
    for layer_node in list(layer_nodes or []):
        try:
            gid = int(getattr(layer_node, "global_id", -1))
        except Exception:
            continue
        node = by_gid.get(int(gid))
        if node is None:
            continue
        _set_node_su(node, int(getattr(layer_node, "su_type", _node_su(node))))
        for attr in (
            "target_hop1_degree",
            "target_fixed_anchor_count",
            "special_anchor_mode",
            "special_anchor_partition",
        ):
            try:
                setattr(node, attr, copy.deepcopy(getattr(layer_node, attr, None)))
            except Exception:
                pass
        try:
            setattr(node, "anchor_mode", getattr(layer_node, "special_anchor_mode", None))
        except Exception:
            pass
        try:
            setattr(node, "anchor_partition", getattr(layer_node, "special_anchor_partition", None))
        except Exception:
            pass
        try:
            setattr(node, "hop1_ids", tuple(int(x) for x in list(getattr(layer_node, "hop1_ids", []) or [])))
        except Exception:
            pass
        try:
            setattr(node, "hop1_su", Counter({int(k): int(v) for k, v in getattr(layer_node, "hop1_su", Counter()).items()}))
            setattr(node, "hop1_ms", tuple(sorted(int(k) for k, v in getattr(layer_node, "hop1_su", Counter()).items() for _ in range(int(v)))))
        except Exception:
            pass
        try:
            setattr(node, "hop2_su", Counter({int(k): int(v) for k, v in getattr(layer_node, "hop2_su", Counter()).items()}))
            setattr(node, "hop2_ms", tuple(sorted(int(k) for k, v in getattr(layer_node, "hop2_su", Counter()).items() for _ in range(int(v)))))
        except Exception:
            pass
    _refresh_node_counters_local(nodes)


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _node_su_changes_by_gid(before_nodes: Sequence[Any], after_nodes: Sequence[Any]) -> List[Dict[str, int]]:
    before = {
        _node_gid(node): _node_su(node)
        for node in list(before_nodes or [])
        if _node_gid(node) >= 0
    }
    after = {
        _node_gid(node): _node_su(node)
        for node in list(after_nodes or [])
        if _node_gid(node) >= 0
    }
    out: List[Dict[str, int]] = []
    for gid in sorted(set(before) & set(after)):
        src = int(before[int(gid)])
        dst = int(after[int(gid)])
        if src == dst:
            continue
        out.append({
            "global_id": int(gid),
            "from": int(src),
            "to": int(dst),
        })
    return out


def _compact_block_c_step_meta(step_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(step_meta or {})
    keys = (
        "backend",
        "phase",
        "block_c_phase",
        "ok",
        "partial_progress",
        "candidate_pending_layer12",
        "n_moves",
        "n_rejected_moves",
        "final_scenario",
        "stop_reason",
        "fail_reasons",
        "initial_penalty",
        "best_penalty",
        "final_penalty",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key in meta:
            value = meta.get(key)
            if isinstance(value, tuple):
                value = list(value)
            out[str(key)] = value
    return out


def _layer12_trace_status(layer12_meta: Dict[str, Any], layer2_r2: Any = None) -> Dict[str, Any]:
    meta = dict(layer12_meta or {})
    mode = str(meta.get("mode", "") or "")
    ran = bool(meta.get("ran", mode in {"layer1_layer2", "layer12_failed"}))
    status: Dict[str, Any] = {
        "ran": bool(ran),
        "mode": str(mode or ("layer1_layer2" if ran else "not_run")),
    }
    if bool(ran):
        status["ok"] = bool(meta.get("ok", False))
    reason = str(meta.get("reason", "") or "")
    error = str(meta.get("error", "") or "")
    if reason:
        status["reason"] = reason
    if error:
        status["error"] = error
    if "layer1_ok" in meta:
        status["layer1_ok"] = bool(meta.get("layer1_ok", False))
    if "layer1_soft" in meta:
        status["layer1_soft"] = int(meta.get("layer1_soft", 0) or 0)
    r2 = _safe_float_or_none(meta.get("r2", layer2_r2))
    if r2 is None:
        r2 = _safe_float_or_none(layer2_r2)
    if r2 is not None:
        status["layer2_r2"] = float(r2)
    template_lib = str(meta.get("template_lib_path", "") or "")
    if template_lib:
        status["template_lib_path"] = template_lib
    return status


def _print_block_c_step_topology_trace(
    *,
    step_index: int,
    previous_nodes: Sequence[Any],
    seed_nodes: Sequence[Any],
    rebuilt_nodes: Sequence[Any],
    layer12_meta: Dict[str, Any],
    layer2_r2: Any = None,
    step_meta: Optional[Dict[str, Any]] = None,
    precomputed_result: Any = None,
    show_allocator_log: bool = False,
    show_debug_details: bool = False,
    show_node_coverage: bool = False,
) -> None:
    prev = _clone_nodes(previous_nodes)
    seed = _clone_nodes(seed_nodes)
    rebuilt = _clone_nodes(rebuilt_nodes)
    for seq in (prev, seed, rebuilt):
        try:
            _refresh_node_counters_local(seq)
        except Exception:
            pass

    print(f"========== Block C Step {int(step_index)} Adjustment Trace ==========")
    print(f"candidate_su_changes: {_json(_node_su_changes_by_gid(prev, seed))}")
    print(f"layer1_2_status: {_json(_layer12_trace_status(layer12_meta, layer2_r2))}")
    print(f"layer1_2_su_changes: {_json(_node_su_changes_by_gid(seed, rebuilt))}")
    print(f"hist_delta_previous_to_seed: {_json(_hist_delta(_nodes_to_histogram(prev), _nodes_to_histogram(seed)))}")
    print(f"hist_delta_seed_to_layer1_2: {_json(_hist_delta(_nodes_to_histogram(seed), _nodes_to_histogram(rebuilt)))}")
    print(f"hist_delta_previous_to_layer1_2: {_json(_hist_delta(_nodes_to_histogram(prev), _nodes_to_histogram(rebuilt)))}")
    compact_meta = _compact_block_c_step_meta(dict(step_meta or {}))
    if compact_meta:
        print(f"block_c_step_meta: {_json(compact_meta)}")
    print("")

    try:
        step_result = precomputed_result if precomputed_result is not None else _allocate_nodes(
            rebuilt,
            show_allocator_log=bool(show_allocator_log),
        )
    except Exception as exc:
        print(f"topology_step_dump_error: {exc}")
        print("")
        return
    dump_args = SimpleNamespace(
        show_debug_details=bool(show_debug_details),
        show_node_coverage=bool(show_node_coverage),
    )
    _print_dump_sections(step_result, rebuilt, dump_args)


def _sync_layer4_fixed_meta_from_nodes(adjuster: Any, nodes: Sequence[Any]) -> None:
    special_degree_meta: Dict[int, Dict[int, int]] = {
        19: {1: 0, 2: 0, 3: 0},
        20: {1: 0, 2: 0, 3: 0},
        21: {2: 0, 3: 0},
    }
    partition_19 = {
        "ether": {1: 0, 2: 0, 3: 0},
        "thio": {1: 0, 2: 0, 3: 0},
    }
    mode_meta = {
        19: {
            "ether_single": {1: 0, 2: 0, 3: 0},
            "ether_double": {2: 0, 3: 0},
            "thio_single": {1: 0, 2: 0, 3: 0},
            "thio_double": {2: 0, 3: 0},
        },
        20: {
            "single": {1: 0, 2: 0, 3: 0},
            "double": {2: 0, 3: 0},
        },
        21: {
            "single": {2: 0, 3: 0},
            "double": {2: 0, 3: 0},
        },
    }
    for node in list(nodes or []):
        su = _node_su(node)
        if su not in {19, 20, 21}:
            continue
        deg = getattr(node, "target_hop1_degree", None)
        try:
            deg_i = int(deg) if deg is not None else None
        except Exception:
            deg_i = None
        if deg_i is None:
            deg_i = max(0, len(list(getattr(node, "hop1_ids", []) or [])))
        if int(deg_i) in special_degree_meta.get(int(su), {}):
            special_degree_meta[int(su)][int(deg_i)] += 1
        fixed_cnt = 0
        try:
            fixed_cnt = max(0, int(getattr(node, "target_fixed_anchor_count", 0) or 0))
        except Exception:
            fixed_cnt = 0
        part = str(
            getattr(node, "special_anchor_partition", None)
            or getattr(node, "anchor_partition", None)
            or ""
        )
        mode = "double" if fixed_cnt >= 2 else "single"
        if su == 19:
            if part not in {"ether", "thio"}:
                part = "ether"
            if int(deg_i) in partition_19[str(part)]:
                partition_19[str(part)][int(deg_i)] += 1
            key = f"{part}_{mode}"
            if key in mode_meta[19] and int(deg_i) in mode_meta[19][key]:
                mode_meta[19][key][int(deg_i)] += 1
        elif su == 20:
            key = "double" if mode == "double" else "single"
            if key in mode_meta[20] and int(deg_i) in mode_meta[20][key]:
                mode_meta[20][key][int(deg_i)] += 1
        elif su == 21:
            key = "double" if mode == "double" else "single"
            if key in mode_meta[21] and int(deg_i) in mode_meta[21][key]:
                mode_meta[21][key][int(deg_i)] += 1

    fixed_meta = {
        "special_degree_meta": special_degree_meta,
        "special_partition_meta": {19: partition_19},
        "special_anchor_mode_meta": mode_meta,
    }
    try:
        adjuster.fixed_partition_meta = copy.deepcopy(fixed_meta)
    except Exception:
        pass
    layer0 = getattr(adjuster, "layer0_estimator", None)
    if layer0 is not None:
        try:
            layer0.fixed_partition_meta = copy.deepcopy(fixed_meta)
            layer0.special_degree_meta = copy.deepcopy(special_degree_meta)
            layer0.special_partition_meta = copy.deepcopy({19: partition_19})
        except Exception:
            pass


def _convert_aromatic_to_11_cycle(nodes: Sequence[Any], max_count: int) -> List[Dict[str, int]]:
    changes: List[Dict[str, int]] = []
    cycle = [12, 12, 13]
    cursor = 0
    for _ in range(max(0, int(max_count))):
        picked = None
        picked_src = None
        for offset in range(len(cycle)):
            idx = (int(cursor) + int(offset)) % len(cycle)
            src = int(cycle[idx])
            cands = [
                node for node in sorted(list(nodes or []), key=_node_sort_key)
                if _node_su(node) == int(src)
            ]
            if cands:
                picked = cands[0]
                picked_src = int(src)
                cursor = (idx + 1) % len(cycle)
                break
        if picked is None or picked_src is None:
            break
        _set_node_su(picked, 11)
        changes.append({"gid": int(_node_gid(picked)), "from": int(picked_src), "to": 11})
    return changes


def _convert_path(nodes: Sequence[Any], path: Sequence[int], max_count: int) -> List[Dict[str, int]]:
    changes: List[Dict[str, int]] = []
    path_i = [int(x) for x in list(path or [])]
    if len(path_i) < 2:
        return changes
    for _ in range(max(0, int(max_count))):
        picked = [
            node for node in sorted(list(nodes or []), key=_node_sort_key)
            if _node_su(node) == int(path_i[0])
        ]
        if not picked:
            break
        node = picked[0]
        prev = int(path_i[0])
        for dst in path_i[1:]:
            _set_node_su(node, int(dst))
            changes.append({"gid": int(_node_gid(node)), "from": int(prev), "to": int(dst)})
            prev = int(dst)
    return changes


def _convert_aromatic_chain(nodes: Sequence[Any], tail_path: Sequence[int], max_count: int) -> List[Dict[str, int]]:
    changes: List[Dict[str, int]] = []
    tail = [int(x) for x in list(tail_path or [])]
    for _ in range(max(0, int(max_count))):
        first = _convert_aromatic_to_11_cycle(nodes, 1)
        if not first:
            break
        changes.extend(first)
        gid = int(first[-1]["gid"])
        node = next((n for n in list(nodes or []) if int(_node_gid(n)) == int(gid)), None)
        if node is None:
            break
        prev = 11
        for dst in tail:
            _set_node_su(node, int(dst))
            changes.append({"gid": int(gid), "from": int(prev), "to": int(dst)})
            prev = int(dst)
    return changes


def _unit_count_for_path_changes(changes: Sequence[Dict[str, int]], path: Sequence[int]) -> int:
    steps = max(1, len([int(x) for x in list(path or [])]) - 1)
    return int(len(list(changes or [])) // int(steps))


def _apply_fallback_recommendation(
    nodes: Sequence[Any],
    rec: Dict[str, Any],
    allowed_phases: set[str],
) -> Optional[Dict[str, Any]]:
    phase = str(rec.get("phase", "closure_repair") or "closure_repair")
    if phase not in allowed_phases:
        return None
    if not bool(rec.get("actionable", True)):
        return None
    op_name = str(rec.get("op", "") or "")
    count = max(1, int(rec.get("count", 1) or 1))
    before_counter = _node_su_counter(nodes)
    changes: List[Dict[str, int]] = []

    path_map = {
        "22->23": [22, 23],
        "24->23": [24, 23],
        "11->23": [11, 23],
        "23->24": [23, 24],
        "25->24": [25, 24],
        "23->22": [23, 22],
        "23->11": [23, 11],
        "22->23->24": [22, 23, 24],
        "11->23->24": [11, 23, 24],
        "24->23->22": [24, 23, 22],
        "11->23->22": [11, 23, 22],
        "22->23->11": [22, 23, 11],
        "24->23->11": [24, 23, 11],
    }
    if op_name == "12/13->11":
        changes.extend(_convert_aromatic_to_11_cycle(nodes, count))
    elif op_name == "12/13->11->23":
        changes.extend(_convert_aromatic_chain(nodes, [23], count))
    elif op_name == "12/13->11->23->24":
        changes.extend(_convert_aromatic_chain(nodes, [23, 24], count))
    elif op_name == "12/13->11->23->22":
        changes.extend(_convert_aromatic_chain(nodes, [23, 22], count))
    elif op_name in path_map:
        changes.extend(_convert_path(nodes, path_map[str(op_name)], count))
    elif op_name == "flex_short_make_F1":
        endpoint_changes: List[Dict[str, int]] = []
        endpoint_changes.extend(_convert_aromatic_to_11_cycle(nodes, count))
        changes.extend(endpoint_changes)

        converted = _convert_path(nodes, [22, 23], count)
        changes.extend(converted)
        internal_units = _unit_count_for_path_changes(converted, [22, 23])
        if internal_units < count:
            converted = _convert_path(nodes, [24, 23], count - internal_units)
            changes.extend(converted)
            internal_units += _unit_count_for_path_changes(converted, [24, 23])
        if internal_units < count:
            changes.extend(_convert_path(nodes, [11, 23], count - internal_units))
    elif op_name == "ordinary_pending_F1_resources":
        endpoint_changes: List[Dict[str, int]] = []
        endpoint_changes.extend(_convert_aromatic_to_11_cycle(nodes, count))
        changes.extend(endpoint_changes)

        converted = _convert_path(nodes, [22, 23], count)
        changes.extend(converted)
        internal_units = _unit_count_for_path_changes(converted, [22, 23])
        if internal_units < count:
            converted = _convert_path(nodes, [24, 23], count - internal_units)
            changes.extend(converted)
            internal_units += _unit_count_for_path_changes(converted, [24, 23])
        if internal_units < count:
            changes.extend(_convert_path(nodes, [11, 23], count - internal_units))
    elif op_name == "ordinary_pending_F2_resources":
        converted = _convert_path(nodes, [22, 23], count)
        changes.extend(converted)
        internal_units = _unit_count_for_path_changes(converted, [22, 23])
        if internal_units < count:
            converted = _convert_path(nodes, [24, 23], count - internal_units)
            changes.extend(converted)
            internal_units += _unit_count_for_path_changes(converted, [24, 23])
        if internal_units < count:
            changes.extend(_convert_path(nodes, [11, 23], count - internal_units))
        side_changes = _convert_path(nodes, [23, 22], count)
        changes.extend(side_changes)
        side_units = _unit_count_for_path_changes(side_changes, [23, 22])
        if side_units < count:
            changes.extend(_convert_path(nodes, [24, 23, 22], count - side_units))
    elif op_name == "like24_unclosed_rebalance":
        converted = _convert_path(nodes, [23, 24], count)
        changes.extend(converted)
        applied_units = _unit_count_for_path_changes(converted, [23, 24])
        if applied_units < count:
            converted = _convert_path(nodes, [25, 24], count - applied_units)
            changes.extend(converted)
            applied_units += _unit_count_for_path_changes(converted, [25, 24])
        if applied_units < count:
            changes.extend(_convert_path(nodes, [22, 23, 24], count - applied_units))
    elif op_name == "reduce_flex_prefer_side":
        changes.extend(_convert_path(nodes, [23, 22], count))
        applied_units = _unit_count_for_path_changes(changes, [23, 22])
        if applied_units < count:
            changes.extend(_convert_path(nodes, [24, 23, 22], count - applied_units))

    if not changes:
        return None
    _refresh_node_counters_local(nodes)
    after_counter = _node_su_counter(nodes)
    return {
        "op": f"fallback_{op_name}x{len(changes)}",
        "stage": "fallback_topology_recommendation",
        "backend": "no_torch_fallback",
        "phase": str(phase),
        "source_op": str(op_name),
        "requested_count": int(count),
        "applied_changes": int(len(changes)),
        "reason": str(rec.get("reason", "") or ""),
        "delta_counts": _counter_delta(before_counter, after_counter),
        "changes": list(changes),
    }


def _allocation_ok_from_result(result: Any) -> bool:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    shortage = dict(diag.get("resource_shortage", {}) or {})
    flex_short = max(0, int(diag.get("flex_short", 0) or 0))
    flex_excess = max(0, int(diag.get("flex_excess", 0) or 0))
    pending_total = int(
        getattr(
            result,
            "pending_closure_total_count",
            len(getattr(result, "pending_closure_topologies", []) or []),
        )
        or 0
    )
    return bool(
        bool(diag.get("ok", False))
        and int(pending_total) == 0
        and int(getattr(result, "unallocated_branch", 0) or 0) == 0
        and all(int(shortage.get(key, 0) or 0) == 0 for key in RESOURCE_KEYS)
        and int(flex_short) == 0
        and int(flex_excess) == 0
    )


def _fallback_result_scenario(result: Any) -> str:
    if _allocation_ok_from_result(result):
        return "ok"
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    shortage = dict(diag.get("resource_shortage", {}) or {})
    pending_total = int(
        getattr(
            result,
            "pending_closure_total_count",
            len(getattr(result, "pending_closure_topologies", []) or []),
        )
        or 0
    )
    for key in RESOURCE_KEYS:
        if int(shortage.get(key, 0) or 0) > 0:
            return f"{key}_shortage"
    if int(pending_total) > 0:
        return "pending_closure"
    if int(getattr(result, "unallocated_branch", 0) or 0) > 0:
        return "abstract_topology_gap"
    if int(diag.get("flex_short", 0) or 0) > 0:
        return "flex_shortage"
    if int(diag.get("flex_excess", 0) or 0) > 0:
        return "flex_excess"
    if not bool(diag.get("ok", False)):
        return "topology_assembly_failed"
    return "not_ok"


def _fallback_penalty_from_result(result: Any) -> Tuple[int, ...]:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    shortage = dict(diag.get("resource_shortage", {}) or {})
    flex_short = max(0, int(diag.get("flex_short", 0) or 0))
    flex_excess = max(0, int(diag.get("flex_excess", 0) or 0))
    shortage_total = sum(max(0, int(shortage.get(key, 0) or 0)) for key in RESOURCE_KEYS)
    pending_total = int(
        getattr(
            result,
            "pending_closure_total_count",
            len(getattr(result, "pending_closure_topologies", []) or []),
        )
        or 0
    )
    return (
        0 if _allocation_ok_from_result(result) else 1,
        max(0, int(pending_total)),
        int(shortage_total),
        max(0, int(shortage.get("11", 0) or 0)),
        max(0, int(shortage.get("22", 0) or 0)),
        max(0, int(shortage.get("23", 0) or 0)),
        max(0, int(shortage.get("24", 0) or 0)),
        max(0, int(shortage.get("25", 0) or 0)),
        max(0, int(getattr(result, "unallocated_branch", 0) or 0)),
        int(flex_short + flex_excess),
        int(flex_excess),
        int(flex_short),
    )


def _fallback_diag_summary(result: Any) -> Dict[str, Any]:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    shortage = dict(diag.get("resource_shortage", {}) or {})
    pending_blocking = int(getattr(result, "pending_closure_count", 0) or 0)
    pending_total = int(
        getattr(
            result,
            "pending_closure_total_count",
            len(getattr(result, "pending_closure_topologies", []) or []),
        )
        or 0
    )
    return {
        "ok": bool(_allocation_ok_from_result(result)),
        "topology_assembly_ok": bool(diag.get("ok", False)),
        "pending_closure_count": int(pending_blocking),
        "pending_closure_total_count": int(pending_total),
        "unallocated_branch": int(getattr(result, "unallocated_branch", 0) or 0),
        "required_extra_11": int(shortage.get("11", 0) or 0),
        "required_extra_22": int(shortage.get("22", 0) or 0),
        "required_extra_23": int(shortage.get("23", 0) or 0),
        "required_extra_24": int(shortage.get("24", 0) or 0),
        "required_extra_25": int(shortage.get("25", 0) or 0),
        "resource_shortage": {key: int(shortage.get(key, 0) or 0) for key in RESOURCE_KEYS},
        "remaining_11": int(getattr(result, "remaining_11", 0) or 0),
        "remaining_22": int(getattr(result, "remaining_22", 0) or 0),
        "remaining_23": int(getattr(result, "remaining_23", 0) or 0),
        "remaining_24": int(getattr(result, "remaining_24", 0) or 0),
        "remaining_25": int(getattr(result, "remaining_25", 0) or 0),
        "flex_count": int(diag.get("flex_count", 0) or 0),
        "flex_lower": int(diag.get("flex_lower", 0) or 0),
        "flex_upper": int(diag.get("flex_upper", 0) or 0),
        "flex_short": int(diag.get("flex_short", 0) or 0),
        "flex_excess": int(diag.get("flex_excess", 0) or 0),
        "recommended_adjustments": list(diag.get("recommended_adjustments", []) or []),
        "resource_ledger": dict(diag.get("resource_ledger", {}) or {}),
    }


def _run_block_c_adjustment_fallback(
    nodes: List[Any],
    phase: str,
    max_steps: int,
    show_block_c_log: bool = False,
    fallback_reason: str = "torch_unavailable",
    trace_steps: bool = True,
    show_allocator_log: bool = False,
    show_debug_details: bool = False,
    show_node_coverage: bool = False,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], Any, Any]:
    _refresh_node_counters_local(nodes)
    hist_before = _nodes_to_histogram(nodes)
    moves: List[Dict[str, Any]] = []
    allowed_phases = {"closure_repair", "flex_repair"}
    final_reason = "max_steps_exhausted"

    for step in range(max(0, int(max_steps))):
        result = _allocate_nodes(nodes, show_allocator_log=False)
        if _allocation_ok_from_result(result):
            final_reason = "ok"
            break
        diag = _fallback_diag_summary(result)
        penalty_before = _fallback_penalty_from_result(result)
        record: Optional[Dict[str, Any]] = None
        for rec in list(diag.get("recommended_adjustments", []) or []):
            candidate_nodes = _clone_nodes(nodes)
            candidate_record = _apply_fallback_recommendation(candidate_nodes, dict(rec or {}), allowed_phases)
            if candidate_record is None:
                continue
            candidate_result = _allocate_nodes(candidate_nodes, show_allocator_log=False)
            penalty_after = _fallback_penalty_from_result(candidate_result)
            if tuple(penalty_after) >= tuple(penalty_before):
                if bool(show_block_c_log):
                    print(
                        "  [BlockC-Fallback] "
                        f"reject step={step} op={candidate_record.get('source_op')} "
                        f"penalty_before={tuple(penalty_before)} penalty_after={tuple(penalty_after)}"
                    )
                continue
            if bool(trace_steps):
                _print_block_c_step_topology_trace(
                    step_index=int(step + 1),
                    previous_nodes=nodes,
                    seed_nodes=candidate_nodes,
                    rebuilt_nodes=candidate_nodes,
                    layer12_meta={
                        "ran": False,
                        "ok": True,
                        "mode": "not_run",
                        "reason": "fallback_backend_no_layer1_layer2",
                    },
                    layer2_r2=float("nan"),
                    step_meta=dict(candidate_record),
                    precomputed_result=candidate_result,
                    show_allocator_log=bool(show_allocator_log),
                    show_debug_details=bool(show_debug_details),
                    show_node_coverage=bool(show_node_coverage),
                )
            nodes[:] = candidate_nodes
            record = dict(candidate_record)
            record["penalty_before"] = list(penalty_before)
            record["penalty_after"] = list(penalty_after)
            record["diagnostic_after"] = _fallback_diag_summary(candidate_result)
            if record is not None:
                break
        if record is None:
            final_reason = "no_improving_fallback_recommendation"
            break
        record["step"] = int(step)
        record["diagnostic_before"] = dict(diag)
        moves.append(record)
        if bool(show_block_c_log):
            print(
                "  [BlockC-Fallback] "
                f"step={step} op={record.get('source_op')} "
                f"phase={record.get('phase')} delta={record.get('delta_counts')}"
            )

    final_result = _allocate_nodes(nodes, show_allocator_log=False)
    final_diag = _fallback_diag_summary(final_result)
    final_ok = bool(_allocation_ok_from_result(final_result))
    scenario = "ok" if final_ok else _fallback_result_scenario(final_result)
    branch_phase = str(phase or "full").strip().lower()
    branch_ok = bool(final_ok) if branch_phase in {"branch", "full"} else True
    extra_ok = bool(final_ok) if branch_phase in {"extra", "full"} else True
    meta = {
        "backend": "no_torch_fallback",
        "fallback_reason": str(fallback_reason),
        "phase": str(branch_phase),
        "block_c_phase": "fallback_topology_recommendations",
        "n_moves": int(len(moves)),
        "ok": bool(final_ok),
        "branch_ok": bool(branch_ok),
        "extra_ok": bool(extra_ok),
        "records": list(moves),
        "final_diag": dict(final_diag),
        "final_allocation": dict(final_diag),
        "final_scenario": str(scenario),
        "stop_reason": str(final_reason),
        "allowed_recommendation_phases": sorted(allowed_phases),
        "branch_meta": {
            "ok": bool(branch_ok),
            "n_moves": int(len(moves)),
            "final_scenario": str(scenario),
        },
        "extra_meta": {
            "ok": bool(extra_ok),
            "n_moves": 0,
            "final_scenario": "fallback_extra_not_run",
            "reason": "no_torch_fallback_only_runs_closure_and_flex_repair",
        },
    }
    return _nodes_to_histogram(nodes), list(moves), meta, hist_before, _nodes_to_histogram(nodes)


def _run_layer4_block_c_adjustment(
    nodes: List[Any],
    phase: str,
    max_steps: int,
    show_block_c_log: bool = False,
    csv_path: Optional[str] = None,
    template_lib_path: Optional[str] = None,
    trace_steps: bool = True,
    show_allocator_log: bool = False,
    show_debug_details: bool = False,
    show_node_coverage: bool = False,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], Any, Any]:
    import torch
    from model.inverse.layer4.inverse_layer4 import Layer4Adjuster

    adjuster = Layer4Adjuster()
    layer_nodes = _nodes_to_layer_nodes(nodes)
    adjuster._refresh_node_counters(layer_nodes)
    _sync_layer4_fixed_meta_from_nodes(adjuster, layer_nodes)
    hist_before = _nodes_to_histogram(nodes)
    S_target = None
    E_target = None
    target_reason = "not_requested"
    if csv_path:
        S_target, E_target, target_reason = _read_block_c_targets_from_final_outputs(str(csv_path))
    if E_target is None:
        E_target = torch.matmul(hist_before.float(), adjuster.E_SU.cpu())
    ppm = None
    diff = None
    trace_previous_nodes: List[Any] = _clone_nodes(layer_nodes)

    def _emit_layer4_step_trace(
        *,
        seed_snapshot: Sequence[Any],
        rebuilt_snapshot: Sequence[Any],
        step_index: int,
        step_meta: Dict[str, Any],
        layer12_meta: Dict[str, Any],
        layer2_r2: Any = None,
    ) -> None:
        nonlocal trace_previous_nodes
        if not bool(trace_steps):
            trace_previous_nodes = _clone_nodes(rebuilt_snapshot)
            return
        _print_block_c_step_topology_trace(
            step_index=int(step_index),
            previous_nodes=trace_previous_nodes,
            seed_nodes=seed_snapshot,
            rebuilt_nodes=rebuilt_snapshot,
            layer12_meta=dict(layer12_meta or {}),
            layer2_r2=layer2_r2,
            step_meta=dict(step_meta or {}),
            show_allocator_log=bool(show_allocator_log),
            show_debug_details=bool(show_debug_details),
            show_node_coverage=bool(show_node_coverage),
        )
        trace_previous_nodes = _clone_nodes(rebuilt_snapshot)

    def _run_layer12_rebuild(
        seed_nodes: List[Any],
        H_seed: Any,
        step_index: int,
        step_meta: Dict[str, Any],
    ) -> Tuple[List[Any], Any, Dict[str, Any], float]:
        from model.inverse.layer1.inverse_layer1 import (
            DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
            Layer1Assigner,
        )
        from model.inverse.layer2.inverse_layer2 import Layer2Estimator
        from model.shared.coarse_graph import E_SU

        seed_snapshot = copy.deepcopy(seed_nodes)
        if S_target is None or E_target is None:
            adjuster._refresh_node_counters(seed_nodes)
            layer12_meta = {
                "ran": False,
                "ok": True,
                "mode": "topology_only",
                "reason": str(target_reason),
            }
            _emit_layer4_step_trace(
                seed_snapshot=seed_snapshot,
                rebuilt_snapshot=seed_nodes,
                step_index=int(step_index),
                step_meta=dict(step_meta or {}),
                layer12_meta=layer12_meta,
                layer2_r2=float("nan"),
            )
            return seed_nodes, _nodes_to_histogram(seed_nodes), layer12_meta, float("nan")

        device = torch.device("cpu")
        assigner = Layer1Assigner(
            device="cpu",
            E_SU_tensor=E_SU,
            layer0_estimator=getattr(adjuster, "layer0_estimator", None),
        )
        try:
            _sync_layer4_fixed_meta_from_nodes(adjuster, seed_nodes)
            fixed_meta = copy.deepcopy(getattr(adjuster, "fixed_partition_meta", {}) or {})
            if getattr(adjuster, "layer0_estimator", None) is not None:
                assigner.layer0_estimator = adjuster.layer0_estimator
            else:
                assigner.layer0_estimator = SimpleNamespace(
                    fixed_partition_meta=copy.deepcopy(fixed_meta),
                    special_degree_meta=copy.deepcopy(fixed_meta.get("special_degree_meta", {}) or {}),
                    special_partition_meta=copy.deepcopy(fixed_meta.get("special_partition_meta", {}) or {}),
                )
        except Exception:
            pass
        layer1_cfg = {
            "eval_nmr": bool(template_lib_path),
            "eval_output_dir": None,
            "eval_lib_path": template_lib_path,
            "eval_hwhm": 1.0,
            "eval_allow_approx": True,
            "build_variant": int(step_index),
            "seed_nodes": copy.deepcopy(seed_nodes),
            "preserve_seed_node_ids": True,
            "enable_carbonyl_joint_adjust": False,
            "enable_hop1_adjust": bool(template_lib_path),
            "hop1_adjust_iterations": 2,
            "hop1_neg_threshold": -0.8,
            "hop1_pos_threshold": 0.8,
        }
        try:
            with redirect_stdout(io.StringIO()):
                rebuilt = assigner.layer1_assign(
                    H_init=H_seed.detach().clone().to(device),
                    S_target=S_target.detach().clone().to(device),
                    E_target=E_target.detach().clone().to(device),
                    **layer1_cfg,
                )
            ok, errors = assigner.validate_graph_consistency(
                rebuilt,
                H_seed.detach().clone().to(device),
                E_target=E_target.detach().clone().to(device),
                verbose=False,
            )
            accept_ok, accept_meta = assigner.assess_consistency_acceptance(
                list(errors or []),
                max_soft_errors=DEFAULT_MAX_SOFT_CONSISTENCY_ERRORS,
            )
        except Exception as exc:
            _emit_layer4_step_trace(
                seed_snapshot=seed_snapshot,
                rebuilt_snapshot=seed_nodes,
                step_index=int(step_index),
                step_meta=dict(step_meta or {}),
                layer12_meta={
                    "ran": True,
                    "ok": False,
                    "mode": "layer12_failed",
                    "error": f"layer1_failed:{exc}",
                    "template_lib_path": str(template_lib_path or ""),
                },
                layer2_r2=float("nan"),
            )
            raise
        if not bool(accept_ok):
            _emit_layer4_step_trace(
                seed_snapshot=seed_snapshot,
                rebuilt_snapshot=seed_nodes,
                step_index=int(step_index),
                step_meta=dict(step_meta or {}),
                layer12_meta={
                    "ran": True,
                    "ok": False,
                    "mode": "layer12_failed",
                    "layer1_ok": bool(ok),
                    "layer1_soft": int(accept_meta.get("n_soft", 0) or 0),
                    "error": f"layer1_graph_invalid:{errors[0] if errors else 'unknown'}",
                    "template_lib_path": str(template_lib_path or ""),
                },
                layer2_r2=float("nan"),
            )
            raise RuntimeError(f"layer1_graph_invalid:{errors[0] if errors else 'unknown'}")

        estimator = Layer2Estimator(device=device, lib_path=template_lib_path, vae_model=None)
        try:
            with redirect_stdout(io.StringIO()):
                rebuilt = estimator.layer2_assign(
                    nodes=rebuilt,
                    S_target=S_target.detach().clone().to(device),
                    E_target=E_target.detach().clone().to(device),
                    output_dir=None,
                    hwhm=1.0,
                )
        except Exception as exc:
            _emit_layer4_step_trace(
                seed_snapshot=seed_snapshot,
                rebuilt_snapshot=seed_nodes,
                step_index=int(step_index),
                step_meta=dict(step_meta or {}),
                layer12_meta={
                    "ran": True,
                    "ok": False,
                    "mode": "layer12_failed",
                    "layer1_ok": bool(ok),
                    "layer1_soft": int(accept_meta.get("n_soft", 0) or 0),
                    "error": f"layer2_failed:{exc}",
                    "template_lib_path": str(template_lib_path or ""),
                },
                layer2_r2=float("nan"),
            )
            raise
        adjuster._refresh_node_counters(rebuilt)
        H_nodes = _nodes_to_histogram(rebuilt)
        r2 = float("nan")
        try:
            S_recon = estimator.reconstruct_spectrum(rebuilt, hwhm=1.0)
            target = S_target.detach().clone().to(device).flatten()
            ss_res = torch.sum((target - S_recon.flatten()) ** 2)
            ss_tot = torch.sum((target - torch.mean(target)) ** 2).clamp(min=1e-12)
            r2 = float((1.0 - ss_res / ss_tot).detach().cpu().item())
        except Exception:
            r2 = float("nan")
        layer12_meta = {
            "ran": True,
            "ok": True,
            "mode": "layer1_layer2",
            "layer1_ok": bool(ok),
            "layer1_soft": int(accept_meta.get("n_soft", 0) or 0),
            "template_lib_path": str(template_lib_path or ""),
            "r2": _safe_float_or_none(r2),
        }
        _emit_layer4_step_trace(
            seed_snapshot=seed_snapshot,
            rebuilt_snapshot=rebuilt,
            step_index=int(step_index),
            step_meta=dict(step_meta or {}),
            layer12_meta=layer12_meta,
            layer2_r2=float(r2),
        )
        return rebuilt, H_nodes, layer12_meta, float(r2)

    def _run_loop() -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
        return adjuster.run_block_c_topology_closure_loop(
            nodes=layer_nodes,
            E_target=E_target,
            S_target=S_target,
            ppm=ppm,
            diff=diff,
            max_steps=max_steps,
            phase=str(phase),
            rebuild_callback=_run_layer12_rebuild,
            keep_unimproved=True,
            stop_on_cycle=True,
            show_log=True,
            return_best_on_exhaustion=False,
        )

    hist_after, moves, meta = _run_loop()
    _sync_nodes_from_layer_nodes(nodes, layer_nodes)
    meta = dict(meta or {})
    meta["target_source"] = "final_outputs" if not target_reason else str(target_reason)
    meta["template_lib_path"] = str(template_lib_path or "")
    return hist_after, list(moves or []), dict(meta or {}), hist_before, _nodes_to_histogram(nodes)


def _run_block_c_adjustment(
    nodes: List[Any],
    phase: str,
    max_steps: int,
    show_block_c_log: bool = False,
    backend: str = "auto",
    csv_path: Optional[str] = None,
    template_lib_path: Optional[str] = None,
    trace_steps: bool = True,
    show_allocator_log: bool = False,
    show_debug_details: bool = False,
    show_node_coverage: bool = False,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], Any, Any]:
    backend_name = str(backend or "auto").strip().lower()
    if backend_name == "fallback":
        return _run_block_c_adjustment_fallback(
            nodes,
            phase=phase,
            max_steps=max_steps,
            show_block_c_log=show_block_c_log,
            fallback_reason="forced_by_cli",
            trace_steps=trace_steps,
            show_allocator_log=show_allocator_log,
            show_debug_details=show_debug_details,
            show_node_coverage=show_node_coverage,
        )
    if backend_name == "layer4":
        return _run_layer4_block_c_adjustment(
            nodes,
            phase=phase,
            max_steps=max_steps,
            show_block_c_log=show_block_c_log,
            csv_path=csv_path,
            template_lib_path=template_lib_path,
            trace_steps=trace_steps,
            show_allocator_log=show_allocator_log,
            show_debug_details=show_debug_details,
            show_node_coverage=show_node_coverage,
        )
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        return _run_block_c_adjustment_fallback(
            nodes,
            phase=phase,
            max_steps=max_steps,
            show_block_c_log=show_block_c_log,
            fallback_reason=f"missing dependency: {getattr(exc, 'name', 'torch') or 'torch'}",
            trace_steps=trace_steps,
            show_allocator_log=show_allocator_log,
            show_debug_details=show_debug_details,
            show_node_coverage=show_node_coverage,
        )
    try:
        return _run_layer4_block_c_adjustment(
            nodes,
            phase=phase,
            max_steps=max_steps,
            show_block_c_log=show_block_c_log,
            csv_path=csv_path,
            template_lib_path=template_lib_path,
            trace_steps=trace_steps,
            show_allocator_log=show_allocator_log,
            show_debug_details=show_debug_details,
            show_node_coverage=show_node_coverage,
        )
    except ModuleNotFoundError as exc:
        return _run_block_c_adjustment_fallback(
            nodes,
            phase=phase,
            max_steps=max_steps,
            show_block_c_log=show_block_c_log,
            fallback_reason=f"missing dependency: {getattr(exc, 'name', '') or str(exc)}",
            trace_steps=trace_steps,
            show_allocator_log=show_allocator_log,
            show_debug_details=show_debug_details,
            show_node_coverage=show_node_coverage,
        )


def _chain_display(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    return str(meta.get("open_topology") or meta.get("display_topology") or "-".join(map(str, chain.composition)))


def _chain_resolved_display(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    return str(
        meta.get("resolved_topology")
        or meta.get("open_topology")
        or meta.get("display_topology")
        or "-".join(map(str, getattr(chain, "composition", []) or []))
    )


def _format_su_path(values: Sequence[Any]) -> str:
    return "-".join(str(int(x)) for x in list(values or []))


def _token_with_branch(su: int, branch: Sequence[Any]) -> str:
    vals = [int(x) for x in list(branch or [])]
    if not vals:
        return str(int(su))
    return f"{int(su)}(-{_format_su_path(vals)})"


def _token_with_branch_slots(
    su: int,
    slots: Sequence[Sequence[Any]],
    labels: Sequence[str],
) -> str:
    token = str(int(su))
    slot_vals = [list(slot or []) for slot in list(slots or [])]
    slot_labels = [str(x) for x in list(labels or [])]
    for idx, vals in enumerate(slot_vals):
        ints = [int(x) for x in vals]
        if ints:
            token += f"(-{_format_su_path(ints)})"
            continue
        label = slot_labels[idx] if idx < len(slot_labels) else ""
        if label:
            token += f"(-{label})"
    return token


def _tokens_with_branches(
    su_values: Sequence[Any],
    slot_names: Sequence[str],
    slot_branch_su: Dict[str, List[int]],
) -> List[str]:
    return [
        _token_with_branch(int(su), slot_branch_su.get(str(slot), []))
        for slot, su in zip(list(slot_names or []), list(su_values or []))
    ]


def _expanded_ring_topology(chain: Any, meta: Dict[str, Any]) -> str:
    ring_type = str(meta.get("ring_type") or getattr(chain, "chain_type", ""))
    slot_branch_su = {
        str(k): [int(x) for x in list(v or [])]
        for k, v in dict(meta.get("slot_branch_su", {}) or {}).items()
    }
    if ring_type == "side_ring":
        slots = [str(x) for x in list(meta.get("side_ring_slot_names", []) or [])]
        ring_su = [int(x) for x in list(meta.get("side_ring_su", []) or [])]
        if slots and len(slots) == len(ring_su):
            body = [
                _token_with_branch(su, slot_branch_su.get(slot, []))
                for slot, su in zip(slots, ring_su)
            ]
            return "11-" + "-".join(body) + "-11"
    if ring_type == "vertical_ring":
        slots = ["base", "right_inner_23", "right", "left_inner_23", "left", "closing_23"]
        ring_su = [int(x) for x in list(meta.get("vertical_ring_su", []) or [])]
        if len(ring_su) == len(slots):
            body = _tokens_with_branches(ring_su, slots, slot_branch_su)
            return "11-" + "-".join(body) + "-(to_first_" + str(ring_su[0]) + ")"
    if ring_type == "fused_side_ring":
        base_slots = ["base_upper", "bridge_upper", "bridge_lower", "base_lower"]
        outer_slots = ["outer_inner_upper", "outer_upper", "outer_lower", "outer_inner_lower"]
        base_su = [int(x) for x in list(meta.get("base_ring_su", []) or [])]
        outer_su = [int(x) for x in list(meta.get("outer_ring_su", []) or [])]
        if len(base_su) == len(base_slots):
            base = _tokens_with_branches(base_su, base_slots, slot_branch_su)
            if len(outer_su) == len(outer_slots):
                outer = _tokens_with_branches(outer_su, outer_slots, slot_branch_su)
                return "11-" + base[0] + "-" + base[1] + "-[" + "-".join(outer) + "]-" + base[2] + "-" + base[3] + "-11"
            return "11-" + "-".join(base) + "-11"
    return _format_su_path(getattr(chain, "composition", []) or [])


def _expanded_linear_topology(chain: Any, meta: Dict[str, Any]) -> str:
    comp = [int(x) for x in list(getattr(chain, "composition", []) or [])]
    branch_su = [int(x) for x in list(meta.get("branch_su", []) or [])]
    branch_slots_su = {
        str(k): [int(x) for x in list(v or [])]
        for k, v in dict(meta.get("branch_slots_su", {}) or {}).items()
    }
    if not comp:
        return ""
    if not branch_su and not branch_slots_su:
        return _format_su_path(comp)

    family = str(meta.get("assembly_family") or "")
    like24_idx = [
        idx for idx, su in enumerate(comp)
        if int(su) in {14, 19, 20, 21, 24, 25}
    ]
    slot_branch_by_idx: Dict[int, List[List[int]]] = {}
    slot_labels_by_idx: Dict[int, List[str]] = {}
    if branch_slots_su and like24_idx:
        label_map = {
            str(k): str(v)
            for k, v in dict(meta.get("branch_slot_labels", {}) or {}).items()
        }
        if {"left", "right"} <= set(branch_slots_su) and len(like24_idx) >= 2:
            for slot, anchor_idx in (("left", like24_idx[0]), ("right", like24_idx[1])):
                slot_branch_by_idx[int(anchor_idx)] = [
                    [int(x) for x in list(branch_slots_su.get(slot, []) or [])]
                ]
                slot_labels_by_idx[int(anchor_idx)] = [
                    str(label_map.get(slot) or ("H" if not branch_slots_su.get(slot) else ""))
                ]
        else:
            anchor_idx = next((idx for idx in like24_idx if int(comp[idx]) == 25), like24_idx[0])
            preferred_slots = [slot for slot in ("fixed22", "H") if slot in branch_slots_su]
            preferred_slots.extend(slot for slot in sorted(branch_slots_su) if slot not in preferred_slots)
            slot_branch_by_idx[int(anchor_idx)] = [
                [int(x) for x in list(branch_slots_su.get(slot, []) or [])]
                for slot in preferred_slots
            ]
            slot_labels_by_idx[int(anchor_idx)] = [
                str(label_map.get(slot) or ("H" if slot == "H" else ""))
                for slot in preferred_slots
            ]

    branch_by_idx: Dict[int, List[int]] = {}
    if slot_branch_by_idx:
        pass
    elif (
        family in {"24_AB_N_CD_BRIDGE", "24_CD_N_CD_BRIDGE"}
        and len(branch_su) == len(like24_idx)
        and all(int(su) == 22 for su in branch_su)
    ):
        for idx, su in zip(like24_idx, branch_su):
            branch_by_idx[int(idx)] = [int(su)]
    elif like24_idx:
        branch_by_idx[int(like24_idx[0])] = list(branch_su)

    tokens = [
        _token_with_branch_slots(su, slot_branch_by_idx[idx], slot_labels_by_idx.get(idx, []))
        if idx in slot_branch_by_idx
        else _token_with_branch(su, branch_by_idx.get(idx, []))
        for idx, su in enumerate(comp)
    ]
    return "-".join(tokens)


def _chain_expanded_display(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    ctype = str(getattr(chain, "chain_type", ""))
    if ctype in {"side_ring", "vertical_ring", "fused_side_ring"}:
        return _expanded_ring_topology(chain, meta)
    return _expanded_linear_topology(chain, meta)


def _chain_actual_topology_values(chain: Any) -> List[int]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    actual_topology = meta.get("actual_topology_su")
    if isinstance(actual_topology, (list, tuple)) and actual_topology:
        return [int(x) for x in list(actual_topology or [])]
    actual_used = meta.get("actual_used_su")
    if isinstance(actual_used, (list, tuple)) and actual_used:
        vals = [int(x) for x in list(actual_used or [])]
        closure = meta.get("closure_resource_su")
        if isinstance(closure, (list, tuple)):
            vals.extend(int(x) for x in list(closure or []))
        return vals
    return []


def _chain_actual_display(chain: Any) -> str:
    vals = _chain_actual_topology_values(chain)
    if vals:
        return _format_su_path(vals)
    return _chain_expanded_display(chain)


def _chain_signature_display(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    return str(
        meta.get("open_topology")
        or meta.get("resolved_topology")
        or meta.get("display_topology")
        or _chain_expanded_display(chain)
    )


def _chain_resource_counts(chain: Any) -> Dict[str, int]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    override = meta.get("resource_requirements_override")
    if isinstance(override, dict):
        return {key: int(override.get(key, 0) or 0) for key in RESOURCE_KEYS}
    return {
        key: int(_chain_required_resource_counts(chain).get(key, 0))
        for key in RESOURCE_KEYS
    }


def _chain_candidate_group_id(chain: Any) -> str:
    meta = dict(getattr(chain, "metadata", {}) or {})
    return str(meta.get("candidate_group_id") or meta.get("source_group_id") or "")


def _source_key(values: Sequence[Any]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in list(values or []) if int(x) >= 0))


def _topology_row(chain: Any, idx: int) -> Dict[str, Any]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    option = dict(meta.get("assembly_option", {}) or {})
    constraints = dict(option.get("constraints", {}) or {})
    return {
        "index": int(idx),
        "chain_type": str(getattr(chain, "chain_type", "")),
        "origin_type": str(getattr(chain, "origin_type", "")),
        "topology": _chain_resolved_display(chain),
        "actual_topology": _chain_actual_display(chain),
        "signature_topology": _chain_signature_display(chain),
        "expanded_topology": _chain_expanded_display(chain),
        "composition": [int(x) for x in list(getattr(chain, "composition", []) or [])],
        "source_ids": [int(x) for x in list(getattr(chain, "source_ids", []) or [])],
        "assembly_family": str(meta.get("assembly_family") or option.get("assembly_family") or ""),
        "usage_mode": str(meta.get("usage_mode") or option.get("usage_mode") or ""),
        "closure_plan": dict(option.get("closure_plan", {}) or {}),
        "resource_cost": _chain_resource_counts(chain),
        "ring_type": str(meta.get("ring_type") or constraints.get("ring_type") or ""),
        "branch_tail_lengths": dict(meta.get("branch_tail_lengths", {}) or {}),
    }


def _closed_topology_chains(result: Any) -> List[Any]:
    all_closed = (
        list(getattr(result, "bridge_chains", []) or [])
        + list(getattr(result, "side_chains", []) or [])
        + list(getattr(result, "branch_chains", []) or [])
    )
    out: List[Any] = []
    seen: set[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = set()
    for chain in all_closed:
        key = (
            str(getattr(chain, "chain_type", "")),
            tuple(int(x) for x in list(getattr(chain, "composition", []) or [])),
            _source_key(getattr(chain, "source_ids", []) or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(chain)
    return out


def _positive_counter_delta(left: Counter[int], right: Counter[int]) -> Counter[int]:
    keys = set(int(k) for k in left) | set(int(k) for k in right)
    return Counter({
        int(k): diff
        for k in keys
        for diff in [max(0, int(left.get(k, 0)) - int(right.get(k, 0)))]
        if int(diff) > 0
    })


def _without_ignored_topology_su(counter: Counter[int]) -> Counter[int]:
    return Counter({
        int(k): int(v)
        for k, v in counter.items()
        if int(k) not in IGNORED_TOPOLOGY_SU
    })


def _slot_branch_su(meta: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    slot_branch_su = dict(meta.get("slot_branch_su", {}) or {})
    for slot in sorted(slot_branch_su):
        out.extend(int(x) for x in list(slot_branch_su.get(slot, []) or []))
    return out


def _chain_topology_su_values(chain: Any) -> List[int]:
    meta = dict(getattr(chain, "metadata", {}) or {})
    ctype = str(getattr(chain, "chain_type", ""))
    actual_vals = _chain_actual_topology_values(chain)
    if actual_vals:
        return list(actual_vals)
    comp = [int(x) for x in list(getattr(chain, "composition", []) or [])]
    if ctype == "side_ring":
        ring_su = [int(x) for x in list(meta.get("side_ring_su", []) or [])]
        if ring_su:
            return [11] + list(ring_su) + _slot_branch_su(meta) + [11]
        return [11] + list(comp) + [11]
    if ctype == "vertical_ring":
        ring_su = [int(x) for x in list(meta.get("vertical_ring_su", []) or [])]
        if ring_su:
            return [11] + list(ring_su) + _slot_branch_su(meta)
        return list(comp)
    if ctype == "fused_side_ring":
        base_su = [int(x) for x in list(meta.get("base_ring_su", []) or [])]
        outer_su = [int(x) for x in list(meta.get("outer_ring_su", []) or [])]
        if base_su or outer_su:
            return [11] + list(base_su) + [11] + list(outer_su) + _slot_branch_su(meta)
        return [11] + list(comp) + [11]
    branch_su = [int(x) for x in list(meta.get("branch_su", []) or [])]
    return list(comp) + list(branch_su)


def _topology_su_counts(chains: Sequence[Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for chain in chains:
        counts.update(_chain_topology_su_values(chain))
    return counts


def _chain_source_ids(chains: Sequence[Any]) -> set[int]:
    out: set[int] = set()
    for chain in list(chains or []):
        for gid in list(getattr(chain, "source_ids", []) or []):
            if int(gid) >= 0:
                out.add(int(gid))
    return out


def _metadata_abstracted_source_ids(meta: Dict[str, Any]) -> set[int]:
    out: set[int] = set()

    def add_rows(rows: Sequence[Any]) -> None:
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            try:
                gid = int(row.get("actual_gid", -1))
            except Exception:
                continue
            if gid >= 0:
                out.add(gid)

    add_rows(list(meta.get("abstracted_neighbors", []) or []))
    for gid in list(meta.get("evidence_ids", []) or []):
        try:
            gid_int = int(gid)
        except Exception:
            continue
        if gid_int >= 0:
            out.add(gid_int)
    for row in list(meta.get("branch_candidate_alternatives", []) or []):
        if not isinstance(row, dict):
            continue
        add_rows(list(row.get("abstracted_neighbors", []) or []))
        for gid in list(row.get("evidence_ids", []) or []):
            try:
                gid_int = int(gid)
            except Exception:
                continue
            if gid_int >= 0:
                out.add(gid_int)
    return out


def _chain_abstracted_source_ids(chains: Sequence[Any]) -> set[int]:
    out: set[int] = set()
    for chain in list(chains or []):
        out.update(_metadata_abstracted_source_ids(dict(getattr(chain, "metadata", {}) or {})))
    return out


def _abstracted_topology_su_counts(
    chains: Sequence[Any],
    nodes: Sequence[Any],
) -> Counter[int]:
    node_by_gid = {int(getattr(node, "global_id", -1)): node for node in nodes}
    counts: Counter[int] = Counter()
    for gid in _chain_abstracted_source_ids(chains):
        node = node_by_gid.get(int(gid))
        if node is None:
            continue
        su = int(getattr(node, "su_type", -1))
        if su not in IGNORED_TOPOLOGY_SU:
            counts[su] += 1
    return counts


def _source_usage_by_gid(result: Any) -> Dict[int, List[str]]:
    usage: Dict[int, List[str]] = defaultdict(list)
    chain_groups = [
        ("closed", _closed_topology_chains(result)),
        ("pending", _default_unclosed_topology_chains(result)),
    ]
    for group_name, chains in chain_groups:
        for idx, chain in enumerate(chains):
            label = f"{group_name}#{idx}:{getattr(chain, 'origin_type', '')}"
            for gid in list(getattr(chain, "source_ids", []) or []):
                if int(gid) >= 0:
                    usage[int(gid)].append(label)
            for gid in _metadata_abstracted_source_ids(dict(getattr(chain, "metadata", {}) or {})):
                usage[int(gid)].append(f"{label}:abstracted")
    return usage


def _unrecognized_topology_nodes(
    nodes: Sequence[Any],
    covered_ids: set[int],
    target_counts: Counter[int],
) -> List[Tuple[Any, str]]:
    remaining = Counter({int(k): int(v) for k, v in target_counts.items() if int(v) > 0})
    out: List[Tuple[Any, str]] = []
    for node in sorted(nodes, key=lambda n: int(getattr(n, "global_id", -1))):
        gid = int(getattr(node, "global_id", -1))
        su = int(getattr(node, "su_type", -1))
        if gid in covered_ids or su in IGNORED_TOPOLOGY_SU:
            continue
        if int(remaining.get(su, 0)) <= 0:
            continue
        out.append((node, "not_covered_by_output_topology"))
        remaining[su] -= 1
    for node in sorted(nodes, key=lambda n: int(getattr(n, "global_id", -1))):
        gid = int(getattr(node, "global_id", -1))
        su = int(getattr(node, "su_type", -1))
        if gid not in covered_ids or su in IGNORED_TOPOLOGY_SU:
            continue
        if int(remaining.get(su, 0)) <= 0:
            continue
        out.append((node, "covered_but_not_counted_in_balance"))
        remaining[su] -= 1
    return out


def _su_matrix_value_width(counters: Sequence[Counter[int]]) -> int:
    width = 1
    for counter in counters:
        for su in SU_MATRIX_VALUES:
            value = int(counter.get(int(su), 0))
            width = max(width, len(str(abs(value))))
    return max(3, int(width))


def _print_su_counter_matrix(title: str, counter: Counter[int], value_width: int) -> None:
    print(f"{title}:")
    for start in range(0, len(SU_MATRIX_VALUES), SU_MATRIX_COLS):
        row = SU_MATRIX_VALUES[start:start + SU_MATRIX_COLS]
        cells = [
            f"SU{int(su):02d}:{int(counter.get(int(su), 0)):>{int(value_width)}d}"
            for su in row
        ]
        print("  " + "  ".join(cells))


def _remaining_after_closed_topology_counts(
    result: Any,
    nodes: Sequence[Any],
) -> Counter[int]:
    input_counts = Counter(int(getattr(node, "su_type", -1)) for node in nodes)
    closed_counts = _topology_su_counts(_closed_topology_chains(result))
    return _without_ignored_topology_su(_positive_counter_delta(input_counts, closed_counts))


def _unrecognized_topology_target_counts(
    result: Any,
    nodes: Sequence[Any],
) -> Counter[int]:
    remaining_after_closed = _remaining_after_closed_topology_counts(result, nodes)
    known_unclosed_counts = _without_ignored_topology_su(
        _topology_su_counts(_default_unclosed_topology_chains(result))
    )
    abstracted_counts = _without_ignored_topology_su(
        _abstracted_topology_su_counts(
            _closed_topology_chains(result) + _default_unclosed_topology_chains(result),
            nodes,
        )
    )
    known_counts = Counter(known_unclosed_counts)
    known_counts.update(abstracted_counts)
    return _positive_counter_delta(remaining_after_closed, known_counts)


def _unclosed_reason_for_chain(
    chain: Any,
    lookup: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]],
) -> str:
    gid = _chain_candidate_group_id(chain)
    source_ids = _source_key(getattr(chain, "source_ids", []) or [])
    reason_row = lookup.get((gid, source_ids)) or lookup.get(("", source_ids)) or {}
    return str(reason_row.get("reason", "unknown"))


def _default_unclosed_topology_chains(result: Any) -> List[Any]:
    lookup = _unclosed_reason_lookup(result)
    alternative_reasons = {
        "candidate_group_already_selected",
        "exclusive_group_conflict",
        "source_id_conflict_with_ordinary_chain",
        "source_id_conflict_with_selected_option",
    }
    out: List[Any] = []
    seen: set[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = set()
    for chain in list(getattr(result, "pending_closure_topologies", []) or []):
        if _unclosed_reason_for_chain(chain, lookup) in alternative_reasons:
            continue
        key = (
            str(getattr(chain, "chain_type", "")),
            tuple(int(x) for x in list(getattr(chain, "composition", []) or [])),
            _source_key(getattr(chain, "source_ids", []) or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(chain)
    return out


def _print_topology_balance(result: Any, nodes: Sequence[Any]) -> None:
    closed_chains = _closed_topology_chains(result)
    input_counts = Counter(int(getattr(node, "su_type", -1)) for node in nodes)
    closed_counts = _topology_su_counts(closed_chains)
    remaining_after_closed = _remaining_after_closed_topology_counts(result, nodes)
    unclosed_counts = Counter(remaining_after_closed)
    balance_counters = [
        input_counts,
        closed_counts,
        unclosed_counts,
        remaining_after_closed,
    ]
    value_width = _su_matrix_value_width(balance_counters)
    print("========== Topology SU Balance ==========")
    print("matrix_shape: 3x11 (SU00-SU32; missing=0)")
    print(f"ignored_topology_su: {_json(sorted(IGNORED_TOPOLOGY_SU))}")
    _print_su_counter_matrix("input_su_counts", input_counts, value_width)
    _print_su_counter_matrix("closed_topology_su_counts", closed_counts, value_width)
    _print_su_counter_matrix("unclosed_topology_su_counts", unclosed_counts, value_width)
    _print_su_counter_matrix("input_remaining_after_closed_topologies_by_su", remaining_after_closed, value_width)
    print("")


def _print_flexible_chain_range(result: Any, nodes: Sequence[Any]) -> None:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    input_counts = Counter(int(getattr(node, "su_type", -1)) for node in nodes)
    flex_lower = int(diag.get("flex_lower", 0) or 0)
    flex_upper = int(diag.get("flex_upper", 0) or 0)
    flex_count = int(diag.get("flex_count", 0) or 0)
    su23_input = int(input_counts.get(23, 0))
    su23_consumed = int(getattr(result, "consumed_23", 0) or 0)
    su23_remaining = int(getattr(result, "remaining_23", 0) or 0)
    su23_shortage = max(0, int(su23_consumed) - int(su23_input))
    print("========== Flexible Chain Range ==========")
    print(
        "flex_window: "
        f"[{flex_lower},{flex_upper}] "
        f"ok={bool(diag.get('flex_window_ok', True))} "
        f"source={str(diag.get('flex_window_source', ''))}"
    )
    print(
        "flex_topology_count: "
        f"{flex_count} "
        f"status={'in_range' if flex_lower <= flex_count <= flex_upper else 'out_of_range'}"
    )
    print(
        "flex_count_detail: "
        f"explicit={int(diag.get('explicit_flex_count', 0) or 0)} "
        f"generated={int(diag.get('generated_flex_count', 0) or 0)} "
        f"short={int(diag.get('flex_short', 0) or 0)} "
        f"excess={int(diag.get('flex_excess', 0) or 0)}"
    )
    print(
        "su23_resource_check: "
        f"input={su23_input} "
        f"consumed={su23_consumed} "
        f"remaining={su23_remaining} "
        f"shortage={su23_shortage} "
        f"status={'shortage' if su23_shortage > 0 else 'ok'}"
    )
    print("")


def _print_compact_topology_group(title: str, chains: Sequence[Any], *, show_reason: bool = False, result: Any = None) -> None:
    print(f"========== {title} ==========")
    print(f"count: {len(chains)}")
    if not chains:
        print("(none)")
        print("")
        return
    lookup = _unclosed_reason_lookup(result) if show_reason and result is not None else {}
    for idx, chain in enumerate(chains):
        topology = _chain_display(chain) if str(getattr(chain, "chain_type", "")) == "pending_closure" else _chain_expanded_display(chain)
        actual = _chain_actual_display(chain)
        line = f"[{idx}] origin={getattr(chain, 'origin_type', '')} topology={topology} actual={actual}"
        if show_reason:
            line += f" reason={_unclosed_reason_for_chain(chain, lookup)}"
        print(line)
    print("")


def _print_compact_topologies(result: Any) -> None:
    groups = _assembled_topology_groups(result)
    _print_compact_topology_group(
        "Flexible Connection Topologies (closed)",
        groups["flex"],
    )
    _print_compact_topology_group(
        "Side Connection Topologies (closed)",
        groups["side"],
    )
    _print_compact_topology_group(
        "Aliphatic Ring Topologies (closed)",
        groups["ring"],
    )
    if groups["other"]:
        _print_compact_topology_group("Other Closed Topologies", groups["other"])
    _print_compact_topology_group(
        "Unclosed / Incomplete Topologies",
        _default_unclosed_topology_chains(result),
        show_reason=True,
        result=result,
    )
    print("")


def _print_unrecognized_topology_nodes_detail(result: Any, nodes: Sequence[Any]) -> None:
    target_counts = _unrecognized_topology_target_counts(result, nodes)
    covered_ids = _chain_source_ids(
        _closed_topology_chains(result) + _default_unclosed_topology_chains(result)
    )
    usage_by_gid = _source_usage_by_gid(result)
    missing_nodes = _unrecognized_topology_nodes(nodes, covered_ids, target_counts)
    node_by_gid = {int(getattr(node, "global_id", -1)): node for node in nodes}
    print("========== Unrecognized / Not Listed Topology Nodes ==========")
    print(f"count: {len(missing_nodes)}")
    if not missing_nodes:
        print("(none)")
        print("")
        return
    value_width = _su_matrix_value_width([
        Counter(int(getattr(node, "su_type", -1)) for node, _status in missing_nodes)
    ])
    _print_su_counter_matrix(
        "unrecognized_topology_node_su_counts",
        Counter(int(getattr(node, "su_type", -1)) for node, _status in missing_nodes),
        value_width,
    )
    for node, status in missing_nodes:
        state = state_for_node(node)
        hop1_ids = [int(x) for x in list(getattr(node, "hop1_ids", ()) or ())]
        hop1_su = [
            int(getattr(node_by_gid[int(gid)], "su_type", -1))
            if int(gid) in node_by_gid else "?"
            for gid in hop1_ids
        ]
        hop1_gid_su = [
            f"{int(gid)}:{int(getattr(node_by_gid[int(gid)], 'su_type', -1))}"
            if int(gid) in node_by_gid else f"{int(gid)}:?"
            for gid in hop1_ids
        ]
        hop1_usage = [
            f"{int(gid)}:{';'.join(usage_by_gid.get(int(gid), ['unused']))}"
            for gid in hop1_ids
        ]
        display_status = status
        if status == "not_covered_by_output_topology" and any(
            int(gid) in usage_by_gid for gid in hop1_ids
        ):
            display_status = "not_covered_but_neighbor_used_by_output_topology"
        print(
            f"gid={int(getattr(node, 'global_id', -1))} "
            f"su={int(getattr(node, 'su_type', -1))} "
            f"label={state.label} "
            f"status={display_status} "
            f"degree={getattr(node, 'target_degree', None)} "
            f"fixed={getattr(node, 'target_fixed_anchor_count', 0)} "
            f"hop1_su={hop1_su} "
            f"hop1_gid_su={hop1_gid_su} "
            f"hop1_usage={hop1_usage}"
        )
    print("")


def _print_topology_group(title: str, chains: Sequence[Any]) -> None:
    print(f"========== {title} ==========")
    print(f"count: {len(chains)}")
    if not chains:
        print("(none)")
        print("")
        return
    for idx, chain in enumerate(chains):
        row = _topology_row(chain, idx)
        print(f"[{idx}] type={row['chain_type']} origin={row['origin_type']}")
        print(f"    topology={row['topology']}")
        print(f"    actual_topology={row['actual_topology']}")
        print(f"    signature_topology={row['signature_topology']}")
        print(f"    expanded_topology={row['expanded_topology']}")
        print(f"    composition={row['composition']}")
        print(f"    source_ids={row['source_ids']}")
        print(f"    assembly_family={row['assembly_family']} usage_mode={row['usage_mode']}")
        if row["closure_plan"]:
            print(f"    closure_plan={_json(row['closure_plan'])}")
        if row["ring_type"]:
            print(f"    ring_type={row['ring_type']} branch_tail_lengths={_json(row['branch_tail_lengths'])}")
        print(f"    resource_cost={_json(row['resource_cost'])}")
    print("")


def _assembled_topology_groups(result: Any) -> Dict[str, List[Any]]:
    groups = {
        "flex": [],
        "side": [],
        "ring": [],
        "other": [],
    }
    for chain in _closed_topology_chains(result):
        ctype = str(getattr(chain, "chain_type", ""))
        if ctype in FLEX_TOPOLOGY_TYPES:
            groups["flex"].append(chain)
        elif ctype in SIDE_TOPOLOGY_TYPES:
            groups["side"].append(chain)
        elif ctype in RING_TOPOLOGY_TYPES:
            groups["ring"].append(chain)
        else:
            groups["other"].append(chain)
    return groups


def _print_assembled_topologies(result: Any) -> None:
    groups = _assembled_topology_groups(result)
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    print("========== Assembled Closed Topology Overview ==========")
    print(f"flexible_topology_count: {len(groups['flex'])}")
    print(f"side_topology_count: {len(groups['side'])}")
    print(f"aliphatic_ring_topology_count: {len(groups['ring'])}")
    print(f"other_closed_topology_count: {len(groups['other'])}")
    print(f"flex_scenario: {str(diag.get('flex_scenario', ''))}")
    print(f"flex_window_source: {str(diag.get('flex_window_source', ''))}")
    if str(diag.get("flex_window_error", "")):
        print(f"flex_window_error: {str(diag.get('flex_window_error', ''))}")
    print(f"flex_window_ok: {bool(diag.get('flex_window_ok', True))}")
    print(f"X/Y/Z: {int(diag.get('X', 0) or 0)}/{int(diag.get('Y', 0) or 0)}/{int(diag.get('Z', 0) or 0)}")
    print(
        "flex_range_raw/effective: "
        f"[{int(diag.get('flex_lower_raw', 0) or 0)},{int(diag.get('flex_upper_raw', 0) or 0)}] / "
        f"[{int(diag.get('flex_lower', 0) or 0)},{int(diag.get('flex_upper', 0) or 0)}]"
    )
    print(f"fixed_flex_count: {int(diag.get('fixed_flex_count', 0) or 0)}")
    print(f"explicit_flex_count: {int(diag.get('explicit_flex_count', 0) or 0)}")
    print(f"ordinary_explicit_flex_count: {int(diag.get('ordinary_explicit_flex_count', 0) or 0)}")
    print(f"mandatory_branch_flex_count: {int(diag.get('mandatory_branch_flex_count', 0) or 0)}")
    print(f"generated_flex_count: {int(diag.get('generated_flex_count', 0) or 0)}")
    print(f"prefill_flex_count: {int(diag.get('prefill_flex_count', 0) or 0)}")
    print(f"final_repair_flex_count: {int(diag.get('final_repair_flex_count', 0) or 0)}")
    print(f"extra_backbone_flex_count: {int(diag.get('extra_backbone_flex_count', 0) or 0)}")
    print(f"flex_prefill_target: {int(diag.get('flex_prefill_target', 0) or 0)}")
    print(f"extra_backbone_topology_count: {int(diag.get('extra_backbone_topology_count', 0) or 0)}")
    print(f"ring_assembly_enabled: {bool(diag.get('ring_assembly_enabled', False))}")
    print(f"ring_pair_option_count: {int(diag.get('ring_pair_option_count', 0) or 0)}")
    print(f"fused_ring_planner_option_count: {int(diag.get('fused_ring_planner_option_count', 0) or 0)}")
    print(f"side_ring_planner_option_count: {int(diag.get('side_ring_planner_option_count', 0) or 0)}")
    print(f"vertical_ring_planner_option_count: {int(diag.get('vertical_ring_planner_option_count', 0) or 0)}")
    print(f"selected_ring_plan_counts_by_type: {dict(diag.get('selected_ring_plan_counts_by_type', {}) or {})}")
    print(f"selected_ring_planner_counts: {dict(diag.get('selected_ring_planner_counts', {}) or {})}")
    print(f"bridgehead_14_replacement_count: {int(diag.get('bridgehead_14_replacement_count', 0) or 0)}")
    print(f"like24_used_by_ring: {int(diag.get('like24_used_by_ring', 0) or 0)}")
    print(f"raw_su24_filler_used: {int(diag.get('raw_su24_filler_used', 0) or 0)}")
    print("")
    _print_topology_group(
        "Flexible Connection Topologies (bridge + branch_bridge)",
        groups["flex"],
    )
    _print_topology_group(
        "Side Connection Topologies (side + branch_side)",
        groups["side"],
    )
    _print_topology_group(
        "Aliphatic Ring Topologies (side_ring + vertical_ring + fused_side_ring)",
        groups["ring"],
    )
    if groups["other"]:
        _print_topology_group("Other Closed Topologies", groups["other"])


def _unclosed_reason_lookup(result: Any) -> Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]]:
    lookup: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    for row in list(getattr(result, "topology_unclosed_candidates", []) or []):
        if not isinstance(row, dict):
            continue
        gid = str(row.get("candidate_group_id") or "")
        source_ids = _source_key(row.get("source_ids", []) or [])
        lookup[(gid, source_ids)] = dict(row)
        lookup[("", source_ids)] = dict(row)
    return lookup


def _print_unclosed_and_incomplete_topologies(result: Any) -> None:
    pending = list(getattr(result, "pending_closure_topologies", []) or [])
    lookup = _unclosed_reason_lookup(result)
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    rejected = list(diag.get("rejected_options", []) or [])
    print("========== Unclosed / Incomplete Topologies ==========")
    print(f"pending_closure_count: {len(pending)}")
    print(f"topology_unclosed_candidate_count: {len(getattr(result, 'topology_unclosed_candidates', []) or [])}")
    print(f"rejected_option_count: {int(diag.get('rejected_option_count', len(rejected)) or 0)}")
    if not pending:
        print("(none)")
    for idx, chain in enumerate(pending):
        gid = _chain_candidate_group_id(chain)
        source_ids = _source_key(getattr(chain, "source_ids", []) or [])
        reason_row = lookup.get((gid, source_ids)) or lookup.get(("", source_ids)) or {}
        meta = dict(getattr(chain, "metadata", {}) or {})
        print(f"[{idx}] type={getattr(chain, 'chain_type', '')} origin={getattr(chain, 'origin_type', '')}")
        print(f"    topology={_chain_display(chain)}")
        print(f"    actual_topology={_chain_actual_display(chain)}")
        print(f"    signature_topology={_chain_signature_display(chain)}")
        print(f"    composition={list(getattr(chain, 'composition', []) or [])}")
        print(f"    source_ids={list(getattr(chain, 'source_ids', []) or [])}")
        print(f"    candidate_group_id={gid or reason_row.get('candidate_group_id', '')}")
        print(f"    reason={reason_row.get('reason', 'unknown')}")
        print(f"    closure_intents={_json(meta.get('closure_intents', {}))}")
        families = reason_row.get("assembly_families")
        if families:
            print(f"    assembly_families={_json(families)}")
    print("")


def _print_backbone_resource_balance(result: Any) -> None:
    diag = dict(getattr(result, "topology_assembly_diagnostics", {}) or {})
    ledger = dict(diag.get("resource_ledger", {}) or {})
    print("========== Backbone Resource Balance ==========")
    allocator_totals = {
        "11": int(getattr(result, "total_11", 0)),
        "22": int(getattr(result, "total_22", 0)),
        "23": int(getattr(result, "total_23", 0)),
        "24": int(getattr(result, "total_24", 0)),
        "25": int(getattr(result, "total_25", 0)),
    }
    allocator_consumed = {
        "11": int(getattr(result, "consumed_11", 0)),
        "22": int(getattr(result, "consumed_22", 0)),
        "23": int(getattr(result, "consumed_23", 0)),
        "24": int(getattr(result, "consumed_24", 0)),
        "25": int(getattr(result, "consumed_25", 0)),
    }
    allocator_remaining = {
        "11": int(getattr(result, "remaining_11", 0)),
        "22": int(getattr(result, "remaining_22", 0)),
        "23": int(getattr(result, "remaining_23", 0)),
        "24": int(getattr(result, "remaining_24", 0)),
        "25": int(getattr(result, "remaining_25", 0)),
    }
    print(f"allocator_total_by_kind: {_json(allocator_totals)}")
    print(f"allocator_consumed_by_kind: {_json(allocator_consumed)}")
    print(f"allocator_remaining_by_kind: {_json(allocator_remaining)}")
    if ledger:
        print(f"assembly_total_by_kind: {_json({key: int(dict(ledger.get('total', {}) or {}).get(key, 0)) for key in RESOURCE_KEYS})}")
        print(f"assembly_consumed_by_kind: {_json({key: int(dict(ledger.get('consumed', {}) or {}).get(key, 0)) for key in RESOURCE_KEYS})}")
        print(f"assembly_remaining_by_kind: {_json({key: int(dict(ledger.get('remaining', {}) or {}).get(key, 0)) for key in RESOURCE_KEYS})}")
        print(f"assembly_shortage_by_kind: {_json({key: int(dict(ledger.get('shortage', {}) or {}).get(key, 0)) for key in RESOURCE_KEYS})}")
    print(f"backbone_residual: {int(diag.get('backbone_residual', 0) or 0)}")
    print(f"backbone_residual_ok: {bool(diag.get('backbone_residual_ok', True))}")
    print(f"recommended_adjustments: {_json(list(diag.get('recommended_adjustments', []) or []))}")
    print("")


def _collect_assignments(result: Any) -> Dict[int, List[str]]:
    assignments: Dict[int, List[str]] = defaultdict(list)
    for row in list(getattr(result, "catalog_fragment_rows", []) or []):
        label = f"fragment:{row.get('kind')}:{row.get('origin_rule')}"
        for gid in list(row.get("node_ids", []) or []):
            if int(gid) >= 0:
                assignments[int(gid)].append(label)
    all_chains = (
        list(getattr(result, "bridge_chains", []) or [])
        + list(getattr(result, "side_chains", []) or [])
        + list(getattr(result, "branch_chains", []) or [])
        + list(getattr(result, "pending_closure_topologies", []) or [])
    )
    for chain in all_chains:
        label = f"chain:{chain.origin_type}"
        for gid in list(getattr(chain, "source_ids", []) or []):
            if int(gid) >= 0:
                assignments[int(gid)].append(label)
    return assignments


def _is_base_backbone_resource(node: Any) -> bool:
    return int(getattr(node, "su_type", -1)) in BASE_BACKBONE_RESOURCE_SU


def _print_summary(result: Any, nodes: Sequence[Any]) -> None:
    su_counts = Counter(int(getattr(n, "su_type", -1)) for n in nodes)
    base_counts = Counter(int(getattr(n, "su_type", -1)) for n in nodes if _is_base_backbone_resource(n))
    required_counts = Counter(int(getattr(n, "su_type", -1)) for n in nodes if not _is_base_backbone_resource(n))
    pending_origin_counts = Counter(str(ch.origin_type) for ch in result.pending_closure_topologies)
    closed_origin_counts = Counter(
        str(ch.origin_type)
        for ch in list(result.bridge_chains) + list(result.side_chains) + list(result.branch_chains)
    )
    branch_alt_counts: Counter[str] = Counter()
    for chain in result.pending_closure_topologies:
        for alt in list(chain.metadata.get("branch_candidate_alternatives", []) or []):
            branch_alt_counts[str(alt.get("abcd_type"))] += 1

    print("========== Topology Candidate Dump ==========")
    print(f"nodes: {len(nodes)}")
    print(f"su_counts: {_json(dict(sorted(su_counts.items())))}")
    print(f"base_backbone_resource_su: {_json(sorted(BASE_BACKBONE_RESOURCE_SU))}")
    print(f"base_backbone_resource_counts: {_json(dict(sorted(base_counts.items())))}")
    print(f"classification_required_counts: {_json(dict(sorted(required_counts.items())))}")
    print(f"catalog_summary: {_json(dict(result.catalog_summary))}")
    print(f"closed_bridge_count: {len(result.bridge_chains)}")
    print(f"closed_side_count: {len(result.side_chains)}")
    print(f"closed_origin_counts: {_json(dict(closed_origin_counts))}")
    print(f"pending_count: {len(result.pending_closure_topologies)}")
    print(f"pending_origin_counts: {_json(dict(pending_origin_counts))}")
    print(f"branch_alternative_type_counts: {_json(dict(branch_alt_counts))}")
    print("")


def _print_base_backbone_resources(nodes: Sequence[Any]) -> None:
    print("========== Base Backbone Resources ==========")
    grouped: Dict[int, List[int]] = defaultdict(list)
    for node in nodes:
        if _is_base_backbone_resource(node):
            grouped[int(node.su_type)].append(int(node.global_id))
    if not grouped:
        print("(none)")
        print("")
        return
    for su in sorted(grouped):
        ids = sorted(grouped[su])
        print(f"su={su} count={len(ids)} global_ids={ids}")
    print("")


def _print_unclassified_required_nodes(nodes: Sequence[Any], assignments: Dict[int, List[str]]) -> None:
    print("========== Unclassified Required Nodes ==========")
    missing = [
        node
        for node in nodes
        if not _is_base_backbone_resource(node) and int(node.global_id) not in assignments
    ]
    missing_counts = Counter(int(node.su_type) for node in missing)
    print(f"unclassified_required_count: {len(missing)}")
    print(f"unclassified_required_counts: {_json(dict(sorted(missing_counts.items())))}")
    if not missing:
        print("(none)")
        return
    for node in sorted(missing, key=lambda n: (int(n.su_type), int(n.global_id))):
        state = state_for_node(node)
        print(
            f"gid={int(node.global_id)} "
            f"su={int(node.su_type)} "
            f"label={state.label} "
            f"name={getattr(node, 'su_name', '') or ''} "
            f"degree={getattr(node, 'target_degree', None)} "
            f"fixed={getattr(node, 'target_fixed_anchor_count', 0)} "
            f"mode={getattr(node, 'anchor_mode', None)} "
            f"hop1_ids={list(getattr(node, 'hop1_ids', ()) or ())} "
            f"hop1_ms={list(getattr(node, 'hop1_ms', ()) or ())}"
        )


def _print_node_assignments(nodes: Sequence[Any], assignments: Dict[int, List[str]]) -> None:
    print("========== Node Coverage ==========")
    for node in sorted(nodes, key=lambda n: int(n.global_id)):
        state = state_for_node(node)
        labels = assignments.get(int(node.global_id), [])
        if labels:
            status = "recognized"
        elif _is_base_backbone_resource(node):
            status = "base_backbone_resource"
        else:
            status = "unclassified_required"
        print(
            f"gid={int(node.global_id)} su={int(node.su_type)} label={state.label} "
            f"status={status} assignments={_json(labels)}"
        )


def _block_c_record_summary_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "op": str(rec.get("op", "")),
        "stage": str(rec.get("stage", "")),
    }
    for key in (
        "backend",
        "phase",
        "source_op",
        "requested_count",
        "applied_changes",
        "reason",
        "loop_step",
        "accepted",
        "accepted_policy",
        "layer2_r2",
    ):
        if key in rec:
            row[str(key)] = rec.get(key)
    if "layer12" in rec:
        layer12 = dict(rec.get("layer12", {}) or {})
        row["layer12"] = {
            "ok": bool(layer12.get("ok", False)),
            "mode": str(layer12.get("mode", "")),
            "reason": str(layer12.get("reason", "")),
            "error": str(layer12.get("error", "")),
            "r2": layer12.get("r2"),
        }
    if "topology_after" in rec:
        topo = dict(rec.get("topology_after", {}) or {})
        shortage = dict(topo.get("resource_shortage", {}) or {})
        row["topology_after"] = {
            "ok": bool(topo.get("ok", False)),
            "assembly_ok": bool(topo.get("topology_assembly_ok", False)),
            "closed": int(topo.get("closed_topology_count", 0) or 0),
            "pending": int(
                topo.get(
                    "pending_closure_total_count",
                    topo.get("pending_closure_count", 0),
                )
                or 0
            ),
            "unallocated": int(topo.get("unallocated_branch", 0) or 0),
            "req11": int(shortage.get("11", topo.get("req_11", 0)) or 0),
            "req22": int(shortage.get("22", topo.get("req_22", 0)) or 0),
            "req23": int(shortage.get("23", topo.get("req_23", 0)) or 0),
            "req24": int(shortage.get("24", topo.get("req_24", 0)) or 0),
            "req25": int(shortage.get("25", topo.get("req_25", 0)) or 0),
            "unrecognized": int(topo.get("unrecognized_topology_node_count", 0) or 0),
        }
    if "delta_counts" in rec:
        row["delta_counts"] = dict(rec.get("delta_counts", {}) or {})
    if "node_conversions" in rec:
        row["node_conversions"] = list(rec.get("node_conversions", []) or [])
    if "penalty_before" in rec:
        row["penalty_before"] = list(rec.get("penalty_before", []) or [])
    if "penalty_after" in rec:
        row["penalty_after"] = list(rec.get("penalty_after", []) or [])
    if "diagnostic_before" in rec:
        diag = dict(rec.get("diagnostic_before", {}) or {})
        row["diagnostic_before"] = {
            "req11": int(diag.get("required_extra_11", diag.get("req_11", 0)) or 0),
            "req22": int(diag.get("required_extra_22", diag.get("req_22", 0)) or 0),
            "req23": int(diag.get("required_extra_23", diag.get("req_23", 0)) or 0),
            "pending": int(
                diag.get(
                    "pending_closure_total_count",
                    diag.get("pending_closure_count", 0),
                )
                or 0
            ),
        }
    return row


def _print_block_c_adjustment_summary(
    moves: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    hist_before: Any,
    hist_after: Any,
) -> None:
    print("========== Block C Structure Adjustment ==========")
    if str(meta.get("backend", "") or ""):
        print(f"backend: {str(meta.get('backend', ''))}")
    if str(meta.get("fallback_reason", "") or ""):
        print(f"fallback_reason: {str(meta.get('fallback_reason', ''))}")
    print(f"ok: {bool(meta.get('ok', False))}")
    print(f"phase: {str(meta.get('phase', ''))}")
    print(f"n_moves: {int(meta.get('n_moves', len(list(moves or []))) or 0)}")
    print(f"branch_ok: {bool(meta.get('branch_ok', False))}")
    print(f"extra_ok: {bool(meta.get('extra_ok', False))}")
    if bool(meta.get("partial_progress", False)):
        print("partial_progress: True")
    print(f"final_scenario: {str(meta.get('final_scenario', ''))}")
    if meta.get("initial_penalty") is not None or meta.get("best_penalty") is not None:
        print(f"initial_penalty: {_json(list(meta.get('initial_penalty', []) or []))}")
        print(f"best_penalty: {_json(list(meta.get('best_penalty', []) or []))}")
        print(f"final_penalty: {_json(list(meta.get('final_penalty', []) or []))}")
    if str(meta.get("stop_reason", "") or ""):
        print(f"stop_reason: {str(meta.get('stop_reason', ''))}")
    if bool(meta.get("returned_best_state", False)):
        print("returned_best_state: True")
    print(f"hist_delta_by_su: {_json(_hist_delta(hist_before, hist_after))}")
    branch_meta = dict(meta.get("branch_meta", {}) or {})
    extra_meta = dict(meta.get("extra_meta", {}) or {})
    if branch_meta:
        print(
            "branch_meta: "
            + _json({
                "ok": bool(branch_meta.get("ok", False)),
                "n_moves": int(branch_meta.get("n_moves", 0) or 0),
                "final_scenario": str(branch_meta.get("final_scenario", "")),
                "partial_progress": bool(branch_meta.get("partial_progress", False)),
            })
        )
        if bool(branch_meta.get("returned_original_h", False)):
            print("returned_original_h: True")
        if int(branch_meta.get("n_rejected_moves", 0) or 0) > 0:
            print(f"n_rejected_moves: {int(branch_meta.get('n_rejected_moves', 0) or 0)}")
    if extra_meta:
        print(
            "extra_meta: "
            + _json({
                "ok": bool(extra_meta.get("ok", False)),
                "n_moves": int(extra_meta.get("n_moves", 0) or 0),
                "final_scenario": str(extra_meta.get("final_scenario", "")),
                "reason": str(extra_meta.get("reason", "")),
            })
        )
    records = list(moves or [])
    print(f"records_count: {len(records)}")
    for idx, rec in enumerate(records[:20]):
        row = _block_c_record_summary_row(dict(rec or {}))
        print(f"[{idx}] {_json(row)}")
    if len(records) > 20:
        print(f"... {len(records) - 20} more records")
    rejected_records = list(branch_meta.get("rejected_records", []) or [])
    if rejected_records:
        print(f"rejected_records_count: {len(rejected_records)}")
        for idx, rec in enumerate(rejected_records[:20]):
            row = _block_c_record_summary_row(dict(rec or {}))
            print(f"[rejected {idx}] {_json(row)}")
        if len(rejected_records) > 20:
            print(f"... {len(rejected_records) - 20} more rejected records")
    print("")


def _print_dump_sections(result: Any, nodes: Sequence[Any], args: Any) -> None:
    if bool(args.show_debug_details):
        assignments = _collect_assignments(result)
        _print_summary(result, nodes)
        _print_assembled_topologies(result)
        _print_unclosed_and_incomplete_topologies(result)
        _print_backbone_resource_balance(result)
        _print_base_backbone_resources(nodes)
        _print_unclassified_required_nodes(nodes, assignments)
        if bool(args.show_node_coverage):
            print("")
            _print_node_assignments(nodes, assignments)
    else:
        _print_topology_balance(result, nodes)
        _print_flexible_chain_range(result, nodes)
        _print_compact_topologies(result)
        _print_unrecognized_topology_nodes_detail(result, nodes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump closed topology structures and SU balance from a final_nodes CSV."
    )
    parser.add_argument("csv_path", help="Path to final_nodes_by_global_id.csv")
    parser.add_argument(
        "--show-allocator-log",
        action="store_true",
        help="Print FlexAllocator's internal summary before this dump.",
    )
    parser.add_argument(
        "--show-node-coverage",
        action="store_true",
        help="With --show-debug-details, also print one line per input node with its recognized assignments.",
    )
    parser.add_argument(
        "--show-debug-details",
        action="store_true",
        help="Print allocator internals, resource ledgers, pending candidates, and node-level diagnostics.",
    )
    parser.add_argument(
        "--run-block-c-adjustment",
        action="store_true",
        help="Run Layer4 block_c skeleton adjustment in memory, then dump the adjusted topology allocation.",
    )
    parser.add_argument(
        "--block-c-phase",
        choices=("branch", "extra", "full"),
        default="branch",
        help="Which block_c skeleton adjustment phase to run when --run-block-c-adjustment is enabled.",
    )
    parser.add_argument(
        "--block-c-backend",
        choices=("auto", "layer4", "fallback"),
        default="auto",
        help="Adjustment backend: auto uses Layer4 when available, fallback forces allocator-guided diagnostic edits.",
    )
    parser.add_argument(
        "--block-c-template-lib",
        default=None,
        help="Optional subgraph template library for the Layer2 rebuild inside --run-block-c-adjustment.",
    )
    parser.add_argument(
        "--block-c-max-steps",
        type=int,
        default=50,
        help="Maximum block_c skeleton adjustment iterations.",
    )
    parser.add_argument(
        "--show-block-c-log",
        action="store_true",
        help="Print block_c adjustment logs instead of suppressing them.",
    )
    parser.add_argument(
        "--no-block-c-step-trace",
        action="store_true",
        help="Disable per-step topology dumps during --run-block-c-adjustment.",
    )
    args = parser.parse_args(argv)

    csv_path = str(Path(args.csv_path).expanduser())
    nodes = _read_nodes(csv_path)
    result = _allocate_nodes(nodes, show_allocator_log=bool(args.show_allocator_log))

    print(f"input_csv: {csv_path}")
    if bool(args.run_block_c_adjustment):
        print("========== Before Block C Adjustment ==========")
        _print_dump_sections(result, nodes, args)
        hist_after, moves, meta, hist_before, hist_after_nodes = _run_block_c_adjustment(
            nodes,
            phase=str(args.block_c_phase),
            max_steps=int(args.block_c_max_steps),
            show_block_c_log=bool(args.show_block_c_log),
            backend=str(args.block_c_backend),
            csv_path=csv_path,
            template_lib_path=(str(Path(args.block_c_template_lib).expanduser()) if args.block_c_template_lib else None),
            trace_steps=not bool(args.no_block_c_step_trace),
            show_allocator_log=bool(args.show_allocator_log),
            show_debug_details=bool(args.show_debug_details),
            show_node_coverage=bool(args.show_node_coverage),
        )
        _print_block_c_adjustment_summary(moves, meta, hist_before, hist_after_nodes)
        adjusted_result = _allocate_nodes(nodes, show_allocator_log=bool(args.show_allocator_log))
        print("========== After Block C Adjustment ==========")
        _print_dump_sections(adjusted_result, nodes, args)
    else:
        _print_dump_sections(result, nodes, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
