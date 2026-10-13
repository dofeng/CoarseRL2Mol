"""G2S spectrum guidance for fixed-topology RL_MTCS assembly.

The inverse pipeline already supplies every structural unit.  During flex and
side placement only external graph edges and aromatic physical-site labels
change.  This adapter therefore scores a complete, coordinate-free graph and
caches predictions by rooted k-hop environment.  Hex-grid rotations and
translations never trigger redundant G2S inference.
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model.forward.g2s.g2s_model import LocalNMRDataset
from model.forward.g2s.g2s_utils import load_g2s_checkpoint
from model.shared.coarse_graph import E_SU, NUM_SU_TYPES, PPM_AXIS, PPM_STEP
from model.shared.inverse_common import (
    compute_r2_score,
    lorentzian_spectrum,
    normalize_spectrum_to_carbon_count,
    resample_spectrum_to_ppm_axis,
)


ELEMENT_NAMES = ("C", "H", "O", "N", "S", "X")


def parse_element_composition(value: Any) -> torch.Tensor:
    """Return target ``[C, H, O, N, S, X]`` counts from common inputs."""
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().float().flatten()
    elif isinstance(value, Mapping):
        upper = {str(key).upper(): float(count) for key, count in value.items()}
        result = torch.tensor(
            [upper.get(name, 0.0) for name in ELEMENT_NAMES], dtype=torch.float
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.tensor(list(value), dtype=torch.float).flatten()
    elif isinstance(value, (str, Path)):
        text = str(value).strip()
        path = Path(text)
        if path.is_file():
            rows: Dict[str, float] = {}
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError(f"element composition CSV has no header: {path}")
                fields = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
                element_field = fields.get("element")
                value_field = fields.get("target") or fields.get("predicted")
                if not element_field or not value_field:
                    raise ValueError(
                        "element composition CSV requires element and target columns"
                    )
                for row in reader:
                    name = str(row.get(element_field, "")).strip().upper()
                    if name in ELEMENT_NAMES:
                        rows[name] = float(row.get(value_field, 0.0) or 0.0)
            result = torch.tensor(
                [rows.get(name, 0.0) for name in ELEMENT_NAMES], dtype=torch.float
            )
        else:
            pairs = re.findall(
                r"([CHONSX])\s*(?:=\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                text.upper(),
            )
            if not pairs:
                raise ValueError(
                    "elements must be a mapping, six values, a composition CSV, "
                    "or an expression such as C=500,H=520,O=166,N=6,S=0,X=0"
                )
            parsed = {name: float(count) for name, count in pairs}
            result = torch.tensor(
                [parsed.get(name, 0.0) for name in ELEMENT_NAMES], dtype=torch.float
            )
    else:
        raise TypeError(f"unsupported element composition: {type(value).__name__}")
    if int(result.numel()) != 6:
        raise ValueError(f"element composition must contain six values, got {result.tolist()}")
    if not torch.isfinite(result).all() or bool((result < 0).any()):
        raise ValueError(f"invalid element composition: {result.tolist()}")
    if float(result[0].item()) <= 0.0:
        raise ValueError("target element composition must contain at least one carbon")
    return result


def load_target_spectrum_from_csv(
    csv_path: str | Path,
    *,
    carbon_count: float,
) -> torch.Tensor:
    """Read a two-column spectrum and align it to the project's PPM axis."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"target spectrum does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(handle, dialect))
    if not rows:
        raise ValueError(f"target spectrum is empty: {path}")
    start = 0
    try:
        float(rows[0][0])
        float(rows[0][1])
    except (IndexError, TypeError, ValueError):
        start = 1
    ppm_values = []
    intensities = []
    for row in rows[start:]:
        if len(row) < 2:
            continue
        try:
            ppm = float(row[0])
            intensity = float(row[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(ppm) and math.isfinite(intensity):
            ppm_values.append(ppm)
            intensities.append(intensity)
    if not ppm_values:
        raise ValueError(f"target spectrum has no numeric ppm/intensity rows: {path}")
    target = resample_spectrum_to_ppm_axis(
        np.asarray(ppm_values), np.asarray(intensities), ppm_axis=PPM_AXIS
    )
    return normalize_spectrum_to_carbon_count(
        target, float(carbon_count), ppm_step=float(PPM_STEP)
    ).cpu()


def _bond_features(
    source_su: int,
    target_su: int,
    *,
    aromatic_ring: bool = False,
    aliphatic_ring: bool = False,
) -> Tuple[float, float]:
    if aromatic_ring:
        return 1.5, 1.0
    pair = {int(source_su), int(target_su)}
    if pair <= {17, 18}:
        return 3.0, float(aliphatic_ring)
    if pair <= {14, 15, 16}:
        return 2.0, float(aliphatic_ring)
    return 1.0, float(aliphatic_ring)


def _cycle_edges(edges):
    """Return graph cycle edges, excluding pendant branches (iterative DFS)."""
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    order, low, parent, bridges = {}, {}, {}, set()
    for root in adjacency:
        if root in order:
            continue
        order[root] = low[root] = len(order)
        parent[root] = None
        stack = [(root, iter(adjacency[root]))]
        while stack:
            a, neighbors = stack[-1]
            b = next(neighbors, None)
            if b is None:
                stack.pop()
                p = parent[a]
                if p is not None:
                    low[p] = min(low[p], low[a])
                    if low[a] > order[p]:
                        bridges.add(tuple(sorted((p, a))))
            elif b != parent[a]:
                if b in order:
                    low[a] = min(low[a], order[b])
                else:
                    parent[b] = a
                    order[b] = low[b] = len(order)
                    stack.append((b, iter(adjacency[b])))
    return {tuple(sorted(e)) for e in edges} - bridges


class NMRGuidanceScorer:
    """Batch G2S scorer with graph-level and rooted k-hop caches."""

    def __init__(
        self,
        *,
        spectrum_csv: str | Path,
        elements: Any,
        model_checkpoint: str | Path,
        device: str = "auto",
        batch_size: int = 1,
        hwhm: float = 1.0,
        graph_cache_limit: int = 2048,
        node_cache_limit: int = 8192,
        record_limit: int = 10000,
        center_limit: int = 8,
        aromatic_response_scale: float = 1.0,
    ) -> None:
        requested = str(device or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"NMR device {device!r} was requested but CUDA is unavailable"
            )
        self.device = torch.device(requested)
        self.model, self.model_config, _metadata = load_g2s_checkpoint(
            model_checkpoint, device=self.device
        )
        self.k_hop = int(self.model_config.k_hop)
        # Batch size controls collation memory, never the number of scored roots.
        self.batch_size = max(1, int(batch_size))
        self.graph_cache_limit = min(256, max(1, int(graph_cache_limit)))
        self.node_cache_limit = min(2048, max(1, int(node_cache_limit)))
        self.record_limit = min(5000, max(0, int(record_limit)))
        self.center_limit = min(32, max(1, int(center_limit)))
        self.aromatic_response_scale = float(aromatic_response_scale)
        self.hwhm = float(hwhm)
        self.elements = parse_element_composition(elements)
        self.target_spectrum = load_target_spectrum_from_csv(
            spectrum_csv, carbon_count=float(self.elements[0].item())
        )
        self.spectrum_csv = str(Path(spectrum_csv).resolve())
        self.model_checkpoint = str(Path(model_checkpoint).resolve())
        self._graph_cache: Dict[str, Dict[str, Any]] = {}
        self._node_cache: Dict[Tuple[Any, ...], float] = {}
        self._records: list[Dict[str, Any]] = []
        self._stats: Dict[str, int] = defaultdict(int)
        self._stage_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @staticmethod
    def _effective_su_map(
        state: Any,
    ) -> Dict[int, int]:
        current = {
            int(node_id): int(su)
            for node_id, su in state.current_node_map().items()
        }
        return current

    def _graph_descriptor(
        self,
        state: Any,
    ) -> Dict[str, Any]:
        all_nodes = sorted(state.node_axial)
        labels = self._effective_su_map(state)
        su_map = {n: labels[n] for n in all_nodes if n in labels}
        if set(all_nodes) != set(su_map):
            missing = sorted(set(all_nodes) - set(su_map))
            raise ValueError(f"assembly SU map is incomplete; missing={missing[:12]}")
        index = {node_id: offset for offset, node_id in enumerate(all_nodes)}
        ownership = state.problem.ownership_map()
        units = {str(unit.unit_id): unit for unit in state.problem.units}
        edge_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for edge in state.edges:
            left, right = sorted((
                int(edge.source_node_id), int(edge.target_node_id)
            ))
            unit = units[ownership[left]]
            internal = ownership[left] == ownership[right]
            aromatic = internal and unit.phase == 'aromatic'
            ring = internal and unit.phase == 'ring'
            edge_map[(left, right)] = _bond_features(su_map[left], su_map[right],
                aromatic_ring=aromatic, aliphatic_ring=ring)

        cycle_edges = _cycle_edges(edge_map)
        edge_map = {edge: (attr[0], float(edge in cycle_edges))
                    for edge, attr in edge_map.items()}
        labels = tuple((node_id, int(su_map[node_id])) for node_id in all_nodes)
        edges = tuple(
            (left, right, float(attr[0]), float(attr[1]))
            for (left, right), attr in sorted(edge_map.items())
        )
        graph_key = hashlib.sha256(repr((labels, edges)).encode("utf-8")).hexdigest()

        su_indices = torch.tensor([su_map[node_id] for node_id in all_nodes], dtype=torch.long)
        if bool(((su_indices < 0) | (su_indices >= NUM_SU_TYPES)).any()):
            raise ValueError("candidate graph contains an invalid SU type")
        x = torch.cat((
            F.one_hot(su_indices, num_classes=NUM_SU_TYPES).float(),
            E_SU[su_indices].float(),
        ), dim=1)
        edge_index = (
            torch.tensor(
                [[index[left], index[right]] for left, right, _bond, _ring in edges],
                dtype=torch.long,
            ).t().contiguous()
            if edges else torch.empty((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor([[bond, ring] for _l, _r, bond, ring in edges], dtype=torch.float)
            if edges else torch.empty((0, 2), dtype=torch.float)
        )
        is_carbon = E_SU[su_indices, 0] > 0.0
        carbon_count = int(is_carbon.sum().item())
        su_hist = torch.bincount(su_indices, minlength=NUM_SU_TYPES).float()
        adjacency: Dict[int, set[int]] = {idx: set() for idx in range(len(all_nodes))}
        edge_attr_by_index: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for left, right, bond, ring in edges:
            a, b = index[left], index[right]
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_attr_by_index[tuple(sorted((a, b)))] = (bond, ring)
        return {
            "graph_key": graph_key,
            "node_ids": all_nodes,
            "su_indices": su_indices,
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "is_carbon": is_carbon,
            "carbon_count": carbon_count,
            "su_hist": su_hist,
            "adjacency": adjacency,
            "edge_attr_by_index": edge_attr_by_index,
        }

    def _root_key(self, descriptor: Mapping[str, Any], center: int) -> Tuple[Any, ...]:
        adjacency = descriptor["adjacency"]
        distance = {int(center): 0}
        queue = deque([int(center)])
        while queue:
            node = queue.popleft()
            if distance[node] >= self.k_hop:
                continue
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        selected = set(distance)
        node_ids = descriptor["node_ids"]
        su_indices = descriptor["su_indices"]
        labels = tuple(sorted(
            (int(node_ids[idx]), int(su_indices[idx].item())) for idx in selected
        ))
        edges = tuple(sorted(
            (
                int(node_ids[left]), int(node_ids[right]),
                float(attr[0]), float(attr[1]),
            )
            for (left, right), attr in descriptor["edge_attr_by_index"].items()
            if left in selected and right in selected
        ))
        # The full nested environment was previously retained as the key for
        # every predicted centre.  Large beam-search layers can contain tens
        # of thousands of distinct environments, making that representation
        # consume several gigabytes even though equality is all the cache
        # needs.  A deterministic digest preserves the cache semantics while
        # keeping each key bounded in size.
        environment_digest = hashlib.sha256(
            repr((labels, edges)).encode("utf-8")
        ).digest()
        return (
            int(node_ids[center]),
            int(self.k_hop),
            environment_digest,
        )

    def _raw_graph(self, descriptor: Mapping[str, Any], centers: Iterable[int]) -> Dict[str, Any]:
        selected_centers = [int(center) for center in centers]
        mask = torch.zeros(int(descriptor["x"].shape[0]), dtype=torch.bool)
        for center in selected_centers:
            mask[center] = True
        count = int(descriptor["x"].shape[0])
        return {
            "x": descriptor["x"],
            "edge_index": descriptor["edge_index"],
            "edge_attr": descriptor["edge_attr"],
            "is_carbon": mask,
            "su_hist": descriptor["su_hist"],
            "total_atom_counts": self.elements,
            "real_ppm": torch.zeros(count, dtype=torch.float),
            "real_intensity": torch.ones(count, dtype=torch.float),
            # LocalNMRDataset uses this optional field to materialise only
            # the requested roots instead of all carbon centres in the graph.
            "center_ids": torch.tensor(selected_centers, dtype=torch.long),
        }

    @torch.no_grad()
    def _predict_missing_roots(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        root_keys: Sequence[Dict[int, Tuple[Any, ...]]],
        *,
        stage: str,
    ) -> Dict[Tuple[Any, ...], float]:
        predictions = {key:self._node_cache[key] for keys in root_keys for key in keys.values()
                       if key in self._node_cache}
        representatives: Dict[Tuple[Any, ...], Tuple[int, int]] = {}
        for graph_index, keys in enumerate(root_keys):
            for center, key in keys.items():
                if key not in self._node_cache:
                    representatives.setdefault(key, (graph_index, int(center)))
        if not representatives:
            return predictions
        centers_by_graph: Dict[int, list[int]] = defaultdict(list)
        key_by_graph_center: Dict[Tuple[int, int], Tuple[Any, ...]] = {}
        for key, (graph_index, center) in representatives.items():
            centers_by_graph[graph_index].append(center)
            key_by_graph_center[(graph_index, center)] = key
        graph_indices = sorted(centers_by_graph)
        missing_count = len(representatives)
        self._stats["node_predictions"] += missing_count
        self._stage_stats[stage]["node_predictions"] += missing_count

        # LocalNMRDataset eagerly materialises one k-hop graph per requested
        # carbon centre.  Building it for an entire search layer can multiply
        # hundreds of candidates by hundreds of centres and exhaust RAM before
        # DataLoader gets a chance to batch inference.  Partition by centre
        # count so both dataset construction and model execution stay bounded.
        # PyG's eager collation has a much larger transient footprint than the
        # raw feature tensors.  Keep this below the configured model batch so
        # a caller can request a large inference batch without also requesting
        # a multi-gigabyte in-memory preprocessing batch.
        # Keep each PyG collation small.  A k-hop graph around an HSQ centre
        # can still be large, so eight roots per batch creates a high
        # allocator watermark even when the model batch is bounded.
        root_limit = self.center_limit
        center_chunks: list[Tuple[int, list[int]]] = []
        for graph_index in graph_indices:
            graph_centers = centers_by_graph[graph_index]
            center_chunks.extend(
                (graph_index, graph_centers[offset:offset + root_limit])
                for offset in range(0, len(graph_centers), root_limit)
            )

        for graph_index, chunk_centers in center_chunks:
            raw_list = [self._raw_graph(
                descriptors[graph_index], chunk_centers
            )]
            dataset = LocalNMRDataset(
                raw_list, k=self.k_hop, require_real_ppm=False
            )
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False
            )
            for batch in loader:
                self._stats["model_calls"] += 1
                self._stage_stats[stage]["model_calls"] += 1
                batch = batch.to(self.device)
                (mu_pred, _pi), _latent = self.model(batch)
                mu_pred = torch.clamp(
                    mu_pred.view(-1), min=0.1, max=240.0
                ).cpu()
                mol_ids = batch.mol_id.view(-1).long().cpu().tolist()
                centers = (
                    batch.original_center_id.view(-1).long().cpu().tolist()
                )
                for value, raw_index, center in zip(
                    mu_pred.tolist(), mol_ids, centers
                ):
                    if int(raw_index) != 0:
                        raise RuntimeError(
                            "unexpected graph index in chunked NMR inference"
                        )
                    key = key_by_graph_center[(graph_index, int(center))]
                    predictions[key] = float(value)
                    self._node_cache[key] = float(value)
                    if len(self._node_cache) > self.node_cache_limit:
                        # Evict an arbitrary oldest insertion.  Cache entries
                        # are an optimization only; dropping one never
                        # changes the assembly result.
                        self._node_cache.pop(next(iter(self._node_cache)))
                del batch, mu_pred, mol_ids, centers, _pi, _latent
            del loader, dataset, raw_list
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return predictions

    def _spectrum_metrics(
        self,
        shifts: torch.Tensor,
        su_types: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        weights = torch.ones_like(shifts).float()
        if su_types is not None:
            from model.shared.inverse_common import reconstruction_peak_scale
            weights = torch.tensor([reconstruction_peak_scale(int(su), self.aromatic_response_scale)
                                    for su in su_types])
        predicted = lorentzian_spectrum(
            shifts.float(), weights, PPM_AXIS.cpu(),
            hwhm=self.hwhm,
        ).cpu()
        target = self.target_spectrum[: predicted.numel()].float()
        predicted = predicted[: target.numel()].float()
        residual = predicted - target
        mse = float(torch.mean(residual.square()).item())
        rmse = math.sqrt(max(0.0, mse))
        mae = float(torch.mean(residual.abs()).item())
        denom = float(torch.linalg.vector_norm(target).item()) * float(
            torch.linalg.vector_norm(predicted).item()
        )
        cosine = (
            float(torch.dot(target, predicted).item()) / denom if denom > 1e-12 else 0.0
        )
        target_rms = float(torch.sqrt(torch.mean(target.square())).item())
        metrics = {
            "loss": float(1.0 - cosine),
            "cosine_similarity": float(cosine),
            "r2": float(compute_r2_score(target, predicted)),
            "rmse": float(rmse),
            "normalized_rmse": float(rmse / max(target_rms, 1e-12)),
            "mae": float(mae),
            "carbon_centers": int(shifts.numel()),
        }
        return metrics, predicted

    def score_candidates(
        self,
        candidates: Sequence[Any],
        *,
        stage: str,
        depth: int,
    ) -> list[Dict[str, Any]]:
        """Score one geometrically valid search layer in a single G2S batch."""
        if stage == "rigid":
            raise ValueError("rigid stage must not request NMR guidance")
        # Bound live graph descriptors independently of the number of proposals.
        if len(candidates) > 1:
            return [self.score_candidates(
                [candidate], stage=stage, depth=depth,
            )[0] for candidate in candidates]
        stage = str(stage)
        descriptors = []
        cached_results: Dict[int, Dict[str, Any]] = {}
        graph_cache_before: Dict[int, bool] = {}
        for index, candidate in enumerate(candidates):
            state = getattr(candidate, "state", candidate)
            descriptor = self._graph_descriptor(state)
            descriptors.append(descriptor)
            cached = self._graph_cache.get(descriptor["graph_key"])
            graph_cache_before[index] = cached is not None
            if cached is not None:
                cached_results[index] = cached
        total = len(candidates)
        self._stats["candidate_evaluations"] += total
        self._stage_stats[stage]["candidate_evaluations"] += total
        self._stats["graph_cache_hits"] += len(cached_results)
        self._stage_stats[stage]["graph_cache_hits"] += len(cached_results)

        representative_by_graph: Dict[str, int] = {}
        representatives: set[int] = set()
        batch_duplicates = 0
        for index, descriptor in enumerate(descriptors):
            if index in cached_results:
                continue
            graph_key = str(descriptor["graph_key"])
            if graph_key in representative_by_graph:
                batch_duplicates += 1
            else:
                representative_by_graph[graph_key] = index
                representatives.add(index)
        self._stats["batch_duplicate_graphs"] += batch_duplicates
        self._stage_stats[stage]["batch_duplicate_graphs"] += batch_duplicates

        root_keys: list[Dict[int, Tuple[Any, ...]]] = []
        for index, descriptor in enumerate(descriptors):
            if index in cached_results or index not in representatives:
                root_keys.append({})
                continue
            centers = torch.nonzero(
                descriptor["is_carbon"], as_tuple=False
            ).view(-1).tolist()
            keys = {center: self._root_key(descriptor, center) for center in centers}
            root_keys.append(keys)
            self._stats["node_cache_hits"] += sum(
                key in self._node_cache for key in keys.values()
            )
            self._stage_stats[stage]["node_cache_hits"] += sum(
                key in self._node_cache for key in keys.values()
            )
        predictions = self._predict_missing_roots(descriptors, root_keys, stage=stage)
        predictions = self._node_cache if predictions is None else predictions

        outputs = []
        for index, descriptor in enumerate(descriptors):
            cached = cached_results.get(index) or self._graph_cache.get(
                descriptor["graph_key"]
            )
            if cached is None:
                keys = root_keys[index]
                available = sum(key in predictions for key in keys.values())
                coverage = available / max(1, descriptor["carbon_count"])
                if available != descriptor["carbon_count"] or not keys:
                    result = dict(scored=False, coverage=coverage, loss=None,
                                  reason="missing_nmr_predictions",
                                  carbon_centers=available, graph_key=descriptor["graph_key"])
                    outputs.append(result)
                    self._stats["unscored_candidates"] += 1
                    self._stage_stats[stage]["unscored_candidates"] += 1
                    if self.record_limit:
                        self._records.append(dict(stage=stage, depth=depth,
                                                  candidate_index=index, **result))
                        del self._records[:-self.record_limit]
                    continue
                shifts = torch.tensor([predictions[key] for key in keys.values()])
                su_types = descriptor["su_indices"][list(keys)]
                metrics, spectrum = self._spectrum_metrics(shifts, su_types)
                metrics.update(scored=True, coverage=1.0, reason="")
                cached = {
                    "metrics": metrics,
                    "spectrum": spectrum,
                    "shifts": shifts,
                }
                self._graph_cache[descriptor["graph_key"]] = cached
                if len(self._graph_cache) > self.graph_cache_limit:
                    self._graph_cache.pop(next(iter(self._graph_cache)))
                self._stats["unique_graphs_scored"] += 1
                self._stage_stats[stage]["unique_graphs_scored"] += 1
            result = {
                **cached["metrics"],
                "graph_key": descriptor["graph_key"],
            }
            outputs.append(result)
            if self.record_limit:
                self._records.append({
                    "stage": stage,
                    "depth": int(depth),
                    "candidate_index": int(index),
                    "graph_key": descriptor["graph_key"],
                    "graph_cache_hit": bool(graph_cache_before[index]),
                    **cached["metrics"],
                })
                if len(self._records) > self.record_limit:
                    del self._records[:len(self._records) - self.record_limit]
        return outputs

    def score_state(self, state: Any, *, stage: str = "final") -> Dict[str, Any]:
        return self.score_candidates([state], stage=stage, depth=0)[0]

    def statistics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        values = self._stage_stats.get(str(stage), {}) if stage else self._stats
        return {
            "enabled": True,
            "device": str(self.device),
            "model_checkpoint": self.model_checkpoint,
            "k_hop": int(self.k_hop),
            "batch_size": int(self.batch_size),
            "candidate_evaluations": int(values.get("candidate_evaluations", 0)),
            "unique_graphs_scored": int(values.get("unique_graphs_scored", 0)),
            "graph_cache_hits": int(values.get("graph_cache_hits", 0)),
            "batch_duplicate_graphs": int(values.get("batch_duplicate_graphs", 0)),
            "node_cache_hits": int(values.get("node_cache_hits", 0)),
            "node_predictions": int(values.get("node_predictions", 0)),
            "model_calls": int(values.get("model_calls", 0)),
            "unscored_candidates": int(values.get("unscored_candidates", 0)),
            "aromatic_response_scale": self.aromatic_response_scale,
            "center_limit": int(self.center_limit),
        }

    def export_outputs(self, output_dir: str | Path, final_state: Any) -> Dict[str, str]:
        directory = Path(output_dir) / "nmr"
        directory.mkdir(parents=True, exist_ok=True)
        final_metrics = self.score_state(final_state, stage="final")
        cached = self._graph_cache.get(final_metrics["graph_key"])
        if cached is None or not final_metrics.get("scored", False):
            path = directory / "summary.json"
            path.write_text(json.dumps(dict(final_metrics=final_metrics,
                                           statistics=self.statistics()), indent=2),
                            encoding="utf-8")
            return {"nmr_summary_json": str(path)}
        spectrum = cached["spectrum"].cpu()
        target = self.target_spectrum[: spectrum.numel()].cpu()
        spectrum_csv = directory / "final_spectrum.csv"
        with spectrum_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("ppm", "target", "predicted", "residual"))
            for ppm, observed, predicted in zip(
                PPM_AXIS.tolist(), target.tolist(), spectrum.tolist()
            ):
                writer.writerow((ppm, observed, predicted, observed - predicted))
        candidate_csv = directory / "candidate_scores.csv"
        if self._records:
            with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in self._records for key in row}))
                writer.writeheader()
                writer.writerows(self._records)
        summary_json = directory / "summary.json"
        summary_json.write_text(json.dumps({
            "target_spectrum_csv": self.spectrum_csv,
            "target_elements": dict(zip(ELEMENT_NAMES, self.elements.tolist())),
            "final_metrics": {
                key: value for key, value in final_metrics.items() if key != "graph_key"
            },
            "statistics": self.statistics(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        plot_png = directory / "final_spectrum.png"
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            2, 1, figsize=(9.0, 5.8), sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
        )
        ppm = PPM_AXIS[: spectrum.numel()].numpy()
        axes[0].plot(ppm, target.numpy(), color="#2878B5", lw=1.3, label="Target")
        axes[0].plot(ppm, spectrum.numpy(), color="#D55E00", lw=1.1, ls="--", label="G2S")
        axes[0].set_ylabel("Intensity")
        axes[0].legend(frameon=False)
        axes[0].set_title(
            f"Final G2S reconstruction, R2={final_metrics['r2']:.4f}, "
            f"cosine={final_metrics['cosine_similarity']:.4f}"
        )
        axes[1].plot(ppm, (target - spectrum).numpy(), color="#444444", lw=0.9)
        axes[1].axhline(0.0, color="black", lw=0.7)
        axes[1].set_xlabel("Chemical shift (ppm)")
        axes[1].set_ylabel("Residual")
        for axis in axes:
            axis.set_xlim(float(ppm.max()), float(ppm.min()))
            axis.tick_params(direction="in")
        fig.tight_layout()
        fig.savefig(plot_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return {
            "nmr_directory": str(directory),
            "nmr_spectrum_csv": str(spectrum_csv),
            "nmr_candidate_scores_csv": str(candidate_csv),
            "nmr_summary_json": str(summary_json),
            "nmr_spectrum_png": str(plot_png),
        }


__all__ = [
    "NMRGuidanceScorer",
    "load_target_spectrum_from_csv",
    "parse_element_composition",
]
