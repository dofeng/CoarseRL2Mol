import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, List, Sequence, Tuple, Dict, Optional, Set, Union
import matplotlib.pyplot as plt

try:
    from .coarse_graph import (
        SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS, PPM_STEP
    )
except ImportError:
    from model.shared.coarse_graph import (
        SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS, PPM_STEP
    )
# SU分类（用于快速索引）
SU_CARBONYL = [0, 1, 2, 3, 4]  
SU_AROMATIC = [5, 6, 7, 8, 9, 10, 11, 12, 13] 
SU_UNSATURATED = [14, 15, 16, 17, 18] 
SU_ALIPHATIC = [19, 20, 21, 22, 23, 24, 25] 

# Effective spectral response groups.  These are deliberately kept separate
# from the structural/topology classification: the response coefficient is
# applied only while constructing a reconstructed spectrum.
AROMATIC_SU_TYPES = frozenset(int(value) for value in SU_AROMATIC)

# PPM分段定义
PPM_SEGMENTS = {
    'carbonyl': (160.0, 240.0),  
    'aromatic': (90.0, 160.0),   
    'aliphatic': (0.0, 90.0),
    'ordinary_aliphatic': (0.0, 50.0),
    'oxygenated_aliphatic': (50.0, 100.0),
}


def compute_aromatic_area_fraction(
    spectrum: torch.Tensor,
    ppm_axis: torch.Tensor = PPM_AXIS,
) -> float:
    """Return the target-spectrum aromatic-area fraction.

    The target spectrum is not modified.  The fraction is used only to choose
    the effective response of aromatic structure-unit peaks in reconstructed
    spectra.  The project convention is the 90--160 ppm area divided by the
    complete 0.1--240 ppm area.
    """
    try:
        values = spectrum.detach().cpu().float().flatten()
    except Exception:
        values = torch.as_tensor(spectrum, dtype=torch.float).flatten()
    try:
        axis = ppm_axis.detach().cpu().float().flatten()
    except Exception:
        axis = torch.as_tensor(ppm_axis, dtype=torch.float).flatten()
    n = min(int(values.numel()), int(axis.numel()))
    if n <= 0:
        return 0.0
    values = values[:n]
    axis = axis[:n]
    finite = torch.isfinite(values) & torch.isfinite(axis)
    if not bool(finite.any()):
        return 0.0
    values = values[finite]
    axis = axis[finite]
    if int(axis.numel()) <= 1:
        return 0.0
    # The input spectra are non-negative after preprocessing.  Clamping here
    # makes the area ratio robust to tiny interpolation/baseline negatives
    # without changing the target tensor itself.
    values = torch.clamp(values, min=0.0)
    order = torch.argsort(axis)
    axis = axis[order]
    values = values[order]
    step = torch.median(torch.diff(axis)).item()
    if not np.isfinite(step) or float(step) <= 0.0:
        step = float(PPM_STEP)
    total = float(values.sum().item()) * float(step)
    if total <= 1e-12:
        return 0.0
    aromatic = float(
        values[(axis >= 90.0) & (axis <= 160.0)].sum().item()
    ) * float(step)
    return float(max(0.0, min(1.0, aromatic / total)))


def aromatic_response_scale_from_fraction(aromatic_fraction: float) -> float:
    """Map aromatic area fraction to the configured reconstruction response."""
    value = float(aromatic_fraction)
    if not np.isfinite(value):
        value = 0.0
    if value < 0.72:
        return 1.00
    if value < 0.75:
        return 0.98
    if value < 0.78:
        return 0.96
    if value < 0.82:
        return 0.94
    return 0.92


def reconstruction_peak_scale(
    su_type: int,
    aromatic_response_scale: float = 1.0,
) -> float:
    """Return the effective peak multiplier for one structure unit."""
    try:
        scale = float(aromatic_response_scale)
    except Exception:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(scale if int(su_type) in AROMATIC_SU_TYPES else 1.0)

SPECIAL_DEGREE_PRIORS: Dict[int, Dict[int, float]] = {
    19: {1: 0.35, 2: 0.60, 3: 0.05},
    20: {1: 0.35, 2: 0.60, 3: 0.05},
    21: {2: 0.90, 3: 0.10},
}

SPECIAL_MODE_FORBIDDEN_NEIGHBORS: Dict[Tuple[int, int], Set[int]] = {
    (19, 1): {28},
}

TERMINAL_PORT_BASE_SU: Set[int] = {1, 4, 16, 18, 22, 28, 32}


@dataclass(frozen=True)
class Hop1PortToken:
    """State-aware neighbor token used by 1-hop port rules."""

    su_type: int
    degrees: Tuple[int, ...] = ()
    modes: Tuple[str, ...] = ()
    partitions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Hop1PortSignature:
    """A complete ordered set of ports, with optional center-node constraints."""

    ports: Tuple[FrozenSet[Union[int, Hop1PortToken]], ...]
    center_modes: Tuple[str, ...] = ()
    center_partitions: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.ports)

    def __iter__(self):
        return iter(self.ports)

    def __getitem__(self, index: int) -> FrozenSet[Union[int, Hop1PortToken]]:
        return self.ports[index]


def hop1_token(su_type: int,
               degrees: Optional[Iterable[int]] = None,
               modes: Optional[Iterable[str]] = None,
               partitions: Optional[Iterable[str]] = None) -> Hop1PortToken:
    return Hop1PortToken(
        su_type=int(su_type),
        degrees=tuple(sorted(set(int(x) for x in list(degrees or [])))),
        modes=tuple(sorted(set(str(x) for x in list(modes or [])))),
        partitions=tuple(sorted(set(str(x) for x in list(partitions or [])))),
    )


def hop1_signature(*ports: Iterable[Union[int, Hop1PortToken]],
                   center_modes: Optional[Iterable[str]] = None,
                   center_partitions: Optional[Iterable[str]] = None) -> Hop1PortSignature:
    return Hop1PortSignature(
        ports=tuple(
            frozenset(_normalize_port_entry(entry) for entry in list(port or []))
            for port in ports
        ),
        center_modes=tuple(sorted(set(str(x) for x in list(center_modes or [])))),
        center_partitions=tuple(sorted(set(str(x) for x in list(center_partitions or [])))),
    )


def _normalize_special_mode(mode: Optional[Any]) -> Optional[str]:
    if mode is None:
        return None
    txt = str(mode)
    if txt.startswith('double'):
        return 'double'
    if txt.startswith('single'):
        return 'single'
    return txt


def _node_su_type(node_or_su: Any) -> Optional[int]:
    try:
        if isinstance(node_or_su, Hop1PortToken):
            return int(node_or_su.su_type)
        if hasattr(node_or_su, 'su_type'):
            return int(getattr(node_or_su, 'su_type'))
        return int(node_or_su)
    except Exception:
        return None


def _node_target_degree(node_or_su: Any) -> Optional[int]:
    try:
        val = getattr(node_or_su, 'target_hop1_degree', None)
        return int(val) if val is not None else None
    except Exception:
        return None


def _node_anchor_mode(node_or_su: Any) -> Optional[str]:
    try:
        mode = _normalize_special_mode(getattr(node_or_su, 'special_anchor_mode', None))
        if mode is not None:
            return mode
        fixed = getattr(node_or_su, 'target_fixed_anchor_count', None)
        if fixed is not None:
            fixed_i = int(fixed)
            if fixed_i >= 2:
                return 'double'
            if fixed_i == 1:
                return 'single'
        return None
    except Exception:
        return None


def _node_anchor_partition(node_or_su: Any) -> Optional[str]:
    try:
        val = getattr(node_or_su, 'special_anchor_partition', None)
        return str(val) if val is not None else None
    except Exception:
        return None


def _normalize_port_entry(entry: Union[int, Hop1PortToken]) -> Union[int, Hop1PortToken]:
    if isinstance(entry, Hop1PortToken):
        return entry
    return int(entry)


def port_entry_base_su(entry: Union[int, Hop1PortToken]) -> int:
    if isinstance(entry, Hop1PortToken):
        return int(entry.su_type)
    return int(entry)


def port_entry_matches(entry: Union[int, Hop1PortToken], neighbor: Any) -> bool:
    neighbor_su = _node_su_type(neighbor)
    if neighbor_su is None:
        return False
    if isinstance(entry, Hop1PortToken):
        if int(neighbor_su) != int(entry.su_type):
            return False

        # Integer-only callers are legacy/template validators.  They can only
        # validate the coarse SU type; node-state constraints are enforced by
        # Layer1/adjusters when actual nodes are available.
        if not hasattr(neighbor, 'su_type'):
            return True

        if entry.degrees:
            degree = _node_target_degree(neighbor)
            if degree is None or int(degree) not in set(int(x) for x in entry.degrees):
                return False
        if entry.modes:
            mode = _node_anchor_mode(neighbor)
            if mode is None or str(mode) not in set(str(x) for x in entry.modes):
                return False
        if entry.partitions:
            partition = _node_anchor_partition(neighbor)
            if partition is None or str(partition) not in set(str(x) for x in entry.partitions):
                return False
        return True
    return int(neighbor_su) == int(entry)


def port_entry_debug(entry: Union[int, Hop1PortToken]) -> str:
    if not isinstance(entry, Hop1PortToken):
        return str(int(entry))
    suffix = []
    if entry.degrees:
        suffix.append('d' + ''.join(str(int(x)) for x in entry.degrees))
    if entry.modes:
        suffix.append('mode=' + '/'.join(str(x) for x in entry.modes))
    if entry.partitions:
        suffix.append('part=' + '/'.join(str(x) for x in entry.partitions))
    return f"{int(entry.su_type)}" + (f"({';'.join(suffix)})" if suffix else "")


def signature_matches_center(signature: Hop1PortSignature, center_node: Optional[Any]) -> bool:
    if center_node is None:
        return True
    if signature.center_modes:
        mode = _node_anchor_mode(center_node)
        if mode is not None and str(mode) not in set(str(x) for x in signature.center_modes):
            return False
    if signature.center_partitions:
        partition = _node_anchor_partition(center_node)
        if partition is not None and str(partition) not in set(str(x) for x in signature.center_partitions):
            return False
    return True


def is_terminal_port_node(node_or_su: Any) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None:
        return False
    if int(su_i) in TERMINAL_PORT_BASE_SU:
        return True
    if int(su_i) in {19, 20}:
        degree = _node_target_degree(node_or_su)
        return bool(degree is not None and int(degree) == 1)
    return False


def terminal_neighbor_count(neighbors: Iterable[Any]) -> int:
    return int(sum(1 for nb in list(neighbors or []) if is_terminal_port_node(nb)))


def violates_special_d3_terminal_limit(su_type: int,
                                       target_degree: Optional[int],
                                       neighbors: Iterable[Any]) -> bool:
    """Return whether a d3 special center would be capped by too many terminals."""
    try:
        if int(su_type) not in {19, 20}:
            return False
        if target_degree is None or int(target_degree) != 3:
            return False
    except Exception:
        return False
    return bool(terminal_neighbor_count(neighbors) > 1)


def is_single_special_d3_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) != 3:
        return False
    mode = _node_anchor_mode(node_or_su)
    return str(mode) != 'double'


def is_double_special_d23_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) not in {2, 3}:
        return False
    return str(_node_anchor_mode(node_or_su)) == 'double'


def is_double_special_d3_port_node(node_or_su: Any, su_type: Optional[int] = None) -> bool:
    su_i = _node_su_type(node_or_su)
    if su_i is None or int(su_i) not in {19, 20}:
        return False
    if su_type is not None and int(su_i) != int(su_type):
        return False
    degree = _node_target_degree(node_or_su)
    if degree is None or int(degree) != 3:
        return False
    return str(_node_anchor_mode(node_or_su)) == 'double'


def fixed_connector_special_pair_ok(center_node: Any, neighbors: Iterable[Any]) -> bool:
    center_su = _node_su_type(center_node)
    if center_su not in {27, 29, 31}:
        return True
    special_su = 20 if int(center_su) == 27 else 19
    relevant = [
        nb for nb in list(neighbors or [])
        if _node_su_type(nb) == int(special_su)
    ]
    if len(relevant) < 2:
        return True
    double_d3_count = int(sum(
        1 for nb in relevant
        if is_double_special_d3_port_node(nb, int(special_su))
    ))
    return bool(double_d3_count < 2)


def has_forbidden_fixed_connector_special_combo(center_node: Any, neighbors: Iterable[Any]) -> bool:
    """Return whether a fixed connector has an impossible special pairing."""
    return not fixed_connector_special_pair_ok(center_node, neighbors)


def get_aliphatic_carbon_policy(E_target: torch.Tensor) -> Dict[str, float]:
    """
    根据 H/C 比例决定脂肪碳初始化和 Layer4 上限策略。

    规则:
      - H/C < 0.9:  初始化保持 0.82 * xN, Layer4 上限 0.9 * xN
      - 0.9 <= H/C < 1.1: 初始化提高到 0.9 * xN, Layer4 上限 1.0 * xN
      - H/C >= 1.1: 初始化提高到 1.0 * xN, Layer4 上限 1.1 * xN
    """
    try:
        e = E_target.detach().cpu().flatten().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float).flatten()
        c_count = float(e[0].item()) if int(e.numel()) > 0 else 0.0
        h_count = float(e[1].item()) if int(e.numel()) > 1 else 0.0
        o_count = float(e[2].item()) if int(e.numel()) > 2 else 0.0
    except Exception:
        c_count = 0.0
        h_count = 0.0
        o_count = 0.0

    hc_ratio = float(h_count / max(c_count, 1e-8)) if c_count > 0.0 else 0.0
    oc_ratio = float(o_count / max(c_count, 1e-8)) if c_count > 0.0 else 0.0
    if float(hc_ratio) < 0.9:
        init_scale = 0.82
        upper_scale = 0.90
    elif float(hc_ratio) < 1.1:
        init_scale = 0.90
        upper_scale = 1.00
    else:
        init_scale = 1.00
        upper_scale = 1.10

    # O-rich biomass often needs more room for oxygenated aliphatic carbons, but
    # the spectrum still decides whether that oxygen is carbonyl/aromatic/aliphatic.
    if float(oc_ratio) >= 0.25:
        upper_scale = max(float(upper_scale), 1.05)

    return {
        'hc_ratio': float(hc_ratio),
        'oc_ratio': float(oc_ratio),
        'init_aliphatic_scale': float(init_scale),
        'layer4_aliphatic_upper_scale': float(upper_scale),
    }


def estimate_region_carbon_budgets(
    S_target: torch.Tensor,
    E_target: torch.Tensor,
    ppm_step: float = PPM_STEP,
) -> Dict[str, float]:
    """
    Estimate carbon budgets from integrated NMR regions.

    The target spectrum is normalized to pi * carbon_count under the project's
    hwhm=1 Lorentzian convention, so area ratios are already approximate carbon
    ratios. We keep H/C policy only as a mild scale for the broad aliphatic
    initialization, while exposing ordinary and oxygenated aliphatic budgets
    separately for Layer0/Layer4 constraints.
    """
    try:
        spec = S_target.detach().cpu().flatten().float()
    except Exception:
        spec = torch.tensor([], dtype=torch.float)
    try:
        e = E_target.detach().cpu().flatten().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float).flatten()
    except Exception:
        e = torch.zeros(6, dtype=torch.float)

    total_C = float(e[0].item()) if int(e.numel()) > 0 else 0.0

    def _area(lo_ppm: float, hi_ppm: float) -> float:
        if int(spec.numel()) <= 0:
            return 0.0
        lo_i = max(0, int(math.floor(float(lo_ppm) / float(ppm_step))))
        hi_i = min(int(spec.numel()), int(math.ceil(float(hi_ppm) / float(ppm_step))))
        if hi_i <= lo_i:
            return 0.0
        return float(spec[lo_i:hi_i].sum().item()) * float(ppm_step)

    total_area = float(spec.sum().item()) * float(ppm_step) if int(spec.numel()) > 0 else 0.0
    if total_area <= 1e-8:
        x_ordinary, x_oxygenated, y, z = 0.20, 0.15, 0.55, 0.10
    else:
        ordinary_area = _area(0.0, 50.0)
        oxygenated_area = _area(50.0, 100.0)
        aliphatic_area = _area(0.0, 90.0)
        aromatic_area = _area(90.0, 160.0)
        carbonyl_area = _area(160.0, 240.0)
        x_ordinary = float(ordinary_area / total_area)
        x_oxygenated = float(oxygenated_area / total_area)
        x_aliphatic = float(aliphatic_area / total_area)
        y = float(aromatic_area / total_area)
        z = float(carbonyl_area / total_area)
        # Keep the historical 0-90 aliphatic ratio for compatibility below.
        x = float(x_aliphatic)
        policy = get_aliphatic_carbon_policy(e)
        aliphatic_scale = float(policy.get('init_aliphatic_scale', 0.82))
        return {
            'ordinary_x': float(x_ordinary),
            'oxygenated_x': float(x_oxygenated),
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'N': float(total_C),
            'ordinary_xN': float(x_ordinary * total_C),
            'oxygenated_xN': float(x_oxygenated * total_C),
            'xN': float(x * total_C),
            'yN': float(y * total_C),
            'zN': float(z * total_C),
            'ordinary_aliphatic_C': float(max(0.0, x_ordinary * total_C)),
            'oxygenated_aliphatic_C': float(max(0.0, x_oxygenated * total_C)),
            'aliphatic_C': float(max(0.0, aliphatic_scale * x * total_C)),
            'aromatic_C': float(max(0.0, total_C - (aliphatic_scale * x * total_C) - (z * total_C))),
            'carbonyl_C': float(max(0.0, z * total_C)),
            'hc_ratio': float(policy.get('hc_ratio', 0.0)),
            'init_aliphatic_scale': float(aliphatic_scale),
            'layer4_aliphatic_upper_scale': float(policy.get('layer4_aliphatic_upper_scale', 0.90)),
            'total_area': float(total_area),
        }

    policy = get_aliphatic_carbon_policy(e)
    aliphatic_scale = float(policy.get('init_aliphatic_scale', 0.82))
    x = float(x_ordinary + min(0.40, x_oxygenated))
    return {
        'ordinary_x': float(x_ordinary),
        'oxygenated_x': float(x_oxygenated),
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'N': float(total_C),
        'ordinary_xN': float(x_ordinary * total_C),
        'oxygenated_xN': float(x_oxygenated * total_C),
        'xN': float(x * total_C),
        'yN': float(y * total_C),
        'zN': float(z * total_C),
        'ordinary_aliphatic_C': float(max(0.0, x_ordinary * total_C)),
        'oxygenated_aliphatic_C': float(max(0.0, x_oxygenated * total_C)),
        'aliphatic_C': float(max(0.0, aliphatic_scale * x * total_C)),
        'aromatic_C': float(max(0.0, total_C - (aliphatic_scale * x * total_C) - (z * total_C))),
        'carbonyl_C': float(max(0.0, z * total_C)),
        'hc_ratio': float(policy.get('hc_ratio', 0.0)),
        'init_aliphatic_scale': float(aliphatic_scale),
        'layer4_aliphatic_upper_scale': float(policy.get('layer4_aliphatic_upper_scale', 0.90)),
        'total_area': float(total_area),
    }


def allocate_special_degree_counts(total: int, ratio_map: Dict[int, float]) -> Dict[int, int]:
    total_i = max(0, int(total))
    if total_i <= 0:
        return {int(k): 0 for k in sorted(ratio_map.keys())}

    keys = [int(k) for k in sorted(ratio_map.keys())]
    raw = [max(0.0, float(ratio_map.get(k, 0.0))) * float(total_i) for k in keys]
    base = [int(math.floor(v)) for v in raw]
    remainder = int(total_i - sum(base))
    if remainder > 0 and keys:
        order = sorted(
            range(len(keys)),
            key=lambda i: (raw[i] - float(base[i]), -keys[i]),
            reverse=True,
        )
        for i in range(remainder):
            base[order[i % len(order)]] += 1
    return {int(k): int(v) for k, v in zip(keys, base)}


def normalize_special_degree_meta(total_counts: Dict[int, int],
                                  special_degree_meta: Optional[Dict[int, Dict[int, int]]] = None,
                                  priors: Optional[Dict[int, Dict[int, float]]] = None) -> Dict[int, Dict[int, int]]:
    priors_map = priors or SPECIAL_DEGREE_PRIORS
    meta_in = dict(special_degree_meta or {})
    out: Dict[int, Dict[int, int]] = {}
    for su_type, ratio_map in priors_map.items():
        total = max(0, int(total_counts.get(int(su_type), 0)))
        src = dict(meta_in.get(int(su_type), meta_in.get(str(int(su_type)), {})) or {})
        current = {
            int(deg): max(0, int(src.get(int(deg), src.get(str(int(deg)), 0)) or 0))
            for deg in ratio_map.keys()
        }
        if total <= 0:
            out[int(su_type)] = {int(deg): 0 for deg in ratio_map.keys()}
            continue
        if sum(current.values()) == int(total):
            out[int(su_type)] = {int(deg): int(cnt) for deg, cnt in current.items()}
            continue
        weight_sum = float(sum(current.values()))
        if weight_sum > 0:
            eff_priors = {int(deg): float(current.get(int(deg), 0)) / float(weight_sum) for deg in ratio_map.keys()}
        else:
            eff_priors = {int(deg): float(ratio_map.get(int(deg), 0.0)) for deg in ratio_map.keys()}
        out[int(su_type)] = allocate_special_degree_counts(int(total), eff_priors)
    return out


def rebuild_su19_partition_meta(total_19: int,
                                o_base_19: int,
                                s_reserved_19: int,
                                special_degree_meta_19: Optional[Dict[int, int]] = None,
                                existing_partition_meta_19: Optional[Dict[str, Dict[int, int]]] = None) -> Dict[str, Dict[int, int]]:
    total_i = max(0, int(total_19))
    ether_count = max(0, min(int(o_base_19), int(total_i)))
    thio_count = max(0, min(int(s_reserved_19), int(total_i - ether_count)))
    if int(ether_count + thio_count) < int(total_i):
        ether_count += int(total_i - (ether_count + thio_count))

    priors_19 = dict(SPECIAL_DEGREE_PRIORS.get(19, {1: 0.5, 2: 0.4, 3: 0.1}))
    total_deg = normalize_special_degree_meta(
        {19: int(total_i)},
        special_degree_meta={19: dict(special_degree_meta_19 or {})},
    ).get(19, {})

    existing_meta = dict(existing_partition_meta_19 or {})
    thio_pref_src = dict(existing_meta.get('thio', {}) or {})
    if int(sum(int(thio_pref_src.get(int(deg), thio_pref_src.get(str(int(deg)), 0)) or 0) for deg in priors_19.keys())) > 0:
        thio_pref = {
            int(deg): max(0, int(thio_pref_src.get(int(deg), thio_pref_src.get(str(int(deg)), 0)) or 0))
            for deg in priors_19.keys()
        }
    else:
        thio_pref = allocate_special_degree_counts(int(thio_count), priors_19)

    thio_alloc = {int(deg): 0 for deg in priors_19.keys()}
    remaining_thio = int(thio_count)

    pref_order = sorted(
        priors_19.keys(),
        key=lambda deg: (
            -int(thio_pref.get(int(deg), 0)),
            -int(total_deg.get(int(deg), 0)),
            int(deg),
        ),
    )
    for deg in pref_order:
        if int(remaining_thio) <= 0:
            break
        take = min(
            int(remaining_thio),
            int(total_deg.get(int(deg), 0)),
            int(thio_pref.get(int(deg), 0)),
        )
        thio_alloc[int(deg)] += int(take)
        remaining_thio -= int(take)

    residual_order = sorted(
        priors_19.keys(),
        key=lambda deg: (
            -int(total_deg.get(int(deg), 0) - thio_alloc.get(int(deg), 0)),
            int(deg),
        ),
    )
    for deg in residual_order:
        if int(remaining_thio) <= 0:
            break
        available = int(total_deg.get(int(deg), 0)) - int(thio_alloc.get(int(deg), 0))
        if int(available) <= 0:
            continue
        take = min(int(remaining_thio), int(available))
        thio_alloc[int(deg)] += int(take)
        remaining_thio -= int(take)

    if int(remaining_thio) > 0:
        fallback = allocate_special_degree_counts(int(thio_count), priors_19)
        thio_alloc = {int(deg): 0 for deg in priors_19.keys()}
        remaining_thio = int(thio_count)
        for deg in sorted(
            priors_19.keys(),
            key=lambda d: (
                -int(fallback.get(int(d), 0)),
                -int(total_deg.get(int(d), 0)),
                int(d),
            ),
        ):
            if int(remaining_thio) <= 0:
                break
            take = min(int(remaining_thio), int(total_deg.get(int(deg), 0)))
            thio_alloc[int(deg)] += int(take)
            remaining_thio -= int(take)

    ether_alloc = {
        int(deg): max(0, int(total_deg.get(int(deg), 0)) - int(thio_alloc.get(int(deg), 0)))
        for deg in priors_19.keys()
    }

    ether_total = int(sum(int(v) for v in ether_alloc.values()))
    thio_total = int(sum(int(v) for v in thio_alloc.values()))
    if int(ether_total) != int(ether_count) or int(thio_total) != int(thio_count):
        ether_alloc = allocate_special_degree_counts(int(ether_count), priors_19)
        thio_alloc = allocate_special_degree_counts(int(thio_count), priors_19)

    return {
        'ether': {int(deg): int(ether_alloc.get(int(deg), 0)) for deg in priors_19.keys()},
        'thio': {int(deg): int(thio_alloc.get(int(deg), 0)) for deg in priors_19.keys()},
    }


def select_port_patterns_for_degree(port_patterns: Any, degree: Optional[int]) -> Any:
    if degree is None:
        return port_patterns
    signatures = [sig for sig in iter_port_signature_objects(port_patterns) if int(len(sig.ports)) == int(degree)]
    return signatures


def get_mode_forbidden_neighbors(su_type: int, target_degree: Optional[int]) -> Set[int]:
    if target_degree is None:
        return set()
    return set(SPECIAL_MODE_FORBIDDEN_NEIGHBORS.get((int(su_type), int(target_degree)), set()))


def is_mode_specific_neighbor_forbidden(su_type: int,
                                        target_degree: Optional[int],
                                        neighbor_su: int) -> bool:
    return int(neighbor_su) in get_mode_forbidden_neighbors(int(su_type), target_degree)


def get_special_hydrogen_count(su_type: int, degree: Optional[int]) -> Optional[float]:
    if degree is None:
        return None
    su_i = int(su_type)
    deg_i = int(degree)
    if su_i in {19, 20}:
        return {1: 3.0, 2: 2.0, 3: 1.0}.get(deg_i)
    if su_i == 21:
        return {2: 2.0, 3: 1.0}.get(deg_i)
    return None


def get_node_degree_hint(node: Any) -> Optional[int]:
    try:
        target_degree = getattr(node, 'target_hop1_degree', None)
        if target_degree is not None and int(target_degree) > 0:
            return int(target_degree)
    except Exception:
        pass
    try:
        hop1_ids = list(getattr(node, 'hop1_ids', []) or [])
        if hop1_ids:
            return int(len(hop1_ids))
    except Exception:
        pass
    try:
        hop1_su = getattr(node, 'hop1_su', None)
        if hop1_su is not None:
            deg = int(sum(int(v) for v in hop1_su.values()))
            if deg > 0:
                return int(deg)
    except Exception:
        pass
    return None


def get_effective_node_element_vector(node: Any,
                                      E_SU_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    tensor = E_SU if E_SU_tensor is None else E_SU_tensor
    su_i = int(getattr(node, 'su_type', -1))
    if su_i < 0 or su_i >= int(tensor.size(0)):
        return torch.zeros(tensor.size(1), dtype=torch.float, device=tensor.device)
    vec = tensor[su_i].detach().clone().to(device=tensor.device, dtype=torch.float)
    degree_hint = get_node_degree_hint(node)
    h_override = get_special_hydrogen_count(su_i, degree_hint)
    if h_override is not None and int(vec.numel()) > 1:
        vec[1] = float(h_override)
    return vec


def get_effective_nodes_element_vector(nodes: Sequence[Any],
                                       E_SU_tensor: Optional[torch.Tensor] = None,
                                       device: Optional[torch.device] = None) -> torch.Tensor:
    tensor = E_SU if E_SU_tensor is None else E_SU_tensor
    out_device = device or tensor.device
    total = torch.zeros(tensor.size(1), dtype=torch.float, device=out_device)
    for node in nodes:
        total = total + get_effective_node_element_vector(node, tensor).to(out_device)
    return total


def get_effective_hist_element_vector(H: torch.Tensor,
                                      special_degree_meta: Optional[Dict[int, Dict[int, int]]] = None,
                                      E_SU_tensor: Optional[torch.Tensor] = None,
                                      device: Optional[torch.device] = None) -> torch.Tensor:
    tensor = E_SU if E_SU_tensor is None else E_SU_tensor
    out_device = device or tensor.device
    H_cpu = torch.clamp(H.detach().cpu(), min=0).long()
    total = torch.matmul(H_cpu.float(), tensor.detach().cpu().float())
    meta = normalize_special_degree_meta(
        {int(su): int(H_cpu[int(su)].item()) for su in SPECIAL_DEGREE_PRIORS.keys() if int(H_cpu.numel()) > int(su)},
        special_degree_meta=special_degree_meta,
    )
    if int(total.numel()) > 1:
        for su_type, degree_map in meta.items():
            if int(H_cpu.numel()) <= int(su_type):
                continue
            base_h = float(tensor.detach().cpu().float()[int(su_type), 1].item())
            for degree, count in degree_map.items():
                h_override = get_special_hydrogen_count(int(su_type), int(degree))
                if h_override is None:
                    continue
                total[1] += float(count) * float(h_override - base_h)
    return total.to(out_device).float()

# ============================================================================
# 结构单元连接规则（基于化学语义）
# ============================================================================

# SU连接度定义
SU_CONNECTION_DEGREE = {
    0: 2,   # Amide_Group: -C(=O)-NH-
    1: 1,   # Carboxylic_Acid: -COOH
    2: 2,   # Ester_Group: -C(=O)-O-
    3: 2,   # Aldehyde_Ketone_C: -C(=O)- 可桥接或末端
    4: 1,   # Nitrile_C: -C≡N
    5: 3,   # O_Substituted_Aro_C: 芳香-氧连接
    6: 3,   # N_Substituted_Aro_C: 芳香-氮连接
    7: 3,   # S_Substituted_Aro_C: 芳香-硫连接
    8: 3,   # X_Substituted_Aro_C: 芳香-卤连接
    9: 3,   # Keto_Substituted_Aro_C: 芳香-羰基连接
    10: 3,  # Aryl_Substituted_Aro_C: 芳基取代
    11: 3,  # Alkyl_Substituted_Aro_C: 烷基取代
    12: 3,  # Aromatic_Bridgehead_C: 稠环桥头
    13: 2,  # Carbocyclic_Aro_CH: 芳香CH
    14: 3,  # Vinyllic_Cq: >C=
    15: 2,  # Vinyllic_CH: -HC=
    16: 1,  # Vinyllic_CH2: =CH2
    17: 2,  # Alkynyl_Cq: -C≡
    18: 1,  # Alkynyl_CH: ≡CH
    19: 3,  # Alcohol_Ether_C: 节点级目标连接度可为1/2/3，运行时由target_hop1_degree约束
    20: 3,  # Amine_C: 节点级目标连接度可为1/2/3，运行时由target_hop1_degree约束
    21: 3,  # Halogenated_C: 节点级目标连接度可为2/3，运行时由target_hop1_degree约束
    22: 1,  # Alkyl_CH3: -CH3
    23: 2,  # Alkyl_CH2: -CH2-
    24: 3,  # Alkyl_CH: -CH<
    25: 4,  # Alkyl_Cq: >C<
    26: 2,  # Heterocyclic_N: 吡啶氮
    27: 2,  # Amine_Nitrogen: -NH-
    28: 1,  # Hydroxyl_O: -OH
    29: 2,  # Ether_O: -O-
    30: 2,  # Heterocyclic_S: 噻吩硫
    31: 2,  # Thioether_S: -S-
    32: 1,  # Halogen_X: -X
}

# ============================================================================
# 1-hop端口组合规则
# 说明:
# - 普通整数表示只关心 SU 类型。
# - Hop1PortToken 表示状态感知邻居，如 19(d=2/3)、20(mode=single)。
# - Hop1PortSignature 可限制中心节点的 single/double/ether/thio 模式。
# ============================================================================
D23 = (2, 3)
D123 = (1, 2, 3)
SINGLE = ('single',)
DOUBLE = ('double',)
ETHER = ('ether',)
THIO = ('thio',)

SU19_D123 = hop1_token(19, degrees=D123)
SU19_D23 = hop1_token(19, degrees=D23)
SU19_D1 = hop1_token(19, degrees=(1,))
SU19_SINGLE_D23 = hop1_token(19, degrees=D23, modes=SINGLE)
SU20_D123 = hop1_token(20, degrees=D123)
SU20_D23 = hop1_token(20, degrees=D23)
SU20_D1 = hop1_token(20, degrees=(1,))
SU20_SINGLE_D23 = hop1_token(20, degrees=D23, modes=SINGLE)
SU21_D23 = hop1_token(21, degrees=D23)

ARO = {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}
ARO_NO9 = {13, 12, 11, 10, 5, 6, 7, 8, 26, 30}
ARO_NO10_12 = {13, 12, 11, 5, 6, 7, 8, 9, 26, 30}
ARO_HET_RING = {13, 12, 11, 10, 5, 6, 7, 8, 9}
SAT_EXT_1 = {23, 11, 22, 24, 25, 2, 29, 31, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2 = {23, 11, 24, 25, 2, 29, 31, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_1_19_ETHER = {23, 11, 22, 24, 25, 2, 29, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2_19_ETHER = {23, 11, 24, 25, 2, 29, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_1_19_THIO = {23, 11, 22, 24, 25, 31, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
SAT_EXT_2_19_THIO = {23, 11, 24, 25, 31, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}
O_ANCHORS = {2, 28, 29}
O_ANCHORS_NO28 = {2, 29}
S_ANCHORS = {31}
HAL_EXT = {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}

HOP1_PORT_COMBINATIONS: Dict[int, Any] = {
    0: [
        {9, 23, 24, 25, 22, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {6, SU20_D123},
    ],
    1: [{9, 23, 24, 25, 14, 15, SU19_D23, SU20_D23, SU21_D23}],
    2: [
        {9, 23, 24, 25, 22, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {5, SU19_D123},
    ],
    3: [
        {9, 23, 24, 25, 22, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {9, 23, 24, 25, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    4: [{23, 24, 25}],
    5: [ARO, ARO, {2, 28, 29}],
    6: [ARO, ARO, {0, 27}],
    7: [ARO, ARO, {31}],
    8: [ARO, ARO, {32}],
    9: [ARO_NO9, ARO_NO9, {0, 1, 2, 3}],
    10: [ARO_NO10_12, ARO_NO10_12, {10}],
    11: [ARO, ARO, {23, 24, 25, 22, 15, 17, SU19_D23, SU20_D23, SU21_D23}],
    12: [ARO, ARO, {12}],
    13: [ARO, ARO],
    14: [
        {23, 24, 25, 22, 2, 1, 0, 3, SU19_D23, SU20_D23, SU21_D23},
        {23, 24, 25, 22, 2, 1, 0, 3, 4, SU19_D23, SU20_D23, SU21_D23},
        {14, 15, 16},
    ],
    15: [{23, 24, 25, 22, 11, 2, 1, 0, 3, SU19_D23, SU20_D23, SU21_D23}, {14, 15, 16}],
    16: [{14, 15}],
    17: [{23, 24, 25, 11, 19, 20, 21, 2, 0, 3}, {17, 18}],
    18: [{17}],
    19: [
        hop1_signature(O_ANCHORS_NO28, center_modes=SINGLE, center_partitions=ETHER),
        hop1_signature(S_ANCHORS, center_modes=SINGLE, center_partitions=THIO),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            O_ANCHORS,
            center_modes=SINGLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            S_ANCHORS,
            center_modes=SINGLE,
            center_partitions=THIO,
        ),
        hop1_signature(O_ANCHORS, O_ANCHORS_NO28, center_modes=DOUBLE, center_partitions=ETHER),
        hop1_signature(S_ANCHORS, S_ANCHORS, center_modes=DOUBLE, center_partitions=THIO),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            SAT_EXT_2_19_ETHER,
            O_ANCHORS,
            center_modes=SINGLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            SAT_EXT_2_19_THIO,
            S_ANCHORS,
            center_modes=SINGLE,
            center_partitions=THIO,
        ),
        hop1_signature(
            SAT_EXT_1_19_ETHER,
            O_ANCHORS,
            O_ANCHORS_NO28,
            center_modes=DOUBLE,
            center_partitions=ETHER,
        ),
        hop1_signature(
            SAT_EXT_1_19_THIO,
            S_ANCHORS,
            S_ANCHORS,
            center_modes=DOUBLE,
            center_partitions=THIO,
        ),
    ],
    20: [
        hop1_signature({0, 27}, center_modes=SINGLE),
        hop1_signature({23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}, {0, 27}, center_modes=SINGLE),
        hop1_signature({0, 27}, {0, 27}, center_modes=DOUBLE),
        hop1_signature(
            {23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {23, 11, 24, 25, 27, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {0, 27},
            center_modes=SINGLE,
        ),
        hop1_signature(
            {23, 11, 22, 24, 25, 27, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
            {0, 27},
            {0, 27},
            center_modes=DOUBLE,
        ),
    ],
    21: [
        hop1_signature(HAL_EXT, {32}),
        hop1_signature(HAL_EXT, HAL_EXT, {32}),
    ],
    22: [{23, 11, 24, 25, 2, 3, 1, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23}],
    23: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    24: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    25: [
        {23, 11, 22, 24, 25, 2, 3, 1, 0, 4, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
        {23, 11, 24, 25, 2, 3, 0, 14, 15, 17, SU19_D23, SU20_D23, SU21_D23},
    ],
    26: [ARO_HET_RING, ARO_HET_RING],
    27: [
        hop1_signature({6, SU20_D123}, {6, SU20_D23}),
        hop1_signature({SU20_D123}, {SU20_D23}),
    ],
    28: [{5, SU19_D23}],
    29: [
        hop1_signature({5, SU19_D123}, {5, SU19_D23}),
        hop1_signature({5}, {SU19_D23}),
        hop1_signature({SU19_D123}, {SU19_D23}),
    ],
    30: [ARO_HET_RING, ARO_HET_RING],
    31: [
        hop1_signature({7, SU19_D123}, {7, SU19_D23}),
        hop1_signature({SU19_D123}, {SU19_D23}),
    ],
    32: [{8, SU21_D23}],
}

def iter_port_signature_objects(port_patterns: Any) -> List[Hop1PortSignature]:
    """返回带中心状态约束的签名对象。"""
    if not port_patterns:
        return []

    if isinstance(port_patterns, Hop1PortSignature):
        return [port_patterns]

    if isinstance(port_patterns, (list, tuple)) and port_patterns:
        first = port_patterns[0]
        if isinstance(first, set) and all(isinstance(item, set) for item in port_patterns):
            return [
                hop1_signature(*[
                    {_normalize_port_entry(v) for v in allowed}
                    for allowed in port_patterns
                ])
            ]

    signatures: List[Hop1PortSignature] = []
    if isinstance(port_patterns, (list, tuple)):
        for signature in port_patterns:
            if isinstance(signature, Hop1PortSignature):
                signatures.append(signature)
                continue
            if not isinstance(signature, (list, tuple)):
                continue
            sig_sets: List[Set[Union[int, Hop1PortToken]]] = []
            ok = True
            for allowed in signature:
                if not isinstance(allowed, set):
                    ok = False
                    break
                sig_sets.append(set(_normalize_port_entry(v) for v in allowed))
            if ok and sig_sets:
                signatures.append(hop1_signature(*sig_sets))
    return signatures


def get_port_pattern_degrees(port_patterns: Any) -> Tuple[int, ...]:
    """返回该端口规则允许的所有合法连接度。"""
    return tuple(sorted(set(len(sig.ports) for sig in iter_port_signature_objects(port_patterns) if sig.ports)))


def format_port_patterns_debug(port_patterns: Any) -> List[List[List[str]]]:
    """将端口规则格式化为便于打印/调试的整数列表。"""
    return [
        [
            sorted(port_entry_debug(v) for v in allowed)
            for allowed in sig.ports
        ]
        for sig in iter_port_signature_objects(port_patterns)
    ]


# 每个SU的所有端口允许的邻居类型的并集
SU_FIXED_CONNECTIONS = {
    su: sorted({
        int(port_entry_base_su(v))
        for sig in iter_port_signature_objects(port_sets)
        for allowed in sig.ports
        for v in allowed
    })
    for su, port_sets in HOP1_PORT_COMBINATIONS.items()
}

# 外接结构要求
SU_EXTERNAL_CONNECTIONS = {
    5: [2, 28, 29],  
    6: [0, 27],      
    7: [31],         
    8: [32],         
    9: [0, 1, 2, 3], 
    10: [10],
    11: [23, 24, 25, 22, 19, 20, 21, 15, 17],  
    19: [2, 28, 29, 31],  
    20: [0, 27],   
    21: [32],       
}


def required_external_candidates_for_node(node: Any,
                                          missing_su19_as_none: bool = False) -> Optional[List[int]]:
    """Return fixed-anchor candidate SU types for a stateful Layer1 node."""
    try:
        su_i = int(getattr(node, 'su_type', -1))
    except Exception:
        return None if bool(missing_su19_as_none) else []

    if int(su_i) == 19:
        part = _node_anchor_partition(node)
        deg = _node_target_degree(node)
        if str(part) == 'thio':
            return [31]
        if str(part) == 'ether':
            if int(deg or 0) == 1:
                return [2, 29]
            return [2, 28, 29]
        return None if bool(missing_su19_as_none) else []

    return [int(x) for x in list(SU_EXTERNAL_CONNECTIONS.get(int(su_i), []) or [])]


def validate_connection(center_su: int,
                        neighbor_su: int,
                        E_target: Optional[torch.Tensor] = None) -> bool:
    """
    对单条候选边做粗粒度 fallback 过滤。

    状态感知的合法性由端口 matcher 负责；这里仅保留类型并集
    和元素预算这类低成本检查，供旧调用点快速剪枝。
    """
    center_i = int(center_su)
    neighbor_i = int(neighbor_su)

    # 14/15/16 are unsaturated non-aromatic units.  SU15 may use SU11 as a
    # backbone port; the other aromatic 5-13 ports are forbidden here.
    if center_i in {14, 16} and 5 <= neighbor_i <= 13:
        return False
    if neighbor_i in {14, 16} and 5 <= center_i <= 13:
        return False
    if center_i == 15 and 5 <= neighbor_i <= 13 and neighbor_i != 11:
        return False
    if neighbor_i == 15 and 5 <= center_i <= 13 and center_i != 11:
        return False

    allowed = SU_FIXED_CONNECTIONS.get(center_i)
    if allowed is not None and neighbor_i not in allowed:
        return False

    if E_target is not None:
        try:
            halogen_budget = float(E_target[5].item())
        except Exception:
            halogen_budget = 0.0
        if halogen_budget <= 0.0 and (center_su in {8, 21} or neighbor_su in {8, 21, 32}):
            return False

    return True


def check_external_connection_requirement(center_su: int, hop1_counter: Counter) -> Tuple[bool, str]:
    """
    检查芳香取代位点和特殊脂肪碳的外接要求
    """
    if center_su not in SU_EXTERNAL_CONNECTIONS:
        return True, ""
    
    required_external = SU_EXTERNAL_CONNECTIONS[center_su]
    hop1_types = set(hop1_counter.keys())

    if not any(ext_su in hop1_types for ext_su in required_external):
        return False, f"SU {center_su} requires external connection to {required_external}"
    
    return True, ""


# ============================================================================
# NMR工具函数
# ============================================================================

def lorentzian_spectrum(mus: torch.Tensor, pis: torch.Tensor, ppm_axis: torch.Tensor,
                         hwhm: float = 1.0) -> torch.Tensor:
    if mus.dim() == 1:
        mus = mus.unsqueeze(1)
    if pis.dim() == 1:
        pis = pis.unsqueeze(1)
    hwhm_sq = hwhm ** 2
    delta_ppm_sq = (ppm_axis.unsqueeze(0) - mus) ** 2
    all_peaks = pis * hwhm_sq / (delta_ppm_sq + hwhm_sq)
    return all_peaks.sum(dim=0)


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """计算R²评分"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return float(r2.item())


def compute_segment_r2(y_true: torch.Tensor, y_pred: torch.Tensor, 
                       ppm_axis: torch.Tensor, segment_name: str) -> float:
    """计算特定PPM分段的R²"""
    lo, hi = PPM_SEGMENTS[segment_name]
    mask = (ppm_axis >= lo) & (ppm_axis <= hi)
    if not mask.any():
        return 0.0
    return compute_r2_score(y_true[mask], y_pred[mask])


def resample_spectrum_to_ppm_axis(
    ppm_values: np.ndarray | torch.Tensor | List[float],
    intensities: np.ndarray | torch.Tensor | List[float],
    ppm_axis: torch.Tensor = PPM_AXIS,
) -> torch.Tensor:
    """
    将任意 ppm 采样顺序/间距的谱图重采样到项目统一的 PPM_AXIS 上。

    这样可避免直接忽略 CSV 第一列 ppm 而导致升降序或非等间距输入被错误解释。
    """
    ppm_np = np.asarray(ppm_values, dtype=np.float64).reshape(-1)
    intensity_np = np.asarray(intensities, dtype=np.float64).reshape(-1)

    mask = np.isfinite(ppm_np) & np.isfinite(intensity_np)
    ppm_np = ppm_np[mask]
    intensity_np = intensity_np[mask]

    if ppm_np.size == 0 or intensity_np.size == 0:
        return torch.zeros_like(ppm_axis, dtype=torch.float)

    order = np.argsort(ppm_np)
    ppm_sorted = ppm_np[order]
    intensity_sorted = intensity_np[order]

    ppm_unique, inverse = np.unique(ppm_sorted, return_inverse=True)
    if ppm_unique.size != ppm_sorted.size:
        sums = np.zeros_like(ppm_unique, dtype=np.float64)
        counts = np.zeros_like(ppm_unique, dtype=np.float64)
        np.add.at(sums, inverse, intensity_sorted)
        np.add.at(counts, inverse, 1.0)
        intensity_sorted = sums / np.maximum(counts, 1.0)
        ppm_sorted = ppm_unique

    axis_np = ppm_axis.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y_interp = np.interp(axis_np, ppm_sorted, intensity_sorted, left=0.0, right=0.0)
    return torch.tensor(y_interp, dtype=torch.float)


def normalize_spectrum_to_carbon_count(
    spectrum: torch.Tensor,
    carbon_count: float,
    ppm_step: float = PPM_STEP,
) -> torch.Tensor:
    """
    将输入谱图面积缩放到与训练数据一致的目标面积: pi * num_carbons。
    """
    y = spectrum.detach().clone().float().flatten()
    carbon = max(0.0, float(carbon_count))
    if carbon <= 0.0:
        return y

    current_area = float(y.sum().item()) * float(ppm_step)
    if current_area <= 1e-8:
        return y

    target_area = float(math.pi) * float(carbon)
    scale = float(target_area / current_area)
    return y * scale


def evaluate_spectrum_reconstruction(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ppm_axis: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Evaluate an inverse reconstruction at its fixed unit-peak scale."""
    y_true_t = y_true.flatten().float()
    y_pred_t = y_pred.flatten().float().to(y_true_t.device)

    n = int(min(y_true_t.numel(), y_pred_t.numel()))
    y_true_t = y_true_t[:n]
    y_pred_t = y_pred_t[:n]

    ppm_eval = None
    if ppm_axis is not None:
        ppm_eval = ppm_axis.flatten().float().to(y_true_t.device)[:n]

    out: Dict[str, Any] = {
        'S_target': y_true_t,
        'S_recon': y_pred_t,
        'r2': float(compute_r2_score(y_true_t, y_pred_t)),
    }

    if ppm_eval is not None and int(ppm_eval.numel()) == int(y_true_t.numel()):
        out['ppm_axis'] = ppm_eval
        out['r2_carbonyl'] = float(compute_segment_r2(y_true_t, y_pred_t, ppm_eval, 'carbonyl'))
        out['r2_aromatic'] = float(compute_segment_r2(y_true_t, y_pred_t, ppm_eval, 'aromatic'))
        out['r2_aliphatic'] = float(compute_segment_r2(y_true_t, y_pred_t, ppm_eval, 'aliphatic'))

    return out


def resolve_eval_inputs(S_target: torch.Tensor,
                        ppm_axis: torch.Tensor,
                        device: str | torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    S_eval = S_target.to(device).flatten()
    ppm_eval = ppm_axis.to(device).flatten()
    if int(S_eval.numel()) != int(ppm_eval.numel()):
        n = int(min(S_eval.numel(), ppm_eval.numel()))
        S_eval = S_eval[:n]
        ppm_eval = ppm_eval[:n]
    return S_eval, ppm_eval


def reconstruct_from_mu_pi(mus: Sequence[float],
                           pis: Sequence[float],
                           ppm_axis: torch.Tensor,
                           hwhm: float,
                           intensity_scale: float,
                           device: str | torch.device,
                           unit_peak_intensity: bool = False,
                           peak_scales: Optional[Sequence[float]] = None) -> torch.Tensor:
    if not mus:
        return torch.zeros_like(ppm_axis, dtype=torch.float, device=ppm_axis.device)

    mu_t = torch.tensor(list(mus), dtype=torch.float, device=device)
    if bool(unit_peak_intensity):
        pi_t = torch.ones(int(mu_t.numel()), dtype=torch.float, device=device)
    else:
        pi_t = torch.tensor(list(pis), dtype=torch.float, device=device)
    if (not bool(unit_peak_intensity)) and float(intensity_scale) != 1.0:
        pi_t = pi_t * float(intensity_scale)
    if peak_scales is not None:
        if torch.is_tensor(peak_scales):
            scales_t = peak_scales.detach().to(device=device, dtype=torch.float).flatten()
        else:
            scales_t = torch.tensor(list(peak_scales), dtype=torch.float, device=device)
        if int(scales_t.numel()) != int(mu_t.numel()):
            raise ValueError(
                "peak_scales must contain one value per reconstructed peak: "
                f"peaks={int(mu_t.numel())}, scales={int(scales_t.numel())}"
            )
        if not bool(torch.isfinite(scales_t).all()) or bool((scales_t <= 0.0).any()):
            raise ValueError("peak_scales must be finite and strictly positive")
        pi_t = pi_t * scales_t
    return lorentzian_spectrum(mu_t, pi_t, ppm_axis, hwhm=float(hwhm))


def evaluate_mu_pi_assignments(S_target: torch.Tensor,
                               ppm_axis: torch.Tensor,
                               mus: Sequence[float],
                               pis: Sequence[float],
                               hwhm: float,
                               intensity_scale: float,
                               device: str | torch.device,
                               unit_peak_intensity: bool = False,
                               peak_scales: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    if not mus:
        return {
            'S_target': S_target,
            'r2': 0.0,
            'r2_carbonyl': 0.0,
            'r2_aromatic': 0.0,
            'r2_aliphatic': 0.0,
            'diff': np.zeros(int(ppm_axis.numel()), dtype=np.float64),
        }

    S_recon = reconstruct_from_mu_pi(
        mus=mus,
        pis=pis,
        ppm_axis=ppm_axis,
        hwhm=float(hwhm),
        intensity_scale=float(intensity_scale),
        device=device,
        unit_peak_intensity=bool(unit_peak_intensity),
        peak_scales=peak_scales,
    )
    eval_info = evaluate_spectrum_reconstruction(
        S_target,
        S_recon,
        ppm_axis=ppm_axis,
    )
    return {
        'S_target': eval_info['S_target'],
        'S_recon': S_recon,
        'r2': float(eval_info.get('r2', 0.0)),
        'r2_carbonyl': float(eval_info.get('r2_carbonyl', 0.0)),
        'r2_aromatic': float(eval_info.get('r2_aromatic', 0.0)),
        'r2_aliphatic': float(eval_info.get('r2_aliphatic', 0.0)),
        'diff': (eval_info['S_target'] - eval_info['S_recon']).detach().cpu().numpy(),
    }


def multiset_from_counter(cnt: Counter) -> Tuple[int, ...]:
    ms = []
    for k, v in cnt.items():
        try:
            kk = int(k)
            vv = int(v)
        except Exception:
            continue
        if vv <= 0:
            continue
        ms.extend([kk] * vv)
    ms.sort()
    return tuple(ms)


def multiset_l1_distance(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    c1 = Counter(ms1)
    c2 = Counter(ms2)
    all_keys = set(c1.keys()) | set(c2.keys())
    return int(sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in all_keys))


def multiset_overlap_size(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    c1 = Counter(ms1)
    c2 = Counter(ms2)
    all_keys = set(c1.keys()) | set(c2.keys())
    return int(sum(min(int(c1.get(k, 0)), int(c2.get(k, 0))) for k in all_keys))


def multiset_diff_nodes(ms1: Tuple[int, ...], ms2: Tuple[int, ...]) -> int:
    ov = multiset_overlap_size(ms1, ms2)
    return int(max(len(ms1), len(ms2)) - ov)


def save_node_peak_rows(rows: Iterable[Dict[str, Any]],
                        output_dir: str,
                        filename: str = 'layer1_library_node_peaks.csv') -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(str(Path(output_dir) / filename), index=False)


def save_spectrum_comparison(S_target: torch.Tensor,
                             ppm: np.ndarray,
                             S_recon: np.ndarray,
                             diff: np.ndarray,
                             output_dir: str,
                             prefix: str = 'layer1_library') -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'ppm': ppm,
        'target': S_target.detach().cpu().numpy(),
        'reconstructed': S_recon,
        'difference': diff,
    }).to_csv(str(Path(output_dir) / f'{prefix}_spectrum_comparison.csv'), index=False)


def save_spectrum_figure(S_target: torch.Tensor,
                         S_recon: torch.Tensor,
                         ppm: np.ndarray,
                         output_dir: str,
                         layer_name: str) -> None:
    visualize_spectrum_comparison(
        S_target.detach().cpu(),
        S_recon.detach().cpu(),
        torch.tensor(ppm, dtype=torch.float32),
        layer_name,
        save_dir=output_dir,
    )

def visualize_su_distribution(su_hist: torch.Tensor, layer_name: str, 
                               save_dir: str = 'inverse_result'):
    """可视化SU分布（保持与v2兼容）"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    su_names = [name for name, _ in SU_DEFS]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(NUM_SU_TYPES)
    counts = su_hist.detach().cpu().numpy()
    
    colors = ['#d62728' if i in SU_CARBONYL else 
              '#2ca02c' if i in SU_AROMATIC else 
              '#1f77b4' for i in range(NUM_SU_TYPES)]
    
    ax.bar(x, counts, color=colors, alpha=0.7)
    ax.set_xlabel('SU Index', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{layer_name} SU Distribution', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(su_names, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{layer_name.lower()}_su_distribution.png', dpi=300)
    plt.close(fig)


def visualize_spectrum_comparison(S_target: torch.Tensor, S_recon: torch.Tensor,
                                   ppm_axis: torch.Tensor, layer_name: str,
                                   save_dir: str = 'inverse_result'):
    """可视化谱图对比"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ppm = ppm_axis.detach().cpu().numpy()
    target = S_target.detach().cpu().numpy()
    recon = S_recon.detach().cpu().numpy()
    diff = target - recon
    
    # 对比图
    ax1.fill_between(ppm, 0, target, alpha=0.3, color='#1f77b4', label='Target (area)')
    ax1.fill_between(ppm, 0, recon, alpha=0.3, color='#ff7f0e', label='Reconstructed (area)')
    ax1.plot(ppm, target, label='Target', lw=2.0, color='#1f77b4')
    ax1.plot(ppm, recon, label=f'Reconstructed ({layer_name})', lw=2.0, 
             color='#ff7f0e', linestyle='--')
    ax1.set_ylabel('Intensity', fontsize=14)
    ax1.set_title(f'Target vs. Reconstructed ({layer_name})', fontsize=16)
    ax1.invert_xaxis()
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(alpha=0.3)
    
    # 差谱图
    ax2.fill_between(ppm, 0, diff, alpha=0.4, color='crimson', label='Difference (area)')
    ax2.plot(ppm, diff, label='Difference (Target - Recon)', lw=1.5, color='crimson')
    ax2.axhline(0.0, color='black', lw=1.0, alpha=0.6, linestyle=':')
    ax2.set_xlabel('Chemical Shift (ppm)', fontsize=14)
    ax2.set_ylabel('Intensity', fontsize=14)
    ax2.set_title(f'Difference Spectrum ({layer_name})', fontsize=16)
    ax2.invert_xaxis()
    ax2.legend(frameon=False, fontsize=12)
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{layer_name.lower()}_comparison.png', dpi=300)
    plt.close(fig)


# ============================================================================
# 节点数据结构
# ============================================================================

class _NodeV3:
    """
    改进的节点类，存储结构单元节点的完整信息
    """
    __slots__ = ['global_id', 'su_type', 'hop1_su', 'hop2_su', 'hop1_ids', 'hop2_ids',
                 'fixed_hop1_ids',
                 'z_vec', 'mu', 'pi', 'z_history', 
                 'constraint_violations', 'score_components', 'template_key',
                 'target_hop1_degree', 'init_target_hop1_degree', 'special_degree_source',
                 'special_anchor_partition',
                 'target_fixed_anchor_count', 'init_target_fixed_anchor_count',
                 'special_anchor_mode',
                 'stage3_topology_owner_id', 'stage3_topology_role',
                 'stage3_locked_neighbor_ids', 'stage3_resource_claimed']
    
    def __init__(self, global_id: int, su_type: int,
                 hop1_su: Optional[Counter] = None, 
                 hop2_su: Optional[Counter] = None,
                 z_vec: Optional[torch.Tensor] = None,
                 mu: float = 0.0, pi: float = 1.0,
                 target_hop1_degree: Optional[int] = None,
                 special_degree_source: Optional[str] = None,
                 special_anchor_partition: Optional[str] = None,
                 target_fixed_anchor_count: Optional[int] = None,
                 special_anchor_mode: Optional[str] = None):
        self.global_id = global_id
        self.su_type = su_type
        self.hop1_su = hop1_su if hop1_su is not None else Counter()
        self.hop2_su = hop2_su if hop2_su is not None else Counter()
        self.hop1_ids = []  # 存储1-hop邻居的全局ID
        self.hop2_ids = []  # 存储去重后的2-hop邻居全局ID
        self.fixed_hop1_ids = set()  # 存储不可拆除的固定1-hop邻居ID
        self.z_vec = z_vec if z_vec is not None else torch.zeros(16)
        self.mu = mu
        self.pi = pi
        self.z_history = [z_vec.clone()] if z_vec is not None and z_vec.numel() > 0 else []
        self.constraint_violations = set()
        self.score_components = {}
        self.template_key = None
        self.target_hop1_degree = int(target_hop1_degree) if target_hop1_degree is not None else None
        self.init_target_hop1_degree = int(target_hop1_degree) if target_hop1_degree is not None else None
        self.special_degree_source = str(special_degree_source) if special_degree_source is not None else None
        self.special_anchor_partition = (
            str(special_anchor_partition) if special_anchor_partition is not None else None
        )
        self.target_fixed_anchor_count = (
            int(target_fixed_anchor_count) if target_fixed_anchor_count is not None else None
        )
        self.init_target_fixed_anchor_count = (
            int(target_fixed_anchor_count) if target_fixed_anchor_count is not None else None
        )
        self.special_anchor_mode = (
            str(special_anchor_mode) if special_anchor_mode is not None else None
        )
        self.stage3_topology_owner_id = None
        self.stage3_topology_role = None
        self.stage3_locked_neighbor_ids = set()
        self.stage3_resource_claimed = False
    
    @property
    def center_su(self):
        """兼容性属性，返回su_type"""
        return self.su_type
    
    @property
    def hop1_counter(self):
        """兼容性属性，返回hop1_su"""
        return self.hop1_su
    
    @property
    def hop2_counter(self):
        """兼容性属性，返回hop2_su"""
        return self.hop2_su
    
    @property
    def z(self):
        """兼容性属性，返回z_vec"""
        return self.z_vec
    
    def get_hop1_degree(self) -> int:
        """获取当前1-hop连接度"""
        return sum(self.hop1_su.values())
    
    def get_max_degree(self) -> int:
        """获取该SU类型的最大连接度"""
        try:
            if self.target_hop1_degree is not None and int(self.target_hop1_degree) > 0:
                return int(self.target_hop1_degree)
        except Exception:
            pass
        max_deg = SU_CONNECTION_DEGREE.get(self.su_type, 4)
        if isinstance(max_deg, tuple):
            return max_deg[1]
        return max_deg
    
    def is_hop1_complete(self) -> bool:
        """检查1-hop是否已完成分配"""
        return self.get_hop1_degree() >= self.get_max_degree()
    
    def is_hop1_empty(self) -> bool:
        """检查1-hop是否为空"""
        return self.get_hop1_degree() == 0
    
    def remaining_hop1_slots(self) -> int:
        """剩余可分配的1-hop槽位"""
        return max(0, self.get_max_degree() - self.get_hop1_degree())
    
    def has_neighbor(self, neighbor_id: int) -> bool:
        """检查是否已连接到指定邻居"""
        return neighbor_id in self.hop1_ids
    
    def validate_hop1_consistency(self) -> Tuple[bool, List[str]]:
        """
        验证hop1_su与hop1_ids的一致性
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 检查数量一致性
        if len(self.hop1_ids) != sum(self.hop1_su.values()):
            errors.append(f"Node {self.global_id}: hop1_ids长度({len(self.hop1_ids)}) != hop1_su总数({sum(self.hop1_su.values())})")
        
        # 检查度数约束
        current_degree = self.get_hop1_degree()
        max_degree = self.get_max_degree()
        if current_degree > max_degree:
            errors.append(f"Node {self.global_id}: 度数超限 {current_degree} > {max_degree}")
        
        # 检查hop1_ids中无重复
        if len(self.hop1_ids) != len(set(self.hop1_ids)):
            errors.append(f"Node {self.global_id}: hop1_ids中存在重复ID")
        
        # 检查hop1_ids中不含自身
        if self.global_id in self.hop1_ids:
            errors.append(f"Node {self.global_id}: hop1_ids中包含自身ID")

        invalid_fixed = [nid for nid in self.fixed_hop1_ids if nid not in self.hop1_ids]
        if invalid_fixed:
            errors.append(f"Node {self.global_id}: fixed_hop1_ids包含不存在的邻居 {sorted(invalid_fixed)}")
        
        return len(errors) == 0, errors
    
    def __repr__(self):
        return (f"NodeV3(id={self.global_id}, su={self.su_type}, "
                f"hop1={dict(self.hop1_su)}, degree={self.get_hop1_degree()}/{self.get_max_degree()}, "
                f"target={self.target_hop1_degree}, part={self.special_anchor_partition}, "
                f"fixed={self.target_fixed_anchor_count}, mode={self.special_anchor_mode})")
