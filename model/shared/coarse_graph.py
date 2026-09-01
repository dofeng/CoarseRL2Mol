import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from typing import List, Dict, Tuple, Optional, Set
import os
import sys
import networkx as nx
import numpy as np
import re
import math
from collections import deque
import itertools as _it
from scipy.signal import find_peaks

# 自定义异常类，用于更好地处理不同的跳过情况
class MoleculeProcessingError(Exception):
    """分子处理过程中的异常基类"""
    pass

class TooFewNodesError(MoleculeProcessingError):
    """粗粒度节点数量过少异常"""
    pass

class NoStructureUnitsError(MoleculeProcessingError):
    """未识别到结构单元异常"""
    pass

class UnassignedAtomsError(MoleculeProcessingError):
    """存在未分配原子异常"""
    pass

class IncompleteShiftMappingError(MoleculeProcessingError):
    """原子级核磁位移与碳原子索引不能一一对应"""
    pass

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

# --- 谱图相关全局常量 ---
NUM_SU_TYPES: int = 33
NMR_DATA_POINTS: int = 2400
PPM_MAX: float = 240.0
PPM_MIN: float = 0.1
PPM_STEP: float = 0.1
# 创建一个新的从0.1到240ppm的坐标轴
PPM_AXIS = torch.arange(PPM_MIN, PPM_MAX + PPM_STEP, PPM_STEP)


SU_DEFS: List[Tuple[str, str]] = [
    # A类: 核心官能团 (红色)（0-4）
    ('Amide_Group', '[#6](=[O])[#7]'),
    ('Carboxylic_Acid', '[CX3](=[O])[OX2H1]'),
    ('Ester_Group', '[#6](=[O])[#8;X2;!H1]'),

    # B类: 其他单原子官能团 (橙色)
    ('Aldehyde_Ketone_C', '[#6;H1,H0](=[O])'),
    ('Nitrile_C', '[#6]#[N]'),

    # C类: 芳香碳 (蓝色)（5-13）
    ('O_Substituted_Aro_C', '[c;$(c-O)]'),
    ('N_Substituted_Aro_C', '[c;$(c-N)]'),
    ('S_Substituted_Aro_C', '[c;$(c-S)]'),
    ('X_Substituted_Aro_C', '[c;$(c-F),$(c-Cl),$(c-Br),$(c-I)]'),
    ('Keto_Substituted_Aro_C', '[c;$(c-C=O)]'),
    ('Aryl_Substituted_Aro_C', '[c;$(c-!@c)]'),
    ('Alkyl_Substituted_Aro_C','[c;$(c-C)]'),
    ('Aromatic_Bridgehead_C', '[c;a;H0;R2,R3,R4]'),
    ('Carbocyclic_Aro_CH', '[cH1;a]'),

    # D类: 链状不饱和碳 (青色)（14-18）
    ('Vinyllic_Cq', '[CX3H0;$(C=C)]'),
    ('Vinyllic_CH', '[CX3H1;$(C=C)]'),
    ('Vinyllic_CH2', '[CX3H2;$(C=C)]'),
    ('Alkynyl_Cq', '[CX2H0;$(C#C)]'),
    ('Alkynyl_CH', '[CX2H1;$(C#C)]'),

    # E类: 饱和脂肪碳 (绿色)（19-25）
    ('Alcohol_Ether_C', '[CX4;$(C-[O,o,S,s])]'),
    ('Amine_C', '[CX4;$(C-[N,n])]'),
    ('Halogenated_C', '[CX4;$(C-[F,Cl,Br,I])]'),
    ('Alkyl_CH3', '[CX4H3]'),
    ('Alkyl_CH2', '[CX4H2]'),
    ('Alkyl_CH', '[CX4H1]'),
    ('Alkyl_Cq', '[CX4H0]'),

    # F类: 杂原子节点 (紫色)（26-32）
    ('Heterocyclic_N', '[n;r;+0;!$(nC=O)]'),
    ('Amine_Nitrogen', '[N;X3;v3;+0;!$(NC=O)]'),
    ('Hydroxyl_O', '[OX2H;$(O-c),$(O-C)]'),
    ('Ether_O', '[$([O;X2;!H1;!$([O]-C(=O)O)]),$([o;r])]'),
    ('Heterocyclic_S', '[s;r]'),
    ('Thioether_S', '[S;X2H0]'),
    ('Halogen_X', '[F,Cl,Br,I]'),
]

# 创建从名称到整数ID的映射和预编译模式
SU_NAME_TO_ID: Dict[str, int] = {name: i for i, (name, _) in enumerate(SU_DEFS)}
SU_PATTERNS: List[Tuple[str, Chem.Mol]] = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in SU_DEFS]

# V-Final-Fix: Degree/Valency constraints for each SU
SU_MAX_DEGREE = {
    # A类: 核心官能团
    'Amide_Group': 2,
    'Carboxylic_Acid': 1,
    'Ester_Group': 2,
    # B类: 其他单原子官能团
    'Aldehyde_Ketone_C': 2,
    'Nitrile_C': 1,
    # C类: 芳香碳
    'O_Substituted_Aro_C': 3,
    'N_Substituted_Aro_C': 3,
    'S_Substituted_Aro_C': 3,
    'X_Substituted_Aro_C': 3,
    'Keto_Substituted_Aro_C': 3,
    'Aryl_Substituted_Aro_C': 3,
    'Alkyl_Substituted_Aro_C': 3,
    'Aromatic_Bridgehead_C': 3,
    'Carbocyclic_Aro_CH': 2,
    # D类: 链状不饱和碳
    'Vinyllic_Cq': 2,
    'Vinyllic_CH': 2,
    'Vinyllic_CH2': 1,
    'Alkynyl_Cq': 2,
    'Alkynyl_CH': 1,
    # E类: 饱和脂肪碳
    'Alcohol_Ether_C': 2,
    'Amine_C': 2,
    'Halogenated_C': 2,
    'Alkyl_CH3': 1,
    'Alkyl_CH2': 2,
    'Alkyl_CH': 3,
    'Alkyl_Cq': 4,
    # F类: 杂原子节点
    'Heterocyclic_N': 3,
    'Amine_Nitrogen': 3,
    'Hydroxyl_O': 1,
    'Ether_O': 2,
    'Heterocyclic_S': 2,
    'Thioether_S': 2,
    'Halogen_X': 1,
}

def calculate_actual_atom_counts_for_su(mol: Chem.Mol, atom_indices: List[int]) -> torch.Tensor:
    """
    为给定的原子索引列表计算精确的原子数。
    假定传入的mol对象已经调用过AddHs()。
    """
    counts = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'X': 0}

    total_h_in_su = 0
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()

        # 统计与该重原子相连的显式氢原子数量（AddHs 之后为显式氢）
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'H':
                total_h_in_su += 1

        # 累加重原子本身的计数
        if symbol in counts:
            counts[symbol] += 1
        elif symbol in ['F', 'Cl', 'Br', 'I']:
            counts['X'] += 1

    # 将累加的总氢数赋值给H
    counts['H'] = total_h_in_su

    return torch.tensor([counts['C'], counts['H'], counts['O'], counts['N'], counts['S'], counts['X']], dtype=torch.float)

SU_PPM_RANGES = {
    # A类: 核心官能团
    'Amide_Group': (150, 210),
    'Carboxylic_Acid': (150, 210),
    'Ester_Group': (150, 230),
    # B类: 其他单原子官能团
    'Aldehyde_Ketone_C': (150, 240),
    'Nitrile_C': (100, 150),
    # C类: 芳香碳
    'O_Substituted_Aro_C': (126, 165),
    'N_Substituted_Aro_C': (126, 165),
    'S_Substituted_Aro_C': (126, 165),
    'X_Substituted_Aro_C': (126, 165),
    'Keto_Substituted_Aro_C': (139, 165),
    'Aryl_Substituted_Aro_C': (134, 165),
    'Alkyl_Substituted_Aro_C':(139, 165),
    'Aromatic_Bridgehead_C': (126, 142),
    'Carbocyclic_Aro_CH':  (100, 131),
    # D类: 链状不饱和碳
    'Vinyllic_Cq': (110, 180),
    'Vinyllic_CH': (110, 180),
    'Vinyllic_CH2':(110, 180),
    'Alkynyl_Cq':  (55, 110),
    'Alkynyl_CH':  (55, 110),
    # E类: 饱和脂肪碳
    'Alcohol_Ether_C': (10, 100),
    'Amine_C': (35, 85),
    'Halogenated_C': (20, 80),
    'Alkyl_CH3': (0, 25),
    'Alkyl_CH2': (18, 45),
    'Alkyl_CH': (28, 60),
    'Alkyl_Cq': (15, 60),
    # F类: 杂原子节点
    'Heterocyclic_N': (-1, 1),
    'Amine_Nitrogen': (-1, 1),
    'Hydroxyl_O': (-1, 1),
    'Ether_O': (-1, 1),
    'Heterocyclic_S': (-1, 1),
    'Thioether_S': (-1, 1),
    'Halogen_X': (-1, 1),
}


def get_su_ppm_range(su_type: int) -> tuple[float, float]:
    """Return the authoritative coarse ppm range for an SU type."""
    su_i = int(su_type)
    if su_i < 0 or su_i >= len(SU_DEFS):
        raise ValueError(f"invalid SU type: {su_i}")
    su_name = str(SU_DEFS[su_i][0])
    if su_name not in SU_PPM_RANGES:
        raise KeyError(f"missing coarse ppm range for SU{su_i} ({su_name})")
    low, high = SU_PPM_RANGES[su_name]
    return float(low), float(high)

_ppm_list = []
for name, _ in SU_DEFS:
    low, high = SU_PPM_RANGES.get(name, (0, 300))
    # 对 Halogen_X 维持极窄范围 (0~1 ppm)，不再额外扩展
    if name == 'Halogen_X':
        _ppm_list.append((low, high))
        continue
    low = max(0, low - 1)
    high = min(300, high + 1)
    _ppm_list.append((low, high))

SU_PPM_MIN_MAX = torch.tensor(_ppm_list, dtype=torch.float)


# ==========================================================
# 核心常量: SU -> 元素组成矩阵 (C, H, O, N, S, X)
# ==========================================================
E_SU = torch.tensor([
    # A类: 核心官能团
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # 0: Amide_Group (-CONH-), H on N is assumed 1 avg.
    [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],  # 1: Carboxylic_Acid (-COOH)
    [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],  # 2: Ester_Group (-COO-)
    # B类: 其他单原子官能团
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 3: Aldehyde_Ketone_C (>C=O), avg H of aldehyde(1) and ketone(0)
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 4: Nitrile_C (-C≡N)
    # C类: 芳香碳
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 5: O_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6: N_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 7: S_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 8: X_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 9: Keto_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10: Aryl_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 11: Alkyl_Substituted_Aro_C, SU is just the carbon
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 12: Aromatic_Bridgehead_C
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 13: Carbocyclic_Aro_CH
    # D类: 链状不饱和碳
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 14: Vinyllic_Cq
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 15: Vinyllic_CH
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 16: Vinyllic_CH2
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 17: Alkynyl_Cq
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 18: Alkynyl_CH
    # E类: 饱和脂肪碳
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 19: Alcohol_Ether_C, sp3 C attached to O/S (e.g. -O/S-CH2-)
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 20: Amine_C, similar avg H to Alcohol_Ether_C
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 21: Halogenated_C, avg H of CH3(3), CH2(2), CH(1) -> ~1.5-2H
    [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],  # 22: Alkyl_CH3
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 23: Alkyl_CH2
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 24: Alkyl_CH
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 25: Alkyl_Cq
    # F类: 杂原子节点
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 26: Heterocyclic_N, aromatic ring N
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # 27: Amine_Nitrogen, amino-like neutral trivalent N
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 28: Hydroxyl_O (-OH)
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 29: Ether_O, ordinary ether O and aromatic ring O
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 30: Heterocyclic_S, aromatic ring S
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 31: Thioether_S, ordinary sulfide S
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 32: Halogen_X
], dtype=torch.float)

# ===== 统一绘图风格参数 =====
NODE_RADIUS = 0.35           # 节点半径 (固定值，基于坐标单位)
NODE_ALPHA = 0.9             # 节点透明度
FONT_SIZE = 9                # 字体大小 (固定)
FONT_COLOR = 'white'         # 字体颜色
EDGE_COLOR = (0.5, 0.5, 0.5) # 统一边颜色（灰色）
EDGE_LW = 1.5                # 统一边线宽
EDGE_LENGTH = 1.0            # 目标边长度 (固定)
PLOT_ROTATION_DEGREES = -90.0
LABEL_ROTATION_DEGREES = -90.0
HEX_DIRECTIONS = ((1, 1), (1, 0), (-1, 0), (-1, -1), (0, 1), (0, -1))
HEX_VERTEX_OFFSETS = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))

def _node_color(su: int):
    """
    根据结构单元类型返回节点颜色。
    """
    color_map = {
        0: (0.70, 0.10, 0.40),
        1: (0.85, 0.10, 0.30),
        2: (0.75, 0.15, 0.50),
        3: (0.60, 0.05, 0.25),
        4: (0.45, 0.05, 0.55),
        5: (0.20, 0.55, 0.90),
        6: (0.15, 0.35, 0.80),
        7: (0.25, 0.65, 0.85),
        8: (0.55, 0.70, 0.85),
        9: (0.10, 0.25, 0.65),
        10: (0.20, 0.50, 1.00),
        11: (0.30, 0.70, 0.95),
        12: (0.05, 0.15, 0.55),
        13: (0.45, 0.80, 1.00),
        14: (0.10, 0.75, 0.45),
        15: (0.15, 0.85, 0.55),
        16: (0.20, 0.90, 0.65),
        17: (0.05, 0.55, 0.35),
        18: (0.10, 0.65, 0.40),
        19: (0.95, 0.50, 0.20),
        20: (0.90, 0.65, 0.35),
        21: (0.80, 0.75, 0.50),
        22: (0.95, 0.75, 0.15),
        23: (0.90, 0.60, 0.10),
        24: (0.85, 0.50, 0.05),
        25: (0.75, 0.40, 0.05),
        26: (0.40, 0.10, 0.85),
        27: (0.60, 0.20, 0.90),
        28: (1.00, 0.20, 0.20),
        29: (0.80, 0.05, 0.05),
        30: (0.85, 0.80, 0.05),
        31: (0.70, 0.65, 0.00),
        32: (0.55, 0.55, 0.55),
    }
    return color_map.get(su, (0.6, 0.6, 0.6))


def _hex_axial_to_cart(q: float, r: float, scale: float = 1.0) -> Tuple[float, float]:
    sqrt3_half = math.sqrt(3.0) / 2.0
    return scale * sqrt3_half * q, scale * (r - 0.5 * q)


def _rotate_positions(pos: Dict[int, Tuple[float, float]],
                      degrees: float) -> Dict[int, Tuple[float, float]]:
    if not pos or abs(float(degrees)) < 1e-9:
        return dict(pos)

    cx = 0.5 * (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values()))
    cy = 0.5 * (min(y for _, y in pos.values()) + max(y for _, y in pos.values()))
    theta = math.radians(float(degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rotated = {}
    for nid, (x, y) in pos.items():
        dx = x - cx
        dy = y - cy
        rotated[nid] = (
            cx + dx * cos_t - dy * sin_t,
            cy + dx * sin_t + dy * cos_t,
        )
    return rotated


def _hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    aq, ar = a
    bq, br = b
    return int((abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) / 2)


def _compute_rdkit_su_positions(mol, su_matches) -> Dict[int, Tuple[float, float]]:
    conf = mol.GetConformer()
    atom_to_su_idx = [-1] * mol.GetNumAtoms()
    for su_idx, su_info in enumerate(su_matches):
        for atom_idx in su_info['atoms']:
            atom_to_su_idx[atom_idx] = su_idx

    boundary_atoms_per_su = {i: set() for i in range(len(su_matches))}
    for bond in mol.GetBonds():
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        su_a = atom_to_su_idx[a_idx]
        su_b = atom_to_su_idx[b_idx]
        if su_a != -1 and su_b != -1 and su_a != su_b:
            boundary_atoms_per_su[su_a].add(a_idx)
            boundary_atoms_per_su[su_b].add(b_idx)

    pos = {}
    for i, su_info in enumerate(su_matches):
        atom_indices = list(boundary_atoms_per_su[i]) or list(su_info['atoms'])
        coords = [conf.GetAtomPosition(j) for j in atom_indices]
        centroid_x = sum(c.x for c in coords) / len(coords)
        centroid_y = sum(c.y for c in coords) / len(coords)
        pos[i] = (centroid_x, centroid_y)
    return pos


def _direction_similarity(rdkit_vec: Tuple[float, float],
                          hex_dir: Tuple[int, int],
                          occupied: bool,
                          parent_axial: Tuple[int, int],
                          candidate_axial: Tuple[int, int]) -> float:
    hx, hy = _hex_axial_to_cart(*hex_dir)
    rx, ry = rdkit_vec
    r_norm = math.hypot(rx, ry)
    h_norm = math.hypot(hx, hy)
    if r_norm <= 1e-8 or h_norm <= 1e-8:
        score = 0.0
    else:
        score = (rx * hx + ry * hy) / (r_norm * h_norm)

    if occupied:
        score -= 3.0
    score -= 0.03 * _hex_distance((0, 0), candidate_axial)
    score -= 0.01 * _hex_distance(parent_axial, candidate_axial)
    return score


def _dir_between(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return b[0] - a[0], b[1] - a[1]


def _turn_dirs(direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    if direction not in HEX_VERTEX_OFFSETS:
        return list(HEX_DIRECTIONS)
    idx = HEX_VERTEX_OFFSETS.index(direction)
    return [HEX_VERTEX_OFFSETS[(idx - 1) % 6], HEX_VERTEX_OFFSETS[(idx + 1) % 6]]


def _score_hex_candidate(node: int,
                         candidate: Tuple[int, int],
                         placed_neighbors: List[int],
                         axial_pos: Dict[int, Tuple[int, int]],
                         rdkit_pos: Dict[int, Tuple[float, float]]) -> float:
    score = 0.03 * _hex_distance((0, 0), candidate)
    nx, ny = rdkit_pos.get(node, (0.0, 0.0))

    for nb in placed_neighbors:
        nb_axial = axial_pos[nb]
        grid_dist = _hex_distance(candidate, nb_axial)
        score += 10.0 * abs(grid_dist - 1)

        bx, by = rdkit_pos.get(nb, (nx, ny))
        hex_delta = (candidate[0] - nb_axial[0], candidate[1] - nb_axial[1])
        direction_score = _direction_similarity(
            (nx - bx, ny - by),
            hex_delta,
            False,
            nb_axial,
            candidate,
        )
        score -= direction_score

    return score


def _candidate_entries_for_side_node(node: int,
                                     placed_neighbors: List[int],
                                     axial_pos: Dict[int, Tuple[int, int]],
                                     blocked_axials: Set[Tuple[int, int]],
                                     ring_exterior_dirs: Dict[int, Set[Tuple[int, int]]],
                                     incoming_dirs: Dict[int, Tuple[int, int]],
                                     rdkit_pos: Dict[int, Tuple[float, float]]) -> List[Tuple[float, Tuple[int, int], int, Tuple[int, int]]]:
    entries = []

    ring_parents = [nb for nb in placed_neighbors if nb in ring_exterior_dirs]
    for parent in ring_parents:
        pq, pr = axial_pos[parent]
        for d in sorted(ring_exterior_dirs[parent]):
            cand = (pq + d[0], pr + d[1])
            if cand in blocked_axials:
                continue
            score = _score_hex_candidate(node, cand, placed_neighbors, axial_pos, rdkit_pos)
            score -= 5.0
            entries.append((score, cand, parent, d))

    if entries:
        entries.sort(key=lambda item: item[0])
        return entries

    chain_parents = [nb for nb in placed_neighbors if nb in incoming_dirs]
    for parent in chain_parents:
        pq, pr = axial_pos[parent]
        for d in _turn_dirs(incoming_dirs[parent]):
            cand = (pq + d[0], pr + d[1])
            if cand in blocked_axials:
                continue
            score = _score_hex_candidate(node, cand, placed_neighbors, axial_pos, rdkit_pos)
            score -= 2.0
            entries.append((score, cand, parent, d))

    if entries:
        entries.sort(key=lambda item: item[0])
        return entries

    for parent in placed_neighbors:
        pq, pr = axial_pos[parent]
        for d in HEX_DIRECTIONS:
            cand = (pq + d[0], pr + d[1])
            if cand in blocked_axials:
                continue
            score = _score_hex_candidate(node, cand, placed_neighbors, axial_pos, rdkit_pos)
            entries.append((score, cand, parent, d))

    entries.sort(key=lambda item: item[0])
    return entries


def _build_atom_to_su_idx(su_matches, num_atoms: int) -> List[int]:
    atom_to_su_idx = [-1] * num_atoms
    for su_idx, su_info in enumerate(su_matches):
        for atom_idx in su_info['atoms']:
            atom_to_su_idx[atom_idx] = su_idx
    return atom_to_su_idx


def _ordered_ring_atoms(mol, ring_atoms: Tuple[int, ...]) -> List[int]:
    ring_set = set(int(a) for a in ring_atoms)
    adj = {}
    for atom_idx in ring_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        adj[atom_idx] = sorted(
            nbr.GetIdx() for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in ring_set
        )

    if not ring_set or any(len(nbs) < 2 for nbs in adj.values()):
        return list(ring_atoms)

    start = min(ring_set)
    path = [start]
    prev = None
    cur = start
    while len(path) < len(ring_set):
        candidates = [n for n in adj[cur] if n != prev and n not in path]
        if not candidates:
            return list(ring_atoms)
        nxt = candidates[0]
        path.append(nxt)
        prev, cur = cur, nxt

    if start not in adj[cur]:
        return list(ring_atoms)
    return path


def _extract_hex_ring_records(mol, su_matches, rdkit_pos: Dict[int, Tuple[float, float]]) -> List[Dict]:
    atom_to_su_idx = _build_atom_to_su_idx(su_matches, mol.GetNumAtoms())
    records = []
    for ring_atoms in mol.GetRingInfo().AtomRings():
        if len(ring_atoms) != 6:
            continue

        ordered_atoms = _ordered_ring_atoms(mol, tuple(ring_atoms))
        nodes = [atom_to_su_idx[a] for a in ordered_atoms]
        if any(n < 0 for n in nodes) or len(set(nodes)) != 6:
            continue

        centroid = (
            sum(rdkit_pos[n][0] for n in nodes) / 6.0,
            sum(rdkit_pos[n][1] for n in nodes) / 6.0,
        )
        records.append({
            'atoms': ordered_atoms,
            'nodes': nodes,
            'centroid': centroid,
        })
    return records


def _ring_offsets(rotation: int, reverse: bool) -> List[Tuple[int, int]]:
    if reverse:
        return [HEX_VERTEX_OFFSETS[(rotation - i) % 6] for i in range(6)]
    return [HEX_VERTEX_OFFSETS[(rotation + i) % 6] for i in range(6)]


def _score_ring_orientation(ring: Dict,
                            offsets: List[Tuple[int, int]],
                            center: Tuple[int, int],
                            rdkit_pos: Dict[int, Tuple[float, float]]) -> float:
    cx, cy = ring['centroid']
    score = 0.0
    for node, offset in zip(ring['nodes'], offsets):
        rx, ry = rdkit_pos.get(node, (cx, cy))
        hx, hy = _hex_axial_to_cart(offset[0], offset[1])
        rv = (rx - cx, ry - cy)
        hv = (hx, hy)
        score += _direction_similarity(rv, hv, False, center, (center[0] + offset[0], center[1] + offset[1]))
    return score


def _ring_assignment_candidates(ring: Dict,
                                axial_pos: Dict[int, Tuple[int, int]],
                                occupied_axials: Set[Tuple[int, int]],
                                ring_centers: Set[Tuple[int, int]],
                                rdkit_pos: Dict[int, Tuple[float, float]],
                                default_center: Tuple[int, int],
                                neighbors: Optional[Dict[int, List[int]]] = None) -> List[Tuple[float, Tuple[int, int], List[Tuple[int, int]]]]:
    candidates = []
    nodes = ring['nodes']
    placed = [(i, n) for i, n in enumerate(nodes) if n in axial_pos]
    ring_node_set = set(nodes)

    anchors = []
    if not placed and neighbors:
        for i, node in enumerate(nodes):
            for nb in neighbors.get(node, []):
                if nb in axial_pos and nb not in ring_node_set:
                    anchors.append((i, node, nb))

    for rotation in range(6):
        for reverse in (False, True):
            offsets = _ring_offsets(rotation, reverse)
            implied_centers = set()
            for i, node in placed:
                aq, ar = axial_pos[node]
                oq, or_ = offsets[i]
                implied_centers.add((aq - oq, ar - or_))

            if len(implied_centers) > 1:
                continue

            center_options = []
            if implied_centers:
                center_options.append((next(iter(implied_centers)), 0.0))
            elif anchors:
                for i, node, nb in anchors:
                    nb_axial = axial_pos[nb]
                    nbx, nby = rdkit_pos.get(nb, (0.0, 0.0))
                    nx, ny = rdkit_pos.get(node, (nbx, nby))
                    for d in HEX_DIRECTIONS:
                        coord = (nb_axial[0] + d[0], nb_axial[1] + d[1])
                        offset = offsets[i]
                        center = (coord[0] - offset[0], coord[1] - offset[1])
                        anchor_score = _direction_similarity(
                            (nx - nbx, ny - nby),
                            d,
                            coord in occupied_axials,
                            nb_axial,
                            coord,
                        )
                        center_options.append((center, anchor_score))
            else:
                center_options.append((default_center, 0.0))

            seen_centers = set()
            for center, anchor_score in center_options:
                if center in seen_centers:
                    continue
                seen_centers.add(center)
                if center in ring_centers or center in occupied_axials:
                    continue

                conflict = False
                for node, offset in zip(nodes, offsets):
                    coord = (center[0] + offset[0], center[1] + offset[1])
                    if node in axial_pos:
                        if axial_pos[node] != coord:
                            conflict = True
                            break
                    elif coord in occupied_axials or coord in ring_centers:
                        conflict = True
                        break
                if conflict:
                    continue

                score = _score_ring_orientation(ring, offsets, center, rdkit_pos)
                score += 5.0 * len(placed)
                score += 2.0 * anchor_score
                candidates.append((score, center, offsets))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _place_hex_rings(rings: List[Dict],
                     edge_list: List[List[int]],
                     rdkit_pos: Dict[int, Tuple[float, float]]) -> Tuple[Dict[int, Tuple[int, int]], Set[Tuple[int, int]], Dict[int, Set[Tuple[int, int]]]]:
    axial_pos: Dict[int, Tuple[int, int]] = {}
    occupied_axials: Set[Tuple[int, int]] = set()
    ring_centers: Set[Tuple[int, int]] = set()
    ring_exterior_dirs: Dict[int, Set[Tuple[int, int]]] = {}
    unplaced = set(range(len(rings)))
    component_offset = 0
    neighbors: Dict[int, List[int]] = {}
    for u, v in edge_list:
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    for node in neighbors:
        neighbors[node] = sorted(set(neighbors[node]))

    while unplaced:
        best = None

        for ri in sorted(unplaced):
            ring = rings[ri]
            placed_count = sum(1 for n in ring['nodes'] if n in axial_pos)
            ring_node_set = set(ring['nodes'])
            anchor_count = sum(
                1
                for n in ring['nodes']
                for nb in neighbors.get(n, [])
                if nb in axial_pos and nb not in ring_node_set
            )
            if axial_pos and placed_count == 0 and anchor_count == 0:
                continue

            default_center = (component_offset, 0)
            while default_center in ring_centers or default_center in occupied_axials:
                component_offset += 4
                default_center = (component_offset, 0)

            candidates = _ring_assignment_candidates(
                ring,
                axial_pos,
                occupied_axials,
                ring_centers,
                rdkit_pos,
                default_center,
                neighbors,
            )
            if not candidates:
                continue

            score, center, offsets = candidates[0]
            choice_key = (placed_count, anchor_count, score, -ri)
            if best is None or choice_key > best[0]:
                best = (choice_key, ri, center, offsets)

        if best is None:
            ri = min(unplaced)
            ring = rings[ri]
            default_center = (component_offset, 0)
            while default_center in ring_centers or default_center in occupied_axials:
                component_offset += 4
                default_center = (component_offset, 0)
            candidates = _ring_assignment_candidates(
                ring,
                axial_pos,
                occupied_axials,
                ring_centers,
                rdkit_pos,
                default_center,
                neighbors,
            )
            if not candidates:
                unplaced.remove(ri)
                continue
            _, center, offsets = candidates[0]
        else:
            _, ri, center, offsets = best

        for node, offset in zip(rings[ri]['nodes'], offsets):
            coord = (center[0] + offset[0], center[1] + offset[1])
            axial_pos[node] = coord
            occupied_axials.add(coord)
            ring_exterior_dirs.setdefault(node, set()).add(offset)
        ring_centers.add(center)
        unplaced.remove(ri)
        component_offset = max([q for q, _ in occupied_axials] + [q for q, _ in ring_centers], default=0) + 4

    return axial_pos, ring_centers, ring_exterior_dirs


def _layout_remaining_nodes(edge_list: List[List[int]],
                            rdkit_pos: Dict[int, Tuple[float, float]],
                            num_nodes: int,
                            axial_pos: Dict[int, Tuple[int, int]],
                            ring_centers: Set[Tuple[int, int]],
                            ring_exterior_dirs: Dict[int, Set[Tuple[int, int]]]) -> Dict[int, Tuple[int, int]]:
    neighbors: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in edge_list:
        if u == v:
            continue
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)

    for node in neighbors:
        neighbors[node] = sorted(set(neighbors[node]))

    blocked_axials = set(axial_pos.values()) | set(ring_centers)
    queue = deque(sorted(axial_pos))
    queued = set(queue)
    incoming_dirs: Dict[int, Tuple[int, int]] = {}

    while len(axial_pos) < num_nodes:
        if not queue:
            unplaced = [n for n in range(num_nodes) if n not in axial_pos]
            seed = max(unplaced, key=lambda n: (len(neighbors.get(n, [])), -n))
            seed_center = (max([q for q, _ in blocked_axials], default=0) + 4, 0)
            while seed_center in blocked_axials:
                seed_center = (seed_center[0] + 4, 0)
            axial_pos[seed] = seed_center
            blocked_axials.add(seed_center)
            queue.append(seed)
            queued.add(seed)

        u = queue.popleft()
        ux, uy = rdkit_pos.get(u, (0.0, 0.0))
        candidates_to_place = [v for v in neighbors.get(u, []) if v not in axial_pos]
        candidates_to_place.sort(
            key=lambda v: (
                -len(neighbors.get(v, [])),
                math.atan2(rdkit_pos.get(v, (0.0, 0.0))[1] - uy,
                           rdkit_pos.get(v, (0.0, 0.0))[0] - ux),
                v,
            )
        )

        for v in candidates_to_place:
            placed_neighbors = [nb for nb in neighbors.get(v, []) if nb in axial_pos]
            candidates = _candidate_entries_for_side_node(
                v,
                placed_neighbors,
                axial_pos,
                blocked_axials,
                ring_exterior_dirs,
                incoming_dirs,
                rdkit_pos,
            )
            if candidates:
                _, chosen, parent, direction = candidates[0]
            else:
                chosen = (axial_pos[u][0] + 1, axial_pos[u][1] + 1)
                parent = u
                direction = _dir_between(axial_pos[u], chosen)

            axial_pos[v] = chosen
            if direction in HEX_DIRECTIONS:
                incoming_dirs[v] = direction
            elif parent in axial_pos:
                incoming_dirs[v] = _dir_between(axial_pos[parent], chosen)
            blocked_axials.add(chosen)
            if v not in queued:
                queue.append(v)
                queued.add(v)

    return axial_pos


def _layout_on_hex_grid(edge_list: List[List[int]],
                        rdkit_pos: Dict[int, Tuple[float, float]],
                        num_nodes: int,
                        mol=None,
                        su_matches=None) -> Dict[int, Tuple[float, float]]:
    if num_nodes <= 0:
        return {}

    rings = _extract_hex_ring_records(mol, su_matches, rdkit_pos) if mol is not None and su_matches is not None else []
    axial_pos, ring_centers, ring_exterior_dirs = _place_hex_rings(rings, edge_list, rdkit_pos)
    axial_pos = _layout_remaining_nodes(edge_list, rdkit_pos, num_nodes, axial_pos, ring_centers, ring_exterior_dirs)

    cart_pos = {
        nid: _hex_axial_to_cart(q, r, scale=EDGE_LENGTH)
        for nid, (q, r) in axial_pos.items()
    }

    cx = 0.5 * (min(x for x, _ in cart_pos.values()) + max(x for x, _ in cart_pos.values()))
    cy = 0.5 * (min(y for _, y in cart_pos.values()) + max(y for _, y in cart_pos.values()))
    return {nid: (x - cx, y - cy) for nid, (x, y) in cart_pos.items()}


def visualize_coarse_grained_graph(data, su_matches, mol, output_path,
                                   rotation_degrees: float = PLOT_ROTATION_DEGREES,
                                   label_rotation_degrees: float = LABEL_ROTATION_DEGREES,
                                   node_radius: float = NODE_RADIUS,
                                   show_su_id: bool = True,
                                   font_size: int = FONT_SIZE):
    """
    可视化粗粒度分子图，使用 axial 六角坐标布局。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    node_radius = float(node_radius)
    edge_list = data.edge_index.t().tolist()
    rdkit_pos = _compute_rdkit_su_positions(mol, su_matches)
    scaled_pos = _layout_on_hex_grid(edge_list, rdkit_pos, len(su_matches), mol=mol, su_matches=su_matches)
    scaled_pos = _rotate_positions(scaled_pos, rotation_degrees)

    all_x = [p[0] for p in scaled_pos.values()]
    all_y = [p[1] for p in scaled_pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    margin = node_radius * 3
    data_width = (max_x - min_x) + 2 * margin
    data_height = (max_y - min_y) + 2 * margin

    min_size = EDGE_LENGTH * 3
    data_width = max(data_width, min_size)
    data_height = max(data_height, min_size)

    pixels_per_unit = 80  # 每单位数据对应的像素数
    fig_width = data_width * pixels_per_unit / 100  # 转换为英寸 (dpi=100)
    fig_height = data_height * pixels_per_unit / 100

    fig_width = max(4, min(fig_width, 20))
    fig_height = max(4, min(fig_height, 20))

    fig = Figure(figsize=(fig_width, fig_height), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    for u, v in edge_list:
        if u not in scaled_pos or v not in scaled_pos:
            continue
        x0, y0 = scaled_pos[u]
        x1, y1 = scaled_pos[v]
        ax.plot([x0, x1], [y0, y1],
                color=EDGE_COLOR,
                linewidth=EDGE_LW,
                zorder=1,
                solid_capstyle='round')

    for i, su_info in enumerate(su_matches):
        if i not in scaled_pos:
            continue
        su_id = SU_NAME_TO_ID[su_info['name']]
        x, y = scaled_pos[i]
        color = _node_color(su_id)

        circle = mpatches.Circle((x, y), node_radius,
                                  facecolor=color,
                                  edgecolor='white',
                                  linewidth=0.8,
                                  alpha=NODE_ALPHA,
                                  zorder=3)
        ax.add_patch(circle)

        if show_su_id:
            ax.text(x, y, str(su_id),
                    ha='center', va='center',
                    fontsize=font_size,
                    fontweight='bold',
                    color=FONT_COLOR,
                    rotation=label_rotation_degrees,
                    rotation_mode='anchor',
                    zorder=4)

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(f"{data.smiles}", fontsize=9, pad=8)

    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white',
                    pad_inches=0.1)
    except Exception as e:
        print(f"  -> 警告: 可视化图 '{output_path}' 生成失败: {e}。跳过图片保存。", file=sys.stderr)
    finally:
        plt.close(fig)


def _assign_atoms_to_su(mol: Chem.Mol) -> Tuple[List[Dict], List[int]]:
    """将分子中的原子分配给结构单元(SU)。返回匹配列表和原子->SU索引的映射。"""
    atom_to_su_idx = [-1] * mol.GetNumAtoms()
    su_matches = []

    for su_name, pattern in SU_PATTERNS:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # 检查这个匹配中的所有原子是否都尚未被分配
            if all(atom_to_su_idx[i] == -1 for i in match):
                su_id = len(su_matches)
                su_matches.append({'name': su_name, 'atoms': match})
                for atom_idx in match:
                    atom_to_su_idx[atom_idx] = su_id

    return su_matches, atom_to_su_idx

def smiles_to_coarse_grained_graph(
    smiles_string: str,
    output_path_prefix: Optional[str] = None,
    y_spectrum: Optional[torch.Tensor] = None,
    real_atom_shifts: Optional[Dict[int, Tuple[float, float]]] = None,
    min_su_nodes: int = 6,
    save_visualization: bool = True,
    require_complete_real_shifts: bool = False,
    plot_rotation_degrees: float = PLOT_ROTATION_DEGREES,
    label_rotation_degrees: float = LABEL_ROTATION_DEGREES,
    node_radius: float = NODE_RADIUS,
    show_su_id: bool = True,
    font_size: int = FONT_SIZE,
) -> Optional[Tuple[Data, List[Dict]]]:
    """
    V27: 将一个SMILES字符串转换为一个包含【最终版增强特征】和【修正后边提取逻辑】的粗粒度图信息的PyG Data对象。
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if not mol:
            print(f"  -> 警告: RDKit无法解析SMILES '{smiles_string}'。跳过该分子。")
            return None
    except Exception as e:
        print(f"  -> 警告: 处理SMILES '{smiles_string}' 时发生意外的RDKit错误: {e}。跳过该分子。")
        return None

    # V49: 确保先执行AddHs，让后续所有函数都操作在带H的完整分子上
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    su_matches, atom_to_su_idx = _assign_atoms_to_su(mol)
    if not su_matches:
        print(f"  -> 警告: 分子 '{smiles_string}' 中未能识别出任何结构单元。跳过该分子。")
        raise NoStructureUnitsError(f"未能识别出任何结构单元: {smiles_string}")

    # 检查粗粒度节点数量。默认保留历史规则：只保留节点数量 >= 6 的分子。
    su_count = len(su_matches)
    min_su_nodes = int(min_su_nodes)
    if min_su_nodes > 0 and su_count < min_su_nodes:
        print(f"  -> 跳过: 分子 '{smiles_string}' 的粗粒度节点数量为 {su_count}，少于{min_su_nodes}个节点。")
        raise TooFewNodesError(f"粗粒度节点数量为 {su_count}，少于{min_su_nodes}个节点: {smiles_string}")

    # 检查是否有任何重原子未被分配到SU
    unassigned_heavy_atoms = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetSymbol() != 'H' and atom_to_su_idx[atom.GetIdx()] == -1
    ]
    if unassigned_heavy_atoms:
        print(f"  -> 警告: 分子 '{smiles_string}' 中存在未定义的结构 (例如原子索引: {unassigned_heavy_atoms})。跳过该分子。")
        raise UnassignedAtomsError(f"存在未定义的结构 (原子索引: {unassigned_heavy_atoms}): {smiles_string}")

    if require_complete_real_shifts:
        real_atom_shifts = real_atom_shifts or {}
        carbon_atom_indices = {
            atom.GetIdx() for atom in mol.GetAtoms()
            if atom.GetSymbol() == 'C'
        }
        provided_indices = set(int(idx) for idx in real_atom_shifts.keys())
        missing_indices = sorted(carbon_atom_indices - provided_indices)
        extra_indices = sorted(provided_indices - carbon_atom_indices)
        if missing_indices or extra_indices:
            raise IncompleteShiftMappingError(
                "原子级核磁位移与碳原子索引不能一一对应: "
                f"missing_carbon_indices={missing_indices}, "
                f"extra_noncarbon_indices={extra_indices}, "
                f"smiles={smiles_string}"
            )

    # --- 稳定节点顺序：按每个 SU 所含原子的最小 RDKit 原子索引升序 ---
    # 这样能够与 SMILES/原子索引保持一致，确保核磁标注映射稳定
    order = list(range(su_count))
    order.sort(key=lambda i: (min(su_matches[i]['atoms']), SU_NAME_TO_ID[su_matches[i]['name']]))

    # 构建旧->新索引映射，并据此重排 su_matches
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    su_matches = [su_matches[i] for i in order]

    # 同步更新 atom_to_su_idx，使其指向新的 SU 索引
    for atom_idx, old_su in enumerate(atom_to_su_idx):
        if old_su != -1:
            atom_to_su_idx[atom_idx] = old_to_new_idx_map[old_su]

    # --- 节点特征提取 (在排序后的 su_matches 上进行) ---
    node_features = []
    for su_info in su_matches:
        su_type_id = SU_NAME_TO_ID[su_info['name']]
        one_hot_su_type = F.one_hot(torch.tensor(su_type_id), num_classes=NUM_SU_TYPES).float()

        atom_counts = calculate_actual_atom_counts_for_su(mol, su_info['atoms'])

        # V-Refactored: 节点特征现在是干净的 39 维: 33 (one-hot SU) + 6 (原子数)
        node_feat = torch.cat([
            one_hot_su_type,
            atom_counts,
        ])
        node_features.append(node_feat)

    node_feat_matrix = torch.stack(node_features)

    # --- 构建边特征 (基于原子级 bond, 使用已更新的 SU 索引) ---
    edge_map = {}
    for bond in mol.GetBonds():
        a_idx, b_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        su_u_new, su_v_new = atom_to_su_idx[a_idx], atom_to_su_idx[b_idx]
        if su_u_new == -1 or su_v_new == -1 or su_u_new == su_v_new:
            continue

        edge_tuple = tuple(sorted((su_u_new, su_v_new)))
        if edge_tuple not in edge_map:
            edge_map[edge_tuple] = torch.tensor([
                bond.GetBondTypeAsDouble(),
                1.0 if bond.IsInRing() else 0.0
            ], dtype=torch.float)

    edge_index = torch.tensor(list(edge_map.keys()), dtype=torch.long).t().contiguous() if edge_map else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.stack(list(edge_map.values())) if edge_map else torch.empty((0, 2), dtype=torch.float)

    # --- 提取全局特征 (V49: 基于最终的、已添加H的mol对象) ---
    total_counts = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'X': 0}
    # 直接遍历处理过的分子（包含所有H原子）以获得精确的全局计数
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == 'H':
            total_counts['H'] += 1
        elif symbol == 'C':
            total_counts['C'] += 1
        elif symbol == 'O':
            total_counts['O'] += 1
        elif symbol == 'N':
            total_counts['N'] += 1
        elif symbol == 'S':
            total_counts['S'] += 1
        elif symbol in ['F', 'Cl', 'Br', 'I']:
            total_counts['X'] += 1

    total_atom_counts = torch.tensor([
        total_counts['C'],
        total_counts['H'],
        total_counts['O'],
        total_counts['N'],
        total_counts['S'],
        total_counts['X']
    ], dtype=torch.float)

    # V-Final-Fix: Add max degree information
    su_max_degrees = torch.tensor([SU_MAX_DEGREE.get(su['name'], 4) for su in su_matches], dtype=torch.long)

    # 创建PyG Data对象
    data = Data(x=node_feat_matrix, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles_string

    # data.y 仅做兼容保留，不再用于训练。
    # 该谱图将被下面基于节点信息新生成的谱图所覆盖。
    if y_spectrum is not None:
        # 临时保留输入谱图，主要为了可视化或调试，但不会是最终数据
        data.y_spectrum = y_spectrum
    else:
        data.y_spectrum = torch.zeros(NMR_DATA_POINTS, dtype=torch.float)

    data.total_atom_counts = total_atom_counts
    data.su_max_degrees = su_max_degrees

    # V-Fix (Robustness #6): 为损失函数预先计算好节点类型标签
    data.su_type_indices = torch.tensor([SU_NAME_TO_ID[su['name']] for su in su_matches], dtype=torch.long)

    # V-Refactored: 从39维特征中提取 one-hot 部分
    su_onehot = data.x[:, 0:NUM_SU_TYPES]
    data.su_hist = su_onehot.sum(0)

    # ❷ is_carbon 标记
    data.is_carbon = (data.x[:, :26].sum(-1) > 0)

    # ==========================================================
    # 新增: 将 *真实* 原子级核磁位移/强度映射到 SU 节点级别
    # ==========================================================
    real_ppm_tensor       = torch.zeros(len(su_matches))
    real_intensity_tensor = torch.zeros(len(su_matches))

    if real_atom_shifts:
        for i, su_info in enumerate(su_matches):
            ppm_list, inten_list = [], []
            for atom_idx in su_info['atoms']:
                if atom_idx in real_atom_shifts:
                    ppm_val, inten_val = real_atom_shifts[atom_idx]
                    ppm_list.append(float(ppm_val))
                    inten_list.append(float(inten_val))

            if ppm_list:
                total_inten = sum(inten_list)
                # 强度加权平均位移作为该 SU 的代表 PPM
                weighted_ppm = sum(p * i for p, i in zip(ppm_list, inten_list)) / (total_inten + 1e-8)
                real_ppm_tensor[i]       = weighted_ppm
                real_intensity_tensor[i] = total_inten

    if require_complete_real_shifts:
        carbon_su_indices = [
            i for i, su_info in enumerate(su_matches)
            if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C' for atom_idx in su_info['atoms'])
        ]
        uncovered_su_indices = [
            i for i in carbon_su_indices
            if float(real_intensity_tensor[i].item()) <= 1e-6
        ]
        if uncovered_su_indices:
            raise IncompleteShiftMappingError(
                "部分含碳 SU 节点没有有效核磁位移: "
                f"uncovered_su_indices={uncovered_su_indices}, smiles={smiles_string}"
            )

    # -------- [V3] 将相对强度转换为基于碳计数的“准绝对强度” --------
    # 1. 计算分子中碳原子的总数
    num_carbons = total_atom_counts[0].item() # C is at index 0

    # 2. 计算当前相对强度的总和
    current_total_intensity = real_intensity_tensor.sum()

    # 3. 计算缩放因子，使得缩放后的总强度等于碳原子数
    if current_total_intensity > 1e-6 and num_carbons > 0:
        scaling_factor = num_carbons / current_total_intensity
        real_intensity_tensor = real_intensity_tensor * scaling_factor

    data.real_ppm       = real_ppm_tensor
    data.real_intensity = real_intensity_tensor

    hwhm = 1.0
    hwhm_sq = hwhm**2

    delta_ppm_sq = (PPM_AXIS.unsqueeze(0) - data.real_ppm.unsqueeze(1))**2

    valid_nodes_mask = data.real_intensity > 1e-6
    if valid_nodes_mask.any():
        # [num_valid_nodes, 1] * scalar / ([num_valid_nodes, num_points] + scalar)
        all_peaks = data.real_intensity[valid_nodes_mask].unsqueeze(1) * hwhm_sq / (delta_ppm_sq[valid_nodes_mask] + hwhm_sq)
        # 沿节点维度求和，得到最终谱图
        data.y_spectrum = all_peaks.sum(dim=0)
    else:
        # 如果没有有效的节点，则生成零谱
        data.y_spectrum = torch.zeros(NMR_DATA_POINTS, dtype=torch.float)

    if output_path_prefix:
        if save_visualization:
            visualize_coarse_grained_graph(
                data,
                su_matches,
                mol,
                f"{output_path_prefix}.png",
                rotation_degrees=plot_rotation_degrees,
                label_rotation_degrees=label_rotation_degrees,
                node_radius=node_radius,
                show_su_id=show_su_id,
                font_size=font_size,
            )

        try:
            torch.save(data.to_dict(), f"{output_path_prefix}.pt")
            print(f"  -> 图数据字典已保存到: {output_path_prefix}.pt")
        except Exception as e:
            print(f"警告: 保存图数据文件 '{output_path_prefix}.pt' 失败: {e}")

    return data, su_matches


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='将SMILES分子转换为粗粒度图并可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 手动指定单个SMILES
  python coarse_graph.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --name Aspirin --output_dir ./demo

  # 使用默认测试分子
  python coarse_graph.py --output_dir ./demo

  # 只输出图片，不保存pt文件
  python coarse_graph.py --smiles "c1ccccc1" --name Benzene --no_pt
        """
    )
    parser.add_argument('--smiles', '-s', type=str, default=None,
                        help='输入的SMILES字符串')
    parser.add_argument('--name', '-n', type=str, default='molecule',
                        help='分子名称（用于输出文件名）')
    parser.add_argument('--output_dir', '-o', type=str, default='coarse_graphs',
                        help='输出目录 (默认: coarse_graphs)')
    parser.add_argument('--no_pt', action='store_true',
                        help='不保存.pt文件，只输出可视化图片')
    parser.add_argument('--demo', action='store_true',
                        help='运行内置测试分子演示')
    parser.add_argument('--plot_rotation', type=float, default=PLOT_ROTATION_DEGREES,
                        help='粗粒度图整体旋转角度；正值为逆时针，默认 -90')
    parser.add_argument('--label_rotation', type=float, default=LABEL_ROTATION_DEGREES,
                        help='节点中心 SU ID 的文字旋转角度')
    parser.add_argument('--node_radius', type=float, default=NODE_RADIUS,
                        help='节点圆圈半径，基于六角网格坐标单位')
    parser.add_argument('--font_size', type=int, default=FONT_SIZE,
                        help='节点中心 SU ID 字号')
    parser.add_argument('--hide_su_id', action='store_true',
                        help='隐藏节点中心 SU ID，只展示粗粒度节点颜色和拓扑')

    args = parser.parse_args()

    if not VISUALIZATION_ENABLED:
        print("警告: 可视化库(networkx, matplotlib)未安装，图片将不会被生成。")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}\n")

    def process_smiles(smiles: str, name: str, save_pt: bool = True):
        """处理单个SMILES"""
        print(f"{'='*60}")
        print(f"分子: {name}")
        print(f"SMILES: {smiles}")
        print(f"{'='*60}")

        output_prefix = os.path.join(args.output_dir, name)

        try:
            result = smiles_to_coarse_grained_graph(
                smiles,
                output_path_prefix=output_prefix,
                plot_rotation_degrees=args.plot_rotation,
                label_rotation_degrees=args.label_rotation,
                node_radius=args.node_radius,
                show_su_id=not args.hide_su_id,
                font_size=args.font_size,
            )
            if result:
                graph_data, su_matches_debug = result
                print(f"✅ 转换成功!")
                print(f"   粗粒度节点数: {graph_data.num_nodes}")
                print(f"   粗粒度边数: {graph_data.num_edges}")
                print(f"   总原子组成 (C,H,O,N,S,X): {graph_data.total_atom_counts.tolist()}")
                print(f"\n📊 结构单元(SU)分配:")
                for i, su_info in enumerate(su_matches_debug):
                    su_id = SU_NAME_TO_ID[su_info['name']]
                    print(f"   节点 {i:2d}: SU {su_id:2d} ({su_info['name']:<25})")
                print(f"\n🖼️  图片: {output_prefix}.png")

                if not save_pt:
                    # 删除pt文件
                    pt_path = f"{output_prefix}.pt"
                    if os.path.exists(pt_path):
                        os.remove(pt_path)
                        print(f"   (已跳过.pt文件)")
                else:
                    print(f"📦 数据: {output_prefix}.pt")
                return True
            else:
                print("❌ 转换失败")
                return False
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False

    if args.smiles:
        # 处理用户指定的SMILES
        process_smiles(args.smiles, args.name, save_pt=not args.no_pt)
    elif args.demo or len(sys.argv) == 1:
        # 运行内置测试分子
        test_smiles = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Vanillin": "COC1=C(C=C(C=C1)C=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Naphthalene": "c1ccc2ccccc2c1",
            "Biphenyl": "c1ccccc1-c1ccccc1",
        }

        print("运行内置测试分子演示...\n")
        for name, smiles in test_smiles.items():
            process_smiles(smiles, name, save_pt=not args.no_pt)
            print()
    else:
        parser.print_help()
