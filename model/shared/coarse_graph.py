import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from typing import List, Dict, Tuple, Optional
import os
import sys
import networkx as nx
import numpy as np
import re
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
    ('Alcohol_Ether_C', '[CX4;$(C-[O,o])]'),
    ('Amine_C', '[CX4;$(C-[N,n])]'),
    ('Halogenated_C', '[CX4;$(C-[F,Cl,Br,I])]'),
    ('Alkyl_CH3', '[CX4H3]'),
    ('Alkyl_CH2', '[CX4H2]'),
    ('Alkyl_CH', '[CX4H1]'),
    ('Alkyl_Cq', '[CX4H0]'),
    
    # F类: 杂原子节点 (紫色)（26-32）
    ('Heterocyclic_N', '[n;+0;!$(nC=O)]'),
    ('Amine_Nitrogen', '[N;v3;!$(NC=O)]'),
    ('Hydroxyl_O', '[OX2H;$(O-c),$(O-C)]'),
    ('Ether_O', '[O;X2;!H1;!$([O]-C(=O)O)]'),
    ('Heterocyclic_S', '[s;r]'),
    ('Thioether_S', '[S;X2]'),
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
    'Alcohol_Ether_C': (40, 100),
    'Amine_C': (35, 85),
    'Halogenated_C': (20, 80),
    'Alkyl_CH3': (8, 25),
    'Alkyl_CH2': (10, 45),
    'Alkyl_CH': (15, 60),
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
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 19: Alcohol_Ether_C, avg H of >CHOH(1), -CH2OH(2), >C(OH)-(0) -> ~1H
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 20: Amine_C, similar avg H to Alcohol_Ether_C
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 21: Halogenated_C, avg H of CH3(3), CH2(2), CH(1) -> ~1.5-2H
    [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],  # 22: Alkyl_CH3
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # 23: Alkyl_CH2
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 24: Alkyl_CH
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 25: Alkyl_Cq
    # F类: 杂原子节点
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 26: Heterocyclic_N (e.g., in pyridine, no H on N)
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # 27: Amine_Nitrogen, avg H of RNH2(2), R2NH(1), R3N(0) -> ~1H
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 28: Hydroxyl_O (-OH)
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 29: Ether_O (-O-)
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 30: Heterocyclic_S
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 31: Thioether_S (-S-)
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 32: Halogen_X
], dtype=torch.float)

# ===== 统一绘图风格参数 (与 mcts_visualization.py 保持一致) =====
NODE_RADIUS = 0.22           # 节点半径 (固定值，基于坐标单位)
NODE_ALPHA = 0.9             # 节点透明度
FONT_SIZE = 7                # 字体大小 (固定)
FONT_COLOR = 'white'         # 字体颜色
EDGE_COLOR = (0.5, 0.5, 0.5) # 统一边颜色（灰色）
EDGE_LW = 1.5                # 统一边线宽
EDGE_LENGTH = 1.0            # 目标边长度 (固定)

def _node_color(su: int):
    """
    根据结构单元类型返回节点颜色 (与 mcts_visualization.py 一致)
    
    颜色方案：
    - 芳香结构(5-13)：蓝绿色系
    - 脂肪结构(19-25)：橙黄色系  
    - 氧结构(28,29)：红色系
    - 氮结构(27)：紫色系
    - 硫结构(31)：黄色系
    - 卤素结构(32)：灰色系
    - 羰基结构(0-4)：深红色系
    - 不饱和结构(14-18)：青色系
    """
    # 芳香结构单元 (5-13)
    if su == 12:  # Aromatic_Bridgehead_C
        return (0.9, 0.2, 0.2)
    elif su == 10:  # Aryl_Substituted_Aro_C
        return (0.2, 0.4, 0.9)
    elif su == 11:  # Alkyl_Substituted_Aro_C
        return (0.2, 0.8, 0.4)
    elif su == 13:  # Carbocyclic_Aro_CH
        return (0.4, 0.7, 0.9)
    elif su == 5:   # O_Substituted_Aro_C
        return (0.8, 0.3, 0.6)
    elif su == 6:   # N_Substituted_Aro_C
        return (0.6, 0.2, 0.8)
    elif su == 7:   # S_Substituted_Aro_C
        return (0.9, 0.8, 0.1)
    elif su == 8:   # X_Substituted_Aro_C
        return (0.5, 0.5, 0.5)
    elif su == 9:   # Keto_Substituted_Aro_C
        return (0.7, 0.1, 0.1)
    # 脂肪结构单元 (19-25)
    elif su in (23, 24, 25):  # Alkyl_CH2, Alkyl_CH, Alkyl_Cq
        return (0.9, 0.6, 0.2)
    elif su == 22:  # Alkyl_CH3
        return (0.8, 0.5, 0.1)
    elif su == 19:  # Alcohol_Ether_C
        return (0.9, 0.4, 0.4)
    elif su == 20:  # Amine_C
        return (0.7, 0.5, 0.9)
    elif su == 21:  # Halogenated_C
        return (0.6, 0.6, 0.6)
    # 氧结构单元 (28, 29)
    elif su == 28:  # Hydroxyl_O
        return (1.0, 0.2, 0.2)
    elif su == 29:  # Ether_O
        return (0.8, 0.0, 0.0)
    # 氮结构单元 (27)
    elif su == 27:  # Amine_Nitrogen
        return (0.5, 0.0, 0.8)
    # 硫结构单元 (31)
    elif su == 31:  # Thioether_S
        return (0.9, 0.9, 0.0)
    # 卤素结构单元 (32)
    elif su == 32:  # Halogen_X
        return (0.4, 0.4, 0.4)
    # 羰基结构单元 (0-4)
    elif su == 0:   # Amide_Group
        return (0.6, 0.0, 0.3)
    elif su == 1:   # Carboxylic_Acid
        return (0.8, 0.0, 0.2)
    elif su == 2:   # Ester_Group
        return (0.7, 0.1, 0.4)
    elif su == 3:   # Aldehyde_Ketone_C
        return (0.5, 0.0, 0.2)
    elif su == 4:   # Nitrile_C
        return (0.4, 0.0, 0.4)
    # 不饱和结构单元 (14-18)
    elif su == 14:  # Vinyllic_Cq
        return (0.0, 0.6, 0.8)
    elif su == 15:  # Vinyllic_CH
        return (0.0, 0.7, 0.9)
    elif su == 16:  # Vinyllic_CH2
        return (0.1, 0.8, 1.0)
    elif su == 17:  # Alkynyl_Cq
        return (0.0, 0.4, 0.6)
    elif su == 18:  # Alkynyl_CH
        return (0.0, 0.5, 0.7)
    # 杂原子结构单元 (26, 30)
    elif su == 26:  # Heterocyclic_N
        return (0.3, 0.0, 0.9)
    elif su == 30:  # Heterocyclic_S
        return (0.9, 0.7, 0.0)
    # 默认颜色
    else:
        return (0.6, 0.6, 0.6)


def visualize_coarse_grained_graph(data, su_matches, mol, output_path):
    """
    可视化粗粒度分子图 (MCTS 风格 + RDKit 布局)

    设计目标：
    - 完全使用 RDKit 计算的 2D 坐标，不再用 spring 布局或 PCA 旋转，
      这样苯环 / 杂环的几何形状与 RDKit 原图保持一致，不会被拉直或扭曲。
    - 仅做整体的等比例缩放 + 平移，以适配画布大小。
    - 节点大小和文字字号固定，与分子大小无关。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ---------- 1. 从 RDKit 2D 坐标计算 SU 质心 ----------
    conf = mol.GetConformer()

    # 1.1 构建原子 -> SU 的映射
    atom_to_su_idx = [-1] * mol.GetNumAtoms()
    for su_idx, su_info in enumerate(su_matches):
        for atom_idx in su_info['atoms']:
            atom_to_su_idx[atom_idx] = su_idx

    # 1.2 统计每个 SU 的“边界原子”（即与其他 SU 相连的原子）
    boundary_atoms_per_su = {i: set() for i in range(len(su_matches))}
    for bond in mol.GetBonds():
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        su_a = atom_to_su_idx[a_idx]
        su_b = atom_to_su_idx[b_idx]
        if su_a != -1 and su_b != -1 and su_a != su_b:
            boundary_atoms_per_su[su_a].add(a_idx)
            boundary_atoms_per_su[su_b].add(b_idx)

    # 1.3 以“边界原子”的几何中心作为 SU 坐标（若没有边界原子则退化为全部原子中心）
    pos = {}
    for i, su_info in enumerate(su_matches):
        atom_indices = list(boundary_atoms_per_su[i]) or list(su_info['atoms'])
        coords = [conf.GetAtomPosition(j) for j in atom_indices]
        centroid_x = sum(c.x for c in coords) / len(coords)
        centroid_y = sum(c.y for c in coords) / len(coords)
        pos[i] = (centroid_x, centroid_y)

    edge_list = data.edge_index.t().tolist()

    # ---------- 2. 计算当前 SU 间平均边长，用于等比例缩放 ----------
    edge_lengths = []
    for u, v in edge_list:
        if u in pos and v in pos:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            edge_lengths.append(np.sqrt(dx * dx + dy * dy))

    if edge_lengths:
        avg_edge_len = float(np.mean(edge_lengths))
        scale_factor = EDGE_LENGTH / avg_edge_len if avg_edge_len > 1e-6 else 1.0
    else:
        scale_factor = 1.0

    # 等比例缩放 + 平移到以 (0,0) 为中心
    scaled_pos = {}
    for nid, (x, y) in pos.items():
        scaled_pos[nid] = (x * scale_factor, y * scale_factor)

    all_x = [p[0] for p in scaled_pos.values()]
    all_y = [p[1] for p in scaled_pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    for nid in scaled_pos:
        x, y = scaled_pos[nid]
        scaled_pos[nid] = (x - cx, y - cy)

    # 重新计算边界
    all_x = [p[0] for p in scaled_pos.values()]
    all_y = [p[1] for p in scaled_pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # ---------- 3. 根据分子大小设定画布尺寸，保持节点视觉大小基本恒定 ----------
    margin = NODE_RADIUS * 3
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

    # ---------- 4. 绘制 ----------
    fig = Figure(figsize=(fig_width, fig_height), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # 边：统一颜色 / 线宽，保持 RDKit 拓扑
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

    # 节点：固定半径 + 固定字号的 SU ID
    for i, su_info in enumerate(su_matches):
        if i not in scaled_pos:
            continue
        su_id = SU_NAME_TO_ID[su_info['name']]
        x, y = scaled_pos[i]
        color = _node_color(su_id)

        circle = mpatches.Circle((x, y), NODE_RADIUS,
                                  facecolor=color,
                                  edgecolor='white',
                                  linewidth=0.8,
                                  alpha=NODE_ALPHA,
                                  zorder=3)
        ax.add_patch(circle)

        ax.text(x, y, str(su_id),
                ha='center', va='center',
                fontsize=FONT_SIZE,
                fontweight='bold',
                color=FONT_COLOR,
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

def smiles_to_coarse_grained_graph(smiles_string: str, output_path_prefix: Optional[str] = None, y_spectrum: Optional[torch.Tensor] = None, real_atom_shifts: Optional[Dict[int, Tuple[float, float]]] = None) -> Optional[Tuple[Data, List[Dict]]]:
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

    # 检查粗粒度节点数量，只保留节点数量 >= 6 的分子
    su_count = len(su_matches)
    if su_count < 6:
        print(f"  -> 跳过: 分子 '{smiles_string}' 的粗粒度节点数量为 {su_count}，少于6个节点。")
        raise TooFewNodesError(f"粗粒度节点数量为 {su_count}，少于6个节点: {smiles_string}")

    # 检查是否有任何重原子未被分配到SU
    unassigned_heavy_atoms = [
        atom.GetIdx() for atom in mol.GetAtoms() 
        if atom.GetSymbol() != 'H' and atom_to_su_idx[atom.GetIdx()] == -1
    ]
    if unassigned_heavy_atoms:
        print(f"  -> 警告: 分子 '{smiles_string}' 中存在未定义的结构 (例如原子索引: {unassigned_heavy_atoms})。跳过该分子。")
        raise UnassignedAtomsError(f"存在未定义的结构 (原子索引: {unassigned_heavy_atoms}): {smiles_string}")

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
        visualize_coarse_grained_graph(data, su_matches, mol, f"{output_path_prefix}.png")

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
            result = smiles_to_coarse_grained_graph(smiles, output_path_prefix=output_prefix)
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