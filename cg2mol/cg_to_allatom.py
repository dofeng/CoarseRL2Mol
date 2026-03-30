#!/usr/bin/env python3
"""
cg_to_allatom.py - 将简化的粗粒度分子转化为全原子分子

输入：简化后的粗粒度JSON（nodes + bonds）
输出：全原子JSON（atoms + bonds）

每个SU的全原子形式：
- 0 : C(=O)-NH  酰胺，C连接芳香侧，N连接脂肪侧(6,20)
- 1 : COOH      羧酸
- 2 : C(=O)-O   酯基，C连接芳香侧，O连接脂肪侧(5,19)
- 3 : C(=O)     羰基（醛酮）
- 4 : C≡N       氰基
- 5-12 : C      单个碳（芳香碳）
- 13 : CH       芳香CH
- 14 : >C=      双键碳（四取代）
- 15 : -HC=     双键碳（三取代）
- 16 : =CH2     双键末端
- 17 : -C≡      三键碳
- 18 : ≡CH      三键末端
- 19-21,23 : CH2 亚甲基
- 22 : CH3      甲基
- 24 : CH       次甲基
- 25 : C        季碳
- 26 : N        氮
- 27 : NH       仲胺
- 28 : OH       羟基
- 29 : O        醚氧
- 30 : S        硫醚
- 31 : S        硫醚
- 32 : X        卤素(默认Cl)
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AtomExpansion:
    """SU展开为全原子的定义"""
    atoms: List[Tuple[str, str]]  # [(元素符号, 原子类型描述), ...]
    internal_bonds: List[Tuple[int, int, str]]  # [(原子i, 原子j, 键类型), ...]
    # 连接点映射：邻居SU类型集合 -> 本SU中用于连接的原子索引
    # 默认连接点是索引0
    # 对于有方向性的SU（如酰胺），需要根据邻居类型选择连接点
    connect_point: int = 0  # 默认连接点
    # 特殊连接规则：{邻居SU类型: 本SU连接原子索引}
    directional_connects: Dict[int, int] = field(default_factory=dict)
    # 坐标偏移（相对于中心原子）
    coord_offsets: List[Tuple[float, float]] = field(default_factory=list)


# 键类型常量
SINGLE = "SINGLE"
DOUBLE = "DOUBLE"
TRIPLE = "TRIPLE"
AROMATIC = "AROMATIC"

# ============================================================
# SU全原子展开定义
# ============================================================

SU_EXPANSIONS: Dict[int, AtomExpansion] = {}

# SU-0: C(=O)-NH 酰胺
# 结构: C(=O)-N(-H)
# 原子: 0=C, 1=O(=O), 2=N, 3=H
# C连接芳香侧(5-13)，N连接脂肪侧(6,19,20,21,23等)
SU_EXPANSIONS[0] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("N", "N-H"), ("H", "H-N")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE), (2, 3, SINGLE)],
    connect_point=0,  # 默认C连接
    directional_connects={
        # N连接的邻居类型
        6: 2, 19: 2, 20: 2, 21: 2, 23: 2, 24: 2, 25: 2,
    },
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.5, 0), (-0.8, 0.3)]
)

# SU-1: COOH 羧酸
# 结构: C(=O)-O-H
# 原子: 0=C, 1=O(=O), 2=O(-H), 3=H
SU_EXPANSIONS[1] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("O", "O-H"), ("H", "H-O")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE), (2, 3, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.3, -0.5), (-0.6, -0.8)]
)

# SU-2: C(=O)-O 酯基
# 结构: C(=O)-O-
# 原子: 0=C, 1=O(=O), 2=O
# C连接芳香侧，O连接脂肪侧(5,19等)
SU_EXPANSIONS[2] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O"), ("O", "O-")],
    internal_bonds=[(0, 1, DOUBLE), (0, 2, SINGLE)],
    connect_point=0,  # 默认C连接
    directional_connects={
        5: 2, 19: 2, 20: 2, 21: 2, 23: 2, 24: 2, 25: 2,
    },
    coord_offsets=[(0, 0), (0.3, 0.5), (-0.5, 0)]
)

# SU-3: C(=O) 羰基
# 原子: 0=C, 1=O
SU_EXPANSIONS[3] = AtomExpansion(
    atoms=[("C", "C=O"), ("O", "=O")],
    internal_bonds=[(0, 1, DOUBLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0, 0.5)]
)

# SU-4: C≡N 氰基
# 原子: 0=C, 1=N
SU_EXPANSIONS[4] = AtomExpansion(
    atoms=[("C", "C≡N"), ("N", "≡N")],
    internal_bonds=[(0, 1, TRIPLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.5, 0)]
)

# SU-5 到 SU-12: 单个芳香碳 C
for su_id in range(5, 13):
    SU_EXPANSIONS[su_id] = AtomExpansion(
        atoms=[("C", "Ar-C")],
        internal_bonds=[],
        connect_point=0,
        coord_offsets=[(0, 0)]
    )

# SU-13: 芳香CH
# 原子: 0=C, 1=H
SU_EXPANSIONS[13] = AtomExpansion(
    atoms=[("C", "Ar-CH"), ("H", "H-Ar")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-14: >C= 双键碳（四取代）
# 原子: 0=C
SU_EXPANSIONS[14] = AtomExpansion(
    atoms=[("C", ">C=")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-15: -HC= 双键碳（三取代）
# 原子: 0=C, 1=H
SU_EXPANSIONS[15] = AtomExpansion(
    atoms=[("C", "-HC="), ("H", "H-C=")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-16: =CH2 双键末端
# 原子: 0=C, 1=H, 2=H
SU_EXPANSIONS[16] = AtomExpansion(
    atoms=[("C", "=CH2"), ("H", "H-C"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0.3), (0.3, -0.3)]
)

# SU-17: -C≡ 三键碳
# 原子: 0=C
SU_EXPANSIONS[17] = AtomExpansion(
    atoms=[("C", "-C≡")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-18: ≡CH 三键末端
# 原子: 0=C, 1=H
SU_EXPANSIONS[18] = AtomExpansion(
    atoms=[("C", "≡CH"), ("H", "H-C≡")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-19, 20, 21, 23: CH2 亚甲基
for su_id in [19, 20, 21, 23]:
    SU_EXPANSIONS[su_id] = AtomExpansion(
        atoms=[("C", "CH2"), ("H", "H-C"), ("H", "H-C")],
        internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE)],
        connect_point=0,
        coord_offsets=[(0, 0), (0.2, 0.35), (0.2, -0.35)]
    )

# SU-22: CH3 甲基
SU_EXPANSIONS[22] = AtomExpansion(
    atoms=[("C", "CH3"), ("H", "H-C"), ("H", "H-C"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE), (0, 2, SINGLE), (0, 3, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.3, 0), (0.15, 0.26), (0.15, -0.26)]
)

# SU-24: CH 次甲基
SU_EXPANSIONS[24] = AtomExpansion(
    atoms=[("C", "CH"), ("H", "H-C")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-25: C 季碳
SU_EXPANSIONS[25] = AtomExpansion(
    atoms=[("C", "C")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-26: N 氮
SU_EXPANSIONS[26] = AtomExpansion(
    atoms=[("N", "N")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-27: NH 仲胺
SU_EXPANSIONS[27] = AtomExpansion(
    atoms=[("N", "NH"), ("H", "H-N")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.4, 0)]
)

# SU-28: OH 羟基
SU_EXPANSIONS[28] = AtomExpansion(
    atoms=[("O", "OH"), ("H", "H-O")],
    internal_bonds=[(0, 1, SINGLE)],
    connect_point=0,
    coord_offsets=[(0, 0), (0.35, 0)]
)

# SU-29: O 醚氧
SU_EXPANSIONS[29] = AtomExpansion(
    atoms=[("O", "O")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-30: S 硫醚
SU_EXPANSIONS[30] = AtomExpansion(
    atoms=[("S", "S")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-31: S 硫醚
SU_EXPANSIONS[31] = AtomExpansion(
    atoms=[("S", "S")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)

# SU-32: X 卤素 (默认Cl)
SU_EXPANSIONS[32] = AtomExpansion(
    atoms=[("Cl", "X")],
    internal_bonds=[],
    connect_point=0,
    coord_offsets=[(0, 0)]
)


# ============================================================
# 芳香SU和双键/三键SU集合
# ============================================================
AROMATIC_SU = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}
DOUBLE_BOND_SU = {14, 15, 16}
TRIPLE_BOND_SU = {17, 18}


def hex_to_cartesian(q: int, r: int, scale: float = 1.0) -> Tuple[float, float]:
    """六角坐标转笛卡尔坐标"""
    x = scale * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = scale * (3 / 2 * r)
    return (x, y)


def get_connect_atom_for_neighbor(su_type: int, neighbor_su_type: int) -> int:
    """
    获取SU中用于连接特定邻居的原子索引
    
    对于有方向性的SU（如酰胺、酯基），根据邻居类型返回正确的连接原子
    """
    if su_type not in SU_EXPANSIONS:
        return 0
    
    expansion = SU_EXPANSIONS[su_type]
    
    # 检查是否有针对该邻居类型的特殊连接规则
    if neighbor_su_type in expansion.directional_connects:
        return expansion.directional_connects[neighbor_su_type]
    
    return expansion.connect_point


def convert_cg_to_allatom(simplified_cg: Dict) -> Dict:
    """
    将简化的粗粒度分子转化为全原子分子
    
    Args:
        simplified_cg: 简化后的粗粒度JSON (nodes + bonds)
    
    Returns:
        全原子JSON (atoms + bonds + metadata)
    """
    cg_nodes = simplified_cg.get('nodes', [])
    cg_bonds = simplified_cg.get('bonds', [])
    
    # ========================================
    # 1. 构建邻居映射（用于确定方向性连接）
    # ========================================
    # cg_seq -> [邻居cg_seq列表]
    neighbors: Dict[int, List[int]] = {node['seq']: [] for node in cg_nodes}
    for bond in cg_bonds:
        a, b = bond[0], bond[1]
        neighbors[a].append(b)
        neighbors[b].append(a)
    
    # cg_seq -> su_type
    seq_to_su: Dict[int, int] = {node['seq']: node['su_type'] for node in cg_nodes}
    
    # ========================================
    # 2. 展开所有CG节点为全原子
    # ========================================
    all_atoms: List[Dict] = []
    all_bonds: List[List] = []
    
    # CG节点seq -> 展开后的原子索引映射
    # {cg_seq: {local_atom_idx: global_atom_idx}}
    cg_to_atom_map: Dict[int, Dict[int, int]] = {}
    
    # CG节点seq -> 该节点的"主连接原子"（继承CG连接的原子）
    # 注意：对于方向性SU，不同邻居可能连接到不同原子
    # {cg_seq: {neighbor_cg_seq: global_atom_idx}}
    cg_connect_atoms: Dict[int, Dict[int, int]] = {}
    
    atom_idx = 0
    
    for node in cg_nodes:
        cg_seq = node['seq']
        cg_id = node['id']
        su_type = node['su_type']
        axial_coord = node['axial_coord']
        xy_coord = node.get('xy_coord')
        
        # 获取SU展开定义
        if su_type not in SU_EXPANSIONS:
            print(f"[Warning] Unknown SU type {su_type}, using default C")
            expansion = AtomExpansion(atoms=[("C", "C")], internal_bonds=[], coord_offsets=[(0, 0)])
        else:
            expansion = SU_EXPANSIONS[su_type]
        
        # 基础笛卡尔坐标：优先使用简化JSON中的xy_coord（与原始pos2d一致），
        # 若不存在则回退到六角坐标投影
        if isinstance(xy_coord, (list, tuple)) and len(xy_coord) >= 2:
            try:
                base_x, base_y = float(xy_coord[0]), float(xy_coord[1])
            except (TypeError, ValueError):
                base_x, base_y = hex_to_cartesian(axial_coord[0], axial_coord[1])
        else:
            base_x, base_y = hex_to_cartesian(axial_coord[0], axial_coord[1])
        
        # 记录该CG节点展开的原子映射
        local_to_global: Dict[int, int] = {}
        cg_to_atom_map[cg_seq] = local_to_global
        cg_connect_atoms[cg_seq] = {}
        
        # 添加原子
        for local_idx, (element, atom_type) in enumerate(expansion.atoms):
            # 计算坐标偏移
            if local_idx < len(expansion.coord_offsets):
                dx, dy = expansion.coord_offsets[local_idx]
            else:
                dx, dy = 0, 0
            
            atom = {
                "idx": atom_idx,
                "element": element,
                "atom_type": atom_type,
                "cg_seq": cg_seq,
                "cg_id": cg_id,
                "su_type": su_type,
                "local_idx": local_idx,
                "coord": [base_x + dx, base_y + dy, 0.0],
                "axial_coord": axial_coord
            }
            all_atoms.append(atom)
            local_to_global[local_idx] = atom_idx
            atom_idx += 1
        
        # 添加内部键
        for (i, j, bond_type) in expansion.internal_bonds:
            global_i = local_to_global[i]
            global_j = local_to_global[j]
            all_bonds.append([global_i, global_j, bond_type, "internal"])
        
        # 预计算该CG节点对各邻居的连接原子
        for neighbor_seq in neighbors[cg_seq]:
            neighbor_su = seq_to_su[neighbor_seq]
            local_connect = get_connect_atom_for_neighbor(su_type, neighbor_su)
            global_connect = local_to_global[local_connect]
            cg_connect_atoms[cg_seq][neighbor_seq] = global_connect
    
    # ========================================
    # 3. 处理CG节点之间的连接
    # ========================================
    for bond in cg_bonds:
        cg_a, cg_b = bond[0], bond[1]
        
        # 获取各自用于连接对方的原子
        atom_a = cg_connect_atoms[cg_a].get(cg_b)
        atom_b = cg_connect_atoms[cg_b].get(cg_a)
        
        if atom_a is None or atom_b is None:
            print(f"[Warning] Cannot find connect atoms for bond {cg_a}-{cg_b}")
            continue
        
        # 确定键类型
        su_a = seq_to_su[cg_a]
        su_b = seq_to_su[cg_b]
        bond_type = determine_bond_type(su_a, su_b)
        
        all_bonds.append([atom_a, atom_b, bond_type, "cg_bond"])
    
    # ========================================
    # 4. 构建输出
    # ========================================
    result = {
        "atoms": all_atoms,
        "bonds": [[b[0], b[1], b[2]] for b in all_bonds],  # 简化输出
        "bond_details": all_bonds,  # 完整信息
        "metadata": {
            "cg_node_count": len(cg_nodes),
            "cg_bond_count": len(cg_bonds),
            "atom_count": len(all_atoms),
            "bond_count": len(all_bonds),
            "created_at": datetime.now().isoformat()
        },
        "cg_to_atom_map": {str(k): v for k, v in cg_to_atom_map.items()}
    }
    
    return result


def determine_bond_type(su_a: int, su_b: int) -> str:
    """
    根据两个SU类型确定键类型
    """
    # 芳香-芳香：芳香键
    if su_a in AROMATIC_SU and su_b in AROMATIC_SU:
        return AROMATIC
    
    # 双键碳之间：双键
    if su_a in DOUBLE_BOND_SU and su_b in DOUBLE_BOND_SU:
        return DOUBLE
    
    # 三键碳之间：三键
    if su_a in TRIPLE_BOND_SU and su_b in TRIPLE_BOND_SU:
        return TRIPLE
    
    # 氰基内部（SU-4的C和N之间已在内部处理，这里是SU-4与其他的连接）
    # 默认单键
    return SINGLE


def convert_simplified_to_allatom(input_path: str, output_path: Optional[str] = None) -> Dict:
    """
    读取简化的CG JSON并转化为全原子
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        simplified_cg = json.load(f)
    
    result = convert_cg_to_allatom(simplified_cg)
    result['metadata']['source_file'] = input_path
    
    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # 去掉_simplified后缀
        if base.endswith('_simplified'):
            base = base[:-11]
        output_path = f"{base}_allatom{ext}"
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[CG->AllAtom] Input: {input_path}")
    print(f"[CG->AllAtom] Output: {output_path}")
    print(f"[CG->AllAtom] CG Nodes: {result['metadata']['cg_node_count']}")
    print(f"[CG->AllAtom] CG Bonds: {result['metadata']['cg_bond_count']}")
    print(f"[CG->AllAtom] Atoms: {result['metadata']['atom_count']}")
    print(f"[CG->AllAtom] Bonds: {result['metadata']['bond_count']}")
    
    return result


def print_allatom_summary(allatom: Dict):
    """打印全原子结果的摘要"""
    atoms = allatom.get('atoms', [])
    bonds = allatom.get('bonds', [])
    
    print("\n" + "=" * 60)
    print("全原子分子转化结果")
    print("=" * 60)
    
    # 统计元素分布
    elem_counts: Dict[str, int] = {}
    for atom in atoms:
        elem = atom.get('element', '?')
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
    
    print(f"\n原子总数: {len(atoms)}")
    print(f"键总数: {len(bonds)}")
    
    print("\n元素分布:")
    for elem, count in sorted(elem_counts.items()):
        print(f"  {elem:2s}: {count:3d} 个")
    
    # 统计键类型
    bond_type_counts: Dict[str, int] = {}
    for bond in bonds:
        bt = bond[2] if len(bond) > 2 else "SINGLE"
        bond_type_counts[bt] = bond_type_counts.get(bt, 0) + 1
    
    print("\n键类型分布:")
    for bt, count in sorted(bond_type_counts.items()):
        print(f"  {bt:10s}: {count:3d} 个")
    
    # 分子式
    formula_parts = []
    for elem in ['C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'I']:
        if elem in elem_counts:
            if elem_counts[elem] == 1:
                formula_parts.append(elem)
            else:
                formula_parts.append(f"{elem}{elem_counts[elem]}")
    print(f"\n分子式: {''.join(formula_parts)}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将简化的粗粒度分子转化为全原子分子')
    parser.add_argument('input', help='输入JSON文件路径（简化的CG格式）')
    parser.add_argument('-o', '--output', help='输出JSON文件路径（可选）')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    
    result = convert_simplified_to_allatom(args.input, args.output)
    
    if args.verbose:
        print_allatom_summary(result)
