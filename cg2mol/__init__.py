"""
cg2mol - 粗粒度分子到全原子分子转换模块

将MCTS生成的粗粒度分子图转换为全原子分子结构，
支持3D坐标生成和PDB/SDF文件导出。

主要模块:
- cg_simplify: MCTS输出JSON → 简化格式（节点+连接）
- cg_to_allatom: 简化格式 → 全原子JSON
- allatom_to_rdkit: 全原子JSON → RDKit分子/PDB/SDF
"""

from .cg_simplify import simplify_cg_molecule, simplify_cg_file, print_simplified_summary
from .cg_to_allatom import (
    convert_cg_to_allatom, 
    convert_simplified_to_allatom,
    print_allatom_summary,
    AtomExpansion,
    SU_EXPANSIONS,
)
from .allatom_to_rdkit import (
    AllAtomToRDKit,
    convert_allatom_to_rdkit,
    convert_allatom_file,
    export_molecule,
    print_molecule_info,
)
from .serialization import (
    save_coarse_grained_molecule,
    load_coarse_grained_molecule,
    serialize_builder,
    get_molecule_summary,
    print_molecule_summary,
)

__all__ = [
    # 简化
    'simplify_cg_molecule',
    'simplify_cg_file',
    'print_simplified_summary',
    # 全原子转换
    'convert_cg_to_allatom',
    'convert_simplified_to_allatom',
    'print_allatom_summary',
    'AtomExpansion',
    'SU_EXPANSIONS',
    # RDKit/导出
    'AllAtomToRDKit',
    'convert_allatom_to_rdkit',
    'convert_allatom_file',
    'export_molecule',
    'print_molecule_info',
    # 序列化
    'save_coarse_grained_molecule',
    'load_coarse_grained_molecule',
    'serialize_builder',
    'get_molecule_summary',
    'print_molecule_summary',
]
