"""
cg2mol - 粗粒度分子到全原子分子转换模块

将MCTS生成的粗粒度分子图转换为全原子分子结构，
支持3D坐标生成和PDB/SDF文件导出。

当前RL_MTCS结果可使用 ``convert_rl_mtcs_file("path/to/final.json")``。
默认严格校验化学结构，再生成一次三维构象并导出JSON、SDF、PDB及审计报告。

    paths = convert_rl_mtcs_file(
        "test_results/YM-1/rl_mtcs_nmr/<run>/final.json",
        output_dir="test_results/YM-1/allatom",
        coord_mode="3d",
        max_iters=5000,
    )

``final_molecule.json`` 同样支持。默认拒绝未完成的组装结果，且不会补回未组装资源、
改变最终SU标签或推断缺失的立体化学信息。SDF为保留键级的主要结构输出。
依赖RDKit与NetworkX。大分子优先以粗粒度布局作为三维初值并进行力场松弛，
小分子使用ETKDGv3；三维生成或优化未成功时不会伪装成已完成的三维导出。

主要模块:
- cg_simplify: MCTS输出JSON → 简化格式（节点+连接）
- cg_to_allatom: 简化格式 → 全原子JSON
- allatom_to_rdkit: 全原子JSON → RDKit分子/PDB/SDF
"""

from .cg_simplify import (
    simplify_cg_molecule,
    simplify_rl_mtcs_molecule,
    simplify_cg_file,
    print_simplified_summary,
)
from .cg_to_allatom import (
    convert_cg_to_allatom, 
    convert_simplified_to_allatom,
    CG2MolConversionError,
    print_allatom_summary,
    AtomExpansion,
    SU_EXPANSIONS,
)
from .allatom_to_rdkit import (
    AllAtomToRDKit,
    RDKitConversionError,
    convert_allatom_to_rdkit,
    convert_allatom_file,
    convert_rl_mtcs_file,
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
    'simplify_rl_mtcs_molecule',
    'simplify_cg_file',
    'print_simplified_summary',
    # 全原子转换
    'convert_cg_to_allatom',
    'convert_simplified_to_allatom',
    'CG2MolConversionError',
    'print_allatom_summary',
    'AtomExpansion',
    'SU_EXPANSIONS',
    # RDKit/导出
    'AllAtomToRDKit',
    'RDKitConversionError',
    'convert_allatom_to_rdkit',
    'convert_allatom_file',
    'convert_rl_mtcs_file',
    'export_molecule',
    'print_molecule_info',
    # 序列化
    'save_coarse_grained_molecule',
    'load_coarse_grained_molecule',
    'serialize_builder',
    'get_molecule_summary',
    'print_molecule_summary',
]
