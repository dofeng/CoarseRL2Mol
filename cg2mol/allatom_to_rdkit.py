#!/usr/bin/env python3
"""
allatom_to_rdkit.py - 将全原子JSON转化为RDKit分子并导出为PDB/SDF等格式

输入：全原子JSON（atoms + bonds）
输出：RDKit Mol对象，以及PDB/SDF/MOL2等文件

处理：
1. 从JSON构建RDKit分子
2. 设置正确的键类型（SINGLE, DOUBLE, TRIPLE, AROMATIC）
3. 设置芳香性标记
4. 处理2D坐标到3D坐标的转换
5. 使用力场优化3D结构
6. 导出为多种格式
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdMolTransforms
    from rdkit.Chem import rdDepictor, rdForceFieldHelpers
    from rdkit.Geometry import Point3D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[Warning] RDKit not available. Install with: pip install rdkit")


# 键类型映射
BOND_TYPE_MAP = {
    "SINGLE": Chem.BondType.SINGLE if RDKIT_AVAILABLE else None,
    "DOUBLE": Chem.BondType.DOUBLE if RDKIT_AVAILABLE else None,
    "TRIPLE": Chem.BondType.TRIPLE if RDKIT_AVAILABLE else None,
    "AROMATIC": Chem.BondType.AROMATIC if RDKIT_AVAILABLE else None,
}

# 芳香SU类型
AROMATIC_SU = {5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 30}


class AllAtomToRDKit:
    """将全原子JSON转化为RDKit分子"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required but not installed")
    
    def convert(self, allatom_json: Dict) -> Optional[Chem.Mol]:
        """
        将全原子JSON转化为RDKit分子
        
        Args:
            allatom_json: 全原子JSON数据
        
        Returns:
            RDKit Mol对象
        """
        atoms = allatom_json.get('atoms', [])
        bonds = allatom_json.get('bonds', [])
        
        if not atoms:
            print("[Error] No atoms in JSON")
            return None
        
        # 创建可编辑分子
        mol = Chem.RWMol()
        
        # ========================================
        # 1. 添加原子
        # ========================================
        atom_idx_map: Dict[int, int] = {}  # JSON idx -> RDKit idx
        
        for atom_data in atoms:
            json_idx = atom_data['idx']
            element = atom_data['element']
            su_type = atom_data.get('su_type', 0)
            local_idx = atom_data.get('local_idx', 0)
            
            # 创建RDKit原子
            rd_atom = Chem.Atom(element)
            
            # 设置芳香性（只有芳香环中的C/N/S原子，不包括H）
            # 且只有主原子(local_idx=0)才是芳香环的一部分
            if su_type in AROMATIC_SU and element not in ['H'] and local_idx == 0:
                rd_atom.SetIsAromatic(True)
            
            # 添加到分子
            rd_idx = mol.AddAtom(rd_atom)
            atom_idx_map[json_idx] = rd_idx
        
        if self.verbose:
            print(f"[RDKit] Added {len(atoms)} atoms")
        
        # ========================================
        # 2. 添加键
        # ========================================
        bond_count = 0
        aromatic_bonds = 0
        
        for bond_data in bonds:
            if len(bond_data) < 3:
                continue
            
            idx_a, idx_b, bond_type_str = bond_data[0], bond_data[1], bond_data[2]
            
            # 映射到RDKit索引
            if idx_a not in atom_idx_map or idx_b not in atom_idx_map:
                if self.verbose:
                    print(f"[Warning] Bond {idx_a}-{idx_b} has invalid atom indices")
                continue
            
            rd_a = atom_idx_map[idx_a]
            rd_b = atom_idx_map[idx_b]
            
            # 检查是否已存在键
            if mol.GetBondBetweenAtoms(rd_a, rd_b) is not None:
                continue
            
            # 获取键类型
            bond_type = BOND_TYPE_MAP.get(bond_type_str, Chem.BondType.SINGLE)
            
            # 添加键
            mol.AddBond(rd_a, rd_b, bond_type)
            bond_count += 1
            
            if bond_type_str == "AROMATIC":
                aromatic_bonds += 1
                # 确保两端原子都标记为芳香
                mol.GetAtomWithIdx(rd_a).SetIsAromatic(True)
                mol.GetAtomWithIdx(rd_b).SetIsAromatic(True)
                mol.GetBondBetweenAtoms(rd_a, rd_b).SetIsAromatic(True)
        
        if self.verbose:
            print(f"[RDKit] Added {bond_count} bonds ({aromatic_bonds} aromatic)")
        
        # ========================================
        # 3. 设置坐标
        # ========================================
        self._set_coordinates(mol, atoms, atom_idx_map)
        
        # ========================================
        # 4. 尝试Sanitize分子
        # ========================================
        try:
            Chem.SanitizeMol(mol)
            if self.verbose:
                print("[RDKit] Sanitization successful")
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Sanitization failed: {e}")
            # 尝试部分sanitize
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                                   Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                                   Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                                   Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION)
            except:
                pass
        
        return mol.GetMol()
    
    def _set_coordinates(self, mol: Chem.RWMol, atoms: List[Dict], 
                         atom_idx_map: Dict[int, int]):
        """设置3D坐标"""
        conf = Chem.Conformer(mol.GetNumAtoms())
        
        for atom_data in atoms:
            json_idx = atom_data['idx']
            coord = atom_data.get('coord', [0, 0, 0])
            
            if json_idx in atom_idx_map:
                rd_idx = atom_idx_map[json_idx]
                # 使用2D坐标作为x,y，z设为0
                x = coord[0] if len(coord) > 0 else 0.0
                y = coord[1] if len(coord) > 1 else 0.0
                z = coord[2] if len(coord) > 2 else 0.0
                conf.SetAtomPosition(rd_idx, Point3D(x, y, z))
        
        mol.AddConformer(conf, assignId=True)
    
    def optimize_3d(self, mol: Chem.Mol, max_iters: int = 500) -> Chem.Mol:
        """
        优化分子3D结构
        
        使用MMFF或UFF力场进行能量最小化
        """
        mol = Chem.RWMol(mol)
        
        # 尝试使用MMFF力场
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
            if self.verbose:
                print("[RDKit] MMFF optimization completed")
            return mol.GetMol()
        except Exception as e:
            if self.verbose:
                print(f"[Warning] MMFF failed: {e}, trying UFF")
        
        # 回退到UFF力场
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            if self.verbose:
                print("[RDKit] UFF optimization completed")
            return mol.GetMol()
        except Exception as e:
            if self.verbose:
                print(f"[Warning] UFF failed: {e}")
        
        return mol.GetMol()
    
    def generate_3d_coords(self, mol: Chem.Mol) -> Chem.Mol:
        """
        使用RDKit生成3D坐标（如果原始坐标不好用）
        """
        mol = Chem.RWMol(mol)
        
        try:
            # 移除现有构象
            mol.RemoveAllConformers()
            
            # 生成3D坐标
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            # 优化
            AllChem.MMFFOptimizeMolecule(mol)
            
            if self.verbose:
                print("[RDKit] Generated new 3D coordinates")
            
            return mol.GetMol()
        except Exception as e:
            if self.verbose:
                print(f"[Warning] 3D generation failed: {e}")
            return mol.GetMol()


def convert_allatom_to_rdkit(allatom_json: Dict, verbose: bool = True) -> Optional[Chem.Mol]:
    """便捷函数：将全原子JSON转化为RDKit分子"""
    converter = AllAtomToRDKit(verbose=verbose)
    return converter.convert(allatom_json)


def export_molecule(mol: Chem.Mol, output_path: str, 
                    format: str = "auto",
                    coord_mode: str = "3d",
                    verbose: bool = True) -> bool:
    """
    导出RDKit分子为文件
    
    Args:
        mol: RDKit分子
        output_path: 输出文件路径
        format: 输出格式 (auto, pdb, sdf, mol, mol2, xyz)
        coord_mode: 坐标模式
            - "3d": 生成3D坐标并优化（默认）
            - "2d": 生成2D平面坐标（初始坐标，便于观察）
            - "none": 不生成坐标，使用原始坐标
        verbose: 是否打印信息
    
    Returns:
        是否成功
    """
    if mol is None:
        print("[Error] No molecule to export")
        return False
    
    # 自动检测格式
    if format == "auto":
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "sdf"
    
    # 添加氢原子
    mol_with_h = Chem.AddHs(mol)
    
    # 根据coord_mode生成坐标
    if coord_mode == "3d":
        try:
            mol_with_h.RemoveAllConformers()
            
            # 随机坐标嵌入 + MMFF优化
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.randomSeed = 42
            params.maxIterations = 1000
            embed_result = AllChem.EmbedMolecule(mol_with_h, params)
            
            if embed_result == -1:
                raise RuntimeError("Embedding failed")
            
            AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=500)
            if verbose:
                print("[Export] 3D coordinates generated and optimized")
                        
        except Exception as e:
            if verbose:
                print(f"[Warning] 3D generation failed: {e}")
    
    elif coord_mode == "2d":
        try:
            mol_with_h.RemoveAllConformers()
            # 生成2D平面坐标（z=0）
            AllChem.Compute2DCoords(mol_with_h)
            if verbose:
                print("[Export] 2D coordinates generated (planar)")
        except Exception as e:
            if verbose:
                print(f"[Warning] 2D generation failed: {e}")
    
    else:  # coord_mode == "none"
        if verbose:
            print("[Export] Using original coordinates")
    
    # 导出
    try:
        if format in ["sdf", "mol"]:
            writer = Chem.SDWriter(output_path)
            writer.write(mol_with_h)
            writer.close()
        elif format == "pdb":
            Chem.MolToPDBFile(mol_with_h, output_path)
        elif format == "xyz":
            _write_xyz(mol_with_h, output_path)
        elif format == "mol2":
            # RDKit不直接支持mol2，使用PDB作为替代
            Chem.MolToPDBFile(mol_with_h, output_path)
            if verbose:
                print("[Warning] MOL2 format not directly supported, using PDB")
        else:
            print(f"[Error] Unknown format: {format}")
            return False
        
        if verbose:
            print(f"[Export] Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Export failed: {e}")
        return False


def _write_xyz(mol: Chem.Mol, output_path: str):
    """写入XYZ格式文件"""
    conf = mol.GetConformer()
    
    with open(output_path, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"Generated by cg2mol\n")
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")


def convert_allatom_file(input_path: str, 
                         output_dir: Optional[str] = None,
                         formats: List[str] = ["sdf", "pdb"],
                         coord_mode: str = "3d",
                         verbose: bool = True) -> Dict[str, str]:
    """
    从全原子JSON文件转化并导出多种格式
    
    Args:
        input_path: 输入JSON文件路径
        output_dir: 输出目录（默认同输入文件目录）
        formats: 输出格式列表
        coord_mode: 坐标模式 ("3d", "2d", "none")
        verbose: 是否打印信息
    
    Returns:
        {格式: 输出文件路径} 字典
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 读取JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        allatom_json = json.load(f)
    
    # 转化为RDKit分子
    converter = AllAtomToRDKit(verbose=verbose)
    mol = converter.convert(allatom_json)
    
    if mol is None:
        raise RuntimeError("Failed to convert to RDKit molecule")
    
    # 确定输出目录和基础名称
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if base_name.endswith('_allatom'):
        base_name = base_name[:-8]
    
    # 根据coord_mode添加后缀
    if coord_mode == "2d":
        base_name += "_2d"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出各格式
    output_files = {}
    
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        if export_molecule(mol, output_path, format=fmt, coord_mode=coord_mode, verbose=verbose):
            output_files[fmt] = output_path
    
    return output_files


def print_molecule_info(mol: Chem.Mol):
    """打印分子信息"""
    if mol is None:
        print("No molecule")
        return
    
    print("\n" + "=" * 60)
    print("分子信息")
    print("=" * 60)
    
    print(f"原子数: {mol.GetNumAtoms()}")
    print(f"键数: {mol.GetNumBonds()}")
    print(f"分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    
    try:
        print(f"分子量: {Chem.Descriptors.MolWt(mol):.2f}")
    except:
        pass
    
    # 统计元素
    elem_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
    
    print("\n元素分布:")
    for elem, count in sorted(elem_counts.items()):
        print(f"  {elem:2s}: {count:3d}")
    
    # 统计键类型
    bond_counts = {}
    for bond in mol.GetBonds():
        bt = str(bond.GetBondType())
        bond_counts[bt] = bond_counts.get(bt, 0) + 1
    
    print("\n键类型分布:")
    for bt, count in sorted(bond_counts.items()):
        print(f"  {bt:15s}: {count:3d}")
    
    # 芳香环
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_bonds = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    print(f"\n芳香原子: {aromatic_atoms}")
    print(f"芳香键: {aromatic_bonds}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将全原子JSON转化为RDKit分子并导出')
    parser.add_argument('input', help='输入JSON文件路径（全原子格式）')
    parser.add_argument('-o', '--output-dir', help='输出目录（可选）')
    parser.add_argument('-f', '--formats', nargs='+', default=['sdf', 'pdb'],
                        help='输出格式列表（默认: sdf pdb）')
    parser.add_argument('-c', '--coord-mode', choices=['3d', '2d', 'none'], default='3d',
                        help='坐标模式: 3d=优化3D坐标, 2d=平面2D坐标, none=原始坐标（默认: 3d）')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    
    output_files = convert_allatom_file(
        args.input,
        output_dir=args.output_dir,
        formats=args.formats,
        coord_mode=args.coord_mode,
        verbose=args.verbose
    )
    
    print(f"\n导出完成:")
    for fmt, path in output_files.items():
        print(f"  {fmt}: {path}")
    
    # 打印分子信息
    if args.verbose:
        with open(args.input, 'r') as f:
            data = json.load(f)
        mol = convert_allatom_to_rdkit(data, verbose=False)
        print_molecule_info(mol)
