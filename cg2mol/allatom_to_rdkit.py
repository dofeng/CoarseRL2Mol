#!/usr/bin/env python3
"""
allatom_to_rdkit.py - 将全原子JSON转化为RDKit分子并导出为PDB/SDF等格式

输入：全原子JSON（atoms + bonds）
输出：RDKit Mol对象，以及PDB/SDF/MOL/XYZ文件

处理：
1. 从JSON构建RDKit分子
2. 设置正确的键类型（SINGLE, DOUBLE, TRIPLE, AROMATIC）
3. 设置芳香性标记
4. 处理2D坐标到3D坐标的转换
5. 使用力场优化3D结构
6. 导出为多种格式
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdMolTransforms, rdMolDescriptors
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


class RDKitConversionError(ValueError):
    """Raised when the generated all-atom graph fails strict sanitization."""


class AllAtomToRDKit:
    """将全原子JSON转化为RDKit分子"""
    
    def __init__(self, verbose: bool = True, strict: bool = True):
        self.verbose = verbose
        self.strict = bool(strict)
        
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
            raise RDKitConversionError("all-atom JSON contains no atoms")
        
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
            if json_idx in atom_idx_map:
                raise RDKitConversionError(f"duplicate all-atom index: {json_idx}")
            rd_atom.SetNoImplicit(True)
            for key in ("cg_id", "cg_seq", "su_type", "local_idx"):
                if key in atom_data:
                    rd_atom.SetProp(key, str(atom_data[key]))
            
            # 设置芳香性（只有芳香环中的C/N/S原子，不包括H）
            # 且只有主原子(local_idx=0)才是芳香环的一部分
            if atom_data.get("aromatic", su_type in AROMATIC_SU and element != 'H' and local_idx == 0):
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
                raise RDKitConversionError(f"invalid all-atom bond: {bond_data}")
            
            idx_a, idx_b, bond_type_str = bond_data[0], bond_data[1], bond_data[2]
            
            # 映射到RDKit索引
            if idx_a not in atom_idx_map or idx_b not in atom_idx_map:
                raise RDKitConversionError(f"bond {idx_a}-{idx_b} references an unknown atom")
            
            rd_a = atom_idx_map[idx_a]
            rd_b = atom_idx_map[idx_b]
            
            # 检查是否已存在键
            if mol.GetBondBetweenAtoms(rd_a, rd_b) is not None:
                raise RDKitConversionError(f"duplicate all-atom bond: {idx_a}-{idx_b}")
            
            # 获取键类型
            if bond_type_str not in BOND_TYPE_MAP:
                raise RDKitConversionError(f"unsupported bond type: {bond_type_str}")
            bond_type = BOND_TYPE_MAP[bond_type_str]
            
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
        
        original_molecule = Chem.Mol(mol)
        sanitize_flag = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            problems = []
            try:
                problems = list(Chem.DetectChemistryProblems(original_molecule))
            except Exception:
                pass
            problem_text = "; ".join(
                f"{problem.GetType()}: {problem.Message()}"
                for problem in problems[:5]
            )
            message = (
                "generated all-atom molecule failed RDKit sanitization: "
                f"flag={sanitize_flag}"
                + (f", problems={problem_text}" if problem_text else "")
            )
            if self.strict:
                raise RDKitConversionError(message)
            if self.verbose:
                print(f"[Warning] {message}")
        elif self.verbose:
            print("[RDKit] Sanitization successful")

        if self.strict:
            radicals = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons()]
            if radicals:
                raise RDKitConversionError(f"unexpected radical atoms: {radicals[:10]}")
            if len(Chem.GetMolFrags(mol)) != 1:
                raise RDKitConversionError("all-atom molecule is disconnected")
            counts = dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))
            expected = allatom_json.get("metadata", {}).get("element_counts")
            if expected is not None and counts != expected:
                raise RDKitConversionError(
                    f"all-atom element accounting mismatch: expected={expected}, actual={counts}"
                )
        source_format = str(allatom_json.get("metadata", {}).get("source_format", ""))
        mol.SetProp("cg2mol_source_format", source_format)
        
        return mol.GetMol()
    
    def _set_coordinates(self, mol: Chem.RWMol, atoms: List[Dict], 
                         atom_idx_map: Dict[int, int]):
        """设置3D坐标"""
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(any(
            len(atom.get("coord", [])) > 2 and abs(float(atom["coord"][2])) > 1e-8
            for atom in atoms
        ))
        
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
    
    def optimize_3d(self, mol: Chem.Mol, max_iters: int = 5000) -> Chem.Mol:
        """Minimize a conformer and fail rather than claim non-convergence."""
        molecule = Chem.Mol(mol)
        if int(max_iters) <= 0 or molecule.GetNumConformers() == 0:
            raise RDKitConversionError("optimization requires a conformer and max_iters > 0")
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            force_field = "MMFF94"
            status = AllChem.MMFFOptimizeMolecule(molecule, maxIters=int(max_iters))
        elif AllChem.UFFHasAllMoleculeParams(molecule):
            force_field = "UFF"
            status = AllChem.UFFOptimizeMolecule(molecule, maxIters=int(max_iters))
        else:
            raise RDKitConversionError("neither MMFF nor UFF covers all atoms")
        if status != 0:
            raise RDKitConversionError(
                f"{force_field} optimization did not converge: status={status}, max_iters={max_iters}"
            )
        positions = molecule.GetConformer().GetPositions()
        if not all(math.isfinite(float(value)) for row in positions for value in row):
            raise RDKitConversionError("optimized coordinates contain non-finite values")
        bond_lengths = [
            math.dist(positions[bond.GetBeginAtomIdx()], positions[bond.GetEndAtomIdx()])
            for bond in molecule.GetBonds()
        ]
        if bond_lengths and (min(bond_lengths) < 0.5 or max(bond_lengths) > 3.0):
            raise RDKitConversionError("optimized conformer contains implausible bond lengths")
        molecule.SetProp("cg2mol_force_field", force_field)
        molecule.SetIntProp("cg2mol_optimization_status", int(status))
        if self.verbose:
            print(f"[RDKit] {force_field} optimization converged")
        return molecule

    def generate_3d_coords(
        self, mol: Chem.Mol, *, random_seed: int = 42,
        max_iters: int = 5000, embed_timeout: int = 300,
    ) -> Chem.Mol:
        """Generate one explicit-H conformer; embedding failure is fatal."""
        molecule = Chem.AddHs(Chem.Mol(mol))
        if (
            molecule.GetNumAtoms() >= 300
            and molecule.GetNumConformers()
            and molecule.HasProp("cg2mol_source_format")
            and molecule.GetProp("cg2mol_source_format") == "rl_mtcs"
        ):
            conformer = molecule.GetConformer()
            positions = conformer.GetPositions()
            primary_positions = [
                tuple(positions[atom.GetIdx()])
                for atom in molecule.GetAtoms()
                if atom.HasProp("local_idx") and atom.GetProp("local_idx") == "0"
            ]
            if len(primary_positions) != len(set(primary_positions)):
                raise RDKitConversionError("CG layout contains overlapping node positions")
            center = positions.mean(axis=0)
            randomizer = random.Random(int(random_seed))
            for atom_index, position in enumerate(positions):
                conformer.SetAtomPosition(atom_index, (
                    float((position[0] - center[0]) * 1.45),
                    float((position[1] - center[1]) * 1.45),
                    float(position[2] - center[2]) + randomizer.uniform(-0.25, 0.25),
                ))
            conformer.Set3D(True)
            if self.verbose:
                print(f"[RDKit] Refining CG-seeded 3D coordinates for {molecule.GetNumAtoms()} atoms", flush=True)
            molecule = self.optimize_3d(molecule, max_iters=max_iters)
            molecule.SetProp("cg2mol_coord_mode", "3d")
            molecule.SetProp("cg2mol_coordinate_method", "cg_layout_seeded")
            return molecule
        molecule.RemoveAllConformers()
        parameters = AllChem.ETKDGv3()
        parameters.useRandomCoords = True
        parameters.randomSeed = int(random_seed)
        parameters.maxIterations = 1000
        if hasattr(parameters, "timeout"):
            parameters.timeout = max(1, int(embed_timeout))
        if self.verbose:
            print(f"[RDKit] Embedding {molecule.GetNumAtoms()} atoms (timeout={embed_timeout}s)", flush=True)
        conformer_id = AllChem.EmbedMolecule(molecule, parameters)
        if conformer_id < 0:
            raise RDKitConversionError("3D embedding failed; no molecule was exported")
        if self.verbose:
            print(f"[RDKit] Optimizing conformer (max_iters={max_iters})", flush=True)
        molecule = self.optimize_3d(molecule, max_iters=max_iters)
        molecule.SetProp("cg2mol_coord_mode", "3d")
        molecule.SetProp("cg2mol_coordinate_method", "ETKDGv3")
        return molecule


def convert_allatom_to_rdkit(
    allatom_json: Dict,
    verbose: bool = True,
    *,
    strict: bool = True,
) -> Optional[Chem.Mol]:
    """便捷函数：将全原子JSON转化为RDKit分子"""
    converter = AllAtomToRDKit(verbose=verbose, strict=strict)
    return converter.convert(allatom_json)


def _prepare_coordinates(
    molecule: Chem.Mol, coord_mode: str, *, verbose: bool,
    random_seed: int = 42, max_iters: int = 5000, embed_timeout: int = 300,
) -> Chem.Mol:
    prepared = Chem.AddHs(Chem.Mol(molecule))
    if coord_mode == "3d":
        return AllAtomToRDKit(verbose=verbose).generate_3d_coords(
            prepared, random_seed=random_seed, max_iters=max_iters,
            embed_timeout=embed_timeout,
        )
    if coord_mode == "2d":
        prepared.RemoveAllConformers()
        AllChem.Compute2DCoords(prepared)
    elif coord_mode not in {"none", "prepared"}:
        raise ValueError(f"unsupported coordinate mode: {coord_mode}")
    if prepared.GetNumConformers() == 0:
        raise RDKitConversionError("molecule has no coordinates")
    prepared.SetProp("cg2mol_coord_mode", coord_mode)
    return prepared


def export_molecule(mol: Chem.Mol, output_path: str, 
                    format: str = "auto",
                    coord_mode: str = "3d",
                    verbose: bool = True,
                    strict: bool = True,
                    random_seed: int = 42,
                    max_iters: int = 5000,
                    embed_timeout: int = 300) -> bool:
    """
    导出RDKit分子为文件
    
    Args:
        mol: RDKit分子
        output_path: 输出文件路径
        format: 输出格式 (auto, pdb, sdf, mol, xyz)
        coord_mode: 坐标模式
            - "3d": 生成3D坐标并优化（默认）
            - "2d": 生成2D平面坐标（初始坐标，便于观察）
            - "none": 不生成坐标，使用原始坐标
        verbose: 是否打印信息
    
    Returns:
        是否成功
    """
    if mol is None:
        raise RDKitConversionError("no molecule to export")
    if format == "auto":
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "sdf"
    format = format.lower()
    if format not in {"sdf", "mol", "pdb", "xyz"}:
        raise ValueError(f"unsupported export format: {format}")
    if coord_mode not in {"3d", "2d", "none", "prepared"}:
        raise ValueError(f"unsupported coordinate mode: {coord_mode}")
    if strict:
        sanitize_flag = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            raise RDKitConversionError(
                f"cannot export unsanitized molecule: flag={sanitize_flag}"
            )

    mol_with_h = _prepare_coordinates(
        mol, coord_mode, verbose=verbose, random_seed=random_seed,
        max_iters=max_iters, embed_timeout=embed_timeout,
    )
    if format == "sdf":
        writer = Chem.SDWriter(output_path)
        try:
            writer.SetForceV3000(mol_with_h.GetNumAtoms() > 999)
            writer.write(mol_with_h)
        finally:
            writer.close()
    elif format == "mol":
        Chem.MolToMolFile(mol_with_h, output_path, forceV3000=mol_with_h.GetNumAtoms() > 999)
    elif format == "pdb":
        Chem.MolToPDBFile(mol_with_h, output_path)
    else:
        _write_xyz(mol_with_h, output_path)
    if verbose:
        print(f"[Export] Saved to {output_path}")
    return True


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
                         formats: Sequence[str] = ("sdf", "pdb"),
                         coord_mode: str = "3d",
                         verbose: bool = True,
                         strict: bool = True,
                         random_seed: int = 42,
                         max_iters: int = 5000,
                         embed_timeout: int = 300) -> Dict[str, str]:
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
    converter = AllAtomToRDKit(verbose=verbose, strict=strict)
    mol = converter.convert(allatom_json)
    
    if mol is None:
        raise RuntimeError("Failed to convert to RDKit molecule")
    prepared = _prepare_coordinates(
        mol, coord_mode, verbose=verbose, random_seed=random_seed,
        max_iters=max_iters, embed_timeout=embed_timeout,
    )
    
    # 确定输出目录和基础名称
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if base_name.endswith('_allatom'):
        base_name = base_name[:-8]
    
    # 根据coord_mode添加后缀
    if coord_mode == "2d":
        base_name += "_2d"
    
    output_dir = output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出各格式
    output_files = {}
    
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        if export_molecule(
            prepared,
            output_path,
            format=fmt,
            coord_mode="prepared",
            verbose=verbose,
            strict=strict,
        ):
            output_files[fmt] = output_path
    
    return output_files


def convert_rl_mtcs_file(
    input_path: str,
    output_dir: Optional[str] = None,
    *,
    formats: Sequence[str] = ("sdf", "pdb"),
    coord_mode: str = "3d",
    random_seed: int = 42,
    max_iters: int = 5000,
    embed_timeout: int = 300,
    allow_partial: bool = False,
    verbose: bool = True,
) -> Dict[str, str]:
    """Reconstruct one RL_MTCS graph and export a shared, audited conformer.

    Returns paths keyed by ``allatom_json``, ``report``, and requested formats.
    SU labels and real node IDs come only from the final assembled graph.
    Coordinates are generated once, then reused by JSON/SDF/PDB exports.
    """
    from .cg_simplify import simplify_rl_mtcs_molecule
    from .cg_to_allatom import CG2MolConversionError, convert_cg_to_allatom

    source_path = Path(input_path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    simplified = simplify_rl_mtcs_molecule(payload)
    metadata = simplified["metadata"]
    if not allow_partial and metadata.get("assembly_complete") is not True:
        raise CG2MolConversionError(
            "RL_MTCS completion is not confirmed; use final.json or final_molecule.json "
            "with a completed assembly audit"
        )
    requested_formats = tuple(str(value).lower() for value in formats)
    if not requested_formats or set(requested_formats) - {"sdf", "mol", "pdb", "xyz"}:
        raise ValueError("formats must contain only sdf, mol, pdb, or xyz")
    allatom = convert_cg_to_allatom(simplified, allow_partial=allow_partial)
    molecule = convert_allatom_to_rdkit(allatom, verbose=verbose)
    prepared = _prepare_coordinates(
        molecule, coord_mode, verbose=verbose, random_seed=random_seed,
        max_iters=max_iters, embed_timeout=embed_timeout,
    )
    if prepared.GetNumAtoms() != len(allatom["atoms"]):
        raise RDKitConversionError("coordinate preparation changed the audited atom inventory")
    conformer = prepared.GetConformer()
    for atom_data in allatom["atoms"]:
        position = conformer.GetAtomPosition(int(atom_data["idx"]))
        atom_data["coord"] = [float(position.x), float(position.y), float(position.z)]
    allatom["metadata"].update({
        "source_file": str(source_path.resolve()),
        "coord_mode": coord_mode,
        "sanitized": True,
        "formula": rdMolDescriptors.CalcMolFormula(prepared),
    })
    report = {
        **allatom["metadata"],
        "connected_component_count": len(Chem.GetMolFrags(prepared)),
        "radical_atom_count": sum(bool(atom.GetNumRadicalElectrons()) for atom in prepared.GetAtoms()),
        "optimization_converged": coord_mode == "3d",
        "optimization_max_iters": int(max_iters) if coord_mode == "3d" else None,
        "random_seed": int(random_seed) if coord_mode == "3d" else None,
        "force_field": prepared.GetProp("cg2mol_force_field") if prepared.HasProp("cg2mol_force_field") else None,
        "coordinate_method": prepared.GetProp("cg2mol_coordinate_method") if prepared.HasProp("cg2mol_coordinate_method") else coord_mode,
        "stereochemistry": "unspecified_centers_are_not_inferred_from_coarse_grained_layout",
    }
    destination = Path(output_dir) if output_dir is not None else source_path.parent / "allatom"
    destination.mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, str] = {}
    for file_format in requested_formats:
        output_path = destination / f"final_allatom.{file_format}"
        export_molecule(prepared, str(output_path), format=file_format, coord_mode="prepared", verbose=verbose)
        output_files[file_format] = str(output_path)
    json_path = destination / "final_allatom.json"
    report_path = destination / "conversion_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(allatom, handle, indent=2, ensure_ascii=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    output_files.update(allatom_json=str(json_path), report=str(report_path))
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
    if not __package__:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        __package__ = "cg2mol"
    
    parser = argparse.ArgumentParser(description='将RL_MTCS或全原子JSON转化并导出')
    parser.add_argument('input', help='输入JSON文件路径')
    parser.add_argument('-o', '--output-dir', help='输出目录（可选）')
    parser.add_argument('-f', '--formats', nargs='+', default=['sdf', 'pdb'],
                        help='输出格式列表（默认: sdf pdb）')
    parser.add_argument('-c', '--coord-mode', choices=['3d', '2d', 'none'], default='3d',
                        help='坐标模式: 3d=优化3D坐标, 2d=平面2D坐标, none=原始坐标（默认: 3d）')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印详细信息')
    parser.add_argument('--rl-mtcs', action='store_true', help='输入是RL_MTCS final.json或final_molecule.json')
    parser.add_argument('--max-iters', type=int, default=5000, help='力场优化迭代上限')
    parser.add_argument('--embed-timeout', type=int, default=300, help='三维嵌入超时秒数')
    
    args = parser.parse_args()
    
    conversion = convert_rl_mtcs_file if args.rl_mtcs else convert_allatom_file
    output_files = conversion(
        args.input,
        output_dir=args.output_dir,
        formats=args.formats,
        coord_mode=args.coord_mode,
        verbose=args.verbose,
        max_iters=args.max_iters,
        embed_timeout=args.embed_timeout,
    )
    
    print(f"\n导出完成:")
    for fmt, path in output_files.items():
        print(f"  {fmt}: {path}")
    
    # 打印分子信息
    if args.verbose and not args.rl_mtcs:
        with open(args.input, 'r') as f:
            data = json.load(f)
        mol = convert_allatom_to_rdkit(data, verbose=False)
        print_molecule_info(mol)
