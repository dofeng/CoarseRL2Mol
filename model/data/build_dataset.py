import os
import argparse
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Optional, List
import numpy as np
import sys
from pathlib import Path

try:
    from ..shared.coarse_graph import (
        smiles_to_coarse_grained_graph,
        TooFewNodesError,
        NoStructureUnitsError,
        UnassignedAtomsError,
        IncompleteShiftMappingError,
    )
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from model.shared.coarse_graph import (
        smiles_to_coarse_grained_graph,
        TooFewNodesError,
        NoStructureUnitsError,
        UnassignedAtomsError,
        IncompleteShiftMappingError,
    )


# Dataset construction is intentionally restricted to neutral CHONS molecules.
# Halogen-containing or charged structures are skipped before coarse graining.
ALLOWED_ELEMENTS = {'C', 'H', 'O', 'N', 'S'}

# Thread-safe statistics counter
class ThreadSafeStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.stats = {
            "processed": 0,
            "skipped_invalid_smiles": 0,
            "skipped_no_csv": 0,
            "skipped_bad_spectrum": 0,
            "skipped_cg_fail": 0,
            "skipped_zero_carbon": 0,
            "skipped_too_few_nodes": 0,
            "skipped_incomplete_shifts": 0,
        }

    def increment(self, key: str, value: int = 1):
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + value
    
    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return self.stats.copy()

def calculate_carbon_count_from_smiles(smiles: str) -> int:
    """
     新增：从SMILES字符串计算总碳原子数
    
    Args:
        smiles: SMILES字符串
    
    Returns:
        总碳原子数，如果解析失败返回0
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        
        # 计算所有碳原子数量
        carbon_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                carbon_count += 1
        
        return carbon_count
    except Exception:
        return 0

def get_carbon_atom_indices_from_smiles(smiles: str) -> Optional[set]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'}
    except Exception:
        return None

def is_valid_molecule(smiles: str) -> Tuple[bool, str]:
    """Checks if a SMILES string represents a valid molecule for our processing."""
    if not smiles:
        return False, "Empty SMILES string"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string, cannot be parsed by RDKit"
        
    # Check for multiple disconnected components
    if len(Chem.GetMolFrags(mol)) > 1:
        return False, "Molecule has disconnected components (contains '.')"
        
    for atom in mol.GetAtoms():
        # Check for unsupported elements
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False, f"Unsupported element: {atom.GetSymbol()}"
        # Check for charges
        if atom.GetFormalCharge() != 0:
            return False, f"Atom {atom.GetIdx()} has a formal charge of {atom.GetFormalCharge()}"
            
    return True, "Valid"

# ------------------------------------------------------------
# 新增: shift_mapping 提供真实核磁位移 (input_smiles ➜ {atom_idx: (ppm,intensity)})
# ------------------------------------------------------------
def process_single_molecule(basename: str, data_dir: str, output_dir: str, stats_counter: ThreadSafeStats,
                            shift_mapping: Dict[str, Dict[int, Tuple[float, float]]],
                            shift_mapping_by_id: Optional[Dict[str, Dict[int, Tuple[float, float]]]] = None,
                            min_su_nodes: int = 6,
                            save_visualization: bool = True,
                            require_shift_by_id: bool = False,
                            require_complete_shifts: bool = False):
    """
    处理单个分子文件，生成并保存其 .pt 图数据文件。
    """
    txt_path = os.path.join(data_dir, f"{basename}.txt")
    
    # 不再需要 .csv 文件，因为谱图是动态生成的
    # if not os.path.exists(csv_path):
    #     stats_counter.increment("skipped_no_csv")
    #     return

    try:
        # 读取SMILES
        with open(txt_path, 'r', encoding='utf-8') as f:
            smiles = f.read().strip()
        
        # 验证分子有效性
        is_valid, reason = is_valid_molecule(smiles)
        if not is_valid:
            stats_counter.increment("skipped_invalid_smiles")
            return
        
        # 🔧 新增：计算总碳数
        total_carbon_count = calculate_carbon_count_from_smiles(smiles)
        if total_carbon_count <= 0:
            stats_counter.increment("skipped_zero_carbon")
            return
            
        # curated 数据必须优先按文件 ID 精确匹配，避免重复 SMILES 互相覆盖。
        # 非严格模式下保留旧数据的 smiles -> atom shift 映射回退。
        real_shift_map = {}
        if shift_mapping_by_id:
            real_shift_map = shift_mapping_by_id.get(basename, {})
        if not real_shift_map and not require_shift_by_id:
            real_shift_map = shift_mapping.get(smiles, {})
        if require_shift_by_id and not real_shift_map:
            stats_counter.increment("skipped_no_csv")
            return

        if require_complete_shifts:
            carbon_atom_indices = get_carbon_atom_indices_from_smiles(smiles)
            if carbon_atom_indices is None:
                stats_counter.increment("skipped_invalid_smiles")
                return
            provided_indices = set(int(idx) for idx in real_shift_map.keys())
            if provided_indices != carbon_atom_indices:
                stats_counter.increment("skipped_incomplete_shifts")
                return

        # 生成粗粒度图，并保存 .pt 与可视化 .png 文件
        # y_spectrum 传入 None, 将在 coarse_graph.py 中动态生成
        output_prefix = os.path.join(output_dir, basename)
        result = smiles_to_coarse_grained_graph(
            smiles_string=smiles,
            output_path_prefix=output_prefix,
            y_spectrum=None, # 谱图将从节点数据动态生成
            real_atom_shifts=real_shift_map,
            min_su_nodes=int(min_su_nodes),
            save_visualization=bool(save_visualization),
            require_complete_real_shifts=bool(require_complete_shifts),
        )
        
        if result:
            stats_counter.increment("processed")
        else:
            # 理论上不应该到这里，因为函数现在会抛出异常而不是返回 None
            stats_counter.increment("skipped_cg_fail")
            
    except TooFewNodesError as e:
        stats_counter.increment("skipped_too_few_nodes")
    except NoStructureUnitsError as e:
        stats_counter.increment("skipped_cg_fail")
    except UnassignedAtomsError as e:
        stats_counter.increment("skipped_cg_fail")
    except IncompleteShiftMappingError as e:
        stats_counter.increment("skipped_incomplete_shifts")
    except Exception as e:
        import traceback
        basename = os.path.basename(txt_path).replace('.txt', '')
        print(f"--- Exception occurred for molecule: {basename} ---")
        try:
            print(f"SMILES: {smiles}")
        except NameError:
            pass # smiles was not defined
        traceback.print_exc()
        print("----------------------------------------------------")
        stats_counter.increment("skipped_cg_fail")


def main(args):
    """Main function to orchestrate the dataset building process with multithreading."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    source_files = os.listdir(args.data_dir)
    basenames = sorted([os.path.splitext(f)[0] for f in source_files if f.endswith('.txt')])
    
    print(f"Found {len(basenames)} unique molecules to process in '{args.data_dir}'")
    print(f"Using {args.num_workers} worker threads for parallel processing.")
    
    stats_counter = ThreadSafeStats()
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_molecule,
                basename,
                args.data_dir,
                args.output_dir,
                stats_counter,
                args.shift_mapping,
                args.shift_mapping_by_id,
                args.min_su_nodes,
                not args.no_png,
                args.require_shift_by_id,
                args.require_complete_shifts,
            ): basename
            for basename in basenames
        }

        pbar = tqdm(total=len(futures), desc="Processing Molecules", unit="mol")

        for future in as_completed(futures):
            try:
                future.result() # Wait for completion and catch potential exceptions
            except Exception as e:
                basename = futures[future]
                print(f"Molecule {basename} failed with an exception: {e}")
            pbar.update(1)
        pbar.close()

    # Final statistics
    final_stats = stats_counter.get_stats()
    
    print("\n--- Processing Complete ---")
    print(f"Successfully processed and generated .pt files for: {final_stats['processed']} molecules")
    print("Skipped molecules breakdown:")
    print(f"  - Invalid SMILES or chemical structure: {final_stats['skipped_invalid_smiles']}")
    print(f"  - Missing corresponding .csv file:      {final_stats.get('skipped_no_csv', 0)}")
    print(f"  - Malformed or inconsistent .csv file: {final_stats.get('skipped_bad_spectrum', 0)}")
    print(f"  - Zero carbon atoms (e.g., pure heteroatom molecules): {final_stats['skipped_zero_carbon']}")
    print(f"  - Too few coarse-grained nodes (< {args.min_su_nodes} nodes): {final_stats.get('skipped_too_few_nodes', 0)}")
    print(f"  - Incomplete atom-level shift mapping: {final_stats.get('skipped_incomplete_shifts', 0)}")
    print(f"  - Coarse-graining failed (e.g., undefined structures): {final_stats['skipped_cg_fail']}")
    print(f"\nIndividual .pt and .png files are saved in: '{args.output_dir}'")

def load_shift_mapping(
    shift_path: Optional[str],
    unit_intensity: bool = False,
) -> Tuple[Dict[str, Dict[int, Tuple[float, float]]], Dict[str, Dict[int, Tuple[float, float]]]]:
    shift_mapping: Dict[str, Dict[int, Tuple[float, float]]] = {}
    shift_mapping_by_id: Dict[str, Dict[int, Tuple[float, float]]] = {}

    if not shift_path:
        return shift_mapping, shift_mapping_by_id

    if not os.path.exists(shift_path):
        print(f"警告: shift_csv 路径 '{shift_path}' 不存在，跳过真实位移映射。")
        return shift_mapping, shift_mapping_by_id

    def parse_shift_file(path_csv: str, basename: Optional[str] = None) -> int:
        df_tmp = pd.read_csv(path_csv)
        if df_tmp.empty:
            return 0

        required_columns = {"input_smiles", "atom_index", "real_chemical_shift"}
        missing_columns = sorted(required_columns - set(df_tmp.columns))
        if missing_columns:
            raise ValueError(f"missing required columns: {missing_columns}")

        local_by_smiles: Dict[str, Dict[int, Tuple[float, float]]] = {}
        local_by_id: Dict[int, Tuple[float, float]] = {}
        seen_indices = set()
        smiles_values = set()

        for _, row in df_tmp.iterrows():
            smi = str(row['input_smiles']).strip()
            if not smi or smi.lower() == "nan":
                raise ValueError("empty input_smiles")
            smiles_values.add(smi)
            if len(smiles_values) > 1 and basename is not None:
                raise ValueError("multiple input_smiles values in one curated shift file")

            idx = int(row['atom_index'])
            if idx in seen_indices:
                raise ValueError(f"duplicate atom_index: {idx}")
            seen_indices.add(idx)

            ppm = float(row['real_chemical_shift'])
            if not np.isfinite(ppm):
                raise ValueError(f"non-finite real_chemical_shift for atom_index {idx}: {ppm}")

            inten = 1.0 if unit_intensity else float(row.get('real_intensity', 1.0))
            if not np.isfinite(inten) or inten < 0:
                raise ValueError(f"invalid real_intensity for atom_index {idx}: {inten}")

            local_by_smiles.setdefault(smi, {})[idx] = (ppm, inten)
            local_by_id[idx] = (ppm, inten)

        for smi, atom_map in local_by_smiles.items():
            shift_mapping.setdefault(smi, {}).update(atom_map)
        if basename is not None:
            shift_mapping_by_id[str(basename)] = local_by_id

        return len(local_by_id)

    if os.path.isdir(shift_path):
        csv_files = sorted(f for f in os.listdir(shift_path) if f.lower().endswith('.csv'))
        parsed_files = 0
        parsed_rows = 0
        for f in csv_files:
            path_csv = os.path.join(shift_path, f)
            basename = os.path.splitext(f)[0]
            try:
                count = parse_shift_file(path_csv, basename=basename)
                if count:
                    parsed_files += 1
                    parsed_rows += count
            except Exception as e:
                print(f"  -> 警告: 解析真实位移文件 '{path_csv}' 失败: {e}")
        print(
            f"已从目录 '{shift_path}' 载入位移数据: {parsed_rows} 条记录, "
            f"{parsed_files} 个文件, {len(shift_mapping)} 个唯一 SMILES"
        )
    else:
        df_shift = pd.read_csv(shift_path)
        for _, row in df_shift.iterrows():
            smi = str(row['input_smiles']).strip()
            idx = int(row['atom_index'])
            ppm = float(row['real_chemical_shift'])
            inten = 1.0 if unit_intensity else float(row.get('real_intensity', 1.0))
            shift_mapping.setdefault(smi, {})[idx] = (ppm, inten)
        print(f"已加载真实核磁位移数据: {len(df_shift)} 条记录，对应 {len(shift_mapping)} 个唯一 SMILES")

    if unit_intensity:
        print("已启用 --unit_intensity: 所有原子级峰强度按 1.0 处理")

    return shift_mapping, shift_mapping_by_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build a dataset for NMR prediction from raw data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--curated_dir',
        type=str,
        default=None,
        help="Curated dataset root containing structure_data/ and shift_data/.\n"
             "Example: --curated_dir /home/dmin/CoarseRL2Mol/data/curated_shift_dataset"
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=None,
        help="Directory containing the source .txt and .csv files.\nExample: --data_dir /path/to/organized_data"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='processed_dataset', 
        help="Directory to save the output `my_raw_data.pt` file.\nDefaults to './processed_dataset'"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=min(8, os.cpu_count() + 4 if os.cpu_count() else 8),
        help="Number of worker threads for parallel processing.\nDefaults to a sensible number based on CPU cores."
    )

    parser.add_argument(
        '--shift_csv',
        type=str,
        default=None,
        help="CSV 文件，列包含: input_smiles,atom_index,real_chemical_shift,real_intensity"
    )
    parser.add_argument(
        '--unit_intensity',
        action='store_true',
        help="Ignore real_intensity in shift CSV and use 1.0 for every atom-level carbon peak."
    )
    parser.add_argument(
        '--min_su_nodes',
        type=int,
        default=6,
        help="Minimum number of coarse-grained SU nodes required. Historical default is 6."
    )
    parser.add_argument(
        '--no_png',
        action='store_true',
        help="Skip coarse graph PNG visualization and only save .pt files."
    )
    parser.add_argument(
        '--require_shift_by_id',
        action='store_true',
        help="Require shift_data/<id>.csv to exist for each structure_data/<id>.txt; do not fall back to SMILES matching."
    )
    parser.add_argument(
        '--require_complete_shifts',
        action='store_true',
        help="Require atom_index values in the shift data to exactly match all RDKit carbon atom indices."
    )
    
    args = parser.parse_args()

    if args.curated_dir:
        args.data_dir = args.data_dir or os.path.join(args.curated_dir, "structure_data")
        args.shift_csv = args.shift_csv or os.path.join(args.curated_dir, "shift_data")
        args.require_shift_by_id = True
        args.require_complete_shifts = True

    if not args.data_dir:
        parser.error("必须提供 --data_dir，或提供包含 structure_data/ 与 shift_data/ 的 --curated_dir")

    shift_mapping, shift_mapping_by_id = load_shift_mapping(
        args.shift_csv,
        unit_intensity=bool(args.unit_intensity),
    )

    args.shift_mapping = shift_mapping
    args.shift_mapping_by_id = shift_mapping_by_id
    main(args) 
