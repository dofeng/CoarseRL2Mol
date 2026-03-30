import os
import argparse
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Optional, List
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from coarse_graph import smiles_to_coarse_grained_graph, TooFewNodesError, NoStructureUnitsError, UnassignedAtomsError


# Define allowed elements
ALLOWED_ELEMENTS = {'C', 'H', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I'}

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
            "skipped_too_few_nodes": 0
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
                            shift_mapping: Dict[str, Dict[int, Tuple[float, float]]]):
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
            
        # 若不存在该分子的节点级真实位移数据：允许继续生成 .pt（无需真实谱，供后续 z_library 使用）
        real_shift_map = shift_mapping.get(smiles, {})

        # 生成粗粒度图，并保存 .pt 与可视化 .png 文件
        # y_spectrum 传入 None, 将在 coarse_graph.py 中动态生成
        output_prefix = os.path.join(output_dir, basename)
        result = smiles_to_coarse_grained_graph(
            smiles_string=smiles,
            output_path_prefix=output_prefix,
            y_spectrum=None, # 谱图将从节点数据动态生成
            real_atom_shifts=real_shift_map
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
        futures = {executor.submit(process_single_molecule, basename, args.data_dir, args.output_dir, stats_counter, args.shift_mapping): basename for basename in basenames}

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
    print(f"  - Too few coarse-grained nodes (< 6 nodes): {final_stats.get('skipped_too_few_nodes', 0)}")
    print(f"  - Coarse-graining failed (e.g., undefined structures): {final_stats['skipped_cg_fail']}")
    print(f"\nIndividual .pt and .png files are saved in: '{args.output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build a dataset for NMR prediction from raw data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
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
    
    args = parser.parse_args()

    shift_mapping: Dict[str, Dict[int, Tuple[float, float]]] = {}
    if args.shift_csv and os.path.exists(args.shift_csv):
        # -------------------------------------------------
        # A) 若提供的是“目录” —— 每个分子一个 csv 文件
        #    文件名须与 SMILES/TXT 基名一致，例如 38.csv
        #    列名: input_smiles,atom_index,real_chemical_shift,real_intensity
        # B) 若提供的是“单个csv文件” —— 包含多分子记录
        # -------------------------------------------------
        if os.path.isdir(args.shift_csv):
            csv_files = [f for f in os.listdir(args.shift_csv) if f.lower().endswith('.csv')]
            for f in csv_files:
                path_csv = os.path.join(args.shift_csv, f)
                try:
                    df_tmp = pd.read_csv(path_csv)
                    if df_tmp.empty: 
                        continue
                    smi = str(df_tmp.iloc[0]['input_smiles']).strip()
                    for _, row in df_tmp.iterrows():
                        idx   = int(row['atom_index'])
                        ppm   = float(row['real_chemical_shift'])
                        inten = float(row['real_intensity'])
                        shift_mapping.setdefault(smi, {})[idx] = (ppm, inten)
                except Exception as e:
                    print(f"  -> 警告: 解析真实位移文件 '{path_csv}' 失败: {e}")
            print(f"已从目录 '{args.shift_csv}' 载入位移数据，对应 {len(shift_mapping)} 个分子")
        else:
            # 单文件模式
            df_shift = pd.read_csv(args.shift_csv)
            for _, row in df_shift.iterrows():
                smi  = str(row['input_smiles']).strip()
                idx  = int(row['atom_index'])
                ppm  = float(row['real_chemical_shift'])
                inten= float(row['real_intensity'])
                shift_mapping.setdefault(smi, {})[idx] = (ppm, inten)
            print(f"已加载真实核磁位移数据: {len(df_shift)} 条记录，对应 {len(shift_mapping)} 个分子")
    else:
        if args.shift_csv:
            print(f"警告: shift_csv 路径 '{args.shift_csv}' 不存在，跳过真实位移映射。")
        shift_mapping = {}

    args.shift_mapping = shift_mapping
    main(args) 