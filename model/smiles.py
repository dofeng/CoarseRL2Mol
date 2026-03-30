import os
import gzip
import argparse
from typing import List
from rdkit import Chem
try:
    from rdkit.Chem import rdMolStandardize as _rdMolStandardize
except Exception:
    _rdMolStandardize = None
from tqdm import tqdm

# 允许的原子：仅 H, C, N, O, S
ALLOWED_ATOMS = {1, 6, 7, 8, 16}

def is_chons_only(mol):
    """检查分子是否仅含 CHONS 元素"""
    return all(atom.GetAtomicNum() in ALLOWED_ATOMS for atom in mol.GetAtoms())

def is_neutral(mol):
    """检查分子是否为中性：任一原子带电即排除（避免内盐）"""
    return all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())

def get_largest_fragment(mol):
    """获取最大有机片段（去除盐和溶剂）。优先使用 rdMolStandardize，缺失时降级。"""
    if _rdMolStandardize is not None:
        chooser = _rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        return chooser.choose(mol)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    organic_frags = [f for f in frags if is_chons_only(f)]
    candidates = organic_frags if organic_frags else frags
    return max(candidates, key=lambda m: m.GetNumHeavyAtoms())

def to_simple_smiles(mol):
    """转换为简单的 SMILES（无立体化学信息）"""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def filter_molecule(smiles, min_heavy=7, max_heavy=140):
    """
    筛选分子：
    - 仅含 CHONS 元素
    - 中性分子
    - 重原子数在 15-40 之间
    - 返回简单 SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 选择最大片段并标准化
    mol = get_largest_fragment(mol)
    
    # 检查是否仅含 CHONS
    if not is_chons_only(mol):
        return None
    
    # 检查是否中性
    if not is_neutral(mol):
        return None
    
    # 检查重原子数
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms < min_heavy or heavy_atoms > max_heavy:
        return None
    
    # 返回简单 SMILES
    return to_simple_smiles(mol)
def detect_delimiter(header: str) -> str:
    if header.count('\t') >= 1:
        return '\t'
    if header.count(',') >= 1:
        return ','
    return None

def find_smiles_index(header: str, delim: str) -> int:
    cols = [c.strip().lower() for c in header.strip().split(delim)]
    candidates = [
        'canonical_smiles', 'smiles', 'can_smiles', 'canonical smiles'
    ]
    for i, c in enumerate(cols):
        if c in candidates:
            return i
    # ChEMBL chemreps.txt 通常第二列是 canonical_smiles
    return 1 if len(cols) > 1 else -1

def find_id_index(header: str, delim: str) -> int:
    cols = [c.strip().lower() for c in header.strip().split(delim)]
    candidates = ['chembl_id', 'molecule_chembl_id', 'chembl id']
    for i, c in enumerate(cols):
        if c in candidates:
            return i
    # ChEMBL chemreps.txt 第一列通常是 chembl_id
    return 0

def open_text_maybe_gzip(path: str):
    return gzip.open(path, 'rt', encoding='utf-8', errors='ignore') if path.endswith('.gz') else open(path, 'r', encoding='utf-8', errors='ignore')

def expand_inputs(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if os.path.isdir(full):
                    continue
                if name.endswith(('.txt', '.csv', '.gz')):
                    files.append(full)
        else:
            files.append(p)
    return files

def process_input_files(inputs: List[str], outdir: str, min_heavy: int, max_heavy: int, overwrite: bool) -> int:
    os.makedirs(outdir, exist_ok=True)
    saved = 0
    pbar = tqdm(desc='Saved', unit='mol')

    for path in expand_inputs(inputs):
        if not os.path.exists(path):
            continue
        with open_text_maybe_gzip(path) as f:
            header = f.readline()
            if not header:
                continue
            delim = detect_delimiter(header) or '\t'
            smiles_idx = find_smiles_index(header, delim)
            id_idx = find_id_index(header, delim)
            if smiles_idx < 0 or id_idx < 0:
                continue
            for line in f:
                parts = line.rstrip('\n').split(delim)
                if len(parts) <= max(smiles_idx, id_idx):
                    continue
                chembl_id = parts[id_idx].strip()
                smi = parts[smiles_idx].strip()
                if not chembl_id or not smi:
                    continue
                out_path = os.path.join(outdir, f"{chembl_id}.txt")
                if not overwrite and os.path.exists(out_path):
                    continue
                filtered = filter_molecule(smi, min_heavy=min_heavy, max_heavy=max_heavy)
                if not filtered:
                    continue
                with open(out_path, 'w', encoding='utf-8') as out:
                    out.write(filtered + '\n')
                saved += 1
                pbar.update(1)
    pbar.close()
    return saved

def main():
    parser = argparse.ArgumentParser(description='从一个或多个 ChEMBL chemreps/CSV 文件筛选 CHONS-only 中性分子，按 chembl_id.txt 保存。')
    parser.add_argument('--inputs', nargs='+', required=True, help='输入文件列表（支持 .txt 或 .gz，首行需包含 canonical_smiles 列）')
    parser.add_argument('--outdir', default='smiles_out', help='输出目录')
    parser.add_argument('--min_heavy', type=int, default=15, help='最小重原子数（含）')
    parser.add_argument('--max_heavy', type=int, default=40, help='最大重原子数（含）')
    parser.add_argument('--overwrite', action='store_true', help='同名 chembl_id 已存在时是否覆盖写入（默认跳过）')
    args = parser.parse_args()

    saved = process_input_files(args.inputs, args.outdir, args.min_heavy, args.max_heavy, args.overwrite)
    print(f"Done. Saved {saved} molecules into '{args.outdir}'.")

if __name__ == '__main__':
    main()