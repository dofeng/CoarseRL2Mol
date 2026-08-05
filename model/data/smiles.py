import os
import gzip
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

from rdkit import Chem
try:
    from rdkit.Chem import rdMolStandardize as _rdMolStandardize
except Exception:
    _rdMolStandardize = None
from tqdm import tqdm

# 允许的原子：仅 H, C, N, O, S
ALLOWED_ATOMS = {1, 6, 7, 8, 16}
Record = Tuple[str, str]
FilteredRecord = Tuple[str, Optional[str]]

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


def _filter_batch(
    records: Sequence[Record],
    min_heavy: int,
    max_heavy: int,
) -> List[FilteredRecord]:
    """Filter one bounded batch inside a worker process."""
    return [
        (
            chembl_id,
            filter_molecule(smi, min_heavy=min_heavy, max_heavy=max_heavy),
        )
        for chembl_id, smi in records
    ]


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


def _iter_input_records(inputs: List[str]) -> Iterator[Record]:
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
            required_idx = max(smiles_idx, id_idx)
            for line in f:
                parts = line.rstrip('\n').split(delim)
                if len(parts) <= required_idx:
                    continue
                chembl_id = parts[id_idx].strip()
                smi = parts[smiles_idx].strip()
                if chembl_id and smi:
                    yield chembl_id, smi


def _iter_pending_batches(
    inputs: List[str],
    outdir: str,
    overwrite: bool,
    batch_size: int,
    stats: Dict[str, int],
    pbar,
) -> Iterator[List[Record]]:
    batch: List[Record] = []
    for chembl_id, smi in _iter_input_records(inputs):
        stats['scanned'] += 1
        out_path = os.path.join(outdir, f"{chembl_id}.txt")
        if not overwrite and os.path.exists(out_path):
            stats['skipped_existing'] += 1
            pbar.update(1)
            continue
        batch.append((chembl_id, smi))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_filtered_batch(
    records: Sequence[FilteredRecord],
    outdir: str,
    overwrite: bool,
    stats: Dict[str, int],
) -> None:
    for chembl_id, filtered in records:
        if not filtered:
            stats['rejected'] += 1
            continue
        out_path = os.path.join(outdir, f"{chembl_id}.txt")
        # Recheck here because duplicate IDs can occur in separate pending batches.
        if not overwrite and os.path.exists(out_path):
            stats['skipped_existing'] += 1
            continue
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(filtered + '\n')
        stats['saved'] += 1


def process_input_files(
    inputs: List[str],
    outdir: str,
    min_heavy: int,
    max_heavy: int,
    overwrite: bool,
    num_workers: int = 1,
    batch_size: int = 256,
) -> int:
    if min_heavy < 0 or max_heavy < min_heavy:
        raise ValueError('require 0 <= min_heavy <= max_heavy')
    if num_workers < 1:
        raise ValueError('num_workers must be at least 1')
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1')

    os.makedirs(outdir, exist_ok=True)
    stats = {
        'scanned': 0,
        'saved': 0,
        'rejected': 0,
        'skipped_existing': 0,
    }
    pbar = tqdm(desc='Processed', unit='mol')
    batches = _iter_pending_batches(
        inputs,
        outdir,
        overwrite,
        batch_size,
        stats,
        pbar,
    )

    def consume(filtered_records: Sequence[FilteredRecord]) -> None:
        _write_filtered_batch(filtered_records, outdir, overwrite, stats)
        pbar.update(len(filtered_records))
        pbar.set_postfix(
            saved=stats['saved'],
            rejected=stats['rejected'],
            skipped=stats['skipped_existing'],
            refresh=False,
        )

    try:
        if num_workers == 1:
            for batch in batches:
                consume(_filter_batch(batch, min_heavy, max_heavy))
        else:
            # Bound pending work so millions of input rows are never retained in RAM.
            max_pending = num_workers * 2
            pending: Deque = deque()
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for batch in batches:
                    pending.append(
                        executor.submit(_filter_batch, batch, min_heavy, max_heavy)
                    )
                    if len(pending) >= max_pending:
                        consume(pending.popleft().result())
                while pending:
                    consume(pending.popleft().result())
    finally:
        pbar.close()

    print(
        'Summary: '
        f"scanned={stats['scanned']}, saved={stats['saved']}, "
        f"rejected={stats['rejected']}, "
        f"skipped_existing={stats['skipped_existing']}"
    )
    return stats['saved']

def main():
    parser = argparse.ArgumentParser(description='从一个或多个 ChEMBL chemreps/CSV 文件筛选 CHONS-only 中性分子，按 chembl_id.txt 保存。')
    parser.add_argument('--inputs', nargs='+', required=True, help='输入文件列表（支持 .txt 或 .gz，首行需包含 canonical_smiles 列）')
    parser.add_argument('--outdir', default='smiles_out', help='输出目录')
    parser.add_argument('--min_heavy', type=int, default=15, help='最小重原子数（含）')
    parser.add_argument('--max_heavy', type=int, default=40, help='最大重原子数（含）')
    parser.add_argument('--overwrite', action='store_true', help='同名 chembl_id 已存在时是否覆盖写入（默认跳过）')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='RDKit 并行工作进程数（默认 1）',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='每个并行任务处理的分子数（默认 256）',
    )
    args = parser.parse_args()

    saved = process_input_files(
        args.inputs,
        args.outdir,
        args.min_heavy,
        args.max_heavy,
        args.overwrite,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    print(f"Done. Saved {saved} molecules into '{args.outdir}'.")

if __name__ == '__main__':
    main()
