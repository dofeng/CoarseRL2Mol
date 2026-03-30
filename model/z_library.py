import torch
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import pickle
from typing import Tuple, List, Set, Dict, Optional

# 从现有模块导入所需组件
from g2s_model import NMR_VAE, LocalNMRDataset, load_raw_data
from coarse_graph import NUM_SU_TYPES, SU_DEFS, SU_MAX_DEGREE, SU_PPM_RANGES
from torch_geometric.utils import k_hop_subgraph, to_undirected

# 从inverse_common导入化学规则（统一定义，避免重复）
from inverse_common import (
    SU_CONNECTION_DEGREE, TERMINAL_SU, HOP1_PORT_COMBINATIONS,
    SU_EXTERNAL_CONNECTIONS, UNSATURATED_PAIRS, FORBIDDEN_CONNECTIONS,
    validate_connection, check_external_connection_requirement
)

# ============================================================================
# 端口匹配辅助函数
# ============================================================================

def _can_match_ports(neighbors: List[int], port_sets: List[Set[int]]) -> bool:
    """检查邻居列表是否能匹配端口集合"""
    if len(neighbors) != len(port_sets):
        return False
    if not neighbors:
        return True

    used = [False] * len(neighbors)
    order = sorted(range(len(port_sets)), key=lambda i: len(port_sets[i]))

    def dfs(port_pos: int) -> bool:
        if port_pos >= len(order):
            return True
        p = order[port_pos]
        allowed = port_sets[p]
        for i, n in enumerate(neighbors):
            if used[i]:
                continue
            if n not in allowed:
                continue
            used[i] = True
            if dfs(port_pos + 1):
                return True
            used[i] = False
        return False

    return dfs(0)


# ============================================================================
# 向后兼容的别名函数
# ============================================================================

def validate_single_connection(center_su: int, neighbor_su: int) -> bool:
    """向后兼容：调用inverse_common.validate_connection"""
    return validate_connection(center_su, neighbor_su)


def check_external_requirement(center_su: int, hop1_counter: Counter) -> Tuple[bool, str]:
    """向后兼容：调用inverse_common.check_external_connection_requirement"""
    # 检查是否至少有一个外接结构
    if center_su not in SU_EXTERNAL_CONNECTIONS:
        return True, ""
    
    required_external = SU_EXTERNAL_CONNECTIONS[center_su]
    hop1_types = set(hop1_counter.keys())
    
    if not any(ext_su in hop1_types for ext_su in required_external):
        return False, f"SU {center_su} requires external connection to {required_external}"
    
    return True, ""


def validate_hop1_multiset(center_su: int, hop1_multiset: tuple) -> Tuple[bool, List[str]]:
    """
    验证1-hop多重集的完整语义合法性
    
    Args:
        center_su: 中心结构单元索引
        hop1_multiset: 1-hop邻居的排序多重集元组，如 (9, 13, 13)
    
    Returns:
        Tuple[bool, List[str]]: (是否合法, 违规信息列表)
    """
    violations = []
    
    # 将多重集转为Counter
    hop1_counter = Counter(hop1_multiset)
    
    # 1. 检查度数约束
    total_degree = len(hop1_multiset)
    
    if center_su in HOP1_PORT_COMBINATIONS:
        port_sets = HOP1_PORT_COMBINATIONS[center_su]
        required_degree = len(port_sets)
        if total_degree != required_degree:
            violations.append(f"Degree {total_degree} != required {required_degree}")
        else:
            if not _can_match_ports(list(hop1_multiset), port_sets):
                violations.append(f"Port combination mismatch for SU {center_su}")
    
        if center_su in FORBIDDEN_CONNECTIONS['double_terminal_bridge']:
            hop1_types = list(hop1_counter.keys())
            if len(hop1_types) > 0 and all(su in TERMINAL_SU for su in hop1_types):
                violations.append(f"Bridge unit {center_su} cannot connect only terminals")
    else:
        max_degree = SU_CONNECTION_DEGREE.get(center_su, 4)
        if isinstance(max_degree, tuple):
            max_degree = max_degree[1]
        if total_degree > max_degree:
            violations.append(f"Degree {total_degree} > max {max_degree}")
    
        for neighbor_su in hop1_counter.keys():
            if not validate_single_connection(center_su, neighbor_su):
                violations.append(f"Invalid connection: {center_su} -> {neighbor_su}")
    
        valid_ext, ext_msg = check_external_requirement(center_su, hop1_counter)
        if not valid_ext:
            violations.append(ext_msg)
    
        if center_su in FORBIDDEN_CONNECTIONS['double_terminal_bridge']:
            hop1_types = list(hop1_counter.keys())
            if len(hop1_types) > 0 and all(su in TERMINAL_SU for su in hop1_types):
                violations.append(f"Bridge unit {center_su} cannot connect only terminals")
    
    return len(violations) == 0, violations


def filter_subgraph_library(
    input_lib_path: str,
    output_lib_path: str,
    strict_mode: bool = True,
    verbose: bool = True
) -> dict:
    """
    根据连接规则筛选子图模板库
    
    Args:
        input_lib_path: 输入库文件路径
        output_lib_path: 输出库文件路径
        strict_mode: 严格模式（True=只保留完全合法的模板）
        verbose: 是否输出详细信息
    
    Returns:
        dict: 筛选后的库
    """
    print(f"\n{'='*60}")
    print("开始筛选子图模板库（1-hop连接规则验证）")
    print(f"{'='*60}")
    print(f"输入库: {input_lib_path}")
    print(f"输出库: {output_lib_path}")
    print(f"严格模式: {strict_mode}")
    
    # 1. 加载原始库
    lib = torch.load(input_lib_path, map_location='cpu')
    templates = lib.get('templates', {})
    su_names = [name for name, _ in SU_DEFS]
    
    print(f"\n原始模板数: {len(templates)}")
    
    # 2. 统计信息
    stats = {
        'total': len(templates),
        'valid': 0,
        'invalid': 0,
        'by_su': defaultdict(lambda: {'total': 0, 'valid': 0, 'invalid': 0}),
        'violation_types': Counter(),
        'invalid_examples': []
    }
    
    # 3. 筛选模板
    filtered_templates = {}
    
    for tpl_key, tpl_info in tqdm(templates.items(), desc="验证1-hop连接规则"):
        center_su, hop1_multiset, hop2_multiset = tpl_key
        
        stats['by_su'][center_su]['total'] += 1
        
        # 验证1-hop多重集
        is_valid, violations = validate_hop1_multiset(center_su, hop1_multiset)
        
        if is_valid:
            stats['valid'] += 1
            stats['by_su'][center_su]['valid'] += 1
            filtered_templates[tpl_key] = tpl_info
        else:
            stats['invalid'] += 1
            stats['by_su'][center_su]['invalid'] += 1
            
            for v in violations:
                stats['violation_types'][v] += 1
            
            # 记录无效示例（最多100个）
            if len(stats['invalid_examples']) < 100:
                stats['invalid_examples'].append({
                    'center_su': center_su,
                    'center_su_name': su_names[center_su],
                    'hop1_multiset': hop1_multiset,
                    'violations': violations,
                    'sample_count': tpl_info.get('sample_count', 0)
                })
    
    # 4. 输出统计信息
    print(f"\n{'='*60}")
    print("筛选结果统计")
    print(f"{'='*60}")
    print(f"有效模板: {stats['valid']} / {stats['total']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"无效模板: {stats['invalid']} / {stats['total']} ({100*stats['invalid']/stats['total']:.1f}%)")
    
    print(f"\n按SU类型统计:")
    for su_idx in sorted(stats['by_su'].keys()):
        su_stats = stats['by_su'][su_idx]
        su_name = su_names[su_idx] if su_idx < len(su_names) else f"SU{su_idx}"
        if su_stats['total'] > 0:
            valid_pct = 100 * su_stats['valid'] / su_stats['total']
            print(f"  SU{su_idx:02d} ({su_name:25s}): "
                  f"有效 {su_stats['valid']:4d} / {su_stats['total']:4d} ({valid_pct:5.1f}%)")
    
    if verbose and stats['violation_types']:
        print(f"\n违规类型统计 (Top 20):")
        for violation, count in stats['violation_types'].most_common(20):
            print(f"  {count:5d}x: {violation}")
    
    if verbose and stats['invalid_examples']:
        print(f"\n无效模板示例 (前10个):")
        for i, ex in enumerate(stats['invalid_examples'][:10]):
            print(f"  [{i+1}] SU{ex['center_su']} ({ex['center_su_name']}): "
                  f"hop1={ex['hop1_multiset']}, samples={ex['sample_count']}")
            for v in ex['violations']:
                print(f"       -> {v}")
    
    # 5. 重建库结构
    # 重建center_index
    new_center_index = defaultdict(list)
    for tpl_key in filtered_templates.keys():
        center_su = tpl_key[0]
        new_center_index[center_su].append(tpl_key)
    
    # 排序
    for center_su in new_center_index:
        new_center_index[center_su] = sorted(new_center_index[center_su])
    
    # 重建role_index（只保留有效模板相关的）
    old_role_index = lib.get('role_index', {})
    new_role_index = {}
    valid_tpl_keys = set(filtered_templates.keys())
    
    for role_key, role_samples in old_role_index.items():
        # 只保留与有效模板相关的角色索引
        filtered_role_samples = [
            s for s in role_samples 
            if not s or s.get('template_key') in valid_tpl_keys
        ]
        if filtered_role_samples:
            new_role_index[role_key] = filtered_role_samples
    
    # 构建新库
    filtered_lib = {
        'templates': filtered_templates,
        'center_index': dict(new_center_index),
        'role_index': new_role_index,
        'su_max_degrees': lib.get('su_max_degrees'),
        'library_type': lib.get('library_type', '3hop_enhanced_v2') + '_filtered',
        'total_templates': len(filtered_templates),
        'filter_stats': {
            'original_count': stats['total'],
            'filtered_count': stats['valid'],
            'removed_count': stats['invalid'],
            'strict_mode': strict_mode
        }
    }
    
    # 6. 保存筛选后的库
    torch.save(filtered_lib, output_lib_path)
    print(f"\n筛选后的库已保存: {output_lib_path}")
    print(f"  -> 有效模板数: {len(filtered_templates)}")
    
    # 7. 生成筛选后的CSV统计
    out_dir = os.path.dirname(output_lib_path) or '.'
    rows = []
    for tpl_key, tpl_info in filtered_templates.items():
        center_su_idx, hop1_multiset, hop2_multiset = tpl_key
        hop1_str = f"[{','.join(str(i) for i in hop1_multiset)}]"
        hop2_str = f"[{','.join(str(i) for i in hop2_multiset)}]"
        rows.append({
            'center_su_idx': int(center_su_idx),
            'center_su': su_names[center_su_idx],
            'hop1_multiset': hop1_str,
            'hop2_multiset': hop2_str,
            'sample_count': int(tpl_info.get('sample_count', 0)),
            'mu_min': float(tpl_info.get('mu_min', 0)),
            'mu_max': float(tpl_info.get('mu_max', 0)),
            'center_mu_median': float(tpl_info.get('center_mu', 0))
        })
    
    if rows:
        rows_sorted = sorted(rows, key=lambda r: (r['center_su_idx'], r['hop1_multiset'], r['hop2_multiset']))
        df = pd.DataFrame(rows_sorted)
        csv_path = os.path.join(out_dir, 'su_multiset_distribution_filtered.csv')
        df.to_csv(csv_path, index=False)
        print(f"  -> 筛选后的多重集统计已保存: {csv_path}")
    
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        idx = np.argsort(values)
        v = values[idx]
        w = weights[idx]
        cw = np.cumsum(w)
        cutoff = float(q) * float(cw[-1])
        pos = int(np.searchsorted(cw, cutoff, side='left'))
        pos = min(max(pos, 0), len(v) - 1)
        return float(v[pos])


    def _compute_common_mu_stats(mu_values: List[float], weights: List[float]) -> Optional[Dict[str, float]]:
        if not mu_values:
            return None
        v = np.asarray(mu_values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w = np.where(w > 0, w, 1.0)

        median = _weighted_quantile(v, w, 0.5)
        abs_dev = np.abs(v - median)
        mad = _weighted_quantile(abs_dev, w, 0.5)
        thr = 3.0 * 1.4826 * mad

        if thr > 0:
            keep = abs_dev <= thr
        else:
            keep = np.ones_like(v, dtype=bool)

        v_keep = v[keep]
        w_keep = w[keep]
        if v_keep.size == 0:
            v_keep = v
            w_keep = w

        return {
            'mu_median': float(median),
            'mu_common_min': float(np.min(v_keep)),
            'mu_common_max': float(np.max(v_keep)),
            'mu_q05': _weighted_quantile(v_keep, w_keep, 0.05),
            'mu_q95': _weighted_quantile(v_keep, w_keep, 0.95),
            'n_total': int(v.size),
            'n_kept': int(v_keep.size),
        }


    su_mu_rows = []
    for su_idx in sorted(new_center_index.keys()):
        mu_values = []
        weights = []
        for tpl_key, tpl_info in filtered_templates.items():
            if tpl_key[0] != su_idx:
                continue
            mu = tpl_info.get('center_mu', 0.0)
            if mu is None:
                continue
            mu = float(mu)
            if mu == 0.0:
                continue
            mu_values.append(mu)
            weights.append(float(tpl_info.get('sample_count', 1)))

        stats_mu = _compute_common_mu_stats(mu_values, weights)
        if stats_mu is None:
            continue
        su_mu_rows.append({
            'center_su_idx': int(su_idx),
            'center_su': su_names[su_idx],
            'mu_median': float(stats_mu['mu_median']),
            'mu_common_min': float(stats_mu['mu_common_min']),
            'mu_common_max': float(stats_mu['mu_common_max']),
            'mu_q05': float(stats_mu['mu_q05']),
            'mu_q95': float(stats_mu['mu_q95']),
            'n_templates_total': int(stats_mu['n_total']),
            'n_templates_kept': int(stats_mu['n_kept']),
            'n_templates_outliers': int(stats_mu['n_total'] - stats_mu['n_kept']),
        })

    if su_mu_rows:
        su_mu_df = pd.DataFrame(su_mu_rows)
        su_mu_csv_path = os.path.join(out_dir, 'su_nmr_common_range_filtered.csv')
        su_mu_df.to_csv(su_mu_csv_path, index=False)
        print(f"  -> 筛选后各SU常见NMR位移范围已保存: {su_mu_csv_path}")
    
    # 8. 按 (center_su, hop1_multiset) 统计NMR范围
    hop1_mu_agg: Dict[Tuple[int, Tuple], List[Dict]] = defaultdict(list)
    for tpl_key, tpl_info in filtered_templates.items():
        center_su_idx, hop1_multiset, _ = tpl_key
        mu = tpl_info.get('center_mu', 0.0)
        if mu is None or float(mu) == 0.0:
            continue
        hop1_mu_agg[(center_su_idx, hop1_multiset)].append({
            'mu': float(mu),
            'mu_min': float(tpl_info.get('mu_min', mu)),
            'mu_max': float(tpl_info.get('mu_max', mu)),
            'sample_count': int(tpl_info.get('sample_count', 1))
        })
    
    hop1_nmr_rows = []
    for (su_idx, hop1_ms), items in hop1_mu_agg.items():
        mu_values = [it['mu'] for it in items]
        weights = [it['sample_count'] for it in items]
        all_mu_min = [it['mu_min'] for it in items]
        all_mu_max = [it['mu_max'] for it in items]
        
        stats_hop1 = _compute_common_mu_stats(mu_values, weights)
        if stats_hop1 is None:
            continue
        
        hop1_str = f"[{','.join(str(i) for i in hop1_ms)}]"
        hop1_nmr_rows.append({
            'center_su_idx': int(su_idx),
            'center_su': su_names[su_idx],
            'hop1_multiset': hop1_str,
            'mu_median': float(stats_hop1['mu_median']),
            'mu_common_min': float(stats_hop1['mu_common_min']),
            'mu_common_max': float(stats_hop1['mu_common_max']),
            'mu_q05': float(stats_hop1['mu_q05']),
            'mu_q95': float(stats_hop1['mu_q95']),
            'mu_global_min': float(min(all_mu_min)),
            'mu_global_max': float(max(all_mu_max)),
            'n_templates': len(items),
            'sample_count_total': sum(weights),
        })
    
    if hop1_nmr_rows:
        hop1_nmr_rows_sorted = sorted(hop1_nmr_rows, key=lambda r: (r['center_su_idx'], r['hop1_multiset']))
        hop1_nmr_df = pd.DataFrame(hop1_nmr_rows_sorted)
        hop1_nmr_csv_path = os.path.join(out_dir, 'su_hop1_nmr_range_filtered.csv')
        hop1_nmr_df.to_csv(hop1_nmr_csv_path, index=False)
        print(f"  -> 筛选后各SU+1-hop组合NMR范围已保存: {hop1_nmr_csv_path}")
    
    # 9. 保存无效模板列表（用于分析）
    if stats['invalid_examples']:
        invalid_rows = []
        for ex in stats['invalid_examples']:
            invalid_rows.append({
                'center_su_idx': ex['center_su'],
                'center_su': ex['center_su_name'],
                'hop1_multiset': str(ex['hop1_multiset']),
                'violations': '; '.join(ex['violations']),
                'sample_count': ex['sample_count']
            })
        invalid_df = pd.DataFrame(invalid_rows)
        invalid_csv_path = os.path.join(out_dir, 'invalid_templates_hop1.csv')
        invalid_df.to_csv(invalid_csv_path, index=False)
        print(f"  -> 无效模板列表已保存: {invalid_csv_path}")
    
    return filtered_lib


def get_allowed_hop1_neighbors(center_su: int) -> Set[int]:
    """
    获取某个中心SU允许的所有1-hop邻居类型
    
    Args:
        center_su: 中心结构单元索引
    
    Returns:
        Set[int]: 允许的邻居SU类型集合
    """
    if center_su not in SU_FIXED_CONNECTIONS:
        # 默认允许所有非末端互连
        allowed = set(range(NUM_SU_TYPES))
        if center_su in TERMINAL_SU:
            allowed -= TERMINAL_SU
        return allowed
    
    allowed = SU_FIXED_CONNECTIONS[center_su]
    if isinstance(allowed, dict):
        all_allowed = set()
        for side_list in allowed.values():
            all_allowed.update(side_list)
        return all_allowed
    else:
        return set(allowed)


def print_connection_rules_summary():
    """打印连接规则摘要"""
    su_names = [name for name, _ in SU_DEFS]
    
    print("\n" + "="*80)
    print("SU连接规则摘要")
    print("="*80)
    
    for su_idx in range(NUM_SU_TYPES):
        su_name = su_names[su_idx] if su_idx < len(su_names) else f"SU{su_idx}"
        max_deg = SU_CONNECTION_DEGREE.get(su_idx, 4)
        is_terminal = su_idx in TERMINAL_SU
        
        allowed = get_allowed_hop1_neighbors(su_idx)
        allowed_str = ', '.join(str(x) for x in sorted(allowed)[:10])
        if len(allowed) > 10:
            allowed_str += f"... ({len(allowed)} total)"
        
        external_req = SU_EXTERNAL_CONNECTIONS.get(su_idx, None)
        external_str = str(external_req) if external_req else "无"
        
        print(f"\nSU{su_idx:02d} ({su_name}):")
        print(f"  最大度数: {max_deg}, 末端: {is_terminal}")
        print(f"  允许连接: [{allowed_str}]")
        print(f"  外接要求: {external_str}")

def get_3hop_template_key_v2(subgraph, su_types_in_subgraph):
    """
    V2: 按多重集精确分桶的3-hop模板键生成
    
    返回:
        template_key: (center_su, hop1_multiset, hop2_multiset) - 多重集桶键
        node_layers: 节点层次信息
    """
    center_id = subgraph.center_id.item()
    center_su_type = su_types_in_subgraph[center_id].item()
    
    # 使用BFS找到不同hop层的节点
    edge_index = subgraph.edge_index
    row, col = edge_index
    
    # 构建邻接表
    adj_list = defaultdict(list)
    for i in range(len(row)):
        adj_list[row[i].item()].append(col[i].item())
        adj_list[col[i].item()].append(row[i].item())
    
    # BFS分层
    visited = set()
    hop_1_nodes = set()
    hop_2_nodes = set()
    hop_3_nodes = set()
    
    # 1-hop邻居
    for neighbor in adj_list[center_id]:
        if neighbor not in visited:
            hop_1_nodes.add(neighbor)
            visited.add(neighbor)
    
    # 2-hop邻居
    for node_1hop in hop_1_nodes:
        for neighbor in adj_list[node_1hop]:
            if neighbor not in visited and neighbor != center_id:
                hop_2_nodes.add(neighbor)
                visited.add(neighbor)
    
    # 3-hop邻居
    for node_2hop in hop_2_nodes:
        for neighbor in adj_list[node_2hop]:
            if neighbor not in visited and neighbor != center_id:
                hop_3_nodes.add(neighbor)
                visited.add(neighbor)
    
    # V2关键改进：使用多重集计数，保留SU类型的重复度信息
    hop_1_su_types = [su_types_in_subgraph[node].item() for node in hop_1_nodes]
    hop_2_su_types = [su_types_in_subgraph[node].item() for node in hop_2_nodes]
    
    # 转换为Counter并转为有序元组(保持多重集的重复信息)
    hop1_counter = Counter(hop_1_su_types)
    hop2_counter = Counter(hop_2_su_types)
    
    # 将Counter转为排序后的多重集元组
    hop1_multiset = tuple(sorted(hop1_counter.elements()))
    hop2_multiset = tuple(sorted(hop2_counter.elements()))
    
    template_key = (center_su_type, hop1_multiset, hop2_multiset)
    
    node_layers = {
        'center': center_id,
        'hop_1': list(hop_1_nodes),
        'hop_2': list(hop_2_nodes), 
        'hop_3': list(hop_3_nodes),
        'hop1_counter': hop1_counter,
        'hop2_counter': hop2_counter
    }
    
    return template_key, node_layers


def build_3hop_library_v2(pt_dir: str, g2s_ckpt: str, k_hop: int = 3,
                         latent_dim: int = 16, hid_dim: int = 384, batch_size: int = 32, device: str = 'cuda',
                         num_workers: int = 4, mols_per_chunk: int = 2000, skip_role_index: bool = False,
                         amp: bool = False, prefetch_factor: int = 2):
    """
    V2: 重构3-hop子图模板库，按多重集精确分桶
    
    核心改进：
    1. 桶键使用多重集：(center_su, hop1_multiset, hop2_multiset)
    2. 桶内存储所有z/μ/π样本，便于KD-Tree检索
    3. 构建center_index和role_index索引
    """
    print("开始构建V2版本3-hop子图模板库...")
    
    # 1. 加载数据和预训练模型
    raw_all = load_raw_data(pt_dir)

    vae = NMR_VAE(hid=hid_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(g2s_ckpt, map_location=device, weights_only=False))
    vae.eval()

    # 2. 收集桶级数据
    template_buckets = defaultdict(list)  # 桶级收集器
    center_index = defaultdict(set)  # center_su -> set(template_keys)
    role_index = defaultdict(list)   # (center_su, hop_role, su_type) -> list(sample_info)

    # cudnn/amp优化
    if device.startswith('cuda'):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    total_mols = len(raw_all)
    processed_mols = 0
    # 分块处理，避免一次性构建超大子图集
    while processed_mols < total_mols:
        end = min(total_mols, processed_mols + int(max(1, mols_per_chunk)))
        raw = raw_all[processed_mols:end]
        processed_mols = end

        dset = LocalNMRDataset(raw, k=k_hop)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None
        )

        desc_txt = f"阶段1/2: 收集桶级样本 [{processed_mols}/{total_mols} mols]"
        if device.startswith('cuda') and amp:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with torch.inference_mode():
            for batch in tqdm(loader, desc=desc_txt):
                batch = batch.to(device)
                with autocast_ctx:
                    pred_params, latent_params = vae(batch)
                mu_pred_batch, pi_pred_unnorm_batch = pred_params
                mu_latent_batch, _ = latent_params

                subgraphs = batch.to_data_list()
                su_names = [name for name, _ in SU_DEFS]

                for i, subgraph in enumerate(subgraphs):
                    subgraph = subgraph.to(device)
                    su_types_in_subgraph = subgraph.x[:, :NUM_SU_TYPES].argmax(-1)

                    template_key, node_layers = get_3hop_template_key_v2(subgraph, su_types_in_subgraph)

                    center_id = node_layers['center']
                    center_su_type = su_types_in_subgraph[center_id].item()
                    center_su_name = su_names[center_su_type]
                    mu_ppm = float(mu_pred_batch[i].item())
                    pi_center = float(pi_pred_unnorm_batch[i].item())
                    z_sample = mu_latent_batch[i].detach().cpu()

                    sample_info = {
                        'z': z_sample,
                        'mu': mu_ppm,
                        'pi': pi_center,
                        'template_key': template_key,
                        'hop1_counter': node_layers['hop1_counter'],
                        'hop2_counter': node_layers['hop2_counter']
                    }
                    template_buckets[template_key].append(sample_info)
                    center_index[center_su_type].add(template_key)

                    if not skip_role_index:
                        def encode_role_node_ppm(node_idx: int, role: str):
                            d = subgraph.clone()
                            d.center_id = torch.tensor([node_idx], device=device)
                            if hasattr(d, 'global_feat'):
                                d.global_feat = d.global_feat.to(device)
                            d = d.to(device)
                            with autocast_ctx:
                                pred_params_n, latent_params_n = vae(d)
                            mu_pred_n, pi_pred_n = pred_params_n
                            z_n, _ = latent_params_n
                            su_name_n = su_names[su_types_in_subgraph[node_idx].item()]
                            l_n, h_n = SU_PPM_RANGES.get(su_name_n, (0, 240))
                            ppm_n_raw = mu_pred_n[0].item()
                            ppm_n = float(max(l_n, min(h_n, ppm_n_raw)))
                            pi_n = float(pi_pred_n[0].item())
                            su_type = su_types_in_subgraph[node_idx].item()
                            role_key = (center_su_type, role, su_type)
                            role_index[role_key].append({
                                'z': z_n[0].detach().cpu(),
                                'mu': ppm_n,
                                'pi': pi_n,
                                'template_key': template_key
                            })
                            return z_n[0].detach().cpu(), ppm_n, pi_n

                        for nidx in node_layers['hop_1']:
                            encode_role_node_ppm(nidx, 'hop1')
                        for nidx in node_layers['hop_2']:
                            encode_role_node_ppm(nidx, 'hop2')

    print(f"收集到 {len(template_buckets)} 个不同的多重集模板桶")

    # 2.b 额外增加：为非碳中心节点收集“结构-only”模板（不做VAE）
    # 目的：逆向约束 1-hop/2-hop 组合，避免凭空捏造
    raw_list_all = raw_all  # load_raw_data 返回的列表
    for mol in tqdm(raw_list_all, desc='阶段1+/2: 收集非碳中心的结构模板'):
        try:
            x_node = mol["x"]
            edge_index = mol["edge_index"]
            is_carbon_mask = (x_node[:, :26].sum(-1) > 0)
            # 确保无向
            if edge_index.numel() > 0:
                edge_index = to_undirected(edge_index)
            num_nodes = x_node.size(0)
            for center in range(num_nodes):
                if bool(is_carbon_mask[center]):
                    continue
                sub_nodes, sub_edge_index, mapping, mask = k_hop_subgraph(
                    center, k_hop, edge_index, relabel_nodes=True, num_nodes=num_nodes)
                # 构造最小 subgraph 结构供键生成函数使用
                class _Mini:
                    pass
                sub = _Mini()
                sub.edge_index = sub_edge_index
                sub.center_id = torch.tensor([mapping])
                su_types_in_subgraph = x_node[sub_nodes][:, :NUM_SU_TYPES].argmax(-1)
                template_key, node_layers = get_3hop_template_key_v2(sub, su_types_in_subgraph)

                # 仅结构：不存 z/μ/π
                sample_info = {
                    'struct_only': True,
                    'template_key': template_key,
                    'hop1_counter': node_layers['hop1_counter'],
                    'hop2_counter': node_layers['hop2_counter']
                }
                template_buckets[template_key].append(sample_info)
                center_index[template_key[0]].add(template_key)

                # 角色索引仅用于“存在性”，不需要 z/μ/π；填充空字典
                center_su_type = template_key[0]
                for su_t, cnt in node_layers['hop1_counter'].items():
                    role_index[(center_su_type, 'hop1', su_t)].append({})
                for su_t, cnt in node_layers['hop2_counter'].items():
                    role_index[(center_su_type, 'hop2', su_t)].append({})
        except Exception:
            continue

    # 3. 聚合桶级统计并构建最终库
    templates = {}
    
    for template_key, samples in tqdm(template_buckets.items(), desc="阶段2/2: 构建桶级统计"):
        if len(samples) < 3:  # 过滤样本数太少的桶
            continue

        # 提取桶内所有样本
        # 对于结构-only 的条目，不含 z/μ/π，需过滤
        z_list = [s['z'] for s in samples if 'z' in s]
        mu_list = [s['mu'] for s in samples if 'mu' in s]
        pi_list = [s['pi'] for s in samples if 'pi' in s]

        # 堆叠为张量
        has_numeric = len(mu_list) > 0 and len(z_list) > 0 and len(pi_list) > 0
        if has_numeric:
            z_samples = torch.stack(z_list)  # (n, latent_dim)
            mu_samples = torch.tensor(mu_list, dtype=torch.float)  # (n,)
            pi_samples = torch.tensor(pi_list, dtype=torch.float)  # (n,)

        # 基本统计量（用于检索和显示；不计算均值/标准差）
        if has_numeric:
            mu_min = float(mu_samples.min().item())
            mu_max = float(mu_samples.max().item())
        else:
            mu_min, mu_max = 0.0, 0.0

        # 提取hop1/hop2计数信息
        hop1_counter = samples[0]['hop1_counter']
        hop2_counter = samples[0]['hop2_counter']

        center_su_type = template_key[0]
        
        # 为外层粗选找到"居中"的z（最接近μ中位数的样本）
        if has_numeric:
            mu_median = float(mu_samples.median().item())
            center_sample_idx = torch.argmin(torch.abs(mu_samples - mu_median)).item()
            center_z = z_samples[center_sample_idx]
            center_mu = float(mu_samples[center_sample_idx].item())
            center_pi = float(pi_samples[center_sample_idx].item())
        else:
            center_sample_idx, center_z, center_mu, center_pi = -1, None, 0.0, 0.0
        
        # 预存按 μ 升序的样本索引，便于逆向阶段定点选择 z
        sorted_idx_by_mu = torch.argsort(mu_samples) if has_numeric else None

        entry = {
            'samples': {
                # 若结构-only，samples 为空字典
            },
            'sorted_idx_by_mu': sorted_idx_by_mu,
            'mu_min': mu_min,
            'mu_max': mu_max,
            # 新增：外层粗选用的"居中"样本
            'center_sample_idx': center_sample_idx,
            'center_z': center_z,
            'center_mu': center_mu,
            'center_pi': center_pi,
            'nmr_range': (mu_min, mu_max),  # 核磁范围
            
            'center_su': int(center_su_type),
            'hop1_counts': hop1_counter,
            'hop2_counts': hop2_counter,
            'sample_count': len(samples)
        }
        if has_numeric:
            entry['samples'] = {
                'z': z_samples,
                'mu': mu_samples,
                'pi': pi_samples
            }
        templates[template_key] = entry

    # 按 (center_su, hop1_multiset, hop2_multiset) 全顺序排序，重建为有序字典
    sorted_items = sorted(templates.items(), key=lambda kv: kv[0])
    templates = {k: v for k, v in sorted_items}

    print(f"生成了 {len(templates)} 个有效的多重集模板桶（已按 center→hop1→hop2 排序）")

    # 4. 构建最终库和索引
    su_names = [name for name, _ in SU_DEFS]
    su_max_degrees = torch.tensor([SU_MAX_DEGREE.get(name, 4) for name in su_names], dtype=torch.long)
    
    # 转换center_index为普通dict（便于序列化）
    center_index_dict = {k: sorted(list(v), key=lambda key: key) for k, v in center_index.items()}
    role_index_dict = dict(role_index)

    final_library = {
        'templates': templates,
        'center_index': center_index_dict,    # center_su -> list(template_keys)
        'role_index': role_index_dict,        # (center_su, role, su_type) -> list(sample_info)
        'su_max_degrees': su_max_degrees,
        'library_type': '3hop_enhanced_v2',
        'total_templates': len(templates)
    }
    
    print("V2版本3-hop模板库构建完成！")
    return final_library


def _concat_samples(a: dict, b: dict) -> dict:
    """合并两组 samples:{z,mu,pi}，返回新字典。任何一方为空则返回另一方。"""
    if not a:
        return b
    if not b:
        return a
    z = torch.cat([a['z'], b['z']], dim=0)
    mu = torch.cat([a['mu'], b['mu']], dim=0)
    pi = torch.cat([a['pi'], b['pi']], dim=0)
    return {'z': z, 'mu': mu, 'pi': pi}


def merge_3hop_libraries_v2(base_lib: dict, add_lib: dict) -> dict:
    """将 add_lib 的模板与索引增量合并到 base_lib 上，返回合并后的库。"""
    if base_lib is None or not isinstance(base_lib, dict):
        return add_lib

    base_templates = base_lib.get('templates', {})
    add_templates = add_lib.get('templates', {})

    for tpl_key, tpl_info in add_templates.items():
        if tpl_key not in base_templates:
            # 直接新增
            base_templates[tpl_key] = tpl_info
            continue
        # 需要合并 samples 与统计
        a = base_templates[tpl_key]
        b = tpl_info

        # 合并 samples（可能为空字典）
        samples_a = a.get('samples', {}) or {}
        samples_b = b.get('samples', {}) or {}
        samples_new = _concat_samples(samples_a, samples_b)

        # 重新计算统计量
        has_numeric = bool(samples_new) and samples_new['mu'].numel() > 0
        if has_numeric:
            mu_samples = samples_new['mu']
            pi_samples = samples_new['pi']
            z_samples = samples_new['z']
            mu_min = float(mu_samples.min().item())
            mu_max = float(mu_samples.max().item())
            mu_median = float(mu_samples.median().item())
            center_sample_idx = torch.argmin(torch.abs(mu_samples - mu_median)).item()
            center_z = z_samples[center_sample_idx]
            center_mu = float(mu_samples[center_sample_idx].item())
            center_pi = float(pi_samples[center_sample_idx].item())
            sorted_idx_by_mu = torch.argsort(mu_samples)
            sample_count = int(a.get('sample_count', 0)) + int(b.get('sample_count', 0))
        else:
            mu_min = float(min(a.get('mu_min', 0.0), b.get('mu_min', 0.0)))
            mu_max = float(max(a.get('mu_max', 0.0), b.get('mu_max', 0.0)))
            center_sample_idx = a.get('center_sample_idx', -1)
            center_z = a.get('center_z', None)
            center_mu = a.get('center_mu', 0.0)
            center_pi = a.get('center_pi', 0.0)
            sorted_idx_by_mu = a.get('sorted_idx_by_mu', None)
            sample_count = int(a.get('sample_count', 0)) + int(b.get('sample_count', 0))

        # 写回
        a['samples'] = samples_new if has_numeric else {}
        a['mu_min'] = mu_min
        a['mu_max'] = mu_max
        a['center_sample_idx'] = center_sample_idx
        a['center_z'] = center_z
        a['center_mu'] = center_mu
        a['center_pi'] = center_pi
        a['sorted_idx_by_mu'] = sorted_idx_by_mu if has_numeric else None
        a['sample_count'] = sample_count
        # 其余字段（center_su, hop1_counts, hop2_counts, nmr_range）保持不变

    base_lib['templates'] = base_templates

    # 合并 center_index（去重）
    ci_base = base_lib.get('center_index', {})
    ci_add = add_lib.get('center_index', {})
    for su, keys in ci_add.items():
        if su not in ci_base:
            ci_base[su] = list(keys)
        else:
            s = set(tuple(k) for k in ci_base[su])
            for k in keys:
                tk = tuple(k)
                if tk not in s:
                    ci_base[su].append(k)
                    s.add(tk)
            # 排序稳定
            try:
                ci_base[su] = sorted(ci_base[su], key=lambda x: tuple(x))
            except Exception:
                pass
    base_lib['center_index'] = ci_base

    # 合并 role_index（直接拼接列表）
    ri_base = base_lib.get('role_index', {})
    ri_add = add_lib.get('role_index', {})
    for k, lst in ri_add.items():
        if k in ri_base:
            ri_base[k].extend(lst)
        else:
            ri_base[k] = list(lst)
    base_lib['role_index'] = ri_base

    base_lib['total_templates'] = len(base_lib.get('templates', {}))
    base_lib['library_type'] = add_lib.get('library_type', base_lib.get('library_type', '3hop_enhanced_v2'))
    if 'su_max_degrees' not in base_lib:
        base_lib['su_max_degrees'] = add_lib.get('su_max_degrees')
    return base_lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build or filter a V2 3-hop enhanced subgraph template library.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ========== build 子命令 ==========
    build_parser = subparsers.add_parser('build', help='构建新的模板库')
    build_parser.add_argument('--pt_dir', required=True,
                      help='Directory containing the source *.pt graph files.')
    build_parser.add_argument('--g2s_ckpt', required=True,
                      help='Path to the pre-trained G2S VAE model checkpoint.')
    build_parser.add_argument('--out', default='subgraph_library_3hop_v2.pt',
                      help='Output path for the generated V2 3-hop library file.')
    build_parser.add_argument('--hid_dim', type=int, default=384, help='Hidden dimension of G2S model.')
    build_parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension of G2S model.')
    build_parser.add_argument('--k_hop', type=int, default=3, help='Subgraph radius (should be 3 for enhanced templates).')
    build_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing.')
    build_parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    build_parser.add_argument('--mols_per_chunk', type=int, default=2000, help='分块规模（分子数），避免一次性占满内存')
    build_parser.add_argument('--skip_role_index', action='store_true', help='跳过 hop1/hop2 角色索引采样以提速降内存')
    build_parser.add_argument('--amp', action='store_true', help='启用半精度推理（需要GPU）')
    build_parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader 预取批次数（workers>0时生效）')
    build_parser.add_argument('--base_lib', type=str, default=None,
                      help='已有库文件路径；提供则将此次构建的库合并进该库')
    build_parser.add_argument('--merge_in_place', action='store_true',
                      help='与 --base_lib 配合使用；合并结果直接覆盖保存到该路径')
    
    # ========== filter 子命令 ==========
    filter_parser = subparsers.add_parser('filter', help='根据1-hop连接规则筛选现有模板库')
    filter_parser.add_argument('--input', '-i', required=True,
                      help='输入库文件路径 (subgraph_library.pt)')
    filter_parser.add_argument('--output', '-o', required=True,
                      help='输出筛选后的库文件路径')
    filter_parser.add_argument('--strict', action='store_true', default=True,
                      help='严格模式（默认开启，只保留完全合法的模板）')
    filter_parser.add_argument('--verbose', '-v', action='store_true', default=True,
                      help='输出详细信息')
    
    # ========== rules 子命令 ==========
    rules_parser = subparsers.add_parser('rules', help='打印SU连接规则摘要')
    
    args = parser.parse_args()
    
    # 处理filter命令
    if args.command == 'filter':
        filter_subgraph_library(
            input_lib_path=args.input,
            output_lib_path=args.output,
            strict_mode=args.strict,
            verbose=args.verbose
        )
        exit(0)
    
    # 处理rules命令
    if args.command == 'rules':
        print_connection_rules_summary()
        exit(0)
    
    # 处理build命令（或无子命令的旧式调用）
    if args.command == 'build':
        pass  # 继续执行下面的build逻辑
    elif args.command is None:
        # 向后兼容：无子命令时检查是否有旧式参数
        # 重新解析为旧式参数
        parser_legacy = argparse.ArgumentParser()
        parser_legacy.add_argument('--pt_dir', required=True)
        parser_legacy.add_argument('--g2s_ckpt', required=True)
        parser_legacy.add_argument('--out', default='subgraph_library_3hop_v2.pt')
        parser_legacy.add_argument('--hid_dim', type=int, default=384)
        parser_legacy.add_argument('--latent_dim', type=int, default=16)
        parser_legacy.add_argument('--k_hop', type=int, default=3)
        parser_legacy.add_argument('--batch_size', type=int, default=32)
        parser_legacy.add_argument('--num_workers', type=int, default=4)
        parser_legacy.add_argument('--mols_per_chunk', type=int, default=2000)
        parser_legacy.add_argument('--skip_role_index', action='store_true')
        parser_legacy.add_argument('--amp', action='store_true')
        parser_legacy.add_argument('--prefetch_factor', type=int, default=2)
        parser_legacy.add_argument('--base_lib', type=str, default=None)
        parser_legacy.add_argument('--merge_in_place', action='store_true')
        args = parser_legacy.parse_args()
    else:
        parser.print_help()
        exit(1) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Building V2 3-hop enhanced template library with k_hop={args.k_hop}")

    lib = build_3hop_library_v2(
        args.pt_dir, 
        args.g2s_ckpt,
        k_hop=args.k_hop,
        hid_dim=args.hid_dim, 
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        mols_per_chunk=args.mols_per_chunk,
        skip_role_index=args.skip_role_index,
        amp=args.amp,
        prefetch_factor=args.prefetch_factor
    )
    
    # 若存在 base_lib，则执行合并
    out_path = args.out
    if getattr(args, 'base_lib', None):
        try:
            base = torch.load(args.base_lib, map_location='cpu', weights_only=False)
            merged = merge_3hop_libraries_v2(base, lib)
            lib = merged
            if args.merge_in_place:
                out_path = args.base_lib
            print(f"已将新库与基库合并，输出路径将为: {out_path}")
        except Exception as e:
            print(f"  -> 警告: 读取/合并 base_lib 失败，将仅保存新库: {e}")

    # 保存库文件
    torch.save(lib, out_path)
    print(f"\nV2 3-hop模板库已保存到: {out_path}")
    print(f" -> 找到 {lib['total_templates']} 个独特的多重集模板桶")
    print(f" -> 库类型: {lib['library_type']}")
    
    # 打印一些统计信息
    template_sizes = [info['sample_count'] for info in lib['templates'].values()]
    print(f" -> 模板样本数统计: 最小={min(template_sizes)}, 最大={max(template_sizes)}, 平均={np.mean(template_sizes):.1f}")
    
    # V2: 分析center_index覆盖度
    center_index = lib['center_index']
    print(f" -> 中心SU覆盖度: {len(center_index)} 种中心SU类型")
    for center_su, template_keys in center_index.items():
        su_name = [name for name, _ in SU_DEFS][center_su]
        print(f"   {su_name}: {len(template_keys)} 个模板桶")
    
    # 生成并保存多重集统计CSV
    su_names = [name for name, _ in SU_DEFS]
    rows = []
    for tpl_key, tpl_info in lib['templates'].items():
        center_su_idx, hop1_multiset, hop2_multiset = tpl_key
        # 按照 su+1-hop+2-hop 的键顺序本身已经排序，直接用
        hop1_str = f"[{','.join(str(i) for i in hop1_multiset)}]"
        hop2_str = f"[{','.join(str(i) for i in hop2_multiset)}]"
        rows.append({
            'center_su_idx': int(center_su_idx),
            'center_su': su_names[center_su_idx],
            'hop1_multiset': hop1_str,
            'hop2_multiset': hop2_str,
            'sample_count': int(tpl_info['sample_count']),
            'mu_min': float(tpl_info['mu_min']),
            'mu_max': float(tpl_info['mu_max']),
            'center_mu_median': float(tpl_info['center_mu'])
        })

    if rows:
        out_dir = os.path.dirname(out_path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        # 根据 (center_su_idx, hop1_multiset, hop2_multiset) 排序
        rows_sorted = sorted(rows, key=lambda r: (r['center_su_idx'], r['hop1_multiset'], r['hop2_multiset']))
        df = pd.DataFrame(rows_sorted, columns=['center_su_idx','center_su','hop1_multiset','hop2_multiset','sample_count','mu_min','mu_max','center_mu_median'])
        csv_path = os.path.join(out_dir, 'su_multiset_distribution.csv')
        df.to_csv(csv_path, index=False)
        print(f" -> 多重集模板统计已保存(已排序): {csv_path}")
        
    # 保存索引文件（便于快速加载）
    index_dir = os.path.join(out_dir, 'indexes')
    os.makedirs(index_dir, exist_ok=True)
    
    with open(os.path.join(index_dir, 'center_index.pkl'), 'wb') as f:
        pickle.dump(lib['center_index'], f)
    with open(os.path.join(index_dir, 'role_index.pkl'), 'wb') as f:
        pickle.dump(lib['role_index'], f)
    print(f" -> 索引文件已保存: {index_dir}/")