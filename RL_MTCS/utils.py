import sys
from pathlib import Path
# ✅ 添加父目录到Python路径，以便导入model模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Optional

from model.g2s_model import NMR_VAE, LocalNMRDataset
from model.coarse_graph import NUM_SU_TYPES, E_SU, PPM_AXIS


class NMRPredictor:
    def __init__(self, model_ckpt: str, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = NMR_VAE(hid=384, latent_dim=16).to(self.device)
        state = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_nodewise(self, raw_mol: Dict, k_hop: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回节点级 (mu, pi) 向量（对齐全图节点索引）以及对应的谱图。
        raw_mol 可通过调整 is_carbon 仅选择部分中心节点，实现增量推理。
        """
        n_nodes = int(raw_mol.get("x", torch.zeros(0, NUM_SU_TYPES + 6)).shape[0])
        if n_nodes == 0:
            return torch.zeros_like(PPM_AXIS), torch.zeros(0), torch.zeros(0)
        dataset = LocalNMRDataset([raw_mol], k=k_hop)
        if len(dataset) == 0:
            return torch.zeros_like(PPM_AXIS), torch.zeros(n_nodes), torch.zeros(n_nodes)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(loader)).to(self.device)
        (mu_pred, pi_pred), _ = self.model(batch)
        
        # ✅ 添加数值稳定性检查
        mu_pred = torch.clamp(mu_pred, min=0.0, max=240.0)  # PPM范围限制
        pi_pred = torch.clamp(pi_pred, min=0.0, max=10.0)   # 强度限制
        
        # 将 batch 顺序映射回原图节点索引
        orig_centers = batch.original_center_id.view(-1).long().tolist()
        mu_full = torch.zeros(n_nodes, device=self.device)
        pi_full = torch.zeros(n_nodes, device=self.device)
        
        # ✅ 添加索引边界检查
        for i, idx in enumerate(orig_centers):
            if 0 <= idx < n_nodes:  # 边界检查
                mu_full[idx] = mu_pred[i]
                pi_full[idx] = pi_pred[i]
            else:
                print(f"⚠️  警告: 节点索引越界 idx={idx}, n_nodes={n_nodes}")
        
        # 汇总谱图（仅使用非零强度条目）
        valid = pi_full > 0
        if valid.sum() > 0:
            S = lorentzian_spectrum(mu_full[valid], pi_full[valid], PPM_AXIS.to(self.device))
        else:
            S = torch.zeros_like(PPM_AXIS).to(self.device)
        
        return S.detach().cpu(), mu_full.detach().cpu(), pi_full.detach().cpu()

    @torch.no_grad()
    def predict_spectrum(self, raw_mol: Dict, k_hop: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S, mu_full, pi_full = self.predict_nodewise(raw_mol, k_hop=k_hop)
        return S, mu_full, pi_full


def lorentzian_spectrum(mus: torch.Tensor, pis: torch.Tensor, ppm_axis: torch.Tensor, hwhm: float = 1.0) -> torch.Tensor:
    if mus.dim() == 1:
        mus = mus.unsqueeze(1)
    if pis.dim() == 1:
        pis = pis.unsqueeze(1)
    hwhm_sq = hwhm ** 2
    delta_ppm_sq = (ppm_axis.unsqueeze(0) - mus) ** 2
    all_peaks = pis * hwhm_sq / (delta_ppm_sq + hwhm_sq)
    return all_peaks.sum(dim=0)


# ============================================================
# SU类型到元素组成的映射 [C, H, O, N, S, X]
# 基于 coarse_graph.py 中的 SU_DEFS 定义
# ============================================================
SU_ELEMENT_COUNTS = {
    # A类: 核心官能团 (0-2)
    0: [1, 0, 1, 1, 0, 0],  # Amide_Group: C(=O)N
    1: [1, 1, 2, 0, 0, 0],  # Carboxylic_Acid: C(=O)OH
    2: [1, 0, 2, 0, 0, 0],  # Ester_Group: C(=O)O
    # B类: 其他单原子官能团 (3-4)
    3: [1, 0, 1, 0, 0, 0],  # Aldehyde_Ketone_C: C=O
    4: [1, 0, 0, 1, 0, 0],  # Nitrile_C: C#N
    # C类: 芳香碳 (5-13)
    5: [1, 0, 0, 0, 0, 0],  # O_Substituted_Aro_C
    6: [1, 0, 0, 0, 0, 0],  # N_Substituted_Aro_C
    7: [1, 0, 0, 0, 0, 0],  # S_Substituted_Aro_C
    8: [1, 0, 0, 0, 0, 0],  # X_Substituted_Aro_C
    9: [1, 0, 0, 0, 0, 0],  # Keto_Substituted_Aro_C
    10: [1, 0, 0, 0, 0, 0], # Aryl_Substituted_Aro_C (刚性连接碳)
    11: [1, 0, 0, 0, 0, 0], # Alkyl_Substituted_Aro_C (柔性端碳)
    12: [1, 0, 0, 0, 0, 0], # Aromatic_Bridgehead_C
    13: [1, 1, 0, 0, 0, 0], # Carbocyclic_Aro_CH
    # D类: 链状不饱和碳 (14-18)
    14: [1, 0, 0, 0, 0, 0], # Vinyllic_Cq
    15: [1, 1, 0, 0, 0, 0], # Vinyllic_CH
    16: [1, 2, 0, 0, 0, 0], # Vinyllic_CH2
    17: [1, 0, 0, 0, 0, 0], # Alkynyl_Cq
    18: [1, 1, 0, 0, 0, 0], # Alkynyl_CH
    # E类: 饱和脂肪碳 (19-25)
    19: [1, 2, 0, 0, 0, 0], # Alcohol_Ether_C (与O相连)
    20: [1, 2, 0, 0, 0, 0], # Amine_C (与N相连)
    21: [1, 2, 0, 0, 0, 0], # Halogenated_C
    22: [1, 3, 0, 0, 0, 0], # Alkyl_CH3 (甲基)
    23: [1, 2, 0, 0, 0, 0], # Alkyl_CH2 (亚甲基)
    24: [1, 1, 0, 0, 0, 0], # Alkyl_CH (叔碳)
    25: [1, 0, 0, 0, 0, 0], # Alkyl_Cq (季碳)
    # F类: 杂原子节点 (26-32) - 不含碳
    26: [0, 0, 0, 1, 0, 0], # Heterocyclic_N
    27: [0, 1, 0, 1, 0, 0], # Amine_Nitrogen (-NH-)
    28: [0, 1, 1, 0, 0, 0], # Hydroxyl_O (-OH)
    29: [0, 0, 1, 0, 0, 0], # Ether_O (-O-)
    30: [0, 0, 0, 0, 1, 0], # Heterocyclic_S
    31: [0, 0, 0, 0, 1, 0], # Thioether_S
    32: [0, 0, 0, 0, 0, 1], # Halogen_X
}


def connection_graph_to_raw_mol(clusters, graph, global_elem_ratio: Optional[torch.Tensor] = None) -> Dict:
    """
    将粗粒度分子图转换为PyG格式的raw_mol，用于G2S模型推理。
    
    Args:
        clusters: 团簇列表
        graph: ConnectionGraph对象
        global_elem_ratio: 全局元素比例 [C,H,O,N,S,X]，归一化后的比例（6维）
                          如果为None，则使用基于SU的局部元素组成（不推荐）
    
    节点特征格式: [33-dim SU one-hot] + [6-dim 全局元素比例]
    
    关键说明：
    - G2S模型的 LocalNMRDataset 会为每个含碳中心节点切割 3-hop 子图
    - is_carbon 字段标记哪些节点是含碳的（SU类型 0-25）
    - 全局特征 global_feat = concat(su_hist_freq, elem_ratio) (39维)
    """
    site_nodes: List[Tuple[str, int]] = []
    chain_nodes: List[Tuple[str, int]] = []
    uid_to_idx: Dict[str, int] = {}
    x_rows: List[torch.Tensor] = []
    su_type_list: List[int] = []  # 记录每个节点的SU类型，用于正确标记is_carbon
    
    # 如果没有提供全局元素比例，使用默认值（全零，表示未知）
    if global_elem_ratio is None:
        global_elem_ratio = torch.zeros(6)
    else:
        global_elem_ratio = global_elem_ratio.float()
        # 确保是归一化的比例
        if global_elem_ratio.sum() > 1.1:  # 如果是绝对计数，归一化
            global_elem_ratio = global_elem_ratio / global_elem_ratio.sum().clamp(min=1.0)

    def _append_node(uid: str, su_type: int):
        idx = len(uid_to_idx)
        uid_to_idx[uid] = idx
        one_hot = torch.zeros(NUM_SU_TYPES)
        # ✅ 添加边界检查
        if 0 <= su_type < NUM_SU_TYPES:
            one_hot[su_type] = 1.0
        else:
            print(f"⚠️  警告: SU类型越界 su_type={su_type}, NUM_SU_TYPES={NUM_SU_TYPES}")
        # 使用全局元素比例而不是单个SU的元素组成
        feat = torch.cat([one_hot, global_elem_ratio])
        x_rows.append(feat)
        su_type_list.append(su_type)  # 记录SU类型
        return idx

    for c in clusters:
        if not getattr(c, "placed", False):
            continue
        for s in c.sites:
            # 跳过被标记为删除的站点（su_type=-1表示已被合并/删除）
            if s.su_type < 0:
                continue
            uid = s.site_uid
            _append_node(uid, int(s.su_type))
            site_nodes.append((uid, int(s.su_type)))

    def _ensure_chain_node(node):
        # 跳过被标记为删除的节点
        if node.su_type < 0:
            return None
        uid = node.node_uid
        if uid in uid_to_idx:
            return uid_to_idx[uid]
        idx = _append_node(uid, int(node.su_type))
        chain_nodes.append((uid, int(node.su_type)))
        return idx

    edges: set = set()

    for c in clusters:
        if not getattr(c, "placed", False):
            continue
        local_map: Dict[int, int] = {}
        for i, s in enumerate(c.sites):
            # 跳过删除的站点，不加入local_map
            if s.su_type >= 0:
                local_map[i] = uid_to_idx.get(s.site_uid, None)
        for a, b in c.intra_edges:
            ia = local_map.get(a, None)
            ib = local_map.get(b, None)
            if ia is None or ib is None:
                continue
            e = (min(ia, ib), max(ia, ib))
            edges.add(e)

    for e in graph.rigid_edges:
        u = uid_to_idx.get(e.u_site)
        v = uid_to_idx.get(e.v_site)
        if u is None or v is None:
            continue
        edges.add((min(u, v), max(u, v)))

    for e in graph.flexible_edges:
        u = uid_to_idx.get(e.u_site)
        prev = u
        for node in e.chain:
            cur = _ensure_chain_node(node)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur
        v = uid_to_idx.get(e.v_site)
        if prev is not None and v is not None:
            edges.add((min(prev, v), max(prev, v)))

    for e in getattr(graph, 'side_edges', []):
        u = uid_to_idx.get(e.u_site)
        prev = u
        for node in e.chain:
            cur = _ensure_chain_node(node)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur

    for e in getattr(graph, 'branch_edges', []):
        base_uid = e.base_uid
        base = uid_to_idx.get(base_uid)
        prev = base
        for node in e.chain:
            cur = _ensure_chain_node(node)
            if prev is not None and cur is not None:
                edges.add((min(prev, cur), max(prev, cur)))
            prev = cur
        if e.target_uid is not None:
            t = uid_to_idx.get(e.target_uid)
            if prev is not None and t is not None:
                edges.add((min(prev, t), max(prev, t)))

    if not x_rows:
        return {
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "edge_attr": torch.empty(0, 2),
            "x": torch.empty(0, NUM_SU_TYPES + 6),
            "is_carbon": torch.empty(0, dtype=torch.bool),
            "su_hist": torch.zeros(NUM_SU_TYPES),
            "total_atom_counts": torch.zeros(6),
            "y_spectrum": torch.zeros_like(PPM_AXIS),
        }

    x = torch.stack(x_rows)
    
    # ============================================================
    # 关键修复：正确标记含碳节点
    # 含碳SU类型是 0-25（A/B/C/D/E类），杂原子F类是 26-32
    # ============================================================
    su_types_tensor = torch.tensor(su_type_list, dtype=torch.long)
    is_carbon = (su_types_tensor <= 25)  # SU类型 0-25 是含碳的
    
    # 计算SU直方图（绝对计数）
    su_hist = torch.zeros(NUM_SU_TYPES)
    for su_type in su_type_list:
        if 0 <= su_type < NUM_SU_TYPES:
            su_hist[su_type] += 1
    
    # ============================================================
    # 计算精确的全局原子计数（基于SU类型）
    # total_atom_counts = [C, H, O, N, S, X]
    # ============================================================
    total_atom_counts = torch.zeros(6)
    for su_type in su_type_list:
        if su_type in SU_ELEMENT_COUNTS:
            elem_counts = SU_ELEMENT_COUNTS[su_type]
            for i in range(6):
                total_atom_counts[i] += elem_counts[i]

    if edges:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((edge_index.shape[1], 2), dtype=torch.float)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, 2)

    # ============================================================
    # 构建完整的 raw_mol 字典
    # G2S的LocalNMRDataset需要: x, edge_index, edge_attr, is_carbon, su_hist, total_atom_counts
    # ============================================================
    raw = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "x": x,
        "is_carbon": is_carbon,
        "su_hist": su_hist,
        "total_atom_counts": total_atom_counts,
        "y_spectrum": torch.zeros_like(PPM_AXIS),  # 推理时不需要真实谱图
        "uid_map": uid_to_idx,
        "su_types": su_types_tensor,  # 保留SU类型信息便于调试
    }
    return raw

def _parse_elements_expr(expr: str) -> torch.Tensor:
    """
    解析元素表达式，例如 "C10H14O"
    返回一个长度为 6 的 Tensor: [C, H, O, N, S, X]
    """
    import re
    if not expr:
        return torch.zeros(6, dtype=torch.float)
    matches = dict(re.findall(r"([CHONSX])\s*=\s*(\d+)", expr.upper()))
    vec = [int(matches.get(sym, 0)) for sym in ['C', 'H', 'O', 'N', 'S', 'X']]
    return torch.tensor(vec, dtype=torch.float)


def load_target_spectrum_from_csv(
    csv_path: str,
    ppm_column: str = 'ppm',
    intensity_column: str = 'intensity',
    ppm_range: Tuple[float, float] = (0.0, 240.0),
    num_points: int = 2400,
    device: Optional[torch.device] = None,
    elements_expr: Optional[str] = None,
) -> torch.Tensor:
    """
    从CSV文件加载目标核磁谱图
    
    参数:
        csv_path: CSV文件路径
        ppm_column: PPM列名（如果CSV有列名）
        intensity_column: 强度列名（如果CSV有列名）
        ppm_range: PPM范围 (min, max)
        num_points: 数据点数量
        device: 目标设备 (cuda/cpu)
    
    返回:
        torch.Tensor: 谱图张量，形状 (num_points,)
    
    CSV格式支持:
        1. 两列无表头: [ppm值, 强度值]
        2. 两列有表头: 自动识别列名
        3. 单列: 直接作为强度值，PPM均匀分布
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"目标谱图文件不存在: {csv_path}")
    
    # 尝试读取CSV，自动检测分隔符
    # 优先尝试无表头读取（数值型数据文件通常没有表头）
    try:
        # 先尝试无表头读取
        df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
        
        # 检查第一行是否为数值（判断是否真的无表头）
        first_val = df.iloc[0, 0]
        try:
            float(first_val)
            has_header = False
        except (ValueError, TypeError):
            # 第一行不是数值，可能是表头
            has_header = True
            df = pd.read_csv(csv_path, sep=None, engine='python')  # 重新读取，有表头
        
        if len(df.columns) == 1:
            # 单列：直接作为强度
            spectrum = df.iloc[:, 0].values
            
        elif len(df.columns) >= 2:
            # 两列或多列：尝试识别PPM和强度列
            if has_header and ppm_column in df.columns and intensity_column in df.columns:
                spectrum = df[intensity_column].values
            else:
                # 没有匹配的列名或无表头，使用第二列作为强度
                spectrum = df.iloc[:, 1].values
        else:
            raise ValueError("CSV文件格式错误")
            
    except Exception as e:
        raise ValueError(f"无法读取CSV文件: {csv_path} ({e})")
    
    # 验证和处理数据
    spectrum = np.array(spectrum, dtype=np.float32)
    
    # 如果数据点数量不匹配，进行插值
    if len(spectrum) != num_points:
        from scipy.interpolate import interp1d
        
        x_old = np.linspace(ppm_range[0], ppm_range[1], len(spectrum))
        x_new = np.linspace(ppm_range[0], ppm_range[1], num_points)
        
        f = interp1d(x_old, spectrum, kind='linear', fill_value='extrapolate')
        spectrum = f(x_new).astype(np.float32)
    
    # 强度缩放：如果提供了元素表达式，则按 C 数量进行面积对齐
    if elements_expr:
        E = _parse_elements_expr(elements_expr)
        C = float(E[0].item())
        if C > 0:
            target_area = float(np.pi) * C
            current_area = float(spectrum.sum()) * 0.1
            if current_area > 1e-6:
                scale = target_area / current_area
                spectrum = (spectrum * scale).astype(np.float32)
    
    # 转换为Tensor（保持原始强度量纲）
    spectrum_tensor = torch.from_numpy(spectrum)
    
    if device is not None:
        spectrum_tensor = spectrum_tensor.to(device)
    
    return spectrum_tensor


def load_su_counts_from_csv(csv_path: str) -> Dict[int, int]:
    """
    从CSV文件加载结构单元数量
    
    参数:
        csv_path: CSV文件路径
        
    返回:
        Dict[int, int]: {SU编号: 数量}
    
    CSV格式:
        - 两列: [SU编号, 数量] 或 [center_su_idx, count]
        - 有/无表头均可
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"SU数据文件不存在: {csv_path}")
    
    print(f"📦 加载结构单元数据: {csv_path}")
    
    try:
        # 尝试有表头，自动检测分隔符
        df = pd.read_csv(csv_path, sep=None, engine='python')
        
        # 识别列名
        su_col = None
        if 'center_su_idx' in df.columns:
            su_col = 'center_su_idx'
        elif 'su_type' in df.columns:
            su_col = 'su_type'
        elif 'su_idx' in df.columns:
            su_col = 'su_idx'

        count_col = None
        if 'count' in df.columns and su_col is not None and 'count' != su_col:
            count_col = 'count'
        elif 'n' in df.columns and su_col is not None and 'n' != su_col:
            count_col = 'n'
        elif 'num' in df.columns and su_col is not None and 'num' != su_col:
            count_col = 'num'

        if su_col is None:
            if len(df.columns) >= 2:
                su_col = df.columns[0]
                count_col = df.columns[1]
            elif len(df.columns) == 1:
                su_col = df.columns[0]
                count_col = None
            else:
                raise ValueError("CSV格式错误")
        
        # 统计每个SU的数量
        if count_col is None or count_col == su_col:
            su_counts = df[su_col].value_counts().to_dict()
        else:
            su_counts = dict(zip(df[su_col], df[count_col]))
        
    except Exception as e:
        # 无表头模式
        print(f"   ⚠️  有表头读取失败，尝试无表头模式...")
        df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
        
        if len(df.columns) == 1:
            su_counts = df.iloc[:, 0].value_counts().to_dict()
        else:
            su_counts = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    
    # 转换为整数
    su_counts = {int(k): int(v) for k, v in su_counts.items()}
    
    total = sum(su_counts.values())
    print(f"   ✓ 加载 {len(su_counts)} 种结构单元，总数: {total}")
    print(f"   ✓ SU范围: {min(su_counts.keys())} - {max(su_counts.keys())}")
    print(f"   ✅ 结构单元数据加载完成")
    
    return su_counts


def validate_inputs(
    target_spectrum: torch.Tensor,
    su_counts: Dict[int, int],
    g2s_ckpt: str
) -> bool:
    """
    验证输入数据的有效性
    
    返回:
        bool: 数据是否有效
    """
    print("\n🔍 验证输入数据...")
    
    valid = True
    
    # 检查谱图
    if target_spectrum.dim() != 1:
        print(f"   ❌ 谱图维度错误: {target_spectrum.shape}，应为1维")
        valid = False
    else:
        print(f"   ✓ 谱图维度正确: {target_spectrum.shape}")
    
    if target_spectrum.min() < 0:
        print(f"   ⚠️  谱图包含负值: min={target_spectrum.min()}")
    
    # 检查SU数据
    if len(su_counts) == 0:
        print(f"   ❌ SU数据为空")
        valid = False
    else:
        print(f"   ✓ SU数据有效: {len(su_counts)} 种结构单元")
    
    # 检查模型文件
    g2s_path = Path(g2s_ckpt)
    if not g2s_path.exists():
        print(f"   ❌ G2S模型不存在: {g2s_ckpt}")
        valid = False
    else:
        print(f"   ✓ G2S模型存在: {g2s_ckpt}")
    
    if valid:
        print("   ✅ 所有输入数据验证通过\n")
    else:
        print("   ❌ 输入数据验证失败\n")
    
    return valid


# 便捷函数：一键加载所有数据
def load_rl_data(
    target_spectrum_csv: str,
    su_counts_csv: str,
    g2s_ckpt: str,
    device: Optional[torch.device] = None,
    elements_expr: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[int, int], str]:
    """
    一键加载所有强化学习所需数据
    
    参数:
        target_spectrum_csv: 目标谱图CSV路径
        su_counts_csv: 结构单元CSV路径
        g2s_ckpt: G2S模型路径
        device: 计算设备
    
    返回:
        (target_spectrum, su_counts, g2s_ckpt)
    """
    print("="*80)
    print("🚀 加载强化学习数据")
    print("="*80)
    
    # 加载目标谱图
    target_spectrum = load_target_spectrum_from_csv(target_spectrum_csv, device=device, elements_expr=elements_expr)
    
    # 加载SU数据
    su_counts = load_su_counts_from_csv(su_counts_csv)
    
    # 验证数据
    if not validate_inputs(target_spectrum, su_counts, g2s_ckpt):
        raise ValueError("输入数据验证失败，请检查文件格式")
    
    print("="*80)
    print("✅ 所有数据加载完成")
    print("="*80 + "\n")
    
    return target_spectrum, su_counts, g2s_ckpt
