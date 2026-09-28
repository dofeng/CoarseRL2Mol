import sys
from pathlib import Path
# ✅ 添加父目录到Python路径，以便导入model模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple, Optional

from model.g2s_model import NMR_VAE, LocalNMRDataset
from model.coarse_graph import NUM_SU_TYPES, PPM_AXIS


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
