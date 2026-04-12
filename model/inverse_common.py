import math
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Any, List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt

from .coarse_graph import (
    SU_DEFS, E_SU, NUM_SU_TYPES, PPM_AXIS, PPM_STEP
)
# ============================================================================
# 全局常量配置
# ============================================================================

TOL_CHONSX = torch.tensor([0.02, 0.05, 0.08, 0.08, 0.08, 0.0], dtype=torch.float)
TOL_VEC_L0 = torch.tensor([0.03, 0.06, 0.10, 0.10, 0.10, 0.0], dtype=torch.float)  
TOL_VEC_L1 = torch.tensor([0.025, 0.055, 0.09, 0.09, 0.09, 0.0], dtype=torch.float)
TOL_VEC_L4 = torch.tensor([0.02, 0.05, 0.08, 0.08, 0.08, 0.0], dtype=torch.float)  

# SU分类（用于快速索引）
SU_CARBONYL = [0, 1, 2, 3, 4]  
SU_AROMATIC = [5, 6, 7, 8, 9, 10, 11, 12, 13] 
SU_UNSATURATED = [14, 15, 16, 17, 18] 
SU_ALIPHATIC = [19, 20, 21, 22, 23, 24, 25] 
SU_HETERO_N = [26, 27]  
SU_HETERO_O = [28, 29] 
SU_HETERO_S = [30, 31] 
SU_HETERO_X = [32] 

# PPM分段定义
PPM_SEGMENTS = {
    'carbonyl': (160.0, 240.0),  
    'aromatic': (90.0, 160.0),   
    'aliphatic': (0.0, 90.0)     
}

MAX_CANDS_PER_NODE = 600  

# Layer4配置
L4_CONFIG = {
    'r2_target': 0.985,                    # 目标R²阈值
    'max_outer_iters': 120,                # 最大外循环次数
    'spike_drop_min': 0.02,                # 尖峰下降最小阈值
    'segment_limits_replace': {'carbonyl': 3, 'aromatic': 6, 'aliphatic': 4},
    'segment_limits_addrm': {'carbonyl': 3, 'aromatic': 6, 'aliphatic': 4},
    'max_no_improve_cycles': 3,            # 连续无改善轮数触发放宽
    'tolerance_relax_step': 0.01,          # 容差放宽步长
}

# 软约束优先级
CONSTRAINT_PRIORITY = {
    'element_C': 100,    
    'element_X': 100,     
    'element_H': 80,      
    'element_O': 60,      
    'element_N': 60,       
    'element_S': 60,     
    'degree': 70,          
    'quota': 50,           
    'semantic': 90,        
    'unsat_ratio': 75,     
}


def get_aliphatic_carbon_policy(E_target: torch.Tensor) -> Dict[str, float]:
    """
    根据 H/C 比例决定脂肪碳初始化和 Layer4 上限策略。

    规则:
      - H/C < 0.9:  初始化保持 0.82 * xN, Layer4 上限 0.9 * xN
      - 0.9 <= H/C < 1.1: 初始化提高到 0.9 * xN, Layer4 上限 1.0 * xN
      - H/C >= 1.1: 初始化提高到 1.0 * xN, Layer4 上限 1.1 * xN
    """
    try:
        e = E_target.detach().cpu().flatten().float() if hasattr(E_target, 'detach') else torch.tensor(E_target, dtype=torch.float).flatten()
        c_count = float(e[0].item()) if int(e.numel()) > 0 else 0.0
        h_count = float(e[1].item()) if int(e.numel()) > 1 else 0.0
    except Exception:
        c_count = 0.0
        h_count = 0.0

    hc_ratio = float(h_count / max(c_count, 1e-8)) if c_count > 0.0 else 0.0
    if float(hc_ratio) < 0.9:
        init_scale = 0.82
        upper_scale = 0.90
    elif float(hc_ratio) < 1.1:
        init_scale = 0.90
        upper_scale = 1.00
    else:
        init_scale = 1.00
        upper_scale = 1.10

    return {
        'hc_ratio': float(hc_ratio),
        'init_aliphatic_scale': float(init_scale),
        'layer4_aliphatic_upper_scale': float(upper_scale),
    }

# ============================================================================
# 结构单元连接规则（基于化学语义）
# ============================================================================

# SU连接度定义
SU_CONNECTION_DEGREE = {
    0: 2,   # Amide_Group: -C(=O)-NH-
    1: 1,   # Carboxylic_Acid: -COOH
    2: 2,   # Ester_Group: -C(=O)-O-
    3: 2,   # Aldehyde_Ketone_C: -C(=O)- 可桥接或末端
    4: 1,   # Nitrile_C: -C≡N
    5: 3,   # O_Substituted_Aro_C: 芳香-氧连接
    6: 3,   # N_Substituted_Aro_C: 芳香-氮连接
    7: 3,   # S_Substituted_Aro_C: 芳香-硫连接
    8: 3,   # X_Substituted_Aro_C: 芳香-卤连接
    9: 3,   # Keto_Substituted_Aro_C: 芳香-羰基连接
    10: 3,  # Aryl_Substituted_Aro_C: 芳基取代
    11: 3,  # Alkyl_Substituted_Aro_C: 烷基取代
    12: 3,  # Aromatic_Bridgehead_C: 稠环桥头
    13: 2,  # Carbocyclic_Aro_CH: 芳香CH
    14: 3,  # Vinyllic_Cq: >C=
    15: 2,  # Vinyllic_CH: -HC=
    16: 1,  # Vinyllic_CH2: =CH2
    17: 2,  # Alkynyl_Cq: -C≡
    18: 1,  # Alkynyl_CH: ≡CH
    19: 2,  # Alcohol_Ether_C: -CH2-O
    20: 2,  # Amine_C: -CH2-N
    21: 2,  # Halogenated_C: -CH2-X
    22: 1,  # Alkyl_CH3: -CH3
    23: 2,  # Alkyl_CH2: -CH2-
    24: 3,  # Alkyl_CH: -CH<
    25: 4,  # Alkyl_Cq: >C<
    26: 2,  # Heterocyclic_N: 吡啶氮
    27: 2,  # Amine_Nitrogen: -NH-
    28: 1,  # Hydroxyl_O: -OH
    29: 2,  # Ether_O: -O-
    30: 2,  # Heterocyclic_S: 噻吩硫
    31: 2,  # Thioether_S: -S-
    32: 1,  # Halogen_X: -X
}

# 末端结构单元
TERMINAL_SU = {1, 4, 16, 18, 22, 28, 32}

# ============================================================================
# 1-hop端口组合规则
# ============================================================================
HOP1_PORT_COMBINATIONS: Dict[int, List[Set[int]]] = {
    0: [{9, 23, 24, 25, 22, 14, 15, 17}, {6, 20}], 
    1: [{9, 23, 24, 25, 19, 20, 21, 14, 15, 17}], 
    2: [{9, 23, 24, 25, 22, 19, 20, 21, 14, 15, 17}, {5, 19}], 
    3: [{9, 23, 24, 25, 22, 19, 20, 21, 14, 15, 17}, {9, 23, 24, 25, 19, 20, 21, 14, 15, 17}], 
    4: [{23, 24, 25, 10}], 
    5: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {2, 28, 29}], 
    6: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {0, 27}], 
    7: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {31}],  
    8: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {32}], 
    9: [{13, 12, 11, 10, 5, 6, 7, 8, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 26, 30}, {0, 1, 2, 3}], 
    10: [{13, 12, 11, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 5, 6, 7, 8, 9, 26, 30}, {4, 10}], 
    11: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {23, 24, 25, 22, 19, 20, 21}], 
    12: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {12}], 
    13: [{13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}, {13, 12, 11, 10, 5, 6, 7, 8, 9, 26, 30}], 
    14: [{23, 24, 25, 22, 19, 20, 21, 2, 1, 0, 3, 4}, {23, 24, 25, 22, 19, 20, 21, 2, 1, 0, 3, 4}, {14, 15, 16}],  
    15: [{23, 24, 25, 22, 19, 20, 21, 2, 1, 0, 3, 4}, {14, 15, 16}], 
    16: [{14, 15}], 
    17: [{23, 24, 25, 19, 20, 21, 2, 0, 3}, {17, 18}], 
    18: [{17}], 
    19: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {2, 28, 29, 31}], 
    20: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {0, 27}], 
    21: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {32}], 
    22: [{25, 24, 19, 20, 21, 23, 11,  2, 3, 1, 0, 14, 15, 17}], 
    23: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}], 
    24: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}], 
    25: [{23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}, {23, 11, 22, 24, 25, 19, 20, 21, 2, 3, 1, 0, 14, 15, 17}], 
    26: [{13, 12, 11, 10, 5, 6, 7, 8, 9}, {13, 12, 11, 10, 5, 6, 7, 8, 9}], 
    27: [{6, 20}, {6, 20}], 
    28: [{5, 19}], 
    29: [{5, 19}, {5, 19}],  
    30: [{13, 12, 11, 10, 5, 6, 7, 8, 9}, {13, 12, 11, 10, 5, 6, 7, 8, 9}], 
    31: [{7, 19}, {7, 19}],  
    32: [{8, 21}],  
}

# 每个SU的所有端口允许的邻居类型的并集
SU_FIXED_CONNECTIONS = {
    su: list(set().union(*port_sets)) if port_sets else []
    for su, port_sets in HOP1_PORT_COMBINATIONS.items()
}

# 外接结构要求
SU_EXTERNAL_CONNECTIONS = {
    5: [2, 28, 29],  
    6: [0, 27],      
    7: [31],         
    8: [32],         
    9: [0, 1, 2, 3], 
    10: [4, 10],     
    11: [23, 24, 25, 22, 19, 20, 21],  
    19: [2, 28, 29, 31],  
    20: [0, 27],   
    21: [31],       
}

# 不饱和键配对
UNSATURATED_PAIRS = {
    14: [15, 16, 14],  
    15: [15, 16, 14],  
    16: [15, 14],      
    17: [17, 18],      
    18: [17],        
}

# 禁止连接规则
FORBIDDEN_CONNECTIONS = {
    'terminal_to_terminal': True,  
    'double_terminal_bridge': [3, 14, 15, 17, 23, 24, 25, 19, 20, 21, 29, 27, 31], 
    '10_10_must_pair': True,  
    'aromatic_no_external': [12, 13, 26, 30],
}

# 连接优先级
SU_CONNECTION_PRIORITY = {
    # 羰基/腈优先级
    'carbonyl_priority': [9, 22, 23, 24, 25, 14, 15, 17],
    # 芳香环成环优先级
    'aromatic_ring_priority': [13, 12, 11, 10, 5, 6, 7, 8, 9],
    # 脂肪链优先级
    'aliphatic_priority': [23, 11, 22, 24, 25, 19, 20, 21],
    # 不饱和结构优先级
    'unsaturated_saturated_priority': [23, 24, 25, 22, 19, 20, 21, 2, 1, 0, 3, 4],
}

def validate_connection(center_su: int, neighbor_su: int, E_target: torch.Tensor) -> bool:
    """
    验证两个SU之间的连接是否符合化学语义
    """
    if center_su in TERMINAL_SU and neighbor_su in TERMINAL_SU:
        return False
    
    if center_su in SU_FIXED_CONNECTIONS:
        allowed = SU_FIXED_CONNECTIONS[center_su]
        if isinstance(allowed, dict):
            all_allowed = set()
            for side_list in allowed.values():
                all_allowed.update(side_list)
            if neighbor_su not in all_allowed:
                return False
        elif neighbor_su not in allowed:
            return False
    
    if E_target[5].item() <= 0:
        if center_su in [8, 21] or neighbor_su in [8, 21, 32]:
            return False
    
    if center_su in UNSATURATED_PAIRS:
        if neighbor_su not in UNSATURATED_PAIRS[center_su]:
            if center_su in [14, 15]:
                return neighbor_su not in [17, 18]
            elif center_su in [17, 18]:
                return neighbor_su not in [14, 15, 16]
    
    return True


def check_external_connection_requirement(center_su: int, hop1_counter: Counter) -> Tuple[bool, str]:
    """
    检查芳香取代位点和特殊脂肪碳的外接要求
    """
    if center_su not in SU_EXTERNAL_CONNECTIONS:
        return True, ""
    
    required_external = SU_EXTERNAL_CONNECTIONS[center_su]
    hop1_types = set(hop1_counter.keys())

    if not any(ext_su in hop1_types for ext_su in required_external):
        return False, f"SU {center_su} requires external connection to {required_external}"
    
    return True, ""


# ============================================================================
# NMR工具函数
# ============================================================================

def lorentzian(x: torch.Tensor, mu: float, pi: float, gamma: float = 1.0) -> torch.Tensor:
    """洛伦兹峰形函数"""
    return pi * gamma**2 / ((x - mu)**2 + gamma**2)


def lorentzian_spectrum(mus: torch.Tensor, pis: torch.Tensor, ppm_axis: torch.Tensor,
                         hwhm: float = 1.0) -> torch.Tensor:
    if mus.dim() == 1:
        mus = mus.unsqueeze(1)
    if pis.dim() == 1:
        pis = pis.unsqueeze(1)
    hwhm_sq = hwhm ** 2
    delta_ppm_sq = (ppm_axis.unsqueeze(0) - mus) ** 2
    all_peaks = pis * hwhm_sq / (delta_ppm_sq + hwhm_sq)
    return all_peaks.sum(dim=0)


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """计算R²评分"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return float(r2.item())


def compute_segment_r2(y_true: torch.Tensor, y_pred: torch.Tensor, 
                       ppm_axis: torch.Tensor, segment_name: str) -> float:
    """计算特定PPM分段的R²"""
    lo, hi = PPM_SEGMENTS[segment_name]
    mask = (ppm_axis >= lo) & (ppm_axis <= hi)
    if not mask.any():
        return 0.0
    return compute_r2_score(y_true[mask], y_pred[mask])


def resample_spectrum_to_ppm_axis(
    ppm_values: np.ndarray | torch.Tensor | List[float],
    intensities: np.ndarray | torch.Tensor | List[float],
    ppm_axis: torch.Tensor = PPM_AXIS,
) -> torch.Tensor:
    """
    将任意 ppm 采样顺序/间距的谱图重采样到项目统一的 PPM_AXIS 上。

    这样可避免直接忽略 CSV 第一列 ppm 而导致升降序或非等间距输入被错误解释。
    """
    ppm_np = np.asarray(ppm_values, dtype=np.float64).reshape(-1)
    intensity_np = np.asarray(intensities, dtype=np.float64).reshape(-1)

    mask = np.isfinite(ppm_np) & np.isfinite(intensity_np)
    ppm_np = ppm_np[mask]
    intensity_np = intensity_np[mask]

    if ppm_np.size == 0 or intensity_np.size == 0:
        return torch.zeros_like(ppm_axis, dtype=torch.float)

    order = np.argsort(ppm_np)
    ppm_sorted = ppm_np[order]
    intensity_sorted = intensity_np[order]

    ppm_unique, inverse = np.unique(ppm_sorted, return_inverse=True)
    if ppm_unique.size != ppm_sorted.size:
        sums = np.zeros_like(ppm_unique, dtype=np.float64)
        counts = np.zeros_like(ppm_unique, dtype=np.float64)
        np.add.at(sums, inverse, intensity_sorted)
        np.add.at(counts, inverse, 1.0)
        intensity_sorted = sums / np.maximum(counts, 1.0)
        ppm_sorted = ppm_unique

    axis_np = ppm_axis.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y_interp = np.interp(axis_np, ppm_sorted, intensity_sorted, left=0.0, right=0.0)
    return torch.tensor(y_interp, dtype=torch.float)


def normalize_spectrum_to_carbon_count(
    spectrum: torch.Tensor,
    carbon_count: float,
    ppm_step: float = PPM_STEP,
) -> torch.Tensor:
    """
    将输入谱图面积缩放到与训练数据一致的目标面积: pi * num_carbons。
    """
    y = spectrum.detach().clone().float().flatten()
    carbon = max(0.0, float(carbon_count))
    if carbon <= 0.0:
        return y

    current_area = float(y.sum().item()) * float(ppm_step)
    if current_area <= 1e-8:
        return y

    target_area = float(math.pi) * float(carbon)
    scale = float(target_area / current_area)
    return y * scale


def fit_spectrum_scale(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    nonnegative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算最优全局缩放因子 alpha，使 alpha * y_pred 最贴近 y_true。
    """
    y_true = y_true.flatten().float()
    y_pred = y_pred.flatten().float().to(y_true.device)

    n = int(min(y_true.numel(), y_pred.numel()))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    denom = torch.sum(y_pred * y_pred).clamp(min=1e-8)
    alpha = torch.sum(y_true * y_pred) / denom
    if bool(nonnegative):
        alpha = torch.clamp(alpha, min=0.0)
    y_fit = alpha * y_pred
    return y_fit, alpha


def evaluate_spectrum_reconstruction(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ppm_axis: Optional[torch.Tensor] = None,
    fit_scale: bool = True,
    nonnegative_alpha: bool = True,
) -> Dict[str, Any]:
    """
    统一的谱图评估入口。

    返回 raw recon、alpha-fitted recon、R² 与分段 R²，确保各层评估口径一致。
    """
    y_true_t = y_true.flatten().float()
    y_pred_t = y_pred.flatten().float().to(y_true_t.device)

    n = int(min(y_true_t.numel(), y_pred_t.numel()))
    y_true_t = y_true_t[:n]
    y_pred_t = y_pred_t[:n]

    ppm_eval = None
    if ppm_axis is not None:
        ppm_eval = ppm_axis.flatten().float().to(y_true_t.device)[:n]

    if bool(fit_scale):
        y_fit_t, alpha_t = fit_spectrum_scale(
            y_true_t,
            y_pred_t,
            nonnegative=bool(nonnegative_alpha),
        )
    else:
        y_fit_t = y_pred_t
        alpha_t = torch.tensor(1.0, dtype=y_true_t.dtype, device=y_true_t.device)

    out: Dict[str, Any] = {
        'S_target': y_true_t,
        'S_recon_raw': y_pred_t,
        'S_fit': y_fit_t,
        'alpha': float(alpha_t.detach().cpu().item()),
        'r2': float(compute_r2_score(y_true_t, y_fit_t)),
    }

    if ppm_eval is not None and int(ppm_eval.numel()) == int(y_true_t.numel()):
        out['ppm_axis'] = ppm_eval
        out['r2_carbonyl'] = float(compute_segment_r2(y_true_t, y_fit_t, ppm_eval, 'carbonyl'))
        out['r2_aromatic'] = float(compute_segment_r2(y_true_t, y_fit_t, ppm_eval, 'aromatic'))
        out['r2_aliphatic'] = float(compute_segment_r2(y_true_t, y_fit_t, ppm_eval, 'aliphatic'))

    return out

def visualize_su_distribution(su_hist: torch.Tensor, layer_name: str, 
                               save_dir: str = 'inverse_result'):
    """可视化SU分布（保持与v2兼容）"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    su_names = [name for name, _ in SU_DEFS]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(NUM_SU_TYPES)
    counts = su_hist.detach().cpu().numpy()
    
    colors = ['#d62728' if i in SU_CARBONYL else 
              '#2ca02c' if i in SU_AROMATIC else 
              '#1f77b4' for i in range(NUM_SU_TYPES)]
    
    ax.bar(x, counts, color=colors, alpha=0.7)
    ax.set_xlabel('SU Index', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{layer_name} SU Distribution', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(su_names, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{layer_name.lower()}_su_distribution.png', dpi=300)
    plt.close(fig)


def visualize_spectrum_comparison(S_target: torch.Tensor, S_recon: torch.Tensor,
                                   ppm_axis: torch.Tensor, layer_name: str,
                                   save_dir: str = 'inverse_result'):
    """可视化谱图对比"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ppm = ppm_axis.detach().cpu().numpy()
    target = S_target.detach().cpu().numpy()
    recon = S_recon.detach().cpu().numpy()
    diff = target - recon
    
    # 对比图
    ax1.fill_between(ppm, 0, target, alpha=0.3, color='#1f77b4', label='Target (area)')
    ax1.fill_between(ppm, 0, recon, alpha=0.3, color='#ff7f0e', label='Reconstructed (area)')
    ax1.plot(ppm, target, label='Target', lw=2.0, color='#1f77b4')
    ax1.plot(ppm, recon, label=f'Reconstructed ({layer_name})', lw=2.0, 
             color='#ff7f0e', linestyle='--')
    ax1.set_ylabel('Intensity', fontsize=14)
    ax1.set_title(f'Target vs. Reconstructed ({layer_name})', fontsize=16)
    ax1.invert_xaxis()
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(alpha=0.3)
    
    # 差谱图
    ax2.fill_between(ppm, 0, diff, alpha=0.4, color='crimson', label='Difference (area)')
    ax2.plot(ppm, diff, label='Difference (Target - Recon)', lw=1.5, color='crimson')
    ax2.axhline(0.0, color='black', lw=1.0, alpha=0.6, linestyle=':')
    ax2.set_xlabel('Chemical Shift (ppm)', fontsize=14)
    ax2.set_ylabel('Intensity', fontsize=14)
    ax2.set_title(f'Difference Spectrum ({layer_name})', fontsize=16)
    ax2.invert_xaxis()
    ax2.legend(frameon=False, fontsize=12)
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{layer_name.lower()}_comparison.png', dpi=300)
    plt.close(fig)


# ============================================================================
# 节点数据结构
# ============================================================================

class _NodeV3:
    """
    改进的节点类，存储结构单元节点的完整信息
    """
    __slots__ = ['global_id', 'su_type', 'hop1_su', 'hop2_su', 'hop1_ids',
                 'z_vec', 'mu', 'pi', 'z_history', 
                 'constraint_violations', 'score_components', 'template_key']
    
    def __init__(self, global_id: int, su_type: int,
                 hop1_su: Optional[Counter] = None, 
                 hop2_su: Optional[Counter] = None,
                 z_vec: Optional[torch.Tensor] = None,
                 mu: float = 0.0, pi: float = 1.0):
        self.global_id = global_id
        self.su_type = su_type
        self.hop1_su = hop1_su if hop1_su is not None else Counter()
        self.hop2_su = hop2_su if hop2_su is not None else Counter()
        self.hop1_ids = []  # 存储1-hop邻居的全局ID
        self.z_vec = z_vec if z_vec is not None else torch.zeros(16)
        self.mu = mu
        self.pi = pi
        self.z_history = [z_vec.clone()] if z_vec is not None and z_vec.numel() > 0 else []
        self.constraint_violations = set()
        self.score_components = {}
        self.template_key = None
    
    @property
    def center_su(self):
        """兼容性属性，返回su_type"""
        return self.su_type
    
    @property
    def hop1_counter(self):
        """兼容性属性，返回hop1_su"""
        return self.hop1_su
    
    @property
    def hop2_counter(self):
        """兼容性属性，返回hop2_su"""
        return self.hop2_su
    
    @property
    def z(self):
        """兼容性属性，返回z_vec"""
        return self.z_vec
    
    def get_hop1_degree(self) -> int:
        """获取当前1-hop连接度"""
        return sum(self.hop1_su.values())
    
    def get_max_degree(self) -> int:
        """获取该SU类型的最大连接度"""
        max_deg = SU_CONNECTION_DEGREE.get(self.su_type, 4)
        if isinstance(max_deg, tuple):
            return max_deg[1]
        return max_deg
    
    def is_hop1_complete(self) -> bool:
        """检查1-hop是否已完成分配"""
        return self.get_hop1_degree() >= self.get_max_degree()
    
    def is_hop1_empty(self) -> bool:
        """检查1-hop是否为空"""
        return self.get_hop1_degree() == 0
    
    def remaining_hop1_slots(self) -> int:
        """剩余可分配的1-hop槽位"""
        return max(0, self.get_max_degree() - self.get_hop1_degree())
    
    def has_neighbor(self, neighbor_id: int) -> bool:
        """检查是否已连接到指定邻居"""
        return neighbor_id in self.hop1_ids
    
    def validate_hop1_consistency(self) -> Tuple[bool, List[str]]:
        """
        验证hop1_su与hop1_ids的一致性
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 检查数量一致性
        if len(self.hop1_ids) != sum(self.hop1_su.values()):
            errors.append(f"Node {self.global_id}: hop1_ids长度({len(self.hop1_ids)}) != hop1_su总数({sum(self.hop1_su.values())})")
        
        # 检查度数约束
        current_degree = self.get_hop1_degree()
        max_degree = self.get_max_degree()
        if current_degree > max_degree:
            errors.append(f"Node {self.global_id}: 度数超限 {current_degree} > {max_degree}")
        
        # 检查hop1_ids中无重复
        if len(self.hop1_ids) != len(set(self.hop1_ids)):
            errors.append(f"Node {self.global_id}: hop1_ids中存在重复ID")
        
        # 检查hop1_ids中不含自身
        if self.global_id in self.hop1_ids:
            errors.append(f"Node {self.global_id}: hop1_ids中包含自身ID")
        
        return len(errors) == 0, errors
    
    def __repr__(self):
        return (f"NodeV3(id={self.global_id}, su={self.su_type}, "
                f"hop1={dict(self.hop1_su)}, degree={self.get_hop1_degree()}/{self.get_max_degree()})")
