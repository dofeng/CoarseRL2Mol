import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GATv2Conv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_undirected
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

def load_raw_data(path:str):
    if os.path.isdir(path):
        pt_files = sorted(glob.glob(os.path.join(path, "*.pt")))
        raw_list = []
        for f in pt_files:
            try:
                # For loading PyG Data objects, weights_only must be False
                raw_list.append(torch.load(f, weights_only=False))
            except Exception as e:
                print(f"  -> 警告: 载入 '{f}' 失败: {e}")
        print(f"已从目录 '{path}' 载入 {len(raw_list)} 个分子 .pt 文件")
        return raw_list
    else:
        print(f"正在加载数据文件 '{path}' ...")
        # For loading PyG Data objects, weights_only must be False
        data = torch.load(path, weights_only=False)
        if isinstance(data, list):
            return data
        else:
            print(f"  -> 检测到单个分子数据，已将其包装为列表以便兼容处理。")
            return [data]

class MyData(Data):
    """自带 __inc__ 规则, 使得 Batch.collate 时为 center_id 自动加 offset"""

    def __inc__(self, key, value, *args, **kwargs):
        # 需要增加节点数偏移的键
        if key in ("center_id",):
            return self.x.size(0)
        # 不需要偏移的键 (保持为0)
        if key in ("mol_id", "original_center_id"):
            return 0
        # 其他情况使用默认行为
        return super().__inc__(key, value, *args, **kwargs)

# ------------------------------------------------------------
# 1. 数据准备：把中心节点的 k-hop 子图切出来
# ------------------------------------------------------------
class LocalNMRDataset(InMemoryDataset):
    def __init__(self, raw_list, k=3):
        """
        raw_list: 列表，每个元素是 dict，含全分子图以及所有节点的局部真谱向量
        k       : k-hop 半径（默认 3，用于提取完整 3-hop 子图）
        仅为“含碳中心节点”切 3-hop 子图，用于节点级 NMR 监督
        """
        self.k = k
        super().__init__(".", transform=None, pre_transform=None)
        self.data, self.slices = self._process(raw_list)

    # =========================================================
    def _process(self, raw_list):
        all_data = []
        for mol_idx, mol in enumerate(raw_list):
            edge_index   = mol["edge_index"]
            edge_attr    = mol["edge_attr"]
            x_node       = mol["x"]                 
            is_carbon    = mol["is_carbon"]            
            y_mol        = mol.get("y_spectrum", mol.get("y_mol"))  # 兼容旧字段

            # ★ 关键修复：补全反向边
            if edge_index.numel() > 0:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr)

            for center in range(x_node.size(0)):
                # 仅对“含碳中心节点”切 3-hop 子图
                if not bool(is_carbon[center]):
                    continue
                # ---- 抽 k-hop 子图（k=3 默认）----
                sub_nodes, sub_edge_index, mapping, mask = k_hop_subgraph(
                    center, self.k, edge_index, relabel_nodes=True, num_nodes=x_node.size(0))
                data = MyData()
                data.x = x_node[sub_nodes]
                data.edge_index = sub_edge_index
                if edge_attr.numel() == 0:
                    data.edge_attr = edge_attr
                else:
                    data.edge_attr = edge_attr[mask]
                
                # 新增：原始中心节点在全分子图中的索引
                data.original_center_id = torch.tensor([center])
                # mapping 是中心节点在子图中的新索引
                data.center_id  = torch.tensor([mapping])
                # --- 存储节点/分子级标签与索引 ---
                data.is_carbon = torch.tensor([True])
                data.mol_id    = torch.tensor([mol_idx])  
                data.y_mol     = y_mol                   

                # ---- 新增: 真实节点位移/强度 ----
                if "real_ppm" in mol:
                    data.real_ppm       = torch.tensor([float(mol["real_ppm"][center])])      # (1,)
                    data.real_intensity = torch.tensor([float(mol["real_intensity"][center])])# (1,)
                else:
                    data.real_ppm       = torch.tensor([0.0])
                    data.real_intensity = torch.tensor([0.0])

                # ---------------- 全局特征 ----------------
                # 拼接 (33) su_hist 与 (6) 元素计数 → 39 维
                su_hist_global   = mol.get("su_hist", torch.zeros(33))  # 绝对计数
                elem_counts_global = mol.get("total_atom_counts", torch.zeros(6))  # 绝对计数

                # ---------------- 归一化 ----------------
                total_nodes = su_hist_global.sum().clamp(min=1.0)
                total_atoms = elem_counts_global.sum().clamp(min=1.0)
                su_hist_freq = su_hist_global.float() / total_nodes  # 频率 0~1
                elem_ratio    = elem_counts_global.float() / total_atoms  # 原子比例 0~1
                g_feat = torch.cat([su_hist_freq, elem_ratio])  # (39,)
                data.global_feat = g_feat.unsqueeze(0)  # (1,39)

                all_data.append(data)

        return self.collate(all_data)   

# ============================================================
# 2. VAE 网络模块
# ============================================================

class EnvironmentEncoder(nn.Module):
    """改进版编码器: (GINE × (L-1)) + 1×GATv2 + Residual + LayerNorm"""
    def __init__(self, n_node_feat=39, n_edge_feat=2, hid=384, latent_dim=64, n_layers: int = 4):
        super().__init__()

        self.hid = hid
        self.edge_enc = nn.Sequential(
            nn.Linear(n_edge_feat, hid), nn.ReLU(), nn.Linear(hid, hid)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if i < n_layers - 1:
                mlp_in = n_node_feat if i == 0 else hid
                mlp = nn.Sequential(nn.Linear(mlp_in, hid), nn.ReLU(), nn.Linear(hid, hid))
                conv = GINEConv(mlp, edge_dim=hid)
            else:
                # 最后一层使用带多头注意力的 GATv2Conv
                conv = GATv2Conv(hid, hid // 8, heads=8, edge_dim=hid, dropout=0.1)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hid))

        self.head_mu = nn.Linear(hid, latent_dim)
        self.head_logvar = nn.Linear(hid, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr, center_id = data.x, data.edge_index, data.edge_attr, data.center_id

        if edge_attr.numel() == 0:
            e_emb = torch.zeros(edge_index.size(1), self.hid, device=x.device)
        else:
            e_emb = self.edge_enc(edge_attr)

        h = x  # 初始节点特征 (维度 n_node_feat)
        for idx, conv in enumerate(self.convs):
            if idx == 0:
                h = conv(h, edge_index, edge_attr=e_emb)  # 无残差
            else:
                h_new = conv(h, edge_index, edge_attr=e_emb)
                h = h + h_new  # Residual
            h = self.norms[idx](h)

        center_vec = h[center_id]  # (B, hid)
        
        mu = self.head_mu(center_vec)
        logvar = self.head_logvar(center_vec)
        
        return mu, logvar

class SimpleConcatDecoder(nn.Module):
    """轻量解码器：直接拼接局部(SU+z)与全局嵌入，再用 MLP 预测 [mu_raw, pi_logit]。"""
    def __init__(self, n_su_feat=33, latent_dim=16, g_dim=2, hid=384, out_dim=2):
        super().__init__()
        in_dim = n_su_feat + latent_dim + g_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid, hid // 2), nn.ReLU(),
            nn.Linear(hid // 2, out_dim)
        )

    def forward(self, su_feat, z, g_embed):
        x = torch.cat([su_feat, z, g_embed], dim=-1)
        return self.net(x)


class NMRDecoder(nn.Module):
    """保持向后兼容的解码器接口"""
    def __init__(self, n_su_feat=33, latent_dim=16, g_dim=2, hid=384, out_dim=2):
        super().__init__()
        # 使用轻量的拼接解码器，弱化全局通道容量
        self.decoder = SimpleConcatDecoder(n_su_feat, latent_dim, g_dim, hid, out_dim)

    def forward(self, su_feat, z, g_embed):
        return self.decoder(su_feat, z, g_embed)

class NMR_VAE(nn.Module):
    """Graph→Spectrum VAE, 结合了局部环境和全局图信息。"""

    def __init__(self, n_node_feat=39, n_edge_feat=2, n_su_feat=33,
                 hid=384, latent_dim=16, g_dim: int = 2):
        super().__init__()
        self.encoder = EnvironmentEncoder(n_node_feat, n_edge_feat, hid, latent_dim)
        # 仅使用 6 维元素比例 (C,H,O,N,S,X) → 2 维嵌入
        self.global_mlp = nn.Sequential(
            nn.Linear(6, g_dim), nn.ReLU()
        )
        # 解码器改为简单拼接版本
        self.decoder = NMRDecoder(n_su_feat, latent_dim, g_dim=g_dim, hid=hid)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu_latent, logvar_latent = self.encoder(data)
        z = self.reparameterize(mu_latent, logvar_latent)

        center_node_x = data.x[data.center_id]
        su_feat = center_node_x[:, :33]  # (B, 33)

        # -------- 全局特征处理 --------
        device_ = su_feat.device
        if hasattr(data, "global_feat"):
            g_raw_full = data.global_feat.to(device_)
            if g_raw_full.dim() != 2:
                g_raw_full = g_raw_full.view(su_feat.size(0), -1)

            # 仅取后 6 维元素比例
            elem_part = g_raw_full[:, 33:]
            # 若提供的是绝对原子数，则归一化到比例
            if elem_part.max() > 1.1:
                elem_sum = elem_part.sum(dim=1, keepdim=True).clamp(min=1.0)
                elem_part = elem_part / elem_sum
            g_raw = elem_part  # (B,6)
        else:
            # 无全局特征时，仅使用元素比例通道，置零 (B,6)
            g_raw = torch.zeros((su_feat.size(0), 6), device=device_)

        g_embed = self.global_mlp(g_raw)  # (B, 2)
        # 进一步削弱全局通道在训练/推理中的权重占比
        g_embed = 0.02 * g_embed

        pred_params_raw = self.decoder(su_feat, z, g_embed)
        
        mu_pred = pred_params_raw[:, 0]
        intensity_logit = pred_params_raw[:, 1]
        
        # 使用 softplus 保证强度为正
        pi_pred_unnormalized = F.softplus(intensity_logit)

        return (mu_pred, pi_pred_unnormalized), (mu_latent, logvar_latent)

# =====================================================================
# 3. VAE 损失函数
# =====================================================================
def vae_loss_function(pred_params, latent_params, true_data, beta=0.01, lambda_mu=2.0, lambda_pi=1.0):
    """
    改进的 VAE 损失函数，使用归一化标签和重新平衡的权重。
    
    Args:
        pred_params: (mu_pred, pi_pred) 预测的 NMR 参数
        latent_params: (mu_latent, logvar_latent) 潜在变量参数
        true_data: 真实数据，包含 real_ppm, real_intensity, is_carbon
        beta: KL 散度权重 (增加到 0.01)
        lambda_mu: 化学位移损失权重 (增加到 2.0)
        lambda_pi: 强度损失权重
    """
    mu_pred, pi_pred = pred_params
    mu_latent, logvar_latent = latent_params
    
    # 统一形状到 (B,)
    mu_pred = mu_pred.view(-1)
    pi_pred = pi_pred.view(-1)
    real_ppm = true_data.real_ppm.view(-1)
    real_intensity = true_data.real_intensity.view(-1)
    is_carbon = true_data.is_carbon.view(-1).bool()
    
    # 只在有真实信号的含碳节点上计算损失
    mask = is_carbon & (real_intensity > 0)
    
    if not mask.any():
        # 如果一个batch里没有任何有效节点，返回“可反向传播的零损失”
        # 用与 mu_pred 共享图的零标量，避免 backward 报 requires_grad 错误
        zero = (mu_pred * 0.0).sum()
        return zero, zero, zero, zero, zero

    # 标签归一化处理
    # ppm 归一化到 [-1, 1] 范围 (对齐 0–240ppm)
    ppm_center, ppm_scale = 120.0, 120.0
    mu_pred_norm = (mu_pred - ppm_center) / ppm_scale
    real_ppm_norm = (real_ppm - ppm_center) / ppm_scale
    
    # 强度使用 log1p 变换减少尺度差异
    pi_pred_log = torch.log1p(torch.clamp(pi_pred, min=0))
    real_intensity_log = torch.log1p(real_intensity)

    # --- 重构损失 (L_nmr) ---
    loss_mu_norm = F.l1_loss(mu_pred_norm[mask], real_ppm_norm[mask])
    loss_pi = F.l1_loss(pi_pred_log[mask], real_intensity_log[mask])
    
    reconstruction_loss = lambda_mu * loss_mu_norm + lambda_pi * loss_pi

    # --- KL散度损失 (L_kl) ---
    kl_loss = -0.5 * torch.sum(1 + logvar_latent[mask] - mu_latent[mask].pow(2) - logvar_latent[mask].exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    total_loss = reconstruction_loss + beta * kl_loss
    
    # 返回原始尺度的 μ 损失用于监控
    loss_mu_original = F.l1_loss(mu_pred[mask], real_ppm[mask])
    
    return total_loss, reconstruction_loss, kl_loss, loss_mu_original, loss_pi

# ------------------------------------------------------------
# 4. 训练与验证脚本
# ------------------------------------------------------------
def train(model, loader, opt, device, epoch, beta):
    """训练一个 epoch"""
    model.train()
    total_loss, total_recon, total_kl, total_mols = 0.0, 0.0, 0.0, 0

    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        
        pred_params, latent_params = model(data)
        
        loss, recon_loss, kl_loss, _, _ = vae_loss_function(
            pred_params, latent_params, data, beta=beta
        )

        if torch.isnan(loss):
            continue

        loss.backward()
        opt.step()

        num_mols = torch.unique(data.mol_id).numel()
        total_loss += loss.item() * num_mols
        total_recon += recon_loss.item() * num_mols
        total_kl += kl_loss.item() * num_mols
        total_mols += num_mols

    return total_loss / max(total_mols, 1), total_recon / max(total_mols, 1), total_kl / max(total_mols, 1)

@torch.no_grad()
def validate(model, loader, device, beta):
    """验证模型在一个 epoch 上的性能（仅返回损失，不计算逐 epoch 的 R2）。"""
    model.eval()
    total_loss, total_recon, total_kl, total_mols = 0.0, 0.0, 0.0, 0
    last_loss_mu, last_loss_pi = 0.0, 0.0

    for data in loader:
        data = data.to(device)
        
        pred_params, latent_params = model(data)
        
        loss, recon_loss, kl_loss, loss_mu, loss_pi = vae_loss_function(
            pred_params, latent_params, data, beta=beta
        )
        
        if torch.isnan(loss):
            continue

        num_mols = torch.unique(data.mol_id).numel()
        total_loss += loss.item() * num_mols
        total_recon += recon_loss.item() * num_mols
        total_kl += kl_loss.item() * num_mols
        total_mols += num_mols

        last_loss_mu = loss_mu.item()
        last_loss_pi = loss_pi.item()

    return (
        total_loss / max(total_mols, 1),
        total_recon / max(total_mols, 1),
        total_kl / max(total_mols, 1),
        last_loss_mu,
        last_loss_pi,
    )


@torch.no_grad()
def collect_val_regression_data(model, loader, device):
    """在完整验证集上收集用于回归评价的 μ / π 预测与真值。"""
    model.eval()
    all_mu_true, all_mu_pred = [], []
    all_pi_true, all_pi_pred = [], []

    for data in loader:
        data = data.to(device)
        (mu_pred, pi_pred), _ = model(data)

        mu_pred_flat = mu_pred.view(-1)
        pi_pred_flat = pi_pred.view(-1)
        real_ppm = data.real_ppm.view(-1).to(mu_pred_flat.device)
        real_intensity = data.real_intensity.view(-1).to(mu_pred_flat.device)
        is_carbon = data.is_carbon.view(-1).bool().to(mu_pred_flat.device)

        mask_valid = is_carbon & (real_intensity > 0)
        if mask_valid.any():
            all_mu_true.append(real_ppm[mask_valid].detach().cpu())
            all_mu_pred.append(mu_pred_flat[mask_valid].detach().cpu())
            all_pi_true.append(real_intensity[mask_valid].detach().cpu())
            all_pi_pred.append(pi_pred_flat[mask_valid].detach().cpu())

    if not all_mu_true:
        return None, None, None, None

    mu_true = torch.cat(all_mu_true).numpy()
    mu_pred = torch.cat(all_mu_pred).numpy()
    pi_true = torch.cat(all_pi_true).numpy()
    pi_pred = torch.cat(all_pi_pred).numpy()
    return mu_true, mu_pred, pi_true, pi_pred

# -------------------- main -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train NMR VAE Model")
    parser.add_argument("data_path", type=str,
                        help="路径: 1) 单个 .pt 列表文件 或 2) 目录，内含多个*.pt")
    parser.add_argument("--hid", type=int, default=384, help="隐藏层维度")          
    parser.add_argument("--latent_dim", type=int, default=16, help="潜变量维度")
    parser.add_argument("--k_hop", type=int, default=3, help="子图的 hop 数")
    parser.add_argument("--beta", type=float, default=0.01, help="KL 损失的权重 (增强到 0.01)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    data_path = args.data_path

    raw_list = load_raw_data(data_path)
    dataset  = LocalNMRDataset(raw_list, k=args.k_hop)
    
    print(f"dataset[0].x.shape: {dataset[0].x.shape}")

    # ---- 训练/验证拆分 (按分子级别) ----
    import numpy as np
    mol_ids = torch.unique(dataset._data.mol_id).tolist()
    rng = np.random.default_rng(42)
    rng.shuffle(mol_ids)
    split = int(0.9 * len(mol_ids))
    train_mols, val_mols = mol_ids[:split], mol_ids[split:]

    train_idx = [i for i,d in enumerate(dataset) if d.mol_id.item() in train_mols]
    val_idx   = [i for i,d in enumerate(dataset) if d.mol_id.item() in val_mols]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=12)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=12)
    
    print(f"数据集拆分: {len(train_set)} 子图(训练), {len(val_set)} 子图(验证)，对应分子 {len(train_mols)}/{len(val_mols)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NMR_VAE(hid=args.hid, latent_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 使用余弦退火调度器替代 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # ---- 初始化 best_val 与保存目录 ----
    os.makedirs("checkpoints_g2s", exist_ok=True)
    best_val = float("inf")

    # 记录训练/验证历史，便于可视化与后处理
    history_epochs = []
    history_train_loss = []
    history_train_recon = []
    history_train_kl = []
    history_val_loss = []
    history_val_recon = []
    history_val_kl = []
    history_val_mu = []
    history_val_pi = []
    history_val_r2_mu = []

    print(f"\n--- 开始在 {device} 上训练 VAE 模型 ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl = train(
            model, train_loader, optimizer, device, epoch, beta=args.beta
        )
        val_loss, val_recon, val_kl, val_mu, val_pi = validate(
            model, val_loader, device, beta=args.beta
        )

        scheduler.step()

        # 记录历史
        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_train_recon.append(train_recon)
        history_train_kl.append(train_kl)
        history_val_loss.append(val_loss)
        history_val_recon.append(val_recon)
        history_val_kl.append(val_kl)
        history_val_mu.append(val_mu)
        history_val_pi.append(val_pi)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} (rec={train_recon:.4f}, kl={train_kl:.4f}) | "
            f"Val Loss: {val_loss:.4f} (rec={val_recon:.4f}, kl={val_kl:.4f}) | "
            f"μ_loss={val_mu:.4f}, π_loss={val_pi:.4f}"
        )

        # 每次验证更优就保存
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints_g2s/g2s_best_model.pt")
            print(f" [✓] New best model saved! Val Loss: {val_loss:.4f}")

    # ---- 训练结束后: 保存历史 CSV 并绘制损失曲线 ----
    if history_epochs:
        history_dict = {
            "epoch": history_epochs,
            "train_loss": history_train_loss,
            "train_recon_loss": history_train_recon,
            "train_kl_loss": history_train_kl,
            "val_loss": history_val_loss,
            "val_recon_loss": history_val_recon,
            "val_kl_loss": history_val_kl,
            "val_mu_L1": history_val_mu,
            "val_pi_L1": history_val_pi,
        }

        df_hist = pd.DataFrame(history_dict)
        csv_path = os.path.join("checkpoints_g2s", "training_history.csv")
        df_hist.to_csv(csv_path, index=False)
        print(f"Training history CSV saved to {csv_path}")

        # 统一字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.linewidth'] = 1.2

        # 损失曲线 (总损失 / 重构 / KL)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        ax = axes[0]
        ax.plot(history_epochs, history_train_loss, 'o-', label="Train", color='#1f77b4', markersize=3)
        ax.plot(history_epochs, history_val_loss, 's-', label="Val", color='#ff7f0e', markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Total Loss", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(direction='in', width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax = axes[1]
        ax.plot(history_epochs, history_train_recon, 'o-', label="Train", color='#1f77b4', markersize=3)
        ax.plot(history_epochs, history_val_recon, 's-', label="Val", color='#ff7f0e', markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Reconstruction Loss", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(direction='in', width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax = axes[2]
        ax.plot(history_epochs, history_train_kl, 'o-', label="Train", color='#1f77b4', markersize=3)
        ax.plot(history_epochs, history_val_kl, 's-', label="Val", color='#ff7f0e', markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("KL Loss", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(direction='in', width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        loss_fig_path = os.path.join("checkpoints_g2s", "loss_curves.png")
        fig.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Loss curves figure saved to {loss_fig_path}")

    # ---- 使用最佳模型在验证集上计算最终回归 (μ 与 π) ----
    best_model_path = os.path.join("checkpoints_g2s", "g2s_best_model.pt")
    if os.path.exists(best_model_path):
        best_model = NMR_VAE(hid=args.hid, latent_dim=args.latent_dim).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))

        mu_true, mu_pred, pi_true, pi_pred = collect_val_regression_data(
            best_model, val_loader, device
        )

        if mu_true is not None:
            # 统一字体
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['axes.linewidth'] = 1.2

            # --- μ 回归 ---
            r2_mu = r2_score(mu_true, mu_pred)
            mae_mu = mean_absolute_error(mu_true, mu_pred)

            coeffs_mu = np.polyfit(mu_true, mu_pred, 1)
            fit_mu = np.poly1d(coeffs_mu)
            x_fit_mu = np.linspace(mu_true.min(), mu_true.max(), 200)
            residuals_mu = mu_pred - fit_mu(mu_true)
            std_err_mu = np.std(residuals_mu)
            ci_upper_mu = fit_mu(x_fit_mu) + 1.96 * std_err_mu
            ci_lower_mu = fit_mu(x_fit_mu) - 1.96 * std_err_mu

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(mu_true, mu_pred, alpha=0.4, s=10, color='#1f77b4', label='Val Predicted')
            ax.plot(x_fit_mu, fit_mu(x_fit_mu), 'r-', linewidth=1.5,
                    label=f'Fit: y={coeffs_mu[0]:.2f}x+{coeffs_mu[1]:.2f}')
            ax.fill_between(x_fit_mu, ci_lower_mu, ci_upper_mu, color='#ff7f0e', alpha=0.2, label='95% CI')
            ax.plot([mu_true.min(), mu_true.max()], [mu_true.min(), mu_true.max()],
                    'k--', linewidth=1.0, label='1:1 Line')

            ax.set_xlabel('Observed Chemical Shift (ppm)', fontsize=12)
            ax.set_ylabel('Predicted Chemical Shift (ppm)', fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc='upper left')
            ax.tick_params(direction='in', width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            text_mu = f'$R^2$ = {r2_mu:.4f}\nMAE = {mae_mu:.4f}'
            ax.text(0.95, 0.05, text_mu, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

            fig.tight_layout()
            r2_mu_path = os.path.join("checkpoints_g2s", "r2_regression_mu.png")
            fig.savefig(r2_mu_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Final μ regression plot saved to {r2_mu_path}")

            # --- π 回归 ---
            r2_pi = r2_score(pi_true, pi_pred)
            mae_pi = mean_absolute_error(pi_true, pi_pred)

            coeffs_pi = np.polyfit(pi_true, pi_pred, 1)
            fit_pi = np.poly1d(coeffs_pi)
            x_fit_pi = np.linspace(pi_true.min(), pi_true.max(), 200)
            residuals_pi = pi_pred - fit_pi(pi_true)
            std_err_pi = np.std(residuals_pi)
            ci_upper_pi = fit_pi(x_fit_pi) + 1.96 * std_err_pi
            ci_lower_pi = fit_pi(x_fit_pi) - 1.96 * std_err_pi

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pi_true, pi_pred, alpha=0.4, s=10, color='#1f77b4', label='Val Predicted')
            ax.plot(x_fit_pi, fit_pi(x_fit_pi), 'r-', linewidth=1.5,
                    label=f'Fit: y={coeffs_pi[0]:.2f}x+{coeffs_pi[1]:.2f}')
            ax.fill_between(x_fit_pi, ci_lower_pi, ci_upper_pi, color='#ff7f0e', alpha=0.2, label='95% CI')
            ax.plot([pi_true.min(), pi_true.max()], [pi_true.min(), pi_true.max()],
                    'k--', linewidth=1.0, label='1:1 Line')

            ax.set_xlabel('Observed Intensity', fontsize=12)
            ax.set_ylabel('Predicted Intensity', fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc='upper left')
            ax.tick_params(direction='in', width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            text_pi = f'$R^2$ = {r2_pi:.4f}\nMAE = {mae_pi:.4f}'
            ax.text(0.95, 0.05, text_pi, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

            fig.tight_layout()
            r2_pi_path = os.path.join("checkpoints_g2s", "r2_regression_pi.png")
            fig.savefig(r2_pi_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Final π regression plot saved to {r2_pi_path}")

            # 保存回归散点数据到 CSV，便于日后自定义绘图
            df_mu = pd.DataFrame({"true_mu": mu_true, "pred_mu": mu_pred})
            df_pi = pd.DataFrame({"true_pi": pi_true, "pred_pi": pi_pred})
            df_mu.to_csv(os.path.join("checkpoints_g2s", "val_mu_regression_points.csv"), index=False)
            df_pi.to_csv(os.path.join("checkpoints_g2s", "val_pi_regression_points.csv"), index=False)
            print("Validation regression points saved to CSV (μ & π).")
