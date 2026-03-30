import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, argparse
from torch.utils.data import Dataset, DataLoader
from model.inverse_common import E_SU, NUM_SU_TYPES, SU_DEFS
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 新增: PAD 类常量 ===
PAD_OFFSET = 1  # 额外留一类给 PAD，占据最后一个索引位置

# === 经验 SU 频率（根据 su_distribution.csv，按 SU_DEFS / SU ID 顺序对齐） ===
_SU_NAME_TO_FREQ = {
    "Carbocyclic_Aro_CH": 23652.,
    "Alkyl_CH2": 15152.,
    "Alkyl_CH3": 10518.,
    "Alcohol_Ether_C": 5118.,
    "Alkyl_Substituted_Aro_C": 3982.,
    "Halogen_X": 3233.,
    "Hydroxyl_O": 2579.,
    "Ether_O": 2486.,
    "O_Substituted_Aro_C": 2337.,
    "Amine_C": 2282.,
    "Alkyl_CH": 2057.,
    "Amine_Nitrogen": 1883.,
    "X_Substituted_Aro_C": 1839.,
    "Aromatic_Bridgehead_C": 1819.,
    "Ester_Group": 1797.,
    "Keto_Substituted_Aro_C": 1698.,
    "Vinyllic_CH": 1695.,
    "Aldehyde_Ketone_C": 1504.,
    "Heterocyclic_N": 1198.,
    "Carboxylic_Acid": 1134.,
    "N_Substituted_Aro_C": 1100.,
    "Amide_Group": 953.,
    "Halogenated_C": 802.,
    "Alkyl_Cq": 759.,
    "Vinyllic_Cq": 591.,
    "Aryl_Substituted_Aro_C": 560.,
    "Thioether_S": 527.,
    "Vinyllic_CH2": 414.,
    "Nitrile_C": 334.,
    "S_Substituted_Aro_C": 260.,
    "Alkynyl_Cq": 259.,
    "Heterocyclic_S": 167.,
    "Alkynyl_CH": 65.,
}

# 按照 coarse_graph.SU_DEFS 的顺序（即 SU ID 顺序 0-32）构建频率向量
SU_FREQ = torch.tensor([
    _SU_NAME_TO_FREQ[name] for name, _ in SU_DEFS
], dtype=torch.float)

class ResidualSEBlock(nn.Module):
    """1-D 残差卷积块 + SE 通道注意力 + GroupNorm (对小 batch 更稳)。"""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, groups: int = 8, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=9, stride=stride, padding=4, bias=False)
        self.gn1   = nn.GroupNorm(groups, out_c)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=9, stride=1, padding=4, bias=False)
        self.gn2   = nn.GroupNorm(groups, out_c)
        # SE 分支
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1  = nn.Linear(out_c, out_c // se_ratio)
        self.se_fc2  = nn.Linear(out_c // se_ratio, out_c)
        # shortcut
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        # SE
        w = self.se_pool(h).squeeze(-1)  # (B,C)
        w = torch.sigmoid(self.se_fc2(self.act(self.se_fc1(w))))
        h = h * w.unsqueeze(-1)
        return self.act(h + res)

class SpectrumEncoderCNN(nn.Module):
    """改进版 1-D ResNet + SE 编码器，将全谱编码为 hid 向量。"""
    def __init__(self, in_channels: int = 1, hid: int = 256, base_c: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_c, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, base_c),
            nn.ReLU(inplace=True)
        )
        # 连续 4 个残差块，逐步下采样
        self.layer1 = ResidualSEBlock(base_c, base_c * 2, stride=2)   # 64 → 128
        self.layer2 = ResidualSEBlock(base_c * 2, hid, stride=2)      # 128 → hid
        self.layer3 = ResidualSEBlock(hid, hid, stride=2)
        self.layer4 = ResidualSEBlock(hid, hid, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).squeeze(-1)
        return h

class S2NModel(nn.Module):

    def __init__(self, hid: int = 256, n_su_types: int = NUM_SU_TYPES):
        super().__init__()
        self.n_su_types = n_su_types

        # 频谱编码器
        self.spec_enc = SpectrumEncoderCNN(in_channels=1, hid=hid)

        # 元素向量 -> hid 嵌入
        self.elem_fc = nn.Linear(6, hid)

        # 融合后投影
        self.fusion = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(inplace=True),
        )

        # 输出头: SU 计数 (33)，在原 head 前增加一层 + Dropout，提升表达并减轻过拟合
        self.head_count = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hid, n_su_types),
        )

    # -----------------------------------------------------
    def forward(self, spectrum: torch.Tensor, elements: torch.Tensor):
        # 1) 频谱编码
        h_spec = self.spec_enc(spectrum.unsqueeze(1))        # (B, hid)
        # 2) 元素编码
        h_elem = F.relu(self.elem_fc(elements))              # (B, hid)
        # 3) 融合
        h = self.fusion(torch.cat([h_spec, h_elem], dim=-1)) # (B, hid)

        # 4) 输出 SU 计数 logits
        n_su_logits = self.head_count(h)                     # (B, 33)
        return n_su_logits

    def infer_su_hist(self, spectrum: torch.Tensor, elements: torch.Tensor):
        """简化推理接口，直接返回预测的SU直方图。"""
        with torch.no_grad():
            n_su_logits = self.forward(spectrum, elements)
        n_su_pred = F.softplus(n_su_logits) # (B,33)
        return n_su_pred

# =====================================================================
#  1. 数据集类
# =====================================================================
class S2NDataset(Dataset):
    """[V2 Refactored] 为 S2N 模型加载和预处理数据。"""
    def __init__(self, pt_files: list):
        self.files = pt_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_dict = torch.load(file_path, map_location='cpu', weights_only=False)

        # 输入
        spectrum = data_dict.get('y_spectrum', data_dict.get('y_mol'))
        elements = data_dict['total_atom_counts']

        # 监督标签 (Ground Truth)
        n_su_true = data_dict['su_hist']
        
        return {
            "spectrum": spectrum,
            "elements": elements,
            "n_su_true": n_su_true,
        }

def s2n_loss(
    n_su_logits,
    n_su_true,
    elements_true,
    w_cnt: float = 1.0,
    w_elem_abs: float = 0.5,
    w_nodesum: float = 0.5,
):
    """
    [V2 Refactored] 简化的损失函数，仅关注SU直方图的准确性。
    """
    # 1. SU计数损失 (L1 Loss，加入稀有 SU 权重)
    n_su_pred = F.softplus(n_su_logits)
    diff = torch.abs(n_su_pred - n_su_true)  # (B, 33)

    # 稀有 SU 权重：
    #  - 常见 SU 权重固定为 1
    #  - 仅对稀有 SU （出现次数低于阈值）放大权重，最大不超过 8
    freq = SU_FREQ.to(n_su_pred.device)                   # (33,)
    weights = torch.ones_like(freq)                       # 默认权重=1（常见 SU）

    rare_threshold = 3000.0                               # 出现次数 < 1000 视为稀有
    min_freq = freq.min()
    mask_rare = freq < rare_threshold

    if mask_rare.any():
        # 在线性区间 [min_freq, rare_threshold] 上，将权重从 8 线性缩放到 1
        freq_rare = freq[mask_rare]
        # 为安全起见，避免除以 0
        denom = max(rare_threshold - min_freq, 1.0)
        w_rare = 1.0 + (rare_threshold - freq_rare) / denom * (8.0 - 1.0)
        w_rare = torch.clamp(w_rare, min=1.0, max=8.0)
        weights[mask_rare] = w_rare

    loss_cnt = (diff * weights.unsqueeze(0)).mean()

    # 2. 元素组成约束 (绝对计数L1损失)
    E_SU_d = E_SU.to(n_su_pred.device)  # (33,6)
    elem_pred_cnt = torch.matmul(n_su_pred, E_SU_d) # (B,6)
    loss_elem_abs = F.l1_loss(elem_pred_cnt, elements_true)

    # 3. 总节点数约束
    total_nodes_pred = n_su_pred.sum(dim=-1) # (B,)
    heavy_atoms_true  = elements_true[:, [0,2,3,4,5]].sum(dim=-1) 
    loss_nodesum = F.l1_loss(total_nodes_pred, heavy_atoms_true)

    # 4. 总损失
    total_loss = (
        w_cnt * loss_cnt
        + w_elem_abs * loss_elem_abs
        + w_nodesum * loss_nodesum
    )
    return total_loss, loss_cnt, loss_elem_abs, loss_nodesum


def train_one_epoch(model, loader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_loss_cnt, total_loss_elem, total_loss_nodesum = 0.0, 0.0, 0.0
    count = 0

    for batch in tqdm(loader, desc="Training"):
        spectrum = batch["spectrum"].to(device)
        elements_true = batch["elements"].to(device)
        n_su_true = batch["n_su_true"].to(device)

        optimizer.zero_grad()
        n_su_logits = model(spectrum, elements_true)
        loss, loss_cnt, loss_elem_abs, loss_nodesum = s2n_loss(
            n_su_logits, n_su_true, elements_true
        )

        if torch.isnan(loss):
            continue

        loss.backward()
        optimizer.step()

        batch_size = spectrum.size(0)
        total_loss += loss.item() * batch_size
        total_loss_cnt += loss_cnt.item() * batch_size
        total_loss_elem += loss_elem_abs.item() * batch_size
        total_loss_nodesum += loss_nodesum.item() * batch_size
        count += batch_size
    
    avg_loss = total_loss / count
    avg_loss_cnt = total_loss_cnt / count
    avg_loss_elem = total_loss_elem / count
    avg_loss_nodesum = total_loss_nodesum / count

    return avg_loss, avg_loss_cnt, avg_loss_elem, avg_loss_nodesum


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_loss_cnt, total_loss_elem, total_loss_nodesum = 0.0, 0.0, 0.0
    count = 0

    for batch in tqdm(loader, desc="Validating"):
        spectrum = batch["spectrum"].to(device)
        n_su_true = batch["n_su_true"].to(device)
        elements_true = batch["elements"].to(device)
        
        n_su_logits = model(spectrum, elements_true)
        loss, loss_cnt, loss_elem_abs, loss_nodesum = s2n_loss(
            n_su_logits, n_su_true, elements_true
        )

        if torch.isnan(loss):
            continue

        batch_size = spectrum.size(0)
        total_loss += loss.item() * batch_size
        total_loss_cnt += loss_cnt.item() * batch_size
        total_loss_elem += loss_elem_abs.item() * batch_size
        total_loss_nodesum += loss_nodesum.item() * batch_size
        count += batch_size

    avg_loss = total_loss / count
    avg_loss_cnt = total_loss_cnt / count
    avg_loss_elem = total_loss_elem / count
    avg_loss_nodesum = total_loss_nodesum / count
    
    return avg_loss, avg_loss_cnt, avg_loss_elem, avg_loss_nodesum

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train S2N (Spectrum to Node counts) model")
    parser.add_argument(
        "data_paths",
        type=str,
        nargs='+',
        help="One or more paths to directories containing .pt files. "
             "Strategy: The first path is for single molecules, subsequent paths are for merged/large molecules."
    )
    parser.add_argument("--hid", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for DataLoader")
    args = parser.parse_args()

    # --- 1. Collect and split dataset files for robust validation ---
    print("--- Preparing Datasets ---")
    if not args.data_paths:
        raise ValueError("At least one data path must be provided.")

    single_mol_path = args.data_paths[0]
    merged_mol_paths = args.data_paths[1:]

    single_mol_files = sorted(glob.glob(os.path.join(single_mol_path, "*.pt")))
    merged_mol_files = []
    for path in merged_mol_paths:
        merged_mol_files.extend(sorted(glob.glob(os.path.join(path, "*.pt"))))

    print(f"Found {len(single_mol_files)} single molecule files from '{single_mol_path}'")
    if merged_mol_paths:
        print(f"Found {len(merged_mol_files)} merged molecule files from {merged_mol_paths}")

    random.seed(42)
    random.shuffle(merged_mol_files)
    
    # Split merged files for training and validation
    val_split = 0.2
    split_idx = int(len(merged_mol_files) * (1 - val_split))
    
    train_files = single_mol_files + merged_mol_files[:split_idx]
    val_files = merged_mol_files[split_idx:]
    
    random.shuffle(train_files)

    print(f"Training set size: {len(train_files)} ({len(single_mol_files)} single, {split_idx} merged)")
    print(f"Validation set size: {len(val_files)} ({len(val_files)} merged)")
    
    # --- 2. Create Datasets and DataLoaders ---
    train_dataset = S2NDataset(train_files)
    val_dataset = S2NDataset(val_files)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True  # 避免最后一个 batch=1 导致 BatchNorm 报错
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    # --- 3. Initialize Model, Optimizer, Scheduler ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S2NModel(hid=args.hid).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    # --- 4. Training Loop ---
    os.makedirs("checkpoints_s2n", exist_ok=True)
    best_val_loss = float("inf")

    history_epochs = []
    history_train_loss = []
    history_val_loss = []
    history_train_cnt = []
    history_val_cnt = []
    history_train_elem = []
    history_val_elem = []
    history_train_nodesum = []
    history_val_nodesum = []

    print(f"\n--- Starting training on {device} ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_loss_cnt, train_loss_elem, train_loss_nodesum = train_one_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_loss_cnt, val_loss_elem, val_loss_nodesum = validate_one_epoch(
            model, val_loader, device
        )
        
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Cnt L1: {val_loss_cnt:.4f} "
            f"| Val Elem L1: {val_loss_elem:.4f}"
            f"| Val Nodesum L1: {val_loss_nodesum:.4f}"
        )

        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_cnt.append(train_loss_cnt)
        history_val_cnt.append(val_loss_cnt)
        history_train_elem.append(train_loss_elem)
        history_val_elem.append(val_loss_elem)
        history_train_nodesum.append(train_loss_nodesum)
        history_val_nodesum.append(val_loss_nodesum)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "checkpoints_s2n/best_s2n_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  [✓] New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    # --- 5. Visualize training history (separate plots, Times New Roman, no grid) ---
    if history_epochs:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.linewidth'] = 1.2
        
        def save_single_curve(epochs, train_data, val_data, ylabel, filename):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(epochs, train_data, 'o-', label="Train", color='#1f77b4', markersize=3, linewidth=1.2)
            ax.plot(epochs, val_data, 's-', label="Val", color='#ff7f0e', markersize=3, linewidth=1.2)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(frameon=False, fontsize=10)
            ax.tick_params(direction='in', width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            fig.tight_layout()
            fig.savefig(os.path.join("checkpoints_s2n", filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        save_single_curve(history_epochs, history_train_loss, history_val_loss, "Total Loss", "loss_total.png")
        save_single_curve(history_epochs, history_train_cnt, history_val_cnt, "SU Count L1", "loss_su_count.png")
        save_single_curve(history_epochs, history_train_elem, history_val_elem, "Element Count L1", "loss_element.png")
        save_single_curve(history_epochs, history_train_nodesum, history_val_nodesum, "Node Sum L1", "loss_nodesum.png")
        print("Training curves saved to checkpoints_s2n/ (loss_total.png, loss_su_count.png, loss_element.png, loss_nodesum.png)")

    # --- 6. Generate R2 regression plot on validation set ---
    print("\n--- Generating R2 regression plot on validation set ---")
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            spectrum = batch["spectrum"].to(device)
            elements_true = batch["elements"].to(device)
            n_su_true = batch["n_su_true"]
            n_su_logits = model(spectrum, elements_true)
            n_su_pred = F.softplus(n_su_logits).cpu()
            # Flatten all SU counts for regression plot
            all_true.append(n_su_true.view(-1))
            all_pred.append(n_su_pred.view(-1))
    
    all_true = torch.cat(all_true).numpy()
    all_pred = torch.cat(all_pred).numpy()
    
    # Compute R2 and MAE
    from sklearn.metrics import r2_score, mean_absolute_error
    r2_val = r2_score(all_true, all_pred)
    mae_val = mean_absolute_error(all_true, all_pred)
    
    # Linear fit
    import numpy as np
    coeffs = np.polyfit(all_true, all_pred, 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(all_true.min(), all_true.max(), 100)
    
    # 95% confidence interval (approximate)
    residuals = all_pred - fit_line(all_true)
    std_err = np.std(residuals)
    ci_upper = fit_line(x_fit) + 1.96 * std_err
    ci_lower = fit_line(x_fit) - 1.96 * std_err
    
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(all_true, all_pred, alpha=0.4, s=10, color='#1f77b4', label='Val Predicted')
    ax.plot(x_fit, fit_line(x_fit), 'r-', linewidth=1.5, label=f'Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
    ax.fill_between(x_fit, ci_lower, ci_upper, color='#ff7f0e', alpha=0.2, label='95% CI')
    ax.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'k--', linewidth=1, label='1:1 Line')
    
    ax.set_xlabel("Observed Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.tick_params(direction='in', width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Add R2 and MAE text
    textstr = f'$R^2$ = {r2_val:.4f}\nMAE = {mae_val:.4f}'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    fig.tight_layout()
    r2_path = os.path.join("checkpoints_s2n", "r2_regression_val.png")
    fig.savefig(r2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"R2 regression plot saved to {r2_path} (R2={r2_val:.4f}, MAE={mae_val:.4f})")