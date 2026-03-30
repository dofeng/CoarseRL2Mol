import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from torch_geometric.loader import DataLoader

try:
    from .g2s_model import NMR_VAE, LocalNMRDataset, load_raw_data
    from .inverse_common import lorentzian_spectrum
    from .coarse_graph import PPM_AXIS
except ImportError:
    from g2s_model import NMR_VAE, LocalNMRDataset, load_raw_data
    from inverse_common import lorentzian_spectrum
    from coarse_graph import PPM_AXIS

def validate_g2s_model(model_path: Path, input_path: Path, num_samples: int, hid: int = 384, latent_dim: int = 16, k_hop: int = 3):
    """
    加载G2S VAE模型，对测试样本进行前向预测，并生成详细的可视化验证结果。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载训练好的VAE模型
    print("正在加载G2S VAE模型...")
    model = NMR_VAE(hid=hid, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print("模型加载成功。")

    # 2. 加载测试数据
    try:
        raw_list = load_raw_data(str(input_path))
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    if len(raw_list) > num_samples:
        print(f"共找到 {len(raw_list)} 个分子，将随机抽取 {num_samples} 个进行分析。")
        raw_list = random.sample(raw_list, num_samples)
    
    output_dir = Path("g2s_validation_results")
    output_dir.mkdir(exist_ok=True)
    print(f"验证结果将保存在 '{output_dir}' 目录中。")

    # 统一绘图风格
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.2

    # 收集每个分子的定量指标，最终汇总为 CSV
    metrics_rows = []

    # 3. 遍历每个抽样的分子进行验证
    for i, mol_data in enumerate(tqdm(raw_list, desc="Validating Molecules")):
        basename = mol_data.get('smiles', f'molecule_{i}')
        
        # 将单个分子包装成list以供LocalNMRDataset使用
        dataset = LocalNMRDataset([mol_data], k=k_hop)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            (mu_pred, pi_pred), _ = model(batch)

        # 聚合所有节点的预测，重构最终的分子谱图（直接使用模型 μ/π，不做区间映射/夹逼）
        S_reconstructed = lorentzian_spectrum(mu_pred, pi_pred, PPM_AXIS.to(device))
        
        # 获取真实谱图和节点级真值
        S_true = batch.y_mol.view(-1, len(PPM_AXIS))[0]  # 取第一个即可
        ppm_true = batch.real_ppm.squeeze()
        pi_true = batch.real_intensity.squeeze()
        is_carbon_mask = batch.is_carbon.squeeze() & (pi_true > 0)

        # ---- 计算定量指标 ----
        S_true_np = S_true.detach().cpu().numpy()
        S_recon_np = S_reconstructed.detach().cpu().numpy()
        try:
            r2_spectrum = float(r2_score(S_true_np, S_recon_np))
            mae_spectrum = float(mean_absolute_error(S_true_np, S_recon_np))
        except Exception:
            r2_spectrum, mae_spectrum = float("nan"), float("nan")

        if is_carbon_mask.any():
            ppm_true_valid = ppm_true[is_carbon_mask].detach().cpu().numpy()
            mu_pred_valid = mu_pred[is_carbon_mask].detach().cpu().numpy()
            pi_true_valid = pi_true[is_carbon_mask].detach().cpu().numpy()
            pi_pred_valid = pi_pred[is_carbon_mask].detach().cpu().numpy()
            try:
                r2_mu = float(r2_score(ppm_true_valid, mu_pred_valid))
                mae_mu = float(mean_absolute_error(ppm_true_valid, mu_pred_valid))
            except Exception:
                r2_mu, mae_mu = float("nan"), float("nan")
            try:
                r2_pi = float(r2_score(pi_true_valid, pi_pred_valid))
                mae_pi = float(mean_absolute_error(pi_true_valid, pi_pred_valid))
            except Exception:
                r2_pi, mae_pi = float("nan"), float("nan")
            num_peaks = int(is_carbon_mask.sum().item())
        else:
            r2_mu = mae_mu = r2_pi = mae_pi = float("nan")
            num_peaks = 0

        metrics_rows.append({
            "molecule": basename,
            "num_peaks": num_peaks,
            "r2_spectrum": r2_spectrum,
            "mae_spectrum": mae_spectrum,
            "r2_mu": r2_mu,
            "mae_mu": mae_mu,
            "r2_pi": r2_pi,
            "mae_pi": mae_pi,
        })

        # --- 分别创建三张独立的图片进行可视化 ---
        sanitized_basename = "".join([c if c.isalnum() else "_" for c in basename])

        # 统一美化函数
        def _beautify_axis(ax):
            ax.tick_params(direction='in', width=1.5, labelsize=14)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        # ========== 图1: 全谱对比 ==========
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(PPM_AXIS.numpy(), S_true_np, label='True Spectrum', color='#1f77b4', lw=2.0)
        ax1.plot(PPM_AXIS.numpy(), S_recon_np, label='Reconstructed', color='#ff7f0e', linestyle='--', lw=2.0)
        ax1.set_xlabel('Chemical Shift (ppm)', fontsize=18)
        ax1.set_ylabel('Absolute Intensity', fontsize=18)
        ax1.invert_xaxis()
        ax1.legend(frameon=False, fontsize=16)
        _beautify_axis(ax1)
        text_spec = f"R$^2$={r2_spectrum:.4f}\nMAE={mae_spectrum:.4f}"
        ax1.text(0.02, 0.98, text_spec, transform=ax1.transAxes,
                 va='top', ha='left', fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        fig1.tight_layout()
        fig1.savefig(output_dir / f"{sanitized_basename}_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # ========== 图2: 化学位移 (μ) 对比 ==========
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        if num_peaks > 0:
            ax2.scatter(ppm_true_valid, mu_pred_valid, alpha=0.6, s=50, color='#1f77b4', edgecolors='white', linewidths=0.5)
        ax2.plot([0, 250], [0, 250], color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('True Chemical Shift (ppm)', fontsize=18)
        ax2.set_ylabel('Predicted Chemical Shift (ppm)', fontsize=18)
        ax2.set_xlim(0, 250)
        ax2.set_ylim(0, 250)
        ax2.set_aspect('equal', adjustable='box')
        _beautify_axis(ax2)
        text_mu = f"R$^2$={r2_mu:.4f}\nMAE={mae_mu:.4f}"
        ax2.text(0.02, 0.98, text_mu, transform=ax2.transAxes,
                 va='top', ha='left', fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        fig2.tight_layout()
        fig2.savefig(output_dir / f"{sanitized_basename}_mu.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # ========== 图3: 峰强度 (π) 对比 ==========
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        if num_peaks > 0:
            ax3.scatter(pi_true_valid, pi_pred_valid, alpha=0.6, s=50, color='#1f77b4', edgecolors='white', linewidths=0.5)
        max_val = max(float(pi_true.max().item()), float(pi_pred.max().item())) * 1.1 if pi_true.numel() > 0 and pi_pred.numel() > 0 else 1.0
        ax3.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('True Intensity', fontsize=18)
        ax3.set_ylabel('Predicted Intensity', fontsize=18)
        ax3.set_xlim(0, max_val)
        ax3.set_ylim(0, max_val)
        ax3.set_aspect('equal', adjustable='box')
        _beautify_axis(ax3)
        text_pi = f"R$^2$={r2_pi:.4f}\nMAE={mae_pi:.4f}"
        ax3.text(0.02, 0.98, text_pi, transform=ax3.transAxes,
                 va='top', ha='left', fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
        fig3.tight_layout()
        fig3.savefig(output_dir / f"{sanitized_basename}_pi.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)

    # 汇总保存所有分子的定量指标
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        metrics_path = output_dir / "validation_metrics.csv"
        df_metrics.to_csv(metrics_path, index=False)
        print(f"定量验证指标已保存至: {metrics_path}")

    print(f"\n--- 验证完成 ---")
    print(f"已在 {len(raw_list)} 个样本上生成验证图。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2S VAE Model Validation Script")
    parser.add_argument('--model_path', type=Path, required=True, help='训练好的G2S VAE模型权重路径')
    parser.add_argument('--input_path', type=Path, required=True, help='包含测试.pt文件的目录或单个.pt文件路径')
    parser.add_argument('--num_samples', type=int, default=10, help='如果输入是目录，则随机抽样的最大样本数')
    parser.add_argument('--hid', type=int, default=384, help='隐藏层维度，需与训练时一致')
    parser.add_argument('--latent_dim', type=int, default=16, help='潜变量维度，需与训练时一致')
    parser.add_argument('--k_hop', type=int, default=3, help='子图的 hop 数，与模型训练保持一致')
    
    args = parser.parse_args()
    
    validate_g2s_model(model_path=args.model_path,
                       input_path=args.input_path,
                       num_samples=args.num_samples,
                       hid=args.hid,
                       latent_dim=args.latent_dim,
                       k_hop=args.k_hop) 
