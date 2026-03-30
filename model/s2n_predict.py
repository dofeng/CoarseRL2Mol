import argparse
import torch
import pandas as pd
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re

from s2n_model import S2NModel
from coarse_graph import SU_DEFS, _node_color
import torch.nn.functional as F

def predict_from_csv_and_elements(model_path: Path, spectrum_csv: Path, elements_str: str, output_name: str):
    """
    从谱图CSV和元素组成直接进行S2N预测。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = S2NModel(hid=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"S2N模型 '{model_path}' 加载成功。")

    # 2. 加载和处理输入数据
    # 从CSV加载谱图
    df_spec = pd.read_csv(spectrum_csv, sep=r'[;, \t]', engine='python', header=None)
    S_target = torch.tensor(df_spec.iloc[:, 1].values, dtype=torch.float).to(device)
    
    # 解析元素字符串
    matches = dict(re.findall(r"([CHONSX])\s*=\s*(\d+)", elements_str.upper()))
    E_target = torch.tensor([int(matches.get(sym, 0)) for sym in ['C','H','O','N','S','X']], dtype=torch.float).to(device)
    
    # 预处理：强度缩放 (与 inverse_pipeline.py 保持一致)
    num_carbons = E_target[0].item()
    if num_carbons > 0:
        target_total_area = torch.pi * num_carbons
        current_spec_sum = S_target.sum() * 0.1
        if current_spec_sum > 1e-6:
            scaling_factor = target_total_area / current_spec_sum
            S_target = S_target * scaling_factor
    
    print("\n--- 输入数据 ---")
    print(f"谱图文件: {spectrum_csv}")
    print(f"元素组成: {elements_str} (C={num_carbons})")
    print(f"谱图总面积缩放至: {S_target.sum().item():.2f}")

    # 3. 模型推理
    with torch.no_grad():
        su_hist_pred = model.infer_su_hist(S_target.unsqueeze(0), E_target.unsqueeze(0)).squeeze(0).cpu()

    # 4. 打印结果
    print("\n--- 预测的SU直方图 ---")
    su_names = [name for name, _ in SU_DEFS]
    df = pd.DataFrame({
        'SU_Name': su_names,
        'Predicted_Count': su_hist_pred.round().long().tolist()
    })
    df_filtered = df[df['Predicted_Count'] > 0]
    print(df_filtered.to_string(index=False))
    
    # 保存结果
    output_dir = Path("s2n_prediction_results")
    output_dir.mkdir(exist_ok=True)
    csv_out_path = output_dir / f"{output_name}_pred_su_hist.csv"
    df_filtered.to_csv(csv_out_path, index=False)
    print(f"\n[✓] 预测结果已保存至: {csv_out_path}")

    # 可视化预测的 SU 直方图（与 dataset_utils.py 风格一致）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    
    # 使用完整的预测结果（包含所有33个SU），按ID顺序
    all_counts = su_hist_pred.round().numpy()
    x_indices = np.arange(len(SU_DEFS))  # 0-32
    
    # 使用 _node_color 为每个 SU 设置颜色
    colors = [_node_color(int(idx)) for idx in x_indices]
    labels = [f"{idx}" for idx in x_indices]

    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(range(len(x_indices)), all_counts, color=colors, edgecolor='white', linewidth=0.5)
    
    # 在柱状图顶部添加数值标签
    for i, (bar, count) in enumerate(zip(bars, all_counts)):
        height = bar.get_height()
        if count > 0:  # 只为非零值添加标签
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=16)
    
    ax.set_xticks(range(len(x_indices)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel('Predicted Count', fontsize=24)
    ax.set_title('Predicted Structural Unit Distribution (Indices 0-32)', fontsize=24, pad=12)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(False)

    fig.tight_layout()
    png_out_path = output_dir / f"{output_name}_pred_su_hist.png"
    fig.savefig(png_out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] 预测直方图已保存至: {png_out_path}")

def validate_s2n_model(model_path: Path, input_path: Path, num_samples: int):
    """
    加载S2N模型并在一系列测试样本上验证其性能。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = S2NModel(hid=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"S2N模型 '{model_path}' 加载成功。")

    # 2. 加载测试数据
    if os.path.isdir(input_path):
        pt_files = sorted(glob.glob(os.path.join(input_path, "*.pt")))
    elif os.path.isfile(input_path):
        pt_files = [str(input_path)]
    else:
        raise FileNotFoundError(f"输入路径 '{input_path}' 无效。")

    if not pt_files:
        raise FileNotFoundError(f"在 '{input_path}' 中未找到.pt文件。")

    if len(pt_files) > num_samples:
        import random
        pt_files = random.sample(pt_files, num_samples)
        print(f"随机抽取 {num_samples} 个样本进行验证。")

    output_dir = Path("s2n_validation_results")
    output_dir.mkdir(exist_ok=True)
    print(f"验证结果将保存在 '{output_dir}' 目录中。")
    
    # 3. 循环验证
    total_mae = 0.0
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        
        S_target = data['y_spectrum'].to(device)
        E_target = data['total_atom_counts'].to(device)
        su_hist_true = data['su_hist']

        # 模型推理
        with torch.no_grad():
            su_hist_pred = model.infer_su_hist(S_target.unsqueeze(0), E_target.unsqueeze(0)).squeeze(0).cpu()

        mae = F.l1_loss(su_hist_pred, su_hist_true).item()
        total_mae += mae
        
        basename = Path(pt_file).stem
        print(f"\n--- 验证样本: {basename} ---")
        print(f"SU 直方图 MAE: {mae:.4f}")

        # 打印对比表格
        su_names = [name for name, _ in SU_DEFS]
        df = pd.DataFrame({
            'SU_Name': su_names,
            'True_Count': su_hist_true.round().long().tolist(),
            'Pred_Count': su_hist_pred.round().long().tolist()
        })
        df_filtered = df[(df['True_Count'] > 0) | (df['Pred_Count'] > 0)]
        print(df_filtered.to_string(index=False))

        # 可视化对比：仅绘制 True 或 Pred 非零的 SU 类型，使用 SU ID
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.linewidth'] = 1.2
        
        su_hist_true_np = su_hist_true.numpy()
        su_hist_pred_np = su_hist_pred.numpy()
        mask = (su_hist_true_np > 0) | (su_hist_pred_np > 0)

        if mask.any():
            # 获取 SU ID 列表
            su_ids_sel = [i for i, keep in enumerate(mask) if keep]
            true_sel = su_hist_true_np[mask]
            pred_sel = su_hist_pred_np[mask]

            indices = np.arange(len(su_ids_sel))
            width = 0.38

            fig, ax = plt.subplots(figsize=(max(8, len(su_ids_sel) * 0.6), 5))
            ax.bar(indices - width/2, true_sel, width, label='True', 
                   color='#4C72B0', edgecolor='#2E4A71', linewidth=0.8)
            ax.bar(indices + width/2, pred_sel, width, label='Predicted', 
                   color='#DD8452', edgecolor='#A85C32', linewidth=0.8)

            ax.set_xlabel('SU ID', fontsize=18)
            ax.set_ylabel('Counts', fontsize=18)
            ax.set_xticks(indices)
            ax.set_xticklabels([str(sid) for sid in su_ids_sel], fontsize=16)
            ax.set_yticklabels([str(sid) for sid in su_ids_sel], fontsize=16)
            ax.legend(frameon=False, fontsize=16, loc='upper right')
            ax.tick_params(direction='in', width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            fig.tight_layout()
            plt.savefig(output_dir / f"{basename}_validation.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f"\n--- 验证完成 ---")
    print(f"在 {len(pt_files)} 个样本上的平均 MAE: {total_mae / len(pt_files):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("S2N模型预测与验证脚本")
    parser.add_argument('--model_path', type=Path, required=True, help='训练好的 S2N 模型权重路径. Aliased to --checkpoint for convenience.')
    
    # Group for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--input_path', type=Path, help='[验证模式] 包含测试.pt文件的目录或单个.pt文件路径')
    mode_group.add_argument('--spectrum_csv', type=Path, help='[预测模式] 目标谱图CSV文件路径')
    
    # Arguments for prediction mode
    parser.add_argument('--elements', type=str, help="[预测模式] 目标元素组成, e.g., 'C=230,H=200,O=30'")
    parser.add_argument('--output_name', type=str, default='prediction', help='[预测模式] 输出文件的前缀名')

    # Arguments for validation mode
    parser.add_argument('--num_samples', type=int, default=10, help='[验证模式] 如果输入是目录，则随机抽样的最大样本数')

    # Alias for convenience to match notebook
    parser.add_argument('--checkpoint', type=Path, dest='model_path', help=argparse.SUPPRESS)

    args = parser.parse_args()
    
    if args.input_path:
        # --- 验证模式 ---
        validate_s2n_model(args.model_path, args.input_path, args.num_samples)
    elif args.spectrum_csv:
        # --- 预测模式 ---
        if not args.elements:
            parser.error("--elements 参数在预测模式下是必需的。")
        predict_from_csv_and_elements(args.model_path, args.spectrum_csv, args.elements, args.output_name) 