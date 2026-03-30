import argparse
import os
import glob
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from torch_geometric.data import Data
import random
import numpy as np
import shutil

# Make sure coarse_graph is importable
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from coarse_graph import SU_DEFS, SU_PPM_RANGES, _node_color


def _set_plot_defaults():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0


def _save_su_distribution(total_su_hist: torch.Tensor, output_dir: Path) -> Path:
    rows = []
    for idx, (name, _) in enumerate(SU_DEFS):
        rows.append({
            'SU_Index': idx,
            'SU_Name': name,
            'Total_Count': int(total_su_hist[idx].item())
        })
    df = pd.DataFrame(rows, columns=['SU_Index', 'SU_Name', 'Total_Count'])
    csv_path = output_dir / "su_distribution.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _save_node_count_distribution(node_counts: list[int], output_dir: Path) -> Path:
    series = pd.Series(node_counts)
    distribution = series.value_counts().sort_index()
    df = distribution.reset_index()
    df.columns = ['Num_SUs', 'Molecule_Count']
    csv_path = output_dir / "node_count_distribution.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _plot_su_distribution(su_df: pd.DataFrame, output_path: Path):
    _set_plot_defaults()
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = [_node_color(int(idx)) for idx in su_df['SU_Index']]
    labels = [f"{idx}" for idx in su_df['SU_Index']]
    bars = ax.bar(range(len(su_df)), su_df['Total_Count'], color=colors, edgecolor='white', linewidth=0.5)
    
    # 在柱状图顶部添加数值标签
    for i, (bar, count) in enumerate(zip(bars, su_df['Total_Count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=16)
    
    ax.set_xticks(range(len(su_df)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel('Count', fontsize=24)
    ax.set_title('Structural Unit Distribution (Indices 0-32)', fontsize=24, pad=12)
    ax.tick_params(axis='both', labelsize=18)  # 调整坐标轴刻度数值大小
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_node_count_distribution(node_df: pd.DataFrame, output_path: Path):
    _set_plot_defaults()
    fig, ax = plt.subplots(figsize=(18, 6))
    counts = node_df['Molecule_Count'].to_numpy()
    num_sus = node_df['Num_SUs'].to_numpy()
    
    # 统一颜色，不使用渐变；对稀有值（count < 10）使用不同颜色突出显示
    colors = ['#d62728' if c < 10 else '#1f77b4' for c in counts]
    
    ax.bar(num_sus, counts, color=colors, edgecolor='white', linewidth=0.4, width=0.8)
    
    # 使用对数刻度y轴以便更好地显示稀有值
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)  # 设置下限，避免log(0)
    
    ax.set_xlabel('Number of Structural Units per Molecule', fontsize=24)
    ax.set_ylabel('Molecule Count (log scale)', fontsize=24)
    ax.set_title('Molecule Size Distribution', fontsize=24, pad=12)
    ax.tick_params(axis='both', labelsize=18)  # 调整坐标轴刻度数值大小
    ax.grid(False)
    
    # 添加图例说明颜色含义
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='white', label='Count ≥ 10'),
        Patch(facecolor='#d62728', edgecolor='white', label='Count < 10 (rare)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_analysis_results(su_csv: Path, node_count_csv: Path, output_dir: Path):
    """
    Generate bar charts for SU distribution and molecule-size distribution
    from precomputed CSV files.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    su_df = pd.read_csv(su_csv)
    node_df = pd.read_csv(node_count_csv)
    _plot_su_distribution(su_df, output_dir / "su_distribution.png")
    _plot_node_count_distribution(node_df, output_dir / "node_count_distribution.png")


def analyze_dataset(data_dir: Path, output_dir: Path):
    """
    Scans a directory of .pt files, analyzes node and SU distributions,
    and saves the analysis results.
    """
    output_dir.mkdir(exist_ok=True)
    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'.")
        return

    print(f"Analyzing {len(pt_files)} graph files from '{data_dir}'...")

    node_counts = []
    total_su_hist = torch.zeros(len(SU_DEFS), dtype=torch.long)

    for f in tqdm(pt_files, desc="Scanning files"):
        try:
            # 修正：加载字典后，将其转换为PyG的Data对象
            data_dict = torch.load(f, weights_only=False)
            data = Data.from_dict(data_dict)
            node_counts.append(data.num_nodes)
            total_su_hist += data.su_hist.long()
        except Exception as e:
            print(f"Warning: Could not load or process file {f}. Error: {e}")
            continue

    if not node_counts:
        print("Error: No valid data could be loaded for analysis.")
        return

    # --- Node Count Analysis ---
    df_nodes = pd.DataFrame(node_counts, columns=['num_nodes'])
    print("\n" + "="*40)
    print("      Node Count Distribution Analysis")
    print("="*40)
    print(df_nodes.describe())

    node_count_csv = _save_node_count_distribution(node_counts, output_dir)
    print(f"[✓] Node count distribution data saved to '{node_count_csv}'")

    # --- SU Distribution Analysis ---
    su_names = [name for name, _ in SU_DEFS]
    df_su = pd.DataFrame({
        'SU_Name': su_names,
        'Total_Count': total_su_hist.tolist()
    }).sort_values('Total_Count', ascending=False)

    print("\n" + "="*40)
    print("      Total Structural Unit Distribution")
    print("="*40)
    print(df_su.to_string(index=False))

    su_csv = _save_su_distribution(total_su_hist, output_dir)
    print(f"[✓] SU distribution data saved to '{su_csv}'")

    visualize_analysis_results(su_csv, node_count_csv, output_dir)
    print("[✓] Visualization images generated.")
    print("\nAnalysis complete.")


def su_real_ppm_stats(data_dir: Path, output_dir: Path):
    """
    统计所有 .pt 图文件中, 按 SU 类型聚合的真实核磁位移(real_ppm)分布范围。
    规则:
      - 仅统计碳相关 SU 节点 (data.is_carbon==True 且 real_intensity>0)
      - 非碳 SU 类型 (索引 26~32) 的 ppm 统计全部输出为 0
    输出: su_real_ppm_stats.csv (列: SU_Name, SU_Index, Count, PPM_Min, PPM_Max, PPM_Mean, PPM_Std)
    """
    output_dir.mkdir(exist_ok=True)
    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'.")
        return

    print(f"Computing SU real PPM stats from {len(pt_files)} graph files in '{data_dir}'...")

    # 初始化每个 SU 的 ppm 收集容器
    num_su = len(SU_DEFS)
    su_ppm_values = [[] for _ in range(num_su)]

    from torch_geometric.data import Data
    for f in tqdm(pt_files, desc="Scanning files"):
        try:
            data_dict = torch.load(f, weights_only=False)
            data = Data.from_dict(data_dict)
        except Exception as e:
            print(f"Warning: Could not load or process file {f}. Error: {e}")
            continue

        if not hasattr(data, 'real_ppm') or not hasattr(data, 'real_intensity') or not hasattr(data, 'su_type_indices') or not hasattr(data, 'is_carbon'):
            continue

        # 掩码: 仅取碳节点且有有效强度
        try:
            is_carbon_mask = data.is_carbon.bool()
        except Exception:
            # 兼容偶发类型
            is_carbon_mask = torch.tensor(data.is_carbon, dtype=torch.bool)

        intensity_mask = (data.real_intensity > 1e-6)
        valid_mask = is_carbon_mask & intensity_mask

        if valid_mask.numel() == 0 or not valid_mask.any():
            continue

        su_ids = data.su_type_indices[valid_mask]
        ppm_vals = data.real_ppm[valid_mask]

        # 收集到各自 SU 桶
        for su_idx, ppm in zip(su_ids.tolist(), ppm_vals.tolist()):
            su_ppm_values[su_idx].append(float(ppm))

    # 聚合统计
    su_names = [name for name, _ in SU_DEFS]
    rows = []
    for su_idx in range(num_su):
        name = su_names[su_idx]
        vals = su_ppm_values[su_idx]
        # 非碳 SU (26~32) 输出 0
        if su_idx >= 26:
            rows.append({
                'SU_Name': name,
                'SU_Index': su_idx,
                'Count': 0 if len(vals) == 0 else len(vals),
                'PPM_Min': 0.0,
                'PPM_Max': 0.0,
                'PPM_Mean': 0.0,
                'PPM_Std': 0.0
            })
            continue

        if len(vals) == 0:
            rows.append({
                'SU_Name': name,
                'SU_Index': su_idx,
                'Count': 0,
                'PPM_Min': 0.0,
                'PPM_Max': 0.0,
                'PPM_Mean': 0.0,
                'PPM_Std': 0.0
            })
        else:
            v = np.array(vals, dtype=float)
            rows.append({
                'SU_Name': name,
                'SU_Index': su_idx,
                'Count': int(v.size),
                'PPM_Min': float(v.min()),
                'PPM_Max': float(v.max()),
                'PPM_Mean': float(v.mean()),
                'PPM_Std': float(v.std())
            })

    df = pd.DataFrame(rows, columns=['SU_Name','SU_Index','Count','PPM_Min','PPM_Max','PPM_Mean','PPM_Std'])
    out_csv = output_dir / "su_real_ppm_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] SU real PPM stats saved to '{out_csv}'")

def intensity_stats(data_dir: Path, output_dir: Path, intensity_threshold: float = 2.0):
    """
    统计所有 .pt 图文件中节点级 NMR 强度 (real_intensity) 的分布。
    输出:
      - intensity_stats.csv: 统计摘要 (count, min, max, mean, std, percentiles)
      - intensity_distribution.png: 强度直方图
      - intensity_outliers.csv: 强度 > threshold 的异常样本列表
    """
    output_dir.mkdir(exist_ok=True)
    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'.")
        return

    print(f"Computing intensity stats from {len(pt_files)} graph files in '{data_dir}'...")

    all_intensities = []  # 收集所有有效节点的强度
    outlier_records = []  # 记录异常强度的样本
    file_intensity_stats = []  # 每个文件的强度统计

    for f in tqdm(pt_files, desc="Scanning files"):
        try:
            data_dict = torch.load(f, weights_only=False)
            data = Data.from_dict(data_dict)
        except Exception as e:
            print(f"Warning: Could not load file {f}. Error: {e}")
            continue

        if not hasattr(data, 'real_intensity') or not hasattr(data, 'is_carbon'):
            continue

        # 掩码: 仅取碳节点且有有效强度
        try:
            is_carbon_mask = data.is_carbon.bool()
        except Exception:
            is_carbon_mask = torch.tensor(data.is_carbon, dtype=torch.bool)

        intensity_vals = data.real_intensity[is_carbon_mask]
        valid_mask = intensity_vals > 1e-6
        valid_intensities = intensity_vals[valid_mask]

        if valid_intensities.numel() == 0:
            continue

        # 转为 numpy
        intensities_np = valid_intensities.numpy().astype(float)
        all_intensities.extend(intensities_np.tolist())

        # 检测异常值
        max_intensity = float(intensities_np.max())
        if max_intensity > intensity_threshold:
            outlier_records.append({
                'file': os.path.basename(f),
                'num_valid_nodes': len(intensities_np),
                'max_intensity': max_intensity,
                'mean_intensity': float(intensities_np.mean()),
                'num_above_threshold': int((intensities_np > intensity_threshold).sum())
            })

        # 文件级统计
        file_intensity_stats.append({
            'file': os.path.basename(f),
            'count': len(intensities_np),
            'min': float(intensities_np.min()),
            'max': max_intensity,
            'mean': float(intensities_np.mean()),
            'std': float(intensities_np.std()) if len(intensities_np) > 1 else 0.0
        })

    if not all_intensities:
        print("Error: No valid intensity data found.")
        return

    all_intensities = np.array(all_intensities)

    # --- 全局统计 ---
    stats_summary = {
        'Metric': ['count', 'min', 'max', 'mean', 'std', 'median',
                   'p25', 'p75', 'p90', 'p95', 'p99',
                   f'count_above_{intensity_threshold}', f'ratio_above_{intensity_threshold}'],
        'Value': [
            len(all_intensities),
            float(all_intensities.min()),
            float(all_intensities.max()),
            float(all_intensities.mean()),
            float(all_intensities.std()),
            float(np.median(all_intensities)),
            float(np.percentile(all_intensities, 25)),
            float(np.percentile(all_intensities, 75)),
            float(np.percentile(all_intensities, 90)),
            float(np.percentile(all_intensities, 95)),
            float(np.percentile(all_intensities, 99)),
            int((all_intensities > intensity_threshold).sum()),
            float((all_intensities > intensity_threshold).sum() / len(all_intensities))
        ]
    }
    df_stats = pd.DataFrame(stats_summary)
    stats_csv = output_dir / "intensity_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"\n[✓] Intensity statistics saved to '{stats_csv}'")
    print(df_stats.to_string(index=False))

    # --- 绘制直方图 ---
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 全范围直方图 (log-scale y)
    ax = axes[0]
    ax.hist(all_intensities, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=intensity_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold={intensity_threshold}')
    ax.set_xlabel('Intensity', fontsize=14)
    ax.set_ylabel('Frequency (log scale)', fontsize=14)
    ax.set_yscale('log')
    ax.set_title('Full Range Intensity Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(direction='in', labelsize=12)

    # 右图: 仅 0~3 范围内的细节
    ax = axes[1]
    clipped = all_intensities[all_intensities <= 3.0]
    ax.hist(clipped, bins=60, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax.axvline(x=intensity_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold={intensity_threshold}')
    ax.set_xlabel('Intensity', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Intensity Distribution (0 ~ 3)', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(direction='in', labelsize=12)

    fig.tight_layout()
    fig_path = output_dir / "intensity_distribution.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Intensity distribution plot saved to '{fig_path}'")

    # --- 保存异常样本列表 ---
    if outlier_records:
        df_outliers = pd.DataFrame(outlier_records)
        df_outliers = df_outliers.sort_values('max_intensity', ascending=False)
        outliers_csv = output_dir / "intensity_outliers.csv"
        df_outliers.to_csv(outliers_csv, index=False)
        print(f"[✓] {len(outlier_records)} files with intensity > {intensity_threshold} saved to '{outliers_csv}'")
    else:
        print(f"[✓] No files found with intensity > {intensity_threshold}")

    print("\nIntensity stats analysis complete.")


def split_by_intensity(
    data_dir: Path,
    ok_dir: Path,
    bad_dir: Path,
    intensity_threshold: float = 2.0,
):
    """
    基于节点级强度阈值对数据集进行质检筛选：
      - 若任意碳节点的 real_intensity > threshold，则判为"异常文件"
      - 否则判为"正常文件"
    并将 .pt 文件分别复制到 ok_dir / bad_dir 下。
    """
    ok_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'.")
        return

    total_ok, total_bad = 0, 0
    bad_examples_printed = 0

    print(f"扫描 {len(pt_files)} 个 .pt 文件，筛选强度 > {intensity_threshold} 的异常数据…")
    for f in tqdm(pt_files, desc="Intensity QC"):
        file_bad = False
        max_intensity = 0.0

        try:
            data_dict = torch.load(f, weights_only=False)
            data = Data.from_dict(data_dict)
        except Exception as e:
            file_bad = True
            reason = f"load_error: {e}"
        else:
            if not hasattr(data, 'real_intensity') or not hasattr(data, 'is_carbon'):
                file_bad = True
                reason = "missing_required_attributes"
            else:
                try:
                    is_carbon_mask = data.is_carbon.bool()
                except Exception:
                    is_carbon_mask = torch.tensor(data.is_carbon, dtype=torch.bool)

                intensity_vals = data.real_intensity[is_carbon_mask]
                valid_mask = intensity_vals > 1e-6
                valid_intensities = intensity_vals[valid_mask]

                if valid_intensities.numel() > 0:
                    max_intensity = float(valid_intensities.max().item())
                    if max_intensity > intensity_threshold:
                        file_bad = True
                        if bad_examples_printed < 5:
                            print(f"[强度异常] {os.path.basename(f)} | max_intensity={max_intensity:.4f} > {intensity_threshold}")
                            bad_examples_printed += 1

        # 复制文件到对应目录
        dst_dir = bad_dir if file_bad else ok_dir
        try:
            shutil.copy2(f, dst_dir / os.path.basename(f))
        except Exception as e:
            print(f"[拷贝失败] {f} -> {dst_dir}: {e}")

        if file_bad:
            total_bad += 1
        else:
            total_ok += 1

    print(f"\n筛选完成：正常 {total_ok} 个，异常(强度>{intensity_threshold}) {total_bad} 个。")
    print(f"正常文件已复制到: {ok_dir}")
    print(f"异常文件已复制到: {bad_dir}")


def split_by_su_ppm(
    data_dir: Path,
    ok_dir: Path,
    bad_dir: Path,
    margin: float = 0.0,
    intensity_threshold: float = 1e-6,
):
    """
    基于 coarse_graph.SU_PPM_RANGES 对数据集中每个 .pt 文件进行核磁标注质检：
      - 仅检查碳相关节点 (is_carbon==True 且 real_intensity>threshold)
      - 若任意有效节点的 real_ppm 落在其 SU 类型的范围之外(可加 margin)，则判为“错误文件”
      - 否则判为“正常文件”

    并将 .pt 文件分别复制到 ok_dir / bad_dir 下。
    """
    ok_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'.")
        return

    # 为每个 SU 索引构造 (low, high) 范围
    su_names = [name for name, _ in SU_DEFS]
    su_ranges = [SU_PPM_RANGES.get(name, (0.0, 240.0)) for name in su_names]

    total_ok, total_bad = 0, 0
    bad_examples = 0

    print(f"扫描 {len(pt_files)} 个 .pt 文件，依据 SU_PPM_RANGES 进行质检…")
    for f in tqdm(pt_files, desc="PPM QC"):
        file_bad = False
        try:
            data_dict = torch.load(f, weights_only=False)
            data = Data.from_dict(data_dict)
        except Exception as e:
            # 解析失败直接归为 bad
            file_bad = True
            reason = f"load_error: {e}"
        else:
            # 必要属性存在性检查
            required_attrs = ['real_ppm', 'real_intensity', 'su_type_indices', 'is_carbon']
            if not all(hasattr(data, attr) for attr in required_attrs):
                file_bad = True
                reason = "missing_required_attributes"
            else:
                # 构造有效节点掩码：仅检查碳节点且有有效强度
                try:
                    is_carbon_mask = data.is_carbon.bool()
                except Exception:
                    is_carbon_mask = torch.tensor(data.is_carbon, dtype=torch.bool)

                intensity_mask = (data.real_intensity > float(intensity_threshold))
                valid_mask = is_carbon_mask & intensity_mask

                if valid_mask.any():
                    idx_list = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
                    for idx in idx_list:
                        try:
                            su_idx = int(data.su_type_indices[idx])
                            if su_idx < 0 or su_idx >= len(su_ranges):
                                file_bad = True
                                reason = f"invalid_su_index:{su_idx}"
                                break
                            low, high = su_ranges[su_idx]
                            low = float(low) - margin
                            high = float(high) + margin
                            ppm_val = float(data.real_ppm[idx])
                            if not (low <= ppm_val <= high):
                                file_bad = True
                                # 仅记录少量示例，避免刷屏
                                if bad_examples < 5:
                                    su_name = su_names[su_idx]
                                    print(f"[标注异常] {os.path.basename(f)} | SU={su_name} (idx={su_idx}) | ppm={ppm_val:.2f} ∉ [{low:.2f}, {high:.2f}]")
                                    bad_examples += 1
                                break
                        except Exception as e:
                            file_bad = True
                            reason = f"node_check_error:{e}"
                            break
                else:
                    # 若没有任何有效碳节点（无真实标注），默认归为 bad 以便人工复核
                    file_bad = True
                    reason = "no_valid_carbon_nodes"

        # 复制文件到对应目录
        dst_dir = bad_dir if file_bad else ok_dir
        try:
            shutil.copy2(f, dst_dir / os.path.basename(f))
        except Exception as e:
            print(f"[拷贝失败] {f} -> {dst_dir}: {e}")

        if file_bad:
            total_bad += 1
        else:
            total_ok += 1

    print(f"\n质检完成：正常 {total_ok} 个，错误 {total_bad} 个。")

def batch_merge_graphs(data_dir: Path, output_dir: Path, target_node_counts: list[int], num_per_target: int, upsample_sus: list[str], upsample_factor: float):
    """
    Automated batch merging of graphs to create larger datasets with specific properties.
    """
    output_dir.mkdir(exist_ok=True)
    print("--- Starting Batch Merge ---")
    print(f"Source Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Node Counts: {target_node_counts}")
    print(f"Graphs per Target: {num_per_target}")
    print(f"Upsampling SUs: {upsample_sus} (Factor: {upsample_factor}x)")

    # --- 1. Pre-scan and index the dataset ---
    print("\nScanning and indexing dataset...")
    graph_metadata = []
    su_name_to_id = {name: i for i, (name, _) in enumerate(SU_DEFS)}
    upsample_su_ids = {su_name_to_id[name] for name in upsample_sus if name in su_name_to_id}

    pt_files = sorted(glob.glob(str(data_dir / "*.pt")))
    if not pt_files:
        print(f"Error: No .pt files found in '{data_dir}'. Aborting.")
        return

    for f_path in tqdm(pt_files, desc="Indexing graphs"):
        try:
            data_dict = torch.load(f_path, weights_only=False)
            data = Data.from_dict(data_dict)
            su_indices = set(data.su_type_indices.tolist())
            graph_metadata.append({
                "path": f_path,
                "num_nodes": data.num_nodes,
                "su_indices": su_indices
            })
        except Exception as e:
            print(f"Warning: Skipping file {f_path} due to error: {e}")
            continue
    
    if not graph_metadata:
        print("Error: No valid graph files could be indexed. Aborting.")
        return

    # --- 2. Calculate sampling weights for upsampling ---
    weights = []
    for meta in graph_metadata:
        if upsample_su_ids.intersection(meta["su_indices"]):
            weights.append(upsample_factor)
        else:
            weights.append(1.0)

    # --- 3. Generation loop ---
    for target_size in target_node_counts:
        print(f"\n--- Generating graphs for target size: ~{target_size} nodes ---")
        for i in range(num_per_target):
            files_to_merge = []
            current_nodes = 0
            
            # Use a copy of available indices to sample without replacement for each large graph
            available_indices = list(range(len(graph_metadata)))
            
            while current_nodes < target_size:
                if not available_indices:
                    print("Warning: Ran out of unique graphs to sample from. Merged graph may be smaller than target.")
                    break
                
                # Create weights corresponding to the currently available indices
                current_weights = [weights[idx] for idx in available_indices]

                # Sample an index from the available indices
                chosen_relative_idx = random.choices(
                    population=range(len(available_indices)),
                    weights=current_weights,
                    k=1
                )[0]
                
                # Get the original index from the list of available indices
                original_idx = available_indices.pop(chosen_relative_idx)
                graph_to_add = graph_metadata[original_idx]
                
                files_to_merge.append(Path(graph_to_add["path"]))
                current_nodes += graph_to_add["num_nodes"]

            print(f"  Generating graph {i+1}/{num_per_target} with {current_nodes} nodes from {len(files_to_merge)} files.")
            
            output_filename = output_dir / f"{target_size}_{i+1:02d}.pt"
            
            # Use the existing merge_graphs function to perform the merge
            merge_graphs(files_to_merge, output_filename)

    print("\n--- Batch merge complete! ---")


def merge_graphs(input_files: list[Path], output_file: Path):
    """
    Merges multiple graph .pt files into a single, larger graph object.
    """
    if len(input_files) < 2:
        print("Error: Merge command requires at least two input files.")
        return

    graphs_to_merge = []
    print(f"Loading {len(input_files)} graphs to merge...")
    for f in input_files:
        try:
            # 修正：加载字典后，将其转换为PyG的Data对象
            data_dict = torch.load(f, weights_only=False)
            graphs_to_merge.append(Data.from_dict(data_dict))
        except Exception as e:
            print(f"Warning: Could not load file {f}. Skipping. Error: {e}")

    if len(graphs_to_merge) < 2:
        print("Error: Not enough valid graphs could be loaded to perform a merge.")
        return

    # Initialize the merged graph with the first graph's data
    merged_data = graphs_to_merge[0].clone()
    
    # Iterate over the rest of the graphs and merge them in
    for i in range(1, len(graphs_to_merge)):
        graph_to_add = graphs_to_merge[i]
        num_nodes_before = merged_data.num_nodes

        # --- Concatenate node-level features ---
        merged_data.x = torch.cat([merged_data.x, graph_to_add.x], dim=0)
        merged_data.real_ppm = torch.cat([merged_data.real_ppm, graph_to_add.real_ppm], dim=0)
        merged_data.real_intensity = torch.cat([merged_data.real_intensity, graph_to_add.real_intensity], dim=0)
        merged_data.is_carbon = torch.cat([merged_data.is_carbon, graph_to_add.is_carbon], dim=0)
        merged_data.su_type_indices = torch.cat([merged_data.su_type_indices, graph_to_add.su_type_indices], dim=0)
        merged_data.su_max_degrees = torch.cat([merged_data.su_max_degrees, graph_to_add.su_max_degrees], dim=0)
        
        # --- Concatenate edge-level features (with offset) ---
        if graph_to_add.edge_index.numel() > 0:
            edge_index_offset = graph_to_add.edge_index + num_nodes_before
            merged_data.edge_index = torch.cat([merged_data.edge_index, edge_index_offset], dim=1)
            merged_data.edge_attr = torch.cat([merged_data.edge_attr, graph_to_add.edge_attr], dim=0)

        # --- Sum global features ---
        merged_data.y_spectrum += graph_to_add.y_spectrum
        merged_data.total_atom_counts += graph_to_add.total_atom_counts
        merged_data.su_hist += graph_to_add.su_hist
        
        # --- Combine SMILES ---
        # 修正：确保SMILES是字符串
        if hasattr(merged_data, 'smiles') and isinstance(merged_data.smiles, list):
            merged_data.smiles = "".join(merged_data.smiles)
        if hasattr(graph_to_add, 'smiles') and isinstance(graph_to_add.smiles, list):
            graph_to_add.smiles = "".join(graph_to_add.smiles)
            
        merged_data.smiles += "." + graph_to_add.smiles

    # Ensure output directory exists
    output_file.parent.mkdir(exist_ok=True)
    # V-Fix: Save as dict to maintain compatibility
    torch.save(merged_data.to_dict(), output_file)
    
    print(f"  - Total nodes in merged graph: {merged_data.num_nodes}")
    print(f"  - Total edges in merged graph: {merged_data.num_edges}")
    print(f"[✓] Merged graph saved to '{output_file}'")

def main():
    parser = argparse.ArgumentParser(
        description="Dataset utilities for analysis and merging.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Analyzer Sub-parser ---
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze the distribution of nodes and SUs in a dataset."
    )
    parser_analyze.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the processed .pt graph files."
    )
    parser_analyze.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset_analysis"),
        help="Directory to save analysis results (plots, csvs)."
    )

    # --- Visualization Sub-parser ---
    parser_visualize = subparsers.add_parser(
        "visualize",
        help="Render charts from previously generated CSV files."
    )
    parser_visualize.add_argument(
        "--su_csv",
        type=Path,
        required=True,
        help="Path to the SU distribution CSV (with SU_Index column)."
    )
    parser_visualize.add_argument(
        "--node_count_csv",
        type=Path,
        required=True,
        help="Path to the node-count distribution CSV."
    )
    parser_visualize.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save generated plots."
    )

    # --- Merger Sub-parser ---
    parser_merge = subparsers.add_parser(
        "merge",
        help="Merge multiple graph .pt files into a single large graph."
    )
    parser_merge.add_argument(
        "--inputs",
        type=Path,
        nargs='+',
        required=True,
        help="Two or more paths to the input .pt files to merge."
    )
    parser_merge.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the final merged .pt file."
    )

    # --- Batch Merger Sub-parser ---
    parser_batch_merge = subparsers.add_parser(
        "batch_merge",
        help="Automatically merge graphs to create large datasets with specific properties."
    )
    parser_batch_merge.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the source .pt graph files."
    )
    parser_batch_merge.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the generated large graph files."
    )
    parser_batch_merge.add_argument(
        "--target_node_counts",
        type=int,
        nargs='+',
        required=True,
        help="A list of target node counts for the merged graphs (e.g., 100 200 400)."
    )
    parser_batch_merge.add_argument(
        "--num_per_target",
        type=int,
        default=5,
        help="Number of merged graphs to generate for each target size."
    )
    parser_batch_merge.add_argument(
        "--upsample_sus",
        type=str,
        nargs='*',
        default=['Alkynyl_CH', 'Alkyl_Cq'],
        help="A list of rare SU names to give priority to during sampling."
    )
    parser_batch_merge.add_argument(
        "--upsample_factor",
        type=float,
        default=5.0,
        help="How much more likely to pick a graph containing a rare SU."
    )

    # --- SU real PPM stats Sub-parser ---
    parser_real_stats = subparsers.add_parser(
        "real_stats",
        help="Compute per-SU real PPM distribution stats from .pt files."
    )
    parser_real_stats.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the processed .pt graph files."
    )
    parser_real_stats.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_report"),
        help="Directory to save the stats CSV."
    )

    # --- Intensity Stats Sub-parser ---
    parser_int_stats = subparsers.add_parser(
        "intensity_stats",
        help="统计节点级 NMR 强度 (real_intensity) 分布，输出统计摘要、直方图和异常列表。"
    )
    parser_int_stats.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the processed .pt graph files."
    )
    parser_int_stats.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_report"),
        help="Directory to save the stats CSV and plots."
    )
    parser_int_stats.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Intensity threshold for outlier detection (default: 2.0)."
    )

    # --- Split by Intensity Sub-parser ---
    parser_split_int = subparsers.add_parser(
        "split_intensity",
        help="根据强度阈值筛选数据，将强度>阈值的异常文件分拣到 bad 目录。"
    )
    parser_split_int.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the source .pt graph files."
    )
    parser_split_int.add_argument(
        "--ok_dir",
        type=Path,
        required=True,
        help="Directory to save intensity-QC passed files."
    )
    parser_split_int.add_argument(
        "--bad_dir",
        type=Path,
        required=True,
        help="Directory to save intensity-QC failed files (intensity > threshold)."
    )
    parser_split_int.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Intensity threshold for filtering (default: 2.0)."
    )

    # --- PPM 质检分拣 Sub-parser ---
    parser_qc_ppm = subparsers.add_parser(
        "split_ppm",
        help="根据 SU_PPM_RANGES 对 real_ppm 进行质检，将 .pt 文件分拣到 ok/bad 目录。"
    )
    parser_qc_ppm.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the source .pt graph files."
    )
    parser_qc_ppm.add_argument(
        "--ok_dir",
        type=Path,
        required=True,
        help="Directory to save PPM-QC passed files."
    )
    parser_qc_ppm.add_argument(
        "--bad_dir",
        type=Path,
        required=True,
        help="Directory to save PPM-QC failed files."
    )
    parser_qc_ppm.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Allowable margin to widen SU_PPM_RANGES when checking (ppm)."
    )
    parser_qc_ppm.add_argument(
        "--intensity_threshold",
        type=float,
        default=1e-6,
        help="Only nodes with real_intensity greater than this will be checked."
    )


    args = parser.parse_args()

    if args.command == "analyze":
        analyze_dataset(args.data_dir, args.output_dir)
    elif args.command == "merge":
        merge_graphs(args.inputs, args.output)
    elif args.command == "batch_merge":
        batch_merge_graphs(
            args.data_dir,
            args.output_dir,
            args.target_node_counts,
            args.num_per_target,
            args.upsample_sus,
            args.upsample_factor
        )
    elif args.command == "real_stats":
        su_real_ppm_stats(args.data_dir, args.output_dir)
    elif args.command == "split_ppm":
        split_by_su_ppm(
            args.data_dir,
            args.ok_dir,
            args.bad_dir,
            args.margin,
            args.intensity_threshold,
        )
    elif args.command == "intensity_stats":
        intensity_stats(args.data_dir, args.output_dir, args.threshold)
    elif args.command == "split_intensity":
        split_by_intensity(
            args.data_dir,
            args.ok_dir,
            args.bad_dir,
            args.threshold,
        )

if __name__ == '__main__':
    main() 