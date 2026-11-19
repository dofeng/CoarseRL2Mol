# Fig. 3：正向模型的整体评价与单样本预测

沿用原图的 Arial、7 pt 正文、蓝色密度图和色盲友好配色，补全真实预测示例。
绘图只读取已有评估结果，不重新训练、不改动模型，也不覆盖 `output_figures/` 的旧图。

## 一键生成

从项目根目录执行：

```bash
MPLCONFIGDIR=/tmp/graphnmr-mplconfig \
  /home/dofengqi/miniconda3/envs/NMRer/bin/python \
  plot/fig3/plot_predictions.py
```

已有 numpy、pandas、matplotlib、Pillow 的其他 Python 环境也可以直接运行。
默认输出到 `plot/fig3/output_figures_revised/`：

| 面板 | 图像 | 含义 |
| --- | --- | --- |
| a | `S2N_regression` | 全部“分子—SU类型”计数对的预测与参考值 |
| b | `G2S_test_regression` | 全部有标签碳中心的化学位移预测与参考值 |
| c | `S2N_group_per_su_mae` | 各类别内的逐 SU 计数 MAE，不是类别总数 MAE |
| d | `G2S_hop_ablation` | 不同子图半径的已有 MAE、RMSE |
| e | `S2N_prediction_example_<ID>` | 一个样本的参考/预测分布，保留全部 33 类及零值 |
| f | `G2S_spectrum_example_<ID>` | 同一样本的参考位移/G2S预测位移重构谱 |

`Fig3_forward_models_complete` 是 **180 × 198 mm** 的六面板排版；
`Fig3_preview.png` 是较小的检查预览，不作为投稿原图。
所有独立图和合图均导出 PDF、SVG、600 dpi PNG、LZW 压缩 TIFF。
SVG 保留文本，PDF 嵌入 TrueType 字体，白色不透明背景。
导出不使用 `bbox_inches='tight'`，因此不会偷偷改变指定的物理尺寸。
已有回归图维持 75 × 72 mm；类别与消融图维持 75 × 58 mm；新增示例为 88 × 58 mm。

## 关键统计修正

令 `e[i,j] = |pred[i,j] - true[i,j]|`，则：

- 整体散点图：`mean(e[i,j])`，包括所有分子和全部 33 个 SU 类型。
- 新类别柱图：`mean(e[i,j] for j in category)`，仍然先取逐 SU 的绝对误差。
- 原类别总数误差：`mean_i(|sum_j pred[i,j] - sum_j true[i,j]|)`，不再用于主图。

新类别指标可按类别 SU 类型数加权，准确还原整体 MAE，而不是对五个柱子简单平均。
这不等于把原类别总数 MAE 除以类型数，因为原口径可能发生误差抵消。
默认测试集的整体 MAE 仍为 **1.6591**；新柱值依次为：

| 类别 | SU索引 | 类型数 | 新逐SU计数MAE |
| --- | --- | --- | --- |
| Carbonyl/nitrile | 0–4 | 5 | 1.4254 |
| Aromatic | 5–13 | 9 | 1.5071 |
| Unsaturated | 14–18 | 5 | 1.4310 |
| Aliphatic | 19–25 | 7 | 2.5366 |
| Heteroatom | 26–32 | 7 | 1.3070 |

碳基类别 0–25 中每个 SU 含一个碳；26–32 不含碳。
`plot.py --use_mae_per_100c` 使用每个样本的真实碳数，对逐 SU 绝对误差归一化后求均值。
它不是原来的类别总数归一化误差。原始评估 CSV 不作修改。

## 示例选择与谱图的解释边界

默认不选“最好看的”预测。先求两个评估表交集中的每个样本的 S2N MAE 与 G2S 位移 MAE，
分别转为百分位秩，再选择两者距 0.5 的绝对距离之和最小的样本；并列按完整来源路径排序。
默认有 495 个共同样本，选中 **15007**；完整选择表导出到 `source_data/example_selection.csv`。
这只是共同测试样本中的中间误差示例，不能代表所有分子尺寸的整体性能。

G2S 示例采用真实保存的 `true_ppm`、`pred_ppm`，不伪造模型输出：

1. 检查所选样本的全部碳中心是否与 S2N 参考 SU 数量逐类一致。
2. 两条谱线使用相同的 Lorentzian 核，默认 HWHM = 1 ppm、FWHM = 2 ppm。
3. 每个碳中心的权重都固定为 1；G2S 只预测位移，并没有独立预测峰强。
4. 使用参考曲线最大值作为**同一个**除数缩放两条曲线，保留相对峰高差异。
5. 不对两条曲线单独最大值归一化，不移动峰位，不额外平滑或拟合展宽。

因此，图注应写 **reference-shift reconstruction vs G2S-shift reconstruction**，
不能称为“实验谱与预测谱对比”。图中 `Shift MAE` 也是位移 MAE，不是谱强 MAE。
S2N 分布示例和 G2S 谱示例使用同一样本，但后者基于其参考分子图，
并不意味着 S2N 的计数输出能直接唯一确定分子图。

### 为什么暂不使用 51121

- `plot/NMR示意/51121.png` 是分子结构图，不是两条谱线的预测结果。
- `51121.csv` 只有一条参考谱，不能从中推导出真实的 G2S 预测结果。
- 该样本不在这两个现有测试预测表中。
- 对 `51121.txt` 的 SMILES 实测调用现有粗粒度构建器时，原子索引 **6、7（零起始）**
  无法分配到当前 33 类词表，抛出 `UnassignedAtomsError`。

本次没有修改词表或强行替换其结构单元，也没有把其他分子的预测冒充为 51121。
指定 `--sample-id 51121` 会明确报错，而不是静默替换为其他样本。
若之后解决并验证词表覆盖问题、获得相应模型的真实预测记录，可提供新的两张评估表再作图。

## 可调整参数

```bash
python plot/fig3/plot_predictions.py \
  --sample-id 15007 \
  --hwhm 1.0 --ppm-step 0.1 --curve-style filled \
  --width-mm 180 --height-mm 198 \
  --example-width-mm 88 --example-height-mm 58 \
  --dpi 600 --font Arial \
  --out-dir plot/fig3/output_custom
```

- `--sample-id auto`：默认可复现选择；也可以指定 ID 或完整来源路径。
- `--s2n-points`、`--g2s-points`、`--g2s-ablation`：替换真实预测与消融源文件。
- `--curve-style lines`：去掉区域填充；默认两条曲线都填充。
- 所有颜色、字号、线宽继承 `plot.py` 的 `NatureStyleConfig`。
- 仅重画原来四张图：`python plot/fig3/plot.py --figure_set all --out_dir OUTPUT`。
- 旧 `--group_csv` 参数仍可传入，但会提示忽略旧类别总数表，从逐点结果重算。
- 缺少 Arial 时显式报错；可安装该字体或传入已有字体，不会悄悄更换。
- 图像不得在排版软件中继续缩小而不检查最终字号；具体单/双栏宽度以目标期刊指南为准。

## 源数据与质量检查

`source_data/` 包括全部回归点、33 类逐类误差、5 类统一口径误差、原始消融结果、
示例原始计数/位移、单位权重、谱重构数据和完整样本选择表。
`figure_metadata.json` 记录源文件 SHA-256、样本数、全局指标、选样规则、
谱展宽/归一化、字体、实际尺寸和未解决的统计报告限制。
本次实际导出检查结果见 `QA.md`。
不凭空添加多随机种子误差线或置信区间；数据划分沿用已有导出，未重新做结构泄漏审计。
密度图使用全部样本与完整 bin 计数范围，不截断高密度色阶；非有限值和重复记录会报错。

```bash
MPLCONFIGDIR=/tmp/graphnmr-mplconfig \
  /home/dofengqi/miniconda3/envs/NMRer/bin/python \
  -m unittest discover -s plot/fig3/tests -v
```

### 可用图注草稿

**Fig. 3 | Forward-model performance and matched single-molecule examples.**
**a**, Predicted versus reference SU counts for 2,698 test samples and all 33 SU types
(89,034 molecule–SU pairs). **b**, Predicted versus reference chemical shifts for 5,480
carbon centers from 495 test molecules. Colour indicates the number of observations
per density bin. **c**, Mean absolute count error per SU type within each structural
category, including zero-count observations; errors are averaged before pooling SU
types and are not errors of category-total counts. **d**, Reported MAE and RMSE for
subgraph radii of 1–4 hops. No seed-to-seed uncertainty estimates were available.
**e**, Reference and S2N-predicted counts for all 33 SU types of sample 15007.
**f**, Spectra reconstructed from the reference and G2S-predicted shifts of the same
sample using identical unit carbon weights and Lorentzian kernels
(HWHM, 1 ppm). Both curves share one reference-derived intensity scale; G2S does not
predict peak amplitudes. The example is selected by proximity to the median
per-molecule error ranks of both models among their common test records.
Source data are provided with the figure bundle.

投稿前还应按确定的目标刊再次核查官方 figure guide 与该刊作者指南。
当前配置是可调整的保守排版起点，不是对所有 Nature 子刊的一揽子格式保证。
