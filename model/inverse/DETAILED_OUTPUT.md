# 逆向推理：简易输出与详细输出

`InversePipelineV3.infer()` 新增 `output_mode`，默认值为 `"simple"`。

| 模式 | 用途 | 输出行为 |
| --- | --- | --- |
| `simple` | 常规运行 | 保持原有推理和输出行为，不生成逐阶段谱线追踪文件。 |
| `detailed` | 论文绘图、检查调整过程 | 在常规结果之外保存谱线、残差、节点、SU计数和动作；同时启用最终完整诊断。 |

该开关只控制观察与导出，不修改候选搜索、接受条件、随机状态、谱线计算公式或最终结果选择。详细模式增加数据复制和磁盘写入，默认不逐步生成PNG。`save_intermediates` 继续控制原有中间结果，与新追踪开关独立；即使设置 `save_intermediates=False`，详细模式仍会输出追踪数据。原有 `final_output_detail="compact"/"full"` 继续有效；`detailed` 自动包含 `full` 的最终诊断。

## 启用方式

在原有调用中增加 `output_mode="detailed"`，保留你原来的其他推理参数。建议为本次重跑使用单独目录：

```python
nodes, H_final = pipeline.infer(
    S_target,
    E_target,
    output_dir="/home/dofengqi/Graphnmr/test_results/HSQ_detailed",
    eval_lib_path="/home/dofengqi/Graphnmr/z_library/subgraph_library_filtered.pt",
    output_mode="detailed",
    # 其他参数保持原有设置。
)
```

在Notebook中使用时，更新代码后先重启内核并重新创建 `pipeline`，再运行推理，避免继续使用内存中的旧类。恢复常规模式可设置 `output_mode="simple"`，或省略该参数。

命令行入口同样支持 `--output-mode detailed`。在原有 `python -m model.inverse.inverse_pipeline ...` 命令后追加该参数即可；其余模型、谱图和元素参数保持原样。

## 数据位置和字段

命令行和逆向 Notebook 默认在重采样后用 1.5 ppm 窗口做二阶 Savitzky–Golay 平滑，然后连接低信号窗口中最低 20% 采样点的中位坐标得到分段直线。默认窗口为 0–7、65–95、186–194 和 223–240 ppm。扣除基线后将负值截到零，再按 `π × 碳数` 归一化。`input_spectrum.csv` 保存重采样谱（`raw`）、平滑谱（`smoothed`）、实际扣除的基线（`baseline`）、扣除后未归一化的谱（`corrected`）和推理输入（`target`）。`observed` 标记 CSV 覆盖的化学位移范围，`baseline_anchor` 标记最接近所选锚点的采样位置。默认窗口在裁剪谱上不足两个时，改用实测范围的两端估计一条直线。

两个入口默认完整扣除估计基线。命令行用 `--smooth-window-ppm` 和 `--baseline-strength` 调节平滑和基线扣除，两者同时设为 0 可复现仅面积归一化的输入。输入谱在默认窗口有明显信号时，可重复使用 `--baseline-window LOW HIGH` 选择窗口，Notebook 中使用 `baseline_windows=[(low, high), ...]`。指定两个窗口以拟合一条直线。

详细输出保存在 `<output_dir>/spectral_trace/<run_id>/`，运行时会打印完整路径。已有同名追踪目录时自动增加后缀，不覆盖早先追踪文件。

```text
spectral_trace/<run_id>/
  manifest.json
  target.csv
  checkpoints.csv
  events.jsonl
  000001_initial_layer1_raw/
    spectrum.csv
    nodes.csv
    su_counts.csv
    node_changes.csv
    metadata.json
  ...
  outer_cycle_1_layer3/
    layer3_spectrum.csv
    layer3_z_moves.csv
```

- `checkpoints.csv`：全部可绘制谱线的索引，包含阶段、外循环编号、父状态、接受标记、谱学计算口径、R²、RMSE、MAE、节点数和相对父状态发生变化的节点数。
- `spectrum.csv`：完整的2400点 `ppm, target, reconstructed, residual`，其中 `residual = target - reconstructed`。同时提供对应的 `*_display` 列；所有阶段统一除以目标谱的最大绝对值，便于堆叠填充绘图。原始数值保留，不对每条曲线分别归一化，也不额外平滑。
- `nodes.csv`：该时刻的节点ID、SU类型、化学位移、模板键、一/二跳邻居、固定连接、评分分量和完整z向量。它记录当时逆向推理的局部环境和节点关系；完整分子的MCTS组装仍属于后续步骤。
- `su_counts.csv`：全部33种SU的计数、父状态计数和变化量。首状态没有父状态时，`has_parent=False`，变化量置零。
- `node_changes.csv`：相对父状态实际发生变化的节点字段，包括SU、位移、模板键、邻居和z，便于标注关键调整。
- `metadata.json`：该状态的指标、动作及接受原因等上下文。额外的分区R²使用0–90、90–160和160–240 ppm分析窗；这些分析窗不代表常规分峰的结构归属边界。
- `events.jsonl`：候选接受/拒绝、Stage3循环决策、待重建转换、Layer3整体接受/回退和最终完成事件。不会因内存中的循环历史截断而丢失已写出的事件。
- `manifest.json`：输入元素、运行配置、模板库信息、共同显示缩放因子、最终选中循环和导出错误数等。`checkpoint_export_seconds` 仅统计谱线检查点的导出耗时，不代表详细模式的全部额外耗时。

`layer3_spectrum.csv` 和 `layer3_z_moves.csv` 在Layer3候选整体接受后保存；如果发生整体回退，候选内部的谱和动作仍保存在相应检查点和事件中。

## 记录哪些阶段

| `stage` | 记录时点 | 绘图用途 |
| --- | --- | --- |
| `initial_layer1_raw` | 最终选中的初始方案，其未细化Layer1结果 | 可选的最初基线。 |
| `initial_layer2_raw` | 同一初始方案未经Layer1细化的Layer2结果 | 诊断分支，不作为后续细化的直接父状态。 |
| `initial_layer1_refined` | 初始Layer1细化完成 | 展示初始局部邻域细化。 |
| `initial_layer2` | 初始Layer1–2正式完成 | 建议作为Layer4调整前的基线。 |
| `outer_cycle_input` | 每轮外循环的实际输入 | 核对循环起点。 |
| `block_a_accepted` | 每次Block A候选经Layer1–2重建、验证并被接受后 | 逐次展示实际接受的A调整。 |
| `block_a_complete` | 本轮Block A结束 | A阶段端点。 |
| `block_b_accepted` | 每次Block B子阶段重建、验证并被接受后 | 区分不同类型的B调整。 |
| `block_b_complete` | 本轮Block B结束 | B阶段端点。 |
| `stage3_spectrum_rebuild_accepted` | Stage3谱驱动重建被正式接受后 | 展示谱驱动调整；SU24后续校验仅在整体接受后记录。 |
| `stage3_external_rebuild_accepted` | Stage3外部拓扑触发的新一代重建被接受后 | 展示拓扑调整后的有效重建。 |
| `stage3_rebuild_accepted` | Stage3其他有效新一代重建被接受后 | 展示资源/拓扑校正的阶段谱。 |
| `stage3_complete` | Stage3完成、进入Layer3之前 | Stage3端点。 |
| `layer3_sample_initialization` | Layer3固定模板样本初始化 | 单独标明谱学口径转换。 |
| `layer3_sample_move` | Layer3每次实际接受的离散样本移动 | 观察候选内部如何改善；整体接受前为临时状态。 |
| `layer3_accepted` / `layer3_rollback` | Layer3候选整体接受或回退后 | 本轮正式保留结果。 |
| `final_selected` | 按原算法选出最终最佳外循环后 | 最终反演谱，与常规最终输出对应。 |

如果关闭资源阶段，Layer3前记录为 `layer3_input`。没有被接受的调整不会人为插入进展；因此阶段端点可能与上一个状态完全一致。Stage3尚未重建的资源转换只记录事件，不导出可能已经过时的谱线。初始阶段只导出最终选中方案的状态，不导出所有竞争方案。

## 如何选择连续绘图状态

先读 `checkpoints.csv`，按 `parent_checkpoint_id` 追溯父状态。`sequence` 是导出顺序，不能单凭行号认定所有曲线组成一条连续改善轨迹。

外循环继续使用原算法保留的Layer2/Stage3状态作为下一轮起点，而不是上一轮Layer3优化后的谱。`spectral_contract` 区分Layer1库位移、Layer2模板中心位移与Layer3样本位移；跨口径或跨外循环的R²未必单调。

Layer3内部检查点标记为 `decision="provisional"`。应结合 `layer3_decision` 事件判断整个候选是否被接受；整体回退的候选不能放进正式改善路径。`final_selected` 的父状态指向实际被选中的最佳外循环，可能不是最后运行的一轮。

图5可先选择同一轮的 `initial_layer2 → block_a_accepted/complete → block_b_accepted/complete → stage3_*_accepted/complete → layer3_accepted → final_selected`。若关键谱变化集中在某个阶段，再从该阶段筛选实际接受且变化明显的检查点；保留上述父状态和口径信息核对连续性。

导出异常会发出提示并写入 `export_errors.jsonl`，不改变推理接受条件。运行结束后检查 `manifest.json` 的 `status` 和 `export_error_count`：正常完成为 `complete`；没有可用完整Stage3/Layer3结果为 `incomplete`；其他意外中止可能保留 `running`，此时已完整写出的检查点仍可检查。

## 回归检查

在项目根目录、`NMRer`环境中运行：

```bash
python -m pytest model/inverse/tests/test_spectral_trace.py -q -p no:cacheprovider
```

用例检查谱/残差一致性、共同显示缩放、SU变化、只读观察、真实Layer3接受/回退，以及两轮推理协调流程的模式一致性和最佳循环选择。测试中的库存与拓扑候选服务使用小规模替身，不替代完整HSQ数据的重跑。


## Stage3收敛与重建上限

Stage3会在端口平衡后再次检查窗口调整的SU10/SU12方向和最终flex范围，避免平衡操作撤销已规划的调整。完整重建后的去重使用保留SU、模板、化学位移、锚点属性和固定边的图同构检查；节点编号仍用于实际资源绑定，不会被改写。失败候选记录按化学图状态保存，返回等价状态时不会重新获得搜索额度。

以下限制同时适用于`simple`和`detailed`，与谱线导出开关独立，均要求正整数：

| block_c_resource参数 | 默认值 | 作用 |
|---|---:|---|
| `max_rebuilds` | 64 | 本次Stage3调用的累计Layer1/2重建次数，包括失败候选、谱学候选和SU24后续评估。 |
| `max_planner_attempts` | 256 | 本次Stage3调用的累计规划次数。 |
| `max_attempts_per_generation` | 32 | 同一代图上的规划和物化尝试次数。 |
| `max_no_progress_rebuilds` | 8 | 连续未接受新化学图状态的重建次数；只有实际提交新状态后才重置。 |

原Notebook参数可继续使用，无需新增配置。需要调整时，在已有配置上更新这些键，例如：

```python
stage_configs["block_c_resource"].update({
    "max_rebuilds": 64,
    "max_planner_attempts": 256,
    "max_attempts_per_generation": 32,
    "max_no_progress_rebuilds": 8,
})
```

达到上限时返回最后完整提交的Layer2状态，保留节点、SU直方图与谱的一致性，并将Stage3标记为未完成；不会把受限停止视为完成或直接进入Layer3。`stage3_meta`的诊断中包含`search_budget`及明确停止原因：`stage3_rebuild_limit`、`stage3_planner_attempt_limit`、`stage3_generation_attempt_limit`或`stage3_no_progress_rebuild_limit`。详细模式额外记录`stage3_search_stopped`事件。

指定`output_dir`后，两种输出模式都会在受限停止时将该完整状态保存到Stage3输出目录下的`stopped_checkpoint/stage3_input_snapshot/`，并在元数据`resume_snapshot_dir`及日志中给出路径。保存格式与原始Stage3输入快照相同，包含节点、H、目标谱、元素、配置和校验和；若谱学调整窗口已经结束，续跑配置会保持其关闭。此快照供继续Stage3使用，不代表完整推理结果。显式再次调用会开始新的搜索额度，不会无限自动重试；保存失败会给出警告和`resume_snapshot_error`。

可用现有`pipeline.replay_stage3_from_snapshot(...)`直接从`stage3_input_snapshot`验证Stage3，避免重复运行Layer0、Layer1/2初始化和Block A/B；其`stage_configs`同样接受上述限制。
