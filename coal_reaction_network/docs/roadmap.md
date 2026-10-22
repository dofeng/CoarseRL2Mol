# 后续开发路线图

> 本文下面的“已完成”小节记录早期阶段设计；v3 的实际行为以 README 和
> `docs/v3_low_memory.md` 为准。v3 已修复 mol 列解析，改为有界任务调度、
> 实例 track / 类别驻留分离、磁盘时间窗口和计量网络界面。
> 默认按文件名排序，raw lineage 不应用比例裁剪，时间 off 跳过昂贵时间分析。

## 已完成：第一阶段——正确性基础

- 伪物种使用稳定的 `C*_R*_O*` 主键，不再依赖发现顺序生成字母 ID。
- lineage 匹配只遍历共享 atom ID 的候选片段。
- 事件由 parent-child 二部图连通分量生成。
- 支持 `persistence`、`transformation`、`fragmentation`、`fusion`、
  `many_to_many`、`birth` 和 `death`。
- 事件保存 atom 守恒率、质量误差、元素差额、置信度和解析等级。
- 完整计量关系保存在事件成员表和 reaction stoichiometry 表中。

## 已完成：第二阶段——大规模轨迹管线

- 按 bonds 文件头中的 timestep 建立稳定帧索引。
- 单帧提取和相邻帧比较采用独立多进程阶段。
- SQLite 主进程串行写入，worker 不竞争写锁。
- atom ID 使用压缩二进制 BLOB 保存。
- 默认不保存完整 atom details 和 ring path。
- 每帧及每个帧对具有 `pending/processing/completed/failed` 状态。
- 重启后自动恢复遗留的 `processing` 任务并跳过已完成任务。
- 配置哈希和输入清单哈希防止不兼容结果混用。

## 已完成：时间状态与可逆派生网络

- 默认 `--temporal-mode off`，一帧片段和快速反应全部保留。
- `annotate` 模式计算 fragment track 生命周期、短寿命候选和反向通道候选，但不删除事件。
- `exclude` 模式只重建 `filtered` 派生视图；`events`、`raw` 和 `annotated` 仍完整保留。
- fragment track、事件标注和网络视图都写入 SQLite，可在改变过滤参数时复用原始解析结果。
- `raw_reaction_network.json` 保存完整反应超边，包含多反应物/多产物计量关系。
- 短生命周期但 atom 守恒的事件标记为 `fast_intermediate_candidate` 并默认保留，避免把快速真实反应误删。

## 第三阶段——ReaxFF 反应去噪

### 键级滞后

将单一 cutoff 改为成键和断键双阈值：

```text
formation_cutoff = 0.35
breaking_cutoff  = 0.25
```

中间区域继承上一帧键状态，降低 cutoff 附近振荡导致的假反应。

### 多帧持续性过滤

新增三帧或五帧窗口，只有连接变化持续指定帧数才确认事件。数据库增加：

```text
raw_event
confirmed_event
suppressed_flicker
```

### 可配置事件拆分

对 `many_to_many` composite event 使用三帧窗口尝试拆分为多个局部事件；无法唯一拆分时继续保留 composite，不强行解释为基元反应。

## 第四阶段——动力学和状态模型

- 根据 `species_states` 计算每个反应物的累计 population exposure。
- 输出离散时间转移概率和经验反应频率。
- 对单分子通道估计一阶经验速率。
- 对多反应物通道保存联合暴露量，不默认宣称满足质量作用定律。
- 增加 timestep 到物理时间的转换配置。
- 输出可用于 ODE、CTMC 或 population-balance 模型的计量矩阵。

建议新增表：

```text
reaction_exposures
reaction_rates
stoichiometric_matrix
```

## 第五阶段——全局数据驱动粗粒化

固定 C/R/O 区间继续作为可解释的主网络。若需要 K-means 或其他聚类：

1. 从完整轨迹进行分层抽样；
2. 全局拟合 scaler 和聚类中心；
3. 固化模型版本；
4. 对所有帧只执行 predict；
5. 保存旧 species 到新 cluster 的映射。

禁止逐帧重新训练聚类，否则 cluster ID 无法跨时间比较。

## 第六阶段——数据库驱动 Dashboard

Dashboard 不再现场解析 bonds 文件，而是查询已完成的 SQLite/DuckDB 数据库，提供：

- species population 和 total mass 时间序列；
- reaction channel 频率、置信度和守恒误差筛选；
- fragment atom-ID lineage 追踪；
- persistence、resolved、composite、unresolved 分层显示；
- 反应超图和普通投影图切换；
- 指定时间区间、物种和反应通道的子网络导出；
- 批处理进度、失败帧和失败帧对检查。

## 第七阶段——生产化验证

- 使用人工小轨迹覆盖所有事件类型和守恒边界。
- 使用连续短时间 ReaxFF 轨迹标注真实成键/断键事件。
- 比较不同 cutoff、滞后阈值和输出间隔的网络稳定性。
- 增加百万片段级性能基准和内存峰值测试。
- 增加数据库迁移版本，允许后续升级 schema 而不丢失已处理结果。
