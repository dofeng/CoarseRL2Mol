# C/R/O 泛粗粒度反应网络

## 范围与定义

网络节点仅由 C/R/O 区间决定，不引入详细分子物种、SMILES、反应中心分类或机器学习。
C 是片段碳原子数，O 是氧原子数，R 是连通片段的独立环数 `E−V+1`，不是芳香环数。
默认区间与现有分类保持一致；通过配置 `classification` 修改三维区间。改变区间须使用新数据库，不能混用不同定义。

数据层次：片段实例 → 原子溯源事件 → 类别计量通道 → 类别物质流投影。
完整通道保留 `A→A+B`、`A→2B`、`2A→2A` 等两侧计量。后两者中的同类别参与者仍是不同的实例。
既不抵消两侧同类，也不把同一反应的多条流边当成多个独立反应。

## 模块管理

```text
src/
  reaxff_fragments.py          原始解析、连通片段与属性
  pseudo_species.py           唯一的 C/R/O 区间分类定义
  fragment_lineage.py          实例关系、完整计量事件
  trajectory_pipeline.py       调度、数据库写入、阶段编排
  network_postprocess.py       驻留 / 时间标注、既有通道聚合
  coarse_network/
    build.py                  逐帧对物质流、成员与数量审计、断点恢复
    queries.py                权重筛选、覆盖率、物质流投影、区间诊断
    export.py                 完整派生统计的流式 CSV 导出
  dashboard_queries.py        通用只读查询、事件证据与组成曲线
  dashboard_app.py             HTTP 路由和后台任务管理
  web/
    coarse-network.js         C/R/O 主图、分类诊断、物质流证据与 JSON 导出
    app.js                    通用交互、计量图、任务、图表
tests/
  test_coarse_network.py       泛粗粒度语义、守恒、恢复和投影测试
  test_dashboard_http.py       HTTP / 任务 / 新增接口集成测试
  benchmark_pipeline.py       包含物质流与查询的合成压力测试
```

新功能采用 v3 的**附加派生表**，不重写现有原始表定义，不兼容错误解析器生成的 v2 数据。
未执行新派生阶段的 v3 数据库仍可查看已有计量通道；主物质流图明确提示需要后处理，而不是显示“成功但没有网络”。

## 运行

所有命令从仓库根目录执行。输入、pattern、排序、cutoff、元素映射、质量表和分类配置必须与原分析一致。

### 对已完成的 v3 结果增加物质流

```bash
python3 coal_reaction_network/src/trajectory_pipeline.py /path/to/original/bonds \
  --database coal_reaction_network/results/reaction_network_v3/network.sqlite \
  --stage postprocess --workers 2 --max-inflight 4 --memory-limit-mib 3072
```

此命令会更新指定数据库的派生表及其导出；需要保留旧版本快照时，请先使用 SQLite 备份方式备份，勿在写入期间只复制主数据库文件。
已完成帧和事件不会重新解析，但仍检查原始输入 manifest；首次构建物质流需读取原始文件获得原子元素清单。
闭合体系复用相同 atom-signature 的清单；开放体系至多缓存两种清单，并拒绝同一共享原子 ID 改变元素。
新物质流逐帧对原子提交、失败回滚、取消后断点恢复。改变时间过滤只影响派生视图，不重新计算已完成的原始物质流。

若原始帧或事件尚未完成，使用 `--stage all --retry-failed`，而不是仅执行后处理。

### 从新轨迹构建

```bash
python3 coal_reaction_network/src/trajectory_pipeline.py /path/to/bonds \
  --database coal_reaction_network/results/my_cro_network/network.sqlite \
  --workers 2 --max-inflight 4 --memory-limit-mib 3072 \
  --expected-frame-step 1 --retry-failed

python3 coal_reaction_network/src/network_dashboard.py --port 8765
```

也可在界面中选择“已有 v3 · 仅构建网络与物质流”。加载已有结果本身是只读操作，不隐式修改数据库。

## 物质流与守恒

`coarse_event_flows` 每行由 `(帧对,事件,来源类别,去向类别)` 唯一确定，保存共享原子数、碳数、氧数和质量。
每个共享原子在单个事件中计一次；原子的去向由原子 ID 确定，而非按产物质量比例估计。

- `persistence` 和 `coarse_persistence` 不贡献反应物质流。
- 类内连接变化、类别不变的原子交换均保留；API 的 `include_internal` 默认仍为 true，界面默认关闭以避免类内累计量主导排名。
- 流量包括反应参与片段中留在同类别的原子，并不表示这些原子的键都发生了变化。
- “隐藏同类别转移”只去掉物质流对角项，不删改完整反应式。碳流/氧流/质量权重也随该选项改变。
- 无碳事件的碳权重为零，选择“事件次数”可查看它们；同理适用于无氧事件。
- `birth/death` 的未匹配原子计入审计，不伪造来源 / 去向流边。
- 累计流可能多次计入同一原子。单位为原子转移次数或质量累加（默认原子质量表下为 u·次）；不是产率、物理通量或速率常数。

`coarse_pair_audits` 校验：每个片段恰好属于一个帧对事件、每个事件类别系数符合实际片段、流原子总数等于事件共享原子数、通道累计的类别净变化等于两帧实际数量变化。
任何上述结构一致性失败都阻止派生网络发布。元素不平衡和未匹配原子另外记录，允许显式开放体系查看边界事件。
类内改变即使净数量为零也保留，不能用类别数量差反推反应。

## 界面与筛选

1. **物质流主图**：横向 C、纵向 R、颜色与标签 O；位置随区间定义固定，节点面积随窗口平均质量占比变化。
2. **主要通道计量图**：显示按同一权重选择的完整反应式，保留反应节点及各参与者系数。
3. **全部计量通道**：沿用完整通道分页，按次数排序；同样支持统计对象与三维区间过滤，不采用权重或覆盖率选择。
4. **分类诊断**：显示最大区间占用率、各类平均数量和质量/碳/氧占比，以及类内/跨类事件与审计摘要。类别缺席的已完成帧按零参与平均。
5. **组成曲线**：支持数量、质量、质量占比、碳占比、氧占比；没有相应元素时占比返回零，不能解释成存在该元素。
6. 点击类别查看组成曲线；点击流边查看真实共享原子的实例证据，再点击片段追溯原子 ID。

主图先按时间窗、raw/annotated/filtered、类别参与者、事件类型和解析等级确定候选通道，再以累计权重选择通道。
覆盖率分母是该候选集在最少次数阈值与显示上限之前的总权重。界面明确报告未达目标，不将显示上限伪装成 100% 覆盖。
类别过滤要求存在同一个参与类别同时满足所有选定 C/R/O 条件；不会裁掉反应式另一侧的参与者。

界面默认最多显示 12 条通道，API 默认 40 条且上限 200 条；流边返回上限 4000 条，超限标记截断。
事件权重的覆盖率按事件 / 通道计数，不按投影流边计数；一条事件可能贡献多条流边。
流边详情中实例查询沿用网络的事件与参与类别筛选，不受权重和通道显示上限约束；证据包含该类别对的全部匹配通道。
本版不保证筛选图保留所有桥接通道或路径，不能把流图中的图论路径解释为计量可达性。

## 输出与接口

- 原有 `reaction_channels.csv` 和 `network_views/` 保留完整计量网络。
- `coarse_channel_flows.csv`：完整 raw 网络按反应式、来源类别和去向类别聚合的原子 / 碳 / 氧 / 质量转移；其中 `event_count` 是该类别对的贡献事件数，不能跨类别对求和当作独立事件总数。
- `coarse_pair_audits.csv`：每个帧对的守恒与完整性摘要。
- `network_summary.json` 中 `coarse_analytics`：审计帧对数、元素不平衡数、未匹配原子次数和派生流记录数。
- UI 可导出当前网络 JSON（含查询、完整所选计量、覆盖率和物质流）及 SVG；不是完整数据库导出。
- `--no-network-export` 跳过新增的大型 CSV 导出，但仍构建数据库内的派生统计。

只读接口：`/api/coarse-network`、`/api/coarse-diagnostics`；事件证据复用 `/api/events` 的 `flow_source` / `flow_target` 参数。
所有 DB 查询有超时和 SQLite 文件临时存储。主网络聚合在临时表中执行，Python 仅保留有限通道与流边；超时会明确报错，不返回冒充完整的部分网络。

## 内存和验证边界

帧对构建只保留相邻帧的片段、最多两种原子清单和一个事件的聚合，不建立全轨迹原子历史。
运行内存预算是软预算，不是操作系统硬上限；数据库新增派生表会增加磁盘占用。
过粗区间可能让大量过程成为自环，这是分类的可见分辨率限制，不应伪造跨类别连接。
本版未增加新反应去噪算法；阈值以下未保存的键不能恢复，现有阈值化连接也不覆盖全部键级变化。

执行回归：`python3 -B coal_reaction_network/tests/run_tests.py`。
执行压力测试：`python3 -B coal_reaction_network/tests/benchmark_pipeline.py --frames 1000 10001 --atoms 96`。
浏览器验收脚本 `tests/browser_network_check.py --database /path/to/network.sqlite` 使用独立安装的 Playwright，检查真实结果、证据、布局和非空 SVG 像素；路径交互另由 `tests/browser_pathway_check.py` 验证。合成和界面测试不能替代反应机理或动力学模型的科学验证。
