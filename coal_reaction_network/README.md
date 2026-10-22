# Coal Reaction Network · v3

从 ReaxFF bonds 轨迹逐帧提取片段、执行 C/R/O 泛粗粒度分类，并通过 atom-ID lineage
构建保留完整计量关系的反应网络。针对约 6GB 可用内存提供有界任务队列、磁盘后处理与分页界面。

## 先读：旧数据库不可直接续跑

v2 及更早解析器没有跳过 bonds 文件的 `mol` 列，导致键级、`abo/nlp/q` 错位。
v3 会拒绝写入这些数据库。**保留旧结果用于比对，使用新的数据库路径重新解析原始轨迹。**
不要修改旧数据库中的版本号绕过检查，也不要只重新计算旧片段的反应网络。

当前解析格式：

```text
id type nb id_1 ... id_nb mol bo_1 ... bo_nb abo nlp q
```

每条原子记录必须恰好有 `7 + 2*nb` 列。每个输入文件只能包含一个 timestep，必须有粒子数头部。
重复原子、未知邻居、未知元素质量、非法数值及多 timestep 文件会明确失败。
若文件声明了生成时的键级 cutoff，请求的 cutoff 不得低于它：文件中未输出的低键级不能恢复。

## 6GB 配置下的完整运行

运行环境：Python 3.10+，Linux / WSL；Dashboard 的类别筛选需要 SQLite JSON 函数支持。
核心、Dashboard 与测试不需要第三方 Python 包或前端 CDN。
在仓库根目录运行：

```bash
python3 coal_reaction_network/src/trajectory_pipeline.py /path/to/bonds \
  --database coal_reaction_network/results/full_network_v3/network.sqlite \
  --workers 2 \
  --pair-workers 2 \
  --max-inflight 4 \
  --memory-limit-mib 3072 \
  --expected-frame-step 1 \
  --retry-failed
```

- `--workers`：单帧解析进程数，默认 2。
- `--pair-workers`：帧对比较进程数，默认继承 workers。
- `--max-inflight`：已提交但未消费的任务上限，默认最大 worker 数的两倍；必须不小于 worker 数。
- `--memory-limit-mib`：分析进程树 RSS 的软保护阈值，默认 3072 MiB；超限退出并保留已提交结果，降低并发后恢复。
- `--expected-frame-step 1`：检查文件名帧号是否连续；若文件名编号按 100 递增，请改为 100。
- `--expected-frame-count N`：可选，验证发现的输入数。
- 不给预期步长时，“完整”仅表示输入清单内全部帧和相邻帧对已处理，不表示验证了原始输出没有缺帧。
- `--minimum-free-disk-mib`：启动时的最低磁盘余量，默认 512 MiB；这不是完整轨迹的容量估算，应预留远大于此的空间。

默认按文件名最后一个数字排序；重复文件名帧号会拒绝，避免混入不同轨迹。Header timestep 可以重复。
`--frame-order timestep` 可显式按头部排序。时间曲线默认使用数据库内部 `frame_id`，不是物理时间。

**只有所有预期帧和帧对成功，才发布完整网络。**单帧/帧对失败时状态为 `partial`，退出码为 2，
不会将不完整运行显示为完成。`frames_completed` 只代表 `--stage frames` 完成。

数据库附近的 `network.progress.json` 保存当前阶段、阶段进度、进程树内存和错误。
内存值为 Linux `/proc` 的采样 RSS 求和，可能重复计算共享页；不是内核硬内存限制，也不含浏览器、VS Code 和系统占用。
在途任务数固定，但小型输入清单本身仍随帧数增长。不要在 6GB 环境同时启动多个大型独立分析。

## 恢复、输入与配置版本

中断后重新运行相同命令：已提交帧/帧对跳过，遗留 `processing` 恢复为 `pending`。
`--retry-failed` 重试相同输入下失败的任务。单数据库文件锁禁止重复写入；worker 不继承父进程的 SQLite 连接。

默认输入指纹包含路径、timestep、文件大小及纳秒修改时间。
`--strict-input-hash` 额外逐文件计算 SHA-256；首次及恢复都需使用相同选项。
输入清单或解析配置改变会拒绝复用；修复/替换已指纹化的错误文件后应使用新的数据库。
分析期间不要修改输入文件。

默认读取 `configs/network.json`，可以用 `--config /path/to/config.json` 替换。CLI 显式参数优先。
配置可包含：

```json
{
  "elements": {"1": "C", "2": "H", "3": "O"},
  "masses": {"C": 12.011, "H": 1.008, "O": 15.999},
  "bond_order_cutoff": 0.3,
  "execution": {"workers": 2, "memory_limit_mib": 3072},
  "matching": {"minimum_event_conservation": 0.9},
  "temporal_filter": {
    "mode": "off", "minimum_lifetime_frames": 1,
    "reversal_window_frames": 0, "preserve_fast_intermediates": true
  }
}
```

可选 `classification` 对象接受 `carbon_bins`、`ring_bins`、`oxygen_bins`，
每个区间为 `[标签, 下界, 上界或null]`，必须从零连续覆盖到无穷。
解析器版本、描述符版本、元素、质量、cutoff 和类别定义属于解析缓存键。
改变时间过滤只重建派生结果；改变守恒阈值更新事件解析等级并重建聚合，不重新解析帧。

旧的 `--minimum-shared-atoms` / overlap 参数保留调用兼容性，但**不删减 raw lineage**；
所有非零 atom-ID 重叠均保留，包括两个大分子间少量原子转移。

## 分类与事件的科学含义

```text
ReaxFF bonds → 片段实例 → C/R/O 类别 → 相邻帧 lineage
             → 计量反应事件 → 全轨迹聚合通道 → 按需派生视图
```

- 默认类别数上限为 `5 × 5 × 4 = 100`；使用稳定 ID，如 `C40+_R10+_O3+`。
- `C40+` 表示碳数 >40，`R10+` 表示独立环数 >10；其他区间见 `pseudo_species.py`。
- 环数统一采用连接分量的独立环数 `E - V + 1`；不是几何环数或芳香环数。
- 质量和估计自由基不参与类别定义。自由基用典型价态与正确解析的 `abo` 差值估计，仍需针对力场校准。
- 相同 atom ID、类别和连接拓扑指纹才判为 `persistence`；相同类别内连接变化记为 `internal_rearrangement`。
- 没有指纹的外部片段记录只能判为 `coarse_persistence`，不能据此宣称无化学变化。
- 指纹覆盖阈值化的原子连接，不覆盖未改变连接的键级变化、电子态或立体化学。
- `fragmentation` / `fusion` / `transformation` / `many_to_many` / `birth` / `death` 保留各自实例证据。
- 事件由相邻帧 lineage 连通分量构建，跨采样间隔的多步机理不能强行解释为基元反应。
- raw 保存全部事件，通道聚合不把 persistence 当作反应；类内重排会进入通道。
- 保留 `A → A + B`、`A+B → B+2C` 两侧完整计量，不抵消相同物种，也不替换为两两 source-target 边。
- 事件频数不是反应速率，启发式质量评分不是统计置信概率。动力学建模仍是独立后续工作。

默认验证所有帧的 atom ID/元素集合不变。真正的开放体系可显式使用 `--allow-atom-changes`，
此时必须人工检查 birth/death、守恒和缺失原子，不能把输入损坏解释成真实反应。

## 轨迹和时间视图

`fragment_tracks` 表示一对一且 atom membership 完全相同的实例连续轨迹。
分裂、融合、原子交换建立新 track；祖先关系在 `track_links` 中单独保存。
同一 track 的类别改变时，新建 `residence_segments`，时间过滤使用这个类别驻留时间，
而不是祖先轨迹长度。轨迹边界/缺口处未完整观测的驻留段按 censored 处理，不直接作为短寿命排除。

```bash
python3 coal_reaction_network/src/trajectory_pipeline.py /path/to/bonds \
  --database coal_reaction_network/results/full_network_v3/network.sqlite \
  --stage postprocess --temporal-mode annotate \
  --minimum-lifetime-frames 2 --reversal-window-frames 1
```

- `off`：默认不构建大规模时间标注；需要 track 浏览时加 `--build-tracks`。
- `annotate`：标记候选，保留全部事件。
- `exclude`：只在 filtered 中排除普通短寿命/逆转候选，不删除原始事件。
- 默认保留短寿命、原子守恒的 `fast_intermediate_candidate`，包括同时满足逆转候选者。
- 逆转只在后续帧、反向计量签名及实例连续关系匹配时标注；同帧对或无关分子的反向通道不算逆转。
- 即便实例匹配，逆转也仅是候选，不是“已证明的假反应”。
- track 构建按帧 checkpoint 恢复；标注及通道先生成临时磁盘结果，再事务替换已发布结果。
  标注/聚合中断会保留旧已发布表，但当前运行不标记 ready；恢复时重建未完成的派生阶段。

## 数据库、导出和阶段

主要表：`frames`、`fragments`、`species`、`species_states`、`frame_pairs`、`lineage_links`、
`events`、`event_members`、`reaction_channels`、`reaction_stoichiometry`、
`fragment_tracks`、`residence_segments`、`track_links`、`event_annotations`、`network_view_channels`。
`network_view_events` 为虚拟视图，不为三个网络复制三份事件成员表。

SQLite 使用 WAL、受限页缓存和磁盘临时表。片段 atom IDs 为压缩二进制；批处理不存完整 atom details 或环路径。
默认不导出可能很大的逐帧 `species_states.csv`，需要时加 `--export-states`；CSV 与网络 JSON 均流式写入。
`--no-network-export` 跳过三套网络视图文件，数据库仍保留完整结果。

`--stage frames` 仅处理帧；`--stage transitions` 复用帧并处理帧对及后处理；
`--stage postprocess` 只重建时间分析/聚合；默认 `all`。

若备份正在使用 WAL 的数据库，应使用 SQLite backup 或停机后的安全快照；不要只复制 `.sqlite` 而忽略 WAL。

## Dashboard

```bash
python3 coal_reaction_network/src/network_dashboard.py --host 127.0.0.1 --port 8765
```

通过浏览器访问本机 `127.0.0.1:8765`；VS Code Remote/WSL 用户可转发端口 8765。
也可继续使用 Notebook 中的 `start_dashboard()`，API 兼容。
只允许本地监听；不应未经认证直接暴露到公网。

界面提供：
- 默认显示按事件次数排序的 12 条主要计量通道，统计对象为类别数量变化；可切换类别数量不变的活动、全部非持续事件和物质流投影。
- 反应事件与持续存在分别计数；跨类别与类别内累计碳/氧/质量转移分项显示，界面默认不计入类别内物质流。
- 数据集选择、受限并行任务、分阶段进度、RSS/峰值、取消与恢复；单个 Dashboard 同时仅启动一个分析。
- 物种—反应—物种的镜像二部图：完整系数、同名物种双侧参与、颜色分类、缩放/平移、SVG 子图导出。
- 视图、帧区间、类别、C/R/O 区间、事件类型、解析等级与频次筛选；网络和实例证据共用事件筛选条件。
- 分页通道表、具体事件、逐帧片段、分页 atom ID 和连接指纹。
- 有限点数的时间曲线，按显示桶求均值；缺席物种记零，population 同时保留 min/max 包络。

展示筛选与分页不会删改完整数据库。左右相同名称代表同一个逻辑类别，不是两个新物种。
对于上百条通道不要试图一次把所有标签铺到屏幕上，优先按物种/时间窗口聚焦。

查询端点均带明确限制或游标：`/api/network`、`/api/events`、`/api/frames`、`/api/fragments`、
`/api/fragment`、`/api/lineage`、`/api/tracks`、`/api/species-states`、`/api/timeseries`。
大 SQL 查询有 30 秒超时保护，超时后请缩小帧范围。

## 单帧与两帧工具

```bash
python3 coal_reaction_network/src/analyze_bonds.py coal_reaction_network/data/bond3512.reaxff \
  --output-dir coal_reaction_network/results/bond3512_v3
python3 coal_reaction_network/src/visualize_frame.py \
  coal_reaction_network/results/bond3512_v3/bond3512_fragments.json
python3 coal_reaction_network/src/compare_bonds.py \
  coal_reaction_network/data/bond0.reaxff coal_reaction_network/data/bond3512.reaxff \
  --output-dir coal_reaction_network/results/transitions_v3
```

`visualize_transitions.py` 现在接受带完整计量的 transition/network JSON；
`--max-edges` 兼容旧选项名，但现在限制的是显示的反应通道数。只有 source-target 投影的旧文件应重新生成。
Notebook 中的 K-means 为单帧探索工具，不作为跨轨迹稳定类别主键。

## 测试与压力验证

```bash
python3 -B coal_reaction_network/tests/run_tests.py
python3 -B coal_reaction_network/tests/benchmark_pipeline.py \
  --frames 100 1000 10001 --atoms 96 --output /tmp/coal-scalability.json
python3 -B coal_reaction_network/tests/benchmark_pipeline.py \
  --frames 100 1000 --atoms 5724 --output /tmp/coal-scalability-5724.json
```

标准库测试 runner 不需要 pytest；HTTP 集成测试需要允许本机临时端口。
压力测试在临时目录生成合成输入并清理，覆盖解析、帧对、track、标注和聚合，记录进程树 RSS，
同时验证帧对完整性、同帧 track 唯一性及通道计数。**合成测试不替代完整真实轨迹的科学验收。**

修改概览及验收边界见 `docs/v3_low_memory.md`；本次观测结果见 `docs/validation_v3.md`。
# C/R/O 主网络扩展

新增以 **C/R/O 区间为唯一物种定义** 的物质流主图、完整计量图切换、碳/氧/质量/次数权重、覆盖率筛选、分类占用诊断和逐帧对守恒审计。
实现位于 `src/coarse_network/`，前端投影独立于 `src/web/coarse-network.js`；不引入详细分子物种网络。

**已有完成的 v3 数据库需要执行一次 `--stage postprocess` 才有新增物质流；不需要重新解析已完成帧。**
加载数据库不会自动修改数据。界面也可选择“已有 v3 · 仅构建网络与物质流”。
完整使用方式、模块划分、统计含义与边界见 [C/R/O 网络说明](docs/cro_network.md)。
回归、合成万帧测试和真实样本验证见 [本次验证记录](docs/validation_cro.md)。
# 时间路径与三维交互

在“视图”中选择**路径 · 阶段概览**，可切换三维阶段流程、压缩事件三维图和原始逐事件二维图。选择起止帧、锚点片段 / 类别后，扫描每个相邻帧对；默认按 4 个阶段汇总，并可展开逐事件证据。阶段摘要不是单步反应，也不保证展示阶段内部的短寿命中间体。连接使用空间曲线，分支分层展开；三维左键拖动旋转、Shift＋拖动平移、左键按住＋滚轮以光标位置缩放；普通滚轮不缩放，二维在固定容器内滚动。
节点显示稳定 P 编号及明确 C/R/O 区间，内部数据库 ID 不变。已有完成的 v3 帧 / 事件可直接查询，无需重建物质流。
使用、模块划分、局部主分支定义和分页 / 任务边界见 [路径说明](docs/pathway.md)。

路径布局统一为**从左到右的时间轴＋稳定分支通道**，持续状态用驻留线段表示，不再蛇形折行。提供紧凑时间 / 真实帧距切换、二维横向滚动、三维底部时间浏览条、5%–400% 缩放、全图适配和选中节点聚焦。三维旋转采用已有 SVG 元素的增量坐标更新，不重建表格和场景；仍为 SVG 正交投影，不是 WebGL 引擎。默认受限旋转便于阅读时间，自由观察模式可能改变屏幕上的时间方向。

界面与统计口径审查记录见 [本次审查](docs/interface_review.md)。
