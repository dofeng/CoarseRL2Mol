# 拓扑读取、实时组装与过程回放

`assembly_viewer` 是独立的可视化目录，复用 `RL_MTCS` 的搜索与装配接口，不修改引擎源码、候选、奖励、随机种子或停止规则。

## 启动完整界面

在项目目录执行：

```bash
/home/dofengqi/miniconda3/envs/NMRer/bin/python -m assembly_viewer ui --port 8765
```

打开 `http://127.0.0.1:8765`。远程机器通过端口转发访问，界面中的路径是计算服务器上的文件路径。

1. 选择样品或填写节点 CSV、拓扑 CSV、输出根目录与 Python 可执行文件，设置参数。
2. 点击 **读取拓扑**。只执行原初始化接口，读取和校验全部资源。芳香团簇、柔性链、脂肪环、侧链与取代请求显示在同一画布；这一步不执行 MCTS 或 NMR。
3. 点击 **开始组装**。正式计算在独立进程执行，同一画布逐步显示刚性、柔性、环、侧链、取代过程。
4. 搜索中显示实际访问的试连接；选中候选不代表接受。前瞻判断、检查点接受及回退分别标记。默认前瞻只显示状态和完成率，切换 **快速前瞻试算** 可以在同一画布播放前瞻连接。
5. 镜头默认聚焦当前动作涉及的真实连接组件，避免临时暂存团簇距离过远而让连接缩成小点。节点坐标不变；点击 **适应画布** 可查看所有已放置组件，取消 **聚焦当前连接** 可持续查看全局。
6. 播放暂停只暂停画面。每个有意义的连接事件按顺序显示，不自动跳到最新帧；界面显示当前播放位置与已记录位置。
7. 阶段按钮恢复该阶段的实际起始状态，然后点击播放。勾选 **只回放所选阶段** 会在其结束事件停止。完整历史保存在磁盘，重新启动界面服务后仍可打开。
8. 运行结束后导出离线回放、阶段 SVG/PDF/PNG 和过程 MP4；界面提供最终 SVG/PDF 链接，最终结构 JSON 由原引擎输出。

不在前端增加预算或调整算法。预览中的分区位移只用于展示，不写入实际坐标。正式帧使用记录中的真实坐标；SU25 固定中心分支、中点取代、节点删除及身份变化均保留。

## Jupyter 中保留参数、缩短入口

先将项目加入路径，再导入接口：

```python
import sys
sys.path.insert(0, "/home/dofengqi/Graphnmr")

from RL_MTCS import AssemblyConfig
from assembly_viewer import open_workbench

config = AssemblyConfig(
    connection_candidate_limit=10,
    forward_checkpoint_interval=4,
    forward_min_completion=0.80,
    forward_max_rollbacks=3,
    clearance_radius=3,
    c_puct=1.414,
    seed=27,
    spectrum_csv=None,
    element_composition_csv=None,
    g2s_checkpoint=None,
    reward_weights=(0.40, 0.15, 0.20, 0.20, 0.05),
    verbose=True,
    save_stage_images=True,
)
ui = open_workbench("YM-3", config=config)
```

点击输出的链接即可进入与命令行相同的界面。不会自动开始计算，也不使用嵌套 iframe。参数仍可在页面修改；开始后本轮参数被冻结。关闭 notebook 或浏览器不会终止独立装配进程。关闭 notebook 中的界面服务：

```python
ui.shutdown()
ui.server_close()
```

## 回放与导出

```bash
python -m assembly_viewer serve /path/to/output/process_trace --port 8765
python -m assembly_viewer export /path/to/output/process_trace --html /path/to/replay.html --frames /path/to/figures
python -m assembly_viewer export /path/to/output/process_trace --video /path/to/process.mp4
python -m assembly_viewer figure /path/to/output/final_molecule.json
```

界面视频使用当前过程范围；命令行可通过 `--scope live` 输出搜索试连接、确认和回退，默认 `formal` 输出已确认的连接过程。

`figure` 从已有分子 JSON 直接导出 SVG、PDF 和 300 DPI PNG 到同目录的 `publication_figures/`，不重新装配，也不需要过程记录。支持包含 `molecule` 的最终结果 JSON。`--output-dir` 可指定目录，`--formats svg pdf` 可仅输出矢量图；PNG/TIFF 的分辨率可用 `--dpi` 调整。

静态图的 SU 文字大小、圆边和连接线宽均按实际显示的节点直径计算，因此放大、缩小以及不同分子跨度不会改变文字与圆的比例。浅色节点使用深色文字。SVG 保留可编辑文本，PDF 嵌入 TrueType 字体；节点身份、实际坐标、连接边和 SU 颜色沿用原结果。静态图目标键长为 25 磅，图幅限制为 3–20 英寸，防止刚性暂存组件的远距离坐标产生巨量像素。大分子需在矢量图中放大查看；缩成论文单栏时应另外配局部放大图。矢量导出改善清晰度，不改变结构的实际疏密。

视频需要所用 Python 环境的 PATH 中存在 `ffmpeg`。离线 HTML 不依赖服务器、外部脚本或 CDN。HTML 包含可回放状态，长运行可能生成较大的文件；在线查看采用分页和最多 12 个状态的缓存。视频帧写入临时磁盘后编码，不把全部帧留在内存。零连接但已有初始化结构的合法部分结果也可以导出。

合成演示：

```bash
python -m assembly_viewer demo
```

演示明确标记为合成数据，不能作为 YM-3 或 LY-2 的实验结果。

## 文件与接口

```text
assembly_viewer/
  workflow.py       持久运行目录，读取/开始生命周期，独立进程
  worker.py         只读初始化和显式开始计算
  resources.py      输入资源清单，配置和源码校验
  adapter.py        实例级观察器，记录原引擎事件
  recorder.py       SQLite 状态差分、事件和阶段起点
  server.py         单界面 HTTP/SSE，读取/开始/历史/导出
  export.py         离线页面、阶段图片和视频
  web/              唯一 HTML、JavaScript、CSS
  examples/         notebook 示例
  tests/            原引擎一致性、流程与浏览器测试
```

每次运行保存：

```text
output/
  run.json                 输入、冻结参数、Python 和运行命令
  run.log                  原引擎运行日志与异常
  final.json               原引擎最终结果
  process_trace/
    trace.sqlite3          有序事件、状态差分、阶段索引
    resources.json         全部原始拓扑资源和节点身份
    manifest.json          配置、来源哈希、状态、完成情况
  replay.html              按需导出
  process_figures/         按需导出
  process.mp4              按需导出
```

运行列表保存在 `assembly_viewer/runs/registry.sqlite3`。也可在页面填写其他输出目录重新打开记录。重新打开不恢复搜索、不重新运行装配。旧格式和旧多画布入口已删除，不提供兼容转接。

输入预览后若 CSV、芳香计划、拓扑资源表或引擎源码变化，开始时会拒绝运行，要求重新读取。浏览器、服务关闭不停止 worker；计算异常不会被包装成“正常无候选”。已确认结构的合法性与是否全部完成分别显示。

同步批处理仍可使用 `assemble_with_trace` 或 `python -m assembly_viewer record`。界面观察器不保留额外分子搜索树，只把状态差分和统计写入磁盘。真实搜索树仍由原引擎负责。

## 记录开销与阶段输出

搜索画面使用原引擎的完整身份、标签、合同和几何合法性检查，关闭仅用于形态诊断的全分子凸包/空洞扫描，标记为 `audit.diagnostics_scope="legality_only"`。搜索奖励、快速前瞻及其审计照常执行；原引擎输出的阶段与最终文件继续包含原有完整审计。记录层不再为评分和重访反复投影同一个状态，也不复制随后会丢弃的动作历史。

事件、状态差分和转移按事务批量写入，产生事件时约每 200 ms 发布一次，搜索轮、检查点、阶段结束、正常关闭或异常退出时提交剩余记录。未跳过连接、访问或回退事件；实时播放可以落后于计算，暂停画面不会暂停计算。记录层最多缓存 4 个显示用 JSON，未缓存分子搜索树；回放重建逐步应用差分，避免每一步都复制整个分子。

五个阶段仍自动生成各自的 `*_molecule.json`、`*_summary.json` 和 `*_search_trace.json`；`save_stage_images=True` 时生成阶段 SVG/PDF/300 DPI PNG。最终结构始终生成三种图片，路径记录在 `final.json` 的 `outputs` 中。最终 JSON、节点/边 CSV、审计和未连接资源报告仍由原引擎保存，无需点击网页导出。页面的阶段完成事件发生在原引擎写完对应文件之前，图片写入期间可能稍晚出现在输出目录。

新渲染器仅接入 `assembly_viewer` 的实例级输出适配，不改动 `RL_MTCS` 的 Python 文件或全局函数。直接运行 `RL_MTCS.assemble_from_csv` 仍使用原绘图接口，之后可用上述 `figure` 命令重新导出。三种格式共用一次图形构建，仅在选定阶段/最终输出时绘制，不在搜索候选或前瞻过程中制图。

更新代码后需重启界面服务，并重新点击“读取拓扑”再开始；已启动的独立 worker 不会热更新代码。

完成状态按实际连接任务判断：`all_complete=True` 允许逆向声明的剩余资源；
`strict_all_nodes_assembled` 单独表示所有原始节点是否均已装配。未用于连接任务的
剩余 SU11 在芳香模板中保持 SU13，节点 ID 和原始 SU11 身份保留，
`aromatic_residual_nodes` 记录这些允许的变化。节点详情同时显示原始类型、
装配目标类型和当前类型；它们不会列入待恢复标签。闲置芳香节点和未装配闭合资源
仍在报告中保留各自的状态与原因，真正缺失合同、错误标签和几何错误仍接受审计。

2026-10-14 性能验证：实际 LY-1 输入、`connection_candidate_limit=10`、`seed=879`、NMR 关闭，刚性单轮搜索（不含前瞻）原引擎耗时 238.9 秒，带记录耗时 293.7 秒，额外开销约 23%。两者均比较 10 个结果、访问 131 个状态，最终状态签名和动作历史完全一致。与旧运行已经保存的前缀比较，1961 条候选、访问和奖励事件一致；旧运行在 1110.7 秒时仍为 7/10 个候选，随后被中断，不能将其当成完整运行耗时。此验证覆盖真实刚性搜索；五阶段输出、图片、实时/离线回放及视频另用合成装配测试验证。
