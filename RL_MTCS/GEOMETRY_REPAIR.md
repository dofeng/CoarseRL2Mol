# 刚性位点与阶段输出修复

运行入口仍为 `assemble_from_csv()`，使用 `AssemblyConfig`。调试时关闭
`spectrum_csv` 和 `g2s_checkpoint`，NMR 批量限制保持单图。

## 本次变更

- 刚性搜索在实际环位点上绑定逻辑 SU10，再执行直接连接；不增加节点。
  合法的三叉、四叉连接不再被度数大于二的规则普遍惩罚。
- 绑定是芳香外围占位 ID 的双射置换。环模板、非芳香单元的内部节点与边不变；
  芳香节点的物理归属及按 GID 表示的芳香内部边可随置换改变。
  `aromatic_site_bindings` 记录实际单元、模板 slot 和局部坐标。
- 移除了旧的固定归属配对交换搜索。现在采用有界可行性探测为 PUCT 提供初始路径，
  每个父节点仍保留，后续搜索可以替换早期动作。
- 每个保留刚性候选分别重建 Flex 分量和候选域，分享总试探预算。
  侧链失败时最多重试 `stage_retry_limit` 个其他刚性候选（默认 2，最大 4），
  额外 Flex 总预算和 Side 总预算分别不超过一份 `search_budget`。
- 只将几何合法性作为完成的硬条件；形状、分量均衡属于偏好。
- Flex 和 Side 的部分结果也提交端点映射后再导出，避免实际边与合同不同步。
- 初始芳香图片只显示 SU12/SU13，隐藏 GID；诊断 JSON 保留计划类型与映射。
- 输出增加 `flex_chain/` 和 `aliphatic_ring/` 独立检查点；保留刚性分量图、
  `side/`、`substitution/`、总图。环未完成时仍不启动侧链。
- 禁止模板外补点。环外节点和环外“芳香边”在初始化时直接报错。

## 当前原始 HSQ 阻塞

`test_results/HSQ-1/outer_cycle_1/final_outputs/final_aromatic_clusters.json`
中的 `aromatic:0` 有 21 个节点、20 个环位点。GID 78 不在环内，
通过 `22–78` 挂接。原始暂存记录合计 300 个真实芳香节点加 1 个五元环取代
暂存节点，模板仅有 300 个环位点。

当前输入会返回 `input_geometry_invalid`，而不是继续搜索或宣称组装完成。
本次未修改逆向 CSV/模板文件，未删除 GID 78，也未新增芳香团簇。
上游现已允许最多 5 个未承担合同的真实 SU13 闲置，模板本身不扩展。
重新导出会保留 GID 78 与 SU30 暂存对，将剩余 SU13 记入
`idle_aromatic_nodes`。源节点 CSV 不删除该节点，RL 装配图不包含闲置节点。
旧快照必须重新导出；读取器不会静默修复旧 JSON 中的环外挂点。

逆向 Stage3 使用统一的柔性链数量区间：`L=X−N10/2+1`、`U=floor(0.8X)`。
区间宽度超过 6 时，专用事务按完整 SU10 对尝试 `10→12`，逐次重算模板、
区间及 SU12 上限，并保留实际柔性链数在区间内。普通资源转换池仍不开放 SU10。
事务复用已有 Layer1/2 重建与失败回退，不把改标签当成完成拓扑转换。
HSQ 数量候选为 6 个 SU10 转为 SU12：17 个芳香单元、11 对刚性连接、
柔性链区间 `[7,13]`。这不是完整几何装配成功的证明。

## 验证

```bash
/home/dofengqi/miniconda3/envs/NMRer/bin/python -m pytest -q \
  tests/test_rl_mtcs_csv.py tests/test_rl_mtcs_sites.py tests/test_aromatic_coordination.py
```

测试分别覆盖原始 HSQ 拒绝、输入库存、三叉/四叉位点连接、端点映射回放、
PUCT 回退、候选预算分配、树上限、环与侧链完整连接及阶段图片。
大拓扑测试使用明确标注的合成 fixture，未回写原始 HSQ。

较大合成阶段运行观测峰值约 256 MiB；这是单次有界测量，尚不是长期内存稳定性证明。
