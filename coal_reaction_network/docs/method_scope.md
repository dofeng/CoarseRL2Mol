# 方法范围与核心对象

## 三层对象

### 1. Fragment instance

某一输出帧中由 bonds 图的 connected component 得到的真实片段，具有稳定的 atom ID 集合、质量和结构描述符。

### 2. Pseudo-species class

第一版只使用三个分类维度：碳数量、环数量、含氧量。自由基位点和分子质量不参与节点分类，而作为每个片段实例及节点统计状态的内在属性保存。该类别是一个集合，不等同于单一确定分子。

### 3. Reaction event

由相邻输出帧之间的 fragment lineage 产生的实例级转化，再聚合为类别级计量式，例如 `A -> A + B + C`。

## 节点状态

每个类别节点在时间 `t` 至少包含：

```text
population:      n_A(t)
total_mass:      M_A(t)
mean_mass:       mean(m | A, t)
mass_distribution
element_totals
descriptor_distribution
```

## 第一版分类区间

类别编码不要同时用 `C5` 表示“5 个碳”和“5 个环”。第一版默认分别使用：

```text
carbon_bin = C0, C1-5, C6-20, C21-40, C40+
ring_bin   = R0, R1, R2-5, R6-10, R10+
oxygen_bin = O0, O1, O2, O3+
```

三个维度的组合构成泛粗粒度节点。节点可以显示为 `A`、`B`、`C`，但底层必须保存完整描述符。例如：

```text
C40+ + R1 + O0   -> A
C40+ + R10+ + O0 -> B
```

`radical_sites` 和 `molecular_mass` 作为节点内部属性保留，允许同一节点在反应前后发生质量和自由基状态变化。
