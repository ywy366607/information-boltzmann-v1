# Transolver 的 slice 坍缩 = G22c 的富人俱乐部：局部储库能否替代正交化（预注册，2026-10-26，实验前写定）

## 1. 对应

Transolver 的 `SliceRead`（`fine_grain/native_mot.py:115`）让每个点在 M 个 slice 上做 softmax 软分配，再做质量归一化池化，然后在 M 个 slice 之间做注意力，最后经 Deslice 写回点场。按体系：

| Transolver | 体系 |
|---|---|
| 点在 slice 上的 softmax | R1/R2 单项扇区 |
| 质量归一化 Σwx/Σw | 质量归一化读出 |
| M 个 slice 间的注意力，O(M²) | P1a：在有限的槽间交互 |
| `use_null_slice`（空槽 ∅） | 排出通道（G16） |
| **slice 坍缩**：少数 slice 吸走全部质量，或分配趋于均匀 | **G22c `shared_potential_forces_nesting`**：所有隔室共享一个势时，强连接必然嵌套成富人俱乐部 |

本仓库现有的防坍缩手段是几何的：Newton–Schulz Stiefel 正交化（`use_stiefel`）、自适应温度、G ≤ C 的秩约束。已有诊断（`results/published/collapse_temperature_sweep.json`、`collapse_sites.json`）显示，温度为 0 时出现 δ 坍缩；上下文的有效秩为 4/32，slice 的有效秩为 2/32。

G22c 的逆否给出另一种、由公理推出的手段：**每个 slice 一个局部储库**。slice 已经吸收的质量越多，它对下一份质量的势越低：logit_m ← logit_m + log r_m，其中 r_m 随该 slice 的已分配质量耗竭。在单次前向中迭代几轮，就是带容量约束的分配。

**已有工作**：这在单次前向中等价于 Sinkhorn / 最优传输式的均衡分配，MoE 路由中已有（BASE layers、S-BASE 等均衡路由）。所以这里是体系推出了一个已知有效的技巧，并不是新发明；要检验的是"它能否在 Transolver 上替代正交化"，以及坍缩与 G22c 的对应是否定量成立。

## 2. 实验（待锁存实验完成后实现）

- 任务：仓库现有的合成任务 `glyph` 与 `lines`（`scripts/train_benchmark.py`），使用 SliceNet 的 slice 读出。
- 四个臂：(a) 基线，不做正交化；(b) Stiefel 正交化；(c) 局部储库，3 轮迭代，不做正交化；(d) 局部储库 + Stiefel。
- 指标：`collapse_stats` 中的 PR_mass、H_mass、H_point，slice 的有效秩，任务准确率。按 [最小 | 中位 | 均值 | 最大] 报告。初步验证用 1 个种子。

## 3. 预言

| 编号 | 预言 |
|---|---|
| S1 | (a) 出现坍缩：PR_mass < M/4，或 slice 有效秩 < M/4 |
| S2 | (c) 的 PR_mass ≥ M/2，并且不低于 (b)：局部储库与正交化一样能防坍缩 |
| S3 | (c) 的任务准确率不低于 (b) 减 1 个百分点 |
| S4（G22c 的定量形式） | 在 (a) 中，slice 按吸收质量排序后呈嵌套结构：质量大的 slice 所服务的点集，包含质量小的 slice 所服务点集的大部分（包含率 > 0.5）；(c) 中这一嵌套消失 |

**失败的出口**：若 (c) 均匀坍缩（H_mass → 1，所有 slice 的内容趋同），说明储库只解决了"少数吞并"的一侧，另一侧（内容同质化）需要的是 G21 意义上的分化来源（涨落、边界、身份），那正是正交化在做的事。这时结论应是"两者分工"，而不是"替代"。
