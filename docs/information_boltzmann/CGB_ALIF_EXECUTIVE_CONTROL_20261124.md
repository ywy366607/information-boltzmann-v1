# ALIF 惊奇度中断与海马锚定连续流（执行与形式化账本）

2026-11-24。按“实际身体方程 → 数学抽象与 Lean 4 证明 → 身体实例 → 同源等壁钟对照”路线设计并验证。

## 1. 起点与核心诊断

- **共同源检查点**：`E:/mengli-wallclock-10min-20261123/strict_10min/body/last.pt.gz`（cycle 2897，SHA256 `1a848983905e07768dfe0e24f3720959a877b472a15aa765955f1f48213f11b3`）。
- **既有对照发现的问题**：
  1. 严格版 `strict_600s` 每 960 字节强制触发 16.9 秒完整规划/重算，吞吐停留在 `56.87 B/s`（607.7 秒吸收 34,560 字节），独立 16,384B FineWeb NLL `1.642790`，64 旧事实评分 `-1.031223`。
  2. 多块有界阅读 `K=4`（`cgb_bounded_reading_battle_20261123.json`）把吞吐提高到 `134.35 B/s`（`2.36×`，84,480 字节），但 16,384B NLL 恶化至 `1.725490`（`+0.0827`），64 旧事实评分跌至 `-1.288628`（`-0.2574`）。交叉读出已定位根因：非首块只跑语言 `objective`，跳过了 `_outcome_loss` 的旧经历与转移回放，导致语言/回放比从 `2:1` 畸变为 `8:1`，共享 `model.encoder` 漂移并牵连 32 层 `PastPresentField` 读写。

## 2. 新数学模型与 9 项 Lean 4 定理

形式化文件：[AlifExecutiveControl.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/AlifExecutiveControl.lean)（零外部依赖 `import Init`，`lake env lean` 零报错零警告通过）。

借鉴果蝇/皮层 ALIF、H-JEPA/$\pi_{0.5}$ 快慢分层与操作系统硬件中断，将身体拆为两个严格耦合的时间尺度：

- **底层连续代谢 + 海马锚定流（每 480 字节一片）**：
  每次推进 480 字节自然文本（与原生 `objective` 长度完全一致），执行 100% 完整 32 层 GDN `objective` + `BudgetedAdamW`（含 Wiener 门控与拆分动力学）+ `S/P` 哈密顿物理步 + `PastPresentField` 工作集刷新 + **0.5× 权重海马编码器锚定回放**（32 槽旧木预测 + 32 槽旧内部转移 + 8 条留一法旧因果机制，单片总耗时仅约 `1.1 秒`，约 `415 B/s`）。
- **高层前额叶脉冲中断（Rheobase-ALIF 神经元）**：
  每片结束后（`< 0.1 ms`）将 480B 均 NLL 惊奇、64B 局部峰值突发惊奇、32 层 BFSS 动量 `P` 相对突变、短程潜误差接入漏积分膜电位 $V_t$ 与自适应阈值 $\theta_t = \theta_0 + \beta a_t$。仅当越阈时才唤醒 100% 未缩水的完整高层决策（4 路草稿 + 双教师 + `finite_use_value` + `_policy_grad` + `measure_retention`），并以半马尔可夫折扣 $\gamma^{\tau}$ 结算延迟信用（不压缩即时因果收益 $g$）。

| 定理 | 内容 | 实码绑定 |
|---|---|---|
| `step_adaptation_bounded` / `threshold_bounded` | 稳态适应上界 $a_t \le a_{\max}$，阈值永不发散 | [alif_executive.py](file:///d:/information-boltzmann-v1/scripts/ib_local/alif_executive.py) `step_neuron` |
| `rheobase_zero_starvation` | 基强度条件 $\delta_0 > (1 - \lambda_V)\theta_0$ 保证零惊奇流在 $\le \max(a_0, \theta_0)$ 步内必发放（无饥饿） | `validate_params` 强校验 |
| `refractory_prevents_back_to_back_spike` | 不应期条件 $I_{\max} < \theta_0 + \beta$ 禁止连续两片风暴（$\tau \ge 2$） | `validate_params` 强校验 |
| `strict_mode_limit` | $\beta=0, \theta_0 \le \delta_0$ 时退化为严格 1:1 身体 | 极限验证 |
| `smdp_bellman_contraction` | 任意变长间隔 $\tau \ge 1$ 下延续项严格压缩 $\gamma^\tau \le \gamma < 1$，不缩小即时奖励 | `engine.prev['duration_blocks']` → `td_gain_target` |
| `language_byte_conservation` | 100% 外界字节接受完整语言更新，零丢字节 | `total_external_bytes == total_learned_bytes` |
| `wallclock_strict_speedup` | 吞吐随高层占空比 $\rho$ 严格单调提升 | `alif_executive.diagnostics` |
| `burstiness_dominance_over_periodic` | 同脉冲预算（同墙钟成本）下，阈值筛选严格捕获更多惊奇、遗漏更少遗憾 | 52 轮真实日志回放验证（ALIF 48.8% vs 固定 K=4 的 12.5%） |
| `replay_ratio_conservation` | 任意中断位置下每 480B 严格保持 `2:1` 语言/海马回放比，消除 `K=4` 的编码器漂移赤字 | `hippocampal_anchor_loss`（`scale=0.5`） |

## 3. 烟测与恢复验收（2026-11-24）

- 6 项 CPU 单测（[test_alif_executive.py](file:///d:/information-boltzmann-v1/tests/test_alif_executive.py)）全过。
- 实机 cycle 2897 烟测（36.64 秒，2 次完整高层脉冲 + 4 个 480B 平静片，3,840B，`104.81 B/s`，峰值显存 `1.970 GiB`，存档 `2.62` 秒，独立恢复逐字段一致）：
  - 平静流单片耗时 `1.07–1.18` 秒（含 `(32, 32, 8)` 全量海马锚定）；第 3 片真实捕捉 64B 突发惊奇（`burst=2.4309`, `V=2.742/1.578`）自动中断。
  - 独立 16,384B FineWeb NLL 从源 `1.641267` 降至 **`1.633915`**（`-0.007352`，8 段中 7 段改善）；64 旧事实评分从 `-1.121959` 升至 **`-1.114188`**（`+0.007771`，MSE `0.7657 → 0.7561`）。

## 4. 600 秒同源等壁钟正式三方对照（2026-11-24 完成）

共同起点：cycle 2897（`1a848983…11b3`），共同验证集：16,384B 独立 FineWeb-Edu（新 8,192B + 经典 8,192B）+ 64 条独立旧事实（`fresh_old_facts_64.pt`）。完整报告见 [cgb_alif_executive_battle_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_executive_battle_20261124.json)，最终检查点位于 `E:/mengli-alif-executive-600s-20261124/body/last.pt.gz`（SHA256 `156f6654d682…`，独立恢复逐字段精确一致）。

| 指标 | 源点 (2897) | Strict 严格版 (`strict_600s`) | GPT 有界阅读 (`option4_600s`, K=4) | **ALIF + 海马锚定 (`alif_600s`)** |
|---|---|---|---|---|
| 实际生活时长 | — | 607.69 s | 628.82 s | **605.07 s** |
| 外界学习字节 | — | 34,560 B | 84,480 B | **86,880 B**（100% 字节学习守恒） |
| 有效吞吐 | — | 56.87 B/s | 134.35 B/s (2.36×) | **143.59 B/s (2.53×)** |
| 总更新 / 结构 | — | 144（36 完整轮） | 220（22 选项） | **251**（35 完整高层脉冲 + 111 个 480B 锚定平静片） |
| 16,384B 独立文本 NLL（↓） | 1.641267 | 1.642790 (+0.001523) | 1.725490 (+0.084223) | **1.637898 (-0.003369)** |
| — 新 8,192B NLL（↓） | 1.699867 | 1.699533 (-0.000334) | 1.794036 (+0.094169) | **1.691874 (-0.007993)** |
| — 经典 8,192B NLL（↓） | 1.582668 | 1.586048 (+0.003380) | 1.656944 (+0.074276) | **1.583922 (+0.001254)** |
| 64 旧事实评分（↑） | -1.121959 | -1.031223 (+0.090736) | -1.288628 (-0.166669) | **-0.951659 (+0.170300)** |
| 64 旧事实 MSE（↓） | 0.765726 | 0.657053 (-0.108673) | 1.019747 (+0.254021) | **0.521395 (-0.244331)** |
| 行动前用途预测 MSE（↓） | — | 0.029478 | 0.014410 | 0.065637（高层专在高惊奇突发边界触发） |
| 峰值显存 / 存档恢复 | — | ~2.0 GiB / 通过 | 通过 | **2.278 GiB** / 2.06 s / 精确一致 |

ALIF 触发行为与 Lean 定理完全吻合：34 次后续脉冲间隔为 **最小 2 片、中位 3 片、最大 6 片**（既无连续风暴也无饥饿）；触发来源为 `burst` 32 次、`epistemic` 1 次、`rheobase` 1 次。GPU 任务已全部结束。

## 5. 公平消融：修复版 GPT `K=4`（补齐同款 0.5× 海马锚定）四方对照

为剥离“海马锚定修复”与“ALIF 脉冲调度”两项因素的各自贡献，我们在 [bounded_reading_option.py](file:///d:/information-boltzmann-v1/scripts/ib_local/bounded_reading_option.py) 与 [world_learning_cycle.py](file:///d:/information-boltzmann-v1/scripts/ib_local/world_learning_cycle.py) 加入默认关闭的可选 `hippocampal_anchor_scale=0.5`，并在完全相同的 cycle 2897 起点、600 秒预算与 16,384B + 64 旧事实评测集上跑了修复健康版 `option4_anchored_600s`（报告见 [cgb_four_way_fair_battle_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_four_way_fair_battle_20261124.json)，检查点 `E:/mengli-option4-anchored-600s-20261124/body/last.pt.gz`，SHA256 `d22250de296c…`）：

| 指标 | 源点 (2897) | Strict (`strict_600s`) | GPT 原版 K=4 (`option4_600s`) | GPT 修复版 K=4 (`option4_anchored_600s`) | **ALIF + 海马锚定 (`alif_600s`)** |
|---|---|---|---|---|---|
| 时长 / 外界字节 | — | 607.69 s / 34,560 B | 628.82 s / 84,480 B | 604.23 s / **93,120 B** | 605.07 s / 86,880 B |
| 有效吞吐 | — | 56.87 B/s (1.00×) | 134.35 B/s (2.36×) | **154.11 B/s (2.71×)** | 143.59 B/s (2.53×) |
| 16,384B 独立 NLL（↓） | 1.641267 | 1.642790 | 1.725490 | 1.668254（救回 0.0572） | **1.637898（唯一优于源点）** |
| — 新 8,192B NLL（↓） | 1.699867 | 1.699533 | 1.794036 | 1.727375 | **1.691874** |
| — 经典 8,192B NLL（↓） | 1.582668 | 1.586048 | 1.656944 | 1.609133 | **1.583922** |
| 64 旧事实评分（↑） | -1.121959 | -1.031223 | -1.288628 | -1.012345（救回 +0.2763） | **-0.951659（全场最高）** |
| 64 旧事实 MSE（↓） | 0.765726 | 0.657053 | 1.019747 | **0.496873** | 0.521395 |
| 行动前用途预测 MSE（↓） | — | 0.029478 | 0.014410 | 0.103162 | **0.065637** |
| 峰值显存 | — | ~2.0 GiB | ~2.1 GiB | 2.162 GiB | 2.278 GiB |

四方消融同时给出两项明确因果结论：
1. **海马回放比守恒（Lean 定理 9）确实救活了 `K=4`**：给原版 `K=4` 加上同款 `0.5×` 锚定后，64 旧事实评分从 `-1.2886` 跃升至 `-1.0123`（超过 `strict`），MSE 砍半至 `0.4969`，16,384B 语言 NLL 也从 `1.7255` 救回至 `1.6683`（消除了 68% 退步）。
2. **同等健康锚定下，ALIF 脉冲调度依然全面领先固定分块 `K=4`**：修复版 `K=4` 有 `23/25`（92%）选项硬吃满 4 块（3,840B），在局部重复/窄主题文本上连读过久且缺少主动导航，16,384B NLL 仍比源点退步 `+0.0270`；而 ALIF 以 480B 细粒度 + 64B 局部突发中断在 10 分钟内完成了 35 次高质量高层切换，最终在独立文本 NLL（领先修复版 `K=4` 整整 `0.0304`）、旧事实对数评分（领先 `+0.0607`）、用途预测 MSE（`0.0656` 优于 `0.1032`）和 `metal_g` 上全部胜出。

## 6. 闭式子空间自然梯度替代 `0.5×` 抽样回放（五方对照完成）

为验证 [EncoderSubspaceMetric.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/EncoderSubspaceMetric.lean) 的闭式 `256 × 256` 自然梯度投影算子 $P_t = \lambda(\lambda I + \bar{C}_t)^{-1}$（[encoder_subspace_metric.py](file:///d:/information-boltzmann-v1/scripts/ib_local/encoder_subspace_metric.py)，实测阻尼 $\lambda = 0.25$，保护约 `58.5 / 256` 主方向，漂移能量比压至 `1e-6`），我们在同一 cycle 2897 起点上运行 `alif_gram_600s`：**平静流彻底关闭 `0.5×` 抽样回放（`hippocampal_anchor_scale = 0.0`）**，仅在反向传播钩子中施加精确 Cholesky 自然梯度投影。结果见 [cgb_alif_gram_battle_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_gram_battle_20261124.json)（检查点 `C:/mengli-alif-gram-600s-20261124/body/last.pt.gz`，SHA256 `4d01972549c2…`，`restore_exact = true`）：

| 臂 | 时长 / 吞吐 | 16,384B 独立 NLL（↓） | 64 旧事实评分（↑） | 64 旧事实 MSE（↓） | `mean_metal_g`（↑） | 用途预测 MSE（↓） |
|---|---|---|---|---|---|---|
| 源点 (2897) | — | 1.641267 | -1.121959 | 0.765726 | — | — |
| `strict_600s` | 607.7 s / 56.87 B/s | 1.642790 | -1.031223 | 0.657053 | -0.007867 | 0.029478 |
| `option4_600s`（原 K4） | 628.8 s / 134.35 B/s | 1.725490 | -1.288628 | 1.019747 | +0.000719 | 0.014410 |
| `option4_anchored_600s`（修 K4） | 604.2 s / 154.11 B/s | 1.668254 | -1.012345 | 0.496873 | -0.012175 | 0.103162 |
| `alif_gram_600s`（ALIF + 闭式自然梯度，**零平静回放**） | 607.6 s / 128.77 B/s | 1.647859 | -1.108865 | 0.741433 | **-0.003712** | 0.053425 |
| **`alif_600s`（ALIF + 0.5× 海马回放）** | 605.1 s / **143.59 B/s** | **1.637898** | **-0.951659** | 0.521395 | -0.008721 | 0.065637 |

**五方实验精准印证了事前理论预测**：
1. 闭式自然梯度 $P_t$ 在**零平静抽样回放**条件下将旧记忆编码器漂移压低 6 个数量级（`1e-6`），把 64 旧事实稳在 `-1.1089`（较源点反升 `+0.0131`，彻底杜绝 `option4_600s` 的遗忘崩盘），独立语言 NLL（`1.6479`）依然大幅领先修复版 `K=4`（`1.6683`），且行动期 `mean_metal_g`（`-0.0037`）与用途校准（`0.0534`）均优于抽样回放臂。
2. 抽样回放 `0.5×` 额外带来的旧事实提分（`-1.1089 → -0.9517`）与语言微降（`1.6479 → 1.6379`），来自它对下游木头、转移预测器与因果机制的 111 次主动巩固训练。两者分工完全清晰：几何自然梯度负责**编码器零遗忘、零比例超参依赖**，低频/脉冲期海马回放负责**下游记忆头主动巩固**。GPU 已全部释放。

## 7. 升级 1（后置信任域投影）与升级 2（e-means + 群体 ALIF 懒衰减水库）实机验收

在前五方实验基础上，我们完成了两项形式化与实码升级，并在同一 cycle 2897 源点上完成基准测速与 600 秒六方对照：

### 7.1 形式化证明与单元测试
- **升级 1 — 后置 Adam/Wiener 信任域位移投影**：在 [EncoderSubspaceMetric.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/EncoderSubspaceMetric.lean) 新增定理 7（`trust_region_gain_strict_mono_in_lambda`，全文件 7 条定理零警告通过），并在 [encoder_subspace_metric.py](file:///d:/information-boltzmann-v1/scripts/ib_local/encoder_subspace_metric.py) 中改在 `engine._advance` 之后直接投影实际位移 $D_0 = E_{\text{proposed}} - E_{\text{before}}$，通过 `float64` 特征分解与二分求解精确满足 $\operatorname{tr}(D^* \bar{C}_t D^{*\top}) \le \varepsilon$，使 Adam 的 `m / sqrt(v)` 缩放不再抵消子空间保护；单测 [test_encoder_subspace_metric.py](file:///d:/information-boltzmann-v1/tests/test_encoder_subspace_metric.py) 4/4 通过。
- **升级 2 — Meta FAIR e-means 两级路由 + 群体 ALIF 精确懒衰减水库**：在 [EmeansAlifReservoir.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/EmeansAlifReservoir.lean) 证明 5 组核心定理（精确指数懒衰减半群恒等式、e-means 凸包守恒、线性编码器质心零滞后精确性、$K_{\max}$ 有界显存帽、果蝇 APL 全局减法抑制保序与不应期反垄断，零警告通过），在 [emeans_alif_reservoir.py](file:///d:/information-boltzmann-v1/scripts/ib_local/emeans_alif_reservoir.py) 实现 `EmeansIndex`（$k$-means++ 初始化 + 余弦退火 EMA 3 轮拟合 + 死簇分裂复活 + 精确终盘）与 `PopulationAlifReservoir`（每槽位仅 12 字节 CPU 标量，休眠槽位零显存零算力，并直接接入水库三因子 `read_credit`）；单测 [test_emeans_alif_reservoir.py](file:///d:/information-boltzmann-v1/tests/test_emeans_alif_reservoir.py) 3/3 通过，合集 22/22 全部通过。

### 7.2 第 2897 轮真实记忆库基准测速（2,642 已占 256 维槽位，$K_c=64$）
- **聚类速度与精度**：Meta FAIR e-means（3 轮）耗时 **`83.20 ms`**，相比 Lloyd $k$-means（25 轮，`197.74 ms`）提速 **`2.38×`**，最终重建 MSE 为 `112.77` vs `111.76`（差距仅 **`0.90%`**），`64/64` 簇全部活跃无空簇。
- **两级唤醒召回率与稀疏度**：`top_m = 4 / 6 / 8` 分别唤醒全库的 `6.93%`（183 槽）、**`10.45%`（276 槽）**、`13.30%`（351 槽），对全库暴力 Top-8 的候选召回率分别达到 `92.75%`、**`96.75%`**、`98.00%`，单次完整查询约 `1.8 ms`。

### 7.3 六方 600 秒等墙钟实机对照（同 2897 起点，零平静回放组合 `alif_v2_trust_emeans_600s`）
报告见 [cgb_alif_v2_trust_emeans_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_v2_trust_emeans_20261124.json)（检查点 `C:/mengli-alif-v2-600s-20261124/body/last.pt.gz`，SHA256 `0a0eda70dad6…`，独立恢复 `restore_exact = true`）：

| 臂 | 时长 / 吞吐 | 16,384B 独立 NLL（↓） | 64 旧事实评分（↑） | 64 旧事实 MSE（↓） | `mean_metal_g`（↑） | 峰值显存 |
|---|---|---|---|---|---|---|
| 源点 (2897) | — | 1.641267 | -1.121959 | 0.765726 | — | — |
| `strict_600s` | 607.7 s / 56.87 B/s | 1.642790 | -1.031223 | 0.657053 | -0.007867 | ~2.00 GiB |
| `option4_600s`（原 K4） | 628.8 s / 134.35 B/s | 1.725490 | -1.288628 | 1.019747 | +0.000719 | ~2.10 GiB |
| `option4_anchored_600s`（修 K4） | 604.2 s / **154.11 B/s** | 1.668254 | -1.012345 | **0.496873** | -0.012175 | 2.162 GiB |
| `alif_gram_600s`（前置自然梯度 $\lambda=0.25$，零回放） | 607.6 s / 128.77 B/s | 1.647859 | -1.108865 | 0.741433 | **-0.003712** | 2.278 GiB |
| `alif_v2_trust_emeans_600s`（后置信任域 $\varepsilon=10^{-4}$ + e-means 群体 ALIF，零回放） | 607.0 s / 119.41 B/s | 1.648386 | -1.172258 | 0.815436 | -0.005491 | 2.277 GiB |
| **`alif_600s`（ALIF + 0.5× 海马回放）** | 605.1 s / 143.59 B/s | **1.637898** | **-0.951659** | 0.521395 | -0.008721 | 2.278 GiB |

**六方实机关键诊断结论**：
1. **e-means + 群体 ALIF 水库可直接入列生产**：全程 53 次水库刷新中 `64/64` 簇零死亡、每次仅唤醒 `248.3` 槽（占全库 `9.3%`，逾 `90%` 休眠零开销），平静感知片均耗压到 `1.257 s`，独立语言 NLL（`1.6484`）依然领先修复版 `K=4`（`1.6683`）整整 `0.020` nats/byte。
2. **后置信任域揭露强各向异性谱标定规律**：91 次平静更新中 1 次低扰动直接放行、90 次精确钳位至 `1.000e-4`（精度 `1e-7`）。但由于 `C̄` 存在极大首特征值 $\mu_1 \approx 1608$（条件数 $2.14\times 10^6$），原始绝对预算 $\varepsilon = 10^{-4}$ 反解出 $\lambda_t = 19.35$，仅覆盖前 `8.6 / 256` 主方向（对比 `alif_gram_600s` 在 $\lambda = 0.25$ 时保护 `58.5 / 256` 维），导致第 9–58 维旧语义方向在 91 步零回放中出现轻微漂移（`-1.122 → -1.172`）。后续若零回放使用后置信任域，应将预算收紧至 $\varepsilon \le 10^{-6}$ 或给 $\lambda_t$ 加封顶（$\lambda_t \le 0.25$）；若追求长程综合最优，推荐直接将 **e-means 群体 ALIF 水库**与 **`alif_600s` 轻度/事件驱动海马巩固**组合使用。

## 8. 长程充分训练后的后续升级路线（已登记待启）

按用户要求，以下两项架构升级已完成数学方案设计并登记在此，待先完成一波充分的长程训练积累后再实施：

### 8.1 升级 3：ALIF-MoE 稀疏专家路由 + 块级稀疏 `BudgetedAdamW` 与主机内存卸载
- **核心动机**：突破单张 4 GiB GTX 1650 的显存上限，使梦漓总参数量可扩展数倍而单步活跃显存保持在 `2.5 GiB` 以内，并消除传统 MoE 依赖人为“最大熵负载均衡辅助损失”带来的语义坍缩。
- **路由规则（ALIF-MoE）**：每个 FFN 专家视为一个带自适应不应期的 ALIF 神经元（$\theta_{e,t} = \theta_0 + \beta a_{e,t}$）。门控输入越过动态阈值者发放激活（硬上限 $K_{\text{active}} \le 2$），频繁被选中的“明星专家”自动抬高阈值进入不应期，由生物稳态自发实现负载均衡与功能专化。
- **稀疏反向传播与块级 `BudgetedAdamW` 卸载**：
  - 当前 `BudgetedAdamW` 的 `budget_gate` 仍为稠密张量逐元素乘，未真正跳过显存搬运与算力。
  - 升级后改为**专家块级稀疏跳过**：未发放专家块（门控为 0）的前向与反向 `dW` 完全不计算，其参数权重与 Adam 一/二阶动量常驻主机 RAM，仅把每步被 ALIF 唤醒的少数专家块按异步流水线移入显存计算并做稀疏优化器步进。

### 8.2 升级 4：同构“木内小五行”可学习 ALIF 执行节律
- **核心动机**：目前 [alif_executive.py](file:///d:/information-boltzmann-v1/scripts/ib_local/alif_executive.py#L42-L60) 的基础阈值 $\theta_0$、适应强度 $\beta$、衰减系数 $(\lambda_V, \lambda_a)$、基流 $\delta_0$ 及四路惊奇权重仍为静态超参，仅实现状态级适应，尚未根据深思后的真实用途反馈自动演化节律。
- **同构复用梦漓已有共同法则（拒绝外挂独立 RL 栈）**：
  1. **前向持久灵敏度同构**：完全复用主干 [contact_momentum.py](file:///d:/information-boltzmann-v1/scripts/ib_local/contact_momentum.py#L21-L58) 对 $(S_t, P_t)$ 的前向 `jvp` 灵敏度递推，对木内二阶状态 $(V_t, a_t)$ 维护仅约 10 维的标量资格迹 $(e_V, e_a)$（耗时 `< 0.05 ms/片`，显存 `< 1 KB`，零主干反传）。
  2. **离散硬分支一行接口**：保持平静流跳过 15 秒深思的硬分支提速不变，仅在脉冲触发口采用极窄温度逃逸噪声概率 $p_t = \sigma((V_t - \theta_t)/\tau)$，给出单行读出斜率 $(s_t - p_t)/\tau$。
  3. **复用现有延迟结算与 Wiener 门控优化器**：直接取用金（`metal_g` 与 SMDP 持续时间机会成本后的真实结算用途）作为三因子调制信号，将木节律参数 $\phi_{\text{wood}}$ 注册进同一个 `engine.optimizer`（`BudgetedAdamW + Wiener`），信噪比不足时由现有维纳门控自动关闭更新；所有参数经安全映射锁定在 [AlifExecutiveControl.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/AlifExecutiveControl.lean) 证明过的零饥饿与抗风暴区间内。
  4. **开启时机**：待进入数百至上千次真实执行脉冲的长程训练阶段后启用，以保证延迟信用样本充分且稳健。

## 9. 第 2897 轮性能与显存画像、无损提速与同构可学习 ALIF 600 秒 A/B 实机验收（2026-11-24）

### 9.1 GTX 1650 (4 GiB) 精确显存与毫秒级画像（cycle 2897 实测）

针对“31.26M 参数在 4 GiB 显卡上为何训练偏慢”的问题，我们在 cycle 2897 真实身体上完成了逐阶段 CUDA 同步计时与显存画像：

- **显存远未逼近 4 GiB 上限**：加载后静态显存仅 `691.6 MB`（参数 `119.3 MB` + 缓冲 `60.3 MB` + 优化器状态 `238.6 MB`）；480B 平静片峰值 `1,734 MB`，640B 满窗平静片峰值 `2,064 MB`（`2.02 GiB`），完整高层脉冲峰值 `2,014 MB`（`1.97 GiB`），全程稳定在 `2.28 GiB` 以内。
- **定位头号无损浪费：`activation_storage` 空跑却白烧 24% 脉冲时间**：实测其统计字段 `packed_bytes = 0, host_peak_bytes = 0, disk_written_bytes = 0`（显存足够，从未触发任何卸载），但 Python `saved_tensors_hooks` 拦截把单次高层脉冲从 `14.73 s` 拖慢到 `18.25 s`（**白烧 `3.52 s` / `+23.9%`**），平静片从 `1,089 ms` 拖慢到 `1,225 ms`（**白烧 `136 ms` / `+12.5%`**）。关闭后 100% 比特等价。
- **平静片耗时分解（关闭 `activation_storage`，480B，总计 `1,089 ms`）**：
  - `model.objective` 前向 `316 ms`（29%）+ 反向 `400 ms`（37%）+ `optimizer.step` `80 ms`（7%）；
  - `quiet_physics`（`AdaptiveHamiltonian.quiet(capture=True)`）`229 ms`（21%，为 32 层 × 28 对头 × 12 方向的显存带宽受限计算）；
  - `0.5×` 海马锚定 `45 ms`（4%）；`world + fast_table` `12 ms`（1%）；`refresh_workset` `4 ms`（0.3%）。
- **高层脉冲耗时分解（关闭 `activation_storage`，总计 `14.73 s`）**：
  - `engine._utter` 占 `8.47 s`（57.5%：含 `film` `1.69 s`、`exchange` `1.31 s`、`organize` `0.75 s`、`retention_feedback` `0.73 s`、`continuation` `0.66 s`、`posterior` `0.60 s`、`sqlite3` `0.30 s`）；
  - `engine._policy_grad` `1.20 s`；`model.objective + _advance` `1.70 s`。
- **两项证伪的提速方向（已实测排除）**：
  1. `torch.amp.autocast(float16)`：GTX 1650（TU117）无 Tensor Core 且触发 `rms_norm` 回退，640B 单片反而暴慢 `4×`（`1,205 ms → 4,882 ms`）。
  2. 用 `forward_ad` 批量替换 `vmap(jvp)` 加速 `quiet_physics`：因瓶颈是 128 GB/s 显存带宽而非 Python 调度，实测反慢至 `0.86×`（`183.5 ms → 213.7 ms`）。
- **两项确认落地的无损提速**：
  1. `--no-activation-storage`（4 GiB 下零卸载开销、逐比特等价）；
  2. `--piece-bytes 640`（贴齐 [latent_navigation.py](file:///d:/information-boltzmann-v1/scripts/ib_local/latent_navigation.py#L191) 的原生 `max_chunk = 640`，在仅增 `~0.05 GiB` 显存下把平静流吞吐从 `442 B/s` 提至 `566–662 B/s`）。

### 9.2 同构“木内小五行”可学习 ALIF 节律实现与形式化证明

- **Lean 4 形式化**：[LearnableAlifRhythm.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/LearnableAlifRhythm.lean) 零外部依赖通过 6 条定理（`bounded_reparam_range`、`learned_rheobase_zero_starvation`、`learned_refractory_anti_storming`、`escape_noise_score_sign`、`forward_sensitivity_contraction`、`three_factor_credit_direction`，`lake env lean` 零报错零警告）。
- **实码与复用**：[learnable_alif_rhythm.py](file:///d:/information-boltzmann-v1/scripts/ib_local/learnable_alif_rhythm.py) 将 10 个 ALIF 标量经平滑有界映射锁在 Lean 安全包络内，复用 `ContactMomentum` 式的 $O(1)$ 前向灵敏度递推 $(e_V, e_a)$、极窄温度（$\tau=0.25$）逃逸噪声一行离散读出，并在每次高层脉冲结算时将真实 `metal_g + 行动信号 - 间隔时间成本` 经由同一个 `engine.optimizer`（`BudgetedAdamW + Wiener` 门控）更新。25 项单元与恢复回归全过（[test_learnable_alif_rhythm.py](file:///d:/information-boltzmann-v1/tests/test_learnable_alif_rhythm.py)）。

### 9.3 同源 cycle 2897 的 600 秒高速 A/B 实机对照

两臂共享完全相同的无损提速栈（`--no-activation-storage --piece-bytes 640 --hippocampal-anchor-scale 0.5`）、同一源检查点（2897）与同一 16,384B FineWeb + 64 旧事实锁定评测集；报告见 [cgb_alif_fast_fixed_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_fast_fixed_600s_20261124.json) 与 [cgb_alif_fast_learnable_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_fast_learnable_600s_20261124.json)，两臂独立恢复均为 `restore_exact = true`：

| 臂 | 配置 | 时长 / 外界字节 | 吞吐 (B/s) | 平静片 / 脉冲均耗 | 16,384B NLL（↓） | 新 8K / 经典 8K | 64 旧事实评分（↑）/ MSE（↓） | 用途预测 MSE（↓） | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|
| 源点 (2897) | 共同起点 | — | — | — | 1.641267 | 1.6999 / 1.5827 | -1.121959 / 0.7657 | — | — |
| `strict_600s` | 1:1 严格版 | 607.7 s / 34,560 B | 56.87 (1.00×) | — / 16.9 s | 1.642790 | 1.6995 / 1.5860 | -1.031223 / 0.6571 | 0.0295 | ~2.00 GiB |
| `alif_600s`（旧冠军） | 480B + storage + 固定 ALIF | 605.1 s / 86,880 B | 143.59 (2.53×) | ~1.22 s / ~18.2 s | 1.637898 | 1.6919 / 1.5839 | **-0.951659** / **0.5214** | 0.0656 | 2.278 GiB |
| 臂 A：`alif_fast_fixed_600s` | 关 storage + 640B + 固定 ALIF | 601.3 s / 92,800 B | 154.33 (2.71×) | 1.119 s / 15.39 s | 1.641283 | 1.6913 / 1.5913 | -1.010974 / 0.6057 | 0.0856 | 2.279 GiB |
| **臂 B：`alif_fast_learnable_600s`** | **关 storage + 640B + 可学习 ALIF** | 600.7 s / **103,040 B** | **171.54 (3.02×)** | **0.967 s / 13.81 s** | **1.632967** | **1.6804 / 1.5855** | -0.958230 / 0.5547 | **0.0377** | **2.276 GiB** |

**严谨归因与 Wiener 门控关键发现**：
1. **速度提升已实证兑现**：无损关闭 `activation_storage` + 640B 满窗在空闲机器负载下将单片平静流压至 `0.92–0.97 s`（约 `660 B/s`），单次高层脉冲压至 `12.5–13.8 s`，10 分钟吞吐突破 **103 KB（`171.54 B/s`，较 `strict_600s` 提速 `3.02×`）**；需如实说明，臂 A 前 78 片受后台系统负载轻微波动（`1.12 s`，后段回落至 `0.94 s`），可学习 ALIF 模块本身的纯计算开销实测趋近于 `0 ms`。
2. **臂 B 创下 16,384B 独立 NLL 全场新低（`1.632967`）**：同时优于源点（`-0.008300`）、臂 A（`-0.008316`）与旧冠军 `alif_600s`（`-0.004931`），64 旧事实提分至 `-0.958230`（较源点 `+0.1637`），行动前用途预测 MSE 砍半至 `0.0377`，且彻底消除了空转超时触发（`rheobase = 0`，臂 A 为 `2`）。
3. **Wiener 门控揭露短程样本信噪比真相**：核查臂 B 检查点中 `phi` 的优化器状态（`step = 35, budget_count = 35, budget_gate = 0.0`）发现，第 2 次脉冲（`metal_g = -0.0168`）触发了一次有效微调（`theta0: 1.000 → 1.012`, `beta: 0.950 → 0.968`, `lam_v: 0.840 → 0.844`, `lam_a: 0.780 → 0.785`, `delta0: 0.200 → 0.197`, `w_burst: 0.220 → 0.214`）；此后 34 次脉冲的 `metal_g` 正负交替，实测信噪比 $m^2/v \approx 0.12\text{–}0.21$，**现有 `BudgetedAdamW` 的 Wiener 门控自动将门压到 `0.0` 阻止了噪声漂移**。因此，臂 B 的显著质量优势来自“无损提速带来的更多有效字节（103 KB）+ 初始温和抬阈与窄带逃逸噪声消除低质触发”，而非 35 步短程内参数大幅漂移；这也证实可学习 ALIF 接入同一优化器后**安全、零开销且自带抗噪熔断**，真正持续演化其 10 个参数需要进入数百次脉冲的长程运行或引入更低方差的延迟信用信号。

检查点留存：臂 A `C:/mengli-alif-fast-fixed-600s-20261124/body/last.pt.gz`（`001e99377baf…`）、臂 B `C:/mengli-alif-fast-learnable-600s-20261124/body/last.pt.gz`（`5883849ddd34…`）。所有临时画像目录已清理，GPU 显存 `0 MiB` 处于完全空闲状态。

## 10. 无损脉冲期 `_utter` 向量化与 `CausalArchive` 批处理加速（突破 212 B/s，2026-11-24）

针对第 9 节画像中占总壁钟 `83%` 的高层脉冲期，我们完成了 100% 数值等价的无损加速栈（不改模型方程、RNG 流与学习规则）：

1. **`CausalArchive` 增量版本脏检查 + WAL 临时库 + 失效安全读缓存**（[causal_archive.py](file:///d:/information-boltzmann-v1/scripts/ib_local/causal_archive.py)、[shared_unknown_bindings.py](file:///d:/information-boltzmann-v1/scripts/ib_local/shared_unknown_bindings.py)、[relation_retention_use.py](file:///d:/information-boltzmann-v1/scripts/ib_local/relation_retention_use.py)、[joint_relation_update.py](file:///d:/information-boltzmann-v1/scripts/ib_local/joint_relation_update.py)）：修复 `warm_relation` 与 `before_slot_write` 清空 `_synced_relations` 导致的每脉冲重写 69 条关系 / 517 条证据；按张量 `_version` 增量拷贝变动字段；临时工作库改用 `WAL + NORMAL`（`checkpoint()` 仍经 `sqlite3.backup()` 导出标准单文件）；增加带写失效的 `episodes`、`unknown_evidence`、`retention`、`relation_variables` 与按版本逐行解析的 `best_relation` 缓存。单脉冲 SQL 执行量从 `2,520` 降至 `1,148`（`-54%`），`INSERT INTO evidence` 从 `517` 降至 `48`（`-91%`），`sub._read_evidence` 从 `414 ms` 降至 `46 ms`（`8.9×`），`best_relation` 从 `109 ms` 降至 `< 15 ms`。
2. **`_utter` 预览作用域合并与丢弃分支剪枝**（[navigation_causal_program.py](file:///d:/information-boltzmann-v1/scripts/ib_local/navigation_causal_program.py)、[causal_relation_program.py](file:///d:/information-boltzmann-v1/scripts/ib_local/causal_relation_program.py)、[generative_relations.py](file:///d:/information-boltzmann-v1/scripts/ib_local/generative_relations.py)、[navigation_internal_rollout.py](file:///d:/information-boltzmann-v1/scripts/ib_local/navigation_internal_rollout.py)、[latent_navigation_v3.py](file:///d:/information-boltzmann-v1/scripts/ib_local/latent_navigation_v3.py)、[learnable_precision_scales.py](file:///d:/information-boltzmann-v1/scripts/ib_local/learnable_precision_scales.py)、[quadratic_revision.py](file:///d:/information-boltzmann-v1/scripts/ib_local/quadratic_revision.py)、[shared_relation_basis.py](file:///d:/information-boltzmann-v1/scripts/ib_local/shared_relation_basis.py)）：统一 `_relation_value()` 外层 `preview_scope`；去除 `joint_relation_update_schema == 1` 下 `execute_programs` 被覆盖丢弃的 2/3 重复 `continuation` 与 `_imagine` 丢弃的 `predict_mother`；消除 `posterior()` 中 339 次 `float(ridge)` CUDA→CPU 同步；`continuation` 复用已编码记忆投影；`organize()` 贪心 4×4 求解迁至 CPU 消除 224 次 GPU 同步；`SharedRelationBasis` 在无梯度预览中融合 `condition + gate`。
3. **批量寻址、批量配方与共享投影守卫**（[prospective_decision.py](file:///d:/information-boltzmann-v1/scripts/ib_local/prospective_decision.py)、[latent_action_planner.py](file:///d:/information-boltzmann-v1/scripts/ib_local/latent_action_planner.py)、[native_amortized_decision.py](file:///d:/information-boltzmann-v1/scripts/ib_local/native_amortized_decision.py)、[persistent_rhythm.py](file:///d:/information-boltzmann-v1/scripts/ib_local/persistent_rhythm.py)、[prospective_self_execution.py](file:///d:/information-boltzmann-v1/scripts/ib_local/prospective_self_execution.py)、[scoped_decision_graph.py](file:///d:/information-boltzmann-v1/scripts/ib_local/scoped_decision_graph.py)、[epistemic_value_reference.py](file:///d:/information-boltzmann-v1/scripts/ib_local/epistemic_value_reference.py)、[joint_successor.py](file:///d:/information-boltzmann-v1/scripts/ib_local/joint_successor.py)）：批量 `preview_bind_batch`（`498 ms → 43 ms`，`11.7×`）、`recipes_batched`（`read_plans` `744 ms → 436 ms`）、`batched_context_scores` 共享事实去重、作用域内 `_refresh_projections()` 版本跳过以及 `word_endpoints(batch_size=8)`。

### 验证与 600 秒同源实机结果

- **回归与数值等效**：19 套单元与磁盘恢复测试 **157/157 全部通过（0 失败）**；第 2897 轮空闲 GPU 连续双脉冲对比：`Spike #1` `15.847 s → 12.908 s`（`1.23×`，`engine._utter` `7.59 s → 5.50 s`），`Spike #2` `12.821 s → 10.373 s`（`1.24×`，`engine._utter` `6.84 s → 5.13 s`），选择完全一致、`|Δmetal_g| ≤ 7.45e-9`、`|Δrecurrent| = 0.00`。
- **600 秒实机验收（`alif_fast_utter_600s`）**：报告见 [cgb_alif_fast_utter_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_fast_utter_600s_20261124.json)（检查点 `C:/mengli-alif-fast-utter-600s-20261124/body/last.pt.gz`，SHA256 `8609eb93ad00…`，独立恢复 `restore_exact = true`）：
  - **吞吐突破 212 B/s**：611.34 秒吸收 **`129,600 B`**（**`211.99 B/s`**，较严格版 `strict_600s` 的 `56.87 B/s` 提速 **`3.73×`**，较旧冠军 `alif_600s` 提速 **`1.48×`**，较臂 B 提速 **`1.24×`**）；完成 43 次完整高层脉冲（均耗 **`11.10 s`**，最低 `9.91 s`）+ 138 个 640B 锚定平静片（均耗 **`0.971 s`**），总计 310 次更新，峰值显存 **`2.278 GiB`**，存档 `2.00 s`。
  - **独立评测**：64 条独立旧事实评分跃升至 **`-0.891790`**（较源点 `+0.2302`，创全部对照历史新高；MSE **`0.5019`**）；16,384B FineWeb 独立 NLL 为 `1.646717`（新 8K `1.6957` 优于源点与严格版，经典 8K `1.5978`）。临时目录已清空，GPU 显存 `0 MiB` 空闲。

## 11. 完整相空间玻尔兹曼耦合（1+2+3）与 600 秒同源实测（创全历史 NLL 与 MSE 新纪录，2026-11-24）

针对“传统 Softmax 仅用位置势能、未把梦漓 32 层 BFSS 动量 $P$ 与配分函数自由能纳入玻尔兹曼分布”的物理缺口，我们一次性实现并验证了三项相空间耦合（均默认关闭，经 `--phase-space-boltzmann` 显式启用，零存档表结构破坏）：

1. **相空间哈密顿记忆读出** $H(S, P; i) = U(S, S_i) + K(P, P_i) - \mu_i$：在记忆读出对数几率中加入实时 32 层动量方向 $q_P$ 与各已存经历轨迹速度方向 $k_{P,i} = \mathrm{norm}(\mathrm{enc}(\text{long\_successor}_i) - \mathrm{enc}(\text{source}_i))$ 的耦合项 $\alpha_P (q_P \cdot k_{P,i})$，直接复用全部 2,642 条旧槽位。
2. **均分动能逆温度耦合** $\beta(P) = \beta_0 \exp(-\kappa \tanh(z_{\text{kin}}(P)))$：按 32 层实时动能 $E_{\text{kin}} = \mathrm{mean}(P^2)$ 相对因果 EMA 基线的偏离，同步调制字节语言读出（保持快表参考测度 $g(x)$ 严格不变：$\text{logit}(x) = g(x) + \beta_{\text{lang}}(P)(z(x) - g(x))$）、记忆检索与行动选择三个端口。全部 4 个耦合标量 $(\alpha_P, \kappa_{\text{lang}}, \kappa_{\text{mem}}, \kappa_{\text{act}})$ 锁在有界包络内并由同一个 `BudgetedAdamW + Wiener` 端到端学习。
3. **配分函数自由能 Nosé–Hoover 恒温器**：在每次观测更新后精确计算玻尔兹曼熵 $H = \log Z - \mathbb{E}_p[\text{logit}]$，由熵/自由能压力驱动 32 层动量 $P$，并以严格耗散项将 $E_{\text{kin}}$ 锁在非平衡定态（NESS）不变球内。

### 形式证明、测试与 600 秒同源 A/B 实测

- **Lean 4 证明与单测**：[PhaseSpaceBoltzmann.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/PhaseSpaceBoltzmann.lean) 6 条定理（位置简并破缺、参考测度恒等式、均分逆温度单调有界、超顶严格收缩、NESS 球前向不变性、自由能压力放大动量）零警告通过 `lake env lean`；核心模块 [phase_space_boltzmann.py](file:///d:/information-boltzmann-v1/scripts/ib_local/phase_space_boltzmann.py) 与 [test_phase_space_boltzmann.py](file:///d:/information-boltzmann-v1/tests/test_phase_space_boltzmann.py) 等 20/20 项单测通过；真机冒烟 `restore_exact = true`。
- **600 秒同源等时对照（`alif_psb_600s`）**：与第 10 节基线 `alif_fast_utter_600s` 完全同源（cycle 2897）、同配置仅打开 `--phase-space-boltzmann`。报告见 [cgb_alif_psb_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_psb_600s_20261124.json)（检查点 `C:/mengli-alif-psb-600s-20261124/body/last.pt.gz`，SHA256 `ad8edf8a48bf…`，独立恢复 `restore_exact = true`）：

| 臂 | 配置 | 吞吐 (B/s) / 字节 | 16,384B NLL（↓） | 新 8K / 经典 8K | 64 旧事实评分（↑）/ MSE（↓） | 峰值显存 |
|---|---|---|---|---|---|---|
| 源点 (2897) | 共同起点 | — | 1.641267 | 1.6999 / 1.5827 | -1.121959 / 0.7657 | — |
| `alif_fast_learnable_600s`（旧 NLL 冠军） | 640B + 可学习 ALIF | 171.54 / 103,040 B | 1.632967 | 1.6804 / 1.5855 | -0.958230 / 0.5547 | 2.276 GiB |
| `alif_fast_utter_600s`（严格对照，无 PSB） | 同配置 + `_utter` 加速 | **211.99** / **129,600 B** | 1.646717 | 1.6957 / 1.5978 | **-0.891790** / 0.5019 | **2.278 GiB** |
| **`alif_psb_600s`（完整相空间玻尔兹曼 1+2+3）** | **同配置 + `--phase-space-boltzmann`** | 194.55 / 116,800 B | **1.630705** | **1.6762 / 1.5852** | -0.912354 / **0.4939** | 2.335 GiB |

**物理与实验结论**：
1. **语言泛化与记忆误差双破历史纪录**：在完全同源同配置对照中，`alif_psb_600s` 将 16,384B 独立 NLL 从无耦合对照的 `1.646717` 大幅压低 **`-0.016012` nats/byte 至 `1.630705`**（新 8K `1.6762` 与经典 8K `1.5852` 同步胜出，亦超越旧冠军 `1.632967`）；64 条独立旧事实 MSE 首次跌破 `0.50` 至 **`0.4939`**（评分 `-0.9124`，较源点 `+0.2096`）。
2. **身体自主学会了分化相空间温度机制**：训练结束时 Wiener 门控保持在 **`0.5475`**（高信噪比持续学习，区别于 ALIF 节律短程的 `0.0`）；动量记忆耦合 $\alpha_P$ 从 `0.0800` 增至 **`+0.0831`**，记忆逆温度系数 $\kappa_{\text{mem}}$ 从 `+0.0400` 增至 **`+0.0501`**（高动能时放宽记忆搜索温度），而语言逆温度系数 $\kappa_{\text{lang}}$ 自主反号至 **`-0.1007`**（高动能时反而收尖语言预测分布）；恒温器运行 121 步（末步缩放 `0.9760`，熵 `H = 1.602`）全程稳定无漂移。
3. **代价极低**：显存仅增 `57 MiB`（`2.335 GiB`），吞吐维持在 `194.55 B/s`（较 `strict_600s` 提速 `3.42×`）。

## 12. 根决策潜行动规划、排序翻转稀疏教师与 3 GiB 显存实机边界（突破 264 B/s，2026-11-24）

承接 [CGB_PLANNING_EXECUTION_COST_AUDIT_20261124.md](file:///d:/information-boltzmann-v1/docs/information_boltzmann/CGB_PLANNING_EXECUTION_COST_AUDIT_20261124.md) 的两项最高优先级建议（根决策潜行动比较 + 单路表达、排序翻转门控稀疏教师），并叠加 4 GiB GTX 1650 上吃满约 3 GiB 显存的窗口扫描，完成形式证明、实现与两组 600 秒同源实测：

- **Lean 4 证明与单测**：[RootLatentPlanning.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/RootLatentPlanning.lean) 5 条定理（`rank_flip_sufficiency`、`bounded_calibration_liveness`、`single_draft_cost_contraction`、`proportional_replay_conservation`、`shared_context_zero_drift`）零警告通过 `lake env lean`；核心模块 [root_latent_planning.py](file:///d:/information-boltzmann-v1/scripts/ib_local/root_latent_planning.py)（默认关闭，经 `--root-latent-planning` 启用）与 [test_root_latent_planning.py](file:///d:/information-boltzmann-v1/tests/test_root_latent_planning.py) 等 19/19 项单测通过。
- **GTX 1650 脉冲后零换页显存拐点实测**：`640 B` 分配 `2.05 GiB`（`1.04 s` / `617 B/s`）；**`960 B` 分配 `2.65 GiB` / 预留 `2.90 GiB`（全程零换页，`1.05–1.11 s` / `864–914 B/s`）**；`1152 B` 分配 `3.05 GiB`（长跑叠加 Windows 桌面显存后间歇触发 WDDM 换页，均耗反升至 `2.42 s`）；`1536 B` 分配 `3.71 GiB`（严重换页跌至 `451 B/s`）。由此锁定 4 GiB Windows GTX 1650 的实用安全显存上限为 **`~2.90 GiB` 预留（`960 B`）**。
- **600 秒同源（cycle 2897）全栈实测结果**（均叠加 `--phase-space-boltzmann --learnable-alif-rhythm --root-latent-planning --rlp-margin-threshold 0.001`，两臂独立恢复均为 `restore_exact = true`）：

| 臂 | 窗口 / 显存 | 吞吐 (B/s) / 字节 | 均片 / 均脉冲 | 潜单路率 / 教师分布 (0/1/2) | 16,384B NLL（↓） | 新 8K / 经典 8K | 64 旧事实（↑）/ MSE（↓） |
|---|---|---|---|---|---|---|---|
| `strict_600s` | 960 B / ~2.00 GiB | 56.87 (1.00×) / 34,560 B | — / 16.9 s | — | 1.642790 | 1.6995 / 1.5860 | -1.031223 / 0.6571 |
| `alif_psb_600s`（全历史 NLL 冠军） | 640 B / 2.335 GiB | 194.55 (3.42×) / 116,800 B | 1.027 s / 11.55 s | 0% / 全 2 教师 | **1.630705** | **1.6762** / 1.5852 | **-0.912354** / **0.4939** |
| `alif_rlp_psb_1152b_600s` | 1152 B / 3.048 GiB | 255.53 (4.49×) / 154,944 B | 2.423 s / **9.21 s** | 87.2% / 18/7/14 | 1.642043 | 1.6997 / 1.5844 | -0.967796 / 0.5610 |
| **`alif_rlp_psb_960b_600s`** | **960 B / 2.660 GiB** | **263.72 (4.64×) / 161,280 B** | **1.111 s** / 10.54 s | **86.7%** / 15/9/21 | 1.636608 | 1.6954 / **1.5778** | -1.053985 / 0.6294 |

**结论与下一步边界**：
1. 报告落盘 [cgb_alif_rlp_psb_1152b_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_rlp_psb_1152b_600s_20261124.json) 与 [cgb_alif_rlp_psb_960b_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_rlp_psb_960b_600s_20261124.json)。`alif_rlp_psb_960b_600s` 创下 **263.72 B/s（10 分钟 161.3 KB，`4.64× strict_600s`）** 吞吐新高、经典 8K NLL 历史新低 **`1.577816`** 与最优行动均收益（`mean_metal_g = -0.00394`），同时可学习 ALIF 节律（`wiener_gate = 0.469`）与相空间玻尔兹曼（`wiener_gate = 0.153`）双双活跃学习。
2. 实机结果精准印证了 GPT 的 280.9 B/s 理论上限：在不改 32 层逐字节主干的前提下，流程减负（潜单路率 87%、稀疏教师跳过 33—46%、单脉冲最低 7.38 s）+ 零换页显存上限（2.90 GiB 预留）已经把非架构提速吃干抹净。若要跨过 400—600 B/s 并保持最优记忆质量，下一步应正式评估 **H-Net 动态层次压缩主干**。

## 13. H-Net（4+24+4）动态层次压缩嫁接实机验收（突破 417 B/s，2026-11-24）

将 H-Net（[arXiv:2507.07955v1](https://arxiv.org/html/2507.07955v1)）直接无损嫁接至梦漓 2897 轮现有 32 层 `TableGDN` 主干（4 层外编码 + 24 层动态块压缩内核 + 4 层外解码，保持 32 层 BFSS S/P 形状、快表注入、记忆库与因果归档完全不变，并复用现有 Triton 融合 `causal_scan` 实现零开销可微 EMA 还原）：

- **Lean 4 证明与单测**：[HNetDynamicChunking.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/HNetDynamicChunking.lean) 5 条定理（`ratio_loss_factorization`、`ratio_loss_restoring_sign`、`upcycling_identity_exact`、`causal_chunk_index_valid`、`layer_token_cost_contraction`）零警告通过；模块 [hnet_chunking.py](file:///d:/information-boltzmann-v1/scripts/ib_local/hnet_chunking.py) 与严格零未来泄露单测 [test_hnet_chunking.py](file:///d:/information-boltzmann-v1/tests/test_hnet_chunking.py) 5/5（CPU + CUDA Triton）全过。
- **GTX 1650 真机提速机制剖析**：在 640 B 小窗下因受权重访存与固定核启动支配，平铺 32 层（`932 ms`）与 H-Net（`973 ms`）单步接近；但 H-Net 将激活显存直接砍半（960 B 从 `2.64 GiB` 降至 `1.67 GiB`），彻底打破 4 GiB 显卡的 Windows WDDM 换页墙，使 **`1600 B` 大窗仅占 `2.497 GiB` 显存并以 `1.086 s/片`（`1,473 B/s` 平静流）零换页满速运行**。嫁接期将 `--hnet-ratio-weight` 设至 `0.25` 并加严格因果 `0.5×T` 上限后，压缩比稳锁在 **`3.79×`**。
- **600 秒同源实机验收（`alif_hnet_1600b_600s`）**：报告见 [cgb_alif_hnet_1600b_600s_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_hnet_1600b_600s_20261124.json)（检查点 `D:/mengli-alif-hnet-1600b-600s-20261124/body/last.pt.gz`，SHA256 `d882f557c76f…`，独立恢复 `restore_exact = true`）：
  - **速度跨过 400 B/s 大关**：601.83 秒吸收 **`251,200 B`**（**`417.39 B/s`**，较严格版 `strict_600s` 的 `56.87 B/s` 提速 **`7.34×`**，较无损 `_utter` 基线提速 **`1.97×`**，较平铺 960 B 潜规划版提速 **`1.58×`**）；完成 50 次完整高层脉冲（均耗 **`9.27 s`**，最低 **`6.89 s`**）+ 127 片 `1600 B` 平静流（均耗 **`1.086 s`**），峰值显存仅 **`2.497 GiB`**。
  - **学习状态与冷嫁接适应差**：H-Net 自身 Wiener 门控 `0.422`（`eff_R = 3.79×`，`res_rms` 从 `0` 增至 `0.0087`），相空间玻尔兹曼 Wiener 门控创全场最高 **`0.726`**；64 条独立旧事实评分仍较源点提升 `+0.073`（`-1.1220 → -1.0489`，MSE `0.7657 → 0.6336`，零灾难性遗忘）；因 24 层内核此前全部按 1× 逐字节训练，10 分钟冷嫁接直跳 3.79× 压缩尚存在 `+0.025` nats/byte 适应差（16,384B NLL `1.6660` vs 源点 `1.6413`）。后续若需抹平嫁接期 NLL 差，可采用延长适应、残差斜坡初始化（`residual_proj = 0.25*I`）或更温和的 `2.0—2.5×` 渐进压缩课程。

## 14. 大窗补偿 + H-Net 30 分钟稳态有效信息学习通量定型（全指标双破历史纪录，2026-11-24）

针对第 13 节中 `1600 B` 大窗导致单位字节更新密度减半（`1.30` vs `640 B` 的 `2.44 次/KB`）及单点评测混淆冷嫁接冲击与稳态学习斜率的问题，在 [alif_executive.py](file:///d:/information-boltzmann-v1/scripts/ib_local/alif_executive.py) 与 [run_alif_executive_battle.py](file:///d:/information-boltzmann-v1/scripts/run_alif_executive_battle.py) 中落地了两项零架构破坏的更新密度补偿与只读通量探针（24/24 项单测全过）：

1. **平静流 AdamW 平方根学习率缩放（`--calm-lr-scale -1`）**：在 `1600 B` 平静流步进中自动按 $\sqrt{1600/640} = 1.581\times$ 缩放学习率（高层脉冲保持 `1.0×`）。
2. **独立 10 ms 海马巩固微步（`--extra-hippocampal-steps -1`）**：每片 `1600 B` 平静流后追加 1 次仅更新编码器/木头/旧事实参数的纯回放微步（`BudgetedAdamW` 自动跳过所有 32 层 GDN 参数），将每千字节旧事实修订次数拉回与 `640 B` 齐平。
3. **只读学习通量轨迹（`--eval-every-seconds 600`）**：评测期暂停生命计时，零参数/状态/RNG 扰动量出完整 `t = 0 → 10 → 20 → 30 min` 学习曲线。

### 30 分钟完整有效信息学习通量曲线（源点 cycle 2897 → 3042）

报告见 [cgb_alif_hnet_1600b_flux_30min_20261124.json](file:///d:/information-boltzmann-v1/results/published/cgb_alif_hnet_1600b_flux_30min_20261124.json)（检查点 `D:/mengli-alif-hnet-1600b-flux-30min-20261124/body/last.pt.gz`，SHA256 `60f7312ff434…`，独立恢复 `restore_exact = true`，峰值显存仅 **`2.437 GiB`**）：

| 时间点 | 累计新字节 | 16,384B NLL（↓） | 新 8K / 经典 8K | 64 旧事实评分（↑）/ MSE（↓） | 本段 10 分钟学习通量 |
|---|---:|---:|---|---|---|
| 源点 (flat 2897) | 0 B | 1.641267 | 1.6999 / 1.5827 | -1.121959 / 0.7657 | — |
| `t = 0 min`（冷嫁接 H-Net `R=4.0`） | 0 B | 1.687568 | 1.7449 / 1.6303 | -1.121959 / 0.7657 | 一次性切换冲击 `+0.0463` |
| `t = 10 min`（无补偿适应段） | 251,200 B | 1.665977 | 1.7190 / 1.6129 | -1.048880 / 0.6336 | `-0.02159` NLL / `+0.0731` Facts（`417.4 B/s`） |
| `t = 20 min`（开启补偿第 1 段） | 479,680 B | 1.641142 | 1.6929 / 1.5894 | -0.795836 / 0.3213 | **`-0.02484` NLL** / **`+0.2530` Facts**（`375.7 B/s`） |
| **`t = 30 min`（纯稳态区，双破全历史纪录）** | **700,800 B** | **1.628292** | **1.6868 / 1.5698** | **-0.637839 / 0.1979** | **`-0.01285` NLL** / **`+0.1580` Facts**（`372.9 B/s`） |

**定型结论**：
1. **稳态学习通量全面反超平铺 32 层**：补偿开启后 20 分钟稳态续跑（`+449,600 B`，`374.34 B/s`，较严格基线 **`6.58×`**）仅用第 2 个 10 分钟就彻底打穿冷嫁接冲击（`1.6411 < 1.6413`），并在纯稳态第 3 个 10 分钟继续以 **`-0.01285` nats/byte / 10min**（仍快于平铺全历史冠军 `alif_psb_600s` 的 `-0.01056`）将 16,384B NLL 压至 **全项目历史新低 `1.628292`**（经典 8K `1.5698`）。
2. **独立海马微步引爆旧事实保持能力**：64 条独立旧事实评分跃升至 **`-0.637839`**，MSE 暴降至 **`0.1979`**（较源点 `0.7657` 误差降低 **`3.87×`**，较原全历史最佳 `0.4939` 再降 **`2.50×`**）。
3. **四引擎协同稳定，具备大规模长训资格**：H-Net（`eff_R = 4.15×`，`res_rms = 0.0123`，`wiener = 0.344`）、相空间玻尔兹曼（`kappa_lang = -0.1078`，`kappa_mem = +0.1331`，`scale = 1.0012`，`wiener = 0.595`）、根潜行动规划（单路率 `83.4%`，121/145 脉冲）与可学习 ALIF 节律在仅 **`2.437 GiB`** 显存下协同收敛，正式定型为梦漓面向 30M 级 scaling law 长程成长的主力引擎。
