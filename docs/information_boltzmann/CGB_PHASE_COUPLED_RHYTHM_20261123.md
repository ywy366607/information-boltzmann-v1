# 事件驱动＋脑波相位耦合多速率节律（2026-11-23）

## 1. 背景与动机

在完成全部严格 1:1 加速栈（单字节五核融合、撤回合批、目标 logits 复用、右填充策略梯度合批、同步消除、`_retain_constraints` 提核、ISA-L 压缩与 E 盘 spool 加载）后，32 层 30.5M 身体的单轮严格等价耗时已从最初 `32.21 秒/轮` 降到 `14.39 秒/轮`（`strict8_baseline`，8 轮 `115.115 秒`）。但真正吸收语言和更新权重（`body.objective`＋反向传播＋`BudgetedAdamW`）每轮只占约 `2.2 秒`，其余约 `12 秒` 消耗在强制每 960 字节锁步执行一次的全套慢思考（四提案草稿、多假设 `imagine`、双教师 `flush_teachers`、轮内两次 `_poll_inside` 与全套 8 分支撤回重放）。

生物大脑并不采用固定 `每 N 轮` 计数器，也不在每个音节都停下来做慢思考和离线海马重放。本模块以真实身体可观测量驱动耦合脑波相位与巩固压力，引入显式开关 `--phase-coupled-rhythm`（默认关闭，不改变旧存档与严格 1:1 路径）。

## 2. 数学与生物机制

核心实现见 [phase_coupled_rhythm.py](file:///d:/information-boltzmann-v1/scripts/ib_local/phase_coupled_rhythm.py)，形式化证明见 [PhaseCoupledEventRhythm.lean](file:///d:/information-boltzmann-v1/lean/CGBAxioms/reviews/PhaseCoupledEventRhythm.lean)（7 条定理全部零报错通过）。

每轮结束后，利用真实 `bfss_momenta` 动能 $K_t$ 与真实 `response_nll` 标准化意外度 $z_t = \max(0, (e_t - \mu_{e,t-1})/\sqrt{v_{e,t-1}+\epsilon})$ 推进两个耦合状态量：

$$\Delta\phi_t = \omega_{\min} + \omega_{\text{body}}\tanh(K_t) + \omega_{\text{err}}\tanh(z_t), \qquad \omega_{\min} = \frac{2\pi}{5} > 0$$

$$\psi_{t+1} = \begin{cases} 0 & \text{本轮触发 SWR 巩固} \\ \min\left(1,\; \psi_t + \alpha_{\min} + \alpha_{\text{err}}\tanh(z_t)\right) & \text{否则} \end{cases}, \qquad \alpha_{\min} = 0.34 > 0$$

- **0 步延迟抢占**：当 $z_{t-1} > 1.25$（蓝斑–去甲肾上腺素/前扣带回意外抢占）时，下一轮立即打开 **Theta 深思＋海马 SWR 巩固**；当轮间边界自主选出连续 ≥2 步静默内省（剔除已完成的轮内静默步）时，立即打开 **Theta 深思**。
- **有界活性（永不饿死）**：Lean 证明 $\omega_{\min} > 0$ 与 $\alpha_{\min} > 0$ 保证即便在完全平稳的文本流中，Theta 窗最迟 5 轮必开、SWR 巩固最迟 3 轮必触发。
- **在线 Gamma 流畅通道（平静流、Theta 谷底）**：
  1. **H-JEPA 跨槽主题保持**：在有界视野（`max_theme_hold = 4`）内保持当前 `unknown_id`，跨局部关系槽刷新约束与编码坐标，省去每轮重复的 `question_choices` 树搜索；
  2. **聚焦单提案话语**：`proposal_budget = 1`，Lean 证明单元素 `log_softmax` 恒为 0，REINFORCE 得分完全无偏，同时省去 `_film` 内多余 `imagine()` 与 3/4 策略反传；
  3. **轻量化摊销估值**：单次 `compose`＋精确局部 `finite_use_value.expected`＋1 点 `persistent_rhythm.read_plans`，将双教师 `flush_teachers` 留给 Theta 窗；
  4. **无打断吸收**：关闭轮内两次冗余 `_poll_inside`，保留完整轮间 `autonomous_cycle.poll` 与准确资格迹；
  5. **海马 SWR 延迟巩固**：平静流不每半段跑 8 分支撤回重放，而是累积压力至 SWR 窗口或意外抢占时爆发。
- **不变核**：100% 真实语言吸收、32 层 GDN `objective`＋反向传播＋`BudgetedAdamW`、`S/P` 物理演化、精确字节快表、`causal_use_value` 结算、`shared_unknown.commit` 与延迟信用在每一轮完整运行。

## 3. 8 轮真实身体头对头实测（2813 → 2821）

同源 `original 2813`（`E:/mengli-native-supply-completed-20261122/original/last.pt.gz`），两臂均跑 8 轮 / 32 次优化更新 / 7680 外界字节 / 0 付费调用，完整报告见 [cgb_phase_coupled_rhythm_8cycles_20261123.json](file:///d:/information-boltzmann-v1/results/published/cgb_phase_coupled_rhythm_8cycles_20261123.json)。

| 指标 | 初始基线（2813，折算） | 严格 1:1 加速栈 (`strict8_baseline`) | 相位耦合节律 (`phase8_rhythm`) |
|---|---:|---:|---:|
| 8 轮纯生活计算时间 | ~257.71 秒 | 115.115 秒 | **72.686 秒** |
| 单轮平均耗时 | 32.214 秒/轮 | 14.389 秒/轮 | **9.086 秒/轮**（Gamma 最低 **5.939 秒**） |
| 外界字节吞吐 | 29.80 B/s | 66.72 B/s (**2.24×**) | **105.66 B/s（3.55× / 相对严格栈 1.584×）** |
| 末尾完整存档提交 | ~24.07 秒 | 2.427 秒 | 2.547 秒 |
| 峰值 CUDA 显存分配 | 2.156 GiB | 2.158 GiB | **1.983 GiB** |
| 64 旧事实均值变化 | 起点 -1.137590 | -1.132076 (**+0.005514**) | -1.133782 (**+0.003808**) |
| 独立 FineWeb 8192B NLL | 起点 1.583043 | 1.580249 (**+0.002794**) | 1.582251 (**+0.000792**) |

实机脑波轨迹显示：8 轮内自然演化出 6 轮 Gamma（`5.94—7.31 秒/轮`）、2 轮 Theta（`14.88—15.21 秒/轮`）、2 次海马 SWR 巩固与 6 次 H-JEPA 跨槽主题保持。第 2818 轮预测误差突跳至 `nll=2.1252`（`z=1.39 > 1.25`）并叠加 Theta 相位满圈，立即将第 2819 轮抢占进 Theta＋SWR 并把主题从 `unknown:1923` 切换为 `unknown:1924`，随后第 2820—2821 轮平滑回到 `6.19s / 5.94s` Gamma 流畅通道。独立加载恢复审计确认 `phase_rhythm_state` 与配置全部精确还原。

## 4. 使用边界与下一步

- 需要与旧架构严格逐字节等价对照时，使用 `strict8_baseline` 全套 1:1 加速开关（`14.39 秒/轮`、`2.43 秒` 存档）。
- 进行长程基础语言生长与多速率生物节律研究时，叠加 `--phase-coupled-rhythm`（`9.09 秒/轮`、`105.66 B/s`）。下一步建议在同等墙钟预算（如各自跑满 10 分钟或 32—64 轮）下对比两条路线在独立 FineWeb 8192 字节与 64 旧事实上的单位时间语言吸收收益。
