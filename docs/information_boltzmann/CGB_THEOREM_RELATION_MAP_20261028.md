# CGB 定理关系图（2026-10-28，Claude）

**用途。** 定理很多（`CGB_PROVEN_INDEX.md` 收 1139 条，100 个模块）。本文不重复定理内容，只回答三件事：

1. 根在哪里：哪些东西由我们手放进去；
2. 从根出发，在什么条件下推出什么（边上注明定理名）；
3. 哪些是本质（对每个生物都成立，写成准入门义务），哪些是现象（只在某些世界里出现，写成预注册的预言）。

原则（用户 2026-10-28）：只放根和世界条件，让模型自己生成其余部分；每一条"必须有"都要有必要性证明。

**检索顺序**：本文 → `CGB_PROVEN_INDEX.md`（Lean 全集，自动生成）→ `CGB_GENERATION_LAYER1_20261025.md`（G1–G25 的中文推导）→ `CGB_UNIFIED_REFERENCE_20261026.md`（语言生命体的实现接线）→ `CGB_HAND_DERIVATIONS.md`（未入 Lean 的纸面推导和数值见证）。新增机制前四处都要查。

---

## 0. 根（手放进去的）

| 根 | 内容 | Lean |
|---|---|---|
| 介质与六条公理 | 有限局部状态、局部接触、守恒（P2a）、时间反演对称（P2b）、边界感知（P3） | `Medium`、`Symmetry`、`AtomicIndependence`、`SubconditionGraph` |
| 能量不是基本量 | 可逆过程存在能量 E = −log π，决定每个事件与逆事件的速率比（T-E） | `JumpProcess.energy_of_reversible`、`Medium.energy_of_p2b` |
| 守恒来自对称 | Q 与 Q² 的期望都平稳 ⇔ 每个事件都保持 Q（Baez–Fong 诺特，T-N）；介质的守恒量是内容 | `JumpProcess.noether`、`NoetherBridge.p2a_iff_closed_*` |
| 有限 | 守恒 + 容量 ⇒ 状态空间有限 | `Medium.finite_reachable`、`Medium.finite_states` |
| 活着 | 可恢复的摄入（实际世界、实际未来） | `Life`、`OperationalLife`、`WorldRecovery.actualRecover_*` |
| 世界条件 | 边界上的驱动可以是任意的：雨、点击、妈妈（LLM）。它们不受公理约束 | `ConstraintFreedom.arbitrary_driven_tables_admitted`、`constant_protocol_can_be_driven` |

世界本身也是介质（`Nesting`：T-N1 毯子来自接触，T-N2 交换守恒，T-N3 冻结外部后内部仍细致平衡；`GenerationNesting` G26）。但最外层的边界驱动是自由的：妈妈是边界，不是生命体，不需要为说话付代价。

---

## 1. 公理直接推出的（所有介质）

- **一切不可逆只从边界进来**：G6/G6DB（`interior_detailed_balance`），内容的一切漂移来自跨边界事件（`content_drift_boundary`）。⇒ 能量（自由能）的来源只能是边界上与平衡值不同的势。
- **速率只经能量差依赖状态**：G1/G1E。⇒ 任何"选择"（选择性排出、选择性遗忘）只能由能量实现。
- **毯子**：`jump_blanket`、`blanket_of_axioms`、`blanket_of_contact`。
- **预测是快边界的 Gibbs 条件分布**：G13（`freezeSlow_conditional_balance`）。
- **瓶颈定理**：全接触时，P2c 迫使电导只依赖两端的身份页（`Models.xhop_pages`）。

---

## 2. 生命的必然（本质 → 准入门义务）

| 结论 | 定理 |
|---|---|
| 活着必须感知 | `Lives.exposed`；不感知就不能自我维持 `not_selfMaintains_of_unexposed` |
| 活着必须代谢（与世界交换内容） | `Lives.metabolizes`、`not_selfMaintains_of_closedMatter`、嵌套版 `Part.no_intake_of_closed` |
| 失去可恢复性不可逆 | `irrecoverable_absorbing`、`Part.irrecoverable_absorbing` |
| 没有可能的摄入就不可能恢复 | `recover_eq_zero`、`actualRecover_zero_without_intake`、`recovery_zero_without_intake` |
| **被动吸收者不算活着**：装满后吸收停止，在任何世界里都不活 | `RecoveryControls.passive_full_not_alive`、`passive_intake_not_persistence` |
| 有限供给养不起持续消耗 | `EnergyBudget.no_sustained_use_on_finite_supply` |
| 石头满足公理，但从来不活 | `Models.rock_cannot_eat` |
| 锁死即死、锁存即失明 | H1 `trap_irrecoverable`、H2 `trap_insensitive` |

准入门（`scripts/cgb_gate.py`）目前检查：感知、代谢、排出、选择性排出、记忆、闭环胜过致盲；种群另加分裂必需。

---

## 3. 有限个体 × 无限流 ⇒ 遗忘、选择、驱动（本质）

```
有限容量（P1a）× 不停摄入
 ├─ 遗忘被迫：累计排出 ≥ 累计摄入 − 容量（G16a export_forced）；只写不删会饿死（G16b）
 ├─ 不选择就全忘（G17a survive_le_pow）⇒ 保留必须选择（G17b retention_needs_selectivity）⇒ 选择只能靠能量（G1）
 ├─ 遗忘要耗散：擦除有 Landauer 下限（positive_erasure_requires_positive_dissipation）
 ├─ 平衡态记忆有上界（G24）、平衡态不能点燃（G23）、不存在精确锁存（hold_vanishes）
 │    ⇒ 要记得久，必须被驱动
 ├─ 线性记忆不能锁存、不能清理（GenerationLinearMemory）；任何线性记忆对长度 L 的突发都要承受总损伤 L（linear_burst_damage_integral）
 ├─ 长期占用守恒预算的条目会饿死别的（Continual.held_budget_starves）
 └─ 时间深度：单个两态池按几何速度遗忘（G20a）；几何排布的池组给出 1/t 痕迹（GenerationCascadeMemory）
```

---

## 4. 预测、记忆与学习

| 结论 | 定理 | 条件 |
|---|---|---|
| 随机世界里预测误差永不归零：流永不枯竭 | T-N7 `stream_never_dry` | 某状态有两个边界读数不同的后继 |
| 无记忆的两端只看到符号频率 | G19b | — |
| 序列结构需要两端有记忆 | G19c | — |
| 要按"落空次数"预测摄入，状态数 ≥ N+1 | G11b `intake_memory_capacity` | — |
| 精确记忆投影到规范的因果状态；均匀精度需要能区分"未来不同"的历史 | `CausalState.minimal`、`separated_histories_capacity` | — |
| 有限字母表推不出有限的精确记忆 | `CausalState.finite_alphabets_unbounded` | — |
| **可预测的观察不提供排除证据（不写入、不学习）** | `CausalLearning.agreement_no_write` | 有限候选模型 |
| 每次有效写入都花一份储备 | `CausalLearning.write_contract`、`paid_elimination` | — |
| 探索不可能是安全的 | `CausalLearning.no_safe_exploration` | — |
| 不变的世界里，本能胜过学习者 | L-1 | 世界平均不变 |
| 变化的世界里，学习者胜过任何本能 | L-2 | 两个不同的平均，学习成本 < 差距的 1/4 |
| 历史与未来无关时，历史无用 | L-3 | — |
| 世界稳定后，固定规则最优 | O-1 | — |
| **每个固定规则在任何时刻之后都会被超越 ⇔ 世界永不稳定** | O-2 `beaten_forever_iff_never_settles` | — |
| 变化的世界需要比较 | C-1 `no_fixed_leave_threshold` | — |
| 漂移禁止增益归零：V(t+1) = (1−K)²V + q + K²r | W2 | 漂移 q > 0 |
| 没有记忆的学习者多付 r²/(P+r) | `memoryless_gain_excess` | — |
| 罕见方向要按出现次数计时 | W3 | — |
| 学习会饱和（从一个来源学到的总量 ≤ 它规律的描述长度） | **待入库**（见 §8 N5） | — |

---

## 5. 没有输入时的活动：三种机制（可选，现象）

（`GenerationPersistence`，2026-10-27）

| 机制 | 必要条件 | 作用 |
|---|---|---|
| 主动放电（被驱动的默认活动） | 持续的循环流必须被驱动（A：`cycle_rates_balance`、`driven_cycle_not_balanced`） | 维持工作范围、可变性、时钟、默认行动 |
| 积分反馈参照（适应） | 不动点处活动回到设定值，缺口为零（B：`integral_feedback_adapts`、`adapted_gap_zero`） | 掌舵，不负责搜索 |
| 锁存 | 环路增益 ≥ 1（C：`latch_needs_unit_gain`）；增益 < 1 时只有一个静息态（`contraction_unique_rest`）；需要驱动（G23/G24）；要健康（H1/H2） | 保持决定、情境、目标；撑开收获窗口（见 §7 外部结果） |

相关：放大器 G25（平衡态 Hill 系数为 1，协同需要多份含量，增益与范围的权衡，固定阈值在宽世界里失败）；健康锁存的优化器形式 B1–B3（`deferLatch_holds/releases/senses`）。

**主动推理**不是一条机制。它是"有限身体 + 可恢复性 + 漂移的世界"推出来的闭环组织，第一级在 `Inference`（毯子与预见）。它的零件分别由上面三种机制和 §4 的记忆提供。

---

## 6. 行动与觅食（现象：由世界条件决定）

| 结论 | 定理 | 世界条件 |
|---|---|---|
| 在会枯竭的世界里，待着不动会饿死 | G26b `stay_starves` | 再生率 r < 消耗 c |
| 最优觅食需要落空计数 | G12 `miss_counter_necessary` | — |
| 移动是与占据者交换（空格不特殊） | `NestedWorld.move_exchanges` | — |
| 在不断变化的世界里，付费重开、试探、辨认、更新能撑过断粮 | `RenewableLearningLife.natural_survival` | 见证模型 |

**"只听不问"为什么不存在**（用户 2026-10-28）。四条已有定理加一条待证：
1. 可预测的输入不让你学到任何东西（`agreement_no_write`）；
2. 被动吸收者装满后就不再活着（`passive_full_not_alive`）；
3. 只要世界永不稳定，"只听"这个固定规则在任何时刻之后都会被超越（O-2）；
4. 不观察时（W2 中 K = 0），对隐藏状态的方差每拍增加 q；
5. 待证（N4）：于是一次观察的信息增益 ½·log(1 + (v₀ + qt)/r) 会超过任何固定的行动代价，行动必在有限时间内发生。

例外是睡觉和冬眠：q = 0，或者代价暂时高于增益。只要 q > 0，这段时间就有尽头。

---

## 7. 语言生命体：能量是什么

- **能量**：同一套 φ/μ（T-E：E = −log π）。概率分布本身就是能量地形：均值是最低点，方差是曲率的倒数；精度要用载体数量（材料、能量）来买（G25e 的增益–范围权衡是它的一种形式）。
- **来源**：只能是边界（G6DB），即语言流本身相对平衡的偏离。
- **两种耦合**（待定，见 §9）：
  - (a) 势差：边界单元与刺激设下的势交换内容。不匹配时才有净流。
  - (b) 低熵载体：把符号副本当燃料"用旧"（信息棘轮）。能预测时取出的功最多；要取时间关联里的功，内部状态必须镜像世界的隐藏状态。
- **外部结果**（Muruga、Ginot、Loos、Bechinger，arXiv:2603.06160，2026）：
  - 测到结果 m 后，可取的功 ≤ kT·KL(ρ_m‖ρ_平衡)；
  - 只测一次取不到隐藏记忆里的功：I((X,Y):M) = I(X:M)；
  - 两次测量的最优间隔等于隐藏自由度的时间尺度；
  - 在双势阱里让粒子"卡住"时，取出的功可以超过粒子本身储存的能量。

  这些是物理学结果，不是我们的定理；要用，先入 Lean（N1–N3）。
- **T1（`GenerationPredictiveHarvest`）的地位**：是一个声明的小生境契约（"猜中就有人付钱"），不是语言生命体的代谢。原因有二：它奖励可预测性本身，于是暗室最优；一句话也不是能量。数学结论（Kelly 上界等）照常成立。
- **已撤回的说法**（2026-10-28）："每个符号的负熵只能收获一次"。信息可以复制，代价是能量，载体守恒；一句话可以说很多遍。

---

## 8. 待证的边（入 Lean 前不做相应实现）

**2026-10-28 已证 N1–N5**（`GenerationInformationEnergy.lean`，完整构建与审计通过）：`freeEnergy_eq_kl`、`freeEnergy_at_law`、`kl_nonneg`（N1）；`snapshot_misses_hidden`（N2）；`finite_memory_bounds_net_work`（N3，代数层：Sagawa–Ueda 上界与 Landauer 擦除作为假设，G16a 给出擦除量）；`passivity_ends`（N4，直接调用 W2 `vanishing_gain_loses_under_drift`，增益取零）；`mixture_regret_le`（N5）。依赖 T1 的草稿文件已删除。剩 N6（可选）。

| 编号 | 命题 | 依赖（已有） |
|---|---|---|
| N1 | 不匹配就是自由能：对 Gibbs 分布 π，F(q) − F(π) = KL(q‖π) ≥ 0，当且仅当 q = π 时为零 | T-E、`gibbs_inequality` |
| N2 | 只测一次取不到隐藏记忆：给定 X 时 M 与 Y 条件独立 ⇒ I((X,Y):M) = I(X:M) | 有限概率 |
| N3 | 有限身体不能从平衡世界得到净收获 | G16a、Landauer 下限 |
| N4 | 不观察时不确定性增长，行动必在有限时间内发生 | W2（K = 0）、对数单调 |
| N5 | 学习会饱和：Bayes 混合的超额编码长度 ≤ log(1/w(θ₀)) | 已写，`GenerationLanguageForaging.lean` 的 `mixture_regret_le`，单文件编译通过，未入根导入。同文件中依赖 T1 的另三条要删除 |
| N6（可选） | 符号流取功上界与"必要多样性"（Boyd–Mandal–Crutchfield） | N1、N2 |

---

## 9. 未决的设计问题

1. 语言流按 (a)、(b) 中哪种方式喂身体，还是两者都要（感知走 (a)，取能走 (b)）。
2. 语言生命体的储备：统一参考里的 `OpenMedium`（有限环境库存）能否充当，还是需要一个像大肠杆菌 E 页那样的储备页。
3. 对妈妈只要求一条：她的回应要和孩子说的话有关（依赖性）。其余（叫名字、重复、长度）都是世界设计，属于现象。
