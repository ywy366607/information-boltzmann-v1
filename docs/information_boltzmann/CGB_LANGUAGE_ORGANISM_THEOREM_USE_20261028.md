# 语言生命体：到目前为止的决定，以及要用哪些定理（2026-10-28）

本文从全部 101 个模块、1262 条已证条目（`CGB_PROVEN_INDEX.md`）中，按语言生命体的部件逐个挑出：
- **用**：前提满足，直接成为设计约束；
- **相关**：给出方向，或是同一结构的另一个实例；
- **缺**：还需要新证。

关系总图见 `CGB_THEOREM_RELATION_MAP_20261028.md`，嘴的设计见 `CGB_LANGUAGE_MOUTH_DESIGN_20261028.md`，10-25 的接口规格见 `CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md`。

## 0. 已定的事（用户，2026-10-27/28）

| 事项 | 决定 |
|---|---|
| 方向 | 语言生命体：自己说话、自己提问，主动学习、主动压缩、吸收负熵 |
| 嘴 | 主动推理：说话要有相对基准的预期收益（锚定话题 = 留下，提问 = 离开）；不设说话的价格 |
| 妈妈 | 是 LLM，不要求她满足我们的公理；唯一要求是她的回应依赖于孩子说的话 |
| 自听 | 要。自我模型由此而来 |
| 信道容量 | 不外加。自听已让说话者受自己的容量约束 |
| 储备 | 不单设储备页。学到的模型就是储备（条件见 §3） |
| 代谢 | = 遗忘（分解）+ 学习（合成）；活着 = 学得比忘得快 |
| 数值来源 | 不用跨代搜索（太慢）。定理定范围，个体一生中的学习（G14 两相 / G15 三因子）定数值 |
| 语言 | 英文；输入输出是原生字节 0..255 |
| 主动放电 | 要：有一股不依赖说话的边界供养，驱动活动层里的循环（A）。GDN 没有输入时只有唯一静止点（`gated_delta_contracts` + `contraction_unique_rest`），锁存和自发活动都要放在活动层 h，不在突触层 S |
| 结构 | 64×64×64 = 64 个"柱"，每柱 64 个单元，柱内全连接（S_m 就是柱内突触，一个接触邻域）；柱与柱放在 3D 中，柱间稀疏连接，近邻为主，加少量长程。柱内用 S 的坐标表示单元身份，只有柱间才谈空间位置。读写头：键 k 按内容寻址（delta 的"先擦后写"就是 NTM 的 erase/add），路由 w 选柱。读写可以在同一个键上，因为时间顺序已经排除了"抄输入"的捷径：读在到达之前，写在到达之后。这与 10-08 的端口记录一致：允许重叠，口径有限（GDN 里就是 ‖k‖ = 1、β ≤ 2），分离端口只作诊断 |
| 约束 | 能在手机上跑：前向递推，无 BPTT，无 FP64 |

## 1. 架构草图（推导所得，待生成器表达力证明）

```
 妈妈的字节 ─┐                                        ┌─> 嘴（门控输出反应）─┐
             ├─> 耳 ─> 快页（预测：G13 弛豫） ─> 读出 ┤                     │
 自己的字节 ─┘          ↕ 慢页（记忆：级联池组）       └─ 学习进展 − 基准 ─┘
     ^                                                    （= 开口信号 = 第三因子）
     └────────────────────────── 自听 ───────────────────────────────┘
```

推导出的结构是一个有限状态、只向前推进的门控循环网络。`GenerationLinearMemory` 已证：Gated DeltaNet、Mamba、RWKV、带衰减的线性注意力都是同一形式：
- 门 α 就是非选择性排出，即 weight decay；
- delta 项是选择性排出；
- 写入是摄入。

所以这类网络是我们框架的一个实例。

**注（用户指出）**：GDN 的 α_t（Mamba 的 Δ_t 同理）是可学习且依赖输入的，每个 token、每个头一个值。所以它对内容不加选择（同一头里所有方向一起衰减），但对时间有选择（遇到话题切换时可以突然多忘）；对内容的选择交给 delta 项。

这正是 W2 + C-1 要求的形式：世界变了，就要多忘。另外，Mamba 把各头的 Δ 按对数均匀分布初始化，各头的时间常数呈几何排布，这就是 `GenerationCascadeMemory` 的几何池组。

但 α 可学并不能绕开 `histories_merge` 和 `linear_hold_keeps_error`：只要 α ≤ ᾱ < 1，记忆深度就有限；α → 1 时能长期保持，但误差也一起保持。

GDN 的 α 是离线用反向传播学的。我们不用 BPTT，但对线性递推，衰减参数的灵敏度可以只向前算：∂S_t/∂α = S_{t−1} + α·∂S_{t−1}/∂α，精确且便宜。但线性记忆不能锁存，也不能纠错（`linear_hold_keeps_error`），因此慢页至少要有一处非线性（吸引子或锁存）。

## 2. 按部件：用哪些定理

### A. 容器与边界
| | 定理 | 在这里说明什么 |
|---|---|---|
| 用 | `Interaction.Machine.conditioned_next`、`trace_prefix_consistent` | 开放交互的数学对象（10-25 规格已采用） |
| 用 | `Medium.blanket_of_axioms`、`Nesting.organism_blanket` | 身体有边界（马尔可夫毯） |
| 用 | `Medium.modulation_needs_boundary`、`rate_interior_stimulus_free` | 驱动只能来自边界：内部不能凭空产生学习信号 |
| 用 | `NoetherBridge.content_drift_boundary` | 含量只能经由边界改变，所以"记忆 = 储备"要求学习必须是边界驱动的、提高含量的交换 |
| 用 | `Life.never_alive_without_intake`、`OperationalLife.alive` | 准入门：总含量大于零，并且能碰到一次摄入 |

### B. 耳与自听（感知、自我模型）
| | 定理 | 说明 |
|---|---|---|
| 用 | G13（`Generation`） | 预测 = 边界在慢内部条件下的 Gibbs 条件分布 |
| 用 | N2 `snapshot_misses_hidden` | 话题、意图这类隐藏量只能靠跨时间的记录取得 |
| 用 | `SensoryRecord.action_content_distinguishes`、`record_state_lower_bound` | 精确记录必须区分"谁说的、说了什么"：自听要带上自己行动的副本（传出副本），否则自己的话和妈妈的话会混在一起 |
| 用 | **`observational_record_does_not_fix_intervention`** | 只看记录，定不了"我这样说会怎样"。要知道自己行动的后果，必须行动：提问在因果上是必要的。这和 N4 是两条独立的理由 |
| 用 | `CausalState.separated_histories_capacity`、`history_readout_exists_iff` | 状态大小的下界：要区分多少种历史，就至少要多少状态 |
| 用 | G12 `miss_counter_necessary`、`intake_memory_capacity` | 基准（别处平均）必须有记忆来承载 |

### C. 记忆与代谢（遗忘 + 学习）
| | 定理 | 说明 |
|---|---|---|
| 用 | G16a `export_forced`、G16b `write_only_intake_bounded`、`endless_intake_endless_export` | 有限身体面对无限流，必须遗忘；无止境的摄入就伴随无止境的排出 |
| 用 | `survive_le_pow`、`unselective_export_forgets`、G17b `retention_needs_selectivity` | 统一衰减会几何地忘光一切，保留必须有选择 |
| 用 | `GenerationLinearMemory`（全部 7 条） | weight decay = 门 α = 非选择性排出；线性记忆的时间深度有限（`histories_merge`）；纠错需要非线性 |
| 用 | `GenerationCascadeMemory` | 多时间尺度叠加，痕迹按 1/t 衰减 |
| 用 | W2 `vanishing_gain_loses_under_drift` | 世界会变时，不遗忘就会输：遗忘是必须的，不是正则化 |
| 用 | `InformationThermodynamics.erased_information_floor`、`positive_erasure_requires_positive_dissipation`、N3 | 代谢的能量代价落在擦除上 |
| 用 | G23、G24 `equilibrium_memory_bound` | 平衡态记忆有界；长程记忆要靠驱动 |
| 用 | H1 `trap_irrecoverable`、H2 `trap_insensitive` | 锁死就是死，锁存时对输入是盲的：记忆不能锁死 |
| 相关 | `TurnoverControls.ledger_not_identified_by_state_dynamics` | 含量不变时，"学的和忘的正好平衡"和"什么也没发生"在状态动力学上无法区分。所以评估代谢必须记录写入和排出两个通量，不能只看状态 |
| 相关 | CLO `ema_forgets_rare_spikes`、`frozen_hold_loses_on_drift`（GPT 分支，只引用） | 统一衰减会丢掉罕见事件，冻结在漂移下必输 |

### D. 学习规则（数值从这里来）
| | 定理 | 说明 |
|---|---|---|
| 用 | G14 `two_phase_drift`、`phases_must_balance`、`sector_law_is_softmax` | 预测学习：局部两相之差就是似然梯度；读出是 softmax |
| 用 | G15 `reward_two_phase_drift`、`expected_reward_deriv`、`outcome_blind_drift_zero`、`reward_phases_must_balance` | 行动学习：三因子 = 策略梯度；调制信号必须读摄入；必须有基准 |
| 用 | **`delayed_credit_requires_history_carrier`**、`no_visible_only_delayed_credit_rule` | 妈妈的回应会延迟到达，只看当前可见量的规则做不了延迟信用分配，必须有历史载体。所以三因子的资格迹是必须的，不是可选项 |
| 用 | `agreement_no_write`、`CausalLearning.write_contract` | 预测对了就不写 |
| 用 | `finite_reweighting_improves_iff`、`LocalEdit.accepts_iff` | 10-25 规格的离散修改判据，是 G14 的离散形式 |
| 用 | W3 `occurrence_clock_beats_time_clock` | 罕见的字节或词要按出现次数计时，不能按拍计时 |
| 用 | `no_fixed_gain_for_two_volatilities`、`no_global_gain_for_heterogeneous_compartments`、`no_shared_shrink_for_two_snr` | 不同单元必须有各自的增益（学习速率），不能共用一个全局值 |
| 用 | **`own_step_biases_noise_estimate`、`residual_removes_own_step`、`in_sample_decrease_bias`、`fresh_decrease_unbiased`** | 自己这一步会让同一批数据上的评估产生偏差，要用新数据评估。这正是自听闭环的风险（见 §4 缺口 1） |
| 用 | L-1 `innate_optimal_of_constant`、L-2 `learner_beats_innate`、L-3 `history_useless_of_independent` | 不变的世界造就本能，代际间变化的世界造就学习者；学习需要过去能预测未来。妈妈的回应依赖于孩子说的话，这个世界会变，所以学习胜过跨代搜索（理论依据） |
| 用 | N5 `mixture_regret_le` | 学习会饱和；中等可预测处学得最多 |
| 相关 | `GenerationOptimizer.reservoir_gain_is_local_anti_dominant`（GPT 分支，只引用） | 由局部资源的可用量决定学习增益：把代谢（局部资源）和学习速率接起来 |
| 相关 | `CausalLearning.empty_write_reserve`、`RenewableLearning`（写入额度由摄入补充） | 现有的有限载体实例里有一个"写入储备"。它只是实例，不是义务：义务是每拍写入预算有限（`PaidRecord.combined_write_budget`），不要求有存量 |

### E. 嘴（行动）
| | 定理 | 说明 |
|---|---|---|
| 用 | C-1 `no_fixed_leave_threshold`、G12 `exists_leave_threshold` | 锚定还是提问：拿当前进展和基准比 |
| 用 | N4 `passivity_ends` | 只听不问不能持续 |
| 用 | N7 `tempo_band`、`peaks_wasted`、`uniform_attains` | 语速是一个带；信息要均匀地说；轮流说话 |
| 用 | G25 `fixed_threshold_fails`、`centred_threshold_resolves`、`gain_times_band` | 开口的门槛要对准信号当前的范围（自适应），固定门槛会失效 |
| 用 | Persistence B `integral_feedback_adapts` | 基准的自适应可以由积分反馈实现 |
| 用 | H1/H2 | 嘴不能锁死（一直说或一直不说） |
| 用 | G26b `stay_starves` / `stay_dies` | 待在原地会饿死，对应总在同一个话题上 |
| 相关 | `CausalLearning.no_safe_exploration` | 提问的安全性无法事先证明：探索有不可消除的风险 |
| 不用 | `PredictiveHarvest`（T1） | 生态位合同，会让"暗室"成为最优；已降级 |

### F. 发育（结构从哪来）
| | 定理 | 说明 |
|---|---|---|
| 用 | G18 `GenerationGrowth`（`slot_reciprocity`、`memory_separates`、`shared_potential_forces_nesting` 等） | 结构由占据、配对和记忆长出来；没有记忆时，同样频率的共同活动只能长出同样的结构 |
| 用 | `NestedWorld.births_only_by_division`、`move_exchanges` | 分裂与交换（和生态球同一个生成器） |
| 用 | `GenerationAxiomAudit`（均匀性迫使接触图规则化）、`Lattice` M4 | 介质是格点，接触图规则 |

### G. 供养与能量
| | 定理 | 说明 |
|---|---|---|
| 用 | N1 `freeEnergy_eq_kl`、T-E | 能量 = −log π；自由能 = KL |
| 用 | `EnergyBudget.no_sustained_use_on_finite_supply` | 供给有限，就不能持续使用 |
| 用 | T-N7 `stream_never_dry` | 流永不枯竭（世界条件） |
| 用 | N3 `finite_memory_bounds_net_work` | 平衡世界给不出净收获：妈妈的话必须是低熵的，可学 |

### 不适用
- `SensoryMaintenance*`、`GraceRecovery*`、`Port*`：大肠杆菌时期的协议细节和审计。
- `Models`、`AtomicIndependence`、`ConstraintAudit`：公理的见证和独立性。
- 各 `GenerationOrganism*`：具体物种。

它们是正确的，但不约束这个设计。

## 3. 三条主链（根 → 这里用到的结论）

1. **有限身体 × 无限流**
   - ⇒ 必须遗忘（G16a）⇒ 必须有选择地遗忘（G17b）⇒ 选择只能靠能量（G1）。
   - ⇒ 遗忘的门就是 weight decay（LinearMemory），衰减速度由世界漂移决定（W2）。
   - ⇒ 代谢 = 遗忘 + 学习；记忆就是储备（`content_drift_boundary` + `OperationalLife.alive`）。
2. **世界会变 + 有延迟**
   - ⇒ 必须比较（C-1）⇒ 基准需要记忆（G12）⇒ 门槛要居中（G25）。
   - ⇒ 学习胜过本能（L-2）⇒ 两相 / 三因子（G14/G15）⇒ 延迟信用需要历史载体（`delayed_credit_…`）。
3. **只看记录定不了干预**（`observational_record_…`）**+ 不观察则不确定性增长**（N4）
   - ⇒ 必须行动（提问）⇒ 行动要看后果（N2：回应必须依赖于行动）⇒ 自听是最强的依赖 ⇒ 自我模型。
   - ⇒ 节奏受双方有限性约束（N7）。

三条链在一个量上汇合：**学习进展 − 基准**。它既是开口信号，也是可塑性的第三因子，也是"活着"的读数（学得比忘得快）。

## 4. 缺口（要新证，实现之前做）

1. **自听闭环下的无偏性**：输入依赖于自己的输出时，G14 两相漂移还是不是似然梯度？按 `residual_removes_own_step` 的模式，用传出副本扣除自己那一步。
2. **资格迹的长度**：级联迹的时间深度 ≥ 妈妈回应的延迟（`delayed_credit_requires_history_carrier` + `GenerationCascadeMemory`）。
3. **衰减时间常数 = 世界漂移的函数**：由 W2 和 GradientFilter 的稳态增益（`gain_error_square`、`steady_gain_worse_for_fresh`）推出最优衰减的**形式**：它要随漂移与噪声之比变化，遇到变化就要多忘。**数值**像 GDN 的 α_t 一样在一生中在线学出，可以用前向灵敏度，也可以在线估计漂移和噪声。要证的是：前向灵敏度对线性递推是精确的，并且在自听闭环下无偏（与缺口 1 合并）。
4. **第三因子读的是摄入**：在语言生命体里，摄入就是学习进展，需要证明它满足 G15c 的"读摄入"条件。
5. **生成器的表达力**：E-mouth（比较后触发）、E-tempo（弛豫到位才输出）、慢页里的非线性（锁存或吸引子，`linear_hold_keeps_error` 要求）。
