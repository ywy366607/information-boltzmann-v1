# 语言生命体：到目前为止的决定，以及要用哪些定理（2026-10-28）

**历史规划，非当前准入结论。** 原文将若干候选设计、物理类比与定理混在一起；以下条目的可用条件应以 [定理关系图](CGB_THEOREM_RELATION_MAP_20261028.md) 和 Lean 声明为准。柱结构、活动层、学习信号的具体形式及世界回应机制尚未由公理唯一推出，也未通过同一语言世界的联合存在性证明。

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
| 妈妈 | 可作为外部语言世界；可以回应、沉默或改变话题。行动影响未来输入是特定世界条件，不要求每次都有回应 |
| 自听 | 要。自我模型由此而来 |
| 信道容量 | 不外加。自听已让说话者受自己的容量约束 |
| 储备 | 不单设储备页。学到的模型就是储备（条件见 §3） |
| 信息维持 | 研究状态保留、更新及未来预测收益；“学得比忘得快”尚未被证明等价于存活 |
| 数值来源 | 不用跨代搜索（太慢）。定理定范围，个体一生中的学习（G14 两相 / G15 三因子）定数值 |
| 语言 | 英文；输入输出是原生字节 0..255 |
| 活动与保持 | 特定收缩条件下的零输入递推有唯一静止点；实际载体能否产生主动活动、健康保持与释放，需要在同一动力学下证明 |
| 结构 | 柱、柱内连接与稀疏跨柱连接是历史草图，不是已由公理唯一推出的规模或结构。原文的 `64×64×64 = 64 柱、每柱 64 单元` 算术及含义不一致，不能作为参数配置；实际规模、路径和成本须随生成规则核验 |
| 约束 | 能在手机上跑：前向递推，无 BPTT，无 FP64 |

## 0a. 当前实数 GDN 联合桥（2026-10-31）

用户已明确：保留神经网络的实数矩阵状态，不为迁就旧门的 `[Fintype State]` 先改造成量化/查表模型。renewable 确定性载体、开放世界和联合门已去除无用的身体有限性；评分字母仍为256字节。真正依赖身体基数或有限求和的定理保留原前提。

新增 `GenerationGDNLife.lean` 固定自然参数/精度、因果 query、时间 prior、观察同化及保存特征上的仿射 head 更新。`carrier_fold` 和 `Process.run_step` 使它们位于同一实数身体—世界轨迹；`finish_future_ce_improves` 和 `Process.next_byte_improves` 给出带实际读出漂移/世界持续性条件的未来收益桥。`ExternalLanguageLaw.information` 将同一状态/行动条件下的当前外部字节—后续外部字节 CMI 注册为世界性质，`recurrent_external_information_forces_activity` 由被动后的 CMI 归零和反复正信息推出不可能最终被动。`externallyUseful` 再把其中实际可学的事件接到后继评分改善；全程不使用任意摄入或机会预算。

**范围**：固定 query 是上一到达字节的地址，encoder/address 未在线学习；精度正只是数值不变域，不是物理存活；外部来源的真实性、被动为何失去有益语言、行动影响、预测余量及反复机会仍须在实际世界兑现。没有证明完整 `Medium.Lives`，也没有把有限精度、完整恢复或资源预算义务取消。储备是状态中可用的未来区别，耗散是这些区别被遗忘或串扰抹去；两者都不是额外资源字段。详见 [实数 GDN 联合桥及因果/条件表](CGB_GDN_REAL_STATE_JOINT_BRIDGE_20261031.md)。

后续完备性核对见 [完整目标与生成元依赖表](CGB_LANGUAGE_LIFE_COMPLETENESS_20261031.md)。`GenerationLanguageExistence` 新增实际精度同化到两态占据动力学的等式、有界记忆不变域、真实静默中的 learned bias 区别保持，以及物理有限表示/持续固定目标改善的边界。`GenerationGraceMemory.remembers` 对未修改的物理 Grace 过程精确算得一次写入与所有 quiet 延迟的 CMI=log2；它仍不满足原 `Persists`，不能替代 GDN 或完整生命证书。

## 1. 架构草图（历史候选，待条件与生成器证明）

```
 妈妈的字节 ─┐                                        ┌─> 嘴（门控输出反应）─┐
             ├─> 耳 ─> 快页（预测：G13 弛豫） ─> 读出 ┤                     │
 自己的字节 ─┘          ↕ 慢页（记忆：级联池组）       └─ 学习进展 − 基准 ─┘
     ^                                                    （= 开口信号 = 第三因子）
     └────────────────────────── 自听 ───────────────────────────────┘
```

下面的门控递推是历史候选。`GenerationLinearMemory` 证明特定线性递推的条件性质；各实际架构是否满足这些前提，需要逐项核对：
- 门 α 就是非选择性排出，即 weight decay；
- delta 项是选择性排出；
- 写入是摄入。

所以这类网络是我们框架的一个实例。

**注（用户指出）**：GDN 的 α_t（Mamba 的 Δ_t 同理）是可学习且依赖输入的，每个 token、每个头一个值。所以它对内容不加选择（同一头里所有方向一起衰减），但对时间有选择（遇到话题切换时可以突然多忘）；对内容的选择交给 delta 项。

W2 和 C-1 在各自声明的模型中约束增益与行动；它们没有证明所有环境变化都要求增加遗忘。几何时间尺度可以作为条件性构造，不能仅由上述结论指定为必需架构。

但 α 可学并不能绕开 `histories_merge` 和 `linear_hold_keeps_error`：只要 α ≤ ᾱ < 1，记忆深度就有限；α → 1 时能长期保持，但误差也一起保持。

对于标量线性递推，衰减参数的灵敏度可向前递推；真实模型还要考虑状态依赖门、自听反馈，以及对参数数目和连接规模的灵敏度开销。“不随流长增长”不自动意味着对模型规模便宜。`linear_hold_keeps_error` 的结论限于其线性保持与扰动条件，不直接指定必须新增慢页或某一锁存结构。

## 2. 按部件：用哪些定理

### A. 容器与边界
| | 定理 | 在这里说明什么 |
|---|---|---|
| 用 | `Interaction.Machine.conditioned_next`、`trace_prefix_consistent` | 开放交互的数学对象（10-25 规格已采用） |
| 用 | `Medium.blanket_of_axioms`、`Nesting.organism_blanket` | 身体有边界（马尔可夫毯） |
| 用 | `Medium.modulation_needs_boundary`、`rate_interior_stimulus_free` | 驱动只能来自边界：内部不能凭空产生学习信号 |
| 相关 | `NoetherBridge.content_drift_boundary` | 在其守恒合同下含量变化由边界通量给出；不等于证明记忆必须对应含量增加 |
| 用 | `Life.never_alive_without_intake`、`OperationalLife.alive` | 准入门：总含量大于零，并且能碰到一次摄入 |

### B. 耳与自听（感知、自我模型）
| | 定理 | 说明 |
|---|---|---|
| 用 | G13（`Generation`） | 预测 = 边界在慢内部条件下的 Gibbs 条件分布 |
| 相关 | N2 `snapshot_misses_hidden` | 在指定测量核只读可见部分的条件下，联合状态的信息等于可见部分的信息；隐藏量可能通过相关性或其他观测获知 |
| 相关 | `SensoryRecord.action_content_distinguishes`、`record_state_lower_bound` | 对指定 16 状态实例，保持行动和含量读出要求区分全部状态；不指定通用自听必须增加独立副本 |
| 相关 | **`observational_record_does_not_fix_intervention`** | 所给反例说明特定观测记录不能唯一识别干预结果；不证明所有因果知识都必须由自己提问取得 |
| 用 | `CausalState.separated_histories_capacity`、`history_readout_exists_iff` | 状态大小的下界：要区分多少种历史，就至少要多少状态 |
| 用 | G12 `miss_counter_necessary`、`intake_memory_capacity` | 基准（别处平均）必须有记忆来承载 |

### C. 记忆与代谢（遗忘 + 学习）
| | 定理 | 说明 |
|---|---|---|
| 相关 | G16a `export_forced`、G16b `write_only_intake_bounded`、`endless_intake_endless_export` | 在所给内容守恒与有限容量合同下，无界累计摄入要求无界累计排出；不自动等价于遗忘某项有用信息 |
| 用 | `survive_le_pow`、`unselective_export_forgets`、G17b `retention_needs_selectivity` | 统一衰减会几何地忘光一切，保留必须有选择 |
| 用 | `GenerationLinearMemory`（全部 7 条） | weight decay = 门 α = 非选择性排出；线性记忆的时间深度有限（`histories_merge`）；纠错需要非线性 |
| 用 | `GenerationCascadeMemory` | 多时间尺度叠加，痕迹按 1/t 衰减 |
| 相关 | W2 `vanishing_gain_loses_under_drift` | 在指定漂移和增益模型中增益消失导致估计误差超界；不证明所有变化世界都需要同一种遗忘 |
| 相关 | `InformationThermodynamics.erased_information_floor`、`positive_erasure_requires_positive_dissipation`、N3 | 指定擦除、工作与热力学假设下的条件性界；语言损失不自动等于物理耗散 |
| 用 | G23、G24 `equilibrium_memory_bound` | 平衡态记忆有界；长程记忆要靠驱动 |
| 用 | H1 `trap_irrecoverable`、H2 `trap_insensitive` | 锁死就是死，锁存时对输入是盲的：记忆不能锁死 |
| 相关 | `TurnoverControls.ledger_not_identified_by_state_dynamics` | 含量不变时，"学的和忘的正好平衡"和"什么也没发生"在状态动力学上无法区分。所以评估代谢必须记录写入和排出两个通量，不能只看状态 |
| 相关 | CLO `ema_forgets_rare_spikes`、`frozen_hold_loses_on_drift`（GPT 分支，只引用） | 统一衰减会丢掉罕见事件，冻结在漂移下必输 |

### D. 学习规则（数值从这里来）
| | 定理 | 说明 |
|---|---|---|
| 用 | G14 `two_phase_drift`、`phases_must_balance`、`sector_law_is_softmax` | 预测学习：局部两相之差就是似然梯度；读出是 softmax |
| 相关 | G15 `reward_two_phase_drift`、`expected_reward_deriv`、`outcome_blind_drift_zero`、`reward_phases_must_balance` | 在指定策略、两相与收益条件下得到对应的策略梯度及盲调制结论；不证明所有学习都只能采用一个三因子公式 |
| 用 | **`delayed_credit_requires_history_carrier`**、`no_visible_only_delayed_credit_rule` | 当不同经历在当前可见状态合并、但要求不同信用时，需要能够访问的历史区别；可以是现有状态或可检索外部记录，不指定唯一资格迹形式 |
| 用 | `agreement_no_write`、`CausalLearning.write_contract` | 预测对了就不写 |
| 用 | `finite_reweighting_improves_iff`、`LocalEdit.accepts_iff` | 10-25 规格的离散修改判据，是 G14 的离散形式 |
| 用 | W3 `occurrence_clock_beats_time_clock` | 罕见的字节或词要按出现次数计时，不能按拍计时 |
| 相关 | `no_fixed_gain_for_two_volatilities`、`no_global_gain_for_heterogeneous_compartments`、`no_shared_shrink_for_two_snr` | 指定估计模型中，异质条件下共享固定增益不能同时达到各自最优；不证明所有系统必须逐单元设置学习率 |
| 用 | **`own_step_biases_noise_estimate`、`residual_removes_own_step`、`in_sample_decrease_bias`、`fresh_decrease_unbiased`** | 自己这一步会让同一批数据上的评估产生偏差，要用新数据评估。这正是自听闭环的风险（见 §4 缺口 1） |
| 相关 | L-1 `innate_optimal_of_constant`、L-2 `learner_beats_innate`、L-3 `history_useless_of_independent` | 分别约束指定觅食收益模型；不能从回应依赖行动直接推出学习胜过跨代搜索 |
| 相关 | N5 `mixture_regret_le` | 指定有限混合相对某正概率假设的超额编码长度有界；不推出学习饱和 |
| 相关 | `GenerationOptimizer.reservoir_gain_is_local_anti_dominant`（GPT 分支，只引用） | 由局部资源的可用量决定学习增益：把代谢（局部资源）和学习速率接起来 |
| 相关 | `CausalLearning.empty_write_reserve`、`RenewableLearning`（写入额度由摄入补充） | 现有的有限载体实例里有一个"写入储备"。它只是实例，不是义务：义务是每拍写入预算有限（`PaidRecord.combined_write_budget`），不要求有存量 |

### E. 嘴（行动）
| | 定理 | 说明 |
|---|---|---|
| 用 | C-1 `no_fixed_leave_threshold`、G12 `exists_leave_threshold` | 锚定还是提问：拿当前进展和基准比 |
| 相关 | N4 `passivity_ends` | 指定正漂移模型中的对数表达式终将超过固定数值阈值；不单独推出行动或禁止只听 |
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
| 相关 | N3 `finite_memory_bounds_net_work` | 在假定工作和擦除两项界的条件下得到净值代数上界；不推出妈妈必须输出某种熵率 |

### 不适用
- `SensoryMaintenance*`、`GraceRecovery*`、`Port*`：大肠杆菌时期的协议细节和审计。
- `Models`、`AtomicIndependence`、`ConstraintAudit`：公理的见证和独立性。
- 各 `GenerationOrganism*`：具体物种。

它们是正确的，但不约束这个设计。

## 3. 三条主链（根 → 这里用到的结论）

1. **有限身体与持续交互**：在特定守恒账本下，无界累计摄入要求累计排出；对未来任务必须区分的历史，有限状态、允许的误差和可访问外部记录共同约束可保持的信息。哪些区别有用，要由未来预测和行动条件决定。
2. **世界变化与延迟结果**：若相同可见状态对应需要不同信用的经历，归因需要可访问的历史区别；已有两相、三因子及前向递推定理提供条件性实现线索，尚不指定唯一算法。
3. **行动与世界**：在被动机会有限且持续可行性依赖指定信号等条件下，已有反事实定理可推出不能最终被动。具体世界的回应、沉默、行动后果及自维持仍须在同一过程上验证。

“学习进展 − 基准”是历史候选信号，不是已证明统一生成三条链或等价于存活的量。

## 3b. 学习：前向信用与三因子的历史线索

- **已有**：
  - `FLY_EVENT_THREE_FACTOR_DESIGN_20261015.md`：事件驱动的三因子设计；资格状态、稠密调制与通信的完整成本仍需核算。
  - `fly_carried_credit.py`：跨窗口传递部分前向灵敏度；窗口内仍有 BPTT，且冻结部分反馈路径，详见下文核对。
  - G15 与 `delayed_credit_requires_history_carrier` 给出指定形式与条件下的推导，不证明所有可行学习法都必须采用该具体资格迹。
- **当前边界**：前向灵敏度和三因子都提供可能的在线学习方式；免于保存随时间增长的计算图，不等于所有此类算法互相等价，也不保证其随模型规模的状态与计算开销可接受。所需历史区别、可访问的结果信号和具体递推要在同一规则上验证。
- **历史柱结构设想，尚未形成联合准入**：
  - 柱内突触 S_m：delta 写本身就是逐字节的在线学习，不需要 BPTT。误差是"到达 − 预测"，每个后端单元局部可得。
  - 线性部分（α、β、门）：前向灵敏度是精确的。
  - 非线性活动层：`fly_carried_credit` 的条件灵敏度有路径限制；两相对比是否适用、是否精确及成本均须另证。
  - 嘴和行动：第三因子 = q − c，广播（类似多巴胺）。
- **果蝇的三个教训**（10-04 的在线学习平台期，以及 10-15 的设计文档）：
  1. 那次特定运行中大量更新小于 float32 的 ULP。随机舍入是待比较的实现选择，不是公理自动要求；量化收益和质量需实测。
  2. 第三因子没有携带信号（多巴胺沉默，门 2 ≈ 0）。训练前必须先审计 q − c 是否真的变化、是否真的到达调制载体。
  3. 那次随机低秩反馈的信用近似存在误差；候选替代路径需要同预算验证。

## 3c. 已证（2026-10-28，`GenerationColumnMemory.lean`，完整构建与审计通过）

| 定理 | 内容 |
|---|---|
| `silent_rest_unique` | 在单位键、指定增益区间、严格衰减且零写入的条件下，所述门控 delta 步只有零静止点 |
| `tied_write_is_error_descent`、`tied_write_error_shrinks`、`exact_prediction_no_write` | 用同一个键先读后写，就是在 ½(⟨k,x⟩ − v)² 上走一步梯度（到达 − 预测，乘以前端键，即 G14 的差）；单位键时误差按 1 − β 缩小；预测准确时不写 |
| `occurrence_decay_retains`、`unused_column_kept` | 按使用衰减（α^w，w ∈ [0,1]）保留的至少和每拍都衰减一样多；不被用到的柱完全保留 |
| `no_latch_below_unit_gain`、`hardSat_lipschitz`、`latch_above_unit_gain` | 在各自声明的标量映射条件下，收缩时只有零静止点，硬饱和且 g ≥ 1 时存在 ±1 静止点；未证明 delta 写入、抗扰保持与释放的联合动力学 |

## 4. 缺口（要新证，实现之前做）

### 2026-10-29 接手核对：自听与已证范围

`GenerationColumnMemory.self_listening_increment` 给出标量闭环例 `x⁺ = αx + βθx + θ` 的精确双轨差分；`self_listening_frozen_state_gap` 和 `self_listening_frozen_state_counterexample` 显示冻结自听路径会漏掉 `βθ` 对历史灵敏度的作用。它是反馈信用缺口的条件反例，不是妈妈响应可微或随机信用无偏的证明。离散输出、外部妈妈的因果反馈和活动层耦合仍需要自己的估计器及独立检验。

该模块局部编译通过；完整审计运行刷新了 1218 条索引，但随后定理解释目录因 `CGB.Generation.dot_self_nonneg` 的源哈希变化而报错，故本次完整审计尚未全部通过。不得将索引刷新等同于整条审计成功。

`fly_carried_credit.py` 的既有实现将跨窗口的条件直接路径灵敏度带到窗口边界，文件自身明确窗口内仍用 BPTT，且冻结其他细胞的尖峰与多巴胺；所以不能将其视为已经实现完整网络无 BPTT 或所有路径精确。`FLY_EVENT_THREE_FACTOR_DESIGN_20261015.md` 是事件驱动设计，原文保留 BPTT 支架，并指出稠密第三因子会影响稀疏复杂度；它不是已证在手机端运行的本柱算法。时间长度 O(1) 不等于参数数目 O(1)，每突触资格状态和第三因子传播的成本需单独计量。

`silent_rest_unique` 依赖单位键、`0 ≤ β ≤ 2`、`0 ≤ α < 1`、零写入；它不覆盖含持续输入、其他状态或非收缩情形的整个 GDN 系统。`latch_above_unit_gain` 给出标量饱和映射的固定点：`g = 1` 的固定点不自动给出抗扰吸引，亦未证明柱中学习规则产生稳定且可释放的锁存。活动层的驱动循环、状态可达性和自听—妈妈闭环仍是未完成的桥。

已证的部分见 §3c。下面仍待证（编号沿用）：

1. **自听闭环下的无偏性**：输入依赖于自己的输出时，G14 两相漂移还是不是似然梯度？按 `residual_removes_own_step` 的模式，用传出副本扣除自己那一步。
2. **资格迹的长度**：级联迹的时间深度 ≥ 妈妈回应的延迟（`delayed_credit_requires_history_carrier` + `GenerationCascadeMemory`）。
3. **衰减时间常数 = 世界漂移的函数**：由 W2 和 GradientFilter 的稳态增益（`gain_error_square`、`steady_gain_worse_for_fresh`）推出最优衰减的**形式**：它要随漂移与噪声之比变化，遇到变化就要多忘。**数值**像 GDN 的 α_t 一样在一生中在线学出，可以用前向灵敏度，也可以在线估计漂移和噪声。要证的是：前向灵敏度对线性递推是精确的，并且在自听闭环下无偏（与缺口 1 合并）。
4. **第三因子读的是摄入**：在语言生命体里，摄入就是学习进展，需要证明它满足 G15c 的"读摄入"条件。
5. **生成器的表达力**：E-mouth（比较后触发）、E-tempo（弛豫到位才输出）、慢页里的非线性（锁存或吸引子，`linear_hold_keeps_error` 要求）。
