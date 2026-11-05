# CGB 公理–定理因果图与最简生成元（2026-11-02）

数据来源：`lean/CGBAxioms/.lake/audit_inventory.json` 与 `CGB_THEOREM_CLASSES.json`。必要或门控定理共 95 条，库存较旧，其中 86 条在库、9 条 `Nonlinearity` 不在库。

本文分三种证据：**Lean 算出**的依赖；**手工归族**（读定理前提后的判断，尚未复核）；**候选机制**（文献和推理，未运行、未进 Lean）。

## 1. Lean 里的真实依赖

| 与公理条款的关系 | 86 条必要定理 | 全部 1688 条已分类定理 |
|---|---|---|
| 依赖某条 P 条款或整组公理 | 3 | 326 |
| 只提到介质对象 | 21 | 403 |
| 完全不碰介质 | 62 | 959 |

依赖 P 条款的 3 条是 `finite_reachable`（P2a）、`modulation_needs_boundary`（P1b）、`ContextBody.body_axioms`（全部六条）。

所以这些定理不是从 P1–P3 推出来的。因果次序写在它们的前提里：例如 `export_forced` 的前提 `c(t+1)+out=c+inn`、`c≤cap` 正是守恒加有限，但 Lean 里没有一条边把它连回 P2a、P1a。

## 2. 哪些成链，哪些是单独的必要条件

- 20 条的证明确实用到另一条必要定理。
- 31 条没有用到项目里的任何定理，是一步引理。
- 连通分量：11、10、7、3 各一组，7 对，41 条孤立。11 条那组内容混杂，靠共用底层引理连起来，不算真链。

**真链：**

1. **历史链（10 条）**：`history_readout_exists_iff`、`history_transition_exists_iff` → `causal_carrier_contract_iff`、`external_causal_readout_exists_iff` → `open_world_body_prediction_iff`、`future_reserve_recoverable_iff`、`step_source_irrelevant`、`self_not_externallyUseful` → `recurrent_external_evidence_forces_activity`、`recurrent_external_information_forces_activity`。
2. **闭环链（7 条）**：`not_unbounded_of_bounded`、`action_condition_required`、`silence_condition_required` → `viable_not_eventually_passive` → `active_closure`；另有 `selective_write_contract`、`candidate_erasure_not_unbounded_signal`。
3. **有限损失链（3 条）**：`finite_body_stationary_loss_eventually_constant` → `recurrent_gain_excludes_eventual_stationary_descent`、`strict_descent_has_no_finite_exact_readout`。

**成对：** `export_forced`–`write_only_intake_bounded`；`unselective_export_forgets`–`retention_needs_selectivity`；`contraction_unique_rest`–`no_latch_below_unit_gain`；`assimilate_order_blind`–`no_read_separates_order`；`no_visible_only_delayed_credit_rule`–`delayed_credit_requires_history_carrier`；`no_fixed_gain_for_two_volatilities`–`no_global_gain_for_heterogeneous_compartments`；`phases_must_balance`–`reward_phases_must_balance`。

其余都是单独的必要条件。

## 3. 按前提归族（手工）

| 族 | 一句话 | 上游条件 | 条数 | 对机制的作用 |
|---|---|---|---|---|
| K 区分律 | 状态里没有的区分读不出来 | 有限（P1a） | 26 | 验收 |
| L 账本律 | 进多少就要出多少，没有流入就活不下去 | 守恒（P2a）+ 边界流 | 17 | 机制直接满足 |
| C 松弛律 | 增益不到 1 就只有一个静息点，记忆会被洗掉 | 松弛（P2b），桥未证 | 13 | 机制直接满足 |
| X 局部律 | 调制要到边界，对称不自破，竞争要读本地需求 | 局部（P1b）+ 同规则（P2c、P3） | 8 | 机制直接满足 |
| N 非线性律 | 叠加态只能线性读，没有交互 | 语言这个任务 | 9（不在库） | 机制直接满足 |
| W 世界律 | 世界在变，固定增益必输，观测不定干预 | 世界 | 15 | 验收 |
| B 身体特例 | 只说某个候选身体 | 无 | 7 | 换身体就作废 |

**K（26）**：`merged_history_preserves_read_and_successor`、`assimilate_order_blind`、`no_read_separates_order`、`causal_carrier_contract_iff`、`delayed_action_credit_requires_state_separation`、`delayed_credit_requires_history_carrier`、`external_causal_readout_exists_iff`、`external_trace_requires_retrieved_distinction`、`history_readout_exists_iff`、`history_transition_exists_iff`、`no_visible_only_delayed_credit_rule`、`separated_histories_capacity`、`record_state_lower_bound`、`merged_histories_binary_logloss_floor`、`precision_distinguishes_equal_mean_histories`、`exact_key_capacity`、`exact_decoder_capacity`、`collision_of_excess_keys`、`observed_memory_information_le`、`future_reserve_recoverable_iff`、`loses_future_language_distinction_iff_not_preserved`、`open_world_body_prediction_iff`、`finite_body_stationary_loss_eventually_constant`、`strict_descent_has_no_finite_exact_readout`、`recurrent_gain_excludes_eventual_stationary_descent`、`finite_reachable`。

**L（17）**：`export_forced`、`write_only_intake_bounded`、`no_sustained_use_on_finite_supply`、`not_persists_without_incoming`、`historical_persistence_requires_alive_change`、`conserving_calm_needs_organizational_viability`、`trap_irrecoverable`、`no_allocation_persists`、`predictive_growth_contract`、`not_unbounded_of_bounded`、`viable_not_eventually_passive`、`active_closure`、`candidate_erasure_not_unbounded_signal`、`counterfactual_passive_bound_forces_activity`、`recurrent_external_evidence_forces_activity`、`recurrent_external_information_forces_activity`、`selective_write_contract`。

**C（13）**：`contraction_unique_rest`、`latch_needs_unit_gain`、`no_latch_below_unit_gain`、`silent_rest_unique`、`contraction_has_no_two_latches`、`common_suffix_address_bound`、`histories_merge`、`linear_hold_keeps_error`、`held_coordinate_survives_suffix`、`no_uniform_positive_lower`、`retention_needs_selectivity`、`unselective_export_forgets`、`bank_power_law_lower_bound`。

**X（8）**：`modulation_needs_boundary`、`events_touch_bounded`、`symmetric_stays_symmetric`、`three_part_balance_impossible`、`phases_must_balance`、`reward_phases_must_balance`、`local_anti_dominance_requires_demand_readout`、`column_budget_allows_wta`。

**N（9，不在库）**：`additive_no_interaction`、`interaction_needs_nonadditive`、`linear_read_of_superposed_state`、`linear_stack_superposes`、`linear_stack_no_interaction`、`linear_recurrence_no_interaction`、`additive_logit_cannot_separate`、`additive_error_floor`、`association_read_additive_in_contents`。

**W（15）**：`frozen_hold_loses_on_drift`、`vanishing_gain_loses_under_drift`、`no_fixed_gain_for_two_volatilities`、`no_global_gain_for_heterogeneous_compartments`、`innate_loss_quarter`、`beaten_forever_iff_never_settles`、`history_useless_of_independent`、`process_noise_gain_floor`、`precision_binary_sample_gain_not_future_guarantee`、`no_safe_exploration`、`observational_record_does_not_fix_intervention`、`indistinguishable_joint_worlds_cannot_both_stay_viable`、`action_condition_required`、`silence_condition_required`、`action_distinction_preserved`。

**B（7）**：`body_axioms`、`order_separated`、`context_write_separates_order`、`association_readout_interference_bound`、`preparation_uses_only_elapsed`、`step_source_irrelevant`、`self_not_externallyUseful`。

## 4. 麦克斯韦形式（候选）

麦克斯韦方程单独也算不出具体结果，还要材料关系和边界条件。这里同样分三层：

- **定律**：有限、局部接触、含量守恒、可逆速率。Lean 已有一次性证明：`PairCRN.axioms`、`LatticeCRN.axioms`、`tableCRN.axioms`、`p2b_of_energy`、`rate_local_of_energy`。一张局部反应表可以同时满足全部六条。
- **材料**：反应表本身，也就是化学和基因组。公理不规定它，`arbitrary_driven_tables_admitted` 说明任意表都能过。
- **边界**：语言字节流。

**候选最简机制是一张两行的反应表（第一稿，已由第 7 节的一轮循环取代）：**

| 反应 | 形式 | 想要它带来什么 |
|---|---|---|
| R1 自催化耗池 | 活 + 活 + 池 ⇌ 活 + 活 + 活；池从边界流入、扩散快，活扩散慢 | 增益大于 1（C）、乘积非线性（N）、账本（L）、局部耗尽带来的竞争和隔室（X） |
| R2 接触结合 | 两端活跃 + 池 ⇌ 接触 | 用得多的连接留下，不用的拆回池里（G18 槽位） |

R1 是 Gray–Scott、Schnakenberg 这一类，文献里已知能出 Turing 斑点，Gray–Scott 的斑点还会自我复制（Pearson 1993，凭记忆引用，未在本项目验证）。这条线可以同时接上图灵、隔室和分裂。

**未核对的风险：**

- R1 是三体反应，可能碰到 `three_part_balance_impossible`（G8）。
- 单一全局池会长成富人俱乐部（`one_pool_nested`，G19a），所以池必须是扩散的局部池。
- `PairCRN` 只有成对事件，R1 要写成同格内状态变化或两步反应才可能进入。
- 只靠平衡反应，混匀体系每个守恒类只有一个平衡态；要有两个稳定态，需要边界驱动或空间相分离。

K、W 两族（41 条）这张表不生成，它们是长成后的验收：状态能否分开历史，世界变了增益能否跟上。

## 5. 缺的桥

每一族和上游条件之间缺一条 Lean 边。最小的一组：

1. P2a + 接触相遇 → 准入 ∝ 可用池（把 `shared_pool_marginal_admission_is_anti_dominant` 的 `1−q` 从前提变成结论，也就是“局部 + 守恒 → 竞争”）。
2. P2a + P1a → `export_forced` 的账本前提。
3. P2b + 有限 → 无驱动时收缩（接 C 族）。
4. P1a → K 族的有限编码。
5. `PairCRN` 实例 + 增益条件 → `latch_needs_unit_gain` 的前提。

## 6. 不能分段验收（11-02 用户要求）

规范已写明“可拆分实现，不可拆分验收”，1、5 也“不是先各做完一个独立玩具”。

穿线、定表、发育、浸入语言的工作顺序保留，每次写代码仍是局部的；要求是做每一步时都记得整个系统。隔室、竞争和留住的结构都应该和内容有关，而内容就是语言，所以定表和发育本身就要对语言有帮助。

- **同一个体：** 发育从出生起就浸在真实语言流里，不另做一个只看隔室、竞争或斑图的玩具再接语言。
- **定理：** 桥定理对一般介质或一般 `PairCRN` 陈述，不对为通过而缩窄的实例陈述。
- **系统 4：** 不是当前前置门，但设计不能把它堵死。边界事件从一开始就要能写出去，并影响后面能接触到什么。

**用五系统回看第 4 节的两行表（设计阅读，未运行）：**

| 系统 | 表里已有 | 还缺或有冲突 |
|---|---|---|
| 1 维生 | 守恒含量、回补 | **有冲突：**规范禁止额外的营养池和能量币，语言是唯一食物，信息又不是守恒物料。所以字节不能当物料流入，只能在边界调制速率；驱动从何而来未定 |
| 2 预测 | 无 | **最大缺口：**表里没有字节进入和读出未来的路径 |
| 3 学习 | R2 的用进废退 | 没有误差和后果怎样改变保留；三层信用没有落到反应上 |
| 4 行动 | 无 | 没有边界写出事件 |
| 5 发育 | R1、R2 | 没有时钟、分裂和基因组的写法 |

这张表现在只是系统 5 的草图，还不是候选身体。

## 7. 一个字节的完整循环（设计推导，未运行）

目标：不分别设计五个系统，只写一轮字节循环，再看每个系统是否都落在这一轮里。

**三个决定：**

1. **字节不是物料，是钳制。** 世界把读出粒子放到实际字节的标签上，净含量不变。`PairCRN` 已允许刺激经边界储库势 `μ` 进入，R1/R2 的协议也已这样写。
2. **注入的自由能等于惊讶。** 由 N1 `freeEnergy_eq_kl`，把分布 π 钳到字节 b 注入 `KL(δ_b‖π) = −log π(b)`。这是 N1 的特例，还没单独写进 Lean；物理解释按 N1 的注释要另行论证。
3. **池是空闲单元和空槽，不是养分。** 身体内部封闭，这样不增加能量币。

**四行反应，全部是成对事件：**

| 反应 | 形式 | 作用 |
|---|---|---|
| R1 合取 | 活 + 到达 ⇌ 复合 | 同时保留上下文和新到达，不擦除 |
| R2 转移 | 复合 + 邻 ⇌ 活′ + 到达′ | 合并历史；没被预测的部分成为邻居的到达 |
| R3 占槽 | 复合 + 空槽 ⇌ 占槽 | 学习；有限槽位的竞争使占用接近条件频率 |
| R4 招募 | 复合 + 空闲单元 ⇌ 复合 + 新单元 | 自催化，也就是分裂；不用的单元拆回 |

**一轮的顺序：** 自由松弛得到预测 → 字节钳制注入惊讶 → 经同一张能量表反推出后验 → 绑定、占槽 → 转移。

**预测和后验是同一条定理的两面：** 能量 `E(i,b) = −log(a_i w_ib)` 下，固定内部得到 `π(b|P)`，固定字节得到 `π(i|b) ∝ a_i w_ib`，都是 G13 `freezeSlow_conditional_balance` 的条件平衡。所以第三层 3.4 留下的缺口“w 同时负责写入和读出”不是缺陷，读写共用 w 就是贝叶斯反推。前提是快变量来得及松弛，这一点未证。

**用同一轮回看五个系统：**

| 系统 | 落点 | 风险 |
|---|---|---|
| 1 维生 | 语言流是打破平衡的驱动，也是唯一食物（见 E3 设计第 12 节），被预测的字节和自回声几乎不注入 | 活动靠惊讶维持；预测变好后驱动下降，要靠占槽的亚稳态留住记忆（`latch_needs_unit_gain`） |
| 2 预测 | 读出、合取、合并 | 合并规则怎样局部学到，是全表最大的未知 |
| 3 学习 | R3 当拍学习；复合的寿命是物理窗口；窗口外靠自发活动重放，把重放当到达再走一轮 | 重放的稀疏触发未设计；R3 只给频率，不给 G14 的两相号，需核对是否够用 |
| 4 行动 | 表达就是从读出分布采样并放出；自己放出的字节被预测，几乎不注入；全被预测时驱动消失，只有改变语言才有新驱动，接 `active_closure` | 只留了开口，没有设计 |
| 5 发育 | R4、局部竞争、回收 | 时钟和基因组写法未定；R2 的合并要由发育提供接触 |

**深度：** 边界单元的“到达”是字节，内部单元的“到达”是邻居的状态。规则相同，层次只来自离边界的距离，符合 P3，也接上 JEPA 式的局部预测。没被预测的部分才继续往里走。这一段完全是推理，没有任何测量。

**已有测量给出的底线：** 这轮循环去掉深度和招募，就是动力学滤波器。按字节转移在 β=1、1 M 字节处是 2.37，order 2 是 2.06（单种子）。理想地址下 64 个单元能到 1.94。所以 R1 的合取方向有证据，但只到压缩器一级，不是语言理解。

**核对：**

- 成对事件避开了三体乘积，但仍需放进 `PairCRN` 或局部版本实例化检查。
- Still 等 2012《Thermodynamics of Prediction》说的“耗散对应非预测信息”可能给合并规则一个方向。这是凭记忆引用，未查证。

## 8. 不存字节，存因果（11-02 用户要求，设计推导，未运行）

用户的例子：舍友让倒垃圾，记住的是“要找时间倒垃圾”，不是原话；上课要理解因果结构和公式，不是背老师的话。

**第 7 节有两处与此冲突：**

1. 把 `th 后来了 a` 这样的复合态存进槽位，就是在存字节组合，属于压缩器。
2. “没被猜中的部分才继续往里走”是错的。措辞细节本身很难猜，惊讶很大，这条规则会把噪声一路存进去。惊讶只说明没猜到，不说明有用。

**修正：**

| 时间尺度 | 载体 | 留什么 |
|---|---|---|
| 几拍到几句 | 复合态，寿命短，自然散掉 | 逐字细节，够当下读懂和复述 |
| 合并之后 | 单元状态 | 后续走向相同的历史变成同一状态，措辞在这里消失 |
| 长期 | 占槽 | 只留后来反复帮上预测的转移 |

- **占槽的对象变了：** 只有最外层碰字节。内部的槽记的是状态到状态的转移，不记字节。
- **加固的时间变了：** 到达时只留一个没有内容的参与标记。后来预测兑现时才加固，没兑现就散掉。这是延迟的三因子形式，第三因子是后来的兑现，由到达语言本身给出。
- **合并的标准：** `causal_carrier_contract_iff` 已经给出，读出和后继都相同的历史才能合并而不损失。措辞不同但后面走向相同，就该合并。这也回答了第 7 节留下的问题的方向：合并尽量丢掉对未来没用的历史。怎样只用局部信息判断，仍未解决。

**用 JEPA 代替手工合并（11-02 用户决定）：** 不设计单独的合并判断。内部单元预测邻居的潜状态，预测不到的措辞细节自然留不住；后续走向相同的历史落到相近的潜状态，合并是连续发生的。`causal_carrier_contract_iff` 退回验收角色。最外层仍锚定真实字节，这是已有决定。这也接上第 7 节：内部单元的“到达”就是邻居的潜状态，局部预测误差就是局部自由能。

**JEPA 带来的三个风险：**

- **防塌缩要局部：** SIGReg 是全局统计量，要看整批样本，也不符合局部性。把所有维度压向各向同性高斯，还可能抹平隔室差异；旧主线正是因此回退了转移 SIGReg（`docs/LEJEPA_LEWM_REFERENCE.md`）。候选做法是用有限槽位和单元的局部竞争防塌缩，未证。
- **可预测不等于有用：** JEPA 会丢掉不可预测的，但会留下可预测的垃圾。果蝇线的时间轴 SIGReg 就被内部时钟骗过，核心变成节拍器，有效秩只有 8 到 10。
- **LeJEPA 本身是多视图一致，没有时序预测器：** 时序要用 LeWM 的形式。

**三个仍未解决的问题：**

- **因果的两层：** 只听语言得到的是预测意义上的因果态。世界因果要靠干预（`observational_record_does_not_fix_intervention`），这要到系统 4，或先在自我编辑里对自身做干预。
- **公式：** 公式能用到没见过的内容上，这比合并状态多一层，需要转移规则能作用于新内容。现在的四行反应给不出这一点。
- **自洽：** 同一结论从不同路径重放都能到达，可能是自洽的物理形式。这只是猜想。

**可证伪预测（真实语言）：** 离边界越远，单元状态里关于精确字节的信息越少，关于远期未来的信息相对越多。两段换词不换意的历史，内部状态应比外层更接近。若内部仍主要编码字节，说明这条设计失败。

## 9. 与逆发育合并（11-02 用户要求，设计，未运行）

用户指出第 7、8 节仍是纯信息状态，没有动力学，生成不出大脑，化学因子也不该人为指定。这个批评成立：“活”“到达”“复合”“占槽”是按职责起的名字，没有空间、时钟和生长。

**决定：第 7、8 节改为功能读法，不是机制。** 它们描述长成的组织应该表现出什么，用来验收；生成靠下面的动力学。这组动力学大部分已经存在：`CGB_CORE_EQUATIONS_20261020.md`、`CONSTRUCTAL_GENERATIVE_BRAIN_FRAMEWORK_20261018.md` §6、`CGB_V2_RESULTS_20261020.md`、E1 系列。

| 层 | 变量 | 动力学 | 谁决定 |
|---|---|---|---|
| 物理 | 流、膜电荷、扩散场 | 电荷守恒 E1；`∂M_k/∂t = D_k∇²M_k − M_k/τ_k + 释放` | 定律 |
| 化学 | 无名物种，只有扩散、寿命等物理性质 | 局部质量作用的反应表 | 表由基因组决定，不指定职责 |
| 基因 | 身份码 c = 空间种子 × 二分位 × 时间计数 | 分裂、迁移、分支生长、表达 `ρ_k(c)` | 短基因组 |
| 经验 | 结构 g、痕迹 z、留存 S | 连接 `N_ij = E_ij · σ(cᵢᵀAcⱼ) · S_ij` | 语言 |

**化学因子不人为指定：** 世界只给物种，不给名字。像多巴胺的东西，是某类细胞释放、别的细胞按受体读到的物种，职责由发育和经历形成。同一物种在不同受体上含义不同。果蝇证据：半谱系预测递质，纯度 0.877，说明递质身份是发育程序的输出。

**核心方程里的酶池模板也是人写的。** 处理方式：把 V2 的模板当初始基因组，不当规则，让它能被改动。这样不必从随机表出发，随机表的路已经失败过：生成个体 s0、s1 都没有记忆，还有赢家诅咒。

**逆发育提供什么：** 不提供果蝇的接线，提供生成器的形式：时钟序列、二分位分裂、相遇 × 匹配、身份决定生理。数字在语言里重新定。

**语言怎样进入每一层：** `N_ij` 三项里，相遇和匹配来自基因组，留存 S 来自经历。所以隔室和竞争长成什么样，本来就由语言决定。分裂从第一次起就有字节在边界上。

**第 7、8 节的功能在动力学里的落点（V2 证据）：**

| 功能 | 动力学形式 | 证据 |
|---|---|---|
| 预测邻居（JEPA） | 先后放电的时序不对称，来自钙加两个酶池 | T2 长出 STDP 方向，未写规则；有三项手工扫描的参数 |
| 局部防塌缩 | BCM 滑动阈值、池耗竭、共享耗竭池的去相关压力 | T2、T4c、T4d |
| 预测残差 | 抑制在膜上抵消输入，净电流就是残差 | T3b：B/A 1.34，未达预注册的 1.5，且是节律输入，不是语言 |
| 只留有因果作用的 | 标签捕获 | T4c：相关组/随机组 1.08 到 1.80，未达 2 倍 |

**风险：**

- 这组方程从未在语言上运行过：V2f 没有语言训练。V0 是另一个简化结构（哈希码加 k-WTA），它比 bigram 差不能算到这组方程头上，此处更正。
- E3（基因组在介质里生长）从未开始，E2 第一轮不敏感。
- 营养因子要是局部化学物质，不能变成规范禁止的全局货币或营养池。核对后才能用。
- 基因组由谁来选未定。选择很贵。第一版可以只用一个固定基因组，不做进化。
- 要分清两件事。**赢者通吃**（富人俱乐部）已由 G22 解决：局部有限储库给出不应期、疲劳、短时程抑制和缩放，G22c 说明储库必须是局部的。**赢家诅咒**（选择把运气当能力）是统计问题，G22 管不到；旧实验的病因是少数固定训练世界。对策是每代用新文本、基因组保持很短，`mixture_regret_le` 的遗憾界提示描述越短越难过拟合，这一点只是类比，选择单个赢家的版本未证。
- 人类语言不是果蝇的世界，果蝇的数字不能带过来。
- 必须满足手机规模，事件驱动。
- 执行前要过 Lean 准入和 `scripts/cgb_gate.py`。
