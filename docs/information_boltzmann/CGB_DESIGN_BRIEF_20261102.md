> **历史阶段记录，已原位归档（2026-11-05）。** 本页保留对应版本的协议、结果和当时判断；当前执行状态与下一步以[平台里程碑](CGB_MILESTONE_CLOSED_LOOP_20261105.md)为准。

# 设计底本：定义、主链、五系统与必要定理

写给你亲自设计用。定义和主链是原文，一字未改，来源是 [CGB_AXIOMS_V0_20261022.md](CGB_AXIOMS_V0_20261022.md) §4.1、§4.1.1 和 [活规格](CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md) §5。定理的白话说明是我对着 Lean 签名逐条重写的，因为索引里有几条描述不准。

这是一份阅读底本，不是新的权威来源。和原文件冲突时，以原文件为准。

**设计还没有开始。** 到 11-02 为止的全部实验都是我跑的诊断，没有一项是身体决定。第 8、9 节是留给你的。

定理的取舍：全部 1700 条里，分类为“必要”的 62 条，加上门引用的，共 95 条，全部列在第 6 节。另外加了 10 条主链上绕不开的定理，一共 105 条。其余定理是候选机制的细节、见证或语义定义，这里不列。

**先看结论：**
- 定理大多在说**不能怎样**，几乎不说**应该怎样**。任何一条都没有规定柱、r×r×r、GDN 或 Transformer。
- 最硬的约束只有一条：状态必须既能读出需要的东西，又能自己递推（`causal_carrier_contract_iff`）。
- 能力没有任何定理管。`arbitrary_driven_tables_admitted` 已经证明满足公理什么都不保证。
- L1 自举目的、L5 生态位、工作量 O(1)、泛化，这四样都没有根定理。
- 第 8 节的 10 件事里，第 1、7 件你已定，还剩 8 件。第 2、3、8 件是现有文档或规则互相矛盾。
- 我写的动力学方程目前不合规，见第 8 节第 8 件。

## 1. 无限流生命体定义（原文）

> 无限流生命体：一个具有有限状态空间和有限计算资源的开放系统，持续暴露于近乎无限的信息流中。它通过自举，自发形成并维持以自身存在为核心的内在目的，利用自身有限的内部自由度，通过记忆、学习、信息压缩等过程形成一定程度的智能，并利用这些智能形成关于周围世界的局部世界模型，据此预测未来、选择行动，主动改变自身与环境之间的相互作用关系，形成操作闭合，进而维护自身生存状态，构建并维持自身生态位，使自身状态保持在由自身定义的可存活区间之内。

定义逐句在体系里的身份：

| 定义中的条件 | 身份 | 现在有什么 |
|---|---|---|
| 生存是生命体的第一需要 | L0，生命定义公理 | 定义 |
| 有限状态空间、有限计算资源 | 状态有限由 P1a 推出；计算资源没有推出 | 状态有定理，计算没有 |
| 开放系统，持续暴露于无限流 | P3 只限定边界感知，流的存在是世界前提 | 世界条件 |
| 自举形成内在目的 | L1，目标定理 | **没有根定理**，只有条件性自我后果模型 |
| 记忆、学习、压缩形成智能 | L2，目标定理 | 必要条件很多，能力没有定理 |
| 局部世界模型，预测未来 | L3，目标定理 | 条件定理 |
| 选择行动，操作闭合 | L4，目标定理 | `active_closure`，依赖世界条件 |
| 构建并维持生态位 | L5，目标定理 | **没有根定理** |
| 保持在自身定义的可存活区间 | L6，目标定理 | 条件合同 |

原文的判断：“活着不是公理，是整个体系的主定理（存在性猜想）”，即存在满足 P1–P3 的程序，使系统满足 L1–L6。这条还没有证出来。

## 2. 三原则（原文要点）

- **P1 有限：信息是物理的。** P1a 状态有代价：可区分的状态都要占用含量，单元容量有上界，恰好一个真空态。P1b 作用要接触：事件只读、只改参与者的状态与它们接收的刺激，参与者两两接触。
- **P2 对称：封闭时定律的不变性。** P2a 守恒，P2b 时间反演，P2c 结构等变（共享规则的约束）。
- **P3 对称破缺：只有三个来源。** 外边界、身份与位置、历史。“除此之外不允许任何手设的不对称，例如预先指定的模块或区域职责。”

## 3. 实现主链（原文，用户确认 2026-10-31）

> 生存是生命体的第一需要：失去可存活性即退出生命过程；这不是外部奖励或预先写入的“自我目的”。在守恒、有限容量、持续演化与近乎无限信息流共同成立时，封闭状态会丧失对未来仍有用的可区分结构，因此生命必须作为开放系统从边界持续吸收信息负熵以维持自身。这里的负熵是同一有限状态中可用于后续语言/世界交互的有效预测、记忆与压缩结构，不另设食物、能量币或储备账户。
>
> 但可摄入的新结构会被学尽，环境也会变化；被动接收不足以持续维持可存活性。因此系统必须主动适应、改造或转换与环境的相互作用，以获得新的可用信息。主动推理由此要求预测、因果关系与自我模型；自举形成的特异性记忆、遗忘、压缩和学习结构，随后反过来维护生存。有限容量又迫使选择性保留、代谢、遗忘与压缩；无限流要求这些更新的工作量不随既往经历长度增长。延迟后果下的归因由此要求广义资格迹，并进一步约束隔室、连接与三因子学习机制。

```text
L0 生存连续性
  → 守恒 + 有限容量 + 持续演化 + 无限流
  → 封闭状态丧失未来仍有用的结构区别
  → 开放边界持续吸收信息负熵
  → 环境变化、已学尽的结构与世界响应差异
  → 主动适应、改造或转换交互
  → 预测、因果、自我模型与操作闭合
  → 选择性保留、代谢、遗忘、压缩、学习
  → 与经历长度无关的递推更新
  → 延迟归因的广义资格迹、隔室、连接与三因子学习
```

原文注明：“这是一条生成依赖链，不是功能模块清单。”

## 4. 五个系统（原文）

| 系统 | 共同状态中的职责 | 依赖关系 |
|---|---|---|
| 1 维生—稳态 | 储备、耗散、可存活区间与恢复 | L0 底座；约束所有其余更新 |
| 2 感知—预测—压缩 | 语言摄入、未来区别与局部世界关系 | 与学习可塑性共同形成 |
| 3 记忆—学习—可塑性 | 选择保留、资格迹、三因子和局部增益 | 读取预测误差与已到达后果 |
| 4 行动—生态位 | 表达、提问、改变后续可达语言 | 建立在维生、预测和信用之上 |
| 5 发育—组织 | 柱、连接、竞争和分化的可达结构 | 从出生起约束全部状态，不预标功能区 |

工程优先级（原文）：“系统 2、3 是能力核心，必须保持并超过现代语言模型的说话、编程、长周期任务与科学推理能力；系统 1、5 是让这套能力可无限流持续运行的底座，不得各自退化为独立玩具。系统 4 建立在 1–3 的闭环之上。”

这五个是同一动力学的投影，不是五个模块。五行是另一套分析框架，不和这五个系统一一对应。

```text
        ┌──────── 1 维生：有限、耗散、恢复 ────────┐
        │                                          │
        │     2 感知预测  ⇄  3 记忆学习            │   → 4 行动生态位
        │     （同一状态，共同形成）               │       （后接）
        │                                          │       │
        └──────── 5 发育：局部、竞争、分化 ────────┘       │
                                                           │
             4 改变世界 → 新语言回到 2、3，并维持 1  ←─────┘
```

## 5. 五系统之间的关系，以及定理依据

活规格只写了依赖方向。下表把定理按“哪个系统逼出哪个系统的什么”归类，**这个归类是我做的**，请你判断。

| 关系 | 它说的是 | 依据 | 状态 |
|---|---|---|---|
| 1 → 全部 | 有限和耗散约束一切更新 | `no_uniform_positive_lower`、`export_forced` | 有定理 |
| 1 → 3 | 容量有限，所以必须选择地留、选择地忘 | `retention_needs_selectivity`、`unselective_export_forgets` | 有定理 |
| 1 → 2 | 摄入靠预测，预测准才有“食物” | `predictive_growth_contract` | 有定理，世界条件 |
| 2 ⇄ 3 | 预测和记忆是同一状态的两面：能读出，又能递推 | `causal_carrier_contract_iff` | **最硬的一条** |
| 2 ⇄ 3 | 语言有交互，线性读写碰不到，要有非线性地址或读取 | `interaction_needs_nonadditive`、`linear_recurrence_no_interaction` | 有定理 |
| 2 → 3 | 预测误差是学习信号，形式被迫是“到达 − 预测” | `phases_must_balance` | 有定理 |
| 3 → 2 | 合并历史有分数下限，记不住就预测不好 | `merged_histories_binary_logloss_floor` | 有定理 |
| 3 → 1 | 经历变成储备，可用恢复性来测 | `future_reserve_recoverable_iff` | 绑定 GDNLife |
| 5 → 全部 | 只能局部作用，外部只从边界进入 | `modulation_needs_boundary` | 有定理 |
| 5 → 3 | 增益必须局部，防垄断要读局部需求，要双侧预算 | `no_global_gain_for_heterogeneous_compartments`、`local_anti_dominance_requires_demand_readout`、`column_budget_allows_wta` | 有定理，模型受限 |
| 5 自身 | 分化需要对称破缺来源 | `symmetric_stays_symmetric` | 有定理 |
| 3 → 5 | 连接和隔室随经历生长 | 无 | **没有必要性定理** |
| 1 ⇄ 5 | 资源怎样进入竞争才防住塌缩 | 无；实测见第 10 节 | **没有定理**，账户形式不合规 |
| 1 → 4 | 学尽和漂移逼出主动 | `viable_not_eventually_passive`、`active_closure` | 依赖世界条件 |
| 2 → 4 | 预测必须以动作为条件 | `action_condition_required` | 有定理，世界条件 |
| 3 → 4 | 延迟动作的信用要状态分开 | `delayed_action_credit_requires_state_separation` | 有定理 |
| 4 → 2 | 被动观测定不了干预结果，只读不说学不全因果 | `observational_record_does_not_fix_intervention` | 有定理 |
| 4 → 1 | 行动改变长期可达语言，维持生存 | 无 | **L5 没有根定理** |

## 6. 逐环定理

范围的意思：**普适**对任何身体成立；**世界条件**在世界满足前提时成立；**绑定模型**只在某个具体模型上证明，原理可能更一般，但不能直接外推。门是第 13 节的义务编号。

### 环 1：生存连续性与有限身体

主链起点：`L0 生存连续性 → 守恒 + 有限容量 + 持续演化 + 无限流`。这一环说明身体是有限的，而且活着是一种会失去的状态。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `finite_reachable`<br><sub>Medium</sub> | 含量不超过任一界的局部状态只有有限个，且事件守恒含量时，从任何初态出发能到达的状态集合是有限的。 | 语义·普适 | O01 | 身体有限是推出来的，不是预算选择。 |
| `no_uniform_positive_lower`<br><sub>GenerationSurvivalBounds</sub> | 每步至少按固定比例衰减的量，不可能永远高于任何正阈值。 | 必要·普适 | — | 没有维护，任何结构都会跌破存活线。 |
| `trap_irrecoverable`<br><sub>GenerationLatchHealth</sub> | 一个状态集合对个体自身事件封闭，且里面没有任何状态能摄入，那么集合里每个状态都无法靠自己恢复。 | 必要·绑定模型 | — | 必须避开吸收态：满池只读、活动饱和锁死、完全静默，都要有出路。 |
| `historical_persistence_requires_alive_change`<br><sub>GenerationConservedMemory</sub> | 如果静息动力学不改变存活指标的期望，这个系统就不满足“维持”的定义。维持要求：有输入时活着，短暂切断能扛住，长期切断会死，什么都不做也会死。 | 必要·绑定模型 | — | 活着必须是可以失去的。一个放着不动也永远好好的系统，不算在维持。 |
| `conserving_calm_needs_organizational_viability`<br><sub>GenerationConservedMemory</sub> | 静息事件都守恒总含量、存活判据又只读总含量时，同样不满足“维持”。 | 必要·绑定模型 | O08 | 存活判据不能只看总量，必须读组织结构。 |
| `no_sustained_use_on_finite_supply`<br><sub>EnergyBudget</sub> | 等温模型里，自由能有下界、累计供给有上界时，不可能永远保持每拍至少 ε 的平均做功和耗散。 | 必要·世界条件 | — | 持续运行需要持续供给，语言这条流不能断。 |

### 环 2：封闭会死，必须开放

主链：`封闭状态丧失未来仍有用的结构区别 → 开放边界持续吸收信息负熵`。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `not_persists_without_incoming`<br><sub>MaterialExchange</sub> | 账本上外部流入恒为零时，任何阈值下都不能持续维持。 | 必要·世界条件 | — | 开放是必要条件。 |
| `actualRecover_zero_without_intake`<br><sub>WorldRecovery</sub> | 所有状态都无法摄入时，任意有限期限内的恢复概率为零。 | 语义·普适 | — | 恢复能力来自摄入能力，不能只靠内部整理。 |
| `exposed`<br><sub>Life</sub> | 满足旧版 `Lives` 条件的介质，必有事件的速率或作用依赖感官刺激。 | 语义·普适 | — | 活体的动力学必须真的读输入。 |
| `modulation_needs_boundary`<br><sub>Generation</sub> | 在局部性和“只有边界感知”下，同一转移的速率如果随刺激变化，它的改变区域或接触范围必定碰到边界。 | 必要·普适 | — | 没有全局广播。调制和第三因子只能从边界经局部接触传进去。 |
| `predictive_growth_contract`<br><sub>InformationEconomics</sub> | 增长率精确等于 `log|B| − 交叉熵`，上限是 `log|B| − 熵`。 | 设计·世界条件 | O18 | 语言的“食物”就是预测：预测越准摄入越多，上限是语言本身的熵。不用另设食物账户。 |
| `no_allocation_persists`<br><sub>GenerationPredictiveHarvest</sub> | 如果世界的熵超过 `log|B| − m`，任何预测分配的增长都低于维持成本 m。 | 必要·世界条件 | O18 | 世界太随机或成本太高时，任何模型都活不下来。这是世界的上限，不是模型的问题。 |

### 环 3：负熵就是可读、可递推的未来区别

主链把负熵定义为“同一有限状态中可用于后续交互的有效预测、记忆与压缩结构”。这一环是整套理论里约束最硬的部分：记忆到底是什么，这里说清楚了。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `history_readout_exists_iff`<br><sub>LanguageLifeKernel</sub> | 存在准确读出，当且仅当编码相同的历史要求相同的输出。 | 必要·普适 | O06 | 状态必须把需要不同答案的历史分开。 |
| `history_transition_exists_iff`<br><sub>LanguageLifeKernel</sub> | 只靠当前状态和输入就能递推，当且仅当编码相同的历史在同样输入后仍然编码相同。 | 必要·普适 | O05 | 这是“不用翻历史”的精确条件，也是无限流可运行的形式。 |
| `causal_carrier_contract_iff`<br><sub>InformationEconomics</sub> | 同时存在读出和递推，当且仅当上面两条一起成立。 | 必要·普适 | O06 | **这份文档里最重要的一条。** 记忆的定义不依赖任何载体：状态既能答出需要的东西，又能自己往前走。2 和 3 共用这一条。 |
| `merged_history_preserves_read_and_successor`<br><sub>GenerationColumnRefinement</sub> | 上一条在 `ColumnRefinement` 身体上的形式：合并的历史必须读出相同、后继相同。 | 设计·绑定模型 | O02 | H/C/S 必须从同一身体读出。这条绑定柱接口。 |
| `merged_histories_binary_logloss_floor`<br><sub>GenerationSelfProduction</sub> | 两段历史要求相反的二值结果却被编码成同一状态，平均 log 损失至少 log 2。 | 必要·普适 | O06 | 遗忘有明确代价：每合并一对该分开的历史，分数就有一个下限。 |
| `separated_histories_capacity`<br><sub>LanguageLifeKernel</sub> | 如果要在误差 ε 内服务一批历史，而它们两两在某个后续上相差超过 2ε，那么编码必须单射，状态数至少等于历史数。 | 必要·普适 | O03 | 容量下界由“要分开多少种历史”决定，和用什么载体无关。这是预算的理论依据。 |
| `record_state_lower_bound`<br><sub>LanguageLifeKernel</sub> | 一个具体例子：要同时读出决策动作和维持成本，感知记录至少需要 16 个状态。 | 必要·普适 | — | 上一条的具体算例，不是新约束。 |
| `observed_memory_information_le`<br><sub>GenerationConservedMemory</sub> | 粗粒化读出能看到的记忆信息，不超过完整状态里的记忆信息。 | 必要·绑定模型 | O08 | 读出看不到不等于没有，但读出看到的一定在状态里。 |
| `loses_future_language_distinction_iff_not_preserved`<br><sub>GenerationGDNLife</sub> | 合并了此前可区分的未来语言签名，等价于储备没有保住。模型绑定 GDNLife。 | 必要·绑定模型 | O06 | 耗散的定义：不是数值变小，而是未来的区别没了。 |
| `future_reserve_recoverable_iff`<br><sub>GenerationGDNLife</sub> | 能从更新后的签名恢复更新前的签名，等价于储备保住了。模型绑定 GDNLife。 | 必要·绑定模型 | O06 | 储备可以用“能不能恢复”来测，不需要储备账户。 |
| `external_causal_readout_exists_iff`<br><sub>LanguageLifeKernel</sub> | 带外部检索时，存在准确读出当且仅当内部状态和检索结果相同的情形要求相同输出。 | 必要·普适 | — | 外部存储是允许的，但判据不变：检索结果和内部状态合起来必须分得开。 |
| `external_trace_requires_retrieved_distinction`<br><sub>LanguageLifeKernel</sub> | 内部状态相同、所需信用不同时，检索出来的外部记录必须不同。 | 必要·普适 | — | 外部记录若要代替内部记忆，必须真的带上那条区别。 |
| `precision_distinguishes_equal_mean_histories`<br><sub>GenerationSelfProduction</sub> | 均值相同、精度不同的两个记忆，收到同一观测后均值不同；只看均值的转移函数做不到。 | 必要·普适 | O16 | 只存“值”不够，还要存“这个值有多少证据”。不限定用什么形式存。 |

### 环 3b：非线性是必要的（2026-11-02 新增）

语言到处是交互：`th` 后面是什么，不等于 `t` 的预测加 `h` 的预测。这组定理说，线性结构碰不到这种交互。真实文本上前提成立：两段不重叠的 100 万字节里，` a/ n/ia/in → n` 的对数几率交互对比约 4.0–4.1 nats。这类对比含零计数格，数值随平滑变化，只能当量级看。这组定理已通过 `lake build`，还没有跑全量审计。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `additive_no_interaction`<br><sub>GenerationNonlinearity</sub> | 两个来源分开作用的映射，交互对比 `f(a,b)+f(a',b')−f(a,b')−f(a',b)` 恒为零。 | 必要·普适 | — | 这是核心事实。 |
| `interaction_needs_nonadditive`<br><sub>GenerationNonlinearity</sub> | 目标有非零交互对比时，任何可加映射都不能精确表示它。 | 必要·普适 | — | 必要性本身。 |
| `linear_read_of_superposed_state`<br><sub>GenerationNonlinearity</sub> | 任意宽度的线性读出作用在叠加状态上仍然可加。 | 必要·普适 | — | 加宽没用。 |
| `linear_stack_superposes`<br><sub>GenerationNonlinearity</sub> | 任意深度的线性层加偏置，都保持两个来源叠加。 | 必要·普适 | — | 下一条的引理。 |
| `linear_stack_no_interaction`<br><sub>GenerationNonlinearity</sub> | 线性栈的线性读出没有交互。 | 必要·普适 | — | 加深也没用。 |
| `linear_recurrence_no_interaction`<br><sub>GenerationNonlinearity</sub> | 线性递推记忆加线性读出，对两个过去的到达可加，嵌入是任意查找表也一样。 | 必要·普适 | — | 学到的嵌入救不了线性记忆。 |
| `additive_logit_cannot_separate`<br><sub>GenerationNonlinearity</sub> | 可加的对数几率取不到 XOR 的符号模式。 | 必要·普适 | — | 末端的 softmax 救不了，它不算有效的非线性。 |
| `additive_error_floor`<br><sub>GenerationNonlinearity</sub> | 可加预测器在四种情形上的平方误差和，不低于对比平方的四分之一。 | 必要·普适 | — | 这个下限和身体规模无关，多大都一样。 |
| `association_read_additive_in_contents`<br><sub>GenerationNonlinearity</sub> | 固定查询时，GDN 式读出对存储内容可加。 | 必要·绑定模型 | — | 检索只能加权，不能把取到的东西组合起来。这是 GDN 要配 MLP 的原因。 |
| `key_query_read_interacts`<br><sub>GenerationNonlinearity</sub> | 一个到达给键、另一个给查询时，读出有交互。 | 见证·绑定模型 | — | 学到的地址本身就是非线性，所以非线性不一定要 MLP。 |
| `one_rectifier_reads_interaction`<br><sub>GenerationNonlinearity</sub> | XOR 有交互，一个整流单元就够读出。 | 见证·普适 | — | 必要性是紧的：要的是有非线性，不是非线性要多复杂。 |

### 环 4：结构会学尽，环境会变

主链：`环境变化、已学尽的结构与世界响应差异`。这一环解释为什么被动学习不够，也解释为什么学习率不能归零。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `finite_body_stationary_loss_eventually_constant`<br><sub>GenerationLanguageExistence</sub> | 有限状态的轨迹上，如果损失一直不增，有限步之后一定恒定。 | 设计·绑定模型 | O17 | 有限身体的学习一定有平台。`context_cells` 在 1.52 附近停住，和这条方向一致。 |
| `strict_descent_has_no_finite_exact_readout`<br><sub>GenerationLanguageExistence</sub> | 损失由有限物理状态精确读出时，不可能永远严格下降。 | 设计·绑定模型 | O17 | “永远进步”在有限身体上不存在。 |
| `recurrent_gain_excludes_eventual_stationary_descent`<br><sub>GenerationLanguageExistence</sub> | 有限状态下，如果损失无穷多次严格下降，就不可能最终单调不增，也就是它必须无穷多次回升。 | 设计·绑定模型 | O17 | 持续有收获的身体一定持续在遗忘。回升不是故障。 |
| `precision_binary_sample_gain_not_future_guarantee`<br><sub>GenerationSelfProduction</sub> | 一步更新让这个样本的损失下降，但对已校准的预测，未来期望损失不会因此变好。 | 设计·普适 | O17 | 当拍分数变好不等于学到东西。验收必须看未来。 |
| `candidate_erasure_not_unbounded_signal`<br><sub>GenerationSelfProduction</sub> | 有限候选集上，靠消除候选得到的信号累积有界。 | 必要·世界条件 | — | 从固定假设集里榨取的信息是有限的，这就是“学尽”。 |
| `frozen_hold_loses_on_drift`<br><sub>GenerationContinualOptimizer</sub> | 误差每步增长 q>0 时，冻结的估计终将超过任何阈值。 | 必要·世界条件 | — | 环境漂移时，不学习必输。 |
| `vanishing_gain_loses_under_drift`<br><sub>GenerationGradientFilter</sub> | 环境持续漂移时，增益长期小于 q/(4M) 的跟踪方差终将超过 M。 | 必要·世界条件 | — | 学习率不能归零。 |
| `beaten_forever_iff_never_settles`<br><sub>GenerationLearning</sub> | 阈值觅食模型里，每个固定规则永远会在某处落后，当且仅当环境均值永不稳定。 | 必要·世界条件 | — | 学习永远有必要，当且仅当世界永远在变。 |
| `innate_loss_quarter`<br><sub>GenerationLearning</sub> | 两个环境里，任何固定阈值都至少在一处损失两均值差的四分之一。 | 必要·世界条件 | — | 先天固定规则有下界损失，这是学习的最低理由。 |
| `history_useless_of_independent`<br><sub>GenerationLearning</sub> | 未来和历史独立时，按历史选规则不会超过固定规则的上界。 | 必要·世界条件 | — | 记忆的价值来自世界的结构。世界没有结构，记忆就没用。 |

### 环 5：必须主动

主链：`主动适应、改造或转换交互`。注意这些定理的共同前提：**被动的累计摄入有界**。这是世界条件，不是证出来的。系统 4 现在延期，但这一环决定它迟早要接。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `not_unbounded_of_bounded`<br><sub>GenerationActiveInference</sub> | 累计信号有上界，就不是无界信号。 | 必要·世界条件 | — | 下面几条的基础步骤。 |
| `viable_not_eventually_passive`<br><sub>GenerationActiveInference</sub> | 生存需要无界累计信号，而最终被动的策略只有有界信号时，存活策略不能最终被动。 | 必要·世界条件 | O20 | 长期活着就不能永远只听不说。 |
| `active_closure`<br><sub>GenerationActiveInference</sub> | 上一条加上世界对动作敏感时：策略不能最终被动，预测也不能忽略动作。 | 必要·世界条件 | O20 | 主动闭合的总结论，覆盖 L4。 |
| `counterfactual_passive_bound_forces_activity`<br><sub>GenerationSelfProduction</sub> | 反事实地看，如果所有最终被动的动作序列摄入都有界，而实际轨迹摄入无界，实际策略就不是最终被动。 | 必要·世界条件 | — | 判据是反事实的：要比较“换成被动会怎样”。 |
| `recurrent_external_evidence_forces_activity`<br><sub>GenerationGDNLife</sub> | 被动后不再有有用外部证据，而有用证据又反复出现时，策略不是最终被动。模型绑定 GDNLife。 | 必要·绑定模型 | — | 上面结论在语言证据上的版本。 |
| `recurrent_external_information_forces_activity`<br><sub>GenerationGDNLife</sub> | 被动后外部信息为零，而正信息反复从外部到达时，策略不是最终被动。模型绑定 GDNLife。 | 必要·绑定模型 | O20 | 同上，用信息量表述。自听不计入外部。 |

### 环 6：预测、因果、自我模型

主链：`预测、因果、自我模型与操作闭合`。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `action_condition_required`<br><sub>GenerationActiveInference</sub> | 世界对动作敏感时，不看动作的预测器不可能处处准确。 | 必要·世界条件 | O20 | 预测必须以动作为条件。 |
| `silence_condition_required`<br><sub>GenerationActiveInference</sub> | 沉默字节的概率随动作变化时，同样如此。 | 必要·世界条件 | O20 | 沉默也是后果，预测要算上它。 |
| `observational_record_does_not_fix_intervention`<br><sub>LanguageLifeKernel</sub> | 两个规则在所有观测里相同，但在一次干预下不同。 | 见证·普适 | O20 | 被动读文本永远学不全因果。语言身体只读不说时，这一层是空的。 |
| `action_distinction_preserved`<br><sub>GenerationSelfProduction</sub> | 两个动作的后继不同时，准确的后果模型必须区分它们。 | 必要·世界条件 | — | 自我后果模型是必要的，不用单独设模块。 |
| `open_world_body_prediction_iff`<br><sub>GenerationSelfProduction</sub> | 只读身体和动作就能精确预测，当且仅当隐藏世界的差别不改变后继。 | 必要·世界条件 | O20 | 世界有隐藏状态时，身体不可能精确预测。局部世界模型只能是局部的。 |
| `indistinguishable_joint_worlds_cannot_both_stay_viable`<br><sub>GenerationSelfProduction</sub> | 两个世界对身体完全一样，却要求不同动作才能活时，一个固定策略不可能在两个世界都活。 | 必要·世界条件 | O20 | 身体必须分开“需要不同对待”的世界。 |
| `no_safe_exploration`<br><sub>CausalLearning</sub> | 每个动作都可能被某个剩余假设判为不安全时，不存在稳健安全的动作。 | 必要·世界条件 | — | 探索一定有风险，死亡是正常结果。 |
| `preparation_uses_only_elapsed`<br><sub>GenerationGDNLife</sub> | GDNLife 的预测准备只依赖经过时间，不依赖到达的字节。 | 设计·绑定模型 | O04 | 先预测后到达的形式。绑定 GDNLife，但要求是通用的。 |
| `step_source_irrelevant`<br><sub>GenerationGDNLife</sub> | GDNLife 里来源标签不改变下一状态。 | 设计·绑定模型 | O19 | 这是一条反例：要区分自己和外界，来源必须写进状态。 |
| `self_not_externallyUseful`<br><sub>GenerationGDNLife</sub> | 自听字节按定义不算有用的外部证据。 | 设计·绑定模型 | O19 | 自己说的话可以用来整合，不算新信息。 |

### 环 7：选择保留、代谢、遗忘、压缩

主链：`选择性保留、代谢、遗忘、压缩、学习`。分四组：流量、容量、顺序、保持与释放。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `export_forced`<br><sub>GenerationExport</sub> | 内容守恒且容量有界时，累计摄入不超过累计排出加容量。 | 设计·绑定模型 | O11 | 想持续吸收就必须持续排出。 |
| `write_only_intake_bounded`<br><sub>GenerationExport</sub> | 只进不出时，累计摄入不超过容量。 | 设计·绑定模型 | O11 | 没有遗忘的身体会饱和。 |
| `retention_needs_selectivity`<br><sub>GenerationExport</sub> | 内容始终有正的存活下界时，它的删除概率不可能永远高于同一个正数。 | 设计·绑定模型 | O11 | 想留住某样东西，就必须让它比别的东西更不容易被删。 |
| `unselective_export_forgets`<br><sub>GenerationExport</sub> | 每次删除概率都有正下界时，内容的存活量会低于任何阈值。 | 设计·绑定模型 | O11 | 不选择的遗忘什么都留不下。 |
| `selective_write_contract`<br><sub>InformationEconomics</sub> | 候选消除模型里，与证据相容的到达不写入也不付费；有储备且候选被反驳时，删除它并付一次费。 | 设计·世界条件 | O11 | 没带来新信息的到达不该改变状态。绑定候选消除模型。 |
| `exact_key_capacity`<br><sub>GenerationAssociationCapacity</sub> | 线性无关的键数量不超过键维度。 | 必要·普适 | O10 | 线性记忆里精确存储的条数受 r 限制。这是 S 为什么要大的理由，也是线性记忆的上限。 |
| `exact_decoder_capacity`<br><sub>GenerationAssociationCapacity</sub> | 存在精确解码器时，存储条数不超过维度。 | 必要·普适 | O10 | 同上。 |
| `collision_of_excess_keys`<br><sub>GenerationAssociationCapacity</sub> | 键数超过维度时，一定存在两组不同内容写出相同状态。 | 必要·普适 | O10 | 超容量必然混叠。只对线性叠加成立，非线性或离散地址不受这条限制。 |
| `association_readout_interference_bound`<br><sub>GenerationAssociationCapacity</sub> | 关联读出的误差受键和查询的交叉相关控制。 | 设计·绑定模型 | O10 | 串扰的量化上界，选地址时用。 |
| `assimilate_order_blind`<br><sub>GenerationContextAssociation</sub> | GDNLife 的写入对到达顺序不敏感：置换到达序列，状态完全相同。 | 必要·绑定模型 | O07 | **只针对 GDNLife。** 这是我们翻译版的缺陷，真正的 GDN-2 没有这个问题。 |
| `no_read_separates_order`<br><sub>GenerationContextAssociation</sub> | 所以任何读出都分不开这种状态里的两种顺序。 | 必要·绑定模型 | O07 | 语言依赖顺序，写入必须带上下文。 |
| `context_write_separates_order`<br><sub>GenerationContextAssociation</sub> | 写入地址依赖此前的字节时，两种顺序写出不同状态。 | 设计·绑定模型 | O07 | 顺序问题的一种修法，不是唯一的。 |
| `order_separated`<br><sub>GenerationContextBody</sub> | 上一条在 `context_cells` 上的实例。 | 设计·绑定模型 | O07 | 只说明这个身体分得开顺序。 |
| `histories_merge`<br><sub>GenerationLinearMemory</sub> | 每步历史差距至多乘 α 时，T 步后的差距不超过初始差距乘所有 α 的积。 | 设计·绑定模型 | O09 | 收缩系统的记忆视界有上限。 |
| `common_suffix_address_bound`<br><sub>GenerationContextAssociation</sub> | 收缩的上下文递推里，共同后缀越长，两个状态越接近，速度是 ρ 的长度次方。 | 设计·绑定模型 | O14 | 相同的最近文本会把状态拉到一起。这是好事（可重复）也是限制（老信息会丢）。 |
| `held_coordinate_survives_suffix`<br><sub>GenerationContextAssociation</sub> | 某个坐标在一段输入下不变时，它上面的区别能穿过整段输入。 | 设计·绑定模型 | O14 | 长记忆要靠不被输入冲刷的坐标。 |
| `contraction_unique_rest`<br><sub>GenerationPersistence</sub> | 严格收缩只有一个不动点。 | 设计·绑定模型 | O14 | 处处收缩的动力学不可能保持多个状态。 |
| `contraction_has_no_two_latches`<br><sub>GenerationContextAssociation</sub> | 上一条的上下文版本：严格收缩不可能有两个自锁状态。 | 设计·绑定模型 | O14 | 同上。 |
| `latch_needs_unit_gain`<br><sub>GenerationPersistence</sub> | 可微函数有两个不动点时，它们之间必有一点导数等于 1。 | 设计·绑定模型 | O14 | 锁存必须有增益达到 1 的区域。 |
| `no_latch_below_unit_gain`<br><sub>GenerationColumnMemory</sub> | 1-Lipschitz 且固定零点的非线性配上小于 1 的增益时，唯一不动点是零。 | 必要·普适 | O14 | 增益小于 1 就锁不住。 |
| `latch_above_unit_gain`<br><sub>GenerationColumnMemory</sub> | 增益至少为 1 的硬饱和反馈有 −1、0、1 三个不动点。 | 设计·绑定模型 | — | 上一条的另一面，说明锁存确实可以有。 |
| `silent_rest_unique`<br><sub>GenerationColumnMemory</sub> | 无写入的 GDN 更新在 α<1 时唯一不动点是零。 | 必要·绑定模型 | O14 | 没有输入时 GDN 的记忆终将归零。绑定 GDN 更新。 |
| `linear_hold_keeps_error`<br><sub>GenerationLinearMemory</sub> | 线性映射原样保持信号时，也原样保持沿信号方向的误差。 | 设计·绑定模型 | O09 | 线性记忆不能同时保持和纠错，纠错要靠非线性。 |
| `bank_power_law_lower_bound`<br><sub>GenerationCascadeMemory</sub> | 有限记忆池里有时间常数落在 t 到 γt 之间时，总轨迹有约 1/t 的下界。 | 必要·绑定模型 | O09 | 要长尾记忆，时间常数就要覆盖相应尺度。这是充分条件形式。 |

### 环 8：工作量与经历长度无关

主链：`与经历长度无关的递推更新`。**这一环没有专门定理**，下面是最接近的两条。它现在是工程要求（O03），不是推出来的。

- `history_transition_exists_iff`（环 3）给出“只靠当前状态递推”的条件，但没说每步要花多少工作。
- `separated_histories_capacity`（环 3）给的是状态下界，不是工作量。

所以“无限流要求工作量不随经历增长”目前是主链的陈述和 O03 的检查，Lean 里没有对应定理。

### 环 9：延迟信用、增益、三因子、竞争

主链末端：`延迟归因的广义资格迹、隔室、连接与三因子学习`。分四组：信用、增益、两相、竞争。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `delayed_credit_requires_history_carrier`<br><sub>LanguageLifeKernel</sub> | 同一个当前输入需要两种不同信用时，只看当前输入的规则做不到。 | 必要·普适 | O15 | 结果到达时，状态必须还记得当时的区别。 |
| `delayed_action_credit_requires_state_separation`<br><sub>LanguageLifeKernel</sub> | 两个动作走到同一后继状态，却要不同信用时，任何只读后继的规则都做不到。 | 必要·普适 | O15 | 资格迹就是这种状态区别，不一定是单独数组。 |
| `no_visible_only_delayed_credit_rule`<br><sub>LanguageLifeKernel</sub> | 延迟信用目标依赖隐藏位时，只看可见值的规则做不到。 | 必要·普适 | O15 | 同上，换成延迟目标的写法。 |
| `contact_response_unsaturated_increment`<br><sub>GenerationContactEligibility</sub> | 活动没有饱和时，接触权重的微小改变经由局部参与度精确传到响应上。 | 设计·绑定模型 | — | 这是最接近生物资格迹的定理：迹是参与度，是响应的局部因子，不是内容。 |
| `contact_response_saturated_increment_zero`<br><sub>GenerationContactEligibility</sub> | 两个响应都饱和时，它们的差为零，饱和面上没有局部接触证据。 | 设计·绑定模型 | — | **这和锁存有矛盾：** 记住要靠饱和，而饱和的单元收不到信用。设计时要处理。 |
| `value_future_log_likelihood_deriv`<br><sub>GenerationGDNLife</sub> | 下一字节对数似然对编码值方向的导数，等于前向资格迹乘以“到达减预测”的语言误差。模型绑定 GDNLife。 | 设计·绑定模型 | — | 前向迹加当拍误差能得到精确信用的形式，但只在这个模型上证明。 |
| `no_fixed_gain_for_two_volatilities`<br><sub>Generation</sub> | 同一噪声下，两个不同波动的环境不能共用一个固定增益而都达到最优。 | 设计·普适 | O13 | 增益必须随局部状况变化。 |
| `no_global_gain_for_heterogeneous_compartments`<br><sub>Generation</sub> | 隔室波动不同时，共享固定增益至少让一个隔室严格次优。 | 必要·世界条件 | O13 | 这是隔室需要局部增益的理由，只在随机游走估计模型里证明。 |
| `process_noise_gain_floor`<br><sub>GenerationAssociationCapacity</sub> | 过程噪声和到达精度都为正时，增益有严格正的下界。 | 设计·绑定模型 | O13、O16 | 时间流逝本身让身体保持可学。 |
| `phases_must_balance`<br><sub>Generation</sub> | 两相学习的漂移对所有读出都是似然梯度的正倍数时，驱动相和自由相权重必须相等。 | 必要·普适 | — | “到达 − 预测”的形式是被迫的，两项不能不等权。 |
| `reward_phases_must_balance`<br><sub>Generation</sub> | 奖励版本的同一结论。 | 必要·普适 | — | 第三因子调制时同样要等权。 |
| `three_part_balance_impossible`<br><sub>Generation</sub> | 细致平衡模型里，三部分加性速率配三因子乘积权重且三项都非恒定时，选定状态的反向总速率被迫为零。 | 必要·普适 | — | 这种写法的三因子不能保持细致平衡。这条是模型形式上的限制。 |
| `local_anti_dominance_requires_demand_readout`<br><sub>GenerationOptimizer</sub> | 局部反垄断的增益不可能和需求无关。 | 设计·绑定模型 | O12 | 防止赢家通吃，局部增益必须读局部需求。 |
| `column_budget_allows_wta`<br><sub>GenerationContinualOptimizer</sub> | 只有单侧预算时，一行可以拿走全部方向。 | 见证·绑定模型 | O12 | 这是一个反例：单侧预算防不住赢家通吃，要两侧都有预算。 |
| `symmetric_stays_symmetric`<br><sub>GenerationGrowth</sub> | 确定性更新和重标记交换时，对称初态永远对称。 | 设计·绑定模型 | O12 | 分化不会自己发生，必须有边界差异、身份或涨落。 |

### 身体外形：公理层

这几条不属于主链某一环，但决定“满足公理”到底说明了什么。

| 定理 | 说什么 | 类别·范围 | 门 | 对设计意味着什么 |
|---|---|---|---|---|
| `arbitrary_driven_tables_admitted`<br><sub>ConstraintFreedom</sub> | 有限状态、唯一零态和非负速率下，任意更新表都能由一个开放单点介质满足公理。 | 语义·普适 | O22 | **公理本身不约束能力。** 满足公理是必要条件，什么都不保证。 |
| `existence_survives_weakening`<br><sub>Foundation</sub> | 已有一个见证时，放宽条件不会破坏它。 | 语义·普适 | — | 不能用“去掉某公理后有坏模型”证明该公理必要。 |
| `body_axioms`<br><sub>GenerationContextBody</sub> | `context_cells` 的外形满足 P1–P3。 | 设计·绑定模型 | O01 | 只说明外形。Lean 里的函数都是自由函数。 |
| `events_touch_bounded`<br><sub>GenerationContextBody</sub> | `context_cells` 的每个事件只碰有界数量的单元，并且总有未被碰到的单元。 | 设计·绑定模型 | O22 | 排除上面那个单点介质的漏洞。任何新身体都要有自己的这一条。 |


## 7. 定理不管的地方：你的设计空间

下面这些定理一条都没有规定，全是你可以选的：

- **载体：** 柱、GDN、Transformer、果蝇、3D、查表，定理都不区分。线性载体才受 `exact_key_capacity` 的限制。
- **状态怎么分：** H/C/S 三页是工程选择。定理只要求能读出、能递推。
- **表示：** 编码、嵌入、地址怎么学，定理都没说。
- **维度和规模：** r、层数、柱数、预算大小都没有定理。下界只有“要分开多少种历史”。
- **学习规则的具体形式：** 定理只要求局部、用到达减预测、有信用载体。具体函数是自由的。
- **能力本身：** 泛化、语义、推理，没有任何定理。全部靠门来量。

没有根定理、只能靠你定义或靠门验收的要求：

| 要求 | 现状 |
|---|---|
| L1 自举目的 | 只有条件性后果模型，目的怎么形成没有定理 |
| L5 生态位 | 没有顶层根定理 |
| 工作量不随经历增长 | 主链有陈述，Lean 没有定理 |
| 泛化（O21） | 工程要求，明确写了不是必要性定理 |
| 连接和隔室随经历生长 | 没有必要性定理 |

## 8. 需要你拍板的 10 件事

**1. BPTT 到底怎么算。已定（11-02）。**
- **你的决定：** 信用分三层：当拍梯度、固定窗口 BPTT、窗口外靠记忆取回。五行规范和 AGENTS.md 已按此修改，下面保留原来的矛盾作记录。
- [活规格](CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md) §7：“BPTT 或有限窗口可作为局部数值工具，却不能充当跨无限流延迟后果的记忆、归因或自我连续性证明”。
- [五行规范](CGB_FIVE_PHASE_DIAGNOSTIC_RULES.md) §9：“CGB语言学习禁止BPTT、手写时域反传及完整历史Jacobian/RTRL”。
- 定理：Lean 里没有禁止 BPTT 的定理。固定窗口的截断 BPTT 工作量也与经历长度无关，超出窗口的信用才需要状态载体。所以这条禁令是你的决定，不是推出来的。这件事影响最大。

**2. 载体是否还固定为柱。文档和你的话矛盾。**
- 活规格 §6.1：“目标载体是柱式共同递归体”，§6 还固定了 H/C/S 字段。
- 你 11-01 说柱、GDN、果蝇、3D 都是实现载体。
- 如果按你的话，活规格这两处要改。

**3. 定理是许可证还是否决权。**
- 现在的工作规则是“机制要先有 Lean 必要性才能用”。
- 活规格 §7 又说现代深度网络原语都允许，只看能否接入同一动力学并以证据胜出。
- 我的建议是定理只否决，能力由门来量。否则编码器和地址这些东西永远不许学。

**4. 预算是多少。**
- 1 MiB 是我为了比较选的，没有依据。
- 单元数 K 同样是选的：动力学方程的带宽预言没有过。
- 活规格 §5 要求“保持并超过现代语言模型”的能力，你又要求能在手机上跑。这两个目标下的预算差好几个数量级。

**5. 出生时带多少先天结构。**
- P3 不允许手设模块，但允许身份、位置和历史。
- 外循环（代际筛选）给出的初值和规则算不算先天结构，现有文档没有说。这决定逆发育和生成规则放在哪。

**6. 第一版要不要接系统 4。**
- 现在是延期。但环 5 和环 6 说明，只读不说时因果这一层是空的。语言能力做到什么程度再接，由你定。

**7. 资格迹能不能是一页独立状态。已定（11-02）。**
- **你的决定：** 加不加页不重要。规则改为不再禁止，但要计入资源、不得隐藏。下面保留原来的讨论作记录。
- [五行规范](CGB_FIVE_PHASE_DIAGNOSTIC_RULES.md) §9：“不另加持久资格页”。
- [活规格](CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md) §6：`Trace` 必须从 `C,S` 读出，不是第四个模块；只有反例强迫，才加最小自由度。
- 定理：`delayed_credit_requires_history_carrier` 只要求某种历史载体或充分统计量，没有规定是不是单独一页。
- 实测：能从现有状态读出的写规则 `e rᵀW` 没有提分。`gdn2_trace` 的迹页方向对齐好得多，但它**正是一页持久资格页，违反上面的规则**。它只是诊断，无论分数如何都没有准入。
- 我的建议：先试不加页。被取回的记忆本身就是 `Trace(C,S)`：把过去取回当拍，用当拍信用改写取回的内容。这符合活规格，也不需要时间反传。这个方向是假设，还没有实现。

**8. 资源到底是什么。我的方程和主链原文冲突。**
- 主链：“不另设食物、能量币或储备账户”。五行规范：“不是额外的能量币”。
- [动力学方程](CGB_DYNAMICS_EQUATIONS_20261102.md)里的 `R_i` 和 `F` 正是单独的账户，全局池 `F` 还连着所有单元，不符合局部接触。所以这组方程目前**不是合规的身体**。
- 活规格允许的形式是 `Demand(Z)`：从局部使用史读出，不是账户。
- 我的建议：去掉 `F`，把 `R_j` 换成单元自己使用史的读出，再重测“同分防塌缩”是否还成立。这还没有测。

**9. 认知卸载归谁，放在哪。**
- 你已经定了：卸载要离开显存，磁盘优先，内存勉强。
- 还没定：表算身体还是算世界。这决定它占不占身体预算，也决定写入算不算行动。`context_cells` 现在把整张表算在 1 MiB 身体内。
- 我的建议（11-02 改）：算世界。表算身体时，哈希一步能碰任意格子，等于连到所有格子；算世界时，身体只在边界发出地址，寻址由世界完成。

**10. 局部性按哪一级算。**
- P1b 只要求参与者两两接触，不限距离和度数。§3.6 明说局部性不能从无限流和有限载体推出，是“信息是物理的”这条原则的选择。
- O22 要求每事件有界，但固定中枢可以绕过它：每个事件都碰同一个固定单元。
- 活规格更强：只在被事件触及的局部邻域，全局密集算子标为“未物理精化”。
- 我的建议：采用活规格这一级，并增加一条：除边界外，没有每个事件都碰的单元。`context_cells` 的 `pool` 过不了这条。

## 9. 设计工作表（空白留给你）

硬约束来自定理和门，证据来自第 10 节，都不是答案。最后一列是你的。

| 要设计的东西 | 硬约束 | 现有证据的倾向 | 你的选择 |
|---|---|---|---|
| 状态是什么 | 能读出、能递推（`causal_carrier_contract_iff`）；有限 | 载体未定，见第 8 节第 2 件 | |
| 地址从哪来 | 不能是可加的（环 3b）；不能读未来（O04） | 外部窗口上限是窗口长；身体状态做地址的配对增益为正 | |
| 单元怎样保持 | 增益不小于 1 才能锁存（环 7） | 线性转移没有记忆，需要 `γ>1` | |
| 单元怎样写入 | 必须选择性（环 7）；顺序要保留（O07） | 只有 `context_cells` 过门 | |
| 教师信号 | 到达减预测（`phases_must_balance`） | 语言每拍给全信息答案，不是多巴胺式低秩信号 | |
| 信用怎样跨时间 | 要有历史载体（环 9）；工作量与历史长度无关 | 时间信用差约 0.27 nats；重读 32 字节已追平 BPTT | **三层：当拍、固定窗口、记忆取回** |
| 记忆里留什么 | 取回内容要带上所需区别（`external_trace_requires_retrieved_distinction`） | 不要求原始记录 | **关键节点，分析后写成新的因果记忆；用 JEPA 式潜表示，不存 token 细节** |
| 记忆怎样改自己 | 增益要局部、读需求（环 9）；不能掉进不可恢复态（`trap_irrecoverable`）；当拍变好不算数（`precision_binary_sample_gain_not_future_guarantee`） | 没有现成定理或实验；自我修改是对自身的干预，不依赖系统 4 | **记忆可以真的改自己的参数** |
| 潜表示怎样防塌缩、怎样挣到字节 | 增长率由字节交叉熵决定（`predictive_growth_contract`）；被动观测定不了干预（`observational_record_does_not_fix_intervention`） | 果蝇线上时间轴 SIGReg 被内部时钟骗过，有效秩 8–10/32；只优化潜空间 MSE 已被项目拒绝 | |
| 取回多稀疏、谁发起 | 每事件只碰有界单元（O22） | 稀疏才能放盘；还没有实现 | **极稀疏；既有触景生情，也有主动回想** |
| 饱和和信用的矛盾 | 锁存需要饱和；饱和的接触收不到信用（`contact_response_saturated_increment_zero`） | 还没有设计 | |
| 竞争与资源 | 防垄断要读局部需求、要双侧预算（环 9）；不设账户 | 资源要和似然同分才防住塌缩，账户形式不合规 | |
| 活动有多稀疏 | 每事件只碰有界单元（O22） | 现在的软竞争每拍碰所有单元，不满足 O22 | |
| 什么离开身体 | 取回内容必须带上所需区别（`external_trace_requires_retrieved_distinction`） | 地址能提前算才能上盘；后缀自动机不适合 | |
| 出生时有什么 | P3 只允许边界、身份位置、历史三种破缺 | 第 8 节第 5 件 | |
| 规模 K 与预算 | 下界只有 `separated_histories_capacity` | 带宽理论失败，强模式约 9–12 个且与 β 无关；K 仍要你选 | |

## 10. 11-02 的实测（设计时用）

全部单种子，真实字节。“诊断”的意思是没有过门，不算机制准入。除特别说明外，GDN 数字取 75 万–100 万字节窗口。

| 问题 | 结果 | 能说明 | 不能说明 |
|---|---|---|---|
| 批更新值多少 | `gdn2` 2.477 → `gdn2_batched` 2.144 | 每字节更新的噪声很大 | 批多大合适 |
| 时间信用值多少 | `gdn2_batched` 2.144，`gdn2_tbptt` 1.875，`transformer_batched` 1.855 | 差约 0.27 nats | 是否违规由第 8 节决定 |
| 重读记录能否代替时间反传 | `transformer_batched` 1.855，优于 `gdn2_tbptt` 1.875 | 它每拍重读 32 字节原文，只用当拍梯度，没有时间反传；短程信用不需要 BPTT | 窗口外的记录怎样取回；层间空间反传是否生物合理 |
| 从读取反推的值规则 | 2.187，比对照差 0.043；带 MLP 时差 0.039 | 对齐 0.77 也没有转成分数，规则不保留 | 所有写规则都没用 |
| 键规则 | MLP 组从 2.277 到 2.164 | 比预言大 7 倍多，对齐度不预测分数 | 机制原因未知；未在无 MLP 组上完成 |
| MLP | 2.277 对 2.144 | 这个预算下没有提分 | 非线性没用：宽度从 64 降到 52 |
| 前向迹方向 | 对缺失梯度余弦 0.878、0.922，规则是 0.485、0.583 | 迹比规则准，但窗口 NLL 只好 0.021（2.123），T1、T2 失败；加键规则 2.105，T3 失败 | 值路径不是 GDN 的主要缺口；门控信用没有测 |
| 表上 TD(λ) | 增益 0.0006 nats | 表的读取没有延迟后果，迹无可桥接 | 迹对有延迟后果的身体没用 |
| 词上下文进键 | 增益 0.0127 nats | 这是上下文，不是迹 | — |
| 字节间互信息 | 1 拍 0.66，2 拍 0.26，4 拍 0.04，12 拍约 0 | 长程信息在多字节组合里 | 组合层面的互信息没有测 |
| 递归地址 | 配对增益 0.057–0.342；状态只记 3–4 字节；K=64 最好 2.518 | 递归有因果贡献 | 不是身体候选，比二阶计数表 2.185 差很多 |
| 资源接法 | 有效单元 9.1 → 30.8（`r ∝ (R v)^β`） | 资源要和似然同分 | 账户形式不合规，见第 8 节第 8 件 |
| 带宽 | 全低于理论带，强模式约 9–12 个 | 单元数不是这版方程给的 | 上限来自哪里：数据、状态太软或只有一拍目标 |
| 后缀自动机上盘 | 读 1.80、写 4.72 次/字节，最坏 106 步 | 功能可上盘，这个结构不适合 | 没有在真盘上测 |
| 日志加哈希 | K=8 查询 0.78 次/字节；32+ 匹配占 1.1%，准确 97% | 地址可提前算 | NLL 没有测 |

`context_cells` 约 1.52，来自 25k 段长跑，窗口不同，只能粗比。

RSI 外部参考（11-02 查证，都是代码智能体，不是神经身体）：

| 来源 | 做法 | 对设计的提醒 |
|---|---|---|
| [DGM](https://arxiv.org/abs/2505.22954) | 改自己的代码，实测定留舍，归档全部变体 | 能力破坏的变体不进归档；自改仍发生作弊 |
| [Hyperagents](https://arxiv.org/abs/2603.19461) | 改进器也可被改写 | 它自己长出持久记忆，存洞见、因果假设和计划；父代选择仍固定 |
| [奖励作弊研究](https://iclr.cc/virtual/2026/10018648) | 公开代理任务上优化，留出任务上验收 | 优化步数越多作弊越多；自我批评不可靠 |
| [SRWM](https://proceedings.mlr.press/v162/irie22b.html) | 权重矩阵在运行时改自己，用外积和 delta 规则 | 最接近神经身体的形式 |

### 公理核对（11-02）

公理很弱：`arbitrary_driven_tables_admitted` 说任意表都能当一个暴露单点通过，P1b 也没有限制参与者数量。所以多数问题落在 O01、O22，不是公理本身。下面的判断是我的阅读，没有 Lean 实例。

| 对象 | P1a 计费 | P1b / O22 局部 | P2a、P2b | P3 手设不对称 | 结论 |
|---|---|---|---|---|---|
| `context_cells` | 计入 | Lean 已证外形；`pool` 是中枢，哈希等于连到所有格子 | Lean 已证外形 | 无 | 只有它证过外形，局部性按最弱一级 |
| GDN、Transformer 现有各组 | **少算**：批累积梯度约 33 万字节没计 | 过不了 O22 的多单元读法 | 只能算单点，守恒是空的 | 无 | 没证明，未准入 |
| `gdn2_tbptt` | **少算**：64 拍状态至少 52 万字节没计 | 同上 | 同上 | 无 | 尺子，不是 1 MiB 身体 |
| 动力学方程 | 计费未写 | 软竞争每拍碰所有单元，O22 不过 | 未检 | 无 | 诊断 |
| 三层信用 | 窗口要计费 | 全局反向过不了 O22，除非只走有界活跃部分 | 未检 | 无 | 要改形式 |
| 固定大小的共享核心 | 计入 | 形式上过 O22，实质是中枢 | 未检 | 无 | **是绕过，不算局部** |
| 字节层、潜层、记忆层、反身层 | — | — | — | **四层按职责命名，违反 P3 的写法** | 要改写：层次只能来自离边界的距离和历史 |
| 不可自改的评价 | — | — | — | 落在边界的字节上就允许 | 只能锚定边界 |
| 自我编辑 | 改动要计费 | 只碰有界单元 | 闭合事件不能增加含量；无驱动时不能单向 | 无 | 只能重排，要有流驱动 |

## 11. 还没进 Lean 的定理

定理维护交给 GPT 和 Gemini。这些都不算已证。

| 内容 | 状态 | 位置 |
|---|---|---|
| 树突：漏积分前向迹是精确梯度，无环路径成立 | 9 条草稿，单文件通过 | [DendriticCreditDraft.lean](../../results/cgb_dynamics_equations_20261102/lean_drafts/DendriticCreditDraft.lean) |
| 读取写规则不反对精确梯度 | 3 条草稿通过；规则本身实测被否，不必进主线 | [WriteCreditDraft.lean](../../results/cgb_dynamics_equations_20261102/lean_drafts/WriteCreditDraft.lean) |
| 环 3b 的 12 条 | `lake build` 通过，已入分类，等全量审计 | `GenerationNonlinearity.lean` |
| 读取次数不少于跳数；固定比例不是最优 | 只有证明思路 | 待写 |
| 软竞争色散、递归一阶消失、`γ=1` 不能锁存 | 只有纸面推导 | [动力学方程](CGB_DYNAMICS_EQUATIONS_20261102.md) |
| 延迟模式增长率 `λ(k)` | 线性响应论证，未验证 | 同上 |
| 工作量与经历长度无关 | 环 8 的空洞，没有思路 | 待写 |

## 12. 现在的证据等级

| 目标 | 当前最高证据 | 不能代替它的东西 |
|---|---|---|
| L0/L6 可存活性 | E1–E2 分散合同 | 某次预测下降或一个满载状态 |
| L1 自举目的 | E0 定义，部分 E1 条件 | 外置奖励、“我要活”变量 |
| L2 记忆学习压缩 | E1–E2 | 无限日志、BPTT 图或一次拟合 |
| L3 世界模型 | E1 局部定义和条件 | 仅当前 token NLL |
| L4 操作闭合 | E1 条件主动闭合 | 内部采样或自听回声 |
| L5 生态位 | E0 目标、E1 局部条件 | 一次请求或静态语料 |

E0 是定义或世界合同，E1 是 Lean 局部机制，E2 是条件组合，E3 是同一 Φ 的精化，E4 是实现对拍，E5 是真实语言检验。来源是 [CGB_THEOREM_RELATION_MAP_20261028.md](CGB_THEOREM_RELATION_MAP_20261028.md) §1、§4。

## 13. 门的 22 项义务

阈值和完整说明在 [CGB_HARD_OBLIGATIONS.json](CGB_HARD_OBLIGATIONS.json)。阈值是工程选择，不是定理。

| 门 | 系统 | 检查什么 | 方式 |
|---|---|---|---|
| O01 | 5 | The body is a Medium | Lean |
| O22 | 5 | The Medium instance is not one exposed site | Lean |
| O02 | 1,2,3 | One body, readable projections | Lean |
| O03 | 1,3 | Finite state and history-independent work | 运行 |
| O04 | 2 | Prediction cannot read the future | 运行 |
| O05 | 1,5 | Complete continuation | 运行 |
| O06 | 2,3 | Future distinctions are held and readable | 真实语言 |
| O07 | 2,3 | Order is held | Lean |
| O08 | 1,3 | Experience becomes reserve | 真实语言 |
| O09 | 2 | Memory horizon is earned | 真实语言 |
| O21 | 2 | Knowledge generalizes to unseen contexts | 真实语言 |
| O10 | 3,5 | Key capacity exceeds the required contexts | 真实语言 |
| O11 | 1,3 | Finite intake forces selective release | 真实语言 |
| O12 | 3,5 | Competition without winner-take-all | 真实语言 |
| O13 | 3 | Gain is local and follows evidence | 运行 |
| O14 | 2,3 | Recent language determines state and older language can matter | 运行 |
| O15 | 3 | Credit acts on held distinctions | 真实语言 |
| O16 | 1 | Time is real | 运行 |
| O17 | 1 | Learning stays honest about its limits | 真实语言 |
| O18 | 1 | Language pays | 真实语言 |
| O19 | 4 | Self-hearing is not external evidence | 延期 |
| O20 | 4 | Active closure | 延期 |
