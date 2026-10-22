# Claude 对 Codex 接手期工作的审计与归档（2026-10-22）

审计对象：[Codex 交接说明](CGB_CLAUDE_REVIEW_HANDOFF_20261022.md) 所列全部工作，也就是 `14bcfa1` 中 Codex 的增量，以及 `eb720ed → dabcc92` 共 25 个提交。这些提交新增约 3.1 万行，其中 73 份文档、约 30 个 `information_boltzmann/cgb/` 模块。

判据只用 CGB 已有的约束：
- 机制来自必然：守恒加有限速率、对称、对称破缺；
- 物理即免费计算；
- 自然是老师：先由真实果蝇约束，再写机制；
- V1→V2→V3 阶梯，并做事前登记的预言；
- 生命要能在手机上跑，见 memory [[project-life-must-run-on-a-phone]]。

不以"实现自洽、账平、恢复逐位一致"作为准入证据。

我核实了以下几项：
- V3d 结果 JSON 中的数字；
- 物理核的速率律代码，`mobile_exchange.event_rate` 和 `growing_material.birth_log_ratio`；
- 各模块来源提交。

其余数字取自 Codex 报告，没有重跑。

## 总判

Codex 有三处做得对：按我的下一步把匹配码钉稳、给结构补材料账、把真实 R7 动态数据接进来。这三项进入主线。

**物理材料核这条链不认可，整体归档。** 它约 4000 行，记录在 50 多份文档中，涉及快场功耦合、有源端口、痕迹、空间沉积、有限端、移动材料、身份材料、边界、域出生、整核执行、原生端口、带电存储和原生信用等环节。

Codex 自己写下的约束（准入审计、阶段定位）大多正确。问题出在执行上：每次自我批评之后，都是再加一层物理结构，而不是先停下来接受自然检验。

## 一、进入主线（认可）

| 工作 | 为什么符合原则 | 主线中的角色 |
|---|---|---|
| **V3d 稳定匹配** | 正是 V3c 之后我建议的下一步。有向 64 维码加几何得 42.18% ± 0.13 点，比几何多 +10.62 点，配对 bootstrap 95% 区间 9.7–11.5 点；对称码只多 +6.87 点；256 维仅再多 0.56 点；冷启动 41.99%；4 条预言全过（已核 JSON）。 | 先天身份匹配的定量规律。**谱参与秩约 6.85，前 16 个模态保留 99.996% 的作用量**，说明身份码本身是低维的，这是"基因组瓶颈"的直接证据，下一步应事前登记低秩重拟合。 |
| **V2f 结构材料账** | 守恒骨架 `d(B+χΣg+W)/dt = Q` 是无条件收支。 | 作为记账层保留。按比例分摊、维持欠款无反馈这两点按 Codex 自己的审计，只作接口，不作生成律。 |
| **逆向支持运输及远端对照** | 是我提案的实现。得到一个真实发现：标签捕获到不了多跳，活动捕获能到达两跳以外。 | 作为"捕获可达性与选择性矛盾"的证据保留，是信用分配必须解决的约束。 |
| **第一性原理准入审计（10-21）** | 把高斯目标场、条件二叉树、sigmoid 电导、独立伯努利发生降级，判断正确。 | 保留为准入判例。 |
| **R7 真实动态数据与观测接口** | 这是"自然是老师"的正确做法：Filo2 固定提交、来源与许可、首帧排除、右删失、按生长锥留出。 | 保留数据与观测层结果：Control 的突起寿命是**双时间尺度**，快 2.09 分钟、慢 16.4 分钟，慢组权重 0.163；单指数在留出锥上明显更差。这是下一条要解释的自然规律，见第三节分歧 1。 |
| **微观到宏观消元原则** | Mori–Zwanzig 式块消元说明：记忆和历史项来自被消去的自由度，不需要另装记忆奖励。 | 进入公理化讨论，作为"有效层记忆从哪来"的推论模板。 |
| **生成元 L 的写法** | `L f = L_motion f + Σ k_a [f(X+ν_a) − f(X)]` 用同一个生成元写所有状态量，正是"不逐个安装功能"。 | 作为公理化的形式骨架。局部详细平衡降为一致性检查，不当作生成器，见分歧 2。 |

## 二、保留为诊断或对照（不进主线，也不删除）

- **E2 切边定位（旧 `m_main`）**：急性功能定位，属于诊断层。
- **空间表型描述、出生校准、条件树联合检验**：已被 Codex 自己降级。作为参考分布和失败证据保留，对应 `morphology.py`、`occurrence.py`、`tree_geometry.py`；`development.py`、`tree_development.py`、`birth.py` 只作统计对照。
- **`CGB_REACTION_GROWTH_DERIVATION_20261021.md`**：我没有逐条审，暂留，公理化时再核。

## 三、归档：物理材料核链

**范围**
- 代码（不删、不移动，测试照常通过，**停止扩展**）：
  - 交换与材料：`structural_exchange`、`finite_exchange`、`exchange_checkpoint`、`flow_exchange`、`driven_exchange`、`trace_exchange`、`spatial_exchange`、`end_exchange`、`mobile_exchange`、`identity_material`、`boundary_material`、`growing_material`、`material_fork`、`charge_storage`；
  - 端口与学习：`native_port`、`sensory_channel`、`local_learning`、`memory_intervention`；
  - 观察：`energy_observation`、`exchange_observation`、`learning_observation`、`readout_observation`、`credit_observation`。
- 文档：开头加了归档标记的那些，清单在文末。
- Git 标签：`archive/cgb-material-kernel-20261022` 指向 `dabcc92`。
- 大状态：`F:/fly_stream_overflow/cgb_*_20261022/` 原样保留。

### 归档理由

1. **顺序反了："自然是老师"被放到最后。** 核心物性是先发明再执行的，主要几类有：
   - 结构能、内聚能、弹性能、痕迹能、捕获耦合、带电能、载体能、结合能、域形成能；
   - 各种尝试率、扩散率、门控高低值。

   单是 dataclass 参数字段就有 30 多个，大部分值手选。真实果蝇数据（R7）到用户纠正之后才进来，一接上，固定物性就失败了：
   - 安装比例超过 99.9%，预测平均在场约 15 根，实测 7.6；
   - 分钟尺度只剩一个慢模，重现不出双时标；
   - 只拟合寿命时，形成能落到平坦区，参数不可识别。

   局部详细平衡只约束正反速率比，**全部内容都在这 30 多个常数和状态图的选择里**，而它们没有一个来自数据。

2. **状态量没有观测对应，不可被自然检验。**
   - 24–55 个域、122 个材料单位，Codex 自己承认"一座桥不是一条细丝""域是否对应膜段尚未成立"。
   - 一个状态变量无法对应任何可测量的模型，在真正重要的层面上无法被证伪。
   - 和真实规模（约 14 万神经元、约 5000 万突触）之间也没有缩放路径。

3. **功能全为零，原因可以说清。**
   - 字节读出 NLL 10.6–11.8，而频率基线 3.4–3.6；
   - 时序差 27.50，同序不同噪声 27.33，二者相当；
   - 标记只带来 +11.5% 的**瞬时**捕获差，未来方向不稳（5/8 对 6/8，关掉快响应后 7/8 对 4/8）；
   - **断源支长到 49 域，比有文字输入的 46 域还多**，源功为 0，靠热浴输入闭账。

   原因：生长是理想混合熵驱动的相变松弛，出生强、回收弱，系统里不存在"按用途选择结构"的通路。这不是参数没调好，是结构上缺这条通路。

4. **违反"物理即免费计算"和"能在手机上跑"。**
   - 全图 Python SSA、TwoSum 补偿账本、1e-8 审计、三 RNG 逐位恢复、回文分裂积分；
   - 精度花在一个未经验证的模型上；
   - 318 项检查验证的是实现与它自己一致，不是与自然一致。

5. **过程上逐层堆叠。** 每一轮的"下一步"都是再加一层：电容、有源端口、移动、身份、边界、域出生、带电存储……没有一层先单独通过真实数据的检验。这正是 memory [[feedback-system-cannot-be-built-piecemeal]] 和 [[feedback-theory-first-negative-results-need-causes]] 要避免的形态。

### 从归档链中提取、保留下来的教训（进入公理化的反例库）

- **生长不等于自维持。** 混合熵加热浴就能让结构数量增加，必须有断源对照，Codex 的这个对照设计是对的。
- **只改共同速率因子（势垒）的标记，不改变封闭系统的平衡分布**，因此不能算作信用。信用必须改变驱动力或接入开放驱动。
- **快导通与资格不能是同一个变量**，否则"标记"和"短时增益"无法分开验收。
- **结构与支持交换的时间尺度若不分离，可见结构只剩一个慢模**，这正是 R7 失败的直接原因，也正好指向下面的分歧 1。

## 四、与 Codex 判断的分歧

1. **R7 慢尾不是"再加一个拟合慢组"，而是捕获机制的直接预言。**
   - Kiral 2020 的生物学结论是：少数突起在尖端形成突触种子（bulb），种子使其稳定；自噬降解种子，从而限制伙伴选择。
   - 这在结构上和我们 V2 的"标签—捕获"相同：一根短寿命突起在缩回前以有限速率被捕获，被捕获后寿命变长。
   - 两态马尔可夫链（未捕获、已捕获）**必然**给出双指数生存曲线，慢组权重等于缩回前被捕获的概率 c/(c + k_fast)。
   - 这不是补丁，而是从"有限速率捕获加时间尺度分离"推出来的定理。
   - 物理核失败，恰恰因为它的支持交换混合太快，被捕获态不够慢、不够稀有。
   - 已下载的数据里还有 **Atg6、Atg7 自噬突变体**（176 和 216 条非 bulb 记录），因此可以事前登记：用 Control 拟合两态捕获模型后，突变体应当**只改变已捕获态的寿命或捕获比例，不改变快速缩回率**。
   - 这是"自然是老师"的下一步，不需要任何材料核。

2. **局部详细平衡不应作为生成器的根。**
   - 发育中的大脑是远离平衡的驱动系统。详细平衡只排除永动机，却没有说清状态图和常数。
   - 我们阶梯里真正被数据约束住的规律有：
     - 相遇（几何）；
     - 低秩有向身份码；
     - 有限资源与周转；
     - 可能的稀有捕获导致双时标。
   - 公理应放在这一层：局部性、守恒加有限速率、有限容量、有限维身份。热力学一致性作为检查项，而不是起点。

3. **验收口径。** "实现符合声明方程"只算工程门槛，不算科学证据。每一层先过真实数据，再叠下一层。

## 五、对接下来公理化讨论的建议（只列框架，留待讨论）

1. **公设层（少而硬）**：
   - 局部性；
   - 守恒加有限速率；
   - 有限容量与有限资源；
   - 身份是有限维码；
   - 开放驱动；
   - 学习只能用局部可得量。
2. **本构登记层**：每个具体函数形式或常数都要写明来源数据集、留出检验和失效范围。没有数据来源的，标为"假设"，不能进入推论链。
3. **推论层（Lean 的主战场）**。可形式化的小定理有：
   - 两态捕获推出双指数生存，且慢权重 = c/(c + k)；
   - 有限池推出竞争与稳态比例；
   - 有向双线性码的规范自由度：C→tC、A→A/t² 不变，以及 k^(−4/3) 正则缩放；
   - 离散更新保持守恒与非负不变区间，例如 V2f 的解析步 `g(t+dt)` 保持 [0, gmax]。
4. **算法层**：Python 实现与 Lean 参考实现在小规模上逐项比对。证明守恒不等于证明实现了指定动力学，这一点 Codex 说得对。
5. **数据层**：每个推论配一个真实果蝇检验，如 maleCNS 留出、R7 Control 加突变体、E1 复本上限。数据负责挑选公设下的本构。

这样主线不会偏：任何新机制要么是公设的推论，要么是登记过数据来源的本构，否则不能进入实现。

## 归档文档清单（已在各文档开头加归档标记）

物理材料核链：
- FINITE_EXCHANGE（PREREG、RESULTS）、STRUCTURE_EXCHANGE_PREREG、PHYSICAL_CORE
  - PHYSICAL_CORE 的 §1–§2 收支与速率比部分，在公理化讨论中引用。
- FAST_WORK、DRIVEN_PORTS、TRACE_EXCHANGE_RESULTS、TRACE_GROWTH_PREREG、SPATIAL_MATERIAL_RESULTS、SPATIAL_SEED_CONTROL
- CONNECTIVITY_STATE_DERIVATION、END_PAIRING、MOBILE_MATERIAL、IDENTITY_MATERIAL
- NATIVE_LEARNING、SENSORY_CHANNEL_PREREG、SENSORY_EXPOSURE、FIELD_SYMMETRY
- BOUNDARY_CONTACTS、BOUNDARY_MEMORY_PROBE、FUTURE_EXCHANGE
- MATERIAL_DOMAIN_BIRTH_DERIVATION、DOMAIN_BIRTH、DOMAIN_RATE、MEMORY_PHASE_COUPLING
- DRIVE_MAINTENANCE、FULL_STEP（DERIVATION、PREREG、RESULTS）
- LEARNING_GENERATOR_BRIDGE、TRACE_FATE、PREDICTION_OBSERVATION（DERIVATION、RESULTS）
- NATIVE_PREDICTIVE_PORT_DERIVATION、NATIVE_PORT、NATIVE_ADAPTATION_DERIVATION、NATIVE_TIMING
- POLARIZED_STORAGE、STORAGE_ACCOUNTING_REPAIR、NATIVE_CREDIT（CLOSURE、RESPONSE_CONTROL、RESULTS）
- R7_GENERATOR_BRIDGE、R7_STRUCTURE（PREREG、CALIBRATION、RESULTS）、R7_JOINT_OBSERVABLES
  - 这几份是物理核在 R7 上的投影；其中的失败结果作为证据引用。

统计对照：SPATIAL_PREVIEW、SPATIAL_IDENTITY_PREREG、BIRTH_CALIBRATION、TREE_JOINT。

未归档：V3D、V2F、SUPPORT_TRANSPORT、REMOTE_SUPPORT、FIRST_PRINCIPLES_AUDIT、MICRO_MACRO_CAUSALITY、SUBSTRATE_INDEPENDENT_GENERATION、R7_DYNAMIC（PREREG、AMENDMENT、RESULTS）、INVERSE_EVIDENCE、E2_SENSITIVITY、PROGRESS_OWNERSHIP、REACTION_GROWTH_DERIVATION，以及交接说明。
