# N 类定理用途复核（2026-11-03）

> 用户后续纠正：本审查用于寻找约束尽量充分的相容集合。下文撤回的是无条件外推，并不排除把相关机制及其前提共同施加于当前候选；不以“必须是所有身体的唯一结构”作为探索许可。具体构造任务见[相容核心第6节](CGB_THEOREM_CONSISTENT_CORE_20261103.md#6-当前任务寻找并构造相容最大集合)。本轮完成的是用途分类，尚未完成相容最大集合搜索。

## 范围与方法

本轮完成 Claude 原遍历中全部 **1403 条 N** 的用途复核，来自127个生产模块。原遍历的其余365条R/M/W没有在本轮重新全审；对原综合的关键外推另外查了源码。

使用3个 `gpt-6-luna` 子代理分六组逐条审阅；每条保留实际Lean声明、编译后签名、原判定和源码位置供核对。主代理抽查新增R、联合见证与低置信项，另对第1/2组R/M及联合见证进行了交叉审查，再统一“算法构造、性质验证、测量工具”的口径。审查标签是模型辅助判断，不是Lean自动证明的分类。

自动化只负责名称覆盖、去重、字段、相关声明解析、签名及127个源码文件哈希核验；没有用关键词程序决定语义分类。没有新增Lean定理、硬义务、身体机制或训练。

## 两个互相独立的分类轴

R/M/W/N是对原遍历的复核建议，**未写回原分类库或门**：

| 建议判定 | 数量 | 含义 |
|---|---:|---|
| R | 17 | 条件性必要限制或充要判据；适用前提不能省略 |
| M | 52 | 可复用更新法、构造或机制充分关系 |
| W | 31 | 具体世界／协议前提 |
| N | 1303 | 保留为其他用途；不是无价值 |

用途标签允许一条有多种用途，因此下表不可相加当作互斥类别：

| 用途 | 数量 |
|---|---:|
| 同一对象的联合见证 | 36 |
| 局部／具体模型见证 | 266 |
| 反例或设计排除 | 184 |
| 测量与账本工具 | 58 |
| 证明辅助 | 800 |
| 公理元理论 | 51 |
| 重复关系 | 1 |
| 旁支 | 55 |

同一对象满足若干性质的联合见证，还分为公理相容、具体恢复/学习、记忆与活动等不同范围。**36条联合见证不等于36个完整语言生命体，也不证明九条候选全部相容。**

## 可直接复用的例子

| 原N中的声明 | 可复用之处及边界 |
|---|---|
| [`CGB.SelfProduction.grace_stochastic_joint_generator_exists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4056) | 显式构造同一策略与过程并证明合取性质；限指定Grace构造。 |
| [`CGB.GraceMemory.remembers_with_reachable_action_change`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L177) | 同一系统保留记忆且可达历史差异影响后续动作；不是语言能力证据。 |
| [`CGB.SeedMemory.physical_memory_component`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L181) | 同一seed兼有公理准入、真实记忆和非平凡静息事件；结论没有Lives。 |
| [`CGB.RecoveryControls.passive_intake_not_persistence`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L269) | 明确反驳“公理＋初始物料＋某处有摄入即可持续生存”的推断；不是摄入无用。 |
| [`CGB.GDNLife.value_future_logit_eligibility_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L735) | 已有更新中单次值扰动到后继logit的前向资格参照；不能外推为全参数任意长信用。 |
| [`CGB.LanguageLifeKernel.no_visible_only_delayed_credit_rule`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L99) | 在可见量相同而正确信用不同的条件下排除仅依赖可见量的规则；需保存或取回区分。 |
| [`CGB.Generation.exact_key_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L271) | 线性独立精确键族满足n≤d；不限制近似、非线性或混叠存储的总条数。 |

## 对 Claude 九条候选综合的修正

1. **联合相容与最大性尚未证明。** 分别存在、甚至若干局部合取都存在，不推出全部条件在同一状态与同一转移上可同时满足。可用现有见证作为构造积木；先写清共享变量及条件的对应，不能靠产品并排或把目标装入合同证明非空。
2. **“活着”与“中断耐受”已有定义及反例。** `Medium.Lives`、`Operational.Persists`、`Operational.ToleratesSensoryInterruption` 等已存在。石头满足介质公理，说明物理可容许性不等于生命准入；不能据此直接新增一条物理公理。门若漏检，应先补已有定义到实际实现的对应。
3. **两相平衡不强制睡眠。** `Generation.lean:579` 的 `phases_must_balance` 在指定钳制/自由相更新与梯度对齐前提下推出λ=μ；负项可以按概率解析计算，定理没有证明独立睡眠期或脑区必要。
4. **二分谱系是条件构造。** `GenerationDivisionTree.lean` 的模块说明明确：二分节点已经写进Lineage定义，尚无到任意Medium.jump的桥；因此不证明所有身体必须分裂生长或采用二分汇聚。
5. **全局收缩排除多固定点，不强制解剖隔室。** `contraction_has_no_two_latches` 可指导把需要保持的区别移出严格收缩部分，或改变该部分动力学；不存在只能加指定隔室这一结论。
6. **对称交换未必抹平。** `symmetric_exchange` 只给差乘以(1−2D)；严格衰减需|1−2D|<1，D=0保持、D=1翻转。共享势的富人俱乐部结论也局限于所声明的occupancy乘积形式及非负条件。
7. **前向标签有作用域与成本。** `tag_is_exact_credit` 固定转移T和预测权重w，对过去一次初始质量方向求导；每个独立tag需要额外状态场。它不提供免费全参历史梯度。
8. **固定点、吸引性与健康锁存不同。** `latch_above_unit_gain` 针对hardSat证明三个固定点；g=1时不能仅凭这条定理宣称抗扰吸引盆。保持、持续感知与反证释放仍需同一转移的证据。
9. **几何时间常数银行是有限区间充分构造。** 其幂律下界依赖给定权重及时间常数覆盖区间，不证明任意延迟、无限记忆或这种参数化唯一必要。

这些修正保留了机制路径，撤回的是从局部前提到唯一身体结构的外推。旧综合已加后续复核提示，未按候选九条重写门。

## 五个开放问题如何使用这些材料

| 问题 | 本次材料能支持什么 | 仍需共同实例解决什么 |
|---|---|---|
| 锁存载体 | 守恒观察量、具体记忆与活动共存见证 | 同一学习动力学的保持、抗扰、释放 |
| 地址 | 精确线性键维数界、关联写入与读出条件 | 可训练表征和受资源限制的可重复寻址 |
| 延迟信用 | 候选单次前向资格、历史区分反例与相关测量 | 所有被学习对象的信用及状态成本 |
| 隔室与时间尺度 | 具体构造和局部前提下的限制 | 这些自由度怎样从共同动力学生成 |
| 静默持续演化 | 静默中记忆与非平凡事件可共存的实例 | 选定信息动力学中的实际维持／活动条件 |

family/questions字段是用途导航，不是已经证明的逻辑依赖或联合证书。related链接按审计库存唯一解析，不能替代project_dependencies中的证明依赖。

## 覆盖、保留判断与查阅方式

- 覆盖1403/1403，缺项0，重名0。高置信1224条，中等178条，低置信1条；置信度是审查者判断，不是证明可信度。
- 唯一低置信条目：`CGB.GDNLife.Process.scored_joint_trajectory`（GenerationGDNLife.lean:1663）。保留N/local_witness：它含任意intake、机会预算、CE余量与重复目标/漂移条件，尚不能给这些量共同的语言语义，不能用作完整生命验收。
- 完整逐条用途：[JSONL](../../results/published/cgb_theorem_n_review_20261103.jsonl)；[统计与源码哈希](../../results/published/cgb_theorem_n_review_20261103.summary.json)。
- 分组审查与交叉复核：`results/cgb_theorem_traversal_20261103/n_review/`。完整源码入口仍是[CGB_PROVEN_INDEX](CGB_PROVEN_INDEX.md)，本轮未改变其1768条数量。
- 汇总校验：`D:/conda_envs/vox/python.exe scripts/review_theorem_n.py --check`；重新生成使用`--build`；渐进查询使用`--name 名称片段`或`--role counterexample`。`--progress`只报告阶段覆盖，不允许生成完整验收产物。

下一步积极把相关约束及条件构造交叠到同一载体和同一转移，继续扩大相容集合，寻找满足丰富约束的少量共同生成律；出现冲突时定位具体前提，不因缺少普遍必要性而放弃有条件的强设计。本轮未新增门、未新增身体模块。
